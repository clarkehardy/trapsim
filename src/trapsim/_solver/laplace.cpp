/*
 * laplace.cpp  –  Red-Black Gauss-Seidel SOR solver, two modes:
 *
 *   electric:   ∇·(ε∇φ) = 0
 *               Dirichlet at electrode-mask voxels, outer faces floating.
 *               One solve per electrode → field.pa<id>.
 *
 *   magnetic:   ∇·(μ∇ψ) = -src   (src = node-centred magnetic source from
 *                                  σ_M = Br·n̂ distributed over magnet shells)
 *               Dirichlet ψ=0 on outer grid faces (far-field BC).  Single
 *               solve → magfield.pa.
 *
 * Usage:
 *   ./laplace electric <grid.txt> <epsilon.raw> <out_dir> <omega> <max_iter> <tol> \
 *             <mask_1.raw> [<mask_2.raw> ...]
 *
 *   ./laplace magnetic <grid.txt> <mu.raw>      <out_dir> <omega> <max_iter> <tol> \
 *             <magnetic_source.raw>
 *
 *   grid.txt          : one line "NX NY NZ DX TX TY TZ"
 *   epsilon.raw/mu.raw: flat float64 array, shape (NZ-1)×(NY-1)×(NX-1)
 *   mask_e.raw        : flat uint8 array, shape NZ×NY×NX; 1 = inside electrode e
 *   magnetic_source.raw: flat float64 array, shape NZ×NY×NX
 *
 * Output PA format (SIMION-compatible 56-byte header + NX·NY·NZ float64, z slow):
 *   Electric mode:
 *     Free-space    : phi_solved × SCALE_REF
 *     This electrode: 2·SCALE_REF + electrode_number   (≥ 1.5·SCALE_REF)
 *     Other electr. : -1.0                              (< 0, sign-bit sentinel)
 *   Magnetic mode (every voxel):
 *     psi_solved × SCALE_REF                            (no sentinels)
 *
 * Build:
 *   clang++ -O3 -std=c++17 -o laplace laplace.cpp
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// ── Constants ─────────────────────────────────────────────────────────────────
static const double SCALE_REF = 100000.0;

// ── Grid globals (set in main, used everywhere) ───────────────────────────────
static int    NX, NY, NZ;
static double DX, TX, TY, TZ;

// Flat 3-D index: layout [k][j][i] — z slowest, x fastest (matches SIMION)
inline int idx(int i, int j, int k) {
    return k * NY * NX + j * NX + i;
}

// ── I/O helpers ───────────────────────────────────────────────────────────────

static void die(const std::string& msg) {
    std::cerr << "ERROR: " << msg << "\n";
    std::exit(1);
}

static std::vector<uint8_t> read_u8(const std::string& path, int n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die("Cannot open " + path);
    std::vector<uint8_t> v(n);
    f.read(reinterpret_cast<char*>(v.data()), n);
    if (!f) die("Short read from " + path);
    return v;
}

static std::vector<double> read_f64(const std::string& path, int n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die("Cannot open " + path);
    std::vector<double> v(n);
    f.read(reinterpret_cast<char*>(v.data()), (long long)n * 8);
    if (!f) die("Short read from " + path);
    return v;
}

static void write_i32(std::ofstream& f, int32_t v) {
    f.write(reinterpret_cast<const char*>(&v), 4);
}
static void write_f64(std::ofstream& f, double v) {
    f.write(reinterpret_cast<const char*>(&v), 8);
}

// ── Precompute node-centred diffusion coefficient (ε or μ) from cell array ───
//
// coef_cell[k][j][i] is at the centre of the cell bounded by grid nodes
// (i,j,k) to (i+1,j+1,k+1).  We interpolate to each grid node (i,j,k) as
// the arithmetic mean of the (up to 8) surrounding cell centres.
//
// coef_node has the same layout as phi: [NZ][NY][NX].
static std::vector<double> build_coef_node(const std::vector<double>& coef_cell) {
    const int NXc = NX - 1, NYc = NY - 1, NZc = NZ - 1;
    auto cidx = [&](int ci, int cj, int ck) {
        return ck * NYc * NXc + cj * NXc + ci;
    };

    std::vector<double> coef_node(NX * NY * NZ, 1.0);

    for (int k = 0; k < NZ; k++) {
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                int ci_lo = std::max(0, i - 1), ci_hi = std::min(NXc - 1, i);
                int cj_lo = std::max(0, j - 1), cj_hi = std::min(NYc - 1, j);
                int ck_lo = std::max(0, k - 1), ck_hi = std::min(NZc - 1, k);

                double sum = 0.0;
                int    cnt = 0;
                for (int ck = ck_lo; ck <= ck_hi; ck++)
                    for (int cj = cj_lo; cj <= cj_hi; cj++)
                        for (int ci = ci_lo; ci <= ci_hi; ci++) {
                            sum += coef_cell[cidx(ci, cj, ck)];
                            cnt++;
                        }
                coef_node[idx(i, j, k)] = sum / cnt;
            }
        }
    }
    return coef_node;
}

// ── Generic SOR sweep for ∇·(coef ∇φ) = -src ─────────────────────────────────
//
// `dirichlet` (optional) is a per-voxel byte mask: non-zero values mark
// Dirichlet-clamped nodes.  Their phi is left as-is by the sweep.
// `src` (optional) is a per-voxel RHS in units of [phi / dx²]; the stencil
// adds src*dx² to the numerator before dividing by the face-weight sum.
// `clamp_outer` clamps phi=0 on the outer-face nodes (i=0/NX-1 etc.).
//
// Returns number of iterations performed.

static int solve_poisson_sor(
    std::vector<double>&        phi,
    const std::vector<uint8_t>* dirichlet,    // may be nullptr
    const std::vector<double>&  coef_node,
    const std::vector<double>*  src,          // may be nullptr
    bool                        clamp_outer,
    double                      omega,
    int                         max_iter,
    double                      tol)
{
    const double dx2 = DX * DX;

    if (clamp_outer) {
        // Clamp ψ=0 on every outer-face node up front.
        for (int k = 0; k < NZ; k++)
        for (int j = 0; j < NY; j++)
        for (int i = 0; i < NX; i++) {
            if (i == 0 || i == NX-1 || j == 0 || j == NY-1 ||
                k == 0 || k == NZ-1) {
                phi[idx(i, j, k)] = 0.0;
            }
        }
    }

    int iter = 0;
    for (iter = 0; iter < max_iter; iter++) {
        double max_delta = 0.0;

        // Two half-sweeps: colour 0 (red) then colour 1 (black)
        for (int colour = 0; colour < 2; colour++) {
#ifdef _OPENMP
#pragma omp parallel for reduction(max:max_delta) schedule(dynamic,4)
#endif
            for (int k = 1; k < NZ - 1; k++) {
                for (int j = 1; j < NY - 1; j++) {
                    int start_i = 1 + ((j + k + colour + 1) % 2);
                    for (int i = start_i; i < NX - 1; i += 2) {
                        int n = idx(i, j, k);
                        if (dirichlet && (*dirichlet)[n]) continue;

                        double cn  = coef_node[n];
                        double exm = 0.5 * (cn + coef_node[idx(i-1,j,  k  )]);
                        double exp = 0.5 * (cn + coef_node[idx(i+1,j,  k  )]);
                        double eym = 0.5 * (cn + coef_node[idx(i,  j-1,k  )]);
                        double eyp = 0.5 * (cn + coef_node[idx(i,  j+1,k  )]);
                        double ezm = 0.5 * (cn + coef_node[idx(i,  j,  k-1)]);
                        double ezp = 0.5 * (cn + coef_node[idx(i,  j,  k+1)]);
                        double denom = exm + exp + eym + eyp + ezm + ezp;

                        double numer =
                            exm * phi[idx(i-1,j,  k  )] +
                            exp * phi[idx(i+1,j,  k  )] +
                            eym * phi[idx(i,  j-1,k  )] +
                            eyp * phi[idx(i,  j+1,k  )] +
                            ezm * phi[idx(i,  j,  k-1)] +
                            ezp * phi[idx(i,  j,  k+1)];
                        if (src) numer += (*src)[n] * dx2;
                        double phi_star = numer / denom;

                        double delta = omega * (phi_star - phi[n]);
                        phi[n] += delta;

                        double ad = std::fabs(delta);
                        if (ad > max_delta) max_delta = ad;
                    }
                }
            }
        }  // colour loop

        if (max_delta < tol) {
            iter++;
            break;
        }
    }
    return iter;
}

// ── Electric solve for one electrode (initialises phi, calls SOR) ────────────
static int solve_electric_one(
    std::vector<double>&        phi,
    const std::vector<uint8_t>& elec_mask,
    const std::vector<double>&  eps_node,
    int   solve_elec,
    double omega,
    int    max_iter,
    double tol)
{
    const int N = NX * NY * NZ;
    for (int n = 0; n < N; n++) {
        phi[n] = (elec_mask[n] == solve_elec) ? 1.0 : 0.0;
    }
    return solve_poisson_sor(phi, &elec_mask, eps_node,
                             /*src=*/nullptr, /*clamp_outer=*/false,
                             omega, max_iter, tol);
}

// ── Magnetic solve (single pass over the whole geometry) ─────────────────────
static int solve_magnetic(
    std::vector<double>&        psi,
    const std::vector<double>&  mu_node,
    const std::vector<double>&  src,
    double omega,
    int    max_iter,
    double tol)
{
    std::fill(psi.begin(), psi.end(), 0.0);
    return solve_poisson_sor(psi, /*dirichlet=*/nullptr, mu_node,
                             &src, /*clamp_outer=*/true,
                             omega, max_iter, tol);
}

// ── Write SIMION-compatible PA binary file ───────────────────────────────────
// Magnetic mode: pass elec_mask=nullptr, solve_elec=0 → every voxel written as
// phi[n] × SCALE_REF (no sentinels).
static void write_pa(
    const std::string&            path,
    const std::vector<double>&    phi,
    const std::vector<uint8_t>*   elec_mask,
    int                           solve_elec)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) die("Cannot create PA file: " + path);

    write_i32(f, -2);
    write_i32(f,  1);
    write_f64(f, SCALE_REF);
    write_i32(f, NX);
    write_i32(f, NY);
    write_i32(f, NZ);
    write_i32(f, 1600);
    write_f64(f, DX);
    write_f64(f, DX);
    write_f64(f, DX);

    const int N = NX * NY * NZ;
    std::vector<double> data(N);
    for (int n = 0; n < N; n++) {
        if (elec_mask) {
            uint8_t em = (*elec_mask)[n];
            if (em == solve_elec) {
                data[n] = 2.0 * SCALE_REF + solve_elec;
            } else if (em != 0) {
                data[n] = -1.0;
            } else {
                data[n] = phi[n] * SCALE_REF;
            }
        } else {
            data[n] = phi[n] * SCALE_REF;
        }
    }
    f.write(reinterpret_cast<const char*>(data.data()), (long long)N * 8);
}

// ── main ─────────────────────────────────────────────────────────────────────
static void usage_die() {
    std::cerr <<
        "Usage:\n"
        "  laplace electric <grid.txt> <epsilon.raw> <out_dir> <omega> <max_iter> <tol>"
        " <mask_1.raw> [<mask_2.raw> ...]\n"
        "  laplace magnetic <grid.txt> <mu.raw>      <out_dir> <omega> <max_iter> <tol>"
        " <magnetic_source.raw>\n";
    std::exit(1);
}

int main(int argc, char* argv[]) {
    if (argc < 8) usage_die();

    const std::string mode      = argv[1];
    const std::string grid_file = argv[2];
    const std::string coef_file = argv[3];   // epsilon.raw or mu.raw
    const std::string out_dir   = argv[4];
    const double      omega     = std::stod(argv[5]);
    const int         max_iter  = std::stoi(argv[6]);
    const double      tol       = std::stod(argv[7]);

    if (mode != "electric" && mode != "magnetic") {
        std::cerr << "ERROR: unknown mode " << mode
                  << " (must be 'electric' or 'magnetic')\n";
        usage_die();
    }

    // Read grid parameters
    {
        std::ifstream gf(grid_file);
        if (!gf) die("Cannot open grid file: " + grid_file);
        gf >> NX >> NY >> NZ >> DX >> TX >> TY >> TZ;
    }
    std::cout << "Mode: " << mode << "\n";
    std::cout << "Grid: " << NX << " × " << NY << " × " << NZ
              << "  dx=" << DX << " mm  (" << (long long)NX*NY*NZ << " nodes)\n";
    std::cout << "SOR: ω=" << omega << "  max_iter=" << max_iter
              << "  tol=" << tol << "\n";
    std::cout.flush();

    const int N  = NX * NY * NZ;
    const int Nc = (NX-1) * (NY-1) * (NZ-1);

    // Read coefficient (ε or μ) array and interpolate to nodes
    auto coef_cell = read_f64(coef_file, Nc);
    std::cout << "Coefficient array: " << Nc << " cells  (shape "
              << (NZ-1) << "×" << (NY-1) << "×" << (NX-1) << ")\n";
    std::cout << "Building node-centred coefficient array ...\n"; std::cout.flush();
    auto coef_node = build_coef_node(coef_cell);
    coef_cell.clear(); coef_cell.shrink_to_fit();

    std::vector<double> phi(N);

    if (mode == "electric") {
        const int N_ELEC = argc - 8;
        if (N_ELEC < 1) usage_die();
        std::vector<std::string> mask_files(N_ELEC);
        for (int e = 0; e < N_ELEC; e++) mask_files[e] = argv[8 + e];
        std::cout << "Electrodes: " << N_ELEC << "\n";

        // Combine electrode masks
        std::vector<uint8_t> elec_mask(N, 0);
        {
            std::vector<uint8_t> tmp(N);
            for (int e = 0; e < N_ELEC; e++) {
                tmp = read_u8(mask_files[e], N);
                for (int n = 0; n < N; n++) {
                    if (tmp[n]) elec_mask[n] = (uint8_t)(e + 1);
                }
            }
            long long elec_voxels = 0;
            for (int n = 0; n < N; n++) elec_voxels += (elec_mask[n] != 0);
            std::cout << "Electrode voxels: " << elec_voxels
                      << " (" << std::fixed << std::setprecision(2)
                      << 100.0 * elec_voxels / N << "%)\n";
        }

        for (int e = 1; e <= N_ELEC; e++) {
            auto t0 = std::chrono::steady_clock::now();
            int iters = solve_electric_one(phi, elec_mask, coef_node, e,
                                            omega, max_iter, tol);
            auto t1 = std::chrono::steady_clock::now();
            double secs = std::chrono::duration<double>(t1 - t0).count();

            std::ostringstream oss;
            oss << out_dir << "/field.pa" << e;
            write_pa(oss.str(), phi, &elec_mask, e);

            std::cout << "  pa" << e << ": " << iters << " iters  "
                      << std::fixed << std::setprecision(1) << secs << " s  → "
                      << oss.str() << "\n";
            std::cout.flush();
        }
    } else {
        // magnetic mode: expect exactly one trailing arg (source file)
        if (argc != 9) usage_die();
        const std::string src_file = argv[8];
        auto src = read_f64(src_file, N);
        long long n_src = 0;
        for (int n = 0; n < N; n++) if (src[n] != 0.0) n_src++;
        std::cout << "Magnetic source voxels: " << n_src
                  << " (" << std::fixed << std::setprecision(2)
                  << 100.0 * n_src / N << "%)\n";

        auto t0 = std::chrono::steady_clock::now();
        int iters = solve_magnetic(phi, coef_node, src, omega, max_iter, tol);
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();

        std::string out_path = out_dir + "/magfield.pa";
        write_pa(out_path, phi, /*elec_mask=*/nullptr, 0);

        std::cout << "  magnetic: " << iters << " iters  "
                  << std::fixed << std::setprecision(1) << secs << " s  → "
                  << out_path << "\n";
    }

    std::cout << "Done.\n";
    return 0;
}
