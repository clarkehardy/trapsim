/*
 * laplace.cpp  –  Red-Black Gauss-Seidel SOR solver for ∇·(ε∇φ) = 0
 *
 * Reads per-electrode voxel masks and a dielectric permittivity array,
 * solves the unit-potential Laplace equation for each electrode in sequence,
 * and writes SIMION-compatible PA binary files.
 *
 * Usage:
 *   ./laplace <grid.txt> <epsilon.raw> <out_dir> <omega> <max_iter> <tol> \
 *             <mask_1.raw> [<mask_2.raw> ...]
 *
 *   grid.txt   : one line "NX NY NZ DX TX TY TZ"
 *   epsilon.raw: flat float64 array, shape (NZ-1)×(NY-1)×(NX-1), row-major
 *   mask_e.raw : flat uint8 array, shape NZ×NY×NX; 1 = inside electrode e
 *   out_dir    : directory to write paulTrap.pa1 ... paulTrap.paN
 *   omega      : SOR relaxation parameter (typically 1.90–1.99)
 *   max_iter   : maximum iterations per electrode
 *   tol        : convergence tolerance (max |Δφ| per sweep)
 *
 * Output PA format (SIMION-compatible binary):
 *   56-byte header  +  NX·NY·NZ  float64 values (z slowest, x fastest)
 *   Free-space    : phi_solved × SCALE_REF
 *   This electrode: 2·SCALE_REF + electrode_number   (≥ 1.5·SCALE_REF)
 *   Other electr. : -1.0                              (< 0, sign-bit sentinel)
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

// ── Precompute node-centred ε from cell-centred epsilon array ─────────────────
//
// eps_cell[k][j][i] is at the centre of the cell bounded by grid nodes
// (i,j,k) to (i+1,j+1,k+1).  We interpolate to each grid node (i,j,k) as
// the arithmetic mean of the (up to 8) surrounding cell centres.
//
// eps_node has the same layout as phi: [NZ][NY][NX].
static std::vector<double> build_eps_node(const std::vector<double>& eps_cell) {
    const int NXc = NX - 1, NYc = NY - 1, NZc = NZ - 1;
    auto cidx = [&](int ci, int cj, int ck) {
        return ck * NYc * NXc + cj * NXc + ci;
    };

    std::vector<double> eps_node(NX * NY * NZ, 1.0);

    for (int k = 0; k < NZ; k++) {
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                // Surrounding cells: ci in [max(0,i-1)..min(NXc-1,i)]
                int ci_lo = std::max(0, i - 1), ci_hi = std::min(NXc - 1, i);
                int cj_lo = std::max(0, j - 1), cj_hi = std::min(NYc - 1, j);
                int ck_lo = std::max(0, k - 1), ck_hi = std::min(NZc - 1, k);

                double sum = 0.0;
                int    cnt = 0;
                for (int ck = ck_lo; ck <= ck_hi; ck++)
                    for (int cj = cj_lo; cj <= cj_hi; cj++)
                        for (int ci = ci_lo; ci <= ci_hi; ci++) {
                            sum += eps_cell[cidx(ci, cj, ck)];
                            cnt++;
                        }
                eps_node[idx(i, j, k)] = sum / cnt;
            }
        }
    }
    return eps_node;
}

// ── SOR solver for one electrode ─────────────────────────────────────────────
//
// elec_mask[n] = electrode number at voxel n (0 = free space, 1..N = electrode)
// eps_node[n]  = node-centred ε_r
// solve_elec   = index of the electrode to set φ=1 (others are clamped to 0)
// Returns number of iterations performed.

static int solve_sor(
    std::vector<double>&       phi,
    const std::vector<uint8_t>& elec_mask,
    const std::vector<double>&  eps_node,
    int   solve_elec,
    double omega,
    int    max_iter,
    double tol)
{
    const int N = NX * NY * NZ;

    // Initialise: 1 on this electrode, 0 elsewhere (cold start).
    for (int n = 0; n < N; n++) {
        phi[n] = (elec_mask[n] == solve_elec) ? 1.0 : 0.0;
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
                    // Starting i for this row so that (i+j+k) has the right parity
                    int start_i = 1 + ((j + k + colour + 1) % 2);
                    for (int i = start_i; i < NX - 1; i += 2) {
                        int n = idx(i, j, k);
                        if (elec_mask[n]) continue;  // fixed BC

                        // Face permittivities: arithmetic mean of adjacent nodes
                        double exm = 0.5 * (eps_node[n] + eps_node[idx(i-1,j,  k  )]);
                        double exp = 0.5 * (eps_node[n] + eps_node[idx(i+1,j,  k  )]);
                        double eym = 0.5 * (eps_node[n] + eps_node[idx(i,  j-1,k  )]);
                        double eyp = 0.5 * (eps_node[n] + eps_node[idx(i,  j+1,k  )]);
                        double ezm = 0.5 * (eps_node[n] + eps_node[idx(i,  j,  k-1)]);
                        double ezp = 0.5 * (eps_node[n] + eps_node[idx(i,  j,  k+1)]);
                        double denom = exm + exp + eym + eyp + ezm + ezp;

                        double phi_star =
                            (exm * phi[idx(i-1,j,  k  )] +
                             exp * phi[idx(i+1,j,  k  )] +
                             eym * phi[idx(i,  j-1,k  )] +
                             eyp * phi[idx(i,  j+1,k  )] +
                             ezm * phi[idx(i,  j,  k-1)] +
                             ezp * phi[idx(i,  j,  k+1)]) / denom;

                        double delta = omega * (phi_star - phi[n]);
                        phi[n] += delta;

                        double ad = std::fabs(delta);
                        if (ad > max_delta) max_delta = ad;
                    }
                }
            }
        }  // colour loop

        if (max_delta < tol) {
            iter++;  // count the converged iteration
            break;
        }
    }
    return iter;
}

// ── Write SIMION-compatible PA binary file ────────────────────────────────────
static void write_pa(
    const std::string&          path,
    const std::vector<double>&  phi,
    const std::vector<uint8_t>& elec_mask,
    int                         solve_elec)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) die("Cannot create PA file: " + path);

    // 56-byte header
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

    // Data
    const int N = NX * NY * NZ;
    std::vector<double> data(N);
    for (int n = 0; n < N; n++) {
        uint8_t em = elec_mask[n];
        if (em == solve_elec) {
            // This electrode: large positive sentinel (≥ 1.5·SCALE_REF)
            data[n] = 2.0 * SCALE_REF + solve_elec;
        } else if (em != 0) {
            // Other electrode: negative sentinel (sign-bit flag)
            data[n] = -1.0;
        } else {
            // Free space: solved potential × SCALE_REF
            data[n] = phi[n] * SCALE_REF;
        }
    }
    f.write(reinterpret_cast<const char*>(data.data()), (long long)N * 8);
}

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc < 8) {
        std::cerr <<
            "Usage: laplace <grid.txt> <epsilon.raw> <out_dir> <omega> <max_iter> <tol>"
            " <mask_1.raw> [<mask_2.raw> ...]\n";
        return 1;
    }

    // Parse fixed positional arguments
    const std::string grid_file  = argv[1];
    const std::string eps_file   = argv[2];
    const std::string out_dir    = argv[3];
    const double      omega      = std::stod(argv[4]);
    const int         max_iter   = std::stoi(argv[5]);
    const double      tol        = std::stod(argv[6]);
    const int         N_ELEC     = argc - 7;
    std::vector<std::string> mask_files(N_ELEC);
    for (int e = 0; e < N_ELEC; e++) mask_files[e] = argv[7 + e];

    // Read grid parameters
    {
        std::ifstream gf(grid_file);
        if (!gf) die("Cannot open grid file: " + grid_file);
        gf >> NX >> NY >> NZ >> DX >> TX >> TY >> TZ;
    }
    std::cout << "Grid: " << NX << " × " << NY << " × " << NZ
              << "  dx=" << DX << " mm  (" << (long long)NX*NY*NZ << " nodes)\n";
    std::cout << "SOR: ω=" << omega << "  max_iter=" << max_iter
              << "  tol=" << tol << "\n";
    std::cout << "Electrodes: " << N_ELEC << "\n";
    std::cout.flush();

    const int N = NX * NY * NZ;

    // Read and combine electrode masks into one array
    // elec_mask[n] = 0 (free) or electrode number 1..N_ELEC
    std::vector<uint8_t> elec_mask(N, 0);
    {
        std::vector<uint8_t> tmp(N);
        for (int e = 0; e < N_ELEC; e++) {
            tmp = read_u8(mask_files[e], N);
            for (int n = 0; n < N; n++) {
                if (tmp[n]) {
                    if (elec_mask[n] != 0 && elec_mask[n] != (uint8_t)(e+1)) {
                        // Overlap: last electrode wins (shouldn't happen with clean geometry)
                    }
                    elec_mask[n] = (uint8_t)(e + 1);
                }
            }
        }
        long long elec_voxels = 0;
        for (int n = 0; n < N; n++) elec_voxels += (elec_mask[n] != 0);
        std::cout << "Electrode voxels: " << elec_voxels
                  << " (" << std::fixed << std::setprecision(2)
                  << 100.0 * elec_voxels / N << "%)\n";
    }

    // Read dielectric array
    const int Nc = (NX-1) * (NY-1) * (NZ-1);
    auto eps_cell = read_f64(eps_file, Nc);
    std::cout << "Dielectric array: " << Nc << " cells  (shape "
              << (NZ-1) << "×" << (NY-1) << "×" << (NX-1) << ")\n";

    // Precompute node-centred ε
    std::cout << "Building node-centred ε array ...\n"; std::cout.flush();
    auto eps_node = build_eps_node(eps_cell);
    eps_cell.clear(); eps_cell.shrink_to_fit();  // free cell array

    // Working arrays
    std::vector<double> phi(N);

    // Solve for each electrode
    for (int e = 1; e <= N_ELEC; e++) {
        auto t0 = std::chrono::steady_clock::now();

        int iters = solve_sor(phi, elec_mask, eps_node, e, omega, max_iter, tol);

        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();

        // Build output path: <out_dir>/paulTrap.pa<e>
        std::ostringstream oss;
        oss << out_dir << "/paulTrap.pa" << e;
        write_pa(oss.str(), phi, elec_mask, e);

        std::cout << "  pa" << e << ": " << iters << " iters  "
                  << std::fixed << std::setprecision(1) << secs << " s  → "
                  << oss.str() << "\n";
        std::cout.flush();
    }

    std::cout << "Done.\n";
    return 0;
}
