"""trapsim  –  Geometry-agnostic particle-in-trap simulation.

Public API:

    from trapsim import load_geometry, load_experiment
    from trapsim.physics import Electrostatic, Gravity, EpsteinDrag, Langevin

See README.md for usage and the geometry.yaml schema.
"""

from .config import load_geometry, load_experiment, GeometryConfig, ExperimentConfig

__all__ = [
    "load_geometry",
    "load_experiment",
    "GeometryConfig",
    "ExperimentConfig",
]

__version__ = "0.1.0"
