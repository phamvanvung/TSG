"""Task-Sensitive Geometry reproducibility core."""

__version__ = "1.0.0"

from .geometry import build_task_geometry, orientation_score, orientation_score_set
from .mapping import (
    CanonicalMaps,
    fit_orthogonal_map,
    fit_teacher_canonical_maps,
    identity_distance,
    map_none_to_teacher,
    map_student_to_teacher,
    map_teacher_to_student,
    orthogonality_error,
    validate_map,
)
from .projection import decompose_flattened_region, decompose_per_cell
from .statistics import cluster_bootstrap, partial_spearman, spearman

__all__ = [
    "CanonicalMaps",
    "fit_orthogonal_map",
    "fit_teacher_canonical_maps",
    "identity_distance",
    "map_none_to_teacher",
    "map_student_to_teacher",
    "map_teacher_to_student",
    "orthogonality_error",
    "validate_map",
    "decompose_flattened_region",
    "decompose_per_cell",
    "build_task_geometry",
    "orientation_score",
    "orientation_score_set",
    "partial_spearman",
    "spearman",
    "cluster_bootstrap",
]
