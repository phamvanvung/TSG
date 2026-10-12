import hashlib

import numpy as np

from tsg_repro.data import gaussian_view_seed, generate_gaussian


def test_gaussian_view_seed_is_frozen():
    assert gaussian_view_seed("train", "abc123", 15) == 11419406632834875237
    assert gaussian_view_seed("val", "abc123", 50) == 185516804861913356
    assert gaussian_view_seed("train", "000000", 25) == 14666755092344060692


def test_gaussian_pixels_are_frozen():
    clean = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    seed = gaussian_view_seed("train", "000000", 25)
    degraded = generate_gaussian(clean, 25, seed)
    assert degraded.dtype == np.uint8
    assert hashlib.sha256(degraded.tobytes()).hexdigest() == (
        "8a39bbc98715b76be53a6c5e148a47b36437aa499648b17eacabc61ef9e0fdd1"
    )
