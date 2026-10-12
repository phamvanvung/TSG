from __future__ import annotations

from collections.abc import Mapping


def compare_numeric_tree(observed, reference, *, atol=1e-8, rtol=1e-6, path=""):
    """Return a list of mismatches for numeric leaves present in reference."""
    mismatches = []
    if isinstance(reference, Mapping):
        if not isinstance(observed, Mapping):
            return [f"{path or '<root>'}: observed is not a mapping"]
        for k, rv in reference.items():
            p = f"{path}.{k}" if path else str(k)
            if k not in observed:
                mismatches.append(f"{p}: missing")
            else:
                mismatches.extend(compare_numeric_tree(observed[k], rv, atol=atol, rtol=rtol, path=p))
        return mismatches
    if isinstance(reference, (int, float)) and not isinstance(reference, bool):
        if not isinstance(observed, (int, float)) or isinstance(observed, bool):
            return [f"{path}: expected numeric {reference!r}, got {observed!r}"]
        diff = abs(float(observed) - float(reference))
        lim = atol + rtol * abs(float(reference))
        if diff > lim:
            mismatches.append(f"{path}: observed={observed!r}, reference={reference!r}, diff={diff:.3e}")
    return mismatches
