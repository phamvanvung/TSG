.PHONY: test self-check contract-audit check compile manifest

test:
	PYTHONPATH=src python -m pytest -q

self-check:
	PYTHONPATH=src python scripts/00_self_check.py

contract-audit:
	PYTHONPATH=src python scripts/audit_scientific_contracts.py

compile:
	PYTHONPATH=src python -m compileall -q src scripts tests

check: compile test self-check contract-audit

manifest:
	python scripts/build_release_manifest.py
