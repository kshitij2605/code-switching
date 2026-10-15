"""Shared utilities for the data / training / evaluation pipeline.

- leak_guard: blocks training rows that overlap protected evaluation data.
- build_eval_blocklist: builds common/eval_blocklist.json (ids + hashes only).
- run_registry: writes a reproducibility record next to every training run.

Scripts in the numbered stage folders import this package by putting the repo
root on sys.path (``Path(__file__).resolve().parents[2]``).
"""
