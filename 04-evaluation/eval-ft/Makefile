# Evaluation harness targets for the frozen benchmark.
# Prefer the locked research environment (docs/environment.md), then a repo .venv, then python3.
LOCKED := $(wildcard /data/code-switching/envs/liquid-ai-research/bin/python)
VENV := $(wildcard ../../.venv/bin/python)
PYTHON ?= $(if $(LOCKED),$(LOCKED),$(if $(VENV),$(VENV),python3))

.PHONY: help test

help:
	@echo "make test       - run the scoring harness unit tests"

test:
	$(PYTHON) tests/test_score.py
