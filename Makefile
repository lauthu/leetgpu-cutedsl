# Developer entry points for this repository. `make help` lists them.
# Works with the GNU Make 3.81 that ships with macOS.

.DEFAULT_GOAL := help

PYTHON      ?= python3
VENV        := .venv
RUFF        := $(VENV)/bin/ruff
LEETGPU     := ./tools/leetgpu/leetgpu
NODE        := $(or $(LEETGPU_NODE),node)
PY_SOURCES  := solutions tools .claude/skills

LANGUAGE    ?= cute
GPU         ?= H100
# The CuTe DSL version to read in the editor. Keep it equal to the version the
# LeetGPU challenge page shows.
CUTLASS_REF ?= v4.4.2
CUTLASS_DIR := .cache/cutlass
# The official challenge definitions: statement, tests, tolerances and starters.
# Read-only reference. Its license forbids redistribution, so it stays in .cache.
CHALLENGES_DIR := .cache/leetgpu-challenges

# macOS signs in once through a separate Chrome window and keeps the refresh
# token in the Keychain, so later commands renew the session by themselves.
# If the sign-in provider refuses that window, use one of:
#   make login LOGIN_FLAGS=--persist            paste from a private window, stay signed in
#   make login LOGIN_FLAGS=--existing-browser   read an open Chrome tab, lasts one hour
# Elsewhere, serve the import page on a fixed port for an SSH tunnel.
ifeq ($(shell uname -s),Darwin)
LOGIN_FLAGS ?= --browser --persist
else
LOGIN_FLAGS ?= --no-browser --port 8765
endif

.PHONY: help setup stubs challenges fmt lint records test check login session preflight run submit history stats

help:
	@echo 'Setup'
	@echo '  make setup       Create .venv with the pinned ruff, fetch CuTe DSL sources and the official challenges'
	@echo '  make stubs       Fetch only the CuTe DSL sources ($(CUTLASS_REF)) into $(CUTLASS_DIR)'
	@echo '  make challenges  Fetch or update the official challenge definitions in $(CHALLENGES_DIR)'
	@echo 'Checks (no login, no GPU)'
	@echo '  make fmt       Sort imports and format Python sources'
	@echo '  make lint      Report lint and formatting problems without changing files'
	@echo '  make records   Verify recorded solution files against submissions.json'
	@echo '  make test      Run the LeetGPU tool tests (needs Node.js 22+)'
	@echo '  make check     lint + records + test'
	@echo 'LeetGPU (needs Node.js 22+ and a login)'
	@echo '  make login     Sign in ($(LOGIN_FLAGS))'
	@echo '  make run FILE=solutions/<challenge>/<version>.py      Run on the platform; not a submission'
	@echo '  make submit FILE=solutions/<challenge>/<version>.py   Submit privately and write a receipt'
	@echo '  make history CHALLENGE=<id-or-directory-name>         List submissions for LANGUAGE and GPU'
	@echo '  make stats CHALLENGE=<directory-name>                 Summarize recorded submissions for a review (offline)'
	@echo 'Options: GPU=$(GPU) LANGUAGE=$(LANGUAGE) CHALLENGE=<id-or-directory-name> (inferred from the directory of FILE)'

setup: $(RUFF) stubs challenges

$(RUFF): requirements-dev.txt
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --quiet --disable-pip-version-check -r requirements-dev.txt
	@touch $(RUFF)

# A sparse, shallow checkout: Python sources and examples only, never executed.
stubs:
	@if [ -d $(CUTLASS_DIR)/.git ]; then \
	  git -C $(CUTLASS_DIR) fetch --quiet --depth 1 origin tag $(CUTLASS_REF) && \
	  git -C $(CUTLASS_DIR) -c advice.detachedHead=false checkout --quiet $(CUTLASS_REF); \
	else \
	  git -c advice.detachedHead=false clone --quiet --depth 1 --branch $(CUTLASS_REF) \
	    --filter=blob:none --sparse https://github.com/NVIDIA/cutlass.git $(CUTLASS_DIR) && \
	  git -C $(CUTLASS_DIR) sparse-checkout set python/CuTeDSL examples/python/CuTeDSL; \
	fi
	@echo 'CuTe DSL $(CUTLASS_REF) sources are in $(CUTLASS_DIR)/python/CuTeDSL'

# The repository is small, so a shallow clone of everything is simplest; an
# existing checkout is moved to the latest main.
challenges:
	@if [ -d $(CHALLENGES_DIR)/.git ]; then \
	  git -C $(CHALLENGES_DIR) fetch --quiet --depth 1 origin main && \
	  git -C $(CHALLENGES_DIR) reset --quiet --hard FETCH_HEAD; \
	else \
	  git clone --quiet --depth 1 https://github.com/AlphaGPU/leetgpu-challenges.git $(CHALLENGES_DIR); \
	fi
	@echo 'Official challenges are in $(CHALLENGES_DIR)/challenges/<difficulty>/<id>_<name>'

fmt: $(RUFF)
	$(RUFF) check --fix-only --quiet $(PY_SOURCES)
	$(RUFF) format $(PY_SOURCES)
	@$(PYTHON) tools/check_records.py

lint: $(RUFF)
	$(RUFF) check $(PY_SOURCES)
	$(RUFF) format --check $(PY_SOURCES)

records:
	@$(PYTHON) tools/check_records.py

test:
	sh -n $(LEETGPU)
	@$(LEETGPU) --help >/dev/null
	$(NODE) --test tools/leetgpu/*.test.mjs

check: lint records test

login:
	$(LEETGPU) login $(LOGIN_FLAGS)

# status succeeds while the access token is valid or can be renewed without
# asking; only otherwise is a login needed.
session:
	@$(LEETGPU) status >/dev/null 2>&1 || $(LEETGPU) login $(LOGIN_FLAGS)

# Catch what would certainly fail on the platform before spending a GPU run:
# syntax errors and undefined names. Style findings never block a run.
preflight: $(RUFF)
	@test -n "$(FILE)" || { echo 'Usage: make $(firstword $(MAKECMDGOALS)) FILE=solutions/<challenge>/<version>.py [GPU=$(GPU)] [CHALLENGE=<id-or-directory-name>]' >&2; exit 2; }
	@test -f "$(FILE)" || { echo 'No such file: $(FILE)' >&2; exit 2; }
	@case "$(FILE)" in *.py) $(RUFF) check --quiet --select F63,F7,F82 "$(FILE)";; esac

run submit: preflight session
	$(LEETGPU) submissions $@ --language $(LANGUAGE) --gpu $(GPU) --file "$(FILE)" $(if $(CHALLENGE),--challenge "$(CHALLENGE)")

history: session
	@test -n "$(CHALLENGE)" || { echo 'Usage: make history CHALLENGE=<id-or-directory-name> [GPU=$(GPU)]' >&2; exit 2; }
	$(LEETGPU) submissions list --challenge "$(CHALLENGE)" --language $(LANGUAGE) --gpu $(GPU)

# Every number in a solution review comes from this summary of submissions.json.
stats:
	@test -n "$(CHALLENGE)" || { echo 'Usage: make stats CHALLENGE=<directory-name>, e.g. 1_vector_add' >&2; exit 2; }
	@$(PYTHON) .claude/skills/review-solution/scripts/stats.py "solutions/$(CHALLENGE)"
