# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

local := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
infra := $(shell appinfra scripts-path)

# Configuration
APPINFRA_DEV_PKG_NAME := llm_kelt
APPINFRA_PYTEST_COVERAGE_THRESHOLD := 50
APPINFRA_PYTEST_COVERAGE_MARKERS := unit or integration
APPINFRA_DEV_DOCSTRING_THRESHOLD := 90

# Test configuration file (used by integration and e2e tests)
export KELT_TEST_CONFIG_FILE := $(local)etc/llm-kelt.yaml

# Code quality strictness
# - true: Fail on any code quality violations (CI mode)
# - false: Report violations but don't fail (development mode)
APPINFRA_DEV_CQ_STRICT := true

# No ruff rules reported as warnings only: they are enforced in pyproject's select
APPINFRA_DEV_CQ_LINT_SELECT := false

# Fail on duplicated code blocks and unused private definitions
APPINFRA_DEV_CQ_DUP := strict
APPINFRA_DEV_CQ_DEAD := strict

# Report tests that sit in another suite's directory, or leak process-wide state
APPINFRA_DEV_TEST_HEALTH := warn

# Enforce SPDX license headers on all source files
APPINFRA_DEV_CQ_SPDX := true

# Run example scripts as a check subcheck (same as `make examples.check`)
# Each example declares its invocation via `# ci-*` header markers.
APPINFRA_DEV_CHECK_EXAMPLES := true

# Exclude examples from function size checks (demo scripts have longer functions)
APPINFRA_DEV_CQ_EXCLUDE := llm_kelt/examples/*

# Skip built-in targets (we define custom ones below)
# - test.e2e: GPU tests must run sequentially to avoid OOM
APPINFRA_DEV_SKIP_TARGETS := test.e2e

# PostgreSQL configuration
APPINFRA_PG_CONFIG_FILE := pg.yaml

# Include framework (config first)
include $(infra)/make/Makefile.config
include $(infra)/make/Makefile.env
include $(infra)/make/Makefile.help
include $(infra)/make/Makefile.utils
include $(infra)/make/Makefile.pg

include $(infra)/make/Makefile.dev
include $(infra)/make/Makefile.pytest
include $(infra)/make/Makefile.clean
include $(infra)/make/Makefile.install

# Custom e2e tests: run sequentially (-n 0) because GPU tests can't share GPU memory
# Note: single colon overrides the infra target (double colon would append)
test.e2e:
	@echo "* running end-to-end tests (sequential for GPU)..."
	@$(PYTHON) -m pytest tests/ -m e2e -n 0 -qq --tb=short || { ec=$$?; [ $$ec -eq 5 ] && exit 0 || exit $$ec; }
	@echo "* end-to-end tests done"
.PHONY: test.e2e

# Database migrations
ALEMBIC := $(PYTHON) -m alembic -c llm_kelt/migrations/alembic.ini
.PHONY: migrate migrate.status migrate.history

migrate: ## Run database migrations to latest
	$(ALEMBIC) upgrade head

migrate.status: ## Show current migration status
	$(ALEMBIC) current

migrate.history: ## Show migration history
	$(ALEMBIC) history
