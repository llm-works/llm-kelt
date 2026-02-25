local := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
infra := $(shell appinfra scripts-path)

# Configuration
INFRA_DEV_PKG_NAME := llm_kelt
INFRA_PYTEST_COVERAGE_THRESHOLD := 30
INFRA_DEV_DOCSTRING_THRESHOLD := 90

# Test configuration file (used by integration and e2e tests)
export KELT_TEST_CONFIG_FILE := $(local)etc/llm-kelt.yaml

# Code quality strictness
# - true: Fail on any code quality violations (CI mode)
# - false: Report violations but don't fail (development mode)
INFRA_DEV_CQ_STRICT := true

# Exclude examples from function size checks (demo scripts have longer functions)
INFRA_DEV_CQ_EXCLUDE := examples/*

# Skip built-in targets (we define custom ones below)
# - type: examples import torch/transformers which cause mypy to hang
# - test.e2e: GPU tests must run sequentially to avoid OOM
INFRA_DEV_SKIP_TARGETS := type test.e2e

# PostgreSQL configuration
INFRA_PG_CONFIG_FILE := pg.yaml

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

# Custom type checking: core package needs full checking, examples need --follow-imports=skip
# (examples import torch/transformers which cause mypy to hang without this flag)
type::
	@$(PYTHON) -m mypy $(INFRA_DEV_PKG_NAME)/ --exclude 'examples/'
	@if [ -d "examples" ]; then $(PYTHON) -m mypy examples/ --follow-imports=skip --ignore-missing-imports; fi

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
