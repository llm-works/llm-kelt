local := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
infra := $(shell appinfra scripts-path)

# Configuration
INFRA_DEV_PKG_NAME := llm_learn
INFRA_PYTEST_COVERAGE_THRESHOLD := 30

# Test configuration file (used by integration and e2e tests)
export LEARN_TEST_CONFIG_FILE := $(local)etc/llm-learn.yaml

# Code quality strictness
# - true: Fail on any code quality violations (CI mode)
# - false: Report violations but don't fail (development mode)
INFRA_DEV_CQ_STRICT := true

# Exclude examples from function size checks (demo scripts have longer functions)
INFRA_DEV_CQ_EXCLUDE := examples/*

# Skip built-in type target (we define custom one below for mixed dependencies)
INFRA_DEV_SKIP_TARGETS := type

# PostgreSQL configuration
INFRA_PG_CONFIG_FILE := pg.yaml

# Include framework (config first)
include $(infra)/make/Makefile.config
include $(infra)/make/Makefile.env
include $(infra)/make/Makefile.help
include $(infra)/make/Makefile.utils
include $(infra)/make/Makefile.pg

# Integration tests share a DB and cannot run in parallel.
# Define before Makefile.dev so this runs first and exits.
check::
	@echo ""
	@echo "ERROR: Use 'make check.seq' for this project."
	@echo "       Integration tests share a database and cannot run in parallel."
	@echo ""
	@exit 1

include $(infra)/make/Makefile.dev
include $(infra)/make/Makefile.pytest
include $(infra)/make/Makefile.clean
include $(infra)/make/Makefile.install

# Custom type checking: core package needs full checking, examples need --follow-imports=skip
# (examples import torch/transformers which cause mypy to hang without this flag)
type::
	@$(PYTHON) -m mypy $(INFRA_DEV_PKG_NAME)/ --exclude 'examples/'
	@if [ -d "examples" ]; then $(PYTHON) -m mypy examples/ --follow-imports=skip --ignore-missing-imports; fi

# Database migrations
ALEMBIC := $(PYTHON) -m alembic -c migrations/alembic.ini
.PHONY: migrate migrate.status migrate.history

migrate: ## Run database migrations to latest
	$(ALEMBIC) upgrade head

migrate.status: ## Show current migration status
	$(ALEMBIC) current

migrate.history: ## Show migration history
	$(ALEMBIC) history
