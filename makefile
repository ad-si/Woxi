.PHONY: help
help: makefile
	@tail -n +4 makefile | grep ".PHONY"


.PHONY: test-unit
test-unit:
	cargo test --quiet


# Alias the CLI command to test before running the tests.
# E.g. `wolframscript -c` or `woxi eval`
#
# $ cat /bin/wo
# #! /usr/bin/env bash
# wolframscript -c "$*"
.PHONY: test-cli
test-cli: install
	shelltest \
		--hide-successes \
		--color \
		tests/cli/*.test


.PHONY: test-cli-wolframscript
test-cli-wolframscript: install
	WOXI_USE_WOLFRAM=true \
	shelltest \
		--precise \
		--color \
		tests/cli/*.test


.PHONY: test-shebang
test-shebang: install
	test "$$(./tests/cli/hello_world.wls)" == 'Hello World!'


.PHONY: test
test: test-unit test-cli test-shebang


.PHONY: format
format:
	cargo fmt


.PHONY: install
install:
	cargo install --path .
