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
	nix fmt


.PHONY: install
install:
	cargo install --path .


.PHONY: release
release:
	@echo '1. `cai changelog <first-commit-hash>`'
	@echo '2. `git add ./changelog.md && git commit -m "Update changelog"`'
	@echo '3. `cargo release major / minor / patch`'
	@echo '4. Create a new GitHub release at https://github.com/ad-si/Woxi/releases/new'
	@echo "5. Announce release on \n" \
		"   - https://x.com \n" \
		"   - https://bsky.app \n" \
		"   - https://this-week-in-rust.org \n" \
		"   - https://news.ycombinator.com \n" \
		"   - https://lobste.rs \n" \
		"   - Reddit \n" \
		"     - https://reddit.com/r/rust \n" \
		"     - https://reddit.com/r/Mathematica/ \n" \
		"     - https://reddit.com/r/math \n" \
		"     - https://reddit.com/r/Physics \n" \
		"     - https://reddit.com/r/ElectricalEngineering \n" \
		"     - https://reddit.com/r/matlab \n" \
		"     - https://reddit.com/r/sympy/ \n" \
		"     - https://reddit.com/r/Julia/ \n" \
		"     - https://reddit.com/r/octave/ \n"
