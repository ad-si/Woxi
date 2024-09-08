.PHONY: help
help: makefile
	@tail -n +4 makefile | grep ".PHONY"


.PHONY: test-unit
test-unit:
	cargo test


.PHONY: test-cli
test-cli:
	shelltest --color cli.test


.PHONY: test
test: test-unit test-cli


.PHONY: install
install:
	cargo install --path .
