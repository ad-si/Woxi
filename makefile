# Prepend local wrappers and cargo-installed binaries to PATH.
# .bin/ contains wolframscript and wo wrappers for Docker environments.
export PATH := $(CURDIR)/.bin:$(HOME)/.cargo/bin:$(PATH)

.PHONY: help
help: makefile
	@tail -n +4 makefile | grep ".PHONY"


.PHONY: test-unit
test-unit:
	cargo nextest run \
		--show-progress=none \
		--status-level=fail \
		--failure-output=final


# Run the heavy `#[ignore]`d script snapshot tests (see `slow_script_test!`
# in tests/script_snapshot_tests.rs). These are excluded from `make test`
# because they take tens of seconds each in debug builds and dominate the
# suite's wall-clock; their performance is tracked by the benchmarks. This
# target re-runs them for full correctness coverage on demand.
.PHONY: test-slow
test-slow:
	cargo nextest run \
		--profile slow \
		--run-ignored only \
		-E 'test(/^script_/)' \
		--show-progress=none \
		--status-level=fail \
		--failure-output=final


# Alias the CLI command to test before running the tests.
# E.g. `wolframscript -c` or `woxi eval`
#
# $ cat /bin/wo
# #! /usr/bin/env bash
# wolframscript -c "$*"
.PHONY: test-cli
test-cli: install-cli
	@if ! command -v scrut &> /dev/null; \
		then cargo install scrut; \
		fi
	scrut test tests/cli


.PHONY: test-cli-wolframscript
test-cli-wolframscript: install-cli
	@if ! command -v scrut &> /dev/null; \
		then cargo install scrut; \
		fi
	WOXI_USE_WOLFRAM=true \
		scrut test tests/cli


.PHONY: test-shebang
test-shebang: install-cli
	test "$$(./tests/woxi/hello_world.wls)" = 'Hello World!'


.PHONY: test-scripts-wolframscript
test-scripts-wolframscript:
	@echo "Testing scripts with wolframscript against snapshots …"
	WOXI_USE_WOLFRAM=true cargo nextest run \
		--profile slow --run-ignored all script_ --test-threads=1
	@echo "All wolframscript script tests passed."


.PHONY: test-unit-wolframscript
test-unit-wolframscript: install-cli
	@echo "Verifying unit tests against wolframscript …"
	node tests/wolframscript/verify_unit_tests.ts
	@echo "All unit test verifications passed."


.PHONY: test
test: test-unit


.PHONY: test-all
test-all: test-unit test-slow test-cli test-shebang


.PHONY: test-conformance
test-conformance: test-unit-wolframscript test-cli-wolframscript test-scripts-wolframscript


# Build a Docker image with a pinned Rust toolchain and run the
# machine-specific tests inside a clean Linux environment. The tests
# themselves derive expected values from the host (env vars,
# `gethostname`, `current_dir`, …), so this target's purpose is to
# exercise the Linux branches of the platform-dependent impl
# (`$OperatingSystem` = "Unix", `$BaseDirectory` = "/usr/share/Wolfram",
# `$UserBaseDirectory` = "$HOME/.Wolfram") on hosts that aren't Linux.
.PHONY: test-docker
test-docker:
	docker build -t woxi-test -f Dockerfile .
	docker run --rm --hostname woxi-test woxi-test


.PHONY: format
format:
	cargo clippy --fix --allow-dirty > /dev/null 2>&1
	cargo fmt
	# nix fmt


.PHONY: install-cli
install-cli:
	cargo install --path .


.PHONY: install
install: install-cli
	@if [ "$$(uname)" = "Darwin" ]; then \
		$(MAKE) install-macos-app; \
	fi


.PHONY: install-debug
install-debug:
	cargo install --debug --path .


# Build Woxi Studio and install it as a macOS .app bundle in
# /Applications. The bundle's icon is generated on the fly from
# images/favicon.png via sips + iconutil (both shipped with macOS).
APP_BUNDLE := /Applications/Woxi Studio.app

.PHONY: install-macos-app
install-macos-app:
	cargo build --release -p woxi-studio
	rm -rf "$(APP_BUNDLE)"
	mkdir -p "$(APP_BUNDLE)/Contents/MacOS"
	mkdir -p "$(APP_BUNDLE)/Contents/Resources"
	cp target/release/woxi-studio "$(APP_BUNDLE)/Contents/MacOS/woxi-studio"
	cp woxi-studio/macos/Info.plist "$(APP_BUNDLE)/Contents/Info.plist"
	@tmp=$$(mktemp -d) && \
		iconset="$$tmp/icon.iconset" && \
		mkdir "$$iconset" && \
		for s in 16 32 64 128 256 512; do \
			sips -z $$s $$s images/favicon.png \
				--out "$$iconset/icon_$${s}x$${s}.png" >/dev/null && \
			d=$$((s*2)) && \
			sips -z $$d $$d images/favicon.png \
				--out "$$iconset/icon_$${s}x$${s}@2x.png" >/dev/null; \
		done && \
		iconutil -c icns "$$iconset" \
			-o "$(APP_BUNDLE)/Contents/Resources/icon.icns" && \
		rm -rf "$$tmp"
	@/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister \
		-f "$(APP_BUNDLE)"
	@echo "Installed $(APP_BUNDLE)"


.PHONY: wasm-build
wasm-build:
	# Invalidate cargo's fingerprint to force recompilation of the woxi crate.
	# Without this, cargo's wasm32 release cache can go stale and skip rebuilding.
	touch src/lib.rs
	wasm-pack build \
		-d tests/playground/pkg \
		--target web \
		--dev \
		--no-default-features \
		--features wasm


.PHONY: wasm-build-production
wasm-build-production:
	wasm-pack build \
		-d tests/playground/pkg \
		--target web \
		--release \
		--no-default-features \
		--features wasm


.PHONY: jupyterlite-kernel-build
jupyterlite-kernel-build:
	cd jupyterlite-woxi-kernel && npm install && npx tsc
	cd jupyterlite-woxi-kernel && \
		uvx --python 3.12 --from jupyter-core --with jupyterlab jupyter labextension build .


.PHONY: jupyterlite-build
jupyterlite-build: wasm-build jupyterlite-kernel-build
	rm -f .jupyterlite.doit.db
	uvx \
		--python 3.12 \
		--no-cache \
		--with jupyterlite-core \
		--with jupyterlab \
		--with ./jupyterlite-woxi-kernel \
		jupyter lite build --output-dir tests/jupyterlite
	cp -r tests/playground/pkg tests/jupyterlite/wasm


.PHONY: docs/summary
docs/summary:
	@if command -v wolframscript >/dev/null 2>&1; \
	then wolframscript -file scripts/build_summary.wls; \
	else woxi run scripts/build_summary.wls; \
	fi


.PHONY: docs/site
docs/site: wasm-build docs/summary
	uvx --python 3.12 --from zensical==0.0.40 \
		zensical build --clean -f tests/zensical.toml


# Assemble the deployment tree:
#   - tests/landing/    → tests/book/             (minimal landing page at /)
#   - tests/playground/ → tests/book/playground/  (full playground at /playground/)
#   - zensical output   → tests/book/docs/        (via site_dir in zensical.toml)
#   - tests/jupyterlite/→ tests/book/jupyterlite/
# Both the landing page and the /playground/ copy share the same WASM
# bundle (built once by wasm-build into tests/playground/pkg/), copied
# into each location so worker.js can find ./pkg/woxi.js.
.PHONY: docs/build
docs/build: jupyterlite-build docs/site
	cp -R tests/landing/. tests/book/
	cp -R tests/playground/pkg tests/book/pkg
	rm -rf tests/book/playground
	cp -R tests/playground tests/book/playground
	rm -rf tests/book/jupyterlite
	cp -R tests/jupyterlite tests/book/jupyterlite
	cp tests/cli/favicon.png tests/book/favicon.png


.PHONY: docs/serve
docs/serve: docs/build
	cd tests/book && npx -y http-server -p 5501 -c-1 -s .


.PHONY: clean
clean:
	cargo clean
	rm -rf tests/playground/pkg
	rm -rf tests/jupyterlite
	rm -rf tests/book
	rm -f .jupyterlite.doit.db


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
