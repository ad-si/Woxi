# Prepend local wrappers and cargo-installed binaries to PATH.
# .bin/ contains wolframscript and wo wrappers for Docker environments.
export PATH := $(CURDIR)/.bin:$(HOME)/.cargo/bin:$(PATH)

ifeq ($(OS),Windows_NT)
DEV_NULL = NUL:
else
DEV_NULL = /dev/null
endif

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
	@# The first calendar computation in a fresh wolframscript install lazily
	@# initializes the CalendarData subsystem (paclet/index fetch), a one-time
	@# cost that can stall far past nextest's per-test timeout. Several script
	@# snapshots use DayRange/DatePlus, so whichever runs first would otherwise
	@# eat that cold start and time out (e.g. find_the_last_sunday_of_each_month
	@# hit the 900s ceiling here). Pay it once, outside any timed test, so the
	@# subsequent `wolframscript -file` runs all init from the warm on-disk cache.
	@echo "Warming wolframscript CalendarData subsystem (one-time lazy init) …"
	@wolframscript -code 'DayRange[{2000,1,1},{2000,1,1},Sunday];' >/dev/null 2>&1 || true
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


# --- Fuzzing (see fuzz/README.md) -----------------------------------------

# Seed the libFuzzer corpora from the existing test scripts (plus generated
# expressions for the interpret target) so the fuzzer starts from valid
# Wolfram Language programs instead of random bytes.
.PHONY: fuzz-corpus
fuzz-corpus:
	mkdir -p fuzz/corpus/parse fuzz/corpus/interpret
	cp tests/scripts/*.wls fuzz/corpus/parse/
	cp tests/scripts/*.wls fuzz/corpus/interpret/
	cargo run --bin woxi-diff-fuzz -- --print-cases --cases 200 --seed 0 \
		| split -l 1 - fuzz/corpus/interpret/gen-

# Coverage-guided crash fuzzing (requires a nightly toolchain and
# cargo-fuzz). Runs until interrupted; crash inputs land in fuzz/artifacts/.
.PHONY: fuzz-parse
fuzz-parse: fuzz-corpus
	@if ! command -v cargo-fuzz &> /dev/null; \
		then cargo install cargo-fuzz; \
		fi
	cargo +nightly fuzz run parse -- -timeout=10 -max_len=4096

.PHONY: fuzz-interpret
fuzz-interpret: fuzz-corpus
	@if ! command -v cargo-fuzz &> /dev/null; \
		then cargo install cargo-fuzz; \
		fi
	cargo +nightly fuzz run interpret -- -timeout=20 -max_len=2048

# Differential fuzzing against wolframscript (local binary or the
# cmd-server.js Docker bridge — auto-detected). Reports and shrinks any
# output divergence; exits non-zero when one is found.
.PHONY: fuzz-diff
fuzz-diff:
	cargo build --bins
	cargo run --bin woxi-diff-fuzz -- --cases 200


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
	cargo clippy --fix --allow-dirty > $(DEV_NULL) 2>&1
	cargo fmt
#	nix fmt


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
	cargo build --profile studio -p woxi-studio
	rm -rf "$(APP_BUNDLE)"
	mkdir -p "$(APP_BUNDLE)/Contents/MacOS"
	mkdir -p "$(APP_BUNDLE)/Contents/Resources"
	cp target/studio/woxi-studio "$(APP_BUNDLE)/Contents/MacOS/woxi-studio"
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


# Inputs that, when changed, require the playground WASM bundle to be rebuilt.
# Mirrors the CI cache key in publish-book.yml (src/**, Cargo.{toml,lock}) plus
# build.rs, which also feeds the wasm32 compile.
WASM_PKG  := tests/playground/pkg/woxi_bg.wasm
WASM_SRCS := $(shell find src -type f) build.rs Cargo.toml Cargo.lock

# Real file target: only recompiles when a source input is newer than the
# bundle. wasm-pack regenerates pkg/ on every run, so the output ends up newer
# than the `touch src/lib.rs` below and the target stays up-to-date afterwards.
$(WASM_PKG): $(WASM_SRCS)
	# Invalidate cargo's fingerprint to force recompilation of the woxi crate.
	# Without this, cargo's wasm32 release cache can go stale and skip rebuilding.
	touch src/lib.rs
	wasm-pack build \
		-d tests/playground/pkg \
		--target web \
		--dev \
		-- \
		--no-default-features \
		--features wasm

# Convenience alias: `make wasm-build` (and dependents) delegate to the real
# file target, so the build is skipped when the bundle is already current.
.PHONY: wasm-build
wasm-build: $(WASM_PKG)


# Bundle CodeMirror (+ LZString) into tests/playground/vendor/codemirror.js so
# the playground loads its editor locally instead of from the esm.sh CDN. The
# committed bundle is what the playground actually serves; run this only to
# regenerate it after changing versions or the export list in
# tests/playground-deps/. `npm ci` needs network access to the npm registry.
CM_BUNDLE := tests/playground/vendor/codemirror.js
CM_SRCS   := tests/playground-deps/entry.mjs tests/playground-deps/package.json

$(CM_BUNDLE): $(CM_SRCS)
	cd tests/playground-deps && npm ci && npm run build

.PHONY: playground-codemirror
playground-codemirror: $(CM_BUNDLE)


.PHONY: wasm-build-production
wasm-build-production:
	wasm-pack build \
		-d tests/playground/pkg \
		--target web \
		--release \
		-- \
		--no-default-features \
		--features wasm


.PHONY: jupyterlite-kernel-build
jupyterlite-kernel-build:
	cd jupyterlite-woxi-kernel && npm install && npx tsc
	cd jupyterlite-woxi-kernel && \
		uvx --python 3.12 --from jupyter-builder jupyter-builder build .


.PHONY: jupyterlite-build
jupyterlite-build: wasm-build jupyterlite-kernel-build
	rm -f .jupyterlite.doit.db
	uvx \
		--python 3.12 \
		--from jupyter-core \
		--with jupyterlite-core \
		--with jupyterlab \
		--with libarchive-c \
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
docs/build: jupyterlite-build docs/site playground-codemirror
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
		"     - https://reddit.com/r/octave/ \n" \
		"     - https://reddit.com/r/buildinpublic/ \n"

