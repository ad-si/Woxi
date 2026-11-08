# Releasing Woxi

`make release` prints the release order. Preparing a release does not itself
publish packages or create a tag.

## Prepare

Choose the version and update these together:

- `Cargo.toml`, `woxi-py/Cargo.toml`, and `woxi-studio/Cargo.toml`.
- `npm/package.json`.
- `jupyterlite-woxi-kernel/package.json`, its root and root-package versions
  in `package-lock.json`, and `pyproject.toml`.
- Both version fields in `jupyterlite-woxi-kernel/src/kernel.ts`.
- Both bundle version fields in `woxi-studio/macos/Info.plist`.

Refresh `Cargo.lock` and `fuzz/Cargo.lock` with Cargo after changing versions.
The `distribution_versions_match_the_interpreter` unit test checks the
public distribution versions together. The parser and reduce crates have
independent versions; update their dependency requirements in `Cargo.toml`
if their versions change.

Move the unreleased changelog under a dated version heading, keeping an empty
Unreleased heading for subsequent work. Start the release notes with a short
list of user-facing highlights; retain the detailed changes underneath.

## Validate

Run `make format`, `make lint`, and `make test`. Also run `make test-studio`,
`make test-python`, `make test-reduce-core`, and `make npm-test` for the
packages not covered by the default interpreter suite. Run CLI documentation
tests against the working-tree executable, as described in `AGENTS.md`.

Check Windows tests, slow tests, fuzzing, all five archive builds, the Python
source-package build, and the website build in GitHub Actions. An archive
build passing does not mean that platform's tests passed. Resolve code
failures and retry infrastructure failures before release.

## Publish

Review and commit the prepared files on `main`. Publishing is a separate,
explicit step:

1. Ensure the required versions of `woxi-parser` and `woxi-reduce` are on
   crates.io. For new versions, dry-run and publish each dependency first,
   then dry-run and publish `woxi`. Do not republish existing versions.
   Use `cargo publish --dry-run -p <package>` followed by
   `cargo publish -p <package>` after reviewing the package contents.
2. Push the release commit and its `v<version>` tag. A version-tag push starts
   the Python wheel builds and PyPI publication. Confirm the Python manifest
   version before pushing: changing the tag alone does not change it.
3. Create the GitHub release for that tag. This starts the CLI/Studio archive
   workflow. Check all five platforms, `SHA256SUMS.txt`, and provenance;
   the workflow can upload a partial set when one platform fails.
4. Run `make npm-pack` to inspect the rebuilt npm package, then
   `make npm-publish`. npm publication is not triggered by the GitHub release.
5. Verify PyPI installation, npm installation, downloadable archives, and
   the deployed website/JupyterLite before announcing. The website deploys
   through its own workflow; JupyterLite metadata is prepared above, but no
   standalone JupyterLite package publication is automated here.

Do not blindly dispatch a publishing workflow to test it: manual dispatch of
the Python workflow can publish to PyPI, and a successful website build can
trigger deployment.
