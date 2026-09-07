// Each test file wraps its tests in a module named after the file, which
// keeps `cargo nextest run <name>` filters matching the file they live in.
#![allow(clippy::module_inception)]

mod lsp_tests {
  mod analysis;
  mod protocol;
  mod server;
}
