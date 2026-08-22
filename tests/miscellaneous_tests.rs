// Each test file wraps its tests in a module named after the file, which
// keeps `cargo nextest run <name>` filters matching the file they live in.
#![allow(clippy::module_inception)]

mod miscellaneous_tests {
  mod attributes;
  mod docs_links;
  mod high_level_functions;
  mod list;
  mod parser;
  mod precision_display;
  mod svg_rendering;
  mod wikidata;
}
