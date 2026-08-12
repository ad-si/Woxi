//! Link integrity for the published site.
//!
//! Two failure modes have bitten before and are cheap to rule out here:
//!
//! 1. A doc page is moved but `tests/SUMMARY.md` keeps pointing at the old
//!    path. The nav entry then 404s on the site and `Information[…]` hands
//!    out a dead `Documentation` URL, since both are derived from
//!    `SUMMARY.md`.
//! 2. The landing page links to `/docs/<page>.html`. The generated site
//!    serves directory URLs (`/docs/<page>/`), so the `.html` form 404s.

use std::path::{Path, PathBuf};

fn tests_dir() -> PathBuf {
  Path::new(env!("CARGO_MANIFEST_DIR")).join("tests")
}

/// All `[label](target)` pairs of a markdown outline.
fn outline_targets(text: &str) -> Vec<String> {
  let mut targets = Vec::new();
  for line in text.lines() {
    let trimmed = line.trim_start();
    if !trimmed.starts_with("- [") {
      continue;
    }
    let Some(open) = trimmed.find("](") else {
      continue;
    };
    let rest = &trimmed[open + 2..];
    let Some(close) = rest.find(')') else {
      continue;
    };
    targets.push(rest[..close].to_string());
  }
  targets
}

#[test]
fn summary_entries_point_at_existing_pages() {
  let tests = tests_dir();
  let summary = std::fs::read_to_string(tests.join("SUMMARY.md"))
    .expect("tests/SUMMARY.md is readable");
  let docs = tests.join("cli");

  let dangling: Vec<String> = outline_targets(&summary)
    .into_iter()
    .filter(|t| t.ends_with(".md"))
    .filter(|t| !docs.join(t).is_file())
    .collect();

  assert!(
    dangling.is_empty(),
    "tests/SUMMARY.md references pages that do not exist under tests/cli/ \
     (moved or deleted?): {dangling:?}\n\
     Fix the paths and re-run `wolframscript -f scripts/build_summary.wls`."
  );
}

/// Pages that are published but deliberately kept out of the nav.
const UNLISTED_PAGES: &[&str] = &[
  // Generated status report over `src/evaluator/dispatch/`, not a guide.
  "undocumented.md",
];

/// Every per-function page must be reachable from the nav, otherwise it is
/// published but orphaned — and `Information[…]` has no URL for it.
#[test]
fn every_doc_page_is_listed_in_summary() {
  let tests = tests_dir();
  let summary = std::fs::read_to_string(tests.join("SUMMARY.md"))
    .expect("tests/SUMMARY.md is readable");
  let docs = tests.join("cli");
  let listed: std::collections::HashSet<String> =
    outline_targets(&summary).into_iter().collect();

  let mut missing = Vec::new();
  let mut stack = vec![docs.clone()];
  while let Some(dir) = stack.pop() {
    for entry in std::fs::read_dir(&dir).expect("docs dir is readable") {
      let path = entry.expect("readable dir entry").path();
      if path.is_dir() {
        stack.push(path);
        continue;
      }
      if path.extension().is_none_or(|e| e != "md") {
        continue;
      }
      let rel = path
        .strip_prefix(&docs)
        .expect("page lives under tests/cli")
        .to_string_lossy()
        .replace('\\', "/");
      if !listed.contains(&rel) && !UNLISTED_PAGES.contains(&rel.as_str()) {
        missing.push(rel);
      }
    }
  }
  missing.sort();

  assert!(
    missing.is_empty(),
    "documentation pages missing from tests/SUMMARY.md: {missing:?}\n\
     Re-run `wolframscript -f scripts/build_summary.wls` to add them."
  );
}

#[test]
fn landing_page_docs_links_use_directory_urls() {
  let landing = std::fs::read_to_string(tests_dir().join("landing/index.html"))
    .expect("tests/landing/index.html is readable");
  let docs = tests_dir().join("cli");

  let mut bad = Vec::new();
  for (_, href) in landing.match_indices("href=\"").map(|(i, m)| {
    let rest = &landing[i + m.len()..];
    let end = rest.find('"').expect("href is terminated");
    (i, &rest[..end])
  }) {
    let Some(page) = href.strip_prefix("/docs/") else {
      continue;
    };
    // `/docs/` itself is the documentation index.
    if page.is_empty() {
      continue;
    }
    // The site serves generated pages as directory URLs; a `.html` suffix
    // only works for hand-written assets copied in verbatim.
    let source = page.trim_end_matches('/');
    if !docs.join(format!("{source}.md")).is_file() {
      bad.push(href.to_string());
    }
  }

  assert!(
    bad.is_empty(),
    "landing page links to documentation pages that do not exist; the site \
     serves `/docs/<page>/`, not `/docs/<page>.html`: {bad:?}"
  );
}
