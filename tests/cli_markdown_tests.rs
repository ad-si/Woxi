//! Guard the conformance catalog's Markdown against Scrut fence ambiguity.

#[test]
fn conformance_gaps_have_unambiguous_scrut_fences() {
  let document = include_str!("cli/comparison/mathematica/conformance_gaps.md");
  let mut open_fence: Option<(&str, usize)> = None;
  for (index, line) in document.lines().enumerate() {
    let line_number = index + 1;
    if let Some((fence, _)) = open_fence {
      if line.starts_with(fence) {
        assert_eq!(
          line.trim_end(),
          fence,
          "ambiguous closing fence at line {line_number}"
        );
        open_fence = None;
      }
    } else if line.starts_with("``") {
      // Scrut 0.4.1 also treats TWO leading backticks as a block opener.
      // Keep multi-backtick inline spans after prose, even though Markdown
      // itself allows them at the beginning of a line.
      let ticks = line.bytes().take_while(|&byte| byte == b'`').count();
      let language = line[ticks..].trim();
      assert!(
        ticks >= 3 && !language.is_empty() && !language.contains('`'),
        "ambiguous Scrut fence at line {line_number}: {line}"
      );
      open_fence = Some((&line[..ticks], line_number));
    }
  }
  assert!(open_fence.is_none(), "unclosed fence: {open_fence:?}");
}
