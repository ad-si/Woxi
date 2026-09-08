//! Tokenisation, definition discovery, diagnostics and position mapping.

use woxi::lsp::analysis::{
  Definition, DefinitionKind, LineIndex, MODIFIER_DECLARATION,
  MODIFIER_DEFAULT_LIBRARY, SEMANTIC_TOKEN_TYPES, Severity, TokenKind,
  diagnostics, find_definitions, occurrences, scan, semantic_tokens,
  spelling_suggestion, symbol_at, tokenize,
};

/// The `(kind, text)` of every token of `source`.
fn tokens_of(source: &str) -> Vec<(TokenKind, &str)> {
  tokenize(source)
    .into_iter()
    .map(|token| (token.kind, token.text(source)))
    .collect()
}

fn definitions_of(source: &str) -> Vec<Definition> {
  find_definitions(source, &tokenize(source))
}

#[test]
fn tokenizes_a_function_call() {
  assert_eq!(
    tokens_of("Sin[x]"),
    vec![
      (TokenKind::Symbol, "Sin"),
      (TokenKind::Open, "["),
      (TokenKind::Symbol, "x"),
      (TokenKind::Close, "]"),
    ]
  );
}

#[test]
fn skips_comments_including_nested_ones() {
  assert_eq!(
    tokens_of("(* outer (* inner *) still *) x"),
    vec![(TokenKind::Symbol, "x")]
  );
}

#[test]
fn an_unterminated_comment_swallows_the_rest() {
  // Half-typed comments must not make the rest of the file look like code.
  assert_eq!(tokens_of("x (* unfinished"), vec![(TokenKind::Symbol, "x")]);
}

#[test]
fn strings_are_one_token_and_hide_their_contents() {
  assert_eq!(
    tokens_of(r#"f["Sin[x] \" still a string"]"#),
    vec![
      (TokenKind::Symbol, "f"),
      (TokenKind::Open, "["),
      (TokenKind::Str, r#""Sin[x] \" still a string""#),
      (TokenKind::Close, "]"),
    ]
  );
}

#[test]
fn an_unterminated_string_runs_to_the_end_of_the_input() {
  assert_eq!(
    tokens_of(r#"f["abc"#),
    vec![
      (TokenKind::Symbol, "f"),
      (TokenKind::Open, "["),
      (TokenKind::Str, r#""abc"#),
    ]
  );
}

#[test]
fn numbers_keep_their_precision_and_exponent_marks() {
  assert_eq!(
    tokens_of("1.5`20 + 2^^1011 - 1.*^-6"),
    vec![
      (TokenKind::Number, "1.5`20"),
      (TokenKind::Operator, "+"),
      (TokenKind::Number, "2^^1011"),
      (TokenKind::Operator, "-"),
      (TokenKind::Number, "1.*^-6"),
    ]
  );
}

#[test]
fn an_implicit_product_keeps_the_symbol_separate() {
  // `2x` is `2*x` in the Wolfram Language, so the symbol must stay
  // visible for hover and completion.
  assert_eq!(
    tokens_of("2x"),
    vec![(TokenKind::Number, "2"), (TokenKind::Symbol, "x")]
  );
}

#[test]
fn multi_character_operators_are_not_split() {
  assert_eq!(
    tokens_of("a === b =!= c ^:= d"),
    vec![
      (TokenKind::Symbol, "a"),
      (TokenKind::Operator, "==="),
      (TokenKind::Symbol, "b"),
      (TokenKind::Operator, "=!="),
      (TokenKind::Symbol, "c"),
      (TokenKind::Operator, "^:="),
      (TokenKind::Symbol, "d"),
    ]
  );
}

#[test]
fn context_marks_belong_to_the_symbol() {
  assert_eq!(
    tokens_of("System`Sin"),
    vec![(TokenKind::Symbol, "System`Sin")]
  );
}

#[test]
fn association_delimiters_are_brackets() {
  assert_eq!(
    tokens_of(r#"<|"a" -> 1|>"#),
    vec![
      (TokenKind::Open, "<|"),
      (TokenKind::Str, r#""a""#),
      (TokenKind::Operator, "->"),
      (TokenKind::Number, "1"),
      (TokenKind::Close, "|>"),
    ]
  );
}

#[test]
fn non_ascii_symbols_are_tokenized() {
  assert_eq!(
    tokens_of("α + 1"),
    vec![
      (TokenKind::Symbol, "α"),
      (TokenKind::Operator, "+"),
      (TokenKind::Number, "1"),
    ]
  );
}

#[test]
fn finds_a_delayed_function_definition() {
  let source = "square[x_] := x^2";
  let definitions = definitions_of(source);
  assert_eq!(definitions.len(), 1);
  assert_eq!(definitions[0].name, "square");
  assert_eq!(definitions[0].kind, DefinitionKind::Function);
  assert_eq!(definitions[0].operator, ":=");
  assert_eq!(definitions[0].depth, 0);
  let (start, end) = definitions[0].name_range;
  assert_eq!(&source[start..end], "square");
  let (start, end) = definitions[0].full_range;
  assert_eq!(&source[start..end], "square[x_] := x^2");
}

#[test]
fn finds_a_variable_definition() {
  let definitions = definitions_of("x = 5");
  assert_eq!(definitions.len(), 1);
  assert_eq!(definitions[0].kind, DefinitionKind::Variable);
  assert_eq!(definitions[0].operator, "=");
}

#[test]
fn a_curried_definition_is_a_function() {
  let definitions = definitions_of("f[x_][y_] := x + y");
  assert_eq!(definitions.len(), 1);
  assert_eq!(definitions[0].name, "f");
  assert_eq!(definitions[0].kind, DefinitionKind::Function);
}

#[test]
fn comparisons_are_not_definitions() {
  assert!(definitions_of("If[x == 1, a, b]").is_empty());
  assert!(definitions_of("x <= 1 && y >= 2").is_empty());
  assert!(definitions_of("x =!= y").is_empty());
}

#[test]
fn statements_end_at_a_semicolon() {
  let source = "a = 1; b = 2";
  let definitions = definitions_of(source);
  assert_eq!(definitions.len(), 2);
  let (start, end) = definitions[0].full_range;
  assert_eq!(&source[start..end], "a = 1");
  let (start, end) = definitions[1].full_range;
  assert_eq!(&source[start..end], "b = 2");
}

#[test]
fn statements_end_at_a_line_break() {
  let source = "a = 1\nb = 2\n";
  let definitions = definitions_of(source);
  assert_eq!(definitions.len(), 2);
  let (start, end) = definitions[0].full_range;
  assert_eq!(&source[start..end], "a = 1");
}

#[test]
fn a_definition_may_span_several_lines() {
  let source = "f[x_] := Module[{y = 1},\n  y + x\n]\n";
  let definitions = definitions_of(source);
  assert_eq!(definitions[0].name, "f");
  let (start, end) = definitions[0].full_range;
  assert_eq!(&source[start..end], "f[x_] := Module[{y = 1},\n  y + x\n]");
}

#[test]
fn locals_are_found_but_marked_as_nested() {
  let definitions = definitions_of("f[x_] := Module[{y = 1}, y + x]");
  let local = definitions.iter().find(|d| d.name == "y").unwrap();
  assert!(local.depth > 0);
  assert_eq!(definitions.iter().find(|d| d.name == "f").unwrap().depth, 0);
}

#[test]
fn definitions_inside_strings_and_comments_are_ignored() {
  assert!(definitions_of(r#"Print["x = 5"]"#).is_empty());
  assert!(definitions_of("(* x = 5 *)").is_empty());
}

#[test]
fn finds_the_symbol_under_the_cursor() {
  let source = "Sin[xyz]";
  let tokens = tokenize(source);
  assert_eq!(symbol_at(&tokens, 0).unwrap().text(source), "Sin");
  assert_eq!(symbol_at(&tokens, 5).unwrap().text(source), "xyz");
  // A cursor directly after a name still resolves to it.
  assert_eq!(symbol_at(&tokens, 3).unwrap().text(source), "Sin");
  assert_eq!(symbol_at(&tokens, 7).unwrap().text(source), "xyz");
  // A cursor that is not on a symbol resolves to nothing.
  let source = "1 + [";
  assert!(symbol_at(&tokenize(source), 0).is_none());
}

#[test]
fn occurrences_skip_strings_and_other_symbols() {
  let source = r#"f[x] + fx + "f" + f"#;
  let found = occurrences(source, &tokenize(source), "f");
  assert_eq!(found.len(), 2);
  assert_eq!(found[0].start, 0);
}

#[test]
fn reports_a_syntax_error_once() {
  let source = "f[1, 2";
  let tokens = tokenize(source);
  let found = diagnostics(source, &tokens, &[]);
  assert_eq!(found.len(), 1);
  assert_eq!(found[0].severity, Severity::Error);
  assert_eq!(found[0].code, "syntax");
  assert!(
    found[0].message.starts_with("Syntax error:"),
    "unexpected message: {}",
    found[0].message
  );
}

#[test]
fn valid_code_has_no_diagnostics() {
  let source = "Map[Sin, Range[3]]";
  assert!(diagnostics(source, &tokenize(source), &[]).is_empty());
}

#[test]
fn empty_input_has_no_diagnostics() {
  assert!(diagnostics("", &tokenize(""), &[]).is_empty());
  assert!(diagnostics("  \n", &tokenize("  \n"), &[]).is_empty());
}

#[test]
fn a_line_continuation_is_not_a_syntax_error() {
  // The interpreter joins a `\`-newline continuation before parsing, so
  // code pasted out of a notebook (tests/scripts/deepcopy.wls is a real
  // example) must not be flagged as broken.
  let source = "a = {\"one\", \\\n{\"two\"}};\n";
  assert!(woxi::parse(source).is_err(), "the raw grammar rejects this");
  assert!(diagnostics(source, &tokenize(source), &[]).is_empty());
}

#[test]
fn statements_separated_by_newlines_only_are_not_syntax_errors() {
  // The interpreter inserts the missing statement separators itself.
  let source = "a = 1\nb = 2\nPrint[a + b]\n";
  assert!(diagnostics(source, &tokenize(source), &[]).is_empty());
}

#[test]
fn warns_about_unimplemented_builtins() {
  // `WordData` is listed in functions.csv as out of scope for Woxi.
  let source = "WordData[\"hello\"]";
  let tokens = tokenize(source);
  let found = diagnostics(source, &tokens, &[]);
  assert_eq!(found.len(), 1);
  assert_eq!(found[0].severity, Severity::Warning);
  assert_eq!(found[0].code, "unsupported");
  assert!(
    found[0].message.contains("WordData"),
    "unexpected message: {}",
    found[0].message
  );
  let (start, end) = found[0].range;
  assert_eq!(&source[start..end], "WordData");
}

#[test]
fn a_definition_in_the_file_shadows_an_unsupported_builtin() {
  let source = "WordData[x_] := x\nWordData[1]\n";
  let tokens = tokenize(source);
  let definitions = find_definitions(source, &tokens);
  assert!(diagnostics(source, &tokens, &definitions).is_empty());
}

#[test]
fn implemented_builtins_are_not_warned_about() {
  let source = "Table[Sin[i], {i, 3}]";
  assert!(diagnostics(source, &tokenize(source), &[]).is_empty());
}

#[test]
fn maps_offsets_to_utf16_positions() {
  let source = "abc\nde√f\n";
  let index = LineIndex::new(source);
  assert_eq!(index.position(source, 0), (0, 0));
  assert_eq!(index.position(source, 3), (0, 3));
  assert_eq!(index.position(source, 4), (1, 0));
  // "√" is three bytes but a single UTF-16 code unit.
  assert_eq!(index.position(source, 9), (1, 3));
}

#[test]
fn maps_utf16_positions_back_to_offsets() {
  let source = "abc\nde√f\n";
  let index = LineIndex::new(source);
  assert_eq!(index.offset(source, 0, 0), 0);
  assert_eq!(index.offset(source, 1, 0), 4);
  assert_eq!(index.offset(source, 1, 3), 9);
  // A character past the end of a line clamps to the line's end.
  assert_eq!(index.offset(source, 0, 99), 3);
  // A line past the end of the document clamps to its end.
  assert_eq!(index.offset(source, 99, 0), source.len());
}

#[test]
fn positions_round_trip_through_offsets() {
  let source = "f[x_] := x^2\n(* ∑ *)\ng[y_] := y\n";
  let index = LineIndex::new(source);
  for (offset, _) in source.char_indices() {
    let (line, character) = index.position(source, offset);
    assert_eq!(index.offset(source, line, character), offset);
  }
}

#[test]
fn an_astral_character_takes_two_utf16_code_units() {
  let source = "x = \"🙂\"; y = 1";
  let index = LineIndex::new(source);
  let end = index.position(source, source.len());
  assert_eq!(end, (0, source.chars().count() as u32 + 1));
}

/// The repository's own script suite is known-good Wolfram Language, so
/// the language server must not flag any of it as a syntax error. This is
/// the regression test for a tokeniser or diagnostic change that starts
/// rejecting valid code.
#[test]
fn no_repository_script_reports_a_syntax_error() {
  let scripts = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("tests")
    .join("scripts");
  let mut checked = 0;
  for entry in std::fs::read_dir(&scripts).unwrap() {
    let path = entry.unwrap().path();
    if path.extension().is_none_or(|ext| ext != "wls") {
      continue;
    }
    // `_`-prefixed scripts are local scratch files (gitignored as
    // `/tests/scripts/_*`), not part of the repository.
    if path
      .file_name()
      .is_some_and(|name| name.to_string_lossy().starts_with('_'))
    {
      continue;
    }
    let source = std::fs::read_to_string(&path).unwrap();
    let source = woxi::without_shebang(&source);
    let errors: Vec<_> = diagnostics(&source, &tokenize(&source), &[])
      .into_iter()
      .filter(|diagnostic| diagnostic.severity == Severity::Error)
      .collect();
    assert!(
      errors.is_empty(),
      "{}: {:?}",
      path.display(),
      errors.first().map(|error| &error.message)
    );
    checked += 1;
  }
  assert!(checked > 100, "expected the script suite, found {checked}");
}

proptest::proptest! {
  /// The tokenizer runs on whatever is in the editor's buffer, which is
  /// arbitrary text far more often than it is valid Wolfram Language. It
  /// must never panic and must always return token ranges that index the
  /// source safely, since every answer of the server slices the text with
  /// them.
  #[test]
  fn tokenizing_arbitrary_text_yields_valid_ranges(source in ".{0,200}") {
    let tokens = tokenize(&source);
    let mut previous_end = 0;
    for token in &tokens {
      proptest::prop_assert!(token.start >= previous_end);
      proptest::prop_assert!(token.start < token.end);
      proptest::prop_assert!(source.is_char_boundary(token.start));
      proptest::prop_assert!(source.is_char_boundary(token.end));
      previous_end = token.end;
    }
    // Definitions and positions are derived from the same ranges.
    let index = LineIndex::new(&source);
    for definition in find_definitions(&source, &tokens) {
      let (start, end) = definition.full_range;
      proptest::prop_assert!(source.is_char_boundary(start));
      proptest::prop_assert!(source.is_char_boundary(end));
      proptest::prop_assert!(start <= end);
      let (line, character) = index.position(&source, start);
      proptest::prop_assert_eq!(index.offset(&source, line, character), start);
    }
  }
}

/// The `(type, modifiers, text)` of every semantic token of `source`.
fn highlights(source: &str) -> Vec<(&'static str, u32, &str)> {
  let definitions = find_definitions(source, &tokenize(source));
  semantic_tokens(source, &definitions)
    .into_iter()
    .map(|token| {
      (
        SEMANTIC_TOKEN_TYPES[token.token_type as usize],
        token.modifiers,
        &source[token.range.0..token.range.1],
      )
    })
    .collect()
}

#[test]
fn scanning_keeps_the_comments_tokenizing_drops() {
  let source = "x = 1 (* one *)";
  assert_eq!(
    scan(source)
      .into_iter()
      .map(|token| (token.kind, token.text(source)))
      .collect::<Vec<_>>(),
    vec![
      (TokenKind::Symbol, "x"),
      (TokenKind::Operator, "="),
      (TokenKind::Number, "1"),
      (TokenKind::Comment, "(* one *)"),
    ]
  );
  assert!(
    tokens_of(source)
      .iter()
      .all(|(kind, _)| *kind != TokenKind::Comment)
  );
}

#[test]
fn a_slot_is_one_token() {
  assert_eq!(
    tokens_of("#1 + ##2 + #name & /@ %3"),
    vec![
      (TokenKind::Operator, "#1"),
      (TokenKind::Operator, "+"),
      (TokenKind::Operator, "##2"),
      (TokenKind::Operator, "+"),
      (TokenKind::Operator, "#name"),
      (TokenKind::Operator, "&"),
      (TokenKind::Operator, "/@"),
      (TokenKind::Operator, "%3"),
    ]
  );
}

#[test]
fn reads_the_multi_character_operators_as_one_token() {
  for (source, operator) in [
    ("<<\"init.m\"", "<<"),
    ("expr >> \"out.txt\"", ">>"),
    ("expr >>> \"out.txt\"", ">>>"),
    ("x //= f", "//="),
    ("g /: f[g[x_]] := x", "/:"),
  ] {
    assert!(
      tokens_of(source)
        .iter()
        .any(|(kind, text)| *kind == TokenKind::Operator && *text == operator),
      "{operator} was not read as one token in: {source}"
    );
  }
}

#[test]
fn a_number_may_start_with_its_decimal_point() {
  assert_eq!(
    tokens_of("{.7, 1.5}"),
    vec![
      (TokenKind::Open, "{"),
      (TokenKind::Number, ".7"),
      (TokenKind::Operator, ","),
      (TokenKind::Number, "1.5"),
      (TokenKind::Close, "}"),
    ]
  );
  // A repetition is still an operator: `.` only starts a number before a
  // digit.
  assert_eq!(
    tokens_of("{1 ..}"),
    vec![
      (TokenKind::Open, "{"),
      (TokenKind::Number, "1"),
      (TokenKind::Operator, ".."),
      (TokenKind::Close, "}"),
    ]
  );
}

#[test]
fn highlights_built_ins_apart_from_the_file_s_own_symbols() {
  assert_eq!(
    highlights("total = Total[{1, \"a\"}] (* sum *)"),
    vec![
      ("variable", MODIFIER_DECLARATION, "total"),
      ("operator", 0, "="),
      ("function", MODIFIER_DEFAULT_LIBRARY, "Total"),
      ("operator", 0, "["),
      ("operator", 0, "{"),
      ("number", 0, "1"),
      ("operator", 0, ","),
      ("string", 0, "\"a\""),
      ("operator", 0, "}"),
      ("operator", 0, "]"),
      ("comment", 0, "(* sum *)"),
    ]
  );
}

#[test]
fn highlights_a_pattern_name_as_a_parameter_throughout_its_definition() {
  assert_eq!(
    highlights("square[x_] := x^2"),
    vec![
      ("function", MODIFIER_DECLARATION, "square"),
      ("operator", 0, "["),
      ("parameter", MODIFIER_DECLARATION, "x"),
      ("operator", 0, "_"),
      ("operator", 0, "]"),
      ("operator", 0, ":="),
      ("parameter", 0, "x"),
      ("operator", 0, "^"),
      ("number", 0, "2"),
    ]
  );
  // Outside the definition that binds it the same name is free again.
  assert_eq!(
    highlights("square[x_] := x^2\nx = 3")[9],
    ("variable", MODIFIER_DECLARATION, "x")
  );
}

#[test]
fn highlights_a_slot_as_a_parameter_and_a_message_as_a_property() {
  assert_eq!(highlights("#1 &")[0], ("parameter", 0, "#1"));
  assert_eq!(highlights("f::usage")[2], ("property", 0, "usage"));
}

#[test]
fn a_multi_line_token_is_highlighted_one_line_at_a_time() {
  assert_eq!(
    highlights("(* one\n   two *)"),
    vec![("comment", 0, "(* one"), ("comment", 0, "   two *)")]
  );
}

#[test]
fn suggests_the_built_in_a_misspelled_name_was_meant_to_be() {
  assert_eq!(spelling_suggestion("Lenght"), Some("Length"));
  assert_eq!(spelling_suggestion("Reverze"), Some("Reverse"));
  assert_eq!(spelling_suggestion("StringJion"), Some("StringJoin"));
  // A name that is spelled right, or is nothing like a built-in, or is a
  // lowercase name of the user's own, is left alone.
  assert_eq!(spelling_suggestion("Length"), None);
  assert_eq!(spelling_suggestion("Wobblefish"), None);
  assert_eq!(spelling_suggestion("lenght"), None);
  assert_eq!(spelling_suggestion("Abc"), None);
  // A numbered name of the author's own, and a name the language's `…Q`
  // convention relates to a built-in, are both deliberate.
  assert_eq!(spelling_suggestion("Option1"), None);
  assert_eq!(spelling_suggestion("Boolean"), None);
}

#[test]
fn reports_a_misspelled_built_in_as_a_hint_with_its_correction() {
  let source = "Lenght[{1, 2}]";
  let tokens = tokenize(source);
  let definitions = find_definitions(source, &tokens);
  let reported = diagnostics(source, &tokens, &definitions);
  assert_eq!(reported.len(), 1);
  assert_eq!(reported[0].code, "spelling");
  assert_eq!(reported[0].severity, Severity::Information);
  assert_eq!(reported[0].range, (0, 6));
  assert!(reported[0].message.contains("`Length`"));
}

#[test]
fn a_symbol_the_file_introduces_is_never_a_misspelling() {
  for source in [
    // Defined here…
    "Lenght[x_] := x; Lenght[{1, 2}]",
    // …or bound as a pattern.
    "f[Lenght_] := Lenght + 1",
  ] {
    let tokens = tokenize(source);
    let definitions = find_definitions(source, &tokens);
    assert!(
      diagnostics(source, &tokens, &definitions)
        .iter()
        .all(|diagnostic| diagnostic.code != "spelling"),
      "spelling reported for: {source}"
    );
  }
}
