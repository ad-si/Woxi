// `CodeParser`` reads source as source — the tokens and where they came
// from — rather than as the expression it evaluates to. Editors use it to
// locate syntax errors, and templating readers to cut a file at boundaries
// only the reader can see. Issue #603.

use super::*;

/// `source` as a Wolfram string literal, so a test can pass arbitrary code
/// through `interpret` without hand-escaping it at every call.
fn wl_string(source: &str) -> String {
  let escaped = source
    .replace('\\', "\\\\")
    .replace('"', "\\\"")
    .replace('\n', "\\n");
  format!("\"{escaped}\"")
}

mod code_tokenize {
  use super::*;

  // Every character of the input belongs to exactly one token, so the
  // tokens concatenate back to the source they were read from.
  #[test]
  fn the_tokens_tile_the_source() {
    clear_state();
    for source in [
      "a\nbb\nc",
      "f[x_] := x^2;",
      "(* a comment *) 1 + 2",
      "\"a string\" <> \"another\"",
      "<|\"k\" -> {1, 2.5, 16^^ff}|>",
      "a /. b :> c // d",
    ] {
      let joined = interpret(&format!(
        "StringJoin[Cases[CodeParser`CodeTokenize[{}], \
           LeafNode[_, s_String, _] :> s]]",
        wl_string(source)
      ))
      .unwrap();
      assert_eq!(joined, source, "tokens should rebuild {source:?}");
    }
  }

  #[test]
  fn each_kind_of_token_is_named() {
    clear_state();
    assert_eq!(
      interpret(
        "ToString[Cases[CodeParser`CodeTokenize[\"f[1] (* c *)\"], \
           LeafNode[k_, _, _] :> k], InputForm]"
      )
      .unwrap(),
      "{Token`Symbol, Token`OpenSquare, Token`Integer, Token`CloseSquare, \
       Token`Whitespace, Token`Comment}"
    );
  }

  // Longest match wins, so `===` is one token and not `==` then `=`.
  #[test]
  fn the_longest_operator_wins() {
    clear_state();
    assert_eq!(
      interpret(
        "ToString[Cases[CodeParser`CodeTokenize[\"a===b\"], \
           LeafNode[k_, _, _] :> k], InputForm]"
      )
      .unwrap(),
      "{Token`Symbol, Token`EqualEqualEqual, Token`Symbol}"
    );
  }
}

mod source_conventions {
  use super::*;

  // `SourceCharacterIndex` reports 1-based character indices, both ends
  // inclusive, so a one-character token names its index twice.
  #[test]
  fn character_indices_are_inclusive() {
    clear_state();
    assert_eq!(
      interpret(
        "ToString[Cases[CodeParser`CodeConcreteParse[\"a\\nbb\\nc\", \
           CodeParser`SourceConvention -> \"SourceCharacterIndex\"][[2]], \
           LeafNode[Token`Newline, _, a_] :> Lookup[a, Source, Nothing]], \
           InputForm]"
      )
      .unwrap(),
      "{{2, 2}, {5, 5}}"
    );
  }

  // Without the option, positions are line and column.
  #[test]
  fn the_default_is_line_and_column() {
    clear_state();
    assert_eq!(
      interpret(
        "ToString[Cases[CodeParser`CodeConcreteParse[\"a\\nbb\"][[2]], \
           LeafNode[Token`Newline, _, a_] :> Lookup[a, Source, Nothing]], \
           InputForm]"
      )
      .unwrap(),
      "{{{1, 2}, {2, 1}}}"
    );
  }

  // The idiom a templating reader uses to cut source at its newlines,
  // which is what `CodeConcreteParse` is called for.
  #[test]
  fn newline_spans_split_the_source() {
    clear_state();
    assert_eq!(
      interpret(
        "str = \"a\\nbb\\nc\"; \
         Select[Select[(StringTake[str, \
           Partition[Join[{1}, #, {StringLength[str]}], 2]] &@ \
           Flatten[{#1 - 1, #2 + 1} & @@@ Sort@Cases[ \
             CodeParser`CodeConcreteParse[str, \
               CodeParser`SourceConvention -> \"SourceCharacterIndex\"][[2]], \
             LeafNode[Token`Newline, _, a_] :> Lookup[a, Source, Nothing]]]), \
           StringQ], (StringLength[#] > 0) &]"
      )
      .unwrap(),
      "{a, bb, c}"
    );
  }
}

mod code_parse {
  use super::*;

  #[test]
  fn source_that_reads_carries_no_error_node() {
    clear_state();
    for source in ["f[x_] := x^2", "1 + 2", "{a, b, c}", "Module[{x = 1}, x]"] {
      assert_eq!(
        interpret(&format!(
          "Length[Cases[CodeParser`CodeParse[{}], \
             (ErrorNode | AbstractSyntaxErrorNode | UnterminatedGroupNode \
              | UnterminatedCallNode)[___], Infinity]]",
          wl_string(source)
        ))
        .unwrap(),
        "0",
        "{source:?} should read cleanly"
      );
    }
  }

  #[test]
  fn source_that_does_not_read_reports_where() {
    clear_state();
    assert_eq!(
      interpret(
        "ToString[Cases[CodeParser`CodeParse[\"f[x_] := ]\"], \
           ErrorNode[_, _, a_] :> Lookup[a, Source, Nothing], Infinity], \
           InputForm]"
      )
      .unwrap(),
      "{{{1, 10}, {1, 10}}}"
    );
  }

  // The position is a real one: a failure on the second line says so.
  #[test]
  fn the_reported_line_follows_the_source() {
    clear_state();
    let reported = interpret(
      "ToString[Cases[CodeParser`CodeParse[\"a = 1\\nb = ]\"], \
         ErrorNode[_, _, s_] :> First[Lookup[s, Source, {{0, 0}}]], \
         Infinity], InputForm]",
    )
    .unwrap();
    assert!(
      reported.starts_with("{{2, "),
      "the error is on line 2, got {reported}"
    );
  }
}

mod needs_the_context {
  use super::*;

  // Woxi provides the context itself, so asking for it is a no-op success
  // rather than a search for a paclet that is not installed.
  #[test]
  fn the_context_is_available() {
    clear_state();
    assert_eq!(
      interpret("Needs[\"CodeParser`\"]; Head[CodeParser`CodeTokenize[\"1\"]]")
        .unwrap(),
      "List"
    );
  }
}
