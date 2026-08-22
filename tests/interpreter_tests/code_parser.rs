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

  // The kinds the language already has a name for keep it — a symbol is a
  // `Symbol`, not a `Token`Symbol` — and only the punctuation that has no
  // name of its own is reported under `Token`.
  #[test]
  fn each_kind_of_token_is_named() {
    clear_state();
    assert_eq!(
      interpret(
        "ToString[Cases[CodeParser`CodeTokenize[\"f[1] (* c *)\"], \
           LeafNode[k_, _, _] :> k], InputForm]"
      )
      .unwrap(),
      "{Symbol, Token`OpenSquare, Integer, Token`CloseSquare, \
       Whitespace, Token`Comment}"
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
      "{Symbol, Token`EqualEqualEqual, Symbol}"
    );
  }

  // The operators whose two halves also mean something on their own: `<>`
  // is not `<` then `>`, `??` is not two `?`, and `%%` is the second-to-last
  // result rather than two `%`.
  #[test]
  fn two_character_operators_are_one_token_each() {
    clear_state();
    for (source, expected) in [
      ("a<>b", "{Symbol, Token`LessGreater, Symbol}"),
      ("??x", "{Token`QuestionQuestion, Symbol}"),
      ("!!x", "{Token`BangBang, Symbol}"),
      ("%%", "{Token`PercentPercent}"),
      (
        "x/:y=z",
        "{Symbol, Token`SlashColon, Symbol, Token`Equal, Symbol}",
      ),
      ("a**b", "{Symbol, Token`StarStar, Symbol}"),
      ("a<->b", "{Symbol, Token`LessMinusGreater, Symbol}"),
      ("a|->b", "{Symbol, Token`BarMinusGreater, Symbol}"),
      ("x//=f", "{Symbol, Token`SlashSlashEqual, Symbol}"),
      ("a::[", "{Symbol, Token`ColonColonOpenSquare}"),
    ] {
      assert_eq!(
        interpret(&format!(
          "ToString[Cases[CodeParser`CodeTokenize[{}], \
             LeafNode[k_, _, _] :> k], InputForm]",
          wl_string(source)
        ))
        .unwrap(),
        expected,
        "{source:?} should be read as {expected}"
      );
    }
  }

  // `[[` is a pair of brackets rather than a token of its own, so `a[[1]]`
  // reads the same as the `Part[a, 1]` it stands for.
  #[test]
  fn a_double_bracket_is_two_brackets() {
    clear_state();
    assert_eq!(
      interpret(
        "ToString[Cases[CodeParser`CodeTokenize[\"a[[1]]\"], \
           LeafNode[k_, _, _] :> k], InputForm]"
      )
      .unwrap(),
      "{Symbol, Token`OpenSquare, Token`OpenSquare, Integer, \
       Token`CloseSquare, Token`CloseSquare}"
    );
  }

  // A slot or an out mark stops before the number that selects which one it
  // is: `#1` is a `#` and a `1`, the way `x1` is not.
  #[test]
  fn a_slot_number_is_a_token_of_its_own() {
    clear_state();
    for (source, expected) in [
      ("#1", "{Token`Hash, Integer}"),
      ("##2", "{Token`HashHash, Integer}"),
      ("#abc", "{Token`Hash, Symbol}"),
      ("%3", "{Token`Percent, Integer}"),
      ("%%1", "{Token`PercentPercent, Integer}"),
    ] {
      assert_eq!(
        interpret(&format!(
          "ToString[Cases[CodeParser`CodeTokenize[{}], \
             LeafNode[k_, _, _] :> k], InputForm]",
          wl_string(source)
        ))
        .unwrap(),
        expected,
        "{source:?} should be read as {expected}"
      );
    }
  }

  // Whitespace is one token per character, so a run of spaces is a run of
  // tokens — a concrete tree keeps every one of them.
  #[test]
  fn each_space_is_its_own_token() {
    clear_state();
    assert_eq!(
      interpret(
        "ToString[Cases[CodeParser`CodeTokenize[\"a   b\"], \
           LeafNode[k_, _, _] :> k], InputForm]"
      )
      .unwrap(),
      "{Symbol, Whitespace, Whitespace, Whitespace, Symbol}"
    );
  }

  // The kind a number is reported under is the kind of its value: `1*^-6`
  // is the exact `1/1000000`, so it is a `Rational` and not a `Real`.
  #[test]
  fn a_number_is_named_after_its_value() {
    clear_state();
    for (source, expected) in [
      ("1", "{Integer}"),
      ("2.5", "{Real}"),
      (".5", "{Real}"),
      ("1.", "{Real}"),
      ("1..", "{Integer, Token`DotDot}"),
      ("16^^ff", "{Integer}"),
      ("16^^f.f", "{Real}"),
      ("1`20", "{Real}"),
      ("1*^6", "{Integer}"),
      ("1*^-6", "{Rational}"),
      ("1.5*^-6", "{Real}"),
    ] {
      assert_eq!(
        interpret(&format!(
          "ToString[Cases[CodeParser`CodeTokenize[{}], \
             LeafNode[k_, _, _] :> k], InputForm]",
          wl_string(source)
        ))
        .unwrap(),
        expected,
        "{source:?} should be read as {expected}"
      );
    }
  }

  // A named character is a letter of a symbol when it names a letter, and a
  // token of its own when it names an operator — written either as
  // `\[Rule]` or as the character that spells.
  #[test]
  fn a_named_character_is_read_as_what_it_names() {
    clear_state();
    for (source, expected) in [
      ("\\[Alpha]", "{Symbol}"),
      ("x\\[Alpha]y", "{Symbol}"),
      ("\\[Pi]", "{Symbol}"),
      ("a\\[Rule]b", "{Symbol, Token`LongName`Rule, Symbol}"),
      ("\\[Element]", "{Token`LongName`Element}"),
      ("\\[Transpose]", "{Token`LongName`Transpose}"),
      ("\\[Continuation]", "{Whitespace}"),
      ("\\[NewLine]", "{Token`Newline}"),
      ("\\[RawStar]", "{Token`Star}"),
    ] {
      assert_eq!(
        interpret(&format!(
          "ToString[Cases[CodeParser`CodeTokenize[{}], \
             LeafNode[k_, _, _] :> k], InputForm]",
          wl_string(source)
        ))
        .unwrap(),
        expected,
        "{source:?} should be read as {expected}"
      );
    }
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
