use woxi::{
  clear_state, interpret, interpret_with_stdout, split_into_statements,
};

mod interpreter_tests {
  use super::*;

  #[test]
  fn test_split_single_expression() {
    assert_eq!(split_into_statements("1 + 2"), vec!["1 + 2"]);
  }

  #[test]
  fn test_split_multiple_lines() {
    assert_eq!(
      split_into_statements(
        "Graphics[Circle[]]\n1 + 3\nGraphics[Rectangle[]]\n5 * 8"
      ),
      vec![
        "Graphics[Circle[]]",
        "1 + 3",
        "Graphics[Rectangle[]]",
        "5 * 8"
      ]
    );
  }

  #[test]
  fn test_split_preserves_multiline_brackets() {
    assert_eq!(
      split_into_statements("Module[{a = 1},\n  a + 2\n]\n3 + 4"),
      vec!["Module[{a = 1},\n  a + 2\n]", "3 + 4"]
    );
  }

  #[test]
  fn test_split_preserves_set_delayed_continuation() {
    assert_eq!(
      split_into_statements("f[x_] :=\n  x^2\nf[3]"),
      vec!["f[x_] :=\n  x^2", "f[3]"]
    );
  }

  #[test]
  fn test_split_preserves_prefix_not_continuation() {
    // `lychrelQ[n_] := !\n  palindromeQ[n]` — the `!` at end of line is
    // a prefix Not awaiting its operand on the next line, not a postfix
    // Factorial. Detected by the prev_code_char being an operator (`=`).
    assert_eq!(
      split_into_statements("f[n_] := !\n  g[n]\nf[5]"),
      vec!["f[n_] := !\n  g[n]", "f[5]"]
    );
  }

  #[test]
  fn test_split_semicolon_lines() {
    assert_eq!(
      split_into_statements("a = 5;\nb = 10;\na + b"),
      vec!["a = 5;", "b = 10;", "a + b"]
    );
  }

  #[test]
  fn test_split_blank_lines() {
    assert_eq!(
      split_into_statements("1 + 2\n\n3 + 4"),
      vec!["1 + 2", "3 + 4"]
    );
  }

  #[test]
  fn test_split_trailing_comment_only() {
    // A trailing comment-only line should not produce a separate statement
    assert_eq!(
      split_into_statements("Sin[123]\n(* comment *)"),
      vec!["Sin[123]"]
    );
  }

  #[test]
  fn test_split_leading_comment_only() {
    // A leading comment-only line should be merged with the next code line
    assert_eq!(
      split_into_statements("(* comment *)\nSin[123]"),
      vec!["(* comment *)\nSin[123]"]
    );
  }

  #[test]
  fn test_split_comment_between_expressions() {
    // A comment between two expressions should not produce an extra statement
    assert_eq!(
      split_into_statements("1 + 1\n(* comment *)\n2 + 2"),
      vec!["1 + 1", "(* comment *)\n2 + 2"]
    );
  }

  #[test]
  fn test_split_multiple_comments_only() {
    // Multiple comment-only lines should not produce statements
    assert_eq!(split_into_statements("(* c1 *)\n(* c2 *)"), vec![""]);
  }

  #[test]
  fn test_split_preserves_multiline_association() {
    assert_eq!(
      split_into_statements(
        "a = <|\n  \"x\" -> 1,\n  \"y\" -> 2\n|>\nPrint[a]"
      ),
      vec!["a = <|\n  \"x\" -> 1,\n  \"y\" -> 2\n|>", "Print[a]"]
    );
  }

  #[test]
  fn test_split_preserves_nested_multiline_association() {
    assert_eq!(
      split_into_statements("a = <|\"x\" -> <|\n  \"n\" -> 42\n|>|>\nPrint[a]"),
      vec!["a = <|\"x\" -> <|\n  \"n\" -> 42\n|>|>", "Print[a]"]
    );
  }

  #[test]
  fn test_split_backslash_line_continuation() {
    assert_eq!(
      split_into_statements(
        "ImaginaryQ[u_] :=\\\n  Head[u]===Complex && Re[u]===0\nImaginaryQ[3 I]"
      ),
      vec![
        "ImaginaryQ[u_] :=  Head[u]===Complex && Re[u]===0",
        "ImaginaryQ[3 I]"
      ]
    );
  }

  #[test]
  fn test_split_backslash_continuation_multi() {
    assert_eq!(split_into_statements("1 +\\\n2 +\\\n3"), vec!["1 +2 +3"]);
  }

  #[test]
  fn test_split_backslash_line_continuation_crlf() {
    // Line continuation should work with CRLF line endings (issue #70)
    assert_eq!(
      split_into_statements(
        "ImaginaryQ[u_] :=\\\r\n  Head[u]===Complex && Re[u]===0\r\nImaginaryQ[3 I]"
      ),
      vec![
        "ImaginaryQ[u_] :=  Head[u]===Complex && Re[u]===0",
        "ImaginaryQ[3 I]"
      ]
    );
  }

  #[test]
  fn test_split_backslash_continuation_multi_crlf() {
    // Multiple line continuations with CRLF (issue #70)
    assert_eq!(
      split_into_statements("1 +\\\r\n2 +\\\r\n3"),
      vec!["1 +2 +3"]
    );
  }

  #[test]
  fn test_interpret_line_continuation_crlf() {
    // Full interpret path with CRLF line endings (issue #70)
    assert_eq!(interpret("f[x_] :=\\\r\n  x + 1\r\nf[5]").unwrap(), "6");
  }

  #[test]
  fn test_guarded_rule_ordering_partitionsp() {
    // Issue #118: a `/;`-guarded rule must be tried before an unguarded but
    // otherwise-more-specific rule (`f[n_Integer, _] /; n<0` before
    // `f[n_Integer, r_Integer]`). Without this, the recursion never hits the
    // n<0 base case and either overflows or stays symbolic.
    clear_state();
    let program = "Unprotect[PartitionsP]\n\
      PartitionsP[n_Integer, _] := 0 /; (n<0)\n\
      PartitionsP[0, 0] := 1\n\
      PartitionsP[_, 0] := 0\n\
      PartitionsP[_, r_Integer] := 0 /; (r<0)\n\
      PartitionsP[n_Integer, 1] := 1 /; (n>0)\n\
      PartitionsP[n_Integer, 2] := Floor[n/2] /; (n>0)\n\
      PartitionsP[n_Integer, r_Integer] := PartitionsP[n-r] /; (r >= n/2)\n\
      PartitionsP[n_Integer, r_Integer] := \
        PartitionsP[n, r] = PartitionsP[n-1, r-1] + PartitionsP[n-r, r]\n\
      Table[PartitionsP[10, r], {r, 0, 10}]";
    assert_eq!(
      interpret(program).unwrap(),
      "{0, 1, 5, 8, 9, 7, 5, 3, 2, 1, 1}"
    );
    clear_state();
  }

  #[test]
  fn test_guarded_rule_downvalue_order() {
    // The guarded rule keeps its definition-order position ahead of a later
    // unguarded, otherwise-more-specific rule (issue #118).
    clear_state();
    interpret("Unprotect[gg]").unwrap();
    interpret("gg[n_Integer, _] := aa /; (n < 0)").unwrap();
    interpret("gg[n_Integer, r_Integer] := bb").unwrap();
    assert_eq!(
      interpret("DownValues[gg]").unwrap(),
      "{HoldPattern[gg[n_Integer, _]] :> aa /; n < 0, \
       HoldPattern[gg[n_Integer, r_Integer]] :> bb}"
    );
    clear_state();
  }

  #[test]
  fn test_same_pattern_guarded_rule_not_overwritten() {
    // Issue #119: redefining an unconditional rule with the SAME base pattern
    // must NOT delete a previously-defined guarded rule. In Wolfram both are
    // distinct DownValues, the guard rule keeps its (earlier) position, so a
    // canonicalizing rule `f[a_,b_] := f[b,a] /; a>b` still fires before the
    // later general `f[a_,b_] := …`.
    clear_state();
    interpret("PP[n1_Integer, n2_Integer] := PP[n2, n1] /; (n1 > n2)").unwrap();
    interpret("PP[n1_Integer, n2_Integer] := gen[n1, n2]").unwrap();
    assert_eq!(
      interpret("DownValues[PP]").unwrap(),
      "{HoldPattern[PP[n1_Integer, n2_Integer]] :> PP[n2, n1] /; n1 > n2, \
       HoldPattern[PP[n1_Integer, n2_Integer]] :> gen[n1, n2]}"
    );
    // The canonicalizing rule fires first for a descending pair.
    assert_eq!(interpret("PP[2, 5]").unwrap(), "gen[2, 5]");
    assert_eq!(interpret("PP[5, 2]").unwrap(), "gen[2, 5]");
    clear_state();
  }

  #[test]
  fn test_bipartite_partitions_canonicalizing_rule() {
    // Issue #119: full bipartite-partition recursion relying on the
    // canonicalizing rule to avoid `PP[n, 0]` divide-by-zero (the
    // `PP[n1_Integer, 0]` base case is intentionally omitted). Matches
    // wolframscript's `{2, 4, 7}`.
    clear_state();
    let program = "PP[n1_Integer, n2_Integer] := PP[n2, n1] /; (n1 > n2)\n\
      PP[n1_Integer, _] := 0 /; (n1<0)\n\
      PP[0, n2_Integer] := PartitionsP[n2]\n\
      PP[n1_Integer, n2_Integer] := PP[n1, n2] = \
        Sum[k PP[n1 - r j, n2 - r k], {j, 0, n1}, {k, n2}, {r, n2/k}]/n2\n\
      { PP[1,1], PP[1,2], PP[1,3] }";
    assert_eq!(interpret(program).unwrap(), "{2, 4, 7}");
    clear_state();
  }

  #[test]
  fn test_list_pattern_literal_element() {
    // Issue #119: a literal element inside a list pattern must be matched
    // exactly — `f[{0, n2_}]` must NOT match `f[{1, 5}]`.
    clear_state();
    interpret("f[{0, n2_Integer}] := matched0[n2]").unwrap();
    interpret("f[{n1_Integer, n2_Integer}] := general[n1, n2]").unwrap();
    assert_eq!(interpret("f[{0, 5}]").unwrap(), "matched0[5]");
    assert_eq!(interpret("f[{1, 5}]").unwrap(), "general[1, 5]");
    assert_eq!(interpret("f[{3, 5}]").unwrap(), "general[3, 5]");
    clear_state();
  }

  #[test]
  fn test_list_pattern_head_constraint() {
    // Issue #119: per-element head constraints inside a list pattern must be
    // enforced — `g[{n1_Integer, n2_Integer}]` must NOT match a non-integer
    // element.
    clear_state();
    interpret("g[{n1_Integer, n2_Integer}] := bothInt[n1, n2]").unwrap();
    assert_eq!(interpret("g[{1, 2}]").unwrap(), "bothInt[1, 2]");
    assert_eq!(interpret("g[{1, \"x\"}]").unwrap(), "g[{1, x}]");
    assert_eq!(interpret("g[{1.5, 2}]").unwrap(), "g[{1.5, 2}]");
    clear_state();
  }

  #[test]
  fn test_list_pattern_literal_more_specific_than_blank() {
    // Issue #119: a list rule with a literal element (`{1, x_}`) must take
    // priority over an all-blank list rule (`{n_, x_}`) regardless of which
    // is defined first — matching Wolfram's specificity ordering.
    clear_state();
    interpret("s[{n_, x_}] := other[n, x]").unwrap();
    interpret("s[{1, x_}] := one[x]").unwrap();
    assert_eq!(interpret("s[{1, 9}]").unwrap(), "one[9]");
    assert_eq!(interpret("s[{2, 9}]").unwrap(), "other[2, 9]");
    clear_state();
  }

  #[test]
  fn test_list_pattern_head_more_specific_than_blank() {
    // Issue #119: a head-constrained list element (`{n_Integer, x_}`) is more
    // specific than a bare blank element (`{n_, x_}`), independent of order.
    clear_state();
    interpret("g[{n_, x_}] := gen[n, x]").unwrap();
    interpret("g[{n_Integer, x_}] := hd[n, x]").unwrap();
    assert_eq!(interpret("g[{1, 9}]").unwrap(), "hd[1, 9]");
    assert_eq!(interpret("g[{1.5, 9}]").unwrap(), "gen[1.5, 9]");
    clear_state();
  }

  #[test]
  fn test_nested_list_pattern_binding() {
    // Issue #119 follow-up: nested list patterns bind their inner elements, so
    // `p[{{a_, b_}, c_}]` binds a, b, c — matching wolframscript.
    clear_state();
    interpret("p[{{a_, b_}, c_}] := f[a, b, c]").unwrap();
    assert_eq!(interpret("p[{{1, 2}, 3}]").unwrap(), "f[1, 2, 3]");
    // Wrong shape must not match.
    assert_eq!(interpret("p[{1, 2, 3}]").unwrap(), "p[{1, 2, 3}]");
    clear_state();
    interpret("q[{{a_, b_}, {c_, d_}}] := g[a, b, c, d]").unwrap();
    assert_eq!(interpret("q[{{1, 2}, {3, 4}}]").unwrap(), "g[1, 2, 3, 4]");
    clear_state();
  }

  #[test]
  fn test_list_pattern_downvalues_reconstruction() {
    // Issue #119 follow-up: DownValues/Definition reconstruct the surface
    // `{…}` list pattern (with element names, body, and `/;` guard) rather than
    // leaking the lowered `_lp0_List` / `Part[_lp0, i]` form.
    clear_state();
    interpret("g[{a_Integer, b_}] := h[a, b]").unwrap();
    assert_eq!(
      interpret("DownValues[g]").unwrap(),
      "{HoldPattern[g[{a_Integer, b_}]] :> h[a, b]}"
    );
    clear_state();
    interpret("ZZ[{n1_Integer, n2_Integer}] := ZZ[{n2, n1}] /; (n1 > n2)")
      .unwrap();
    interpret("ZZ[{n1_Integer, n2_Integer}] := gen[n1, n2]").unwrap();
    assert_eq!(
      interpret("DownValues[ZZ]").unwrap(),
      "{HoldPattern[ZZ[{n1_Integer, n2_Integer}]] :> ZZ[{n2, n1}] /; n1 > n2, \
       HoldPattern[ZZ[{n1_Integer, n2_Integer}]] :> gen[n1, n2]}"
    );
    clear_state();
    interpret("p[{{a_, b_}, c_}] := f[a, b, c]").unwrap();
    assert_eq!(
      interpret("DownValues[p]").unwrap(),
      "{HoldPattern[p[{{a_, b_}, c_}]] :> f[a, b, c]}"
    );
    clear_state();
  }

  #[test]
  fn test_list_pattern_guard_over_elements() {
    // Issue #119: a body-level `/;` guard that references destructured list
    // elements must be checked against the bound element values, so the
    // list-argument recursion produces the same result as the scalar form.
    clear_state();
    let program = "ZZ[{n1_Integer, n2_Integer}] := ZZ[{n2, n1}] /; (n1 > n2)\n\
      ZZ[{n1_Integer, _}] := 0 /; (n1<0)\n\
      ZZ[{0, n2_Integer}] := PartitionsP[n2]\n\
      ZZ[{n1_Integer, n2_Integer}] := ZZ[{n1, n2}] = \
        Sum[k ZZ[{n1 - r j, n2 - r k}], {j, 0, n1}, {k, n2}, {r, n2/k}]/n2\n\
      { ZZ[{1,1}], ZZ[{1,2}], ZZ[{1,3}], ZZ[{2,2}], ZZ[{2,3}] }";
    assert_eq!(interpret(program).unwrap(), "{2, 4, 7, 9, 16}");
    clear_state();
  }

  #[test]
  fn test_split_condition_continuation() {
    // /; (Condition) at end of line means the expression continues
    assert_eq!(
      split_into_statements("Foo[x_] :=\n  -x /;\nx > 1\nFoo[2]"),
      vec!["Foo[x_] :=\n  -x /;\nx > 1", "Foo[2]"]
    );
  }

  #[test]
  fn test_split_operator_continuation() {
    // Lines ending with operators should continue to the next line
    assert_eq!(split_into_statements("x = 1 +\n2"), vec!["x = 1 +\n2"]);
  }

  #[test]
  fn test_comment_only_input() {
    // A standalone comment should not cause an error
    clear_state();
    let result = interpret("(* comment *)");
    assert!(result.is_err());
    assert!(matches!(result, Err(woxi::InterpreterError::EmptyInput)));
  }

  #[test]
  fn test_percent_history_in_visual_mode() {
    // In visual mode (woxi-studio), `%` should resolve to the previous
    // `interpret_with_stdout` call's top-level result so cells like
    // `N[%]` work as expected. CLI mode keeps wolframscript's behaviour
    // of returning `Out[0]` (no history), which is exercised elsewhere.
    clear_state();
    woxi::clear_last_output();
    let r1 = interpret_with_stdout("2 + 3").unwrap();
    assert_eq!(r1.result, "5");
    let r2 = interpret_with_stdout("N[%]").unwrap();
    assert_eq!(r2.result, "5.");
  }

  #[test]
  fn test_percent_in_cli_mode_collapses_to_out_zero() {
    // `interpret` (CLI / wolframscript-equivalent path) must not consume
    // the visual-mode history. `%` collapses to `Out[0]` exactly as
    // wolframscript does inside a single `-code` invocation.
    clear_state();
    woxi::clear_last_output();
    let _ = interpret_with_stdout("123").unwrap(); // would populate history
    // Bare `interpret` ignores history:
    assert_eq!(interpret("%").unwrap(), "Out[0]");
  }

  #[test]
  fn test_accented_named_characters_decode() {
    // Wolfram named characters for accented Latin letters must decode to
    // their Unicode chars, so e.g. imported text ("Curaçao") compares
    // equal to source written with escapes ("Cura\[CCedilla]ao").
    clear_state();
    assert_eq!(interpret("\"Cura\\[CCedilla]ao\"").unwrap(), "Curaçao");
    assert_eq!(
      interpret("\"Cura\\[CCedilla]ao\" == \"Curaçao\"").unwrap(),
      "True"
    );
    // A lookup keyed by the escaped form must hit when queried with the
    // decoded (imported) form — the exact pattern the FIFA notebook uses.
    assert_eq!(
      interpret("Lookup[<|\"Cura\\[CCedilla]ao\" -> 0.152|>, \"Curaçao\"]")
        .unwrap(),
      "0.152"
    );
    // Spot-check a few more across the Latin-1 range.
    assert_eq!(interpret("\"\\[ODoubleDot]\"").unwrap(), "ö");
    assert_eq!(interpret("\"\\[NTilde]\"").unwrap(), "ñ");
    assert_eq!(interpret("\"\\[CapitalATilde]\\[Section]\"").unwrap(), "Ã§");
    assert_eq!(interpret("\"\\[SZ]\"").unwrap(), "ß");
  }

  #[test]
  fn test_expression_then_comment() {
    // Expression followed by comment should evaluate the expression
    clear_state();
    assert_eq!(interpret("Sin[123]\n(* comment *)").unwrap(), "Sin[123]");
  }

  #[test]
  fn test_column_with_tableform_and_headings_full_example() {
    // Regression test for the playground rendering of a Column that mixes
    // text headings, a TableForm with column headings, and trailing text.
    clear_state();
    let r = interpret_with_stdout(
      "names = {\"2\\[Euro]\", \"1\\[Euro]\", \"50c\", \"20c\"};\n\
       weights = {8.50, 7.50, 7.80, 5.74};\n\
       best = {10, 2, 0, 0};\n\
       Column[{\n\
         \"=== Fewest Euro coins to make exactly 100 g ===\",\n\
         TableForm[\n\
           Select[Transpose[{names, best, best * weights}], #[[2]] > 0 &],\n\
           TableHeadings -> {None, {\"Coin\", \"Count\", \"Weight (g)\"}}\n\
         ],\n\
         \"Total coins\"\n\
       }]",
    )
    .unwrap();
    let svg = r.graphics.expect("expected graphics output");
    assert!(svg.matches("<svg").count() >= 2);
    assert!(!svg.contains("TableForm["));
    assert!(svg.contains("Fewest Euro coins"));
    assert!(svg.contains("Coin"));
  }

  #[test]
  fn test_column_with_nested_tableform_renders_as_graphics() {
    // In visual mode (playground / woxi-studio), a Column containing a
    // TableForm must pre-render the table as a sub-SVG instead of falling
    // back to the literal `TableForm[…]` text echo.
    clear_state();
    let r =
      interpret_with_stdout("Column[{\"hello\", TableForm[{{1, 2}, {3, 4}}]}]")
        .unwrap();
    let svg = r.graphics.expect("Column should produce a graphics SVG");
    // A nested <svg> child is the marker that the TableForm got embedded
    // as a sub-SVG (vs. being stringified as plain text).
    assert!(
      svg.matches("<svg").count() >= 2,
      "Column SVG should embed the TableForm as a nested <svg>:\n{svg}"
    );
    // The text item is still rendered as a <text> element.
    assert!(
      svg.contains(">hello<"),
      "Column SVG missing text item:\n{svg}"
    );
    // The fall-back stringified table should NOT appear.
    assert!(
      !svg.contains("TableForm["),
      "Column SVG should not contain raw TableForm[…] text:\n{svg}"
    );
  }

  #[test]
  fn test_comment_then_expression() {
    // Comment followed by expression should evaluate the expression
    clear_state();
    assert_eq!(interpret("(* comment *)\nSin[123]").unwrap(), "Sin[123]");
  }

  #[test]
  fn test_inline_comment() {
    // Inline comment should not affect the result
    clear_state();
    assert_eq!(interpret("5 + (* inline *) 3").unwrap(), "8");
  }

  #[test]
  fn test_comment_after_condition_operator() {
    // A comment after /; should not cause an infinite loop
    clear_state();
    assert_eq!(interpret("x /; (* foo *) True").unwrap(), "x /; True");
  }

  #[test]
  fn test_modifier_circumflex_as_power() {
    // Regression: the modifier-letter circumflex `ˆ` (U+02C6, emitted by the
    // macOS `^` dead key) must act as the Power operator, identical to `^`.
    clear_state();
    assert_eq!(interpret("2ˆ10").unwrap(), interpret("2^10").unwrap());
    assert_eq!(interpret("2ˆ10").unwrap(), "1024");
    clear_state();
    assert_eq!(interpret("xˆ2 /. x->3").unwrap(), "9");
    clear_state();
    assert_eq!(
      interpret("r=(1.+2. I)ˆI; {r, Abs[r], Im[r]}").unwrap(),
      "{0.2291401859804338 + 0.23817011512167555*I, \
       0.3304999675767306, 0.23817011512167555}"
    );
    // The circumflex must stay literal inside string content.
    clear_state();
    assert_eq!(interpret("\"aˆb\"").unwrap(), "aˆb");
  }

  #[test]
  fn test_comment_after_condition_in_set_delayed() {
    // SetDelayed with Condition and inline comment should work
    clear_state();
    assert_eq!(
      interpret("f[x_] := x^2 /; (* positive *) True; f[3]").unwrap(),
      "9"
    );
  }

  #[test]
  fn test_condition_in_module_as_guard() {
    // Condition inside Module should act as a guard for the function definition.
    // If test is True, return the value. If not, the overload doesn't match.
    clear_state();
    interpret("Foo[u_,x_Symbol] := Module[{}, 3 /; u == 1]").unwrap();
    // x != 1, so Foo[x, x] should remain unevaluated
    assert_eq!(interpret("Foo[x, x]").unwrap(), "Foo[x, x]");
    // 1 == 1 is True, so Foo[1, x] should return 3
    assert_eq!(interpret("Foo[1, x]").unwrap(), "3");
  }

  #[test]
  fn test_condition_in_block_as_guard() {
    // Same behavior with Block instead of Module
    clear_state();
    interpret("Bar[n_] := Block[{}, n^2 /; n > 0]").unwrap();
    assert_eq!(interpret("Bar[3]").unwrap(), "9");
    assert_eq!(interpret("Bar[-1]").unwrap(), "Bar[-1]");
  }

  #[test]
  fn test_condition_in_module_with_expression() {
    // Condition guard with non-trivial expression in Module
    clear_state();
    interpret("Sqr[n_] := Module[{}, n^2 /; n > 0]").unwrap();
    assert_eq!(interpret("Sqr[3]").unwrap(), "9");
    assert_eq!(interpret("Sqr[-1]").unwrap(), "Sqr[-1]");
  }

  #[test]
  fn test_replace_repeated_with_conditional_rule_stored_in_variable() {
    // Bubble-sort-style rule with a Condition guard, stored in an OwnValue
    // and applied via //.. Regression for two bugs:
    //   1. Set was stringifying RuleDelayed, dropping the parens around
    //      `(p /; c)` and re-parsing as `Condition[p, RuleDelayed[c, body]]`.
    //   2. Pattern matching with `/;` didn't backtrack through sequence
    //      splits, so the first split `b=1, c=2` failed the condition and
    //      the rule never fired even though `b=2, c=1` satisfies `b > c`.
    clear_state();
    interpret("sort = ({a___, b_, c_, d___} /; b > c) :> {a, c, b, d}")
      .unwrap();
    assert_eq!(interpret("{1, 2, 1} //. sort").unwrap(), "{1, 1, 2}");
    assert_eq!(
      interpret("{3, 1, 2, 5, 4} //. sort").unwrap(),
      "{1, 2, 3, 4, 5}"
    );
  }

  #[test]
  fn test_match_q_with_condition_backtracks_through_sequence_splits() {
    // MatchQ must enumerate sequence splits when the LHS has a Condition,
    // returning True if any split satisfies the guard.
    clear_state();
    assert_eq!(
      interpret("MatchQ[{1, 2, 1}, {a___, b_, c_, d___} /; b > c]").unwrap(),
      "True",
    );
    assert_eq!(
      interpret("MatchQ[{1, 2, 3}, {a___, b_, c_, d___} /; b > c]").unwrap(),
      "False",
    );
  }

  #[test]
  fn test_nested_comment() {
    clear_state();
    assert_eq!(
      interpret("1 + (* outer (* inner *) outer *) 2").unwrap(),
      "3"
    );
  }

  #[test]
  fn test_nested_comment_only() {
    clear_state();
    assert!(interpret("(* outer (* inner *) *)").is_err());
  }

  #[test]
  fn test_deeply_nested_comment() {
    clear_state();
    assert_eq!(
      interpret("10 + (* a (* b (* c *) b *) a *) 5").unwrap(),
      "15"
    );
  }

  #[test]
  fn test_nested_comment_multiline() {
    clear_state();
    assert_eq!(
      interpret("1 + (* outer\n(* inner *)\nouter *) 2").unwrap(),
      "3"
    );
  }

  #[test]
  fn test_split_nested_comment() {
    assert_eq!(
      split_into_statements("1 + 1\n(* outer (* inner *) *)\n2 + 2"),
      vec!["1 + 1", "(* outer (* inner *) *)\n2 + 2"]
    );
  }

  #[test]
  fn test_multi_statement_results() {
    // When a cell has multiple expressions, each should produce output
    clear_state();
    let statements = split_into_statements("a = 1 + 2\n2^a");
    assert_eq!(statements, vec!["a = 1 + 2", "2^a"]);

    let mut results = Vec::new();
    for stmt in &statements {
      match interpret_with_stdout(stmt) {
        Ok(result) => {
          if result.result != "\0" {
            results.push(result.result);
          }
        }
        Err(_) => {}
      }
    }
    assert_eq!(results, vec!["3", "8"]);
  }

  #[test]
  fn test_unary_plus() {
    clear_state();
    assert_eq!(interpret("(+q)").unwrap(), "q");
    assert_eq!(interpret("+5").unwrap(), "5");
    assert_eq!(interpret("+x").unwrap(), "x");
    assert_eq!(interpret("1 + +2").unwrap(), "3");
    assert_eq!(interpret("+x^2").unwrap(), "x^2");
  }

  #[test]
  fn test_circle_minus() {
    clear_state();
    // CircleMinus is a symbolic operator displayed with the ⊖ glyph
    assert_eq!(interpret("CircleMinus[a, b]").unwrap(), "a \u{2296} b");
    assert_eq!(
      interpret("CircleMinus[a, b, c]").unwrap(),
      "a \u{2296} b \u{2296} c"
    );
    // Single argument stays in CircleMinus[...] form, matching wolframscript
    assert_eq!(interpret("CircleMinus[5]").unwrap(), "CircleMinus[5]");
  }

  #[test]
  fn notation_wrappers_stay_symbolic_without_warning() {
    // Notation/display wrapper heads stay unevaluated as their canonical form
    // in wolframscript and must NOT emit a spurious "not yet implemented"
    // warning (like Subscript/Superscript/Framed already behave).
    let cases = [
      ("Overscript[x, 2]", "Overscript[x, 2]"),
      ("Underscript[x, 2]", "Underscript[x, 2]"),
      ("Underoverscript[x, 1, 2]", "Underoverscript[x, 1, 2]"),
      ("Underlined[\"x\"]", "Underlined[x]"),
      ("Highlighted[\"x\"]", "Highlighted[x]"),
      ("Mouseover[a, b]", "Mouseover[a, b]"),
      ("Magnify[x, 2]", "Magnify[x, 2]"),
      ("Ket[0]", "Ket[0]"),
      ("Bra[0]", "Bra[0]"),
    ];
    for (input, expected) in cases {
      let r = interpret_with_stdout(input).unwrap();
      assert_eq!(r.result, expected, "result mismatch for {input}");
      assert!(
        !r.warnings.iter().any(|w| w.contains("not yet implemented")),
        "unexpected 'not yet implemented' warning for {input}: {:?}",
        r.warnings
      );
    }
  }

  #[test]
  fn control_wrappers_stay_symbolic_without_warning() {
    // Interactive control / display / annotation wrapper heads stay
    // unevaluated as their canonical form in wolframscript's script mode and
    // must NOT emit a spurious "not yet implemented" warning.
    let cases = [
      ("Button[a, b]", "Button[a, b]"),
      ("ActionMenu[a, b]", "ActionMenu[a, b]"),
      ("Tooltip[a, b]", "Tooltip[a, b]"),
      ("Interpretation[a, b]", "Interpretation[a, b]"),
      ("Invisible[x]", "Invisible[x]"),
      ("Subsuperscript[x, 1, 2]", "Subsuperscript[x, 1, 2]"),
      ("Deploy[x]", "Deploy[x]"),
      ("MouseAppearance[a, b]", "MouseAppearance[a, b]"),
      ("Editable[x]", "Editable[x]"),
      ("Selectable[x]", "Selectable[x]"),
      ("DynamicWrapper[a, b]", "DynamicWrapper[a, b]"),
      ("Dynamic[x]", "Dynamic[x]"),
      ("Setter[a, b]", "Setter[a, b]"),
      ("Slider[0.5]", "Slider[0.5]"),
      ("Toggler[a, b]", "Toggler[a, b]"),
      ("Manipulator[x]", "Manipulator[x]"),
      ("ColorSlider[x]", "ColorSlider[x]"),
      ("Opener[x]", "Opener[x]"),
      ("TabView[x]", "TabView[x]"),
      ("MenuView[x]", "MenuView[x]"),
      ("SlideView[x]", "SlideView[x]"),
      ("FlipView[x]", "FlipView[x]"),
    ];
    for (input, expected) in cases {
      let r = interpret_with_stdout(input).unwrap();
      assert_eq!(r.result, expected, "result mismatch for {input}");
      assert!(
        !r.warnings.iter().any(|w| w.contains("not yet implemented")),
        "unexpected 'not yet implemented' warning for {input}: {:?}",
        r.warnings
      );
    }
  }

  mod case_helpers;

  mod algebra;
  mod arg_count;
  mod arithmetic;
  mod association;
  mod attributes;
  mod batch_wrappers;
  mod calculus;
  mod cellular_automaton;
  mod column;
  mod control_flow;
  mod dataset;
  mod datetime;
  mod distributions;
  mod element_data;
  mod entity;
  mod function_application;
  mod function_definitions;
  mod functions;
  mod geometry;
  mod graph_theory;
  mod graphics;
  mod image;
  mod interval;
  mod io;
  mod large_number_and_memoization;
  mod linear_algebra;
  mod list;
  mod machine_specific;
  mod math;
  mod patterns;
  mod property;
  mod quantity;
  mod rosetta_script_fixes;
  mod row;
  mod special_functions;
  mod statistics;
  mod string;
  mod styling;
  mod syntax;
  mod tabular;
  mod turing_machine;
}
