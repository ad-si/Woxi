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
  fn test_guarded_rule_three_arg_incomparable_order() {
    // Issue #121: a `/;`-guarded rule and a structurally-more-specific
    // unguarded rule are INCOMPARABLE and must fire in definition order, even
    // when the unguarded rule has more head constraints. The guard rule was
    // entered first, so it must win for the overlapping `(-1, 3, 2)` case.
    clear_state();
    interpret("g3[a_Integer, _, _] := neg /; (a < 0)").unwrap();
    interpret("g3[a_Integer, b_Integer, c_Integer] := three").unwrap();
    assert_eq!(interpret("g3[-1, 3, 2]").unwrap(), "neg");
    assert_eq!(interpret("g3[1, 3, 2]").unwrap(), "three");
    // Reversed definition order: the unguarded rule is entered first and wins,
    // since the two rules are incomparable (entry order is preserved).
    clear_state();
    interpret("h3[a_Integer, b_Integer, c_Integer] := three").unwrap();
    interpret("h3[a_Integer, _, _] := neg /; (a < 0)").unwrap();
    assert_eq!(interpret("h3[-1, 3, 2]").unwrap(), "three");
    assert_eq!(interpret("h3[1, 3, 2]").unwrap(), "three");
    clear_state();
  }

  #[test]
  fn test_nested_pattern_more_specific_than_blank_either_order() {
    // A nested structural pattern (`f[g[x_]]`) is more specific than a bare
    // blank (`f[x_]`) and must fire first regardless of definition order. The
    // partial-order insertion falls back to the specificity score for
    // structural patterns, so this keeps working in both orders.
    clear_state();
    interpret("f7[g[x_]] := ng[x]").unwrap();
    interpret("f7[x_] := gen[x]").unwrap();
    assert_eq!(interpret("f7[g[5]]").unwrap(), "ng[5]");
    assert_eq!(interpret("f7[5]").unwrap(), "gen[5]");
    clear_state();
    // Reversed: the general rule is entered first, but the structural pattern
    // still wins for `f8[g[5]]`.
    interpret("f8[x_] := gen[x]").unwrap();
    interpret("f8[g[x_]] := ng[x]").unwrap();
    assert_eq!(interpret("f8[g[5]]").unwrap(), "ng[5]");
    assert_eq!(interpret("f8[5]").unwrap(), "gen[5]");
    clear_state();
  }

  #[test]
  fn test_exact_arity_more_specific_than_optional_arg() {
    // `f[x_]` (exact arity 1) is more specific than `f[x_, y_:0]` (which can
    // default `y`), so a single-arg call fires `f[x_]` regardless of definition
    // order; a two-arg call still falls to the optional rule. Matches WL.
    clear_state();
    interpret("f1[x_, y_:0] := opt[x, y]").unwrap();
    interpret("f1[x_] := one[x]").unwrap();
    assert_eq!(interpret("f1[5]").unwrap(), "one[5]");
    assert_eq!(interpret("f1[5, 6]").unwrap(), "opt[5, 6]");
    clear_state();
    // Reversed definition order yields the same dispatch.
    interpret("f2[x_] := one[x]").unwrap();
    interpret("f2[x_, y_:0] := opt[x, y]").unwrap();
    assert_eq!(interpret("f2[5]").unwrap(), "one[5]");
    assert_eq!(interpret("f2[5, 6]").unwrap(), "opt[5, 6]");
    clear_state();
  }

  #[test]
  fn test_optional_arg_overload_kept_distinct() {
    // `f[x_, y_]` and `f[x_, y_:0]` are distinct DownValues — defining the
    // optional-arg rule must NOT delete the exact-arity rule. For a two-arg call
    // the exact rule wins; for a one-arg call only the optional rule applies.
    clear_state();
    interpret("q1[x_, y_] := req2[x, y]").unwrap();
    interpret("q1[x_, y_:0] := opt[x, y]").unwrap();
    assert_eq!(interpret("q1[5]").unwrap(), "opt[5, 0]");
    assert_eq!(interpret("q1[5, 6]").unwrap(), "req2[5, 6]");
    clear_state();
    // Redefining the SAME optional-arg pattern still replaces it.
    interpret("q4[x_, y_:0] := a[x, y]").unwrap();
    interpret("q4[x_, y_:0] := b[x, y]").unwrap();
    assert_eq!(interpret("q4[3]").unwrap(), "b[3, 0]");
    assert_eq!(interpret("q4[3, 4]").unwrap(), "b[3, 4]");
    clear_state();
  }

  #[test]
  fn test_nested_pattern_inner_head_specificity() {
    // Among nested structural patterns, a tighter inner pattern wins: `g[x_Integer]`
    // is more specific than `g[x_]` and fires for an integer argument, while a
    // non-integer falls to the looser rule. Matches wolframscript.
    clear_state();
    interpret("f12[g[x_]] := a[x]").unwrap();
    interpret("f12[g[x_Integer]] := b[x]").unwrap();
    assert_eq!(interpret("f12[g[5]]").unwrap(), "b[5]");
    assert_eq!(interpret("f12[g[1.5]]").unwrap(), "a[1.5]");
    clear_state();
  }

  #[test]
  fn test_bipartite_partitions_three_arg_recursion() {
    // Issue #121: the three-index bipartite-partition recursion relies on the
    // `BiPartitionsP[n1_Integer, _, _] := 0 /; n1<0` guard firing before the
    // unguarded memoizing rule. A wrong order made `BiPartitionsP[-1, 3, 2]`
    // evaluate to 1, inflating the total. Matches wolframscript exactly.
    clear_state();
    let program = "Unprotect[PartitionsP]\n\
      PartitionsP[n_Integer, _] := 0 /; (n < 0)\n\
      PartitionsP[0, 0] := 1\n\
      PartitionsP[_, 0] := 0\n\
      PartitionsP[_, r_Integer] := 0 /; (r < 0)\n\
      PartitionsP[n_Integer, 1] := 1 /; (n > 0)\n\
      PartitionsP[n_Integer, 2] := Floor[n/2] /; (n > 0)\n\
      PartitionsP[n_Integer, r_Integer] := PartitionsP[n-r] /; (r >= n/2)\n\
      PartitionsP[n_Integer, r_Integer] := \
        PartitionsP[n, r] = PartitionsP[n-1, r-1] + PartitionsP[n-r, r]\n\
      BiPartitionsP[n1_Integer, n2_Integer, r_Integer] := \
        BiPartitionsP[n2, n1, r] /; (n1 > n2)\n\
      BiPartitionsP[n1_Integer, _, _] := 0 /; (n1 < 0)\n\
      BiPartitionsP[0, n2_Integer, r_Integer] := PartitionsP[n2, r]\n\
      BiPartitionsP[0, 0, 0] := 1\n\
      BiPartitionsP[_, _, 0] := 0\n\
      BiPartitionsP[0, 0, _] := 0\n\
      BiPartitionsP[_, _, 1] := 1\n\
      BiPartitionsP[n1_Integer, n2_Integer, r_Integer] := \
        0 /; ((r < 0) || (r > n1+n2))\n\
      BiPartitionsP[n1_Integer, n2_Integer, r_Integer] := \
        BiPartitionsP[n1, n2, r] = BiPartitionsP[n1-1, n2, r-1] + \
        BiPartitionsP[n1-r, n2, r] + \
        Sum[BiPartitionsP[n1-i, n2-j, i] PartitionsP[j, r-i], \
          {i, Min[r-1, n1]}, {j, r-i, Min[n2, n1+n2-2i]}]\n\
      { BiPartitionsP[-1, 3, 2], \
        Table[BiPartitionsP[2, 3, r], {r, 0, 5}], \
        Sum[BiPartitionsP[2, 3, r], {r, 0, 5}] }";
    assert_eq!(interpret(program).unwrap(), "{0, {0, 1, 5, 6, 3, 1}, 16}");
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
  fn test_part_partw_mirrored_to_captured_stdout() {
    // wolframscript prints Part::partw to stdout in script mode, so the
    // library path (interpret_with_stdout — snapshot tests, playground,
    // Jupyter) must capture it too. Regression test for the
    // stem-and-leaf_plot.wls snapshot divergence.
    clear_state();
    let r = interpret_with_stdout("RealDigits[Quotient[x, 10]][[2]]").unwrap();
    assert_eq!(
      r.stdout,
      "\nPart::partw: Part 2 of RealDigits[Quotient[x, 10]] does not \
       exist.\n"
    );
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
  fn test_export_graphic_does_not_render_inline() {
    // Exporting a graphic (e.g. BarChart) to a file writes the file and
    // returns the filename; it must NOT also surface the chart as inline
    // graphics in visual frontends (playground, woxi-studio). Evaluating the
    // second argument populates the capture buffer, so Export has to drop that
    // entry.
    clear_state();
    let path = std::env::temp_dir().join("woxi_test_export_barchart.svg");
    let code = format!(
      "Export[\"{}\", BarChart[{{5, 8, 3, 9, 6, 4, 7}}]]",
      path.display()
    );
    let r = interpret_with_stdout(&code).unwrap();
    assert_eq!(r.result, path.display().to_string());
    assert!(
      r.graphics.is_none(),
      "Export should not surface inline graphics, got:\n{:?}",
      r.graphics
    );
    // The file itself must still contain the rendered chart.
    let written = std::fs::read_to_string(&path).unwrap();
    assert!(written.contains("<svg"), "exported file should be an SVG");
    std::fs::remove_file(&path).ok();
  }

  #[test]
  fn test_piechart_chartstyle_and_chartlabels() {
    // Regression: PieChart must honor `ChartStyle` (per-slice fill colors,
    // keyed by data index) and `ChartLabels` (text drawn on each wedge).
    clear_state();
    let svg = interpret_with_stdout(
      "PieChart[{1, 2, 3, 4}, \
       ChartStyle -> {Pink, LightBlue, LightGreen, LightOrange}, \
       ChartLabels -> {\"one\", \"two\", \"three\", \"four\"}]",
    )
    .unwrap()
    .graphics
    .expect("PieChart should produce a graphics SVG");
    // ChartStyle colors, one per data index (Pink, LightBlue, LightGreen,
    // LightOrange). The default PLOT_COLORS palette must not appear.
    for rgb in [
      "rgb(255,128,128)", // Pink
      "rgb(222,240,255)", // LightBlue
      "rgb(224,255,224)", // LightGreen
      "rgb(255,230,204)", // LightOrange
    ] {
      assert!(
        svg.contains(rgb),
        "PieChart SVG missing ChartStyle color {rgb}:\n{svg}"
      );
    }
    // ChartLabels rendered as wedge text.
    for label in ["one", "two", "three", "four"] {
      assert!(
        svg.contains(&format!(">{label}</text>")),
        "PieChart SVG missing ChartLabels text `{label}`:\n{svg}"
      );
    }
  }

  #[test]
  fn test_piechart_input_order_and_black_border() {
    // Regression: PieChart draws slices in the order given by the value
    // array (no smallest-to-largest sorting) and every wedge has a black
    // border by default, matching wolframscript (EdgeForm GrayLevel[0]).
    clear_state();
    let svg = interpret_with_stdout("PieChart[{30, 20, 10}]")
      .unwrap()
      .graphics
      .expect("PieChart should produce a graphics SVG");
    // Borders must be black, never white.
    assert!(
      !svg.contains("stroke=\"white\""),
      "PieChart wedges must use a black border:\n{svg}"
    );
    assert!(
      svg.contains("stroke=\"black\""),
      "PieChart wedges must use a black border:\n{svg}"
    );
    // Slices appear in input order: the first `<title>` is 30, then 20, 10.
    let order: Vec<&str> = svg
      .match_indices("<title>")
      .map(|(i, _)| {
        let rest = &svg[i + "<title>".len()..];
        &rest[..rest.find("</title>").unwrap()]
      })
      .collect();
    assert_eq!(
      order,
      vec!["30", "20", "10"],
      "PieChart slices must follow the input value order:\n{svg}"
    );
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
  fn test_tableform_decimal_alignment_lines_up_dots() {
    // `TableAlignments -> "."` must line up the numbers on their decimal
    // point. Each cell is start-anchored in the SVG, so the dot's x-position
    // is `x + (chars before the dot) * char_width` (char_width = 8.4). All
    // cells must share the same dot x.
    clear_state();
    let svg = interpret(
      "ExportString[TableForm[0.12345 * 10^Range[4], TableAlignments -> \".\"], \"SVG\"]",
    )
    .unwrap();

    let char_width = 8.4_f64;
    let mut dot_positions: Vec<f64> = Vec::new();
    for chunk in svg.split("<text ").skip(1) {
      // Parse the `x="…"` attribute.
      let x_val = chunk
        .split("x=\"")
        .nth(1)
        .and_then(|s| s.split('"').next())
        .and_then(|s| s.parse::<f64>().ok())
        .expect("text element must have an x attribute");
      // Parse the element's text content.
      let content = chunk
        .split('>')
        .nth(1)
        .and_then(|s| s.split('<').next())
        .expect("text element must have content");
      // Only numeric cells participate in decimal alignment.
      let int_chars = match content.find('.') {
        Some(pos) => content[..pos].chars().count(),
        None => content.chars().count(),
      };
      dot_positions.push(x_val + int_chars as f64 * char_width);
    }

    assert_eq!(dot_positions.len(), 4, "expected 4 rows:\n{svg}");
    let first = dot_positions[0];
    for dp in &dot_positions {
      assert!(
        (dp - first).abs() < 1e-6,
        "decimal points must line up; got {dot_positions:?}\n{svg}"
      );
    }
  }

  #[test]
  fn test_real_output_svg_has_no_precision_backtick() {
    // Regression: the playground/Studio SVG for results containing machine
    // Reals (e.g. `NMinimize[(x-1)^2, x]` → `{0., {x -> 1.}}`) must not show
    // the box-form precision marker backtick (`0.``/`1.``). The typeset
    // display suppresses it.
    clear_state();
    let r = interpret_with_stdout("NMinimize[(x - 1)^2, x]").unwrap();
    let svg = r
      .output_svg
      .expect("expected output SVG for real-valued result");
    assert!(
      !svg.contains('`'),
      "Real result SVG must not contain a precision-marker backtick:\n{svg}"
    );
    // The numeric values themselves are still present.
    assert!(svg.contains("0.") && svg.contains("1."), "SVG: {svg}");
  }

  #[test]
  fn test_scientific_real_output_svg_uses_superscript() {
    // Regression: a machine Real in scientific notation (`10.^10` → `1.*^10`)
    // must be typeset as `1. × 10^10` in the Playground/Studio SVG — a `×`
    // factor with the exponent as a smaller superscript — rather than the raw
    // InputForm `*^` operator.
    for code in ["10.^10", "1.5*^-8", "3.4*^10"] {
      clear_state();
      let svg = interpret_with_stdout(code)
        .unwrap()
        .output_svg
        .unwrap_or_else(|| panic!("expected output SVG for {code}"));
      assert!(
        !svg.contains("*^"),
        "scientific SVG for {code} must not contain the literal `*^`:\n{svg}"
      );
      assert!(
        svg.contains('\u{00d7}'),
        "scientific SVG for {code} must contain the × factor:\n{svg}"
      );
      // The exponent renders in the reduced superscript font size (14 * 0.7).
      assert!(
        svg.contains("font-size=\"9.8\""),
        "scientific SVG for {code} must have a superscript exponent:\n{svg}"
      );
    }
  }

  #[test]
  fn test_large_number_output_svg_groups_digits() {
    // The Wolfram notebook groups the integer part of large numbers into
    // 3-digit blocks (`10^10` → `10 000 000 000`). In the Playground/Studio SVG
    // each group renders as its own `<text>` atom, so the full ungrouped run
    // never appears and the leading/interior groups do.
    clear_state();
    let svg = interpret_with_stdout("10^10").unwrap().output_svg.unwrap();
    assert!(
      !svg.contains(">10000000000<"),
      "large integer digits must be grouped:\n{svg}"
    );
    assert!(
      svg.contains(">10<") && svg.contains(">000<"),
      "expected 3-digit groups:\n{svg}"
    );

    // Grouping starts at five integer digits: `10^3` (1000) stays ungrouped,
    // `10^4` (10000) becomes `10 000`.
    clear_state();
    let four = interpret_with_stdout("10^3").unwrap().output_svg.unwrap();
    assert!(
      four.contains(">1000<"),
      "4-digit number must not group:\n{four}"
    );
    clear_state();
    let five = interpret_with_stdout("10^4").unwrap().output_svg.unwrap();
    assert!(
      five.contains(">10<") && five.contains(">000<"),
      "5-digit number must group:\n{five}"
    );

    // A non-scientific real groups only its integer part (`10.^5` → `100 000.`).
    clear_state();
    let real = interpret_with_stdout("10.^5").unwrap().output_svg.unwrap();
    assert!(
      real.contains(">100<") && real.contains(">000.<"),
      "real integer part must group, fractional dot kept:\n{real}"
    );
  }

  #[test]
  fn test_bare_literal_output_svg_groups_digits() {
    // A bare number literal (which the interpreter serves from a fast path)
    // still gets a typeset, digit-grouped SVG in visual hosts, like a computed
    // result: `10000` → `10 000`, `100000.` → `100 000.`, and a list literal
    // `{10000, 20000}` groups each element.
    for (code, present, absent) in [
      ("10000", ">10<", ">10000<"),
      ("100000.", ">000.<", ">100000.<"),
      ("{10000, 20000}", ">000<", ">10000<"),
    ] {
      clear_state();
      let svg = interpret_with_stdout(code)
        .unwrap()
        .output_svg
        .unwrap_or_else(|| {
          panic!("bare literal {code} should have an output SVG")
        });
      assert!(
        svg.contains(present),
        "{code}: expected {present} in:\n{svg}"
      );
      assert!(
        !svg.contains(absent),
        "{code}: must not contain {absent}:\n{svg}"
      );
    }
    // Below the 5-digit threshold a bare literal stays ungrouped.
    clear_state();
    let small = interpret_with_stdout("1000").unwrap().output_svg.unwrap();
    assert!(
      small.contains(">1000<"),
      "1000 must stay ungrouped:\n{small}"
    );
  }

  #[test]
  fn test_play_synthesizes_audio_in_visual_mode() {
    // In visual mode (playground / woxi-studio), Play[f, {t, …}] synthesizes a
    // playable WAV exposed via the `sound` channel instead of the -Sound- echo.
    clear_state();
    let r = interpret_with_stdout("Play[Sin[440*2*Pi*t], {t, 0, 1}]").unwrap();
    let audio = r.sound.expect("Play should produce synthesized audio");
    assert_eq!(audio.mime, "audio/wav");
    // Decoded bytes start with the RIFF/WAVE magic.
    let bytes = base64::engine::Engine::decode(
      &base64::engine::general_purpose::STANDARD,
      &audio.base64,
    )
    .expect("sound should be valid base64");
    assert_eq!(&bytes[0..4], b"RIFF", "WAV should start with RIFF magic");
    assert_eq!(&bytes[8..12], b"WAVE", "WAV should declare WAVE format");
    // 1 second at 8000 Hz, 16-bit mono => 44-byte header + 8000*2 data bytes.
    assert_eq!(bytes.len(), 44 + 8000 * 2);
  }

  #[test]
  fn test_sound_list_of_plays_synthesizes_audio_in_visual_mode() {
    // Sound[{Play[…], Play[…]}] concatenates its segments into one WAV.
    clear_state();
    let r = interpret_with_stdout(
      "Sound[{Play[Sin[1000*t], {t, 0, 0.2}], Play[Sin[500*t], {t, 0, 0.5}]}]",
    )
    .unwrap();
    let audio = r.sound.expect("Sound should produce synthesized audio");
    let bytes = base64::engine::Engine::decode(
      &base64::engine::general_purpose::STANDARD,
      &audio.base64,
    )
    .expect("sound should be valid base64");
    // 0.2s + 0.5s = 0.7s at 8000 Hz, 16-bit mono.
    assert_eq!(bytes.len(), 44 + (8000 * 7 / 10) * 2);
  }

  #[test]
  fn test_list_play_synthesizes_audio_in_visual_mode() {
    // In visual mode (playground / woxi-studio), ListPlay[{levels…}] is
    // normalized, encoded as a WAV, and exposed via the `sound` channel so the
    // hosts render a playable audio widget instead of the -Sound- echo.
    clear_state();
    let r = interpret_with_stdout("ListPlay[{0.1, 0.2, 0.3, -0.1}]").unwrap();
    let audio = r.sound.expect("ListPlay should produce synthesized audio");
    let bytes = base64::engine::Engine::decode(
      &base64::engine::general_purpose::STANDARD,
      &audio.base64,
    )
    .expect("sound should be valid base64");
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    // Default ListPlay sample rate is 8000 Hz.
    assert_eq!(u32::from_le_bytes(bytes[24..28].try_into().unwrap()), 8000);
    // Four normalized samples, 16-bit mono, byte-verified against wolframscript.
    assert_eq!(bytes.len(), 44 + 4 * 2);
    assert_eq!(&bytes[44..52], &[0, 0, 0, 64, 255, 127, 0, 128]);
  }

  #[test]
  fn test_list_play_target_waveform_in_visual_mode() {
    // The motivating example: a 50 Hz sine sampled at 2000 Hz over 1 second
    // (2001 samples) plays as a Sound in the visual hosts.
    clear_state();
    let r = interpret_with_stdout(
      "ListPlay[Table[Sin[2 Pi 50 t], {t, 0, 1, 1./2000}]]",
    )
    .unwrap();
    assert_eq!(r.result, "-Sound-");
    let audio = r.sound.expect("ListPlay should produce synthesized audio");
    let bytes = base64::engine::Engine::decode(
      &base64::engine::general_purpose::STANDARD,
      &audio.base64,
    )
    .expect("sound should be valid base64");
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(u32::from_le_bytes(bytes[24..28].try_into().unwrap()), 8000);
    // 2001 samples, 16-bit mono.
    assert_eq!(bytes.len(), 44 + 2001 * 2);
  }

  /// Decode the base64 WAV payload of a captured audio output.
  fn decode_wav_bytes(audio: &woxi::AudioOutput) -> Vec<u8> {
    assert_eq!(audio.mime, "audio/wav");
    base64::engine::Engine::decode(
      &base64::engine::general_purpose::STANDARD,
      &audio.base64,
    )
    .expect("sound should be valid base64")
  }

  #[test]
  fn test_audio_from_samples_renders_player_in_visual_mode() {
    // In visual mode (playground / woxi-studio), Audio[{samples…}] is encoded
    // as a WAV and exposed via the `sound` channel so the hosts render a
    // graphical audio player. The CLI keeps the symbolic Audio[…] form.
    clear_state();
    let r = interpret_with_stdout("Audio[{0, 0.5, -0.5, 1}]").unwrap();
    assert_eq!(r.result, "-Audio-");
    let audio = r.sound.expect("Audio should produce playable audio");
    assert_eq!(audio.label, None);
    let bytes = decode_wav_bytes(&audio);
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    // Default sample rate for sample-data Audio is 44100 Hz.
    assert_eq!(u32::from_le_bytes(bytes[24..28].try_into().unwrap()), 44100);
    // 4 samples, 16-bit mono.
    assert_eq!(bytes.len(), 44 + 4 * 2);
  }

  #[test]
  fn test_audio_sample_rate_option_sets_wav_rate() {
    clear_state();
    let r =
      interpret_with_stdout("Audio[{0, 1, 0}, SampleRate -> 8000]").unwrap();
    let audio = r.sound.expect("Audio should produce playable audio");
    let bytes = decode_wav_bytes(&audio);
    assert_eq!(u32::from_le_bytes(bytes[24..28].try_into().unwrap()), 8000);
  }

  #[test]
  fn test_audio_multichannel_samples_encode_interleaved_wav() {
    clear_state();
    let r = interpret_with_stdout("Audio[{{0, 1, 0}, {1, 0, 1}}]").unwrap();
    let audio = r.sound.expect("Audio should produce playable audio");
    let bytes = decode_wav_bytes(&audio);
    // Channel count lives in bytes 22..24 of the fmt chunk.
    assert_eq!(u16::from_le_bytes(bytes[22..24].try_into().unwrap()), 2);
    // 3 frames × 2 channels × 16-bit.
    assert_eq!(bytes.len(), 44 + 3 * 2 * 2);
  }

  #[test]
  fn test_audio_file_renders_player_in_visual_mode() {
    // Audio[File["path"]] embeds the file's bytes so visual hosts render a
    // graphical audio player that can actually play the file.
    clear_state();
    let wav = interpret_with_stdout("Audio[{0, 1, 0}]")
      .unwrap()
      .sound
      .unwrap();
    let bytes = decode_wav_bytes(&wav);
    let path = std::env::temp_dir().join("woxi_test_audio_file.wav");
    std::fs::write(&path, &bytes).unwrap();

    clear_state();
    let r =
      interpret_with_stdout(&format!("Audio[File[\"{}\"]]", path.display()))
        .unwrap();
    assert_eq!(r.result, "-Audio-");
    let audio = r.sound.expect("file-backed Audio should produce audio");
    assert_eq!(audio.mime, "audio/wav");
    assert_eq!(audio.label.as_deref(), Some("woxi_test_audio_file.wav"));
    // The player carries the file's bytes verbatim.
    assert_eq!(decode_wav_bytes(&audio), bytes);
    std::fs::remove_file(&path).ok();
  }

  #[test]
  fn test_audio_file_path_string_renders_player_in_visual_mode() {
    // A bare string with an audio extension works like File["path"].
    clear_state();
    let wav = interpret_with_stdout("Audio[{0, 1, 0}]")
      .unwrap()
      .sound
      .unwrap();
    let bytes = decode_wav_bytes(&wav);
    let path = std::env::temp_dir().join("woxi_test_audio_str.wav");
    std::fs::write(&path, &bytes).unwrap();

    clear_state();
    let r =
      interpret_with_stdout(&format!("Audio[\"{}\"]", path.display())).unwrap();
    assert_eq!(r.result, "-Audio-");
    let audio = r.sound.expect("file-backed Audio should produce audio");
    assert_eq!(audio.label.as_deref(), Some("woxi_test_audio_str.wav"));
    assert_eq!(decode_wav_bytes(&audio), bytes);
    std::fs::remove_file(&path).ok();
  }

  #[test]
  fn test_export_image_to_svg_embeds_png() {
    // Exporting an Image to an .svg file must produce a valid SVG that wraps
    // the raster pixels as a base64-encoded PNG <image> element, rather than
    // erroring because the image crate has no SVG raster encoder.
    clear_state();
    let path = std::env::temp_dir().join("woxi_test_export_image.svg");
    let _ = std::fs::remove_file(&path);
    let code = format!(
      "Export[\"{}\", Image[ConstantArray[{{0, 1, 0.5}}, {{4, 4}}]]]",
      path.display()
    );
    // Export returns the filename it wrote to.
    assert_eq!(interpret(&code).unwrap(), path.display().to_string());

    let svg = std::fs::read_to_string(&path).unwrap();
    // Matches wolframscript, which opens the file with the XML declaration.
    assert!(
      svg.starts_with("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<svg"),
      "not an SVG document: {}",
      &svg[..40.min(svg.len())]
    );
    assert!(
      svg.contains("width='4'") && svg.contains("height='4'"),
      "wrong dims"
    );
    assert!(
      svg.contains("data:image/png;base64,"),
      "raster not embedded as a base64 PNG"
    );
    std::fs::remove_file(&path).ok();
  }

  #[test]
  fn test_export_string_image_svg_embeds_png() {
    // ExportString[image, "SVG"] uses the same embedded-PNG rendering.
    clear_state();
    let svg = interpret(
      "ExportString[Image[ConstantArray[{0, 1, 0.5}, {2, 2}]], \"SVG\"]",
    )
    .unwrap();
    assert!(
      svg.starts_with("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<svg")
    );
    assert!(svg.contains("data:image/png;base64,"));
  }

  #[test]
  fn test_export_string_svg_embeds_used_fonts() {
    // A standalone exported SVG carries the fonts it uses so it renders the
    // same on systems where they aren't installed. Any text pulls in the
    // sans-serif face (Atkinson Hyperlegible Next) as an @font-face with the
    // font bytes inlined as a base64 data URL.
    clear_state();
    let svg =
      interpret("ExportString[Plot[Sin[x], {x, 0, 6}], \"SVG\"]").unwrap();
    assert!(svg.contains("@font-face"), "no @font-face block");
    assert!(
      svg.contains("font-family: \"Atkinson Hyperlegible Next\""),
      "sans-serif face not embedded"
    );
    assert!(
      svg.contains("src: url(\"data:font/ttf;base64,"),
      "font bytes not inlined as a data URL"
    );
    // The style block sits inside the SVG document, right after the root tag.
    assert!(svg.contains("<defs><style"), "no <style> block");
  }

  #[test]
  fn test_export_string_svg_embeds_monospace_only_when_used() {
    // The Mono face is embedded only for documents that actually use
    // monospace text (typeset expressions, datasets, …), not for every graphic.
    clear_state();
    let mono =
      interpret("ExportString[Dataset[<|\"a\" -> 1|>], \"SVG\"]").unwrap();
    assert!(
      mono.contains("font-family: \"Atkinson Hyperlegible Mono\""),
      "monospace face missing for a monospace-using export"
    );

    clear_state();
    let sans =
      interpret("ExportString[Plot[Sin[x], {x, 0, 6}], \"SVG\"]").unwrap();
    assert!(
      !sans.contains("Atkinson Hyperlegible Mono"),
      "Mono face embedded into a graphic that uses no monospace text"
    );

    // A text label that merely *reads* "monospace" must not pull in the Mono
    // face — only a `font-family` requesting one does. (The label itself is
    // drawn with the default sans-serif family.)
    clear_state();
    let label = interpret(
      "ExportString[Graphics[{Text[\"monospace\", {0, 0}]}], \"SVG\"]",
    )
    .unwrap();
    assert!(
      !label.contains("Atkinson Hyperlegible Mono"),
      "Mono face embedded because of text content, not a font-family"
    );
  }

  #[test]
  fn test_export_string_svg_without_text_embeds_no_fonts() {
    // A text-free graphic needs no fonts, so none are embedded.
    clear_state();
    let svg = interpret("ExportString[Graphics[{Disk[]}], \"SVG\"]").unwrap();
    assert!(
      !svg.contains("@font-face"),
      "fonts embedded into a text-free graphic"
    );
  }

  #[test]
  fn test_plot_aspect_ratio_sizes_frame_not_canvas() {
    // AspectRatio sets the height/width ratio of the plotting *area* (the data
    // frame), not the whole image. A short ratio must therefore NOT squash the
    // plot: the total image is the frame plus the label/tick margins, so its
    // height exceeds `width * ratio`. Regression for a bug where AspectRatio
    // sized the entire canvas, collapsing ticks/data into a thin band.
    let dims = |svg: &str| -> (f64, f64) {
      let grab = |attr: &str| -> f64 {
        let key = format!("{attr}=\"");
        let start = svg.find(&key).expect("attr present") + key.len();
        let end = svg[start..].find('"').unwrap() + start;
        svg[start..end].parse().unwrap()
      };
      (grab("width"), grab("height"))
    };

    clear_state();
    let short = interpret(
      "ExportString[Plot[Sin[x], {x, 0, 4 Pi}, AspectRatio -> 1/3], \"SVG\"]",
    )
    .unwrap();
    let (w, h) = dims(&short);
    // Frame height alone would be w/3; the real image must be taller because
    // axis ticks and labels live outside the frame.
    assert!(
      h > w / 3.0 + 30.0,
      "AspectRatio 1/3 collapsed the frame: {w}x{h} (expected height > w/3 + margins)"
    );

    // A taller ratio yields a taller image, and the height grows linearly with
    // the ratio (frame = w' * ratio, margins constant) — never proportional to
    // the whole canvas.
    clear_state();
    let tall = interpret(
      "ExportString[Plot[Sin[x], {x, 0, 4 Pi}, AspectRatio -> 2/3], \"SVG\"]",
    )
    .unwrap();
    let (_, h2) = dims(&tall);
    assert!(
      h2 > h,
      "doubling AspectRatio did not increase image height: {h} -> {h2}"
    );

    // ListPlot / ListLinePlot honor AspectRatio the same way (previously they
    // ignored it and stayed at the default height).
    for head in ["ListPlot", "ListLinePlot"] {
      clear_state();
      let with_ar = interpret(&format!(
        "ExportString[{head}[Table[Sin[t], {{t, 0, 10, 0.2}}], AspectRatio -> 1/3], \"SVG\"]"
      ))
      .unwrap();
      let (lw, lh) = dims(&with_ar);
      assert!(
        lh < lw && (lh - lw / 3.0).abs() < lw / 3.0,
        "{head} ignored AspectRatio 1/3: {lw}x{lh}"
      );
    }
  }

  #[test]
  fn test_audio_missing_file_still_renders_player_chrome() {
    // A file-backed Audio whose file cannot be read (missing here; any local
    // path in the browser playground) still renders the player chrome: the
    // audio output is captured with an empty payload and the file's name.
    clear_state();
    let r =
      interpret_with_stdout("Audio[File[\"/nonexistent/jazz_no1.flac\"]]")
        .unwrap();
    assert_eq!(r.result, "-Audio-");
    let audio = r.sound.expect("file-backed Audio should produce audio");
    assert_eq!(audio.base64, "");
    assert_eq!(audio.mime, "audio/flac");
    assert_eq!(audio.label.as_deref(), Some("jazz_no1.flac"));
  }

  #[test]
  fn test_audio_symbolic_data_keeps_text_echo_in_visual_mode() {
    // Audio with non-numeric data cannot become a player and keeps its
    // symbolic echo even in visual mode.
    clear_state();
    let r = interpret_with_stdout("Audio[{a, b}]").unwrap();
    assert_eq!(r.result, "Audio[{a, b}]");
    assert!(r.sound.is_none());
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
  fn test_replace_all_descends_into_rule_and_association() {
    // Regression: a blank pattern (x_Real, x_Integer, ...) must descend into
    // the pattern/replacement of a Rule subexpression and into Association
    // keys/values, matching wolframscript. Previously the AST pattern path
    // handled only List/FunctionCall/BinaryOp and fell through to a
    // string-based fallback that never reached inside a Rule.
    clear_state();
    // Into a bare Rule's replacement.
    assert_eq!(
      interpret("({0} -> {1.5, 2.5}) /. x_Real :> Round[x]").unwrap(),
      "{0} -> {2, 2}",
    );
    // Into a Rule nested in a list.
    assert_eq!(interpret("{a -> 1.5} /. x_Real :> 9").unwrap(), "{a -> 9}");
    assert_eq!(interpret("(a -> 5) /. x_Integer :> 9").unwrap(), "a -> 9");
    // Into an Association value.
    assert_eq!(
      interpret("<|k -> 1.5|> /. x_Real :> 9").unwrap(),
      "<|k -> 9|>",
    );
    // A symbol blank descends into the Rule parts (a, b) AND its `Rule` head,
    // matching wolframscript — rather than the buggy string fallback binding
    // the whole `a -> b` as a Symbol. See test_replace_all_rewrites_head_symbol.
    assert_eq!(
      interpret("(a -> b) /. x_Symbol :> foo[x]").unwrap(),
      "foo[Rule][foo[a], foo[b]]",
    );
  }

  #[test]
  fn test_replace_all_rewrites_head_symbol() {
    // ReplaceAll treats a compound's head as an ordinary subexpression, so a
    // symbol-blank rule rewrites the head too, producing `f[h][...]` (a
    // CurriedCall) exactly like wolframscript. Regression for woxi previously
    // leaving heads untouched under `x_Symbol :> ...`.
    clear_state();
    assert_eq!(
      interpret("h[a, b] /. x_Symbol :> f[x]").unwrap(),
      "f[h][f[a], f[b]]",
    );
    // List / Rule heads are subexpressions too.
    assert_eq!(
      interpret("{a, b} /. x_Symbol :> f[x]").unwrap(),
      "f[List][f[a], f[b]]",
    );
    assert_eq!(
      interpret("(a -> b) /. x_Symbol :> f[x]").unwrap(),
      "f[Rule][f[a], f[b]]",
    );
    // Non-symbol heads (integers) are left alone; the head still rewrites.
    assert_eq!(
      interpret("h[1, 2] /. x_Symbol :> f[x]").unwrap(),
      "f[h][1, 2]"
    );
    // Curried calls rewrite every layer's head. Previously `h[a][b]` matched
    // `x_Symbol` at the top level because get_expr_head returned "Symbol".
    assert_eq!(
      interpret("h[a][b] /. x_Symbol :> f[x]").unwrap(),
      "f[h][f[a]][f[b]]",
    );
    assert_eq!(interpret("h[a][b] /. x_h :> 99").unwrap(), "99[b]");
    // The multi-rule path (list of rules) behaves identically.
    assert_eq!(
      interpret("h[a, b] /. {x_Symbol :> f[x]}").unwrap(),
      "f[h][f[a], f[b]]",
    );
    // A literal head rule still rewrites only the matching head.
    assert_eq!(interpret("x[a] /. x -> 3").unwrap(), "3[a]");
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
  fn test_insphere_simplex() {
    clear_state();
    // A 2-simplex is a triangle; Insphere[Simplex[...]] must match the
    // Triangle[...] wrapper form. 3-4-5 right triangle → incircle of radius 1
    // centred at {1, 1}; 6-8-10 right triangle → radius 2 centred at {2, 2}.
    assert_eq!(
      interpret("Insphere[Simplex[{{0, 0}, {4, 0}, {0, 3}}]]").unwrap(),
      "Sphere[{1, 1}, 1]"
    );
    assert_eq!(
      interpret("Insphere[Simplex[{{0, 0}, {6, 0}, {0, 8}}]]").unwrap(),
      "Sphere[{2, 2}, 2]"
    );
    // A 3-simplex is a tetrahedron; Simplex and Tetrahedron wrappers must
    // agree on the inscribed sphere for the same vertices.
    let verts = "{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}";
    assert_eq!(
      interpret(&format!("Insphere[Simplex[{verts}]]")).unwrap(),
      interpret(&format!("Insphere[Tetrahedron[{verts}]]")).unwrap()
    );
  }

  #[test]
  fn test_mixed_radix_quantity_stays_symbolic() {
    clear_state();
    // MixedRadixQuantity[digits, radixList] is an inert container: wolframscript
    // leaves it symbolic (arguments evaluate, the head stays) and emits no
    // message. It must NOT produce a "not yet implemented" warning.
    let r =
      interpret_with_stdout("MixedRadixQuantity[{1, 2, 3}, {60, 60}]").unwrap();
    assert_eq!(r.result, "MixedRadixQuantity[{1, 2, 3}, {60, 60}]");
    assert!(
      !r.warnings.iter().any(|w| w.contains("not yet implemented")),
      "unexpected warning: {:?}",
      r.warnings
    );
    // N threads into the arguments while the head is preserved.
    assert_eq!(
      interpret("N[MixedRadixQuantity[{1, 2, 3}, {60, 60}]]").unwrap(),
      "MixedRadixQuantity[{1., 2., 3.}, {60., 60.}]"
    );
    assert_eq!(
      interpret("Head[MixedRadixQuantity[{1, 2, 3}, {60, 60}]]").unwrap(),
      "MixedRadixQuantity"
    );
  }

  #[test]
  fn n_of_exact_zero_stays_exact() {
    clear_state();
    // N[0, p] on an exact zero stays the exact integer 0 (Head Integer,
    // Precision Infinity) — wolframscript never fabricates a
    // precision-tagged BigFloat zero. Non-zero exacts still pick up the tag.
    assert_eq!(interpret("N[0, 20]").unwrap(), "0");
    assert_eq!(interpret("N[0, 30]").unwrap(), "0");
    assert_eq!(interpret("Head[N[0, 20]]").unwrap(), "Integer");
    assert_eq!(interpret("Precision[N[0, 20]]").unwrap(), "Infinity");
    // Exact zeros arising from evaluation collapse too.
    assert_eq!(interpret("N[Sin[0], 20]").unwrap(), "0");
    assert_eq!(interpret("N[2 - 2, 25]").unwrap(), "0");
    assert_eq!(interpret("N[Cos[Pi/2], 40]").unwrap(), "0");
    // A machine Real 0. is left unchanged, and non-zero exacts keep the tag.
    assert_eq!(interpret("N[0., 20]").unwrap(), "0.");
    assert_eq!(interpret("N[2, 20]").unwrap(), "2.`20.");
    // Lists collapse the zero elements element-wise while others keep the tag.
    assert_eq!(
      interpret("N[{1, 0, 2}, 20]").unwrap(),
      "{1.`20., 0, 2.`20.}"
    );
    // RealDigits sees the exact 0, not a padded BigFloat zero.
    assert_eq!(interpret("RealDigits[N[0, 20]]").unwrap(), "{{0}, 1}");
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
      // Highlighted is intentionally omitted here: like Framed it renders to
      // an SVG box (`-Graphics-`) in visual mode rather than staying symbolic.
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
      // Interactive manipulation heads from the InteractiveManipulation
      // guide. In script mode these stay unevaluated as their canonical form
      // (they only become interactive objects inside a notebook), so they
      // must not emit a spurious "not yet implemented" warning.
      ("Animate[x, {x, 0, 1}]", "Animate[x, {x, 0, 1}]"),
      ("Animator[0]", "Animator[0]"),
      ("ListAnimate[{1, 2, 3}]", "ListAnimate[{1, 2, 3}]"),
      (
        "ControllerManipulate[x, {x, 0, 1}]",
        "ControllerManipulate[x, {x, 0, 1}]",
      ),
      ("Trigger[Dynamic[x]]", "Trigger[Dynamic[x]]"),
      ("SetterBar[1, {1, 2, 3}]", "SetterBar[1, {1, 2, 3}]"),
      ("CheckboxBar[{1}, {1, 2}]", "CheckboxBar[{1}, {1, 2}]"),
      ("TogglerBar[{1}, {1, 2}]", "TogglerBar[{1}, {1, 2}]"),
      ("RadioButton[1]", "RadioButton[1]"),
      ("ProgressIndicator[0.5]", "ProgressIndicator[0.5]"),
      ("PaneSelector[{1 -> a}, 1]", "PaneSelector[{1 -> a}, 1]"),
      ("PopupView[{a, b}]", "PopupView[{a, b}]"),
      ("IntervalSlider[{2, 4}]", "IntervalSlider[{2, 4}]"),
      ("Slider2D[{0, 0}]", "Slider2D[{0, 0}]"),
      // Manipulate option symbols used on their own stay symbolic too.
      ("Bookmarks", "Bookmarks"),
      ("ContinuousAction", "ContinuousAction"),
      ("AppearanceElements", "AppearanceElements"),
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
  fn control_active_returns_inactive_form_in_script_mode() {
    // ControlActive[activeform, normalform] displays as `activeform` only
    // while it sits inside a control that is being actively manipulated. In
    // script mode nothing is ever actively manipulated, so it evaluates to
    // its inactive form `normalform` (with the argument fully evaluated).
    let cases = [
      ("ControlActive[1, 2]", "2"),
      ("ControlActive[1 + 1, 2 + 2]", "4"),
      ("ControlActive[\"fast\", \"slow\"]", "slow"),
      // ControlActive[] with no arguments queries whether a control is being
      // actively manipulated; outside a notebook nothing is, so it is False.
      ("ControlActive[]", "False"),
      // Other non-two-argument forms have no active/normal split, so they stay
      // symbolic (and must not warn about being unimplemented).
      ("ControlActive[5]", "ControlActive[5]"),
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
  fn polar_curves_stay_symbolic_in_script_mode() {
    // PolarCurve / FilledPolarCurve are lightweight graphics primitives that
    // the playground and Woxi Studio render as graphics. In the plain CLI
    // (script mode) they stay unevaluated as their canonical form — rather
    // than being lowered to a ParametricRegion the way wolframscript does —
    // so the visual hosts can draw them.
    let cases = [
      (
        "PolarCurve[1 + Cos[t], {t, 0, 2 Pi}]",
        "PolarCurve[1 + Cos[t], {t, 0, 2*Pi}]",
      ),
      (
        "FilledPolarCurve[PolarCurve[Sin[2 t], {t, 0, 2 Pi}]]",
        "FilledPolarCurve[PolarCurve[Sin[2*t], {t, 0, 2*Pi}]]",
      ),
      (
        "FilledPolarCurve[1 - Cos[t], t]",
        "FilledPolarCurve[1 - Cos[t], t]",
      ),
      // Wrapped in Graphics the head is Graphics (they render as a curve /
      // filled region in visual hosts).
      (
        "Head[Graphics[PolarCurve[1 + Cos[t], {t, 0, 2 Pi}]]]",
        "Graphics",
      ),
      (
        "Head[Graphics[FilledPolarCurve[PolarCurve[Sin[2 t], {t, 0, 2 Pi}]]]]",
        "Graphics",
      ),
      (
        "Head[Graphics[FilledPolarCurve[1 - Cos[t], t]]]",
        "Graphics",
      ),
    ];
    for (input, expected) in cases {
      assert_eq!(
        interpret(input).unwrap(),
        expected,
        "result mismatch for {input}"
      );
    }
  }

  #[test]
  fn held_graphics_argument_summarizes_as_graphics_placeholder() {
    // A Graphics[...] argument held inside a symbolic wrapper (LocatorPane,
    // ClickPane) still summarizes to the -Graphics- placeholder in OutputForm,
    // matching wolframscript — the full Graphics expression is only shown by
    // InputForm / FullForm.
    let cases = [
      (
        "LocatorPane[Dynamic[p], Graphics[Point[p]]]",
        "LocatorPane[Dynamic[p], -Graphics-]",
      ),
      ("ClickPane[Graphics[{}], f]", "ClickPane[-Graphics-, f]"),
    ];
    for (input, expected) in cases {
      let r = interpret_with_stdout(input).unwrap();
      assert_eq!(r.result, expected, "result mismatch for {input}");
    }
  }

  #[test]
  fn drop_shadowing_canonicalizes_with_defaults() {
    // DropShadowing arguments are matched positionally in the order
    // offset (2-element numeric list), radius (number), color (color
    // directive or None), each slot optional, and the missing slots are
    // filled with the defaults {-3, -3}, 2 and
    // Opacity[1/3, ThemeColor[Foreground]] — matching wolframscript.
    let cases = [
      (
        "DropShadowing[]",
        "DropShadowing[{-3, -3}, 2, Opacity[1/3, ThemeColor[Foreground]]]",
      ),
      (
        "DropShadowing[{1, 2}]",
        "DropShadowing[{1, 2}, 2, Opacity[1/3, ThemeColor[Foreground]]]",
      ),
      (
        "DropShadowing[5]",
        "DropShadowing[{-3, -3}, 5, Opacity[1/3, ThemeColor[Foreground]]]",
      ),
      (
        "DropShadowing[2.5]",
        "DropShadowing[{-3, -3}, 2.5, Opacity[1/3, ThemeColor[Foreground]]]",
      ),
      (
        "DropShadowing[Red]",
        "DropShadowing[{-3, -3}, 2, RGBColor[1, 0, 0]]",
      ),
      (
        "DropShadowing[Opacity[0.5]]",
        "DropShadowing[{-3, -3}, 2, Opacity[0.5]]",
      ),
      ("DropShadowing[None]", "DropShadowing[{-3, -3}, 2, None]"),
      (
        "DropShadowing[{1, 2}, 5]",
        "DropShadowing[{1, 2}, 5, Opacity[1/3, ThemeColor[Foreground]]]",
      ),
      (
        "DropShadowing[{1, 2}, Red]",
        "DropShadowing[{1, 2}, 2, RGBColor[1, 0, 0]]",
      ),
      (
        "DropShadowing[5, Red]",
        "DropShadowing[{-3, -3}, 5, RGBColor[1, 0, 0]]",
      ),
      (
        "DropShadowing[{1, 2}, 5, Red]",
        "DropShadowing[{1, 2}, 5, RGBColor[1, 0, 0]]",
      ),
      // The canonical form is a fixed point of evaluation.
      (
        "DropShadowing[{-3, -3}, 2, Opacity[1/3, ThemeColor[Foreground]]]",
        "DropShadowing[{-3, -3}, 2, Opacity[1/3, ThemeColor[Foreground]]]",
      ),
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
  fn drop_shadowing_invalid_specs_stay_unevaluated() {
    // Argument lists that don't fit the offset/radius/color pattern
    // (wrong types, wrong slot order, too many arguments) are left
    // unevaluated with evaluated arguments, and must NOT emit a spurious
    // "not yet implemented" warning (like Glow/EdgeForm/Opacity).
    let cases = [
      ("DropShadowing[True]", "DropShadowing[True]"),
      ("DropShadowing[False]", "DropShadowing[False]"),
      ("DropShadowing[x]", "DropShadowing[x]"),
      ("DropShadowing[{a, b}]", "DropShadowing[{a, b}]"),
      ("DropShadowing[{1, 2, 3}]", "DropShadowing[{1, 2, 3}]"),
      // Color before offset is out of order.
      (
        "DropShadowing[Red, {1, 2}]",
        "DropShadowing[RGBColor[1, 0, 0], {1, 2}]",
      ),
      (
        "DropShadowing[{1, 2}, 5, Red, 7]",
        "DropShadowing[{1, 2}, 5, RGBColor[1, 0, 0], 7]",
      ),
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
  mod assessment;
  mod association;
  mod astronomy;
  mod attributes;
  mod audio;
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
  mod molecule;
  mod music;
  mod patterns;
  mod polyhedron_data;
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
  mod timeseries;
  mod turing_machine;
  mod wavelets;
  mod wxf;
}

#[cfg(test)]
mod tmp_dbg4 {
  #[test]
  fn t() {
    woxi::clear_state();
    let _ = woxi::interpret_with_stdout("{1, 2, 3}");
  }
}
