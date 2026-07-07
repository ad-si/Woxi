use woxi::interpret;

mod list {
  use super::*;

  #[test]
  fn parse() {
    assert_eq!(interpret("{1, 2, 3}").unwrap(), "{1, 2, 3}");
    assert_eq!(interpret("{a, b, c}").unwrap(), "{a, b, c}");
    assert_eq!(
      interpret("{True, False, True}").unwrap(),
      "{True, False, True}"
    );
  }

  #[test]
  fn first() {
    assert_eq!(interpret("First[{1, 2, 3}]").unwrap(), "1");
    assert_eq!(interpret("First[{a, b, c}]").unwrap(), "a");
    assert_eq!(interpret("First[{True, False, False}]").unwrap(), "True");
  }

  #[test]
  fn last() {
    assert_eq!(interpret("Last[{1, 2, 3}]").unwrap(), "3");
    assert_eq!(interpret("Last[{a, b, c}]").unwrap(), "c");
    assert_eq!(interpret("Last[{True, True, False}]").unwrap(), "False");
  }

  #[test]
  fn gather() {
    assert_eq!(
      interpret("Gather[{1, 1, 2, 2, 1}]").unwrap(),
      "{{1, 1, 1}, {2, 2}}"
    );
    assert_eq!(
      interpret("Gather[{a, b, a, c, b}]").unwrap(),
      "{{a, a}, {b, b}, {c}}"
    );
    assert_eq!(interpret("Gather[{}]").unwrap(), "{}");
  }

  #[test]
  fn gather_by() {
    assert_eq!(
      interpret("GatherBy[{1, 2, 3, 4, 5}, EvenQ]").unwrap(),
      "{{1, 3, 5}, {2, 4}}"
    );
    assert_eq!(
      interpret("GatherBy[{-2, -1, 0, 1, 2}, Sign]").unwrap(),
      "{{-2, -1}, {0}, {1, 2}}"
    );
  }

  #[test]
  fn gather_by_nested_funcs() {
    // GatherBy with a list of functions does a nested gather: first by
    // f1, then each resulting sublist is gathered by f2, and so on.
    assert_eq!(
      interpret("GatherBy[{1, 2, 3, 4, 5, 6}, {OddQ, PrimeQ}]").unwrap(),
      "{{{1}, {3, 5}}, {{2}, {4, 6}}}"
    );
    assert_eq!(
      interpret("GatherBy[Range[10], {EvenQ, PrimeQ}]").unwrap(),
      "{{{1, 9}, {3, 5, 7}}, {{2}, {4, 6, 8, 10}}}"
    );
    // A list with a single function behaves like GatherBy[list, f1] but
    // wraps the result in an extra level.
    assert_eq!(
      interpret("GatherBy[{{1,3},{2,2},{1,5},{2,1},{3,6}}, {First}]").unwrap(),
      "{{{1, 3}, {1, 5}}, {{2, 2}, {2, 1}}, {{3, 6}}}"
    );
  }

  #[test]
  fn split() {
    assert_eq!(
      interpret("Split[{1, 1, 2, 2, 3}]").unwrap(),
      "{{1, 1}, {2, 2}, {3}}"
    );
    assert_eq!(
      interpret("Split[{a, a, b, c, c, c}]").unwrap(),
      "{{a, a}, {b}, {c, c, c}}"
    );
    assert_eq!(interpret("Split[{}]").unwrap(), "{}");
  }

  #[test]
  fn split_by() {
    assert_eq!(
      interpret("SplitBy[{1, 2, 3, 4, 5}, EvenQ]").unwrap(),
      "{{1}, {2}, {3}, {4}, {5}}"
    );
  }

  #[test]
  fn split_by_list_of_funcs_single() {
    // Singleton list of funcs behaves like SplitBy[list, f] wrapped one extra level.
    assert_eq!(
      interpret("SplitBy[{1, 1, 2, 2, 3, 3, 3, 4}, {OddQ}]").unwrap(),
      "{{1, 1}, {2, 2}, {3, 3, 3}, {4}}"
    );
  }

  #[test]
  fn split_by_list_of_funcs_nested() {
    assert_eq!(
      interpret("SplitBy[{1, 1, 2, 2, 3, 3, 3, 4}, {OddQ, EvenQ}]").unwrap(),
      "{{{1, 1}}, {{2, 2}}, {{3, 3, 3}}, {{4}}}"
    );
  }

  #[test]
  fn split_by_list_of_funcs_with_structured_items() {
    assert_eq!(
      interpret("SplitBy[{{1, a}, {1, b}, {2, a}, {2, b}}, {First, Last}]")
        .unwrap(),
      "{{{{1, a}}, {{1, b}}}, {{{2, a}}, {{2, b}}}}"
    );
  }

  #[test]
  fn extract() {
    assert_eq!(interpret("Extract[{a, b, c, d}, 2]").unwrap(), "b");
    assert_eq!(
      interpret("Extract[{a, {b1, b2, b3}, c, d}, {2, 3}]").unwrap(),
      "b3"
    );
  }

  #[test]
  fn extract_all_span_part_specs() {
    let m = "{{a, b, c}, {d, e, f}, {g, h, i}}";
    // `All` selects every element at that level: column 2.
    assert_eq!(
      interpret(&format!("Extract[{m}, {{All, 2}}]")).unwrap(),
      "{b, e, h}"
    );
    // `All` in the last position returns whole rows.
    assert_eq!(
      interpret(&format!("Extract[{m}, {{2, All}}]")).unwrap(),
      "{d, e, f}"
    );
    // `All` at both levels reproduces the matrix.
    assert_eq!(
      interpret(&format!("Extract[{m}, {{All, All}}]")).unwrap(),
      m
    );
    // A list of indices selects several columns per row.
    assert_eq!(
      interpret(&format!("Extract[{m}, {{All, {{1, 3}}}}]")).unwrap(),
      "{{a, c}, {d, f}, {g, i}}"
    );
    // A single-path All spec wrapped in an outer list is a one-element
    // multi-path spec.
    assert_eq!(
      interpret(&format!("Extract[{m}, {{{{All, 2}}}}]")).unwrap(),
      "{{b, e, h}}"
    );
    // The optional head wraps the extracted result once.
    assert_eq!(
      interpret(&format!("Extract[{m}, {{All, 2}}, Hold]")).unwrap(),
      "Hold[{b, e, h}]"
    );
    // A Span selector inside a path.
    assert_eq!(
      interpret(&format!("Extract[{m}, {{1 ;; 2, 2}}]")).unwrap(),
      "{b, e}"
    );
  }

  #[test]
  fn catenate() {
    assert_eq!(
      interpret("Catenate[{{1, 2}, {3, 4}}]").unwrap(),
      "{1, 2, 3, 4}"
    );
    assert_eq!(
      interpret("Catenate[{{a, b}, {c}, {d, e, f}}]").unwrap(),
      "{a, b, c, d, e, f}"
    );
  }

  #[test]
  fn catenate_associations() {
    assert_eq!(
      interpret("Catenate[{<|a -> 1|>, <|b -> 2|>}]").unwrap(),
      "{1, 2}"
    );
  }

  #[test]
  fn catenate_association_of_lists() {
    // An association argument is catenated over its values.
    assert_eq!(
      interpret("Catenate[<|a -> {1}, b -> {2, 3}|>]").unwrap(),
      "{1, 2, 3}"
    );
    assert_eq!(
      interpret("Catenate[<|a -> {1, 2}, b -> {3}, c -> {4, 5}|>]").unwrap(),
      "{1, 2, 3, 4, 5}"
    );
  }

  #[test]
  fn catenate_empty_association() {
    assert_eq!(interpret("Catenate[<||>]").unwrap(), "{}");
  }

  #[test]
  fn apply() {
    assert_eq!(interpret("Apply[Plus, {1, 2, 3}]").unwrap(), "6");
    assert_eq!(interpret("Apply[Times, {2, 3, 4}]").unwrap(), "24");
  }

  #[test]
  fn apply_replaces_head_with_any_expr() {
    // Apply replaces the head with `func` regardless of what `func` is —
    // not only symbols/functions. Regression: Woxi used to error with
    // "Apply: first argument must be a function".
    assert_eq!(interpret("Apply[{g, h}, {1, 2}]").unwrap(), "{g, h}[1, 2]");
    assert_eq!(interpret("Apply[3, {1, 2}]").unwrap(), "3[1, 2]");
    assert_eq!(interpret("Apply[f[a], {1, 2}]").unwrap(), "f[a][1, 2]");
    // Applicable compound heads reduce.
    assert_eq!(
      interpret("Apply[Composition[f, g], {1, 2}]").unwrap(),
      "f[g[1, 2]]"
    );
  }

  #[test]
  fn apply_map_scan_treat_rational_complex_as_atoms() {
    // Rational / Complex are atoms: Apply leaves them unchanged rather than
    // splatting their internal {num, den} / {re, im} as arguments.
    assert_eq!(interpret("Apply[f, 1/2]").unwrap(), "1/2");
    assert_eq!(interpret("Apply[f, 1 + 2 I]").unwrap(), "1 + 2*I");
    // A list head is still replaced.
    assert_eq!(interpret("Apply[Plus, {1/2, 3/4}]").unwrap(), "5/4");

    // Map at level 1 over an atom returns it unchanged; level {-1} maps the
    // atom itself, and a Rational inside a sum is mapped as a whole.
    assert_eq!(interpret("Map[f, 1/2]").unwrap(), "1/2");
    assert_eq!(interpret("Map[f, 1/2, {-1}]").unwrap(), "f[1/2]");
    assert_eq!(
      interpret("Map[f, 3 + x/2, {-1}]").unwrap(),
      "f[3] + f[1/2]*f[x]"
    );

    // Scan has no parts to visit over an atom, so nothing is sown.
    assert_eq!(interpret("Reap[Scan[Sow, 1/2]]").unwrap(), "{Null, {}}");
    // The 1/2 inside x/2 is part of the atomic Rational, not sown separately.
    assert_eq!(
      interpret("Reap[Scan[Sow, 3 + x/2]]").unwrap(),
      "{Null, {{3, x/2}}}"
    );
  }

  #[test]
  fn replace_treats_rational_complex_as_atoms() {
    // ReplaceAll must not descend into a Rational's {num, den} or a complex
    // number's tree: 1/2 /. x_Integer -> z stays 1/2 (not z/z).
    assert_eq!(interpret("1/2 /. x_Integer -> z").unwrap(), "1/2");
    assert_eq!(interpret("{1/2, 3} /. x_Integer -> z").unwrap(), "{1/2, z}");
    assert_eq!(interpret("3 + x/2 /. n_Integer -> z").unwrap(), "x/2 + z");
    assert_eq!(interpret("1 + 2 I /. x_Integer -> z").unwrap(), "1 + 2*I");
    // The whole atom can still be matched by a structured pattern.
    assert_eq!(interpret("1/2 /. Rational[a_, b_] -> a + b").unwrap(), "3");
    assert_eq!(
      interpret("1 + 2 I /. Complex[a_, b_] -> a + b").unwrap(),
      "3"
    );
    // Leveled Replace likewise treats them as atoms.
    assert_eq!(
      interpret("Replace[1/2, x_Integer -> z, Infinity]").unwrap(),
      "1/2"
    );
    assert_eq!(
      interpret("Replace[{1/2}, x_Integer -> z, {-1}]").unwrap(),
      "{1/2}"
    );
    // Ordinary integers are still replaced at the leaf level.
    assert_eq!(
      interpret("Replace[{1, 2}, n_Integer -> z, {-1}]").unwrap(),
      "{z, z}"
    );
  }

  #[test]
  fn find_repeat() {
    // Finds the shortest tiling sub-sequence; a partial final rep is allowed.
    assert_eq!(
      interpret("FindRepeat[{1, 2, 3, 1, 2, 3}]").unwrap(),
      "{1, 2, 3}"
    );
    assert_eq!(
      interpret("FindRepeat[{1, 2, 3, 1, 2, 3, 1}]").unwrap(),
      "{1, 2, 3}"
    );
    assert_eq!(interpret("FindRepeat[{5, 5, 5, 5}]").unwrap(), "{5}");
    // No repetition: the whole list is returned.
    assert_eq!(
      interpret("FindRepeat[{1, 2, 3, 4, 5}]").unwrap(),
      "{1, 2, 3, 4, 5}"
    );
    // A lead-in element breaks the tiling from the start.
    assert_eq!(
      interpret("FindRepeat[{7, 1, 2, 1, 2, 1, 2}]").unwrap(),
      "{7, 1, 2, 1, 2, 1, 2}"
    );
    assert_eq!(
      interpret("FindRepeat[{1, 2, 1, 2, 3}]").unwrap(),
      "{1, 2, 1, 2, 3}"
    );
    // Reals compare by value.
    assert_eq!(
      interpret("FindRepeat[{1.5, 2.5, 1.5, 2.5}]").unwrap(),
      "{1.5, 2.5}"
    );
    // Second argument: the pattern must repeat at least n whole times.
    assert_eq!(
      interpret("FindRepeat[{1, 2, 1, 2, 1, 2}, 2]").unwrap(),
      "{1, 2}"
    );
    // When no period repeats n times, the empty list is returned.
    assert_eq!(
      interpret("FindRepeat[{1, 2, 3, 1, 2, 3}, 3]").unwrap(),
      "{}"
    );
    assert_eq!(interpret("FindRepeat[{1, 2, 3, 4, 5}, 2]").unwrap(), "{}");
    // Strings.
    assert_eq!(interpret("FindRepeat[\"abcabc\"]").unwrap(), "abc");
    assert_eq!(interpret("FindRepeat[\"abcd\"]").unwrap(), "abcd");
    // Associations reduce over their values, keeping the keys.
    assert_eq!(
      interpret("FindRepeat[<|1 -> a, 2 -> b, 3 -> a, 4 -> b|>]").unwrap(),
      "<|1 -> a, 2 -> b|>"
    );
    // Empty list.
    assert_eq!(interpret("FindRepeat[{}]").unwrap(), "{}");
  }

  #[test]
  fn find_transient_repeat() {
    // {transient, repeat}: the repeat is the shortest end-cycle occurring at
    // least n times; the transient is the (shortest) leading remainder.
    assert_eq!(
      interpret("FindTransientRepeat[{1, 2, 3, 2, 3}, 2]").unwrap(),
      "{{1}, {2, 3}}"
    );
    // Extra repetitions still report a single period.
    assert_eq!(
      interpret("FindTransientRepeat[{1, 2, 3, 2, 3, 2, 3}, 2]").unwrap(),
      "{{1}, {2, 3}}"
    );
    assert_eq!(
      interpret("FindTransientRepeat[{0, 1, 2, 3, 1, 2, 3}, 2]").unwrap(),
      "{{0}, {1, 2, 3}}"
    );
    // A longer transient before the cycle.
    assert_eq!(
      interpret("FindTransientRepeat[{9, 8, 1, 2, 1, 2, 1, 2}, 3]").unwrap(),
      "{{9, 8}, {1, 2}}"
    );
    // The smallest period wins when several tile the end.
    assert_eq!(
      interpret("FindTransientRepeat[{1, 1, 1, 1}, 2]").unwrap(),
      "{{}, {1}}"
    );
    // No transient.
    assert_eq!(
      interpret("FindTransientRepeat[{1, 2, 1, 2}, 2]").unwrap(),
      "{{}, {1, 2}}"
    );
    // No cycle repeats n times: the whole list is the transient.
    assert_eq!(
      interpret("FindTransientRepeat[{1, 2, 3, 4, 5}, 2]").unwrap(),
      "{{1, 2, 3, 4, 5}, {}}"
    );
    assert_eq!(
      interpret("FindTransientRepeat[{1, 2, 3, 2, 3}, 3]").unwrap(),
      "{{1, 2, 3, 2, 3}, {}}"
    );
    // n = 1: the whole list is one repetition (minimal transient wins).
    assert_eq!(
      interpret("FindTransientRepeat[{1, 2, 3, 2, 3}, 1]").unwrap(),
      "{{}, {1, 2, 3, 2, 3}}"
    );
    // Empty list.
    assert_eq!(interpret("FindTransientRepeat[{}, 2]").unwrap(), "{{}, {}}");
    // Strings and associations keep the input head in both parts.
    assert_eq!(
      interpret("FindTransientRepeat[\"abcbc\", 2]").unwrap(),
      "{a, bc}"
    );
    assert_eq!(
      interpret("FindTransientRepeat[<|1 -> a, 2 -> b, 3 -> a, 4 -> b|>, 2]")
        .unwrap(),
      "{<||>, <|1 -> a, 2 -> b|>}"
    );
  }

  #[test]
  fn find_transient_repeat_messages() {
    // A non-positive-integer count emits intp and stays unevaluated.
    assert_eq!(
      interpret("FindTransientRepeat[{1, 2, 3, 2, 3}, 0]").unwrap(),
      "FindTransientRepeat[{1, 2, 3, 2, 3}, 0]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "FindTransientRepeat::intp: Positive integer expected at position 2 in FindTransientRepeat[{1, 2, 3, 2, 3}, 0]."
      )),
      "expected intp message, got {:?}",
      msgs
    );
  }

  #[test]
  fn array_reduce() {
    // Reduce a matrix over each dimension; f gets the gathered elements.
    assert_eq!(
      interpret("ArrayReduce[Total, {{1, 2}, {3, 4}}, 1]").unwrap(),
      "{4, 6}"
    );
    assert_eq!(
      interpret("ArrayReduce[Total, {{1, 2}, {3, 4}}, 2]").unwrap(),
      "{3, 7}"
    );
    // Reducing every dimension yields a scalar.
    assert_eq!(
      interpret("ArrayReduce[Total, {{1, 2}, {3, 4}}, {1, 2}]").unwrap(),
      "10"
    );
    // Works with any reducing function.
    assert_eq!(
      interpret("ArrayReduce[Max, {{1, 2}, {3, 4}}, 1]").unwrap(),
      "{3, 4}"
    );
    assert_eq!(
      interpret("ArrayReduce[Mean, {{1, 2, 3}, {4, 5, 6}}, 2]").unwrap(),
      "{2, 5}"
    );
    // The gathered list is in row-major order over the reduced dims.
    assert_eq!(
      interpret("ArrayReduce[Identity, {{1, 2}, {3, 4}}, 1]").unwrap(),
      "{{1, 3}, {2, 4}}"
    );
    assert_eq!(
      interpret("ArrayReduce[#[[1]] - #[[2]] &, {{1, 2}, {3, 4}}, 1]").unwrap(),
      "{-2, -2}"
    );
    // A 1-D vector.
    assert_eq!(
      interpret("ArrayReduce[Total, {1, 2, 3, 4}, 1]").unwrap(),
      "10"
    );
    // A 3-D array, reducing single and multiple dimensions.
    assert_eq!(
      interpret("ArrayReduce[Total, {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, 1]")
        .unwrap(),
      "{{6, 8}, {10, 12}}"
    );
    assert_eq!(
      interpret(
        "ArrayReduce[Identity, {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, {1, 3}]"
      )
      .unwrap(),
      "{{1, 2, 5, 6}, {3, 4, 7, 8}}"
    );
    assert_eq!(
      interpret(
        "ArrayReduce[Total, {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, {2, 3}]"
      )
      .unwrap(),
      "{10, 26}"
    );
    // Out-of-range level specs stay unevaluated.
    assert_eq!(
      interpret("ArrayReduce[Total, {{1, 2}, {3, 4}}, 3]").unwrap(),
      "ArrayReduce[Total, {{1, 2}, {3, 4}}, 3]"
    );
    assert_eq!(
      interpret("ArrayReduce[Total, {{1, 2}, {3, 4}}, 0]").unwrap(),
      "ArrayReduce[Total, {{1, 2}, {3, 4}}, 0]"
    );
  }

  #[test]
  fn comap_apply() {
    // ComapApply applies each function to the spread sequence of arguments.
    assert_eq!(
      interpret("ComapApply[{f, g}, {1, 2}]").unwrap(),
      "{f[1, 2], g[1, 2]}"
    );
    assert_eq!(
      interpret("ComapApply[{f, g, h}, {a, b}]").unwrap(),
      "{f[a, b], g[a, b], h[a, b]}"
    );
    // Real functions are evaluated.
    assert_eq!(
      interpret("ComapApply[{Plus, Times}, {3, 4}]").unwrap(),
      "{7, 12}"
    );
    // Associations map over their values, keeping the keys.
    assert_eq!(
      interpret("ComapApply[<|a -> f, b -> g|>, {1, 2}]").unwrap(),
      "<|a -> f[1, 2], b -> g[1, 2]|>"
    );
    // General heads keep their structure.
    assert_eq!(
      interpret("ComapApply[h[f, g], {1, 2}]").unwrap(),
      "h[f[1, 2], g[1, 2]]"
    );
    // Atomic first argument is returned unchanged.
    assert_eq!(interpret("ComapApply[f, {1, 2}]").unwrap(), "f");
    assert_eq!(interpret("ComapApply[3, {1, 2}]").unwrap(), "3");
    // Default level {1}: a nested list is applied as a whole head.
    assert_eq!(
      interpret("ComapApply[{f, {g, h}}, {1, 2}]").unwrap(),
      "{f[1, 2], {g, h}[1, 2]}"
    );
    // Non-list second argument: Apply on an atom leaves it unchanged.
    assert_eq!(interpret("ComapApply[{f, g}, x]").unwrap(), "{x, x}");
    // Operator form.
    assert_eq!(
      interpret("ComapApply[{f, g}][{1, 2}]").unwrap(),
      "{f[1, 2], g[1, 2]}"
    );
    assert_eq!(
      interpret("ComapApply[{f, g}]").unwrap(),
      "ComapApply[{f, g}]"
    );
  }

  #[test]
  fn identity() {
    assert_eq!(interpret("Identity[5]").unwrap(), "5");
    assert_eq!(interpret("Identity[{1, 2, 3}]").unwrap(), "{1, 2, 3}");
  }

  #[test]
  fn outer() {
    assert_eq!(
      interpret("Outer[Times, {1, 2}, {3, 4}]").unwrap(),
      "{{3, 4}, {6, 8}}"
    );
    assert_eq!(
      interpret("Outer[Plus, {1, 2}, {10, 20}]").unwrap(),
      "{{11, 21}, {12, 22}}"
    );
  }

  #[test]
  fn outer_rational_and_complex_are_atoms() {
    // Regression: Rational/Complex must be treated as atoms, not descended
    // into as if {num, den} / {re, im} were a sub-level.
    assert_eq!(interpret("Outer[Times, {1/2}, {1/3}]").unwrap(), "{{1/6}}");
    assert_eq!(
      interpret("Outer[Times, {2/3, -2/3, 1/3}, {1/3, 2/3, 2/3}]").unwrap(),
      "{{2/9, 4/9, 4/9}, {-2/9, -4/9, -4/9}, {1/9, 2/9, 2/9}}"
    );
    assert_eq!(
      interpret("Outer[Plus, {1/2, 1/4}, {1/3}]").unwrap(),
      "{{5/6}, {7/12}}"
    );
    assert_eq!(
      interpret("Outer[Times, {1 + I, 2}, {1}]").unwrap(),
      "{{1 + I}, {2}}"
    );
  }

  #[test]
  fn level_treats_rational_and_complex_as_atoms() {
    // Regression: Level must not descend into a Rational's {num, den} or a
    // Complex's structure.
    assert_eq!(interpret("Level[1/2, {-1}]").unwrap(), "{1/2}");
    assert_eq!(interpret("Level[1/2, {1}]").unwrap(), "{}");
    assert_eq!(
      interpret("Level[3 + x/2, Infinity]").unwrap(),
      "{3, 1/2, x, x/2}"
    );
    assert_eq!(interpret("Level[1 + 2 I, {-1}]").unwrap(), "{1 + 2*I}");
    assert_eq!(interpret("Level[{1/2, 1/3}, {-1}]").unwrap(), "{1/2, 1/3}");
    assert_eq!(interpret("Level[f[1/2, a], {-1}]").unwrap(), "{1/2, a}");
  }

  #[test]
  fn depth_treats_rational_as_atom() {
    assert_eq!(interpret("Depth[1/2]").unwrap(), "1");
    assert_eq!(interpret("Depth[1 + 2 I]").unwrap(), "1");
    assert_eq!(interpret("Depth[3 + x/2]").unwrap(), "3");
  }

  #[test]
  fn conjugate_does_not_distribute_over_bare_symbol_sums() {
    // A fully-symbolic sum stays grouped under one Conjugate (Wolfram does not
    // auto-distribute Conjugate over Plus without a simplifiable term).
    assert_eq!(interpret("Conjugate[a + b]").unwrap(), "Conjugate[a + b]");
    assert_eq!(
      interpret("Conjugate[a + b + c]").unwrap(),
      "Conjugate[a + b + c]"
    );
    assert_eq!(
      interpret("Conjugate[a b + c]").unwrap(),
      "Conjugate[a*b + c]"
    );
    // Terms with a numeric coefficient or a pure-number term DO distribute,
    // and the remaining bare terms are grouped.
    assert_eq!(
      interpret("Conjugate[a - b]").unwrap(),
      "Conjugate[a] - Conjugate[b]"
    );
    assert_eq!(
      interpret("Conjugate[2 a + 3 b]").unwrap(),
      "2*Conjugate[a] + 3*Conjugate[b]"
    );
    assert_eq!(interpret("Conjugate[a + 5]").unwrap(), "5 + Conjugate[a]");
    assert_eq!(
      interpret("Conjugate[2 a + b + c]").unwrap(),
      "2*Conjugate[a] + Conjugate[b + c]"
    );
    assert_eq!(
      interpret("Conjugate[a b + 2 c]").unwrap(),
      "Conjugate[a*b] + 2*Conjugate[c]"
    );
    assert_eq!(
      interpret("Conjugate[a^2 + b]").unwrap(),
      "Conjugate[a]^2 + Conjugate[b]"
    );
    // Purely numeric arguments still conjugate fully.
    assert_eq!(interpret("Conjugate[2 + 3 I]").unwrap(), "2 - 3*I");
  }

  #[test]
  fn head_of_directed_infinity() {
    // Infinity, -Infinity and ComplexInfinity are DirectedInfinity[…] objects.
    assert_eq!(interpret("Head[Infinity]").unwrap(), "DirectedInfinity");
    assert_eq!(interpret("Head[-Infinity]").unwrap(), "DirectedInfinity");
    assert_eq!(
      interpret("Head[ComplexInfinity]").unwrap(),
      "DirectedInfinity"
    );
    assert_eq!(
      interpret("Head[DirectedInfinity[-1]]").unwrap(),
      "DirectedInfinity"
    );
    assert_eq!(
      interpret("Head[DirectedInfinity[I]]").unwrap(),
      "DirectedInfinity"
    );
    // Ordinary symbols, numbers and products are unaffected.
    assert_eq!(interpret("Head[Pi]").unwrap(), "Symbol");
    assert_eq!(interpret("Head[x]").unwrap(), "Symbol");
    assert_eq!(interpret("Head[a b]").unwrap(), "Times");
  }

  #[test]
  fn atom_q_of_directed_infinity() {
    // Infinity objects are DirectedInfinity[…], hence not atoms.
    assert_eq!(interpret("AtomQ[Infinity]").unwrap(), "False");
    assert_eq!(interpret("AtomQ[-Infinity]").unwrap(), "False");
    assert_eq!(interpret("AtomQ[ComplexInfinity]").unwrap(), "False");
    assert_eq!(interpret("AtomQ[DirectedInfinity[I]]").unwrap(), "False");
    // Ordinary atoms (symbols, numbers, Rational/Complex) stay atomic.
    assert_eq!(interpret("AtomQ[Pi]").unwrap(), "True");
    assert_eq!(interpret("AtomQ[x]").unwrap(), "True");
    assert_eq!(interpret("AtomQ[1/2]").unwrap(), "True");
    assert_eq!(interpret("AtomQ[2 + 3 I]").unwrap(), "True");
    assert_eq!(interpret("AtomQ[a + b]").unwrap(), "False");
  }

  #[test]
  fn element_of_infinity_is_false_in_every_domain() {
    // Infinity objects are not members of any number domain.
    assert_eq!(interpret("Element[Infinity, Reals]").unwrap(), "False");
    assert_eq!(interpret("Element[-Infinity, Reals]").unwrap(), "False");
    assert_eq!(interpret("Element[Infinity, Integers]").unwrap(), "False");
    assert_eq!(interpret("Element[Infinity, Complexes]").unwrap(), "False");
    assert_eq!(interpret("Element[Infinity, Algebraics]").unwrap(), "False");
    assert_eq!(
      interpret("Element[ComplexInfinity, Complexes]").unwrap(),
      "False"
    );
    assert_eq!(interpret("NotElement[Infinity, Reals]").unwrap(), "True");
    // Ordinary numbers and constants remain members.
    assert_eq!(interpret("Element[5, Reals]").unwrap(), "True");
    assert_eq!(interpret("Element[Pi, Reals]").unwrap(), "True");
    assert_eq!(interpret("Element[I, Complexes]").unwrap(), "True");
  }

  #[test]
  fn pattern_head_of_directed_infinity() {
    // Infinity objects match _DirectedInfinity and not _Symbol.
    assert_eq!(
      interpret("MatchQ[Infinity, _DirectedInfinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("MatchQ[-Infinity, _DirectedInfinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("MatchQ[ComplexInfinity, _DirectedInfinity]").unwrap(),
      "True"
    );
    assert_eq!(interpret("MatchQ[Infinity, _Symbol]").unwrap(), "False");
    // Cases filters accordingly.
    assert_eq!(
      interpret("Cases[{Infinity, x, 3}, _Symbol]").unwrap(),
      "{x}"
    );
    assert_eq!(
      interpret("Cases[{Infinity, 3, x}, _DirectedInfinity]").unwrap(),
      "{Infinity}"
    );
    // Ordinary head-constrained blanks are unaffected.
    assert_eq!(interpret("MatchQ[y, _Symbol]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[3, _Integer]").unwrap(), "True");
  }

  #[test]
  fn directed_infinity_structured_pattern() {
    // DirectedInfinity[d_] destructures the infinity symbols into their
    // direction: Infinity -> 1, -Infinity -> -1, ComplexInfinity -> {}.
    assert_eq!(
      interpret("Infinity /. DirectedInfinity[d_] :> d").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("-Infinity /. DirectedInfinity[d_] :> d").unwrap(),
      "-1"
    );
    assert_eq!(
      interpret("ComplexInfinity /. DirectedInfinity[d___] :> {d}").unwrap(),
      "{}"
    );
    assert_eq!(
      interpret("MatchQ[Infinity, DirectedInfinity[_]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("Cases[{Infinity, -Infinity, 3}, DirectedInfinity[d_] :> d]")
        .unwrap(),
      "{1, -1}"
    );
    // Nested and explicit DirectedInfinity forms still work.
    assert_eq!(
      interpret("f[Infinity] /. f[DirectedInfinity[d_]] :> d").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("MatchQ[DirectedInfinity[I], DirectedInfinity[_]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn depth_descends_into_operator_forms() {
    // Power (x^2 / Sqrt[x]) and other operator/special forms are stored as
    // BinaryOp/Comparison/Rule etc.; Depth must descend into their canonical
    // FullForm just like Level does.
    assert_eq!(interpret("Depth[x^2]").unwrap(), "2");
    assert_eq!(interpret("Depth[Sqrt[x]]").unwrap(), "2");
    assert_eq!(interpret("Depth[2^x^y]").unwrap(), "3");
    assert_eq!(interpret("Depth[a == b]").unwrap(), "2");
    assert_eq!(interpret("Depth[a -> b]").unwrap(), "2");
    assert_eq!(interpret("Depth[!a]").unwrap(), "2");
    assert_eq!(interpret("Depth[#]").unwrap(), "2");
    assert_eq!(interpret("Depth[#1 + #2]").unwrap(), "3");
    assert_eq!(interpret("Depth[Infinity]").unwrap(), "2");
    // Atoms (including Rational/Complex/constants) stay depth 1.
    assert_eq!(interpret("Depth[Pi]").unwrap(), "1");
    assert_eq!(interpret("Depth[x]").unwrap(), "1");
    // Plain nesting is unchanged.
    assert_eq!(interpret("Depth[f[g[h[x]]]]").unwrap(), "4");
  }

  #[test]
  fn cases_position_count_skip_rational_internals() {
    // A leveled pattern search must not match a Rational's numerator/denom.
    assert_eq!(
      interpret("Cases[{1/2, 3, 5/4}, _Integer, {-1}]").unwrap(),
      "{3}"
    );
    assert_eq!(interpret("Position[1/2, 1]").unwrap(), "{}");
    assert_eq!(interpret("Count[{1/2, 3/4}, _Integer, {-1}]").unwrap(), "0");
  }

  #[test]
  fn inner() {
    assert_eq!(
      interpret("Inner[Times, {1, 2, 3}, {4, 5, 6}, Plus]").unwrap(),
      "32"
    );
    assert_eq!(
      interpret("Inner[Plus, {1, 2}, {3, 4}, Times]").unwrap(),
      "24"
    );
  }

  #[test]
  fn kronecker_product() {
    // vector ⊗ vector → outer-product matrix
    assert_eq!(
      interpret("KroneckerProduct[{1, 2}, {3, 4}]").unwrap(),
      "{{3, 4}, {6, 8}}"
    );
    // unequal-length vectors → m×n matrix
    assert_eq!(
      interpret("KroneckerProduct[{1, 2, 3}, {4, 5}]").unwrap(),
      "{{4, 5}, {8, 10}, {12, 15}}"
    );
    // vector ⊗ matrix → (|v|·rows)×cols
    assert_eq!(
      interpret("KroneckerProduct[{1, 2}, {{1, 0}, {0, 1}}]").unwrap(),
      "{{1, 0}, {0, 1}, {2, 0}, {0, 2}}"
    );
    // matrix ⊗ vector → rows×(cols·|v|)
    assert_eq!(
      interpret("KroneckerProduct[{{1, 0}, {0, 1}}, {3, 4}]").unwrap(),
      "{{3, 4, 0, 0}, {0, 0, 3, 4}}"
    );
    // matrix ⊗ matrix → block matrix
    assert_eq!(
      interpret("KroneckerProduct[{{1, 2}, {3, 4}}, {{0, 1}, {1, 0}}]")
        .unwrap(),
      "{{0, 1, 0, 2}, {1, 0, 2, 0}, {0, 3, 0, 4}, {3, 0, 4, 0}}"
    );
    // n-ary: folds left over all arguments
    assert_eq!(
      interpret("KroneckerProduct[{1, 2}, {3, 4}, {5, 6}]").unwrap(),
      "{{15, 18, 20, 24}, {30, 36, 40, 48}}"
    );
  }

  #[test]
  fn partition_alignment() {
    // Basic forms (no overhang).
    assert_eq!(
      interpret("Partition[{1, 2, 3, 4, 5}, 2]").unwrap(),
      "{{1, 2}, {3, 4}}"
    );
    assert_eq!(
      interpret("Partition[{1, 2, 3, 4, 5}, 2, 1]").unwrap(),
      "{{1, 2}, {2, 3}, {3, 4}, {4, 5}}"
    );
    // Cyclic alignment {kL, kR} with no padding.
    assert_eq!(
      interpret("Partition[{1, 2, 3, 4, 5}, 3, 2, {1, 1}]").unwrap(),
      "{{1, 2, 3}, {3, 4, 5}, {5, 1, 2}}"
    );
    // Scalar alignment k is shorthand for {k, k}: cyclic, no padding.
    assert_eq!(
      interpret("Partition[{a, b, c, d, e}, 2, 1, 2]").unwrap(),
      "{{e, a}, {a, b}, {b, c}, {c, d}, {d, e}}"
    );
    // Scalar alignment with explicit padding keeps and pads the tail block.
    assert_eq!(
      interpret("Partition[{a, b, c, d}, 2, 1, 1, x]").unwrap(),
      "{{a, b}, {b, c}, {c, d}, {d, x}}"
    );
    // Scalar alignment 2 with padding pads both ends.
    assert_eq!(
      interpret("Partition[{a, b, c, d, e}, 3, 2, 2, x]").unwrap(),
      "{{x, a, b}, {b, c, d}, {d, e, x}}"
    );
  }

  #[test]
  fn replace_part() {
    assert_eq!(
      interpret("ReplacePart[{a, b, c}, 2 -> x]").unwrap(),
      "{a, x, c}"
    );
    assert_eq!(
      interpret("ReplacePart[{1, 2, 3, 4}, 1 -> 0]").unwrap(),
      "{0, 2, 3, 4}"
    );
    assert_eq!(
      interpret("ReplacePart[{a, b, c}, -1 -> z]").unwrap(),
      "{a, b, z}"
    );
  }

  #[test]
  fn replace_list() {
    // Enumerates every way a sequence pattern can match, with a fixed
    // Blank slot (b_) between two BlankNullSequence slots.
    assert_eq!(
      interpret("ReplaceList[{1, 2, 3}, {a___, b_, c___} :> {b}]").unwrap(),
      "{{1}, {2}, {3}}"
    );
    // RuleDelayed and Rule behave the same here.
    assert_eq!(
      interpret("ReplaceList[{1, 2, 3}, {a___, b_, c___} -> {b}]").unwrap(),
      "{{1}, {2}, {3}}"
    );
    // Two adjacent fixed slots — every adjacent pair.
    assert_eq!(
      interpret("ReplaceList[{1, 2, 3, 4, 5}, {___, x_, y_, ___} :> x + y]")
        .unwrap(),
      "{3, 5, 7, 9}"
    );
    // Head-constrained slot only matches integer elements.
    assert_eq!(
      interpret("ReplaceList[{1, a, 2, b}, {___, x_Integer, ___} -> x]")
        .unwrap(),
      "{1, 2}"
    );
    // Flat (Orderless) partition enumeration over Plus.
    assert_eq!(
      interpret("ReplaceList[a + b + c, x_ + y_ -> {x, y}]").unwrap(),
      "{{a, b + c}, {b, a + c}, {c, a + b}, {a + b, c}, {a + c, b}, \
       {b + c, a}}"
    );
    // Whole-expression match where the result equals the input.
    assert_eq!(
      interpret("ReplaceList[{1, 2, 3}, x_ -> x]").unwrap(),
      "{{1, 2, 3}}"
    );
    assert_eq!(
      interpret("ReplaceList[f[a, b], x_ -> x]").unwrap(),
      "{f[a, b]}"
    );
    // ReplaceList does not descend into subparts (level-0 only).
    assert_eq!(interpret("ReplaceList[{1, 2, 3}, 2 -> x]").unwrap(), "{}");
    // No match at all.
    assert_eq!(interpret("ReplaceList[a, b -> x]").unwrap(), "{}");
    // Cap the number of matches with the third argument.
    assert_eq!(
      interpret("ReplaceList[{1, 2, 3, 4}, {___, x_, y_, ___} :> {x, y}, 2]")
        .unwrap(),
      "{{1, 2}, {2, 3}}"
    );
  }

  #[test]
  fn array() {
    assert_eq!(interpret("Array[#^2 &, 3]").unwrap(), "{1, 4, 9}");
    assert_eq!(interpret("Array[# + 1 &, 4]").unwrap(), "{2, 3, 4, 5}");
  }

  #[test]
  fn through_two_args() {
    // Through[{f, g}, h] returns {f, g} when head doesn't match h
    assert_eq!(interpret("Through[{Sin, Cos}, 0]").unwrap(), "{Sin, Cos}");
    assert_eq!(
      interpret("Through[{Abs, Sign}, -5]").unwrap(),
      "{Abs, Sign}"
    );
  }

  #[test]
  fn comap() {
    // Comap applies each function in a list to the same argument.
    assert_eq!(
      interpret("Comap[{f, g, h}, x]").unwrap(),
      "{f[x], g[x], h[x]}"
    );
    // The whole second argument is passed to each function.
    assert_eq!(
      interpret("Comap[{f, g}, {1, 2}]").unwrap(),
      "{f[{1, 2}], g[{1, 2}]}"
    );
    // Real functions are evaluated.
    assert_eq!(
      interpret("Comap[{Total, Mean, Max}, {1, 2, 3, 4}]").unwrap(),
      "{10, 5/2, 4}"
    );
    // Associations map over their values, keeping the keys.
    assert_eq!(
      interpret("Comap[<|a -> f, b -> g|>, x]").unwrap(),
      "<|a -> f[x], b -> g[x]|>"
    );
    // The structure of a general head is preserved.
    assert_eq!(interpret("Comap[h[f, g], x]").unwrap(), "h[f[x], g[x]]");
    // Atomic first argument is returned unchanged.
    assert_eq!(interpret("Comap[f, x]").unwrap(), "f");
    assert_eq!(interpret("Comap[3, x]").unwrap(), "3");
    // Default level is {1}: nested lists are applied as a whole.
    assert_eq!(
      interpret("Comap[{f, {g, h}}, x]").unwrap(),
      "{f[x], {g, h}[x]}"
    );
    // An explicit level spec descends further.
    assert_eq!(
      interpret("Comap[{f, {g, h}}, x, 2]").unwrap(),
      "{f[x], {g[x], h[x]}[x]}"
    );
    // Operator form: Comap[funs][x] == Comap[funs, x].
    assert_eq!(interpret("Comap[{f, g}][x]").unwrap(), "{f[x], g[x]}");
    assert_eq!(interpret("Comap[{Min, Max}][{3, 1, 2}]").unwrap(), "{1, 3}");
    // Operator form alone stays symbolic until applied.
    assert_eq!(interpret("Comap[{f, g}]").unwrap(), "Comap[{f, g}]");
    // Empty list.
    assert_eq!(interpret("Comap[{}, x]").unwrap(), "{}");
  }

  #[test]
  fn delete_cases_with_count() {
    // DeleteCases[list, pattern, levelspec, n] deletes at most n matches
    assert_eq!(
      interpret("DeleteCases[{a, b, a, c, a}, a, 1, 1]").unwrap(),
      "{b, a, c, a}"
    );
    assert_eq!(
      interpret("DeleteCases[{a, b, a, c, a}, a, 1, 2]").unwrap(),
      "{b, c, a}"
    );
    // Without count, deletes all
    assert_eq!(
      interpret("DeleteCases[{a, b, a, c, a}, a]").unwrap(),
      "{b, c}"
    );
  }

  #[test]
  fn flatten_with_level() {
    assert_eq!(
      interpret("Flatten[{{{1, 2}}, {{3, 4}}}, 1]").unwrap(),
      "{{1, 2}, {3, 4}}"
    );
    assert_eq!(interpret("Flatten[{{{1}}, {{2}}}, 2]").unwrap(), "{1, 2}");
    // Flatten[list, 0] returns list unchanged
    assert_eq!(
      interpret("Flatten[{{1, 2}, {3, 4}}, 0]").unwrap(),
      "{{1, 2}, {3, 4}}"
    );
  }

  #[test]
  fn flatten_single_level_spec_preserves_others() {
    // Flatten[m, {{2}}] for 3D m: unmentioned old levels 1 and 3 are
    // appended as singleton groups, so the result is equivalent to
    // Flatten[m, {{2}, {1}, {3}}]. Regression for mathics
    // test_structure.py:37.
    assert_eq!(
      interpret("m = {{{1, 2}, {3}}, {{4}, {5, 6}}}; Flatten[m, {{2}}]")
        .unwrap(),
      "{{{1, 2}, {4}}, {{3}, {5, 6}}}"
    );
  }

  #[test]
  fn flatten_with_custom_head() {
    // Flatten[f[g[a, b], g[c, d]], Infinity, g] splices g children into f
    assert_eq!(
      interpret("Flatten[f[g[a, b], g[c, d]], Infinity, g]").unwrap(),
      "f[a, b, c, d]"
    );
  }

  #[test]
  fn flatten_with_custom_head_in_list() {
    assert_eq!(
      interpret("Flatten[{f[a, b], f[c, d]}, Infinity, f]").unwrap(),
      "{a, b, c, d}"
    );
  }

  #[test]
  fn flatten_with_custom_head_nested() {
    assert_eq!(
      interpret("Flatten[f[g[a, b], g[c, g[d, e]]], Infinity, g]").unwrap(),
      "f[a, b, c, d, e]"
    );
  }

  #[test]
  fn flatten_with_custom_head_mixed() {
    // h[b] is not g, so it stays
    assert_eq!(
      interpret("Flatten[f[g[a], h[b], g[c]], Infinity, g]").unwrap(),
      "f[a, h[b], c]"
    );
  }

  // --- FlattenAt ---
  #[test]
  fn flatten_at_positive_index() {
    assert_eq!(
      interpret("FlattenAt[{a, {b, c}, {d, e}, {f}}, 2]").unwrap(),
      "{a, b, c, {d, e}, {f}}"
    );
  }

  #[test]
  fn flatten_at_negative_index() {
    assert_eq!(
      interpret("FlattenAt[{a, {b, c}, {d, e}, {f}}, -1]").unwrap(),
      "{a, {b, c}, {d, e}, f}"
    );
    assert_eq!(
      interpret("FlattenAt[{a, {b, c}, {d, e}, {f}}, -2]").unwrap(),
      "{a, {b, c}, d, e, {f}}"
    );
  }

  #[test]
  fn flatten_at_singleton_position() {
    // {2} is a length-1 position vector — same as integer 2
    assert_eq!(
      interpret("FlattenAt[{a, {b, c}, {d, e}, {f}}, {2}]").unwrap(),
      "{a, b, c, {d, e}, {f}}"
    );
  }

  #[test]
  fn flatten_at_list_of_positions() {
    // {{2}, {4}} flattens at positions 2 and 4
    assert_eq!(
      interpret("FlattenAt[{a, {b, c}, {d, e}, {f}}, {{2}, {4}}]").unwrap(),
      "{a, b, c, {d, e}, f}"
    );
  }

  #[test]
  fn flatten_at_list_of_positions_with_negative() {
    assert_eq!(
      interpret("FlattenAt[{a, {b, c}, {d, e}, {f}}, {{-1}, {2}}]").unwrap(),
      "{a, b, c, {d, e}, f}"
    );
  }

  #[test]
  fn flatten_at_nested_position() {
    // {2, 1} is a single nested position — flatten expr[[2, 1]]
    assert_eq!(
      interpret("FlattenAt[{a, {{b, c}, {d, e}}, {f}}, {2, 1}]").unwrap(),
      "{a, {b, c, {d, e}}, {f}}"
    );
  }

  #[test]
  fn flatten_at_list_of_nested_positions() {
    assert_eq!(
      interpret(
        "FlattenAt[{{a, b}, {c, {d, e}}, {f, {g, h}}}, {{2, 2}, {3, 2}}]"
      )
      .unwrap(),
      "{{a, b}, {c, d, e}, {f, g, h}}"
    );
  }

  #[test]
  fn flatten_at_non_list_head() {
    assert_eq!(
      interpret("FlattenAt[f[a, g[b, c], d], 2]").unwrap(),
      "f[a, b, c, d]"
    );
  }

  #[test]
  fn flatten_at_operator_form() {
    assert_eq!(
      interpret("FlattenAt[2][{a, {b, c}, {d, e}, {f}}]").unwrap(),
      "{a, b, c, {d, e}, {f}}"
    );
  }

  #[test]
  fn flatten_at_empty_position() {
    // FlattenAt[expr, {}] is a no-op
    assert_eq!(interpret("FlattenAt[{a, b}, {}]").unwrap(), "{a, b}");
  }

  // --- SubsetPosition ---
  #[test]
  fn subset_position_literal() {
    assert_eq!(
      interpret("SubsetPosition[{a, b, c, d, e, f}, {b, d}]").unwrap(),
      "{{2, 4}}"
    );
  }

  #[test]
  fn subset_position_pattern() {
    assert_eq!(
      interpret("SubsetPosition[{a, b, c, d, e, f}, {_, c}]").unwrap(),
      "{{1, 3}, {2, 3}, {4, 3}, {5, 3}, {6, 3}}"
    );
  }

  #[test]
  fn subset_position_all_pairs() {
    assert_eq!(
      interpret("SubsetPosition[{1, 2, 3, 4, 5}, {_, _}]").unwrap(),
      "{{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 4}, {2, 5}, {3, 4}, {3, 5}, {4, 5}}"
    );
  }

  #[test]
  fn subset_position_pattern_test() {
    assert_eq!(
      interpret("SubsetPosition[{1, 2, 3, 4, 5}, {_?(# > 3 &)}]").unwrap(),
      "{{4}, {5}}"
    );
  }

  // --- SubsetCases ---
  #[test]
  fn subset_cases_basic() {
    assert_eq!(
      interpret("SubsetCases[{1, 2, 3, 4, 5}, {_, _}]").unwrap(),
      "{{1, 2}, {3, 4}}"
    );
  }

  #[test]
  fn subset_cases_triples() {
    assert_eq!(
      interpret("SubsetCases[{1, 2, 3, 4, 5, 6}, {_, _, _}]").unwrap(),
      "{{1, 2, 3}, {4, 5, 6}}"
    );
  }

  #[test]
  fn subset_cases_with_pattern() {
    assert_eq!(
      interpret("SubsetCases[{1, 2, 3, 4, 5}, {_?(# > 3 &)}]").unwrap(),
      "{{4}, {5}}"
    );
  }

  #[test]
  fn subset_cases_with_condition() {
    assert_eq!(
      interpret("SubsetCases[{a, b, c, d}, {x_, y_} /; OrderedQ[{x, y}]]")
        .unwrap(),
      "{{a, b}, {c, d}}"
    );
  }

  #[test]
  fn subset_cases_with_limit() {
    assert_eq!(
      interpret("SubsetCases[{1, 2, 3, 4, 5}, {_, _}, 1]").unwrap(),
      "{{1, 2}}"
    );
  }

  // --- SubsetCount ---
  #[test]
  fn subset_count_basic() {
    assert_eq!(interpret("SubsetCount[{a, b, c, d}, {a, c}]").unwrap(), "1");
  }

  #[test]
  fn subset_count_pairs() {
    assert_eq!(
      interpret("SubsetCount[{1, 2, 3, 4, 5}, {_, _}]").unwrap(),
      "2"
    );
  }

  #[test]
  fn subset_count_pattern_test() {
    assert_eq!(
      interpret("SubsetCount[{1, 2, 3, 4, 5}, {_?(# > 3 &)}]").unwrap(),
      "2"
    );
  }

  #[test]
  fn range_imax() {
    // Range[imax] generates {1, 2, ..., imax}
    assert_eq!(interpret("Range[5]").unwrap(), "{1, 2, 3, 4, 5}");
    assert_eq!(interpret("Range[1]").unwrap(), "{1}");
    assert_eq!(interpret("Range[0]").unwrap(), "{}");
    // Non-integer upper bound: step defaults to 1, min defaults to 1
    // so the elements are integers (matches wolframscript).
    assert_eq!(interpret("Range[3.5]").unwrap(), "{1, 2, 3}");
  }

  #[test]
  fn range_imin_imax() {
    // Range[imin, imax] generates {imin, ..., imax}
    assert_eq!(interpret("Range[2, 8]").unwrap(), "{2, 3, 4, 5, 6, 7, 8}");
    assert_eq!(interpret("Range[5, 5]").unwrap(), "{5}");
    // Empty when imin > imax and step is positive (the default).
    assert_eq!(interpret("Range[5, 1]").unwrap(), "{}");
    // Integer min with Real (non-integer) max -> integer elements.
    assert_eq!(interpret("Range[1, 3.5]").unwrap(), "{1, 2, 3}");
    // Real min with integer max -> Real elements.
    assert_eq!(interpret("Range[1.0, 5]").unwrap(), "{1., 2., 3., 4., 5.}");
    // Negative bounds.
    assert_eq!(interpret("Range[-2, 2]").unwrap(), "{-2, -1, 0, 1, 2}");
  }

  #[test]
  fn range_imin_imax_step() {
    // Range[imin, imax, di] uses step di.
    assert_eq!(interpret("Range[1, 10, 2]").unwrap(), "{1, 3, 5, 7, 9}");
    // Negative step iterates downward.
    assert_eq!(
      interpret("Range[10, 1, -1]").unwrap(),
      "{10, 9, 8, 7, 6, 5, 4, 3, 2, 1}"
    );
    // Fractional rational step preserves exact rationals.
    assert_eq!(
      interpret("Range[1, 2, 1/4]").unwrap(),
      "{1, 5/4, 3/2, 7/4, 2}"
    );
    // Real step -> Real outputs.
    assert_eq!(
      interpret("Range[0, 1, 0.25]").unwrap(),
      "{0., 0.25, 0.5, 0.75, 1.}"
    );
    // Step that overshoots -> single element.
    assert_eq!(interpret("Range[1, 5, 10]").unwrap(), "{1}");
  }

  #[test]
  fn range_zero_step_errors() {
    // wolframscript emits Range::range and returns the call unevaluated
    // (verified 2026-06-12); the old hard-error behavior diverged.
    assert_eq!(interpret("Range[1, 5, 0]").unwrap(), "Range[1, 5, 0]");
  }

  #[test]
  fn range_symbolic_step_with_numeric_ratio() {
    // Range[a, b, (b - a)/n]: step is symbolic but (max - min)/step
    // simplifies to a numeric value, so the count is determined.
    // Verify count and endpoints rather than exact string formatting.
    assert_eq!(interpret("Length[Range[a, b, (b - a)/4]]").unwrap(), "5");
    assert_eq!(interpret("First[Range[a, b, (b - a)/4]]").unwrap(), "a");
    assert_eq!(interpret("Last[Range[a, b, (b - a)/4]]").unwrap(), "b");

    assert_eq!(interpret("Length[Range[a, b, (b - a)/3]]").unwrap(), "4");
    assert_eq!(interpret("First[Range[a, b, (b - a)/3]]").unwrap(), "a");
    assert_eq!(interpret("Last[Range[a, b, (b - a)/3]]").unwrap(), "b");

    // Mixed numeric/symbolic ratio: (b - a) / ((b - a)/2.5) = 2.5,
    // so count = floor(2.5) + 1 = 3 elements.
    assert_eq!(interpret("Length[Range[a, b, (b - a)/2.5]]").unwrap(), "3");
    assert_eq!(interpret("First[Range[a, b, (b - a)/2.5]]").unwrap(), "a");

    // Adjacent differences should equal the step.
    assert_eq!(
      interpret("Simplify[Range[a, b, (b - a)/4][[2]] - a]").unwrap(),
      interpret("Simplify[(b - a)/4]").unwrap()
    );
  }

  #[test]
  fn range_irrational_ratio() {
    // When (max - min)/step simplifies to an irrational positive real
    // like Pi (~3.14), Range should still enumerate using floor(k) + 1.
    assert_eq!(interpret("Length[Range[a, b, (b - a)/Pi]]").unwrap(), "4");
    assert_eq!(interpret("First[Range[a, b, (b - a)/Pi]]").unwrap(), "a");

    // E ~ 2.718 -> 3 elements
    assert_eq!(interpret("Length[Range[a, b, (b - a)/E]]").unwrap(), "3");

    // GoldenRatio ~ 1.618 -> 2 elements
    assert_eq!(
      interpret("Length[Range[a, b, (b - a)/GoldenRatio]]").unwrap(),
      "2"
    );

    // Sqrt[2] ~ 1.414 -> 2 elements
    assert_eq!(
      interpret("Length[Range[a, b, (b - a)/Sqrt[2]]]").unwrap(),
      "2"
    );
  }

  #[test]
  fn inverse_permutation_list() {
    assert_eq!(
      interpret("InversePermutation[{2, 3, 1}]").unwrap(),
      "{3, 1, 2}"
    );
    assert_eq!(
      interpret("InversePermutation[{2, 3, 1, 4}]").unwrap(),
      "{3, 1, 2, 4}"
    );
    assert_eq!(
      interpret("InversePermutation[{1, 2, 3}]").unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn inverse_permutation_cycles() {
    // Basic two-cycle inversion: each cycle is reversed, then canonicalised
    // so its smallest element comes first and cycles are sorted by their
    // smallest element.
    assert_eq!(
      interpret("InversePermutation[Cycles[{{3, 2, 5, 1}, {4, 7}}]]").unwrap(),
      "Cycles[{{1, 5, 2, 3}, {4, 7}}]"
    );
    // Involutions are their own inverse.
    assert_eq!(
      interpret("InversePermutation[Cycles[{{1, 2}, {3, 4}}]]").unwrap(),
      "Cycles[{{1, 2}, {3, 4}}]"
    );
    // Empty cycle list.
    assert_eq!(
      interpret("InversePermutation[Cycles[{}]]").unwrap(),
      "Cycles[{}]"
    );
    // Single longer cycle requiring rotation.
    assert_eq!(
      interpret("InversePermutation[Cycles[{{2, 3, 5, 1, 4}}]]").unwrap(),
      "Cycles[{{1, 5, 3, 2, 4}}]"
    );
    // Two-element cycles in canonical order are unchanged.
    assert_eq!(
      interpret("InversePermutation[Cycles[{{5, 4, 3, 2, 1}}]]").unwrap(),
      "Cycles[{{1, 2, 3, 4, 5}}]"
    );
    // Multiple cycles get sorted by smallest element after inversion.
    assert_eq!(
      interpret("InversePermutation[Cycles[{{1, 3, 2}, {7, 6, 8}}]]").unwrap(),
      "Cycles[{{1, 2, 3}, {6, 7, 8}}]"
    );
  }

  #[test]
  fn symmetric_group() {
    // SymmetricGroup[n] is a symbolic group object that stays unevaluated
    // as its canonical form (matching wolframscript), with no spurious
    // "not yet implemented" warning.
    assert_eq!(interpret("SymmetricGroup[3]").unwrap(), "SymmetricGroup[3]");
    assert_eq!(interpret("SymmetricGroup[5]").unwrap(), "SymmetricGroup[5]");
    assert_eq!(
      interpret("Head[SymmetricGroup[3]]").unwrap(),
      "SymmetricGroup"
    );

    // GroupOrder[SymmetricGroup[n]] == n! (including the n = 0 trivial case).
    assert_eq!(interpret("GroupOrder[SymmetricGroup[0]]").unwrap(), "1");
    assert_eq!(interpret("GroupOrder[SymmetricGroup[1]]").unwrap(), "1");
    assert_eq!(interpret("GroupOrder[SymmetricGroup[4]]").unwrap(), "24");
    assert_eq!(interpret("GroupOrder[SymmetricGroup[5]]").unwrap(), "120");

    // GroupGenerators[SymmetricGroup[n]]:
    //   n <= 1 -> {}; n == 2 -> single transposition;
    //   n >= 3 -> transposition (1 2) plus the n-cycle.
    assert_eq!(
      interpret("GroupGenerators[SymmetricGroup[0]]").unwrap(),
      "{}"
    );
    assert_eq!(
      interpret("GroupGenerators[SymmetricGroup[1]]").unwrap(),
      "{}"
    );
    assert_eq!(
      interpret("GroupGenerators[SymmetricGroup[2]]").unwrap(),
      "{Cycles[{{1, 2}}]}"
    );
    assert_eq!(
      interpret("GroupGenerators[SymmetricGroup[3]]").unwrap(),
      "{Cycles[{{1, 2}}], Cycles[{{1, 2, 3}}]}"
    );
    assert_eq!(
      interpret("GroupGenerators[SymmetricGroup[4]]").unwrap(),
      "{Cycles[{{1, 2}}], Cycles[{{1, 2, 3, 4}}]}"
    );
  }

  #[test]
  fn dihedral_group() {
    // DihedralGroup[n] is a symbolic group object that stays unevaluated as
    // its canonical form (matching wolframscript), with no spurious
    // "not yet implemented" warning.
    assert_eq!(interpret("DihedralGroup[4]").unwrap(), "DihedralGroup[4]");
    assert_eq!(
      interpret("Head[DihedralGroup[4]]").unwrap(),
      "DihedralGroup"
    );

    // GroupOrder[DihedralGroup[n]] == 2n (with DihedralGroup[1] of order 2).
    assert_eq!(interpret("GroupOrder[DihedralGroup[1]]").unwrap(), "2");
    assert_eq!(interpret("GroupOrder[DihedralGroup[3]]").unwrap(), "6");
    assert_eq!(interpret("GroupOrder[DihedralGroup[4]]").unwrap(), "8");
    assert_eq!(interpret("GroupOrder[DihedralGroup[5]]").unwrap(), "10");

    // GroupGenerators[DihedralGroup[n]].
    assert_eq!(
      interpret("GroupGenerators[DihedralGroup[1]]").unwrap(),
      "{Cycles[{{1, 2}}]}"
    );
    assert_eq!(
      interpret("GroupGenerators[DihedralGroup[2]]").unwrap(),
      "{Cycles[{{1, 2}}], Cycles[{{3, 4}}]}"
    );
    assert_eq!(
      interpret("GroupGenerators[DihedralGroup[4]]").unwrap(),
      "{Cycles[{{1, 4}, {2, 3}}], Cycles[{{1, 2, 3, 4}}]}"
    );

    // GroupElements[DihedralGroup[n]] - all 2n elements, ordered exactly as
    // wolframscript does (lexicographically by image list).
    assert_eq!(
      interpret("GroupElements[DihedralGroup[1]]").unwrap(),
      "{Cycles[{}], Cycles[{{1, 2}}]}"
    );
    assert_eq!(
      interpret("GroupElements[DihedralGroup[2]]").unwrap(),
      "{Cycles[{}], Cycles[{{3, 4}}], Cycles[{{1, 2}}], Cycles[{{1, 2}, {3, 4}}]}"
    );
    assert_eq!(
      interpret("GroupElements[DihedralGroup[3]]").unwrap(),
      "{Cycles[{}], Cycles[{{2, 3}}], Cycles[{{1, 2}}], \
       Cycles[{{1, 2, 3}}], Cycles[{{1, 3, 2}}], Cycles[{{1, 3}}]}"
    );
    assert_eq!(
      interpret("GroupElements[DihedralGroup[4]]").unwrap(),
      "{Cycles[{}], Cycles[{{2, 4}}], Cycles[{{1, 2}, {3, 4}}], \
       Cycles[{{1, 2, 3, 4}}], Cycles[{{1, 3}}], Cycles[{{1, 3}, {2, 4}}], \
       Cycles[{{1, 4, 3, 2}}], Cycles[{{1, 4}, {2, 3}}]}"
    );
    assert_eq!(
      interpret("Length[GroupElements[DihedralGroup[5]]]").unwrap(),
      "10"
    );
  }

  #[test]
  fn alternating_group() {
    // AlternatingGroup[n] is a symbolic group object that stays unevaluated as
    // its canonical form (matching wolframscript), with no spurious
    // "not yet implemented" warning.
    assert_eq!(
      interpret("AlternatingGroup[4]").unwrap(),
      "AlternatingGroup[4]"
    );
    assert_eq!(
      interpret("Head[AlternatingGroup[4]]").unwrap(),
      "AlternatingGroup"
    );

    // GroupOrder[AlternatingGroup[n]] == n!/2 (with order 1 for n <= 1).
    assert_eq!(interpret("GroupOrder[AlternatingGroup[0]]").unwrap(), "1");
    assert_eq!(interpret("GroupOrder[AlternatingGroup[1]]").unwrap(), "1");
    assert_eq!(interpret("GroupOrder[AlternatingGroup[3]]").unwrap(), "3");
    assert_eq!(interpret("GroupOrder[AlternatingGroup[4]]").unwrap(), "12");
    assert_eq!(interpret("GroupOrder[AlternatingGroup[5]]").unwrap(), "60");

    // GroupGenerators[AlternatingGroup[n]].
    assert_eq!(
      interpret("GroupGenerators[AlternatingGroup[4]]").unwrap(),
      "{Cycles[{{1, 2, 3}}], Cycles[{{2, 3, 4}}]}"
    );

    // GroupElements[AlternatingGroup[n]] - all even permutations, ordered
    // lexicographically by image list (matching wolframscript).
    assert_eq!(
      interpret("GroupElements[AlternatingGroup[0]]").unwrap(),
      "{Cycles[{}]}"
    );
    assert_eq!(
      interpret("GroupElements[AlternatingGroup[1]]").unwrap(),
      "{Cycles[{}]}"
    );
    assert_eq!(
      interpret("GroupElements[AlternatingGroup[2]]").unwrap(),
      "{Cycles[{}]}"
    );
    assert_eq!(
      interpret("GroupElements[AlternatingGroup[3]]").unwrap(),
      "{Cycles[{}], Cycles[{{1, 2, 3}}], Cycles[{{1, 3, 2}}]}"
    );
    assert_eq!(
      interpret("GroupElements[AlternatingGroup[4]]").unwrap(),
      "{Cycles[{}], Cycles[{{2, 3, 4}}], Cycles[{{2, 4, 3}}], \
       Cycles[{{1, 2}, {3, 4}}], Cycles[{{1, 2, 3}}], Cycles[{{1, 2, 4}}], \
       Cycles[{{1, 3, 2}}], Cycles[{{1, 3, 4}}], Cycles[{{1, 3}, {2, 4}}], \
       Cycles[{{1, 4, 2}}], Cycles[{{1, 4, 3}}], Cycles[{{1, 4}, {2, 3}}]}"
    );
    assert_eq!(
      interpret("Length[GroupElements[AlternatingGroup[5]]]").unwrap(),
      "60"
    );
  }

  #[test]
  fn permutation_power_list() {
    // List-form permutations (regression coverage for existing behaviour).
    assert_eq!(
      interpret("PermutationPower[{2, 3, 1}, 3]").unwrap(),
      "{1, 2, 3}"
    );
    assert_eq!(
      interpret("PermutationPower[{2, 3, 1}, 2]").unwrap(),
      "{3, 1, 2}"
    );
    assert_eq!(
      interpret("PermutationPower[{2, 3, 1}, -1]").unwrap(),
      "{3, 1, 2}"
    );
  }

  #[test]
  fn permutation_power_cycles() {
    // Sixth power of a permutation with cycle lengths 3 and 4. The
    // 3-cycle returns to identity (6 mod 3 = 0) and is dropped; the
    // 4-cycle becomes its square (two swaps).
    assert_eq!(
      interpret("PermutationPower[Cycles[{{4, 2, 5}, {6, 3, 1, 7}}], 6]")
        .unwrap(),
      "Cycles[{{1, 6}, {3, 7}}]"
    );
    // Inverse squared on the same permutation.
    assert_eq!(
      interpret("PermutationPower[Cycles[{{4, 2, 5}, {6, 3, 1, 7}}], -2]")
        .unwrap(),
      "Cycles[{{1, 6}, {2, 5, 4}, {3, 7}}]"
    );
    // Zeroth power: identity.
    assert_eq!(
      interpret("PermutationPower[Cycles[{{1, 2, 3}}], 0]").unwrap(),
      "Cycles[{}]"
    );
    // Cube of a 3-cycle is identity.
    assert_eq!(
      interpret("PermutationPower[Cycles[{{1, 2, 3}}], 3]").unwrap(),
      "Cycles[{}]"
    );
    // Square of a 3-cycle is its inverse.
    assert_eq!(
      interpret("PermutationPower[Cycles[{{1, 2, 3}}], 2]").unwrap(),
      "Cycles[{{1, 3, 2}}]"
    );
    // Negative power of a 3-cycle.
    assert_eq!(
      interpret("PermutationPower[Cycles[{{1, 2, 3}}], -1]").unwrap(),
      "Cycles[{{1, 3, 2}}]"
    );
    // Odd power of a 2-cycle is itself.
    assert_eq!(
      interpret("PermutationPower[Cycles[{{1, 2}}], 5]").unwrap(),
      "Cycles[{{1, 2}}]"
    );
    // Any power of the identity is the identity.
    assert_eq!(
      interpret("PermutationPower[Cycles[{}], 5]").unwrap(),
      "Cycles[{}]"
    );
  }

  #[test]
  fn permutation_length_cycles() {
    // PermutationLength counts the number of moved points (sum of
    // cycle lengths in Cycles form).
    assert_eq!(
      interpret("PermutationLength[Cycles[{{1, 7, 3, 5}, {2, 12, 9}}]]")
        .unwrap(),
      "7"
    );
    assert_eq!(interpret("PermutationLength[Cycles[{}]]").unwrap(), "0");
  }

  #[test]
  fn permutation_max_cycles() {
    assert_eq!(
      interpret("PermutationMax[Cycles[{{1, 6, 3}, {2, 5, 12, 9}}]]").unwrap(),
      "12"
    );
  }

  #[test]
  fn permutation_min_cycles() {
    assert_eq!(
      interpret("PermutationMin[Cycles[{{3, 4, 6}, {2, 7}}]]").unwrap(),
      "2"
    );
  }

  #[test]
  fn permutation_order_cycles() {
    // Order is the LCM of cycle lengths: LCM(3, 5, 2) = 30.
    assert_eq!(
      interpret(
        "PermutationOrder[Cycles[{{2, 3, 5}, {1, 6, 7, 4, 10}, {8, 9}}]]"
      )
      .unwrap(),
      "30"
    );
    // Identity has order 1.
    assert_eq!(interpret("PermutationOrder[Cycles[{}]]").unwrap(), "1");
  }

  #[test]
  fn permutation_support_cycles() {
    // Support is the sorted union of integers across all cycles.
    assert_eq!(
      interpret("PermutationSupport[Cycles[{{1, 7}, {2, 5, 10, 9}, {4, 6}}]]")
        .unwrap(),
      "{1, 2, 4, 5, 6, 7, 9, 10}"
    );
    // Empty Cycles -> empty support.
    assert_eq!(interpret("PermutationSupport[Cycles[{}]]").unwrap(), "{}");
  }

  #[test]
  fn harmonic_mean_symbolic() {
    // n / Plus[1/x1, ..., 1/xn], with reciprocals printed as x^(-1).
    assert_eq!(
      interpret("HarmonicMean[{a, b, c, d}]").unwrap(),
      "4/(a^(-1) + b^(-1) + c^(-1) + d^(-1))"
    );
    assert_eq!(
      interpret("HarmonicMean[{a, b}]").unwrap(),
      "2/(a^(-1) + b^(-1))"
    );
    // Singleton symbolic list collapses to the single element.
    assert_eq!(interpret("HarmonicMean[{a}]").unwrap(), "a");
    // Mixed numeric and symbolic: numeric reciprocals combine into a rational.
    assert_eq!(
      interpret("HarmonicMean[{2, 4, x}]").unwrap(),
      "3/(3/4 + x^(-1))"
    );
  }

  #[test]
  fn trimmed_mean_asymmetric() {
    // TrimmedMean[list, {f1, f2}]: floor(f1*n) smallest and floor(f2*n)
    // largest elements are removed before averaging.
    assert_eq!(
      interpret("TrimmedMean[{-10, 1, 1, 1, 1, 20}, {0.2, 0}]").unwrap(),
      "24/5"
    );
    assert_eq!(
      interpret("TrimmedMean[{-10, 1, 1, 1, 1, 20}, {0, 0.2}]").unwrap(),
      "-6/5"
    );
    // Drop 1 from start (floor(0.1*10)=1) and 3 from end (floor(0.3*10)=3).
    assert_eq!(
      interpret("TrimmedMean[{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {0.1, 0.3}]")
        .unwrap(),
      "9/2"
    );
    // {0, 0} drops nothing and falls back to the plain mean.
    assert_eq!(
      interpret("TrimmedMean[{1, 2, 3, 4, 5}, {0, 0}]").unwrap(),
      "3"
    );
  }

  #[test]
  fn trimmed_mean_default_fraction() {
    // TrimmedMean[list] defaults to the 5% trimmed mean, i.e. f=0.05.
    // For n=10 with f=0.05, floor(0.5)=0 so nothing is trimmed.
    assert_eq!(
      interpret("TrimmedMean[{-10, 1, 1, 1, 1, 20}]").unwrap(),
      "7/3"
    );
    assert_eq!(
      interpret("TrimmedMean[{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}]").unwrap(),
      "11/2"
    );
  }

  #[test]
  fn trimmed_mean_floor_rounding() {
    // Regression: 0.05 * 10 = 0.5 should floor to 0 (not round to 1),
    // matching wolframscript. Without this fix, an outlier at the end of
    // the list would be incorrectly dropped and the mean shifted.
    assert_eq!(
      interpret("TrimmedMean[{0, 1, 2, 3, 4, 5, 6, 7, 8, 100}, 0.05]").unwrap(),
      "68/5"
    );
    // 0.025 * 20 = 0.5 → floor = 0 → no trimming
    assert_eq!(interpret("TrimmedMean[Range[20], 0.025]").unwrap(), "21/2");
  }

  #[test]
  fn take_list_negative_counts() {
    // Negative counts take from the end of the remaining slice and
    // remove the taken elements from it.
    assert_eq!(
      interpret("TakeList[{a, b, c, d, e, f, g, h}, {-2, -3, -1}]").unwrap(),
      "{{g, h}, {d, e, f}, {c}}"
    );
    // Mixed signs interleave front/back takes.
    assert_eq!(
      interpret("TakeList[{a, b, c, d, e, f, g, h}, {-2, 3, -1}]").unwrap(),
      "{{g, h}, {a, b, c}, {f}}"
    );
  }

  #[test]
  fn take_list_custom_head() {
    // Custom heads are preserved on each sublist.
    assert_eq!(
      interpret("TakeList[h[a, b, c, d, e, f, g], {2, 3, 1}]").unwrap(),
      "{h[a, b], h[c, d, e], h[f]}"
    );
  }

  #[test]
  fn take_list_all_and_upto() {
    // All takes everything remaining.
    assert_eq!(
      interpret("TakeList[{a, b, c, d, e, f, g, h}, {2, 3, All}]").unwrap(),
      "{{a, b}, {c, d, e}, {f, g, h}}"
    );
    // UpTo[n] takes min(n, remaining); an UpTo past the end yields an
    // empty trailing sublist instead of an error.
    assert_eq!(
      interpret("TakeList[Range[12], {2, 3, UpTo[10], UpTo[5]}]").unwrap(),
      "{{1, 2}, {3, 4, 5}, {6, 7, 8, 9, 10, 11, 12}, {}}"
    );
  }

  #[test]
  fn take_list_overrun_emits_iseqs() {
    // An integer spec demanding more than is left aborts with iseqs,
    // referencing the whole spec list and original input, and returns
    // the call unevaluated. (wolframscript parity)
    assert_eq!(
      interpret("TakeList[Range[10], {2, 10}]").unwrap(),
      "TakeList[{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {2, 10}]"
    );
    assert_eq!(
      interpret("TakeList[Range[3], {5}]").unwrap(),
      "TakeList[{1, 2, 3}, {5}]"
    );
    // Negative overrun is also an iseqs case.
    assert_eq!(
      interpret("TakeList[Range[3], {-5}]").unwrap(),
      "TakeList[{1, 2, 3}, {-5}]"
    );
    // Overrun is detected before a later malformed spec is reached.
    assert_eq!(
      interpret("TakeList[Range[5], {10, x}]").unwrap(),
      "TakeList[{1, 2, 3, 4, 5}, {10, x}]"
    );
  }

  #[test]
  fn take_list_malformed_spec_stays_unevaluated() {
    // A non-sequence-spec atom yields seqs and an unevaluated result.
    assert_eq!(
      interpret("TakeList[Range[5], {foo}]").unwrap(),
      "TakeList[{1, 2, 3, 4, 5}, {foo}]"
    );
    assert_eq!(
      interpret("TakeList[Range[5], {x, 10}]").unwrap(),
      "TakeList[{1, 2, 3, 4, 5}, {x, 10}]"
    );
  }

  #[test]
  fn threshold_default() {
    // Threshold[data] uses 10^-10 as the default threshold and replaces
    // elements with |x| <= threshold by zero. The zero substituted in is
    // an integer 0 for integer/rational inputs.
    assert_eq!(
      interpret("Threshold[{1, 10^(-1), 10^(-2), 10^(-8), 10^(-11)}]").unwrap(),
      "{1, 1/10, 1/100, 1/100000000, 0}"
    );
  }

  #[test]
  fn threshold_numeric_list() {
    // |x| <= t triggers replacement (note the boundary case: |2| <= 2 → 0).
    assert_eq!(
      interpret("Threshold[{-3, 1, -2, 0, 2, -1, 0, 1, -3, 3, 2}, 3/2]")
        .unwrap(),
      "{-3, 0, -2, 0, 2, 0, 0, 0, -3, 3, 2}"
    );
    assert_eq!(
      interpret("Threshold[{-3, 1, -2, 0, 2, -1, 0, 1, -3, 3, 2}, 2]").unwrap(),
      "{-3, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0}"
    );
    // Real input: zeros come back as 0. (Real), not 0 (Integer).
    assert_eq!(
      interpret("Threshold[{1.5, -0.5, 0.1, -2.0, 3.0}, 1.0]").unwrap(),
      "{1.5, 0., 0., -2., 3.}"
    );
  }

  #[test]
  fn threshold_rectangular_array() {
    // Nested numeric lists are thresholded element-wise.
    assert_eq!(
      interpret("Threshold[{{1, 2}, {3, 4}}, 2]").unwrap(),
      "{{0, 0}, {3, 4}}"
    );
  }

  #[test]
  fn threshold_methods_firm_hyperbola_smoothgarrote() {
    // "Firm" with a single cutoff is identical in value to "Hard" (|x| <= t
    // → 0, else x), but an inexact threshold promotes the result to Real.
    assert_eq!(
      interpret("Threshold[{0.5, 1., 2., 3., -2.5}, {\"Firm\", 1.5}]").unwrap(),
      "{0., 0., 2., 3., -2.5}"
    );
    assert_eq!(
      interpret("Threshold[{1, 2, 3}, {\"Firm\", 3/2}]").unwrap(),
      "{0, 2, 3}"
    );
    // An inexact threshold makes Firm promote an exact array to reals,
    // whereas bare "Hard" keeps it exact.
    assert_eq!(
      interpret("Threshold[{1, 2, 3}, {\"Firm\", 1.5}]").unwrap(),
      "{0., 2., 3.}"
    );
    assert_eq!(
      interpret("Threshold[{1, 2, 3}, {\"Hard\", 1.5}]").unwrap(),
      "{0, 2, 3}"
    );
    // "Hyperbola": Sign[x] Sqrt[x^2 - t^2] for |x| > t, else 0. Exact inputs
    // stay exact/symbolic.
    assert_eq!(
      interpret("Threshold[{2}, {\"Hyperbola\", 3/2}]").unwrap(),
      "{Sqrt[7]/2}"
    );
    assert_eq!(
      interpret("Threshold[{-3., -1., 1., 3.}, {\"Hyperbola\", 1.5}]").unwrap(),
      "{-2.598076211353316, 0., 0., 2.598076211353316}"
    );
    // "SmoothGarrote": x^3 / (x^2 + t^2) for every element.
    assert_eq!(
      interpret("Threshold[{1, 2, 3}, {\"SmoothGarrote\", 3/2}]").unwrap(),
      "{4/13, 32/25, 12/5}"
    );
    assert_eq!(
      interpret("Threshold[{-3., -0.5, 2., 1.}, {\"SmoothGarrote\", 1.}]")
        .unwrap(),
      "{-2.7, -0.1, 1.6, 0.5}"
    );
  }

  #[test]
  fn threshold_soft_garrote_exact_inputs() {
    // Exact data + exact threshold stays exact for the arithmetic methods
    // (previously these were forced to machine reals).
    assert_eq!(
      interpret("Threshold[{1, 2, 3}, {\"Soft\", 3/2}]").unwrap(),
      "{0, 1/2, 3/2}"
    );
    assert_eq!(
      interpret("Threshold[{1, 2, 3}, {\"PiecewiseGarrote\", 3/2}]").unwrap(),
      "{0, 7/8, 9/4}"
    );
  }

  #[test]
  fn threshold_non_numeric_returns_unevaluated() {
    // A non-numeric element should leave the call unevaluated rather
    // than coerce or silently drop the symbol.
    assert_eq!(
      interpret("Threshold[{1, 2, x}, 1]").unwrap(),
      "Threshold[{1, 2, x}, 1]"
    );
  }

  #[test]
  fn indexed_concrete_vector() {
    // Indexed[list, i] behaves like Part[list, i] for a concrete list.
    assert_eq!(interpret("Indexed[{a, b}, 1]").unwrap(), "a");
    assert_eq!(interpret("Indexed[{a, b}, 2]").unwrap(), "b");
    // Negative indices count from the end.
    assert_eq!(interpret("Indexed[{a, b, c}, -1]").unwrap(), "c");
  }

  #[test]
  fn indexed_concrete_matrix() {
    // Indexed[expr, {i, j, ...}] descends one level per index.
    assert_eq!(
      interpret("Indexed[{{a, b, c}, {d, e, f}}, {1, 2}]").unwrap(),
      "b"
    );
    assert_eq!(
      interpret("Indexed[{{a, b, c}, {d, e, f}}, {2, 3}]").unwrap(),
      "f"
    );
    // A one-element index list still indexes one level.
    assert_eq!(
      interpret("Indexed[{{a, b, c}, {d, e, f}}, {2}]").unwrap(),
      "{d, e, f}"
    );
    assert_eq!(interpret("Indexed[{a, b, c}, {2}]").unwrap(), "b");
  }

  #[test]
  fn indexed_out_of_range_returns_unevaluated() {
    // Out-of-range indices leave the call unevaluated (matching
    // wolframscript's Indexed::partw message).
    assert_eq!(
      interpret("Indexed[{a, b}, 3]").unwrap(),
      "Indexed[{a, b}, 3]"
    );
    // Zero is rejected (Indexed::ind: not a nonzero integer).
    assert_eq!(
      interpret("Indexed[{a, b}, 0]").unwrap(),
      "Indexed[{a, b}, 0]"
    );
  }

  #[test]
  fn indexed_symbolic_normalises_to_list_index() {
    // A non-list argument cannot be indexed concretely, so the call
    // stays symbolic — but the integer index is canonicalised into a
    // singleton list, matching wolframscript.
    assert_eq!(interpret("Indexed[x, 1]").unwrap(), "Indexed[x, {1}]");
  }

  #[test]
  fn center_array_scalar() {
    // Scalar input is centered in a length-n vector; padding split
    // floor((n-k)/2) left, ceil((n-k)/2) right (so for even n a scalar
    // sits just left of middle).
    assert_eq!(interpret("CenterArray[x, 5]").unwrap(), "{0, 0, x, 0, 0}");
    assert_eq!(interpret("CenterArray[x, 4]").unwrap(), "{0, x, 0, 0}");
    assert_eq!(
      interpret("CenterArray[x, 6]").unwrap(),
      "{0, 0, x, 0, 0, 0}"
    );
    // Custom padding element.
    assert_eq!(
      interpret("CenterArray[x, 5, y]").unwrap(),
      "{y, y, x, y, y}"
    );
  }

  #[test]
  fn center_array_no_arg_defaults_to_1() {
    // CenterArray[n] is shorthand for CenterArray[1, n].
    assert_eq!(interpret("CenterArray[5]").unwrap(), "{0, 0, 1, 0, 0}");
  }

  #[test]
  fn center_array_list_in_vector() {
    // Asymmetric padding when (n - k) is odd: 1 left, 2 right.
    assert_eq!(
      interpret("CenterArray[{a, b}, 5]").unwrap(),
      "{0, a, b, 0, 0}"
    );
    // Symmetric padding when (n - k) is even.
    assert_eq!(
      interpret("CenterArray[{a, b, c}, 7]").unwrap(),
      "{0, 0, a, b, c, 0, 0}"
    );
  }

  #[test]
  fn center_array_multi_dim_scalar() {
    // 2D specification with a scalar value.
    assert_eq!(
      interpret("CenterArray[x, {5, 5}]").unwrap(),
      "{{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, x, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}}"
    );
    // Even dims: scalar sits at row 2, col 2 (1-indexed).
    assert_eq!(
      interpret("CenterArray[x, {4, 4}]").unwrap(),
      "{{0, 0, 0, 0}, {0, x, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}"
    );
  }

  #[test]
  fn center_array_multi_dim_block() {
    // 2D centered block.
    assert_eq!(
      interpret("CenterArray[{{a, b}, {c, d}}, {4, 4}]").unwrap(),
      "{{0, 0, 0, 0}, {0, a, b, 0}, {0, c, d, 0}, {0, 0, 0, 0}}"
    );
  }

  #[test]
  fn frechet_mean_symbolic() {
    // Mean exists only when shape > 1; the Piecewise default is Infinity.
    assert_eq!(
      interpret("Mean[FrechetDistribution[a, b]]").unwrap(),
      "Piecewise[{{b*Gamma[1 - a^(-1)], 1 < a}}, Infinity]"
    );
  }

  #[test]
  fn cumulant_basic() {
    // First cumulant is the mean.
    assert_eq!(interpret("Cumulant[{1, 2, 3, 4, 5}, 1]").unwrap(), "3");
    // Second cumulant is the (population) variance.
    assert_eq!(interpret("Cumulant[{1, 2, 3, 4, 5}, 2]").unwrap(), "2");
    // Third cumulant equals the third central moment (zero for symmetric data).
    assert_eq!(interpret("Cumulant[{1, 2, 3, 4, 5}, 3]").unwrap(), "0");
    // Fourth cumulant: m4 - 3 m2^2 = -26/5.
    assert_eq!(interpret("Cumulant[{1, 2, 3, 4, 5}, 4]").unwrap(), "-26/5");
    // Order 0 is always 0.
    assert_eq!(interpret("Cumulant[{1, 2, 3, 4, 5}, 0]").unwrap(), "0");
  }

  #[test]
  fn cumulant_higher_order_exact() {
    // Exact rational results for higher orders (verified against wolframscript).
    assert_eq!(
      interpret("Cumulant[{1, 2, 3, 5, 8, 13, 21}, 5]").unwrap(),
      "-1139161500/16807"
    );
    assert_eq!(
      interpret("Cumulant[{1, 2, 3, 5, 8, 13, 21}, 6]").unwrap(),
      "-96518711318/117649"
    );
  }

  #[test]
  fn cumulant_numeric_and_unevaluated() {
    // Real-valued data yields a machine-precision result.
    assert_eq!(interpret("Cumulant[{1.0, 2, 3, 4, 5}, 2]").unwrap(), "2.");
    assert_eq!(
      interpret("N[Cumulant[{1, 2, 3, 4, 5}, 4]]").unwrap(),
      "-5.2"
    );
    // Symbolic data evaluates via the moment-cumulant recursion. Woxi returns
    // the (population) variance in raw-moment form while wolframscript returns
    // the algebraically-equal central-moment form, so assert the value at a
    // concrete substitution (the population variance of {1, 2, 5} = 26/9),
    // which both engines agree on.
    assert_eq!(
      interpret("Cumulant[{a, b, c}, 2] /. {a -> 1, b -> 2, c -> 5}").unwrap(),
      "26/9"
    );
  }

  #[test]
  fn frechet_variance_symbolic() {
    // Variance exists only when shape > 2.
    assert_eq!(
      interpret("Variance[FrechetDistribution[a, b]]").unwrap(),
      "Piecewise[{{b^2*(Gamma[1 - 2/a] - Gamma[1 - a^(-1)]^2), a > 2}}, Infinity]"
    );
  }

  #[test]
  fn median_exact_rational_and_symbolic() {
    // Rational lists keep an exact result (regression: previously returned
    // the call unevaluated because the f64 sort path rejected rationals).
    assert_eq!(interpret("Median[{1/2, 1/3, 1/4}]").unwrap(), "1/3");
    assert_eq!(interpret("Median[{1/2, 1/3, 1/4, 1/5}]").unwrap(), "7/24");
    // Symbolic-but-real elements are ordered by numeric value and the exact
    // element / mean is preserved.
    assert_eq!(interpret("Median[{Pi, E, 1}]").unwrap(), "E");
    assert_eq!(interpret("Median[{Pi, E, 1, 2}]").unwrap(), "(2 + E)/2");
    assert_eq!(
      interpret("Median[{Sin[1], Cos[1]}]").unwrap(),
      "(Cos[1] + Sin[1])/2"
    );
    // A single inexact element does not coerce the selected exact element.
    assert_eq!(interpret("Median[{1.5, 1/2, 1/3}]").unwrap(), "1/2");
    // Plain integer/real lists are unchanged.
    assert_eq!(interpret("Median[{1, 2, 3, 4}]").unwrap(), "5/2");
    assert_eq!(interpret("Median[{1, 2, 3, 4, 5}]").unwrap(), "3");
    assert_eq!(interpret("Median[{1., 2., 3., 4.}]").unwrap(), "2.5");
    // Non-numeric symbolic lists stay unevaluated (Median::rectn).
    assert_eq!(interpret("Median[{a, b, c}]").unwrap(), "Median[{a, b, c}]");
  }

  #[test]
  fn frechet_median_symbolic() {
    // Median is always defined: b * Log[2]^(-1/a).
    assert_eq!(
      interpret("Median[FrechetDistribution[a, b]]").unwrap(),
      "b/Log[2]^a^(-1)"
    );
  }

  #[test]
  fn frechet_mean_numeric_branches() {
    // Concrete a > 1 collapses the Piecewise into the first branch.
    assert_eq!(
      interpret("Mean[FrechetDistribution[2, 3]]").unwrap(),
      "3*Sqrt[Pi]"
    );
    // a ≤ 1 falls through to the default branch (Infinity).
    assert_eq!(
      interpret("Mean[FrechetDistribution[0.5, 1.5]]").unwrap(),
      "Infinity"
    );
  }

  #[test]
  fn frechet_variance_numeric_branches() {
    // a ≤ 2 → Infinity.
    assert_eq!(
      interpret("Variance[FrechetDistribution[2, 3]]").unwrap(),
      "Infinity"
    );
  }

  #[test]
  fn half_normal_mean_and_variance_symbolic() {
    // Mean = 1/theta; rendered with the negative-exponent form.
    assert_eq!(
      interpret("Mean[HalfNormalDistribution[a]]").unwrap(),
      "a^(-1)"
    );
    // Variance = (Pi - 2) / (2 theta^2).
    assert_eq!(
      interpret("Variance[HalfNormalDistribution[a]]").unwrap(),
      "(-2 + Pi)/(2*a^2)"
    );
  }

  #[test]
  fn half_normal_median_symbolic() {
    // Median = Sqrt[Pi] * InverseErf[1/2] / theta.
    assert_eq!(
      interpret("Median[HalfNormalDistribution[a]]").unwrap(),
      "(Sqrt[Pi]*InverseErf[1/2])/a"
    );
  }

  #[test]
  fn half_normal_numeric_values() {
    // Concrete parameters should collapse to exact rationals.
    assert_eq!(interpret("Mean[HalfNormalDistribution[2]]").unwrap(), "1/2");
    assert_eq!(
      interpret("Variance[HalfNormalDistribution[2]]").unwrap(),
      "(-2 + Pi)/8"
    );
  }

  #[test]
  fn hypoexponential_mean_symbolic() {
    // Mean = sum_i 1/lambda_i. Woxi renders reciprocals as x^(-1).
    assert_eq!(
      interpret("Mean[HypoexponentialDistribution[{a, b, c}]]").unwrap(),
      "a^(-1) + b^(-1) + c^(-1)"
    );
    assert_eq!(
      interpret("Mean[HypoexponentialDistribution[{a, b}]]").unwrap(),
      "a^(-1) + b^(-1)"
    );
  }

  #[test]
  fn hypoexponential_variance_symbolic() {
    // Variance of independent exponentials is sum_i 1/lambda_i^2.
    assert_eq!(
      interpret("Variance[HypoexponentialDistribution[{a, b, c}]]").unwrap(),
      "a^(-2) + b^(-2) + c^(-2)"
    );
  }

  #[test]
  fn hypoexponential_numeric_values() {
    // Concrete rates collapse to a single rational.
    assert_eq!(
      interpret("Mean[HypoexponentialDistribution[{1, 2, 3}]]").unwrap(),
      "11/6"
    );
    assert_eq!(
      interpret("Mean[HypoexponentialDistribution[{1, 2}]]").unwrap(),
      "3/2"
    );
  }

  #[test]
  fn inverse_gaussian_mean_variance() {
    // Mean = mu; Variance = mu^3 / lambda.
    assert_eq!(
      interpret("Mean[InverseGaussianDistribution[m, l]]").unwrap(),
      "m"
    );
    assert_eq!(
      interpret("Variance[InverseGaussianDistribution[m, l]]").unwrap(),
      "m^3/l"
    );
    // Concrete parameters.
    assert_eq!(
      interpret("Mean[InverseGaussianDistribution[2, 3]]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("Variance[InverseGaussianDistribution[2, 3]]").unwrap(),
      "8/3"
    );
  }

  #[test]
  fn gaussian_matrix() {
    // (2r+1)×(2r+1) discrete-Gaussian matrix.
    assert_eq!(
      interpret("Dimensions[GaussianMatrix[3]]").unwrap(),
      "{7, 7}"
    );
    assert_eq!(
      interpret("Dimensions[GaussianMatrix[{2, 1}]]").unwrap(),
      "{5, 5}"
    );
    // Weights normalise to 1.
    assert_eq!(
      interpret("Round[Total[Flatten[GaussianMatrix[2]]]]").unwrap(),
      "1"
    );
    // GaussianMatrix[r] uses sigma = r/2, so it equals the {r, r/2} form.
    assert_eq!(
      interpret("GaussianMatrix[2] === GaussianMatrix[{2, 1}]").unwrap(),
      "True"
    );
    // Rounded values match wolframscript byte-for-byte.
    assert_eq!(
      interpret("Round[GaussianMatrix[1] * 10^6]").unwrap(),
      "{{9876, 79628, 9876}, {79628, 641984, 79628}, {9876, 79628, 9876}}"
    );
    // Non-numeric / invalid arguments stay symbolic.
    assert_eq!(interpret("GaussianMatrix[x]").unwrap(), "GaussianMatrix[x]");
  }

  #[test]
  fn logistic_mean_variance_median() {
    // Mean = mu; Variance = beta^2 * Pi^2 / 3; Median = mu.
    assert_eq!(interpret("Mean[LogisticDistribution[m, b]]").unwrap(), "m");
    assert_eq!(
      interpret("Variance[LogisticDistribution[m, b]]").unwrap(),
      "(b^2*Pi^2)/3"
    );
    assert_eq!(
      interpret("Median[LogisticDistribution[m, b]]").unwrap(),
      "m"
    );
    // Zero-arg form defaults to mu = 0, beta = 1.
    assert_eq!(interpret("Mean[LogisticDistribution[]]").unwrap(), "0");
    assert_eq!(
      interpret("Variance[LogisticDistribution[]]").unwrap(),
      "Pi^2/3"
    );
  }

  #[test]
  fn extreme_value_mean_variance_median_symbolic() {
    // Mean = a + b*EulerGamma; Variance = (b^2 Pi^2)/6;
    // Median = a - b*Log[Log[2]].
    assert_eq!(
      interpret("Mean[ExtremeValueDistribution[a, b]]").unwrap(),
      "a + b*EulerGamma"
    );
    assert_eq!(
      interpret("Variance[ExtremeValueDistribution[a, b]]").unwrap(),
      "(b^2*Pi^2)/6"
    );
    assert_eq!(
      interpret("Median[ExtremeValueDistribution[a, b]]").unwrap(),
      "a - b*Log[Log[2]]"
    );
  }

  #[test]
  fn extreme_value_mean_variance_median_numeric() {
    // Concrete parameters collapse to closed form.
    assert_eq!(
      interpret("Mean[ExtremeValueDistribution[1, 2]]").unwrap(),
      "1 + 2*EulerGamma"
    );
    assert_eq!(
      interpret("Variance[ExtremeValueDistribution[1, 2]]").unwrap(),
      "(2*Pi^2)/3"
    );
    assert_eq!(
      interpret("Median[ExtremeValueDistribution[1, 2]]").unwrap(),
      "1 - 2*Log[Log[2]]"
    );
  }

  #[test]
  fn chi_median_symbolic() {
    // Median[ChiDistribution[v]] = Sqrt[2]*Sqrt[InverseGammaRegularized[v/2, 0, 1/2]].
    assert_eq!(
      interpret("Median[ChiDistribution[v]]").unwrap(),
      "Sqrt[2]*Sqrt[InverseGammaRegularized[v/2, 0, 1/2]]"
    );
  }

  #[test]
  fn chi_median_numeric() {
    // Concrete v: collapses the v/2 argument but keeps the special-function
    // form symbolic since InverseGammaRegularized isn't fully evaluated.
    assert_eq!(
      interpret("Median[ChiDistribution[3]]").unwrap(),
      "Sqrt[2]*Sqrt[InverseGammaRegularized[3/2, 0, 1/2]]"
    );
  }

  #[test]
  fn beta_median_symbolic() {
    // Median[BetaDistribution[a, b]] = InverseBetaRegularized[1/2, a, b]
    // (symbolic; the inverse regularized beta has no elementary closed
    // form for general a, b).
    assert_eq!(
      interpret("Median[BetaDistribution[a, b]]").unwrap(),
      "InverseBetaRegularized[1/2, a, b]"
    );
  }

  #[test]
  fn cycles_canonicalisation() {
    // Cycles drops length-1 cycles, rotates each cycle so the smallest
    // element comes first, and sorts cycles by that first element.
    assert_eq!(
      interpret("Cycles[{{4, 10, 2, 5}, {9}, {7, 1, 18}}]").unwrap(),
      "Cycles[{{1, 18, 7}, {2, 5, 4, 10}}]"
    );
  }

  #[test]
  fn cycles_already_canonical() {
    assert_eq!(
      interpret("Cycles[{{1, 18, 7}, {2, 5, 4, 10}}]").unwrap(),
      "Cycles[{{1, 18, 7}, {2, 5, 4, 10}}]"
    );
  }

  #[test]
  fn cycles_drop_singletons() {
    assert_eq!(interpret("Cycles[{{3}, {4}}]").unwrap(), "Cycles[{}]");
  }

  #[test]
  fn cycles_rotate_smallest_first() {
    // Single cycle: rotate so the smallest is first.
    assert_eq!(
      interpret("Cycles[{{3, 1, 2}}]").unwrap(),
      "Cycles[{{1, 2, 3}}]"
    );
    assert_eq!(
      interpret("Cycles[{{5, 4, 3, 2, 1}}]").unwrap(),
      "Cycles[{{1, 5, 4, 3, 2}}]"
    );
  }

  #[test]
  fn bernoulli_median_symbolic() {
    // Median[BernoulliDistribution[p]] = Piecewise[{{1, p > 1/2}}, 0].
    assert_eq!(
      interpret("Median[BernoulliDistribution[p]]").unwrap(),
      "Piecewise[{{1, p > 1/2}}, 0]"
    );
  }

  #[test]
  fn bernoulli_median_numeric() {
    // p < 1/2 collapses to 0.
    assert_eq!(
      interpret("Median[BernoulliDistribution[0.3]]").unwrap(),
      "0"
    );
    // p > 1/2 collapses to 1.
    assert_eq!(
      interpret("Median[BernoulliDistribution[0.7]]").unwrap(),
      "1"
    );
  }

  #[test]
  fn dagum_median_symbolic() {
    // Median[DagumDistribution[p, a, b]] = b / (-1 + 2^(1/p))^(1/a),
    // from inverting (1 + (b/x)^a)^(-p) = 1/2.
    assert_eq!(
      interpret("Median[DagumDistribution[p, a, b]]").unwrap(),
      "b/(-1 + 2^p^(-1))^a^(-1)"
    );
  }

  #[test]
  fn dagum_median_numeric() {
    // p = 1, a = 2, b = 3: (-1 + 2)^(1/2) = 1, so Median = 3.
    assert_eq!(
      interpret("Median[DagumDistribution[1, 2, 3]]").unwrap(),
      "3"
    );
  }

  #[test]
  fn pareto_median_symbolic() {
    // Median[ParetoDistribution[k, a]] = k * 2^(1/a).
    assert_eq!(
      interpret("Median[ParetoDistribution[k, a]]").unwrap(),
      "2^a^(-1)*k"
    );
  }

  #[test]
  fn pareto_median_numeric() {
    // a = 2: Median = k * Sqrt[2].
    assert_eq!(
      interpret("Median[ParetoDistribution[3, 2]]").unwrap(),
      "3*Sqrt[2]"
    );
    // a = 1: Median = 2k.
    assert_eq!(interpret("Median[ParetoDistribution[5, 1]]").unwrap(), "10");
  }

  #[test]
  fn weibull_median_symbolic() {
    // Median[WeibullDistribution[a, b]] = b * Log[2]^(1/a).
    assert_eq!(
      interpret("Median[WeibullDistribution[a, b]]").unwrap(),
      "b*Log[2]^a^(-1)"
    );
  }

  #[test]
  fn weibull_median_numeric() {
    // a = 2: Median = b * Sqrt[Log[2]].
    assert_eq!(
      interpret("Median[WeibullDistribution[2, 3]]").unwrap(),
      "3*Sqrt[Log[2]]"
    );
    // a = 1 (exponential) collapses to b * Log[2].
    assert_eq!(
      interpret("Median[WeibullDistribution[1, 2]]").unwrap(),
      "2*Log[2]"
    );
  }

  #[test]
  fn cauchy_median_symbolic() {
    // Median[CauchyDistribution[a, b]] = a (Cauchy is symmetric about
    // its location parameter even though Mean is Indeterminate).
    assert_eq!(interpret("Median[CauchyDistribution[a, b]]").unwrap(), "a");
  }

  #[test]
  fn cauchy_median_numeric() {
    assert_eq!(interpret("Median[CauchyDistribution[5, 2]]").unwrap(), "5");
    // Zero-arg form defaults to (0, 1).
    assert_eq!(interpret("Median[CauchyDistribution[]]").unwrap(), "0");
  }

  #[test]
  fn normal_median_symbolic() {
    // The normal distribution is symmetric about its mean, so the
    // median equals mu.
    assert_eq!(interpret("Median[NormalDistribution[m, s]]").unwrap(), "m");
  }

  #[test]
  fn normal_median_numeric() {
    assert_eq!(interpret("Median[NormalDistribution[3, 2]]").unwrap(), "3");
    // Zero-arg form defaults to mu = 0, sigma = 1.
    assert_eq!(interpret("Median[NormalDistribution[]]").unwrap(), "0");
  }

  #[test]
  fn exponential_median_symbolic() {
    // Median[ExponentialDistribution[lambda]] = Log[2]/lambda.
    assert_eq!(
      interpret("Median[ExponentialDistribution[a]]").unwrap(),
      "Log[2]/a"
    );
  }

  #[test]
  fn exponential_median_numeric() {
    // Concrete lambda.
    assert_eq!(
      interpret("Median[ExponentialDistribution[2]]").unwrap(),
      "Log[2]/2"
    );
    // Numeric rate evaluates to a Real.
    let result = interpret("Median[ExponentialDistribution[2.]]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.3465735902799726).abs() < 1e-12, "got {}", val);
  }

  #[test]
  fn lognormal_median_symbolic() {
    // Median[LogNormalDistribution[mu, sigma]] = E^mu (the median of
    // a lognormal is determined entirely by the location parameter).
    assert_eq!(
      interpret("Median[LogNormalDistribution[m, s]]").unwrap(),
      "E^m"
    );
  }

  #[test]
  fn lognormal_median_numeric() {
    // Concrete location collapses to E^1 == E.
    assert_eq!(
      interpret("Median[LogNormalDistribution[1, 2]]").unwrap(),
      "E"
    );
    // E^0 == 1.
    assert_eq!(
      interpret("Median[LogNormalDistribution[0, 1]]").unwrap(),
      "1"
    );
  }

  #[test]
  fn stable_mean_type_zero_symbolic() {
    // Type-0 parametrisation: Mean = mu - beta*sigma*Tan[Pi*alpha/2]
    // when 1 < alpha <= 2; otherwise Indeterminate.
    assert_eq!(
      interpret("Mean[StableDistribution[0, a, b, m, s]]").unwrap(),
      "Piecewise[{{m - b*s*Tan[(a*Pi)/2], Inequality[2, GreaterEqual, a, Greater, 1]}}, Indeterminate]"
    );
  }

  #[test]
  fn stable_mean_type_one_symbolic() {
    // Type-1 parametrisation: Mean = mu when 1 < alpha <= 2;
    // otherwise Indeterminate.
    assert_eq!(
      interpret("Mean[StableDistribution[1, a, b, m, s]]").unwrap(),
      "Piecewise[{{m, Inequality[2, GreaterEqual, a, Greater, 1]}}, Indeterminate]"
    );
  }

  #[test]
  fn stable_variance_symbolic() {
    // Variance is finite only for alpha == 2 (Gaussian limit) for both
    // type-0 and type-1 parametrisations.
    assert_eq!(
      interpret("Variance[StableDistribution[0, a, b, m, s]]").unwrap(),
      "Piecewise[{{2*s^2, a == 2}}, Indeterminate]"
    );
    assert_eq!(
      interpret("Variance[StableDistribution[1, a, b, m, s]]").unwrap(),
      "Piecewise[{{2*s^2, a == 2}}, Indeterminate]"
    );
  }

  #[test]
  fn stable_mean_numeric_branches() {
    // Concrete alpha > 1, beta = 0, mu = 0 collapses to 0 (or 0.).
    assert_eq!(
      interpret("Mean[StableDistribution[0, 1.5, 0, 0, 1]]").unwrap(),
      "0."
    );
    // alpha = 2 (Gaussian) collapses to mu.
    assert_eq!(
      interpret("Mean[StableDistribution[0, 2, 0, 0, 1]]").unwrap(),
      "0"
    );
    // alpha <= 1 falls through to Indeterminate.
    assert_eq!(
      interpret("Mean[StableDistribution[0, 1/2, 0, 0, 1]]").unwrap(),
      "Indeterminate"
    );
  }

  #[test]
  fn arcsin_distribution_median_symbolic() {
    // Median[ArcSinDistribution[{a, b}]] = (a + b)/2 (the distribution
    // is symmetric about the midpoint of its support).
    assert_eq!(
      interpret("Median[ArcSinDistribution[{a, b}]]").unwrap(),
      "(a + b)/2"
    );
  }

  #[test]
  fn arcsin_distribution_median_numeric() {
    // Concrete bounds.
    assert_eq!(
      interpret("Median[ArcSinDistribution[{0, 1}]]").unwrap(),
      "1/2"
    );
    assert_eq!(
      interpret("Median[ArcSinDistribution[{2, 6}]]").unwrap(),
      "4"
    );
    // Zero-argument form defaults to {0, 1}.
    assert_eq!(interpret("Median[ArcSinDistribution[]]").unwrap(), "1/2");
  }

  #[test]
  fn uniform_median_symbolic() {
    // Median[UniformDistribution[{a, b}]] = (a + b)/2.
    assert_eq!(
      interpret("Median[UniformDistribution[{a, b}]]").unwrap(),
      "(a + b)/2"
    );
  }

  #[test]
  fn uniform_median_numeric() {
    // Concrete bounds collapse to a rational.
    assert_eq!(
      interpret("Median[UniformDistribution[{0, 1}]]").unwrap(),
      "1/2"
    );
    assert_eq!(
      interpret("Median[UniformDistribution[{2, 6}]]").unwrap(),
      "4"
    );
    // Zero-arg form defaults to {0, 1}.
    assert_eq!(interpret("Median[UniformDistribution[]]").unwrap(), "1/2");
  }

  #[test]
  fn student_t_median_symbolic() {
    // The t-distribution is symmetric about 0, so Median = 0 for every ν,
    // even when the Mean (Piecewise) is Indeterminate.
    assert_eq!(interpret("Median[StudentTDistribution[v]]").unwrap(), "0");
  }

  #[test]
  fn student_t_median_numeric() {
    // Concrete ν > 1.
    assert_eq!(interpret("Median[StudentTDistribution[3]]").unwrap(), "0");
    // ν = 1 is the Cauchy case; Mean is Indeterminate but Median is still 0.
    assert_eq!(interpret("Median[StudentTDistribution[1]]").unwrap(), "0");
  }

  #[test]
  fn laplace_median_symbolic() {
    // Median[LaplaceDistribution[μ, β]] = μ.
    assert_eq!(interpret("Median[LaplaceDistribution[m, b]]").unwrap(), "m");
  }

  #[test]
  fn laplace_median_numeric() {
    // Concrete location parameter collapses to the integer.
    assert_eq!(interpret("Median[LaplaceDistribution[3, 2]]").unwrap(), "3");
  }

  #[test]
  fn laplace_zero_arg_form() {
    // LaplaceDistribution[] defaults to mean 0, scale 1; Mean, Variance,
    // and Median all collapse to the documented defaults.
    assert_eq!(
      interpret("LaplaceDistribution[]").unwrap(),
      "LaplaceDistribution[0, 1]"
    );
    assert_eq!(interpret("Mean[LaplaceDistribution[]]").unwrap(), "0");
    assert_eq!(interpret("Variance[LaplaceDistribution[]]").unwrap(), "2");
    assert_eq!(interpret("Median[LaplaceDistribution[]]").unwrap(), "0");
  }

  #[test]
  fn rayleigh_median_symbolic() {
    // Median[RayleighDistribution[σ]] = σ*Sqrt[Log[4]].
    assert_eq!(
      interpret("Median[RayleighDistribution[s]]").unwrap(),
      "s*Sqrt[Log[4]]"
    );
  }

  #[test]
  fn rayleigh_median_numeric() {
    // Concrete σ keeps the symbolic Log[4] factor.
    assert_eq!(
      interpret("Median[RayleighDistribution[2]]").unwrap(),
      "2*Sqrt[Log[4]]"
    );
  }

  #[test]
  fn discrete_uniform_median_symbolic() {
    // Median[DiscreteUniformDistribution[{min, max}]]
    //   = -1 + min + Max[1, Ceiling[(1 + max - min)/2]]
    assert_eq!(
      interpret("Median[DiscreteUniformDistribution[{min, max}]]").unwrap(),
      "-1 + min + Max[1, Ceiling[(1 + max - min)/2]]"
    );
  }

  #[test]
  fn discrete_uniform_median_numeric() {
    // Odd-length support: median is the exact middle element.
    assert_eq!(
      interpret("Median[DiscreteUniformDistribution[{1, 5}]]").unwrap(),
      "3"
    );
    // Even-length support uses Ceiling to pick the lower-middle index.
    assert_eq!(
      interpret("Median[DiscreteUniformDistribution[{1, 6}]]").unwrap(),
      "3"
    );
    // Negative-shifted support: {-2, -2, -1, 0, 1, 2} → median 0.
    assert_eq!(
      interpret("Median[DiscreteUniformDistribution[{-2, 2}]]").unwrap(),
      "0"
    );
  }

  #[test]
  fn hypergeometric_mean() {
    // Mean[HypergeometricDistribution[n, ns, nt]] = (n*ns)/nt.
    assert_eq!(
      interpret("Mean[HypergeometricDistribution[n, ns, nt]]").unwrap(),
      "(n*ns)/nt"
    );
    // The audit case with subscripted parameters.
    assert_eq!(
      interpret(
        "Mean[HypergeometricDistribution[n, Subscript[n, succ], \
         Subscript[n, tot]]]"
      )
      .unwrap(),
      "(n*Subscript[n, succ])/Subscript[n, tot]"
    );
    // Concrete parameters collapse to an exact number / a machine real.
    assert_eq!(
      interpret("Mean[HypergeometricDistribution[5, 10, 50]]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Mean[HypergeometricDistribution[10, 20, 100]]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("N[Mean[HypergeometricDistribution[5, 10, 50]]]").unwrap(),
      "1."
    );
  }

  #[test]
  fn binomial_mean_variance_symbolic() {
    // Mean[BinomialDistribution[n, p]] = n*p.
    assert_eq!(
      interpret("Mean[BinomialDistribution[n, p]]").unwrap(),
      "n*p"
    );
    // Variance[BinomialDistribution[n, p]] = n*(1 - p)*p.
    assert_eq!(
      interpret("Variance[BinomialDistribution[n, p]]").unwrap(),
      "n*(1 - p)*p"
    );
  }

  #[test]
  fn binomial_mean_variance_numeric() {
    // Concrete parameters collapse the Mean and Variance to exact numbers.
    assert_eq!(
      interpret("Mean[BinomialDistribution[10, 1/2]]").unwrap(),
      "5"
    );
    assert_eq!(
      interpret("Variance[BinomialDistribution[10, 1/2]]").unwrap(),
      "5/2"
    );
    assert_eq!(
      interpret("Mean[BinomialDistribution[10, 0.5]]").unwrap(),
      "5."
    );
    assert_eq!(
      interpret("Variance[BinomialDistribution[10, 0.5]]").unwrap(),
      "2.5"
    );
  }

  #[test]
  fn inverse_gamma_mean_variance_median_symbolic() {
    // Mean exists only for a > 1; the Piecewise default is Indeterminate.
    assert_eq!(
      interpret("Mean[InverseGammaDistribution[a, b]]").unwrap(),
      "Piecewise[{{b/(-1 + a), a > 1}}, Indeterminate]"
    );
    // Variance exists only for a > 2.
    assert_eq!(
      interpret("Variance[InverseGammaDistribution[a, b]]").unwrap(),
      "Piecewise[{{b^2/((-2 + a)*(-1 + a)^2), a > 2}}, Indeterminate]"
    );
    // Median is defined for all a > 0 via the regularized inverse gamma.
    assert_eq!(
      interpret("Median[InverseGammaDistribution[a, b]]").unwrap(),
      "b/InverseGammaRegularized[a, 1/2]"
    );
  }

  #[test]
  fn inverse_gamma_mean_variance_numeric_branches() {
    // a = 3 > 2, so both Mean and Variance collapse to the first branch.
    assert_eq!(
      interpret("Mean[InverseGammaDistribution[3, 2]]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Variance[InverseGammaDistribution[3, 2]]").unwrap(),
      "1"
    );
    // a = 1/2 < 1 → Mean falls through to the Indeterminate default.
    assert_eq!(
      interpret("Mean[InverseGammaDistribution[1/2, 2]]").unwrap(),
      "Indeterminate"
    );
    // 1 < a = 3/2 < 2 → Variance is Indeterminate even though Mean is finite.
    assert_eq!(
      interpret("Variance[InverseGammaDistribution[3/2, 2]]").unwrap(),
      "Indeterminate"
    );
  }

  #[test]
  fn inverse_gamma_median_numeric() {
    // Median stays symbolic because InverseGammaRegularized is itself
    // symbolic for these arguments.
    assert_eq!(
      interpret("Median[InverseGammaDistribution[3, 2]]").unwrap(),
      "2/InverseGammaRegularized[3, 1/2]"
    );
  }

  #[test]
  fn gompertz_makeham_mean_and_median_symbolic() {
    // Gompertz (two-arg) form: Mean = (E^xi*Gamma[0, xi])/lambda;
    // Median = Log[1 + Log[2]/xi]/lambda.
    assert_eq!(
      interpret("Mean[GompertzMakehamDistribution[l, x]]").unwrap(),
      "(E^x*Gamma[0, x])/l"
    );
    assert_eq!(
      interpret("Median[GompertzMakehamDistribution[l, x]]").unwrap(),
      "Log[1 + Log[2]/x]/l"
    );
  }

  #[test]
  fn gompertz_makeham_mean_and_median_numeric() {
    // Concrete parameters keep the closed form unevaluated where
    // wolframscript also keeps it symbolic (Gamma[0, 3], Log[2]/3).
    assert_eq!(
      interpret("Mean[GompertzMakehamDistribution[2, 3]]").unwrap(),
      "(E^3*Gamma[0, 3])/2"
    );
    assert_eq!(
      interpret("Median[GompertzMakehamDistribution[2, 3]]").unwrap(),
      "Log[1 + Log[2]/3]/2"
    );
  }

  #[test]
  fn extreme_value_zero_arg_form() {
    // Zero-arg form defaults to a = 0, b = 1.
    assert_eq!(
      interpret("ExtremeValueDistribution[]").unwrap(),
      "ExtremeValueDistribution[0, 1]"
    );
    assert_eq!(
      interpret("Mean[ExtremeValueDistribution[]]").unwrap(),
      "EulerGamma"
    );
    assert_eq!(
      interpret("Variance[ExtremeValueDistribution[]]").unwrap(),
      "Pi^2/6"
    );
    assert_eq!(
      interpret("Median[ExtremeValueDistribution[]]").unwrap(),
      "-Log[Log[2]]"
    );
  }

  #[test]
  fn harmonic_mean_matrix() {
    // List-of-lists input → column-wise harmonic mean.
    assert_eq!(
      interpret("HarmonicMean[{{1, 2}, {5, 10}, {5, 2}, {4, 8}}]").unwrap(),
      "{80/33, 160/49}"
    );
    assert_eq!(
      interpret("HarmonicMean[{{a, b}, {c, d}}]").unwrap(),
      "{2/(a^(-1) + c^(-1)), 2/(b^(-1) + d^(-1))}"
    );
  }

  #[test]
  fn permutation_replace() {
    // Single point under a cycle.
    assert_eq!(
      interpret("PermutationReplace[3, Cycles[{{1, 2, 3}}]]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("PermutationReplace[1, Cycles[{{1, 2, 3}}]]").unwrap(),
      "2"
    );
    // Point not moved by the permutation stays fixed.
    assert_eq!(
      interpret("PermutationReplace[5, Cycles[{{1, 2, 3}}]]").unwrap(),
      "5"
    );
    // List of points mapped element-wise.
    assert_eq!(
      interpret("PermutationReplace[{1, 2, 3, 4}, Cycles[{{1, 2, 3}}]]")
        .unwrap(),
      "{2, 3, 1, 4}"
    );
    // Permutation given as a list: point i maps to list[[i]].
    assert_eq!(interpret("PermutationReplace[2, {3, 1, 2}]").unwrap(), "1");
    assert_eq!(
      interpret("PermutationReplace[{1, 2, 3}, {3, 1, 2}]").unwrap(),
      "{3, 1, 2}"
    );
    // Index beyond the permutation list length is fixed.
    assert_eq!(interpret("PermutationReplace[4, {3, 1, 2}]").unwrap(), "4");
    // Non-integer points are left unchanged.
    assert_eq!(
      interpret("PermutationReplace[{a, b}, Cycles[{{1, 2}}]]").unwrap(),
      "{a, b}"
    );
    assert_eq!(
      interpret("PermutationReplace[x, Cycles[{{1, 2}}]]").unwrap(),
      "x"
    );
    // Nested lists are mapped recursively; multiple cycles.
    assert_eq!(
      interpret("PermutationReplace[{{1, 2}, {3, 4}}, Cycles[{{1, 2, 3}}]]")
        .unwrap(),
      "{{2, 3}, {1, 4}}"
    );
    assert_eq!(
      interpret(
        "PermutationReplace[{1, 2, 3, 4, 5}, Cycles[{{1, 2}, {3, 4, 5}}]]"
      )
      .unwrap(),
      "{2, 1, 4, 5, 3}"
    );
  }

  #[test]
  fn distance_matrix() {
    // Default EuclideanDistance: symmetric matrix with zero diagonal.
    assert_eq!(
      interpret("DistanceMatrix[{{0, 0}, {3, 0}, {0, 4}}]").unwrap(),
      "{{0, 3, 4}, {3, 0, 5}, {4, 5, 0}}"
    );
    // Symbolic Euclidean distances are kept exact.
    assert_eq!(
      interpret("DistanceMatrix[{{1, 2}, {3, 4}, {5, 6}, {7, 8}}]").unwrap(),
      "{{0, 2*Sqrt[2], 4*Sqrt[2], 6*Sqrt[2]}, \
{2*Sqrt[2], 0, 2*Sqrt[2], 4*Sqrt[2]}, \
{4*Sqrt[2], 2*Sqrt[2], 0, 2*Sqrt[2]}, \
{6*Sqrt[2], 4*Sqrt[2], 2*Sqrt[2], 0}}"
    );
    // One-dimensional points.
    assert_eq!(
      interpret("DistanceMatrix[{{1}, {4}, {9}}]").unwrap(),
      "{{0, 3, 8}, {3, 0, 5}, {8, 5, 0}}"
    );
    // Single point.
    assert_eq!(interpret("DistanceMatrix[{{0, 0}}]").unwrap(), "{{0}}");
    // DistanceFunction option.
    assert_eq!(
      interpret(
        "DistanceMatrix[{{0, 0}, {3, 0}, {0, 4}}, \
         DistanceFunction -> ManhattanDistance]"
      )
      .unwrap(),
      "{{0, 3, 4}, {3, 0, 7}, {4, 7, 0}}"
    );
    assert_eq!(
      interpret(
        "DistanceMatrix[{{1, 2}, {3, 4}}, \
         DistanceFunction -> SquaredEuclideanDistance]"
      )
      .unwrap(),
      "{{0, 8}, {8, 0}}"
    );
  }

  #[test]
  fn test_random_permutation() {
    // Head is always Cycles, regardless of the drawn permutation.
    assert_eq!(interpret("Head[RandomPermutation[6]]").unwrap(), "Cycles");

    // n = 0 and n = 1 have only the identity permutation.
    assert_eq!(interpret("RandomPermutation[0]").unwrap(), "Cycles[{}]");
    assert_eq!(interpret("RandomPermutation[1]").unwrap(), "Cycles[{}]");

    // n = 2 has exactly two permutations: identity Cycles[{}] and the swap.
    let p2 = interpret("RandomPermutation[2]").unwrap();
    assert!(
      p2 == "Cycles[{}]" || p2 == "Cycles[{{1, 2}}]",
      "unexpected RandomPermutation[2]: {p2}"
    );

    // A list like {4} is not a valid point count / permutation group, so the
    // call stays unevaluated (matching wolframscript's RandomPermutation::grp).
    assert_eq!(
      interpret("Head[RandomPermutation[{4}]]").unwrap(),
      "RandomPermutation"
    );

    // The result is always a valid permutation of 1..n (its list form,
    // padded to length n, sorts back to 1..n).
    assert_eq!(
      interpret("Sort[PermutationList[RandomPermutation[8], 8]]").unwrap(),
      "{1, 2, 3, 4, 5, 6, 7, 8}"
    );
    assert_eq!(
      interpret("PermutationListQ[PermutationList[RandomPermutation[7], 7]]")
        .unwrap(),
      "True"
    );

    // RandomPermutation[n, k] returns a length-k list of Cycles objects.
    assert_eq!(interpret("Length[RandomPermutation[5, 3]]").unwrap(), "3");
    assert_eq!(
      interpret("Map[Head, RandomPermutation[5, 3]]").unwrap(),
      "{Cycles, Cycles, Cycles}"
    );
    assert_eq!(interpret("RandomPermutation[4, 0]").unwrap(), "{}");
  }

  #[test]
  fn array_components() {
    // Identical elements get the same index, by order of first appearance.
    assert_eq!(
      interpret("ArrayComponents[{a, b, a, c, a, b}]").unwrap(),
      "{1, 2, 1, 3, 1, 2}"
    );
    // Works for any element type; structural equality.
    assert_eq!(
      interpret("ArrayComponents[{5, 7, 5, 9, 5, 7}]").unwrap(),
      "{1, 2, 1, 3, 1, 2}"
    );
    assert_eq!(
      interpret("ArrayComponents[{{a, b}, {b, a}}]").unwrap(),
      "{{1, 2}, {2, 1}}"
    );

    // Integer 0 is the background element: mapped to 0 by default.
    assert_eq!(
      interpret("ArrayComponents[{7, 0, 7, 3, 0}]").unwrap(),
      "{1, 0, 1, 2, 0}"
    );
    assert_eq!(
      interpret("ArrayComponents[{0, 0, 0}]").unwrap(),
      "{0, 0, 0}"
    );
    // Only integer 0 is special; real 0. is an ordinary element.
    assert_eq!(interpret("ArrayComponents[{0., a}]").unwrap(), "{1, 2}");

    // Level argument controls labeling depth.
    assert_eq!(
      interpret("ArrayComponents[{{a, b}, {c, a}}, 1]").unwrap(),
      "{1, 2}"
    );
    assert_eq!(
      interpret("ArrayComponents[{{a, b}, {c, a}}, 2]").unwrap(),
      "{{1, 2}, {3, 1}}"
    );
    // Default level descends to the leaves.
    assert_eq!(
      interpret("ArrayComponents[{{{a}, {b}}, {{a}, {c}}}]").unwrap(),
      "{{{1}, {2}}, {{1}, {3}}}"
    );
    // Elements shallower than the target level are labeled as units.
    assert_eq!(
      interpret("ArrayComponents[{{a, b}, c, {a, b}}, 2]").unwrap(),
      "{{1, 2}, 3, {1, 2}}"
    );
    assert_eq!(
      interpret("ArrayComponents[{a, {b, c}}, 1]").unwrap(),
      "{1, 2}"
    );

    // Rules give explicit labels; the rest auto-number with smallest unused.
    assert_eq!(
      interpret("ArrayComponents[{a, b, a, c}, 1, {a -> 99}]").unwrap(),
      "{99, 1, 99, 2}"
    );
    assert_eq!(
      interpret("ArrayComponents[{a, b, c, d, e}, 1, {c -> 1}]").unwrap(),
      "{2, 3, 1, 4, 5}"
    );
    assert_eq!(
      interpret("ArrayComponents[{a, b, c, d}, 1, {a -> 10, b -> 2}]").unwrap(),
      "{10, 2, 1, 3}"
    );
    // A rule can override the default 0 -> 0, or map elements to 0.
    assert_eq!(
      interpret("ArrayComponents[{0, a, b}, 1, {0 -> 5}]").unwrap(),
      "{5, 1, 2}"
    );
    assert_eq!(
      interpret("ArrayComponents[{x, 0, y}, 1, {x -> 0}]").unwrap(),
      "{0, 0, 1}"
    );

    // Empty list.
    assert_eq!(interpret("ArrayComponents[{}]").unwrap(), "{}");

    // Invalid inputs stay unevaluated (matching wolframscript).
    assert_eq!(
      interpret("ArrayComponents[foo]").unwrap(),
      "ArrayComponents[foo]"
    );
    assert_eq!(
      interpret("ArrayComponents[{a, b}, {1}]").unwrap(),
      "ArrayComponents[{a, b}, {1}]"
    );
  }

  #[test]
  fn transpose_permutation_forms() {
    // Partial permutation: trailing levels are left in place. {1} is the
    // identity on level 1, so the tensor is returned unchanged (regression:
    // previously leaked "permutation must cover every level").
    assert_eq!(
      interpret("Transpose[{{1, 2}, {3, 4}}, {1}]").unwrap(),
      "{{1, 2}, {3, 4}}"
    );
    assert_eq!(interpret("Transpose[{1, 2, 3}, {1}]").unwrap(), "{1, 2, 3}");
    // A full permutation transposes as usual.
    assert_eq!(
      interpret("Transpose[{{1, 2, 3}, {4, 5, 6}}, {2, 1}]").unwrap(),
      "{{1, 4}, {2, 5}, {3, 6}}"
    );
    // Repeated destination collapses levels onto a diagonal.
    assert_eq!(
      interpret("Transpose[{{1, 2}, {3, 4}}, {1, 1}]").unwrap(),
      "{1, 4}"
    );
    // Identity permutation longer than the rank is accepted (trailing levels).
    assert_eq!(
      interpret("Transpose[{{1, 2}, {3, 4}}, {1, 2, 3}]").unwrap(),
      "{{1, 2}, {3, 4}}"
    );
    // Rank-3 partial swap of the first two levels.
    assert_eq!(
      interpret("Transpose[Array[a, {2, 2, 2}], {2, 1}]").unwrap(),
      "{{{a[1, 1, 1], a[1, 1, 2]}, {a[2, 1, 1], a[2, 1, 2]}}, \
       {{a[1, 2, 1], a[1, 2, 2]}, {a[2, 2, 1], a[2, 2, 2]}}}"
    );
    // Invalid permutations stay unevaluated (messages go to stderr).
    assert_eq!(
      interpret("Transpose[{{1, 2}, {3, 4}}, {2, 2}]").unwrap(),
      "Transpose[{{1, 2}, {3, 4}}, {2, 2}]"
    );
    assert_eq!(
      interpret("Transpose[{{1, 2}, {3, 4}}, {3}]").unwrap(),
      "Transpose[{{1, 2}, {3, 4}}, {3}]"
    );
    assert_eq!(
      interpret("Transpose[{{1, 2}, {3, 4}}, {0}]").unwrap(),
      "Transpose[{{1, 2}, {3, 4}}, {0}]"
    );
    assert_eq!(
      interpret("Transpose[{{1, 2, 3}, {4, 5, 6}}, {1, 1}]").unwrap(),
      "Transpose[{{1, 2, 3}, {4, 5, 6}}, {1, 1}]"
    );
  }

  #[test]
  fn test_tensor_transpose() {
    // Matrix transpose via explicit permutation.
    assert_eq!(
      interpret("TensorTranspose[{{1, 2}, {3, 4}}, {2, 1}]").unwrap(),
      "{{1, 3}, {2, 4}}"
    );
    // Non-square matrix.
    assert_eq!(
      interpret("TensorTranspose[{{1, 2, 3}, {4, 5, 6}}, {2, 1}]").unwrap(),
      "{{1, 4}, {2, 5}, {3, 6}}"
    );
    // Default permutation swaps the first two levels (same as Transpose).
    assert_eq!(
      interpret("TensorTranspose[{{1, 2, 3}, {4, 5, 6}}]").unwrap(),
      "{{1, 4}, {2, 5}, {3, 6}}"
    );
    // Rank-3 tensor: cycle the levels {2, 3, 1}.
    assert_eq!(
      interpret("Dimensions[TensorTranspose[Array[a, {2, 3, 4}], {2, 3, 1}]]")
        .unwrap(),
      "{4, 2, 3}"
    );
    assert_eq!(
      interpret("Dimensions[TensorTranspose[Array[a, {2, 3, 4}], {3, 1, 2}]]")
        .unwrap(),
      "{3, 4, 2}"
    );
    // Default permutation on a rank-3 tensor only swaps the first two levels.
    assert_eq!(
      interpret("Dimensions[TensorTranspose[Array[a, {2, 3, 4}]]]").unwrap(),
      "{3, 2, 4}"
    );
    // A partial permutation shorter than the rank leaves trailing levels fixed.
    assert_eq!(
      interpret("Dimensions[TensorTranspose[Array[a, {2, 3, 4}], {2, 1}]]")
        .unwrap(),
      "{3, 2, 4}"
    );
    // Identity permutation returns the tensor unchanged.
    assert_eq!(
      interpret("TensorTranspose[{1, 2, 3}, {1}]").unwrap(),
      "{1, 2, 3}"
    );
    // Empty permutation on a scalar is the identity.
    assert_eq!(interpret("TensorTranspose[5, {}]").unwrap(), "5");

    // A spot-check of an actual element permutation on a rank-3 tensor.
    assert_eq!(
      interpret(
        "TensorTranspose[{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, {3, 2, 1}]"
      )
      .unwrap(),
      "{{{1, 5}, {3, 7}}, {{2, 6}, {4, 8}}}"
    );

    // Error: not a valid permutation of 1..Length[perm] (symmperm).
    assert_eq!(
      interpret("TensorTranspose[{{1, 2}, {3, 4}}, {1, 1}]").unwrap(),
      "TensorTranspose[{{1, 2}, {3, 4}}, {1, 1}]"
    );
    assert_eq!(
      interpret("TensorTranspose[{{1, 2}, {3, 4}}, {3, 1}]").unwrap(),
      "TensorTranspose[{{1, 2}, {3, 4}}, {3, 1}]"
    );
    // Error: valid permutation but moves slots beyond tensor rank (ttrank).
    assert_eq!(
      interpret("TensorTranspose[{1, 2, 3}, {2, 1}]").unwrap(),
      "TensorTranspose[{1, 2, 3}, {2, 1}]"
    );
    assert_eq!(
      interpret("TensorTranspose[{1, 2, 3}]").unwrap(),
      "TensorTranspose[{1, 2, 3}]"
    );
  }

  #[test]
  fn test_sequence_replace() {
    // Literal subsequence replacement (non-overlapping, left to right).
    assert_eq!(
      interpret("SequenceReplace[{1, 2, 3, 4, 5}, {2, 3} -> \"X\"]").unwrap(),
      "{1, X, 4, 5}"
    );
    assert_eq!(
      interpret("SequenceReplace[{1, 2, 1, 2, 3, 1, 2}, {1, 2} -> \"X\"]")
        .unwrap(),
      "{X, X, 3, X}"
    );
    // Pattern variables with delayed rule; trailing unmatched element passes through.
    assert_eq!(
      interpret("SequenceReplace[{1, 2, 3, 4, 5}, {a_, b_} :> a + b]").unwrap(),
      "{3, 7, 5}"
    );
    assert_eq!(
      interpret("SequenceReplace[{1, 2, 3, 4, 5, 6}, {a_, b_} :> a + b]")
        .unwrap(),
      "{3, 7, 11}"
    );
    // Condition guard.
    assert_eq!(
      interpret("SequenceReplace[{2, 1, 5, 3, 1, 4}, {x_, y_} /; x > y :> xy]")
        .unwrap(),
      "{xy, xy, 1, 4}"
    );
    // Empty list pattern leaves the list unchanged.
    assert_eq!(
      interpret("SequenceReplace[{1, 2, 3}, {} -> \"X\"]").unwrap(),
      "{1, 2, 3}"
    );
    // Multiple rules tried in order; longer match wins at a given position.
    assert_eq!(
      interpret(
        "SequenceReplace[{1, 2, 3, 4, 5, 6}, {{1, 2} -> \"A\", {3, 4} -> \"B\"}]"
      )
      .unwrap(),
      "{A, B, 5, 6}"
    );
    assert_eq!(
      interpret(
        "SequenceReplace[{1, 2, 3, 1, 2}, {{1, 2, 3} -> \"T\", {1, 2} -> \"D\"}]"
      )
      .unwrap(),
      "{T, D}"
    );
    // Max-replacement count argument.
    assert_eq!(
      interpret("SequenceReplace[{0, 1, 0, 0, 1, 1}, {0, 0} -> \"X\", 1]")
        .unwrap(),
      "{0, 1, X, 1, 1}"
    );
    // BlankSequence is greedy.
    assert_eq!(
      interpret("SequenceReplace[{1, 2, 3, 4, 5}, {a__, b_} :> {a, b}]")
        .unwrap(),
      "{{1, 2, 3, 4, 5}}"
    );
    // Empty input list.
    assert_eq!(interpret("SequenceReplace[{}, {1} -> 0]").unwrap(), "{}");
    // Single-element pattern with evaluated RHS.
    assert_eq!(
      interpret("SequenceReplace[{5, 6, 7}, {x_} :> x^2]").unwrap(),
      "{25, 36, 49}"
    );
  }

  #[test]
  fn test_subset_replace() {
    // A literal contiguous subset is replaced and the rest left in place.
    assert_eq!(
      interpret("SubsetReplace[{1, 2, 3, 4}, {2, 3} -> x]").unwrap(),
      "{1, x, 4}"
    );
    // Every non-overlapping occurrence is replaced.
    assert_eq!(
      interpret("SubsetReplace[{1, 2, 3, 1, 2, 3}, {2, 3} -> x]").unwrap(),
      "{1, x, 1, x}"
    );
    // A length-2 pattern consumes pairs left to right; the odd tail remains.
    assert_eq!(
      interpret("SubsetReplace[{1, 2, 3, 4, 5}, {a_, b_} :> a + b]").unwrap(),
      "{3, 7, 5}"
    );
    // Subsets are combinations, not just contiguous runs: {1, 3} matches the
    // non-adjacent positions 1 and 3, leaving 2.
    assert_eq!(
      interpret("SubsetReplace[{1, 2, 3}, {1, 3} -> x]").unwrap(),
      "{x, 2}"
    );
    // The replacement is emitted at the smallest position of the matched
    // subset; matching is over combinations so {1, 1} pairs positions 2 and 4.
    assert_eq!(
      interpret("SubsetReplace[{5, 1, 5, 1}, {1, 1} -> 0]").unwrap(),
      "{5, 0, 5}"
    );
    // A longer pattern consumes more positions per match.
    assert_eq!(
      interpret("SubsetReplace[{1, 2, 3, 4}, {a_, b_, c_} :> a]").unwrap(),
      "{1, 4}"
    );
    // Conditions on the pattern are honoured.
    assert_eq!(
      interpret(
        "SubsetReplace[{1, 2, 3, 4, 5, 6}, {a_, b_} /; a < b :> a + b]"
      )
      .unwrap(),
      "{3, 7, 11}"
    );
    // Rules are tried in the given order: the length-2 rule wins over the
    // length-1 rule even though it is listed first.
    assert_eq!(
      interpret("SubsetReplace[{1, 2}, {{a_, b_} :> 10, {c_} :> 20}]").unwrap(),
      "{10}"
    );
    // Different-length rules combine, each consuming its own positions.
    assert_eq!(
      interpret("SubsetReplace[{1, 2, 3, 4}, {{1} -> a, {3, 4} -> b}]")
        .unwrap(),
      "{a, 2, b}"
    );
    // No matching subset leaves the list unchanged.
    assert_eq!(
      interpret("SubsetReplace[{1, 2, 3}, {9, 9} -> x]").unwrap(),
      "{1, 2, 3}"
    );
    // The operator form SubsetReplace[rule][list].
    assert_eq!(
      interpret("SubsetReplace[{2, 3} -> x][{1, 2, 3, 4}]").unwrap(),
      "{1, x, 4}"
    );
    // An empty list yields an empty list.
    assert_eq!(interpret("SubsetReplace[{}, {1, 2} -> x]").unwrap(), "{}");
    // A scalar (non-list) rule LHS does not apply: the call is unevaluated.
    assert_eq!(
      interpret("SubsetReplace[{1, 2, 3, 2}, 2 -> x]").unwrap(),
      "SubsetReplace[{1, 2, 3, 2}, 2 -> x]"
    );
    // A non-list first argument stays unevaluated.
    assert_eq!(
      interpret("SubsetReplace[5, {1, 2} -> x]").unwrap(),
      "SubsetReplace[5, {1, 2} -> x]"
    );
  }

  #[test]
  fn fold_while() {
    // FoldWhile[f, x, list, test] returns the first accumulator value that
    // fails the test (the failing value is included).
    assert_eq!(
      interpret("FoldWhile[Times, 2, {2, 3, 4, 5}, # < 100 &]").unwrap(),
      "240"
    );
    assert_eq!(
      interpret("FoldWhile[Plus, 0, {1, 2, 3, 4, 5}, # < 6 &]").unwrap(),
      "6"
    );
    // When the test never fails, the final fold result is returned.
    assert_eq!(
      interpret("FoldWhile[Times, 1, {2, 3, 4}, # < 100 &]").unwrap(),
      "24"
    );
    assert_eq!(
      interpret("FoldWhile[f, x, {a, b, c}, True &]").unwrap(),
      "f[f[f[x, a], b], c]"
    );
    // The initial value is tested first: if it already fails, x is returned
    // with no folding at all.
    assert_eq!(
      interpret("FoldWhile[Times, 2, {2, 3, 4, 5}, # < 1 &]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("FoldWhile[Plus, 100, {1, 2, 3}, # < 6 &]").unwrap(),
      "100"
    );
    // An empty list returns the initial value.
    assert_eq!(interpret("FoldWhile[Plus, 0, {}, # < 6 &]").unwrap(), "0");
    // The 3-argument form takes the initial value from the head of the list.
    assert_eq!(
      interpret("FoldWhile[Plus, {1, 2, 3, 4, 5}, # < 6 &]").unwrap(),
      "6"
    );
    // The `m` argument supplies the last m results to the test.
    assert_eq!(
      interpret("FoldWhile[Times, 1, {2, 2, 3, 3, 4}, Unequal, 2]").unwrap(),
      "144"
    );
    // `All` supplies the entire history to the test.
    assert_eq!(
      interpret("FoldWhile[Times, 2, {2, 3, 4, 5}, # < 100 &, All]").unwrap(),
      "240"
    );
    // A negative `n` steps back n results from where the test failed.
    assert_eq!(
      interpret("FoldWhile[Plus, 0, {1, 2, 3, 4, 5}, # < 6 &, 1, -1]").unwrap(),
      "3"
    );
    // A positive `n` folds n extra times past the failing value.
    assert_eq!(
      interpret("FoldWhile[Plus, 0, {1, 2, 3, 4, 5}, # < 6 &, 1, 1]").unwrap(),
      "10"
    );
  }
}
