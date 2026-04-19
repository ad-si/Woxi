use super::*;

mod anonymous_function_call {
  use super::*;

  #[test]
  fn identity_anonymous() {
    // #&[1] should return 1
    assert_eq!(interpret("#&[1]").unwrap(), "1");
  }

  #[test]
  fn power_anonymous() {
    // #^2&[{1, 2, 3}] should map squaring
    assert_eq!(interpret("#^2 &[{1, 2, 3}]").unwrap(), "{1, 4, 9}");
  }

  #[test]
  fn anonymous_with_addition() {
    assert_eq!(interpret("#+10&[5]").unwrap(), "15");
  }
}

mod function_name_substitution {
  use super::*;

  #[test]
  fn pass_function_as_argument() {
    clear_state();
    assert_eq!(
      interpret_with_stdout("g[f_] := f[]; g[Print[\"Hello\"] &]")
        .unwrap()
        .stdout,
      "Hello\n"
    );
  }

  #[test]
  fn repeat_with_anonymous_function() {
    clear_state();
    assert_eq!(
      interpret_with_stdout(
        "repeat[f_, n_] := Do[f[], {n}]; repeat[Print[\"hi\"] &, 3]"
      )
      .unwrap()
      .stdout,
      "hi\nhi\nhi\n"
    );
  }
}

mod function_head {
  use super::*;

  #[test]
  fn function_one_arg_is_pure_function() {
    // Function[body] is equivalent to body &
    assert_eq!(interpret("Function[# + 1][5]").unwrap(), "6");
  }

  #[test]
  fn function_one_arg_with_multiple_slots() {
    assert_eq!(interpret("Function[#1 + #2][3, 4]").unwrap(), "7");
  }

  #[test]
  fn function_named_param_single() {
    assert_eq!(interpret("Function[x, x + 1][5]").unwrap(), "6");
  }

  #[test]
  fn function_named_param_power() {
    assert_eq!(interpret("Function[x, x^2][3]").unwrap(), "9");
  }

  #[test]
  fn function_named_param_multi() {
    assert_eq!(interpret("Function[{x, y}, x + y][3, 4]").unwrap(), "7");
  }

  #[test]
  fn function_named_param_multiply() {
    assert_eq!(interpret("Function[{x, y}, x*y][3, 4]").unwrap(), "12");
  }

  #[test]
  fn function_named_identity() {
    assert_eq!(interpret("Function[x, x][10]").unwrap(), "10");
  }

  // Regression test for https://github.com/ad-si/Woxi/issues/96
  // Pattern variable names must not leak across parameters in lambdas.
  #[test]
  fn function_named_param_no_variable_leak() {
    assert_eq!(
      interpret("Function[{u, a}, {u, a}][a + 1, 42]").unwrap(),
      "{1 + a, 42}"
    );
  }

  #[test]
  fn function_named_param_no_variable_leak_apply() {
    assert_eq!(
      interpret("Apply[Function[{u, a}, {u, a}], {a + 1, 42}]").unwrap(),
      "{1 + a, 42}"
    );
  }

  #[test]
  fn function_named_param_no_variable_leak_select() {
    // Select uses the two-arg utility path
    assert_eq!(
      interpret("Select[{1, 2, 3}, Function[a, a > 1]]").unwrap(),
      "{2, 3}"
    );
  }

  // With[] must also use simultaneous substitution.
  #[test]
  fn with_no_variable_leak() {
    assert_eq!(
      interpret("With[{u = a + 1, a = 42}, {u, a}]").unwrap(),
      "{1 + a, 42}"
    );
  }

  #[test]
  fn function_display_one_arg() {
    // Function[body] displays as body &
    assert_eq!(interpret("Function[# + 1]").unwrap(), "#1 + 1 & ");
  }

  #[test]
  fn function_display_named_single() {
    assert_eq!(
      interpret("Function[x, x + 1]").unwrap(),
      "Function[x, x + 1]"
    );
  }

  #[test]
  fn function_display_named_multi() {
    assert_eq!(
      interpret("Function[{x, y}, x + y]").unwrap(),
      "Function[{x, y}, x + y]"
    );
  }

  #[test]
  fn function_assigned_to_variable() {
    clear_state();
    assert_eq!(interpret("f = Function[x, x + 1]; f[10]").unwrap(), "11");
  }

  #[test]
  fn function_multi_param_assigned() {
    clear_state();
    assert_eq!(
      interpret("g = Function[{x, y}, x^2 + y^2]; g[3, 4]").unwrap(),
      "25"
    );
  }

  #[test]
  fn function_in_map() {
    assert_eq!(
      interpret("Map[Function[x, x^2], {1, 2, 3, 4}]").unwrap(),
      "{1, 4, 9, 16}"
    );
  }

  #[test]
  fn function_in_select() {
    assert_eq!(
      interpret("Select[{1, 2, 3, 4, 5}, Function[x, OddQ[x]]]").unwrap(),
      "{1, 3, 5}"
    );
  }

  #[test]
  fn function_in_apply() {
    assert_eq!(
      interpret("Apply[Function[{x, y}, x + y], {10, 20}]").unwrap(),
      "30"
    );
  }

  #[test]
  fn function_holdall_body() {
    // Function should not evaluate the body prematurely
    assert_eq!(interpret("Function[x, OddQ[x]][3]").unwrap(), "True");
  }

  #[test]
  fn function_head_is_function() {
    assert_eq!(interpret("Head[Function[x, x + 1]]").unwrap(), "Function");
  }
}

mod paren_anonymous_function {
  use super::*;

  #[test]
  fn paren_anonymous_with_comparison() {
    // (# === "")& is an anonymous function testing for empty string
    // Uses postfix @ operator since direct call syntax is not supported
    assert_eq!(interpret("(# === \"\")& @ \"hello\"").unwrap(), "False");
    assert_eq!(interpret("(# === \"\")& @ \"\"").unwrap(), "True");
  }

  #[test]
  fn paren_anonymous_with_arithmetic() {
    assert_eq!(interpret("(# + 1)& @ 5").unwrap(), "6");
    assert_eq!(interpret("(# * 2 + 3)& @ 4").unwrap(), "11");
  }

  #[test]
  fn paren_anonymous_in_map() {
    assert_eq!(interpret("Map[(# + 1)&, {1, 2, 3}]").unwrap(), "{2, 3, 4}");
  }

  #[test]
  fn paren_anonymous_with_if() {
    assert_eq!(interpret("(If[# > 0, #, 0])& @ 5").unwrap(), "5");
    assert_eq!(interpret("(If[# > 0, #, 0])& @ -3").unwrap(), "0");
  }

  #[test]
  fn paren_anonymous_in_postfix() {
    // If[# === "", i, #]& @ "hello" should return "hello" (strings displayed without quotes)
    assert_eq!(
      interpret("(If[# === \"\", \"empty\", #])& @ \"hello\"").unwrap(),
      "hello"
    );
    assert_eq!(
      interpret("(If[# === \"\", \"empty\", #])& @ \"\"").unwrap(),
      "empty"
    );
  }

  #[test]
  fn paren_anonymous_with_compound_expression() {
    // (expr1; expr2)& should create an anonymous function with CompoundExpression body
    assert_eq!(
      interpret("Reap[Nest[(Sow[#]; 3*# + 1) &, 7, 5]]").unwrap(),
      "{1822, {{7, 22, 67, 202, 607}}}"
    );
  }

  #[test]
  fn paren_anonymous_direct_call() {
    // (expr)&[args] should call the anonymous function directly
    assert_eq!(interpret("(# + 1) &[5]").unwrap(), "6");
    assert_eq!(interpret("(# * 2 + 3) &[4]").unwrap(), "11");
    assert_eq!(interpret("(#1 + #2) &[3, 4]").unwrap(), "7");
  }

  #[test]
  fn function_anonymous_direct_call() {
    // If[...]&[args] should call the anonymous function directly
    assert_eq!(interpret("If[# > 0, #, 0] &[5]").unwrap(), "5");
    assert_eq!(interpret("If[# > 0, #, 0] &[-3]").unwrap(), "0");
  }

  #[test]
  fn list_anonymous_direct_call() {
    // {expr}&[args] should call the anonymous function directly
    assert_eq!(interpret("{#, #^2} &[3]").unwrap(), "{3, 9}");
    assert_eq!(interpret("{#1, #2, #1 + #2} &[2, 5]").unwrap(), "{2, 5, 7}");
  }

  #[test]
  fn paren_call_with_ampersand_inside() {
    // (expr &)[args] — anonymous function with & inside parens, called with bracket args
    assert_eq!(interpret("(# + 1 &)[5]").unwrap(), "6");
    assert_eq!(interpret("(#^2 &)[3]").unwrap(), "9");
    assert_eq!(interpret("(#1 + #2 &)[3, 4]").unwrap(), "7");
  }

  #[test]
  fn paren_call_derivative_anonymous() {
    // (D[#, x] &)[expr] — derivative as anonymous function with & inside parens
    assert_eq!(
      interpret("(D[#, x] &)[x^3 + Sin[x]]").unwrap(),
      "3*x^2 + Cos[x]"
    );
  }

  #[test]
  fn paren_call_if_anonymous() {
    // (If[...] &)[args] — If as anonymous function with & inside parens
    assert_eq!(interpret("(If[# > 0, #, -#] &)[5]").unwrap(), "5");
    assert_eq!(interpret("(If[# > 0, #, -#] &)[-5]").unwrap(), "5");
  }

  #[test]
  fn paren_call_chained() {
    // (expr)[a][b] — chained calls on parenthesized expression
    assert_eq!(interpret("(# &)[#^2 &][3]").unwrap(), "9");
  }
}

mod part_anonymous_function {
  use super::*;

  #[test]
  fn slot_part_simple() {
    // #[[n]]& extracts the nth element
    assert_eq!(interpret("#[[2]] &[{3, 4, 5}]").unwrap(), "4");
    assert_eq!(interpret("#[[1]] &[{10, 20, 30}]").unwrap(), "10");
    assert_eq!(interpret("#[[3]] &[{10, 20, 30}]").unwrap(), "30");
  }

  #[test]
  fn slot_part_in_list_anonymous() {
    // {#[[1]], #[[2]]}& is a list anonymous function with Part extracts
    assert_eq!(interpret("{#[[2]], #[[1]]} &[{3, 4}]").unwrap(), "{4, 3}");
  }

  #[test]
  fn slot_part_with_arithmetic() {
    // #[[1]] + #[[2]]& performs arithmetic on parts
    assert_eq!(interpret("#[[1]] + #[[2]] &[{3, 4}]").unwrap(), "7");
    assert_eq!(interpret("#[[1]] * #[[2]] &[{5, 6}]").unwrap(), "30");
  }

  #[test]
  fn slot_part_in_nest() {
    // Fibonacci via Nest with Part-based anonymous function
    assert_eq!(
      interpret("Nest[{#[[2]], #[[1]] + #[[2]]} &, {0, 1}, 10]").unwrap(),
      "{55, 89}"
    );
  }

  #[test]
  fn slot_part_in_map() {
    // Map with Part-based anonymous function
    assert_eq!(
      interpret("Map[#[[1]] &, {{1, 2}, {3, 4}, {5, 6}}]").unwrap(),
      "{1, 3, 5}"
    );
  }

  #[test]
  fn slot_part_without_anonymous_function() {
    // Part extraction on slots in evaluated contexts (no &)
    assert_eq!(interpret("{1, 2, 3}[[2]]").unwrap(), "2");
    assert_eq!(interpret("x = {10, 20, 30}; x[[3]]").unwrap(), "30");
  }
}

mod postfix_with_anonymous_function {
  use super::*;

  #[test]
  fn postfix_at_with_simple_anonymous() {
    assert_eq!(interpret("#^2& @ 3").unwrap(), "9");
    assert_eq!(interpret("#+1& @ 5").unwrap(), "6");
  }

  #[test]
  fn postfix_at_with_function_anonymous() {
    assert_eq!(interpret("Sqrt[#]& @ 16").unwrap(), "4");
  }

  #[test]
  fn postfix_at_with_string_result() {
    // Anonymous function that returns a string (strings displayed without quotes)
    assert_eq!(
      interpret("If[# > 0, \"positive\", \"non-positive\"]& @ 5").unwrap(),
      "positive"
    );
    assert_eq!(
      interpret("If[# > 0, \"positive\", \"non-positive\"]& @ -3").unwrap(),
      "non-positive"
    );
  }

  #[test]
  fn postfix_at_preserves_string_arg() {
    // When the argument is a string, it should be preserved
    assert_eq!(interpret("StringLength[#]& @ \"hello\"").unwrap(), "5");
  }
}

mod prefix_application_associativity {
  use super::*;

  #[test]
  fn right_associative_chaining() {
    // f @ g @ x should be f[g[x]] (right-associative)
    assert_eq!(
      interpret("Double[x_] := x * 2; Double @ Sin @ (Pi/2)").unwrap(),
      "2"
    );
  }

  #[test]
  fn single_prefix() {
    assert_eq!(interpret("Sqrt @ 16").unwrap(), "4");
  }
}

mod operator_precedence_at_map_apply {
  use super::*;

  #[test]
  fn apply_looser_than_map() {
    // @@ binds looser than /@ — f @@ g /@ h = Apply[f, Map[g, h]]
    assert_eq!(
      interpret("StringJoin @@ ToString /@ IntegerDigits[50, 2]").unwrap(),
      "110010"
    );
  }

  #[test]
  fn prefix_at_tighter_than_map() {
    // @ binds tighter than /@ — f @ g /@ h = Map[f[g], h]
    assert_eq!(
      interpret("FullForm[Hold[f @ g /@ h]]").unwrap(),
      "Hold[Map[f[g], h]]"
    );
  }

  #[test]
  fn prefix_at_tighter_than_apply() {
    // @ binds tighter than @@ — f @ g @@ h = Apply[f[g], h]
    assert_eq!(
      interpret("FullForm[Hold[f @ g @@ h]]").unwrap(),
      "Hold[Apply[f[g], h]]"
    );
  }

  #[test]
  fn anon_func_map_continuation() {
    // f & /@ {1, 2} — & then /@ should work even for single-term body
    assert_eq!(
      interpret("FullForm[Hold[f & /@ g @ h]]").unwrap(),
      "Hold[Map[Function[f], g[h]]]"
    );
  }

  #[test]
  fn anon_func_single_term_with_map() {
    // x & /@ {1, 2} — single identifier before & with Map continuation
    assert_eq!(interpret("42 & /@ {1, 2, 3}").unwrap(), "{42, 42, 42}");
  }
}

mod variable_as_function_head {
  use super::*;

  #[test]
  fn variable_holding_function_name() {
    clear_state();
    assert_eq!(
      interpret("t = Flatten; t @ {{1, 2}, {3, 4}}").unwrap(),
      "{1, 2, 3, 4}"
    );
  }

  #[test]
  fn variable_in_apply() {
    clear_state();
    assert_eq!(interpret("f = Plus; f @@ {1, 2, 3}").unwrap(), "6");
  }

  #[test]
  fn variable_in_map() {
    clear_state();
    assert_eq!(
      interpret("f = ToString; f /@ {1, 2, 3}").unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn variable_in_function_call() {
    clear_state();
    assert_eq!(interpret("f = Length; f[{1, 2, 3}]").unwrap(), "3");
  }
}

mod expression_level_anonymous_function {
  use super::*;

  #[test]
  fn multi_operator_body() {
    // #^2 + 1 & — body has multiple operators, not just Slot op Term
    assert_eq!(interpret("Map[#^2 + 1 &, {3, 0}]").unwrap(), "{10, 1}");
    assert_eq!(interpret("Map[# * 2 - 3 &, {5, 10}]").unwrap(), "{7, 17}");
  }

  #[test]
  fn replace_all_body() {
    // # /. {rules} & — body contains ReplaceAll with conditional patterns
    assert_eq!(
      interpret(
        "Map[# /. {n_ /; EvenQ[n] :> n/2, n_ /; OddQ[n] :> 3 n + 1} &, {27, 6}]"
      )
      .unwrap(),
      "{82, 3}"
    );
  }

  #[test]
  fn replace_all_simple_rule() {
    // # /. rule & — body contains ReplaceAll with a single rule
    assert_eq!(
      interpret("Map[# /. x_ /; x > 3 :> 0 &, {1, 2, 5, 4, 3}]").unwrap(),
      "{1, 2, 0, 0, 3}"
    );
  }

  #[test]
  fn nestlist_collatz() {
    // Full Collatz sequence via NestList with ReplaceAll in anonymous function
    assert_eq!(
      interpret("NestList[# /. {n_ /; EvenQ[n] :> n/2, n_ /; OddQ[n] :> 3 n + 1} &, 27, 10]")
        .unwrap(),
      "{27, 82, 41, 124, 62, 31, 94, 47, 142, 71, 214}"
    );
  }

  #[test]
  fn replace_all_variable_rules() {
    // # /. r & — RHS of /. is a variable holding rules
    assert_eq!(
      interpret("r = {x -> 1, y -> 2}; {x, y, z} /. r").unwrap(),
      "{1, 2, z}"
    );
    // In anonymous function context
    assert_eq!(
      interpret("r = {x_ /; EvenQ[x] :> x/2}; Map[# /. r &, {4, 7}]").unwrap(),
      "{2, 7}"
    );
    // Nest with variable rules
    assert_eq!(
      interpret("r = {a -> b, b -> c}; Nest[# /. r &, a, 2]").unwrap(),
      "c"
    );
  }

  #[test]
  fn replace_repeated_variable_rules() {
    // # //. r — RHS of //. is a variable holding rules
    assert_eq!(interpret("r = {a -> b, b -> c}; a //. r").unwrap(), "c");
  }

  #[test]
  fn replace_repeated_with_pattern() {
    // ReplaceRepeated should handle pattern rules like f[y_] -> y
    assert_eq!(interpret("f[f[f[x]]] //. f[y_] -> y").unwrap(), "x");
    assert_eq!(
      interpret("ReplaceRepeated[f[f[f[x]]], f[y_] -> y]").unwrap(),
      "x"
    );
  }

  #[test]
  fn replace_repeated_evaluates_after_substitution() {
    // After substitution, the result should be re-evaluated
    assert_eq!(interpret("x^2 //. x -> 3").unwrap(), "9");
    assert_eq!(interpret("{1 + a, 2 + a} //. a -> 0").unwrap(), "{1, 2}");
  }

  #[test]
  fn postfix_application_body() {
    // body // func & — body contains postfix application
    assert_eq!(
      interpret("Map[# // Abs &, {-3, 2, -1}]").unwrap(),
      "{3, 2, 1}"
    );
  }

  #[test]
  fn existing_forms_still_work() {
    // Simple: Slot op Term (uses SimpleAnonymousFunction)
    assert_eq!(interpret("Map[# + 1 &, {5}]").unwrap(), "{6}");
    // Function call (uses FunctionAnonymousFunction)
    assert_eq!(interpret("Map[Sin[#] &, {0}]").unwrap(), "{0}");
    // Paren (uses ParenAnonymousFunction)
    assert_eq!(interpret("(# + 1) &[5]").unwrap(), "6");
    // List (uses ListAnonymousFunction)
    assert_eq!(interpret("{#, #^2} &[3]").unwrap(), "{3, 9}");
  }
}

mod parallel_map {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("ParallelMap[f, {1, 2, 3}]").unwrap(),
      "{f[1], f[2], f[3]}"
    );
  }

  #[test]
  fn with_function() {
    assert_eq!(
      interpret("ParallelMap[#^2 &, {1, 2, 3}]").unwrap(),
      "{1, 4, 9}"
    );
  }
}

mod map_apply_function {
  use super::*;

  #[test]
  fn map_apply_basic() {
    assert_eq!(
      interpret("MapApply[f, {{a, b}, {c, d}}]").unwrap(),
      "{f[a, b], f[c, d]}"
    );
  }

  #[test]
  fn map_apply_with_plus() {
    assert_eq!(
      interpret("MapApply[Plus, {{1, 2}, {3, 4}}]").unwrap(),
      "{3, 7}"
    );
  }

  #[test]
  fn map_apply_with_times() {
    assert_eq!(
      interpret("MapApply[Times, {{2, 3}, {4, 5}}]").unwrap(),
      "{6, 20}"
    );
  }

  #[test]
  fn map_apply_matches_operator_form() {
    // MapApply[f, list] should give the same result as f @@@ list
    assert_eq!(
      interpret("MapApply[f, {{a, b}, {c, d}}]").unwrap(),
      interpret("f @@@ {{a, b}, {c, d}}").unwrap()
    );
  }

  #[test]
  fn map_apply_with_user_function() {
    assert_eq!(
      interpret("g[x_, y_] := x + y; MapApply[g, {{1, 2}, {3, 4}}]").unwrap(),
      "{3, 7}"
    );
  }

  #[test]
  fn map_apply_empty_list() {
    assert_eq!(interpret("MapApply[f, {}]").unwrap(), "{}");
  }

  #[test]
  fn map_apply_single_element() {
    assert_eq!(
      interpret("MapApply[f, {{a, b, c}}]").unwrap(),
      "{f[a, b, c]}"
    );
  }

  #[test]
  fn map_apply_with_pure_function() {
    assert_eq!(
      interpret("MapApply[Function[{x, y}, x^y], {{2, 3}, {3, 2}}]").unwrap(),
      "{8, 9}"
    );
  }
}

mod block_map {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("BlockMap[g, {a, b, c, d, e, h}, 2]").unwrap(),
      "{g[{a, b}], g[{c, d}], g[{e, h}]}"
    );
  }

  #[test]
  fn with_total() {
    assert_eq!(
      interpret("BlockMap[Total, {1, 2, 3, 4, 5, 6}, 2]").unwrap(),
      "{3, 7, 11}"
    );
  }

  #[test]
  fn block_size_3() {
    assert_eq!(
      interpret("BlockMap[g, {a, b, c, d, e, h}, 3]").unwrap(),
      "{g[{a, b, c}], g[{d, e, h}]}"
    );
  }

  #[test]
  fn drops_remainder() {
    assert_eq!(
      interpret("BlockMap[g, {a, b, c, d, e}, 3]").unwrap(),
      "{g[{a, b, c}]}"
    );
  }

  #[test]
  fn with_offset_1() {
    assert_eq!(
      interpret("BlockMap[g, {a, b, c, d, e}, 3, 1]").unwrap(),
      "{g[{a, b, c}], g[{b, c, d}], g[{c, d, e}]}"
    );
  }

  #[test]
  fn with_offset_2() {
    assert_eq!(
      interpret("BlockMap[g, {a, b, c, d, e, h, i}, 3, 2]").unwrap(),
      "{g[{a, b, c}], g[{c, d, e}], g[{e, h, i}]}"
    );
  }
}

mod operate {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(interpret("Operate[p, f[a, b]]").unwrap(), "p[f][a, b]");
  }

  #[test]
  fn level_1() {
    assert_eq!(interpret("Operate[p, f[a, b], 1]").unwrap(), "p[f][a, b]");
  }

  #[test]
  fn level_0() {
    assert_eq!(
      interpret("Operate[p, f[a][b][c], 0]").unwrap(),
      "p[f[a][b][c]]"
    );
  }

  #[test]
  fn atomic_argument_unchanged() {
    // Operate[p, f] — f is atomic, nothing to operate on; return f unchanged.
    assert_eq!(interpret("Operate[p, f]").unwrap(), "f");
  }

  #[test]
  fn depth_exceeds_nesting_returns_unchanged() {
    // f[a][b][c] has nesting depth 3; Operate at depth 4 returns it unchanged.
    assert_eq!(
      interpret("Operate[p, f[a][b][c], 4]").unwrap(),
      "f[a][b][c]"
    );
  }

  #[test]
  fn level_3_on_triple_curried() {
    // Operate[p, f[a][b][c], 3] wraps the innermost symbol `f`.
    assert_eq!(
      interpret("Operate[p, f[a][b][c], 3]").unwrap(),
      "p[f][a][b][c]"
    );
  }
}

mod dimension_mismatch_returns_unevaluated {
  use super::*;

  #[test]
  fn thread_unequal_lengths() {
    // Thread with unequal length lists should return unevaluated, not error
    assert_eq!(
      interpret("Thread[f[{a, b, c}, {1, 2}]]").unwrap(),
      "f[{a, b, c}, {1, 2}]"
    );
  }

  #[test]
  fn map_thread_unequal_lengths() {
    assert_eq!(
      interpret("MapThread[f, {{a, b}, {1, 2, 3}}]").unwrap(),
      "MapThread[f, {{a, b}, {1, 2, 3}}]"
    );
  }

  #[test]
  fn transpose_ragged() {
    assert_eq!(
      interpret("Transpose[{{1, 2}, {3}}]").unwrap(),
      "Transpose[{{1, 2}, {3}}]"
    );
  }

  #[test]
  fn association_thread_unequal() {
    assert_eq!(
      interpret("AssociationThread[{a, b, c}, {1, 2}]").unwrap(),
      "AssociationThread[{a, b, c}, {1, 2}]"
    );
  }

  #[test]
  fn inner_incompatible() {
    assert_eq!(
      interpret("Inner[f, {a, b}, {x, y, z}, g]").unwrap(),
      "Inner[f, {a, b}, {x, y, z}, g]"
    );
  }
}

mod named_function_arity_check {
  use super::*;

  #[test]
  fn too_few_args_returns_unevaluated() {
    // Function[{x, y}, body] called with 1 argument should not partially bind
    assert_eq!(
      interpret("Function[{x, y}, x + y][{1, 2}]").unwrap(),
      "Function[{x, y}, x + y][{1, 2}]"
    );
  }

  #[test]
  fn exact_args_works() {
    assert_eq!(interpret("Function[{x, y}, x + y][1, 2]").unwrap(), "3");
  }

  #[test]
  fn single_param_single_arg_works() {
    assert_eq!(interpret("Function[{x}, x^2][3]").unwrap(), "9");
  }
}
