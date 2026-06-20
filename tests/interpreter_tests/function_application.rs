use super::*;

mod map_and_part_application {
  use super::*;

  #[test]
  fn map_named_function() {
    // Map of a named Function must apply it (bind the parameter), not append
    // the element as another argument. The closure `i^2 &` captures i.
    assert_eq!(
      interpret("Function[i, i^2 &] /@ Range[3]").unwrap(),
      "{1^2 & , 2^2 & , 3^2 & }"
    );
    assert_eq!(
      interpret("Function[x, x*2] /@ {10, 20}").unwrap(),
      "{20, 40}"
    );
  }

  #[test]
  fn apply_part_result() {
    // A Part result can be applied as a function: a[[i]][args].
    assert_eq!(
      interpret("fs = {1^2 &, 2^2 &, 3^2 &}; fs[[2]][]").unwrap(),
      "4"
    );
    assert_eq!(interpret("g = {f, h}; g[[1]][3]").unwrap(), "f[3]");
  }
}

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

  // A single-parameter `Function[{y}, body]` keeps its braces on display,
  // distinct from the no-braces form `Function[y, body]`. Matches Wolfram.
  #[test]
  fn function_single_param_preserves_braces() {
    assert_eq!(
      interpret("Function[{y}, f[y]]").unwrap(),
      "Function[{y}, f[y]]"
    );
    assert_eq!(interpret("Function[y, f[y]]").unwrap(), "Function[y, f[y]]");
  }

  // ReplaceAll into the body must preserve the original parameter form:
  // bracketed-in stays bracketed-out, bare stays bare.
  #[test]
  fn function_replace_all_preserves_param_form() {
    assert_eq!(
      interpret("Function[{y}, f[x, y]] /. x -> y").unwrap(),
      "Function[{y}, f[y, y]]"
    );
    assert_eq!(
      interpret("Function[y, f[x, y]] /. x -> y").unwrap(),
      "Function[y, f[y, y]]"
    );
  }

  // Function application does capture-avoiding substitution: when the
  // argument contains an identifier that clashes with an inner Function's
  // bound parameter, the inner parameter is alpha-renamed. Matches
  // Wolfram's convention of appending `$` to the name.
  #[test]
  fn function_application_alpha_renames_captured_params() {
    assert_eq!(
      interpret("Function[{x}, Function[{y}, f[x, y]]][y]").unwrap(),
      "Function[{y$}, f[y, y$]]"
    );
  }

  // Two-stage application with the same name in inner and outer scopes
  // — the inner param (bare) gets renamed on the first step so the second
  // step doesn't see capture. Result is `x^y`, not `y^y`.
  #[test]
  fn function_double_application_resolves_hygiene() {
    assert_eq!(
      interpret("Function[y, Function[x, y^x]][x][y]").unwrap(),
      "x^y"
    );
  }

  // Chained `& [...]` anonymous-function calls: `g[#] & [h[#]] & [5]`
  // parses as `((g[#] &)[h[#]] &)[5]`. The grammar previously allowed
  // only one anonymous-function suffix per Expression, so the outer
  // `& [5]` triggered a parse error. Matches wolframscript output
  // `g[h[5]]`.
  #[test]
  fn anonymous_function_chained_application() {
    assert_eq!(interpret("g[#] & [h[#]] & [5]").unwrap(), "g[h[5]]");
  }

  #[test]
  fn anonymous_function_chained_constant_ignores_inner_slot() {
    // `(f &)[x]` returns `f` regardless of the slot argument, and wrapping
    // that in another `& [y]` still returns `f`.
    assert_eq!(interpret("f & [x] & [y]").unwrap(), "f");
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
  fn apply_descends_into_binary_op_at_infinity() {
    // Apply[List, expr, {0, Infinity}] must descend into binary ops like
    // Power that the parser leaves as Expr::BinaryOp. Matches wolframscript.
    assert_eq!(
      interpret("Apply[List, a + b * c ^ e * f[g], {0, Infinity}]").unwrap(),
      "{a, {b, {c, e}, {g}}}"
    );
  }

  #[test]
  fn apply_negative_levelspec() {
    // Negative level -k selects subexpressions of Depth k (here -3 means
    // Depth 3 subexpressions). For {{{{{a}}}}} (Depth 6), {2, -3} = {2, 3}.
    assert_eq!(
      interpret("Apply[f, {{{{{a}}}}}, {2, -3}]").unwrap(),
      "{{f[f[{a}]]}}"
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
      "FullForm[Hold[f[g] /@ h]]"
    );
  }

  #[test]
  fn prefix_at_tighter_than_apply() {
    // @ binds tighter than @@ — f @ g @@ h = Apply[f[g], h]
    assert_eq!(
      interpret("FullForm[Hold[f @ g @@ h]]").unwrap(),
      "FullForm[Hold[f[g] @@ h]]"
    );
  }

  #[test]
  fn anon_func_map_continuation() {
    // f & /@ {1, 2} — & then /@ should work even for single-term body
    assert_eq!(
      interpret("FullForm[Hold[f & /@ g @ h]]").unwrap(),
      "FullForm[Hold[(f & ) /@ g[h]]]"
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

  // ParallelMap[f, expr, levelspec] should behave like Map[f, expr,
  // levelspec] (we don't actually parallelise in Woxi). With levelspec
  // 2, f is applied at levels 1 AND 2, so the squared leaves get
  // squared again by the outer mapping into the list.
  #[test]
  fn with_level_spec_integer() {
    assert_eq!(
      interpret("ParallelMap[#^2 &, {{1, 2}, {3, 4}}, 2]").unwrap(),
      "{{1, 16}, {81, 256}}"
    );
  }

  // Levelspec {1, 2} applies f at level 1 and 2 — same as 2 here.
  #[test]
  fn with_level_spec_range() {
    assert_eq!(
      interpret("ParallelMap[#^2 &, {{1, 2}, {3, 4}}, {1, 2}]").unwrap(),
      "{{1, 16}, {81, 256}}"
    );
  }

  // Named function shows the nesting clearly.
  #[test]
  fn with_level_spec_named_function() {
    assert_eq!(
      interpret("ParallelMap[f, {{a, b}, {c, d}}, 2]").unwrap(),
      "{f[{f[a], f[b]}], f[{f[c], f[d]}]}"
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

  // A single-element block specification {n} is equivalent to the integer n
  // (non-overlapping blocks), matching wolframscript.
  #[test]
  fn block_spec_single_element() {
    assert_eq!(
      interpret("BlockMap[f, Range[6], {2}]").unwrap(),
      "{f[{1, 2}], f[{3, 4}], f[{5, 6}]}"
    );
    assert_eq!(
      interpret("BlockMap[f, Range[6], {3}]").unwrap(),
      "{f[{1, 2, 3}], f[{4, 5, 6}]}"
    );
    // Leftover elements that don't fill a block are dropped.
    assert_eq!(
      interpret("BlockMap[f, Range[7], {2}]").unwrap(),
      "{f[{1, 2}], f[{3, 4}], f[{5, 6}]}"
    );
    assert_eq!(
      interpret("BlockMap[Total, Range[10], {2}]").unwrap(),
      "{3, 7, 11, 15, 19}"
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

mod slot_zero_self_reference {
  use super::*;

  #[test]
  fn factorial_via_slot_zero() {
    // Classic recursive anonymous-function idiom.
    assert_eq!(interpret("If[#1<=1, 1, #1 #0[#1-1]]& [5]").unwrap(), "120");
  }

  #[test]
  fn factorial_via_slot_zero_larger() {
    assert_eq!(
      interpret("If[#1<=1, 1, #1 #0[#1-1]]& [10]").unwrap(),
      "3628800"
    );
  }

  #[test]
  fn unnumbered_slot_self_reference() {
    assert_eq!(interpret("If[# <= 1, 1, # #0[#-1]]& [4]").unwrap(), "24");
  }

  // #0 must self-reference through the prefix-application form `#0@arg`, not
  // only the bracket form `#0[arg]`. Regression: the slot-zero substitution
  // pass did not descend into Expr::PrefixApply, so `#0` was wrongly filled
  // with the first argument (e.g. `fact[5]` produced `5*5[4]`).
  #[test]
  fn factorial_via_slot_zero_prefix_apply() {
    assert_eq!(
      interpret("If[# <= 1, 1, # * #0@(# - 1)]& [5]").unwrap(),
      "120"
    );
  }

  // Mirrors the balanced-ternary `tobt` idiom: `#0@Quotient[...]` recursion.
  #[test]
  fn slot_zero_prefix_apply_with_quotient() {
    assert_eq!(
      interpret(r#"If[# == 0, "z", #0@Quotient[#, 3]]& [10]"#).unwrap(),
      "z"
    );
  }

  // #0 self-reference must also survive inside a ReplaceAll right-hand side.
  #[test]
  fn slot_zero_inside_replace_all() {
    assert_eq!(
      interpret("If[# <= 0, 0, (# + #0[# - 1]) /. x_ -> x]& [4]").unwrap(),
      "10"
    );
  }

  // #0 self-reference through the Apply form `#0 @@ {arg}` (Expr::Apply).
  #[test]
  fn factorial_via_slot_zero_apply() {
    assert_eq!(
      interpret("If[# <= 1, 1, # * #0 @@ {# - 1}]& [5]").unwrap(),
      "120"
    );
  }

  // #0 self-reference through the Map form `#0 /@ list` (Expr::MapApply):
  // recursively totals an arbitrarily nested list.
  #[test]
  fn slot_zero_via_map_over_nested_list() {
    assert_eq!(
      interpret("If[ListQ[#], Total[#0 /@ #], #]& [{1, {2, 3}, {{4}}}]")
        .unwrap(),
      "10"
    );
  }
}

mod sub_value_assignments {
  use super::*;

  #[test]
  fn set_delayed_sub_value_returns_null() {
    // f[1][x_] := x is a SubValue assignment; return Null so the CLI shows
    // no output (matches wolframscript's surface behaviour).
    clear_state();
    assert_eq!(interpret("freshSubA[1][x_] := x").unwrap(), "\0");
  }

  #[test]
  fn set_delayed_deep_sub_value_returns_null() {
    clear_state();
    assert_eq!(interpret("freshSubB[a][b][c] := whatever").unwrap(), "\0");
  }
}

mod dynamic_head_apply {
  use super::*;

  // Regression: `If[cond, A, B] @ x` must evaluate the head before
  // applying it, so the result is `B[x]` (or `A[x]`), not the 4-arg form
  // `If[cond, A, B, x]` which would silently drop x. Same for `// f`.
  // Surfaced via primeTurn[80] in the Spiral of Prime Numbers notebook,
  // where `If[dot, AbsolutePointSize, AbsoluteThickness]@Large` lost
  // its argument and the blue line never rendered.

  #[test]
  fn prefix_apply_with_if_head() {
    assert_eq!(
      interpret(
        "FullForm[If[False, AbsolutePointSize, AbsoluteThickness]@Large]"
      )
      .unwrap(),
      "FullForm[AbsoluteThickness[Large]]"
    );
  }

  #[test]
  fn prefix_apply_with_if_head_true_branch() {
    assert_eq!(
      interpret("FullForm[If[True, Point, Line]@{{1, 2}, {3, 4}}]").unwrap(),
      "FullForm[Point[{{1, 2}, {3, 4}}]]"
    );
  }

  #[test]
  fn postfix_with_if_head() {
    assert_eq!(
      interpret("FullForm[{1, 2, 3} // If[False, Reverse, Length]]").unwrap(),
      "FullForm[3]"
    );
  }
}

mod cases {
  use super::super::case_helpers::assert_case;

  #[test]
  fn at_1() {
    assert_case(r#"a @ b"#, r#"a[b]"#);
  }
  #[test]
  fn at_2() {
    assert_case(r#"a @ b; a @ b @ c"#, r#"a[b[c]]"#);
  }
  #[test]
  fn f_1() {
    assert_case(
      r#"f[x, Sequence[a, b], y]; Attributes[Set]; a = Sequence[b, c]; a; list = {1, 2, 3}; f[Sequence @@ list]"#,
      r#"f[1, 2, 3]"#,
    );
  }
  #[test]
  fn hold_1() {
    assert_case(
      r#"f[x, Sequence[a, b], y]; Attributes[Set]; a = Sequence[b, c]; a; list = {1, 2, 3}; f[Sequence @@ list]; Hold[a, Sequence[b, c], d]"#,
      r#"Hold[a, b, c, d]"#,
    );
  }
  #[test]
  fn hold_2() {
    assert_case(
      r#"f[x, Sequence[a, b], y]; Attributes[Set]; a = Sequence[b, c]; a; list = {1, 2, 3}; f[Sequence @@ list]; Hold[a, Sequence[b, c], d]; Hold[{a, Sequence[b, c], d}]"#,
      r#"Hold[{a, Sequence[b, c], d}]"#,
    );
  }
  #[test]
  fn at_3() {
    assert_case(
      r#"\(x \^ 2\); \(x \_ 2\); \( a \+ b \% c\); \( a \& b \% c\); \( \@ 5 \)"#,
      r#"SqrtBox["5"]"#,
    );
  }
  #[test]
  fn anonymous_function_1() {
    assert_case(
      r#"\(x \^ 2\); \(x \_ 2\); \( a \+ b \% c\); \( a \& b \% c\); \( \@ 5 \); \(x \& y \)"#,
      r#"OverscriptBox["x", "y"]"#,
    );
  }
  #[test]
  fn plus_1() {
    assert_case(
      r#"\(x \^ 2\); \(x \_ 2\); \( a \+ b \% c\); \( a \& b \% c\); \( \@ 5 \); \(x \& y \); \(x \+ y \)"#,
      r#"UnderscriptBox["x", "y"]"#,
    );
  }
  #[test]
  fn power() {
    assert_case(
      r#"\(x \^ 2\); \(x \_ 2\); \( a \+ b \% c\); \( a \& b \% c\); \( \@ 5 \); \(x \& y \); \(x \+ y \); \( x \^ 2 \_ 4 \)"#,
      r#"SuperscriptBox["x", SubscriptBox["2", "4"]]"#,
    );
  }
  #[test]
  fn plus_2() {
    assert_case(
      r#"\(x \^ 2\); \(x \_ 2\); \( a \+ b \% c\); \( a \& b \% c\); \( \@ 5 \); \(x \& y \); \(x \+ y \); \( x \^ 2 \_ 4 \); (a + b)[x]"#,
      r#"(a + b)[x]"#,
    );
  }
  #[test]
  fn expr() {
    assert_case(
      r#"\(x \^ 2\); \(x \_ 2\); \( a \+ b \% c\); \( a \& b \% c\); \( \@ 5 \); \(x \& y \); \(x \+ y \); \( x \^ 2 \_ 4 \); (a + b)[x]; (a b)[x]"#,
      r#"(a*b)[x]"#,
    );
  }
  #[test]
  fn f_2() {
    assert_case(
      r#"a + b + c /. a + b -> t; a + 2 + b + c + x * y /. n_Integer + s__Symbol + rest_ -> {n, s, rest}; f[a, b, c, d] /. f[first_, rest___] -> {first, {rest}}; f[4] /. f[x_?(# > 0&)] -> x ^ 2"#,
      r#"16"#,
    );
  }
  #[test]
  fn greater_1() {
    assert_case(
      r#"a + b + c /. a + b -> t; a + 2 + b + c + x * y /. n_Integer + s__Symbol + rest_ -> {n, s, rest}; f[a, b, c, d] /. f[first_, rest___] -> {first, {rest}}; f[4] /. f[x_?(# > 0&)] -> x ^ 2; f[4] /. f[x_] /; x > 0 -> x ^ 2"#,
      r#"16"#,
    );
  }
  #[test]
  fn f_3() {
    assert_case(
      r#"a + b + c /. a + b -> t; a + 2 + b + c + x * y /. n_Integer + s__Symbol + rest_ -> {n, s, rest}; f[a, b, c, d] /. f[first_, rest___] -> {first, {rest}}; f[4] /. f[x_?(# > 0&)] -> x ^ 2; f[4] /. f[x_] /; x > 0 -> x ^ 2; f[a, b, c, d] /. f[start__, end__] -> {{start}, {end}}"#,
      r#"{{a}, {b, c, d}}"#,
    );
  }
  #[test]
  fn f_4() {
    assert_case(
      r#"a + b + c /. a + b -> t; a + 2 + b + c + x * y /. n_Integer + s__Symbol + rest_ -> {n, s, rest}; f[a, b, c, d] /. f[first_, rest___] -> {first, {rest}}; f[4] /. f[x_?(# > 0&)] -> x ^ 2; f[4] /. f[x_] /; x > 0 -> x ^ 2; f[a, b, c, d] /. f[start__, end__] -> {{start}, {end}}; f[a] /. f[x_, y_:3] -> {x, y}"#,
      r#"{a, 3}"#,
    );
  }
  #[test]
  fn equal_1() {
    assert_case(
      r#"IntegerLength[123456]; IntegerLength[10^10000]; IntegerLength[-10^1000]; IntegerLength[8, 2]; IntegerLength /@ (10 ^ Range[100]) == Range[2, 101]"#,
      r#"True"#,
    );
  }
  #[test]
  fn apply_1() {
    assert_case(
      r#"1 + 2; a + b + a; a + a + 3 * a; a + b + 4.5 + a + b + a + 2 + 1.5 b; Plus @@ {2, 4, 6}"#,
      r#"12"#,
    );
  }
  #[test]
  fn apply_2() {
    assert_case(
      r#"1 + 2; a + b + a; a + a + 3 * a; a + b + 4.5 + a + b + a + 2 + 1.5 b; Plus @@ {2, 4, 6}; Plus @@ Range[1000]"#,
      r#"500500"#,
    );
  }
  #[test]
  fn default_values_1() {
    assert_case(
      r#"1 + 2; a + b + a; a + a + 3 * a; a + b + 4.5 + a + b + a + 2 + 1.5 b; Plus @@ {2, 4, 6}; Plus @@ Range[1000]; DefaultValues[Plus]"#,
      r#"{HoldPattern[Default[Plus]] :> 0}"#,
    );
  }
  #[test]
  fn greater_2() {
    assert_case(
      r#"1 + 2; a + b + a; a + a + 3 * a; a + b + 4.5 + a + b + a + 2 + 1.5 b; Plus @@ {2, 4, 6}; Plus @@ Range[1000]; DefaultValues[Plus]; a /. n_. + x_ :> {n, x}"#,
      r#"{0, a}"#,
    );
  }
  #[test]
  fn apply_3() {
    assert_case(
      r#"10 * 2; 10 2; a * a; x ^ 10 * x ^ -2; {1, 2, 3} * 4; Times @@ {1, 2, 3, 4}"#,
      r#"24"#,
    );
  }
  #[test]
  fn integer_length() {
    assert_case(
      r#"10 * 2; 10 2; a * a; x ^ 10 * x ^ -2; {1, 2, 3} * 4; Times @@ {1, 2, 3, 4}; IntegerLength[Times@@Range[5000]]"#,
      r#"16326"#,
    );
  }
  #[test]
  fn default_values_2() {
    assert_case(
      r#"10 * 2; 10 2; a * a; x ^ 10 * x ^ -2; {1, 2, 3} * 4; Times @@ {1, 2, 3, 4}; IntegerLength[Times@@Range[5000]]; DefaultValues[Times]"#,
      r#"{HoldPattern[Default[Times]] :> 1}"#,
    );
  }
  #[test]
  fn greater_3() {
    assert_case(
      r#"10 * 2; 10 2; a * a; x ^ 10 * x ^ -2; {1, 2, 3} * 4; Times @@ {1, 2, 3, 4}; IntegerLength[Times@@Range[5000]]; DefaultValues[Times]; a /. n_. * x_ :> {n, x}"#,
      r#"{1, a}"#,
    );
  }
  #[test]
  fn operate_1() {
    assert_case(r#"Operate[p, f[a, b]]"#, r#"p[f][a, b]"#);
  }
  #[test]
  fn operate_2() {
    assert_case(
      r#"Operate[p, f[a, b]]; Operate[p, f[a, b], 1]"#,
      r#"p[f][a, b]"#,
    );
  }
  #[test]
  fn operate_3() {
    assert_case(
      r#"Operate[p, f[a, b]]; Operate[p, f[a, b], 1]; Operate[p, f[a][b][c], 0]"#,
      r#"p[f[a][b][c]]"#,
    );
  }
  #[test]
  fn through_1() {
    assert_case(r#"Through[f[g][x]]"#, r#"f[g[x]]"#);
  }
  #[test]
  fn through_2() {
    assert_case(
      r#"Through[f[g][x]]; Through[p[f, g][x]]"#,
      r#"p[f[x], g[x]]"#,
    );
  }
  #[test]
  fn apply_4() {
    assert_case(r#"f @@@ {{a, b}, {c, d}}"#, r#"{f[a, b], f[c, d]}"#);
  }
  #[test]
  fn minus() {
    assert_case(r#"InverseErf /@ {-1, 0, 1}"#, r#"{-Infinity, 0, Infinity}"#);
  }
  #[test]
  fn divide_1() {
    assert_case(r#"InverseErfc /@ {0, 1, 2}"#, r#"{Infinity, 0, -Infinity}"#);
  }
  #[test]
  fn f_5() {
    assert_case(r#"f := # ^ 2 &; f[3]"#, r#"9"#);
  }
  #[test]
  fn anonymous_function_2() {
    assert_case(r#"f := # ^ 2 &; f[3]; #^3& /@ {1, 2, 3}"#, r#"{1, 8, 27}"#);
  }
  #[test]
  fn anonymous_function_3() {
    assert_case(
      r#"f := # ^ 2 &; f[3]; #^3& /@ {1, 2, 3}; #1+#2&[4, 5]"#,
      r#"9"#,
    );
  }
  #[test]
  fn function_1() {
    assert_case(
      r#"f := # ^ 2 &; f[3]; #^3& /@ {1, 2, 3}; #1+#2&[4, 5]; Function[{x, y}, x * y][2, 3]"#,
      r#"6"#,
    );
  }
  #[test]
  fn function_2() {
    assert_case(
      r#"f := # ^ 2 &; f[3]; #^3& /@ {1, 2, 3}; #1+#2&[4, 5]; Function[{x, y}, x * y][2, 3]; Function[{x}, Function[{y}, f[x, y]]][y]"#,
      r#"Function[{y$}, f[y, y$]]"#,
    );
  }
  #[test]
  fn function_3() {
    assert_case(
      r#"f := # ^ 2 &; f[3]; #^3& /@ {1, 2, 3}; #1+#2&[4, 5]; Function[{x, y}, x * y][2, 3]; Function[{x}, Function[{y}, f[x, y]]][y]; Function[{y}, f[x, y]] /. x->y"#,
      r#"Function[{y}, f[y, y]]"#,
    );
  }
  #[test]
  fn function_4() {
    assert_case(
      r#"f := # ^ 2 &; f[3]; #^3& /@ {1, 2, 3}; #1+#2&[4, 5]; Function[{x, y}, x * y][2, 3]; Function[{x}, Function[{y}, f[x, y]]][y]; Function[{y}, f[x, y]] /. x->y; Function[y, Function[x, y^x]][x][y]"#,
      r#"x ^ y"#,
    );
  }
  #[test]
  fn function_5() {
    assert_case(
      r#"f := # ^ 2 &; f[3]; #^3& /@ {1, 2, 3}; #1+#2&[4, 5]; Function[{x, y}, x * y][2, 3]; Function[{x}, Function[{y}, f[x, y]]][y]; Function[{y}, f[x, y]] /. x->y; Function[y, Function[x, y^x]][x][y]; Function[x, Function[y, x^y]][x][y]"#,
      r#"x ^ y"#,
    );
  }
  #[test]
  fn g_1() {
    assert_case(
      r#"f := # ^ 2 &; f[3]; #^3& /@ {1, 2, 3}; #1+#2&[4, 5]; Function[{x, y}, x * y][2, 3]; Function[{x}, Function[{y}, f[x, y]]][y]; Function[{y}, f[x, y]] /. x->y; Function[y, Function[x, y^x]][x][y]; Function[x, Function[y, x^y]][x][y]; g[#] & [h[#]] & [5]"#,
      r#"g[h[5]]"#,
    );
  }
  #[test]
  fn h_1() {
    assert_case(
      r#"f := # ^ 2 &; f[3]; #^3& /@ {1, 2, 3}; #1+#2&[4, 5]; Function[{x, y}, x * y][2, 3]; Function[{x}, Function[{y}, f[x, y]]][y]; Function[{y}, f[x, y]] /. x->y; Function[y, Function[x, y^x]][x][y]; Function[x, Function[y, x^y]][x][y]; g[#] & [h[#]] & [5]; h := Function[{x}, Hold[1+x]]; h[1 + 1]"#,
      r#"Hold[1 + 2]"#,
    );
  }
  #[test]
  fn h_2() {
    assert_case(
      r#"f := # ^ 2 &; f[3]; #^3& /@ {1, 2, 3}; #1+#2&[4, 5]; Function[{x, y}, x * y][2, 3]; Function[{x}, Function[{y}, f[x, y]]][y]; Function[{y}, f[x, y]] /. x->y; Function[y, Function[x, y^x]][x][y]; Function[x, Function[y, x^y]][x][y]; g[#] & [h[#]] & [5]; h := Function[{x}, Hold[1+x]]; h[1 + 1]; h:= Function[{x}, Hold[1+x], HoldAll]; h[1+1]"#,
      r#"Hold[1 + (1 + 1)]"#,
    );
  }
  #[test]
  fn slot() {
    assert_case(r#"#"#, r#"#1"#);
  }
  #[test]
  fn anonymous_function_4() {
    assert_case(r#"#; {#1, #2, #3}&[1, 2, 3, 4, 5]"#, r#"{1, 2, 3}"#);
  }
  #[test]
  fn composition_1() {
    assert_case(r#"Composition[f, g][x]"#, r#"f[g[x]]"#);
  }
  #[test]
  fn composition_2() {
    assert_case(
      r#"Composition[f, g][x]; Composition[f, g, h][x, y, z]"#,
      r#"f[g[h[x, y, z]]]"#,
    );
  }
  #[test]
  fn composition_3() {
    assert_case(
      r#"Composition[f, g][x]; Composition[f, g, h][x, y, z]; Composition[]"#,
      r#"Identity"#,
    );
  }
  #[test]
  fn composition_4() {
    assert_case(
      r#"Composition[f, g][x]; Composition[f, g, h][x, y, z]; Composition[]; Composition[][x]"#,
      r#"x"#,
    );
  }
  #[test]
  fn attributes() {
    assert_case(
      r#"Composition[f, g][x]; Composition[f, g, h][x, y, z]; Composition[]; Composition[][x]; Attributes[Composition]"#,
      r#"{Flat, OneIdentity, Protected}"#,
    );
  }
  #[test]
  fn composition_5() {
    assert_case(
      r#"Composition[f, g][x]; Composition[f, g, h][x, y, z]; Composition[]; Composition[][x]; Attributes[Composition]; Composition[f, Composition[g, h]]"#,
      r#"f @* g @* h"#,
    );
  }
  #[test]
  fn identity() {
    assert_case(r#"Identity[x]"#, r#"x"#);
  }
  #[test]
  fn apply_5() {
    assert_case(r#"f @@ {1, 2, 3}"#, r#"f[1, 2, 3]"#);
  }
  #[test]
  fn apply_6() {
    assert_case(r#"f @@ {1, 2, 3}; Plus @@ {1, 2, 3}"#, r#"6"#);
  }
  #[test]
  fn plus_3() {
    assert_case(
      r#"f @@ {1, 2, 3}; Plus @@ {1, 2, 3}; f @@ (a + b + c)"#,
      r#"f[a, b, c]"#,
    );
  }
  #[test]
  fn apply_7() {
    assert_case(
      r#"f @@ {1, 2, 3}; Plus @@ {1, 2, 3}; f @@ (a + b + c); Apply[f][a + b + c]"#,
      r#"f[a, b, c]"#,
    );
  }
  #[test]
  fn apply_8() {
    assert_case(
      r#"f @@ {1, 2, 3}; Plus @@ {1, 2, 3}; f @@ (a + b + c); Apply[f][a + b + c]; Apply[f, {a + b, g[c, d, e * f], 3}, {1}]"#,
      r#"{f[a, b], f[c, d, e*f], 3}"#,
    );
  }
  #[test]
  fn apply_9() {
    assert_case(
      r#"f @@ {1, 2, 3}; Plus @@ {1, 2, 3}; f @@ (a + b + c); Apply[f][a + b + c]; Apply[f, {a + b, g[c, d, e * f], 3}, {1}]; Apply[f, {a, b, c}, {0}]"#,
      r#"f[a, b, c]"#,
    );
  }
  #[test]
  fn apply_10() {
    assert_case(
      r#"f @@ {1, 2, 3}; Plus @@ {1, 2, 3}; f @@ (a + b + c); Apply[f][a + b + c]; Apply[f, {a + b, g[c, d, e * f], 3}, {1}]; Apply[f, {a, b, c}, {0}]; Apply[f, {{{{{a}}}}}, {2, -3}]"#,
      r#"{{f[f[{a}]]}}"#,
    );
  }
  #[test]
  fn apply_11() {
    assert_case(
      r#"f @@ {1, 2, 3}; Plus @@ {1, 2, 3}; f @@ (a + b + c); Apply[f][a + b + c]; Apply[f, {a + b, g[c, d, e * f], 3}, {1}]; Apply[f, {a, b, c}, {0}]; Apply[f, {{{{{a}}}}}, {2, -3}]; Apply[List, a + b * c ^ e * f[g], {0, Infinity}]"#,
      r#"{a, {b, {c, e}, {g}}}"#,
    );
  }
  #[test]
  fn divide_2() {
    assert_case(r#"f /@ {1, 2, 3}"#, r#"{f[1], f[2], f[3]}"#);
  }
  #[test]
  fn anonymous_function_5() {
    assert_case(
      r#"f /@ {1, 2, 3}; #^2& /@ {1, 2, 3, 4}"#,
      r#"{1, 4, 9, 16}"#,
    );
  }
  #[test]
  fn map_1() {
    assert_case(
      r#"f /@ {1, 2, 3}; #^2& /@ {1, 2, 3, 4}; Map[f, {{a, b}, {c, d, e}}, {2}]"#,
      r#"{{f[a], f[b]}, {f[c], f[d], f[e]}}"#,
    );
  }
  #[test]
  fn map_2() {
    assert_case(
      r#"f /@ {1, 2, 3}; #^2& /@ {1, 2, 3, 4}; Map[f, {{a, b}, {c, d, e}}, {2}]; Map[f, <|"a" -> 1, "b" -> 2, "c" -> 3, "d" -> 4|>]"#,
      r#"<|"a" -> f[1], "b" -> f[2], "c" -> f[3], "d" -> f[4]|>"#,
    );
  }
  #[test]
  fn map_3() {
    assert_case(
      r#"f /@ {1, 2, 3}; #^2& /@ {1, 2, 3, 4}; Map[f, {{a, b}, {c, d, e}}, {2}]; Map[f, <|"a" -> 1, "b" -> 2, "c" -> 3, "d" -> 4|>]; Map[f, a + b + c, Heads->True]"#,
      r#"f[Plus][f[a], f[b], f[c]]"#,
    );
  }
  #[test]
  fn map_4() {
    assert_case(
      r#"f /@ {1, 2, 3}; #^2& /@ {1, 2, 3, 4}; Map[f, {{a, b}, {c, d, e}}, {2}]; Map[f, <|"a" -> 1, "b" -> 2, "c" -> 3, "d" -> 4|>]; Map[f, a + b + c, Heads->True]; Map[f][{a, b, c}]"#,
      r#"{f[a], f[b], f[c]}"#,
    );
  }
  #[test]
  fn map_at_1() {
    assert_case(r#"MapAt[f, {a, b, c}, 2]"#, r#"{a, f[b], c}"#);
  }
  #[test]
  fn map_at_2() {
    assert_case(
      r#"MapAt[f, {a, b, c}, 2]; MapAt[0&, {{1, 1}, {1, 1}}, {2, 1}]"#,
      r#"{{1, 1}, {0, 1}}"#,
    );
  }
  #[test]
  fn map_at_3() {
    assert_case(
      r#"MapAt[f, {a, b, c}, 2]; MapAt[0&, {{1, 1}, {1, 1}}, {2, 1}]; MapAt[0&, {{0, 1}, {1, 0}}, 2]"#,
      r#"{{0, 1}, 0}"#,
    );
  }
  #[test]
  fn map_at_4() {
    assert_case(
      r#"MapAt[f, {a, b, c}, 2]; MapAt[0&, {{1, 1}, {1, 1}}, {2, 1}]; MapAt[0&, {{0, 1}, {1, 0}}, 2]; MapAt[0&, {{0, 1}, {1, 0}}, {{2}, {1}}]"#,
      r#"{0, 0}"#,
    );
  }
  #[test]
  fn map_at_5() {
    assert_case(
      r#"MapAt[f, {a, b, c}, 2]; MapAt[0&, {{1, 1}, {1, 1}}, {2, 1}]; MapAt[0&, {{0, 1}, {1, 0}}, 2]; MapAt[0&, {{0, 1}, {1, 0}}, {{2}, {1}}]; MapAt[f, {a, b, c}, -1]"#,
      r#"{a, b, f[c]}"#,
    );
  }
  #[test]
  fn map_at_6() {
    assert_case(
      r#"MapAt[f, {a, b, c}, 2]; MapAt[0&, {{1, 1}, {1, 1}}, {2, 1}]; MapAt[0&, {{0, 1}, {1, 0}}, 2]; MapAt[0&, {{0, 1}, {1, 0}}, {{2}, {1}}]; MapAt[f, {a, b, c}, -1]; MapAt[f, -1][{a, b, c}]"#,
      r#"{a, b, f[c]}"#,
    );
  }
  #[test]
  fn map_at_7() {
    assert_case(
      r#"MapAt[f, {a, b, c}, 2]; MapAt[0&, {{1, 1}, {1, 1}}, {2, 1}]; MapAt[0&, {{0, 1}, {1, 0}}, 2]; MapAt[0&, {{0, 1}, {1, 0}}, {{2}, {1}}]; MapAt[f, {a, b, c}, -1]; MapAt[f, -1][{a, b, c}]; MapAt[f, <|"a" -> 1, "b" -> 2, "c" -> 3, "d" -> 4|>, 2]"#,
      r#"<|"a" -> 1, "b" -> f[2], "c" -> 3, "d" -> 4|>"#,
    );
  }
  #[test]
  fn map_at_8() {
    assert_case(
      r#"MapAt[f, {a, b, c}, 2]; MapAt[0&, {{1, 1}, {1, 1}}, {2, 1}]; MapAt[0&, {{0, 1}, {1, 0}}, 2]; MapAt[0&, {{0, 1}, {1, 0}}, {{2}, {1}}]; MapAt[f, {a, b, c}, -1]; MapAt[f, -1][{a, b, c}]; MapAt[f, <|"a" -> 1, "b" -> 2, "c" -> 3, "d" -> 4|>, 2]; MapAt[f, <|"a" -> 1, "b" -> 2, "c" -> 3, "d" -> 4|>, -2]"#,
      r#"<|"a" -> 1, "b" -> 2, "c" -> f[3], "d" -> 4|>"#,
    );
  }
  #[test]
  fn map_indexed_1() {
    assert_case(
      r#"MapIndexed[f, {a, b, c}]"#,
      r#"{f[a, {1}], f[b, {2}], f[c, {3}]}"#,
    );
  }
  #[test]
  fn map_indexed_2() {
    assert_case(
      r#"MapIndexed[f, {a, b, c}]; MapIndexed[f][{a, b, c}]"#,
      r#"{f[a, {1}], f[b, {2}], f[c, {3}]}"#,
    );
  }
  #[test]
  fn map_indexed_3() {
    assert_case(
      r#"MapIndexed[f, {a, b, c}]; MapIndexed[f][{a, b, c}]; MapIndexed[f, {a, b, c}, Heads->True]"#,
      r#"f[List, {0}][f[a, {1}], f[b, {2}], f[c, {3}]]"#,
    );
  }
  #[test]
  fn map_indexed_4() {
    assert_case(
      r#"MapIndexed[f, {a, b, c}]; MapIndexed[f][{a, b, c}]; MapIndexed[f, {a, b, c}, Heads->True]; MapIndexed[f, a + b + c * d, {0, 1}]"#,
      r#"f[f[a, {1}] + f[b, {2}] + f[c*d, {3}], {}]"#,
    );
  }
  #[test]
  fn map_indexed_5() {
    assert_case(
      r#"MapIndexed[f, {a, b, c}]; MapIndexed[f][{a, b, c}]; MapIndexed[f, {a, b, c}, Heads->True]; MapIndexed[f, a + b + c * d, {0, 1}]; expr = a + b * f[g] * c ^ e; listified = Apply[List, expr, {0, Infinity}]; MapIndexed[#2 &, listified, {-1}]"#,
      r#"{{1}, {{2, 1}, {{2, 2, 1}, {2, 2, 2}}, {{2, 3, 1}}}}"#,
    );
  }
  #[test]
  fn map_indexed_6() {
    assert_case(
      r#"MapIndexed[f, {a, b, c}]; MapIndexed[f][{a, b, c}]; MapIndexed[f, {a, b, c}, Heads->True]; MapIndexed[f, a + b + c * d, {0, 1}]; expr = a + b * f[g] * c ^ e; listified = Apply[List, expr, {0, Infinity}]; MapIndexed[#2 &, listified, {-1}]; MapIndexed[#2 &, listified, {-1}, Heads -> True]"#,
      r#"{0}[{1}, {2, 0}[{2, 1}, {2, 2, 0}[{2, 2, 1}, {2, 2, 2}], {2, 3, 0}[{2, 3, 1}]]]"#,
    );
  }
  #[test]
  fn map_indexed_7() {
    assert_case(
      r#"MapIndexed[f, {a, b, c}]; MapIndexed[f][{a, b, c}]; MapIndexed[f, {a, b, c}, Heads->True]; MapIndexed[f, a + b + c * d, {0, 1}]; expr = a + b * f[g] * c ^ e; listified = Apply[List, expr, {0, Infinity}]; MapIndexed[#2 &, listified, {-1}]; MapIndexed[#2 &, listified, {-1}, Heads -> True]; MapIndexed[Extract[expr, #2] &, listified, {-1}, Heads -> True]"#,
      r#"a + b*c^e*f[g]"#,
    );
  }
  #[test]
  fn map_thread_1() {
    assert_case(
      r#"MapThread[f, {{a, b, c}, {1, 2, 3}}]"#,
      r#"{f[a, 1], f[b, 2], f[c, 3]}"#,
    );
  }
  #[test]
  fn map_thread_2() {
    assert_case(
      r#"MapThread[f, {{a, b, c}, {1, 2, 3}}]; MapThread[f, {{{a, b}, {c, d}}, {{e, f}, {g, h}}}, 2]"#,
      r#"{{f[a, e], f[b, f]}, {f[c, g], f[d, h]}}"#,
    );
  }
  #[test]
  fn map_thread_3() {
    assert_case(
      r#"MapThread[f, {{a, b, c}, {1, 2, 3}}]; MapThread[f, {{{a, b}, {c, d}}, {{e, f}, {g, h}}}, 2]; MapThread[f][{{a, b, c}, {1, 2, 3}}]"#,
      r#"{f[a, 1], f[b, 2], f[c, 3]}"#,
    );
  }
  #[test]
  fn thread_1() {
    assert_case(r#"Thread[f[{a, b, c}]]"#, r#"{f[a], f[b], f[c]}"#);
  }
  #[test]
  fn thread_2() {
    assert_case(
      r#"Thread[f[{a, b, c}]]; Thread[f[{a, b, c}, t]]"#,
      r#"{f[a, t], f[b, t], f[c, t]}"#,
    );
  }
  #[test]
  fn thread_3() {
    assert_case(
      r#"Thread[f[{a, b, c}]]; Thread[f[{a, b, c}, t]]; Thread[f[a + b + c], Plus]"#,
      r#"f[a] + f[b] + f[c]"#,
    );
  }
  #[test]
  fn list_literal() {
    assert_case(
      r#"Thread[f[{a, b, c}]]; Thread[f[{a, b, c}, t]]; Thread[f[a + b + c], Plus]; {a, b, c} + {d, e, f} + g"#,
      r#"{a + d + g, b + e + g, c + f + g}"#,
    );
  }
  #[test]
  fn apply_12() {
    assert_case(r#"Plus@@uniformTable"#, r#"uniformTable"#);
  }
  #[test]
  fn divide_3() {
    assert_case(
      r#"N[Sqrt[2], 41]//Precision; N[Sqrt[2], 40]//Precision; N[Sqrt[2], 41]//Precision; N[Sqrt[2], 40]//Precision; N[Sqrt[2], 41]; Precision/@Table[N[Pi,p],{p, {5, 100, MachinePrecision, 20}}]"#,
      r#"{5., 100., MachinePrecision, 20.}"#,
    );
  }
  #[test]
  fn divide_4() {
    assert_case(
      r#"N[Sqrt[2], 41]//Precision; N[Sqrt[2], 40]//Precision; N[Sqrt[2], 41]//Precision; N[Sqrt[2], 40]//Precision; N[Sqrt[2], 41]; Precision/@Table[N[Pi,p],{p, {5, 100, MachinePrecision, 20}}]; Precision/@Table[N[Sin[1],p],{p, {5, 100, MachinePrecision, 20}}]"#,
      r#"{5., 100., MachinePrecision, 20.}"#,
    );
  }
  #[test]
  fn n_1() {
    assert_case(
      r#"N[Sqrt[2], 41]//Precision; N[Sqrt[2], 40]//Precision; N[Sqrt[2], 41]//Precision; N[Sqrt[2], 40]//Precision; N[Sqrt[2], 41]; Precision/@Table[N[Pi,p],{p, {5, 100, MachinePrecision, 20}}]; Precision/@Table[N[Sin[1],p],{p, {5, 100, MachinePrecision, 20}}]; N[Sqrt[2], 40]"#,
      r#"1.4142135623730950488016887242096980785696718753769480731767`40."#,
    );
  }
  #[test]
  fn n_2() {
    assert_case(
      r#"N[Sqrt[2], 41]//Precision; N[Sqrt[2], 40]//Precision; N[Sqrt[2], 41]//Precision; N[Sqrt[2], 40]//Precision; N[Sqrt[2], 41]; Precision/@Table[N[Pi,p],{p, {5, 100, MachinePrecision, 20}}]; Precision/@Table[N[Sin[1],p],{p, {5, 100, MachinePrecision, 20}}]; N[Sqrt[2], 40]; N[Sqrt[2], 4]"#,
      r#"1.4142135623730950488`4."#,
    );
  }
  #[test]
  fn n_3() {
    assert_case(
      r#"N[Sqrt[2], 41]//Precision; N[Sqrt[2], 40]//Precision; N[Sqrt[2], 41]//Precision; N[Sqrt[2], 40]//Precision; N[Sqrt[2], 41]; Precision/@Table[N[Pi,p],{p, {5, 100, MachinePrecision, 20}}]; Precision/@Table[N[Sin[1],p],{p, {5, 100, MachinePrecision, 20}}]; N[Sqrt[2], 40]; N[Sqrt[2], 4]; N[Pi, 40]"#,
      r#"3.1415926535897932384626433832795028841971693993751058209749`40."#,
    );
  }
  #[test]
  fn n_4() {
    assert_case(
      r#"N[Sqrt[2], 41]//Precision; N[Sqrt[2], 40]//Precision; N[Sqrt[2], 41]//Precision; N[Sqrt[2], 40]//Precision; N[Sqrt[2], 41]; Precision/@Table[N[Pi,p],{p, {5, 100, MachinePrecision, 20}}]; Precision/@Table[N[Sin[1],p],{p, {5, 100, MachinePrecision, 20}}]; N[Sqrt[2], 40]; N[Sqrt[2], 4]; N[Pi, 40]; N[Pi, 4]"#,
      r#"3.1415926535897932385`4."#,
    );
  }
  #[test]
  fn n_5() {
    assert_case(
      r#"N[Sqrt[2], 41]//Precision; N[Sqrt[2], 40]//Precision; N[Sqrt[2], 41]//Precision; N[Sqrt[2], 40]//Precision; N[Sqrt[2], 41]; Precision/@Table[N[Pi,p],{p, {5, 100, MachinePrecision, 20}}]; Precision/@Table[N[Sin[1],p],{p, {5, 100, MachinePrecision, 20}}]; N[Sqrt[2], 40]; N[Sqrt[2], 4]; N[Pi, 40]; N[Pi, 4]; N[Pi, 41]"#,
      r#"3.1415926535897932384626433832795028841971693993751058209749`41."#,
    );
  }
  #[test]
  fn n_6() {
    assert_case(
      r#"N[Sqrt[2], 41]//Precision; N[Sqrt[2], 40]//Precision; N[Sqrt[2], 41]//Precision; N[Sqrt[2], 40]//Precision; N[Sqrt[2], 41]; Precision/@Table[N[Pi,p],{p, {5, 100, MachinePrecision, 20}}]; Precision/@Table[N[Sin[1],p],{p, {5, 100, MachinePrecision, 20}}]; N[Sqrt[2], 40]; N[Sqrt[2], 4]; N[Pi, 40]; N[Pi, 4]; N[Pi, 41]; N[Sqrt[2], 41]"#,
      r#"1.4142135623730950488016887242096980785696718753769480731767`41."#,
    );
  }
  #[test]
  fn equal_2() {
    assert_case(
      r#"IntegerLength /@ (10 ^ Range[100] - 1) == Range[1, 100]"#,
      r#"True"#,
    );
  }
  #[test]
  fn map_5() {
    assert_case(
      r#"Map[F, Q[a->1, b:>Association[p->3,q->4]]]"#,
      r#"Q[F[a->1], F[b:>Association[p->3, q->4]]]"#,
    );
  }
  #[test]
  fn map_6() {
    assert_case(
      r#"Map[F, Q[a->1, b:>Association[p->3,q->4]]]; Map[F, Q[a->1, b:>Association[p->3,q->4]],{2}]"#,
      r#"Q[F[a]->F[1], F[b]:>F[Association[p->3, q->4]]]"#,
    );
  }
  #[test]
  fn map_7() {
    assert_case(
      r#"Map[F, Q[a->1, b:>Association[p->3,q->4]]]; Map[F, Q[a->1, b:>Association[p->3,q->4]],{2}]; Map[F, Association[a->1, b:>Association[p->3,q->4]], {0}]"#,
      r#"F[<|a -> 1, b :> Association[p -> 3, q -> 4]|>]"#,
    );
  }
  #[test]
  fn map_8() {
    assert_case(
      r#"Map[F, Q[a->1, b:>Association[p->3,q->4]]]; Map[F, Q[a->1, b:>Association[p->3,q->4]],{2}]; Map[F, Association[a->1, b:>Association[p->3,q->4]], {0}]; Map[F, Association[a->1, b:>2]]"#,
      r#"<|a -> F[1], b :> F[2]|>"#,
    );
  }
  #[test]
  fn map_9() {
    assert_case(
      r#"Map[F, Q[a->1, b:>Association[p->3,q->4]]]; Map[F, Q[a->1, b:>Association[p->3,q->4]],{2}]; Map[F, Association[a->1, b:>Association[p->3,q->4]], {0}]; Map[F, Association[a->1, b:>2]]; Map[F, Association[a->1, b:>Association[p->3,q->4]]]"#,
      r#"<|a -> F[1], b :> F[Association[p -> 3, q -> 4]]|>"#,
    );
  }
  #[test]
  fn map_10() {
    assert_case(
      r#"Map[F, Q[a->1, b:>Association[p->3,q->4]]]; Map[F, Q[a->1, b:>Association[p->3,q->4]],{2}]; Map[F, Association[a->1, b:>Association[p->3,q->4]], {0}]; Map[F, Association[a->1, b:>2]]; Map[F, Association[a->1, b:>Association[p->3,q->4]]]; Map[F, Association[a->1, b:>Association[p->3,q->4]], {1}]"#,
      r#"<|a -> F[1], b :> F[Association[p -> 3, q -> 4]]|>"#,
    );
  }
  #[test]
  fn map_11() {
    assert_case(
      r#"Map[F, Q[a->1, b:>Association[p->3,q->4]]]; Map[F, Q[a->1, b:>Association[p->3,q->4]],{2}]; Map[F, Association[a->1, b:>Association[p->3,q->4]], {0}]; Map[F, Association[a->1, b:>2]]; Map[F, Association[a->1, b:>Association[p->3,q->4]]]; Map[F, Association[a->1, b:>Association[p->3,q->4]], {1}]; Map[F, Association[a->1,b:>2,q]]"#,
      r#"Association[F[a->1], F[b:>2], F[q]]"#,
    );
  }
  #[test]
  fn map_12() {
    assert_case(
      r#"Map[F, Q[a->1, b:>Association[p->3,q->4]]]; Map[F, Q[a->1, b:>Association[p->3,q->4]],{2}]; Map[F, Association[a->1, b:>Association[p->3,q->4]], {0}]; Map[F, Association[a->1, b:>2]]; Map[F, Association[a->1, b:>Association[p->3,q->4]]]; Map[F, Association[a->1, b:>Association[p->3,q->4]], {1}]; Map[F, Association[a->1,b:>2,q]]; Map[F, Association[a->1, b:>Association[p->3,q->4]], {2}]"#,
      r#"<|a -> 1, b :> Association[F[p -> 3], F[q -> 4]]|>"#,
    );
  }
  #[test]
  fn map_13() {
    assert_case(
      r#"Map[F, Q[a->1, b:>Association[p->3,q->4]]]; Map[F, Q[a->1, b:>Association[p->3,q->4]],{2}]; Map[F, Association[a->1, b:>Association[p->3,q->4]], {0}]; Map[F, Association[a->1, b:>2]]; Map[F, Association[a->1, b:>Association[p->3,q->4]]]; Map[F, Association[a->1, b:>Association[p->3,q->4]], {1}]; Map[F, Association[a->1,b:>2,q]]; Map[F, Association[a->1, b:>Association[p->3,q->4]], {2}]; Map[F, Association[a->1, b:>Q[p->3, q->4]], {2}]"#,
      r#"<|a -> 1, b :> Q[F[p -> 3], F[q -> 4]]|>"#,
    );
  }
  #[test]
  fn g_2() {
    assert_case(
      r#"g[x_,y_] := x+y;g[Sequence@@Slot/@Range[2]]&[1,2]"#,
      r#"#1 + #2"#,
    );
  }
  #[test]
  fn evaluate() {
    assert_case(
      r#"g[x_,y_] := x+y;g[Sequence@@Slot/@Range[2]]&[1,2]; Evaluate[g[Sequence@@Slot/@Range[2]]]&[1,2]"#,
      r#"3"#,
    );
  }
  #[test]
  fn divide_5() {
    assert_case(
      r#"g[x_,y_] := x+y;g[Sequence@@Slot/@Range[2]]&[1,2]; Evaluate[g[Sequence@@Slot/@Range[2]]]&[1,2]; # // InputForm"#,
      r#"InputForm[#1]"#,
    );
  }
  #[test]
  fn divide_6() {
    assert_case(
      r#"g[x_,y_] := x+y;g[Sequence@@Slot/@Range[2]]&[1,2]; Evaluate[g[Sequence@@Slot/@Range[2]]]&[1,2]; # // InputForm; #0 // InputForm"#,
      r#"InputForm[#0]"#,
    );
  }
  #[test]
  fn divide_7() {
    assert_case(
      r#"g[x_,y_] := x+y;g[Sequence@@Slot/@Range[2]]&[1,2]; Evaluate[g[Sequence@@Slot/@Range[2]]]&[1,2]; # // InputForm; #0 // InputForm; ## // InputForm"#,
      r#"InputForm[##1]"#,
    );
  }
}

mod named_slots {
  use super::*;

  #[test]
  fn named_slot_fills_from_association() {
    assert_eq!(interpret(r#"(#name &)[<|"name" -> 7|>]"#).unwrap(), "7");
    assert_eq!(
      interpret(r#"(#a + #b &)[<|"a" -> 1, "b" -> 2|>]"#).unwrap(),
      "3"
    );
  }

  #[test]
  fn named_slot_alphanumeric_key() {
    assert_eq!(interpret(r#"(#abc1 &)[<|"abc1" -> 9|>]"#).unwrap(), "9");
  }

  #[test]
  fn named_slot_extra_arguments_ignored() {
    assert_eq!(interpret(r#"(#a &)[<|"a" -> 9|>, 99]"#).unwrap(), "9");
  }

  #[test]
  fn named_slot_in_operator_forms() {
    assert_eq!(
      interpret(r#"Select[#a > 1 &][{<|"a" -> 1|>, <|"a" -> 3|>}]"#).unwrap(),
      "{<|a -> 3|>}"
    );
    assert_eq!(
      interpret(r#"Map[#x &, {<|"x" -> 1|>, <|"x" -> 2|>}]"#).unwrap(),
      "{1, 2}"
    );
  }

  #[test]
  fn named_slot_display_form() {
    // #a & round-trips; bare Slot["a"] also displays as #a
    assert_eq!(interpret("#a &").unwrap(), "#a & ");
    assert_eq!(interpret(r#"Slot["a"]"#).unwrap(), "#a");
    assert_eq!(interpret("#abc + #2 &").unwrap(), "#abc + #2 & ");
  }

  #[test]
  fn named_slot_via_stored_function() {
    assert_eq!(
      interpret(r#"f = #total &; f[<|"total" -> 42|>]"#).unwrap(),
      "42"
    );
  }

  #[test]
  fn named_slot_missing_key_emits_slota() {
    // The filled slots substitute; missing ones stay as #key
    assert_eq!(interpret(r#"(#a + #b &)[<|"a" -> 1|>]"#).unwrap(), "1 + #b");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "Function::slota: Named slot b in #a + #b &  cannot be filled from <|a -> 1|>."
      )),
      "expected Function::slota message, got {:?}",
      msgs
    );
  }

  #[test]
  fn named_slot_non_association_emits_slot1() {
    assert_eq!(interpret("(#a &)[5]").unwrap(), "#a");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "Function::slot1: (#a & )[5] is expected to have an Association as the first argument."
      )),
      "expected Function::slot1 message, got {:?}",
      msgs
    );
  }

  #[test]
  fn named_slot_no_arguments_emits_slot1() {
    assert_eq!(interpret("(#a &)[]").unwrap(), "#a");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "Function::slot1: (#a & )[] is expected to have an Association as the first argument."
      )),
      "expected Function::slot1 message, got {:?}",
      msgs
    );
  }

  #[test]
  fn named_slot_in_map_with_missing_key() {
    assert_eq!(
      interpret(r#"Map[#a &, {<|"a" -> 1|>, <|"b" -> 2|>}]"#).unwrap(),
      "{1, #a}"
    );
  }
}

mod part_of_atoms {
  use super::*;

  #[test]
  fn part_of_number_parses_and_warns() {
    // 5[["a"]] must parse (number base) and emit Part::pspec1
    assert_eq!(interpret(r#"5[["a"]]"#).unwrap(), r#"5[[a]]"#);
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs
        .iter()
        .any(|m| m
          .contains("Part::pspec1: Part specification a is not applicable.")),
      "expected Part::pspec1 message, got {:?}",
      msgs
    );
  }

  #[test]
  fn string_part_spec_on_list_emits_pspec1() {
    assert_eq!(interpret(r#"{1, 2}[["a"]]"#).unwrap(), "{1, 2}[[a]]");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs
        .iter()
        .any(|m| m
          .contains("Part::pspec1: Part specification a is not applicable.")),
      "expected Part::pspec1 message, got {:?}",
      msgs
    );
  }

  #[test]
  fn string_part_spec_on_symbol_emits_pspec1() {
    // Multiple string indices: the message names the first one
    assert_eq!(interpret(r#"x[["a", "b"]]"#).unwrap(), "x[[a,b]]");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs
        .iter()
        .any(|m| m
          .contains("Part::pspec1: Part specification a is not applicable.")),
      "expected Part::pspec1 message, got {:?}",
      msgs
    );
  }

  #[test]
  fn part_deeper_than_object_emits_partd() {
    // Multi-index spec that hits an atom mid-way: partd with the
    // evaluated base displayed in OutputForm (strings unquoted)
    assert_eq!(
      interpret(r#"{{1, 2}, {3, 4}}[[1, 2, 3]]"#).unwrap(),
      "{{1, 2}, {3, 4}}[[1,2,3]]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "Part::partd: Part specification {{1, 2}, {3, 4}}[[1,2,3]] is longer than depth of object."
      )),
      "expected Part::partd message, got {:?}",
      msgs
    );
  }

  #[test]
  fn part_of_string_literal_parses() {
    // String bases parse: "Alice"[[2]] stays unevaluated with partd
    assert_eq!(interpret(r#""Alice"[[2]]"#).unwrap(), "Alice[[2]]");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "Part::partd: Part specification Alice[[2]] is longer than depth of object."
      )),
      "expected Part::partd message, got {:?}",
      msgs
    );
  }

  #[test]
  fn part_of_number_literal_parses() {
    assert_eq!(interpret("5[[1]]").unwrap(), "5[[1]]");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "Part::partd: Part specification 5[[1]] is longer than depth of object."
      )),
      "expected Part::partd message, got {:?}",
      msgs
    );
  }
}

mod curry {
  use super::*;

  // Bare `Curry[f]` builds a two-argument operator that applies its
  // arguments in REVERSED order: `Curry[f][a][b]` is `f[b, a]`.
  #[test]
  fn bare_curry_reverses_two_args() {
    assert_eq!(interpret("Curry[f][a][b]").unwrap(), "f[b, a]");
    // Both arguments supplied in a single application are still reversed.
    assert_eq!(interpret("Curry[f][a, b]").unwrap(), "f[b, a]");
  }

  #[test]
  fn bare_curry_partial_application_keeps_head() {
    // One argument is not enough; the operator keeps accumulating.
    assert_eq!(interpret("Curry[f][a]").unwrap(), "Curry[f][a]");
    assert_eq!(interpret("Head[Curry[f][a]]").unwrap(), "Curry[f]");
  }

  #[test]
  fn bare_curry_extra_args_apply_to_result() {
    // After collecting two args, further applications apply to the result.
    assert_eq!(interpret("Curry[f][a][b][c]").unwrap(), "f[b, a][c]");
  }

  #[test]
  fn bare_curry_on_builtins() {
    // Curry[Power][2][3] -> Power[3, 2] -> 9 (reversed).
    assert_eq!(interpret("Curry[Power][2][3]").unwrap(), "9");
    // Curry[Subtract][3][10] -> Subtract[10, 3] -> 7.
    assert_eq!(interpret("Curry[Subtract][3][10]").unwrap(), "7");
  }

  // `Curry[f, n]` collects `n` arguments in order across applications.
  #[test]
  fn curry_n_collects_in_order() {
    assert_eq!(interpret("Curry[f, 2][a][b]").unwrap(), "f[a, b]");
    assert_eq!(interpret("Curry[f, 3][a][b][c]").unwrap(), "f[a, b, c]");
    assert_eq!(
      interpret("Curry[f, 4][a][b][c][d]").unwrap(),
      "f[a, b, c, d]"
    );
    assert_eq!(interpret("Curry[Plus, 3][1][2][3]").unwrap(), "6");
  }

  #[test]
  fn curry_n_one_is_immediate() {
    assert_eq!(interpret("Curry[f, 1][a]").unwrap(), "f[a]");
    // With n satisfied, the next application applies to the result.
    assert_eq!(interpret("Curry[f, 1][a][b]").unwrap(), "f[a][b]");
  }

  #[test]
  fn curry_n_multiple_args_per_application() {
    // Reaching the count in one application finishes immediately.
    assert_eq!(interpret("Curry[f, 2][a, b][c]").unwrap(), "f[a, b][c]");
    assert_eq!(interpret("Curry[f, 3][a, b][c]").unwrap(), "f[a, b, c]");
    assert_eq!(interpret("Curry[f, 3][a][b, c]").unwrap(), "f[a, b, c]");
  }

  #[test]
  fn curry_n_overshoot_leftover_applies_to_result() {
    // Collecting more than n in one application leaves the surplus to apply
    // to the produced result.
    assert_eq!(
      interpret("Curry[f, 3][a, b][c, d]").unwrap(),
      "f[a, b, c][d]"
    );
    assert_eq!(
      interpret("Curry[f, 2][a][b][c][d]").unwrap(),
      "f[a, b][c][d]"
    );
  }

  // `Curry[f, {p1, …}]` arranges the collected args by an explicit
  // permutation: output position i receives the perm[i]-th supplied arg.
  #[test]
  fn curry_with_permutation_spec() {
    assert_eq!(interpret("Curry[f, {1, 2}][a][b]").unwrap(), "f[a, b]");
    assert_eq!(interpret("Curry[f, {2, 1}][a][b]").unwrap(), "f[b, a]");
    assert_eq!(
      interpret("Curry[f, {3, 1, 2}][a][b][c]").unwrap(),
      "f[c, a, b]"
    );
  }

  // A partially-applied curry operator used as a mapping function must fire
  // once its final argument arrives. Regression: it previously stayed
  // unevaluated (`{Curry[Power][2][1], …}`).
  #[test]
  fn partial_curry_as_map_function() {
    assert_eq!(
      interpret("Map[Curry[Power][2], {1, 2, 3}]").unwrap(),
      "{1, 4, 9}"
    );
    // A plain (non-curry) curried head still accumulates symbolically.
    assert_eq!(
      interpret("Map[g[a][b], {1, 2}]").unwrap(),
      "{g[a][b][1], g[a][b][2]}"
    );
  }
}

mod operator_applied {
  use super::*;

  // `OperatorApplied` is the public spelling of the same operator as `Curry`.
  // Bare `OperatorApplied[f]` reverses two arguments.
  #[test]
  fn bare_reverses_two_args() {
    assert_eq!(interpret("OperatorApplied[f][a][b]").unwrap(), "f[b, a]");
    assert_eq!(interpret("OperatorApplied[Power][2][3]").unwrap(), "9");
    assert_eq!(interpret("OperatorApplied[Subtract][2][10]").unwrap(), "8");
  }

  #[test]
  fn partial_application_keeps_head() {
    assert_eq!(
      interpret("OperatorApplied[f][a]").unwrap(),
      "OperatorApplied[f][a]"
    );
    assert_eq!(
      interpret("Head[OperatorApplied[f]]").unwrap(),
      "OperatorApplied"
    );
    // The bare operator object stays unevaluated.
    assert_eq!(
      interpret("OperatorApplied[f]").unwrap(),
      "OperatorApplied[f]"
    );
  }

  // `OperatorApplied[f, n]` collects n arguments in order.
  #[test]
  fn n_collects_in_order() {
    assert_eq!(interpret("OperatorApplied[f, 1][a]").unwrap(), "f[a]");
    assert_eq!(interpret("OperatorApplied[f, 2][a][b]").unwrap(), "f[a, b]");
    assert_eq!(
      interpret("OperatorApplied[f, 3][a][b][c]").unwrap(),
      "f[a, b, c]"
    );
  }

  // `OperatorApplied[f, {perm}]` arranges by an explicit permutation.
  #[test]
  fn permutation_spec() {
    assert_eq!(
      interpret("OperatorApplied[f, {2, 1}][a][b]").unwrap(),
      "f[b, a]"
    );
    assert_eq!(
      interpret("OperatorApplied[f, {1, 3, 2}][a][b][c]").unwrap(),
      "f[a, c, b]"
    );
  }

  #[test]
  fn as_map_function() {
    assert_eq!(
      interpret("Map[OperatorApplied[Power][2], {1, 2, 3}]").unwrap(),
      "{1, 4, 9}"
    );
  }
}

mod append_prepend_nest_firstcase_operator_forms {
  use super::*;

  #[test]
  fn append_prepend_operator_forms() {
    assert_eq!(interpret("Append[x][{1, 2}]").unwrap(), "{1, 2, x}");
    assert_eq!(interpret("Prepend[x][{1, 2}]").unwrap(), "{x, 1, 2}");
    assert_eq!(
      interpret("Map[Append[0], {{1}, {2}}]").unwrap(),
      "{{1, 0}, {2, 0}}"
    );
  }

  #[test]
  fn nest_operator_form() {
    assert_eq!(interpret("Nest[f, 3][x]").unwrap(), "f[f[f[x]]]");
  }

  #[test]
  fn first_case_operator_form() {
    assert_eq!(interpret("FirstCase[_?OddQ][{2, 4, 5, 6}]").unwrap(), "5");
  }

  #[test]
  fn unapplied_operator_forms_stay_unevaluated() {
    // The bare operator object is valid (no arg-count error) and stays put.
    assert_eq!(interpret("Append[x]").unwrap(), "Append[x]");
    assert_eq!(interpret("Prepend[x]").unwrap(), "Prepend[x]");
    assert_eq!(interpret("Nest[f, 3]").unwrap(), "Nest[f, 3]");
    assert_eq!(interpret("FirstCase[p]").unwrap(), "FirstCase[p]");
  }

  #[test]
  fn replace_operator_form() {
    use woxi::interpret_with_stdout;
    // Replace[rules][expr] applies the rules — without a spurious argbu error.
    let r = interpret_with_stdout("Replace[x_ :> x^2][5]").unwrap();
    assert_eq!(r.result, "25");
    assert!(r.warnings.is_empty());
    assert_eq!(interpret("Replace[{a -> 1, b -> 2}][a]").unwrap(), "1");
    assert_eq!(
      interpret("Map[Replace[x_ :> x^2], {1, 2, 3}]").unwrap(),
      "{1, 4, 9}"
    );
    // Bare operator object stays unevaluated.
    assert_eq!(interpret("Replace[r]").unwrap(), "Replace[r]");
  }
}

mod curry_applied {
  use super::*;

  // `CurryApplied[f, n]` collects n arguments in order, like Curry[f, n].
  #[test]
  fn collects_n_in_order() {
    assert_eq!(interpret("CurryApplied[f, 2][a][b]").unwrap(), "f[a, b]");
    assert_eq!(interpret("CurryApplied[Plus, 3][1][2][3]").unwrap(), "6");
    assert_eq!(interpret("CurryApplied[Power, 2][2][3]").unwrap(), "8");
  }

  #[test]
  fn count_one_is_immediate() {
    assert_eq!(interpret("CurryApplied[f, 1][a]").unwrap(), "f[a]");
  }

  // `CurryApplied[f, {perm}]` arranges by an explicit permutation.
  #[test]
  fn explicit_permutation() {
    assert_eq!(
      interpret("CurryApplied[f, {2, 1}][a][b]").unwrap(),
      "f[b, a]"
    );
  }

  // Unlike Curry/OperatorApplied, a bare CurryApplied[f] (no count) does not
  // curry — it stays inert.
  #[test]
  fn no_count_stays_inert() {
    assert_eq!(
      interpret("CurryApplied[g][a][b]").unwrap(),
      "CurryApplied[g][a][b]"
    );
  }

  #[test]
  fn bare_and_partial_forms_stay_symbolic() {
    assert_eq!(
      interpret("CurryApplied[f, 2]").unwrap(),
      "CurryApplied[f, 2]"
    );
    assert_eq!(
      interpret("CurryApplied[f, 2][a]").unwrap(),
      "CurryApplied[f, 2][a]"
    );
  }
}

mod reverse_applied {
  use super::*;

  // ReverseApplied[f][x1, …, xn] = f[xn, …, x1] (all arguments reversed).
  #[test]
  fn reverses_all_arguments() {
    assert_eq!(interpret("ReverseApplied[f][a, b]").unwrap(), "f[b, a]");
    assert_eq!(
      interpret("ReverseApplied[f][a, b, c]").unwrap(),
      "f[c, b, a]"
    );
    assert_eq!(interpret("ReverseApplied[f][a]").unwrap(), "f[a]");
    assert_eq!(interpret("ReverseApplied[f][]").unwrap(), "f[]");
  }

  #[test]
  fn applies_to_builtins() {
    assert_eq!(interpret("ReverseApplied[Subtract][2, 10]").unwrap(), "8");
    assert_eq!(interpret("ReverseApplied[Divide][2, 10]").unwrap(), "5");
    assert_eq!(interpret("ReverseApplied[Plus][1, 2, 3]").unwrap(), "6");
    assert_eq!(
      interpret("ReverseApplied[List][1, 2, 3, 4]").unwrap(),
      "{4, 3, 2, 1}"
    );
  }

  // ReverseApplied[f, n] reverses only the first n arguments.
  #[test]
  fn reverses_first_n_arguments() {
    assert_eq!(
      interpret("ReverseApplied[f, 2][a, b, c]").unwrap(),
      "f[b, a, c]"
    );
    assert_eq!(
      interpret("ReverseApplied[f, 3][a, b, c, d]").unwrap(),
      "f[c, b, a, d]"
    );
    assert_eq!(interpret("ReverseApplied[f, 1][a, b]").unwrap(), "f[a, b]");
  }

  // It fires on the first application and does not accumulate like Curry.
  #[test]
  fn does_not_accumulate() {
    assert_eq!(interpret("ReverseApplied[f][a][b]").unwrap(), "f[a][b]");
  }

  // The explicit Apply form also reverses.
  #[test]
  fn via_apply() {
    assert_eq!(
      interpret("Apply[ReverseApplied[Rule], {1, 2}]").unwrap(),
      "2 -> 1"
    );
  }

  #[test]
  fn unapplied_stays_symbolic() {
    assert_eq!(
      interpret("Head[ReverseApplied[f]]").unwrap(),
      "ReverseApplied"
    );
  }
}

mod apply_operator_composite_head {
  use super::*;

  // `@@` replaces the head with the whole left-hand expression, not just its
  // head symbol: `g[a] @@ {1, 2}` is `g[a][1, 2]`, not `g[1, 2]`.
  #[test]
  fn composite_function_call_head() {
    assert_eq!(interpret("g[a] @@ {1, 2}").unwrap(), "g[a][1, 2]");
    assert_eq!(interpret("h[x] @@ {1, 2}").unwrap(), "h[x][1, 2]");
  }

  #[test]
  fn operator_heads_reduce() {
    assert_eq!(
      interpret("Composition[f, g] @@ {1, 2}").unwrap(),
      "f[g[1, 2]]"
    );
    assert_eq!(
      interpret("OperatorApplied[Rule] @@ {1, 2}").unwrap(),
      "2 -> 1"
    );
  }

  // MapApply (@@@) threads the composite head over each sublist.
  #[test]
  fn map_apply_composite_head() {
    assert_eq!(
      interpret("ReverseApplied[Rule] @@@ {{1, 2}, {3, 4}}").unwrap(),
      "{2 -> 1, 4 -> 3}"
    );
    assert_eq!(
      interpret("Composition[f, g] @@@ {{1, 2}}").unwrap(),
      "{f[g[1, 2]]}"
    );
  }

  // An arithmetic head stays an inert curried call, correctly parenthesized.
  #[test]
  fn arithmetic_head_stays_inert() {
    assert_eq!(interpret("(f + g) @@ {1, 2}").unwrap(), "(f + g)[1, 2]");
  }

  // Plain symbol and head-variable cases still work.
  #[test]
  fn simple_heads_unaffected() {
    assert_eq!(interpret("f @@ {1, 2}").unwrap(), "f[1, 2]");
    assert_eq!(interpret("Plus @@ {1, 2, 3}").unwrap(), "6");
    assert_eq!(interpret("myf = Plus; myf @@ {1, 2, 3}").unwrap(), "6");
  }
}

mod apply_held_arguments {
  use super::*;

  // When the source head held its arguments (Hold, HoldForm, …), replacing it
  // with a non-holding head must evaluate those arguments, while replacing it
  // with another holding head must not — i.e. the new head's attributes govern.

  #[test]
  fn list_evaluates_held_arguments() {
    assert_eq!(
      interpret("Apply[List, Hold[1 + 1, 2 + 2]]").unwrap(),
      "{2, 4}"
    );
  }

  #[test]
  fn operator_form_evaluates_held_arguments() {
    assert_eq!(interpret("List @@ Hold[1 + 1]").unwrap(), "{2}");
    assert_eq!(interpret("Times @@ Hold[2 + 1, 3]").unwrap(), "9");
  }

  #[test]
  fn unknown_head_evaluates_held_arguments() {
    assert_eq!(interpret("Apply[f, Hold[1 + 1]]").unwrap(), "f[2]");
  }

  #[test]
  fn holdform_source_is_evaluated_under_list() {
    assert_eq!(interpret("Apply[List, HoldForm[1 + 1]]").unwrap(), "{2}");
  }

  #[test]
  fn holding_new_head_keeps_arguments_unevaluated() {
    // Hold has HoldAll, so the lifted (still unevaluated) argument stays put.
    assert_eq!(
      interpret("Apply[Hold, Hold[1 + 1]]").unwrap(),
      "Hold[1 + 1]"
    );
    assert_eq!(
      interpret("HoldForm @@ Hold[1 + 1]").unwrap(),
      "HoldForm[1 + 1]"
    );
  }

  #[test]
  fn list_source_already_evaluated_then_held() {
    // The source list evaluates 1 + 1 to 2 before Apply runs, so Hold[2].
    assert_eq!(interpret("Apply[Hold, {1 + 1}]").unwrap(), "Hold[2]");
  }
}

mod map_operator_composite_head {
  use super::*;

  // Map applies the whole composite head to each element as a curried call:
  // g[a] /@ {1, 2} → {g[a][1], g[a][2]}, not {g[a, 1], g[a, 2]}.
  #[test]
  fn map_composite_function_call_head() {
    assert_eq!(interpret("g[a] /@ {1, 2}").unwrap(), "{g[a][1], g[a][2]}");
    assert_eq!(
      interpret("Map[g[a], {1, 2}]").unwrap(),
      "{g[a][1], g[a][2]}"
    );
    assert_eq!(
      interpret("g[a, b] /@ {1, 2}").unwrap(),
      "{g[a, b][1], g[a, b][2]}"
    );
  }

  #[test]
  fn map_operator_heads_reduce() {
    assert_eq!(
      interpret("Composition[f, g] /@ {1, 2}").unwrap(),
      "{f[g[1]], f[g[2]]}"
    );
    assert_eq!(
      interpret("OperatorApplied[Power][2] /@ {1, 2, 3}").unwrap(),
      "{1, 4, 9}"
    );
    assert_eq!(
      interpret("ReverseApplied[f] /@ {{1, 2}, {3, 4}}").unwrap(),
      "{f[{1, 2}], f[{3, 4}]}"
    );
  }

  // Postfix (//) and prefix (@) with a composite head also curry.
  #[test]
  fn postfix_and_prefix_composite_head() {
    assert_eq!(interpret("x // g[a]").unwrap(), "g[a][x]");
    assert_eq!(interpret("g[a] @ x").unwrap(), "g[a][x]");
  }

  // Plain heads are unaffected.
  #[test]
  fn simple_head_unaffected() {
    assert_eq!(interpret("f /@ {1, 2}").unwrap(), "{f[1], f[2]}");
  }
}

mod arithmetic_head_application {
  use super::*;

  // A compound arithmetic head stays an inert, parenthesized curried call
  // across every application operator — not mis-associated as f + g[x].
  #[test]
  fn prefix_at() {
    assert_eq!(interpret("(f + g) @ x").unwrap(), "(f + g)[x]");
    assert_eq!(interpret("(a*b) @ x").unwrap(), "(a*b)[x]");
    assert_eq!(interpret("(f - g) @ x").unwrap(), "(f - g)[x]");
  }

  #[test]
  fn postfix_slashslash() {
    assert_eq!(interpret("x // (f + g)").unwrap(), "(f + g)[x]");
  }

  #[test]
  fn map_and_apply() {
    assert_eq!(
      interpret("(f + g) /@ {1, 2}").unwrap(),
      "{(f + g)[1], (f + g)[2]}"
    );
    assert_eq!(interpret("(f + g) @@ {1, 2}").unwrap(), "(f + g)[1, 2]");
  }

  #[test]
  fn full_form_is_curried() {
    assert_eq!(interpret("Head[(f + g) @ x]").unwrap(), "f + g");
  }
}
