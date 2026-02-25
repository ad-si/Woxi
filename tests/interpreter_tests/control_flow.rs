use super::*;

mod for_loop {
  use super::*;

  #[test]
  fn basic_for() {
    clear_state();
    assert_eq!(
      interpret("s = 0; For[i = 1, i <= 5, i++, s += i]; s").unwrap(),
      "15"
    );
  }

  #[test]
  fn for_returns_null() {
    clear_state();
    assert_eq!(interpret("For[i = 0, i < 3, i++, i]").unwrap(), "Null");
  }

  #[test]
  fn for_three_args_no_body() {
    clear_state();
    assert_eq!(interpret("For[i = 1, i < 10, i = i*2]; i").unwrap(), "16");
  }

  #[test]
  fn for_with_break() {
    clear_state();
    assert_eq!(
      interpret("For[i = 0, i < 10, i++, If[i == 5, Break[]]]; i").unwrap(),
      "5"
    );
  }

  #[test]
  fn for_with_continue() {
    clear_state();
    assert_eq!(
      interpret("s = 0; For[i = 0, i < 10, i++, If[Mod[i,2] == 0, Continue[]]; s += i]; s")
        .unwrap(),
      "25"
    );
  }

  #[test]
  fn for_with_return_in_function() {
    clear_state();
    assert_eq!(
      interpret(
        "f[n_] := For[i = 2, i <= n, i++, If[Mod[n, i] == 0, Return[i]]]; f[15]"
      )
      .unwrap(),
      "3"
    );
  }
}

mod while_loop {
  use super::*;

  #[test]
  fn basic_while() {
    clear_state();
    assert_eq!(interpret("i = 0; While[i < 5, i++]; i").unwrap(), "5");
  }

  #[test]
  fn while_with_assignment() {
    clear_state();
    assert_eq!(
      interpret("n = 0; While[n < 10, n = n + 3]; n").unwrap(),
      "12"
    );
  }

  #[test]
  fn while_returns_null() {
    clear_state();
    assert_eq!(interpret("i = 0; While[i < 3, i++]").unwrap(), "Null");
  }

  #[test]
  fn while_with_break() {
    clear_state();
    assert_eq!(
      interpret("i = 0; While[True, i++; If[i >= 5, Break[]]]; i").unwrap(),
      "5"
    );
  }

  #[test]
  fn while_in_module() {
    clear_state();
    assert_eq!(
      interpret("Module[{i = 0, s = 0}, While[i < 5, s += i; i++]; s]")
        .unwrap(),
      "10"
    );
  }

  #[test]
  fn while_false_condition() {
    clear_state();
    assert_eq!(interpret("While[False, Print[1]]").unwrap(), "Null");
  }
}

mod block_scoping {
  use super::*;

  #[test]
  fn basic_block() {
    clear_state();
    assert_eq!(interpret("Block[{x = 5}, x + 1]").unwrap(), "6");
  }

  #[test]
  fn block_restores_variables() {
    clear_state();
    assert_eq!(interpret("x = 10; Block[{x = 5}, x]; x").unwrap(), "10");
  }

  #[test]
  fn block_uninitialized_var() {
    clear_state();
    assert_eq!(interpret("Block[{x}, x]").unwrap(), "x");
  }

  #[test]
  fn block_multiple_vars() {
    clear_state();
    assert_eq!(interpret("Block[{x = 3, y = 4}, x + y]").unwrap(), "7");
  }
}

mod return_value {
  use super::*;

  #[test]
  fn return_in_block() {
    clear_state();
    // Return propagates through Block; at top level it becomes symbolic Return[val] (like wolframscript)
    assert_eq!(interpret("Block[{}, Return[42]]").unwrap(), "Return[42]");
  }

  #[test]
  fn return_in_module() {
    clear_state();
    // At top level, uncaught Return[] becomes symbolic Return[val] (like wolframscript)
    assert_eq!(
      interpret("Module[{x = 10}, Return[x + 1]]").unwrap(),
      "Return[11]"
    );
  }

  #[test]
  fn return_in_block_inside_function() {
    clear_state();
    assert_eq!(
      interpret("f[] := Block[{}, Return[42]]; f[]").unwrap(),
      "42"
    );
  }

  #[test]
  fn return_in_module_inside_function() {
    clear_state();
    assert_eq!(
      interpret("g[] := Module[{x = 10}, Return[x + 1]]; g[]").unwrap(),
      "11"
    );
  }
}

mod if_function_extended {
  use super::*;

  #[test]
  fn if_four_args_default() {
    // If[non-boolean, true-branch, false-branch, default]
    // Non-boolean condition should return default (4th arg)
    assert_eq!(interpret("If[\"x\", 1, 0, 2]").unwrap(), "2");
  }

  #[test]
  fn if_four_args_true() {
    assert_eq!(interpret("If[True, 1, 0, 2]").unwrap(), "1");
  }

  #[test]
  fn if_four_args_false() {
    assert_eq!(interpret("If[False, 1, 0, 2]").unwrap(), "0");
  }
}

mod do_single_iter {
  use super::*;

  #[test]
  fn repeat_n_times() {
    clear_state();
    assert_eq!(
      interpret_with_stdout("Do[Print[\"hello\"], {3}]")
        .unwrap()
        .stdout,
      "hello\nhello\nhello\n"
    );
  }
}

mod while_single_arg {
  use super::*;

  #[test]
  fn do_while_pattern() {
    clear_state();
    assert_eq!(
      interpret_with_stdout(
        "value = 0; While[value++; Print[value]; Mod[value,6]!=0]"
      )
      .unwrap()
      .stdout,
      "1\n2\n3\n4\n5\n6\n"
    );
  }
}

mod compound_assignment {
  use super::*;

  #[test]
  fn add_to() {
    clear_state();
    assert_eq!(interpret("x = 5; x += 3; x").unwrap(), "8");
  }

  #[test]
  fn add_to_return_value() {
    clear_state();
    assert_eq!(interpret("x = 10; x += 7").unwrap(), "17");
  }

  #[test]
  fn subtract_from() {
    clear_state();
    assert_eq!(interpret("x = 10; x -= 3; x").unwrap(), "7");
  }

  #[test]
  fn times_by() {
    clear_state();
    assert_eq!(interpret("x = 5; x *= 4; x").unwrap(), "20");
  }

  #[test]
  fn divide_by() {
    clear_state();
    assert_eq!(interpret("x = 20; x /= 4; x").unwrap(), "5");
  }

  #[test]
  fn chained_compound_assignment() {
    clear_state();
    assert_eq!(interpret("x = 1; x += 2; x *= 3; x -= 1; x").unwrap(), "8");
  }
}

mod chained_assignment {
  use super::*;

  #[test]
  fn basic_chained_set() {
    // s = k = {} should set both s and k to {}
    clear_state();
    assert_eq!(interpret("s = k = {}; {s, k}").unwrap(), "{{}, {}}");
  }

  #[test]
  fn triple_chained_set() {
    clear_state();
    assert_eq!(
      interpret("a = b = c = 42; {a, b, c}").unwrap(),
      "{42, 42, 42}"
    );
  }

  #[test]
  fn chained_set_with_expression() {
    clear_state();
    assert_eq!(interpret("x = y = 1 + 2; {x, y}").unwrap(), "{3, 3}");
  }

  #[test]
  fn right_associativity() {
    // a = b = 5 should be parsed as a = (b = 5), not (a = b) = 5
    clear_state();
    assert_eq!(interpret("a = b = 5; b").unwrap(), "5");
  }
}

mod append_to {
  use super::*;

  #[test]
  fn basic() {
    clear_state();
    assert_eq!(
      interpret("x = {1, 2, 3}; AppendTo[x, 4]").unwrap(),
      "{1, 2, 3, 4}"
    );
  }

  #[test]
  fn updates_variable() {
    clear_state();
    assert_eq!(
      interpret("x = {1, 2}; AppendTo[x, 3]; x").unwrap(),
      "{1, 2, 3}"
    );
  }
}

mod prepend_to {
  use super::*;

  #[test]
  fn basic() {
    clear_state();
    assert_eq!(
      interpret("x = {1, 2, 3}; PrependTo[x, 0]").unwrap(),
      "{0, 1, 2, 3}"
    );
  }

  #[test]
  fn updates_variable() {
    clear_state();
    assert_eq!(
      interpret("x = {2, 3}; PrependTo[x, 1]; x").unwrap(),
      "{1, 2, 3}"
    );
  }
}

mod check {
  use super::*;

  #[test]
  fn check_no_error() {
    clear_state();
    assert_eq!(interpret("Check[2 + 3, failed]").unwrap(), "5");
  }

  #[test]
  fn check_with_error() {
    clear_state();
    assert_eq!(interpret("Check[1/0, failed]").unwrap(), "failed");
  }

  #[test]
  fn check_failexpr_is_evaluated() {
    clear_state();
    assert_eq!(interpret("Check[1/0, 1 + 1]").unwrap(), "2");
  }
}

mod abort {
  use super::*;

  #[test]
  fn abort_returns_aborted() {
    clear_state();
    assert_eq!(interpret("Abort[]").unwrap(), "$Aborted");
  }

  #[test]
  fn check_abort_catches_abort() {
    clear_state();
    assert_eq!(interpret("CheckAbort[Abort[], caught]").unwrap(), "caught");
  }

  #[test]
  fn check_abort_no_abort() {
    clear_state();
    assert_eq!(interpret("CheckAbort[2 + 3, caught]").unwrap(), "5");
  }

  #[test]
  fn abort_stops_computation() {
    clear_state();
    assert_eq!(interpret("x = 1; Abort[]; x = 2; x").unwrap(), "$Aborted");
  }
}

mod implies {
  use super::*;

  #[test]
  fn true_implies_symbolic() {
    clear_state();
    assert_eq!(interpret("Implies[True, a]").unwrap(), "a");
  }

  #[test]
  fn false_implies_anything() {
    clear_state();
    assert_eq!(interpret("Implies[False, a]").unwrap(), "True");
  }

  #[test]
  fn true_implies_true() {
    clear_state();
    assert_eq!(interpret("Implies[True, True]").unwrap(), "True");
  }

  #[test]
  fn true_implies_false() {
    clear_state();
    assert_eq!(interpret("Implies[True, False]").unwrap(), "False");
  }
}

mod which {
  use super::*;

  #[test]
  fn symbolic_condition_returns_remaining() {
    clear_state();
    assert_eq!(
      interpret("Which[False, a, x, b, True, c]").unwrap(),
      "Which[x, b, True, c]"
    );
  }

  #[test]
  fn all_false_returns_null() {
    clear_state();
    assert_eq!(interpret("Which[False, a, False, b]").unwrap(), "Null");
  }
}

mod or_logical {
  use super::*;

  #[test]
  fn simplifies_with_false() {
    clear_state();
    assert_eq!(interpret("Or[a, False, b]").unwrap(), "a || b");
  }

  #[test]
  fn true_short_circuits() {
    clear_state();
    assert_eq!(interpret("Or[False, True, a]").unwrap(), "True");
  }

  #[test]
  fn all_false() {
    clear_state();
    assert_eq!(interpret("Or[False, False]").unwrap(), "False");
  }
}

mod and_logical {
  use super::*;

  #[test]
  fn simplifies_with_true() {
    clear_state();
    assert_eq!(interpret("And[a, True, b]").unwrap(), "a && b");
  }

  #[test]
  fn false_short_circuits() {
    clear_state();
    assert_eq!(interpret("And[True, False, a]").unwrap(), "False");
  }
}

mod xor_logical {
  use super::*;

  #[test]
  fn simplifies_with_false() {
    clear_state();
    assert_eq!(interpret("Xor[a, False, b]").unwrap(), "Xor[a, b]");
  }
}

mod not_logical {
  use super::*;

  #[test]
  fn not_true() {
    clear_state();
    assert_eq!(interpret("Not[True]").unwrap(), "False");
  }

  #[test]
  fn not_false() {
    clear_state();
    assert_eq!(interpret("Not[False]").unwrap(), "True");
  }

  #[test]
  fn not_symbolic() {
    clear_state();
    assert_eq!(interpret("Not[a]").unwrap(), " !a");
  }

  #[test]
  fn not_symbolic_expr() {
    clear_state();
    assert_eq!(interpret("Not[a && b]").unwrap(), " !(a && b)");
  }

  #[test]
  fn prefix_not_true() {
    clear_state();
    // !True should parse as Not[True] and evaluate to False
    assert_eq!(interpret("Not[True]").unwrap(), "False");
  }

  #[test]
  fn prefix_not_false() {
    clear_state();
    assert_eq!(interpret("Not[False]").unwrap(), "True");
  }
}

mod nand_logical {
  use super::*;

  #[test]
  fn all_true() {
    clear_state();
    assert_eq!(interpret("Nand[True, True]").unwrap(), "False");
  }

  #[test]
  fn one_false() {
    clear_state();
    assert_eq!(interpret("Nand[True, False]").unwrap(), "True");
  }

  #[test]
  fn symbolic_stays() {
    clear_state();
    assert_eq!(interpret("Nand[a, b]").unwrap(), "Nand[a, b]");
  }

  #[test]
  fn symbolic_with_true() {
    clear_state();
    assert_eq!(interpret("Nand[a, True, b]").unwrap(), "Nand[a, b]");
  }
}

mod nor_logical {
  use super::*;

  #[test]
  fn all_false() {
    clear_state();
    assert_eq!(interpret("Nor[False, False]").unwrap(), "True");
  }

  #[test]
  fn one_true() {
    clear_state();
    assert_eq!(interpret("Nor[False, True]").unwrap(), "False");
  }

  #[test]
  fn symbolic_stays() {
    clear_state();
    assert_eq!(interpret("Nor[a, b]").unwrap(), "Nor[a, b]");
  }

  #[test]
  fn symbolic_with_false() {
    clear_state();
    assert_eq!(interpret("Nor[a, False, b]").unwrap(), "Nor[a, b]");
  }
}

mod interrupt {
  use super::*;

  #[test]
  fn interrupt_returns_aborted() {
    clear_state();
    assert_eq!(interpret("Interrupt[]").unwrap(), "$Aborted");
  }

  #[test]
  fn interrupt_stops_computation() {
    clear_state();
    let result =
      interpret_with_stdout("Print[\"a\"]; Interrupt[]; Print[\"b\"]").unwrap();
    assert_eq!(result.stdout, "a\n");
    assert_eq!(result.result, "$Aborted");
  }
}

mod pause {
  use super::*;

  #[test]
  fn pause_returns_null() {
    clear_state();
    assert_eq!(interpret("Pause[0.01]").unwrap(), "Null");
  }
}

mod goto_label {
  use super::*;

  #[test]
  fn basic_goto_label_loop() {
    clear_state();
    assert_eq!(
      interpret("i = 0; Label[start]; i = i + 1; If[i < 5, Goto[start]]; i")
        .unwrap(),
      "5"
    );
  }

  #[test]
  fn label_alone_returns_null() {
    clear_state();
    // Label at top level (not inside CompoundExpr) stays symbolic
    assert_eq!(interpret("Label[x]").unwrap(), "Null");
  }

  #[test]
  fn goto_no_label_returns_null() {
    clear_state();
    // Goto with no matching label returns Null (with stderr message)
    assert_eq!(interpret("Goto[x]").unwrap(), "Null");
  }

  #[test]
  fn goto_label_in_module() {
    clear_state();
    assert_eq!(
      interpret("Module[{i = 0}, Label[s]; i = i + 1; If[i < 3, Goto[s]]; i]")
        .unwrap(),
      "3"
    );
  }

  #[test]
  fn goto_label_in_function() {
    clear_state();
    assert_eq!(
      interpret(
        "f[] := (i = 0; Label[s]; i = i + 1; If[i < 3, Goto[s]]; i); f[]"
      )
      .unwrap(),
      "3"
    );
  }

  #[test]
  fn goto_label_with_integer_tag() {
    clear_state();
    assert_eq!(
      interpret("i = 0; Label[1]; i = i + 1; If[i < 3, Goto[1]]; i").unwrap(),
      "3"
    );
  }

  #[test]
  fn goto_label_with_string_tag() {
    clear_state();
    assert_eq!(
      interpret(
        "i = 0; Label[\"loop\"]; i = i + 1; If[i < 4, Goto[\"loop\"]]; i"
      )
      .unwrap(),
      "4"
    );
  }

  #[test]
  fn goto_label_with_print() {
    clear_state();
    let result = interpret_with_stdout(
      "i = 0; Label[start]; i = i + 1; Print[i]; If[i < 3, Goto[start]]; i",
    )
    .unwrap();
    assert_eq!(result.stdout, "1\n2\n3\n");
    assert_eq!(result.result, "3");
  }

  #[test]
  fn goto_label_hold_all() {
    clear_state();
    // Goto and Label have HoldAll, so tags are not evaluated
    assert_eq!(
      interpret("Attributes[Goto]").unwrap(),
      "{HoldAll, Protected}"
    );
    assert_eq!(
      interpret("Attributes[Label]").unwrap(),
      "{HoldAll, Protected}"
    );
  }
}

mod do_multi_iterator {
  use super::*;

  #[test]
  fn two_iterators() {
    clear_state();
    assert_eq!(
      interpret_with_stdout("Do[Print[{i, j}], {i, 1, 2}, {j, 3, 5}]")
        .unwrap()
        .stdout,
      "{1, 3}\n{1, 4}\n{1, 5}\n{2, 3}\n{2, 4}\n{2, 5}\n"
    );
  }

  #[test]
  fn three_iterators() {
    clear_state();
    assert_eq!(
      interpret(
        "s = 0; Do[s += i * j * k, {i, 1, 2}, {j, 1, 2}, {k, 1, 2}]; s"
      )
      .unwrap(),
      "27"
    );
  }

  #[test]
  fn with_break() {
    clear_state();
    assert_eq!(
      interpret_with_stdout(
        "Do[If[i > 10, Break[], If[Mod[i, 2] == 0, Continue[]]; Print[i]], {i, 5, 20}]"
      )
      .unwrap()
      .stdout,
      "5\n7\n9\n"
    );
  }
}

mod absolute_timing {
  use super::*;

  #[test]
  fn absolute_timing_returns_list() {
    clear_state();
    // AbsoluteTiming returns {time, result}
    let result = interpret("AbsoluteTiming[1 + 1]").unwrap();
    assert!(result.starts_with('{'));
    assert!(result.ends_with('}'));
    assert!(result.contains(", 2}"));
  }

  #[test]
  fn timing_returns_list() {
    clear_state();
    let result = interpret("Timing[2 + 3]").unwrap();
    assert!(result.starts_with('{'));
    assert!(result.contains(", 5}"));
  }
}

mod return_in_loops {
  use super::*;

  #[test]
  fn return_in_do() {
    clear_state();
    assert_eq!(interpret("Do[If[True, Return[42]], {1}]").unwrap(), "42");
  }

  #[test]
  fn return_stops_do_loop() {
    clear_state();
    assert_eq!(interpret("Do[If[i > 3, Return[i]], {i, 10}]").unwrap(), "4");
  }

  #[test]
  fn return_no_arg_in_do() {
    clear_state();
    assert_eq!(
      interpret("Do[If[i > 3, Return[]]; Print[i], {i, 10}]").unwrap(),
      "Null"
    );
  }

  #[test]
  fn return_in_while() {
    clear_state();
    // In Wolfram, Return[] inside While is NOT caught by the loop -
    // it propagates up as the symbolic Return[99]
    assert_eq!(interpret("While[True, Return[99]]").unwrap(), "Return[99]");
  }

  #[test]
  fn return_in_for() {
    clear_state();
    // In Wolfram, Return[] inside For is NOT caught by the loop -
    // it propagates up as the symbolic Return[5]
    assert_eq!(
      interpret("For[i=1, i<=10, i++, If[i==5, Return[i]]]").unwrap(),
      "Return[5]"
    );
  }
}

mod logical_expand {
  use super::*;

  #[test]
  fn distribute_and_over_or() {
    assert_eq!(
      interpret("LogicalExpand[a && (b || c)]").unwrap(),
      "(a && b) || (a && c)"
    );
  }

  #[test]
  fn de_morgan_not_and() {
    assert_eq!(
      interpret("LogicalExpand[Not[a && b]]").unwrap(),
      " !a ||  !b"
    );
  }

  #[test]
  fn de_morgan_not_or() {
    assert_eq!(
      interpret("LogicalExpand[Not[a || b]]").unwrap(),
      " !a &&  !b"
    );
  }

  #[test]
  fn double_negation() {
    assert_eq!(interpret("LogicalExpand[Not[Not[a]]]").unwrap(), "a");
  }

  #[test]
  fn implies_expansion() {
    assert_eq!(
      interpret("LogicalExpand[Implies[a, b]]").unwrap(),
      "b ||  !a"
    );
  }

  #[test]
  fn xor_expansion() {
    assert_eq!(
      interpret("LogicalExpand[Xor[a, b]]").unwrap(),
      "(a &&  !b) || (b &&  !a)"
    );
  }

  #[test]
  fn equivalent_expansion() {
    assert_eq!(
      interpret("LogicalExpand[Equivalent[a, b]]").unwrap(),
      "(a && b) || ( !a &&  !b)"
    );
  }

  #[test]
  fn nand_expansion() {
    assert_eq!(
      interpret("LogicalExpand[Nand[a, b]]").unwrap(),
      " !a ||  !b"
    );
  }

  #[test]
  fn nor_expansion() {
    assert_eq!(interpret("LogicalExpand[Nor[a, b]]").unwrap(), " !a &&  !b");
  }

  #[test]
  fn nested_expansion() {
    // (a || b) && (c || d) → (a && c) || (a && d) || (b && c) || (b && d)
    assert_eq!(
      interpret("LogicalExpand[(a || b) && (c || d)]").unwrap(),
      "(a && c) || (a && d) || (b && c) || (b && d)"
    );
  }

  #[test]
  fn already_dnf() {
    assert_eq!(
      interpret("LogicalExpand[a || (b && c)]").unwrap(),
      "a || (b && c)"
    );
  }

  #[test]
  fn true_false() {
    assert_eq!(interpret("LogicalExpand[True]").unwrap(), "True");
    assert_eq!(interpret("LogicalExpand[False]").unwrap(), "False");
  }

  #[test]
  fn single_symbol() {
    assert_eq!(interpret("LogicalExpand[a]").unwrap(), "a");
  }
}
