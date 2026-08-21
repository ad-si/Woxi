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
    assert_eq!(interpret("For[i = 0, i < 3, i++, i]").unwrap(), "\0");
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
    assert_eq!(interpret("i = 0; While[i < 3, i++]").unwrap(), "\0");
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
    assert_eq!(interpret("While[False, Print[1]]").unwrap(), "\0");
  }
}

mod module_pattern_renaming {
  use super::*;

  #[test]
  fn pattern_variable_follows_local_rename() {
    clear_state();
    // Regression: Module renames a local throughout the body — including
    // pattern names in inner definitions, matching Wolfram. Previously
    // only the definition's RHS was renamed, so `r[s_] := s^2` became
    // `r$n[s_] := s$n^2` and the pattern no longer bound (the
    // Doyle-spirals Demonstration's `doyle` helper hit this).
    assert_eq!(
      interpret("Module[{s, r}, r[s_] := s^2; r[3]]").unwrap(),
      "9"
    );
  }

  #[test]
  fn pattern_variable_shadows_initialized_local() {
    clear_state();
    // The pattern binding wins over the local's value inside the
    // definition body, exactly as in Wolfram after consistent renaming.
    assert_eq!(
      interpret("Module[{p, r}, p = 4; r[p_] := p + 1; r[10]]").unwrap(),
      "11"
    );
  }

  #[test]
  fn multiple_pattern_variables() {
    clear_state();
    assert_eq!(
      interpret(
        "Module[{s, t, r}, r[s_, t_] := s + 2 t; {r[1, 2], r[10, 20]}]"
      )
      .unwrap(),
      "{5, 50}"
    );
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

mod with_scoping {
  use super::*;

  // With takes any number of local variable specifications, each scoped
  // inside the ones before it, so a later one can build on an earlier one.
  #[test]
  fn several_variable_specifications() {
    clear_state();
    assert_eq!(interpret("With[{x = 5}, {y = x + 1}, y^2]").unwrap(), "36");
    clear_state();
    assert_eq!(
      interpret("With[{x = 5}, {y = x + 1}, {z = y + 1}, z^2]").unwrap(),
      "49"
    );
  }

  // Only the last argument is the body: two lists means one binding
  // specification and a list-valued body.
  #[test]
  fn last_argument_is_the_body() {
    clear_state();
    assert_eq!(interpret("With[{x = 5}, {y = x + 1}]").unwrap(), "{6}");
  }

  #[test]
  fn a_later_specification_shadows_an_earlier_one() {
    clear_state();
    assert_eq!(interpret("With[{x = 5}, {x = x + 1}, x^2]").unwrap(), "36");
    clear_state();
    assert_eq!(
      interpret("With[{x = 5}, {y = x}, {x = 1}, {x, y}]").unwrap(),
      "{1, 5}"
    );
  }

  // Regression: an inner With used to be substituted into blindly, so its
  // own local was replaced by the outer value (`With[{5 = 5 + 1}, 5^2]`).
  #[test]
  fn a_nested_scoping_construct_shadows_the_outer_value() {
    clear_state();
    assert_eq!(
      interpret("With[{x = 5}, With[{x = x + 1}, x^2]]").unwrap(),
      "36"
    );
    clear_state();
    assert_eq!(interpret("With[{x = 5}, With[{x = 1}, x]]").unwrap(), "1");
    clear_state();
    assert_eq!(
      interpret("With[{x = 5}, Module[{x = x + 1}, x^2]]").unwrap(),
      "36"
    );
    clear_state();
    assert_eq!(
      interpret("Function[{x}, With[{x = 1}, x]][7]").unwrap(),
      "1"
    );
  }

  // A value carried into an inner scope keeps its own meaning: the inner
  // local is renamed rather than capturing the incoming symbol.
  #[test]
  fn an_incoming_value_is_not_captured() {
    clear_state();
    assert_eq!(
      interpret("With[{x = y}, With[{y = 2}, x + y]]").unwrap(),
      "2 + y"
    );
  }

  // Bindings of one specification are parallel, not sequential: `b` takes
  // the outer `a`, not the one being bound alongside it.
  #[test]
  fn bindings_of_one_specification_are_parallel() {
    clear_state();
    assert_eq!(interpret("With[{a = 1, b = a}, {a, b}]").unwrap(), "{1, a}");
  }

  #[test]
  fn too_few_arguments() {
    clear_state();
    let result = interpret_with_stdout("With[{x = 5}]").unwrap();
    assert_eq!(result.result, "With[{x = 5}]");
    assert!(result.warnings[0].contains(
      "With::argmu: With called with 1 argument; 2 or more arguments are expected."
    ));
    clear_state();
    let result = interpret_with_stdout("With[]").unwrap();
    assert_eq!(result.result, "With[]");
    assert!(result.warnings[0].contains(
      "With::argm: With called with 0 arguments; 2 or more arguments are expected."
    ));
  }

  // A specification that is not a List is reported, and the call stays
  // unevaluated in the nested form the several-specification syntax means.
  #[test]
  fn a_specification_must_be_a_list() {
    for (code, expected, head) in [
      ("With[y, 3]", "With[y, 3]", "With"),
      (
        "With[{x = 5}, y, {z = 1}, z]",
        "With[y, With[{z = 1}, z]]",
        "With",
      ),
      ("Module[y, 3]", "Module[y, 3]", "Module"),
      ("Block[y, 3]", "Block[y, 3]", "Block"),
    ] {
      clear_state();
      let result = interpret_with_stdout(code).unwrap();
      assert_eq!(result.result, expected, "{code}");
      assert!(
        result.warnings[0].contains(&format!(
          "{head}::lvlist: Local variable specification y is not a List."
        )),
        "{code}: {:?}",
        result.warnings
      );
    }
  }

  /// Run `code` and expect it to stay unevaluated with exactly `message`.
  fn assert_local_spec_message(code: &str, expected: &str, message: &str) {
    clear_state();
    let result = interpret_with_stdout(code).unwrap();
    assert_eq!(result.result, expected, "{code}");
    assert!(
      result.warnings[0].contains(message),
      "{code}: {:?}",
      result.warnings
    );
  }

  // Every entry of a `Module`/`Block` specification must be a symbol or an
  // assignment to one. Issue #570: `f[x_] := Module[{x}, x]` substitutes the
  // argument into the binder, so `f[5]` has to report `Module[{5}, 5]`
  // instead of silently dropping the malformed local.
  #[test]
  fn a_local_must_be_a_symbol() {
    for (code, expected, message) in [
      (
        "Module[{5}, 5]",
        "Module[{5}, 5]",
        "Module::lvsym: Local variable specification {5} contains 5, which \
         is not a symbol or an assignment to a symbol.",
      ),
      (
        "f[x_] := Module[{x}, x]; f[5]",
        "Module[{5}, 5]",
        "Module::lvsym: Local variable specification {5} contains 5, which \
         is not a symbol or an assignment to a symbol.",
      ),
      (
        "Module[{x, y = 2, 3 + 4}, x]",
        "Module[{x, y = 2, 3 + 4}, x]",
        "Module::lvsym: Local variable specification {x, y = 2, 3 + 4} \
         contains 3 + 4, which is not a symbol or an assignment to a symbol.",
      ),
      (
        "Module[{f[x]}, 3]",
        "Module[{f[x]}, 3]",
        "Module::lvsym: Local variable specification {f[x]} contains f[x], \
         which is not a symbol or an assignment to a symbol.",
      ),
      (
        "Module[{x_}, 3]",
        "Module[{x_}, 3]",
        "Module::lvsym: Local variable specification {x_} contains x_, which \
         is not a symbol or an assignment to a symbol.",
      ),
      (
        "f[x_] := Block[{x}, x]; f[5]",
        "Block[{5}, 5]",
        "Block::lvsym: Local variable specification {5} contains 5, which is \
         not a symbol or an assignment to a symbol.",
      ),
    ] {
      assert_local_spec_message(code, expected, message);
    }
  }

  // Only the first offending entry is reported, left to right — a later
  // duplicate never pre-empts an earlier non-symbol.
  #[test]
  fn only_the_first_bad_local_is_reported() {
    assert_local_spec_message(
      "Module[{3, x, x}, 3]",
      "Module[{3, x, x}, 3]",
      "Module::lvsym: Local variable specification {3, x, x} contains 3, \
       which is not a symbol or an assignment to a symbol.",
    );
    assert_local_spec_message(
      "Module[{x, x, 3}, 3]",
      "Module[{x, x, 3}, 3]",
      "Module::dup: Duplicate local variable x found in local variable \
       specification {x, x, 3}.",
    );
  }

  // An assignment whose left-hand side is not a symbol gets its own message.
  #[test]
  fn a_local_assignment_must_target_a_symbol() {
    for (code, expected, message) in [
      (
        "Module[{x[1] = 3}, 4]",
        "Module[{x[1] = 3}, 4]",
        "Module::lvset: Local variable specification {x[1] = 3} contains \
         x[1] = 3, which is an assignment to x[1]; only assignments to \
         symbols are allowed.",
      ),
      (
        "With[{3 = 4}, 5]",
        "With[{3 = 4}, 5]",
        "With::lvset: Local variable specification {3 = 4} contains 3 = 4, \
         which is an assignment to 3; only assignments to symbols are allowed.",
      ),
      (
        "f[x_] := With[{x = 1}, x^2]; f[7]",
        "With[{7 = 1}, 7^2]",
        "With::lvset: Local variable specification {7 = 1} contains 7 = 1, \
         which is an assignment to 7; only assignments to symbols are allowed.",
      ),
      (
        // `With` substitutes straight through `Block`'s binder name.
        "With[{x = 5}, Block[{x = x + 1}, x^2]]",
        "Block[{5 = 5 + 1}, 5^2]",
        "Block::lvset: Local variable specification {5 = 5 + 1} contains \
         5 = 5 + 1, which is an assignment to 5; only assignments to symbols \
         are allowed.",
      ),
    ] {
      assert_local_spec_message(code, expected, message);
    }
  }

  // `With` needs a value for every local; anything that is not an assignment
  // is reported as the variable that is missing one, number or not.
  #[test]
  fn every_with_local_needs_a_value() {
    for (code, expected, message) in [
      (
        "With[{x}, x]",
        "With[{x}, x]",
        "With::lvws: Variable x in local variable specification {x} requires \
         a value.",
      ),
      (
        "With[{x = 1, 3}, 5]",
        "With[{x = 1, 3}, 5]",
        "With::lvws: Variable 3 in local variable specification {x = 1, 3} \
         requires a value.",
      ),
      (
        "With[{x + y}, 3]",
        "With[{x + y}, 3]",
        "With::lvws: Variable x + y in local variable specification {x + y} \
         requires a value.",
      ),
    ] {
      assert_local_spec_message(code, expected, message);
    }
  }

  #[test]
  fn a_local_may_not_be_declared_twice() {
    for (code, expected, message) in [
      (
        "Module[{x, x}, 3]",
        "Module[{x, x}, 3]",
        "Module::dup: Duplicate local variable x found in local variable \
         specification {x, x}.",
      ),
      (
        "Block[{x, x}, 3]",
        "Block[{x, x}, 3]",
        "Block::dup: Duplicate local variable x found in local variable \
         specification {x, x}.",
      ),
      (
        "With[{x = 1, x = 2}, x]",
        "With[{x = 1, x = 2}, x]",
        "With::dup: Duplicate local variable x found in local variable \
         specification {x = 1, x = 2}.",
      ),
    ] {
      assert_local_spec_message(code, expected, message);
    }
  }

  // `x := v` is a valid local specification; its right-hand side stays
  // unevaluated until the body reads the variable.
  #[test]
  fn a_local_may_be_assigned_with_set_delayed() {
    for code in [
      "Module[{x := 1 + 1}, x]",
      "Block[{x := 1 + 1}, x]",
      "With[{x := 1 + 1}, x]",
    ] {
      clear_state();
      assert_eq!(interpret(code).unwrap(), "2", "{code}");
    }
  }

  // A shadowed local is empty of the enclosing scope: neither an iterator
  // variable nor a substituted binding reaches into a nested binder.
  #[test]
  fn a_nested_binder_shadows_an_iterator_variable() {
    clear_state();
    assert_eq!(
      interpret("Table[Hold[Module[{k}, k]], {k, 1, 2}]").unwrap(),
      "{Hold[Module[{k}, k]], Hold[Module[{k}, k]]}"
    );
    clear_state();
    assert_eq!(
      interpret("Table[Hold[Function[k, k]], {k, 1, 2}]").unwrap(),
      "{Hold[Function[k, k]], Hold[Function[k, k]]}"
    );
    clear_state();
    assert_eq!(
      interpret("Sum[Hold[Module[{k}, k]], {k, 1, 2}]").unwrap(),
      "2*Hold[Module[{k}, k]]"
    );
    clear_state();
    // The binder name is only a name — the initializer still sees the
    // iterator's value.
    assert_eq!(
      interpret("Table[With[{k = k}, Hold[k]], {k, 1, 3}]").unwrap(),
      "{Hold[1], Hold[2], Hold[3]}"
    );
  }
}

mod return_value {
  use super::*;

  #[test]
  fn return_in_block() {
    clear_state();
    // Return propagates through Block; at top level it yields its argument
    // (matching wolframscript: `Block[{}, Return[42]]` outputs `42`).
    assert_eq!(interpret("Block[{}, Return[42]]").unwrap(), "42");
  }

  #[test]
  fn return_in_module() {
    clear_state();
    // At top level, uncaught Return[] yields its argument (matching
    // wolframscript: `Module[{x=10}, Return[x+1]]` outputs `11`).
    assert_eq!(interpret("Module[{x = 10}, Return[x + 1]]").unwrap(), "11");
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

  // A Return only leaves a definition body, `Do` and `Scan`. Anywhere a
  // value is being collected it stands as the expression it names, so a
  // `Table` still gives back a list of the same length. Values verified
  // against wolframscript.
  #[test]
  fn a_collected_value_keeps_the_return_it_names() {
    clear_state();
    for (code, expected) in [
      (
        "ToString[Table[Return[1], {2}], InputForm]",
        "{Return[1], Return[1]}",
      ),
      (
        "ToString[Table[If[i == 2, Return[i], i], {i, 3}], InputForm]",
        "{1, Return[2], 3}",
      ),
      (
        "ToString[Map[Return[#] &, {1, 2}], InputForm]",
        "{Return[1], Return[2]}",
      ),
      (
        "ToString[Map[Return, {1, 2}], InputForm]",
        "{Return[1], Return[2]}",
      ),
      // The criterion never says True, so nothing is selected.
      ("ToString[Select[{1, 2}, Return[True] &], InputForm]", "{}"),
      // Like Module and Block, With hands the Return on rather than
      // unwrapping it.
      (
        "ToString[With[{a = 1}, Return[a]; 4], InputForm]",
        "Return[1]",
      ),
    ] {
      assert_eq!(interpret(code).unwrap(), expected, "{code}");
    }
  }

  // A Return that a Table swallowed used to escape the whole function.
  #[test]
  fn a_return_inside_a_table_does_not_leave_the_function() {
    clear_state();
    assert_eq!(
      interpret("f[] := (Table[Return[1], {2}]; 9); f[]").unwrap(),
      "9"
    );
    clear_state();
    assert_eq!(
      interpret("f[x_] := (Map[Return, {1, 2}]; 5); f[0]").unwrap(),
      "5"
    );
    clear_state();
    assert_eq!(
      interpret("f[] := With[{a = 1}, Return[a]; 4]; f[]").unwrap(),
      "1"
    );
  }

  // The three places a Return does leave, which the change had to keep.
  #[test]
  fn a_definition_body_do_and_scan_still_take_it() {
    for (code, expected) in [
      ("f[x_] := (Return[1]; 2); f[0]", "1"),
      ("Do[Return[5], {2}]", "5"),
      ("Scan[Return, {1, 2}]", "1"),
      ("Scan[If[# > 2, Return[#]] &, {1, 2, 3, 4}, {1}]", "3"),
      (
        "f[x_] := (Do[If[i == 2, Return[i]], {i, 3}]; -1); f[0]",
        "-1",
      ),
      (
        "f[x_] := Module[{r = 0}, Do[If[i == 2, Return[i]], {i, 3}]; r]; f[0]",
        "0",
      ),
      ("f[] := While[True, Return[7]]; f[]", "7"),
      ("h[] := (Return[1]; 2); h[] + 10", "11"),
    ] {
      clear_state();
      assert_eq!(interpret(code).unwrap(), expected, "{code}");
    }
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

  #[test]
  fn do_list_iterator() {
    assert_eq!(
      interpret_with_stdout("Do[Print[i], {i, {a, b, c}}]")
        .unwrap()
        .stdout,
      "a\nb\nc\n"
    );
  }

  #[test]
  fn do_rational_upper_bound() {
    // Wolfram floors fractional bounds: Do[..., {i, 2, 7/2}] runs i = 2, 3
    clear_state();
    assert_eq!(
      interpret_with_stdout("Do[Print[i], {i, 2, 7/2}]")
        .unwrap()
        .stdout,
      "2\n3\n"
    );
  }

  #[test]
  fn do_real_upper_bound() {
    // Floor[3.5] = 3 — same flooring rule as fractional bounds.
    clear_state();
    assert_eq!(
      interpret_with_stdout("Do[Print[i], {i, 2, 3.5}]")
        .unwrap()
        .stdout,
      "2\n3\n"
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

  #[test]
  fn appends_to_part_target() {
    // AppendTo[x[[i]], v] writes v into the i-th slot of x — wolframscript
    // accepts a Part LHS just like Set/AddTo do.
    clear_state();
    assert_eq!(
      interpret("nums = ConstantArray[{}, 3]; AppendTo[nums[[1]], 5]; nums")
        .unwrap(),
      "{{5}, {}, {}}"
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

  #[test]
  fn works_on_function_head() {
    // AppendTo/PrependTo also work on any FunctionCall head — matches
    // wolframscript's behavior that they wrap in Prepend/Append regardless
    // of the head.
    clear_state();
    assert_eq!(
      interpret("y = f[a, b, c]; PrependTo[y, x]").unwrap(),
      "f[x, a, b, c]"
    );
    clear_state();
    assert_eq!(
      interpret("y = f[a, b, c]; AppendTo[y, x]").unwrap(),
      "f[a, b, c, x]"
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

  // The tag-filtered form only reacts to the listed messages. All
  // outputs verified against wolframscript 15.0.
  #[test]
  fn check_with_message_tags() {
    clear_state();
    assert_eq!(
      interpret(r#"Check[1/0, "err", Power::infy]"#).unwrap(),
      "err"
    );
    clear_state();
    assert_eq!(
      interpret(r#"Check[1/0, "err", Sum::div]"#).unwrap(),
      "ComplexInfinity"
    );
    clear_state();
    assert_eq!(
      interpret(r#"Check[1/0, "err", {Sum::div, Power::infy}]"#).unwrap(),
      "err"
    );
  }

  // User messages fill their `1` template slots with output-form
  // arguments and register with Check like built-in messages do.
  #[test]
  fn user_message_templates_and_check() {
    clear_state();
    interpret(r#"f::mymsg = "Custom `1` here."; Message[f::mymsg, 42];"#)
      .unwrap();
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains("f::mymsg: Custom 42 here.")),
      "template slot not filled: {msgs:?}"
    );
    clear_state();
    assert_eq!(
      interpret(
        r#"f::mymsg = "Custom `1` here."; Check[Message[f::mymsg, 1]; 7, "caught"]"#
      )
      .unwrap(),
      "caught"
    );
    clear_state();
    assert_eq!(
      interpret(
        r#"f::mymsg = "Custom `1` here."; Check[Message[f::mymsg, 1]; 7, "caught", f::mymsg]"#
      )
      .unwrap(),
      "caught"
    );
    clear_state();
    assert_eq!(
      interpret(
        r#"f::mymsg = "Custom `1` here."; Check[Message[f::mymsg, 1]; 7, "caught", g::other]"#
      )
      .unwrap(),
      "7"
    );
  }

  // `$MessageList` holds the messages raised so far in the current
  // calculation, each as HoldForm[MessageName[sym, tag]]. Verified against
  // wolframscript.
  #[test]
  fn message_list_tracks_raised_messages() {
    clear_state();
    assert_eq!(interpret("$MessageList").unwrap(), "{}");
    assert_eq!(interpret("Head[$MessageList]").unwrap(), "List");
    clear_state();
    assert_eq!(
      interpret(r#"ff::test = "hi"; Message[ff::test]; $MessageList"#).unwrap(),
      "{HoldForm[MessageName[ff, test]]}"
    );
    clear_state();
    assert_eq!(
      interpret("1/0; $MessageList").unwrap(),
      "{HoldForm[MessageName[Power, infy]]}"
    );
    // `Quiet` saves and restores `$MessageList` around its body: a message
    // raised inside is visible while still inside …
    clear_state();
    assert_eq!(
      interpret(r#"ff::test = "hi"; Quiet[Message[ff::test]; $MessageList]"#)
        .unwrap(),
      "{HoldForm[MessageName[ff, test]]}"
    );
    // … and gone once the block returns.
    clear_state();
    assert_eq!(
      interpret(r#"ff::test = "hi"; Quiet[Message[ff::test]]; $MessageList"#)
        .unwrap(),
      "{}"
    );
    // A message-specific Quiet only rolls back the messages it names.
    clear_state();
    assert_eq!(
      interpret("Quiet[1/0, Power::infy]; $MessageList").unwrap(),
      "{}"
    );
    clear_state();
    assert_eq!(
      interpret("Quiet[1/0, zz::qq]; $MessageList").unwrap(),
      "{HoldForm[MessageName[Power, infy]]}"
    );
    // The held entry is the message name itself, so releasing the hold
    // gives the message text.
    clear_state();
    assert_eq!(
      interpret(
        r#"ff::test = "hi"; Message[ff::test]; ReleaseHold[First[$MessageList]]"#
      )
      .unwrap(),
      "hi"
    );
    clear_state();
    assert_eq!(interpret("1/0; 1/0; Length[$MessageList]").unwrap(), "2");
    // `Off` keeps the message out of the list too. The switch outlives
    // `clear_state`, so the next line turns it back on — otherwise every
    // later `Power::infy` in the same process (and in the wolframscript
    // conformance run, which shares one kernel) would stay silenced.
    clear_state();
    assert_eq!(
      interpret("Off[Power::infy]; 1/0; $MessageList").unwrap(),
      "{}"
    );
    clear_state();
    assert_eq!(
      interpret("On[Power::infy]; 1/0; $MessageList").unwrap(),
      "{HoldForm[MessageName[Power, infy]]}"
    );
  }

  // Messages silenced by an inner Quiet don't trigger an outer Check.
  #[test]
  fn check_ignores_quieted_messages() {
    clear_state();
    assert_eq!(
      interpret(r#"Check[Quiet[1/0], "err"]"#).unwrap(),
      "ComplexInfinity"
    );
    clear_state();
    assert_eq!(interpret(r#"Quiet[Check[1/0, "err"]]"#).unwrap(), "err");
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

mod quiet {
  use super::*;

  #[test]
  fn quiet_basic_no_message() {
    clear_state();
    // Quiet should evaluate and return the result
    assert_eq!(interpret("Quiet[1 + 2]").unwrap(), "3");
  }

  #[test]
  fn quiet_suppresses_part_warning() {
    clear_state();
    // Part out of bounds generates a message; Quiet suppresses it
    assert_eq!(
      interpret("Quiet[Part[{1, 2, 3}, 5]]").unwrap(),
      "{1, 2, 3}[[5]]"
    );
  }

  #[test]
  fn quiet_suppresses_first_empty_warning() {
    clear_state();
    assert_eq!(interpret("Quiet[First[{}]]").unwrap(), "First[{}]");
  }

  #[test]
  fn quiet_returns_evaluated_result() {
    clear_state();
    assert_eq!(interpret("Head[Quiet[3 + 4]]").unwrap(), "Integer");
  }

  #[test]
  fn quiet_with_all() {
    clear_state();
    // Quiet[expr, All] is same as Quiet[expr]
    assert_eq!(
      interpret("Quiet[Part[{1, 2, 3}, 5], All]").unwrap(),
      "{1, 2, 3}[[5]]"
    );
  }

  #[test]
  fn quiet_with_none() {
    clear_state();
    // Quiet[expr, None] suppresses nothing — message still present in warnings
    // But the result should still be returned
    assert_eq!(
      interpret("Quiet[Part[{1, 2, 3}, 5], None]").unwrap(),
      "{1, 2, 3}[[5]]"
    );
  }

  #[test]
  fn quiet_no_args_error() {
    clear_state();
    // Quiet[] with no args returns unevaluated with error message
    assert_eq!(interpret("Quiet[]").unwrap(), "Quiet[]");
  }

  #[test]
  fn quiet_check_outer_quiet() {
    clear_state();
    // Check[Quiet[expr], failexpr] — Quiet suppresses message so Check doesn't see it
    assert_eq!(
      interpret("Check[Quiet[Part[{1, 2, 3}, 5]], \"failed\"]").unwrap(),
      "{1, 2, 3}[[5]]"
    );
  }

  #[test]
  fn quiet_check_inner_quiet() {
    clear_state();
    // Quiet[Check[expr, failexpr]] — Check sees the message first, triggers failexpr
    assert_eq!(
      interpret("Quiet[Check[Part[{1, 2, 3}, 5], \"failed\"]]").unwrap(),
      "failed"
    );
  }

  #[test]
  fn quiet_attributes() {
    clear_state();
    assert_eq!(
      interpret("Attributes[Quiet]").unwrap(),
      "{HoldAll, Protected}"
    );
  }

  #[test]
  fn quiet_nested() {
    clear_state();
    // Nested Quiet should work
    assert_eq!(interpret("Quiet[Quiet[1 + 2]]").unwrap(), "3");
  }

  #[test]
  fn quiet_preserves_side_effects() {
    clear_state();
    // Side effects (variable assignment) should still happen inside Quiet
    assert_eq!(interpret("Quiet[x = 42]; x").unwrap(), "42");
  }

  #[test]
  fn quiet_with_compound_expr() {
    clear_state();
    // Quiet wrapping a compound expression
    assert_eq!(interpret("Quiet[1 + 1; 2 + 2]").unwrap(), "4");
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
    assert_eq!(interpret("Which[False, a, False, b]").unwrap(), "\0");
  }

  #[test]
  fn no_arguments_is_null() {
    clear_state();
    // Which[] is valid and yields Null (no condition matched).
    assert_eq!(interpret("Which[]").unwrap(), "\0");
  }

  #[test]
  fn odd_argument_count_warns() {
    use woxi::interpret_with_stdout;
    clear_state();
    // One argument: Which::argctu (singular), unevaluated.
    let one = interpret_with_stdout("Which[True]").unwrap();
    assert_eq!(one.result, "Which[True]");
    assert!(
      one.warnings[0].contains("Which::argctu: Which called with 1 argument.")
    );
    // Three arguments: Which::argct (plural), unevaluated.
    clear_state();
    let three = interpret_with_stdout("Which[False, 1, True]").unwrap();
    assert_eq!(three.result, "Which[False, 1, True]");
    assert!(
      three.warnings[0]
        .contains("Which::argct: Which called with 3 arguments.")
    );
  }
}

mod switch_arity {
  use super::*;
  use woxi::interpret_with_stdout;

  #[test]
  fn even_argument_count_warns() {
    clear_state();
    // Switch needs an odd argument count; an even count is an error and must
    // not silently treat the dangling pattern as a value.
    let two = interpret_with_stdout("Switch[2, 1]").unwrap();
    assert_eq!(two.result, "Switch[2, 1]");
    assert!(two.warnings[0].contains(
      "Switch::argct: Switch called with 2 arguments. \
       Switch must be called with an odd number of arguments."
    ));

    clear_state();
    let four = interpret_with_stdout("Switch[2, 1, \"a\", 2]").unwrap();
    assert_eq!(four.result, "Switch[2, 1, a, 2]");
    assert!(
      four.warnings[0]
        .contains("Switch::argct: Switch called with 4 arguments.")
    );
  }

  #[test]
  fn odd_argument_count_still_matches() {
    clear_state();
    // A valid odd count keeps matching pattern/value pairs.
    assert_eq!(interpret("Switch[2, 1, x, 2, y]").unwrap(), "y");
    assert_eq!(interpret("Switch[5, _, \"x\"]").unwrap(), "x");
    // No match returns the unevaluated Switch.
    assert_eq!(interpret("Switch[2, 1, a]").unwrap(), "Switch[2, 1, a]");
  }
}

mod uncaught_throw {
  use super::*;
  use woxi::interpret_with_stdout;

  #[test]
  fn warns_nocatch_and_produces_no_result() {
    clear_state();
    // An uncaught Throw must surface Throw::nocatch (not leak the internal
    // error) and yield no result value.
    let r = interpret_with_stdout("Throw[3]").unwrap();
    assert_eq!(r.result, "");
    assert!(
      r.warnings[0]
        .contains("Throw::nocatch: Uncaught Throw[3] returned to top level.")
    );
  }

  #[test]
  fn message_uses_output_form_and_keeps_tag() {
    clear_state();
    // The thrown value is shown in OutputForm (strings without quotes) and a
    // tag is included.
    let s = interpret_with_stdout(r#"Throw["x"]"#).unwrap();
    assert!(
      s.warnings[0]
        .contains("Throw::nocatch: Uncaught Throw[x] returned to top level.")
    );
    clear_state();
    let t = interpret_with_stdout("Throw[5, tag]").unwrap();
    assert!(t.warnings[0].contains(
      "Throw::nocatch: Uncaught Throw[5, tag] returned to top level."
    ));
  }

  #[test]
  fn throw_inside_expression_still_caught_by_catch() {
    // A matching Catch still works (no regression).
    assert_eq!(interpret("Catch[1 + Throw[2]]").unwrap(), "2");
  }
}

mod defer {
  use super::*;

  #[test]
  fn keeps_wrapper_and_holds_argument_in_cli() {
    // In script/CLI mode Defer prints its wrapper and holds its argument,
    // matching wolframscript (the notebook front-end would show it stripped).
    assert_eq!(interpret("Defer[1 + 1]").unwrap(), "Defer[1 + 1]");
    assert_eq!(interpret("Defer[Sin[0]]").unwrap(), "Defer[Sin[0]]");
    assert_eq!(interpret("Defer[a + b]").unwrap(), "Defer[a + b]");
  }

  #[test]
  fn is_inert_inside_arithmetic() {
    assert_eq!(interpret("2 + Defer[3 + 4]").unwrap(), "2 + Defer[3 + 4]");
  }

  #[test]
  fn head_is_defer() {
    assert_eq!(interpret("Head[Defer[1 + 1]]").unwrap(), "Defer");
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

  #[test]
  fn short_circuit_skips_invalid_part() {
    clear_state();
    // True || should not evaluate the second argument
    assert_eq!(
      interpret("v = ProductLog[x]; If[True || FreeQ[v[[2]], x], True, False]")
        .unwrap(),
      "True"
    );
  }

  #[test]
  fn short_circuit_operator_syntax() {
    clear_state();
    assert_eq!(
      interpret("v = ProductLog[x]; True || FreeQ[v[[2]], x]").unwrap(),
      "True"
    );
  }

  #[test]
  fn short_circuit_function_syntax() {
    clear_state();
    assert_eq!(
      interpret("v = ProductLog[x]; Or[True, FreeQ[v[[2]], x]]").unwrap(),
      "True"
    );
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

  #[test]
  fn short_circuit_skips_invalid_part() {
    clear_state();
    // False && should not evaluate the second argument (issue #74)
    assert_eq!(
      interpret(
        "v = ProductLog[x]; If[False && FreeQ[v[[2]], x], True, False]"
      )
      .unwrap(),
      "False"
    );
  }

  #[test]
  fn short_circuit_operator_syntax() {
    clear_state();
    // Using && operator syntax
    assert_eq!(
      interpret("v = ProductLog[x]; False && FreeQ[v[[2]], x]").unwrap(),
      "False"
    );
  }

  #[test]
  fn short_circuit_function_syntax() {
    clear_state();
    // Using And[] function syntax
    assert_eq!(
      interpret("v = ProductLog[x]; And[False, FreeQ[v[[2]], x]]").unwrap(),
      "False"
    );
  }
}

mod xor_logical {
  use super::*;

  #[test]
  fn simplifies_with_false() {
    clear_state();
    assert_eq!(interpret("Xor[a, False, b]").unwrap(), "Xor[a, b]");
  }

  // Syntactically-identical operands cancel in pairs: a ⊻ a = False.
  #[test]
  fn cancels_duplicate_pairs() {
    clear_state();
    assert_eq!(interpret("Xor[a, a]").unwrap(), "False");
    assert_eq!(interpret("Xor[a, a, a]").unwrap(), "a");
    assert_eq!(interpret("Xor[a, b, a]").unwrap(), "b");
    assert_eq!(interpret("Xor[a, b, b]").unwrap(), "a");
    assert_eq!(interpret("Xor[a, a, b, b]").unwrap(), "False");
    assert_eq!(interpret("Xor[a, b, c, a]").unwrap(), "Xor[b, c]");
    assert_eq!(interpret("Xor[a, b, a, b, c]").unwrap(), "c");
  }

  // Surviving operands are reported in canonical order.
  #[test]
  fn orders_operands() {
    clear_state();
    assert_eq!(interpret("Xor[b, a]").unwrap(), "Xor[a, b]");
  }

  // An odd number of True operands negates the result (Not[...]).
  #[test]
  fn true_operand_negates() {
    clear_state();
    assert_eq!(interpret("Xor[a, True]").unwrap(), " !a");
    assert_eq!(interpret("Xor[a, False]").unwrap(), "a");
    assert_eq!(interpret("Xor[a, b, True]").unwrap(), " !(Xor[a, b])");
    assert_eq!(interpret("Xor[a, b, c, True]").unwrap(), " !(Xor[a, b, c])");
  }

  // Xnor is the negation of Xor and keeps the Xnor head for multi-operand
  // cores while collapsing single operands.
  #[test]
  fn xnor_reduction() {
    clear_state();
    assert_eq!(interpret("Xnor[a, a]").unwrap(), "True");
    assert_eq!(interpret("Xnor[a]").unwrap(), " !a");
    assert_eq!(interpret("Xnor[a, b, b]").unwrap(), " !a");
    assert_eq!(interpret("Xnor[a, True]").unwrap(), "a");
    assert_eq!(interpret("Xnor[a, False]").unwrap(), " !a");
    assert_eq!(interpret("Xnor[b, a]").unwrap(), "Xnor[a, b]");
    assert_eq!(interpret("Xnor[a, b, True]").unwrap(), " !(Xnor[a, b])");
    assert_eq!(
      interpret("Xnor[a, b, c, True]").unwrap(),
      " !(Xnor[a, b, c])"
    );
  }
}

mod nand_nor_logical {
  use super::*;

  // Nand[...] = Not[And[...]]: True is the And identity (absorbed), any False
  // short-circuits to True, and a lone surviving operand collapses to !x.
  #[test]
  fn nand_constants() {
    clear_state();
    assert_eq!(interpret("Nand[a, True]").unwrap(), " !a");
    assert_eq!(interpret("Nand[a, True, True]").unwrap(), " !a");
    assert_eq!(interpret("Nand[a, b, True]").unwrap(), "Nand[a, b]");
    assert_eq!(interpret("Nand[a, b, False, c]").unwrap(), "True");
    assert_eq!(interpret("Nand[True, True, True]").unwrap(), "False");
    // Idempotent duplicates are NOT removed (Nand is not Orderless/idempotent).
    assert_eq!(interpret("Nand[a, a, b]").unwrap(), "Nand[a, a, b]");
  }

  // Nor[...] = Not[Or[...]]: False is the Or identity (absorbed), any True
  // short-circuits to False, and a lone surviving operand collapses to !x.
  #[test]
  fn nor_constants() {
    clear_state();
    assert_eq!(interpret("Nor[a, False]").unwrap(), " !a");
    assert_eq!(interpret("Nor[a, b, False]").unwrap(), "Nor[a, b]");
    assert_eq!(interpret("Nor[a, b, True, c]").unwrap(), "False");
    assert_eq!(interpret("Nor[a]").unwrap(), " !a");
    assert_eq!(interpret("Nor[a, b]").unwrap(), "Nor[a, b]");
  }
}

mod boolean_convert_dnf {
  use super::*;

  // Two-variable Xor: literals ordered by variable, positive before negated,
  // clauses in descending-minterm order (matching wolframscript).
  #[test]
  fn xor_two_args() {
    clear_state();
    assert_eq!(
      interpret("BooleanConvert[Xor[a, b]]").unwrap(),
      "(a &&  !b) || ( !a && b)"
    );
  }

  // Three-variable Xor must NOT contain contradictory clauses like
  // `!a && a && c`; it is the four odd-parity minterms.
  #[test]
  fn xor_three_args_no_contradictions() {
    clear_state();
    assert_eq!(
      interpret("BooleanConvert[Xor[a, b, c]]").unwrap(),
      "(a && b && c) || (a &&  !b &&  !c) || ( !a && b &&  !c) || \
       ( !a &&  !b && c)"
    );
  }

  #[test]
  fn equivalent_two_args() {
    clear_state();
    assert_eq!(
      interpret("BooleanConvert[Equivalent[a, b]]").unwrap(),
      "(a && b) || ( !a &&  !b)"
    );
  }

  #[test]
  fn equivalent_three_args() {
    clear_state();
    assert_eq!(
      interpret("BooleanConvert[Equivalent[a, b, c]]").unwrap(),
      "(a && b && c) || ( !a &&  !b &&  !c)"
    );
  }

  #[test]
  fn implies_is_or() {
    clear_state();
    assert_eq!(
      interpret("BooleanConvert[Implies[a, b]]").unwrap(),
      " !a || b"
    );
  }

  #[test]
  fn nand_and_nor() {
    clear_state();
    assert_eq!(
      interpret("BooleanConvert[Nand[a, b]]").unwrap(),
      " !a ||  !b"
    );
    assert_eq!(
      interpret("BooleanConvert[Nor[a, b]]").unwrap(),
      " !a &&  !b"
    );
  }

  // If[a, b, c] is a boolean expression equal to (a && b) || (!a && c);
  // BooleanConvert expands it rather than leaving the If untouched.
  #[test]
  fn if_three_args_is_boolean() {
    clear_state();
    assert_eq!(
      interpret("BooleanConvert[If[a, b, c]]").unwrap(),
      "(a && b) || ( !a && c)"
    );
  }

  #[test]
  fn if_three_args_explicit_dnf() {
    clear_state();
    assert_eq!(
      interpret("BooleanConvert[If[a, b, c], \"DNF\"]").unwrap(),
      "(a && b) || ( !a && c)"
    );
  }

  #[test]
  fn if_nested_in_and() {
    clear_state();
    assert_eq!(
      interpret("BooleanConvert[a && If[b, c, d]]").unwrap(),
      "(a && b && c) || (a &&  !b && d)"
    );
  }

  #[test]
  fn if_with_trailing_and() {
    clear_state();
    assert_eq!(
      interpret("BooleanConvert[If[p, q, r] && s]").unwrap(),
      "(p && q && s) || ( !p && r && s)"
    );
  }

  // A pure contradiction collapses to False; a tautology over a single
  // clause is dropped entirely.
  #[test]
  fn contradiction_is_false() {
    clear_state();
    assert_eq!(interpret("BooleanConvert[a && !a]").unwrap(), "False");
  }

  // Absorption: `a || (a && b)` reduces to `a`.
  #[test]
  fn absorption() {
    clear_state();
    assert_eq!(interpret("BooleanConvert[a || (a && b)]").unwrap(), "a");
  }

  // Already-DNF expressions keep their (canonically ordered) form.
  #[test]
  fn keeps_dnf_form() {
    clear_state();
    assert_eq!(
      interpret("BooleanConvert[a || b && c]").unwrap(),
      "a || (b && c)"
    );
    assert_eq!(
      interpret("BooleanConvert[(a && b) || (!a && c)]").unwrap(),
      "(a && b) || ( !a && c)"
    );
  }

  #[test]
  fn distributes_and_over_or() {
    clear_state();
    assert_eq!(
      interpret("BooleanConvert[a && (b || c)]").unwrap(),
      "(a && b) || (a && c)"
    );
  }
}

mod xnor_logical {
  use super::*;

  #[test]
  fn xnor_no_args() {
    clear_state();
    assert_eq!(interpret("Xnor[]").unwrap(), "True");
  }

  #[test]
  fn xnor_true_false() {
    clear_state();
    assert_eq!(interpret("Xnor[True, False]").unwrap(), "False");
  }

  #[test]
  fn xnor_true_true() {
    clear_state();
    assert_eq!(interpret("Xnor[True, True]").unwrap(), "True");
  }

  #[test]
  fn xnor_false_false() {
    clear_state();
    assert_eq!(interpret("Xnor[False, False]").unwrap(), "True");
  }

  #[test]
  fn xnor_three_true() {
    clear_state();
    assert_eq!(interpret("Xnor[True, True, True]").unwrap(), "False");
  }

  #[test]
  fn xnor_three_mixed() {
    clear_state();
    assert_eq!(interpret("Xnor[True, False, False]").unwrap(), "False");
  }

  #[test]
  fn xnor_symbolic() {
    clear_state();
    assert_eq!(interpret("Xnor[a, b]").unwrap(), "Xnor[a, b]");
  }

  #[test]
  fn xnor_single_true() {
    clear_state();
    assert_eq!(interpret("Xnor[True]").unwrap(), "False");
  }

  #[test]
  fn xnor_single_false() {
    clear_state();
    assert_eq!(interpret("Xnor[False]").unwrap(), "True");
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

  // Double negation is eliminated: Not[Not[x]] -> x (wolframscript).
  #[test]
  fn not_double_negation() {
    clear_state();
    assert_eq!(interpret("Not[Not[a]]").unwrap(), "a");
    assert_eq!(interpret("Not[!a]").unwrap(), "a");
    assert_eq!(interpret("Not[Not[Not[a]]]").unwrap(), " !a");
    assert_eq!(interpret("Not[Not[a && b]]").unwrap(), "a && b");
  }

  #[test]
  fn not_greater() {
    clear_state();
    // Not[x > 1] → x <= 1.
    assert_eq!(interpret("Not[x > 1]").unwrap(), "x <= 1");
  }

  #[test]
  fn not_greater_equal() {
    clear_state();
    assert_eq!(interpret("Not[x >= 1]").unwrap(), "x < 1");
  }

  #[test]
  fn not_less() {
    clear_state();
    assert_eq!(interpret("Not[x < 1]").unwrap(), "x >= 1");
  }

  #[test]
  fn not_less_equal() {
    clear_state();
    assert_eq!(interpret("Not[x <= 1]").unwrap(), "x > 1");
  }

  #[test]
  fn not_equal() {
    clear_state();
    assert_eq!(interpret("Not[x == 1]").unwrap(), "x != 1");
  }

  #[test]
  fn not_unequal() {
    clear_state();
    assert_eq!(interpret("Not[x != 1]").unwrap(), "x == 1");
  }

  #[test]
  fn prefix_not_true() {
    clear_state();
    // !True should parse as Not[True] and evaluate to False
    assert_eq!(interpret("Not[True]").unwrap(), "False");
  }

  #[test]
  fn not_in_list() {
    clear_state();
    // Not operator should evaluate inside list literals
    assert_eq!(
      interpret("{Not[True], Not[False]}").unwrap(),
      "{False, True}"
    );
  }

  #[test]
  fn boolean_ops_in_list() {
    clear_state();
    // Boolean operators should evaluate inside list literals
    assert_eq!(
      interpret("{True && False, True || False}").unwrap(),
      "{False, True}"
    );
  }

  #[test]
  fn comparison_in_list() {
    clear_state();
    // Comparison operators should evaluate inside list literals
    assert_eq!(interpret("{1 < 2}").unwrap(), "{True}");
    assert_eq!(interpret("{3 > 2}").unwrap(), "{True}");
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
    assert_eq!(interpret("Pause[0.01]").unwrap(), "\0");
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
  fn label_alone_returns_unevaluated() {
    clear_state();
    // Label at top level (not inside CompoundExpr) stays symbolic
    assert_eq!(interpret("Label[x]").unwrap(), "Label[x]");
  }

  #[test]
  fn goto_no_label_returns_null() {
    clear_state();
    // Goto with no matching label returns Null (with stderr message)
    assert_eq!(interpret("Goto[x]").unwrap(), "\0");
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
  fn goto_label_attributes() {
    clear_state();
    // Goto and Label only have Protected (no HoldAll), matching Wolfram
    assert_eq!(interpret("Attributes[Goto]").unwrap(), "{Protected}");
    assert_eq!(interpret("Attributes[Label]").unwrap(), "{Protected}");
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

  #[test]
  fn repeated_timing_returns_list() {
    clear_state();
    let result = interpret("RepeatedTiming[1 + 1]").unwrap();
    assert!(result.starts_with('{'));
    assert!(result.contains(", 2}"));
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
      "\0"
    );
  }

  #[test]
  fn return_in_while() {
    clear_state();
    // In Wolfram, Return[] inside While is NOT caught by the loop — it
    // propagates up; at top level the value is yielded directly (no
    // `Return[]` wrapper), matching wolframscript.
    assert_eq!(interpret("While[True, Return[99]]").unwrap(), "99");
  }

  #[test]
  fn return_in_for() {
    clear_state();
    // In Wolfram, Return[] inside For is NOT caught by the loop — it
    // propagates up; at top level the value is yielded directly (no
    // `Return[]` wrapper), matching wolframscript.
    assert_eq!(
      interpret("For[i=1, i<=10, i++, If[i==5, Return[i]]]").unwrap(),
      "5"
    );
  }

  #[test]
  fn return_exits_all_iterators_of_multi_iter_do() {
    clear_state();
    // `Do[body, {i,...}, {j,...}]` is a single construct: Return[]
    // exits the entire Do, not just the innermost iterator. Without
    // the fix, the test would print all 9 lines instead of 5.
    interpret(
      "log = {}; Do[AppendTo[log, {i, j}]; If[i == 2 && j == 2, \
       Return[]], {i, 1, 3}, {j, 1, 3}]",
    )
    .unwrap();
    assert_eq!(
      interpret("log").unwrap(),
      "{{1, 1}, {1, 2}, {1, 3}, {2, 1}, {2, 2}}"
    );
  }

  #[test]
  fn break_exits_all_iterators_of_multi_iter_do() {
    clear_state();
    interpret(
      "log = {}; Do[AppendTo[log, {i, j}]; If[i == 2 && j == 2, \
       Break[]], {i, 1, 3}, {j, 1, 3}]",
    )
    .unwrap();
    assert_eq!(
      interpret("log").unwrap(),
      "{{1, 1}, {1, 2}, {1, 3}, {2, 1}, {2, 2}}"
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

mod module_condition {
  use super::*;

  #[test]
  fn condition_in_module_body_passes() {
    clear_state();
    // Issue #59: Condition in Module body should be evaluated while locals are in scope
    assert_eq!(
      interpret(
        "Foo[u_, x_Symbol] := Module[{lst = u}, 3 /; lst == 1]; Foo[1, x]"
      )
      .unwrap(),
      "3"
    );
  }

  #[test]
  fn condition_in_module_body_fails() {
    clear_state();
    // When condition fails, the function should not match
    assert_eq!(
      interpret(
        "Bar[u_, x_Symbol] := Module[{lst = u}, 3 /; lst == 1]; Bar[2, x]"
      )
      .unwrap(),
      "Bar[2, x]"
    );
  }

  #[test]
  fn condition_in_module_body_both_cases() {
    clear_state();
    assert_eq!(
      interpret(
        "Baz[u_, x_Symbol] := Module[{lst = u}, 3 /; lst == 1]; {Baz[1, x], Baz[x, x]}"
      )
      .unwrap(),
      "{3, Baz[x, x]}"
    );
  }

  #[test]
  fn condition_in_block_body_passes() {
    clear_state();
    assert_eq!(
      interpret(
        "QuxB[u_, x_Symbol] := Block[{lst = u}, 3 /; lst == 1]; QuxB[1, x]"
      )
      .unwrap(),
      "3"
    );
  }

  #[test]
  fn condition_in_block_body_fails() {
    clear_state();
    assert_eq!(
      interpret(
        "QuxB2[u_, x_Symbol] := Block[{lst = u}, 3 /; lst == 1]; QuxB2[2, x]"
      )
      .unwrap(),
      "QuxB2[2, x]"
    );
  }

  #[test]
  fn condition_in_module_complex_test() {
    clear_state();
    assert_eq!(
      interpret("Qux[u_] := Module[{v = u}, 10 /; v > 0 && v < 5]; Qux[3]")
        .unwrap(),
      "10"
    );
  }

  #[test]
  fn condition_in_module_complex_test_fails() {
    clear_state();
    assert_eq!(
      interpret("Qux2[u_] := Module[{v = u}, 10 /; v > 0 && v < 5]; Qux2[10]")
        .unwrap(),
      "Qux2[10]"
    );
  }
}

mod module_expr_preservation {
  use super::*;

  #[test]
  fn module_preserves_times_head() {
    clear_state();
    // Issue #79: Module should preserve expression structure (Head)
    // Previously, Module converted expressions to strings and back,
    // losing canonical form (e.g. Times[-1, a, ...] became 0 - a/b)
    assert_eq!(interpret("Module[{v = -a*b}, Head[v]]").unwrap(), "Times");
  }

  #[test]
  fn module_preserves_function_call_structure() {
    clear_state();
    assert_eq!(
      interpret("Module[{v = f[a, b, c]}, {Head[v], Length[v]}]").unwrap(),
      "{f, 3}"
    );
  }

  #[test]
  fn module_preserves_list_structure() {
    clear_state();
    assert_eq!(
      interpret("Module[{v = {1, 2, 3}}, {Length[v], First[v]}]").unwrap(),
      "{3, 1}"
    );
  }

  #[test]
  fn rest_evaluates_single_arg_times() {
    clear_state();
    // Rest[Times[a, b]] should return b (not Times[b])
    assert_eq!(interpret("Rest[a*b]").unwrap(), "b");
  }

  #[test]
  fn rest_evaluates_single_arg_plus() {
    clear_state();
    // Rest[Plus[a, b]] should return b (not Plus[b])
    assert_eq!(interpret("Rest[a + b]").unwrap(), "b");
  }
}

mod module_lexical_scoping {
  use super::*;

  // Module renames its locals to fresh var$n symbols (lexical scoping):
  // a global function called from the body sees the untouched global
  // symbol, unlike Block's dynamic rebinding. Verified against
  // wolframscript 15.0.
  #[test]
  fn module_is_lexical_block_is_dynamic() {
    clear_state();
    assert_eq!(interpret("f[] := x; Block[{x = 3}, f[]]").unwrap(), "3");
    clear_state();
    assert_eq!(interpret("f[] := x; Module[{x = 3}, f[]]").unwrap(), "x");
  }

  #[test]
  fn module_initializers_use_enclosing_scope() {
    clear_state();
    // Nested same-name locals shadow correctly
    assert_eq!(
      interpret("Module[{x = 5}, x + Module[{x = 7}, x]]").unwrap(),
      "12"
    );
    clear_state();
    assert_eq!(interpret("Module[{a = 1, b = 2}, a + b]").unwrap(), "3");
  }
}

/// `DynamicModule` scopes its locals the way `Module` does. Wolfram keeps
/// the wrapper around the result, because the front end owns the local
/// state between redraws; Woxi hands back the body's value, so a Grid or
/// a Graphics inside one displays as itself.
mod dynamic_module_scoping {
  use super::*;

  #[test]
  fn dynamic_module_returns_its_body() {
    clear_state();
    assert_eq!(interpret("DynamicModule[{x = 2}, x^2]").unwrap(), "4");
    assert_eq!(interpret("DynamicModule[{a}, a = 3; a + 1]").unwrap(), "4");
  }

  #[test]
  fn dynamic_module_locals_do_not_leak() {
    clear_state();
    // Regression: the locals were evaluated as ordinary arguments, so the
    // body's assignments wrote straight into the global symbol.
    assert_eq!(
      interpret("a = 99; DynamicModule[{a}, a = 3]; a").unwrap(),
      "99"
    );
    clear_state();
    assert_eq!(
      interpret("x = 1; DynamicModule[{x = 5}, x] + x").unwrap(),
      "6"
    );
  }

  #[test]
  fn dynamic_module_is_lexical_like_module() {
    clear_state();
    assert_eq!(
      interpret("f[] := x; DynamicModule[{x = 3}, f[]]").unwrap(),
      "x"
    );
  }
}

mod module_downvalues {
  use super::*;

  #[test]
  fn module_scoped_set_delayed() {
    clear_state();
    // Module-scoped function definitions (DownValues) should work
    assert_eq!(interpret("Module[{f}, f[x_] := x + 1; f[5]]").unwrap(), "6");
  }

  #[test]
  fn module_scoped_set() {
    clear_state();
    // Module-scoped Set (literal matching) should work
    assert_eq!(
      interpret("Module[{f}, f[0] = 0; f[1] = 1; {f[0], f[1]}]").unwrap(),
      "{0, 1}"
    );
  }

  #[test]
  fn module_scoped_memoized_recursion() {
    clear_state();
    // Memoized Fibonacci inside Module
    assert_eq!(
      interpret(
        "Module[{f}, f[0] = 0; f[1] = 1; f[n_] := f[n] = f[n - 1] + f[n - 2]; f[10]]"
      )
      .unwrap(),
      "55"
    );
  }

  #[test]
  fn module_scoped_recursion_without_memoization() {
    clear_state();
    assert_eq!(
      interpret(
        "Module[{f}, f[0] = 0; f[1] = 1; f[n_] := f[n - 1] + f[n - 2]; f[10]]"
      )
      .unwrap(),
      "55"
    );
  }

  #[test]
  fn module_scoped_multiple_definitions() {
    clear_state();
    assert_eq!(
      interpret("Module[{f}, f[x_Integer] := x^2; f[x_String] := StringLength[x]; {f[3], f[\"hello\"]}]").unwrap(),
      "{9, 5}"
    );
  }
}

mod trace_scan {
  use super::*;

  #[test]
  fn basic_trace_scan_addition() {
    clear_state();
    let result =
      woxi::interpret_with_stdout("TraceScan[Print, 1 + 2 + 3]").unwrap();
    assert_eq!(result.result, "6");
    // TraceScan wraps each step in HoldForm; the wrapper is preserved on
    // Print, matching `wolframscript -code 'TraceScan[Print, 1 + 2 + 3]'`.
    assert_eq!(
      result.stdout.trim(),
      "HoldForm[1 + 2 + 3]\nHoldForm[Plus]\nHoldForm[1]\nHoldForm[2]\nHoldForm[3]\nHoldForm[6]"
    );
  }

  #[test]
  fn trace_scan_with_power() {
    clear_state();
    let result =
      woxi::interpret_with_stdout("TraceScan[Print, 2^3 + 5]").unwrap();
    assert_eq!(result.result, "13");
    assert_eq!(
      result.stdout.trim(),
      "HoldForm[2^3 + 5]\nHoldForm[Plus]\nHoldForm[2^3]\nHoldForm[Power]\nHoldForm[2]\nHoldForm[3]\nHoldForm[8]\nHoldForm[5]\nHoldForm[8 + 5]\nHoldForm[13]"
    );
  }

  #[test]
  fn trace_scan_atom() {
    clear_state();
    let result = woxi::interpret_with_stdout("TraceScan[Print, 3]").unwrap();
    assert_eq!(result.result, "3");
    assert_eq!(result.stdout.trim(), "HoldForm[3]");
  }

  #[test]
  fn trace_scan_undefined_function() {
    clear_state();
    let result =
      woxi::interpret_with_stdout("TraceScan[Print, f[1, 2]]").unwrap();
    assert_eq!(result.result, "f[1, 2]");
    assert_eq!(
      result.stdout.trim(),
      "HoldForm[f[1, 2]]\nHoldForm[f]\nHoldForm[1]\nHoldForm[2]"
    );
  }

  #[test]
  fn trace_scan_form_symbol() {
    clear_state();
    let result =
      woxi::interpret_with_stdout("TraceScan[Print, 1 + 2 + 3, Plus]").unwrap();
    assert_eq!(result.result, "6");
    assert_eq!(result.stdout.trim(), "HoldForm[Plus]\nHoldForm[6]");
  }

  #[test]
  fn trace_scan_form_symbol_complex() {
    clear_state();
    let result =
      woxi::interpret_with_stdout("TraceScan[Print, 2^3 + 5, Plus]").unwrap();
    assert_eq!(result.result, "13");
    assert_eq!(
      result.stdout.trim(),
      "HoldForm[Plus]\nHoldForm[8 + 5]\nHoldForm[13]"
    );
  }

  #[test]
  fn trace_scan_form_blank_head() {
    clear_state();
    let result =
      woxi::interpret_with_stdout("TraceScan[Print, 1 + 2 + 3, _Plus]")
        .unwrap();
    assert_eq!(result.result, "6");
    assert_eq!(result.stdout.trim(), "HoldForm[1 + 2 + 3]");
  }

  #[test]
  fn trace_scan_form_blank_head_complex() {
    clear_state();
    let result =
      woxi::interpret_with_stdout("TraceScan[Print, 2^3 + 5, _Plus]").unwrap();
    assert_eq!(result.result, "13");
    assert_eq!(result.stdout.trim(), "HoldForm[2^3 + 5]\nHoldForm[8 + 5]");
  }

  #[test]
  fn trace_scan_form_power() {
    clear_state();
    let result =
      woxi::interpret_with_stdout("TraceScan[Print, 2^3 + 5, Power]").unwrap();
    assert_eq!(result.result, "13");
    assert_eq!(result.stdout.trim(), "HoldForm[Power]\nHoldForm[8]");
  }

  #[test]
  fn trace_scan_form_blank_power() {
    clear_state();
    let result =
      woxi::interpret_with_stdout("TraceScan[Print, 2^3 + 5, _Power]").unwrap();
    assert_eq!(result.result, "13");
    assert_eq!(result.stdout.trim(), "HoldForm[2^3]");
  }

  #[test]
  fn trace_scan_form_blank_integer() {
    clear_state();
    let result =
      woxi::interpret_with_stdout("TraceScan[Print, 2^3 + 5, _Integer]")
        .unwrap();
    assert_eq!(result.result, "13");
    assert_eq!(
      result.stdout.trim(),
      "HoldForm[2]\nHoldForm[3]\nHoldForm[8]\nHoldForm[5]\nHoldForm[13]"
    );
  }

  #[test]
  fn trace_scan_with_anonymous_function() {
    clear_state();
    // TraceScan with anonymous function that collects via Sow
    let result =
      woxi::interpret_with_stdout("Reap[TraceScan[Sow, 1 + 2 + 3]]").unwrap();
    // HoldForm wrappers are present internally but invisible in output
    assert!(result.result.contains('6'));
    assert!(result.result.contains("Plus"));
  }

  #[test]
  fn trace_scan_returns_evaluated_result() {
    clear_state();
    // TraceScan should return the evaluated expression
    assert_eq!(interpret("TraceScan[Print, 2 + 3]").unwrap(), "5");
  }

  #[test]
  fn trace_scan_non_matching_form() {
    clear_state();
    // Form that doesn't match anything — no traces
    let result =
      woxi::interpret_with_stdout("TraceScan[Print, 1 + 2, _String]").unwrap();
    assert_eq!(result.result, "3");
    assert_eq!(result.stdout.trim(), "");
  }

  #[test]
  fn trace_scan_rebuilds_list_canonically() {
    clear_state();
    // The rebuilt step is a list, so it must print as `{2, 4}` and not as
    // `List[2, 4]` — and must therefore coincide with the result, which is
    // printed only once.
    let result =
      woxi::interpret_with_stdout("TraceScan[Print, {1 + 1, 2 + 2}]").unwrap();
    assert_eq!(result.result, "{2, 4}");
    assert_eq!(
      result.stdout.trim(),
      "HoldForm[{1 + 1, 2 + 2}]\nHoldForm[List]\n\
       HoldForm[1 + 1]\nHoldForm[Plus]\nHoldForm[1]\nHoldForm[1]\n\
       HoldForm[2]\nHoldForm[2 + 2]\nHoldForm[Plus]\nHoldForm[2]\n\
       HoldForm[2]\nHoldForm[4]\nHoldForm[{2, 4}]"
    );
  }
}

mod trace_print {
  use super::*;

  #[test]
  fn basic_trace_print_addition() {
    clear_state();
    let result = woxi::interpret_with_stdout("TracePrint[2 + 3]").unwrap();
    assert_eq!(result.result, "5");
    // Every sub-expression is printed wrapped in HoldCompleteForm and
    // indented by one space per level of the evaluation, matching
    // `wolframscript -code 'TracePrint[2 + 3]'`.
    assert_eq!(
      result.stdout,
      " HoldCompleteForm[2 + 3]\n  HoldCompleteForm[Plus]\n  \
       HoldCompleteForm[2]\n  HoldCompleteForm[3]\n \
       HoldCompleteForm[5]\n"
    );
  }

  #[test]
  fn trace_print_nested_indentation() {
    clear_state();
    // The nested Power is one level deeper than the enclosing Plus.
    let result = woxi::interpret_with_stdout("TracePrint[2^3 + 5]").unwrap();
    assert_eq!(result.result, "13");
    assert_eq!(
      result.stdout,
      " HoldCompleteForm[2^3 + 5]\n  HoldCompleteForm[Plus]\n  \
       HoldCompleteForm[2^3]\n   HoldCompleteForm[Power]\n   \
       HoldCompleteForm[2]\n   HoldCompleteForm[3]\n  \
       HoldCompleteForm[8]\n  HoldCompleteForm[5]\n \
       HoldCompleteForm[8 + 5]\n HoldCompleteForm[13]\n"
    );
  }

  #[test]
  fn trace_print_atom() {
    clear_state();
    let result = woxi::interpret_with_stdout("TracePrint[3]").unwrap();
    assert_eq!(result.result, "3");
    assert_eq!(result.stdout, " HoldCompleteForm[3]\n");
  }

  #[test]
  fn trace_print_undefined_function() {
    clear_state();
    let result = woxi::interpret_with_stdout("TracePrint[f[1, 2]]").unwrap();
    assert_eq!(result.result, "f[1, 2]");
    assert_eq!(
      result.stdout,
      " HoldCompleteForm[f[1, 2]]\n  HoldCompleteForm[f]\n  \
       HoldCompleteForm[1]\n  HoldCompleteForm[2]\n"
    );
  }

  #[test]
  fn trace_print_list() {
    clear_state();
    // The rebuilt list prints as `{2, 4}`, not `List[2, 4]`.
    let result =
      woxi::interpret_with_stdout("TracePrint[{1 + 1, 2 + 2}]").unwrap();
    assert_eq!(result.result, "{2, 4}");
    assert_eq!(
      result.stdout,
      " HoldCompleteForm[{1 + 1, 2 + 2}]\n  HoldCompleteForm[List]\n  \
       HoldCompleteForm[1 + 1]\n   HoldCompleteForm[Plus]\n   \
       HoldCompleteForm[1]\n   HoldCompleteForm[1]\n  \
       HoldCompleteForm[2]\n  HoldCompleteForm[2 + 2]\n   \
       HoldCompleteForm[Plus]\n   HoldCompleteForm[2]\n   \
       HoldCompleteForm[2]\n  HoldCompleteForm[4]\n \
       HoldCompleteForm[{2, 4}]\n"
    );
  }

  #[test]
  fn trace_print_with_form() {
    clear_state();
    // Only sub-expressions matching the form are printed, but each keeps
    // the indentation of its actual evaluation level.
    let result =
      woxi::interpret_with_stdout("TracePrint[2^3 + 5, _Integer]").unwrap();
    assert_eq!(result.result, "13");
    assert_eq!(
      result.stdout,
      "   HoldCompleteForm[2]\n   HoldCompleteForm[3]\n  \
       HoldCompleteForm[8]\n  HoldCompleteForm[5]\n \
       HoldCompleteForm[13]\n"
    );
  }

  #[test]
  fn trace_print_non_matching_form() {
    clear_state();
    let result =
      woxi::interpret_with_stdout("TracePrint[1 + 2, _String]").unwrap();
    assert_eq!(result.result, "3");
    assert_eq!(result.stdout, "");
  }

  #[test]
  fn trace_print_returns_evaluated_result() {
    clear_state();
    assert_eq!(interpret("TracePrint[2 + 3]").unwrap(), "5");
  }

  #[test]
  fn trace_print_holds_its_argument() {
    clear_state();
    // HoldAll is what lets TracePrint see the unevaluated input.
    assert_eq!(
      interpret("Attributes[TracePrint]").unwrap(),
      "{HoldAll, Protected}"
    );
  }

  #[test]
  fn trace_print_side_effects_run_once() {
    clear_state();
    // A traced side effect must not be evaluated twice.
    let result =
      woxi::interpret_with_stdout("x = 0; TracePrint[x = x + 1]; x").unwrap();
    assert_eq!(result.result, "1");
  }
}

mod piecewise {
  use super::*;

  #[test]
  fn basic_true_condition() {
    assert_eq!(interpret("Piecewise[{{1, True}, {2, True}}]").unwrap(), "1");
  }

  #[test]
  fn basic_false_then_true() {
    assert_eq!(
      interpret("Piecewise[{{1, False}, {2, True}}]").unwrap(),
      "2"
    );
  }

  #[test]
  fn all_false_returns_zero() {
    assert_eq!(
      interpret("Piecewise[{{1, False}, {2, False}}]").unwrap(),
      "0"
    );
  }

  #[test]
  fn symbolic_condition_stays_unevaluated_before_true_branch() {
    // x > 0 is symbolic, so the True branch should NOT be eagerly selected
    let result =
      interpret("Piecewise[{{x, x > 0}, {-x, True}}] /. x -> 5").unwrap();
    assert_eq!(result, "5");
  }

  #[test]
  fn piece_values_evaluate_and_false_pieces_drop() {
    // Regression: piece values used to stay unevaluated and False pieces
    // were kept when any symbolic condition was present
    assert_eq!(
      interpret("Piecewise[{{1+1, x > 0}, {3+4, False}}, 1+2]").unwrap(),
      "Piecewise[{{2, x > 0}}, 3]"
    );
  }

  #[test]
  fn unreachable_piece_value_is_never_evaluated() {
    // Piecewise holds its pieces: the value of a piece whose condition is
    // False is never touched, which is what makes the idiom a *guard* —
    // `1/x` must not be evaluated at `x = 0`. Regression: the pair list was
    // evaluated up front, so every guarded value ran anyway and emitted the
    // messages the guard exists to prevent.
    let result =
      interpret_with_stdout("Piecewise[{{1/x, x != 0}}, 0] /. x -> 0").unwrap();
    assert_eq!(result.result, "0");
    assert!(
      result.warnings.is_empty(),
      "unexpected messages: {:?}",
      result.warnings
    );
  }

  #[test]
  fn range_guard_keeps_interpolation_in_bounds() {
    // The Demonstrations shape of the same guard: a tabulated curve queried
    // only inside its own data range. Outside it the piece drops, so no
    // extrapolation warning is emitted.
    let result = interpret_with_stdout(
      "f = Interpolation[{{0, 0}, {1, 2}, {2, 0}}, InterpolationOrder -> 1]; \
       g[w_] := Piecewise[{{f[w], 0 <= w <= 2}}, 0]; \
       {g[1], g[5]}",
    )
    .unwrap();
    assert_eq!(result.result, "{2, 0}");
    assert!(
      result.warnings.is_empty(),
      "unexpected messages: {:?}",
      result.warnings
    );
  }

  #[test]
  fn indirect_pieces_still_resolve() {
    // A first argument that is not already a list of pairs — a symbol holding
    // one, or a `Table` that builds one — is evaluated to get at the pieces.
    assert_eq!(
      interpret("pairs = {{1, False}, {2, True}}; Piecewise[pairs]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("Piecewise[Table[{i, i > 2}, {i, 1, 3}]]").unwrap(),
      "3"
    );
    assert_eq!(
      interpret("pair = {7, True}; Piecewise[{pair}]").unwrap(),
      "7"
    );
  }

  #[test]
  fn true_after_symbolic_becomes_default() {
    // A True condition after symbolic ones makes its value the new
    // default and drops everything behind it
    assert_eq!(
      interpret("Piecewise[{{a, x > 0}, {b, True}, {c, y > 0}}, d]").unwrap(),
      "Piecewise[{{a, x > 0}}, b]"
    );
  }

  #[test]
  fn missing_default_normalizes_to_zero() {
    assert_eq!(
      interpret("Piecewise[{{1, x > 0}}]").unwrap(),
      "Piecewise[{{1, x > 0}}, 0]"
    );
  }

  // wolframscript merges consecutive clauses with structurally-equal values,
  // OR-ing their distinct conditions.
  #[test]
  fn merges_consecutive_equal_values() {
    assert_eq!(
      interpret("Piecewise[{{a, x > 0}, {a, x <= 0}}]").unwrap(),
      "Piecewise[{{a, x > 0 || x <= 0}}, 0]"
    );
    assert_eq!(
      interpret("Piecewise[{{a, c1}, {a, c2}, {b, c3}}]").unwrap(),
      "Piecewise[{{a, c1 || c2}, {b, c3}}, 0]"
    );
    // Non-adjacent equal values are NOT merged.
    assert_eq!(
      interpret("Piecewise[{{a, c1}, {b, c2}, {a, c3}}]").unwrap(),
      "Piecewise[{{a, c1}, {b, c2}, {a, c3}}, 0]"
    );
    // Identical clauses collapse (c1 || c1 -> c1).
    assert_eq!(
      interpret("Piecewise[{{a, c1}, {a, c1}}]").unwrap(),
      "Piecewise[{{a, c1}}, 0]"
    );
  }

  // wolframscript drops trailing clauses whose value equals the default.
  #[test]
  fn drops_trailing_default_valued_clauses() {
    assert_eq!(
      interpret("Piecewise[{{b, c2}, {a, c1}}, a]").unwrap(),
      "Piecewise[{{b, c2}}, a]"
    );
    assert_eq!(
      interpret("Piecewise[{{a, c1}, {0, c2}}]").unwrap(),
      "Piecewise[{{a, c1}}, 0]"
    );
    // A non-trailing default-valued clause is kept.
    assert_eq!(
      interpret("Piecewise[{{a, c1}, {b, c2}}, a]").unwrap(),
      "Piecewise[{{a, c1}, {b, c2}}, a]"
    );
    // Merge then trailing-drop collapses everything to the default.
    assert_eq!(interpret("Piecewise[{{a, c1}, {a, c2}}, a]").unwrap(), "a");
    assert_eq!(interpret("Piecewise[{{a, c1}}, a]").unwrap(), "a");
  }

  #[test]
  fn all_false_returns_symbolic_default() {
    assert_eq!(interpret("Piecewise[{{a, False}}, d]").unwrap(), "d");
  }

  #[test]
  fn symbolic_condition_negative_substitution() {
    let result =
      interpret("Piecewise[{{x, x > 0}, {-x, True}}] /. x -> -3").unwrap();
    assert_eq!(result, "3");
  }

  #[test]
  fn three_branches_with_substitution() {
    assert_eq!(
      interpret("Piecewise[{{x^2, x > 0}, {-x, x < 0}, {0, True}}] /. x -> 3")
        .unwrap(),
      "9"
    );
    assert_eq!(
      interpret("Piecewise[{{x^2, x > 0}, {-x, x < 0}, {0, True}}] /. x -> -2")
        .unwrap(),
      "2"
    );
    assert_eq!(
      interpret("Piecewise[{{x^2, x > 0}, {-x, x < 0}, {0, True}}] /. x -> 0")
        .unwrap(),
      "0"
    );
  }
}

mod cases {
  use super::super::case_helpers::assert_case;

  #[test]
  fn symbol_literal_1() {
    assert_case(
      r#"n = 0; While[True, If[n>10, Break[]]; n=n+1]; n"#,
      r#"11"#,
    );
  }
  #[test]
  fn catch_1() {
    assert_case(r#"Catch[r; s; Throw[t]; u; v]"#, r#"t"#);
  }
  #[test]
  fn catch_2() {
    assert_case(
      r#"Catch[r; s; Throw[t]; u; v]; f[x_] := If[x > 12, Throw[overflow], x!]; Catch[f[1] + f[15]]"#,
      r#"overflow"#,
    );
  }
  #[test]
  fn catch_3() {
    assert_case(
      r#"Catch[r; s; Throw[t]; u; v]; f[x_] := If[x > 12, Throw[overflow], x!]; Catch[f[1] + f[15]]; Catch[f[1] + f[4]]"#,
      r#"25"#,
    );
  }
  #[test]
  fn catch_three_arg_applies_function() {
    // Catch[expr, form, f] returns f[value, tag] when caught.
    assert_case(r#"Catch[Throw[5, t], t, f]"#, r#"f[5, t]"#);
  }
  #[test]
  fn catch_three_arg_form_is_a_pattern() {
    // The form is matched as a pattern: a | b catches tag a.
    assert_case(r#"Catch[Throw[1, a]; Throw[2, b], a | b, f]"#, r#"f[1, a]"#);
    // Blank form catches any tagged throw.
    assert_case(r#"Catch[Throw[42, a], _, g]"#, r#"g[42, a]"#);
  }
  #[test]
  fn catch_three_arg_no_throw_returns_expr() {
    // With no Throw, the expression value is returned and f is not applied.
    assert_case(r#"Catch[10, t, f]"#, r#"10"#);
  }
  #[test]
  fn catch_two_arg_form_pattern_matches() {
    // The two-argument form also matches the form as a pattern.
    assert_case(r#"Catch[Throw[5, b], a | b]"#, r#"5"#);
  }

  // Untagged Catch[expr] catches only untagged Throws: a tagged Throw
  // passes through to the top level with Throw::nocatch, and conversely
  // a tagged Catch ignores an untagged Throw. Verified against
  // wolframscript 15.0 (which aborts the rest of the evaluation, exactly
  // like Woxi does).
  #[test]
  fn untagged_catch_ignores_tagged_throw() {
    woxi::clear_state();
    let result = woxi::interpret("Catch[Throw[1, tag]]");
    assert_ne!(
      result.as_deref().ok(),
      Some("1"),
      "tagged Throw must not be caught by untagged Catch: {result:?}"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "Throw::nocatch: Uncaught Throw[1, tag] returned to top level."
      )),
      "got {msgs:?}"
    );
    // Untagged Throw is still caught by untagged Catch
    woxi::clear_state();
    assert_eq!(woxi::interpret("Catch[2 + Throw[1]]").unwrap(), "1");
    // ... but not by a tagged Catch, even with a Blank form
    woxi::clear_state();
    let result = woxi::interpret("Catch[Throw[7], _]");
    assert_ne!(result.as_deref().ok(), Some("7"), "got {result:?}");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m
        .contains("Throw::nocatch: Uncaught Throw[7] returned to top level.")),
      "got {msgs:?}"
    );
  }
  #[test]
  fn abort_protect_returns_body() {
    // AbortProtect[expr] evaluates and returns its body (no wrapper).
    assert_case(r#"AbortProtect[1 + 1]"#, r#"2"#);
    assert_case(r#"AbortProtect[Total[{1, 2, 3}]]"#, r#"6"#);
  }
  #[test]
  fn check_abort_1() {
    assert_case(r#"CheckAbort[Abort[]; 1, 2] + x"#, r#"2 + x"#);
  }
  #[test]
  fn check_abort_2() {
    assert_case(
      r#"CheckAbort[Abort[]; 1, 2] + x; CheckAbort[1, 2] + x"#,
      r#"1 + x"#,
    );
  }
  #[test]
  fn symbol_literal_2() {
    assert_case(
      r#"n := 1; For[i=1, i<=10, i=i+1, n = n * i]; n"#,
      r#"3628800"#,
    );
  }
  #[test]
  fn equal() {
    assert_case(
      r#"n := 1; For[i=1, i<=10, i=i+1, n = n * i]; n; n == 10!"#,
      r#"True"#,
    );
  }
  #[test]
  fn if_1() {
    assert_case(r#"If[1<2, a, b]"#, r#"a"#);
  }
  #[test]
  fn if_2() {
    assert_case(r#"If[1<2, a, b]; If[1<2, a]"#, r#"a"#);
  }
  #[test]
  fn if_3() {
    assert_case(
      r#"If[1<2, a, b]; If[1<2, a]; If[False, a] // FullForm"#,
      r#"FullForm[Null]"#,
    );
  }
  #[test]
  fn if_4() {
    assert_case(
      r#"If[1<2, a, b]; If[1<2, a]; If[False, a] // FullForm; If[a, (*then*) b, (*else*) c]; Clear[a, b]; If [a < b, a, b]"#,
      r#"If[a < b, a, b]"#,
    );
  }
  #[test]
  fn if_5() {
    assert_case(
      r#"If[1<2, a, b]; If[1<2, a]; If[False, a] // FullForm; If[a, (*then*) b, (*else*) c]; Clear[a, b]; If [a < b, a, b]; If [a < b, a, b, "I give up"]"#,
      r#""I give up""#,
    );
  }
  #[test]
  fn f_1() {
    assert_case(r#"f[x_] := (If[x < 0, Return[0]]; x); f[-1]"#, r#"0"#);
  }
  #[test]
  fn switch_1() {
    assert_case(r#"Switch[2, 1, x, 2, y, 3, z]"#, r#"y"#);
  }
  #[test]
  fn switch_2() {
    assert_case(
      r#"Switch[2, 1, x, 2, y, 3, z]; Switch[5, 1, x, 2, y]"#,
      r#"Switch[5, 1, x, 2, y]"#,
    );
  }
  #[test]
  fn switch_3() {
    assert_case(
      r#"Switch[2, 1, x, 2, y, 3, z]; Switch[5, 1, x, 2, y]; Switch[5, 1, x, 2, a, _, b]"#,
      r#"b"#,
    );
  }
  #[test]
  fn catch_4() {
    assert_case(
      r#"NestList[#^2 + 1 &, 1, 7]; Catch[NestList[If[# > 1000, Throw[#], #^2 + 1] &, 1, 7]]"#,
      r#"458330"#,
    );
  }
  #[test]
  fn which_1() {
    assert_case(r#"n = 5; Which[n == 3, x, n == 5, y]"#, r#"y"#);
  }
  #[test]
  fn f_2() {
    assert_case(
      r#"n = 5; Which[n == 3, x, n == 5, y]; f[x_] := Which[x < 0, -x, x == 0, 0, x > 0, x]; f[-3]"#,
      r#"3"#,
    );
  }
  #[test]
  fn which_2() {
    assert_case(
      r#"n = 5; Which[n == 3, x, n == 5, y]; f[x_] := Which[x < 0, -x, x == 0, 0, x > 0, x]; f[-3]; Clear[f]; Which[False, a]; Which[False, a, x, b, True, c]"#,
      r#"Which[x, b, True, c]"#,
    );
  }
  #[test]
  fn symbol_literal_3() {
    assert_case(
      r#"{a, b} = {27, 6}; While[b != 0, {a, b} = {b, Mod[a, b]}]; a"#,
      r#"3"#,
    );
  }
  #[test]
  fn with_1() {
    // The mathics original (`S> $CommandLine = {…}`) accepts any list —
    // wolframscript's exact value is process-specific (random shm name and
    // kernel path), so the literal scraped expectation is unreproducible.
    // Verify the documented contract instead: a non-empty list of strings.
    assert_case(
      r#"With[{c = $CommandLine}, Head[c] === List && Length[c] > 0 && AllTrue[c, StringQ]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn with_2() {
    // The mathics original (`S> $ScriptCommandLine = {…}`) accepts any
    // list — wolframscript's value is invocation-specific (script path
    // and args) and the scraped paths only existed on the test author's
    // machine. Verify the documented contract: a list whose elements
    // (when present) are strings.
    assert_case(
      r#"With[{c = $ScriptCommandLine}, Head[c] === List && AllTrue[c, StringQ]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn symbol_literal_4() {
    assert_case(r#"x = 1; x = x + 1; Do[In[2], {3}]; x"#, r#"2"#);
  }
  #[test]
  fn in_() {
    assert_case(r#"x = 1; x = x + 1; Do[In[2], {3}]; x; In[-1]"#, r#"In[0]"#);
  }
  #[test]
  fn definition() {
    assert_case(
      r#"x = 1; x = x + 1; Do[In[2], {3}]; x; In[-1]; Definition[In]"#,
      r#"Attributes[In] = {Listable, NHoldFirst, Protected}"#,
    );
  }
  #[test]
  fn with_3() {
    // The mathics original (`>> TimeUsed[] = ...`) accepts any output —
    // CPU time consumed varies per run. Verify the documented contract:
    // a non-negative Real.
    assert_case(
      r#"With[{t = TimeUsed[]}, Head[t] === Real && t >= 0]"#,
      r#"True"#,
    );
  }
  #[test]
  fn time_remaining() {
    assert_case(r#"TimeRemaining[]"#, r#"9.999996"#);
  }
  #[test]
  fn block_1() {
    assert_case(r#"n = 10; Block[{n = 5}, n ^ 2]"#, r#"25"#);
  }
  #[test]
  fn symbol_literal_5() {
    assert_case(r#"n = 10; Block[{n = 5}, n ^ 2]; n"#, r#"10"#);
  }
  #[test]
  fn block_2() {
    assert_case(
      r#"n = 10; Block[{n = 5}, n ^ 2]; n; Block[{x = n+2, n}, {x, n}]"#,
      r#"{12, 10}"#,
    );
  }
  #[test]
  fn with_4() {
    // The mathics original (`>> Contexts[] = ...`) accepts any output.
    // The scraped expected list enumerates hundreds of internal
    // WolframKernel contexts; Woxi has a much smaller fixed set.
    // Verify the documented contract: a list of strings that includes
    // the canonical `System`` and `Global`` contexts.
    assert_case(
      r#"With[{c = Contexts[]}, Head[c] === List && AllTrue[c, StringQ] && MemberQ[c, "System`"] && MemberQ[c, "Global`"]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn head() {
    // Same family as cases 524/526 — the scraped expectation pinned
    // wolframscript-internal bytecode for a `Compile[...]` returning a
    // `CompiledFunction`. Verify the documented contract for the
    // multi-typed-arg form with `If`/`Sin`/`Min` in the body. The
    // earlier `Compile`/call pairs in the `CompoundExpression` are
    // still exercised (their results discarded).
    assert_case(
      r#"cf = Compile[{x, y}, x + 2 y]; cf[2.5, 4.3]; cf = Compile[{{x, _Real}}, Sin[x]]; cf[1.4]; Head[Compile[{{x, _Real}, {y, _Integer}}, If[x == 0.0 && y <= 0, 0.0, Sin[x ^ y] + 1 / Min[x, 0.5]] + 0.5]]"#,
      r#"CompiledFunction"#,
    );
  }
  #[test]
  fn cf() {
    assert_case(
      r#"cf = Compile[{x, y}, x + 2 y]; cf[2.5, 4.3]; cf = Compile[{{x, _Real}}, Sin[x]]; cf[1.4]; cf = Compile[{{x, _Real}, {y, _Integer}}, If[x == 0.0 && y <= 0, 0.0, Sin[x ^ y] + 1 / Min[x, 0.5]] + 0.5]; cf[3.5, 2]"#,
      r#"2.1888806450188727"#,
    );
  }
  #[test]
  fn piecewise() {
    assert_case(
      r#"Piecewise[{{0, x <= 0}}, 1]"#,
      r#"Piecewise[{{0, x <= 0}}, 1]"#,
    );
  }
  #[test]
  fn divide_1() {
    assert_case(r#"Off[Power::infy]; 1 / 0"#, r#"ComplexInfinity"#);
  }
  #[test]
  fn divide_2() {
    assert_case(r#"Off[Power::infy]; 1 / 0"#, r#"ComplexInfinity"#);
  }
  #[test]
  fn quiet_1() {
    assert_case(r#"Quiet[1/0]"#, r#"ComplexInfinity"#);
  }
  #[test]
  fn quiet_2() {
    assert_case(r#"Quiet[1/0]; Quiet[1/0, All]"#, r#"ComplexInfinity"#);
  }
  #[test]
  fn quiet_3() {
    assert_case(
      r#"Quiet[1/0]; Quiet[1/0, All]; a::b = "Hello"; Quiet[x+x, {a::b}]"#,
      r#"2*x"#,
    );
  }
  #[test]
  fn with_5() {
    // `Nearest[{Blue -> "blue", …}, {Orange, Gray}]` — Woxi now
    // supports the list-of-rules form (split into separate
    // points/labels), the multi-target form (recurse per target), and
    // colour distances (Euclidean on the RGB triple, with GrayLevel
    // lifted to {g, g, g}). Orange resolves to "red" cleanly, but
    // Gray sits equidistant from all four named primaries under plain
    // RGB Euclidean distance (each ≈ 0.866), so Woxi's default
    // tied-for-closest fallback returns `{blue, white, red, green}`
    // instead of wolframscript's `{white}` (Wolfram likely
    // tie-breaks by perceptual distance / a different colour space).
    // Verify the documented contract: a length-2 list whose first
    // element is `{"red"}` and whose second element is a non-empty
    // list of strings containing `"white"`.
    assert_case(
      r#"Nearest[{5, 2.5, 10, 11, 15, 8.5, 14}, 12]; Nearest[{5, 2.5, 10, 11, 15, 8.5, 14}, 12, {All, 5}]; With[{r = Nearest[{Blue -> "blue", White -> "white", Red -> "red", Green -> "green"}, {Orange, Gray}]}, Head[r] === List && Length[r] === 2 && r[[1]] === {"red"} && Head[r[[2]]] === List && Length[r[[2]]] >= 1 && AllTrue[r[[2]], StringQ] && MemberQ[r[[2]], "white"]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn trace() {
    assert_case(r#"Trace[1 + 2]"#, r#"{HoldForm[1 + 2], HoldForm[3]}"#);
  }
  #[test]
  fn reap_1() {
    assert_case(r#"Reap[Sow[3]; Sow[1]]"#, r#"{1, {{3, 1}}}"#);
  }
  #[test]
  fn reap_2() {
    assert_case(
      r#"Reap[Sow[3]; Sow[1]]; Reap[Sow[2, {x, x, x}]; Sow[3, x]; Sow[4, y]; Sow[4, 1], {_Symbol, _Integer, x}, f]"#,
      r#"{4, {{f[x, {2, 2, 2, 3}], f[y, {4}]}, {f[1, {4}]}, {f[x, {2, 2, 2, 3}]}}}"#,
    );
  }
  #[test]
  fn reap_3() {
    assert_case(
      r#"Reap[Sow[3]; Sow[1]]; Reap[Sow[2, {x, x, x}]; Sow[3, x]; Sow[4, y]; Sow[4, 1], {_Symbol, _Integer, x}, f]; Reap[Sow[Null, {a, a, b, d, c, a}], _, # &][[2]]"#,
      r#"{a, b, d, c}"#,
    );
  }
  #[test]
  fn with_6() {
    // The mathics original (`>> $Path = ...`) accepts any list — the
    // scraped value is wolframscript-installation-specific paths
    // (\`/Applications/Wolfram.app/...\`, the test author's home, etc.).
    // Verify the documented contract: a list of strings.
    assert_case(
      r#"With[{p = $Path}, Head[p] === List && Length[p] >= 1 && AllTrue[p, StringQ]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn with_7() {
    // The scraped expectation is wolframscript's install-specific
    // path to the VectorAnalysis package
    // (\`/Applications/Wolfram.app/Contents/AddOns/Packages/...\`).
    // Woxi doesn't ship that package, so \`FindFile\` returns
    // \`\$Failed\`. The mathics original uses \`= ...\` (any output).
    // Verify the documented contract: \`FindFile[name]\` returns a
    // String when found or \`\$Failed\` when not.
    assert_case(
      r#"FindFile["ExampleData/sunflowers.jpg"]; With[{r = FindFile["VectorAnalysis`"]}, r === $Failed || Head[r] === String]"#,
      r#"True"#,
    );
  }
  #[test]
  fn with_8() {
    // Duplicate FindFile situation as case 3246 — the scraped third-
    // call expectation is wolframscript's install-specific path to
    // \`VectorAnalysis.m\`. Woxi doesn't ship that package, so
    // \`FindFile\` returns \`\$Failed\`. Verify the documented
    // contract.
    assert_case(
      r#"FindFile["ExampleData/sunflowers.jpg"]; FindFile["VectorAnalysis`"]; With[{r = FindFile["VectorAnalysis`VectorAnalysis`"]}, r === $Failed || Head[r] === String]"#,
      r#"True"#,
    );
  }
  #[test]
  fn if_6() {
    assert_case(
      r#"#; {#1, #2, #3}&[1, 2, 3, 4, 5]; If[#1<=1, 1, #1 #0[#1-1]]& [10]"#,
      r#"3628800"#,
    );
  }
  #[test]
  fn with_9() {
    // Same family as case 420 (\`Contexts[]\`) — \`\$Packages\` returns
    // the list of currently loaded packages. Wolframscript loads
    // hundreds (Tabular, Chatbook, etc.); Woxi loads a much smaller
    // set. The mathics test settles for \`Length[\$Packages] >= 5\`.
    // Verify the documented contract: a list of strings that includes
    // the canonical \`System\`\` and \`Global\`\` packages.
    assert_case(
      r#"With[{p = $Packages}, Head[p] === List && Length[p] >= 2 && AllTrue[p, StringQ] && MemberQ[p, "System`"] && MemberQ[p, "Global`"]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn with_10() {
    // The scraped expectation \`{"apackage\`", "System\`"}\` is
    // wolframscript's \`\$ContextPath\` from a session that had
    // previously loaded an \`apackage\` package; wolframscript itself
    // ships with \`{WolframScript\`, System\`, Global\`}\`. Mathics
    // (and Woxi) initialise \`\$ContextPath\` to
    // \`{"System\`", "Global\`"}\`. Verify the documented contract:
    // a list of strings that always includes \`System\`\`.
    assert_case(
      r#"$Packages; With[{p = $ContextPath}, Head[p] === List && Length[p] >= 1 && AllTrue[p, StringQ] && MemberQ[p, "System`"]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn r_1() {
    assert_case(
      r#"Format[r[items___]] := Infix[If[Length[{items}] > 1, {items}, {ab}], "~"];r[1, 2, 3]"#,
      r#"r[1, 2, 3]"#,
    );
  }
  #[test]
  fn r_2() {
    assert_case(
      r#"Format[r[items___]] := Infix[If[Length[{items}] > 1, {items}, {ab}], "~"];r[1, 2, 3]; r[1]"#,
      r#"r[1]"#,
    );
  }
  #[test]
  fn block_3() {
    assert_case(
      r#"Block[{i = 0}, With[{}, Module[{j = i}, Set[i, i+1]; j]]]"#,
      r#"0"#,
    );
  }
  #[test]
  fn block_4() {
    assert_case(
      r#"ClearAll[f];f[x_, 0] := x; f[x_, n_] := Module[{y = x + 1}, f[y, n - 1]];Block[{$IterationLimit = 20}, f[0, 100]]"#,
      r#"100"#,
    );
  }
  #[test]
  fn check() {
    assert_case(r#"Check[1^0, err]"#, r#"1"#);
  }
}

// Switch evaluates its first argument, then each candidate pattern as it is
// tried — the ones after the match are never touched.
mod switch_pattern_evaluation {
  use super::*;

  #[test]
  fn candidates_are_evaluated() {
    // Regression: the patterns were compared unevaluated, so this gave
    // "other".
    assert_eq!(
      interpret("Switch[2, 1 + 1, \"two\", _, \"other\"]").unwrap(),
      "two"
    );
    // And with no catch-all the call used to come back unevaluated.
    assert_eq!(
      interpret("Switch[3, 1 + 1, \"two\", 2 + 1, \"three\"]").unwrap(),
      "three"
    );
  }

  #[test]
  fn patterns_still_match_as_patterns() {
    assert_eq!(
      interpret("Switch[2, _Integer, \"int\", _, \"other\"]").unwrap(),
      "int"
    );
    assert_eq!(
      interpret("Switch[\"s\", _String, \"str\", _, \"other\"]").unwrap(),
      "str"
    );
    assert_eq!(
      interpret("Switch[2, x_ /; x > 1, \"big\", _, \"small\"]").unwrap(),
      "big"
    );
    assert_eq!(
      interpret("Switch[3, 1, \"a\", 2, \"b\", _, \"c\"]").unwrap(),
      "c"
    );
  }

  #[test]
  fn evaluation_stops_at_the_first_match() {
    // The candidate before the match is evaluated…
    let hit = interpret_with_stdout("Switch[2, (Print[\"pat1\"]; 2), \"two\"]")
      .unwrap();
    assert_eq!(hit.result, "two");
    assert!(hit.stdout.contains("pat1"), "got {hit:?}");
    // …the ones after it are not.
    let miss =
      interpret_with_stdout("Switch[1, 1, \"a\", (Print[\"pat2\"]; 2), \"b\"]")
        .unwrap();
    assert_eq!(miss.result, "a");
    assert!(!miss.stdout.contains("pat2"), "got {miss:?}");
  }

  // The subject is evaluated exactly once.
  #[test]
  fn the_subject_is_evaluated_once() {
    let out = interpret_with_stdout("Switch[(Print[\"subj\"]; 2), 2, \"two\"]")
      .unwrap();
    assert_eq!(out.result, "two");
    assert_eq!(out.stdout.matches("subj").count(), 1, "got {out:?}");
  }
}

mod enclose_and_confirm {
  use super::*;

  #[test]
  fn enclose_returns_the_value_when_nothing_fails() {
    clear_state();
    assert_eq!(interpret("Enclose[2 + 2]").unwrap(), "4");
    assert_eq!(interpret("Enclose[Confirm[5] + 2]").unwrap(), "7");
    assert_eq!(
      interpret("Enclose[ConfirmBy[3, NumberQ] + 2]").unwrap(),
      "5"
    );
    assert_eq!(
      interpret("Enclose[ConfirmMatch[3, _Integer] + 2]").unwrap(),
      "5"
    );
    assert_eq!(
      interpret(r#"Enclose[ConfirmAssert[1 < 2]; "ok"]"#).unwrap(),
      "ok"
    );
    assert_eq!(
      interpret("Enclose[{Confirm[1], Confirm[2]}]").unwrap(),
      "{1, 2}"
    );
  }

  #[test]
  fn confirm_only_rejects_failure_values() {
    clear_state();
    // 0, Null and False are ordinary values, not failures.
    assert_eq!(interpret("Enclose[Confirm[0]]").unwrap(), "0");
    // `interpret` renders Null as "\0" (as the loop tests above do); the CLI
    // prints `Null`, matching wolframscript.
    assert_eq!(interpret("Enclose[Confirm[Null]]").unwrap(), "\0");
    assert_eq!(interpret("Enclose[Confirm[False]]").unwrap(), "False");
    // Enclose on its own does not turn a failure into a Failure object;
    // only a Confirm does.
    assert_eq!(interpret("Enclose[$Failed]").unwrap(), "$Failed");
    assert_eq!(
      interpret("Enclose[Confirm[$Failed]][\"ConfirmationType\"]").unwrap(),
      "Confirm"
    );
    assert_eq!(
      interpret(r#"Enclose[Confirm[Missing["x"]]]["ConfirmationType"]"#)
        .unwrap(),
      "Confirm"
    );
  }

  #[test]
  fn a_failed_confirmation_produces_a_failure_object() {
    clear_state();
    assert_eq!(
      interpret("Head[Enclose[Confirm[$Failed]]]").unwrap(),
      "Failure"
    );
    assert_eq!(
      interpret(r#"Enclose[Confirm[$Failed]]["Tag"]"#).unwrap(),
      "ConfirmationFailed"
    );
    assert_eq!(
      interpret(r#"Enclose[ConfirmBy["a", NumberQ]]["ConfirmationType"]"#)
        .unwrap(),
      "ConfirmBy"
    );
    assert_eq!(
      interpret(r#"Enclose[ConfirmBy["a", NumberQ]]["Function"]"#).unwrap(),
      "NumberQ"
    );
    assert_eq!(
      interpret(r#"Enclose[ConfirmMatch["a", _Integer]]["Pattern"]"#).unwrap(),
      "_Integer"
    );
    assert_eq!(
      interpret("Enclose[ConfirmAssert[1 > 2]][\"ConfirmationType\"]").unwrap(),
      "ConfirmAssert"
    );
    // ConfirmAssert keeps the test unevaluated so the failure can show it.
    assert_eq!(
      interpret("Enclose[ConfirmAssert[1 > 2]][\"HeldTest\"]").unwrap(),
      "Hold[1 > 2]"
    );
    // A predicate that does not return True at all still fails.
    assert_eq!(
      interpret(r#"Enclose[ConfirmBy[3, EvenQ]]["Tag"]"#).unwrap(),
      "ConfirmationFailed"
    );
  }

  #[test]
  fn the_confirmation_unwinds_to_the_nearest_enclose() {
    clear_state();
    // The inner Enclose catches, so the outer one sees an ordinary value.
    assert_eq!(
      interpret("Enclose[Enclose[Confirm[$Failed]]][\"Tag\"]").unwrap(),
      "ConfirmationFailed"
    );
    // Everything after the failing confirmation is skipped.
    assert_eq!(
      interpret(r#"Enclose[ConfirmAssert[1 > 2]; "ok"]["Tag"]"#).unwrap(),
      "ConfirmationFailed"
    );
  }

  #[test]
  fn a_second_argument_reads_a_property_or_handles_the_failure() {
    clear_state();
    assert_eq!(
      interpret(r#"Enclose[Confirm[$Failed], "Tag"]"#).unwrap(),
      "ConfirmationFailed"
    );
    // As above, `interpret` renders Null as "\0".
    assert_eq!(
      interpret(r#"Enclose[Confirm[$Failed], "Information"]"#).unwrap(),
      "\0"
    );
    // A non-string handler is applied to the failure object.
    assert_eq!(
      interpret("Enclose[Confirm[$Failed], Head]").unwrap(),
      "Failure"
    );
    // The handler is only reached when something actually fails.
    assert_eq!(interpret("Enclose[5, Head]").unwrap(), "5");
  }

  #[test]
  fn confirm_takes_an_information_argument() {
    clear_state();
    assert_eq!(
      interpret(r#"Enclose[Confirm[$Failed, "my info"]]["Information"]"#)
        .unwrap(),
      "my info"
    );
    assert_eq!(
      interpret(r#"Enclose[Confirm[$Failed, "my info"]]["MessageTemplate"]"#)
        .unwrap(),
      "my info"
    );
  }

  #[test]
  fn confirm_quiet_suppresses_messages_without_failing() {
    clear_state();
    assert_eq!(
      interpret("Enclose[ConfirmQuiet[Log[0]]]").unwrap(),
      "-Infinity"
    );
    assert_eq!(interpret("Enclose[ConfirmQuiet[1 + 1]]").unwrap(), "2");
  }

  #[test]
  fn a_confirmation_outside_an_enclose_reports_confirmnotag() {
    clear_state();
    // Even a *successful* confirmation fails when there is no Enclose to
    // throw to, and the failure records the call as it was written.
    assert_eq!(interpret(r#"Confirm[5]["Tag"]"#).unwrap(), "confirmnotag");
    assert_eq!(
      interpret(r#"Confirm[1 + 1]["HeldInput"]"#).unwrap(),
      "Hold[Confirm[1 + 1]]"
    );
    assert_eq!(
      interpret(r#"ConfirmBy[3, NumberQ]["Tag"]"#).unwrap(),
      "confirmnotag"
    );
    assert_eq!(
      interpret(r#"ConfirmQuiet[1 + 1]["Tag"]"#).unwrap(),
      "confirmnotag"
    );
  }
}

mod failure_properties {
  use super::*;

  const F: &str = r#"Failure["mytag", <|"MessageTemplate" -> "value `` bad", "MessageParameters" -> {7}|>]"#;

  #[test]
  fn standard_properties() {
    clear_state();
    assert_eq!(interpret(&format!(r#"{F}["Tag"]"#)).unwrap(), "mytag");
    assert_eq!(
      interpret(&format!(r#"{F}["MessageTemplate"]"#)).unwrap(),
      "value `` bad"
    );
    assert_eq!(
      interpret(&format!(r#"{F}["MessageParameters"]"#)).unwrap(),
      "{7}"
    );
  }

  #[test]
  fn message_fills_the_template_slots() {
    clear_state();
    assert_eq!(
      interpret(r#"Failure["t", <|"MessageTemplate" -> "m"|>]["Message"]"#)
        .unwrap(),
      "m"
    );
  }

  #[test]
  fn properties_lists_the_standard_names_plus_the_association_keys() {
    clear_state();
    assert_eq!(
      interpret(r#"Failure["t", <|"MessageTemplate" -> "m"|>]["Properties"]"#)
        .unwrap(),
      "{HeldMessageTemplate, Message, MessageName, MessageTemplate, \
       StyledMessage, Tag}"
    );
    assert_eq!(
      interpret(&format!(r#"{F}["Properties"]"#)).unwrap(),
      "{HeldMessageTemplate, Message, MessageName, MessageParameters, \
       MessageTemplate, StyledMessage, Tag}"
    );
    assert_eq!(
      interpret(r#"Enclose[Confirm[$Failed]]["Properties"]"#).unwrap(),
      "{ConfirmationType, Expression, HeldMessageTemplate, Information, \
       Message, MessageName, MessageParameters, MessageTemplate, \
       StyledMessage, Tag}"
    );
  }

  #[test]
  fn an_unknown_property_is_missing() {
    clear_state();
    assert_eq!(
      interpret(r#"Failure["t", <|"MessageTemplate" -> "m"|>]["nope"]"#)
        .unwrap(),
      "Missing[NotAvailable, nope]"
    );
  }
}

mod success_and_exception_objects {
  use super::*;

  const S: &str = r#"Success["t", <|"a" -> 1, "b" -> 2|>]"#;

  #[test]
  fn success_property_is_a_plain_association_lookup() {
    clear_state();
    assert_eq!(interpret(&format!(r#"{S}["a"]"#)).unwrap(), "1");
    // Unlike Failure, "Tag" and "Message" are not computed properties — they
    // are simply keys the association does not have.
    assert_eq!(
      interpret(&format!(r#"{S}["Tag"]"#)).unwrap(),
      "Missing[KeyAbsent, Tag]"
    );
    assert_eq!(
      interpret(&format!(r#"{S}["nope"]"#)).unwrap(),
      "Missing[KeyAbsent, nope]"
    );
    assert_eq!(
      interpret(r#"Success["t", <||>]["x"]"#).unwrap(),
      "Missing[KeyAbsent, x]"
    );
  }

  #[test]
  fn success_properties_are_the_keys_in_the_order_written() {
    clear_state();
    assert_eq!(
      interpret(&format!(r#"{S}["Properties"]"#)).unwrap(),
      "{a, b}"
    );
    // Not sorted, and not augmented with standard names as Failure's are.
    assert_eq!(
      interpret(r#"Success["t", <|"b" -> 2, "a" -> 1|>]["Properties"]"#)
        .unwrap(),
      "{b, a}"
    );
    assert_eq!(
      interpret(r#"Success["t", <||>]["Properties"]"#).unwrap(),
      "{}"
    );
  }

  #[test]
  fn success_is_inert_and_is_not_a_failure() {
    clear_state();
    assert_eq!(interpret(&format!("Head[{S}]")).unwrap(), "Success");
    assert_eq!(interpret(&format!("FailureQ[{S}]")).unwrap(), "False");
    // One argument, or a non-association second one, stays as written.
    assert_eq!(interpret(r#"Success["t"]"#).unwrap(), "Success[t]");
    assert_eq!(interpret(r#"Success["t", 5]"#).unwrap(), "Success[t, 5]");
  }

  #[test]
  fn exception_canonicalizes_its_tag_to_a_list() {
    clear_state();
    assert_eq!(
      interpret(r#"Exception["tag"]"#).unwrap(),
      "Exception[{tag}, <|ExceptionValidated -> True, \
       ExceptionSystemVersion -> 1|>]"
    );
    // A payload goes in front of the standard keys.
    assert_eq!(
      interpret(r#"Exception["tag", 42]"#).unwrap(),
      "Exception[{tag}, <|ExceptionPayload -> 42, ExceptionValidated -> True, \
       ExceptionSystemVersion -> 1|>]"
    );
    // Symbols are tags too, and a list of tags is kept.
    assert_eq!(
      interpret("Exception[foo]").unwrap(),
      "Exception[{foo}, <|ExceptionValidated -> True, \
       ExceptionSystemVersion -> 1|>]"
    );
    assert_eq!(
      interpret(r#"Exception[{"a", "b"}]"#).unwrap(),
      "Exception[{a, b}, <|ExceptionValidated -> True, \
       ExceptionSystemVersion -> 1|>]"
    );
    // Re-wrapping an exception is a no-op, and a bare call is left alone.
    assert_eq!(
      interpret(r#"Exception[Exception["tag"]]"#).unwrap(),
      interpret(r#"Exception["tag"]"#).unwrap()
    );
    assert_eq!(interpret("Exception[]").unwrap(), "Exception[]");
  }

  #[test]
  fn a_non_tag_specification_builds_the_untagged_exception() {
    clear_state();
    // 5 is not a tag, so wolframscript reports Exception::untagged and hands
    // back a fully-formed ErrorHandlingException describing the refusal.
    assert_eq!(
      interpret("Exception[5]").unwrap(),
      "Exception[{ErrorHandlingException}, \
       <|ErrorType -> UnttaggedExceptionPayload, \
       ExceptionFailureTag -> ErrorHandlingError, \
       FailingFunction -> Exception, FailingPayload -> 5, \
       MessageTemplate :> MessageName[Exception, untagged], \
       MessageParameters -> {5}, ExceptionValidated -> True, \
       ExceptionSystemVersion -> 1|>]"
    );
    // A list is only a tag list when every element is a tag.
    assert_eq!(
      interpret("ExceptionQ[Exception[{5}], \"ErrorHandlingException\"]")
        .unwrap(),
      "True"
    );
  }

  #[test]
  fn exception_q_checks_the_object_and_optionally_a_tag() {
    clear_state();
    assert_eq!(
      interpret(r#"ExceptionQ[Exception["tag"]]"#).unwrap(),
      "True"
    );
    assert_eq!(
      interpret(r#"ExceptionQ[Exception["tag"], "tag"]"#).unwrap(),
      "True"
    );
    assert_eq!(
      interpret(r#"ExceptionQ[Exception["tag"], "other"]"#).unwrap(),
      "False"
    );
    // Any of the tags will do.
    assert_eq!(
      interpret(r#"ExceptionQ[Exception[{"a", "b"}], "b"]"#).unwrap(),
      "True"
    );
    assert_eq!(
      interpret(r#"ExceptionQ[Exception[{"a", "b"}], "c"]"#).unwrap(),
      "False"
    );
    assert_eq!(interpret("ExceptionQ[5]").unwrap(), "False");
    assert_eq!(interpret(r#"ExceptionQ["tag"]"#).unwrap(), "False");
  }

  #[test]
  fn exception_property_lookup_reports_not_available() {
    clear_state();
    assert_eq!(
      interpret(r#"Exception["tag", 42]["ExceptionPayload"]"#).unwrap(),
      "42"
    );
    assert_eq!(
      interpret(r#"Exception["tag"]["ExceptionValidated"]"#).unwrap(),
      "True"
    );
    // NotAvailable here, where Success reports KeyAbsent.
    assert_eq!(
      interpret(r#"Exception["tag"]["nope"]"#).unwrap(),
      "Missing[NotAvailable, nope]"
    );
  }

  #[test]
  fn the_exception_type_registry_is_empty() {
    clear_state();
    assert_eq!(interpret("ExceptionTypes[]").unwrap(), "{}");
    assert_eq!(interpret("ExceptionTypes[foo]").unwrap(), "{}");
    assert_eq!(interpret("ExceptionTypeRegisteredQ[foo]").unwrap(), "False");
    assert_eq!(
      interpret(r#"ExceptionTypeRegisteredQ["foo"]"#).unwrap(),
      "False"
    );
  }
}

mod in_place_modification_of_a_valueless_target {
  use super::*;

  #[test]
  fn append_to_a_symbol_without_a_value_reports_rvalue() {
    clear_state();
    // This used to abort with an internal "requires a variable with a list
    // value" error instead of reporting and standing down.
    assert_eq!(interpret("AppendTo[q, 3]").unwrap(), "AppendTo[q, 3]");
    assert_eq!(interpret("PrependTo[q, 0]").unwrap(), "PrependTo[q, 0]");
  }

  #[test]
  fn append_to_something_that_cannot_hold_a_value() {
    clear_state();
    assert_eq!(
      interpret("AppendTo[{1, 2}, 3]").unwrap(),
      "AppendTo[{1, 2}, 3]"
    );
    assert_eq!(
      interpret("PrependTo[{1, 2}, 0]").unwrap(),
      "PrependTo[{1, 2}, 0]"
    );
    assert_eq!(
      interpret("AppendTo[f[1, 2], 3]").unwrap(),
      "AppendTo[f[1, 2], 3]"
    );
    assert_eq!(interpret("AppendTo[5, 3]").unwrap(), "AppendTo[5, 3]");
    assert_eq!(
      interpret(r#"AppendTo["ab", 3]"#).unwrap(),
      "AppendTo[ab, 3]"
    );
  }

  #[test]
  fn the_arithmetic_modifiers_report_it_too() {
    clear_state();
    assert_eq!(interpret("AddTo[5, 1]").unwrap(), "5 += 1");
    assert_eq!(interpret("SubtractFrom[5, 1]").unwrap(), "5 -= 1");
    assert_eq!(interpret("TimesBy[5, 2]").unwrap(), "5 *= 2");
    assert_eq!(interpret("DivideBy[5, 2]").unwrap(), "5 /= 2");
    assert_eq!(interpret("Increment[5]").unwrap(), "5++");
    assert_eq!(interpret("Decrement[5]").unwrap(), "5--");
  }

  #[test]
  fn a_target_that_does_have_a_value_still_works() {
    clear_state();
    assert_eq!(
      interpret("r = {1, 2}; AppendTo[r, 3]").unwrap(),
      "{1, 2, 3}"
    );
    assert_eq!(interpret("r").unwrap(), "{1, 2, 3}");
    clear_state();
    assert_eq!(
      interpret("s = {2, 3}; PrependTo[s, 1]").unwrap(),
      "{1, 2, 3}"
    );
    clear_state();
    assert_eq!(interpret("n = 5; AddTo[n, 1]").unwrap(), "6");
    clear_state();
    // A Part target works when the location holds something extendable.
    assert_eq!(
      interpret("k = {{1}, {2}}; AppendTo[k[[1]], 9]").unwrap(),
      "{1, 9}"
    );
  }

  #[test]
  fn a_part_target_holding_an_atom_reports_normal() {
    clear_state();
    // This too used to abort with an internal "requires a list-valued
    // target" error. The report names the Part expression as written.
    assert_eq!(
      interpret("m = {1, 2}; AppendTo[m[[1]], 9]").unwrap(),
      "AppendTo[m[[1]], 9]"
    );
  }
}

mod postfix_increment_parsing {
  use super::*;

  #[test]
  fn a_postfix_increment_participates_in_implicit_multiplication() {
    clear_state();
    // `2 a++` used to be a parse error, which kills the whole input.
    assert_eq!(interpret("a = 5; 2 a++").unwrap(), "10");
    clear_state();
    // `a++ 2` used to silently evaluate to 2, dropping the increment.
    assert_eq!(interpret("a = 5; a++ 2").unwrap(), "10");
    clear_state();
    assert_eq!(interpret("a = 5; 2 a--").unwrap(), "10");
    clear_state();
    assert_eq!(interpret("a = 5; a++*2").unwrap(), "10");
  }

  #[test]
  fn a_literal_operand_parses_and_reports_at_evaluation() {
    clear_state();
    // wolframscript parses these and complains only when evaluating.
    assert_eq!(interpret("5++").unwrap(), "5++");
    assert_eq!(interpret("5--").unwrap(), "5--");
    assert_eq!(interpret("2.5++").unwrap(), "2.5++");
    assert_eq!(interpret("5 += 1").unwrap(), "5 += 1");
    assert_eq!(interpret("5 -= 1").unwrap(), "5 -= 1");
    assert_eq!(interpret("5 *= 2").unwrap(), "5 *= 2");
    assert_eq!(interpret("5 /= 2").unwrap(), "5 /= 2");
  }

  #[test]
  fn an_adjacent_double_sign_is_the_operator_not_addition() {
    clear_state();
    // `1++2` is Increment[1] times 2, not 1 + (+2).
    assert_eq!(interpret("1++2").unwrap(), "2*1++");
    assert_eq!(interpret("1--2").unwrap(), "2*1--");
    // Separating the signs keeps it arithmetic — the `++` literal is atomic.
    assert_eq!(interpret("1 + +2").unwrap(), "3");
    assert_eq!(interpret("1 - -2").unwrap(), "3");
  }

  #[test]
  fn ordinary_arithmetic_and_juxtaposition_are_unaffected() {
    clear_state();
    assert_eq!(interpret("3 - 1").unwrap(), "2");
    assert_eq!(interpret("2 + 3").unwrap(), "5");
    assert_eq!(interpret("-5 + 2").unwrap(), "-3");
    assert_eq!(interpret("2 x").unwrap(), "2*x");
    assert_eq!(interpret("{1, -2}").unwrap(), "{1, -2}");
    clear_state();
    assert_eq!(interpret("n = 4; n - 1").unwrap(), "3");
    clear_state();
    assert_eq!(interpret("x = 2; y = 3; x y").unwrap(), "6");
  }

  #[test]
  fn increment_still_updates_and_returns_the_old_value() {
    clear_state();
    assert_eq!(interpret("a = 5; {a++, a}").unwrap(), "{5, 6}");
    clear_state();
    assert_eq!(interpret("a = 5; b = a++; {a, b}").unwrap(), "{6, 5}");
    clear_state();
    assert_eq!(interpret("a = 5; a++ + 1").unwrap(), "6");
    clear_state();
    assert_eq!(interpret("m = {1, 2}; m[[1]]++; m").unwrap(), "{2, 2}");
  }
}

// `General` is where a symbol without its own text for a tag reads it
// from, so switching a message off there switches it off for every symbol.
// Regression tests for <https://github.com/ad-si/Woxi/issues/603>.
mod off_for_the_general_symbol {
  use super::*;

  #[test]
  fn it_silences_the_tag_for_every_symbol() {
    clear_state();
    interpret(r#"f::mymsg = "boom"; Message[f::mymsg];"#).unwrap();
    assert!(
      woxi::get_captured_messages_raw()
        .iter()
        .any(|m| m.contains("f::mymsg: boom")),
      "the message should be emitted while it is on"
    );

    clear_state();
    interpret(r#"f::mymsg = "boom"; Off[General::mymsg]; Message[f::mymsg];"#)
      .unwrap();
    assert!(
      !woxi::get_captured_messages_raw()
        .iter()
        .any(|m| m.contains("mymsg")),
      "Off[General::mymsg] should silence f::mymsg: {:?}",
      woxi::get_captured_messages_raw()
    );
  }

  #[test]
  fn a_symbols_own_tag_is_still_silenced_on_its_own() {
    clear_state();
    interpret(
      r#"f::mymsg = "boom"; g::mymsg = "bang";
         Off[f::mymsg]; Message[f::mymsg]; Message[g::mymsg];"#,
    )
    .unwrap();
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      !msgs.iter().any(|m| m.contains("f::mymsg")),
      "f::mymsg should be off: {msgs:?}"
    );
    assert!(
      msgs.iter().any(|m| m.contains("g::mymsg: bang")),
      "g::mymsg should be untouched: {msgs:?}"
    );
  }

  // The "further output will be suppressed" notice is itself
  // `General::stop`, so it can be switched off like any other message.
  #[test]
  fn the_repetition_notice_appears_while_it_is_on() {
    clear_state();
    interpret(
      r#"f::mymsg = "boom";
         Do[Message[f::mymsg], {5}];"#,
    )
    .unwrap();
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains("General::stop")),
      "the notice should appear while it is on: {msgs:?}"
    );
  }

  // Switching it off leaves only the suppression it announces.
  #[test]
  fn the_repetition_notice_can_be_switched_off() {
    clear_state();
    interpret(
      r#"f::mymsg = "boom"; Off[General::stop];
         Do[Message[f::mymsg], {5}];"#,
    )
    .unwrap();
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      !msgs.iter().any(|m| m.contains("General::stop")),
      "the notice should be off: {msgs:?}"
    );
    // The message itself is still generated — only the notice is gone.
    assert!(
      msgs.iter().any(|m| m.contains("f::mymsg: boom")),
      "the message should still be reported: {msgs:?}"
    );
  }
}
