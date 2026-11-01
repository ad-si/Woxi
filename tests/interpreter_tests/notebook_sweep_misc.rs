//! Regressions found while making Wolfram Demonstrations notebooks run:
//! semantic bugs that surfaced as generic Part/list/argument messages.
use super::*;

mod internal_messages {
  use super::*;

  // The elimination in Solve/Reduce tries the branch where a denominator
  // vanishes and meets `1/0` there. That is the solver's own bookkeeping:
  // the message must not be generated at all, so `Check` does not fire, it
  // is not in `$MessageList`, and a front end (which shows the captured
  // messages) reports nothing.
  const SYSTEM: &str = "Solve[{8.314 == (22.4*p)/a + (22.4*p)/b, \
    0.0036639754618530535*a == (0.009871668311944718*p)^(2/7), \
    2269.12 == 44.8*p}, {a, b, p}]";

  #[test]
  fn solve_elimination_raises_no_messages() {
    clear_state();
    let result = interpret_with_stdout(SYSTEM).unwrap();
    assert!(
      result.result.starts_with("{{a -> 223.89"),
      "{}",
      result.result
    );
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
  }

  #[test]
  fn solve_elimination_does_not_trip_check() {
    clear_state();
    assert_eq!(
      interpret(&format!("Head[Check[{SYSTEM}, failed]]")).unwrap(),
      "List"
    );
    assert_eq!(interpret(&format!("{SYSTEM}; $MessageList")).unwrap(), "{}");
  }

  #[test]
  fn user_division_by_zero_still_reported() {
    clear_state();
    let result = interpret_with_stdout("1/0").unwrap();
    assert_eq!(result.result, "ComplexInfinity");
    assert!(result.warnings.iter().any(|w| w.contains("Power::infy")));
  }
}

mod element_alternatives {
  use super::*;

  #[test]
  fn list_becomes_flat_alternatives() {
    clear_state();
    assert_eq!(
      interpret("ToString[FullForm[Element[{a, 2, b, c}, Integers]]]").unwrap(),
      "Element[Alternatives[a, b, c], Integers]"
    );
    assert_eq!(
      interpret("Element[{a, 2, b}, Integers]").unwrap(),
      "Element[a | b, Integers]"
    );
    assert_eq!(
      interpret("Element[{a, 2.5, b}, Integers]").unwrap(),
      "False"
    );
  }

  // A long chain used to be a nested binary `a | (b | (c | …))` as deep as
  // the list was long, and re-evaluating the stored value hit
  // $RecursionLimit (an integer program over 1640 variables).
  #[test]
  fn long_variable_list_reevaluates() {
    clear_state();
    assert_eq!(
      interpret(
        "c = Thread[Array[z, 1640] \\[Element] Integers]; {Head[c], Length[c[[1]]]}"
      )
      .unwrap(),
      "{Element, 1640}"
    );
  }
}

mod count_rule_pattern {
  use super::*;

  // Count has no transformation-rule form (unlike Cases): a rule is a
  // pattern that matches rules.
  #[test]
  fn rule_is_a_pattern() {
    clear_state();
    assert_eq!(
      interpret(
        "{Count[{a -> 1, b -> 0}, _ -> 1], Count[{a :> 1, b -> 0}, _ :> 1], \
         Count[{f[a -> 1], b -> 1}, _ -> 1, Infinity]}"
      )
      .unwrap(),
      "{1, 1, 2}"
    );
    assert_eq!(
      interpret("Cases[{a -> 1, b -> 0}, _ -> 1]").unwrap(),
      "{1, 1}"
    );
  }
}

mod integer_programs {
  use super::*;

  // An inexact integer program is solved in machine precision (as in
  // Wolfram); the exact search over the rationals of decimal inputs took
  // minutes on a few hundred variables. The continuous variables of a mixed
  // problem still come out at the exact vertex.
  #[test]
  fn mixed_linear_programming_keeps_exact_vertex() {
    clear_state();
    assert_eq!(
      interpret(
        "LinearProgramming[{1., 2.5}, {{1, 2}, {3, 1}}, {3.5, 2}, {0, 0}, {Integers, Reals}]"
      )
      .unwrap(),
      "{3, 0.25}"
    );
    assert_eq!(
      interpret(
        "LinearProgramming[{1.5, 2}, {{1, 1}}, {{2.5, 1}}, {{0, 3}, {0, 3}}, {Integers, Reals}]"
      )
      .unwrap(),
      "{2, 0.5}"
    );
  }

  // The facility-location program of the "Facilities Location Problem"
  // Demonstration: 600 binary variables x[i, j], y[j].
  #[test]
  fn facility_location_over_indexed_variables() {
    clear_state();
    let result = interpret(
      "n = 24; pts = Table[{Cos[2. Pi i/n] (1 + i/10.), Sin[2. Pi i/n]}, {i, n}]; \
       var = Flatten@Join[Table[x[i, j], {i, n}, {j, n}], Table[y[j], {j, n}]]; \
       obj = Sum[EuclideanDistance[pts[[i]], pts[[j]]]*x[i, j], {i, n}, {j, n}]; \
       cons = Flatten@{Thread[0 <= var <= 1], Thread[var \\[Element] Integers], \
         Table[x[i, j] <= y[j], {i, n}, {j, n}], Sum[y[j], {j, n}] <= 3, \
         Table[Sum[x[i, j], {j, n}] == 1, {i, n}]}; \
       sol = NMinimize[{obj, cons}, var]; \
       {sol[[1]], Cases[sol[[2]], HoldPattern[y[j_] -> 1] :> j], Count[sol[[2]], _ -> 1]}",
    )
    .unwrap();
    assert_eq!(result, "{18.96359150724976, {4, 11, 22}, 27}");
  }
}

mod combinatorica_tableaux {
  use super::*;

  #[test]
  fn tableaux_of_a_shape_in_package_order() {
    clear_state();
    assert_eq!(
      interpret(
        "Quiet@Get[\"Combinatorica`\"]; \
         {Tableaux[{2, 2}], Tableaux[{3, 1}], Combinatorica`Tableaux[{1}]}"
      )
      .unwrap(),
      "{{{{1, 3}, {2, 4}}, {{1, 2}, {3, 4}}}, \
       {{{1, 3, 4}, {2}}, {{1, 2, 4}, {3}}, {{1, 2, 3}, {4}}}, {{{1}}}}"
    );
    assert_eq!(
      interpret("Take[Combinatorica`Tableaux[{3, 2, 1}], 3]").unwrap(),
      "{{{1, 4, 6}, {2, 5}, {3}}, {{1, 3, 6}, {2, 5}, {4}}, \
       {{1, 2, 6}, {3, 5}, {4}}}"
    );
  }

  #[test]
  fn tableaux_of_every_partition() {
    clear_state();
    assert_eq!(
      interpret("Combinatorica`Tableaux[3]").unwrap(),
      "{{{1, 2, 3}}, {{1, 3}, {2}}, {{1, 2}, {3}}, {{1}, {2}, {3}}}"
    );
  }

  #[test]
  fn number_of_tableaux_by_hook_lengths() {
    clear_state();
    assert_eq!(
      interpret(
        "{Combinatorica`NumberOfTableaux[{3, 2, 1}], \
         Combinatorica`NumberOfTableaux[{4, 3, 1}], \
         Combinatorica`NumberOfTableaux[5], \
         Length[Combinatorica`Tableaux[{4, 3, 1}]]}"
      )
      .unwrap(),
      "{16, 70, 26, 70}"
    );
  }

  #[test]
  fn non_shapes_stay_unevaluated() {
    clear_state();
    assert_eq!(
      interpret("Combinatorica`Tableaux[{2, 3}]").unwrap(),
      "Combinatorica`Tableaux[{2, 3}]"
    );
  }
}
