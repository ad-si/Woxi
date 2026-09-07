use num_bigint::Sign;
use woxi_reduce::rational_qe::*;
use woxi_reduce::*;

#[cfg(test)]
mod tests {
  use std::collections::{BTreeMap, BTreeSet};

  use num_bigint::BigInt;
  use proptest::prelude::*;
  use proptest::test_runner::RngSeed;

  use super::super::lower::formula_from_expr;
  use super::*;

  fn parse(source: &str) -> Formula {
    formula_from_expr(&crate::parse_to_expr(source).unwrap()).unwrap()
  }

  fn eliminate(source: &str) -> Formula {
    eliminate_quantifiers(parse(source)).unwrap()
  }

  #[test]
  fn elimination_order_prefers_an_exact_pivot_and_is_permutation_stable() {
    fn selection(source: &str) -> ((bool, usize, usize, usize), Formula) {
      let formula = parse(source).normalized();
      let Formula::Quantified(_, variables, body) = &formula else {
        panic!("test input must contain one quantifier block");
      };
      let selected = choose_elimination_variable(
        body,
        &variables.iter().cloned().collect::<BTreeSet<_>>(),
      )
      .unwrap();
      (
        rational_elimination_cost(body, &selected),
        eliminate_quantifiers(formula).unwrap(),
      )
    }

    let body = "x > a && x < b && x > c && x < d && y == a + b";
    let first = selection(&format!("Exists[{{x, y}}, {body}]"));
    let permuted = selection(&format!("Exists[{{y, x}}, {body}]"));
    assert!(
      !first.0.0,
      "the selected variable must have an equality pivot"
    );
    assert_eq!(first.0, permuted.0);
    assert_eq!(first.1, permuted.1);
  }

  #[test]
  fn repeated_quantified_subformulas_use_the_shared_memo() {
    let repeated = parse("Exists[x, x == 1]");
    let source = Formula::Or(vec![
      Formula::And(vec![repeated.clone(), parse("a < 0")]),
      Formula::And(vec![repeated, parse("a >= 0")]),
    ])
    .normalized();
    let mut memo = FormulaMemo::default();
    let result = eliminate_recursive(source, &mut memo).unwrap().normalized();
    assert_eq!(result, Formula::True);
    assert!(memo.hits() > 0);
  }

  #[test]
  fn projects_open_interval_and_preserves_free_parameters() {
    assert_eq!(
      eliminate("Exists[y, 0 < y && y < x && x < 2]"),
      parse("0 < x && x < 2")
    );
  }

  #[test]
  fn detects_incompatible_strict_and_closed_bounds() {
    assert_eq!(eliminate("Exists[x, x < a && x >= a]"), Formula::False);
    assert_eq!(eliminate("Exists[x, x <= a && x >= a]"), Formula::True);
  }

  #[test]
  fn unbounded_dense_intervals_have_witnesses() {
    assert_eq!(eliminate("Exists[x, x < a]"), Formula::True);
    assert_eq!(eliminate("Exists[x, x > a]"), Formula::True);
    assert_eq!(eliminate("Exists[x, x != a]"), Formula::True);
  }

  #[test]
  fn equality_substitution_is_exact_and_removes_the_binder() {
    let result = eliminate("Exists[x, 2*x + y == 0 && x > 1]");
    assert_eq!(
      result.free_variables(),
      BTreeSet::from([Variable::free("y")])
    );
    assert!(result.is_nnf());
    assert!(
      result
        .all_variables()
        .iter()
        .all(|variable| variable.binder.is_none())
    );
  }

  #[test]
  fn universal_quantification_uses_checked_duality() {
    assert_eq!(eliminate("ForAll[x, x <= a || x > a]"), Formula::True);
    assert_eq!(eliminate("ForAll[x, x < a]"), Formula::False);
  }

  #[test]
  fn nested_alternation_is_eliminated_innermost_first() {
    let result = eliminate("Exists[x, ForAll[y, y < x || y >= x]]");
    assert_eq!(result, Formula::True);
    assert!(result.all_variables().is_empty());
  }

  #[test]
  fn divisibility_is_rejected_by_the_dense_theory() {
    assert!(
      eliminate_quantifiers(parse("Exists[x, Divisible[x, 2]]")).is_none()
    );
  }

  fn evaluate_term(
    term: &AffineTerm,
    environment: &BTreeMap<Variable, Rational>,
  ) -> Rational {
    term.coefficients.iter().fold(
      term.constant.clone(),
      |value, (variable, coefficient)| {
        value.add(
          &coefficient.multiply(
            environment
              .get(variable)
              .expect("every free variable must have a valuation"),
          ),
        )
      },
    )
  }

  fn evaluate_atom(
    atom: &Atom,
    environment: &BTreeMap<Variable, Rational>,
  ) -> bool {
    let Atom::Relation(relation, term) = atom else {
      panic!("dense generated formulas contain no divisibility atoms");
    };
    let sign = evaluate_term(term, environment).numerator.sign();
    match relation {
      Relation::Equal => sign == Sign::NoSign,
      Relation::NotEqual => sign != Sign::NoSign,
      Relation::Less => sign == Sign::Minus,
      Relation::LessEqual => sign != Sign::Plus,
      Relation::Greater => sign == Sign::Plus,
      Relation::GreaterEqual => sign != Sign::Minus,
    }
  }

  fn collect_boundaries(
    formula: &Formula,
    variable: &Variable,
    environment: &BTreeMap<Variable, Rational>,
    output: &mut Vec<Rational>,
  ) {
    match formula {
      Formula::Atom(Atom::Relation(_, term)) => {
        let coefficient = term.coefficient(variable);
        if coefficient.is_zero() {
          return;
        }
        let mut rest = term.clone();
        rest.coefficients.remove(variable);
        output.push(
          evaluate_term(&rest, environment)
            .negated()
            .checked_divide(&coefficient)
            .unwrap(),
        );
      }
      Formula::And(children) | Formula::Or(children) => {
        for child in children {
          collect_boundaries(child, variable, environment, output);
        }
      }
      Formula::Not(inner) => {
        collect_boundaries(inner, variable, environment, output);
      }
      Formula::True | Formula::False => {}
      Formula::Atom(Atom::Divides { .. }) => {
        panic!("dense generated formulas contain no divisibility atoms");
      }
      Formula::Quantified(_, _, _) => {
        panic!("the independent oracle handles one binder at a time");
      }
    }
  }

  fn representative_points(
    body: &Formula,
    variable: &Variable,
    environment: &BTreeMap<Variable, Rational>,
  ) -> Vec<Rational> {
    let mut boundaries = Vec::new();
    collect_boundaries(body, variable, environment, &mut boundaries);
    boundaries.sort_by(Rational::numeric_cmp);
    boundaries.dedup_by(|left, right| left.numeric_cmp(right).is_eq());
    if boundaries.is_empty() {
      return vec![Rational::zero()];
    }
    let one = Rational::one();
    let two = Rational::integer(BigInt::from(2));
    let mut points = Vec::with_capacity(boundaries.len() * 2 + 1);
    points.push(boundaries[0].subtract(&one));
    for (index, boundary) in boundaries.iter().enumerate() {
      points.push(boundary.clone());
      if let Some(next) = boundaries.get(index + 1) {
        points.push(boundary.add(next).checked_divide(&two).unwrap());
      }
    }
    points.push(boundaries.last().unwrap().add(&one));
    points
  }

  fn evaluate_formula(
    formula: &Formula,
    environment: &mut BTreeMap<Variable, Rational>,
  ) -> bool {
    match formula {
      Formula::True => true,
      Formula::False => false,
      Formula::Atom(atom) => evaluate_atom(atom, environment),
      Formula::And(children) => children
        .iter()
        .all(|child| evaluate_formula(child, environment)),
      Formula::Or(children) => children
        .iter()
        .any(|child| evaluate_formula(child, environment)),
      Formula::Not(inner) => !evaluate_formula(inner, environment),
      Formula::Quantified(quantifier, variables, body) => {
        assert_eq!(variables.len(), 1);
        let variable = &variables[0];
        let points = representative_points(body, variable, environment);
        match quantifier {
          Quantifier::Exists => points.into_iter().any(|point| {
            environment.insert(variable.clone(), point);
            let result = evaluate_formula(body, environment);
            environment.remove(variable);
            result
          }),
          Quantifier::ForAll => points.into_iter().all(|point| {
            environment.insert(variable.clone(), point);
            let result = evaluate_formula(body, environment);
            environment.remove(variable);
            result
          }),
        }
      }
    }
  }

  fn generated_atom(
    bound: &Variable,
    free: &Variable,
    bound_coefficient: i8,
    free_coefficient: i8,
    constant: i8,
    relation: u8,
  ) -> Formula {
    let term = AffineTerm::variable(bound.clone())
      .scaled(&Rational::integer(BigInt::from(bound_coefficient)))
      .add(
        &AffineTerm::variable(free.clone())
          .scaled(&Rational::integer(BigInt::from(free_coefficient))),
      )
      .add(&AffineTerm::constant(Rational::integer(BigInt::from(
        constant,
      ))));
    Formula::Atom(Atom::Relation(
      match relation % 6 {
        0 => Relation::Equal,
        1 => Relation::NotEqual,
        2 => Relation::Less,
        3 => Relation::LessEqual,
        4 => Relation::Greater,
        _ => Relation::GreaterEqual,
      },
      term,
    ))
  }

  fn smt_relation(relation: u8, term: &str) -> String {
    match relation % 6 {
      0 => format!("(= {term} 0)"),
      1 => format!("(not (= {term} 0))"),
      2 => format!("(< {term} 0)"),
      3 => format!("(<= {term} 0)"),
      4 => format!("(> {term} 0)"),
      _ => format!("(>= {term} 0)"),
    }
  }

  #[test]
  #[ignore = "requires a Z3 development oracle"]
  fn generated_closed_formulas_agree_with_z3() {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let mut state = 0x5eed_2000_u64;
    let mut next = || {
      state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
      state
    };
    let bound = Variable::bound("x", 0);
    let unused = Variable::free("unused");
    let mut expected = Vec::with_capacity(500);
    let mut query = String::from("(set-logic LRA)\n");
    for _ in 0..500 {
      let a = i8::try_from(next() % 11).unwrap() - 5;
      let b = i8::try_from(next() % 17).unwrap() - 8;
      let c = i8::try_from(next() % 11).unwrap() - 5;
      let d = i8::try_from(next() % 17).unwrap() - 8;
      let first_relation = u8::try_from(next() % 6).unwrap();
      let second_relation = u8::try_from(next() % 6).unwrap();
      let conjunction = next() & 1 == 0;
      let universal = next() & 1 == 0;
      let first = generated_atom(&bound, &unused, a, 0, b, first_relation);
      let second = generated_atom(&bound, &unused, c, 0, d, second_relation);
      let body = if conjunction {
        Formula::And(vec![first, second])
      } else {
        Formula::Or(vec![first, second])
      };
      let formula = Formula::Quantified(
        if universal {
          Quantifier::ForAll
        } else {
          Quantifier::Exists
        },
        vec![bound.clone()],
        Box::new(body),
      )
      .normalized();
      expected.push(match eliminate_quantifiers(formula).unwrap() {
        Formula::True => "sat",
        Formula::False => "unsat",
        result => panic!("a closed formula must decide to truth: {result:?}"),
      });

      let first_term = format!("(+ (* {a} x) {b})");
      let second_term = format!("(+ (* {c} x) {d})");
      let connective = if conjunction { "and" } else { "or" };
      let quantifier = if universal { "forall" } else { "exists" };
      query.push_str("(push)\n(assert ");
      query.push_str(&format!(
        "({quantifier} ((x Real)) ({connective} {} {}))",
        smt_relation(first_relation, &first_term),
        smt_relation(second_relation, &second_term),
      ));
      query.push_str(")\n(check-sat)\n(pop)\n");
    }
    query.push_str("(exit)\n");

    let executable = std::env::var("WOXI_Z3").unwrap_or_else(|_| "z3".into());
    let mut child = Command::new(executable)
      .arg("-in")
      .stdin(Stdio::piped())
      .stdout(Stdio::piped())
      .spawn()
      .expect("Z3 must be available for this development-only oracle test");
    child
      .stdin
      .take()
      .unwrap()
      .write_all(query.as_bytes())
      .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    let actual = String::from_utf8(output.stdout).unwrap();
    let actual = actual.lines().collect::<Vec<_>>();
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
      assert_eq!(*actual, expected, "generated oracle case {index}");
    }
  }

  proptest! {
    #![proptest_config(ProptestConfig {
      cases: 10_000,
      rng_seed: RngSeed::Fixed(0x5eed_1000),
      max_shrink_iters: 20_000,
      ..ProptestConfig::default()
    })]

    #[test]
    fn generated_projection_agrees_with_independent_boundary_sampling(
      atoms in proptest::collection::vec(
        (-5_i8..=5, -5_i8..=5, -8_i8..=8, 0_u8..6),
        1..=4,
      ),
      connective_bits in any::<u8>(),
      universal in any::<bool>(),
    ) {
      let bound = Variable::bound("y", 0);
      let free = Variable::free("x");
      let mut children = atoms
        .into_iter()
        .map(|(a, b, c, relation)| {
          generated_atom(&bound, &free, a, b, c, relation)
        });
      let mut body = children.next().unwrap();
      for (index, child) in children.enumerate() {
        body = if connective_bits & (1 << index) == 0 {
          Formula::And(vec![body, child])
        } else {
          Formula::Or(vec![body, child])
        };
      }
      let original = Formula::Quantified(
        if universal { Quantifier::ForAll } else { Quantifier::Exists },
        vec![bound],
        Box::new(body),
      ).into_nnf().normalized();
      let eliminated = eliminate_quantifiers(original.clone()).unwrap();
      prop_assert!(!eliminated.contains_quantifier());

      for value in [
        Rational::integer(BigInt::from(-3)),
        Rational::integer(BigInt::from(-1)),
        Rational::zero(),
        Rational::new(BigInt::from(1), BigInt::from(2)).unwrap(),
        Rational::integer(BigInt::from(2)),
      ] {
        let mut environment = BTreeMap::from([(free.clone(), value)]);
        let before = evaluate_formula(&original, &mut environment);
        let after = evaluate_formula(&eliminated, &mut environment);
        prop_assert_eq!(before, after);
      }
    }
  }
}
