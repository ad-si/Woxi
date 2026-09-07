//! Transactional lowering from Woxi expressions to the linear formula IR.

use std::collections::{BTreeMap, BTreeSet};

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};
use woxi_reduce::{
  AffineTerm, Atom, Formula, Quantifier, Rational, Relation, Variable,
};

use crate::syntax::{BinaryOperator, ComparisonOp, Expr, UnaryOperator};

use super::{integer_value, rational_value};

#[derive(Default)]
struct LoweringContext {
  next_binder: u32,
  scopes: Vec<BTreeMap<String, Variable>>,
}

pub(super) fn formula_from_expr(expr: &Expr) -> Option<Formula> {
  LoweringContext::default()
    .formula(expr)
    .map(|formula| formula.into_nnf().normalized())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct LinearReduceRequest {
  pub(super) formula: Formula,
  pub(super) targets: Vec<Variable>,
  pub(super) domain: super::ReduceDomain,
}

pub(super) fn request_from_args(args: &[Expr]) -> Option<LinearReduceRequest> {
  if args.len() != 3
    || !matches!(
      super::request_domain(args),
      super::ReduceDomain::Reals
        | super::ReduceDomain::Rationals
        | super::ReduceDomain::Integers
    )
  {
    return None;
  }
  let target_names = quantifier_names(&args[1])?;
  if target_names.is_empty()
    || target_names.iter().collect::<BTreeSet<_>>().len() != target_names.len()
  {
    return None;
  }
  Some(LinearReduceRequest {
    formula: formula_from_expr(&args[0])?,
    targets: target_names.into_iter().map(Variable::free).collect(),
    domain: super::request_domain(args),
  })
}

impl LoweringContext {
  fn variable(&self, name: &str) -> Variable {
    self
      .scopes
      .iter()
      .rev()
      .find_map(|scope| scope.get(name))
      .cloned()
      .unwrap_or_else(|| Variable::free(name))
  }

  fn term(&mut self, expr: &Expr) -> Option<AffineTerm> {
    if let Some(value) = rational_value(expr) {
      return Some(AffineTerm::constant(value));
    }
    match expr {
      Expr::Identifier(name)
        if !matches!(
          name.as_str(),
          "True" | "False" | "I" | "Infinity" | "ComplexInfinity"
        ) =>
      {
        Some(AffineTerm::variable(self.variable(name)))
      }
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand,
      } => Some(
        self
          .term(operand)?
          .scaled(&Rational::integer(BigInt::from(-1))),
      ),
      Expr::BinaryOp { op, left, right } => match op {
        BinaryOperator::Plus => Some(self.term(left)?.add(&self.term(right)?)),
        BinaryOperator::Minus => {
          Some(self.term(left)?.subtract(&self.term(right)?))
        }
        BinaryOperator::Times => {
          self.term(left)?.checked_multiply(&self.term(right)?)
        }
        BinaryOperator::Divide => {
          self.term(left)?.checked_divide(&self.term(right)?)
        }
        BinaryOperator::Power => self.power(left, right),
        BinaryOperator::And
        | BinaryOperator::Or
        | BinaryOperator::StringJoin
        | BinaryOperator::Alternatives => None,
      },
      Expr::FunctionCall { name, args } => match name.as_str() {
        "Plus" => {
          let mut sum = AffineTerm::zero();
          for argument in args {
            sum = sum.add(&self.term(argument)?);
          }
          Some(sum)
        }
        "Times" => {
          let mut product = AffineTerm::constant(Rational::one());
          for argument in args {
            product = product.checked_multiply(&self.term(argument)?)?;
          }
          Some(product)
        }
        "Power" if args.len() == 2 => self.power(&args[0], &args[1]),
        _ => None,
      },
      _ => None,
    }
  }

  fn power(&mut self, base: &Expr, exponent: &Expr) -> Option<AffineTerm> {
    match integer_value(exponent)? {
      exponent if exponent.is_zero() => {
        Some(AffineTerm::constant(Rational::one()))
      }
      exponent if exponent.is_one() => self.term(base),
      _ => None,
    }
  }

  fn formula(&mut self, expr: &Expr) -> Option<Formula> {
    match expr {
      Expr::Identifier(name) if name == "True" => Some(Formula::True),
      Expr::Identifier(name) if name == "False" => Some(Formula::False),
      Expr::UnaryOp {
        op: UnaryOperator::Not,
        operand,
      } => Some(Formula::Not(Box::new(self.formula(operand)?))),
      Expr::BinaryOp { op, left, right } => match op {
        BinaryOperator::And => Some(Formula::And(vec![
          self.formula(left)?,
          self.formula(right)?,
        ])),
        BinaryOperator::Or => {
          Some(Formula::Or(vec![self.formula(left)?, self.formula(right)?]))
        }
        _ => None,
      },
      Expr::Comparison {
        operands,
        operators,
      } if operands.len() == operators.len() + 1 => {
        let mut formulas = Vec::with_capacity(operators.len());
        for (index, operator) in operators.iter().enumerate() {
          formulas.push(self.comparison(
            relation_from_comparison(*operator)?,
            &operands[index],
            &operands[index + 1],
          )?);
        }
        Some(Formula::And(formulas))
      }
      Expr::FunctionCall { name, args } => self.function_formula(name, args),
      _ => None,
    }
  }

  fn function_formula(&mut self, name: &str, args: &[Expr]) -> Option<Formula> {
    if let Some(relation) = relation_from_name(name) {
      return self.relation_chain(relation, args);
    }
    match name {
      "And" => Some(Formula::And(
        args
          .iter()
          .map(|argument| self.formula(argument))
          .collect::<Option<Vec<_>>>()?,
      )),
      "Or" => Some(Formula::Or(
        args
          .iter()
          .map(|argument| self.formula(argument))
          .collect::<Option<Vec<_>>>()?,
      )),
      "Not" if args.len() == 1 => {
        Some(Formula::Not(Box::new(self.formula(&args[0])?)))
      }
      "Nand" => Some(Formula::Not(Box::new(Formula::And(
        args
          .iter()
          .map(|argument| self.formula(argument))
          .collect::<Option<Vec<_>>>()?,
      )))),
      "Nor" => Some(Formula::Not(Box::new(Formula::Or(
        args
          .iter()
          .map(|argument| self.formula(argument))
          .collect::<Option<Vec<_>>>()?,
      )))),
      "Implies" if args.len() == 2 => Some(Formula::Or(vec![
        Formula::Not(Box::new(self.formula(&args[0])?)),
        self.formula(&args[1])?,
      ])),
      "Equivalent" if args.len() >= 2 => self.equivalent(args),
      "Xor" => self.xor(args),
      "Exists" | "ForAll" if args.len() == 2 => {
        self.quantified(name, &args[0], &args[1])
      }
      "Inequality" if args.len() >= 5 && args.len() % 2 == 1 => {
        let mut formulas = Vec::new();
        for index in (1..args.len()).step_by(2) {
          let Expr::Identifier(operator) = &args[index] else {
            return None;
          };
          formulas.push(self.comparison(
            relation_from_name(operator)?,
            &args[index - 1],
            &args[index + 1],
          )?);
        }
        Some(Formula::And(formulas))
      }
      "Divisible" if args.len() == 2 => {
        let modulus = integer_value(&args[1])?;
        if modulus <= BigInt::zero() {
          return None;
        }
        Some(Formula::Atom(Atom::divides(
          modulus,
          self.term(&args[0])?,
          false,
        )?))
      }
      _ => None,
    }
  }

  fn relation_chain(
    &mut self,
    relation: Relation,
    args: &[Expr],
  ) -> Option<Formula> {
    if args.len() < 2 {
      return None;
    }
    let mut formulas = Vec::new();
    if relation == Relation::NotEqual && args.len() > 2 {
      for left in 0..args.len() {
        for right in left + 1..args.len() {
          formulas.push(self.comparison(
            relation,
            &args[left],
            &args[right],
          )?);
        }
      }
    } else {
      for pair in args.windows(2) {
        formulas.push(self.comparison(relation, &pair[0], &pair[1])?);
      }
    }
    Some(Formula::And(formulas))
  }

  fn comparison(
    &mut self,
    relation: Relation,
    left: &Expr,
    right: &Expr,
  ) -> Option<Formula> {
    if matches!(relation, Relation::Equal | Relation::NotEqual) {
      if let Some(formula) = self.mod_comparison(relation, left, right) {
        return formula;
      }
      if let Some(formula) = self.mod_comparison(relation, right, left) {
        return formula;
      }
    }
    Some(Formula::Atom(Atom::Relation(
      relation,
      self.term(left)?.subtract(&self.term(right)?),
    )))
  }

  /// Returns `None` when this is not a `Mod[term, modulus] relation residue`
  /// shape. Invalid Mod shapes are subsequently rejected by ordinary affine
  /// lowering because `Mod` is not an affine term.
  fn mod_comparison(
    &mut self,
    relation: Relation,
    possible_mod: &Expr,
    possible_residue: &Expr,
  ) -> Option<Option<Formula>> {
    let Expr::FunctionCall { name, args } = possible_mod else {
      return None;
    };
    if name != "Mod" || args.len() != 2 {
      return None;
    }
    let modulus = integer_value(&args[1])?;
    if modulus <= BigInt::zero() {
      return Some(None);
    }
    let residue = integer_value(possible_residue)?;
    let equal = relation == Relation::Equal;
    if residue.is_negative() || residue >= modulus {
      return Some(Some(if equal { Formula::False } else { Formula::True }));
    }
    let term = self
      .term(&args[0])?
      .subtract(&AffineTerm::constant(Rational::integer(residue)));
    Some(Some(Formula::Atom(Atom::divides(modulus, term, !equal)?)))
  }

  fn quantified(
    &mut self,
    name: &str,
    variables: &Expr,
    body: &Expr,
  ) -> Option<Formula> {
    let names = quantifier_names(variables)?;
    if names.is_empty()
      || names.iter().collect::<BTreeSet<_>>().len() != names.len()
    {
      return None;
    }
    let mut scope = BTreeMap::new();
    let mut bound = Vec::with_capacity(names.len());
    for name in names {
      let binder = self.next_binder;
      self.next_binder = self.next_binder.checked_add(1)?;
      let variable = Variable::bound(&name, binder);
      scope.insert(name, variable.clone());
      bound.push(variable);
    }
    self.scopes.push(scope);
    let lowered_body = self.formula(body);
    self.scopes.pop();
    Some(Formula::Quantified(
      if name == "Exists" {
        Quantifier::Exists
      } else {
        Quantifier::ForAll
      },
      bound,
      Box::new(lowered_body?),
    ))
  }

  fn equivalent(&mut self, args: &[Expr]) -> Option<Formula> {
    let formulas = args
      .iter()
      .map(|argument| self.formula(argument))
      .collect::<Option<Vec<_>>>()?;
    Some(Formula::And(
      formulas
        .windows(2)
        .map(|pair| {
          Formula::Or(vec![
            Formula::And(vec![pair[0].clone(), pair[1].clone()]),
            Formula::And(vec![
              Formula::Not(Box::new(pair[0].clone())),
              Formula::Not(Box::new(pair[1].clone())),
            ]),
          ])
        })
        .collect(),
    ))
  }

  fn xor(&mut self, args: &[Expr]) -> Option<Formula> {
    let mut parity = Formula::False;
    for argument in args {
      let next = self.formula(argument)?;
      parity = Formula::Or(vec![
        Formula::And(vec![
          parity.clone(),
          Formula::Not(Box::new(next.clone())),
        ]),
        Formula::And(vec![Formula::Not(Box::new(parity)), next]),
      ]);
    }
    Some(parity)
  }
}

fn relation_from_comparison(operator: ComparisonOp) -> Option<Relation> {
  match operator {
    ComparisonOp::Equal => Some(Relation::Equal),
    ComparisonOp::NotEqual => Some(Relation::NotEqual),
    ComparisonOp::Less => Some(Relation::Less),
    ComparisonOp::LessEqual => Some(Relation::LessEqual),
    ComparisonOp::Greater => Some(Relation::Greater),
    ComparisonOp::GreaterEqual => Some(Relation::GreaterEqual),
    ComparisonOp::SameQ | ComparisonOp::UnsameQ => None,
  }
}

fn relation_from_name(name: &str) -> Option<Relation> {
  match name {
    "Equal" => Some(Relation::Equal),
    "Unequal" => Some(Relation::NotEqual),
    "Less" => Some(Relation::Less),
    "LessEqual" => Some(Relation::LessEqual),
    "Greater" => Some(Relation::Greater),
    "GreaterEqual" => Some(Relation::GreaterEqual),
    _ => None,
  }
}

fn quantifier_names(expr: &Expr) -> Option<Vec<String>> {
  match expr {
    Expr::Identifier(name) => Some(vec![name.clone()]),
    Expr::List(items) => items
      .iter()
      .map(|item| match item {
        Expr::Identifier(name) => Some(name.clone()),
        _ => None,
      })
      .collect(),
    _ => None,
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn id(name: &str) -> Expr {
    Expr::Identifier(name.to_string())
  }

  fn relation(left: Expr, operator: ComparisonOp, right: Expr) -> Expr {
    Expr::Comparison {
      operands: vec![left, right],
      operators: vec![operator],
    }
  }

  fn call(name: &str, args: Vec<Expr>) -> Expr {
    Expr::FunctionCall {
      name: name.to_string(),
      args: args.into(),
    }
  }

  #[test]
  fn affine_lowering_accepts_constants_and_rejects_nonlinearity() {
    let linear = relation(
      Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(3)),
          right: Box::new(id("x")),
        }),
        right: Box::new(Expr::Integer(2)),
      },
      ComparisonOp::LessEqual,
      Expr::Integer(7),
    );
    assert!(formula_from_expr(&linear).is_some());

    let nonlinear = relation(
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(id("x")),
        right: Box::new(id("y")),
      },
      ComparisonOp::Equal,
      Expr::Integer(0),
    );
    assert!(formula_from_expr(&nonlinear).is_none());
  }

  #[test]
  fn binders_are_capture_safe_and_alpha_invariant() {
    let make_formula = |bound_name: &str| {
      call(
        "Exists",
        vec![
          id(bound_name),
          Expr::BinaryOp {
            op: BinaryOperator::And,
            left: Box::new(relation(
              id(bound_name),
              ComparisonOp::Less,
              id("a"),
            )),
            right: Box::new(relation(
              id("z"),
              ComparisonOp::Greater,
              Expr::Integer(0),
            )),
          },
        ],
      )
    };
    let first = formula_from_expr(&make_formula("x")).unwrap();
    let second = formula_from_expr(&make_formula("y")).unwrap();
    assert_eq!(first, second);
    assert_eq!(
      first.free_variables(),
      BTreeSet::from([Variable::free("a"), Variable::free("z")])
    );
  }

  #[test]
  fn mod_lowering_handles_canonical_and_impossible_residues() {
    let congruence = relation(
      call("Mod", vec![id("x"), Expr::Integer(6)]),
      ComparisonOp::Equal,
      Expr::Integer(4),
    );
    assert!(matches!(
      formula_from_expr(&congruence),
      Some(Formula::Atom(Atom::Divides { modulus, .. }))
        if modulus == BigInt::from(6)
    ));

    let impossible = relation(
      call("Mod", vec![id("x"), Expr::Integer(6)]),
      ComparisonOp::Equal,
      Expr::Integer(7),
    );
    assert_eq!(formula_from_expr(&impossible), Some(Formula::False));
  }

  #[test]
  fn logical_surface_forms_finish_in_nnf() {
    let x_lt_zero = relation(id("x"), ComparisonOp::Less, Expr::Integer(0));
    let x_gt_one = relation(id("x"), ComparisonOp::Greater, Expr::Integer(1));
    for name in ["Xor", "Equivalent", "Nand", "Nor"] {
      let lowered = formula_from_expr(&call(
        name,
        vec![x_lt_zero.clone(), x_gt_one.clone()],
      ))
      .unwrap();
      assert!(lowered.is_nnf(), "surface form: {name}");
    }
  }

  #[test]
  fn duplicate_quantifier_names_and_variable_denominators_are_rejected() {
    let duplicate = call(
      "Exists",
      vec![
        Expr::List(vec![id("x"), id("x")].into()),
        relation(id("x"), ComparisonOp::Equal, Expr::Integer(0)),
      ],
    );
    assert!(formula_from_expr(&duplicate).is_none());

    let variable_denominator = relation(
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(id("x")),
        right: Box::new(id("y")),
      },
      ComparisonOp::Equal,
      Expr::Integer(1),
    );
    assert!(formula_from_expr(&variable_denominator).is_none());
  }

  #[test]
  fn reduce_request_requires_an_explicit_scoped_domain_and_unique_targets() {
    let equation = relation(id("x"), ComparisonOp::Equal, Expr::Integer(1));
    let request =
      request_from_args(&[equation.clone(), id("x"), id("Rationals")]).unwrap();
    assert_eq!(request.targets, vec![Variable::free("x")]);
    assert_eq!(request.domain, super::super::ReduceDomain::Rationals);
    assert!(request_from_args(&[equation.clone(), id("x")]).is_none());
    assert!(
      request_from_args(&[
        equation,
        Expr::List(vec![id("x"), id("x")].into()),
        id("Reals"),
      ])
      .is_none()
    );
  }

  #[test]
  fn complete_surface_grammar_matrix_lowers_transactionally() {
    for source in [
      "3*x + 2/5 <= y - 7",
      "Plus[Times[3, x], Rational[2, 5]] == y",
      "x^1 != 4",
      "-x < 2",
      "-1 < x <= 2",
      "Equal[x, y, z]",
      "Unequal[x, y, z]",
      "x < 0 && !(y >= 2)",
      "x < 0 || y > 2",
      "Xor[x < 0, y > 2, z == 1]",
      "Implies[x < 0, y > 2]",
      "Equivalent[x < 0, y > 2, z == 1]",
      "Nand[x < 0, y > 2]",
      "Nor[x < 0, y > 2]",
      "Exists[{x, y}, 2*x + y == a]",
      "ForAll[x, Exists[y, x < y || Mod[y, 3] == 1]]",
      "Mod[2*x + 1, 5] != 3",
      "Divisible[2*x + 1, 5]",
      "Inequality[0, Less, x, LessEqual, 3]",
    ] {
      let expression = crate::parse_to_expr(source).unwrap();
      assert!(
        formula_from_expr(&expression).is_some(),
        "failed to lower: {source}"
      );
    }
  }

  #[test]
  fn rejection_matrix_never_returns_a_partial_formula() {
    for source in [
      "x*y == 1",
      "x^2 < 3",
      "x/y == 1",
      "Sin[x] < 1",
      "x < 1.5",
      "Mod[x, -3] == 1",
      "Mod[x, 3] == 1/2",
      "Divisible[x/2, 3]",
      "Divisible[x, 0]",
      "Exists[{x, x}, x == 1]",
      "Exists[{x, 2}, x == 1]",
    ] {
      let expression = crate::parse_to_expr(source).unwrap();
      assert!(
        formula_from_expr(&expression).is_none(),
        "partially accepted out-of-scope input: {source}"
      );
    }
  }
}
