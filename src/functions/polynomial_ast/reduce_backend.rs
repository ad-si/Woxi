//! Systematic backends for `Reduce`.
//!
//! The built-in reducer remains the cheap, always-available implementation.
//! This module supplies a theory-neutral representation of exact algebraic
//! constraints and an optional native bridge to SMT-RAT's CAlC quantifier
//! elimination backend.  Keeping the representation independent from both
//! Woxi's evaluator AST and SMT-LIB makes it possible to add other decision
//! procedures (Presburger arithmetic, complex algebra, ...) without teaching
//! each of them about parser-specific expression shapes.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use num_bigint::{BigInt, Sign};
use num_traits::{One, ToPrimitive, Zero};

use crate::helpers::call;
use crate::syntax::{BinaryOperator, ComparisonOp, Expr, UnaryOperator};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ReduceDomain {
  Default,
  Reals,
  Integers,
  Rationals,
  Complexes,
  Modulus,
  Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ReduceRoute {
  RealAlgebraic,
  Integer,
  Rational,
  ComplexAlgebraic,
  Transcendental,
  Unsupported,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Rational {
  numerator: BigInt,
  denominator: BigInt,
}

impl Rational {
  fn new(mut numerator: BigInt, mut denominator: BigInt) -> Option<Self> {
    if denominator.is_zero() {
      return None;
    }
    if denominator.sign() == Sign::Minus {
      numerator = -numerator;
      denominator = -denominator;
    }
    Some(Self {
      numerator,
      denominator,
    })
  }

  fn integer(value: BigInt) -> Self {
    Self {
      numerator: value,
      denominator: BigInt::one(),
    }
  }

  fn reciprocal(&self) -> Option<Self> {
    Self::new(self.denominator.clone(), self.numerator.clone())
  }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum AlgebraicTerm {
  Number(Rational),
  Variable(String),
  Add(Vec<Self>),
  Multiply(Vec<Self>),
  Power(Box<Self>, u32),
}

impl AlgebraicTerm {
  fn variables(&self, output: &mut BTreeSet<String>) {
    match self {
      Self::Number(_) => {}
      Self::Variable(name) => {
        output.insert(name.clone());
      }
      Self::Add(terms) | Self::Multiply(terms) => {
        for term in terms {
          term.variables(output);
        }
      }
      Self::Power(base, _) => base.variables(output),
    }
  }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Relation {
  Equal,
  NotEqual,
  Less,
  LessEqual,
  Greater,
  GreaterEqual,
}

impl Relation {
  fn from_comparison(op: ComparisonOp) -> Option<Self> {
    match op {
      ComparisonOp::Equal => Some(Self::Equal),
      ComparisonOp::NotEqual => Some(Self::NotEqual),
      ComparisonOp::Less => Some(Self::Less),
      ComparisonOp::LessEqual => Some(Self::LessEqual),
      ComparisonOp::Greater => Some(Self::Greater),
      ComparisonOp::GreaterEqual => Some(Self::GreaterEqual),
      ComparisonOp::SameQ | ComparisonOp::UnsameQ => None,
    }
  }

  fn is_ordered(self) -> bool {
    matches!(
      self,
      Self::Less | Self::LessEqual | Self::Greater | Self::GreaterEqual
    )
  }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Quantifier {
  Exists,
  ForAll,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ConstraintFormula {
  True,
  False,
  Atom(Relation, AlgebraicTerm, AlgebraicTerm),
  And(Vec<Self>),
  Or(Vec<Self>),
  Xor(Vec<Self>),
  Not(Box<Self>),
  Quantified(Quantifier, Vec<String>, Box<Self>),
}

impl ConstraintFormula {
  fn all_variables(&self, output: &mut BTreeSet<String>) {
    match self {
      Self::True | Self::False => {}
      Self::Atom(_, left, right) => {
        left.variables(output);
        right.variables(output);
      }
      Self::And(items) | Self::Or(items) | Self::Xor(items) => {
        for item in items {
          item.all_variables(output);
        }
      }
      Self::Not(item) => item.all_variables(output),
      Self::Quantified(_, variables, body) => {
        output.extend(variables.iter().cloned());
        body.all_variables(output);
      }
    }
  }

  fn free_variables(&self) -> BTreeSet<String> {
    fn collect(
      formula: &ConstraintFormula,
      bound: &mut Vec<String>,
      output: &mut BTreeSet<String>,
    ) {
      match formula {
        ConstraintFormula::True | ConstraintFormula::False => {}
        ConstraintFormula::Atom(_, left, right) => {
          let mut variables = BTreeSet::new();
          left.variables(&mut variables);
          right.variables(&mut variables);
          output.extend(
            variables
              .into_iter()
              .filter(|name| !bound.iter().rev().any(|item| item == name)),
          );
        }
        ConstraintFormula::And(items)
        | ConstraintFormula::Or(items)
        | ConstraintFormula::Xor(items) => {
          for item in items {
            collect(item, bound, output);
          }
        }
        ConstraintFormula::Not(item) => collect(item, bound, output),
        ConstraintFormula::Quantified(_, variables, body) => {
          let old_len = bound.len();
          bound.extend(variables.iter().cloned());
          collect(body, bound, output);
          bound.truncate(old_len);
        }
      }
    }

    let mut output = BTreeSet::new();
    collect(self, &mut Vec::new(), &mut output);
    output
  }

  fn ordered_variables(&self, output: &mut BTreeSet<String>) {
    match self {
      Self::Atom(relation, left, right) if relation.is_ordered() => {
        left.variables(output);
        right.variables(output);
      }
      Self::Atom(_, _, _) | Self::True | Self::False => {}
      Self::And(items) | Self::Or(items) | Self::Xor(items) => {
        for item in items {
          item.ordered_variables(output);
        }
      }
      Self::Not(item) => item.ordered_variables(output),
      Self::Quantified(_, _, body) => body.ordered_variables(output),
    }
  }

  fn contains_quantifier(&self) -> bool {
    match self {
      Self::Quantified(_, _, _) => true,
      Self::And(items) | Self::Or(items) | Self::Xor(items) => {
        items.iter().any(Self::contains_quantifier)
      }
      Self::Not(item) => item.contains_quantifier(),
      Self::True | Self::False | Self::Atom(_, _, _) => false,
    }
  }
}

#[derive(Clone, Debug)]
pub(super) struct NormalizedReduce {
  formula: ConstraintFormula,
  pub(super) variables: Vec<String>,
  pub(super) domain: ReduceDomain,
}

impl NormalizedReduce {
  pub(super) fn contains_quantifier(&self) -> bool {
    self.formula.contains_quantifier()
  }
}

fn integer_value(expr: &Expr) -> Option<BigInt> {
  match expr {
    Expr::Integer(value) => Some(BigInt::from(*value)),
    Expr::BigInteger(value) => Some(value.clone()),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => integer_value(operand).map(std::ops::Neg::neg),
    _ => None,
  }
}

fn rational_value(expr: &Expr) -> Option<Rational> {
  if let Some(value) = integer_value(expr) {
    return Some(Rational::integer(value));
  }
  match expr {
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      Rational::new(integer_value(&args[0])?, integer_value(&args[1])?)
    }
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      let numerator = integer_value(left)?;
      let denominator = integer_value(right)?;
      Rational::new(numerator, denominator)
    }
    _ => None,
  }
}

fn term_from_expr(expr: &Expr) -> Option<AlgebraicTerm> {
  if let Some(value) = rational_value(expr) {
    return Some(AlgebraicTerm::Number(value));
  }
  match expr {
    Expr::Identifier(name)
      if !matches!(
        name.as_str(),
        "True" | "False" | "I" | "Infinity" | "ComplexInfinity"
      ) =>
    {
      Some(AlgebraicTerm::Variable(name.clone()))
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => Some(AlgebraicTerm::Multiply(vec![
      AlgebraicTerm::Number(Rational::integer(BigInt::from(-1))),
      term_from_expr(operand)?,
    ])),
    Expr::BinaryOp { op, left, right } => match op {
      BinaryOperator::Plus => Some(AlgebraicTerm::Add(vec![
        term_from_expr(left)?,
        term_from_expr(right)?,
      ])),
      BinaryOperator::Minus => Some(AlgebraicTerm::Add(vec![
        term_from_expr(left)?,
        AlgebraicTerm::Multiply(vec![
          AlgebraicTerm::Number(Rational::integer(BigInt::from(-1))),
          term_from_expr(right)?,
        ]),
      ])),
      BinaryOperator::Times => Some(AlgebraicTerm::Multiply(vec![
        term_from_expr(left)?,
        term_from_expr(right)?,
      ])),
      BinaryOperator::Divide => {
        let inverse = rational_value(right)?.reciprocal()?;
        Some(AlgebraicTerm::Multiply(vec![
          term_from_expr(left)?,
          AlgebraicTerm::Number(inverse),
        ]))
      }
      BinaryOperator::Power => {
        let exponent = integer_value(right)?.to_u32()?;
        Some(AlgebraicTerm::Power(
          Box::new(term_from_expr(left)?),
          exponent,
        ))
      }
      BinaryOperator::And
      | BinaryOperator::Or
      | BinaryOperator::StringJoin
      | BinaryOperator::Alternatives => None,
    },
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Plus" => Some(AlgebraicTerm::Add(
        args
          .iter()
          .map(term_from_expr)
          .collect::<Option<Vec<_>>>()?,
      )),
      "Times" => Some(AlgebraicTerm::Multiply(
        args
          .iter()
          .map(term_from_expr)
          .collect::<Option<Vec<_>>>()?,
      )),
      "Power" if args.len() == 2 => Some(AlgebraicTerm::Power(
        Box::new(term_from_expr(&args[0])?),
        integer_value(&args[1])?.to_u32()?,
      )),
      _ => None,
    },
    _ => None,
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

fn relation_chain(
  relation: Relation,
  args: &[Expr],
) -> Option<ConstraintFormula> {
  if args.len() < 2 {
    return None;
  }
  let mut atoms = Vec::new();
  if relation == Relation::NotEqual && args.len() > 2 {
    for left in 0..args.len() {
      for right in left + 1..args.len() {
        atoms.push(ConstraintFormula::Atom(
          relation,
          term_from_expr(&args[left])?,
          term_from_expr(&args[right])?,
        ));
      }
    }
  } else {
    for pair in args.windows(2) {
      atoms.push(ConstraintFormula::Atom(
        relation,
        term_from_expr(&pair[0])?,
        term_from_expr(&pair[1])?,
      ));
    }
  }
  Some(if atoms.len() == 1 {
    atoms.pop().unwrap()
  } else {
    ConstraintFormula::And(atoms)
  })
}

fn quantifier_variables(expr: &Expr) -> Option<Vec<String>> {
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

fn formula_from_expr(expr: &Expr) -> Option<ConstraintFormula> {
  match expr {
    Expr::Identifier(name) if name == "True" => Some(ConstraintFormula::True),
    Expr::Identifier(name) if name == "False" => Some(ConstraintFormula::False),
    Expr::UnaryOp {
      op: UnaryOperator::Not,
      operand,
    } => Some(ConstraintFormula::Not(Box::new(formula_from_expr(
      operand,
    )?))),
    Expr::BinaryOp { op, left, right } => match op {
      BinaryOperator::And => Some(ConstraintFormula::And(vec![
        formula_from_expr(left)?,
        formula_from_expr(right)?,
      ])),
      BinaryOperator::Or => Some(ConstraintFormula::Or(vec![
        formula_from_expr(left)?,
        formula_from_expr(right)?,
      ])),
      _ => None,
    },
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == operators.len() + 1 => {
      let mut atoms = Vec::with_capacity(operators.len());
      for (index, operator) in operators.iter().enumerate() {
        atoms.push(ConstraintFormula::Atom(
          Relation::from_comparison(*operator)?,
          term_from_expr(&operands[index])?,
          term_from_expr(&operands[index + 1])?,
        ));
      }
      Some(if atoms.len() == 1 {
        atoms.pop().unwrap()
      } else {
        ConstraintFormula::And(atoms)
      })
    }
    Expr::FunctionCall { name, args } => {
      if let Some(relation) = relation_from_name(name) {
        return relation_chain(relation, args);
      }
      match name.as_str() {
        "And" => Some(ConstraintFormula::And(
          args
            .iter()
            .map(formula_from_expr)
            .collect::<Option<Vec<_>>>()?,
        )),
        "Or" => Some(ConstraintFormula::Or(
          args
            .iter()
            .map(formula_from_expr)
            .collect::<Option<Vec<_>>>()?,
        )),
        "Xor" => Some(ConstraintFormula::Xor(
          args
            .iter()
            .map(formula_from_expr)
            .collect::<Option<Vec<_>>>()?,
        )),
        "Not" if args.len() == 1 => Some(ConstraintFormula::Not(Box::new(
          formula_from_expr(&args[0])?,
        ))),
        "Nand" => {
          Some(ConstraintFormula::Not(Box::new(ConstraintFormula::And(
            args
              .iter()
              .map(formula_from_expr)
              .collect::<Option<Vec<_>>>()?,
          ))))
        }
        "Nor" => Some(ConstraintFormula::Not(Box::new(ConstraintFormula::Or(
          args
            .iter()
            .map(formula_from_expr)
            .collect::<Option<Vec<_>>>()?,
        )))),
        "Implies" if args.len() == 2 => Some(ConstraintFormula::Or(vec![
          ConstraintFormula::Not(Box::new(formula_from_expr(&args[0])?)),
          formula_from_expr(&args[1])?,
        ])),
        "Equivalent" if args.len() >= 2 => {
          let formulas = args
            .iter()
            .map(formula_from_expr)
            .collect::<Option<Vec<_>>>()?;
          let mut pairs = Vec::new();
          for pair in formulas.windows(2) {
            pairs.push(ConstraintFormula::Or(vec![
              ConstraintFormula::And(vec![pair[0].clone(), pair[1].clone()]),
              ConstraintFormula::And(vec![
                ConstraintFormula::Not(Box::new(pair[0].clone())),
                ConstraintFormula::Not(Box::new(pair[1].clone())),
              ]),
            ]));
          }
          Some(ConstraintFormula::And(pairs))
        }
        "Exists" | "ForAll" if args.len() == 2 => {
          let quantifier = if name == "Exists" {
            Quantifier::Exists
          } else {
            Quantifier::ForAll
          };
          Some(ConstraintFormula::Quantified(
            quantifier,
            quantifier_variables(&args[0])?,
            Box::new(formula_from_expr(&args[1])?),
          ))
        }
        "Inequality" if args.len() >= 5 && args.len() % 2 == 1 => {
          let mut atoms = Vec::new();
          for index in (1..args.len()).step_by(2) {
            let Expr::Identifier(operator) = &args[index] else {
              return None;
            };
            atoms.push(ConstraintFormula::Atom(
              relation_from_name(operator)?,
              term_from_expr(&args[index - 1])?,
              term_from_expr(&args[index + 1])?,
            ));
          }
          Some(ConstraintFormula::And(atoms))
        }
        _ => None,
      }
    }
    _ => None,
  }
}

fn request_domain(args: &[Expr]) -> ReduceDomain {
  if args.len() < 3 {
    return ReduceDomain::Default;
  }
  match &args[2] {
    Expr::Identifier(name) => match name.as_str() {
      "Reals" => ReduceDomain::Reals,
      "Integers" => ReduceDomain::Integers,
      "Rationals" => ReduceDomain::Rationals,
      "Complexes" => ReduceDomain::Complexes,
      _ => ReduceDomain::Unknown,
    },
    Expr::Rule { pattern, .. } if matches!(pattern.as_ref(), Expr::Identifier(name) if name == "Modulus") => {
      ReduceDomain::Modulus
    }
    _ => ReduceDomain::Unknown,
  }
}

fn contains_transcendental(expr: &Expr) -> bool {
  match expr {
    Expr::FunctionCall { name, args } => {
      if matches!(
        name.as_str(),
        "Sin"
          | "Cos"
          | "Tan"
          | "Cot"
          | "Sec"
          | "Csc"
          | "Sinh"
          | "Cosh"
          | "Tanh"
          | "Coth"
          | "Sech"
          | "Csch"
          | "Exp"
          | "Log"
          | "ArcSin"
          | "ArcCos"
          | "ArcTan"
          | "ArcCot"
          | "ArcSec"
          | "ArcCsc"
      ) {
        return true;
      }
      args.iter().any(contains_transcendental)
    }
    Expr::BinaryOp { left, right, .. } => {
      contains_transcendental(left) || contains_transcendental(right)
    }
    Expr::UnaryOp { operand, .. } => contains_transcendental(operand),
    Expr::Comparison { operands, .. } | Expr::CompoundExpr(operands) => {
      operands.iter().any(contains_transcendental)
    }
    Expr::List(items) => items.iter().any(contains_transcendental),
    _ => false,
  }
}

pub(super) fn classify_reduce_request(args: &[Expr]) -> ReduceRoute {
  if args.len() < 2 || args.len() > 3 {
    return ReduceRoute::Unsupported;
  }
  match request_domain(args) {
    ReduceDomain::Integers | ReduceDomain::Modulus => ReduceRoute::Integer,
    ReduceDomain::Rationals => ReduceRoute::Rational,
    ReduceDomain::Complexes => ReduceRoute::ComplexAlgebraic,
    ReduceDomain::Unknown => ReduceRoute::Unsupported,
    ReduceDomain::Default | ReduceDomain::Reals => {
      if contains_transcendental(&args[0]) {
        return ReduceRoute::Transcendental;
      }
      let Some(formula) = formula_from_expr(&args[0]) else {
        return ReduceRoute::Unsupported;
      };
      if request_domain(args) == ReduceDomain::Reals {
        return ReduceRoute::RealAlgebraic;
      }
      let mut all = BTreeSet::new();
      formula.all_variables(&mut all);
      let mut ordered = BTreeSet::new();
      formula.ordered_variables(&mut ordered);
      if !ordered.is_empty() && all.is_subset(&ordered) {
        ReduceRoute::RealAlgebraic
      } else {
        ReduceRoute::ComplexAlgebraic
      }
    }
  }
}

pub(super) fn normalize_real_reduce(args: &[Expr]) -> Option<NormalizedReduce> {
  if classify_reduce_request(args) != ReduceRoute::RealAlgebraic {
    return None;
  }
  Some(NormalizedReduce {
    formula: formula_from_expr(&args[0])?,
    variables: quantifier_variables(&args[1])?,
    domain: request_domain(args),
  })
}

#[derive(Debug)]
struct SmtNames {
  original_to_smt: BTreeMap<String, String>,
  smt_to_original: BTreeMap<String, String>,
}

impl SmtNames {
  fn for_request(request: &NormalizedReduce) -> Self {
    let mut variables = BTreeSet::new();
    request.formula.all_variables(&mut variables);
    let mut ordered_variables = request
      .variables
      .iter()
      .filter(|variable| variables.remove(*variable))
      .cloned()
      .collect::<Vec<_>>();
    ordered_variables.extend(variables);
    let mut original_to_smt = BTreeMap::new();
    let mut smt_to_original = BTreeMap::new();
    for (index, original) in ordered_variables.into_iter().enumerate() {
      let smt = format!("woxi_v_{index}");
      original_to_smt.insert(original.clone(), smt.clone());
      smt_to_original.insert(smt, original);
    }
    Self {
      original_to_smt,
      smt_to_original,
    }
  }

  fn smt<'a>(&'a self, original: &'a str) -> &'a str {
    self
      .original_to_smt
      .get(original)
      .map_or(original, String::as_str)
  }
}

fn smt_integer(value: &BigInt) -> String {
  if value.sign() == Sign::Minus {
    format!("(- {})", -value)
  } else {
    value.to_string()
  }
}

fn term_to_smt(term: &AlgebraicTerm, names: &SmtNames) -> String {
  match term {
    AlgebraicTerm::Number(value) => {
      if value.denominator.is_one() {
        smt_integer(&value.numerator)
      } else {
        format!(
          "(/ {} {})",
          smt_integer(&value.numerator),
          smt_integer(&value.denominator)
        )
      }
    }
    AlgebraicTerm::Variable(name) => names.smt(name).to_string(),
    AlgebraicTerm::Add(terms) => {
      let values = terms
        .iter()
        .map(|term| term_to_smt(term, names))
        .collect::<Vec<_>>();
      if values.is_empty() {
        "0".to_string()
      } else if values.len() == 1 {
        values[0].clone()
      } else {
        format!("(+ {})", values.join(" "))
      }
    }
    AlgebraicTerm::Multiply(terms) => {
      let values = terms
        .iter()
        .map(|term| term_to_smt(term, names))
        .collect::<Vec<_>>();
      if values.is_empty() {
        "1".to_string()
      } else if values.len() == 1 {
        values[0].clone()
      } else {
        format!("(* {})", values.join(" "))
      }
    }
    AlgebraicTerm::Power(base, exponent) => {
      format!("(^ {} {exponent})", term_to_smt(base, names))
    }
  }
}

fn formula_to_smt(formula: &ConstraintFormula, names: &SmtNames) -> String {
  match formula {
    ConstraintFormula::True => "true".to_string(),
    ConstraintFormula::False => "false".to_string(),
    ConstraintFormula::Atom(relation, left, right) => {
      let operator = match relation {
        Relation::Equal => "=",
        Relation::NotEqual => "distinct",
        Relation::Less => "<",
        Relation::LessEqual => "<=",
        Relation::Greater => ">",
        Relation::GreaterEqual => ">=",
      };
      format!(
        "({operator} {} {})",
        term_to_smt(left, names),
        term_to_smt(right, names)
      )
    }
    ConstraintFormula::And(items)
    | ConstraintFormula::Or(items)
    | ConstraintFormula::Xor(items) => {
      let operator = match formula {
        ConstraintFormula::And(_) => "and",
        ConstraintFormula::Or(_) => "or",
        ConstraintFormula::Xor(_) => "xor",
        _ => unreachable!(),
      };
      if items.is_empty() {
        return if operator == "and" {
          "true".to_string()
        } else {
          "false".to_string()
        };
      }
      format!(
        "({operator} {})",
        items
          .iter()
          .map(|item| formula_to_smt(item, names))
          .collect::<Vec<_>>()
          .join(" ")
      )
    }
    ConstraintFormula::Not(item) => {
      format!("(not {})", formula_to_smt(item, names))
    }
    ConstraintFormula::Quantified(quantifier, variables, body) => {
      let operator = match quantifier {
        Quantifier::Exists => "exists",
        Quantifier::ForAll => "forall",
      };
      let bindings = variables
        .iter()
        .map(|variable| format!("({} Real)", names.smt(variable)))
        .collect::<Vec<_>>()
        .join(" ");
      format!("({operator} ({bindings}) {})", formula_to_smt(body, names))
    }
  }
}

fn build_smt_query(request: &NormalizedReduce) -> (String, SmtNames) {
  let names = SmtNames::for_request(request);
  let mut query = String::from("(set-logic NRA)\n");
  for variable in request.formula.free_variables() {
    query
      .push_str(&format!("(declare-fun {} () Real)\n", names.smt(&variable)));
  }
  query.push_str("(assert ");
  query.push_str(&formula_to_smt(&request.formula, &names));
  query.push_str(")\n(apply qe)\n");
  (query, names)
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum SExpression {
  Atom(String),
  List(Vec<Self>),
}

fn tokenize_smt(input: &str) -> Result<Vec<String>, String> {
  let mut tokens = Vec::new();
  let mut chars = input.chars().peekable();
  while let Some(ch) = chars.next() {
    match ch {
      ch if ch.is_whitespace() => {}
      ';' => {
        for next in chars.by_ref() {
          if next == '\n' {
            break;
          }
        }
      }
      '(' | ')' => tokens.push(ch.to_string()),
      '|' => {
        let mut token = String::new();
        let mut closed = false;
        for next in chars.by_ref() {
          if next == '|' {
            closed = true;
            break;
          }
          token.push(next);
        }
        if !closed {
          return Err("unterminated SMT-LIB quoted symbol".to_string());
        }
        tokens.push(token);
      }
      '"' => {
        let mut token = String::from("\"");
        let mut closed = false;
        while let Some(next) = chars.next() {
          token.push(next);
          if next == '"' {
            if chars.peek() == Some(&'"') {
              token.push(chars.next().unwrap());
            } else {
              closed = true;
              break;
            }
          }
        }
        if !closed {
          return Err("unterminated SMT-LIB string".to_string());
        }
        tokens.push(token);
      }
      _ => {
        let mut token = String::from(ch);
        while let Some(next) = chars.peek() {
          if next.is_whitespace() || matches!(next, '(' | ')' | ';') {
            break;
          }
          token.push(chars.next().unwrap());
        }
        tokens.push(token);
      }
    }
  }
  Ok(tokens)
}

fn parse_s_expression(
  tokens: &[String],
  position: &mut usize,
) -> Result<SExpression, String> {
  let Some(token) = tokens.get(*position) else {
    return Err("unexpected end of SMT-LIB output".to_string());
  };
  *position += 1;
  if token == "(" {
    let mut items = Vec::new();
    while tokens.get(*position).is_some_and(|next| next != ")") {
      items.push(parse_s_expression(tokens, position)?);
    }
    if tokens.get(*position).is_none() {
      return Err("unterminated SMT-LIB list".to_string());
    }
    *position += 1;
    Ok(SExpression::List(items))
  } else if token == ")" {
    Err("unexpected ')' in SMT-LIB output".to_string())
  } else {
    Ok(SExpression::Atom(token.clone()))
  }
}

fn parse_smt_expressions(input: &str) -> Result<Vec<SExpression>, String> {
  let tokens = tokenize_smt(input)?;
  let mut position = 0;
  let mut expressions = Vec::new();
  while position < tokens.len() {
    expressions.push(parse_s_expression(&tokens, &mut position)?);
  }
  Ok(expressions)
}

fn expand_smt_lets(
  expression: &SExpression,
  environment: &BTreeMap<String, SExpression>,
) -> Result<SExpression, String> {
  match expression {
    SExpression::Atom(name) => Ok(
      environment
        .get(name)
        .cloned()
        .unwrap_or_else(|| expression.clone()),
    ),
    SExpression::List(items) if items.is_empty() => Ok(expression.clone()),
    SExpression::List(items) if matches!(items.first(), Some(SExpression::Atom(name)) if name == "let") =>
    {
      if items.len() != 3 {
        return Err("malformed SMT-LIB let expression".to_string());
      }
      let SExpression::List(bindings) = &items[1] else {
        return Err("malformed SMT-LIB let bindings".to_string());
      };
      let mut extended = environment.clone();
      for binding in bindings {
        let SExpression::List(pair) = binding else {
          return Err("malformed SMT-LIB let binding".to_string());
        };
        if pair.len() != 2 {
          return Err("malformed SMT-LIB let binding".to_string());
        }
        let SExpression::Atom(name) = &pair[0] else {
          return Err("SMT-LIB let binding name is not a symbol".to_string());
        };
        // SMT-LIB let bindings are simultaneous, so their right-hand sides
        // are expanded in the outer environment.
        extended.insert(name.clone(), expand_smt_lets(&pair[1], environment)?);
      }
      expand_smt_lets(&items[2], &extended)
    }
    SExpression::List(items) if matches!(items.first(), Some(SExpression::Atom(name)) if name == "!") =>
    {
      if items.len() < 2 {
        return Err("malformed annotated SMT-LIB expression".to_string());
      }
      expand_smt_lets(&items[1], environment)
    }
    SExpression::List(items) => Ok(SExpression::List(
      items
        .iter()
        .map(|item| expand_smt_lets(item, environment))
        .collect::<Result<Vec<_>, _>>()?,
    )),
  }
}

fn decimal_rational(text: &str) -> Option<Rational> {
  let (negative, unsigned) = text
    .strip_prefix('-')
    .map_or((false, text), |rest| (true, rest));
  let (whole, fractional) = unsigned.split_once('.')?;
  if whole.is_empty()
    || fractional.is_empty()
    || !whole.chars().all(|ch| ch.is_ascii_digit())
    || !fractional.chars().all(|ch| ch.is_ascii_digit())
  {
    return None;
  }
  let denominator = BigInt::from(10u8).pow(fractional.len() as u32);
  let mut numerator = whole.parse::<BigInt>().ok()? * &denominator
    + fractional.parse::<BigInt>().ok()?;
  if negative {
    numerator = -numerator;
  }
  Rational::new(numerator, denominator)
}

fn bigint_expr(value: &BigInt) -> Expr {
  value
    .to_i128()
    .map_or_else(|| Expr::BigInteger(value.clone()), Expr::Integer)
}

fn rational_expr(value: &Rational) -> Expr {
  if value.denominator.is_one() {
    bigint_expr(&value.numerator)
  } else {
    call(
      "Rational",
      vec![
        bigint_expr(&value.numerator),
        bigint_expr(&value.denominator),
      ],
    )
  }
}

fn atom_integer(expression: &SExpression) -> Option<BigInt> {
  let SExpression::Atom(atom) = expression else {
    return None;
  };
  atom.parse::<BigInt>().ok()
}

fn fold_call(name: &str, items: Vec<Expr>, identity: Expr) -> Expr {
  match items.len() {
    0 => identity,
    1 => items.into_iter().next().unwrap(),
    _ => call(name, items),
  }
}

fn term_from_smt(
  expression: &SExpression,
  names: &SmtNames,
) -> Result<Expr, String> {
  match expression {
    SExpression::Atom(atom) => {
      if let Ok(value) = atom.parse::<BigInt>() {
        return Ok(bigint_expr(&value));
      }
      if let Some(value) = decimal_rational(atom) {
        return Ok(rational_expr(&value));
      }
      Ok(Expr::Identifier(
        names
          .smt_to_original
          .get(atom)
          .cloned()
          .unwrap_or_else(|| atom.clone()),
      ))
    }
    SExpression::List(items) if items.is_empty() => {
      Err("empty SMT-LIB term".to_string())
    }
    SExpression::List(items) => {
      let SExpression::Atom(operator) = &items[0] else {
        return Err("SMT-LIB term operator is not a symbol".to_string());
      };
      let arguments = &items[1..];
      match operator.as_str() {
        "+" => Ok(fold_call(
          "Plus",
          arguments
            .iter()
            .map(|item| term_from_smt(item, names))
            .collect::<Result<Vec<_>, _>>()?,
          Expr::Integer(0),
        )),
        "*" => Ok(fold_call(
          "Times",
          arguments
            .iter()
            .map(|item| term_from_smt(item, names))
            .collect::<Result<Vec<_>, _>>()?,
          Expr::Integer(1),
        )),
        "-" if arguments.len() == 1 => Ok(Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand: Box::new(term_from_smt(&arguments[0], names)?),
        }),
        "-" if arguments.len() >= 2 => {
          let mut values = arguments.iter();
          let first = term_from_smt(values.next().unwrap(), names)?;
          let mut difference = first;
          for right in values {
            difference = Expr::BinaryOp {
              op: BinaryOperator::Minus,
              left: Box::new(difference),
              right: Box::new(term_from_smt(right, names)?),
            };
          }
          Ok(difference)
        }
        "/" if arguments.len() == 2 => Ok(Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(term_from_smt(&arguments[0], names)?),
          right: Box::new(term_from_smt(&arguments[1], names)?),
        }),
        "^" if arguments.len() == 2 => Ok(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(term_from_smt(&arguments[0], names)?),
          right: Box::new(term_from_smt(&arguments[1], names)?),
        }),
        "to_real" if arguments.len() == 1 => {
          term_from_smt(&arguments[0], names)
        }
        "ite" if arguments.len() == 3 => Ok(call(
          "Piecewise",
          vec![
            Expr::List(
              vec![Expr::List(
                vec![
                  term_from_smt(&arguments[1], names)?,
                  formula_from_smt(&arguments[0], names)?,
                ]
                .into(),
              )]
              .into(),
            ),
            term_from_smt(&arguments[2], names)?,
          ],
        )),
        "root" if arguments.len() == 3 => {
          let polynomial = term_from_smt(&arguments[0], names)?;
          let index = atom_integer(&arguments[1])
            .and_then(|value| value.to_i128())
            .filter(|value| *value >= 1)
            .ok_or_else(|| "invalid SMT-RAT indexed-root index".to_string())?;
          let SExpression::Atom(root_variable) = &arguments[2] else {
            return Err("invalid SMT-RAT indexed-root variable".to_string());
          };
          let root_variable = names
            .smt_to_original
            .get(root_variable)
            .map_or(root_variable.as_str(), String::as_str);
          let body = crate::syntax::substitute_variable(
            &polynomial,
            root_variable,
            &Expr::Slot(1),
          );
          Ok(call(
            "Root",
            vec![
              Expr::Function {
                body: Box::new(body),
              },
              Expr::Integer(index),
              Expr::Integer(0),
            ],
          ))
        }
        _ => Err(format!("unsupported SMT-LIB term operator '{operator}'")),
      }
    }
  }
}

fn comparison_formula(
  operator: &str,
  arguments: &[SExpression],
  names: &SmtNames,
) -> Result<Expr, String> {
  if arguments.len() < 2 {
    return Err(format!("SMT-LIB '{operator}' needs at least two operands"));
  }
  let comparison = match operator {
    "=" => ComparisonOp::Equal,
    "distinct" => ComparisonOp::NotEqual,
    "<" => ComparisonOp::Less,
    "<=" => ComparisonOp::LessEqual,
    ">" => ComparisonOp::Greater,
    ">=" => ComparisonOp::GreaterEqual,
    _ => return Err(format!("unsupported SMT-LIB relation '{operator}'")),
  };
  let terms = arguments
    .iter()
    .map(|item| term_from_smt(item, names))
    .collect::<Result<Vec<_>, _>>()?;
  if comparison == ComparisonOp::NotEqual && terms.len() > 2 {
    let mut pairs = Vec::new();
    for left in 0..terms.len() {
      for right in left + 1..terms.len() {
        pairs.push(Expr::Comparison {
          operands: vec![terms[left].clone(), terms[right].clone()],
          operators: vec![comparison],
        });
      }
    }
    return Ok(fold_call(
      "And",
      pairs,
      Expr::Identifier("True".to_string()),
    ));
  }
  Ok(Expr::Comparison {
    operands: terms,
    operators: vec![comparison; arguments.len() - 1],
  })
}

fn formula_from_smt(
  expression: &SExpression,
  names: &SmtNames,
) -> Result<Expr, String> {
  match expression {
    SExpression::Atom(atom) if atom == "true" => {
      Ok(Expr::Identifier("True".to_string()))
    }
    SExpression::Atom(atom) if atom == "false" => {
      Ok(Expr::Identifier("False".to_string()))
    }
    SExpression::Atom(atom) => Err(format!(
      "unexpected SMT-LIB atom '{atom}' where a formula was expected"
    )),
    SExpression::List(items) if items.is_empty() => {
      Err("empty SMT-LIB formula".to_string())
    }
    SExpression::List(items) => {
      let SExpression::Atom(operator) = &items[0] else {
        return Err("SMT-LIB formula operator is not a symbol".to_string());
      };
      let arguments = &items[1..];
      match operator.as_str() {
        "and" | "or" | "xor" => {
          let name = match operator.as_str() {
            "and" => "And",
            "or" => "Or",
            _ => "Xor",
          };
          let identity = if operator == "and" { "True" } else { "False" };
          Ok(fold_call(
            name,
            arguments
              .iter()
              .map(|item| formula_from_smt(item, names))
              .collect::<Result<Vec<_>, _>>()?,
            Expr::Identifier(identity.to_string()),
          ))
        }
        "not" if arguments.len() == 1 => {
          Ok(call("Not", vec![formula_from_smt(&arguments[0], names)?]))
        }
        "=>" if arguments.len() == 2 => Ok(call(
          "Implies",
          vec![
            formula_from_smt(&arguments[0], names)?,
            formula_from_smt(&arguments[1], names)?,
          ],
        )),
        "=" | "distinct" | "<" | "<=" | ">" | ">=" => {
          comparison_formula(operator, arguments, names)
        }
        "ite" if arguments.len() == 3 => Ok(call(
          "Or",
          vec![
            call(
              "And",
              vec![
                formula_from_smt(&arguments[0], names)?,
                formula_from_smt(&arguments[1], names)?,
              ],
            ),
            call(
              "And",
              vec![
                call("Not", vec![formula_from_smt(&arguments[0], names)?]),
                formula_from_smt(&arguments[2], names)?,
              ],
            ),
          ],
        )),
        "exists" | "forall" if arguments.len() == 2 => {
          let SExpression::List(bindings) = &arguments[0] else {
            return Err("malformed SMT-LIB quantifier bindings".to_string());
          };
          let mut variables = Vec::new();
          for binding in bindings {
            let SExpression::List(pair) = binding else {
              return Err("malformed SMT-LIB quantifier binding".to_string());
            };
            let [SExpression::Atom(variable), SExpression::Atom(sort)] =
              pair.as_slice()
            else {
              return Err("malformed SMT-LIB quantifier binding".to_string());
            };
            if sort != "Real" {
              return Err(format!(
                "unsupported SMT-LIB quantified sort '{sort}'"
              ));
            }
            variables.push(Expr::Identifier(
              names
                .smt_to_original
                .get(variable)
                .cloned()
                .unwrap_or_else(|| variable.clone()),
            ));
          }
          Ok(call(
            if operator == "exists" {
              "Exists"
            } else {
              "ForAll"
            },
            vec![
              Expr::List(variables.into()),
              formula_from_smt(&arguments[1], names)?,
            ],
          ))
        }
        _ => Err(format!("unsupported SMT-LIB formula operator '{operator}'")),
      }
    }
  }
}

fn parse_smtrat_output(output: &str, names: &SmtNames) -> Result<Expr, String> {
  let expressions = parse_smt_expressions(output)?;
  let candidate = expressions
    .iter()
    .rev()
    .find(|expression| {
      !matches!(expression, SExpression::Atom(atom) if matches!(atom.as_str(), "success" | "sat" | "unsat"))
    })
    .ok_or_else(|| "SMT-RAT returned no quantifier-elimination formula".to_string())?;
  if matches!(candidate, SExpression::Atom(atom) if atom == "unknown") {
    return Err("SMT-RAT returned unknown".to_string());
  }
  let expanded = expand_smt_lets(candidate, &BTreeMap::new())?;
  formula_from_smt(&expanded, names)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BackendMode {
  Auto,
  Internal,
  SmtRat,
}

pub(super) fn backend_mode() -> BackendMode {
  match std::env::var("WOXI_REDUCE_BACKEND")
    .unwrap_or_else(|_| "auto".to_string())
    .to_ascii_lowercase()
    .as_str()
  {
    "internal" | "builtin" | "off" => BackendMode::Internal,
    "smtrat" | "calc" => BackendMode::SmtRat,
    _ => BackendMode::Auto,
  }
}

#[cfg(not(target_arch = "wasm32"))]
fn run_smtrat(query: &str) -> Result<String, String> {
  use std::time::Duration;

  let executable = std::env::var("WOXI_SMTRAT")
    .unwrap_or_else(|_| "smtrat-shared".to_string());
  let timeout = std::env::var("WOXI_SMTRAT_TIMEOUT_MS")
    .ok()
    .and_then(|value| value.parse::<u64>().ok())
    .map_or(Duration::from_secs(30), Duration::from_millis);
  run_smtrat_executable(query, &executable, timeout)
}

#[cfg(not(target_arch = "wasm32"))]
fn run_smtrat_executable(
  query: &str,
  executable: &str,
  timeout: std::time::Duration,
) -> Result<String, String> {
  use std::io::{Read, Write};
  use std::process::{Command, Stdio};
  use std::time::{Duration, Instant};

  let mut child = Command::new(executable)
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .spawn()
    .map_err(|error| format!("cannot start '{executable}': {error}"))?;

  let mut stdin = child
    .stdin
    .take()
    .ok_or_else(|| "cannot open SMT-RAT stdin".to_string())?;
  let mut stdout = child
    .stdout
    .take()
    .ok_or_else(|| "cannot open SMT-RAT stdout".to_string())?;
  let mut stderr = child
    .stderr
    .take()
    .ok_or_else(|| "cannot open SMT-RAT stderr".to_string())?;

  // Drain both output pipes while the solver is running.  SMT-RAT normally
  // waits for the complete query, but diagnostics or a large early result
  // must not be able to fill a pipe and deadlock the input writer.
  let stdout_reader = std::thread::spawn(move || {
    let mut bytes = Vec::new();
    stdout.read_to_end(&mut bytes).map(|_| bytes)
  });
  let stderr_reader = std::thread::spawn(move || {
    let mut bytes = Vec::new();
    stderr.read_to_end(&mut bytes).map(|_| bytes)
  });
  let query = query.as_bytes().to_vec();
  let stdin_writer = std::thread::spawn(move || {
    let result = stdin.write_all(&query);
    drop(stdin);
    result
  });

  let start = Instant::now();
  let status = loop {
    match child.try_wait() {
      Ok(Some(status)) => break Ok(status),
      Ok(None) if start.elapsed() < timeout => {
        std::thread::sleep(Duration::from_millis(5));
      }
      Ok(None) => {
        let _ = child.kill();
        let _ = child.wait();
        break Err(format!(
          "SMT-RAT exceeded the {} ms time limit",
          timeout.as_millis()
        ));
      }
      Err(error) => {
        let _ = child.kill();
        let _ = child.wait();
        break Err(format!("cannot wait for SMT-RAT: {error}"));
      }
    }
  };

  let write_result = stdin_writer
    .join()
    .map_err(|_| "SMT-RAT stdin writer panicked".to_string())?;
  let stdout = stdout_reader
    .join()
    .map_err(|_| "SMT-RAT stdout reader panicked".to_string())?
    .map_err(|error| format!("cannot read SMT-RAT stdout: {error}"))?;
  let stderr = stderr_reader
    .join()
    .map_err(|_| "SMT-RAT stderr reader panicked".to_string())?
    .map_err(|error| format!("cannot read SMT-RAT stderr: {error}"))?;
  let stdout = String::from_utf8(stdout)
    .map_err(|error| format!("SMT-RAT stdout is not UTF-8: {error}"))?;
  let status = status?;
  write_result
    .map_err(|error| format!("cannot write SMT-RAT query: {error}"))?;

  let diagnostic = String::from_utf8_lossy(&stderr);
  if !smtrat_exit_status_is_success(status) {
    return Err(format!(
      "SMT-RAT exited with {status}{}",
      if diagnostic.trim().is_empty() {
        String::new()
      } else {
        format!(": {}", diagnostic.trim())
      }
    ));
  }

  // SMT-LIB convention uses exit status 10 for sat and 20 for unsat.  QE
  // emits a formula and deliberately uses those statuses too, so a non-zero
  // code is not by itself an error.
  if stdout.trim().is_empty() {
    return Err(format!(
      "SMT-RAT exited with {status} without a formula{}",
      if diagnostic.trim().is_empty() {
        String::new()
      } else {
        format!(": {}", diagnostic.trim())
      }
    ));
  }
  Ok(stdout)
}

#[cfg(not(target_arch = "wasm32"))]
fn smtrat_exit_status_is_success(status: std::process::ExitStatus) -> bool {
  status.success() || matches!(status.code(), Some(10 | 20))
}

#[cfg(target_arch = "wasm32")]
fn run_smtrat(_query: &str) -> Result<String, String> {
  Err("SMT-RAT subprocesses are unavailable on wasm32".to_string())
}

pub(super) fn try_smtrat_reduce(
  request: &NormalizedReduce,
) -> Result<Expr, String> {
  try_smtrat_reduce_with(request, run_smtrat)
}

fn try_smtrat_reduce_with(
  request: &NormalizedReduce,
  runner: impl FnOnce(&str) -> Result<String, String>,
) -> Result<Expr, String> {
  let (query, names) = build_smt_query(request);
  let output = runner(&query)?;
  parse_smtrat_output(&output, &names)
}

fn flatten_boolean(name: &str, expression: &Expr, output: &mut Vec<Expr>) {
  match expression {
    Expr::FunctionCall {
      name: nested_name,
      args,
    } if nested_name == name => {
      for item in args {
        flatten_boolean(name, item, output);
      }
    }
    Expr::BinaryOp { op, left, right }
      if (name == "And" && *op == BinaryOperator::And)
        || (name == "Or" && *op == BinaryOperator::Or) =>
    {
      flatten_boolean(name, left, output);
      flatten_boolean(name, right, output);
    }
    other => output.push(other.clone()),
  }
}

fn shape_boolean_expression(expression: &Expr) -> Expr {
  match expression {
    Expr::FunctionCall { name, args } if name == "And" || name == "Or" => {
      let identity = if name == "And" { "True" } else { "False" };
      let absorbing = if name == "And" { "False" } else { "True" };
      let mut items = Vec::new();
      for item in args {
        let item = shape_boolean_expression(item);
        flatten_boolean(name, &item, &mut items);
      }
      if items.iter().any(
        |item| matches!(item, Expr::Identifier(value) if value == absorbing),
      ) {
        return Expr::Identifier(absorbing.to_string());
      }
      items.retain(
        |item| !matches!(item, Expr::Identifier(value) if value == identity),
      );
      items.sort_by(|left, right| {
        crate::syntax::expr_to_string(left)
          .cmp(&crate::syntax::expr_to_string(right))
      });
      items.dedup_by(|left, right| {
        crate::syntax::expr_to_string(left)
          == crate::syntax::expr_to_string(right)
      });
      match items.len() {
        0 => Expr::Identifier(identity.to_string()),
        1 => items.pop().unwrap(),
        _ => call(name, items),
      }
    }
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .into_iter()
        .map(shape_boolean_expression)
        .collect::<Vec<_>>()
        .into(),
    },
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(shape_boolean_expression(left)),
      right: Box::new(shape_boolean_expression(right)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(shape_boolean_expression(operand)),
    },
    other => other.clone(),
  }
}

fn ordered_variables_in_result(
  expression: &Expr,
  output: &mut HashSet<String>,
) {
  match expression {
    Expr::Comparison {
      operands,
      operators,
    } => {
      for (index, operator) in operators.iter().enumerate() {
        if matches!(
          operator,
          ComparisonOp::Less
            | ComparisonOp::LessEqual
            | ComparisonOp::Greater
            | ComparisonOp::GreaterEqual
        ) {
          super::simplify::collect_variables(&operands[index], output);
          super::simplify::collect_variables(&operands[index + 1], output);
        }
      }
    }
    Expr::FunctionCall { args, .. } | Expr::List(args) => {
      for argument in args {
        ordered_variables_in_result(argument, output);
      }
    }
    Expr::BinaryOp { left, right, .. } => {
      ordered_variables_in_result(left, output);
      ordered_variables_in_result(right, output);
    }
    Expr::UnaryOp { operand, .. } => {
      ordered_variables_in_result(operand, output);
    }
    _ => {}
  }
}

pub(super) fn shape_smtrat_result(
  expression: &Expr,
  request: &NormalizedReduce,
) -> Expr {
  let mut expression = shape_boolean_expression(expression);

  // With an implicit domain, ordered input constraints make their free
  // quantities real.  QE can replace inequalities with equations or remove
  // them entirely, so explicitly retain the real-domain assumptions for any
  // free variable that no longer occurs in an ordered output atom.  Bound
  // variables are deliberately excluded.  An explicit Reals argument already
  // supplies the domain and needs no residual membership condition.
  if request.domain == ReduceDomain::Default
    && !matches!(&expression, Expr::Identifier(value) if value == "False")
  {
    let input_variables = request.formula.free_variables();
    let mut ordered_output_variables = HashSet::new();
    ordered_variables_in_result(&expression, &mut ordered_output_variables);
    let memberships = input_variables
      .into_iter()
      .filter(|variable| !ordered_output_variables.contains(variable))
      .map(|variable| {
        call(
          "Element",
          vec![
            Expr::Identifier(variable),
            Expr::Identifier("Reals".to_string()),
          ],
        )
      })
      .collect::<Vec<_>>();
    if !memberships.is_empty() {
      let mut conditions = Vec::with_capacity(memberships.len() + 1);
      conditions.push(expression);
      conditions.extend(memberships);
      expression =
        fold_call("And", conditions, Expr::Identifier("True".to_string()));
      expression = shape_boolean_expression(&expression);
    }
  }

  crate::evaluator::evaluate_expr_to_expr(&expression).unwrap_or(expression)
}

#[cfg(test)]
mod tests {
  use super::*;

  fn id(name: &str) -> Expr {
    Expr::Identifier(name.to_string())
  }

  #[test]
  fn normalizes_quantified_real_polynomial_formula() {
    let formula = Expr::FunctionCall {
      name: "Exists".to_string(),
      args: vec![
        Expr::List(vec![id("x")].into()),
        Expr::Comparison {
          operands: vec![
            Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(id("x")),
              right: Box::new(Expr::Integer(2)),
            },
            id("a"),
          ],
          operators: vec![ComparisonOp::LessEqual],
        },
      ]
      .into(),
    };
    let args = vec![formula, id("a"), id("Reals")];
    let request = normalize_real_reduce(&args).unwrap();
    assert!(request.contains_quantifier());
    let (query, _) = build_smt_query(&request);
    assert!(query.contains("(exists ((woxi_v_1 Real))"));
    assert!(query.contains("(<= (^ woxi_v_1 2) woxi_v_0)"));
    assert!(query.ends_with("(apply qe)\n"));
  }

  #[test]
  fn default_domain_does_not_send_pure_complex_equations_to_real_solver() {
    let equation = Expr::Comparison {
      operands: vec![
        Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(id("x")),
            right: Box::new(Expr::Integer(2)),
          }),
          right: Box::new(Expr::Integer(1)),
        },
        Expr::Integer(0),
      ],
      operators: vec![ComparisonOp::Equal],
    };
    assert_eq!(
      classify_reduce_request(&[equation, id("x")]),
      ReduceRoute::ComplexAlgebraic
    );
  }

  #[test]
  fn implicit_real_route_requires_every_variable_to_be_ordered() {
    let real_only = Expr::Comparison {
      operands: vec![id("x"), Expr::Integer(0)],
      operators: vec![ComparisonOp::Greater],
    };
    assert_eq!(
      classify_reduce_request(&[real_only, id("x")]),
      ReduceRoute::RealAlgebraic
    );

    let mixed = Expr::BinaryOp {
      op: BinaryOperator::And,
      left: Box::new(Expr::Comparison {
        operands: vec![id("x"), Expr::Integer(0)],
        operators: vec![ComparisonOp::Greater],
      }),
      right: Box::new(Expr::Comparison {
        operands: vec![id("y"), Expr::Integer(0)],
        operators: vec![ComparisonOp::Equal],
      }),
    };
    assert_eq!(
      classify_reduce_request(&[
        mixed,
        Expr::List(vec![id("x"), id("y")].into())
      ]),
      ReduceRoute::ComplexAlgebraic
    );
  }

  #[test]
  fn classifies_non_real_theories_onto_specialist_routes() {
    let equation = Expr::Comparison {
      operands: vec![id("x"), Expr::Integer(1)],
      operators: vec![ComparisonOp::Equal],
    };
    assert_eq!(
      classify_reduce_request(&[equation.clone(), id("x"), id("Integers")]),
      ReduceRoute::Integer
    );
    assert_eq!(
      classify_reduce_request(&[equation.clone(), id("x"), id("Rationals")]),
      ReduceRoute::Rational
    );
    assert_eq!(
      classify_reduce_request(&[equation.clone(), id("x"), id("Complexes")]),
      ReduceRoute::ComplexAlgebraic
    );

    let transcendental = Expr::Comparison {
      operands: vec![call("Sin", vec![id("x")]), Expr::Integer(0)],
      operators: vec![ComparisonOp::Greater],
    };
    assert_eq!(
      classify_reduce_request(&[transcendental, id("x")]),
      ReduceRoute::Transcendental
    );

    assert_eq!(
      classify_reduce_request(&[equation, id("x"), id("Booleans")]),
      ReduceRoute::Unsupported
    );
  }

  #[test]
  fn parses_boolean_formula_and_exact_rationals() {
    let names = SmtNames {
      original_to_smt: BTreeMap::from([(
        "x".to_string(),
        "woxi_v_0".to_string(),
      )]),
      smt_to_original: BTreeMap::from([(
        "woxi_v_0".to_string(),
        "x".to_string(),
      )]),
    };
    let expression = parse_smtrat_output(
      "(or (< woxi_v_0 (- 2)) (>= woxi_v_0 (/ 3 2)))\n",
      &names,
    )
    .unwrap();
    assert_eq!(
      crate::syntax::expr_to_string(&expression),
      "x < -2 || x >= 3/2"
    );
  }

  #[test]
  fn runs_the_full_query_parse_and_shape_pipeline() {
    let formula = Expr::Comparison {
      operands: vec![
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(id("x")),
          right: Box::new(Expr::Integer(2)),
        },
        Expr::Integer(1),
      ],
      operators: vec![ComparisonOp::Greater],
    };
    let request = normalize_real_reduce(&[formula, id("x")]).unwrap();
    let result = try_smtrat_reduce_with(&request, |query| {
      assert!(query.contains("(> (^ woxi_v_0 2) 1)"));
      Ok("(or (< woxi_v_0 (- 1)) (> woxi_v_0 1))".to_string())
    })
    .unwrap();
    let shaped = shape_smtrat_result(&result, &request);
    assert_eq!(crate::syntax::expr_to_string(&shaped), "x < -1 || x > 1");
  }

  #[test]
  fn result_shaping_retains_implicit_real_domains_but_not_bound_variables() {
    let positive = Expr::Comparison {
      operands: vec![id("x"), Expr::Integer(0)],
      operators: vec![ComparisonOp::Greater],
    };
    let implicit = normalize_real_reduce(&[positive.clone(), id("x")]).unwrap();
    let shaped = shape_smtrat_result(&id("True"), &implicit);
    assert_eq!(crate::syntax::expr_to_string(&shaped), "Element[x, Reals]");

    let explicit =
      normalize_real_reduce(&[positive, id("x"), id("Reals")]).unwrap();
    let shaped = shape_smtrat_result(&id("True"), &explicit);
    assert_eq!(crate::syntax::expr_to_string(&shaped), "True");

    let quantified = Expr::FunctionCall {
      name: "Exists".to_string(),
      args: vec![
        id("x"),
        Expr::Comparison {
          operands: vec![id("x"), Expr::Integer(0)],
          operators: vec![ComparisonOp::Greater],
        },
      ]
      .into(),
    };
    let request = normalize_real_reduce(&[quantified, id("x")]).unwrap();
    let shaped = shape_smtrat_result(&id("True"), &request);
    assert_eq!(crate::syntax::expr_to_string(&shaped), "True");
  }

  #[cfg(all(unix, not(target_arch = "wasm32")))]
  #[test]
  fn native_adapter_round_trips_standard_io() {
    use std::time::Duration;

    let output =
      run_smtrat_executable("true\n", "/bin/cat", Duration::from_secs(2))
        .unwrap();
    assert_eq!(output, "true\n");
  }

  #[cfg(all(unix, not(target_arch = "wasm32")))]
  #[test]
  fn native_adapter_accepts_smtlib_success_exit_codes() {
    use std::os::unix::process::ExitStatusExt;

    assert!(smtrat_exit_status_is_success(
      std::process::ExitStatus::from_raw(0)
    ));
    assert!(smtrat_exit_status_is_success(
      std::process::ExitStatus::from_raw(10 << 8)
    ));
    assert!(smtrat_exit_status_is_success(
      std::process::ExitStatus::from_raw(20 << 8)
    ));
    assert!(!smtrat_exit_status_is_success(
      std::process::ExitStatus::from_raw(1 << 8)
    ));
  }

  #[test]
  fn parses_let_and_indexed_root() {
    let names = SmtNames {
      original_to_smt: BTreeMap::from([
        ("a".to_string(), "woxi_v_0".to_string()),
        ("x".to_string(), "woxi_v_1".to_string()),
      ]),
      smt_to_original: BTreeMap::from([
        ("woxi_v_0".to_string(), "a".to_string()),
        ("woxi_v_1".to_string(), "x".to_string()),
      ]),
    };
    let expression = parse_smtrat_output(
      "(let ((r (root (+ (^ woxi_v_1 2) woxi_v_0) 1 woxi_v_1))) (= woxi_v_1 r))",
      &names,
    )
    .unwrap();
    let rendered = crate::syntax::expr_to_string(&expression);
    assert!(rendered.contains("Root"));
    assert!(rendered.contains("#1^2"));
    assert!(rendered.contains("a"));
  }
}
