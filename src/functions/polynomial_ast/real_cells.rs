//! Decide a boolean combination of univariate real inequalities by sign
//! tests: the real roots and poles of every comparison (and the breakpoints
//! of the `Abs` terms in it) cut the line into open intervals and points, on
//! each of which every comparison keeps one truth value.
//!
//! `Reduce` presents the pieces where the formula holds as a disjunction of
//! intervals; `Probability` measures them under a distribution.

use super::*;
use crate::functions::math_ast::try_eval_to_f64;

/// A boolean formula over the comparison atoms.
enum Formula {
  Const(bool),
  Atom(usize),
  And(Vec<Formula>),
  Or(Vec<Formula>),
  Not(Box<Formula>),
  Xor(Vec<Formula>),
  Nand(Vec<Formula>),
  Nor(Vec<Formula>),
  Implies(Box<Formula>, Box<Formula>),
  Equivalent(Vec<Formula>),
}

impl Formula {
  fn holds(&self, atoms: &[bool]) -> bool {
    match self {
      Formula::Const(b) => *b,
      Formula::Atom(i) => atoms[*i],
      Formula::And(fs) => fs.iter().all(|f| f.holds(atoms)),
      Formula::Or(fs) => fs.iter().any(|f| f.holds(atoms)),
      Formula::Not(f) => !f.holds(atoms),
      Formula::Xor(fs) => fs.iter().filter(|f| f.holds(atoms)).count() % 2 == 1,
      Formula::Nand(fs) => !fs.iter().all(|f| f.holds(atoms)),
      Formula::Nor(fs) => !fs.iter().any(|f| f.holds(atoms)),
      Formula::Implies(a, b) => !a.holds(atoms) || b.holds(atoms),
      Formula::Equivalent(fs) => {
        let values: Vec<bool> = fs.iter().map(|f| f.holds(atoms)).collect();
        values.iter().all(|v| *v) || values.iter().all(|v| !*v)
      }
    }
  }
}

/// One comparison `diff op 0`, with the poles of `diff`.
struct Atom {
  diff: Expr,
  op: CompOp,
  poles: Vec<f64>,
}

fn comp_op(op: &ComparisonOp) -> Option<CompOp> {
  Some(match op {
    ComparisonOp::Equal => CompOp::Equal,
    ComparisonOp::NotEqual => CompOp::NotEqual,
    ComparisonOp::Less => CompOp::Less,
    ComparisonOp::LessEqual => CompOp::LessEqual,
    ComparisonOp::Greater => CompOp::Greater,
    ComparisonOp::GreaterEqual => CompOp::GreaterEqual,
    _ => return None,
  })
}

fn comp_op_named(name: &str) -> Option<CompOp> {
  Some(match name {
    "Equal" => CompOp::Equal,
    "Unequal" => CompOp::NotEqual,
    "Less" => CompOp::Less,
    "LessEqual" => CompOp::LessEqual,
    "Greater" => CompOp::Greater,
    "GreaterEqual" => CompOp::GreaterEqual,
    _ => return None,
  })
}

struct Parser {
  atoms: Vec<(Expr, CompOp)>,
}

impl Parser {
  fn chain(&mut self, operands: &[Expr], ops: &[CompOp]) -> Formula {
    let mut parts: Vec<Formula> = Vec::new();
    for (i, op) in ops.iter().enumerate() {
      self.atoms.push((
        call(
          "Plus",
          vec![
            operands[i].clone(),
            call("Times", vec![Expr::Integer(-1), operands[i + 1].clone()]),
          ],
        ),
        *op,
      ));
      parts.push(Formula::Atom(self.atoms.len() - 1));
    }
    if parts.len() == 1 {
      parts.pop().unwrap()
    } else {
      Formula::And(parts)
    }
  }

  fn all(&mut self, args: &[Expr]) -> Option<Vec<Formula>> {
    args.iter().map(|a| self.parse(a)).collect()
  }

  fn parse(&mut self, e: &Expr) -> Option<Formula> {
    match e {
      Expr::Identifier(b) if b == "True" => Some(Formula::Const(true)),
      Expr::Identifier(b) if b == "False" => Some(Formula::Const(false)),
      Expr::Comparison {
        operands,
        operators,
      } if operands.len() >= 2 && operators.len() + 1 == operands.len() => {
        let ops: Option<Vec<CompOp>> = operators.iter().map(comp_op).collect();
        Some(self.chain(operands, &ops?))
      }
      Expr::BinaryOp { op, left, right } => {
        let parts = vec![self.parse(left)?, self.parse(right)?];
        match op {
          BinaryOperator::And => Some(Formula::And(parts)),
          BinaryOperator::Or => Some(Formula::Or(parts)),
          _ => None,
        }
      }
      Expr::UnaryOp {
        op: UnaryOperator::Not,
        operand,
      } => Some(Formula::Not(Box::new(self.parse(operand)?))),
      Expr::FunctionCall { name, args } => match name.as_str() {
        "And" => Some(Formula::And(self.all(args)?)),
        "Or" => Some(Formula::Or(self.all(args)?)),
        "Xor" => Some(Formula::Xor(self.all(args)?)),
        "Nand" => Some(Formula::Nand(self.all(args)?)),
        "Nor" => Some(Formula::Nor(self.all(args)?)),
        "Equivalent" => Some(Formula::Equivalent(self.all(args)?)),
        "Not" if args.len() == 1 => {
          Some(Formula::Not(Box::new(self.parse(&args[0])?)))
        }
        "Implies" if args.len() == 2 => Some(Formula::Implies(
          Box::new(self.parse(&args[0])?),
          Box::new(self.parse(&args[1])?),
        )),
        "Inequality" if args.len() >= 3 && args.len() % 2 == 1 => {
          let mut operands = Vec::new();
          let mut ops = Vec::new();
          for (i, a) in args.iter().enumerate() {
            if i % 2 == 0 {
              operands.push(a.clone());
            } else if let Expr::Identifier(n) = a {
              ops.push(comp_op_named(n)?);
            } else {
              return None;
            }
          }
          Some(self.chain(&operands, &ops))
        }
        _ if args.len() >= 2 => {
          let op = comp_op_named(name)?;
          Some(self.chain(args, &vec![op; args.len() - 1]))
        }
        _ => None,
      },
      _ => None,
    }
  }
}

/// The distinct `Abs[u]` subexpressions of `e` that involve `var`.
fn abs_terms(e: &Expr, var: &str, out: &mut Vec<Expr>) {
  if let Expr::FunctionCall { name, args } = e
    && name == "Abs"
    && args.len() == 1
    && !crate::functions::calculus_ast::is_constant_wrt(&args[0], var)
  {
    let key = crate::syntax::expr_to_string(e);
    if !out.iter().any(|o| crate::syntax::expr_to_string(o) == key) {
      out.push(e.clone());
    }
    return;
  }
  match e {
    Expr::FunctionCall { args, .. } => {
      args.iter().for_each(|a| abs_terms(a, var, out));
    }
    Expr::List(items) => items.iter().for_each(|a| abs_terms(a, var, out)),
    Expr::BinaryOp { left, right, .. } => {
      abs_terms(left, var, out);
      abs_terms(right, var, out);
    }
    Expr::UnaryOp { operand, .. } => abs_terms(operand, var, out),
    _ => {}
  }
}

/// The real roots and poles of the rational function `e` of `var`.
fn rational_roots(e: &Expr, var: &str) -> Option<Vec<(f64, Expr, bool)>> {
  let together = crate::evaluator::evaluate_function_call_ast(
    "Together",
    std::slice::from_ref(e),
  )
  .ok()?;
  let (numerator, denominator) = super::together::extract_num_den(&together);
  let numerator = expand_and_combine(&numerator);
  let denominator = expand_and_combine(&denominator);
  let numerator_degree = max_power_int(&numerator, var)?;
  let denominator_degree = max_power_int(&denominator, var)?;
  if numerator_degree < 0 || denominator_degree < 0 {
    return None;
  }
  real_roots_and_poles(
    &numerator,
    numerator_degree,
    &denominator,
    denominator_degree,
    var,
  )
}

/// The pieces of the real line for a formula: the critical points in
/// increasing order (numeric value and exact form) and whether the formula
/// holds on each piece — interval 0, point 0, interval 1, …, point n-1,
/// interval n.
pub struct RealCells {
  pub points: Vec<(f64, Expr)>,
  pub holds: Vec<bool>,
}

/// Decide `formula`, a boolean combination of comparisons between exact
/// rational functions of `var` (possibly with `Abs`), on every piece of the
/// real line. `None` when the formula has another shape, mentions another
/// symbol or an inexact number, or a root can't be found.
pub fn univariate_real_cells(formula: &Expr, var: &str) -> Option<RealCells> {
  let mut parser = Parser { atoms: Vec::new() };
  let tree = parser.parse(formula)?;
  let mut points: Vec<(f64, Expr)> = Vec::new();
  let mut add_point = |x: f64, e: Expr| {
    if !points
      .iter()
      .any(|(p, _)| (p - x).abs() <= 1e-12 * (1.0 + x.abs()))
    {
      points.push((x, e));
    }
  };
  let mut atoms: Vec<Atom> = Vec::new();
  for (diff, op) in parser.atoms {
    let diff = crate::evaluator::evaluate_expr_to_expr(&diff).ok()?;
    let mut symbols = Vec::new();
    constraint_variables(&diff, &mut symbols);
    if symbols.iter().any(|s| s != var)
      || crate::functions::math_ast::contains_inexact_real(&diff)
    {
      return None;
    }
    let mut abs = Vec::new();
    abs_terms(&diff, var, &mut abs);
    if abs.len() > 4 {
      return None;
    }
    let mut poles = Vec::new();
    // The breakpoints of each `Abs`, then the roots of every sign choice.
    for a in &abs {
      if let Expr::FunctionCall { args, .. } = a {
        for (x, e, pole) in rational_roots(&args[0], var)? {
          if !pole {
            add_point(x, e);
          }
        }
      }
    }
    for signs in 0..(1u32 << abs.len()) {
      let rules: Vec<Expr> = abs
        .iter()
        .enumerate()
        .map(|(i, a)| {
          let Expr::FunctionCall { args, .. } = a else {
            unreachable!()
          };
          let value = if signs & (1 << i) == 0 {
            args[0].clone()
          } else {
            call("Times", vec![Expr::Integer(-1), args[0].clone()])
          };
          Expr::Rule {
            pattern: Box::new(a.clone()),
            replacement: Box::new(value),
          }
        })
        .collect();
      let variant = if rules.is_empty() {
        diff.clone()
      } else {
        crate::evaluator::evaluate_expr_to_expr(&call(
          "ReplaceAll",
          vec![diff.clone(), Expr::List(rules.into())],
        ))
        .ok()?
      };
      for (x, e, pole) in rational_roots(&variant, var)? {
        if pole {
          poles.push(x);
        }
        add_point(x, e);
      }
    }
    atoms.push(Atom { diff, op, poles });
  }
  points
    .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

  // The truth of every atom at `x`; `None` when an atom has a pole there.
  let atom_values = |x: f64, at_point: bool| -> Option<Option<Vec<bool>>> {
    let mut values = Vec::with_capacity(atoms.len());
    for atom in &atoms {
      if at_point
        && atom
          .poles
          .iter()
          .any(|p| (p - x).abs() <= 1e-12 * (1.0 + x.abs()))
      {
        return Some(None);
      }
      let mut v = try_eval_to_f64(&crate::syntax::substitute_variable(
        &atom.diff,
        var,
        &Expr::Real(x),
      ))?;
      if !v.is_finite() {
        return None;
      }
      if at_point && v.abs() <= 1e-9 * (1.0 + x.abs()) {
        v = 0.0;
      }
      values.push(match atom.op {
        CompOp::Equal => v == 0.0,
        CompOp::NotEqual => v != 0.0,
        CompOp::Less => v < 0.0,
        CompOp::LessEqual => v <= 0.0,
        CompOp::Greater => v > 0.0,
        CompOp::GreaterEqual => v >= 0.0,
      });
    }
    Some(Some(values))
  };

  let n = points.len();
  let mut holds = Vec::with_capacity(2 * n + 1);
  for i in 0..=n {
    let x = match (i, n) {
      (_, 0) => 0.0,
      (0, _) => points[0].0 - 1.0,
      (i, n) if i == n => points[n - 1].0 + 1.0,
      (i, _) => f64::midpoint(points[i - 1].0, points[i].0),
    };
    holds.push(atom_values(x, false)?.is_some_and(|v| tree.holds(&v)));
    if i < n {
      holds
        .push(atom_values(points[i].0, true)?.is_some_and(|v| tree.holds(&v)));
    }
  }
  Some(RealCells { points, holds })
}

impl RealCells {
  /// The maximal runs of pieces where the formula holds, as index ranges.
  fn runs(&self) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut i = 0;
    while i < self.holds.len() {
      if !self.holds[i] {
        i += 1;
        continue;
      }
      let first = i;
      while i + 1 < self.holds.len() && self.holds[i + 1] {
        i += 1;
      }
      runs.push((first, i));
      i += 1;
    }
    runs
  }

  /// The solution set as `Reduce` writes it: `True`, `False`, or the
  /// disjunction of its intervals and points in increasing order.
  pub fn to_condition(&self, var: &str) -> Expr {
    if self.holds.iter().all(|h| *h) {
      return bool_expr(true);
    }
    build_or(
      self
        .runs()
        .into_iter()
        .map(|(first, last)| run_condition(var, first, last, &self.points))
        .collect(),
    )
  }

  /// The solution set as intervals: each end is `None` when unbounded, or
  /// the end point and whether it is included. A single point is an interval
  /// with equal closed ends.
  pub fn intervals(&self) -> Vec<RealInterval> {
    let n = self.points.len();
    self
      .runs()
      .into_iter()
      .map(|(first, last)| {
        let lower = if first % 2 == 1 {
          Some((self.points[first / 2].1.clone(), true))
        } else if first == 0 {
          None
        } else {
          Some((self.points[first / 2 - 1].1.clone(), false))
        };
        let upper = if last % 2 == 1 {
          Some((self.points[last / 2].1.clone(), true))
        } else if last / 2 == n {
          None
        } else {
          Some((self.points[last / 2].1.clone(), false))
        };
        RealInterval { lower, upper }
      })
      .collect()
  }
}

/// An interval of the real line; see [`RealCells::intervals`].
pub struct RealInterval {
  pub lower: Option<(Expr, bool)>,
  pub upper: Option<(Expr, bool)>,
}
