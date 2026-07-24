#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator, unevaluated};

use crate::functions::calculus_ast::is_constant_wrt;
use crate::functions::math_ast::rat_reduce;

// ─── Rational exponent helpers ─────────────────────────────────────

/// A simple rational number (numerator, denominator) with denominator > 0.
#[derive(Clone, Copy, Debug)]
pub struct Rat {
  n: i128,
  d: i128,
}

impl Rat {
  fn new(n: i128, d: i128) -> Self {
    assert!(d != 0);
    let (n, d) = rat_reduce(n, d);
    Rat { n: n, d: d }
  }

  fn zero() -> Self {
    Rat { n: 0, d: 1 }
  }

  fn from_int(v: i128) -> Self {
    Rat { n: v, d: 1 }
  }

  fn add(self, other: Rat) -> Rat {
    Rat::new(self.n * other.d + other.n * self.d, self.d * other.d)
  }

  fn max(self, other: Rat) -> Rat {
    // Compare self.n/self.d vs other.n/other.d
    if self.n * other.d >= other.n * self.d {
      self
    } else {
      other
    }
  }

  fn min(self, other: Rat) -> Rat {
    if self.n * other.d <= other.n * self.d {
      self
    } else {
      other
    }
  }

  /// Returns the integer value if this rational is a whole number.
  pub fn as_int(self) -> Option<i128> {
    if self.d == 1 { Some(self.n) } else { None }
  }

  pub fn to_expr(self) -> Expr {
    if self.d == 1 {
      Expr::Integer(self.n)
    } else {
      Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(self.n), Expr::Integer(self.d)].into(),
      }
    }
  }
}

/// Try to interpret an Expr as a rational number.
fn expr_to_rat(e: &Expr) -> Option<Rat> {
  match e {
    Expr::Integer(n) => Some(Rat::from_int(*n)),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        Some(Rat::new(*n, *d))
      } else {
        None
      }
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      let r = expr_to_rat(operand)?;
      Some(Rat::new(-r.n, r.d))
    }
    _ => None,
  }
}

/// Convenience wrapper: returns max power as integer, or None if non-integer.
pub fn max_power_int(expr: &Expr, var: &str) -> Option<i128> {
  max_power(expr, var).and_then(|r| r.as_int())
}

// ─── Exponent ───────────────────────────────────────────────────────

/// Exponent[expr, var] - Returns the maximum power of var in expr
pub fn exponent_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Exponent expects 2 or 3 arguments".into(),
    ));
  }

  // SeriesData input: reduce via `Normal` first so the polynomial path can
  // measure its degree (Exponent[Series[Exp[x],{x,0,5}],x] -> 5), matching
  // wolframscript.
  if let Expr::FunctionCall { name, .. } = &args[0]
    && name == "SeriesData"
  {
    let normalized = crate::evaluator::evaluate_function_call_ast(
      "Normal",
      &[args[0].clone()],
    )?;
    let mut new_args = args.to_vec();
    new_args[0] = normalized;
    return exponent_ast(&new_args);
  }

  let var = match &args[1] {
    Expr::Identifier(name) => name.as_str(),
    // A non-symbol form (a bare number, or a compound like x + 1 or Sin[x])
    // is treated as an atomic unit: Exponent returns the highest power of
    // that form across the additive terms, without expanding. A form that
    // never appears (any bare number) gives 0. Matches wolframscript.
    _ => {
      return exponent_of_form(args);
    }
  };

  // Exponent[0, x] -> -Infinity
  if matches!(&args[0], Expr::Integer(0)) {
    return Ok(Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(Expr::Identifier("Infinity".to_string())),
    });
  }

  // Expand and combine like terms first to handle things like (x^2+1)^3-1
  let expanded = expand_and_combine(&args[0]);

  // Determine if we need Max (default), Min, or List
  let use_min =
    args.len() == 3 && matches!(&args[2], Expr::Identifier(s) if s == "Min");
  let use_list =
    args.len() == 3 && matches!(&args[2], Expr::Identifier(s) if s == "List");

  if use_list {
    match collect_powers(&expanded, var) {
      Some(powers) => {
        let mut sorted: Vec<Rat> = powers;
        // Sort ascending by rational value
        sorted.sort_by(|a, b| {
          let lhs = a.n * b.d;
          let rhs = b.n * a.d;
          lhs.cmp(&rhs)
        });
        // Deduplicate
        sorted.dedup_by(|a, b| a.n * b.d == b.n * a.d);
        let elems: Vec<Expr> =
          sorted.into_iter().map(|r| r.to_expr()).collect();
        Ok(Expr::List(elems.into()))
      }
      None => Ok(unevaluated("Exponent", args)),
    }
  } else if use_min {
    match min_power(&expanded, var) {
      Some(r) => Ok(r.to_expr()),
      None => Ok(unevaluated("Exponent", args)),
    }
  } else {
    match max_power(&expanded, var) {
      Some(r) => Ok(r.to_expr()),
      None => {
        // Symbolic fallback: collect per-term exponents, retaining symbolic
        // ones as Expr; then build a Max[...] over all distinct values.
        if let Some(exprs) = collect_term_powers(&expanded, var) {
          return Ok(build_max_expr(exprs));
        }
        Ok(unevaluated("Exponent", args))
      }
    }
  }
}

/// Exponent[expr, form] for a non-symbol `form`: the highest (Max, default),
/// lowest (Min), or every (List) power of `form` treated as an atomic unit
/// across the additive terms of `expr`. The expression is NOT expanded, so a
/// compound form such as `x + 1` keeps its structure
/// (Exponent[(x + 1)^2, x + 1] = 2). A form that never appears contributes 0.
fn exponent_of_form(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let form = &args[1];
  let terms = super::coefficient::collect_additive_terms(&args[0]);
  let mut powers: Vec<Rat> =
    terms.iter().map(|t| term_power_of_form(t, form)).collect();
  if powers.is_empty() {
    powers.push(Rat::zero());
  }

  let use_min =
    args.len() == 3 && matches!(&args[2], Expr::Identifier(s) if s == "Min");
  let use_list =
    args.len() == 3 && matches!(&args[2], Expr::Identifier(s) if s == "List");

  if use_list {
    powers.sort_by(|a, b| (a.n * b.d).cmp(&(b.n * a.d)));
    powers.dedup_by(|a, b| a.n * b.d == b.n * a.d);
    let elems: Vec<Expr> = powers.into_iter().map(|r| r.to_expr()).collect();
    Ok(Expr::List(elems.into()))
  } else if use_min {
    let m = powers.into_iter().reduce(|a, b| a.min(b)).unwrap();
    Ok(m.to_expr())
  } else {
    let m = powers.into_iter().reduce(|a, b| a.max(b)).unwrap();
    Ok(m.to_expr())
  }
}

/// Power to which `form` (an atomic unit) is raised in a single multiplicative
/// term. `form^n` gives `n`, a factor equal to `form` gives 1, a product sums
/// the powers across its factors, and a term without `form` gives 0.
fn term_power_of_form(term: &Expr, form: &Expr) -> Rat {
  use crate::evaluator::pattern_matching::expr_equal;
  if expr_equal(term, form) {
    return Rat::from_int(1);
  }
  match term {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      if expr_equal(left, form)
        && let Some(r) = expr_to_rat(right)
      {
        return r;
      }
      Rat::zero()
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      if expr_equal(&args[0], form)
        && let Some(r) = expr_to_rat(&args[1])
      {
        return r;
      }
      Rat::zero()
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => term_power_of_form(left, form).add(term_power_of_form(right, form)),
    Expr::FunctionCall { name, args } if name == "Times" => args
      .iter()
      .fold(Rat::zero(), |acc, a| acc.add(term_power_of_form(a, form))),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => term_power_of_form(operand, form),
    _ => Rat::zero(),
  }
}

/// Collect the (possibly symbolic) power of `var` in each additive term of
/// `expr`. Returns `None` if any term has an unrecognizable form.
fn collect_term_powers(expr: &Expr, var: &str) -> Option<Vec<Expr>> {
  let terms = super::coefficient::collect_additive_terms(expr);
  let mut result = Vec::with_capacity(terms.len());
  for t in &terms {
    result.push(term_power_of(t, var)?);
  }
  Some(result)
}

/// Power of `var` in a single multiplicative term. Supports symbolic
/// exponents like `x^(1+n)` — returns the exponent expression unchanged
/// when it can't be reduced to a rational.
fn term_power_of(term: &Expr, var: &str) -> Option<Expr> {
  if is_constant_wrt(term, var) {
    return Some(Expr::Integer(0));
  }
  match term {
    Expr::Identifier(n) if n == var => Some(Expr::Integer(1)),
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      if is_constant_wrt(left, var) {
        Some(Expr::Integer(0))
      } else if is_constant_wrt(right, var)
        && matches!(left.as_ref(), Expr::Identifier(n) if n == var)
      {
        Some((**right).clone())
      } else {
        None
      }
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      if is_constant_wrt(&args[0], var) {
        Some(Expr::Integer(0))
      } else if is_constant_wrt(&args[1], var)
        && matches!(&args[0], Expr::Identifier(n) if n == var)
      {
        Some(args[1].clone())
      } else {
        None
      }
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let l = term_power_of(left, var)?;
      let r = term_power_of(right, var)?;
      Some(add_exponents(l, r))
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut acc = Expr::Integer(0);
      for a in args {
        let p = term_power_of(a, var)?;
        acc = add_exponents(acc, p);
      }
      Some(acc)
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => term_power_of(operand, var),
    _ => None,
  }
}

fn add_exponents(a: Expr, b: Expr) -> Expr {
  if matches!(&a, Expr::Integer(0)) {
    return b;
  }
  if matches!(&b, Expr::Integer(0)) {
    return a;
  }
  // Defer to the evaluator so numeric exponents collapse.
  let sum = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![a.clone(), b.clone()].into(),
  };
  crate::evaluator::evaluate_expr_to_expr(&sum).unwrap_or(sum)
}

fn build_max_expr(mut exprs: Vec<Expr>) -> Expr {
  // Drop duplicates (by structural string) and Integer(0) (since any other
  // term is at least 0).
  let mut seen: std::collections::HashSet<String> =
    std::collections::HashSet::new();
  exprs.retain(|e| {
    let key = crate::syntax::expr_to_string(e);
    seen.insert(key)
  });
  // Defer to Max's own evaluator: it will pick the max when all entries are
  // comparable numerically, and stay as `Max[…]` otherwise.
  let call = Expr::FunctionCall {
    name: "Max".to_string(),
    args: exprs.into(),
  };
  crate::evaluator::evaluate_expr_to_expr(&call).unwrap_or(call)
}

/// Find the maximum power of `var` in `expr`.  Returns None for non-polynomial forms.
pub fn max_power(expr: &Expr, var: &str) -> Option<Rat> {
  match expr {
    Expr::Integer(_)
    | Expr::BigInteger(_)
    | Expr::Real(_)
    | Expr::Constant(_)
    | Expr::String(_) => Some(Rat::zero()),
    Expr::Identifier(name) => {
      if name == var {
        Some(Rat::from_int(1))
      } else {
        Some(Rat::zero())
      }
    }
    Expr::BinaryOp { op, left, right } => match op {
      BinaryOperator::Plus | BinaryOperator::Minus => {
        let l = max_power(left, var)?;
        let r = max_power(right, var)?;
        Some(l.max(r))
      }
      BinaryOperator::Times => {
        let l = max_power(left, var)?;
        let r = max_power(right, var)?;
        Some(l.add(r))
      }
      BinaryOperator::Power => {
        if is_constant_wrt(left, var) {
          Some(Rat::zero())
        } else if is_constant_wrt(right, var) {
          let exp = expr_to_rat(right)?;
          let base_pow = max_power(left, var)?;
          Some(Rat::new(base_pow.n * exp.n, base_pow.d * exp.d))
        } else {
          None
        }
      }
      BinaryOperator::Divide => {
        if is_constant_wrt(right, var) {
          max_power(left, var)
        } else {
          None
        }
      }
      _ => {
        if is_constant_wrt(expr, var) {
          Some(Rat::zero())
        } else {
          None
        }
      }
    },
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => max_power(operand, var),
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Plus" => {
        let mut m = Rat::zero();
        for a in args {
          m = m.max(max_power(a, var)?);
        }
        Some(m)
      }
      "Times" => {
        let mut s = Rat::zero();
        for a in args {
          s = s.add(max_power(a, var)?);
        }
        Some(s)
      }
      "Power" if args.len() == 2 => {
        if is_constant_wrt(&args[0], var) {
          Some(Rat::zero())
        } else if is_constant_wrt(&args[1], var) {
          let exp = expr_to_rat(&args[1])?;
          let base_pow = max_power(&args[0], var)?;
          Some(Rat::new(base_pow.n * exp.n, base_pow.d * exp.d))
        } else {
          None
        }
      }
      _ => {
        if is_constant_wrt(expr, var) {
          Some(Rat::zero())
        } else {
          None
        }
      }
    },
    _ => {
      if is_constant_wrt(expr, var) {
        Some(Rat::zero())
      } else {
        None
      }
    }
  }
}

/// Collect all distinct powers of `var` appearing as additive terms in `expr`.
fn collect_powers(expr: &Expr, var: &str) -> Option<Vec<Rat>> {
  match expr {
    // A constant term has power 0
    Expr::Integer(_)
    | Expr::BigInteger(_)
    | Expr::Real(_)
    | Expr::Constant(_)
    | Expr::String(_) => Some(vec![Rat::zero()]),
    Expr::Identifier(name) => {
      if name == var {
        Some(vec![Rat::from_int(1)])
      } else {
        Some(vec![Rat::zero()])
      }
    }
    Expr::BinaryOp { op, left, right } => match op {
      BinaryOperator::Plus | BinaryOperator::Minus => {
        let mut l = collect_powers(left, var)?;
        let r = collect_powers(right, var)?;
        l.extend(r);
        Some(l)
      }
      BinaryOperator::Times => {
        // Power of a product = sum of powers of factors
        let l = max_power(left, var)?;
        let r = max_power(right, var)?;
        Some(vec![l.add(r)])
      }
      BinaryOperator::Power => {
        if is_constant_wrt(left, var) {
          Some(vec![Rat::zero()])
        } else if is_constant_wrt(right, var) {
          let exp = expr_to_rat(right)?;
          let base_pow = max_power(left, var)?;
          Some(vec![Rat::new(base_pow.n * exp.n, base_pow.d * exp.d)])
        } else {
          None
        }
      }
      BinaryOperator::Divide => {
        if is_constant_wrt(right, var) {
          collect_powers(left, var)
        } else {
          None
        }
      }
      _ => {
        if is_constant_wrt(expr, var) {
          Some(vec![Rat::zero()])
        } else {
          None
        }
      }
    },
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => collect_powers(operand, var),
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Plus" => {
        let mut all = Vec::new();
        for a in args {
          all.extend(collect_powers(a, var)?);
        }
        Some(all)
      }
      "Times" => {
        let mut s = Rat::zero();
        for a in args {
          s = s.add(max_power(a, var)?);
        }
        Some(vec![s])
      }
      "Power" if args.len() == 2 => {
        if is_constant_wrt(&args[0], var) {
          Some(vec![Rat::zero()])
        } else if is_constant_wrt(&args[1], var) {
          let exp = expr_to_rat(&args[1])?;
          let base_pow = max_power(&args[0], var)?;
          Some(vec![Rat::new(base_pow.n * exp.n, base_pow.d * exp.d)])
        } else {
          None
        }
      }
      _ => {
        if is_constant_wrt(expr, var) {
          Some(vec![Rat::zero()])
        } else {
          None
        }
      }
    },
    _ => {
      if is_constant_wrt(expr, var) {
        Some(vec![Rat::zero()])
      } else {
        None
      }
    }
  }
}

/// Find the minimum power of `var` in `expr`.
pub fn min_power(expr: &Expr, var: &str) -> Option<Rat> {
  match expr {
    Expr::Integer(_)
    | Expr::BigInteger(_)
    | Expr::Real(_)
    | Expr::Constant(_)
    | Expr::String(_) => Some(Rat::zero()),
    Expr::Identifier(name) => {
      if name == var {
        Some(Rat::from_int(1))
      } else {
        Some(Rat::zero())
      }
    }
    Expr::BinaryOp { op, left, right } => match op {
      BinaryOperator::Plus | BinaryOperator::Minus => {
        let l = min_power(left, var)?;
        let r = min_power(right, var)?;
        Some(l.min(r))
      }
      BinaryOperator::Times => {
        let l = min_power(left, var)?;
        let r = min_power(right, var)?;
        Some(l.add(r))
      }
      BinaryOperator::Power => {
        if is_constant_wrt(left, var) {
          Some(Rat::zero())
        } else if is_constant_wrt(right, var) {
          let exp = expr_to_rat(right)?;
          let base_pow = min_power(left, var)?;
          Some(Rat::new(base_pow.n * exp.n, base_pow.d * exp.d))
        } else {
          None
        }
      }
      BinaryOperator::Divide => {
        if is_constant_wrt(right, var) {
          min_power(left, var)
        } else {
          None
        }
      }
      _ => {
        if is_constant_wrt(expr, var) {
          Some(Rat::zero())
        } else {
          None
        }
      }
    },
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => min_power(operand, var),
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Plus" => {
        let mut m: Option<Rat> = None;
        for a in args {
          let p = min_power(a, var)?;
          m = Some(match m {
            None => p,
            Some(prev) => prev.min(p),
          });
        }
        m
      }
      "Times" => {
        let mut s = Rat::zero();
        for a in args {
          s = s.add(min_power(a, var)?);
        }
        Some(s)
      }
      "Power" if args.len() == 2 => {
        if is_constant_wrt(&args[0], var) {
          Some(Rat::zero())
        } else if is_constant_wrt(&args[1], var) {
          let exp = expr_to_rat(&args[1])?;
          let base_pow = min_power(&args[0], var)?;
          Some(Rat::new(base_pow.n * exp.n, base_pow.d * exp.d))
        } else {
          None
        }
      }
      _ => {
        if is_constant_wrt(expr, var) {
          Some(Rat::zero())
        } else {
          None
        }
      }
    },
    _ => {
      if is_constant_wrt(expr, var) {
        Some(Rat::zero())
      } else {
        None
      }
    }
  }
}
