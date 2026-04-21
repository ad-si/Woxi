#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator};

use crate::functions::calculus_ast::is_constant_wrt;

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
    let g = gcd(n.unsigned_abs(), d.unsigned_abs()) as i128;
    let (n2, d2) = (n / g, d / g);
    // Keep denominator positive
    if d2 < 0 {
      Rat { n: -n2, d: -d2 }
    } else {
      Rat { n: n2, d: d2 }
    }
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
        args: vec![Expr::Integer(self.n), Expr::Integer(self.d)],
      }
    }
  }
}

fn gcd(a: u128, b: u128) -> u128 {
  if b == 0 { a } else { gcd(b, a % b) }
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

/// Convenience wrapper: returns min power as integer, or None if non-integer.
pub fn min_power_int(expr: &Expr, var: &str) -> Option<i128> {
  min_power(expr, var).and_then(|r| r.as_int())
}

// ─── Exponent ───────────────────────────────────────────────────────

/// Exponent[expr, var] - Returns the maximum power of var in expr
pub fn exponent_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Exponent expects 2 or 3 arguments".into(),
    ));
  }
  let var = match &args[1] {
    Expr::Identifier(name) => name.as_str(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Second argument of Exponent must be a symbol".into(),
      ));
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
        Ok(Expr::List(elems))
      }
      None => Ok(Expr::FunctionCall {
        name: "Exponent".to_string(),
        args: args.to_vec(),
      }),
    }
  } else if use_min {
    match min_power(&expanded, var) {
      Some(r) => Ok(r.to_expr()),
      None => Ok(Expr::FunctionCall {
        name: "Exponent".to_string(),
        args: args.to_vec(),
      }),
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
        Ok(Expr::FunctionCall {
          name: "Exponent".to_string(),
          args: args.to_vec(),
        })
      }
    }
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
    args: vec![a.clone(), b.clone()],
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
    args: exprs,
  };
  crate::evaluator::evaluate_expr_to_expr(&call).unwrap_or(call)
}

/// Find the maximum power of `var` in `expr`.  Returns None for non-polynomial forms.
pub fn max_power(expr: &Expr, var: &str) -> Option<Rat> {
  match expr {
    Expr::Integer(_) | Expr::Real(_) | Expr::Constant(_) | Expr::String(_) => {
      Some(Rat::zero())
    }
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
    Expr::Integer(_) | Expr::Real(_) | Expr::Constant(_) | Expr::String(_) => {
      Some(vec![Rat::zero()])
    }
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
    Expr::Integer(_) | Expr::Real(_) | Expr::Constant(_) | Expr::String(_) => {
      Some(Rat::zero())
    }
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
