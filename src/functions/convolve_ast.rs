//! Convolve[f, g, x, y] — symbolic convolution over the real line for a
//! recognized family of integrands:
//!
//! - DiracDelta[x] ⊛ g  →  g with x → y
//! - UnitBox[x] ⊛ UnitBox[x]  →  UnitTriangle[y]
//! - UnitStep[x] ⊛ UnitStep[x]  →  y*UnitStep[y]
//! - K·E^(-a(x-p)²) ⊛ K'·E^(-a'(x-p')²)
//!   →  K K' Sqrt[Pi/(a + a')] · E^(-(a a'/(a + a')) (y-p-p')²)
//!   Recognized via `Log`/`PowerExpand`/`Expand`, so any algebraic packaging
//!   of a (possibly shifted, possibly scaled) Gaussian matches — a bare
//!   `E^(-a x^2)`, `PDF[NormalDistribution[mu, sigma], x]`, etc.
//! - E^(-a x) UnitStep[x] ⊛ E^(-a x) UnitStep[x]  →  y E^(-a y) UnitStep[y]
//!
//! Results are constructed to match wolframscript's printed forms
//! (e.g. Sqrt[Pi/2]/E^(y^2/2), (y*UnitStep[y])/E^(2*y)).

#[allow(unused_imports)]
use super::*;
use crate::functions::math_ast::rat_reduce;

type Frac = (i128, i128);

fn frac(n: i128, d: i128) -> Frac {
  rat_reduce(n, d)
}

fn frac_to_expr(f: Frac) -> Expr {
  if f.1 == 1 {
    Expr::Integer(f.0)
  } else {
    call("Rational", vec![Expr::Integer(f.0), Expr::Integer(f.1)])
  }
}

pub fn convolve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| unevaluated("Convolve", args);
  if args.len() != 4 {
    return Ok(unevaluated(args));
  }
  let x_var = match &args[2] {
    Expr::Identifier(v) => v.clone(),
    _ => return Ok(unevaluated(args)),
  };
  let y = match &args[3] {
    y @ Expr::Identifier(_) => y.clone(),
    _ => return Ok(unevaluated(args)),
  };

  let is_call_on_x = |e: &Expr, head: &str| {
    matches!(e, Expr::FunctionCall { name, args }
      if name == head
        && args.len() == 1
        && matches!(&args[0], Expr::Identifier(v) if *v == x_var))
  };

  // DiracDelta[x] ⊛ g → g(y) (convolution is commutative; try both sides)
  for (f, g) in [(&args[0], &args[1]), (&args[1], &args[0])] {
    if is_call_on_x(f, "DiracDelta") {
      let substituted = crate::syntax::substitute_variable(g, &x_var, &y);
      return crate::evaluator::evaluate_expr_to_expr(&substituted);
    }
  }

  // UnitBox[x] ⊛ UnitBox[x] → UnitTriangle[y]
  if is_call_on_x(&args[0], "UnitBox") && is_call_on_x(&args[1], "UnitBox") {
    return Ok(call1("UnitTriangle", y));
  }

  let unit_step_y = || call1("UnitStep", y.clone());
  let times = |factors: Vec<Expr>| call("Times", factors);
  let e_sym = || id_expr("E");

  // UnitStep[x] ⊛ UnitStep[x] → y*UnitStep[y]
  if is_call_on_x(&args[0], "UnitStep") && is_call_on_x(&args[1], "UnitStep") {
    return Ok(times(vec![y.clone(), unit_step_y()]));
  }

  // K·E^(-a(x-p)²) ⊛ K'·E^(-a'(x-p')²) → see `gaussian_shape` doc comment.
  if let (Some((a, shift_f, k_f)), Some((b, shift_g, k_g))) = (
    gaussian_shape(&args[0], &x_var),
    gaussian_shape(&args[1], &x_var),
  ) {
    let s = frac(a.0 * b.1 + b.0 * a.1, a.1 * b.1); // a + b
    let q = frac(a.0 * b.0 * s.1, a.1 * b.1 * s.0); // a*b/(a + b)
    let sqrt_part = call1("Sqrt", div2(const_expr("Pi"), frac_to_expr(s)));
    let y_shifted = minus2(y.clone(), call("Plus", vec![shift_f, shift_g]));
    let y_sq = pow2(y_shifted, Expr::Integer(2));
    let exponent = match q {
      (1, 1) => y_sq,
      (1, r) => div2(y_sq, Expr::Integer(r)),
      (p, 1) => times(vec![Expr::Integer(p), y_sq]),
      (p, r) => div2(times(vec![Expr::Integer(p), y_sq]), Expr::Integer(r)),
    };
    let gaussian_part = div2(sqrt_part, pow2(e_sym(), exponent));
    let result = times(vec![k_f, k_g, gaussian_part]);
    return crate::evaluator::evaluate_expr_to_expr(&result);
  }

  // E^(-a x) UnitStep[x] ⊛ (same a) → (y*UnitStep[y])/E^(a y)
  if let (Some(a), Some(b)) = (
    exp_step_rate(&args[0], &x_var),
    exp_step_rate(&args[1], &x_var),
  ) && a == b
  {
    let rate_term = match a {
      (1, 1) => y.clone(),
      (p, 1) => times(vec![Expr::Integer(p), y.clone()]),
      (p, r) => {
        div2(times(vec![Expr::Integer(p), y.clone()]), Expr::Integer(r))
      }
    };
    return Ok(div2(
      times(vec![y.clone(), unit_step_y()]),
      pow2(e_sym(), rate_term),
    ));
  }

  Ok(unevaluated(args))
}

/// Recognize `expr` (a function of `x_var`) as `K·E^(-a·(x_var-shift)²)` for
/// a literal positive rational `a`, returning `(a, shift, K)`.
///
/// Works by expanding `Log[expr]` into a polynomial in `x_var` (via
/// `PowerExpand` then `Expand`) and reading off its quadratic, linear and
/// constant coefficients: `Log[K·E^(-a(x-p)²)] = -a·x² + 2ap·x + (Log[K] -
/// a·p²)`. This is agnostic to how the Gaussian is algebraically packaged
/// (a bare `E^(-a x^2)`, a reciprocal product such as
/// `PDF[NormalDistribution[mu, sigma], x]`, shifted or not), unlike matching
/// the `Power[E, …]` shape directly.
fn gaussian_shape(expr: &Expr, x_var: &str) -> Option<(Frac, Expr, Expr)> {
  let log_expr = call1("PowerExpand", call1("Log", expr.clone()));
  let expanded =
    crate::evaluator::evaluate_expr_to_expr(&call1("Expand", log_expr)).ok()?;

  let mut coeff2_terms = Vec::new();
  let mut coeff1_terms = Vec::new();
  let mut coeff0_terms = Vec::new();
  for term in collect_additive_terms(&expanded) {
    let (power, coeff) = term_var_power_and_coeff(&term, x_var);
    match power {
      2 => coeff2_terms.push(coeff),
      1 => coeff1_terms.push(coeff),
      0 => coeff0_terms.push(coeff),
      _ => return None,
    }
  }
  if coeff2_terms.is_empty() {
    return None;
  }
  let coeff2 =
    crate::evaluator::evaluate_expr_to_expr(&call("Plus", coeff2_terms))
      .ok()?;
  let a = negative_frac(&coeff2)?;

  let coeff1 = if coeff1_terms.is_empty() {
    Expr::Integer(0)
  } else {
    call("Plus", coeff1_terms)
  };
  let coeff0 = if coeff0_terms.is_empty() {
    Expr::Integer(0)
  } else {
    call("Plus", coeff0_terms)
  };

  // shift = coeff1 / (2a)
  let two_a = frac_to_expr(frac(2 * a.0, a.1));
  let shift =
    crate::evaluator::evaluate_expr_to_expr(&div2(coeff1, two_a)).ok()?;

  // K = E^(coeff0 + a·shift²) — the x-independent remainder of Log[expr],
  // with the shift's own quadratic contribution (-a·shift²) added back in.
  let correction =
    times2(frac_to_expr(a), pow2(shift.clone(), Expr::Integer(2)));
  let log_k = call("Plus", vec![coeff0, correction]);
  let k =
    crate::evaluator::evaluate_expr_to_expr(&pow2(id_expr("E"), log_k)).ok()?;

  Some((a, shift, k))
}

/// Match E^(-a·x)·UnitStep[x] (factors in any order) and return a.
fn exp_step_rate(expr: &Expr, x_var: &str) -> Option<Frac> {
  let factors: Vec<&Expr> = match expr {
    Expr::FunctionCall { name, args } if name == "Times" && args.len() == 2 => {
      args.iter().collect()
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => vec![left, right],
    _ => return None,
  };
  let mut rate: Option<Frac> = None;
  let mut has_step = false;
  for f in factors {
    if matches!(f, Expr::FunctionCall { name, args }
      if name == "UnitStep"
        && args.len() == 1
        && matches!(&args[0], Expr::Identifier(v) if v == x_var))
    {
      has_step = true;
    } else {
      rate = neg_coeff_of(&exp_exponent(f)?, x_var, 1);
    }
  }
  if has_step { rate } else { None }
}

/// If expr is E^exponent, return the exponent.
fn exp_exponent(expr: &Expr) -> Option<Expr> {
  match expr {
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      is_e(&args[0]).then(|| args[1].clone())
    }
    Expr::FunctionCall { name, args } if name == "Exp" && args.len() == 1 => {
      Some(args[0].clone())
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => is_e(left).then(|| (**right).clone()),
    _ => None,
  }
}

fn is_e(expr: &Expr) -> bool {
  matches!(expr, Expr::Identifier(n) | Expr::Constant(n) if n == "E")
}

/// Match exponent == -c·x^deg with positive rational c.
fn neg_coeff_of(exponent: &Expr, x_var: &str, deg: i128) -> Option<Frac> {
  let x_pow_matches = |e: &Expr| -> bool {
    if deg == 1 {
      matches!(e, Expr::Identifier(v) if v == x_var)
    } else {
      match e {
        Expr::FunctionCall { name, args }
          if name == "Power" && args.len() == 2 =>
        {
          matches!(&args[0], Expr::Identifier(v) if v == x_var)
            && matches!(&args[1], Expr::Integer(d) if *d == deg)
        }
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left,
          right,
        } => {
          matches!(left.as_ref(), Expr::Identifier(v) if v == x_var)
            && matches!(right.as_ref(), Expr::Integer(d) if *d == deg)
        }
        _ => false,
      }
    }
  };
  match exponent {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      if x_pow_matches(operand) {
        return Some((1, 1));
      }
      // -(c·x^deg) with positive c
      if let Expr::FunctionCall { name, args } = operand.as_ref()
        && name == "Times"
        && args.len() == 2
        && x_pow_matches(&args[1])
      {
        return positive_frac(&args[0]);
      }
      None
    }
    Expr::FunctionCall { name, args } if name == "Times" && args.len() == 2 => {
      if !x_pow_matches(&args[1]) {
        return None;
      }
      negative_frac(&args[0])
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      if !x_pow_matches(right) {
        return None;
      }
      negative_frac(left)
    }
    _ => None,
  }
}

fn positive_frac(e: &Expr) -> Option<Frac> {
  match e {
    Expr::Integer(n) if *n > 0 => Some((*n, 1)),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1])
        && *n > 0
        && *d > 0
      {
        Some(frac(*n, *d))
      } else {
        None
      }
    }
    _ => None,
  }
}

fn negative_frac(e: &Expr) -> Option<Frac> {
  match e {
    Expr::Integer(n) if *n < 0 => Some((-n, 1)),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1])
        && *n < 0
        && *d > 0
      {
        Some(frac(-n, *d))
      } else {
        None
      }
    }
    _ => None,
  }
}
