//! Convolve[f, g, x, y] — symbolic convolution over the real line for a
//! recognized family of integrands:
//!
//! - DiracDelta[x] ⊛ g  →  g with x → y
//! - UnitBox[x] ⊛ UnitBox[x]  →  UnitTriangle[y]
//! - UnitStep[x] ⊛ UnitStep[x]  →  y*UnitStep[y]
//! - E^(-a x²) ⊛ E^(-b x²)  →  Sqrt[Pi/(a + b)] · E^(-(a b/(a + b)) y²)
//! - E^(-a x) UnitStep[x] ⊛ E^(-a x) UnitStep[x]  →  y E^(-a y) UnitStep[y]
//!
//! Results are constructed to match wolframscript's printed forms
//! (e.g. Sqrt[Pi/2]/E^(y^2/2), (y*UnitStep[y])/E^(2*y)).

use crate::InterpreterError;
use crate::syntax::Expr;

type Frac = (i128, i128);

fn frac(n: i128, d: i128) -> Frac {
  fn gcd(a: i128, b: i128) -> i128 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
      let t = b;
      b = a % b;
      a = t;
    }
    a
  }
  let g = gcd(n, d).max(1);
  let (mut n, mut d) = (n / g, d / g);
  if d < 0 {
    n = -n;
    d = -d;
  }
  (n, d)
}

fn frac_to_expr(f: Frac) -> Expr {
  if f.1 == 1 {
    Expr::Integer(f.0)
  } else {
    Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(f.0), Expr::Integer(f.1)].into(),
    }
  }
}

pub fn convolve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "Convolve".to_string(),
    args: args.to_vec().into(),
  };
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
    return Ok(Expr::FunctionCall {
      name: "UnitTriangle".to_string(),
      args: vec![y].into(),
    });
  }

  let unit_step_y = || Expr::FunctionCall {
    name: "UnitStep".to_string(),
    args: vec![y.clone()].into(),
  };
  let times = |factors: Vec<Expr>| Expr::FunctionCall {
    name: "Times".to_string(),
    args: factors.into(),
  };
  let div = |n: Expr, d: Expr| Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left: Box::new(n),
    right: Box::new(d),
  };
  let pow = |b: Expr, e: Expr| Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Power,
    left: Box::new(b),
    right: Box::new(e),
  };
  let e_sym = || Expr::Identifier("E".to_string());

  // UnitStep[x] ⊛ UnitStep[x] → y*UnitStep[y]
  if is_call_on_x(&args[0], "UnitStep") && is_call_on_x(&args[1], "UnitStep") {
    return Ok(times(vec![y.clone(), unit_step_y()]));
  }

  // E^(-a x²) ⊛ E^(-b x²) → Sqrt[Pi/(a + b)]/E^((a b/(a + b)) y²)
  if let (Some(a), Some(b)) = (
    gaussian_rate(&args[0], &x_var),
    gaussian_rate(&args[1], &x_var),
  ) {
    let s = frac(a.0 * b.1 + b.0 * a.1, a.1 * b.1); // a + b
    let q = frac(a.0 * b.0 * s.1, a.1 * b.1 * s.0); // a*b/(a + b)
    let sqrt_part = Expr::FunctionCall {
      name: "Sqrt".to_string(),
      args: vec![div(Expr::Constant("Pi".to_string()), frac_to_expr(s))].into(),
    };
    let y_sq = pow(y.clone(), Expr::Integer(2));
    let exponent = match q {
      (1, 1) => y_sq,
      (1, r) => div(y_sq, Expr::Integer(r)),
      (p, 1) => times(vec![Expr::Integer(p), y_sq]),
      (p, r) => div(times(vec![Expr::Integer(p), y_sq]), Expr::Integer(r)),
    };
    return Ok(div(sqrt_part, pow(e_sym(), exponent)));
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
      (p, r) => div(times(vec![Expr::Integer(p), y.clone()]), Expr::Integer(r)),
    };
    return Ok(div(
      times(vec![y.clone(), unit_step_y()]),
      pow(e_sym(), rate_term),
    ));
  }

  Ok(unevaluated(args))
}

/// Match E^(-a·x²) and return the positive rational rate a.
fn gaussian_rate(expr: &Expr, x_var: &str) -> Option<Frac> {
  let exponent = exp_exponent(expr)?;
  neg_coeff_of(&exponent, x_var, 2)
}

/// Match E^(-a·x)·UnitStep[x] (factors in any order) and return a.
fn exp_step_rate(expr: &Expr, x_var: &str) -> Option<Frac> {
  let factors: Vec<&Expr> = match expr {
    Expr::FunctionCall { name, args } if name == "Times" && args.len() == 2 => {
      args.iter().collect()
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
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
    } else if let Some(exponent) = exp_exponent(f) {
      rate = neg_coeff_of(&exponent, x_var, 1);
    } else {
      return None;
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
      op: crate::syntax::BinaryOperator::Power,
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
          op: crate::syntax::BinaryOperator::Power,
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
      op: crate::syntax::UnaryOperator::Minus,
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
      op: crate::syntax::BinaryOperator::Times,
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
