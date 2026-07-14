#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{Expr, expr_to_string, unevaluated};

/// PolynomialExtendedGCD[p, q, x] - extended GCD of two polynomials.
///
/// Returns `{g, {s, t}}` where `g` is the (monic) GCD of `p` and `q` in the
/// variable `x`, and `s`, `t` are polynomials satisfying
/// `s*p + t*q == g`.
pub fn polynomial_extended_gcd_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  // Separate an optional `Modulus -> n` from the positional arguments.
  let mut pos: Vec<Expr> = Vec::new();
  let mut modulus: Option<i128> = None;
  for a in args {
    if let Some(m) = extract_modulus_option(a) {
      modulus = Some(m);
    } else {
      pos.push(a.clone());
    }
  }
  if pos.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "PolynomialExtendedGCD expects 3 arguments".into(),
    ));
  }

  let var = match &pos[2] {
    Expr::Identifier(name) => name.clone(),
    _ => {
      return Ok(unevaluated("PolynomialExtendedGCD", args));
    }
  };

  let p = crate::evaluator::evaluate_expr_to_expr(&pos[0])?;
  let q = crate::evaluator::evaluate_expr_to_expr(&pos[1])?;

  // Modular extended GCD over GF(m).
  if let Some(m) = modulus {
    let pc = poly_to_coeffs_mod(&p, &var, m)?;
    let qc = poly_to_coeffs_mod(&q, &var, m)?;
    let (g, s, t) = poly_extended_gcd_mod(&pc, &qc, m);
    return Ok(Expr::List(
      vec![
        coeffs_to_poly(&g, &var, m),
        Expr::List(
          vec![coeffs_to_poly(&s, &var, m), coeffs_to_poly(&t, &var, m)].into(),
        ),
      ]
      .into(),
    ));
  }

  // Both arguments constant in `var`: over a field they are units, so the GCD
  // is 1 with `s = 1/p`, `t = 0` (matching wolframscript's choice).
  let p_deg = max_power_int(&expand_and_combine(&p), &var).unwrap_or(0);
  let q_deg = max_power_int(&expand_and_combine(&q), &var).unwrap_or(0);
  if p_deg == 0 && q_deg == 0 {
    let p_zero = expr_to_string(&p) == "0";
    let q_zero = expr_to_string(&q) == "0";
    if p_zero && q_zero {
      return Ok(Expr::List(
        vec![
          Expr::Integer(0),
          Expr::List(vec![Expr::Integer(0), Expr::Integer(0)].into()),
        ]
        .into(),
      ));
    }
    // Prefer the first non-zero argument as the inverted coefficient.
    let (s, t) = if !p_zero {
      (
        poly_divide_by_const(&Expr::Integer(1), &p, &var)?,
        Expr::Integer(0),
      )
    } else {
      (
        Expr::Integer(0),
        poly_divide_by_const(&Expr::Integer(1), &q, &var)?,
      )
    };
    return Ok(Expr::List(
      vec![Expr::Integer(1), Expr::List(vec![s, t].into())].into(),
    ));
  }

  // Extended Euclidean algorithm over the field of rational coefficients.
  // Invariant: old_s*p + old_t*q == old_r  and  s*p + t*q == r.
  let mut old_r = expand_and_combine(&p);
  let mut r = expand_and_combine(&q);
  let mut old_s = Expr::Integer(1);
  let mut s = Expr::Integer(0);
  let mut old_t = Expr::Integer(0);
  let mut t = Expr::Integer(1);

  // Guard against runaway loops; well-formed inputs terminate quickly.
  for _ in 0..1000 {
    if expr_to_string(&crate::evaluator::evaluate_expr_to_expr(&r)?) == "0" {
      break;
    }

    let (quot, _rem) = poly_divide_symbolic(&old_r, &r, &var)?;
    let quot = crate::evaluator::evaluate_expr_to_expr(&quot)?;

    // new_r = old_r - quot*r  (use exact polynomial remainder to avoid
    // accumulated rounding when the division is not exact)
    let new_r = poly_sub_mul(&old_r, &quot, &r, &var)?;
    let new_s = poly_sub_mul(&old_s, &quot, &s, &var)?;
    let new_t = poly_sub_mul(&old_t, &quot, &t, &var)?;

    old_r = r;
    r = new_r;
    old_s = s;
    s = new_s;
    old_t = t;
    t = new_t;
  }

  // old_r is now the GCD (up to a constant factor). Normalize so it is monic
  // in `var`: divide g, s, t by the leading coefficient of g.
  let g = crate::evaluator::evaluate_expr_to_expr(&old_r)?;
  let g_deg = max_power_int(&expand_and_combine(&g), &var).unwrap_or(0);
  let lead = coefficient_ast(&[
    g.clone(),
    Expr::Identifier(var.clone()),
    Expr::Integer(g_deg),
  ])?;
  let lead = crate::evaluator::evaluate_expr_to_expr(&lead)?;

  let g_norm = poly_divide_by_const(&g, &lead, &var)?;
  let s_norm = poly_divide_by_const(&old_s, &lead, &var)?;
  let t_norm = poly_divide_by_const(&old_t, &lead, &var)?;

  // wolframscript collapses common rational denominators into a single factor,
  // e.g. `1/2 - x/2` is displayed as `(1 - x)/2`. `Together` reproduces this.
  let g_norm = together(&g_norm)?;
  let s_norm = together(&s_norm)?;
  let t_norm = together(&t_norm)?;

  Ok(Expr::List(
    vec![g_norm, Expr::List(vec![s_norm, t_norm].into())].into(),
  ))
}

/// Apply `Together` to collapse common rational denominators.
fn together(expr: &Expr) -> Result<Expr, InterpreterError> {
  crate::functions::polynomial_ast::together_ast(&[expr.clone()])
}

/// Compute `a - quot*b` and return it expanded/combined.
fn poly_sub_mul(
  a: &Expr,
  quot: &Expr,
  b: &Expr,
  _var: &str,
) -> Result<Expr, InterpreterError> {
  let prod = build_mul(quot, b);
  let expr = build_sub(a, &prod);
  let result = crate::evaluator::evaluate_expr_to_expr(&expr)?;
  crate::evaluator::evaluate_expr_to_expr(&expand_and_combine(&result))
}

/// Divide a polynomial by a (nonzero) constant and expand the result.
fn poly_divide_by_const(
  poly: &Expr,
  c: &Expr,
  _var: &str,
) -> Result<Expr, InterpreterError> {
  if expr_to_string(c) == "1" {
    return crate::evaluator::evaluate_expr_to_expr(&expand_and_combine(poly));
  }
  let inv = Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![c.clone(), Expr::Integer(-1)].into(),
  };
  let prod = build_mul(&inv, poly);
  let result = crate::evaluator::evaluate_expr_to_expr(&prod)?;
  crate::evaluator::evaluate_expr_to_expr(&expand_and_combine(&result))
}
