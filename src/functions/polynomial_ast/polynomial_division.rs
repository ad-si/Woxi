#[allow(unused_imports)]
use super::*;
use crate::evaluator::expr_equal;

/// Split `[p, q, x]` or `[p, q, x, Modulus -> n]` into the three positional
/// arguments and an optional modulus.
fn split_modulus(args: &[Expr]) -> (Vec<Expr>, Option<i128>) {
  let mut positional = Vec::new();
  let mut modulus = None;
  for a in args {
    if let Some(m) = extract_modulus_option(a) {
      modulus = Some(m);
    } else {
      positional.push(a.clone());
    }
  }
  (positional, modulus)
}

/// Modular polynomial division `p / q` over GF(p) returning (quotient,
/// remainder) as polynomial expressions in `var`.
fn poly_divmod_mod_exprs(
  p_poly: &Expr,
  q_poly: &Expr,
  var: &str,
  m: i128,
) -> Result<(Expr, Expr), InterpreterError> {
  let pc = poly_to_coeffs_mod(p_poly, var, m)?;
  let qc = poly_to_coeffs_mod(q_poly, var, m)?;
  let (quot, rem) = poly_divmod_mod(&pc, &qc, m);
  Ok((coeffs_to_poly(&quot, var, m), coeffs_to_poly(&rem, var, m)))
}

/// PolynomialRemainder[p, q, x] - remainder of polynomial division
pub fn polynomial_remainder_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let (pos, modulus) = split_modulus(args);
  if pos.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "PolynomialRemainder expects 3 arguments".into(),
    ));
  }
  let var = match &pos[2] {
    Expr::Identifier(name) => name.as_str(),
    _ => {
      return Ok(unevaluated("PolynomialRemainder", args));
    }
  };
  if let Some(m) = modulus {
    return Ok(poly_divmod_mod_exprs(&pos[0], &pos[1], var, m)?.1);
  }

  let (_, remainder) = poly_divide_symbolic(&pos[0], &pos[1], var)?;
  crate::evaluator::evaluate_expr_to_expr(&remainder)
}

/// PolynomialQuotient[p, q, x] - quotient of polynomial division
pub fn polynomial_quotient_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let (pos, modulus) = split_modulus(args);
  if pos.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "PolynomialQuotient expects 3 arguments".into(),
    ));
  }
  let var = match &pos[2] {
    Expr::Identifier(name) => name.as_str(),
    _ => {
      return Ok(unevaluated("PolynomialQuotient", args));
    }
  };
  if let Some(m) = modulus {
    return Ok(poly_divmod_mod_exprs(&pos[0], &pos[1], var, m)?.0);
  }

  let (quotient, _) = poly_divide_symbolic(&pos[0], &pos[1], var)?;
  crate::evaluator::evaluate_expr_to_expr(&quotient)
}

/// PolynomialQuotientRemainder[p, q, x] - {quotient, remainder} of polynomial division
pub fn polynomial_quotient_remainder_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let (pos, modulus) = split_modulus(args);
  if pos.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "PolynomialQuotientRemainder expects 3 arguments".into(),
    ));
  }
  let var = match &pos[2] {
    Expr::Identifier(name) => name.as_str(),
    _ => {
      return Ok(unevaluated("PolynomialQuotientRemainder", args));
    }
  };
  if let Some(m) = modulus {
    let (q, r) = poly_divmod_mod_exprs(&pos[0], &pos[1], var, m)?;
    return Ok(Expr::List(vec![q, r].into()));
  }

  let (quotient, remainder) = poly_divide_symbolic(&pos[0], &pos[1], var)?;
  let q = crate::evaluator::evaluate_expr_to_expr(&quotient)?;
  let r = crate::evaluator::evaluate_expr_to_expr(&remainder)?;
  Ok(Expr::List(vec![q, r].into()))
}

/// PolynomialReduce[poly, {p1, p2, ...}, x] — reduce `poly` modulo the
/// polynomials `pi` in the single variable `x`. Returns `{{a1, a2, ...}, b}`
/// where `a1 p1 + a2 p2 + ... + b == poly` and `b` is the (minimal) remainder.
///
/// A variable list `{x1, x2, …}` of two or more entries instead uses
/// lexicographic multivariate division; a single variable given that way
/// falls through to the single-variable path below.
pub fn polynomial_reduce_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("PolynomialReduce", args);
  if args.len() != 3 {
    return Ok(unevaluated());
  }
  // A single divisor may be given without the surrounding list, the way the
  // other polynomial-division functions take one; the result still reports
  // its quotient inside a one-element list:
  // PolynomialReduce[x^2, x, x] → {{x}, 0}.
  let divisors: Vec<Expr> = match &args[1] {
    Expr::List(items) => items.to_vec(),
    other => vec![other.clone()],
  };
  // Multivariate form: {x1, x2, …} with two or more variables uses
  // lexicographic multivariate division. A "variable" need not be a bare
  // identifier — `p[a]`, say, is just as valid an indeterminate to real
  // Mathematica as `x`. The exact-rational engine handles the overwhelming
  // majority of real input (and is what every existing multivariate test
  // exercises) far faster than evaluator-driven symbolic arithmetic, so it
  // is tried first; only a coefficient that isn't a plain rational number —
  // a free parameter, `Sqrt[3]`, `Pi`, … — falls through to the slower,
  // fully general path.
  if let Expr::List(vs) = &args[2]
    && vs.len() >= 2
  {
    if let Some(fast) =
      crate::functions::groebner_ast::polynomial_reduce_multivar(
        &args[0], &divisors, vs,
      )
    {
      return Ok(fast);
    }
    return Ok(
      polynomial_reduce_multivar_symbolic(&args[0], &divisors, vs)
        .unwrap_or_else(&unevaluated),
    );
  }
  let var = match &args[2] {
    Expr::Identifier(v) => v.clone(),
    Expr::List(vs) if vs.len() == 1 => match &vs[0] {
      Expr::Identifier(v) => v.clone(),
      _ => return Ok(unevaluated()),
    },
    _ => return Ok(unevaluated()),
  };

  let eval = |e: Expr| crate::evaluator::evaluate_expr_to_expr(&e);
  let var_id = Expr::Identifier(var.clone());
  let expand = |e: &Expr| -> Expr {
    eval(call1("Expand", e.clone())).unwrap_or_else(|_| e.clone())
  };
  let exponent = |e: &Expr| -> Option<i128> {
    match eval(call("Exponent", vec![e.clone(), var_id.clone()])) {
      Ok(Expr::Integer(n)) => Some(n),
      _ => None,
    }
  };
  let coeff = |e: &Expr, d: i128| -> Expr {
    eval(call(
      "Coefficient",
      vec![e.clone(), var_id.clone(), Expr::Integer(d)],
    ))
    .unwrap_or(Expr::Integer(0))
  };
  let is_zero = |e: &Expr| matches!(e, Expr::Integer(0));
  let var_pow = |d: i128| -> Expr {
    if d == 0 {
      Expr::Integer(1)
    } else if d == 1 {
      var_id.clone()
    } else {
      call("Power", vec![var_id.clone(), Expr::Integer(d)])
    }
  };

  // Precompute divisor leading data: (degree, leading_coeff, expanded_divisor).
  let mut div_info: Vec<Option<(i128, Expr, Expr)>> = Vec::new();
  for d in &divisors {
    let de = expand(d);
    if is_zero(&de) {
      div_info.push(None);
      continue;
    }
    match exponent(&de) {
      Some(deg) => {
        let lc = coeff(&de, deg);
        div_info.push(Some((deg, lc, de)));
      }
      None => return Ok(unevaluated()), // not a polynomial in `var`
    }
  }

  let k = divisors.len();
  let mut quotients = vec![Expr::Integer(0); k];
  let mut remainder = Expr::Integer(0);
  let mut p = expand(&args[0]);

  let mut guard = 0usize;
  while !is_zero(&p) {
    guard += 1;
    if guard > 100_000 {
      return Ok(unevaluated());
    }
    let Some(dp) = exponent(&p) else {
      return Ok(unevaluated());
    };
    let lcp = coeff(&p, dp);

    // Find the first divisor whose leading term divides the leading term of p.
    let mut reduced = false;
    for (i, info) in div_info.iter().enumerate() {
      let Some((ddeg, dlc, de)) = info else {
        continue;
      };
      if *ddeg <= dp {
        // t = (lcp / dlc) * x^(dp - ddeg)
        let ratio = build_div(&lcp, dlc);
        let t = eval(build_mul(&ratio, &var_pow(dp - ddeg)))
          .unwrap_or_else(|_| build_mul(&ratio, &var_pow(dp - ddeg)));
        quotients[i] =
          eval(call("Plus", vec![quotients[i].clone(), t.clone()]))
            .unwrap_or_else(|_| quotients[i].clone());
        p = expand(&build_sub(&p, &build_mul(&t, de)));
        reduced = true;
        break;
      }
    }
    if !reduced {
      // Move the leading term of p into the remainder.
      let lt = eval(build_mul(&lcp, &var_pow(dp)))
        .unwrap_or_else(|_| build_mul(&lcp, &var_pow(dp)));
      remainder = eval(call("Plus", vec![remainder.clone(), lt.clone()]))
        .unwrap_or_else(|_| remainder.clone());
      p = expand(&build_sub(&p, &lt));
    }
  }

  // A divisor written with a rational denominator — `(1 + x + x^2 - x^3)/2`
  // — is reduced against its cleared numerator, and the clearing factor is
  // left standing in front of the quotient rather than distributed into it:
  // wolframscript answers `{{2*(1 - x)}, 0}`, not `{{2 - 2*x}, 0}`. Only
  // `PolynomialReduce` presents it that way; `PolynomialQuotient` expands.
  for (i, divisor) in divisors.iter().enumerate() {
    if is_zero(&quotients[i]) {
      continue;
    }
    let Ok(Expr::Integer(m)) =
      eval(call1("Denominator", call1("Together", divisor.clone())))
    else {
      continue;
    };
    if m <= 1 {
      continue;
    }
    let scaled = expand(&build_div(&quotients[i], &Expr::Integer(m)));
    quotients[i] = build_mul(&Expr::Integer(m), &scaled);
  }

  Ok(Expr::List(
    vec![Expr::List(quotients.into()), remainder].into(),
  ))
}

/// One monomial's exponent vector, one per entry of `vars` (lexicographic
/// order: `vars[0]` is most significant, matching wolframscript's
/// `PolynomialReduce`/`MonomialOrder -> "Lexicographic"` default).
type MExp = Vec<u32>;

/// `term`'s exponent in each of `vars`, plus whatever is left over as
/// `(factors to Times together)` — the term's coefficient. A "variable"
/// need not be a bare identifier (`p[a]` counts, matched structurally), and
/// the coefficient may be any expression at all: a free parameter (`a`), an
/// irrational constant (`Sqrt[3]`), `Pi`, … — anything not itself one of
/// `vars` just rides along symbolically, exactly as real Mathematica treats
/// it. Returns `None` only when `term` raises one of the `vars` to a
/// negative or non-integer power, which is not a polynomial in that
/// variable.
fn term_to_mexp_symbolic(
  term: &Expr,
  vars: &[Expr],
) -> Option<(MExp, Vec<Expr>)> {
  fn flatten<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    match e {
      Expr::FunctionCall { name, args } if name == "Times" => {
        for a in args {
          flatten(a, out);
        }
      }
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        flatten(left, out);
        flatten(right, out);
      }
      other => out.push(other),
    }
  }
  let mut factors: Vec<&Expr> = Vec::new();
  flatten(term, &mut factors);
  let mut mono = vec![0u32; vars.len()];
  let mut coef_factors: Vec<Expr> = Vec::new();
  for f in factors {
    if let Some(i) = vars.iter().position(|w| expr_equal(f, w)) {
      mono[i] += 1;
      continue;
    }
    let power_parts = match f {
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        Some((&args[0], &args[1]))
      }
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => Some((&**left, &**right)),
      _ => None,
    };
    if let Some((base, exp)) = power_parts
      && let Some(i) = vars.iter().position(|w| expr_equal(base, w))
    {
      match exp {
        Expr::Integer(k) if *k >= 1 && *k <= u32::MAX as i128 => {
          mono[i] += *k as u32;
          continue;
        }
        // A negative or non-integer power of an actual variable: not a
        // polynomial in that variable, so the whole reduction bails.
        _ => return None,
      }
    }
    coef_factors.push(f.clone());
  }
  Some((mono, coef_factors))
}

/// Merge `(mono, coeff)` into `poly`, summing coefficients on a repeated
/// monomial (symbolically, through the evaluator) and dropping any entry
/// that cancels to exactly `0`.
fn mexp_add_term(
  poly: &mut Vec<(MExp, Expr)>,
  mono: MExp,
  coeff: Expr,
) -> Option<()> {
  if matches!(coeff, Expr::Integer(0)) {
    return Some(());
  }
  if let Some(pos) = poly.iter().position(|(m, _)| *m == mono) {
    let summed = crate::evaluator::evaluate_expr_to_expr(&plus2(
      poly[pos].1.clone(),
      coeff,
    ))
    .ok()?;
    if matches!(summed, Expr::Integer(0)) {
      poly.remove(pos);
    } else {
      poly[pos].1 = summed;
    }
  } else {
    poly.push((mono, coeff));
  }
  Some(())
}

/// Expanded expression -> sparse multivariate polynomial with symbolic
/// coefficients, as `(exponents, coefficient)` pairs.
fn expr_to_mexp_terms(expr: &Expr, vars: &[Expr]) -> Option<Vec<(MExp, Expr)>> {
  fn split<'a>(e: &'a Expr, sign: i128, out: &mut Vec<(&'a Expr, i128)>) {
    match e {
      Expr::FunctionCall { name, args } if name == "Plus" => {
        for a in args {
          split(a, sign, out);
        }
      }
      Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left,
        right,
      } => {
        split(left, sign, out);
        split(right, sign, out);
      }
      Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left,
        right,
      } => {
        split(left, sign, out);
        split(right, -sign, out);
      }
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand,
      } => split(operand, -sign, out),
      other => out.push((other, sign)),
    }
  }
  let mut raw: Vec<(&Expr, i128)> = Vec::new();
  split(expr, 1, &mut raw);
  let mut poly: Vec<(MExp, Expr)> = Vec::new();
  for (term, sign) in raw {
    let (mono, mut coef_factors) = term_to_mexp_symbolic(term, vars)?;
    if sign < 0 {
      coef_factors.push(Expr::Integer(-1));
    }
    let coeff = match coef_factors.len() {
      0 => Expr::Integer(1),
      1 => coef_factors.remove(0),
      _ => {
        crate::evaluator::evaluate_expr_to_expr(&call("Times", coef_factors))
          .ok()?
      }
    };
    mexp_add_term(&mut poly, mono, coeff)?;
  }
  Some(poly)
}

/// Render `(exponents, coefficient)` terms back to an `Expr` sum.
fn mexp_terms_to_expr(poly: &[(MExp, Expr)], vars: &[Expr]) -> Expr {
  if poly.is_empty() {
    return Expr::Integer(0);
  }
  let mut terms: Vec<Expr> = Vec::new();
  for (mono, coeff) in poly {
    let mut factors: Vec<Expr> = Vec::new();
    if !matches!(coeff, Expr::Integer(1)) || mono.iter().all(|&e| e == 0) {
      factors.push(coeff.clone());
    }
    for (j, &e) in mono.iter().enumerate() {
      if e == 1 {
        factors.push(vars[j].clone());
      } else if e > 1 {
        factors.push(call(
          "Power",
          vec![vars[j].clone(), Expr::Integer(e as i128)],
        ));
      }
    }
    terms.push(match factors.len() {
      1 => factors.remove(0),
      _ => call("Times", factors),
    });
  }
  let sum = match terms.len() {
    1 => terms.remove(0),
    _ => call("Plus", terms),
  };
  crate::evaluator::evaluate_expr_to_expr(&sum).unwrap_or(sum)
}

/// PolynomialReduce[poly, {g1, …, gk}, {x1, …, xn}] — multivariate division
/// in lexicographic order, with symbolic (not just rational) coefficients:
/// any part of a term that is not one of `vars` — a free parameter, `Pi`,
/// `Sqrt[3]`, … — rides along as an ordinary coefficient, added, multiplied
/// and divided through the evaluator exactly as the single-variable
/// division above does. Returns `None` (so the caller stays unevaluated)
/// for non-polynomial input (in the given `vars`) or a runaway reduction.
fn polynomial_reduce_multivar_symbolic(
  dividend: &Expr,
  divisors: &[Expr],
  vars: &[Expr],
) -> Option<Expr> {
  if vars.is_empty() || vars.len() > 6 {
    return None;
  }
  let eval = |e: &Expr| {
    crate::evaluator::evaluate_expr_to_expr(e).unwrap_or_else(|_| e.clone())
  };
  let expand = |e: &Expr| -> Expr { eval(&call1("Expand", e.clone())) };

  let mut p = expr_to_mexp_terms(&expand(dividend), vars)?;
  let mut div_terms: Vec<Vec<(MExp, Expr)>> =
    Vec::with_capacity(divisors.len());
  for d in divisors {
    div_terms.push(expr_to_mexp_terms(&expand(d), vars)?);
  }
  for dt in &mut div_terms {
    dt.sort_by(|a, b| b.0.cmp(&a.0));
  }
  let k = div_terms.len();
  let mut quotients: Vec<Vec<(MExp, Expr)>> = vec![Vec::new(); k];
  let mut remainder: Vec<(MExp, Expr)> = Vec::new();

  let mut guard = 0usize;
  loop {
    p.sort_by(|a, b| b.0.cmp(&a.0));
    let Some((lm_p, lc_p)) = p.first().cloned() else {
      break;
    };
    guard += 1;
    if guard > 100_000 {
      return None;
    }
    let mut reduced = false;
    for i in 0..k {
      let Some((lm_i, lc_i)) = div_terms[i].first().cloned() else {
        continue;
      };
      if lm_i.iter().zip(lm_p.iter()).all(|(di, pi)| di <= pi) {
        let factor = eval(&build_div(&lc_p, &lc_i));
        let shift: MExp =
          lm_p.iter().zip(lm_i.iter()).map(|(a, b)| a - b).collect();
        // `lm_i` (divisor `i`'s own leading monomial) never changes, and
        // `p`'s leading monomial strictly decreases every iteration, so
        // `shift` does too whenever this same divisor is chosen again —
        // this quotient can never receive the same monomial twice.
        quotients[i].push((shift.clone(), factor.clone()));
        // The divisor's own leading term exactly cancels `p`'s leading
        // term by construction (that is how `factor`/`shift` were chosen)
        // — drop it directly rather than spending two more evaluator
        // round trips proving a cancellation already known to happen, and
        // only walk the divisor's remaining terms.
        p.remove(0);
        for (dm, dc) in &div_terms[i][1..] {
          let sm: MExp =
            dm.iter().zip(shift.iter()).map(|(a, b)| a + b).collect();
          let sc = eval(&call(
            "Times",
            vec![Expr::Integer(-1), factor.clone(), dc.clone()],
          ));
          mexp_add_term(&mut p, sm, sc)?;
        }
        reduced = true;
        break;
      }
    }
    if !reduced {
      // `p`'s leading monomial strictly decreases every iteration (whether
      // reduced or moved to the remainder), so it can never recur — no
      // divisor's leading term divides it, and no future subtraction can
      // ever reintroduce a monomial this large. Moving it is therefore a
      // plain append, with no merge-search or cancelling evaluator call
      // needed on either side.
      remainder.push((lm_p.clone(), lc_p));
      p.remove(0);
    }
  }

  let q_exprs: Vec<Expr> = quotients
    .iter()
    .map(|q| mexp_terms_to_expr(q, vars))
    .collect();
  let r_expr = mexp_terms_to_expr(&remainder, vars);
  Some(Expr::List(vec![Expr::List(q_exprs.into()), r_expr].into()))
}

/// Perform polynomial long division p / q in variable var.
/// Returns (quotient, remainder) as expressions.
pub fn poly_divide_symbolic(
  p: &Expr,
  q: &Expr,
  var: &str,
) -> Result<(Expr, Expr), InterpreterError> {
  // Get coefficients of both polynomials
  let p_expanded = expand_and_combine(p);
  let q_expanded = expand_and_combine(q);

  let p_deg = max_power_int(&p_expanded, var).unwrap_or(0);
  let q_deg = max_power_int(&q_expanded, var).unwrap_or(0);

  if q_deg == 0 {
    // Dividing by a constant - check if it's zero
    let q_coeff = coefficient_ast(&[
      q.clone(),
      Expr::Identifier(var.to_string()),
      Expr::Integer(0),
    ])?;
    let q_str =
      expr_to_string(&crate::evaluator::evaluate_expr_to_expr(&q_coeff)?);
    if q_str == "0" {
      return Err(InterpreterError::EvaluationError(
        "PolynomialRemainder: division by zero polynomial".into(),
      ));
    }
  }

  // Extract coefficients for p
  let mut p_coeffs: Vec<Expr> = Vec::new();
  for i in 0..=p_deg {
    let c = coefficient_ast(&[
      p.clone(),
      Expr::Identifier(var.to_string()),
      Expr::Integer(i),
    ])?;
    p_coeffs.push(crate::evaluator::evaluate_expr_to_expr(&c)?);
  }

  // Extract coefficients for q
  let mut q_coeffs: Vec<Expr> = Vec::new();
  for i in 0..=q_deg {
    let c = coefficient_ast(&[
      q.clone(),
      Expr::Identifier(var.to_string()),
      Expr::Integer(i),
    ])?;
    q_coeffs.push(crate::evaluator::evaluate_expr_to_expr(&c)?);
  }

  if p_deg < q_deg {
    // Remainder is p itself, quotient is 0
    return Ok((Expr::Integer(0), p_expanded));
  }

  // Polynomial long division using Expr arithmetic
  let mut remainder = p_coeffs;
  let mut quotient_coeffs =
    vec![Expr::Integer(0); (p_deg - q_deg + 1) as usize];
  let lead_q = q_coeffs.last().unwrap().clone();

  for i in (0..quotient_coeffs.len()).rev() {
    let rem_idx = i + q_coeffs.len() - 1;
    if rem_idx >= remainder.len() {
      continue;
    }

    // q_i = remainder[rem_idx] / lead_q
    let qi = build_div(&remainder[rem_idx], &lead_q);
    let qi = crate::evaluator::evaluate_expr_to_expr(&qi)?;

    quotient_coeffs[i] = qi.clone();

    // Subtract qi * q from remainder
    for j in 0..q_coeffs.len() {
      let sub = build_mul(&qi, &q_coeffs[j]);
      let sub = crate::evaluator::evaluate_expr_to_expr(&sub)?;
      let new_val = build_sub(&remainder[i + j], &sub);
      remainder[i + j] = crate::evaluator::evaluate_expr_to_expr(&new_val)?;
    }
  }

  // Build quotient expression
  let quotient = coeffs_to_expr_symbolic(&quotient_coeffs, var);
  let rem = coeffs_to_expr_symbolic(&remainder, var);

  Ok((quotient, rem))
}

/// Build a division expression
fn build_div(a: &Expr, b: &Expr) -> Expr {
  if expr_to_string(b) == "1" {
    return a.clone();
  }
  div2(a.clone(), b.clone())
}

/// Build a multiplication expression
pub fn build_mul(a: &Expr, b: &Expr) -> Expr {
  call("Times", vec![a.clone(), b.clone()])
}

/// Build a subtraction expression
pub fn build_sub(a: &Expr, b: &Expr) -> Expr {
  call(
    "Plus",
    vec![a.clone(), call("Times", vec![Expr::Integer(-1), b.clone()])],
  )
}

/// Build polynomial from symbolic coefficients
fn coeffs_to_expr_symbolic(coeffs: &[Expr], var: &str) -> Expr {
  let mut terms = Vec::new();
  for (i, coeff) in coeffs.iter().enumerate() {
    let c_str = expr_to_string(coeff);
    if c_str == "0" {
      continue;
    }
    let term = if i == 0 {
      coeff.clone()
    } else if i == 1 {
      if c_str == "1" {
        Expr::Identifier(var.to_string())
      } else {
        call(
          "Times",
          vec![coeff.clone(), Expr::Identifier(var.to_string())],
        )
      }
    } else {
      let var_power = call(
        "Power",
        vec![Expr::Identifier(var.to_string()), Expr::Integer(i as i128)],
      );
      if c_str == "1" {
        var_power
      } else {
        call("Times", vec![coeff.clone(), var_power])
      }
    };
    terms.push(term);
  }

  if terms.is_empty() {
    Expr::Integer(0)
  } else if terms.len() == 1 {
    terms.into_iter().next().unwrap()
  } else {
    call("Plus", terms)
  }
}
