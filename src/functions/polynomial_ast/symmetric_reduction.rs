//! `SymmetricReduction[f, {x1, ..., xn}]` and the three-argument
//! `SymmetricReduction[f, {x1, ..., xn}, {s1, ..., sn}]`.
//!
//! Writes the symmetric part of `f` as a polynomial in the elementary symmetric
//! polynomials `e_k`, returning `{p, q}` where `p` is that symmetric part and
//! `q` is the non-symmetric remainder (`f == p + q`). With the two-argument
//! form the `e_k` are substituted back as their variable expressions; with the
//! three-argument form they are written in terms of the supplied symbols
//! `s_k`. Uses the classical reduction by elementary symmetric polynomials in
//! lexicographic monomial order.

use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator};

fn eval(e: Expr) -> Result<Expr, InterpreterError> {
  crate::evaluator::evaluate_expr_to_expr(&e)
}

fn call(name: &str, args: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args: args.into(),
  }
}

fn sum(terms: Vec<Expr>) -> Expr {
  match terms.len() {
    0 => Expr::Integer(0),
    1 => terms.into_iter().next().unwrap(),
    _ => Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms.into(),
    },
  }
}

fn pow(base: Expr, e: i128) -> Expr {
  if e == 1 {
    return base;
  }
  Expr::BinaryOp {
    op: BinaryOperator::Power,
    left: Box::new(base),
    right: Box::new(Expr::Integer(e)),
  }
}

/// Extract `(exponent vector, coefficient)` pairs from
/// `CoefficientRules[poly, vars]`, or None if it isn't a clean polynomial.
fn coeff_rules(
  poly: &Expr,
  vars: &[Expr],
) -> Result<Option<Vec<(Vec<i128>, Expr)>>, InterpreterError> {
  let rules = eval(call(
    "CoefficientRules",
    vec![poly.clone(), Expr::List(vars.to_vec().into())],
  ))?;
  let Expr::List(ref items) = rules else {
    return Ok(None);
  };
  let mut out = Vec::new();
  for it in items.iter() {
    let Expr::Rule {
      pattern,
      replacement,
    } = it
    else {
      return Ok(None);
    };
    let Expr::List(exps) = pattern.as_ref() else {
      return Ok(None);
    };
    let mut ev = Vec::with_capacity(exps.len());
    for e in exps.iter() {
      match e {
        Expr::Integer(n) => ev.push(*n),
        _ => return Ok(None),
      }
    }
    if ev.len() != vars.len() {
      return Ok(None);
    }
    out.push((ev, replacement.as_ref().clone()));
  }
  Ok(Some(out))
}

/// True for explicitly negative numeric values (integer, rational, real) in
/// any of their expression representations.
fn negative_numeric(e: &Expr) -> bool {
  match e {
    Expr::Integer(n) => *n < 0,
    Expr::Real(x) => *x < 0.0,
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      matches!(
        (&args[0], &args[1]),
        (Expr::Integer(a), Expr::Integer(b)) if a.signum() * b.signum() < 0
      )
    }
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      matches!(
        (&**left, &**right),
        (Expr::Integer(a), Expr::Integer(b)) if a.signum() * b.signum() < 0
      )
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      matches!(
        &**operand,
        Expr::Integer(n) if *n > 0
      ) || matches!(
        &**operand,
        Expr::Real(x) if *x > 0.0
      )
    }
    _ => false,
  }
}

/// `PowerSymmetricPolynomial[r, {x1, ..., xn}]` — the power sum
/// `x1^r + ... + xn^r` (threading listably over nested data), and
/// `PowerSymmetricPolynomial[{r1, ..., rk}, {{...}, ...}]` — the multivariate
/// power sum over k-tuples. The one-argument formal form, any spec with an
/// explicitly negative numeric exponent, and tuple data whose rows do not
/// match the spec length all stay unevaluated, matching wolframscript.
pub fn power_symmetric_polynomial_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "PowerSymmetricPolynomial".to_string(),
      args: args.to_vec().into(),
    })
  };
  if args.len() != 2 {
    // The one-argument form is a formal object.
    return unevaluated();
  }
  let rspec = eval(args[0].clone())?;
  let data = match eval(args[1].clone())? {
    Expr::List(ref items) => items.to_vec(),
    _ => return unevaluated(),
  };

  let power = |base: &Expr, exp: &Expr| Expr::BinaryOp {
    op: BinaryOperator::Power,
    left: Box::new(base.clone()),
    right: Box::new(exp.clone()),
  };

  match &rspec {
    Expr::List(rs) => {
      if rs.iter().any(negative_numeric) {
        return unevaluated();
      }
      let mut terms = Vec::with_capacity(data.len());
      for row in &data {
        let Expr::List(cells) = row else {
          return unevaluated();
        };
        if cells.len() != rs.len() {
          return unevaluated();
        }
        let factors: Vec<Expr> = cells
          .iter()
          .zip(rs.iter())
          .map(|(c, r)| power(c, r))
          .collect();
        terms.push(match factors.len() {
          0 => Expr::Integer(1),
          1 => factors.into_iter().next().unwrap(),
          _ => Expr::FunctionCall {
            name: "Times".to_string(),
            args: factors.into(),
          },
        });
      }
      eval(sum(terms))
    }
    r => {
      if negative_numeric(r) {
        return unevaluated();
      }
      let terms: Vec<Expr> = data.iter().map(|e| power(e, r)).collect();
      eval(sum(terms))
    }
  }
}

pub fn symmetric_reduction_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "SymmetricReduction".to_string(),
      args: args.to_vec().into(),
    })
  };
  if args.len() != 2 && args.len() != 3 {
    return unevaluated();
  }
  let f = &args[0];
  let vars: Vec<Expr> = match &args[1] {
    Expr::List(v) => v.to_vec(),
    _ => return unevaluated(),
  };
  let n = vars.len();
  if n == 0 {
    return unevaluated();
  }

  // Elementary symmetric polynomials e_1, ..., e_n in the variables (used both
  // for the subtraction step and, in the 2-arg form, as the symbolic names).
  let mut elem: Vec<Expr> = Vec::with_capacity(n);
  for k in 1..=n {
    elem.push(eval(call(
      "SymmetricPolynomial",
      vec![Expr::Integer(k as i128), Expr::List(vars.clone().into())],
    ))?);
  }
  // Symbols the symmetric part is written in.
  let sym: Vec<Expr> = if args.len() == 3 {
    match &args[2] {
      Expr::List(s) if s.len() == n => s.to_vec(),
      _ => return unevaluated(),
    }
  } else {
    elem.clone()
  };

  let mut g = eval(call("Expand", vec![f.clone()]))?;
  let mut p_terms: Vec<Expr> = Vec::new();
  let mut q_terms: Vec<Expr> = Vec::new();

  for _ in 0..100_000 {
    if matches!(g, Expr::Integer(0)) {
      break;
    }
    let Some(mut rules) = coeff_rules(&g, &vars)? else {
      return unevaluated();
    };
    if rules.is_empty() {
      break;
    }
    // Leading term in lexicographic order (x1 > x2 > ... > xn).
    rules.sort_by(|a, b| b.0.cmp(&a.0));
    let (exp, coeff) = rules.into_iter().next().unwrap();

    let weakly_decreasing =
      exp.windows(2).all(|w| w[0] >= w[1]) && *exp.last().unwrap() >= 0;
    if weakly_decreasing {
      // Match with e_1^d1 e_2^d2 ... e_n^dn where d_k = a_k - a_{k+1}
      // (and d_n = a_n). Subtract its expansion from g, record the s-term.
      let mut d = vec![0i128; n];
      for k in 0..n - 1 {
        d[k] = exp[k] - exp[k + 1];
      }
      d[n - 1] = exp[n - 1];

      let mut p_factors = vec![coeff.clone()];
      let mut e_factors = vec![coeff.clone()];
      for (k, &dk) in d.iter().enumerate() {
        if dk > 0 {
          p_factors.push(pow(sym[k].clone(), dk));
          e_factors.push(pow(elem[k].clone(), dk));
        }
      }
      p_terms.push(eval(call("Times", p_factors))?);
      let e_prod = call("Times", e_factors);
      g = eval(call(
        "Expand",
        vec![sum(vec![
          g.clone(),
          eval(call("Times", vec![Expr::Integer(-1), e_prod]))?,
        ])],
      ))?;
    } else {
      // The leading monomial cannot belong to the symmetric part: move it to
      // the remainder and continue with the rest of g.
      let mut m_factors = vec![coeff.clone()];
      for (i, &ai) in exp.iter().enumerate() {
        if ai > 0 {
          m_factors.push(pow(vars[i].clone(), ai));
        }
      }
      let mono = eval(call("Times", m_factors))?;
      q_terms.push(mono.clone());
      g = eval(call(
        "Expand",
        vec![sum(vec![
          g.clone(),
          eval(call("Times", vec![Expr::Integer(-1), mono]))?,
        ])],
      ))?;
    }
  }

  let p = eval(sum(p_terms))?;
  let q = eval(sum(q_terms))?;
  Ok(Expr::List(vec![p, q].into()))
}
