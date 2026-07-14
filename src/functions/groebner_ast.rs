//! GroebnerBasis[{polys}, {vars}] — reduced Gröbner basis in
//! lexicographic order over the rationals, normalized the way
//! wolframscript prints it: primitive integer coefficients, positive
//! leading coefficient, basis sorted ascending by leading monomial.
//!
//! Exact i128 fraction arithmetic with checked operations; coefficient
//! blowup or unrecognized (non-polynomial) input returns the call
//! unevaluated.

use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator};
use std::collections::BTreeMap;

type Frac = (i128, i128);
type Mono = Vec<u32>;
type Poly = BTreeMap<std::cmp::Reverse<Mono>, Frac>; // leading term first

// With `Modulus -> p` (prime) the same engine runs over GF(p): every
// coefficient is kept as (v, 1) with v in [0, p), which `norm` — the
// funnel all fraction arithmetic passes through — enforces via modular
// inversion of denominators.
thread_local! {
  static GB_MODULUS: std::cell::Cell<Option<i128>> =
    const { std::cell::Cell::new(None) };
}

/// Set the thread-local modulus for the lifetime of the returned guard.
struct ModulusGuard;
impl Drop for ModulusGuard {
  fn drop(&mut self) {
    GB_MODULUS.with(|m| m.set(None));
  }
}
fn set_modulus(p: i128) -> ModulusGuard {
  GB_MODULUS.with(|m| m.set(Some(p)));
  ModulusGuard
}

fn mod_inverse(a: i128, p: i128) -> Option<i128> {
  let (mut old_r, mut r) = (a.rem_euclid(p), p);
  let (mut old_s, mut s) = (1i128, 0i128);
  if old_r == 0 {
    return None;
  }
  while r != 0 {
    let q = old_r / r;
    (old_r, r) = (r, old_r - q * r);
    (old_s, s) = (s, old_s - q * s);
  }
  if old_r != 1 {
    return None;
  }
  Some(old_s.rem_euclid(p))
}

fn gcd(a: i128, b: i128) -> i128 {
  let (mut a, mut b) = (a.abs(), b.abs());
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a
}

fn norm(f: Frac) -> Option<Frac> {
  if f.1 == 0 {
    return None;
  }
  if let Some(p) = GB_MODULUS.with(|m| m.get()) {
    let inv = mod_inverse(f.1, p)?;
    return Some(((f.0.rem_euclid(p) * inv).rem_euclid(p), 1));
  }
  let g = gcd(f.0, f.1).max(1);
  let (mut n, mut d) = (f.0 / g, f.1 / g);
  if d < 0 {
    n = n.checked_neg()?;
    d = d.checked_neg()?;
  }
  Some((n, d))
}

fn f_add(a: Frac, b: Frac) -> Option<Frac> {
  norm((
    a.0.checked_mul(b.1)?.checked_add(b.0.checked_mul(a.1)?)?,
    a.1.checked_mul(b.1)?,
  ))
}

fn f_mul(a: Frac, b: Frac) -> Option<Frac> {
  norm((a.0.checked_mul(b.0)?, a.1.checked_mul(b.1)?))
}

fn f_div(a: Frac, b: Frac) -> Option<Frac> {
  if b.0 == 0 {
    return None;
  }
  norm((a.0.checked_mul(b.1)?, a.1.checked_mul(b.0)?))
}

fn f_neg(a: Frac) -> Option<Frac> {
  norm((a.0.checked_neg()?, a.1))
}

fn poly_add_term(p: &mut Poly, m: Mono, c: Frac) -> Option<()> {
  let key = std::cmp::Reverse(m);
  let existing = p.get(&key).copied().unwrap_or((0, 1));
  let sum = f_add(existing, c)?;
  if sum.0 == 0 {
    p.remove(&key);
  } else {
    p.insert(key, sum);
  }
  Some(())
}

fn leading(p: &Poly) -> Option<(&Mono, Frac)> {
  p.iter().next().map(|(k, &c)| (&k.0, c))
}

fn mono_divides(a: &Mono, b: &Mono) -> bool {
  a.iter().zip(b.iter()).all(|(x, y)| x <= y)
}

fn mono_div(b: &Mono, a: &Mono) -> Mono {
  b.iter().zip(a.iter()).map(|(x, y)| x - y).collect()
}

fn mono_mul(a: &Mono, b: &Mono) -> Mono {
  a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn mono_lcm(a: &Mono, b: &Mono) -> Mono {
  a.iter().zip(b.iter()).map(|(x, y)| *x.max(y)).collect()
}

/// p - c * m * q
fn poly_sub_scaled(p: &Poly, q: &Poly, c: Frac, m: &Mono) -> Option<Poly> {
  let mut out = p.clone();
  for (key, &qc) in q {
    let term_m = mono_mul(&key.0, m);
    let term_c = f_neg(f_mul(qc, c)?)?;
    poly_add_term(&mut out, term_m, term_c)?;
  }
  Some(out)
}

/// Fully reduce p modulo the basis (all terms, not just the leading one).
fn reduce(p: &Poly, basis: &[Poly]) -> Option<Poly> {
  let mut current = p.clone();
  'outer: loop {
    for (key, &c) in current.iter() {
      for b in basis {
        if let Some((lm, lc)) = leading(b)
          && mono_divides(lm, &key.0)
        {
          let factor = f_div(c, lc)?;
          let shift = mono_div(&key.0, lm);
          current = poly_sub_scaled(&current, b, factor, &shift)?;
          continue 'outer;
        }
      }
    }
    return Some(current);
  }
}

/// Render a `Poly` (lex order, leading term first) back to an `Expr`, then
/// evaluate it so the result is canonicalized like wolframscript.
fn poly_to_expr(p: &Poly, vars: &[String]) -> Expr {
  if p.is_empty() {
    return Expr::Integer(0);
  }
  let mut terms: Vec<Expr> = Vec::new();
  for (key, &(n, d)) in p.iter() {
    let mono = &key.0;
    let mut factors: Vec<Expr> = Vec::new();
    for (j, &e) in mono.iter().enumerate() {
      if e == 1 {
        factors.push(Expr::Identifier(vars[j].clone()));
      } else if e > 1 {
        factors.push(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(Expr::Identifier(vars[j].clone())),
          right: Box::new(Expr::Integer(e as i128)),
        });
      }
    }
    let coeff = if d == 1 {
      Expr::Integer(n)
    } else {
      Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(n), Expr::Integer(d)].into(),
      }
    };
    let term = if factors.is_empty() {
      coeff
    } else {
      let mono_expr = factors
        .into_iter()
        .reduce(|a, b| Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(a),
          right: Box::new(b),
        })
        .unwrap();
      if n == 1 && d == 1 {
        mono_expr
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(coeff),
          right: Box::new(mono_expr),
        }
      }
    };
    terms.push(term);
  }
  let sum = terms
    .into_iter()
    .reduce(|a, b| Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(a),
      right: Box::new(b),
    })
    .unwrap();
  crate::evaluator::evaluate_expr_to_expr(&sum).unwrap_or(sum)
}

/// PolynomialReduce[poly, {g1, …, gk}, {x1, …, xn}] — multivariate division in
/// lexicographic order. Returns `{{q1, …, qk}, r}` such that
/// `poly = q1 g1 + … + qk gk + r`. Returns None (so the caller stays
/// unevaluated) for non-polynomial input or coefficients that aren't rational
/// in the given variables (e.g. a divisor free of all the variables).
pub fn polynomial_reduce_multivar(
  dividend: &Expr,
  divisors: &[Expr],
  vars: &[String],
) -> Option<Expr> {
  if vars.is_empty() || vars.len() > 6 {
    return None;
  }
  let expand = |e: &Expr| -> Expr {
    crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Expand".to_string(),
      args: vec![e.clone()].into(),
    })
    .unwrap_or_else(|_| e.clone())
  };

  let mut p = expr_to_poly(&expand(dividend), vars)?;
  let mut div_polys: Vec<Poly> = Vec::with_capacity(divisors.len());
  for d in divisors {
    div_polys.push(expr_to_poly(&expand(d), vars)?);
  }
  let k = div_polys.len();
  let mut quotients: Vec<Poly> = vec![Poly::new(); k];
  let mut remainder = Poly::new();

  let mut guard = 0usize;
  while let Some((lm_p, lc_p)) = leading(&p).map(|(m, c)| (m.clone(), c)) {
    guard += 1;
    if guard > 100_000 {
      return None;
    }
    let mut reduced = false;
    for i in 0..k {
      let Some((lm_i, lc_i)) =
        leading(&div_polys[i]).map(|(m, c)| (m.clone(), c))
      else {
        continue;
      };
      if mono_divides(&lm_i, &lm_p) {
        let factor = f_div(lc_p, lc_i)?;
        let shift = mono_div(&lm_p, &lm_i);
        poly_add_term(&mut quotients[i], shift.clone(), factor)?;
        p = poly_sub_scaled(&p, &div_polys[i], factor, &shift)?;
        reduced = true;
        break;
      }
    }
    if !reduced {
      // The leading term is divisible by no divisor: move it to the remainder.
      poly_add_term(&mut remainder, lm_p.clone(), lc_p)?;
      poly_add_term(&mut p, lm_p, f_neg(lc_p)?)?;
    }
  }

  let q_exprs: Vec<Expr> =
    quotients.iter().map(|q| poly_to_expr(q, vars)).collect();
  let r_expr = poly_to_expr(&remainder, vars);
  Some(Expr::List(vec![Expr::List(q_exprs.into()), r_expr].into()))
}

fn s_poly(a: &Poly, b: &Poly) -> Option<Poly> {
  let (lma, lca) = leading(a)?;
  let (lmb, lcb) = leading(b)?;
  let l = mono_lcm(lma, lmb);
  // a * (l/lma)/lca - b * (l/lmb)/lcb
  let mut left = Poly::new();
  let sa = mono_div(&l, lma);
  let ca = f_div((1, 1), lca)?;
  for (key, &c) in a {
    poly_add_term(&mut left, mono_mul(&key.0, &sa), f_mul(c, ca)?)?;
  }
  let sb = mono_div(&l, lmb);
  let cb = f_div((1, 1), lcb)?;
  poly_sub_scaled(&left, b, cb, &sb)
}

/// Extract `Modulus -> m` (m >= 0) from an option argument.
fn groebner_modulus_option(opt: &Expr) -> Option<i128> {
  let value = |v: &Expr| match v {
    Expr::Integer(m) if *m >= 0 => Some(*m),
    _ => None,
  };
  if let Expr::Rule {
    pattern,
    replacement,
  } = opt
    && matches!(pattern.as_ref(), Expr::Identifier(s) if s == "Modulus")
  {
    return value(replacement);
  }
  if let Expr::FunctionCall { name, args } = opt
    && (name == "Rule" || name == "RuleDelayed")
    && args.len() == 2
    && matches!(&args[0], Expr::Identifier(s) if s == "Modulus")
  {
    return value(&args[1]);
  }
  None
}

fn is_prime_i128(n: i128) -> bool {
  if n < 2 {
    return false;
  }
  let mut d = 2i128;
  while d * d <= n {
    if n % d == 0 {
      return false;
    }
    d += 1;
  }
  true
}

fn unevaluated_call(name: &str, args: &[Expr]) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args: args.to_vec().into(),
  }
}

pub fn groebner_basis_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| unevaluated_call("GroebnerBasis", args);
  // GroebnerBasis[polys, vars, Modulus -> p] computes over GF(p);
  // Modulus -> 0 is ordinary arithmetic, and composite moduli (which
  // would need strong Z/m bases) stay unevaluated.
  let mut modulus: Option<i128> = None;
  if args.len() == 3 {
    match groebner_modulus_option(&args[2]) {
      Some(0) => return groebner_basis_ast(&args[..2]),
      Some(p) if is_prime_i128(p) => modulus = Some(p),
      _ => return Ok(unevaluated(args)),
    }
  } else if args.len() != 2 {
    return Ok(unevaluated(args));
  }
  let _guard = modulus.map(set_modulus);
  let full_args = args;
  let args = &args[..2];
  let unevaluated = |_: &[Expr]| unevaluated_call("GroebnerBasis", full_args);
  let polys_in: Vec<Expr> = match &args[0] {
    Expr::List(items) if !items.is_empty() => items.iter().cloned().collect(),
    single => vec![single.clone()],
  };
  let vars: Vec<String> = match &args[1] {
    Expr::List(items) => {
      let mut out = Vec::with_capacity(items.len());
      for v in items {
        match v {
          Expr::Identifier(name) => out.push(name.clone()),
          _ => return Ok(unevaluated(args)),
        }
      }
      out
    }
    Expr::Identifier(name) => vec![name.clone()],
    _ => return Ok(unevaluated(args)),
  };
  if vars.is_empty() || vars.len() > 6 {
    return Ok(unevaluated(args));
  }

  // Parse the input polynomials
  let mut basis: Vec<Poly> = Vec::new();
  for p in &polys_in {
    let expanded =
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Expand".to_string(),
        args: vec![p.clone()].into(),
      })?;
    match expr_to_poly(&expanded, &vars) {
      Some(poly) if !poly.is_empty() => basis.push(poly),
      Some(_) => {} // zero polynomial contributes nothing
      None => return Ok(unevaluated(args)),
    }
  }
  if basis.is_empty() {
    // Over GF(p) inputs that all vanish mod p leave an empty basis;
    // over the rationals wolframscript prints {0}.
    if modulus.is_some() {
      return Ok(Expr::List(vec![].into()));
    }
    return Ok(Expr::List(vec![Expr::Integer(0)].into()));
  }

  // Buchberger's algorithm with a work cap
  let result = (|| -> Option<Vec<Poly>> {
    let mut basis = basis;
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..basis.len() {
      for j in (i + 1)..basis.len() {
        pairs.push((i, j));
      }
    }
    let mut steps = 0usize;
    while let Some((i, j)) = pairs.pop() {
      steps += 1;
      if steps > 2000 {
        return None;
      }
      let s = s_poly(&basis[i], &basis[j])?;
      let r = reduce(&s, &basis)?;
      if !r.is_empty() {
        let new_idx = basis.len();
        for k in 0..new_idx {
          pairs.push((k, new_idx));
        }
        basis.push(r);
      }
    }
    // Inter-reduce: drop polynomials whose leading term is divisible by
    // another's, then fully reduce each against the rest
    let mut keep: Vec<Poly> = Vec::new();
    for i in 0..basis.len() {
      let (lm_i, _) = leading(&basis[i])?;
      let redundant = basis.iter().enumerate().any(|(j, q)| {
        if i == j {
          return false;
        }
        match leading(q) {
          Some((lm_j, _)) => {
            mono_divides(lm_j, lm_i) && (lm_j != lm_i || j < i)
          }
          None => false,
        }
      });
      if !redundant {
        keep.push(basis[i].clone());
      }
    }
    let snapshot = keep.clone();
    let mut reduced: Vec<Poly> = Vec::new();
    for (i, p) in snapshot.iter().enumerate() {
      let others: Vec<Poly> = snapshot
        .iter()
        .enumerate()
        .filter(|(j, _)| *j != i)
        .map(|(_, q)| q.clone())
        .collect();
      let r = reduce(p, &others)?;
      if !r.is_empty() {
        reduced.push(r);
      }
    }
    Some(reduced)
  })();
  let Some(mut reduced) = result else {
    return Ok(unevaluated(args));
  };

  // Trivial ideal: any nonzero constant
  if reduced
    .iter()
    .any(|p| leading(p).is_some_and(|(m, _)| m.iter().all(|&e| e == 0)))
  {
    return Ok(Expr::List(vec![Expr::Integer(1)].into()));
  }

  // Sort ascending by leading monomial (lex)
  reduced.sort_by(|a, b| {
    let la = leading(a).map(|(m, _)| m.clone()).unwrap_or_default();
    let lb = leading(b).map(|(m, _)| m.clone()).unwrap_or_default();
    la.cmp(&lb)
  });

  // Normalize to primitive integer coefficients with a positive leading
  // coefficient, then render. Over GF(p) the basis is made monic first,
  // so the primitive/sign logic below is a no-op (content 1, lead +1).
  let mut out: Vec<Expr> = Vec::with_capacity(reduced.len());
  for p in &reduced {
    let monic;
    let p = if let Some(pm) = modulus {
      let Some((_, lc)) = leading(p) else {
        return Ok(unevaluated(args));
      };
      let Some(inv) = mod_inverse(lc.0, pm) else {
        return Ok(unevaluated(args));
      };
      let mut q = Poly::new();
      for (k, &c) in p {
        q.insert(k.clone(), ((c.0 * inv).rem_euclid(pm), 1));
      }
      monic = q;
      &monic
    } else {
      p
    };
    let rendered = (|| -> Option<Expr> {
      let mut denom_lcm: i128 = 1;
      for &c in p.values() {
        denom_lcm = denom_lcm.checked_mul(c.1 / gcd(denom_lcm, c.1))?;
      }
      let mut nums: Vec<(Mono, i128)> = Vec::new();
      let mut content: i128 = 0;
      for (key, &c) in p {
        let scaled = c.0.checked_mul(denom_lcm / c.1)?;
        content = gcd(content, scaled);
        nums.push((key.0.clone(), scaled));
      }
      let content = content.max(1);
      let lead_sign = nums.first().map(|(_, c)| *c < 0).unwrap_or(false);
      let sign = if lead_sign { -1 } else { 1 };
      let mut terms: Vec<Expr> = Vec::new();
      for (m, c) in &nums {
        let c = sign * c / content;
        let mut factors: Vec<Expr> = Vec::new();
        if c != 1 || m.iter().all(|&e| e == 0) {
          factors.push(Expr::Integer(c));
        }
        for (vi, &e) in m.iter().enumerate() {
          if e == 1 {
            factors.push(Expr::Identifier(vars[vi].clone()));
          } else if e > 1 {
            factors.push(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(Expr::Identifier(vars[vi].clone())),
              right: Box::new(Expr::Integer(e as i128)),
            });
          }
        }
        terms.push(match factors.len() {
          1 => factors.remove(0),
          _ => Expr::FunctionCall {
            name: "Times".to_string(),
            args: factors.into(),
          },
        });
      }
      let sum = if terms.len() == 1 {
        terms.remove(0)
      } else {
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: terms.into(),
        }
      };
      crate::evaluator::evaluate_expr_to_expr(&sum).ok()
    })();
    match rendered {
      Some(e) => out.push(e),
      None => return Ok(unevaluated(args)),
    }
  }
  Ok(Expr::List(out.into()))
}

/// Expanded expression -> sparse multivariate polynomial. None for
/// anything outside c * v1^e1 * ... terms with rational c.
fn expr_to_poly(expr: &Expr, vars: &[String]) -> Option<Poly> {
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
  let mut terms: Vec<(&Expr, i128)> = Vec::new();
  split(expr, 1, &mut terms);
  let mut poly = Poly::new();
  for (term, sign) in terms {
    let (m, c) = term_to_mono(term, vars)?;
    poly_add_term(&mut poly, m, f_mul(c, (sign, 1))?)?;
  }
  Some(poly)
}

fn term_to_mono(term: &Expr, vars: &[String]) -> Option<(Mono, Frac)> {
  let mut mono = vec![0u32; vars.len()];
  let mut coef: Frac = (1, 1);
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
  for f in factors {
    match f {
      Expr::Integer(n) => coef = f_mul(coef, (*n, 1))?,
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
          coef = f_mul(coef, (*n, *d))?;
        } else {
          return None;
        }
      }
      Expr::Identifier(v) => {
        let i = vars.iter().position(|w| w == v)?;
        mono[i] += 1;
      }
      _ => {
        // var^k
        let (base, exp) = match f {
          Expr::FunctionCall { name, args }
            if name == "Power" && args.len() == 2 =>
          {
            (&args[0], &args[1])
          }
          Expr::BinaryOp {
            op: BinaryOperator::Power,
            left,
            right,
          } => (&**left as &Expr, &**right as &Expr),
          _ => return None,
        };
        let (base, exp) = (base, exp);
        if let (Expr::Identifier(v), Expr::Integer(k)) = (base, exp)
          && *k >= 1
          && *k <= u32::MAX as i128
        {
          let i = vars.iter().position(|w| w == v)?;
          mono[i] += *k as u32;
        } else {
          return None;
        }
      }
    }
  }
  Some((mono, coef))
}
