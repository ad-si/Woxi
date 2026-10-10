//! Multivariate polynomial factorization over the integers.
//!
//! The polynomial is split into its integer content, its monomial content
//! and its content with respect to a main variable, then decomposed into
//! square-free parts (Yun's algorithm over multivariate gcds). Each
//! square-free part is evaluated at an integer point of the remaining
//! variables, the univariate image is factored, and binary splits of the
//! univariate factors are Hensel-lifted back to the full polynomial
//! (Wang-style lifting with the leading coefficient imposed on both
//! factors). A split whose lift reproduces the polynomial exactly is a
//! genuine factorization; when no split lifts, the part is irreducible.
//!
//! Arithmetic is exact over Q with big integers, so there is no overflow.

#[allow(unused_imports)]
use super::*;
use crate::functions::calculus_ast::simplify;
use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::collections::BTreeMap;

// ─── rationals ───────────────────────────────────────────────────────

fn big_gcd(a: &BigInt, b: &BigInt) -> BigInt {
  let (mut a, mut b) = (a.abs(), b.abs());
  while !b.is_zero() {
    let r = &a % &b;
    a = b;
    b = r;
  }
  a
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Q {
  n: BigInt,
  d: BigInt,
}

impl Q {
  fn int(n: BigInt) -> Q {
    Q {
      n,
      d: BigInt::one(),
    }
  }
  fn new(n: BigInt, d: BigInt) -> Q {
    let g = big_gcd(&n, &d);
    let (mut n, mut d) = if g.is_one() || g.is_zero() {
      (n, d)
    } else {
      (n / &g, d / &g)
    };
    if d.is_negative() {
      n = -n;
      d = -d;
    }
    if n.is_zero() {
      d = BigInt::one();
    }
    Q { n, d }
  }
  fn zero() -> Q {
    Q::int(BigInt::zero())
  }
  fn one() -> Q {
    Q::int(BigInt::one())
  }
  fn is_zero(&self) -> bool {
    self.n.is_zero()
  }
  fn add(&self, o: &Q) -> Q {
    if self.d == o.d {
      return Q::new(&self.n + &o.n, self.d.clone());
    }
    Q::new(&self.n * &o.d + &o.n * &self.d, &self.d * &o.d)
  }
  fn neg(&self) -> Q {
    Q {
      n: -&self.n,
      d: self.d.clone(),
    }
  }
  fn sub(&self, o: &Q) -> Q {
    self.add(&o.neg())
  }
  fn mul(&self, o: &Q) -> Q {
    Q::new(&self.n * &o.n, &self.d * &o.d)
  }
  fn div(&self, o: &Q) -> Q {
    Q::new(&self.n * &o.d, &self.d * &o.n)
  }
}

// ─── sparse multivariate polynomials ─────────────────────────────────

type Mono = Vec<u32>;

/// Terms keyed by exponent vector; the lexicographically largest key (the
/// first variable most significant) is the leading term.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Poly {
  nv: usize,
  t: BTreeMap<Mono, Q>,
}

impl Poly {
  fn zero(nv: usize) -> Poly {
    Poly {
      nv,
      t: BTreeMap::new(),
    }
  }
  fn constant(nv: usize, c: Q) -> Poly {
    let mut p = Poly::zero(nv);
    if !c.is_zero() {
      p.t.insert(vec![0; nv], c);
    }
    p
  }
  fn one(nv: usize) -> Poly {
    Poly::constant(nv, Q::one())
  }
  /// `var^k`
  fn var_pow(nv: usize, var: usize, k: u32) -> Poly {
    let mut m = vec![0; nv];
    m[var] = k;
    let mut p = Poly::zero(nv);
    p.t.insert(m, Q::one());
    p
  }
  fn is_zero(&self) -> bool {
    self.t.is_empty()
  }
  fn is_const(&self) -> bool {
    self.t.keys().all(|m| m.iter().all(|&e| e == 0))
  }
  fn add_term(&mut self, m: Mono, c: Q) {
    if c.is_zero() {
      return;
    }
    match self.t.get_mut(&m) {
      Some(v) => {
        *v = v.add(&c);
        if v.is_zero() {
          self.t.remove(&m);
        }
      }
      None => {
        self.t.insert(m, c);
      }
    }
  }
  fn add(&self, o: &Poly) -> Poly {
    let mut r = self.clone();
    for (m, c) in &o.t {
      r.add_term(m.clone(), c.clone());
    }
    r
  }
  fn sub(&self, o: &Poly) -> Poly {
    let mut r = self.clone();
    for (m, c) in &o.t {
      r.add_term(m.clone(), c.neg());
    }
    r
  }
  fn mul(&self, o: &Poly) -> Poly {
    let mut r = Poly::zero(self.nv);
    for (ma, ca) in &self.t {
      for (mb, cb) in &o.t {
        let m: Mono = ma.iter().zip(mb).map(|(a, b)| a + b).collect();
        r.add_term(m, ca.mul(cb));
      }
    }
    r
  }
  fn scale(&self, c: &Q) -> Poly {
    if c.is_zero() {
      return Poly::zero(self.nv);
    }
    Poly {
      nv: self.nv,
      t: self.t.iter().map(|(m, v)| (m.clone(), v.mul(c))).collect(),
    }
  }
  fn deg(&self, var: usize) -> u32 {
    self.t.keys().map(|m| m[var]).max().unwrap_or(0)
  }
  fn vars(&self) -> Vec<usize> {
    (0..self.nv).filter(|&v| self.deg(v) > 0).collect()
  }
  fn leading_coeff(&self) -> Option<&Q> {
    self.t.values().next_back()
  }
  /// Coefficient of `var^k`, as a polynomial with `var` removed.
  fn coeff(&self, var: usize, k: u32) -> Poly {
    let mut r = Poly::zero(self.nv);
    for (m, c) in &self.t {
      if m[var] == k {
        let mut m2 = m.clone();
        m2[var] = 0;
        r.t.insert(m2, c.clone());
      }
    }
    r
  }
  /// Leading coefficient with respect to `var`.
  fn lc(&self, var: usize) -> Poly {
    self.coeff(var, self.deg(var))
  }
  /// Substitute `var = 0`.
  fn at_zero(&self, var: usize) -> Poly {
    self.coeff(var, 0)
  }
  fn derivative(&self, var: usize) -> Poly {
    let mut r = Poly::zero(self.nv);
    for (m, c) in &self.t {
      if m[var] > 0 {
        let mut m2 = m.clone();
        m2[var] -= 1;
        r.add_term(m2, c.mul(&Q::int(BigInt::from(m[var]))));
      }
    }
    r
  }
  /// Substitute `var -> var + a`.
  fn shift(&self, var: usize, a: &BigInt) -> Poly {
    if a.is_zero() || self.deg(var) == 0 {
      return self.clone();
    }
    let lin = Poly::var_pow(self.nv, var, 1)
      .add(&Poly::constant(self.nv, Q::int(a.clone())));
    let mut r = Poly::zero(self.nv);
    for k in (0..=self.deg(var)).rev() {
      r = r.mul(&lin).add(&self.coeff(var, k));
    }
    r
  }
  /// Exact division; None when `b` does not divide `self`.
  fn div_exact(&self, b: &Poly) -> Option<Poly> {
    let (bm, bc) = b.t.iter().next_back()?;
    let mut r = self.clone();
    let mut q = Poly::zero(self.nv);
    while let Some((rm, rc)) = r.t.iter().next_back() {
      if rm.iter().zip(bm).any(|(a, b)| a < b) {
        return None;
      }
      let m: Mono = rm.iter().zip(bm).map(|(a, b)| a - b).collect();
      let c = rc.div(bc);
      let mut t = Poly::zero(self.nv);
      t.t.insert(m.clone(), c.clone());
      r = r.sub(&t.mul(b));
      q.add_term(m, c);
    }
    Some(q)
  }
  /// Split off the rational content: `self = unit * prim` where `prim`
  /// has coprime integer coefficients and a positive leading coefficient.
  fn normalize(&self) -> (Q, Poly) {
    if self.is_zero() {
      return (Q::one(), self.clone());
    }
    let mut num_gcd = BigInt::zero();
    let mut den_lcm = BigInt::one();
    for c in self.t.values() {
      num_gcd = big_gcd(&num_gcd, &c.n);
      let g = big_gcd(&den_lcm, &c.d);
      den_lcm = &den_lcm / g * &c.d;
    }
    let mut unit = Q::new(num_gcd, den_lcm);
    if self.leading_coeff().unwrap().n.is_negative() {
      unit = unit.neg();
    }
    let inv = Q::one().div(&unit);
    (unit, self.scale(&inv))
  }
  fn prim(&self) -> Poly {
    self.normalize().1
  }
}

// ─── gcd, content, square-free decomposition ─────────────────────────

/// Pseudo-remainder of `a` by `b` with respect to `var`.
fn prem(a: &Poly, b: &Poly, var: usize) -> Poly {
  let db = b.deg(var);
  let lb = b.lc(var);
  let mut r = a.clone();
  while !r.is_zero() && r.deg(var) >= db {
    let dr = r.deg(var);
    let t = r.lc(var).mul(&Poly::var_pow(r.nv, var, dr - db));
    r = r.mul(&lb).sub(&t.mul(b)).prim();
  }
  r
}

/// Content of `p` with respect to `var` (gcd of its coefficients).
fn content(p: &Poly, var: usize) -> Poly {
  let mut g = Poly::zero(p.nv);
  for k in 0..=p.deg(var) {
    let c = p.coeff(var, k);
    if c.is_zero() {
      continue;
    }
    g = gcd(&g, &c);
    if g.is_const() {
      return Poly::one(p.nv);
    }
  }
  g
}

/// Greatest common divisor, normalized (integer primitive, positive lead).
fn gcd(a: &Poly, b: &Poly) -> Poly {
  if a.is_zero() {
    return b.prim();
  }
  if b.is_zero() {
    return a.prim();
  }
  if a.is_const() || b.is_const() {
    return Poly::one(a.nv);
  }
  if let Some(g) = gcd_heu(&a.prim(), &b.prim(), 0) {
    return g;
  }
  let var = (0..a.nv).find(|&v| a.deg(v) > 0 || b.deg(v) > 0).unwrap();
  if a.deg(var) == 0 {
    return gcd(a, &content(b, var));
  }
  if b.deg(var) == 0 {
    return gcd(&content(a, var), b);
  }
  let (ca, cb) = (content(a, var), content(b, var));
  let g = gcd(&ca, &cb);
  let mut pa = a.div_exact(&ca).unwrap();
  let mut pb = b.div_exact(&cb).unwrap();
  if pa.deg(var) < pb.deg(var) {
    std::mem::swap(&mut pa, &mut pb);
  }
  loop {
    let r = prem(&pa, &pb, var);
    if r.is_zero() {
      break;
    }
    if r.deg(var) == 0 {
      return g;
    }
    pa = pb;
    pb = r.div_exact(&content(&r, var)).unwrap().prim();
  }
  g.mul(&pb.div_exact(&content(&pb, var)).unwrap()).prim()
}

/// Heuristic gcd (Char–Geddes–Gonnet) of integer-primitive polynomials:
/// evaluate one variable at a large integer, take the gcd of the images
/// recursively, reconstruct the candidate from its balanced ξ-adic digits
/// and accept it only when it divides both inputs. None = inconclusive.
fn gcd_heu(a: &Poly, b: &Poly, depth: usize) -> Option<Poly> {
  let nv = a.nv;
  if a.is_const() && b.is_const() {
    let g = big_gcd(&a.t.values().next()?.n, &b.t.values().next()?.n);
    return Some(Poly::constant(nv, Q::int(g)));
  }
  if depth > 8 {
    return None;
  }
  let var = (0..nv).find(|&v| a.deg(v) > 0 || b.deg(v) > 0)?;
  let max_abs = |p: &Poly| p.t.values().map(|c| c.n.abs()).max().unwrap();
  let deg = a.deg(var).max(b.deg(var)) as u64;
  let mut xi: BigInt = max_abs(a).min(max_abs(b)) * 2 + 29;
  for _ in 0..6 {
    if xi.bits() * deg.max(1) > 4000 {
      return None;
    }
    let eval = |p: &Poly| {
      let mut r = Poly::zero(nv);
      for (m, c) in &p.t {
        let mut m2 = m.clone();
        m2[var] = 0;
        r.add_term(m2, c.mul(&Q::int(xi.pow(m[var]))));
      }
      r
    };
    let (ea, eb) = (eval(a), eval(b));
    if !ea.is_zero() && !eb.is_zero() {
      let (ua, pa) = ea.normalize();
      let (ub, pb) = eb.normalize();
      if let Some(gamma) = gcd_heu(&pa, &pb, depth + 1) {
        // Restore the integer content gcd dropped by the normalization.
        let cg = big_gcd(&ua.n, &ub.n);
        let mut gamma = gamma.scale(&Q::int(cg));
        // Balanced ξ-adic digits of gamma give the coefficients in var.
        let mut cand = Poly::zero(nv);
        let mut k = 0u32;
        while !gamma.is_zero() {
          let mut digit = Poly::zero(nv);
          for (m, c) in &gamma.t {
            let mut r = c.n.clone() % &xi;
            if r < BigInt::zero() {
              r += &xi;
            }
            if &r * 2 > xi {
              r -= &xi;
            }
            digit.add_term(m.clone(), Q::int(r));
          }
          cand = cand.add(&digit.mul(&Poly::var_pow(nv, var, k)));
          gamma = gamma.sub(&digit).scale(&Q::new(BigInt::one(), xi.clone()));
          k += 1;
          if k as u64 > deg + 1 {
            break;
          }
        }
        if gamma.is_zero() && !cand.is_zero() {
          let cand = cand.prim();
          if a.div_exact(&cand).is_some() && b.div_exact(&cand).is_some() {
            return Some(cand);
          }
        }
      }
    }
    xi = xi * 73794 / 27011;
  }
  None
}

/// True when `p` (primitive w.r.t. `var`) is square-free, decided cheaply
/// from a univariate image whose degree in `var` is preserved; None when
/// no conclusive image was found.
fn is_square_free_by_image(p: &Poly, var: usize) -> Option<bool> {
  let others: Vec<usize> = p.vars().into_iter().filter(|&v| v != var).collect();
  let lc = p.lc(var);
  let mut seed = 0x2545f4914f6cdd1du64;
  for attempt in 0..8 {
    let point = eval_point(attempt, others.len(), &mut seed);
    let eval = |q: &Poly| {
      others
        .iter()
        .zip(&point)
        .fold(q.clone(), |acc, (&v, a)| acc.shift(v, a).at_zero(v))
    };
    if eval(&lc).is_zero() {
      continue;
    }
    let image = u_from(&eval(p), var);
    let (s, _) = u_ext_gcd(&image, &u_deriv(&image)).unzip();
    if s.is_some() {
      return Some(true);
    }
  }
  None
}

fn u_deriv(a: &UPoly) -> UPoly {
  u_trim(
    a.iter()
      .enumerate()
      .skip(1)
      .map(|(k, c)| c.mul(&Q::int(BigInt::from(k))))
      .collect(),
  )
}

/// Yun's square-free decomposition of a primitive polynomial with respect
/// to `var`: pairs of (square-free part, multiplicity).
fn square_free(p: &Poly, var: usize) -> Vec<(Poly, usize)> {
  if is_square_free_by_image(p, var) == Some(true) {
    return vec![(p.clone(), 1)];
  }
  let dp = p.derivative(var);
  let g = gcd(p, &dp);
  if g.is_const() {
    return vec![(p.clone(), 1)];
  }
  let mut out = Vec::new();
  let mut b = p.div_exact(&g).unwrap();
  let c = dp.div_exact(&g).unwrap();
  let mut d = c.sub(&b.derivative(var));
  let mut i = 1;
  while !b.is_const() {
    let a = gcd(&b, &d);
    if !a.is_const() {
      out.push((a.clone(), i));
    }
    b = b.div_exact(&a).unwrap();
    let c = d.div_exact(&a).unwrap();
    d = c.sub(&b.derivative(var));
    i += 1;
  }
  out
}

// ─── univariate helpers over Q ───────────────────────────────────────

type UPoly = Vec<Q>; // ascending, trimmed

fn u_trim(mut a: UPoly) -> UPoly {
  while a.last().is_some_and(Q::is_zero) {
    a.pop();
  }
  a
}

fn u_from(p: &Poly, var: usize) -> UPoly {
  let mut v = vec![Q::zero(); p.deg(var) as usize + 1];
  for (m, c) in &p.t {
    v[m[var] as usize] = c.clone();
  }
  u_trim(v)
}

fn u_to(a: &UPoly, nv: usize, var: usize) -> Poly {
  let mut p = Poly::zero(nv);
  for (k, c) in a.iter().enumerate() {
    let mut m = vec![0; nv];
    m[var] = k as u32;
    p.add_term(m, c.clone());
  }
  p
}

fn u_mul(a: &UPoly, b: &UPoly) -> UPoly {
  if a.is_empty() || b.is_empty() {
    return vec![];
  }
  let mut r = vec![Q::zero(); a.len() + b.len() - 1];
  for (i, x) in a.iter().enumerate() {
    for (j, y) in b.iter().enumerate() {
      r[i + j] = r[i + j].add(&x.mul(y));
    }
  }
  u_trim(r)
}

fn u_sub(a: &UPoly, b: &UPoly) -> UPoly {
  let mut r = vec![Q::zero(); a.len().max(b.len())];
  for (i, x) in a.iter().enumerate() {
    r[i] = x.clone();
  }
  for (i, y) in b.iter().enumerate() {
    r[i] = r[i].sub(y);
  }
  u_trim(r)
}

fn u_divrem(a: &UPoly, b: &UPoly) -> (UPoly, UPoly) {
  let mut r = a.clone();
  if r.len() < b.len() {
    return (vec![], r);
  }
  let mut q = vec![Q::zero(); r.len() - b.len() + 1];
  let lb = b.last().unwrap();
  while r.len() >= b.len() && !r.is_empty() {
    let shift = r.len() - b.len();
    let c = r.last().unwrap().div(lb);
    for (i, bi) in b.iter().enumerate() {
      r[i + shift] = r[i + shift].sub(&c.mul(bi));
    }
    q[shift] = c;
    r.pop();
    r = u_trim(r);
  }
  (u_trim(q), r)
}

/// `(s, t)` with `s*a + t*b = 1`, or None when `a` and `b` share a factor.
fn u_ext_gcd(a: &UPoly, b: &UPoly) -> Option<(UPoly, UPoly)> {
  let (mut r0, mut r1) = (a.clone(), b.clone());
  let (mut s0, mut s1) = (vec![Q::one()], vec![]);
  let (mut t0, mut t1) = (vec![], vec![Q::one()]);
  while !r1.is_empty() {
    let (q, r) = u_divrem(&r0, &r1);
    r0 = std::mem::replace(&mut r1, r);
    let s = u_sub(&s0, &u_mul(&q, &s1));
    s0 = std::mem::replace(&mut s1, s);
    let t = u_sub(&t0, &u_mul(&q, &t1));
    t0 = std::mem::replace(&mut t1, t);
  }
  if r0.len() != 1 {
    return None;
  }
  let inv = Q::one().div(&r0[0]);
  let sc = |v: &UPoly| v.iter().map(|c| c.mul(&inv)).collect::<UPoly>();
  Some((sc(&s0), sc(&t0)))
}

// ─── Hensel lifting ──────────────────────────────────────────────────

struct Lifter {
  nv: usize,
  main: usize,
  /// Degree bound for the diophantine corrections in each variable.
  max_deg: u32,
}

impl Lifter {
  /// Solve `s1*a2 + s2*a1 = c` with `deg_main(s1) < deg_main(a1)`, where
  /// every variable in `vars` is expanded about 0.
  fn diophant(
    &self,
    a1: &Poly,
    a2: &Poly,
    c: &Poly,
    vars: &[usize],
  ) -> Option<(Poly, Poly)> {
    let Some((&y, rest)) = vars.split_last() else {
      let ua1 = u_from(a1, self.main);
      let ua2 = u_from(a2, self.main);
      let uc = u_from(c, self.main);
      // s*a2 + t*a1 = 1  ⇒  s1 = (s*c) mod a1,  s2 = (c - s1*a2) / a1.
      let (s, _) = u_ext_gcd(&ua2, &ua1)?;
      let (_, s1) = u_divrem(&u_mul(&s, &uc), &ua1);
      let (s2, rem) = u_divrem(&u_sub(&uc, &u_mul(&s1, &ua2)), &ua1);
      if !rem.is_empty() {
        return None;
      }
      return Some((
        u_to(&s1, self.nv, self.main),
        u_to(&s2, self.nv, self.main),
      ));
    };
    let a1n = a1.at_zero(y);
    let a2n = a2.at_zero(y);
    let (mut s1, mut s2) = self.diophant(&a1n, &a2n, &c.at_zero(y), rest)?;
    let residual = |s1: &Poly, s2: &Poly| c.sub(&s1.mul(a2)).sub(&s2.mul(a1));
    let mut e = residual(&s1, &s2);
    for m in 1..=self.max_deg {
      if e.is_zero() {
        break;
      }
      let cm = e.coeff(y, m);
      if cm.is_zero() {
        continue;
      }
      let (d1, d2) = self.diophant(&a1n, &a2n, &cm, rest)?;
      let ym = Poly::var_pow(self.nv, y, m);
      s1 = s1.add(&d1.mul(&ym));
      s2 = s2.add(&d2.mul(&ym));
      e = residual(&s1, &s2);
    }
    Some((s1, s2))
  }

  /// Lift `a(main, 0) = g0 * h0` to `lc_main(a) * a = G * H`, with both
  /// lifted factors carrying `lc_main(a)` as their leading coefficient.
  /// `others` are the variables to lift, already shifted so the
  /// evaluation point is the origin.
  fn lift(
    &self,
    a: &Poly,
    others: &[usize],
    g0: &UPoly,
    h0: &UPoly,
  ) -> Option<(Poly, Poly)> {
    let l = a.lc(self.main);
    let target = l.mul(a);
    let restrict = |p: &Poly, upto: usize| {
      others[upto..]
        .iter()
        .fold(p.clone(), |acc, &v| acc.at_zero(v))
    };
    // Scale the images so that both carry lc(a) at the origin.
    let l0 = restrict(&l, 0);
    let l0c = l0.t.values().next().cloned().unwrap_or_else(Q::zero);
    let mut g =
      u_to(g0, self.nv, self.main).scale(&l0c.div(g0.last().unwrap()));
    let mut h =
      u_to(h0, self.nv, self.main).scale(&l0c.div(h0.last().unwrap()));
    let (dg, dh) = (g0.len() as u32 - 1, h0.len() as u32 - 1);
    let mut e = Poly::zero(self.nv);
    for j in 0..others.len() {
      let y = others[j];
      let aj = restrict(&target, j + 1);
      let lj = restrict(&l, j + 1);
      // Impose the true leading coefficient on both factors.
      g = g
        .sub(&g.lc(self.main).mul(&Poly::var_pow(self.nv, self.main, dg)))
        .add(&lj.mul(&Poly::var_pow(self.nv, self.main, dg)));
      h = h
        .sub(&h.lc(self.main).mul(&Poly::var_pow(self.nv, self.main, dh)))
        .add(&lj.mul(&Poly::var_pow(self.nv, self.main, dh)));
      let (g_prev, h_prev) = (g.at_zero(y), h.at_zero(y));
      e = aj.sub(&g.mul(&h));
      for m in 1..=aj.deg(y) {
        if e.is_zero() {
          break;
        }
        let c = e.coeff(y, m);
        if c.is_zero() {
          continue;
        }
        let (s1, s2) = self.diophant(&g_prev, &h_prev, &c, &others[..j])?;
        let ym = Poly::var_pow(self.nv, y, m);
        g = g.add(&s1.mul(&ym));
        h = h.add(&s2.mul(&ym));
        e = aj.sub(&g.mul(&h));
      }
      if !e.is_zero() {
        return None;
      }
    }
    if others.is_empty() || !e.is_zero() {
      return None;
    }
    Some((g, h))
  }
}

// ─── factorization driver ────────────────────────────────────────────

/// Upper bound on the number of univariate image factors whose binary
/// splits are tried (2^(r-1) lifts in the worst case).
const MAX_IMAGE_FACTORS: usize = 12;

/// Deterministic small evaluation points: the origin first, then
/// pseudo-random points in a slowly widening box.
fn eval_point(attempt: usize, n: usize, seed: &mut u64) -> Vec<BigInt> {
  if attempt == 0 {
    return vec![BigInt::zero(); n];
  }
  let radius = 1 + (attempt / 4) as i64;
  (0..n)
    .map(|_| {
      *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
      let r = ((*seed >> 33) % (2 * radius as u64 + 1)) as i64 - radius;
      BigInt::from(r)
    })
    .collect()
}

/// Univariate integer polynomial (in `var`) → irreducible primitive
/// factors via the existing univariate factorizer.
fn factor_univariate(p: &Poly, var: usize) -> Option<Vec<UPoly>> {
  let up = u_from(&p.prim(), var);
  let coeffs: Vec<i128> = up
    .iter()
    .map(|c| if c.d.is_one() { c.n.to_i128() } else { None })
    .collect::<Option<_>>()?;
  let factors = super::factor_integer_poly(&coeffs, "x");
  let mut out: Vec<UPoly> = Vec::new();
  for f in &factors {
    let fc = extract_poly_coeffs(f, "x")?;
    if fc.len() > 1 {
      out.push(fc.into_iter().map(|c| Q::int(BigInt::from(c))).collect());
    }
  }
  if out.is_empty() {
    out.push(up);
  }
  // Every image factor must account for the whole degree.
  let total: usize = out.iter().map(|f| f.len() - 1).sum();
  (total == p.deg(var) as usize).then_some(out)
}

/// Factor a square-free polynomial that is primitive with respect to
/// `main` into irreducible factors.
fn factor_square_free(p: &Poly, main: usize) -> Option<Vec<Poly>> {
  if p.deg(main) <= 1 {
    return Some(vec![p.prim()]);
  }
  let others: Vec<usize> =
    p.vars().into_iter().filter(|&v| v != main).collect();
  if others.is_empty() {
    return Some(
      factor_univariate(p, main)?
        .iter()
        .map(|f| u_to(f, p.nv, main).prim())
        .collect(),
    );
  }

  // Find good evaluation points and keep the one with the fewest image
  // factors: lc must not vanish and the image must stay square-free.
  let lc = p.lc(main);
  let mut best: Option<(Vec<BigInt>, Vec<UPoly>)> = None;
  let mut good = 0;
  let mut seed = 0x9e3779b97f4a7c15u64;
  for attempt in 0..60 {
    let point = eval_point(attempt, others.len(), &mut seed);
    let eval = |q: &Poly| {
      others
        .iter()
        .zip(&point)
        .fold(q.clone(), |acc, (&v, a)| acc.shift(v, a).at_zero(v))
    };
    if eval(&lc).is_zero() {
      continue;
    }
    let image = eval(p);
    let uimage = u_from(&image, main);
    if u_ext_gcd(&uimage, &u_deriv(&uimage)).is_none() {
      continue;
    }
    let factors = factor_univariate(&image, main)?;
    if factors.len() == 1 {
      return Some(vec![p.prim()]); // irreducible
    }
    if best.as_ref().is_none_or(|(_, b)| factors.len() < b.len()) {
      best = Some((point, factors));
    }
    good += 1;
    if good >= 3 {
      break;
    }
  }
  let (point, images) = best?;
  if images.len() > MAX_IMAGE_FACTORS {
    return None;
  }
  let shifted = others
    .iter()
    .zip(&point)
    .fold(p.clone(), |acc, (&v, a)| acc.shift(v, a));
  let lifter = Lifter {
    nv: p.nv,
    main,
    max_deg: (0..p.nv).map(|v| p.deg(v)).max().unwrap_or(0),
  };
  let unshift = |q: &Poly| {
    others
      .iter()
      .zip(&point)
      .fold(q.clone(), |acc, (&v, a)| acc.shift(v, &-a))
  };
  split(&shifted, &images, &others, &lifter, &unshift)
}

/// Find a binary split of `p` (shifted to the origin) consistent with the
/// image factors, recursing into both halves. No split ⇒ irreducible.
fn split(
  p: &Poly,
  images: &[UPoly],
  others: &[usize],
  lifter: &Lifter,
  unshift: &dyn Fn(&Poly) -> Poly,
) -> Option<Vec<Poly>> {
  let r = images.len();
  if r <= 1 {
    return Some(vec![unshift(p).prim()]);
  }
  let main = lifter.main;
  for size in 1..=r / 2 {
    for subset in subsets(r, size) {
      // An even split is tried once, from the side holding factor 0.
      if 2 * size == r && subset[0] != 0 {
        continue;
      }
      let rest: Vec<usize> = (0..r).filter(|i| !subset.contains(i)).collect();
      let prod = |idx: &[usize]| {
        idx
          .iter()
          .fold(vec![Q::one()], |acc, &i| u_mul(&acc, &images[i]))
      };
      let Some((g, h)) = lifter.lift(p, others, &prod(&subset), &prod(&rest))
      else {
        continue;
      };
      // Strip the imposed leading coefficient: primitive parts w.r.t. main.
      let g = g.div_exact(&content(&g, main))?.prim();
      let h = h.div_exact(&content(&h, main))?.prim();
      let gi: Vec<UPoly> = subset.iter().map(|&i| images[i].clone()).collect();
      let hi: Vec<UPoly> = rest.iter().map(|&i| images[i].clone()).collect();
      let mut out = split(&g, &gi, others, lifter, unshift)?;
      out.extend(split(&h, &hi, others, lifter, unshift)?);
      return Some(out);
    }
  }
  Some(vec![unshift(p).prim()])
}

/// All size-`k` subsets of `0..n` in lexicographic order.
fn subsets(n: usize, k: usize) -> Vec<Vec<usize>> {
  fn go(
    start: usize,
    n: usize,
    k: usize,
    cur: &mut Vec<usize>,
    out: &mut Vec<Vec<usize>>,
  ) {
    if cur.len() == k {
      out.push(cur.clone());
      return;
    }
    for i in start..n {
      cur.push(i);
      go(i + 1, n, k, cur, out);
      cur.pop();
    }
  }
  let mut out = Vec::new();
  go(0, n, k, &mut Vec::new(), &mut out);
  out
}

/// Full factorization of an integer-primitive polynomial with positive
/// leading coefficient: irreducible factors with multiplicities.
fn factor_primitive(p: &Poly) -> Option<Vec<(Poly, usize)>> {
  if p.is_const() {
    return Some(vec![]);
  }
  let mut out: Vec<(Poly, usize)> = Vec::new();
  // Monomial content.
  let mut p = p.clone();
  for v in 0..p.nv {
    let k = p.t.keys().map(|m| m[v]).min().unwrap_or(0);
    if k > 0 {
      out.push((Poly::var_pow(p.nv, v, 1), k as usize));
      let mut t = BTreeMap::new();
      for (m, c) in &p.t {
        let mut m2 = m.clone();
        m2[v] -= k;
        t.insert(m2, c.clone());
      }
      p.t = t;
    }
  }
  if p.is_const() {
    return Some(out);
  }
  // Main variable: the one of least positive degree.
  let main = p.vars().into_iter().min_by_key(|&v| (p.deg(v), v)).unwrap();
  let c = content(&p, main);
  if !c.is_const() {
    out.extend(factor_primitive(&c)?);
    out.extend(factor_primitive(&p.div_exact(&c)?.prim())?);
    return Some(out);
  }
  for (s, mult) in square_free(&p, main) {
    for f in factor_square_free(&s.prim(), main)? {
      out.push((f, mult));
    }
  }
  Some(out)
}

// ─── Expr bridge ─────────────────────────────────────────────────────

fn coeff_to_q(e: &Expr) -> Option<Q> {
  match e {
    Expr::Integer(n) => Some(Q::int(BigInt::from(*n))),
    Expr::BigInteger(n) => Some(Q::int(n.clone())),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      let n = coeff_to_q(&args[0])?;
      let d = coeff_to_q(&args[1])?;
      (!d.is_zero()).then(|| n.div(&d))
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => coeff_to_q(operand).map(|q| q.neg()),
    _ => None,
  }
}

/// Split a non-numeric factor into (generator, exponent): symbols and
/// positive integer powers map to their base; anything else (`Sin[x]`,
/// `E^t`, …) is an opaque generator, as in wolframscript, where
/// `Factor[2 a x + 2 a Sin[x]]` is `2 a (x + Sin[x])`.
fn generator_power(f: &Expr) -> (Expr, u32) {
  let (base, k) = match f {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => (left.as_ref(), right.as_ref()),
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      (&args[0], &args[1])
    }
    _ => return (f.clone(), 1),
  };
  match k {
    Expr::Integer(k) if *k > 0 => match u32::try_from(*k) {
      Ok(k) => (base.clone(), k),
      Err(_) => (f.clone(), 1),
    },
    _ => (f.clone(), 1),
  }
}

/// Expanded expression → polynomial over its generators, which are
/// returned sorted by their printed form (alphabetical for symbols).
fn expr_to_poly(expr: &Expr) -> Option<(Poly, Vec<Expr>)> {
  let mut terms: Vec<(Q, Vec<(String, Expr, u32)>)> = Vec::new();
  let mut gens: BTreeMap<String, Expr> = BTreeMap::new();
  for term in collect_additive_terms(expr) {
    let (num, _key, var_factors) = decompose_term(&term);
    let c = coeff_to_q(&simplify(num))?;
    let mut powers = Vec::new();
    for f in &var_factors {
      let (g, k) = generator_power(f);
      let key = expr_to_string(&g);
      gens.entry(key.clone()).or_insert_with(|| g.clone());
      powers.push((key, g, k));
    }
    terms.push((c, powers));
  }
  let keys: Vec<&String> = gens.keys().collect();
  let mut p = Poly::zero(keys.len());
  for (c, powers) in terms {
    let mut m = vec![0u32; keys.len()];
    for (key, _, k) in powers {
      let pos = keys.iter().position(|g| **g == key)?;
      m[pos] = m[pos].checked_add(k)?;
    }
    p.add_term(m, c);
  }
  Some((p, gens.into_values().collect()))
}

fn big_to_expr(n: &BigInt) -> Expr {
  match n.to_i128() {
    Some(v) => Expr::Integer(v),
    None => Expr::BigInteger(n.clone()),
  }
}

fn q_to_expr(q: &Q) -> Expr {
  if q.d.is_one() {
    big_to_expr(&q.n)
  } else {
    Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![big_to_expr(&q.n), big_to_expr(&q.d)].into(),
    }
  }
}

fn poly_to_expr(p: &Poly, gens: &[Expr]) -> Expr {
  let mut terms: Vec<Expr> = Vec::new();
  for (m, c) in &p.t {
    let mut factors: Vec<Expr> = Vec::new();
    for (i, &e) in m.iter().enumerate() {
      let v = gens[i].clone();
      match e {
        0 => {}
        1 => factors.push(v),
        _ => factors.push(pow2(v, Expr::Integer(e as i128))),
      }
    }
    let term = if factors.is_empty() {
      q_to_expr(c)
    } else if c.n.is_one() && c.d.is_one() {
      build_product(factors)
    } else if (-c.n.clone()).is_one() && c.d.is_one() {
      negate_term(&build_product(factors))
    } else {
      multiply_exprs(&q_to_expr(c), &build_product(factors))
    };
    terms.push(term);
  }
  if terms.is_empty() {
    return Expr::Integer(0);
  }
  combine_and_build(&terms)
}

/// Factor an expanded polynomial.
/// Returns the numeric content (its sign that of the lexicographically
/// leading term) and the irreducible factors with multiplicities, each
/// with coprime integer coefficients and a positive leading term. None
/// when the coefficients are not rational or a cap is exceeded.
pub(super) fn factor_multivariate_poly(
  expanded: &Expr,
) -> Option<(Expr, Vec<(Expr, usize)>)> {
  let (p, gens) = expr_to_poly(expanded)?;
  if p.is_zero() {
    return None;
  }
  let (unit, prim) = p.normalize();
  let mut factors = factor_primitive(&prim)?;
  // Merge equal factors.
  let mut merged: Vec<(Poly, usize)> = Vec::new();
  for (f, k) in factors.drain(..) {
    match merged.iter_mut().find(|(g, _)| *g == f) {
      Some(entry) => entry.1 += k,
      None => merged.push((f, k)),
    }
  }
  Some((
    q_to_expr(&unit),
    merged
      .iter()
      .map(|(f, k)| (poly_to_expr(f, &gens), *k))
      .collect(),
  ))
}
