//! Zassenhaus factorization of square-free primitive univariate integer
//! polynomials: factor mod a good prime with the GF(p) Berlekamp engine,
//! Hensel-lift the modular factors to p^k beyond the Landau–Mignotte
//! coefficient bound, then recombine subsets by trial division over Z.
//!
//! All arithmetic is checked i128 with conservative caps; any overflow or
//! cap breach returns None so the caller can fall back to other methods.

use super::gf_factor::gf_factor_coeffs;
use super::poly_div;

const PRIMES: [i128; 15] =
  [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];
const MAX_MODP_FACTORS: usize = 12;
const MAX_DEGREE: usize = 24;
/// Keep the lift target below 2^30 so the (possibly squared) working
/// modulus stays multiplication-safe in i128.
const MAX_LIFT_TARGET: i128 = 1 << 30;

// ─── modular polynomial arithmetic (ascending trimmed vectors) ───────

fn trim(v: &mut Vec<i128>) {
  while v.len() > 1 && *v.last().unwrap() == 0 {
    v.pop();
  }
}

fn is_zero(v: &[i128]) -> bool {
  v == [0]
}

fn deg(v: &[i128]) -> usize {
  v.len() - 1
}

fn mod_inv(a: i128, m: i128) -> Option<i128> {
  let (mut old_r, mut r) = (a.rem_euclid(m), m);
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
  Some(old_s.rem_euclid(m))
}

fn pmul(a: &[i128], b: &[i128], m: i128) -> Vec<i128> {
  if is_zero(a) || is_zero(b) {
    return vec![0];
  }
  let mut out = vec![0i128; a.len() + b.len() - 1];
  for (i, &x) in a.iter().enumerate() {
    if x == 0 {
      continue;
    }
    for (j, &y) in b.iter().enumerate() {
      out[i + j] = (out[i + j] + x * y).rem_euclid(m);
    }
  }
  trim(&mut out);
  out
}

fn padd(a: &[i128], b: &[i128], m: i128) -> Vec<i128> {
  let mut out = vec![0i128; a.len().max(b.len())];
  for (i, &x) in a.iter().enumerate() {
    out[i] = x;
  }
  for (i, &y) in b.iter().enumerate() {
    out[i] = (out[i] + y).rem_euclid(m);
  }
  trim(&mut out);
  out
}

fn psub(a: &[i128], b: &[i128], m: i128) -> Vec<i128> {
  let mut out = vec![0i128; a.len().max(b.len())];
  for (i, &x) in a.iter().enumerate() {
    out[i] = x;
  }
  for (i, &y) in b.iter().enumerate() {
    out[i] = (out[i] - y).rem_euclid(m);
  }
  trim(&mut out);
  out
}

/// (quotient, remainder) of a by b mod m; requires lc(b) invertible.
fn pdivrem(a: &[i128], b: &[i128], m: i128) -> Option<(Vec<i128>, Vec<i128>)> {
  let db = deg(b);
  let inv = mod_inv(*b.last().unwrap(), m)?;
  let mut r: Vec<i128> = a.iter().map(|c| c.rem_euclid(m)).collect();
  trim(&mut r);
  if is_zero(&r) || deg(&r) < db {
    return Some((vec![0], r));
  }
  let mut q = vec![0i128; deg(&r) - db + 1];
  while !is_zero(&r) && deg(&r) >= db {
    let dr = deg(&r);
    let c = (r[dr] * inv).rem_euclid(m);
    q[dr - db] = c;
    for (i, &bc) in b.iter().enumerate() {
      let idx = dr - db + i;
      r[idx] = (r[idx] - c * bc).rem_euclid(m);
    }
    trim(&mut r);
    if dr == 0 {
      break;
    }
  }
  trim(&mut q);
  Some((q, r))
}

/// Monic gcd over GF(p) with Bezout coefficients: g = s*a + t*b.
fn ext_gcd(
  a: &[i128],
  b: &[i128],
  p: i128,
) -> Option<(Vec<i128>, Vec<i128>, Vec<i128>)> {
  let (mut r0, mut r1) = (a.to_vec(), b.to_vec());
  let (mut s0, mut s1) = (vec![1i128], vec![0i128]);
  let (mut t0, mut t1) = (vec![0i128], vec![1i128]);
  for v in r0.iter_mut() {
    *v = v.rem_euclid(p);
  }
  for v in r1.iter_mut() {
    *v = v.rem_euclid(p);
  }
  trim(&mut r0);
  trim(&mut r1);
  while !is_zero(&r1) {
    let (q, r) = pdivrem(&r0, &r1, p)?;
    let s = psub(&s0, &pmul(&q, &s1, p), p);
    let t = psub(&t0, &pmul(&q, &t1, p), p);
    (r0, r1) = (r1, r);
    (s0, s1) = (s1, s);
    (t0, t1) = (t1, t);
  }
  // Normalize to a monic gcd.
  let inv = mod_inv(*r0.last().unwrap(), p)?;
  let scale = |v: &[i128]| -> Vec<i128> {
    let mut out: Vec<i128> =
      v.iter().map(|c| (c * inv).rem_euclid(p)).collect();
    trim(&mut out);
    out
  };
  Some((scale(&r0), scale(&s0), scale(&t0)))
}

fn derivative(v: &[i128]) -> Vec<i128> {
  if v.len() <= 1 {
    return vec![0];
  }
  let mut out: Vec<i128> = v
    .iter()
    .enumerate()
    .skip(1)
    .map(|(i, &c)| c * i as i128)
    .collect();
  trim(&mut out);
  out
}

/// Map coefficients to symmetric representatives in (-m/2, m/2].
fn symmetric(v: &[i128], m: i128) -> Vec<i128> {
  let half = m / 2;
  let mut out: Vec<i128> = v
    .iter()
    .map(|c| {
      let r = c.rem_euclid(m);
      if r > half { r - m } else { r }
    })
    .collect();
  trim(&mut out);
  out
}

fn gcd_i128(a: i128, b: i128) -> i128 {
  let (mut a, mut b) = (a.abs(), b.abs());
  while b != 0 {
    (a, b) = (b, a % b);
  }
  a
}

/// Primitive part with positive leading coefficient; also returns the
/// removed (signed) content.
fn primitive_part(v: &[i128]) -> (Vec<i128>, i128) {
  let mut content = 0i128;
  for &c in v {
    content = gcd_i128(content, c);
  }
  let content = content.max(1);
  let sign = if *v.last().unwrap() < 0 { -1 } else { 1 };
  let out: Vec<i128> = v.iter().map(|c| c / content / sign).collect();
  (out, content * sign)
}

// ─── Hensel lifting ──────────────────────────────────────────────────

/// One quadratic Hensel step: from f ≡ g·h (mod m), s·g + t·h ≡ 1 (mod m)
/// with h monic, to the same congruences mod m². All values reduced mod m².
#[allow(clippy::type_complexity)]
fn hensel_step(
  f: &[i128],
  g: &[i128],
  h: &[i128],
  s: &[i128],
  t: &[i128],
  m: i128,
) -> Option<(Vec<i128>, Vec<i128>, Vec<i128>, Vec<i128>)> {
  let m2 = m.checked_mul(m)?;
  let e = psub(f, &pmul(g, h, m2), m2);
  let (q, r) = pdivrem(&pmul(s, &e, m2), h, m2)?;
  let g1 = padd(g, &padd(&pmul(t, &e, m2), &pmul(&q, g, m2), m2), m2);
  let h1 = padd(h, &r, m2);
  let b = psub(&padd(&pmul(s, &g1, m2), &pmul(t, &h1, m2), m2), &[1], m2);
  let (c, d) = pdivrem(&pmul(s, &b, m2), &h1, m2)?;
  let s1 = psub(s, &d, m2);
  let t1 = psub(t, &padd(&pmul(t, &b, m2), &pmul(&c, &g1, m2), m2), m2);
  Some((g1, h1, s1, t1))
}

/// Lift the monic mod-p factorization f ≡ lc·g1···gr (mod p) to factors
/// mod M >= target (M a power of p), returned monic mod M.
fn hensel_lift_tree(
  f: &[i128],
  factors: &[Vec<i128>],
  p: i128,
  target: i128,
) -> Option<Vec<Vec<i128>>> {
  if factors.len() == 1 {
    // Single factor: f is lc * g1; return the monic reduction of f itself.
    let mut m = p;
    while m < target {
      m = m.checked_mul(m)?;
    }
    let inv = mod_inv((*f.last().unwrap()).rem_euclid(m), m)?;
    let mut g: Vec<i128> = f.iter().map(|c| (c * inv).rem_euclid(m)).collect();
    trim(&mut g);
    return Some(vec![g]);
  }
  let mid = factors.len() / 2;
  let (left, right) = factors.split_at(mid);
  // g carries the leading coefficient; h is monic.
  let lc = (*f.last().unwrap()).rem_euclid(p);
  let mut g = vec![lc];
  for fac in left {
    g = pmul(&g, fac, p);
  }
  let mut h = vec![1i128];
  for fac in right {
    h = pmul(&h, fac, p);
  }
  let (one, s, t) = ext_gcd(&g, &h, p)?;
  if one != [1] {
    return None; // factors not coprime mod p (should not happen)
  }
  let (mut gl, mut hl, mut sl, mut tl) =
    (g.clone(), h.clone(), s.clone(), t.clone());
  let mut m = p;
  while m < target {
    let (g1, h1, s1, t1) = hensel_step(f, &gl, &hl, &sl, &tl, m)?;
    (gl, hl, sl, tl) = (g1, h1, s1, t1);
    m = m.checked_mul(m)?;
  }
  // Recurse into both halves with the lifted images as the new targets.
  let mut out = hensel_lift_tree(&gl, left, p, target)?;
  out.extend(hensel_lift_tree(&hl, right, p, target)?);
  Some(out)
}

// ─── main entry ──────────────────────────────────────────────────────

/// Factor a primitive square-free integer polynomial into irreducible
/// primitive factors with positive leading coefficients, plus a -1
/// content entry when the signs require it. None = bail (caps/overflow
/// or no usable prime); Some with one factor = proven irreducible.
pub(super) fn zassenhaus_int_factors(
  coeffs: &[i128],
) -> Option<Vec<Vec<i128>>> {
  let mut f = coeffs.to_vec();
  trim(&mut f);
  let n = deg(&f);
  if !(2..=MAX_DEGREE).contains(&n) {
    return None;
  }
  // Work with the positive-lc primitive part; the removed (signed)
  // content is reattached as a constant entry at the end.
  let (f, content) = primitive_part(&f);
  let lc = *f.last().unwrap();

  // Pick the prime (lc nonzero mod p, f squarefree mod p) that yields the
  // fewest modular factors among the first few usable ones.
  let mut best: Option<(i128, Vec<Vec<i128>>)> = None;
  let mut tried = 0;
  for &p in &PRIMES {
    if lc % p == 0 {
      continue;
    }
    let fp = derivative(&f);
    let Some((g, _, _)) = ext_gcd(&f, &fp, p) else {
      continue;
    };
    if g != [1] {
      continue; // not squarefree mod p
    }
    let Some((_, facs)) = gf_factor_coeffs(&f, p) else {
      continue;
    };
    let facs: Vec<Vec<i128>> = facs.into_iter().map(|(v, _)| v).collect();
    if best.as_ref().is_none_or(|(_, b)| facs.len() < b.len()) {
      best = Some((p, facs));
    }
    tried += 1;
    if tried >= 4 {
      break;
    }
  }
  let (p, modp_factors) = best?;
  if modp_factors.len() == 1 {
    let (prim, _) = primitive_part(&f);
    return Some(vec![prim]);
  }
  if modp_factors.len() > MAX_MODP_FACTORS {
    return None;
  }

  // Landau–Mignotte bound: |coeff of any factor| <= 2^n * ||f||_2, and
  // candidates are premultiplied by lc.
  let norm_sq: i128 = f
    .iter()
    .try_fold(0i128, |acc, &c| acc.checked_add(c.checked_mul(c)?))?;
  let norm = (norm_sq as f64).sqrt().ceil() as i128;
  let bound = (1i128 << n.min(60))
    .checked_mul(norm)?
    .checked_mul(lc.abs())?;
  let target = bound.checked_mul(2)?.checked_add(1)?;
  if target > MAX_LIFT_TARGET {
    return None;
  }

  let lifted = hensel_lift_tree(&f, &modp_factors, p, target)?;
  // The working modulus the lift ended at.
  let mut modulus = p;
  while modulus < target {
    modulus = modulus.checked_mul(modulus)?;
  }

  // Subset recombination by trial division over Z.
  let mut avail: Vec<Vec<i128>> = lifted;
  let mut remaining = f.clone();
  let mut result: Vec<Vec<i128>> = Vec::new();
  let mut size = 1usize;
  'outer: while 2 * size <= avail.len() {
    let idxs: Vec<usize> = (0..avail.len()).collect();
    for combo in combinations(&idxs, size) {
      let lc_r = (*remaining.last().unwrap()).rem_euclid(modulus);
      let mut cand = vec![lc_r];
      for &i in &combo {
        cand = pmul(&cand, &avail[i], modulus);
      }
      let cand = symmetric(&cand, modulus);
      let (cand_prim, _) = primitive_part(&cand);
      if deg(&cand_prim) == 0 {
        continue;
      }
      if let Some((q, rem)) = poly_div(&remaining, &cand_prim)
        && rem.iter().all(|&c| c == 0)
      {
        result.push(cand_prim);
        remaining = q;
        trim(&mut remaining);
        // Remove the consumed modular factors, restart at this size.
        let mut combo = combo;
        combo.sort_unstable_by(|a, b| b.cmp(a));
        for i in combo {
          avail.remove(i);
        }
        continue 'outer;
      }
    }
    size += 1;
  }
  if deg(&remaining) > 0 {
    let (prim, _) = primitive_part(&remaining);
    result.push(prim);
  }
  if result.len() == 1 && content == 1 {
    return Some(result); // proven irreducible
  }
  if content != 1 {
    result.push(vec![content]);
  }
  Some(result)
}

/// All size-k index combinations (lexicographic).
fn combinations(items: &[usize], k: usize) -> Vec<Vec<usize>> {
  let mut out = Vec::new();
  let n = items.len();
  if k > n {
    return out;
  }
  let mut idx: Vec<usize> = (0..k).collect();
  loop {
    out.push(idx.iter().map(|&i| items[i]).collect());
    // advance
    let mut i = k;
    loop {
      if i == 0 {
        return out;
      }
      i -= 1;
      if idx[i] != i + n - k {
        break;
      }
    }
    idx[i] += 1;
    for j in i + 1..k {
      idx[j] = idx[j - 1] + 1;
    }
  }
}
