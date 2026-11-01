//! Machine-precision Weierstrass functions for complex arguments and
//! invariants: the lattice of the invariants {g2, g3} is found from the roots
//! of 4t^3 - g2 t - g3 (via the arithmetic-geometric mean), reduced to a
//! basis whose period ratio lies in the fundamental domain, and ℘, ℘', ζ and
//! σ are evaluated through the Jacobi theta function θ1 of that lattice.

type C = (f64, f64);

fn add(a: C, b: C) -> C {
  (a.0 + b.0, a.1 + b.1)
}
fn sub(a: C, b: C) -> C {
  (a.0 - b.0, a.1 - b.1)
}
fn mul(a: C, b: C) -> C {
  (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
fn div(a: C, b: C) -> C {
  let d = b.0 * b.0 + b.1 * b.1;
  ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}
fn scale(a: C, s: f64) -> C {
  (a.0 * s, a.1 * s)
}
fn abs(a: C) -> f64 {
  a.0.hypot(a.1)
}
fn sqrt(a: C) -> C {
  let r = abs(a);
  if r == 0.0 {
    return (0.0, 0.0);
  }
  let re = f64::midpoint(r, a.0).sqrt();
  let im = ((r - a.0) / 2.0).sqrt();
  (re, if a.1 < 0.0 { -im } else { im })
}
fn exp(a: C) -> C {
  let m = a.0.exp();
  (m * a.1.cos(), m * a.1.sin())
}
fn sin(a: C) -> C {
  (a.0.sin() * a.1.cosh(), a.0.cos() * a.1.sinh())
}
fn cos(a: C) -> C {
  (a.0.cos() * a.1.cosh(), -a.0.sin() * a.1.sinh())
}
const I: C = (0.0, 1.0);
const PI: f64 = std::f64::consts::PI;

/// The roots of 4t^3 - g2 t - g3, polished by Newton's method.
fn cubic_roots(g2: C, g3: C) -> [C; 3] {
  // Durand-Kerner on t^3 - (g2/4) t - g3/4.
  let p = scale(g2, -0.25);
  let q = scale(g3, -0.25);
  let f = |t: C| add(add(mul(mul(t, t), t), mul(p, t)), q);
  let scale_r = 1.0 + abs(p).sqrt() + abs(q).cbrt();
  let mut roots = [
    scale((0.4, 0.9), scale_r),
    scale((-0.65, 0.72), scale_r),
    scale((0.2, -1.1), scale_r),
  ];
  for _ in 0..500 {
    let mut delta = 0.0f64;
    for i in 0..3 {
      let mut den = (1.0, 0.0);
      for j in 0..3 {
        if i != j {
          den = mul(den, sub(roots[i], roots[j]));
        }
      }
      if abs(den) == 0.0 {
        continue;
      }
      let step = div(f(roots[i]), den);
      roots[i] = sub(roots[i], step);
      delta = delta.max(abs(step));
    }
    if delta <= 1e-17 * scale_r {
      break;
    }
  }
  for r in &mut roots {
    for _ in 0..3 {
      let d = add(scale(mul(*r, *r), 3.0), p);
      if abs(d) == 0.0 {
        break;
      }
      *r = sub(*r, div(f(*r), d));
    }
  }
  roots
}

/// The arithmetic-geometric mean of a and b, taking at each step the square
/// root closer to the arithmetic mean (the principal value of the AGM).
fn agm(mut a: C, mut b: C) -> C {
  for _ in 0..100 {
    let a1 = scale(add(a, b), 0.5);
    let mut b1 = sqrt(mul(a, b));
    if abs(sub(a1, b1)) > abs(add(a1, b1)) {
      b1 = scale(b1, -1.0);
    }
    a = a1;
    b = b1;
    if abs(sub(a, b)) <= 1e-16 * abs(a) {
      break;
    }
  }
  a
}

/// A basis {ω1, ω3} of half-periods with τ = ω3/ω1 in the fundamental
/// domain (|Re τ| ≤ 1/2, |τ| ≥ 1, Im τ > 0).
fn reduce_basis(mut w1: C, mut w3: C) -> (C, C) {
  let mut tau = div(w3, w1);
  if tau.1 < 0.0 {
    w3 = scale(w3, -1.0);
    tau = scale(tau, -1.0);
  }
  for _ in 0..200 {
    let shift = tau.0.round();
    if shift != 0.0 {
      w3 = sub(w3, scale(w1, shift));
      tau = div(w3, w1);
    }
    if abs(tau) < 1.0 - 1e-15 {
      // τ → -1/τ: (ω1, ω3) → (ω3, -ω1).
      let old = w1;
      w1 = w3;
      w3 = scale(old, -1.0);
      tau = div(w3, w1);
    } else {
      break;
    }
  }
  (w1, w3)
}

/// The lattice of the invariants {g2, g3}, as a reduced basis of
/// half-periods. `None` for a degenerate lattice (zero discriminant).
fn lattice(g2: C, g3: C) -> Option<(C, C)> {
  let disc = sub(mul(mul(g2, g2), g2), scale(mul(g3, g3), 27.0));
  let size = abs(g2).powf(1.5).max(abs(g3)).max(1e-300);
  if abs(disc) <= 1e-12 * size * size {
    return None;
  }
  let e = cubic_roots(g2, g3);
  let target = abs(g2).max(abs(g3)).max(1e-300);
  let perms = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
  ];
  for perm in perms {
    let (e1, e2, e3) = (e[perm[0]], e[perm[1]], e[perm[2]]);
    let s13 = sqrt(sub(e1, e3));
    let w1 = div((PI / 2.0, 0.0), agm(s13, sqrt(sub(e1, e2))));
    let w3 = mul(I, div((PI / 2.0, 0.0), agm(s13, sqrt(sub(e2, e3)))));
    for w3 in [w3, scale(w3, -1.0)] {
      let tau = div(w3, w1);
      if tau.1.abs() < 1e-12 {
        continue;
      }
      let (b1, b3) = reduce_basis(w1, w3);
      if let Some((h2, h3)) =
        super::elliptic::weierstrass_invariants_numeric(b1, b3)
        && abs(sub(h2, g2)) <= 1e-8 * target
        && abs(sub(h3, g3)) <= 1e-8 * target
      {
        return Some((b1, b3));
      }
    }
  }
  None
}

/// θ1 and its first three derivatives at v, for the nome q = e^(iπτ).
fn theta1(v: C, tau: C) -> [C; 4] {
  let mut out = [(0.0, 0.0); 4];
  for n in 0..60 {
    let k = 2 * n + 1;
    let e = (n as f64 + 0.5).powi(2);
    // q^((n+1/2)^2) = e^(iπτ (n+1/2)^2)
    let qn = exp(scale(mul(I, tau), PI * e));
    if abs(qn) < 1e-30 && n > 1 {
      break;
    }
    let sign = if n % 2 == 0 { 2.0 } else { -2.0 };
    let c = scale(qn, sign);
    let kv = scale(v, k as f64);
    let kf = k as f64;
    let (s, co) = (sin(kv), cos(kv));
    out[0] = add(out[0], mul(c, s));
    out[1] = add(out[1], mul(scale(c, kf), co));
    out[2] = add(out[2], mul(scale(c, -kf * kf), s));
    out[3] = add(out[3], mul(scale(c, -kf * kf * kf), co));
  }
  out
}

/// The quantities shared by ℘, ζ and σ for one lattice.
struct Lattice {
  w1: C,
  w3: C,
  tau: C,
  eta1: C,
  eta3: C,
  theta1_prime_0: C,
}

impl Lattice {
  fn new(g2: C, g3: C) -> Option<Self> {
    let (w1, w3) = lattice(g2, g3)?;
    let tau = div(w3, w1);
    let t0 = theta1((0.0, 0.0), tau);
    // η1 = -π² θ1'''(0) / (12 ω1 θ1'(0))
    let eta1 = scale(div(t0[3], mul(scale(w1, 12.0), t0[1])), -PI * PI);
    // Legendre's relation η1 ω3 - η3 ω1 = iπ/2.
    let eta3 = div(sub(mul(eta1, w3), scale(I, PI / 2.0)), w1);
    Some(Lattice {
      w1,
      w3,
      tau,
      eta1,
      eta3,
      theta1_prime_0: t0[1],
    })
  }

  /// z reduced into the period parallelogram around the origin, and the
  /// numbers (m, n) of periods 2ω1, 2ω3 taken off.
  fn reduce(&self, z: C) -> (C, f64, f64) {
    // Solve z = x (2ω1) + y (2ω3) for real x, y.
    let (a, b) = (scale(self.w1, 2.0), scale(self.w3, 2.0));
    let det = a.0 * b.1 - a.1 * b.0;
    let x = (z.0 * b.1 - z.1 * b.0) / det;
    let y = (a.0 * z.1 - a.1 * z.0) / det;
    let (m, n) = (x.round(), y.round());
    (sub(sub(z, scale(a, m)), scale(b, n)), m, n)
  }

  /// θ1-based pieces at z: v-derivative ratios θ1'/θ1, θ1''/θ1, θ1'''/θ1.
  fn ratios(&self, z: C) -> (C, C, C) {
    let v = div(scale(z, PI / 2.0), self.w1);
    let t = theta1(v, self.tau);
    (div(t[1], t[0]), div(t[2], t[0]), div(t[3], t[0]))
  }

  fn zeta(&self, z: C) -> C {
    let (z0, m, n) = self.reduce(z);
    let (r1, _, _) = self.ratios(z0);
    let k = div((PI / 2.0, 0.0), self.w1);
    let base = add(div(mul(self.eta1, z0), self.w1), mul(k, r1));
    add(
      base,
      add(scale(self.eta1, 2.0 * m), scale(self.eta3, 2.0 * n)),
    )
  }

  fn p(&self, z: C) -> C {
    let (z0, _, _) = self.reduce(z);
    let (r1, r2, _) = self.ratios(z0);
    let k = div((PI / 2.0, 0.0), self.w1);
    // ℘ = -η1/ω1 + k² ((θ1'/θ1)² - θ1''/θ1)
    add(
      scale(div(self.eta1, self.w1), -1.0),
      mul(mul(k, k), sub(mul(r1, r1), r2)),
    )
  }

  fn p_prime(&self, z: C) -> C {
    let (z0, _, _) = self.reduce(z);
    let (r1, r2, r3) = self.ratios(z0);
    let k = div((PI / 2.0, 0.0), self.w1);
    // d/dv[(θ'/θ)² - θ''/θ] = 2 r1 (r2 - r1²) - (r3 - r2 r1)
    let d = sub(
      scale(mul(r1, sub(r2, mul(r1, r1))), 2.0),
      sub(r3, mul(r2, r1)),
    );
    mul(mul(mul(k, k), k), d)
  }

  fn sigma(&self, z: C) -> C {
    let v = div(scale(z, PI / 2.0), self.w1);
    let t = theta1(v, self.tau);
    // σ = (2ω1/π) exp(η1 z² / (2ω1)) θ1(v) / θ1'(0)
    let pre = scale(self.w1, 2.0 / PI);
    let ex = exp(div(mul(self.eta1, mul(z, z)), scale(self.w1, 2.0)));
    mul(mul(pre, ex), div(t[0], self.theta1_prime_0))
  }
}

/// Which Weierstrass function to evaluate.
#[derive(Clone, Copy)]
pub(crate) enum WeierstrassKind {
  P,
  PPrime,
  Zeta,
  Sigma,
}

/// The Weierstrass function `kind` at complex `z` for complex invariants.
pub(crate) fn weierstrass_complex(
  kind: WeierstrassKind,
  z: C,
  g2: C,
  g3: C,
) -> Option<C> {
  let lattice = Lattice::new(g2, g3)?;
  let value = match kind {
    WeierstrassKind::P => lattice.p(z),
    WeierstrassKind::PPrime => lattice.p_prime(z),
    WeierstrassKind::Zeta => lattice.zeta(z),
    WeierstrassKind::Sigma => lattice.sigma(z),
  };
  (value.0.is_finite() && value.1.is_finite()).then_some(value)
}

/// `InverseWeierstrassP[p, {g2, g3}]` for machine complex `p`, `g2`, `g3`:
/// `u = -R_F(p - e1, p - e2, p - e3)` (so `℘(u) = p`), the representative WL
/// returns. For real invariants WL moves a `u` whose real part lies left of
/// `-ω/2` (`ω` the real half-period `R_F(0, e1 - e2, e1 - e3)`, `e1` the
/// largest real root) one real period right, into `[-ω/2, 3ω/2)`.
pub(crate) fn inverse_weierstrass_p_complex(p: C, g2: C, g3: C) -> Option<C> {
  let e = cubic_roots(g2, g3);
  let rf = super::carlson_rf_complex(sub(p, e[0]), sub(p, e[1]), sub(p, e[2]));
  let mut u = scale(rf, -1.0);
  if g2.1 == 0.0 && g3.1 == 0.0 {
    // The largest real root (the roots with |Im| at rounding level).
    let tol = 1e-9 * (1.0 + abs(g2).sqrt() + abs(g3).cbrt());
    let e1 = e
      .iter()
      .filter(|r| r.1.abs() <= tol)
      .map(|r| r.0)
      .fold(f64::NEG_INFINITY, f64::max);
    if e1.is_finite() {
      let others: Vec<C> = e
        .iter()
        .copied()
        .filter(|r| (r.0 - e1).abs() > tol || r.1.abs() > tol)
        .collect();
      if let [a, b] = others.as_slice() {
        let w = super::carlson_rf_complex(
          (0.0, 0.0),
          sub((e1, 0.0), *a),
          sub((e1, 0.0), *b),
        );
        if w.0.is_finite() && u.0 < -w.0 / 2.0 {
          u.0 += 2.0 * w.0;
        }
        // On [e1, Infinity) the real period segment maps onto p: u is real
        // (the imaginary part left is rounding noise of the complex roots).
        if p.1 == 0.0 && p.0 >= e1 {
          u.1 = 0.0;
        }
      }
    }
  }
  (u.0.is_finite() && u.1.is_finite()).then_some(u)
}
