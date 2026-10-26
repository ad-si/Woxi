//! Deterministic randomness for the generated example data.
//!
//! Every generator draws from an [`Rng`] seeded from the collection and entry
//! name (plus a salt per use), so the same request yields the same data on
//! every run and on every platform — tests can check exact values.

/// FNV-1a, used to turn names into seeds.
pub fn hash(parts: &[&str]) -> u64 {
  let mut h: u64 = 0xcbf2_9ce4_8422_2325;
  for part in parts {
    for b in part.bytes().chain(std::iter::once(0xff)) {
      h ^= b as u64;
      h = h.wrapping_mul(0x0100_0000_01b3);
    }
  }
  h
}

/// SplitMix64: small, fast, and good enough for synthetic data.
#[derive(Clone)]
pub struct Rng(u64);

impl Rng {
  pub fn new(seed: u64) -> Self {
    Rng(seed)
  }

  /// A generator seeded from `collection`, `entry` and a `salt` naming what
  /// the numbers are for.
  pub fn for_entry(collection: &str, entry: &str, salt: &str) -> Self {
    Rng(hash(&[collection, entry, salt]))
  }

  pub fn next_u64(&mut self) -> u64 {
    self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = self.0;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
  }

  /// Uniform in `[0, 1)`.
  pub fn uniform(&mut self) -> f64 {
    (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
  }

  /// Uniform in `[lo, hi)`.
  pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * self.uniform()
  }

  /// Uniform integer in `0..n` (`n > 0`).
  pub fn below(&mut self, n: usize) -> usize {
    ((self.next_u64() as u128 * n as u128) >> 64) as usize
  }

  /// Uniform integer in `lo..=hi`.
  pub fn int_in(&mut self, lo: i64, hi: i64) -> i64 {
    if hi <= lo {
      return lo;
    }
    lo + self.below((hi - lo + 1) as usize) as i64
  }

  /// Standard normal (Box–Muller).
  pub fn normal(&mut self) -> f64 {
    let u1 = self.uniform().max(1e-300);
    let u2 = self.uniform();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
  }

  pub fn chance(&mut self, p: f64) -> bool {
    self.uniform() < p
  }

  /// A random element of a non-empty slice.
  pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
    &items[self.below(items.len())]
  }

  /// Fisher–Yates shuffle.
  pub fn shuffle<T>(&mut self, items: &mut [T]) {
    for i in (1..items.len()).rev() {
      let j = self.below(i + 1);
      items.swap(i, j);
    }
  }
}

/// Smooth 2D value noise in roughly `[-1, 1]`, periodic with period
/// `period` lattice cells in both directions when `period > 0` (so textures
/// built from it tile seamlessly).
pub struct Noise {
  perm: [u8; 512],
  grad: [f64; 256],
  period: usize,
}

impl Noise {
  pub fn new(seed: u64, period: usize) -> Self {
    let mut rng = Rng::new(seed);
    let mut p: Vec<u8> = (0..=255).collect();
    rng.shuffle(&mut p);
    let mut perm = [0u8; 512];
    for i in 0..512 {
      perm[i] = p[i & 255];
    }
    let mut grad = [0.0; 256];
    for g in &mut grad {
      *g = rng.range(-1.0, 1.0);
    }
    Noise { perm, grad, period }
  }

  fn lattice(&self, x: i64, y: i64) -> f64 {
    let (x, y) = if self.period > 0 {
      let p = self.period as i64;
      (x.rem_euclid(p), y.rem_euclid(p))
    } else {
      (x, y)
    };
    let i = self.perm[(x & 255) as usize] as usize;
    self.grad[self.perm[(i + (y & 255) as usize) & 511] as usize]
  }

  /// Value noise at `(x, y)` with quintic smoothing.
  pub fn at(&self, x: f64, y: f64) -> f64 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (xi, yi) = (x0 as i64, y0 as i64);
    let s = |t: f64| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    let (sx, sy) = (s(fx), s(fy));
    let a = self.lattice(xi, yi);
    let b = self.lattice(xi + 1, yi);
    let c = self.lattice(xi, yi + 1);
    let d = self.lattice(xi + 1, yi + 1);
    let top = a + (b - a) * sx;
    let bottom = c + (d - c) * sx;
    top + (bottom - top) * sy
  }

  /// Fractal sum of `octaves` noise layers, each at twice the frequency and
  /// `gain` times the amplitude of the previous one; normalized to roughly
  /// `[-1, 1]`.
  pub fn fbm(&self, x: f64, y: f64, octaves: u32, gain: f64) -> f64 {
    let (mut sum, mut amp, mut freq, mut norm) = (0.0, 1.0, 1.0, 0.0);
    for _ in 0..octaves {
      sum += amp * self.at(x * freq, y * freq);
      norm += amp;
      amp *= gain;
      freq *= 2.0;
    }
    sum / norm
  }

  /// Turbulence: fractal sum of absolute noise, in `[0, 1]`.
  pub fn turbulence(&self, x: f64, y: f64, octaves: u32) -> f64 {
    let (mut sum, mut amp, mut freq, mut norm) = (0.0, 1.0, 1.0, 0.0);
    for _ in 0..octaves {
      sum += amp * self.at(x * freq, y * freq).abs();
      norm += amp;
      amp *= 0.5;
      freq *= 2.0;
    }
    sum / norm
  }

  /// Gradient (Perlin) noise at `(x, y)` in roughly `[-1, 1]`: smooth and
  /// free of the axis-aligned blocks value noise shows.
  pub fn perlin(&self, x: f64, y: f64) -> f64 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (xi, yi) = (x0 as i64, y0 as i64);
    let grad = |ix: i64, iy: i64, dx: f64, dy: f64| -> f64 {
      let (ix, iy) = if self.period > 0 {
        let p = self.period as i64;
        (ix.rem_euclid(p), iy.rem_euclid(p))
      } else {
        (ix, iy)
      };
      let h = self.perm
        [(self.perm[(ix & 255) as usize] as usize + (iy & 255) as usize) & 511];
      let a = h as f64 * (std::f64::consts::TAU / 256.0);
      a.cos() * dx + a.sin() * dy
    };
    let s = |t: f64| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    let (sx, sy) = (s(fx), s(fy));
    let n00 = grad(xi, yi, fx, fy);
    let n10 = grad(xi + 1, yi, fx - 1.0, fy);
    let n01 = grad(xi, yi + 1, fx, fy - 1.0);
    let n11 = grad(xi + 1, yi + 1, fx - 1.0, fy - 1.0);
    let a = n00 + (n10 - n00) * sx;
    let b = n01 + (n11 - n01) * sx;
    (a + (b - a) * sy) * 1.41
  }

  /// Fractal sum of Perlin octaves, normalized to roughly `[-1, 1]`.
  pub fn pfbm(&self, x: f64, y: f64, octaves: u32, gain: f64) -> f64 {
    let (mut sum, mut amp, mut freq, mut norm) = (0.0, 1.0, 1.0, 0.0);
    for o in 0..octaves {
      // Offset each octave so lattice points do not line up.
      let off = o as f64 * 17.31;
      sum += amp * self.perlin(x * freq + off, y * freq - off);
      norm += amp;
      amp *= gain;
      freq *= 2.0;
    }
    sum / norm
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn deterministic() {
    let mut a = Rng::for_entry("TestImage", "Moon", "x");
    let mut b = Rng::for_entry("TestImage", "Moon", "x");
    for _ in 0..10 {
      assert_eq!(a.next_u64(), b.next_u64());
    }
  }

  #[test]
  fn periodic_noise_tiles() {
    let n = Noise::new(7, 8);
    for i in 0..20 {
      let y = i as f64 * 0.37;
      assert!((n.at(0.0, y) - n.at(8.0, y)).abs() < 1e-12);
    }
  }
}
