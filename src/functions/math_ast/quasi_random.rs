//! Quasi-random (low-discrepancy) generators selected by
//! `SeedRandom[seed, Method -> {"MKL", Method -> {"Sobol", "Dimension" -> d}}]`
//! and its `"Niederreiter"` sibling.
//!
//! Unlike the pseudo-random generators, these produce a deterministic
//! sequence of `d`-dimensional points that fill the unit cube evenly. Each
//! draw returns the next coordinate, cycling through the dimensions, so
//! `RandomReal[1, {n, d}]` yields the first `n` points of the sequence.
//! Both generators use 32-bit Gray-code construction (Antonov–Saleev) and
//! start at the origin.

use std::convert::Infallible;

/// Number of bits of every coordinate.
const BITS: usize = 32;

/// Primitive polynomials (as bit patterns, leading and constant terms
/// included) and initial direction numbers `m_1, m_2, …` of the Sobol
/// sequence for dimensions 2–40, after Bratley and Fox (ACM TOMS 659).
/// Dimension 1 is the van der Corput sequence and needs no entry.
const SOBOL_INIT: &[(u32, &[u32])] = &[
  (3, &[1]),
  (7, &[1, 1]),
  (11, &[1, 3, 7]),
  (13, &[1, 1, 5]),
  (19, &[1, 3, 1, 1]),
  (25, &[1, 1, 3, 7]),
  (37, &[1, 3, 3, 9, 9]),
  (59, &[1, 3, 7, 13, 3]),
  (47, &[1, 1, 5, 11, 27]),
  (61, &[1, 3, 5, 1, 15]),
  (55, &[1, 1, 7, 3, 29]),
  (41, &[1, 3, 7, 7, 21]),
  (67, &[1, 1, 1, 9, 23, 37]),
  (97, &[1, 3, 3, 5, 19, 33]),
  (91, &[1, 1, 3, 13, 11, 7]),
  (109, &[1, 1, 7, 13, 25, 5]),
  (103, &[1, 3, 5, 11, 7, 11]),
  (115, &[1, 1, 1, 3, 13, 39]),
  (131, &[1, 3, 1, 15, 17, 63, 13]),
  (193, &[1, 1, 5, 5, 1, 27, 33]),
  (137, &[1, 3, 3, 3, 25, 17, 115]),
  (145, &[1, 1, 3, 15, 29, 15, 41]),
  (143, &[1, 3, 1, 7, 3, 23, 79]),
  (241, &[1, 3, 7, 9, 31, 29, 17]),
  (157, &[1, 1, 5, 13, 11, 3, 29]),
  (185, &[1, 3, 1, 9, 5, 21, 119]),
  (167, &[1, 1, 3, 1, 23, 13, 75]),
  (229, &[1, 3, 3, 11, 27, 31, 73]),
  (171, &[1, 1, 7, 7, 19, 25, 105]),
  (213, &[1, 3, 5, 5, 21, 9, 7]),
  (191, &[1, 1, 1, 15, 5, 49, 59]),
  (253, &[1, 1, 1, 1, 1, 33, 65]),
  (203, &[1, 3, 5, 15, 17, 19, 21]),
  (211, &[1, 1, 7, 11, 13, 29, 3]),
  (239, &[1, 3, 7, 5, 7, 11, 113]),
  (247, &[1, 1, 5, 3, 15, 19, 61]),
  (285, &[1, 3, 1, 1, 9, 27, 89, 7]),
  (369, &[1, 1, 3, 7, 31, 15, 45, 23]),
  (299, &[1, 3, 3, 9, 9, 25, 107, 39]),
];

/// The largest dimension the Sobol generator supports.
const SOBOL_MAX_DIMENSION: usize = SOBOL_INIT.len() + 1;

/// The largest dimension the Niederreiter generator supports (the
/// irreducible polynomials it needs stay well below 32 bits of degree).
const NIEDERREITER_MAX_DIMENSION: usize = 318;

/// Which low-discrepancy sequence a [`QuasiRng`] draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuasiKind {
  Sobol,
  Niederreiter,
}

/// A Gray-code quasi-random point generator.
#[derive(Clone, Debug)]
pub struct QuasiRng {
  /// `directions[r][i]`: the bits XORed into coordinate `i` when bit `r`
  /// of the point index is the lowest zero bit.
  directions: Vec<Vec<u32>>,
  /// The current point, as 32-bit binary fractions.
  point: Vec<u32>,
  /// Index of the current point.
  index: u64,
  /// The coordinate of `point` the next draw returns.
  coordinate: usize,
}

impl QuasiRng {
  /// A generator of `dimension`-dimensional points, or `None` when the
  /// dimension is out of range for the sequence.
  pub fn new(kind: QuasiKind, dimension: usize) -> Option<Self> {
    let max = match kind {
      QuasiKind::Sobol => SOBOL_MAX_DIMENSION,
      QuasiKind::Niederreiter => NIEDERREITER_MAX_DIMENSION,
    };
    if dimension == 0 || dimension > max {
      return None;
    }
    let columns: Vec<Vec<u32>> = (0..dimension)
      .map(|i| match kind {
        QuasiKind::Sobol => sobol_directions(i),
        QuasiKind::Niederreiter => niederreiter_directions(i),
      })
      .collect();
    let directions = (0..BITS)
      .map(|r| columns.iter().map(|c| c[r]).collect())
      .collect();
    Some(QuasiRng {
      directions,
      point: vec![0; dimension],
      index: 0,
      coordinate: 0,
    })
  }

  /// The next coordinate as a 32-bit binary fraction.
  fn next_coordinate(&mut self) -> u32 {
    let value = self.point[self.coordinate];
    self.coordinate += 1;
    if self.coordinate == self.point.len() {
      self.coordinate = 0;
      // Gray-code step: flip by the direction of the lowest zero bit.
      let r = (self.index.trailing_ones() as usize).min(BITS - 1);
      for (x, d) in self.point.iter_mut().zip(&self.directions[r]) {
        *x ^= d;
      }
      self.index += 1;
    }
    value
  }
}

/// A draw puts the coordinate's binary fraction in the high bits, so a
/// uniform float sampled from it is exactly that coordinate.
impl rand::TryRng for QuasiRng {
  type Error = Infallible;

  fn try_next_u32(&mut self) -> Result<u32, Infallible> {
    Ok(self.next_coordinate())
  }

  fn try_next_u64(&mut self) -> Result<u64, Infallible> {
    Ok((self.next_coordinate() as u64) << 32)
  }

  fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> {
    for chunk in dst.chunks_mut(4) {
      let bytes = self.next_coordinate().to_be_bytes();
      chunk.copy_from_slice(&bytes[..chunk.len()]);
    }
    Ok(())
  }
}

/// Sobol direction numbers `v_r = m_r / 2^r` (as 32-bit fractions) of the
/// zero-based dimension `i`.
fn sobol_directions(i: usize) -> Vec<u32> {
  let mut m: Vec<u32> = Vec::with_capacity(BITS);
  if i == 0 {
    m.resize(BITS, 1);
  } else {
    let (poly, init) = SOBOL_INIT[i - 1];
    let degree = poly.ilog2() as usize;
    m.extend_from_slice(init);
    // m_k = 2 a_1 m_{k-1} ⊕ 4 a_2 m_{k-2} ⊕ … ⊕ 2^s m_{k-s} ⊕ m_{k-s}
    for k in degree..BITS {
      let mut next = m[k - degree] ^ (m[k - degree] << degree);
      for j in 1..degree {
        if (poly >> (degree - j)) & 1 == 1 {
          next ^= m[k - j] << j;
        }
      }
      m.push(next);
    }
  }
  m.iter()
    .enumerate()
    .map(|(r, &mr)| mr.wrapping_shl((BITS - 1 - r) as u32))
    .collect()
}

/// The `n`-th (zero-based) irreducible polynomial over GF(2) of degree at
/// least one, in increasing order of its bit pattern: x, x+1, x²+x+1, …
fn irreducible_polynomial(n: usize) -> u64 {
  let is_irreducible = |p: u64| {
    let degree = p.ilog2();
    // Every possible factor has at most half the degree.
    (2u64..1 << (degree / 2 + 1)).all(|q| gf2_rem(p, q) != 0)
  };
  (2u64..).filter(|&p| is_irreducible(p)).nth(n).unwrap()
}

/// Remainder of the GF(2) polynomial division `a mod b`.
fn gf2_rem(mut a: u64, b: u64) -> u64 {
  let db = b.ilog2();
  while a != 0 && a.ilog2() >= db {
    a ^= b << (a.ilog2() - db);
  }
  a
}

/// Product of two GF(2) polynomials given as coefficient vectors (lowest
/// degree first).
fn gf2_mul(a: &[u8], b: &[u8]) -> Vec<u8> {
  let mut out = vec![0u8; a.len() + b.len() - 1];
  for (i, &x) in a.iter().enumerate() {
    if x == 1 {
      for (j, &y) in b.iter().enumerate() {
        out[i + j] ^= y;
      }
    }
  }
  out
}

/// Base-2 Niederreiter generator columns of the zero-based dimension `i`,
/// after Bratley, Fox and Niederreiter (ACM TOMS 738): the matrix entries
/// come from the Laurent coefficients of powers of the dimension's
/// irreducible polynomial, with every free choice set to one.
fn niederreiter_directions(i: usize) -> Vec<u32> {
  let poly = irreducible_polynomial(i);
  let degree = poly.ilog2() as usize;
  let px: Vec<u8> = (0..=degree).map(|k| ((poly >> k) & 1) as u8).collect();
  let mut pb: Vec<u8> = vec![1];
  let mut v: Vec<u8> = vec![0; BITS + degree * (BITS / degree + 2) + 1];
  // c[r][j]: output bit j of the column for index bit r.
  let mut c = vec![[0u8; BITS]; BITS];
  let mut u = 0;
  for j in 0..BITS {
    if u == 0 {
      // B ← B·P; recompute V from the new B (whose old degree is K_j).
      let kj = pb.len() - 1;
      pb = gf2_mul(&px, &pb);
      let m = pb.len() - 1;
      v.fill(0);
      v[kj] = 1;
      for x in v.iter_mut().take(m).skip(kj + 1) {
        *x = 1;
      }
      for r in 0..(v.len() - m) {
        let mut term = 0;
        for k in 0..m {
          term ^= pb[k] & v[r + k];
        }
        v[r + m] = term;
      }
    }
    for (r, row) in c.iter_mut().enumerate() {
      row[j] = v[r + u];
    }
    u += 1;
    if u == degree {
      u = 0;
    }
  }
  c.iter()
    .map(|row| row.iter().fold(0u32, |acc, &b| (acc << 1) | b as u32))
    .collect()
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Each Sobol polynomial has as many initial direction numbers as its
  /// degree, and every `m_k` is odd and below `2^k`.
  #[test]
  fn sobol_table_is_well_formed() {
    for &(poly, init) in SOBOL_INIT {
      let degree = poly.ilog2() as usize;
      assert_eq!(init.len(), degree, "polynomial {poly}");
      assert_eq!(poly & 1, 1, "polynomial {poly}");
      for (k, &m) in init.iter().enumerate() {
        assert!(m % 2 == 1 && m < 1 << (k + 1), "polynomial {poly}");
      }
    }
  }

  /// Rank over GF(2) of the top `k` rows of the first `m` columns of a
  /// generator matrix given by its columns.
  fn leading_rank(columns: &[u32], k: usize, m: usize) -> usize {
    let mut rows: Vec<u32> = (0..k)
      .map(|j| {
        (0..m).fold(0, |acc, r| acc | (((columns[r] >> (31 - j)) & 1) << r))
      })
      .collect();
    let mut rank = 0;
    for bit in 0..m {
      if let Some(p) = (rank..k).find(|&i| (rows[i] >> bit) & 1 == 1) {
        rows.swap(rank, p);
        for i in 0..k {
          if i != rank && (rows[i] >> bit) & 1 == 1 {
            rows[i] ^= rows[rank];
          }
        }
        rank += 1;
      }
    }
    rank
  }

  /// A Sobol coordinate is a (0,1)-sequence; a Niederreiter coordinate
  /// whose polynomial has degree `e` is a (t,1)-sequence with `t < e`.
  #[test]
  fn coordinates_have_the_expected_t_value() {
    for i in 0..SOBOL_MAX_DIMENSION {
      let columns = sobol_directions(i);
      for m in 1..=BITS {
        assert_eq!(leading_rank(&columns, m, m), m, "Sobol dimension {i}");
      }
    }
    for i in 0..60 {
      let columns = niederreiter_directions(i);
      let e = irreducible_polynomial(i).ilog2() as usize;
      for m in e..=BITS {
        let k = m + 1 - e;
        assert_eq!(
          leading_rank(&columns, k, m),
          k,
          "Niederreiter dimension {i}, m = {m}"
        );
      }
    }
  }

  #[test]
  fn irreducible_polynomials_in_order() {
    let first: Vec<u64> = (0..8).map(irreducible_polynomial).collect();
    assert_eq!(first, vec![2, 3, 7, 11, 13, 19, 25, 31]);
  }

  #[test]
  fn dimensions_out_of_range_are_rejected() {
    assert!(QuasiRng::new(QuasiKind::Sobol, 0).is_none());
    assert!(QuasiRng::new(QuasiKind::Sobol, SOBOL_MAX_DIMENSION + 1).is_none());
    assert!(QuasiRng::new(QuasiKind::Sobol, SOBOL_MAX_DIMENSION).is_some());
    assert!(
      QuasiRng::new(QuasiKind::Niederreiter, NIEDERREITER_MAX_DIMENSION)
        .is_some()
    );
  }
}
