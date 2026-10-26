//! Procedural stand-ins for `"Texture"` and `"ColorTexture"` entries.
//!
//! Each entry has its own recipe, written to resemble the original's
//! structure — vertical bricks where the original's bricks stand upright,
//! cracked plates for bark, chevron columns for herringbone, rows of
//! shingles, a hexagonal hole grid, straw strokes, ripples running the way
//! the original's do — but drawn from noise, cells and strokes, never from
//! the original's pixels. Most recipes produce a brightness field; the
//! caller maps it onto the recorded per-channel mean and spread, which gives
//! the original's colours (a two-tone pattern's field is the motif mask, so
//! it becomes exactly two colours). A few multi-coloured entries (flowers,
//! rust, fur) are painted in colour directly.

use std::f64::consts::{PI, TAU};

use super::raster::*;
use super::rng::{Noise, Rng};

/// A scalar image, `0` dark to `1` bright.
pub struct Field {
  pub w: usize,
  pub h: usize,
  pub v: Vec<f64>,
}

impl Field {
  pub fn new(w: usize, h: usize, f: impl Fn(f64, f64) -> f64) -> Field {
    let mut v = Vec::with_capacity(w * h);
    for y in 0..h {
      for x in 0..w {
        v.push(f(x as f64 + 0.5, y as f64 + 0.5));
      }
    }
    Field { w, h, v }
  }

  pub fn get(&self, x: isize, y: isize) -> f64 {
    let x = x.clamp(0, self.w as isize - 1) as usize;
    let y = y.clamp(0, self.h as isize - 1) as usize;
    self.v[y * self.w + x]
  }

  /// Shade the field as a height map lit from `light` (unit-ish vector in
  /// the image plane): `0.5 + strength · slope along the light`.
  #[must_use]
  pub fn relief(&self, light: (f64, f64), strength: f64) -> Field {
    let mut out = Vec::with_capacity(self.v.len());
    for y in 0..self.h as isize {
      for x in 0..self.w as isize {
        let gx = self.get(x + 1, y) - self.get(x - 1, y);
        let gy = self.get(x, y + 1) - self.get(x, y - 1);
        out.push(0.5 + strength * (gx * light.0 + gy * light.1));
      }
    }
    Field {
      w: self.w,
      h: self.h,
      v: out,
    }
  }

  pub fn map(&mut self, f: impl Fn(f64) -> f64) {
    for v in &mut self.v {
      *v = f(*v);
    }
  }

  pub fn combine(&mut self, other: &Field, f: impl Fn(f64, f64) -> f64) {
    for (a, b) in self.v.iter_mut().zip(&other.v) {
      *a = f(*a, *b);
    }
  }

  /// A dark band of `rows` rows at the bottom (or top), like the frame edge
  /// visible in some of the original photographs.
  pub fn band(&mut self, rows: usize, bottom: bool, value: f64) {
    let rows = rows.min(self.h);
    let range = if bottom {
      self.h - rows..self.h
    } else {
      0..rows
    };
    for y in range {
      for x in 0..self.w {
        self.v[y * self.w + x] = value;
      }
    }
  }

  pub fn to_canvas(&self) -> Canvas {
    let mut c = Canvas::new(self.w, self.h, BLACK);
    for y in 0..self.h {
      for x in 0..self.w {
        let g = self.v[y * self.w + x].clamp(0.0, 1.0);
        c.set(x, y, [g, g, g]);
      }
    }
    c
  }

  /// The luma of a canvas as a field.
  pub fn from_canvas(c: &Canvas) -> Field {
    Field::new(c.width, c.height, |x, y| {
      luma(c.get(x as usize, y as usize))
    })
  }
}

/// Random points with a grid index for nearest-neighbour queries (cells,
/// stones, grains). `aspect` stretches cells horizontally (> 1) or
/// vertically (< 1).
pub struct Cells {
  pts: Vec<(f64, f64)>,
  tone: Vec<f64>,
  grid: Vec<Vec<usize>>,
  gw: usize,
  gh: usize,
  cell: f64,
  aspect: f64,
}

impl Cells {
  pub fn new(
    rng: &mut Rng,
    w: usize,
    h: usize,
    n: usize,
    aspect: f64,
  ) -> Cells {
    Cells::with_jitter(rng, w, h, n, aspect, 0.8)
  }

  /// As [`Cells::new`], with `jitter` (0 regular … 1 fully random within
  /// a grid cell); at full jitter some points are dropped, for irregular
  /// sizes.
  pub fn with_jitter(
    rng: &mut Rng,
    w: usize,
    h: usize,
    n: usize,
    aspect: f64,
    jitter: f64,
  ) -> Cells {
    // Work in a space where cells are round: x is divided by `aspect`.
    let (sw, sh) = (w as f64 / aspect, h as f64);
    let cell = ((sw * sh) / n.max(1) as f64).sqrt().max(1.0);
    let gw = (sw / cell).ceil() as usize + 1;
    let gh = (sh / cell).ceil() as usize + 1;
    let mut grid = vec![Vec::new(); gw * gh];
    let mut pts = Vec::with_capacity(n);
    let mut tone = Vec::with_capacity(n);
    // Jittered grid placement keeps the cells evenly sized.
    for gy in 0..gh {
      for gx in 0..gw {
        let lo = 0.5 - jitter / 2.0;
        let copies = if jitter > 0.9 {
          [0, 1, 1, 1][rng.below(4)]
        } else {
          1
        };
        for _ in 0..copies {
          let p = (
            (gx as f64 + rng.range(lo, 1.0 - lo)) * cell,
            (gy as f64 + rng.range(lo, 1.0 - lo)) * cell,
          );
          grid[gy * gw + gx].push(pts.len());
          pts.push(p);
          tone.push(rng.uniform());
        }
      }
    }
    Cells {
      pts,
      tone,
      grid,
      gw,
      gh,
      cell,
      aspect,
    }
  }

  /// `(nearest distance, second nearest distance, nearest index)` in
  /// pixels (of the stretched space).
  pub fn query(&self, x: f64, y: f64) -> (f64, f64, usize) {
    let sx = x / self.aspect;
    let (cx, cy) = ((sx / self.cell) as isize, (y / self.cell) as isize);
    let (mut d1, mut d2, mut k1) = (f64::MAX, f64::MAX, 0);
    for gy in cy - 2..=cy + 2 {
      for gx in cx - 2..=cx + 2 {
        if gx < 0 || gy < 0 || gx >= self.gw as isize || gy >= self.gh as isize
        {
          continue;
        }
        for &k in &self.grid[gy as usize * self.gw + gx as usize] {
          let (px, py) = self.pts[k];
          let d = ((sx - px).powi(2) + (y - py).powi(2)).sqrt();
          if d < d1 {
            d2 = d1;
            d1 = d;
            k1 = k;
          } else if d < d2 {
            d2 = d;
          }
        }
      }
    }
    (d1, d2, k1)
  }

  pub fn tone(&self, k: usize) -> f64 {
    self.tone[k]
  }

  pub fn size(&self) -> f64 {
    self.cell
  }
}

/// Draw `n` straight strokes: position uniform, angle `angle ± spread`
/// (radians, 0 = pointing right, measured clockwise because y grows down),
/// length in `len`, width `width`, brightness from `tone`.
#[allow(clippy::too_many_arguments)]
fn strokes(
  c: &mut Canvas,
  rng: &mut Rng,
  n: usize,
  angle: f64,
  spread: f64,
  len: (f64, f64),
  width: f64,
  tone: (f64, f64),
) {
  let (w, h) = (c.width as f64, c.height as f64);
  for _ in 0..n {
    let (x, y) = (rng.range(-0.1, 1.1) * w, rng.range(-0.1, 1.1) * h);
    let a = angle + rng.range(-spread, spread);
    let l = rng.range(len.0, len.1);
    let g = rng.range(tone.0, tone.1);
    let (dx, dy) = (a.cos() * l / 2.0, a.sin() * l / 2.0);
    c.line(
      (x - dx, y - dy),
      (x + dx, y + dy),
      width,
      &Paint::Solid([g; 3]),
    );
  }
}

/// How brick courses are laid.
struct Bricks {
  /// Brick length and height in pixels (along and across the course).
  long: f64,
  short: f64,
  /// Courses run vertically (upright bricks) instead of horizontally.
  vertical: bool,
  /// Offset of alternate courses as a fraction of a brick (0.5 running
  /// bond, 0 stack bond).
  offset: f64,
  /// Mortar width as a fraction of the brick height.
  mortar: f64,
  /// Mortar brightness relative to the bricks' mean (> 0 lighter).
  mortar_tone: f64,
}

/// Brick brightness at pattern coordinates `(u, v)` (pixels).
fn brick_at(b: &Bricks, noise: &Noise, u: f64, v: f64) -> f64 {
  let (along, across) = if b.vertical { (v, u) } else { (u, v) };
  let course = (across / b.short).floor();
  let fa = across / b.short - course;
  let shift = if course as i64 % 2 == 0 {
    0.0
  } else {
    b.offset
  };
  let pos = along / b.long + shift;
  let idx = pos.floor();
  let fb = pos - idx;
  let joint = b.mortar / 2.0;
  let in_mortar = fa < joint
    || fa > 1.0 - joint
    || fb < joint * b.short / b.long
    || fb > 1.0 - joint * b.short / b.long;
  let grain = noise.pfbm(u / 3.0, v / 3.0, 3, 0.6) * 0.12;
  if in_mortar {
    0.5 + b.mortar_tone + grain * 0.5
  } else {
    let tone = noise
      .perlin(idx * 3.17 + course * 1.31, course * 2.71 - idx * 0.7)
      * 0.07;
    0.5 + tone + grain + noise.pfbm(u / 25.0 + 50.0, v / 25.0, 2, 0.5) * 0.06
  }
}

/// Perspective: pattern coordinates seen on a wall receding towards the
/// right (`k > 0`) or left (`k < 0`); rows converge on the vanishing side.
fn recede(x: f64, y: f64, w: f64, h: f64, k: f64) -> (f64, f64) {
  let t = if k >= 0.0 { x / w } else { 1.0 - x / w };
  let depth = 1.0 + k.abs() * t;
  let u = if k >= 0.0 {
    x * (1.0 + k.abs() * 0.5) / depth
  } else {
    w - (w - x) * (1.0 + k.abs() * 0.5) / depth
  };
  (u, h / 2.0 + (y - h / 2.0) * depth)
}

/// The recipe for one texture entry: a brightness field, or `None` for
/// names without one (handled by the colour painter).
#[allow(clippy::too_many_lines)]
fn field(
  name: &str,
  w: usize,
  h: usize,
  rng: &mut Rng,
  noise: &Noise,
  noise2: &Noise,
) -> Option<Field> {
  let (wf, hf) = (w as f64, h as f64);
  let s = wf.min(hf) / 512.0; // scale relative to a 512-pixel texture
  let base = name.trim_end_matches(|c: char| c.is_ascii_digit());
  let variant = name[base.len()..].parse::<usize>().unwrap_or(1);
  Some(match base {
    "Bark" => {
      // Plates separated by deep, irregular cracks, elongated vertically.
      let n = (wf * hf / (55.0 * s).powi(2)) as usize + 20;
      let cells = Cells::new(rng, w, h, n, 0.4);
      let mut f = Field::new(w, h, |x, y| {
        let (wx, wy) = (
          x + noise.pfbm(x / 30.0, y / 30.0, 4, 0.55) * 18.0 * s,
          y + noise.pfbm(x / 30.0 + 9.0, y / 30.0, 4, 0.55) * 18.0 * s,
        );
        let (d1, d2, k) = cells.query(wx, wy);
        let width =
          cells.size() * (0.1 + 0.08 * noise2.perlin(x / 20.0, y / 20.0));
        let crack = ((d2 - d1) / width.max(0.5)).clamp(0.0, 1.0);
        let plate = 0.6
          + 0.12 * cells.tone(k)
          + noise2.pfbm(x / 3.0, y / 3.0, 4, 0.65) * 0.3;
        plate * crack.powf(0.7)
      });
      if variant == 2 {
        f.map(|v| v.powf(1.4));
      }
      f
    }
    "BrickWall" => {
      let b = Bricks {
        long: 36.0 * s,
        short: 14.0 * s,
        vertical: false,
        offset: 0.0,
        mortar: 0.18,
        mortar_tone: 0.15,
      };
      let mut f = Field::new(w, h, |x, y| brick_at(&b, noise, x, y) * 0.8);
      // A ledge and a dark base with openings, as in the photograph.
      let ledge = hf * 0.88;
      for y in 0..h {
        for x in 0..w {
          let (xf, yf) = (x as f64, y as f64);
          let v = &mut f.v[y * w + x];
          if yf > ledge && yf < ledge + 0.04 * hf {
            *v = 0.75;
          } else if yf >= ledge + 0.04 * hf {
            let opening = ((xf / (wf / 5.5)).fract() - 0.5).abs() < 0.12
              && yf > ledge + 0.08 * hf;
            *v = if opening { 0.05 } else { 0.3 };
          }
        }
      }
      f
    }
    "Bricks" => {
      let (long, short, vertical, offset, k) = match variant {
        1 => (110.0, 56.0, true, 0.5, 0.0),
        2 => (120.0, 48.0, true, 0.5, 0.0),
        3 => (150.0, 90.0, true, 0.5, 0.0),
        4 => (60.0, 14.0, false, 0.5, -1.6),
        5 => (90.0, 18.0, true, 0.0, 0.0),
        6 => (90.0, 20.0, true, 0.5, -1.4),
        7 => (90.0, 40.0, true, 0.5, 0.0),
        _ => (70.0, 16.0, false, 0.5, 0.0),
      };
      let b = Bricks {
        long: long * s,
        short: short * s,
        vertical,
        offset,
        mortar: 0.16,
        mortar_tone: if variant == 2 { 0.25 } else { 0.1 },
      };
      let mut f = Field::new(w, h, |x, y| {
        let (u, v) = if k == 0.0 {
          (x, y)
        } else {
          recede(x, y, wf, hf, k)
        };
        // Bricks7 is a pavement seen at an angle: courses narrow upwards.
        let (u, v) = if variant == 7 {
          (wf / 2.0 + (u - wf / 2.0) * (1.0 + 0.4 * (1.0 - y / hf)), v)
        } else {
          (u, v)
        };
        brick_at(&b, noise, u, v)
      });
      if variant == 2 {
        f.combine(
          &Field::new(w, h, |x, y| noise2.perlin(x / 1.5, y / 1.5)),
          |a, n| (a + n * 0.25).clamp(0.0, 1.0),
        );
      }
      if matches!(variant, 4 | 6) {
        f.band((0.03 * hf) as usize, true, 0.0);
      }
      if variant == 5 {
        f.band((0.03 * hf) as usize, true, 0.05);
        f.band((0.02 * hf) as usize, false, 0.05);
      }
      f
    }
    "Bubbles" => {
      // Foam: two layers of cells with bright thin walls, darker inside.
      let a = Cells::with_jitter(
        rng,
        w,
        h,
        (wf * hf / (34.0 * s).powi(2)) as usize + 10,
        1.0,
        1.0,
      );
      let b = Cells::with_jitter(
        rng,
        w,
        h,
        (wf * hf / (60.0 * s).powi(2)) as usize + 6,
        1.0,
        1.0,
      );
      Field::new(w, h, |x, y| {
        let wall = |c: &Cells| {
          let (d1, d2, _) = c.query(x, y);
          ((-(d2 - d1) / (1.5 * s.max(0.6))).exp(), d1 / c.size())
        };
        let (wa, ra) = wall(&a);
        let (wb, _) = wall(&b);
        let inner =
          0.15 + 0.25 * ra + noise.pfbm(x / 15.0, y / 15.0, 3, 0.5) * 0.12;
        (inner + 0.8 * wa + 0.4 * wb).min(1.0)
      })
    }
    "Cloth" => {
      let threads = 3.0 * s.max(0.6);
      let mottle_scale = if variant == 1 { 70.0 } else { 22.0 };
      Field::new(w, h, |x, y| {
        let (i, j) =
          ((x / threads).floor() as i64, (y / threads).floor() as i64);
        let weave = if (i + j) % 2 == 0 { 0.1 } else { -0.1 };
        let mottle = noise.pfbm(x / mottle_scale, y / mottle_scale, 5, 0.6)
          * if variant == 1 { 0.18 } else { 0.4 };
        let speck = noise2.perlin(x / 2.0, y / 2.0) * 0.15;
        0.5 + weave + mottle + speck
      })
    }
    "Grass" => {
      let mut c = Canvas::new(w, h, [0.18; 3]);
      let blades = (16000.0 * s * s) as usize;
      // Later variants are photographed at a lower angle: shorter, more
      // upright blades.
      let (len, spread) = if variant >= 4 {
        ((6.0 * s, 16.0 * s), 0.5)
      } else {
        ((8.0 * s, 26.0 * s), 1.1)
      };
      strokes(
        &mut c,
        rng,
        blades,
        -PI / 2.0,
        spread,
        len,
        1.4 * s.max(0.6),
        (0.05, 0.95),
      );
      let mut f = Field::from_canvas(&c);
      if variant == 4 {
        f.band((0.02 * hf) as usize, false, 0.05);
        f.band((0.015 * hf) as usize, true, 0.05);
      }
      f
    }
    "Gravel" => {
      let cells = Cells::with_jitter(
        rng,
        w,
        h,
        (wf * hf / (38.0 * s).powi(2)) as usize + 20,
        1.2,
        1.0,
      );
      Field::new(w, h, |x, y| {
        let (d1, d2, k) = cells.query(
          x + noise.perlin(x / 15.0, y / 15.0) * 6.0,
          y + noise.perlin(x / 15.0 + 7.0, y / 15.0) * 6.0,
        );
        let edge = ((d2 - d1) / (cells.size() * 0.12)).clamp(0.0, 1.0);
        let dome = 1.0 - (d1 / cells.size()).min(1.0) * 0.4;
        (0.55 + 0.35 * cells.tone(k)) * dome * edge.powf(0.5)
          + noise2.perlin(x / 3.0, y / 3.0) * 0.05
      })
    }
    "Herringbone" => {
      // Columns of twill, the diagonal flipping from column to column.
      let col = wf / 7.0;
      let pitch = 4.0 * s.max(0.5);
      Field::new(w, h, |x, y| {
        let c = (x / col).floor() as i64;
        let dir = if c % 2 == 0 { 1.0 } else { -1.0 };
        let local = x - c as f64 * col;
        let t = ((y + dir * local) / pitch).rem_euclid(1.0);
        let twill = if t < 0.5 { 0.12 } else { -0.12 };
        let fiber = noise.perlin(x / 1.5, y / 6.0)
          * if variant == 1 { 0.1 } else { 0.25 };
        let mottle = noise2.pfbm(x / 90.0, y / 90.0, 3, 0.5) * 0.06;
        0.5 + twill + fiber + mottle
      })
    }
    "HexagonalHoles" => {
      let d = wf / 46.0;
      let r = d * 0.36;
      Field::new(w, h, |x, y| {
        let row = (y / (d * 0.866)).round();
        let cy = row * d * 0.866;
        let shift = if row as i64 % 2 == 0 { 0.0 } else { d / 2.0 };
        let cx = ((x - shift) / d).round() * d + shift;
        let dist = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
        if dist < r {
          0.05
        } else {
          0.92 + noise.perlin(x / 30.0, y / 30.0) * 0.03
        }
      })
    }
    "Leather" | "Pigskin" => {
      // Fine crinkled relief; leather's grain runs vertically.
      let stretch = if base == "Leather" { 3.0 } else { 1.0 };
      let scale = if base == "Leather" { 6.0 } else { 11.0 } * s.max(0.5);
      let height = Field::new(w, h, |x, y| {
        noise.pfbm(x / scale, y / (scale * stretch), 4, 0.55)
          + noise2.perlin(x / (scale * 4.0), y / (scale * 4.0)) * 0.5
      });
      let mut f = height.relief((0.7, -0.7), 3.0);
      if base == "Pigskin" && variant <= 2 {
        f.band((0.05 * hf) as usize, true, 0.0);
      }
      f
    }
    "MetalGrates" => {
      // A grating seen at an angle: bars in two directions, receding.
      let mut f = Field::new(w, h, |x, y| {
        let depth = 0.4 + 1.2 * y / hf;
        let u = (x - wf * 0.3) / depth + y * 0.6;
        let v = y / depth;
        let a = ((u + v * 0.5) / (14.0 * s)).fract() < 0.35;
        let b = ((v * 1.8 - u * 0.3) / (14.0 * s)).fract() < 0.3;
        if a || b {
          0.7 + noise.perlin(x / 4.0, y / 4.0) * 0.05
        } else {
          0.35
        }
      });
      f.band((0.06 * hf) as usize, true, 0.05);
      f
    }
    "Mosaic" => return Some(mosaic(variant, w, h, rng, noise, noise2)),
    "Raffia" => {
      let t = wf / 35.0;
      let height = Field::new(w, h, |x, y| {
        let (i, j) = (
          (x / (2.0 * t)).floor() as i64,
          (y / (2.0 * t)).floor() as i64,
        );
        let horizontal = (i + j) % 2 == 0;
        let (a, b) = if horizontal { (x, y) } else { (y, x) };
        let ridge = ((b / t) * PI).sin().abs();
        ridge * 0.6
          + noise.perlin(a / 8.0, b / 2.0) * 0.4
          + noise2.perlin(x / 3.0, y / 3.0) * 0.2
      });
      height.relief((0.6, -0.8), 1.6)
    }
    "Sand" => {
      // Grains: small cells of random brightness, slightly domed.
      let grain = match variant {
        2 | 3 => 5.5,
        4 | 5 => 2.0,
        _ => 3.5,
      } * s.max(0.5);
      let cells = Cells::with_jitter(
        rng,
        w,
        h,
        (wf * hf / (grain * grain)) as usize + 10,
        1.0,
        1.0,
      );
      let contrast = if variant == 2 { 0.7 } else { 0.35 };
      let mut f = Field::new(w, h, |x, y| {
        let (d1, d2, k) = cells.query(x, y);
        let edge = ((d2 - d1) / (grain * 0.25)).clamp(0.0, 1.0);
        (0.6 + contrast * (cells.tone(k) - 0.5)) * (0.7 + 0.3 * edge)
          + noise2.pfbm(x / 60.0, y / 60.0, 2, 0.5) * 0.05
      });
      if matches!(variant, 1 | 6) {
        for _ in 0..(0.002 * wf * hf) as usize {
          let (x, y) = (rng.below(w), rng.below(h));
          f.v[y * w + x] = 0.05;
        }
      }
      if matches!(variant, 4 | 5) {
        f.band((0.04 * hf) as usize, true, 0.0);
      }
      f
    }
    "Shingles" => {
      // Rows of rectangular wooden shingles with dark joints.
      let row = hf / if variant == 1 { 22.0 } else { 12.0 };
      let slope = if variant == 2 { -0.08 } else { 0.0 };
      let mut f = Field::new(w, h, |x, y| {
        let yy = y - x * slope;
        let r = (yy / row).floor();
        let fy = yy / row - r;
        let mut sx = x / (row * 1.2) + noise.perlin(r * 3.3, 0.5) * 3.0;
        // Uneven shingle widths.
        sx += noise.perlin(sx * 0.7, r) * 0.4;
        let k = sx.floor();
        let fx = sx - k;
        let tone = noise2.perlin(k * 1.7, r * 2.3) * 0.15;
        let streak = noise.perlin(x / 2.0, y / 20.0) * 0.08;
        if fx < 0.06 {
          0.12
        } else if fy > 0.85 {
          0.25
        } else {
          0.5 + tone + streak + 0.15 * fy
        }
      });
      if variant == 1 {
        f.band((0.03 * hf) as usize, false, 0.85);
        f.band((0.1 * hf) as usize, true, 0.3);
      } else {
        f.band((0.06 * hf) as usize, true, 0.25);
      }
      f
    }
    "Straw" => {
      let mut c = Canvas::new(w, h, [0.15; 3]);
      let angle = if variant == 3 { -PI * 0.45 } else { -PI * 0.3 };
      strokes(
        &mut c,
        rng,
        (3500.0 * s * s) as usize,
        angle,
        0.35,
        (40.0 * s, 220.0 * s),
        2.0 * s.max(0.6),
        (0.4, 1.0),
      );
      Field::from_canvas(&c)
    }
    "TileRoof" => {
      // Sky, a roof of curved tiles, and the facade beneath it.
      let mut f = Field::new(w, h, |x, y| {
        let t = y / hf;
        if t < 0.33 {
          0.75 + noise.pfbm(x / 120.0, y / 60.0, 3, 0.5) * 0.05
        } else if t < 0.4 {
          0.45
        } else if t < 0.68 {
          let rows = (t - 0.4) / 0.28 * 12.0;
          let fy = rows.fract();
          let col = ((x / (wf / 40.0)) * PI).sin().abs();
          0.3 + 0.35 * col * (1.0 - 0.5 * fy)
        } else {
          let window = ((x / (wf / 4.0)).fract() - 0.5).abs() < 0.3 && t > 0.76;
          if window {
            0.08
          } else {
            0.55 + noise.perlin(x / 10.0, y / 10.0) * 0.05
          }
        }
      });
      f.band((0.03 * hf) as usize, false, 0.0);
      f
    }
    "Wall" => {
      // Stucco: lumpy relief.
      let height = Field::new(w, h, |x, y| {
        noise.pfbm(x / (26.0 * s), y / (26.0 * s), 5, 0.55)
          + noise2.perlin(x / (6.0 * s), y / (6.0 * s)) * 0.15
      });
      height.relief((0.7, -0.7), 3.5)
    }
    "Water" => {
      // Ripples: short streaks along a slanted direction.
      let a: f64 = if variant == 2 { 1.15 } else { 1.25 };
      let (ca, sa) = (a.cos(), a.sin());
      let height = Field::new(w, h, |x, y| {
        let u = x * ca + y * sa;
        let v = -x * sa + y * ca;
        noise.pfbm(u / (40.0 * s), v / (3.5 * s), 4, 0.5)
          + noise2.perlin(u / (12.0 * s), v / (2.0 * s)) * 0.3
      });
      height.relief((0.0, 1.0), if variant == 2 { 6.0 } else { 2.0 })
    }
    "Wood" => wood(
      w,
      h,
      rng,
      noise,
      noise2,
      &WoodKind {
        cathedral: true,
        pores: true,
        ..WoodKind::default()
      },
    ),
    "WoodFence" => {
      // Planks in perspective (narrower to the left), a wedge of sky above
      // the far end, shrubs in front of it.
      let mut f = Field::new(w, h, |x, y| {
        let t = x / wf;
        let top = hf * 0.16 * (1.0 - t);
        let bottom = hf * (1.0 - 0.12 * (1.0 - t));
        if y < top {
          0.85
        } else if y > bottom
          || (t < 0.4
            && y
              > hf * (0.8 + 0.3 * t) + noise.pfbm(x / 20.0, 1.0, 3, 0.5) * 30.0)
        {
          0.25 + noise.pfbm(x / 6.0, y / 6.0, 3, 0.5) * 0.3
        } else {
          let u = (t * 3.0 + t * t * 4.0) * wf / (24.0 * s);
          let fu = u.fract();
          let plank = if fu < 0.18 {
            0.15
          } else {
            0.45 + 0.1 * (fu * PI).sin()
          };
          plank + noise.perlin(x / 3.0, y / 40.0) * 0.05
        }
      });
      f.band((0.03 * hf) as usize, false, 0.0);
      f
    }
    // Colour textures with a brightness recipe.
    "Amboyna" | "Vavona" => wood(
      w,
      h,
      rng,
      noise,
      noise2,
      &WoodKind {
        vertical: false,
        burl: 1.0,
        ..WoodKind::default()
      },
    ),
    "BurlOak" => wood(
      w,
      h,
      rng,
      noise,
      noise2,
      &WoodKind {
        vertical: false,
        burl: 0.7,
        ..WoodKind::default()
      },
    ),
    "Ash" => wood(
      w,
      h,
      rng,
      noise,
      noise2,
      &WoodKind {
        cathedral: true,
        spacing: 18.0,
        ..WoodKind::default()
      },
    ),
    "GoldenOak" => wood(
      w,
      h,
      rng,
      noise,
      noise2,
      &WoodKind {
        vertical: false,
        spacing: 14.0,
        warp: 5.0,
        ..WoodKind::default()
      },
    ),
    "Kingwood" => wood(
      w,
      h,
      rng,
      noise,
      noise2,
      &WoodKind {
        knot: Some((0.65, 0.72)),
        spacing: 16.0,
        warp: 10.0,
        ..WoodKind::default()
      },
    ),
    "Laurel" => wood(
      w,
      h,
      rng,
      noise,
      noise2,
      &WoodKind {
        spacing: 40.0,
        warp: 2.0,
        ..WoodKind::default()
      },
    ),
    "LightCherry" => wood(
      w,
      h,
      rng,
      noise,
      noise2,
      &WoodKind {
        spacing: 6.0,
        warp: 2.0,
        ..WoodKind::default()
      },
    ),
    "RedApple" => {
      // Soft vertical streaks over a strong top-to-bottom change of shade.
      Field::new(w, h, |x, y| {
        0.72 - 0.45 * y / hf
          + noise.perlin(x / 6.0, y / 120.0) * 0.06
          + noise2.pfbm(x / 40.0, y / 300.0, 3, 0.5) * 0.08
      })
    }
    "GrayMarble" | "YellowMarble" => Field::new(w, h, |x, y| {
      let speck = noise.perlin(x / 2.0, y / 2.0) * 0.25
        + noise2.perlin(x / 6.0, y / 6.0) * 0.15;
      let cloud = noise.pfbm(x / 120.0, y / 120.0, 4, 0.5) * 0.25;
      let vein = noise2.pfbm(x / 60.0 + 5.0, y / 60.0, 4, 0.55).abs();
      let crack = if vein < 0.02 { -0.45 } else { 0.0 };
      0.6 + speck * if base == "GrayMarble" { 1.0 } else { 0.3 } + cloud + crack
    }),
    "WhiteMarble" => Field::new(w, h, |x, y| {
      // Mottling crossed by a fine network of crinkled veins.
      let mottle = noise.pfbm(x / 40.0, y / 40.0, 5, 0.6) * 0.35;
      let net = noise2.pfbm(x / 25.0, y / 25.0, 5, 0.6).abs();
      let vein = if net < 0.03 { -0.5 } else { 0.0 };
      0.6 + mottle + vein
    }),
    "OrangeConcrete" => Field::new(w, h, |x, y| {
      0.5
        + noise.perlin(x / 1.8, y / 1.8) * 0.3
        + noise2.perlin(x / 6.0, y / 6.0) * 0.15
        + noise.pfbm(x / 150.0, y / 150.0, 2, 0.5) * 0.08
    }),
    "StoneWall" => {
      // Irregular rectangular stones in rough courses.
      let row = hf / 9.0;
      Field::new(w, h, |x, y| {
        let r = (y / row + noise.perlin(x / 80.0, 3.0) * 0.15).floor();
        let fy = (y / row + noise.perlin(x / 80.0, 3.0) * 0.15).rem_euclid(1.0);
        let len = row * (1.0 + 0.6 * noise.perlin(r * 1.7, 0.3).abs());
        let sx = x / len + noise.perlin(r * 2.9, 9.1) * 2.0;
        let fx = sx.rem_euclid(1.0);
        let k = sx.floor();
        let gap = fx < 0.05 || fy < 0.06;
        let moss = noise2.pfbm(x / 12.0, y / 12.0, 4, 0.6) * 0.25;
        if gap {
          0.08
        } else {
          0.4 + noise.perlin(k * 2.1, r * 1.3) * 0.15 + moss
        }
      })
    }
    _ => return None,
  })
}

struct WoodKind {
  vertical: bool,
  /// Arched "cathedral" figure instead of straight lines.
  cathedral: bool,
  /// Open pores: short dark dashes along the grain.
  pores: bool,
  /// Centre of a knot (fractions of width and height).
  knot: Option<(f64, f64)>,
  /// Burl: grain swirling around many small eyes.
  burl: f64,
  /// Distance between grain lines (pixels at 512) and how much they wander.
  spacing: f64,
  warp: f64,
}

impl Default for WoodKind {
  fn default() -> Self {
    WoodKind {
      vertical: true,
      cathedral: false,
      pores: false,
      knot: None,
      burl: 0.0,
      spacing: 9.0,
      warp: 6.0,
    }
  }
}

fn wood(
  w: usize,
  h: usize,
  rng: &mut Rng,
  noise: &Noise,
  noise2: &Noise,
  k: &WoodKind,
) -> Field {
  let (wf, hf) = (w as f64, h as f64);
  let s = wf.min(hf) / 512.0;
  let eyes: Vec<(f64, f64)> = (0..(30.0 * k.burl) as usize)
    .map(|_| (rng.range(0.0, wf), rng.range(0.0, hf)))
    .collect();
  let mut f = Field::new(w, h, |x, y| {
    // Coordinates with the grain running along `along`.
    let (along, across) = if k.vertical { (y, x) } else { (x, y) };
    let (span_al, span_ac) = if k.vertical { (hf, wf) } else { (wf, hf) };
    let mut phase = across / (k.spacing * s)
      + noise.pfbm(along / 200.0, across / 200.0, 3, 0.5) * k.warp;
    if k.cathedral {
      // Arches: nested parabolas around the middle of the board.
      let c = across - span_ac * 0.5;
      phase = (c * c / (span_ac * 0.25) - along * 0.6) / (k.spacing * 1.4 * s)
        + noise.pfbm(along / 150.0, across / 150.0, 3, 0.5) * k.warp * 0.5;
    }
    if k.burl > 0.0 {
      // Burl: contours of a lumpy surface instead of parallel lines.
      phase = noise.pfbm(x / (90.0 * s), y / (90.0 * s), 4, 0.55) * 9.0
        + across / (60.0 * s);
    }
    if let Some((kx, ky)) = k.knot {
      let (dx, dy) = (x - kx * wf, y - ky * hf);
      let r = (dx * dx * 4.0 + dy * dy).sqrt();
      phase += 40.0 * (-r / (0.25 * span_al.min(span_ac))).exp();
    }
    for &(ex, ey) in &eyes {
      let r = ((x - ex).powi(2) + (y - ey).powi(2)).sqrt();
      phase += 3.0 * (-r / (25.0 * s)).exp() * k.burl;
    }
    let rings = (phase * PI).sin() * 0.5 + 0.5;
    let rings = if k.burl > 0.0 {
      (rings * 0.6 + noise2.pfbm(x / 25.0, y / 25.0, 3, 0.5) * 0.4)
        .clamp(0.0, 1.0)
    } else {
      rings
    };
    // Pore-wood boards (oak, ash) show mostly fibre, the figure only faintly.
    let figure = if k.pores { 0.08 } else { 0.3 };
    let fiber = noise2.perlin(across / 1.5, along / 25.0)
      * if k.pores { 0.2 } else { 0.12 };
    0.5
      + figure * rings.powf(1.5)
      + fiber
      + noise.pfbm(along / 300.0, across / 120.0, 3, 0.5) * 0.12
  });
  if k.pores {
    let mut c = f.to_canvas();
    let n = (9000.0 * s * s) as usize;
    let angle = if k.vertical { PI / 2.0 } else { 0.0 };
    strokes(
      &mut c,
      rng,
      n,
      angle,
      0.05,
      (3.0 * s, 9.0 * s),
      1.2,
      (0.1, 0.3),
    );
    f = Field::from_canvas(&c);
  }
  f
}

/// Patchwork textures: rectangles each filled with a different texture
/// (Mosaic, 2, 4, 5), a quadtree of flat grays (Mosaic3), or flat regions
/// with curved and straight boundaries (Mosaic6).
fn mosaic(
  variant: usize,
  w: usize,
  h: usize,
  rng: &mut Rng,
  noise: &Noise,
  noise2: &Noise,
) -> Field {
  let (wf, hf) = (w as f64, h as f64);
  match variant {
    3 => Field::new(w, h, |x, y| {
      // Finer squares towards the bottom right.
      let (u, v) = (x / wf, y / hf);
      let level = if v < 0.5 {
        4.0
      } else if u < 0.5 {
        8.0
      } else if v < 0.75 {
        12.0
      } else {
        16.0
      };
      let (i, j) = ((u * level).floor(), (v * level).floor());
      let t = noise.perlin(i * 1.7 + level, j * 2.3);
      (0.5 + t * 0.9).clamp(0.0, 1.0)
    }),
    6 => Field::new(w, h, |x, y| {
      let (u, v) = (x / wf, y / hf);
      let (col, row) = ((u * 3.0).floor(), (v * 3.0).floor());
      let (fu, fv) = (u * 3.0 - col, v * 3.0 - row);
      let tone = |k: f64| 0.15 + 0.7 * ((k * 0.37).fract());
      match (row as i64, col as i64) {
        (0, 0) => {
          if (fu - 1.0).powi(2) + (fv - 1.0).powi(2) < 0.8 {
            tone(1.0)
          } else {
            tone(2.0)
          }
        }
        (0, _) => {
          if (fu - 0.5).abs() < fv / 2.0 {
            tone(3.0 + col)
          } else {
            tone(5.0 + col)
          }
        }
        (1, _) => {
          if fv > 0.5 + 0.35 * (fu * TAU * (col + 1.0)).sin() {
            tone(7.0 + col)
          } else {
            tone(9.0 + col)
          }
        }
        _ => {
          let blob = noise.pfbm(x / 160.0, y / 160.0, 2, 0.5);
          tone((blob * 3.0).floor() + 11.0)
        }
      }
    }),
    _ => {
      // Split into rectangles, each with its own texture.
      let mut rects = vec![(0.0, 0.0, wf, hf)];
      for _ in 0..(6 + variant) {
        rects.sort_by(|a, b| {
          ((b.2 - b.0) * (b.3 - b.1)).total_cmp(&((a.2 - a.0) * (a.3 - a.1)))
        });
        let (x0, y0, x1, y1) = rects.remove(0);
        let t = rng.range(0.35, 0.65);
        if x1 - x0 > y1 - y0 {
          let xm = x0 + (x1 - x0) * t;
          rects.push((x0, y0, xm, y1));
          rects.push((xm, y0, x1, y1));
        } else {
          let ym = y0 + (y1 - y0) * t;
          rects.push((x0, y0, x1, ym));
          rects.push((x0, ym, x1, y1));
        }
      }
      let sources = [
        "Grass",
        "Straw",
        "Raffia",
        "Bark",
        "Herringbone",
        "Leather",
        "Sand",
        "Water",
        "Gravel",
        "Cloth",
      ];
      let mut out = Field::new(w, h, |_, _| 0.5);
      for (i, &(x0, y0, x1, y1)) in rects.iter().enumerate() {
        let src = sources[(i + variant) % sources.len()];
        let mut sub_rng = Rng::new(rng.next_u64());
        let Some(tex) = field(src, w, h, &mut sub_rng, noise2, noise) else {
          continue;
        };
        for y in y0 as usize..(y1 as usize).min(h) {
          for x in x0 as usize..(x1 as usize).min(w) {
            out.v[y * w + x] = tex.v[y * w + x];
          }
        }
      }
      out
    }
  }
}

/// Two-tone patterns: the motif drawn as a mask (1 = motif) on a
/// background, returned as a brightness field. `light_motif` motifs are
/// brighter than their background.
#[allow(clippy::too_many_lines)]
fn pattern(name: &str, w: usize, h: usize, rng: &mut Rng) -> Option<Field> {
  let (wf, hf) = (w as f64, h as f64);
  let s = wf.min(hf) / 400.0;
  let mut c = Canvas::new(w, h, BLACK);
  let ink = Paint::Solid(WHITE);
  let mut light_motif = false;
  let leaf = |c: &mut Canvas, x: f64, y: f64, len: f64, a: f64| {
    c.ellipse(
      x + a.cos() * len / 2.0,
      y + a.sin() * len / 2.0,
      len / 2.0,
      len / 5.0,
      a,
      &ink,
    );
  };
  match name {
    "BerriesPattern" => {
      for _ in 0..(14.0 * s * s).max(6.0) as usize {
        let (x, y) = (rng.range(0.0, wf), rng.range(0.0, hf));
        let a = rng.range(0.0, TAU);
        let stem = bezier(
          (x, y),
          (x + 30.0 * s * a.cos(), y + 10.0 * s),
          (x + 50.0 * s * a.cos(), y - 20.0 * s * a.sin()),
          (x + 70.0 * s * a.cos(), y + 70.0 * s * a.sin()),
          20,
        );
        c.polyline(&stem, 2.0 * s, &ink);
        for (k, p) in stem.iter().enumerate().step_by(3) {
          leaf(
            &mut c,
            p.0,
            p.1,
            22.0 * s,
            a + if k % 2 == 0 { 0.9 } else { -0.9 },
          );
        }
        for k in 0..4 {
          c.circle(
            x + rng.range(-8.0, 8.0) * s,
            y + rng.range(-8.0, 8.0) * s - k as f64,
            3.5 * s,
            &ink,
          );
        }
      }
    }
    "DelisPattern" => {
      // Damask: mirrored lobes and teardrops on a staggered lattice.
      let cell = wf / 2.2;
      for j in -1..(hf / cell) as i64 + 2 {
        for i in -1..(wf / cell) as i64 + 2 {
          let (cx, cy) = (
            (i as f64 + if j % 2 == 0 { 0.0 } else { 0.5 }) * cell,
            j as f64 * cell * 0.9,
          );
          for side in [-1.0, 1.0] {
            c.circle(
              cx + side * cell * 0.18,
              cy - cell * 0.22,
              cell * 0.14,
              &ink,
            );
            c.circle(
              cx + side * cell * 0.32,
              cy + cell * 0.02,
              cell * 0.11,
              &ink,
            );
          }
          c.ellipse(cx, cy + cell * 0.1, cell * 0.12, cell * 0.28, 0.0, &ink);
          c.polygon(
            &[
              (cx - cell * 0.4, cy - cell * 0.05),
              (cx + cell * 0.4, cy - cell * 0.05),
              (cx, cy + cell * 0.45),
            ],
            &ink,
          );
          c.circle(cx, cy - cell * 0.42, cell * 0.06, &ink);
        }
      }
    }
    "DiamondsPattern" => {
      light_motif = true;
      let cell = wf / 3.2;
      c.map(|x, y, _| {
        let u = (x / cell + y / (cell * 1.7)).rem_euclid(1.0);
        let v = (x / cell - y / (cell * 1.7)).rem_euclid(1.0);
        let line = |t: f64| (t - 0.5).abs() > 0.43 || (t - 0.5).abs() < 0.05;
        if line(u) || line(v) { WHITE } else { BLACK }
      });
    }
    "FleurPattern" => {
      let step = 75.0 * s;
      let s = s * 2.0;
      let mut j = 0;
      let mut y = step / 2.0;
      while y < hf + step {
        let mut x = if j % 2 == 0 { step / 2.0 } else { step * 1.5 };
        while x < wf + step {
          c.ellipse(x, y - 5.0 * s, 3.0 * s, 7.0 * s, 0.0, &ink);
          c.ellipse(x - 7.0 * s, y - 1.0 * s, 3.0 * s, 6.0 * s, 0.7, &ink);
          c.ellipse(x + 7.0 * s, y - 1.0 * s, 3.0 * s, 6.0 * s, -0.7, &ink);
          x += step * 2.0;
        }
        y += step * 0.75;
        j += 1;
      }
    }
    "FloralPattern1" => {
      let cell = wf / 2.0;
      for j in -1..(hf / cell) as i64 + 2 {
        for i in -1..3 {
          let (cx, cy) = (
            (i as f64 + if j % 2 == 0 { 0.0 } else { 0.5 }) * cell,
            j as f64 * cell,
          );
          // A cup-shaped flower on a curved stem.
          c.polygon(
            &[
              (cx - 30.0 * s, cy - 12.0 * s),
              (cx + 30.0 * s, cy - 12.0 * s),
              (cx + 18.0 * s, cy + 12.0 * s),
              (cx - 18.0 * s, cy + 12.0 * s),
            ],
            &ink,
          );
          for k in 0..5 {
            let px = cx - 24.0 * s + k as f64 * 12.0 * s;
            c.polygon(
              &[
                (px - 4.0 * s, cy - 12.0 * s),
                (px + 4.0 * s, cy - 12.0 * s),
                (px, cy - 22.0 * s),
              ],
              &ink,
            );
          }
          let stem = bezier(
            (cx, cy + 12.0 * s),
            (cx + 25.0 * s, cy + 60.0 * s),
            (cx - 25.0 * s, cy + 100.0 * s),
            (cx + 10.0 * s, cy + 150.0 * s),
            30,
          );
          c.polyline(&stem, 5.0 * s, &ink);
          c.ellipse(
            cx + 35.0 * s,
            cy + 95.0 * s,
            10.0 * s,
            18.0 * s,
            0.3,
            &ink,
          );
        }
      }
    }
    "FloralPattern2" => {
      for k in 0..3 {
        let x = wf * (0.2 + 0.3 * k as f64);
        c.line((x, 0.0), (x, hf * 0.8), 2.5 * s, &ink);
        c.line(
          (x + 12.0 * s, hf * 0.1),
          (x + 12.0 * s, hf * 0.9),
          2.5 * s,
          &ink,
        );
        let curl = bezier(
          (x, hf * 0.8),
          (x - 20.0 * s, hf * 0.9),
          (x - 40.0 * s, hf * 0.8),
          (x - 30.0 * s, hf * 0.75),
          20,
        );
        c.polyline(&curl, 2.5 * s, &ink);
        let (bx, by) = (x + 25.0 * s, hf * (0.2 + 0.25 * (k % 2) as f64));
        c.polygon(
          &[
            (bx, by),
            (bx + 30.0 * s, by - 20.0 * s),
            (bx + 40.0 * s, by + 10.0 * s),
            (bx + 20.0 * s, by + 25.0 * s),
          ],
          &ink,
        );
      }
    }
    "FloralPattern3" | "LeavesPattern" | "FrogsPattern" | "PaisleyPattern" => {
      let n = match name {
        "LeavesPattern" => 7,
        "FrogsPattern" => 6,
        "PaisleyPattern" => 10,
        _ => 30,
      };
      for _ in 0..n {
        let (x, y) = (rng.range(0.0, wf), rng.range(0.0, hf));
        let a = rng.range(0.0, TAU);
        let r = rng.range(18.0, 28.0) * s;
        match name {
          "LeavesPattern" => {
            let r = r * 1.6;
            c.polygon(&star(x, y, r * 1.4, Some(r * 0.75), 5, a), &ink);
            for k in 0..5 {
              let b = a + TAU * k as f64 / 5.0;
              c.circle(
                x + r * 0.9 * b.cos(),
                y + r * 0.9 * b.sin(),
                r * 0.35,
                &ink,
              );
            }
            c.line(
              (x, y),
              (x + r * 2.0 * a.cos(), y + r * 2.0 * a.sin()),
              2.0 * s,
              &ink,
            );
          }
          "FrogsPattern" => {
            // Orbits (an atom) and a frog: body, head, legs.
            let r = r * 1.4;
            for k in 0..3 {
              let b = a + PI * k as f64 / 3.0;
              let orbit: Vec<(f64, f64)> = (0..=60)
                .map(|i| {
                  let t = TAU * i as f64 / 60.0;
                  let (ex, ey) = (r * 2.2 * t.cos(), r * 0.8 * t.sin());
                  (
                    x + ex * b.cos() - ey * b.sin(),
                    y + ex * b.sin() + ey * b.cos(),
                  )
                })
                .collect();
              c.polyline(&orbit, 2.0 * s, &ink);
            }
            c.ellipse(x, y, r * 0.9, r * 0.55, a, &ink);
            c.circle(
              x + r * 0.9 * a.cos(),
              y + r * 0.9 * a.sin(),
              r * 0.35,
              &ink,
            );
            for side in [-1.0, 1.0] {
              let b = a + side * 2.3;
              c.line(
                (x, y),
                (x + r * 1.2 * b.cos(), y + r * 1.2 * b.sin()),
                3.0 * s,
                &ink,
              );
              let b2 = a + side * 0.9;
              c.line(
                (x, y),
                (x + r * 1.0 * b2.cos(), y + r * 1.0 * b2.sin()),
                3.0 * s,
                &ink,
              );
            }
          }
          "PaisleyPattern" => {
            // A teardrop with a curled tip, outlined, with a dot inside.
            let r = r * 1.6;
            let shape = |c: &mut Canvas, k: f64, paint: &Paint| {
              c.circle(x, y, r * 0.6 * k, paint);
              let tail = bezier(
                (x - r * 0.45, y),
                (x - r * 0.5, y - r * 1.3),
                (x + r * 0.4, y - r * 1.5),
                (x + r * 0.7, y - r * 1.1),
                16,
              );
              c.polyline(&tail, r * 0.35 * k, paint);
            };
            shape(&mut c, 1.0, &ink);
            shape(&mut c, 0.7, &Paint::Solid(BLACK));
            c.circle(x, y, r * 0.15, &ink);
          }
          _ => {
            // Vine with leaves and trumpet flowers.
            let vine = bezier(
              (x, y),
              (x + 40.0 * s, y - 30.0 * s),
              (x + 60.0 * s, y + 20.0 * s),
              (x + 100.0 * s, y - 10.0 * s),
              20,
            );
            c.polyline(&vine, 1.5 * s, &ink);
            leaf(&mut c, vine[6].0, vine[6].1, 22.0 * s, a);
            c.polygon(
              &star(vine[14].0, vine[14].1, r * 0.7, Some(r * 0.4), 5, a),
              &ink,
            );
          }
        }
      }
    }
    "GloriesPattern" => return None,
    "MultiSpiralsPattern" => {
      for _ in 0..(wf * hf / (60.0 * s).powi(2)) as usize + 8 {
        let (cx, cy) = (rng.range(0.0, wf), rng.range(0.0, hf));
        let r = rng.range(40.0, 70.0) * s;
        // Each spiral sits on a disc of background, covering what is below.
        c.circle(cx, cy, r, &Paint::Solid(BLACK));
        let pts: Vec<(f64, f64)> = (0..200)
          .map(|i| {
            let t = i as f64 / 200.0;
            let a = t * 3.0 * TAU;
            (cx + r * t * a.cos(), cy + r * t * a.sin())
          })
          .collect();
        c.polyline(&pts, 6.0 * s, &ink);
        c.ring(cx, cy, r, 6.0 * s, &ink);
      }
    }
    "ScallopPattern" => {
      // Rows of arches, each with an inner arch, overlapping like scales.
      let r = wf / 6.0;
      let arc = |c: &mut Canvas, cx: f64, cy: f64, rr: f64, width: f64| {
        let pts: Vec<(f64, f64)> = (0..=40)
          .map(|i| {
            let a = PI + PI * i as f64 / 40.0;
            (cx + rr * a.cos(), cy + rr * a.sin())
          })
          .collect();
        c.polyline(&pts, width, &ink);
      };
      // Top rows first: each lower row covers the feet of the arches above.
      let mut row = 0;
      let mut y = -r;
      while y < hf + 2.0 * r {
        let shift = if row % 2 == 0 { 0.0 } else { r };
        let mut x = -r + shift;
        while x < wf + r {
          // Clear the arch, then draw its outline and inner arch.
          c.ellipse(x, y, r, r, 0.0, &Paint::Solid(BLACK));
          arc(&mut c, x, y, r * 0.95, 8.0 * s);
          arc(&mut c, x, y, r * 0.62, 5.0 * s);
          x += 2.0 * r;
        }
        y += r * 0.75;
        row += 1;
      }
    }
    "TriWeavePattern" => {
      let r = wf / 4.0;
      for j in -1..(hf / r) as i64 + 2 {
        for i in -1..(wf / r) as i64 + 2 {
          let (x, y) = (
            (i as f64 + if j % 2 == 0 { 0.0 } else { 0.5 }) * r,
            j as f64 * r * 0.87,
          );
          c.ring(x, y, r * 0.5, 4.0 * s, &ink);
        }
      }
    }
    "VinesPattern" => {
      for _ in 0..(wf * hf / (40.0 * s).powi(2)) as usize + 20 {
        let (x, y) = (rng.range(0.0, wf), rng.range(0.0, hf));
        let a = rng.range(0.0, TAU);
        let curl: Vec<(f64, f64)> = (0..40)
          .map(|i| {
            let t = i as f64 / 40.0;
            let r = 18.0 * s * (1.0 - t * 0.7);
            (
              x + 40.0 * s * t * a.cos() + r * (t * 9.0).cos() * 0.4,
              y + 40.0 * s * t * a.sin() + r * (t * 9.0).sin() * 0.4,
            )
          })
          .collect();
        c.polyline(&curl, 1.5 * s, &ink);
        leaf(&mut c, curl[20].0, curl[20].1, 8.0 * s, a + 1.2);
      }
    }
    "WavesPattern" => {
      let period = wf / 2.2;
      let rows = (hf / (period * 0.6)).ceil() as i64 + 1;
      for j in 0..rows {
        let y0 = j as f64 * period * 0.6 + period * 0.2;
        let shift = if j % 2 == 0 { 0.0 } else { period / 2.0 };
        let wave: Vec<(f64, f64)> = (0..=400)
          .map(|i| {
            let x = -period + i as f64 / 400.0 * (wf + 2.0 * period);
            (x, y0 + 0.12 * period * ((x - shift) / period * TAU).sin())
          })
          .collect();
        c.polyline(&wave, 6.0 * s, &ink);
        let mut x = shift - period;
        while x < wf + period {
          let curl: Vec<(f64, f64)> = (0..60)
            .map(|i| {
              let t = i as f64 / 60.0;
              let a = PI + t * 1.6 * TAU;
              let r = 0.13 * period * (1.0 - 0.7 * t);
              (
                x + period * 0.3 + r * a.cos(),
                y0 - 0.12 * period + r * a.sin(),
              )
            })
            .collect();
          c.polyline(&curl, 6.0 * s, &ink);
          x += period;
        }
      }
    }
    "WheatPattern" => {
      let step = wf / 6.0;
      for k in 0..7 {
        let x = step * (k as f64 + 0.5);
        let (y0, y1) = if k % 2 == 0 {
          (0.0, hf)
        } else {
          (hf * 0.15, hf * 0.85)
        };
        c.line((x, y0), (x, y1), 2.5 * s, &ink);
        if k % 2 == 1 {
          let mut y = y0 + 10.0 * s;
          while y < y0 + (y1 - y0) * 0.6 {
            leaf(&mut c, x, y, 16.0 * s, -PI / 2.0 - 0.6);
            leaf(&mut c, x, y, 16.0 * s, -PI / 2.0 + 0.6);
            y += 18.0 * s;
          }
        }
      }
    }
    _ => return None,
  }
  let mut f = Field::from_canvas(&c);
  if !light_motif {
    f.map(|v| 1.0 - v);
  }
  Some(f)
}

/// Multi-coloured entries painted in colour.
fn painted(
  name: &str,
  w: usize,
  h: usize,
  rng: &mut Rng,
  noise: &Noise,
  noise2: &Noise,
) -> Option<Canvas> {
  let (wf, hf) = (w as f64, h as f64);
  let s = wf.min(hf) / 400.0;
  let mut c = Canvas::new(w, h, BLACK);
  match name {
    "GloriesPattern" => {
      c.fill(&Paint::Solid(rgb8(80, 170, 70)));
      for _ in 0..(wf * hf / (85.0 * s).powi(2)) as usize + 8 {
        let (x, y) = (rng.range(0.0, wf), rng.range(0.0, hf));
        let r = rng.range(16.0, 22.0) * s;
        if rng.chance(0.5) {
          c.polygon(
            &star(x, y, r, Some(r * 0.5), 5, rng.range(0.0, TAU)),
            &Paint::Solid(rgb8(80, 50, 140)),
          );
          c.circle(x, y, r * 0.3, &Paint::Solid(rgb8(20, 10, 40)));
        } else {
          for k in 0..12 {
            let a = TAU * k as f64 / 12.0;
            c.ellipse(
              x + r * 0.6 * a.cos(),
              y + r * 0.6 * a.sin(),
              r * 0.5,
              r * 0.15,
              a,
              &Paint::Solid(WHITE),
            );
          }
          c.circle(x, y, r * 0.25, &Paint::Solid(rgb8(230, 200, 40)));
        }
      }
    }
    "CatFur" => {
      c.fill(&Paint::Fn(&|x, y| {
        let stripe = ((x * 0.6 + y) / (40.0 * s)
          + noise.pfbm(x / 60.0, y / 60.0, 3, 0.5) * 2.0)
          .sin();
        let g = 0.5 + 0.2 * stripe;
        [g, g * 0.9, g * 0.75]
      }));
      strokes(
        &mut c,
        rng,
        (6000.0 * s * s) as usize,
        -0.9,
        0.3,
        (6.0 * s, 16.0 * s),
        1.0,
        (0.1, 0.9),
      );
    }
    "CheetahFur" => {
      c.fill(&Paint::Fn(&|x, y| {
        let g = 0.75 + noise.pfbm(x / 5.0, y / 20.0, 3, 0.5) * 0.15;
        [g, g * 0.85, g * 0.6]
      }));
      // Rosettes: rings of dark blobs.
      for _ in 0..(wf * hf / (62.0 * s).powi(2)) as usize + 10 {
        let (x, y) = (rng.range(0.0, wf), rng.range(0.0, hf));
        let r = rng.range(14.0, 24.0) * s;
        // A rosette: broken ring of blotches around a slightly darker centre.
        c.ellipse(
          x,
          y,
          r * 0.8,
          r * 0.7,
          0.0,
          &Paint::Solid(rgb8(185, 150, 100)),
        );
        for k in 0..rng.int_in(3, 5) {
          let a = TAU * k as f64 / 4.0 + rng.range(-0.4, 0.4);
          c.ellipse(
            x + r * a.cos(),
            y + r * a.sin(),
            r * 0.55,
            r * 0.3,
            a + PI / 2.0,
            &Paint::Solid(rgb8(30, 20, 15)),
          );
        }
      }
    }
    "GiraffeFur" => {
      let cells = Cells::new(
        rng,
        w,
        h,
        (wf * hf / (75.0 * s).powi(2)) as usize + 12,
        1.0,
      );
      c.fill(&Paint::Fn(&|x, y| {
        let (d1, d2, k) = cells.query(
          x + noise.perlin(x / 20.0, y / 20.0) * 8.0,
          y + noise2.perlin(x / 20.0, y / 20.0) * 8.0,
        );
        if d2 - d1 < 9.0 * s {
          rgb8(220, 210, 195)
        } else {
          shade(
            rgb8(120, 60, 30),
            0.7 + 0.5 * cells.tone(k) + noise.perlin(x / 4.0, y / 4.0) * 0.1,
          )
        }
      }));
    }
    "Metal1" => {
      c.fill(&Paint::Fn(&|x, y| {
        let rust = noise.pfbm(x / 160.0, y / 160.0, 6, 0.6)
          + 0.15 * (1.0 - y / hf)
          - 0.1;
        let flake = noise2.pfbm(x / 70.0, y / 90.0, 5, 0.6);
        if rust > 0.25 {
          rgb8(170, 80, 30)
        } else if flake > 0.2 {
          rgb8(230, 225, 215)
        } else {
          shade(rgb8(130, 145, 160), 1.0 + flake * 0.3)
        }
      }));
    }
    "Metal2" | "Metal3" | "Metal4" => {
      // Diamond plate: raised lozenges alternating direction cell by cell.
      let cell = 45.0 * s;
      let base = match name {
        "Metal2" => rgb8(140, 110, 90),
        "Metal3" => rgb8(100, 120, 110),
        _ => rgb8(150, 150, 150),
      };
      c.fill(&Paint::Fn(&|x, y| {
        let (u, v) = ((x + y) / cell, (x - y) / cell);
        let (i, j) = (u.floor(), v.floor());
        let (fu, fv) = (u - i - 0.5, v - j - 0.5);
        let horizontal = (i + j) as i64 % 2 == 0;
        let (a, b) = if horizontal { (fu, fv) } else { (fv, fu) };
        let lozenge = if name == "Metal4" {
          // Groups of slender parallel bars.
          (-0.3..0.3).contains(&b)
            && ((b + 0.3) / 0.15).fract() < 0.55
            && a.abs() < 0.42 - 0.6 * (((b + 0.3) / 0.15).fract() - 0.27).abs()
        } else {
          (a / 0.42).powi(2) + (b / 0.13).powi(2) < 1.0
        };
        let grime = noise.pfbm(x / 30.0, y / 30.0, 4, 0.6) * 0.3;
        if lozenge {
          shade(base, 1.35 + grime - 0.6 * b)
        } else {
          shade(base, 0.85 + grime)
        }
      }));
    }
    "GreenMarble" => {
      c.fill(&Paint::Fn(&|x, y| {
        let mottle = noise.pfbm(x / 110.0, y / 110.0, 5, 0.6);
        let net = noise2.pfbm(x / 45.0, y / 45.0, 5, 0.6).abs();
        let base = if mottle > 0.0 {
          rgb8(70, 120, 50)
        } else {
          rgb8(150, 120, 100)
        };
        if net < 0.035 {
          rgb8(90, 50, 40)
        } else {
          shade(base, 1.0 + noise.perlin(x / 4.0, y / 4.0) * 0.15)
        }
      }));
    }
    "Roof" => {
      // Rows of shingles with rounded bottoms, staggered and varied.
      let row = hf / 10.0;
      let width = row * 1.3;
      c.fill(&Paint::Solid(rgb8(40, 25, 20)));
      // Bottom rows first: each row overlaps the top of the row below.
      for r in (-1..12).rev() {
        let y0 = r as f64 * row;
        let shift = if r % 2 == 0 { 0.0 } else { width / 2.0 };
        let mut x = -width + shift;
        while x < wf + width {
          let tone = shade(rgb8(140, 75, 50), rng.range(0.6, 1.3));
          let poly: Vec<(f64, f64)> = [(x + 2.0, y0)]
            .into_iter()
            .chain((0..=10).map(|i| {
              // Left to right along the rounded bottom edge.
              let a = PI - PI * i as f64 / 10.0;
              (
                x + width / 2.0 + (width / 2.0 - 2.0) * a.cos(),
                y0 + row * 1.25 + row * 0.25 * a.sin(),
              )
            }))
            .chain([(x + width - 2.0, y0)])
            .collect();
          c.polygon(
            &poly,
            &Paint::Fn(&|px, py| {
              shade(tone, 1.0 + noise.perlin(px / 3.0, py / 12.0) * 0.15)
            }),
          );
          x += width;
        }
      }
    }
    _ => return None,
  }
  Some(c)
}

/// The texture for `name` at `w × h` pixels.
pub fn texture(
  name: &str,
  w: usize,
  h: usize,
  seed: u64,
  rng: &mut Rng,
) -> Canvas {
  let noise = Noise::new(seed, 0);
  let noise2 = Noise::new(seed ^ 0x9e37_79b9, 0);
  if let Some(c) = painted(name, w, h, rng, &noise, &noise2) {
    return c;
  }
  if let Some(f) = pattern(name, w, h, rng) {
    return f.to_canvas();
  }
  if let Some(f) = field(name, w, h, rng, &noise, &noise2) {
    return f.to_canvas();
  }
  Field::new(w, h, |x, y| {
    0.5 + 0.4 * noise.pfbm(x / 40.0, y / 40.0, 6, 0.55)
  })
  .to_canvas()
}
