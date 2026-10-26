//! A small 2D rasterizer for generated example images.
//!
//! Shapes are described by signed distance functions, which gives
//! anti-aliased edges for free: a pixel's coverage is
//! `clamp(0.5 - distance, 0, 1)`. Everything is drawn in RGB with values in
//! `[0, 1]`; [`Canvas::into_image`] converts to the requested channel count
//! and quantizes to the image type, producing an `Expr::Image` directly.

use super::rng::{Noise, Rng};
use crate::syntax::{Expr, ImageType};

pub type Rgb = [f64; 3];

pub const BLACK: Rgb = [0.0, 0.0, 0.0];
pub const WHITE: Rgb = [1.0, 1.0, 1.0];

/// Linear interpolation between two colours.
pub fn mix(a: Rgb, b: Rgb, t: f64) -> Rgb {
  let t = t.clamp(0.0, 1.0);
  [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ]
}

/// Scale a colour's brightness.
pub fn shade(c: Rgb, f: f64) -> Rgb {
  [
    (c[0] * f).clamp(0.0, 1.0),
    (c[1] * f).clamp(0.0, 1.0),
    (c[2] * f).clamp(0.0, 1.0),
  ]
}

/// A colour from 8-bit components.
pub const fn rgb8(r: u8, g: u8, b: u8) -> Rgb {
  [r as f64 / 255.0, g as f64 / 255.0, b as f64 / 255.0]
}

/// Rec. 601 luma, the grayscale conversion Wolfram uses.
pub fn luma(c: Rgb) -> f64 {
  0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
}

/// How a shape is filled.
pub enum Paint<'a> {
  Solid(Rgb),
  /// A gradient from `c0` at `p0` to `c1` at `p1`.
  Linear {
    p0: (f64, f64),
    c0: Rgb,
    p1: (f64, f64),
    c1: Rgb,
  },
  /// A radial gradient from `c0` at the centre to `c1` at `radius`.
  Radial {
    center: (f64, f64),
    radius: f64,
    c0: Rgb,
    c1: Rgb,
  },
  /// An arbitrary colour per pixel.
  Fn(&'a dyn Fn(f64, f64) -> Rgb),
}

impl Paint<'_> {
  fn at(&self, x: f64, y: f64) -> Rgb {
    match self {
      Paint::Solid(c) => *c,
      Paint::Linear { p0, c0, p1, c1 } => {
        let (dx, dy) = (p1.0 - p0.0, p1.1 - p0.1);
        let len2 = (dx * dx + dy * dy).max(1e-12);
        let t = ((x - p0.0) * dx + (y - p0.1) * dy) / len2;
        mix(*c0, *c1, t)
      }
      Paint::Radial {
        center,
        radius,
        c0,
        c1,
      } => {
        let d = ((x - center.0).powi(2) + (y - center.1).powi(2)).sqrt();
        mix(*c0, *c1, d / radius.max(1e-12))
      }
      Paint::Fn(f) => f(x, y),
    }
  }
}

/// An RGB drawing surface; `(0, 0)` is the top-left corner, `x` grows to
/// the right and `y` downwards, pixel centres sit at half-integers.
pub struct Canvas {
  pub width: usize,
  pub height: usize,
  data: Vec<f64>,
}

impl Canvas {
  pub fn new(width: usize, height: usize, background: Rgb) -> Self {
    let mut data = Vec::with_capacity(width * height * 3);
    for _ in 0..width * height {
      data.extend_from_slice(&background);
    }
    Canvas {
      width,
      height,
      data,
    }
  }

  pub fn get(&self, x: usize, y: usize) -> Rgb {
    let i = (y * self.width + x) * 3;
    [self.data[i], self.data[i + 1], self.data[i + 2]]
  }

  pub fn set(&mut self, x: usize, y: usize, c: Rgb) {
    let i = (y * self.width + x) * 3;
    self.data[i..i + 3].copy_from_slice(&c);
  }

  /// Blend `c` over pixel `(x, y)` with opacity `a`.
  pub fn blend(&mut self, x: usize, y: usize, c: Rgb, a: f64) {
    if a <= 0.0 {
      return;
    }
    let a = a.min(1.0);
    let i = (y * self.width + x) * 3;
    for k in 0..3 {
      self.data[i + k] += (c[k] - self.data[i + k]) * a;
    }
  }

  /// Fill everything with `paint`.
  pub fn fill(&mut self, paint: &Paint) {
    for y in 0..self.height {
      for x in 0..self.width {
        let c = paint.at(x as f64 + 0.5, y as f64 + 0.5);
        self.set(x, y, c);
      }
    }
  }

  /// Rewrite every pixel: `f(x, y, colour) -> colour`.
  pub fn map(&mut self, f: impl Fn(f64, f64, Rgb) -> Rgb) {
    for y in 0..self.height {
      for x in 0..self.width {
        let c = f(x as f64 + 0.5, y as f64 + 0.5, self.get(x, y));
        self.set(x, y, c);
      }
    }
  }

  /// Fill the region where `sdf < 0` (signed distance in pixels), limited
  /// to the bounding box `[x0, x1] × [y0, y1]`, with anti-aliased edges.
  pub fn fill_sdf(
    &mut self,
    bbox: (f64, f64, f64, f64),
    sdf: impl Fn(f64, f64) -> f64,
    paint: &Paint,
    opacity: f64,
  ) {
    let (x0, y0, x1, y1) = bbox;
    let xa = (x0 - 1.0).floor().max(0.0) as usize;
    let ya = (y0 - 1.0).floor().max(0.0) as usize;
    let xb = ((x1 + 1.0).ceil().max(0.0) as usize).min(self.width);
    let yb = ((y1 + 1.0).ceil().max(0.0) as usize).min(self.height);
    for y in ya..yb {
      for x in xa..xb {
        let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
        let coverage = (0.5 - sdf(px, py)).clamp(0.0, 1.0);
        if coverage > 0.0 {
          self.blend(x, y, paint.at(px, py), coverage * opacity);
        }
      }
    }
  }

  /// A filled ellipse with semi-axes `rx`, `ry`, rotated by `angle`
  /// radians.
  pub fn ellipse(
    &mut self,
    cx: f64,
    cy: f64,
    rx: f64,
    ry: f64,
    angle: f64,
    paint: &Paint,
  ) {
    let r = rx.max(ry);
    let (s, c) = angle.sin_cos();
    let (rx, ry) = (rx.max(0.3), ry.max(0.3));
    self.fill_sdf(
      (cx - r, cy - r, cx + r, cy + r),
      |x, y| {
        let (dx, dy) = (x - cx, y - cy);
        let (u, v) = (dx * c + dy * s, -dx * s + dy * c);
        // First-order distance approximation to the ellipse boundary.
        let k = ((u / rx).powi(2) + (v / ry).powi(2)).sqrt();
        let grad = ((u / (rx * rx)).powi(2) + (v / (ry * ry)).powi(2)).sqrt();
        if grad < 1e-12 {
          -rx.min(ry)
        } else {
          (k - 1.0) * k / grad
        }
      },
      paint,
      1.0,
    );
  }

  pub fn circle(&mut self, cx: f64, cy: f64, r: f64, paint: &Paint) {
    let r = r.max(0.3);
    self.fill_sdf(
      (cx - r, cy - r, cx + r, cy + r),
      |x, y| ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() - r,
      paint,
      1.0,
    );
  }

  /// A ring (annulus) of radius `r` and stroke width `w`.
  pub fn ring(&mut self, cx: f64, cy: f64, r: f64, w: f64, paint: &Paint) {
    let half = (w / 2.0).max(0.3);
    let o = r + half;
    self.fill_sdf(
      (cx - o, cy - o, cx + o, cy + o),
      |x, y| (((x - cx).powi(2) + (y - cy).powi(2)).sqrt() - r).abs() - half,
      paint,
      1.0,
    );
  }

  /// An axis-aligned rectangle.
  pub fn rect(&mut self, x0: f64, y0: f64, x1: f64, y1: f64, paint: &Paint) {
    let (cx, cy) = (f64::midpoint(x0, x1), f64::midpoint(y0, y1));
    let (hx, hy) = ((x1 - x0).abs() / 2.0, (y1 - y0).abs() / 2.0);
    self.fill_sdf(
      (x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1)),
      |x, y| {
        let dx = (x - cx).abs() - hx;
        let dy = (y - cy).abs() - hy;
        let outside = (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt();
        outside + dx.max(dy).min(0.0)
      },
      paint,
      1.0,
    );
  }

  /// A filled polygon (even-odd rule).
  pub fn polygon(&mut self, points: &[(f64, f64)], paint: &Paint) {
    self.polygon_alpha(points, paint, 1.0);
  }

  pub fn polygon_alpha(
    &mut self,
    points: &[(f64, f64)],
    paint: &Paint,
    opacity: f64,
  ) {
    if points.len() < 3 {
      return;
    }
    let (mut x0, mut y0, mut x1, mut y1) =
      (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for &(x, y) in points {
      x0 = x0.min(x);
      y0 = y0.min(y);
      x1 = x1.max(x);
      y1 = y1.max(y);
    }
    self.fill_sdf(
      (x0, y0, x1, y1),
      |x, y| polygon_sdf(points, x, y),
      paint,
      opacity,
    );
  }

  /// A stroked polyline of width `w` with round joins and caps.
  pub fn polyline(&mut self, points: &[(f64, f64)], w: f64, paint: &Paint) {
    let half = (w / 2.0).max(0.35);
    for seg in points.windows(2) {
      let (a, b) = (seg[0], seg[1]);
      self.fill_sdf(
        (
          a.0.min(b.0) - half,
          a.1.min(b.1) - half,
          a.0.max(b.0) + half,
          a.1.max(b.1) + half,
        ),
        |x, y| segment_distance(a, b, x, y) - half,
        paint,
        1.0,
      );
    }
  }

  pub fn line(&mut self, a: (f64, f64), b: (f64, f64), w: f64, paint: &Paint) {
    self.polyline(&[a, b], w, paint);
  }

  /// Add value noise to every pixel: `amp` is the amplitude, `scale` the
  /// feature size in pixels.
  pub fn grain(&mut self, noise: &Noise, scale: f64, amp: f64) {
    self.map(|x, y, c| {
      let n = noise.fbm(x / scale, y / scale, 4, 0.5) * amp;
      [
        (c[0] + n).clamp(0.0, 1.0),
        (c[1] + n).clamp(0.0, 1.0),
        (c[2] + n).clamp(0.0, 1.0),
      ]
    });
  }

  /// Add per-pixel sensor noise of standard deviation `sd`.
  pub fn speckle(&mut self, rng: &mut Rng, sd: f64) {
    for v in &mut self.data {
      *v = (*v + rng.normal() * sd).clamp(0.0, 1.0);
    }
  }

  /// Box blur of radius `r` (separable: rows, then columns).
  pub fn blur(&mut self, r: usize) {
    if r == 0 {
      return;
    }
    let (w, h) = (self.width, self.height);
    let mut tmp = vec![0.0; self.data.len()];
    for y in 0..h {
      for x in 0..w {
        let (lo, hi) = (x.saturating_sub(r), (x + r + 1).min(w));
        for c in 0..3 {
          let sum: f64 = (lo..hi).map(|k| self.data[(y * w + k) * 3 + c]).sum();
          tmp[(y * w + x) * 3 + c] = sum / (hi - lo) as f64;
        }
      }
    }
    for y in 0..h {
      let (lo, hi) = (y.saturating_sub(r), (y + r + 1).min(h));
      for x in 0..w {
        for c in 0..3 {
          let sum: f64 = (lo..hi).map(|k| tmp[(k * w + x) * 3 + c]).sum();
          self.data[(y * w + x) * 3 + c] = sum / (hi - lo) as f64;
        }
      }
    }
  }

  /// Resample to `width × height` (bilinear).
  #[must_use]
  pub fn resized(&self, width: usize, height: usize) -> Canvas {
    let mut out = Canvas::new(width, height, BLACK);
    let sx = self.width as f64 / width as f64;
    let sy = self.height as f64 / height as f64;
    for y in 0..height {
      for x in 0..width {
        let fx = ((x as f64 + 0.5) * sx - 0.5).max(0.0);
        let fy = ((y as f64 + 0.5) * sy - 0.5).max(0.0);
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) =
          ((x0 + 1).min(self.width - 1), (y0 + 1).min(self.height - 1));
        let (tx, ty) = (fx - x0 as f64, fy - y0 as f64);
        let top = mix(self.get(x0, y0), self.get(x1, y0), tx);
        let bottom = mix(self.get(x0, y1), self.get(x1, y1), tx);
        out.set(x, y, mix(top, bottom, ty));
      }
    }
    out
  }

  /// Mean of the luma (grayscale) values.
  pub fn mean_luma(&self) -> f64 {
    let n = (self.width * self.height).max(1);
    self
      .data
      .chunks(3)
      .map(|c| luma([c[0], c[1], c[2]]))
      .sum::<f64>()
      / n as f64
  }

  /// Linearly rescale every channel so that the image has the given mean
  /// and standard deviation per channel (as recorded for the original), so
  /// histogram-based demos behave the same way.
  pub fn match_stats(&mut self, means: &[f64], sds: &[f64], gray: bool) {
    let n = (self.width * self.height).max(1) as f64;
    if gray {
      let l: Vec<f64> = self
        .data
        .chunks(3)
        .map(|c| luma([c[0], c[1], c[2]]))
        .collect();
      let m = l.iter().sum::<f64>() / n;
      let sd = (l.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n)
        .sqrt()
        .max(1e-6);
      let (tm, ts) = (means[0], sds[0]);
      for (i, v) in l.iter().enumerate() {
        let g = ((v - m) / sd * ts + tm).clamp(0.0, 1.0);
        self.data[i * 3..i * 3 + 3].copy_from_slice(&[g, g, g]);
      }
      return;
    }
    for ch in 0..3 {
      let m = self.data.iter().skip(ch).step_by(3).sum::<f64>() / n;
      let sd = (self
        .data
        .iter()
        .skip(ch)
        .step_by(3)
        .map(|v| (v - m).powi(2))
        .sum::<f64>()
        / n)
        .sqrt()
        .max(1e-6);
      let (tm, ts) =
        (means[ch.min(means.len() - 1)], sds[ch.min(sds.len() - 1)]);
      for v in self.data.iter_mut().skip(ch).step_by(3) {
        *v = ((*v - m) / sd * ts + tm).clamp(0.0, 1.0);
      }
    }
  }

  /// The pixels as `channels`-channel data in `[0, 1]` (1: luma, 3: RGB,
  /// 4: RGB plus opaque alpha), top row first.
  pub fn pixels(&self, channels: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(self.width * self.height * channels);
    for c in self.data.chunks(3) {
      match channels {
        1 => out.push(luma([c[0], c[1], c[2]])),
        2 => out.extend_from_slice(&[luma([c[0], c[1], c[2]]), 1.0]),
        4 => out.extend_from_slice(&[c[0], c[1], c[2], 1.0]),
        _ => out.extend_from_slice(c),
      }
    }
    out
  }

  /// Convert to an `Expr::Image` with `channels` channels, quantized to
  /// `image_type`.
  pub fn into_image(
    self,
    channels: usize,
    image_type: ImageType,
    color_space: Option<&'static str>,
  ) -> Expr {
    let mut data = self.pixels(channels);
    quantize(&mut data, image_type);
    Expr::Image {
      width: self.width as u32,
      height: self.height as u32,
      channels: channels as u8,
      data: std::sync::Arc::new(data),
      image_type,
      color_space,
      display_size: None,
    }
  }
}

/// Round values to the levels of `image_type` (`k/255` for `"Byte"`, …).
pub fn quantize(data: &mut [f64], image_type: ImageType) {
  let levels = match image_type {
    ImageType::Bit => 1.0,
    ImageType::Byte => 255.0,
    ImageType::Bit16 => 65535.0,
    _ => return,
  };
  for v in data.iter_mut() {
    *v = (v.clamp(0.0, 1.0) * levels).round() / levels;
  }
}

/// Distance from `(x, y)` to the segment `a`–`b`.
pub fn segment_distance(a: (f64, f64), b: (f64, f64), x: f64, y: f64) -> f64 {
  let (dx, dy) = (b.0 - a.0, b.1 - a.1);
  let len2 = dx * dx + dy * dy;
  let t = if len2 < 1e-12 {
    0.0
  } else {
    (((x - a.0) * dx + (y - a.1) * dy) / len2).clamp(0.0, 1.0)
  };
  ((x - a.0 - t * dx).powi(2) + (y - a.1 - t * dy).powi(2)).sqrt()
}

/// Signed distance to a polygon (negative inside, even-odd rule).
pub fn polygon_sdf(points: &[(f64, f64)], x: f64, y: f64) -> f64 {
  let n = points.len();
  let mut d = f64::MAX;
  let mut inside = false;
  for i in 0..n {
    let a = points[i];
    let b = points[(i + 1) % n];
    d = d.min(segment_distance(a, b, x, y));
    if (a.1 > y) != (b.1 > y) && x < (b.0 - a.0) * (y - a.1) / (b.1 - a.1) + a.0
    {
      inside = !inside;
    }
  }
  if inside { -d } else { d }
}

/// Points along a cubic Bézier curve.
pub fn bezier(
  p0: (f64, f64),
  p1: (f64, f64),
  p2: (f64, f64),
  p3: (f64, f64),
  steps: usize,
) -> Vec<(f64, f64)> {
  (0..=steps)
    .map(|i| {
      let t = i as f64 / steps as f64;
      let u = 1.0 - t;
      let (a, b, c, d) =
        (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
      (
        a * p0.0 + b * p1.0 + c * p2.0 + d * p3.0,
        a * p0.1 + b * p1.1 + c * p2.1 + d * p3.1,
      )
    })
    .collect()
}

/// A regular `n`-gon (or star when `inner` is given) around `(cx, cy)`.
pub fn star(
  cx: f64,
  cy: f64,
  r: f64,
  inner: Option<f64>,
  n: usize,
  rotation: f64,
) -> Vec<(f64, f64)> {
  let mut pts = Vec::new();
  let k = if inner.is_some() { 2 * n } else { n };
  for i in 0..k {
    let rr = match inner {
      Some(ri) if i % 2 == 1 => ri,
      _ => r,
    };
    let a = rotation + std::f64::consts::TAU * i as f64 / k as f64;
    pts.push((cx + rr * a.cos(), cy + rr * a.sin()));
  }
  pts
}
