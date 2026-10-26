//! Drawn stand-ins for the photographs of `"TestImageSet"` (Lisbon 2016,
//! Oxford 2011): each frame has its own composition — the lawn with a
//! distant tower, the tram street, the canal with narrowboats, the domed
//! rotunda, the gilded gate — sketched with simple shapes. Coordinates are
//! fractions of the frame (`u` across, `v` down).

use std::f64::consts::{PI, TAU};

use super::raster::*;
use super::rng::{Noise, Rng};

/// A drawing surface in frame fractions.
pub struct Sketch {
  pub c: Canvas,
  pub rng: Rng,
  pub noise: Noise,
  pub w: f64,
  pub h: f64,
}

/// Colours used across scenes.
pub mod col {
  use super::super::raster::{Rgb, rgb8};
  pub const STONE: Rgb = rgb8(200, 175, 120);
  pub const PALE_STONE: Rgb = rgb8(220, 205, 170);
  pub const WHITE_STONE: Rgb = rgb8(225, 222, 210);
  pub const BRICK: Rgb = rgb8(150, 70, 50);
  pub const GRASS: Rgb = rgb8(90, 140, 50);
  pub const TREE: Rgb = rgb8(55, 85, 40);
  pub const ROAD: Rgb = rgb8(85, 85, 88);
  pub const COBBLE: Rgb = rgb8(150, 140, 125);
  pub const WATER: Rgb = rgb8(80, 95, 90);
  pub const DARK: Rgb = rgb8(35, 30, 30);
  pub const YELLOW: Rgb = rgb8(230, 190, 40);
  pub const ROOF: Rgb = rgb8(190, 90, 60);
}

/// Sky kinds.
#[derive(Clone, Copy)]
pub enum Sky {
  Overcast,
  Clouds,
  Blue,
  Dusk,
}

impl Sketch {
  pub fn new(w: usize, h: usize, seed: u64) -> Sketch {
    Sketch {
      c: Canvas::new(w, h, WHITE),
      rng: Rng::new(seed),
      noise: Noise::new(seed ^ 0xabc, 0),
      w: w as f64,
      h: h as f64,
    }
  }

  fn px(&self, u: f64, v: f64) -> (f64, f64) {
    (u * self.w, v * self.h)
  }

  /// Fill the region `v < horizon` with sky.
  pub fn sky(&mut self, kind: Sky, horizon: f64) {
    let (w, h) = (self.w, self.h);
    let n = &self.noise;
    let f = |x: f64, y: f64| -> Rgb {
      let t = y / (horizon * h).max(1.0);
      let cloud = (n.pfbm(x / (w * 0.15), y / (h * 0.08), 5, 0.55) * 0.5 + 0.5)
        .clamp(0.0, 1.0);
      match kind {
        Sky::Overcast => shade(rgb8(225, 228, 232), 0.92 + 0.12 * cloud),
        Sky::Clouds => mix(
          mix(rgb8(110, 150, 210), rgb8(170, 195, 230), t),
          rgb8(240, 240, 242),
          ((cloud - 0.45) * 2.5).clamp(0.0, 1.0),
        ),
        Sky::Blue => mix(
          mix(rgb8(70, 120, 200), rgb8(150, 190, 235), t),
          WHITE,
          ((cloud - 0.65) * 2.0).clamp(0.0, 0.6),
        ),
        Sky::Dusk => mix(rgb8(70, 80, 130), rgb8(170, 160, 175), t),
      }
    };
    let y1 = horizon * h;
    self
      .c
      .fill_sdf((0.0, 0.0, w, y1), |_, y| y - y1, &Paint::Fn(&f), 1.0);
  }

  /// A textured region between `v0` and `v1` across the whole frame.
  pub fn ground(
    &mut self,
    v0: f64,
    v1: f64,
    base: Rgb,
    grain: f64,
    stripes: bool,
  ) {
    let (w, h) = (self.w, self.h);
    let n = &self.noise;
    let f = |x: f64, y: f64| -> Rgb {
      let t = (y / h - v0) / (v1 - v0).max(1e-6);
      let g = n.pfbm(x / (grain * w), y / (grain * h * 0.5), 3, 0.5) * 0.15;
      let stripe =
        if stripes && ((x / w * 8.0 + t * 3.0).floor() as i64) % 2 == 0 {
          0.08
        } else {
          0.0
        };
      shade(base, 0.85 + 0.25 * t + g + stripe)
    };
    self.c.rect(0.0, v0 * h, w, v1 * h, &Paint::Fn(&f));
  }

  pub fn rect(&mut self, u0: f64, v0: f64, u1: f64, v1: f64, col: Rgb) {
    let (a, b) = (self.px(u0, v0), self.px(u1, v1));
    self.c.rect(a.0, a.1, b.0, b.1, &Paint::Solid(col));
  }

  /// A stone or plaster surface with a little texture.
  pub fn wall(&mut self, u0: f64, v0: f64, u1: f64, v1: f64, col: Rgb) {
    let (a, b) = (self.px(u0, v0), self.px(u1, v1));
    let n = &self.noise;
    let f = |x: f64, y: f64| {
      shade(col, 1.0 + n.pfbm(x / 12.0, y / 12.0, 3, 0.5) * 0.12)
    };
    self.c.rect(a.0, a.1, b.0, b.1, &Paint::Fn(&f));
  }

  pub fn poly(&mut self, pts: &[(f64, f64)], col: Rgb) {
    let p: Vec<(f64, f64)> = pts.iter().map(|&(u, v)| self.px(u, v)).collect();
    self.c.polygon(&p, &Paint::Solid(col));
  }

  pub fn line(&mut self, a: (f64, f64), b: (f64, f64), width: f64, col: Rgb) {
    let (p, q) = (self.px(a.0, a.1), self.px(b.0, b.1));
    self
      .c
      .line(p, q, width * self.w.min(self.h), &Paint::Solid(col));
  }

  pub fn ellipse(&mut self, u: f64, v: f64, ru: f64, rv: f64, col: Rgb) {
    let (x, y) = self.px(u, v);
    self
      .c
      .ellipse(x, y, ru * self.w, rv * self.h, 0.0, &Paint::Solid(col));
  }

  /// A grid of windows inside a rectangle.
  #[allow(clippy::too_many_arguments)]
  pub fn windows(
    &mut self,
    u0: f64,
    v0: f64,
    u1: f64,
    v1: f64,
    cols: usize,
    rows: usize,
    col: Rgb,
    arched: bool,
  ) {
    let (cw, rh) = ((u1 - u0) / cols as f64, (v1 - v0) / rows as f64);
    for j in 0..rows {
      for i in 0..cols {
        let (x0, y0) =
          (u0 + cw * (i as f64 + 0.32), v0 + rh * (j as f64 + 0.22));
        let (x1, y1) = (x0 + cw * 0.36, y0 + rh * 0.5);
        self.rect(x0, y0, x1, y1, col);
        if arched {
          self.ellipse(
            f64::midpoint(x0, x1),
            y0,
            cw * 0.18,
            cw * 0.18 * self.w / self.h,
            col,
          );
        }
      }
    }
  }

  /// A building front: wall, windows, cornice and roof line.
  #[allow(clippy::too_many_arguments)]
  pub fn facade(
    &mut self,
    u0: f64,
    v0: f64,
    u1: f64,
    v1: f64,
    col: Rgb,
    cols: usize,
    rows: usize,
    gothic: bool,
  ) {
    self.wall(u0, v0, u1, v1, col);
    self.windows(
      u0,
      v0 + (v1 - v0) * 0.08,
      u1,
      v1 - (v1 - v0) * 0.05,
      cols,
      rows,
      shade(col, 0.35),
      gothic,
    );
    self.rect(u0, v0, u1, v0 + (v1 - v0) * 0.04, shade(col, 0.8));
    if gothic {
      // Crenellations and pinnacles along the top.
      let n = (cols * 2).max(4);
      for i in 0..n {
        if i % 2 == 0 {
          let x = u0 + (u1 - u0) * i as f64 / n as f64;
          self.rect(
            x,
            v0 - (v1 - v0) * 0.05,
            x + (u1 - u0) / n as f64,
            v0,
            col,
          );
        }
      }
      for i in 0..=cols / 2 {
        let x = u0 + (u1 - u0) * i as f64 / (cols / 2).max(1) as f64;
        self.poly(
          &[(x - 0.008, v0), (x + 0.008, v0), (x, v0 - (v1 - v0) * 0.18)],
          col,
        );
      }
    }
  }

  /// A tower with a spire, a dome or a flat crenellated top.
  pub fn tower(
    &mut self,
    u: f64,
    width: f64,
    v_top: f64,
    v_base: f64,
    col: Rgb,
    top: Top,
  ) {
    let body_top = match top {
      Top::Spire => v_top + (v_base - v_top) * 0.45,
      Top::Dome => v_top + (v_base - v_top) * 0.3,
      Top::Flat => v_top,
    };
    self.wall(u - width / 2.0, body_top, u + width / 2.0, v_base, col);
    self.windows(
      u - width / 2.0,
      body_top + 0.02,
      u + width / 2.0,
      v_base - 0.05,
      2,
      3,
      shade(col, 0.35),
      true,
    );
    match top {
      Top::Spire => {
        self.poly(
          &[
            (u - width * 0.4, body_top),
            (u + width * 0.4, body_top),
            (u, v_top),
          ],
          shade(col, 0.9),
        );
        for side in [-1.0, 1.0] {
          let x = u + side * width / 2.0;
          self.poly(
            &[
              (x - 0.006, body_top),
              (x + 0.006, body_top),
              (x, body_top - (v_base - v_top) * 0.12),
            ],
            col,
          );
        }
      }
      Top::Dome => {
        self.ellipse(
          u,
          body_top,
          width * 0.55,
          (body_top - v_top) * 0.9,
          rgb8(130, 135, 140),
        );
        self.rect(
          u - width * 0.06,
          v_top - 0.01,
          u + width * 0.06,
          v_top + (body_top - v_top) * 0.2,
          col,
        );
      }
      Top::Flat => {
        for i in 0..5 {
          if i % 2 == 0 {
            let x = u - width / 2.0 + width * i as f64 / 5.0;
            self.rect(x, v_top - 0.025, x + width / 5.0, v_top, col);
          }
        }
      }
    }
  }

  /// A domed rotunda (drum with columns, dome, lantern).
  pub fn rotunda(&mut self, u: f64, width: f64, v_top: f64, v_base: f64) {
    let drum_top = v_top + (v_base - v_top) * 0.38;
    let (x0, x1) = (u - width / 2.0, u + width / 2.0);
    // Drum shaded like a cylinder.
    let (a, b) = (self.px(x0, drum_top), self.px(x1, v_base));
    let n = &self.noise;
    let f = |x: f64, y: f64| {
      let t = (x - a.0) / (b.0 - a.0);
      shade(
        col::STONE,
        0.65 + 0.5 * (t * PI).sin() + n.perlin(x / 10.0, y / 10.0) * 0.06,
      )
    };
    self.c.rect(a.0, a.1, b.0, b.1, &Paint::Fn(&f));
    for k in 0..9 {
      let x = x0 + width * (k as f64 + 0.5) / 9.0;
      self.line(
        (x, drum_top + 0.02),
        (x, v_base - (v_base - drum_top) * 0.35),
        0.006,
        shade(col::STONE, 1.15),
      );
    }
    self.windows(
      x0,
      drum_top + (v_base - drum_top) * 0.68,
      x1,
      v_base,
      7,
      1,
      col::DARK,
      true,
    );
    // A balustrade, then the dome (a half ellipse) with ribs and a lantern.
    self.rect(x0, drum_top - 0.02, x1, drum_top, shade(col::STONE, 1.1));
    let (cx, cy) = self.px(u, drum_top - 0.02);
    let (rx, ry) = (width * 0.4 * self.w, (drum_top - v_top) * 0.7 * self.h);
    self.c.fill_sdf(
      (cx - rx, cy - ry, cx + rx, cy),
      |x, y| {
        let d = ((x - cx) / rx).powi(2) + ((y - cy) / ry).powi(2) - 1.0;
        if y > cy { 1.0 } else { d * rx.min(ry) / 2.0 }
      },
      &Paint::Solid(rgb8(120, 128, 135)),
      1.0,
    );
    for k in 1..8 {
      let t = k as f64 / 8.0 * PI;
      let (ex, ey) = (cx - rx * t.cos(), cy - ry * t.sin());
      self.c.line(
        (cx + (ex - cx) * 0.2, cy - ry * 0.95),
        (ex, ey.max(cy - ry * 0.98)),
        1.5,
        &Paint::Solid(rgb8(160, 165, 170)),
      );
    }
    self.rect(
      u - width * 0.04,
      v_top,
      u + width * 0.04,
      drum_top - 0.02 - (drum_top - v_top) * 0.68,
      col::PALE_STONE,
    );
  }

  /// A round-headed opening.
  pub fn arch(
    &mut self,
    u: f64,
    width: f64,
    v_top: f64,
    v_base: f64,
    col: Rgb,
  ) {
    let r = width / 2.0;
    let rv = r * self.w / self.h;
    self.rect(u - r, v_top + rv, u + r, v_base, col);
    self.ellipse(u, v_top + rv, r, rv, col);
  }

  /// An iron gate: bars with gilded ornament.
  pub fn gate(&mut self, u0: f64, v0: f64, u1: f64, v1: f64) {
    let n = 18;
    for i in 0..=n {
      let x = u0 + (u1 - u0) * i as f64 / n as f64;
      self.line((x, v0), (x, v1), 0.004, col::DARK);
    }
    for k in 0..4 {
      let y = v0 + (v1 - v0) * (0.15 + 0.25 * k as f64);
      self.line((u0, y), (u1, y), 0.006, col::DARK);
    }
    for _ in 0..12 {
      let (x, y) = (
        self.rng.range(u0, u1),
        self.rng.range(v0, v0 + (v1 - v0) * 0.45),
      );
      self.ellipse(x, y, 0.012, 0.01, col::YELLOW);
    }
  }

  /// A tree: trunk and a clump of foliage.
  pub fn tree(&mut self, u: f64, v_base: f64, size: f64, col: Rgb) {
    self.line(
      (u, v_base),
      (u, v_base - size * 0.5),
      size * 0.08,
      rgb8(70, 55, 40),
    );
    for _ in 0..9 {
      let (dx, dy) = (
        self.rng.range(-0.5, 0.5) * size,
        self.rng.range(-1.0, -0.4) * size,
      );
      let r = self.rng.range(0.2, 0.35) * size;
      let tone = shade(col, self.rng.range(0.75, 1.2));
      self.ellipse(
        u + dx * self.h / self.w,
        v_base + dy,
        r * self.h / self.w,
        r,
        tone,
      );
    }
  }

  /// A weeping willow: a mound of drooping strands.
  pub fn willow(&mut self, u: f64, v_base: f64, size: f64) {
    self.tree(u, v_base, size * 0.7, rgb8(110, 140, 70));
    for _ in 0..140 {
      let x = u + self.rng.range(-0.5, 0.5) * size * self.h / self.w;
      let y0 = v_base - size * self.rng.range(0.4, 0.9);
      let len = size * self.rng.range(0.3, 0.7);
      let tone = shade(rgb8(130, 160, 80), self.rng.range(0.7, 1.2));
      self.line((x, y0), (x + 0.005, y0 + len), 0.003, tone);
    }
  }

  /// A band of trees along a horizon.
  pub fn treeline(&mut self, u0: f64, u1: f64, v_base: f64, height: f64) {
    let n = ((u1 - u0) * 14.0).max(2.0) as usize;
    for i in 0..n {
      let u = u0 + (u1 - u0) * (i as f64 + self.rng.range(0.0, 1.0)) / n as f64;
      let s = height * self.rng.range(0.6, 1.1);
      self.tree(u, v_base, s, col::TREE);
    }
  }

  /// A standing figure.
  pub fn person(&mut self, u: f64, v_base: f64, height: f64, shirt: Rgb) {
    let hw = height * 0.12 * self.h / self.w;
    self.rect(
      u - hw,
      v_base - height * 0.85,
      u + hw,
      v_base - height * 0.4,
      shirt,
    );
    self.rect(
      u - hw * 0.8,
      v_base - height * 0.4,
      u - hw * 0.1,
      v_base,
      rgb8(50, 50, 70),
    );
    self.rect(
      u + hw * 0.1,
      v_base - height * 0.4,
      u + hw * 0.8,
      v_base,
      rgb8(50, 50, 70),
    );
    self.ellipse(
      u,
      v_base - height * 0.92,
      hw * 0.7,
      height * 0.08,
      rgb8(200, 160, 130),
    );
  }

  /// A box-shaped vehicle (tram, van, car, boat) with windows.
  #[allow(clippy::too_many_arguments)]
  pub fn vehicle(
    &mut self,
    u0: f64,
    v0: f64,
    u1: f64,
    v1: f64,
    body: Rgb,
    windows: usize,
    wheels: bool,
  ) {
    self.rect(u0, v0, u1, v1, body);
    if windows > 0 {
      self.windows(
        u0,
        v0 + (v1 - v0) * 0.1,
        u1,
        v0 + (v1 - v0) * 0.55,
        windows,
        1,
        rgb8(60, 70, 80),
        false,
      );
    }
    if wheels {
      let r = (v1 - v0) * 0.2;
      for u in [u0 + (u1 - u0) * 0.2, u1 - (u1 - u0) * 0.2] {
        self.ellipse(u, v1, r * self.h / self.w, r, col::DARK);
      }
    }
  }

  /// Water with reflections of what stands above it.
  pub fn water(&mut self, u0: f64, v0: f64, u1: f64, v1: f64, base: Rgb) {
    let (a, b) = (self.px(u0, v0), self.px(u1, v1));
    let n = &self.noise;
    let f = |x: f64, y: f64| {
      shade(base, 0.9 + n.pfbm(x / 60.0, y / 4.0, 3, 0.5) * 0.25)
    };
    self.c.rect(a.0, a.1, b.0, b.1, &Paint::Fn(&f));
  }

  /// Railings: a row of thin vertical bars.
  pub fn railing(&mut self, u0: f64, u1: f64, v_top: f64, v_base: f64) {
    let n = ((u1 - u0) * 80.0) as usize;
    self.line((u0, v_top), (u1, v_top), 0.004, col::DARK);
    for i in 0..=n {
      let x = u0 + (u1 - u0) * i as f64 / n.max(1) as f64;
      self.line((x, v_top), (x, v_base), 0.0025, col::DARK);
    }
  }

  /// Fine texture over everything, like a photograph.
  pub fn finish(&mut self, seed: u64) {
    let grain = Noise::new(seed ^ 0x77, 0);
    self.c.grain(&grain, 1.5, 0.025);
  }
}

#[derive(Clone, Copy)]
pub enum Top {
  Spire,
  Dome,
  Flat,
}

/// Draw the photograph `name` (`"Oxford2011_04"`, …) at `w × h`.
#[allow(clippy::too_many_lines)]
pub fn photo(name: &str, w: usize, h: usize, seed: u64) -> Canvas {
  let mut s = Sketch::new(w, h, seed);
  let frame: usize = name
    .rsplit('_')
    .next()
    .and_then(|n| n.parse().ok())
    .unwrap_or(1);
  if name.starts_with("Lisbon") {
    lisbon(&mut s, frame);
  } else {
    oxford(&mut s, frame);
  }
  s.finish(seed);
  s.c
}

#[allow(clippy::too_many_lines)]
fn oxford(s: &mut Sketch, frame: usize) {
  use col::*;
  match frame {
    1 | 18 => {
      // A wide lawn, a line of trees and college buildings, a tower.
      s.sky(Sky::Clouds, 0.62);
      s.facade(0.0, 0.42, 0.25, 0.7, PALE_STONE, 6, 2, true);
      s.treeline(0.2, 1.0, 0.68, 0.18);
      s.tower(0.66, 0.06, 0.3, 0.68, STONE, Top::Flat);
      s.rect(0.0, 0.66, 1.0, 0.7, rgb8(160, 150, 130));
      s.ground(0.7, 1.0, GRASS, 0.05, false);
    }
    2 => {
      // A golden facade with a carved doorway and a statue before it.
      s.wall(0.0, 0.0, 1.0, 1.0, STONE);
      for k in 0..6 {
        let x = 0.08 + 0.17 * k as f64;
        s.line((x, 0.0), (x, 1.0), 0.01, shade(STONE, 0.8));
      }
      s.rect(0.2, 0.05, 0.8, 0.45, shade(STONE, 1.1));
      s.rect(0.3, 0.12, 0.7, 0.25, rgb8(60, 55, 45));
      s.arch(0.5, 0.42, 0.42, 0.95, rgb8(40, 35, 30));
      s.person(0.5, 0.9, 0.42, rgb8(50, 50, 50));
      s.railing(0.25, 0.75, 0.82, 0.98);
      s.ground(0.97, 1.0, COBBLE, 0.05, false);
    }
    3 => {
      // A canal under willows, narrowboats, a towpath, a footbridge.
      s.sky(Sky::Overcast, 0.4);
      s.willow(0.15, 0.62, 0.6);
      s.treeline(0.45, 1.0, 0.45, 0.25);
      s.rect(0.3, 0.3, 0.75, 0.34, rgb8(70, 70, 70));
      s.water(0.0, 0.55, 0.7, 1.0, WATER);
      for (k, c) in [rgb8(60, 140, 140), rgb8(120, 40, 40), rgb8(40, 60, 120)]
        .into_iter()
        .enumerate()
      {
        let x = 0.3 + 0.12 * k as f64;
        s.vehicle(
          x,
          0.48 + 0.02 * k as f64,
          x + 0.1,
          0.53 + 0.02 * k as f64,
          c,
          2,
          false,
        );
      }
      s.poly(
        &[
          (0.55, 1.0),
          (0.65, 0.55),
          (0.75, 0.55),
          (1.0, 0.7),
          (1.0, 1.0),
        ],
        rgb8(140, 125, 100),
      );
      s.ground(0.62, 1.0, GRASS, 0.04, false);
      s.poly(
        &[(0.6, 1.0), (0.72, 0.6), (0.82, 0.6), (0.9, 1.0)],
        rgb8(150, 135, 110),
      );
    }
    4 | 11 | 12 => {
      // The domed rotunda between college buildings.
      let wide = frame == 12;
      s.sky(
        if frame == 12 {
          Sky::Clouds
        } else {
          Sky::Overcast
        },
        0.85,
      );
      s.facade(0.0, 0.45, 0.22, 0.85, STONE, 4, 3, true);
      s.facade(0.78, 0.45, 1.0, 0.85, STONE, 4, 3, true);
      let (u, wd) = if wide { (0.5, 0.18) } else { (0.5, 0.42) };
      s.rotunda(u, wd, 0.12, 0.88);
      s.ground(
        0.86,
        1.0,
        if frame == 11 { COBBLE } else { GRASS },
        0.05,
        frame == 12,
      );
      if wide {
        s.railing(0.2, 0.8, 0.86, 0.95);
      }
    }
    5 => {
      // A long college front, a curving railing and a lawn.
      s.sky(Sky::Overcast, 0.5);
      s.facade(0.0, 0.32, 1.0, 0.72, STONE, 12, 3, true);
      s.ground(0.72, 1.0, ROAD, 0.05, false);
      s.poly(&[(0.4, 1.0), (0.55, 0.75), (1.0, 0.72), (1.0, 1.0)], GRASS);
      s.railing(0.45, 1.0, 0.72, 0.8);
      s.vehicle(0.02, 0.68, 0.12, 0.76, rgb8(230, 230, 230), 2, true);
    }
    6 | 8 => {
      // A church with a tall spire.
      s.sky(Sky::Overcast, 0.85);
      let u = if frame == 6 { 0.5 } else { 0.55 };
      s.facade(0.0, 0.45, 0.3, 0.85, STONE, 3, 3, true);
      s.facade(0.7, 0.5, 1.0, 0.85, STONE, 3, 3, true);
      s.tower(u, 0.18, 0.02, 0.88, STONE, Top::Spire);
      if frame == 8 {
        s.ground(0.84, 1.0, GRASS, 0.05, true);
        s.railing(0.4, 1.0, 0.82, 0.9);
      } else {
        s.treeline(0.0, 1.0, 1.0, 0.18);
      }
    }
    7 | 13 => {
      // A crenellated college front with turrets and a gate tower.
      s.sky(Sky::Overcast, 0.75);
      s.facade(0.0, 0.32, 1.0, 0.8, PALE_STONE, 10, 3, true);
      s.tower(
        if frame == 7 { 0.35 } else { 0.5 },
        0.16,
        0.05,
        0.85,
        PALE_STONE,
        Top::Flat,
      );
      s.arch(
        if frame == 7 { 0.35 } else { 0.5 },
        0.08,
        0.66,
        0.85,
        rgb8(50, 45, 40),
      );
      s.ground(0.8, 1.0, ROAD, 0.05, false);
    }
    9 => {
      // Looking through a dark archway at twin towers across a lawn.
      s.sky(Sky::Overcast, 0.5);
      s.facade(0.2, 0.35, 0.8, 0.6, PALE_STONE, 6, 2, true);
      s.tower(0.37, 0.07, 0.15, 0.6, PALE_STONE, Top::Spire);
      s.tower(0.63, 0.07, 0.15, 0.6, PALE_STONE, Top::Spire);
      s.ground(0.6, 0.8, GRASS, 0.05, true);
      s.ground(0.8, 1.0, rgb8(170, 160, 145), 0.05, false);
      // The arch frame.
      s.rect(0.0, 0.0, 0.12, 1.0, rgb8(70, 60, 50));
      s.rect(0.88, 0.0, 1.0, 1.0, rgb8(70, 60, 50));
      s.rect(0.0, 0.0, 1.0, 0.08, rgb8(70, 60, 50));
      s.poly(&[(0.12, 0.08), (0.5, 0.08), (0.12, 0.3)], rgb8(70, 60, 50));
      s.poly(&[(0.88, 0.08), (0.5, 0.08), (0.88, 0.3)], rgb8(70, 60, 50));
    }
    10 => {
      // A gothic front with two pinnacled towers.
      s.sky(Sky::Overcast, 0.9);
      s.facade(0.1, 0.4, 0.9, 0.9, PALE_STONE, 6, 3, true);
      s.tower(0.32, 0.14, 0.05, 0.9, PALE_STONE, Top::Spire);
      s.tower(0.68, 0.14, 0.05, 0.9, PALE_STONE, Top::Spire);
      s.ground(0.88, 1.0, GRASS, 0.05, false);
    }
    14 => {
      // A canal, a towpath, a road with yellow lines.
      s.sky(Sky::Overcast, 0.35);
      s.treeline(0.0, 0.8, 0.45, 0.25);
      s.water(0.0, 0.45, 0.3, 1.0, WATER);
      s.ground(0.45, 1.0, GRASS, 0.04, false);
      s.poly(
        &[(0.3, 1.0), (0.42, 0.45), (0.48, 0.45), (0.5, 1.0)],
        rgb8(160, 145, 120),
      );
      s.poly(&[(0.7, 1.0), (0.75, 0.42), (1.0, 0.42), (1.0, 1.0)], ROAD);
      s.line((0.8, 1.0), (0.78, 0.45), 0.004, YELLOW);
      s.line((0.82, 1.0), (0.79, 0.45), 0.004, YELLOW);
      s.rect(0.62, 0.42, 0.7, 1.0, BRICK);
    }
    15 | 19 => {
      // A gilded iron gate in a stone arch (and its gatehouse).
      s.sky(Sky::Overcast, 0.1);
      s.wall(0.0, 0.0, 1.0, 1.0, PALE_STONE);
      if frame == 19 {
        s.tower(0.3, 0.12, 0.0, 0.35, PALE_STONE, Top::Dome);
        s.tower(0.7, 0.12, 0.0, 0.35, PALE_STONE, Top::Dome);
        s.ground(0.85, 1.0, ROAD, 0.05, false);
      }
      let (v0, v1) = if frame == 15 {
        (0.1, 1.0)
      } else {
        (0.45, 0.85)
      };
      s.arch(0.5, 0.75, v0, v1, rgb8(60, 55, 50));
      s.gate(0.18, v0 + 0.15, 0.82, v1);
    }
    16 => {
      // A classical archway with columns, a gate, a lane.
      s.wall(0.0, 0.0, 1.0, 0.75, PALE_STONE);
      for x in [0.1, 0.25, 0.75, 0.9] {
        s.rect(x - 0.03, 0.08, x + 0.03, 0.7, shade(PALE_STONE, 1.08));
      }
      s.arch(0.5, 0.4, 0.15, 0.7, rgb8(40, 40, 40));
      s.wall(0.38, 0.4, 0.62, 0.7, STONE);
      s.gate(0.33, 0.45, 0.67, 0.72);
      s.ground(0.72, 1.0, ROAD, 0.05, false);
      s.line((0.0, 0.95), (0.3, 0.72), 0.005, YELLOW);
    }
    17 => {
      // A river under trees, an iron footbridge, ducks on the bank.
      s.sky(Sky::Overcast, 0.1);
      s.treeline(0.0, 1.0, 0.5, 0.45);
      s.water(0.0, 0.48, 1.0, 0.85, WATER);
      s.line((0.65, 0.42), (0.95, 0.4), 0.02, rgb8(60, 60, 60));
      s.c.ring(
        0.8 * s.w,
        0.5 * s.h,
        0.12 * s.w,
        0.01 * s.w,
        &Paint::Solid(rgb8(60, 60, 60)),
      );
      s.poly(
        &[(0.0, 1.0), (0.4, 0.82), (1.0, 0.6), (1.0, 1.0)],
        rgb8(120, 120, 80),
      );
      s.poly(
        &[(0.55, 1.0), (1.0, 0.68), (1.0, 0.75), (0.65, 1.0)],
        rgb8(150, 130, 110),
      );
      for k in 0..4 {
        s.ellipse(
          0.1 + 0.08 * k as f64,
          0.9 - 0.02 * k as f64,
          0.025,
          0.015,
          rgb8(110, 90, 70),
        );
      }
    }
    _ => {
      // Dusk on the canal: modern brick blocks, their lights in the water.
      s.sky(Sky::Dusk, 0.5);
      s.treeline(0.0, 0.3, 0.55, 0.3);
      s.facade(0.55, 0.05, 1.0, 0.6, BRICK, 5, 6, false);
      s.facade(0.4, 0.35, 0.56, 0.6, BRICK, 2, 3, false);
      s.water(0.38, 0.6, 1.0, 1.0, rgb8(60, 60, 80));
      s.poly(
        &[(0.0, 1.0), (0.0, 0.6), (0.38, 0.6), (0.3, 1.0)],
        rgb8(60, 80, 50),
      );
      s.poly(
        &[(0.1, 1.0), (0.32, 0.6), (0.36, 0.6), (0.25, 1.0)],
        rgb8(130, 125, 120),
      );
    }
  }
}

#[allow(clippy::too_many_lines)]
fn lisbon(s: &mut Sketch, frame: usize) {
  use col::*;
  match frame {
    1 => {
      // The side of a yellow van with a slogan, its wheel, the street.
      s.ground(0.0, 1.0, COBBLE, 0.02, false);
      s.vehicle(0.0, 0.0, 1.0, 0.65, YELLOW, 0, false);
      for k in 0..10 {
        s.rect(
          0.3 + 0.06 * k as f64,
          0.28,
          0.34 + 0.06 * k as f64,
          0.35,
          DARK,
        );
      }
      s.line((0.36, 0.42), (0.9, 0.42), 0.012, DARK);
      s.rect(0.0, 0.65, 1.0, 0.72, rgb8(60, 60, 60));
      s.ellipse(0.52, 0.8, 0.12, 0.17, DARK);
      s.ellipse(0.52, 0.8, 0.05, 0.07, rgb8(160, 160, 160));
    }
    2 => {
      // Glazed tiles: a repeating ornament in yellow, orange and blue.
      s.rect(0.0, 0.0, 1.0, 1.0, rgb8(235, 235, 225));
      let n = 4;
      for j in 0..n {
        for i in 0..n {
          let (u, v) =
            ((i as f64 + 0.5) / n as f64, (j as f64 + 0.5) / n as f64);
          let r = 0.5 / n as f64;
          s.poly(
            &[(u - r, v), (u, v - r), (u + r, v), (u, v + r)],
            rgb8(220, 170, 60),
          );
          s.poly(
            &[
              (u - r * 0.6, v),
              (u, v - r * 0.6),
              (u + r * 0.6, v),
              (u, v + r * 0.6),
            ],
            rgb8(240, 240, 230),
          );
          s.ellipse(u, v, r * 0.25, r * 0.25, rgb8(60, 90, 150));
          for k in 0..4 {
            let a = PI / 4.0 + TAU * k as f64 / 4.0;
            s.ellipse(
              u + r * 0.7 * a.cos(),
              v + r * 0.7 * a.sin(),
              r * 0.18,
              r * 0.18,
              rgb8(190, 90, 40),
            );
          }
          s.line((u - r, v - r), (u + r, v - r), 0.004, rgb8(100, 150, 120));
        }
      }
    }
    3 => {
      // A steep shopping street: facades closing in, the lift tower at the
      // end, people on the paving.
      s.sky(Sky::Overcast, 0.4);
      s.poly(
        &[(0.0, 0.0), (0.38, 0.25), (0.38, 0.75), (0.0, 1.0)],
        rgb8(190, 180, 160),
      );
      s.poly(
        &[(1.0, 0.0), (0.62, 0.25), (0.62, 0.75), (1.0, 1.0)],
        rgb8(175, 165, 150),
      );
      for k in 0..6 {
        let t = k as f64 / 6.0;
        s.line(
          (0.38 * t, 0.15 + 0.1 * t),
          (0.38 * t, 0.85 - 0.1 * t),
          0.004,
          rgb8(90, 80, 70),
        );
        s.line(
          (1.0 - 0.38 * t, 0.15 + 0.1 * t),
          (1.0 - 0.38 * t, 0.85 - 0.1 * t),
          0.004,
          rgb8(90, 80, 70),
        );
      }
      s.rect(0.45, 0.08, 0.55, 0.6, rgb8(120, 120, 125));
      s.windows(0.45, 0.1, 0.55, 0.55, 2, 6, rgb8(220, 225, 230), false);
      s.poly(
        &[(0.38, 0.75), (0.62, 0.75), (1.0, 1.0), (0.0, 1.0)],
        COBBLE,
      );
      for k in 0..7 {
        let u = 0.25 + 0.08 * k as f64;
        let size = 0.15 + 0.03 * (k % 3) as f64;
        let shirt = [s.rng.uniform(), s.rng.uniform(), s.rng.uniform()];
        s.person(u, 0.92 + 0.02 * (k % 2) as f64, size, shirt);
      }
    }
    4 => {
      // Looking up into a coffered vault, daylight in the arch below.
      s.rect(0.0, 0.0, 1.0, 1.0, rgb8(200, 185, 160));
      for k in 0..12 {
        let a = PI * k as f64 / 11.0;
        s.line(
          (0.5, 0.4),
          (0.5 + 0.7 * a.cos(), 0.4 - 0.6 * a.sin()),
          0.015,
          rgb8(150, 135, 115),
        );
      }
      for k in 1..4 {
        s.c.ring(
          0.5 * s.w,
          0.4 * s.h,
          k as f64 * 0.15 * s.w,
          0.01 * s.w,
          &Paint::Solid(rgb8(160, 145, 120)),
        );
      }
      s.ellipse(0.5, 0.4, 0.05, 0.06, rgb8(170, 150, 120));
      s.ellipse(0.5, 1.05, 0.35, 0.18, rgb8(245, 245, 245));
    }
    5 => {
      // The cathedral: two square towers, a rose window, the door; cars.
      s.sky(Sky::Clouds, 0.8);
      s.wall(0.15, 0.25, 0.85, 0.85, rgb8(190, 170, 130));
      s.tower(0.27, 0.22, 0.1, 0.85, rgb8(185, 165, 125), Top::Flat);
      s.tower(0.73, 0.22, 0.1, 0.85, rgb8(185, 165, 125), Top::Flat);
      s.ellipse(0.5, 0.4, 0.06, 0.045, rgb8(80, 70, 60));
      s.arch(0.5, 0.18, 0.5, 0.85, rgb8(50, 45, 40));
      s.ground(0.85, 1.0, ROAD, 0.05, false);
      s.vehicle(0.05, 0.8, 0.45, 0.98, rgb8(230, 230, 230), 2, true);
      s.vehicle(0.55, 0.82, 0.95, 0.98, rgb8(40, 40, 40), 2, true);
      s.line((0.0, 0.05), (1.0, 0.2), 0.002, DARK);
    }
    6 => {
      // A white statue on its pedestal in a square.
      s.sky(Sky::Clouds, 0.55);
      s.facade(0.0, 0.42, 0.45, 0.62, rgb8(220, 200, 170), 6, 2, false);
      s.facade(0.55, 0.45, 1.0, 0.62, rgb8(235, 225, 215), 6, 2, false);
      s.ground(0.6, 1.0, rgb8(180, 170, 155), 0.02, false);
      s.rect(0.4, 0.5, 0.6, 0.92, rgb8(220, 215, 205));
      s.rect(0.43, 0.15, 0.57, 0.5, rgb8(225, 220, 210));
      s.ellipse(0.5, 0.12, 0.04, 0.035, rgb8(225, 220, 210));
      for k in 0..3 {
        s.ellipse(0.72 + 0.1 * k as f64, 0.55, 0.05, 0.02, WHITE);
      }
    }
    7 => {
      // A tiled peacock on a whitewashed wall.
      s.wall(0.0, 0.0, 1.0, 1.0, rgb8(235, 233, 225));
      s.wall(0.0, 0.0, 0.92, 0.66, rgb8(240, 238, 232));
      s.poly(
        &[(0.3, 0.3), (0.95, 0.55), (0.9, 0.62), (0.3, 0.45)],
        rgb8(70, 130, 70),
      );
      for k in 0..6 {
        s.ellipse(
          0.45 + 0.08 * k as f64,
          0.42 + 0.025 * k as f64,
          0.02,
          0.025,
          rgb8(200, 140, 40),
        );
      }
      s.ellipse(0.25, 0.4, 0.07, 0.12, rgb8(40, 80, 160));
      s.ellipse(0.2, 0.25, 0.03, 0.04, rgb8(40, 80, 160));
      s.ellipse(0.2, 0.62, 0.06, 0.05, rgb8(60, 110, 170));
      s.ellipse(0.32, 0.6, 0.05, 0.05, rgb8(220, 170, 40));
      s.line((0.0, 0.67), (0.92, 0.67), 0.01, rgb8(200, 195, 185));
      s.line((0.85, 0.75), (0.75, 0.95), 0.003, rgb8(150, 145, 140));
    }
    8 => {
      // A domed white church against a blue sky.
      s.sky(Sky::Blue, 0.85);
      s.facade(0.25, 0.45, 0.75, 0.88, WHITE_STONE, 5, 3, false);
      s.rotunda(0.5, 0.28, 0.08, 0.5);
      s.facade(0.0, 0.7, 0.2, 0.88, rgb8(220, 160, 120), 2, 2, false);
      s.facade(0.82, 0.72, 1.0, 0.88, rgb8(230, 200, 150), 2, 2, false);
      s.ground(0.88, 1.0, rgb8(170, 165, 160), 0.03, false);
    }
    9 => {
      // A heart of flowers hanging in a white courtyard, a red door.
      s.wall(0.0, 0.0, 1.0, 0.85, rgb8(235, 235, 230));
      s.rect(0.85, 0.15, 1.0, 0.8, rgb8(160, 40, 30));
      s.rect(0.75, 0.0, 0.83, 0.85, rgb8(220, 220, 215));
      s.ground(0.85, 1.0, COBBLE, 0.02, false);
      let heart = |t: f64| -> (f64, f64) {
        let x = 16.0 * t.sin().powi(3);
        let y = 13.0 * t.cos()
          - 5.0 * (2.0 * t).cos()
          - 2.0 * (3.0 * t).cos()
          - (4.0 * t).cos();
        (0.42 + x * 0.018, 0.4 - y * 0.02)
      };
      let pts: Vec<(f64, f64)> =
        (0..60).map(|i| heart(TAU * i as f64 / 60.0)).collect();
      s.poly(&pts, rgb8(150, 50, 60));
      for _ in 0..300 {
        let t = s.rng.range(0.0, TAU);
        let k = s.rng.range(0.0, 1.0);
        let (x, y) = heart(t);
        let (u, v) = (0.42 + (x - 0.42) * k, 0.4 + (y - 0.4) * k);
        let c = [s.rng.uniform(), s.rng.uniform(), s.rng.uniform()];
        s.ellipse(u, v, 0.012, 0.012, c);
      }
    }
    10 => {
      // Over red roofs to the river and its bridge.
      s.sky(Sky::Blue, 0.3);
      s.rect(0.0, 0.28, 1.0, 0.35, rgb8(120, 140, 160));
      s.line((0.55, 0.3), (0.8, 0.31), 0.004, rgb8(150, 60, 50));
      s.ground(0.35, 1.0, rgb8(200, 180, 160), 0.05, false);
      for _ in 0..260 {
        let (u, v) = (s.rng.uniform(), s.rng.range(0.36, 0.95));
        let size = 0.015 + 0.03 * (v - 0.35);
        let roof = shade(ROOF, s.rng.range(0.8, 1.2));
        s.poly(
          &[
            (u - size, v),
            (u + size, v),
            (u + size * 0.6, v - size * 0.8),
            (u - size * 0.6, v - size * 0.8),
          ],
          roof,
        );
        s.rect(u - size, v, u + size, v + size * 0.6, rgb8(235, 225, 210));
      }
      s.tree(0.05, 1.0, 0.35, TREE);
      s.tree(0.95, 1.0, 0.3, TREE);
    }
    11 => {
      // A peacock on a rock under trees.
      s.rect(0.0, 0.0, 1.0, 1.0, rgb8(60, 90, 50));
      s.treeline(0.0, 1.0, 0.6, 0.55);
      s.poly(
        &[(0.0, 1.0), (0.0, 0.62), (0.5, 0.55), (1.0, 0.6), (1.0, 1.0)],
        rgb8(150, 150, 140),
      );
      s.ellipse(0.48, 0.5, 0.1, 0.05, rgb8(40, 70, 110));
      s.poly(&[(0.55, 0.48), (0.85, 0.5), (0.58, 0.53)], rgb8(70, 90, 60));
      s.line((0.4, 0.5), (0.36, 0.38), 0.012, rgb8(40, 70, 110));
      s.line((0.47, 0.55), (0.47, 0.62), 0.004, rgb8(120, 110, 90));
      s.line((0.52, 0.55), (0.52, 0.62), 0.004, rgb8(120, 110, 90));
    }
    12 => {
      // A café table under trees, people around it.
      s.rect(0.0, 0.0, 1.0, 1.0, rgb8(70, 100, 60));
      s.treeline(0.0, 1.0, 0.45, 0.4);
      s.wall(0.55, 0.15, 0.8, 0.5, rgb8(220, 210, 190));
      s.ground(0.5, 1.0, rgb8(160, 150, 120), 0.05, false);
      s.person(0.25, 0.7, 0.3, rgb8(180, 170, 160));
      s.person(0.6, 0.78, 0.4, rgb8(230, 230, 240));
      for k in 0..5 {
        s.rect(
          0.5,
          0.48 + 0.05 * k as f64,
          0.7,
          0.5 + 0.05 * k as f64,
          rgb8(40, 40, 90),
        );
      }
      s.poly(
        &[(0.1, 0.78), (0.75, 0.78), (0.85, 1.0), (0.0, 1.0)],
        rgb8(110, 60, 40),
      );
      s.line((0.92, 0.0), (0.88, 1.0), 0.06, rgb8(110, 90, 70));
    }
    _ => {
      // A tram street: tall houses, the yellow tram, a scooter.
      s.sky(Sky::Overcast, 0.2);
      s.facade(0.0, 0.0, 0.4, 0.75, rgb8(200, 190, 175), 3, 6, false);
      s.facade(0.4, 0.05, 1.0, 0.6, rgb8(225, 215, 200), 5, 5, false);
      s.ground(0.6, 1.0, ROAD, 0.05, false);
      s.vehicle(0.25, 0.35, 0.8, 0.72, YELLOW, 4, true);
      s.rect(0.25, 0.62, 0.8, 0.66, rgb8(150, 40, 30));
      s.line((0.3, 0.0), (0.4, 0.35), 0.003, DARK);
      s.vehicle(0.08, 0.68, 0.18, 0.82, rgb8(40, 40, 50), 0, true);
      s.person(0.13, 0.75, 0.15, rgb8(40, 40, 60));
    }
  }
}
