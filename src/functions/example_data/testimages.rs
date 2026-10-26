//! Drawn stand-ins for the `"TestImage"` photographs and test charts, one
//! composition per entry, following each original's layout: what is shown,
//! where, from which viewpoint (a tank seen from above on scrubland, an
//! alarm clock on a pile of books, a lake framed by pines with a sailboat).
//! The synthetic charts are drawn as such: `"Gray21"` is the 3 × 7 grid of
//! the 21 gray levels, `"Ruler"` a millimetre grid with a scale,
//! `"ResolutionChart"` groups of bars at decreasing sizes.

use std::f64::consts::{PI, TAU};

use super::photos::{Sketch, Sky, Top, col};
use super::raster::*;

/// Scrubland seen from above: sandy ground with clumps of bushes.
fn scrub(s: &mut Sketch, density: f64, base: Rgb) {
  s.ground(0.0, 1.0, base, 0.03, false);
  let (w, h) = (s.w, s.h);
  let n = (1300.0 * density) as usize;
  for _ in 0..n {
    let (x, y) = (s.rng.range(0.0, w), s.rng.range(0.0, h));
    // Smaller and denser towards the top (further away).
    let r = s.rng.range(0.6, 2.2) * (0.5 + y / h) * w / 512.0 * 3.0;
    let tone = shade(rgb8(105, 105, 90), s.rng.range(0.6, 1.2));
    for _ in 0..4 {
      let (dx, dy) = (s.rng.range(-r, r), s.rng.range(-r, r) * 0.7);
      s.c.circle(x + dx, y + dy, r * 0.6, &Paint::Solid(tone));
    }
  }
}

/// A tracked vehicle seen obliquely from above.
fn tank(s: &mut Sketch, u: f64, v: f64, size: f64, angle: f64, turret: bool) {
  let olive = rgb8(80, 80, 65);
  let (ca, sa) = (angle.cos(), angle.sin());
  let rot = |x: f64, y: f64| {
    (
      u + (x * ca - y * sa) * size,
      v + (x * sa + y * ca) * size * 1.3,
    )
  };
  let hull: Vec<(f64, f64)> =
    [(-1.0, -0.6), (1.0, -0.6), (1.0, 0.6), (-1.0, 0.6)]
      .iter()
      .map(|&(x, y)| rot(x, y))
      .collect();
  // Shadow, tracks, hull, turret and gun.
  let shadow: Vec<(f64, f64)> = hull
    .iter()
    .map(|&(x, y)| (x + size * 0.15, y + size * 0.2))
    .collect();
  s.poly(&shadow, rgb8(40, 40, 35));
  for side in [-0.55, 0.55] {
    let side: f64 = side * 1.2;
    let track: Vec<(f64, f64)> = [
      (-1.05, side - 0.14),
      (1.05, side - 0.14),
      (1.05, side + 0.14),
      (-1.05, side + 0.14),
    ]
    .iter()
    .map(|&(x, y)| rot(x, y))
    .collect();
    s.poly(&track, rgb8(35, 35, 30));
  }
  s.poly(&hull, olive);
  if turret {
    let t: Vec<(f64, f64)> = (0..12)
      .map(|i| {
        let a = TAU * i as f64 / 12.0;
        (0.45 * a.cos() - 0.1, 0.42 * a.sin())
      })
      .map(|(x, y)| rot(x, y))
      .collect();
    s.poly(&t, shade(olive, 1.2));
    s.line(rot(0.3, 0.0), rot(1.6, 0.0), size * 0.12, shade(olive, 0.8));
  } else {
    let deck: Vec<(f64, f64)> =
      [(-0.7, -0.35), (0.6, -0.35), (0.6, 0.35), (-0.7, 0.35)]
        .iter()
        .map(|&(x, y)| rot(x, y))
        .collect();
    s.poly(&deck, shade(olive, 1.15));
  }
}

/// A box truck (or van) seen obliquely: box, cab, wheels.
fn truck(s: &mut Sketch, u: f64, v: f64, size: f64, flip: bool) {
  let d = if flip { -1.0 } else { 1.0 };
  let body = rgb8(95, 95, 85);
  s.poly(
    &[
      (u - size, v),
      (u + size * 0.5 * d, v),
      (u + size * 0.5 * d, v - size * 0.9),
      (u - size, v - size * 0.9),
    ],
    body,
  );
  s.poly(
    &[
      (u - size, v - size * 0.9),
      (u + size * 0.5 * d, v - size * 0.9),
      (u + size * 0.7 * d, v - size * 1.15),
      (u - size * 0.8, v - size * 1.15),
    ],
    shade(body, 1.3),
  );
  s.poly(
    &[
      (u + size * 0.5 * d, v),
      (u + size * 1.1 * d, v),
      (u + size * 1.1 * d, v - size * 0.55),
      (u + size * 0.5 * d, v - size * 0.6),
    ],
    shade(body, 0.8),
  );
  for x in [u - size * 0.6, u + size * 0.8 * d] {
    s.ellipse(x, v, size * 0.18, size * 0.22, rgb8(25, 25, 25));
  }
}

/// A car seen obliquely.
fn car(s: &mut Sketch, u: f64, v: f64, size: f64, body: Rgb) {
  s.poly(
    &[
      (u - size, v),
      (u + size, v - size * 0.15),
      (u + size, v - size * 0.45),
      (u - size, v - size * 0.3),
    ],
    body,
  );
  s.poly(
    &[
      (u - size * 0.4, v - size * 0.35),
      (u + size * 0.5, v - size * 0.45),
      (u + size * 0.3, v - size * 0.7),
      (u - size * 0.3, v - size * 0.6),
    ],
    shade(body, 0.7),
  );
  for x in [u - size * 0.6, u + size * 0.6] {
    s.ellipse(
      x,
      v - size * 0.05,
      size * 0.15,
      size * 0.18,
      rgb8(20, 20, 20),
    );
  }
}

/// An aircraft seen from above: fuselage, swept wings, tailplane.
fn aircraft(
  s: &mut Sketch,
  u: f64,
  v: f64,
  size: f64,
  angle: f64,
  body: Rgb,
  span: f64,
) {
  let (ca, sa) = (angle.cos(), angle.sin());
  let aspect = 1.0;
  let rot = |x: f64, y: f64| {
    (
      u + (x * ca - y * sa) * size,
      v + (x * sa + y * ca) * size * aspect,
    )
  };
  let wing: Vec<(f64, f64)> = [
    (0.15, 0.0),
    (-0.25, span),
    (-0.4, span),
    (-0.2, 0.0),
    (-0.4, -span),
    (-0.25, -span),
  ]
  .iter()
  .map(|&(x, y)| rot(x, y))
  .collect();
  s.poly(&wing, shade(body, 0.85));
  let tail: Vec<(f64, f64)> = [
    (-0.75, 0.0),
    (-0.95, span * 0.4),
    (-1.0, span * 0.4),
    (-0.92, 0.0),
    (-1.0, -span * 0.4),
    (-0.95, -span * 0.4),
  ]
  .iter()
  .map(|&(x, y)| rot(x, y))
  .collect();
  s.poly(&tail, shade(body, 0.85));
  let fuselage: Vec<(f64, f64)> = (0..24)
    .map(|i| {
      let t = TAU * i as f64 / 24.0;
      rot(t.cos(), 0.09 * t.sin())
    })
    .collect();
  s.poly(&fuselage, body);
  let (cx, cy) = rot(0.6, 0.0);
  s.ellipse(cx, cy, size * 0.08, size * 0.035, rgb8(40, 50, 70));
}

/// A head-and-shoulders portrait.
#[allow(clippy::too_many_arguments)]
fn portrait(
  s: &mut Sketch,
  u: f64,
  v: f64,
  size: f64,
  skin: Rgb,
  hair: Rgb,
  long_hair: bool,
  top: Rgb,
) {
  let k = 1.0;
  s.ellipse(u, v + size * 1.9, size * 1.4 * k, size * 1.0, top);
  s.rect(
    u - size * 0.2,
    v + size * 0.7,
    u + size * 0.2,
    v + size * 1.0,
    shade(skin, 0.85),
  );
  if long_hair {
    s.ellipse(u, v + size * 0.5, size * 0.85, size * 1.4, hair);
  }
  s.ellipse(u, v, size * 0.6, size * 0.85, skin);
  s.ellipse(u, v - size * 0.6, size * 0.68, size * 0.42, hair);
  for side in [-1.0, 1.0] {
    let ex = u + side * size * 0.24;
    s.ellipse(ex, v - size * 0.05, size * 0.1, size * 0.05, WHITE);
    s.ellipse(
      ex,
      v - size * 0.05,
      size * 0.05,
      size * 0.05,
      rgb8(60, 50, 40),
    );
  }
  s.line(
    (u, v),
    (u - size * 0.04, v + size * 0.25),
    size * 0.03,
    shade(skin, 0.7),
  );
  s.ellipse(
    u,
    v + size * 0.45,
    size * 0.18,
    size * 0.05,
    rgb8(170, 70, 70),
  );
}

/// A figure seen whole, in a dark suit or a dress.
fn figure(
  s: &mut Sketch,
  u: f64,
  v_base: f64,
  height: f64,
  clothes: Rgb,
  dress: bool,
) {
  let hw = height * 0.11;
  if dress {
    s.poly(
      &[
        (u - hw * 0.8, v_base - height * 0.82),
        (u + hw * 0.8, v_base - height * 0.82),
        (u + hw * 1.3, v_base - height * 0.3),
        (u - hw * 1.3, v_base - height * 0.3),
      ],
      clothes,
    );
    s.line(
      (u - hw * 0.4, v_base - height * 0.3),
      (u - hw * 0.4, v_base),
      height * 0.03,
      rgb8(200, 160, 130),
    );
    s.line(
      (u + hw * 0.4, v_base - height * 0.3),
      (u + hw * 0.4, v_base),
      height * 0.03,
      rgb8(200, 160, 130),
    );
  } else {
    s.rect(
      u - hw,
      v_base - height * 0.82,
      u + hw,
      v_base - height * 0.45,
      clothes,
    );
    s.rect(
      u - hw * 0.9,
      v_base - height * 0.45,
      u - hw * 0.1,
      v_base,
      shade(clothes, 0.8),
    );
    s.rect(
      u + hw * 0.1,
      v_base - height * 0.45,
      u + hw * 0.9,
      v_base,
      shade(clothes, 0.8),
    );
  }
  s.rect(
    u - hw * 0.25,
    v_base - height * 0.86,
    u + hw * 0.25,
    v_base - height * 0.82,
    rgb8(200, 160, 130),
  );
  s.ellipse(
    u,
    v_base - height * 0.92,
    hw * 0.65,
    height * 0.07,
    rgb8(205, 165, 135),
  );
  s.ellipse(
    u,
    v_base - height * 0.97,
    hw * 0.7,
    height * 0.035,
    rgb8(150, 110, 60),
  );
}

/// Interior of a room: walls, curtains, furniture.
fn room(s: &mut Sketch) {
  s.wall(0.0, 0.0, 1.0, 0.75, rgb8(170, 150, 120));
  s.rect(0.55, 0.05, 0.9, 0.55, rgb8(220, 215, 200));
  s.rect(0.5, 0.05, 0.56, 0.6, rgb8(140, 110, 90));
  s.rect(0.89, 0.05, 0.95, 0.6, rgb8(140, 110, 90));
  s.ground(0.75, 1.0, rgb8(150, 120, 100), 0.04, false);
  s.rect(0.02, 0.55, 0.25, 0.85, rgb8(110, 80, 60));
  s.rect(0.7, 0.62, 0.98, 0.8, rgb8(90, 60, 40));
}

/// Fruit or vegetables filling the frame.
fn heap(
  s: &mut Sketch,
  palette: &[Rgb],
  n: usize,
  r: (f64, f64),
  stretch: f64,
) {
  for _ in 0..n {
    let col = *s.rng.pick(palette);
    let (u, v) = (s.rng.range(-0.05, 1.05), s.rng.range(-0.05, 1.05));
    let size = s.rng.range(r.0, r.1);
    let (ru, rv) = (size, size * stretch * s.rng.range(0.8, 1.2));
    s.ellipse(u + 0.01, v + 0.012, ru, rv, shade(col, 0.35));
    s.ellipse(u, v, ru, rv, col);
    s.ellipse(
      u - ru * 0.2,
      v - rv * 0.2,
      ru * 0.6,
      rv * 0.6,
      shade(col, 1.18),
    );
    s.ellipse(
      u - ru * 0.35,
      v - rv * 0.45,
      ru * 0.15,
      rv * 0.1,
      mix(col, WHITE, 0.85),
    );
  }
}

/// Draw the test image `name` at `w × h`.
#[allow(clippy::too_many_lines)]
pub fn draw(name: &str, w: usize, h: usize, seed: u64) -> Option<Canvas> {
  use col::*;
  let mut s = Sketch::new(w, h, seed);
  match name {
    "Gray21" => return Some(gray21(w, h)),
    "Ruler" => return Some(ruler(w, h)),
    "ResolutionChart" => return Some(resolution_chart(w, h)),
    "APC" => {
      scrub(&mut s, 0.35, rgb8(170, 160, 140));
      tank(&mut s, 0.45, 0.52, 0.11, 0.0, false);
    }
    "Tank" => {
      scrub(&mut s, 0.6, rgb8(165, 160, 145));
      tank(&mut s, 0.55, 0.45, 0.17, -1.1, true);
    }
    "Tank2" => {
      scrub(&mut s, 1.0, rgb8(140, 140, 125));
      tank(&mut s, 0.33, 0.4, 0.1, -0.1, true);
    }
    "Tank3" => {
      scrub(&mut s, 0.8, rgb8(150, 150, 135));
      s.line((0.35, 0.0), (0.42, 1.0), 0.02, rgb8(185, 180, 165));
      tank(&mut s, 0.45, 0.52, 0.1, -PI / 2.0, true);
    }
    "Truck" => {
      scrub(&mut s, 0.7, rgb8(160, 155, 140));
      s.line((0.0, 0.85), (1.0, 0.1), 0.05, rgb8(190, 185, 170));
      truck(&mut s, 0.47, 0.6, 0.08, false);
    }
    "TruckAndAPC" | "TruckAndAPC2" => {
      scrub(&mut s, 0.9, rgb8(150, 150, 135));
      tank(&mut s, 0.25, 0.15, 0.06, 0.1, false);
      truck(&mut s, 0.65, 0.4, 0.06, false);
      truck(
        &mut s,
        if name == "TruckAndAPC" { 0.6 } else { 0.12 },
        0.85,
        0.08,
        name == "TruckAndAPC2",
      );
    }
    "CarAndAPC" | "CarAndAPC2" => {
      scrub(&mut s, 0.9, rgb8(150, 150, 135));
      tank(&mut s, 0.28, 0.25, 0.06, 0.0, false);
      tank(&mut s, 0.8, 0.1, 0.045, 0.0, false);
      car(&mut s, 0.3, 0.6, 0.08, rgb8(110, 110, 110));
    }
    "Airplane" => {
      s.rect(0.0, 0.0, 1.0, 1.0, rgb8(230, 230, 230));
      s.line((0.05, 1.0), (0.2, 0.0), 0.004, rgb8(50, 50, 50));
      s.line((0.9, 0.0), (0.9, 0.35), 0.004, rgb8(50, 50, 50));
      s.line((0.9, 0.35), (1.0, 0.35), 0.004, rgb8(50, 50, 50));
      s.line((0.8, 1.0), (0.8, 0.8), 0.004, rgb8(50, 50, 50));
      aircraft(&mut s, 0.45, 0.65, 0.25, -2.4, rgb8(200, 200, 200), 0.55);
    }
    "Airplane2" => {
      s.ground(0.0, 1.0, rgb8(220, 220, 215), 0.05, false);
      for side in [-1.0, 1.0] {
        s.line(
          (0.5 + side * 0.22, 0.35),
          (0.5 + side * 0.1, 0.75),
          0.02,
          rgb8(60, 60, 60),
        );
      }
      aircraft(&mut s, 0.52, 0.52, 0.18, -2.2, rgb8(90, 90, 90), 0.5);
    }
    "F16" => {
      s.sky(Sky::Clouds, 0.3);
      // Snowy mountain ridges.
      s.ground(0.3, 1.0, rgb8(220, 225, 235), 0.05, false);
      for k in 0..9 {
        let u = 0.05 + 0.11 * k as f64;
        s.poly(
          &[
            (u - 0.12, 1.0),
            (u, 0.45 + 0.1 * (k % 3) as f64),
            (u + 0.1, 1.0),
          ],
          rgb8(90, 80, 130),
        );
        s.poly(
          &[
            (u - 0.06, 0.75),
            (u, 0.47 + 0.1 * (k % 3) as f64),
            (u + 0.04, 0.75),
          ],
          rgb8(235, 235, 240),
        );
      }
      aircraft(&mut s, 0.5, 0.32, 0.3, 0.05, rgb8(235, 235, 235), 0.4);
      s.poly(
        &[(0.2, 0.28), (0.26, 0.15), (0.32, 0.28)],
        rgb8(180, 40, 50),
      );
    }
    "U2" => {
      s.ground(0.0, 1.0, rgb8(30, 30, 30), 0.02, false);
      aircraft(&mut s, 0.42, 0.45, 0.35, 2.1, rgb8(130, 130, 130), 0.9);
    }
    "Boat" => {
      s.sky(Sky::Overcast, 0.4);
      s.ground(0.4, 1.0, rgb8(150, 150, 140), 0.03, false);
      s.tower(0.35, 0.04, 0.28, 0.45, rgb8(220, 220, 220), Top::Dome);
      for (u, size) in [(0.2, 0.12), (0.62, 0.2), (0.9, 0.12)] {
        s.poly(
          &[
            (u - size, 0.55),
            (u + size, 0.55),
            (u + size * 0.7, 0.75),
            (u - size * 0.8, 0.72),
          ],
          rgb8(70, 70, 70),
        );
        s.line((u - size, 0.6), (u + size, 0.6), 0.008, WHITE);
        s.rect(
          u - size * 0.2,
          0.42,
          u + size * 0.3,
          0.55,
          rgb8(200, 200, 200),
        );
        s.line((u, 0.05), (u, 0.42), 0.006, rgb8(50, 50, 50));
        s.line(
          (u + size * 0.5, 0.15),
          (u + size * 0.5, 0.45),
          0.004,
          rgb8(50, 50, 50),
        );
      }
    }
    "Sailboat" => {
      s.sky(Sky::Clouds, 0.5);
      s.treeline(0.15, 0.85, 0.55, 0.12);
      s.water(0.0, 0.55, 1.0, 1.0, rgb8(110, 150, 190));
      for (u0, u1) in [(0.0, 0.18), (0.82, 1.0)] {
        for k in 0..6 {
          let u = u0 + (u1 - u0) * k as f64 / 5.0;
          s.line((u, 1.0), (u, 0.0), 0.012, rgb8(110, 60, 40));
          s.tree(u, 0.45, 0.35, rgb8(40, 80, 50));
        }
      }
      s.poly(&[(0.47, 0.72), (0.5, 0.58), (0.52, 0.72)], WHITE);
      s.rect(0.45, 0.72, 0.54, 0.75, rgb8(240, 240, 240));
    }
    "Bridge" => {
      // A white wooden footbridge across a stream, trees behind.
      s.rect(0.0, 0.0, 1.0, 1.0, rgb8(100, 100, 95));
      s.treeline(0.0, 1.0, 0.4, 0.45);
      s.water(0.0, 0.55, 1.0, 1.0, rgb8(120, 120, 115));
      for _ in 0..40 {
        let (u, v) = (s.rng.uniform(), s.rng.range(0.6, 1.0));
        s.ellipse(u, v, 0.05, 0.012, rgb8(220, 220, 220));
      }
      s.poly(
        &[(0.0, 0.5), (0.35, 0.45), (0.35, 0.65), (0.0, 0.75)],
        rgb8(70, 70, 65),
      );
      s.poly(
        &[(0.0, 0.42), (1.0, 0.36), (1.0, 0.41), (0.0, 0.48)],
        rgb8(60, 60, 55),
      );
      // Railings: two rails and posts.
      for rail in [0.25, 0.32] {
        s.line((0.0, rail + 0.1), (1.0, rail), 0.01, rgb8(240, 240, 240));
      }
      for k in 0..14 {
        let u = k as f64 / 13.0;
        let v = 0.42 - 0.06 * u;
        s.line((u, v - 0.18), (u, v), 0.008, rgb8(235, 235, 235));
      }
    }
    "Clock" => {
      s.rect(0.0, 0.0, 1.0, 1.0, rgb8(235, 235, 235));
      s.rect(0.0, 0.72, 1.0, 1.0, rgb8(210, 210, 210));
      s.rect(0.52, 0.6, 0.95, 0.72, rgb8(60, 50, 40));
      s.rect(0.55, 0.52, 0.92, 0.6, rgb8(90, 75, 60));
      s.ellipse(0.71, 0.36, 0.13, 0.16, rgb8(170, 170, 170));
      s.ellipse(0.71, 0.36, 0.11, 0.135, rgb8(25, 25, 25));
      for k in 0..12 {
        let a = TAU * k as f64 / 12.0;
        s.ellipse(
          0.71 + 0.085 * a.cos(),
          0.36 + 0.105 * a.sin(),
          0.008,
          0.01,
          WHITE,
        );
      }
      s.line((0.71, 0.36), (0.66, 0.3), 0.008, WHITE);
      s.line((0.71, 0.36), (0.76, 0.33), 0.006, WHITE);
      for side in [-1.0, 1.0] {
        s.ellipse(0.71 + side * 0.09, 0.2, 0.04, 0.035, rgb8(160, 160, 160));
      }
      s.rect(0.05, 0.6, 0.15, 0.73, rgb8(80, 80, 80));
      s.rect(0.16, 0.62, 0.26, 0.73, rgb8(90, 90, 90));
    }
    "Couple" => {
      room(&mut s);
      figure(&mut s, 0.3, 0.98, 0.85, rgb8(30, 35, 50), false);
      figure(&mut s, 0.68, 0.98, 0.75, rgb8(70, 100, 60), true);
    }
    "Couple2" => {
      room(&mut s);
      figure(&mut s, 0.32, 0.8, 0.45, rgb8(70, 70, 70), false);
      figure(&mut s, 0.52, 0.78, 0.4, rgb8(150, 150, 150), true);
    }
    "Elaine" | "Tiffany" | "Girl" | "Girl2" | "Girl3" | "Man" => {
      let (bg, skin, hair, long, top) = match name {
        "Elaine" => (
          rgb8(90, 90, 90),
          rgb8(200, 190, 180),
          rgb8(170, 150, 120),
          true,
          rgb8(200, 160, 120),
        ),
        "Tiffany" => (
          rgb8(200, 190, 160),
          rgb8(230, 180, 150),
          rgb8(230, 200, 120),
          true,
          rgb8(230, 190, 150),
        ),
        "Girl" => (
          rgb8(80, 60, 50),
          rgb8(220, 170, 140),
          rgb8(170, 120, 60),
          false,
          rgb8(160, 30, 30),
        ),
        "Girl2" => (
          rgb8(170, 170, 170),
          rgb8(225, 185, 160),
          rgb8(110, 70, 40),
          false,
          rgb8(100, 160, 130),
        ),
        "Girl3" => (
          rgb8(150, 140, 170),
          rgb8(220, 170, 140),
          rgb8(50, 30, 25),
          true,
          rgb8(230, 60, 120),
        ),
        _ => (
          rgb8(110, 110, 110),
          rgb8(190, 170, 150),
          rgb8(60, 60, 60),
          false,
          rgb8(40, 40, 40),
        ),
      };
      let n = &s.noise;
      let f = |x: f64, y: f64| {
        shade(bg, 0.85 + 0.3 * n.pfbm(x / 120.0, y / 120.0, 3, 0.5))
      };
      s.c.fill(&Paint::Fn(&f));
      let (u, v, size) = match name {
        "Tiffany" => (0.5, 0.42, 0.33),
        "Elaine" => (0.42, 0.4, 0.22),
        "Man" => (0.45, 0.25, 0.16),
        _ => (0.5, 0.4, 0.22),
      };
      portrait(&mut s, u, v, size, skin, hair, long, top);
      if name == "Elaine" {
        s.ellipse(
          u,
          v - size * 0.75,
          size * 1.2,
          size * 0.45,
          rgb8(230, 230, 225),
        );
        s.ellipse(
          u + 0.05,
          v + size * 2.0,
          size * 0.9,
          size * 0.6,
          rgb8(200, 200, 190),
        );
      }
      if name == "Man" {
        s.ellipse(u, v - size * 0.75, size * 1.0, size * 0.3, rgb8(30, 30, 30));
        for k in 0..6 {
          s.line(
            (u - size * 0.8, v - size * 0.8),
            (u - size * 2.0 + 0.03 * k as f64, v + size * 2.0),
            0.006,
            rgb8(220, 220, 220),
          );
        }
        s.ellipse(
          u + 0.05,
          v + size * 3.2,
          size * 0.5,
          size * 0.4,
          rgb8(220, 210, 200),
        );
      }
      if name == "Girl" {
        // A bouquet held at the bottom left.
        for _ in 0..40 {
          let (du, dv) = (s.rng.range(-0.15, 0.15), s.rng.range(-0.08, 0.08));
          let c = *s.rng.pick(&[
            rgb8(240, 200, 40),
            rgb8(220, 40, 30),
            rgb8(240, 150, 30),
          ]);
          s.ellipse(0.3 + du, 0.9 + dv, 0.035, 0.035, c);
        }
      }
    }
    "Flower" => {
      s.rect(0.0, 0.0, 1.0, 1.0, rgb8(40, 70, 35));
      s.treeline(0.0, 1.0, 0.5, 0.45);
      for _ in 0..14 {
        let (u, v) = (s.rng.uniform(), s.rng.range(0.4, 1.0));
        s.ellipse(u, v, 0.08, 0.04, rgb8(50, 100, 40));
      }
      for _ in 0..12 {
        let (u, v) = (s.rng.range(0.0, 0.4), s.rng.range(0.0, 0.3));
        s.ellipse(u, v, 0.012, 0.012, rgb8(220, 40, 30));
      }
      for k in 0..4 {
        let a = -PI / 2.0 + (k as f64 - 1.5) * 0.7;
        s.ellipse(
          0.55 + 0.12 * a.cos(),
          0.4 + 0.15 * a.sin(),
          0.08,
          0.07,
          rgb8(250, 220, 30),
        );
      }
      s.ellipse(0.55, 0.45, 0.06, 0.06, rgb8(240, 190, 20));
      s.line((0.55, 0.5), (0.53, 1.0), 0.015, rgb8(80, 130, 50));
    }
    "House" => {
      s.sky(Sky::Blue, 1.0);
      s.wall(0.0, 0.3, 0.75, 1.0, rgb8(160, 70, 50));
      s.wall(0.3, 0.05, 0.42, 0.4, rgb8(160, 70, 50));
      s.poly(
        &[
          (0.4, 0.4),
          (0.75, 0.15),
          (1.0, 0.35),
          (1.0, 0.45),
          (0.75, 0.25),
          (0.45, 0.47),
        ],
        rgb8(120, 70, 60),
      );
      s.poly(
        &[
          (0.6, 0.85),
          (0.9, 0.65),
          (1.0, 0.7),
          (1.0, 0.8),
          (0.6, 0.95),
        ],
        rgb8(110, 60, 50),
      );
      s.rect(0.75, 0.4, 0.95, 0.75, rgb8(160, 70, 50));
      s.rect(0.52, 0.5, 0.6, 0.65, WHITE);
      s.rect(0.62, 0.5, 0.7, 0.85, rgb8(220, 220, 220));
      s.line((0.75, 0.15), (1.0, 0.33), 0.01, WHITE);
    }
    "House2" => {
      s.sky(Sky::Blue, 0.45);
      s.treeline(0.0, 1.0, 0.45, 0.3);
      s.ground(0.6, 1.0, rgb8(110, 130, 70), 0.04, false);
      s.wall(0.3, 0.3, 0.9, 0.6, rgb8(230, 230, 225));
      s.poly(
        &[(0.25, 0.32), (0.55, 0.12), (0.95, 0.32)],
        rgb8(150, 155, 160),
      );
      s.windows(0.32, 0.36, 0.88, 0.58, 6, 1, rgb8(60, 70, 80), false);
      s.tree(0.12, 0.65, 0.4, rgb8(180, 130, 60));
      car(&mut s, 0.5, 0.88, 0.22, rgb8(60, 140, 160));
    }
    "JellyBeans" | "JellyBeans2" => {
      s.rect(0.0, 0.0, 1.0, 1.0, rgb8(225, 225, 205));
      let palette = [
        rgb8(200, 20, 30),
        rgb8(240, 200, 30),
        rgb8(40, 150, 50),
        rgb8(110, 30, 90),
        rgb8(60, 20, 40),
        rgb8(240, 140, 30),
      ];
      let (u0, v0, u1, v1) = if name == "JellyBeans" {
        (0.17, 0.3, 0.75, 0.62)
      } else {
        (0.08, 0.28, 0.92, 0.75)
      };
      let n = if name == "JellyBeans" { 140 } else { 200 };
      for _ in 0..n {
        let col = *s.rng.pick(&palette);
        let (u, v) = (s.rng.range(u0, u1), s.rng.range(v0, v1));
        let a = s.rng.range(0.0, PI);
        let (x, y) = (u * w as f64, v * h as f64);
        let r = w as f64 * 0.018;
        s.c.ellipse(
          x + 1.5,
          y + 2.0,
          r * 1.3,
          r * 0.8,
          a,
          &Paint::Solid(shade(col, 0.4)),
        );
        s.c.ellipse(x, y, r * 1.3, r * 0.8, a, &Paint::Solid(col));
        s.c.ellipse(
          x - r * 0.3,
          y - r * 0.3,
          r * 0.3,
          r * 0.15,
          a,
          &Paint::Solid(mix(col, WHITE, 0.8)),
        );
      }
    }
    "Peppers" => {
      // Peppers filling the frame: long ones lying across, bells in front.
      s.rect(0.0, 0.0, 1.0, 1.0, rgb8(120, 20, 20));
      let palette = [
        rgb8(200, 20, 20),
        rgb8(170, 15, 20),
        rgb8(70, 150, 40),
        rgb8(130, 180, 50),
        rgb8(225, 200, 60),
      ];
      for _ in 0..45 {
        let col = *s.rng.pick(&palette);
        let (x, y) =
          (s.rng.range(-0.1, 1.1) * s.w, s.rng.range(-0.1, 1.1) * s.h);
        let r = s.rng.range(0.1, 0.18) * s.w;
        let (ra, rb) = if s.rng.chance(0.4) {
          (r * 1.8, r * 0.45)
        } else {
          (r, r * 0.9)
        };
        let a = s.rng.range(0.0, PI);
        s.c.ellipse(
          x + r * 0.08,
          y + r * 0.1,
          ra,
          rb,
          a,
          &Paint::Solid(shade(col, 0.3)),
        );
        s.c.ellipse(x, y, ra, rb, a, &Paint::Solid(col));
        s.c.ellipse(
          x - r * 0.15,
          y - r * 0.2,
          ra * 0.6,
          rb * 0.5,
          a,
          &Paint::Solid(shade(col, 1.2)),
        );
        s.c.ellipse(
          x - r * 0.3,
          y - r * 0.35,
          ra * 0.12,
          rb * 0.08,
          a,
          &Paint::Solid(mix(col, WHITE, 0.8)),
        );
        if s.rng.chance(0.4) {
          s.c.line(
            (x, y - rb),
            (x + r * 0.2, y - rb - r * 0.35),
            r * 0.12,
            &Paint::Solid(rgb8(60, 110, 40)),
          );
        }
      }
    }
    "Apples" => {
      s.rect(0.0, 0.0, 1.0, 1.0, rgb8(60, 40, 30));
      heap(
        &mut s,
        &[
          rgb8(190, 25, 30),
          rgb8(150, 200, 60),
          rgb8(200, 60, 40),
          rgb8(230, 200, 70),
        ],
        70,
        (0.06, 0.075),
        1.0,
      );
    }
    "Mandrill" => return None,
    "Marruecos" => {
      s.sky(Sky::Clouds, 0.45);
      s.ground(0.45, 1.0, rgb8(200, 130, 80), 0.04, false);
      for k in 0..6 {
        let u = 0.1 + 0.17 * k as f64;
        s.poly(
          &[
            (u - 0.2, 0.75),
            (u, 0.42 + 0.05 * (k % 2) as f64),
            (u + 0.15, 0.75),
          ],
          shade(rgb8(210, 140, 80), 0.8 + 0.1 * (k % 3) as f64),
        );
      }
      s.ellipse(0.12, 0.45, 0.01, 0.01, DARK);
    }
    "Moon" => {
      // The lunar surface seen at a low sun: plains, ridges, craters.
      let n = &s.noise;
      let f = |x: f64, y: f64| {
        let g = 0.55 + 0.12 * n.pfbm(x / 120.0, y / 120.0, 5, 0.55);
        [g, g, g]
      };
      s.c.fill(&Paint::Fn(&f));
      let (wf, hf) = (w as f64, h as f64);
      for _ in 0..140 {
        let (x, y) = (s.rng.range(0.0, wf), s.rng.range(0.0, hf));
        let r = wf * 0.06 * s.rng.uniform().powi(3) + 1.5;
        s.c
          .ellipse(x - r * 0.2, y, r, r, 0.0, &Paint::Solid([0.35; 3]));
        s.c.ellipse(
          x + r * 0.15,
          y,
          r * 0.85,
          r * 0.85,
          0.0,
          &Paint::Solid([0.62; 3]),
        );
      }
      // A wrinkle ridge: a soft bright band with a dark side.
      let ridge = bezier(
        (wf * 0.45, 0.0),
        (wf * 0.35, hf * 0.35),
        (wf * 0.6, hf * 0.6),
        (wf * 0.5, hf),
        60,
      );
      let shadow: Vec<(f64, f64)> =
        ridge.iter().map(|&(x, y)| (x + wf * 0.025, y)).collect();
      s.c.polyline(&shadow, wf * 0.04, &Paint::Solid([0.45; 3]));
      s.c.polyline(&ridge, wf * 0.05, &Paint::Solid([0.68; 3]));
      s.c.blur(2);
    }
    "RadcliffeCamera" => {
      s.sky(Sky::Overcast, 0.85);
      s.facade(0.0, 0.5, 0.18, 0.85, STONE, 3, 3, true);
      s.facade(0.82, 0.5, 1.0, 0.85, STONE, 3, 3, true);
      s.rotunda(0.5, 0.5, 0.08, 0.88);
      s.ground(0.86, 1.0, GRASS, 0.05, true);
    }
    "Splash" => {
      // A milk drop's crown in a dish on a red background.
      s.rect(0.0, 0.0, 1.0, 1.0, rgb8(200, 40, 40));
      s.rect(0.0, 0.0, 1.0, 0.12, rgb8(150, 30, 35));
      s.rect(0.0, 0.0, 0.03, 1.0, rgb8(60, 20, 30));
      s.ellipse(0.5, 0.72, 0.36, 0.12, rgb8(150, 60, 40));
      s.ellipse(0.5, 0.62, 0.3, 0.1, rgb8(240, 230, 220));
      s.ellipse(0.5, 0.63, 0.22, 0.065, rgb8(200, 180, 160));
      for k in 0..16 {
        let a = TAU * k as f64 / 16.0;
        let (u, v) = (0.5 + 0.28 * a.cos(), 0.58 + 0.09 * a.sin());
        let height = 0.06 + 0.03 * (k % 3) as f64;
        s.poly(
          &[(u - 0.02, v + 0.03), (u + 0.02, v + 0.03), (u, v - height)],
          rgb8(245, 240, 235),
        );
        s.ellipse(u, v - height, 0.012, 0.016, WHITE);
      }
    }
    "Stall" => {
      s.rect(0.0, 0.0, 1.0, 1.0, rgb8(60, 40, 30));
      s.rect(0.0, 0.0, 1.0, 0.12, rgb8(150, 150, 140));
      let palette = [
        rgb8(130, 140, 40),
        rgb8(100, 120, 30),
        rgb8(200, 170, 50),
        rgb8(150, 40, 40),
        rgb8(230, 200, 60),
        rgb8(190, 120, 50),
      ];
      for k in 0..9 {
        let (u, v) = ((k % 3) as f64 * 0.35 + 0.15, (k / 3) as f64 * 0.3 + 0.3);
        let col = palette[k % palette.len()];
        let (x, y, r) = (u * w as f64, v * h as f64, w as f64 * 0.14);
        s.c.ellipse(
          x,
          y + r * 0.4,
          r * 1.1,
          r * 0.35,
          0.0,
          &Paint::Solid(rgb8(40, 30, 25)),
        );
        s.c.polygon(
          &[
            (x - r, y + r * 0.35),
            (x, y - r * 0.6),
            (x + r, y + r * 0.35),
          ],
          &Paint::Solid(col),
        );
        for _ in 0..60 {
          let (dx, dy) =
            (s.rng.range(-0.8, 0.8) * r, s.rng.range(-0.5, 0.35) * r);
          if dy > -0.6 * r + dx.abs() * 0.9 {
            s.c.circle(
              x + dx,
              y + dy,
              r * 0.06,
              &Paint::Solid(shade(col, s.rng.range(0.7, 1.3))),
            );
          }
        }
      }
    }
    "Tree" => {
      // A windswept cypress on a rocky point above the sea.
      s.sky(Sky::Blue, 0.5);
      s.water(0.0, 0.42, 1.0, 0.62, rgb8(60, 110, 170));
      s.ground(0.58, 1.0, rgb8(130, 90, 90), 0.04, false);
      for (u, v, r) in [
        (0.08, 0.55, 0.1),
        (0.2, 0.58, 0.07),
        (0.85, 0.72, 0.08),
        (0.72, 0.62, 0.05),
      ] {
        s.ellipse(u, v, r, r * 0.8, rgb8(205, 195, 185));
      }
      let trunk = |s: &mut Sketch, a: (f64, f64), b: (f64, f64), wd: f64| {
        s.line(a, b, wd, rgb8(80, 50, 60));
      };
      trunk(&mut s, (0.42, 0.85), (0.5, 0.55), 0.05);
      trunk(&mut s, (0.5, 0.55), (0.3, 0.32), 0.03);
      trunk(&mut s, (0.5, 0.55), (0.62, 0.35), 0.035);
      trunk(&mut s, (0.62, 0.35), (0.85, 0.3), 0.025);
      trunk(&mut s, (0.62, 0.35), (0.55, 0.15), 0.02);
      for (u, v, r) in [
        (0.25, 0.25, 0.13),
        (0.45, 0.12, 0.18),
        (0.68, 0.18, 0.15),
        (0.88, 0.25, 0.12),
        (0.15, 0.32, 0.09),
        (0.58, 0.3, 0.1),
      ] {
        for _ in 0..6 {
          let (du, dv) = (s.rng.range(-r, r) * 0.7, s.rng.range(-0.03, 0.03));
          s.ellipse(u + du, v + dv, r * 0.5, r * 0.18, rgb8(50, 30, 60));
        }
      }
    }
    "Volubilis" => {
      s.sky(Sky::Blue, 0.6);
      s.ground(0.6, 1.0, rgb8(200, 175, 140), 0.04, false);
      s.wall(0.25, 0.15, 0.78, 0.65, rgb8(190, 160, 120));
      for k in 0..5 {
        s.arch(
          0.31 + 0.11 * k as f64,
          0.075,
          0.3,
          0.65,
          rgb8(140, 175, 225),
        );
      }
      s.wall(0.02, 0.5, 0.2, 0.65, rgb8(180, 150, 110));
      s.person(0.05, 0.66, 0.08, rgb8(60, 60, 80));
    }
    "WalterCronkite" => {
      // A news anchor in a dark suit against wood panelling, a framed
      // picture behind his shoulder.
      let n = &s.noise;
      let f = |x: f64, y: f64| {
        shade(
          rgb8(110, 95, 85),
          0.8
            + 0.2 * ((x / 6.0).sin() * 0.5 + 0.5)
            + n.perlin(x / 3.0, y / 40.0) * 0.1,
        )
      };
      s.c.fill(&Paint::Fn(&f));
      s.rect(0.0, 0.0, 0.3, 0.3, rgb8(30, 30, 30));
      s.rect(0.03, 0.03, 0.27, 0.27, rgb8(80, 80, 80));
      portrait(
        &mut s,
        0.5,
        0.38,
        0.22,
        rgb8(190, 160, 140),
        rgb8(60, 55, 50),
        false,
        rgb8(30, 30, 40),
      );
      s.poly(
        &[(0.42, 0.62), (0.58, 0.62), (0.5, 0.95)],
        rgb8(230, 230, 230),
      );
      s.poly(
        &[(0.48, 0.66), (0.52, 0.66), (0.54, 1.0), (0.46, 1.0)],
        rgb8(90, 90, 100),
      );
    }
    "ChemicalPlant" | "ChemicalPlantClose" | "ChemicalPlantFar" => {
      // An oblique aerial view: hills on top, a freeway crossing the valley,
      // the plant's tanks and sheds in the middle, orchards below.
      let n = &s.noise;
      let (w, h) = (s.w, s.h);
      let f = |x: f64, y: f64| {
        let v = y / h;
        if v < 0.35 {
          let ridge = n.pfbm(x / 60.0, y / 25.0, 5, 0.55);
          let g = 0.45 + 0.35 * ridge - 0.1 * (0.35 - v);
          [g, g, g]
        } else {
          let rows = ((y / 3.0).sin() * 0.5 + 0.5) * 0.05;
          let g = 0.3 + rows + n.pfbm(x / 40.0, y / 40.0, 3, 0.5) * 0.12;
          [g, g, g]
        }
      };
      s.c.fill(&Paint::Fn(&f));
      let _ = w;
      s.line((0.05, 1.0), (0.6, 0.35), 0.03, rgb8(160, 160, 160));
      s.line((0.1, 1.0), (0.65, 0.35), 0.006, rgb8(90, 90, 90));
      for _ in 0..40 {
        let (u, v) = (s.rng.range(0.0, 0.45), s.rng.range(0.38, 0.6));
        s.rect(u, v, u + 0.02, v + 0.012, rgb8(200, 200, 200));
      }
      s.rect(0.48, 0.58, 0.62, 0.64, rgb8(235, 235, 235));
      for k in 0..12 {
        let u = 0.55 + 0.03 * k as f64;
        s.line((u, 0.52), (u, 0.62), 0.004, rgb8(70, 70, 70));
      }
      for (u, v) in [(0.78, 0.5), (0.82, 0.52), (0.86, 0.5)] {
        s.ellipse(u, v, 0.015, 0.012, rgb8(240, 240, 240));
      }
    }
    "ToyVehicles" => {
      // Toys on a white tabletop in front of a lake and hills.
      s.sky(Sky::Clouds, 0.15);
      s.poly(
        &[
          (0.0, 0.25),
          (0.3, 0.12),
          (0.6, 0.15),
          (1.0, 0.1),
          (1.0, 0.3),
          (0.0, 0.3),
        ],
        rgb8(140, 140, 130),
      );
      s.water(0.4, 0.28, 1.0, 0.6, rgb8(170, 170, 165));
      s.poly(
        &[(0.0, 0.28), (0.4, 0.28), (0.75, 0.6), (0.0, 0.6)],
        rgb8(210, 205, 195),
      );
      s.ground(0.6, 1.0, rgb8(215, 215, 210), 0.05, false);
      s.rect(0.88, 0.35, 0.98, 0.72, rgb8(235, 235, 230));
      s.ellipse(0.85, 0.68, 0.06, 0.05, rgb8(220, 220, 215));
      s.vehicle(0.08, 0.25, 0.18, 0.32, rgb8(240, 240, 240), 2, true);
    }
    _ => return None,
  }
  s.finish(seed);
  Some(s.c)
}

/// The 21 gray levels, as Wolfram's chart has them.
const GRAY_LEVELS: [u8; 21] = [
  0, 12, 25, 38, 51, 64, 77, 90, 103, 116, 128, 140, 152, 164, 177, 190, 203,
  216, 229, 242, 255,
];

/// `"Gray21"`: three columns of seven steps — the first darkest at the top
/// and lightening downwards, the second continuing upwards, the third
/// continuing downwards to white.
fn gray21(w: usize, h: usize) -> Canvas {
  let mut c = Canvas::new(w, h, BLACK);
  c.map(|x, y, _| {
    // Block boundaries at whole pixels: pixel i is in block ⌊3i/w⌋.
    let col = (((x - 0.5) * 3.0 / w as f64).floor() as usize).min(2);
    let row = (((y - 0.5) * 7.0 / h as f64).floor() as usize).min(6);
    let level = match col {
      0 => row,
      1 => 13 - row,
      _ => 14 + row,
    };
    [GRAY_LEVELS[level] as f64 / 255.0; 3]
  });
  c
}

/// `"Ruler"`: a fine grid with a heavier line every eighth, crossed by a
/// graduated scale along both axes.
fn ruler(w: usize, h: usize) -> Canvas {
  let (wf, hf) = (w as f64, h as f64);
  let mut c = Canvas::new(w, h, WHITE);
  let ink = Paint::Solid(BLACK);
  let step = wf / 32.0;
  let mut k = 0.0;
  while k * step <= wf {
    c.line((k * step, 0.0), (k * step, hf), 1.0, &ink);
    k += 1.0;
  }
  k = 0.0;
  while k * step <= hf {
    c.line((0.0, k * step), (wf, k * step), 1.0, &ink);
    k += 1.0;
  }
  // The scales: a cleared band with ticks, through the middle.
  let (cx, cy) = (wf * 0.53, hf * 0.53);
  c.rect(cx - 6.0, 0.0, cx + 22.0, hf, &Paint::Solid(WHITE));
  c.rect(0.0, cy - 22.0, wf, cy + 6.0, &Paint::Solid(WHITE));
  let tick = step / 8.0;
  let mut i = 0;
  while i as f64 * tick < wf {
    let len = if i % 8 == 0 {
      16.0
    } else if i % 4 == 0 {
      10.0
    } else {
      6.0
    };
    let t = i as f64 * tick;
    c.line((cx, t), (cx + len, t), 1.0, &ink);
    c.line((t, cy), (t, cy - len), 1.0, &ink);
    i += 1;
  }
  c.line((cx, 0.0), (cx, hf), 1.0, &ink);
  c.line((0.0, cy), (wf, cy), 1.0, &ink);
  c
}

/// `"ResolutionChart"`: groups of three bars, horizontal and vertical, in
/// sizes that halve from group to group, with group numbers left as gaps.
fn resolution_chart(w: usize, h: usize) -> Canvas {
  let (wf, hf) = (w as f64, h as f64);
  let mut c = Canvas::new(w, h, WHITE);
  let ink = Paint::Solid(BLACK);
  let element = |c: &mut Canvas, x: f64, y: f64, bar: f64| {
    for k in 0..3 {
      let yy = y + k as f64 * 2.0 * bar;
      c.rect(x, yy, x + 5.0 * bar, yy + bar, &ink);
      let xx = x + 6.0 * bar + k as f64 * 2.0 * bar;
      c.rect(xx, y, xx + bar, y + 5.0 * bar, &ink);
    }
  };
  // Large groups down the left, a nested smaller copy in the middle.
  let mut bar = wf / 60.0;
  let mut y = hf * 0.16;
  for _ in 0..5 {
    element(&mut c, wf * 0.12, y, bar);
    y += bar * 7.0;
    bar *= 0.89;
  }
  let mut bar = wf / 60.0;
  let mut y = hf * 0.16;
  for _ in 0..4 {
    element(&mut c, wf * 0.55, y, bar);
    y += bar * 7.0;
    bar *= 0.89;
  }
  let mut bar = wf / 220.0;
  let mut y = hf * 0.35;
  for _ in 0..6 {
    element(&mut c, wf * 0.4, y, bar);
    y += bar * 7.0;
    bar *= 0.89;
  }
  c.rect(wf * 0.42, hf * 0.16, wf * 0.5, hf * 0.24, &ink);
  element(&mut c, wf * 0.65, hf * 0.78, wf / 50.0);
  c
}
