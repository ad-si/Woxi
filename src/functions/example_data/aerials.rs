//! Top-down terrain for the aerial photographs (`"AerialImage"` and the
//! aerial `"TestImage"` entries): water on the side where the original has
//! its coast or bay, fine street grids where it shows towns, field
//! patchworks for farmland, shaded hills, runways and bridges where the
//! original has them. One layout per entry.

use std::f64::consts::{PI, TAU};

use super::raster::*;
use super::rng::{Noise, Rng};

/// The layout of one aerial view.
#[derive(Clone, Copy)]
struct Layout {
  /// Water: the side it lies on (direction it extends towards, radians;
  /// 0 = right, π/2 = down) and how much of the frame it covers.
  water: Option<(f64, f64)>,
  /// Share of the land covered by town.
  urban: f64,
  /// Field patchwork on the remaining land.
  fields: bool,
  /// Relief strength of hills.
  hills: f64,
  /// Runways: centre (fractions) and direction.
  runway: Option<(f64, f64, f64)>,
  /// A bridge across the water: from and to (fractions).
  bridge: Option<((f64, f64), (f64, f64))>,
  /// Street grid angle.
  street_angle: f64,
}

const fn lay(
  water: Option<(f64, f64)>,
  urban: f64,
  fields: bool,
  hills: f64,
) -> Layout {
  Layout {
    water,
    urban,
    fields,
    hills,
    runway: None,
    bridge: None,
    street_angle: 0.0,
  }
}

fn layout(name: &str) -> Layout {
  match name {
    "FosterCity" => Layout {
      street_angle: -0.6,
      ..lay(Some((-0.8, 0.55)), 0.5, false, 0.2)
    },
    "Oakland" => Layout {
      street_angle: 0.2,
      ..lay(Some((-1.0, 0.12)), 0.8, false, 0.8)
    },
    "Oakland2" => Layout {
      runway: Some((0.3, 0.45, 0.25)),
      ..lay(Some((PI, 0.45)), 0.6, false, 0.1)
    },
    "Richmond" => lay(Some((PI, 0.35)), 0.7, false, 0.6),
    "Richmond2" => lay(Some((-1.2, 0.6)), 0.3, false, 0.6),
    "SanDiego" | "SanDiego11" => Layout {
      runway: Some((0.3, 0.7, -0.6)),
      street_angle: -0.5,
      ..lay(None, 0.3, false, 1.0)
    },
    "SanDiego10" => lay(Some((0.7, 0.12)), 0.3, false, 1.0),
    "SanDiego12" => lay(Some((0.3, 0.3)), 0.4, false, 1.0),
    "SanDiego13" => lay(Some((PI / 2.0, 0.55)), 0.4, false, 0.4),
    "SanDiego2" => Layout {
      street_angle: 0.4,
      ..lay(Some((-2.4, 0.1)), 0.85, false, 0.3)
    },
    "SanDiego3" => Layout {
      runway: Some((0.5, 0.45, -1.2)),
      ..lay(Some((-2.4, 0.3)), 0.3, false, 0.1)
    },
    "SanDiego4" => Layout {
      street_angle: 0.0,
      ..lay(None, 0.95, false, 0.2)
    },
    "SanDiego5" => lay(Some((0.0, 0.45)), 0.2, false, 1.0),
    "SanDiego6" => lay(Some((PI / 2.0, 0.35)), 0.7, false, 0.3),
    "SanDiego7" => lay(None, 0.75, false, 0.9),
    "SanDiego8" => lay(Some((-2.5, 0.3)), 0.4, false, 1.0),
    "SanDiego9" => lay(Some((0.6, 0.6)), 0.6, false, 0.3),
    "SanFrancisco" => Layout {
      bridge: Some(((0.42, 0.45), (0.38, 0.75))),
      ..lay(Some((0.0, 0.6)), 0.4, false, 0.8)
    },
    "SanFrancisco2" => Layout {
      bridge: Some(((0.25, 0.65), (0.8, 0.0))),
      ..lay(Some((0.0, 0.8)), 0.9, false, 0.1)
    },
    "SanFrancisco3" => lay(Some((-0.6, 0.5)), 0.6, false, 0.5),
    "SanFrancisco4" => Layout {
      bridge: Some(((0.15, 0.35), (0.35, 0.3))),
      ..lay(Some((PI, 0.55)), 0.7, false, 0.3)
    },
    "SanFrancisco5" => Layout {
      bridge: Some(((0.15, 0.3), (0.4, 0.05))),
      ..lay(Some((-PI / 2.0, 0.45)), 0.95, false, 0.2)
    },
    "SanFrancisco6" => lay(Some((-PI / 2.0, 0.4)), 0.95, false, 0.1),
    "SanFrancisco7" => Layout {
      street_angle: 0.15,
      ..lay(None, 0.85, false, 0.5)
    },
    "SanFrancisco8" => lay(Some((PI, 0.08)), 0.85, false, 0.3),
    "SanFrancisco9" => lay(Some((PI, 0.5)), 0.75, false, 0.2),
    "Shreveport" => lay(Some((0.0, 0.05)), 0.35, true, 0.1),
    "Stockton" | "Stockton3" | "Stockton4" | "Stockton6" => Layout {
      street_angle: 0.4,
      ..lay(None, 0.25, true, 0.0)
    },
    "Stockton2" => lay(None, 0.2, true, 0.0),
    "Stockton5" => lay(Some((-1.0, 0.12)), 0.1, true, 0.0),
    "WashingtonIR" => lay(None, 0.5, true, 0.3),
    "WoodlandHills" => lay(None, 0.4, false, 1.2),
    // TestImage aerials.
    "Aerial" => Layout {
      street_angle: -0.6,
      ..lay(Some((-0.9, 0.12)), 0.85, false, 0.2)
    },
    "Aerial2" => lay(None, 0.6, false, 0.1),
    "Airport" => lay(None, 0.6, false, 0.0),
    "ChemicalPlant" | "ChemicalPlantClose" | "ChemicalPlantFar" => Layout {
      street_angle: 0.5,
      ..lay(None, 0.25, true, 0.3)
    },
    _ => lay(None, 0.5, false, 0.3),
  }
}

/// Draw the aerial view `name` at `w × h`.
pub fn draw(
  name: &str,
  w: usize,
  h: usize,
  seed: u64,
  rng: &mut Rng,
) -> Canvas {
  let (wf, hf) = (w as f64, h as f64);
  let noise = Noise::new(seed, 0);
  let noise2 = Noise::new(seed ^ 0x55, 0);
  if name == "Earth" {
    return earth(w, h, &noise, &noise2);
  }
  let lay = if name.starts_with("ChemicalPlant") {
    layout("ChemicalPlant")
  } else {
    layout(name)
  };
  let scale = wf.min(hf) / 512.0;
  // Patchwork of fields: a rotated grid of cells with their own tone.
  let field_angle = lay.street_angle + 0.05;
  let (fs, fc) = field_angle.sin_cos();
  let cell = 22.0 * scale;
  let block = 5.0 * scale;
  let (ss, sc) = lay.street_angle.sin_cos();
  let mut c = Canvas::new(w, h, BLACK);
  c.fill(&Paint::Fn(&|x, y| {
    let (u, v) = (x / wf, y / hf);
    // Water.
    if let Some((dir, extent)) = lay.water {
      let d = (u - 0.5) * dir.cos() + (v - 0.5) * dir.sin();
      let shore = 0.5 - extent
        + noise.pfbm(x / (180.0 * scale), y / (180.0 * scale), 4, 0.55) * 0.12;
      if d > shore {
        return shade(
          rgb8(70, 95, 110),
          0.9 + noise2.pfbm(x / 50.0, y / 6.0, 3, 0.5) * 0.08
            - (d - shore) * 0.2,
        );
      }
    }
    // Hills: shaded relief.
    let hh = |x: f64, y: f64| {
      noise.pfbm(x / (140.0 * scale), y / (140.0 * scale), 6, 0.55)
    };
    let relief =
      (hh(x + 2.0, y) - hh(x - 2.0, y) - (hh(x, y + 2.0) - hh(x, y - 2.0)))
        * 12.0
        * lay.hills;
    let base = if lay.fields {
      let (fu, fv) = (x * fc + y * fs, -x * fs + y * fc);
      // Columns of uneven width, rows of uneven height.
      let fu2 = fu / cell + 0.4 * noise2.perlin(fu / (cell * 5.0), 0.5);
      let fv2 = fv / (cell * 0.8) + 0.4 * noise2.perlin(1.5, fv / (cell * 4.0));
      let (i, j) = (fu2.floor(), fv2.floor());
      let t = noise2.perlin(i * 1.7 + 0.3, j * 2.9 + 0.7) * 0.6;
      let edge = fu2.rem_euclid(1.0) < 0.04 || fv2.rem_euclid(1.0) < 0.05;
      let speckle = noise.perlin(x / 2.0, y / 2.0) * 0.06;
      if edge {
        rgb8(150, 145, 130)
      } else {
        shade(
          mix(rgb8(100, 115, 80), rgb8(165, 155, 130), 0.5 + t),
          1.0 + speckle,
        )
      }
    } else {
      mix(
        rgb8(110, 115, 80),
        rgb8(170, 155, 125),
        0.5 + noise2.pfbm(x / 60.0, y / 60.0, 4, 0.5),
      )
    };
    let mut col = shade(base, 1.0 + relief);
    // Town: a fine grid of streets and roofs where the urban mask is on.
    let town =
      noise.pfbm(x / (220.0 * scale) + 40.0, y / (220.0 * scale), 3, 0.5)
        + lay.urban
        - 0.5;
    if town > 0.0 {
      let (su, sv) = (x * sc + y * ss, -x * ss + y * sc);
      // Blocks vary in size and tone; some are parks.
      let bu = su / block;
      let bv = sv / (block * 1.6);
      // Arterials every sixth street are wider.
      let arterial =
        (bu / 6.0).rem_euclid(1.0) < 0.06 || (bv / 6.0).rem_euclid(1.0) < 0.05;
      let street =
        arterial || bu.rem_euclid(1.0) < 0.16 || bv.rem_euclid(1.0) < 0.12;
      let lot = noise2.perlin(bu.floor() * 0.37, bv.floor() * 0.53);
      let roof = 0.8 + noise2.perlin(su / 2.0, sv / 2.0) * 0.35 + lot * 0.2;
      let block_col = if lot > 0.45 {
        rgb8(100, 120, 80)
      } else {
        shade(mix(rgb8(160, 155, 150), base, 0.35), roof)
      };
      let street_col = rgb8(175, 175, 170);
      // Soft edge of town.
      col = mix(
        col,
        if street { street_col } else { block_col },
        (town * 8.0).min(1.0),
      );
    }
    col
  }));
  if let Some((u, v, a)) = lay.runway {
    for da in [0.0, 0.9] {
      let (dx, dy) = ((a + da).cos() * 0.25 * wf, (a + da).sin() * 0.25 * wf);
      let (x, y) = (u * wf, v * hf);
      c.line(
        (x - dx, y - dy),
        (x + dx, y + dy),
        9.0 * scale,
        &Paint::Solid(rgb8(225, 225, 220)),
      );
      c.line(
        (x - dx * 0.95, y - dy * 0.95),
        (x + dx * 0.95, y + dy * 0.95),
        1.0,
        &Paint::Solid(rgb8(120, 120, 120)),
      );
    }
  }
  if let Some(((u0, v0), (u1, v1))) = lay.bridge {
    c.line(
      (u0 * wf, v0 * hf),
      (u1 * wf, v1 * hf),
      2.5 * scale,
      &Paint::Solid(rgb8(200, 200, 200)),
    );
  }
  match name {
    "Pentagon" => pentagon(&mut c, wf, hf),
    "Airport" => {
      // Two terminals with gates around an oval, and a parking grid.
      for (x, y) in [(0.25, 0.3), (0.25, 0.75)] {
        c.ellipse(
          x * wf,
          y * hf,
          0.08 * wf,
          0.12 * hf,
          0.0,
          &Paint::Solid(rgb8(200, 200, 200)),
        );
        for k in 0..8 {
          let a = TAU * k as f64 / 8.0;
          c.line(
            (x * wf, y * hf),
            ((x + 0.12 * a.cos()) * wf, (y + 0.16 * a.sin()) * hf),
            3.0 * scale,
            &Paint::Solid(rgb8(210, 210, 210)),
          );
        }
      }
      c.rect(
        0.45 * wf,
        0.05 * hf,
        0.85 * wf,
        0.95 * hf,
        &Paint::Fn(&|x, y| {
          if (x / (6.0 * scale)).rem_euclid(1.0) < 0.3 {
            rgb8(60, 60, 60)
          } else {
            shade(
              rgb8(150, 150, 150),
              0.8 + noise.perlin(x / 2.0, y / 2.0) * 0.3,
            )
          }
        }),
      );
    }
    _ if name.starts_with("ChemicalPlant") => {
      for _ in 0..12 {
        let (x, y) = (rng.range(0.35, 0.75) * wf, rng.range(0.5, 0.8) * hf);
        let r = rng.range(0.008, 0.02) * wf;
        c.circle(x, y, r, &Paint::Solid(rgb8(230, 230, 230)));
      }
      c.rect(
        0.4 * wf,
        0.62 * hf,
        0.55 * wf,
        0.68 * hf,
        &Paint::Solid(rgb8(240, 240, 240)),
      );
      for k in 0..20 {
        let x = rng.range(0.45, 0.8) * wf;
        c.line(
          (x, 0.55 * hf),
          (x + 3.0, 0.65 * hf + k as f64 * 0.3),
          1.5,
          &Paint::Solid(rgb8(60, 60, 60)),
        );
      }
    }
    _ => {}
  }
  c
}

fn pentagon(c: &mut Canvas, wf: f64, hf: f64) {
  let (cx, cy, r) = (wf * 0.45, hf * 0.45, wf.min(hf) * 0.33);
  let ring = |c: &mut Canvas, rr: f64, col: Rgb| {
    c.polygon(
      &star(cx, cy, rr, None, 5, -PI / 2.0 + 0.2),
      &Paint::Solid(col),
    );
  };
  ring(c, r * 1.05, rgb8(120, 120, 120));
  for k in 0..5 {
    let t = k as f64 / 5.0;
    ring(c, r * (1.0 - 0.12 * t), rgb8(210, 210, 205));
    ring(c, r * (0.94 - 0.12 * t), rgb8(90, 90, 90));
  }
  ring(c, r * 0.36, rgb8(120, 140, 110));
  // Spokes between the rings.
  for k in 0..10 {
    let a = -PI / 2.0 + 0.2 + TAU * k as f64 / 10.0;
    c.line(
      (cx + 0.36 * r * a.cos(), cy + 0.36 * r * a.sin()),
      (cx + r * a.cos(), cy + r * a.sin()),
      2.0,
      &Paint::Solid(rgb8(210, 210, 205)),
    );
  }
  // Parking lots below.
  c.rect(
    0.05 * wf,
    0.82 * hf,
    0.6 * wf,
    0.98 * hf,
    &Paint::Fn(&|x, _| {
      if (x / 5.0).rem_euclid(1.0) < 0.4 {
        rgb8(70, 70, 70)
      } else {
        rgb8(170, 170, 170)
      }
    }),
  );
}

/// The Earth from space: a large globe partly out of frame, clouds swirling.
fn earth(w: usize, h: usize, noise: &Noise, noise2: &Noise) -> Canvas {
  let (wf, hf) = (w as f64, h as f64);
  let mut c = Canvas::new(w, h, rgb8(30, 10, 40));
  let (cx, cy, r) = (wf * 0.62, hf * 0.62, wf.min(hf) * 0.75);
  let globe = |x: f64, y: f64| {
    let d = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() / r;
    let land = noise.pfbm(x / (r / 3.0), y / (r / 3.0), 5, 0.55);
    let base = if land > 0.15 {
      mix(rgb8(150, 110, 90), rgb8(210, 180, 150), land)
    } else {
      rgb8(60, 40, 120)
    };
    // Cloud bands curling around the globe.
    let a = (y - cy).atan2(x - cx);
    let swirl =
      noise2.pfbm(a * 3.0 + d * 4.0, d * 12.0 + (a * 2.0).sin(), 5, 0.55);
    let cloud = ((swirl + 0.05) * 2.0).clamp(0.0, 1.0);
    shade(mix(base, WHITE, cloud), 1.05 - 0.5 * d.powi(4))
  };
  c.circle(cx, cy, r, &Paint::Fn(&globe));
  c
}
