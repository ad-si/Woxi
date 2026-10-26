//! Crude drawings standing in for Wolfram's test photographs, aerial views
//! and textures. Every scene is chosen from the entry name and drawn with
//! the shared [`raster`](super::raster) primitives plus noise; the caller
//! then matches the recorded per-channel mean and spread, so histograms and
//! thresholds behave like the original's.

use std::f64::consts::{PI, TAU};

use super::raster::*;
use super::rng::{Noise, Rng, hash};

/// Draw the scene for `collection`/`name` at `w × h` pixels.
pub fn draw(collection: &str, name: &str, w: usize, h: usize) -> Canvas {
  let mut rng = Rng::for_entry(collection, name, "scene");
  let seed = hash(&[collection, name, "noise"]);
  match collection {
    "Texture" | "ColorTexture" => {
      super::textures::texture(name, w, h, seed, &mut rng)
    }
    "AerialImage" => super::aerials::draw(name, w, h, seed, &mut rng),
    _ => test_image(name, w, h, seed, &mut rng),
  }
}

/// Which family of picture a name stands for.
fn base(name: &str) -> &str {
  name.trim_end_matches(|c: char| c.is_ascii_digit() || c == '_')
}

fn test_image(
  name: &str,
  w: usize,
  h: usize,
  seed: u64,
  rng: &mut Rng,
) -> Canvas {
  if let Some(c) = super::testimages::draw(name, w, h, seed) {
    return c;
  }
  let (wf, hf) = (w as f64, h as f64);
  let noise = Noise::new(seed, 0);
  let b = base(name);
  let mut c = match b {
    "Couple" | "Girl" | "Elaine" | "Tiffany" | "Man" | "WalterCronkite" => {
      portrait(b, w, h, rng, &noise)
    }
    "Mandrill" | "Baboon" => mandrill(w, h, rng, &noise),
    "Peppers" | "Apples" | "JellyBeans" => produce(b, w, h, rng),
    "Boat" | "Sailboat" => boat(b, w, h, rng, &noise),
    "Airplane" | "F" | "U" => aircraft(w, h, rng, &noise),
    "Tank" | "Truck" | "APC" | "CarAndAPC" | "TruckAndAPC" | "ToyVehicles" => {
      vehicles(b, w, h, rng, &noise)
    }
    "Aerial" | "Airport" | "ChemicalPlant" | "ChemicalPlantClose"
    | "ChemicalPlantFar" => {
      super::aerials::draw(name, w, h, rng.next_u64(), rng)
    }
    "Oxford" | "Lisbon" => super::photos::photo(name, w, h, rng.next_u64()),
    "House" | "Stall" | "RadcliffeCamera" | "Bridge" | "Volubilis"
    | "Marruecos" => building(b, w, h, rng, &noise),
    "Moon" => moon(w, h, rng, &noise),
    "Clock" => clock(w, h, rng),
    "Tree" => tree(w, h, rng, &noise),
    "Flower" => flower(w, h, rng),
    "Splash" => splash(w, h, rng),
    "Gray" => gray_steps(w, h),
    "ResolutionChart" => resolution_chart(w, h),
    "Ruler" => ruler(w, h),
    _ => landscape(w, h, rng, &noise),
  };
  // A little texture everywhere, like a photograph.
  if !matches!(b, "Gray" | "ResolutionChart" | "Ruler") {
    let grain = Noise::new(seed ^ 0x5eed, 0);
    c.grain(&grain, (wf.min(hf) / 90.0).max(1.5), 0.03);
  }
  c
}

// ---- people -------------------------------------------------------------

const SKIN: [Rgb; 4] = [
  rgb8(224, 172, 140),
  rgb8(198, 140, 110),
  rgb8(150, 100, 75),
  rgb8(236, 190, 160),
];
const HAIR: [Rgb; 4] = [
  rgb8(40, 28, 20),
  rgb8(110, 70, 35),
  rgb8(200, 160, 90),
  rgb8(20, 18, 18),
];

fn portrait(
  kind: &str,
  w: usize,
  h: usize,
  rng: &mut Rng,
  noise: &Noise,
) -> Canvas {
  let (wf, hf) = (w as f64, h as f64);
  let wall = mix(rgb8(90, 110, 140), rgb8(200, 180, 150), rng.uniform());
  let mut c = Canvas::new(w, h, wall);
  c.fill(&Paint::Linear {
    p0: (0.0, 0.0),
    c0: shade(wall, 1.15),
    p1: (wf, hf),
    c1: shade(wall, 0.7),
  });
  // A window or picture frame in the background.
  if rng.chance(0.6) {
    let x0 = rng.range(0.05, 0.6) * wf;
    let y0 = rng.range(0.05, 0.25) * hf;
    c.rect(
      x0,
      y0,
      x0 + 0.3 * wf,
      y0 + 0.35 * hf,
      &Paint::Solid(shade(wall, 1.3)),
    );
    c.rect(
      x0 + 0.02 * wf,
      y0 + 0.02 * hf,
      x0 + 0.28 * wf,
      y0 + 0.33 * hf,
      &Paint::Solid(rgb8(170, 200, 230)),
    );
  }
  let people = if kind == "Couple" { 2 } else { 1 };
  for p in 0..people {
    let cx = if people == 1 {
      wf * rng.range(0.42, 0.58)
    } else {
      wf * (0.3 + 0.4 * p as f64)
    };
    let s = hf * if people == 1 { 0.27 } else { 0.2 };
    let long_hair = matches!(kind, "Girl" | "Elaine" | "Tiffany")
      || (kind == "Couple" && p == 1);
    person(&mut c, cx, hf * 0.45, s, long_hair, rng, noise);
  }
  if kind == "Elaine" || kind == "Tiffany" {
    // A hat brim.
    let cx = wf * 0.5;
    c.ellipse(
      cx,
      hf * 0.22,
      wf * 0.3,
      hf * 0.05,
      -0.1,
      &Paint::Solid(rgb8(120, 60, 50)),
    );
  }
  c
}

fn person(
  c: &mut Canvas,
  cx: f64,
  cy: f64,
  s: f64,
  long_hair: bool,
  rng: &mut Rng,
  noise: &Noise,
) {
  let skin = *rng.pick(&SKIN);
  let hair = *rng.pick(&HAIR);
  let shirt = [rng.uniform(), rng.uniform(), rng.uniform()];
  // Shoulders and torso.
  c.ellipse(
    cx,
    cy + 1.9 * s,
    1.5 * s,
    1.1 * s,
    0.0,
    &Paint::Linear {
      p0: (cx, cy + s),
      c0: shirt,
      p1: (cx, cy + 3.0 * s),
      c1: shade(shirt, 0.5),
    },
  );
  c.rect(
    cx - 0.25 * s,
    cy + 0.6 * s,
    cx + 0.25 * s,
    cy + 1.0 * s,
    &Paint::Solid(shade(skin, 0.85)),
  );
  if long_hair {
    c.ellipse(
      cx,
      cy + 0.4 * s,
      0.95 * s,
      1.35 * s,
      0.0,
      &Paint::Solid(hair),
    );
  }
  // Head with shading.
  let shading = |x: f64, y: f64| {
    let n = noise.fbm(x / (0.3 * s), y / (0.3 * s), 3, 0.5) * 0.05;
    let d = ((x - cx + 0.3 * s).powi(2) + (y - cy + 0.3 * s).powi(2)).sqrt()
      / (1.6 * s);
    shade(skin, 1.15 - 0.45 * d + n)
  };
  c.ellipse(cx, cy, 0.72 * s, 0.95 * s, 0.0, &Paint::Fn(&shading));
  // Hair on top.
  c.ellipse(
    cx,
    cy - 0.55 * s,
    0.78 * s,
    0.5 * s,
    0.0,
    &Paint::Solid(hair),
  );
  // Eyes, brows, nose and mouth.
  for side in [-1.0, 1.0] {
    let ex = cx + side * 0.28 * s;
    c.ellipse(
      ex,
      cy - 0.05 * s,
      0.13 * s,
      0.07 * s,
      0.0,
      &Paint::Solid(WHITE),
    );
    c.circle(ex, cy - 0.05 * s, 0.06 * s, &Paint::Solid(rgb8(60, 40, 30)));
    c.line(
      (ex - 0.14 * s, cy - 0.2 * s),
      (ex + 0.14 * s, cy - 0.22 * s),
      0.05 * s,
      &Paint::Solid(shade(hair, 0.8)),
    );
  }
  c.line(
    (cx, cy),
    (cx - 0.05 * s, cy + 0.3 * s),
    0.05 * s,
    &Paint::Solid(shade(skin, 0.7)),
  );
  let mouth = bezier(
    (cx - 0.22 * s, cy + 0.48 * s),
    (cx - 0.1 * s, cy + 0.56 * s),
    (cx + 0.1 * s, cy + 0.56 * s),
    (cx + 0.22 * s, cy + 0.48 * s),
    12,
  );
  c.polyline(&mouth, 0.06 * s, &Paint::Solid(rgb8(150, 60, 60)));
}

fn mandrill(w: usize, h: usize, rng: &mut Rng, noise: &Noise) -> Canvas {
  let (wf, hf) = (w as f64, h as f64);
  let mut c = Canvas::new(w, h, BLACK);
  // Fur: streaky noise in olive and brown.
  let fur = |x: f64, y: f64| {
    let n = noise.fbm(x / 6.0, y / 40.0, 5, 0.6);
    let t = noise.turbulence(x / 60.0, y / 60.0, 4);
    mix(
      rgb8(60, 55, 25),
      rgb8(190, 170, 90),
      0.5 + 0.8 * n * (0.6 + t),
    )
  };
  c.fill(&Paint::Fn(&fur));
  let (cx, cy) = (wf * 0.5, hf * 0.55);
  // Blue ridged cheeks.
  for side in [-1.0, 1.0] {
    c.ellipse(
      cx + side * 0.17 * wf,
      cy + 0.05 * hf,
      0.11 * wf,
      0.3 * hf,
      side * 0.12,
      &Paint::Linear {
        p0: (cx + side * 0.08 * wf, cy),
        c0: rgb8(70, 110, 200),
        p1: (cx + side * 0.28 * wf, cy),
        c1: rgb8(130, 160, 220),
      },
    );
    for k in 0..5 {
      let x = cx + side * (0.11 + 0.025 * k as f64) * wf;
      c.line(
        (x, cy - 0.15 * hf),
        (x + side * 0.02 * wf, cy + 0.25 * hf),
        0.006 * wf,
        &Paint::Solid(rgb8(40, 60, 120)),
      );
    }
  }
  // The long red nose.
  c.ellipse(
    cx,
    cy + 0.02 * hf,
    0.07 * wf,
    0.33 * hf,
    0.0,
    &Paint::Linear {
      p0: (cx - 0.07 * wf, cy),
      c0: rgb8(200, 40, 40),
      p1: (cx + 0.07 * wf, cy),
      c1: rgb8(240, 90, 70),
    },
  );
  c.ellipse(
    cx,
    cy + 0.33 * hf,
    0.09 * wf,
    0.05 * hf,
    0.0,
    &Paint::Solid(rgb8(220, 70, 60)),
  );
  // Yellow eyes.
  for side in [-1.0, 1.0] {
    let ex = cx + side * 0.12 * wf;
    let ey = cy - 0.3 * hf;
    c.ellipse(
      ex,
      ey,
      0.04 * wf,
      0.03 * hf,
      0.0,
      &Paint::Solid(rgb8(230, 180, 40)),
    );
    c.circle(ex, ey, 0.015 * wf, &Paint::Solid(BLACK));
  }
  // Whiskers.
  for _ in 0..30 {
    let side = if rng.chance(0.5) { -1.0 } else { 1.0 };
    let x = cx + side * rng.range(0.25, 0.45) * wf;
    let y = cy + rng.range(0.0, 0.4) * hf;
    c.line(
      (x, y),
      (
        x + side * rng.range(0.03, 0.08) * wf,
        y + rng.range(0.02, 0.1) * hf,
      ),
      1.0,
      &Paint::Solid(rgb8(235, 230, 210)),
    );
  }
  c
}

fn produce(kind: &str, w: usize, h: usize, rng: &mut Rng) -> Canvas {
  let (wf, hf) = (w as f64, h as f64);
  let (bg, palette, size, n): (Rgb, Vec<Rgb>, f64, usize) = match kind {
    "JellyBeans" => (
      rgb8(230, 230, 235),
      vec![
        rgb8(220, 30, 40),
        rgb8(250, 200, 30),
        rgb8(40, 160, 60),
        rgb8(240, 120, 30),
        rgb8(130, 40, 150),
        rgb8(250, 250, 240),
        rgb8(240, 110, 160),
        rgb8(60, 40, 30),
      ],
      0.07,
      26,
    ),
    "Apples" => (
      rgb8(60, 40, 30),
      vec![
        rgb8(190, 30, 30),
        rgb8(150, 190, 50),
        rgb8(210, 60, 40),
        rgb8(230, 200, 60),
      ],
      0.17,
      9,
    ),
    _ => (
      rgb8(40, 30, 25),
      vec![
        rgb8(200, 20, 20),
        rgb8(40, 140, 40),
        rgb8(230, 190, 30),
        rgb8(160, 20, 30),
        rgb8(30, 100, 30),
      ],
      0.2,
      12,
    ),
  };
  let mut c = Canvas::new(w, h, bg);
  for _ in 0..n {
    let col = *rng.pick(&palette);
    let x = rng.range(-0.05, 1.05) * wf;
    let y = rng.range(-0.05, 1.05) * hf;
    let r = size * wf.min(hf) * rng.range(0.7, 1.3);
    let (rx, ry, a) = match kind {
      "JellyBeans" => (r * 1.4, r * 0.85, rng.range(0.0, PI)),
      "Apples" => (r, r * 0.92, 0.0),
      _ => (r * 1.1, r * rng.range(0.6, 1.4), rng.range(0.0, PI)),
    };
    let shading = |px: f64, py: f64| {
      let d = ((px - x + 0.3 * rx).powi(2) + (py - y + 0.3 * ry).powi(2))
        .sqrt()
        / (1.6 * rx.max(ry));
      shade(col, 1.25 - 0.7 * d)
    };
    // A shadow, the body, then a highlight.
    c.ellipse(
      x + 0.08 * r,
      y + 0.1 * r,
      rx,
      ry,
      a,
      &Paint::Solid(shade(bg, 0.5)),
    );
    c.ellipse(x, y, rx, ry, a, &Paint::Fn(&shading));
    c.ellipse(
      x - 0.35 * rx,
      y - 0.4 * ry,
      0.18 * rx,
      0.1 * ry,
      a,
      &Paint::Solid(mix(col, WHITE, 0.85)),
    );
    if kind != "JellyBeans" {
      c.line(
        (x, y - ry * 0.9),
        (x + 0.1 * r, y - ry * 1.2),
        0.08 * r,
        &Paint::Solid(rgb8(60, 110, 40)),
      );
    }
  }
  c
}

/// Sky over water or land, as a backdrop.
fn backdrop(
  w: usize,
  h: usize,
  horizon: f64,
  water: bool,
  noise: &Noise,
) -> Canvas {
  let hf = h as f64;
  let hy = horizon * hf;
  let mut c = Canvas::new(w, h, BLACK);
  let sky = |x: f64, y: f64| {
    if y < hy {
      let cloud = (noise.fbm(x / 80.0, y / 30.0, 4, 0.5) * 1.5).max(0.0);
      mix(
        mix(rgb8(110, 160, 220), rgb8(200, 220, 240), y / hy),
        WHITE,
        cloud,
      )
    } else if water {
      let wave = noise.fbm(x / 60.0, (y - hy) / 6.0, 3, 0.5) * 0.15;
      shade(
        mix(rgb8(60, 110, 140), rgb8(20, 50, 70), (y - hy) / (hf - hy)),
        1.0 + wave,
      )
    } else {
      let g = noise.fbm(x / 25.0, y / 25.0, 4, 0.5) * 0.25;
      shade(
        mix(rgb8(120, 140, 70), rgb8(80, 90, 50), (y - hy) / (hf - hy)),
        1.0 + g,
      )
    }
  };
  c.fill(&Paint::Fn(&sky));
  c
}

fn boat(
  kind: &str,
  w: usize,
  h: usize,
  rng: &mut Rng,
  noise: &Noise,
) -> Canvas {
  let (wf, hf) = (w as f64, h as f64);
  let mut c = backdrop(w, h, rng.range(0.45, 0.6), true, noise);
  let n = if kind == "Boat" { 3 } else { 2 };
  for i in 0..n {
    let s = wf * rng.range(0.15, 0.3) / (1.0 + i as f64 * 0.5);
    let x = wf * rng.range(0.2, 0.8);
    let y = hf * rng.range(0.6, 0.8);
    let hull = [
      (x - s, y),
      (x + s, y),
      (x + 0.75 * s, y + 0.25 * s),
      (x - 0.8 * s, y + 0.25 * s),
    ];
    c.polygon(
      &hull,
      &Paint::Solid(*rng.pick(&[
        rgb8(240, 240, 235),
        rgb8(150, 40, 30),
        rgb8(30, 40, 70),
      ])),
    );
    c.line(
      (x, y),
      (x, y - 1.6 * s),
      0.03 * s,
      &Paint::Solid(rgb8(80, 60, 40)),
    );
    c.polygon(
      &[
        (x + 0.03 * s, y - 1.55 * s),
        (x + 0.03 * s, y - 0.1 * s),
        (x + 0.9 * s, y - 0.1 * s),
      ],
      &Paint::Solid(rgb8(245, 245, 240)),
    );
    c.polygon(
      &[
        (x - 0.03 * s, y - 1.3 * s),
        (x - 0.03 * s, y - 0.1 * s),
        (x - 0.6 * s, y - 0.1 * s),
      ],
      &Paint::Solid(rgb8(230, 225, 215)),
    );
    // Reflection.
    c.polygon_alpha(
      &[
        (x - s, y + 0.25 * s),
        (x + s, y + 0.25 * s),
        (x + 0.6 * s, y + 0.7 * s),
        (x - 0.6 * s, y + 0.7 * s),
      ],
      &Paint::Solid(BLACK),
      0.15,
    );
  }
  c
}

fn aircraft(w: usize, h: usize, rng: &mut Rng, noise: &Noise) -> Canvas {
  let (wf, hf) = (w as f64, h as f64);
  let mut c = backdrop(w, h, rng.range(0.75, 1.1), false, noise);
  let (cx, cy) = (wf * rng.range(0.4, 0.6), hf * rng.range(0.4, 0.55));
  let s = wf.min(hf) * 0.42;
  let a = rng.range(-0.3, 0.3);
  let rot = |x: f64, y: f64| {
    (
      cx + x * a.cos() - y * a.sin(),
      cy + x * a.sin() + y * a.cos(),
    )
  };
  let body = rgb8(200, 200, 205);
  let pts = |v: &[(f64, f64)]| {
    v.iter()
      .map(|&(x, y)| rot(x * s, y * s))
      .collect::<Vec<_>>()
  };
  // Wings, tail, fuselage, cockpit.
  c.polygon(
    &pts(&[(-0.15, -0.02), (0.15, -0.02), (0.05, -0.75), (-0.05, -0.75)]),
    &Paint::Solid(shade(body, 0.85)),
  );
  c.polygon(
    &pts(&[(-0.15, 0.02), (0.15, 0.02), (0.05, 0.75), (-0.05, 0.75)]),
    &Paint::Solid(shade(body, 0.8)),
  );
  c.polygon(
    &pts(&[(-0.95, -0.02), (-0.75, -0.02), (-0.85, -0.3), (-0.95, -0.3)]),
    &Paint::Solid(shade(body, 0.75)),
  );
  c.polygon(
    &pts(&[(-0.95, 0.02), (-0.75, 0.02), (-0.85, 0.25), (-0.95, 0.25)]),
    &Paint::Solid(shade(body, 0.75)),
  );
  c.ellipse(
    cx,
    cy,
    s,
    0.09 * s,
    a,
    &Paint::Linear {
      p0: rot(0.0, -0.09 * s),
      c0: shade(body, 1.15),
      p1: rot(0.0, 0.09 * s),
      c1: shade(body, 0.6),
    },
  );
  let (kx, ky) = rot(0.6 * s, -0.02 * s);
  c.ellipse(
    kx,
    ky,
    0.15 * s,
    0.04 * s,
    a,
    &Paint::Solid(rgb8(40, 50, 70)),
  );
  c
}

fn vehicles(
  kind: &str,
  w: usize,
  h: usize,
  rng: &mut Rng,
  noise: &Noise,
) -> Canvas {
  let (wf, hf) = (w as f64, h as f64);
  let mut c = backdrop(w, h, rng.range(0.3, 0.5), false, noise);
  let n = if kind.contains("And") || kind == "ToyVehicles" {
    2
  } else {
    1
  };
  for i in 0..n {
    let s = wf * 0.3 / n as f64;
    let x = wf * (0.3 + 0.4 * i as f64) + rng.range(-0.05, 0.05) * wf;
    let y = hf * rng.range(0.6, 0.75);
    let olive = mix(rgb8(90, 95, 60), rgb8(130, 120, 80), rng.uniform());
    let tank =
      kind == "Tank" || (kind.contains("APC") && i == 1) || kind == "APC";
    c.ellipse(
      x,
      y + 0.35 * s,
      1.2 * s,
      0.12 * s,
      0.0,
      &Paint::Solid(shade(olive, 0.4)),
    );
    c.rect(
      x - s,
      y - 0.25 * s,
      x + s,
      y + 0.25 * s,
      &Paint::Solid(olive),
    );
    if tank {
      c.rect(
        x - 0.45 * s,
        y - 0.55 * s,
        x + 0.35 * s,
        y - 0.25 * s,
        &Paint::Solid(shade(olive, 1.1)),
      );
      c.line(
        (x + 0.3 * s, y - 0.42 * s),
        (x + 1.4 * s, y - 0.5 * s),
        0.08 * s,
        &Paint::Solid(shade(olive, 0.8)),
      );
    } else {
      c.rect(
        x + 0.3 * s,
        y - 0.7 * s,
        x + s,
        y - 0.2 * s,
        &Paint::Solid(shade(olive, 1.15)),
      );
      c.rect(
        x + 0.45 * s,
        y - 0.6 * s,
        x + 0.9 * s,
        y - 0.4 * s,
        &Paint::Solid(rgb8(60, 80, 100)),
      );
    }
    for k in 0..5 {
      let wx = x - 0.8 * s + 0.4 * s * k as f64;
      c.circle(wx, y + 0.3 * s, 0.17 * s, &Paint::Solid(rgb8(30, 30, 30)));
      c.circle(wx, y + 0.3 * s, 0.07 * s, &Paint::Solid(rgb8(90, 90, 90)));
    }
  }
  c
}

fn building(
  kind: &str,
  w: usize,
  h: usize,
  rng: &mut Rng,
  noise: &Noise,
) -> Canvas {
  let (wf, hf) = (w as f64, h as f64);
  let horizon = rng.range(0.55, 0.75);
  let mut c = backdrop(w, h, horizon, kind == "Bridge", noise);
  let gy = horizon * hf;
  let stone = mix(rgb8(200, 180, 140), rgb8(170, 150, 130), rng.uniform());
  let stone_tex = |x: f64, y: f64| {
    shade(stone, 1.0 + noise.fbm(x / 8.0, y / 8.0, 3, 0.5) * 0.15)
  };
  match kind {
    "Bridge" => {
      c.rect(
        0.0,
        gy - 0.25 * hf,
        wf,
        gy - 0.15 * hf,
        &Paint::Fn(&stone_tex),
      );
      let arches = 4;
      for k in 0..arches {
        let ax = wf * (k as f64 + 0.5) / arches as f64;
        c.rect(
          ax - 0.03 * wf + wf / arches as f64 / 2.0,
          gy - 0.16 * hf,
          ax + 0.03 * wf + wf / arches as f64 / 2.0,
          gy + 0.05 * hf,
          &Paint::Fn(&stone_tex),
        );
        c.ring(
          ax,
          gy,
          wf / arches as f64 / 2.0 - 0.03 * wf,
          0.03 * hf,
          &Paint::Fn(&stone_tex),
        );
      }
    }
    "RadcliffeCamera" => {
      let cx = wf * 0.5;
      c.rect(
        cx - 0.3 * wf,
        gy - 0.35 * hf,
        cx + 0.3 * wf,
        gy,
        &Paint::Fn(&stone_tex),
      );
      c.ellipse(
        cx,
        gy - 0.35 * hf,
        0.22 * wf,
        0.2 * hf,
        0.0,
        &Paint::Solid(rgb8(110, 120, 110)),
      );
      c.rect(
        cx - 0.03 * wf,
        gy - 0.62 * hf,
        cx + 0.03 * wf,
        gy - 0.5 * hf,
        &Paint::Fn(&stone_tex),
      );
      for k in 0..9 {
        let x = cx - 0.27 * wf + 0.0675 * wf * k as f64;
        c.rect(
          x - 0.01 * wf,
          gy - 0.32 * hf,
          x + 0.01 * wf,
          gy - 0.05 * hf,
          &Paint::Solid(shade(stone, 0.7)),
        );
      }
    }
    "Volubilis" => {
      for k in 0..6 {
        let x = wf * (0.15 + 0.14 * k as f64);
        let top = gy - hf * rng.range(0.2, 0.45);
        c.rect(
          x - 0.025 * wf,
          top,
          x + 0.025 * wf,
          gy,
          &Paint::Fn(&stone_tex),
        );
        c.rect(
          x - 0.04 * wf,
          top - 0.02 * hf,
          x + 0.04 * wf,
          top,
          &Paint::Fn(&stone_tex),
        );
      }
      c.rect(
        0.12 * wf,
        gy - 0.5 * hf,
        0.55 * wf,
        gy - 0.45 * hf,
        &Paint::Fn(&stone_tex),
      );
    }
    "Marruecos" => {
      c.rect(0.0, gy - 0.6 * hf, wf, gy, &Paint::Fn(&stone_tex));
      for k in 0..4 {
        let x = wf * (0.15 + 0.23 * k as f64);
        c.rect(
          x - 0.07 * wf,
          gy - 0.35 * hf,
          x + 0.07 * wf,
          gy,
          &Paint::Solid(shade(stone, 0.35)),
        );
        c.ellipse(
          x,
          gy - 0.35 * hf,
          0.07 * wf,
          0.07 * wf,
          0.0,
          &Paint::Solid(shade(stone, 0.35)),
        );
      }
    }
    "Stall" => {
      c.rect(
        0.15 * wf,
        gy - 0.3 * hf,
        0.85 * wf,
        gy,
        &Paint::Solid(rgb8(120, 80, 50)),
      );
      for k in 0..10 {
        let x0 = 0.1 * wf + 0.08 * wf * k as f64;
        let col = if k % 2 == 0 {
          rgb8(200, 40, 40)
        } else {
          rgb8(240, 240, 230)
        };
        c.polygon(
          &[
            (x0, gy - 0.45 * hf),
            (x0 + 0.08 * wf, gy - 0.45 * hf),
            (x0 + 0.08 * wf, gy - 0.32 * hf),
            (x0, gy - 0.32 * hf),
          ],
          &Paint::Solid(col),
        );
      }
      for _ in 0..40 {
        let col = [rng.uniform(), rng.uniform(), rng.uniform() * 0.5];
        c.circle(
          rng.range(0.18, 0.82) * wf,
          gy - rng.range(0.05, 0.25) * hf,
          0.015 * wf,
          &Paint::Solid(col),
        );
      }
    }
    _ => {
      // A house: walls, roof, windows and a door.
      let x0 = wf * rng.range(0.15, 0.3);
      let x1 = wf * rng.range(0.7, 0.85);
      let top = gy - hf * rng.range(0.3, 0.4);
      let wall = mix(rgb8(230, 220, 200), rgb8(170, 70, 60), rng.uniform());
      c.rect(x0, top, x1, gy, &Paint::Solid(wall));
      c.polygon(
        &[
          (x0 - 0.03 * wf, top),
          (x1 + 0.03 * wf, top),
          (f64::midpoint(x0, x1), top - 0.2 * hf),
        ],
        &Paint::Solid(rgb8(90, 50, 40)),
      );
      for k in 0..3 {
        let wx = x0 + (x1 - x0) * (0.2 + 0.3 * k as f64);
        c.rect(
          wx - 0.04 * wf,
          top + 0.06 * hf,
          wx + 0.04 * wf,
          top + 0.15 * hf,
          &Paint::Solid(rgb8(70, 90, 120)),
        );
        c.rect(
          wx - 0.003 * wf,
          top + 0.06 * hf,
          wx + 0.003 * wf,
          top + 0.15 * hf,
          &Paint::Solid(WHITE),
        );
      }
      let dx = f64::midpoint(x0, x1);
      c.rect(
        dx - 0.04 * wf,
        gy - 0.12 * hf,
        dx + 0.04 * wf,
        gy,
        &Paint::Solid(rgb8(80, 50, 30)),
      );
    }
  }
  c
}

fn moon(w: usize, h: usize, rng: &mut Rng, noise: &Noise) -> Canvas {
  let (wf, hf) = (w as f64, h as f64);
  let mut c = Canvas::new(w, h, BLACK);
  let (cx, cy, r) = (wf * 0.5, hf * 0.5, wf.min(hf) * 0.45);
  let surface = |x: f64, y: f64| {
    let n = noise.fbm(x / (r / 3.0), y / (r / 3.0), 5, 0.55);
    let maria = (noise.fbm(x / r + 7.0, y / r, 3, 0.5) > 0.1) as i32 as f64;
    let d = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() / r;
    let g = (0.7 + 0.2 * n - 0.25 * maria) * (1.0 - 0.35 * d * d);
    [g, g, g]
  };
  c.circle(cx, cy, r, &Paint::Fn(&surface));
  for _ in 0..60 {
    let a = rng.range(0.0, TAU);
    let d = r * rng.uniform().sqrt() * 0.95;
    let (x, y) = (cx + d * a.cos(), cy + d * a.sin());
    let cr = r * 0.12 * rng.uniform().powi(3) + 1.0;
    c.circle(x, y, cr, &Paint::Solid([0.45; 3]));
    c.ring(x, y, cr, (cr * 0.25).max(1.0), &Paint::Solid([0.85; 3]));
  }
  c
}

fn clock(w: usize, h: usize, rng: &mut Rng) -> Canvas {
  let (wf, hf) = (w as f64, h as f64);
  let mut c = Canvas::new(w, h, rgb8(150, 130, 110));
  let (cx, cy, r) = (wf * 0.5, hf * 0.5, wf.min(hf) * 0.4);
  c.circle(cx, cy, r * 1.1, &Paint::Solid(rgb8(60, 50, 40)));
  c.circle(
    cx,
    cy,
    r,
    &Paint::Radial {
      center: (cx - 0.3 * r, cy - 0.3 * r),
      radius: 1.6 * r,
      c0: rgb8(250, 248, 240),
      c1: rgb8(200, 195, 180),
    },
  );
  for k in 0..60 {
    let a = TAU * k as f64 / 60.0;
    let (inner, width) = if k % 5 == 0 {
      (0.8, 0.035)
    } else {
      (0.9, 0.012)
    };
    c.line(
      (cx + inner * r * a.cos(), cy + inner * r * a.sin()),
      (cx + 0.95 * r * a.cos(), cy + 0.95 * r * a.sin()),
      width * r,
      &Paint::Solid(BLACK),
    );
  }
  let hour = rng.range(0.0, TAU);
  let minute = rng.range(0.0, TAU);
  for (a, len, width) in [(hour, 0.5, 0.06), (minute, 0.78, 0.035)] {
    c.line(
      (cx, cy),
      (
        cx + len * r * (a - PI / 2.0).cos(),
        cy + len * r * (a - PI / 2.0).sin(),
      ),
      width * r,
      &Paint::Solid(BLACK),
    );
  }
  c.circle(cx, cy, 0.04 * r, &Paint::Solid(BLACK));
  c
}

fn tree(w: usize, h: usize, rng: &mut Rng, noise: &Noise) -> Canvas {
  let (wf, hf) = (w as f64, h as f64);
  let mut c = backdrop(w, h, 0.75, false, noise);
  let (bx, by) = (wf * 0.5, hf * 0.82);
  let bark = rgb8(90, 60, 40);
  c.polygon(
    &[
      (bx - 0.05 * wf, by),
      (bx + 0.05 * wf, by),
      (bx + 0.025 * wf, by - 0.4 * hf),
      (bx - 0.025 * wf, by - 0.4 * hf),
    ],
    &Paint::Solid(bark),
  );
  let mut branches =
    vec![((bx, by - 0.4 * hf), -PI / 2.0, 0.22 * hf, 0.035 * wf)];
  let mut tips = Vec::new();
  for _ in 0..4 {
    let mut next = Vec::new();
    for ((x, y), a, len, width) in branches {
      for da in [-0.5, 0.45] {
        let na: f64 = a + da + rng.range(-0.2, 0.2);
        let end = (x + len * na.cos(), y + len * na.sin());
        c.line((x, y), end, width, &Paint::Solid(bark));
        next.push((end, na, len * 0.72, width * 0.65));
      }
    }
    tips = next.iter().map(|b| b.0).collect();
    branches = next;
  }
  for (x, y) in tips {
    let g = mix(rgb8(40, 100, 30), rgb8(110, 160, 50), rng.uniform());
    c.ellipse(
      x,
      y,
      rng.range(0.04, 0.08) * wf,
      rng.range(0.03, 0.06) * hf,
      rng.range(0.0, PI),
      &Paint::Solid(g),
    );
  }
  c
}

fn flower(w: usize, h: usize, rng: &mut Rng) -> Canvas {
  let (wf, hf) = (w as f64, h as f64);
  let mut c = Canvas::new(w, h, rgb8(40, 80, 40));
  c.fill(&Paint::Radial {
    center: (wf * 0.5, hf * 0.5),
    radius: wf.max(hf) * 0.8,
    c0: rgb8(80, 130, 60),
    c1: rgb8(20, 40, 20),
  });
  let (cx, cy, r) = (wf * 0.5, hf * 0.48, wf.min(hf) * 0.38);
  let petal = mix(rgb8(240, 200, 40), rgb8(220, 60, 120), rng.uniform());
  let n = 8 + rng.below(6);
  for k in 0..n {
    let a = TAU * k as f64 / n as f64;
    c.ellipse(
      cx + 0.55 * r * a.cos(),
      cy + 0.55 * r * a.sin(),
      0.5 * r,
      0.18 * r,
      a,
      &Paint::Radial {
        center: (cx, cy),
        radius: r,
        c0: shade(petal, 0.7),
        c1: mix(petal, WHITE, 0.3),
      },
    );
  }
  c.circle(cx, cy, 0.22 * r, &Paint::Solid(rgb8(120, 70, 20)));
  for k in 0..40 {
    let a = k as f64 * 2.4;
    let d = 0.2 * r * (k as f64 / 40.0).sqrt();
    c.circle(
      cx + d * a.cos(),
      cy + d * a.sin(),
      0.02 * r,
      &Paint::Solid(rgb8(60, 30, 10)),
    );
  }
  c
}

fn splash(w: usize, h: usize, rng: &mut Rng) -> Canvas {
  let (wf, hf) = (w as f64, h as f64);
  let mut c = Canvas::new(w, h, rgb8(30, 20, 40));
  let liquid = mix(rgb8(200, 30, 40), rgb8(230, 200, 40), rng.uniform());
  let surface = hf * 0.65;
  c.rect(
    0.0,
    surface,
    wf,
    hf,
    &Paint::Linear {
      p0: (0.0, surface),
      c0: liquid,
      p1: (0.0, hf),
      c1: shade(liquid, 0.4),
    },
  );
  for k in 0..6 {
    let rx = wf * (0.1 + 0.08 * k as f64);
    c.ring(
      wf * 0.5,
      surface + 0.05 * hf,
      rx,
      2.0,
      &Paint::Solid(mix(liquid, WHITE, 0.4)),
    );
  }
  let crown = star(
    wf * 0.5,
    surface - 0.05 * hf,
    wf * 0.22,
    Some(wf * 0.1),
    14,
    -PI / 2.0,
  );
  c.polygon(&crown, &Paint::Solid(liquid));
  c.ellipse(
    wf * 0.5,
    surface,
    wf * 0.2,
    hf * 0.06,
    0.0,
    &Paint::Solid(shade(liquid, 0.7)),
  );
  for _ in 0..14 {
    let a = rng.range(PI * 1.1, PI * 1.9);
    let d = wf * rng.range(0.25, 0.4);
    c.circle(
      wf * 0.5 + d * a.cos(),
      surface + d * a.sin() * 0.8,
      wf * rng.range(0.006, 0.015),
      &Paint::Solid(liquid),
    );
  }
  c
}

fn landscape(w: usize, h: usize, rng: &mut Rng, noise: &Noise) -> Canvas {
  let mut c = backdrop(w, h, rng.range(0.4, 0.6), rng.chance(0.3), noise);
  let (wf, hf) = (w as f64, h as f64);
  let mut hills = vec![(0.0, hf * 0.6)];
  for k in 0..=12 {
    hills.push((
      wf * k as f64 / 12.0,
      hf * (0.45 + 0.1 * noise.at(k as f64 * 0.7, 3.0)),
    ));
  }
  hills.push((wf, hf * 0.6));
  c.polygon_alpha(&hills, &Paint::Solid(rgb8(90, 110, 80)), 0.7);
  c
}

/// 21 gray steps from black to white.
fn gray_steps(w: usize, h: usize) -> Canvas {
  let mut c = Canvas::new(w, h, BLACK);
  c.map(|x, _, _| {
    let k = (x / w as f64 * 21.0).floor().min(20.0);
    [k / 20.0; 3]
  });
  c
}

/// Converging line bundles and bars, the elements of a resolution chart.
fn resolution_chart(w: usize, h: usize) -> Canvas {
  let (wf, hf) = (w as f64, h as f64);
  let mut c = Canvas::new(w, h, WHITE);
  let ink = Paint::Solid(BLACK);
  for k in 0..24 {
    let a = PI * k as f64 / 24.0;
    c.line(
      (wf * 0.5, hf * 0.5),
      (
        wf * 0.5 + 0.45 * wf * a.cos(),
        hf * 0.5 - 0.45 * hf * a.sin(),
      ),
      1.0,
      &ink,
    );
  }
  for k in 0..6 {
    let s = 2.0 + k as f64 * 2.0;
    let y0 = hf * 0.62 + k as f64 * hf * 0.06;
    for j in 0..5 {
      let x = wf * 0.08 + j as f64 * 2.0 * s;
      c.rect(x, y0, x + s, y0 + hf * 0.05, &ink);
    }
  }
  c.ring(wf * 0.75, hf * 0.8, wf * 0.08, 2.0, &ink);
  c.rect(wf * 0.02, hf * 0.02, wf * 0.98, hf * 0.04, &ink);
  c
}

/// A ruler: graduated ticks along an edge.
fn ruler(w: usize, h: usize) -> Canvas {
  let (wf, hf) = (w as f64, h as f64);
  let mut c = Canvas::new(w, h, rgb8(235, 225, 190));
  let ink = Paint::Solid(BLACK);
  let step = (wf / 100.0).max(3.0);
  let mut k = 0;
  let mut x = step;
  while x < wf - step / 2.0 {
    let len = if k % 10 == 9 {
      0.45
    } else if k % 5 == 4 {
      0.3
    } else {
      0.18
    };
    c.line((x, 0.0), (x, len * hf), 1.0, &ink);
    x += step;
    k += 1;
  }
  c
}
