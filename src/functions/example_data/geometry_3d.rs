//! `ExampleData[{"Geometry3D", …}]`: generated 3D models.
//!
//! Mathematical surfaces (`"Torus"`, `"KleinBottle"`, `"MoebiusStrip"`, …)
//! are sampled exactly; scanned objects are crude models made of primitives
//! (a cow is an ellipsoid body on four tube legs with a head and a tail).
//! Every model is built with exactly the recorded number of vertices and of
//! faces of each arity (see [`mesh::fit`]) and mapped onto the recorded
//! bounding box; all properties derive from that one vertex and face list.

use std::collections::BTreeMap;
use std::f64::consts::{PI, TAU};

use super::images::cached;
use super::mesh::{self, Faces, Mesh, P3};
use super::rng::{Noise, Rng};
use super::{Collection, Entry};
use crate::functions::*;

const X: P3 = [1.0, 0.0, 0.0];
const Y: P3 = [0.0, 1.0, 0.0];
const Z: P3 = [0.0, 0.0, 1.0];
const AXES: (P3, P3, P3) = (X, Y, Z);

/// Grid resolution `k · density`, at least `min`.
fn res(d: f64, k: f64, min: usize) -> usize {
  ((k * d).round() as usize).max(min)
}

/// A straight tube (frustum) from `a` to `b`.
fn limb(m: &mut Mesh, d: f64, f: Faces, a: P3, b: P3, r0: f64, r1: f64) {
  m.tube(
    res(d, 6.0, 2),
    res(d, 8.0, 3),
    f,
    |t| mesh::add(a, mesh::scale(mesh::sub(b, a), t)),
    |t| r0 + (r1 - r0) * t,
  );
}

fn blob(m: &mut Mesh, d: f64, f: Faces, c: P3, r: P3) {
  m.ellipsoid(c, r, AXES, res(d, 10.0, 3), res(d, 14.0, 3), f, |_| 0.0);
}

fn rotated_blob(
  m: &mut Mesh,
  d: f64,
  f: Faces,
  c: P3,
  r: P3,
  yaw: f64,
  pitch: f64,
) {
  let (sy, cy) = yaw.sin_cos();
  let (sp, cp) = pitch.sin_cos();
  let ax = [cy * cp, sy * cp, sp];
  let ay = [-sy, cy, 0.0];
  let az = mesh::cross(ax, ay);
  m.ellipsoid(
    c,
    r,
    (ax, ay, az),
    res(d, 8.0, 3),
    res(d, 10.0, 3),
    f,
    |_| 0.0,
  );
}

/// A four-legged animal along x, standing on z = 0.
fn quadruped(
  m: &mut Mesh,
  d: f64,
  f: Faces,
  neck_up: f64,
  horns: bool,
  frill: bool,
) {
  blob(m, d * 2.0, f, [0.0, 0.0, 1.1], [1.0, 0.42, 0.45]);
  for (x, y) in [(0.6, 0.25), (0.6, -0.25), (-0.65, 0.25), (-0.65, -0.25)] {
    limb(m, d, f, [x, y, 0.95], [x, y, 0.0], 0.13, 0.08);
  }
  let neck_top = [1.25, 0.0, 1.2 + neck_up];
  limb(m, d, f, [0.8, 0.0, 1.2], neck_top, 0.2, 0.15);
  rotated_blob(
    m,
    d * 1.4,
    f,
    mesh::add(neck_top, [0.25, 0.0, -0.05]),
    [0.35, 0.16, 0.18],
    0.0,
    -0.4,
  );
  limb(m, d, f, [-0.95, 0.0, 1.25], [-1.45, 0.0, 0.6], 0.06, 0.02);
  if horns {
    for y in [-0.12, 0.12] {
      limb(
        m,
        d,
        f,
        mesh::add(neck_top, [0.2, y, 0.1]),
        mesh::add(neck_top, [0.55, y * 1.6, 0.4]),
        0.05,
        0.0,
      );
    }
    limb(
      m,
      d,
      f,
      mesh::add(neck_top, [0.5, 0.0, 0.0]),
      mesh::add(neck_top, [0.75, 0.0, 0.15]),
      0.04,
      0.0,
    );
  }
  if frill {
    rotated_blob(
      m,
      d * 1.4,
      f,
      mesh::add(neck_top, [-0.05, 0.0, 0.15]),
      [0.08, 0.45, 0.4],
      0.0,
      0.3,
    );
  }
}

/// The model for `name` at grid density `d`.
fn model(name: &str, d: f64, f: Faces, rng_seed: u64) -> Mesh {
  let mut m = Mesh::default();
  let noise = Noise::new(rng_seed, 0);
  match name {
    "Cone" => {
      m.revolve([0.0; 3], Z, res(d, 10.0, 2), res(d, 30.0, 3), f, |t| {
        1.0 - t
      });
    }
    "Torus" => m.torus_grid(res(d, 30.0, 3), res(d, 60.0, 3), f, |u, v| {
      let (a, b) = (u * TAU, v * TAU);
      [
        (3.0 + b.cos()) * a.cos(),
        (3.0 + b.cos()) * a.sin(),
        b.sin(),
      ]
    }),
    "KleinBottle" => {
      m.torus_grid(res(d, 30.0, 3), res(d, 70.0, 3), f, |u, v| {
        // The classic bottle immersion.
        let (u, v) = (u * TAU, v * TAU);
        let r = 4.0 * (1.0 - u.cos() / 2.0);
        if u < PI {
          [
            6.0 * u.cos() * (1.0 + u.sin()) + r * u.cos() * v.cos(),
            16.0 * u.sin() + r * u.sin() * v.cos(),
            r * v.sin(),
          ]
        } else {
          [
            6.0 * u.cos() * (1.0 + u.sin()) + r * (v + PI).cos(),
            16.0 * u.sin(),
            r * v.sin(),
          ]
        }
      });
    }
    "MoebiusStrip" => {
      m.grid(res(d, 4.0, 2), res(d, 50.0, 3), true, f, |u, v| {
        let a = u * TAU;
        let s = v * 2.0 - 1.0;
        let r = 3.0 + s * (a / 2.0).cos();
        [r * a.cos(), r * a.sin(), s * (a / 2.0).sin()]
      });
    }
    "Seashell" => {
      m.grid(res(d, 20.0, 2), res(d, 40.0, 3), true, f, |u, v| {
        let (a, t) = (u * TAU, v * 3.0 * TAU);
        let s = (t / (3.0 * TAU)).exp() - 0.9;
        [
          s * t.cos() * (1.0 + a.cos()),
          s * t.sin() * (1.0 + a.cos()),
          s * a.sin() + 1.5 * s * s,
        ]
      });
    }
    "Deimos" | "Phobos" => {
      let lumpy = |p: P3| {
        noise.fbm(p[0] * 2.0 + 3.0, p[1] * 2.0 + p[2] * 1.3, 4, 0.5) * 0.18
      };
      m.ellipsoid(
        [0.0; 3],
        [1.3, 1.0, 0.9],
        AXES,
        res(d, 40.0, 3),
        res(d, 80.0, 3),
        f,
        lumpy,
      );
    }
    "Vase" => {
      m.revolve([0.0; 3], Z, res(d, 40.0, 2), res(d, 60.0, 3), f, |t| {
        0.25
          + 0.3 * (PI * t * 1.2).sin().max(0.0)
          + if t > 0.85 { (t - 0.85) * 1.5 } else { 0.0 }
      });
    }
    "Cow" => quadruped(&mut m, d, f, 0.0, false, false),
    "Horse" => quadruped(&mut m, d, f, 0.6, false, false),
    "Triceratops" => quadruped(&mut m, d, f, -0.2, true, true),
    "StanfordBunny" => {
      blob(&mut m, d * 3.0, f, [0.0, 0.0, 0.8], [0.9, 0.7, 0.75]);
      blob(&mut m, d * 2.0, f, [0.75, 0.0, 1.4], [0.45, 0.38, 0.4]);
      for y in [-0.15, 0.15] {
        rotated_blob(
          &mut m,
          d * 1.5,
          f,
          [0.6, y, 2.05],
          [0.12, 0.08, 0.5],
          0.0,
          0.25,
        );
      }
      blob(&mut m, d, f, [-0.9, 0.0, 0.7], [0.18, 0.18, 0.18]);
    }
    "Beethoven" => {
      m.revolve(
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.4],
        res(d, 8.0, 2),
        res(d, 24.0, 3),
        f,
        |t| 1.3 - 0.9 * t,
      );
      limb(&mut m, d, f, [0.0, 0.0, 0.2], [0.0, 0.0, 0.8], 0.35, 0.33);
      m.ellipsoid(
        [0.0, 0.0, 1.5],
        [0.6, 0.5, 0.75],
        AXES,
        res(d, 20.0, 3),
        res(d, 28.0, 3),
        f,
        |p| {
          // Wild hair on the upper half.
          if p[2] > -0.1 {
            0.12
              * (noise.fbm(p[0] * 4.0, p[1] * 4.0 + p[2] * 3.0, 3, 0.5) + 0.6)
          } else {
            0.0
          }
        },
      );
    }
    "HammerheadShark" => {
      m.tube(
        res(d, 20.0, 2),
        res(d, 14.0, 3),
        f,
        |t| [2.0 * t - 1.0, 0.0, 0.05 * (PI * t).sin()],
        |t| 0.25 * (PI * t.powf(0.8)).sin().max(0.02),
      );
      rotated_blob(
        &mut m,
        d * 1.5,
        f,
        [1.0, 0.0, 0.0],
        [0.12, 0.5, 0.07],
        0.0,
        0.0,
      );
      let fin = |m: &mut Mesh, origin: P3, u: P3, v: P3| {
        m.extrude(
          origin,
          u,
          v,
          &[(0.0, 0.0), (0.35, 0.0), (0.05, 0.4)],
          0.03,
          res(d, 2.0, 2),
          f,
        );
      };
      fin(&mut m, [-0.1, 0.0, 0.2], X, Z);
      fin(
        &mut m,
        [-1.0, 0.0, 0.0],
        X,
        mesh::normalize([0.0, 0.0, 1.0]),
      );
      fin(
        &mut m,
        [-1.0, 0.0, 0.0],
        X,
        mesh::normalize([0.0, 0.0, -1.0]),
      );
      fin(
        &mut m,
        [0.3, 0.2, -0.05],
        X,
        mesh::normalize([0.0, 1.0, -0.3]),
      );
      fin(
        &mut m,
        [0.3, -0.2, -0.05],
        X,
        mesh::normalize([0.0, -1.0, -0.3]),
      );
    }
    "Galleon" => {
      m.tube(
        res(d, 16.0, 2),
        res(d, 16.0, 3),
        f,
        |t| [2.0 * t - 1.0, 0.0, 0.3],
        |t| 0.35 * (PI * (0.05 + 0.9 * t)).sin(),
      );
      for (x, h) in [(-0.5, 1.6), (0.1, 2.0), (0.6, 1.5)] {
        limb(&mut m, d, f, [x, 0.0, 0.4], [x, 0.0, 0.4 + h], 0.03, 0.02);
        for k in 0..3 {
          let z = 0.6 + h * (0.12 + 0.28 * k as f64);
          m.cuboid(
            [x + 0.03, 0.0, z],
            [0.02, 0.45 - 0.08 * k as f64, h * 0.12],
            AXES,
            res(d, 3.0, 2),
            f,
          );
        }
      }
      limb(&mut m, d, f, [0.9, 0.0, 0.45], [1.4, 0.0, 0.9], 0.03, 0.01);
    }
    "SpaceShuttle" => {
      m.tube(
        res(d, 10.0, 2),
        res(d, 10.0, 3),
        f,
        |t| [2.0 * t - 1.0, 0.0, 0.0],
        |t| 0.18 * (PI * (0.5 * t + 0.02)).sin().powf(0.4).min(1.0),
      );
      m.extrude(
        [-0.9, 0.0, -0.08],
        X,
        Y,
        &[(0.0, -1.1), (0.9, -0.15), (0.9, 0.15), (0.0, 1.1)],
        0.05,
        2,
        f,
      );
      m.extrude(
        [-0.95, 0.0, 0.1],
        X,
        Z,
        &[(0.0, 0.0), (0.4, 0.0), (0.05, 0.6)],
        0.04,
        2,
        f,
      );
    }
    "SedanCar" | "UtahVWBug" => {
      // The long axis is y for these two.
      m.cuboid([0.0, 0.0, 0.45], [0.45, 1.1, 0.22], AXES, res(d, 6.0, 2), f);
      if name == "UtahVWBug" {
        m.ellipsoid(
          [0.0, -0.1, 0.65],
          [0.45, 0.9, 0.45],
          AXES,
          res(d, 10.0, 3),
          res(d, 14.0, 3),
          f,
          |_| 0.0,
        );
      } else {
        m.cuboid([0.0, -0.1, 0.8], [0.4, 0.55, 0.16], AXES, res(d, 5.0, 2), f);
      }
      for (x, y) in [(0.45f64, 0.7), (-0.45, 0.7), (0.45, -0.7), (-0.45, -0.7)]
      {
        limb(
          &mut m,
          d,
          f,
          [x - 0.06 * x.signum(), y, 0.22],
          [x + 0.06 * x.signum(), y, 0.22],
          0.22,
          0.22,
        );
      }
    }
    "Tugboat" => {
      m.tube(
        res(d, 30.0, 2),
        res(d, 30.0, 3),
        f,
        |t| [2.0 * t - 1.0, 0.0, 0.0],
        |t| 0.35 * (PI * (0.03 + 0.94 * t)).sin().powf(0.5),
      );
      m.cuboid(
        [0.1, 0.0, 0.45],
        [0.35, 0.22, 0.18],
        AXES,
        res(d, 8.0, 2),
        f,
      );
      m.cuboid([0.2, 0.0, 0.72], [0.15, 0.18, 0.1], AXES, res(d, 6.0, 2), f);
      limb(&mut m, d, f, [-0.2, 0.0, 0.5], [-0.2, 0.0, 1.0], 0.08, 0.07);
    }
    "UtahTeapot" => {
      // Body and lid as one profile closing at both poles, so the only
      // triangles are the few at the poles (the original is nearly all
      // quads); spout and handle taper to points inside the body.
      m.revolve(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.4],
        res(d, 8.0, 3),
        res(d, 12.0, 3),
        f,
        |t| {
          let p = [
            (0.0, 0.0),
            (0.03, 0.7),
            (0.35, 1.05),
            (0.68, 0.8),
            (0.72, 0.6),
            (0.9, 0.3),
            (0.93, 0.08),
            (0.97, 0.12),
            (1.0, 0.0),
          ];
          let i = p
            .iter()
            .position(|&(x, _)| x >= t)
            .unwrap_or(p.len() - 1)
            .max(1);
          let ((x0, y0), (x1, y1)) = (p[i - 1], p[i]);
          y0 + (y1 - y0) * (t - x0) / (x1 - x0).max(1e-9)
        },
      );
      m.tube(
        res(d, 4.0, 2),
        res(d, 5.0, 3),
        f,
        |t| [0.6 + 1.1 * t, 0.0, 0.3 + 0.7 * t * t],
        |t| {
          0.18
            * (1.0 - t * t)
              .max(0.0)
              .sqrt()
              .max(if t < 1.0 { 0.3 } else { 0.0 })
            * (1.0 - 0.4 * t)
        },
      );
      m.tube(
        res(d, 6.0, 2),
        res(d, 5.0, 3),
        f,
        |t| {
          let a = PI * (0.4 + 1.2 * t);
          [-0.7 + 0.55 * a.cos(), 0.0, 0.6 + 0.4 * a.sin()]
        },
        |t| if t <= 0.0 || t >= 1.0 { 0.0 } else { 0.08 },
      );
    }
    "Wrench" => {
      m.cuboid([0.0, 0.0, 0.0], [1.0, 0.06, 0.12], AXES, res(d, 8.0, 2), f);
      for (x, open) in [(-1.1f64, false), (1.1, true)] {
        m.tube(
          res(d, 16.0, 2),
          res(d, 8.0, 3),
          f,
          |t| {
            let span = if open { 1.4 * PI } else { TAU * 0.999 };
            let a = (if open { 0.8 * PI } else { 0.0 }) + span * t;
            [x + 0.25 * a.cos() * x.signum(), 0.0, 0.25 * a.sin()]
          },
          |_| 0.07,
        );
      }
    }
    "BassGuitar" => {
      let body: Vec<(f64, f64)> = (0..res(d, 24.0, 8))
        .map(|k| {
          let a = TAU * k as f64 / res(d, 24.0, 8) as f64;
          let waist = 1.0 - 0.25 * (2.0 * a).cos().powi(2);
          (0.9 * a.cos(), 0.6 * a.sin() * waist)
        })
        .collect();
      m.extrude([-1.2, 0.0, 0.0], X, Y, &body, 0.12, 2, f);
      m.cuboid([0.6, 0.0, 0.05], [1.1, 0.06, 0.03], AXES, res(d, 6.0, 2), f);
      m.cuboid(
        [1.85, 0.0, 0.05],
        [0.18, 0.1, 0.02],
        AXES,
        res(d, 3.0, 2),
        f,
      );
    }
    "CastleWall" => {
      m.cuboid([0.0, 0.0, 0.4], [2.0, 0.15, 0.4], AXES, res(d, 12.0, 2), f);
      for k in 0..12 {
        let x = -1.85 + k as f64 * 0.335;
        m.cuboid([x, 0.0, 0.88], [0.08, 0.15, 0.08], AXES, res(d, 3.0, 2), f);
      }
      for x in [-2.0, 2.0] {
        m.revolve(
          [x, 0.0, 0.0],
          [0.0, 0.0, 1.2],
          res(d, 8.0, 2),
          res(d, 16.0, 3),
          f,
          |_| 0.35,
        );
        m.revolve(
          [x, 0.0, 1.2],
          [0.0, 0.0, 0.5],
          res(d, 4.0, 2),
          res(d, 16.0, 3),
          f,
          |t| 0.42 * (1.0 - t),
        );
      }
    }
    "PottedPlant" => {
      m.revolve(
        [0.0; 3],
        [0.0, 0.0, 0.6],
        res(d, 6.0, 2),
        res(d, 20.0, 3),
        f,
        |t| 0.3 + 0.12 * t,
      );
      let mut rng = Rng::new(rng_seed);
      for _ in 0..40 {
        let a = rng.range(0.0, TAU);
        let tilt = rng.range(0.3, 1.2);
        let len = rng.range(0.5, 1.4);
        let c = [
          len * 0.5 * tilt.sin() * a.cos(),
          len * 0.5 * tilt.sin() * a.sin(),
          0.6 + len * 0.5 * tilt.cos(),
        ];
        rotated_blob(
          &mut m,
          d * 0.8,
          f,
          c,
          [len * 0.5, 0.12, 0.02],
          a,
          PI / 2.0 - tilt,
        );
      }
    }
    "Tree" => {
      limb(&mut m, d * 1.5, f, [0.0; 3], [0.0, 0.0, 1.2], 0.12, 0.08);
      let mut rng = Rng::new(rng_seed);
      for _ in 0..60 {
        let a = rng.range(0.0, TAU);
        let z = rng.range(0.8, 1.6);
        let c = [
          0.7 * rng.uniform() * a.cos(),
          0.7 * rng.uniform() * a.sin(),
          z + rng.range(0.0, 0.6),
        ];
        limb(&mut m, d * 0.5, f, [0.0, 0.0, z * 0.8], c, 0.04, 0.02);
        blob(&mut m, d * 0.6, f, c, [0.25, 0.25, 0.2]);
      }
    }
    "VikingLander" => {
      let hex: Vec<(f64, f64)> = (0..6)
        .map(|k| {
          let a = TAU * k as f64 / 6.0;
          (0.6 * a.cos(), 0.6 * a.sin())
        })
        .collect();
      m.extrude([0.0, 0.0, 0.9], X, Y, &hex, 0.3, res(d, 6.0, 2), f);
      for k in 0..3 {
        let a = TAU * k as f64 / 3.0 + 0.5;
        limb(
          &mut m,
          d * 2.0,
          f,
          [0.5 * a.cos(), 0.5 * a.sin(), 0.8],
          [1.3 * a.cos(), 1.3 * a.sin(), 0.05],
          0.05,
          0.04,
        );
        m.revolve(
          [1.3 * a.cos(), 1.3 * a.sin(), 0.0],
          [0.0, 0.0, 0.06],
          2,
          res(d, 12.0, 3),
          f,
          |_| 0.15,
        );
      }
      limb(
        &mut m,
        d * 2.0,
        f,
        [0.0, 0.0, 1.05],
        [0.0, 0.0, 1.6],
        0.03,
        0.03,
      );
      m.revolve(
        [0.0, 0.0, 1.6],
        [0.0, 0.0, 0.25],
        res(d, 8.0, 2),
        res(d, 24.0, 3),
        f,
        |t| 0.05 + 0.5 * t.sqrt(),
      );
    }
    "Zeppelin" => {
      m.ellipsoid(
        [0.0; 3],
        [1.0, 0.15, 0.15],
        AXES,
        res(d, 30.0, 3),
        res(d, 24.0, 3),
        f,
        |_| 0.0,
      );
      for (u, v) in [
        (Y, X),
        (mesh::scale(Y, -1.0), X),
        (Z, X),
        (mesh::scale(Z, -1.0), X),
      ] {
        m.extrude(
          mesh::add([-0.95, 0.0, 0.0], mesh::scale(u, 0.05)),
          v,
          u,
          &[(0.0, 0.0), (0.25, 0.0), (0.0, 0.18)],
          0.01,
          2,
          f,
        );
      }
      m.cuboid(
        [0.2, 0.0, -0.17],
        [0.12, 0.03, 0.03],
        AXES,
        res(d, 3.0, 2),
        f,
      );
    }
    _ => blob(&mut m, d * 3.0, f, [0.0; 3], [1.0, 1.0, 1.0]),
  }
  m
}

/// `@FaceSizes`: `{{arity, count}, …}`.
fn face_targets(entry: &Entry) -> BTreeMap<usize, usize> {
  let mut out = BTreeMap::new();
  if let Some(Expr::List(ref t)) = entry.stat("@FaceSizes") {
    for pair in t {
      if let Expr::List(p) = pair
        && let (Some(Expr::Integer(k)), Some(Expr::Integer(n))) =
          (p.first(), p.get(1))
      {
        out.insert(*k as usize, *n as usize);
      }
    }
  }
  out
}

/// The arities in the order the original lists its polygon groups.
fn face_order(entry: &Entry) -> Vec<usize> {
  match entry.stat("@FaceSizes") {
    Some(Expr::List(ref t)) => t
      .iter()
      .filter_map(|p| match p {
        Expr::List(p) => match p.first() {
          Some(Expr::Integer(k)) => Some(*k as usize),
          _ => None,
        },
        _ => None,
      })
      .collect(),
    _ => Vec::new(),
  }
}

/// The vertex count, from the `"VertexData"` skeleton `Arr[{n, 3}, …]`.
fn vertex_target(entry: &Entry) -> usize {
  match entry.skeleton("VertexData") {
    Some(Expr::FunctionCall { ref args, .. }) => match args.first() {
      Some(Expr::List(d)) => match d.first() {
        Some(Expr::Integer(n)) => *n as usize,
        _ => 0,
      },
      _ => 0,
    },
    _ => 0,
  }
}

/// The finished mesh of an entry: faces grouped by arity in the recorded
/// order, vertices on the recorded bounding box.
fn build(entry: &Entry) -> Mesh {
  let targets = face_targets(entry);
  let main = targets
    .iter()
    .max_by_key(|(_, n)| **n)
    .map_or(3, |(k, _)| *k);
  let faces = if main == 4 {
    Faces::Quads
  } else {
    Faces::Triangles
  };
  let seed = super::rng::hash(&["Geometry3D", entry.name, "mesh"]);
  let mut m = mesh::fit(
    |d| model(entry.name, d, faces, seed),
    vertex_target(entry),
    &targets,
  );
  if let Some(Expr::List(ref bb)) = entry.exact("BoundingBox") {
    let mut lo = [0.0; 3];
    let mut hi = [1.0; 3];
    for (k, range) in bb.iter().take(3).enumerate() {
      let r = super::images::numbers(range);
      if r.len() == 2 {
        lo[k] = r[0];
        hi[k] = r[1];
      }
    }
    m.fit_box(lo, hi);
  }
  let order = face_order(entry);
  m.faces.sort_by_key(|f| {
    order
      .iter()
      .position(|&k| k == f.len())
      .unwrap_or(usize::MAX)
  });
  m
}

fn point(p: &P3) -> Expr {
  Expr::List(vec![Expr::Real(p[0]), Expr::Real(p[1]), Expr::Real(p[2])].into())
}

fn index_list(f: &[usize]) -> Expr {
  Expr::List(
    f.iter()
      .map(|&i| Expr::Integer(i as i128 + 1))
      .collect::<Vec<_>>()
      .into(),
  )
}

/// One `Polygon[{faces…}]` per arity, in face order.
fn polygon_groups(m: &Mesh) -> Vec<Expr> {
  let mut groups: Vec<(usize, Vec<Expr>)> = Vec::new();
  for f in &m.faces {
    match groups.last_mut() {
      Some((k, g)) if *k == f.len() => g.push(index_list(f)),
      _ => groups.push((f.len(), vec![index_list(f)])),
    }
  }
  groups
    .into_iter()
    .map(|(_, g)| call("Polygon", vec![Expr::List(g.into())]))
    .collect()
}

fn vertex_list(m: &Mesh) -> Expr {
  Expr::List(m.vertices.iter().map(point).collect::<Vec<_>>().into())
}

fn has_normals(entry: &Entry) -> bool {
  entry.skeleton("VertexNormals").is_some()
}

fn graphics_complex(entry: &Entry, m: &Mesh, with_normals: bool) -> Expr {
  let mut args = vec![vertex_list(m), Expr::List(polygon_groups(m).into())];
  if with_normals && has_normals(entry) {
    args.push(rule_expr(
      id_expr("VertexNormals"),
      Expr::List(m.normals().iter().map(point).collect::<Vec<_>>().into()),
    ));
  }
  call("GraphicsComplex", args)
}

pub struct Geometry3D;

impl Collection for Geometry3D {
  fn value(&self, entry: &Entry, property: Option<&str>) -> Option<Expr> {
    let property = property?;
    let key = format!("Geometry3D/{}/{property}", entry.name);
    let mesh = || build(entry);
    Some(match property {
      "VertexData" => cached(key, || vertex_list(&mesh())),
      "PolygonData" => cached(key, || {
        Expr::List(
          mesh()
            .faces
            .iter()
            .map(|f| index_list(f))
            .collect::<Vec<_>>()
            .into(),
        )
      }),
      "PolygonObjects" => cached(key, || {
        let m = mesh();
        Expr::List(
          m.faces
            .iter()
            .map(|f| {
              call(
                "Polygon",
                vec![Expr::List(
                  f.iter()
                    .map(|&i| point(&m.vertices[i]))
                    .collect::<Vec<_>>()
                    .into(),
                )],
              )
            })
            .collect::<Vec<_>>()
            .into(),
        )
      }),
      "VertexNormals" => {
        if !has_normals(entry) {
          return None;
        }
        cached(key, || {
          Expr::List(
            mesh()
              .normals()
              .iter()
              .map(point)
              .collect::<Vec<_>>()
              .into(),
          )
        })
      }
      "GraphicsComplex" => {
        cached(key, || graphics_complex(entry, &mesh(), false))
      }
      "Graphics3D" => cached(key, || {
        let m = mesh();
        let mut args = vec![Expr::List(
          vec![call0("EdgeForm"), graphics_complex(entry, &m, true)].into(),
        )];
        if let Some(Expr::List(ref opts)) = entry.stat("@Graphics3DOptions") {
          args.extend(opts.iter().cloned());
        }
        call("Graphics3D", args)
      }),
      "MeshRegion" | "Region" => cached(key, || {
        let m = mesh();
        call(
          "MeshRegion",
          vec![vertex_list(&m), Expr::List(polygon_groups(&m).into())],
        )
      }),
      "BoundaryMeshRegion" => {
        entry.skeleton("BoundaryMeshRegion")?;
        cached(key, || {
          let m = mesh();
          call(
            "BoundaryMeshRegion",
            vec![vertex_list(&m), Expr::List(polygon_groups(&m).into())],
          )
        })
      }
      _ => return None,
    })
  }
}
