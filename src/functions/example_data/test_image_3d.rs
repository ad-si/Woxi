//! `ExampleData[{"TestImage3D", …}]`: generated volumes.
//!
//! Each volume is a phantom made of nested ellipsoids, cylinders and boxes
//! of different intensities — a skull and brain for the head scans, bones
//! for the knee, an engine block, electron orbitals — plus noise, rescaled
//! to the recorded mean and spread. Volumes are `Image3D[{slice, …}]` stacks
//! of packed 2D images, front to back.

use super::images::{cached, image_parts, numbers, resized, shape_of};
use super::rng::Noise;
use super::{Collection, Entry};
use crate::functions::*;
use crate::syntax::ImageType;

/// Signed "inside" test helpers in normalized coordinates `[-1, 1]^3`.
fn ellipsoid(p: [f64; 3], c: [f64; 3], r: [f64; 3]) -> bool {
  ((p[0] - c[0]) / r[0]).powi(2)
    + ((p[1] - c[1]) / r[1]).powi(2)
    + ((p[2] - c[2]) / r[2]).powi(2)
    <= 1.0
}

fn cylinder_z(p: [f64; 3], c: [f64; 2], r: f64, z0: f64, z1: f64) -> bool {
  (p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) <= r * r
    && p[2] >= z0
    && p[2] <= z1
}

/// Density in `[0, 1]` at `p` for the phantom `name`.
fn density(name: &str, p: [f64; 3], noise: &Noise) -> f64 {
  let n = noise.fbm(p[0] * 6.0 + p[2] * 3.1, p[1] * 6.0 - p[2] * 2.3, 3, 0.5);
  match name {
    "CThead" => {
      if !ellipsoid(p, [0.0; 3], [0.72, 0.88, 0.95]) {
        0.0
      } else if !ellipsoid(p, [0.0; 3], [0.64, 0.8, 0.87]) {
        0.95
      } else if ellipsoid(p, [0.0, -0.1, 0.1], [0.1, 0.25, 0.2]) {
        0.15
      } else {
        0.42 + 0.04 * n
      }
    }
    "MRbrain" => {
      if !ellipsoid(p, [0.0; 3], [0.7, 0.86, 0.9]) {
        0.02
      } else if !ellipsoid(p, [0.0; 3], [0.62, 0.78, 0.82]) {
        0.25
      } else if ellipsoid(p, [-0.12, 0.0, 0.1], [0.07, 0.3, 0.15])
        || ellipsoid(p, [0.12, 0.0, 0.1], [0.07, 0.3, 0.15])
      {
        0.08
      } else {
        // Gray and white matter with folds.
        let folds = (n * 9.0).sin();
        if folds > 0.3 { 0.7 } else { 0.5 }
      }
    }
    "MRknee" => {
      let femur = cylinder_z(p, [0.0, 0.0], 0.3, 0.08, 1.0)
        || ellipsoid(p, [0.0, 0.0, 0.08], [0.42, 0.32, 0.2]);
      let tibia = cylinder_z(p, [0.0, 0.05], 0.28, -1.0, -0.12)
        || ellipsoid(p, [0.0, 0.05, -0.12], [0.45, 0.35, 0.12]);
      if femur || tibia {
        0.75 + 0.1 * n
      } else if (p[2]).abs() < 0.12 && ellipsoid(p, [0.0; 3], [0.45, 0.4, 0.3])
      {
        0.45
      } else if ellipsoid(p, [0.0, 0.0, 0.0], [0.8, 0.7, 2.0]) {
        0.3 + 0.08 * n
      } else {
        0.0
      }
    }
    "CTengine" => {
      let block = p[0].abs() < 0.8 && p[1].abs() < 0.5 && p[2].abs() < 0.7;
      let bores = (0..4)
        .any(|k| cylinder_z(p, [-0.6 + 0.4 * k as f64, 0.0], 0.15, -0.6, 0.8));
      if block && !bores { 0.9 } else { 0.0 }
    }
    _ => {
      // Orbitals: a few lobes of |Y_lm|^2-like densities.
      let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt().max(1e-9);
      let cos_t = p[2] / r;
      let lobes =
        (3.0 * cos_t * cos_t - 1.0).abs() * (-3.0 * r).exp() * r * r * 8.0;
      lobes.min(1.0)
    }
  }
}

/// The depth in an `Sk["Image3D", {w, h, d}, …]` skeleton.
fn skeleton_depth(skeleton: &Expr) -> usize {
  match skeleton {
    Expr::FunctionCall { args, .. } => match args.get(1) {
      Some(Expr::List(d)) => match d.get(2) {
        Some(Expr::Integer(n)) => *n as usize,
        _ => 1,
      },
      _ => 1,
    },
    _ => 1,
  }
}

/// The volume's slices as `Expr::Image`s.
fn volume(entry: &Entry) -> Option<Expr> {
  let shape = shape_of(&entry.skeleton("Image3D")?)?;
  let depth = skeleton_depth(&entry.skeleton("Image3D")?);
  Some(cached(format!("TestImage3D/{}/volume", entry.name), || {
    let (w, h) = (shape.width, shape.height);
    let noise = Noise::new(super::rng::hash(&["TestImage3D", entry.name]), 0);
    let mut vals = Vec::with_capacity(w * h * depth);
    for z in 0..depth {
      for y in 0..h {
        for x in 0..w {
          let p = [
            2.0 * (x as f64 + 0.5) / w as f64 - 1.0,
            2.0 * (y as f64 + 0.5) / h as f64 - 1.0,
            2.0 * (z as f64 + 0.5) / depth as f64 - 1.0,
          ];
          vals.push(density(entry.name, p, &noise));
        }
      }
    }
    // Rescale to the recorded mean and spread.
    if let Some(stats) = entry.stat("@Stats") {
      let s = numbers(&stats);
      if s.len() == 2 {
        let n = vals.len().max(1) as f64;
        let m = vals.iter().sum::<f64>() / n;
        let sd = (vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n)
          .sqrt()
          .max(1e-6);
        for v in &mut vals {
          *v = ((*v - m) / sd * s[1] + s[0]).clamp(0.0, 1.0);
        }
      }
    }
    let levels = if matches!(shape.image_type, ImageType::Bit16) {
      65535.0
    } else {
      255.0
    };
    let slices: Vec<Expr> = vals
      .chunks(w * h)
      .map(|plane| Expr::Image {
        width: w as u32,
        height: h as u32,
        channels: 1,
        data: std::sync::Arc::new(
          plane
            .iter()
            .map(|v| (v * levels).round() / levels)
            .collect(),
        ),
        image_type: shape.image_type,
        color_space: shape.color_space,
        display_size: None,
      })
      .collect();
    stack(slices, shape.color_space)
  }))
}

fn stack(slices: Vec<Expr>, color_space: Option<&'static str>) -> Expr {
  let mut args = vec![Expr::List(slices.into())];
  if let Some(cs) = color_space {
    args.push(rule_expr(id_expr("ColorSpace"), str_expr(cs)));
  }
  args.push(rule_expr(id_expr("Interleaving"), id_expr("None")));
  call("Image3D", args)
}

fn slices(vol: &Expr) -> Vec<Expr> {
  crate::functions::image_ast::image3d_slices(vol).unwrap_or_default()
}

/// A per-voxel list built from each slice's pixels (depth, rows, columns).
fn voxels(vol: &Expr, f: impl Fn(f64, f64) -> Expr) -> Expr {
  Expr::List(
    slices(vol)
      .iter()
      .filter_map(|s| {
        let (w, h, _, t, data) = image_parts(s)?;
        let m = if matches!(t, ImageType::Bit16) {
          65535.0
        } else {
          255.0
        };
        Some(Expr::List(
          (0..h)
            .map(|y| {
              Expr::List(
                (0..w)
                  .map(|x| f(data[y * w + x], m))
                  .collect::<Vec<_>>()
                  .into(),
              )
            })
            .collect::<Vec<_>>()
            .into(),
        ))
      })
      .collect::<Vec<_>>()
      .into(),
  )
}

pub struct TestImages3D;

impl Collection for TestImages3D {
  fn value(&self, entry: &Entry, property: Option<&str>) -> Option<Expr> {
    let vol = || volume(entry);
    Some(match property? {
      "Image3D" => vol()?,
      "ImageList" => Expr::List(slices(&vol()?).into()),
      "Data" => voxels(&vol()?, |v, m| Expr::Integer((v * m).round() as i128)),
      "GrayLevels" => voxels(&vol()?, |v, _| Expr::Real(v)),
      "RGBColorArray" => voxels(&vol()?, |v, _| {
        call(
          "RGBColor",
          vec![Expr::Real(v), Expr::Real(v), Expr::Real(v)],
        )
      }),
      "Thumbnail" => {
        let shape = shape_of(&entry.skeleton("Thumbnail")?)?;
        let target_depth = skeleton_depth(&entry.skeleton("Thumbnail")?);
        let all = slices(&vol()?);
        let picked: Vec<Expr> = (0..target_depth)
          .filter_map(|k| all.get(k * all.len() / target_depth.max(1)))
          .map(|s| resized(s, shape.width, shape.height))
          .collect();
        stack(picked, shape.color_space)
      }
      "FilePath" => super::images::file_path(entry, "tiff", || {
        vol().unwrap_or(Expr::Integer(0))
      }),
      _ => return None,
    })
  }
}
