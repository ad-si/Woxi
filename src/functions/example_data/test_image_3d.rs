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

/// Density in `[0, 1]` at `p` for the phantom `name`.
fn density(name: &str, p: [f64; 3], noise: &Noise) -> f64 {
  let n = noise.fbm(p[0] * 6.0 + p[2] * 3.1, p[1] * 6.0 - p[2] * 2.3, 3, 0.5);
  match name {
    "CThead" => {
      // Coronal slices: skull outline, brain, the jaw and spine below.
      let skull = ellipsoid(p, [0.0, -0.15, 0.0], [0.62, 0.72, 0.9]);
      let brain = ellipsoid(p, [0.0, -0.15, 0.0], [0.56, 0.65, 0.85]);
      let jaw = ellipsoid(p, [0.0, 0.62, 0.0], [0.25, 0.3, 0.6]);
      if brain {
        0.42 + 0.04 * n
      } else if skull {
        0.95
      } else if jaw {
        if (p[0].abs() < 0.05)
          || ellipsoid(p, [0.0, 0.62, 0.0], [0.18, 0.24, 0.5])
        {
          0.15 + 0.3 * (p[1] * 30.0).sin().abs()
        } else {
          0.85
        }
      } else {
        0.0
      }
    }
    "MRbrain" => {
      // Sagittal slices: head in profile facing left, folded brain, spine.
      let head = ellipsoid(p, [0.05, -0.1, 0.0], [0.7, 0.75, 0.9]);
      let face = ellipsoid(p, [-0.5, 0.35, 0.0], [0.25, 0.35, 0.5]);
      let neck = p[0] > -0.2 && p[0] < 0.25 && p[1] > 0.4;
      let brain = ellipsoid(p, [0.1, -0.25, 0.0], [0.58, 0.5, 0.8]);
      if brain {
        let folds = (n * 9.0).sin();
        if folds > 0.3 { 0.7 } else { 0.4 }
      } else if ellipsoid(p, [0.35, 0.3, 0.0], [0.2, 0.15, 0.5]) {
        0.6
      } else if head || face || neck {
        0.2 + 0.1 * n
      } else {
        0.02
      }
    }
    "MRknee" => {
      // A bright irregular blob: the knee in cross-section.
      let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
      let a = p[1].atan2(p[0]);
      let edge = 0.55
        + 0.1 * (2.0 * a).sin()
        + 0.06 * (3.0 * a + 1.0).cos()
        + 0.1 * p[2];
      if r < edge { 1.0 } else { 0.0 }
    }
    "CTengine" => {
      // An engine casting: an irregular vertical outline with ribs.
      let outline = p[0].abs() < 0.35 + 0.1 * (p[1] * 4.0).sin()
        && p[1].abs() < 0.85
        && p[2].abs() < 0.8;
      let cavity = ellipsoid(p, [0.08, 0.1, 0.0], [0.18, 0.45, 2.0]);
      let rib = (p[1] * 10.0).sin() > 0.9;
      if outline && (!cavity || rib) {
        0.6 + 0.3 * (1.0 - p[0].abs())
      } else {
        0.0
      }
    }
    _ => {
      // An orbital: a bright core fading outwards.
      let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
      (1.0 - (r / 0.55).powi(4)).clamp(0.0, 1.0)
        + 0.2 * (-((r - 0.6) / 0.08).powi(2)).exp()
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
