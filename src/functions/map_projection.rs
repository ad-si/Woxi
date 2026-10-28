//! Forward map projections of the spherical Earth, used by
//! `CountryData[country, {"Shape", {projection, {lat, lon}}}]`.
//!
//! Points are projected on the unit sphere, so the result is in units of the
//! Earth's radius, with the origin at the projection center.

use super::*;
use crate::functions::geographics::country_outline_rings;
use crate::helpers::call;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

/// Largest latitude (radians) fed to projections that blow up at the poles.
const MAX_LAT: f64 = 89.9 * PI / 180.0;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
  Albers,
  Bonne,
  CylindricalEqualArea(f64),
  CylindricalEquidistant(f64),
  EckertIV,
  EckertVI,
  Equirectangular,
  LambertAzimuthal,
  LambertConic,
  LambertCylindrical,
  Mercator,
  MillerCylindrical,
  Mollweide,
  Orthographic,
  Polyconic,
  Robinson,
  Sinusoidal,
  VanDerGrinten,
  WinkelTripel,
}

/// A projection together with its center `(lat, lon)` in radians.
pub struct MapProjection {
  kind: Kind,
  lat0: f64,
  lon0: f64,
}

fn name_of(e: &Expr) -> Option<&str> {
  match e {
    Expr::String(s) | Expr::Identifier(s) => Some(s),
    _ => None,
  }
}

fn kind_from_name(name: &str, param: Option<f64>) -> Option<Kind> {
  let p = param.unwrap_or(0.0).to_radians();
  Some(match name {
    "Albers" => Kind::Albers,
    "Bonne" => Kind::Bonne,
    "CylindricalEqualArea" => Kind::CylindricalEqualArea(p),
    "CylindricalEquidistant" => Kind::CylindricalEquidistant(p),
    "EckertIV" => Kind::EckertIV,
    "EckertVI" => Kind::EckertVI,
    "Equirectangular" => Kind::Equirectangular,
    "LambertAzimuthal" => Kind::LambertAzimuthal,
    "LambertConic" => Kind::LambertConic,
    "LambertCylindrical" => Kind::LambertCylindrical,
    "Mercator" => Kind::Mercator,
    "MillerCylindrical" => Kind::MillerCylindrical,
    "Mollweide" => Kind::Mollweide,
    "Orthographic" => Kind::Orthographic,
    "Polyconic" => Kind::Polyconic,
    "Robinson" => Kind::Robinson,
    "Sinusoidal" => Kind::Sinusoidal,
    "VanDerGrinten" => Kind::VanDerGrinten,
    "WinkelTripel" => Kind::WinkelTripel,
    _ => return None,
  })
}

impl MapProjection {
  /// Parse `proj` (`"Name"` or `{"Name", parameter}`) and `center`
  /// (`{lat, lon}`).
  pub fn from_exprs(proj: &Expr, center: &Expr) -> Option<Self> {
    let kind = match proj {
      Expr::List(items) if items.len() == 2 => kind_from_name(
        name_of(&items[0])?,
        Some(crate::functions::graphics::expr_to_f64(&items[1])?),
      )?,
      Expr::List(items) if items.len() == 1 => {
        kind_from_name(name_of(&items[0])?, None)?
      }
      other => kind_from_name(name_of(other)?, None)?,
    };
    let (lat0, lon0) = match center {
      Expr::List(c) if c.len() == 2 => (
        crate::functions::graphics::expr_to_f64(&c[0])?,
        crate::functions::graphics::expr_to_f64(&c[1])?,
      ),
      _ => return None,
    };
    Some(MapProjection {
      kind,
      lat0: lat0.to_radians(),
      lon0: lon0.to_radians(),
    })
  }

  /// Project a point (degrees); `None` when it is not representable (the far
  /// side of a hemispheric projection, or a singular point).
  pub fn project(&self, lon_deg: f64, lat_deg: f64) -> Option<(f64, f64)> {
    let d = lon_deg.to_radians() - self.lon0;
    self.project_rel((d + PI).rem_euclid(2.0 * PI) - PI, lat_deg)
  }

  /// Like [`Self::project`], for a longitude relative to the center
  /// (radians, within `[-π, π]`).
  fn project_rel(&self, d: f64, lat_deg: f64) -> Option<(f64, f64)> {
    let phi = lat_deg.to_radians();
    let phi0 = self.lat0;
    let sphere = |phi: f64, d: f64| {
      // Cosine of the angular distance from the center.
      phi0.sin() * phi.sin() + phi0.cos() * phi.cos() * d.cos()
    };
    let r = match self.kind {
      Kind::Equirectangular => (d, phi),
      Kind::CylindricalEquidistant(p1) => (d * p1.cos(), phi),
      Kind::CylindricalEqualArea(p1) => (d * p1.cos(), phi.sin() / p1.cos()),
      Kind::LambertCylindrical => (d, phi.sin()),
      Kind::Mercator => (d, mercator(phi, 1.0)),
      Kind::MillerCylindrical => (d, 1.25 * mercator(phi, 0.8)),
      Kind::Sinusoidal => (d * phi.cos(), phi),
      Kind::Mollweide => {
        let t = solve_theta(phi, |t| 2.0 * t + (2.0 * t).sin(), PI * phi.sin());
        (2.0 * 2f64.sqrt() / PI * d * t.cos(), 2f64.sqrt() * t.sin())
      }
      Kind::EckertIV => {
        let t = solve_theta(
          phi,
          |t| t + t.sin() * t.cos() + 2.0 * t.sin(),
          (2.0 + FRAC_PI_2) * phi.sin(),
        );
        (
          2.0 / (4.0 * PI + PI * PI).sqrt() * d * (1.0 + t.cos()),
          2.0 * PI.sqrt() / (4.0 + PI).sqrt() * t.sin(),
        )
      }
      Kind::EckertVI => {
        let t =
          solve_theta(phi, |t| t + t.sin(), (1.0 + FRAC_PI_2) * phi.sin());
        let k = (2.0 + PI).sqrt();
        (d * (1.0 + t.cos()) / k, 2.0 * t / k)
      }
      Kind::Robinson => robinson(d, phi),
      Kind::WinkelTripel => {
        let alpha = sphere_central_angle(phi, d);
        let sinc = if alpha.abs() < 1e-12 {
          1.0
        } else {
          alpha.sin() / alpha
        };
        let (ax, ay) =
          (2.0 * phi.cos() * (d / 2.0).sin() / sinc, phi.sin() / sinc);
        (f64::midpoint(d * (2.0 / PI), ax), f64::midpoint(phi, ay))
      }
      Kind::VanDerGrinten => van_der_grinten(d, phi),
      Kind::LambertAzimuthal => {
        let c = sphere(phi, d);
        if c < -1.0 + 1e-9 {
          return None;
        }
        let k = (2.0 / (1.0 + c)).sqrt();
        (
          k * phi.cos() * d.sin(),
          k * (phi0.cos() * phi.sin() - phi0.sin() * phi.cos() * d.cos()),
        )
      }
      Kind::Orthographic => {
        if sphere(phi, d) < 0.0 {
          return None;
        }
        (
          phi.cos() * d.sin(),
          phi0.cos() * phi.sin() - phi0.sin() * phi.cos() * d.cos(),
        )
      }
      Kind::Albers => {
        let (p1, p2) = (29.5f64.to_radians(), 45.5f64.to_radians());
        let n = f64::midpoint(p1.sin(), p2.sin());
        let c = p1.cos().powi(2) + 2.0 * n * p1.sin();
        let rho = |p: f64| (c - 2.0 * n * p.sin()).max(0.0).sqrt() / n;
        let (rho_p, rho0) = (rho(phi), rho(phi0));
        (rho_p * (n * d).sin(), rho0 - rho_p * (n * d).cos())
      }
      Kind::LambertConic => {
        let p1 = phi0.clamp(-MAX_LAT, MAX_LAT);
        if p1.abs() < 1e-6 {
          (d, mercator(phi, 1.0))
        } else {
          let n = p1.sin();
          let t = |p: f64| (FRAC_PI_4 + p.clamp(-MAX_LAT, MAX_LAT) / 2.0).tan();
          let f = p1.cos() * t(p1).powf(n) / n;
          let (rho, rho0) = (f / t(phi).powf(n), f / t(p1).powf(n));
          (rho * (n * d).sin(), rho0 - rho * (n * d).cos())
        }
      }
      Kind::Bonne => {
        if phi0.abs() < 1e-6 {
          (d * phi.cos(), phi)
        } else {
          let cot = 1.0 / phi0.tan();
          let rho = cot + phi0 - phi;
          let e = if rho.abs() < 1e-12 {
            0.0
          } else {
            d * phi.cos() / rho
          };
          (rho * e.sin(), cot - rho * e.cos())
        }
      }
      Kind::Polyconic => {
        if phi.abs() < 1e-9 {
          (d, 0.0)
        } else {
          let e = d * phi.sin();
          let cot = 1.0 / phi.tan();
          (cot * e.sin(), phi + cot * (1.0 - e.cos()))
        }
      }
    };
    // Cylindrical, pseudocylindrical and similar projections are drawn
    // relative to the equator; the conics and Bonne already contain `phi0`.
    let y_shift = match self.kind {
      Kind::Equirectangular
      | Kind::CylindricalEquidistant(_)
      | Kind::CylindricalEqualArea(_)
      | Kind::LambertCylindrical
      | Kind::Mercator
      | Kind::MillerCylindrical => self.cyl_y_center(),
      _ => 0.0,
    };
    (r.0.is_finite() && r.1.is_finite()).then_some((r.0, r.1 - y_shift))
  }

  /// Vertical offset of the map center for the cylindrical projections.
  fn cyl_y_center(&self) -> f64 {
    let phi0 = self.lat0;
    match self.kind {
      Kind::Equirectangular | Kind::CylindricalEquidistant(_) => phi0,
      Kind::CylindricalEqualArea(p1) => phi0.sin() / p1.cos(),
      Kind::LambertCylindrical => phi0.sin(),
      Kind::Mercator => mercator(phi0, 1.0),
      Kind::MillerCylindrical => 1.25 * mercator(phi0, 0.8),
      _ => 0.0,
    }
  }
}

/// `ln tan(π/4 + k φ/2)` with the latitude kept away from the poles.
fn mercator(phi: f64, k: f64) -> f64 {
  (FRAC_PI_4 + k * phi.clamp(-MAX_LAT, MAX_LAT) / 2.0)
    .tan()
    .ln()
}

/// Angular distance from the point on the equator at the central meridian.
fn sphere_central_angle(phi: f64, d: f64) -> f64 {
  (phi.cos() * (d / 2.0).cos()).clamp(-1.0, 1.0).acos()
}

/// Newton iteration for the auxiliary angle θ with `f(θ) = rhs`.
fn solve_theta(phi: f64, f: impl Fn(f64) -> f64, rhs: f64) -> f64 {
  if phi.abs() >= FRAC_PI_2 - 1e-9 {
    return phi.signum() * FRAC_PI_2;
  }
  let mut t = phi;
  for _ in 0..50 {
    let h = 1e-7;
    let df = (f(t + h) - f(t - h)) / (2.0 * h);
    if df.abs() < 1e-12 {
      break;
    }
    let step = (f(t) - rhs) / df;
    t = (t - step).clamp(-FRAC_PI_2, FRAC_PI_2);
    if step.abs() < 1e-12 {
      break;
    }
  }
  t
}

/// Robinson's tabulated projection, interpolated linearly in latitude.
fn robinson(d: f64, phi: f64) -> (f64, f64) {
  // (parallel length, distance from equator) every 5 degrees from 0 to 90.
  const X: [f64; 19] = [
    1.0, 0.9986, 0.9954, 0.99, 0.9822, 0.973, 0.96, 0.9427, 0.9216, 0.8962,
    0.8679, 0.835, 0.7986, 0.7597, 0.7186, 0.6732, 0.6213, 0.5722, 0.5322,
  ];
  const Y: [f64; 19] = [
    0.0, 0.062, 0.124, 0.186, 0.248, 0.31, 0.372, 0.434, 0.4958, 0.5571,
    0.6176, 0.6769, 0.7346, 0.7903, 0.8435, 0.8936, 0.9394, 0.9761, 1.0,
  ];
  let a = phi.abs().to_degrees().min(90.0) / 5.0;
  let i = (a.floor() as usize).min(17);
  let f = a - i as f64;
  let x = X[i] + (X[i + 1] - X[i]) * f;
  let y = Y[i] + (Y[i + 1] - Y[i]) * f;
  (0.8487 * x * d, 1.3523 * y * phi.signum())
}

fn van_der_grinten(d: f64, phi: f64) -> (f64, f64) {
  if phi.abs() < 1e-9 {
    return (d, 0.0);
  }
  if d.abs() < 1e-9 || (phi.abs() - FRAC_PI_2).abs() < 1e-9 {
    let t = (phi.abs() / FRAC_PI_2).min(1.0).asin();
    return (0.0, PI * (t / 2.0).tan() * phi.signum());
  }
  let a = 0.5 * (PI / d - d / PI).abs();
  let theta = (phi.abs() / FRAC_PI_2).min(1.0).asin();
  let g = theta.cos() / (theta.sin() + theta.cos() - 1.0);
  let p = g * (2.0 / theta.sin() - 1.0);
  let q = a * a + g;
  let x = d.signum()
    * PI
    * (a * (g - p * p)
      + (a * a * (g - p * p).powi(2) - (p * p + a * a) * (g * g - p * p))
        .max(0.0)
        .sqrt())
    / (p * p + a * a);
  let y = phi.signum()
    * PI
    * (p * q - a * ((a * a + 1.0) * (p * p + a * a) - q * q).max(0.0).sqrt())
    / (p * p + a * a);
  (x, y)
}

/// Clip a closed ring (points `(x, y)`) to `lo <= x <= hi`
/// (Sutherland–Hodgman).
fn clip_x(ring: &[(f64, f64)], lo: f64, hi: f64) -> Vec<(f64, f64)> {
  let clip_side = |pts: Vec<(f64, f64)>, edge: f64, keep_above: bool| {
    let inside =
      |p: &(f64, f64)| if keep_above { p.0 >= edge } else { p.0 <= edge };
    let mut out = Vec::new();
    for i in 0..pts.len() {
      let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
      let cut = |a: (f64, f64), b: (f64, f64)| {
        let t = (edge - a.0) / (b.0 - a.0);
        (edge, a.1 + t * (b.1 - a.1))
      };
      match (inside(&a), inside(&b)) {
        (true, true) => out.push(b),
        (true, false) => out.push(cut(a, b)),
        (false, true) => {
          out.push(cut(a, b));
          out.push(b);
        }
        _ => {}
      }
    }
    out
  };
  let pts = clip_side(ring.to_vec(), lo, true);
  if pts.is_empty() {
    return pts;
  }
  clip_side(pts, hi, false)
}

/// The pieces of a ring (longitude, latitude in degrees) that lie within half
/// a turn of the center meridian, with the rest wrapped around the globe.
fn wrapped_pieces(ring: &[(f32, f32)], lon0_deg: f64) -> Vec<Vec<(f64, f64)>> {
  let mut unwrapped: Vec<(f64, f64)> = Vec::with_capacity(ring.len());
  for &(lon, lat) in ring {
    let mut x = (lon as f64 - lon0_deg + 180.0).rem_euclid(360.0) - 180.0;
    if let Some(&(prev, _)) = unwrapped.last() {
      while x - prev > 180.0 {
        x -= 360.0;
      }
      while prev - x > 180.0 {
        x += 360.0;
      }
    }
    unwrapped.push((x, lat as f64));
  }
  let mut pieces = Vec::new();
  for k in -2..=2 {
    let shift = 360.0 * k as f64;
    let piece = clip_x(&unwrapped, -180.0 + shift, 180.0 + shift);
    if piece.len() >= 3 {
      pieces.push(piece.into_iter().map(|(x, y)| (x - shift, y)).collect());
    }
  }
  pieces
}

/// `CountryData[country, {"Shape", {projection, {lat, lon}}}]`: the country
/// (or the whole world for `"World"`) projected as a `Graphics` of polygons.
pub fn country_shape(
  country: Option<&str>,
  proj: &Expr,
  center: &Expr,
) -> Option<Expr> {
  let projection = MapProjection::from_exprs(proj, center)?;
  let rings = country_outline_rings(country)?;
  let lon0_deg = projection.lon0.to_degrees();
  let polygons: Vec<Expr> = rings
    .iter()
    .flat_map(|ring| wrapped_pieces(ring, lon0_deg))
    .filter_map(|piece| {
      let pts: Vec<Expr> = piece
        .iter()
        .filter_map(|&(d, lat)| projection.project_rel(d.to_radians(), lat))
        .map(|(x, y)| Expr::List(vec![Expr::Real(x), Expr::Real(y)].into()))
        .collect();
      (pts.len() >= 3).then(|| call("Polygon", vec![Expr::List(pts.into())]))
    })
    .collect();
  Some(call("Graphics", vec![Expr::List(polygons.into())]))
}
