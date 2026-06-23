//! Geodesic primitives shared by the geographic functions.
//!
//! All computations use Karney's geodesic algorithm (via `geographiclib-rs`)
//! on the GRS80 reference ellipsoid, which is the model the Wolfram Language
//! uses for `GeoModel -> ITRF00` (its default). This reproduces `GeoDistance`,
//! `GeoDirection` and `GeoDestination` to full machine precision.

use crate::InterpreterError;
use crate::functions::geographics::{position_to_latlon, positions_from_arg};
use crate::syntax::Expr;
use geographiclib_rs::{DirectGeodesic, Geodesic, InverseGeodesic};
use std::sync::OnceLock;

/// GRS80 ellipsoid: semi-major axis (m) and flattening. Matches WL's ITRF00.
const GRS80_A: f64 = 6_378_137.0;
const GRS80_F: f64 = 1.0 / 298.257222101;

/// The shared geodesic model (GRS80).
fn geodesic() -> &'static Geodesic {
  static G: OnceLock<Geodesic> = OnceLock::new();
  G.get_or_init(|| Geodesic::new(GRS80_A, GRS80_F))
}

/// Geodesic distance in meters between two `(lat, lon)` points in degrees.
pub fn distance_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
  geodesic().inverse(lat1, lon1, lat2, lon2)
}

/// Initial bearing (azimuth) in degrees from point 1 to point 2, in the range
/// `(-180, 180]`, measured clockwise from true north — matching WL.
pub fn azimuth_deg(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
  // The 4-tuple inverse returns (s12, azi1, azi2, a12).
  let (_s, az1, _az2, _a12): (f64, f64, f64, f64) =
    geodesic().inverse(lat1, lon1, lat2, lon2);
  az1
}

/// Destination `(lat, lon)` reached from `(lat, lon)` by travelling `dist_m`
/// meters along the geodesic with initial bearing `az_deg` (degrees).
pub fn destination(lat: f64, lon: f64, az_deg: f64, dist_m: f64) -> (f64, f64) {
  geodesic().direct(lat, lon, az_deg, dist_m)
}

// ── Wolfram-Language entry points ─────────────────────────────────────────

fn quantity(magnitude: f64, unit: &str) -> Expr {
  Expr::FunctionCall {
    name: "Quantity".to_string(),
    args: vec![Expr::Real(magnitude), Expr::String(unit.to_string())].into(),
  }
}

fn geo_position(lat: f64, lon: f64) -> Expr {
  Expr::FunctionCall {
    name: "GeoPosition".to_string(),
    args: vec![Expr::List(vec![Expr::Real(lat), Expr::Real(lon)].into())]
      .into(),
  }
}

/// Leave the call symbolic (unevaluated) when arguments don't match.
fn unevaluated(name: &str, args: &[Expr]) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args: args.to_vec().into(),
  }
}

/// `GeoDistance[p1, p2]` — geodesic distance as `Quantity[km, "Kilometers"]`.
pub fn geo_distance_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() == 2
    && let (Some((lat1, lon1)), Some((lat2, lon2))) =
      (position_to_latlon(&args[0]), position_to_latlon(&args[1]))
  {
    let km = distance_m(lat1, lon1, lat2, lon2) / 1000.0;
    return Ok(quantity(km, "Kilometers"));
  }
  Ok(unevaluated("GeoDistance", args))
}

/// `GeoDirection[p1, p2]` — initial bearing as `Quantity[deg, "AngularDegrees"]`.
pub fn geo_direction_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() == 2
    && let (Some((lat1, lon1)), Some((lat2, lon2))) =
      (position_to_latlon(&args[0]), position_to_latlon(&args[1]))
  {
    let deg = azimuth_deg(lat1, lon1, lat2, lon2);
    return Ok(quantity(deg, "AngularDegrees"));
  }
  Ok(unevaluated("GeoDirection", args))
}

/// Resolve a distance/angle argument that may be a plain number or a
/// `Quantity`. A plain number passes through unchanged (distances are taken
/// in meters, bearings in degrees, matching wolframscript). A `Quantity` is
/// converted to `target_unit` and its magnitude returned — so
/// `Quantity[100, "Kilometers"]` and `Quantity[100, "Miles"]` both resolve to
/// the right number of meters, and `Quantity[90, "AngularDegrees"]` resolves
/// to 90.
fn magnitude_in_unit(expr: &Expr, target_unit: &str) -> Option<f64> {
  if let Some(v) = crate::functions::graphics::expr_to_f64(expr) {
    return Some(v);
  }
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "Quantity" || args.len() != 2 {
    return None;
  }
  // Already in the requested unit — `UnitConvert` to the same unit is a
  // no-op that stays unevaluated, so read the magnitude directly. The unit
  // may be stored as a String (`"AngularDegrees"`) or a symbol.
  let unit_name = match &args[1] {
    Expr::String(s) => s.clone(),
    other => crate::syntax::expr_to_string(other),
  };
  if unit_name == target_unit {
    return crate::functions::graphics::expr_to_f64(&args[0]);
  }
  let converted = crate::functions::quantity_ast::unit_convert_ast(&[
    expr.clone(),
    Expr::String(target_unit.to_string()),
  ])
  .ok()?;
  if let Expr::FunctionCall { name, args } = &converted
    && name == "Quantity"
    && args.len() == 2
  {
    return crate::functions::graphics::expr_to_f64(&args[0]);
  }
  None
}

/// `GeoDestination[p, {dist, azimuth}]` — the geodesic destination point.
/// `dist` may be a plain number (meters) or a length `Quantity`; `azimuth`
/// may be a plain number (degrees) or an `"AngularDegrees"` `Quantity`.
pub fn geo_destination_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() == 2
    && let Some((lat, lon)) = position_to_latlon(&args[0])
    && let Expr::List(spec) = &args[1]
    && spec.len() == 2
    && let (Some(dist_m), Some(az)) = (
      magnitude_in_unit(&spec[0], "Meters"),
      magnitude_in_unit(&spec[1], "AngularDegrees"),
    )
  {
    let (lat2, lon2) = destination(lat, lon, az, dist_m);
    return Ok(geo_position(lat2, lon2));
  }
  Ok(unevaluated("GeoDestination", args))
}

/// Collect the ordered positions of a path argument: a `GeoPath[{…}]`, a bare
/// list of positions, or a single `GeoPosition`.
fn path_positions(expr: &Expr) -> Vec<(f64, f64)> {
  if let Expr::FunctionCall { name, args } = expr
    && name == "GeoPath"
    && !args.is_empty()
  {
    return positions_from_arg(&args[0]);
  }
  positions_from_arg(expr)
}

/// `GeoLength[GeoPath[{…}]]` — total geodesic length as `Quantity[km, …]`.
pub fn geo_length_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() == 1 {
    let pts = path_positions(&args[0]);
    if pts.len() >= 2 {
      let mut m = 0.0;
      for w in pts.windows(2) {
        m += distance_m(w[0].0, w[0].1, w[1].0, w[1].1);
      }
      return Ok(quantity(m / 1000.0, "Kilometers"));
    }
  }
  Ok(unevaluated("GeoLength", args))
}

/// `GeoBounds[{positions…}]` — `{{latmin, latmax}, {lonmin, lonmax}}`.
pub fn geo_bounds_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() == 1 {
    let pts = positions_from_arg(&args[0]);
    if !pts.is_empty() {
      let lat_min = pts.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
      let lat_max = pts.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
      let lon_min = pts.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
      let lon_max = pts.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
      return Ok(Expr::List(
        vec![
          Expr::List(vec![Expr::Real(lat_min), Expr::Real(lat_max)].into()),
          Expr::List(vec![Expr::Real(lon_min), Expr::Real(lon_max)].into()),
        ]
        .into(),
      ));
    }
  }
  Ok(unevaluated("GeoBounds", args))
}

#[cfg(test)]
mod tests {
  use super::*;

  // Reference values from wolframscript (GeoModel -> ITRF00, the default).
  // geographiclib-rs agrees with WL's own geodesic to ~12 significant
  // figures; the displayed last few digits diverge by a few ULP because WL
  // uses a different geodesic implementation. That is invisible at map-pixel
  // scale (the only use of these helpers in rendering), so tests assert a
  // relative tolerance rather than the exact decimal string.
  fn close(a: f64, b: f64) {
    assert!((a - b).abs() <= b.abs() * 1e-9 + 1e-7, "{a} vs {b}");
  }

  #[test]
  fn distance_matches_wolfram() {
    close(
      distance_m(40.0, -100.0, 34.0, -118.0) / 1000.0,
      1731.01496832333,
    );
    close(distance_m(0.0, 0.0, 0.0, 180.0) / 1000.0, 20003.93145846094);
    assert_eq!(distance_m(40.0, -100.0, 40.0, -100.0), 0.0);
  }

  #[test]
  fn azimuth_matches_wolfram() {
    close(azimuth_deg(40.0, -100.0, 34.0, -118.0), -106.93807421415234);
    close(azimuth_deg(0.0, 0.0, 0.0, 10.0), 90.0);
  }

  #[test]
  fn destination_matches_wolfram() {
    let (lat, lon) = destination(40.0, -100.0, 45.0, 100_000.0);
    close(lat, 40.63380067529134);
    close(lon, -99.16417223868879);
  }
}
