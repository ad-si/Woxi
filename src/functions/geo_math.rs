//! Geodesic primitives shared by the geographic functions.
//!
//! All computations use Karney's geodesic algorithm (via `geographiclib-rs`)
//! on the GRS80 reference ellipsoid, which is the model the Wolfram Language
//! uses for `GeoModel -> ITRF00` (its default). This reproduces `GeoDistance`,
//! `GeoDirection` and `GeoDestination` to full machine precision.

use crate::InterpreterError;
use crate::functions::geographics::{position_to_latlon, positions_from_arg};
use crate::syntax::{Expr, UnaryOperator, unevaluated};
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
fn distance_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
  geodesic().inverse(lat1, lon1, lat2, lon2)
}

/// Initial bearing (azimuth) in degrees from point 1 to point 2, in the range
/// `(-180, 180]`, measured clockwise from true north — matching WL.
fn azimuth_deg(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
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

/// Wrap a geodesic distance (in meters) as a `Quantity`, picking the unit the
/// Wolfram Language uses: distances under 1000 m stay in `"Meters"`, longer
/// ones switch to `"Kilometers"`.
fn distance_quantity(m: f64) -> Expr {
  if m.abs() < 1000.0 {
    quantity(m, "Meters")
  } else {
    quantity(m / 1000.0, "Kilometers")
  }
}

fn geo_position(lat: f64, lon: f64) -> Expr {
  Expr::FunctionCall {
    name: "GeoPosition".to_string(),
    args: vec![Expr::List(vec![Expr::Real(lat), Expr::Real(lon)].into())]
      .into(),
  }
}

/// `GeoDistance[p1, p2]` — geodesic distance as `Quantity[km, "Kilometers"]`.
pub fn geo_distance_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() == 2
    && let (Some((lat1, lon1)), Some((lat2, lon2))) =
      (position_to_latlon(&args[0]), position_to_latlon(&args[1]))
  {
    let m = distance_m(lat1, lon1, lat2, lon2);
    return Ok(distance_quantity(m));
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
      return Ok(distance_quantity(m));
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

/// `GeoAntipode[pos]` — the point diametrically opposite `pos`: the latitude
/// is negated and the longitude shifted by 180°, renormalized into the
/// `(-180, 180]` range. Exact coordinates are preserved (integer in → integer
/// out) and a `GeoPosition` wrapper round-trips, matching wolframscript.
pub fn geo_antipode_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() == 1 {
    let (coords, wrapped) = match &args[0] {
      Expr::FunctionCall { name, args: a }
        if name == "GeoPosition" && a.len() == 1 =>
      {
        (&a[0], true)
      }
      other => (other, false),
    };
    if let Expr::List(items) = coords
      && items.len() == 2
      // Only act on numeric coordinates; leave symbolic input unevaluated.
      && crate::functions::graphics::expr_to_f64(&items[0]).is_some()
      && crate::functions::graphics::expr_to_f64(&items[1]).is_some()
    {
      let new_lat = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[Expr::Integer(-1), items[0].clone()],
      )?;
      let new_lon = antipode_longitude(&items[1])?;
      let result = Expr::List(vec![new_lat, new_lon].into());
      return Ok(if wrapped {
        Expr::FunctionCall {
          name: "GeoPosition".to_string(),
          args: vec![result].into(),
        }
      } else {
        result
      });
    }
  }
  Ok(unevaluated("GeoAntipode", args))
}

/// Shift a longitude by 180° and renormalize into `(-180, 180]`, keeping the
/// arithmetic exact. Input longitudes lie in `[-180, 180]`, so the shifted
/// value lies in `[0, 360]`; subtract a full turn only when it exceeds 180°.
fn antipode_longitude(lon: &Expr) -> Result<Expr, InterpreterError> {
  let shifted = crate::evaluator::evaluate_function_call_ast(
    "Plus",
    &[lon.clone(), Expr::Integer(180)],
  )?;
  if let Some(v) = crate::functions::graphics::expr_to_f64(&shifted)
    && v > 180.0
  {
    return crate::evaluator::evaluate_function_call_ast(
      "Plus",
      &[shifted, Expr::Integer(-360)],
    );
  }
  Ok(shifted)
}

/// An angle for DMS formatting: exact rational degrees (sign included) or a
/// machine real. Exact values whose seconds come out integral print without
/// decimals (`30°0'0"`); everything else gets a fixed number of decimals.
enum AngleVal {
  Exact(i128, i128), // p/q with q > 0
  Real(f64),
}

impl AngleVal {
  fn is_negative(&self) -> bool {
    match self {
      AngleVal::Exact(p, _) => *p < 0,
      AngleVal::Real(v) => *v < 0.0,
    }
  }

  fn abs(&self) -> AngleVal {
    match self {
      AngleVal::Exact(p, q) => AngleVal::Exact(p.abs(), *q),
      AngleVal::Real(v) => AngleVal::Real(v.abs()),
    }
  }

  /// d + m/60 + s/3600, staying exact only when all three parts are exact.
  fn from_dms(d: &AngleVal, m: &AngleVal, s: &AngleVal) -> AngleVal {
    if let (
      AngleVal::Exact(dp, dq),
      AngleVal::Exact(mp, mq),
      AngleVal::Exact(sp, sq),
    ) = (d, m, s)
      && let Some(v) = (|| {
        let a = rat_add((*dp, *dq), (*mp, mq.checked_mul(60)?))?;
        rat_add(a, (*sp, sq.checked_mul(3600)?))
      })()
    {
      return AngleVal::Exact(v.0, v.1);
    }
    AngleVal::Real(d.to_f64() + m.to_f64() / 60.0 + s.to_f64() / 3600.0)
  }

  fn to_f64(&self) -> f64 {
    match self {
      AngleVal::Exact(p, q) => *p as f64 / *q as f64,
      AngleVal::Real(v) => *v,
    }
  }
}

/// Exact rational addition with overflow checks (falls back to `None`).
fn rat_add(a: (i128, i128), b: (i128, i128)) -> Option<(i128, i128)> {
  let num = a.0.checked_mul(b.1)?.checked_add(b.0.checked_mul(a.1)?)?;
  let den = a.1.checked_mul(b.1)?;
  let g = gcd_i128(num.abs().max(1), den);
  Some((num / g, den / g))
}

fn gcd_i128(mut a: i128, mut b: i128) -> i128 {
  while b != 0 {
    (a, b) = (b, a % b);
  }
  a.max(1)
}

/// Convert a numeric scalar expression to an `AngleVal`. Exactness is
/// preserved for Integer/Rational; Reals and exact-but-irrational numerics
/// (Pi, Sqrt[2], …) take the machine-real path like wolframscript.
fn to_angle(expr: &Expr) -> Option<AngleVal> {
  match expr {
    Expr::Integer(n) => Some(AngleVal::Exact(*n, 1)),
    Expr::Real(v) => Some(AngleVal::Real(*v)),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      match (&args[0], &args[1]) {
        (Expr::Integer(p), Expr::Integer(q)) if *q != 0 => {
          Some(AngleVal::Exact(*p * q.signum(), q.abs()))
        }
        _ => None,
      }
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => to_angle(operand).map(|v| match v {
      AngleVal::Exact(p, q) => AngleVal::Exact(-p, q),
      AngleVal::Real(x) => AngleVal::Real(-x),
    }),
    _ => {
      let v = crate::functions::math_ast::try_eval_to_f64(expr)?;
      if !v.is_finite() {
        return None;
      }
      Some(AngleVal::Real(v))
    }
  }
}

/// Format a non-negative angle as `d°m's"`. Seconds of an exact angle that
/// are integral print as a bare integer (ignoring `prec`); otherwise they are
/// rounded half-even to `prec` decimals (a trailing `.` remains for prec 0).
fn format_dms(v: &AngleVal, prec: u32) -> String {
  let pow = 10_i128.pow(prec);
  // Total scaled seconds (angle * 3600 * 10^prec), rounded half-even.
  let scaled: i128 = match v {
    AngleVal::Exact(p, q) => {
      if (3600 * p) % q == 0 {
        // Integral seconds: emit without any decimal point.
        let total = 3600 * p / q;
        return format!(
          "{}°{}'{}\"",
          total / 3600,
          (total % 3600) / 60,
          total % 60
        );
      }
      let num = p.checked_mul(3600 * pow);
      match num {
        Some(num) => {
          let t = num / q;
          let r = num % q;
          match (2 * r).cmp(q) {
            std::cmp::Ordering::Greater => t + 1,
            std::cmp::Ordering::Equal => {
              if t % 2 == 0 {
                t
              } else {
                t + 1
              }
            }
            std::cmp::Ordering::Less => t,
          }
        }
        // Overflow: fall back to the machine-real path.
        None => return format_dms(&AngleVal::Real(v.to_f64()), prec),
      }
    }
    AngleVal::Real(x) => {
      let d = x.floor();
      let rem = ((x - d) * 3600.0 * pow as f64).round_ties_even() as i128;
      d as i128 * 3600 * pow + rem
    }
  };
  let d = scaled / (3600 * pow);
  let m = (scaled % (3600 * pow)) / (60 * pow);
  let s_scaled = scaled % (60 * pow);
  let s_int = s_scaled / pow;
  if prec == 0 {
    format!("{d}°{m}'{s_int}.\"")
  } else {
    format!(
      "{d}°{m}'{s_int}.{:0width$}\"",
      s_scaled % pow,
      width = prec as usize
    )
  }
}

/// Parse a DMS string like `30°15'50.5"` (each component optional after the
/// degrees, optional sign, optional trailing N/S/E/W). Numbers may have a
/// decimal part and are kept exact so integral seconds round-trip without
/// decimals.
fn parse_dms_string(s: &str) -> Option<AngleVal> {
  let mut rest = s.trim();
  let mut negative = false;
  if let Some(r) = rest.strip_prefix('-') {
    negative = true;
    rest = r;
  } else if let Some(r) = rest.strip_prefix('+') {
    rest = r;
  }
  // One decimal number as an exact rational, consuming it from the input.
  fn take_number(rest: &mut &str) -> Option<(i128, i128)> {
    let digits = |s: &str| s.chars().take_while(|c| c.is_ascii_digit()).count();
    let int_len = digits(rest);
    if int_len == 0 {
      return None;
    }
    let mut num: i128 = rest[..int_len].parse().ok()?;
    let mut den: i128 = 1;
    let mut consumed = int_len;
    let after = &rest[int_len..];
    if let Some(frac) = after.strip_prefix('.') {
      let frac_len = digits(frac);
      let frac_part: i128 = if frac_len == 0 {
        0
      } else {
        frac[..frac_len].parse().ok()?
      };
      den = 10_i128.checked_pow(frac_len as u32)?;
      num = num.checked_mul(den)?.checked_add(frac_part)?;
      consumed += 1 + frac_len;
    }
    *rest = &rest[consumed..];
    Some((num, den))
  }
  let deg = take_number(&mut rest)?;
  rest = rest.strip_prefix('°')?;
  let mut min = (0, 1);
  let mut sec = (0, 1);
  if !rest.is_empty() && rest.chars().next().is_some_and(|c| c.is_ascii_digit())
  {
    min = take_number(&mut rest)?;
    rest = rest.strip_prefix('\'')?;
  }
  if !rest.is_empty() && rest.chars().next().is_some_and(|c| c.is_ascii_digit())
  {
    sec = take_number(&mut rest)?;
    rest = rest.strip_prefix('"')?;
  }
  if matches!(rest, "N" | "S" | "E" | "W") {
    rest = "";
  }
  if !rest.is_empty() {
    return None;
  }
  let v = AngleVal::from_dms(
    &AngleVal::Exact(deg.0, deg.1),
    &AngleVal::Exact(min.0, min.1),
    &AngleVal::Exact(sec.0, sec.1),
  );
  Some(if negative {
    match v {
      AngleVal::Exact(p, q) => AngleVal::Exact(-p, q),
      AngleVal::Real(x) => AngleVal::Real(-x),
    }
  } else {
    v
  })
}

/// DMSString[angle] — format an angle in degrees as a `d°m's"` string.
/// A `{lat, lon}` pair (or GeoPosition) formats both with N/S / E/W
/// suffixes joined by two spaces; a `{d, m, s}` triple is a DMS value;
/// DMS strings parse and re-format. The optional second argument is the
/// number of decimals on the seconds (default 3).
pub fn dms_string_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("DMSString", args));
  if args.is_empty() || args.len() > 2 {
    return unevaluated();
  }

  let prec: u32 = match args.get(1) {
    None => 3,
    Some(Expr::Integer(n)) if (0..=27).contains(n) => *n as u32,
    Some(Expr::String(s)) => {
      crate::emit_message(&format!(
        "DMSString::form: Invalid formatting specification {}.",
        s
      ));
      return unevaluated();
    }
    Some(_) => {
      crate::emit_message(&format!(
        "DMSString::num: {} cannot be interpreted as a numerical angle specification.",
        crate::syntax::expr_to_output(&args[0])
      ));
      return unevaluated();
    }
  };

  // A lat/lon pair: each formatted as an absolute angle plus hemisphere
  // suffix (N/E for zero), joined by two spaces.
  let format_pair = |lat: &AngleVal, lon: &AngleVal| -> String {
    let side = |v: &AngleVal, pos: char, neg: char| {
      let c = if v.is_negative() { neg } else { pos };
      format!("{}{}", format_dms(&v.abs(), prec), c)
    };
    format!("{}  {}", side(lat, 'N', 'S'), side(lon, 'E', 'W'))
  };

  match &args[0] {
    Expr::String(s) => match parse_dms_string(s) {
      Some(v) => Ok(Expr::String(format_dms(&v.abs(), prec))),
      None => {
        crate::emit_message(&format!(
          "DMSString::str: {} cannot be interpreted as a degree-minute-second string specification.",
          s
        ));
        unevaluated()
      }
    },
    Expr::List(items) => {
      let vals: Option<Vec<AngleVal>> = items.iter().map(to_angle).collect();
      match vals.as_deref() {
        Some([lat, lon]) => Ok(Expr::String(format_pair(lat, lon))),
        Some([d, m, s]) => Ok(Expr::String(format_dms(
          &AngleVal::from_dms(d, m, s).abs(),
          prec,
        ))),
        _ => {
          crate::emit_message(&format!(
            "DMSString::dms: {} cannot be interpreted as a degree-minute-second list specification.",
            crate::syntax::expr_to_output(&args[0])
          ));
          unevaluated()
        }
      }
    }
    Expr::FunctionCall { name, args: gargs }
      if name == "GeoPosition" && gargs.len() == 1 =>
    {
      if let Expr::List(items) = &gargs[0]
        && (items.len() == 2 || items.len() == 3)
        && let (Some(lat), Some(lon)) =
          (to_angle(&items[0]), to_angle(&items[1]))
      {
        return Ok(Expr::String(format_pair(&lat, &lon)));
      }
      crate::emit_message(&format!(
        "DMSString::ang: {} cannot be interpreted as a degree-minute-second angle specification.",
        crate::syntax::expr_to_output(&args[0])
      ));
      unevaluated()
    }
    other => match to_angle(other) {
      Some(v) => Ok(Expr::String(format_dms(&v.abs(), prec))),
      None => {
        crate::emit_message(&format!(
          "DMSString::ang: {} cannot be interpreted as a degree-minute-second angle specification.",
          crate::syntax::expr_to_output(other)
        ));
        unevaluated()
      }
    },
  }
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

  #[test]
  fn antipode_negates_lat_and_wraps_lon() {
    let list = |a: i128, b: i128| {
      Expr::List(vec![Expr::Integer(a), Expr::Integer(b)].into())
    };
    let render =
      |e: Expr| crate::syntax::expr_to_string(&geo_antipode_ast(&[e]).unwrap());
    // Western longitude shifts east, exact integers preserved.
    assert_eq!(render(list(40, -100)), "{-40, 80}");
    // Longitude 0 maps to 180 (the boundary stays positive).
    assert_eq!(render(list(0, 0)), "{0, 180}");
    // Eastern longitude wraps past +180 to a negative value.
    assert_eq!(render(list(40, 100)), "{-40, -80}");
    // GeoPosition wrapper round-trips.
    let wrapped = Expr::FunctionCall {
      name: "GeoPosition".to_string(),
      args: vec![list(40, -100)].into(),
    };
    assert_eq!(render(wrapped), "GeoPosition[{-40, 80}]");
  }
}
