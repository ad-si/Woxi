use crate::InterpreterError;
use crate::syntax::Expr;

// ─── Unit dimension system ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dimension {
  Length,
  Mass,
  Time,
  Volume,
}

/// Conversion factor to SI base unit as a rational (numerator, denominator).
/// E.g. Kilometers → Meters is (1000, 1).
struct UnitInfo {
  dimension: Dimension,
  /// to_si_numer / to_si_denom = number of SI base units per 1 of this unit
  to_si_numer: i128,
  to_si_denom: i128,
}

fn get_unit_info(name: &str) -> Option<UnitInfo> {
  let info = match name {
    // Length → Meters
    "Meters" => UnitInfo {
      dimension: Dimension::Length,
      to_si_numer: 1,
      to_si_denom: 1,
    },
    "Kilometers" => UnitInfo {
      dimension: Dimension::Length,
      to_si_numer: 1000,
      to_si_denom: 1,
    },
    "Centimeters" => UnitInfo {
      dimension: Dimension::Length,
      to_si_numer: 1,
      to_si_denom: 100,
    },
    "Millimeters" => UnitInfo {
      dimension: Dimension::Length,
      to_si_numer: 1,
      to_si_denom: 1000,
    },
    "Feet" => UnitInfo {
      dimension: Dimension::Length,
      to_si_numer: 381,
      to_si_denom: 1250,
    },
    "Inches" => UnitInfo {
      dimension: Dimension::Length,
      to_si_numer: 127,
      to_si_denom: 5000,
    },
    "Miles" => UnitInfo {
      dimension: Dimension::Length,
      to_si_numer: 201168,
      to_si_denom: 125,
    },
    "Yards" => UnitInfo {
      dimension: Dimension::Length,
      to_si_numer: 1143,
      to_si_denom: 1250,
    },

    // Mass → Kilograms
    "Kilograms" => UnitInfo {
      dimension: Dimension::Mass,
      to_si_numer: 1,
      to_si_denom: 1,
    },
    "Grams" => UnitInfo {
      dimension: Dimension::Mass,
      to_si_numer: 1,
      to_si_denom: 1000,
    },
    "Milligrams" => UnitInfo {
      dimension: Dimension::Mass,
      to_si_numer: 1,
      to_si_denom: 1000000,
    },
    "Pounds" => UnitInfo {
      dimension: Dimension::Mass,
      to_si_numer: 45359237,
      to_si_denom: 100000000,
    },

    // Time → Seconds
    "Seconds" => UnitInfo {
      dimension: Dimension::Time,
      to_si_numer: 1,
      to_si_denom: 1,
    },
    "Minutes" => UnitInfo {
      dimension: Dimension::Time,
      to_si_numer: 60,
      to_si_denom: 1,
    },
    "Hours" => UnitInfo {
      dimension: Dimension::Time,
      to_si_numer: 3600,
      to_si_denom: 1,
    },
    "Days" => UnitInfo {
      dimension: Dimension::Time,
      to_si_numer: 86400,
      to_si_denom: 1,
    },

    // Volume → Liters
    "Liters" => UnitInfo {
      dimension: Dimension::Volume,
      to_si_numer: 1,
      to_si_denom: 1,
    },
    "Milliliters" => UnitInfo {
      dimension: Dimension::Volume,
      to_si_numer: 1,
      to_si_denom: 1000,
    },
    "Gallons" => UnitInfo {
      dimension: Dimension::Volume,
      to_si_numer: 473176473,
      to_si_denom: 125000000,
    },

    _ => return None,
  };
  Some(info)
}

fn gcd(mut a: i128, mut b: i128) -> i128 {
  a = a.abs();
  b = b.abs();
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a
}

// ─── Quantity helpers ───────────────────────────────────────────────────────

/// Extract (magnitude, unit_string) from a Quantity FunctionCall.
/// Unit can be an Identifier or a String.
pub fn is_quantity(expr: &Expr) -> Option<(&Expr, &Expr)> {
  if let Expr::FunctionCall { name, args } = expr
    && name == "Quantity"
    && args.len() == 2
  {
    return Some((&args[0], &args[1]));
  }
  None
}

/// Get the unit name from a unit expression (Identifier or String).
fn unit_name(expr: &Expr) -> Option<&str> {
  match expr {
    Expr::Identifier(s) => Some(s.as_str()),
    Expr::String(s) => Some(s.as_str()),
    _ => None,
  }
}

/// Check if two unit expressions represent compatible (same dimension) units.
fn units_compatible(u1: &Expr, u2: &Expr) -> bool {
  let n1 = unit_name(u1);
  let n2 = unit_name(u2);
  if let (Some(n1), Some(n2)) = (n1, n2)
    && let (Some(info1), Some(info2)) = (get_unit_info(n1), get_unit_info(n2))
  {
    return info1.dimension == info2.dimension;
  }
  // If we can't look up the units, they're compatible only if identical
  crate::syntax::expr_to_string(u1) == crate::syntax::expr_to_string(u2)
}

/// Check if two unit expressions are the same unit.
fn units_equal(u1: &Expr, u2: &Expr) -> bool {
  crate::syntax::expr_to_string(u1) == crate::syntax::expr_to_string(u2)
}

/// Recursively normalize unit expressions: String → Identifier
fn normalize_unit(unit: Expr) -> Expr {
  match unit {
    Expr::String(s) => Expr::Identifier(s),
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op,
      left: Box::new(normalize_unit(*left)),
      right: Box::new(normalize_unit(*right)),
    },
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name,
      args: args.into_iter().map(normalize_unit).collect(),
    },
    other => other,
  }
}

fn make_quantity(magnitude: Expr, unit: Expr) -> Expr {
  let unit = normalize_unit(unit);
  Expr::FunctionCall {
    name: "Quantity".to_string(),
    args: vec![magnitude, unit],
  }
}

/// Convert a magnitude from `from_unit` to `to_unit` (must be same dimension).
/// Returns magnitude * (from_factor / to_factor) as exact rational arithmetic.
fn convert_magnitude(
  magnitude: &Expr,
  from_unit: &str,
  to_unit: &str,
) -> Result<Expr, InterpreterError> {
  let from = get_unit_info(from_unit).ok_or_else(|| {
    InterpreterError::EvaluationError(format!("Unknown unit: {}", from_unit))
  })?;
  let to = get_unit_info(to_unit).ok_or_else(|| {
    InterpreterError::EvaluationError(format!("Unknown unit: {}", to_unit))
  })?;

  if from.dimension != to.dimension {
    return Err(InterpreterError::EvaluationError(format!(
      "{} and {} are incompatible units.",
      from_unit, to_unit
    )));
  }

  // conversion = (from_numer * to_denom) / (from_denom * to_numer)
  let conv_numer = from.to_si_numer * to.to_si_denom;
  let conv_denom = from.to_si_denom * to.to_si_numer;

  multiply_magnitude_by_rational(magnitude, conv_numer, conv_denom)
}

/// Multiply a magnitude expression by a rational (numer/denom).
fn multiply_magnitude_by_rational(
  magnitude: &Expr,
  numer: i128,
  denom: i128,
) -> Result<Expr, InterpreterError> {
  // Simplify the conversion factor
  let g = gcd(numer, denom);
  let numer = numer / g;
  let denom = denom / g;

  if numer == 1 && denom == 1 {
    return Ok(magnitude.clone());
  }

  match magnitude {
    Expr::Integer(m) => {
      let result_numer = m * numer;
      let result_denom = denom;
      let g2 = gcd(result_numer, result_denom);
      let rn = result_numer / g2;
      let rd = result_denom / g2;
      if rd == 1 {
        Ok(Expr::Integer(rn))
      } else {
        Ok(crate::functions::math_ast::make_rational_pub(rn, rd))
      }
    }
    Expr::Real(f) => Ok(Expr::Real(f * (numer as f64) / (denom as f64))),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(mn), Expr::Integer(md)) = (&args[0], &args[1]) {
        let rn = mn * numer;
        let rd = md * denom;
        Ok(crate::functions::math_ast::make_rational_pub(rn, rd))
      } else {
        // Symbolic magnitude — wrap in Times
        Ok(crate::functions::math_ast::times_ast(&[
          magnitude.clone(),
          crate::functions::math_ast::make_rational_pub(numer, denom),
        ])?)
      }
    }
    _ => {
      // Symbolic magnitude — wrap in Times
      let factor = if denom == 1 {
        Expr::Integer(numer)
      } else {
        crate::functions::math_ast::make_rational_pub(numer, denom)
      };
      crate::functions::math_ast::times_ast(&[magnitude.clone(), factor])
    }
  }
}

// ─── Quantity[mag, unit] constructor ────────────────────────────────────────

pub fn quantity_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  match args.len() {
    1 => {
      // Quantity["Meters"] → Quantity[1, Meters]
      let unit = args[0].clone();
      Ok(make_quantity(Expr::Integer(1), unit))
    }
    2 => {
      let magnitude = args[0].clone();
      let unit = args[1].clone();
      Ok(make_quantity(magnitude, unit))
    }
    _ => Err(InterpreterError::EvaluationError(
      "Quantity expects 1 or 2 arguments".into(),
    )),
  }
}

// ─── QuantityMagnitude ──────────────────────────────────────────────────────

pub fn quantity_magnitude_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  match args.len() {
    1 => {
      // QuantityMagnitude[Quantity[m, u]] → m
      if let Some((mag, _unit)) = is_quantity(&args[0]) {
        Ok(mag.clone())
      } else {
        Ok(Expr::FunctionCall {
          name: "QuantityMagnitude".to_string(),
          args: args.to_vec(),
        })
      }
    }
    2 => {
      // QuantityMagnitude[Quantity[m, u], target_unit] → convert, return magnitude
      if let Some((mag, unit)) = is_quantity(&args[0]) {
        let target = &args[1];
        let from_name = unit_name(unit);
        let to_name = unit_name(target);
        if let (Some(from), Some(to)) = (from_name, to_name) {
          convert_magnitude(mag, from, to)
        } else {
          Ok(Expr::FunctionCall {
            name: "QuantityMagnitude".to_string(),
            args: args.to_vec(),
          })
        }
      } else {
        Ok(Expr::FunctionCall {
          name: "QuantityMagnitude".to_string(),
          args: args.to_vec(),
        })
      }
    }
    _ => Err(InterpreterError::EvaluationError(
      "QuantityMagnitude expects 1 or 2 arguments".into(),
    )),
  }
}

// ─── QuantityUnit ───────────────────────────────────────────────────────────

pub fn quantity_unit_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "QuantityUnit expects exactly 1 argument".into(),
    ));
  }
  if let Some((_mag, unit)) = is_quantity(&args[0]) {
    Ok(unit.clone())
  } else {
    Ok(Expr::FunctionCall {
      name: "QuantityUnit".to_string(),
      args: args.to_vec(),
    })
  }
}

// ─── QuantityQ ──────────────────────────────────────────────────────────────

pub fn quantity_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "QuantityQ expects exactly 1 argument".into(),
    ));
  }
  let result = is_quantity(&args[0]).is_some();
  Ok(Expr::Identifier(
    if result { "True" } else { "False" }.to_string(),
  ))
}

// ─── CompatibleUnitQ ────────────────────────────────────────────────────────

pub fn compatible_unit_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "CompatibleUnitQ expects exactly 2 arguments".into(),
    ));
  }
  let u1 = if let Some((_m, u)) = is_quantity(&args[0]) {
    u
  } else {
    &args[0]
  };
  let u2 = if let Some((_m, u)) = is_quantity(&args[1]) {
    u
  } else {
    &args[1]
  };
  let result = units_compatible(u1, u2);
  Ok(Expr::Identifier(
    if result { "True" } else { "False" }.to_string(),
  ))
}

// ─── UnitConvert ────────────────────────────────────────────────────────────

pub fn unit_convert_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "UnitConvert expects exactly 2 arguments".into(),
    ));
  }
  if let Some((mag, unit)) = is_quantity(&args[0]) {
    let target = &args[1];
    let from_name = unit_name(unit);
    let to_name = unit_name(target);
    if let (Some(from), Some(to)) = (from_name, to_name) {
      let new_mag = convert_magnitude(mag, from, to)?;
      Ok(make_quantity(new_mag, target.clone()))
    } else {
      Ok(Expr::FunctionCall {
        name: "UnitConvert".to_string(),
        args: args.to_vec(),
      })
    }
  } else {
    Ok(Expr::FunctionCall {
      name: "UnitConvert".to_string(),
      args: args.to_vec(),
    })
  }
}

// ─── Arithmetic integration ────────────────────────────────────────────────

/// Handle Plus when Quantity arguments are present.
/// Returns Some(result) if handled, None if not applicable.
pub fn try_quantity_plus(
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  // Check if any argument is a Quantity
  let has_quantity = args.iter().any(|a| is_quantity(a).is_some());
  if !has_quantity {
    return None;
  }

  // Separate quantity args from non-quantity args
  let mut quantity_args: Vec<&Expr> = Vec::new();
  let mut other_args: Vec<Expr> = Vec::new();

  for arg in args {
    if is_quantity(arg).is_some() {
      quantity_args.push(arg);
    } else {
      other_args.push(arg.clone());
    }
  }

  if quantity_args.is_empty() {
    return None;
  }

  // Get the target unit from the first quantity
  let (first_mag, first_unit) = is_quantity(quantity_args[0]).unwrap();
  let target_unit_name = unit_name(first_unit);

  // Check all quantities are compatible
  for q in &quantity_args[1..] {
    let (_m, u) = is_quantity(q).unwrap();
    if !units_compatible(first_unit, u) {
      // Incompatible units — print error and return unevaluated
      let u1_name = crate::syntax::expr_to_string(first_unit);
      let u2_name = crate::syntax::expr_to_string(u);
      eprintln!();
      eprintln!(
        "Quantity::compat: {} and {} are incompatible units.",
        u1_name, u2_name
      );
      use std::io::{self, Write};
      io::stderr().flush().ok();
      // Return unevaluated Plus
      return Some(Ok(Expr::FunctionCall {
        name: "Plus".to_string(),
        args: args.to_vec(),
      }));
    }
  }

  // All quantities are compatible — convert to first unit and sum magnitudes
  let mut magnitudes: Vec<Expr> = vec![first_mag.clone()];
  for q in &quantity_args[1..] {
    let (m, u) = is_quantity(q).unwrap();
    let from = unit_name(u);
    if let (Some(from), Some(to)) = (from, target_unit_name) {
      if from == to {
        magnitudes.push(m.clone());
      } else {
        match convert_magnitude(m, from, to) {
          Ok(converted) => magnitudes.push(converted),
          Err(e) => return Some(Err(e)),
        }
      }
    } else {
      // Unknown units but same string — just add directly
      magnitudes.push(m.clone());
    }
  }

  // Sum the magnitudes
  let sum_mag = match crate::functions::math_ast::plus_ast(&magnitudes) {
    Ok(s) => s,
    Err(e) => return Some(Err(e)),
  };

  let result = make_quantity(sum_mag, first_unit.clone());

  if other_args.is_empty() {
    Some(Ok(result))
  } else {
    // Mix of Quantity and non-Quantity — return unevaluated Plus with quantities combined
    other_args.push(result);
    // Sort: non-quantity first, quantity last (matching Wolfram behavior)
    Some(Ok(Expr::FunctionCall {
      name: "Plus".to_string(),
      args: other_args,
    }))
  }
}

/// Handle Times when Quantity arguments are present.
pub fn try_quantity_times(
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  let has_quantity = args.iter().any(|a| is_quantity(a).is_some());
  if !has_quantity {
    return None;
  }

  let mut quantity_args: Vec<(Expr, Expr)> = Vec::new(); // (magnitude, unit)
  let mut scalar_args: Vec<Expr> = Vec::new();

  for arg in args {
    if let Some((m, u)) = is_quantity(arg) {
      quantity_args.push((m.clone(), u.clone()));
    } else {
      scalar_args.push(arg.clone());
    }
  }

  if quantity_args.len() == 1 {
    // scalar * Quantity → Quantity[scalar * mag, unit]
    let (mag, unit) = &quantity_args[0];
    let mut all_mags = scalar_args;
    all_mags.push(mag.clone());
    let new_mag = match crate::functions::math_ast::times_ast(&all_mags) {
      Ok(m) => m,
      Err(e) => return Some(Err(e)),
    };
    Some(Ok(make_quantity(new_mag, unit.clone())))
  } else {
    // Multiple Quantities → compound unit
    // Quantity[a, u1] * Quantity[b, u2] → Quantity[a*b, u1*u2]
    let mut all_mags: Vec<Expr> = scalar_args;
    let mut unit_parts: Vec<Expr> = Vec::new();

    for (mag, unit) in &quantity_args {
      all_mags.push(mag.clone());
      unit_parts.push(unit.clone());
    }

    let new_mag = match crate::functions::math_ast::times_ast(&all_mags) {
      Ok(m) => m,
      Err(e) => return Some(Err(e)),
    };

    // Build compound unit: u1*u2*...
    let compound_unit = if unit_parts.len() == 1 {
      unit_parts.remove(0)
    } else {
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: unit_parts,
      }
    };

    Some(Ok(make_quantity(new_mag, compound_unit)))
  }
}

/// Handle Divide when Quantity arguments are present.
pub fn try_quantity_divide(
  a: &Expr,
  b: &Expr,
) -> Option<Result<Expr, InterpreterError>> {
  let q_a = is_quantity(a);
  let q_b = is_quantity(b);

  match (q_a, q_b) {
    (Some((mag_a, unit_a)), Some((mag_b, unit_b))) => {
      // Quantity / Quantity → Quantity[a/b, u1/u2]
      let new_mag = match crate::functions::math_ast::divide_ast(&[
        mag_a.clone(),
        mag_b.clone(),
      ]) {
        Ok(m) => m,
        Err(e) => return Some(Err(e)),
      };
      if units_equal(unit_a, unit_b) {
        // Same units cancel out
        Some(Ok(new_mag))
      } else {
        let compound_unit = Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: Box::new(unit_a.clone()),
          right: Box::new(unit_b.clone()),
        };
        Some(Ok(make_quantity(new_mag, compound_unit)))
      }
    }
    (Some((mag, unit)), None) => {
      // Quantity / scalar
      let new_mag =
        match crate::functions::math_ast::divide_ast(&[mag.clone(), b.clone()])
        {
          Ok(m) => m,
          Err(e) => return Some(Err(e)),
        };
      Some(Ok(make_quantity(new_mag, unit.clone())))
    }
    (None, Some((mag, unit))) => {
      // scalar / Quantity → Quantity[scalar/mag, unit^-1]
      let new_mag =
        match crate::functions::math_ast::divide_ast(&[a.clone(), mag.clone()])
        {
          Ok(m) => m,
          Err(e) => return Some(Err(e)),
        };
      let inv_unit = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Power,
        left: Box::new(unit.clone()),
        right: Box::new(Expr::Integer(-1)),
      };
      Some(Ok(make_quantity(new_mag, inv_unit)))
    }
    _ => None,
  }
}

/// Handle Power when base is Quantity.
pub fn try_quantity_power(
  base: &Expr,
  exp: &Expr,
) -> Option<Result<Expr, InterpreterError>> {
  let q = is_quantity(base);
  q?;

  let (mag, unit) = q.unwrap();

  let new_mag =
    match crate::functions::math_ast::power_ast(&[mag.clone(), exp.clone()]) {
      Ok(m) => m,
      Err(e) => return Some(Err(e)),
    };

  let new_unit = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Power,
    left: Box::new(unit.clone()),
    right: Box::new(exp.clone()),
  };

  Some(Ok(make_quantity(new_mag, new_unit)))
}

/// Compare two Quantity expressions. Returns Some(ordering) if comparable.
pub fn try_quantity_compare(
  left: &Expr,
  right: &Expr,
) -> Option<std::cmp::Ordering> {
  let (mag_l, unit_l) = is_quantity(left)?;
  let (mag_r, unit_r) = is_quantity(right)?;

  if !units_compatible(unit_l, unit_r) {
    return None;
  }

  // Convert right to left's unit
  let from = unit_name(unit_r)?;
  let to = unit_name(unit_l)?;

  let converted_r = if from == to {
    mag_r.clone()
  } else {
    convert_magnitude(mag_r, from, to).ok()?
  };

  // Try numeric comparison
  let l = crate::functions::math_ast::try_eval_to_f64(mag_l)?;
  let r = crate::functions::math_ast::try_eval_to_f64(&converted_r)?;
  l.partial_cmp(&r)
}
