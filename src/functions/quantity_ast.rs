use crate::InterpreterError;
use crate::syntax::Expr;
use std::collections::BTreeMap;

// ─── Unit dimension system ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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

// ─── Compound unit decomposition ────────────────────────────────────────────

/// A decomposed compound unit: list of (base_unit_name, exponent) pairs
/// plus a combined SI conversion factor as a rational.
struct CompoundUnitInfo {
  /// Each base unit and its exponent, e.g. [("Kilometers", 1), ("Seconds", -2)]
  components: Vec<(String, i64)>,
  /// Combined SI conversion factor as rational numer/denom
  si_numer: i128,
  si_denom: i128,
  /// Dimension exponent map for compatibility checking
  dimensions: BTreeMap<Dimension, i64>,
}

/// Raise a rational to an integer power.
fn rational_pow(numer: i128, denom: i128, exp: i64) -> (i128, i128) {
  if exp == 0 {
    return (1, 1);
  }
  if exp > 0 {
    (numer.pow(exp as u32), denom.pow(exp as u32))
  } else {
    // Negative exponent: swap numer/denom
    (denom.pow((-exp) as u32), numer.pow((-exp) as u32))
  }
}

/// Resolve common unit abbreviation strings to full unit names.
fn resolve_unit_abbreviation(s: &str) -> Option<Expr> {
  use crate::syntax::BinaryOperator;

  // Simple abbreviations → single unit
  let simple = match s {
    "m" => "Meters",
    "km" => "Kilometers",
    "cm" => "Centimeters",
    "mm" => "Millimeters",
    "ft" => "Feet",
    "in" => "Inches",
    "mi" => "Miles",
    "yd" => "Yards",
    "kg" => "Kilograms",
    "g" => "Grams",
    "mg" => "Milligrams",
    "lb" | "lbs" => "Pounds",
    "s" => "Seconds",
    "min" => "Minutes",
    "h" | "hr" => "Hours",
    "d" => "Days",
    "L" => "Liters",
    "mL" => "Milliliters",
    "gal" => "Gallons",
    _ => "",
  };
  if !simple.is_empty() {
    return Some(Expr::Identifier(simple.to_string()));
  }

  // Compound abbreviations
  let make_div = |n: &str, d: &str| -> Expr {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::Identifier(n.to_string())),
      right: Box::new(Expr::Identifier(d.to_string())),
    }
  };
  match s {
    "mph" => Some(make_div("Miles", "Hours")),
    "km/h" | "kph" => Some(make_div("Kilometers", "Hours")),
    "m/s" => Some(make_div("Meters", "Seconds")),
    _ => {
      // Try parsing "X/Y" pattern
      if let Some((num, den)) = s.split_once('/') {
        let num_expr = resolve_unit_abbreviation(num).or_else(|| {
          get_unit_info(num).map(|_| Expr::Identifier(num.to_string()))
        })?;
        let den_expr = resolve_unit_abbreviation(den).or_else(|| {
          get_unit_info(den).map(|_| Expr::Identifier(den.to_string()))
        })?;
        Some(Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(num_expr),
          right: Box::new(den_expr),
        })
      } else {
        None
      }
    }
  }
}

/// Resolve named compound constants like SpeedOfLight.
/// Returns (magnitude_factor_numer, magnitude_factor_denom, unit_components).
fn resolve_compound_constant(
  name: &str,
) -> Option<(i128, i128, Vec<(String, i64)>)> {
  match name {
    "SpeedOfLight" => Some((
      299792458,
      1,
      vec![("Meters".to_string(), 1), ("Seconds".to_string(), -1)],
    )),
    _ => None,
  }
}

/// Recursively decompose a unit Expr into a CompoundUnitInfo.
fn decompose_unit_expr(expr: &Expr) -> Option<CompoundUnitInfo> {
  match expr {
    Expr::Identifier(name) | Expr::String(name) => {
      // Try direct unit lookup
      if let Some(info) = get_unit_info(name) {
        let mut dims = BTreeMap::new();
        dims.insert(info.dimension, 1);
        return Some(CompoundUnitInfo {
          components: vec![(name.clone(), 1)],
          si_numer: info.to_si_numer,
          si_denom: info.to_si_denom,
          dimensions: dims,
        });
      }
      // Try compound constants (e.g. SpeedOfLight)
      if let Some((mag_n, mag_d, components)) = resolve_compound_constant(name)
      {
        let mut si_numer: i128 = mag_n;
        let mut si_denom: i128 = mag_d;
        let mut dims = BTreeMap::new();
        for (unit_name, exp) in &components {
          let uinfo = get_unit_info(unit_name)?;
          let (pn, pd) =
            rational_pow(uinfo.to_si_numer, uinfo.to_si_denom, *exp);
          si_numer *= pn;
          si_denom *= pd;
          *dims.entry(uinfo.dimension).or_insert(0) += exp;
        }
        let g = gcd(si_numer, si_denom);
        return Some(CompoundUnitInfo {
          components,
          si_numer: si_numer / g,
          si_denom: si_denom / g,
          dimensions: dims,
        });
      }
      // Try abbreviation resolution
      if let Some(resolved) = resolve_unit_abbreviation(name) {
        return decompose_unit_expr(&resolved);
      }
      None
    }
    Expr::BinaryOp { op, left, right } => {
      use crate::syntax::BinaryOperator;
      match op {
        BinaryOperator::Divide => {
          let mut left_info = decompose_unit_expr(left)?;
          let right_info = decompose_unit_expr(right)?;
          // Negate right exponents
          for (name, exp) in &right_info.components {
            left_info.components.push((name.clone(), -exp));
          }
          // SI factor: left / right
          left_info.si_numer *= right_info.si_denom;
          left_info.si_denom *= right_info.si_numer;
          let g = gcd(left_info.si_numer, left_info.si_denom);
          left_info.si_numer /= g;
          left_info.si_denom /= g;
          // Merge dimensions
          for (dim, exp) in &right_info.dimensions {
            *left_info.dimensions.entry(*dim).or_insert(0) -= exp;
          }
          left_info.dimensions.retain(|_, v| *v != 0);
          Some(left_info)
        }
        BinaryOperator::Times => {
          let mut left_info = decompose_unit_expr(left)?;
          let right_info = decompose_unit_expr(right)?;
          for (name, exp) in &right_info.components {
            left_info.components.push((name.clone(), *exp));
          }
          left_info.si_numer *= right_info.si_numer;
          left_info.si_denom *= right_info.si_denom;
          let g = gcd(left_info.si_numer, left_info.si_denom);
          left_info.si_numer /= g;
          left_info.si_denom /= g;
          for (dim, exp) in &right_info.dimensions {
            *left_info.dimensions.entry(*dim).or_insert(0) += exp;
          }
          left_info.dimensions.retain(|_, v| *v != 0);
          Some(left_info)
        }
        BinaryOperator::Power => {
          let base_info = decompose_unit_expr(left)?;
          let exp_val = match right.as_ref() {
            Expr::Integer(n) => *n as i64,
            _ => return None,
          };
          let components: Vec<(String, i64)> = base_info
            .components
            .iter()
            .map(|(n, e)| (n.clone(), e * exp_val))
            .collect();
          let (pn, pd) =
            rational_pow(base_info.si_numer, base_info.si_denom, exp_val);
          let g = gcd(pn, pd);
          let dims: BTreeMap<Dimension, i64> = base_info
            .dimensions
            .iter()
            .map(|(d, e)| (*d, e * exp_val))
            .filter(|(_, e)| *e != 0)
            .collect();
          Some(CompoundUnitInfo {
            components,
            si_numer: pn / g,
            si_denom: pd / g,
            dimensions: dims,
          })
        }
        _ => None,
      }
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut result: Option<CompoundUnitInfo> = None;
      for arg in args {
        let info = decompose_unit_expr(arg)?;
        match &mut result {
          None => result = Some(info),
          Some(r) => {
            for (name, exp) in &info.components {
              r.components.push((name.clone(), *exp));
            }
            r.si_numer *= info.si_numer;
            r.si_denom *= info.si_denom;
            let g = gcd(r.si_numer, r.si_denom);
            r.si_numer /= g;
            r.si_denom /= g;
            for (dim, exp) in &info.dimensions {
              *r.dimensions.entry(*dim).or_insert(0) += exp;
            }
            r.dimensions.retain(|_, v| *v != 0);
          }
        }
      }
      result
    }
    _ => None,
  }
}

/// Simplify compound unit components by merging same-dimension units.
/// Returns (simplified_components, conversion_factor_numer, conversion_factor_denom).
/// The conversion factor should be applied to the magnitude.
fn simplify_compound_unit(
  components: &[(String, i64)],
) -> (Vec<(String, i64)>, i128, i128) {
  // Group by dimension
  let mut dim_groups: BTreeMap<Dimension, Vec<(String, i64)>> = BTreeMap::new();
  let mut unknown: Vec<(String, i64)> = Vec::new();

  for (name, exp) in components {
    if let Some(info) = get_unit_info(name) {
      dim_groups
        .entry(info.dimension)
        .or_default()
        .push((name.clone(), *exp));
    } else {
      unknown.push((name.clone(), *exp));
    }
  }

  let mut conv_numer: i128 = 1;
  let mut conv_denom: i128 = 1;
  let mut simplified: Vec<(String, i64)> = Vec::new();

  for units in dim_groups.values() {
    // Find the "canonical" unit: the one with the largest total absolute exponent
    let canonical =
      units.iter().max_by_key(|(_, e)| e.abs()).unwrap().0.clone();
    let canonical_info = get_unit_info(&canonical).unwrap();

    let mut total_exp: i64 = 0;
    for (name, exp) in units {
      if name == &canonical {
        total_exp += exp;
      } else {
        // Convert this unit to canonical: factor = (this_si / canonical_si)^exp
        let this_info = get_unit_info(name).unwrap();
        // this_unit in SI = this_si_numer/this_si_denom
        // canonical in SI = can_si_numer/can_si_denom
        // conversion per unit = (this_si_n * can_si_d) / (this_si_d * can_si_n)
        let unit_conv_n = this_info.to_si_numer * canonical_info.to_si_denom;
        let unit_conv_d = this_info.to_si_denom * canonical_info.to_si_numer;
        let (pn, pd) = rational_pow(unit_conv_n, unit_conv_d, *exp);
        conv_numer *= pn;
        conv_denom *= pd;
        let g = gcd(conv_numer, conv_denom);
        conv_numer /= g;
        conv_denom /= g;
        total_exp += exp;
      }
    }

    if total_exp != 0 {
      simplified.push((canonical, total_exp));
    }
  }

  simplified.extend(unknown);

  (simplified, conv_numer, conv_denom)
}

/// Build a unit Expr from simplified components.
/// Positive exponents go in numerator, negative in denominator.
fn components_to_unit_expr(components: &[(String, i64)]) -> Expr {
  use crate::syntax::BinaryOperator;

  let mut numer_parts: Vec<Expr> = Vec::new();
  let mut denom_parts: Vec<Expr> = Vec::new();

  for (name, exp) in components {
    let base = Expr::Identifier(name.clone());
    let abs_exp = exp.abs();
    let part = if abs_exp == 1 {
      base
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(base),
        right: Box::new(Expr::Integer(abs_exp as i128)),
      }
    };
    if *exp > 0 {
      numer_parts.push(part);
    } else {
      denom_parts.push(part);
    }
  }

  let numer = if numer_parts.is_empty() {
    Expr::Integer(1)
  } else if numer_parts.len() == 1 {
    numer_parts.remove(0)
  } else {
    Expr::FunctionCall {
      name: "Times".to_string(),
      args: numer_parts,
    }
  };

  if denom_parts.is_empty() {
    numer
  } else {
    let denom = if denom_parts.len() == 1 {
      denom_parts.remove(0)
    } else {
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: denom_parts,
      }
    };
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(numer),
      right: Box::new(denom),
    }
  }
}

/// Decompose, simplify, and rebuild a compound unit expression.
/// Returns (simplified_unit_expr, conversion_factor_numer, conversion_factor_denom).
fn simplify_unit_expr(unit: &Expr) -> Option<(Expr, i128, i128)> {
  let info = decompose_unit_expr(unit)?;
  let (simplified, conv_n, conv_d) = simplify_compound_unit(&info.components);
  if simplified.is_empty() {
    return Some((Expr::Integer(1), conv_n, conv_d));
  }
  Some((components_to_unit_expr(&simplified), conv_n, conv_d))
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
  if let (Some(info1), Some(info2)) =
    (decompose_unit_expr(u1), decompose_unit_expr(u2))
  {
    return info1.dimensions == info2.dimensions;
  }
  // If we can't decompose, they're compatible only if identical
  crate::syntax::expr_to_string(u1) == crate::syntax::expr_to_string(u2)
}

/// Check if two unit expressions are the same unit.
fn units_equal(u1: &Expr, u2: &Expr) -> bool {
  crate::syntax::expr_to_string(u1) == crate::syntax::expr_to_string(u2)
}

/// Recursively normalize unit expressions: String → Identifier,
/// and expand abbreviations like "km/h" → Kilometers/Hours.
fn normalize_unit(unit: Expr) -> Expr {
  match unit {
    Expr::String(ref s) => {
      // Try abbreviation expansion first
      if get_unit_info(s).is_some() {
        Expr::Identifier(s.clone())
      } else if let Some(expanded) = resolve_unit_abbreviation(s) {
        normalize_unit(expanded)
      } else {
        Expr::Identifier(s.clone())
      }
    }
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

    // Try compound unit decomposition for both sides
    let from_info = decompose_unit_expr(unit);
    let to_info = decompose_unit_expr(target);

    if let (Some(from), Some(to)) = (from_info, to_info) {
      // Check dimension compatibility
      if from.dimensions != to.dimensions {
        return Err(InterpreterError::EvaluationError(format!(
          "{} and {} are incompatible units.",
          crate::syntax::expr_to_string(unit),
          crate::syntax::expr_to_string(target)
        )));
      }
      // Convert: new_mag = mag * (from_si / to_si)
      let conv_numer = from.si_numer * to.si_denom;
      let conv_denom = from.si_denom * to.si_numer;
      let new_mag =
        multiply_magnitude_by_rational(mag, conv_numer, conv_denom)?;
      // Build target unit expression from decomposed target components
      // (preserves the user's target unit naming)
      Ok(make_quantity(new_mag, normalize_unit(target.clone())))
    } else {
      // Fallback: return unevaluated
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
  let first_decomposed = decompose_unit_expr(first_unit);
  let mut magnitudes: Vec<Expr> = vec![first_mag.clone()];
  for q in &quantity_args[1..] {
    let (m, u) = is_quantity(q).unwrap();
    // Try compound unit conversion via decomposition
    if let (Some(from_info), Some(to_info)) =
      (&first_decomposed, decompose_unit_expr(u))
    {
      let conv_numer = to_info.si_numer * from_info.si_denom;
      let conv_denom = to_info.si_denom * from_info.si_numer;
      if conv_numer == conv_denom {
        magnitudes.push(m.clone());
      } else {
        match multiply_magnitude_by_rational(m, conv_numer, conv_denom) {
          Ok(converted) => magnitudes.push(converted),
          Err(e) => return Some(Err(e)),
        }
      }
    } else {
      // Fallback: simple unit names
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
        magnitudes.push(m.clone());
      }
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
    // Multiple Quantities → compound unit with simplification
    let mut all_mags: Vec<Expr> = scalar_args;
    let mut unit_parts: Vec<Expr> = Vec::new();

    for (mag, unit) in &quantity_args {
      all_mags.push(mag.clone());
      unit_parts.push(unit.clone());
    }

    let raw_compound = if unit_parts.len() == 1 {
      unit_parts.remove(0)
    } else {
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: unit_parts,
      }
    };

    // Try to simplify compound unit (merge same-dimension units)
    let (final_unit, extra_conv_n, extra_conv_d) =
      simplify_unit_expr(&raw_compound).unwrap_or((raw_compound, 1, 1));

    let mut new_mag = match crate::functions::math_ast::times_ast(&all_mags) {
      Ok(m) => m,
      Err(e) => return Some(Err(e)),
    };

    // Apply conversion factor from unit simplification
    if extra_conv_n != 1 || extra_conv_d != 1 {
      new_mag = match multiply_magnitude_by_rational(
        &new_mag,
        extra_conv_n,
        extra_conv_d,
      ) {
        Ok(m) => m,
        Err(e) => return Some(Err(e)),
      };
    }

    // If all units cancelled out, return bare magnitude
    if matches!(&final_unit, Expr::Integer(1)) {
      Some(Ok(new_mag))
    } else {
      Some(Ok(make_quantity(new_mag, final_unit)))
    }
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
      let mut new_mag = match crate::functions::math_ast::divide_ast(&[
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
        let raw_compound = Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: Box::new(unit_a.clone()),
          right: Box::new(unit_b.clone()),
        };
        // Try to simplify (merge same-dimension units)
        let (final_unit, conv_n, conv_d) =
          simplify_unit_expr(&raw_compound).unwrap_or((raw_compound, 1, 1));
        if conv_n != 1 || conv_d != 1 {
          new_mag =
            match multiply_magnitude_by_rational(&new_mag, conv_n, conv_d) {
              Ok(m) => m,
              Err(e) => return Some(Err(e)),
            };
        }
        if matches!(&final_unit, Expr::Integer(1)) {
          Some(Ok(new_mag))
        } else {
          Some(Ok(make_quantity(new_mag, final_unit)))
        }
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
      // Try to simplify
      let (final_unit, conv_n, conv_d) =
        simplify_unit_expr(&inv_unit).unwrap_or((inv_unit, 1, 1));
      let final_mag = if conv_n != 1 || conv_d != 1 {
        match multiply_magnitude_by_rational(&new_mag, conv_n, conv_d) {
          Ok(m) => m,
          Err(e) => return Some(Err(e)),
        }
      } else {
        new_mag
      };
      Some(Ok(make_quantity(final_mag, final_unit)))
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

  // Convert right magnitude to left's unit via decomposition
  let left_info = decompose_unit_expr(unit_l);
  let right_info = decompose_unit_expr(unit_r);

  let converted_r = if let (Some(li), Some(ri)) = (left_info, right_info) {
    let conv_numer = ri.si_numer * li.si_denom;
    let conv_denom = ri.si_denom * li.si_numer;
    if conv_numer == conv_denom {
      mag_r.clone()
    } else {
      multiply_magnitude_by_rational(mag_r, conv_numer, conv_denom).ok()?
    }
  } else {
    // Fallback to simple conversion
    let from = unit_name(unit_r)?;
    let to = unit_name(unit_l)?;
    if from == to {
      mag_r.clone()
    } else {
      convert_magnitude(mag_r, from, to).ok()?
    }
  };

  // Try numeric comparison
  let l = crate::functions::math_ast::try_eval_to_f64(mag_l)?;
  let r = crate::functions::math_ast::try_eval_to_f64(&converted_r)?;
  l.partial_cmp(&r)
}
