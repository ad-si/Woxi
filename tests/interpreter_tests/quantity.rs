use super::*;

// ─── Construction ───────────────────────────────────────────────────────────

#[test]
fn quantity_two_args() {
  assert_eq!(
    interpret("Quantity[3, \"Meters\"]").unwrap(),
    "Quantity[3, Meters]"
  );
}

#[test]
fn quantity_one_arg() {
  assert_eq!(
    interpret("Quantity[\"Meters\"]").unwrap(),
    "Quantity[1, Meters]"
  );
}

#[test]
fn quantity_zero_magnitude() {
  assert_eq!(
    interpret("Quantity[0, \"Meters\"]").unwrap(),
    "Quantity[0, Meters]"
  );
}

#[test]
fn quantity_rational_magnitude() {
  assert_eq!(
    interpret("Quantity[1/3, \"Meters\"]").unwrap(),
    "Quantity[1/3, Meters]"
  );
}

#[test]
fn quantity_real_magnitude() {
  assert_eq!(
    interpret("Quantity[2.5, \"Meters\"]").unwrap(),
    "Quantity[2.5, Meters]"
  );
}

#[test]
fn quantity_unit_normalized_to_identifier() {
  // String unit should become Identifier in output (no quotes)
  let result = interpret("Quantity[1, \"Kilometers\"]").unwrap();
  assert!(result.contains("Kilometers"));
  assert!(!result.contains("\"Kilometers\""));
}

// ─── Compound units ─────────────────────────────────────────────────────────

#[test]
fn quantity_compound_divide_unit() {
  assert_eq!(
    interpret("Quantity[60, \"Miles\"/\"Hours\"]").unwrap(),
    "Quantity[60, Miles/Hours]"
  );
}

#[test]
fn quantity_compound_power_unit() {
  assert_eq!(
    interpret("Quantity[1, \"Meters\"/\"Seconds\"^2]").unwrap(),
    "Quantity[1, Meters/Seconds^2]"
  );
}

// ─── Same-unit addition ─────────────────────────────────────────────────────

#[test]
fn quantity_add_same_unit() {
  assert_eq!(
    interpret("Quantity[3, \"Meters\"] + Quantity[2, \"Meters\"]").unwrap(),
    "Quantity[5, Meters]"
  );
}

#[test]
fn quantity_subtract_same_unit() {
  assert_eq!(
    interpret("Quantity[3, \"Meters\"] - Quantity[1, \"Meters\"]").unwrap(),
    "Quantity[2, Meters]"
  );
}

// ─── Cross-unit addition ────────────────────────────────────────────────────

#[test]
fn quantity_add_compatible_units() {
  assert_eq!(
    interpret("Quantity[3, \"Meters\"] + Quantity[2, \"Kilometers\"]").unwrap(),
    "Quantity[2003, Meters]"
  );
}

#[test]
fn quantity_add_incompatible_units() {
  // Should return unevaluated (plus an error message on stderr)
  let result =
    interpret("Quantity[3, \"Meters\"] + Quantity[2, \"Seconds\"]").unwrap();
  assert!(result.contains("Quantity"));
  assert!(result.contains("Plus") || result.contains("+"));
}

// ─── Mixed addition (Quantity + number) ─────────────────────────────────────

#[test]
fn quantity_add_with_number() {
  let result = interpret("Quantity[3, \"Meters\"] + 5").unwrap();
  assert_eq!(result, "5 + Quantity[3, Meters]");
}

// ─── Scalar multiplication ──────────────────────────────────────────────────

#[test]
fn quantity_multiply_by_scalar() {
  assert_eq!(
    interpret("Quantity[3, \"Meters\"] * 5").unwrap(),
    "Quantity[15, Meters]"
  );
}

#[test]
fn quantity_scalar_times_quantity() {
  assert_eq!(
    interpret("2 * Quantity[3, \"Meters\"]").unwrap(),
    "Quantity[6, Meters]"
  );
}

// ─── Quantity × Quantity ────────────────────────────────────────────────────

#[test]
fn quantity_multiply_different_units() {
  assert_eq!(
    interpret("Quantity[3, \"Meters\"] * Quantity[2, \"Seconds\"]").unwrap(),
    "Quantity[6, Meters*Seconds]"
  );
}

// ─── Division ───────────────────────────────────────────────────────────────

#[test]
fn quantity_divide_different_units() {
  assert_eq!(
    interpret("Quantity[10, \"Meters\"] / Quantity[2, \"Seconds\"]").unwrap(),
    "Quantity[5, Meters/Seconds]"
  );
}

#[test]
fn quantity_divide_same_units() {
  assert_eq!(
    interpret("Quantity[10, \"Meters\"] / Quantity[2, \"Meters\"]").unwrap(),
    "5"
  );
}

#[test]
fn quantity_divide_by_scalar() {
  assert_eq!(
    interpret("Quantity[10, \"Meters\"] / 2").unwrap(),
    "Quantity[5, Meters]"
  );
}

// ─── Power ──────────────────────────────────────────────────────────────────

#[test]
fn quantity_power() {
  assert_eq!(
    interpret("Quantity[3, \"Meters\"]^2").unwrap(),
    "Quantity[9, Meters^2]"
  );
}

// ─── QuantityMagnitude ──────────────────────────────────────────────────────

#[test]
fn quantity_magnitude_basic() {
  assert_eq!(
    interpret("QuantityMagnitude[Quantity[5, \"Meters\"]]").unwrap(),
    "5"
  );
}

#[test]
fn quantity_magnitude_with_conversion() {
  assert_eq!(
    interpret("QuantityMagnitude[Quantity[1, \"Kilometers\"], \"Meters\"]")
      .unwrap(),
    "1000"
  );
}

// ─── QuantityUnit ───────────────────────────────────────────────────────────

#[test]
fn quantity_unit_basic() {
  assert_eq!(
    interpret("QuantityUnit[Quantity[5, \"Meters\"]]").unwrap(),
    "Meters"
  );
}

// ─── QuantityQ ──────────────────────────────────────────────────────────────

#[test]
fn quantity_q_true() {
  assert_eq!(
    interpret("QuantityQ[Quantity[5, \"Meters\"]]").unwrap(),
    "True"
  );
}

#[test]
fn quantity_q_false() {
  assert_eq!(interpret("QuantityQ[5]").unwrap(), "False");
}

// ─── CompatibleUnitQ ────────────────────────────────────────────────────────

#[test]
fn compatible_unit_q_true() {
  assert_eq!(
    interpret(
      "CompatibleUnitQ[Quantity[1, \"Meters\"], Quantity[1, \"Kilometers\"]]"
    )
    .unwrap(),
    "True"
  );
}

#[test]
fn compatible_unit_q_false() {
  assert_eq!(
    interpret(
      "CompatibleUnitQ[Quantity[1, \"Meters\"], Quantity[1, \"Seconds\"]]"
    )
    .unwrap(),
    "False"
  );
}

// ─── UnitConvert ────────────────────────────────────────────────────────────

#[test]
fn unit_convert_km_to_m() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Kilometers\"], \"Meters\"]").unwrap(),
    "Quantity[1000, Meters]"
  );
}

#[test]
fn unit_convert_cm_to_m() {
  assert_eq!(
    interpret("UnitConvert[Quantity[100, \"Centimeters\"], \"Meters\"]")
      .unwrap(),
    "Quantity[1, Meters]"
  );
}

#[test]
fn unit_convert_hours_to_seconds() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Hours\"], \"Seconds\"]").unwrap(),
    "Quantity[3600, Seconds]"
  );
}

#[test]
fn unit_convert_minutes_to_seconds() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Minutes\"], \"Seconds\"]").unwrap(),
    "Quantity[60, Seconds]"
  );
}

#[test]
fn unit_convert_kg_to_g() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Kilograms\"], \"Grams\"]").unwrap(),
    "Quantity[1000, Grams]"
  );
}

#[test]
fn unit_convert_liters_to_ml() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Liters\"], \"Milliliters\"]").unwrap(),
    "Quantity[1000, Milliliters]"
  );
}

#[test]
fn unit_convert_feet_to_meters() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Feet\"], \"Meters\"]").unwrap(),
    "Quantity[381/1250, Meters]"
  );
}

#[test]
fn unit_convert_inches_to_cm() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Inches\"], \"Centimeters\"]").unwrap(),
    "Quantity[127/50, Centimeters]"
  );
}

#[test]
fn unit_convert_miles_to_km() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Miles\"], \"Kilometers\"]").unwrap(),
    "Quantity[25146/15625, Kilometers]"
  );
}

#[test]
fn unit_convert_miles_to_m() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Miles\"], \"Meters\"]").unwrap(),
    "Quantity[201168/125, Meters]"
  );
}

#[test]
fn unit_convert_pounds_to_kg() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Pounds\"], \"Kilograms\"]").unwrap(),
    "Quantity[45359237/100000000, Kilograms]"
  );
}

// ─── Comparisons ────────────────────────────────────────────────────────────

#[test]
fn quantity_greater_same_unit() {
  assert_eq!(
    interpret("Quantity[5, \"Meters\"] > Quantity[3, \"Meters\"]").unwrap(),
    "True"
  );
}

#[test]
fn quantity_less_same_unit() {
  assert_eq!(
    interpret("Quantity[3, \"Meters\"] < Quantity[5, \"Meters\"]").unwrap(),
    "True"
  );
}

#[test]
fn quantity_equal_same_unit() {
  assert_eq!(
    interpret("Quantity[5, \"Meters\"] == Quantity[5, \"Meters\"]").unwrap(),
    "True"
  );
}

#[test]
fn quantity_greater_cross_unit() {
  assert_eq!(
    interpret("Quantity[1, \"Kilometers\"] > Quantity[500, \"Meters\"]")
      .unwrap(),
    "True"
  );
}

#[test]
fn quantity_less_cross_unit() {
  assert_eq!(
    interpret("Quantity[500, \"Meters\"] < Quantity[1, \"Kilometers\"]")
      .unwrap(),
    "True"
  );
}

// ─── Compound unit simplification ──────────────────────────────────────────

#[test]
fn quantity_divide_speed_by_time() {
  // km/h divided by Seconds → Kilometers/Seconds^2
  // (merges Hours and Seconds into Seconds)
  let result = interpret(
    "Quantity[100, \"Kilometers\"/\"Hours\"] / Quantity[3.2, \"Seconds\"]",
  )
  .unwrap();
  assert_eq!(
    result,
    "Quantity[0.008680555555555556, Kilometers/Seconds^2]"
  );
}

#[test]
fn quantity_multiply_accel_by_time() {
  // Kilometers/Seconds^2 * Days → Kilometers/Seconds
  let result = interpret(
    "Quantity[0.008680555555555556, \"Kilometers\"/\"Seconds\"^2] * Quantity[10, \"Days\"]",
  )
  .unwrap();
  assert_eq!(result, "Quantity[7500., Kilometers/Seconds]");
}

// ─── Compound UnitConvert ─────────────────────────────────────────────────

#[test]
fn unit_convert_compound_km_s_to_km_h() {
  assert_eq!(
    interpret(
      "UnitConvert[Quantity[7500, \"Kilometers\"/\"Seconds\"], \"Kilometers\"/\"Hours\"]"
    )
    .unwrap(),
    "Quantity[27000000, Kilometers/Hours]"
  );
}

#[test]
fn unit_convert_m_s_to_km_h() {
  assert_eq!(
    interpret(
      "UnitConvert[Quantity[1, \"Meters\"/\"Seconds\"], \"Kilometers\"/\"Hours\"]"
    )
    .unwrap(),
    "Quantity[18/5, Kilometers/Hours]"
  );
}

// ─── Unit abbreviations ───────────────────────────────────────────────────

#[test]
fn quantity_abbreviation_km_h() {
  assert_eq!(
    interpret("Quantity[1, \"km/h\"]").unwrap(),
    "Quantity[1, Kilometers/Hours]"
  );
}

#[test]
fn quantity_abbreviation_m_s() {
  assert_eq!(
    interpret("Quantity[1, \"m/s\"]").unwrap(),
    "Quantity[1, Meters/Seconds]"
  );
}

#[test]
fn quantity_abbreviation_mph() {
  assert_eq!(
    interpret("Quantity[1, \"mph\"]").unwrap(),
    "Quantity[1, Miles/Hours]"
  );
}

#[test]
fn unit_convert_with_abbreviation_target() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Meters\"/\"Seconds\"], \"km/h\"]")
      .unwrap(),
    "Quantity[18/5, Kilometers/Hours]"
  );
}

// ─── SpeedOfLight ─────────────────────────────────────────────────────────

#[test]
fn unit_convert_speed_of_light_to_m_s() {
  assert_eq!(
    interpret(
      "UnitConvert[Quantity[1, \"SpeedOfLight\"], \"Meters\"/\"Seconds\"]"
    )
    .unwrap(),
    "Quantity[299792458, Meters/Seconds]"
  );
}

#[test]
fn unit_convert_speed_of_light_to_km_h() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"SpeedOfLight\"], \"km/h\"]").unwrap(),
    "Quantity[5396264244/5, Kilometers/Hours]"
  );
}

// ─── Compound unit addition ───────────────────────────────────────────────

#[test]
fn quantity_add_compound_compatible() {
  // km/h + m/s should add (both are velocity)
  let result = interpret(
    "Quantity[1, \"Kilometers\"/\"Hours\"] + Quantity[1, \"Meters\"/\"Seconds\"]",
  )
  .unwrap();
  // 1 + 18/5 = 23/5, but 18/5 as float is 3.6, so 1 + 3.6 = 4.6
  assert!(result.contains("Kilometers/Hours"));
}

// ─── Compound unit comparison ─────────────────────────────────────────────

#[test]
fn quantity_compare_compound_units() {
  assert_eq!(
    interpret(
      "Quantity[100, \"Kilometers\"/\"Hours\"] > Quantity[10, \"Meters\"/\"Seconds\"]"
    )
    .unwrap(),
    "True"
  );
}

// ─── CompatibleUnitQ with compound units ──────────────────────────────────

#[test]
fn compatible_unit_q_compound() {
  assert_eq!(
    interpret(
      "CompatibleUnitQ[Quantity[1, \"Kilometers\"/\"Hours\"], Quantity[1, \"Meters\"/\"Seconds\"]]"
    )
    .unwrap(),
    "True"
  );
}

#[test]
fn compatible_unit_q_compound_false() {
  assert_eq!(
    interpret(
      "CompatibleUnitQ[Quantity[1, \"Kilometers\"/\"Hours\"], Quantity[1, \"Meters\"]]"
    )
    .unwrap(),
    "False"
  );
}
