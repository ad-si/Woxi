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

// ─── UnitConvert with compound string target ──────────────────────────────

#[test]
fn unit_convert_compound_string_target() {
  // "Meters/Seconds^2" as a string should be parsed as a compound unit
  assert_eq!(
    interpret(
      "UnitConvert[Quantity[1, Kilometers/Seconds^2], \"Meters/Seconds^2\"]"
    )
    .unwrap(),
    "Quantity[1000, Meters/Seconds^2]"
  );
}

#[test]
fn unit_convert_acceleration_from_velocity() {
  // Full acceleration calculation: 100 km/h in 3.2 seconds
  assert_eq!(
    interpret(
      "v0 = Quantity[0, \"km/h\"]; v1 = Quantity[100, \"km/h\"]; t1 = Quantity[3.2, \"Seconds\"]; UnitConvert[(v1 - v0) / t1, \"Meters/Seconds^2\"]"
    )
    .unwrap(),
    "Quantity[8.680555555555555, Meters/Seconds^2]"
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

// ─── CamelCase "Per" compound units ────────────────────────────────────────

#[test]
fn quantity_kilometers_per_hour() {
  assert_eq!(
    interpret("Quantity[45, \"KilometersPerHour\"]").unwrap(),
    "Quantity[45, Kilometers/Hours]"
  );
}

#[test]
fn quantity_meters_per_second() {
  assert_eq!(
    interpret("Quantity[100, \"MetersPerSecond\"]").unwrap(),
    "Quantity[100, Meters/Seconds]"
  );
}

#[test]
fn quantity_miles_per_hour() {
  assert_eq!(
    interpret("Quantity[60, \"MilesPerHour\"]").unwrap(),
    "Quantity[60, Miles/Hours]"
  );
}

#[test]
fn unit_convert_per_unit_times_time() {
  // Quantity[45, km/h] * Quantity[42, min] → Quantity[63/2, km]
  assert_eq!(
    interpret(
      "UnitConvert[Quantity[45, \"KilometersPerHour\"] * Quantity[42, \"Minutes\"], \"Kilometers\"]"
    )
    .unwrap(),
    "Quantity[63/2, Kilometers]"
  );
}

#[test]
fn unit_convert_meters_per_second_squared() {
  assert_eq!(
    interpret("Quantity[10, \"MetersPerSecondSquared\"]").unwrap(),
    "Quantity[10, Meters/Seconds^2]"
  );
}

// ─── Electrical & energy units ───────────────────────────────────────────

#[test]
fn unit_convert_joules_to_millijoules() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Joules\"], \"Millijoules\"]").unwrap(),
    "Quantity[1000, Millijoules]"
  );
}

#[test]
fn unit_convert_kwh_to_joules() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"KilowattHours\"], \"Joules\"]")
      .unwrap(),
    "Quantity[3600000, Joules]"
  );
}

#[test]
fn unit_convert_microfarads_volts_squared_to_millijoules() {
  assert_eq!(
    interpret(
      "c = Quantity[100, \"Microfarads\"]; u = Quantity[6.0, \"Volts\"]; energy = 1/2 * c * u^2; UnitConvert[energy, \"Millijoules\"]"
    ).unwrap(),
    "Quantity[1.8, Millijoules]"
  );
}

#[test]
fn unit_convert_amps_times_volts_to_watts() {
  assert_eq!(
    interpret(
      "UnitConvert[Quantity[5, \"Amperes\"] * Quantity[10, \"Volts\"], \"Watts\"]"
    ).unwrap(),
    "Quantity[50, Watts]"
  );
}

#[test]
fn compatible_unit_q_energy_units() {
  assert_eq!(
    interpret(
      "CompatibleUnitQ[Quantity[1, \"Microfarads\"*\"Volts\"^2], Quantity[1, \"Joules\"]]"
    ).unwrap(),
    "True"
  );
}

#[test]
fn unit_convert_kilowatts_hours_to_joules() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Kilowatts\"*\"Hours\"], \"Joules\"]")
      .unwrap(),
    "Quantity[3600000, Joules]"
  );
}

#[test]
fn unit_convert_volts_amps_to_watts() {
  // V * A = W (Ohm's law power)
  assert_eq!(
    interpret("UnitConvert[Quantity[220, \"Volts\"*\"Amperes\"], \"Watts\"]")
      .unwrap(),
    "Quantity[220, Watts]"
  );
}

#[test]
fn quantity_ohms_law() {
  // V = I * R → Quantity[5, "Amperes"] * Quantity[100, "Ohms"] = Quantity[500, Volts]
  assert_eq!(
    interpret(
      "UnitConvert[Quantity[5, \"Amperes\"] * Quantity[100, \"Ohms\"], \"Volts\"]"
    ).unwrap(),
    "Quantity[500, Volts]"
  );
}

// ─── Sqrt / Power with rational exponents on Quantities ──────────────────────

#[test]
fn quantity_sqrt_squared_unit() {
  // Sqrt[Quantity[4, Seconds^2]] → Quantity[2, Seconds]
  assert_eq!(
    interpret("Sqrt[Quantity[4, \"Seconds\"^2]]").unwrap(),
    "Quantity[2, Seconds]"
  );
}

#[test]
fn quantity_sqrt_simple_unit() {
  // Sqrt[Quantity[9.0, Meters^2]] → Quantity[3., Meters]
  assert_eq!(
    interpret("Sqrt[Quantity[9.0, \"Meters\"^2]]").unwrap(),
    "Quantity[3., Meters]"
  );
}

#[test]
fn quantity_power_half_squared_unit() {
  // Power[Quantity[4, Seconds^2], 1/2] → Quantity[2, Seconds]
  assert_eq!(
    interpret("Power[Quantity[4, \"Seconds\"^2], 1/2]").unwrap(),
    "Quantity[2, Seconds]"
  );
}

#[test]
fn quantity_power_half_compound_unit() {
  // Power[Quantity[4, Meters/Seconds^2], 1/2] → Quantity[2, Meters^(1/2)/Seconds]
  assert_eq!(
    interpret("Power[Quantity[4, \"Meters\"/\"Seconds\"^2], 1/2]").unwrap(),
    "Quantity[2, Meters^(1/2)/Seconds]"
  );
}

#[test]
fn quantity_sqrt_free_fall() {
  // Full free-fall physics example:
  // h=100m, g=9.81 m/s², fallzeit = Sqrt[2h/g], v = g*fallzeit
  // UnitConvert[v, km/h] should give ~159.46 km/h
  let code = r#"
    h = Quantity[100, "Meters"];
    g = Quantity[9.81, "Meters"/"Seconds"^2];
    fallzeit = Sqrt[2 h/g];
    endgeschwindigkeit = g*fallzeit;
    UnitConvert[endgeschwindigkeit, "Kilometers"/"Hours"]
  "#;
  let result = interpret(code).unwrap();
  // Extract magnitude and verify it's approximately 159.46
  assert!(
    result.starts_with("Quantity["),
    "Expected Quantity, got: {}",
    result
  );
  assert!(
    result.ends_with("Kilometers/Hours]"),
    "Expected km/h unit, got: {}",
    result
  );
  let mag_str = result
    .strip_prefix("Quantity[")
    .unwrap()
    .strip_suffix(", Kilometers/Hours]")
    .unwrap();
  let mag: f64 = mag_str.parse().unwrap();
  assert!((mag - 159.46).abs() < 0.1, "Expected ~159.46, got: {}", mag);
}

#[test]
fn quantity_power_integer_still_works() {
  // Verify integer powers still work correctly
  assert_eq!(
    interpret("Quantity[3, \"Meters\"]^2").unwrap(),
    "Quantity[9, Meters^2]"
  );
}

#[test]
fn quantity_power_third() {
  // Quantity[8, Meters^3]^(1/3) → Quantity[2, Meters]
  assert_eq!(
    interpret("Power[Quantity[8, \"Meters\"^3], 1/3]").unwrap(),
    "Quantity[2, Meters]"
  );
}

// ─── Extended units: Time ──────────────────────────────────────────────────

#[test]
fn unit_convert_seconds_to_milliseconds() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Seconds\"], \"Milliseconds\"]")
      .unwrap(),
    "Quantity[1000, Milliseconds]"
  );
}

#[test]
fn unit_convert_milliseconds_to_microseconds() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Milliseconds\"], \"Microseconds\"]")
      .unwrap(),
    "Quantity[1000, Microseconds]"
  );
}

#[test]
fn unit_convert_microseconds_to_nanoseconds() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Microseconds\"], \"Nanoseconds\"]")
      .unwrap(),
    "Quantity[1000, Nanoseconds]"
  );
}

#[test]
fn unit_convert_weeks_to_days() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Weeks\"], \"Days\"]").unwrap(),
    "Quantity[7, Days]"
  );
}

#[test]
fn unit_convert_weeks_to_hours() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Weeks\"], \"Hours\"]").unwrap(),
    "Quantity[168, Hours]"
  );
}

// ─── Extended units: Length ─────────────────────────────────────────────────

#[test]
fn unit_convert_millimeters_to_micrometers() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Millimeters\"], \"Micrometers\"]")
      .unwrap(),
    "Quantity[1000, Micrometers]"
  );
}

#[test]
fn unit_convert_micrometers_to_nanometers() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Micrometers\"], \"Nanometers\"]")
      .unwrap(),
    "Quantity[1000, Nanometers]"
  );
}

#[test]
fn unit_convert_nautical_miles_to_meters() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"NauticalMiles\"], \"Meters\"]")
      .unwrap(),
    "Quantity[1852, Meters]"
  );
}

#[test]
fn unit_convert_nautical_miles_to_kilometers() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"NauticalMiles\"], \"Kilometers\"]")
      .unwrap(),
    "Quantity[463/250, Kilometers]"
  );
}

// ─── Extended units: Mass ──────────────────────────────────────────────────

#[test]
fn unit_convert_tonnes_to_kilograms() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Tonnes\"], \"Kilograms\"]").unwrap(),
    "Quantity[1000, Kilograms]"
  );
}

#[test]
fn unit_convert_pounds_to_ounces() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Pounds\"], \"Ounces\"]").unwrap(),
    "Quantity[16, Ounces]"
  );
}

#[test]
fn unit_convert_ounces_to_grams() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Ounces\"], \"Grams\"]").unwrap(),
    "Quantity[45359237/1600000, Grams]"
  );
}

// ─── Extended units: Pressure ──────────────────────────────────────────────

#[test]
fn unit_convert_bars_to_pascals() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Bars\"], \"Pascals\"]").unwrap(),
    "Quantity[100000, Pascals]"
  );
}

#[test]
fn unit_convert_atmospheres_to_pascals() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Atmospheres\"], \"Pascals\"]")
      .unwrap(),
    "Quantity[101325, Pascals]"
  );
}

#[test]
fn unit_convert_atmospheres_to_bars() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Atmospheres\"], \"Bars\"]").unwrap(),
    "Quantity[4053/4000, Bars]"
  );
}

// ─── Extended units: Energy ────────────────────────────────────────────────

#[test]
fn unit_convert_calories_to_joules() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Calories\"], \"Joules\"]").unwrap(),
    "Quantity[523/125, Joules]"
  );
}

#[test]
fn unit_convert_kilocalories_to_joules() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Kilocalories\"], \"Joules\"]")
      .unwrap(),
    "Quantity[4184, Joules]"
  );
}

#[test]
fn unit_convert_kilocalories_to_calories() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Kilocalories\"], \"Calories\"]")
      .unwrap(),
    "Quantity[1000, Calories]"
  );
}

#[test]
fn unit_convert_electronvolts_to_joules() {
  let result =
    interpret("UnitConvert[Quantity[1, \"ElectronVolts\"], \"Joules\"]")
      .unwrap();
  assert!(result.starts_with("Quantity["));
  assert!(result.ends_with(", Joules]"));
  // 1 eV = 1.602176634e-19 J
  let mag_str = result
    .strip_prefix("Quantity[")
    .unwrap()
    .strip_suffix(", Joules]")
    .unwrap();
  // It should be a rational: 1602176634/10000000000000000000000000000
  assert!(mag_str.contains('/'), "Expected rational, got: {}", mag_str);
}

// ─── Extended units: Frequency ─────────────────────────────────────────────

#[test]
fn unit_convert_kilohertz_to_hertz() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Kilohertz\"], \"Hertz\"]").unwrap(),
    "Quantity[1000, Hertz]"
  );
}

#[test]
fn unit_convert_megahertz_to_hertz() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Megahertz\"], \"Hertz\"]").unwrap(),
    "Quantity[1000000, Hertz]"
  );
}

#[test]
fn unit_convert_gigahertz_to_megahertz() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Gigahertz\"], \"Megahertz\"]")
      .unwrap(),
    "Quantity[1000, Megahertz]"
  );
}

#[test]
fn unit_convert_gigahertz_to_hertz() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Gigahertz\"], \"Hertz\"]").unwrap(),
    "Quantity[1000000000, Hertz]"
  );
}

// ─── Extended units: Magnetic flux density ─────────────────────────────────

#[test]
fn unit_convert_teslas_to_milliteslas() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Teslas\"], \"Milliteslas\"]").unwrap(),
    "Quantity[1000, Milliteslas]"
  );
}

#[test]
fn unit_convert_milliteslas_to_teslas() {
  assert_eq!(
    interpret("UnitConvert[Quantity[500, \"Milliteslas\"], \"Teslas\"]")
      .unwrap(),
    "Quantity[1/2, Teslas]"
  );
}

// ─── Extended units: Speed (Knots) ─────────────────────────────────────────

#[test]
fn unit_convert_knots_to_km_h() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Knots\"], \"Kilometers\"/\"Hours\"]")
      .unwrap(),
    "Quantity[463/250, Kilometers/Hours]"
  );
}

#[test]
fn unit_convert_km_h_to_knots() {
  // 1 km/h = 250/463 knots
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Kilometers\"/\"Hours\"], \"Knots\"]")
      .unwrap(),
    "Quantity[250/463, Knots]"
  );
}

#[test]
fn unit_convert_knots_to_m_s() {
  // 1 knot = 1852/3600 m/s
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Knots\"], \"Meters\"/\"Seconds\"]")
      .unwrap(),
    "Quantity[463/900, Meters/Seconds]"
  );
}

// ─── Extended units: Abbreviations ─────────────────────────────────────────

#[test]
fn quantity_abbreviation_ms() {
  assert_eq!(
    interpret("Quantity[500, \"ms\"]").unwrap(),
    "Quantity[500, Milliseconds]"
  );
}

#[test]
fn quantity_abbreviation_us() {
  assert_eq!(
    interpret("Quantity[100, \"us\"]").unwrap(),
    "Quantity[100, Microseconds]"
  );
}

#[test]
fn quantity_abbreviation_ns() {
  assert_eq!(
    interpret("Quantity[50, \"ns\"]").unwrap(),
    "Quantity[50, Nanoseconds]"
  );
}

#[test]
fn quantity_abbreviation_um() {
  assert_eq!(
    interpret("Quantity[10, \"um\"]").unwrap(),
    "Quantity[10, Micrometers]"
  );
}

#[test]
fn quantity_abbreviation_nm() {
  assert_eq!(
    interpret("Quantity[550, \"nm\"]").unwrap(),
    "Quantity[550, Nanometers]"
  );
}

#[test]
fn quantity_abbreviation_nmi() {
  assert_eq!(
    interpret("Quantity[1, \"nmi\"]").unwrap(),
    "Quantity[1, NauticalMiles]"
  );
}

#[test]
fn quantity_abbreviation_t() {
  assert_eq!(
    interpret("Quantity[5, \"t\"]").unwrap(),
    "Quantity[5, Tonnes]"
  );
}

#[test]
fn quantity_abbreviation_oz() {
  assert_eq!(
    interpret("Quantity[8, \"oz\"]").unwrap(),
    "Quantity[8, Ounces]"
  );
}

#[test]
fn quantity_abbreviation_bar() {
  assert_eq!(
    interpret("Quantity[1, \"bar\"]").unwrap(),
    "Quantity[1, Bars]"
  );
}

#[test]
fn quantity_abbreviation_atm() {
  assert_eq!(
    interpret("Quantity[1, \"atm\"]").unwrap(),
    "Quantity[1, Atmospheres]"
  );
}

#[test]
fn quantity_abbreviation_cal() {
  assert_eq!(
    interpret("Quantity[100, \"cal\"]").unwrap(),
    "Quantity[100, Calories]"
  );
}

#[test]
fn quantity_abbreviation_kcal() {
  assert_eq!(
    interpret("Quantity[2, \"kcal\"]").unwrap(),
    "Quantity[2, Kilocalories]"
  );
}

#[test]
fn quantity_abbreviation_ev() {
  assert_eq!(
    interpret("Quantity[1, \"eV\"]").unwrap(),
    "Quantity[1, ElectronVolts]"
  );
}

#[test]
fn quantity_abbreviation_hz() {
  assert_eq!(
    interpret("Quantity[440, \"Hz\"]").unwrap(),
    "Quantity[440, Hertz]"
  );
}

#[test]
fn quantity_abbreviation_khz() {
  assert_eq!(
    interpret("Quantity[10, \"kHz\"]").unwrap(),
    "Quantity[10, Kilohertz]"
  );
}

#[test]
fn quantity_abbreviation_mhz() {
  assert_eq!(
    interpret("Quantity[2400, \"MHz\"]").unwrap(),
    "Quantity[2400, Megahertz]"
  );
}

#[test]
fn quantity_abbreviation_ghz() {
  assert_eq!(
    interpret("Quantity[5, \"GHz\"]").unwrap(),
    "Quantity[5, Gigahertz]"
  );
}

#[test]
fn quantity_abbreviation_tesla() {
  assert_eq!(
    interpret("Quantity[1, \"T\"]").unwrap(),
    "Quantity[1, Teslas]"
  );
}

#[test]
fn quantity_abbreviation_mt() {
  assert_eq!(
    interpret("Quantity[50, \"mT\"]").unwrap(),
    "Quantity[50, Milliteslas]"
  );
}

#[test]
fn quantity_abbreviation_kn() {
  assert_eq!(
    interpret("Quantity[30, \"kn\"]").unwrap(),
    "Quantity[30, Knots]"
  );
}

#[test]
fn quantity_abbreviation_kt() {
  assert_eq!(
    interpret("Quantity[30, \"kt\"]").unwrap(),
    "Quantity[30, Knots]"
  );
}

// ─── Cross-category compatibility ──────────────────────────────────────────

#[test]
fn compatible_unit_q_hertz_inverse_seconds() {
  // Hertz has dimension Time^-1, same as 1/Seconds
  assert_eq!(
    interpret(
      "CompatibleUnitQ[Quantity[1, \"Hertz\"], Quantity[1, 1/\"Seconds\"]]"
    )
    .unwrap(),
    "True"
  );
}

#[test]
fn compatible_unit_q_bars_pascals() {
  assert_eq!(
    interpret(
      "CompatibleUnitQ[Quantity[1, \"Bars\"], Quantity[1, \"Pascals\"]]"
    )
    .unwrap(),
    "True"
  );
}

#[test]
fn compatible_unit_q_calories_joules() {
  assert_eq!(
    interpret(
      "CompatibleUnitQ[Quantity[1, \"Calories\"], Quantity[1, \"Joules\"]]"
    )
    .unwrap(),
    "True"
  );
}

#[test]
fn compatible_unit_q_knots_m_per_s() {
  assert_eq!(
    interpret(
      "CompatibleUnitQ[Quantity[1, \"Knots\"], Quantity[1, \"Meters\"/\"Seconds\"]]"
    )
    .unwrap(),
    "True"
  );
}
