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
  // Unit name is shown as an identifier (without quotes) in output
  let result = interpret("Quantity[1, \"Kilometers\"]").unwrap();
  assert!(result.contains("Kilometers"));
}

// ─── Lowercase unit names ───────────────────────────────────────────────────

#[test]
fn quantity_lowercase_days() {
  assert_eq!(
    interpret("Quantity[7, \"days\"]").unwrap(),
    "Quantity[7, Days]"
  );
}

#[test]
fn quantity_lowercase_weeks() {
  assert_eq!(
    interpret("Quantity[2, \"weeks\"]").unwrap(),
    "Quantity[2, Weeks]"
  );
}

#[test]
fn quantity_lowercase_addition() {
  assert_eq!(
    interpret("Quantity[7, \"days\"] + Quantity[2, \"weeks\"]").unwrap(),
    "Quantity[21, Days]"
  );
}

#[test]
fn quantity_lowercase_meters() {
  assert_eq!(
    interpret("Quantity[5, \"meters\"]").unwrap(),
    "Quantity[5, Meters]"
  );
}

#[test]
fn quantity_lowercase_kilometers() {
  // Adding km + m picks the smaller unit (meters) to match Mathematica.
  assert_eq!(
    interpret("Quantity[1, \"kilometers\"] + Quantity[500, \"meters\"]")
      .unwrap(),
    "Quantity[1500, Meters]"
  );
}

#[test]
fn quantity_lowercase_hours_minutes() {
  // Adding hours + minutes picks the smaller unit (minutes).
  assert_eq!(
    interpret("Quantity[3, \"hours\"] + Quantity[30, \"minutes\"]").unwrap(),
    "Quantity[210, Minutes]"
  );
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
fn quantity_add_picks_smaller_unit_order_independent() {
  // Result unit is picked by SI scale, not argument order: meter + cm = cm.
  assert_eq!(
    interpret("Quantity[6, \"meter\"] + Quantity[3, \"centimeter\"]").unwrap(),
    "Quantity[603, Centimeters]"
  );
  assert_eq!(
    interpret("Quantity[3, \"centimeter\"] + Quantity[6, \"meter\"]").unwrap(),
    "Quantity[603, Centimeters]"
  );
}

#[test]
fn quantity_add_picks_smaller_unit_gram_vs_kilogram() {
  assert_eq!(
    interpret("Quantity[1, \"kilogram\"] + Quantity[500, \"gram\"]").unwrap(),
    "Quantity[1500, Grams]"
  );
}

#[test]
fn quantity_add_incompatible_units_canonically_ordered() {
  // Incompatible units stay unevaluated inside Plus, but the operands are
  // sorted canonically (magnitude first): 3 < 6 ⇒ Seconds precedes Meters.
  assert_eq!(
    interpret("Quantity[6, \"meter\"] + Quantity[3, \"second\"]").unwrap(),
    "Quantity[3, Seconds] + Quantity[6, Meters]"
  );
  assert_eq!(
    interpret("Quantity[3, \"seconds\"] + Quantity[6, \"meters\"]").unwrap(),
    "Quantity[3, Seconds] + Quantity[6, Meters]"
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

// The factors of a compound product unit are ordered alphabetically by name,
// matching wolframscript (e.g. Watts*Hours displays as Hours*Watts),
// independent of the operand order.
#[test]
fn quantity_multiply_alphabetical_unit_order() {
  assert_eq!(
    interpret("Quantity[100, \"Watts\"] * Quantity[2, \"Hours\"]").unwrap(),
    "Quantity[200, Hours*Watts]"
  );
  assert_eq!(
    interpret("Quantity[1, \"Hours\"] * Quantity[1, \"Watts\"]").unwrap(),
    "Quantity[1, Hours*Watts]"
  );
  assert_eq!(
    interpret("Quantity[1, \"Volts\"] * Quantity[1, \"Amperes\"]").unwrap(),
    "Quantity[1, Amperes*Volts]"
  );
}

// ─── Unary functions thread through the magnitude ───────────────────────────

// Abs/Floor/Ceiling/Round/IntegerPart/Re/Im/Conjugate apply to the magnitude
// and keep the unit; Sign reduces to the dimensionless sign.
#[test]
fn quantity_unary_keeps_unit() {
  assert_eq!(
    interpret("Abs[Quantity[-5, \"Meters\"]]").unwrap(),
    "Quantity[5, Meters]"
  );
  assert_eq!(
    interpret("Floor[Quantity[5.7, \"Meters\"]]").unwrap(),
    "Quantity[5, Meters]"
  );
  assert_eq!(
    interpret("Ceiling[Quantity[5.2, \"Meters\"]]").unwrap(),
    "Quantity[6, Meters]"
  );
  assert_eq!(
    interpret("Round[Quantity[5.6, \"Meters\"]]").unwrap(),
    "Quantity[6, Meters]"
  );
  assert_eq!(
    interpret("IntegerPart[Quantity[5.7, \"Meters\"]]").unwrap(),
    "Quantity[5, Meters]"
  );
  // Abs of a complex magnitude collapses to its modulus, keeping the unit.
  assert_eq!(
    interpret("Abs[Quantity[3 - 4 I, \"Meters\"]]").unwrap(),
    "Quantity[5, Meters]"
  );
}

#[test]
fn quantity_sign_is_dimensionless() {
  assert_eq!(interpret("Sign[Quantity[-5, \"Meters\"]]").unwrap(), "-1");
  assert_eq!(interpret("Sign[Quantity[5, \"Meters\"]]").unwrap(), "1");
}

// Mod over two compatible quantities converts the dividend to the divisor's
// unit, takes the modulus, and returns the result in the divisor's unit.
#[test]
fn quantity_mod_same_unit() {
  assert_eq!(
    interpret("Mod[Quantity[7, \"Meters\"], Quantity[3, \"Meters\"]]").unwrap(),
    "Quantity[1, Meters]"
  );
  // Wolfram's Mod is non-negative.
  assert_eq!(
    interpret("Mod[Quantity[-7, \"Meters\"], Quantity[3, \"Meters\"]]")
      .unwrap(),
    "Quantity[2, Meters]"
  );
}

#[test]
fn quantity_mod_converts_to_divisor_unit() {
  // 7 m = 700 cm; 700 mod 300 = 100, returned in centimeters.
  assert_eq!(
    interpret("Mod[Quantity[7, \"Meters\"], Quantity[300, \"Centimeters\"]]")
      .unwrap(),
    "Quantity[100, Centimeters]"
  );
}

#[test]
fn quantity_mod_incompatible_units_unevaluated() {
  assert_eq!(
    interpret("Mod[Quantity[7, \"Meters\"], Quantity[2, \"Seconds\"]]")
      .unwrap(),
    "Mod[Quantity[7, Meters], Quantity[2, Seconds]]"
  );
}

// Sign predicates test the magnitude and return a Boolean.
#[test]
fn quantity_sign_predicates() {
  assert_eq!(
    interpret("Positive[Quantity[5, \"Meters\"]]").unwrap(),
    "True"
  );
  assert_eq!(
    interpret("Positive[Quantity[-5, \"Meters\"]]").unwrap(),
    "False"
  );
  assert_eq!(
    interpret("Negative[Quantity[-3, \"Meters\"]]").unwrap(),
    "True"
  );
  assert_eq!(
    interpret("NonNegative[Quantity[0, \"Meters\"]]").unwrap(),
    "True"
  );
  assert_eq!(
    interpret("NonPositive[Quantity[-1, \"Meters\"]]").unwrap(),
    "True"
  );
}

// Max/Min over quantities compare magnitudes after unit conversion and return
// the winning quantity in its original unit.
#[test]
fn quantity_max_min_same_unit() {
  assert_eq!(
    interpret("Max[Quantity[1, \"Meters\"], Quantity[2, \"Meters\"]]").unwrap(),
    "Quantity[2, Meters]"
  );
  assert_eq!(
    interpret("Min[Quantity[1, \"Meters\"], Quantity[2, \"Meters\"]]").unwrap(),
    "Quantity[1, Meters]"
  );
  // List argument form.
  assert_eq!(
    interpret("Max[{Quantity[1, \"Meters\"], Quantity[2, \"Meters\"]}]")
      .unwrap(),
    "Quantity[2, Meters]"
  );
}

#[test]
fn quantity_max_min_mixed_units() {
  // 1 m > 50 cm, returned in its original unit.
  assert_eq!(
    interpret("Max[Quantity[1, \"Meters\"], Quantity[50, \"Centimeters\"]]")
      .unwrap(),
    "Quantity[1, Meters]"
  );
  assert_eq!(
    interpret("Min[Quantity[1, \"Meters\"], Quantity[50, \"Centimeters\"]]")
      .unwrap(),
    "Quantity[50, Centimeters]"
  );
}

#[test]
fn quantity_max_incompatible_units_unevaluated() {
  // Length vs time cannot be compared, so the call stays unevaluated.
  assert_eq!(
    interpret("Max[Quantity[1, \"Meters\"], Quantity[2, \"Seconds\"]]")
      .unwrap(),
    "Max[Quantity[1, Meters], Quantity[2, Seconds]]"
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

#[test]
fn quantity_q_unknown_unit_false() {
  // 'Maters' isn't a known unit, so Quantity[3, "Maters"] fails QuantityQ —
  // matching wolframscript's stricter behaviour.
  assert_eq!(
    interpret(r#"QuantityQ[Quantity[3, "Maters"]]"#).unwrap(),
    "False"
  );
}

#[test]
fn quantity_q_compound_known_unit_true() {
  assert_eq!(
    interpret(r#"QuantityQ[Quantity[3, "Meters"/"Seconds"]]"#).unwrap(),
    "True"
  );
}

#[test]
fn quantity_q_bare_identifier_unit_false() {
  // Bare symbols (not strings) are not valid unit specifications in Wolfram —
  // Quantity[2, Second] stays unevaluated and QuantityQ returns False.
  assert_eq!(
    interpret("QuantityQ[Quantity[2, Second]]").unwrap(),
    "False"
  );
  assert_eq!(
    interpret("QuantityQ[Quantity[2, Seconds]]").unwrap(),
    "False"
  );
  assert_eq!(
    interpret("QuantityQ[Quantity[2, Meters]]").unwrap(),
    "False"
  );
}

#[test]
fn quantity_bare_identifier_unit_preserved() {
  // Bare identifiers should be preserved as-is, not coerced to canonical strings.
  assert_eq!(
    interpret("Quantity[2, Second]").unwrap(),
    "Quantity[2, Second]"
  );
  assert_eq!(
    interpret("Quantity[2, Meters]").unwrap(),
    "Quantity[2, Meters]"
  );
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

// One-argument UnitConvert converts to SI base units. Volume is Length^3, so
// the SI base of a volume unit is Meters^3 (not Liters).
#[test]
fn unit_convert_one_arg_to_si_base() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Liters\"]]").unwrap(),
    "Quantity[1/1000, Meters^3]"
  );
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Hours\"]]").unwrap(),
    "Quantity[3600, Seconds]"
  );
}

// Derived SI units display their compound base with Wolfram-matching
// parenthesization: product numerators and product denominators are grouped,
// so a/(b*c) and (a*b)/c render unambiguously.
#[test]
fn unit_convert_derived_unit_display() {
  // Single numerator, product denominator → a/(b*c).
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Bars\"]]").unwrap(),
    "Quantity[100000, Kilograms/(Meters*Seconds^2)]"
  );
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Pascals\"]]").unwrap(),
    "Quantity[1, Kilograms/(Meters*Seconds^2)]"
  );
  // Product numerator, single denominator → (a*b)/c.
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Newtons\"]]").unwrap(),
    "Quantity[1, (Kilograms*Meters)/Seconds^2]"
  );
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Joules\"]]").unwrap(),
    "Quantity[1, (Kilograms*Meters^2)/Seconds^2]"
  );
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Watts\"]]").unwrap(),
    "Quantity[1, (Kilograms*Meters^2)/Seconds^3]"
  );
}

// Volume units (Length^3) interconvert with Meters^3 and with each other.
#[test]
fn unit_convert_volume_units() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Liters\"], \"Meters\"^3]").unwrap(),
    "Quantity[1/1000, Meters^3]"
  );
  assert_eq!(
    interpret("UnitConvert[Quantity[1000, \"Milliliters\"], \"Liters\"]")
      .unwrap(),
    "Quantity[1, Liters]"
  );
  assert_eq!(
    interpret("UnitDimensions[Quantity[1, \"Liters\"]]").unwrap(),
    "{{LengthUnit, 3}}"
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

// Regression: the magnitude-scaling used to compute `(f * numer) / denom`,
// which introduced a ULP-sized error (1.7236510059999999 instead of
// 1.723651006). Collapsing `numer/denom` to f64 first matches wolframscript.
#[test]
fn unit_convert_real_pounds_rounding() {
  assert_eq!(
    interpret("UnitConvert[Quantity[3.8, \"Pounds\"]]").unwrap(),
    "Quantity[1.723651006, Kilograms]"
  );
}

// ─── Temperature (affine) conversions ───────────────────────────────────────

#[test]
fn temperature_unit_names_canonicalize() {
  // Temperature aliases canonicalize to their Wolfram output spelling at
  // Quantity construction time.
  assert_eq!(
    interpret("Quantity[100, \"Celsius\"]").unwrap(),
    "Quantity[100, DegreesCelsius]"
  );
  assert_eq!(
    interpret("Quantity[32, \"Fahrenheit\"]").unwrap(),
    "Quantity[32, DegreesFahrenheit]"
  );
  assert_eq!(
    interpret("Quantity[100, \"Kelvin\"]").unwrap(),
    "Quantity[100, Kelvins]"
  );
  // Already-canonical names are unchanged.
  assert_eq!(
    interpret("Quantity[100, \"DegreesCelsius\"]").unwrap(),
    "Quantity[100, DegreesCelsius]"
  );
  // QuantityUnit reflects the canonical name.
  assert_eq!(
    interpret("QuantityUnit[Quantity[5, \"Celsius\"]]").unwrap(),
    "DegreesCelsius"
  );
}

#[test]
fn unit_convert_celsius_to_kelvin() {
  assert_eq!(
    interpret("UnitConvert[Quantity[100, \"Celsius\"], \"Kelvin\"]").unwrap(),
    "Quantity[7463/20, Kelvins]"
  );
  assert_eq!(
    interpret("UnitConvert[Quantity[0, \"Celsius\"], \"Kelvin\"]").unwrap(),
    "Quantity[5463/20, Kelvins]"
  );
}

#[test]
fn unit_convert_kelvin_to_celsius_real() {
  assert_eq!(
    interpret("UnitConvert[Quantity[273.15, \"Kelvin\"], \"Celsius\"]")
      .unwrap(),
    "Quantity[0., DegreesCelsius]"
  );
}

#[test]
fn unit_convert_fahrenheit_celsius() {
  assert_eq!(
    interpret("UnitConvert[Quantity[32, \"Fahrenheit\"], \"Celsius\"]")
      .unwrap(),
    "Quantity[0, DegreesCelsius]"
  );
  assert_eq!(
    interpret("UnitConvert[Quantity[212, \"Fahrenheit\"], \"Celsius\"]")
      .unwrap(),
    "Quantity[100, DegreesCelsius]"
  );
}

#[test]
fn unit_convert_celsius_to_fahrenheit() {
  assert_eq!(
    interpret("UnitConvert[Quantity[100, \"Celsius\"], \"Fahrenheit\"]")
      .unwrap(),
    "Quantity[212, DegreesFahrenheit]"
  );
  assert_eq!(
    interpret("UnitConvert[Quantity[37, \"Celsius\"], \"Fahrenheit\"]")
      .unwrap(),
    "Quantity[493/5, DegreesFahrenheit]"
  );
}

#[test]
fn unit_convert_fahrenheit_to_kelvin() {
  assert_eq!(
    interpret("UnitConvert[Quantity[100, \"Fahrenheit\"], \"Kelvin\"]")
      .unwrap(),
    "Quantity[55967/180, Kelvins]"
  );
  assert_eq!(
    interpret("UnitConvert[Quantity[0, \"Kelvin\"], \"Fahrenheit\"]").unwrap(),
    "Quantity[-45967/100, DegreesFahrenheit]"
  );
}

// The 1-argument form converts to the SI base, Kelvin.
#[test]
fn unit_convert_celsius_one_arg() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Celsius\"]]").unwrap(),
    "Quantity[5483/20, Kelvins]"
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
  // Wolfram doesn't recognize compound expressions with bare symbols as units.
  // Kilometers/Seconds^2 (unquoted) returns unevaluated.
  assert_eq!(
    interpret(
      "UnitConvert[Quantity[1, Kilometers/Seconds^2], \"Meters/Seconds^2\"]"
    )
    .unwrap(),
    "UnitConvert[Quantity[1, Kilometers/Seconds^2], Meters/Seconds^2]"
  );
}

#[test]
fn unit_convert_acceleration_from_velocity() {
  // Full acceleration calculation: 100 km/h in 3.2 seconds
  // Use compound expression "Meters"/"Seconds"^2 (not string "Meters/Seconds^2"
  // which wolframscript can't parse)
  assert_eq!(
    interpret(
      r#"v0 = Quantity[0, "km/h"]; v1 = Quantity[100, "km/h"]; t1 = Quantity[3.2, "Seconds"]; UnitConvert[(v1 - v0) / t1, "Meters"/"Seconds"^2]"#
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
  // Result should be in Kilometers/Hours
  assert!(result.contains("Kilometers"));
  assert!(result.contains("Hours"));
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
fn unit_convert_kwh_abbreviation_to_joules() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"kWh\"], \"Joules\"]").unwrap(),
    "Quantity[3600000, Joules]"
  );
}

#[test]
fn quantity_kwh_display() {
  assert_eq!(
    interpret("Quantity[1, \"KilowattHours\"]").unwrap(),
    "Quantity[1, Hours*Kilowatts]"
  );
  assert_eq!(
    interpret("Quantity[1, \"kWh\"]").unwrap(),
    "Quantity[1, Hours*Kilowatts]"
  );
  assert_eq!(
    interpret("Quantity[1, \"WattHours\"]").unwrap(),
    "Quantity[1, Hours*Watts]"
  );
  assert_eq!(
    interpret("Quantity[1, \"Wh\"]").unwrap(),
    "Quantity[1, Hours*Watts]"
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
  // Power[Quantity[4, Meters/Seconds^2], 1/2] → Quantity[2, Sqrt[Meters]/Seconds] (matching wolframscript)
  assert_eq!(
    interpret("Power[Quantity[4, \"Meters\"/\"Seconds\"^2], 1/2]").unwrap(),
    "Quantity[2, Sqrt[\"Meters\"]/Seconds]"
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

#[test]
fn quantity_real_exponent_no_parens() {
  // A Real exponent in a Quantity unit renders without parentheses,
  // matching wolframscript: Watts^0.24, not Watts^(0.24).
  assert_eq!(
    interpret(r#"Quantity[4., "watt"]^(.24)"#).unwrap(),
    "Quantity[1.3947436663504054, Watts^0.24]"
  );
}

#[test]
fn quantity_rational_exponent_keeps_parens() {
  // A Rational exponent prints as `n/d` so it needs parens to disambiguate
  // from `Watts^n / d`. Wolfram displays it as Watts^(1/3).
  assert_eq!(
    interpret(r#"Quantity[8, "Meters"^3]^(1/3)"#).unwrap(),
    "Quantity[2, Meters]"
  );
  // Confirm the parens form on a unit that doesn't simplify away.
  assert_eq!(
    interpret(r#"Quantity[4., "watt"]^(1/3)"#).unwrap(),
    "Quantity[1.5874010519681994, Watts^(1/3)]"
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
    "Quantity[4184, Joules]"
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
    "Quantity[1, DietaryCalories]"
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
  // String abbreviations are preserved as-is in display
  // (wolframscript under Quiet does not expand them)
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
    "Quantity[5, MetricTons]"
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
    "Quantity[100, ThermochemicalCalories]"
  );
}

#[test]
fn quantity_abbreviation_kcal() {
  assert_eq!(
    interpret("Quantity[2, \"kcal\"]").unwrap(),
    "Quantity[2, ThermochemicalKilocalories]"
  );
}

#[test]
fn quantity_abbreviation_ev() {
  assert_eq!(
    interpret("Quantity[1, \"eV\"]").unwrap(),
    "Quantity[1, Electronvolts]"
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
    "Quantity[30, MetricKilotons]"
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

// ─── Singular unit normalization ─────────────────────────────────────────────

#[test]
fn quantity_singular_meter() {
  assert_eq!(
    interpret("Quantity[5, \"Meter\"]").unwrap(),
    "Quantity[5, Meters]"
  );
}

#[test]
fn quantity_singular_second() {
  assert_eq!(
    interpret("Quantity[10, \"Second\"]").unwrap(),
    "Quantity[10, Seconds]"
  );
}

#[test]
fn quantity_singular_compound() {
  assert_eq!(
    interpret("Quantity[12.345, \"Meter\"/\"Second\"]").unwrap(),
    "Quantity[12.345, Meters/Seconds]"
  );
}

#[test]
fn quantity_singular_foot() {
  assert_eq!(
    interpret("Quantity[100, \"Foot\"]").unwrap(),
    "Quantity[100, Feet]"
  );
}

#[test]
fn quantity_singular_inch() {
  assert_eq!(
    interpret("Quantity[5, \"Inch\"]").unwrap(),
    "Quantity[5, Inches]"
  );
}

#[test]
fn quantity_singular_henry() {
  assert_eq!(
    interpret("Quantity[3, \"Henry\"]").unwrap(),
    "Quantity[3, Henries]"
  );
}

#[test]
fn quantity_singular_mole() {
  // "Mole" canonicalizes to the plural "Moles" (amount of substance).
  assert_eq!(
    interpret("Quantity[1, \"Mole\"]").unwrap(),
    "Quantity[1, Moles]"
  );
  assert_eq!(
    interpret("N[Quantity[1, \"Mole\"]]").unwrap(),
    "Quantity[1., Moles]"
  );
}

#[test]
fn quantity_mole_arithmetic_and_convert() {
  assert_eq!(
    interpret("Quantity[3, \"Mole\"] + Quantity[2, \"Mole\"]").unwrap(),
    "Quantity[5, Moles]"
  );
  assert_eq!(
    interpret("UnitConvert[Quantity[1000, \"Millimoles\"], \"Moles\"]")
      .unwrap(),
    "Quantity[1, Moles]"
  );
}

#[test]
fn quantity_mole_dimensions() {
  assert_eq!(
    interpret("UnitDimensions[Quantity[1, \"Moles\"]]").unwrap(),
    "{{AmountUnit, 1}}"
  );
}

// ─── SVG rendering (playground) ─────────────────────────────────────────────

#[test]
fn quantity_svg_simple_unit() {
  clear_state();
  let result = interpret_with_stdout("Quantity[5, \"Meters\"]").unwrap();
  assert_eq!(result.result, "5 m");
  let svg = result.output_svg.expect("Quantity should produce SVG");
  assert!(
    svg.contains(">5</text>") && svg.contains(">m</text>"),
    "SVG should show abbreviated unit: {svg}"
  );
}

#[test]
fn quantity_svg_compound_divide() {
  clear_state();
  let result =
    interpret_with_stdout("Quantity[12.345, \"Meters\"/\"Seconds\"]").unwrap();
  assert_eq!(result.result, "12.345 m/s");
  let svg = result.output_svg.expect("Quantity should produce SVG");
  assert!(
    svg.contains(">m</text>") && svg.contains(">s</text>"),
    "SVG should show abbreviated compound unit: {svg}"
  );
}

#[test]
fn quantity_svg_power_unit() {
  clear_state();
  let result =
    interpret_with_stdout("Quantity[9.8, \"Meters\"/\"Seconds\"^2]").unwrap();
  assert_eq!(result.result, "9.8 m/s^2");
  let svg = result.output_svg.expect("Quantity should produce SVG");
  assert!(
    svg.contains(">m</text>") && svg.contains(">s</text>"),
    "SVG should show abbreviated unit: {svg}"
  );
  assert!(
    svg.contains(">2</text>"),
    "SVG should contain exponent: {svg}"
  );
}

#[test]
fn quantity_svg_kilogram() {
  clear_state();
  let result = interpret_with_stdout("Quantity[70, \"Kilograms\"]").unwrap();
  assert_eq!(result.result, "70 kg");
  let svg = result.output_svg.expect("Quantity should produce SVG");
  assert!(svg.contains(">kg</text>"), "SVG should show 'kg': {svg}");
}

#[test]
fn quantity_svg_days_plural() {
  clear_state();
  let result = interpret_with_stdout("Quantity[21, \"Days\"]").unwrap();
  assert_eq!(result.result, "21 days");
  let svg = result.output_svg.expect("Quantity should produce SVG");
  assert!(
    svg.contains(">days</text>"),
    "SVG should show 'days': {svg}"
  );
}

#[test]
fn quantity_svg_days_singular() {
  clear_state();
  let result = interpret_with_stdout("Quantity[1, \"Days\"]").unwrap();
  assert_eq!(result.result, "1 day");
  let svg = result.output_svg.expect("Quantity should produce SVG");
  assert!(svg.contains(">day</text>"), "SVG should show 'day': {svg}");
}

#[test]
fn quantity_svg_singular_compound() {
  clear_state();
  let result =
    interpret_with_stdout("Quantity[12.345, \"Meter\"/\"Second\"]").unwrap();
  assert_eq!(result.result, "12.345 m/s");
}

// ─── Box representation for Quantity ─────────────────────────────────────────

#[test]
fn quantity_box_simple_unit() {
  assert_eq!(
    interpret("ToBoxes[Quantity[5, \"Meters\"]]").unwrap(),
    "RowBox[{5,  , m}]"
  );
}

#[test]
fn quantity_box_compound_divide() {
  // Wolfram emits a trailing backtick on machine-precision floats inside
  // box output (e.g. `12.345\``), matching MakeBoxes/ToBoxes precision marker.
  assert_eq!(
    interpret("ToBoxes[Quantity[12.345, \"Meters\"/\"Seconds\"]]").unwrap(),
    "RowBox[{12.345`,  , RowBox[{m, /, s}]}]"
  );
}

#[test]
fn quantity_box_power_unit() {
  let result =
    interpret("ToBoxes[Quantity[9.8, \"Meters\"/\"Seconds\"^2]]").unwrap();
  assert!(
    result.contains("SuperscriptBox"),
    "Power unit box should contain SuperscriptBox: {result}"
  );
  assert!(
    result.contains("FractionBox") || result.contains("/"),
    "Compound unit box should show division: {result}"
  );
}

#[test]
fn quantity_box_kilogram() {
  assert_eq!(
    interpret("ToBoxes[Quantity[70, \"Kilograms\"]]").unwrap(),
    "RowBox[{70,  , kg}]"
  );
}

#[test]
fn quantity_box_days_plural() {
  assert_eq!(
    interpret("ToBoxes[Quantity[21, \"Days\"]]").unwrap(),
    "RowBox[{21,  , days}]"
  );
}

#[test]
fn quantity_box_days_singular() {
  assert_eq!(
    interpret("ToBoxes[Quantity[1, \"Days\"]]").unwrap(),
    "RowBox[{1,  , day}]"
  );
}

// ─── UnitConvert with singular unit names ───────────────────────────────────

#[test]
fn unit_convert_singular_target_kilometer_hour() {
  assert_eq!(
    interpret(
      "UnitConvert[Quantity[12.345, \"Meter\"/\"Second\"], \"Kilometer\"/\"Hour\"]"
    )
    .unwrap(),
    "Quantity[44.442, Kilometers/Hours]"
  );
}

#[test]
fn unit_convert_singular_target_foot() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Meters\"], \"Foot\"]").unwrap(),
    "Quantity[1250/381, Feet]"
  );
}

#[test]
fn unit_convert_singular_target_inch() {
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Meters\"], \"Inch\"]").unwrap(),
    "Quantity[5000/127, Inches]"
  );
}

#[test]
fn unit_convert_all_target_formats_m_s_to_km_h() {
  // All of these should produce the same result
  let expected = "Quantity[44.442, Kilometers/Hours]";
  assert_eq!(
    interpret(
      "UnitConvert[Quantity[12.345, \"Meter\"/\"Second\"], \"Kilometers\"/\"Hours\"]"
    )
    .unwrap(),
    expected
  );
  assert_eq!(
    interpret(
      "UnitConvert[Quantity[12.345, \"Meter\"/\"Second\"], \"Kilometer\"/\"Hour\"]"
    )
    .unwrap(),
    expected
  );
  assert_eq!(
    interpret(
      "UnitConvert[Quantity[12.345, \"Meter\"/\"Second\"], \"km\"/\"h\"]"
    )
    .unwrap(),
    expected
  );
  assert_eq!(
    interpret("UnitConvert[Quantity[12.345, \"Meter\"/\"Second\"], \"km/h\"]")
      .unwrap(),
    expected
  );
  assert_eq!(
    interpret("UnitConvert[Quantity[12.345, \"Meter\"/\"Second\"], \"kph\"]")
      .unwrap(),
    expected
  );
}

#[test]
fn unit_convert_singular_target_playground_display() {
  clear_state();
  let result = interpret_with_stdout(
    "UnitConvert[Quantity[12.345, \"Meter\"/\"Second\"], \"Kilometer\"/\"Hour\"]",
  )
  .unwrap();
  assert_eq!(result.result, "44.442 km/h");
}

/// Regression tests for Gigapascals, Megapascals, Kilopascals unit support
#[test]
fn gigapascals_to_pascals() {
  clear_state();
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Gigapascals\"], \"Pascals\"]")
      .unwrap(),
    "Quantity[1000000000, Pascals]"
  );
}

#[test]
fn megapascals_to_pascals() {
  clear_state();
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Megapascals\"], \"Pascals\"]")
      .unwrap(),
    "Quantity[1000000, Pascals]"
  );
}

#[test]
fn kilopascals_to_pascals() {
  clear_state();
  assert_eq!(
    interpret("UnitConvert[Quantity[1, \"Kilopascals\"], \"Pascals\"]")
      .unwrap(),
    "Quantity[1000, Pascals]"
  );
}

#[test]
fn gigapascals_200_to_pascals() {
  clear_state();
  assert_eq!(
    interpret("UnitConvert[Quantity[200, \"Gigapascals\"], \"Pascals\"]")
      .unwrap(),
    "Quantity[200000000000, Pascals]"
  );
}

// ─── IndependentUnit ──────────────────────────────────────────────────────────

#[test]
fn independent_unit_basic() {
  assert_eq!(
    interpret("IndependentUnit[\"Foo\"]").unwrap(),
    "IndependentUnit[Foo]"
  );
}

#[test]
fn independent_unit_head() {
  assert_eq!(
    interpret("Head[IndependentUnit[\"Foo\"]]").unwrap(),
    "IndependentUnit"
  );
}

#[test]
fn independent_unit_with_quantity() {
  assert_eq!(
    interpret("Quantity[3, IndependentUnit[\"Foo\"]]").unwrap(),
    "Quantity[3, IndependentUnit[\"Foo\"]]"
  );
}

// ─── DatedUnit ──────────────────────────────────────────────────────────────

#[test]
fn dated_unit_basic() {
  assert_eq!(
    interpret(r#"DatedUnit["USDollars", 1990]"#).unwrap(),
    "DatedUnit[USDollars, 1990]"
  );
}

#[test]
fn dated_unit_symbolic() {
  assert_eq!(
    interpret(r#"DatedUnit["Euros", 2020]"#).unwrap(),
    "DatedUnit[Euros, 2020]"
  );
}

#[test]
fn dated_unit_with_quantity() {
  assert_eq!(
    interpret(r#"Quantity[100, DatedUnit["USDollars", 1990]]"#).unwrap(),
    r#"Quantity[100, DatedUnit["USDollars", 1990]]"#
  );
}

#[test]
fn dated_unit_single_arg() {
  assert_eq!(
    interpret(r#"DatedUnit["USDollars"]"#).unwrap(),
    "DatedUnit[USDollars]"
  );
}

// ─── QuantityDistribution ───────────────────────────────────────────────────

#[test]
fn quantity_distribution_inert() {
  assert_eq!(
    interpret(r#"QuantityDistribution[NormalDistribution[0, 1], "Meters"]"#)
      .unwrap(),
    "QuantityDistribution[NormalDistribution[0, 1], Meters]"
  );
}

#[test]
fn quantity_distribution_mean() {
  assert_eq!(
    interpret(
      r#"Mean[QuantityDistribution[NormalDistribution[0, 1], "Meters"]]"#
    )
    .unwrap(),
    "Quantity[0, Meters]"
  );
}

#[test]
fn quantity_distribution_mean_exponential() {
  assert_eq!(
    interpret(
      r#"Mean[QuantityDistribution[ExponentialDistribution[2], "Seconds"]]"#
    )
    .unwrap(),
    "Quantity[1/2, Seconds]"
  );
}

#[test]
fn quantity_distribution_variance() {
  assert_eq!(
    interpret(
      r#"Variance[QuantityDistribution[NormalDistribution[0, 1], "Meters"]]"#
    )
    .unwrap(),
    "Quantity[1, Meters^2]"
  );
}

#[test]
fn quantity_distribution_variance_exponential() {
  assert_eq!(
    interpret(
      r#"Variance[QuantityDistribution[ExponentialDistribution[2], "Seconds"]]"#
    )
    .unwrap(),
    "Quantity[1/4, Seconds^2]"
  );
}

#[test]
fn quantity_with_integer_unit_multiplies() {
  assert_eq!(interpret("Quantity[2, 3]").unwrap(), "6");
}

#[test]
fn quantity_with_rational_unit_multiplies() {
  assert_eq!(interpret("Quantity[2, 3/2]").unwrap(), "3");
}

#[test]
fn quantity_with_real_unit_multiplies() {
  assert_eq!(interpret("Quantity[2, 1.5]").unwrap(), "3.");
}

#[test]
fn quantity_with_unit_one_keeps_magnitude() {
  assert_eq!(interpret("Quantity[3/2, 1]").unwrap(), "3/2");
}

#[test]
fn quantity_unit_threads_over_list_from_quantity() {
  assert_eq!(
    interpret(r#"QuantityUnit[Quantity[{10,20}, "Meters"]]"#).unwrap(),
    "{Meters, Meters}"
  );
}

#[test]
fn quantity_unit_threads_over_list_of_quantities() {
  assert_eq!(
    interpret(r#"QuantityUnit[{Quantity[1, "m"], Quantity[2, "s"]}]"#).unwrap(),
    "{Meters, Seconds}"
  );
}

#[test]
fn unit_convert_one_arg_miles_to_meters() {
  assert_eq!(
    interpret(r#"UnitConvert[Quantity[5.2, "miles"]]"#).unwrap(),
    "Quantity[8368.588800000001, Meters]"
  );
}

#[test]
fn unit_convert_one_arg_hour_to_seconds() {
  assert_eq!(
    interpret(r#"UnitConvert[Quantity[1, "hour"]]"#).unwrap(),
    "Quantity[3600, Seconds]"
  );
}

#[test]
fn unit_convert_one_arg_centimeters_to_meters() {
  assert_eq!(
    interpret(r#"UnitConvert[Quantity[100, "centimeter"]]"#).unwrap(),
    "Quantity[1, Meters]"
  );
}

#[test]
fn unit_convert_one_arg_grams_to_kilograms() {
  assert_eq!(
    interpret(r#"UnitConvert[Quantity[1000, "gram"]]"#).unwrap(),
    "Quantity[1, Kilograms]"
  );
}

#[test]
fn known_unit_q_canonical_plural_true() {
  assert_eq!(interpret(r#"KnownUnitQ["Feet"]"#).unwrap(), "True");
  assert_eq!(interpret(r#"KnownUnitQ["Meters"]"#).unwrap(), "True");
  assert_eq!(interpret(r#"KnownUnitQ["Kilograms"]"#).unwrap(), "True");
}

#[test]
fn known_unit_q_singular_false() {
  // Wolframscript treats singular forms as unknown.
  assert_eq!(interpret(r#"KnownUnitQ["Foot"]"#).unwrap(), "False");
  assert_eq!(interpret(r#"KnownUnitQ["Meter"]"#).unwrap(), "False");
}

#[test]
fn known_unit_q_unknown_string_false() {
  assert_eq!(interpret(r#"KnownUnitQ["Foo"]"#).unwrap(), "False");
}

#[test]
fn known_unit_q_compound_true() {
  assert_eq!(
    interpret(r#"KnownUnitQ["Meters"/"Seconds"^2]"#).unwrap(),
    "True"
  );
  assert_eq!(
    interpret(r#"KnownUnitQ["Meters"*"Seconds"]"#).unwrap(),
    "True"
  );
}

#[test]
fn known_unit_q_compound_lowercase_false() {
  // Lowercase names don't count in Wolfram (mathics is more lenient).
  assert_eq!(
    interpret(r#"KnownUnitQ["meter"^2/"second"]"#).unwrap(),
    "False"
  );
}

#[test]
fn known_unit_q_dimensionless_one() {
  assert_eq!(interpret("KnownUnitQ[1]").unwrap(), "True");
}

#[test]
fn known_unit_q_other_numbers_false() {
  assert_eq!(interpret("KnownUnitQ[2]").unwrap(), "False");
  assert_eq!(interpret("KnownUnitQ[3.14]").unwrap(), "False");
  assert_eq!(interpret("KnownUnitQ[1/2]").unwrap(), "False");
}

mod cases {
  use super::super::case_helpers::assert_case;

  #[test]
  fn known_unit_q_1() {
    assert_case(r#"KnownUnitQ["Feet"]"#, r#"True"#);
  }
  #[test]
  fn known_unit_q_2() {
    assert_case(r#"KnownUnitQ["Feet"]; KnownUnitQ["Foo"]"#, r#"False"#);
  }
  #[test]
  fn known_unit_q_3() {
    assert_case(
      r#"KnownUnitQ["Feet"]; KnownUnitQ["Foo"]; KnownUnitQ["meter"^2/"second"]"#,
      r#"False"#,
    );
  }
  #[test]
  fn quantity_1() {
    assert_case(r#"Quantity["Kilogram"]"#, r#"Quantity[1, "Kilograms"]"#);
  }
  #[test]
  fn quantity_2() {
    assert_case(
      r#"Quantity["Kilogram"]; Quantity[10, "Meters"]"#,
      r#"Quantity[10, "Meters"]"#,
    );
  }
  #[test]
  fn quantity_3() {
    assert_case(
      r#"Quantity["Kilogram"]; Quantity[10, "Meters"]; Quantity[{10, 20}, "Meters"]"#,
      r#"{Quantity[10, "Meters"], Quantity[20, "Meters"]}"#,
    );
  }
  #[test]
  fn quantity_4() {
    assert_case(
      r#"Quantity["Kilogram"]; Quantity[10, "Meters"]; Quantity[{10, 20}, "Meters"]; Quantity[2, 3/2]"#,
      r#"3"#,
    );
  }
  #[test]
  fn quantity_q_1() {
    assert_case(
      r#"Quantity["Kilogram"]; Quantity[10, "Meters"]; Quantity[{10, 20}, "Meters"]; Quantity[2, 3/2]; QuantityQ[Quantity[2, Second]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn quantity_5() {
    assert_case(
      r#"Quantity["Kilogram"]; Quantity[10, "Meters"]; Quantity[{10, 20}, "Meters"]; Quantity[2, 3/2]; QuantityQ[Quantity[2, Second]]; Quantity[3, "centimeter"] / Quantity[2, "second"]^2"#,
      r#"Quantity[3/4, "Centimeters"/"Seconds"^2]"#,
    );
  }
  #[test]
  fn quantity_6() {
    assert_case(
      r#"Quantity["Kilogram"]; Quantity[10, "Meters"]; Quantity[{10, 20}, "Meters"]; Quantity[2, 3/2]; QuantityQ[Quantity[2, Second]]; Quantity[3, "centimeter"] / Quantity[2, "second"]^2; Quantity[3, "centimeter"] Quantity[2, "meter"]"#,
      r#"Quantity[3/50, "Meters"^2]"#,
    );
  }
  #[test]
  fn quantity_7() {
    assert_case(
      r#"Quantity["Kilogram"]; Quantity[10, "Meters"]; Quantity[{10, 20}, "Meters"]; Quantity[2, 3/2]; QuantityQ[Quantity[2, Second]]; Quantity[3, "centimeter"] / Quantity[2, "second"]^2; Quantity[3, "centimeter"] Quantity[2, "meter"]; Quantity[6, "meter"] + Quantity[3, "centimeter"]"#,
      r#"Quantity[603, "Centimeters"]"#,
    );
  }
  #[test]
  fn quantity_magnitude_1() {
    assert_case(r#"QuantityMagnitude[Quantity["Kilogram"]]"#, r#"1"#);
  }
  #[test]
  fn quantity_magnitude_2() {
    assert_case(
      r#"QuantityMagnitude[Quantity["Kilogram"]]; QuantityMagnitude[Quantity[10, "Meters"]]"#,
      r#"10"#,
    );
  }
  #[test]
  fn quantity_magnitude_3() {
    assert_case(
      r#"QuantityMagnitude[Quantity["Kilogram"]]; QuantityMagnitude[Quantity[10, "Meters"]]; QuantityMagnitude[Quantity[{10,20}, "Meters"]]"#,
      r#"{10, 20}"#,
    );
  }
  #[test]
  fn quantity_q_2() {
    assert_case(r#"QuantityQ[Quantity[3, "Meters"]]"#, r#"True"#);
  }
  #[test]
  fn quantity_unit_1() {
    assert_case(r#"QuantityUnit[Quantity["Kilogram"]]"#, r#""Kilograms""#);
  }
  #[test]
  fn quantity_unit_2() {
    assert_case(
      r#"QuantityUnit[Quantity["Kilogram"]]; QuantityUnit[Quantity[10, "Meters"]]"#,
      r#""Meters""#,
    );
  }
  #[test]
  fn quantity_unit_3() {
    assert_case(
      r#"QuantityUnit[Quantity["Kilogram"]]; QuantityUnit[Quantity[10, "Meters"]]; QuantityUnit[Quantity[{10,20}, "Meters"]]"#,
      r#"{"Meters", "Meters"}"#,
    );
  }
  #[test]
  fn unit_convert_1() {
    assert_case(
      r#"UnitConvert[Quantity[5.2, "miles"], "kilometers"]"#,
      r#"Quantity[8.368588800000001, "Kilometers"]"#,
    );
  }
  #[test]
  fn unit_convert_2() {
    assert_case(
      r#"UnitConvert[Quantity[5.2, "miles"], "kilometers"]; UnitConvert[Quantity[3.8, "Pounds"]]"#,
      r#"Quantity[1.723651006, "Kilograms"]"#,
    );
  }
  #[test]
  fn quantity_8() {
    assert_case(r#"Quantity[10, Meters]"#, r#"Quantity[10, Meters]"#);
  }
  #[test]
  fn quantity_9() {
    assert_case(
      r#"Quantity[4., "watt"]^(1/2)"#,
      r#"Quantity[2., Sqrt["Watts"]]"#,
    );
  }
  #[test]
  fn quantity_10() {
    assert_case(
      r#"Quantity[4., "watt"]^(1/3)"#,
      r#"Quantity[1.5874010519681994, "Watts"^(1/3)]"#,
    );
  }
  #[test]
  fn quantity_11() {
    assert_case(
      r#"Quantity[4., "watt"]^(.24)"#,
      r#"Quantity[1.3947436663504054, "Watts"^0.24]"#,
    );
  }
}

// ─── UnitDimensions ─────────────────────────────────────────────────────────

#[test]
fn unit_dimensions_base_units() {
  assert_eq!(
    interpret("UnitDimensions[\"Meters\"]").unwrap(),
    "{{LengthUnit, 1}}"
  );
  assert_eq!(
    interpret("UnitDimensions[\"Kilograms\"]").unwrap(),
    "{{MassUnit, 1}}"
  );
  assert_eq!(
    interpret("UnitDimensions[\"Seconds\"]").unwrap(),
    "{{TimeUnit, 1}}"
  );
  assert_eq!(
    interpret("UnitDimensions[\"Amperes\"]").unwrap(),
    "{{ElectricCurrentUnit, 1}}"
  );
}

// Derived units decompose, ordered alphabetically by dimension name.
#[test]
fn unit_dimensions_derived_units() {
  assert_eq!(
    interpret("UnitDimensions[\"Newtons\"]").unwrap(),
    "{{LengthUnit, 1}, {MassUnit, 1}, {TimeUnit, -2}}"
  );
  assert_eq!(
    interpret("UnitDimensions[\"Joules\"]").unwrap(),
    "{{LengthUnit, 2}, {MassUnit, 1}, {TimeUnit, -2}}"
  );
}

// Volume folds into LengthUnit cubed.
#[test]
fn unit_dimensions_volume_is_length_cubed() {
  assert_eq!(
    interpret("UnitDimensions[\"Liters\"]").unwrap(),
    "{{LengthUnit, 3}}"
  );
}

// Temperature scales are a separate dimension.
#[test]
fn unit_dimensions_temperature() {
  assert_eq!(
    interpret("UnitDimensions[\"Kelvins\"]").unwrap(),
    "{{TemperatureUnit, 1}}"
  );
  assert_eq!(
    interpret("UnitDimensions[\"Celsius\"]").unwrap(),
    "{{TemperatureUnit, 1}}"
  );
}

#[test]
fn unit_dimensions_compound_and_quantity() {
  assert_eq!(
    interpret("UnitDimensions[\"Meters\"/\"Seconds\"]").unwrap(),
    "{{LengthUnit, 1}, {TimeUnit, -1}}"
  );
  assert_eq!(
    interpret("UnitDimensions[Quantity[1, \"Newtons\"]]").unwrap(),
    "{{LengthUnit, 1}, {MassUnit, 1}, {TimeUnit, -2}}"
  );
}
