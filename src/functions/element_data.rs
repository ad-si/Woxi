use crate::InterpreterError;
use crate::syntax::Expr;

/// Data for a single element.
struct Element {
  atomic_number: i128,
  standard_name: &'static str,
  abbreviation: &'static str,
  atomic_weight: &'static str,
  group: Option<i128>,
  period: i128,
  block: &'static str,
  electronegativity: Option<f64>,
  melting_point: Option<f64>,
  boiling_point: Option<f64>,
  atomic_radius: Option<f64>,
  electron_configuration: &'static [&'static [u8]],
}

/// Atomic weight precisions from Wolfram Language for Z=1..118
static ATOMIC_WEIGHT_PRECISIONS: &[f64] = &[
  5.0, 7.0, 3.0, 8.0, 4.0, 5.0, 5.0, 5.0, 11.0, 6.0, // 1-10
  10.0, 5.0, 9.0, 5.0, 11.0, 4.0, 4.0, 4.0, 6.0, 5.0, // 11-20
  8.0, 5.0, 6.0, 6.0, 8.0, 5.0, 8.0, 6.0, 5.0, 4.0, // 21-30
  5.0, 5.0, 8.0, 5.0, 5.0, 5.0, 6.0, 4.0, 8.0, 5.0, // 31-40
  7.0, 4.0, 2.0, 5.0, 8.0, 5.0, 7.0, 6.0, 6.0, 6.0, // 41-50
  6.0, 5.0, 8.0, 6.0, 11.0, 6.0, 8.0, 6.0, 8.0, 6.0, // 51-60
  3.0, 5.0, 6.0, 5.0, 9.0, 6.0, 9.0, 6.0, 9.0, 6.0, // 61-70
  7.0, 6.0, 8.0, 5.0, 6.0, 5.0, 6.0, 6.0, 9.0, 6.0, // 71-80
  5.0, 4.0, 8.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 7.0, // 81-90
  8.0, 8.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, // 91-100
  3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, // 101-110
  3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, // 111-118
];

fn make_quantity(magnitude: Expr, unit: &str) -> Expr {
  Expr::FunctionCall {
    name: "Quantity".to_string(),
    args: vec![magnitude, Expr::String(unit.to_string())],
  }
}

fn make_bigfloat(value: f64, precision: f64) -> Expr {
  Expr::BigFloat(
    format_real_for_precision(value, precision as usize),
    precision,
  )
}

/// Format a real number, trimming trailing zeros but keeping decimal point
fn format_real_for_precision(value: f64, _sig_digits: usize) -> String {
  if value == 0.0 {
    return "0.".to_string();
  }
  // Use enough decimal places to capture all digits, then trim trailing zeros
  let sign = if value < 0.0 { "-" } else { "" };
  let abs_val = value.abs();
  let s = format!("{:.15}", abs_val);
  let s = s.trim_end_matches('0');

  if s.ends_with('.') {
    format!("{}{}", sign, s)
  } else {
    format!("{}{}", sign, s)
  }
}

fn missing_not_available() -> Expr {
  Expr::FunctionCall {
    name: "Missing".to_string(),
    args: vec![Expr::String("NotAvailable".to_string())],
  }
}

fn missing_not_applicable() -> Expr {
  Expr::FunctionCall {
    name: "Missing".to_string(),
    args: vec![Expr::String("NotApplicable".to_string())],
  }
}

fn missing_not_found() -> Expr {
  Expr::FunctionCall {
    name: "Missing".to_string(),
    args: vec![Expr::String("NotFound".to_string())],
  }
}

static ELEMENTS: &[Element] = &[
  Element {
    atomic_number: 1,
    standard_name: "Hydrogen",
    abbreviation: "H",
    atomic_weight: "1.008",
    group: Some(1),
    period: 1,
    block: "s",
    electronegativity: Some(2.2),
    melting_point: Some(-259.14),
    boiling_point: Some(-252.87),
    atomic_radius: Some(53.0),
    electron_configuration: &[&[1]],
  },
  Element {
    atomic_number: 2,
    standard_name: "Helium",
    abbreviation: "He",
    atomic_weight: "4.002602",
    group: Some(18),
    period: 1,
    block: "s",
    electronegativity: None,
    melting_point: None,
    boiling_point: Some(-268.93),
    atomic_radius: Some(31.0),
    electron_configuration: &[&[2]],
  },
  Element {
    atomic_number: 3,
    standard_name: "Lithium",
    abbreviation: "Li",
    atomic_weight: "6.94",
    group: Some(1),
    period: 2,
    block: "s",
    electronegativity: Some(0.98),
    melting_point: Some(180.54),
    boiling_point: Some(1342.0),
    atomic_radius: Some(167.0),
    electron_configuration: &[&[2], &[1]],
  },
  Element {
    atomic_number: 4,
    standard_name: "Beryllium",
    abbreviation: "Be",
    atomic_weight: "9.0121831",
    group: Some(2),
    period: 2,
    block: "s",
    electronegativity: Some(1.57),
    melting_point: Some(1287.0),
    boiling_point: Some(2470.0),
    atomic_radius: Some(112.0),
    electron_configuration: &[&[2], &[2]],
  },
  Element {
    atomic_number: 5,
    standard_name: "Boron",
    abbreviation: "B",
    atomic_weight: "10.81",
    group: Some(13),
    period: 2,
    block: "p",
    electronegativity: Some(2.04),
    melting_point: Some(2075.0),
    boiling_point: Some(4000.0),
    atomic_radius: Some(87.0),
    electron_configuration: &[&[2], &[2, 1]],
  },
  Element {
    atomic_number: 6,
    standard_name: "Carbon",
    abbreviation: "C",
    atomic_weight: "12.011",
    group: Some(14),
    period: 2,
    block: "p",
    electronegativity: Some(2.55),
    melting_point: Some(3550.0),
    boiling_point: Some(4027.0),
    atomic_radius: Some(67.0),
    electron_configuration: &[&[2], &[2, 2]],
  },
  Element {
    atomic_number: 7,
    standard_name: "Nitrogen",
    abbreviation: "N",
    atomic_weight: "14.007",
    group: Some(15),
    period: 2,
    block: "p",
    electronegativity: Some(3.04),
    melting_point: Some(-210.1),
    boiling_point: Some(-195.79),
    atomic_radius: Some(56.0),
    electron_configuration: &[&[2], &[2, 3]],
  },
  Element {
    atomic_number: 8,
    standard_name: "Oxygen",
    abbreviation: "O",
    atomic_weight: "15.999",
    group: Some(16),
    period: 2,
    block: "p",
    electronegativity: Some(3.44),
    melting_point: Some(-218.3),
    boiling_point: Some(-182.9),
    atomic_radius: Some(48.0),
    electron_configuration: &[&[2], &[2, 4]],
  },
  Element {
    atomic_number: 9,
    standard_name: "Fluorine",
    abbreviation: "F",
    atomic_weight: "18.998403162",
    group: Some(17),
    period: 2,
    block: "p",
    electronegativity: Some(3.98),
    melting_point: Some(-219.6),
    boiling_point: Some(-188.12),
    atomic_radius: Some(42.0),
    electron_configuration: &[&[2], &[2, 5]],
  },
  Element {
    atomic_number: 10,
    standard_name: "Neon",
    abbreviation: "Ne",
    atomic_weight: "20.1797",
    group: Some(18),
    period: 2,
    block: "p",
    electronegativity: None,
    melting_point: Some(-248.59),
    boiling_point: Some(-246.08),
    atomic_radius: Some(38.0),
    electron_configuration: &[&[2], &[2, 6]],
  },
  Element {
    atomic_number: 11,
    standard_name: "Sodium",
    abbreviation: "Na",
    atomic_weight: "22.98976928",
    group: Some(1),
    period: 3,
    block: "s",
    electronegativity: Some(0.93),
    melting_point: Some(97.72),
    boiling_point: Some(883.0),
    atomic_radius: Some(190.0),
    electron_configuration: &[&[2], &[2, 6], &[1]],
  },
  Element {
    atomic_number: 12,
    standard_name: "Magnesium",
    abbreviation: "Mg",
    atomic_weight: "24.305",
    group: Some(2),
    period: 3,
    block: "s",
    electronegativity: Some(1.31),
    melting_point: Some(650.0),
    boiling_point: Some(1090.0),
    atomic_radius: Some(145.0),
    electron_configuration: &[&[2], &[2, 6], &[2]],
  },
  Element {
    atomic_number: 13,
    standard_name: "Aluminum",
    abbreviation: "Al",
    atomic_weight: "26.9815384",
    group: Some(13),
    period: 3,
    block: "p",
    electronegativity: Some(1.61),
    melting_point: Some(660.32),
    boiling_point: Some(2519.0),
    atomic_radius: Some(118.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 1]],
  },
  Element {
    atomic_number: 14,
    standard_name: "Silicon",
    abbreviation: "Si",
    atomic_weight: "28.085",
    group: Some(14),
    period: 3,
    block: "p",
    electronegativity: Some(1.9),
    melting_point: Some(1414.0),
    boiling_point: Some(2900.0),
    atomic_radius: Some(111.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 2]],
  },
  Element {
    atomic_number: 15,
    standard_name: "Phosphorus",
    abbreviation: "P",
    atomic_weight: "30.973761998",
    group: Some(15),
    period: 3,
    block: "p",
    electronegativity: Some(2.19),
    melting_point: Some(44.2),
    boiling_point: Some(280.5),
    atomic_radius: Some(98.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 3]],
  },
  Element {
    atomic_number: 16,
    standard_name: "Sulfur",
    abbreviation: "S",
    atomic_weight: "32.06",
    group: Some(16),
    period: 3,
    block: "p",
    electronegativity: Some(2.58),
    melting_point: Some(115.21),
    boiling_point: Some(444.72),
    atomic_radius: Some(88.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 4]],
  },
  Element {
    atomic_number: 17,
    standard_name: "Chlorine",
    abbreviation: "Cl",
    atomic_weight: "35.45",
    group: Some(17),
    period: 3,
    block: "p",
    electronegativity: Some(3.16),
    melting_point: Some(-101.5),
    boiling_point: Some(-34.04),
    atomic_radius: Some(79.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 5]],
  },
  Element {
    atomic_number: 18,
    standard_name: "Argon",
    abbreviation: "Ar",
    atomic_weight: "39.95",
    group: Some(18),
    period: 3,
    block: "p",
    electronegativity: None,
    melting_point: Some(-189.3),
    boiling_point: Some(-185.8),
    atomic_radius: Some(71.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6]],
  },
  Element {
    atomic_number: 19,
    standard_name: "Potassium",
    abbreviation: "K",
    atomic_weight: "39.0983",
    group: Some(1),
    period: 4,
    block: "s",
    electronegativity: Some(0.82),
    melting_point: Some(63.38),
    boiling_point: Some(759.0),
    atomic_radius: Some(243.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6], &[1]],
  },
  Element {
    atomic_number: 20,
    standard_name: "Calcium",
    abbreviation: "Ca",
    atomic_weight: "40.078",
    group: Some(2),
    period: 4,
    block: "s",
    electronegativity: Some(1.0),
    melting_point: Some(842.0),
    boiling_point: Some(1484.0),
    atomic_radius: Some(194.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6], &[2]],
  },
  Element {
    atomic_number: 21,
    standard_name: "Scandium",
    abbreviation: "Sc",
    atomic_weight: "44.955907",
    group: Some(3),
    period: 4,
    block: "d",
    electronegativity: Some(1.36),
    melting_point: Some(1541.0),
    boiling_point: Some(2830.0),
    atomic_radius: Some(184.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 1], &[2]],
  },
  Element {
    atomic_number: 22,
    standard_name: "Titanium",
    abbreviation: "Ti",
    atomic_weight: "47.867",
    group: Some(4),
    period: 4,
    block: "d",
    electronegativity: Some(1.54),
    melting_point: Some(1668.0),
    boiling_point: Some(3287.0),
    atomic_radius: Some(176.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 2], &[2]],
  },
  Element {
    atomic_number: 23,
    standard_name: "Vanadium",
    abbreviation: "V",
    atomic_weight: "50.9415",
    group: Some(5),
    period: 4,
    block: "d",
    electronegativity: Some(1.63),
    melting_point: Some(1910.0),
    boiling_point: Some(3407.0),
    atomic_radius: Some(171.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 3], &[2]],
  },
  Element {
    atomic_number: 24,
    standard_name: "Chromium",
    abbreviation: "Cr",
    atomic_weight: "51.9961",
    group: Some(6),
    period: 4,
    block: "d",
    electronegativity: Some(1.66),
    melting_point: Some(1907.0),
    boiling_point: Some(2671.0),
    atomic_radius: Some(166.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 5], &[1]],
  },
  Element {
    atomic_number: 25,
    standard_name: "Manganese",
    abbreviation: "Mn",
    atomic_weight: "54.938043",
    group: Some(7),
    period: 4,
    block: "d",
    electronegativity: Some(1.55),
    melting_point: Some(1246.0),
    boiling_point: Some(2061.0),
    atomic_radius: Some(161.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 5], &[2]],
  },
  Element {
    atomic_number: 26,
    standard_name: "Iron",
    abbreviation: "Fe",
    atomic_weight: "55.845",
    group: Some(8),
    period: 4,
    block: "d",
    electronegativity: Some(1.83),
    melting_point: Some(1538.0),
    boiling_point: Some(2861.0),
    atomic_radius: Some(156.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 6], &[2]],
  },
  Element {
    atomic_number: 27,
    standard_name: "Cobalt",
    abbreviation: "Co",
    atomic_weight: "58.933194",
    group: Some(9),
    period: 4,
    block: "d",
    electronegativity: Some(1.88),
    melting_point: Some(1495.0),
    boiling_point: Some(2927.0),
    atomic_radius: Some(152.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 7], &[2]],
  },
  Element {
    atomic_number: 28,
    standard_name: "Nickel",
    abbreviation: "Ni",
    atomic_weight: "58.6934",
    group: Some(10),
    period: 4,
    block: "d",
    electronegativity: Some(1.91),
    melting_point: Some(1455.0),
    boiling_point: Some(2913.0),
    atomic_radius: Some(149.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 8], &[2]],
  },
  Element {
    atomic_number: 29,
    standard_name: "Copper",
    abbreviation: "Cu",
    atomic_weight: "63.546",
    group: Some(11),
    period: 4,
    block: "d",
    electronegativity: Some(1.9),
    melting_point: Some(1084.62),
    boiling_point: Some(2562.0),
    atomic_radius: Some(145.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 10], &[1]],
  },
  Element {
    atomic_number: 30,
    standard_name: "Zinc",
    abbreviation: "Zn",
    atomic_weight: "65.38",
    group: Some(12),
    period: 4,
    block: "d",
    electronegativity: Some(1.65),
    melting_point: Some(419.53),
    boiling_point: Some(907.0),
    atomic_radius: Some(142.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 10], &[2]],
  },
  Element {
    atomic_number: 31,
    standard_name: "Gallium",
    abbreviation: "Ga",
    atomic_weight: "69.723",
    group: Some(13),
    period: 4,
    block: "p",
    electronegativity: Some(1.81),
    melting_point: Some(29.76),
    boiling_point: Some(2204.0),
    atomic_radius: Some(136.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 10], &[2, 1]],
  },
  Element {
    atomic_number: 32,
    standard_name: "Germanium",
    abbreviation: "Ge",
    atomic_weight: "72.63",
    group: Some(14),
    period: 4,
    block: "p",
    electronegativity: Some(2.01),
    melting_point: Some(938.3),
    boiling_point: Some(2820.0),
    atomic_radius: Some(125.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 10], &[2, 2]],
  },
  Element {
    atomic_number: 33,
    standard_name: "Arsenic",
    abbreviation: "As",
    atomic_weight: "74.921595",
    group: Some(15),
    period: 4,
    block: "p",
    electronegativity: Some(2.18),
    melting_point: Some(817.0),
    boiling_point: Some(614.0),
    atomic_radius: Some(114.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 10], &[2, 3]],
  },
  Element {
    atomic_number: 34,
    standard_name: "Selenium",
    abbreviation: "Se",
    atomic_weight: "78.971",
    group: Some(16),
    period: 4,
    block: "p",
    electronegativity: Some(2.55),
    melting_point: Some(221.0),
    boiling_point: Some(685.0),
    atomic_radius: Some(103.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 10], &[2, 4]],
  },
  Element {
    atomic_number: 35,
    standard_name: "Bromine",
    abbreviation: "Br",
    atomic_weight: "79.904",
    group: Some(17),
    period: 4,
    block: "p",
    electronegativity: Some(2.96),
    melting_point: Some(-7.3),
    boiling_point: Some(59.0),
    atomic_radius: Some(94.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 10], &[2, 5]],
  },
  Element {
    atomic_number: 36,
    standard_name: "Krypton",
    abbreviation: "Kr",
    atomic_weight: "83.798",
    group: Some(18),
    period: 4,
    block: "p",
    electronegativity: Some(3.0),
    melting_point: Some(-157.36),
    boiling_point: Some(-153.22),
    atomic_radius: Some(88.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 10], &[2, 6]],
  },
  Element {
    atomic_number: 37,
    standard_name: "Rubidium",
    abbreviation: "Rb",
    atomic_weight: "85.4678",
    group: Some(1),
    period: 5,
    block: "s",
    electronegativity: Some(0.82),
    melting_point: Some(39.31),
    boiling_point: Some(688.0),
    atomic_radius: Some(265.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 10], &[2, 6], &[1]],
  },
  Element {
    atomic_number: 38,
    standard_name: "Strontium",
    abbreviation: "Sr",
    atomic_weight: "87.62",
    group: Some(2),
    period: 5,
    block: "s",
    electronegativity: Some(0.95),
    melting_point: Some(777.0),
    boiling_point: Some(1382.0),
    atomic_radius: Some(219.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 10], &[2, 6], &[2]],
  },
  Element {
    atomic_number: 39,
    standard_name: "Yttrium",
    abbreviation: "Y",
    atomic_weight: "88.905838",
    group: Some(3),
    period: 5,
    block: "d",
    electronegativity: Some(1.22),
    melting_point: Some(1526.0),
    boiling_point: Some(3345.0),
    atomic_radius: Some(212.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 10], &[2, 6, 1], &[2]],
  },
  Element {
    atomic_number: 40,
    standard_name: "Zirconium",
    abbreviation: "Zr",
    atomic_weight: "91.224",
    group: Some(4),
    period: 5,
    block: "d",
    electronegativity: Some(1.33),
    melting_point: Some(1855.0),
    boiling_point: Some(4409.0),
    atomic_radius: Some(206.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 10], &[2, 6, 2], &[2]],
  },
  Element {
    atomic_number: 41,
    standard_name: "Niobium",
    abbreviation: "Nb",
    atomic_weight: "92.90637",
    group: Some(5),
    period: 5,
    block: "d",
    electronegativity: Some(1.6),
    melting_point: Some(2477.0),
    boiling_point: Some(4744.0),
    atomic_radius: Some(198.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 10], &[2, 6, 4], &[1]],
  },
  Element {
    atomic_number: 42,
    standard_name: "Molybdenum",
    abbreviation: "Mo",
    atomic_weight: "95.95",
    group: Some(6),
    period: 5,
    block: "d",
    electronegativity: Some(2.16),
    melting_point: Some(2623.0),
    boiling_point: Some(4639.0),
    atomic_radius: Some(190.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 10], &[2, 6, 5], &[1]],
  },
  Element {
    atomic_number: 43,
    standard_name: "Technetium",
    abbreviation: "Tc",
    atomic_weight: "97.",
    group: Some(7),
    period: 5,
    block: "d",
    electronegativity: Some(1.9),
    melting_point: Some(2157.0),
    boiling_point: Some(4265.0),
    atomic_radius: Some(183.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 10], &[2, 6, 5], &[2]],
  },
  Element {
    atomic_number: 44,
    standard_name: "Ruthenium",
    abbreviation: "Ru",
    atomic_weight: "101.07",
    group: Some(8),
    period: 5,
    block: "d",
    electronegativity: Some(2.2),
    melting_point: Some(2334.0),
    boiling_point: Some(4150.0),
    atomic_radius: Some(178.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 10], &[2, 6, 7], &[1]],
  },
  Element {
    atomic_number: 45,
    standard_name: "Rhodium",
    abbreviation: "Rh",
    atomic_weight: "102.90549",
    group: Some(9),
    period: 5,
    block: "d",
    electronegativity: Some(2.28),
    melting_point: Some(1964.0),
    boiling_point: Some(3695.0),
    atomic_radius: Some(173.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 10], &[2, 6, 8], &[1]],
  },
  Element {
    atomic_number: 46,
    standard_name: "Palladium",
    abbreviation: "Pd",
    atomic_weight: "106.42",
    group: Some(10),
    period: 5,
    block: "d",
    electronegativity: Some(2.2),
    melting_point: Some(1554.9),
    boiling_point: Some(2963.0),
    atomic_radius: Some(169.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 10], &[2, 6, 10]],
  },
  Element {
    atomic_number: 47,
    standard_name: "Silver",
    abbreviation: "Ag",
    atomic_weight: "107.8682",
    group: Some(11),
    period: 5,
    block: "d",
    electronegativity: Some(1.93),
    melting_point: Some(961.78),
    boiling_point: Some(2162.0),
    atomic_radius: Some(165.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 10], &[2, 6, 10], &[1]],
  },
  Element {
    atomic_number: 48,
    standard_name: "Cadmium",
    abbreviation: "Cd",
    atomic_weight: "112.414",
    group: Some(12),
    period: 5,
    block: "d",
    electronegativity: Some(1.69),
    melting_point: Some(321.07),
    boiling_point: Some(767.0),
    atomic_radius: Some(161.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 10], &[2, 6, 10], &[2]],
  },
  Element {
    atomic_number: 49,
    standard_name: "Indium",
    abbreviation: "In",
    atomic_weight: "114.818",
    group: Some(13),
    period: 5,
    block: "p",
    electronegativity: Some(1.78),
    melting_point: Some(156.6),
    boiling_point: Some(2072.0),
    atomic_radius: Some(156.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 10], &[2, 6, 10], &[2, 1]],
  },
  Element {
    atomic_number: 50,
    standard_name: "Tin",
    abbreviation: "Sn",
    atomic_weight: "118.71",
    group: Some(14),
    period: 5,
    block: "p",
    electronegativity: Some(1.96),
    melting_point: Some(231.93),
    boiling_point: Some(2602.0),
    atomic_radius: Some(145.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 10], &[2, 6, 10], &[2, 2]],
  },
  Element {
    atomic_number: 51,
    standard_name: "Antimony",
    abbreviation: "Sb",
    atomic_weight: "121.76",
    group: Some(15),
    period: 5,
    block: "p",
    electronegativity: Some(2.05),
    melting_point: Some(630.63),
    boiling_point: Some(1587.0),
    atomic_radius: Some(133.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 10], &[2, 6, 10], &[2, 3]],
  },
  Element {
    atomic_number: 52,
    standard_name: "Tellurium",
    abbreviation: "Te",
    atomic_weight: "127.6",
    group: Some(16),
    period: 5,
    block: "p",
    electronegativity: Some(2.1),
    melting_point: Some(449.51),
    boiling_point: Some(988.0),
    atomic_radius: Some(123.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 10], &[2, 6, 10], &[2, 4]],
  },
  Element {
    atomic_number: 53,
    standard_name: "Iodine",
    abbreviation: "I",
    atomic_weight: "126.90447",
    group: Some(17),
    period: 5,
    block: "p",
    electronegativity: Some(2.66),
    melting_point: Some(113.7),
    boiling_point: Some(184.3),
    atomic_radius: Some(115.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 10], &[2, 6, 10], &[2, 5]],
  },
  Element {
    atomic_number: 54,
    standard_name: "Xenon",
    abbreviation: "Xe",
    atomic_weight: "131.293",
    group: Some(18),
    period: 5,
    block: "p",
    electronegativity: Some(2.6),
    melting_point: Some(-111.8),
    boiling_point: Some(-108.0),
    atomic_radius: Some(108.0),
    electron_configuration: &[&[2], &[2, 6], &[2, 6, 10], &[2, 6, 10], &[2, 6]],
  },
  Element {
    atomic_number: 55,
    standard_name: "Cesium",
    abbreviation: "Cs",
    atomic_weight: "132.90545196",
    group: Some(1),
    period: 6,
    block: "s",
    electronegativity: Some(0.79),
    melting_point: Some(28.44),
    boiling_point: Some(671.0),
    atomic_radius: Some(298.0),
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10],
      &[2, 6],
      &[1],
    ],
  },
  Element {
    atomic_number: 56,
    standard_name: "Barium",
    abbreviation: "Ba",
    atomic_weight: "137.327",
    group: Some(2),
    period: 6,
    block: "s",
    electronegativity: Some(0.89),
    melting_point: Some(727.0),
    boiling_point: Some(1870.0),
    atomic_radius: Some(253.0),
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10],
      &[2, 6],
      &[2],
    ],
  },
  Element {
    atomic_number: 57,
    standard_name: "Lanthanum",
    abbreviation: "La",
    atomic_weight: "138.90547",
    group: None,
    period: 6,
    block: "f",
    electronegativity: Some(1.1),
    melting_point: Some(920.0),
    boiling_point: Some(3464.0),
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10],
      &[2, 6, 1],
      &[2],
    ],
  },
  Element {
    atomic_number: 58,
    standard_name: "Cerium",
    abbreviation: "Ce",
    atomic_weight: "140.116",
    group: None,
    period: 6,
    block: "f",
    electronegativity: Some(1.12),
    melting_point: Some(798.0),
    boiling_point: Some(3360.0),
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 1],
      &[2, 6],
      &[2],
    ],
  },
  Element {
    atomic_number: 59,
    standard_name: "Praseodymium",
    abbreviation: "Pr",
    atomic_weight: "140.90766",
    group: None,
    period: 6,
    block: "f",
    electronegativity: Some(1.13),
    melting_point: Some(931.0),
    boiling_point: Some(3290.0),
    atomic_radius: Some(247.0),
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 3],
      &[2, 6],
      &[2],
    ],
  },
  Element {
    atomic_number: 60,
    standard_name: "Neodymium",
    abbreviation: "Nd",
    atomic_weight: "144.242",
    group: None,
    period: 6,
    block: "f",
    electronegativity: Some(1.14),
    melting_point: Some(1021.0),
    boiling_point: Some(3100.0),
    atomic_radius: Some(206.0),
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 4],
      &[2, 6],
      &[2],
    ],
  },
  Element {
    atomic_number: 61,
    standard_name: "Promethium",
    abbreviation: "Pm",
    atomic_weight: "145.",
    group: None,
    period: 6,
    block: "f",
    electronegativity: None,
    melting_point: Some(1100.0),
    boiling_point: Some(3000.0),
    atomic_radius: Some(205.0),
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 5],
      &[2, 6],
      &[2],
    ],
  },
  Element {
    atomic_number: 62,
    standard_name: "Samarium",
    abbreviation: "Sm",
    atomic_weight: "150.36",
    group: None,
    period: 6,
    block: "f",
    electronegativity: Some(1.17),
    melting_point: Some(1072.0),
    boiling_point: Some(1803.0),
    atomic_radius: Some(238.0),
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 6],
      &[2, 6],
      &[2],
    ],
  },
  Element {
    atomic_number: 63,
    standard_name: "Europium",
    abbreviation: "Eu",
    atomic_weight: "151.964",
    group: None,
    period: 6,
    block: "f",
    electronegativity: None,
    melting_point: Some(822.0),
    boiling_point: Some(1527.0),
    atomic_radius: Some(231.0),
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 7],
      &[2, 6],
      &[2],
    ],
  },
  Element {
    atomic_number: 64,
    standard_name: "Gadolinium",
    abbreviation: "Gd",
    atomic_weight: "157.25",
    group: None,
    period: 6,
    block: "f",
    electronegativity: Some(1.2),
    melting_point: Some(1313.0),
    boiling_point: Some(3250.0),
    atomic_radius: Some(233.0),
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 7],
      &[2, 6, 1],
      &[2],
    ],
  },
  Element {
    atomic_number: 65,
    standard_name: "Terbium",
    abbreviation: "Tb",
    atomic_weight: "158.925354",
    group: None,
    period: 6,
    block: "f",
    electronegativity: None,
    melting_point: Some(1356.0),
    boiling_point: Some(3230.0),
    atomic_radius: Some(225.0),
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 9],
      &[2, 6],
      &[2],
    ],
  },
  Element {
    atomic_number: 66,
    standard_name: "Dysprosium",
    abbreviation: "Dy",
    atomic_weight: "162.5",
    group: None,
    period: 6,
    block: "f",
    electronegativity: Some(1.22),
    melting_point: Some(1412.0),
    boiling_point: Some(2567.0),
    atomic_radius: Some(228.0),
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 10],
      &[2, 6],
      &[2],
    ],
  },
  Element {
    atomic_number: 67,
    standard_name: "Holmium",
    abbreviation: "Ho",
    atomic_weight: "164.930329",
    group: None,
    period: 6,
    block: "f",
    electronegativity: Some(1.23),
    melting_point: Some(1474.0),
    boiling_point: Some(2700.0),
    atomic_radius: Some(226.0),
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 11],
      &[2, 6],
      &[2],
    ],
  },
  Element {
    atomic_number: 68,
    standard_name: "Erbium",
    abbreviation: "Er",
    atomic_weight: "167.259",
    group: None,
    period: 6,
    block: "f",
    electronegativity: Some(1.24),
    melting_point: Some(1497.0),
    boiling_point: Some(2868.0),
    atomic_radius: Some(226.0),
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 12],
      &[2, 6],
      &[2],
    ],
  },
  Element {
    atomic_number: 69,
    standard_name: "Thulium",
    abbreviation: "Tm",
    atomic_weight: "168.934219",
    group: None,
    period: 6,
    block: "f",
    electronegativity: Some(1.25),
    melting_point: Some(1545.0),
    boiling_point: Some(1950.0),
    atomic_radius: Some(222.0),
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 13],
      &[2, 6],
      &[2],
    ],
  },
  Element {
    atomic_number: 70,
    standard_name: "Ytterbium",
    abbreviation: "Yb",
    atomic_weight: "173.045",
    group: None,
    period: 6,
    block: "f",
    electronegativity: None,
    melting_point: Some(819.0),
    boiling_point: Some(1196.0),
    atomic_radius: Some(222.0),
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6],
      &[2],
    ],
  },
  Element {
    atomic_number: 71,
    standard_name: "Lutetium",
    abbreviation: "Lu",
    atomic_weight: "174.9668",
    group: Some(3),
    period: 6,
    block: "d",
    electronegativity: Some(1.27),
    melting_point: Some(1663.0),
    boiling_point: Some(3402.0),
    atomic_radius: Some(217.0),
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 1],
      &[2],
    ],
  },
  Element {
    atomic_number: 72,
    standard_name: "Hafnium",
    abbreviation: "Hf",
    atomic_weight: "178.486",
    group: Some(4),
    period: 6,
    block: "d",
    electronegativity: Some(1.3),
    melting_point: Some(2233.0),
    boiling_point: Some(4603.0),
    atomic_radius: Some(208.0),
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 2],
      &[2],
    ],
  },
  Element {
    atomic_number: 73,
    standard_name: "Tantalum",
    abbreviation: "Ta",
    atomic_weight: "180.94788",
    group: Some(5),
    period: 6,
    block: "d",
    electronegativity: Some(1.5),
    melting_point: Some(3017.0),
    boiling_point: Some(5458.0),
    atomic_radius: Some(200.0),
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 3],
      &[2],
    ],
  },
  Element {
    atomic_number: 74,
    standard_name: "Tungsten",
    abbreviation: "W",
    atomic_weight: "183.84",
    group: Some(6),
    period: 6,
    block: "d",
    electronegativity: Some(2.36),
    melting_point: Some(3422.0),
    boiling_point: Some(5555.0),
    atomic_radius: Some(193.0),
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 4],
      &[2],
    ],
  },
  Element {
    atomic_number: 75,
    standard_name: "Rhenium",
    abbreviation: "Re",
    atomic_weight: "186.207",
    group: Some(7),
    period: 6,
    block: "d",
    electronegativity: Some(1.9),
    melting_point: Some(3186.0),
    boiling_point: Some(5596.0),
    atomic_radius: Some(188.0),
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 5],
      &[2],
    ],
  },
  Element {
    atomic_number: 76,
    standard_name: "Osmium",
    abbreviation: "Os",
    atomic_weight: "190.23",
    group: Some(8),
    period: 6,
    block: "d",
    electronegativity: Some(2.2),
    melting_point: Some(3033.0),
    boiling_point: Some(5012.0),
    atomic_radius: Some(185.0),
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 6],
      &[2],
    ],
  },
  Element {
    atomic_number: 77,
    standard_name: "Iridium",
    abbreviation: "Ir",
    atomic_weight: "192.217",
    group: Some(9),
    period: 6,
    block: "d",
    electronegativity: Some(2.2),
    melting_point: Some(2466.0),
    boiling_point: Some(4428.0),
    atomic_radius: Some(180.0),
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 7],
      &[2],
    ],
  },
  Element {
    atomic_number: 78,
    standard_name: "Platinum",
    abbreviation: "Pt",
    atomic_weight: "195.084",
    group: Some(10),
    period: 6,
    block: "d",
    electronegativity: Some(2.28),
    melting_point: Some(1768.3),
    boiling_point: Some(3825.0),
    atomic_radius: Some(177.0),
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 9],
      &[1],
    ],
  },
  Element {
    atomic_number: 79,
    standard_name: "Gold",
    abbreviation: "Au",
    atomic_weight: "196.96657",
    group: Some(11),
    period: 6,
    block: "d",
    electronegativity: Some(2.54),
    melting_point: Some(1064.18),
    boiling_point: Some(2856.0),
    atomic_radius: Some(174.0),
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10],
      &[1],
    ],
  },
  Element {
    atomic_number: 80,
    standard_name: "Mercury",
    abbreviation: "Hg",
    atomic_weight: "200.592",
    group: Some(12),
    period: 6,
    block: "d",
    electronegativity: Some(2.0),
    melting_point: Some(-38.83),
    boiling_point: Some(356.73),
    atomic_radius: Some(171.0),
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10],
      &[2],
    ],
  },
  Element {
    atomic_number: 81,
    standard_name: "Thallium",
    abbreviation: "Tl",
    atomic_weight: "204.38",
    group: Some(13),
    period: 6,
    block: "p",
    electronegativity: Some(1.62),
    melting_point: Some(304.0),
    boiling_point: Some(1473.0),
    atomic_radius: Some(156.0),
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10],
      &[2, 1],
    ],
  },
  Element {
    atomic_number: 82,
    standard_name: "Lead",
    abbreviation: "Pb",
    atomic_weight: "207.2",
    group: Some(14),
    period: 6,
    block: "p",
    electronegativity: Some(2.33),
    melting_point: Some(327.46),
    boiling_point: Some(1749.0),
    atomic_radius: Some(154.0),
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10],
      &[2, 2],
    ],
  },
  Element {
    atomic_number: 83,
    standard_name: "Bismuth",
    abbreviation: "Bi",
    atomic_weight: "208.9804",
    group: Some(15),
    period: 6,
    block: "p",
    electronegativity: Some(2.02),
    melting_point: Some(271.3),
    boiling_point: Some(1564.0),
    atomic_radius: Some(143.0),
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10],
      &[2, 3],
    ],
  },
  Element {
    atomic_number: 84,
    standard_name: "Polonium",
    abbreviation: "Po",
    atomic_weight: "209.",
    group: Some(16),
    period: 6,
    block: "p",
    electronegativity: Some(2.0),
    melting_point: Some(254.0),
    boiling_point: Some(962.0),
    atomic_radius: Some(135.0),
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10],
      &[2, 4],
    ],
  },
  Element {
    atomic_number: 85,
    standard_name: "Astatine",
    abbreviation: "At",
    atomic_weight: "210.",
    group: Some(17),
    period: 6,
    block: "p",
    electronegativity: Some(2.2),
    melting_point: Some(302.0),
    boiling_point: None,
    atomic_radius: Some(127.0),
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10],
      &[2, 5],
    ],
  },
  Element {
    atomic_number: 86,
    standard_name: "Radon",
    abbreviation: "Rn",
    atomic_weight: "222.",
    group: Some(18),
    period: 6,
    block: "p",
    electronegativity: None,
    melting_point: Some(-71.0),
    boiling_point: Some(-61.7),
    atomic_radius: Some(120.0),
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10],
      &[2, 6],
    ],
  },
  Element {
    atomic_number: 87,
    standard_name: "Francium",
    abbreviation: "Fr",
    atomic_weight: "223.",
    group: Some(1),
    period: 7,
    block: "s",
    electronegativity: Some(0.7),
    melting_point: None,
    boiling_point: None,
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10],
      &[2, 6],
      &[1],
    ],
  },
  Element {
    atomic_number: 88,
    standard_name: "Radium",
    abbreviation: "Ra",
    atomic_weight: "226.",
    group: Some(2),
    period: 7,
    block: "s",
    electronegativity: Some(0.9),
    melting_point: Some(700.0),
    boiling_point: Some(1737.0),
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10],
      &[2, 6],
      &[2],
    ],
  },
  Element {
    atomic_number: 89,
    standard_name: "Actinium",
    abbreviation: "Ac",
    atomic_weight: "227.",
    group: None,
    period: 7,
    block: "f",
    electronegativity: Some(1.1),
    melting_point: Some(1050.0),
    boiling_point: Some(3200.0),
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10],
      &[2, 6, 1],
      &[2],
    ],
  },
  Element {
    atomic_number: 90,
    standard_name: "Thorium",
    abbreviation: "Th",
    atomic_weight: "232.0377",
    group: None,
    period: 7,
    block: "f",
    electronegativity: Some(1.3),
    melting_point: Some(1750.0),
    boiling_point: Some(4820.0),
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10],
      &[2, 6, 2],
      &[2],
    ],
  },
  Element {
    atomic_number: 91,
    standard_name: "Protactinium",
    abbreviation: "Pa",
    atomic_weight: "231.03588",
    group: None,
    period: 7,
    block: "f",
    electronegativity: Some(1.5),
    melting_point: Some(1572.0),
    boiling_point: Some(4000.0),
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10, 2],
      &[2, 6, 1],
      &[2],
    ],
  },
  Element {
    atomic_number: 92,
    standard_name: "Uranium",
    abbreviation: "U",
    atomic_weight: "238.02891",
    group: None,
    period: 7,
    block: "f",
    electronegativity: Some(1.38),
    melting_point: Some(1135.0),
    boiling_point: Some(3927.0),
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10, 3],
      &[2, 6, 1],
      &[2],
    ],
  },
  Element {
    atomic_number: 93,
    standard_name: "Neptunium",
    abbreviation: "Np",
    atomic_weight: "237.",
    group: None,
    period: 7,
    block: "f",
    electronegativity: Some(1.36),
    melting_point: Some(644.0),
    boiling_point: Some(4000.0),
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10, 4],
      &[2, 6, 1],
      &[2],
    ],
  },
  Element {
    atomic_number: 94,
    standard_name: "Plutonium",
    abbreviation: "Pu",
    atomic_weight: "244.",
    group: None,
    period: 7,
    block: "f",
    electronegativity: Some(1.28),
    melting_point: Some(640.0),
    boiling_point: Some(3230.0),
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10, 6],
      &[2, 6],
      &[2],
    ],
  },
  Element {
    atomic_number: 95,
    standard_name: "Americium",
    abbreviation: "Am",
    atomic_weight: "243.",
    group: None,
    period: 7,
    block: "f",
    electronegativity: Some(1.3),
    melting_point: Some(1176.0),
    boiling_point: Some(2011.0),
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10, 7],
      &[2, 6],
      &[2],
    ],
  },
  Element {
    atomic_number: 96,
    standard_name: "Curium",
    abbreviation: "Cm",
    atomic_weight: "247.",
    group: None,
    period: 7,
    block: "f",
    electronegativity: Some(1.3),
    melting_point: Some(1345.0),
    boiling_point: Some(3110.0),
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10, 7],
      &[2, 6, 1],
      &[2],
    ],
  },
  Element {
    atomic_number: 97,
    standard_name: "Berkelium",
    abbreviation: "Bk",
    atomic_weight: "247.",
    group: None,
    period: 7,
    block: "f",
    electronegativity: Some(1.3),
    melting_point: Some(1050.0),
    boiling_point: None,
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10, 9],
      &[2, 6],
      &[2],
    ],
  },
  Element {
    atomic_number: 98,
    standard_name: "Californium",
    abbreviation: "Cf",
    atomic_weight: "251.",
    group: None,
    period: 7,
    block: "f",
    electronegativity: Some(1.3),
    melting_point: Some(900.0),
    boiling_point: None,
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10, 10],
      &[2, 6],
      &[2],
    ],
  },
  Element {
    atomic_number: 99,
    standard_name: "Einsteinium",
    abbreviation: "Es",
    atomic_weight: "252.",
    group: None,
    period: 7,
    block: "f",
    electronegativity: Some(1.3),
    melting_point: Some(860.0),
    boiling_point: None,
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10, 11],
      &[2, 6],
      &[2],
    ],
  },
  Element {
    atomic_number: 100,
    standard_name: "Fermium",
    abbreviation: "Fm",
    atomic_weight: "257.",
    group: None,
    period: 7,
    block: "f",
    electronegativity: Some(1.3),
    melting_point: Some(1527.0),
    boiling_point: None,
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10, 12],
      &[2, 6],
      &[2],
    ],
  },
  Element {
    atomic_number: 101,
    standard_name: "Mendelevium",
    abbreviation: "Md",
    atomic_weight: "258.",
    group: None,
    period: 7,
    block: "f",
    electronegativity: Some(1.3),
    melting_point: Some(827.0),
    boiling_point: None,
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10, 13],
      &[2, 6],
      &[2],
    ],
  },
  Element {
    atomic_number: 102,
    standard_name: "Nobelium",
    abbreviation: "No",
    atomic_weight: "259.",
    group: None,
    period: 7,
    block: "f",
    electronegativity: Some(1.3),
    melting_point: Some(827.0),
    boiling_point: None,
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10, 14],
      &[2, 6],
      &[2],
    ],
  },
  Element {
    atomic_number: 103,
    standard_name: "Lawrencium",
    abbreviation: "Lr",
    atomic_weight: "262.",
    group: Some(3),
    period: 7,
    block: "d",
    electronegativity: None,
    melting_point: Some(1627.0),
    boiling_point: None,
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10, 14],
      &[2, 6, 1],
      &[2],
    ],
  },
  Element {
    atomic_number: 104,
    standard_name: "Rutherfordium",
    abbreviation: "Rf",
    atomic_weight: "267.",
    group: Some(4),
    period: 7,
    block: "d",
    electronegativity: None,
    melting_point: None,
    boiling_point: None,
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10, 14],
      &[2, 6, 2],
      &[2],
    ],
  },
  Element {
    atomic_number: 105,
    standard_name: "Dubnium",
    abbreviation: "Db",
    atomic_weight: "268.",
    group: Some(5),
    period: 7,
    block: "d",
    electronegativity: None,
    melting_point: None,
    boiling_point: None,
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10, 14],
      &[2, 6, 3],
      &[2],
    ],
  },
  Element {
    atomic_number: 106,
    standard_name: "Seaborgium",
    abbreviation: "Sg",
    atomic_weight: "269.",
    group: Some(6),
    period: 7,
    block: "d",
    electronegativity: None,
    melting_point: None,
    boiling_point: None,
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10, 14],
      &[2, 6, 4],
      &[2],
    ],
  },
  Element {
    atomic_number: 107,
    standard_name: "Bohrium",
    abbreviation: "Bh",
    atomic_weight: "270.",
    group: Some(7),
    period: 7,
    block: "d",
    electronegativity: None,
    melting_point: None,
    boiling_point: None,
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10, 14],
      &[2, 6, 5],
      &[2],
    ],
  },
  Element {
    atomic_number: 108,
    standard_name: "Hassium",
    abbreviation: "Hs",
    atomic_weight: "269.",
    group: Some(8),
    period: 7,
    block: "d",
    electronegativity: None,
    melting_point: None,
    boiling_point: None,
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10, 14],
      &[2, 6, 6],
      &[2],
    ],
  },
  Element {
    atomic_number: 109,
    standard_name: "Meitnerium",
    abbreviation: "Mt",
    atomic_weight: "277.",
    group: Some(9),
    period: 7,
    block: "d",
    electronegativity: None,
    melting_point: None,
    boiling_point: None,
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10, 14],
      &[2, 6, 7],
      &[2],
    ],
  },
  Element {
    atomic_number: 110,
    standard_name: "Darmstadtium",
    abbreviation: "Ds",
    atomic_weight: "281.",
    group: Some(10),
    period: 7,
    block: "d",
    electronegativity: None,
    melting_point: None,
    boiling_point: None,
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10, 14],
      &[2, 6, 8],
      &[2],
    ],
  },
  Element {
    atomic_number: 111,
    standard_name: "Roentgenium",
    abbreviation: "Rg",
    atomic_weight: "282.",
    group: Some(11),
    period: 7,
    block: "d",
    electronegativity: None,
    melting_point: None,
    boiling_point: None,
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10, 14],
      &[2, 6, 9],
      &[2],
    ],
  },
  Element {
    atomic_number: 112,
    standard_name: "Copernicium",
    abbreviation: "Cn",
    atomic_weight: "285.",
    group: Some(12),
    period: 7,
    block: "d",
    electronegativity: None,
    melting_point: None,
    boiling_point: None,
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10, 14],
      &[2, 6, 10],
      &[2],
    ],
  },
  Element {
    atomic_number: 113,
    standard_name: "Nihonium",
    abbreviation: "Nh",
    atomic_weight: "286.",
    group: Some(13),
    period: 7,
    block: "p",
    electronegativity: None,
    melting_point: None,
    boiling_point: None,
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10, 14],
      &[2, 6, 10],
      &[2, 1],
    ],
  },
  Element {
    atomic_number: 114,
    standard_name: "Flerovium",
    abbreviation: "Fl",
    atomic_weight: "290.",
    group: Some(14),
    period: 7,
    block: "p",
    electronegativity: None,
    melting_point: None,
    boiling_point: None,
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10, 14],
      &[2, 6, 10],
      &[2, 2],
    ],
  },
  Element {
    atomic_number: 115,
    standard_name: "Moscovium",
    abbreviation: "Mc",
    atomic_weight: "290.",
    group: Some(15),
    period: 7,
    block: "p",
    electronegativity: None,
    melting_point: None,
    boiling_point: None,
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10, 14],
      &[2, 6, 10],
      &[2, 3],
    ],
  },
  Element {
    atomic_number: 116,
    standard_name: "Livermorium",
    abbreviation: "Lv",
    atomic_weight: "293.",
    group: Some(16),
    period: 7,
    block: "p",
    electronegativity: None,
    melting_point: None,
    boiling_point: None,
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10, 14],
      &[2, 6, 10],
      &[2, 4],
    ],
  },
  Element {
    atomic_number: 117,
    standard_name: "Tennessine",
    abbreviation: "Ts",
    atomic_weight: "294.",
    group: Some(17),
    period: 7,
    block: "p",
    electronegativity: None,
    melting_point: None,
    boiling_point: None,
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10, 14],
      &[2, 6, 10],
      &[2, 5],
    ],
  },
  Element {
    atomic_number: 118,
    standard_name: "Oganesson",
    abbreviation: "Og",
    atomic_weight: "294.",
    group: Some(18),
    period: 7,
    block: "p",
    electronegativity: None,
    melting_point: None,
    boiling_point: None,
    atomic_radius: None,
    electron_configuration: &[
      &[2],
      &[2, 6],
      &[2, 6, 10],
      &[2, 6, 10, 14],
      &[2, 6, 10, 14],
      &[2, 6, 10],
      &[2, 6],
    ],
  },
];

/// Look up an element by name (case-insensitive), abbreviation, or atomic number
fn find_element(identifier: &Expr) -> Option<&'static Element> {
  match identifier {
    Expr::Integer(n) => {
      let n = *n;
      if (1..=118).contains(&n) {
        Some(&ELEMENTS[(n - 1) as usize])
      } else {
        None
      }
    }
    Expr::String(s) => find_element_by_string(s),
    Expr::Identifier(s) => find_element_by_string(s),
    _ => None,
  }
}

fn find_element_by_string(s: &str) -> Option<&'static Element> {
  let lower = s.to_lowercase();
  // Try by standard name (case-insensitive)
  for elem in ELEMENTS.iter() {
    if elem.standard_name.to_lowercase() == lower {
      return Some(elem);
    }
  }
  // Try by abbreviation (case-sensitive first, then insensitive)
  for elem in ELEMENTS.iter() {
    if elem.abbreviation == s {
      return Some(elem);
    }
  }
  ELEMENTS
    .iter()
    .find(|&elem| elem.abbreviation.to_lowercase() == lower)
    .map(|v| v as _)
}

fn get_property(elem: &Element, property: &str) -> Expr {
  match property {
    "Name" => Expr::String(elem.standard_name.to_lowercase()),
    "StandardName" => Expr::String(elem.standard_name.to_string()),
    "Abbreviation" => Expr::String(elem.abbreviation.to_string()),
    "AtomicNumber" => Expr::Integer(elem.atomic_number),
    "AtomicWeight" => {
      let prec = ATOMIC_WEIGHT_PRECISIONS[(elem.atomic_number - 1) as usize];
      make_quantity(
        Expr::BigFloat(elem.atomic_weight.to_string(), prec),
        "AtomicMassUnit",
      )
    }
    "Group" => match elem.group {
      Some(g) => Expr::Integer(g),
      None => Expr::FunctionCall {
        name: "Missing".to_string(),
        args: vec![Expr::String("Undefined".to_string())],
      },
    },
    "Period" => Expr::Integer(elem.period),
    "Block" => Expr::String(elem.block.to_string()),
    "Electronegativity" | "ElectroNegativity" => match elem.electronegativity {
      Some(v) => {
        let prec = electronegativity_precision(v);
        make_bigfloat(v, prec)
      }
      None => missing_not_applicable(),
    },
    "MeltingPoint" => match elem.melting_point {
      Some(v) => {
        let prec = temp_precision(v);
        make_quantity(make_bigfloat(v, prec), "DegreesCelsius")
      }
      None => {
        if elem.electronegativity.is_none() && elem.melting_point.is_none() {
          missing_not_available()
        } else {
          missing_not_applicable()
        }
      }
    },
    "BoilingPoint" => match elem.boiling_point {
      Some(v) => {
        let prec = temp_precision(v);
        make_quantity(make_bigfloat(v, prec), "DegreesCelsius")
      }
      None => missing_not_available(),
    },
    "AbsoluteBoilingPoint" => match elem.boiling_point {
      Some(v) => {
        // Celsius → Kelvin. Round to 2 decimals to mask IEEE-754 drift
        // (source data has 2 decimal places).
        let kelvin = ((v + 273.15) * 100.0).round() / 100.0;
        Expr::Real(kelvin)
      }
      None => missing_not_available(),
    },
    "AbsoluteMeltingPoint" => match elem.melting_point {
      Some(v) => {
        let kelvin = ((v + 273.15) * 100.0).round() / 100.0;
        Expr::Real(kelvin)
      }
      None => {
        if elem.electronegativity.is_none() && elem.melting_point.is_none() {
          missing_not_available()
        } else {
          missing_not_applicable()
        }
      }
    },
    "AtomicRadius" => match elem.atomic_radius {
      Some(v) => {
        let prec = if v >= 100.0 { 3.0 } else { 2.0 };
        make_quantity(make_bigfloat(v, prec), "Picometers")
      }
      None => missing_not_available(),
    },
    "ElectronConfiguration" => {
      let shells: Vec<Expr> = elem
        .electron_configuration
        .iter()
        .map(|shell| {
          Expr::List(shell.iter().map(|&n| Expr::Integer(n as i128)).collect())
        })
        .collect();
      Expr::List(shells)
    }
    "ElectronConfigurationString" => {
      Expr::String(format_electron_configuration(elem))
    }
    // Properties we recognise by name but don't yet have tabulated data for —
    // mirror Mathematica's behaviour of returning Missing[NotAvailable] so
    // callers can distinguish "unknown datum" from "unrecognised property".
    "SpecificHeat"
    | "BrinellHardness"
    | "BulkModulus"
    | "CovalentRadius"
    | "CrustAbundance"
    | "Density"
    | "DiscoveryYear"
    | "ElectronAffinity"
    | "ElectronShellConfiguration"
    | "FusionHeat"
    | "IonizationEnergies"
    | "LiquidDensity"
    | "MohsHardness"
    | "PoissonRatio"
    | "Series"
    | "ShearModulus"
    | "ThermalConductivity"
    | "VanDerWaalsRadius"
    | "VaporizationHeat"
    | "VickersHardness"
    | "YoungModulus" => missing_not_available(),
    _ => missing_not_found(),
  }
}

/// Format an element's electron configuration as a string like
/// `[Ne] 3s2 3p4`. Uses the largest noble gas whose shells are a prefix of
/// the element's shells; the remainder becomes explicit `nS<count>` terms.
fn format_electron_configuration(elem: &Element) -> String {
  // Noble gases (group 18), ordered by atomic number descending.
  const NOBLE_GASES: &[(i128, &str)] = &[
    (118, "Og"),
    (86, "Rn"),
    (54, "Xe"),
    (36, "Kr"),
    (18, "Ar"),
    (10, "Ne"),
    (2, "He"),
  ];

  let elem_shells = elem.electron_configuration;

  // Find the largest noble gas whose config is a subshell-level prefix of
  // elem's config. "Subshell prefix" means for each shell of the noble gas,
  // elem's corresponding shell starts with the same subshell occupancies;
  // any shell beyond the noble gas's last is considered entirely "remaining".
  let mut prefix_sym: Option<&str> = None;
  let mut prefix_ng_shells: &[&[u8]] = &[];
  for &(z, sym) in NOBLE_GASES {
    if z >= elem.atomic_number {
      continue;
    }
    let ng_shells = ELEMENTS[(z - 1) as usize].electron_configuration;
    if ng_shells.len() > elem_shells.len() {
      continue;
    }
    // Each of the noble gas's shells must be a prefix of elem's shell.
    let is_prefix = (0..ng_shells.len()).all(|i| {
      let ng = ng_shells[i];
      let el = elem_shells[i];
      ng.len() <= el.len() && ng.iter().zip(el.iter()).all(|(a, b)| a == b)
    });
    if is_prefix {
      prefix_sym = Some(sym);
      prefix_ng_shells = ng_shells;
      break;
    }
  }

  let mut parts: Vec<String> = Vec::new();
  if let Some(sym) = prefix_sym {
    parts.push(format!("[{}]", sym));
  }
  const SUBSHELL_LETTERS: &[char] = &['s', 'p', 'd', 'f', 'g', 'h', 'i'];
  for (shell_idx, shell) in elem_shells.iter().enumerate() {
    let n = shell_idx + 1;
    let ng_shell_opt = prefix_ng_shells.get(shell_idx).copied();
    for (sub_idx, &count) in shell.iter().enumerate() {
      if count == 0 {
        continue;
      }
      // Skip subshells already covered by the noble gas prefix.
      if let Some(ng) = ng_shell_opt
        && sub_idx < ng.len()
        && ng[sub_idx] == count
      {
        continue;
      }
      let letter = SUBSHELL_LETTERS.get(sub_idx).copied().unwrap_or('?');
      parts.push(format!("{}{}{}", n, letter, count));
    }
  }
  parts.join(" ")
}

fn electronegativity_precision(v: f64) -> f64 {
  let frac = v - v.floor();
  if frac.abs() < 1e-10 {
    // Integer value (e.g. 1.0, 2.0, 3.0) → precision 2
    2.0
  } else if v >= 1.0 {
    // e.g. 1.9, 2.2, 3.98 → precision 3
    3.0
  } else {
    // e.g. 0.7 → 1, 0.82 → 2, 0.98 → 2
    let s = format!("{}", v);
    let s = s.trim_start_matches("0.");
    s.len() as f64
  }
}

fn temp_precision(v: f64) -> f64 {
  let s = format!("{}", v);
  let s_clean = s.trim_start_matches('-');
  let digits: String = s_clean.chars().filter(|c| c.is_ascii_digit()).collect();
  let digits = digits.trim_start_matches('0');
  if digits.is_empty() {
    1.0
  } else {
    let count = digits.len() as f64;
    // At least 2 digits of precision for temperatures
    if count < 2.0 { 2.0 } else { count }
  }
}

static SUPPORTED_PROPERTIES: &[&str] = &[
  "Abbreviation",
  "AtomicNumber",
  "AtomicRadius",
  "AtomicWeight",
  "Block",
  "BoilingPoint",
  "ElectronConfiguration",
  "Electronegativity",
  "Group",
  "MeltingPoint",
  "Name",
  "Period",
  "StandardName",
];

pub fn element_data_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  match args.len() {
    0 => {
      // ElementData[] — return list of all elements as Entity expressions
      let entities: Vec<Expr> = ELEMENTS
        .iter()
        .map(|elem| Expr::FunctionCall {
          name: "Entity".to_string(),
          args: vec![
            Expr::String("Element".to_string()),
            Expr::String(elem.standard_name.to_string()),
          ],
        })
        .collect();
      Ok(Expr::List(entities))
    }
    1 => {
      // ElementData["Properties"] or ElementData[All] or ElementData[element]
      match &args[0] {
        Expr::String(s) if s == "Properties" => {
          let props: Vec<Expr> = SUPPORTED_PROPERTIES
            .iter()
            .map(|p| Expr::String(p.to_string()))
            .collect();
          Ok(Expr::List(props))
        }
        Expr::Identifier(s) if s == "All" => {
          let entities: Vec<Expr> = ELEMENTS
            .iter()
            .map(|elem| Expr::FunctionCall {
              name: "Entity".to_string(),
              args: vec![
                Expr::String("Element".to_string()),
                Expr::String(elem.standard_name.to_string()),
              ],
            })
            .collect();
          Ok(Expr::List(entities))
        }
        identifier => {
          // ElementData[element] returns Entity[Element, name]
          match find_element(identifier) {
            Some(elem) => Ok(Expr::FunctionCall {
              name: "Entity".to_string(),
              args: vec![
                Expr::String("Element".to_string()),
                Expr::String(elem.standard_name.to_string()),
              ],
            }),
            None => Ok(missing_not_found()),
          }
        }
      }
    }
    2 => {
      // ElementData[element, property]
      let elem = match find_element(&args[0]) {
        Some(e) => e,
        None => {
          return Ok(Expr::FunctionCall {
            name: "ElementData".to_string(),
            args: args.to_vec(),
          });
        }
      };
      match &args[1] {
        Expr::String(prop) => Ok(get_property(elem, prop)),
        _ => Ok(Expr::FunctionCall {
          name: "ElementData".to_string(),
          args: args.to_vec(),
        }),
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "ElementData".to_string(),
      args: args.to_vec(),
    }),
  }
}
