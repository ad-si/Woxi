//! Minimal built-in country knowledge base.
//!
//! Woxi has no connection to the Wolfram Knowledgebase, but a handful of
//! deterministic, static facts (most importantly populations) are easy to
//! bundle. This module backs three entry points:
//!
//!   * `Interpreter["Country"][name]` — resolve a free-form country name to
//!     `Entity["Country", canonical]` (or `Missing` when unknown).
//!   * `CountryData[name, "Population"]` — `Quantity[n, "People"]`.
//!   * `EntityValue[Entity["Country", name], "Population"]` — same value.
//!
//! Populations are recent (≈2023) round-number estimates. They are not meant
//! to match the Wolfram Knowledgebase to the last digit; they give Woxi a
//! self-contained, reproducible country dataset.

use crate::InterpreterError;
use crate::syntax::{Expr, unevaluated};

/// A single sovereign country: a canonical display name, its population in
/// number of people, and alternate spellings/abbreviations that should
/// resolve to it.
struct Country {
  canonical: &'static str,
  population: i64,
  aliases: &'static [&'static str],
}

#[rustfmt::skip]
static COUNTRIES: &[Country] = &[
  Country { canonical: "Afghanistan", population: 41_128_771, aliases: &[] },
  Country { canonical: "Albania", population: 2_842_321, aliases: &[] },
  Country { canonical: "Algeria", population: 44_903_225, aliases: &[] },
  Country { canonical: "Andorra", population: 79_824, aliases: &[] },
  Country { canonical: "Angola", population: 35_588_987, aliases: &[] },
  Country { canonical: "Antigua and Barbuda", population: 93_763, aliases: &["Antigua & Barbuda"] },
  Country { canonical: "Argentina", population: 46_234_830, aliases: &[] },
  Country { canonical: "Armenia", population: 2_780_469, aliases: &[] },
  Country { canonical: "Australia", population: 26_177_413, aliases: &[] },
  Country { canonical: "Austria", population: 8_939_617, aliases: &[] },
  Country { canonical: "Azerbaijan", population: 10_358_074, aliases: &[] },
  Country { canonical: "Bahamas", population: 409_984, aliases: &["The Bahamas"] },
  Country { canonical: "Bahrain", population: 1_472_233, aliases: &[] },
  Country { canonical: "Bangladesh", population: 171_186_372, aliases: &[] },
  Country { canonical: "Barbados", population: 281_635, aliases: &[] },
  Country { canonical: "Belarus", population: 9_534_954, aliases: &[] },
  Country { canonical: "Belgium", population: 11_655_930, aliases: &[] },
  Country { canonical: "Belize", population: 405_272, aliases: &[] },
  Country { canonical: "Benin", population: 13_352_864, aliases: &[] },
  Country { canonical: "Bhutan", population: 782_455, aliases: &[] },
  Country { canonical: "Bolivia", population: 12_224_110, aliases: &[] },
  Country { canonical: "Bosnia and Herzegovina", population: 3_233_526, aliases: &["Bosnia & Herzegovina", "Bosnia", "Bosnia-Herzegovina"] },
  Country { canonical: "Botswana", population: 2_630_296, aliases: &[] },
  Country { canonical: "Brazil", population: 215_313_498, aliases: &[] },
  Country { canonical: "Brunei", population: 449_002, aliases: &["Brunei Darussalam"] },
  Country { canonical: "Bulgaria", population: 6_781_953, aliases: &[] },
  Country { canonical: "Burkina Faso", population: 22_673_762, aliases: &[] },
  Country { canonical: "Burundi", population: 12_889_576, aliases: &[] },
  Country { canonical: "Cambodia", population: 16_767_842, aliases: &[] },
  Country { canonical: "Cameroon", population: 27_914_536, aliases: &[] },
  Country { canonical: "Canada", population: 38_929_902, aliases: &[] },
  Country { canonical: "Cape Verde", population: 593_149, aliases: &["Cabo Verde"] },
  Country { canonical: "Central African Republic", population: 5_579_144, aliases: &[] },
  Country { canonical: "Chad", population: 17_723_315, aliases: &[] },
  Country { canonical: "Chile", population: 19_603_733, aliases: &[] },
  Country { canonical: "China", population: 1_412_175_000, aliases: &["People's Republic of China", "PRC", "Mainland China"] },
  Country { canonical: "Colombia", population: 51_874_024, aliases: &[] },
  Country { canonical: "Comoros", population: 836_774, aliases: &[] },
  Country { canonical: "Congo", population: 5_970_424, aliases: &["Republic of the Congo", "Congo-Brazzaville"] },
  Country { canonical: "Costa Rica", population: 5_180_829, aliases: &[] },
  Country { canonical: "Croatia", population: 3_899_000, aliases: &[] },
  Country { canonical: "Cuba", population: 11_212_191, aliases: &[] },
  Country { canonical: "Cyprus", population: 1_251_488, aliases: &[] },
  Country { canonical: "Czech Republic", population: 10_672_103, aliases: &["Czechia"] },
  Country { canonical: "Democratic Republic of the Congo", population: 99_010_212, aliases: &["DR Congo", "DRC", "Congo-Kinshasa", "Congo (Kinshasa)", "Zaire"] },
  Country { canonical: "Denmark", population: 5_903_037, aliases: &[] },
  Country { canonical: "Djibouti", population: 1_120_849, aliases: &[] },
  Country { canonical: "Dominica", population: 72_737, aliases: &[] },
  Country { canonical: "Dominican Republic", population: 11_228_821, aliases: &[] },
  Country { canonical: "Ecuador", population: 18_001_000, aliases: &[] },
  Country { canonical: "Egypt", population: 110_990_103, aliases: &[] },
  Country { canonical: "El Salvador", population: 6_336_392, aliases: &[] },
  Country { canonical: "Equatorial Guinea", population: 1_674_908, aliases: &[] },
  Country { canonical: "Eritrea", population: 3_684_032, aliases: &[] },
  Country { canonical: "Estonia", population: 1_326_062, aliases: &[] },
  Country { canonical: "Eswatini", population: 1_201_670, aliases: &["Swaziland"] },
  Country { canonical: "Ethiopia", population: 123_379_924, aliases: &[] },
  Country { canonical: "Fiji", population: 929_766, aliases: &[] },
  Country { canonical: "Finland", population: 5_540_745, aliases: &[] },
  Country { canonical: "France", population: 67_971_311, aliases: &[] },
  Country { canonical: "Gabon", population: 2_388_992, aliases: &[] },
  Country { canonical: "Gambia", population: 2_705_992, aliases: &["The Gambia"] },
  Country { canonical: "Georgia", population: 3_744_385, aliases: &[] },
  Country { canonical: "Germany", population: 83_797_985, aliases: &[] },
  Country { canonical: "Ghana", population: 33_475_870, aliases: &[] },
  Country { canonical: "Greece", population: 10_426_919, aliases: &[] },
  Country { canonical: "Grenada", population: 124_610, aliases: &[] },
  Country { canonical: "Guatemala", population: 17_357_886, aliases: &[] },
  Country { canonical: "Guinea", population: 13_859_341, aliases: &[] },
  Country { canonical: "Guinea-Bissau", population: 2_105_566, aliases: &[] },
  Country { canonical: "Guyana", population: 808_726, aliases: &[] },
  Country { canonical: "Haiti", population: 11_584_996, aliases: &[] },
  Country { canonical: "Honduras", population: 10_432_860, aliases: &[] },
  Country { canonical: "Hungary", population: 9_643_048, aliases: &[] },
  Country { canonical: "Iceland", population: 375_318, aliases: &[] },
  Country { canonical: "India", population: 1_417_173_173, aliases: &[] },
  Country { canonical: "Indonesia", population: 275_501_339, aliases: &[] },
  Country { canonical: "Iran", population: 88_550_570, aliases: &["Islamic Republic of Iran", "Persia"] },
  Country { canonical: "Iraq", population: 44_496_122, aliases: &[] },
  Country { canonical: "Ireland", population: 5_127_170, aliases: &["Republic of Ireland"] },
  Country { canonical: "Israel", population: 9_557_500, aliases: &[] },
  Country { canonical: "Italy", population: 58_940_425, aliases: &[] },
  Country { canonical: "Ivory Coast", population: 28_160_542, aliases: &["Cote d'Ivoire", "Côte d'Ivoire"] },
  Country { canonical: "Jamaica", population: 2_827_377, aliases: &[] },
  Country { canonical: "Japan", population: 124_516_650, aliases: &[] },
  Country { canonical: "Jordan", population: 11_285_869, aliases: &[] },
  Country { canonical: "Kazakhstan", population: 19_621_972, aliases: &[] },
  Country { canonical: "Kenya", population: 54_027_487, aliases: &[] },
  Country { canonical: "Kiribati", population: 131_232, aliases: &[] },
  Country { canonical: "Kosovo", population: 1_761_985, aliases: &[] },
  Country { canonical: "Kuwait", population: 4_268_873, aliases: &[] },
  Country { canonical: "Kyrgyzstan", population: 6_630_623, aliases: &[] },
  Country { canonical: "Laos", population: 7_529_475, aliases: &["Lao People's Democratic Republic"] },
  Country { canonical: "Latvia", population: 1_879_383, aliases: &[] },
  Country { canonical: "Lebanon", population: 5_489_739, aliases: &[] },
  Country { canonical: "Lesotho", population: 2_305_825, aliases: &[] },
  Country { canonical: "Liberia", population: 5_302_681, aliases: &[] },
  Country { canonical: "Libya", population: 6_812_341, aliases: &[] },
  Country { canonical: "Liechtenstein", population: 39_327, aliases: &[] },
  Country { canonical: "Lithuania", population: 2_831_639, aliases: &[] },
  Country { canonical: "Luxembourg", population: 647_599, aliases: &[] },
  Country { canonical: "Madagascar", population: 29_611_714, aliases: &[] },
  Country { canonical: "Malawi", population: 20_405_317, aliases: &[] },
  Country { canonical: "Malaysia", population: 33_938_221, aliases: &[] },
  Country { canonical: "Maldives", population: 523_787, aliases: &[] },
  Country { canonical: "Mali", population: 22_593_590, aliases: &[] },
  Country { canonical: "Malta", population: 531_113, aliases: &[] },
  Country { canonical: "Marshall Islands", population: 41_569, aliases: &[] },
  Country { canonical: "Mauritania", population: 4_736_139, aliases: &[] },
  Country { canonical: "Mauritius", population: 1_262_523, aliases: &[] },
  Country { canonical: "Mexico", population: 127_504_125, aliases: &[] },
  Country { canonical: "Micronesia", population: 114_164, aliases: &["Federated States of Micronesia"] },
  Country { canonical: "Moldova", population: 2_601_315, aliases: &[] },
  Country { canonical: "Monaco", population: 36_469, aliases: &[] },
  Country { canonical: "Mongolia", population: 3_398_366, aliases: &[] },
  Country { canonical: "Montenegro", population: 627_082, aliases: &[] },
  Country { canonical: "Morocco", population: 37_457_971, aliases: &[] },
  Country { canonical: "Mozambique", population: 32_969_517, aliases: &[] },
  Country { canonical: "Myanmar", population: 54_179_306, aliases: &["Burma"] },
  Country { canonical: "Namibia", population: 2_567_012, aliases: &[] },
  Country { canonical: "Nauru", population: 12_668, aliases: &[] },
  Country { canonical: "Nepal", population: 30_547_580, aliases: &[] },
  Country { canonical: "Netherlands", population: 17_700_982, aliases: &["Holland", "The Netherlands"] },
  Country { canonical: "New Zealand", population: 5_124_100, aliases: &[] },
  Country { canonical: "Nicaragua", population: 6_948_392, aliases: &[] },
  Country { canonical: "Niger", population: 26_207_977, aliases: &[] },
  Country { canonical: "Nigeria", population: 218_541_212, aliases: &[] },
  Country { canonical: "North Korea", population: 26_069_416, aliases: &["Democratic People's Republic of Korea", "DPRK", "Korea, North"] },
  Country { canonical: "North Macedonia", population: 2_093_599, aliases: &["Macedonia"] },
  Country { canonical: "Norway", population: 5_457_127, aliases: &[] },
  Country { canonical: "Oman", population: 4_576_298, aliases: &[] },
  Country { canonical: "Pakistan", population: 235_824_862, aliases: &[] },
  Country { canonical: "Palau", population: 18_055, aliases: &[] },
  Country { canonical: "Palestine", population: 5_043_612, aliases: &["State of Palestine", "Palestinian Territories"] },
  Country { canonical: "Panama", population: 4_408_581, aliases: &[] },
  Country { canonical: "Papua New Guinea", population: 10_142_619, aliases: &[] },
  Country { canonical: "Paraguay", population: 6_780_744, aliases: &[] },
  Country { canonical: "Peru", population: 34_049_588, aliases: &[] },
  Country { canonical: "Philippines", population: 115_559_009, aliases: &[] },
  Country { canonical: "Poland", population: 36_821_749, aliases: &[] },
  Country { canonical: "Portugal", population: 10_409_704, aliases: &[] },
  Country { canonical: "Qatar", population: 2_695_122, aliases: &[] },
  Country { canonical: "Romania", population: 19_047_009, aliases: &[] },
  Country { canonical: "Russia", population: 144_236_933, aliases: &["Russian Federation"] },
  Country { canonical: "Rwanda", population: 13_776_698, aliases: &[] },
  Country { canonical: "Saint Kitts and Nevis", population: 47_657, aliases: &["St. Kitts and Nevis"] },
  Country { canonical: "Saint Lucia", population: 179_857, aliases: &["St. Lucia"] },
  Country { canonical: "Saint Vincent and the Grenadines", population: 103_948, aliases: &["St. Vincent and the Grenadines"] },
  Country { canonical: "Samoa", population: 222_382, aliases: &[] },
  Country { canonical: "San Marino", population: 33_660, aliases: &[] },
  Country { canonical: "Sao Tome and Principe", population: 227_380, aliases: &["São Tomé and Príncipe"] },
  Country { canonical: "Saudi Arabia", population: 36_408_820, aliases: &[] },
  Country { canonical: "Senegal", population: 17_316_449, aliases: &[] },
  Country { canonical: "Serbia", population: 6_664_449, aliases: &[] },
  Country { canonical: "Seychelles", population: 107_118, aliases: &[] },
  Country { canonical: "Sierra Leone", population: 8_605_718, aliases: &[] },
  Country { canonical: "Singapore", population: 5_637_022, aliases: &[] },
  Country { canonical: "Slovakia", population: 5_431_752, aliases: &[] },
  Country { canonical: "Slovenia", population: 2_111_986, aliases: &[] },
  Country { canonical: "Solomon Islands", population: 723_995, aliases: &[] },
  Country { canonical: "Somalia", population: 17_597_511, aliases: &[] },
  Country { canonical: "South Africa", population: 59_893_885, aliases: &[] },
  Country { canonical: "South Korea", population: 51_815_810, aliases: &["Republic of Korea", "Korea, South", "Korea"] },
  Country { canonical: "South Sudan", population: 10_913_164, aliases: &[] },
  Country { canonical: "Spain", population: 47_558_630, aliases: &[] },
  Country { canonical: "Sri Lanka", population: 22_181_000, aliases: &[] },
  Country { canonical: "Sudan", population: 46_874_204, aliases: &[] },
  Country { canonical: "Suriname", population: 618_040, aliases: &[] },
  Country { canonical: "Sweden", population: 10_549_347, aliases: &[] },
  Country { canonical: "Switzerland", population: 8_775_760, aliases: &[] },
  Country { canonical: "Syria", population: 22_125_249, aliases: &["Syrian Arab Republic"] },
  Country { canonical: "Taiwan", population: 23_893_394, aliases: &["Republic of China", "Chinese Taipei"] },
  Country { canonical: "Tajikistan", population: 9_952_787, aliases: &[] },
  Country { canonical: "Tanzania", population: 65_497_748, aliases: &[] },
  Country { canonical: "Thailand", population: 71_697_030, aliases: &[] },
  Country { canonical: "Timor-Leste", population: 1_341_296, aliases: &["East Timor"] },
  Country { canonical: "Togo", population: 8_848_699, aliases: &[] },
  Country { canonical: "Tonga", population: 106_858, aliases: &[] },
  Country { canonical: "Trinidad and Tobago", population: 1_531_044, aliases: &["Trinidad & Tobago"] },
  Country { canonical: "Tunisia", population: 12_356_117, aliases: &[] },
  Country { canonical: "Turkey", population: 85_341_241, aliases: &["Türkiye", "Turkiye"] },
  Country { canonical: "Turkmenistan", population: 6_430_770, aliases: &[] },
  Country { canonical: "Tuvalu", population: 11_312, aliases: &[] },
  Country { canonical: "Uganda", population: 47_249_585, aliases: &[] },
  Country { canonical: "Ukraine", population: 38_000_000, aliases: &[] },
  Country { canonical: "United Arab Emirates", population: 9_441_129, aliases: &["UAE", "Emirates"] },
  Country { canonical: "United Kingdom", population: 66_971_411, aliases: &["UK", "Britain", "Great Britain", "United Kingdom of Great Britain and Northern Ireland"] },
  Country { canonical: "United States", population: 333_287_557, aliases: &["USA", "U.S.A.", "U.S.", "US", "United States of America", "America"] },
  Country { canonical: "Uruguay", population: 3_422_794, aliases: &[] },
  Country { canonical: "Uzbekistan", population: 35_648_100, aliases: &[] },
  Country { canonical: "Vanuatu", population: 326_740, aliases: &[] },
  Country { canonical: "Vatican City", population: 825, aliases: &["Holy See"] },
  Country { canonical: "Venezuela", population: 28_301_696, aliases: &[] },
  Country { canonical: "Vietnam", population: 98_186_856, aliases: &["Viet Nam"] },
  Country { canonical: "Yemen", population: 33_696_614, aliases: &[] },
  Country { canonical: "Zambia", population: 20_017_675, aliases: &[] },
  Country { canonical: "Zimbabwe", population: 16_320_537, aliases: &[] },
];

/// Normalize a country name for matching: trim, lowercase, replace `&` with
/// `and`, drop characters that aren't ASCII alphanumerics or spaces, and
/// collapse internal whitespace. So `"Bosnia & Herzegovina"`,
/// `" bosnia and herzegovina "`, and `"Bosnia-Herzegovina"` all normalize to
/// the same key.
fn normalize(name: &str) -> String {
  let lowered = name.trim().to_lowercase().replace('&', " and ");
  let mut out = String::with_capacity(lowered.len());
  let mut prev_space = false;
  for ch in lowered.chars() {
    if ch.is_ascii_alphanumeric() {
      out.push(ch);
      prev_space = false;
    } else if ch.is_whitespace() || ch == '-' {
      if !prev_space && !out.is_empty() {
        out.push(' ');
      }
      prev_space = true;
    }
    // All other characters (punctuation, diacritics) are dropped.
  }
  out.trim_end().to_string()
}

/// Resolve a free-form country name (canonical or alias) to its canonical
/// display name, or `None` if the country is unknown. Used to key geographic
/// geometry to the knowledge base.
pub fn canonical_name(name: &str) -> Option<&'static str> {
  lookup(name).map(|c| c.canonical)
}

/// Resolve a free-form country name to its dataset entry.
fn lookup(name: &str) -> Option<&'static Country> {
  let key = normalize(name);
  if key.is_empty() {
    return None;
  }
  COUNTRIES.iter().find(|c| {
    normalize(c.canonical) == key
      || c.aliases.iter().any(|a| normalize(a) == key)
  })
}

fn make_quantity(magnitude: Expr, unit: &str) -> Expr {
  Expr::FunctionCall {
    name: "Quantity".to_string(),
    args: vec![magnitude, Expr::String(unit.to_string())].into(),
  }
}

fn missing(reason: &str, name: &str) -> Expr {
  Expr::FunctionCall {
    name: "Missing".to_string(),
    args: vec![
      Expr::String(reason.to_string()),
      Expr::String(name.to_string()),
    ]
    .into(),
  }
}

/// Look up a property of a country given by canonical name (or alias), for the
/// `EntityValue[Entity["Country", name], property]` / `CountryData` paths.
/// Returns `None` only when the country itself is unknown, so the caller can
/// fall back to its own handling; an unknown *property* returns `Missing`.
pub fn country_property(name: &str, property: &str) -> Option<Expr> {
  let country = lookup(name)?;
  Some(match property {
    "Population" => {
      make_quantity(Expr::Integer(country.population as i128), "People")
    }
    "Name" => Expr::String(country.canonical.to_string()),
    _ => missing("NotAvailable", property),
  })
}

/// `CountryData[name, property]` — currently `"Population"` and `"Name"`.
/// `CountryData[name]` returns the canonical `Entity["Country", name]`.
pub fn country_data_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  match args {
    [Expr::String(name)] => Ok(match lookup(name) {
      Some(c) => entity(c.canonical),
      None => missing("UnknownCountry", name),
    }),
    [Expr::String(name), Expr::String(property)] => Ok(
      country_property(name, property)
        .unwrap_or_else(|| missing("UnknownCountry", name)),
    ),
    _ => Ok(unevaluated("CountryData", args)),
  }
}

fn entity(canonical: &str) -> Expr {
  Expr::FunctionCall {
    name: "Entity".to_string(),
    args: vec![
      Expr::String("Country".to_string()),
      Expr::String(canonical.to_string()),
    ]
    .into(),
  }
}

/// Parse `val` according to a scalar Interpreter type. Returns `None` when the
/// type is unknown or the value cannot be interpreted (so the caller leaves the
/// expression unevaluated).
fn interpret_scalar(ty: &str, val: &Expr) -> Option<Expr> {
  // Is the string an integer literal (optional sign, digits only)?
  let parse_integer = |s: &str| -> Option<Expr> {
    let t = s.trim();
    match t.parse::<i128>() {
      Ok(n) => Some(Expr::Integer(n)),
      Err(_) => match t.parse::<num_bigint::BigInt>() {
        Ok(n) => Some(Expr::BigInteger(n)),
        Err(_) => None,
      },
    }
  };
  let parse_real =
    |s: &str| -> Option<Expr> { s.trim().parse::<f64>().ok().map(Expr::Real) };

  match ty {
    "String" => match val {
      Expr::String(s) => Some(Expr::String(s.clone())),
      _ => None,
    },
    "Boolean" => {
      let s = match val {
        Expr::String(s) => s.trim().to_lowercase(),
        _ => return None,
      };
      match s.as_str() {
        "true" => Some(Expr::Identifier("True".to_string())),
        "false" => Some(Expr::Identifier("False".to_string())),
        _ => None,
      }
    }
    "Integer" => match val {
      Expr::Integer(n) => Some(Expr::Integer(*n)),
      Expr::BigInteger(n) => Some(Expr::BigInteger(n.clone())),
      Expr::String(s) => parse_integer(s),
      _ => None,
    },
    "Real" => match val {
      Expr::Integer(n) => Some(Expr::Real(*n as f64)),
      Expr::Real(r) => Some(Expr::Real(*r)),
      Expr::String(s) => parse_real(s),
      _ => None,
    },
    "Number" => match val {
      Expr::Integer(n) => Some(Expr::Integer(*n)),
      Expr::Real(r) => Some(Expr::Real(*r)),
      // Integer literals stay integers; anything else numeric becomes a Real.
      Expr::String(s) => parse_integer(s).or_else(|| parse_real(s)),
      _ => None,
    },
    _ => None,
  }
}

/// `Interpreter["Country"][input]` — resolve `input` to `Entity["Country", …]`
/// or `Missing` when it can't be interpreted. `domain` is the argument to
/// `Interpreter` (here `"Country"`); `applied` are the arguments it is then
/// called with. Other domains stay unevaluated.
pub fn apply_interpreter(
  domain: &Expr,
  applied: &[Expr],
) -> Result<Expr, InterpreterError> {
  if let Expr::String(d) = domain
    && d == "Country"
    && let [Expr::String(input)] = applied
  {
    return Ok(match lookup(input) {
      Some(c) => entity(c.canonical),
      None => missing("NoInterpretation", input),
    });
  }

  // Scalar interpreter types: parse a string (or numeric input) into a typed
  // value, matching wolframscript. Unparseable inputs stay unevaluated (the
  // Failure object wolframscript returns is not reproduced here).
  if let Expr::String(d) = domain
    && let [val] = applied
    && let Some(result) = interpret_scalar(d, val)
  {
    return Ok(result);
  }

  // Unhandled domain/arguments: keep the curried form unevaluated.
  Ok(Expr::CurriedCall {
    func: Box::new(Expr::FunctionCall {
      name: "Interpreter".to_string(),
      args: vec![domain.clone()].into(),
    }),
    args: applied.to_vec(),
  })
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn resolves_common_aliases() {
    assert_eq!(lookup("USA").unwrap().canonical, "United States");
    assert_eq!(
      lookup("Bosnia & Herzegovina").unwrap().canonical,
      "Bosnia and Herzegovina"
    );
    assert_eq!(lookup("Czechia").unwrap().canonical, "Czech Republic");
    assert_eq!(lookup("Cabo Verde").unwrap().canonical, "Cape Verde");
    assert_eq!(lookup("Türkiye").unwrap().canonical, "Turkey");
    assert_eq!(lookup("  south korea ").unwrap().canonical, "South Korea");
  }

  #[test]
  fn non_sovereign_names_do_not_resolve() {
    assert!(lookup("Scotland").is_none());
    assert!(lookup("Curaçao").is_none());
  }

  #[test]
  fn population_is_a_quantity() {
    let p = country_property("Germany", "Population").unwrap();
    match &p {
      Expr::FunctionCall { name, args } => {
        assert_eq!(name, "Quantity");
        assert!(matches!(&args[0], Expr::Integer(n) if *n > 0));
        assert!(matches!(&args[1], Expr::String(u) if u == "People"));
      }
      _ => panic!("expected a Quantity FunctionCall"),
    }
  }
}
