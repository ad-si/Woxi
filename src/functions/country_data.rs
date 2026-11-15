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

use super::*;

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
  legacy(name).map(|c| c.canonical)
}

/// All countries as `Entity["Country", canonical]` — the built-in "Country"
/// `EntityList`/`EntityValue["Entities"]` knowledge base.
pub fn country_entities() -> Vec<Expr> {
  RECORDS.iter().map(|r| entity(r.entity)).collect()
}

/// Countries (by canonical name) in which each language is an official
/// language; backs the `"<Language>Speaking"` properties.
#[rustfmt::skip]
static LANGUAGE_COUNTRIES: &[(&str, &[&str])] = &[
  ("Arabic", &["Algeria", "Bahrain", "Chad", "Comoros", "Djibouti", "Egypt", "Eritrea", "Iraq", "Israel", "Jordan", "Kuwait", "Lebanon", "Libya", "Mauritania", "Morocco", "Oman", "Palestine", "Qatar", "Saudi Arabia", "Somalia", "South Sudan", "Sudan", "Syria", "Tunisia", "United Arab Emirates", "Yemen"]),
  ("Danish", &["Denmark"]),
  ("Dutch", &["Belgium", "Netherlands", "Suriname"]),
  ("English", &["Antigua and Barbuda", "Australia", "Bahamas", "Barbados", "Belize", "Botswana", "Burundi", "Cameroon", "Canada", "Dominica", "Eritrea", "Eswatini", "Fiji", "Gambia", "Ghana", "Grenada", "Guyana", "India", "Ireland", "Jamaica", "Kenya", "Kiribati", "Lesotho", "Liberia", "Malawi", "Malta", "Marshall Islands", "Mauritius", "Micronesia", "Namibia", "Nauru", "New Zealand", "Nigeria", "Pakistan", "Palau", "Papua New Guinea", "Philippines", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "Seychelles", "Sierra Leone", "Singapore", "Solomon Islands", "South Africa", "South Sudan", "Sudan", "Tanzania", "Tonga", "Trinidad and Tobago", "Tuvalu", "Uganda", "United Kingdom", "United States", "Zambia", "Zimbabwe"]),
  ("French", &["Belgium", "Benin", "Burkina Faso", "Burundi", "Cameroon", "Canada", "Central African Republic", "Chad", "Comoros", "Congo", "Democratic Republic of the Congo", "Djibouti", "Equatorial Guinea", "France", "Gabon", "Guinea", "Haiti", "Ivory Coast", "Luxembourg", "Madagascar", "Mali", "Monaco", "Niger", "Rwanda", "Senegal", "Seychelles", "Switzerland", "Togo", "Vanuatu"]),
  ("German", &["Austria", "Belgium", "Germany", "Liechtenstein", "Luxembourg", "Switzerland"]),
  ("Italian", &["Italy", "San Marino", "Switzerland", "Vatican City"]),
  ("Portuguese", &["Angola", "Brazil", "Cape Verde", "Equatorial Guinea", "Guinea-Bissau", "Mozambique", "Portugal", "Sao Tome and Principe", "Timor-Leste"]),
  ("Spanish", &["Argentina", "Bolivia", "Chile", "Colombia", "Costa Rica", "Cuba", "Dominican Republic", "Ecuador", "El Salvador", "Equatorial Guinea", "Guatemala", "Honduras", "Mexico", "Nicaragua", "Panama", "Paraguay", "Peru", "Spain", "Uruguay", "Venezuela"]),
];

/// Country classes accepted by `CountryData[class]`, listing member countries
/// by canonical name.
#[rustfmt::skip]
static COUNTRY_CLASSES: &[(&str, &[&str])] = &[
  ("EU", &["Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"]),
  ("Europe", &["Albania", "Andorra", "Austria", "Belarus", "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Kosovo", "Latvia", "Liechtenstein", "Lithuania", "Luxembourg", "Malta", "Moldova", "Monaco", "Montenegro", "Netherlands", "North Macedonia", "Norway", "Poland", "Portugal", "Romania", "Russia", "San Marino", "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "Ukraine", "United Kingdom", "Vatican City"]),
];

/// Resolve a free-form country name to its dataset entry. Names written
/// without spaces (`"CzechRepublic"`) resolve like their spaced forms.
fn lookup(name: &str) -> Option<&'static Country> {
  let key = normalize(name);
  if key.is_empty() {
    return None;
  }
  let exact = COUNTRIES.iter().find(|c| {
    normalize(c.canonical) == key
      || c.aliases.iter().any(|a| normalize(a) == key)
  });
  exact.or_else(|| {
    let squashed = key.replace(' ', "");
    COUNTRIES.iter().find(|c| {
      normalize(c.canonical).replace(' ', "") == squashed
        || c
          .aliases
          .iter()
          .any(|a| normalize(a).replace(' ', "") == squashed)
    })
  })
}

use crate::functions::element_data::make_quantity;

/// First year of the yearly series in `resources/country_data.tsv.gz`.
const FIRST_YEAR: i64 = 1960;

/// One row of `resources/country_data.tsv.gz` (World Bank statistics under
/// the Wolfram Language's entity names; see `scripts/gen_country_data.py`).
struct Record {
  entity: &'static str,
  iso3: &'static str,
  name: &'static str,
  full_name: &'static str,
  continents: Vec<&'static str>,
  area: Option<f64>,
  land_area: Option<f64>,
  population: Vec<Option<f64>>,
  gdp: Vec<Option<f64>>,
}

static RECORDS: std::sync::LazyLock<Vec<Record>> =
  std::sync::LazyLock::new(|| {
    use flate2::read::GzDecoder;
    use std::io::Read;
    let compressed = include_bytes!("../../resources/country_data.tsv.gz");
    let mut text = String::new();
    GzDecoder::new(&compressed[..])
      .read_to_string(&mut text)
      .expect("country_data.tsv.gz is valid gzip");
    let text: &'static str = Box::leak(text.into_boxed_str());
    let series = |s: &str| -> Vec<Option<f64>> {
      s.split(',').map(|v| v.parse().ok()).collect()
    };
    text
      .lines()
      .filter(|l| !l.starts_with('#') && !l.is_empty())
      .filter_map(|line| {
        let f: Vec<&'static str> = line.split('\t').collect();
        (f.len() == 9).then(|| Record {
          entity: f[0],
          iso3: f[1],
          name: f[2],
          full_name: f[3],
          continents: f[4].split(',').filter(|c| !c.is_empty()).collect(),
          area: f[5].parse().ok(),
          land_area: f[6].parse().ok(),
          population: series(f[7]),
          gdp: series(f[8]),
        })
      })
      .collect()
  });

fn squash(s: &str) -> String {
  normalize(s).replace(' ', "")
}

/// The statistics row of a country given by entity name (`"UnitedStates"`),
/// display name, official name or any alias of the legacy table.
fn record(name: &str) -> Option<&'static Record> {
  let records = &*RECORDS;
  if let Some(r) = records.iter().find(|r| r.entity == name) {
    return Some(r);
  }
  let key = squash(name);
  if key.is_empty() {
    return None;
  }
  records
    .iter()
    .find(|r| {
      squash(r.entity) == key
        || squash(r.name) == key
        || squash(r.full_name) == key
    })
    .or_else(|| {
      let canonical = squash(lookup(name)?.canonical);
      records
        .iter()
        .find(|r| squash(r.name) == canonical || squash(r.entity) == canonical)
        .or_else(|| {
          // The legacy table's aliases name the record (`"Congo"` is the
          // `"Republic of the Congo"`).
          let c = lookup(name)?;
          records.iter().find(|r| {
            c.aliases.iter().any(|a| {
              squash(a) == squash(r.name) || squash(a) == squash(r.entity)
            })
          })
        })
    })
}

/// The legacy entry (population, aliases, geometry key) of a country named
/// any way `record` understands.
fn legacy(name: &str) -> Option<&'static Country> {
  lookup(name).or_else(|| {
    let r = record(name)?;
    lookup(r.name).or_else(|| lookup(r.full_name))
  })
}

/// The Wolfram Language entity name (`"UnitedStates"`) of a country named
/// any way the knowledge base understands.
pub fn wl_entity_name(name: &str) -> Option<&'static str> {
  record(name)
    .map(|r| r.entity)
    .or_else(|| legacy(name).map(|c| c.canonical))
}

/// The Wolfram Language entity name of a legacy country.
fn entity_name(canonical: &'static str) -> &'static str {
  record(canonical).map_or(canonical, |r| r.entity)
}

fn latest(series: &[Option<f64>]) -> Option<(i64, f64)> {
  series
    .iter()
    .enumerate()
    .rev()
    .find_map(|(i, v)| v.map(|v| (FIRST_YEAR + i as i64, v)))
}

fn at_year(series: &[Option<f64>], year: i64) -> Option<f64> {
  let i = usize::try_from(year - FIRST_YEAR).ok()?;
  series.get(i).copied().flatten()
}

/// A yearly statistic as a Quantity, or `None` when the table has no such
/// series.
fn series_of(r: &'static Record, prop: &str) -> Option<Vec<Option<f64>>> {
  match prop {
    "Population" => Some(r.population.clone()),
    "GDP" => Some(r.gdp.clone()),
    "GDPPerCapita" => Some(
      r.gdp
        .iter()
        .zip(&r.population)
        .map(|(g, p)| match (g, p) {
          (Some(g), Some(p)) if *p > 0.0 => Some(g / p),
          _ => None,
        })
        .collect(),
    ),
    _ => None,
  }
}

fn eval_wl(src: &str) -> Result<Expr, InterpreterError> {
  let parsed = crate::functions::string_ast::parse_program_to_expr(src)?;
  crate::evaluator::evaluate_expr_to_expr(&parsed)
}

/// `Quantity[value, unit]` with a compound unit written as WL source.
fn quantity_in(value: Expr, unit_src: &str) -> Result<Expr, InterpreterError> {
  Ok(call("Quantity", vec![value, eval_wl(unit_src)?]))
}

fn statistic_quantity(
  prop: &str,
  value: f64,
) -> Result<Expr, InterpreterError> {
  match prop {
    "Population" => Ok(make_quantity(
      Expr::Integer(value.round() as i128),
      "People",
    )),
    "GDP" => quantity_in(Expr::Real(value), "\"USDollars\"/\"Years\""),
    _ => quantity_in(Expr::Real(value), "\"USDollars\"/(\"People\"*\"Years\")"),
  }
}

/// `CountryData[c, {{prop}, {y1, y2}}]` — the statistic over the years
/// as a `TimeSeries` of `{{year}, value}` pairs.
fn time_series(
  r: &'static Record,
  prop: &str,
  from: i64,
  to: i64,
) -> Result<Expr, InterpreterError> {
  let Some(series) = series_of(r, prop) else {
    return Ok(missing("NotAvailable", prop));
  };
  let pairs: Vec<Expr> = (from..=to)
    .filter_map(|y| {
      let v = at_year(&series, y)?;
      let v = if prop == "Population" {
        Expr::Integer(v.round() as i128)
      } else {
        Expr::Real(v)
      };
      Some(Expr::List(
        vec![Expr::List(vec![Expr::Integer(y as i128)].into()), v].into(),
      ))
    })
    .collect();
  if pairs.is_empty() {
    return Ok(missing("NotAvailable", prop));
  }
  crate::evaluator::evaluate_expr_to_expr(&call(
    "TimeSeries",
    vec![Expr::List(pairs.into())],
  ))
}

/// Properties answered from the statistics table.
fn record_property(
  r: &'static Record,
  property: &str,
) -> Result<Option<Expr>, InterpreterError> {
  Ok(Some(match property {
    "Name" => str_expr(r.name),
    "FullName" => str_expr(r.full_name),
    "StandardName" => str_expr(r.entity),
    "UNCode" if r.iso3.len() == 3 => str_expr(r.iso3),
    "Continent" => match r.continents.first() {
      Some(c) => call("EntityClass", vec![str_expr("Country"), str_expr(c)]),
      None => missing("NotApplicable", property),
    },
    "Area" | "LandArea" => {
      match if property == "Area" {
        r.area
      } else {
        r.land_area
      } {
        Some(a) => quantity_in(Expr::Real(a), "\"Kilometers\"^2")?,
        None => missing("NotAvailable", property),
      }
    }
    "Population" | "GDP" | "GDPPerCapita" => {
      match series_of(r, property).as_deref().and_then(latest) {
        Some((_, v)) => statistic_quantity(property, v)?,
        None => missing("NotAvailable", property),
      }
    }
    "Flag" => match flag_png(r.entity) {
      Some(png) => crate::functions::image_ast::import_image_from_bytes(&png)?,
      None => missing("NotAvailable", property),
    },
    "FlagDescription" => match FLAG_DESCRIPTIONS.get(r.entity) {
      Some(text) => str_expr(text),
      None => missing("NotAvailable", property),
    },
    _ => match indicator_unit(property) {
      Some(unit) => match indicator_value(r.entity, property) {
        Some(v) => indicator_quantity(unit, v)?,
        None => missing("NotAvailable", property),
      },
      None => return Ok(None),
    },
  }))
}

/// How the value of a property of `resources/country_indicators.tsv.gz`
/// is expressed.
#[derive(Clone, Copy, PartialEq, Debug)]
enum IndicatorUnit {
  /// `Quantity[n, "People"]`
  People,
  /// `Quantity[n, "People"/"Years"]`
  PeoplePerYear,
  /// `Quantity[x, 1/"Years"]` — a yearly rate as a fraction.
  PerYear,
  /// `Quantity[x, "Years"]`
  Years,
  /// `Quantity[x, "USDollars"/"Years"]`
  DollarsPerYear,
  /// `Quantity[x, "USDollars"]`
  Dollars,
  /// `Quantity[x, "Kilometers"]`
  Kilometers,
  /// A plain integer count.
  Count,
  /// A plain real number (fractions, indices, fertility rates).
  Number,
}

/// The unit of a property answered from the indicator table, or `None`
/// for properties the table does not cover.
fn indicator_unit(property: &str) -> Option<IndicatorUnit> {
  use IndicatorUnit::{
    Count, Dollars, DollarsPerYear, Kilometers, Number, People, PeoplePerYear,
    PerYear, Years,
  };
  Some(match property {
    "ChildPopulation"
    | "AdultPopulation"
    | "ElderlyPopulation"
    | "FemalePopulation"
    | "FemaleChildPopulation"
    | "FemaleAdultPopulation"
    | "FemaleElderlyPopulation"
    | "MalePopulation"
    | "MaleChildPopulation"
    | "MaleAdultPopulation"
    | "MaleElderlyPopulation"
    | "LaborForce"
    | "InternetUsers" => People,
    "AnnualBirths" | "AnnualDeaths" => PeoplePerYear,
    "PopulationGrowth"
    | "BirthRateFraction"
    | "DeathRateFraction"
    | "MigrationRateFraction"
    | "GDPRealGrowth"
    | "IndustrialProductionGrowth"
    | "InflationRate" => PerYear,
    "LifeExpectancy" | "FemaleLifeExpectancy" | "MaleLifeExpectancy" => Years,
    "CurrentAccountBalance"
    | "EconomicAid"
    | "GDPAtParity"
    | "ManufacturingValueAdded"
    | "NationalIncome"
    | "AgriculturalValueAdded"
    | "IndustrialValueAdded"
    | "TotalConsumption"
    | "HouseholdConsumption"
    | "GovernmentConsumption"
    | "GrossInvestment"
    | "FixedInvestment"
    | "ExportValue"
    | "ImportValue"
    | "InventoryChange" => DollarsPerYear,
    "ExternalDebt" | "ForeignExchangeReserves" => Dollars,
    "RailwayLength" => Kilometers,
    "PhoneLines" | "CellularPhones" => Count,
    "TotalFertilityRate"
    | "InfantMortalityFraction"
    | "FemaleInfantMortalityFraction"
    | "MaleInfantMortalityFraction"
    | "HIVAIDSFraction"
    | "LiteracyFraction"
    | "FemaleLiteracyFraction"
    | "MaleLiteracyFraction"
    | "UnemploymentFraction"
    | "PovertyFraction"
    | "GiniIndex" => Number,
    _ => return None,
  })
}

fn indicator_quantity(
  unit: IndicatorUnit,
  v: f64,
) -> Result<Expr, InterpreterError> {
  use IndicatorUnit::{
    Count, Dollars, DollarsPerYear, Kilometers, Number, People, PeoplePerYear,
    PerYear, Years,
  };
  let int = || Expr::Integer(v.round() as i128);
  Ok(match unit {
    People => make_quantity(int(), "People"),
    PeoplePerYear => quantity_in(int(), "\"People\"/\"Years\"")?,
    PerYear => quantity_in(Expr::Real(v), "1/\"Years\"")?,
    Years => make_quantity(Expr::Real(v), "Years"),
    DollarsPerYear => quantity_in(Expr::Real(v), "\"USDollars\"/\"Years\"")?,
    Dollars => make_quantity(Expr::Real(v), "USDollars"),
    Kilometers => make_quantity(Expr::Real(v), "Kilometers"),
    Count => int(),
    Number => Expr::Real(v),
  })
}

/// The indicator table: property names from its header line, and each
/// country's latest values (see `scripts/gen_country_indicators.py`).
struct Indicators {
  properties: Vec<&'static str>,
  rows: std::collections::HashMap<&'static str, Vec<Option<f64>>>,
}

static INDICATORS: std::sync::LazyLock<Indicators> =
  std::sync::LazyLock::new(|| {
    use flate2::read::GzDecoder;
    use std::io::Read;
    let compressed =
      include_bytes!("../../resources/country_indicators.tsv.gz");
    let mut text = String::new();
    GzDecoder::new(&compressed[..])
      .read_to_string(&mut text)
      .expect("country_indicators.tsv.gz is valid gzip");
    let text: &'static str = Box::leak(text.into_boxed_str());
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    let properties = lines
      .next()
      .map_or_default(|h| h.split('\t').skip(1).collect());
    let rows = lines
      .filter_map(|line| {
        let mut f = line.split('\t');
        let entity = f.next().filter(|e| !e.is_empty())?;
        Some((entity, f.map(|v| v.parse().ok()).collect()))
      })
      .collect();
    Indicators { properties, rows }
  });

/// The latest value of an indicator-table property of a country.
fn indicator_value(entity: &str, property: &str) -> Option<f64> {
  let table = &*INDICATORS;
  let col = table.properties.iter().position(|p| *p == property)?;
  table.rows.get(entity)?.get(col).copied().flatten()
}

/// Base64-encoded PNG flags keyed by entity name, from
/// `resources/country_flags.tsv.gz` (see `scripts/gen_country_flags.py`).
static FLAGS: std::sync::LazyLock<
  std::collections::HashMap<&'static str, &'static str>,
> = std::sync::LazyLock::new(|| {
  use flate2::read::GzDecoder;
  use std::io::Read;
  let compressed = include_bytes!("../../resources/country_flags.tsv.gz");
  let mut text = String::new();
  GzDecoder::new(&compressed[..])
    .read_to_string(&mut text)
    .expect("country_flags.tsv.gz is valid gzip");
  let text: &'static str = Box::leak(text.into_boxed_str());
  text
    .lines()
    .filter(|l| !l.starts_with('#'))
    .filter_map(|l| l.split_once('\t'))
    .collect()
});

/// Plain-text flag descriptions keyed by entity name, from
/// `resources/country_flag_descriptions.tsv.gz` (the CIA World Factbook,
/// see `scripts/gen_country_flag_descriptions.py`).
static FLAG_DESCRIPTIONS: std::sync::LazyLock<
  std::collections::HashMap<&'static str, &'static str>,
> = std::sync::LazyLock::new(|| {
  use flate2::read::GzDecoder;
  use std::io::Read;
  let compressed =
    include_bytes!("../../resources/country_flag_descriptions.tsv.gz");
  let mut text = String::new();
  GzDecoder::new(&compressed[..])
    .read_to_string(&mut text)
    .expect("country_flag_descriptions.tsv.gz is valid gzip");
  let text: &'static str = Box::leak(text.into_boxed_str());
  text
    .lines()
    .filter(|l| !l.starts_with('#'))
    .filter_map(|l| l.split_once('\t'))
    .collect()
});

/// The decoded PNG bytes of a country's flag.
fn flag_png(entity: &str) -> Option<Vec<u8>> {
  use base64::Engine as _;
  let b64 = FLAGS.get(entity)?;
  base64::engine::general_purpose::STANDARD.decode(b64).ok()
}

/// `Entity["Country", name]` objects of every country whose continent list
/// contains `continent`.
fn continent_members(continent: &str) -> Vec<Expr> {
  RECORDS
    .iter()
    .filter(|r| r.continents.contains(&continent))
    .map(|r| entity(r.entity))
    .collect()
}

fn continents() -> Vec<&'static str> {
  // Antarctica is a continent class even though no country belongs to it.
  let mut out: Vec<&'static str> = vec!["Antarctica"];
  for r in RECORDS.iter() {
    for c in &r.continents {
      if !out.contains(c) {
        out.push(c);
      }
    }
  }
  out.sort_unstable();
  out
}

/// Members of a class name: a continent, `"Countries"`, or one of the
/// static classes (`"EU"`, `"Europe"`).
fn class_members(class: &str) -> Option<Vec<Expr>> {
  if class == "Countries" {
    return Some(country_entities());
  }
  if continents().contains(&class) {
    return Some(continent_members(class));
  }
  let members = COUNTRY_CLASSES
    .iter()
    .find_map(|(c, ms)| (*c == class).then_some(ms))?;
  let mut names: Vec<&'static str> =
    members.iter().map(|m| entity_name(m)).collect();
  names.sort_unstable();
  Some(names.into_iter().map(entity).collect())
}

/// Borders between countries: two countries border when their Natural Earth
/// outlines share a vertex.
fn bordering_countries(r: &'static Record) -> Option<Expr> {
  use std::collections::{HashMap, HashSet};
  static ADJ: std::sync::OnceLock<HashMap<&'static str, Vec<&'static str>>> =
    std::sync::OnceLock::new();
  let adj = ADJ.get_or_init(|| {
    let mut owners: HashMap<(i32, i32), HashSet<&'static str>> = HashMap::new();
    for rec in RECORDS.iter() {
      let Some(rings) =
        crate::functions::geographics::country_outline_rings(Some(rec.name))
          .or_else(|| {
            crate::functions::geographics::country_outline_rings(Some(
              rec.full_name,
            ))
          })
      else {
        continue;
      };
      for ring in rings {
        for &(lon, lat) in ring {
          owners
            .entry(((lon * 100.0).round() as i32, (lat * 100.0).round() as i32))
            .or_default()
            .insert(rec.entity);
        }
      }
    }
    let mut adj: HashMap<&'static str, Vec<&'static str>> = HashMap::new();
    for set in owners.values() {
      if set.len() < 2 {
        continue;
      }
      for a in set {
        for b in set {
          if a != b {
            let list = adj.entry(a).or_default();
            if !list.contains(b) {
              list.push(b);
            }
          }
        }
      }
    }
    for list in adj.values_mut() {
      list.sort_unstable();
    }
    adj
  });
  crate::functions::geographics::country_outline_rings(Some(r.name))?;
  Some(Expr::List(
    adj
      .get(r.entity)
      .map_or_default(|l| l.iter().map(|e| entity(e)).collect::<Vec<_>>())
      .into(),
  ))
}

fn missing(reason: &str, name: &str) -> Expr {
  call("Missing", vec![str_expr(reason), str_expr(name)])
}

/// Look up a property of a country given by canonical name (or alias), for the
/// `EntityValue[Entity["Country", name], property]` / `CountryData` paths.
/// Returns `None` only when the country itself is unknown, so the caller can
/// fall back to its own handling; an unknown *property* returns `Missing`.
pub fn country_property(name: &str, property: &str) -> Option<Expr> {
  if name == "World" && property == "SchematicCoordinates" {
    return Some(
      crate::functions::geographics::world_schematic_coordinates()
        .unwrap_or_else(|| missing("NotAvailable", property)),
    );
  }
  if property == "SchematicCoordinates" && lookup(name).is_none() {
    // Territories outside the sovereign list (e.g. Antarctica).
    return crate::functions::geographics::country_schematic_coordinates(name);
  }
  let rec = record(name);
  if let Some(r) = rec {
    match property {
      "CenterCoordinates" => {
        if let Some((lat, lon)) =
          crate::functions::geographics::country_center(r.name).or_else(|| {
            crate::functions::geographics::country_center(r.full_name)
          })
        {
          return Some(Expr::List(
            vec![Expr::Real(lat), Expr::Real(lon)].into(),
          ));
        }
      }
      "BorderingCountries" => {
        return Some(
          bordering_countries(r)
            .unwrap_or_else(|| missing("NotAvailable", property)),
        );
      }
      _ => {}
    }
    if let Ok(Some(v)) = record_property(r, property) {
      return Some(v);
    }
  }
  let Some(country) = legacy(name) else {
    return rec.map(|_| missing("NotAvailable", property));
  };
  if let Some(lang) = property.strip_suffix("Speaking")
    && let Some(countries) = LANGUAGE_COUNTRIES
      .iter()
      .find_map(|(l, cs)| (*l == lang).then_some(cs))
  {
    return Some(Expr::Identifier(
      if countries.contains(&country.canonical) {
        "True"
      } else {
        "False"
      }
      .to_string(),
    ));
  }
  if property == "SchematicPolygon" || property == "Polygon" {
    return Some(
      crate::functions::geographics::country_schematic_polygon(
        country.canonical,
      )
      .unwrap_or_else(|| missing("NotAvailable", property)),
    );
  }
  if property == "SchematicCoordinates"
    || property == "Coordinates"
    || property == "FullCoordinates"
  {
    return Some(
      crate::functions::geographics::country_schematic_coordinates(
        country.canonical,
      )
      .unwrap_or_else(|| missing("NotAvailable", property)),
    );
  }
  Some(match property {
    "Population" => {
      make_quantity(Expr::Integer(country.population as i128), "People")
    }
    "Name" => str_expr(country.canonical),
    "FullName" => str_expr(country.canonical),
    // WL's `Graphics[{fill, EdgeForm[Black], Polygon[{ring, …}]}]`, here
    // with the rings in plain {lon, lat} degrees.
    "Shape" => match crate::functions::geographics::country_outline_rings(Some(
      country.canonical,
    )) {
      Some(rings) if !rings.is_empty() => {
        let rings: Vec<Expr> = rings
          .iter()
          .map(|ring| {
            Expr::List(
              ring
                .iter()
                .map(|&(lon, lat)| {
                  Expr::List(
                    vec![Expr::Real(lon as f64), Expr::Real(lat as f64)].into(),
                  )
                })
                .collect::<Vec<_>>()
                .into(),
            )
          })
          .collect();
        eval_wl("{RGBColor[0.896, 0.8878, 0.8548], EdgeForm[GrayLevel[0]]}")
          .map_or_else(
            |_| missing("NotAvailable", property),
            |style| {
              let mut prims = match &style {
                Expr::List(items) => items.to_vec(),
                other => vec![other.clone()],
              };
              prims.push(call("Polygon", vec![Expr::List(rings.into())]));
              call("Graphics", vec![Expr::List(prims.into())])
            },
          )
      }
      _ => missing("NotAvailable", property),
    },
    _ => missing("NotAvailable", property),
  })
}

/// `CountryData[name, property]` — `"Population"`, `"Name"`, `"FullName"`,
/// `"Polygon"` and the `"<Language>Speaking"` flags. `CountryData[class]`
/// (`"EU"`, `"Europe"`) lists the member countries.
/// `CountryData[name]` returns the canonical `Entity["Country", name]`.
pub fn country_data_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Entity["Country", name] is accepted wherever a country name is.
  let normalized: Vec<Expr> = args
    .iter()
    .enumerate()
    .map(|(i, a)| match (i, a) {
      (0, Expr::FunctionCall { name, args: ea })
        if (name == "Entity" || name == "EntityClass")
          && ea.len() == 2
          && matches!(&ea[0], Expr::String(t) if t == "Country") =>
      {
        ea[1].clone()
      }
      _ => a.clone(),
    })
    .collect();
  let args = &normalized[..];
  match args {
    [] => Ok(Expr::List(country_entities().into())),
    [Expr::Identifier(all)] if all == "All" => {
      Ok(Expr::List(country_entities().into()))
    }
    [Expr::Identifier(all), prop] if all == "All" => {
      let values = country_entities()
        .iter()
        .map(|m| country_data_ast(&[m.clone(), prop.clone()]))
        .collect::<Result<Vec<_>, _>>()?;
      Ok(Expr::List(values.into()))
    }
    [Expr::String(s)] if s == "Continents" => Ok(Expr::List(
      continents()
        .into_iter()
        .map(|c| call("EntityClass", vec![str_expr("Country"), str_expr(c)]))
        .collect::<Vec<_>>()
        .into(),
    )),
    [Expr::String(class)]
      if class_members(class).is_some() && record(class).is_none() =>
    {
      Ok(Expr::List(class_members(class).unwrap().into()))
    }
    [Expr::String(name)] => Ok(match record(name) {
      Some(r) => entity(r.entity),
      None => match lookup(name) {
        Some(c) => entity(c.canonical),
        None => missing("UnknownCountry", name),
      },
    }),
    // A class with a property maps it over the members.
    [Expr::String(class), prop]
      if record(class).is_none() && class_members(class).is_some() =>
    {
      let members = class_members(class).unwrap();
      let values = members
        .iter()
        .map(|m| country_data_ast(&[m.clone(), prop.clone()]))
        .collect::<Result<Vec<_>, _>>()?;
      Ok(Expr::List(values.into()))
    }
    [Expr::String(name), Expr::String(property)] => Ok(
      country_property(name, property)
        .unwrap_or_else(|| missing("UnknownCountry", name)),
    ),
    // `{prop, year}`: the statistic in that year.
    [Expr::String(name), Expr::List(spec)]
      if matches!(spec.as_slice(), [Expr::String(_), Expr::Integer(_)]) =>
    {
      let (Expr::String(prop), Expr::Integer(year)) = (&spec[0], &spec[1])
      else {
        unreachable!()
      };
      let Some(r) = record(name) else {
        return Ok(missing("UnknownCountry", name));
      };
      match series_of(r, prop).and_then(|s| at_year(&s, *year as i64)) {
        Some(v) => statistic_quantity(prop, v),
        None if series_of(r, prop).is_some() => {
          Ok(missing("NotAvailable", prop))
        }
        None => Ok(unevaluated("CountryData", args)),
      }
    }
    // `{{prop}, {y1, y2}}` / `{{prop, currency}, {y1, y2}}` / `{prop, {y1,
    // y2}}`: the statistic over those years as a TimeSeries.
    [Expr::String(name), Expr::List(spec)]
      if spec.len() == 2
        && matches!(&spec[1], Expr::List(ys) if ys.len() == 2
          && ys.iter().all(|y| matches!(y, Expr::Integer(_)))) =>
    {
      let prop = match &spec[0] {
        Expr::String(p) => p.clone(),
        Expr::List(ps) => match ps.first() {
          Some(Expr::String(p)) => p.clone(),
          _ => return Ok(unevaluated("CountryData", args)),
        },
        _ => return Ok(unevaluated("CountryData", args)),
      };
      let Expr::List(ys) = &spec[1] else {
        unreachable!()
      };
      let (Expr::Integer(a), Expr::Integer(b)) = (&ys[0], &ys[1]) else {
        unreachable!()
      };
      match record(name) {
        Some(r) if series_of(r, &prop).is_some() => {
          time_series(r, &prop, *a as i64, *b as i64)
        }
        Some(_) => Ok(missing("NotAvailable", &prop)),
        None => Ok(missing("UnknownCountry", name)),
      }
    }
    [Expr::String(name), Expr::List(spec)]
      if spec.len() == 2
        && matches!(&spec[0], Expr::String(p) if p == "Shape" || p == "Polygon") =>
    {
      let polygon = matches!(&spec[0], Expr::String(p) if p == "Polygon");
      let country = (name != "World").then_some(name.as_str());
      // `{prop, {projection, {lat, lon}}}`, or a bare projection name that
      // is centered on the origin.
      let shape = match &spec[1] {
        Expr::List(pc) if pc.len() == 2 => {
          crate::functions::map_projection::country_shape(
            country, &pc[0], &pc[1],
          )
        }
        proj @ Expr::String(_) => {
          crate::functions::map_projection::country_shape(
            country,
            proj,
            &Expr::List(vec![Expr::Integer(0), Expr::Integer(0)].into()),
          )
        }
        _ => None,
      };
      Ok(match shape {
        // "Polygon" is the bare projected polygon, "Shape" the Graphics.
        Some(Expr::FunctionCall { args: ref g, .. })
          if polygon && g.len() == 1 =>
        {
          match &g[0] {
            Expr::List(polys) => {
              let rings: Vec<Expr> = polys
                .iter()
                .filter_map(|p| match p {
                  Expr::FunctionCall { name, args } if name == "Polygon" => {
                    args.first().cloned()
                  }
                  _ => None,
                })
                .collect();
              call("Polygon", vec![Expr::List(rings.into())])
            }
            _ => unevaluated("CountryData", args),
          }
        }
        Some(g) => g,
        None => unevaluated("CountryData", args),
      })
    }
    _ => Ok(unevaluated("CountryData", args)),
  }
}

fn entity(name: &str) -> Expr {
  let name = legacy(name).map_or(name, |c| entity_name(c.canonical));
  call("Entity", vec![str_expr("Country"), str_expr(name)])
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
      Err(_) => match t.parse::<BigInt>() {
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
        "true" => Some(bool_expr(true)),
        "false" => Some(bool_expr(false)),
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

  // A list of inputs is interpreted element by element.
  if let [Expr::List(items)] = applied {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|item| apply_interpreter(domain, std::slice::from_ref(item)))
      .collect();
    return Ok(Expr::List(results?.into()));
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
    func: Box::new(call1("Interpreter", domain.clone())),
    args: applied.to_vec(),
  })
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn language_tables_use_canonical_names() {
    for (lang, countries) in LANGUAGE_COUNTRIES {
      for c in *countries {
        assert_eq!(canonical_name(c), Some(*c), "{lang}: {c}");
      }
    }
  }

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
