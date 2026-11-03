//! Minimal built-in city knowledge base backing `CityData`.
//!
//! Woxi has no connection to the Wolfram Knowledgebase, so this bundles a
//! static table of major world cities (coordinates in degrees, population in
//! people, rounded). It supports the common forms:
//!
//!   * `CityData[name]` — `Entity["City", {name, region, country}]`
//!   * `CityData[name, "Coordinates"]` — `{latitude, longitude}`
//!   * `CityData[name, "Latitude" | "Longitude" | "Population" | "Country" |
//!     "Region" | "Name"]`
//!
//! `name` may be a string, or a `{name, region, country}` list / the matching
//! `Entity["City", …]`.

use super::*;
use crate::functions::element_data::make_quantity;

struct City {
  name: &'static str,
  region: &'static str,
  country: &'static str,
  lat: f64,
  lon: f64,
  population: i64,
}

macro_rules! c {
  ($n:expr, $r:expr, $c:expr, $la:expr, $lo:expr, $p:expr) => {
    City {
      name: $n,
      region: $r,
      country: $c,
      lat: $la,
      lon: $lo,
      population: $p,
    }
  };
}

#[rustfmt::skip]
static CITIES: &[City] = &[
  c!("Amsterdam", "NoordHolland", "Netherlands", 52.3667, 4.9, 842_000),
  c!("Athens", "Attica", "Greece", 37.9667, 23.7167, 643_000),
  c!("Barcelona", "Catalonia", "Spain", 41.3833, 2.1833, 1_620_000),
  c!("Belgrade", "Belgrade", "Serbia", 44.8186, 20.468, 1_197_000),
  c!("Berlin", "Berlin", "Germany", 52.5167, 13.4, 3_645_000),
  c!("Bern", "Bern", "Switzerland", 46.9167, 7.4667, 134_000),
  c!("Brussels", "BrusselsCapital", "Belgium", 50.85, 4.35, 1_209_000),
  c!("Bucharest", "Bucharest", "Romania", 44.4333, 26.1, 1_883_000),
  c!("Budapest", "Budapest", "Hungary", 47.5, 19.0833, 1_752_000),
  c!("Copenhagen", "Hovedstaden", "Denmark", 55.6667, 12.5833, 602_000),
  c!("Dublin", "Leinster", "Ireland", 53.3333, -6.25, 544_000),
  c!("Edinburgh", "Scotland", "United Kingdom", 55.9533, -3.1883, 507_000),
  c!("Frankfurt", "Hessen", "Germany", 50.1167, 8.6833, 753_000),
  c!("Hamburg", "Hamburg", "Germany", 53.55, 10.0, 1_841_000),
  c!("Helsinki", "Uusimaa", "Finland", 60.1756, 24.9342, 656_000),
  c!("Kyiv", "Kyiv", "Ukraine", 50.4333, 30.5167, 2_884_000),
  c!("Lisbon", "Lisbon", "Portugal", 38.7167, -9.1333, 545_000),
  c!("London", "England", "United Kingdom", 51.5, -0.1167, 8_982_000),
  c!("Madrid", "Madrid", "Spain", 40.4, -3.6833, 3_223_000),
  c!("Milan", "Lombardy", "Italy", 45.4667, 9.2, 1_372_000),
  c!("Moscow", "Moscow", "Russia", 55.75, 37.6167, 12_506_000),
  c!("Munich", "Bavaria", "Germany", 48.1333, 11.5667, 1_472_000),
  c!("Oslo", "Oslo", "Norway", 59.9167, 10.75, 697_000),
  c!("Paris", "IleDeFrance", "France", 48.8667, 2.3333, 2_161_000),
  c!("Prague", "Prague", "Czech Republic", 50.0833, 14.4667, 1_309_000),
  c!("Rome", "Lazio", "Italy", 41.9, 12.4833, 2_873_000),
  c!("Sofia", "Sofia", "Bulgaria", 42.6833, 23.3167, 1_242_000),
  c!("Stockholm", "Stockholm", "Sweden", 59.3333, 18.05, 975_000),
  c!("Vienna", "Vienna", "Austria", 48.2, 16.3667, 1_911_000),
  c!("Warsaw", "Masovia", "Poland", 52.25, 21.0, 1_794_000),
  c!("Zurich", "Zurich", "Switzerland", 47.3667, 8.55, 421_000),
  c!("Cairo", "Cairo", "Egypt", 30.05, 31.25, 9_540_000),
  c!("Johannesburg", "Gauteng", "South Africa", -26.2, 28.0833, 5_635_000),
  c!("Lagos", "Lagos", "Nigeria", 6.45, 3.4, 14_368_000),
  c!("Nairobi", "Nairobi", "Kenya", -1.2833, 36.8167, 4_397_000),
  c!("Bangkok", "Bangkok", "Thailand", 13.75, 100.5167, 10_539_000),
  c!("Beijing", "Beijing", "China", 39.9167, 116.3833, 21_540_000),
  c!("Delhi", "Delhi", "India", 28.6667, 77.2167, 16_788_000),
  c!("Dubai", "Dubai", "United Arab Emirates", 25.2667, 55.3, 3_331_000),
  c!("Hong Kong", "HongKong", "China", 22.3, 114.2, 7_482_000),
  c!("Istanbul", "Istanbul", "Turkey", 41.0167, 28.9667, 15_462_000),
  c!("Jakarta", "Jakarta", "Indonesia", -6.1744, 106.8294, 10_562_000),
  c!("Mumbai", "Maharashtra", "India", 19.0167, 72.85, 12_442_000),
  c!("Seoul", "Seoul", "South Korea", 37.5667, 126.9833, 9_776_000),
  c!("Shanghai", "Shanghai", "China", 31.2, 121.5, 24_870_000),
  c!("Singapore", "Singapore", "Singapore", 1.2931, 103.8558, 5_454_000),
  c!("Tehran", "Tehran", "Iran", 35.6944, 51.4215, 8_694_000),
  c!("Tokyo", "Tokyo", "Japan", 35.685, 139.7514, 13_960_000),
  c!("Melbourne", "Victoria", "Australia", -37.8167, 144.9667, 5_078_000),
  c!("Sydney", "NewSouthWales", "Australia", -33.8833, 151.2167, 5_312_000),
  c!("Auckland", "Auckland", "New Zealand", -36.8667, 174.7667, 1_657_000),
  c!("Buenos Aires", "BuenosAires", "Argentina", -34.5875, -58.6725, 3_121_000),
  c!("Lima", "Lima", "Peru", -12.05, -77.05, 9_751_000),
  c!("Mexico City", "MexicoCity", "Mexico", 19.4333, -99.1333, 9_209_000),
  c!("Rio de Janeiro", "RioDeJaneiro", "Brazil", -22.9, -43.2333, 6_748_000),
  c!("Santiago", "Santiago", "Chile", -33.45, -70.6667, 5_614_000),
  c!("Sao Paulo", "SaoPaulo", "Brazil", -23.5333, -46.6167, 12_325_000),
  c!("Chicago", "Illinois", "United States", 41.85, -87.65, 2_746_000),
  c!("Los Angeles", "California", "United States", 34.05, -118.25, 3_898_000),
  c!("Miami", "Florida", "United States", 25.7739, -80.1939, 442_000),
  c!("New York", "NewYork", "United States", 40.7167, -74.0, 8_336_000),
  c!("San Francisco", "California", "United States", 37.775, -122.4183, 873_000),
  c!("Seattle", "Washington", "United States", 47.6097, -122.3331, 737_000),
  c!("Washington", "DistrictOfColumbia", "United States", 38.9047, -77.0164, 689_000),
  c!("Montreal", "Quebec", "Canada", 45.5, -73.5833, 1_762_000),
  c!("Toronto", "Ontario", "Canada", 43.7, -79.4, 2_794_000),
  c!("Vancouver", "BritishColumbia", "Canada", 49.25, -123.1333, 662_000),
];

/// Normalize a city name for matching: lowercase ASCII alphanumerics only.
fn normalize(name: &str) -> String {
  name
    .chars()
    .filter(char::is_ascii_alphanumeric)
    .map(|c| c.to_ascii_lowercase())
    .collect()
}

fn find_city(
  name: &str,
  region: Option<&str>,
  country: Option<&str>,
) -> Option<&'static City> {
  let key = normalize(name);
  CITIES.iter().find(|c| {
    normalize(c.name) == key
      && region.is_none_or(|r| normalize(r) == normalize(c.region))
      && country.is_none_or(|k| normalize(k) == normalize(c.country))
  })
}

/// Resolve the first argument — a name, `{name, region, country}` or
/// `Entity["City", …]` — to a city in the table.
fn resolve(spec: &Expr) -> Option<&'static City> {
  let strings = |items: &[Expr]| -> Option<Vec<String>> {
    items
      .iter()
      .map(|e| match e {
        Expr::String(s) => Some(s.clone()),
        _ => None,
      })
      .collect()
  };
  match spec {
    Expr::String(s) => find_city(s, None, None),
    Expr::List(items) => {
      let s = strings(items)?;
      match s.as_slice() {
        [n] => find_city(n, None, None),
        [n, r, k] => find_city(n, Some(r), Some(k)),
        [n, k] => find_city(n, None, Some(k)),
        _ => None,
      }
    }
    Expr::FunctionCall { name, args }
      if name == "Entity"
        && args.len() == 2
        && matches!(&args[0], Expr::String(t) if t == "City") =>
    {
      resolve(&args[1])
    }
    _ => None,
  }
}

fn entity(city: &City) -> Expr {
  call(
    "Entity",
    vec![
      str_expr("City"),
      Expr::List(
        vec![
          str_expr(city.name),
          str_expr(city.region),
          str_expr(city.country),
        ]
        .into(),
      ),
    ],
  )
}

/// A property of a city from the table.
fn city_property(city: &City, property: &str) -> Expr {
  match property {
    "Coordinates" => {
      Expr::List(vec![Expr::Real(city.lat), Expr::Real(city.lon)].into())
    }
    "Latitude" => Expr::Real(city.lat),
    "Longitude" => Expr::Real(city.lon),
    "Population" => {
      make_quantity(Expr::Integer(city.population as i128), "People")
    }
    "Name" => str_expr(city.name),
    "Country" => {
      call("Entity", vec![str_expr("Country"), str_expr(city.country)])
    }
    "Region" => str_expr(city.region),
    _ => call(
      "Missing",
      vec![str_expr("NotAvailable"), str_expr(property)],
    ),
  }
}

/// `CityData[city]` and `CityData[city, property]`.
pub fn city_data_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(call("CityData", args.to_vec()));
  match args {
    [spec] => match resolve(spec) {
      Some(c) => Ok(entity(c)),
      None => unevaluated(),
    },
    [spec, Expr::String(prop)] => match resolve(spec) {
      Some(c) => Ok(city_property(c, prop)),
      None => unevaluated(),
    },
    _ => unevaluated(),
  }
}
