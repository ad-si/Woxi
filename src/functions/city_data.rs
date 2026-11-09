//! Bundled city identities and properties backing CityData.
//!
//! Identity matches, labels and coordinates were verified against Wolfram
//! Engine 15.0.0. Population estimates retain the existing rounded table.
//! Property data covers the primary city of each name; secondary identity
//! matches do not borrow that primary city's coordinates or population.

use super::*;
use crate::functions::element_data::make_quantity;

#[derive(serde::Deserialize)]
struct City {
  name: String,
  label: String,
  entities: Vec<[String; 3]>,
  coordinates: [f64; 2],
  population: i64,
}

/// The verified cities of `city_data.json` first, then every other city of
/// at least 100 000 people from GeoNames (`resources/cities.tsv.gz`, see
/// `scripts/gen_city_data.py`).
static CITIES: std::sync::LazyLock<Vec<City>> =
  std::sync::LazyLock::new(|| {
    let mut cities: Vec<City> =
      serde_json::from_str(include_str!("../../resources/city_data.json"))
        .expect("bundled city data is valid");
    use flate2::read::GzDecoder;
    use std::io::Read;
    let compressed = include_bytes!("../../resources/cities.tsv.gz");
    let mut text = String::new();
    GzDecoder::new(&compressed[..])
      .read_to_string(&mut text)
      .expect("cities.tsv.gz is valid gzip");
    let verified = cities.len();
    for line in text.lines().filter(|l| !l.starts_with('#')) {
      let f: Vec<&str> = line.split('\t').collect();
      if f.len() < 7 {
        continue;
      }
      let id = [f[0].to_string(), f[1].to_string(), f[2].to_string()];
      // A verified city keeps its own (Wolfram-checked) identity.
      if cities[..verified].iter().any(|c| {
        c.entities.first().is_some_and(|e| {
          normalize(&e[0]) == normalize(&id[0])
            && normalize(&e[2]) == normalize(&id[2])
        })
      }) {
        continue;
      }
      let (Ok(lat), Ok(lon), Ok(population)) =
        (f[4].parse(), f[5].parse(), f[6].parse())
      else {
        continue;
      };
      cities.push(City {
        name: f[0].to_string(),
        label: if f[3].is_empty() { f[0] } else { f[3] }.to_string(),
        entities: vec![id],
        coordinates: [lat, lon],
        population,
      });
    }
    cities
  });

/// `CityData[{All, country}]`: the country's cities, largest first.
fn cities_of_country(country: &str) -> Vec<Expr> {
  let key = normalize(country);
  let mut found: Vec<(&City, &[String; 3])> = CITIES
    .iter()
    .filter_map(|c| {
      let id = c.entities.first()?;
      (normalize(&id[2]) == key).then_some((c, id))
    })
    .collect();
  found.sort_by_key(|(c, _)| std::cmp::Reverse(c.population));
  found.into_iter().map(|(_, id)| entity(id)).collect()
}

fn normalize(s: &str) -> String {
  s.chars()
    .filter(|c| c.is_alphanumeric())
    .flat_map(char::to_lowercase)
    .collect()
}

/// A name, a {name,country}/{name,region,country} query, or a city entity.
fn city_spec(spec: &Expr) -> Option<Vec<String>> {
  match spec {
    Expr::String(s) => Some(vec![s.clone()]),
    Expr::List(items) if (2..=3).contains(&items.len()) => items
      .iter()
      .map(|e| match e {
        Expr::String(s) => Some(s.clone()),
        _ => None,
      })
      .collect(),
    Expr::FunctionCall { name, args }
      if name == "Entity"
        && args.len() == 2
        && matches!(&args[0], Expr::String(s) if s == "City") =>
    {
      city_spec(&args[1])
    }
    _ => None,
  }
}

fn matches(spec: &Expr) -> Option<Vec<(&City, &[String; 3])>> {
  let parts = city_spec(spec)?;
  let mut found = Vec::new();
  for city in CITIES.iter() {
    let alias = normalize(&parts[0]) == normalize(&city.name)
      || normalize(&parts[0]) == normalize(&city.label);
    for id in &city.entities {
      let name = alias || normalize(&parts[0]) == normalize(&id[0]);
      let matched = match parts.as_slice() {
        [_] => name,
        [_, country] => name && normalize(country) == normalize(&id[2]),
        [_, region, country] => {
          name
            && normalize(region) == normalize(&id[1])
            && normalize(country) == normalize(&id[2])
        }
        _ => false,
      };
      if matched && !found.iter().any(|(_, existing)| *existing == id) {
        found.push((city, id));
      }
    }
  }
  Some(found)
}

fn entity(id: &[String; 3]) -> Expr {
  call(
    "Entity",
    vec![
      str_expr("City"),
      Expr::List(id.iter().map(|s| str_expr(s)).collect()),
    ],
  )
}

fn missing() -> Expr {
  call("Missing", vec![str_expr("NotAvailable")])
}

fn property(city: &City, id: &[String; 3], prop: &str) -> Expr {
  if prop == "Country" {
    return call("Entity", vec![str_expr("Country"), str_expr(&id[2])]);
  }
  if city.entities.first() != Some(id) {
    return missing();
  }
  let [lat, lon] = city.coordinates;
  match prop {
    "Coordinates" => Expr::List(vec![Expr::Real(lat), Expr::Real(lon)].into()),
    "Latitude" => make_quantity(Expr::Real(lat), "AngularDegrees"),
    "Longitude" => make_quantity(Expr::Real(lon), "AngularDegrees"),
    "Population" => {
      make_quantity(Expr::Integer(i128::from(city.population)), "People")
    }
    "Name" => str_expr(&city.label),
    "Region" => str_expr(&id[1]),
    _ => call("Missing", vec![str_expr("NotAvailable"), str_expr(prop)]),
  }
}

pub fn city_data_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unchanged = || Ok(call("CityData", args.to_vec()));
  if let [Expr::List(spec)] = args
    && let [Expr::Identifier(all), country] = spec.as_slice()
    && all == "All"
  {
    let name = match country {
      Expr::String(s) => s.clone(),
      Expr::FunctionCall { name, args: ea }
        if name == "Entity" && ea.len() == 2 =>
      {
        match &ea[1] {
          Expr::String(s) => s.clone(),
          _ => return unchanged(),
        }
      }
      _ => return unchanged(),
    };
    let name = crate::functions::country_data::wl_entity_name(&name)
      .map_or(name, str::to_string);
    return Ok(Expr::List(cities_of_country(&name).into()));
  }
  match args {
    [spec] => {
      let Some(found) = matches(spec) else {
        return unchanged();
      };
      let mut cities: Vec<Expr> =
        found.iter().map(|(_, id)| entity(id)).collect();
      // String and two-part queries are name searches. A fully qualified
      // identity resolves directly when it names exactly one city.
      if cities.len() == 1
        && !matches!(spec, Expr::String(_))
        && !matches!(spec, Expr::List(items) if items.len() == 2)
      {
        Ok(cities.remove(0))
      } else {
        Ok(Expr::List(cities.into()))
      }
    }
    [spec, Expr::String(prop)] => match matches(spec) {
      Some(found) => Ok(
        found
          .first()
          .map_or_else(missing, |(c, id)| property(c, id, prop)),
      ),
      None => Ok(if prop == "Coordinates" {
        Expr::List(vec![missing(), missing()].into())
      } else {
        missing()
      }),
    },
    _ => unchanged(),
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn bundled_identities_and_coordinates_are_valid() {
    let mut aliases = std::collections::HashSet::new();
    assert!(!CITIES.is_empty());
    for city in CITIES.iter() {
      // Names repeat across countries; a full identity never does.
      assert!(aliases.insert(city.entities[0].join("|")), "{}", city.name);
      assert_ne!(
        city.entities.len(),
        0,
        "{} has no entity identifiers",
        city.name
      );
      assert!(
        city
          .entities
          .iter()
          .all(|id| id.iter().all(|s| !s.is_empty()))
      );
      assert!((-90.0..=90.0).contains(&city.coordinates[0]));
      assert!((-180.0..=180.0).contains(&city.coordinates[1]));
    }
  }
}
