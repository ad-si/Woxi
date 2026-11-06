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

static CITIES: std::sync::LazyLock<Vec<City>> =
  std::sync::LazyLock::new(|| {
    serde_json::from_str(include_str!("../../resources/city_data.json"))
      .expect("bundled city data is valid")
  });

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
      assert!(aliases.insert(normalize(&city.name)));
      assert!(!city.entities.is_empty());
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
