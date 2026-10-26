//! `ExampleData[{"Dataset", …}]`: generated datasets.
//!
//! The nesting, keys and schema are the recorded ones (planet names,
//! country and region entities, passenger columns); the values — masses,
//! radii, populations, ages — are generated: physical quantities on a
//! plausible logarithmic scale per unit, tabular columns with the recorded
//! category frequencies, ranges and missing values (see [`tabular`]).

use super::images::cached_opt;
use super::instantiate::{Filler, instantiate};
use super::rng::Rng;
use super::tabular::{generate, parse_columns};
use super::words::Lexicon;
use super::{Collection, Entry};
use crate::functions::*;

pub struct Datasets;

/// A plausible magnitude for a quantity in `unit`: log-uniform over the
/// unit's usual range.
fn magnitude(rng: &mut Rng, unit: &str, integer: bool) -> Expr {
  magnitude_at(rng, unit, integer, false)
}

/// As [`magnitude`]; `nested` values (moons inside planets) are smaller.
fn magnitude_at(
  rng: &mut Rng,
  unit: &str,
  integer: bool,
  nested: bool,
) -> Expr {
  let (lo, hi) = match (unit, nested) {
    ("Kilograms", false) => (23.0, 27.5),
    ("Kilograms", true) => (15.0, 22.9),
    ("Kilometers", false) => (3.3, 4.9),
    ("Kilometers", true) => (0.5, 3.3),
    ("People", _) => (3.0, 7.5),
    _ => (0.0, 3.0),
  };
  let v = 10f64.powf(rng.range(lo, hi));
  if integer {
    Expr::Integer(v.round() as i128)
  } else {
    // Three significant digits, like measured values.
    let digits = 10f64.powi(v.log10().floor() as i32 - 2);
    Expr::Real((v / digits).round() * digits)
  }
}

/// Replace `Sk["Quantity", Real | Integer, unit]` by a generated quantity
/// before the generic instantiation.
fn quantities(sk: &Expr, rng: &mut Rng) -> Expr {
  quantities_at(sk, rng, false)
}

fn quantities_at(sk: &Expr, rng: &mut Rng, nested: bool) -> Expr {
  let quantities = |e: &Expr, rng: &mut Rng| quantities_at(e, rng, nested);
  match sk {
    Expr::FunctionCall { name, args }
      if name == "Sk"
        && matches!(args.first(), Some(Expr::String(s)) if s == "Quantity") =>
    {
      let integer =
        matches!(args.get(1), Some(Expr::Identifier(h)) if h == "Integer");
      let unit = match args.get(2) {
        Some(Expr::String(u)) => u.clone(),
        _ => String::new(),
      };
      call(
        "Quantity",
        vec![magnitude_at(rng, &unit, integer, nested), str_expr(&unit)],
      )
    }
    Expr::FunctionCall { name, args } => {
      call(name, args.iter().map(|a| quantities(a, rng)).collect())
    }
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|a| quantities(a, rng))
        .collect::<Vec<_>>()
        .into(),
    ),
    // Values below a key such as "Moons" belong to smaller bodies.
    Expr::Association(pairs) => Expr::Association(
      pairs
        .iter()
        .map(|(k, v)| {
          let deeper = nested || matches!(k, Expr::String(s) if s == "Moons");
          (k.clone(), quantities_at(v, rng, deeper))
        })
        .collect(),
    ),
    Expr::Rule {
      pattern,
      replacement,
    } => rule_expr(pattern.as_ref().clone(), quantities(replacement, rng)),
    other => other.clone(),
  }
}

/// Country → region entities, from the recorded key lists: each region
/// entity names its country as the last part of its specification.
fn nested_by_country(entry: &Entry, keys: &[Expr]) -> Option<Expr> {
  let (Some(Expr::List(countries)), Some(Expr::List(regions))) =
    (keys.first(), keys.get(1))
  else {
    return None;
  };
  let country_name = |e: &Expr| -> Option<String> {
    let Expr::FunctionCall { args, .. } = e else {
      return None;
    };
    match args.get(1)? {
      Expr::String(s) => Some(s.clone()),
      Expr::List(parts) => match parts.last()? {
        Expr::String(s) => Some(s.clone()),
        _ => None,
      },
      _ => None,
    }
  };
  let mut rng = entry.rng("regions");
  Some(Expr::Association(
    countries
      .iter()
      .map(|c| {
        let name = country_name(c);
        let inner: Vec<(Expr, Expr)> = regions
          .iter()
          .filter(|r| country_name(r) == name)
          .map(|r| {
            (
              r.clone(),
              call(
                "Quantity",
                vec![magnitude(&mut rng, "People", true), str_expr("People")],
              ),
            )
          })
          .collect();
        (c.clone(), Expr::Association(inner))
      })
      .collect(),
  ))
}

fn raw_data(entry: &Entry) -> Option<Expr> {
  let keys: Vec<Expr> = match entry.stat("@Keys") {
    Some(Expr::List(ref k)) => k.iter().cloned().collect(),
    _ => Vec::new(),
  };
  // A list of records: a generated table with the recorded columns.
  if let (Some(Expr::Integer(n)), Some(Expr::List(names))) =
    (keys.first(), keys.get(1))
    && let Some(stats) = entry.stat("@Stats")
  {
    let cols = parse_columns(&stats);
    if cols.len() == names.len() {
      let mut rng = entry.rng("rows");
      let rows =
        generate(&mut rng, &cols, *n as usize, None, &Lexicon::English, false);
      return Some(Expr::List(
        rows
          .into_iter()
          .map(|r| Expr::Association(names.iter().cloned().zip(r).collect()))
          .collect::<Vec<_>>()
          .into(),
      ));
    }
  }
  // Entity-keyed nesting (regions by country).
  if matches!(keys.first(), Some(Expr::List(c)) if c.first().is_some_and(|e| matches!(e, Expr::FunctionCall { name, .. } if name == "Entity")))
  {
    return nested_by_country(entry, &keys);
  }
  // Anything else follows its skeleton.
  let sk = entry.skeleton("RawData")?;
  let mut rng = entry.rng("values");
  let sk = quantities(&sk, &mut rng);
  let lex = Lexicon::English;
  let mut f = Filler {
    rng,
    spec: super::tabular::NumSpec::default(),
    lexicon: &lex,
  };
  Some(instantiate(&sk, &mut f))
}

impl Collection for Datasets {
  fn value(&self, entry: &Entry, property: Option<&str>) -> Option<Expr> {
    let raw =
      || cached_opt(format!("Dataset/{}/raw", entry.name), || raw_data(entry));
    Some(match property {
      None => {
        let d = call("Dataset", vec![raw()?]);
        crate::evaluator::evaluate_expr_to_expr(&d).unwrap_or(d)
      }
      Some("RawData") => raw()?,
      Some("Description") => str_expr(&format!(
        "Generated stand-in for the {} dataset: the keys and structure of the original, \
         with synthetic values.",
        entry.name
      )),
      _ => return None,
    })
  }
}
