//! `WikidataData` and `ExternalIdentifier` support.
//!
//! `ExternalIdentifier` is a purely symbolic wrapper for an identifier in an
//! external system (`ExternalIdentifier["WikidataID", "Q405", <|…|>]`).
//! `WikidataData` looks up the values of a Wikidata property for a Wikidata
//! item through the public wikidata.org API (via `curl`, like the other
//! URL-import paths) and translates the Wikidata datatypes to Wolfram
//! Language expressions:
//!
//! - quantities → `Quantity[magnitude, unit]` (Wikidata unit items are
//!   mapped to Wolfram unit names through their English label)
//! - Commons media files and URLs → `URL["…"]`
//! - items / properties → `ExternalIdentifier["WikidataID", id, <|…|>]`
//! - points in time → `DateObject[…]` at the stated precision
//! - globe coordinates → `GeoPosition[{lat, lon}]`
//! - strings, external ids and monolingual texts → plain strings

use crate::InterpreterError;
use crate::syntax::Expr;

/// `ExternalIdentifier[type, id]` / `ExternalIdentifier[type, id, assoc]` —
/// inert symbolic construct; the arguments are kept exactly as given.
pub fn external_identifier_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  Ok(Expr::FunctionCall {
    name: "ExternalIdentifier".to_string(),
    args: args.to_vec().into(),
  })
}

/// An item or property specification: a single identifier or a list of them.
enum Spec {
  Single(String),
  Multiple(Vec<String>),
}

/// `WikidataData[itemspec, propspec]` — values of a Wikidata property for a
/// Wikidata item. List specifications map to the corresponding array of
/// results (one list of values per item/property combination).
pub fn wikidata_data_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "WikidataData".to_string(),
      args: args.to_vec().into(),
    })
  };
  if args.len() != 2 {
    return unevaluated();
  }
  let (Some(items), Some(props)) =
    (resolve_spec(&args[0], 'Q'), resolve_spec(&args[1], 'P'))
  else {
    return unevaluated();
  };

  #[cfg(target_arch = "wasm32")]
  {
    let _ = (items, props);
    Err(InterpreterError::EvaluationError(
      "WikidataData: network access is not available in the browser".into(),
    ))
  }
  #[cfg(not(target_arch = "wasm32"))]
  {
    let result_for_item = |item: &str| -> Result<Expr, InterpreterError> {
      match &props {
        Spec::Single(p) => property_values(item, p),
        Spec::Multiple(ps) => Ok(Expr::List(
          ps.iter()
            .map(|p| property_values(item, p))
            .collect::<Result<Vec<_>, _>>()?
            .into(),
        )),
      }
    };
    match &items {
      Spec::Single(item) => result_for_item(item),
      Spec::Multiple(is) => Ok(Expr::List(
        is.iter()
          .map(|i| result_for_item(i))
          .collect::<Result<Vec<_>, _>>()?
          .into(),
      )),
    }
  }
}

/// Resolve an item/property specification to Wikidata identifiers of the
/// expected kind (`Q…` items or `P…` properties). Accepts
/// `ExternalIdentifier["WikidataID", id, …]`, raw identifier strings,
/// wikidata.org URLs (plain or wrapped in `URL[…]`) and lists thereof.
fn resolve_spec(expr: &Expr, kind: char) -> Option<Spec> {
  match expr {
    Expr::List(items) => {
      let ids = items
        .iter()
        .map(|e| resolve_id(e, kind))
        .collect::<Option<Vec<_>>>()?;
      Some(Spec::Multiple(ids))
    }
    other => Some(Spec::Single(resolve_id(other, kind)?)),
  }
}

/// Resolve a single item/property specification to its Wikidata identifier.
fn resolve_id(expr: &Expr, kind: char) -> Option<String> {
  let id = match expr {
    Expr::FunctionCall { name, args }
      if name == "ExternalIdentifier"
        && args.len() >= 2
        && matches!(&args[0], Expr::String(t) if t == "WikidataID") =>
    {
      match &args[1] {
        Expr::String(id) => id.clone(),
        _ => return None,
      }
    }
    Expr::FunctionCall { name, args }
      if name == "URL"
        && args.len() == 1
        && matches!(&args[0], Expr::String(_)) =>
    {
      match &args[0] {
        Expr::String(url) => id_from_url(url),
        _ => return None,
      }
    }
    Expr::String(s) if s.contains("://") => id_from_url(s),
    Expr::String(s) => s.clone(),
    _ => return None,
  };
  if id.starts_with(kind)
    && id.len() > 1
    && id[1..].bytes().all(|b| b.is_ascii_digit())
  {
    Some(id)
  } else {
    None
  }
}

/// Extract the identifier from a wikidata.org URL such as
/// `https://www.wikidata.org/wiki/Q405` or
/// `http://www.wikidata.org/entity/P31` (also `…/wiki/Property:P31`).
fn id_from_url(url: &str) -> String {
  let last = url.rsplit('/').next().unwrap_or(url);
  last.strip_prefix("Property:").unwrap_or(last).to_string()
}

// ─── Network-backed lookup (CLI only — uses curl) ──────────────────────────

#[cfg(not(target_arch = "wasm32"))]
use serde_json::Value;

/// User agent for wikidata.org requests, per the Wikimedia API etiquette.
#[cfg(not(target_arch = "wasm32"))]
const USER_AGENT: &str = concat!(
  "woxi/",
  env!("CARGO_PKG_VERSION"),
  " (https://github.com/ad-si/Woxi)"
);

/// Fetch a wikidata.org API URL and parse the JSON response. Responses are
/// cached for the lifetime of the interpreter so repeated lookups (and the
/// follow-up label queries) do not hit the network again.
#[cfg(not(target_arch = "wasm32"))]
fn fetch_json(url: &str) -> Result<Value, InterpreterError> {
  use std::cell::RefCell;
  use std::collections::HashMap;
  thread_local! {
    static CACHE: RefCell<HashMap<String, String>> = RefCell::new(HashMap::new());
  }
  let cached = CACHE.with(|c| c.borrow().get(url).cloned());
  let body = match cached {
    Some(body) => body,
    None => {
      let output = std::process::Command::new("curl")
        .args([
          "-fsSL",
          "--max-time",
          "30",
          "--retry",
          "3",
          "--retry-all-errors",
          "-A",
          USER_AGENT,
          url,
        ])
        .output()
        .map_err(|e| {
          InterpreterError::EvaluationError(format!(
            "WikidataData: failed to run curl: {}",
            e
          ))
        })?;
      if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(InterpreterError::EvaluationError(format!(
          "WikidataData: failed to download \"{}\": {}",
          url,
          stderr.trim()
        )));
      }
      let body = String::from_utf8_lossy(&output.stdout).into_owned();
      CACHE.with(|c| c.borrow_mut().insert(url.to_string(), body.clone()));
      body
    }
  };
  serde_json::from_str(&body).map_err(|e| {
    InterpreterError::EvaluationError(format!(
      "WikidataData: invalid API response: {}",
      e
    ))
  })
}

/// The values of one property for one item, as a Wolfram Language list.
#[cfg(not(target_arch = "wasm32"))]
fn property_values(
  item: &str,
  property: &str,
) -> Result<Expr, InterpreterError> {
  let url = format!(
    "https://www.wikidata.org/w/api.php?action=wbgetclaims&entity={}&property={}&format=json",
    item, property
  );
  let json = fetch_json(&url)?;
  let empty = Vec::new();
  let claims = json["claims"][property].as_array().unwrap_or(&empty);

  // "Truthy" statement selection, matching Wikidata's wdt: semantics used by
  // WikidataData's SPARQL queries: deprecated statements are dropped, and if
  // any statement is ranked preferred only the preferred ones count.
  let ranked: Vec<&Value> = claims
    .iter()
    .filter(|c| c["rank"].as_str() != Some("deprecated"))
    .collect();
  let has_preferred = ranked
    .iter()
    .any(|c| c["rank"].as_str() == Some("preferred"));
  let truthy: Vec<&Value> = ranked
    .into_iter()
    .filter(|c| !has_preferred || c["rank"].as_str() == Some("preferred"))
    .collect();

  // Entity labels (for item-valued statements and quantity units) are
  // resolved in a single batched follow-up query.
  let mut label_ids: Vec<String> = Vec::new();
  for claim in &truthy {
    let snak = &claim["mainsnak"];
    match snak["datatype"].as_str() {
      Some("wikibase-item") | Some("wikibase-property") => {
        if let Some(id) = snak["datavalue"]["value"]["id"].as_str() {
          label_ids.push(id.to_string());
        }
      }
      Some("quantity") => {
        if let Some(id) =
          unit_entity_id(snak["datavalue"]["value"]["unit"].as_str())
        {
          label_ids.push(id);
        }
      }
      _ => {}
    }
  }
  let labels = fetch_labels(&label_ids)?;

  let values = truthy
    .iter()
    .map(|claim| snak_to_expr(&claim["mainsnak"], &labels))
    .collect::<Result<Vec<_>, _>>()?;
  Ok(Expr::List(values.into()))
}

/// English label and description for a batch of entity ids.
#[cfg(not(target_arch = "wasm32"))]
fn fetch_labels(
  ids: &[String],
) -> Result<
  std::collections::HashMap<String, (Option<String>, Option<String>)>,
  InterpreterError,
> {
  let mut labels = std::collections::HashMap::new();
  if ids.is_empty() {
    return Ok(labels);
  }
  let mut unique: Vec<&String> = ids.iter().collect();
  unique.sort();
  unique.dedup();
  // wbgetentities accepts at most 50 ids per request.
  for chunk in unique.chunks(50) {
    let joined = chunk
      .iter()
      .map(|s| s.as_str())
      .collect::<Vec<_>>()
      .join("|");
    let url = format!(
      "https://www.wikidata.org/w/api.php?action=wbgetentities&ids={}&props=labels%7Cdescriptions&languages=en&format=json",
      joined
    );
    let json = fetch_json(&url)?;
    if let Some(entities) = json["entities"].as_object() {
      for (id, entity) in entities {
        let label = entity["labels"]["en"]["value"].as_str().map(String::from);
        let description = entity["descriptions"]["en"]["value"]
          .as_str()
          .map(String::from);
        labels.insert(id.clone(), (label, description));
      }
    }
  }
  Ok(labels)
}

/// The entity id of a quantity's unit, or `None` for dimensionless
/// quantities (whose unit is the literal string `"1"`).
#[cfg(not(target_arch = "wasm32"))]
fn unit_entity_id(unit: Option<&str>) -> Option<String> {
  let unit = unit?;
  if unit == "1" {
    return None;
  }
  let id = unit.rsplit('/').next()?;
  if id.starts_with('Q') {
    Some(id.to_string())
  } else {
    None
  }
}

/// Translate one Wikidata snak (statement value) to a Wolfram Language
/// expression according to its datatype.
#[cfg(not(target_arch = "wasm32"))]
fn snak_to_expr(
  snak: &Value,
  labels: &std::collections::HashMap<String, (Option<String>, Option<String>)>,
) -> Result<Expr, InterpreterError> {
  match snak["snaktype"].as_str() {
    Some("somevalue") => {
      return Ok(missing("Unknown"));
    }
    Some("novalue") => {
      return Ok(missing("NotAvailable"));
    }
    _ => {}
  }
  let value = &snak["datavalue"]["value"];
  match snak["datatype"].as_str() {
    Some("quantity") => {
      let magnitude = parse_amount(value["amount"].as_str().unwrap_or("0"));
      match unit_entity_id(value["unit"].as_str()) {
        None => Ok(magnitude),
        Some(id) => {
          let unit = labels
            .get(&id)
            .and_then(|(label, _)| label.clone())
            .ok_or_else(|| {
              InterpreterError::EvaluationError(format!(
                "WikidataData: missing label for unit entity {}",
                id
              ))
            })?;
          evaluate(Expr::FunctionCall {
            name: "Quantity".to_string(),
            args: vec![magnitude, unit_to_expr(&unit)].into(),
          })
        }
      }
    }
    Some("commonsMedia") => {
      let name = value.as_str().unwrap_or_default();
      Ok(url_expr(format!(
        "http://commons.wikimedia.org/wiki/Special:FilePath/{}",
        percent_encode(name)
      )))
    }
    Some("url") => Ok(url_expr(value.as_str().unwrap_or_default().to_string())),
    Some("wikibase-item") | Some("wikibase-property") => {
      let id = value["id"].as_str().unwrap_or_default().to_string();
      let mut meta: Vec<(Expr, Expr)> = Vec::new();
      if let Some((label, description)) = labels.get(&id) {
        if let Some(label) = label {
          meta.push((
            Expr::String("Label".to_string()),
            Expr::String(label.clone()),
          ));
        }
        if let Some(description) = description {
          meta.push((
            Expr::String("Description".to_string()),
            Expr::String(description.clone()),
          ));
        }
      }
      Ok(Expr::FunctionCall {
        name: "ExternalIdentifier".to_string(),
        args: vec![
          Expr::String("WikidataID".to_string()),
          Expr::String(id),
          Expr::Association(meta),
        ]
        .into(),
      })
    }
    Some("time") => time_to_date_object(
      value["time"].as_str().unwrap_or_default(),
      value["precision"].as_i64().unwrap_or(11),
    ),
    Some("globe-coordinate") => {
      let lat = value["latitude"].as_f64().unwrap_or(0.0);
      let lon = value["longitude"].as_f64().unwrap_or(0.0);
      evaluate(Expr::FunctionCall {
        name: "GeoPosition".to_string(),
        args: vec![Expr::List(vec![Expr::Real(lat), Expr::Real(lon)].into())]
          .into(),
      })
    }
    Some("monolingualtext") => Ok(Expr::String(
      value["text"].as_str().unwrap_or_default().to_string(),
    )),
    // string, external-id, math, musical-notation, … — all plain strings.
    _ => Ok(Expr::String(value.as_str().unwrap_or_default().to_string())),
  }
}

/// `Missing["reason"]`.
#[cfg(not(target_arch = "wasm32"))]
fn missing(reason: &str) -> Expr {
  Expr::FunctionCall {
    name: "Missing".to_string(),
    args: vec![Expr::String(reason.to_string())].into(),
  }
}

/// `URL["…"]`.
#[cfg(not(target_arch = "wasm32"))]
fn url_expr(url: String) -> Expr {
  Expr::FunctionCall {
    name: "URL".to_string(),
    args: vec![Expr::String(url)].into(),
  }
}

/// Run an expression through the evaluator so constructors like `Quantity`,
/// `DateObject` and `GeoPosition` canonicalize the same way as typed input.
#[cfg(not(target_arch = "wasm32"))]
fn evaluate(expr: Expr) -> Result<Expr, InterpreterError> {
  crate::evaluator::evaluate_expr_to_expr(&expr)
}

/// Parse a Wikidata quantity amount (`"+73.4767"`, `"-1"`, …). Amounts
/// without a fractional part stay exact integers.
#[cfg(not(target_arch = "wasm32"))]
fn parse_amount(amount: &str) -> Expr {
  let s = amount.strip_prefix('+').unwrap_or(amount);
  if !s.contains(['.', 'e', 'E'])
    && let Ok(n) = s.parse::<i128>()
  {
    return Expr::Integer(n);
  }
  match s.parse::<f64>() {
    Ok(f) => Expr::Real(f),
    Err(_) => Expr::String(amount.to_string()),
  }
}

/// Percent-encode a Commons file name the way Wikidata's SPARQL endpoint
/// renders `Special:FilePath` URLs: RFC 3986 unreserved characters plus the
/// sub-delimiters that stay literal in IRIs are kept, everything else
/// (notably spaces) is escaped.
#[cfg(not(target_arch = "wasm32"))]
fn percent_encode(name: &str) -> String {
  let mut out = String::with_capacity(name.len());
  for byte in name.bytes() {
    match byte {
      b'A'..=b'Z'
      | b'a'..=b'z'
      | b'0'..=b'9'
      | b'-'
      | b'.'
      | b'_'
      | b'~'
      | b'!'
      | b'$'
      | b'&'
      | b'\''
      | b'('
      | b')'
      | b'*'
      | b'+'
      | b','
      | b';'
      | b'=' => out.push(byte as char),
      _ => out.push_str(&format!("%{:02X}", byte)),
    }
  }
  out
}

/// `DateObject` for a Wikidata point in time (`"+2020-09-05T00:00:00Z"`) at
/// the stated precision (9 = year, 10 = month, 11 = day, 12–14 = hour,
/// minute, second; coarser precisions fall back to the year).
#[cfg(not(target_arch = "wasm32"))]
fn time_to_date_object(
  time: &str,
  precision: i64,
) -> Result<Expr, InterpreterError> {
  let (sign, rest) = match time.as_bytes().first() {
    Some(b'-') => (-1i64, &time[1..]),
    Some(b'+') => (1, &time[1..]),
    _ => (1, time),
  };
  let mut date_part = rest.split('T');
  let ymd: Vec<i64> = date_part
    .next()
    .unwrap_or_default()
    .split('-')
    .filter_map(|p| p.parse().ok())
    .collect();
  let hms: Vec<i64> = date_part
    .next()
    .unwrap_or_default()
    .trim_end_matches('Z')
    .split(':')
    .filter_map(|p| p.parse().ok())
    .collect();
  let (year, month, day) = (
    sign * ymd.first().copied().unwrap_or(0),
    ymd.get(1).copied().unwrap_or(1),
    ymd.get(2).copied().unwrap_or(1),
  );
  let mut parts = vec![Expr::Integer(year as i128)];
  if precision >= 10 {
    parts.push(Expr::Integer(month as i128));
  }
  if precision >= 11 {
    parts.push(Expr::Integer(day as i128));
  }
  for (i, part) in hms.iter().enumerate() {
    // hour at precision 12, minute at 13, second at 14
    if precision as usize >= 12 + i {
      parts.push(Expr::Integer(*part as i128));
    }
  }
  evaluate(Expr::FunctionCall {
    name: "DateObject".to_string(),
    args: vec![Expr::List(parts.into())].into(),
  })
}

// ─── Wikidata unit label → Wolfram unit name ────────────────────────────────

/// Map the English label of a Wikidata unit item to a Wolfram Language unit
/// specification. Simple units become pluralized CamelCase names
/// ("yottagram" → "Yottagrams", "degree Celsius" → "DegreesCelsius") and
/// "per" separates numerator and denominator into a unit quotient
/// ("metre per second" → "Meters"/"Seconds").
#[cfg(not(target_arch = "wasm32"))]
fn unit_to_expr(label: &str) -> Expr {
  match label.split_once(" per ") {
    Some((num, den)) => Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(Expr::String(unit_name(num))),
      right: Box::new(Expr::String(unit_name(den))),
    },
    None => Expr::String(unit_name(label)),
  }
}

/// Pluralized CamelCase Wolfram unit name for a (quotient-free) unit label.
#[cfg(not(target_arch = "wasm32"))]
fn unit_name(label: &str) -> String {
  let words: Vec<String> = label
    .split([' ', '-'])
    .filter(|w| !w.is_empty())
    .map(americanize)
    .collect();
  if words.is_empty() {
    return String::new();
  }
  // Temperature-style labels pluralize the leading "degree", not the scale
  // name: "degree Celsius" → "DegreesCelsius".
  if words.len() > 1 && words[0].eq_ignore_ascii_case("degree") {
    let rest: String = words[1..].iter().map(|w| capitalize(w)).collect();
    return format!("Degrees{}", rest);
  }
  // Powers keep the modifier as a prefix: "square kilometre" →
  // "SquareKilometers", "cubic metre" → "CubicMeters".
  let mut name = String::new();
  for (i, word) in words.iter().enumerate() {
    if i == words.len() - 1 {
      name.push_str(&capitalize(&pluralize(word)));
    } else {
      name.push_str(&capitalize(word));
    }
  }
  name
}

/// Rewrite British unit spellings to the American forms Wolfram uses
/// ("kilometre" → "kilometer", "litre" → "liter").
#[cfg(not(target_arch = "wasm32"))]
fn americanize(word: &str) -> String {
  for (from, to) in [("metre", "meter"), ("litre", "liter")] {
    if let Some(stem) = word.strip_suffix(from) {
      return format!("{}{}", stem, to);
    }
    // Also handle already-plural British spellings ("metres").
    let plural = format!("{}s", from);
    if let Some(stem) = word.strip_suffix(plural.as_str()) {
      return format!("{}{}s", stem, to);
    }
  }
  word.to_string()
}

/// English pluralization for unit words. Irregular unit names Wolfram uses
/// are special-cased; invariant words (ending in "s", "z" or "x", like
/// "hertz" or "siemens") stay unchanged.
#[cfg(not(target_arch = "wasm32"))]
fn pluralize(word: &str) -> String {
  match word {
    "foot" => return "feet".to_string(),
    "inch" => return "inches".to_string(),
    "tonne" => return "metricTons".to_string(),
    _ => {}
  }
  match word.as_bytes().last() {
    Some(b's') | Some(b'z') | Some(b'x') => word.to_string(),
    Some(b'y')
      if word.len() > 1
        && !matches!(
          word.as_bytes()[word.len() - 2],
          b'a' | b'e' | b'i' | b'o' | b'u'
        ) =>
    {
      format!("{}ies", &word[..word.len() - 1])
    }
    _ => format!("{}s", word),
  }
}

/// Uppercase the first letter of a word.
#[cfg(not(target_arch = "wasm32"))]
fn capitalize(word: &str) -> String {
  let mut chars = word.chars();
  match chars.next() {
    Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    None => String::new(),
  }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
  use super::*;
  use crate::syntax::expr_to_string;

  #[test]
  fn unit_names_pluralize_and_camel_case() {
    assert_eq!(unit_name("yottagram"), "Yottagrams");
    assert_eq!(unit_name("kilogram"), "Kilograms");
    assert_eq!(unit_name("kilometre"), "Kilometers");
    assert_eq!(unit_name("litre"), "Liters");
    assert_eq!(unit_name("second"), "Seconds");
    assert_eq!(unit_name("astronomical unit"), "AstronomicalUnits");
    assert_eq!(unit_name("square kilometre"), "SquareKilometers");
    assert_eq!(unit_name("cubic metre"), "CubicMeters");
  }

  #[test]
  fn unit_names_handle_irregular_forms() {
    assert_eq!(unit_name("foot"), "Feet");
    assert_eq!(unit_name("inch"), "Inches");
    assert_eq!(unit_name("hertz"), "Hertz");
    assert_eq!(unit_name("siemens"), "Siemens");
    assert_eq!(unit_name("tonne"), "MetricTons");
    assert_eq!(unit_name("century"), "Centuries");
    assert_eq!(unit_name("degree Celsius"), "DegreesCelsius");
    assert_eq!(unit_name("degree Fahrenheit"), "DegreesFahrenheit");
  }

  #[test]
  fn unit_quotients_split_on_per() {
    assert_eq!(
      expr_to_string(&unit_to_expr("metre per second")),
      "\"Meters\"/\"Seconds\""
    );
    assert_eq!(
      expr_to_string(&unit_to_expr("kilometre per hour")),
      "\"Kilometers\"/\"Hours\""
    );
    assert_eq!(expr_to_string(&unit_to_expr("gram")), "\"Grams\"");
  }

  #[test]
  fn amounts_stay_exact_when_integral() {
    assert!(matches!(parse_amount("+8848"), Expr::Integer(8848)));
    assert!(matches!(parse_amount("-42"), Expr::Integer(-42)));
    assert!(
      matches!(parse_amount("+73.4767"), Expr::Real(f) if (f - 73.4767).abs() < 1e-12)
    );
  }

  #[test]
  fn commons_file_names_are_percent_encoded() {
    assert_eq!(
      percent_encode("Flag of Germany.svg"),
      "Flag%20of%20Germany.svg"
    );
    assert_eq!(
      percent_encode("Foo (bar), baz.jpg"),
      "Foo%20(bar),%20baz.jpg"
    );
  }

  #[test]
  fn specs_resolve_from_all_supported_forms() {
    let ext_id = Expr::FunctionCall {
      name: "ExternalIdentifier".to_string(),
      args: vec![
        Expr::String("WikidataID".to_string()),
        Expr::String("Q405".to_string()),
        Expr::Association(vec![]),
      ]
      .into(),
    };
    assert_eq!(resolve_id(&ext_id, 'Q'), Some("Q405".to_string()));
    // The identifier kind is enforced: an item is not a property.
    assert_eq!(resolve_id(&ext_id, 'P'), None);
    assert_eq!(
      resolve_id(&Expr::String("P2067".to_string()), 'P'),
      Some("P2067".to_string())
    );
    assert_eq!(
      resolve_id(
        &Expr::String("https://www.wikidata.org/wiki/Q405".to_string()),
        'Q'
      ),
      Some("Q405".to_string())
    );
    assert_eq!(
      resolve_id(
        &Expr::String(
          "https://www.wikidata.org/wiki/Property:P2067".to_string()
        ),
        'P'
      ),
      Some("P2067".to_string())
    );
    assert_eq!(resolve_id(&Expr::Integer(1), 'Q'), None);
    assert_eq!(resolve_id(&Expr::String("Q40x5".to_string()), 'Q'), None);
  }

  #[test]
  fn times_map_to_date_objects_at_precision() {
    use crate::syntax::expr_to_output;
    let day = time_to_date_object("+1879-03-14T00:00:00Z", 11).unwrap();
    assert_eq!(expr_to_output(&day), "DateObject[{1879, 3, 14}, Day]");
    let month = time_to_date_object("+1879-03-14T00:00:00Z", 10).unwrap();
    assert_eq!(expr_to_output(&month), "DateObject[{1879, 3}, Month]");
    let year = time_to_date_object("+1879-03-14T00:00:00Z", 9).unwrap();
    assert_eq!(expr_to_output(&year), "DateObject[{1879}, Year]");
  }
}
