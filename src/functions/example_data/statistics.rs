//! `ExampleData[{"Statistics", …}]`: generated statistical datasets.
//!
//! The table (`"DataElements"`) is generated column by column from the
//! recorded column types, ranges, spreads, category frequencies and rank
//! correlations (see [`tabular`]); the time series, event series and event
//! data are built from it in Wolfram's `TemporalData` / `EventData` forms.
//! Headings, column types, dimensions, data type and application area are
//! the recorded metadata; descriptions are Woxi's.

use super::images::cached_opt;
use super::instantiate::{Filler, instantiate};
use super::tabular::{
  Column, NumSpec, generate, parse_columns, parse_correlations,
};
use super::words::Lexicon;
use super::{Collection, Entry, GENERATED_SOURCE};
use crate::functions::*;

pub struct StatisticsData;

fn strings(e: Option<&Expr>) -> Vec<String> {
  match e {
    Some(Expr::List(l)) => l
      .iter()
      .filter_map(|x| match x {
        Expr::String(s) => Some(s.clone()),
        _ => None,
      })
      .collect(),
    _ => Vec::new(),
  }
}

fn dims(e: &Expr) -> Vec<usize> {
  match e {
    Expr::List(d) => d
      .iter()
      .filter_map(|x| match x {
        Expr::Integer(n) => Some(*n as usize),
        _ => None,
      })
      .collect(),
    _ => Vec::new(),
  }
}

/// `(rows, columns)` of a table skeleton: `Arr[{n, m}]`, `Arr[{n}]`,
/// `Rep[n, {Rep[m, …] | Arr[{m}] | {…m…}, …}]`.
fn table_shape(sk: &Expr) -> Option<(usize, Option<usize>)> {
  let Expr::FunctionCall { name, args } = sk else {
    return None;
  };
  match name.as_str() {
    "Arr" => {
      let d = dims(args.first()?);
      Some((*d.first()?, d.get(1).copied()))
    }
    "Rep" => {
      let n = match args.first()? {
        Expr::Integer(n) => *n as usize,
        _ => return None,
      };
      let m = match args.get(1)? {
        Expr::List(v) => v.iter().find_map(|row| match row {
          Expr::FunctionCall { name, args } if name == "Rep" => {
            match args.first() {
              Some(Expr::Integer(m)) => Some(*m as usize),
              _ => None,
            }
          }
          Expr::FunctionCall { name, args } if name == "Arr" => {
            args.first().and_then(|d| dims(d).first().copied())
          }
          Expr::List(cells) => Some(cells.len()),
          _ => None,
        }),
        _ => None,
      };
      Some((n, m))
    }
    _ => None,
  }
}

fn columns(entry: &Entry) -> Vec<Column> {
  entry
    .stat("@Stats")
    .map(|s| parse_columns(&s))
    .unwrap_or_default()
}

fn numeric_spec(cols: &[Column], j: usize) -> NumSpec {
  cols
    .iter()
    .skip(j)
    .chain(cols.iter())
    .find_map(|c| match c {
      Column::Numeric(s) => Some(s.clone()),
      _ => None,
    })
    .unwrap_or_default()
}

fn filler<'a>(
  entry: &Entry,
  salt: &str,
  spec: NumSpec,
  lex: &'a Lexicon,
) -> Filler<'a> {
  Filler {
    rng: entry.rng(salt),
    spec,
    lexicon: lex,
  }
}

/// The generated table: rows of cells (one cell per row for a vector).
fn rows(entry: &Entry) -> Option<(Vec<Vec<Expr>>, bool)> {
  let sk = entry.skeleton("DataElements")?;
  let (n, m) = table_shape(&sk)?;
  let cols = columns(entry);
  let lex = Lexicon::English;
  let width = m.unwrap_or(1);
  if cols.len() == width {
    let corr = parse_correlations(entry.stat("@Corr").as_ref());
    let mut rng = entry.rng("table");
    let temporal = matches!(entry.exact("DataType"), Some(Expr::String(ref t)) if t.contains("Series"));
    return Some((
      generate(&mut rng, &cols, n, corr.as_ref(), &lex, temporal),
      m.is_none(),
    ));
  }
  // Irregular tables (ragged rows, missing statistics): follow the skeleton.
  let mut f = filler(entry, "table", numeric_spec(&cols, 0), &lex);
  let value = instantiate(&sk, &mut f);
  let Expr::List(ref items) = value else {
    return None;
  };
  Some((
    items
      .iter()
      .map(|r| match r {
        Expr::List(c) => c.iter().cloned().collect(),
        other => vec![other.clone()],
      })
      .collect(),
    m.is_none(),
  ))
}

fn data_elements(entry: &Entry) -> Option<Expr> {
  let (rows, vector) = rows(entry)?;
  Some(Expr::List(
    rows
      .into_iter()
      .map(|r| {
        if vector && r.len() == 1 {
          r.into_iter().next().unwrap_or(Expr::Integer(0))
        } else {
          Expr::List(r.into())
        }
      })
      .collect::<Vec<_>>()
      .into(),
  ))
}

/// The numeric cells of each row.
fn numeric_rows(entry: &Entry) -> Vec<Vec<Expr>> {
  rows(entry)
    .map(|(r, _)| {
      r.into_iter()
        .map(|row| {
          row
            .into_iter()
            .filter(|c| matches!(c, Expr::Integer(_) | Expr::Real(_)))
            .collect()
        })
        .collect()
    })
    .unwrap_or_default()
}

/// The value path of a temporal property with `k` points of dimension `d`
/// (`None` for scalars), taken from the table where its shape allows.
fn path_values(entry: &Entry, k: usize, d: Option<usize>) -> Expr {
  let num = numeric_rows(entry);
  let n = num.len();
  let width = num.first().map_or(0, std::vec::Vec::len);
  let pick = |row: &Vec<Expr>, cnt: usize| -> Vec<Expr> {
    row[row.len().saturating_sub(cnt)..].to_vec()
  };
  let values: Option<Vec<Expr>> = match d {
    Some(d) if n >= k && width >= d => Some(
      num
        .iter()
        .take(k)
        .map(|r| Expr::List(pick(r, d).into()))
        .collect(),
    ),
    None if n == k && width >= 1 => {
      Some(num.iter().map(|r| r[width - 1].clone()).collect())
    }
    // A year-by-month table read row-wise.
    None if width > 1 && n * (width - 1) >= k => {
      Some(num.iter().flat_map(|r| r[1..].to_vec()).take(k).collect())
    }
    _ => None,
  };
  let values = values.unwrap_or_else(|| {
    let lex = Lexicon::English;
    let mut f = filler(entry, "path", numeric_spec(&columns(entry), 0), &lex);
    let dims = match d {
      Some(d) => vec![Expr::Integer(k as i128), Expr::Integer(d as i128)],
      None => vec![Expr::Integer(k as i128)],
    };
    match instantiate(
      &call(
        "Arr",
        vec![
          Expr::List(dims.into()),
          Expr::List(vec![id_expr("Real")].into()),
        ],
      ),
      &mut f,
    ) {
      Expr::List(ref v) => v.iter().cloned().collect(),
      _ => Vec::new(),
    }
  });
  Expr::List(vec![Expr::List(values.into())].into())
}

/// Replace the `Arr[…]` placeholders in a recorded time specification by
/// increasing integers.
fn concrete_times(spec: &Expr) -> Expr {
  match spec {
    Expr::FunctionCall { name, args } if name == "Arr" => {
      let k = args
        .first()
        .map(dims)
        .and_then(|d| d.first().copied())
        .unwrap_or(0);
      Expr::List(
        (1..=k)
          .map(|i| Expr::Integer(i as i128))
          .collect::<Vec<_>>()
          .into(),
      )
    }
    Expr::FunctionCall { name, args } => {
      call(name, args.iter().map(concrete_times).collect())
    }
    Expr::List(items) => {
      Expr::List(items.iter().map(concrete_times).collect::<Vec<_>>().into())
    }
    other => other.clone(),
  }
}

/// `TemporalData[kind, {{path}, times, 1, {"Continuous", 1}, {"Discrete",
/// 1}, 1, options}, dated, 10.]`, Wolfram's form of a time or event series.
fn temporal(entry: &Entry, column: &str, kind: &str) -> Option<Expr> {
  let sk = entry.skeleton(column)?;
  let Expr::FunctionCall { args, .. } = &sk else {
    return None;
  };
  let Some(Expr::List(fields)) = args.get(2) else {
    return None;
  };
  let path_dims = match fields.first() {
    Some(Expr::FunctionCall { name, args }) if name == "Arr" => {
      args.first().map(dims).unwrap_or_default()
    }
    _ => Vec::new(),
  };
  let k = path_dims.get(1).copied().unwrap_or(0);
  let d = path_dims.get(2).copied();
  let times = entry
    .stat("@Times")
    .filter(|t| !matches!(t, Expr::Identifier(n) if n == "None"))
    .map_or_else(
      || {
        Expr::List(
          vec![Expr::List(
            vec![Expr::Integer(1), Expr::Integer(k as i128), Expr::Integer(1)]
              .into(),
          )]
          .into(),
        )
      },
      |t| concrete_times(&t),
    );
  let resampling = if kind == "EventSeries" {
    id_expr("None")
  } else {
    Expr::List(
      vec![
        str_expr("Interpolation"),
        rule_expr(id_expr("InterpolationOrder"), Expr::Integer(1)),
      ]
      .into(),
    )
  };
  let mut options = Vec::new();
  if let Some(d) = d {
    options.push(rule_expr(
      id_expr("ValueDimensions"),
      Expr::Integer(d as i128),
    ));
  }
  options.push(rule_expr(id_expr("ResamplingMethod"), resampling));
  let dated = matches!(args.get(3), Some(Expr::Identifier(t)) if t == "True");
  let pair = |s: &str| Expr::List(vec![str_expr(s), Expr::Integer(1)].into());
  let value = call(
    "TemporalData",
    vec![
      id_expr(kind),
      Expr::List(
        vec![
          path_values(entry, k, d),
          times,
          Expr::Integer(1),
          pair("Continuous"),
          pair("Discrete"),
          Expr::Integer(1),
          Expr::List(options.into()),
        ]
        .into(),
      ),
      bool_expr(dated),
      Expr::Real(10.0),
    ],
  );
  Some(crate::evaluator::evaluate_expr_to_expr(&value).unwrap_or(value))
}

/// `EventData[Automatic, {times, censoring, None}]` (or a list of them,
/// one per group).
fn event_data(entry: &Entry) -> Option<Expr> {
  let sk = entry.skeleton("EventData")?;
  let cols = columns(entry);
  let lex = Lexicon::English;
  let times_spec = NumSpec {
    min: numeric_spec(&cols, 0).min.max(0.0),
    ..numeric_spec(&cols, 0)
  };
  let mut f = filler(entry, "events", times_spec, &lex);
  let mut rng = entry.rng("censoring");
  let one = |e: &Expr, f: &mut Filler, rng: &mut super::rng::Rng| -> Expr {
    let Expr::FunctionCall { args, .. } = e else {
      return instantiate(e, f);
    };
    let Some(Expr::List(parts)) = args.get(2) else {
      return instantiate(e, f);
    };
    let times = parts.first().map_or(id_expr("None"), |t| instantiate(t, f));
    let censoring = match parts.get(1) {
      Some(Expr::FunctionCall { name, args }) if name == "Arr" => {
        let k = args
          .first()
          .map(dims)
          .and_then(|d| d.first().copied())
          .unwrap_or(0);
        Expr::List(
          (0..k)
            .map(|_| Expr::Integer(i128::from(rng.chance(0.3))))
            .collect::<Vec<_>>()
            .into(),
        )
      }
      Some(other) => instantiate(other, f),
      None => id_expr("None"),
    };
    let rest = parts.get(2).cloned().unwrap_or(id_expr("None"));
    call(
      "EventData",
      vec![
        id_expr("Automatic"),
        Expr::List(vec![times, censoring, rest].into()),
      ],
    )
  };
  Some(match &sk {
    Expr::List(groups) => Expr::List(
      groups
        .iter()
        .map(|g| one(g, &mut f, &mut rng))
        .collect::<Vec<_>>()
        .into(),
    ),
    other => one(other, &mut f, &mut rng),
  })
}

fn heading_list(entry: &Entry) -> Vec<String> {
  strings(entry.exact("ColumnHeadings").as_ref())
}

fn describe(entry: &Entry) -> String {
  let name = match entry.exact("Name") {
    Some(Expr::String(ref n)) => n.clone(),
    _ => entry.name.to_string(),
  };
  let kind = match entry.exact("DataType") {
    Some(Expr::String(ref k)) => k.clone(),
    _ => "data".to_string(),
  };
  let n = match entry.exact("ObservationCount") {
    Some(Expr::Integer(n)) => n,
    _ => 0,
  };
  format!("{name}: {n} generated observations ({kind}).")
}

impl Collection for StatisticsData {
  fn value(&self, entry: &Entry, property: Option<&str>) -> Option<Expr> {
    let key = |p: &str| format!("Statistics/{}/{p}", entry.name);
    Some(match property? {
      "DataElements" => {
        cached_opt(key("DataElements"), || data_elements(entry))?
      }
      "TimeSeries" => cached_opt(key("TimeSeries"), || {
        temporal(entry, "TimeSeries", "TimeSeries")
      })?,
      "EventSeries" => cached_opt(key("EventSeries"), || {
        temporal(entry, "EventSeries", "EventSeries")
      })?,
      "EventData" => cached_opt(key("EventData"), || event_data(entry))?,
      "ColumnDescriptions" => {
        let types = strings(entry.exact("ColumnTypes").as_ref());
        super::string_list(
          &heading_list(entry)
            .iter()
            .enumerate()
            .map(|(i, h)| {
              let t = types.get(i).map_or("values", |t| match t.as_str() {
                "Categorical" => "category labels",
                "String" => "names",
                _ => "numeric values",
              });
              format!("{h} ({t}).")
            })
            .collect::<Vec<_>>(),
        )
      }
      "Description" => str_expr(&describe(entry)),
      "LongDescription" => {
        let headings = heading_list(entry).join(", ");
        str_expr(&format!(
          "{} The columns are {headings}. Woxi generates the values to match the \
           original dataset's column types, ranges and spread; they are not the \
           original observations.",
          describe(entry)
        ))
      }
      "Source" => str_expr(GENERATED_SOURCE),
      _ => return None,
    })
  }
}
