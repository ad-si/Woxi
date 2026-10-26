//! `ExampleData[{"MachineLearning", …}]`: generated training sets.
//!
//! Examples have the recorded form (`features -> label`, an image or a
//! sentence as input, …), the recorded training and test sizes and the
//! recorded label frequencies. The outputs really depend on the inputs — a
//! hidden linear score of the features decides the class (or the value, for
//! regression) — so classifiers trained on the data learn something.
//! Digits are drawn as pen strokes with random slant, size and position;
//! sentences use a vocabulary that differs between classes.

use super::images::cached_opt;
use super::instantiate::{Filler, instantiate};
use super::raster::{Canvas, Paint};
use super::rng::Rng;
use super::tabular::{Column, NumSpec, generate, parse_columns, quantize};
use super::words::{Lexicon, capitalize};
use super::{Collection, Entry, GENERATED_SOURCE};
use crate::functions::*;
use crate::syntax::ImageType;

pub struct MachineLearning;

fn sizes(entry: &Entry) -> (usize, usize) {
  let s = entry
    .stat("@Sizes")
    .map(|s| super::images::numbers(&s))
    .unwrap_or_default();
  (
    s.first().copied().unwrap_or(0.0) as usize,
    s.get(1).copied().unwrap_or(0.0) as usize,
  )
}

/// `(input columns, output column)` statistics.
fn stats(entry: &Entry) -> (Vec<Column>, Option<Column>) {
  match entry.stat("@Stats") {
    Some(Expr::List(ref s)) if s.len() == 2 => (
      parse_columns(&s[0]),
      parse_columns(&s[1]).into_iter().next(),
    ),
    _ => (Vec::new(), None),
  }
}

/// Pen strokes of the digits 0–9 in a unit box (y down).
fn digit_strokes(d: usize) -> Vec<Vec<(f64, f64)>> {
  let arc =
    |cx: f64, cy: f64, rx: f64, ry: f64, a0: f64, a1: f64| -> Vec<(f64, f64)> {
      (0..=16)
        .map(|i| {
          let a = a0 + (a1 - a0) * i as f64 / 16.0;
          (cx + rx * a.cos(), cy + ry * a.sin())
        })
        .collect()
    };
  use std::f64::consts::PI;
  match d {
    0 => vec![arc(0.5, 0.5, 0.28, 0.4, 0.0, 2.0 * PI)],
    1 => vec![vec![(0.35, 0.25), (0.55, 0.1), (0.55, 0.9)]],
    2 => vec![
      [
        arc(0.5, 0.32, 0.25, 0.22, PI, 2.2 * PI),
        vec![(0.7, 0.45), (0.25, 0.9), (0.78, 0.9)],
      ]
      .concat(),
    ],
    3 => vec![
      arc(0.48, 0.3, 0.24, 0.2, -0.8 * PI, 0.5 * PI),
      arc(0.48, 0.7, 0.26, 0.2, -0.5 * PI, 0.8 * PI),
    ],
    4 => vec![vec![(0.62, 0.9), (0.62, 0.1), (0.2, 0.65), (0.8, 0.65)]],
    5 => vec![
      [
        vec![(0.75, 0.1), (0.3, 0.1), (0.28, 0.45)],
        arc(0.48, 0.65, 0.26, 0.24, -0.7 * PI, 0.8 * PI),
      ]
      .concat(),
    ],
    6 => vec![
      [
        vec![(0.68, 0.12), (0.4, 0.4)],
        arc(0.5, 0.68, 0.22, 0.22, 0.0, 2.0 * PI),
      ]
      .concat(),
    ],
    7 => vec![vec![(0.22, 0.1), (0.8, 0.1), (0.42, 0.9)]],
    8 => vec![
      arc(0.5, 0.3, 0.2, 0.2, 0.0, 2.0 * PI),
      arc(0.5, 0.7, 0.25, 0.21, 0.0, 2.0 * PI),
    ],
    _ => vec![
      [
        arc(0.5, 0.32, 0.22, 0.22, 0.0, 2.0 * PI),
        vec![(0.72, 0.32), (0.6, 0.9)],
      ]
      .concat(),
    ],
  }
}

/// A handwritten-looking digit as a 28×28 byte image, dark on white like
/// the original scans.
fn digit_image(rng: &mut Rng, d: usize, w: usize, h: usize) -> Expr {
  let mut c = Canvas::new(w, h, [1.0; 3]);
  let slant = rng.range(-0.25, 0.25);
  let s = rng.range(0.6, 0.8) * w as f64;
  let (ox, oy) = (
    rng.range(0.1, 0.25) * w as f64,
    rng.range(0.08, 0.18) * h as f64,
  );
  let width = rng.range(1.6, 3.0);
  for stroke in digit_strokes(d % 10) {
    let pts: Vec<(f64, f64)> = stroke
      .iter()
      .map(|&(x, y)| {
        (
          ox + (x + slant * (0.5 - y)) * s * 0.85 + rng.range(-0.3, 0.3),
          oy + y * s * 1.05 + rng.range(-0.3, 0.3),
        )
      })
      .collect();
    c.polyline(&pts, width, &Paint::Solid([0.0; 3]));
  }
  c.into_image(1, ImageType::Byte, None)
}

/// Class-specific vocabulary for text classification.
const POSITIVE: &[&str] = &[
  "wonderful",
  "moving",
  "brilliant",
  "charming",
  "delightful",
  "beautiful",
  "clever",
  "warm",
  "funny",
  "gripping",
  "fresh",
  "honest",
  "memorable",
  "rich",
  "smart",
];
const NEGATIVE: &[&str] = &[
  "dull",
  "tedious",
  "clumsy",
  "flat",
  "boring",
  "messy",
  "predictable",
  "shallow",
  "weak",
  "tired",
  "awkward",
  "empty",
  "hollow",
  "stale",
  "silly",
];

/// All examples (in generated order) and the positions of the test set.
fn examples(entry: &Entry) -> Option<(Vec<Expr>, Vec<bool>)> {
  let (train, test) = sizes(entry);
  let n = train + test;
  if n == 0 {
    return None;
  }
  let example = entry.stat("@Example")?;
  let Expr::Rule {
    pattern: input_sk, ..
  } = &example
  else {
    return None;
  };
  let (inputs, output) = stats(entry);
  let mut rng = entry.rng("examples");
  let lex = Lexicon::English;

  // Labels: the recorded class frequencies, or regression values.
  let labels: Vec<Expr> = match &output {
    Some(col @ Column::Categorical(_)) => {
      let mut l: Vec<Expr> =
        generate(&mut rng, std::slice::from_ref(col), n, None, &lex, false)
          .into_iter()
          .map(|mut r| r.remove(0))
          .collect();
      // Ordered by class so a score can be mapped onto them below.
      l.sort_by_key(crate::syntax::expr_to_string);
      l
    }
    Some(Column::Numeric(spec)) => {
      super::tabular::sorted_draws(&mut rng, spec, n)
        .into_iter()
        .map(|v| quantize(v, spec))
        .collect()
    }
    _ => (0..n).map(|i| Expr::Integer((i % 10) as i128)).collect(),
  };

  let input_values: Vec<Expr> = match input_sk.as_ref() {
    // Images (digits) and free text are drawn per label below.
    Expr::FunctionCall { name, args }
      if name == "Sk"
        && matches!(args.first(), Some(Expr::String(s)) if s == "Image") =>
    {
      Vec::new()
    }
    Expr::Identifier(s) if s == "String" => Vec::new(),
    _ => {
      if inputs.is_empty() {
        let mut f = Filler {
          rng: entry.rng("inputs"),
          spec: NumSpec::default(),
          lexicon: &lex,
        };
        (0..n)
          .map(|_| instantiate(input_sk, &mut f))
          .collect::<Vec<_>>()
      } else {
        generate(&mut rng, &inputs, n, None, &lex, false)
          .into_iter()
          .map(|r| Expr::List(r.into()))
          .collect()
      }
    }
  };

  // Pair inputs with labels through a hidden score: a random linear
  // combination of the numeric features (rank-matched to the sorted
  // labels), so the label is predictable from the inputs.
  let mut order: Vec<usize> = (0..n).collect();
  if !input_values.is_empty() {
    let k = match &input_values[0] {
      Expr::List(v) => v.len(),
      _ => 1,
    };
    let weights: Vec<f64> = (0..k).map(|_| rng.normal()).collect();
    let score = |e: &Expr| -> f64 {
      match e {
        Expr::List(v) => v
          .iter()
          .zip(&weights)
          .map(|(x, w)| {
            w * match x {
              Expr::Integer(i) => *i as f64,
              Expr::Real(r) => *r,
              other => (crate::syntax::expr_to_string(other).len() % 7) as f64,
            }
          })
          .sum(),
        _ => 0.0,
      }
    };
    let mut scored: Vec<(f64, usize)> = input_values
      .iter()
      .enumerate()
      .map(|(i, v)| (score(v) + rng.normal() * 0.05, i))
      .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0));
    order = scored.into_iter().map(|(_, i)| i).collect();
  }
  // order[r] = the input that gets the r-th label.
  let mut paired: Vec<(Expr, Expr)> = Vec::with_capacity(n);
  for (r, label) in labels.iter().enumerate() {
    let input = if !input_values.is_empty() {
      input_values[order[r]].clone()
    } else if matches!(input_sk.as_ref(), Expr::Identifier(s) if s == "String")
    {
      // A sentence leaning on its class's vocabulary.
      let positive =
        crate::syntax::expr_to_string(label).contains("pos") || r % 2 == 0;
      let words = 8 + rng.below(18);
      let mut s = String::new();
      for i in 0..words {
        let w = if rng.chance(0.25) {
          rng
            .pick(if positive { POSITIVE } else { NEGATIVE })
            .to_string()
        } else {
          lex.word(&mut rng)
        };
        if i > 0 {
          s.push(' ');
        }
        s.push_str(&if i == 0 { capitalize(&w) } else { w });
      }
      s.push_str(" . ");
      str_expr(&s)
    } else {
      let shape = super::images::shape_of(input_sk)?;
      let digit = match label {
        Expr::Integer(d) => *d as usize,
        _ => r % 10,
      };
      digit_image(&mut rng, digit, shape.width, shape.height)
    };
    paired.push((input, label.clone()));
  }
  // Shuffle into a natural order, then pick a stratified-ish test set.
  rng.shuffle(&mut paired);
  let mut is_test = vec![false; n];
  let mut idx: Vec<usize> = (0..n).collect();
  rng.shuffle(&mut idx);
  for &i in idx.iter().take(test) {
    is_test[i] = true;
  }
  Some((
    paired.into_iter().map(|(i, o)| rule_expr(i, o)).collect(),
    is_test,
  ))
}

fn all_examples(entry: &Entry) -> Option<(Vec<Expr>, Vec<bool>)> {
  let data = cached_opt(format!("ML/{}/data", entry.name), || {
    let (ex, t) = examples(entry)?;
    Some(Expr::List(
      vec![
        Expr::List(ex.into()),
        Expr::List(t.into_iter().map(bool_expr).collect::<Vec<_>>().into()),
      ]
      .into(),
    ))
  })?;
  let Expr::List(ref parts) = data else {
    return None;
  };
  let (Expr::List(ex), Expr::List(t)) = (&parts[0], &parts[1]) else {
    return None;
  };
  Some((
    ex.iter().cloned().collect(),
    t.iter()
      .map(|b| matches!(b, Expr::Identifier(s) if s == "True"))
      .collect(),
  ))
}

impl Collection for MachineLearning {
  fn value(&self, entry: &Entry, property: Option<&str>) -> Option<Expr> {
    let name = match entry.exact("Name") {
      Some(Expr::String(ref n)) => n.clone(),
      _ => entry.name.to_string(),
    };
    Some(match property? {
      "Data" => Expr::List(all_examples(entry)?.0.into()),
      "TrainingData" | "TestData" => {
        let (ex, t) = all_examples(entry)?;
        let want = property == Some("TestData");
        Expr::List(
          ex.into_iter()
            .zip(t)
            .filter(|(_, is)| *is == want)
            .map(|(e, _)| e)
            .collect::<Vec<_>>()
            .into(),
        )
      }
      "Description" => str_expr(&format!("{name} (generated)")),
      "LongDescription" => {
        let (train, test) = sizes(entry);
        str_expr(&format!(
          "{name}: {train} training and {test} test examples generated by Woxi. They have \
           the form, label frequencies and value ranges of the original dataset, and the \
           outputs depend on the inputs, but the examples are synthetic."
        ))
      }
      "Source" => str_expr(GENERATED_SOURCE),
      "VariableDescriptions" => {
        let sk = entry.skeleton("VariableDescriptions")?;
        let Expr::Rule { pattern, .. } = &sk else {
          return None;
        };
        let k = match pattern.as_ref() {
          Expr::List(v) => v.len(),
          Expr::FunctionCall { name, args } if name == "Rep" => {
            match args.first() {
              Some(Expr::Integer(n)) => *n as usize,
              _ => 1,
            }
          }
          _ => 1,
        };
        let inputs: Vec<Expr> = (1..=k)
          .map(|i| str_expr(&format!("Input feature {i}")))
          .collect();
        let lhs = if k == 1 && matches!(pattern.as_ref(), Expr::Identifier(_)) {
          inputs[0].clone()
        } else {
          Expr::List(inputs.into())
        };
        rule_expr(lhs, str_expr("output"))
      }
      _ => return None,
    })
  }
}
