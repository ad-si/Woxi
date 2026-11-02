//! `BulletGauge[values, references, {min, r1, …, max}]`: a horizontal
//! bullet graph as a symbolic `Graphics` — the qualitative ranges as grey
//! bands, each value as a bar from `min` (longest at the back, so every one
//! shows), and each reference as a marker line across the bar.

use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;
use crate::syntax::Expr;

fn call(name: &str, args: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args: args.into(),
  }
}

fn rule(key: &str, value: Expr) -> Expr {
  Expr::Rule {
    pattern: Box::new(Expr::Identifier(key.to_string())),
    replacement: Box::new(value),
  }
}

fn pt(x: f64, y: f64) -> Expr {
  Expr::List(vec![Expr::Real(x), Expr::Real(y)].into())
}

/// The numbers of a value or list of values.
fn numbers(e: &Expr) -> Option<Vec<f64>> {
  match e {
    Expr::List(items) => items.iter().map(try_eval_to_f64).collect(),
    other => try_eval_to_f64(other).map(|v| vec![v]),
  }
}

/// Wolfram's default gauge colour.
fn default_style() -> Expr {
  call(
    "RGBColor",
    vec![
      Expr::Real(0.368417),
      Expr::Real(0.506779),
      Expr::Real(0.709798),
    ],
  )
}

pub fn bullet_gauge_ast(
  args: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  let is_option =
    |e: &Expr| matches!(e, Expr::Rule { .. } | Expr::RuleDelayed { .. });
  let positional: Vec<&Expr> =
    args.iter().take_while(|a| !is_option(a)).collect();
  if positional.len() != 3 {
    return Ok(None);
  }
  let (Some(values), Some(references), Some(range)) = (
    numbers(positional[0]),
    numbers(positional[1]),
    numbers(positional[2]),
  ) else {
    return Ok(None);
  };
  if range.len() < 2 {
    return Ok(None);
  }
  let (lo, hi) = (range[0], range[range.len() - 1]);
  if !(hi > lo) {
    return Ok(None);
  }

  let mut styles: Vec<Expr> = Vec::new();
  let mut label = None;
  let mut divisions = None;
  let mut options = Vec::new();
  for opt in &args[positional.len()..] {
    let (Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    }) = opt
    else {
      continue;
    };
    let value = || {
      evaluate_expr_to_expr(replacement)
        .unwrap_or_else(|_| replacement.as_ref().clone())
    };
    match pattern.as_ref() {
      Expr::Identifier(k) if k == "GaugeStyle" => {
        let v = value();
        styles = match &v {
          Expr::List(items) => items.to_vec(),
          other => vec![other.clone()],
        }
      }
      Expr::Identifier(k) if k == "GaugeLabels" => {
        let v = value();
        label = matches!(v, Expr::String(_)).then_some(v);
      }
      Expr::Identifier(k) if k == "ScaleDivisions" => {
        divisions = match &value() {
          Expr::Integer(n) if *n > 0 => Some(*n as usize),
          Expr::List(items) => match items.first() {
            Some(Expr::Integer(n)) if *n > 0 => Some(*n as usize),
            _ => None,
          },
          _ => None,
        }
      }
      Expr::Identifier(k)
        if matches!(
          k.as_str(),
          "ScaleRangeStyle"
            | "ReferenceMarkers"
            | "ScaleRanges"
            | "ScalePadding"
            | "TicksStyle"
            | "PlotTheme"
            | "GaugeMarkers"
            | "GaugeFaceStyle"
            | "GaugeFrameStyle"
            | "GaugeFrameSize"
            | "ScaleOrigin"
            | "PerformanceGoal"
            | "TargetUnits"
        ) => {}
      _ => options.push(opt.clone()),
    }
  }
  if styles.is_empty() {
    styles.push(default_style());
  }

  let mut primitives = Vec::new();
  // The qualitative ranges, darkest first.
  let bands = range.len() - 1;
  for k in 0..bands {
    let level = 0.6 + 0.3 * k as f64 / bands.max(2) as f64;
    primitives.push(Expr::List(
      vec![
        call("GrayLevel", vec![Expr::Real(level)]),
        call("Rectangle", vec![pt(range[k], 0.0), pt(range[k + 1], 1.0)]),
      ]
      .into(),
    ));
  }
  // The values, longest bar behind.
  let mut order: Vec<usize> = (0..values.len()).collect();
  order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
  for i in order {
    let v = values[i].clamp(lo, hi);
    primitives.push(Expr::List(
      vec![
        styles[i % styles.len()].clone(),
        call("Rectangle", vec![pt(lo, 0.35), pt(v, 0.65)]),
      ]
      .into(),
    ));
  }
  for r in references {
    if (lo..=hi).contains(&r) {
      primitives.push(Expr::List(
        vec![
          call("GrayLevel", vec![Expr::Integer(0)]),
          call("AbsoluteThickness", vec![Expr::Integer(2)]),
          call(
            "Line",
            vec![Expr::List(vec![pt(r, 0.2), pt(r, 0.8)].into())],
          ),
        ]
        .into(),
      ));
    }
  }

  let ticks = match divisions {
    Some(n) => Expr::List(
      (0..=n)
        .map(|k| {
          let x = lo + (hi - lo) * k as f64 / n as f64;
          // Labelled to two decimals at most (`1.93`, `14`).
          let text = format!("{x:.2}");
          let text = text.trim_end_matches('0').trim_end_matches('.');
          // Many divisions label only every few ticks, so the labels fit.
          let text = if k % n.div_ceil(7) == 0 || k == n {
            text
          } else {
            ""
          };
          Expr::List(vec![Expr::Real(x), Expr::String(text.to_string())].into())
        })
        .collect(),
    ),
    None => Expr::Identifier("Automatic".into()),
  };
  let mut graphics_args = vec![Expr::List(primitives.into())];
  graphics_args.extend(options);
  graphics_args.push(rule(
    "Axes",
    Expr::List(
      vec![
        Expr::Identifier("True".into()),
        Expr::Identifier("False".into()),
      ]
      .into(),
    ),
  ));
  graphics_args.push(rule("AxesOrigin", pt(lo, 0.0)));
  graphics_args.push(rule(
    "Ticks",
    Expr::List(vec![ticks, Expr::Identifier("None".into())].into()),
  ));
  graphics_args.push(rule("AspectRatio", Expr::Real(0.2)));
  if let Some(label) = label {
    graphics_args.push(rule("PlotLabel", label));
  }
  Ok(Some(evaluate_expr_to_expr(&call(
    "Graphics",
    graphics_args,
  ))?))
}
