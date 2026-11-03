//! `BulletGauge[values, references, {min, r1, …, max}]`: a horizontal
//! bullet graph, built as the `Graphics` wolframscript returns. The scale
//! runs over `{0, 1}`: the qualitative ranges are grey bands (`{min, max}`
//! alone splits at a half and three quarters), the scale line carries
//! `FindDivisions` ticks, each value is a bar (rotated onto the scale,
//! longest first) and each reference a translucent marker.

use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;
use crate::syntax::{Expr, expr_to_source_form};

fn call(name: &str, args: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args: args.into(),
  }
}

fn eval(e: Expr) -> Expr {
  evaluate_expr_to_expr(&e).unwrap_or(e)
}

/// A value or list of values, as given.
fn items(e: &Expr) -> Vec<Expr> {
  match e {
    Expr::List(items) => items.to_vec(),
    other => vec![other.clone()],
  }
}

const GRAY_50: &str =
  "RGBColor[0.5001144397259117, 0.5001144397259117, 0.5001144397259117]";
const GRAY_25: &str =
  "RGBColor[0.25005721986295587, 0.25005721986295587, 0.25005721986295587]";
const BAND_GRAYS: [&str; 3] = [
  "RGBColor[0.600079345256126, 0.600079345256126, 0.600079345256126]",
  "RGBColor[0.7501007083058716, 0.7501007083058716, 0.7501007083058716]",
  "RGBColor[0.8999984736740948, 0.8999984736740948, 0.8999984736740948]",
];
const FONT: &str =
  "{GrayLevel[0.25], FontSize -> 10, FontFamily -> \"Helvetica\"}";

/// The colour of the `k`-th series (1-based), counting the value bars
/// first: hues `9/10 φ⁻¹` apart (modulo 0.9) at saturation and brightness
/// 0.73, starting from `0.9 (1/60 + φ⁻¹)`.
fn series_color(k: usize) -> Expr {
  let phi = f64::midpoint(1.0, 5f64.sqrt());
  let step = 0.9 * (phi - 1.0);
  let mut h = 0.9 * (1.0 / 60.0 + 1.0 / phi);
  for _ in 1..k {
    h += step;
    if h >= 0.9 {
      h -= 0.9;
    }
  }
  let (s, v) = (0.73_f64, 0.73_f64);
  let x = h * 6.0;
  let i = x.floor();
  let f = x - i;
  let p = v * (1.0 - s);
  let q = v * (1.0 - s * f);
  let t = v * (1.0 - s * (1.0 - f));
  let (r, g, b) = match i as i64 % 6 {
    0 => (v, t, p),
    1 => (q, v, p),
    2 => (p, v, t),
    3 => (p, q, v),
    4 => (t, p, v),
    _ => (v, p, q),
  };
  call(
    "RGBColor",
    vec![Expr::Real(r), Expr::Real(g), Expr::Real(b)],
  )
}

/// Decimal places of a step such as `2.5` or `0.25` (at most 6).
fn decimals(step: f64) -> usize {
  (0..=6)
    .find(|&d| {
      let scaled = step * 10f64.powi(d as i32);
      (scaled - scaled.round()).abs() < 1e-9
    })
    .unwrap_or(6)
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
  let values = items(positional[0]);
  let references = items(positional[1]);
  let range = items(positional[2]);
  let numeric = |es: &[Expr]| es.iter().all(|e| try_eval_to_f64(e).is_some());
  if !numeric(&values) || !numeric(&references) || !numeric(&range) {
    return Ok(None);
  }
  if range.len() < 2 {
    return Ok(None);
  }
  let (lo, hi) = (range[0].clone(), range[range.len() - 1].clone());
  let (lo_f, hi_f) = (
    try_eval_to_f64(&lo).unwrap_or(0.0),
    try_eval_to_f64(&hi).unwrap_or(1.0),
  );
  if hi_f.partial_cmp(&lo_f) != Some(std::cmp::Ordering::Greater) {
    return Ok(None);
  }

  let mut styles: Vec<Expr> = Vec::new();
  let mut label = None;
  let mut divisions = None;
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
    let value = || eval(replacement.as_ref().clone());
    match pattern.as_ref() {
      Expr::Identifier(k) if k == "GaugeStyle" => styles = items(&value()),
      Expr::Identifier(k) if k == "GaugeLabels" => {
        let v = value();
        label = matches!(v, Expr::String(_)).then_some(v);
      }
      Expr::Identifier(k) if k == "ScaleDivisions" => {
        divisions = match &value() {
          Expr::Integer(n) if *n > 0 => Some(*n),
          Expr::List(items) => match items.first() {
            Some(Expr::Integer(n)) if *n > 0 => Some(*n),
            _ => None,
          },
          _ => None,
        }
      }
      _ => {}
    }
  }

  let src = expr_to_source_form;
  // `(x - min)/(max - min)` exactly, and as a machine number.
  let fraction = |x: &Expr| {
    eval(call(
      "Times",
      vec![
        call("Subtract", vec![x.clone(), lo.clone()]),
        call(
          "Power",
          vec![
            call("Subtract", vec![hi.clone(), lo.clone()]),
            Expr::Integer(-1),
          ],
        ),
      ],
    ))
  };
  let machine = |e: Expr| eval(call("N", vec![e]));

  // The qualitative ranges.
  let inner: Vec<Expr> = if range.len() == 2 {
    vec![Expr::Real(0.5), Expr::Real(0.75)]
  } else {
    range[1..range.len() - 1].iter().map(fraction).collect()
  };
  let mut edges = vec![Expr::Integer(0)];
  edges.extend(inner);
  edges.push(Expr::Integer(1));
  let bands: Vec<String> = edges
    .windows(2)
    .enumerate()
    .map(|(i, w)| {
      format!(
        "{{Directive[{}], Rectangle[{{{}, 0}}, {{{}, 0.1}}, \"RoundingRadius\" -> 0]}}",
        BAND_GRAYS[i % 3],
        src(&w[0]),
        src(&w[1])
      )
    })
    .collect();

  // The scale: tick marks and their labels.
  let ticks: Vec<(Expr, Expr)> = if let Some(n) = divisions {
    {
      let step = (hi_f - lo_f) / n as f64;
      let d = decimals(step);
      (0..=n)
        .map(|k| {
          let value = eval(call(
            "Plus",
            vec![
              lo.clone(),
              call(
                "Times",
                vec![
                  Expr::Integer(k),
                  call(
                    "Times",
                    vec![
                      call("Subtract", vec![hi.clone(), lo.clone()]),
                      call("Power", vec![Expr::Integer(n), Expr::Integer(-1)]),
                    ],
                  ),
                ],
              ),
            ],
          ));
          let text = if k == 0 || d == 0 {
            value.clone()
          } else {
            call(
              "NumberForm",
              vec![
                machine(value.clone()),
                Expr::List(
                  vec![
                    Expr::Identifier("Infinity".into()),
                    Expr::Integer(d as i128),
                  ]
                  .into(),
                ),
              ],
            )
          };
          (machine(fraction(&value)), text)
        })
        .collect()
    }
  } else {
    {
      let divs = eval(call(
        "FindDivisions",
        vec![
          Expr::List(vec![lo.clone(), hi.clone()].into()),
          Expr::Integer(5),
        ],
      ));
      items(&divs)
        .into_iter()
        .filter(|t| {
          try_eval_to_f64(t)
            .is_some_and(|v| v >= lo_f - 1e-12 && v <= hi_f + 1e-12)
        })
        .map(|t| (machine(fraction(&t)), t))
        .collect()
    }
  };
  let tick_lines: Vec<String> = ticks
    .iter()
    .map(|(x, _)| {
      let x = src(x);
      format!("{{{{{x}, 0.}}, Scaled[{{0., -0.06}}, {{{x}, 0.}}]}}")
    })
    .collect();
  let tick_texts: Vec<String> = ticks
    .iter()
    .map(|(x, t)| {
      format!(
        "Text[{}, Offset[{{0., -3.}}, Scaled[{{0., -0.06}}, {{{}, 0.}}]], {{0., 1.}}, {{1, 0}}]",
        src(t),
        src(x)
      )
    })
    .collect();

  let label = label.map_or_else(String::new, |l| {
    format!(
      "Inset[Style[Style[DisplayForm[FormBox[{}, TraditionalForm]], {GRAY_50}], {FONT}], {{0.5000000000000001, -0.12}}, Top, Automatic]",
      src(&l)
    )
  });

  // The values, longest bar first so every one shows.
  let mut order: Vec<usize> = (0..values.len()).collect();
  order.sort_by(|&a, &b| {
    try_eval_to_f64(&values[b])
      .unwrap_or(0.0)
      .total_cmp(&try_eval_to_f64(&values[a]).unwrap_or(0.0))
  });
  let clamp = |v: &Expr| {
    let x = try_eval_to_f64(v).unwrap_or(0.0);
    if x < lo_f {
      machine(lo.clone())
    } else if x > hi_f {
      machine(hi.clone())
    } else {
      v.clone()
    }
  };
  let mut marks: Vec<String> = Vec::new();
  for (rank, &i) in order.iter().enumerate() {
    let v = clamp(&values[i]);
    let color = styles
      .get(rank % styles.len().max(1))
      .map_or_else(|| "Hue[0, 1, 0.8]".to_string(), src);
    marks.push(format!(
      "Rotate[{{Tooltip[{{Directive[{color}], Rectangle[{{-0.06875, 0}}, {{-0.03125, {}}}, \"RoundingRadius\" -> 0]}}, {}], {{}}}}, -1/2*Pi, {{0, 0}}]",
      src(&fraction(&v)),
      src(&v)
    ));
  }
  for (j, r) in references.iter().enumerate() {
    let color = src(&series_color(values.len() + j));
    let style = format!("Directive[{color}, Opacity[0.5]]");
    marks.push(format!(
      "{{{style}, GeometricTransformation[{{{{}}, {style}, {{}}, \"MarkerRotation\" -> Pi/2, Tooltip[Rectangle[{{-0.5, -0.075}}, {{0.5, 0.075}}], {}]}}, {{{{{{0., -0.07499999999999996}}, {{0.07500000000000001, 0.}}}}, {{{}, 0.05}}}}]}}",
      src(r),
      src(&machine(fraction(r)))
    ));
  }

  let source = format!(
    "Graphics[{{{{}}, {{}}, Rotate[{{{{{bands}}}}}, 0, {{0, 0}}], \
     {{Style[{{Antialiasing -> False, Style[Line[{{{{0., 0.}}, {{1., 0.}}}}], {GRAY_25}], \
     Style[Style[{{{{Style[Line[{{{tick_lines}}}], {{}}], Style[Line[{{}}], {{}}]}}, \
     Style[Style[{{{{Style[{{{tick_texts}}}, {{}}, {{{GRAY_50}, {GRAY_50}}}], Style[{{}}, {{}}, {{{GRAY_50}, {GRAY_50}}}]}}, {{}}}}, {{{GRAY_50}, {GRAY_50}}}], {FONT}]}}, \
     {{Directive[CapForm[\"Butt\"], AbsoluteThickness[1], GrayLevel[0.5]], Directive[CapForm[\"Butt\"], AbsoluteThickness[0.5], GrayLevel[0.5]]}}], {GRAY_25}]}}, \
     Directive[AbsoluteThickness[1], GrayLevel[0.5]]], {{{{}}, {{}}, {{}}}}}}, {{{label}}}, {{{marks}}}, {{}}, Rotate[{{}}, 0, {{0, 0}}]}}, \
     {{DisplayFunction -> Identity, PlotInteractivity :> $PlotInteractivity, DefaultBaseStyle -> {{\"PlotGraphics\", \"Graphics\"}}, \
     AlignmentPoint -> Center, AspectRatio -> Automatic, Axes -> False, AxesLabel -> None, AxesOrigin -> Automatic, AxesStyle -> {GRAY_25}, \
     Background -> None, BaselinePosition -> Automatic, BaseStyle -> {{}}, ColorOutput -> Automatic, ContentSelectable -> Automatic, \
     CoordinatesToolOptions -> Automatic, DisplayFunction -> Identity, Epilog -> {{}}, FormatType -> TraditionalForm, Frame -> False, \
     FrameLabel -> None, FrameStyle -> {{}}, FrameTicks -> Automatic, FrameTicksStyle -> {{}}, GridLines -> None, GridLinesStyle -> {{}}, \
     ImageMargins -> 0., ImagePadding -> All, ImageSize -> Small, ImageSizeRaw -> Automatic, LabelStyle -> {GRAY_50}, \
     Method -> {{\"BulletGauge\", \"NumericDisplayLocation\" -> Above, \"MarkerBackground\" -> None, \"DefaultBoundaryStyle\" -> Automatic, \
     \"DefaultGraphicsInteraction\" -> {{\"Version\" -> 1.2, \"TrackMousePosition\" -> {{True, False}}, \"Effects\" -> {{\"Highlight\" -> {{\"ratio\" -> 2}}, \
     \"HighlightPoint\" -> {{\"ratio\" -> 2}}, \"Droplines\" -> {{\"freeformCursorMode\" -> True, \"placement\" -> {{\"x\" -> \"All\", \"y\" -> \"None\"}}}}}}}}, \
     \"DefaultPlotStyle\" -> Automatic}}, PlotInteractivity -> Automatic, PlotLabel -> None, PlotRange -> All, PlotRangeClipping -> False, \
     PlotRangePadding -> Automatic, PlotRegion -> Automatic, PreserveImageOptions -> Automatic, Prolog -> {{}}, RotateLabel -> True, \
     Ticks -> Automatic, TicksStyle -> {{}}, Axes -> False, Method -> Automatic}}]",
    bands = bands.join(", "),
    tick_lines = tick_lines.join(", "),
    tick_texts = tick_texts.join(", "),
    marks = marks.join(", "),
  );
  let expr = crate::syntax::string_to_expr(&source)?;
  Ok(Some(evaluate_expr_to_expr(&expr)?))
}
