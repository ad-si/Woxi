//! `RulePlot[CellularAutomaton[spec]]` — the picture of a cellular
//! automaton's rule: one small diagram per neighborhood, the cells of the
//! neighborhood in a row with the cell they produce centred underneath.

use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::helpers::{call, call1};
use crate::syntax::Expr;

/// Largest number of neighborhoods drawn; beyond this the picture would be
/// unreadably wide (and costly to compute), so the call stays a placeholder.
const MAX_CASES: usize = 4096;

/// `(rule, colors, radius, totalistic)` of a numeric `CellularAutomaton[spec]`
/// rule. `spec` is `n`, `{n, k, r}`, or `{n, {k, …}, r}`; `k` defaults to 2
/// and `r` to 1. A `{k, 1}` colour entry marks a totalistic rule, whose
/// result depends only on the sum of the neighborhood.
fn parse_rule_spec(spec: &Expr) -> Option<(Expr, usize, usize, bool)> {
  let small = |e: &Expr| match e {
    Expr::Integer(n) if (1..=64).contains(n) => Some(*n as usize),
    _ => None,
  };
  match spec {
    Expr::Integer(_) => Some((spec.clone(), 2, 1, false)),
    Expr::List(items) if (1..=3).contains(&items.len()) => {
      let k = match items.get(1) {
        None => 2,
        Some(Expr::List(inner)) => small(inner.first()?)?,
        Some(e) => small(e)?,
      };
      let r = match items.get(2) {
        None => 1,
        Some(Expr::List(inner)) if inner.len() == 1 => small(&inner[0])?,
        Some(e) => small(e)?,
      };
      let totalistic = matches!(
        items.get(1),
        Some(Expr::List(inner))
          if inner.len() == 2 && matches!(inner[1], Expr::Integer(1))
      );
      matches!(items[0], Expr::Integer(_))
        .then(|| (items[0].clone(), k, r, totalistic))
    }
    _ => None,
  }
}

/// The graphics for a `RulePlot` of a cellular automaton, or `None` when
/// the first argument is not a numeric `CellularAutomaton[spec]` rule.
pub fn rule_plot_graphics(
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  let Expr::FunctionCall { name, args: ca } = &args[0] else {
    return None;
  };
  if name == "SubstitutionSystem" && ca.len() == 1 && args.len() == 1 {
    return substitution_system_plot(&ca[0]);
  }
  if name != "CellularAutomaton" || ca.len() != 1 {
    return None;
  }
  let (_, k, r, totalistic) = parse_rule_spec(&ca[0])?;
  let width = 2 * r + 1;
  // A totalistic rule has one case per neighborhood sum, not per pattern.
  let cases = if totalistic {
    (k - 1) * width + 1
  } else {
    k.checked_pow(width as u32).filter(|c| *c <= MAX_CASES)?
  };

  let mut color_rules: Option<Expr> = None;
  let mut graphics_opts: Vec<Expr> = Vec::new();
  for opt in &args[1..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && let Expr::Identifier(n) = pattern.as_ref()
    {
      match n.as_str() {
        "ColorRules" => {
          color_rules = evaluate_expr_to_expr(replacement).ok();
        }
        "ImageSize" => graphics_opts.push(opt.clone()),
        _ => {}
      }
    }
  }

  let color_of = |v: usize| -> Expr {
    if let Some(rules) = &color_rules
      && let Ok(c) = crate::evaluator::pattern_matching::apply_replace_all_ast(
        &Expr::Integer(v as i128),
        rules,
      )
      && crate::functions::graphics::parse_color(&c).is_some()
    {
      return c;
    }
    let level = 1.0 - v as f64 / (k - 1).max(1) as f64;
    call1("GrayLevel", Expr::Real(level))
  };
  let pt =
    |x: f64, y: f64| Expr::List(vec![Expr::Real(x), Expr::Real(y)].into());
  let cell = |x: f64, y: f64, v: usize| -> Vec<Expr> {
    vec![
      color_of(v),
      call("Rectangle", vec![pt(x, y), pt(x + 1.0, y + 1.0)]),
    ]
  };

  let mut prims: Vec<Expr> =
    vec![call1("EdgeForm", call1("GrayLevel", Expr::Real(0.6)))];
  let case_pitch = width as f64 + 1.0;
  for (slot, code) in (0..cases).rev().enumerate() {
    // Neighborhood digits, most significant (leftmost cell) first.
    let mut digits = vec![0usize; width];
    let mut rest = code;
    if totalistic {
      // Representative neighborhood: the sum spent on the largest colours
      // first, leftmost.
      for d in &mut digits {
        *d = rest.min(k - 1);
        rest -= *d;
      }
    } else {
      for d in digits.iter_mut().rev() {
        *d = rest % k;
        rest /= k;
      }
    }
    // The canonical `CellularAutomaton` evaluates the rule on a cyclic
    // row exactly one neighborhood wide, so its centre cell is the result.
    let init = Expr::List(
      digits
        .iter()
        .map(|d| Expr::Integer(*d as i128))
        .collect::<Vec<_>>()
        .into(),
    );
    let stepped = evaluate_expr_to_expr(&call(
      "CellularAutomaton",
      vec![ca[0].clone(), init, Expr::Integer(1)],
    ))
    .ok()?;
    let result = match &stepped {
      Expr::List(rows) => match rows.last() {
        Some(Expr::List(cells)) => match cells.get(r) {
          Some(Expr::Integer(v)) if (0..k as i128).contains(v) => *v as usize,
          _ => return None,
        },
        _ => return None,
      },
      _ => return None,
    };
    let x0 = slot as f64 * case_pitch;
    for (i, d) in digits.iter().enumerate() {
      prims.extend(cell(x0 + i as f64, 1.0, *d));
    }
    prims.extend(cell(x0 + r as f64, 0.0, result));
  }

  let mut gargs = vec![Expr::List(prims.into())];
  gargs.extend(graphics_opts);
  Some(crate::functions::graphics::show_ast(&[call(
    "Graphics", gargs,
  )]))
}

/// `n/2` as exact Wolfram source (`3/2`, `-1`, `0`).
fn half(n: i64) -> String {
  if n % 2 == 0 {
    (n / 2).to_string()
  } else {
    format!("{n}/2")
  }
}

/// `RulePlot[SubstitutionSystem[{v -> rhs, …}]]`, as wolframscript draws
/// it: one inset per rule — the cell, an arrow and the block it becomes (a
/// 2D rule), or the cell above the row it becomes (a 1D rule) — side by
/// side between light grey rules. Colours run from white (0) to black (the
/// largest value).
fn substitution_system_plot(
  rules: &Expr,
) -> Option<Result<Expr, InterpreterError>> {
  let Expr::List(rules) = rules else {
    return None;
  };
  let value = |e: &Expr| match e {
    Expr::Integer(v) if (0..=255).contains(v) => Some(*v as i64),
    _ => None,
  };
  let mut parsed: Vec<(i64, Vec<Vec<i64>>, bool)> = Vec::new();
  for rule in rules {
    let Expr::Rule {
      pattern,
      replacement,
    } = rule
    else {
      return None;
    };
    let lhs = value(pattern)?;
    let Expr::List(rows) = replacement.as_ref() else {
      return None;
    };
    if rows.iter().all(|r| matches!(r, Expr::List(_))) {
      let grid: Option<Vec<Vec<i64>>> = rows
        .iter()
        .map(|r| match r {
          Expr::List(cells) => cells.iter().map(value).collect(),
          _ => None,
        })
        .collect();
      parsed.push((lhs, grid?, true));
    } else {
      parsed.push((
        lhs,
        vec![rows.iter().map(value).collect::<Option<_>>()?],
        false,
      ));
    }
  }
  let (Some(first), true) = (parsed.first(), !parsed.is_empty()) else {
    return None;
  };
  let two_d = first.2;
  let (r, c) = (first.1.len() as i64, first.1.first()?.len() as i64);
  if r == 0
    || c == 0
    || parsed.iter().any(|(_, g, d)| {
      *d != two_d
        || g.len() as i64 != r
        || g.iter().any(|row| row.len() as i64 != c)
    })
  {
    return None;
  }
  let max = parsed
    .iter()
    .flat_map(|(l, g, _)| {
      std::iter::once(*l).chain(g.iter().flatten().copied())
    })
    .max()
    .unwrap_or(1)
    .max(1);
  let gray = |v: i64| {
    let (num, den) = (max - v, max);
    let g = gcd(num, den);
    let (num, den) = (num / g, den / g);
    if den == 1 {
      format!("GrayLevel[{num}]")
    } else {
      format!("GrayLevel[{num}/{den}]")
    }
  };
  let edge = "EdgeForm[GrayLevel[-1 + GoldenRatio]]";
  // The inset's plot range, in halves (2D) or quarters (1D).
  let (width, height, range) = if two_d {
    (
      (c + 4) as f64,
      (r + 1) as f64,
      format!(
        "{{{{-3/2, {}}}, {{{}, {}}}}}",
        half(2 * (2 + c) + 1),
        half(-r),
        half((r - 1) + 2 + 1)
      ),
    )
  } else {
    (
      (c + 1) as f64,
      13.0 / 4.0,
      format!("{{{{-1, {c}}}, {{-7/4, 3/2}}}}"),
    )
  };
  let s = width / height;
  let real = |x: f64| crate::syntax::expr_to_source_form(&Expr::Real(x));
  let mut insets = Vec::new();
  for (k, (lhs, grid, _)) in parsed.iter().enumerate() {
    let body = if two_d {
      let rows: Vec<String> = grid
        .iter()
        .enumerate()
        .map(|(i, row)| {
          // Row i spans y from (r - 1)/2 - i to one above.
          let y0 = (r - 1) - 2 * i as i64;
          let cells: Vec<String> = row
            .iter()
            .enumerate()
            .map(|(j, v)| {
              format!(
                "{{{}, Rectangle[{{{}, {}}}, {{{}, {}}}]}}",
                gray(*v),
                2 + j as i64,
                half(y0),
                3 + j as i64,
                half(y0 + 2)
              )
            })
            .collect();
          format!("{{{edge}, {{{}}}}}", cells.join(", "))
        })
        .collect();
      format!(
        "{{{{{{{edge}, {{{{{}, Rectangle[{{-1, 0}}, {{0, 1}}]}}}}}}}}, \
         {{Arrowheads[{{{{Automatic, Automatic, Graphics[Line[{{{{-1/2*Sqrt[3], 1/2}}, {{0, 0}}, {{-1/2*Sqrt[3], -1/2}}}}]]}}}}], Arrow[{{{{0.5, 1/2}}, {{1.5, 1/2}}}}]}}, \
         {{{}}}}}",
        gray(*lhs),
        rows.join(", ")
      )
    } else {
      let cells: Vec<String> = grid[0]
        .iter()
        .enumerate()
        .map(|(j, v)| {
          format!(
            "{{{}, Rectangle[{{{}, -5/4}}, {{{}, -1/4}}]}}",
            gray(*v),
            half(2 * j as i64 - 1),
            half(2 * j as i64 + 1)
          )
        })
        .collect();
      format!(
        "{{{{{edge}, {{{{{}, Rectangle[{{-1/2, 0}}, {{1/2, 1}}]}}}}}}, {{{edge}, {{{}}}}}}}",
        gray(*lhs),
        cells.join(", ")
      )
    };
    insets.push(format!(
      "Inset[Graphics[{body}, PlotRange -> {range}], {{{}, 1/2}}, {{Center, Center}}, {{{}, 1}}]",
      real(s * (k as f64 + 0.5)),
      real(s)
    ));
  }
  let n = parsed.len();
  let xs: Vec<String> = (0..=n)
    .map(|k| {
      if k == 0 {
        "0".to_string()
      } else {
        real(s * k as f64)
      }
    })
    .collect();
  let vlines: Vec<String> = xs
    .iter()
    .map(|x| format!("Line[{{{{{x}, 0}}, {{{x}, 1}}}}]"))
    .collect();
  let end = &xs[n];
  let source = format!(
    "Graphics[{{{{{{{}}}}}, {{GrayLevel[0.8], {{{}}}, {{Line[{{{{0, 0}}, {{{end}, 0}}}}], Line[{{{{0, 1}}, {{{end}, 1}}}}]}}}}}}]",
    insets.join(", "),
    vlines.join(", ")
  );
  Some(
    crate::syntax::string_to_expr(&source)
      .and_then(|e| evaluate_expr_to_expr(&e)),
  )
}

fn gcd(a: i64, b: i64) -> i64 {
  if b == 0 {
    a.abs().max(1)
  } else {
    gcd(b, a % b)
  }
}
