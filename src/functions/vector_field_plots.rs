//! The legacy `VectorFieldPlots\`` package (Wolfram Language 6–8), which
//! Demonstrations from before `VectorPlot` still load. Its plots are plain
//! `Graphics[{Thickness[Small], Arrowheads[0.02], arrows}, opts]`, one
//! `{color…, Arrow[{base, tip}]}` (or `Point[base]` for a zero vector) per
//! sampled vector, and notebooks take them apart by position — so this
//! follows the package source rather than the modern `ListVectorPlot`.

use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;
use crate::helpers::{call, call1, id_expr};
use crate::syntax::Expr;

const CTX: &str = "VectorFieldPlots`";

/// The symbols the package exports, created when it is loaded so their bare
/// names read as the package's.
pub const EXPORTS: &[&str] = &[
  "GradientFieldPlot",
  "HamiltonianFieldPlot",
  "ListVectorFieldPlot",
  "MaxArrowLength",
  "PolyaFieldPlot",
  "ScaleFactor",
  "ScaleFunction",
  "VectorFieldPlot",
];

/// Dispatch for `VectorFieldPlots\`name[args]`; `None` for other names.
pub fn vector_field_plots_ast(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  let short = name.strip_prefix(CTX)?;
  let result = match short {
    "ListVectorFieldPlot" => list_vector_field_plot(args),
    "VectorFieldPlot" => vector_field_plot(args, Vec::new()),
    "GradientFieldPlot" | "HamiltonianFieldPlot" => {
      derived_field_plot(args, short == "HamiltonianFieldPlot")
    }
    "PolyaFieldPlot" => polya_field_plot(args),
    _ => return None,
  };
  Some(
    result
      .map(|r| r.unwrap_or_else(|| crate::helpers::unevaluated(name, args))),
  )
}

/// `opts` flattened (nested option lists are spliced in), or `None` if some
/// entry is not a rule.
fn flat_options(opts: &[Expr]) -> Option<Vec<Expr>> {
  let mut out = Vec::new();
  for o in opts {
    match o {
      Expr::List(items) => out.extend(flat_options(items)?),
      _ if crate::evaluator::dispatch::list_operations::as_rule(o)
        .is_some() =>
      {
        out.push(o.clone());
      }
      _ => return None,
    }
  }
  Some(out)
}

fn short_name(e: &Expr) -> Option<&str> {
  match e {
    Expr::Identifier(n) => Some(n.rsplit('`').next().unwrap_or(n)),
    _ => None,
  }
}

/// The first value given for option `name` (in any context).
fn option<'a>(opts: &'a [Expr], name: &str) -> Option<&'a Expr> {
  opts.iter().find_map(|o| {
    crate::evaluator::dispatch::list_operations::as_rule(o)
      .filter(|(k, _)| short_name(k) == Some(name))
      .map(|(_, v)| v)
  })
}

fn is_symbol(e: Option<&Expr>, name: &str) -> bool {
  matches!(e, Some(Expr::Identifier(n)) if n == name)
}

fn real_pair(e: &Expr) -> Option<(f64, f64)> {
  let Expr::List(items) = e else { return None };
  match items.as_slice() {
    [x, y] => Some((try_eval_to_f64(x)?, try_eval_to_f64(y)?)),
    _ => None,
  }
}

fn pt((x, y): (f64, f64)) -> Expr {
  Expr::List(vec![Expr::Real(x), Expr::Real(y)].into())
}

fn apply(f: &Expr, x: Expr) -> Result<Expr, InterpreterError> {
  evaluate_expr_to_expr(&Expr::CurriedCall {
    func: Box::new(f.clone()),
    args: vec![x],
  })
}

/// `ListVectorFieldPlot[{{base, vector}, …}, opts]` or
/// `ListVectorFieldPlot[matrix of vectors, opts]` (bases on the integer grid,
/// the first row at the top).
fn list_vector_field_plot(
  args: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  let Some((data, opts)) = args.split_first() else {
    return Ok(None);
  };
  let Some(opts) = flat_options(opts) else {
    return Ok(None);
  };
  let Expr::List(items) = data else {
    return lpvf();
  };
  let is_pair = |e: &Expr| {
    matches!(e, Expr::List(p) if p.len() == 2
      && p.iter().all(|q| matches!(q, Expr::List(c) if c.len() == 2)))
  };
  let pairs: Vec<((f64, f64), (f64, f64))> =
    if !items.is_empty() && items.iter().all(is_pair) {
      items
        .iter()
        .filter_map(|e| {
          let Expr::List(p) = e else { return None };
          Some((real_pair(&p[0])?, real_pair(&p[1])?))
        })
        .collect()
    } else if !items.is_empty()
      && items.iter().all(|row| {
        matches!(row, Expr::List(r) if r.iter().all(|v| matches!(v, Expr::List(_))))
      })
    {
      // `MapIndexed[{Reverse[#2], #1} &, Reverse[vects], {2}]`.
      let mut out = Vec::new();
      for (i, row) in items.iter().rev().enumerate() {
        let Expr::List(row) = row else { continue };
        for (j, v) in row.iter().enumerate() {
          if let Some(v) = real_pair(v) {
            out.push((((j + 1) as f64, (i + 1) as f64), v));
          }
        }
      }
      out
    } else {
      return lpvf();
    };
  if pairs.is_empty() {
    return Ok(None);
  }
  draw(&pairs, &opts).map(Some)
}

fn lpvf() -> Result<Option<Expr>, InterpreterError> {
  crate::emit_message(
    "ListVectorFieldPlot::lpvf: ListVectorFieldPlot requires a rectangular \
     array of vectors or a list of {base, vector} pairs.",
  );
  Ok(None)
}

/// The body of the package's `ListVectorFieldPlot`: colors by relative
/// magnitude, the `ScaleFunction` rescaling, the `MaxArrowLength` cut and
/// the `ScaleFactor` (the longest arrow's length) applied in that order.
fn draw(
  pairs: &[((f64, f64), (f64, f64))],
  opts: &[Expr],
) -> Result<Expr, InterpreterError> {
  let none = |o: Option<&Expr>| o.is_none_or(|e| is_symbol(Some(e), "None"));
  let mut vectors: Vec<(f64, f64)> = pairs.iter().map(|p| p.1).collect();
  let mut mags: Vec<f64> = vectors
    .iter()
    .map(|(x, y)| (x * x + y * y).sqrt())
    .collect();

  let color_function = option(opts, "ColorFunction")
    .filter(|_| !none(option(opts, "ColorFunction")));
  let colors: Vec<Option<Expr>> = match color_function {
    None => vec![None; mags.len()],
    Some(f) => {
      let min = mags.iter().copied().fold(f64::INFINITY, f64::min);
      if mags.iter().all(|m| *m == mags[0]) {
        let c = apply(f, Expr::Integer(0))?;
        vec![Some(c); mags.len()]
      } else {
        let span = mags.iter().map(|m| m - min).fold(0.0, f64::max);
        mags
          .iter()
          .map(|m| apply(f, Expr::Real((m - min) / span)).map(Some))
          .collect::<Result<_, _>>()?
      }
    }
  };

  if let Some(f) = option(opts, "ScaleFunction")
    .filter(|_| !none(option(opts, "ScaleFunction")))
  {
    for (v, m) in vectors.iter_mut().zip(mags.iter_mut()) {
      let scaled = if *m == 0.0 {
        Some(0.0)
      } else {
        try_eval_to_f64(&apply(f, Expr::Real(*m))?)
      };
      match scaled {
        Some(s) if *m != 0.0 => {
          *v = (v.0 * s / *m, v.1 * s / *m);
          *m = s;
        }
        _ => {
          *v = (0.0, 0.0);
          *m = 0.0;
        }
      }
    }
  }

  let max_length = option(opts, "MaxArrowLength").and_then(try_eval_to_f64);
  let scale = match option(opts, "ScaleFactor").and_then(try_eval_to_f64) {
    Some(s) => s / mags.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    None => 1.0,
  };

  let mut arrows = Vec::new();
  for (i, &(base, _)) in pairs.iter().enumerate() {
    if max_length.is_some_and(|l| mags[i] > l) {
      continue;
    }
    let (vx, vy) = (scale * vectors[i].0, scale * vectors[i].1);
    let mut item = Vec::new();
    match &colors[i] {
      Some(Expr::List(c)) => item.extend(c.iter().cloned()),
      Some(c) => item.push(c.clone()),
      None => {}
    }
    item.push(if vx == 0.0 && vy == 0.0 {
      call1("Point", pt(base))
    } else {
      call1(
        "Arrow",
        Expr::List(vec![pt(base), pt((base.0 + vx, base.1 + vy))].into()),
      )
    });
    arrows.push(Expr::List(item.into()));
  }

  let own = [
    "ScaleFactor",
    "ScaleFunction",
    "MaxArrowLength",
    "ColorFunction",
    "PlotPoints",
  ];
  let mut graphics_opts: Vec<Expr> = opts
    .iter()
    .filter(|o| {
      crate::evaluator::dispatch::list_operations::as_rule(o)
        .and_then(|(k, _)| short_name(k))
        .is_some_and(|k| !own.contains(&k))
    })
    .cloned()
    .collect();
  for (name, value) in [("AspectRatio", "Automatic"), ("PlotRange", "All")] {
    if option(&graphics_opts, name).is_none() {
      graphics_opts
        .push(crate::helpers::rule_expr(id_expr(name), id_expr(value)));
    }
  }
  let mut parts = vec![Expr::List(
    vec![
      call1("Thickness", id_expr("Small")),
      call1("Arrowheads", Expr::Real(0.02)),
      Expr::List(arrows.into()),
    ]
    .into(),
  )];
  parts.extend(graphics_opts);
  Ok(call("Graphics", parts))
}

/// `VectorFieldPlot[f, {u, u0, u1[, du]}, {v, v0, v1[, dv]}, opts]`: `f`
/// sampled on the grid (`PlotPoints -> n` spaces n points per side, else the
/// step `du` or 15 points), the longest arrow as long as the smaller step
/// unless `ScaleFactor` says otherwise. `extra` are options appended after
/// the caller's (the defaults of a wrapper such as `PolyaFieldPlot`).
fn vector_field_plot(
  args: &[Expr],
  extra: Vec<Expr>,
) -> Result<Option<Expr>, InterpreterError> {
  let [f, Expr::List(xs), Expr::List(ys), rest @ ..] = args else {
    return Ok(None);
  };
  let Some(mut opts) = flat_options(rest) else {
    return Ok(None);
  };
  opts.extend(extra);
  let (Some(xs), Some(ys)) = (axis(xs)?, axis(ys)?) else {
    return Ok(None);
  };
  let (xpp, ypp) = match option(&opts, "PlotPoints") {
    Some(Expr::List(pp)) if !pp.is_empty() => {
      (pp.first().cloned(), pp.last().cloned())
    }
    Some(p) => (Some(p.clone()), Some(p.clone())),
    None => (None, None),
  };
  let step = |(_, lo, hi, d): &(Expr, Expr, Expr, Option<Expr>),
              pp: &Option<Expr>| {
    let span = call("Subtract", vec![hi.clone(), lo.clone()]);
    let expr = match pp {
      Some(Expr::Integer(n)) => {
        call("Divide", vec![span, Expr::Integer(n - 1)])
      }
      _ => match d {
        Some(d) if !is_symbol(Some(d), "Automatic") => d.clone(),
        _ => call("Divide", vec![span, Expr::Integer(14)]),
      },
    };
    evaluate_expr_to_expr(&expr)
  };
  let dx = step(&xs, &xpp)?;
  let dy = step(&ys, &ypp)?;
  let sf = match option(&opts, "ScaleFactor") {
    Some(s) if is_symbol(Some(s), "None") || try_eval_to_f64(s).is_some() => {
      s.clone()
    }
    _ => evaluate_expr_to_expr(&call1(
      "N",
      call("Min", vec![dx.clone(), dy.clone()]),
    ))?,
  };
  let table = call(
    "Table",
    vec![
      Expr::List(
        vec![
          call1("N", Expr::List(vec![xs.0.clone(), ys.0.clone()].into())),
          call1("N", f.clone()),
        ]
        .into(),
      ),
      Expr::List(vec![xs.0, xs.1, xs.2, dx].into()),
      Expr::List(vec![ys.0, ys.1, ys.2, dy].into()),
    ],
  );
  let table = evaluate_expr_to_expr(&table)?;
  let Expr::List(rows) = &table else {
    return Ok(None);
  };
  let mut samples = Vec::new();
  for row in rows {
    let Expr::List(row) = row else {
      return Ok(None);
    };
    for s in row {
      match s {
        Expr::List(p)
          if p.len() == 2 && p.iter().all(|q| matches!(q, Expr::List(_))) =>
        {
          samples.push(s.clone());
        }
        _ => return Ok(None),
      }
    }
  }
  let mut list_args = vec![Expr::List(samples.into())];
  list_args.push(crate::helpers::rule_expr(id_expr("ScaleFactor"), sf));
  list_args.extend(opts);
  list_vector_field_plot(&list_args)
}

/// `{u, u0, u1[, du]}` with numeric bounds.
#[allow(clippy::type_complexity)]
fn axis(
  spec: &[Expr],
) -> Result<Option<(Expr, Expr, Expr, Option<Expr>)>, InterpreterError> {
  let (var, lo, hi, d) = match spec {
    [v, lo, hi] => (v, lo, hi, None),
    [v, lo, hi, d] => (v, lo, hi, Some(d.clone())),
    _ => return Ok(None),
  };
  let lo = evaluate_expr_to_expr(lo)?;
  let hi = evaluate_expr_to_expr(hi)?;
  if try_eval_to_f64(&lo).is_none() || try_eval_to_f64(&hi).is_none() {
    return Ok(None);
  }
  Ok(Some((var.clone(), lo, hi, d)))
}

/// `GradientFieldPlot[f, …]` plots `{D[f, u], D[f, v]}`,
/// `HamiltonianFieldPlot[f, …]` plots `{D[f, v], -D[f, u]}`.
fn derived_field_plot(
  args: &[Expr],
  hamiltonian: bool,
) -> Result<Option<Expr>, InterpreterError> {
  let [f, Expr::List(xs), Expr::List(ys), ..] = args else {
    return Ok(None);
  };
  let (Some(u), Some(v)) = (xs.first(), ys.first()) else {
    return Ok(None);
  };
  let d = |var: &Expr| call("D", vec![f.clone(), var.clone()]);
  let field = if hamiltonian {
    vec![d(v), call("Times", vec![Expr::Integer(-1), d(u)])]
  } else {
    vec![d(u), d(v)]
  };
  let field = evaluate_expr_to_expr(&Expr::List(field.into()))?;
  let mut new_args = vec![field];
  new_args.extend(args[1..].iter().cloned());
  vector_field_plot(&new_args, Vec::new())
}

/// `PolyaFieldPlot[f, …]`: the field `{Re[f], -Im[f]}`, its magnitudes
/// scaled by `Log[# + 1] &` unless the caller gives a `ScaleFunction`.
fn polya_field_plot(args: &[Expr]) -> Result<Option<Expr>, InterpreterError> {
  let Some((f, rest)) = args.split_first() else {
    return Ok(None);
  };
  let field = Expr::List(
    vec![
      call1("Re", f.clone()),
      call("Times", vec![Expr::Integer(-1), call1("Im", f.clone())]),
    ]
    .into(),
  );
  let mut new_args = vec![field];
  new_args.extend(rest.iter().cloned());
  let log1p = crate::syntax::string_to_expr("Log[# + 1] &")?;
  vector_field_plot(
    &new_args,
    vec![crate::helpers::rule_expr(id_expr("ScaleFunction"), log1p)],
  )
}
