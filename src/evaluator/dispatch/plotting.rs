#[allow(unused_imports)]
use super::*;

/// Wrap a plot function call in Quiet mode so that messages emitted during
/// function sampling (e.g. Power::indet for 0^0) are suppressed and discarded.
/// This matches Wolfram Language behavior where Plot internally uses Quiet[Check[…]].
fn quiet_plot(
  f: impl FnOnce() -> Result<Expr, InterpreterError>,
) -> Result<Expr, InterpreterError> {
  let snapshot = crate::snapshot_warnings();
  crate::push_quiet();
  let result = f();
  crate::pop_quiet();
  crate::restore_warnings(snapshot);
  result
}

/// Data that is no list (once every data form the plot accepts has been
/// tried) leaves the call unevaluated with `head::lpn`, as in
/// wolframscript, rather than the plot's internal error.
fn lpn_for_non_list(
  name: &str,
  args: &[Expr],
  result: Result<Expr, InterpreterError>,
) -> Result<Expr, InterpreterError> {
  match result {
    Err(InterpreterError::EvaluationError(msg))
      if msg == "ListPlot: first argument must be a list" =>
    {
      let shown = crate::evaluator::evaluate_expr_to_expr(&args[0])
        .unwrap_or_else(|_| args[0].clone());
      crate::emit_message(&format!(
        "{name}::lpn: {} is not a list of numbers or pairs of numbers.",
        expr_to_string(&shown)
      ));
      Ok(unevaluated(name, args))
    }
    other => other,
  }
}

/// Data that is no list at all for an array plot: `head::arrayerr`
/// (issued outside the plot's own `Quiet`) and the call left unevaluated.
/// `more` extends the message for heads that also take a list of arrays.
fn not_an_array(name: &str, args: &[Expr], more: &str) -> Option<Expr> {
  if matches!(args[0], Expr::List(_)) {
    return None;
  }
  let data = crate::evaluator::evaluate_expr_to_expr(&args[0])
    .unwrap_or_else(|_| args[0].clone());
  if matches!(data, Expr::List(_)) {
    return None;
  }
  crate::emit_message(&format!(
    "{name}::arrayerr: {} must be a valid array{more}.",
    expr_to_string(&data)
  ));
  Some(unevaluated(name, args))
}

/// A plot range limit that is not a number (`Plot[f, {x, a, 1}]` with `a`
/// unset) is `head::plln` with the call left unevaluated, as in
/// wolframscript, not an aborted evaluation.
pub fn non_numeric_limit_is_plln(
  name: &str,
  args: &[Expr],
  result: Result<Expr, InterpreterError>,
) -> Result<Expr, InterpreterError> {
  let Err(InterpreterError::EvaluationError(msg)) = &result else {
    return result;
  };
  let limit_error = msg.contains("cannot evaluate")
    && ["xmin", "xmax", "iterator min", "iterator max"]
      .iter()
      .any(|w| msg.contains(w));
  if !limit_error {
    return result;
  }
  // The first iterator `{var, min, max}` with a limit that is no number.
  for spec in args.iter().skip(1) {
    let Expr::List(items) = spec else { continue };
    if items.len() < 3 {
      continue;
    }
    for limit in &items[1..3] {
      let value = crate::evaluator::evaluate_expr_to_expr(limit)
        .unwrap_or_else(|_| limit.clone());
      if crate::functions::math_ast::try_eval_to_f64(&value).is_none() {
        let spec_shown = crate::evaluator::evaluate_expr_to_expr(spec)
          .unwrap_or_else(|_| spec.clone());
        crate::emit_message(&format!(
          "{name}::plln: Limiting value {} in {} is not a machine-sized real number.",
          expr_to_string(&value),
          expr_to_string(&spec_shown)
        ));
        return Ok(unevaluated(name, args));
      }
    }
  }
  result
}

/// A function plot whose function never evaluates to a number draws
/// nothing: wolframscript returns the empty picture (`Graphics3D[{{}, {}}]`
/// for `ParametricPlot3D[pos[t], {t, 0, 5}]` with `pos` undefined), not an
/// error, so a Manipulate shows its axes and frame instead of failing.
pub fn empty_plot_for_nothing_to_draw(
  name: &str,
  args: &[Expr],
  result: Result<Expr, InterpreterError>,
) -> Result<Expr, InterpreterError> {
  let Err(InterpreterError::EvaluationError(msg)) = &result else {
    return result;
  };
  let nothing_drawn = [
    "no finite values",
    "no renderable",
    "has no points",
    "evaluates to no points",
  ]
  .iter()
  .any(|m| msg.contains(m));
  let function_plot = !name.starts_with("List") && name.contains("Plot");
  if !nothing_drawn || !function_plot {
    return result;
  }
  let options: Vec<Expr> = args
    .iter()
    .filter(|a| matches!(a, Expr::Rule { .. } | Expr::RuleDelayed { .. }))
    .cloned()
    .collect();
  let mut gargs = vec![Expr::List(
    vec![Expr::List(vec![].into()), Expr::List(vec![].into())].into(),
  )];
  gargs.extend(options);
  if name.contains("3D") {
    crate::functions::plot3d::graphics3d_ast(&gargs)
  } else {
    crate::functions::graphics::graphics_ast(&gargs)
  }
}

/// The plot heads that refuse an iterator whose endpoints coincide. Wolfram
/// reports `head::plld` for each of these and leaves the call unevaluated.
/// (`DensityPlot` prints the message under its own name but files it under
/// an internal context, so it is missing from `$MessageList` there.)
const PLLD_HEADS: &[&str] = &[
  "Plot",
  "ParametricPlot",
  "PolarPlot",
  "Plot3D",
  "ParametricPlot3D",
  "ContourPlot",
  "ContourPlot3D",
  "RegionPlot",
  "RegionPlot3D",
  "VectorPlot",
  "VectorPlot3D",
  "StreamPlot",
  "StreamDensityPlot",
  "VectorDensityPlot",
  "DensityPlot",
  "ReImPlot",
  "AbsArgPlot",
  "LogPlot",
  "LogLinearPlot",
  "LogLogPlot",
  "RevolutionPlot3D",
];

/// The head a plot reports `plld` under: the log plots are `Plot` inside,
/// and `RevolutionPlot3D` is a `ParametricPlot3D`.
fn plld_message_head(name: &str) -> &str {
  match name {
    "LogPlot" | "LogLinearPlot" | "LogLogPlot" => "Plot",
    "RevolutionPlot3D" => "ParametricPlot3D",
    other => other,
  }
}

/// A plot whose iterator names a compound variable (`{Subscript[w, f], 0,
/// 1}`, `{x[1], 0, 1}`) runs with each such variable replaced by a fresh
/// symbol, in the iterator and in the plotted expression alike. `None` when
/// every variable is already a symbol.
fn with_symbol_iterators(args: &[Expr]) -> Option<Vec<Expr>> {
  let mut out = args.to_vec();
  let mut renamed = false;
  for (k, spec) in args.iter().enumerate().skip(1) {
    let Expr::List(items) = spec else { continue };
    if items.len() < 2 {
      continue;
    }
    let var = &items[0];
    if !matches!(var, Expr::FunctionCall { .. })
      || crate::functions::predicate_ast::is_numeric_q(var)
    {
      continue;
    }
    let fresh = Expr::Identifier(format!("Plot$var{k}"));
    out[0] =
      crate::functions::polynomial_ast::substitute_expr(&out[0], var, &fresh);
    let mut items = items.to_vec();
    items[0] = fresh;
    out[k] = Expr::List(items.into());
    renamed = true;
  }
  renamed.then_some(out)
}

/// The arguments with every `opt -> (Evaluate[…] &)` option value
/// evaluated, or `None` when there is none. (A pure function whose body is
/// not wrapped in `Evaluate` evaluates to itself; other option values are
/// evaluated by each plot as it reads them.)
fn with_evaluated_options(args: &[Expr]) -> Option<Vec<Expr>> {
  let evaluate_body = |e: &Expr| match e {
    Expr::Function { body } | Expr::NamedFunction { body, .. } => {
      matches!(body.as_ref(), Expr::FunctionCall { name, .. } if name == "Evaluate")
    }
    _ => false,
  };
  let mut out = args.to_vec();
  let mut changed = false;
  for arg in out.iter_mut().skip(1) {
    let Expr::Rule {
      pattern,
      replacement,
    } = arg
    else {
      continue;
    };
    if !evaluate_body(replacement) {
      continue;
    }
    if let Ok(value) =
      crate::evaluator::evaluate_expr_to_expr(replacement.as_ref())
    {
      *arg = Expr::Rule {
        pattern: pattern.clone(),
        replacement: Box::new(value),
      };
      changed = true;
    }
  }
  changed.then_some(out)
}

/// `PlotLegends -> BarLegend[…]` on a plot gives `Legended[plot,
/// Placed[BarLegend[…], After]]`, as in Wolfram: the plot is drawn without
/// the option and the color bar set beside it by `Legended`, so every plot
/// shares the one `BarLegend` renderer.
fn with_bar_legend(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  if !(name.ends_with("Plot") || name.ends_with("Plot3D")) {
    return None;
  }
  let pos = args.iter().position(|a| {
    matches!(a, Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. }
      if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "PlotLegends"))
  })?;
  let (Expr::Rule { replacement, .. } | Expr::RuleDelayed { replacement, .. }) =
    &args[pos]
  else {
    return None;
  };
  let mut legend = match replacement.as_ref() {
    e @ Expr::FunctionCall { name, .. } if name == "BarLegend" => e.clone(),
    // A legend kept in a variable (`PlotLegends -> leg`) of a held plot.
    Expr::Identifier(_) => {
      let e = crate::evaluator::evaluate_expr_to_expr(replacement).ok()?;
      if !matches!(&e, Expr::FunctionCall { name, .. } if name == "BarLegend") {
        return None;
      }
      e
    }
    _ => return None,
  };
  if let Expr::FunctionCall {
    args: legend_args, ..
  } = &mut legend
  {
    let defaults = [
      ("LabelStyle", Expr::List(Vec::new().into())),
      ("LegendLayout", Expr::String("Column".into())),
      ("LegendMarkerSize", Expr::Integer(150)),
      ("Charting`AxisLabel", id_expr("None")),
      ("Charting`TickSide", id_expr("Right")),
      ("ColorFunctionScaling", id_expr("True")),
    ];
    let existing: Vec<&Expr> = legend_args
      .iter()
      .skip(1)
      .flat_map(|e| match e {
        Expr::List(items) => items.iter().collect::<Vec<_>>(),
        other => vec![other],
      })
      .collect();
    let options: Vec<Expr> = defaults.into_iter().filter(|(key, _)| {
      !existing.iter().any(|e| matches!(e, Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. } if matches!(pattern.as_ref(), Expr::Identifier(k) if k == key)))
    }).map(|(key, value)| Expr::Rule { pattern: Box::new(id_expr(key)), replacement: Box::new(value) }).collect();
    if let Some(Expr::List(items)) = legend_args.last_mut()
      && items
        .iter()
        .all(|e| matches!(e, Expr::Rule { .. } | Expr::RuleDelayed { .. }))
    {
      items.extend(options);
    } else {
      legend_args.push(Expr::List(options.into()));
    }
  }
  let mut rest = args.to_vec();
  rest.remove(pos);
  Some(
    crate::evaluator::evaluate_function_call_ast(name, &rest).map(|plot| {
      call(
        "Legended",
        vec![
          plot,
          call(
            "Placed",
            vec![legend, id_expr("After"), id_expr("Identity")],
          ),
        ],
      )
    }),
  )
}

pub fn dispatch_plotting(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  if PLLD_HEADS.contains(&name)
    && args.len() >= 2
    && let Some(renamed) = with_symbol_iterators(args)
  {
    return dispatch_plotting(name, &renamed);
  }
  // The function plots hold their arguments, but Wolfram evaluates their
  // options once up front: the `Evaluate[…]` in a
  // `ColorFunction -> (Evaluate[…] &)` runs once, not at every sample
  // point.
  if PLLD_HEADS.contains(&name)
    && args.len() >= 2
    && let Some(evaluated) = with_evaluated_options(args)
  {
    return dispatch_plotting(name, &evaluated);
  }
  // A range like `{x, 1, 1}` has nothing to sample. The check sits here,
  // ahead of `quiet_plot`, because the plot bodies run with messages
  // suppressed — and in one place, so every plot head refuses alike.
  if PLLD_HEADS.contains(&name)
    && args.len() >= 2
    && args.iter().skip(1).any(|a| {
      crate::functions::plot::degenerate_iterator(
        plld_message_head(name),
        a,
        name == "RegionPlot3D",
      )
    })
  {
    // RevolutionPlot3D gives up with `$Failed` rather than staying
    // unevaluated.
    if name == "RevolutionPlot3D" {
      return Some(Ok(fail_expr()));
    }
    return Some(Ok(call(name, args.to_vec())));
  }
  // `{z, r}` is the square `{z, -|r| (1 + I), |r| (1 + I)}`.
  if matches!(name, "ComplexPlot" | "ComplexPlot3D")
    && let Some(Expr::List(items)) = args.get(1)
    && items.len() == 2
    && matches!(&items[0], Expr::Identifier(_))
  {
    let half = crate::evaluator::evaluate_expr_to_expr(&times(vec![
      call1("Abs", items[1].clone()),
      plus(vec![Expr::Integer(1), id_expr("I")]),
    ]))
    .ok()?;
    let low =
      crate::evaluator::evaluate_expr_to_expr(&neg(half.clone())).ok()?;
    let mut expanded = args.to_vec();
    expanded[1] = Expr::List(vec![items[0].clone(), low, half].into());
    return dispatch_plotting(name, &expanded);
  }
  // A range spec held by the plot (all of them are HoldAll) that is not
  // written as a list is evaluated to the one it names:
  // `Plot[f, range /. settings]` with `range` a stored `{x, 0, 2}`.
  if matches!(
    name,
    "Plot"
      | "LogPlot"
      | "LogLinearPlot"
      | "LogLogPlot"
      | "ParametricPlot"
      | "PolarPlot"
      | "Plot3D"
      | "ContourPlot"
      | "DensityPlot"
      | "ParametricPlot3D"
      | "RegionPlot"
  ) && args.len() >= 2
  {
    let is_option =
      |e: &Expr| matches!(e, Expr::Rule { .. } | Expr::RuleDelayed { .. });
    // Only options after the function: no range at all.
    if is_option(&args[1]) && name == "Plot" {
      crate::emit_message(&format!(
        "{name}::argr: {name} called with 1 argument; 2 arguments are expected."
      ));
      return Some(Ok(call(name, args.to_vec())));
    }
    let mut expanded = args.to_vec();
    let mut changed = false;
    for spec in expanded.iter_mut().skip(1) {
      if is_option(spec) {
        break;
      }
      if !matches!(spec, Expr::List(_))
        && let Ok(value @ Expr::List(_)) =
          crate::evaluator::evaluate_expr_to_expr(spec)
      {
        *spec = value;
        changed = true;
      }
    }
    if changed {
      return dispatch_plotting(name, &expanded);
    }
  }
  // A complex plot's two corners must differ in both their real and their
  // imaginary part, or they span no rectangle.
  if matches!(name, "ComplexPlot" | "ComplexPlot3D")
    && args.len() >= 2
    && crate::functions::plot::degenerate_complex_corners(name, &args[1])
  {
    return Some(Ok(call(name, args.to_vec())));
  }
  if let Some(result) = with_bar_legend(name, args) {
    return Some(result);
  }
  match name {
    #[cfg(not(target_arch = "wasm32"))]
    "Run" if args.len() == 1 => {
      if let Expr::String(cmd) = &args[0] {
        use std::process::Command;
        let status = Command::new("sh").arg("-c").arg(cmd).status();
        Some(match status {
          Ok(s) => {
            let code = s.code().unwrap_or(-1) as i128;
            Ok(Expr::Integer(code * 256))
          }
          Err(e) => Err(InterpreterError::EvaluationError(format!(
            "Run: failed to execute command: {e}"
          ))),
        })
      } else {
        Some(Err(InterpreterError::EvaluationError(
          "Run expects a string argument".into(),
        )))
      }
    }
    "Plot" if args.len() >= 2 => {
      Some(quiet_plot(|| crate::functions::plot::plot_ast(args)))
    }
    // RulePlot[CellularAutomaton[rule]] draws each neighborhood and its
    // result. The other systems (substitution systems, Turing machines)
    // still produce an empty Graphics placeholder so the expected
    // `-Graphics-` output renders; anything that is not a system
    // (`RulePlot[x]`, a bare rule number) stays unevaluated, as in WL.
    "RulePlot"
      if matches!(args.first(), Some(Expr::FunctionCall { name, .. })
      if matches!(
        name.as_str(),
        "CellularAutomaton"
          | "SubstitutionSystem"
          | "TuringMachine"
          | "MultiwaySystem"
      )) =>
    {
      Some(quiet_plot(|| {
        if let Some(result) =
          crate::functions::rule_plot::rule_plot_graphics(args)
        {
          return result;
        }
        crate::functions::graphics::show_ast(&[call1(
          "Graphics",
          Expr::List(vec![].into()),
        )])
      }))
    }
    "RulePlot" if !args.is_empty() => Some(Ok(unevaluated(name, args))),
    // ReImPlot[f, {x, xmin, xmax}, opts...] plots Re[f] and Im[f] on the
    // same axes. We forward to Plot[{Re[f], Im[f]}, …]. When f is a list,
    // splat the real and imaginary parts across all entries. Falls through
    // to the symbolic form when the second argument isn't a valid plot
    // range spec.
    "ReImPlot"
      if args.len() >= 2
        && matches!(
          &args[1],
          Expr::List(items) if items.len() == 3 && matches!(&items[0], Expr::Identifier(_))
        ) =>
    {
      let reim_list = match &args[0] {
        Expr::List(items) => {
          let mut out: Vec<Expr> = Vec::with_capacity(items.len() * 2);
          for it in items {
            out.push(call1("Re", it.clone()));
            out.push(call1("Im", it.clone()));
          }
          Expr::List(out.into())
        }
        f => Expr::List(
          vec![call1("Re", f.clone()), call1("Im", f.clone())].into(),
        ),
      };
      let functions = match &args[0] {
        Expr::List(items) => items.len(),
        _ => 1,
      };
      let mut new_args = Vec::with_capacity(args.len() + 1);
      new_args.push(reim_list);
      new_args.extend(args[1..].iter().cloned());
      // Wolfram draws each function's real and imaginary parts in the
      // function's color, the imaginary part dotted.
      let styled = args[2..].iter().any(|opt| {
        crate::functions::graphics::option_name_value(opt)
          .is_some_and(|(name, _)| name == "PlotStyle")
      });
      if !styled {
        let palette = crate::functions::plot::PLOT_COLORS_RGB;
        let mut styles = Vec::with_capacity(2 * functions);
        for k in 0..functions {
          let (r, g, b) = palette[k % palette.len()];
          let color = Expr::FunctionCall {
            name: "RGBColor".into(),
            args: vec![Expr::Real(r), Expr::Real(g), Expr::Real(b)].into(),
          };
          styles.push(color.clone());
          styles.push(Expr::FunctionCall {
            name: "Directive".into(),
            args: vec![
              color,
              call1(
                "Dashing",
                Expr::List(
                  vec![Expr::Integer(0), Expr::Identifier("Small".into())]
                    .into(),
                ),
              ),
            ]
            .into(),
          });
        }
        new_args.push(Expr::Rule {
          pattern: Box::new(Expr::Identifier("PlotStyle".into())),
          replacement: Box::new(Expr::List(styles.into())),
        });
      }
      Some(quiet_plot(|| crate::functions::plot::plot_ast(&new_args)))
    }
    // ComplexContourPlot[f, {z, zmin, zmax}, opts...] is ContourPlot of f
    // over the rectangle zmin..zmax of the complex plane, with z = x + I y.
    // A real-valued f (or list of them, or equations) gives its level
    // curves directly.
    "ComplexContourPlot"
      if args.len() >= 2
        && matches!(
          &args[1],
          Expr::List(items) if items.len() == 3 && matches!(&items[0], Expr::Identifier(_))
        ) =>
    {
      let Expr::List(iter) = &args[1] else {
        unreachable!()
      };
      let Expr::Identifier(zvar) = &iter[0] else {
        unreachable!()
      };
      let corner = |e: &Expr| {
        let v = crate::evaluator::evaluate_expr_to_expr(e).ok()?;
        crate::functions::math_ast::try_extract_complex_f64(&v)
      };
      let (Some((re0, im0)), Some((re1, im1))) =
        (corner(&iter[1]), corner(&iter[2]))
      else {
        return Some(Ok(unevaluated(name, args)));
      };
      let (xs, ys) = (
        Expr::Identifier("ComplexContourPlot`x".into()),
        Expr::Identifier("ComplexContourPlot`y".into()),
      );
      let z = Expr::FunctionCall {
        name: "Plus".into(),
        args: vec![
          xs.clone(),
          Expr::FunctionCall {
            name: "Times".into(),
            args: vec![Expr::Identifier("I".into()), ys.clone()].into(),
          },
        ]
        .into(),
      };
      let body = crate::functions::plot::substitute_var(&args[0], zvar, &z);
      let range = |v: Expr, lo: f64, hi: f64| {
        Expr::List(vec![v, Expr::Real(lo), Expr::Real(hi)].into())
      };
      let mut new_args = vec![body, range(xs, re0, re1), range(ys, im0, im1)];
      new_args.extend(args[2..].iter().cloned());
      Some(quiet_plot(|| {
        crate::functions::field_plot::contour_plot_ast(&new_args)
      }))
    }
    // AbsArgPlot[f, {x, xmin, xmax}, opts...] plots Abs[f] colored by
    // Arg[f] on the cyclic "MidShiftBalancedHue" gradient (red where f is
    // positive, cyan where it is negative), as Wolfram does: it is Plot of
    // Abs[f] with that ColorFunction.
    "AbsArgPlot"
      if args.len() >= 2
        && matches!(
          &args[1],
          Expr::List(items) if items.len() == 3 && matches!(&items[0], Expr::Identifier(_))
        ) =>
    {
      let Expr::List(iter) = &args[1] else {
        unreachable!()
      };
      let var = iter[0].clone();
      let stops: Vec<Expr> =
        crate::functions::field_plot::MID_SHIFT_BALANCED_HUE
          .iter()
          .map(|&(l, a, b)| Expr::FunctionCall {
            name: "LABColor".into(),
            args: vec![Expr::Real(l), Expr::Real(a), Expr::Real(b)].into(),
          })
          .collect();
      let pi = Expr::Identifier("Pi".into());
      // (Arg[f] + Pi) / (2 Pi)
      let t = Expr::FunctionCall {
        name: "Times".into(),
        args: vec![
          Expr::FunctionCall {
            name: "Plus".into(),
            args: vec![call1("Arg", args[0].clone()), pi.clone()].into(),
          },
          Expr::FunctionCall {
            name: "Power".into(),
            args: vec![
              Expr::FunctionCall {
                name: "Times".into(),
                args: vec![Expr::Integer(2), pi].into(),
              },
              Expr::Integer(-1),
            ]
            .into(),
          },
        ]
        .into(),
      };
      let color_function = Expr::FunctionCall {
        name: "Function".into(),
        args: vec![
          Expr::List(vec![var, Expr::Identifier("AbsArgPlot`y".into())].into()),
          Expr::FunctionCall {
            name: "Blend".into(),
            args: vec![Expr::List(stops.into()), t].into(),
          },
        ]
        .into(),
      };
      let rule = |name: &str, value: Expr| Expr::Rule {
        pattern: Box::new(Expr::Identifier(name.into())),
        replacement: Box::new(value),
      };
      let mut new_args = vec![call1("Abs", args[0].clone())];
      new_args.extend(args[1..].iter().cloned());
      new_args.push(rule("ColorFunction", color_function));
      new_args.push(rule(
        "ColorFunctionScaling",
        Expr::Identifier("False".into()),
      ));
      Some(quiet_plot(|| crate::functions::plot::plot_ast(&new_args)))
    }
    // Manipulate holds its arguments (see core_eval.rs) and, in a text
    // front-end, simply echoes itself back with the body and variable
    // specs preserved. A non-list variable spec emits a Manipulate::vsform
    // message but the expression is still returned unchanged.
    "Manipulate" => Some(crate::functions::graphics::manipulate_ast(args)),
    // Animate is a Manipulate with animator controls (wolframscript
    // evaluates it to one).
    "Animate" => crate::functions::graphics::animate_to_manipulate(args).map(
      |m| match &m {
        Expr::FunctionCall { args, .. } => {
          crate::functions::graphics::manipulate_ast(args)
        }
        _ => Ok(m),
      },
    ),
    // Control holds its argument (see core_eval.rs) and echoes itself back
    // with the variable spec's bounds evaluated. Playground / Studio detect
    // the held Control[…] and render an interactive control widget.
    "Control" => Some(crate::functions::graphics::control_ast(args)),
    // GeometricScene holds its arguments (see core_eval.rs) and evaluates
    // the point-definition rules in order, leaving the primitives symbolic
    // until a `["Graphics"]` property call resolves them (see
    // apply_curried_call in function_application.rs).
    "GeometricScene" if args.len() == 2 || args.len() == 3 => {
      Some(crate::functions::graphics::geometric_scene_ast(args))
    }
    "Plot3D" if args.len() >= 3 => {
      let domain = crate::functions::fem2d::region_domain_args(args);
      let args = domain.as_deref().unwrap_or(args);
      Some(quiet_plot(|| crate::functions::plot3d::plot3d_ast(args)))
    }
    "ParametricPlot3D" if args.len() >= 2 => Some(quiet_plot(|| {
      crate::functions::plot3d::parametric_plot3d_ast(args)
    })),
    "Graphics" if !args.is_empty() => {
      // Keep Graphics as a FunctionCall during evaluation so that Show[]
      // can merge primitives from multiple Graphics expressions.
      // Rendering to SVG happens at the output stage via render_graphics_fc_if_needed.
      None
    }
    "GeoGraphics" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::geographics::geographics_ast(args)
    })),
    "GeoRegionValuePlot" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::geographics::geo_region_value_plot_ast(args)
    })),
    "GeoHistogram" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::geographics::geo_histogram_ast(args)
    })),
    "Show" if !args.is_empty() => {
      Some(crate::functions::graphics::show_ast(args))
    }
    "GraphicsRow" if !args.is_empty() => {
      Some(crate::functions::graphics::graphics_row_ast(args))
    }
    "GraphicsColumn" if !args.is_empty() => {
      Some(crate::functions::graphics::graphics_column_ast(args))
    }
    "GraphicsGrid" if !args.is_empty() => {
      Some(crate::functions::graphics::graphics_grid_ast(args))
    }
    // Overlay is a display wrapper, not a picture: `Head[Overlay[…]]` is
    // `Overlay`, and it echoes as `Overlay[{-Graphics-, …}]`. Returning
    // None keeps it symbolic; the output stage composes the stacked
    // picture (see `expr_to_svg`'s `Overlay` arm).
    "Overlay" if !args.is_empty() => None,
    "PlotGrid" if !args.is_empty() => {
      Some(crate::functions::graphics::plot_grid_ast(args))
    }
    "TreeForm" if !args.is_empty() => {
      // TreeForm stays as a wrapper in OutputForm (matching wolframscript)
      Some(Ok(unevaluated("TreeForm", args)))
    }
    "TreeGraph" if !args.is_empty() => {
      Some(crate::functions::tree_form::tree_graph_ast(args))
    }
    "Graphics3D" if !args.is_empty() => {
      Some(crate::functions::plot3d::graphics3d_ast(args))
    }
    "KochCurve" if !args.is_empty() => {
      Some(crate::functions::graphics::koch_curve_ast(args))
    }
    "HilbertCurve" if !args.is_empty() => {
      Some(crate::functions::graphics::hilbert_curve_ast(args))
    }
    "PeanoCurve" if !args.is_empty() => {
      Some(crate::functions::graphics::peano_curve_ast(args))
    }
    "SierpinskiCurve" if !args.is_empty() => {
      Some(crate::functions::graphics::sierpinski_curve_ast(args))
    }
    "LinearGradientFilling" => Some(
      crate::functions::graphics::linear_gradient_filling_ast(args),
    ),
    "DropShadowing" => {
      Some(crate::functions::graphics::drop_shadowing_ast(args))
    }
    "ListPlot3D" if !args.is_empty() => Some(
      not_an_array(name, args, " or a list of valid arrays").map_or_else(
        || quiet_plot(|| crate::functions::plot3d::list_plot3d_ast(args)),
        Ok,
      ),
    ),
    "ListSurfacePlot3D" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::plot3d::list_surface_plot3d_ast(args)
    })),
    "ListContourPlot3D" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::plot3d::list_contour_plot3d_ast(args)
    })),
    "ListPointPlot3D" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::plot3d::list_point_plot3d_ast(args)
    })),
    "ListLinePlot3D" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::plot3d::list_line_plot3d_ast(args)
    })),
    "RevolutionPlot3D" if args.len() >= 2 => Some(quiet_plot(|| {
      crate::functions::plot3d::revolution_plot3d_ast(args)
    })),
    "RegionPlot3D" if args.len() >= 4 => Some(quiet_plot(|| {
      crate::functions::plot3d::region_plot3d_ast(args)
    })),
    "SphericalPlot3D" if args.len() >= 3 => Some(quiet_plot(|| {
      crate::functions::plot3d::spherical_plot3d_ast(args)
    })),
    "ListPlot" | "ListLinePlot"
      if !args.is_empty()
        && let Some(rewritten) =
          crate::functions::list_plot::expand_highlighted_data(args) =>
    {
      Some(crate::evaluator::evaluate_function_call_ast(
        name, &rewritten,
      ))
    }
    "ListPlot" if !args.is_empty() => {
      // Data that is no list (once every data form ListPlot accepts has been
      // tried) leaves the call unevaluated with ListPlot::lpn, as in
      // wolframscript.
      Some(lpn_for_non_list(
        name,
        args,
        quiet_plot(|| crate::functions::list_plot::list_plot_ast(args)),
      ))
    }
    "DiscretePlot" if args.len() >= 2 => Some(quiet_plot(|| {
      crate::functions::list_plot::discrete_plot_ast(args)
    })),
    "DiscretePlot3D" if args.len() >= 3 => Some(
      match crate::functions::plot3d::discrete_plot3d_bad_iterator(args) {
        Some(unevaluated) => Ok(unevaluated),
        None => {
          quiet_plot(|| crate::functions::plot3d::discrete_plot3d_ast(args))
        }
      },
    ),
    "ListLinePlot" if !args.is_empty() => {
      // Match wolframscript: if the first argument doesn't evaluate
      // to a List, emit `ListLinePlot::lpn` (outside Quiet) and
      // return the call unevaluated. wolframscript on
      // `ListLinePlot[list]` prints the message and leaves the
      // expression symbolic.
      match crate::evaluator::evaluate_expr_to_expr(&args[0]) {
        // A TimeSeries / TemporalData, an Association or labeled
        // `data -> labels` is a valid (non-List) data source, and so is a display wrapper
        // (`Tooltip`, `Style`, ...) around any of those.
        Ok(evaluated)
          if {
            let stripped =
              crate::functions::list_plot::strip_plot_data_display_wrapper(
                &evaluated,
              );
            // `data -> labels` labels the points.
            !matches!(
              stripped,
              Expr::List(_) | Expr::Association(_) | Expr::Rule { .. }
            ) && crate::functions::timeseries_ast::temporal_paths(&stripped)
              .is_none()
          } =>
        {
          crate::emit_message(&format!(
            "ListLinePlot::lpn: {} is not a list of numbers or pairs of numbers.",
            expr_to_string(&evaluated)
          ));
          Some(Ok(unevaluated("ListLinePlot", args)))
        }
        _ => Some(quiet_plot(|| {
          crate::functions::list_plot::list_line_plot_ast(args)
        })),
      }
    }
    "ListFitPlot" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::list_plot::list_fit_plot_ast(args)
    })),
    "QuantilePlot" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::quantile_plot::quantile_plot_ast(args)
    })),
    "ProbabilityPlot" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::quantile_plot::probability_plot_ast(args)
    })),
    "ProbabilityScalePlot" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::quantile_plot::probability_scale_plot_ast(args)
    })),
    "AudioPlot" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::list_plot::audio_plot_ast(args)
    })),
    "ListLogPlot" if !args.is_empty() => Some(lpn_for_non_list(
      name,
      args,
      quiet_plot(|| crate::functions::list_plot::list_log_plot_ast(args)),
    )),
    "ListLogLogPlot" if !args.is_empty() => Some(lpn_for_non_list(
      name,
      args,
      quiet_plot(|| crate::functions::list_plot::list_log_log_plot_ast(args)),
    )),
    "ListLogLinearPlot" if !args.is_empty() => Some(lpn_for_non_list(
      name,
      args,
      quiet_plot(|| {
        crate::functions::list_plot::list_log_linear_plot_ast(args)
      }),
    )),
    "ListPolarPlot" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::list_plot::list_polar_plot_ast(args)
    })),
    "ListStepPlot" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::list_plot::list_step_plot_ast(args)
    })),
    "StackedListPlot" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::list_plot::stacked_list_plot_ast(args)
    })),
    "LogLogPlot" if args.len() >= 2 => Some(quiet_plot(|| {
      crate::functions::plot::log_log_plot_ast(args)
    })),
    "LogPlot" if args.len() >= 2 => {
      Some(quiet_plot(|| crate::functions::plot::log_plot_ast(args)))
    }
    "LogLinearPlot" if args.len() >= 2 => Some(quiet_plot(|| {
      crate::functions::plot::log_linear_plot_ast(args)
    })),
    "ParametricPlot" if args.len() >= 2 => Some(quiet_plot(|| {
      crate::functions::parametric_plot::parametric_plot_ast(args)
    })),
    "PolarPlot" if args.len() >= 2 => Some(quiet_plot(|| {
      crate::functions::parametric_plot::polar_plot_ast(args)
    })),
    "ComplexPlot" if args.len() >= 2 => Some(quiet_plot(|| {
      crate::functions::field_plot::complex_plot_ast(args)
    })),
    "ComplexListPlot" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::list_plot::complex_list_plot_ast(args)
    })),
    // ComplexPlot3D[f, {z, zmin, zmax}] — plot |f(x + i y)| as a 3D
    // surface over the rectangle in the complex plane. We forward to
    // Plot3D with f's modulus and reuse its rendering. Falls through
    // when the iterator isn't a valid `{var, zmin, zmax}` triple.
    "ComplexPlot3D"
      if args.len() >= 2
        && matches!(
          &args[1],
          Expr::List(items) if items.len() == 3 && matches!(&items[0], Expr::Identifier(_))
        ) =>
    {
      let (zvar, z_lo, z_hi) = match &args[1] {
        Expr::List(items) => match &items[0] {
          Expr::Identifier(name) => {
            (name.clone(), items[1].clone(), items[2].clone())
          }
          _ => unreachable!(),
        },
        _ => unreachable!(),
      };
      let xv = "x".to_string();
      let yv = "y".to_string();
      // Substitute z -> x + I*y in f.
      let xy_expr = plus2(
        Expr::Identifier(xv.clone()),
        times2(id_expr("I"), Expr::Identifier(yv.clone())),
      );
      let body = crate::syntax::substitute_variable(&args[0], &zvar, &xy_expr);
      let modulus = call1("Abs", body);
      // Plot ranges: Re(zmin)..Re(zmax) and Im(zmin)..Im(zmax).
      // Evaluate them to concrete numbers so Plot3D's iterator parser
      // accepts them.
      let eval_part = |e: &Expr, part: &str| -> Expr {
        let wrapped = call1(part, e.clone());
        crate::evaluator::evaluate_expr_to_expr(&wrapped).unwrap_or(wrapped)
      };
      let mut plot_args = vec![
        modulus,
        Expr::List(
          vec![
            Expr::Identifier(xv),
            eval_part(&z_lo, "Re"),
            eval_part(&z_hi, "Re"),
          ]
          .into(),
        ),
        Expr::List(
          vec![
            Expr::Identifier(yv),
            eval_part(&z_lo, "Im"),
            eval_part(&z_hi, "Im"),
          ]
          .into(),
        ),
      ];
      // Drop options like PlotLegends -> Automatic that Plot3D doesn't
      // model the same way; pass others through.
      for opt in &args[2..] {
        if let Expr::Rule { pattern, .. } = opt
          && matches!(&**pattern, Expr::Identifier(s) if s == "PlotLegends")
        {
          continue;
        }
        plot_args.push(opt.clone());
      }
      Some(quiet_plot(|| {
        match crate::functions::plot3d::plot3d_ast(&plot_args) {
          Ok(g) => Ok(g),
          // When the function isn't numerically evaluable (e.g. symbolic
          // special functions), fall back to an empty Graphics3D so the
          // caller still gets a placeholder, matching wolframscript's
          // `-Graphics3D-` output.
          Err(_) => crate::functions::plot3d::graphics3d_ast(&[Expr::List(
            vec![].into(),
          )]),
        }
      }))
    }
    // ComplexRegionPlot[pred, {z, zmin, zmax}] — plot the region in the
    // complex plane where pred holds. We substitute z → x + i·y and
    // forward to RegionPlot. `{z, r}` is treated as `{z, -r - r I, r + r I}`.
    "ComplexRegionPlot"
      if args.len() >= 2
        && matches!(
          &args[1],
          Expr::List(items) if (items.len() == 2 || items.len() == 3)
            && matches!(&items[0], Expr::Identifier(_))
        ) =>
    {
      let (zvar, z_lo, z_hi) = match &args[1] {
        Expr::List(items) => match &items[0] {
          Expr::Identifier(name) => {
            if items.len() == 3 {
              (name.clone(), items[1].clone(), items[2].clone())
            } else {
              // {z, r} → corners (-r - r·I, r + r·I)
              let r = items[1].clone();
              let neg_r = call1("Minus", r.clone());
              let corner_lo =
                plus2(neg_r.clone(), times2(neg_r.clone(), id_expr("I")));
              let corner_hi = plus2(r.clone(), times2(r, id_expr("I")));
              (name.clone(), corner_lo, corner_hi)
            }
          }
          _ => unreachable!(),
        },
        _ => unreachable!(),
      };
      let xv = "x".to_string();
      let yv = "y".to_string();
      let xy_expr = plus2(
        Expr::Identifier(xv.clone()),
        times2(id_expr("I"), Expr::Identifier(yv.clone())),
      );
      let pred = crate::syntax::substitute_variable(&args[0], &zvar, &xy_expr);
      let eval_part = |e: &Expr, part: &str| -> Expr {
        let wrapped = call1(part, e.clone());
        crate::evaluator::evaluate_expr_to_expr(&wrapped).unwrap_or(wrapped)
      };
      let mut region_args = vec![
        pred,
        Expr::List(
          vec![
            Expr::Identifier(xv),
            eval_part(&z_lo, "Re"),
            eval_part(&z_hi, "Re"),
          ]
          .into(),
        ),
        Expr::List(
          vec![
            Expr::Identifier(yv),
            eval_part(&z_lo, "Im"),
            eval_part(&z_hi, "Im"),
          ]
          .into(),
        ),
      ];
      for opt in &args[2..] {
        if let Expr::Rule { pattern, .. } = opt
          && matches!(&**pattern, Expr::Identifier(s) if s == "PlotLegends")
        {
          continue;
        }
        region_args.push(opt.clone());
      }
      Some(quiet_plot(|| {
        crate::functions::field_plot::region_plot_ast(&region_args)
      }))
    }
    "DensityPlot" if args.len() >= 3 => Some(quiet_plot(|| {
      let domain = crate::functions::fem2d::region_domain_args(args);
      let args = domain.as_deref().unwrap_or(args);
      crate::functions::field_plot::density_plot_ast(
        &crate::functions::field_plot::with_evaluated_options(args, 3)?,
      )
    })),
    "ContourPlot" if args.len() >= 3 => Some(quiet_plot(|| {
      let domain = crate::functions::fem2d::region_domain_args(args);
      let args = domain.as_deref().unwrap_or(args);
      crate::functions::field_plot::contour_plot_ast(
        &crate::functions::field_plot::with_evaluated_options(args, 3)?,
      )
    })),
    // ContourPlot3D[f == c, {x, x0, x1}, {y, y0, y1}, {z, z0, z1}, opts…] —
    // extracts the level-set surface with marching tetrahedra.
    "ContourPlot3D"
      if args.len() >= 4
        && matches!(
          &args[1],
          Expr::List(items) if items.len() == 3 && matches!(&items[0], Expr::Identifier(_))
        )
        && matches!(
          &args[2],
          Expr::List(items) if items.len() == 3 && matches!(&items[0], Expr::Identifier(_))
        )
        && matches!(
          &args[3],
          Expr::List(items) if items.len() == 3 && matches!(&items[0], Expr::Identifier(_))
        ) =>
    {
      Some(quiet_plot(|| {
        crate::functions::plot3d::contour_plot3d_ast(args)
      }))
    }
    "DensityPlot3D" | "SliceDensityPlot3D" | "SliceContourPlot3D"
      if args.len() >= 4 =>
    {
      use crate::functions::volume_plots as vp;
      Some(quiet_plot(|| {
        let plotted = match name {
          "DensityPlot3D" => vp::density_plot3d_ast(args)?,
          "SliceDensityPlot3D" => vp::slice_density_plot3d_ast(args)?,
          _ => vp::slice_contour_plot3d_ast(args)?,
        };
        Ok(plotted.unwrap_or_else(|| unevaluated(name, args)))
      }))
    }
    "RegionPlot" if args.len() >= 3 => Some(quiet_plot(|| {
      crate::functions::field_plot::region_plot_ast(
        &crate::functions::field_plot::with_evaluated_options(args, 3)?,
      )
    })),
    "VectorPlot" if args.len() >= 3 => Some(quiet_plot(|| {
      crate::functions::field_plot::vector_plot_ast(args)
    })),
    "ListVectorPlot" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::field_plot::list_vector_plot_ast(args)
    })),
    "VectorPlot3D" if args.len() >= 4 => Some(quiet_plot(|| {
      crate::functions::plot3d::vector_plot3d_ast(args)
    })),
    "StreamPlot3D" if args.len() >= 4 => Some(quiet_plot(|| {
      crate::functions::plot3d::stream_plot3d_ast(args)
    })),
    "ListStreamPlot" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::field_plot::list_stream_plot_ast(args)
    })),
    "StreamPlot" if args.len() >= 3 => Some(quiet_plot(|| {
      crate::functions::field_plot::stream_plot_ast(args)
    })),
    "StreamDensityPlot" if args.len() >= 3 => Some(quiet_plot(|| {
      crate::functions::field_plot::stream_density_plot_ast(args)
    })),
    "VectorDensityPlot" if args.len() >= 3 => Some(quiet_plot(|| {
      crate::functions::field_plot::vector_density_plot_ast(args)
    })),
    "ListDensityPlot" if !args.is_empty() => {
      Some(not_an_array(name, args, "").map_or_else(
        || {
          quiet_plot(|| {
            crate::functions::field_plot::list_density_plot_ast(args)
          })
        },
        Ok,
      ))
    }
    "SmoothHistogram" if !args.is_empty() => {
      Some(crate::functions::field_plot::smooth_histogram_ast(args))
    }
    "SmoothDensityHistogram" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::field_plot::smooth_density_histogram_ast(args)
    })),
    "SmoothHistogram3D" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::field_plot::smooth_histogram_3d_ast(args)
    })),
    "ReliefPlot" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::field_plot::relief_plot_ast(args)
    })),
    "ListContourPlot" if !args.is_empty() => {
      Some(not_an_array(name, args, "").map_or_else(
        || {
          quiet_plot(|| {
            crate::functions::field_plot::list_contour_plot_ast(args)
          })
        },
        Ok,
      ))
    }
    "ArrayPlot" | "MatrixPlot" if !args.is_empty() => Some(
      match crate::functions::field_plot::array_plot_argument_error(name, args)
      {
        Some(unevaluated) => Ok(unevaluated),
        None if name == "ArrayPlot" => {
          quiet_plot(|| crate::functions::field_plot::array_plot_ast(args))
        }
        None => {
          quiet_plot(|| crate::functions::field_plot::matrix_plot_ast(args))
        }
      },
    ),
    "BarChart" if !args.is_empty() => {
      Some(crate::functions::chart::bar_chart_ast(args))
    }
    "PairedBarChart" | "PairedHistogram" if args.len() >= 2 => {
      Some(crate::functions::chart::paired_chart_ast(name, args))
    }
    "DensityHistogram" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::density_histogram::density_histogram_ast(args)
    })),
    "PairedSmoothHistogram" if args.len() >= 2 => Some(quiet_plot(|| {
      crate::functions::chart::paired_smooth_histogram_ast(args)
    })),
    // Association, temporal and `Quantity` data become plain list data
    // (keys filling the `Automatic` labels and legends) through the one
    // shared rewrite. BubbleChart reads its (point-level) associations
    // itself.
    "BarChart3D" | "PieChart3D" | "SectorChart" | "SectorChart3D"
    | "RectangleChart" | "RectangleChart3D" | "BubbleChart"
      if !args.is_empty()
        && (name != "BubbleChart"
          || !matches!(&args[0], Expr::Association(_)))
        && let Some(new_args) =
          crate::functions::chart::chart_data_args(args, false) =>
    {
      Some(crate::evaluator::evaluate_function_call_ast(
        name, &new_args,
      ))
    }
    // The 3-D charts render quietly, but a first argument that is no
    // dataset still reports `head::ldata` and stays unevaluated.
    "BarChart3D" | "PieChart3D" | "SectorChart3D"
      if !args.is_empty() && !matches!(&args[0], Expr::List(_)) =>
    {
      Some(Ok(crate::functions::chart::invalid_dataset(name, args)))
    }
    "BarChart3D" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::chart::bar_chart_3d_ast(args)
    })),
    "PieChart" if !args.is_empty() => {
      Some(crate::functions::chart::pie_chart_ast(args))
    }
    "PieChart3D" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::chart::pie_chart_3d_ast(args)
    })),
    "Histogram3D" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::chart::histogram_3d_ast(args)
    })),
    "Histogram" if !args.is_empty() => {
      Some(crate::functions::chart::histogram_ast(args))
    }
    // The legacy `Histograms`` package spelling: `HistogramCategories` gave
    // the bin edges and `HistogramRange` the plotted range, both folded into
    // the modern bin specification.
    "Histograms`Histogram" if !args.is_empty() => {
      let mut new_args = vec![args[0].clone()];
      let mut bins = None;
      for opt in &args[1..] {
        match super::list_operations::as_rule(opt) {
          Some((Expr::Identifier(key), value))
            if key == "Histograms`HistogramCategories" =>
          {
            // A plain list of category boundaries is `{{e1, e2, ...}}`
            // in the modern bin specification.
            bins = Some(match value {
              Expr::List(_) => Expr::List(vec![value.clone()].into()),
              _ => value.clone(),
            });
          }
          Some((Expr::Identifier(key), _))
            if key == "Histograms`HistogramRange" => {}
          _ => new_args.push(opt.clone()),
        }
      }
      if let Some(bins) = bins {
        new_args.insert(1, bins);
      }
      Some(crate::functions::chart::histogram_ast(&new_args))
    }
    // Data that is no list reports `::ldata` (outside the plot's `Quiet`).
    "LineIntegralConvolutionPlot" if args.len() >= 3 => {
      Some(quiet_plot(|| {
        Ok(
          crate::functions::lic_plot::line_integral_convolution_plot_ast(args)?
            .unwrap_or_else(|| unevaluated(name, args)),
        )
      }))
    }
    "BulletGauge" if args.len() >= 3 => Some(
      crate::functions::bullet_gauge::bullet_gauge_ast(args)
        .map(|g| g.unwrap_or_else(|| unevaluated(name, args))),
    ),
    "JuliaSetPlot" if !args.is_empty() => Some(quiet_plot(|| {
      Ok(
        crate::functions::julia_set::julia_set_plot_ast(args)?
          .unwrap_or_else(|| unevaluated(name, args)),
      )
    })),
    "DistributionChart"
      if !args.is_empty() && !matches!(&args[0], Expr::List(_)) =>
    {
      Some(Ok(crate::functions::chart::invalid_dataset(name, args)))
    }
    "DistributionChart" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::distribution_chart::distribution_chart_ast(args)
    })),
    "BoxWhiskerChart" if !args.is_empty() => Some(
      crate::functions::box_whisker_chart::box_whisker_chart_ast(args),
    ),
    "RectangleChart" if !args.is_empty() => Some(
      crate::functions::rectangle_chart::rectangle_chart_ast(args, false),
    ),
    "BarCharts`GeneralizedBarChart" if !args.is_empty() => {
      Some(crate::functions::rectangle_chart::generalized_bar_chart_ast(args))
    }
    "RectangleChart3D" if !args.is_empty() => Some(
      crate::functions::rectangle_chart::rectangle_chart_ast(args, true),
    ),
    "BubbleChart" if !args.is_empty() => {
      Some(crate::functions::bubble_chart::bubble_chart_ast(args))
    }
    "BubbleChart3D" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::chart3d::bubble_chart_3d_ast(args)
    })),
    "BubbleHistogram" if !args.is_empty() => {
      Some(crate::functions::chart::bubble_histogram_ast(args))
    }
    "SectorChart" if !args.is_empty() => {
      Some(crate::functions::chart::sector_chart_ast(args))
    }
    "SectorChart3D" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::chart::sector_chart_3d_ast(args)
    })),
    "DateListPlot" if !args.is_empty() => {
      Some(crate::functions::chart::date_list_plot_ast(args))
    }
    "CandlestickChart" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::chart::candlestick_chart_ast(args)
    })),
    "StackedDateListPlot" if !args.is_empty() => {
      Some(crate::functions::chart::stacked_date_list_plot_ast(args))
    }
    "DateListLogPlot" if !args.is_empty() => {
      Some(crate::functions::chart::date_list_log_plot_ast(args))
    }
    "NumberLinePlot" if !args.is_empty() => Some(
      crate::functions::number_line_plot::number_line_plot_ast(args),
    ),
    "TimelinePlot" if !args.is_empty() => {
      Some(crate::functions::timeline_plot::timeline_plot_ast(args))
    }
    "Dendrogram" if !args.is_empty() => {
      Some(crate::functions::dendrogram::dendrogram_ast(args))
    }
    "TernaryListPlot" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::ternary_list_plot::ternary_list_plot_ast(args)
    })),
    // Not quiet_plot: the elmntav/inpt argument messages must reach the
    // user, exactly as they do in wolframscript.
    "PeriodicTablePlot" => {
      Some(crate::functions::periodic_table_plot::periodic_table_plot_ast(args))
    }
    "WordCloud" if !args.is_empty() => {
      Some(crate::functions::chart::word_cloud_ast(args))
    }
    // PrintTemporary prints like Print in script mode (the notebook
    // "temporary cell" behavior does not apply) and returns Null.
    "Print" | "PrintTemporary" => {
      if args.is_empty() {
        if !crate::is_quiet_print() {
          println!();
        }
        crate::capture_stdout("");
        return Some(Ok(null_expr()));
      }
      // Printing consumes its arguments, so an `Unevaluated[…]` wrapper
      // comes off first: `Print[Unevaluated[Symbol]]` writes `Symbol`.
      let display_str: String = args
        .iter()
        .map(|arg| expr_to_output(&crate::evaluator::strip_unevaluated(arg)))
        .collect::<String>();
      if !crate::is_quiet_print() {
        println!("{display_str}");
      }
      crate::capture_stdout(&display_str);
      Some(Ok(null_expr()))
    }
    _ => None,
  }
}
