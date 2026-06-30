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

pub fn dispatch_plotting(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
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
            "Run: failed to execute command: {}",
            e
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
    // RulePlot[obj] — currently produces a Graphics placeholder so the
    // expected `-Graphics-` output renders, matching wolframscript.
    // Full visual rendering of substitution rules, CellularAutomaton
    // rules, etc. isn't implemented.
    "RulePlot" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::graphics::show_ast(&[Expr::FunctionCall {
        name: "Graphics".to_string(),
        args: vec![Expr::List(vec![].into())].into(),
      }])
    })),
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
          for it in items.iter() {
            out.push(Expr::FunctionCall {
              name: "Re".to_string(),
              args: vec![it.clone()].into(),
            });
            out.push(Expr::FunctionCall {
              name: "Im".to_string(),
              args: vec![it.clone()].into(),
            });
          }
          Expr::List(out.into())
        }
        f => Expr::List(
          vec![
            Expr::FunctionCall {
              name: "Re".to_string(),
              args: vec![f.clone()].into(),
            },
            Expr::FunctionCall {
              name: "Im".to_string(),
              args: vec![f.clone()].into(),
            },
          ]
          .into(),
        ),
      };
      let mut new_args = Vec::with_capacity(args.len());
      new_args.push(reim_list);
      new_args.extend(args[1..].iter().cloned());
      Some(quiet_plot(|| crate::functions::plot::plot_ast(&new_args)))
    }
    // Manipulate holds its arguments (see core_eval.rs) and, in a text
    // front-end, simply echoes itself back with the body and variable
    // specs preserved. A non-list variable spec emits a Manipulate::vsform
    // message but the expression is still returned unchanged.
    "Manipulate" => Some(crate::functions::graphics::manipulate_ast(args)),
    "Plot3D" if args.len() >= 3 => {
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
    "TreeForm" if !args.is_empty() => {
      // TreeForm stays as a wrapper in OutputForm (matching wolframscript)
      Some(Ok(Expr::FunctionCall {
        name: "TreeForm".to_string(),
        args: args.to_vec().into(),
      }))
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
    "LinearGradientFilling" => Some(
      crate::functions::graphics::linear_gradient_filling_ast(args),
    ),
    "ListPlot3D" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::plot3d::list_plot3d_ast(args)
    })),
    "ListPointPlot3D" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::plot3d::list_point_plot3d_ast(args)
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
    "ListPlot" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::list_plot::list_plot_ast(args)
    })),
    "DiscretePlot" if args.len() >= 2 => Some(quiet_plot(|| {
      crate::functions::list_plot::discrete_plot_ast(args)
    })),
    "DiscretePlot3D" if args.len() >= 3 => Some(quiet_plot(|| {
      crate::functions::plot3d::discrete_plot3d_ast(args)
    })),
    "ListLinePlot" if !args.is_empty() => {
      // Match wolframscript: if the first argument doesn't evaluate
      // to a List, emit `ListLinePlot::lpn` (outside Quiet) and
      // return the call unevaluated. wolframscript on
      // `ListLinePlot[list]` prints the message and leaves the
      // expression symbolic.
      match crate::evaluator::evaluate_expr_to_expr(&args[0]) {
        // A TimeSeries / TemporalData is a valid (non-List) data source.
        Ok(evaluated)
          if !matches!(evaluated, Expr::List(_))
            && crate::functions::timeseries_ast::temporal_paths(&evaluated)
              .is_none() =>
        {
          crate::emit_message(&format!(
            "ListLinePlot::lpn: {} is not a list of numbers or pairs of numbers.",
            crate::syntax::expr_to_string(&evaluated)
          ));
          Some(Ok(Expr::FunctionCall {
            name: "ListLinePlot".to_string(),
            args: args.to_vec().into(),
          }))
        }
        _ => Some(quiet_plot(|| {
          crate::functions::list_plot::list_line_plot_ast(args)
        })),
      }
    }
    "ListLogPlot" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::list_plot::list_log_plot_ast(args)
    })),
    "ListLogLogPlot" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::list_plot::list_log_log_plot_ast(args)
    })),
    "ListLogLinearPlot" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::list_plot::list_log_linear_plot_ast(args)
    })),
    "ListPolarPlot" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::list_plot::list_polar_plot_ast(args)
    })),
    "ListStepPlot" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::list_plot::list_step_plot_ast(args)
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
      let xy_expr = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Plus,
        left: Box::new(Expr::Identifier(xv.clone())),
        right: Box::new(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(Expr::Identifier("I".to_string())),
          right: Box::new(Expr::Identifier(yv.clone())),
        }),
      };
      let body = crate::syntax::substitute_variable(&args[0], &zvar, &xy_expr);
      let modulus = Expr::FunctionCall {
        name: "Abs".to_string(),
        args: vec![body].into(),
      };
      // Plot ranges: Re(zmin)..Re(zmax) and Im(zmin)..Im(zmax).
      // Evaluate them to concrete numbers so Plot3D's iterator parser
      // accepts them.
      let eval_part = |e: &Expr, part: &str| -> Expr {
        let wrapped = Expr::FunctionCall {
          name: part.to_string(),
          args: vec![e.clone()].into(),
        };
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
              let neg_r = Expr::FunctionCall {
                name: "Minus".to_string(),
                args: vec![r.clone()].into(),
              };
              let corner_lo = Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Plus,
                left: Box::new(neg_r.clone()),
                right: Box::new(Expr::BinaryOp {
                  op: crate::syntax::BinaryOperator::Times,
                  left: Box::new(neg_r.clone()),
                  right: Box::new(Expr::Identifier("I".to_string())),
                }),
              };
              let corner_hi = Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Plus,
                left: Box::new(r.clone()),
                right: Box::new(Expr::BinaryOp {
                  op: crate::syntax::BinaryOperator::Times,
                  left: Box::new(r),
                  right: Box::new(Expr::Identifier("I".to_string())),
                }),
              };
              (name.clone(), corner_lo, corner_hi)
            }
          }
          _ => unreachable!(),
        },
        _ => unreachable!(),
      };
      let xv = "x".to_string();
      let yv = "y".to_string();
      let xy_expr = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Plus,
        left: Box::new(Expr::Identifier(xv.clone())),
        right: Box::new(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(Expr::Identifier("I".to_string())),
          right: Box::new(Expr::Identifier(yv.clone())),
        }),
      };
      let pred = crate::syntax::substitute_variable(&args[0], &zvar, &xy_expr);
      let eval_part = |e: &Expr, part: &str| -> Expr {
        let wrapped = Expr::FunctionCall {
          name: part.to_string(),
          args: vec![e.clone()].into(),
        };
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
      crate::functions::field_plot::density_plot_ast(args)
    })),
    "ContourPlot" if args.len() >= 3 => Some(quiet_plot(|| {
      crate::functions::field_plot::contour_plot_ast(args)
    })),
    // ContourPlot3D[f, {x, x0, x1}, {y, y0, y1}, {z, z0, z1}, opts…] —
    // returns a placeholder Graphics3D matching wolframscript's
    // `-Graphics3D-`. The full level-set surface is not yet computed.
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
        crate::functions::plot3d::graphics3d_ast(&[Expr::List(vec![].into())])
      }))
    }
    "RegionPlot" if args.len() >= 3 => Some(quiet_plot(|| {
      crate::functions::field_plot::region_plot_ast(args)
    })),
    "VectorPlot" if args.len() >= 3 => Some(quiet_plot(|| {
      crate::functions::field_plot::vector_plot_ast(args)
    })),
    "VectorPlot3D" if args.len() >= 4 => Some(quiet_plot(|| {
      crate::functions::plot3d::vector_plot3d_ast(args)
    })),
    "StreamPlot" if args.len() >= 3 => Some(quiet_plot(|| {
      crate::functions::field_plot::stream_plot_ast(args)
    })),
    "StreamDensityPlot" if args.len() >= 3 => Some(quiet_plot(|| {
      crate::functions::field_plot::stream_density_plot_ast(args)
    })),
    "ListDensityPlot" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::field_plot::list_density_plot_ast(args)
    })),
    "ListContourPlot" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::field_plot::list_contour_plot_ast(args)
    })),
    "ArrayPlot" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::field_plot::array_plot_ast(args)
    })),
    "MatrixPlot" if !args.is_empty() => Some(quiet_plot(|| {
      crate::functions::field_plot::matrix_plot_ast(args)
    })),
    "BarChart" if !args.is_empty() => {
      Some(crate::functions::chart::bar_chart_ast(args))
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
    "Histogram" if !args.is_empty() => {
      Some(crate::functions::chart::histogram_ast(args))
    }
    "BoxWhiskerChart" if !args.is_empty() => {
      Some(crate::functions::chart::box_whisker_chart_ast(args))
    }
    "BubbleChart" if !args.is_empty() => {
      Some(crate::functions::chart::bubble_chart_ast(args))
    }
    "SectorChart" if !args.is_empty() => {
      Some(crate::functions::chart::sector_chart_ast(args))
    }
    "DateListPlot" if !args.is_empty() => {
      Some(crate::functions::chart::date_list_plot_ast(args))
    }
    "NumberLinePlot" if !args.is_empty() => Some(
      crate::functions::number_line_plot::number_line_plot_ast(args),
    ),
    "WordCloud" if !args.is_empty() => {
      Some(crate::functions::chart::word_cloud_ast(args))
    }
    "Print" => {
      if args.is_empty() {
        if !crate::is_quiet_print() {
          println!();
        }
        crate::capture_stdout("");
        return Some(Ok(Expr::Identifier("Null".to_string())));
      }
      let display_str: String = args
        .iter()
        .map(crate::syntax::expr_to_output)
        .collect::<Vec<_>>()
        .join("");
      if !crate::is_quiet_print() {
        println!("{}", display_str);
      }
      crate::capture_stdout(&display_str);
      Some(Ok(Expr::Identifier("Null".to_string())))
    }
    _ => None,
  }
}
