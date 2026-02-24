#[allow(unused_imports)]
use super::*;

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
    "Plot" if args.len() >= 2 => Some(crate::functions::plot::plot_ast(args)),
    "Plot3D" if args.len() >= 3 => {
      Some(crate::functions::plot3d::plot3d_ast(args))
    }
    "Graphics" if !args.is_empty() => {
      Some(crate::functions::graphics::graphics_ast(args))
    }
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
        args: args.to_vec(),
      }))
    }
    "Graphics3D" if !args.is_empty() => {
      Some(crate::functions::plot3d::graphics3d_ast(args))
    }
    "ListPlot3D" if !args.is_empty() => {
      Some(crate::functions::plot3d::list_plot3d_ast(args))
    }
    "ListPlot" if !args.is_empty() => {
      Some(crate::functions::list_plot::list_plot_ast(args))
    }
    "ListLinePlot" if !args.is_empty() => {
      Some(crate::functions::list_plot::list_line_plot_ast(args))
    }
    "ListLogPlot" if !args.is_empty() => {
      Some(crate::functions::list_plot::list_log_plot_ast(args))
    }
    "ListLogLogPlot" if !args.is_empty() => {
      Some(crate::functions::list_plot::list_log_log_plot_ast(args))
    }
    "ListLogLinearPlot" if !args.is_empty() => {
      Some(crate::functions::list_plot::list_log_linear_plot_ast(args))
    }
    "ListPolarPlot" if !args.is_empty() => {
      Some(crate::functions::list_plot::list_polar_plot_ast(args))
    }
    "ListStepPlot" if !args.is_empty() => {
      Some(crate::functions::list_plot::list_step_plot_ast(args))
    }
    "ParametricPlot" if args.len() >= 2 => {
      Some(crate::functions::parametric_plot::parametric_plot_ast(args))
    }
    "PolarPlot" if args.len() >= 2 => {
      Some(crate::functions::parametric_plot::polar_plot_ast(args))
    }
    "DensityPlot" if args.len() >= 3 => {
      Some(crate::functions::field_plot::density_plot_ast(args))
    }
    "ContourPlot" if args.len() >= 3 => {
      Some(crate::functions::field_plot::contour_plot_ast(args))
    }
    "RegionPlot" if args.len() >= 3 => {
      Some(crate::functions::field_plot::region_plot_ast(args))
    }
    "VectorPlot" if args.len() >= 3 => {
      Some(crate::functions::field_plot::vector_plot_ast(args))
    }
    "StreamPlot" if args.len() >= 3 => {
      Some(crate::functions::field_plot::stream_plot_ast(args))
    }
    "StreamDensityPlot" if args.len() >= 3 => {
      Some(crate::functions::field_plot::stream_density_plot_ast(args))
    }
    "ListDensityPlot" if !args.is_empty() => {
      Some(crate::functions::field_plot::list_density_plot_ast(args))
    }
    "ListContourPlot" if !args.is_empty() => {
      Some(crate::functions::field_plot::list_contour_plot_ast(args))
    }
    "ArrayPlot" if !args.is_empty() => {
      Some(crate::functions::field_plot::array_plot_ast(args))
    }
    "MatrixPlot" if !args.is_empty() => {
      Some(crate::functions::field_plot::matrix_plot_ast(args))
    }
    "BarChart" if !args.is_empty() => {
      Some(crate::functions::chart::bar_chart_ast(args))
    }
    "PieChart" if !args.is_empty() => {
      Some(crate::functions::chart::pie_chart_ast(args))
    }
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
    "WordCloud" if !args.is_empty() => {
      Some(crate::functions::chart::word_cloud_ast(args))
    }
    "Print" => {
      if args.is_empty() {
        println!();
        crate::capture_stdout("");
        return Some(Ok(Expr::Identifier("Null".to_string())));
      }
      let display_str: String = args
        .iter()
        .map(crate::syntax::expr_to_output)
        .collect::<Vec<_>>()
        .join("");
      println!("{}", display_str);
      crate::capture_stdout(&display_str);
      Some(Ok(Expr::Identifier("Null".to_string())))
    }
    _ => None,
  }
}
