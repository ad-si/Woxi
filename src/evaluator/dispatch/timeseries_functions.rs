//! Dispatch for temporal-data heads (`TemporalData`, `TimeSeries`,
//! `TimeSeriesResample`) plus interception of descriptive statistics applied
//! directly to a `TimeSeries` (e.g. `Mean[ts]`, `Total[ts]`). This dispatcher
//! runs before `list_operations` so the statistics heads can pull the value
//! path out of a `TimeSeries` before the generic list handlers see it.

use crate::InterpreterError;
use crate::functions::timeseries_ast;
use crate::syntax::Expr;

pub fn dispatch_timeseries_functions(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "TemporalData" => Some(timeseries_ast::temporal_data_ast(args)),
    "TimeSeries" => Some(timeseries_ast::time_series_ast(args)),
    "TimeSeriesResample" => {
      Some(timeseries_ast::time_series_resample_ast(args))
    }
    // Descriptive statistics over a TimeSeries operate on its value path.
    "Mean" | "Total" | "Min" | "Max" | "Median" | "Variance"
    | "StandardDeviation" | "Commonest" | "Length"
      if args.len() == 1 =>
    {
      let values = timeseries_ast::time_series_values(&args[0])?;
      Some(crate::evaluator::evaluate_expr_to_expr(
        &Expr::FunctionCall {
          name: name.to_string(),
          args: vec![values].into(),
        },
      ))
    }
    _ => None,
  }
}
