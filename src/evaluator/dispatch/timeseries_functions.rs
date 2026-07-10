//! Dispatch for temporal-data heads (`TemporalData`, `TimeSeries`,
//! `TimeSeriesResample`) plus interception of descriptive statistics applied
//! directly to a `TimeSeries` (e.g. `Mean[ts]`, `Total[ts]`). This dispatcher
//! runs before `list_operations` so the statistics heads can pull the value
//! path out of a `TimeSeries` before the generic list handlers see it.

use crate::InterpreterError;
use crate::functions::timeseries_ast;
use crate::syntax::Expr;

pub(super) fn dispatch_timeseries_functions(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "TemporalData" => Some(timeseries_ast::temporal_data_ast(args)),
    "TimeSeries" => Some(timeseries_ast::time_series_ast(args)),
    "TimeSeriesResample" => {
      Some(timeseries_ast::time_series_resample_ast(args))
    }
    // `Length` of a TimeSeries reports the arity of the underlying
    // `TemporalData` object, which is always 4 (`TemporalData[tag, dataspec,
    // bool, version]`) — independent of the number of data points.
    "Length"
      if args.len() == 1
        && timeseries_ast::time_series_pairs(&args[0]).is_some() =>
    {
      Some(Ok(Expr::Integer(4)))
    }
    // `Values[ts]` is the value path; `Normal[ts]` is the explicit
    // `{{date, value}, …}` list with each stamp as an `Instant` DateObject.
    "Values"
      if args.len() == 1
        && timeseries_ast::time_series_pairs(&args[0]).is_some() =>
    {
      Some(Ok(timeseries_ast::time_series_values_output(&args[0])?))
    }
    "Normal"
      if args.len() == 1
        && timeseries_ast::time_series_pairs(&args[0]).is_some() =>
    {
      Some(Ok(timeseries_ast::time_series_normal(&args[0])?))
    }
    // Descriptive statistics over a TimeSeries operate on its value path.
    "Mean" | "Total" | "Min" | "Max" | "Median" | "Variance"
    | "StandardDeviation" | "Commonest"
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
