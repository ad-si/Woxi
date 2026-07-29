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
    "TimeSeriesResample" if args.len() == 1 => {
      Some(timeseries_ast::time_series_resample_default(args))
    }
    "TimeSeriesResample" => {
      Some(timeseries_ast::time_series_resample_ast(args))
    }
    "TimeSeriesWindow" => Some(timeseries_ast::time_series_window_ast(args)),
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
    // `Normal` unwraps an EventSeries as well as a TimeSeries.
    "Normal"
      if args.len() == 1
        && timeseries_ast::series_pairs_of(&args[0]).is_some() =>
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
    // EventSeries[{{t, v}, …}] stays inert like TimeSeries; its property
    // queries are answered by the shared path handler.
    "EventSeries"
      if args.len() == 1
        && timeseries_ast::series_pairs_of(&Expr::FunctionCall {
          name: "EventSeries".to_string(),
          args: vec![args[0].clone()].into(),
        })
        .is_some() =>
    {
      Some(Ok(crate::syntax::unevaluated("EventSeries", args)))
    }
    "TimeSeriesShift" => Some(timeseries_ast::time_series_shift_ast(args)),
    "TimeSeriesRescale" => Some(timeseries_ast::time_series_rescale_ast(args)),
    "TimeSeriesMap" => Some(timeseries_ast::time_series_map_ast(args)),
    "TimeSeriesThread" => Some(timeseries_ast::time_series_thread_ast(args)),
    "TimeSeriesInsert" => Some(timeseries_ast::time_series_insert_ast(args)),
    "RegularlySampledQ" => Some(timeseries_ast::regularly_sampled_q_ast(args)),
    "EventSeriesQ" if args.len() == 1 => {
      Some(timeseries_ast::event_series_q_ast(args))
    }
    "EventSeriesLookup" if args.len() == 2 => {
      Some(timeseries_ast::event_series_lookup_ast(args))
    }
    "EventSeriesAccumulate" if args.len() == 1 => {
      Some(timeseries_ast::event_series_accumulate_ast(args))
    }
    // MovingAverage over a series averages the values and keeps the later
    // time stamp of each window, the way wolframscript reports it.
    "MovingAverage"
      if args.len() == 2
        && timeseries_ast::series_pairs_of(&args[0]).is_some() =>
    {
      Some(timeseries_ast::time_series_moving_average_ast(args))
    }
    _ => None,
  }
}
