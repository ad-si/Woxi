//! TemporalData / TimeSeries / TimeSeriesResample and the `CompressedData`
//! binary-serialization reader that backs them. The reference values were
//! confirmed against wolframscript.

use super::*;

/// The wolframscript step-count time series used throughout these tests:
/// 154 daily values from 2013-04-01, stored as a `CompressedData` packed array.
const STEP_DATA: &str = r#"stepdata = TemporalData[TimeSeries, {CompressedData["
1:eJwdkk1IVGEUhh+FjCRTzJrMn3Eya/Ina3TKMQdEVEIMyhYKgoIL0RIjWhYF
IVTQxhKDBEUUgqlFhKJgtptFuBCUIoIWoRGBtXAjKIJPXjic+93znvd9z/lu
qOdO22AqkGK8NkKl8KcCZvIgkQPjR+FIAbRUw0IZrF6AZWP9PMSMlCz4dBnC
YXjreewiZMeh/D+uCK5mQ+QsfCmGRfmul8CvY/CvCt7Le/cQ+0/xAbglpk/t
bnsy1Hopx7a5SM7UCHTWwdBxseXQXgn3zFP6WtTrjUJ4ZB60d1vur9Zbrd2U
I67e7yg81U9pEL7pM2FMyjst7tVJeKDuG9/r7Ilece4a6LoEyRhsyRl0Lx8z
YaAW1uRZt6e60f5TMOK8NX5fUueJ+5o1T2dAvZgW9xPTe5de0+X5oLc0uWud
e+Ic3D4IjblwX3/P9RM5487k28h3VvU7miFX/LD+KuQd0UdQj6t6Tao54z6y
7H2nt2F1ctRJsy8sLi6mROyctXm5v3v+bL3X/EzsuDmq1qg7eew+rnn+q86K
uVncgLwJffQcdjbvLuTdJb2PhwEION9P/Td49z/ccZ/4DTn6ne+FuJ02/412
2JU/0ASVp+Xy25D73DwBez7dZgU=
"], {
TemporalData`DateSpecification[{2013, 4, 1, 0, 0, 0.}, {2013, 9, 1, 0, 0, 0.}, {1, "Day"}]}, 1, {"Discrete", 1}, {"Discrete", 1}, 1, {ValueDimensions -> 1}}, True, 10.];"#;

mod compressed_data {
  use super::*;

  #[test]
  fn decodes_wl_binary_packed_array() {
    // First five daily step counts of the embedded packed Integer32 array.
    let code = format!(
      "Take[CompressedData[\"\n1:eJwdkk1IVGEUhh+FjCRTzJrMn3Eya/Ina3TKMQdEVEIMyhYKgoIL0RIjWhYF\nIVTQxhKDBEUUgqlFhKJgtptFuBCUIoIWoRGBtXAjKIJPXjic+93znvd9z/lu\nqOdO22AqkGK8NkKl8KcCZvIgkQPjR+FIAbRUw0IZrF6AZWP9PMSMlCz4dBnC\nYXjreewiZMeh/D+uCK5mQ+QsfCmGRfmul8CvY/CvCt7Le/cQ+0/xAbglpk/t\nbnsy1Hopx7a5SM7UCHTWwdBxseXQXgn3zFP6WtTrjUJ4ZB60d1vur9Zbrd2U\nI67e7yg81U9pEL7pM2FMyjst7tVJeKDuG9/r7Ilece4a6LoEyRhsyRl0Lx8z\nYaAW1uRZt6e60f5TMOK8NX5fUueJ+5o1T2dAvZgW9xPTe5de0+X5oLc0uWud\ne+Ic3D4IjblwX3/P9RM5487k28h3VvU7miFX/LD+KuQd0UdQj6t6Tao54z6y\n7H2nt2F1ctRJsy8sLi6mROyctXm5v3v+bL3X/EzsuDmq1qg7eew+rnn+q86K\nuVncgLwJffQcdjbvLuTdJb2PhwEION9P/Td49z/ccZ/4DTn6ne+FuJ02/412\n2JU/0ASVp+Xy25D73DwBez7dZgU=\n\"], 5]"
    );
    assert_eq!(
      interpret(&code).unwrap(),
      "{10785, 11753, 7092, 5290, 5022}"
    );
  }

  #[test]
  fn woxi_compress_roundtrips() {
    assert_eq!(
      interpret(r#"Uncompress[Compress[{1, 2, 3, "hi"}]]"#).unwrap(),
      "{1, 2, 3, hi}"
    );
  }
}

mod temporal_data {
  use super::*;

  #[test]
  fn weekday_means_match_wolframscript() {
    let code = format!(
      "{STEP_DATA}\n\
       days = {{Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday}};\n\
       wk = Map[TimeSeriesResample[stepdata, #] &, days];\n\
       Map[Floor[Mean[#]] &, wk]"
    );
    assert_eq!(
      interpret(&code).unwrap(),
      "{10904, 10755, 11368, 10575, 10999, 9167, 9808}"
    );
  }

  #[test]
  fn renders_bar_chart() {
    let code = format!(
      "{STEP_DATA}\n\
       days = {{Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday}};\n\
       avg = Map[Floor[Mean[TimeSeriesResample[stepdata, #]]] &, days];\n\
       BarChart[avg, ChartStyle -> \"DarkRainbow\", \
         LabelingFunction -> (Placed[#, Below] &), \
         ChartLabels -> {{avg, Placed[days, Center]}}]"
    );
    assert_eq!(interpret(&code).unwrap(), "-Graphics-");
  }

  // The category labels live inside a nested `Placed[days, Center, styleFn]`
  // whose styling function rotates them vertically; they must all render.
  #[test]
  fn nested_placed_renders_rotated_day_labels() {
    let days = [
      "Monday",
      "Tuesday",
      "Wednesday",
      "Thursday",
      "Friday",
      "Saturday",
      "Sunday",
    ];
    let code = format!(
      "{STEP_DATA}\n\
       days = {{Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday}};\n\
       avg = Map[Floor[Mean[TimeSeriesResample[stepdata, #]]] &, days];\n\
       ExportString[BarChart[avg, ChartStyle -> \"DarkRainbow\", \
         LabelingFunction -> (Placed[#, Below] &), \
         ChartLabels -> {{avg, Placed[days, Center, \
           Style[Rotate[#, Pi/2], 16, Bold, Opacity[1]] &]}}], \"SVG\"]"
    );
    let svg = interpret(&code).unwrap();
    for day in days {
      let line = svg
        .lines()
        .find(|l| l.contains(&format!(">{day}</text>")))
        .unwrap_or_else(|| panic!("no label for {day}"));
      assert!(
        line.contains("rotate(-90"),
        "expected {day} label to be rotated vertical"
      );
      // Centered on the bar in both axes (so the rotated glyphs don't hang to
      // one side of the bar's center).
      assert!(
        line.contains("text-anchor=\"middle\"")
          && line.contains("dominant-baseline=\"central\""),
        "expected {day} label to be centered on its bar"
      );
      // wolframscript draws centered bar labels in the default dark color,
      // not white.
      assert!(
        line.contains("fill=\"#333\"") && !line.contains("fill=\"white\""),
        "expected dark fill for {day} label, got: {line}"
      );
    }
    // The LabelingFunction (`Placed[#, Below] &`) draws the integer value
    // below each bar — no trailing dot, matching wolframscript.
    for value in ["10904", "10755", "11368", "10575", "10999", "9167", "9808"] {
      assert!(
        svg.contains(&format!(">{value}</text>")),
        "expected value label {value} below its bar in chart SVG"
      );
    }
  }
}

mod labeling_function {
  use super::*;

  // A vertical BarChart must draw the LabelingFunction value below each bar;
  // integer-valued bars render without a trailing dot.
  #[test]
  fn vertical_bar_renders_integer_value_labels() {
    let svg = interpret(
      "ExportString[BarChart[{3, 5, 2}, \
       LabelingFunction -> (Placed[#, Below] &)], \"SVG\"]",
    )
    .unwrap();
    for value in ["3", "5", "2"] {
      assert!(
        svg.contains(&format!(">{value}</text>")),
        "expected value label {value} in vertical BarChart SVG"
      );
    }
    assert!(
      !svg.contains(">3.</text>"),
      "integer value label should not have a trailing dot"
    );
  }

  // Wide multi-digit value labels on many bars must shrink to fit so adjacent
  // labels don't overlap (regression for the crowded 7-bar weekday chart).
  #[test]
  fn wide_value_labels_shrink_to_avoid_overlap() {
    let values = ["10904", "10755", "11368", "10575", "10999", "9167", "9808"];
    let svg = interpret(&format!(
      "ExportString[BarChart[{{{}}}, \
       LabelingFunction -> (Placed[#, Below] &)], \"SVG\"]",
      values.join(", ")
    ))
    .unwrap();

    // Locate each value label's <text> line and read its x and font-size.
    let mut labels: Vec<(f64, f64, usize)> = Vec::new();
    for value in values {
      let needle = format!(">{value}</text>");
      let line = svg
        .lines()
        .find(|l| l.contains(&needle) && l.contains("fill=\"#666\""))
        .unwrap_or_else(|| panic!("no value label for {value}"));
      let x = attr(line, "x=\"").expect("x");
      let fs = attr(line, "font-size=\"").expect("font-size");
      labels.push((x, fs, value.len()));
    }
    labels.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Adjacent labels must not overlap: estimated text width < bar spacing.
    for pair in labels.windows(2) {
      let (x0, fs, len) = pair[0];
      let spacing = pair[1].0 - x0;
      let est_text_width = len as f64 * fs * 0.6;
      assert!(
        est_text_width < spacing,
        "value labels overlap: width ~{est_text_width:.0} >= spacing {spacing:.0}"
      );
    }
  }

  fn attr(line: &str, key: &str) -> Option<f64> {
    let start = line.find(key)? + key.len();
    let rest = &line[start..];
    let end = rest.find('"')?;
    rest[..end].parse().ok()
  }
}

// The user-facing constructor `TemporalData[values, {times}]`: a flat scalar
// path normalizes to a TimeSeries, while a list of paths stays a multi-path
// TemporalData. Both feed ListLinePlot / ListPlot, one line per path.
mod temporal_data_constructor {
  use super::*;

  const S: &str = "s = {2, 1, 6, 5, 7, 4}; t = {1, 2, 5, 10, 12, 15};";
  const MULTI: &str = "s1 = {2, 1, 6, 5, 7, 4}; s2 = {4, 7, 5, 6, 1, 2}; \
    t = {1, 2, 5, 10, 12, 15};";

  #[test]
  fn single_path_normalizes_to_time_series() {
    assert_eq!(
      interpret(&format!("{S} TemporalData[s, {{t}}]")).unwrap(),
      "TimeSeries[{{1, 2}, {2, 1}, {5, 6}, {10, 5}, {12, 7}, {15, 4}}]"
    );
  }

  #[test]
  fn single_path_stats_use_value_path() {
    assert_eq!(
      interpret(&format!("{S} Mean[TemporalData[s, {{t}}]]")).unwrap(),
      "25/6"
    );
    assert_eq!(
      interpret(&format!("{S} TemporalData[s, {{t}}][\"Values\"]")).unwrap(),
      "{2, 1, 6, 5, 7, 4}"
    );
  }

  #[test]
  fn single_path_list_line_plot_renders_graphics() {
    assert_eq!(
      interpret(&format!("{S} ListLinePlot[TemporalData[s, {{t}}]]")).unwrap(),
      "-Graphics-"
    );
    assert_eq!(
      interpret(&format!("{S} ListPlot[TemporalData[s, {{t}}]]")).unwrap(),
      "-Graphics-"
    );
  }

  #[test]
  fn multi_path_stays_temporal_data() {
    assert_eq!(
      interpret(&format!("{MULTI} TemporalData[{{s1, s2}}, {{t}}]")).unwrap(),
      "TemporalData[{{2, 1, 6, 5, 7, 4}, {4, 7, 5, 6, 1, 2}}, \
       {{1, 2, 5, 10, 12, 15}}]"
    );
  }

  #[test]
  fn multi_path_list_line_plot_renders_graphics() {
    assert_eq!(
      interpret(&format!(
        "{MULTI} ListLinePlot[TemporalData[{{s1, s2}}, {{t}}]]"
      ))
      .unwrap(),
      "-Graphics-"
    );
  }

  // Each path is drawn against the shared time axis (its own line in a distinct
  // ColorData[97] palette color), not the 1,2,3,… index axis.
  #[test]
  fn multi_path_plot_draws_one_line_per_path() {
    let svg = interpret(&format!(
      "{MULTI} ExportString[ListLinePlot[TemporalData[{{s1, s2}}, {{t}}]], \"SVG\"]"
    ))
    .unwrap();
    assert!(svg.contains("#5E81B5"), "first path uses the default blue");
    assert!(
      svg.contains("#E0932C"),
      "second path uses the default orange"
    );
  }
}

mod time_series {
  use super::*;

  #[test]
  fn mean_of_value_path() {
    assert_eq!(
      interpret("Mean[TimeSeries[{{1, 10}, {2, 20}, {3, 30}}]]").unwrap(),
      "20"
    );
  }

  #[test]
  fn total_of_value_path() {
    assert_eq!(
      interpret("Total[TimeSeries[{{1, 10}, {2, 20}, {3, 30}}]]").unwrap(),
      "60"
    );
  }

  #[test]
  fn length_is_temporal_data_arity() {
    // A TimeSeries materializes as a 4-argument TemporalData object in WL, so
    // its Length is always 4 — not the number of data points.
    assert_eq!(
      interpret("Length[TimeSeries[{{1, 10}, {2, 20}}]]").unwrap(),
      "4"
    );
    assert_eq!(
      interpret("Length[TimeSeries[{{1, 10}, {2, 20}, {3, 30}}]]").unwrap(),
      "4"
    );
  }

  #[test]
  fn bare_value_path_assigns_integer_times() {
    assert_eq!(
      interpret("Mean[TimeSeries[{10, 20, 30, 40}]]").unwrap(),
      "25"
    );
  }

  // A single numeric start advances the time stamps by 1: 5, 6, 7, 8.
  #[test]
  fn numeric_start_advances_by_one() {
    assert_eq!(
      interpret("TimeSeries[{10, 20, 30, 40}, {5}][\"Path\"]").unwrap(),
      "{{5, 10}, {6, 20}, {7, 30}, {8, 40}}"
    );
  }
}

// TimeSeriesResample[ts, rspec] — sample the piecewise-linear path at evenly
// spaced (or explicitly listed) stamps. All values verified against
// wolframscript.
mod time_series_resample_numeric {
  use super::*;

  const TS: &str = "ts = TimeSeries[{{1, 10}, {3, 30}, {4, 40}, {7, 70}}]; ";

  #[test]
  fn a_bare_step_spans_the_whole_series() {
    assert_eq!(
      interpret(&format!("{TS}TimeSeriesResample[ts, 1][\"Path\"]")).unwrap(),
      "{{1, 10}, {2, 20}, {3, 30}, {4, 40}, {5, 50}, {6, 60}, {7, 70}}"
    );
    assert_eq!(
      interpret(&format!("{TS}TimeSeriesResample[ts, 2][\"Path\"]")).unwrap(),
      "{{1, 10}, {3, 30}, {5, 50}, {7, 70}}"
    );
    assert_eq!(
      interpret(&format!("{TS}Head[TimeSeriesResample[ts, 1]]")).unwrap(),
      "TimeSeries"
    );
  }

  #[test]
  fn no_spec_uses_the_minimum_time_increment() {
    // The gaps are 2, 1 and 3, so the default step is 1.
    assert_eq!(
      interpret(&format!("{TS}TimeSeriesResample[ts][\"Path\"]")).unwrap(),
      "{{1, 10}, {2, 20}, {3, 30}, {4, 40}, {5, 50}, {6, 60}, {7, 70}}"
    );
  }

  #[test]
  fn range_specs() {
    assert_eq!(
      interpret(&format!(
        "{TS}TimeSeriesResample[ts, {{1, 7, 2}}][\"Path\"]"
      ))
      .unwrap(),
      "{{1, 10}, {3, 30}, {5, 50}, {7, 70}}"
    );
    // Without a step the minimum increment is used.
    assert_eq!(
      interpret(&format!("{TS}TimeSeriesResample[ts, {{2, 6}}][\"Path\"]"))
        .unwrap(),
      "{{2, 20}, {3, 30}, {4, 40}, {5, 50}, {6, 60}}"
    );
    // A doubly-nested list is an explicit set of stamps.
    assert_eq!(
      interpret(&format!(
        "{TS}TimeSeriesResample[ts, {{{{1, 4, 7}}}}][\"Path\"]"
      ))
      .unwrap(),
      "{{1, 10}, {4, 40}, {7, 70}}"
    );
  }

  // Outside the sampled span the path is held flat, not extrapolated.
  #[test]
  fn stamps_outside_the_span_clamp() {
    assert_eq!(
      interpret(&format!(
        "{TS}TimeSeriesResample[ts, {{0, 9, 3}}][\"Path\"]"
      ))
      .unwrap(),
      "{{0, 10}, {3, 30}, {6, 60}, {9, 70}}"
    );
    assert_eq!(
      interpret(&format!("{TS}{{ts[0], ts[9]}}")).unwrap(),
      "{10, 70}"
    );
  }

  // Interpolation is exact: integer data at integer stamps stays integral,
  // and an uneven split gives a Rational rather than a float.
  #[test]
  fn interpolation_stays_exact() {
    assert_eq!(interpret(&format!("{TS}ts[2]")).unwrap(), "20");
    assert_eq!(
      interpret("TimeSeriesResample[TimeSeries[{{1, 10}, {2, 15}, {4, 20}}], 1][\"Path\"]")
        .unwrap(),
      "{{1, 10}, {2, 15}, {3, 35/2}, {4, 20}}"
    );
    assert_eq!(
      interpret(
        "TimeSeriesResample[TimeSeries[{{0, 0}, {2, 1}}], 1][\"Path\"]"
      )
      .unwrap(),
      "{{0, 0}, {1, 1/2}, {2, 1}}"
    );
  }

  // A Real step carries a Real time axis, and the values follow it.
  #[test]
  fn an_inexact_step_numericizes_the_values() {
    assert_eq!(
      interpret(&format!("{TS}TimeSeriesResample[ts, 0.5][\"Path\"]")).unwrap(),
      "{{1., 10.}, {1.5, 15.}, {2., 20.}, {2.5, 25.}, {3., 30.}, {3.5, 35.}, \
       {4., 40.}, {4.5, 45.}, {5., 50.}, {5.5, 55.}, {6., 60.}, {6.5, 65.}, \
       {7., 70.}}"
    );
    assert_eq!(
      interpret(&format!(
        "{TS}TimeSeriesResample[ts, {{{{2.5}}}}][\"Path\"]"
      ))
      .unwrap(),
      "{{2.5, 25.}}"
    );
  }

  // The weekday form over date stamps is unaffected.
  #[test]
  fn weekday_form_still_selects_by_day_name() {
    assert_eq!(
      interpret(
        "TimeSeriesResample[TimeSeries[{{{2026, 7, 20}, 1}, \
         {{2026, 7, 21}, 2}, {{2026, 7, 27}, 3}}], Monday][\"PathLength\"]"
      )
      .unwrap(),
      "2"
    );
  }
}

// TimeSeriesWindow[ts, {tmin, tmax}] — the points whose stamps fall in the
// window, both ends included. All values verified against wolframscript.
mod time_series_window {
  use super::*;

  const TS: &str =
    "ts = TimeSeries[{{1, 10}, {2, 20}, {3, 30}, {4, 40}, {5, 50}}]; ";

  #[test]
  fn window_is_inclusive_at_both_ends() {
    assert_eq!(
      interpret(&format!("{TS}TimeSeriesWindow[ts, {{2, 4}}][\"Path\"]"))
        .unwrap(),
      "{{2, 20}, {3, 30}, {4, 40}}"
    );
    assert_eq!(
      interpret(&format!("{TS}TimeSeriesWindow[ts, {{2, 4}}][\"Times\"]"))
        .unwrap(),
      "{2, 3, 4}"
    );
    assert_eq!(
      interpret(&format!(
        "{TS}TimeSeriesWindow[ts, {{2, 4}}][\"PathLength\"]"
      ))
      .unwrap(),
      "3"
    );
    // A degenerate window still catches the point sitting on it.
    assert_eq!(
      interpret(&format!("{TS}TimeSeriesWindow[ts, {{3, 3}}][\"Path\"]"))
        .unwrap(),
      "{{3, 30}}"
    );
  }

  #[test]
  fn result_is_a_time_series() {
    assert_eq!(
      interpret(&format!("{TS}Head[TimeSeriesWindow[ts, {{2, 4}}]]")).unwrap(),
      "TimeSeries"
    );
  }

  #[test]
  fn bounds_need_not_coincide_with_sample_times() {
    assert_eq!(
      interpret(&format!("{TS}TimeSeriesWindow[ts, {{0, 10}}][\"Path\"]"))
        .unwrap(),
      "{{1, 10}, {2, 20}, {3, 30}, {4, 40}, {5, 50}}"
    );
    assert_eq!(
      interpret(&format!("{TS}TimeSeriesWindow[ts, {{2.5, 4.5}}][\"Path\"]"))
        .unwrap(),
      "{{3, 30}, {4, 40}}"
    );
  }

  #[test]
  fn open_ends_and_reversed_bounds() {
    assert_eq!(
      interpret(&format!(
        "{TS}TimeSeriesWindow[ts, {{-Infinity, 3}}][\"Path\"]"
      ))
      .unwrap(),
      "{{1, 10}, {2, 20}, {3, 30}}"
    );
    assert_eq!(
      interpret(&format!(
        "{TS}TimeSeriesWindow[ts, {{3, Infinity}}][\"Path\"]"
      ))
      .unwrap(),
      "{{3, 30}, {4, 40}, {5, 50}}"
    );
    // The bounds may be given the other way round.
    assert_eq!(
      interpret(&format!("{TS}TimeSeriesWindow[ts, {{4, 2}}][\"Path\"]"))
        .unwrap(),
      "{{2, 20}, {3, 30}, {4, 40}}"
    );
  }

  #[test]
  fn date_stamps_are_compared_as_dates() {
    assert_eq!(
      interpret(
        "d = TimeSeries[{{{2020, 1, 1}, 1}, {{2020, 1, 5}, 5}, \
         {{2020, 1, 9}, 9}}]; \
         TimeSeriesWindow[d, {{2020, 1, 4}, {2020, 1, 10}}][\"PathLength\"]"
      )
      .unwrap(),
      "2"
    );
  }

  // An empty window yields an empty series and reports tswndt.
  #[test]
  fn empty_window_reports_tswndt() {
    assert_eq!(
      interpret(&format!("{TS}TimeSeriesWindow[ts, {{10, 20}}][\"Path\"]"))
        .unwrap(),
      "{}"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.starts_with(
        "TimeSeriesWindow::tswndt: The window {10, 20} contains no values on the path(s)"
      )),
      "expected tswndt message, got {msgs:?}"
    );
    assert_eq!(
      interpret(&format!(
        "{TS}TimeSeriesWindow[ts, {{10, 20}}][\"PathLength\"]"
      ))
      .unwrap(),
      "0"
    );
  }

  // Properties stay meaningful on the empty series a missed window produces.
  #[test]
  fn empty_time_series_still_answers_properties() {
    assert_eq!(interpret("TimeSeries[{}][\"Path\"]").unwrap(), "{}");
    assert_eq!(interpret("TimeSeries[{}][\"Times\"]").unwrap(), "{}");
    assert_eq!(interpret("TimeSeries[{}][\"PathLength\"]").unwrap(), "0");
  }
}

// `TimeSeries[values, {DateObject[…]}]` — a single start date spaces the values
// one day apart, and the series can then be sampled, queried, and plotted.
mod date_start {
  use super::*;

  const TS: &str =
    "ts = TimeSeries[{23.1, 24.4, 21.8, 25.5}, {DateObject[{2025, 9, 1}]}];";

  #[test]
  fn assigns_daily_dates_to_all_values() {
    // All four values are retained (the start date is not consumed as data).
    assert_eq!(interpret(&format!("{TS} Length[ts]")).unwrap(), "4");
    assert_eq!(
      interpret(&format!("{TS} ts[\"Values\"]")).unwrap(),
      "{23.1, 24.4, 21.8, 25.5}"
    );
    assert_eq!(interpret(&format!("{TS} Mean[ts]")).unwrap(), "23.7");
  }

  #[test]
  fn lookup_at_data_point_is_exact() {
    // 2025-09-03 is the third daily stamp → 21.8 exactly.
    assert_eq!(
      interpret(&format!("{TS} ts[DateObject[{{2025, 9, 3}}]]")).unwrap(),
      "21.8"
    );
  }

  #[test]
  fn lookup_is_clamped_past_the_last_point() {
    // Outside the sampled span the path is held flat at its end value rather
    // than extrapolating the final segment's slope:
    //   wolframscript -code 'ts = TimeSeries[{23.1, 24.4, 21.8, 25.5},
    //     {DateObject[{2025, 9, 1}]}]; ts[DateObject[{2025, 9, 10}]]'
    //   25.5
    assert_eq!(
      interpret(&format!("{TS} ts[DateObject[{{2025, 9, 10}}]]")).unwrap(),
      "25.5"
    );
    // Same before the first point.
    assert_eq!(
      interpret(&format!("{TS} ts[DateObject[{{2025, 8, 20}}]]")).unwrap(),
      "23.1"
    );
  }

  #[test]
  fn times_are_absolute_seconds() {
    assert_eq!(
      interpret(&format!("{TS} ts[\"Times\"]")).unwrap(),
      "{3.9656736*^9, 3.96576*^9, 3.9658464*^9, 3.9659328*^9}"
    );
  }

  #[test]
  fn first_date_is_a_date_object() {
    assert_eq!(
      interpret(&format!("{TS} ts[\"FirstDate\"]")).unwrap(),
      "DateObject[{2025, 9, 1, 0, 0, 0.}, Instant, Gregorian, 0.]"
    );
  }

  #[test]
  fn date_list_plot_renders_graphics() {
    assert_eq!(
      interpret(&format!("{TS} DateListPlot[ts]")).unwrap(),
      "-Graphics-"
    );
  }
}

// Vector-valued series with explicit date stamps: each value is a list (mixed
// numeric/string), the times are given as `{{date1, date2, …}}`, and Values /
// Normal / point lookup expose the path.
mod vector_valued {
  use super::*;

  const TS: &str = "ts = TimeSeries[\
    {{.1, \"cat\"}, {.2, \"dog\"}, {.3, \"fox\"}}, \
    {{DateObject[{2025, 1, 1}], DateObject[{2025, 1, 2}], \
      DateObject[{2025, 1, 3}]}}];";

  #[test]
  fn lookup_returns_the_vector_value() {
    assert_eq!(
      interpret(&format!("{TS} ts[DateObject[{{2025, 1, 2}}]]")).unwrap(),
      "{0.2, dog}"
    );
  }

  // `Today` resolves to a DateObject that exactly matches the middle stamp,
  // independent of the actual calendar date.
  #[test]
  fn lookup_by_today_symbol() {
    let ts = "ts = TimeSeries[{{.1, \"cat\"}, {.2, \"dog\"}, {.3, \"fox\"}}, \
      {{Yesterday, Today, Tomorrow}}];";
    assert_eq!(interpret(&format!("{ts} ts[Today]")).unwrap(), "{0.2, dog}");
  }

  #[test]
  fn values_returns_the_value_path() {
    assert_eq!(
      interpret(&format!("{TS} Values[ts]")).unwrap(),
      "{{0.1, cat}, {0.2, dog}, {0.3, fox}}"
    );
  }

  // A trailing list of string keys (the third positional argument) names the
  // components of each vector value (WL 15): each value becomes a keyed
  // association, the point lookup returns that association, and Values
  // materializes as a Tabular.
  const KEYED: &str = "ts = TimeSeries[\
    {{.1, \"cat\"}, {.2, \"dog\"}, {.3, \"fox\"}}, \
    {{DateObject[{2025, 1, 1}], DateObject[{2025, 1, 2}], \
      DateObject[{2025, 1, 3}]}}, {\"a\", \"b\"}];";

  #[test]
  fn keyed_lookup_returns_association() {
    assert_eq!(
      interpret(&format!("{KEYED} ts[DateObject[{{2025, 1, 2}}]]")).unwrap(),
      "<|a -> 0.2, b -> dog|>"
    );
  }

  #[test]
  fn keyed_values_is_a_tabular() {
    assert_eq!(
      interpret(&format!("{KEYED} Head[Values[ts]]")).unwrap(),
      "Tabular"
    );
  }

  #[test]
  fn keyed_normal_pairs_dates_with_associations() {
    assert_eq!(
      interpret(&format!("{KEYED} Normal[ts]")).unwrap(),
      "{{DateObject[{2025, 1, 1, 0, 0, 0}, Instant, Gregorian, 0.], \
        <|a -> 0.1, b -> cat|>}, \
       {DateObject[{2025, 1, 2, 0, 0, 0}, Instant, Gregorian, 0.], \
        <|a -> 0.2, b -> dog|>}, \
       {DateObject[{2025, 1, 3, 0, 0, 0}, Instant, Gregorian, 0.], \
        <|a -> 0.3, b -> fox|>}}"
    );
  }

  // Normal surfaces each stamp as an Instant DateObject. A `DateObject[{y,m,d}]`
  // source pads the time fields with integer zeros (matching WL).
  #[test]
  fn normal_pairs_instant_dates_with_values() {
    assert_eq!(
      interpret(&format!("{TS} Normal[ts]")).unwrap(),
      "{{DateObject[{2025, 1, 1, 0, 0, 0}, Instant, Gregorian, 0.], \
        {0.1, cat}}, \
       {DateObject[{2025, 1, 2, 0, 0, 0}, Instant, Gregorian, 0.], \
        {0.2, dog}}, \
       {DateObject[{2025, 1, 3, 0, 0, 0}, Instant, Gregorian, 0.], \
        {0.3, fox}}}"
    );
  }

  // A daily date-list series stores `0.`-seconds dates; Normal preserves the
  // Real zero (vs. the integer zeros above), as WL does.
  #[test]
  fn normal_preserves_real_seconds_of_generated_dates() {
    assert_eq!(
      interpret("Normal[TimeSeries[{23.1, 24.4}, {DateObject[{2025, 9, 1}]}]]")
        .unwrap(),
      "{{DateObject[{2025, 9, 1, 0, 0, 0.}, Instant, Gregorian, 0.], 23.1}, \
       {DateObject[{2025, 9, 2, 0, 0, 0.}, Instant, Gregorian, 0.], 24.4}}"
    );
  }

  // `{start, Automatic, "Day"}` is a date-range spec, not three explicit time
  // stamps — even when it happens to have as many elements as there are
  // values. The generated dates advance one day at a time from the start.
  #[test]
  fn range_spec_start_automatic_step_generates_dates() {
    assert_eq!(
      interpret(
        "Normal[TimeSeries[{10, 20, 30}, {{2013, 1, 1}, Automatic, \"Day\"}]]"
      )
      .unwrap(),
      "{{DateObject[{2013, 1, 1, 0, 0, 0.}, Instant, Gregorian, 0.], 10}, \
       {DateObject[{2013, 1, 2, 0, 0, 0.}, Instant, Gregorian, 0.], 20}, \
       {DateObject[{2013, 1, 3, 0, 0, 0.}, Instant, Gregorian, 0.], 30}}"
    );
  }
}

// Multi-component TimeSeries (ComponentKeys) plotting: ListPlot draws one
// series per component, and `ts -> "key"` selects named components.
mod component_time_series_plot {
  use super::*;

  const TS: &str = "ts = TimeSeries[Transpose@{{4, 9, 18}, {1, 9, 11}, \
     {5, 6, 7}}, {{2013, 1, 1}, Automatic, \"Day\"}, \
     ComponentKeys -> {\"a\", \"b\", \"c\"}];";

  fn svg_circles(code: &str) -> usize {
    clear_state();
    let svg =
      interpret(&format!("{TS} ExportString[{code}, \"SVG\"]")).unwrap();
    svg.matches("<circle").count()
  }

  #[test]
  fn component_keys_build_component_associations() {
    clear_state();
    let out = interpret(&format!("{TS} ts[[1, 1, 2]]")).unwrap();
    // The first value is the association <|a -> 4, b -> 1, c -> 5|>.
    assert_eq!(out, "<|a -> 4, b -> 1, c -> 5|>");
  }

  #[test]
  fn list_plot_draws_one_series_per_component() {
    // Three components × three time points = nine markers.
    assert_eq!(svg_circles("ListPlot[ts]"), 9);
  }

  #[test]
  fn list_plot_selects_single_component() {
    assert_eq!(svg_circles("ListPlot[ts -> \"b\"]"), 3);
  }

  #[test]
  fn list_plot_selects_multiple_components() {
    assert_eq!(svg_circles("ListPlot[ts -> {\"a\", \"c\"}]"), 6);
  }
}

mod series_transformations {
  use super::*;

  const TS: &str = "ts = TimeSeries[{{1, 10}, {2, 20}}]; ";

  #[test]
  fn time_series_shift_moves_the_stamps() {
    assert_eq!(
      interpret(&format!("{TS}TimeSeriesShift[ts, 5][\"Path\"]")).unwrap(),
      "{{6, 10}, {7, 20}}"
    );
    assert_eq!(
      interpret(&format!("{TS}TimeSeriesShift[ts, -1][\"Times\"]")).unwrap(),
      "{0, 1}"
    );
    assert_eq!(
      interpret(&format!("{TS}Head[TimeSeriesShift[ts, 5]]")).unwrap(),
      "TimeSeries"
    );
  }

  #[test]
  fn time_series_map_transforms_the_values() {
    assert_eq!(
      interpret(&format!("{TS}TimeSeriesMap[f, ts][\"Path\"]")).unwrap(),
      "{{1, f[10]}, {2, f[20]}}"
    );
    assert_eq!(
      interpret(&format!("{TS}TimeSeriesMap[# + 1 &, ts][\"Path\"]")).unwrap(),
      "{{1, 11}, {2, 21}}"
    );
  }

  // The values several series share at a time stamp go to the function as a
  // list.
  #[test]
  fn time_series_thread_combines_series() {
    assert_eq!(
      interpret(&format!(
        "{TS}TimeSeriesThread[Total, {{ts, TimeSeries[{{{{1, 1}}, {{2, 2}}}}]}}][\"Path\"]"
      ))
      .unwrap(),
      "{{1, 11}, {2, 22}}"
    );
    assert_eq!(
      interpret(&format!(
        "{TS}TimeSeriesThread[Mean, {{ts, TimeSeries[{{{{1, 20}}, {{2, 40}}}}]}}][\"Path\"]"
      ))
      .unwrap(),
      "{{1, 15}, {2, 30}}"
    );
  }

  // Fewer than three stamps are trivially evenly spaced.
  #[test]
  fn regularly_sampled_q() {
    assert_eq!(
      interpret("RegularlySampledQ[TimeSeries[{{1, 10}, {2, 20}, {3, 30}}]]")
        .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("RegularlySampledQ[TimeSeries[{{1, 10}, {2, 20}, {4, 40}}]]")
        .unwrap(),
      "False"
    );
    assert_eq!(
      interpret("RegularlySampledQ[TimeSeries[{{1, 10}}]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn time_series_insert_keeps_the_path_sorted() {
    assert_eq!(
      interpret(
        "TimeSeriesInsert[TimeSeries[{{1, 10}, {3, 30}}], {2, 20}][\"Path\"]"
      )
      .unwrap(),
      "{{1, 10}, {2, 20}, {3, 30}}"
    );
  }

  #[test]
  fn first_and_last_time_and_value_dimensions() {
    assert_eq!(interpret(&format!("{TS}ts[\"FirstTime\"]")).unwrap(), "1");
    assert_eq!(interpret(&format!("{TS}ts[\"LastTime\"]")).unwrap(), "2");
    assert_eq!(
      interpret(&format!("{TS}ts[\"ValueDimensions\"]")).unwrap(),
      "1"
    );
  }

  // An EventSeries answers the same path queries and survives a map.
  #[test]
  fn event_series() {
    assert_eq!(
      interpret(r#"EventSeries[{{1, "a"}, {2, "b"}}]["Times"]"#).unwrap(),
      "{1, 2}"
    );
    assert_eq!(
      interpret(r#"EventSeries[{{1, "a"}, {2, "b"}}]["Path"]"#).unwrap(),
      "{{1, a}, {2, b}}"
    );
    assert_eq!(
      interpret("Head[TimeSeriesMap[# * 2 &, EventSeries[{{1, 10}}]]]")
        .unwrap(),
      "EventSeries"
    );
  }

  // Arithmetic keeps the time stamps and works on the values.
  #[test]
  fn arithmetic_threads_over_the_values() {
    assert_eq!(
      interpret(&format!("{TS}Normal[ts + 1]")).unwrap(),
      "{{1, 11}, {2, 21}}"
    );
    assert_eq!(
      interpret(&format!("{TS}Normal[2*ts]")).unwrap(),
      "{{1, 20}, {2, 40}}"
    );
    assert_eq!(
      interpret(&format!("{TS}Normal[ts^2]")).unwrap(),
      "{{1, 100}, {2, 400}}"
    );
    assert_eq!(
      interpret(&format!("{TS}Normal[-ts]")).unwrap(),
      "{{1, -10}, {2, -20}}"
    );
    assert_eq!(
      interpret(&format!("{TS}Normal[ts/2]")).unwrap(),
      "{{1, 5}, {2, 10}}"
    );
    assert_eq!(
      interpret("Normal[Sqrt[TimeSeries[{{1, 4}, {2, 9}}]]]").unwrap(),
      "{{1, 2}, {2, 3}}"
    );
  }

  #[test]
  fn two_series_combine_point_by_point() {
    assert_eq!(
      interpret(&format!(
        "{TS}Normal[ts + TimeSeries[{{{{1, 1}}, {{2, 2}}}}]]"
      ))
      .unwrap(),
      "{{1, 11}, {2, 22}}"
    );
  }

  // A moving average over a series is stamped with the last time of each
  // window.
  #[test]
  fn moving_average_over_a_series() {
    assert_eq!(
      interpret(
        "MovingAverage[TimeSeries[{{1, 10}, {2, 20}, {3, 30}}], 2][\"Path\"]"
      )
      .unwrap(),
      "{{2, 15.}, {3, 25.}}"
    );
  }
}

// TimeSeriesRescale carries the time stamps linearly onto a given span,
// keeping the values and the head. Values verified against wolframscript.
mod time_series_rescale {
  use super::*;

  /// The result of `code`, written the way `InputForm` writes it.
  fn form(code: &str) -> String {
    interpret(&format!("ToString[{code}, InputForm]")).unwrap()
  }

  #[test]
  fn the_stamps_land_on_the_span_keeping_their_spacing() {
    clear_state();
    for (code, expected) in [
      (
        "TimeSeriesRescale[TimeSeries[{{1, 10}, {2, 20}}], {0, 1}][\"Times\"]",
        "{0, 1}",
      ),
      (
        "TimeSeriesRescale[TimeSeries[{{1, 10}, {2, 20}}], {5, 7}][\"Times\"]",
        "{5, 7}",
      ),
      (
        "TimeSeriesRescale[TimeSeries[{{0, 1}, {10, 2}}], {0, 100}][\"Times\"]",
        "{0, 100}",
      ),
      // Uneven spacing survives, exactly.
      (
        "TimeSeriesRescale[TimeSeries[{{1, 10}, {2, 20}, {4, 40}}], \
         {0, 1}][\"Times\"]",
        "{0, 1/3, 1}",
      ),
      (
        "TimeSeriesRescale[TimeSeries[{{1, 10}, {2, 20}, {3, 30}}], \
         {0, 1}][\"Times\"]",
        "{0, 1/2, 1}",
      ),
      (
        "TimeSeriesRescale[TimeSeries[{{1, 10}, {2, 20}, {4, 40}}], \
         {0, 6}][\"Times\"]",
        "{0, 2, 6}",
      ),
      // Bare values are stamped 1, 2, 3, … before rescaling.
      (
        "TimeSeriesRescale[TimeSeries[{10, 20, 30}], {0, 1}][\"Times\"]",
        "{0, 1/2, 1}",
      ),
    ] {
      let code = code.replace("         ", "");
      assert_eq!(form(&code), expected, "{code}");
    }
  }

  #[test]
  fn the_values_and_the_head_are_left_alone() {
    clear_state();
    assert_eq!(
      form(
        "TimeSeriesRescale[TimeSeries[{{1, 10}, {2, 20}, {4, 40}}], \
         {0, 1}][\"Path\"]"
          .replace("         ", "")
          .as_str()
      ),
      "{{0, 10}, {1/3, 20}, {1, 40}}"
    );
    assert_eq!(
      interpret(
        "Head[TimeSeriesRescale[TimeSeries[{{1, 10}, {2, 20}}], \
       {0, 1}]]"
          .replace("       ", "")
          .as_str()
      )
      .unwrap(),
      "TimeSeries"
    );
    // An EventSeries stays one.
    assert_eq!(
      interpret(
        "Head[TimeSeriesRescale[EventSeries[{{1, 10}, {2, 20}}], \
       {0, 1}]]"
          .replace("       ", "")
          .as_str()
      )
      .unwrap(),
      "EventSeries"
    );
    assert_eq!(
      form(
        "TimeSeriesRescale[EventSeries[{{1, 10}, {2, 20}}], {0, 1}][\"Times\"]"
      ),
      "{0, 1}"
    );
  }

  #[test]
  fn a_span_that_does_not_increase_is_refused() {
    clear_state();
    for code in [
      "TimeSeriesRescale[TimeSeries[{{1, 10}, {2, 20}}], {1, 0}]",
      "TimeSeriesRescale[TimeSeries[{{1, 10}, {2, 20}}], {0, 0}]",
    ] {
      let result = interpret_with_stdout(code).unwrap();
      assert!(
        result
          .warnings
          .iter()
          .any(|w| w.contains("TimeSeriesRescale::trng")),
        "expected ::trng for {code}, got {:?}",
        result.warnings
      );
    }
    let result =
      interpret_with_stdout("TimeSeriesRescale[TimeSeries[{{1, 10}}]]")
        .unwrap();
    assert!(
      result
        .warnings
        .iter()
        .any(|w| w.contains("TimeSeriesRescale::argr")),
      "expected ::argr, got {:?}",
      result.warnings
    );
    // A span that is not a pair leaves the series as it was.
    assert_eq!(
      form("TimeSeriesRescale[TimeSeries[{{1, 10}, {2, 20}}], 5][\"Times\"]"),
      "{1, 2}"
    );
  }

  // A series runs in time order however its points were written.
  #[test]
  fn out_of_order_stamps_are_sorted() {
    clear_state();
    assert_eq!(
      form("TimeSeries[{{1, 10}, {5, 50}, {2, 20}}][\"Path\"]"),
      "{{1, 10}, {2, 20}, {5, 50}}"
    );
    assert_eq!(
      form("TimeSeries[{{1, 10}, {5, 50}, {2, 20}}][\"Times\"]"),
      "{1, 2, 5}"
    );
    assert_eq!(
      form(
        "TimeSeriesRescale[TimeSeries[{{1, 10}, {5, 50}, {2, 20}}], \
         {0, 1}][\"Times\"]"
          .replace("         ", "")
          .as_str()
      ),
      "{0, 1/4, 1}"
    );
  }
}

// MovingMap over a series windows by TIME rather than by count: `f` sees the
// values whose stamps fall in `[t - n, t]`, and the result is stamped at `t`.
// Values verified against wolframscript.
mod moving_map_series {
  use super::*;

  /// The result of `code`, written the way `InputForm` writes it.
  fn form(code: &str) -> String {
    interpret(&format!("ToString[{code}, InputForm]")).unwrap()
  }

  #[test]
  fn the_window_is_a_span_of_time() {
    clear_state();
    for (code, expected) in [
      (
        "MovingMap[Total, TimeSeries[{{1, 1}, {2, 2}, {3, 3}}], 1][\"Path\"]",
        "{{2, 3}, {3, 5}}",
      ),
      // Unevenly spaced stamps put different numbers of points in a window.
      (
        "MovingMap[Total, TimeSeries[{{1, 1}, {2, 2}, {4, 4}, {7, 7}}], \
         2][\"Path\"]",
        "{{4, 6}, {7, 7}}",
      ),
      (
        "MovingMap[Total, TimeSeries[{{1, 1}, {2, 2}, {4, 4}, {7, 7}}], \
         3][\"Path\"]",
        "{{4, 7}, {7, 11}}",
      ),
      (
        "MovingMap[Length, TimeSeries[{{1, 1}, {2, 2}, {4, 4}, {7, 7}}], \
         3][\"Path\"]",
        "{{4, 3}, {7, 2}}",
      ),
      // A one-element spec means the same as the bare width.
      (
        "MovingMap[Total, TimeSeries[{{1, 1}, {2, 2}, {3, 3}}], {1}][\"Path\"]",
        "{{2, 3}, {3, 5}}",
      ),
      // The width need not be whole.
      (
        "MovingMap[Total, TimeSeries[{{1, 1}, {2, 2}, {3, 3}}], 1.5][\"Path\"]",
        "{{3, 5}}",
      ),
      (
        "MovingMap[Max, TimeSeries[{{1, 5}, {2, 2}, {3, 9}}], 1][\"Path\"]",
        "{{2, 5}, {3, 9}}",
      ),
      // Any function sees the window as a list.
      (
        "MovingMap[f, TimeSeries[{{1, 1}, {2, 2}, {3, 3}}], 1][\"Path\"]",
        "{{2, f[{1, 2}]}, {3, f[{2, 3}]}}",
      ),
    ] {
      let code = code.replace("         ", "");
      assert_eq!(form(&code), expected, "{code}");
    }
  }

  // A window that would reach back past the start of the series is not one,
  // so the result is shorter at the front — and empty when nothing fits.
  #[test]
  fn a_window_that_does_not_fit_is_dropped() {
    clear_state();
    assert_eq!(
      form(
        "MovingMap[Total, TimeSeries[{{1, 1}, {2, 2}, {3, 3}}], 5][\"Path\"]"
      ),
      "{}"
    );
    assert_eq!(
      form(
        "MovingMap[Total, TimeSeries[{{1, 1}, {2, 2}, {3, 3}, {4, 4}}], \
         2][\"Path\"]"
          .replace("         ", "")
          .as_str()
      ),
      "{{3, 6}, {4, 9}}"
    );
    assert_eq!(
      form(
        "MovingMap[Total, TimeSeries[{{1, 1}, {2, 2}, {3, 3}}], 1][\"Times\"]"
      ),
      "{2, 3}"
    );
  }

  #[test]
  fn the_head_of_the_series_is_kept() {
    clear_state();
    assert_eq!(
      interpret(
        "Head[MovingMap[Total, TimeSeries[{{1, 1}, {2, 2}, {3, 3}}], 1]]"
      )
      .unwrap(),
      "TimeSeries"
    );
    assert_eq!(
      interpret(
        "Head[MovingMap[Total, EventSeries[{{1, 1}, {2, 2}, {3, 3}}], 1]]"
      )
      .unwrap(),
      "EventSeries"
    );
    assert_eq!(
      form(
        "MovingMap[Total, EventSeries[{{1, 1}, {2, 2}, {3, 3}}], 1][\"Path\"]"
      ),
      "{{2, 3}, {3, 5}}"
    );
  }

  // A plain list still windows by count, which is a different thing.
  #[test]
  fn a_list_still_windows_by_count() {
    clear_state();
    for (code, expected) in [
      ("MovingMap[Total, {1, 2, 3, 4}, 1]", "{3, 5, 7}"),
      ("MovingMap[Total, {1, 2, 3, 4}, {2}]", "{6, 9}"),
      ("MovingMap[f, {1, 2, 3}, 1]", "{f[{1, 2}], f[{2, 3}]}"),
    ] {
      assert_eq!(form(code), expected, "{code}");
    }
  }
}
