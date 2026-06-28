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
}
