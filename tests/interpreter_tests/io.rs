use super::*;

mod date_string {
  use super::*;

  #[test]
  fn date_string_returns_string() {
    // DateString should return a string (not cause parse error)
    let result = interpret("StringQ[DateString[]]").unwrap();
    assert_eq!(result, "True");
  }

  #[test]
  fn date_string_with_now() {
    let result = interpret("StringQ[DateString[Now]]").unwrap();
    assert_eq!(result, "True");
  }

  #[test]
  fn date_string_iso_format() {
    // ISODateTime format should contain T separator
    let result =
      interpret("StringContainsQ[DateString[Now, \"ISODateTime\"], \"T\"]")
        .unwrap();
    assert_eq!(result, "True");
  }
}

mod directory {
  use super::*;

  #[test]
  fn directory_returns_string() {
    let result = interpret("StringQ[Directory[]]").unwrap();
    assert_eq!(result, "True");
  }

  #[test]
  fn directory_returns_nonempty_string() {
    let result = interpret("StringLength[Directory[]] > 0").unwrap();
    assert_eq!(result, "True");
  }

  #[test]
  fn directory_contains_separator() {
    // Any real directory path should contain a path separator
    let result = interpret("StringContainsQ[Directory[], \"/\"]").unwrap();
    assert_eq!(result, "True");
  }
}

mod create_file {
  use super::*;

  #[test]
  fn create_file_returns_string() {
    // CreateFile should return a string path (not cause parse error)
    let result = interpret("StringQ[CreateFile[]]").unwrap();
    assert_eq!(result, "True");
  }

  #[test]
  fn create_file_path_exists() {
    // The returned path should be a non-empty string
    let result = interpret("StringLength[CreateFile[]] > 0").unwrap();
    assert_eq!(result, "True");
  }
}

mod run {
  use super::*;

  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn run_echo() {
    assert_eq!(interpret(r#"Run["true"]"#).unwrap(), "0");
  }
}

mod plot {
  use super::*;

  #[test]
  fn plot_returns_graphics() {
    clear_state();
    assert_eq!(
      interpret("Plot[Sin[x], {x, 0, 2 Pi}]").unwrap(),
      "-Graphics-"
    );
  }

  #[test]
  fn plot_produces_svg() {
    clear_state();
    let result = interpret_with_stdout("Plot[Sin[x], {x, 0, 2 Pi}]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("</svg>"));
    assert!(
      svg.contains("<polyline")
        || svg.contains("<line")
        || svg.contains("<path")
        || svg.contains("<circle")
    );
  }

  #[test]
  fn plot_cos() {
    clear_state();
    let result = interpret_with_stdout("Plot[Cos[x], {x, 0, 2 Pi}]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
  }

  #[test]
  fn plot_polynomial() {
    clear_state();
    let result = interpret_with_stdout("Plot[x^2, {x, -2, 2}]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
  }

  #[test]
  fn plot_expression() {
    clear_state();
    let result = interpret_with_stdout("Plot[x^2 - 1, {x, -3, 3}]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
  }

  #[test]
  fn plot_with_numeric_range() {
    clear_state();
    let result = interpret_with_stdout("Plot[Sin[x], {x, 0, 6}]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
  }

  #[test]
  fn plot_unevaluated_with_one_arg() {
    clear_state();
    // Like Wolfram, Plot with wrong args returns unevaluated
    assert_eq!(interpret("Plot[Sin[x]]").unwrap(), "Plot[Sin[x]]");
  }

  #[test]
  fn plot_error_bad_iterator() {
    clear_state();
    assert!(interpret("Plot[Sin[x], 5]").is_err());
  }

  #[test]
  fn plot_no_graphics_for_non_plot() {
    clear_state();
    let result = interpret_with_stdout("1 + 2").unwrap();
    assert_eq!(result.result, "3");
    assert!(result.graphics.is_none());
  }

  #[test]
  fn plot_svg_has_viewbox() {
    clear_state();
    let result = interpret_with_stdout("Plot[Sin[x], {x, 0, 6}]").unwrap();
    let svg = result.graphics.unwrap();
    assert!(svg.contains("viewBox="));
    assert!(svg.contains("width=\"360\""));
    assert!(svg.contains("height=\"225\""));
  }

  #[test]
  fn plot_image_size_integer() {
    clear_state();
    let result =
      interpret_with_stdout("Plot[Sin[x], {x, 0, 6}, ImageSize -> 600]")
        .unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.contains("width=\"600\""));
    assert!(svg.contains("height=\"375\""));
  }

  #[test]
  fn plot_image_size_pair() {
    clear_state();
    let result =
      interpret_with_stdout("Plot[Sin[x], {x, 0, 6}, ImageSize -> {800, 300}]")
        .unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.contains("width=\"800\""));
    assert!(svg.contains("height=\"300\""));
  }

  #[test]
  fn plot_image_size_named() {
    clear_state();
    let result =
      interpret_with_stdout("Plot[Sin[x], {x, 0, 6}, ImageSize -> Large]")
        .unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.contains("width=\"480\""));
    assert!(svg.contains("height=\"300\""));
  }

  #[test]
  fn plot_image_size_tiny() {
    clear_state();
    let result =
      interpret_with_stdout("Plot[Sin[x], {x, 0, 6}, ImageSize -> Tiny]")
        .unwrap();
    let svg = result.graphics.unwrap();
    assert!(svg.contains("width=\"100\""));
    assert!(svg.contains("height=\"63\""));
  }

  #[test]
  fn plot_image_size_small() {
    clear_state();
    let result =
      interpret_with_stdout("Plot[Sin[x], {x, 0, 6}, ImageSize -> Small]")
        .unwrap();
    let svg = result.graphics.unwrap();
    assert!(svg.contains("width=\"200\""));
    assert!(svg.contains("height=\"125\""));
  }

  #[test]
  fn plot_image_size_medium() {
    clear_state();
    let result =
      interpret_with_stdout("Plot[Sin[x], {x, 0, 6}, ImageSize -> Medium]")
        .unwrap();
    let svg = result.graphics.unwrap();
    assert!(svg.contains("width=\"360\""));
    assert!(svg.contains("height=\"225\""));
  }

  #[test]
  fn plot_image_size_full() {
    clear_state();
    let result =
      interpret_with_stdout("Plot[Sin[x], {x, 0, 6}, ImageSize -> Full]")
        .unwrap();
    let svg = result.graphics.unwrap();
    // Full uses width="100%" to scale with the container
    assert!(svg.contains("width=\"100%\""));
    // viewBox still uses the 720x450 render base for proper aspect ratio
    assert!(svg.contains("viewBox=\"0 0 7200 4500\""));
  }

  #[test]
  fn plot_image_size_default_unchanged() {
    // Without ImageSize option, default 360x225 is used
    clear_state();
    let result = interpret_with_stdout("Plot[Sin[x], {x, 0, 6}]").unwrap();
    let svg = result.graphics.unwrap();
    assert!(svg.contains("width=\"360\""));
    assert!(svg.contains("height=\"225\""));
  }

  #[test]
  fn export_plot_to_svg() {
    clear_state();
    let path = "/tmp/woxi_test_export_plot.svg";
    let result =
      interpret(&format!("Export[\"{path}\", Plot[Sin[x], {{x, -10, 10}}]]"))
        .unwrap();
    assert_eq!(result, path);
    let content = std::fs::read_to_string(path).unwrap();
    assert!(content.starts_with("<svg"));
    assert!(content.contains("</svg>"));
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn export_string_to_file() {
    clear_state();
    let path = "/tmp/woxi_test_export_string.txt";
    let result =
      interpret(&format!("Export[\"{path}\", \"hello world\"]")).unwrap();
    assert_eq!(result, path);
    let content = std::fs::read_to_string(path).unwrap();
    assert_eq!(content, "hello world");
    std::fs::remove_file(path).ok();
  }
}

mod echo {
  use super::*;

  #[test]
  fn single_arg_returns_value() {
    clear_state();
    let result = interpret_with_stdout("Echo[42]").unwrap();
    assert_eq!(result.result, "42");
    assert_eq!(result.stdout, ">> 42\n");
  }

  #[test]
  fn single_arg_with_expression() {
    clear_state();
    let result = interpret_with_stdout("Echo[2 + 3]").unwrap();
    assert_eq!(result.result, "5");
    assert_eq!(result.stdout, ">> 5\n");
  }

  #[test]
  fn two_args_custom_label() {
    clear_state();
    let result = interpret_with_stdout("Echo[2 + 3, \"result: \"]").unwrap();
    assert_eq!(result.result, "5");
    assert_eq!(result.stdout, ">> result:  5\n");
  }

  #[test]
  fn three_args_with_function() {
    clear_state();
    let result =
      interpret_with_stdout("Echo[{1, 2, 3}, \"length: \", Length]").unwrap();
    assert_eq!(result.result, "{1, 2, 3}");
    assert_eq!(result.stdout, ">> length:  3\n");
  }

  #[test]
  fn map_echo() {
    clear_state();
    let result = interpret_with_stdout("Map[Echo, {1, 2, 3}]").unwrap();
    assert_eq!(result.result, "{1, 2, 3}");
    assert_eq!(result.stdout, ">> 1\n>> 2\n>> 3\n");
  }

  #[test]
  fn echo_with_list() {
    clear_state();
    let result = interpret_with_stdout("Echo[{a, b, c}]").unwrap();
    assert_eq!(result.result, "{a, b, c}");
    assert_eq!(result.stdout, ">> {a, b, c}\n");
  }

  #[test]
  fn echo_with_string() {
    clear_state();
    let result = interpret_with_stdout("Echo[\"hello\"]").unwrap();
    assert_eq!(result.result, "hello");
    assert_eq!(result.stdout, ">> hello\n");
  }
}
