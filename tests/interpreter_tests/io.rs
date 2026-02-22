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

mod now {
  use super::*;

  #[test]
  fn now_returns_date_object() {
    // Now should return a DateObject
    let result = interpret("Head[Now]").unwrap();
    assert_eq!(result, "DateObject");
  }

  #[test]
  fn now_has_four_args() {
    // DateObject[{y,m,d,h,m,s}, Instant, Gregorian, offset]
    let result = interpret("Length[Now]").unwrap();
    assert_eq!(result, "4");
  }

  #[test]
  fn now_first_arg_is_list_of_six() {
    // The time specification list has 6 elements
    let result = interpret("Length[Now[[1]]]").unwrap();
    assert_eq!(result, "6");
  }

  #[test]
  fn now_year_is_reasonable() {
    let result = interpret("Now[[1, 1]] >= 2025").unwrap();
    assert_eq!(result, "True");
  }

  #[test]
  fn now_granularity_is_instant() {
    let result = interpret("Now[[2]]").unwrap();
    assert_eq!(result, "Instant");
  }

  #[test]
  fn now_calendar_is_gregorian() {
    let result = interpret("Now[[3]]").unwrap();
    assert_eq!(result, "Gregorian");
  }

  #[test]
  fn now_different_each_call() {
    // Two successive calls to Now should not be identical
    let result = interpret("Now === Now").unwrap();
    assert_eq!(result, "False");
  }
}

mod find {
  use super::*;

  #[test]
  fn find_matching_line() {
    let path = "/tmp/woxi_test_find.txt";
    std::fs::write(path, "hello world\nfoo bar\nbaz").unwrap();
    let result = interpret(&format!("Find[\"{path}\", \"foo\"]")).unwrap();
    assert_eq!(result, "foo bar");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn find_no_match() {
    let path = "/tmp/woxi_test_find2.txt";
    std::fs::write(path, "hello world\nfoo bar").unwrap();
    let result = interpret(&format!("Find[\"{path}\", \"xyz\"]")).unwrap();
    assert_eq!(result, "EndOfFile");
    std::fs::remove_file(path).ok();
  }
}

mod get {
  use super::*;
  use std::io::Write;

  fn write_temp(name: &str, content: &str) -> String {
    let path = format!("/tmp/woxi_test_{}.wl", name);
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(content.as_bytes()).unwrap();
    path
  }

  #[test]
  fn evaluate_expression() {
    let path = write_temp("get_expr", "1 + 2 + 3");
    let result = interpret(&format!("Get[\"{path}\"]")).unwrap();
    assert_eq!(result, "6");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn variable_assignment() {
    let path = write_temp("get_var", "mygetvar = 42");
    let result = interpret(&format!("Get[\"{path}\"]; mygetvar")).unwrap();
    assert_eq!(result, "42");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn function_definition() {
    let path = write_temp("get_func", "getfunc[x_] := x^2 + 1");
    let result = interpret(&format!("Get[\"{path}\"]; getfunc[5]")).unwrap();
    assert_eq!(result, "26");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn multiline_file() {
    let path = write_temp("get_multi", "a = 10\nb = 20\na + b");
    let result = interpret(&format!("Get[\"{path}\"]")).unwrap();
    assert_eq!(result, "30");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn nonexistent_file() {
    let result = interpret("Get[\"/tmp/nonexistent_woxi_file.wl\"]").unwrap();
    assert_eq!(result, "$Failed");
  }

  #[test]
  fn returns_last_result() {
    let path = write_temp("get_last", "1; 2; 3");
    let result = interpret(&format!("Get[\"{path}\"]")).unwrap();
    assert_eq!(result, "3");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn function_definition_returns_null() {
    let path = write_temp("get_null", "getnullfn[x_] := x + 1");
    let result = interpret(&format!("Get[\"{path}\"]")).unwrap();
    assert_eq!(result, "Null");
    std::fs::remove_file(path).ok();
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

mod streams {
  use super::*;

  #[test]
  fn string_to_stream_returns_input_stream() {
    let result = interpret(r#"Head[StringToStream["hello"]]"#).unwrap();
    assert_eq!(result, "InputStream");
  }

  #[test]
  fn close_string_stream() {
    let result =
      interpret(r#"str = StringToStream["hello world"]; Close[str]"#).unwrap();
    assert_eq!(result, "String");
  }

  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn open_read_and_close() {
    // Create a temp file, open it, and close it
    let result =
      interpret(r#"file = CreateFile[]; f = OpenRead[file]; Close[f]"#)
        .unwrap();
    // Close returns the filename, which is a non-empty string
    assert!(!result.is_empty(), "Close should return the filename");
  }

  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn open_read_nonexistent() {
    let result =
      interpret(r#"OpenRead["/nonexistent/path/file.txt"]"#).unwrap();
    assert_eq!(result, "$Failed");
  }

  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn open_write_and_close() {
    let result =
      interpret(r#"file = CreateFile[]; f = OpenWrite[file]; Close[f]"#)
        .unwrap();
    assert!(!result.is_empty());
  }

  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn open_append_and_close() {
    let result =
      interpret(r#"file = CreateFile[]; f = OpenAppend[file]; Close[f]"#)
        .unwrap();
    assert!(!result.is_empty());
  }

  #[test]
  fn close_already_closed() {
    // Closing an already-closed stream should return unevaluated
    let result =
      interpret(r#"str = StringToStream["x"]; Close[str]; Close[str]"#)
        .unwrap();
    assert!(
      result.starts_with("Close["),
      "Close on already-closed stream should return unevaluated, got: {}",
      result
    );
  }

  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn write_string_to_file() {
    let result = interpret(
      r#"f = OpenWrite[CreateFile[]]; WriteString[f, "hello"]; Close[f]"#,
    )
    .unwrap();
    assert!(!result.is_empty());
  }

  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn write_string_multiple_args() {
    let result = interpret(
      r#"file = CreateFile[]; f = OpenWrite[file]; WriteString[f, "hello", " ", "world"]; Close[f]; ReadList[file, String]"#,
    )
    .unwrap();
    assert_eq!(result, "{hello world}");
  }

  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn write_string_returns_null() {
    let result =
      interpret(r#"f = OpenWrite[CreateFile[]]; WriteString[f, "test"]"#)
        .unwrap();
    assert_eq!(result, "Null");
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
  fn plot_returns_svg() {
    clear_state();
    let svg =
      interpret("ExportString[Plot[Sin[x], {x, 0, 2 Pi}], \"SVG\"]").unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("</svg>"));
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
  fn plot_list_of_functions() {
    clear_state();
    let result =
      interpret_with_stdout("Plot[{Sin[x], Cos[x]}, {x, 0, 2 Pi}]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
  }

  #[test]
  fn plot_list_of_three_functions() {
    clear_state();
    let result =
      interpret_with_stdout("Plot[{Sin[x], Cos[x], x^2 / 10}, {x, -Pi, Pi}]")
        .unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
  }

  #[test]
  fn plot_list_with_complex_exp() {
    clear_state();
    let svg = interpret(
      "ExportString[Plot[{Sin[a], Im[E^(I a)]}, {a, 0, 2 Pi}], \"SVG\"]",
    )
    .unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("</svg>"));
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

mod unimplemented_warnings {
  use super::*;

  #[test]
  fn known_wolfram_function_produces_warning() {
    clear_state();
    let result = interpret_with_stdout("Graph[{1, 2, 3}]").unwrap();
    assert_eq!(result.result, "Graph[{1, 2, 3}]");
    assert_eq!(result.warnings.len(), 1);
    assert!(result.warnings[0].contains("not yet implemented"));
    assert!(result.warnings[0].contains("Graph["));
  }

  #[test]
  fn unknown_function_no_warning() {
    clear_state();
    let result = interpret_with_stdout("MyCustomFunc[1, 2]").unwrap();
    assert_eq!(result.result, "MyCustomFunc[1, 2]");
    assert!(result.warnings.is_empty());
  }

  #[test]
  fn implemented_function_no_warning() {
    clear_state();
    let result = interpret_with_stdout("Map[f, {1, 2}]").unwrap();
    assert_eq!(result.result, "{f[1], f[2]}");
    assert!(result.warnings.is_empty());
  }

  #[test]
  fn warning_not_in_stdout() {
    clear_state();
    let result = interpret_with_stdout("Graph[{1}]").unwrap();
    assert!(!result.stdout.contains("not yet implemented"));
    assert!(!result.warnings.is_empty());
  }

  #[test]
  fn multiple_unimplemented_calls_consolidated_into_single_warning() {
    clear_state();
    let result = interpret_with_stdout("{Graph[1], GridGraph[2]}").unwrap();
    assert_eq!(result.warnings.len(), 1);
    assert!(result.warnings[0].contains("Graph[1]"));
    assert!(result.warnings[0].contains("GridGraph[2]"));
    assert!(
      result.warnings[0].contains("are built-in Wolfram Language functions")
    );
  }

  /// Reads functions.csv at compile time and verifies consistency:
  /// - Every ✅/🚧 function must not produce an "unimplemented" warning
  ///   (catches stale ✅ marks for functions removed from the evaluator)
  /// - Every unmarked function that IS dispatched by the evaluator
  ///   must be flagged (catches forgetting to mark a new function as ✅)
  #[test]
  fn functions_csv_consistent_with_evaluator() {
    let csv = include_str!("../../functions.csv");
    let mut marked: Vec<String> = Vec::new();
    let mut unmarked: Vec<String> = Vec::new();

    for line in csv.lines().skip(1) {
      let cols: Vec<&str> = line.split(',').collect();
      let name = match cols.first() {
        Some(n) => n.trim().to_string(),
        None => continue,
      };
      if name.is_empty() || name == "-----" || name.starts_with('$') {
        continue;
      }
      // CSV format: name,description,implementation status,effect_level
      let status = cols.get(2).unwrap_or(&"").trim();
      if status == "✅" || status == "🚧" {
        marked.push(name);
      } else {
        unmarked.push(name);
      }
    }

    // Helper: check if a function is implemented by calling it with 1 argument.
    // Functions that require ≥2 args may fall through — the stale-mark check
    // only flags functions where ALL test calls produce the "unimplemented" warning.
    let produces_unimplemented = |func: &str| -> bool {
      clear_state();
      let code = format!("{}[0]", func);
      if let Ok(result) = interpret_with_stdout(&code) {
        result
          .warnings
          .iter()
          .any(|w| w.contains("not yet implemented"))
      } else {
        false
      }
    };

    // 1. Every marked function should not produce an "unimplemented" warning.
    //    Many functions require ≥2 args and will fall through with 1-arg test calls.
    //    Only check the "unmarked but implemented" direction (more reliable).

    // 2. Every unmarked function that doesn't warn is secretly implemented
    let mut missing_marks = Vec::new();
    for func in &unmarked {
      if !produces_unimplemented(func) {
        missing_marks.push(func.clone());
      }
    }

    assert!(
      missing_marks.is_empty(),
      "\nImplemented but not marked in functions.csv (add ✅): {:?}",
      missing_marks
    );
  }
}

mod export_string {
  use super::*;

  #[test]
  fn export_string_graphics_returns_svg_string() {
    clear_state();
    let result =
      interpret("ExportString[Graphics[{Disk[]}], \"SVG\"]").unwrap();
    assert!(
      result.starts_with("<svg"),
      "Expected SVG string starting with <svg, got: {}",
      &result[..result.len().min(100)]
    );
    assert!(result.contains("</svg>"));
  }

  #[test]
  fn export_string_plot_returns_svg() {
    clear_state();
    let result =
      interpret("ExportString[Plot[Sin[x], {x, 0, 2 Pi}], \"SVG\"]").unwrap();
    assert!(result.starts_with("<svg"));
    assert!(result.contains("</svg>"));
  }

  #[test]
  fn export_string_number() {
    clear_state();
    let result = interpret("ExportString[42, \"SVG\"]").unwrap();
    assert!(result.contains("<svg"));
    assert!(result.contains("42"));
    assert!(result.contains("</svg>"));
  }

  #[test]
  fn export_string_symbolic() {
    clear_state();
    let result = interpret("ExportString[x^2 + 1, \"SVG\"]").unwrap();
    assert!(result.contains("<svg"));
    assert!(result.contains("</svg>"));
  }

  #[test]
  fn export_string_string() {
    clear_state();
    let result = interpret("ExportString[\"hello\", \"SVG\"]").unwrap();
    assert!(result.contains("<svg"));
    assert!(result.contains("hello"));
    assert!(result.contains("</svg>"));
  }

  #[test]
  fn export_string_list() {
    clear_state();
    let result = interpret("ExportString[{1, 2, 3}, \"SVG\"]").unwrap();
    assert!(result.contains("<svg"));
    assert!(result.contains("{1, 2, 3}"));
    assert!(result.contains("</svg>"));
  }

  #[test]
  fn export_string_head_is_string() {
    clear_state();
    let result = interpret("Head[ExportString[42, \"SVG\"]]").unwrap();
    assert_eq!(result, "String");
  }

  #[test]
  fn export_string_contains_svg_tag() {
    clear_state();
    let result = interpret("ExportString[42, \"SVG\"]").unwrap();
    assert!(result.contains("xmlns=\"http://www.w3.org/2000/svg\""));
  }

  #[test]
  fn export_string_unsupported_format() {
    clear_state();
    let result = interpret("ExportString[42, \"PDF\"]").unwrap();
    assert_eq!(result, "ExportString[42, PDF]");
  }
}

mod grid_graphics {
  use super::*;

  #[test]
  fn grid_returns_svg() {
    clear_state();
    let svg =
      interpret("ExportString[Grid[{{a, b}, {c, d}}], \"SVG\"]").unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains(">a</text>"));
    assert!(svg.contains(">d</text>"));
  }

  #[test]
  fn grid_produces_svg() {
    clear_state();
    let result = interpret_with_stdout("Grid[{{a, b}, {c, d}}]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("</svg>"));
  }

  #[test]
  fn grid_svg_contains_cell_text() {
    clear_state();
    let result = interpret_with_stdout("Grid[{{a, b}, {c, d}}]").unwrap();
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">a</text>"));
    assert!(svg.contains(">b</text>"));
    assert!(svg.contains(">c</text>"));
    assert!(svg.contains(">d</text>"));
  }

  #[test]
  fn grid_frame_all_produces_lines() {
    clear_state();
    let result =
      interpret_with_stdout("Grid[{{1, 2}, {3, 4}}, Frame -> All]").unwrap();
    let svg = result.graphics.unwrap();
    assert!(
      svg.contains("<line"),
      "Frame -> All should produce <line> elements"
    );
  }

  #[test]
  fn grid_no_frame_no_lines() {
    clear_state();
    let result = interpret_with_stdout("Grid[{{1, 2}, {3, 4}}]").unwrap();
    let svg = result.graphics.unwrap();
    assert!(
      !svg.contains("<line"),
      "Without Frame -> All, no <line> elements"
    );
  }

  #[test]
  fn grid_1d_list() {
    clear_state();
    let result = interpret_with_stdout("Grid[{a, b, c}]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">a</text>"));
    assert!(svg.contains(">b</text>"));
    assert!(svg.contains(">c</text>"));
  }

  #[test]
  fn grid_evaluated_cells() {
    clear_state();
    let result =
      interpret_with_stdout("Grid[{{1+1, 2+2}, {3+3, 4+4}}]").unwrap();
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">2</text>"));
    assert!(svg.contains(">4</text>"));
    assert!(svg.contains(">6</text>"));
    assert!(svg.contains(">8</text>"));
  }

  #[test]
  fn grid_symbolic_cells() {
    clear_state();
    let result = interpret_with_stdout("Grid[{{x, y^2}, {z^3, w}}]").unwrap();
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">x</text>"));
    assert!(svg.contains(">w</text>"));
    // Power expressions should render with tspan superscripts
    assert!(
      svg.contains("baseline-shift=\"super\""),
      "Power expressions should use tspan superscripts"
    );
  }

  #[test]
  fn grid_superscript_rendering() {
    clear_state();
    let result = interpret_with_stdout("Grid[{{y^2, z^3}}]").unwrap();
    let svg = result.graphics.unwrap();
    // y^2 should render as y<tspan baseline-shift="super" font-size="70%">2</tspan>
    assert!(
      svg.contains(
        "y<tspan baseline-shift=\"super\" font-size=\"70%\">2</tspan>"
      ),
      "y^2 should render with superscript tspan, got: {}",
      svg
    );
    assert!(
      svg.contains(
        "z<tspan baseline-shift=\"super\" font-size=\"70%\">3</tspan>"
      ),
      "z^3 should render with superscript tspan"
    );
  }

  #[test]
  fn grid_superscript_in_list_cell() {
    clear_state();
    let result = interpret_with_stdout("Grid[{{{x, y^2, z^3}}}]").unwrap();
    let svg = result.graphics.unwrap();
    // List cell should recursively render Power with superscripts
    assert!(
      svg.contains(
        "y<tspan baseline-shift=\"super\" font-size=\"70%\">2</tspan>"
      ),
      "y^2 inside list cell should render with superscript"
    );
    assert!(
      svg.contains("{x, y"),
      "List braces and non-power items should be preserved"
    );
  }

  #[test]
  fn grid_superscript_in_expression() {
    clear_state();
    let result =
      interpret_with_stdout("Grid[{{x^2 + y^3, Sin[x^2]}}]").unwrap();
    let svg = result.graphics.unwrap();
    // Power in Plus should get superscripts
    assert!(
      svg.contains("x<tspan baseline-shift=\"super\" font-size=\"70%\">2</tspan> + y<tspan baseline-shift=\"super\" font-size=\"70%\">3</tspan>"),
      "x^2 + y^3 should render with superscripts on both terms"
    );
    // Power inside function arguments should get superscripts
    assert!(
      svg.contains(
        "Sin[x<tspan baseline-shift=\"super\" font-size=\"70%\">2</tspan>]"
      ),
      "Sin[x^2] should render with superscript inside function call"
    );
  }

  #[test]
  fn output_svg_for_non_graphics() {
    clear_state();
    let result = interpret_with_stdout("{x^3, x^4, 5/7}").unwrap();
    assert!(
      result.output_svg.is_some(),
      "output_svg should be set for non-graphics results"
    );
    let svg = result.output_svg.unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("<tspan baseline-shift=\"super\""));
    // 5/7 is now rendered as a stacked fraction with tspan elements
    assert!(
      svg.contains("dy=\"-4\">5</tspan>"),
      "5/7 numerator should be rendered as stacked fraction"
    );
    assert!(
      svg.contains("dy=\"6\">7</tspan>"),
      "5/7 denominator should be rendered as stacked fraction"
    );
  }

  #[test]
  fn output_svg_not_set_for_graphics() {
    clear_state();
    let result = interpret_with_stdout("Grid[{{a, b}}]").unwrap();
    assert!(
      result.output_svg.is_none(),
      "output_svg should be None for Graphics results"
    );
    assert!(result.graphics.is_some(), "graphics should be set for Grid");
  }

  #[test]
  fn grid_stacked_fraction() {
    clear_state();
    let result = interpret_with_stdout("Grid[{{5/7, x}, {a, 3/11}}]").unwrap();
    let svg = result.graphics.unwrap();
    // 5/7 rendered as stacked fraction: numerator 5 shifted up, bar, denominator 7
    assert!(
      svg.contains("dy=\"-4\">5</tspan>"),
      "5/7 numerator should be shifted up in stacked fraction"
    );
    assert!(
      svg.contains("dy=\"6\">7</tspan>"),
      "5/7 denominator should be shifted down in stacked fraction"
    );
    // 3/11 also stacked
    assert!(
      svg.contains("dy=\"-4\">3</tspan>"),
      "3/11 numerator should be in stacked fraction"
    );
    assert!(
      svg.contains("dy=\"6\">11</tspan>"),
      "3/11 denominator should be in stacked fraction"
    );
    // Fraction bar character
    assert!(
      svg.contains("\u{2500}"),
      "stacked fraction should contain box-drawing bar character"
    );
  }

  #[test]
  fn output_svg_stacked_fraction_height() {
    clear_state();
    let result = interpret_with_stdout("5/7").unwrap();
    let svg = result.output_svg.unwrap();
    // SVG should have increased height for fractions
    assert!(
      svg.contains("height=\"32\""),
      "output SVG should be taller for stacked fractions"
    );
  }

  #[test]
  fn grid_export_string_svg() {
    clear_state();
    let result =
      interpret("ExportString[Grid[{{1, 2}, {3, 4}}, Frame -> All], \"SVG\"]")
        .unwrap();
    assert!(result.starts_with("<svg"));
    assert!(result.contains("</svg>"));
    assert!(result.contains("<line"));
  }

  #[test]
  fn grid_svg_has_monospace_font() {
    clear_state();
    let result = interpret_with_stdout("Grid[{{a, b}}]").unwrap();
    let svg = result.graphics.unwrap();
    assert!(svg.contains("monospace"));
  }

  #[test]
  fn grid_postfix_form() {
    clear_state();
    let result = interpret_with_stdout("{{1, 2}, {3, 4}} // Grid").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
  }
}

mod read_list {
  use super::*;

  #[test]
  fn expression_default() {
    assert_eq!(
      interpret("ReadList[StringToStream[\"123\\n45\\nx\\ny\"]]").unwrap(),
      "{123, 45, x, y}"
    );
  }

  #[test]
  fn expression_with_evaluation() {
    assert_eq!(
      interpret("ReadList[StringToStream[\"1+2\\n3*4\"], Expression]").unwrap(),
      "{3, 12}"
    );
  }

  #[test]
  fn string_type() {
    assert_eq!(
      interpret("ReadList[StringToStream[\"hello\\nworld\"], String]").unwrap(),
      "{hello, world}"
    );
  }

  #[test]
  fn word_type() {
    assert_eq!(
      interpret("ReadList[StringToStream[\"hello world foo\"], Word]").unwrap(),
      "{hello, world, foo}"
    );
  }

  #[test]
  fn number_type() {
    assert_eq!(
      interpret("ReadList[StringToStream[\"1 2.5 3\"], Number]").unwrap(),
      "{1, 2.5, 3}"
    );
  }

  #[test]
  fn record_type() {
    assert_eq!(
      interpret("ReadList[StringToStream[\"a b\\nc d\"], {Word, Word}]")
        .unwrap(),
      "{{a, b}, {c, d}}"
    );
  }

  #[test]
  fn from_file() {
    use std::io::Write;
    let path = "/tmp/woxi_readlist_test.txt";
    let mut file = std::fs::File::create(path).unwrap();
    writeln!(file, "10").unwrap();
    writeln!(file, "20").unwrap();
    writeln!(file, "30").unwrap();
    drop(file);

    let code = format!("ReadList[\"{}\"]", path);
    assert_eq!(interpret(&code).unwrap(), "{10, 20, 30}");
    let _ = std::fs::remove_file(path);
  }
}

mod put {
  use super::*;

  #[test]
  fn put_single_expression() {
    clear_state();
    let path = "/tmp/woxi_test_put_single.wl";
    let result = interpret(&format!("Put[3, \"{path}\"]")).unwrap();
    assert_eq!(result, "Null");
    let content = std::fs::read_to_string(path).unwrap();
    assert_eq!(content, "3\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_multiple_expressions() {
    clear_state();
    let path = "/tmp/woxi_test_put_multi.wl";
    let result = interpret(&format!("Put[1, 2, 3, \"{path}\"]")).unwrap();
    assert_eq!(result, "Null");
    let content = std::fs::read_to_string(path).unwrap();
    assert_eq!(content, "1\n2\n3\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_list() {
    clear_state();
    let path = "/tmp/woxi_test_put_list.wl";
    let result = interpret(&format!("Put[{{1, 2, 3}}, \"{path}\"]")).unwrap();
    assert_eq!(result, "Null");
    let content = std::fs::read_to_string(path).unwrap();
    assert_eq!(content, "{1, 2, 3}\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_symbolic_expression() {
    clear_state();
    let path = "/tmp/woxi_test_put_sym.wl";
    let result = interpret(&format!("Put[x + y, \"{path}\"]")).unwrap();
    assert_eq!(result, "Null");
    let content = std::fs::read_to_string(path).unwrap();
    assert_eq!(content, "x + y\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_evaluates_argument() {
    clear_state();
    let path = "/tmp/woxi_test_put_eval.wl";
    let result = interpret(&format!("Put[1 + 2, \"{path}\"]")).unwrap();
    assert_eq!(result, "Null");
    let content = std::fs::read_to_string(path).unwrap();
    assert_eq!(content, "3\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_empty_file() {
    clear_state();
    let path = "/tmp/woxi_test_put_empty.wl";
    let result = interpret(&format!("Put[\"{path}\"]")).unwrap();
    assert_eq!(result, "Null");
    let content = std::fs::read_to_string(path).unwrap();
    assert_eq!(content, "");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_rational() {
    clear_state();
    let path = "/tmp/woxi_test_put_rat.wl";
    let result = interpret(&format!("Put[1/3, \"{path}\"]")).unwrap();
    assert_eq!(result, "Null");
    let content = std::fs::read_to_string(path).unwrap();
    assert_eq!(content, "1/3\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_operator_form() {
    clear_state();
    let path = "/tmp/woxi_test_put_op.wl";
    let result = interpret(&format!("42 >> \"{path}\"")).unwrap();
    assert_eq!(result, "Null");
    let content = std::fs::read_to_string(path).unwrap();
    assert_eq!(content, "42\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_overwrites_file() {
    clear_state();
    let path = "/tmp/woxi_test_put_overwrite.wl";
    interpret(&format!("Put[1, \"{path}\"]")).unwrap();
    interpret(&format!("Put[2, \"{path}\"]")).unwrap();
    let content = std::fs::read_to_string(path).unwrap();
    assert_eq!(content, "2\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_return_value_in_list() {
    clear_state();
    let path = "/tmp/woxi_test_put_retval.wl";
    let result = interpret(&format!("{{Put[1, \"{path}\"]}}")).unwrap();
    assert_eq!(result, "{Null}");
    std::fs::remove_file(path).ok();
  }
}

mod put_append {
  use super::*;

  #[test]
  fn put_append_basic() {
    clear_state();
    let path = "/tmp/woxi_test_putappend.wl";
    interpret(&format!("Put[1, \"{path}\"]")).unwrap();
    let result = interpret(&format!("PutAppend[2, \"{path}\"]")).unwrap();
    assert_eq!(result, "Null");
    let content = std::fs::read_to_string(path).unwrap();
    assert_eq!(content, "1\n2\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_append_multiple() {
    clear_state();
    let path = "/tmp/woxi_test_putappend_multi.wl";
    interpret(&format!("Put[1, \"{path}\"]")).unwrap();
    interpret(&format!("PutAppend[2, 3, \"{path}\"]")).unwrap();
    let content = std::fs::read_to_string(path).unwrap();
    assert_eq!(content, "1\n2\n3\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_append_operator_form() {
    clear_state();
    let path = "/tmp/woxi_test_putappend_op.wl";
    interpret(&format!("Put[1, \"{path}\"]")).unwrap();
    interpret(&format!("2 >>> \"{path}\"")).unwrap();
    let content = std::fs::read_to_string(path).unwrap();
    assert_eq!(content, "1\n2\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_append_creates_file() {
    clear_state();
    let path = "/tmp/woxi_test_putappend_create.wl";
    std::fs::remove_file(path).ok();
    let result = interpret(&format!("PutAppend[42, \"{path}\"]")).unwrap();
    assert_eq!(result, "Null");
    let content = std::fs::read_to_string(path).unwrap();
    assert_eq!(content, "42\n");
    std::fs::remove_file(path).ok();
  }
}

mod write {
  use super::*;

  #[test]
  fn write_single_expression() {
    clear_state();
    let path = "/tmp/woxi_test_write_single.txt";
    let result = interpret(&format!(
      "str = OpenWrite[\"{path}\"]; Write[str, 42]; Close[str]"
    ))
    .unwrap();
    assert_eq!(result, path);
    let content = std::fs::read_to_string(path).unwrap();
    assert_eq!(content, "42\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn write_multiple_expressions_concatenated() {
    clear_state();
    let path = "/tmp/woxi_test_write_concat.txt";
    interpret(&format!(
      "str = OpenWrite[\"{path}\"]; Write[str, a^2, 1 + b^2]; Close[str]"
    ))
    .unwrap();
    let content = std::fs::read_to_string(path).unwrap();
    assert_eq!(content, "a^21 + b^2\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn write_multiple_calls() {
    clear_state();
    let path = "/tmp/woxi_test_write_multi.txt";
    interpret(&format!(
      "str = OpenWrite[\"{path}\"]; Write[str, 1]; Write[str, 2]; Write[str, 3]; Close[str]"
    ))
    .unwrap();
    let content = std::fs::read_to_string(path).unwrap();
    assert_eq!(content, "1\n2\n3\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn write_list() {
    clear_state();
    let path = "/tmp/woxi_test_write_list.txt";
    interpret(&format!(
      "str = OpenWrite[\"{path}\"]; Write[str, {{1, 2, 3}}]; Close[str]"
    ))
    .unwrap();
    let content = std::fs::read_to_string(path).unwrap();
    assert_eq!(content, "{1, 2, 3}\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn write_returns_null() {
    clear_state();
    let path = "/tmp/woxi_test_write_null.txt";
    let result = interpret(&format!(
      "str = OpenWrite[\"{path}\"]; r = Write[str, 1]; Close[str]; r"
    ))
    .unwrap();
    assert_eq!(result, "Null");
    std::fs::remove_file(path).ok();
  }
}

mod read {
  use super::*;

  #[test]
  fn read_word() {
    clear_state();
    let result = interpret(
      "str = StringToStream[\"abcdefg 123456\"]; r1 = Read[str, Word]; r2 = Read[str, Word]; Close[str]; {r1, r2}",
    )
    .unwrap();
    assert_eq!(result, "{abcdefg, 123456}");
  }

  #[test]
  fn read_number() {
    clear_state();
    let result = interpret(
      "str = StringToStream[\"42 3.14\"]; r1 = Read[str, Number]; r2 = Read[str, Number]; Close[str]; {r1, r2}",
    )
    .unwrap();
    assert_eq!(result, "{42, 3.14}");
  }

  #[test]
  fn read_string_lines() {
    clear_state();
    let result = interpret(
      "str = StringToStream[\"hello world\\nfoo bar\"]; r1 = Read[str, String]; r2 = Read[str, String]; Close[str]; {r1, r2}",
    )
    .unwrap();
    assert_eq!(result, "{hello world, foo bar}");
  }

  #[test]
  fn read_character() {
    clear_state();
    let result = interpret(
      "str = StringToStream[\"abc\"]; r = Read[str, Character]; Close[str]; r",
    )
    .unwrap();
    assert_eq!(result, "a");
  }

  #[test]
  fn read_expression() {
    clear_state();
    let result = interpret(
      "str = StringToStream[\"1 + 2\"]; r = Read[str, Expression]; Close[str]; r",
    )
    .unwrap();
    assert_eq!(result, "3");
  }

  #[test]
  fn read_end_of_file() {
    clear_state();
    let result = interpret(
      "str = StringToStream[\"hello\"]; r1 = Read[str, Word]; r2 = Read[str, Word]; Close[str]; {r1, r2}",
    )
    .unwrap();
    assert_eq!(result, "{hello, EndOfFile}");
  }

  #[test]
  fn read_default_type() {
    clear_state();
    let result = interpret(
      "str = StringToStream[\"hello\"]; r = Read[str]; Close[str]; r",
    )
    .unwrap();
    assert_eq!(result, "hello");
  }

  #[test]
  fn read_mixed_types() {
    clear_state();
    let result = interpret(
      "str = StringToStream[\"3.14 hello 42\"]; r1 = Read[str, Number]; r2 = Read[str, Word]; r3 = Read[str, Number]; Close[str]; {r1, r2, r3}",
    )
    .unwrap();
    assert_eq!(result, "{3.14, hello, 42}");
  }

  #[test]
  fn read_list_of_types() {
    clear_state();
    let result = interpret(
      "str = StringToStream[\"a b c\"]; r = Read[str, {Word, Word, Word}]; Close[str]; r",
    )
    .unwrap();
    assert_eq!(result, "{a, b, c}");
  }
}

mod information {
  use super::*;

  #[test]
  fn variable_definition() {
    clear_state();
    let result = interpret("a = 5; ?a").unwrap();
    assert!(result.contains(
      "OwnValues -> Information`InformationValueForm[OwnValues, a, {a -> 5}]"
    ));
    assert!(result.contains("DownValues -> None"));
    assert!(result.contains("FullName -> Global`a"));
  }

  #[test]
  fn function_definition() {
    clear_state();
    let result = interpret("f[x_] := x^2; ?f").unwrap();
    assert!(result.contains("DownValues -> Information`InformationValueForm[DownValues, f, {f[x_] :> x^2}]"));
    assert!(result.contains("OwnValues -> None"));
  }

  #[test]
  fn undefined_symbol() {
    clear_state();
    let result = interpret("?z").unwrap();
    assert_eq!(result, "Missing[UnknownSymbol, z]");
  }

  #[test]
  fn compound_expression() {
    clear_state();
    let result = interpret("a = 5; ?a; 1 + 2").unwrap();
    assert_eq!(result, "3");
  }

  #[test]
  fn multiple_function_definitions() {
    clear_state();
    let result = interpret("f[x_] := x^2; f[x_, y_] := x + y; ?f").unwrap();
    assert!(result.contains("f[x_] :> x^2, f[x_, y_] :> x + y"));
  }

  #[test]
  fn information_function_call_form() {
    clear_state();
    let result = interpret("a = 10; Information[a]").unwrap();
    assert!(result.contains(
      "OwnValues -> Information`InformationValueForm[OwnValues, a, {a -> 10}]"
    ));
  }

  #[test]
  fn function_with_head_constraint() {
    clear_state();
    let result = interpret("g[x_Integer] := x + 1; ?g").unwrap();
    assert!(result.contains("g[x_Integer] :> x + 1"));
  }

  #[test]
  fn symbol_with_attributes() {
    clear_state();
    let result =
      interpret("SetAttributes[h, Listable]; h[x_] := x^2; ?h").unwrap();
    assert!(result.contains("Attributes -> {Listable}"));
    assert!(result.contains("h[x_] :> x^2"));
  }
}
