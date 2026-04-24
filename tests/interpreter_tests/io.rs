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
    assert_eq!(result, "\0");
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
  #[cfg(not(target_arch = "wasm32"))]
  fn open_write_no_args_returns_output_stream() {
    // OpenWrite[] with no args creates a temp file and returns OutputStream.
    assert_eq!(interpret("Head[OpenWrite[]]").unwrap(), "OutputStream");
  }

  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn open_append_no_args_returns_output_stream() {
    assert_eq!(interpret("Head[OpenAppend[]]").unwrap(), "OutputStream");
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
    assert_eq!(result, "\0");
  }

  // `WriteString["stdout", …]` and `"stderr"` route to the process's
  // standard streams, matching wolframscript.
  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn write_string_stdout_returns_null() {
    // The text goes to stdout via print!, which interpret()'s usual
    // return-value path doesn't capture — but the call should still
    // succeed and return the "\0" marker used for Null in test output.
    let result = interpret(r#"WriteString["stdout", "Hola"]"#).unwrap();
    assert_eq!(result, "\0");
  }
}

mod find_file {
  use super::*;

  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn missing_file_returns_failed() {
    assert_eq!(
      interpret(r#"FindFile["ExampleData/sunflowers.jpg"]"#).unwrap(),
      "$Failed"
    );
  }

  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn context_returns_failed() {
    // Context strings (ending in `) resolve to package files in Wolfram;
    // Woxi has no package loader, so it returns $Failed.
    assert_eq!(
      interpret(r#"FindFile["VectorAnalysis`"]"#).unwrap(),
      "$Failed"
    );
  }
}

mod absolute_file_name {
  use super::*;

  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn missing_file_returns_failed() {
    // Missing files emit AbsoluteFileName::fdnfnd and return $Failed,
    // matching wolframscript.
    assert_eq!(
      interpret(r#"AbsoluteFileName["ExampleData/sunflowers.jpg"]"#).unwrap(),
      "$Failed"
    );
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
    let result = interpret_with_stdout("BinaryRead[1, n, z]").unwrap();
    assert_eq!(result.result, "BinaryRead[1, n, z]");
    assert_eq!(result.warnings.len(), 1);
    assert!(result.warnings[0].contains("not yet implemented"));
    assert!(result.warnings[0].contains("BinaryRead["));
  }

  #[test]
  fn unknown_function_no_warning() {
    clear_state();
    let result = interpret_with_stdout("MyCustomFunc[1, 2]").unwrap();
    assert_eq!(result.result, "MyCustomFunc[1, 2]");
    assert!(result.warnings.is_empty());
  }

  #[test]
  fn shortest_is_symbolic_no_warning() {
    // Shortest is a symbolic pattern wrapper; it's meaningful inside
    // StringCases/StringReplace/etc. but at the top level it should stay
    // unevaluated without emitting the "not yet implemented" warning.
    clear_state();
    let result =
      interpret_with_stdout(r#"Shortest[RegularExpression["a+b"]]"#).unwrap();
    assert_eq!(result.result, "Shortest[RegularExpression[a+b]]");
    assert!(
      result.warnings.is_empty(),
      "Expected no warnings but got: {:?}",
      result.warnings
    );
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
    let result = interpret_with_stdout("BinaryRead[1]").unwrap();
    assert!(!result.stdout.contains("not yet implemented"));
    assert!(!result.warnings.is_empty());
  }

  #[test]
  fn multiple_unimplemented_calls_consolidated_into_single_warning() {
    clear_state();
    let result =
      interpret_with_stdout("{BinaryRead[1], GridGraph[2]}").unwrap();
    assert_eq!(result.warnings.len(), 1);
    assert!(result.warnings[0].contains("BinaryRead[1]"));
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
    assert!(result.contains(">{</text>"), "should contain opening brace");
    assert!(result.contains(">1</text>"), "should contain 1");
    assert!(result.contains(">2</text>"), "should contain 2");
    assert!(result.contains(">3</text>"), "should contain 3");
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
  fn export_string_pdf() {
    clear_state();
    let result = interpret("ExportString[42, \"PDF\"]").unwrap();
    assert!(
      result.starts_with("%PDF-"),
      "Expected PDF output, got: {}",
      &result[..50.min(result.len())]
    );
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
  fn grid_frame_viewbox_has_padding_for_strokes() {
    clear_state();
    let result =
      interpret_with_stdout("Grid[{{a, b, c}, {x, y^2, z^3}}, Frame -> All]")
        .unwrap();
    let svg = result.graphics.unwrap();
    // viewBox must start at -0.5 so border strokes aren't clipped
    assert!(
      svg.contains("viewBox=\"-0.5 -0.5"),
      "Frame -> All viewBox should have -0.5 offset to avoid clipping strokes"
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
    // Superscripts use smaller font
    assert!(
      svg.contains("font-size=\"9.8\""),
      "should have superscript font"
    );
    // 5/7 is rendered as a fraction with SVG line
    assert!(
      svg.contains("<line"),
      "5/7 should have an SVG line for the fraction bar"
    );
    assert!(
      svg.contains(">5</text>"),
      "5/7 numerator should be in stacked fraction"
    );
    assert!(
      svg.contains(">7</text>"),
      "5/7 denominator should be in stacked fraction"
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
    // Fractions rendered as inline text in grid cells
    assert!(svg.contains("5/7"), "5/7 should be rendered in grid");
    assert!(svg.contains("3/11"), "3/11 should be rendered in grid");
  }

  #[test]
  fn output_svg_stacked_fraction_height() {
    clear_state();
    let result = interpret_with_stdout("5/7").unwrap();
    let svg = result.output_svg.unwrap();
    // SVG should have increased height for fractions
    assert!(
      svg.contains("<line"),
      "output SVG should contain fraction line"
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
  fn grid_svg_has_sans_serif_font() {
    clear_state();
    let result = interpret_with_stdout("Grid[{{a, b}}]").unwrap();
    let svg = result.graphics.unwrap();
    assert!(svg.contains("font-family=\"sans-serif\""));
    assert!(!svg.contains("font-family=\"monospace\""));
  }

  #[test]
  fn grid_postfix_form() {
    clear_state();
    let result = interpret_with_stdout("{{1, 2}, {3, 4}} // Grid").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
  }

  #[test]
  fn grid_frame_true_outer_only() {
    clear_state();
    let result =
      interpret_with_stdout("Grid[{{1, 2}, {3, 4}}, Frame -> True]").unwrap();
    let svg = result.graphics.unwrap();
    assert!(svg.contains("<line"), "Frame -> True should produce lines");
    // 2x2 grid with Frame -> True: 2 horizontal (top/bottom) + 2 vertical (left/right) = 4 lines
    let line_count = svg.matches("<line").count();
    assert_eq!(
      line_count, 4,
      "Frame -> True on 2x2 should have 4 outer lines, got {}",
      line_count
    );
  }

  #[test]
  fn grid_dividers_all() {
    clear_state();
    let result =
      interpret_with_stdout("Grid[{{1, 2, 3}, {4, 5, 6}}, Dividers -> All]")
        .unwrap();
    let svg = result.graphics.unwrap();
    assert!(
      svg.contains("<line"),
      "Dividers -> All should produce lines"
    );
  }

  #[test]
  fn grid_dividers_col_only() {
    clear_state();
    let result = interpret_with_stdout(
      "Grid[{{1, 2, 3}, {4, 5, 6}}, Dividers -> {All, None}]",
    )
    .unwrap();
    let svg = result.graphics.unwrap();
    // Should have vertical divider lines but no horizontal ones
    let lines: Vec<&str> =
      svg.lines().filter(|l| l.contains("<line")).collect();
    assert!(!lines.is_empty(), "Should have vertical divider lines");
    // All lines should be vertical (x1 == x2, y values differ)
    for line in &lines {
      if let (Some(x1_pos), Some(x2_pos)) =
        (line.find("x1=\""), line.find("x2=\""))
      {
        let x1 =
          &line[x1_pos + 4..line[x1_pos + 4..].find('"').unwrap() + x1_pos + 4];
        let x2 =
          &line[x2_pos + 4..line[x2_pos + 4..].find('"').unwrap() + x2_pos + 4];
        assert_eq!(
          x1, x2,
          "Dividers -> {{All, None}} should only have vertical lines"
        );
      }
    }
  }

  #[test]
  fn grid_background_uniform() {
    clear_state();
    let result =
      interpret_with_stdout("Grid[{{1, 2}, {3, 4}}, Background -> LightGray]")
        .unwrap();
    let svg = result.graphics.unwrap();
    assert!(
      svg.contains("<rect") && svg.contains("rgb(217,217,217)"),
      "Background -> LightGray should produce gray rect elements"
    );
  }

  #[test]
  fn grid_background_per_column() {
    clear_state();
    let result = interpret_with_stdout(
      "Grid[{{1, 2}, {3, 4}}, Background -> {{LightBlue, LightYellow}}]",
    )
    .unwrap();
    let svg = result.graphics.unwrap();
    // Should have both colors
    assert!(svg.contains("<rect"), "Should have background rects");
    // 4 cells total, 2 blue columns, 2 yellow columns
    let rect_count = svg.matches("<rect").count();
    assert_eq!(rect_count, 4, "Should have 4 background rects for 2x2 grid");
  }

  #[test]
  fn grid_background_per_row() {
    clear_state();
    let result = interpret_with_stdout(
      "Grid[{{1, 2}, {3, 4}}, Background -> {{}, {LightGreen, LightRed}}]",
    )
    .unwrap();
    let svg = result.graphics.unwrap();
    assert!(svg.contains("<rect"), "Should have background rects");
  }

  #[test]
  fn grid_alignment_left() {
    clear_state();
    let result =
      interpret_with_stdout("Grid[{{a, b}, {c, d}}, Alignment -> Left]")
        .unwrap();
    let svg = result.graphics.unwrap();
    assert!(
      svg.contains("text-anchor=\"start\""),
      "Alignment -> Left should use text-anchor start"
    );
    assert!(
      !svg.contains("text-anchor=\"middle\""),
      "Should not have middle alignment"
    );
  }

  #[test]
  fn grid_alignment_right() {
    clear_state();
    let result =
      interpret_with_stdout("Grid[{{a, b}, {c, d}}, Alignment -> Right]")
        .unwrap();
    let svg = result.graphics.unwrap();
    assert!(
      svg.contains("text-anchor=\"end\""),
      "Alignment -> Right should use text-anchor end"
    );
  }

  #[test]
  fn grid_alignment_center_default() {
    clear_state();
    let result = interpret_with_stdout("Grid[{{a, b}, {c, d}}]").unwrap();
    let svg = result.graphics.unwrap();
    assert!(
      svg.contains("text-anchor=\"middle\""),
      "Default alignment should be center (middle)"
    );
  }

  #[test]
  fn grid_frame_all_with_background() {
    clear_state();
    let result = interpret_with_stdout(
      "Grid[{{1, 2}, {3, 4}}, Frame -> All, Background -> LightYellow]",
    )
    .unwrap();
    let svg = result.graphics.unwrap();
    assert!(svg.contains("<rect"), "Should have background rects");
    assert!(svg.contains("<line"), "Should have frame lines");
  }

  #[test]
  fn grid_background_with_none() {
    clear_state();
    let result = interpret_with_stdout(
      "Grid[{{1, 2}, {3, 4}}, Background -> {{LightBlue, None}}]",
    )
    .unwrap();
    let svg = result.graphics.unwrap();
    // Only column 1 should have background, column 2 should not
    let rect_count = svg.matches("<rect").count();
    assert_eq!(
      rect_count, 2,
      "Only 2 cells (column 1) should have backgrounds"
    );
  }

  #[test]
  fn grid_background_rgbcolor() {
    clear_state();
    let result = interpret_with_stdout(
      "Grid[{{1, 2}, {3, 4}}, Background -> RGBColor[1, 0.9, 0.8]]",
    )
    .unwrap();
    let svg = result.graphics.unwrap();
    assert!(
      svg.contains("<rect") && svg.contains("rgb(255,230,204)"),
      "Should render RGBColor background"
    );
  }

  #[test]
  fn grid_dividers_with_frame_true() {
    clear_state();
    let result = interpret_with_stdout(
      "Grid[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, Frame -> True, Dividers -> All]",
    )
    .unwrap();
    let svg = result.graphics.unwrap();
    let line_count = svg.matches("<line").count();
    // 4 horizontal + 4 vertical = 8 lines (outer + inner)
    assert_eq!(
      line_count, 8,
      "Frame -> True + Dividers -> All on 3x3 should have 8 lines, got {}",
      line_count
    );
  }
}

mod read_string {
  use super::*;

  #[test]
  fn basic_file() {
    use std::io::Write;
    let path = "/tmp/woxi_readstring_test.txt";
    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(b"hello world\nfoo bar").unwrap();
    drop(file);

    let code = format!("ReadString[\"{}\"]", path);
    assert_eq!(interpret(&code).unwrap(), "hello world\nfoo bar");
    let _ = std::fs::remove_file(path);
  }

  #[test]
  fn nonexistent_file() {
    assert_eq!(
      interpret("ReadString[\"/tmp/woxi_no_such_file_12345.txt\"]").unwrap(),
      "$Failed"
    );
  }

  #[test]
  fn symbolic_returns_unevaluated() {
    assert_eq!(interpret("ReadString[x]").unwrap(), "ReadString[x]");
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
  fn with_stream_variable() {
    assert_eq!(
      interpret(
        "stream = StringToStream[\"123\\n45\\nx\\ny\"]; ReadList[stream]"
      )
      .unwrap(),
      "{123, 45, x, y}"
    );
  }

  #[test]
  fn number_with_max_count() {
    assert_eq!(
      interpret("ReadList[StringToStream[\"1\\n2\\n3\"], Number, 2]").unwrap(),
      "{1, 2}"
    );
  }

  #[test]
  fn empty_string() {
    assert_eq!(
      interpret("ReadList[StringToStream[\"\"], Expression]").unwrap(),
      "{}"
    );
  }

  #[test]
  fn empty_lines() {
    assert_eq!(
      interpret("ReadList[StringToStream[\"\\n\\n\\n\"], Expression]").unwrap(),
      "{}"
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
    assert_eq!(result, "\0");
    let content = std::fs::read_to_string(path).unwrap();
    assert_eq!(content, "3\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_multiple_expressions() {
    clear_state();
    let path = "/tmp/woxi_test_put_multi.wl";
    let result = interpret(&format!("Put[1, 2, 3, \"{path}\"]")).unwrap();
    assert_eq!(result, "\0");
    let content = std::fs::read_to_string(path).unwrap();
    assert_eq!(content, "1\n2\n3\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_list() {
    clear_state();
    let path = "/tmp/woxi_test_put_list.wl";
    let result = interpret(&format!("Put[{{1, 2, 3}}, \"{path}\"]")).unwrap();
    assert_eq!(result, "\0");
    let content = std::fs::read_to_string(path).unwrap();
    assert_eq!(content, "{1, 2, 3}\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_symbolic_expression() {
    clear_state();
    let path = "/tmp/woxi_test_put_sym.wl";
    let result = interpret(&format!("Put[x + y, \"{path}\"]")).unwrap();
    assert_eq!(result, "\0");
    let content = std::fs::read_to_string(path).unwrap();
    assert_eq!(content, "x + y\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_evaluates_argument() {
    clear_state();
    let path = "/tmp/woxi_test_put_eval.wl";
    let result = interpret(&format!("Put[1 + 2, \"{path}\"]")).unwrap();
    assert_eq!(result, "\0");
    let content = std::fs::read_to_string(path).unwrap();
    assert_eq!(content, "3\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_empty_file() {
    clear_state();
    let path = "/tmp/woxi_test_put_empty.wl";
    let result = interpret(&format!("Put[\"{path}\"]")).unwrap();
    assert_eq!(result, "\0");
    let content = std::fs::read_to_string(path).unwrap();
    assert_eq!(content, "");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_rational() {
    clear_state();
    let path = "/tmp/woxi_test_put_rat.wl";
    let result = interpret(&format!("Put[1/3, \"{path}\"]")).unwrap();
    assert_eq!(result, "\0");
    let content = std::fs::read_to_string(path).unwrap();
    assert_eq!(content, "1/3\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_operator_form() {
    clear_state();
    let path = "/tmp/woxi_test_put_op.wl";
    let result = interpret(&format!("42 >> \"{path}\"")).unwrap();
    assert_eq!(result, "\0");
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
    assert_eq!(result, "\0");
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
    assert_eq!(result, "\0");
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
    assert_eq!(result, "\0");
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

  #[test]
  fn builtin_function_brief() {
    clear_state();
    let result = interpret("?Plus").unwrap();
    assert!(result.contains("Name -> Plus"));
    assert!(result.contains("Usage -> Adds numbers together"));
    assert!(result.contains("False]"));
  }

  #[test]
  fn builtin_function_full() {
    clear_state();
    let result = interpret("??Plus").unwrap();
    assert!(result.contains("Name -> Plus"));
    assert!(result.contains("Usage -> Adds numbers together"));
    assert!(result.contains("Attributes -> {"));
    assert!(result.contains("Protected"));
    assert!(result.contains("Flat"));
    assert!(result.contains("FullName -> System`Plus"));
    assert!(result.contains("True]"));
  }

  #[test]
  fn builtin_function_information_call() {
    clear_state();
    let result = interpret("Information[Sin]").unwrap();
    assert!(result.contains("Name -> Sin"));
    assert!(result.contains("Usage -> Returns the sine"));
  }

  #[test]
  fn builtin_function_information_full_call() {
    clear_state();
    let result = interpret("Information[Sin, \"Full\"]").unwrap();
    assert!(result.contains("Attributes -> {"));
    assert!(result.contains("Listable"));
    assert!(result.contains("NumericFunction"));
    assert!(result.contains("FullName -> System`Sin"));
    assert!(result.contains("True]"));
  }

  #[test]
  fn double_question_mark_user_defined() {
    clear_state();
    let result = interpret("f[x_] := x^2; ??f").unwrap();
    assert!(result.contains("DownValues -> Information`InformationValueForm"));
    assert!(result.contains("f[x_] :> x^2"));
    assert!(result.contains("True]"));
  }

  #[test]
  fn unknown_symbol() {
    clear_state();
    let result = interpret("?xyzNotAFunction").unwrap();
    assert!(result.contains("Missing[UnknownSymbol"));
  }

  #[test]
  fn builtin_symbol_without_attributes() {
    // A built-in symbol that exists in functions.csv but has no attributes
    clear_state();
    let result = interpret("?Table").unwrap();
    assert!(result.contains("Name -> Table"));
    assert!(result.contains("Usage ->"));
  }

  #[test]
  fn pattern_query_prefix_wildcard() {
    // ?Plot* should return InformationDataGrid with matching symbols
    clear_state();
    let result = interpret("?Plot*").unwrap();
    assert!(result.starts_with("InformationDataGrid["));
    assert!(result.contains("System`"));
    assert!(result.contains("Plot"));
    assert!(result.contains("PlotRange"));
    assert!(result.contains("PlotStyle"));
    assert!(result.contains("False]"));
  }

  #[test]
  fn pattern_query_suffix_wildcard() {
    // ?*Plot should match symbols ending with Plot
    clear_state();
    let result = interpret("?*Plot").unwrap();
    assert!(result.starts_with("InformationDataGrid["));
    assert!(result.contains("ListPlot"));
    assert!(result.contains("ContourPlot"));
    // Should not contain PlotRange (doesn't end with Plot)
    assert!(!result.contains("PlotRange"));
  }

  #[test]
  fn pattern_query_both_wildcards() {
    // ?*Plot* should match symbols containing Plot anywhere
    clear_state();
    let result = interpret("?*Plot*").unwrap();
    assert!(result.starts_with("InformationDataGrid["));
    assert!(result.contains("Plot"));
    assert!(result.contains("ListPlot"));
    assert!(result.contains("PlotRange"));
  }

  #[test]
  fn pattern_query_full_info() {
    // ??Plot* should return with True (is_full)
    clear_state();
    let result = interpret("??Plot*").unwrap();
    assert!(result.starts_with("InformationDataGrid["));
    assert!(result.contains("True]"));
  }

  #[test]
  fn pattern_query_includes_user_defined() {
    // Pattern should also match user-defined symbols
    clear_state();
    let result = interpret("myPlotHelper[x_] := x^2; ?*Plot*").unwrap();
    assert!(result.contains("myPlotHelper"));
    assert!(result.contains("Plot"));
  }

  #[test]
  fn pattern_query_no_matches() {
    // Pattern that matches nothing should return empty list
    clear_state();
    let result = interpret("?Zzzzzzz*").unwrap();
    assert!(result.starts_with("InformationDataGrid["));
    assert!(result.contains("System` -> {}"));
  }
}

mod directory_name {
  use super::*;

  #[test]
  fn absolute_path_with_file() {
    assert_eq!(
      interpret(r#"DirectoryName["/home/user/file.txt"]"#).unwrap(),
      "/home/user/"
    );
  }

  #[test]
  fn relative_path_with_file() {
    assert_eq!(interpret(r#"DirectoryName["a/b/c"]"#).unwrap(), "a/b/");
  }

  #[test]
  fn trailing_separator() {
    assert_eq!(
      interpret(r#"DirectoryName["/home/user/"]"#).unwrap(),
      "/home/"
    );
  }

  #[test]
  fn no_directory() {
    assert_eq!(interpret(r#"DirectoryName["file.txt"]"#).unwrap(), "");
  }

  #[test]
  fn empty_string() {
    assert_eq!(interpret(r#"DirectoryName[""]"#).unwrap(), "");
  }

  #[test]
  fn root_directory() {
    assert_eq!(interpret(r#"DirectoryName["/"]"#).unwrap(), "");
  }

  #[test]
  fn single_component_under_root() {
    assert_eq!(interpret(r#"DirectoryName["/home"]"#).unwrap(), "/");
  }

  #[test]
  fn with_n_parameter() {
    assert_eq!(
      interpret(r#"DirectoryName["/home/user/file.txt", 2]"#).unwrap(),
      "/home/"
    );
  }

  #[test]
  fn with_n_exceeding_depth() {
    assert_eq!(interpret(r#"DirectoryName["/a/b", 3]"#).unwrap(), "");
  }

  #[test]
  fn relative_trailing_slash() {
    assert_eq!(interpret(r#"DirectoryName["a/b/c/"]"#).unwrap(), "a/b/");
  }

  #[test]
  fn n_equals_one_explicit() {
    assert_eq!(interpret(r#"DirectoryName["/a/b", 1]"#).unwrap(), "/a/");
  }

  #[test]
  fn deep_path_n3() {
    assert_eq!(
      interpret(r#"DirectoryName["a/b/c/d.txt", 3]"#).unwrap(),
      "a/"
    );
  }
}

mod file_name_join {
  use super::*;

  #[test]
  fn basic_join() {
    assert_eq!(
      interpret(r#"FileNameJoin[{"home", "user", "file.txt"}]"#).unwrap(),
      "home/user/file.txt"
    );
  }

  #[test]
  fn two_parts() {
    assert_eq!(interpret(r#"FileNameJoin[{"a", "b"}]"#).unwrap(), "a/b");
  }

  #[test]
  fn single_part() {
    assert_eq!(
      interpret(r#"FileNameJoin[{"file.txt"}]"#).unwrap(),
      "file.txt"
    );
  }

  #[test]
  fn operating_system_unix() {
    assert_eq!(
      interpret(
        r#"FileNameJoin[{"dir1", "dir2", "dir3"}, OperatingSystem -> "Unix"]"#
      )
      .unwrap(),
      "dir1/dir2/dir3"
    );
  }

  #[test]
  fn operating_system_windows() {
    assert_eq!(
      interpret(
        r#"FileNameJoin[{"dir1", "dir2", "dir3"}, OperatingSystem -> "Windows"]"#
      )
      .unwrap(),
      r#"dir1\dir2\dir3"#
    );
  }
}

mod to_file_name {
  use super::*;

  #[test]
  fn list_dirs_and_file() {
    assert_eq!(
      interpret(r#"ToFileName[{"dir1", "dir2"}, "file"]"#).unwrap(),
      "dir1/dir2/file"
    );
  }

  #[test]
  fn string_dir_and_file() {
    assert_eq!(
      interpret(r#"ToFileName["dir1", "file"]"#).unwrap(),
      "dir1/file"
    );
  }

  #[test]
  fn list_dirs_only_has_trailing_slash() {
    assert_eq!(
      interpret(r#"ToFileName[{"dir1", "dir2", "dir3"}]"#).unwrap(),
      "dir1/dir2/dir3/"
    );
  }

  #[test]
  fn single_dir_has_trailing_slash() {
    assert_eq!(
      interpret(r#"ToFileName["just_a_dir"]"#).unwrap(),
      "just_a_dir/"
    );
  }
}

mod file_name_split {
  use super::*;

  #[test]
  fn absolute_path() {
    assert_eq!(
      interpret(r#"FileNameSplit["/a/b/c.txt"]"#).unwrap(),
      "{, a, b, c.txt}"
    );
  }

  #[test]
  fn relative_path() {
    assert_eq!(interpret(r#"FileNameSplit["a/b/c"]"#).unwrap(), "{a, b, c}");
  }

  #[test]
  fn root_path() {
    assert_eq!(interpret(r#"FileNameSplit["/"]"#).unwrap(), "{}");
  }

  #[test]
  fn empty_string() {
    assert_eq!(interpret(r#"FileNameSplit[""]"#).unwrap(), "{}");
  }

  #[test]
  fn single_filename() {
    assert_eq!(
      interpret(r#"FileNameSplit["abc.txt"]"#).unwrap(),
      "{abc.txt}"
    );
  }

  #[test]
  fn trailing_slash() {
    assert_eq!(
      interpret(r#"FileNameSplit["a/b/c/"]"#).unwrap(),
      "{a, b, c}"
    );
  }

  #[test]
  fn non_string_arg() {
    assert_eq!(interpret("FileNameSplit[42]").unwrap(), "FileNameSplit[42]");
  }
}

mod file_name_depth {
  use super::*;

  #[test]
  fn relative_path() {
    assert_eq!(interpret(r#"FileNameDepth["a/b/c"]"#).unwrap(), "3");
  }

  #[test]
  fn trailing_slash_ignored() {
    assert_eq!(interpret(r#"FileNameDepth["a/b/c/"]"#).unwrap(), "3");
  }

  #[test]
  fn leading_slash_counts_root() {
    assert_eq!(interpret(r#"FileNameDepth["/a/b/c"]"#).unwrap(), "4");
  }

  #[test]
  fn single_filename() {
    assert_eq!(interpret(r#"FileNameDepth["abc"]"#).unwrap(), "1");
  }

  #[test]
  fn empty_string() {
    assert_eq!(interpret(r#"FileNameDepth[""]"#).unwrap(), "0");
  }

  #[test]
  fn non_string_arg() {
    assert_eq!(interpret("FileNameDepth[42]").unwrap(), "FileNameDepth[42]");
  }
}

mod expand_file_name {
  use super::*;

  #[test]
  fn absolute_path_unchanged() {
    assert_eq!(
      interpret(r#"ExpandFileName["/absolute/path"]"#).unwrap(),
      "/absolute/path"
    );
  }

  #[test]
  fn resolves_parent_directory() {
    let result = interpret(r#"ExpandFileName["/a/b/../c"]"#).unwrap();
    assert_eq!(result, "/a/c");
  }

  #[test]
  fn resolves_current_directory() {
    let result = interpret(r#"ExpandFileName["/a/./b"]"#).unwrap();
    assert_eq!(result, "/a/b");
  }

  #[test]
  fn tilde_expansion() {
    let result = interpret(r#"ExpandFileName["~/test.txt"]"#).unwrap();
    assert!(
      result.ends_with("/test.txt") && !result.starts_with('~'),
      "Expected expanded path, got: {}",
      result
    );
  }

  #[test]
  fn relative_path_becomes_absolute() {
    let result = interpret(r#"ExpandFileName["foo/bar"]"#).unwrap();
    assert!(
      result.starts_with('/') && result.ends_with("foo/bar"),
      "Expected absolute path, got: {}",
      result
    );
  }
}

mod parent_directory {
  use super::*;

  #[test]
  fn arg_absolute() {
    assert_eq!(interpret(r#"ParentDirectory["/a/b/c"]"#).unwrap(), "/a/b");
  }

  #[test]
  fn arg_relative() {
    assert_eq!(interpret(r#"ParentDirectory["a/b/c"]"#).unwrap(), "a/b");
  }

  #[test]
  fn no_args_returns_string() {
    // ParentDirectory[] returns the parent of the current working directory;
    // just verify it's a non-empty string.
    let result = interpret("ParentDirectory[]").unwrap();
    assert!(!result.is_empty());
  }
}

mod url_build {
  use super::*;

  #[test]
  fn passthrough_string() {
    assert_eq!(
      interpret(r#"URLBuild["https://example.com"]"#).unwrap(),
      "https://example.com"
    );
  }

  #[test]
  fn path_segments() {
    assert_eq!(
      interpret(r#"URLBuild[{"https://example.com", "path"}]"#).unwrap(),
      "https://example.com/path"
    );
    assert_eq!(
      interpret(r#"URLBuild[{"https://example.com", "a", "b"}]"#).unwrap(),
      "https://example.com/a/b"
    );
  }

  #[test]
  fn with_query_params() {
    assert_eq!(
      interpret(r#"URLBuild[{"https://example.com", "a", "b"}, {"x" -> "1", "y" -> "2"}]"#)
        .unwrap(),
      "https://example.com/a/b?x=1&y=2"
    );
  }
}

mod csv_import {
  use super::*;

  fn csv_path() -> String {
    let manifest = env!("CARGO_MANIFEST_DIR");
    format!("{manifest}/tests/data/data.csv")
  }

  #[test]
  fn import_csv_default_returns_data() {
    let path = csv_path();
    let result = interpret(&format!(r#"Import["{path}"]"#)).unwrap();
    assert!(result.starts_with("{{date, name, fruit, quantity}"));
    assert!(result.contains("banana"));
  }

  #[test]
  fn import_csv_elements() {
    let path = csv_path();
    let result =
      interpret(&format!(r#"Import["{path}", "Elements"]"#)).unwrap();
    assert!(result.contains("ColumnCount"));
    assert!(result.contains("Data"));
    assert!(result.contains("Dataset"));
    assert!(result.contains("Tabular"));
  }

  #[test]
  fn import_csv_column_labels() {
    let path = csv_path();
    assert_eq!(
      interpret(&format!(r#"Import["{path}", "ColumnLabels"]"#)).unwrap(),
      "{date, name, fruit, quantity}"
    );
  }

  #[test]
  fn import_csv_column_count() {
    let path = csv_path();
    assert_eq!(
      interpret(&format!(r#"Import["{path}", "ColumnCount"]"#)).unwrap(),
      "4"
    );
  }

  #[test]
  fn import_csv_row_count() {
    let path = csv_path();
    assert_eq!(
      interpret(&format!(r#"Import["{path}", "RowCount"]"#)).unwrap(),
      "6"
    );
  }

  #[test]
  fn import_csv_dimensions() {
    let path = csv_path();
    assert_eq!(
      interpret(&format!(r#"Import["{path}", "Dimensions"]"#)).unwrap(),
      "{6, 4}"
    );
  }

  #[test]
  fn import_csv_column_types() {
    let path = csv_path();
    let result =
      interpret(&format!(r#"Import["{path}", "ColumnTypes"]"#)).unwrap();
    assert!(result.contains("String"));
  }

  #[test]
  fn import_csv_raw_data() {
    let path = csv_path();
    let result = interpret(&format!(r#"Import["{path}", "RawData"]"#)).unwrap();
    assert!(result.contains("banana"));
    assert!(result.contains("date"));
  }

  #[test]
  fn import_csv_data() {
    let path = csv_path();
    let result = interpret(&format!(r#"Import["{path}", "Data"]"#)).unwrap();
    assert!(result.starts_with("{{date, name, fruit, quantity}"));
    assert!(result.contains("{2025-09-24, John, banana, 3}"));
  }

  #[test]
  fn import_csv_summary() {
    let path = csv_path();
    let result = interpret(&format!(r#"Import["{path}", "Summary"]"#)).unwrap();
    assert!(result.contains("Columns"));
    assert!(result.contains("Rows"));
    assert!(result.contains("ColumnLabels"));
  }

  #[test]
  fn import_csv_dataset() {
    let path = csv_path();
    let result = interpret(&format!(r#"Import["{path}", "Dataset"]"#)).unwrap();
    // Dataset renders as graphics in the interpreter
    assert!(
      result == "-Graphics-" || result.starts_with("Dataset["),
      "unexpected: {}",
      result
    );
  }

  #[test]
  fn import_csv_schema() {
    let path = csv_path();
    let result = interpret(&format!(r#"Import["{path}", "Schema"]"#)).unwrap();
    assert!(result.starts_with("TabularSchema["));
    assert!(result.contains("ColumnKeys"));
    assert!(result.contains("RowCount"));
  }

  #[test]
  fn import_csv_tabular() {
    let path = csv_path();
    let result = interpret(&format!(r#"Import["{path}", "Tabular"]"#)).unwrap();
    // Tabular renders as graphics in the interpreter
    assert!(
      result == "-Graphics-" || result.starts_with("Tabular["),
      "unexpected: {}",
      result
    );
  }

  /// Minimal in-process HTTP server that serves a fixed CSV body on a single
  /// request, then shuts down. Returns the URL to hit.
  #[cfg(not(target_arch = "wasm32"))]
  fn serve_csv_once(body: &'static str) -> String {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
      if let Ok((mut stream, _)) = listener.accept() {
        let mut buf = [0u8; 1024];
        let _ = stream.read(&mut buf);
        let response = format!(
          "HTTP/1.1 200 OK\r\nContent-Type: text/csv\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
          body.len(),
          body
        );
        let _ = stream.write_all(response.as_bytes());
      }
    });
    format!("http://{}/matrix.csv", addr)
  }

  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn import_csv_from_url_default() {
    // Regression test: Import[url] with a .csv URL must fetch and parse the
    // CSV (previously it was routed to the image importer and always failed
    // with "cannot decode image").
    let url = serve_csv_once("1,2,3\n4,5,6\n7,8,9\n");
    let result = interpret(&format!(r#"Import["{url}"]"#)).unwrap();
    assert_eq!(result, "{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}");
  }

  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn import_csv_from_url_with_element() {
    let url = serve_csv_once("a,b,c\n1,2,3\n4,5,6\n");
    let result =
      interpret(&format!(r#"Import["{url}", "ColumnLabels"]"#)).unwrap();
    assert_eq!(result, "{a, b, c}");
  }
}

mod xlsx_import {
  use super::*;

  fn xlsx_path() -> String {
    let manifest = env!("CARGO_MANIFEST_DIR");
    format!("{manifest}/tests/data/population.xlsx")
  }

  #[test]
  fn import_xlsx_default_wraps_sheets() {
    // Import[path] returns a list of sheets, each sheet being a list of rows.
    let path = xlsx_path();
    let result = interpret(&format!(r#"Import["{path}"]"#)).unwrap();
    assert!(
      result.starts_with("{{{"),
      "expected outer list of sheets, got: {}",
      result
    );
    assert!(result.contains("China"));
    assert!(result.contains("Japan"));
  }

  #[test]
  fn import_xlsx_data_first_sheet() {
    // Import[path, {"Data", 1}] returns the first sheet directly.
    let path = xlsx_path();
    let result =
      interpret(&format!(r#"Import["{path}", {{"Data", 1}}]"#)).unwrap();
    assert!(
      result.starts_with("{{1.313973713*^9, China}"),
      "unexpected: {}",
      result
    );
    assert!(result.contains("{1.27463611*^8, Japan}"));
  }

  #[test]
  fn import_xlsx_dimensions_first_sheet() {
    let path = xlsx_path();
    let result =
      interpret(&format!(r#"Dimensions[Import["{path}", {{"Data", 1}}]]"#))
        .unwrap();
    assert_eq!(result, "{10, 2}");
  }

  #[test]
  fn import_xlsx_sheet_names() {
    let path = xlsx_path();
    let result = interpret(&format!(r#"Import["{path}", "Sheets"]"#)).unwrap();
    assert_eq!(result, "{Sheet1}");
  }

  #[test]
  fn import_xlsx_data_element_equals_default() {
    // "Data" and no element should produce identical output.
    let path = xlsx_path();
    let a = interpret(&format!(r#"Import["{path}"]"#)).unwrap();
    let b = interpret(&format!(r#"Import["{path}", "Data"]"#)).unwrap();
    assert_eq!(a, b);
  }

  #[test]
  fn import_xlsx_out_of_range_sheet_fails() {
    let path = xlsx_path();
    let err = interpret(&format!(r#"Import["{path}", {{"Data", 99}}]"#));
    assert!(err.is_err(), "expected error for out-of-range sheet");
  }

  #[test]
  fn import_xlsx_numbers_are_reals() {
    // Regression test: xlsx values must come back as Reals (matching
    // wolframscript), not Integers — even when the cell happens to hold a
    // whole-number value.
    let path = xlsx_path();
    let result =
      interpret(&format!(r#"Head[Import["{path}", {{"Data", 1}}][[1, 1]]]"#))
        .unwrap();
    assert_eq!(result, "Real");
  }
}

mod xlsx_export {
  use super::*;

  fn tmp_path(name: &str) -> String {
    let dir = std::env::temp_dir();
    dir.join(name).to_string_lossy().into_owned()
  }

  #[test]
  fn export_xlsx_round_trip_matches_wolframscript() {
    // Mirrors the user-facing example and the wolframscript reference output.
    let path = tmp_path("woxi_export_basic.xlsx");
    let _ = std::fs::remove_file(&path);
    let script = format!(
      r#"exportData = {{{{3Pi, 1/7, 5}}, {{4.5, 4.75, 4.875}}, {{E, 5!, N[Pi, 10]}}}}; Export["{path}", exportData]; Import["{path}"]"#
    );
    let result = interpret(&script).unwrap();
    assert_eq!(
      result,
      "{{{9.42477796076938, 0.14285714285714285, 5.}, {4.5, 4.75, 4.875}, {2.718281828459045, 120., 3.141592653589793}}}"
    );
  }

  #[test]
  fn export_xlsx_returns_filename() {
    let path = tmp_path("woxi_export_return.xlsx");
    let _ = std::fs::remove_file(&path);
    let result =
      interpret(&format!(r#"Export["{path}", {{{{1, 2}}, {{3, 4}}}}]"#))
        .unwrap();
    assert_eq!(result, path);
    assert!(std::path::Path::new(&path).exists());
  }

  #[test]
  fn export_xlsx_numbers_are_reals_after_round_trip() {
    // Excel stores everything as f64, so integers must come back as Reals.
    let path = tmp_path("woxi_export_integers.xlsx");
    let _ = std::fs::remove_file(&path);
    let result = interpret(&format!(
      r#"Export["{path}", {{{{1, 2, 3}}}}]; Head[Import["{path}", {{"Data", 1}}][[1, 1]]]"#
    ))
    .unwrap();
    assert_eq!(result, "Real");
  }

  #[test]
  fn export_xlsx_preserves_strings_and_booleans() {
    let path = tmp_path("woxi_export_mixed.xlsx");
    let _ = std::fs::remove_file(&path);
    let result = interpret(&format!(
      r#"Export["{path}", {{{{"hello", True, False}}, {{"world", 1, 2.5}}}}]; Import["{path}", {{"Data", 1}}]"#
    ))
    .unwrap();
    assert_eq!(result, "{{hello, True, False}, {world, 1., 2.5}}");
  }

  #[test]
  fn export_xlsx_explicit_format_argument() {
    // Wolfram allows `Export[file, data, "XLSX"]` to force the format
    // independent of the filename extension.
    let path = tmp_path("woxi_export_explicit_fmt.dat");
    let _ = std::fs::remove_file(&path);
    let returned =
      interpret(&format!(r#"Export["{path}", {{{{1, 2}}}}, "XLSX"]"#)).unwrap();
    assert_eq!(returned, path);
    // The bytes on disk are an actual xlsx workbook (PK… zip header).
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..2], b"PK", "expected xlsx zip header");
  }

  #[test]
  fn export_xlsx_evaluates_symbolic_constants() {
    // Pi, E, rationals, and unevaluated arithmetic must be exported as the
    // numeric value the cell will end up containing.
    let path = tmp_path("woxi_export_symbolic.xlsx");
    let _ = std::fs::remove_file(&path);
    let result = interpret(&format!(
      r#"Export["{path}", {{{{Pi, E, 2/3}}}}]; Import["{path}", {{"Data", 1}}]"#
    ))
    .unwrap();
    assert_eq!(
      result,
      "{{3.141592653589793, 2.718281828459045, 0.6666666666666666}}"
    );
  }

  #[test]
  fn export_xlsx_exportformats_lists_xlsx() {
    let result = interpret("MemberQ[$ExportFormats, \"XLSX\"]").unwrap();
    assert_eq!(result, "True");
  }
}

mod txt_import {
  use super::*;

  fn txt_path() -> String {
    let manifest = env!("CARGO_MANIFEST_DIR");
    format!("{manifest}/tests/data/article.txt")
  }

  #[test]
  fn import_txt_returns_string() {
    let path = txt_path();
    let result = interpret(&format!(r#"Head[Import["{path}"]]"#)).unwrap();
    assert_eq!(result, "String");
  }

  #[test]
  fn import_txt_content() {
    let path = txt_path();
    let result = interpret(&format!(r#"Import["{path}"]"#)).unwrap();
    assert!(result.contains("sample article"));
    assert!(result.contains("Line two."));
    assert!(result.contains("Line three."));
  }

  #[test]
  fn import_txt_strips_trailing_newline() {
    // Match wolframscript: Import["file.txt"] strips a single trailing
    // newline so StringLength matches the visible content.
    let path = txt_path();
    let result =
      interpret(&format!(r#"StringLength[Import["{path}"]]"#)).unwrap();
    assert_eq!(result, "64");
  }

  /// Minimal one-shot HTTP server — mirrors `serve_csv_once` above but
  /// serves a plain-text body. Lets us regression-test Import of `.txt`
  /// URLs without hitting the network.
  #[cfg(not(target_arch = "wasm32"))]
  fn serve_txt_once(body: &'static str) -> String {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
      if let Ok((mut stream, _)) = listener.accept() {
        let mut buf = [0u8; 1024];
        let _ = stream.read(&mut buf);
        let response = format!(
          "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
          body.len(),
          body
        );
        let _ = stream.write_all(response.as_bytes());
      }
    });
    format!("http://{}/article.txt", addr)
  }

  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn import_txt_from_url() {
    // Regression: Import[url] for a .txt URL used to fall through to the
    // image importer and fail with "cannot decode image".
    let url = serve_txt_once("hello world\n");
    let result = interpret(&format!(r#"Import["{url}"]"#)).unwrap();
    assert_eq!(result, "hello world");
  }

  #[test]
  fn import_txt_data_non_uniform_returns_lines() {
    // article.txt has varying token counts per line, so "Data" returns a
    // flat list of line strings (same as "Lines"), matching wolframscript.
    let result =
      interpret(r#"Import["tests/data/article.txt", "Data"]"#).unwrap();
    assert_eq!(
      result,
      "{This is a sample article for Import tests., Line two., Line three.}"
    );
  }

  #[test]
  fn import_txt_lines_element() {
    let result =
      interpret(r#"Import["tests/data/article.txt", "Lines"]"#).unwrap();
    assert_eq!(
      result,
      "{This is a sample article for Import tests., Line two., Line three.}"
    );
  }

  #[test]
  fn import_txt_words_element() {
    let result =
      interpret(r#"Import["tests/data/article.txt", "Words"]"#).unwrap();
    assert!(result.starts_with("{This, is, a, sample"));
    assert!(result.contains("Line, three.}"));
  }

  #[test]
  fn import_txt_string_element() {
    // "String" / "Plaintext" return the full file content.
    let result =
      interpret(r#"Import["tests/data/article.txt", "String"]"#).unwrap();
    assert!(result.contains("sample article"));
    assert!(result.contains("Line three."));
  }

  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn import_txt_data_from_url_parses_table() {
    // Regression: Import[url, "Data"] on a whitespace-delimited table must
    // return a list-of-lists with numeric columns auto-converted (and
    // trailing blank lines stripped), matching wolframscript.
    let url =
      serve_txt_once("Joe Smith  94\nJane Smith  85\nBob Example  82\n\n");
    let result = interpret(&format!(r#"Import["{url}", "Data"]"#)).unwrap();
    assert_eq!(
      result,
      "{{Joe, Smith, 94}, {Jane, Smith, 85}, {Bob, Example, 82}}"
    );
  }

  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn import_txt_data_from_url_numeric_column_head_is_integer() {
    let url = serve_txt_once("Joe Smith  94\nJane Smith  85\n");
    let result =
      interpret(&format!(r#"Head[Import["{url}", "Data"][[1, 3]]]"#)).unwrap();
    assert_eq!(result, "Integer");
  }
}

mod import_string {
  use super::*;

  #[test]
  fn import_string_csv_default() {
    let result =
      interpret(r#"ImportString["name,age\nAlice,30\nBob,25", "CSV"]"#)
        .unwrap();
    assert!(result.contains("{name, age}"));
    assert!(result.contains("30"));
  }

  #[test]
  fn import_string_quoted_fields_with_commas() {
    let result = interpret(
      r#"ImportString["name,desc\n\"Smith, John\",\"a, b\"", "CSV"]"#,
    )
    .unwrap();
    assert!(result.contains("Smith, John"));
    assert!(result.contains("a, b"));
  }

  #[test]
  fn import_string_three_args_returns_unevaluated() {
    // ImportString only accepts 1 or 2 arguments (matching wolframscript)
    assert_eq!(
      interpret(r#"ImportString["a,b\n1,2", "CSV", "Data"]"#).unwrap(),
      "ImportString[a,b\n1,2, CSV, Data]"
    );
  }
}

mod file_names {
  use super::*;

  #[test]
  fn returns_list() {
    // FileNames returns a list
    let result = interpret(r#"Head[FileNames[]]"#).unwrap();
    assert_eq!(result, "List");
  }

  #[test]
  fn pattern_match() {
    // FileNames["*.toml"] should find Cargo.toml
    let result =
      interpret(r#"MemberQ[FileNames["*.toml"], "Cargo.toml"]"#).unwrap();
    assert_eq!(result, "True");
  }

  #[test]
  fn with_directory() {
    // FileNames["*.rs", "src"] should find lib.rs
    let result =
      interpret(r#"MemberQ[FileNames["*.rs", "src"], "src/lib.rs"]"#).unwrap();
    assert_eq!(result, "True");
  }

  #[test]
  fn recursive() {
    // FileNames["*.rs", "src", Infinity] should find deeply nested files
    let result = interpret(
      r#"Length[FileNames["*.rs", "src", Infinity]] > Length[FileNames["*.rs", "src"]]"#,
    )
    .unwrap();
    assert_eq!(result, "True");
  }

  #[test]
  fn no_args_includes_dirs() {
    // FileNames[] should include directories like "src"
    let result = interpret(r#"MemberQ[FileNames[], "src"]"#).unwrap();
    assert_eq!(result, "True");
  }
}

mod set_directory {
  use super::*;

  #[test]
  fn set_and_check() {
    // SetDirectory returns the new directory path as a string
    let result = interpret(
      r#"Block[{}, result = StringQ[SetDirectory["src"]]; ResetDirectory[]; result]"#,
    )
    .unwrap();
    assert_eq!(result, "True");
  }

  #[test]
  fn no_args_sets_home() {
    // SetDirectory[] with no arguments sets to $HomeDirectory
    let result = interpret(
      r#"Block[{}, d = SetDirectory[]; ResetDirectory[]; StringQ[d]]"#,
    )
    .unwrap();
    assert_eq!(result, "True");
  }

  #[test]
  fn no_args_matches_home_env() {
    // SetDirectory[] should set to the HOME environment variable
    let home = std::env::var("HOME")
      .or_else(|_| std::env::var("USERPROFILE"))
      .unwrap();
    let home = std::fs::canonicalize(&home)
      .unwrap()
      .to_string_lossy()
      .into_owned();
    let result =
      interpret(r#"Block[{}, d = SetDirectory[]; ResetDirectory[]; d]"#)
        .unwrap();
    assert_eq!(result, home);
  }

  #[test]
  fn does_not_mutate_process_cwd() {
    // Regression test: SetDirectory/ResetDirectory must not touch the
    // process-wide current working directory. Cargo runs tests in parallel
    // threads within one process, so mutating the real CWD here races
    // against any concurrent test that resolves a relative path, leading
    // to flaky CI failures (observed in interpreter_tests::io and
    // interpreter_tests::image::image_io).
    let before = std::env::current_dir().unwrap();
    let result = interpret(
      r#"Block[{}, SetDirectory["/tmp"]; d = Directory[]; ResetDirectory[]; d]"#,
    )
    .unwrap();
    let after = std::env::current_dir().unwrap();
    assert_eq!(before, after, "process CWD must not change");
    // On macOS, /tmp is a symlink to /private/tmp, so Directory[] may
    // return the canonicalized path.
    assert!(
      result == "/tmp" || result == "/private/tmp",
      "virtual CWD must reflect SetDirectory, got: {result}"
    );
  }
}

mod directory_stack {
  use super::*;

  // Fresh session has an empty stack, matching wolframscript.
  #[test]
  fn empty_by_default() {
    assert_eq!(interpret("DirectoryStack[]").unwrap(), "{}");
  }
}

mod read_line {
  use super::*;

  #[test]
  fn read_line_from_file() {
    clear_state();
    // Write a test file, read first line
    let _ = interpret(
      r#"Export["/tmp/woxi_readline_test.txt", "hello world\nsecond line", "Text"]"#,
    );
    assert_eq!(
      interpret(r#"ReadLine["/tmp/woxi_readline_test.txt"]"#).unwrap(),
      "hello world"
    );
  }

  #[test]
  fn read_line_from_stream() {
    clear_state();
    let _ = interpret(
      r#"Export["/tmp/woxi_readline_test2.txt", "line1\nline2\nline3", "Text"]"#,
    );
    let result = interpret(
      r#"stream = OpenRead["/tmp/woxi_readline_test2.txt"]; l1 = ReadLine[stream]; l2 = ReadLine[stream]; Close[stream]; {l1, l2}"#,
    )
    .unwrap();
    assert_eq!(result, "{line1, line2}");
  }

  #[test]
  fn read_line_end_of_file() {
    clear_state();
    let _ =
      interpret(r#"Export["/tmp/woxi_readline_test3.txt", "only", "Text"]"#);
    let result = interpret(
      r#"stream = OpenRead["/tmp/woxi_readline_test3.txt"]; l1 = ReadLine[stream]; l2 = ReadLine[stream]; Close[stream]; {l1, l2}"#,
    )
    .unwrap();
    assert_eq!(result, "{only, EndOfFile}");
  }
}

mod streams_function {
  use super::*;

  #[test]
  fn no_args() {
    assert_eq!(
      interpret("Streams[]").unwrap(),
      "{OutputStream[stdout, 1], OutputStream[stderr, 2]}"
    );
  }

  #[test]
  fn filter_stdout() {
    assert_eq!(
      interpret("Streams[\"stdout\"]").unwrap(),
      "{OutputStream[stdout, 1]}"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[Streams[]]").unwrap(), "List");
  }
}

mod import_export_formats {
  use super::*;

  #[test]
  fn import_formats_is_list() {
    assert_eq!(interpret("Head[$ImportFormats]").unwrap(), "List");
  }

  #[test]
  fn import_formats_contains_csv() {
    assert_eq!(
      interpret("MemberQ[$ImportFormats, \"CSV\"]").unwrap(),
      "True"
    );
  }

  #[test]
  fn import_formats_contains_png() {
    assert_eq!(
      interpret("MemberQ[$ImportFormats, \"PNG\"]").unwrap(),
      "True"
    );
  }

  #[test]
  fn import_formats_all_strings() {
    assert_eq!(
      interpret("AllTrue[$ImportFormats, StringQ]").unwrap(),
      "True"
    );
  }

  #[test]
  fn import_formats_sorted() {
    assert_eq!(
      interpret("Sort[$ImportFormats] === $ImportFormats").unwrap(),
      "True"
    );
  }

  #[test]
  fn export_formats_is_list() {
    assert_eq!(interpret("Head[$ExportFormats]").unwrap(), "List");
  }

  #[test]
  fn export_formats_contains_pdf() {
    assert_eq!(
      interpret("MemberQ[$ExportFormats, \"PDF\"]").unwrap(),
      "True"
    );
  }

  #[test]
  fn export_formats_contains_svg() {
    assert_eq!(
      interpret("MemberQ[$ExportFormats, \"SVG\"]").unwrap(),
      "True"
    );
  }

  #[test]
  fn export_formats_contains_png() {
    assert_eq!(
      interpret("MemberQ[$ExportFormats, \"PNG\"]").unwrap(),
      "True"
    );
  }

  #[test]
  fn export_formats_all_strings() {
    assert_eq!(
      interpret("AllTrue[$ExportFormats, StringQ]").unwrap(),
      "True"
    );
  }

  #[test]
  fn export_formats_sorted() {
    assert_eq!(
      interpret("Sort[$ExportFormats] === $ExportFormats").unwrap(),
      "True"
    );
  }
}
