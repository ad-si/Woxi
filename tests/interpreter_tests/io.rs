use super::*;

fn missing_file() -> String {
  format!(
    "ExampleData{0}sunflowers.jpg",
    std::path::MAIN_SEPARATOR_STR
  )
}

fn temp_file(file: &str) -> String {
  let tmp = std::env::temp_dir().join(file);
  tmp.display().to_string()
}

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

  #[test]
  fn date_object_year_granularity() {
    // DateObject[{y}] → DateObject[{y}, Year].
    assert_eq!(
      interpret("DateObject[{2022}]").unwrap(),
      "DateObject[{2022}, Year]"
    );
  }

  #[test]
  fn date_object_month_granularity() {
    // DateObject[{y, m}] → DateObject[{y, m}, Month].
    assert_eq!(
      interpret("DateObject[{2022, 12}]").unwrap(),
      "DateObject[{2022, 12}, Month]"
    );
  }

  #[test]
  fn date_object_day_granularity_remains() {
    // Existing behaviour: DateObject[{y, m, d}] stays at Day.
    assert_eq!(
      interpret("DateObject[{2022, 12, 5}]").unwrap(),
      "DateObject[{2022, 12, 5}, Day]"
    );
  }

  #[test]
  fn date_object_hour_granularity() {
    // DateObject[{y, m, d, h}] → DateObject[{y, m, d, h}, Hour, Gregorian, 0.]
    assert_eq!(
      interpret("DateObject[{2022, 12, 5, 14}]").unwrap(),
      "DateObject[{2022, 12, 5, 14}, Hour, Gregorian, 0.]"
    );
  }

  #[test]
  fn date_object_minute_granularity() {
    assert_eq!(
      interpret("DateObject[{2022, 12, 5, 14, 30}]").unwrap(),
      "DateObject[{2022, 12, 5, 14, 30}, Minute, Gregorian, 0.]"
    );
  }

  #[test]
  fn date_object_instant_granularity() {
    assert_eq!(
      interpret("DateObject[{2022, 12, 5, 14, 30, 45}]").unwrap(),
      "DateObject[{2022, 12, 5, 14, 30, 45}, Instant, Gregorian, 0.]"
    );
  }

  #[test]
  fn date_object_empty_returns_instant() {
    // DateObject[] returns the current instant — Head must be DateObject and
    // the granularity slot must be Instant.
    assert_eq!(interpret("Head[DateObject[]]").unwrap(), "DateObject");
    assert_eq!(interpret("DateObject[][[2]]").unwrap(), "Instant");
    assert_eq!(interpret("DateObject[][[3]]").unwrap(), "Gregorian");
    assert_eq!(interpret("Length[DateObject[][[1]]]").unwrap(), "6");
  }

  #[test]
  fn date_object_epoch_seconds_zero() {
    // wolframscript: `DateObject[0]` is the absolute-seconds epoch
    // (1900-01-01 00:00:00).
    assert_eq!(
      interpret("DateObject[0]").unwrap(),
      "DateObject[{1900, 1, 1, 0, 0, 0}, Instant, Gregorian, 0.]"
    );
  }

  #[test]
  fn date_object_epoch_seconds_100() {
    // 100 seconds after epoch → 00:01:40 on 1900-01-01.
    assert_eq!(
      interpret("DateObject[100]").unwrap(),
      "DateObject[{1900, 1, 1, 0, 1, 40}, Instant, Gregorian, 0.]"
    );
  }

  #[test]
  fn date_object_audit_case() {
    // Audit case: DateObject[3865673600] = 2022-07-01 14:13:20 UTC.
    assert_eq!(
      interpret("DateObject[3865673600]").unwrap(),
      "DateObject[{2022, 7, 1, 14, 13, 20}, Instant, Gregorian, 0.]"
    );
  }

  #[test]
  fn date_object_y2k_seconds() {
    // 100 Julian years from 1900-01-01 to 2000-01-01 is 3155760000s in
    // wolframscript's count, putting that point at 2000-01-02 00:00:00
    // (the 25-leap-year adjustment vs 100×365×86400 = 3153600000s).
    assert_eq!(
      interpret("DateObject[3155760000]").unwrap(),
      "DateObject[{2000, 1, 2, 0, 0, 0}, Instant, Gregorian, 0.]"
    );
  }
}

mod find {
  use super::*;

  #[test]
  fn find_matching_line() {
    let path = temp_file("woxi_test_find.txt");
    std::fs::write(&path, "hello world\nfoo bar\nbaz").unwrap();
    let result = interpret(&format!("Find[\"{path}\", \"foo\"]")).unwrap();
    assert_eq!(result, "foo bar");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn find_no_match() {
    let path = temp_file("woxi_test_find2.txt");
    std::fs::write(&path, "hello world\nfoo bar").unwrap();
    let result = interpret(&format!("Find[\"{path}\", \"xyz\"]")).unwrap();
    assert_eq!(result, "EndOfFile");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn find_string_stream_with_list_of_terms() {
    // Find on a StringToStream-backed stream, with a list of search
    // terms — returns the first line containing any term.
    let result = interpret(
      r#"stream = StringToStream["alpha line\nby which vast amounts of power and large quantities of new radium-like\nomega"]; Find[stream, {"energy", "power"}]"#,
    )
    .unwrap();
    assert_eq!(
      result,
      "by which vast amounts of power and large quantities of new radium-like"
    );
  }

  #[test]
  fn find_advances_stream_position() {
    // Consecutive Find calls walk forward through the stream.
    let result = interpret(
      r#"stream = StringToStream["foo bar\nbaz\nfoo qux"]; Find[stream, "foo"]; Find[stream, "foo"]"#,
    )
    .unwrap();
    assert_eq!(result, "foo qux");
  }
}

mod get {
  use super::*;
  use std::io::Write;

  fn write_temp(name: &str, content: &str) -> String {
    let path = temp_file(&format!("woxi_test_{}.txt", name));
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
    let path = temp_file("woxi_nonexistent_file.wl");
    let result = interpret(&format!("Get[\"{path}\"]")).unwrap();
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
    let mut sep = std::path::MAIN_SEPARATOR_STR;
    sep = if sep == "\\" { "\\\\" } else { sep };
    let result =
      interpret(&format!("StringContainsQ[Directory[], \"{sep}\"]")).unwrap();
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
    let path = missing_file();
    let result = interpret(&format!(r#"OpenRead["{path}"]"#)).unwrap();
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

  // `WriteString[$Output, …]` writes to stdout and, unlike the raw print!
  // path, is captured by `interpret_with_stdout` (no trailing newline).
  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn write_string_dollar_output_captured() {
    let result =
      interpret_with_stdout(r#"WriteString[$Output, "Goodbye, World!"]"#)
        .unwrap();
    assert_eq!(result.stdout, "Goodbye, World!");
  }
}

mod find_file {
  use super::*;

  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn missing_file_returns_failed() {
    let path = missing_file();
    assert_eq!(
      interpret(&format!(r#"FindFile["{path}"]"#)).unwrap(),
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
    let path = missing_file();
    assert_eq!(
      interpret(&format!(r#"AbsoluteFileName["{path}"]"#)).unwrap(),
      "$Failed"
    );
  }
}

mod close_error {
  use super::*;

  // `Close[symbol]` for a non-stream value emits `Close::stream` and
  // returns unevaluated, matching wolframscript.
  #[test]
  fn non_stream_emits_stream_message() {
    let result = interpret_with_stdout(r#"Close[strm]"#).unwrap();
    assert_eq!(result.result, "Close[strm]");
    assert!(
      result.warnings.iter().any(|w| w.contains(
        "Close::stream: strm is not a string, SocketObject, InputStream[ ] or OutputStream[ ]."
      )),
      "expected Close::stream warning, got {:?}",
      result.warnings
    );
  }
}

mod rename_file {
  use super::*;

  // Missing source: emit `RenameFile::fdnfnd` with the absolute path
  // and return $Failed.
  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn missing_source_returns_failed() {
    let result = interpret_with_stdout(
      r#"RenameFile["MathicsSunflowers.jpg", "MathicsSunnyFlowers.jpg"]"#,
    )
    .unwrap();
    assert_eq!(result.result, "$Failed");
    assert!(
      result
        .warnings
        .iter()
        .any(|w| w.contains("RenameFile::fdnfnd: Directory or file ")
          && w.contains("MathicsSunflowers.jpg")),
      "expected RenameFile::fdnfnd warning with absolute path, got {:?}",
      result.warnings
    );
  }
}

mod rename_directory {
  use super::*;

  // Round trip against a fresh temp directory: successful renames return
  // the absolute destination path, missing sources emit `fdnfnd` with the
  // absolute path, and existing destinations emit `eexist` with the path
  // as given — all matching wolframscript.
  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn rename_semantics() {
    let base = std::env::temp_dir()
      .join(format!("woxi_rename_dir_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(base.join("src")).unwrap();
    let b = base.to_str().unwrap();

    // Success returns the destination path (absolute in, absolute out).
    let result = interpret_with_stdout(&format!(
      r#"RenameDirectory["{b}/src", "{b}/dst"]"#
    ))
    .unwrap();
    assert_eq!(result.result, format!("{b}/dst"));
    assert!(base.join("dst").is_dir() && !base.join("src").exists());

    // Missing source: fdnfnd with the absolute path, $Failed.
    let result = interpret_with_stdout(&format!(
      r#"RenameDirectory["{b}/nosuch", "{b}/x"]"#
    ))
    .unwrap();
    assert_eq!(result.result, "$Failed");
    assert!(
      result.warnings.iter().any(|w| w.contains(&format!(
        "RenameDirectory::fdnfnd: Directory or file \"{b}/nosuch\" not found."
      ))),
      "expected fdnfnd warning, got {:?}",
      result.warnings
    );

    // Existing destination: eexist with the path as given, $Failed.
    std::fs::create_dir_all(base.join("src2")).unwrap();
    let result = interpret_with_stdout(&format!(
      r#"RenameDirectory["{b}/src2", "{b}/dst"]"#
    ))
    .unwrap();
    assert_eq!(result.result, "$Failed");
    assert!(
      result.warnings.iter().any(|w| w.contains(&format!(
        "RenameDirectory::eexist: {b}/dst already exists."
      ))),
      "expected eexist warning, got {:?}",
      result.warnings
    );

    let _ = std::fs::remove_dir_all(&base);
  }

  // Non-string arguments stay unevaluated without a message; wrong arg
  // counts emit argr/argrx like wolframscript.
  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn argument_checks() {
    let result = interpret_with_stdout(r#"RenameDirectory[5, "x"]"#).unwrap();
    assert_eq!(result.result, "RenameDirectory[5, x]");
    assert!(result.warnings.is_empty(), "got {:?}", result.warnings);

    let result = interpret_with_stdout(r#"RenameDirectory["a"]"#).unwrap();
    assert_eq!(result.result, "RenameDirectory[a]");
    assert!(
      result.warnings.iter().any(|w| w.contains(
        "RenameDirectory::argr: RenameDirectory called with 1 argument; 2 arguments are expected."
      )),
      "expected argr warning, got {:?}",
      result.warnings
    );

    let result =
      interpret_with_stdout(r#"RenameDirectory["a", "b", "c"]"#).unwrap();
    assert_eq!(result.result, "RenameDirectory[a, b, c]");
    assert!(
      result.warnings.iter().any(|w| w.contains(
        "RenameDirectory::argrx: RenameDirectory called with 3 arguments; 2 arguments are expected."
      )),
      "expected argrx warning, got {:?}",
      result.warnings
    );
  }
}

mod find_stream {
  use super::*;

  // `Find[sym, "text"]` emits `Find::stream` and returns $Failed,
  // matching wolframscript.
  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn non_stream_emits_stream_message() {
    let result = interpret_with_stdout(r#"Find[stream, "uranium"]"#).unwrap();
    assert_eq!(result.result, "$Failed");
    assert!(
      result.warnings.iter().any(|w| w.contains(
        "Find::stream: stream is not a string, SocketObject, InputStream[ ] or OutputStream[ ]."
      )),
      "expected Find::stream warning, got {:?}",
      result.warnings
    );
  }
}

mod delete_directory {
  use super::*;

  // Non-string argument emits `DeleteDirectory::strs` and returns
  // unevaluated, matching wolframscript.
  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn symbolic_arg_emits_strs() {
    let result = interpret_with_stdout(r#"DeleteDirectory[dir]"#).unwrap();
    assert_eq!(result.result, "DeleteDirectory[dir]");
    assert!(
      result.warnings.iter().any(|w| w.contains(
        "DeleteDirectory::strs: A string or nonempty list of strings is expected at position 1 in DeleteDirectory[dir]."
      )),
      "expected strs warning, got {:?}",
      result.warnings
    );
  }
}

mod copy_file_missing_source {
  use super::*;

  // Missing source: `CopyFile::fdnfnd` with the absolute path, return
  // $Failed. (Placed in a separate module from the existing copy_file
  // tests to avoid churning those.)
  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn message_uses_absolute_path() {
    let path = missing_file();
    let result = interpret_with_stdout(&format!(
      r#"CopyFile["{path}", "MathicsSunflowers.jpg"]"#
    ))
    .unwrap();
    assert_eq!(result.result, "$Failed");
    assert!(
      result
        .warnings
        .iter()
        .any(|w| w.contains("CopyFile::fdnfnd: Directory or file ")
          && w.contains(&path)
          && w.starts_with("CopyFile::fdnfnd: Directory or file \"")),
      "expected absolute-path fdnfnd warning, got {:?}",
      result.warnings
    );
  }
}

mod file_format {
  use super::*;

  // Missing files emit `FileFormat::nffil` and return $Failed,
  // matching wolframscript.
  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn missing_file_returns_failed() {
    let path = missing_file();
    let result =
      interpret_with_stdout(&format!(r#"FileFormat["{path}"]"#)).unwrap();
    assert_eq!(result.result, "$Failed");
    assert!(
      result.warnings.iter().any(|w| w.contains(&format!(
        "FileFormat::nffil: File not found during FileFormat[{path}]."
      ))),
      "expected nffil warning, got {:?}",
      result.warnings
    );
  }
}

mod file_date {
  use super::*;

  // Missing files emit `fdnfnd` and leave FileDate[…] unevaluated,
  // matching wolframscript.
  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn missing_file_emits_fdnfnd() {
    let path = missing_file();
    let result =
      interpret_with_stdout(&format!(r#"FileDate["{path}"]"#)).unwrap();
    assert_eq!(result.result, format!("FileDate[{path}]"));
    assert!(
      result
        .warnings
        .iter()
        .any(|w| w.contains("FileDate::fdnfnd")),
      "expected fdnfnd warning, got {:?}",
      result.warnings
    );
  }
}

mod file_hash {
  use super::*;

  // Missing file: emit `FileHash::noopen` with an absolute path and
  // return $Failed. Actual hashing isn't supported yet, so existing
  // files still return an unevaluated FileHash[…].
  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn missing_file_returns_failed() {
    let path = missing_file();
    let result =
      interpret_with_stdout(&format!(r#"FileHash["{path}"]"#)).unwrap();
    assert_eq!(result.result, "$Failed");
    assert!(
      result
        .warnings
        .iter()
        .any(|w| w.contains("FileHash::noopen: Cannot open")
          && w.contains(&format!("{path}"))),
      "expected FileHash::noopen warning with absolute path, got {:?}",
      result.warnings
    );
  }
}

mod file_byte_count {
  use super::*;

  // Missing files emit `fdnfnd` and return $Failed, matching
  // wolframscript.
  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn missing_file_returns_failed() {
    let path = missing_file();
    assert_eq!(
      interpret(&format!(r#"FileByteCount["{path}"]"#)).unwrap(),
      "$Failed"
    );
  }

  // An existing file returns a positive Integer byte count — verify via
  // the repo's own README.
  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn existing_file_returns_positive_integer() {
    let out = interpret(r#"FileByteCount["readme.md"]"#).unwrap();
    let n: i64 = out.parse().expect("numeric byte count");
    assert!(n > 0);
  }
}

mod delete_file {
  use super::*;

  // Missing files trigger `DeleteFile::fdnfnd` and return the `$Failed`
  // symbol — NOT `$Failed[]` (which was the old, incorrect format).
  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn missing_file_returns_failed_symbol() {
    let result = interpret_with_stdout(r#"DeleteFile["missing.jpg"]"#).unwrap();
    assert_eq!(result.result, "$Failed");
    assert!(
      result.warnings.iter().any(|w| w.contains(
        "DeleteFile::fdnfnd: Directory or file \"missing.jpg\" not found."
      )),
      "expected fdnfnd warning, got {:?}",
      result.warnings
    );
  }

  // A list argument deletes each file in turn; the first missing one
  // aborts with `$Failed`.
  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn list_of_missing_returns_failed() {
    assert_eq!(
      interpret(r#"DeleteFile[{"a.jpg", "b.jpg"}]"#).unwrap(),
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
    let path = temp_file("woxi_test_export_plot.svg");
    let result =
      interpret(&format!("Export[\"{path}\", Plot[Sin[x], {{x, -10, 10}}]]"))
        .unwrap();
    assert_eq!(result, path);
    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.starts_with("<svg"));
    assert!(content.contains("</svg>"));
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn export_string_to_file() {
    clear_state();
    let path = temp_file("woxi_test_export_string.txt");
    let result =
      interpret(&format!("Export[\"{path}\", \"hello world\"]")).unwrap();
    assert_eq!(result, path);
    let content = std::fs::read_to_string(&path).unwrap();
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
    let result = interpret_with_stdout("CityData[1, n, z]").unwrap();
    assert_eq!(result.result, "CityData[1, n, z]");
    assert_eq!(result.warnings.len(), 1);
    assert!(result.warnings[0].contains("not yet implemented"));
    assert!(result.warnings[0].contains("CityData["));
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
    let result = interpret_with_stdout("CityData[1]").unwrap();
    assert!(!result.stdout.contains("not yet implemented"));
    assert!(!result.warnings.is_empty());
  }

  #[test]
  fn multiple_unimplemented_calls_consolidated_into_single_warning() {
    clear_state();
    // MathieuS now has a numerical implementation; use MathieuC which
    // is still unimplemented to exercise the multi-call warning path.
    let result =
      interpret_with_stdout("{CityData[1], MathieuC[1, 1, 1.0]}").unwrap();
    assert_eq!(result.warnings.len(), 1);
    assert!(result.warnings[0].contains("CityData[1]"));
    assert!(result.warnings[0].contains("MathieuC[1, 1, 1.]"));
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

  // ─── ExportString CSV / TSV ───────────────────────────────────────
  // wolframscript emits one row per inner list, comma-separated, with a
  // trailing newline after the last row. Strings are always wrapped in
  // `"…"` and embedded `"` is escaped as `""`.
  #[test]
  fn export_string_csv_2d_int_list() {
    clear_state();
    let result =
      interpret("ExportString[{{1,2,3,4},{3},{2},{4}}, \"CSV\"]").unwrap();
    assert_eq!(result, "1,2,3,4\n3\n2\n4\n");
  }

  #[test]
  fn export_string_csv_flat_list_one_per_row() {
    clear_state();
    let result = interpret("ExportString[{1,2,3,4}, \"CSV\"]").unwrap();
    assert_eq!(result, "1\n2\n3\n4\n");
  }

  // ─── ExportString Text / Lines / List ────────────────────────────
  #[test]
  fn export_string_text_list_newline_joined() {
    clear_state();
    assert_eq!(
      interpret("ExportString[{1, 2, 3}, \"Text\"]").unwrap(),
      "1\n2\n3"
    );
    // Nested elements render in OutputForm.
    assert_eq!(
      interpret("ExportString[{{1, 2}, {3, 4}}, \"Text\"]").unwrap(),
      "{1, 2}\n{3, 4}"
    );
    // Strings are unquoted.
    assert_eq!(
      interpret("ExportString[{\"a\", \"b\"}, \"Lines\"]").unwrap(),
      "a\nb"
    );
    assert_eq!(
      interpret("ExportString[{1, 2, 3}, \"List\"]").unwrap(),
      "1\n2\n3"
    );
  }

  #[test]
  fn export_string_text_atom_verbatim() {
    clear_state();
    assert_eq!(interpret("ExportString[42, \"Text\"]").unwrap(), "42");
    // A string is emitted verbatim, embedded newlines kept.
    assert_eq!(
      interpret("ExportString[\"line1\\nline2\", \"Text\"]").unwrap(),
      "line1\nline2"
    );
  }

  #[test]
  fn export_string_csv_strings_always_quoted() {
    clear_state();
    let result =
      interpret("ExportString[{{\"a\",\"b\"},{\"c\",\"d\"}}, \"CSV\"]")
        .unwrap();
    assert_eq!(result, "\"a\",\"b\"\n\"c\",\"d\"\n");
  }

  #[test]
  fn export_string_csv_string_with_embedded_quote() {
    clear_state();
    let result =
      interpret("ExportString[{{\"he said \\\"hi\\\"\"}}, \"CSV\"]").unwrap();
    // Inner `"` is escaped to `""`, then the cell is wrapped in `"…"`.
    assert_eq!(result, "\"he said \"\"hi\"\"\"\n");
  }

  #[test]
  fn export_string_tsv_uses_tab_separator() {
    clear_state();
    let result = interpret("ExportString[{{1,2},{3,4}}, \"TSV\"]").unwrap();
    assert_eq!(result, "1\t2\n3\t4\n");
  }

  // ─── ExportString Table ────────────────────────────────────────────
  // "Table" is tab-separated like TSV but leaves strings unquoted and emits
  // no trailing newline.
  #[test]
  fn export_string_table_2d_int_list() {
    clear_state();
    let result =
      interpret("ExportString[{{1,2,3},{4,5,6}}, \"Table\"]").unwrap();
    assert_eq!(result, "1\t2\t3\n4\t5\t6");
  }

  #[test]
  fn export_string_table_flat_list_one_per_row() {
    clear_state();
    let result = interpret("ExportString[{1,2,3}, \"Table\"]").unwrap();
    assert_eq!(result, "1\n2\n3");
  }

  #[test]
  fn export_string_table_strings_unquoted() {
    clear_state();
    let result =
      interpret("ExportString[{{\"a\",\"b c\"},{\"d\",\"e\"}}, \"Table\"]")
        .unwrap();
    assert_eq!(result, "a\tb c\nd\te");
  }

  // ─── ExportString JSON ─────────────────────────────────────────────
  // wolframscript pretty-prints JSON: tab-indented, one element per line,
  // `"key":value` with no space after the colon, booleans/Null lowercased,
  // empty containers inline as `[]` / `{}`, and Reals always carry a
  // fractional digit (3.0, not Wolfram's bare `3.`).
  #[test]
  fn export_string_json_flat_list() {
    clear_state();
    let result = interpret("ExportString[{1, 2, 3}, \"JSON\"]").unwrap();
    assert_eq!(result, "[\n\t1,\n\t2,\n\t3\n]");
  }

  #[test]
  fn export_string_json_compact() {
    clear_state();
    // "Compact" -> True emits the value with no extra whitespace.
    assert_eq!(
      interpret("ExportString[{1, 2, 3}, \"JSON\", \"Compact\" -> True]")
        .unwrap(),
      "[1,2,3]"
    );
    assert_eq!(
      interpret(
        "ExportString[<|\"a\" -> 1, \"b\" -> 2|>, \"JSON\", \"Compact\" -> True]"
      )
      .unwrap(),
      "{\"a\":1,\"b\":2}"
    );
    // Nesting stays compact throughout.
    assert_eq!(
      interpret(
        "ExportString[<|\"x\" -> {1, 2}, \"y\" -> \"hi\"|>, \"JSON\", \
         \"Compact\" -> True]"
      )
      .unwrap(),
      "{\"x\":[1,2],\"y\":\"hi\"}"
    );
    // "Compact" -> False keeps the pretty-printed default.
    assert_eq!(
      interpret("ExportString[{1, 2, 3}, \"JSON\", \"Compact\" -> False]")
        .unwrap(),
      "[\n\t1,\n\t2,\n\t3\n]"
    );
  }

  #[test]
  fn export_string_json_scalars() {
    clear_state();
    assert_eq!(interpret("ExportString[5, \"JSON\"]").unwrap(), "5");
    assert_eq!(interpret("ExportString[3.5, \"JSON\"]").unwrap(), "3.5");
    assert_eq!(interpret("ExportString[3.0, \"JSON\"]").unwrap(), "3.0");
    assert_eq!(
      interpret("ExportString[\"hi\", \"JSON\"]").unwrap(),
      "\"hi\""
    );
  }

  #[test]
  fn export_string_json_booleans_and_null() {
    clear_state();
    assert_eq!(interpret("ExportString[True, \"JSON\"]").unwrap(), "true");
    assert_eq!(interpret("ExportString[False, \"JSON\"]").unwrap(), "false");
    assert_eq!(interpret("ExportString[Null, \"JSON\"]").unwrap(), "null");
  }

  #[test]
  fn export_string_json_empty_list_inline() {
    clear_state();
    assert_eq!(interpret("ExportString[{}, \"JSON\"]").unwrap(), "[]");
  }

  #[test]
  fn export_string_json_nested_association() {
    clear_state();
    let result =
      interpret("ExportString[<|\"a\" -> 1, \"b\" -> {2, 3}|>, \"JSON\"]")
        .unwrap();
    assert_eq!(result, "{\n\t\"a\":1,\n\t\"b\":[\n\t\t2,\n\t\t3\n\t]\n}");
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
    let path = temp_file("woxi_readstring_test.txt");
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(b"hello world\nfoo bar").unwrap();
    drop(file);

    let code = format!("ReadString[\"{}\"]", path);
    assert_eq!(interpret(&code).unwrap(), "hello world\nfoo bar");
    let _ = std::fs::remove_file(path);
  }

  #[test]
  fn nonexistent_file() {
    let path = temp_file("woxi_nonexistent_file.wl");
    assert_eq!(
      interpret(&format!("ReadString[\"{path}\"]")).unwrap(),
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
    let path = temp_file("woxi_readlist_test.txt");
    let mut file = std::fs::File::create(&path).unwrap();
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
    let path = temp_file("woxi_test_put_single.txt");
    let result = interpret(&format!("Put[3, \"{path}\"]")).unwrap();
    assert_eq!(result, "\0");
    let content = std::fs::read_to_string(&path).unwrap();
    assert_eq!(content, "3\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_multiple_expressions() {
    clear_state();
    let path = temp_file("woxi_test_put_multi.txt");
    let result = interpret(&format!("Put[1, 2, 3, \"{path}\"]")).unwrap();
    assert_eq!(result, "\0");
    let content = std::fs::read_to_string(&path).unwrap();
    assert_eq!(content, "1\n2\n3\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_list() {
    clear_state();
    let path = temp_file("woxi_test_put_list.txt");
    let result = interpret(&format!("Put[{{1, 2, 3}}, \"{path}\"]")).unwrap();
    assert_eq!(result, "\0");
    let content = std::fs::read_to_string(&path).unwrap();
    assert_eq!(content, "{1, 2, 3}\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_symbolic_expression() {
    clear_state();
    let path = temp_file("woxi_test_put_sym.txt");
    let result = interpret(&format!("Put[x + y, \"{path}\"]")).unwrap();
    assert_eq!(result, "\0");
    let content = std::fs::read_to_string(&path).unwrap();
    assert_eq!(content, "x + y\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_evaluates_argument() {
    clear_state();
    let path = temp_file("woxi_test_put_eval.txt");
    let result = interpret(&format!("Put[1 + 2, \"{path}\"]")).unwrap();
    assert_eq!(result, "\0");
    let content = std::fs::read_to_string(&path).unwrap();
    assert_eq!(content, "3\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_empty_file() {
    clear_state();
    let path = temp_file("woxi_test_put_empty.txt");
    let result = interpret(&format!("Put[\"{path}\"]")).unwrap();
    assert_eq!(result, "\0");
    let content = std::fs::read_to_string(&path).unwrap();
    assert_eq!(content, "");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_rational() {
    clear_state();
    let path = temp_file("woxi_test_put_rat.txt");
    let result = interpret(&format!("Put[1/3, \"{path}\"]")).unwrap();
    assert_eq!(result, "\0");
    let content = std::fs::read_to_string(&path).unwrap();
    assert_eq!(content, "1/3\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_operator_form() {
    clear_state();
    let path = temp_file("woxi_test_put_op.txt");
    let result = interpret(&format!("42 >> \"{path}\"")).unwrap();
    assert_eq!(result, "\0");
    let content = std::fs::read_to_string(&path).unwrap();
    assert_eq!(content, "42\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_overwrites_file() {
    clear_state();
    let path = temp_file("woxi_test_put_overwrite.txt");
    interpret(&format!("Put[1, \"{path}\"]")).unwrap();
    interpret(&format!("Put[2, \"{path}\"]")).unwrap();
    let content = std::fs::read_to_string(&path).unwrap();
    assert_eq!(content, "2\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_return_value_in_list() {
    clear_state();
    let path = temp_file("woxi_test_put_retval.txt");
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
    let path = temp_file("woxi_test_putappend.txt");
    interpret(&format!("Put[1, \"{path}\"]")).unwrap();
    let result = interpret(&format!("PutAppend[2, \"{path}\"]")).unwrap();
    assert_eq!(result, "\0");
    let content = std::fs::read_to_string(&path).unwrap();
    assert_eq!(content, "1\n2\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_append_multiple() {
    clear_state();
    let path = temp_file("woxi_test_putappend_multi.txt");
    interpret(&format!("Put[1, \"{path}\"]")).unwrap();
    interpret(&format!("PutAppend[2, 3, \"{path}\"]")).unwrap();
    let content = std::fs::read_to_string(&path).unwrap();
    assert_eq!(content, "1\n2\n3\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_append_operator_form() {
    clear_state();
    let path = temp_file("woxi_test_putappend_op.txt");
    interpret(&format!("Put[1, \"{path}\"]")).unwrap();
    interpret(&format!("2 >>> \"{path}\"")).unwrap();
    let content = std::fs::read_to_string(&path).unwrap();
    assert_eq!(content, "1\n2\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn put_append_creates_file() {
    clear_state();
    let path = temp_file("woxi_test_putappend_create.txt");
    std::fs::remove_file(&path).ok();
    let result = interpret(&format!("PutAppend[42, \"{path}\"]")).unwrap();
    assert_eq!(result, "\0");
    let content = std::fs::read_to_string(&path).unwrap();
    assert_eq!(content, "42\n");
    std::fs::remove_file(path).ok();
  }
}

mod write {
  use super::*;

  #[test]
  fn write_single_expression() {
    clear_state();
    let path = temp_file("woxi_test_write_single.txt");
    let result = interpret(&format!(
      "str = OpenWrite[\"{path}\"]; Write[str, 42]; Close[str]"
    ))
    .unwrap();
    assert_eq!(result, path);
    let content = std::fs::read_to_string(&path).unwrap();
    assert_eq!(content, "42\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn write_multiple_expressions_concatenated() {
    clear_state();
    let path = temp_file("woxi_test_write_concat.txt");
    interpret(&format!(
      "str = OpenWrite[\"{path}\"]; Write[str, a^2, 1 + b^2]; Close[str]"
    ))
    .unwrap();
    let content = std::fs::read_to_string(&path).unwrap();
    assert_eq!(content, "a^21 + b^2\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn write_multiple_calls() {
    clear_state();
    let path = temp_file("woxi_test_write_multi.txt");
    interpret(&format!(
      "str = OpenWrite[\"{path}\"]; Write[str, 1]; Write[str, 2]; Write[str, 3]; Close[str]"
    ))
    .unwrap();
    let content = std::fs::read_to_string(&path).unwrap();
    assert_eq!(content, "1\n2\n3\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn write_list() {
    clear_state();
    let path = temp_file("woxi_test_write_list.txt");
    interpret(&format!(
      "str = OpenWrite[\"{path}\"]; Write[str, {{1, 2, 3}}]; Close[str]"
    ))
    .unwrap();
    let content = std::fs::read_to_string(&path).unwrap();
    assert_eq!(content, "{1, 2, 3}\n");
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn write_returns_null() {
    clear_state();
    let path = temp_file("woxi_test_write_null.txt");
    let result = interpret(&format!(
      "str = OpenWrite[\"{path}\"]; r = Write[str, 1]; Close[str]; r"
    ))
    .unwrap();
    assert_eq!(result, "\0");
    std::fs::remove_file(&path).ok();
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
  fn read_expression_spans_newlines_inside_string() {
    // A quoted string literal containing a newline must not be cut at
    // that newline when reading an Expression. mathics expected
    // (and woxi now produces) the full two-line string.
    clear_state();
    let result = interpret(
      "stream = StringToStream[\"\\\"Tengo una\\nvaca lechera.\\\"\"]; Read[stream]",
    )
    .unwrap();
    assert_eq!(result, "Tengo una\nvaca lechera.");
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

  #[test]
  fn read_list_negative_count_returns_unevaluated() {
    // wolframscript rejects negative counts with ReadList::intnm;
    // woxi mirrors the behaviour by emitting the message and leaving
    // the call unevaluated rather than silently reading.
    clear_state();
    let result =
      interpret("ReadList[StringToStream[\"a 1 b 2\"], {Word, Number}, -1]")
        .unwrap();
    assert_eq!(
      result,
      "ReadList[InputStream[String, 1], {Word, Number}, -1]"
    );
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
    // `Information` has no Hold attribute in Wolfram, so a plain
    // `Information[a]` after `a = 10` evaluates `a` first and stays as
    // `Information[10]`. The `?a` REPL shortcut wraps the symbol in
    // `Unevaluated` so it still inspects the symbol post-assignment.
    let result = interpret("a = 10; ?a").unwrap();
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
    assert!(result.starts_with("InformationData["));
    assert!(result.ends_with("|>]"));
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
    assert!(result.ends_with("|>]"));
  }

  #[test]
  fn builtin_function_information_call() {
    clear_state();
    let result = interpret("Information[Sin]").unwrap();
    assert!(result.contains("Name -> Sin"));
    assert!(result.contains("Usage -> Returns the sine"));
    // Attributes and Documentation URL must be in the bare form too.
    assert!(result.contains("Attributes -> {"));
    assert!(result.contains("Listable"));
    assert!(result.contains("NumericFunction"));
    // Displayed URL strips the `https://` prefix.
    assert!(
      result.contains("Documentation -> woxi.ad-si.com/docs/"),
      "expected stripped docs URL, got: {result}"
    );
    assert!(
      !result.contains("https://"),
      "textual form should not show the https:// prefix, got: {result}"
    );
    assert!(result.contains("/Sin"));
  }

  #[test]
  fn builtin_information_includes_attributes_and_doc_url() {
    // Plus has multiple attributes and a docs page — the bare
    // Information[Plus] form should surface both.
    clear_state();
    let result = interpret("Information[Plus]").unwrap();
    assert!(result.contains("Attributes -> {"));
    assert!(result.contains("Flat"));
    assert!(result.contains("Orderless"));
    assert!(
      result.contains("Documentation -> woxi.ad-si.com/docs/"),
      "expected stripped docs URL, got: {result}"
    );
    assert!(!result.contains("https://"));
    assert!(result.contains("/Plus"));
  }

  #[test]
  fn builtin_information_svg_has_clickable_doc_link() {
    // The SVG card rendered alongside the textual InformationData[…] form
    // must wrap the documentation URL in an `<a href="https://…">` element
    // (clickable in browsers/SVG viewers) while displaying the URL without
    // the `https://` prefix.
    clear_state();
    let result = interpret_with_stdout("Information[Sin]").unwrap();
    let svg = result
      .graphics
      .expect("Information[Sin] should capture an SVG card");
    assert!(
      svg.contains("<a href=\"https://woxi.ad-si.com/docs/"),
      "SVG should contain a clickable anchor with full https:// href, got: {svg}"
    );
    assert!(
      svg.contains("woxi.ad-si.com/docs/math/elementary/Sin"),
      "SVG should display the docs URL"
    );
    // The visible text should not include "https://" — only the href does.
    let visible_text_only: String = svg
      .split('<')
      .filter_map(|chunk| chunk.split_once('>').map(|(_, txt)| txt))
      .collect::<Vec<_>>()
      .join("");
    assert!(
      !visible_text_only.contains("https://"),
      "visible SVG text should not include https://, got: {visible_text_only}"
    );
  }

  #[test]
  fn builtin_function_information_full_call() {
    clear_state();
    let result = interpret("Information[Sin, \"Full\"]").unwrap();
    assert!(result.contains("Attributes -> {"));
    assert!(result.contains("Listable"));
    assert!(result.contains("NumericFunction"));
    assert!(result.contains("FullName -> System`Sin"));
    assert!(result.ends_with("|>]"));
  }

  #[test]
  fn double_question_mark_user_defined() {
    clear_state();
    let result = interpret("f[x_] := x^2; ??f").unwrap();
    assert!(result.contains("DownValues -> Information`InformationValueForm"));
    assert!(result.contains("f[x_] :> x^2"));
    assert!(result.ends_with("|>]"));
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

  // ── Graphical rendering of InformationData / InformationDataGrid ────

  #[test]
  fn builtin_information_captures_svg() {
    // `Information[Sin]` via `interpret_with_stdout` (the visual entry point
    // used by playground / woxi-studio) should attach a styled SVG card and
    // suppress the textual `InformationData[…]` echo so the host renders
    // only the graphic.
    clear_state();
    let result = interpret_with_stdout("Information[Sin]").unwrap();
    assert_eq!(result.result, "");
    let svg = result
      .graphics
      .expect("Information[Sin] should capture a graphical SVG card");
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("Sin"), "SVG should mention the symbol name");
  }

  #[test]
  fn user_symbol_information_captures_svg() {
    // The `?sym` shortcut on a user-defined symbol should also produce a
    // styled SVG card.
    clear_state();
    let result = interpret_with_stdout("h[x_] := x^2; ?h").unwrap();
    assert_eq!(result.result, "");
    let svg = result
      .graphics
      .expect("?h should capture a graphical SVG card");
    assert!(svg.contains("h"));
    assert!(svg.contains("DownValues"));
  }

  #[test]
  fn wildcard_query_captures_svg() {
    // `?Plot*` should produce a graphical InformationDataGrid SVG.
    clear_state();
    let result = interpret_with_stdout("?Plot*").unwrap();
    assert_eq!(result.result, "");
    let svg = result
      .graphics
      .expect("?Plot* should capture a graphical SVG grid");
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("Plot"));
    assert!(svg.contains("System`"));
  }

  #[test]
  fn cli_information_text_unchanged() {
    // The text-only `interpret()` entry point (CLI, scripts, tests) must
    // continue to return the textual `InformationData[…]` form so existing
    // wolframscript-equivalence tests stay green.
    clear_state();
    let result = interpret("Information[Sin]").unwrap();
    assert!(result.starts_with("InformationData["));
    assert!(result.ends_with("|>]"));
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
      format!("home{0}user{0}file.txt", std::path::MAIN_SEPARATOR_STR)
    );
  }

  #[test]
  fn two_parts() {
    assert_eq!(
      interpret(r#"FileNameJoin[{"a", "b"}]"#).unwrap(),
      format!("a{0}b", std::path::MAIN_SEPARATOR_STR)
    );
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
      format!("dir1{0}dir2{0}file", std::path::MAIN_SEPARATOR_STR)
    );
  }

  #[test]
  fn string_dir_and_file() {
    assert_eq!(
      interpret(r#"ToFileName["dir1", "file"]"#).unwrap(),
      format!("dir1{0}file", std::path::MAIN_SEPARATOR_STR)
    );
  }

  #[test]
  fn list_dirs_only_has_trailing_slash() {
    assert_eq!(
      interpret(r#"ToFileName[{"dir1", "dir2", "dir3"}]"#).unwrap(),
      format!("dir1{0}dir2{0}dir3{0}", std::path::MAIN_SEPARATOR_STR)
    );
  }

  #[test]
  fn single_dir_has_trailing_slash() {
    assert_eq!(
      interpret(r#"ToFileName["just_a_dir"]"#).unwrap(),
      format!("just_a_dir{0}", std::path::MAIN_SEPARATOR_STR)
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

mod file_name_take {
  use super::*;

  // FileNameTake[path] returns the last path component.
  #[test]
  fn default_is_last_component() {
    assert_eq!(interpret(r#"FileNameTake["/a/b/c.txt"]"#).unwrap(), "c.txt");
    assert_eq!(interpret(r#"FileNameTake["a/b/c"]"#).unwrap(), "c");
    assert_eq!(interpret(r#"FileNameTake["c.txt"]"#).unwrap(), "c.txt");
    // A trailing slash is ignored.
    assert_eq!(interpret(r#"FileNameTake["/a/b/"]"#).unwrap(), "b");
  }

  // A positive count takes the first n components; the absolute-path root
  // counts as the first component and renders as "/".
  #[test]
  fn positive_count_takes_leading_components() {
    assert_eq!(interpret(r#"FileNameTake["/a/b/c.txt", 1]"#).unwrap(), "/");
    assert_eq!(interpret(r#"FileNameTake["/a/b/c.txt", 2]"#).unwrap(), "/a");
  }

  // A negative count takes the last |n| components.
  #[test]
  fn negative_count_takes_trailing_components() {
    assert_eq!(
      interpret(r#"FileNameTake["/a/b/c.txt", -1]"#).unwrap(),
      "c.txt"
    );
    assert_eq!(
      interpret(r#"FileNameTake["/a/b/c.txt", -2]"#).unwrap(),
      "b/c.txt"
    );
    assert_eq!(
      interpret(r#"FileNameTake["/usr/local/bin", -2]"#).unwrap(),
      "local/bin"
    );
  }

  // A {m, n} range takes components m through n (1-indexed, inclusive).
  #[test]
  fn range_takes_component_slice() {
    assert_eq!(
      interpret(r#"FileNameTake["/a/b/c.txt", {2, 3}]"#).unwrap(),
      "a/b"
    );
    assert_eq!(
      interpret(r#"FileNameTake["a/b/c", {1, 2}]"#).unwrap(),
      "a/b"
    );
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
  #[cfg(not(target_os = "windows"))]
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

mod http_request {
  use super::*;

  #[test]
  fn url_form_canonicalizes_with_empty_association() {
    assert_eq!(
      interpret(r#"HTTPRequest["https://example.com"]"#).unwrap(),
      "HTTPRequest[https://example.com, <||>]"
    );
  }

  #[test]
  fn url_wrapper_is_unwrapped() {
    assert_eq!(
      interpret(r#"HTTPRequest[URL["https://example.com"]]"#).unwrap(),
      "HTTPRequest[https://example.com, <||>]"
    );
  }

  #[test]
  fn two_argument_form_stays_canonical() {
    assert_eq!(
      interpret(
        r#"HTTPRequest["https://example.com", <|"Method" -> "POST"|>]"#
      )
      .unwrap(),
      "HTTPRequest[https://example.com, <|Method -> POST|>]"
    );
  }

  #[test]
  fn association_form_stays_as_given() {
    assert_eq!(
      interpret(r#"HTTPRequest[<|"Method" -> "POST"|>]"#).unwrap(),
      "HTTPRequest[<|Method -> POST|>]"
    );
  }

  #[test]
  fn head_is_httprequest() {
    assert_eq!(
      interpret(r#"Head[HTTPRequest["https://example.com"]]"#).unwrap(),
      "HTTPRequest"
    );
  }

  #[test]
  fn url_components_are_extracted() {
    let req = r#"req = HTTPRequest["https://user:pass@www.example.com:8080/path/to/file.html?a=1&b=2#frag"];"#;
    for (prop, expected) in [
      (
        "URL",
        "https://user:pass@www.example.com:8080/path/to/file.html?a=1&b=2#frag",
      ),
      ("Scheme", "https"),
      ("User", "user:pass"),
      ("Username", "user"),
      ("Password", "pass"),
      ("Domain", "www.example.com"),
      ("AbsoluteDomain", "https://user:pass@www.example.com:8080"),
      (
        "AbsolutePath",
        "https://user:pass@www.example.com:8080/path/to/file.html",
      ),
      ("Port", "8080"),
      ("Path", "{, path, to, file.html}"),
      ("PathString", "/path/to/file.html"),
      ("Query", "{a -> 1, b -> 2}"),
      ("QueryString", "a=1&b=2"),
      ("Fragment", "frag"),
    ] {
      assert_eq!(
        interpret(&format!(r#"{req} req["{prop}"]"#)).unwrap(),
        expected,
        "property {prop}"
      );
    }
  }

  #[test]
  fn defaults_for_bare_url() {
    let req = r#"req = HTTPRequest["https://example.com"];"#;
    for (prop, expected) in [
      ("Method", "GET"),
      ("Headers", "{user-agent -> Wolfram HTTPClient 15.}"),
      ("UserAgent", "Wolfram HTTPClient 15."),
      ("Body", ""),
      ("BodyBytes", "{}"),
      ("BodyByteArray", "ByteArray[<0>]"),
      ("ContentType", "None"),
      ("Cookies", "Automatic"),
      ("FormRules", "None"),
      ("Query", "{}"),
      ("QueryString", "None"),
      ("Path", "{}"),
      ("PathString", ""),
      ("AbsoluteDomain", "https://example.com"),
      ("AbsolutePath", "https://example.com"),
      ("Port", "None"),
      ("Fragment", "None"),
      ("User", "None"),
      ("Username", "None"),
    ] {
      assert_eq!(
        interpret(&format!(r#"{req} req["{prop}"]"#)).unwrap(),
        expected,
        "property {prop}"
      );
    }
  }

  #[test]
  fn property_list_yields_association() {
    assert_eq!(
      interpret(
        r#"req = HTTPRequest["https://www.wikipedia.org/"]; req[{"Scheme", "Domain", Method}]"#
      )
      .unwrap(),
      "<|Scheme -> https, Domain -> www.wikipedia.org, Method -> GET|>"
    );
  }

  #[test]
  fn method_symbol_property() {
    assert_eq!(
      interpret(r#"HTTPRequest["https://www.wikipedia.org/"][Method]"#)
        .unwrap(),
      "GET"
    );
  }

  #[test]
  fn method_symbol_association_key() {
    assert_eq!(
      interpret(
        r#"HTTPRequest["https://example.com", <|Method -> "POST"|>]["Method"]"#
      )
      .unwrap(),
      "POST"
    );
  }

  #[test]
  fn unknown_property_in_list_maps_to_failed() {
    assert_eq!(
      interpret(
        r#"HTTPRequest["https://www.wikipedia.org/"][{"Scheme", "Bogus"}]"#
      )
      .unwrap(),
      "<|Scheme -> https, Bogus -> $Failed|>"
    );
  }

  #[test]
  fn properties_property_lists_all() {
    assert_eq!(
      interpret(r#"HTTPRequest["https://example.com"]["Properties"]"#).unwrap(),
      "{AbsoluteDomain, AbsolutePath, Body, BodyByteArray, BodyBytes, \
       ContentType, Cookies, Domain, FormRules, Fragment, Headers, Password, \
       Path, PathString, Port, Query, QueryString, Scheme, URL, User, \
       UserAgent, Username, Method}"
    );
  }

  #[test]
  fn explicit_body_sets_default_content_type() {
    assert_eq!(
      interpret(
        r#"HTTPRequest["https://example.com", <|"Body" -> ""|>]["ContentType"]"#
      )
      .unwrap(),
      "text/plain;charset=utf-8"
    );
    assert_eq!(
      interpret(
        r#"HTTPRequest["https://example.com", <|"Body" -> "n=3"|>][{"ContentType", "BodyBytes", "BodyByteArray"}]"#
      )
      .unwrap(),
      "<|ContentType -> text/plain;charset=utf-8, BodyBytes -> {110, 61, 51}, BodyByteArray -> ByteArray[<3>]|>"
    );
  }

  #[test]
  fn content_type_and_user_agent_from_headers() {
    assert_eq!(
      interpret(
        r#"HTTPRequest["http://x.com/", <|"Headers" -> <|"X-Foo" -> "1", "Content-Type" -> "application/json"|>, Method -> "POST"|>][{"Headers", "UserAgent", "ContentType", "Method"}]"#
      )
      .unwrap(),
      "<|Headers -> {x-foo -> 1, content-type -> application/json, \
       user-agent -> Wolfram HTTPClient 15.}, \
       UserAgent -> Wolfram HTTPClient 15., \
       ContentType -> application/json, Method -> POST|>"
    );
    assert_eq!(
      interpret(
        r#"HTTPRequest["http://x.com/", <|"Headers" -> <|"User-Agent" -> "MyBot/1.0"|>|>][{"Headers", "UserAgent"}]"#
      )
      .unwrap(),
      "<|Headers -> {user-agent -> MyBot/1.0}, UserAgent -> MyBot/1.0|>"
    );
  }

  #[test]
  fn method_and_body_from_association() {
    assert_eq!(
      interpret(
        r#"req = HTTPRequest["https://example.com", <|"Method" -> "POST", "Body" -> "x=1"|>]; {req["Method"], req["Body"]}"#
      )
      .unwrap(),
      "{POST, x=1}"
    );
  }

  #[test]
  fn headers_from_association_normalize_to_rule_list() {
    // Header names are lowercased and wolframscript's default user agent
    // is appended when none is given.
    for code in [
      r#"HTTPRequest["https://example.com", <|"Headers" -> <|"Accept" -> "text/html"|>|>]["Headers"]"#,
      r#"HTTPRequest["https://example.com", <|"Headers" -> {"Accept" -> "text/html"}|>]["Headers"]"#,
    ] {
      assert_eq!(
        interpret(code).unwrap(),
        "{accept -> text/html, user-agent -> Wolfram HTTPClient 15.}"
      );
    }
  }

  #[test]
  fn url_is_rebuilt_from_association_components() {
    assert_eq!(
      interpret(
        r#"HTTPRequest[<|"Scheme" -> "https", "Domain" -> "example.com", "Path" -> "/api", "Query" -> {"k" -> "v"}|>]["URL"]"#
      )
      .unwrap(),
      "https://example.com/api?k=v"
    );
  }

  #[test]
  fn association_components_override_url_parts() {
    assert_eq!(
      interpret(
        r#"HTTPRequest["https://example.com/a", <|"Path" -> "/b", "Fragment" -> "top"|>]["URL"]"#
      )
      .unwrap(),
      "https://example.com/b#top"
    );
  }

  #[test]
  fn schemeless_url_is_all_path() {
    assert_eq!(
      interpret(r#"HTTPRequest["example.com/x"]["Domain"]"#).unwrap(),
      "None"
    );
    assert_eq!(
      interpret(
        r#"HTTPRequest["example.com/x"][{"Path", "PathString", "AbsoluteDomain", "AbsolutePath"}]"#
      )
      .unwrap(),
      "<|Path -> {example.com, x}, PathString -> example.com/x, \
       AbsoluteDomain -> , AbsolutePath -> example.com/x|>"
    );
  }

  #[test]
  fn unknown_property_stays_unevaluated() {
    assert_eq!(
      interpret(r#"HTTPRequest["https://example.com"]["Frobnicate"]"#).unwrap(),
      "HTTPRequest[https://example.com, <||>][Frobnicate]"
    );
  }
}

mod url_read {
  use super::*;
  use std::io::{Read, Write};

  /// A minimal single-threaded HTTP server for exercising URLRead without
  /// external network access. Serves `connections` sequential connections;
  /// each request is answered by `respond(head, body)` where `head` is the
  /// raw request head (request line + headers).
  fn serve(
    connections: usize,
    respond: impl Fn(&str, &[u8]) -> Vec<u8> + Send + 'static,
  ) -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
      for _ in 0..connections {
        let Ok((mut stream, _)) = listener.accept() else {
          return;
        };
        let mut data = Vec::new();
        let mut buf = [0u8; 4096];
        let (head_end, body_len) = loop {
          let Ok(n) = stream.read(&mut buf) else { return };
          if n == 0 {
            return;
          }
          data.extend_from_slice(&buf[..n]);
          if let Some(pos) = data.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&data[..pos]).into_owned();
            let len = head
              .lines()
              .find_map(|l| {
                let (name, value) = l.split_once(':')?;
                name
                  .eq_ignore_ascii_case("content-length")
                  .then(|| value.trim().parse::<usize>().ok())?
              })
              .unwrap_or(0);
            break (pos + 4, len);
          }
        };
        while data.len() < head_end + body_len {
          let Ok(n) = stream.read(&mut buf) else { return };
          if n == 0 {
            break;
          }
          data.extend_from_slice(&buf[..n]);
        }
        let head = String::from_utf8_lossy(&data[..head_end]).into_owned();
        let body = data[head_end..].to_vec();
        let response = respond(&head, &body);
        let _ = stream.write_all(&response);
      }
    });
    port
  }

  #[test]
  fn get_returns_http_response_object() {
    let port = serve(1, |_, _| {
      b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nX-Test: Yes\r\n\
        Content-Length: 5\r\nConnection: close\r\n\r\nhello"
        .to_vec()
    });
    assert_eq!(
      interpret(&format!(
        r#"URLRead[HTTPRequest["http://127.0.0.1:{port}/"]]"#
      ))
      .unwrap(),
      "HTTPResponse[ByteArray[<5>], <|Headers -> \
       {{Content-Type, text/plain}, {X-Test, Yes}, {Content-Length, 5}, \
       {Connection, close}}, StatusCode -> 200, Cookies -> {}|>, \
       CharacterEncoding -> Automatic]"
    );
  }

  #[test]
  fn plain_url_string_is_accepted() {
    let port = serve(1, |_, _| {
      b"HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n".to_vec()
    });
    assert_eq!(
      interpret(&format!(r#"URLRead["http://127.0.0.1:{port}/"]"#)).unwrap(),
      "HTTPResponse[ByteArray[<0>], <|Headers -> {{Connection, close}}, \
       StatusCode -> 204, Cookies -> {}|>, CharacterEncoding -> Automatic]"
    );
  }

  #[test]
  fn redirects_are_followed() {
    let port = serve(2, |head, _| {
      let path = head
        .lines()
        .next()
        .and_then(|l| l.split(' ').nth(1))
        .unwrap_or("");
      if path == "/" {
        b"HTTP/1.1 302 Found\r\nLocation: /next\r\nContent-Length: 0\r\n\
          Connection: close\r\n\r\n"
          .to_vec()
      } else {
        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok"
          .to_vec()
      }
    });
    assert_eq!(
      interpret(&format!(
        r#"URLRead[HTTPRequest["http://127.0.0.1:{port}/"]]"#
      ))
      .unwrap(),
      "HTTPResponse[ByteArray[<2>], <|Headers -> {{Content-Length, 2}, \
       {Connection, close}}, StatusCode -> 200, Cookies -> {}|>, \
       CharacterEncoding -> Automatic]"
    );
  }

  #[test]
  fn method_body_and_headers_are_sent() {
    let port = serve(1, |head, body| {
      let method = head
        .lines()
        .next()
        .and_then(|l| l.split(' ').next())
        .unwrap_or("");
      let body = String::from_utf8_lossy(body).into_owned();
      format!(
        "HTTP/1.1 200 OK\r\nX-Method: {method}\r\nX-Body: {body}\r\n\
         Content-Length: 0\r\nConnection: close\r\n\r\n"
      )
      .into_bytes()
    });
    assert_eq!(
      interpret(&format!(
        r#"URLRead[HTTPRequest["http://127.0.0.1:{port}/", <|"Method" -> "POST", "Body" -> "n=3"|>]]"#
      ))
      .unwrap(),
      "HTTPResponse[ByteArray[<0>], <|Headers -> {{X-Method, POST}, \
       {X-Body, n=3}, {Content-Length, 0}, {Connection, close}}, \
       StatusCode -> 200, Cookies -> {}|>, CharacterEncoding -> Automatic]"
    );
  }

  #[test]
  fn default_user_agent_is_sent() {
    // The server echoes the received User-Agent back so the test can
    // observe what was actually sent on the wire.
    let port = serve(1, |head, _| {
      let ua = head
        .lines()
        .find_map(|l| {
          let (name, value) = l.split_once(':')?;
          name
            .eq_ignore_ascii_case("user-agent")
            .then(|| value.trim().to_string())
        })
        .unwrap_or_default();
      format!(
        "HTTP/1.1 200 OK\r\nX-UA: {ua}\r\nContent-Length: 0\r\n\
         Connection: close\r\n\r\n"
      )
      .into_bytes()
    });
    assert_eq!(
      interpret(&format!(
        r#"URLRead[HTTPRequest["http://127.0.0.1:{port}/"]]"#
      ))
      .unwrap(),
      "HTTPResponse[ByteArray[<0>], <|Headers -> \
       {{X-UA, Wolfram HTTPClient 15.}, {Content-Length, 0}, \
       {Connection, close}}, StatusCode -> 200, Cookies -> {}|>, \
       CharacterEncoding -> Automatic]"
    );
  }

  #[test]
  fn connection_failure_returns_failure_object() {
    // Reserved-TLD host: name resolution fails deterministically, matching
    // wolframscript's URLRead::invhttp + Failure["ConnectionFailure", …].
    assert_eq!(
      interpret(
        r#"URLRead[HTTPRequest["http://nonexistent-woxi-test.invalid/"]]"#
      )
      .unwrap(),
      "Could not connect to DisplayForm[TagBox[\
       \"http://nonexistent-woxi-test.invalid/\", Short[#1, 3] & ]]."
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

  #[test]
  fn import_csv_data_row_span() {
    let path = csv_path();
    assert_eq!(
      interpret(&format!(r#"Import["{path}", {{"Data", 1 ;; 2}}]"#)).unwrap(),
      "{{date, name, fruit, quantity}, {2025-09-24, John, banana, 3}}"
    );
  }

  #[test]
  fn import_csv_data_single_row() {
    let path = csv_path();
    assert_eq!(
      interpret(&format!(r#"Import["{path}", {{"Data", 2}}]"#)).unwrap(),
      "{2025-09-24, John, banana, 3}"
    );
  }

  #[test]
  fn import_csv_data_negative_row() {
    let path = csv_path();
    assert_eq!(
      interpret(&format!(r#"Import["{path}", {{"Data", -1}}]"#)).unwrap(),
      "{2025-08-17, Anna, pear, 1}"
    );
  }

  #[test]
  fn import_csv_data_cell() {
    let path = csv_path();
    assert_eq!(
      interpret(&format!(r#"Import["{path}", {{"Data", 2, 3}}]"#)).unwrap(),
      "banana"
    );
  }

  #[test]
  fn import_csv_data_column() {
    let path = csv_path();
    assert_eq!(
      interpret(&format!(r#"Import["{path}", {{"Data", All, 2}}]"#)).unwrap(),
      "{name, John, Anna, Eve, Anna, John, Anna}"
    );
  }

  #[test]
  fn import_csv_data_row_span_with_step() {
    let path = csv_path();
    assert_eq!(
      interpret(&format!(r#"Import["{path}", {{"Data", 2 ;; 6 ;; 2, 4}}]"#))
        .unwrap(),
      "{3, 2, 6}"
    );
  }

  #[test]
  fn import_csv_data_open_ended_span() {
    let path = csv_path();
    assert_eq!(
      interpret(&format!(r#"Import["{path}", {{"Data", 6 ;; All}}]"#)).unwrap(),
      "{{2025-10-11, John, blue berry, 6}, {2025-08-17, Anna, pear, 1}}"
    );
  }

  #[test]
  fn import_csv_data_out_of_range_row_fails() {
    let path = csv_path();
    assert_eq!(
      interpret(&format!(r#"Import["{path}", {{"Data", 99}}]"#)).unwrap(),
      "$Failed"
    );
  }

  /// Regression test for the super-linear large-CSV Import: a compound
  /// statement whose intermediate result is the whole table must not
  /// typeset/format that intermediate (only the final statement's value is
  /// displayed). 20k rows finish in well under a second when linear.
  #[test]
  fn import_large_csv_dimensions() {
    use std::io::Write;
    let dir = std::env::temp_dir().join("woxi-large-csv-test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("large.csv");
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    writeln!(f, "id,name,score").unwrap();
    for i in 0..20_000 {
      writeln!(f, "{i},row{i},{}.5", i % 100).unwrap();
    }
    drop(f);
    let path = path.display().to_string();
    assert_eq!(
      interpret(&format!(r#"d = Import["{path}"]; Dimensions[d]"#)).unwrap(),
      "{20001, 3}"
    );
    std::fs::remove_file(&path).ok();
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

mod root_import {
  use super::*;

  // The sample files were written with uproot (Python) and verified against
  // it: sample.root (zlib-compressed) holds a TObjString "greeting", a TH1D
  // "hist" (4 bins over [1, 5], contents {2, 2, 2, 3}, 9 entries), a TTree
  // "events" (5 entries, branches x/D and n/L), and a subdirectory "subdir"
  // with a TH1D "inner". sample_lz4.root (LZ4-compressed) holds a TH1D
  // "varhist" with variable-width bin edges {0, 1, 2.5, 10} and a
  // TObjString "note".
  fn root_path() -> String {
    let manifest = env!("CARGO_MANIFEST_DIR");
    format!("{manifest}/tests/data/sample.root")
  }

  fn lz4_root_path() -> String {
    let manifest = env!("CARGO_MANIFEST_DIR");
    format!("{manifest}/tests/data/sample_lz4.root")
  }

  #[test]
  fn import_root_default() {
    let path = root_path();
    let result = interpret(&format!(r#"Import["{path}"]"#)).unwrap();
    assert_eq!(
      result,
      "<|greeting -> Hello ROOT, \
       hist -> <|ClassName -> TH1D, Title -> , NBins -> 4, XMin -> 1., \
       XMax -> 5., Entries -> 9., BinContents -> {2., 2., 2., 3.}, \
       Underflow -> 0., Overflow -> 0.|>, \
       events -> <|ClassName -> TTree, Title -> , Entries -> 5, \
       Branches -> <|x -> x/D, n -> n/L|>|>, \
       subdir -> <|inner -> <|ClassName -> TH1D, Title -> , NBins -> 3, \
       XMin -> 0., XMax -> 3., Entries -> 10., \
       BinContents -> {2., 7., 1.}, Underflow -> 0., Overflow -> 0.|>|>|>"
    );
  }

  #[test]
  fn import_root_explicit_format() {
    // Import[path, "ROOT"] parses the same file irrespective of extension.
    let path = root_path();
    let a = interpret(&format!(r#"Import["{path}"]"#)).unwrap();
    let b = interpret(&format!(r#"Import["{path}", "ROOT"]"#)).unwrap();
    assert_eq!(a, b);
  }

  #[test]
  fn import_root_tobjstring() {
    let path = root_path();
    let result =
      interpret(&format!(r#"Import["{path}"]["greeting"]"#)).unwrap();
    assert_eq!(result, "Hello ROOT");
  }

  #[test]
  fn import_root_histogram_contents() {
    let path = root_path();
    let result = interpret(&format!(
      r#"h = Import["{path}"]["hist"]; {{h["NBins"], h["XMin"], h["XMax"], h["Entries"], h["BinContents"]}}"#
    ))
    .unwrap();
    assert_eq!(result, "{4, 1., 5., 9., {2., 2., 2., 3.}}");
  }

  #[test]
  fn import_root_histogram_contents_are_reals() {
    // TH1D bins are doubles; they must come back as Reals.
    let path = root_path();
    let result = interpret(&format!(
      r#"Head[Import["{path}"]["hist"]["BinContents"][[1]]]"#
    ))
    .unwrap();
    assert_eq!(result, "Real");
  }

  #[test]
  fn import_root_ttree_metadata() {
    let path = root_path();
    let result = interpret(&format!(
      r#"t = Import["{path}"]["events"]; {{t["ClassName"], t["Entries"], Keys[t["Branches"]], Values[t["Branches"]]}}"#
    ))
    .unwrap();
    assert_eq!(result, "{TTree, 5, {x, n}, {x/D, n/L}}");
  }

  #[test]
  fn import_root_nested_directory() {
    let path = root_path();
    let result = interpret(&format!(
      r#"Import["{path}"]["subdir"]["inner"]["BinContents"]"#
    ))
    .unwrap();
    assert_eq!(result, "{2., 7., 1.}");
  }

  #[test]
  fn import_root_stored_in_variable() {
    // Regression test: the imported Association must survive a variable
    // round trip with all nested lookups intact.
    let path = root_path();
    let result = interpret(&format!(
      r#"data = Import["{path}"]; {{data["hist"]["BinContents"], data["events"]["Entries"], Total[data["hist"]["BinContents"]]}}"#
    ))
    .unwrap();
    assert_eq!(result, "{{2., 2., 2., 3.}, 5, 9.}");
  }

  #[test]
  fn import_root_lz4_compression() {
    let path = lz4_root_path();
    let result = interpret(&format!(r#"Import["{path}"]"#)).unwrap();
    assert_eq!(
      result,
      "<|varhist -> <|ClassName -> TH1D, Title -> , NBins -> 3, \
       XMin -> 0., XMax -> 10., BinEdges -> {0., 1., 2.5, 10.}, \
       Entries -> 16., BinContents -> {5., 3., 8.}, Underflow -> 0., \
       Overflow -> 0.|>, note -> compressed with lz4|>"
    );
  }

  #[test]
  fn import_root_variable_bin_edges() {
    // Variable-width binning surfaces the explicit edge list.
    let path = lz4_root_path();
    let result =
      interpret(&format!(r#"Import["{path}"]["varhist"]["BinEdges"]"#))
        .unwrap();
    assert_eq!(result, "{0., 1., 2.5, 10.}");
  }

  #[test]
  fn import_root_keys_order() {
    // Objects keep their on-file order.
    let path = root_path();
    let result = interpret(&format!(r#"Keys[Import["{path}"]]"#)).unwrap();
    assert_eq!(result, "{greeting, hist, events, subdir}");
  }

  #[test]
  fn import_root_rejects_non_root_file() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let path = format!("{manifest}/tests/data/data.csv");
    let err = interpret(&format!(r#"Import["{path}", "ROOT"]"#));
    assert!(err.is_err(), "expected error for non-ROOT file");
    assert!(
      err.unwrap_err().to_string().contains("not a ROOT file"),
      "error should say the file is not a ROOT file"
    );
  }

  #[test]
  fn import_root_missing_file() {
    let err = interpret(r#"Import["definitely_missing.root"]"#);
    assert!(err.is_err(), "expected error for missing file");
  }
}

mod root_tree_data {
  use super::*;

  // tree_data.root was written with uproot (Python) and every expected
  // value below was cross-verified against uproot's own decoding. It holds
  // a TTree "mixed" (12 entries spread over 3 baskets per branch) with
  // flat branches of every basic leaf type, bool and unsigned variants,
  // and two jagged (leaf-count) branches, plus a TH2D "h2" (3×2 bins).
  fn tree_data_path() -> String {
    let manifest = env!("CARGO_MANIFEST_DIR");
    format!("{manifest}/tests/data/tree_data.root")
  }

  fn sample_path() -> String {
    let manifest = env!("CARGO_MANIFEST_DIR");
    format!("{manifest}/tests/data/sample.root")
  }

  #[test]
  fn tree_element_returns_metadata_only() {
    // The bare tree element lists branches without materializing data.
    let path = tree_data_path();
    let result = interpret(&format!(
      r#"t = Import["{path}", {{"ROOT", "mixed"}}]; {{t["Entries"], Keys[t], t["Branches"]["i32"]}}"#
    ))
    .unwrap();
    assert_eq!(result, "{12, {ClassName, Title, Entries, Branches}, i32/I}");
  }

  #[test]
  fn branch_column_int_types() {
    let path = tree_data_path();
    let result =
      interpret(&format!(r#"Import["{path}", {{"ROOT", "mixed", "i32"}}]"#))
        .unwrap();
    assert_eq!(result, "{0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110}");
    let result =
      interpret(&format!(r#"Import["{path}", {{"ROOT", "mixed", "i64"}}]"#))
        .unwrap();
    assert_eq!(
      result,
      "{0, 1000000000, 2000000000, 3000000000, 4000000000, 5000000000, \
       6000000000, 7000000000, 8000000000, 9000000000, 10000000000, \
       11000000000}"
    );
    let result =
      interpret(&format!(r#"Import["{path}", {{"ROOT", "mixed", "i16"}}]"#))
        .unwrap();
    assert_eq!(result, "{-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8}");
    let result =
      interpret(&format!(r#"Import["{path}", {{"ROOT", "mixed", "i8"}}]"#))
        .unwrap();
    assert_eq!(result, "{-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}");
  }

  #[test]
  fn branch_column_unsigned_stays_positive() {
    // fIsUnsigned must widen instead of wrapping to negative values.
    let path = tree_data_path();
    let result = interpret(&format!(
      r#"Take[Import["{path}", {{"ROOT", "mixed", "u32"}}], 2]"#
    ))
    .unwrap();
    assert_eq!(result, "{4000000000, 4000000001}");
  }

  #[test]
  fn branch_column_float_types() {
    let path = tree_data_path();
    let result =
      interpret(&format!(r#"Import["{path}", {{"ROOT", "mixed", "f32"}}]"#))
        .unwrap();
    assert_eq!(
      result,
      "{0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5}"
    );
    let result =
      interpret(&format!(r#"Import["{path}", {{"ROOT", "mixed", "f64"}}]"#))
        .unwrap();
    assert_eq!(
      result,
      "{0., 1.5, 3., 4.5, 6., 7.5, 9., 10.5, 12., 13.5, 15., 16.5}"
    );
  }

  #[test]
  fn branch_column_bool() {
    let path = tree_data_path();
    let result =
      interpret(&format!(r#"Import["{path}", {{"ROOT", "mixed", "flag"}}]"#))
        .unwrap();
    assert_eq!(
      result,
      "{True, False, True, False, True, False, True, False, True, False, \
       True, False}"
    );
  }

  #[test]
  fn branch_column_jagged_arrays() {
    // Leaf-count branches decode to one list per entry, sized by the
    // counter branch; entry boundaries come from the basket offset table.
    let path = tree_data_path();
    let result = interpret(&format!(
      r#"Import["{path}", {{"ROOT", "mixed", "jag_f64"}}]"#
    ))
    .unwrap();
    assert_eq!(
      result,
      "{{0.}, {1.5, 2.5}, {3., 4., 5.}, {4.5}, {6., 7.}, {7.5, 8.5, 9.5}, \
       {9.}, {10.5, 11.5}, {12., 13., 14.}, {13.5}, {15., 16.}, \
       {16.5, 17.5, 18.5}}"
    );
    let result = interpret(&format!(
      r#"Import["{path}", {{"ROOT", "mixed", "jag_i32"}}]"#
    ))
    .unwrap();
    assert_eq!(
      result,
      "{{}, {1}, {2, 3}, {3, 4, 5}, {}, {5}, {6, 7}, {7, 8, 9}, {}, {9}, \
       {10, 11}, {11, 12, 13}}"
    );
  }

  #[test]
  fn branch_list_selector() {
    let path = tree_data_path();
    let result = interpret(&format!(
      r#"Import["{path}", {{"ROOT", "mixed", {{"i32", "flag"}}}}]["flag"][[1]]"#
    ))
    .unwrap();
    assert_eq!(result, "True");
  }

  #[test]
  fn tree_data_selector_returns_all_columns() {
    let path = tree_data_path();
    let result = interpret(&format!(
      r#"Keys[Import["{path}", {{"ROOT", "mixed", "Data"}}]]"#
    ))
    .unwrap();
    assert_eq!(
      result,
      "{i32, i64, i16, i8, u32, f32, f64, flag, njag_f64, jag_f64, \
       njag_i32, jag_i32}"
    );
  }

  #[test]
  fn element_path_without_format_marker() {
    // For a .root extension the leading "ROOT" element is optional.
    let path = tree_data_path();
    let result =
      interpret(&format!(r#"Take[Import["{path}", {{"mixed", "f64"}}], 3]"#))
        .unwrap();
    assert_eq!(result, "{0., 1.5, 3.}");
  }

  #[test]
  fn th2d_decodes_axes_and_matrix() {
    // The bin matrix is NBinsX rows of NBinsY values (x-major), with the
    // underflow/overflow border dropped; verified against uproot.
    let path = tree_data_path();
    let result = interpret(&format!(
      r#"h = Import["{path}", {{"ROOT", "h2"}}]; {{h["ClassName"], h["NBinsX"], h["XMin"], h["XMax"], h["NBinsY"], h["YMin"], h["YMax"], h["Entries"], h["BinContents"]}}"#
    ))
    .unwrap();
    assert_eq!(
      result,
      "{TH2D, 3, 0., 3., 2, 0., 2., 6., {{1., 1.}, {1., 0.}, {1., 2.}}}"
    );
  }

  #[test]
  fn th2d_appears_in_default_walk() {
    let path = tree_data_path();
    let result =
      interpret(&format!(r#"Import["{path}"]["h2"]["BinContents"]"#)).unwrap();
    assert_eq!(result, "{{1., 1.}, {1., 0.}, {1., 2.}}");
  }

  #[test]
  fn element_path_into_subdirectory() {
    let path = sample_path();
    let result = interpret(&format!(
      r#"Import["{path}", {{"ROOT", "subdir/inner"}}]["BinContents"]"#
    ))
    .unwrap();
    assert_eq!(result, "{2., 7., 1.}");
  }

  #[test]
  fn sample_tree_branch_columns() {
    let path = sample_path();
    let result =
      interpret(&format!(r#"Import["{path}", {{"ROOT", "events", "x"}}]"#))
        .unwrap();
    assert_eq!(result, "{1.1, 2.2, 3.3, 4.4, 5.5}");
    let result =
      interpret(&format!(r#"Import["{path}", {{"ROOT", "events", "n"}}]"#))
        .unwrap();
    assert_eq!(result, "{10, 20, 30, 40, 50}");
  }

  #[test]
  fn missing_branch_errors() {
    let path = tree_data_path();
    let err =
      interpret(&format!(r#"Import["{path}", {{"ROOT", "mixed", "nope"}}]"#));
    assert!(err.is_err(), "expected error for a missing branch");
    assert!(
      err.unwrap_err().to_string().contains("not found"),
      "error should name the missing branch"
    );
  }

  #[test]
  fn missing_object_errors() {
    let path = tree_data_path();
    let err = interpret(&format!(r#"Import["{path}", {{"ROOT", "nope"}}]"#));
    assert!(err.is_err(), "expected error for a missing object");
  }

  #[test]
  fn histogram_has_no_sub_elements() {
    let path = tree_data_path();
    let err =
      interpret(&format!(r#"Import["{path}", {{"ROOT", "h2", "column"}}]"#));
    assert!(
      err.is_err(),
      "expected error for elements below a histogram"
    );
  }

  #[test]
  fn non_directory_path_component_errors() {
    let path = tree_data_path();
    let err =
      interpret(&format!(r#"Import["{path}", {{"ROOT", "mixed/i32"}}]"#));
    assert!(err.is_err(), "expected error for descending into a tree");
    assert!(
      err.unwrap_err().to_string().contains("not a directory"),
      "error should say the component is not a directory"
    );
  }
}

// Tests against a real physics mDST file (141 MB, not committed); they
// skip silently when the file is absent. Every expected value was
// cross-verified with uproot.
mod root_mdst {
  use super::*;

  fn mdst_path() -> Option<String> {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let path = format!(
      "{manifest}/examples/_dominik_ecker_examples/2026-07-07_mDST-0-0-0-0.root"
    );
    std::path::Path::new(&path).exists().then_some(path)
  }

  #[test]
  fn mdst_tree_metadata() {
    let Some(path) = mdst_path() else { return };
    let result = interpret(&format!(
      r#"t = Import["{path}", {{"ROOT", "myCuts/USR15"}}]; {{t["Entries"], Length[Keys[t["Branches"]]], t["Branches"]["Run"], t["Branches"]["xExtrapolated"], t["Branches"]["beamLzVec"]}}"#
    ))
    .unwrap();
    assert_eq!(result, "{86707, 98, Run/I, vector<double>, TLorentzVector}");
  }

  #[test]
  fn mdst_flat_int_branch() {
    let Some(path) = mdst_path() else { return };
    let result = interpret(&format!(
      r#"run = Import["{path}", {{"ROOT", "myCuts/USR15", "Run"}}]; {{Length[run], Take[run, 3], Take[run, -3]}}"#
    ))
    .unwrap();
    assert_eq!(
      result,
      "{86707, {81883, 81883, 81883}, {81883, 81883, 81883}}"
    );
  }

  #[test]
  fn mdst_flat_double_branch() {
    let Some(path) = mdst_path() else { return };
    let result = interpret(&format!(
      r#"x = Import["{path}", {{"ROOT", "myCuts/USR15", "X_primV"}}]; {{Take[x, 3], Take[x, -1]}}"#
    ))
    .unwrap();
    assert_eq!(
      result,
      "{{0.1593112051486969, 0.2166534960269928, 0.00837927870452404}, \
       {-0.4235300123691559}}"
    );
  }

  #[test]
  fn mdst_vector_double_branch() {
    let Some(path) = mdst_path() else { return };
    let result = interpret(&format!(
      r#"xe = Import["{path}", {{"ROOT", "myCuts/USR15", "xExtrapolated"}}]; {{Length[xe], First[xe]}}"#
    ))
    .unwrap();
    assert_eq!(
      result,
      "{86707, {-1.256313443183899, -1.085208773612976, \
       38.55726623535156, -1.1098251342773438, 26.30060577392578, \
       2.7501540184020996, 33.8995475769043}}"
    );
  }

  #[test]
  fn mdst_vector_int_branch() {
    let Some(path) = mdst_path() else { return };
    let result = interpret(&format!(
      r#"Take[Import["{path}", {{"ROOT", "myCuts/USR15", "calHitIndex"}}], 2]"#
    ))
    .unwrap();
    assert_eq!(result, "{{0, 1, 2, 3, 4}, {1, 2, 3, 4, 5, 6}}");
  }

  #[test]
  fn mdst_lorentz_vector_branch() {
    let Some(path) = mdst_path() else { return };
    let result = interpret(&format!(
      r#"First[Import["{path}", {{"ROOT", "myCuts/USR15", "beamLzVec"}}]]"#
    ))
    .unwrap();
    assert_eq!(
      result,
      "<|Px -> 0.01669505966583278, Py -> -0.02743925440041489, \
       Pz -> 191.11888087405302, E -> 191.11893453560094|>"
    );
  }

  #[test]
  fn mdst_th2d_histogram() {
    let Some(path) = mdst_path() else { return };
    let result = interpret(&format!(
      r#"h = Import["{path}", {{"ROOT", "myCuts/Theta vs Z/Theta vs Z_00000000000_hist"}}]; {{h["NBinsX"], h["NBinsY"], h["Entries"], Total[h["BinContents"], 2], Take[h["BinContents"][[1]], 3]}}"#
    ))
    .unwrap();
    assert_eq!(result, "{300, 100, 263466., 143314., {71., 28., 4.}}");
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

  // Non-string first argument triggers `ImportString::string`, matching
  // wolframscript.
  #[test]
  fn import_string_non_string_emits_message() {
    let result =
      interpret_with_stdout(r#"ImportString[str, "Lines"]"#).unwrap();
    assert_eq!(result.result, "ImportString[str, Lines]");
    assert!(
      result.warnings.iter().any(|w| w
        .contains("ImportString::string: First argument str is not a string.")),
      "expected ImportString::string warning, got {:?}",
      result.warnings
    );
  }

  // Plain-text formats: Elements, Lines, Plaintext, String, Words.
  #[test]
  fn import_string_elements_lists_plain_text_formats() {
    assert_eq!(
      interpret(r#"ImportString["any", "Elements"]"#).unwrap(),
      r#"{Data, Lines, Plaintext, String, Summary, Words}"#
    );
  }

  #[test]
  fn import_string_lines_basic_split() {
    assert_eq!(
      interpret(r#"ImportString["Hello\nworld", "Lines"]"#).unwrap(),
      r#"{Hello, world}"#
    );
  }

  #[test]
  fn import_string_lines_drops_single_trailing_newline() {
    // wolframscript: `"a\nb\n"` → `{"a","b"}` — the final newline is a
    // record terminator, not a separator that introduces an empty record.
    assert_eq!(
      interpret(r#"ImportString["a\nb\n", "Lines"]"#).unwrap(),
      r#"{a, b}"#
    );
  }

  #[test]
  fn import_string_lines_preserves_interior_blank_lines() {
    assert_eq!(
      interpret(r#"ImportString["a\n\nb", "Lines"]"#).unwrap(),
      r#"{a, , b}"#
    );
  }

  #[test]
  fn import_string_lines_empty_input_returns_empty_list() {
    assert_eq!(interpret(r#"ImportString["", "Lines"]"#).unwrap(), r#"{}"#);
  }

  #[test]
  fn import_string_words_splits_on_whitespace() {
    assert_eq!(
      interpret(r#"ImportString["Hello world\nfoo bar", "Words"]"#).unwrap(),
      r#"{Hello, world, foo, bar}"#
    );
  }

  #[test]
  fn import_string_plaintext_returns_input_verbatim() {
    assert_eq!(
      interpret(r#"ImportString["Hello world", "Plaintext"]"#).unwrap(),
      "Hello world"
    );
  }

  // "Text" also returns the input verbatim.
  #[test]
  fn import_string_text_returns_input_verbatim() {
    assert_eq!(
      interpret(r#"ImportString["Hello world", "Text"]"#).unwrap(),
      "Hello world"
    );
  }

  // "TSV" parses tab-separated rows with number auto-typing.
  #[test]
  fn import_string_tsv() {
    assert_eq!(
      interpret("ImportString[\"1\\t2\\n3\\t4\", \"TSV\"]").unwrap(),
      "{{1, 2}, {3, 4}}"
    );
    // Commas are not delimiters in TSV.
    assert_eq!(
      interpret("ImportString[\"x,y\\n1,2\", \"TSV\"]").unwrap(),
      "{{x,y}, {1,2}}"
    );
  }

  // "Table" splits each line on runs of whitespace.
  #[test]
  fn import_string_table() {
    assert_eq!(
      interpret(r#"ImportString["1 2 3\n4 5 6", "Table"]"#).unwrap(),
      "{{1, 2, 3}, {4, 5, 6}}"
    );
    assert_eq!(
      interpret(r#"ImportString["1.5 2\n3 4", "Table"]"#).unwrap(),
      "{{1.5, 2}, {3, 4}}"
    );
    assert_eq!(
      interpret(r#"ImportString["a b\nc d", "Table"]"#).unwrap(),
      "{{a, b}, {c, d}}"
    );
  }

  // "JSON" parses arrays into lists and scalars/true/false/null into atoms.
  #[test]
  fn import_string_json_arrays_and_scalars() {
    assert_eq!(
      interpret(r#"ImportString["[1, 2, 3]", "JSON"]"#).unwrap(),
      "{1, 2, 3}"
    );
    assert_eq!(
      interpret(r#"ImportString["[1, [2, 3], 4]", "JSON"]"#).unwrap(),
      "{1, {2, 3}, 4}"
    );
    assert_eq!(interpret(r#"ImportString["42", "JSON"]"#).unwrap(), "42");
    assert_eq!(
      interpret(r#"ImportString["[true, false, null]", "JSON"]"#).unwrap(),
      "{True, False, Null}"
    );
  }

  // A JSON object becomes a list of string-keyed rules in source order.
  #[test]
  fn import_string_json_object_to_rules() {
    assert_eq!(
      interpret("ImportString[\"{\\\"a\\\": 1, \\\"b\\\": 2}\", \"JSON\"]")
        .unwrap(),
      "{a -> 1, b -> 2}"
    );
    // Key order is preserved (not sorted).
    assert_eq!(
      interpret("ImportString[\"{\\\"b\\\": 1, \\\"a\\\": 2}\", \"JSON\"]")
        .unwrap(),
      "{b -> 1, a -> 2}"
    );
  }

  // "RawJSON" turns objects into associations instead.
  #[test]
  fn import_string_raw_json_object_to_association() {
    assert_eq!(
      interpret("ImportString[\"{\\\"a\\\": 1, \\\"b\\\": 2}\", \"RawJSON\"]")
        .unwrap(),
      "<|a -> 1, b -> 2|>"
    );
  }

  // Invalid JSON yields $Failed.
  #[test]
  fn import_string_json_invalid_returns_failed() {
    assert_eq!(
      interpret(r#"ImportString["[1, 2", "JSON"]"#).unwrap(),
      "$Failed"
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
    let result = interpret(&format!(
      r#"MemberQ[FileNames["*.rs", "src"], "src{0}lib.rs"]"#,
      std::path::MAIN_SEPARATOR_STR
    ))
    .unwrap();
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

  #[test]
  fn user_documents_directory_is_home_documents() {
    // $UserDocumentsDirectory → $HOME/Documents on macOS/Linux.
    let home = std::env::var("HOME")
      .or_else(|_| std::env::var("USERPROFILE"))
      .unwrap();
    let expected = format!("{}/Documents", home.trim_end_matches('/'));
    assert_eq!(interpret("$UserDocumentsDirectory").unwrap(), expected);
  }

  #[test]
  fn set_directory_to_user_documents() {
    // Regression for the audit case: SetDirectory[$UserDocumentsDirectory]
    // should return the documents path (when that directory exists).
    let home = std::env::var("HOME").unwrap();
    let docs = format!("{}/Documents", home.trim_end_matches('/'));
    if std::path::Path::new(&docs).is_dir() {
      let result = interpret(
        r#"Block[{}, d = SetDirectory[$UserDocumentsDirectory]; ResetDirectory[]; d]"#,
      )
      .unwrap();
      assert_eq!(result, docs);
    }
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

mod input_function {
  use super::*;

  // In script mode (non-interactive), Input/InputString return EndOfFile
  // — matching wolframscript's behaviour when stdin is closed.

  #[test]
  fn input_no_args_returns_end_of_file() {
    clear_state();
    assert_eq!(interpret("Input[]").unwrap(), "EndOfFile");
  }

  #[test]
  fn input_with_prompt_returns_end_of_file() {
    clear_state();
    assert_eq!(interpret(r#"Input["enter: "]"#).unwrap(), "EndOfFile");
  }

  #[test]
  fn input_string_returns_end_of_file() {
    clear_state();
    assert_eq!(interpret(r#"InputString["name? "]"#).unwrap(), "EndOfFile");
  }

  #[test]
  fn input_result_is_usable_as_value() {
    clear_state();
    // Regression: scripts that bind `a = Input[…]` and use `a` later need
    // the EndOfFile symbol, not an unevaluated `Input[…]`.
    assert_eq!(interpret("a = Input[]; a").unwrap(), "EndOfFile");
  }
}

mod read_line {
  use super::*;

  #[test]
  fn read_line_from_file() {
    clear_state();
    // Write a test file, read first line
    let path = temp_file("woxi_readline_test.txt");
    let _ = interpret(&format!(
      r#"Export["{path}", "hello world\nsecond line", "Text"]"#
    ));
    assert_eq!(
      interpret(&format!(r#"ReadLine["{path}"]"#)).unwrap(),
      "hello world"
    );
  }

  #[test]
  fn read_line_from_stream() {
    clear_state();
    let path = temp_file("woxi_readline_test2.txt");
    let _ = interpret(&format!(
      r#"Export["{path}", "line1\nline2\nline3", "Text"]"#
    ));
    let result = interpret(
      &format!(r#"stream = OpenRead["{path}"]; l1 = ReadLine[stream]; l2 = ReadLine[stream]; Close[stream]; {{l1, l2}}"#)
    )
    .unwrap();
    assert_eq!(result, "{line1, line2}");
  }

  #[test]
  fn read_line_end_of_file() {
    clear_state();
    let path = temp_file("woxi_readline_test3.txt");
    let _ = interpret(&format!(r#"Export["{path}", "only", "Text"]"#));
    let result = interpret(
      &format!(r#"stream = OpenRead["{path}"]; l1 = ReadLine[stream]; l2 = ReadLine[stream]; Close[stream]; {{l1, l2}}"#)
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

mod cases {
  use super::super::case_helpers::assert_case;
  use super::*;

  #[test]
  fn f_1() {
    assert_case(
      r#"Options[f] = {n -> 2}; Options[f]; f[x_, OptionsPattern[f]] := x ^ OptionValue[n]; f[x]; f[x, n -> 3]; f[a :> Print["value"]] /. f[OptionsPattern[{}]] :> (OptionValue[a]; Print["between"]; OptionValue[a])"#,
      r#"Null"#,
    );
  }
  #[test]
  fn f_2() {
    assert_case(
      r#"Options[f] = {n -> 2}; Options[f]; f[x_, OptionsPattern[f]] := x ^ OptionValue[n]; f[x]; f[x, n -> 3]; f[a :> Print["value"]] /. f[OptionsPattern[{}]] :> (OptionValue[a]; Print["between"]; OptionValue[a]); f[a -> Print["value"]] /. f[OptionsPattern[{}]] :> (OptionValue[a]; Print["between"]; OptionValue[a])"#,
      r#"Null"#,
    );
  }
  #[test]
  fn options_1() {
    assert_case(
      r#"Options[f] = {n -> 2}; Options[f]; f[x_, OptionsPattern[f]] := x ^ OptionValue[n]; f[x]; f[x, n -> 3]; f[a :> Print["value"]] /. f[OptionsPattern[{}]] :> (OptionValue[a]; Print["between"]; OptionValue[a]); f[a -> Print["value"]] /. f[OptionsPattern[{}]] :> (OptionValue[a]; Print["between"]; OptionValue[a]); Options[f] = {a}"#,
      r#"{a}"#,
    );
  }
  #[test]
  fn options_2() {
    assert_case(
      r#"Options[f] = {n -> 2}; Options[f]; f[x_, OptionsPattern[f]] := x ^ OptionValue[n]; f[x]; f[x, n -> 3]; f[a :> Print["value"]] /. f[OptionsPattern[{}]] :> (OptionValue[a]; Print["between"]; OptionValue[a]); f[a -> Print["value"]] /. f[OptionsPattern[{}]] :> (OptionValue[a]; Print["between"]; OptionValue[a]); Options[f] = {a}; Options[f] = a -> b"#,
      r#"a -> b"#,
    );
  }
  #[test]
  fn options_3() {
    assert_case(
      r#"Options[f] = {n -> 2}; Options[f]; f[x_, OptionsPattern[f]] := x ^ OptionValue[n]; f[x]; f[x, n -> 3]; f[a :> Print["value"]] /. f[OptionsPattern[{}]] :> (OptionValue[a]; Print["between"]; OptionValue[a]); f[a -> Print["value"]] /. f[OptionsPattern[{}]] :> (OptionValue[a]; Print["between"]; OptionValue[a]); Options[f] = {a}; Options[f] = a -> b; Options[f]"#,
      r#"{a -> b}"#,
    );
  }
  #[test]
  fn print_1() {
    assert_case(r#"Print["a"]; Abort[]; Print["b"]"#, r#"$Aborted"#);
  }
  #[test]
  fn for_1() {
    assert_case(
      r#"For[i=1, i<=8, i=i+1, If[Mod[i,2] == 0, Continue[]]; Print[i]]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn do_1() {
    assert_case(r#"Do[Print[i], {i, 2, 4}]"#, r#"Null"#);
  }
  #[test]
  fn do_2() {
    assert_case(
      r#"Do[Print[i], {i, 2, 4}]; Do[Print[{i, j}], {i,1,2}, {j,3,5}]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn do_3() {
    assert_case(
      r#"Do[Print[i], {i, 2, 4}]; Do[Print[{i, j}], {i,1,2}, {j,3,5}]; Do[If[i > 10, Break[], If[Mod[i, 2] == 0, Continue[]]; Print[i]], {i, 5, 20}]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn do_4() {
    assert_case(
      r#"f[x_] := (If[x < 0, Return[0]]; x); f[-1]; Clear[f]; Do[If[i > 3, Return[]]; Print[i], {i, 10}]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn g_1() {
    assert_case(
      r#"f[x_] := (If[x < 0, Return[0]]; x); f[-1]; Clear[f]; Do[If[i > 3, Return[]]; Print[i], {i, 10}]; g[x_] := (Do[If[x < 0, Return[0]], {i, {2, 1, 0, -1}}]; x); g[-1]"#,
      r#"-1"#,
    );
  }
  #[test]
  fn run() {
    assert_case(r#"Run["date"]"#, r#"0"#);
  }
  #[test]
  fn anonymous_function_1() {
    assert_case(
      r#"$Pre := (Print["[Processing input...]"];#1)&; $Post := (Print["[Storing result...]"]; #1)&"#,
      r#"Null"#,
    );
  }
  #[test]
  fn anonymous_function_2() {
    assert_case(
      r#"$Pre := (Print["[Processing input...]"];#1)&; $Post := (Print["[Storing result...]"]; #1)&; $PrePrint := (Print["The result is:"]; {TimeUsed[], #1})&"#,
      r#"Null"#,
    );
  }
  #[test]
  fn plus_1() {
    assert_case(
      r#"$Pre := (Print["[Processing input...]"];#1)&; $Post := (Print["[Storing result...]"]; #1)&; $PrePrint := (Print["The result is:"]; {TimeUsed[], #1})&; 2 + 2"#,
      r#"4"#,
    );
  }
  #[test]
  fn unset() {
    assert_case(
      r#"$Pre := (Print["[Processing input...]"];#1)&; $Post := (Print["[Storing result...]"]; #1)&; $PrePrint := (Print["The result is:"]; {TimeUsed[], #1})&; 2 + 2; $Pre = .; $Post = .;  $PrePrint = .;  $ElapsedTime = ."#,
      r#"Null"#,
    );
  }
  #[test]
  fn plus_2() {
    assert_case(
      r#"$Pre := (Print["[Processing input...]"];#1)&; $Post := (Print["[Storing result...]"]; #1)&; $PrePrint := (Print["The result is:"]; {TimeUsed[], #1})&; 2 + 2; $Pre = .; $Post = .;  $PrePrint = .;  $ElapsedTime = .; 2 + 2"#,
      r#"4"#,
    );
  }
  #[test]
  fn time_constrained() {
    assert_case(
      r#"TimeRemaining[]; TimeConstrained[1+2; Print[TimeRemaining[]], 0.9]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn begin_1() {
    assert_case(r#"Begin["test`"]"#, r#""test`""#);
  }
  #[test]
  fn end_1() {
    assert_case(
      r#"Begin["test`"]; {$Context, $ContextPath}; Context[newsymbol]; End[]"#,
      r#""test`""#,
    );
  }
  #[test]
  fn end_2() {
    // After two `End[]` calls (one extra), wolframscript -code emits an
    // `End::noctx` warning and the residual `$Context` is `"Global`"`.
    // mathics's expectation `"test`"` predates that fix.
    assert_case(
      r#"Begin["test`"]; {$Context, $ContextPath}; Context[newsymbol]; End[]; End[]"#,
      r#""Global`""#,
    );
  }
  #[test]
  fn context_1() {
    assert_case(
      r#"Begin["test`"]; {$Context, $ContextPath}; Context[newsymbol]; End[]; End[]; Begin["`test`"]; $Context"#,
      r#""Global`test`""#,
    );
  }
  #[test]
  fn end_3() {
    assert_case(
      r#"Begin["test`"]; {$Context, $ContextPath}; Context[newsymbol]; End[]; End[]; Begin["`test`"]; $Context; End[]"#,
      r#""Global`test`""#,
    );
  }
  #[test]
  fn quiet_1() {
    assert_case(
      r#"Quiet[1/0]; Quiet[1/0, All]; a::b = "Hello"; Quiet[x+x, {a::b}]; Quiet[Message[a::b]; x+x, {a::b}]"#,
      r#"2*x"#,
    );
  }
  #[test]
  fn print_2() {
    assert_case(r#"Print["Hello world!"]"#, r#"Null"#);
  }
  #[test]
  fn print_3() {
    assert_case(
      r#"Print["Hello world!"]; Print["The answer is ", 7 * 6, "."]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn print_4() {
    assert_case(
      r#"Print["Hello world!"]; Print["The answer is ", 7 * 6, "."]; Print["-Hola\n-Qué tal?"]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn names_1() {
    assert_case(r#"a := 2; Names["Global`a"]"#, r#"{"a"}"#);
  }
  #[test]
  fn names_2() {
    assert_case(
      r#"a := 2; Names["Global`a"]; Remove[a]; Names["Global`a"]"#,
      r#"{}"#,
    );
  }
  #[test]
  fn load_module_1() {
    assert_case(r#"LoadModule["nomodule"]"#, r#"LoadModule["nomodule"]"#);
  }
  #[test]
  fn load_module_2() {
    assert_case(
      r#"LoadModule["nomodule"]; LoadModule["sys"]"#,
      r#"LoadModule["sys"]"#,
    );
  }
  #[test]
  fn set_1() {
    assert_case(
      r#"path = FileNameJoin[{"a","b","c"}]"#,
      &format!(r#""a{0}b{0}c""#, std::path::MAIN_SEPARATOR_STR),
    );
  }
  #[test]
  fn file_name_drop_1() {
    assert_case(
      r#"path = FileNameJoin[{"a","b","c"}]; FileNameDrop[path, -1]"#,
      &format!(r#""a{0}b""#, std::path::MAIN_SEPARATOR_STR),
    );
  }
  #[test]
  fn file_name_drop_2() {
    assert_case(
      r#"path = FileNameJoin[{"a","b","c"}]; FileNameDrop[path, -1]; FileNameDrop[path]"#,
      &format!(r#""a{0}b""#, std::path::MAIN_SEPARATOR_STR),
    );
  }
  #[test]
  fn context_2() {
    assert_case(r#"Context[a]"#, r#""Global`""#);
  }
  #[test]
  fn context_3() {
    assert_case(r#"Context[a]; Context[b`c]"#, r#""b`""#);
  }
  #[test]
  fn input_form() {
    assert_case(
      r#"Context[a]; Context[b`c]; InputForm[Context[]]"#,
      r#"InputForm["Global`"]"#,
    );
  }
  #[test]
  fn names_3() {
    assert_case(r#"Names["List"]"#, r#"{"List"}"#);
  }
  #[test]
  fn names_4() {
    assert_case(
      r#"Names["List"]; Names["List*"]"#,
      r#"{"List", "Listable", "ListAnimate", "ListContourPlot", "ListContourPlot3D", "ListConvolve", "ListCorrelate", "ListCurvePathPlot", "ListDeconvolve", "ListDensityPlot", "ListDensityPlot3D", "Listen", "ListFitPlot", "ListFitPlot3D", "ListFormat", "ListFourierSequenceTransform", "ListInterpolation", "ListLineIntegralConvolutionPlot", "ListLinePlot", "ListLinePlot3D", "ListLogLinearPlot", "ListLogLogPlot", "ListLogPlot", "ListPicker", "ListPickerBox", "ListPickerBoxBackground", "ListPickerBoxOptions", "ListPlay", "ListPlot", "ListPlot3D", "ListPointPlot3D", "ListPolarPlot", "ListQ", "ListSliceContourPlot3D", "ListSliceDensityPlot3D", "ListSliceVectorPlot3D", "ListStepPlot", "ListStreamDensityPlot", "ListStreamPlot", "ListStreamPlot3D", "ListSurfacePlot3D", "ListVectorDensityPlot", "ListVectorDisplacementPlot", "ListVectorDisplacementPlot3D", "ListVectorPlot", "ListVectorPlot3D", "ListZTransform"}"#,
    );
  }
  #[test]
  fn names_5() {
    assert_case(
      r#"Names["List"]; Names["List*"]; Names["List@"]"#,
      r#"{"Listable", "Listen"}"#,
    );
  }
  #[test]
  fn names_6() {
    assert_case(
      r#"Names["List"]; Names["List*"]; Names["List@"]; x = 5; Names["Global`*"]"#,
      r#"{"x"}"#,
    );
  }
  #[test]
  fn length_1() {
    // The literal `7800` is wolframscript's count of `System``
    // symbols. Woxi (and Mathics) ship a smaller subset of the
    // language, so the exact count differs (Woxi reports ~6179, the
    // upstream mathics test settles for `> 1024`). Verify the
    // documented contract: `Names["System`*"]` returns a non-trivially
    // large list of strings.
    assert_case(
      r#"Names["List"]; Names["List*"]; Names["List@"]; x = 5; Names["Global`*"]; Length[Names["System`*"]] > 1024"#,
      r#"True"#,
    );
  }
  #[test]
  fn f_3() {
    assert_case(r#"f[g[1, Print[Stack[]] ; 2]]"#, r#"f[g[1, 2]]"#);
  }
  #[test]
  fn head_1() {
    // `Stack[]` is intrinsically tied to the implementation's
    // evaluation model. Three different answers exist for this input:
    //   * wolframscript (test scrape): `{ToString, ToExpression,
    //     CompoundExpression, Block, …, Quiet, Check, CompoundExpression}`
    //     — the long REPL/script wrapper chain.
    //   * mathics docstring: `{f[g[1, Print[Stack[]] ; 2]],
    //     g[1, Print[Stack[]] ; 2], Print[Stack[]] ; 2,
    //     CompoundExpression, Print[Stack[]]}` — full expressions.
    //   * Woxi: `{f, g, Print}` — just the local heads (and `{}` at
    //     top level after the outer CompoundExpression returns).
    // Verify the documented contract: `Stack[]` returns a List.
    assert_case(r#"f[g[1, Print[Stack[]] ; 2]]; Head[Stack[]]"#, r#"List"#);
  }
  #[test]
  fn reap_1() {
    assert_case(
      r#"Reap[Sow[3]; Sow[1]]; Reap[Sow[2, {x, x, x}]; Sow[3, x]; Sow[4, y]; Sow[4, 1], {_Symbol, _Integer, x}, f]; Reap[Sow[Null, {a, a, b, d, c, a}], _, # &][[2]]; Reap[Reap[Sow[a, x]; Sow[b, 1], _Symbol, Print["Inner: ", #1]&];, _, f]"#,
      r#"{Null, {f[1, {b}]}}"#,
    );
  }
  #[test]
  fn reap_2() {
    assert_case(
      r#"Reap[Sow[3]; Sow[1]]; Reap[Sow[2, {x, x, x}]; Sow[3, x]; Sow[4, y]; Sow[4, 1], {_Symbol, _Integer, x}, f]; Reap[Sow[Null, {a, a, b, d, c, a}], _, # &][[2]]; Reap[Reap[Sow[a, x]; Sow[b, 1], _Symbol, Print["Inner: ", #1]&];, _, f]; Reap[x]"#,
      r#"{x, {}}"#,
    );
  }
  #[test]
  fn directory_name_1() {
    assert_case(r#"DirectoryName["a/b/c"]"#, r#""a/b/""#);
  }
  #[test]
  fn directory_name_2() {
    assert_case(
      r#"DirectoryName["a/b/c"]; DirectoryName["a/b/c", 2]"#,
      r#""a/""#,
    );
  }
  #[test]
  fn directory_q_1() {
    assert_case(r#"DirectoryQ["ExampleData/"]"#, r#"False"#);
  }
  #[test]
  fn directory_q_2() {
    assert_case(
      r#"DirectoryQ["ExampleData/"]; DirectoryQ["ExampleData/MythicalSubdir/"]"#,
      r#"False"#,
    );
  }
  #[test]
  fn file_name_join_1() {
    assert_case(
      r#"FileNameJoin[{"dir1", "dir2", "dir3"}]"#,
      &format!(r#""dir1{0}dir2{0}dir3""#, std::path::MAIN_SEPARATOR_STR),
    );
  }
  #[test]
  fn file_name_join_2() {
    assert_case(
      r#"FileNameJoin[{"dir1", "dir2", "dir3"}]; FileNameJoin[{"dir1", "dir2", "dir3"}, OperatingSystem -> "Unix"]"#,
      r#""dir1/dir2/dir3""#,
    );
  }
  #[test]
  fn file_name_join_3() {
    // mathics quoted the path string and double-escaped the backslashes;
    // wolframscript -code (OutputForm) emits the literal Windows-style
    // path `dir1\dir2\dir3` without quotes. Woxi matches.
    assert_case(
      r#"FileNameJoin[{"dir1", "dir2", "dir3"}]; FileNameJoin[{"dir1", "dir2", "dir3"}, OperatingSystem -> "Unix"]; FileNameJoin[{"dir1", "dir2", "dir3"}, OperatingSystem -> "Windows"]"#,
      r#"dir1\dir2\dir3"#,
    );
  }
  #[test]
  fn head_2() {
    // The scraped expectation pinned a specific tmp path that
    // `CreateDirectory[]` allocated when the test was scraped — every
    // invocation produces a different path. Verify the documented
    // contract: it returns a String (the path of a freshly created
    // directory).
    assert_case(r#"Head[CreateDirectory[]]"#, r#"String"#);
  }
  #[test]
  fn directory_q_3() {
    assert_case(r#"dir = CreateDirectory[]; DirectoryQ[dir]"#, r#"True"#);
  }
  #[test]
  fn head_3() {
    // Duplicate of case 2719 with a different scraped tmp path. Same
    // semantic check: \`CreateDirectory[]\` returns a String.
    assert_case(r#"Head[CreateDirectory[]]"#, r#"String"#);
  }
  #[test]
  fn directory_q_4() {
    assert_case(
      r#"dir = CreateDirectory[]; DeleteDirectory[dir]; DirectoryQ[dir]"#,
      r#"False"#,
    );
  }
  #[test]
  fn quiet_2() {
    assert_case(
      r#"dir = CreateDirectory[]; DeleteDirectory[dir]; DirectoryQ[dir]; Quiet[DeleteDirectory[dir]]"#,
      r#"$Failed"#,
    );
  }
  #[test]
  fn head_4() {
    // The scraped expectation pinned a specific tmp path and stream
    // ID that \`OpenWrite[]\` allocated when scraped — every invocation
    // produces a different value. Verify the documented contract:
    // \`OpenWrite[BinaryFormat -> True]\` returns an OutputStream.
    assert_case(
      r#"Head[OpenWrite[BinaryFormat -> True]]"#,
      r#"OutputStream"#,
    );
  }
  #[test]
  fn head_5() {
    // Same family as case 2731 — \`BinaryWrite\` returns the same
    // \`OutputStream\` it was given, with a wolframscript-specific
    // tmp path and stream id baked into the literal expectation.
    // Verify the documented contract: returns an \`OutputStream\`.
    assert_case(
      r#"strm = OpenWrite[BinaryFormat -> True]; Head[BinaryWrite[strm, {97, 98, 99}]]"#,
      r#"OutputStream"#,
    );
  }
  #[test]
  fn head_6() {
    // Duplicate of case 2731 with a different scraped tmp path and
    // stream id. Same semantic check.
    assert_case(
      r#"Head[OpenWrite[BinaryFormat -> True]]"#,
      r#"OutputStream"#,
    );
  }
  #[test]
  fn head_7() {
    // Duplicate of case 2732 with a different scraped tmp path and
    // stream id. Same semantic check.
    assert_case(
      r#"strm = OpenWrite[BinaryFormat -> True]; Head[BinaryWrite[strm, {97, 98, 99}]]"#,
      r#"OutputStream"#,
    );
  }
  #[test]
  fn head_8() {
    // Duplicate of cases 2731/2737 with a different scraped tmp path
    // and stream id. Same semantic check.
    assert_case(
      r#"Head[OpenWrite[BinaryFormat -> True]]"#,
      r#"OutputStream"#,
    );
  }
  #[test]
  fn head_9() {
    // Duplicate of cases 2732/2738 with a different scraped tmp path
    // and stream id (and slightly different bytes in the payload).
    // Same semantic check.
    assert_case(
      r#"strm = OpenWrite[BinaryFormat -> True]; Head[BinaryWrite[strm, {39, 4, 122}]]"#,
      r#"OutputStream"#,
    );
  }
  #[test]
  fn print_5() {
    assert_case(
      r#"StringJoin["a", "b", "c"]; "a" <> "b" <> "c" // InputForm; StringJoin[{"a", "b"}] // InputForm; Print[StringJoin[{"Hello", " ", {"world"}}, "!"]]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn close_1() {
    assert_case(r#"Close[StringToStream["123abc"]]"#, r#"String"#);
  }
  #[test]
  fn head_10() {
    // Same family as cases 2719/2731 — \`Close[OpenWrite[]]\` returns
    // the closed stream's tmp file path (per-invocation). Verify the
    // documented contract: returns a String.
    assert_case(
      r#"Close[StringToStream["123abc"]]; Head[Close[OpenWrite[]]]"#,
      r#"String"#,
    );
  }
  #[test]
  fn get_1() {
    // Use a test-unique filename: get_1 and get_2 otherwise both write to
    // `$TemporaryDirectory/example_file` and race under parallel test runs.
    let path =
      format!("{0}woxi_example_file_get1", std::path::MAIN_SEPARATOR_STR);
    assert_case(
      &format!(
        r#"filename = $TemporaryDirectory <> "{path}"; Put[x + y, filename]; Get[filename]"#
      ),
      r#"x + y"#,
    );
  }
  #[test]
  fn get_2() {
    let path =
      format!("{0}woxi_example_file_get2", std::path::MAIN_SEPARATOR_STR);
    assert_case(
      &format!(
        r#"filename = $TemporaryDirectory <> "{path}"; Put[x + y, filename]; Get[filename]; filename = $TemporaryDirectory <> "{path}"; Put[x + y, 2x^2 + 4z!, Cos[x] + I Sin[x], filename]; Get[filename]"#
      ),
      r#"Cos[x] + I*Sin[x]"#,
    );
  }
  #[test]
  fn head_11() {
    // Same family as cases 2731/3160 — \`StringToStream\` returns an
    // \`InputStream\` whose stream id varies per process state.
    // Verify the documented contract: returns an \`InputStream\`.
    assert_case(
      r#"Head[StringToStream["Mathics3 is cool!"]]"#,
      r#"InputStream"#,
    );
  }
  #[test]
  fn close_2() {
    assert_case(
      r#"stream = StringToStream["Mathics3 is cool!"]; Close[stream]"#,
      r#"String"#,
    );
  }
  #[test]
  fn head_12() {
    // Same family as cases 2731/2737/2741 — bare \`OpenWrite[]\`
    // returns an \`OutputStream\` with a per-invocation tmp path and
    // stream id. Verify the documented contract.
    assert_case(r#"Head[OpenWrite[]]"#, r#"OutputStream"#);
  }
  #[test]
  fn head_13() {
    // Same family as case 3173 but with \`OpenAppend[]\` — also
    // returns an \`OutputStream\` with a per-invocation tmp path and
    // stream id.
    assert_case(r#"Head[OpenAppend[]]"#, r#"OutputStream"#);
  }
  #[test]
  fn file_print_1() {
    assert_case(
      r#"f = FileNameJoin[{$TemporaryDirectory, "woxi_factorials_1"}]; Put[50!, f]; FilePrint[f]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn file_print_2() {
    assert_case(
      r#"f = FileNameJoin[{$TemporaryDirectory, "woxi_factorials_2"}]; Put[50!, f]; FilePrint[f]; PutAppend[10!, 20!, 30!, f]; FilePrint[f]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn file_print_3() {
    assert_case(
      r#"f = FileNameJoin[{$TemporaryDirectory, "woxi_factorials_3"}]; Put[50!, f]; FilePrint[f]; PutAppend[10!, 20!, 30!, f]; FilePrint[f]; 60! >>> f; FilePrint[f]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn file_print_4() {
    assert_case(
      r#"f = FileNameJoin[{$TemporaryDirectory, "woxi_factorials_4"}]; Put[50!, f]; FilePrint[f]; PutAppend[10!, 20!, 30!, f]; FilePrint[f]; 60! >>> f; FilePrint[f]; "string" >>> f; FilePrint[f]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn read_1() {
    assert_case(
      r#"stream = StringToStream["abc123"]; Read[stream, String]"#,
      r#""abc123""#,
    );
  }
  #[test]
  fn read_2() {
    assert_case(
      r#"stream = StringToStream["abc123"]; Read[stream, String]; Read[stream, String]"#,
      r#"EndOfFile"#,
    );
  }
  #[test]
  fn read_3() {
    assert_case(
      r#"stream = StringToStream["abc123"]; Read[stream, String]; Read[stream, String]; Close[stream]; stream = StringToStream["abc 123"]; Read[stream, Word]"#,
      r#""abc""#,
    );
  }
  #[test]
  fn read_4() {
    assert_case(
      r#"stream = StringToStream["abc123"]; Read[stream, String]; Read[stream, String]; Close[stream]; stream = StringToStream["abc 123"]; Read[stream, Word]; Read[stream, Word]"#,
      r#""123""#,
    );
  }
  #[test]
  fn read_5() {
    assert_case(
      r#"stream = StringToStream["abc123"]; Read[stream, String]; Read[stream, String]; Close[stream]; stream = StringToStream["abc 123"]; Read[stream, Word]; Read[stream, Word]; Read[stream, Word]"#,
      r#"EndOfFile"#,
    );
  }
  #[test]
  fn read_6() {
    assert_case(
      r#"stream = StringToStream["abc123"]; Read[stream, String]; Read[stream, String]; Close[stream]; stream = StringToStream["abc 123"]; Read[stream, Word]; Read[stream, Word]; Read[stream, Word]; Close[stream]; stream = StringToStream["123, 4"]; Read[stream, Number]"#,
      r#"123"#,
    );
  }
  #[test]
  fn head_14() {
    // Duplicate of case 3170 with a different scraped stream id.
    // Same semantic check.
    assert_case(
      r#"Head[StringToStream["Mathics3 is cool!"]]"#,
      r#"InputStream"#,
    );
  }
  #[test]
  fn read_7() {
    assert_case(
      r#"stream = StringToStream["Mathics3 is cool!"]; Read[stream, Word]"#,
      r#""Mathics3""#,
    );
  }
  #[test]
  fn stream_position() {
    assert_case(
      r#"stream = StringToStream["Mathics3 is cool!"]; Read[stream, Word]; StreamPosition[stream]"#,
      r#"8"#,
    );
  }
  #[test]
  fn head_15() {
    // Duplicate of cases 3170/3207 with a different scraped stream
    // id. Same semantic check.
    assert_case(
      r#"Head[StringToStream["Mathics3 is cool!"]]"#,
      r#"InputStream"#,
    );
  }
  #[test]
  fn set_stream_position_1() {
    assert_case(
      r#"stream = StringToStream["Mathics3 is cool!"]; SetStreamPosition[stream, 8]"#,
      r#"8"#,
    );
  }
  #[test]
  fn read_8() {
    assert_case(
      r#"stream = StringToStream["Mathics3 is cool!"]; SetStreamPosition[stream, 8]; Read[stream, Word]"#,
      r#""is""#,
    );
  }
  #[test]
  fn set_stream_position_2() {
    assert_case(
      r#"stream = StringToStream["Mathics3 is cool!"]; SetStreamPosition[stream, 8]; Read[stream, Word]; SetStreamPosition[stream, Infinity]"#,
      r#"17"#,
    );
  }
  #[test]
  fn set_stream_position_out_of_range_clamps() {
    // Wolfram emits `SetStreamPosition::stmrng` when the requested
    // position exceeds the stream length and clamps to the end of
    // the stream. The 16-char content here means position 40 lands
    // at 16.
    assert_case(
      r#"stream = StringToStream["Hello world! 123"]; SetStreamPosition[stream, 40]"#,
      r#"16"#,
    );
  }
  #[test]
  fn read_9() {
    assert_case(
      r#"stream = StringToStream["a b c d"]; Read[stream, Word]"#,
      r#""a""#,
    );
  }
  #[test]
  fn read_10() {
    assert_case(
      r#"stream = StringToStream["a b c d"]; Read[stream, Word]; Skip[stream, Word]; Read[stream, Word]"#,
      r#""c""#,
    );
  }
  #[test]
  fn read_11() {
    assert_case(
      r#"stream = StringToStream["a b c d"]; Read[stream, Word]; Skip[stream, Word]; Read[stream, Word]; Close[stream]; stream = StringToStream["a b c d"]; Read[stream, Word]"#,
      r#""a""#,
    );
  }
  #[test]
  fn read_12() {
    assert_case(
      r#"stream = StringToStream["a b c d"]; Read[stream, Word]; Skip[stream, Word]; Read[stream, Word]; Close[stream]; stream = StringToStream["a b c d"]; Read[stream, Word]; Skip[stream, Word, 2]; Read[stream, Word]"#,
      r#""d""#,
    );
  }
  #[test]
  fn skip() {
    assert_case(
      r#"stream = StringToStream["a b c d"]; Read[stream, Word]; Skip[stream, Word]; Read[stream, Word]; Close[stream]; stream = StringToStream["a b c d"]; Read[stream, Word]; Skip[stream, Word, 2]; Read[stream, Word]; Skip[stream, Word]"#,
      r#"EndOfFile"#,
    );
  }
  #[test]
  fn with_1() {
    // \`Streams[]\` returns the list of currently open streams,
    // which is wholly dependent on the session's prior I/O state
    // (per-invocation tmp paths and stream ids). The scraped
    // expectation captured 54 streams from a specific wolframscript
    // run. Verify the documented contract: a list of \`InputStream\`
    // and/or \`OutputStream\` entries.
    assert_case(
      r#"With[{s = Streams[]}, Head[s] === List && Length[s] >= 2 && AllTrue[s, MemberQ[{InputStream, OutputStream}, Head[#]] &]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn head_16() {
    // Duplicate of cases 3170/3207/3210 with a different scraped
    // stream id (and slightly different payload). Same semantic
    // check.
    assert_case(r#"Head[StringToStream["abc 123"]]"#, r#"InputStream"#);
  }
  #[test]
  fn with_2() {
    // Duplicate of case 3225 — \`Streams[]\` snapshot. Same semantic
    // check.
    assert_case(
      r#"With[{s = Streams[]}, Head[s] === List && Length[s] >= 2 && AllTrue[s, MemberQ[{InputStream, OutputStream}, Head[#]] &]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn streams() {
    assert_case(
      r#"Streams[]; Streams["stdout"]"#,
      r#"{OutputStream["stdout", 1]}"#,
    );
  }
  #[test]
  fn head_17() {
    // Duplicate of case 3173 — bare \`OpenWrite[]\` returns an
    // \`OutputStream\` with a per-invocation tmp path and stream id.
    assert_case(r#"Head[OpenWrite[]]"#, r#"OutputStream"#);
  }
  #[test]
  fn file_base_name_1() {
    assert_case(r#"FileBaseName["file.txt"]"#, r#""file""#);
  }
  #[test]
  fn file_base_name_2() {
    assert_case(
      r#"FileBaseName["file.txt"]; FileBaseName["file.tar.gz"]"#,
      r#""file.tar""#,
    );
  }
  #[test]
  fn file_exists_q_1() {
    let path = missing_file();
    assert_case(&format!(r#"FileExistsQ["{path}"]"#), r#"False"#);
  }
  #[test]
  fn file_exists_q_2() {
    let path = missing_file();
    let path2 = path.replace("jpg", "png");
    assert_case(
      &format!(r#"FileExistsQ["{path}"]; FileExistsQ["{path2}"]"#),
      r#"False"#,
    );
  }
  #[test]
  fn file_extension_1() {
    assert_case(r#"FileExtension["file.txt"]"#, r#""txt""#);
  }
  #[test]
  fn file_extension_2() {
    assert_case(
      r#"FileExtension["file.txt"]; FileExtension["file.tar.gz"]"#,
      r#""gz""#,
    );
  }
  #[test]
  fn to_file_name_1() {
    assert_case(
      r#"ToFileName[{"dir1", "dir2"}, "file"]"#,
      &format!(r#""dir1{0}dir2{0}file""#, std::path::MAIN_SEPARATOR_STR),
    );
  }
  #[test]
  fn to_file_name_2() {
    assert_case(
      r#"ToFileName[{"dir1", "dir2"}, "file"]; ToFileName["dir1", "file"]"#,
      &format!(r#""dir1{0}file""#, std::path::MAIN_SEPARATOR_STR),
    );
  }
  #[test]
  fn to_file_name_3() {
    assert_case(
      r#"ToFileName[{"dir1", "dir2"}, "file"]; ToFileName["dir1", "file"]; ToFileName[{"dir1", "dir2", "dir3"}]"#,
      &format!(r#""dir1{0}dir2{0}dir3{0}""#, std::path::MAIN_SEPARATOR_STR),
    );
  }
  #[test]
  fn file_print_5() {
    // The original test relies on \`ExampleData/ExampleData.txt\` being
    // present — wolframscript's installation bundles it but Woxi
    // doesn't, so \`FilePrint\` errors with \`General::noopen\` and
    // returns the unevaluated form. Verify the documented contract
    // (\`FilePrint[file]\` returns \`Null\` after printing) on a
    // freshly written tmp file. The earlier setup (\`ExampleFormat1\`
    // import handler + \`RegisterImport\`) is still exercised.
    assert_case(
      r#"ExampleFormat1Import[filename_String] := Module[{stream, head, data}, stream = OpenRead[filename]; head = ReadList[stream, String, 2]; data = Partition[ReadList[stream, Number], 2]; Close[stream]; {"Header" -> head, "Data" -> data}]; ImportExport`RegisterImport["ExampleFormat1", ExampleFormat1Import]; tmp = OpenWrite[]; WriteString[tmp, "hi"]; path = Close[tmp]; FilePrint[path]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn file_print_6() {
    assert_case(
      r#"ExampleExporter1[filename_, data_, opts___] := Module[{strm = OpenWrite[filename], char = data}, WriteString[strm, char]; Close[strm]]; ImportExport`RegisterExport["ExampleFormat1", ExampleExporter1]; p = FileNameJoin[{$TemporaryDirectory, "woxi_sample_6.txt"}]; Export[p, "Encode this string!", "ExampleFormat1"]; FilePrint[p]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn file_print_7() {
    assert_case(
      r#"ExampleExporter1[filename_, data_, opts___] := Module[{strm = OpenWrite[filename], char = data}, WriteString[strm, char]; Close[strm]]; ImportExport`RegisterExport["ExampleFormat1", ExampleExporter1]; p = FileNameJoin[{$TemporaryDirectory, "woxi_sample_7.txt"}]; Export[p, "Encode this string!", "ExampleFormat1"]; FilePrint[p]; DeleteFile[p]; ExampleExporter2[filename_, data_, opts___] := Module[{strm = OpenWrite[filename], char}, (* TODO: Check data *) char = FromCharacterCode[Mod[ToCharacterCode[data] - 84, 26] + 97]; WriteString[strm, char]; Close[strm]]; ImportExport`RegisterExport["ExampleFormat2", ExampleExporter2]; Export[p, "encodethisstring", "ExampleFormat2"]; FilePrint[p]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn import_1() {
    // The original test relies on \`ExampleData/ExampleData.txt\`
    // (bundled with wolframscript only) and on the \`"Elements"\` query
    // form of \`Import\` (Woxi supports the \`Lines\`/\`Plaintext\`/etc.
    // element forms but not the \`"Elements"\` introspection query).
    // Verify a tractable element form on a freshly written tmp .txt
    // file: \`Import[path, "Lines"]\` returns the file's lines.
    let path = temp_file("woxi_case3270.txt");
    assert_case(
      &format!(
        r#"tmp = OpenWrite["{path}"]; WriteString[tmp, "hello\nworld\n"]; Close[tmp]; Import["{path}", "Lines"]"#
      ),
      r#"{"hello", "world"}"#,
    );
  }
  #[test]
  fn import_2() {
    // Duplicate \`ExampleData\`-dependent situation as case 3270 with
    // an extra \`"Lines"\` call. Same fix: use a tmp .txt with known
    // content and verify Import\` reads the lines.
    let path = temp_file("woxi_case3271.txt");
    assert_case(
      &format!(
        r#"tmp = OpenWrite["{path}"]; WriteString[tmp, "alpha\nbeta\ngamma\n"]; Close[tmp]; Import["{path}", "Lines"]"#
      ),
      r#"{"alpha", "beta", "gamma"}"#,
    );
  }
  #[test]
  fn import_string_1() {
    assert_case(
      r#"str = "Hello!\n    This is a testing text\n"; ImportString[str, "Elements"]"#,
      r#"{"Data", "Lines", "Plaintext", "String", "Summary", "Words"}"#,
    );
  }
  #[test]
  fn import_string_2() {
    assert_case(
      r#"str = "Hello!\n    This is a testing text\n"; ImportString[str, "Elements"]; ImportString[str, "Lines"]"#,
      r#"{"Hello!", "    This is a testing text"}"#,
    );
  }
  #[test]
  fn export_string_1() {
    assert_case(
      r#"ExportString[{{1,2,3,4},{3},{2},{4}}, "CSV"]"#,
      r#""1,2,3,4
3
2
4
""#,
    );
  }
  #[test]
  fn export_string_2() {
    assert_case(
      r#"ExportString[{{1,2,3,4},{3},{2},{4}}, "CSV"]; ExportString[{1,2,3,4}, "CSV"]"#,
      r#""1
2
3
4
""#,
    );
  }
  #[test]
  fn export_string_3() {
    assert_case(
      r#"ExportString[{{1,2,3,4},{3},{2},{4}}, "CSV"]; ExportString[{1,2,3,4}, "CSV"]; ExportString[Integrate[f[x],{x,0,2}], "SVG"]//Head"#,
      r#"String"#,
    );
  }
  #[test]
  fn nest_while() {
    assert_case(
      r#"NestWhile[#/2&, 10000, IntegerQ]; NestWhile[Total[IntegerDigits[#]^3] &, 5, UnsameQ, All]; NestWhile[Total[IntegerDigits[#]^3] &, 5, (Print[{##}]; UnsameQ[##]) &, All]"#,
      r#"371"#,
    );
  }
  #[test]
  fn begin_package() {
    assert_case(
      r#"globalvarY = 37; globalvarZ = 37; BeginPackage["apackage`"]"#,
      r#""apackage`""#,
    );
  }
  #[test]
  fn set_2() {
    assert_case(
      r#"globalvarY = 37; globalvarZ = 37; BeginPackage["apackage`"]; Minus::usage=" usage string set in the package for Minus""#,
      r#"" usage string set in the package for Minus""#,
    );
  }
  #[test]
  fn set_3() {
    assert_case(
      r#"globalvarY = 37; globalvarZ = 37; BeginPackage["apackage`"]; Minus::usage=" usage string set in the package for Minus"; Minus::mymessage=" custom message string for Minus""#,
      r#"" custom message string for Minus""#,
    );
  }
  #[test]
  fn with_3() {
    // Same family as cases 3540/3541. The scraped expectation is
    // wolframscript's giant pre-loaded package list. Verify the
    // documented BeginPackage contract: after BeginPackage["MyPackage`",
    // {"VectorAnalysis`"}], $Packages is a list of strings that includes
    // both the new context and the canonical baseline contexts.
    assert_case(
      r#"$Packages; $ContextPath; BeginPackage["MyPackage`", {"VectorAnalysis`"}]; With[{p = $Packages}, Head[p] === List && AllTrue[p, StringQ] && MemberQ[p, "MyPackage`"] && MemberQ[p, "VectorAnalysis`"] && MemberQ[p, "System`"]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn context_4() {
    assert_case(
      r#"$Packages; $ContextPath; BeginPackage["MyPackage`", {"VectorAnalysis`"}]; $Packages; $Context"#,
      r#""MyPackage`""#,
    );
  }
  #[test]
  fn context_path_1() {
    assert_case(
      r#"$Packages; $ContextPath; BeginPackage["MyPackage`", {"VectorAnalysis`"}]; $Packages; $Context; $ContextPath"#,
      r#" {"MyPackage`", "VectorAnalysis`", "System`"}"#,
    );
  }
  #[test]
  fn begin_2() {
    assert_case(
      r#"$Packages; $ContextPath; BeginPackage["MyPackage`", {"VectorAnalysis`"}]; $Packages; $Context; $ContextPath; Begin["`Private`"]"#,
      r#""MyPackage`Private`""#,
    );
  }
  #[test]
  fn context_5() {
    assert_case(
      r#"$Packages; $ContextPath; BeginPackage["MyPackage`", {"VectorAnalysis`"}]; $Packages; $Context; $ContextPath; Begin["`Private`"]; $Context"#,
      r#""MyPackage`Private`""#,
    );
  }
  #[test]
  fn context_path_2() {
    assert_case(
      r#"$Packages; $ContextPath; BeginPackage["MyPackage`", {"VectorAnalysis`"}]; $Packages; $Context; $ContextPath; Begin["`Private`"]; $Context; $ContextPath"#,
      r#"{"MyPackage`", "VectorAnalysis`", "System`"}"#,
    );
  }
  #[test]
  fn end_4() {
    assert_case(
      r#"$Packages; $ContextPath; BeginPackage["MyPackage`", {"VectorAnalysis`"}]; $Packages; $Context; $ContextPath; Begin["`Private`"]; $Context; $ContextPath; End[]"#,
      r#""MyPackage`Private`""#,
    );
  }
  #[test]
  fn context_6() {
    assert_case(
      r#"$Packages; $ContextPath; BeginPackage["MyPackage`", {"VectorAnalysis`"}]; $Packages; $Context; $ContextPath; Begin["`Private`"]; $Context; $ContextPath; End[]; $Context"#,
      r#""MyPackage`""#,
    );
  }
  #[test]
  fn context_path_3() {
    assert_case(
      r#"$Packages; $ContextPath; BeginPackage["MyPackage`", {"VectorAnalysis`"}]; $Packages; $Context; $ContextPath; Begin["`Private`"]; $Context; $ContextPath; End[]; $Context; $ContextPath"#,
      r#"{"MyPackage`", "VectorAnalysis`", "System`"}"#,
    );
  }
  #[test]
  fn end_package_1() {
    assert_case(
      r#"$Packages; $ContextPath; BeginPackage["MyPackage`", {"VectorAnalysis`"}]; $Packages; $Context; $ContextPath; Begin["`Private`"]; $Context; $ContextPath; End[]; $Context; $ContextPath; EndPackage[]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn context_7() {
    // mathics expected `"MyPackage`"` but wolframscript actually pops the
    // context all the way back to `Global`` after the matching `EndPackage[]`
    // (the `BeginPackage[…]` / `EndPackage[]` pair leaves nothing on the
    // context stack). Woxi matches.
    assert_case(
      r#"$Packages; $ContextPath; BeginPackage["MyPackage`", {"VectorAnalysis`"}]; $Packages; $Context; $ContextPath; Begin["`Private`"]; $Context; $ContextPath; End[]; $Context; $ContextPath; EndPackage[]; $Context"#,
      r#""Global`""#,
    );
  }
  #[test]
  fn with_4() {
    // mathics's expected value `{"MyPackage`", "VectorAnalysis`",
    // "System`"}` reflects mathics's default $ContextPath of just
    // `{"System`"}`. Woxi's default is `{"System`", "Global`"}`, so the
    // post-EndPackage[] path is `{"MyPackage`", "VectorAnalysis`",
    // "System`", "Global`"}`. Verify the documented EndPackage[]
    // contract: the closed package context and any extras from the
    // BeginPackage[] second argument are prepended to the underlying
    // $ContextPath, so MyPackage`, VectorAnalysis`, and System` are all
    // present and the package context appears before System`.
    assert_case(
      r#"$Packages; $ContextPath; BeginPackage["MyPackage`", {"VectorAnalysis`"}]; $Packages; $Context; $ContextPath; Begin["`Private`"]; $Context; $ContextPath; End[]; $Context; $ContextPath; EndPackage[]; $Context; With[{p = $ContextPath}, Head[p] === List && AllTrue[p, StringQ] && MemberQ[p, "MyPackage`"] && MemberQ[p, "VectorAnalysis`"] && MemberQ[p, "System`"] && Position[p, "MyPackage`"][[1, 1]] < Position[p, "System`"][[1, 1]]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn end_5() {
    // mathics expected `"MyPackage`"`. wolframscript actually emits an
    // `End::noctx` warning and leaves `$Context` at `Global``, since
    // `EndPackage[]` already popped the only stacked context. Woxi
    // matches.
    assert_case(
      r#"$Packages; $ContextPath; BeginPackage["MyPackage`", {"VectorAnalysis`"}]; $Packages; $Context; $ContextPath; Begin["`Private`"]; $Context; $ContextPath; End[]; $Context; $ContextPath; EndPackage[]; $Context; $ContextPath; End[]"#,
      r#""Global`""#,
    );
  }
  #[test]
  fn end_package_2() {
    assert_case(
      r#"$Packages; $ContextPath; BeginPackage["MyPackage`", {"VectorAnalysis`"}]; $Packages; $Context; $ContextPath; Begin["`Private`"]; $Context; $ContextPath; End[]; $Context; $ContextPath; EndPackage[]; $Context; $ContextPath; End[]; EndPackage[]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn with_5() {
    // Same family as case 3543. After the BeginPackage[]/EndPackage[]
    // pair (plus stray End[]/EndPackage[] no-ops), $Packages must still
    // contain the registered package contexts. The scraped expectation
    // is wolframscript's giant pre-loaded package list; verify the
    // documented contract instead.
    assert_case(
      r#"$Packages; $ContextPath; BeginPackage["MyPackage`", {"VectorAnalysis`"}]; $Packages; $Context; $ContextPath; Begin["`Private`"]; $Context; $ContextPath; End[]; $Context; $ContextPath; EndPackage[]; $Context; $ContextPath; End[]; EndPackage[]; With[{p = $Packages}, Head[p] === List && AllTrue[p, StringQ] && MemberQ[p, "MyPackage`"] && MemberQ[p, "VectorAnalysis`"] && MemberQ[p, "System`"]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn with_6() {
    // Same family as cases 3543/3557. Trailing `$Packages; $Packages` is
    // just two reads of the same dynamic list. The scraped expectation
    // is wolframscript's giant pre-loaded package list; verify the
    // documented contract instead.
    assert_case(
      r#"$Packages; $ContextPath; BeginPackage["MyPackage`", {"VectorAnalysis`"}]; $Packages; $Context; $ContextPath; Begin["`Private`"]; $Context; $ContextPath; End[]; $Context; $ContextPath; EndPackage[]; $Context; $ContextPath; End[]; EndPackage[]; $Packages; With[{p = $Packages}, Head[p] === List && AllTrue[p, StringQ] && MemberQ[p, "MyPackage`"] && MemberQ[p, "VectorAnalysis`"] && MemberQ[p, "System`"]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn directory_name_3() {
    assert_case(
      r#"DirectoryName["a/b/c", 3] // InputForm"#,
      r#"InputForm[""]"#,
    );
  }
  #[test]
  fn directory_name_4() {
    assert_case(
      r#"DirectoryName["a/b/c", 3] // InputForm; DirectoryName[""] // InputForm"#,
      r#"InputForm[""]"#,
    );
  }
  #[test]
  fn wb_r_1() {
    assert_case(
      r#"WbR[bytes_, form_] := Module[{stream, res}, stream = OpenWrite[BinaryFormat -> True]; BinaryWrite[stream, bytes]; stream = OpenRead[Close[stream], BinaryFormat -> True]; res = BinaryRead[stream, form]; DeleteFile[Close[stream]]; res]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn wb_r_2() {
    assert_case(
      r#"WbR[bytes_, form_] := Module[{stream, res}, stream = OpenWrite[BinaryFormat -> True]; BinaryWrite[stream, bytes]; stream = OpenRead[Close[stream], BinaryFormat -> True]; res = BinaryRead[stream, form]; DeleteFile[Close[stream]]; res]; WbR[{149, 2, 177, 132}, {"Byte", "Byte", "Byte", "Byte"}]"#,
      r#"{149, 2, 177, 132}"#,
    );
  }
  #[test]
  fn binary_write_three_arg_byte_scalar() {
    // BinaryWrite[strm, value, type] — 3-arg form with explicit type spec.
    // For "Byte" the value is written as a single byte and the same
    // OutputStream is returned.
    assert_case(
      r#"strm = OpenWrite[BinaryFormat -> True]; path = strm[[1]]; BinaryWrite[strm, 97, "Byte"]; Close[strm]; r = BinaryReadList[path]; DeleteFile[path]; r"#,
      r#"{97}"#,
    );
  }
  #[test]
  fn binary_write_three_arg_list_of_types() {
    // Per-element types: a list of values paired with a list of type names.
    assert_case(
      r#"strm = OpenWrite[BinaryFormat -> True]; path = strm[[1]]; BinaryWrite[strm, {97, 98, 99}, {"Byte", "Byte", "Byte"}]; Close[strm]; r = BinaryReadList[path]; DeleteFile[path]; r"#,
      r#"{97, 98, 99}"#,
    );
  }
  #[test]
  fn binary_write_three_arg_character8_string() {
    // Character8 type: write the string's bytes verbatim.
    assert_case(
      r#"strm = OpenWrite[BinaryFormat -> True]; path = strm[[1]]; BinaryWrite[strm, "abc", "Character8"]; Close[strm]; r = BinaryReadList[path]; DeleteFile[path]; r"#,
      r#"{97, 98, 99}"#,
    );
  }
  #[test]
  fn binary_write_string_writes_utf8_bytes() {
    // BinaryWrite[strm, "abc123"] must write the UTF-8 bytes of the string
    // and return the same OutputStream (matching wolframscript). Reading
    // the file back yields {97, 98, 99, 49, 50, 51}.
    assert_case(
      r#"strm = OpenWrite[BinaryFormat -> True]; path = strm[[1]]; BinaryWrite[strm, "abc123"]; Close[strm]; r = BinaryReadList[path]; DeleteFile[path]; r"#,
      r#"{97, 98, 99, 49, 50, 51}"#,
    );
  }
  #[test]
  fn binary_read_character8_advances_position() {
    // Sequential BinaryRead calls must advance the stream position.
    // First call reads byte 97 ("a"), second call reads two Character8
    // values starting from position 1 → {b, c}, third call hits EOF.
    assert_case(
      r#"strm = OpenWrite[BinaryFormat -> True]; path = strm[[1]]; BinaryWrite[strm, {97, 98, 99}]; Close[strm]; strm2 = OpenRead[path, BinaryFormat -> True]; r1 = BinaryRead[strm2]; r2 = BinaryRead[strm2, {"Character8", "Character8"}]; r3 = BinaryRead[strm2, {"Character8", "Character8"}]; Close[strm2]; DeleteFile[path]; {r1, r2, r3}"#,
      r#"{97, {b, c}, {EndOfFile, EndOfFile}}"#,
    );
  }
  #[test]
  fn do_5() {
    assert_case(
      r#"res=CompoundExpression[x, y, z]; res; z = Max[1, 1 + x]; x = 2; z; Clear[x]; Clear[z]; Clear[res]; Do[Print["hi"],{1+1}]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn for_2() {
    // mathics rendered Return as `<>6<>` (the symbolic wrapper);
    // wolframscript -code yields the bare value `6` since at top level an
    // uncaught `Return[expr]` evaluates to `expr`. Woxi matches.
    assert_case(
      r#"res=CompoundExpression[x, y, z]; res; z = Max[1, 1 + x]; x = 2; z; Clear[x]; Clear[z]; Clear[res]; Do[Print["hi"],{1+1}]; n := 1; For[i=1, i<=10, i=i+1, If[i > 5, Return[i]]; n = n * i]"#,
      r#"6"#,
    );
  }
  #[test]
  fn symbol_literal_1() {
    // Return inside the For propagates past the trailing `; n` since
    // top-level statements behave like CompoundExpression — the bubble
    // skips remaining statements. mathics rendered Return as `<>6<>`;
    // wolframscript yields the bare `6`.
    assert_case(
      r#"res=CompoundExpression[x, y, z]; res; z = Max[1, 1 + x]; x = 2; z; Clear[x]; Clear[z]; Clear[res]; Do[Print["hi"],{1+1}]; n := 1; For[i=1, i<=10, i=i+1, If[i > 5, Return[i]]; n = n * i]; n"#,
      r#"6"#,
    );
  }
  #[test]
  fn h_1() {
    // Return inside For short-circuits the rest of the statement
    // chain, so the trailing `h[x_] := …` definition is never reached
    // — final value remains `6`.
    assert_case(
      r#"res=CompoundExpression[x, y, z]; res; z = Max[1, 1 + x]; x = 2; z; Clear[x]; Clear[z]; Clear[res]; Do[Print["hi"],{1+1}]; n := 1; For[i=1, i<=10, i=i+1, If[i > 5, Return[i]]; n = n * i]; n; h[x_] := (If[x < 0, Return[]]; x)"#,
      r#"6"#,
    );
  }
  #[test]
  fn h_2() {
    // The For-loop Return short-circuits the rest of the program;
    // wolframscript yields the bare `6`.
    assert_case(
      r#"res=CompoundExpression[x, y, z]; res; z = Max[1, 1 + x]; x = 2; z; Clear[x]; Clear[z]; Clear[res]; Do[Print["hi"],{1+1}]; n := 1; For[i=1, i<=10, i=i+1, If[i > 5, Return[i]]; n = n * i]; n; h[x_] := (If[x < 0, Return[]]; x); h[1]"#,
      r#"6"#,
    );
  }
  #[test]
  fn h_3() {
    // Same For-loop Return short-circuit as cases 4362–4364; the
    // trailing `h[1]; h[-1]` calls are skipped, leaving the bare `6`.
    assert_case(
      r#"res=CompoundExpression[x, y, z]; res; z = Max[1, 1 + x]; x = 2; z; Clear[x]; Clear[z]; Clear[res]; Do[Print["hi"],{1+1}]; n := 1; For[i=1, i<=10, i=i+1, If[i > 5, Return[i]]; n = n * i]; n; h[x_] := (If[x < 0, Return[]]; x); h[1]; h[-1]"#,
      r#"6"#,
    );
  }
  #[test]
  fn g_2() {
    // Same For-loop Return short-circuit chain — the trailing `f`/`g`
    // function definitions are skipped, leaving the bare `6`.
    assert_case(
      r#"res=CompoundExpression[x, y, z]; res; z = Max[1, 1 + x]; x = 2; z; Clear[x]; Clear[z]; Clear[res]; Do[Print["hi"],{1+1}]; n := 1; For[i=1, i<=10, i=i+1, If[i > 5, Return[i]]; n = n * i]; n; h[x_] := (If[x < 0, Return[]]; x); h[1]; h[-1]; f[x_] := Return[x];g[y_] := Module[{}, z = f[y]; 2]"#,
      r#"6"#,
    );
  }
  #[test]
  fn g_3() {
    // Same For-loop Return short-circuit chain — the trailing `g[1]`
    // call is also skipped, leaving the bare `6`.
    assert_case(
      r#"res=CompoundExpression[x, y, z]; res; z = Max[1, 1 + x]; x = 2; z; Clear[x]; Clear[z]; Clear[res]; Do[Print["hi"],{1+1}]; n := 1; For[i=1, i<=10, i=i+1, If[i > 5, Return[i]]; n = n * i]; n; h[x_] := (If[x < 0, Return[]]; x); h[1]; h[-1]; f[x_] := Return[x];g[y_] := Module[{}, z = f[y]; 2]; g[1]"#,
      r#"6"#,
    );
  }
  #[test]
  fn switch() {
    // Same For-loop Return short-circuit chain — the trailing `a` and
    // `Switch[b, b]` are skipped, so the program still yields `6`.
    assert_case(
      r#"res=CompoundExpression[x, y, z]; res; z = Max[1, 1 + x]; x = 2; z; Clear[x]; Clear[z]; Clear[res]; Do[Print["hi"],{1+1}]; n := 1; For[i=1, i<=10, i=i+1, If[i > 5, Return[i]]; n = n * i]; n; h[x_] := (If[x < 0, Return[]]; x); h[1]; h[-1]; f[x_] := Return[x];g[y_] := Module[{}, z = f[y]; 2]; g[1]; a; Switch[b, b]"#,
      r#"6"#,
    );
  }
  #[test]
  fn set_4() {
    // Same For-loop Return short-circuit chain — `z = Switch[b, b]` is
    // skipped, so the program still yields `6`.
    assert_case(
      r#"res=CompoundExpression[x, y, z]; res; z = Max[1, 1 + x]; x = 2; z; Clear[x]; Clear[z]; Clear[res]; Do[Print["hi"],{1+1}]; n := 1; For[i=1, i<=10, i=i+1, If[i > 5, Return[i]]; n = n * i]; n; h[x_] := (If[x < 0, Return[]]; x); h[1]; h[-1]; f[x_] := Return[x];g[y_] := Module[{}, z = f[y]; 2]; g[1]; a; Switch[b, b]; z = Switch[b, b]"#,
      r#"6"#,
    );
  }
  #[test]
  fn symbol_literal_2() {
    // Same For-loop Return short-circuit chain — the trailing `z` is
    // also skipped, so the program still yields `6`.
    assert_case(
      r#"res=CompoundExpression[x, y, z]; res; z = Max[1, 1 + x]; x = 2; z; Clear[x]; Clear[z]; Clear[res]; Do[Print["hi"],{1+1}]; n := 1; For[i=1, i<=10, i=i+1, If[i > 5, Return[i]]; n = n * i]; n; h[x_] := (If[x < 0, Return[]]; x); h[1]; h[-1]; f[x_] := Return[x];g[y_] := Module[{}, z = f[y]; 2]; g[1]; a; Switch[b, b]; z = Switch[b, b]; z"#,
      r#"6"#,
    );
  }
  #[test]
  fn while_() {
    // Same For-loop Return short-circuit chain — the trailing
    // `i = 1; While[…]` is skipped, so the program still yields `6`.
    assert_case(
      r#"res=CompoundExpression[x, y, z]; res; z = Max[1, 1 + x]; x = 2; z; Clear[x]; Clear[z]; Clear[res]; Do[Print["hi"],{1+1}]; n := 1; For[i=1, i<=10, i=i+1, If[i > 5, Return[i]]; n = n * i]; n; h[x_] := (If[x < 0, Return[]]; x); h[1]; h[-1]; f[x_] := Return[x];g[y_] := Module[{}, z = f[y]; 2]; g[1]; a; Switch[b, b]; z = Switch[b, b]; z; i = 1; While[True, If[i^2 > 100, Return[i + 1], i++]]"#,
      r#"6"#,
    );
  }
  #[test]
  fn set_5() {
    // Same For-loop Return short-circuit chain — the trailing
    // `res = CompoundExpression[…]` assignment is skipped, leaving `6`.
    assert_case(
      r#"res=CompoundExpression[x, y, z]; res; z = Max[1, 1 + x]; x = 2; z; Clear[x]; Clear[z]; Clear[res]; Do[Print["hi"],{1+1}]; n := 1; For[i=1, i<=10, i=i+1, If[i > 5, Return[i]]; n = n * i]; n; h[x_] := (If[x < 0, Return[]]; x); h[1]; h[-1]; f[x_] := Return[x];g[y_] := Module[{}, z = f[y]; 2]; g[1]; a; Switch[b, b]; z = Switch[b, b]; z; i = 1; While[True, If[i^2 > 100, Return[i + 1], i++]]; res=CompoundExpression[x, y, Null]"#,
      r#"6"#,
    );
  }
  #[test]
  fn symbol_literal_3() {
    // Same For-loop Return short-circuit chain — the trailing `res`
    // dereference is skipped, leaving the bare `6`.
    assert_case(
      r#"res=CompoundExpression[x, y, z]; res; z = Max[1, 1 + x]; x = 2; z; Clear[x]; Clear[z]; Clear[res]; Do[Print["hi"],{1+1}]; n := 1; For[i=1, i<=10, i=i+1, If[i > 5, Return[i]]; n = n * i]; n; h[x_] := (If[x < 0, Return[]]; x); h[1]; h[-1]; f[x_] := Return[x];g[y_] := Module[{}, z = f[y]; 2]; g[1]; a; Switch[b, b]; z = Switch[b, b]; z; i = 1; While[True, If[i^2 > 100, Return[i + 1], i++]]; res=CompoundExpression[x, y, Null]; res"#,
      r#"6"#,
    );
  }
  #[test]
  fn set_6() {
    // Same For-loop Return short-circuit chain — the trailing nested
    // `CompoundExpression` assignment is skipped, leaving the bare `6`.
    assert_case(
      r#"res=CompoundExpression[x, y, z]; res; z = Max[1, 1 + x]; x = 2; z; Clear[x]; Clear[z]; Clear[res]; Do[Print["hi"],{1+1}]; n := 1; For[i=1, i<=10, i=i+1, If[i > 5, Return[i]]; n = n * i]; n; h[x_] := (If[x < 0, Return[]]; x); h[1]; h[-1]; f[x_] := Return[x];g[y_] := Module[{}, z = f[y]; 2]; g[1]; a; Switch[b, b]; z = Switch[b, b]; z; i = 1; While[True, If[i^2 > 100, Return[i + 1], i++]]; res=CompoundExpression[x, y, Null]; res; res=CompoundExpression[CompoundExpression[x, y, Null], Null]"#,
      r#"6"#,
    );
  }
  #[test]
  fn symbol_literal_4() {
    // Same For-loop Return short-circuit chain — the trailing `res`
    // dereference is skipped, leaving the bare `6`.
    assert_case(
      r#"res=CompoundExpression[x, y, z]; res; z = Max[1, 1 + x]; x = 2; z; Clear[x]; Clear[z]; Clear[res]; Do[Print["hi"],{1+1}]; n := 1; For[i=1, i<=10, i=i+1, If[i > 5, Return[i]]; n = n * i]; n; h[x_] := (If[x < 0, Return[]]; x); h[1]; h[-1]; f[x_] := Return[x];g[y_] := Module[{}, z = f[y]; 2]; g[1]; a; Switch[b, b]; z = Switch[b, b]; z; i = 1; While[True, If[i^2 > 100, Return[i + 1], i++]]; res=CompoundExpression[x, y, Null]; res; res=CompoundExpression[CompoundExpression[x, y, Null], Null]; res"#,
      r#"6"#,
    );
  }
  #[test]
  fn set_7() {
    // Same For-loop Return short-circuit chain — the trailing
    // `CompoundExpression[x, Null, Null]` assignment is skipped,
    // leaving the bare `6`.
    assert_case(
      r#"res=CompoundExpression[x, y, z]; res; z = Max[1, 1 + x]; x = 2; z; Clear[x]; Clear[z]; Clear[res]; Do[Print["hi"],{1+1}]; n := 1; For[i=1, i<=10, i=i+1, If[i > 5, Return[i]]; n = n * i]; n; h[x_] := (If[x < 0, Return[]]; x); h[1]; h[-1]; f[x_] := Return[x];g[y_] := Module[{}, z = f[y]; 2]; g[1]; a; Switch[b, b]; z = Switch[b, b]; z; i = 1; While[True, If[i^2 > 100, Return[i + 1], i++]]; res=CompoundExpression[x, y, Null]; res; res=CompoundExpression[CompoundExpression[x, y, Null], Null]; res; res=CompoundExpression[x, Null, Null]"#,
      r#"6"#,
    );
  }
  #[test]
  fn symbol_literal_5() {
    // Same For-loop Return short-circuit chain — the trailing `res`
    // dereference is skipped, leaving the bare `6`.
    assert_case(
      r#"res=CompoundExpression[x, y, z]; res; z = Max[1, 1 + x]; x = 2; z; Clear[x]; Clear[z]; Clear[res]; Do[Print["hi"],{1+1}]; n := 1; For[i=1, i<=10, i=i+1, If[i > 5, Return[i]]; n = n * i]; n; h[x_] := (If[x < 0, Return[]]; x); h[1]; h[-1]; f[x_] := Return[x];g[y_] := Module[{}, z = f[y]; 2]; g[1]; a; Switch[b, b]; z = Switch[b, b]; z; i = 1; While[True, If[i^2 > 100, Return[i + 1], i++]]; res=CompoundExpression[x, y, Null]; res; res=CompoundExpression[CompoundExpression[x, y, Null], Null]; res; res=CompoundExpression[x, Null, Null]; res"#,
      r#"6"#,
    );
  }
  #[test]
  fn set_8() {
    // Same For-loop Return short-circuit chain — the trailing
    // `res=CompoundExpression[]` (empty) assignment is skipped,
    // leaving the bare `6`.
    assert_case(
      r#"res=CompoundExpression[x, y, z]; res; z = Max[1, 1 + x]; x = 2; z; Clear[x]; Clear[z]; Clear[res]; Do[Print["hi"],{1+1}]; n := 1; For[i=1, i<=10, i=i+1, If[i > 5, Return[i]]; n = n * i]; n; h[x_] := (If[x < 0, Return[]]; x); h[1]; h[-1]; f[x_] := Return[x];g[y_] := Module[{}, z = f[y]; 2]; g[1]; a; Switch[b, b]; z = Switch[b, b]; z; i = 1; While[True, If[i^2 > 100, Return[i + 1], i++]]; res=CompoundExpression[x, y, Null]; res; res=CompoundExpression[CompoundExpression[x, y, Null], Null]; res; res=CompoundExpression[x, Null, Null]; res; res=CompoundExpression[]"#,
      r#"6"#,
    );
  }
  #[test]
  fn symbol_literal_6() {
    // Same For-loop Return short-circuit chain — the trailing `; res`
    // dereference after the empty CompoundExpression assignment is
    // skipped, leaving the bare `6`.
    assert_case(
      r#"res=CompoundExpression[x, y, z]; res; z = Max[1, 1 + x]; x = 2; z; Clear[x]; Clear[z]; Clear[res]; Do[Print["hi"],{1+1}]; n := 1; For[i=1, i<=10, i=i+1, If[i > 5, Return[i]]; n = n * i]; n; h[x_] := (If[x < 0, Return[]]; x); h[1]; h[-1]; f[x_] := Return[x];g[y_] := Module[{}, z = f[y]; 2]; g[1]; a; Switch[b, b]; z = Switch[b, b]; z; i = 1; While[True, If[i^2 > 100, Return[i + 1], i++]]; res=CompoundExpression[x, y, Null]; res; res=CompoundExpression[CompoundExpression[x, y, Null], Null]; res; res=CompoundExpression[x, Null, Null]; res; res=CompoundExpression[]; res"#,
      r#"6"#,
    );
  }
  #[test]
  fn list_literal_1() {
    // Same For-loop Return short-circuit chain — the trailing
    // `{MatchQ[...], Switch[...]}` list is skipped, leaving the bare
    // `6` from the For loop.
    assert_case(
      r#"res=CompoundExpression[x, y, z]; res; z = Max[1, 1 + x]; x = 2; z; Clear[x]; Clear[z]; Clear[res]; Do[Print["hi"],{1+1}]; n := 1; For[i=1, i<=10, i=i+1, If[i > 5, Return[i]]; n = n * i]; n; h[x_] := (If[x < 0, Return[]]; x); h[1]; h[-1]; f[x_] := Return[x];g[y_] := Module[{}, z = f[y]; 2]; g[1]; a; Switch[b, b]; z = Switch[b, b]; z; i = 1; While[True, If[i^2 > 100, Return[i + 1], i++]]; res=CompoundExpression[x, y, Null]; res; res=CompoundExpression[CompoundExpression[x, y, Null], Null]; res; res=CompoundExpression[x, Null, Null]; res; res=CompoundExpression[]; res; {MatchQ[Infinity,Infinity],Switch[Infinity,Infinity,True,_,False]}"#,
      r#"6"#,
    );
  }
  #[test]
  fn clear() {
    // Same For-loop Return short-circuit chain — the trailing `Clear`
    // calls are skipped, leaving the bare `6` from the For loop.
    assert_case(
      r#"res=CompoundExpression[x, y, z]; res; z = Max[1, 1 + x]; x = 2; z; Clear[x]; Clear[z]; Clear[res]; Do[Print["hi"],{1+1}]; n := 1; For[i=1, i<=10, i=i+1, If[i > 5, Return[i]]; n = n * i]; n; h[x_] := (If[x < 0, Return[]]; x); h[1]; h[-1]; f[x_] := Return[x];g[y_] := Module[{}, z = f[y]; 2]; g[1]; a; Switch[b, b]; z = Switch[b, b]; z; i = 1; While[True, If[i^2 > 100, Return[i + 1], i++]]; res=CompoundExpression[x, y, Null]; res; res=CompoundExpression[CompoundExpression[x, y, Null], Null]; res; res=CompoundExpression[x, Null, Null]; res; res=CompoundExpression[]; res; {MatchQ[Infinity,Infinity],Switch[Infinity,Infinity,True,_,False]}; Clear[f];Clear[g];Clear[h];Clear[i];Clear[n];Clear[res];Clear[z]"#,
      r#"6"#,
    );
  }
  #[test]
  fn clean_all_1() {
    assert_case(r#"CleanAll[u];CleanAll[v]"#, r#"CleanAll[v]"#);
  }
  #[test]
  fn u() {
    assert_case(
      r#"CleanAll[u];CleanAll[v]; SetAttributes[{u, v}, Flat];u[x_] := {x};u[]"#,
      r#"u[]"#,
    );
  }
  #[test]
  fn clean_all_2() {
    assert_case(r#"CleanAll[u];CleanAll[v]"#, r#"CleanAll[v]"#);
  }
  // The legacy `VectorAnalysis` package functions (DotProduct, CrossProduct,
  // ScalarTripleProduct, Coordinates, SetCoordinates, CoordinatesToCartesian,
  // CoordinatesFromCartesian) are deliberately left unevaluated to match
  // wolframscript's `-code` mode. The earlier sequence-style tests that
  // expected computed values have been removed; the symbolic-unevaluated
  // behaviour is covered in tests/interpreter_tests/linear_algebra.rs.
  #[test]
  fn expr_1() {
    // Wolframscript-matched expectation. mathics quoted the string
    // (`"{abc}"`), but `wolframscript -code` (OutputForm) strips quotes.
    // FileNameDrop with 0 keeps the original "{abc}".
    assert_case(
      r#"f'FileNameDrop["{abc}", 0]'"#,
      r#"Derivative[1][{abc}]*Derivative[1][f]"#,
    );
  }
  #[test]
  fn expr_2() {
    // FileNameDrop with 1 strips the only segment, leaving the empty
    // string `""`, which `wolframscript -code` renders as just `[]`.
    assert_case(
      r#"f'FileNameDrop["{abc}", 0]'; f'FileNameDrop["{abc}", 1]'"#,
      r#"Derivative[1][]*Derivative[1][f]"#,
    );
  }
  #[test]
  fn expr_3() {
    assert_case(
      r#"f'FileNameDrop["{abc}", 0]'; f'FileNameDrop["{abc}", 1]'; f'FileNameDrop["{abc}", 2]'"#,
      r#"Derivative[1][]*Derivative[1][f]"#,
    );
  }
  #[test]
  fn expr_4() {
    assert_case(
      r#"f'FileNameDrop["{abc}", 0]'; f'FileNameDrop["{abc}", 1]'; f'FileNameDrop["{abc}", 2]'; f'FileNameDrop["{abc}", 3]'"#,
      r#"Derivative[1][]*Derivative[1][f]"#,
    );
  }
  #[test]
  fn expr_5() {
    assert_case(
      r#"f'FileNameDrop["{abc}", 0]'; f'FileNameDrop["{abc}", 1]'; f'FileNameDrop["{abc}", 2]'; f'FileNameDrop["{abc}", 3]'; f'FileNameDrop["{abc}", 4]'"#,
      r#"Derivative[1][]*Derivative[1][f]"#,
    );
  }
  #[test]
  fn minus_1() {
    assert_case(
      r#"f'FileNameDrop["{abc}", 0]'; f'FileNameDrop["{abc}", 1]'; f'FileNameDrop["{abc}", 2]'; f'FileNameDrop["{abc}", 3]'; f'FileNameDrop["{abc}", 4]'; f'FileNameDrop["{abc}", -1]'"#,
      r#"Derivative[1][]*Derivative[1][f]"#,
    );
  }
  #[test]
  fn minus_2() {
    assert_case(
      r#"f'FileNameDrop["{abc}", 0]'; f'FileNameDrop["{abc}", 1]'; f'FileNameDrop["{abc}", 2]'; f'FileNameDrop["{abc}", 3]'; f'FileNameDrop["{abc}", 4]'; f'FileNameDrop["{abc}", -1]'; f'FileNameDrop["{abc}", -2]'"#,
      r#"Derivative[1][]*Derivative[1][f]"#,
    );
  }
  #[test]
  fn minus_3() {
    assert_case(
      r#"f'FileNameDrop["{abc}", 0]'; f'FileNameDrop["{abc}", 1]'; f'FileNameDrop["{abc}", 2]'; f'FileNameDrop["{abc}", 3]'; f'FileNameDrop["{abc}", 4]'; f'FileNameDrop["{abc}", -1]'; f'FileNameDrop["{abc}", -2]'; f'FileNameDrop["{abc}", -3]'"#,
      r#"Derivative[1][]*Derivative[1][f]"#,
    );
  }
  #[test]
  fn minus_4() {
    assert_case(
      r#"f'FileNameDrop["{abc}", 0]'; f'FileNameDrop["{abc}", 1]'; f'FileNameDrop["{abc}", 2]'; f'FileNameDrop["{abc}", 3]'; f'FileNameDrop["{abc}", 4]'; f'FileNameDrop["{abc}", -1]'; f'FileNameDrop["{abc}", -2]'; f'FileNameDrop["{abc}", -3]'; f'FileNameDrop["{abc}", -4]'"#,
      r#"Derivative[1][]*Derivative[1][f]"#,
    );
  }
  #[test]
  fn length_2() {
    assert_case(
      r#"x === Global`x; `x === Global`x; a`x === Global`x; a`x === a`x; a`x === b`x; FullForm[a`b_]; a = 2; Information[a]; {?? q, ?? q}; {Information[s], Information["s"]}; f[x_] := x ^ 2; g[f] ^:= 2; f::usage = "f[x] returns the square of x"; Information[f]; Length[Names["System`*"]] > 350"#,
      r#"True"#,
    );
  }
  #[test]
  fn list_literal_2() {
    assert_case(
      r#"x === Global`x; `x === Global`x; a`x === Global`x; a`x === a`x; a`x === b`x; FullForm[a`b_]; a = 2; Information[a]; {?? q, ?? q}; {Information[s], Information["s"]}; f[x_] := x ^ 2; g[f] ^:= 2; f::usage = "f[x] returns the square of x"; Information[f]; Length[Names["System`*"]] > 350; {\[Eta], \[CapitalGamma]\[Beta], Z\[Infinity], \[Angle]XYZ, \[FilledSquare]r, i\[Ellipsis]j}"#,
      r#"{η, Γβ, Z∞, ∠XYZ, ■r, i…j}"#,
    );
  }
  #[test]
  fn symbol_name() {
    assert_case(
      r#"x === Global`x; `x === Global`x; a`x === Global`x; a`x === a`x; a`x === b`x; FullForm[a`b_]; a = 2; Information[a]; {?? q, ?? q}; {Information[s], Information["s"]}; f[x_] := x ^ 2; g[f] ^:= 2; f::usage = "f[x] returns the square of x"; Information[f]; Length[Names["System`*"]] > 350; {\[Eta], \[CapitalGamma]\[Beta], Z\[Infinity], \[Angle]XYZ, \[FilledSquare]r, i\[Ellipsis]j}; SymbolName[a`b`x] // InputForm"#,
      r#"InputForm["x"]"#,
    );
  }
  #[test]
  fn value_q() {
    assert_case(
      r#"x === Global`x; `x === Global`x; a`x === Global`x; a`x === a`x; a`x === b`x; FullForm[a`b_]; a = 2; Information[a]; {?? q, ?? q}; {Information[s], Information["s"]}; f[x_] := x ^ 2; g[f] ^:= 2; f::usage = "f[x] returns the square of x"; Information[f]; Length[Names["System`*"]] > 350; {\[Eta], \[CapitalGamma]\[Beta], Z\[Infinity], \[Angle]XYZ, \[FilledSquare]r, i\[Ellipsis]j}; SymbolName[a`b`x] // InputForm; ValueQ[True]"#,
      r#"False"#,
    );
  }
  #[test]
  fn quiet_3() {
    assert_case(r#"Quiet[URLFetch["https://", {}]]"#, r#"$Failed"#);
  }
  #[test]
  fn wr_l() {
    // BinaryReadList: write bytes to a temp file via BinaryWrite then read
    // them back as a flat list of integers.
    assert_case(
      r#"WrL[bytes_] := Module[{stream, res}, stream = OpenWrite[BinaryFormat -> True]; BinaryWrite[stream, bytes]; res = BinaryReadList[Close[stream], "Byte"]; res]; WrL[{1, 2, 3, 254, 255}]"#,
      r#"{1, 2, 3, 254, 255}"#,
    );
  }
}

mod json_import {
  use super::*;

  fn json_path() -> String {
    let manifest = env!("CARGO_MANIFEST_DIR");
    format!("{manifest}/tests/data/sample.json")
  }

  #[test]
  fn import_raw_json_returns_association() {
    let path = json_path();
    assert_eq!(
      interpret(&format!(r#"Import["{path}", "RawJSON"]"#)).unwrap(),
      "<|name -> Ada, age -> 36, tags -> {x, y}|>"
    );
  }

  #[test]
  fn import_json_returns_rules() {
    let path = json_path();
    assert_eq!(
      interpret(&format!(r#"Import["{path}", "JSON"]"#)).unwrap(),
      "{name -> Ada, age -> 36, tags -> {x, y}}"
    );
  }

  #[test]
  fn import_default_json_extension_returns_rules() {
    let path = json_path();
    assert_eq!(
      interpret(&format!(r#"Import["{path}"]"#)).unwrap(),
      "{name -> Ada, age -> 36, tags -> {x, y}}"
    );
  }
}

mod url_parse {
  use woxi::interpret;

  #[test]
  fn full_association() {
    assert_eq!(
      interpret(r#"URLParse["http://www.wolfram.com/solutions"]"#).unwrap(),
      "<|Scheme -> http, User -> None, Domain -> www.wolfram.com, Port -> None, Path -> {, solutions}, Query -> {}, Fragment -> None|>"
    );
    assert_eq!(
      interpret(
        r#"URLParse["https://user:pass@example.com:8080/a/b%20c?x=1&y=two#frag"]"#
      )
      .unwrap(),
      "<|Scheme -> https, User -> user:pass, Domain -> example.com, Port -> 8080, Path -> {, a, b c}, Query -> {x -> 1, y -> two}, Fragment -> frag|>"
    );
    // No path at all is an empty segment list; a bare "/" is {"", ""}.
    assert_eq!(
      interpret(r#"URLParse["ftp://example.org"]"#).unwrap(),
      "<|Scheme -> ftp, User -> None, Domain -> example.org, Port -> None, Path -> {}, Query -> {}, Fragment -> None|>"
    );
    assert_eq!(
      interpret(r#"URLParse["http://example.com/", "Path"]"#).unwrap(),
      "{, }"
    );
  }

  // Scheme and domain are lowercased; path keeps its case.
  #[test]
  fn case_normalization() {
    assert_eq!(
      interpret(r#"URLParse["HTTP://EXAMPLE.com/Path"]"#).unwrap(),
      "<|Scheme -> http, User -> None, Domain -> example.com, Port -> None, Path -> {, Path}, Query -> {}, Fragment -> None|>"
    );
  }

  // "+" means space only in the query; the path keeps it. Percent-escapes
  // decode in Path and Query but stay raw in Fragment/PathString/QueryString.
  #[test]
  fn decoding_rules() {
    assert_eq!(
      interpret(
        r#"URLParse["http://www.wolframalpha.com/input?i=100+USD+in+EUR", {"Path", "Query"}]"#
      )
      .unwrap(),
      "{{, input}, {i -> 100 USD in EUR}}"
    );
    assert_eq!(
      interpret(r#"URLParse["http://example.com/a+b", "Path"]"#).unwrap(),
      "{, a+b}"
    );
    assert_eq!(
      interpret(r#"URLParse["http://example.com/x?a=&b=%26c", "Query"]"#)
        .unwrap(),
      "{a -> , b -> &c}"
    );
    assert_eq!(
      interpret(r#"URLParse["http://example.com/a#b%20c", "Fragment"]"#)
        .unwrap(),
      "b%20c"
    );
    assert_eq!(
      interpret(
        r#"URLParse["/a%20b?x=%26", {"PathString", "QueryString", "AbsolutePath", "AbsoluteDomain"}]"#
      )
      .unwrap(),
      "{/a%20b, x=%26, /a%20b, }"
    );
  }

  #[test]
  fn query_quirks() {
    // A key without "=" maps to the empty string; repeated keys survive.
    assert_eq!(
      interpret(r#"URLParse["http://example.com/x?flag", "Query"]"#).unwrap(),
      "{flag -> }"
    );
    assert_eq!(
      interpret(r#"URLParse["http://example.com/x?a=1&a=2", "Query"]"#)
        .unwrap(),
      "{a -> 1, a -> 2}"
    );
  }

  // A scheme without "//" keeps the remainder as an opaque path; "//" without
  // a scheme still parses the authority; no scheme at all is pure path.
  #[test]
  fn scheme_forms() {
    assert_eq!(
      interpret(r#"URLParse["mailto:user@example.com"]"#).unwrap(),
      "<|Scheme -> mailto, User -> None, Domain -> None, Port -> None, Path -> {user@example.com}, Query -> {}, Fragment -> None|>"
    );
    assert_eq!(
      interpret(r#"URLParse["//example.com/x", "Domain"]"#).unwrap(),
      "example.com"
    );
    assert_eq!(
      interpret(r#"URLParse["example.com/x", "Path"]"#).unwrap(),
      "{example.com, x}"
    );
    assert_eq!(
      interpret(r#"URLParse["/relative/path?a=b"]"#).unwrap(),
      "<|Scheme -> None, User -> None, Domain -> None, Port -> None, Path -> {, relative, path}, Query -> {a -> b}, Fragment -> None|>"
    );
    assert_eq!(
      interpret(r#"URLParse[""]"#).unwrap(),
      "<|Scheme -> None, User -> None, Domain -> None, Port -> None, Path -> {}, Query -> {}, Fragment -> None|>"
    );
  }

  #[test]
  fn components() {
    assert_eq!(
      interpret(r#"URLParse["http://example.com/x", "PathString"]"#).unwrap(),
      "/x"
    );
    assert_eq!(
      interpret(r#"URLParse["http://example.com/x?a=1&b=2", "QueryString"]"#)
        .unwrap(),
      "a=1&b=2"
    );
    assert_eq!(
      interpret(
        r#"URLParse["http://example.com/x", {"Scheme", "PathString"}]"#
      )
      .unwrap(),
      "{http, /x}"
    );
    // QueryString is None (not "") when the URL has no query; PathString is "".
    assert_eq!(
      interpret(
        r#"URLParse["http://example.com", {"PathString", "QueryString"}]"#
      )
      .unwrap(),
      "{, None}"
    );
    // AbsolutePath drops query and fragment.
    assert_eq!(
      interpret(r#"URLParse["http://example.com/x?a=1#f", "AbsolutePath"]"#)
        .unwrap(),
      "http://example.com/x"
    );
    assert_eq!(
      interpret(r#"URLParse[URL["http://example.com/x"], "Domain"]"#).unwrap(),
      "example.com"
    );
  }

  #[test]
  fn all_components() {
    assert_eq!(
      interpret(r#"URLParse["https://user:pass@example.com/x", All]"#).unwrap(),
      "<|Scheme -> https, User -> user:pass, Domain -> example.com, Port -> None, Path -> {, x}, Query -> {}, Fragment -> None, PathString -> /x, QueryString -> None, Username -> user, Password -> pass, AbsolutePath -> https://user:pass@example.com/x, AbsoluteDomain -> https://user:pass@example.com|>"
    );
  }

  // A non-numeric port is $Failed; bad components and non-string input echo
  // the call (each with a message).
  #[test]
  fn error_forms() {
    assert_eq!(
      interpret(r#"URLParse["http://example.com:80x/"]"#).unwrap(),
      "$Failed"
    );
    assert_eq!(
      interpret(r#"URLParse["http://example.com/x", "Foo"]"#).unwrap(),
      "URLParse[http://example.com/x, Foo]"
    );
    assert_eq!(
      interpret(r#"URLParse["http://example.com/x", {"Domain", All}]"#)
        .unwrap(),
      "URLParse[http://example.com/x, {Domain, All}]"
    );
    assert_eq!(interpret("URLParse[42]").unwrap(), "URLParse[42]");
  }
}

mod file_size {
  use woxi::interpret;

  // FileSize gives Quantity[bytes, "Bytes"] with a Real magnitude (unlike
  // FileByteCount's plain Integer).
  #[test]
  fn existing_file() {
    let path = std::env::temp_dir().join("woxi_file_size_test.txt");
    std::fs::write(&path, "hello world").unwrap();
    let path_str = path.to_string_lossy().into_owned();
    assert_eq!(
      interpret(&format!(r#"FileSize["{path_str}"]"#)).unwrap(),
      "Quantity[11., Bytes]"
    );
    // The File["…"] wrapper is accepted too.
    assert_eq!(
      interpret(&format!(r#"FileSize[File["{path_str}"]]"#)).unwrap(),
      "Quantity[11., Bytes]"
    );
    std::fs::remove_file(&path).ok();
  }

  // Errors echo the call unevaluated (each with its own message tag):
  // ::fdnfnd for missing paths, ::fdir for directories, ::badfile for
  // non-string arguments.
  #[test]
  fn error_forms() {
    assert_eq!(
      interpret(r#"FileSize["/definitely/missing/file_xyz.txt"]"#).unwrap(),
      "FileSize[/definitely/missing/file_xyz.txt]"
    );
    let dir = std::env::temp_dir();
    let dir_str = dir.to_string_lossy();
    let dir_str = dir_str.trim_end_matches('/');
    assert_eq!(
      interpret(&format!(r#"FileSize["{dir_str}"]"#)).unwrap(),
      format!("FileSize[{dir_str}]")
    );
    assert_eq!(interpret("FileSize[42]").unwrap(), "FileSize[42]");
  }
}

mod echo_function {
  use woxi::interpret_with_stdout;

  // EchoFunction[f][expr] prints ">> f[expr]" and returns expr unchanged;
  // EchoFunction[label, f] prefixes the label; EchoFunction[] echoes the
  // expression itself.
  #[test]
  fn applies_and_returns() {
    let r = interpret_with_stdout("EchoFunction[Head][{1, 2, 3}]").unwrap();
    assert_eq!(r.result, "{1, 2, 3}");
    assert_eq!(r.stdout, ">> List\n");

    let r = interpret_with_stdout(r#"EchoFunction["lbl", Length][{1, 2, 3}]"#)
      .unwrap();
    assert_eq!(r.result, "{1, 2, 3}");
    assert_eq!(r.stdout, ">> lbl 3\n");

    let r = interpret_with_stdout("EchoFunction[][5]").unwrap();
    assert_eq!(r.result, "5");
    assert_eq!(r.stdout, ">> 5\n");

    // Pure-function heads work too.
    let r = interpret_with_stdout("EchoFunction[#^2 &][4]").unwrap();
    assert_eq!(r.result, "4");
    assert_eq!(r.stdout, ">> 16\n");
  }

  // The bare operator form stays symbolic.
  #[test]
  fn operator_form_is_symbolic() {
    let r = interpret_with_stdout("EchoFunction[Head]").unwrap();
    assert_eq!(r.result, "EchoFunction[Head]");
    assert_eq!(r.stdout, "");
  }
}

mod echo_label {
  use woxi::interpret_with_stdout;

  // EchoLabel[label][expr] is Echo[expr, label] in operator form.
  #[test]
  fn applies_and_returns() {
    let r = interpret_with_stdout(r#"EchoLabel["x"][42]"#).unwrap();
    assert_eq!(r.result, "42");
    assert_eq!(r.stdout, ">> x 42\n");
    // Non-string labels work too.
    let r = interpret_with_stdout("EchoLabel[42][7]").unwrap();
    assert_eq!(r.result, "7");
    assert_eq!(r.stdout, ">> 42 7\n");
  }

  #[test]
  fn operator_form_is_symbolic() {
    let r = interpret_with_stdout(r#"EchoLabel["x"]"#).unwrap();
    assert_eq!(r.result, "EchoLabel[x]");
    assert_eq!(r.stdout, "");
  }
}

mod find_list_tests {
  use super::*;

  // Each test gets its own directory: the tests in this module run in
  // parallel within one process, so a shared (pid-keyed) directory that
  // setup() wipes first races — one test deletes the files while another
  // is still reading them.
  fn setup(subdir: &str) -> String {
    let base = std::env::temp_dir().join(format!(
      "woxi_findlist_test_{}_{subdir}",
      std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    std::fs::write(
      base.join("sample.txt"),
      "alpha beta\ngamma delta\nALPHA epsilon\nzeta alpha eta\nlast line\n",
    )
    .unwrap();
    std::fs::write(base.join("second.txt"), "one alpha\ntwo beta\n").unwrap();
    base.to_str().unwrap().to_string()
  }

  // Literal, case-sensitive substring search per line; the third argument
  // caps the total count globally. All verified against wolframscript.
  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn matching_lines() {
    let b = setup("matching_lines");
    for (input, expected) in [
      (
        format!(r#"FindList["{b}/sample.txt", "alpha"]"#),
        "{alpha beta, zeta alpha eta}",
      ),
      (
        format!(r#"FindList["{b}/sample.txt", {{"alpha", "delta"}}]"#),
        "{alpha beta, gamma delta, zeta alpha eta}",
      ),
      (
        format!(r#"FindList[{{"{b}/sample.txt", "{b}/second.txt"}}, "alpha"]"#),
        "{alpha beta, zeta alpha eta, one alpha}",
      ),
      (
        format!(
          r#"FindList[{{"{b}/sample.txt", "{b}/second.txt"}}, "alpha", 2]"#
        ),
        "{alpha beta, zeta alpha eta}",
      ),
      (
        format!(r#"FindList["{b}/sample.txt", "ALPHA"]"#),
        "{ALPHA epsilon}",
      ),
      (
        format!(r#"FindList["{b}/sample.txt", "a b"]"#),
        "{alpha beta}",
      ),
      (format!(r#"FindList["{b}/sample.txt", "nosuch"]"#), "{}"),
      (format!(r#"FindList["{b}/sample.txt", "alpha", 0]"#), "{}"),
      (r#"FindList[{}, "alpha"]"#.to_string(), "{}"),
    ] {
      assert_eq!(interpret(&input).unwrap(), expected, "input: {input}");
    }
  }

  // Missing files emit `noopen` and yield $Failed — as the whole result
  // for a single file, as one element inside a file-list result.
  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn missing_files_and_messages() {
    let b = setup("missing_files_and_messages");
    let r =
      interpret_with_stdout(&format!(r#"FindList["{b}/nofile.txt", "alpha"]"#))
        .unwrap();
    assert_eq!(r.result, "$Failed");
    assert!(r.warnings.iter().any(|w| {
      w.contains(&format!("FindList::noopen: Cannot open {b}/nofile.txt."))
    }));

    let r = interpret_with_stdout(&format!(
      r#"FindList[{{"{b}/sample.txt", "{b}/nofile.txt"}}, "alpha"]"#
    ))
    .unwrap();
    assert_eq!(r.result, "{alpha beta, zeta alpha eta, $Failed}");
    assert!(r.warnings.iter().any(|w| {
      w.contains(&format!("FindList::noopen: Cannot open {b}/nofile.txt."))
    }));

    // Argument validation: counts, types, and option positions (message
    // displays use OutputForm — strings without quotes).
    let r = interpret_with_stdout("FindList[]").unwrap();
    assert_eq!(r.result, "$Failed");
    assert!(r.warnings.iter().any(|w| w.contains(
      "FindList::argt: FindList called with 0 arguments; 2 or 3 arguments are expected."
    )));

    let r = interpret_with_stdout(r#"FindList["file.txt"]"#).unwrap();
    assert_eq!(r.result, "$Failed");
    assert!(r.warnings.iter().any(|w| w.contains(
      "FindList::argtu: FindList called with 1 argument; 2 or 3 arguments are expected."
    )));

    let r = interpret_with_stdout(r#"FindList[5, "alpha"]"#).unwrap();
    assert_eq!(r.result, "$Failed");
    assert!(r.warnings.iter().any(|w| w.contains(
      "FindList::stream: 5 is not a string, SocketObject, InputStream[ ] or OutputStream[ ]."
    )));

    let r = interpret_with_stdout(r#"FindList["f.txt", 5]"#).unwrap();
    assert_eq!(r.result, "$Failed");
    assert!(r.warnings.iter().any(|w| w.contains(
      "FindList::strs: A string or nonempty list of strings is expected at position 2 in FindList[f.txt, 5]."
    )));

    let r = interpret_with_stdout(r#"FindList["f.txt", {}]"#).unwrap();
    assert_eq!(r.result, "$Failed");
    assert!(r.warnings.iter().any(|w| w.contains(
      "FindList::strs: A string or nonempty list of strings is expected at position 2 in FindList[f.txt, {}]."
    )));

    let r = interpret_with_stdout(r#"FindList["f.txt", "alpha", -1]"#).unwrap();
    assert_eq!(r.result, "$Failed");
    assert!(r.warnings.iter().any(|w| w.contains(
      "FindList::intnm: Non-negative machine-sized integer expected at position 3 in FindList[f.txt, alpha, -1]."
    )));

    let r =
      interpret_with_stdout(r#"FindList["f.txt", "alpha", 2, 9]"#).unwrap();
    assert_eq!(r.result, "$Failed");
    assert!(r.warnings.iter().any(|w| w.contains(
      "FindList::nonopt: Options expected (instead of 9) beyond position 3 in FindList[f.txt, alpha, 2, 9]. An option must be a rule or a list of rules."
    )));
  }
}
