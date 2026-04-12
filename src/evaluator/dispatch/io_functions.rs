#[allow(unused_imports)]
use super::*;

// Virtual working-directory stack used by SetDirectory / ResetDirectory.
//
// We deliberately do NOT call `std::env::set_current_dir` here: that mutates
// process-wide state, and cargo runs tests in parallel threads within a
// single process. Mutating the real CWD from one test races against any
// other test that resolves relative paths (Import, FileNames, etc.), causing
// flaky failures in CI. Instead we track a per-thread virtual stack; the top
// of the stack is what `Directory[]` reports, and the process CWD is used as
// the fallback when the stack is empty.
#[cfg(not(target_arch = "wasm32"))]
thread_local! {
  static DIRECTORY_STACK: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(not(target_arch = "wasm32"))]
fn virtual_current_dir() -> String {
  DIRECTORY_STACK
    .with(|s| s.borrow().last().cloned())
    .unwrap_or_else(|| {
      std::env::current_dir()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default()
    })
}

pub fn dispatch_io_functions(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    // Streams[] — return list of open streams (stdout and stderr)
    "Streams" if args.is_empty() => {
      return Some(Ok(Expr::List(vec![
        Expr::FunctionCall {
          name: "OutputStream".to_string(),
          args: vec![Expr::String("stdout".to_string()), Expr::Integer(1)],
        },
        Expr::FunctionCall {
          name: "OutputStream".to_string(),
          args: vec![Expr::String("stderr".to_string()), Expr::Integer(2)],
        },
      ])));
    }
    // Streams["name"] — filter streams by name
    "Streams" if args.len() == 1 => {
      if let Expr::String(name_filter) = &args[0] {
        let all_streams = [("stdout", 1), ("stderr", 2)];
        let matching: Vec<Expr> = all_streams
          .iter()
          .filter(|(n, _)| *n == name_filter.as_str())
          .map(|(n, id)| Expr::FunctionCall {
            name: "OutputStream".to_string(),
            args: vec![Expr::String(n.to_string()), Expr::Integer(*id)],
          })
          .collect();
        return Some(Ok(Expr::List(matching)));
      }
      return Some(Ok(Expr::List(vec![])));
    }
    // ReadList[source] or ReadList[source, type] or ReadList[source, type, n]
    "ReadList" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::string_ast::read_list_ast(args));
    }
    // ReadString["file"] — read file contents as a string
    #[cfg(not(target_arch = "wasm32"))]
    "ReadString" if args.len() == 1 => {
      let filename = match &args[0] {
        Expr::String(s) => s.clone(),
        _ => {
          return Some(Ok(Expr::FunctionCall {
            name: "ReadString".to_string(),
            args: args.to_vec(),
          }));
        }
      };
      let content = match std::fs::read_to_string(&filename) {
        Ok(c) => c,
        Err(_) => {
          crate::emit_message(&format!(
            "ReadString::noopen: Cannot open {}.",
            filename
          ));
          return Some(Ok(Expr::Identifier("$Failed".to_string())));
        }
      };
      return Some(Ok(Expr::String(content)));
    }
    // Get[file] — read and evaluate a file, returning the last result
    #[cfg(not(target_arch = "wasm32"))]
    "Get" if args.len() == 1 => {
      let filename = match &args[0] {
        Expr::String(s) => s.clone(),
        Expr::Identifier(s) => s.clone(),
        _ => {
          return Some(Ok(Expr::FunctionCall {
            name: "Get".to_string(),
            args: args.to_vec(),
          }));
        }
      };
      let content = match std::fs::read_to_string(&filename) {
        Ok(c) => c,
        Err(_) => {
          crate::emit_message(&format!(
            "Get::noopen: Cannot open {}.",
            filename
          ));
          return Some(Ok(Expr::Identifier("$Failed".to_string())));
        }
      };
      // Use interpret to evaluate the file content (handles all node types
      // including FunctionDefinition, Expression, etc.)
      let result_str = match crate::interpret(&content) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
      };
      let result = crate::syntax::string_to_expr(&result_str)
        .unwrap_or(Expr::Identifier(result_str));
      return Some(Ok(result));
    }
    // Put[expr1, expr2, ..., "file"] — write expressions to a file
    #[cfg(not(target_arch = "wasm32"))]
    "Put" if !args.is_empty() => {
      let filename = match args.last().unwrap() {
        Expr::String(s) => s.clone(),
        _ => {
          return Some(Ok(Expr::FunctionCall {
            name: "Put".to_string(),
            args: args.to_vec(),
          }));
        }
      };
      let exprs = &args[..args.len() - 1];
      let content = exprs
        .iter()
        .map(crate::syntax::expr_to_string)
        .collect::<Vec<_>>()
        .join("\n");
      let to_write = if exprs.is_empty() {
        String::new()
      } else {
        format!("{}\n", content)
      };
      match std::fs::write(&filename, to_write) {
        Ok(_) => return Some(Ok(Expr::Identifier("Null".to_string()))),
        Err(_e) => {
          crate::emit_message(&format!(
            "Put::noopen: Cannot open {}.",
            filename
          ));
          return Some(Ok(Expr::Identifier("$Failed".to_string())));
        }
      }
    }
    // PutAppend[expr1, expr2, ..., "file"] — append expressions to a file
    #[cfg(not(target_arch = "wasm32"))]
    "PutAppend" if !args.is_empty() => {
      let filename = match args.last().unwrap() {
        Expr::String(s) => s.clone(),
        _ => {
          return Some(Ok(Expr::FunctionCall {
            name: "PutAppend".to_string(),
            args: args.to_vec(),
          }));
        }
      };
      let exprs = &args[..args.len() - 1];
      let content = exprs
        .iter()
        .map(crate::syntax::expr_to_string)
        .collect::<Vec<_>>()
        .join("\n");
      if !exprs.is_empty() {
        use std::io::Write;
        let to_write = format!("{}\n", content);
        match std::fs::OpenOptions::new()
          .create(true)
          .append(true)
          .open(&filename)
        {
          Ok(mut file) => {
            if file.write_all(to_write.as_bytes()).is_err() {
              crate::emit_message(&format!(
                "PutAppend::noopen: Cannot open {}.",
                filename
              ));
              return Some(Ok(Expr::Identifier("$Failed".to_string())));
            }
          }
          Err(_) => {
            crate::emit_message(&format!(
              "PutAppend::noopen: Cannot open {}.",
              filename
            ));
            return Some(Ok(Expr::Identifier("$Failed".to_string())));
          }
        }
      }
      return Some(Ok(Expr::Identifier("Null".to_string())));
    }
    #[cfg(not(target_arch = "wasm32"))]
    "Export" if args.len() >= 2 => {
      let filename = match &args[0] {
        Expr::String(s) => s.clone(),
        other => {
          return Some(Err(InterpreterError::EvaluationError(format!(
            "Export: first argument must be a filename string, got {}",
            crate::syntax::expr_to_string(other)
          ))));
        }
      };
      // Handle Image export
      if let Expr::Image {
        width,
        height,
        channels,
        data,
        ..
      } = &args[1]
      {
        if let Err(e) = crate::functions::image_ast::export_image(
          &filename, *width, *height, *channels, data,
        ) {
          return Some(Err(e));
        }
        return Some(Ok(Expr::String(filename)));
      }
      // Determine the export format from the explicit third argument or,
      // failing that, from the filename extension.
      let explicit_fmt = args.get(2).and_then(|a| {
        if let Expr::String(s) = a {
          Some(s.to_ascii_uppercase())
        } else {
          None
        }
      });
      let ext_fmt = std::path::Path::new(&filename)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_uppercase());
      let fmt = explicit_fmt.or(ext_fmt).unwrap_or_default();

      if fmt == "XLSX" {
        if let Err(e) =
          crate::functions::xlsx_ast::xlsx_export_file(&filename, &args[1])
        {
          return Some(Err(e));
        }
        return Some(Ok(Expr::String(filename)));
      }

      if fmt == "PDF" {
        let svg = expr_to_svg(&args[1]);
        match svg_to_pdf_bytes(&svg) {
          Ok(pdf_bytes) => {
            if let Err(e) = std::fs::write(&filename, &pdf_bytes).map_err(|e| {
              InterpreterError::EvaluationError(format!("Export: {e}"))
            }) {
              return Some(Err(e));
            }
            return Some(Ok(Expr::String(filename)));
          }
          Err(e) => return Some(Err(e)),
        }
      }

      // Raster image formats: rasterize the SVG and write via the image crate.
      if matches!(
        fmt.as_str(),
        "PNG" | "JPG" | "JPEG" | "GIF" | "BMP" | "TIF" | "TIFF"
      ) {
        // Parse ImageResolution option (default 96 DPI to match
        // usvg's default output resolution).
        let mut dpi: f64 = 96.0;
        for opt in &args[2..] {
          if let Expr::Rule {
            pattern,
            replacement,
          } = opt
            && let Expr::Identifier(k) = pattern.as_ref()
            && k == "ImageResolution"
          {
            match replacement.as_ref() {
              Expr::Integer(n) => dpi = *n as f64,
              Expr::Real(f) => dpi = *f,
              _ => {
                return Some(Err(InterpreterError::EvaluationError(
                  "Export: ImageResolution must be a numeric value".into(),
                )));
              }
            }
          }
        }
        let svg = expr_to_svg(&args[1]);
        match crate::functions::image_ast::rasterize_svg(&svg, dpi) {
          Ok(Expr::Image {
            width,
            height,
            channels,
            ref data,
            ..
          }) => {
            if let Err(e) = crate::functions::image_ast::export_image(
              &filename, width, height, channels, data,
            ) {
              return Some(Err(e));
            }
            return Some(Ok(Expr::String(filename)));
          }
          Ok(_) => unreachable!("rasterize_svg returns Expr::Image"),
          Err(e) => return Some(Err(e)),
        }
      }
      // The second argument has already been evaluated, which triggers
      // capture_graphics() for Plot expressions.  Grab the SVG.
      let content = match &args[1] {
        Expr::Graphics { svg: svg_data, .. } => svg_data.clone(),
        Expr::Identifier(s) if s == "-Graphics-" || s == "-Graphics3D-" => {
          match crate::get_captured_graphics().ok_or_else(|| {
            InterpreterError::EvaluationError(
              "Export: no graphics to export".into(),
            )
          }) {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
          }
        }
        Expr::String(s) => s.clone(),
        other => crate::syntax::expr_to_string(other),
      };
      if let Err(e) = std::fs::write(&filename, &content)
        .map_err(|e| InterpreterError::EvaluationError(format!("Export: {e}")))
      {
        return Some(Err(e));
      }
      return Some(Ok(Expr::String(filename)));
    }
    "ExportString" if args.len() == 2 => {
      // ExportString[expr, "format"] - return string representation
      let format_str = match &args[1] {
        Expr::String(s) => s.clone(),
        _ => {
          // Return unevaluated for non-string format
          return Some(Ok(Expr::FunctionCall {
            name: "ExportString".to_string(),
            args: args.to_vec(),
          }));
        }
      };
      if format_str == "SVG" || format_str == "PDF" {
        let svg = expr_to_svg(&args[0]);
        if format_str == "PDF" {
          #[cfg(not(target_arch = "wasm32"))]
          {
            match svg_to_pdf_bytes(&svg) {
              Ok(pdf_bytes) => {
                // Return raw PDF bytes as a String (binary content)
                let pdf_str =
                  pdf_bytes.into_iter().map(|b| b as char).collect::<String>();
                return Some(Ok(Expr::String(pdf_str)));
              }
              Err(e) => return Some(Err(e)),
            }
          }
          #[cfg(target_arch = "wasm32")]
          {
            return Some(Ok(Expr::FunctionCall {
              name: "ExportString".to_string(),
              args: args.to_vec(),
            }));
          }
        }
        return Some(Ok(Expr::String(svg)));
      }
      if format_str != "SVG" && format_str != "PDF" {
        // Return unevaluated for unsupported formats
        return Some(Ok(Expr::FunctionCall {
          name: "ExportString".to_string(),
          args: args.to_vec(),
        }));
      }
      unreachable!()
    }
    #[cfg(not(target_arch = "wasm32"))]
    "Find" if args.len() == 2 => {
      // Find[stream_or_file, "text"] - find first line containing text
      let filename = match &args[0] {
        Expr::String(s) => s.clone(),
        _ => {
          return Some(Ok(Expr::FunctionCall {
            name: "Find".to_string(),
            args: args.to_vec(),
          }));
        }
      };
      let search = match &args[1] {
        Expr::String(s) => s.clone(),
        _ => {
          return Some(Err(InterpreterError::EvaluationError(
            "Find: second argument must be a string".into(),
          )));
        }
      };
      let content = match std::fs::read_to_string(&filename)
        .map_err(|e| InterpreterError::EvaluationError(format!("Find: {e}")))
      {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
      };
      for line in content.lines() {
        if line.contains(&search) {
          return Some(Ok(Expr::String(line.to_string())));
        }
      }
      return Some(Ok(Expr::Identifier("EndOfFile".to_string())));
    }
    #[cfg(not(target_arch = "wasm32"))]
    "CreateFile" => {
      let filename_opt = if args.is_empty() {
        None
      } else if let Expr::String(s) = &args[0] {
        Some(s.clone())
      } else {
        let s = expr_to_raw_string(&args[0]);
        Some(s)
      };
      return Some(match crate::utils::create_file(filename_opt) {
        Ok(path) => Ok(Expr::String(path.to_string_lossy().into_owned())),
        Err(err) => Err(InterpreterError::EvaluationError(err.to_string())),
      });
    }
    #[cfg(not(target_arch = "wasm32"))]
    "Directory" if args.is_empty() => {
      return Some(Ok(Expr::String(virtual_current_dir())));
    }
    // DirectoryName["path"] or DirectoryName["path", n]
    "DirectoryName" if args.len() == 1 || args.len() == 2 => {
      let path_str = match &args[0] {
        Expr::String(s) => s.clone(),
        _ => {
          return Some(Ok(Expr::FunctionCall {
            name: "DirectoryName".to_string(),
            args: args.to_vec(),
          }));
        }
      };
      let n = if args.len() == 2 {
        match &args[1] {
          Expr::Integer(i) if *i >= 1 => *i as usize,
          Expr::Integer(_) => {
            crate::emit_message(
              "DirectoryName::intpm: Positive machine-sized integer expected at position 2 in DirectoryName.",
            );
            return Some(Ok(Expr::FunctionCall {
              name: "DirectoryName".to_string(),
              args: args.to_vec(),
            }));
          }
          _ => {
            return Some(Ok(Expr::FunctionCall {
              name: "DirectoryName".to_string(),
              args: args.to_vec(),
            }));
          }
        }
      } else {
        1
      };

      let mut result = path_str;
      for _ in 0..n {
        if result.is_empty() {
          break;
        }
        // "/" has no parent
        let trimmed = result.trim_end_matches('/');
        if trimmed.is_empty() {
          // input was "/" or "///" etc.
          result = String::new();
          break;
        }
        // Find the last separator
        if let Some(pos) = trimmed.rfind('/') {
          result = trimmed[..=pos].to_string();
        } else {
          result = String::new();
          break;
        }
      }
      return Some(Ok(Expr::String(result)));
    }
    "FileNameJoin" if args.len() == 1 => {
      if let Expr::List(parts) = &args[0] {
        let segments: Vec<String> = parts
          .iter()
          .filter_map(|e| {
            if let Expr::String(s) = e {
              Some(s.clone())
            } else {
              None
            }
          })
          .collect();
        if segments.len() == parts.len() {
          let mut path = std::path::PathBuf::new();
          for seg in &segments {
            path.push(seg);
          }
          return Some(Ok(Expr::String(path.to_string_lossy().into_owned())));
        }
      }
      return Some(Ok(Expr::FunctionCall {
        name: "FileNameJoin".to_string(),
        args: args.to_vec(),
      }));
    }
    "FileNameSplit" if args.len() == 1 => {
      if let Expr::String(s) = &args[0] {
        if s.is_empty() {
          return Some(Ok(Expr::List(vec![])));
        }
        let parts: Vec<Expr> = s
          .split('/')
          .collect::<Vec<&str>>()
          .into_iter()
          .enumerate()
          .filter(|(i, part)| !(*i > 0 && part.is_empty()))
          .map(|(_, part)| Expr::String(part.to_string()))
          .collect();
        return Some(Ok(Expr::List(parts)));
      }
      return Some(Ok(Expr::FunctionCall {
        name: "FileNameSplit".to_string(),
        args: args.to_vec(),
      }));
    }
    "ExpandFileName" if args.len() == 1 => {
      if let Expr::String(s) = &args[0] {
        let expanded = if s.starts_with('~') {
          if let Ok(home) = std::env::var("HOME") {
            format!("{}{}", home, &s[1..])
          } else {
            s.clone()
          }
        } else {
          s.clone()
        };
        let path = std::path::PathBuf::from(&expanded);
        let abs = if path.is_relative() {
          if let Ok(cwd) = std::env::current_dir() {
            cwd.join(&path)
          } else {
            path
          }
        } else {
          path
        };
        // Normalize path components (resolve . and ..)
        let mut components = Vec::new();
        for component in abs.components() {
          match component {
            std::path::Component::ParentDir => {
              components.pop();
            }
            std::path::Component::CurDir => {}
            _ => components.push(component),
          }
        }
        let normalized: std::path::PathBuf = components.iter().collect();
        return Some(Ok(Expr::String(
          normalized.to_string_lossy().into_owned(),
        )));
      }
      return Some(Ok(Expr::FunctionCall {
        name: "ExpandFileName".to_string(),
        args: args.to_vec(),
      }));
    }
    "URLBuild" if args.len() == 1 || args.len() == 2 => {
      // URLBuild["url"] => "url"
      // URLBuild[{"base", "path1", ...}] => "base/path1/..."
      // URLBuild[{"base", ...}, {"key" -> "val", ...}] => "base/...?key=val&..."
      let parts = match &args[0] {
        Expr::String(s) => vec![s.clone()],
        Expr::List(items) => {
          let mut strs = Vec::new();
          for item in items {
            match item {
              Expr::String(s) => strs.push(s.clone()),
              other => strs.push(crate::syntax::expr_to_string(other)),
            }
          }
          strs
        }
        _ => {
          return Some(Ok(Expr::FunctionCall {
            name: "URLBuild".to_string(),
            args: args.to_vec(),
          }));
        }
      };

      // Build base URL from parts
      let mut url = if parts.is_empty() {
        String::new()
      } else {
        let base = parts[0].trim_end_matches('/').to_string();
        let mut result = base;
        for part in &parts[1..] {
          let segment = part.trim_matches('/');
          if !segment.is_empty() {
            result.push('/');
            result.push_str(segment);
          }
        }
        result
      };

      // Add query parameters
      if args.len() == 2 {
        let query_pairs: Vec<(String, String)> = match &args[1] {
          Expr::List(items) => {
            let mut pairs = Vec::new();
            for item in items {
              match item {
                Expr::Rule {
                  pattern,
                  replacement,
                }
                | Expr::RuleDelayed {
                  pattern,
                  replacement,
                } => {
                  let key = match pattern.as_ref() {
                    Expr::String(s) => s.clone(),
                    other => crate::syntax::expr_to_string(other),
                  };
                  let val = match replacement.as_ref() {
                    Expr::String(s) => s.clone(),
                    other => crate::syntax::expr_to_string(other),
                  };
                  pairs.push((key, val));
                }
                _ => {}
              }
            }
            pairs
          }
          _ => vec![],
        };
        if !query_pairs.is_empty() {
          url.push('?');
          for (i, (key, val)) in query_pairs.iter().enumerate() {
            if i > 0 {
              url.push('&');
            }
            url.push_str(key);
            url.push('=');
            url.push_str(val);
          }
        }
      }

      return Some(Ok(Expr::String(url)));
    }
    // OpenRead[file] — open a file for reading, return InputStream[name, id]
    #[cfg(not(target_arch = "wasm32"))]
    "OpenRead" if args.len() == 1 => {
      let filename = match &args[0] {
        Expr::String(s) => s.clone(),
        other => {
          return Some(Ok(Expr::FunctionCall {
            name: "OpenRead".to_string(),
            args: vec![other.clone()],
          }));
        }
      };
      if !std::path::Path::new(&filename).exists() {
        crate::emit_message(&format!(
          "OpenRead::noopen: Cannot open {}.",
          filename
        ));
        return Some(Ok(Expr::Identifier("$Failed".to_string())));
      }
      let id = crate::register_stream(
        filename.clone(),
        crate::StreamKind::FileStream(filename.clone()),
      );
      return Some(Ok(Expr::FunctionCall {
        name: "InputStream".to_string(),
        args: vec![Expr::String(filename), Expr::Integer(id as i128)],
      }));
    }
    // OpenWrite[file] — open a file for writing, return OutputStream[name, id]
    #[cfg(not(target_arch = "wasm32"))]
    "OpenWrite" if args.len() <= 1 => {
      let filename = if args.is_empty() {
        let path = match crate::utils::create_file(None)
          .map_err(|e| InterpreterError::EvaluationError(e.to_string()))
        {
          Ok(v) => v,
          Err(e) => return Some(Err(e)),
        };
        path.to_string_lossy().into_owned()
      } else {
        match &args[0] {
          Expr::String(s) => s.clone(),
          other => {
            return Some(Ok(Expr::FunctionCall {
              name: "OpenWrite".to_string(),
              args: vec![other.clone()],
            }));
          }
        }
      };
      // Create or truncate the file
      if let Err(e) = std::fs::File::create(&filename).map_err(|e| {
        InterpreterError::EvaluationError(format!(
          "OpenWrite: cannot open {}: {}",
          filename, e
        ))
      }) {
        return Some(Err(e));
      }
      let id = crate::register_stream(
        filename.clone(),
        crate::StreamKind::FileStream(filename.clone()),
      );
      return Some(Ok(Expr::FunctionCall {
        name: "OutputStream".to_string(),
        args: vec![Expr::String(filename), Expr::Integer(id as i128)],
      }));
    }
    // OpenAppend[file] — open a file for appending, return OutputStream[name, id]
    #[cfg(not(target_arch = "wasm32"))]
    "OpenAppend" if args.len() == 1 => {
      let filename = match &args[0] {
        Expr::String(s) => s.clone(),
        other => {
          return Some(Ok(Expr::FunctionCall {
            name: "OpenAppend".to_string(),
            args: vec![other.clone()],
          }));
        }
      };
      // Open for appending (create if not exists)
      if let Err(e) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&filename)
        .map_err(|e| {
          InterpreterError::EvaluationError(format!(
            "OpenAppend: cannot open {}: {}",
            filename, e
          ))
        })
      {
        return Some(Err(e));
      }
      let id = crate::register_stream(
        filename.clone(),
        crate::StreamKind::FileStream(filename.clone()),
      );
      return Some(Ok(Expr::FunctionCall {
        name: "OutputStream".to_string(),
        args: vec![Expr::String(filename), Expr::Integer(id as i128)],
      }));
    }
    // StringToStream["text"] — create an input stream from a string
    "StringToStream" if args.len() == 1 => {
      let text = match &args[0] {
        Expr::String(s) => s.clone(),
        other => {
          return Some(Err(InterpreterError::EvaluationError(format!(
            "StringToStream: argument must be a string, got {}",
            crate::syntax::expr_to_string(other)
          ))));
        }
      };
      let id = crate::register_stream(
        "String".to_string(),
        crate::StreamKind::StringStream(text),
      );
      return Some(Ok(Expr::FunctionCall {
        name: "InputStream".to_string(),
        args: vec![
          Expr::String("String".to_string()),
          Expr::Integer(id as i128),
        ],
      }));
    }
    // Close[stream] — close an open stream
    "Close" if args.len() == 1 => {
      // Extract stream ID from InputStream[name, id] or OutputStream[name, id]
      match &args[0] {
        Expr::FunctionCall {
          name: stream_head,
          args: stream_args,
        } if (stream_head == "InputStream"
          || stream_head == "OutputStream")
          && stream_args.len() == 2 =>
        {
          let id = match &stream_args[1] {
            Expr::Integer(n) => *n as usize,
            _ => {
              return Some(Ok(Expr::FunctionCall {
                name: "Close".to_string(),
                args: args.to_vec(),
              }));
            }
          };
          match crate::close_stream(id) {
            Some(name) => return Some(Ok(Expr::Identifier(name))),
            None => {
              let stream_str = crate::syntax::expr_to_string(&args[0]);
              crate::emit_message(&format!("{} is not open.", stream_str));
              return Some(Ok(Expr::FunctionCall {
                name: "Close".to_string(),
                args: args.to_vec(),
              }));
            }
          }
        }
        Expr::String(s) => {
          crate::emit_message(&format!("{} is not open.", s));
          return Some(Ok(Expr::FunctionCall {
            name: "Close".to_string(),
            args: args.to_vec(),
          }));
        }
        _ => {
          return Some(Ok(Expr::FunctionCall {
            name: "Close".to_string(),
            args: args.to_vec(),
          }));
        }
      }
    }
    // StreamPosition[stream] — get the current position of a stream
    "StreamPosition" if args.len() == 1 => {
      let stream = &args[0];
      match stream {
        Expr::FunctionCall {
          name: stream_head,
          args: stream_args,
        } if (stream_head == "InputStream"
          || stream_head == "OutputStream")
          && stream_args.len() == 2 =>
        {
          if let Expr::Integer(id) = &stream_args[1] {
            match crate::get_stream_position(*id as usize) {
              Some(pos) => return Some(Ok(Expr::Integer(pos as i128))),
              None => {
                let stream_str = crate::syntax::expr_to_string(stream);
                crate::emit_message(&format!(
                  "StreamPosition::openx: {} is not open.",
                  stream_str
                ));
                return Some(Ok(Expr::FunctionCall {
                  name: "StreamPosition".to_string(),
                  args: args.to_vec(),
                }));
              }
            }
          } else {
            return Some(Ok(Expr::FunctionCall {
              name: "StreamPosition".to_string(),
              args: args.to_vec(),
            }));
          }
        }
        Expr::String(s) => {
          crate::emit_message(&format!(
            "StreamPosition::openx: {} is not open.",
            s
          ));
          return Some(Ok(Expr::FunctionCall {
            name: "StreamPosition".to_string(),
            args: args.to_vec(),
          }));
        }
        _ => {
          return Some(Ok(Expr::FunctionCall {
            name: "StreamPosition".to_string(),
            args: args.to_vec(),
          }));
        }
      }
    }
    // SetStreamPosition[stream, pos] — set the current position of a stream
    "SetStreamPosition" if args.len() == 2 => {
      let stream = &args[0];
      let pos = match &args[1] {
        Expr::Integer(n) => *n as usize,
        _ => {
          return Some(Ok(Expr::FunctionCall {
            name: "SetStreamPosition".to_string(),
            args: args.to_vec(),
          }));
        }
      };
      match stream {
        Expr::FunctionCall {
          name: stream_head,
          args: stream_args,
        } if (stream_head == "InputStream"
          || stream_head == "OutputStream")
          && stream_args.len() == 2 =>
        {
          if let Expr::Integer(id) = &stream_args[1] {
            if crate::is_stream_open(*id as usize) {
              crate::set_stream_position(*id as usize, pos);
              return Some(Ok(Expr::Integer(pos as i128)));
            } else {
              let stream_str = crate::syntax::expr_to_string(stream);
              crate::emit_message(&format!(
                "SetStreamPosition::openx: {} is not open.",
                stream_str
              ));
              return Some(Ok(Expr::FunctionCall {
                name: "SetStreamPosition".to_string(),
                args: args.to_vec(),
              }));
            }
          } else {
            return Some(Ok(Expr::FunctionCall {
              name: "SetStreamPosition".to_string(),
              args: args.to_vec(),
            }));
          }
        }
        _ => {
          return Some(Ok(Expr::FunctionCall {
            name: "SetStreamPosition".to_string(),
            args: args.to_vec(),
          }));
        }
      }
    }
    // ReadLine[stream] — read one line from a stream
    // ReadLine["file"] — read first line from a file
    #[cfg(not(target_arch = "wasm32"))]
    "ReadLine" if args.len() == 1 => {
      let (content, position, stream_id) = match &args[0] {
        Expr::String(path) => {
          // ReadLine["file"] - read first line from file directly
          match std::fs::read_to_string(path) {
            Ok(content) => (content, 0usize, None),
            Err(_) => {
              crate::emit_message(&format!(
                "OpenRead::noopen: Cannot open {}.",
                path
              ));
              return Some(Ok(Expr::Identifier("$Failed".to_string())));
            }
          }
        }
        Expr::FunctionCall {
          name: stream_head,
          args: stream_args,
        } if stream_head == "InputStream" && stream_args.len() == 2 => {
          if let Expr::Integer(id) = &stream_args[1] {
            let id = *id as usize;
            match crate::get_stream_content(id) {
              Some((content, pos)) => (content, pos, Some(id)),
              None => {
                return Some(Ok(Expr::Identifier("EndOfFile".to_string())));
              }
            }
          } else {
            return Some(Ok(Expr::FunctionCall {
              name: "ReadLine".to_string(),
              args: args.to_vec(),
            }));
          }
        }
        _ => {
          return Some(Ok(Expr::FunctionCall {
            name: "ReadLine".to_string(),
            args: args.to_vec(),
          }));
        }
      };

      let remaining = &content[position.min(content.len())..];
      if remaining.is_empty() {
        return Some(Ok(Expr::Identifier("EndOfFile".to_string())));
      }

      // Find end of line
      let (line, advance) = if let Some(idx) = remaining.find('\n') {
        (&remaining[..idx], idx + 1)
      } else {
        (remaining, remaining.len())
      };

      let result = Expr::String(line.to_string());

      // Advance position if it's a stream
      if let Some(id) = stream_id {
        crate::advance_stream_position(id, position + advance);
      }

      return Some(Ok(result));
    }
    // Read[stream] or Read[stream, type] — read from a stream
    "Read" if !args.is_empty() && args.len() <= 2 => {
      let stream = &args[0];
      let stream_id = match stream {
        Expr::FunctionCall {
          name: stream_head,
          args: stream_args,
        } if (stream_head == "InputStream"
          || stream_head == "OutputStream")
          && stream_args.len() == 2 =>
        {
          if let Expr::Integer(id) = &stream_args[1] {
            Some(*id as usize)
          } else {
            None
          }
        }
        _ => None,
      };

      if let Some(id) = stream_id
        && let Some((content, position)) = crate::get_stream_content(id)
      {
        let remaining = &content[position.min(content.len())..];

        // Determine the read type
        let read_type = if args.len() == 2 {
          &args[1]
        } else {
          &Expr::Identifier("Expression".to_string())
        };

        // Handle list of types: Read[stream, {type1, type2, ...}]
        if let Expr::List(types) = read_type {
          let mut results = Vec::new();
          let mut current_pos = position;
          for t in types {
            let rem = &content[current_pos.min(content.len())..];
            let (val, advance) = read_single_type(rem, t);
            current_pos += advance;
            results.push(val);
          }
          crate::advance_stream_position(id, current_pos);
          return Some(Ok(Expr::List(results)));
        }

        let (result, advance) = read_single_type(remaining, read_type);
        crate::advance_stream_position(id, position + advance);
        return Some(Ok(result));
      }

      return Some(Ok(Expr::FunctionCall {
        name: "Read".to_string(),
        args: args.to_vec(),
      }));
    }
    // Write[stream, expr1, expr2, ...] — write expressions to a stream in OutputForm
    #[cfg(not(target_arch = "wasm32"))]
    "Write" if args.len() >= 2 => {
      let stream = &args[0];
      let file_path = match stream {
        Expr::FunctionCall {
          name: stream_head,
          args: stream_args,
        } if (stream_head == "OutputStream"
          || stream_head == "InputStream")
          && stream_args.len() == 2 =>
        {
          if let Expr::Integer(id) = &stream_args[1] {
            let stream_id = *id as usize;
            crate::STREAM_REGISTRY.with(|reg| {
              let registry = reg.borrow();
              registry.get(&stream_id).and_then(|s| match &s.kind {
                crate::StreamKind::FileStream(path) => Some(path.clone()),
                _ => None,
              })
            })
          } else {
            None
          }
        }
        Expr::String(path) => Some(path.clone()),
        _ => None,
      };

      if let Some(path) = file_path {
        use std::io::Write;
        let mut file = match std::fs::OpenOptions::new()
          .create(true)
          .append(true)
          .open(&path)
          .map_err(|e| {
            InterpreterError::EvaluationError(format!(
              "Write: cannot open {}: {}",
              path, e
            ))
          }) {
          Ok(v) => v,
          Err(e) => return Some(Err(e)),
        };
        let mut content = String::new();
        for arg in &args[1..] {
          content.push_str(&crate::syntax::expr_to_string(arg));
        }
        content.push('\n');
        if let Err(e) = file.write_all(content.as_bytes()).map_err(|e| {
          InterpreterError::EvaluationError(format!(
            "Write: write error: {}",
            e
          ))
        }) {
          return Some(Err(e));
        }
        return Some(Ok(Expr::Identifier("Null".to_string())));
      }

      return Some(Ok(Expr::FunctionCall {
        name: "Write".to_string(),
        args: args.to_vec(),
      }));
    }
    // WriteString[stream, "text1", "text2", ...] — write strings to a stream
    #[cfg(not(target_arch = "wasm32"))]
    "WriteString" if args.len() >= 2 => {
      let stream = &args[0];
      // Extract stream info
      let file_path = match stream {
        Expr::FunctionCall {
          name: stream_head,
          args: stream_args,
        } if (stream_head == "OutputStream"
          || stream_head == "InputStream")
          && stream_args.len() == 2 =>
        {
          if let Expr::Integer(id) = &stream_args[1] {
            let stream_id = *id as usize;
            crate::STREAM_REGISTRY.with(|reg| {
              let registry = reg.borrow();
              registry.get(&stream_id).and_then(|s| match &s.kind {
                crate::StreamKind::FileStream(path) => Some(path.clone()),
                _ => None,
              })
            })
          } else {
            None
          }
        }
        Expr::String(path) => Some(path.clone()),
        _ => None,
      };

      if let Some(path) = file_path {
        use std::io::Write;
        let mut file = match std::fs::OpenOptions::new()
          .create(true)
          .append(true)
          .open(&path)
          .map_err(|e| {
            InterpreterError::EvaluationError(format!(
              "WriteString: cannot open {}: {}",
              path, e
            ))
          }) {
          Ok(v) => v,
          Err(e) => return Some(Err(e)),
        };
        for arg in &args[1..] {
          let text = match arg {
            Expr::String(s) => s.clone(),
            other => crate::syntax::expr_to_string(other),
          };
          if let Err(e) = file.write_all(text.as_bytes()).map_err(|e| {
            InterpreterError::EvaluationError(format!(
              "WriteString: write error: {}",
              e
            ))
          }) {
            return Some(Err(e));
          }
        }
        return Some(Ok(Expr::Identifier("Null".to_string())));
      }

      return Some(Ok(Expr::FunctionCall {
        name: "WriteString".to_string(),
        args: args.to_vec(),
      }));
    }
    // Save["filename", symbol] or Save["filename", {sym1, sym2, ...}]
    // Saves symbol definitions (OwnValues, DownValues, Attributes, Options) to a file
    #[cfg(not(target_arch = "wasm32"))]
    "Save" if args.len() == 2 => {
      let filename = match &args[0] {
        Expr::String(s) => s.clone(),
        _ => {
          return Some(Ok(Expr::FunctionCall {
            name: "Save".to_string(),
            args: args.to_vec(),
          }));
        }
      };

      // Collect symbol names from the second argument (held)
      let symbols: Vec<String> = match &args[1] {
        Expr::Identifier(s) => vec![s.clone()],
        Expr::String(s) => vec![s.clone()],
        Expr::List(items) => items
          .iter()
          .filter_map(|item| match item {
            Expr::Identifier(s) => Some(s.clone()),
            Expr::String(s) => Some(s.clone()),
            _ => None,
          })
          .collect(),
        _ => {
          return Some(Ok(Expr::FunctionCall {
            name: "Save".to_string(),
            args: args.to_vec(),
          }));
        }
      };

      // Collect all definition lines for all symbols
      let mut all_lines: Vec<String> = Vec::new();

      for sym in &symbols {
        let mut sym_lines: Vec<String> = Vec::new();

        // 1. Attributes (user-set only)
        let user_attrs =
          crate::FUNC_ATTRS.with(|m| m.borrow().get(sym).cloned());
        if let Some(attrs) = user_attrs
          && !attrs.is_empty()
        {
          sym_lines.push(format!(
            "Attributes[{}] = {{{}}}",
            sym,
            attrs.join(", ")
          ));
        }

        // 2. DownValues (function definitions)
        let down_values = crate::FUNC_DEFS.with(|m| {
          let defs = m.borrow();
          defs.get(sym).cloned()
        });
        if let Some(overloads) = down_values {
          for (params, conditions, defaults, heads, _blank_types, body) in
            &overloads
          {
            let params_str = params
              .iter()
              .enumerate()
              .map(|(i, p)| {
                // Check if this is a literal-dispatch parameter (_dvN with SameQ condition)
                if (p.starts_with("_dv") || p.starts_with("_lp"))
                  && let Some(Some(cond)) = conditions.get(i)
                  && let Expr::Comparison {
                    operands,
                    operators,
                  } = cond
                  && operators
                    .iter()
                    .any(|op| matches!(op, crate::syntax::ComparisonOp::SameQ))
                  && operands.len() == 2
                {
                  // Literal value dispatch: use the value directly
                  return crate::syntax::expr_to_string(&operands[1]);
                }

                let head = heads.get(i).and_then(|h| h.as_ref());
                let default = defaults.get(i).and_then(|d| d.as_ref());
                let condition = conditions.get(i).and_then(|c| c.as_ref());

                let mut param_str = if let Some(h) = head {
                  format!("{}_{}", p, h)
                } else {
                  format!("{}_", p)
                };

                if let Some(def) = default {
                  param_str = format!(
                    "{}:{}",
                    param_str,
                    crate::syntax::expr_to_string(def)
                  );
                }

                if let Some(cond) = condition {
                  param_str = format!(
                    "{} /; {}",
                    param_str,
                    crate::syntax::expr_to_string(cond)
                  );
                }

                param_str
              })
              .collect::<Vec<_>>()
              .join(", ");

            let body_str = crate::syntax::expr_to_string(body);

            // Use = for literal-dispatch (all params are _dvN), := otherwise
            let is_literal_dispatch = params
              .iter()
              .all(|p| p.starts_with("_dv") || p.starts_with("_lp"));
            let assign_op = if is_literal_dispatch { "=" } else { ":=" };

            sym_lines.push(format!(
              "{}[{}] {} {}",
              sym, params_str, assign_op, body_str
            ));
          }
        }

        // 3. OwnValues (variable assignments)
        let own_value = crate::ENV.with(|e| {
          let env = e.borrow();
          env.get(sym).cloned()
        });
        if let Some(stored) = own_value {
          let val_str = match stored {
            crate::StoredValue::ExprVal(e) => crate::syntax::expr_to_string(&e),
            crate::StoredValue::Raw(val) => val,
            crate::StoredValue::Association(items) => {
              let parts: Vec<String> = items
                .iter()
                .map(|(k, v)| format!("{} -> {}", k, v))
                .collect();
              format!("<|{}|>", parts.join(", "))
            }
          };
          sym_lines.push(format!("{} = {}", sym, val_str));
        }

        // 4. Options
        let options =
          crate::FUNC_OPTIONS.with(|m| m.borrow().get(sym).cloned());
        if let Some(opts) = options
          && !opts.is_empty()
        {
          let opts_str = opts
            .iter()
            .map(crate::syntax::expr_to_string)
            .collect::<Vec<_>>()
            .join(", ");
          sym_lines.push(format!("Options[{}] = {{{}}}", sym, opts_str));
        }

        all_lines.extend(sym_lines);
      }

      // Join definitions with "\n \n" separator and add trailing newline
      let content = if all_lines.is_empty() {
        "\n".to_string()
      } else {
        format!("{}\n", all_lines.join("\n \n"))
      };

      if filename == "stdout" {
        print!("{}", content);
        crate::capture_stdout(content.trim_end());
      } else {
        match std::fs::write(&filename, &content) {
          Ok(_) => {}
          Err(_e) => {
            crate::emit_message(&format!(
              "Save::noopen: Cannot open {}.",
              filename
            ));
            return Some(Ok(Expr::Identifier("$Failed".to_string())));
          }
        }
      }

      return Some(Ok(Expr::Identifier("Null".to_string())));
    }
    // FileNames[] — list all files in current directory
    // FileNames["pattern"] — list files matching pattern
    // FileNames["pattern", "dir"] — list files in dir matching pattern
    // FileNames["pattern", "dir", Infinity] — recursive search
    #[cfg(not(target_arch = "wasm32"))]
    "FileNames" if args.len() <= 3 => {
      let pattern = if args.is_empty() {
        "*".to_string()
      } else {
        match &args[0] {
          Expr::String(s) => s.clone(),
          _ => {
            return Some(Ok(Expr::FunctionCall {
              name: "FileNames".to_string(),
              args: args.to_vec(),
            }));
          }
        }
      };

      let dir = if args.len() >= 2 {
        match &args[1] {
          Expr::String(s) => s.clone(),
          Expr::List(dirs) => {
            // FileNames["pat", {"dir1", "dir2"}] — search multiple dirs
            let mut all_files = Vec::new();
            let recursive = args.len() >= 3
              && matches!(&args[2], Expr::Identifier(s) if s == "Infinity");
            for d in dirs {
              if let Expr::String(dir_str) = d {
                let mut files =
                  collect_file_names(&pattern, dir_str, recursive);
                all_files.append(&mut files);
              }
            }
            all_files.sort();
            return Some(Ok(Expr::List(
              all_files.into_iter().map(Expr::String).collect(),
            )));
          }
          _ => ".".to_string(),
        }
      } else {
        ".".to_string()
      };

      let recursive = args.len() >= 3
        && matches!(&args[2], Expr::Identifier(s) if s == "Infinity");

      let mut files = collect_file_names(&pattern, &dir, recursive);
      files.sort();
      return Some(Ok(Expr::List(
        files.into_iter().map(Expr::String).collect(),
      )));
    }
    // SetDirectory["dir"] — push "dir" onto the virtual directory stack.
    // Does not mutate the process CWD; see the note on DIRECTORY_STACK.
    #[cfg(not(target_arch = "wasm32"))]
    "SetDirectory" if args.len() == 1 => {
      let dir = match &args[0] {
        Expr::String(s) => s.clone(),
        _ => {
          return Some(Ok(Expr::FunctionCall {
            name: "SetDirectory".to_string(),
            args: args.to_vec(),
          }));
        }
      };
      // Resolve the requested path against the current virtual directory so
      // that relative paths behave like the real Wolfram SetDirectory.
      let requested = std::path::Path::new(&dir);
      let resolved = if requested.is_absolute() {
        requested.to_path_buf()
      } else {
        std::path::PathBuf::from(virtual_current_dir()).join(requested)
      };
      // Canonicalize both to validate existence and normalize the result.
      match std::fs::canonicalize(&resolved) {
        Ok(canonical) if canonical.is_dir() => {
          let new_dir = canonical.to_string_lossy().into_owned();
          DIRECTORY_STACK.with(|s| s.borrow_mut().push(new_dir.clone()));
          return Some(Ok(Expr::String(new_dir)));
        }
        Ok(_) => {
          return Some(Err(InterpreterError::EvaluationError(format!(
            "SetDirectory: {} is not a directory.",
            dir
          ))));
        }
        Err(e) => {
          return Some(Err(InterpreterError::EvaluationError(format!(
            "SetDirectory: {}",
            e
          ))));
        }
      }
    }
    // ResetDirectory[] — pop the virtual directory stack and return the
    // restored directory (or the process CWD if the stack becomes empty).
    #[cfg(not(target_arch = "wasm32"))]
    "ResetDirectory" if args.is_empty() => {
      let popped = DIRECTORY_STACK.with(|s| s.borrow_mut().pop());
      match popped {
        Some(_) => {
          return Some(Ok(Expr::String(virtual_current_dir())));
        }
        None => {
          return Some(Err(InterpreterError::EvaluationError(
            "ResetDirectory: directory stack is empty.".into(),
          )));
        }
      }
    }
    // FileNameDrop["path", n] — drop n path components
    "FileNameDrop" if !args.is_empty() && args.len() <= 2 => {
      if let Expr::String(path) = &args[0] {
        let n = if args.len() == 2 {
          expr_to_i128(&args[1])?
        } else {
          -1 // default: drop last component
        };
        let parts: Vec<&str> = path.split('/').collect();
        let total = parts.len() as i128;
        let result = if n >= 0 {
          // Drop first n components
          let skip = (n as usize).min(parts.len());
          parts[skip..].join("/")
        } else {
          // Drop last |n| components
          let keep = (total + n).max(0) as usize;
          parts[..keep].join("/")
        };
        return Some(Ok(Expr::String(result)));
      }
    }
    _ => {}
  }
  None
}

/// Collect file names matching a glob pattern in a directory.
#[cfg(not(target_arch = "wasm32"))]
fn collect_file_names(
  pattern: &str,
  dir: &str,
  recursive: bool,
) -> Vec<String> {
  use std::path::Path;

  let dir_path = Path::new(dir);
  if !dir_path.is_dir() {
    return Vec::new();
  }

  let mut results = Vec::new();
  collect_files_recursive(dir_path, dir, pattern, recursive, &mut results);
  results
}

#[cfg(not(target_arch = "wasm32"))]
fn collect_files_recursive(
  path: &std::path::Path,
  base_dir: &str,
  pattern: &str,
  recursive: bool,
  results: &mut Vec<String>,
) {
  let entries = match std::fs::read_dir(path) {
    Ok(e) => e,
    Err(_) => return,
  };

  for entry in entries.flatten() {
    let file_name = entry.file_name().to_string_lossy().to_string();
    let file_type = entry.file_type();

    if let Ok(ft) = file_type {
      if glob_match(pattern, &file_name) {
        if base_dir == "." {
          results.push(file_name.clone());
        } else {
          let rel = entry.path();
          let rel_str = rel.to_string_lossy().to_string();
          results.push(rel_str);
        }
      }
      if ft.is_dir() && recursive {
        collect_files_recursive(
          &entry.path(),
          base_dir,
          pattern,
          true,
          results,
        );
      }
    }
  }
}

/// Simple glob pattern matching supporting * and ?
#[cfg(not(target_arch = "wasm32"))]
fn glob_match(pattern: &str, text: &str) -> bool {
  let p: Vec<char> = pattern.chars().collect();
  let t: Vec<char> = text.chars().collect();
  glob_match_impl(&p, &t)
}

#[cfg(not(target_arch = "wasm32"))]
fn glob_match_impl(pattern: &[char], text: &[char]) -> bool {
  if pattern.is_empty() {
    return text.is_empty();
  }
  if pattern[0] == '*' {
    // Try matching * with 0 or more characters
    for i in 0..=text.len() {
      if glob_match_impl(&pattern[1..], &text[i..]) {
        return true;
      }
    }
    false
  } else if text.is_empty() {
    false
  } else if pattern[0] == '?' || pattern[0] == text[0] {
    glob_match_impl(&pattern[1..], &text[1..])
  } else {
    false
  }
}

/// Render a text-mode SVG fallback for non-graphics expressions.
fn expr_text_svg(expr: &Expr) -> String {
  let boxes = super::complex_and_special::expr_to_box_form(expr);
  boxes_to_text_svg(&boxes)
}

/// Render box-form expressions to a text SVG.
fn boxes_to_text_svg(boxes: &Expr) -> String {
  let layout = crate::functions::graphics::layout_box(boxes, 14.0);
  crate::functions::graphics::layout_to_svg(&layout, "currentColor")
}

/// Convert an expression to its SVG string representation.
fn expr_to_svg(expr: &Expr) -> String {
  match expr {
    Expr::Graphics { svg: svg_data, .. } => svg_data.clone(),
    Expr::Identifier(s) if s == "-Graphics-" || s == "-Graphics3D-" => {
      crate::get_captured_graphics().unwrap_or_default()
    }
    Expr::FunctionCall {
      name: gfx_name,
      args: gfx_args,
    } if (gfx_name == "Graphics" || gfx_name == "Graphics3D")
      && !gfx_args.is_empty() =>
    {
      if let Ok(ref rendered) =
        crate::functions::graphics::graphics_ast(gfx_args)
      {
        if let Expr::Graphics { svg: svg_data, .. } = rendered {
          svg_data.clone()
        } else {
          String::new()
        }
      } else {
        String::new()
      }
    }
    Expr::FunctionCall {
      name: grid_name,
      args: grid_args,
    } if grid_name == "Grid" && !grid_args.is_empty() => {
      if crate::functions::graphics::grid_ast(grid_args).is_ok() {
        crate::get_captured_graphics().unwrap_or_default()
      } else {
        String::new()
      }
    }
    Expr::FunctionCall {
      name: style_name,
      args: style_args,
    } if style_name == "Style"
      && style_args.len() >= 2
      && matches!(
        &style_args[0],
        Expr::FunctionCall { name, args }
        if name == "Grid" && !args.is_empty()
      ) =>
    {
      if let Expr::FunctionCall {
        args: grid_args, ..
      } = &style_args[0]
      {
        let style =
          crate::functions::graphics::parse_grid_style(&style_args[1..]);
        if crate::functions::graphics::grid_ast_styled(grid_args, &style)
          .is_ok()
        {
          crate::get_captured_graphics().unwrap_or_default()
        } else {
          String::new()
        }
      } else {
        String::new()
      }
    }
    Expr::FunctionCall {
      name: ds_name,
      args: ds_args,
    } if ds_name == "Dataset" && !ds_args.is_empty() => {
      if let Some(svg) = crate::functions::graphics::dataset_to_svg(&ds_args[0])
      {
        svg
      } else {
        expr_text_svg(expr)
      }
    }
    Expr::FunctionCall {
      name: tab_name,
      args: tab_args,
    } if tab_name == "Tabular" && tab_args.len() >= 2 => {
      if let Some(svg) =
        crate::functions::graphics::tabular_to_svg(&tab_args[0], &tab_args[1])
      {
        svg
      } else {
        expr_text_svg(expr)
      }
    }
    Expr::FunctionCall {
      name: tf_name,
      args: tf_args,
    } if tf_name == "TreeForm" && !tf_args.is_empty() => {
      if let Ok(rendered) = crate::functions::tree_form::tree_form_ast(tf_args)
      {
        if let Expr::Graphics {
          svg: ref svg_data, ..
        } = rendered
        {
          svg_data.clone()
        } else {
          String::new()
        }
      } else {
        String::new()
      }
    }
    Expr::FunctionCall {
      name: tg_name,
      args: tg_args,
    } if tg_name == "TreeGraph" && !tg_args.is_empty() => {
      if let Ok(rendered) = crate::functions::tree_form::tree_graph_ast(tg_args)
      {
        if let Expr::Graphics {
          svg: ref svg_data, ..
        } = rendered
        {
          svg_data.clone()
        } else {
          String::new()
        }
      } else {
        String::new()
      }
    }
    Expr::FunctionCall {
      name: g_name,
      args: g_args,
    } if g_name == "Graph" && g_args.len() >= 2 => {
      if let Ok(rendered) = crate::functions::graph::graph_ast(g_args) {
        if let Expr::Graphics {
          svg: ref svg_data, ..
        } = rendered
        {
          svg_data.clone()
        } else {
          String::new()
        }
      } else {
        String::new()
      }
    }
    Expr::FunctionCall {
      name: mr_name,
      args: mr_args,
    } if mr_name == "MeshRegion" && mr_args.len() == 2 => {
      if let Some(svg) =
        crate::functions::voronoi::mesh_region_to_svg(&mr_args[0], &mr_args[1])
      {
        svg
      } else {
        expr_text_svg(expr)
      }
    }
    // DisplayForm[boxes] / RawBoxes[boxes] — render box expressions to SVG
    Expr::FunctionCall {
      name: box_name,
      args: box_args,
    } if (box_name == "DisplayForm" || box_name == "RawBoxes")
      && box_args.len() == 1 =>
    {
      boxes_to_text_svg(&box_args[0])
    }
    other => expr_text_svg(other),
  }
}

/// Convert an SVG string to PDF bytes using svg2pdf.
#[cfg(not(target_arch = "wasm32"))]
fn svg_to_pdf_bytes(svg_str: &str) -> Result<Vec<u8>, InterpreterError> {
  use std::sync::Arc as StdArc;

  let mut fontdb = svg2pdf::usvg::fontdb::Database::new();
  // Load system fonts first, then embedded fonts + generic-family aliases.
  // load_system_fonts() resets the generic family mappings, so our
  // set_sans_serif_family() etc. must come *after* it.
  fontdb.load_system_fonts();
  fontdb.load_font_data(
    include_bytes!(
      "../../../resources/AtkinsonHyperlegibleMono-VariableFont_wght.ttf"
    )
    .to_vec(),
  );
  fontdb.load_font_data(
    include_bytes!(
      "../../../resources/AtkinsonHyperlegibleNext-VariableFont_wght.ttf"
    )
    .to_vec(),
  );
  fontdb.set_monospace_family("Atkinson Hyperlegible Mono");
  fontdb.set_sans_serif_family("Atkinson Hyperlegible Next");
  fontdb.set_serif_family("Atkinson Hyperlegible Next");
  fontdb.set_cursive_family("Atkinson Hyperlegible Next");
  fontdb.set_fantasy_family("Atkinson Hyperlegible Next");

  let mut opt = svg2pdf::usvg::Options::default();
  opt.fontdb = StdArc::new(fontdb);

  let tree = svg2pdf::usvg::Tree::from_str(svg_str, &opt).map_err(|e| {
    InterpreterError::EvaluationError(format!(
      "Export PDF: SVG parse error: {e}"
    ))
  })?;

  let pdf_bytes = svg2pdf::to_pdf(
    &tree,
    svg2pdf::ConversionOptions::default(),
    svg2pdf::PageOptions::default(),
  )
  .map_err(|e| {
    InterpreterError::EvaluationError(format!(
      "Export PDF: conversion error: {e}"
    ))
  })?;

  Ok(pdf_bytes)
}
