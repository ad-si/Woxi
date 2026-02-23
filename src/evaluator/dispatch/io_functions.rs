#[allow(unused_imports)]
use super::*;

pub fn dispatch_io_functions(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    // ReadList[source] or ReadList[source, type] or ReadList[source, type, n]
    "ReadList" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::string_ast::read_list_ast(args));
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
          eprintln!("Get::noopen: Cannot open {}.", filename);
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
          eprintln!("Put::noopen: Cannot open {}.", filename);
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
              eprintln!("PutAppend::noopen: Cannot open {}.", filename);
              return Some(Ok(Expr::Identifier("$Failed".to_string())));
            }
          }
          Err(_) => {
            eprintln!("PutAppend::noopen: Cannot open {}.", filename);
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
      // ExportString[expr, "SVG"] - return SVG string representation
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
      if format_str != "SVG" {
        // Only SVG supported for now; return unevaluated for other formats
        return Some(Ok(Expr::FunctionCall {
          name: "ExportString".to_string(),
          args: args.to_vec(),
        }));
      }
      let svg = match &args[0] {
        Expr::Graphics { svg: svg_data, .. } => svg_data.clone(),
        Expr::Identifier(s) if s == "-Graphics-" || s == "-Graphics3D-" => {
          crate::get_captured_graphics().unwrap_or_default()
        }
        Expr::FunctionCall {
          name: grid_name,
          args: grid_args,
        } if grid_name == "Grid" && !grid_args.is_empty() => {
          // Grid[...] → render as SVG table
          if crate::functions::graphics::grid_ast(grid_args).is_ok() {
            crate::get_captured_graphics().unwrap_or_default()
          } else {
            String::new()
          }
        }
        Expr::FunctionCall {
          name: ds_name,
          args: ds_args,
        } if ds_name == "Dataset" && !ds_args.is_empty() => {
          // Dataset[data, ...] → render as SVG table
          if let Some(svg) =
            crate::functions::graphics::dataset_to_svg(&ds_args[0])
          {
            svg
          } else {
            // Fallback: render as text SVG
            let markup =
              crate::functions::graphics::expr_to_svg_markup(&args[0]);
            let char_width = 8.4_f64;
            let font_size = 14_usize;
            let display_width =
              crate::functions::graphics::estimate_display_width(&args[0]);
            let width = (display_width * char_width).ceil() as usize;
            let (height, text_y) =
              if crate::functions::graphics::has_fraction(&args[0]) {
                (32_usize, 18_usize)
              } else {
                (font_size + 4, font_size)
              };
            format!(
              "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\">\
               <text x=\"0\" y=\"{text_y}\" font-family=\"monospace\" font-size=\"{font_size}\">{markup}</text>\
               </svg>",
              width, height
            )
          }
        }
        other => {
          // Non-graphics: render expression as SVG text with superscripts
          let markup = crate::functions::graphics::expr_to_svg_markup(other);
          let char_width = 8.4_f64;
          let font_size = 14_usize;
          let display_width =
            crate::functions::graphics::estimate_display_width(other);
          let width = (display_width * char_width).ceil() as usize;
          let (height, text_y) =
            if crate::functions::graphics::has_fraction(other) {
              (32_usize, 18_usize)
            } else {
              (font_size + 4, font_size)
            };
          format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\">\
             <text x=\"0\" y=\"{text_y}\" font-family=\"monospace\" font-size=\"{font_size}\">{markup}</text>\
             </svg>",
            width, height
          )
        }
      };
      return Some(Ok(Expr::String(svg)));
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
      return Some(match std::env::current_dir() {
        Ok(path) => Ok(Expr::String(path.to_string_lossy().into_owned())),
        Err(err) => Err(InterpreterError::EvaluationError(err.to_string())),
      });
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
        eprintln!("OpenRead::noopen: Cannot open {}.", filename);
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
            Some(name) => return Some(Ok(Expr::String(name))),
            None => {
              let stream_str = crate::syntax::expr_to_string(&args[0]);
              eprintln!("{} is not open.", stream_str);
              return Some(Ok(Expr::FunctionCall {
                name: "Close".to_string(),
                args: args.to_vec(),
              }));
            }
          }
        }
        Expr::String(s) => {
          eprintln!("{} is not open.", s);
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
    _ => {}
  }
  None
}
