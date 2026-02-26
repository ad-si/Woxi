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
      if format_str == "PDF" {
        // Generate minimal PDF with expression text
        let text = crate::syntax::expr_to_output(&args[0]);
        return Some(Ok(Expr::String(generate_minimal_pdf(&text))));
      }
      if format_str != "SVG" {
        // Return unevaluated for unsupported formats
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
        Expr::FunctionCall {
          name: tab_name,
          args: tab_args,
        } if tab_name == "Tabular" && tab_args.len() >= 2 => {
          // Tabular[data, schema] → render as SVG table
          if let Some(svg) =
            crate::functions::graphics::tabular_to_svg(
              &tab_args[0],
              &tab_args[1],
            )
          {
            svg
          } else {
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
            Some(name) => return Some(Ok(Expr::Identifier(name))),
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
          for (params, conditions, defaults, heads, body) in &overloads {
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
            eprintln!("Save::noopen: Cannot open {}.", filename);
            return Some(Ok(Expr::Identifier("$Failed".to_string())));
          }
        }
      }

      return Some(Ok(Expr::Identifier("Null".to_string())));
    }
    _ => {}
  }
  None
}

/// Generate a minimal valid PDF containing the given text.
fn generate_minimal_pdf(text: &str) -> String {
  // Escape special PDF characters in text
  let escaped: String = text
    .chars()
    .map(|c| match c {
      '(' => "\\(".to_string(),
      ')' => "\\)".to_string(),
      '\\' => "\\\\".to_string(),
      _ => c.to_string(),
    })
    .collect();

  let content_stream = format!("BT /F1 12 Tf 72 720 Td ({}) Tj ET", escaped);
  let content_len = content_stream.len();

  let mut pdf = String::new();
  let mut offsets: Vec<usize> = Vec::new();

  pdf.push_str("%PDF-1.4\n");

  // Object 1: Catalog
  offsets.push(pdf.len());
  pdf.push_str("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

  // Object 2: Pages
  offsets.push(pdf.len());
  pdf.push_str("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

  // Object 3: Page
  offsets.push(pdf.len());
  pdf.push_str(
    "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n   \
     /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n",
  );

  // Object 4: Content stream
  offsets.push(pdf.len());
  pdf.push_str(&format!(
    "4 0 obj\n<< /Length {} >>\nstream\n{}\nendstream\nendobj\n",
    content_len, content_stream
  ));

  // Object 5: Font
  offsets.push(pdf.len());
  pdf.push_str(
    "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
  );

  // Cross-reference table
  let xref_offset = pdf.len();
  let num_objects = offsets.len() + 1; // +1 for the free object 0
  pdf.push_str("xref\n");
  pdf.push_str(&format!("0 {}\n", num_objects));
  pdf.push_str("0000000000 65535 f \n");
  for offset in &offsets {
    pdf.push_str(&format!("{:010} 00000 n \n", offset));
  }

  // Trailer
  pdf.push_str(&format!(
    "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
    num_objects, xref_offset
  ));

  pdf
}
