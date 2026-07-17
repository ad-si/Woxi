#[allow(unused_imports)]
use super::*;
use crate::syntax::unevaluated;

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
    // Message[sym::tag, args...] — emit a message and return Null. Only matches
    // when the first argument is a MessageName; other shapes fall through to
    // stay unevaluated.
    "Message" if !args.is_empty() => {
      if let Expr::FunctionCall {
        name: mn_name,
        args: mn_args,
      } = &args[0]
        && mn_name == "MessageName"
        && mn_args.len() == 2
      {
        let sym_name = match &mn_args[0] {
          Expr::Identifier(s) => s.clone(),
          other => crate::syntax::expr_to_string(other),
        };
        let tag = match &mn_args[1] {
          Expr::String(s) => s.clone(),
          Expr::Identifier(s) => s.clone(),
          other => crate::syntax::expr_to_string(other),
        };
        // Evaluate MessageName[sym, tag]. If it resolves to a String, use it
        // as the text; otherwise treat the text as unset.
        let resolved = crate::evaluator::evaluate_expr_to_expr(&args[0]);
        let text = match &resolved {
          Ok(Expr::String(s)) => s.clone(),
          _ => "-- Message text not found --".to_string(),
        };
        // Fill the `1`, `2`, ... template slots with the extra arguments
        // (rendered in output form, so strings appear unquoted), matching
        // wolframscript: Message[f::mymsg, 42] shows "Custom 42 here.".
        let mut filled = text;
        for (i, arg) in args[1..].iter().enumerate() {
          let placeholder = format!("`{}`", i + 1);
          if filled.contains(&placeholder) {
            let shown =
              crate::syntax::format_expr(arg, crate::syntax::ExprForm::Output);
            filled = filled.replace(&placeholder, &shown);
          }
        }
        // Route through emit_message so the message is captured (Check
        // reacts to user messages), respects Quiet/Off, participates in
        // General::stop suppression, and reaches the same stream as
        // built-in messages.
        crate::emit_message(&format!("{}::{}: {}", sym_name, tag, filled));
        return Some(Ok(Expr::Identifier("Null".to_string())));
      }
    }
    // HTTPRequest[url] / HTTPRequest[url, assoc] / HTTPRequest[assoc] —
    // symbolic HTTP request object; no network access is performed.
    // The one-argument URL form canonicalizes to HTTPRequest[url, <||>],
    // matching wolframscript; other shapes stay as given.
    "HTTPRequest" if !args.is_empty() => {
      return Some(crate::functions::http_ast::http_request_ast(args));
    }
    // URLRead[req] / URLRead[url] — send the HTTP request through curl and
    // return the HTTPResponse object (or Failure["ConnectionFailure", …]).
    #[cfg(not(target_arch = "wasm32"))]
    "URLRead" if args.len() == 1 => {
      return Some(crate::functions::http_ast::url_read_ast(&args[0]));
    }
    // URLFetch[url] / URLFetch[url, params] — minimal stub.
    // Returns $Failed for URLs that lack a host (e.g. "https://"), matching
    // wolframscript's behavior. Network fetches are out of scope for the
    // CLI/snapshot test loop, so any other URL also returns $Failed.
    "URLFetch" if args.len() == 1 || args.len() == 2 => {
      if let Expr::String(_) = &args[0] {
        return Some(Ok(Expr::Identifier("$Failed".to_string())));
      }
    }
    // Environment["name"] — return the named environment variable value
    "Environment" if args.len() == 1 => {
      if let Expr::String(var_name) = &args[0] {
        return Some(Ok(match std::env::var(var_name) {
          Ok(val) => Expr::String(val),
          Err(_) => Expr::Identifier("$Failed".to_string()),
        }));
      }
      return Some(Ok(unevaluated("Environment", args)));
    }
    // GetEnvironment[] — all environment variables as a List of rules.
    "GetEnvironment" if args.is_empty() => {
      let rules: Vec<Expr> = std::env::vars()
        .map(|(k, v)| Expr::Rule {
          pattern: Box::new(Expr::String(k)),
          replacement: Box::new(Expr::String(v)),
        })
        .collect();
      return Some(Ok(Expr::List(rules.into())));
    }
    // GetEnvironment["name"] — return "name" -> "value" rule
    // GetEnvironment[{"n1","n2"}] — list of rules
    "GetEnvironment" if args.len() == 1 => {
      let make_rule = |var: &str| -> Expr {
        Expr::Rule {
          pattern: Box::new(Expr::String(var.to_string())),
          replacement: Box::new(match std::env::var(var) {
            Ok(val) => Expr::String(val),
            Err(_) => Expr::Identifier("None".to_string()),
          }),
        }
      };
      match &args[0] {
        Expr::String(var) => return Some(Ok(make_rule(var))),
        Expr::List(items) => {
          let rules: Vec<Expr> = items
            .iter()
            .map(|item| match item {
              Expr::String(v) => make_rule(v),
              _ => item.clone(),
            })
            .collect();
          return Some(Ok(Expr::List(rules.into())));
        }
        _ => {}
      }
      return Some(Ok(unevaluated("GetEnvironment", args)));
    }
    // SetEnvironment["name" -> "value"]   — set an environment variable
    // SetEnvironment["name" -> None]      — unset an environment variable
    // SetEnvironment[{rule1, rule2, ...}] — apply multiple rules
    // Returns Null on success, $Failed if any value is not a string or None.
    "SetEnvironment" if args.len() == 1 => {
      fn apply_rule(rule: &Expr) -> Option<bool> {
        let (pat, val) = match rule {
          Expr::Rule {
            pattern,
            replacement,
          }
          | Expr::RuleDelayed {
            pattern,
            replacement,
          } => (pattern.as_ref(), replacement.as_ref()),
          _ => return None,
        };
        let var = match pat {
          Expr::String(s) => s.clone(),
          _ => return Some(false),
        };
        match val {
          Expr::String(v) => {
            // SAFETY: Woxi is single-threaded in the REPL / CLI path.
            unsafe { std::env::set_var(&var, v) };
            Some(true)
          }
          Expr::Identifier(name) if name == "None" => {
            unsafe { std::env::remove_var(&var) };
            Some(true)
          }
          _ => {
            eprintln!(
              "SetEnvironment::setraw: {} must be a string or None.",
              crate::syntax::expr_to_string(val)
            );
            Some(false)
          }
        }
      }
      let ok = match &args[0] {
        Expr::List(rules) => {
          let mut all_ok = true;
          for r in rules {
            match apply_rule(r) {
              Some(true) => {}
              _ => all_ok = false,
            }
          }
          all_ok
        }
        other => matches!(apply_rule(other), Some(true)),
      };
      return Some(Ok(if ok {
        Expr::Identifier("Null".to_string())
      } else {
        Expr::Identifier("$Failed".to_string())
      }));
    }
    // Streams[] — return list of open streams (stdout and stderr)
    "Streams" if args.is_empty() => {
      return Some(Ok(Expr::List(
        vec![
          Expr::FunctionCall {
            name: "OutputStream".to_string(),
            args: vec![Expr::String("stdout".to_string()), Expr::Integer(1)]
              .into(),
          },
          Expr::FunctionCall {
            name: "OutputStream".to_string(),
            args: vec![Expr::String("stderr".to_string()), Expr::Integer(2)]
              .into(),
          },
        ]
        .into(),
      )));
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
            args: vec![Expr::String(n.to_string()), Expr::Integer(*id)].into(),
          })
          .collect();
        return Some(Ok(Expr::List(matching.into())));
      }
      return Some(Ok(Expr::List(vec![].into())));
    }
    // ReadList[source] or ReadList[source, type] or ReadList[source, type, n]
    "ReadList" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::string_ast::read_list_ast(args));
    }
    // ReadString["file"] — read a host-registered virtual file (WASM).
    // The browser has no local filesystem, so the virtual store registered
    // via `set_virtual_file` is the only file source.
    #[cfg(target_arch = "wasm32")]
    "ReadString" if args.len() == 1 => {
      let filename = match &args[0] {
        Expr::String(s) => s.clone(),
        _ => {
          return Some(Ok(unevaluated("ReadString", args)));
        }
      };
      let Some(bytes) = crate::wasm::virtual_file(&filename) else {
        crate::emit_message(&format!(
          "ReadString::noopen: Cannot open {}.",
          filename
        ));
        return Some(Ok(Expr::Identifier("$Failed".to_string())));
      };
      return Some(match String::from_utf8(bytes) {
        Ok(content) => Ok(Expr::String(content)),
        Err(_) => Err(InterpreterError::EvaluationError(format!(
          "ReadString: \"{}\" is not valid UTF-8 text",
          filename
        ))),
      });
    }
    // ReadString["file"] — read file contents as a string
    #[cfg(not(target_arch = "wasm32"))]
    "ReadString" if args.len() == 1 => {
      let filename = match &args[0] {
        Expr::String(s) => s.clone(),
        _ => {
          return Some(Ok(unevaluated("ReadString", args)));
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
    // FileTemplate[src] / FileTemplate[src, args] — read a template file from
    // disk and produce a TemplateObject (the same object StringTemplate would
    // build from the file's contents). `src` may be a path string or a
    // File["path"] wrapper.
    #[cfg(not(target_arch = "wasm32"))]
    "FileTemplate" if args.len() == 1 || args.len() == 2 => {
      let filename = match &args[0] {
        Expr::String(s) => s.clone(),
        Expr::FunctionCall { name, args: inner }
          if name == "File"
            && inner.len() == 1
            && matches!(&inner[0], Expr::String(_)) =>
        {
          match &inner[0] {
            Expr::String(s) => s.clone(),
            _ => unreachable!(),
          }
        }
        // URL[…] / CloudObject[…] and other specifications are left
        // unevaluated (network access is out of scope).
        _ => {
          return Some(Ok(unevaluated("FileTemplate", args)));
        }
      };
      // Resolve relative paths against the virtual working directory.
      let requested = std::path::Path::new(&filename);
      let resolved = if requested.is_absolute() {
        requested.to_path_buf()
      } else {
        std::path::PathBuf::from(virtual_current_dir()).join(requested)
      };
      let content = match std::fs::read_to_string(&resolved) {
        Ok(c) => c,
        Err(_) => {
          crate::emit_message_to_stdout(&format!(
            "StringTemplate::fnfnd: File \"{}\" not found.",
            filename
          ));
          return Some(Ok(Expr::Identifier("$Failed".to_string())));
        }
      };
      let bound_args = if args.len() == 2 {
        Some(args[1].clone())
      } else {
        None
      };
      return Some(Ok(crate::functions::string_ast::build_template_object(
        &content,
        bound_args,
        "TextString",
      )));
    }
    // XMLTemplate[src] / XMLTemplate[src, args] — like StringTemplate but with
    // InsertionFunction -> HTMLFragment. `src` may be a literal template string
    // or a File["path"] wrapper that is read from disk. The template string may
    // embed `<* expr *>` sections in addition to `` `slot` `` markers.
    "XMLTemplate" if args.len() == 1 || args.len() == 2 => {
      let content = match &args[0] {
        Expr::String(s) => s.clone(),
        #[cfg(not(target_arch = "wasm32"))]
        Expr::FunctionCall { name, args: inner }
          if name == "File"
            && inner.len() == 1
            && matches!(&inner[0], Expr::String(_)) =>
        {
          let filename = match &inner[0] {
            Expr::String(s) => s.clone(),
            _ => unreachable!(),
          };
          let requested = std::path::Path::new(&filename);
          let resolved = if requested.is_absolute() {
            requested.to_path_buf()
          } else {
            std::path::PathBuf::from(virtual_current_dir()).join(requested)
          };
          match std::fs::read_to_string(&resolved) {
            Ok(c) => c,
            Err(_) => {
              crate::emit_message_to_stdout(&format!(
                "XMLTemplate::fnfnd: File \"{}\" not found.",
                filename
              ));
              return Some(Ok(Expr::Identifier("$Failed".to_string())));
            }
          }
        }
        // URL[…] / CloudObject[…] and other specifications are left
        // unevaluated (network access is out of scope).
        _ => {
          return Some(Ok(unevaluated("XMLTemplate", args)));
        }
      };
      let bound_args = if args.len() == 2 {
        Some(args[1].clone())
      } else {
        None
      };
      return Some(Ok(crate::functions::string_ast::build_template_object(
        &content,
        bound_args,
        "HTMLFragment",
      )));
    }
    // Get[file] — read and evaluate a file, returning the last result
    #[cfg(not(target_arch = "wasm32"))]
    "Get" if args.len() == 1 => {
      let filename = match &args[0] {
        Expr::String(s) => s.clone(),
        Expr::Identifier(s) => s.clone(),
        _ => {
          return Some(Ok(unevaluated("Get", args)));
        }
      };
      let content = match std::fs::read_to_string(&filename) {
        Ok(c) => c,
        Err(_) => {
          // wolframscript prints this message to stdout (verified with
          // `wolframscript -file`), so mirror it into the captured buffer to
          // keep snapshot/playground/Jupyter output byte-for-byte consistent.
          crate::emit_message_to_stdout(&format!(
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
          return Some(Ok(unevaluated("Put", args)));
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
          return Some(Ok(unevaluated("PutAppend", args)));
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

      // Handle Image export.  Vector formats (SVG) wrap the raster in a
      // base64-encoded PNG <image> element so the file is a valid SVG;
      // every other format is written as a raster file by the image crate.
      if let Expr::Image {
        width,
        height,
        channels,
        data,
        ..
      } = &args[1]
      {
        if fmt == "SVG" {
          let svg = crate::functions::image_ast::image_to_svg_document(
            *width, *height, *channels, data,
          );
          if let Err(e) = std::fs::write(&filename, &svg).map_err(|e| {
            InterpreterError::EvaluationError(format!("Export: {e}"))
          }) {
            return Some(Err(e));
          }
          return Some(Ok(Expr::String(filename)));
        }
        if let Err(e) = crate::functions::image_ast::export_image(
          &filename, *width, *height, *channels, data,
        ) {
          return Some(Err(e));
        }
        return Some(Ok(Expr::String(filename)));
      }

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

      if fmt == "SVG" {
        let svg = expr_to_svg(&args[1]);
        // expr_to_svg returns an empty string when a graphics head fails to
        // render; fall back to the text rendering so the file stays valid SVG.
        let svg = if svg.is_empty() {
          expr_text_svg(&args[1])
        } else {
          svg
        };
        if let Err(e) = std::fs::write(&filename, &svg).map_err(|e| {
          InterpreterError::EvaluationError(format!("Export: {e}"))
        }) {
          return Some(Err(e));
        }
        return Some(Ok(Expr::String(filename)));
      }

      // Raster image formats: rasterize the SVG and write via the image crate.
      if matches!(
        fmt.as_str(),
        "PNG" | "JPG" | "JPEG" | "GIF" | "BMP" | "TIF" | "TIFF"
      ) {
        // Parse ImageResolution option (default 96 DPI to match
        // usvg's default output resolution).
        let mut dpi: f64 = 96.0;
        // Frame delay in hundredths of a second (GIF's native unit).
        // Default 1/8 s = 12 (matches Mathematica's 8 fps default for
        // animated GIF export).
        let mut frame_delay_hundredths: u16 = 12;
        for opt in &args[2..] {
          if let Expr::Rule {
            pattern,
            replacement,
          } = opt
            && let Expr::Identifier(k) = pattern.as_ref()
          {
            match k.as_str() {
              "ImageResolution" => match replacement.as_ref() {
                Expr::Integer(n) => dpi = *n as f64,
                Expr::Real(f) => dpi = *f,
                _ => {
                  return Some(Err(InterpreterError::EvaluationError(
                    "Export: ImageResolution must be a numeric value".into(),
                  )));
                }
              },
              "AnimationRate" | "FrameRate" => {
                // Frames per second → hundredths-of-a-second per frame.
                let fps = match replacement.as_ref() {
                  Expr::Integer(n) => *n as f64,
                  Expr::Real(f) => *f,
                  _ => 30.0,
                };
                if fps > 0.0 {
                  frame_delay_hundredths =
                    (100.0 / fps).round().clamp(1.0, 65535.0) as u16;
                }
              }
              _ => {}
            }
          }
        }

        // Animated GIF path: when exporting a list of graphics to GIF,
        // rasterize each element as a frame.
        if fmt == "GIF"
          && let Expr::List(items) = &args[1]
          && items.len() >= 2
          && items.iter().all(is_rasterizable_frame)
        {
          let mut frames =
            Vec::<crate::functions::image_ast::GifFrame>::with_capacity(
              items.len(),
            );
          for item in items {
            let svg = expr_to_svg(item);
            match crate::functions::image_ast::rasterize_svg(&svg, dpi) {
              Ok(Expr::Image {
                width,
                height,
                channels,
                ref data,
                ..
              }) => {
                let dyn_img =
                  crate::functions::image_ast::expr_to_dynamic_image(
                    width, height, channels, data,
                  );
                frames.push(crate::functions::image_ast::GifFrame {
                  image: dyn_img.to_rgba8(),
                  delay_hundredths: frame_delay_hundredths,
                });
              }
              Ok(_) => unreachable!("rasterize_svg returns Expr::Image"),
              Err(e) => return Some(Err(e)),
            }
          }
          if let Err(e) =
            crate::functions::image_ast::export_animated_gif(&filename, frames)
          {
            return Some(Err(e));
          }
          return Some(Ok(Expr::String(filename)));
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
      // WAV export of a playable sound (Play[…] / Sound[…] / Audio[…]).
      if matches!(fmt.as_str(), "WAV" | "WAVE")
        && let Some(bytes) =
          crate::functions::sound::expr_to_wav_bytes(&args[1])
      {
        if let Err(e) = std::fs::write(&filename, &bytes).map_err(|e| {
          InterpreterError::EvaluationError(format!("Export: {e}"))
        }) {
          return Some(Err(e));
        }
        return Some(Ok(Expr::String(filename)));
      }

      // MIDI export of a computational-music object (MusicScore / MusicVoice / …).
      if (fmt == "MID" || fmt == "MIDI")
        && let Some(bytes) =
          crate::functions::music_midi::music_to_midi(&args[1])
      {
        if let Err(e) = std::fs::write(&filename, &bytes).map_err(|e| {
          InterpreterError::EvaluationError(format!("Export: {e}"))
        }) {
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
    // Browser (WASM) `Export`: there is no filesystem, so instead of writing to
    // disk we serialize the value and hand the bytes to the host via
    // `record_exported_file`, which surfaces them as downloads. Only the
    // formats whose encoders compile to `wasm32` are supported; native-only
    // formats (raster images, PDF, XLSX) return a clear error.
    #[cfg(target_arch = "wasm32")]
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
      // Format from an explicit third-argument string, else the file extension.
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
      let data = &args[1];

      let bytes: Vec<u8> = match fmt.as_str() {
        "CSV" => export_string_csv(data, ',', true, true).into_bytes(),
        "TSV" => export_string_csv(data, '\t', true, true).into_bytes(),
        "JSON" | "RAWJSON" => match export_string_json(data, 0, false) {
          Some(mut json) => {
            json.push('\n');
            json.into_bytes()
          }
          None => {
            return Some(Err(InterpreterError::EvaluationError(
              "Export: value cannot be serialized to JSON".into(),
            )));
          }
        },
        "SVG" => {
          let svg = expr_to_svg(data);
          // expr_to_svg is empty when a graphics head fails to render; fall
          // back to the text rendering so the file stays valid SVG.
          let svg = if svg.is_empty() {
            expr_text_svg(data)
          } else {
            svg
          };
          svg.into_bytes()
        }
        "WAV" | "WAVE" => {
          match crate::functions::sound::expr_to_wav_bytes(data) {
            Some(b) => b,
            None => {
              return Some(Err(InterpreterError::EvaluationError(
                "Export: value is not a playable sound".into(),
              )));
            }
          }
        }
        "MID" | "MIDI" => {
          match crate::functions::music_midi::music_to_midi(data) {
            Some(b) => b,
            None => {
              return Some(Err(InterpreterError::EvaluationError(
                "Export: value is not a music object".into(),
              )));
            }
          }
        }
        // Raster image formats. Encoding an existing Image works in the
        // browser (the `image` crate compiles to wasm); rasterizing a plot or
        // other graphics does not, because the SVG rasterizer (resvg) is
        // native-only. So only Image values are accepted here.
        "PNG" | "JPG" | "JPEG" | "GIF" | "BMP" | "TIF" | "TIFF" => match data {
          Expr::Image {
            width,
            height,
            channels,
            data: pixels,
            ..
          } => match crate::functions::image_ast::export_image_bytes(
            &fmt, *width, *height, *channels, pixels,
          ) {
            Ok(b) => b,
            Err(e) => return Some(Err(e)),
          },
          _ => {
            return Some(Err(InterpreterError::EvaluationError(format!(
              "Export: {} export of a non-image expression (e.g. a plot) \
                 is not supported in the browser; export as SVG instead",
              fmt
            ))));
          }
        },
        // Formats whose encoders are native-only in the WASM build.
        "PDF" | "XLSX" => {
          return Some(Err(InterpreterError::EvaluationError(format!(
            "Export: \"{}\" export is not supported in the browser",
            fmt
          ))));
        }
        // Text and unrecognized formats: strings verbatim, a list one element
        // per line, graphics as their SVG, other expressions rendered directly.
        _ => {
          let elem = |e: &Expr| match e {
            Expr::String(s) => s.clone(),
            _ => crate::syntax::format_expr(e, crate::syntax::ExprForm::Output),
          };
          let content = match data {
            Expr::Graphics { svg, .. } => svg.clone(),
            Expr::Identifier(s) if s == "-Graphics-" || s == "-Graphics3D-" => {
              crate::get_captured_graphics().unwrap_or_default()
            }
            Expr::String(s) => s.clone(),
            Expr::List(items) => {
              items.iter().map(elem).collect::<Vec<_>>().join("\n")
            }
            other => elem(other),
          };
          content.into_bytes()
        }
      };

      crate::wasm::record_exported_file(&filename, &bytes);
      return Some(Ok(Expr::String(filename)));
    }
    "ExportString" if args.len() == 2 || args.len() == 3 => {
      // ExportString[expr, "format"] - return string representation.
      // An optional third argument carries format options; for JSON the
      // "Compact" -> True option emits the value with no extra whitespace.
      let compact = matches!(args.get(2), Some(Expr::Rule { pattern, replacement })
        if matches!(pattern.as_ref(), Expr::String(s) if s == "Compact")
          && matches!(replacement.as_ref(), Expr::Identifier(v) if v == "True"));
      let format_str = match &args[1] {
        Expr::String(s) => s.clone(),
        _ => {
          // Return unevaluated for non-string format
          return Some(Ok(unevaluated("ExportString", args)));
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
            return Some(Ok(unevaluated("ExportString", args)));
          }
        }
        return Some(Ok(Expr::String(svg)));
      }
      if format_str == "CSV" || format_str == "TSV" {
        let sep = if format_str == "CSV" { ',' } else { '\t' };
        return Some(Ok(Expr::String(export_string_csv(
          &args[0], sep, true, true,
        ))));
      }
      // "Table" is tab-separated like TSV but leaves strings unquoted and
      // emits no trailing newline.
      if format_str == "Table" {
        return Some(Ok(Expr::String(export_string_csv(
          &args[0], '\t', false, false,
        ))));
      }
      if (format_str == "JSON" || format_str == "RawJSON")
        && let Some(json) = export_string_json(&args[0], 0, compact)
      {
        return Some(Ok(Expr::String(json)));
      }
      // "Text"/"Lines"/"List": a string is emitted verbatim; a list has each
      // element rendered (OutputForm, strings unquoted) on its own line; an
      // atom is rendered directly.
      if format_str == "Text" || format_str == "Lines" || format_str == "List" {
        let elem = |e: &Expr| match e {
          Expr::String(s) => s.clone(),
          _ => crate::syntax::format_expr(e, crate::syntax::ExprForm::Output),
        };
        let s = match &args[0] {
          Expr::String(s) => s.clone(),
          Expr::List(items) => {
            items.iter().map(elem).collect::<Vec<_>>().join("\n")
          }
          other => elem(other),
        };
        return Some(Ok(Expr::String(s)));
      }
      // Return unevaluated for unsupported formats
      return Some(Ok(unevaluated("ExportString", args)));
    }
    #[cfg(not(target_arch = "wasm32"))]
    "Find" if args.len() == 2 => {
      // Find[stream_or_file, "text" | {"a", "b", …}] - find first line
      // that contains any of the search strings. Accepts file paths,
      // InputStream[…] / OutputStream[…] backed by either a file or a
      // string buffer. Advances the stream's position past the matched
      // line so consecutive Find calls walk forward.
      let search_terms: Vec<String> = match &args[1] {
        Expr::String(s) => vec![s.clone()],
        Expr::List(items) => {
          let mut terms = Vec::with_capacity(items.len());
          for item in items {
            match item {
              Expr::String(s) => terms.push(s.clone()),
              _ => {
                return Some(Err(InterpreterError::EvaluationError(
                  "Find: second argument must be a string or a list of strings"
                    .into(),
                )));
              }
            }
          }
          terms
        }
        _ => {
          return Some(Err(InterpreterError::EvaluationError(
            "Find: second argument must be a string or a list of strings"
              .into(),
          )));
        }
      };

      // (content, start_pos, optional stream id for position advance)
      let (content, start_pos, stream_id) = match &args[0] {
        Expr::String(path) => {
          let body = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
              return Some(Err(InterpreterError::EvaluationError(format!(
                "Find: {e}"
              ))));
            }
          };
          (body, 0usize, None)
        }
        Expr::FunctionCall {
          name: stream_head,
          args: stream_args,
        } if (stream_head == "InputStream"
          || stream_head == "OutputStream")
          && stream_args.len() == 2 =>
        {
          if let Expr::Integer(id) = &stream_args[1] {
            let id_usize = *id as usize;
            match crate::get_stream_content(id_usize) {
              Some((c, p)) => (c, p, Some(id_usize)),
              None => return Some(Ok(Expr::Identifier("$Failed".to_string()))),
            }
          } else {
            return Some(Ok(Expr::Identifier("$Failed".to_string())));
          }
        }
        _ => {
          let arg_str = crate::syntax::expr_to_string(&args[0]);
          crate::emit_message(&format!(
            "Find::stream: {} is not a string, SocketObject, InputStream[ ] or OutputStream[ ].",
            arg_str
          ));
          return Some(Ok(Expr::Identifier("$Failed".to_string())));
        }
      };

      let remaining = &content[start_pos.min(content.len())..];
      let mut consumed = 0usize;
      for line in remaining.split_inclusive('\n') {
        let stripped = line
          .strip_suffix('\n')
          .unwrap_or(line)
          .trim_end_matches('\r');
        consumed += line.len();
        if search_terms.iter().any(|t| stripped.contains(t)) {
          if let Some(id) = stream_id {
            crate::set_stream_position(id, start_pos + consumed);
          }
          return Some(Ok(Expr::String(stripped.to_string())));
        }
      }
      if let Some(id) = stream_id {
        crate::set_stream_position(id, content.len());
      }
      return Some(Ok(Expr::Identifier("EndOfFile".to_string())));
    }
    #[cfg(not(target_arch = "wasm32"))]
    "FindList" => {
      // FindList[file(s), text(s)[, n]] — all lines containing any of the
      // search strings (literal, case-sensitive substrings). Errors return
      // $Failed with the matching wolframscript message; a failed file in
      // a file LIST contributes a $Failed element instead.
      let failed = || Some(Ok(Expr::Identifier("$Failed".to_string())));
      let call_display =
        || crate::syntax::expr_to_output(&unevaluated("FindList", args));
      if args.len() < 2 {
        let (tag, noun) = if args.len() == 1 {
          ("argtu", "1 argument")
        } else {
          ("argt", "0 arguments")
        };
        crate::emit_message(&format!(
          "FindList::{}: FindList called with {}; 2 or 3 arguments are expected.",
          tag, noun
        ));
        return failed();
      }
      if args.len() > 3 {
        for extra in &args[3..] {
          let is_opt = matches!(
            extra,
            Expr::Rule { .. } | Expr::RuleDelayed { .. } | Expr::List(_)
          );
          if !is_opt {
            crate::emit_message(&format!(
              "FindList::nonopt: Options expected (instead of {}) beyond position 3 in {}. An option must be a rule or a list of rules.",
              crate::syntax::expr_to_string(extra),
              call_display()
            ));
            return failed();
          }
        }
      }
      let terms: Vec<String> = match &args[1] {
        Expr::String(s) => vec![s.clone()],
        Expr::List(items)
          if !items.is_empty()
            && items.iter().all(|it| matches!(it, Expr::String(_))) =>
        {
          items
            .iter()
            .map(|it| {
              let Expr::String(s) = it else { unreachable!() };
              s.clone()
            })
            .collect()
        }
        _ => {
          crate::emit_message(&format!(
            "FindList::strs: A string or nonempty list of strings is expected at position 2 in {}.",
            call_display()
          ));
          return failed();
        }
      };
      let limit: usize = match args.get(2) {
        None => usize::MAX,
        Some(Expr::Integer(n)) if *n >= 0 => *n as usize,
        Some(_) => {
          crate::emit_message(&format!(
            "FindList::intnm: Non-negative machine-sized integer expected at position 3 in {}.",
            call_display()
          ));
          return failed();
        }
      };
      let (files, is_list): (Vec<Expr>, bool) = match &args[0] {
        Expr::String(_) => (vec![args[0].clone()], false),
        Expr::List(items) => (items.iter().cloned().collect(), true),
        other => {
          crate::emit_message(&format!(
            "FindList::stream: {} is not a string, SocketObject, InputStream[ ] or OutputStream[ ].",
            crate::syntax::expr_to_string(other)
          ));
          return failed();
        }
      };
      let mut out: Vec<Expr> = Vec::new();
      let mut found = 0usize;
      for f in &files {
        if found >= limit {
          break;
        }
        let Expr::String(path) = f else {
          crate::emit_message(&format!(
            "FindList::stream: {} is not a string, SocketObject, InputStream[ ] or OutputStream[ ].",
            crate::syntax::expr_to_string(f)
          ));
          if !is_list {
            return failed();
          }
          out.push(Expr::Identifier("$Failed".to_string()));
          continue;
        };
        match std::fs::read_to_string(path) {
          Err(_) => {
            crate::emit_message(&format!(
              "FindList::noopen: Cannot open {}.",
              path
            ));
            if !is_list {
              return failed();
            }
            out.push(Expr::Identifier("$Failed".to_string()));
          }
          Ok(content) => {
            for line in content.lines() {
              if found >= limit {
                break;
              }
              let stripped = line.trim_end_matches('\r');
              if terms.iter().any(|t| stripped.contains(t.as_str())) {
                out.push(Expr::String(stripped.to_string()));
                found += 1;
              }
            }
          }
        }
      }
      return Some(Ok(Expr::List(out.into())));
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
    "NotebookDirectory" if args.is_empty() => {
      return Some(match crate::get_notebook_directory() {
        Some(dir) => Ok(Expr::String(dir)),
        None => {
          crate::emit_message(
            "NotebookDirectory::nosv: The notebook directory is not available outside a notebook front-end.",
          );
          Ok(unevaluated("NotebookDirectory", args))
        }
      });
    }
    #[cfg(not(target_arch = "wasm32"))]
    "ParentDirectory" if args.is_empty() || args.len() == 1 => {
      let base = if args.is_empty() {
        virtual_current_dir()
      } else if let Expr::String(s) = &args[0] {
        s.clone()
      } else {
        return Some(Ok(unevaluated("ParentDirectory", args)));
      };
      let parent = std::path::Path::new(&base)
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| base.clone());
      return Some(Ok(Expr::String(parent)));
    }
    // DirectoryName["path"] or DirectoryName["path", n]
    "DirectoryName" if args.len() == 1 || args.len() == 2 => {
      let path_str = match &args[0] {
        Expr::String(s) => s.clone(),
        _ => {
          return Some(Ok(unevaluated("DirectoryName", args)));
        }
      };
      let n = if args.len() == 2 {
        match &args[1] {
          Expr::Integer(i) if *i >= 1 => *i as usize,
          Expr::Integer(_) => {
            crate::emit_message(
              "DirectoryName::intpm: Positive machine-sized integer expected at position 2 in DirectoryName.",
            );
            return Some(Ok(unevaluated("DirectoryName", args)));
          }
          _ => {
            return Some(Ok(unevaluated("DirectoryName", args)));
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
    "ToFileName" if args.len() == 1 || args.len() == 2 => {
      let sep = std::path::MAIN_SEPARATOR.to_string();
      let collect_dirs = |expr: &Expr| -> Option<Vec<String>> {
        match expr {
          Expr::String(s) => Some(vec![s.clone()]),
          Expr::List(parts) => {
            let mut segments = Vec::with_capacity(parts.len());
            for p in parts {
              if let Expr::String(s) = p {
                segments.push(s.clone());
              } else {
                return None;
              }
            }
            Some(segments)
          }
          _ => None,
        }
      };
      if args.len() == 1 {
        if let Some(dirs) = collect_dirs(&args[0]) {
          let joined = dirs.join(&sep);
          return Some(Ok(Expr::String(format!("{}{}", joined, sep))));
        }
      } else if let (Some(dirs), Expr::String(file)) =
        (collect_dirs(&args[0]), &args[1])
      {
        let mut all = dirs;
        all.push(file.clone());
        return Some(Ok(Expr::String(all.join(&sep))));
      }
      return Some(Ok(unevaluated("ToFileName", args)));
    }
    "FileNameJoin" if args.len() == 1 || args.len() == 2 => {
      // Detect OperatingSystem option from second argument (a Rule).
      let sep: char = if args.len() == 2 {
        let mut s = std::path::MAIN_SEPARATOR;
        if let Expr::Rule {
          pattern,
          replacement,
        } = &args[1]
          && matches!(pattern.as_ref(),
            Expr::Identifier(n) if n == "OperatingSystem")
          && let Expr::String(os) = replacement.as_ref()
        {
          s = if os == "Windows" { '\\' } else { '/' };
        }
        s
      } else {
        std::path::MAIN_SEPARATOR
      };
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
          let joined = segments.join(&sep.to_string());
          return Some(Ok(Expr::String(joined)));
        }
      }
      return Some(Ok(unevaluated("FileNameJoin", args)));
    }
    "FileNameSplit" if args.len() == 1 => {
      if let Expr::String(s) = &args[0] {
        if s.is_empty() {
          return Some(Ok(Expr::List(vec![].into())));
        }
        let parts: Vec<Expr> = s
          .split('/')
          .collect::<Vec<&str>>()
          .into_iter()
          .enumerate()
          .filter(|(i, part)| !(*i > 0 && part.is_empty()))
          .map(|(_, part)| Expr::String(part.to_string()))
          .collect();
        return Some(Ok(Expr::List(parts.into())));
      }
      return Some(Ok(unevaluated("FileNameSplit", args)));
    }
    "FileNameDepth" if args.len() == 1 => {
      if let Expr::String(s) = &args[0] {
        if s.is_empty() {
          return Some(Ok(Expr::Integer(0)));
        }
        let count = s
          .split('/')
          .enumerate()
          .filter(|(i, part)| !(*i > 0 && part.is_empty()))
          .count() as i128;
        return Some(Ok(Expr::Integer(count)));
      }
      return Some(Ok(unevaluated("FileNameDepth", args)));
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
      return Some(Ok(unevaluated("ExpandFileName", args)));
    }
    "URLParse" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::http_ast::url_parse_ast(args));
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
          return Some(Ok(unevaluated("URLBuild", args)));
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
    // OpenRead[file, BinaryFormat -> True] — same; binary mode is handled
    // by BinaryRead at read time, so the option is accepted as a pass-through.
    #[cfg(not(target_arch = "wasm32"))]
    "OpenRead" if (1..=2).contains(&args.len()) => {
      let (filename_arg, _opts) = io_split_filename_and_options(args);
      let filename = match filename_arg {
        Some(Expr::String(s)) => s.clone(),
        Some(other) => {
          return Some(Ok(Expr::FunctionCall {
            name: "OpenRead".to_string(),
            args: vec![other.clone()].into(),
          }));
        }
        None => {
          return Some(Ok(unevaluated("OpenRead", args)));
        }
      };
      if !std::path::Path::new(&filename).exists() {
        crate::emit_message_to_stdout(&format!(
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
        args: vec![Expr::String(filename), Expr::Integer(id as i128)].into(),
      }));
    }
    // OpenWrite[file] — open a file for writing, return OutputStream[name, id]
    // OpenWrite[BinaryFormat -> True] — same, options pass-through.
    #[cfg(not(target_arch = "wasm32"))]
    "OpenWrite" if args.len() <= 2 => {
      let (filename_arg, _opts) = io_split_filename_and_options(args);
      let filename = match filename_arg {
        Some(Expr::String(s)) => s.clone(),
        Some(other) => {
          return Some(Ok(Expr::FunctionCall {
            name: "OpenWrite".to_string(),
            args: vec![other.clone()].into(),
          }));
        }
        None => {
          let path = match crate::utils::create_file(None)
            .map_err(|e| InterpreterError::EvaluationError(e.to_string()))
          {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
          };
          path.to_string_lossy().into_owned()
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
        args: vec![Expr::String(filename), Expr::Integer(id as i128)].into(),
      }));
    }
    // BinaryWrite[stream, bytes]              — write bytes (Integers in 0..255)
    // BinaryWrite[stream, bytes, type]        — write with explicit type spec
    // BinaryWrite[stream, bytes, {types…}]    — per-element types
    //
    // Returns the same `stream`. Supported types: "Byte" (Integer → 1 byte),
    // "Character8" (Integer or 1-char String → 1 byte). The 2-arg form
    // infers the type from the value (Integer → Byte, String → Character8).
    #[cfg(not(target_arch = "wasm32"))]
    "BinaryWrite" if (2..=3).contains(&args.len()) => {
      let path = match io_stream_path(&args[0]) {
        Some(p) => p,
        None => {
          return Some(Ok(unevaluated("BinaryWrite", args)));
        }
      };
      // Render a single value at the given type into the byte buffer.
      // Returns false on an unsupported pairing so the caller can fall
      // back to the unevaluated form.
      fn write_value(out: &mut Vec<u8>, value: &Expr, ty: &str) -> bool {
        match (value, ty) {
          (Expr::Integer(n), "Byte" | "Character8") => {
            out.push((*n & 0xff) as u8);
            true
          }
          (Expr::String(s), "Byte" | "Character8") => {
            out.extend_from_slice(s.as_bytes());
            true
          }
          _ => false,
        }
      }
      let unevaluated = || unevaluated("BinaryWrite", args);
      let bytes: Vec<u8> = if args.len() == 3 {
        // Explicit type spec — either a single type string applied to
        // every value or a list of per-element types.
        let mut out: Vec<u8> = Vec::new();
        let values: Vec<&Expr> = match &args[1] {
          Expr::List(items) => items.iter().collect(),
          v => vec![v],
        };
        match &args[2] {
          Expr::String(ty) => {
            for v in &values {
              if !write_value(&mut out, v, ty) {
                return Some(Ok(unevaluated()));
              }
            }
          }
          Expr::List(types) => {
            if types.len() != values.len() {
              return Some(Ok(unevaluated()));
            }
            for (v, t) in values.iter().zip(types.iter()) {
              let Expr::String(ty) = t else {
                return Some(Ok(unevaluated()));
              };
              if !write_value(&mut out, v, ty) {
                return Some(Ok(unevaluated()));
              }
            }
          }
          _ => return Some(Ok(unevaluated())),
        }
        out
      } else {
        // 2-arg form: infer type from the value(s).
        match &args[1] {
          Expr::Integer(n) => vec![(*n & 0xff) as u8],
          Expr::String(s) => s.as_bytes().to_vec(),
          Expr::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
              match it {
                Expr::Integer(n) => out.push((*n & 0xff) as u8),
                Expr::String(s) => out.extend_from_slice(s.as_bytes()),
                _ => return Some(Ok(unevaluated())),
              }
            }
            out
          }
          _ => return Some(Ok(unevaluated())),
        }
      };
      use std::io::Write;
      let mut f = match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
      {
        Ok(f) => f,
        Err(e) => {
          return Some(Err(InterpreterError::EvaluationError(format!(
            "BinaryWrite: cannot open {}: {}",
            path, e
          ))));
        }
      };
      if let Err(e) = f.write_all(&bytes) {
        return Some(Err(InterpreterError::EvaluationError(format!(
          "BinaryWrite: write failed on {}: {}",
          path, e
        ))));
      }
      return Some(Ok(args[0].clone()));
    }
    // BinaryRead[stream]                — read 1 Byte (Integer 0..255)
    // BinaryRead[stream, "Byte"]        — read 1 Byte
    // BinaryRead[stream, "Character8"]  — read 1 Char (String of length 1)
    // BinaryRead[stream, {forms…}]      — read N items, returning a List
    //
    // Sequential calls advance the stream's read position so a chain like
    //   BinaryRead[s]; BinaryRead[s, {"Character8","Character8"}]
    // returns the next bytes after the first call rather than starting
    // from offset 0 every time.
    #[cfg(not(target_arch = "wasm32"))]
    "BinaryRead" if (1..=2).contains(&args.len()) => {
      let path = match io_stream_path(&args[0]) {
        Some(p) => p,
        None => {
          return Some(Ok(unevaluated("BinaryRead", args)));
        }
      };
      // Extract stream id (second arg of InputStream[path, id]) so we can
      // track and advance the read position. Falls back to id-less reads
      // (always from offset 0) when the structure doesn't match.
      let stream_id = if let Expr::FunctionCall {
        name: sname,
        args: sargs,
      } = &args[0]
        && (sname == "InputStream" || sname == "OutputStream")
        && sargs.len() == 2
        && let Expr::Integer(id) = &sargs[1]
      {
        Some(*id as usize)
      } else {
        None
      };
      let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
          return Some(Err(InterpreterError::EvaluationError(format!(
            "BinaryRead: cannot read {}: {}",
            path, e
          ))));
        }
      };
      let start_pos =
        stream_id.and_then(crate::get_stream_position).unwrap_or(0);
      let form = if args.len() == 2 {
        args[1].clone()
      } else {
        Expr::String("Byte".to_string())
      };
      // Render a single-byte form (Byte/Character8) at the given offset,
      // returning EndOfFile when out of range.
      let read_one = |form: &Expr, offset: usize| -> Option<Expr> {
        match form {
          Expr::String(s) if s == "Byte" => {
            if offset < bytes.len() {
              Some(Expr::Integer(bytes[offset] as i128))
            } else {
              Some(Expr::Identifier("EndOfFile".to_string()))
            }
          }
          Expr::String(s) if s == "Character8" => {
            if offset < bytes.len() {
              // Character8 is a raw byte rendered as a 1-char string;
              // values >127 use the Latin-1 mapping.
              let c = bytes[offset] as char;
              Some(Expr::String(c.to_string()))
            } else {
              Some(Expr::Identifier("EndOfFile".to_string()))
            }
          }
          _ => None,
        }
      };
      match &form {
        Expr::String(_) => {
          let Some(result) = read_one(&form, start_pos) else {
            return Some(Ok(unevaluated("BinaryRead", args)));
          };
          if let Some(id) = stream_id {
            let advance = if matches!(&result, Expr::Identifier(s) if s == "EndOfFile")
            {
              0
            } else {
              1
            };
            crate::set_stream_position(id, start_pos + advance);
          }
          return Some(Ok(result));
        }
        Expr::List(items) => {
          let mut out = Vec::with_capacity(items.len());
          let mut offset = start_pos;
          for it in items.iter() {
            let Some(value) = read_one(it, offset) else {
              return Some(Ok(unevaluated("BinaryRead", args)));
            };
            if !matches!(&value, Expr::Identifier(s) if s == "EndOfFile") {
              offset += 1;
            }
            out.push(value);
          }
          if let Some(id) = stream_id {
            crate::set_stream_position(id, offset);
          }
          return Some(Ok(Expr::List(out.into())));
        }
        _ => {
          return Some(Ok(unevaluated("BinaryRead", args)));
        }
      }
    }
    // BinaryReadList[file]            — read all bytes from `file`
    // BinaryReadList[file, "Byte"]    — same
    // BinaryReadList[stream]          — read remaining bytes from stream
    // BinaryReadList[stream, "Byte"]  — same
    //
    // Returns a List of Integers in 0..255. Returns {} on EOF.
    #[cfg(not(target_arch = "wasm32"))]
    "BinaryReadList" if (1..=2).contains(&args.len()) => {
      let path = match &args[0] {
        Expr::String(s) => s.clone(),
        _ => match io_stream_path(&args[0]) {
          Some(p) => p,
          None => {
            return Some(Ok(unevaluated("BinaryReadList", args)));
          }
        },
      };
      let form = if args.len() == 2 {
        args[1].clone()
      } else {
        Expr::String("Byte".to_string())
      };
      // Only "Byte" is supported; other forms fall through unevaluated so
      // callers see the same behaviour as for BinaryRead.
      match &form {
        Expr::String(s) if s == "Byte" => {}
        _ => {
          return Some(Ok(unevaluated("BinaryReadList", args)));
        }
      }
      let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
          return Some(Err(InterpreterError::EvaluationError(format!(
            "BinaryReadList: cannot read {}: {}",
            path, e
          ))));
        }
      };
      let out: Vec<Expr> = bytes
        .into_iter()
        .map(|b| Expr::Integer(b as i128))
        .collect();
      return Some(Ok(Expr::List(out.into())));
    }
    // OpenAppend[file] — open a file for appending, return OutputStream[name, id]
    #[cfg(not(target_arch = "wasm32"))]
    "OpenAppend" if args.len() <= 1 => {
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
              name: "OpenAppend".to_string(),
              args: vec![other.clone()].into(),
            }));
          }
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
        args: vec![Expr::String(filename), Expr::Integer(id as i128)].into(),
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
      // Use Symbol `String` (not the string literal "String") so the
      // formatted form matches wolframscript: `InputStream[String, id]`.
      return Some(Ok(Expr::FunctionCall {
        name: "InputStream".to_string(),
        args: vec![
          Expr::Identifier("String".to_string()),
          Expr::Integer(id as i128),
        ]
        .into(),
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
              return Some(Ok(unevaluated("Close", args)));
            }
          };
          match crate::close_stream(id) {
            // Close[FileStream] returns the file path as a String;
            // Close[StringToStream[…]] returns the symbol `String`.
            Some((name, crate::StreamKind::StringStream(_))) => {
              let _ = name;
              return Some(Ok(Expr::Identifier("String".to_string())));
            }
            Some((name, _)) => return Some(Ok(Expr::String(name))),
            None => {
              let stream_str = crate::syntax::expr_to_string(&args[0]);
              crate::emit_message(&format!("{} is not open.", stream_str));
              return Some(Ok(unevaluated("Close", args)));
            }
          }
        }
        Expr::String(s) => {
          crate::emit_message(&format!("{} is not open.", s));
          return Some(Ok(unevaluated("Close", args)));
        }
        _ => {
          // Anything else is a type error — match wolframscript's message.
          let arg_str = crate::syntax::expr_to_string(&args[0]);
          crate::emit_message_to_stdout(&format!(
            "Close::stream: {} is not a string, SocketObject, InputStream[ ] or OutputStream[ ].",
            arg_str
          ));
          return Some(Ok(unevaluated("Close", args)));
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
                return Some(Ok(unevaluated("StreamPosition", args)));
              }
            }
          } else {
            return Some(Ok(unevaluated("StreamPosition", args)));
          }
        }
        Expr::String(s) => {
          crate::emit_message(&format!(
            "StreamPosition::openx: {} is not open.",
            s
          ));
          return Some(Ok(unevaluated("StreamPosition", args)));
        }
        _ => {
          return Some(Ok(unevaluated("StreamPosition", args)));
        }
      }
    }
    // SetStreamPosition[stream, pos] — set the current position of a stream.
    // `pos` is either a non-negative integer (absolute byte offset) or
    // `Infinity` (seek to end of stream). Returns the new position.
    "SetStreamPosition" if args.len() == 2 => {
      let stream = &args[0];
      let is_infinity =
        matches!(&args[1], Expr::Identifier(s) if s == "Infinity");
      let pos_explicit = match &args[1] {
        Expr::Integer(n) => Some(*n as usize),
        _ if is_infinity => None,
        _ => {
          return Some(Ok(unevaluated("SetStreamPosition", args)));
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
            let id_usize = *id as usize;
            if crate::is_stream_open(id_usize) {
              let stream_len = crate::get_stream_content(id_usize)
                .map(|(content, _)| content.len())
                .unwrap_or(0);
              let pos = match pos_explicit {
                Some(p) => {
                  if p > stream_len {
                    // wolframscript emits SetStreamPosition::stmrng
                    // and clamps the position to the end of stream.
                    let stream_str = crate::syntax::expr_to_string(stream);
                    crate::emit_message(&format!(
                      "SetStreamPosition::stmrng: Cannot set the current point in {} to position {}; the requested position exceeds the length of the stream.",
                      stream_str, p
                    ));
                    stream_len
                  } else {
                    p
                  }
                }
                None => stream_len, // Infinity → end of stream
              };
              crate::set_stream_position(id_usize, pos);
              return Some(Ok(Expr::Integer(pos as i128)));
            } else {
              let stream_str = crate::syntax::expr_to_string(stream);
              crate::emit_message(&format!(
                "SetStreamPosition::openx: {} is not open.",
                stream_str
              ));
              return Some(Ok(unevaluated("SetStreamPosition", args)));
            }
          } else {
            return Some(Ok(unevaluated("SetStreamPosition", args)));
          }
        }
        _ => {
          return Some(Ok(unevaluated("SetStreamPosition", args)));
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
            return Some(Ok(unevaluated("ReadLine", args)));
          }
        }
        _ => {
          return Some(Ok(unevaluated("ReadLine", args)));
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
    // Skip[stream, type] / Skip[stream, type, n] — read and discard `n`
    // (default 1) values of the given type, advancing the stream position.
    "Skip" if args.len() == 2 || args.len() == 3 => {
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

      let count = if args.len() == 3 {
        match &args[2] {
          Expr::Integer(n) if *n >= 0 => *n as usize,
          _ => {
            return Some(Ok(unevaluated("Skip", args)));
          }
        }
      } else {
        1
      };

      if let Some(id) = stream_id
        && let Some((content, mut position)) = crate::get_stream_content(id)
      {
        let mut hit_eof = false;
        for _ in 0..count {
          let remaining = &content[position.min(content.len())..];
          let (val, advance) = read_single_type(remaining, &args[1]);
          if matches!(&val, Expr::Identifier(s) if s == "EndOfFile") {
            hit_eof = true;
            position = content.len();
            break;
          }
          if advance == 0 {
            hit_eof = true;
            break;
          }
          position += advance;
        }
        crate::advance_stream_position(id, position);
        return Some(Ok(Expr::Identifier(
          if hit_eof { "EndOfFile" } else { "Null" }.to_string(),
        )));
      }

      return Some(Ok(unevaluated("Skip", args)));
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
          return Some(Ok(Expr::List(results.into())));
        }

        let (result, advance) = read_single_type(remaining, read_type);
        crate::advance_stream_position(id, position + advance);
        return Some(Ok(result));
      }

      return Some(Ok(unevaluated("Read", args)));
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

      return Some(Ok(unevaluated("Write", args)));
    }
    // WriteString[stream, "text1", "text2", ...] — write strings to a stream
    #[cfg(not(target_arch = "wasm32"))]
    "WriteString" if args.len() >= 2 => {
      let stream = &args[0];
      // Special-case the standard streams so `WriteString["stdout", …]` and
      // `WriteString[$Output, …]` write to the process's stdout, matching
      // wolframscript. `$Output`/`"stdout"` map to stdout, `$Messages`/
      // `"stderr"` to stderr. Stdout writes also go through the captured
      // buffer (like Print) so they appear in `interpret_with_stdout`.
      let std_target = match stream {
        Expr::String(name) if name == "stdout" => Some(true),
        Expr::String(name) if name == "stderr" => Some(false),
        Expr::Identifier(name) if name == "$Output" => Some(true),
        Expr::Identifier(name) if name == "$Messages" => Some(false),
        _ => None,
      };
      if let Some(is_stdout) = std_target {
        use std::io::Write;
        for arg in &args[1..] {
          let text = match arg {
            Expr::String(s) => s.clone(),
            other => crate::syntax::expr_to_string(other),
          };
          if is_stdout {
            if !crate::is_quiet_print() {
              print!("{}", text);
              let _ = std::io::stdout().flush();
            }
            crate::capture_stdout_raw(&text);
          } else {
            eprint!("{}", text);
            let _ = std::io::stderr().flush();
          }
        }
        return Some(Ok(Expr::Identifier("Null".to_string())));
      }
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

      return Some(Ok(unevaluated("WriteString", args)));
    }
    // Save["filename", symbol] or Save["filename", {sym1, sym2, ...}]
    // Saves symbol definitions (OwnValues, DownValues, Attributes, Options) to a file
    #[cfg(not(target_arch = "wasm32"))]
    "Save" if args.len() == 2 => {
      let filename = match &args[0] {
        Expr::String(s) => s.clone(),
        _ => {
          return Some(Ok(unevaluated("Save", args)));
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
          return Some(Ok(unevaluated("Save", args)));
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

        // 2. DownValues (function definitions). Includes literal-argument
        // memoizations (e.g. `f[1] = 42`), which live in MEMO_VALUES.
        let down_values = crate::down_values_with_memo(sym);
        if let Some(overloads) = down_values {
          for (params, conditions, defaults, heads, blank_types, body) in
            &overloads
          {
            // List-pattern params (`_lp{i}`) reconstruct to a surface `{…}`
            // pattern with the original element names, body, and `/;` guard.
            if let Some((pattern_args, display_body)) =
              crate::evaluator::assignment::reconstruct_list_downvalue(
                params,
                conditions,
                heads,
                blank_types,
                body,
              )
            {
              let params_str = pattern_args
                .iter()
                .map(crate::syntax::expr_to_string)
                .collect::<Vec<_>>()
                .join(", ");
              sym_lines.push(format!(
                "{}[{}] := {}",
                sym,
                params_str,
                crate::syntax::expr_to_string(&display_body)
              ));
              continue;
            }
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
                    .any(|op| matches!(op, ComparisonOp::SameQ))
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
                .map(|(k, v)| {
                  format!("{} -> {}", k, crate::syntax::expr_to_string(v))
                })
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
            return Some(Ok(unevaluated("FileNames", args)));
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
    // SetDirectory[] — with no arguments, set to $HomeDirectory.
    #[cfg(not(target_arch = "wasm32"))]
    "SetDirectory" if args.is_empty() => {
      let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_default();
      if home.is_empty() {
        return Some(Err(InterpreterError::EvaluationError(
          "SetDirectory: cannot determine home directory.".into(),
        )));
      }
      match std::fs::canonicalize(&home) {
        Ok(canonical) if canonical.is_dir() => {
          let new_dir = canonical.to_string_lossy().into_owned();
          DIRECTORY_STACK.with(|s| s.borrow_mut().push(new_dir.clone()));
          return Some(Ok(Expr::String(new_dir)));
        }
        _ => {
          return Some(Err(InterpreterError::EvaluationError(
            "SetDirectory: home directory does not exist.".into(),
          )));
        }
      }
    }
    // SetDirectory["dir"] — push "dir" onto the virtual directory stack.
    // Does not mutate the process CWD; see the note on DIRECTORY_STACK.
    #[cfg(not(target_arch = "wasm32"))]
    "SetDirectory" if args.len() == 1 => {
      let dir = match &args[0] {
        Expr::String(s) => s.clone(),
        _ => {
          return Some(Ok(unevaluated("SetDirectory", args)));
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
    // DirectoryStack[] — return the directory stack maintained by
    // SetDirectory/ResetDirectory. Fresh sessions report `{}`.
    #[cfg(not(target_arch = "wasm32"))]
    "DirectoryStack" if args.is_empty() => {
      let stack = DIRECTORY_STACK
        .with(|s| s.borrow().iter().cloned().collect::<Vec<_>>());
      return Some(Ok(Expr::List(
        stack.into_iter().map(Expr::String).collect(),
      )));
    }
    // FileFormat["name"] — return the format string for a file, or
    // emit `FileFormat::nffil` and `$Failed` when missing. Actual
    // format detection isn't implemented yet.
    #[cfg(not(target_arch = "wasm32"))]
    "FileFormat" if args.len() == 1 => {
      let Expr::String(name) = &args[0] else {
        return Some(Ok(unevaluated("FileFormat", args)));
      };
      if !std::path::Path::new(name).exists() {
        crate::emit_message(&format!(
          "FileFormat::nffil: File not found during FileFormat[{}].",
          name
        ));
        return Some(Ok(Expr::Identifier("$Failed".to_string())));
      }
      return Some(Ok(unevaluated("FileFormat", args)));
    }
    // FileDate["name"] / FileDate["name", "type"] — file timestamps.
    // Woxi doesn't implement the date lookup yet; for missing files it
    // still reproduces wolframscript's error path (`fdnfnd` message
    // and an unevaluated FileDate[…] result).
    #[cfg(not(target_arch = "wasm32"))]
    "FileDate" if args.len() == 1 || args.len() == 2 => {
      let Expr::String(name) = &args[0] else {
        return Some(Ok(unevaluated("FileDate", args)));
      };
      if !std::path::Path::new(name).exists() {
        crate::emit_message(&format!(
          "FileDate::fdnfnd: Directory or file \"{}\" not found.",
          name
        ));
      }
      return Some(Ok(unevaluated("FileDate", args)));
    }
    // FileHash["name"] / FileHash["name", "Algorithm"] — return the
    // hash as an Integer. Missing files emit `FileHash::noopen` and
    // return `$Failed`, matching wolframscript.
    #[cfg(not(target_arch = "wasm32"))]
    "FileHash" if args.len() == 1 || args.len() == 2 => {
      let Expr::String(name) = &args[0] else {
        return Some(Ok(unevaluated("FileHash", args)));
      };
      // Only emit the matching error message — actual hashing isn't
      // supported yet. wolframscript reports the absolute path, so
      // resolve relative paths against the current working directory.
      let path = std::path::Path::new(name);
      if !path.exists() {
        let abs = if path.is_absolute() {
          name.clone()
        } else {
          std::env::current_dir()
            .map(|cwd| cwd.join(path).to_string_lossy().into_owned())
            .unwrap_or_else(|_| name.clone())
        };
        crate::emit_message(&format!("FileHash::noopen: Cannot open {}.", abs));
        return Some(Ok(Expr::Identifier("$Failed".to_string())));
      }
      return Some(Ok(unevaluated("FileHash", args)));
    }
    // FileSize["name"] — the size as Quantity[bytes, "Bytes"] with a Real
    // magnitude. Unlike FileByteCount, errors echo the call unevaluated:
    // ::fdnfnd for a missing path, ::fdir for a directory, ::badfile for a
    // non-string argument. A File["…"] wrapper is accepted.
    #[cfg(not(target_arch = "wasm32"))]
    "FileSize" if args.len() == 1 => {
      let unevaluated = || Some(Ok(unevaluated("FileSize", args)));
      let name = match &args[0] {
        Expr::String(s) => s.clone(),
        Expr::FunctionCall { name, args: fargs }
          if name == "File"
            && fargs.len() == 1
            && matches!(&fargs[0], Expr::String(_)) =>
        {
          match &fargs[0] {
            Expr::String(s) => s.clone(),
            _ => unreachable!(),
          }
        }
        other => {
          crate::emit_message(&format!(
            "FileSize::badfile: The specified argument, {}, should be a valid string or File object.",
            crate::syntax::expr_to_output(other)
          ));
          return unevaluated();
        }
      };
      match std::fs::metadata(&name) {
        Ok(meta) if meta.is_file() => {
          return Some(Ok(Expr::FunctionCall {
            name: "Quantity".to_string(),
            args: vec![
              Expr::Real(meta.len() as f64),
              Expr::String("Bytes".to_string()),
            ]
            .into(),
          }));
        }
        Ok(meta) if meta.is_dir() => {
          crate::emit_message(&format!(
            "FileSize::fdir: The specified path {} refers to a directory; a file path was expected.",
            name
          ));
          return unevaluated();
        }
        _ => {
          crate::emit_message(&format!(
            "FileSize::fdnfnd: Directory or file \"{}\" not found.",
            name
          ));
          return unevaluated();
        }
      }
    }
    // FileByteCount["name"] — size in bytes, or emit `fdnfnd` and
    // return `$Failed` when the file is missing.
    #[cfg(not(target_arch = "wasm32"))]
    "FileByteCount" if args.len() == 1 => {
      let Expr::String(name) = &args[0] else {
        return Some(Ok(unevaluated("FileByteCount", args)));
      };
      match std::fs::metadata(name) {
        Ok(meta) if meta.is_file() => {
          return Some(Ok(Expr::Integer(meta.len() as i128)));
        }
        _ => {
          crate::emit_message(&format!(
            "FileByteCount::fdnfnd: Directory or file \"{}\" not found.",
            name
          ));
          return Some(Ok(Expr::Identifier("$Failed".to_string())));
        }
      }
    }
    // AbsoluteFileName["name"] — return the absolute path if the file
    // exists, otherwise emit `AbsoluteFileName::fdnfnd` and return
    // `$Failed` (matching wolframscript).
    #[cfg(not(target_arch = "wasm32"))]
    "AbsoluteFileName" if args.len() == 1 => {
      let Expr::String(name) = &args[0] else {
        return Some(Ok(unevaluated("AbsoluteFileName", args)));
      };
      match std::fs::canonicalize(name) {
        Ok(p) => {
          return Some(Ok(Expr::String(p.to_string_lossy().into_owned())));
        }
        Err(_) => {
          crate::emit_message(&format!(
            "AbsoluteFileName::fdnfnd: Directory or file \"{}\" not found.",
            name
          ));
          return Some(Ok(Expr::Identifier("$Failed".to_string())));
        }
      }
    }
    // FindFile["name"] — return the absolute path if the file exists,
    // else `$Failed`. Context strings like "VectorAnalysis`" are also
    // accepted but always fail (no package loader).
    #[cfg(not(target_arch = "wasm32"))]
    "FindFile" if args.len() == 1 => {
      let Expr::String(name) = &args[0] else {
        return Some(Ok(unevaluated("FindFile", args)));
      };
      // Context names end with a backtick and can't be resolved to a file
      // on disk. Match wolframscript's `$Failed` return.
      if name.contains('`') {
        return Some(Ok(Expr::Identifier("$Failed".to_string())));
      }
      return Some(Ok(match std::fs::canonicalize(name) {
        Ok(p) => Expr::String(p.to_string_lossy().into_owned()),
        Err(_) => Expr::Identifier("$Failed".to_string()),
      }));
    }
    // FileNameDrop["path", n] — drop n path components
    "FileNameDrop" if !args.is_empty() && args.len() <= 2 => {
      if let Expr::String(path) = &args[0] {
        let n = if args.len() == 2 {
          expr_to_i128(&args[1])?
        } else {
          -1 // default: drop last component
        };
        let sep = std::path::MAIN_SEPARATOR_STR;
        let parts: Vec<&str> = path.split(sep).collect();
        let total = parts.len() as i128;
        let result = if n >= 0 {
          // Drop first n components
          let skip = (n as usize).min(parts.len());
          parts[skip..].join(sep)
        } else {
          // Drop last |n| components
          let keep = (total + n).max(0) as usize;
          parts[..keep].join(sep)
        };
        return Some(Ok(Expr::String(result)));
      }
    }
    "FileNameTake" if !args.is_empty() && args.len() <= 2 => {
      if let Expr::String(path) = &args[0] {
        // Path components, matching FileNameSplit: split on '/', dropping
        // empty segments except a leading one (the absolute-root marker).
        let components: Vec<String> = path
          .split('/')
          .enumerate()
          .filter(|(i, part)| !(*i > 0 && part.is_empty()))
          .map(|(_, part)| part.to_string())
          .collect();
        let total = components.len() as i128;
        // Root-aware join: a slice consisting only of the leading "" marker
        // (or otherwise joining to nothing) is the absolute root "/".
        let join = |parts: &[String]| -> String {
          let joined = parts.join("/");
          if joined.is_empty() && !parts.is_empty() {
            "/".to_string()
          } else {
            joined
          }
        };
        // Resolve the take specification into a 0-indexed `[start, end)` range.
        let slice: Option<(usize, usize)> = match args.get(1) {
          // Default: just the last component.
          None => {
            if total == 0 {
              Some((0, 0))
            } else {
              Some(((total - 1) as usize, total as usize))
            }
          }
          Some(Expr::Integer(n)) => {
            if *n >= 0 {
              Some((0, (*n).clamp(0, total) as usize))
            } else {
              Some(((total + *n).max(0) as usize, total as usize))
            }
          }
          Some(Expr::List(range)) if range.len() == 2 => {
            if let (Expr::Integer(m), Expr::Integer(nn)) =
              (&range[0], &range[1])
            {
              let resolve =
                |idx: i128| if idx < 0 { total + idx } else { idx - 1 };
              let s = resolve(*m);
              let e = resolve(*nn);
              if s < 0 || e >= total || s > e {
                None
              } else {
                Some((s as usize, (e + 1) as usize))
              }
            } else {
              None
            }
          }
          _ => None,
        };
        if let Some((s, e)) = slice
          && s <= e
          && e <= components.len()
        {
          return Some(Ok(Expr::String(join(&components[s..e]))));
        }
      }
      return Some(Ok(unevaluated("FileNameTake", args)));
    }
    // Input[] / Input[prompt] / InputString[] / InputString[prompt] —
    // wolframscript in script mode prints the prompt to stdout (no trailing
    // newline) and returns `EndOfFile` since interactive stdin isn't
    // available. Match that so non-interactive scripts behave identically.
    "Input" | "InputString" if args.len() <= 1 => {
      if let Some(arg) = args.first() {
        let prompt = match arg {
          Expr::String(p) => p.clone(),
          _ => crate::syntax::expr_to_string(arg),
        };
        if !crate::is_quiet_print() {
          use std::io::Write as _;
          print!("{prompt}");
          let _ = std::io::stdout().flush();
        }
        crate::capture_stdout_raw(&prompt);
      }
      return Some(Ok(Expr::Identifier("EndOfFile".to_string())));
    }
    _ => {}
  }
  None
}

/// Split an `OpenWrite[…]` / `OpenRead[…]` arg list into the optional
/// filename argument and the remaining option-Rule arguments. Used so the
/// `BinaryFormat -> True` option can be passed through alongside (or
/// instead of) a filename.
#[cfg(not(target_arch = "wasm32"))]
fn io_split_filename_and_options(args: &[Expr]) -> (Option<&Expr>, Vec<&Expr>) {
  let mut filename = None;
  let mut opts = Vec::new();
  for a in args {
    if matches!(a, Expr::Rule { .. } | Expr::RuleDelayed { .. }) {
      opts.push(a);
    } else if filename.is_none() {
      filename = Some(a);
    } else {
      opts.push(a);
    }
  }
  (filename, opts)
}

/// Extract the file path backing an `InputStream[name, id]` /
/// `OutputStream[name, id]` expression. Used by `BinaryWrite` /
/// `BinaryRead` to find the underlying file.
#[cfg(not(target_arch = "wasm32"))]
fn io_stream_path(expr: &Expr) -> Option<String> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "InputStream" && name != "OutputStream" {
    return None;
  }
  if args.is_empty() {
    return None;
  }
  match &args[0] {
    Expr::String(s) => Some(s.clone()),
    _ => None,
  }
}

/// Render a single delimited-table cell. Numeric/symbolic atoms are emitted
/// bare. When `quote_strings` is set (CSV/TSV) strings are wrapped in `"…"`
/// with embedded `"` doubled, matching wolframscript; the `"Table"` format
/// passes `false`, emitting strings verbatim.
fn csv_cell(expr: &Expr, quote_strings: bool) -> String {
  match expr {
    Expr::String(s) if quote_strings => {
      format!("\"{}\"", s.replace('"', "\"\""))
    }
    Expr::String(s) => s.clone(),
    _ => crate::syntax::expr_to_string(expr),
  }
}

/// Serialize an expression to Wolfram's pretty-printed JSON (tab-indented,
/// `"key":value` with no space after the colon, `true`/`false`/`null`, empty
/// containers inline as `[]` / `{}`). `indent` is the tab depth of the value's
/// opening bracket. Returns `None` for any value JSON cannot represent, so the
/// caller leaves `ExportString` unevaluated.
fn export_string_json(
  expr: &Expr,
  indent: usize,
  compact: bool,
) -> Option<String> {
  // Format a Real as JSON: a finite decimal with at least one fractional
  // digit (3.0 -> "3.0", not Wolfram's bare "3.").
  fn real_json(f: f64) -> Option<String> {
    if !f.is_finite() {
      return None;
    }
    let s = format!("{}", f);
    Some(if s.contains('.') || s.contains('e') || s.contains('E') {
      s
    } else {
      format!("{}.0", s)
    })
  }
  fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
      match c {
        '"' => out.push_str("\\\""),
        '\\' => out.push_str("\\\\"),
        '\n' => out.push_str("\\n"),
        '\t' => out.push_str("\\t"),
        '\r' => out.push_str("\\r"),
        _ => out.push(c),
      }
    }
    out
  }
  match expr {
    Expr::Integer(n) => Some(n.to_string()),
    Expr::BigInteger(n) => Some(n.to_string()),
    Expr::Real(f) => real_json(*f),
    Expr::String(s) => Some(format!("\"{}\"", escape(s))),
    Expr::Identifier(s) if s == "True" => Some("true".to_string()),
    Expr::Identifier(s) if s == "False" => Some("false".to_string()),
    Expr::Identifier(s) if s == "Null" || s == "None" => {
      Some("null".to_string())
    }
    Expr::List(items) => {
      if items.is_empty() {
        return Some("[]".to_string());
      }
      let mut parts = Vec::with_capacity(items.len());
      for it in items.iter() {
        parts.push(export_string_json(it, indent + 1, compact)?);
      }
      if compact {
        return Some(format!("[{}]", parts.join(",")));
      }
      let inner = "\t".repeat(indent + 1);
      let body: Vec<String> =
        parts.iter().map(|p| format!("{}{}", inner, p)).collect();
      Some(format!("[\n{}\n{}]", body.join(",\n"), "\t".repeat(indent)))
    }
    Expr::Association(pairs) => {
      if pairs.is_empty() {
        return Some("{}".to_string());
      }
      let mut parts = Vec::with_capacity(pairs.len());
      for (k, v) in pairs.iter() {
        let key = match k {
          Expr::String(s) => s.clone(),
          other => crate::syntax::expr_to_string(other),
        };
        parts.push(format!(
          "\"{}\":{}",
          escape(&key),
          export_string_json(v, indent + 1, compact)?
        ));
      }
      if compact {
        return Some(format!("{{{}}}", parts.join(",")));
      }
      let inner = "\t".repeat(indent + 1);
      let body: Vec<String> =
        parts.iter().map(|p| format!("{}{}", inner, p)).collect();
      Some(format!(
        "{{\n{}\n{}}}",
        body.join(",\n"),
        "\t".repeat(indent)
      ))
    }
    _ => None,
  }
}

/// Serialize an expression to CSV (or TSV when `sep` is `\t`).
/// A list-of-lists is rendered one row per inner list; a flat list is
/// rendered one element per row. Other expressions become a single row.
/// Each row is terminated with a newline (Wolfram's `ExportString` always
/// emits a trailing newline after the last record).
fn export_string_csv(
  expr: &Expr,
  sep: char,
  quote_strings: bool,
  trailing_newline: bool,
) -> String {
  let cell = |e: &Expr| csv_cell(e, quote_strings);
  let row_strs = |row: &Expr| -> String {
    if let Expr::List(items) = row {
      items
        .iter()
        .map(&cell)
        .collect::<Vec<_>>()
        .join(&sep.to_string())
    } else {
      cell(row)
    }
  };
  let rows: Vec<String> = match expr {
    Expr::List(items) => {
      let any_nested = items.iter().any(|e| matches!(e, Expr::List(_)));
      if any_nested {
        items.iter().map(row_strs).collect()
      } else {
        items.iter().map(&cell).collect()
      }
    }
    _ => vec![cell(expr)],
  };
  let mut out = rows.join("\n");
  if trailing_newline {
    out.push('\n');
  }
  out
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
/// True if `expr` is a graphics-like value that `expr_to_svg` will render
/// to a non-trivial SVG. Used to decide whether a `List` passed to
/// `Export[..., "gif"]` should be treated as an animated frame sequence.
#[cfg(not(target_arch = "wasm32"))]
fn is_rasterizable_frame(expr: &Expr) -> bool {
  matches!(expr, Expr::Graphics { .. } | Expr::Image { .. })
    || matches!(
      expr,
      Expr::FunctionCall { name, args }
        if (name == "Graphics" || name == "Graphics3D") && !args.is_empty()
    )
}

pub(crate) fn expr_to_svg(expr: &Expr) -> String {
  match expr {
    Expr::Graphics { svg: svg_data, .. } => svg_data.clone(),
    // Legended[graphics, legend]: the wrapped graphics carry the legend
    // baked into their SVG (e.g. PeriodicTablePlot["Phase"]).
    Expr::FunctionCall { name, args }
      if name == "Legended" && !args.is_empty() =>
    {
      expr_to_svg(&args[0])
    }
    // ComputationalMusic objects render as musical-staff notation.
    Expr::FunctionCall { name, .. }
      if crate::functions::music_ast::MUSIC_OBJECT_HEADS
        .contains(&name.as_str())
        && crate::functions::music_render::music_to_svg(expr).is_some() =>
    {
      crate::functions::music_render::music_to_svg(expr).unwrap_or_default()
    }
    // A plain list of music events (e.g. {MusicNote[…], MusicNote[…]}) keeps
    // its list structure: `{ <staff>, <staff>, … }`, each element drawn as its
    // own staff rather than a bracketed expression dump.
    _ if crate::functions::music_ast::is_music_object_list(expr) => {
      if let Some(svg) = crate::functions::music_render::music_list_to_svg(expr)
      {
        svg
      } else {
        expr_text_svg(expr)
      }
    }
    Expr::Identifier(s) if s == "-Graphics-" || s == "-Graphics3D-" => {
      crate::get_captured_graphics().unwrap_or_default()
    }
    Expr::FunctionCall {
      name: gfx_name,
      args: gfx_args,
    } if gfx_name == "Graphics" && !gfx_args.is_empty() => {
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
      name: gfx_name,
      args: gfx_args,
    } if gfx_name == "Graphics3D" && !gfx_args.is_empty() => {
      if let Ok(ref rendered) =
        crate::functions::plot3d::graphics3d_ast(gfx_args)
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
    } if (grid_name == "Grid" || grid_name == "TextGrid")
      && !grid_args.is_empty() =>
    {
      if crate::functions::graphics::grid_ast(grid_args).is_ok() {
        crate::get_captured_graphics().unwrap_or_default()
      } else {
        String::new()
      }
    }
    // TraditionalForm[Grid[...]] or TraditionalForm[TextGrid[...]]
    Expr::FunctionCall {
      name: tf_name,
      args: tf_args,
    } if tf_name == "TraditionalForm"
      && tf_args.len() == 1
      && matches!(
        &tf_args[0],
        Expr::FunctionCall { name, args }
        if (name == "Grid" || name == "TextGrid") && !args.is_empty()
      ) =>
    {
      if let Expr::FunctionCall {
        args: grid_args, ..
      } = &tf_args[0]
      {
        if crate::functions::graphics::grid_ast(grid_args).is_ok() {
          crate::get_captured_graphics().unwrap_or_default()
        } else {
          String::new()
        }
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
        if (name == "Grid" || name == "TextGrid") && !args.is_empty()
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
      name: row_name,
      args: row_args,
    } if row_name == "Row" && !row_args.is_empty() => {
      if let Some(svg) = crate::row_svg_with_rendered_items(row_args) {
        svg
      } else {
        expr_text_svg(expr)
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
      name: rg_name,
      args: rg_args,
    } if rg_name == "Region" && !rg_args.is_empty() => {
      if let Some(Expr::Graphics { ref svg, .. }) =
        crate::functions::region::region_to_graphics(rg_args)
      {
        svg.clone()
      } else {
        expr_text_svg(expr)
      }
    }
    Expr::FunctionCall {
      name: pc_name,
      args: pc_args,
    } if (pc_name == "PolarCurve" || pc_name == "FilledPolarCurve")
      && !pc_args.is_empty() =>
    {
      if let Some(Expr::Graphics { ref svg, .. }) =
        crate::functions::graphics::polar_curve_to_graphics(pc_name, pc_args)
      {
        svg.clone()
      } else {
        expr_text_svg(expr)
      }
    }
    Expr::FunctionCall { name: do_name, .. } if do_name == "DateObject" => {
      crate::functions::datetime_ast::date_object_panel_svg(expr)
        .unwrap_or_else(|| expr_text_svg(expr))
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
    // QuestionObject[…] — render as a question panel (prompt, answer
    // choices / input field, Submit button).
    Expr::FunctionCall { name: qo_name, .. } if qo_name == "QuestionObject" => {
      crate::functions::assessment_render::question_object_to_svg(expr)
        .unwrap_or_else(|| expr_text_svg(expr))
    }
    // Molecule[…] — render the 2-D structure diagram, prefixed with an XML
    // declaration as wolframscript's SVG export is (a standalone document).
    Expr::FunctionCall { name: mol_name, .. } if mol_name == "Molecule" => {
      match crate::functions::molecule_render::molecule_to_svg(expr) {
        Some(svg) => {
          format!("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n{svg}")
        }
        None => expr_text_svg(expr),
      }
    }
    // MoleculePlot[mol] — render the full 2-D skeletal structure diagram.
    Expr::FunctionCall {
      name: mp_name,
      args: mp_args,
    } if mp_name == "MoleculePlot" && mp_args.len() == 1 => {
      crate::functions::molecule_render::molecule_to_svg(&mp_args[0])
        .unwrap_or_else(|| expr_text_svg(expr))
    }
    // Image[…] — embed the raster as a base64 PNG inside an <image> element
    // so the SVG stays a valid vector wrapper around the pixel data.
    Expr::Image {
      width,
      height,
      channels,
      data,
      ..
    } => crate::functions::image_ast::image_to_svg_document(
      *width, *height, *channels, data,
    ),
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
