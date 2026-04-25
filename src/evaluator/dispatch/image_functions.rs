#[allow(unused_imports)]
use super::*;

/// Extract a lowercase file extension from a path or URL.
/// Strips any `?query` or `#fragment` first, so
/// `"http://host/file.csv?x=1"` yields `"csv"`.
fn import_extension(path: &str) -> String {
  let mut cleaned = path;
  if let Some((p, _)) = cleaned.split_once('?') {
    cleaned = p;
  }
  if let Some((p, _)) = cleaned.split_once('#') {
    cleaned = p;
  }
  cleaned.rsplit('.').next().unwrap_or("").to_lowercase()
}

pub fn dispatch_image_functions(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "Image" if !args.is_empty() && args.len() <= 2 => {
      // Invalid image data shouldn't error — return unevaluated so wrapping
      // predicates like ImageQ can still classify it as False.
      let result = crate::functions::image_ast::image_constructor_ast(args);
      return Some(match result {
        Ok(expr) => Ok(expr),
        Err(_) => Ok(Expr::FunctionCall {
          name: "Image".to_string(),
          args: args.to_vec(),
        }),
      });
    }
    "ImageQ" if args.len() == 1 => {
      return Some(crate::functions::image_ast::image_q_ast(args));
    }
    "ImageDimensions" if args.len() == 1 => {
      return Some(crate::functions::image_ast::image_dimensions_ast(args));
    }
    "ImageAspectRatio" if args.len() == 1 => {
      return Some(crate::functions::image_ast::image_aspect_ratio_ast(args));
    }
    "ImageChannels" if args.len() == 1 => {
      return Some(crate::functions::image_ast::image_channels_ast(args));
    }
    "ImageType" if args.len() == 1 => {
      return Some(crate::functions::image_ast::image_type_ast(args));
    }
    "ImageData" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::image_ast::image_data_ast(args));
    }
    "ImageColorSpace" if args.len() == 1 => {
      return Some(crate::functions::image_ast::image_color_space_ast(args));
    }
    "ColorNegate" if args.len() == 1 => {
      return Some(crate::functions::image_ast::color_negate_ast(args));
    }
    "Binarize" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::image_ast::binarize_ast(args));
    }
    "Blur" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::image_ast::blur_ast(args));
    }
    "Sharpen" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::image_ast::sharpen_ast(args));
    }
    "ImageAdjust" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::image_ast::image_adjust_ast(args));
    }
    "ImageReflect" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::image_ast::image_reflect_ast(args));
    }
    "ImageRotate" if args.len() == 2 => {
      return Some(crate::functions::image_ast::image_rotate_ast(args));
    }
    "ImageResize" if args.len() == 2 => {
      return Some(crate::functions::image_ast::image_resize_ast(args));
    }
    "ImageTrim" if args.len() == 2 => {
      return Some(crate::functions::image_ast::image_trim_ast(args));
    }
    "ImageCrop" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::image_ast::image_crop_ast(args));
    }
    "ImageTake" if args.len() >= 2 && args.len() <= 3 => {
      return Some(crate::functions::image_ast::image_take_ast(args));
    }
    "EdgeDetect" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::image_ast::edge_detect_ast(args));
    }
    "DominantColors" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::image_ast::dominant_colors_ast(args));
    }
    "ImageApply" if args.len() == 2 => {
      return Some(crate::functions::image_ast::image_apply_ast(
        args,
        &evaluate_expr_to_expr,
      ));
    }
    "ColorConvert" if args.len() == 2 => {
      if matches!(&args[0], Expr::Image { .. }) {
        return Some(crate::functions::image_ast::color_convert_ast(args));
      }
    }
    // ColorData[]: list of available data categories.
    "ColorData" if args.is_empty() => {
      return Some(Ok(Expr::List(vec![
        Expr::String("Gradients".to_string()),
        Expr::String("Indexed".to_string()),
        Expr::String("Named".to_string()),
        Expr::String("Physical".to_string()),
      ])));
    }
    "ImageCompose" if args.len() == 2 => {
      return Some(crate::functions::image_ast::image_compose_ast(args));
    }
    "ImageAdd" if args.len() == 2 => {
      return Some(crate::functions::image_ast::image_add_ast(args));
    }
    "ImageSubtract" if args.len() == 2 => {
      return Some(crate::functions::image_ast::image_subtract_ast(args));
    }
    "ImageMultiply" if args.len() == 2 => {
      return Some(crate::functions::image_ast::image_multiply_ast(args));
    }
    "RandomImage" if args.len() <= 2 => {
      return Some(crate::functions::image_ast::random_image_ast(args));
    }
    "ConstantImage" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::image_ast::constant_image_ast(args));
    }
    "ImageCollage" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::image_ast::image_collage_ast(args));
    }
    "ImageAssemble" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::image_ast::image_assemble_ast(args));
    }
    #[cfg(not(target_arch = "wasm32"))]
    "Rasterize" if !args.is_empty() => {
      return Some(crate::functions::image_ast::rasterize_ast(args));
    }
    "Import" if args.len() == 1 => {
      let path = match &args[0] {
        Expr::String(s) => s.clone(),
        _ => {
          return Some(Ok(Expr::FunctionCall {
            name: "Import".to_string(),
            args: args.to_vec(),
          }));
        }
      };
      let is_url = path.starts_with("http://") || path.starts_with("https://");
      let ext = import_extension(&path);

      if is_url {
        if ext == "csv" {
          #[cfg(not(target_arch = "wasm32"))]
          {
            return Some(crate::functions::csv_ast::csv_import_from_url(
              &path, None,
            ));
          }
          #[cfg(target_arch = "wasm32")]
          {
            return Some(crate::wasm::csv_import_from_url_wasm(&path, None));
          }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
          if ext == "txt" {
            return Some(crate::functions::xlsx_ast::download_url(&path).map(
              |bytes| {
                let content = String::from_utf8_lossy(&bytes).into_owned();
                Expr::String(crate::functions::txt_ast::strip_trailing_newline(
                  content,
                ))
              },
            ));
          }
          if matches!(ext.as_str(), "xlsx" | "xls" | "xlsb" | "ods") {
            return Some(crate::functions::xlsx_ast::xlsx_import_from_url(
              &path, None,
            ));
          }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
          return Some(crate::functions::image_ast::import_image_from_url(
            &path,
          ));
        }
        #[cfg(target_arch = "wasm32")]
        {
          return Some(crate::wasm::import_image_from_url_wasm(&path));
        }
      }

      #[cfg(not(target_arch = "wasm32"))]
      {
        match ext.as_str() {
          "csv" => {
            return Some(crate::functions::csv_ast::csv_import_file(
              &path, None,
            ));
          }
          "xlsx" | "xls" | "xlsb" | "ods" => {
            return Some(crate::functions::xlsx_ast::xlsx_import_file(
              &path, None,
            ));
          }
          "txt" => {
            return Some(
              std::fs::read_to_string(&path)
                .map(|s| {
                  Expr::String(
                    crate::functions::txt_ast::strip_trailing_newline(s),
                  )
                })
                .map_err(|e| {
                  InterpreterError::EvaluationError(format!(
                    "Import: cannot open \"{}\": {}",
                    path, e
                  ))
                }),
            );
          }
          "jpg" | "jpeg" | "png" | "gif" | "bmp" | "tiff" | "tif" => {
            return Some(crate::functions::image_ast::import_image(&path));
          }
          _ => {
            return Some(Err(InterpreterError::EvaluationError(format!(
              "Import: unsupported file format \"{}\"",
              ext
            ))));
          }
        }
      }
      #[cfg(target_arch = "wasm32")]
      {
        return Some(Err(InterpreterError::EvaluationError(
          "Import: local file access is not available in the browser".into(),
        )));
      }
    }
    "Import" if args.len() == 2 => {
      let path = match &args[0] {
        Expr::String(p) => p.clone(),
        _ => {
          return Some(Ok(Expr::FunctionCall {
            name: "Import".to_string(),
            args: args.to_vec(),
          }));
        }
      };
      let is_url = path.starts_with("http://") || path.starts_with("https://");
      let ext = import_extension(&path);

      if ext == "csv" {
        let element = match &args[1] {
          Expr::String(e) => e.clone(),
          _ => {
            return Some(Ok(Expr::FunctionCall {
              name: "Import".to_string(),
              args: args.to_vec(),
            }));
          }
        };
        if is_url {
          #[cfg(not(target_arch = "wasm32"))]
          {
            return Some(crate::functions::csv_ast::csv_import_from_url(
              &path,
              Some(&element),
            ));
          }
          #[cfg(target_arch = "wasm32")]
          {
            return Some(crate::wasm::csv_import_from_url_wasm(
              &path,
              Some(&element),
            ));
          }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
          return Some(crate::functions::csv_ast::csv_import_file(
            &path,
            Some(&element),
          ));
        }
      }

      if matches!(ext.as_str(), "xlsx" | "xls" | "xlsb" | "ods") {
        #[cfg(not(target_arch = "wasm32"))]
        {
          if is_url {
            return Some(crate::functions::xlsx_ast::xlsx_import_from_url(
              &path,
              Some(&args[1]),
            ));
          }
          return Some(crate::functions::xlsx_ast::xlsx_import_file(
            &path,
            Some(&args[1]),
          ));
        }
        #[cfg(target_arch = "wasm32")]
        {
          return Some(Err(InterpreterError::EvaluationError(
            "Import: xlsx import is not available in the browser".into(),
          )));
        }
      }

      if ext == "txt" {
        // The element must be a string like "Data", "Lines", "Words", etc.
        let element = match &args[1] {
          Expr::String(e) => e.clone(),
          _ => {
            return Some(Ok(Expr::FunctionCall {
              name: "Import".to_string(),
              args: args.to_vec(),
            }));
          }
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
          let content = if is_url {
            match crate::functions::xlsx_ast::download_url(&path) {
              Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
              Err(e) => return Some(Err(e)),
            }
          } else {
            match std::fs::read_to_string(&path) {
              Ok(s) => s,
              Err(e) => {
                return Some(Err(InterpreterError::EvaluationError(format!(
                  "Import: cannot open \"{}\": {}",
                  path, e
                ))));
              }
            }
          };
          match crate::functions::txt_ast::import_element(&content, &element) {
            Some(expr) => return Some(Ok(expr)),
            None => {
              return Some(Err(InterpreterError::EvaluationError(format!(
                "Import: unsupported element \"{}\" for txt file",
                element
              ))));
            }
          }
        }
        #[cfg(target_arch = "wasm32")]
        {
          let _ = element;
          return Some(Err(InterpreterError::EvaluationError(
            "Import: txt element import is not available in the browser".into(),
          )));
        }
      }

      // Fall through for unsupported formats
      return Some(Ok(Expr::FunctionCall {
        name: "Import".to_string(),
        args: args.to_vec(),
      }));
    }
    "ImportString" if !args.is_empty() && args.len() <= 2 => {
      let content = match &args[0] {
        Expr::String(s) => s.clone(),
        _ => {
          // wolframscript emits an error when the first argument isn't a
          // string — match that message format.
          crate::emit_message(&format!(
            "ImportString::string: First argument {} is not a string.",
            crate::syntax::expr_to_string(&args[0])
          ));
          return Some(Ok(Expr::FunctionCall {
            name: "ImportString".to_string(),
            args: args.to_vec(),
          }));
        }
      };

      // ImportString[str] — default to CSV
      // ImportString[str, "CSV"] — explicit CSV
      let format = if args.len() == 2 {
        match &args[1] {
          Expr::String(s) => s.as_str(),
          _ => {
            return Some(Ok(Expr::FunctionCall {
              name: "ImportString".to_string(),
              args: args.to_vec(),
            }));
          }
        }
      } else {
        "CSV"
      };

      if format != "CSV" {
        return Some(Ok(Expr::FunctionCall {
          name: "ImportString".to_string(),
          args: args.to_vec(),
        }));
      }

      let rows = crate::functions::csv_ast::parse_csv(&content);
      return Some(Ok(crate::functions::csv_ast::csv_import_element(
        &rows, None,
      )));
    }
    _ => {}
  }
  None
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn plain_path_extension() {
    assert_eq!(import_extension("/tmp/file.csv"), "csv");
    assert_eq!(import_extension("file.PNG"), "png");
    assert_eq!(import_extension("no_extension"), "no_extension");
  }

  #[test]
  fn url_with_query_and_fragment() {
    assert_eq!(import_extension("http://host/file.csv?x=1&y=2"), "csv");
    assert_eq!(import_extension("https://host/path/data.CSV#row=3"), "csv");
    assert_eq!(import_extension("http://host/a.png?v=1#frag"), "png");
  }
}
