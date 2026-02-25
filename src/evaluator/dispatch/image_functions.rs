#[allow(unused_imports)]
use super::*;

pub fn dispatch_image_functions(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "Image" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::image_ast::image_constructor_ast(args));
    }
    "ImageQ" if args.len() == 1 => {
      return Some(crate::functions::image_ast::image_q_ast(args));
    }
    "ImageDimensions" if args.len() == 1 => {
      return Some(crate::functions::image_ast::image_dimensions_ast(args));
    }
    "ImageChannels" if args.len() == 1 => {
      return Some(crate::functions::image_ast::image_channels_ast(args));
    }
    "ImageType" if args.len() == 1 => {
      return Some(crate::functions::image_ast::image_type_ast(args));
    }
    "ImageData" if args.len() == 1 => {
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

      if is_url {
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
        let ext = path.rsplit('.').next().unwrap_or("").to_lowercase();
        match ext.as_str() {
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
    _ => {}
  }
  None
}
