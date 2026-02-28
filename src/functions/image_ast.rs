use crate::InterpreterError;
use crate::syntax::{Expr, ImageType};
use std::sync::Arc;

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Convert an Expr::Image to an `image::DynamicImage`.
pub fn expr_to_dynamic_image(
  width: u32,
  height: u32,
  channels: u8,
  data: &[f64],
) -> image::DynamicImage {
  match channels {
    1 => {
      let buf: Vec<u8> = data
        .iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
      let gray = image::GrayImage::from_raw(width, height, buf).unwrap();
      image::DynamicImage::ImageLuma8(gray)
    }
    3 => {
      let buf: Vec<u8> = data
        .iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
      let rgb = image::RgbImage::from_raw(width, height, buf).unwrap();
      image::DynamicImage::ImageRgb8(rgb)
    }
    4 => {
      let buf: Vec<u8> = data
        .iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
      let rgba = image::RgbaImage::from_raw(width, height, buf).unwrap();
      image::DynamicImage::ImageRgba8(rgba)
    }
    _ => unreachable!(),
  }
}

/// Convert an `image::DynamicImage` to an Expr::Image (normalized [0,1] f64).
fn dynamic_image_to_expr(img: &image::DynamicImage) -> Expr {
  let width = img.width();
  let height = img.height();
  match img {
    image::DynamicImage::ImageLuma8(g) => {
      let data: Vec<f64> =
        g.as_raw().iter().map(|&v| v as f64 / 255.0).collect();
      Expr::Image {
        width,
        height,
        channels: 1,
        data: Arc::new(data),
        image_type: ImageType::Byte,
      }
    }
    image::DynamicImage::ImageRgba8(rgba) => {
      let data: Vec<f64> =
        rgba.as_raw().iter().map(|&v| v as f64 / 255.0).collect();
      Expr::Image {
        width,
        height,
        channels: 4,
        data: Arc::new(data),
        image_type: ImageType::Byte,
      }
    }
    _ => {
      // Convert to RGB8 for other formats
      let rgb = img.to_rgb8();
      let data: Vec<f64> =
        rgb.as_raw().iter().map(|&v| v as f64 / 255.0).collect();
      Expr::Image {
        width,
        height,
        channels: 3,
        data: Arc::new(data),
        image_type: ImageType::Byte,
      }
    }
  }
}

/// Encode an Expr::Image as a base64 PNG data URI `<img>` tag.
pub fn image_to_html_img(
  width: u32,
  height: u32,
  channels: u8,
  data: &[f64],
) -> String {
  let dyn_img = expr_to_dynamic_image(width, height, channels, data);
  let mut buf = Vec::new();
  dyn_img
    .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
    .expect("PNG encoding failed");
  let b64 =
    base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &buf);
  format!(
    "<svg xmlns='http://www.w3.org/2000/svg' width='{}' height='{}'>\
     <image href='data:image/png;base64,{}' width='{}' height='{}'/>\
     </svg>",
    width, height, b64, width, height
  )
}

// ─── Core functions (Phase 1) ──────────────────────────────────────────────

/// Image[data] or Image[data, type] - Construct image from nested lists.
/// `{{0,0.5,1},{1,0.5,0}}` → grayscale 3x2
/// `{{{r,g,b},...},...}` → RGB
/// Optional second arg: "Byte", "Bit16", "Real32", "Real64"
pub fn image_constructor_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Image expects 1 or 2 arguments".into(),
    ));
  }

  // Parse optional type specifier
  let requested_type = if args.len() == 2 {
    match &args[1] {
      Expr::String(s) | Expr::Identifier(s) => match s.as_str() {
        "Byte" => Some(ImageType::Byte),
        "Bit16" => Some(ImageType::Bit16),
        "Real32" => Some(ImageType::Real32),
        "Real64" => Some(ImageType::Real64),
        _ => None,
      },
      _ => None,
    }
  } else {
    None
  };

  // The argument should be a 2D or 3D nested list
  let rows = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Image expects a nested list of pixel data".into(),
      ));
    }
  };

  if rows.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "Image: pixel data must not be empty".into(),
    ));
  }

  // Determine if this is grayscale (2D: {{v,...},...}) or color (3D: {{{r,g,b},...},...})
  let first_row = match &rows[0] {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Image expects a 2D or 3D nested list".into(),
      ));
    }
  };

  if first_row.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "Image: rows must not be empty".into(),
    ));
  }

  // Check first element of first row: if it's a List, we have color channels
  let is_color = matches!(&first_row[0], Expr::List(_));

  let height = rows.len() as u32;

  if is_color {
    // Color image: {{{r,g,b}, ...}, ...}
    let width = first_row.len() as u32;
    let channels = match &first_row[0] {
      Expr::List(pixel) => pixel.len() as u8,
      _ => unreachable!(),
    };
    if channels != 3 && channels != 4 {
      return Err(InterpreterError::EvaluationError(format!(
        "Image: color pixels must have 3 (RGB) or 4 (RGBA) channels, got {}",
        channels
      )));
    }

    let expected_len =
      (width as usize) * (height as usize) * (channels as usize);
    let mut data = Vec::with_capacity(expected_len);

    for (i, row) in rows.iter().enumerate() {
      let row_items = match row {
        Expr::List(items) => items,
        _ => {
          return Err(InterpreterError::EvaluationError(format!(
            "Image: row {} is not a list",
            i
          )));
        }
      };
      if row_items.len() as u32 != width {
        return Err(InterpreterError::EvaluationError(format!(
          "Image: row {} has {} pixels, expected {}",
          i,
          row_items.len(),
          width
        )));
      }
      for pixel in row_items {
        let pixel_vals = match pixel {
          Expr::List(vals) => vals,
          _ => {
            return Err(InterpreterError::EvaluationError(
              "Image: each pixel must be a list of channel values".into(),
            ));
          }
        };
        if pixel_vals.len() as u8 != channels {
          return Err(InterpreterError::EvaluationError(format!(
            "Image: pixel has {} channels, expected {}",
            pixel_vals.len(),
            channels
          )));
        }
        for v in pixel_vals {
          data.push(expr_to_f64(v)?);
        }
      }
    }

    Ok(Expr::Image {
      width,
      height,
      channels,
      data: Arc::new(data),
      image_type: requested_type.unwrap_or(ImageType::Real32),
    })
  } else {
    // Grayscale image: {{v, v, ...}, ...}
    let width = first_row.len() as u32;
    let expected_len = (width as usize) * (height as usize);
    let mut data = Vec::with_capacity(expected_len);

    for (i, row) in rows.iter().enumerate() {
      let row_items = match row {
        Expr::List(items) => items,
        _ => {
          return Err(InterpreterError::EvaluationError(format!(
            "Image: row {} is not a list",
            i
          )));
        }
      };
      if row_items.len() as u32 != width {
        return Err(InterpreterError::EvaluationError(format!(
          "Image: row {} has {} values, expected {}",
          i,
          row_items.len(),
          width
        )));
      }
      for v in row_items {
        data.push(expr_to_f64(v)?);
      }
    }

    Ok(Expr::Image {
      width,
      height,
      channels: 1,
      data: Arc::new(data),
      image_type: requested_type.unwrap_or(ImageType::Real32),
    })
  }
}

/// Helper: extract f64 from Expr, resolving constants and arithmetic
fn expr_to_f64(expr: &Expr) -> Result<f64, InterpreterError> {
  crate::functions::math_ast::try_eval_to_f64(expr).ok_or_else(|| {
    InterpreterError::EvaluationError(format!(
      "Image: expected a number, got {}",
      crate::syntax::expr_to_string(expr)
    ))
  })
}

/// ImageQ[expr] - True if Expr::Image, False otherwise
pub fn image_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ImageQ expects exactly 1 argument".into(),
    ));
  }
  Ok(Expr::Identifier(
    if matches!(&args[0], Expr::Image { .. }) {
      "True"
    } else {
      "False"
    }
    .to_string(),
  ))
}

/// ImageDimensions[img] - {width, height}
pub fn image_dimensions_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ImageDimensions expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Image { width, height, .. } => Ok(Expr::List(vec![
      Expr::Integer(*width as i128),
      Expr::Integer(*height as i128),
    ])),
    _ => Ok(Expr::FunctionCall {
      name: "ImageDimensions".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// ImageChannels[img] - Channel count (1/3/4)
pub fn image_channels_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ImageChannels expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Image { channels, .. } => Ok(Expr::Integer(*channels as i128)),
    _ => Err(InterpreterError::EvaluationError(
      "ImageChannels: argument is not an Image".into(),
    )),
  }
}

/// ImageType[img] - "Byte", "Bit16", "Real32", "Real64"
pub fn image_type_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ImageType expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Image { image_type, .. } => {
      let type_str = match image_type {
        ImageType::Bit => "Bit",
        ImageType::Byte => "Byte",
        ImageType::Bit16 => "Bit16",
        ImageType::Real32 => "Real32",
        ImageType::Real64 => "Real64",
      };
      Ok(Expr::String(type_str.to_string()))
    }
    _ => Err(InterpreterError::EvaluationError(
      "ImageType: argument is not an Image".into(),
    )),
  }
}

/// ImageData[img] - Extract pixel values as nested List
pub fn image_data_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ImageData expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Image {
      width,
      height,
      channels,
      data,
      image_type,
    } => {
      let w = *width as usize;
      let h = *height as usize;
      let ch = *channels as usize;

      // Convert values to the appropriate precision.
      // Bit images (from Binarize) return integers 0 and 1.
      // Real32 images store f64 internally but should output f32-precision values.
      let to_expr = |v: f64| -> Expr {
        match image_type {
          crate::syntax::ImageType::Bit => Expr::Integer(v.round() as i128),
          crate::syntax::ImageType::Real32 => Expr::Real((v as f32) as f64),
          _ => Expr::Real(v),
        }
      };

      let mut rows = Vec::with_capacity(h);
      for y in 0..h {
        if ch == 1 {
          // Grayscale: {{v, v, ...}, ...}
          let mut row = Vec::with_capacity(w);
          for x in 0..w {
            let idx = y * w + x;
            row.push(to_expr(data[idx]));
          }
          rows.push(Expr::List(row));
        } else {
          // Color: {{{r, g, b}, ...}, ...}
          let mut row = Vec::with_capacity(w);
          for x in 0..w {
            let base = (y * w + x) * ch;
            let pixel: Vec<Expr> =
              (0..ch).map(|c| to_expr(data[base + c])).collect();
            row.push(Expr::List(pixel));
          }
          rows.push(Expr::List(row));
        }
      }

      Ok(Expr::List(rows))
    }
    _ => Err(InterpreterError::EvaluationError(
      "ImageData: argument is not an Image".into(),
    )),
  }
}

/// ImageColorSpace[img] - "Grayscale" or "RGB"
pub fn image_color_space_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ImageColorSpace expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Image { .. } => {
      // Wolfram returns Automatic for ImageColorSpace (matching wolframscript)
      Ok(Expr::Identifier("Automatic".to_string()))
    }
    _ => Err(InterpreterError::EvaluationError(
      "ImageColorSpace: argument is not an Image".into(),
    )),
  }
}

// ─── Processing functions (Phase 2) ────────────────────────────────────────

/// ColorNegate[img] - Each value v → 1-v (alpha unchanged)
pub fn color_negate_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ColorNegate expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Image {
      width,
      height,
      channels,
      data,
      image_type,
    } => {
      let ch = *channels as usize;
      let mut new_data = Vec::with_capacity(data.len());

      if ch == 4 {
        // RGBA: negate R,G,B but keep alpha
        for i in 0..data.len() {
          if i % 4 == 3 {
            new_data.push(data[i]); // alpha unchanged
          } else {
            new_data.push(1.0 - data[i]);
          }
        }
      } else {
        for v in data.iter() {
          new_data.push(1.0 - v);
        }
      }

      Ok(Expr::Image {
        width: *width,
        height: *height,
        channels: *channels,
        data: Arc::new(new_data),
        image_type: *image_type,
      })
    }
    _ => Err(InterpreterError::EvaluationError(
      "ColorNegate: argument is not an Image".into(),
    )),
  }
}

/// Binarize[img] or Binarize[img, threshold]
pub fn binarize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Binarize expects 1 or 2 arguments".into(),
    ));
  }

  let threshold = if args.len() == 2 {
    expr_to_f64(&args[1])?
  } else {
    0.5
  };

  match &args[0] {
    Expr::Image {
      width,
      height,
      channels,
      data,
      ..
    } => {
      let ch = *channels as usize;
      let w = *width as usize;
      let h = *height as usize;

      // If color, convert to grayscale first for comparison, output is 1-channel
      let mut new_data = Vec::with_capacity(w * h);
      for y in 0..h {
        for x in 0..w {
          let lum = if ch == 1 {
            data[y * w + x]
          } else {
            let base = (y * w + x) * ch;
            // Luminance: 0.299*R + 0.587*G + 0.114*B
            0.299 * data[base] + 0.587 * data[base + 1] + 0.114 * data[base + 2]
          };
          new_data.push(if lum >= threshold { 1.0 } else { 0.0 });
        }
      }

      Ok(Expr::Image {
        width: *width,
        height: *height,
        channels: 1,
        data: Arc::new(new_data),
        image_type: crate::syntax::ImageType::Bit,
      })
    }
    _ => Err(InterpreterError::EvaluationError(
      "Binarize: first argument is not an Image".into(),
    )),
  }
}

/// Blur[img] or Blur[img, r] - Gaussian blur
pub fn blur_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Blur expects 1 or 2 arguments".into(),
    ));
  }

  let sigma = if args.len() == 2 {
    expr_to_f64(&args[1])? as f32
  } else {
    1.0_f32
  };

  match &args[0] {
    Expr::Image {
      width,
      height,
      channels,
      data,
      ..
    } => {
      let dyn_img = expr_to_dynamic_image(*width, *height, *channels, data);
      let blurred = dyn_img.blur(sigma);
      Ok(dynamic_image_to_expr(&blurred))
    }
    _ => Err(InterpreterError::EvaluationError(
      "Blur: first argument is not an Image".into(),
    )),
  }
}

/// Sharpen[img] or Sharpen[img, r] - Unsharp mask
pub fn sharpen_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Sharpen expects 1 or 2 arguments".into(),
    ));
  }

  let sigma = if args.len() == 2 {
    expr_to_f64(&args[1])? as f32
  } else {
    1.0_f32
  };

  match &args[0] {
    Expr::Image {
      width,
      height,
      channels,
      data,
      ..
    } => {
      let dyn_img = expr_to_dynamic_image(*width, *height, *channels, data);
      let sharpened = dyn_img.unsharpen(sigma, 1);
      Ok(dynamic_image_to_expr(&sharpened))
    }
    _ => Err(InterpreterError::EvaluationError(
      "Sharpen: first argument is not an Image".into(),
    )),
  }
}

/// ImageAdjust[img] or ImageAdjust[img, contrast]
/// Auto-rescale to [0,1]; optional contrast adjustment
pub fn image_adjust_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "ImageAdjust expects 1 or 2 arguments".into(),
    ));
  }

  match &args[0] {
    Expr::Image {
      width,
      height,
      channels,
      data,
      image_type,
    } => {
      let contrast = if args.len() == 2 {
        expr_to_f64(&args[1])?
      } else {
        0.0
      };

      // Find min and max values
      let mut min_val = f64::MAX;
      let mut max_val = f64::MIN;
      for &v in data.iter() {
        if v < min_val {
          min_val = v;
        }
        if v > max_val {
          max_val = v;
        }
      }

      let range = max_val - min_val;
      let new_data: Vec<f64> = if range > 0.0 {
        data
          .iter()
          .map(|&v| {
            let normalized = (v - min_val) / range;
            // Apply contrast: shift midtones away from 0.5
            if contrast != 0.0 {
              let shifted = (normalized - 0.5) * (1.0 + contrast) + 0.5;
              shifted.clamp(0.0, 1.0)
            } else {
              normalized
            }
          })
          .collect()
      } else {
        vec![0.5; data.len()]
      };

      Ok(Expr::Image {
        width: *width,
        height: *height,
        channels: *channels,
        data: Arc::new(new_data),
        image_type: *image_type,
      })
    }
    _ => Err(InterpreterError::EvaluationError(
      "ImageAdjust: first argument is not an Image".into(),
    )),
  }
}

/// ImageReflect[img] - Flip left-right (default)
/// ImageReflect[img, Top -> Bottom] for vertical flip
pub fn image_reflect_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "ImageReflect expects 1 or 2 arguments".into(),
    ));
  }

  // Determine flip direction
  let vertical = if args.len() == 2 {
    // Check for Top -> Bottom rule
    match &args[1] {
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => {
        matches!(
          (pattern.as_ref(), replacement.as_ref()),
          (Expr::Identifier(p), Expr::Identifier(r))
            if p == "Top" && r == "Bottom"
        )
      }
      _ => false,
    }
  } else {
    false
  };

  match &args[0] {
    Expr::Image {
      width,
      height,
      channels,
      data,
      ..
    } => {
      let dyn_img = expr_to_dynamic_image(*width, *height, *channels, data);
      let flipped = if vertical {
        dyn_img.flipv()
      } else {
        dyn_img.fliph()
      };
      Ok(dynamic_image_to_expr(&flipped))
    }
    _ => Err(InterpreterError::EvaluationError(
      "ImageReflect: first argument is not an Image".into(),
    )),
  }
}

/// ImageRotate[img, angle] - Rotate by angle in radians
pub fn image_rotate_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ImageRotate expects exactly 2 arguments".into(),
    ));
  }

  let angle = expr_to_f64(&args[1])?;

  match &args[0] {
    Expr::Image {
      width,
      height,
      channels,
      data,
      ..
    } => {
      let dyn_img = expr_to_dynamic_image(*width, *height, *channels, data);

      // Normalize angle to [0, 2π)
      let pi = std::f64::consts::PI;
      let norm = ((angle % (2.0 * pi)) + 2.0 * pi) % (2.0 * pi);

      // Snap to nearest 90° increment
      let rotated = if (norm - pi / 2.0).abs() < 0.01 {
        dyn_img.rotate90()
      } else if (norm - pi).abs() < 0.01 {
        dyn_img.rotate180()
      } else if (norm - 3.0 * pi / 2.0).abs() < 0.01 {
        dyn_img.rotate270()
      } else if norm < 0.01 || (norm - 2.0 * pi).abs() < 0.01 {
        dyn_img
      } else {
        // For arbitrary angles, we only support 90° increments
        // Return unevaluated for now
        return Ok(Expr::FunctionCall {
          name: "ImageRotate".to_string(),
          args: args.to_vec(),
        });
      };
      Ok(dynamic_image_to_expr(&rotated))
    }
    _ => Err(InterpreterError::EvaluationError(
      "ImageRotate: first argument is not an Image".into(),
    )),
  }
}

/// ImageResize[img, {w, h}] - Resize to target dimensions
pub fn image_resize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ImageResize expects exactly 2 arguments".into(),
    ));
  }

  let (new_w, new_h) = match &args[1] {
    Expr::List(dims) if dims.len() == 2 => {
      let w = expr_to_f64(&dims[0])? as u32;
      let h = expr_to_f64(&dims[1])? as u32;
      (w, h)
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "ImageResize: second argument must be {width, height}".into(),
      ));
    }
  };

  match &args[0] {
    Expr::Image {
      width,
      height,
      channels,
      data,
      ..
    } => {
      let dyn_img = expr_to_dynamic_image(*width, *height, *channels, data);
      let resized = dyn_img.resize_exact(
        new_w,
        new_h,
        image::imageops::FilterType::Lanczos3,
      );
      Ok(dynamic_image_to_expr(&resized))
    }
    _ => Err(InterpreterError::EvaluationError(
      "ImageResize: first argument is not an Image".into(),
    )),
  }
}

/// ImageCrop[img, {{x1,y1},{x2,y2}}] - Crop to region
pub fn image_crop_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "ImageCrop expects 1 or 2 arguments".into(),
    ));
  }

  match &args[0] {
    Expr::Image {
      width,
      height,
      channels,
      data,
      ..
    } => {
      if args.len() == 2 {
        // Wolfram's ImageCrop[image, size] expects a flat {w,h} size spec.
        // Nested lists like {{x1,y1},{x2,y2}} are not valid — return unevaluated.
        return Ok(Expr::FunctionCall {
          name: "ImageCrop".to_string(),
          args: args.to_vec(),
        });
      }
      // Auto-crop: trim uniform border
      let dyn_img = expr_to_dynamic_image(*width, *height, *channels, data);
      let cropped = auto_crop(&dyn_img);
      Ok(dynamic_image_to_expr(&cropped))
    }
    _ => Err(InterpreterError::EvaluationError(
      "ImageCrop: first argument is not an Image".into(),
    )),
  }
}

/// Simple auto-crop: trim borders that match the top-left corner pixel
fn auto_crop(img: &image::DynamicImage) -> image::DynamicImage {
  use image::GenericImageView;
  let (w, h) = img.dimensions();
  if w == 0 || h == 0 {
    return img.clone();
  }

  let corner = img.get_pixel(0, 0);
  let threshold = 10u8;

  let pixel_matches = |x: u32, y: u32| -> bool {
    let p = img.get_pixel(x, y);
    p.0
      .iter()
      .zip(corner.0.iter())
      .all(|(a, b)| (*a as i16 - *b as i16).unsigned_abs() <= threshold as u16)
  };

  // Find top
  let mut top = 0;
  'top: for y in 0..h {
    for x in 0..w {
      if !pixel_matches(x, y) {
        top = y;
        break 'top;
      }
    }
    top = y + 1;
  }

  // Find bottom
  let mut bottom = h;
  'bottom: for y in (0..h).rev() {
    for x in 0..w {
      if !pixel_matches(x, y) {
        bottom = y + 1;
        break 'bottom;
      }
    }
    bottom = y;
  }

  // Find left
  let mut left = 0;
  'left: for x in 0..w {
    for y in top..bottom {
      if !pixel_matches(x, y) {
        left = x;
        break 'left;
      }
    }
    left = x + 1;
  }

  // Find right
  let mut right = w;
  'right: for x in (0..w).rev() {
    for y in top..bottom {
      if !pixel_matches(x, y) {
        right = x + 1;
        break 'right;
      }
    }
    right = x;
  }

  if left >= right || top >= bottom {
    return img.clone();
  }

  img.crop_imm(left, top, right - left, bottom - top)
}

// ─── Advanced functions (Phase 3) ──────────────────────────────────────────

/// EdgeDetect[img], EdgeDetect[img, r], EdgeDetect[img, r, t]
/// Gaussian smoothing + Sobel edge detection + binarization.
/// r = Gaussian blur radius (default 2), t = threshold (default: automatic via Otsu).
pub fn edge_detect_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "EdgeDetect expects 1 to 3 arguments".into(),
    ));
  }

  // r is the Gaussian kernel radius; σ = r/√2 (Wolfram: variance = r²/2)
  let r = if args.len() >= 2 {
    expr_to_f64(&args[1])?
  } else {
    2.0
  };
  let gauss_radius = r.round().max(0.0) as usize;
  let sigma = r / std::f64::consts::SQRT_2;

  let user_threshold = if args.len() == 3 {
    Some(expr_to_f64(&args[2])?)
  } else {
    None
  };

  match &args[0] {
    Expr::Image {
      width,
      height,
      channels,
      data,
      ..
    } => {
      let w = *width as usize;
      let h = *height as usize;
      let ch = *channels as usize;

      // Convert to grayscale luminance
      let mut gray = vec![0.0_f64; w * h];
      for y in 0..h {
        for x in 0..w {
          gray[y * w + x] = if ch == 1 {
            data[y * w + x]
          } else {
            let base = (y * w + x) * ch;
            0.299 * data[base] + 0.587 * data[base + 1] + 0.114 * data[base + 2]
          };
        }
      }

      // Gaussian pre-smoothing (separable, clamped borders)
      if sigma > 0.0 && gauss_radius > 0 {
        let kernel = gaussian_kernel_1d(sigma, gauss_radius);
        gray = separable_convolve(&gray, w, h, &kernel, gauss_radius);
      }

      // Sobel gradient computation with clamped border access
      let gx_kernel: [[f64; 3]; 3] =
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
      let gy_kernel: [[f64; 3]; 3] =
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

      let mut magnitudes = vec![0.0_f64; w * h];
      let mut directions = vec![0.0_f64; w * h];

      let clamp_y = |y: i32| y.clamp(0, h as i32 - 1) as usize;
      let clamp_x = |x: i32| x.clamp(0, w as i32 - 1) as usize;

      for y in 0..h {
        for x in 0..w {
          let mut gx = 0.0;
          let mut gy = 0.0;
          for ky in 0..3_i32 {
            for kx in 0..3_i32 {
              let sy = clamp_y(y as i32 + ky - 1);
              let sx = clamp_x(x as i32 + kx - 1);
              let pixel = gray[sy * w + sx];
              gx += pixel * gx_kernel[ky as usize][kx as usize];
              gy += pixel * gy_kernel[ky as usize][kx as usize];
            }
          }
          magnitudes[y * w + x] = (gx * gx + gy * gy).sqrt();
          directions[y * w + x] = gy.atan2(gx);
        }
      }

      // Non-maximum suppression: thin edges to 1 pixel width
      let mut nms = vec![0.0_f64; w * h];
      for y in 0..h {
        for x in 0..w {
          let mag = magnitudes[y * w + x];
          if mag == 0.0 {
            continue;
          }
          let angle = directions[y * w + x];
          // Quantize gradient direction to one of 4 axes (0°, 45°, 90°, 135°)
          // and compare magnitude to neighbors along that direction
          let (dy, dx) = quantize_gradient_direction(angle);
          let n1y = clamp_y(y as i32 + dy);
          let n1x = clamp_x(x as i32 + dx);
          let n2y = clamp_y(y as i32 - dy);
          let n2x = clamp_x(x as i32 - dx);
          let m1 = magnitudes[n1y * w + n1x];
          let m2 = magnitudes[n2y * w + n2x];
          if mag >= m1 && mag >= m2 {
            nms[y * w + x] = mag;
          }
        }
      }

      // Binarize: determine threshold and apply
      let max_mag = nms.iter().cloned().fold(0.0_f64, f64::max);

      // If the maximum gradient is negligible, the image is effectively uniform
      let result = if max_mag < 1e-10 {
        vec![0.0_f64; w * h]
      } else {
        let threshold = if let Some(t) = user_threshold {
          t * max_mag
        } else {
          otsu_threshold(&nms)
        };
        nms
          .iter()
          .map(|&m| if m >= threshold { 1.0 } else { 0.0 })
          .collect()
      };

      Ok(Expr::Image {
        width: *width,
        height: *height,
        channels: 1,
        data: Arc::new(result),
        image_type: ImageType::Bit,
      })
    }
    _ => Err(InterpreterError::EvaluationError(
      "EdgeDetect: argument is not an Image".into(),
    )),
  }
}

/// Build a 1D Gaussian kernel with the given sigma, truncated at ±radius.
fn gaussian_kernel_1d(sigma: f64, radius: usize) -> Vec<f64> {
  let mut kernel = Vec::with_capacity(2 * radius + 1);
  let mut sum = 0.0;
  for i in 0..=(2 * radius) {
    let x = i as f64 - radius as f64;
    let val = (-x * x / (2.0 * sigma * sigma)).exp();
    kernel.push(val);
    sum += val;
  }
  for v in &mut kernel {
    *v /= sum;
  }
  kernel
}

/// Apply a separable convolution (horizontal then vertical) with clamped borders.
fn separable_convolve(
  data: &[f64],
  w: usize,
  h: usize,
  kernel: &[f64],
  radius: usize,
) -> Vec<f64> {
  // Horizontal pass
  let mut tmp = vec![0.0_f64; w * h];
  for y in 0..h {
    for x in 0..w {
      let mut acc = 0.0;
      for k in 0..kernel.len() {
        let sx =
          (x as i32 + k as i32 - radius as i32).clamp(0, w as i32 - 1) as usize;
        acc += data[y * w + sx] * kernel[k];
      }
      tmp[y * w + x] = acc;
    }
  }
  // Vertical pass
  let mut out = vec![0.0_f64; w * h];
  for y in 0..h {
    for x in 0..w {
      let mut acc = 0.0;
      for k in 0..kernel.len() {
        let sy =
          (y as i32 + k as i32 - radius as i32).clamp(0, h as i32 - 1) as usize;
        acc += tmp[sy * w + x] * kernel[k];
      }
      out[y * w + x] = acc;
    }
  }
  out
}

/// Compute an automatic threshold using Otsu's method.
/// Finds the threshold that minimizes within-class variance of the values.
fn otsu_threshold(data: &[f64]) -> f64 {
  let n_bins = 256;
  let max_val = data.iter().cloned().fold(0.0_f64, f64::max);
  if max_val <= 0.0 {
    return 0.0;
  }

  // Build histogram
  let mut hist = vec![0u64; n_bins];
  for &v in data {
    let bin = ((v / max_val) * (n_bins - 1) as f64).round() as usize;
    hist[bin.min(n_bins - 1)] += 1;
  }

  let total = data.len() as f64;
  let mut sum_total = 0.0;
  for (i, &count) in hist.iter().enumerate() {
    sum_total += i as f64 * count as f64;
  }

  let mut best_thresh = 0.0_f64;
  let mut best_var = -1.0_f64;
  let mut w0 = 0.0_f64;
  let mut sum0 = 0.0_f64;

  for (i, &count) in hist.iter().enumerate() {
    w0 += count as f64;
    if w0 == 0.0 {
      continue;
    }
    let w1 = total - w0;
    if w1 == 0.0 {
      break;
    }
    sum0 += i as f64 * count as f64;
    let mean0 = sum0 / w0;
    let mean1 = (sum_total - sum0) / w1;
    let between_var = w0 * w1 * (mean0 - mean1) * (mean0 - mean1);
    if between_var > best_var {
      best_var = between_var;
      best_thresh = (i as f64 + 0.5) / (n_bins - 1) as f64;
    }
  }

  best_thresh * max_val
}

/// Quantize gradient direction to one of 4 axes and return (dy, dx) neighbor offset.
fn quantize_gradient_direction(angle: f64) -> (i32, i32) {
  use std::f64::consts::PI;
  // Normalize angle to [0, π) — we only care about the axis, not the sign
  let a = ((angle % PI) + PI) % PI;
  if !(PI / 8.0..7.0 * PI / 8.0).contains(&a) {
    (0, 1) // horizontal edge → compare left/right
  } else if a < 3.0 * PI / 8.0 {
    (1, 1) // 45° diagonal
  } else if a < 5.0 * PI / 8.0 {
    (1, 0) // vertical edge → compare up/down
  } else {
    (-1, 1) // 135° diagonal
  }
}

/// DominantColors[img] or DominantColors[img, n] - K-means clustering
pub fn dominant_colors_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "DominantColors expects 1 or 2 arguments".into(),
    ));
  }

  let n = if args.len() == 2 {
    expr_to_f64(&args[1])? as usize
  } else {
    5
  };

  match &args[0] {
    Expr::Image {
      width,
      height,
      channels,
      data,
      ..
    } => {
      let w = *width as usize;
      let h = *height as usize;
      let ch = *channels as usize;

      if ch < 3 {
        return Err(InterpreterError::EvaluationError(
          "DominantColors: requires an RGB or RGBA image".into(),
        ));
      }

      // Collect RGB pixels
      let num_pixels = w * h;
      let mut pixels: Vec<[f64; 3]> = Vec::with_capacity(num_pixels);
      for i in 0..num_pixels {
        let base = i * ch;
        pixels.push([data[base], data[base + 1], data[base + 2]]);
      }

      // Simple k-means
      let k = n.min(num_pixels);
      let centers = kmeans_colors(&pixels, k, 20);

      // Return as list of RGBColor[r, g, b]
      let colors: Vec<Expr> = centers
        .iter()
        .map(|c| Expr::FunctionCall {
          name: "RGBColor".to_string(),
          args: vec![Expr::Real(c[0]), Expr::Real(c[1]), Expr::Real(c[2])],
        })
        .collect();

      Ok(Expr::List(colors))
    }
    _ => Err(InterpreterError::EvaluationError(
      "DominantColors: first argument is not an Image".into(),
    )),
  }
}

/// Simple k-means for color quantization
fn kmeans_colors(
  pixels: &[[f64; 3]],
  k: usize,
  max_iters: usize,
) -> Vec<[f64; 3]> {
  if pixels.is_empty() || k == 0 {
    return vec![];
  }

  // Initialize centers by evenly sampling from pixels
  let step = pixels.len().max(1) / k.max(1);
  let mut centers: Vec<[f64; 3]> = (0..k)
    .map(|i| pixels[(i * step).min(pixels.len() - 1)])
    .collect();

  let mut assignments = vec![0usize; pixels.len()];

  for _ in 0..max_iters {
    // Assign each pixel to nearest center
    let mut changed = false;
    for (i, pixel) in pixels.iter().enumerate() {
      let mut best = 0;
      let mut best_dist = f64::MAX;
      for (j, center) in centers.iter().enumerate() {
        let dist = (pixel[0] - center[0]).powi(2)
          + (pixel[1] - center[1]).powi(2)
          + (pixel[2] - center[2]).powi(2);
        if dist < best_dist {
          best_dist = dist;
          best = j;
        }
      }
      if assignments[i] != best {
        assignments[i] = best;
        changed = true;
      }
    }

    if !changed {
      break;
    }

    // Update centers
    let mut sums = vec![[0.0_f64; 3]; k];
    let mut counts = vec![0usize; k];
    for (i, pixel) in pixels.iter().enumerate() {
      let c = assignments[i];
      sums[c][0] += pixel[0];
      sums[c][1] += pixel[1];
      sums[c][2] += pixel[2];
      counts[c] += 1;
    }
    for j in 0..k {
      if counts[j] > 0 {
        centers[j] = [
          sums[j][0] / counts[j] as f64,
          sums[j][1] / counts[j] as f64,
          sums[j][2] / counts[j] as f64,
        ];
      }
    }
  }

  centers
}

/// ImageApply[f, img] - Apply function to each pixel value
pub fn image_apply_ast(
  args: &[Expr],
  eval_fn: &dyn Fn(&Expr) -> Result<Expr, InterpreterError>,
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ImageApply expects exactly 2 arguments".into(),
    ));
  }

  let func = &args[0];

  match &args[1] {
    Expr::Image {
      width,
      height,
      channels,
      data,
      image_type,
    } => {
      let ch = *channels as usize;
      let w = *width as usize;
      let h = *height as usize;
      let num_pixels = w * h;
      let mut new_data = Vec::with_capacity(data.len());

      for i in 0..num_pixels {
        if ch == 1 {
          // Scalar pixel
          let pixel_expr = Expr::Real(data[i]);
          let call = Expr::FunctionCall {
            name: crate::syntax::expr_to_string(func),
            args: vec![pixel_expr],
          };
          let result = eval_fn(&call)?;
          new_data.push(expr_to_f64(&result)?);
        } else {
          // Vector pixel {r, g, b}
          let base = i * ch;
          let pixel_list: Vec<Expr> =
            (0..ch).map(|c| Expr::Real(data[base + c])).collect();
          let pixel_expr = Expr::List(pixel_list);
          let call = Expr::FunctionCall {
            name: crate::syntax::expr_to_string(func),
            args: vec![pixel_expr],
          };
          let mut result = eval_fn(&call)?;
          match &mut result {
            Expr::List(vals) => {
              for v in vals {
                new_data.push(expr_to_f64(v)?);
              }
            }
            _ => {
              // If the function returns a scalar, use it for all channels
              let val = expr_to_f64(&result)?;
              for _ in 0..ch {
                new_data.push(val);
              }
            }
          }
        }
      }

      Ok(Expr::Image {
        width: *width,
        height: *height,
        channels: *channels,
        data: Arc::new(new_data),
        image_type: *image_type,
      })
    }
    _ => Err(InterpreterError::EvaluationError(
      "ImageApply: second argument is not an Image".into(),
    )),
  }
}

/// ColorConvert[img, "Grayscale"] - Convert between color spaces
pub fn color_convert_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ColorConvert expects exactly 2 arguments".into(),
    ));
  }

  let target_space = match &args[1] {
    Expr::String(s) => s.as_str(),
    Expr::Identifier(s) => s.as_str(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "ColorConvert: second argument must be a color space string".into(),
      ));
    }
  };

  match &args[0] {
    Expr::Image {
      width,
      height,
      channels,
      data,
      image_type,
    } => {
      let w = *width as usize;
      let h = *height as usize;
      let ch = *channels as usize;

      match target_space {
        "Grayscale" => {
          if ch == 1 {
            return Ok(args[0].clone()); // Already grayscale
          }
          let mut new_data = Vec::with_capacity(w * h);
          for i in 0..(w * h) {
            let base = i * ch;
            let lum = 0.299 * data[base]
              + 0.587 * data[base + 1]
              + 0.114 * data[base + 2];
            new_data.push(lum);
          }
          Ok(Expr::Image {
            width: *width,
            height: *height,
            channels: 1,
            data: Arc::new(new_data),
            image_type: *image_type,
          })
        }
        "RGB" => {
          if ch == 3 {
            return Ok(args[0].clone()); // Already RGB
          }
          if ch == 1 {
            // Grayscale → RGB
            let mut new_data = Vec::with_capacity(w * h * 3);
            for &v in data.iter() {
              new_data.push(v);
              new_data.push(v);
              new_data.push(v);
            }
            Ok(Expr::Image {
              width: *width,
              height: *height,
              channels: 3,
              data: Arc::new(new_data),
              image_type: *image_type,
            })
          } else if ch == 4 {
            // RGBA → RGB (drop alpha)
            let mut new_data = Vec::with_capacity(w * h * 3);
            for i in 0..(w * h) {
              let base = i * 4;
              new_data.push(data[base]);
              new_data.push(data[base + 1]);
              new_data.push(data[base + 2]);
            }
            Ok(Expr::Image {
              width: *width,
              height: *height,
              channels: 3,
              data: Arc::new(new_data),
              image_type: *image_type,
            })
          } else {
            Err(InterpreterError::EvaluationError(
              "ColorConvert: unsupported channel count".into(),
            ))
          }
        }
        _ => Err(InterpreterError::EvaluationError(format!(
          "ColorConvert: unsupported color space \"{}\"",
          target_space
        ))),
      }
    }
    _ => Err(InterpreterError::EvaluationError(
      "ColorConvert: first argument is not an Image".into(),
    )),
  }
}

/// ImageCompose[img1, img2] - Overlay img2 on img1 (centered)
pub fn image_compose_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ImageCompose expects exactly 2 arguments".into(),
    ));
  }

  match (&args[0], &args[1]) {
    (
      Expr::Image {
        width: w1,
        height: h1,
        channels: ch1,
        data: data1,
        ..
      },
      Expr::Image {
        width: w2,
        height: h2,
        channels: ch2,
        data: data2,
        ..
      },
    ) => {
      let dyn1 = expr_to_dynamic_image(*w1, *h1, *ch1, data1);
      let dyn2 = expr_to_dynamic_image(*w2, *h2, *ch2, data2);

      // Overlay img2 centered on img1
      let mut base = dyn1.to_rgba8();
      let overlay = dyn2.to_rgba8();
      let offset_x = (*w1 as i64 - *w2 as i64) / 2;
      let offset_y = (*h1 as i64 - *h2 as i64) / 2;

      image::imageops::overlay(&mut base, &overlay, offset_x, offset_y);
      Ok(dynamic_image_to_expr(&image::DynamicImage::ImageRgba8(
        base,
      )))
    }
    _ => Err(InterpreterError::EvaluationError(
      "ImageCompose: both arguments must be Images".into(),
    )),
  }
}

/// Helper for pointwise image operations
fn pointwise_image_op(
  args: &[Expr],
  name: &str,
  op: fn(f64, f64) -> f64,
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(format!(
      "{} expects exactly 2 arguments",
      name
    )));
  }

  match (&args[0], &args[1]) {
    (
      Expr::Image {
        width: w1,
        height: h1,
        channels: ch1,
        data: data1,
        image_type: t1,
      },
      Expr::Image {
        width: w2,
        height: h2,
        channels: ch2,
        data: data2,
        ..
      },
    ) => {
      if w1 != w2 || h1 != h2 || ch1 != ch2 {
        return Err(InterpreterError::EvaluationError(format!(
          "{}: images must have the same dimensions and channels",
          name
        )));
      }

      let new_data: Vec<f64> = data1
        .iter()
        .zip(data2.iter())
        .map(|(&a, &b)| op(a, b))
        .collect();

      Ok(Expr::Image {
        width: *w1,
        height: *h1,
        channels: *ch1,
        data: Arc::new(new_data),
        image_type: *t1,
      })
    }
    _ => Err(InterpreterError::EvaluationError(format!(
      "{}: both arguments must be Images",
      name
    ))),
  }
}

/// ImageAdd[img1, img2] - Pointwise add, clamped [0,1]
pub fn image_add_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  pointwise_image_op(args, "ImageAdd", |a, b| a + b)
}

/// ImageSubtract[img1, img2] - Pointwise subtract
pub fn image_subtract_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  pointwise_image_op(args, "ImageSubtract", |a, b| a - b)
}

/// ImageMultiply[img1, img2] - Pointwise multiply
pub fn image_multiply_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  pointwise_image_op(args, "ImageMultiply", |a, b| a * b)
}

/// RandomImage[{w, h}] - Random pixel values using crate::with_rng
pub fn random_image_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "RandomImage expects 0 to 2 arguments".into(),
    ));
  }

  // RandomImage[] → 150x150, [0,1]
  // RandomImage[max] → 150x150, [0,max]
  // RandomImage[max, {w, h}] → w×h, [0,max]
  let max_val = if !args.is_empty() {
    expr_to_f64(&args[0])?
  } else {
    1.0
  };

  let (w, h) = if args.len() == 2 {
    match &args[1] {
      Expr::List(dims) if dims.len() == 2 => {
        let w = expr_to_f64(&dims[0])? as u32;
        let h = expr_to_f64(&dims[1])? as u32;
        (w, h)
      }
      Expr::Integer(n) => {
        let s = *n as u32;
        (s, s)
      }
      _ => {
        return Err(InterpreterError::EvaluationError(
          "RandomImage: second argument must be {width, height}".into(),
        ));
      }
    }
  } else {
    (150u32, 150u32)
  };

  use rand::Rng;
  let channels = 1u8; // Grayscale by default (matches Wolfram)
  let len = (w as usize) * (h as usize) * (channels as usize);
  let data: Vec<f64> = crate::with_rng(|rng| {
    (0..len).map(|_| rng.gen_range(0.0..max_val)).collect()
  });

  Ok(Expr::Image {
    width: w,
    height: h,
    channels,
    data: Arc::new(data),
    image_type: ImageType::Real32,
  })
}

/// ImageTake[img, n] - take first n rows
/// ImageTake[img, {r1, r2}] - take rows r1..r2 (1-indexed inclusive)
/// ImageTake[img, {r1, r2}, {c1, c2}] - take rows r1..r2 and columns c1..c2
pub fn image_take_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "ImageTake expects 2 or 3 arguments".into(),
    ));
  }

  match &args[0] {
    Expr::Image {
      width,
      height,
      channels,
      data,
      image_type,
    } => {
      let w = *width as usize;
      let h = *height as usize;
      let ch = *channels as usize;

      // Parse row spec
      let (r1, r2) = match &args[1] {
        Expr::Integer(n) => {
          let n = *n as usize;
          (0usize, n.min(h))
        }
        Expr::List(pair) if pair.len() == 2 => {
          let a = expr_to_f64(&pair[0])? as usize;
          let b = expr_to_f64(&pair[1])? as usize;
          // 1-indexed to 0-indexed
          let r1 = a.saturating_sub(1).min(h);
          let r2 = b.min(h);
          (r1, r2)
        }
        _ => {
          return Err(InterpreterError::EvaluationError(
            "ImageTake: second argument must be n or {r1, r2}".into(),
          ));
        }
      };

      // Parse column spec (optional)
      let (c1, c2) = if args.len() == 3 {
        match &args[2] {
          Expr::List(pair) if pair.len() == 2 => {
            let a = expr_to_f64(&pair[0])? as usize;
            let b = expr_to_f64(&pair[1])? as usize;
            let c1 = a.saturating_sub(1).min(w);
            let c2 = b.min(w);
            (c1, c2)
          }
          _ => {
            return Err(InterpreterError::EvaluationError(
              "ImageTake: third argument must be {c1, c2}".into(),
            ));
          }
        }
      } else {
        (0, w)
      };

      if r1 >= r2 || c1 >= c2 {
        return Err(InterpreterError::EvaluationError(
          "ImageTake: invalid range".into(),
        ));
      }

      let new_h = r2 - r1;
      let new_w = c2 - c1;
      let mut new_data = Vec::with_capacity(new_h * new_w * ch);

      for y in r1..r2 {
        for x in c1..c2 {
          let base = (y * w + x) * ch;
          for c in 0..ch {
            new_data.push(data[base + c]);
          }
        }
      }

      Ok(Expr::Image {
        width: new_w as u32,
        height: new_h as u32,
        channels: *channels,
        data: Arc::new(new_data),
        image_type: *image_type,
      })
    }
    _ => Err(InterpreterError::EvaluationError(
      "ImageTake: first argument is not an Image".into(),
    )),
  }
}

// ─── ImageCollage ─────────────────────────────────────────────────────────

/// Rectangle for collage layout
struct LayoutRect {
  x: f64,
  y: f64,
  w: f64,
  h: f64,
}

/// Lay out weighted items in rows on a canvas.
///
/// Chooses the number of rows so that cells are as square as possible,
/// distributes items across rows to balance total weight per row,
/// then assigns widths proportional to weight within each row.
/// Returns (item_index, rectangle) pairs.
fn layout_rows(
  items: &[(usize, f64)],
  canvas_w: f64,
  canvas_h: f64,
) -> Vec<(usize, LayoutRect)> {
  let n = items.len();
  if n == 0 {
    return vec![];
  }
  if n == 1 {
    return vec![(
      items[0].0,
      LayoutRect {
        x: 0.0,
        y: 0.0,
        w: canvas_w,
        h: canvas_h,
      },
    )];
  }

  // Choose number of rows so cells are closest to square.
  // For n items on a W×H canvas with r rows, each cell is roughly
  // (W * r / n) wide and (H / r) tall.  Aspect ratio = W·r² / (n·H).
  // Setting that to 1 gives r = sqrt(n · H / W).
  let r_ideal = (n as f64 * canvas_h / canvas_w).sqrt();
  let num_rows = r_ideal.round().max(1.0) as usize;
  let num_rows = num_rows.min(n); // can't have more rows than items

  // Distribute items across rows as evenly as possible.
  // base = items per row, remainder rows get one extra.
  let base = n / num_rows;
  let remainder = n % num_rows;
  let mut rows: Vec<&[(usize, f64)]> = Vec::with_capacity(num_rows);
  let mut offset = 0;
  for r in 0..num_rows {
    let count = base + if r < remainder { 1 } else { 0 };
    rows.push(&items[offset..offset + count]);
    offset += count;
  }

  // Assign each row equal height, then within each row
  // assign widths proportional to weight.
  let row_h = canvas_h / num_rows as f64;
  let mut result = Vec::with_capacity(n);
  let mut y = 0.0;

  for row_items in &rows {
    let row_weight: f64 = row_items.iter().map(|(_, w)| w).sum();
    let mut x = 0.0;
    for (i, &(idx, w)) in row_items.iter().enumerate() {
      let cell_w = if i == row_items.len() - 1 {
        // Last item in row: fill remaining width to avoid rounding gaps
        canvas_w - x
      } else {
        (w / row_weight) * canvas_w
      };
      result.push((
        idx,
        LayoutRect {
          x,
          y,
          w: cell_w,
          h: row_h,
        },
      ));
      x += cell_w;
    }
    y += row_h;
  }

  result
}

/// Extract an (image, weight) pair from an expression.
/// Handles: Image (weight=1), w*Image, Image*w, {Image, w}
fn extract_image_weight(
  expr: &Expr,
) -> Option<(u32, u32, u8, &std::sync::Arc<Vec<f64>>, &ImageType, f64)> {
  // Direct image → weight 1.0
  if let Expr::Image {
    width,
    height,
    channels,
    data,
    image_type,
  } = expr
  {
    return Some((*width, *height, *channels, data, image_type, 1.0));
  }

  // w * Image or Image * w
  if let Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Times,
    left,
    right,
  } = expr
  {
    if let Some(w) = crate::functions::math_ast::try_eval_to_f64(left)
      && let Expr::Image {
        width,
        height,
        channels,
        data,
        image_type,
      } = right.as_ref()
    {
      return Some((*width, *height, *channels, data, image_type, w));
    }
    if let Some(w) = crate::functions::math_ast::try_eval_to_f64(right)
      && let Expr::Image {
        width,
        height,
        channels,
        data,
        image_type,
      } = left.as_ref()
    {
      return Some((*width, *height, *channels, data, image_type, w));
    }
  }

  // Times[w, Image] as FunctionCall
  if let Expr::FunctionCall { name, args } = expr
    && name == "Times"
    && args.len() == 2
  {
    if let Some(w) = crate::functions::math_ast::try_eval_to_f64(&args[0])
      && let Expr::Image {
        width,
        height,
        channels,
        data,
        image_type,
      } = &args[1]
    {
      return Some((*width, *height, *channels, data, image_type, w));
    }
    if let Some(w) = crate::functions::math_ast::try_eval_to_f64(&args[1])
      && let Expr::Image {
        width,
        height,
        channels,
        data,
        image_type,
      } = &args[0]
    {
      return Some((*width, *height, *channels, data, image_type, w));
    }
  }

  None
}

/// ImageCollage[{img1, img2, ...}] — create a collage from images.
/// ImageCollage[{w1*img1, w2*img2, ...}] — weighted collage.
/// ImageCollage[{{img1, w1}, ...}] — paired format.
/// Optional fitting argument: "Fill", "Fit", "Stretch".
/// Optional size argument: width or {width, height}.
pub fn image_collage_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "ImageCollage expects 1 to 3 arguments".into(),
    ));
  }

  // Parse fitting mode (2nd arg is always the fitting string)
  let fitting = if args.len() >= 2 {
    match &args[1] {
      Expr::String(s) => match s.as_str() {
        "Fill" | "Fit" | "Stretch" => s.clone(),
        _ => "Fit".to_string(),
      },
      _ => "Fit".to_string(),
    }
  } else {
    "Fit".to_string()
  };

  // Parse output size (3rd arg only)
  let explicit_size: Option<(u32, u32)> = if args.len() == 3 {
    match &args[2] {
      Expr::Integer(w) => Some((*w as u32, 0)), // width only, height computed later
      Expr::Real(w) => Some((*w as u32, 0)),
      Expr::List(dims) if dims.len() == 2 => {
        let w = crate::functions::math_ast::try_eval_to_f64(&dims[0])
          .unwrap_or(800.0) as u32;
        let h = crate::functions::math_ast::try_eval_to_f64(&dims[1])
          .unwrap_or(600.0) as u32;
        Some((w, h))
      }
      _ => None,
    }
  } else {
    None
  };

  // Parse image list from first argument
  struct ImageEntry {
    dyn_img: image::DynamicImage,
    orig_w: u32,
    orig_h: u32,
    weight: f64,
  }

  let mut entries: Vec<ImageEntry> = Vec::new();
  let mut max_channels: u8 = 1;

  let list = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "ImageCollage: first argument must be a list of images".into(),
      ));
    }
  };

  for item in list {
    // Try direct image/weighted image
    if let Some((w, h, ch, data, _, weight)) = extract_image_weight(item) {
      if ch > max_channels {
        max_channels = ch;
      }
      entries.push(ImageEntry {
        dyn_img: expr_to_dynamic_image(w, h, ch, data),
        orig_w: w,
        orig_h: h,
        weight,
      });
      continue;
    }

    // Try paired format: {image, weight}
    if let Expr::List(pair) = item
      && pair.len() == 2
      && let Expr::Image {
        width,
        height,
        channels,
        data,
        ..
      } = &pair[0]
    {
      let weight =
        crate::functions::math_ast::try_eval_to_f64(&pair[1]).unwrap_or(1.0);
      if *channels > max_channels {
        max_channels = *channels;
      }
      entries.push(ImageEntry {
        dyn_img: expr_to_dynamic_image(*width, *height, *channels, data),
        orig_w: *width,
        orig_h: *height,
        weight,
      });
      continue;
    }

    return Err(InterpreterError::EvaluationError(
      "ImageCollage: list elements must be images, weighted images, or {image, weight} pairs".into(),
    ));
  }

  if entries.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "ImageCollage: no images provided".into(),
    ));
  }

  // Normalize weights
  let total_weight: f64 = entries.iter().map(|e| e.weight.abs()).sum();
  if total_weight < 1e-12 {
    return Err(InterpreterError::EvaluationError(
      "ImageCollage: total weight must be positive".into(),
    ));
  }
  let norm_weights: Vec<f64> = entries
    .iter()
    .map(|e| e.weight.abs() / total_weight)
    .collect();

  // Determine canvas size
  let total_area: f64 = entries
    .iter()
    .map(|e| (e.orig_w as f64) * (e.orig_h as f64))
    .sum();
  let (canvas_w, canvas_h) = match explicit_size {
    Some((w, 0)) => {
      // Width only: compute height from total area
      let h = (total_area / w as f64).max(1.0).round() as u32;
      (w, h)
    }
    Some((w, h)) => (w, h),
    None => {
      // Default: roughly preserve total area with 4:3 aspect
      let cw = (total_area * 4.0 / 3.0).sqrt().round() as u32;
      let ch = (total_area / cw as f64).round() as u32;
      (cw.max(1), ch.max(1))
    }
  };

  // Build layout items (index, weight)
  let items: Vec<(usize, f64)> = norm_weights
    .iter()
    .enumerate()
    .map(|(i, &w)| (i, w))
    .collect();

  let layout = layout_rows(&items, canvas_w as f64, canvas_h as f64);

  // Create canvas with background GrayLevel[0.2] = ~51
  let bg = image::Rgba([51u8, 51, 51, 255]);
  let mut canvas = image::RgbaImage::from_pixel(canvas_w, canvas_h, bg);

  // Place each image in its allocated rectangle
  for (idx, rect) in &layout {
    let entry = &entries[*idx];
    let cell_w = rect.w.round() as u32;
    let cell_h = rect.h.round() as u32;
    if cell_w == 0 || cell_h == 0 {
      continue;
    }

    let rgba_img = entry.dyn_img.to_rgba8();
    let placed = fit_image_to_cell(
      &rgba_img,
      entry.orig_w,
      entry.orig_h,
      cell_w,
      cell_h,
      &fitting,
      bg,
    );

    // Paste onto canvas
    image::imageops::overlay(
      &mut canvas,
      &placed,
      rect.x.round() as i64,
      rect.y.round() as i64,
    );
  }

  // Convert the RGBA canvas to the appropriate channel count
  let dyn_img = match max_channels {
    1 => image::DynamicImage::ImageLuma8(
      image::DynamicImage::ImageRgba8(canvas).to_luma8(),
    ),
    3 => image::DynamicImage::ImageRgb8(
      image::DynamicImage::ImageRgba8(canvas).to_rgb8(),
    ),
    _ => image::DynamicImage::ImageRgba8(canvas),
  };
  Ok(dynamic_image_to_expr(&dyn_img))
}

// ─── ImageAssemble ────────────────────────────────────────────────────────

/// Extract an image from an Expr, returning (DynamicImage, width, height).
/// Returns None for non-image exprs (e.g. Missing[]).
fn extract_image_for_assemble(
  expr: &Expr,
) -> Option<(image::DynamicImage, u32, u32)> {
  if let Expr::Image {
    width,
    height,
    channels,
    data,
    ..
  } = expr
  {
    Some((
      expr_to_dynamic_image(*width, *height, *channels, data),
      *width,
      *height,
    ))
  } else {
    None
  }
}

/// Resize or fit an image into a target cell according to fitting mode.
/// Returns an RGBA image of exactly (cell_w, cell_h).
fn fit_image_to_cell(
  img: &image::RgbaImage,
  orig_w: u32,
  orig_h: u32,
  cell_w: u32,
  cell_h: u32,
  fitting: &str,
  bg: image::Rgba<u8>,
) -> image::RgbaImage {
  match fitting {
    "Stretch" => image::imageops::resize(
      img,
      cell_w,
      cell_h,
      image::imageops::FilterType::Lanczos3,
    ),
    "Fit" => {
      let scale_w = cell_w as f64 / orig_w as f64;
      let scale_h = cell_h as f64 / orig_h as f64;
      let scale = scale_w.min(scale_h);
      let new_w = (orig_w as f64 * scale).round() as u32;
      let new_h = (orig_h as f64 * scale).round() as u32;
      let resized = image::imageops::resize(
        img,
        new_w.max(1),
        new_h.max(1),
        image::imageops::FilterType::Lanczos3,
      );
      let mut cell = image::RgbaImage::from_pixel(cell_w, cell_h, bg);
      let ox = cell_w.saturating_sub(new_w) / 2;
      let oy = cell_h.saturating_sub(new_h) / 2;
      image::imageops::overlay(&mut cell, &resized, ox as i64, oy as i64);
      cell
    }
    "Fill" => {
      let scale_w = cell_w as f64 / orig_w as f64;
      let scale_h = cell_h as f64 / orig_h as f64;
      let scale = scale_w.max(scale_h);
      let new_w = (orig_w as f64 * scale).round() as u32;
      let new_h = (orig_h as f64 * scale).round() as u32;
      let resized = image::imageops::resize(
        img,
        new_w.max(1),
        new_h.max(1),
        image::imageops::FilterType::Lanczos3,
      );
      let cx = new_w.saturating_sub(cell_w) / 2;
      let cy = new_h.saturating_sub(cell_h) / 2;
      image::imageops::crop_imm(
        &resized,
        cx,
        cy,
        cell_w.min(new_w),
        cell_h.min(new_h),
      )
      .to_image()
    }
    _ => {
      // None / default: place as-is, centered, with background fill
      let mut cell = image::RgbaImage::from_pixel(cell_w, cell_h, bg);
      let ox = cell_w.saturating_sub(orig_w) / 2;
      let oy = cell_h.saturating_sub(orig_h) / 2;
      image::imageops::overlay(&mut cell, img, ox as i64, oy as i64);
      cell
    }
  }
}

/// ImageAssemble[{{im11,...,im1n},...,{imm1,...,immn}}] — assemble grid.
/// ImageAssemble[{im1,...,imn}] — assemble as single row.
/// ImageAssemble[grid, fitting] — with fitting mode.
pub fn image_assemble_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "ImageAssemble expects 1 or 2 arguments".into(),
    ));
  }

  // Parse fitting mode
  let fitting = if args.len() == 2 {
    match &args[1] {
      Expr::String(s) => s.clone(),
      Expr::Identifier(s) if s == "None" => "None".to_string(),
      _ => "None".to_string(),
    }
  } else {
    "None".to_string()
  };

  let bg = image::Rgba([0u8, 0, 0, 255]); // Black background (Automatic)

  // Parse the grid from the first argument
  let outer = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "ImageAssemble: first argument must be a list".into(),
      ));
    }
  };

  if outer.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "ImageAssemble: empty list".into(),
    ));
  }

  // Determine max channels across all input images
  let mut max_channels: u8 = 1;
  fn count_channels_in_list(items: &[Expr], max_ch: &mut u8) {
    for item in items {
      if let Expr::Image { channels, .. } = item {
        if *channels > *max_ch {
          *max_ch = *channels;
        }
      } else if let Expr::List(sub) = item {
        count_channels_in_list(sub, max_ch);
      }
    }
  }
  count_channels_in_list(outer, &mut max_channels);

  // Determine if it's a 2D grid or a flat list (single row)
  let is_2d = outer.iter().all(|e| matches!(e, Expr::List(_)));

  // Build grid: Vec<Vec<Option<(DynamicImage, w, h)>>>
  let grid: Vec<Vec<Option<(image::DynamicImage, u32, u32)>>> = if is_2d {
    outer
      .iter()
      .map(|row_expr| {
        if let Expr::List(row_items) = row_expr {
          row_items.iter().map(extract_image_for_assemble).collect()
        } else {
          vec![extract_image_for_assemble(row_expr)]
        }
      })
      .collect()
  } else {
    // Flat list → single row
    vec![outer.iter().map(extract_image_for_assemble).collect()]
  };

  let num_rows = grid.len();
  let num_cols = grid.iter().map(|row| row.len()).max().unwrap_or(0);
  if num_cols == 0 {
    return Err(InterpreterError::EvaluationError(
      "ImageAssemble: no images found".into(),
    ));
  }

  // Compute column widths and row heights
  // Each column gets the max width of images in that column
  // Each row gets the max height of images in that row
  let mut col_widths = vec![0u32; num_cols];
  let mut row_heights = vec![0u32; num_rows];

  for (r, row) in grid.iter().enumerate() {
    for (c, cell) in row.iter().enumerate() {
      if let Some((_, w, h)) = cell {
        if *w > col_widths[c] {
          col_widths[c] = *w;
        }
        if *h > row_heights[r] {
          row_heights[r] = *h;
        }
      }
    }
  }

  // When no fitting is specified, verify commensurate sizes:
  // all images in a row must have the same height,
  // all images in a column must have the same width.
  if fitting == "None" && args.len() < 2 {
    for (r, row) in grid.iter().enumerate() {
      let expected_h = row_heights[r];
      for cell in row.iter() {
        if let Some((_, _, h)) = cell
          && *h != expected_h
        {
          eprintln!(
            "\nImageAssemble::row: \
               Expecting images of the same height in one row."
          );
          return Ok(Expr::FunctionCall {
            name: "ImageAssemble".to_string(),
            args: args.to_vec(),
          });
        }
      }
    }
    for c in 0..num_cols {
      let expected_w = col_widths[c];
      for row in &grid {
        if let Some(Some((_, w, _))) = row.get(c)
          && *w != expected_w
        {
          eprintln!(
            "\nImageAssemble::col: \
               Expecting images of the same width in one column."
          );
          return Ok(Expr::FunctionCall {
            name: "ImageAssemble".to_string(),
            args: args.to_vec(),
          });
        }
      }
    }
  }

  // Total canvas size
  let canvas_w: u32 = col_widths.iter().sum();
  let canvas_h: u32 = row_heights.iter().sum();
  if canvas_w == 0 || canvas_h == 0 {
    return Err(InterpreterError::EvaluationError(
      "ImageAssemble: resulting image has zero dimensions".into(),
    ));
  }

  // Create canvas
  let mut canvas = image::RgbaImage::from_pixel(canvas_w, canvas_h, bg);

  // Place each image
  let mut y_offset = 0u32;
  for (r, row) in grid.iter().enumerate() {
    let mut x_offset = 0u32;
    for (c, cell) in row.iter().enumerate() {
      let cell_w = col_widths[c];
      let cell_h = row_heights[r];

      if cell_w > 0
        && cell_h > 0
        && let Some((dyn_img, orig_w, orig_h)) = cell
      {
        let rgba = dyn_img.to_rgba8();
        let placed = if *orig_w == cell_w && *orig_h == cell_h {
          rgba
        } else {
          fit_image_to_cell(
            &rgba, *orig_w, *orig_h, cell_w, cell_h, &fitting, bg,
          )
        };
        image::imageops::overlay(
          &mut canvas,
          &placed,
          x_offset as i64,
          y_offset as i64,
        );
      }
      // Missing cells stay as background

      x_offset += cell_w;
    }
    y_offset += row_heights[r];
  }

  // Convert the RGBA canvas to the appropriate channel count
  let dyn_img = match max_channels {
    1 => image::DynamicImage::ImageLuma8(
      image::DynamicImage::ImageRgba8(canvas).to_luma8(),
    ),
    3 => image::DynamicImage::ImageRgb8(
      image::DynamicImage::ImageRgba8(canvas).to_rgb8(),
    ),
    _ => image::DynamicImage::ImageRgba8(canvas),
  };
  Ok(dynamic_image_to_expr(&dyn_img))
}

// ─── I/O functions (Phase 4) ──────────────────────────────────────────────

/// Decode raw image bytes (PNG, JPEG, etc.) into an Expr::Image.
pub fn import_image_from_bytes(bytes: &[u8]) -> Result<Expr, InterpreterError> {
  let img = image::load_from_memory(bytes).map_err(|e| {
    InterpreterError::EvaluationError(format!(
      "Import: cannot decode image: {}",
      e
    ))
  })?;
  Ok(dynamic_image_to_expr(&img))
}

/// Import an image file and return an Expr::Image
#[cfg(not(target_arch = "wasm32"))]
pub fn import_image(path: &str) -> Result<Expr, InterpreterError> {
  let img = match image::open(path) {
    Ok(img) => img,
    Err(_) => {
      // Wolfram returns $Failed when the file cannot be opened
      return Ok(Expr::Identifier("$Failed".to_string()));
    }
  };
  Ok(dynamic_image_to_expr(&img))
}

/// Download a URL and import as image (CLI only — uses curl)
#[cfg(not(target_arch = "wasm32"))]
pub fn import_image_from_url(url: &str) -> Result<Expr, InterpreterError> {
  let output = std::process::Command::new("curl")
    .args(["-fsSL", "--max-time", "15", url])
    .output()
    .map_err(|e| {
      InterpreterError::EvaluationError(format!(
        "Import: failed to run curl: {}",
        e
      ))
    })?;
  if !output.status.success() {
    let stderr = String::from_utf8_lossy(&output.stderr);
    return Err(InterpreterError::EvaluationError(format!(
      "Import: failed to download \"{}\": {}",
      url,
      stderr.trim()
    )));
  }
  import_image_from_bytes(&output.stdout)
}

// ─── Rasterize ──────────────────────────────────────────────────────────────

/// Rasterize[expr] or Rasterize[expr, ImageResolution -> n]
/// Converts a Graphics, Grid, or other visual expression to a raster image.
#[cfg(not(target_arch = "wasm32"))]
pub fn rasterize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // If input is already an image, return it directly
  if let Expr::Image { .. } = &args[0] {
    return Ok(args[0].clone());
  }

  // Parse ImageResolution option (default 96 DPI to match usvg default)
  let mut dpi: f64 = 96.0;
  for opt in &args[1..] {
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
        _ => {}
      }
    }
  }

  // Get SVG string from the expression
  let svg_str = match &args[0] {
    Expr::Graphics { svg, .. } => svg.clone(),
    Expr::FunctionCall { name, args: fargs } if name == "Grid" => {
      crate::functions::graphics::grid_svg_with_gaps(fargs, &[])?
    }
    Expr::FunctionCall { name, args: fargs } if name == "Column" => {
      crate::functions::graphics::column_to_svg(fargs).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "Rasterize: failed to render Column".into(),
        )
      })?
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Rasterize: unsupported expression type".into(),
      ));
    }
  };

  rasterize_svg(&svg_str, dpi)
}

/// Rasterize an SVG string to an Expr::Image at the given DPI.
#[cfg(not(target_arch = "wasm32"))]
pub fn rasterize_svg(
  svg_str: &str,
  dpi: f64,
) -> Result<Expr, InterpreterError> {
  use std::sync::Arc as StdArc;

  // Set up font database with embedded monospace font
  let mut fontdb = resvg::usvg::fontdb::Database::new();
  fontdb.load_font_data(
    include_bytes!("../../resources/CourierPrime-Regular.ttf").to_vec(),
  );
  fontdb.load_font_data(
    include_bytes!("../../resources/CourierPrime-Bold.ttf").to_vec(),
  );
  // Set monospace family to our embedded font
  fontdb.set_monospace_family("Courier Prime");

  // Parse SVG
  let mut opt = resvg::usvg::Options::default();
  opt.fontdb = StdArc::new(fontdb);

  let tree = resvg::usvg::Tree::from_str(svg_str, &opt).map_err(|e| {
    InterpreterError::EvaluationError(format!(
      "Rasterize: SVG parse error: {}",
      e
    ))
  })?;

  // Compute output pixel dimensions scaled by DPI
  let svg_size = tree.size();
  let scale = dpi / 96.0; // usvg default DPI is 96
  let pix_w = (svg_size.width() as f64 * scale).ceil() as u32;
  let pix_h = (svg_size.height() as f64 * scale).ceil() as u32;

  if pix_w == 0 || pix_h == 0 {
    return Err(InterpreterError::EvaluationError(
      "Rasterize: resulting image has zero size".into(),
    ));
  }

  // Create pixmap and fill with white background
  let mut pixmap =
    resvg::tiny_skia::Pixmap::new(pix_w, pix_h).ok_or_else(|| {
      InterpreterError::EvaluationError(
        "Rasterize: failed to create pixel buffer".into(),
      )
    })?;
  pixmap.fill(resvg::tiny_skia::Color::WHITE);

  // Render SVG into pixmap
  let transform =
    resvg::tiny_skia::Transform::from_scale(scale as f32, scale as f32);
  resvg::render(&tree, transform, &mut pixmap.as_mut());

  // Convert RGBA pixel data to Expr::Image (normalized f64 values)
  let rgba_data = pixmap.data();
  let data: Vec<f64> = rgba_data.iter().map(|&v| v as f64 / 255.0).collect();

  Ok(Expr::Image {
    width: pix_w,
    height: pix_h,
    channels: 4,
    data: Arc::new(data),
    image_type: ImageType::Byte,
  })
}

/// Export an Expr::Image to a file
#[cfg(not(target_arch = "wasm32"))]
pub fn export_image(
  path: &str,
  width: u32,
  height: u32,
  channels: u8,
  data: &[f64],
) -> Result<(), InterpreterError> {
  let dyn_img = expr_to_dynamic_image(width, height, channels, data);
  dyn_img.save(path).map_err(|e| {
    InterpreterError::EvaluationError(format!(
      "Export: cannot save \"{}\": {}",
      path, e
    ))
  })
}
