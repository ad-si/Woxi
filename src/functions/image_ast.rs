use crate::InterpreterError;
use crate::syntax::{Expr, ImageType};
use std::sync::Arc;

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Convert an Expr::Image to an `image::DynamicImage`.
fn expr_to_dynamic_image(
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
      image_type: requested_type.unwrap_or(ImageType::Real64),
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
      image_type: requested_type.unwrap_or(ImageType::Real64),
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
    _ => Err(InterpreterError::EvaluationError(
      "ImageDimensions: argument is not an Image".into(),
    )),
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
        ImageType::Byte => "Byte",
        ImageType::Bit16 => "Bit16",
        ImageType::Real32 => "Real32",
        ImageType::Real64 => "Real64",
      };
      Ok(Expr::Identifier(type_str.to_string()))
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
      ..
    } => {
      let w = *width as usize;
      let h = *height as usize;
      let ch = *channels as usize;
      let mut rows = Vec::with_capacity(h);

      for y in 0..h {
        if ch == 1 {
          // Grayscale: {{v, v, ...}, ...}
          let mut row = Vec::with_capacity(w);
          for x in 0..w {
            let idx = y * w + x;
            row.push(Expr::Real(data[idx]));
          }
          rows.push(Expr::List(row));
        } else {
          // Color: {{{r, g, b}, ...}, ...}
          let mut row = Vec::with_capacity(w);
          for x in 0..w {
            let base = (y * w + x) * ch;
            let pixel: Vec<Expr> =
              (0..ch).map(|c| Expr::Real(data[base + c])).collect();
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
    Expr::Image { channels, .. } => {
      let space = match channels {
        1 => "Grayscale",
        _ => "RGB",
      };
      Ok(Expr::Identifier(space.to_string()))
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
      image_type,
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
        image_type: *image_type,
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
      let dyn_img = expr_to_dynamic_image(*width, *height, *channels, data);

      let cropped = if args.len() == 2 {
        // Manual crop: {{x1,y1},{x2,y2}}
        let (x1, y1, x2, y2) = match &args[1] {
          Expr::List(coords) if coords.len() == 2 => {
            let p1 = match &coords[0] {
              Expr::List(p) if p.len() == 2 => {
                (expr_to_f64(&p[0])? as u32, expr_to_f64(&p[1])? as u32)
              }
              _ => {
                return Err(InterpreterError::EvaluationError(
                  "ImageCrop: coordinates must be {{x1,y1},{x2,y2}}".into(),
                ));
              }
            };
            let p2 = match &coords[1] {
              Expr::List(p) if p.len() == 2 => {
                (expr_to_f64(&p[0])? as u32, expr_to_f64(&p[1])? as u32)
              }
              _ => {
                return Err(InterpreterError::EvaluationError(
                  "ImageCrop: coordinates must be {{x1,y1},{x2,y2}}".into(),
                ));
              }
            };
            (p1.0, p1.1, p2.0, p2.1)
          }
          _ => {
            return Err(InterpreterError::EvaluationError(
              "ImageCrop: second argument must be {{x1,y1},{x2,y2}}".into(),
            ));
          }
        };
        let crop_w = x2.saturating_sub(x1);
        let crop_h = y2.saturating_sub(y1);
        dyn_img.crop_imm(x1, y1, crop_w, crop_h)
      } else {
        // Auto-crop: trim uniform border
        // Simple implementation: find bounding box of non-uniform pixels
        auto_crop(&dyn_img)
      };

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

/// EdgeDetect[img] - Sobel edge detection
pub fn edge_detect_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "EdgeDetect expects exactly 1 argument".into(),
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

      // Sobel kernels
      let gx_kernel: [[f64; 3]; 3] =
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
      let gy_kernel: [[f64; 3]; 3] =
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

      let mut result = vec![0.0_f64; w * h];
      let mut max_mag = 0.0_f64;

      for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
          let mut gx = 0.0;
          let mut gy = 0.0;
          for ky in 0..3 {
            for kx in 0..3 {
              let pixel = gray[(y + ky - 1) * w + (x + kx - 1)];
              gx += pixel * gx_kernel[ky][kx];
              gy += pixel * gy_kernel[ky][kx];
            }
          }
          let mag = (gx * gx + gy * gy).sqrt();
          result[y * w + x] = mag;
          if mag > max_mag {
            max_mag = mag;
          }
        }
      }

      // Normalize to [0, 1]
      if max_mag > 0.0 {
        for v in result.iter_mut() {
          *v /= max_mag;
        }
      }

      Ok(Expr::Image {
        width: *width,
        height: *height,
        channels: 1,
        data: Arc::new(result),
        image_type: ImageType::Real64,
      })
    }
    _ => Err(InterpreterError::EvaluationError(
      "EdgeDetect: argument is not an Image".into(),
    )),
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
        .map(|(&a, &b)| op(a, b).clamp(0.0, 1.0))
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
  let img = image::open(path).map_err(|e| {
    InterpreterError::EvaluationError(format!(
      "Import: cannot open \"{}\": {}",
      path, e
    ))
  })?;
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
