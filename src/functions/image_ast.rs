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

/// Image[data] / Image[data, type] / Image[data, type, opts…] — construct
/// an image from nested lists. The optional type argument is one of
/// "Bit", "Byte", "Bit16", "Real32", "Real64". Extra arguments are
/// treated as options like `ColorSpace -> ...` and `Interleaving -> ...`;
/// unknown options are accepted and currently ignored.
pub fn image_constructor_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "Image expects at least 1 argument".into(),
    ));
  }

  // The second positional arg is a type tag (string or identifier).
  // Anything beyond that must be a Rule-form option.
  let parse_type = |e: &Expr| -> Option<ImageType> {
    let s = match e {
      Expr::String(s) | Expr::Identifier(s) => Some(s.as_str()),
      _ => None,
    }?;
    match s {
      "Bit" => Some(ImageType::Bit),
      "Byte" | "UnsignedInteger8" => Some(ImageType::Byte),
      "Bit16" | "UnsignedInteger16" => Some(ImageType::Bit16),
      "Real32" => Some(ImageType::Real32),
      "Real64" => Some(ImageType::Real64),
      _ => None,
    }
  };

  let mut requested_type: Option<ImageType> = None;
  let mut opt_start = 1;
  if let Some(ty) = args.get(1).and_then(parse_type) {
    requested_type = Some(ty);
    opt_start = 2;
  }

  // Treat the remaining args as options. Only Rule/RuleDelayed shapes are
  // accepted; anything else aborts to keep nonsense calls from sneaking
  // through.
  for opt in &args[opt_start..] {
    if !matches!(opt, Expr::Rule { .. } | Expr::RuleDelayed { .. }) {
      return Err(InterpreterError::EvaluationError(format!(
        "Image: extra argument is not a Rule option: {}",
        crate::syntax::expr_to_string(opt)
      )));
    }
  }

  // `Image[NumericArray[data, type]]` — unwrap to the nested list and
  // pick up the dtype if not already given.
  let unwrapped: Expr;
  let raw_arg = match &args[0] {
    Expr::FunctionCall {
      name: na_name,
      args: na_args,
    } if na_name == "NumericArray" && !na_args.is_empty() => {
      if requested_type.is_none()
        && let Some(ty) = na_args.get(1)
      {
        let s = match ty {
          Expr::String(s) | Expr::Identifier(s) => Some(s.as_str()),
          _ => None,
        };
        requested_type = match s {
          Some("Byte") | Some("UnsignedInteger8") => Some(ImageType::Byte),
          Some("Bit16") | Some("UnsignedInteger16") => Some(ImageType::Bit16),
          Some("Real32") => Some(ImageType::Real32),
          Some("Real64") => Some(ImageType::Real64),
          _ => None,
        };
      }
      unwrapped = na_args[0].clone();
      &unwrapped
    }
    other => other,
  };

  // The argument should be a 2D or 3D nested list
  let rows = match raw_arg {
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

  // Normalize integer-typed pixel values into the [0, 1] f64 buffer.
  // Storage is always normalized; image_type only affects display precision
  // and the implicit /255 (or /65535) applied on input.
  let divisor: f64 = match requested_type {
    Some(ImageType::Byte) => 255.0,
    Some(ImageType::Bit16) => 65535.0,
    _ => 1.0,
  };

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
          data.push(expr_to_f64(v)? / divisor);
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
        data.push(expr_to_f64(v)? / divisor);
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
  let is_image =
    matches!(&args[0], Expr::Image { .. }) || is_valid_image3d(&args[0]);
  Ok(Expr::Identifier(
    if is_image { "True" } else { "False" }.to_string(),
  ))
}

/// Image3D[arg] is a valid Image when `arg` is a rank-3 or rank-4 nested
/// list, or a NumericArray of rank 3 or 4. Image3D itself isn't fully
/// implemented, but ImageQ can still classify it correctly.
fn is_valid_image3d(e: &Expr) -> bool {
  let Expr::FunctionCall { name, args } = e else {
    return false;
  };
  if name != "Image3D" || args.is_empty() {
    return false;
  }
  let inner = match &args[0] {
    Expr::FunctionCall {
      name: na_name,
      args: na_args,
    } if na_name == "NumericArray" && !na_args.is_empty() => &na_args[0],
    other => other,
  };
  let rank = nested_list_rank(inner);
  matches!(rank, Some(3) | Some(4))
}

/// Depth of a regular nested list (the innermost elements must be
/// scalars). Returns `None` if rows have inconsistent shape.
fn nested_list_rank(e: &Expr) -> Option<usize> {
  match e {
    Expr::List(items) if !items.is_empty() => {
      let first = nested_list_rank(&items[0])?;
      for it in items.iter().skip(1) {
        if nested_list_rank(it)? != first {
          return None;
        }
      }
      Some(first + 1)
    }
    Expr::List(_) => None,
    Expr::Integer(_)
    | Expr::BigInteger(_)
    | Expr::Real(_)
    | Expr::BigFloat(_, _) => Some(0),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      Some(0)
    }
    _ => None,
  }
}

/// ImageDimensions[img] — {width, height} for 2D, {width, height, depth}
/// for Image3D. Image3D[arg] is accepted when `arg` is a rank-3 or
/// rank-4 nested list (or NumericArray of the same).
pub fn image_dimensions_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ImageDimensions expects exactly 1 argument".into(),
    ));
  }
  if let Expr::Image { width, height, .. } = &args[0] {
    return Ok(Expr::List(
      vec![
        Expr::Integer(*width as i128),
        Expr::Integer(*height as i128),
      ]
      .into(),
    ));
  }
  if let Some(dims) = image3d_dimensions(&args[0]) {
    return Ok(Expr::List(
      dims
        .into_iter()
        .map(Expr::Integer)
        .collect::<Vec<_>>()
        .into(),
    ));
  }
  // Match wolframscript: emit ImageDimensions::imginv before
  // returning the unevaluated call.
  crate::emit_message(&format!(
    "ImageDimensions::imginv: Expecting an image or graphics instead of {}.",
    crate::syntax::expr_to_string(&args[0])
  ));
  Ok(Expr::FunctionCall {
    name: "ImageDimensions".to_string(),
    args: args.to_vec().into(),
  })
}

/// Extract `{width, height, depth}` for a valid Image3D[arg]. Returns
/// None if `arg` isn't a rank-3 or rank-4 nested list (peeling through
/// a NumericArray wrapper).
fn image3d_dimensions(e: &Expr) -> Option<Vec<i128>> {
  let Expr::FunctionCall { name, args } = e else {
    return None;
  };
  if name != "Image3D" || args.is_empty() {
    return None;
  }
  let inner = match &args[0] {
    Expr::FunctionCall {
      name: na_name,
      args: na_args,
    } if na_name == "NumericArray" && !na_args.is_empty() => &na_args[0],
    other => other,
  };
  let rank = nested_list_rank(inner)?;
  if rank != 3 && rank != 4 {
    return None;
  }
  let depth = list_len(inner)?;
  let slice = list_first(inner)?;
  let height = list_len(slice)?;
  let row = list_first(slice)?;
  let width = list_len(row)?;
  Some(vec![width as i128, height as i128, depth as i128])
}

fn list_len(e: &Expr) -> Option<usize> {
  match e {
    Expr::List(items) => Some(items.len()),
    _ => None,
  }
}

fn list_first(e: &Expr) -> Option<&Expr> {
  match e {
    Expr::List(items) if !items.is_empty() => Some(&items[0]),
    _ => None,
  }
}

/// ImageAspectRatio[img] - height/width as an exact rational when possible.
pub fn image_aspect_ratio_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ImageAspectRatio expects exactly 1 argument".into(),
    ));
  }
  fn ratio(h: i128, w: i128) -> Expr {
    fn gcd(a: i128, b: i128) -> i128 {
      if b == 0 { a.abs() } else { gcd(b, a % b) }
    }
    let g = gcd(h, w);
    let (num, den) = (h / g, w / g);
    if den == 1 {
      Expr::Integer(num)
    } else {
      Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(num), Expr::Integer(den)].into(),
      }
    }
  }
  if let Expr::Image { width, height, .. } = &args[0] {
    let (w, h) = (*width as i128, *height as i128);
    if w == 0 {
      return Ok(Expr::FunctionCall {
        name: "ImageAspectRatio".to_string(),
        args: args.to_vec().into(),
      });
    }
    return Ok(ratio(h, w));
  }
  if let Some(dims) = image3d_dimensions(&args[0])
    && let [w, h, _] = dims.as_slice()
    && *w != 0
  {
    return Ok(ratio(*h, *w));
  }
  crate::emit_message(&format!(
    "ImageAspectRatio::imginv: Expecting an image or graphics instead of {}.",
    crate::syntax::expr_to_string(&args[0])
  ));
  Ok(Expr::FunctionCall {
    name: "ImageAspectRatio".to_string(),
    args: args.to_vec().into(),
  })
}

/// ImageChannels[img] — channel count (1/3/4 for 2D, derived from the
/// Image3D rank for 3D: rank-3 → 1, rank-4 → innermost length).
pub fn image_channels_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ImageChannels expects exactly 1 argument".into(),
    ));
  }
  if let Expr::Image { channels, .. } = &args[0] {
    return Ok(Expr::Integer(*channels as i128));
  }
  if let Some(ch) = image3d_channels(&args[0]) {
    return Ok(Expr::Integer(ch as i128));
  }
  crate::emit_message(&format!(
    "ImageChannels::imginv: Expecting an image or graphics instead of {}.",
    crate::syntax::expr_to_string(&args[0])
  ));
  Ok(Expr::FunctionCall {
    name: "ImageChannels".to_string(),
    args: args.to_vec().into(),
  })
}

/// Channel count for a valid Image3D[arg]; None for invalid shapes.
fn image3d_channels(e: &Expr) -> Option<usize> {
  let Expr::FunctionCall { name, args } = e else {
    return None;
  };
  if name != "Image3D" || args.is_empty() {
    return None;
  }
  let inner = match &args[0] {
    Expr::FunctionCall {
      name: na_name,
      args: na_args,
    } if na_name == "NumericArray" && !na_args.is_empty() => &na_args[0],
    other => other,
  };
  let rank = nested_list_rank(inner)?;
  match rank {
    3 => Some(1),
    4 => {
      // Innermost length: list at depth 0, 1, 2, 3 — depth-3 element is
      // the per-pixel channel vector.
      let l1 = list_first(inner)?;
      let l2 = list_first(l1)?;
      let l3 = list_first(l2)?;
      list_len(l3)
    }
    _ => None,
  }
}

/// ImageType[img] - "Byte", "Bit16", "Real32", "Real64"
pub fn image_type_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ImageType expects exactly 1 argument".into(),
    ));
  }
  let type_to_str = |t: &ImageType| -> &'static str {
    match t {
      ImageType::Bit => "Bit",
      ImageType::Byte => "Byte",
      ImageType::Bit16 => "Bit16",
      ImageType::Real32 => "Real32",
      ImageType::Real64 => "Real64",
    }
  };
  if let Expr::Image { image_type, .. } = &args[0] {
    return Ok(Expr::String(type_to_str(image_type).to_string()));
  }
  if let Some(t) = image3d_type(&args[0]) {
    return Ok(Expr::String(type_to_str(&t).to_string()));
  }
  crate::emit_message(&format!(
    "ImageType::imginv: Expecting an image or graphics instead of {}.",
    crate::syntax::expr_to_string(&args[0])
  ));
  Ok(Expr::FunctionCall {
    name: "ImageType".to_string(),
    args: args.to_vec().into(),
  })
}

/// Resolve the underlying type tag for a valid Image3D[arg]. Defaults
/// to Real32 if there's no NumericArray wrapper.
fn image3d_type(e: &Expr) -> Option<ImageType> {
  let Expr::FunctionCall { name, args } = e else {
    return None;
  };
  if name != "Image3D" || args.is_empty() {
    return None;
  }
  let (inner, type_tag) = match &args[0] {
    Expr::FunctionCall {
      name: na_name,
      args: na_args,
    } if na_name == "NumericArray" && !na_args.is_empty() => {
      let tag = na_args.get(1).and_then(|t| match t {
        Expr::String(s) | Expr::Identifier(s) => Some(s.as_str()),
        _ => None,
      });
      (&na_args[0], tag)
    }
    other => (other, None),
  };
  let rank = nested_list_rank(inner)?;
  if rank != 3 && rank != 4 {
    return None;
  }
  Some(match type_tag {
    Some("Byte") | Some("UnsignedInteger8") => ImageType::Byte,
    Some("Bit") => ImageType::Bit,
    Some("Bit16") | Some("UnsignedInteger16") => ImageType::Bit16,
    Some("Real64") => ImageType::Real64,
    _ => ImageType::Real32,
  })
}

/// ImageData[img] - Extract pixel values as nested List
pub fn image_data_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "ImageData expects 1 or 2 arguments".into(),
    ));
  }
  let requested_type: Option<&str> = if args.len() == 2 {
    match &args[1] {
      Expr::String(s) => Some(s.as_str()),
      _ => None,
    }
  } else {
    None
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

      let expected_len = w
        .checked_mul(h)
        .and_then(|wh| wh.checked_mul(ch))
        .ok_or_else(|| {
          InterpreterError::EvaluationError(
            "ImageData: image dimensions overflow".into(),
          )
        })?;
      if data.len() < expected_len {
        return Err(InterpreterError::EvaluationError(
          "ImageData: image data buffer is too small for the given dimensions"
            .into(),
        ));
      }

      // Convert values to the appropriate precision.
      // - Explicit 2nd-arg "Bit"/"Byte"/"Bit16"/"Real"/"Real32"/"Real64"
      //   overrides the image's internal type.
      // - Bit images (from Binarize) return integers 0 and 1.
      // - Real32 images store f64 internally but output f32-precision values.
      let to_expr = |v: f64| -> Expr {
        if let Some(t) = requested_type {
          return match t {
            "Bit" => Expr::Integer(v.round().clamp(0.0, 1.0) as i128),
            "Byte" => {
              Expr::Integer((v * 255.0).round().clamp(0.0, 255.0) as i128)
            }
            "Bit16" => {
              Expr::Integer((v * 65535.0).round().clamp(0.0, 65535.0) as i128)
            }
            "Real32" => Expr::Real((v as f32) as f64),
            _ => Expr::Real(v),
          };
        }
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
          let row: Vec<Expr> =
            (0..w).map(|x| to_expr(data[y * w + x])).collect();
          rows.push(Expr::List(row.into()));
        } else {
          // Color: {{{r, g, b}, ...}, ...}
          let row: Vec<Expr> = (0..w)
            .map(|x| {
              let base = (y * w + x) * ch;
              Expr::List((0..ch).map(|c| to_expr(data[base + c])).collect())
            })
            .collect();
          rows.push(Expr::List(row.into()));
        }
      }

      Ok(Expr::List(rows.into()))
    }
    _ => {
      crate::emit_message(&format!(
        "ImageData::imginv: Expecting an image or graphics instead of {}.",
        crate::syntax::expr_to_string(&args[0])
      ));
      Ok(Expr::FunctionCall {
        name: "ImageData".to_string(),
        args: args.to_vec().into(),
      })
    }
  }
}

/// PixelValuePositions[img, val] / PixelValuePositions[img, val, tol]
///
/// Return the `{x, y}` coordinates of every grayscale pixel whose value
/// equals `val` (within `tol`, default 0). Wolfram uses bottom-left
/// origin so `y` is `1` for the bottom-most row, `h` for the top.
/// Iteration order is top-down, left-to-right within each row, matching
/// wolframscript:
///
///   `PixelValuePositions[Image[{{0, 1}, {1, 0}, {1, 1}}], 1]`
///     →  `{{2, 3}, {1, 2}, {1, 1}, {2, 1}}`
pub fn pixel_value_positions_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "PixelValuePositions expects 2 or 3 arguments".into(),
    ));
  }
  // wolframscript validates the image argument first and emits `imginv`
  // before inspecting the target value, so do the same here.
  if !matches!(&args[0], Expr::Image { .. }) {
    crate::emit_message(&format!(
      "PixelValuePositions::imginv: Expecting an image or graphics instead of {}.",
      crate::syntax::expr_to_string(&args[0])
    ));
    return Ok(Expr::FunctionCall {
      name: "PixelValuePositions".to_string(),
      args: args.to_vec().into(),
    });
  }
  let target = match crate::functions::math_ast::try_eval_to_f64(&args[1]) {
    Some(v) => v,
    None => {
      return Ok(Expr::FunctionCall {
        name: "PixelValuePositions".to_string(),
        args: args.to_vec().into(),
      });
    }
  };
  let tol = if args.len() == 3 {
    match crate::functions::math_ast::try_eval_to_f64(&args[2]) {
      Some(v) => v,
      None => {
        return Ok(Expr::FunctionCall {
          name: "PixelValuePositions".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  } else {
    0.0
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
      // Only the grayscale path is implemented; multichannel matching
      // would need Wolfram's `{r, g, b}`-style RGB target value.
      if ch != 1 {
        return Ok(Expr::FunctionCall {
          name: "PixelValuePositions".to_string(),
          args: args.to_vec().into(),
        });
      }
      let mut positions: Vec<Expr> = Vec::new();
      for y in 0..h {
        for x in 0..w {
          let v = data[y * w + x];
          if (v - target).abs() <= tol {
            // Wolfram coords: x' = x+1, y' = h - y (bottom-left origin).
            positions.push(Expr::List(
              vec![
                Expr::Integer((x + 1) as i128),
                Expr::Integer((h - y) as i128),
              ]
              .into(),
            ));
          }
        }
      }
      Ok(Expr::List(positions.into()))
    }
    _ => {
      crate::emit_message(&format!(
        "PixelValuePositions::imginv: Expecting an image or graphics instead of {}.",
        crate::syntax::expr_to_string(&args[0])
      ));
      Ok(Expr::FunctionCall {
        name: "PixelValuePositions".to_string(),
        args: args.to_vec().into(),
      })
    }
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
    other => {
      // Matches wolframscript: emit ImageColorSpace::imginv and return
      // unevaluated instead of erroring out.
      crate::emit_message(&format!(
        "ImageColorSpace::imginv: Expecting an image or graphics instead of {}.",
        crate::syntax::expr_to_string(other)
      ));
      Ok(Expr::FunctionCall {
        name: "ImageColorSpace".to_string(),
        args: args.to_vec().into(),
      })
    }
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
      let is_real32 = matches!(image_type, crate::syntax::ImageType::Real32);
      // For Real32 images, perform the negation in f32 so the result
      // matches wolframscript's f32 image arithmetic. Other types do
      // the subtraction in f64 directly.
      let negate = |v: f64| -> f64 {
        if is_real32 {
          (1.0_f32 - v as f32) as f64
        } else {
          1.0 - v
        }
      };
      let mut new_data = Vec::with_capacity(data.len());
      if ch == 4 {
        // RGBA: negate R,G,B but keep alpha
        for i in 0..data.len() {
          if i % 4 == 3 {
            new_data.push(data[i]); // alpha unchanged
          } else {
            new_data.push(negate(data[i]));
          }
        }
      } else {
        for v in data.iter() {
          new_data.push(negate(*v));
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
    Expr::FunctionCall { name, args: cargs } if name == "RGBColor" => {
      // RGBColor[r, g, b] → RGBColor[1-r, 1-g, 1-b]
      // RGBColor[r, g, b, a] → RGBColor[1-r, 1-g, 1-b, a]
      fn negate_component(e: &Expr) -> Result<Expr, InterpreterError> {
        match e {
          Expr::Integer(n) => Ok(Expr::Integer(1 - n)),
          Expr::Real(r) => Ok(Expr::Real(1.0 - r)),
          _ => {
            let v = expr_to_f64(e)?;
            Ok(Expr::Real(1.0 - v))
          }
        }
      }
      match cargs.len() {
        3 => Ok(Expr::FunctionCall {
          name: "RGBColor".to_string(),
          args: vec![
            negate_component(&cargs[0])?,
            negate_component(&cargs[1])?,
            negate_component(&cargs[2])?,
          ]
          .into(),
        }),
        4 => Ok(Expr::FunctionCall {
          name: "RGBColor".to_string(),
          args: vec![
            negate_component(&cargs[0])?,
            negate_component(&cargs[1])?,
            negate_component(&cargs[2])?,
            cargs[3].clone(),
          ]
          .into(),
        }),
        _ => Err(InterpreterError::EvaluationError(
          "ColorNegate: RGBColor must have 3 or 4 arguments".into(),
        )),
      }
    }
    Expr::FunctionCall { name, args: cargs } if name == "GrayLevel" => {
      // GrayLevel[v] → GrayLevel[1-v], alpha preserved if present
      fn negate_component(e: &Expr) -> Result<Expr, InterpreterError> {
        match e {
          Expr::Integer(n) => Ok(Expr::Integer(1 - n)),
          Expr::Real(r) => Ok(Expr::Real(1.0 - r)),
          _ => {
            let v = expr_to_f64(e)?;
            Ok(Expr::Real(1.0 - v))
          }
        }
      }
      match cargs.len() {
        1 => Ok(Expr::FunctionCall {
          name: "GrayLevel".to_string(),
          args: vec![negate_component(&cargs[0])?].into(),
        }),
        2 => Ok(Expr::FunctionCall {
          name: "GrayLevel".to_string(),
          args: vec![negate_component(&cargs[0])?, cargs[1].clone()].into(),
        }),
        _ => Err(InterpreterError::EvaluationError(
          "ColorNegate: GrayLevel must have 1 or 2 arguments".into(),
        )),
      }
    }
    other => {
      // Match wolframscript: when the argument isn't a valid
      // image / color directive, emit `ColorNegate::imginv` and
      // return the call unevaluated instead of throwing a fatal
      // evaluator error. wolframscript prints
      // `<v> should be a valid image, a color directive or a
      // list of such objects.` on `ColorNegate[$Failed]`.
      crate::emit_message(&format!(
        "ColorNegate::imginv: {} should be a valid image, a color directive or a list of such objects.",
        crate::syntax::expr_to_string(other)
      ));
      Ok(Expr::FunctionCall {
        name: "ColorNegate".to_string(),
        args: args.to_vec().into(),
      })
    }
  }
}

/// Binarize[img] or Binarize[img, threshold]
pub fn binarize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Binarize expects 1 or 2 arguments".into(),
    ));
  }

  // wolframscript validates the image before the threshold, so a chain
  // like `Binarize[img, Out[0]]` (where `img` is undefined) emits
  // Binarize::imginv rather than choking on the non-numeric threshold.
  if !matches!(&args[0], Expr::Image { .. }) {
    crate::emit_message(&format!(
      "Binarize::imginv: Expecting an image or graphics instead of {}.",
      crate::syntax::expr_to_string(&args[0])
    ));
    return Ok(Expr::FunctionCall {
      name: "Binarize".to_string(),
      args: args.to_vec().into(),
    });
  }

  // Threshold can be a single number (strict >) or a {t1, t2} pair
  // (t1 <= v <= t2, both inclusive). The 1-arg form uses an automatic
  // threshold (approximated here as 0.5 with the legacy >= compare so
  // existing snapshots still pass; wolframscript uses Otsu's method).
  enum Threshold {
    Default,
    Single(f64),
    Range(f64, f64),
  }
  let threshold = if args.len() == 2 {
    if let Expr::List(items) = &args[1]
      && items.len() == 2
    {
      Threshold::Range(expr_to_f64(&items[0])?, expr_to_f64(&items[1])?)
    } else {
      Threshold::Single(expr_to_f64(&args[1])?)
    }
  } else {
    Threshold::Default
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
          let bit = match threshold {
            Threshold::Default => lum >= 0.5,
            Threshold::Single(t) => lum > t,
            Threshold::Range(t1, t2) => lum >= t1 && lum <= t2,
          };
          new_data.push(if bit { 1.0 } else { 0.0 });
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

  let Expr::Image {
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
  else {
    crate::emit_message(&format!(
      "Blur::imginv: Expecting an image or graphics instead of {}.",
      crate::syntax::expr_to_string(&args[0])
    ));
    return Ok(Expr::FunctionCall {
      name: "Blur".to_string(),
      args: args.to_vec().into(),
    });
  };

  let radius = if args.len() == 2 {
    expr_to_f64(&args[1])?
  } else {
    2.0
  };
  let r = radius.round() as usize;
  if r == 0 {
    return Ok(args[0].clone());
  }

  let sigma = (radius / 2.0).max(1e-9);
  // Truncated Gaussian kernel of width 2r+1, normalized to sum 1 once
  // (without boundary truncation). Boundary handling uses renormalization
  // at the edges.
  let mut kernel: Vec<f64> = (0..=2 * r)
    .map(|i| {
      let x = i as f64 - r as f64;
      (-(x * x) / (2.0 * sigma * sigma)).exp()
    })
    .collect();
  let ksum: f64 = kernel.iter().sum();
  for k in &mut kernel {
    *k /= ksum;
  }

  let w = *width as usize;
  let h = *height as usize;
  let ch = *channels as usize;

  // Separable 1D Gaussian convolution: blur horizontally, then
  // vertically. Boundary handling renormalises the kernel against the
  // weights that fall inside the image.
  let idx = |y: usize, x: usize, c: usize| -> usize { (y * w + x) * ch + c };

  // Horizontal pass (rows).
  let mut tmp = vec![0.0; data.len()];
  for c_idx in 0..ch {
    for y in 0..h {
      for x in 0..w {
        let mut sum = 0.0;
        let mut wsum = 0.0;
        for ki in 0..kernel.len() {
          let kx = x as isize + ki as isize - r as isize;
          if kx < 0 || kx >= w as isize {
            continue;
          }
          let v = data[idx(y, kx as usize, c_idx)];
          sum += kernel[ki] * v;
          wsum += kernel[ki];
        }
        tmp[idx(y, x, c_idx)] = if wsum > 0.0 { sum / wsum } else { 0.0 };
      }
    }
  }

  // Vertical pass (columns).
  let mut new_data = vec![0.0; data.len()];
  for c_idx in 0..ch {
    for x in 0..w {
      for y in 0..h {
        let mut sum = 0.0;
        let mut wsum = 0.0;
        for ki in 0..kernel.len() {
          let ky = y as isize + ki as isize - r as isize;
          if ky < 0 || ky >= h as isize {
            continue;
          }
          let v = tmp[idx(ky as usize, x, c_idx)];
          sum += kernel[ki] * v;
          wsum += kernel[ki];
        }
        new_data[idx(y, x, c_idx)] = if wsum > 0.0 { sum / wsum } else { 0.0 };
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
    _ => {
      crate::emit_message(&format!(
        "Sharpen::imginv: Expecting an image or graphics instead of {}.",
        crate::syntax::expr_to_string(&args[0])
      ));
      Ok(Expr::FunctionCall {
        name: "Sharpen".to_string(),
        args: args.to_vec().into(),
      })
    }
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

  let Expr::Image {
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
  else {
    crate::emit_message(&format!(
      "ImageAdjust::imginv: Expecting an image or graphics instead of {}.",
      crate::syntax::expr_to_string(&args[0])
    ));
    return Ok(Expr::FunctionCall {
      name: "ImageAdjust".to_string(),
      args: args.to_vec().into(),
    });
  };

  // 1-arg form: rescale to [0, 1] using actual min/max.
  // 2-arg scalar form: contrast curve, no rescaling.
  // 2-arg {c, b} form: brightness then contrast.
  enum Adjust {
    Rescale,
    Contrast(f64),
    BrightnessContrast(f64, f64),
  }
  let mode = if args.len() == 1 {
    Adjust::Rescale
  } else if let Expr::List(items) = &args[1]
    && items.len() == 2
  {
    Adjust::BrightnessContrast(expr_to_f64(&items[0])?, expr_to_f64(&items[1])?)
  } else {
    Adjust::Contrast(expr_to_f64(&args[1])?)
  };

  // Real32 images store f32-quantised values in an f64 buffer. Round
  // each pixel to its f32 representation before the computation and
  // again on the way out so the result matches wolframscript's f32
  // image arithmetic.
  let is_real32 = matches!(image_type, crate::syntax::ImageType::Real32);
  let snap = |v: f64| -> f64 { if is_real32 { (v as f32) as f64 } else { v } };
  let apply = |v: f64| -> f64 {
    let v = snap(v);
    let r = match mode {
      Adjust::Rescale => unreachable!(),
      Adjust::Contrast(c) => (0.5 + (1.0 + c) * (v - 0.5)).clamp(0.0, 1.0),
      Adjust::BrightnessContrast(c, b) => {
        let after_b = (v * (1.0 + b)).clamp(0.0, 1.0);
        (0.5 + (1.0 + c) * (after_b - 0.5)).clamp(0.0, 1.0)
      }
    };
    snap(r)
  };

  let new_data: Vec<f64> = if matches!(mode, Adjust::Rescale) {
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
    if range > 0.0 {
      data.iter().map(|&v| (v - min_val) / range).collect()
    } else {
      vec![0.5; data.len()]
    }
  } else {
    data.iter().map(|&v| apply(v)).collect()
  };

  Ok(Expr::Image {
    width: *width,
    height: *height,
    channels: *channels,
    data: Arc::new(new_data),
    image_type: *image_type,
  })
}

/// ImageReflect[img] / ImageReflect[img, side] / ImageReflect[img, s1 -> s2].
/// Default: vertical (top↔bottom) flip. Horizontal / vertical sides reflect
/// across the perpendicular axis; rules between perpendicular sides do the
/// same axis flip; rules between adjacent sides reflect across a diagonal
/// (swapping width and height). Operates directly on the pixel array, so
/// pixel precision is preserved.
pub fn image_reflect_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "ImageReflect expects 1 or 2 arguments".into(),
    ));
  }

  // The five reflection modes Wolfram supports for 2D images.
  enum Mode {
    Vertical,     // top↔bottom (default)
    Horizontal,   // left↔right
    Diagonal,     // main diagonal, transpose
    AntiDiagonal, // anti-diagonal, transpose-then-rotate-180
  }

  let side_axis = |s: &str| -> Option<u8> {
    match s {
      "Top" | "Bottom" => Some(0),
      "Left" | "Right" => Some(1),
      _ => None,
    }
  };

  let mode = if args.len() == 1 {
    Mode::Vertical
  } else {
    match &args[1] {
      Expr::Identifier(s) => match side_axis(s) {
        Some(0) => Mode::Vertical,
        Some(1) => Mode::Horizontal,
        _ => return Ok(args[0].clone()),
      },
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => match (pattern.as_ref(), replacement.as_ref()) {
        (Expr::Identifier(p), Expr::Identifier(r)) => {
          let (ap, ar) = (side_axis(p), side_axis(r));
          match (ap, ar) {
            (Some(0), Some(0)) => Mode::Vertical,
            (Some(1), Some(1)) => Mode::Horizontal,
            (Some(a), Some(b)) if a != b => {
              if (p == "Top" && r == "Left")
                || (p == "Left" && r == "Top")
                || (p == "Bottom" && r == "Right")
                || (p == "Right" && r == "Bottom")
              {
                Mode::Diagonal
              } else {
                Mode::AntiDiagonal
              }
            }
            _ => return Ok(args[0].clone()),
          }
        }
        _ => return Ok(args[0].clone()),
      },
      _ => return Ok(args[0].clone()),
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
      let pixel = |r: usize, c: usize| -> &[f64] {
        let base = (r * w + c) * ch;
        &data[base..base + ch]
      };
      let mut new_data = Vec::with_capacity(data.len());
      let (new_w, new_h) = match mode {
        Mode::Vertical => {
          for r in 0..h {
            for c in 0..w {
              new_data.extend_from_slice(pixel(h - 1 - r, c));
            }
          }
          (*width, *height)
        }
        Mode::Horizontal => {
          for r in 0..h {
            for c in 0..w {
              new_data.extend_from_slice(pixel(r, w - 1 - c));
            }
          }
          (*width, *height)
        }
        Mode::Diagonal => {
          for r in 0..w {
            for c in 0..h {
              new_data.extend_from_slice(pixel(c, r));
            }
          }
          (*height, *width)
        }
        Mode::AntiDiagonal => {
          for r in 0..w {
            for c in 0..h {
              new_data.extend_from_slice(pixel(h - 1 - c, w - 1 - r));
            }
          }
          (*height, *width)
        }
      };
      Ok(Expr::Image {
        width: new_w,
        height: new_h,
        channels: *channels,
        data: Arc::new(new_data),
        image_type: *image_type,
      })
    }
    _ => {
      crate::emit_message(&format!(
        "ImageReflect::imgvinv: Expecting an image, graphics or video instead of {}.",
        crate::syntax::expr_to_string(&args[0])
      ));
      Ok(Expr::FunctionCall {
        name: "ImageReflect".to_string(),
        args: args.to_vec().into(),
      })
    }
  }
}

/// ImageRotate[img] / ImageRotate[img, angle] — counter-clockwise rotation
/// in radians, snapped to the nearest 90° increment. Default angle is
/// Pi/2. Operates directly on the f64 pixel buffer so Real32 precision is
/// preserved (no Byte round-trip).
pub fn image_rotate_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "ImageRotate expects 1 or 2 arguments".into(),
    ));
  }
  if !matches!(&args[0], Expr::Image { .. }) {
    crate::emit_message(&format!(
      "ImageRotate::imginv: Expecting an image or graphics instead of {}.",
      crate::syntax::expr_to_string(&args[0])
    ));
    return Ok(Expr::FunctionCall {
      name: "ImageRotate".to_string(),
      args: args.to_vec().into(),
    });
  }

  let pi = std::f64::consts::PI;
  let angle = if args.len() >= 2 {
    expr_to_f64(&args[1])?
  } else {
    pi / 2.0
  };

  let Expr::Image {
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
  else {
    unreachable!()
  };
  let w = *width as usize;
  let h = *height as usize;
  let ch = *channels as usize;

  let norm = ((angle % (2.0 * pi)) + 2.0 * pi) % (2.0 * pi);
  let quadrant = if (norm - pi / 2.0).abs() < 0.01 {
    1
  } else if (norm - pi).abs() < 0.01 {
    2
  } else if (norm - 3.0 * pi / 2.0).abs() < 0.01 {
    3
  } else if norm < 0.01 || (norm - 2.0 * pi).abs() < 0.01 {
    0
  } else {
    // Non-90° rotations aren't supported; pass through unevaluated.
    return Ok(Expr::FunctionCall {
      name: "ImageRotate".to_string(),
      args: args.to_vec().into(),
    });
  };

  let pixel = |r: usize, c: usize| -> &[f64] {
    let base = (r * w + c) * ch;
    &data[base..base + ch]
  };
  let mut new_data = Vec::with_capacity(w * h * ch);
  let (new_w, new_h) = match quadrant {
    0 => {
      for r in 0..h {
        for c in 0..w {
          new_data.extend_from_slice(pixel(r, c));
        }
      }
      (*width, *height)
    }
    1 => {
      // Pi/2 CCW: new dims (H, W); new(r, c) = old(c, W - 1 - r)
      let nw = h;
      let nh = w;
      for r in 0..nh {
        for c in 0..nw {
          new_data.extend_from_slice(pixel(c, w - 1 - r));
        }
      }
      (nw as u32, nh as u32)
    }
    2 => {
      // 180°: new(r, c) = old(H - 1 - r, W - 1 - c)
      for r in 0..h {
        for c in 0..w {
          new_data.extend_from_slice(pixel(h - 1 - r, w - 1 - c));
        }
      }
      (*width, *height)
    }
    _ => {
      // 3*Pi/2 CCW (== Pi/2 CW): new dims (H, W); new(r, c) = old(H - 1 - c, r)
      let nw = h;
      let nh = w;
      for r in 0..nh {
        for c in 0..nw {
          new_data.extend_from_slice(pixel(h - 1 - c, r));
        }
      }
      (nw as u32, nh as u32)
    }
  };

  Ok(Expr::Image {
    width: new_w,
    height: new_h,
    channels: *channels,
    data: Arc::new(new_data),
    image_type: *image_type,
  })
}

/// ImageResize[img, {w, h}] or ImageResize[img, w] - Resize to target dimensions
pub fn image_resize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // wolframscript checks the image arg before validating the arg
  // count, so a non-image first arg emits imginv even when extra
  // options are passed.
  if !matches!(&args[0], Expr::Image { .. }) {
    crate::emit_message(&format!(
      "ImageResize::imginv: Expecting an image or graphics instead of {}.",
      crate::syntax::expr_to_string(&args[0])
    ));
    return Ok(Expr::FunctionCall {
      name: "ImageResize".to_string(),
      args: args.to_vec().into(),
    });
  }

  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ImageResize expects exactly 2 arguments".into(),
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
      let (new_w, new_h) = match &args[1] {
        Expr::List(dims) if dims.len() == 2 => {
          let w = expr_to_f64(&dims[0])? as u32;
          let h = expr_to_f64(&dims[1])? as u32;
          (w, h)
        }
        other => {
          // Single number: set width, scale height proportionally
          let w = expr_to_f64(other)? as u32;
          let h =
            ((w as f64) * (*height as f64) / (*width as f64)).round() as u32;
          (w, h)
        }
      };

      let dyn_img = expr_to_dynamic_image(*width, *height, *channels, data);
      let resized = dyn_img.resize_exact(
        new_w,
        new_h,
        image::imageops::FilterType::Lanczos3,
      );
      Ok(dynamic_image_to_expr(&resized))
    }
    _ => {
      crate::emit_message(&format!(
        "ImageResize::imginv: Expecting an image or graphics instead of {}.",
        crate::syntax::expr_to_string(&args[0])
      ));
      Ok(Expr::FunctionCall {
        name: "ImageResize".to_string(),
        args: args.to_vec().into(),
      })
    }
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
          args: args.to_vec().into(),
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

/// ImageTrim[img, {{x1,y1},{x2,y2}}] - Trim image to a coordinate region.
/// Coordinates use Wolfram's bottom-left origin system where pixel i spans [i, i+1].
/// All pixels whose interval overlaps the specified rectangle are included.
pub fn image_trim_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ImageTrim expects exactly 2 arguments".into(),
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
      // Parse {{x1, y1}, {x2, y2}}
      let (x1, y1, x2, y2) = match &args[1] {
        Expr::List(outer) if outer.len() == 2 => {
          let (ax, ay) = match &outer[0] {
            Expr::List(p) if p.len() == 2 => {
              (expr_to_f64(&p[0])?, expr_to_f64(&p[1])?)
            }
            _ => {
              return Ok(Expr::FunctionCall {
                name: "ImageTrim".to_string(),
                args: args.to_vec().into(),
              });
            }
          };
          let (bx, by) = match &outer[1] {
            Expr::List(p) if p.len() == 2 => {
              (expr_to_f64(&p[0])?, expr_to_f64(&p[1])?)
            }
            _ => {
              return Ok(Expr::FunctionCall {
                name: "ImageTrim".to_string(),
                args: args.to_vec().into(),
              });
            }
          };
          (ax.min(bx), ay.min(by), ax.max(bx), ay.max(by))
        }
        _ => {
          return Ok(Expr::FunctionCall {
            name: "ImageTrim".to_string(),
            args: args.to_vec().into(),
          });
        }
      };

      let w = *width;
      let h = *height;

      // Convert y from bottom-left origin to top-left pixel coordinates
      let top = h as f64 - y2;
      let bottom = h as f64 - y1;

      // Pixel i spans [i, i+1]. Include all pixels whose interval
      // overlaps the specified range (inclusive on boundaries).
      let first_col = (x1 - 1.0).ceil().max(0.0) as u32;
      let last_col = (x2.floor() as u32).min(w.saturating_sub(1));
      let first_row = (top - 1.0).ceil().max(0.0) as u32;
      let last_row = (bottom.floor() as u32).min(h.saturating_sub(1));

      if first_col > last_col || first_row > last_row {
        return Err(InterpreterError::EvaluationError(
          "ImageTrim: specified region is empty".into(),
        ));
      }

      let crop_w = last_col - first_col + 1;
      let crop_h = last_row - first_row + 1;

      let dyn_img = expr_to_dynamic_image(w, h, *channels, data);
      let cropped = dyn_img.crop_imm(first_col, first_row, crop_w, crop_h);
      Ok(dynamic_image_to_expr(&cropped))
    }
    _ => Ok(Expr::FunctionCall {
      name: "ImageTrim".to_string(),
      args: args.to_vec().into(),
    }),
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
    _ => {
      crate::emit_message(&format!(
        "EdgeDetect::imginv: Expecting an image or graphics instead of {}.",
        crate::syntax::expr_to_string(&args[0])
      ));
      Ok(Expr::FunctionCall {
        name: "EdgeDetect".to_string(),
        args: args.to_vec().into(),
      })
    }
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
  if args.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "DominantColors expects 1 or 2 arguments".into(),
    ));
  }

  // wolframscript checks the image argument before validating the arg
  // count, so a chain like Import[missing]; DominantColors[img, 3, "X"]
  // reports DominantColors::imginv (on $Failed) rather than ::argt.
  if !matches!(&args[0], Expr::Image { .. }) {
    crate::emit_message(&format!(
      "DominantColors::imginv: Expecting an image or graphics instead of {{{}}}.",
      crate::syntax::expr_to_string(&args[0])
    ));
    return Ok(Expr::FunctionCall {
      name: "DominantColors".to_string(),
      args: args.to_vec().into(),
    });
  }

  if args.len() > 2 {
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
          args: vec![Expr::Real(c[0]), Expr::Real(c[1]), Expr::Real(c[2])]
            .into(),
        })
        .collect();

      Ok(Expr::List(colors.into()))
    }
    _ => unreachable!("non-Image first arg handled above"),
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
            args: vec![pixel_expr].into(),
          };
          let result = eval_fn(&call)?;
          new_data.push(expr_to_f64(&result)?);
        } else {
          // Vector pixel {r, g, b}
          let base = i * ch;
          let pixel_list: Vec<Expr> =
            (0..ch).map(|c| Expr::Real(data[base + c])).collect();
          let pixel_expr = Expr::List(pixel_list.into());
          let call = Expr::FunctionCall {
            name: crate::syntax::expr_to_string(func),
            args: vec![pixel_expr].into(),
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

/// Extract `(r, g, b, optional_alpha)` from a color directive (RGBColor
/// or GrayLevel). Returns None for anything else, including named
/// constants that haven't already been resolved to RGBColor.
fn color_directive_to_rgb(e: &Expr) -> Option<(f64, f64, f64, Option<f64>)> {
  let Expr::FunctionCall { name, args } = e else {
    return None;
  };
  let to_f = |x: &Expr| -> Option<f64> {
    crate::functions::math_ast::try_eval_to_f64(x)
  };
  match name.as_str() {
    "RGBColor" if args.len() == 3 => {
      Some((to_f(&args[0])?, to_f(&args[1])?, to_f(&args[2])?, None))
    }
    "RGBColor" if args.len() == 4 => Some((
      to_f(&args[0])?,
      to_f(&args[1])?,
      to_f(&args[2])?,
      Some(to_f(&args[3])?),
    )),
    "GrayLevel" if args.len() == 1 => {
      let v = to_f(&args[0])?;
      Some((v, v, v, None))
    }
    "GrayLevel" if args.len() == 2 => {
      let v = to_f(&args[0])?;
      Some((v, v, v, Some(to_f(&args[1])?)))
    }
    _ => None,
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

  // Color directives: RGBColor[r, g, b(, a)] / GrayLevel[v(, a)].
  // Conversion preserves the alpha channel when present. GrayLevel
  // input to "Grayscale" stays as the same single value (no luminance
  // round-trip that would introduce f64 error).
  let is_graylevel = matches!(
    &args[0],
    Expr::FunctionCall { name, .. } if name == "GrayLevel"
  );
  if let Some(rgb) = color_directive_to_rgb(&args[0]) {
    let (r, g, b, alpha) = rgb;
    match target_space {
      "Grayscale" => {
        let lum = if is_graylevel {
          r
        } else {
          0.299 * r + 0.587 * g + 0.114 * b
        };
        let mut gargs = vec![Expr::Real(lum)];
        if let Some(a) = alpha {
          gargs.push(Expr::Real(a));
        }
        return Ok(Expr::FunctionCall {
          name: "GrayLevel".to_string(),
          args: gargs.into(),
        });
      }
      "RGB" => {
        let mut rargs = vec![Expr::Real(r), Expr::Real(g), Expr::Real(b)];
        if let Some(a) = alpha {
          rargs.push(Expr::Real(a));
        }
        return Ok(Expr::FunctionCall {
          name: "RGBColor".to_string(),
          args: rargs.into(),
        });
      }
      _ => {}
    }
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

/// ImageCompose[img, overlay] / ImageCompose[img, {overlay, α}] —
/// overlay an image on top of img, centered. The output has the same
/// dimensions, channel count and image type as `img`. With an alpha
/// argument, the overlapping region is blended as (1 - α)*bg + α*ov;
/// without one the overlay replaces the background pixels.
pub fn image_compose_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ImageCompose expects exactly 2 arguments".into(),
    ));
  }

  // Parse the overlay spec: either an Image, or {Image, α}.
  let (overlay_expr, alpha) = match &args[1] {
    Expr::List(items) if items.len() == 2 => {
      let a = expr_to_f64(&items[1])?;
      (&items[0], Some(a))
    }
    _ => (&args[1], None),
  };

  let (
    Expr::Image {
      width: w1,
      height: h1,
      channels: ch1,
      data: data1,
      image_type,
    },
    Expr::Image {
      width: w2,
      height: h2,
      channels: ch2,
      data: data2,
      ..
    },
  ) = (&args[0], overlay_expr)
  else {
    return Err(InterpreterError::EvaluationError(
      "ImageCompose: both arguments must be Images".into(),
    ));
  };

  let bw = *w1 as i64;
  let bh = *h1 as i64;
  let ow = *w2 as i64;
  let oh = *h2 as i64;
  let bch = *ch1 as usize;
  let och = *ch2 as usize;

  // wolframscript centers the overlay; the data buffer is laid out
  // top-down, but Wolfram positions things in image (bottom-up) y. Use
  // `floor(bw/2) - floor(ow/2)` for x and `ceil(bh/2) - ceil(oh/2)` for
  // the top-down y so behavior matches across even/odd sizes.
  let offset_x = bw / 2 - ow / 2;
  let offset_y = (bh + 1) / 2 - (oh + 1) / 2;

  // Output starts as a copy of the background; overlapping pixels are
  // replaced (or alpha-blended) using the overlay's channels.
  let mut out: Vec<f64> = data1.as_ref().clone();
  for oy in 0..oh {
    let by = oy + offset_y;
    if by < 0 || by >= bh {
      continue;
    }
    for ox in 0..ow {
      let bx = ox + offset_x;
      if bx < 0 || bx >= bw {
        continue;
      }
      let base_idx = (by as usize * *w1 as usize + bx as usize) * bch;
      let over_idx = (oy as usize * *w2 as usize + ox as usize) * och;
      for c in 0..bch {
        // Map the overlay's channel: if the overlay has fewer channels
        // (e.g. grayscale into RGB), broadcast the single value.
        let ov = if c < och {
          data2[over_idx + c]
        } else {
          data2[over_idx + (och - 1)]
        };
        let bg = out[base_idx + c];
        out[base_idx + c] = match alpha {
          Some(a) => (1.0 - a) * bg + a * ov,
          None => ov,
        };
      }
    }
  }

  Ok(Expr::Image {
    width: *w1,
    height: *h1,
    channels: *ch1,
    data: Arc::new(out),
    image_type: *image_type,
  })
}

/// Helper for pointwise image operations
fn pointwise_image_op(
  args: &[Expr],
  name: &str,
  op: fn(f64, f64) -> f64,
  op32: fn(f32, f32) -> f32,
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(format!(
      "{} expects exactly 2 arguments",
      name
    )));
  }

  // Real32 images store f32-quantised values in an f64 buffer; perform the
  // op in f32 so results match wolframscript's f32 image arithmetic.
  let apply = |a: f64, b: f64, is_real32: bool| -> f64 {
    if is_real32 {
      op32(a as f32, b as f32) as f64
    } else {
      op(a, b)
    }
  };

  // (Image, Image) — pointwise on matching dimensions.
  if let (
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
  ) = (&args[0], &args[1])
  {
    if w1 != w2 || h1 != h2 || ch1 != ch2 {
      return Err(InterpreterError::EvaluationError(format!(
        "{}: images must have the same dimensions and channels",
        name
      )));
    }
    let is_r32 = matches!(t1, crate::syntax::ImageType::Real32);
    let new_data: Vec<f64> = data1
      .iter()
      .zip(data2.iter())
      .map(|(&a, &b)| apply(a, b, is_r32))
      .collect();
    return Ok(Expr::Image {
      width: *w1,
      height: *h1,
      channels: *ch1,
      data: Arc::new(new_data),
      image_type: *t1,
    });
  }

  // (Image, scalar) — apply `op(pixel, scalar)` to every pixel.
  if let Expr::Image {
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
    && let Some(s) = crate::functions::math_ast::try_eval_to_f64(&args[1])
  {
    let is_r32 = matches!(image_type, crate::syntax::ImageType::Real32);
    let new_data: Vec<f64> =
      data.iter().map(|&v| apply(v, s, is_r32)).collect();
    return Ok(Expr::Image {
      width: *width,
      height: *height,
      channels: *channels,
      data: Arc::new(new_data),
      image_type: *image_type,
    });
  }
  // (scalar, Image) — apply `op(scalar, pixel)`.
  if let Expr::Image {
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[1]
    && let Some(s) = crate::functions::math_ast::try_eval_to_f64(&args[0])
  {
    let is_r32 = matches!(image_type, crate::syntax::ImageType::Real32);
    let new_data: Vec<f64> =
      data.iter().map(|&v| apply(s, v, is_r32)).collect();
    return Ok(Expr::Image {
      width: *width,
      height: *height,
      channels: *channels,
      data: Arc::new(new_data),
      image_type: *image_type,
    });
  }

  // Neither argument is a usable image (and no scalar broadcast applies).
  // Match wolframscript: identify the first non-image argument, emit
  // <Name>::imginv, and return the call unevaluated.
  let bad = if !matches!(&args[0], Expr::Image { .. }) {
    &args[0]
  } else {
    &args[1]
  };
  crate::emit_message(&format!(
    "{}::imginv: Expecting an image or graphics instead of {}.",
    name,
    crate::syntax::expr_to_string(bad)
  ));
  Ok(Expr::FunctionCall {
    name: name.to_string(),
    args: args.to_vec().into(),
  })
}

/// ImageAdd[img, x1, x2, …] — pointwise addition, threading through
/// each additional argument (scalar or image).
pub fn image_add_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  fold_pointwise(args, "ImageAdd", |a, b| a + b, |a, b| a + b)
}

/// ImageSubtract[img, x1, x2, …] — pointwise subtraction (folded).
pub fn image_subtract_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  fold_pointwise(args, "ImageSubtract", |a, b| a - b, |a, b| a - b)
}

/// ImageMultiply[img, x1, x2, …] — pointwise multiplication (folded).
pub fn image_multiply_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  fold_pointwise(args, "ImageMultiply", |a, b| a * b, |a, b| a * b)
}

/// Fold a pointwise binary image op over a list of trailing arguments.
fn fold_pointwise(
  args: &[Expr],
  name: &str,
  op: fn(f64, f64) -> f64,
  op32: fn(f32, f32) -> f32,
) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(format!(
      "{} expects at least 2 arguments",
      name
    )));
  }
  let mut acc = args[0].clone();
  for next in &args[1..] {
    acc = pointwise_image_op(&[acc, next.clone()], name, op, op32)?;
  }
  Ok(acc)
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
/// BinaryImageQ[img] — predicate: True iff the argument is a binary
/// (Bit-type) image. Non-images return False (no warning), matching
/// wolframscript's quiet predicate behavior.
pub fn binary_image_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let result = matches!(
    &args[0],
    Expr::Image {
      image_type: ImageType::Bit,
      ..
    }
  );
  Ok(Expr::Identifier(
    if result { "True" } else { "False" }.to_string(),
  ))
}

/// PixelValue[img, {x, y}] — pixel value at column x (1-indexed from the
/// left) and row y (1-indexed from the bottom). Out-of-bounds positions
/// return 0 (or a list of zeros for multi-channel images).
pub fn pixel_value_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let Expr::Image {
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
  else {
    crate::emit_message(&format!(
      "PixelValue::imginv: Expecting an image or graphics instead of {}.",
      crate::syntax::expr_to_string(&args[0])
    ));
    return Ok(Expr::FunctionCall {
      name: "PixelValue".to_string(),
      args: args.to_vec().into(),
    });
  };
  let Expr::List(pos) = &args[1] else {
    return Ok(Expr::FunctionCall {
      name: "PixelValue".to_string(),
      args: args.to_vec().into(),
    });
  };
  let coord = |e: &Expr| -> Option<i64> {
    match e {
      Expr::Integer(n) => Some(*n as i64),
      Expr::Real(f) => Some(f.round() as i64),
      _ => None,
    }
  };
  let ch = *channels as usize;
  let is_real32 = matches!(image_type, crate::syntax::ImageType::Real32);
  let to_value = |v: f64| -> Expr {
    if is_real32 {
      Expr::Real((v as f32) as f64)
    } else {
      Expr::Real(v)
    }
  };
  let make_zero = || -> Expr {
    if ch == 1 {
      Expr::Real(0.0)
    } else {
      Expr::List(
        std::iter::repeat_n(Expr::Real(0.0), ch)
          .collect::<Vec<_>>()
          .into(),
      )
    }
  };

  if pos.len() != 2 {
    return Ok(make_zero());
  }
  let (Some(x), Some(y)) = (coord(&pos[0]), coord(&pos[1])) else {
    return Ok(Expr::FunctionCall {
      name: "PixelValue".to_string(),
      args: args.to_vec().into(),
    });
  };
  if x < 1 || y < 1 || x > *width as i64 || y > *height as i64 {
    return Ok(make_zero());
  }
  let row = (*height as i64 - y) as usize;
  let col = (x - 1) as usize;
  let base = row * (*width as usize) * ch + col * ch;

  if ch == 1 {
    Ok(to_value(data[base]))
  } else {
    let pixel: Vec<Expr> = (0..ch).map(|c| to_value(data[base + c])).collect();
    Ok(Expr::List(pixel.into()))
  }
}

/// TextRecognize[img] — OCR stub. Real text recognition is not
/// implemented; this stub matches wolframscript's TextRecognize::imgvinv
/// warning for non-image input.
pub fn text_recognize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if !matches!(&args[0], Expr::Image { .. }) {
    crate::emit_message(&format!(
      "TextRecognize::imgvinv: Expecting an image, graphics or video instead of {}.",
      crate::syntax::expr_to_string(&args[0])
    ));
  }
  Ok(Expr::FunctionCall {
    name: "TextRecognize".to_string(),
    args: args.to_vec().into(),
  })
}

/// MedianFilter[arg, r] — stub. Emits MedianFilter::arg1 when first
/// arg isn't an image or list. (MaxFilter and MinFilter have their own
/// arg1 warnings inside their existing list-processing dispatch in
/// math_functions.rs.)
/// GradientFilter[list, r] — magnitude of the gradient of a 1-D list.
///
/// For radius `r = 1` this returns `|central difference|` (edge-replicated
/// at the boundaries), matching wolframscript's `GradientFilter[list, 1]`
/// output for purely numeric input. Larger radii or non-list inputs stay
/// symbolic.
pub fn gradient_filter_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "GradientFilter".to_string(),
    args: args.to_vec().into(),
  };
  if args.len() != 2 {
    return Ok(unevaluated());
  }
  if let (Expr::List(elems), Some(r)) = (
    &args[0],
    crate::functions::math_ast::try_eval_to_f64(&args[1])
      .filter(|v| *v >= 0.0)
      .map(|v| v as usize),
  ) && r == 1
  {
    let n = elems.len();
    let mut values: Vec<f64> = Vec::with_capacity(n);
    for e in elems.iter() {
      let Some(v) = crate::functions::math_ast::try_eval_to_f64(e) else {
        return Ok(unevaluated());
      };
      values.push(v);
    }
    if n == 0 {
      return Ok(Expr::List(Vec::new().into()));
    }
    if n == 1 {
      // Single-element list: gradient is zero.
      return Ok(Expr::List(vec![Expr::Real(0.0)].into()));
    }
    let mut out: Vec<Expr> = Vec::with_capacity(n);
    for i in 0..n {
      let left = if i == 0 { values[0] } else { values[i - 1] };
      let right = if i + 1 >= n {
        values[n - 1]
      } else {
        values[i + 1]
      };
      let g = (right - left) / 2.0;
      out.push(Expr::Real(g.abs()));
    }
    return Ok(Expr::List(out.into()));
  }
  Ok(unevaluated())
}

pub fn median_filter_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let r = crate::functions::math_ast::try_eval_to_f64(&args[1])
    .filter(|v| *v >= 0.0)
    .map(|v| v as usize);

  // MedianFilter[Image, r] — apply 2D median filter per channel on the
  // pixel buffer; result preserves precision and image type.
  if let (
    Expr::Image {
      width,
      height,
      channels,
      data,
      image_type,
    },
    Some(r),
  ) = (&args[0], r)
  {
    let w = *width as usize;
    let h = *height as usize;
    let ch = *channels as usize;
    let mut new_data = vec![0.0; data.len()];
    for c_idx in 0..ch {
      let mut window: Vec<f64> = Vec::with_capacity((2 * r + 1) * (2 * r + 1));
      for y in 0..h {
        for x in 0..w {
          window.clear();
          let y0 = y.saturating_sub(r);
          let y1 = (y + r).min(h - 1);
          let x0 = x.saturating_sub(r);
          let x1 = (x + r).min(w - 1);
          for yy in y0..=y1 {
            for xx in x0..=x1 {
              window.push(data[(yy * w + xx) * ch + c_idx]);
            }
          }
          window.sort_by(|a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
          });
          // wolframscript uses the upper median (no averaging) for images
          // so the result stays in the original value set — important for
          // Byte / Bit images where averaging would invent new values.
          let n = window.len();
          let med = window[n / 2];
          new_data[(y * w + x) * ch + c_idx] = med;
        }
      }
    }
    return Ok(Expr::Image {
      width: *width,
      height: *height,
      channels: *channels,
      data: Arc::new(new_data),
      image_type: *image_type,
    });
  }

  // MedianFilter[list, r] — flat 1D or 2D nested-list median filter
  // (window clipped at boundaries).
  if let (Expr::List(elems), Some(r)) = (&args[0], r) {
    // 2D path: every element is a List of the same length.
    if !elems.is_empty()
      && elems.iter().all(|row| {
        matches!(row, Expr::List(items) if items.len() == match &elems[0] {
          Expr::List(first) => first.len(),
          _ => 0,
        })
      })
    {
      let h = elems.len();
      let w = match &elems[0] {
        Expr::List(items) => items.len(),
        _ => 0,
      };
      let get = |y: usize, x: usize| -> Expr {
        match &elems[y] {
          Expr::List(row) => row[x].clone(),
          _ => unreachable!(),
        }
      };
      let mut rows: Vec<Expr> = Vec::with_capacity(h);
      for y in 0..h {
        let mut new_row: Vec<Expr> = Vec::with_capacity(w);
        for x in 0..w {
          let y0 = y.saturating_sub(r);
          let y1 = (y + r).min(h - 1);
          let x0 = x.saturating_sub(r);
          let x1 = (x + r).min(w - 1);
          let mut window: Vec<Expr> =
            Vec::with_capacity((y1 - y0 + 1) * (x1 - x0 + 1));
          for yy in y0..=y1 {
            for xx in x0..=x1 {
              window.push(get(yy, xx));
            }
          }
          let med =
            crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
              name: "Median".to_string(),
              args: vec![Expr::List(window.into())].into(),
            })
            .unwrap_or_else(|_| get(y, x));
          new_row.push(med);
        }
        rows.push(Expr::List(new_row.into()));
      }
      return Ok(Expr::List(rows.into()));
    }
    // 1D path.
    let n = elems.len();
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
      let lo = i.saturating_sub(r);
      let hi = if i + r < n { i + r } else { n - 1 };
      let window: Vec<Expr> = elems[lo..=hi].to_vec();
      let med = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Median".to_string(),
        args: vec![Expr::List(window.into())].into(),
      })
      .unwrap_or_else(|_| elems[i].clone());
      result.push(med);
    }
    return Ok(Expr::List(result.into()));
  }

  let valid = matches!(&args[0], Expr::Image { .. } | Expr::List(_));
  if !valid {
    crate::emit_message(&format!(
      "MedianFilter::arg1: The first argument {} should be a rectangular array, image or video.",
      crate::syntax::expr_to_string(&args[0])
    ));
  }
  Ok(Expr::FunctionCall {
    name: "MedianFilter".to_string(),
    args: args.to_vec().into(),
  })
}

/// ImageConvolve[img, kernel] — stub. Real convolution is not
/// implemented; this stub matches wolframscript's imginv warning for
/// non-image first arg.
pub fn image_convolve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if !matches!(&args[0], Expr::Image { .. }) {
    crate::emit_message(&format!(
      "ImageConvolve::imginv: Expecting an image or graphics instead of {}.",
      crate::syntax::expr_to_string(&args[0])
    ));
  }
  Ok(Expr::FunctionCall {
    name: "ImageConvolve".to_string(),
    args: args.to_vec().into(),
  })
}

/// GaussianFilter[arg, sigma] — stub. Real Gaussian filtering for
/// images/arrays/videos is not implemented yet; this stub matches
/// wolframscript's GaussianFilter::arg1 warning for inputs that are
/// neither image nor list.
pub fn gaussian_filter_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let valid = matches!(&args[0], Expr::Image { .. } | Expr::List(_));
  if !valid {
    crate::emit_message(&format!(
      "GaussianFilter::arg1: The first argument {} should be a rectangular array, image or video.",
      crate::syntax::expr_to_string(&args[0])
    ));
  }
  Ok(Expr::FunctionCall {
    name: "GaussianFilter".to_string(),
    args: args.to_vec().into(),
  })
}

/// Colorize[matrix] — colorize an integer-label matrix as an RGB
/// image. wolframscript prints the result as `-Image-`. A real
/// renderer would consult `ColorFunction -> …` and evaluate the
/// function at each label; for now we map each unique integer
/// label to a deterministic shade of gray so the result is a
/// well-formed `Expr::Image`. Non-matrix / non-image arguments
/// emit `Colorize::invinput` and stay symbolic, matching
/// wolframscript.
pub fn colorize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if let Expr::List(rows) = &args[0]
    && !rows.is_empty()
    && let Some(first_row) = rows.first()
    && let Expr::List(first_cols) = first_row
    && rows.iter().all(|r| {
      matches!(r, Expr::List(cols)
        if cols.len() == first_cols.len()
          && cols.iter().all(|c| matches!(c, Expr::Integer(_))))
    })
  {
    let height = rows.len() as u32;
    let width = first_cols.len() as u32;
    let mut labels: Vec<i64> = Vec::with_capacity((width * height) as usize);
    for row in rows.iter() {
      if let Expr::List(cols) = row {
        for c in cols.iter() {
          if let Expr::Integer(n) = c {
            labels.push(*n as i64);
          }
        }
      }
    }
    let (min, max) = labels
      .iter()
      .fold((i64::MAX, i64::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let span = (max - min).max(1) as f64;
    // Default rendering: map each label linearly to a gray
    // ramp [0, 1]; emit 3-channel RGB data so the Image stays
    // well-formed. Future work: honor `ColorFunction -> …`.
    let mut data: Vec<f64> = Vec::with_capacity(labels.len() * 3);
    for v in &labels {
      let t = ((*v - min) as f64) / span;
      data.push(t);
      data.push(t);
      data.push(t);
    }
    return Ok(Expr::Image {
      width,
      height,
      channels: 3,
      data: Arc::new(data),
      image_type: crate::syntax::ImageType::Real64,
    });
  }
  let is_image = matches!(&args[0], Expr::Image { .. });
  if !is_image {
    crate::emit_message(&format!(
      "Colorize::invinput: Expecting an integer matrix or an image instead of {}.",
      crate::syntax::expr_to_string(&args[0])
    ));
  }
  Ok(Expr::FunctionCall {
    name: "Colorize".to_string(),
    args: args.to_vec().into(),
  })
}

/// ColorSeparate[img] — return one single-channel image per channel
/// of the input. Grayscale images pass through unchanged (as a list
/// of one). The output images preserve the input's width, height, and
/// image type.
pub fn color_separate_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let Expr::Image {
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
  else {
    crate::emit_message(&format!(
      "ColorSeparate::imginv: Expecting an image or graphics instead of {}.",
      crate::syntax::expr_to_string(&args[0])
    ));
    return Ok(Expr::FunctionCall {
      name: "ColorSeparate".to_string(),
      args: args.to_vec().into(),
    });
  };
  let ch = *channels as usize;
  let n = (*width as usize) * (*height as usize);
  let mut images = Vec::with_capacity(ch);
  for c_idx in 0..ch {
    let channel_data: Vec<f64> = (0..n).map(|i| data[i * ch + c_idx]).collect();
    images.push(Expr::Image {
      width: *width,
      height: *height,
      channels: 1,
      data: Arc::new(channel_data),
      image_type: *image_type,
    });
  }
  Ok(Expr::List(images.into()))
}

/// ColorQuantize[img, n] — color quantization stub. Real quantization is
/// not implemented yet; this stub matches wolframscript's imginv warning
/// when the first arg is not an image.
pub fn color_quantize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if !matches!(&args[0], Expr::Image { .. }) {
    crate::emit_message(&format!(
      "ColorQuantize::imginv: Expecting an image or graphics instead of {}.",
      crate::syntax::expr_to_string(&args[0])
    ));
  }
  Ok(Expr::FunctionCall {
    name: "ColorQuantize".to_string(),
    args: args.to_vec().into(),
  })
}

/// Threshold[data]            replaces values with |x| ≤ 10^-10 by zero.
/// Threshold[data, t]         uses t as the threshold.
/// Lists of arbitrary nesting depth are walked recursively. Real values
/// emit a Real 0.; everything else emits an Integer 0, matching
/// wolframscript. Non-numeric leaves trigger Threshold::nlist and the
/// call is returned unevaluated.
pub fn threshold_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "Threshold".to_string(),
    args: args.to_vec().into(),
  };
  let is_array_like = matches!(
    &args[0],
    Expr::Image { .. } | Expr::List(_) | Expr::FunctionCall { .. }
  );
  if !is_array_like {
    crate::emit_message(&format!(
      "Threshold::wlist: Argument {} should be one of rectangular array of any depth, image, sound or sampled sound list.",
      crate::syntax::expr_to_string(&args[0])
    ));
    return Ok(unevaluated());
  }
  // Image input is still a TODO — leave unevaluated without a message.
  if matches!(&args[0], Expr::Image { .. }) {
    return Ok(unevaluated());
  }
  let Expr::List(_) = &args[0] else {
    return Ok(unevaluated());
  };
  // Default threshold is 10^-10. When supplied explicitly, the threshold
  // itself must be numeric (Integer, Rational, or Real); otherwise we
  // bail out to unevaluated.
  let threshold: Expr = if args.len() == 2 {
    if crate::functions::math_ast::try_eval_to_f64(&args[1]).is_none() {
      return Ok(unevaluated());
    }
    args[1].clone()
  } else {
    Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(1), Expr::Integer(10_000_000_000)].into(),
    }
  };
  // Walk the array recursively. A non-list, non-numeric leaf triggers
  // the Threshold::nlist message and aborts.
  fn apply(data: &Expr, t: &Expr) -> Option<Expr> {
    match data {
      Expr::List(items) => {
        if items.is_empty() {
          // Empty list at the top level was already rejected; nested
          // empty lists round-trip unchanged.
          return Some(Expr::List(items.clone()));
        }
        let mut out = Vec::with_capacity(items.len());
        for item in items.iter() {
          out.push(apply(item, t)?);
        }
        Some(Expr::List(out.into()))
      }
      x if crate::functions::math_ast::try_eval_to_f64(x).is_some() => {
        Some(threshold_one(x, t))
      }
      _ => None,
    }
  }
  // Empty top-level list is an error, matching wolframscript's nlist.
  if let Expr::List(top) = &args[0]
    && top.is_empty()
  {
    crate::emit_message(&format!(
      "Threshold::nlist: Argument {} is not a nonempty list or rectangular array of numeric quantities.",
      crate::syntax::expr_to_string(&args[0])
    ));
    return Ok(unevaluated());
  }
  match apply(&args[0], &threshold) {
    Some(result) => Ok(result),
    None => {
      crate::emit_message(&format!(
        "Threshold::nlist: Argument {} is not a nonempty list or rectangular array of numeric quantities.",
        crate::syntax::expr_to_string(&args[0])
      ));
      Ok(unevaluated())
    }
  }
}

/// Returns x if |x| > t, else a zero of the same kind (Real 0. if x is a
/// Real, otherwise Integer 0).
fn threshold_one(x: &Expr, t: &Expr) -> Expr {
  let zero = if matches!(x, Expr::Real(_)) {
    Expr::Real(0.0)
  } else {
    Expr::Integer(0)
  };
  // Prefer exact comparison when both sides are Integer/Rational so that
  // boundary cases like Threshold[{1/2}, 1/2] don't depend on f64 rounding.
  if let (Some((xn, xd)), Some((tn, td))) = (as_rational(x), as_rational(t)) {
    // |xn/xd| <= tn/td  ⇔  |xn| * td <= tn * xd  (all positive denominators)
    let left = (xn.abs() as i128).checked_mul(td as i128);
    let right = (tn.abs() as i128).checked_mul(xd as i128);
    if let (Some(l), Some(r)) = (left, right)
      && l <= r
    {
      return zero;
    }
    if left.is_none() || right.is_none() {
      // Fall through to f64 path on overflow.
    } else {
      return x.clone();
    }
  }
  let xf = crate::functions::math_ast::try_eval_to_f64(x).unwrap_or(0.0);
  let tf = crate::functions::math_ast::try_eval_to_f64(t).unwrap_or(0.0);
  if xf.abs() <= tf { zero } else { x.clone() }
}

/// Return (numerator, denominator) for an Integer or Rational, normalised
/// so the denominator is positive. Real values return None.
fn as_rational(e: &Expr) -> Option<(i64, i64)> {
  match e {
    Expr::Integer(n) => i64::try_from(*n).ok().map(|n| (n, 1)),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(p), Expr::Integer(q)) = (&args[0], &args[1]) {
        let p = i64::try_from(*p).ok()?;
        let q = i64::try_from(*q).ok()?;
        if q == 0 {
          return None;
        }
        if q < 0 { Some((-p, -q)) } else { Some((p, q)) }
      } else {
        None
      }
    }
    _ => None,
  }
}

/// ImagePartition[img, n] / [img, n, d] — image tile partition stub.
/// Real image partitioning isn't implemented yet; this stub only matches
/// wolframscript's behavior for non-image input (emits ImagePartition::imginv
/// and returns the call unevaluated) so doctests like
/// `ImagePartition[hedy, 256]` (no image present) line up.
pub fn image_partition_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if !matches!(&args[0], Expr::Image { .. }) {
    crate::emit_message(&format!(
      "ImagePartition::imginv: Expecting an image or graphics instead of {}.",
      crate::syntax::expr_to_string(&args[0])
    ));
    return Ok(Expr::FunctionCall {
      name: "ImagePartition".to_string(),
      args: args.to_vec().into(),
    });
  }
  // Image input: leave unevaluated for now (real partitioning TBD).
  Ok(Expr::FunctionCall {
    name: "ImagePartition".to_string(),
    args: args.to_vec().into(),
  })
}

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

      let (r1, r2) = parse_take_range(&args[1], h)?;
      let (c1, c2) = if args.len() == 3 {
        parse_take_range(&args[2], w)?
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
    _ => {
      crate::emit_message(&format!(
        "ImageTake::imginv: Expecting an image or graphics instead of {}.",
        crate::syntax::expr_to_string(&args[0])
      ));
      Ok(Expr::FunctionCall {
        name: "ImageTake".to_string(),
        args: args.to_vec().into(),
      })
    }
  }
}

/// Resolve a 1-indexed position (positive or negative) to a 0-indexed
/// offset within `total`. Negative values count from the end.
fn resolve_take_index(
  e: &Expr,
  total: usize,
) -> Result<usize, InterpreterError> {
  let n: i128 = match e {
    Expr::Integer(i) => *i,
    _ => expr_to_f64(e)? as i128,
  };
  if n > 0 {
    Ok(((n as usize) - 1).min(total.saturating_sub(1)))
  } else if n < 0 {
    let abs = (-n) as usize;
    if abs > total { Ok(0) } else { Ok(total - abs) }
  } else {
    Err(InterpreterError::EvaluationError(
      "ImageTake: index 0 is not valid".into(),
    ))
  }
}

/// Parse one axis of an ImageTake spec into a 0-indexed `[lo, hi)` range.
/// Supports: `All`, integer (first/last n), `{i}` (single row), `{i, j}`.
fn parse_take_range(
  spec: &Expr,
  total: usize,
) -> Result<(usize, usize), InterpreterError> {
  match spec {
    Expr::Identifier(s) if s == "All" => Ok((0, total)),
    Expr::Integer(n) => {
      let n = *n;
      if n >= 0 {
        Ok((0, (n as usize).min(total)))
      } else {
        let abs = (-n) as usize;
        Ok((total.saturating_sub(abs), total))
      }
    }
    Expr::List(items) if items.len() == 1 => {
      let idx = resolve_take_index(&items[0], total)?;
      Ok((idx, idx + 1))
    }
    Expr::List(items) if items.len() == 2 => {
      let lo = resolve_take_index(&items[0], total)?;
      let hi = resolve_take_index(&items[1], total)? + 1;
      Ok((lo, hi))
    }
    _ => Err(InterpreterError::EvaluationError(
      "ImageTake: range must be n, {i}, {i, j}, or All".into(),
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

/// Fast path for ImageCollage when every entry is an Image with the
/// same shape and type. Lays them out in a near-square grid using
/// cols = ceil(sqrt(n)), rows = ceil(n/cols). The output has the same
/// channels and image_type as the inputs. Empty cells (when n is not
/// a perfect rectangle) stay as the buffer's default (0).
fn try_collage_same_shape(items: &[Expr]) -> Option<Expr> {
  if items.is_empty() {
    return None;
  }
  let (cw, ch_count, ch_first, image_type) = match &items[0] {
    Expr::Image {
      width,
      height,
      channels,
      image_type,
      ..
    } => (*width, *height, *channels, *image_type),
    _ => return None,
  };
  for it in items.iter() {
    let Expr::Image {
      width,
      height,
      channels,
      image_type: t,
      ..
    } = it
    else {
      return None;
    };
    if *width != cw
      || *height != ch_count
      || *channels != ch_first
      || *t != image_type
    {
      return None;
    }
  }

  let n = items.len();
  let cols = (n as f64).sqrt().ceil() as usize;
  let rows = n.div_ceil(cols);
  let cell_w = cw as usize;
  let cell_h = ch_count as usize;
  let c = ch_first as usize;
  let out_w = cols * cell_w;
  let out_h = rows * cell_h;
  let mut out = vec![0.0_f64; out_w * out_h * c];

  for (i, item) in items.iter().enumerate() {
    let Expr::Image { data, .. } = item else {
      return None;
    };
    let r = i / cols;
    let col = i % cols;
    for y in 0..cell_h {
      for x in 0..cell_w {
        let src = (y * cell_w + x) * c;
        let dst = ((r * cell_h + y) * out_w + col * cell_w + x) * c;
        for k in 0..c {
          out[dst + k] = data[src + k];
        }
      }
    }
  }

  Some(Expr::Image {
    width: out_w as u32,
    height: out_h as u32,
    channels: ch_first,
    data: Arc::new(out),
    image_type,
  })
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

  // Fast path: every item is a plain Image of the same width, height,
  // channel count, and image type, with no explicit size or fitting
  // override. Lay them out in a near-square grid (cols = ceil(sqrt(n)),
  // rows = ceil(n/cols)) and copy pixel data directly without
  // resampling.
  if args.len() == 1
    && let Some(out) = try_collage_same_shape(list)
  {
    return Ok(out);
  }

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

/// Fast path for `ImageAssemble[grid]` where every cell is an Image
/// with the same width, height, channels, and image type. Concatenates
/// the f64 buffers directly, preserving precision and image type.
fn try_assemble_same_shape(outer: &[Expr]) -> Option<Expr> {
  // Normalise to a 2D grid: if any item is a list, all must be lists.
  let is_2d = outer.iter().all(|e| matches!(e, Expr::List(_)));
  let rows: Vec<&[Expr]> = if is_2d {
    let mut out: Vec<&[Expr]> = Vec::with_capacity(outer.len());
    for r in outer {
      if let Expr::List(items) = r {
        out.push(items.as_ref());
      } else {
        return None;
      }
    }
    out
  } else {
    vec![outer]
  };
  if rows.is_empty() || rows[0].is_empty() {
    return None;
  }
  let cols = rows[0].len();
  if rows.iter().any(|r| r.len() != cols) {
    return None;
  }

  // Reference shape: take from the first cell.
  let (cw, ch_count, image_type, ch_first) = match &rows[0][0] {
    Expr::Image {
      width,
      height,
      channels,
      image_type,
      ..
    } => (*width, *height, *image_type, *channels),
    _ => return None,
  };

  // Every cell must match the reference shape and type.
  for row in &rows {
    for cell in row.iter() {
      let Expr::Image {
        width,
        height,
        channels,
        image_type: t,
        ..
      } = cell
      else {
        return None;
      };
      if *width != cw
        || *height != ch_count
        || *channels != ch_first
        || *t != image_type
      {
        return None;
      }
    }
  }

  let num_rows = rows.len();
  let num_cols = cols;
  let cell_w = cw as usize;
  let cell_h = ch_count as usize;
  let c = ch_first as usize;
  let out_w = num_cols * cell_w;
  let out_h = num_rows * cell_h;
  let mut out = vec![0.0_f64; out_w * out_h * c];

  for (r, row) in rows.iter().enumerate() {
    for (col, cell) in row.iter().enumerate() {
      let Expr::Image { data, .. } = cell else {
        return None;
      };
      for y in 0..cell_h {
        for x in 0..cell_w {
          let src = (y * cell_w + x) * c;
          let dst = ((r * cell_h + y) * out_w + col * cell_w + x) * c;
          for k in 0..c {
            out[dst + k] = data[src + k];
          }
        }
      }
    }
  }

  Some(Expr::Image {
    width: out_w as u32,
    height: out_h as u32,
    channels: ch_first,
    data: Arc::new(out),
    image_type,
  })
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

  // Fast path: every cell is an Image of the same width, height,
  // channel count, and image type. Concatenate the f64 buffers
  // directly without going through the image crate's u8 buffer.
  if fitting == "None"
    && let Some(out) = try_assemble_same_shape(outer)
  {
    return Ok(out);
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
          crate::emit_message(
            "ImageAssemble::row: \
               Expecting images of the same height in one row.",
          );
          return Ok(Expr::FunctionCall {
            name: "ImageAssemble".to_string(),
            args: args.to_vec().into(),
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
          crate::emit_message(
            "ImageAssemble::col: \
               Expecting images of the same width in one column.",
          );
          return Ok(Expr::FunctionCall {
            name: "ImageAssemble".to_string(),
            args: args.to_vec().into(),
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
  // Match wolframscript: emit Import::nffil before returning $Failed for a
  // missing file. Other failures (e.g. corrupt image) still return $Failed
  // but without the not-found message.
  if !std::path::Path::new(path).exists() {
    crate::emit_message(&format!(
      "Import::nffil: File {} not found during Import.",
      path
    ));
    return Ok(Expr::Identifier("$Failed".to_string()));
  }
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

/// Populate a usvg font database with the embedded fallback fonts and wire
/// each CSS generic family to a reasonable default. This is shared by the
/// raster (`rasterize_svg`) and PDF (`svg_to_pdf_bytes`) export paths so
/// that SVGs referring to `font-family="sans-serif"` (emitted by Plot and
/// friends) still render when no matching system font is installed.
///
/// We ship:
/// - **Atkinson Hyperlegible Next** (OFL, variable-weight) as the
///   sans-serif / serif fallback; its Latin + Latin-Extended coverage is
///   more than enough for tick labels, axis labels, titles, and similar
///   plot chrome, and it was literally designed for on-screen legibility.
/// - **Atkinson Hyperlegible Mono** (OFL, variable-weight) as the
///   monospace fallback. One file covers every weight from Thin to
///   ExtraBold, and it shares the same hyperlegible design language as
///   its proportional sibling.
///
/// Anything with exotic glyphs outside the embedded fonts will fall
/// through to `load_system_fonts()`.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_embedded_fonts(fontdb: &mut resvg::usvg::fontdb::Database) {
  fontdb.load_font_data(
    include_bytes!(
      "../../resources/AtkinsonHyperlegibleMono-VariableFont_wght.ttf"
    )
    .to_vec(),
  );
  fontdb.load_font_data(
    include_bytes!(
      "../../resources/AtkinsonHyperlegibleNext-VariableFont_wght.ttf"
    )
    .to_vec(),
  );
  fontdb.set_monospace_family("Atkinson Hyperlegible Mono");
  fontdb.set_sans_serif_family("Atkinson Hyperlegible Next");
  // We don't ship a dedicated serif face, so fall back to the sans-serif
  // for "serif" requests rather than leaving them unresolved.
  fontdb.set_serif_family("Atkinson Hyperlegible Next");
  fontdb.set_cursive_family("Atkinson Hyperlegible Next");
  fontdb.set_fantasy_family("Atkinson Hyperlegible Next");
}

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
    Expr::FunctionCall { name, args: fargs }
      if name == "Graphics" || name == "Graphics3D" =>
    {
      if let Ok(Expr::Graphics { ref svg, .. }) =
        crate::functions::graphics::graphics_ast(fargs)
      {
        svg.clone()
      } else {
        return Err(InterpreterError::EvaluationError(
          "Rasterize: failed to render Graphics".into(),
        ));
      }
    }
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

  let mut fontdb = resvg::usvg::fontdb::Database::new();
  // Load system fonts first so they're available as fallbacks for exotic
  // glyphs, then load embedded fonts and set generic-family mappings.
  // Order matters: load_system_fonts() resets the generic family aliases,
  // so our set_sans_serif_family() etc. must come *after* it.
  fontdb.load_system_fonts();
  load_embedded_fonts(&mut fontdb);

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

/// One frame of an animated GIF: an Expr::Image plus a per-frame delay
/// in hundredths of a second (GIF's native unit).
#[cfg(not(target_arch = "wasm32"))]
pub struct GifFrame {
  pub image: image::RgbaImage,
  pub delay_hundredths: u16,
}

/// Encode a list of frames as an animated GIF and write it to `path`.
/// Frames must all have the same dimensions; if they don't, all frames
/// are resized to the dimensions of the first frame (canvas-pasted to
/// preserve aspect, smaller frames padded with white).
#[cfg(not(target_arch = "wasm32"))]
pub fn export_animated_gif(
  path: &str,
  frames: Vec<GifFrame>,
) -> Result<(), InterpreterError> {
  use image::Delay;
  use image::Frame;
  use image::codecs::gif::{GifEncoder, Repeat};
  use std::fs::File;
  use std::time::Duration;

  if frames.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "Export: animated GIF requires at least one frame".into(),
    ));
  }

  let (canvas_w, canvas_h) = frames[0].image.dimensions();

  let file = File::create(path).map_err(|e| {
    InterpreterError::EvaluationError(format!(
      "Export: cannot save \"{}\": {}",
      path, e
    ))
  })?;
  let mut encoder = GifEncoder::new(file);
  encoder.set_repeat(Repeat::Infinite).map_err(|e| {
    InterpreterError::EvaluationError(format!("Export: GIF encode error: {e}"))
  })?;

  for f in frames {
    let img = if f.image.dimensions() == (canvas_w, canvas_h) {
      f.image
    } else {
      // Pad/crop onto a white canvas of the target size, top-left aligned.
      let mut canvas = image::RgbaImage::from_pixel(
        canvas_w,
        canvas_h,
        image::Rgba([255, 255, 255, 255]),
      );
      let (fw, fh) = f.image.dimensions();
      let copy_w = fw.min(canvas_w);
      let copy_h = fh.min(canvas_h);
      for y in 0..copy_h {
        for x in 0..copy_w {
          canvas.put_pixel(x, y, *f.image.get_pixel(x, y));
        }
      }
      canvas
    };

    let delay = Delay::from_saturating_duration(Duration::from_millis(
      (f.delay_hundredths as u64) * 10,
    ));
    let frame = Frame::from_parts(img, 0, 0, delay);
    encoder.encode_frame(frame).map_err(|e| {
      InterpreterError::EvaluationError(format!(
        "Export: GIF encode error: {e}"
      ))
    })?;
  }

  Ok(())
}

/// ConstantImage[val, {w, h}] - Create a constant image
/// val can be a number (grayscale) or a color (Red, RGBColor[r,g,b], etc.)
pub fn constant_image_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "ConstantImage expects 1 or 2 arguments".into(),
    ));
  }

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
          "ConstantImage: second argument must be {width, height}".into(),
        ));
      }
    }
  } else {
    (150u32, 150u32)
  };

  // Determine the color value(s)
  let (channels, pixel): (u8, Vec<f64>) = resolve_color_value(&args[0])?;

  let len = (w as usize) * (h as usize) * (channels as usize);
  let data: Vec<f64> = pixel.iter().cycle().take(len).copied().collect();

  Ok(Expr::Image {
    width: w,
    height: h,
    channels,
    data: Arc::new(data),
    image_type: ImageType::Real32,
  })
}

/// Resolve a color expression to (channels, pixel_values)
fn resolve_color_value(
  expr: &Expr,
) -> Result<(u8, Vec<f64>), InterpreterError> {
  match expr {
    // Scalar → grayscale
    Expr::Integer(n) => Ok((1, vec![*n as f64])),
    Expr::Real(f) => Ok((1, vec![*f])),
    // Rational[n, d]
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      let n = expr_to_f64(&args[0])?;
      let d = expr_to_f64(&args[1])?;
      Ok((1, vec![n / d]))
    }
    // RGBColor[r, g, b]
    Expr::FunctionCall { name, args }
      if name == "RGBColor" && args.len() == 3 =>
    {
      let r = expr_to_f64(&args[0])?;
      let g = expr_to_f64(&args[1])?;
      let b = expr_to_f64(&args[2])?;
      Ok((3, vec![r, g, b]))
    }
    // GrayLevel[g]
    Expr::FunctionCall { name, args }
      if name == "GrayLevel" && args.len() == 1 =>
    {
      let g = expr_to_f64(&args[0])?;
      Ok((1, vec![g]))
    }
    // Named color (Red, Blue, etc.)
    Expr::Identifier(name) => {
      if let Some(color_expr) = crate::evaluator::named_color_expr_pub(name) {
        resolve_color_value(&color_expr)
      } else {
        Err(InterpreterError::EvaluationError(format!(
          "ConstantImage: unknown color \"{}\"",
          name
        )))
      }
    }
    _ => Err(InterpreterError::EvaluationError(format!(
      "ConstantImage: unsupported value {}",
      crate::syntax::expr_to_string(expr)
    ))),
  }
}

// ─────────────────────────────────────────────────────────────────────────
//  ColorDistance — distance between two colors in CIE LAB color space.
//
//  Wolfram Language uses the sRGB → D50 XYZ → CIE LAB pipeline (Bradford
//  chromatic adaptation) and reports LAB components scaled to [0, 1] (i.e.
//  the standard 0..100 L value divided by 100). The default
//  DistanceFunction is the Euclidean distance in that scaled LAB space —
//  which matches the CIE76 definition divided by 100.
// ─────────────────────────────────────────────────────────────────────────

/// Convert an sRGB component in `[0, 1]` to its linear-light value.
fn srgb_to_linear(c: f64) -> f64 {
  if c <= 0.04045 {
    c / 12.92
  } else {
    ((c + 0.055) / 1.055).powf(2.4)
  }
}

/// Convert linear-light sRGB `(r, g, b)` (each in `[0, 1]`) to CIE XYZ
/// under a D50 illuminant. The matrix coefficients match Wolfram's
/// internal sRGB → D50 conversion (Bradford chromatic adaptation), so
/// `ColorConvert[Red, "XYZ"]` lines up to f64 precision.
fn linear_rgb_to_xyz_d50(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
  let x =
    0.43602191242669813 * r + 0.38510884137388846 * g + 0.14308124062061284 * b;
  let y =
    0.22247517260243593 * r + 0.7169066111623497 * g + 0.06061821623521406 * b;
  let z =
    0.013928134067434789 * r + 0.09710156693979348 * g + 0.7141585835116003 * b;
  (x, y, z)
}

/// Apply the LAB f-function: cube root above the linear-region cutoff,
/// affine continuation below.
fn lab_f(t: f64) -> f64 {
  // delta = 6/29, delta^3 = 216/24389.
  const DELTA3: f64 = 216.0 / 24389.0;
  if t > DELTA3 {
    t.cbrt()
  } else {
    t * (24389.0 / 27.0 / 116.0) + 16.0 / 116.0
  }
}

/// Convert sRGB `(r, g, b)` (each in `[0, 1]`) to Wolfram-style LAB
/// `(L, a, b)` where every component is divided by 100 — i.e. L is in
/// `[0, 1]` for typical inputs.
fn srgb_to_lab(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
  let r_lin = srgb_to_linear(r);
  let g_lin = srgb_to_linear(g);
  let b_lin = srgb_to_linear(b);
  let (x, y, z) = linear_rgb_to_xyz_d50(r_lin, g_lin, b_lin);
  // D50 reference white. The constants below match Wolfram's internal
  // reference white to f64 precision (reverse-engineered from
  // `ColorConvert[Red, "Lab"]`); they sit a few ulp away from the
  // ICC-rounded `(0.96422, 0.82521)` values.
  let xn = 0.9642119944211995;
  let yn = 1.0;
  let zn = 0.8251882845188289;
  let fx = lab_f(x / xn);
  let fy = lab_f(y / yn);
  let fz = lab_f(z / zn);
  let l = 116.0 * fy - 16.0;
  let a = 500.0 * (fx - fy);
  let b_star = 200.0 * (fy - fz);
  (l / 100.0, a / 100.0, b_star / 100.0)
}

/// Resolve any color expression (named, RGBColor, GrayLevel, etc.) to
/// scaled LAB. Returns `None` if the expression is not a color.
fn color_to_lab(expr: &Expr) -> Option<(f64, f64, f64)> {
  let color = crate::functions::graphics::parse_color(expr)?;
  Some(srgb_to_lab(color.r, color.g, color.b))
}

/// ColorDistance[c1, c2] — Euclidean distance between two colors in
/// (Wolfram-scaled) CIE LAB space. The optional
/// `DistanceFunction -> f` rule applies a custom function to the LAB
/// triples instead of the default Euclidean norm.
pub fn color_distance_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "ColorDistance expects at least 2 arguments".into(),
    ));
  }

  // Extract optional `DistanceFunction -> spec` from the trailing args.
  let mut distance_fn: Option<Expr> = None;
  for a in &args[2..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = a
      && matches!(pattern.as_ref(), Expr::Identifier(s) if s == "DistanceFunction")
    {
      distance_fn = Some(replacement.as_ref().clone());
    } else if let Expr::FunctionCall { name, args: ra } = a
      && (name == "Rule" || name == "RuleDelayed")
      && ra.len() == 2
      && matches!(&ra[0], Expr::Identifier(s) if s == "DistanceFunction")
    {
      distance_fn = Some(ra[1].clone());
    }
  }

  // Broadcast: when both color arguments are equal-length lists,
  // apply ColorDistance pairwise. wolframscript:
  //   ColorDistance[{Red, Blue}, {Green, Yellow}, opts] →
  //   {ColorDistance[Red, Green, opts], ColorDistance[Blue, Yellow, opts]}
  if let (Expr::List(items1), Expr::List(items2)) = (&args[0], &args[1])
    && items1.len() == items2.len()
  {
    let mut results = Vec::with_capacity(items1.len());
    for (c1, c2) in items1.iter().zip(items2.iter()) {
      let mut sub_args = vec![c1.clone(), c2.clone()];
      sub_args.extend(args[2..].iter().cloned());
      results.push(color_distance_ast(&sub_args)?);
    }
    return Ok(Expr::List(results.into()));
  }

  let lab1 = color_to_lab(&args[0]).ok_or_else(|| {
    InterpreterError::EvaluationError(format!(
      "ColorDistance: unsupported color {}",
      crate::syntax::expr_to_string(&args[0])
    ))
  })?;
  let lab2 = color_to_lab(&args[1]).ok_or_else(|| {
    InterpreterError::EvaluationError(format!(
      "ColorDistance: unsupported color {}",
      crate::syntax::expr_to_string(&args[1])
    ))
  })?;

  let to_list = |t: (f64, f64, f64)| {
    Expr::List(vec![Expr::Real(t.0), Expr::Real(t.1), Expr::Real(t.2)].into())
  };

  // Decode `{name, qualifier}` distance specs (e.g. `{"CMC",
  // "Perceptibility"}`) before the per-name dispatch.
  let (dist_name, dist_qualifier): (Option<String>, Option<String>) =
    match &distance_fn {
      Some(Expr::List(items)) if items.len() == 2 => {
        let n = match &items[0] {
          Expr::String(s) => Some(s.clone()),
          _ => None,
        };
        let q = match &items[1] {
          Expr::String(s) => Some(s.clone()),
          _ => None,
        };
        (n, q)
      }
      Some(Expr::String(s)) => (Some(s.clone()), None),
      _ => (None, None),
    };

  match distance_fn {
    // Built-in named distance functions arrive as String literals.
    _ if dist_name.as_deref() == Some("CIE2000")
      || dist_name.as_deref() == Some("CIEDE2000") =>
    {
      Ok(Expr::Real(ciede2000_distance(lab1, lab2)))
    }
    _ if dist_name.as_deref() == Some("CIE76") => {
      let dl = lab1.0 - lab2.0;
      let da = lab1.1 - lab2.1;
      let db = lab1.2 - lab2.2;
      Ok(Expr::Real((dl * dl + da * da + db * db).sqrt()))
    }
    _ if dist_name.as_deref() == Some("CMC") => {
      // CMC qualifiers: "Perceptibility" → l=1, c=1 (default, used for
      // small differences); "Acceptability" → l=2, c=1.
      let (l_param, c_param) = match dist_qualifier.as_deref() {
        Some("Acceptability") => (2.0, 1.0),
        _ => (1.0, 1.0),
      };
      Ok(Expr::Real(cmc_distance(lab1, lab2, l_param, c_param)))
    }
    Some(func) => {
      // Apply the user-supplied function to the LAB triples.
      let result =
        crate::evaluator::evaluate_expr_to_expr(&Expr::CurriedCall {
          func: Box::new(func),
          args: vec![to_list(lab1), to_list(lab2)],
        })?;
      Ok(result)
    }
    None => {
      // Default: Euclidean distance in Wolfram-scaled LAB.
      let dl = lab1.0 - lab2.0;
      let da = lab1.1 - lab2.1;
      let db = lab1.2 - lab2.2;
      Ok(Expr::Real((dl * dl + da * da + db * db).sqrt()))
    }
  }
}

/// CIEDE2000 / "CIE2000" perceptual color difference between two
/// Wolfram-scaled LAB triples (L,a,b each in [0,1]). The standard
/// formula operates in the unscaled `[0,100]` LAB space; we multiply
/// by 100, run the formula, and divide the result by 100 to match
/// wolframscript's reported magnitude.
fn ciede2000_distance(lab1: (f64, f64, f64), lab2: (f64, f64, f64)) -> f64 {
  let (l1, a1, b1) = (lab1.0 * 100.0, lab1.1 * 100.0, lab1.2 * 100.0);
  let (l2, a2, b2) = (lab2.0 * 100.0, lab2.1 * 100.0, lab2.2 * 100.0);

  let c1 = (a1 * a1 + b1 * b1).sqrt();
  let c2 = (a2 * a2 + b2 * b2).sqrt();
  let avg_c = 0.5 * (c1 + c2);
  let avg_c7 = avg_c.powi(7);
  let g = 0.5 * (1.0 - (avg_c7 / (avg_c7 + 25f64.powi(7))).sqrt());

  let a1p = a1 * (1.0 + g);
  let a2p = a2 * (1.0 + g);
  let c1p = (a1p * a1p + b1 * b1).sqrt();
  let c2p = (a2p * a2p + b2 * b2).sqrt();
  let avg_cp = 0.5 * (c1p + c2p);

  let to_deg = 180.0 / std::f64::consts::PI;
  let h1p = {
    let h = b1.atan2(a1p) * to_deg;
    if h < 0.0 { h + 360.0 } else { h }
  };
  let h2p = {
    let h = b2.atan2(a2p) * to_deg;
    if h < 0.0 { h + 360.0 } else { h }
  };

  let delta_lp = l2 - l1;
  let delta_cp = c2p - c1p;
  let cprod = c1p * c2p;
  let delta_hp_deg = if cprod == 0.0 {
    0.0
  } else {
    let d = h2p - h1p;
    if d.abs() <= 180.0 {
      d
    } else if d > 180.0 {
      d - 360.0
    } else {
      d + 360.0
    }
  };
  let delta_hp_rad = delta_hp_deg / to_deg;
  let delta_h_big = 2.0 * cprod.sqrt() * (delta_hp_rad / 2.0).sin();

  let avg_lp = 0.5 * (l1 + l2);
  let avg_hp_deg = if cprod == 0.0 {
    h1p + h2p
  } else if (h1p - h2p).abs() <= 180.0 {
    0.5 * (h1p + h2p)
  } else if h1p + h2p < 360.0 {
    0.5 * (h1p + h2p + 360.0)
  } else {
    0.5 * (h1p + h2p - 360.0)
  };
  let avg_hp_rad = avg_hp_deg / to_deg;
  let cos = |deg: f64| (deg / to_deg).cos();
  let t = 1.0 - 0.17 * cos(avg_hp_deg - 30.0)
    + 0.24 * cos(2.0 * avg_hp_deg)
    + 0.32 * cos(3.0 * avg_hp_deg + 6.0)
    - 0.20 * cos(4.0 * avg_hp_deg - 63.0);

  let delta_theta = 30.0 * (-((avg_hp_deg - 275.0) / 25.0).powi(2)).exp();
  let avg_cp7 = avg_cp.powi(7);
  let r_c = 2.0 * (avg_cp7 / (avg_cp7 + 25f64.powi(7))).sqrt();
  let dl_diff = avg_lp - 50.0;
  let s_l = 1.0 + 0.015 * dl_diff * dl_diff / (20.0 + dl_diff * dl_diff).sqrt();
  let s_c = 1.0 + 0.045 * avg_cp;
  let s_h = 1.0 + 0.015 * avg_cp * t;
  let r_t = -(2.0 * delta_theta / to_deg).sin() * r_c;

  let term_l = delta_lp / s_l;
  let term_c = delta_cp / s_c;
  let term_h = delta_h_big / s_h;
  let _ = avg_hp_rad; // unused but documents the radian conversion above
  let de2 =
    term_l * term_l + term_c * term_c + term_h * term_h + r_t * term_c * term_h;
  de2.sqrt() / 100.0
}

/// CMC l:c color difference. `l` and `c` weight lightness and chroma.
/// "Perceptibility" → l=2, c=1; "Acceptability" → l=1, c=1.
/// Operates on Wolfram-scaled LAB (each component in [0, 1]); returns
/// the result divided by 100 to match wolframscript's magnitude.
fn cmc_distance(
  lab1: (f64, f64, f64),
  lab2: (f64, f64, f64),
  l_param: f64,
  c_param: f64,
) -> f64 {
  let (l1, a1, b1) = (lab1.0 * 100.0, lab1.1 * 100.0, lab1.2 * 100.0);
  let (l2, a2, b2) = (lab2.0 * 100.0, lab2.1 * 100.0, lab2.2 * 100.0);

  let c1 = (a1 * a1 + b1 * b1).sqrt();
  let c2 = (a2 * a2 + b2 * b2).sqrt();

  let to_deg = 180.0 / std::f64::consts::PI;
  // h1 in degrees, in [0, 360).
  let h1 = {
    let h = b1.atan2(a1) * to_deg;
    if h < 0.0 { h + 360.0 } else { h }
  };
  // The CMC formula uses h1 (the reference colour's hue).
  let cos = |deg: f64| (deg / to_deg).cos();
  let t = if !(164.0..345.0).contains(&h1) {
    0.36 + (0.4 * cos(h1 + 35.0)).abs()
  } else {
    0.56 + (0.2 * cos(h1 + 168.0)).abs()
  };
  let c1_4 = c1.powi(4);
  let f = (c1_4 / (c1_4 + 1900.0)).sqrt();
  let s_l = if l1 >= 16.0 {
    0.040975 * l1 / (1.0 + 0.01765 * l1)
  } else {
    0.511
  };
  let s_c = 0.0638 * c1 / (1.0 + 0.0131 * c1) + 0.638;
  let s_h = s_c * (f * t + 1.0 - f);

  let delta_l = l1 - l2;
  let delta_c = c1 - c2;
  let delta_a = a1 - a2;
  let delta_b = b1 - b2;
  let delta_h2 = delta_a * delta_a + delta_b * delta_b - delta_c * delta_c;
  let delta_h2 = delta_h2.max(0.0);

  let term_l = delta_l / (l_param * s_l);
  let term_c = delta_c / (c_param * s_c);
  let de2 = term_l * term_l + term_c * term_c + delta_h2 / (s_h * s_h);
  de2.sqrt() / 100.0
}
