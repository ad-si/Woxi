use super::*;
use crate::functions::math_ast::rat_reduce;
use crate::syntax::ImageType;
use std::borrow::Cow;
use std::sync::Arc;

// ─── Helpers ────────────────────────────────────────────────────────────────

/// The image behind a `Texture[…]` argument: an image as is, or any other
/// expression (typically a `Graphics`, which `Texture` accepts) rasterized
/// to one. `None` when it does not evaluate to an image.
pub fn texture_source_image(arg: &Expr) -> Option<Expr> {
  if matches!(arg, Expr::Image { .. }) {
    return Some(arg.clone());
  }
  let value = crate::evaluator::evaluate_expr_to_expr(arg).ok()?;
  if matches!(value, Expr::Image { .. }) {
    return Some(value);
  }
  let call = call1("Rasterize", value);
  crate::evaluator::evaluate_expr_to_expr(&call)
    .ok()
    .filter(|v| matches!(v, Expr::Image { .. }))
}

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
    2 => {
      let buf: Vec<u8> = data
        .iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
      let la = image::GrayAlphaImage::from_raw(width, height, buf).unwrap();
      image::DynamicImage::ImageLumaA8(la)
    }
    // An image with more channels (a multispectral one) shows its first
    // three as red, green and blue.
    n => {
      let buf: Vec<u8> = data
        .chunks(n as usize)
        .flat_map(|px| px[..3].iter())
        .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
      let rgb = image::RgbImage::from_raw(width, height, buf).unwrap();
      image::DynamicImage::ImageRgb8(rgb)
    }
  }
}

/// Encode an Expr::Image's pixel data to in-memory file bytes in a raster
/// format. Unlike `export_image`, this writes to a buffer rather than a path,
/// so it works in the browser (WASM) where `Export` has no filesystem. The
/// `image` crate's encoders compile to `wasm32`; only the SVG rasterizer
/// (resvg) is native-only, so plots must be exported as SVG instead.
pub fn export_image_bytes(
  fmt: &str,
  width: u32,
  height: u32,
  channels: u8,
  data: &[f64],
) -> Result<Vec<u8>, InterpreterError> {
  use std::io::Cursor;
  let format = match fmt {
    "PNG" => image::ImageFormat::Png,
    "JPG" | "JPEG" => image::ImageFormat::Jpeg,
    "GIF" => image::ImageFormat::Gif,
    "BMP" => image::ImageFormat::Bmp,
    "TIF" | "TIFF" => image::ImageFormat::Tiff,
    _ => {
      return Err(InterpreterError::EvaluationError(format!(
        "Export: unsupported image format \"{fmt}\""
      )));
    }
  };
  let dyn_img = expr_to_dynamic_image(width, height, channels, data);
  // JPEG has no alpha channel; flatten to RGB so encoding a 4-channel image
  // doesn't fail.
  let dyn_img = if matches!(format, image::ImageFormat::Jpeg) {
    image::DynamicImage::ImageRgb8(dyn_img.to_rgb8())
  } else {
    dyn_img
  };
  let mut buf = Cursor::new(Vec::new());
  dyn_img.write_to(&mut buf, format).map_err(|e| {
    InterpreterError::EvaluationError(format!(
      "Export: image encode error: {e}"
    ))
  })?;
  Ok(buf.into_inner())
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
        color_space: None,
        display_size: None,
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
        color_space: None,
        display_size: None,
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
        color_space: None,
        display_size: None,
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
///
/// `display_size` is the `ImageSize` the image is shown at; the embedded
/// raster keeps its own pixel dimensions and is scaled without smoothing.
pub fn image_to_html_img(
  width: u32,
  height: u32,
  channels: u8,
  data: &[f64],
  display_size: Option<(u32, u32)>,
) -> String {
  let (shown_w, shown_h) = display_size.unwrap_or((width, height));
  let rendering = if display_size.is_some() {
    " style='image-rendering:pixelated'"
  } else {
    ""
  };
  let dyn_img = expr_to_dynamic_image(width, height, channels, data);
  let mut buf = Vec::new();
  dyn_img
    .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
    .expect("PNG encoding failed");
  let b64 =
    base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &buf);
  format!(
    "<svg xmlns='http://www.w3.org/2000/svg' width='{shown_w}' height='{shown_h}'>\
     <image href='data:image/png;base64,{b64}' width='{shown_w}' height='{shown_h}'{rendering}/>\
     </svg>"
  )
}

/// A standalone SVG *document* wrapping an image's raster as a base64 PNG.
///
/// Same body as [`image_to_html_img`], but prefixed with the XML declaration
/// that wolframscript emits for `Export[…, "SVG"]` / `ExportString[img,
/// "SVG"]`.  Use this for file/string export (where the SVG stands alone);
/// use [`image_to_html_img`] for inline HTML display, where a leading
/// `<?xml?>` processing instruction would be invalid.
pub fn image_to_svg_document(
  width: u32,
  height: u32,
  channels: u8,
  data: &[f64],
  display_size: Option<(u32, u32)>,
) -> String {
  format!(
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n{}",
    image_to_html_img(width, height, channels, data, display_size)
  )
}

// ─── Core functions (Phase 1) ──────────────────────────────────────────────

/// Image[data] / Image[data, type] / Image[data, type, opts…] — construct
/// an image from nested lists. The optional type argument is one of
/// "Bit", "Byte", "Bit16", "Real32", "Real64". Extra arguments are
/// treated as options like `ColorSpace -> ...` and `Interleaving -> ...`;
/// unknown options are accepted and currently ignored.
pub fn image_constructor_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let mut result = image_constructor_base(args)?;
  if let Expr::Image {
    width,
    height,
    display_size,
    ..
  } = &mut result
    && let Some(size) = image_size_option(&args[1..], *width, *height)
  {
    *display_size = Some(size);
  }
  Ok(result)
}

/// Resolve an `ImageSize -> n` / `ImageSize -> {w, h}` option into the pixel
/// size an image of `width` x `height` is displayed at.  A single number
/// fixes the width; a pair is a bounding box the image is scaled to fit
/// while keeping its aspect ratio (an `Automatic` side leaves the other
/// side to decide).
fn image_size_option(
  opts: &[Expr],
  width: u32,
  height: u32,
) -> Option<(u32, u32)> {
  if width == 0 || height == 0 {
    return None;
  }
  let number = |e: &Expr| match e {
    Expr::Integer(n) if *n > 0 => Some(*n as f64),
    Expr::Real(f) if *f > 0.0 => Some(*f),
    _ => None,
  };
  let value = opts.iter().find_map(|opt| match opt {
    Expr::Rule {
      pattern,
      replacement,
    } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "ImageSize") => {
      Some(replacement.as_ref())
    }
    _ => None,
  })?;
  let scale = match value {
    // `Automatic` leaves that side of the bounding box unconstrained, so
    // `{w, Automatic}` fixes the width and `{Automatic, h}` the height.
    Expr::List(dims) if dims.len() == 2 => {
      let is_auto =
        |e: &Expr| matches!(e, Expr::Identifier(n) if n == "Automatic");
      match (is_auto(&dims[0]), is_auto(&dims[1])) {
        (false, true) => number(&dims[0])? / width as f64,
        (true, false) => number(&dims[1])? / height as f64,
        _ => {
          let (bw, bh) = (number(&dims[0])?, number(&dims[1])?);
          (bw / width as f64).min(bh / height as f64)
        }
      }
    }
    other => number(other)? / width as f64,
  };
  Some((
    ((width as f64 * scale).round() as u32).max(1),
    ((height as f64 * scale).round() as u32).max(1),
  ))
}

fn image_constructor_base(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "Image expects at least 1 argument".into(),
    ));
  }
  let unevaluated = || unevaluated("Image", args);

  // Image[image] is idempotent: wolframscript returns the inner image
  // unchanged.
  if args.len() == 1
    && let Expr::Image { .. } = &args[0]
  {
    return Ok(args[0].clone());
  }

  // The second positional arg is a type tag (string or identifier).
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
      // "Real" is the legacy spelling of "Real64".
      "Real64" | "Real" => Some(ImageType::Real64),
      _ => None,
    }
  };

  let mut requested_type: Option<ImageType> = None;
  let mut opt_start = 1;
  if let Some(second) = args.get(1)
    && !matches!(second, Expr::Rule { .. } | Expr::RuleDelayed { .. })
  {
    if let Some(ty) = parse_type(second) {
      requested_type = Some(ty);
      opt_start = 2;
    } else {
      let shown = match second {
        Expr::String(s) => s.clone(),
        e => expr_to_string(e),
      };
      crate::emit_message(&format!(
        "Image::imgdtype: The specified data type {shown} should be \"Bit\", \"Byte\", \"Bit16\", \"Real32\" or \"Real64\"."
      ));
      return Ok(unevaluated());
    }
  }

  // `Interleaving -> False` gives the data channel-first, `{c, h, w}`:
  // transpose it to the interleaved `{h, w, c}` layout and build from that.
  let interleaving_false = |o: &Expr| {
    matches!(o, Expr::Rule { pattern, replacement }
      if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "Interleaving")
        && matches!(replacement.as_ref(), Expr::Identifier(b) if b == "False"))
  };
  // The planes may sit inside a NumericArray / RawArray wrapper, which is
  // kept around the transposed data.
  let planes = match &args[0] {
    Expr::FunctionCall { name, args: na }
      if name == "NumericArray" && !na.is_empty() =>
    {
      &na[0]
    }
    Expr::FunctionCall { name, args: ra }
      if name == "RawArray" && ra.len() >= 2 =>
    {
      &ra[1]
    }
    other => other,
  };
  if args[opt_start..].iter().any(interleaving_false)
    && matches!(planes, Expr::List(planes)
      if planes.iter().all(|p| matches!(p, Expr::List(rows)
        if rows.iter().all(|r| matches!(r, Expr::List(_))))))
  {
    let transposed = crate::evaluator::evaluate_expr_to_expr(&call(
      "Transpose",
      vec![
        planes.clone(),
        Expr::List(
          vec![Expr::Integer(3), Expr::Integer(1), Expr::Integer(2)].into(),
        ),
      ],
    ))?;
    let interleaved = match &args[0] {
      Expr::FunctionCall {
        name,
        args: wrapper,
      } if name == "NumericArray" => {
        let mut wrapper = wrapper.to_vec();
        wrapper[0] = transposed;
        call("NumericArray", wrapper)
      }
      Expr::FunctionCall {
        name,
        args: wrapper,
      } if name == "RawArray" => {
        let mut wrapper = wrapper.to_vec();
        wrapper[1] = transposed;
        call("RawArray", wrapper)
      }
      _ => transposed,
    };
    let mut new_args = vec![interleaved];
    new_args
      .extend(args[1..].iter().filter(|o| !interleaving_false(o)).cloned());
    return image_constructor_ast(&new_args);
  }

  // Treat the remaining args as options. Only Rule/RuleDelayed shapes are
  // accepted; anything else aborts to keep nonsense calls from sneaking
  // through.
  for opt in &args[opt_start..] {
    if !matches!(opt, Expr::Rule { .. } | Expr::RuleDelayed { .. }) {
      return Err(InterpreterError::EvaluationError(format!(
        "Image: extra argument is not a Rule option: {}",
        expr_to_string(opt)
      )));
    }
  }

  // `Image[Graphics[…]]` is the rasterized graphic; options such as
  // `ImageSize` only affect how it is displayed, not its pixel data.
  // An explicit `ImageSize -> {w, h}` pins both dimensions, so the graphic
  // is laid out at exactly that pixel size.
  #[cfg(not(target_arch = "wasm32"))]
  {
    let graphics_call = match &args[0] {
      Expr::FunctionCall { name, .. } if name == "Graphics" => {
        Some(args[0].clone())
      }
      Expr::Graphics {
        structure: Some(s),
        is_3d: false,
        ..
      } if matches!(s.as_ref(), Expr::FunctionCall { name, .. } if name == "Graphics") => {
        Some(s.as_ref().clone())
      }
      _ => None,
    };
    if let Some(call) = graphics_call {
      if let Some(size) = pixel_size_option(&args[opt_start..])
        && let Expr::FunctionCall { name, args: gargs } = &call
      {
        let mut gargs: Vec<Expr> = gargs
          .iter()
          .filter(|a| {
            !matches!(a, Expr::Rule { pattern, .. }
              if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "ImageSize"))
          })
          .cloned()
          .collect();
        gargs.push(rule_expr(id_expr("ImageSize"), size));
        let sized = Expr::FunctionCall {
          name: name.clone(),
          args: gargs.into(),
        };
        return rasterize_ast(&[sized]);
      }
      return rasterize_ast(&[call]);
    }
    if matches!(&args[0], Expr::Graphics { .. }) {
      return rasterize_ast(&args[..1]);
    }
  }

  // Image[image, type] re-quantizes the NORMALIZED data into the new
  // type (unlike raw list input, which is read on the type's own scale).
  if let Expr::Image {
    width,
    height,
    channels,
    ref data,
    color_space,
    ..
  } = args[0]
  {
    let ty = requested_type.unwrap_or(ImageType::Real32);
    let quant = match ty {
      ImageType::Bit => Some(1.0),
      ImageType::Byte => Some(255.0),
      ImageType::Bit16 => Some(65535.0),
      _ => None,
    };
    let new_data: Vec<f64> = match quant {
      Some(m) => data
        .iter()
        .map(|&v| (v * m).round_ties_even().clamp(0.0, m) / m)
        .collect(),
      None => data.as_ref().clone(),
    };
    return Ok(Expr::Image {
      color_space,
      display_size: None,
      width,
      height,
      channels,
      data: Arc::new(new_data),
      image_type: ty,
    });
  }

  // `Image[NumericArray[data, type]]` / `Image[RawArray[type, data]]` —
  // unwrap to the nested list and pick up the dtype if not already given.
  // RawArray (the legacy spelling embedded in `CompressedData` payloads,
  // e.g. from `Image[CompressedData["…"]]` in saved notebooks) puts the
  // type string first and the data second.
  let raw_arg = match &args[0] {
    Expr::FunctionCall {
      name: ra_name,
      args: ra_args,
    } if ra_name == "RawArray" && ra_args.len() >= 2 => {
      if requested_type.is_none() {
        let s = match &ra_args[0] {
          Expr::String(s) | Expr::Identifier(s) => Some(s.as_str()),
          _ => None,
        };
        requested_type = match s {
          Some("Bit") => Some(ImageType::Bit),
          Some("Byte" | "UnsignedInteger8") => Some(ImageType::Byte),
          Some("Bit16" | "UnsignedInteger16") => Some(ImageType::Bit16),
          Some("Real32") => Some(ImageType::Real32),
          Some("Real64") => Some(ImageType::Real64),
          _ => None,
        };
      }
      &ra_args[1]
    }
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
          Some("Bit") => Some(ImageType::Bit),
          Some("Byte" | "UnsignedInteger8") => Some(ImageType::Byte),
          Some("Bit16" | "UnsignedInteger16") => Some(ImageType::Bit16),
          Some("Real32") => Some(ImageType::Real32),
          Some("Real64") => Some(ImageType::Real64),
          _ => None,
        };
      }
      &na_args[0]
    }
    other => other,
  };

  // Parse a strictly rectangular rank-2 or rank-3 numeric array. Any
  // shape or value problem gets wolframscript's imgarray message and an
  // unevaluated echo.
  let invalid = || {
    crate::emit_message(&format!(
      "Image::imgarray: The specified argument {} should be an array of rank 2 or 3 with machine-sized numbers.",
      expr_to_string(&args[0])
    ));
  };
  let parsed = parse_image_array(raw_arg);
  let Some((width, height, channels, mut data)) = parsed else {
    invalid();
    return Ok(unevaluated());
  };

  // An explicit `ColorSpace` has to agree with how many channels the data
  // actually carries — one per component, plus an optional alpha channel.
  // wolframscript rejects the pair outright rather than reinterpreting the
  // data, so `Image[{{0, 128, 255}, {255, 128, 0}}, ColorSpace -> "RGB"]`
  // (a one-channel 3x2 image) is an error, not a 1x2 RGB image.
  let explicit_space = args[opt_start..].iter().find_map(|opt| match opt {
    Expr::Rule {
      pattern,
      replacement,
    } if matches!(pattern.as_ref(), Expr::Identifier(n) | Expr::String(n) if n == "ColorSpace") => {
      match replacement.as_ref() {
        Expr::String(s) | Expr::Identifier(s) => Some(s.clone()),
        _ => None,
      }
    }
    _ => None,
  });
  if let Some(space) = args[opt_start..].iter().find_map(|opt| match opt {
    Expr::Rule {
      pattern,
      replacement,
    } if matches!(pattern.as_ref(), Expr::Identifier(n) | Expr::String(n) if n == "ColorSpace") => {
      match replacement.as_ref() {
        Expr::String(s) | Expr::Identifier(s) => Some(s.clone()),
        _ => None,
      }
    }
    _ => None,
  }) && let Some(base) = color_space_components(&space)
    && channels != base
    && channels != base + 1
  {
    crate::emit_message(&format!(
      "Image::imgcsmis: The specified color space {space} and the number of channels {channels} are not compatible."
    ));
    return Ok(unevaluated());
  }

  // Integer-typed images read raw values on their own scale (a real 0.5
  // with "Byte" is the byte value 0.5, rounded half-even and clamped) and
  // store them normalized to [0, 1].
  let quant: Option<f64> = match requested_type {
    Some(ImageType::Bit) => Some(1.0),
    Some(ImageType::Byte) => Some(255.0),
    Some(ImageType::Bit16) => Some(65535.0),
    _ => None,
  };
  if let Some(m) = quant {
    for v in &mut data {
      *v = v.round_ties_even().clamp(0.0, m) / m;
    }
  }

  Ok(Expr::Image {
    // An explicit color space is remembered (ImageColorSpace, Options).
    color_space: explicit_space.as_deref().and_then(color_space_tag),
    display_size: None,
    width,
    height,
    channels,
    data: Arc::new(data),
    image_type: requested_type.unwrap_or(ImageType::Real32),
  })
}

/// How many colour components a named colour space has, ignoring alpha.
/// `None` for a name Woxi does not know, which then goes unchecked.
fn color_space_components(name: &str) -> Option<u8> {
  match name {
    "Grayscale" => Some(1),
    "RGB" | "HSB" | "LAB" | "LCH" | "LUV" | "XYZ" => Some(3),
    "CMYK" => Some(4),
    _ => None,
  }
}

/// Extract (width, height, channels, row-major interleaved data) from a
/// rectangular rank-2 or rank-3 nested list; None for anything malformed.
/// A rank-3 array with single-element pixels collapses to one channel.
fn parse_image_array(arg: &Expr) -> Option<(u32, u32, u8, Vec<f64>)> {
  let Expr::List(rows) = arg else { return None };
  if rows.is_empty() {
    return None;
  }
  let Expr::List(first_row) = &rows[0] else {
    return None;
  };
  if first_row.is_empty() {
    return None;
  }
  let width = first_row.len();
  let height = rows.len();
  let is_color = matches!(&first_row[0], Expr::List(_));
  let channels = if is_color {
    match &first_row[0] {
      Expr::List(pixel) if !pixel.is_empty() => pixel.len(),
      _ => return None,
    }
  } else {
    1
  };
  if channels > u8::MAX as usize {
    return None;
  }

  let mut data = Vec::with_capacity(width * height * channels);
  let leaf = |e: &Expr| -> Option<f64> {
    if matches!(e, Expr::List(_)) {
      return None;
    }
    let v = crate::functions::math_ast::try_eval_to_f64(e)?;
    v.is_finite().then_some(v)
  };
  for row in rows {
    let Expr::List(items) = row else { return None };
    if items.len() != width {
      return None;
    }
    for cell in items {
      if is_color {
        let Expr::List(pixel) = cell else { return None };
        if pixel.len() != channels {
          return None;
        }
        for v in pixel {
          data.push(leaf(v)?);
        }
      } else {
        data.push(leaf(cell)?);
      }
    }
  }
  Some((width as u32, height as u32, channels as u8, data))
}

/// Helper: extract f64 from Expr, resolving constants and arithmetic
fn expr_to_f64(expr: &Expr) -> Result<f64, InterpreterError> {
  crate::functions::math_ast::try_eval_to_f64(expr).ok_or_else(|| {
    InterpreterError::EvaluationError(format!(
      "Image: expected a number, got {}",
      expr_to_string(expr)
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
  Ok(bool_expr(is_image))
}

/// `Image3D[{img1, img2, …}]` stacks equally sized 2D images into a volume.
/// Returns `{width, height, depth}`, the channel count and the shared type.
fn image_stack_info(e: &Expr) -> Option<(Vec<i128>, usize, ImageType)> {
  let Expr::FunctionCall { name, args } = e else {
    return None;
  };
  if name != "Image3D" || args.is_empty() {
    return None;
  }
  let Expr::List(items) = &args[0] else {
    return None;
  };
  let Some(Expr::Image {
    width,
    height,
    channels,
    image_type,
    ..
  }) = items.first()
  else {
    return None;
  };
  let same = items.iter().all(|it| {
    matches!(it, Expr::Image { width: w, height: h, channels: c, .. }
      if w == width && h == height && c == channels)
  });
  same.then(|| {
    (
      vec![*width as i128, *height as i128, items.len() as i128],
      *channels as usize,
      *image_type,
    )
  })
}

/// The slices of a volume as 2D images, front to back: the images of a
/// stacked `Image3D[{img1, …}]`, or one `Image[plane, type]` per plane of
/// `Image3D[data]` / `Image3D[NumericArray[data, …], type]`. The canonical
/// view of a volume for `ImageData` and `Image3DSlices`.
pub fn image3d_slices(e: &Expr) -> Option<Vec<Expr>> {
  if image_stack_info(e).is_some() {
    let Expr::FunctionCall { args, .. } = e else {
      return None;
    };
    let Expr::List(items) = &args[0] else {
      return None;
    };
    return Some(items.iter().cloned().collect());
  }
  let t = image3d_type(e)?;
  let Expr::FunctionCall { args, .. } = e else {
    return None;
  };
  let data = match &args[0] {
    Expr::FunctionCall { name, args: na }
      if name == "NumericArray" && !na.is_empty() =>
    {
      &na[0]
    }
    other => other,
  };
  let Expr::List(planes) = data else {
    return None;
  };
  let tag = match t {
    ImageType::Bit => "Bit",
    ImageType::Byte => "Byte",
    ImageType::Bit16 => "Bit16",
    ImageType::Real32 => "Real32",
    ImageType::Real64 => "Real64",
  };
  planes
    .iter()
    .map(|plane| {
      crate::evaluator::evaluate_expr_to_expr(&call(
        "Image",
        vec![plane.clone(), str_expr(tag)],
      ))
      .ok()
    })
    .collect()
}

/// `Image3DSlices[vol]` — the slices of a volume; `Image3DSlices[vol, n]` the
/// `n`-th, `Image3DSlices[vol, {n1, …}]` a list of them.
pub fn image3d_slices_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Only slicing along the depth axis (direction 1) is supported.
  if args.len() == 3 && !matches!(args[2], Expr::Integer(1)) {
    return Ok(unevaluated("Image3DSlices", args));
  }
  let Some(slices) = args.first().and_then(image3d_slices) else {
    return Ok(unevaluated("Image3DSlices", args));
  };
  let len = slices.len() as i128;
  let pick = |n: &Expr| -> Option<Expr> {
    let Expr::Integer(k) = n else { return None };
    (1..=len)
      .contains(k)
      .then(|| slices[(*k - 1) as usize].clone())
  };
  let invalid = |spec: &Expr| {
    crate::emit_message(&format!(
      "Image3DSlices::slices: Expecting All, an integer value or a vector of \
       integer values between 1 and {len} instead of {}.",
      expr_to_string(spec)
    ));
    Ok(unevaluated("Image3DSlices", args))
  };
  match args.get(1) {
    None => Ok(Expr::List(slices.into())),
    Some(Expr::Identifier(a)) if a == "All" => Ok(Expr::List(slices.into())),
    Some(spec @ Expr::List(ns)) => {
      match ns.iter().map(pick).collect::<Option<Vec<_>>>() {
        Some(v) => Ok(Expr::List(v.into())),
        None => invalid(spec),
      }
    }
    Some(n) => match pick(n) {
      Some(s) => Ok(s),
      None => invalid(n),
    },
  }
}

/// Flatten a stacked `Image3D[{img1, img2, …}]` into a 2D picture for hosts
/// that can only show flat images: the mean of the slices inside `ClipRange`
/// along the depth axis. `ClipRange -> {{x0, x1}, {y0, y1}, {z0, z1}}` is in
/// voxel coordinates and `All` keeps an axis whole. Returns
/// `(width, height, channels, data)`.
pub fn image3d_stack_projection(e: &Expr) -> Option<(u32, u32, u8, Vec<f64>)> {
  let (dims, channels, _) = image_stack_info(e)?;
  let Expr::FunctionCall { args, .. } = e else {
    return None;
  };
  let Expr::List(slices) = &args[0] else {
    return None;
  };
  let (w, h, d) = (dims[0] as usize, dims[1] as usize, dims[2] as usize);
  let ch = channels;

  let mut clip: [(f64, f64); 3] = [
    (f64::NEG_INFINITY, f64::INFINITY),
    (f64::NEG_INFINITY, f64::INFINITY),
    (f64::NEG_INFINITY, f64::INFINITY),
  ];
  for opt in &args[1..] {
    let Expr::Rule {
      pattern,
      replacement,
    } = opt
    else {
      continue;
    };
    if !matches!(pattern.as_ref(), Expr::Identifier(n) if n == "ClipRange") {
      continue;
    }
    if let Expr::List(axes) = replacement.as_ref() {
      for (i, axis) in axes.iter().take(3).enumerate() {
        if let Expr::List(range) = axis
          && range.len() == 2
          && let (Some(lo), Some(hi)) = (
            crate::functions::math_ast::try_eval_to_f64(&range[0]),
            crate::functions::math_ast::try_eval_to_f64(&range[1]),
          )
        {
          clip[i] = (lo, hi);
        }
      }
    }
  }
  // Voxel `i` (1-based) is inside when it lies within the closed range.
  let inside = |axis: usize, i: usize| {
    let v = (i + 1) as f64;
    v >= clip[axis].0.floor().max(1.0) && v <= clip[axis].1.ceil()
  };

  let mut acc = vec![0.0; w * h * ch];
  let mut count = 0usize;
  for (z, slice) in slices.iter().enumerate().take(d) {
    if !inside(2, z) {
      continue;
    }
    let Expr::Image { data, .. } = slice else {
      return None;
    };
    count += 1;
    for y in 0..h {
      for x in 0..w {
        if inside(0, x) && inside(1, y) {
          for c in 0..ch {
            let k = (y * w + x) * ch + c;
            acc[k] += data.get(k).copied().unwrap_or(0.0);
          }
        }
      }
    }
  }
  if count > 0 {
    for v in &mut acc {
      *v /= count as f64;
    }
  }
  Some((w as u32, h as u32, ch as u8, acc))
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
  if image_stack_info(e).is_some() {
    return true;
  }
  let inner = match &args[0] {
    Expr::FunctionCall {
      name: na_name,
      args: na_args,
    } if na_name == "NumericArray" && !na_args.is_empty() => &na_args[0],
    other => other,
  };
  let rank = nested_list_rank(inner);
  matches!(rank, Some(3 | 4))
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
    expr_to_string(&args[0])
  ));
  Ok(unevaluated("ImageDimensions", args))
}

/// Extract `{width, height, depth}` for a valid Image3D[arg]. Returns
/// None if `arg` isn't a rank-3 or rank-4 nested list (peeling through
/// a NumericArray wrapper).
fn image3d_dimensions(e: &Expr) -> Option<Vec<i128>> {
  if let Some((dims, _, _)) = image_stack_info(e) {
    return Some(dims);
  }
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
    let (num, den) = rat_reduce(h, w);
    if den == 1 {
      Expr::Integer(num)
    } else {
      call("Rational", vec![Expr::Integer(num), Expr::Integer(den)])
    }
  }
  if let Expr::Image { width, height, .. } = &args[0] {
    let (w, h) = (*width as i128, *height as i128);
    if w == 0 {
      return Ok(unevaluated("ImageAspectRatio", args));
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
    expr_to_string(&args[0])
  ));
  Ok(unevaluated("ImageAspectRatio", args))
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
    expr_to_string(&args[0])
  ));
  Ok(unevaluated("ImageChannels", args))
}

/// Channel count for a valid Image3D[arg]; None for invalid shapes.
fn image3d_channels(e: &Expr) -> Option<usize> {
  if let Some((_, ch, _)) = image_stack_info(e) {
    return Some(ch);
  }
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
    return Ok(str_expr(type_to_str(image_type)));
  }
  if let Some(t) = image3d_type(&args[0]) {
    return Ok(str_expr(type_to_str(&t)));
  }
  crate::emit_message(&format!(
    "ImageType::imginv: Expecting an image or graphics instead of {}.",
    expr_to_string(&args[0])
  ));
  Ok(unevaluated("ImageType", args))
}

/// Resolve the underlying type tag for a valid Image3D[arg]. Defaults
/// to Real32 if there's no NumericArray wrapper.
fn image3d_type(e: &Expr) -> Option<ImageType> {
  if let Some((_, _, t)) = image_stack_info(e) {
    return Some(t);
  }
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
  // `Image3D[data, "Byte"]`: a positional type wins over the array's.
  let type_tag = match args.get(1) {
    Some(Expr::String(t)) => Some(t.as_str()),
    _ => type_tag,
  };
  Some(match type_tag {
    Some("Byte" | "UnsignedInteger8") => ImageType::Byte,
    Some("Bit") => ImageType::Bit,
    Some("Bit16" | "UnsignedInteger16") => ImageType::Bit16,
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
  let coerced;
  let args: &[Expr] = if !matches!(&args[0], Expr::Image { .. })
    && let Some(img) = coerce_graphics_to_image(&args[0])
  {
    let mut v = args.to_vec();
    v[0] = img;
    coerced = v;
    &coerced
  } else {
    args
  };
  // A volume: the data of each slice, front to back.
  if !matches!(&args[0], Expr::Image { .. })
    && let Some(slices) = image3d_slices(&args[0])
  {
    let mut out = Vec::with_capacity(slices.len());
    for slice in slices {
      let mut a = vec![slice];
      a.extend(args[1..].iter().cloned());
      out.push(image_data_ast(&a)?);
    }
    return Ok(Expr::List(out.into()));
  }
  let requested_type: Option<&str> = if args.len() == 2 {
    match &args[1] {
      Expr::String(s) => Some(s.as_str()),
      _ => None,
    }
  } else {
    None
  };
  if let Expr::Image {
    color_space: _,
    display_size: _,
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
  {
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
        ImageType::Bit => Expr::Integer(v.round() as i128),
        ImageType::Real32 => Expr::Real((v as f32) as f64),
        _ => Expr::Real(v),
      }
    };

    let mut rows = Vec::with_capacity(h);
    for y in 0..h {
      if ch == 1 {
        // Grayscale: {{v, v, ...}, ...}
        let row: Vec<Expr> = (0..w).map(|x| to_expr(data[y * w + x])).collect();
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
  } else {
    crate::emit_message(&format!(
      "ImageData::imginv: Expecting an image or graphics instead of {}.",
      expr_to_string(&args[0])
    ));
    Ok(unevaluated("ImageData", args))
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
  // before inspecting the target value.
  let Expr::Image {
    width,
    height,
    channels,
    data,
    ..
  } = &args[0]
  else {
    crate::emit_message(&format!(
      "PixelValuePositions::imginv: Expecting an image or graphics instead of {}.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated("PixelValuePositions", args));
  };

  let ch = *channels as usize;

  // Parse the target value: scalar for grayscale, list for multi-channel.
  let target: Vec<f64> = match &args[1] {
    Expr::List(items) => {
      let mut vs = Vec::with_capacity(items.len());
      for it in items {
        match crate::functions::math_ast::try_eval_to_f64(it) {
          Some(v) => vs.push(v),
          None => {
            return Ok(unevaluated("PixelValuePositions", args));
          }
        }
      }
      vs
    }
    other => match crate::functions::math_ast::try_eval_to_f64(other) {
      Some(v) => vec![v],
      None => {
        return Ok(unevaluated("PixelValuePositions", args));
      }
    },
  };

  // Target rank must match channel count.
  if target.len() != ch {
    return Ok(Expr::List(Vec::new().into()));
  }

  let tol = if args.len() == 3 {
    match crate::functions::math_ast::try_eval_to_f64(&args[2]) {
      Some(v) => v,
      None => {
        return Ok(unevaluated("PixelValuePositions", args));
      }
    }
  } else {
    0.0
  };

  let w = *width as usize;
  let h = *height as usize;
  let mut positions: Vec<Expr> = Vec::new();
  for y in 0..h {
    for x in 0..w {
      let base = (y * w + x) * ch;
      // L∞ (Chebyshev) distance: max channel-wise absolute difference.
      let max_diff = (0..ch)
        .map(|c| (data[base + c] - target[c]).abs())
        .fold(0.0_f64, f64::max);
      if max_diff <= tol {
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

/// ImageColorSpace[img] - "Grayscale" or "RGB"
pub fn image_color_space_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ImageColorSpace expects exactly 1 argument".into(),
    ));
  }
  // wolframscript treats both Image and Image3D as valid inputs and
  // returns Automatic when no explicit colour space is set on the
  // image (the common case for inputs built from raw NumericArrays).
  if let Expr::Image { color_space, .. } = &args[0] {
    return Ok(match color_space {
      Some(cs) => str_expr(cs),
      None => id_expr("Automatic"),
    });
  }
  if is_valid_image3d(&args[0]) {
    // An explicit ColorSpace option, else the slices' colour space.
    if let Expr::FunctionCall { args: a, .. } = &args[0] {
      for opt in a.iter().skip(1) {
        if let Expr::Rule {
          pattern,
          replacement,
        } = opt
          && matches!(pattern.as_ref(), Expr::Identifier(n) if n == "ColorSpace")
        {
          return Ok(replacement.as_ref().clone());
        }
      }
    }
    if image_stack_info(&args[0]).is_some()
      && let Some(Expr::Image {
        color_space: Some(cs),
        ..
      }) = image3d_slices(&args[0]).and_then(|s| s.into_iter().next())
    {
      return Ok(str_expr(cs));
    }
    return Ok(id_expr("Automatic"));
  }
  // Matches wolframscript: emit ImageColorSpace::imginv and return
  // unevaluated instead of erroring out.
  crate::emit_message(&format!(
    "ImageColorSpace::imginv: Expecting an image or graphics instead of {}.",
    expr_to_string(&args[0])
  ));
  Ok(unevaluated("ImageColorSpace", args))
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
      color_space: _,
      display_size: _,
      width,
      height,
      channels,
      data,
      image_type,
    } => {
      let ch = *channels as usize;
      let is_real32 = matches!(image_type, ImageType::Real32);
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
        color_space: None,
        display_size: None,
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
        // ColorNegate numericizes components: an exact 1 negates to the
        // machine real `0.`, matching wolframscript's RGBColor[0., 1., 1.].
        if let Expr::Real(r) = e {
          Ok(Expr::Real(1.0 - r))
        } else {
          let v = expr_to_f64(e)?;
          Ok(Expr::Real(1.0 - v))
        }
      }
      match cargs.len() {
        3 => Ok(call(
          "RGBColor",
          vec![
            negate_component(&cargs[0])?,
            negate_component(&cargs[1])?,
            negate_component(&cargs[2])?,
          ],
        )),
        4 => Ok(call(
          "RGBColor",
          vec![
            negate_component(&cargs[0])?,
            negate_component(&cargs[1])?,
            negate_component(&cargs[2])?,
            cargs[3].clone(),
          ],
        )),
        _ => Err(InterpreterError::EvaluationError(
          "ColorNegate: RGBColor must have 3 or 4 arguments".into(),
        )),
      }
    }
    Expr::FunctionCall { name, args: cargs } if name == "GrayLevel" => {
      // GrayLevel[v] → GrayLevel[1-v], alpha preserved if present
      fn negate_component(e: &Expr) -> Result<Expr, InterpreterError> {
        // ColorNegate numericizes components: an exact 1 negates to the
        // machine real `0.`, matching wolframscript's RGBColor[0., 1., 1.].
        if let Expr::Real(r) = e {
          Ok(Expr::Real(1.0 - r))
        } else {
          let v = expr_to_f64(e)?;
          Ok(Expr::Real(1.0 - v))
        }
      }
      match cargs.len() {
        1 => Ok(call1("GrayLevel", negate_component(&cargs[0])?)),
        2 => Ok(call(
          "GrayLevel",
          vec![negate_component(&cargs[0])?, cargs[1].clone()],
        )),
        _ => Err(InterpreterError::EvaluationError(
          "ColorNegate: GrayLevel must have 1 or 2 arguments".into(),
        )),
      }
    }
    // Hue and CMYKColor negate via RGB but stay in their own color space:
    // negate the RGB components, then convert back. (Lighter/Darker, by
    // contrast, return RGBColor.)
    Expr::FunctionCall { name, args: cargs }
      if (name == "Hue"
        && (cargs.len() == 1 || cargs.len() == 3 || cargs.len() == 4))
        || (name == "CMYKColor" && (cargs.len() == 4 || cargs.len() == 5)) =>
    {
      let Some((r, g, b, alpha)) = color_directive_to_rgb(&args[0]) else {
        // Symbolic components: leave unevaluated.
        return Ok(unevaluated("ColorNegate", args));
      };
      let (nr, ng, nb) = (1.0 - r, 1.0 - g, 1.0 - b);
      let mut out = if name == "Hue" {
        let (h, s, v) = rgb_to_hsv(nr, ng, nb);
        vec![Expr::Real(h), Expr::Real(s), Expr::Real(v)]
      } else {
        let (c, m, y, k) = rgb_to_cmyk(nr, ng, nb);
        vec![Expr::Real(c), Expr::Real(m), Expr::Real(y), Expr::Real(k)]
      };
      if let Some(a) = alpha {
        out.push(Expr::Real(a));
      }
      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: out.into(),
      })
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
        expr_to_string(other)
      ));
      Ok(unevaluated("ColorNegate", args))
    }
  }
}

/// Binarize[img], Binarize[img, threshold] or Binarize[img, Method -> m]
///
/// `threshold` is a number (strict `>`), a pair `{t1, t2}` (inclusive range)
/// or a function applied to each pixel value that returns `True` / `False`.
/// `Method` selects an automatic threshold: `"Cluster"`, `"Mean"`,
/// `"Median"`, `"Entropy"`, `"MinimumError"` or `{"BlackFraction", f}`.
pub fn binarize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Binarize expects 1 to 3 arguments".into(),
    ));
  }

  // wolframscript validates the image before the threshold, so a chain
  // like `Binarize[img, Out[0]]` (where `img` is undefined) emits
  // Binarize::imginv rather than choking on the non-numeric threshold.
  if !matches!(&args[0], Expr::Image { .. }) {
    crate::emit_message(&format!(
      "Binarize::imginv: Expecting an image or graphics instead of {}.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated("Binarize", args));
  }

  // The 1-arg form uses an automatic threshold (approximated here as 0.5
  // with the legacy >= compare so existing snapshots still pass;
  // wolframscript uses Otsu's method).
  enum Threshold<'a> {
    Default,
    Single(f64),
    Range(f64, f64),
    Function(&'a Expr),
    Method(BinarizeMethod),
  }
  let (options, positional): (Vec<&Expr>, Vec<&Expr>) = args[1..]
    .iter()
    .partition(|a| matches!(a, Expr::Rule { .. }));
  if positional.len() > 1 {
    return Ok(unevaluated("Binarize", args));
  }
  let method = options.iter().find_map(|opt| match opt {
    Expr::Rule {
      pattern,
      replacement,
    } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "Method") => {
      Some(replacement.as_ref())
    }
    _ => None,
  });
  let mut threshold = match positional.first() {
    None => Threshold::Default,
    Some(Expr::List(items)) if items.len() == 2 => {
      Threshold::Range(expr_to_f64(&items[0])?, expr_to_f64(&items[1])?)
    }
    Some(f @ (Expr::Function { .. } | Expr::NamedFunction { .. })) => {
      Threshold::Function(f)
    }
    Some(t) => Threshold::Single(expr_to_f64(t)?),
  };
  if let (Threshold::Default, Some(m)) = (&threshold, method) {
    match BinarizeMethod::parse(m) {
      Some(m) => threshold = Threshold::Method(m),
      None => return Ok(unevaluated("Binarize", args)),
    }
  }

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
      let lums: Vec<f64> = (0..w * h)
        .map(|i| {
          if ch == 1 {
            data[i]
          } else {
            let base = i * ch;
            // Luminance: 0.299*R + 0.587*G + 0.114*B
            0.299 * data[base] + 0.587 * data[base + 1] + 0.114 * data[base + 2]
          }
        })
        .collect();
      let method_threshold = match &threshold {
        Threshold::Method(m) => m.threshold(&lums),
        _ => 0.5,
      };
      let mut new_data = Vec::with_capacity(w * h);
      for (i, &lum) in lums.iter().enumerate() {
        let bit = match &threshold {
          Threshold::Default => lum >= 0.5,
          Threshold::Single(t) => lum > *t,
          Threshold::Range(t1, t2) => lum >= *t1 && lum <= *t2,
          Threshold::Method(_) => lum > method_threshold,
          Threshold::Function(f) => {
            // The function sees the pixel value: a number for a
            // single-channel image, otherwise the color channels.
            let pixel = if ch == 1 {
              Expr::Real(data[i])
            } else {
              let base = i * ch;
              Expr::List(
                data[base..base + ch.min(3)]
                  .iter()
                  .map(|v| Expr::Real(*v))
                  .collect(),
              )
            };
            let result =
              crate::evaluator::function_application::apply_function_to_arg(
                f, &pixel,
              )?;
            matches!(&result, Expr::Identifier(n) if n == "True")
          }
        };
        new_data.push(if bit { 1.0 } else { 0.0 });
      }

      Ok(Expr::Image {
        color_space: None,
        display_size: None,
        width: *width,
        height: *height,
        channels: 1,
        data: Arc::new(new_data),
        image_type: ImageType::Bit,
      })
    }
    _ => Err(InterpreterError::EvaluationError(
      "Binarize: first argument is not an Image".into(),
    )),
  }
}

/// The automatic threshold selectors `Binarize` accepts as `Method`.
enum BinarizeMethod {
  Cluster,
  Mean,
  Median,
  Entropy,
  MinimumError,
  BlackFraction(f64),
}

impl BinarizeMethod {
  fn parse(spec: &Expr) -> Option<Self> {
    match spec {
      Expr::String(s) => match s.as_str() {
        "Cluster" => Some(Self::Cluster),
        "Mean" => Some(Self::Mean),
        "Median" => Some(Self::Median),
        "Entropy" => Some(Self::Entropy),
        "MinimumError" => Some(Self::MinimumError),
        _ => None,
      },
      Expr::List(items) if items.len() == 2 => match (&items[0], &items[1]) {
        (Expr::String(s), f) if s == "BlackFraction" => {
          let f = expr_to_f64(f).ok()?;
          (0.0..=1.0).contains(&f).then_some(Self::BlackFraction(f))
        }
        _ => None,
      },
      _ => None,
    }
  }

  /// The gray level at or below which pixels become black.
  fn threshold(&self, values: &[f64]) -> f64 {
    if values.is_empty() {
      return 0.5;
    }
    let sorted = || {
      let mut sorted = values.to_vec();
      sorted.sort_by(f64::total_cmp);
      sorted
    };
    match self {
      Self::Cluster => cluster_variance_threshold(values),
      Self::Mean => values.iter().sum::<f64>() / values.len() as f64,
      Self::Median => sorted()[values.len() / 2],
      Self::BlackFraction(f) => {
        let black = (f * values.len() as f64).round() as usize;
        match black {
          0 => -1.0,
          n if n >= values.len() => 2.0,
          n => sorted()[n - 1],
        }
      }
      Self::Entropy | Self::MinimumError => {
        let histogram = gray_histogram(values);
        let bin = if matches!(self, Self::Entropy) {
          max_entropy_bin(&histogram)
        } else {
          minimum_error_bin(&histogram)
        };
        (bin as f64 + 1.0) / histogram.len() as f64
      }
    }
  }
}

/// A 256-bin histogram of values in `[0, 1]`, normalised to sum to 1.
fn gray_histogram(values: &[f64]) -> Vec<f64> {
  const BINS: usize = 256;
  let mut histogram = vec![0.0; BINS];
  for &v in values {
    let bin = ((v.clamp(0.0, 1.0) * BINS as f64) as usize).min(BINS - 1);
    histogram[bin] += 1.0;
  }
  let total = values.len() as f64;
  for c in &mut histogram {
    *c /= total;
  }
  histogram
}

/// Kapur's method: the split maximising the summed entropy of the two
/// classes. Returns the last bin that is black.
fn max_entropy_bin(histogram: &[f64]) -> usize {
  let entropy = |range: std::ops::Range<usize>| -> f64 {
    let mass: f64 = histogram[range.clone()].iter().sum();
    if mass <= 0.0 {
      return 0.0;
    }
    -histogram[range]
      .iter()
      .filter(|&&p| p > 0.0)
      .map(|&p| (p / mass) * (p / mass).ln())
      .sum::<f64>()
  };
  (0..histogram.len() - 1)
    .map(|t| (t, entropy(0..t + 1) + entropy(t + 1..histogram.len())))
    .max_by(|a, b| a.1.total_cmp(&b.1))
    .map_or(histogram.len() / 2, |(t, _)| t)
}

/// Kittler–Illingworth minimum error thresholding: fit two Gaussians to
/// the classes either side of each split and minimise the criterion.
/// Returns the last bin that is black.
fn minimum_error_bin(histogram: &[f64]) -> usize {
  let stats = |range: std::ops::Range<usize>| -> Option<(f64, f64)> {
    let mass: f64 = histogram[range.clone()].iter().sum();
    if mass <= 0.0 {
      return None;
    }
    let mean =
      range.clone().map(|i| i as f64 * histogram[i]).sum::<f64>() / mass;
    let var = range
      .map(|i| (i as f64 - mean).powi(2) * histogram[i])
      .sum::<f64>()
      / mass;
    Some((mass, var.max(1e-9)))
  };
  (0..histogram.len() - 1)
    .filter_map(|t| {
      let (p1, v1) = stats(0..t + 1)?;
      let (p2, v2) = stats(t + 1..histogram.len())?;
      let j = 1.0 + 2.0 * (p1 * v1.sqrt().ln() + p2 * v2.sqrt().ln())
        - 2.0 * (p1 * p1.ln() + p2 * p2.ln());
      Some((t, j))
    })
    .min_by(|a, b| a.1.total_cmp(&b.1))
    .map_or(histogram.len() / 2, |(t, _)| t)
}

/// Blur[image] / Blur[image, r] — blur an image over pixel radius `r`
/// (default 2). This is exactly `GaussianFilter[image, r]` with the
/// standard deviation left at its default `r/2`, so the kernel and the
/// convolution are shared with `GaussianFilter` rather than duplicated.
/// `Blur` differs in its radius specification, which may be a pair
/// `{r_rows, r_columns}`, in capping that radius at half the image (see
/// `blur_radius_axes`), and in keeping an integer image integral.
pub fn blur_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Blur expects 1 or 2 arguments".into(),
    ));
  }
  let Some((vertical, horizontal)) = blur_radius_axes("Blur", args) else {
    return Ok(unevaluated("Blur", args));
  };
  Ok(requantize_filtered(
    gaussian_filter_image(&args[0], vertical, horizontal)
      .unwrap_or_else(|| args[0].clone()),
  ))
}

/// Sharpen[image] / Sharpen[image, r] — sharpen an image over pixel
/// radius `r` (default 2). This is the unsharp mask
/// `image + 2 (image - Blur[image, r])`, so it shares `Blur`'s kernel,
/// radius specification and half-the-image cap; only the combination
/// afterwards is its own.
pub fn sharpen_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Sharpen expects 1 or 2 arguments".into(),
    ));
  }
  let Some((vertical, horizontal)) = blur_radius_axes("Sharpen", args) else {
    return Ok(unevaluated("Sharpen", args));
  };
  let Some(blurred) = gaussian_filter_image(&args[0], vertical, horizontal)
  else {
    return Ok(args[0].clone());
  };
  let (
    Expr::Image {
      color_space,
      display_size: _,
      width,
      height,
      channels,
      data,
      image_type,
    },
    Expr::Image { data: blurred, .. },
  ) = (&args[0], &blurred)
  else {
    unreachable!("blur_radius_axes checked for an image")
  };
  let sharpened = data
    .iter()
    .zip(blurred.iter())
    .map(|(original, blurred)| 3.0 * original - 2.0 * blurred)
    .collect();
  Ok(requantize_filtered(Expr::Image {
    color_space: *color_space,
    display_size: None,
    width: *width,
    height: *height,
    channels: *channels,
    data: Arc::new(sharpened),
    image_type: *image_type,
  }))
}

/// ImageEffect[image, "effect"] / ImageEffect[image, {"effect", params…}] —
/// apply an image effect. The noise effects draw from the shared
/// interpreter RNG, so `SeedRandom` makes them reproducible:
///
/// - `"SaltPepperNoise"` (density `d`, default 0.05): each pixel is
///   replaced, with probability `d`, by pure black or pure white (equal
///   odds). An alpha channel is left untouched.
/// - `"GaussianNoise"` (standard deviation `σ`, default 0.1): adds
///   zero-mean Gaussian noise to every color channel. The result is *not*
///   clipped to [0, 1] — wolframscript lets a noisy real image run out of
///   the unit range, and clipping would bias the noise at both ends.
/// - `"Noise"` (amplitude `a`, default 0.1): adds zero-mean uniform noise
///   drawn from `[-a, a]` to every color channel, also unclipped.
///
/// - `"Solarization"` (threshold `t`, default 0.5): color values above `t`
///   are inverted, the rest are kept.
/// - `"OilPainting"` (radius `r`, default 2): each pixel takes the mean
///   color of the most frequent intensity level in its `(2r+1)²` window.
/// - `"Charcoal"` (radius `r`, default 1): dark strokes on a white ground,
///   drawn where the intensity changes over a distance of `r` pixels.
/// - `"MotionBlur"` (radius `r`, default 5; angle `θ` in degrees, default
///   0): averages along a line of length `2r + 1` in direction `θ`.
///
/// Other effects are not implemented and return the call unevaluated.
pub fn image_effect_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use rand::RngExt;

  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(format!(
      "ImageEffect expects 2 arguments, got {}",
      args.len()
    )));
  }
  let Expr::Image {
    color_space,
    display_size: _,
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
  else {
    crate::emit_message(&format!(
      "ImageEffect::imginv: Expecting an image or graphics instead of {}.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated("ImageEffect", args));
  };

  // The effect spec is either a bare name or `{name, params…}`.
  let (effect, params): (String, &[Expr]) = match &args[1] {
    Expr::String(name) => (name.clone(), &[]),
    Expr::List(items) if !items.is_empty() => match &items[0] {
      Expr::String(name) => (name.clone(), &items[1..]),
      _ => return Ok(unevaluated("ImageEffect", args)),
    },
    _ => return Ok(unevaluated("ImageEffect", args)),
  };
  let numeric_param = |idx: usize, default: f64| -> Option<f64> {
    match params.get(idx) {
      Some(expr) => crate::functions::math_ast::try_eval_to_f64(expr),
      None => Some(default),
    }
  };

  // Noise touches the color channels only; a trailing alpha channel (2 for
  // grayscale + alpha, 4 for RGBA) keeps its original values.
  let ch = *channels as usize;
  let color_channels = match ch {
    2 => 1,
    4 => 3,
    n => n,
  };

  let new_data: Vec<f64> = match effect.as_str() {
    "SaltPepperNoise" => {
      let Some(density) = numeric_param(0, 0.05) else {
        return Ok(unevaluated("ImageEffect", args));
      };
      let mut out = data.as_ref().clone();
      crate::with_rng(|rng| {
        for pixel in out.chunks_mut(ch) {
          if rng.random_range(0.0..1.0) < density {
            let value = if rng.random_range(0.0..1.0) < 0.5 {
              0.0
            } else {
              1.0
            };
            for channel in pixel.iter_mut().take(color_channels) {
              *channel = value;
            }
          }
        }
      });
      out
    }
    "GaussianNoise" => {
      let Some(sigma) = numeric_param(0, 0.1) else {
        return Ok(unevaluated("ImageEffect", args));
      };
      let mut out = data.as_ref().clone();
      crate::with_rng(|rng| {
        let normal = rand_distr::Normal::new(0.0f64, sigma.abs())
          .unwrap_or(rand_distr::Normal::new(0.0f64, 0.0).unwrap());
        for pixel in out.chunks_mut(ch) {
          for channel in pixel.iter_mut().take(color_channels) {
            *channel += rng.sample(normal);
          }
        }
      });
      out
    }
    // `{"Noise", a}`: zero-mean uniform noise of amplitude `a`, i.e. each
    // channel gets an independent draw from `[-a, a]` added to it. Not
    // clipped to [0, 1], matching `"GaussianNoise"` above.
    "Noise" => {
      let Some(amplitude) = numeric_param(0, 0.1) else {
        return Ok(unevaluated("ImageEffect", args));
      };
      let mut out = data.as_ref().clone();
      crate::with_rng(|rng| {
        for pixel in out.chunks_mut(ch) {
          for channel in pixel.iter_mut().take(color_channels) {
            *channel += rng.random_range(-amplitude.abs()..=amplitude.abs());
          }
        }
      });
      out
    }
    "Solarization" => {
      let Some(threshold) = numeric_param(0, 0.5) else {
        return Ok(unevaluated("ImageEffect", args));
      };
      let mut out = data.as_ref().clone();
      for pixel in out.chunks_mut(ch) {
        for channel in pixel.iter_mut().take(color_channels) {
          if *channel > threshold {
            *channel = 1.0 - *channel;
          }
        }
      }
      out
    }
    "OilPainting" | "Charcoal" => {
      let default_radius = if effect == "OilPainting" { 2.0 } else { 1.0 };
      let Some(radius) = numeric_param(0, default_radius) else {
        return Ok(unevaluated("ImageEffect", args));
      };
      let radius = radius.round().max(0.0) as i64;
      let (w, h) = (*width as i64, *height as i64);
      let intensity = |x: i64, y: i64| -> f64 {
        let base = ((y.clamp(0, h - 1) * w + x.clamp(0, w - 1)) as usize) * ch;
        data[base..base + color_channels].iter().sum::<f64>()
          / color_channels as f64
      };
      let mut out = data.as_ref().clone();
      for y in 0..h {
        for x in 0..w {
          let base = ((y * w + x) as usize) * ch;
          if effect == "Charcoal" {
            let dx = intensity(x + radius, y) - intensity(x - radius, y);
            let dy = intensity(x, y + radius) - intensity(x, y - radius);
            let value = 1.0 - ((dx.abs() + dy.abs()) * 2.0).min(1.0);
            for channel in &mut out[base..base + color_channels] {
              *channel = value;
            }
          } else {
            const LEVELS: usize = 20;
            let mut count = [0usize; LEVELS];
            let mut sums = vec![vec![0.0; color_channels]; LEVELS];
            for ny in (y - radius).max(0)..=(y + radius).min(h - 1) {
              for nx in (x - radius).max(0)..=(x + radius).min(w - 1) {
                let level = ((intensity(nx, ny) * LEVELS as f64) as usize)
                  .min(LEVELS - 1);
                let nbase = ((ny * w + nx) as usize) * ch;
                count[level] += 1;
                for (sum, value) in sums[level]
                  .iter_mut()
                  .zip(&data[nbase..nbase + color_channels])
                {
                  *sum += value;
                }
              }
            }
            // Ties go to the darker level so the result is deterministic.
            let best =
              (0..LEVELS).max_by_key(|&l| (count[l], LEVELS - l)).unwrap();
            for (channel, sum) in
              out[base..base + color_channels].iter_mut().zip(&sums[best])
            {
              *channel = sum / count[best] as f64;
            }
          }
        }
      }
      out
    }
    // `{"MotionBlur", r, θ}`: the mean of 2r + 1 bilinear samples, one
    // pixel apart, on the line through the pixel at θ degrees from the
    // horizontal (counterclockwise, y pointing up); edges are clamped.
    "MotionBlur" => {
      let (Some(radius), Some(angle)) =
        (numeric_param(0, 5.0), numeric_param(1, 0.0))
      else {
        return Ok(unevaluated("ImageEffect", args));
      };
      let radius = radius.round().max(0.0) as i64;
      let (dy, dx) = angle.to_radians().sin_cos();
      let (w, h) = (*width as usize, *height as usize);
      let at = |x: usize, y: usize, c: usize| data[(y * w + x) * ch + c];
      let sample = |fx: f64, fy: f64, c: usize| -> f64 {
        let fx = fx.clamp(0.0, (w - 1) as f64);
        let fy = fy.clamp(0.0, (h - 1) as f64);
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
        let (tx, ty) = (fx - x0 as f64, fy - y0 as f64);
        let top = at(x0, y0, c) * (1.0 - tx) + at(x1, y0, c) * tx;
        let bottom = at(x0, y1, c) * (1.0 - tx) + at(x1, y1, c) * tx;
        top * (1.0 - ty) + bottom * ty
      };
      let taps = (2 * radius + 1) as f64;
      let mut out = data.as_ref().clone();
      if w > 0 && h > 0 {
        for y in 0..h {
          for x in 0..w {
            let base = (y * w + x) * ch;
            for c in 0..color_channels {
              let sum: f64 = (-radius..=radius)
                .map(|t| {
                  let t = t as f64;
                  sample(x as f64 + t * dx, y as f64 - t * dy, c)
                })
                .sum();
              out[base + c] = sum / taps;
            }
          }
        }
      }
      out
    }
    _ => return Ok(unevaluated("ImageEffect", args)),
  };

  Ok(requantize_filtered(Expr::Image {
    color_space: *color_space,
    display_size: None,
    width: *width,
    height: *height,
    channels: *channels,
    data: Arc::new(new_data),
    image_type: *image_type,
  }))
}

/// Resolve the radius specification shared by `Blur`, `Sharpen` and
/// `EdgeDetect` into the pair `(rows, columns)` of per-axis radii.
/// `None` means the call was reported and should be left unevaluated:
/// `<head>::imginv` for a first argument that is not an image,
/// `<head>::bdrad` for a radius that is neither a non-negative number
/// nor a pair of them.
fn blur_radius_pair(head: &str, args: &[Expr]) -> Option<(f64, f64)> {
  if !matches!(&args[0], Expr::Image { .. }) {
    crate::emit_message(&format!(
      "{}::imginv: Expecting an image or graphics instead of {}.",
      head,
      expr_to_string(&args[0])
    ));
    return None;
  }

  // A radius is usable only if it is a plain non-negative number.
  let non_negative = |expr: &Expr| {
    crate::functions::math_ast::try_eval_to_f64(expr)
      .filter(|value| *value >= 0.0)
  };
  let bdrad = |spec: &Expr| {
    crate::emit_message(&format!(
      "{}::bdrad: The specified radius {} should be either a \
       non-negative number or a list of 2 non-negative numbers.",
      head,
      crate::syntax::format_expr(spec, crate::syntax::ExprForm::Output)
    ));
  };

  match args.get(1) {
    None => Some((2.0, 2.0)),
    // `{r_rows, r_columns}` — the first radius applies across rows, i.e.
    // vertically, matching wolframscript's row-then-column convention.
    Some(Expr::List(items)) if items.len() == 2 => {
      if let (Some(rows), Some(columns)) =
        (non_negative(&items[0]), non_negative(&items[1]))
      {
        Some((rows, columns))
      } else {
        bdrad(&args[1]);
        None
      }
    }
    Some(spec) => {
      if let Some(radius) = non_negative(spec) {
        Some((radius, radius))
      } else {
        bdrad(spec);
        None
      }
    }
  }
}

/// The kernels `Blur` and `Sharpen` filter with. Neither reaches further
/// than half way across the image: along an axis of `n` pixels the radius
/// is capped at `Ceiling[n/2]`, standard deviation included, so
/// `Blur[image, 5]` on a two-pixel-wide image is `Blur[image, 1]`.
/// `GaussianFilter`, `GradientFilter` and `EdgeDetect` apply no such cap.
fn blur_radius_axes(
  head: &str,
  args: &[Expr],
) -> Option<(GaussianAxis, GaussianAxis)> {
  let (rows, columns) = blur_radius_pair(head, args)?;
  let Expr::Image { width, height, .. } = &args[0] else {
    return None;
  };
  let cap = |extent: u32| ((extent as f64) / 2.0).ceil().max(1.0);
  Some((
    gaussian_axis_from_radius(rows.min(cap(*height))),
    gaussian_axis_from_radius(columns.min(cap(*width))),
  ))
}

/// Thumbnail[image] / Thumbnail[image, n] — return a smaller version
/// of the image whose longer side is capped at `n` pixels (default
/// 150). The aspect ratio is preserved. Equivalent to
/// `ImageResize[image, {n}]` for the down-scaling case.
pub fn thumbnail_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Thumbnail expects 1 or 2 arguments".into(),
    ));
  }
  if !matches!(&args[0], Expr::Image { .. }) {
    crate::emit_message(&format!(
      "Thumbnail::imginv: Expecting an image or graphics instead of {}.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated("Thumbnail", args));
  }
  let size_arg = if args.len() == 2 {
    expr_to_f64(&args[1])? as i64
  } else {
    150
  };
  image_resize_ast(&[
    args[0].clone(),
    Expr::List(vec![Expr::Integer(size_arg as i128)].into()),
  ])
}
/// ImageAdjust[img] or ImageAdjust[img, contrast]
/// Auto-rescale to [0,1]; optional contrast adjustment
/// A scalar volume `Image3D[{{{v, …}, …}, …}, opts]`: its dimensions
/// (depth, rows, columns as nested), the values in row-major order and the
/// trailing options. Leaves may be any exact or inexact numeric constant.
/// `None` for anything else (stacks of images, colour volumes, …).
fn image3d_scalar_volume(
  e: &Expr,
) -> Option<([usize; 3], Vec<f64>, Vec<Expr>)> {
  let Expr::FunctionCall { name, args } = e else {
    return None;
  };
  if name != "Image3D" || args.is_empty() {
    return None;
  }
  let Expr::List(planes) = &args[0] else {
    return None;
  };
  let mut vals = Vec::new();
  let mut dims = [planes.len(), 0, 0];
  for (i, plane) in planes.iter().enumerate() {
    let Expr::List(rows) = plane else {
      return None;
    };
    if i == 0 {
      dims[1] = rows.len();
    } else if rows.len() != dims[1] {
      return None;
    }
    for (j, row) in rows.iter().enumerate() {
      let Expr::List(cells) = row else {
        return None;
      };
      if i == 0 && j == 0 {
        dims[2] = cells.len();
      } else if cells.len() != dims[2] {
        return None;
      }
      for c in cells {
        vals.push(crate::functions::math_ast::try_eval_to_f64(c)?);
      }
    }
  }
  if vals.is_empty() {
    return None;
  }
  Some((dims, vals, args[1..].to_vec()))
}

/// Rebuild `Image3D[volume, opts]` from row-major values.
fn image3d_from_volume(
  dims: [usize; 3],
  vals: &[f64],
  opts: Vec<Expr>,
) -> Expr {
  let [d0, d1, d2] = dims;
  let volume = Expr::List(
    (0..d0)
      .map(|i| {
        Expr::List(
          (0..d1)
            .map(|j| {
              Expr::List(
                (0..d2)
                  .map(|k| Expr::Real(vals[(i * d1 + j) * d2 + k]))
                  .collect(),
              )
            })
            .collect(),
        )
      })
      .collect(),
  );
  let mut args = vec![volume];
  args.extend(opts);
  call("Image3D", args)
}

pub fn image_adjust_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 4 {
    return Err(InterpreterError::EvaluationError(
      "ImageAdjust expects 1 to 4 arguments".into(),
    ));
  }
  // A scalar volume is stretched to the full [0, 1] range, like a 2D image.
  if args.len() == 1
    && let Some((dims, vals, opts)) = image3d_scalar_volume(&args[0])
  {
    let lo = vals.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let scaled: Vec<f64> = if hi > lo {
      vals.iter().map(|v| (v - lo) / (hi - lo)).collect()
    } else {
      vals
    };
    return Ok(image3d_from_volume(dims, &scaled, opts));
  }
  let coerced;
  let args: &[Expr] = if !matches!(&args[0], Expr::Image { .. })
    && let Some(img) = coerce_graphics_to_image(&args[0])
  {
    let mut v = args.to_vec();
    v[0] = img;
    coerced = v;
    &coerced
  } else {
    args
  };

  let Expr::Image {
    color_space: _,
    display_size: _,
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
  else {
    crate::emit_message(&format!(
      "ImageAdjust::imginv: Expecting an image or graphics instead of {}.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated("ImageAdjust", args));
  };

  // 1-arg form: rescale to [0, 1] using actual min/max.
  // 2-arg scalar form: contrast curve, no rescaling.
  // 2-arg {c, b} / {c, b, γ} form: gamma correction first, then
  // brightness, then contrast — the order wolframscript applies them in
  // (`ImageAdjust[Image[{{0.5}}], {0.5, 0.2, 2}]` is 0.2: 0.5^2 = 0.25,
  // times 1.2 is 0.3, then 0.5 + 1.5*(0.3 - 0.5)).
  enum Adjust {
    Rescale,
    /// contrast, brightness, gamma
    Curve(f64, f64, f64),
  }
  let mode = if args.len() == 1 {
    Adjust::Rescale
  } else if let Expr::List(items) = &args[1]
    && (items.len() == 2 || items.len() == 3)
  {
    let gamma = match items.get(2) {
      Some(g) => expr_to_f64(g)?,
      None => 1.0,
    };
    Adjust::Curve(expr_to_f64(&items[0])?, expr_to_f64(&items[1])?, gamma)
  } else if let Expr::List(_) = &args[1] {
    // A list of any other length isn't a spec ImageAdjust knows.
    crate::emit_message(&format!(
      "ImageAdjust::arg2: Invalid correction parameters {}.",
      expr_to_string(&args[1])
    ));
    return Ok(unevaluated("ImageAdjust", args));
  } else {
    Adjust::Curve(expr_to_f64(&args[1])?, 0.0, 1.0)
  };

  // Real32 images store f32-quantised values in an f64 buffer. Round
  // each pixel to its f32 representation before the computation and
  // again on the way out so the result matches wolframscript's f32
  // image arithmetic.
  let is_real32 = matches!(image_type, ImageType::Real32);
  let snap = |v: f64| -> f64 { if is_real32 { (v as f32) as f64 } else { v } };
  // `{imin, imax}` (third argument) maps that input range onto [0, 1]
  // before the correction; `{omin, omax}` (fourth) maps the result onto
  // that output range.
  let range_arg = |i: usize| -> Result<Option<(f64, f64)>, InterpreterError> {
    match args.get(i) {
      Some(Expr::List(r)) if r.len() == 2 => {
        Ok(Some((expr_to_f64(&r[0])?, expr_to_f64(&r[1])?)))
      }
      _ => Ok(None),
    }
  };
  let input_range = range_arg(2)?;
  let output_range = range_arg(3)?;
  let apply = |v: f64| -> f64 {
    let v = snap(v);
    let v = match input_range {
      Some((lo, hi)) if hi != lo => ((v - lo) / (hi - lo)).clamp(0.0, 1.0),
      _ => v,
    };
    let r = match mode {
      Adjust::Rescale => unreachable!(),
      Adjust::Curve(c, b, gamma) => {
        let after_g = if gamma == 1.0 {
          v
        } else {
          v.max(0.0).powf(gamma)
        };
        let after_b = (after_g * (1.0 + b)).clamp(0.0, 1.0);
        (0.5 + (1.0 + c) * (after_b - 0.5)).clamp(0.0, 1.0)
      }
    };
    let r = match output_range {
      Some((lo, hi)) => lo + r * (hi - lo),
      None => r,
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

  // An integer image stores quantised levels, and wolframscript rounds the
  // adjusted values back onto that grid — an adjusted `"Byte"` pixel is
  // always some n/255, an adjusted `"Bit16"` one some n/65535.
  let levels = match image_type {
    ImageType::Bit => Some(1.0),
    ImageType::Byte => Some(255.0),
    ImageType::Bit16 => Some(65535.0),
    ImageType::Real32 | ImageType::Real64 => None,
  };
  let new_data: Vec<f64> = match levels {
    Some(max) => new_data.iter().map(|v| (v * max).round() / max).collect(),
    None => new_data,
  };

  Ok(Expr::Image {
    color_space: None,
    display_size: None,
    width: *width,
    height: *height,
    channels: *channels,
    data: Arc::new(new_data),
    image_type: *image_type,
  })
}

/// HistogramTransform[img] — histogram equalization: pixel values are
/// replaced by their empirical cumulative distribution, so the resulting
/// histogram is as flat as the (discrete) data allows. Wolfram documents
/// this as `HistogramTransform[img, UniformDistribution[{0, 1}]]` and, for
/// a multichannel image, as operating separately on each channel; both are
/// what happens here.
///
/// The map is the textbook one,
/// `(cdf[v] - cdf[min]) / (1 - cdf[min])`, which pins the darkest value to
/// 0 and the brightest to 1. An already-uniform ramp is therefore left
/// alone, and a constant channel — an opaque alpha channel, say, where the
/// denominator vanishes — passes through untouched.
///
/// Only the one-argument form is implemented; a second argument (a
/// distribution or reference image) is left unevaluated.
pub fn histogram_transform_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let Expr::Image {
    color_space,
    display_size: _,
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
  else {
    crate::emit_message(&format!(
      "HistogramTransform::imginv: {} should be an image, a dataset or a list of datasets.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated("HistogramTransform", args));
  };
  if args.len() != 1 {
    return Ok(unevaluated("HistogramTransform", args));
  }

  let ch = *channels as usize;
  let pixels = (*width as usize) * (*height as usize);
  if pixels == 0 || ch == 0 {
    return Ok(args[0].clone());
  }
  let mut new_data = data.as_ref().clone();
  for c_idx in 0..ch {
    let mut sorted: Vec<f64> =
      (0..pixels).map(|i| data[i * ch + c_idx]).collect();
    sorted
      .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len() as f64;
    for i in 0..pixels {
      let v = data[i * ch + c_idx];
      // The *midpoint* of the value's own step of the cumulative
      // distribution — the share of pixels strictly below it plus half the
      // share sitting on it. Every pixel of one value therefore lands in the
      // middle of the band the equalization gives that value, so a channel
      // whose values are the 256 display levels comes back unchanged and a
      // constant channel comes back at 0.5.
      let below = sorted.partition_point(|&s| s < v) as f64;
      let equal = sorted.partition_point(|&s| s <= v) as f64 - below;
      let mid = (below + equal / 2.0) / n;
      // The distribution runs over 256 display levels but the levels
      // themselves are `k/255`, so the band midpoints are rescaled from the
      // one range to the other.
      new_data[i * ch + c_idx] = ((mid * 256.0 - 0.5) / 255.0).clamp(0.0, 1.0);
    }
  }
  Ok(Expr::Image {
    color_space: *color_space,
    display_size: None,
    width: *width,
    height: *height,
    channels: *channels,
    data: Arc::new(new_data),
    image_type: *image_type,
  })
}

/// The colors `ImageHistogram` draws each channel's curve in: gray for a
/// single-channel image, otherwise red/green/blue for the first three
/// channels and gray for any beyond that (e.g. an alpha channel).
fn image_histogram_channel_colors(
  channels: usize,
) -> Vec<crate::functions::graphics::Color> {
  use crate::functions::graphics::Color;
  if channels <= 1 {
    return vec![Color::new(0.5, 0.5, 0.5)];
  }
  [
    Color::new(1.0, 0.0, 0.0),
    Color::new(0.0, 0.65, 0.0),
    Color::new(0.0, 0.0, 1.0),
  ]
  .into_iter()
  .chain(std::iter::repeat(Color::new(0.5, 0.5, 0.5)))
  .take(channels)
  .collect()
}

fn image_histogram_chart_options() -> crate::functions::chart::ChartOptions {
  crate::functions::chart::ChartOptions {
    svg_width: crate::functions::plot::DEFAULT_WIDTH,
    svg_height: crate::functions::plot::DEFAULT_HEIGHT,
    full_width: false,
    chart_labels: Vec::new(),
    chart_label_position: crate::functions::chart::LabelPosition::Below,
    plot_label: None,
    axes_label: None,
    frame_label: None,
    chart_style: Vec::new(),
    chart_style_scheme: None,
    chart_legends: Vec::new(),
    chart_legends_auto: false,
    legend_layout: crate::functions::plot::LegendLayout::Column,
    plot_range_x: Some((0.0, 255.0)),
    plot_range_y: None,
    bar_origin_left: false,
    labeling_function: None,
    ticks_x: None,
    ticks_y: None,
    ticks: true,
    stacked: false,
    epilog: Vec::new(),
    color_function: None,
    color_function_scaling: true,
  }
}

/// ImageHistogram[image] / ImageHistogram[image, opts...] — the
/// distribution of each channel's pixel values, scaled to the usual 0-255
/// display range. `Appearance -> "Separated"` draws one histogram per
/// channel side by side; any other setting (the default, `"RGB"`,
/// `"Transparent"`, `"Stacked"`) overlays them translucently in one plot,
/// the same treatment `Histogram` already gives several datasets.
pub fn image_histogram_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(unevaluated("ImageHistogram", args));
  }
  let Expr::Image {
    width,
    height,
    channels,
    data,
    ..
  } = &args[0]
  else {
    crate::emit_message(&format!(
      "ImageHistogram::imginv: Expecting an image or graphics instead of {}.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated("ImageHistogram", args));
  };

  let separated = args[1..].iter().any(|opt| {
    crate::functions::graphics::option_name_value(opt).is_some_and(
      |(name, value)| {
        name == "Appearance"
          && matches!(value.as_ref(), Expr::String(s) if s == "Separated")
      },
    )
  });

  let w = *width as usize;
  let h = *height as usize;
  let ch = (*channels as usize).max(1);
  let pixels = w * h;
  let colors = image_histogram_channel_colors(ch);
  let datasets: Vec<Vec<f64>> = (0..ch)
    .map(|c| (0..pixels).map(|i| data[i * ch + c] * 255.0).collect())
    .collect();
  let bin_spec = crate::functions::plot::BinSpec::Width(1.0);

  if separated {
    let mut items = Vec::with_capacity(ch);
    for (c, dataset) in datasets.iter().enumerate() {
      let mut opts = image_histogram_chart_options();
      opts.chart_style = vec![colors[c]];
      let svg = crate::functions::plot::generate_histogram_svg(
        std::slice::from_ref(dataset),
        Some(&bin_spec),
        crate::functions::plot::HistogramHeight::Count,
        &mut opts,
      )?;
      items.push(Expr::Graphics {
        svg,
        is_3d: false,
        source: None,
        head: None,
        structure: None,
      });
    }
    return crate::functions::graphics::graphics_row_ast(&[Expr::List(
      items.into(),
    )]);
  }

  let mut opts = image_histogram_chart_options();
  opts.chart_style = colors;
  let svg = crate::functions::plot::generate_histogram_svg(
    &datasets,
    Some(&bin_spec),
    crate::functions::plot::HistogramHeight::Count,
    &mut opts,
  )?;
  Ok(crate::graphics_result(svg))
}

/// ImageLevels[image] / ImageLevels[image, spec] — `{value, count}` pairs for
/// the pixel values of each channel. A single-channel image gives one flat
/// list of pairs, a multichannel image one such list per channel. With `n`
/// the range splits into n bins `[k/n, (k+1)/n)` labelled by their lower
/// edge (the top value counting in the last); without a spec a bit image
/// counts its two values `0` and `1` and every other image uses 256 bins;
/// `"Byte"` counts the 256 byte values and `All` every distinct value.
pub fn image_levels_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Ok(unevaluated("ImageLevels", args));
  }
  let Expr::Image {
    width,
    height,
    channels,
    data,
    image_type,
    ..
  } = &args[0]
  else {
    crate::emit_message(&format!(
      "ImageLevels::imginv: Expecting an image or graphics instead of {}.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated("ImageLevels", args));
  };

  enum Spec {
    Bins(usize),
    Bits,
    Bytes,
    Distinct,
  }
  let spec = match args.get(1) {
    None if matches!(image_type, ImageType::Bit) => Spec::Bits,
    None => Spec::Bins(256),
    Some(Expr::Integer(n)) if *n >= 1 => Spec::Bins(*n as usize),
    Some(Expr::String(s)) if s == "Byte" => Spec::Bytes,
    Some(Expr::Identifier(s)) if s == "All" => Spec::Distinct,
    Some(other) => {
      crate::emit_message(&format!(
        "ImageLevels::arg2: {} is not a positive integer, \"Byte\" or All.",
        expr_to_string(other)
      ));
      return Ok(unevaluated("ImageLevels", args));
    }
  };

  let pixels = (*width as usize) * (*height as usize);
  let ch = (*channels as usize).max(1);
  let pair = |value: Expr, count: i128| {
    Expr::List(vec![value, Expr::Integer(count)].into())
  };

  let per_channel: Vec<Expr> = (0..ch)
    .map(|c| {
      let values = (0..pixels).map(|i| data[i * ch + c].clamp(0.0, 1.0));
      match spec {
        Spec::Distinct => {
          let mut sorted: Vec<f64> = values.collect();
          sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
          let mut out: Vec<(f64, i128)> = Vec::new();
          for v in sorted {
            match out.last_mut() {
              Some((last, n)) if *last == v => *n += 1,
              _ => out.push((v, 1)),
            }
          }
          Expr::List(
            out
              .into_iter()
              .map(|(v, n)| pair(Expr::Real(v), n))
              .collect(),
          )
        }
        Spec::Bits | Spec::Bytes => {
          let top = if matches!(spec, Spec::Bits) {
            1.0
          } else {
            255.0
          };
          let mut counts = vec![0i128; top as usize + 1];
          for v in values {
            counts[(v * top).round() as usize] += 1;
          }
          Expr::List(
            counts
              .iter()
              .enumerate()
              .map(|(k, n)| pair(Expr::Integer(k as i128), *n))
              .collect(),
          )
        }
        Spec::Bins(n) => {
          let mut counts = vec![0i128; n];
          for v in values {
            counts[((v * n as f64).floor() as usize).min(n - 1)] += 1;
          }
          Expr::List(
            counts
              .iter()
              .enumerate()
              .map(|(k, count)| pair(Expr::Real(k as f64 / n as f64), *count))
              .collect(),
          )
        }
      }
    })
    .collect();

  if ch == 1 {
    Ok(per_channel.into_iter().next().unwrap())
  } else {
    Ok(Expr::List(per_channel.into()))
  }
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

  if let Expr::Image {
    color_space: _,
    display_size: _,
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
  {
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
      color_space: None,
      display_size: None,
      width: new_w,
      height: new_h,
      channels: *channels,
      data: Arc::new(new_data),
      image_type: *image_type,
    })
  } else {
    crate::emit_message(&format!(
      "ImageReflect::imgvinv: Expecting an image, graphics or video instead of {}.",
      expr_to_string(&args[0])
    ));
    Ok(unevaluated("ImageReflect", args))
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
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated("ImageRotate", args));
  }

  let pi = std::f64::consts::PI;
  // The angle may be given as a number or as the edge the current top should end
  // up at. Those four names are the right-angle rotations woxi already performs,
  // so they map straight onto angles.
  let angle = if args.len() >= 2 {
    match &args[1] {
      Expr::Identifier(name) | Expr::Constant(name) if name == "Top" => 0.0,
      Expr::Identifier(name) | Expr::Constant(name) if name == "Left" => {
        pi / 2.0
      }
      Expr::Identifier(name) | Expr::Constant(name) if name == "Bottom" => pi,
      Expr::Identifier(name) | Expr::Constant(name) if name == "Right" => {
        -pi / 2.0
      }
      other => {
        if let Ok(value) = expr_to_f64(other) {
          value
        } else {
          crate::emit_message(&format!(
            "ImageRotate::imgang: Angle {} should be a real number; one of \
           Top, Bottom, Left or Right; or a rule from one to another.",
            crate::syntax::format_expr(other, crate::syntax::ExprForm::Output)
          ));
          return Ok(unevaluated("ImageRotate", args));
        }
      }
    }
  } else {
    pi / 2.0
  };

  let Expr::Image {
    color_space: _,
    display_size: _,
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
    // Any other angle: the bounding box of the turned image, each pixel
    // taking the source pixel under its centre (black outside), the way
    // wolframscript resamples.
    let (sin, cos) = angle.sin_cos();
    let (wf, hf) = (w as f64, h as f64);
    let size = |v: f64| (v - 1e-9).ceil().max(1.0) as usize;
    let bw = wf * cos.abs() + hf * sin.abs();
    let bh = wf * sin.abs() + hf * cos.abs();
    let (nw, nh) = (size(bw), size(bh));
    let mut out = Vec::with_capacity(nw * nh * ch);
    for r in 0..nh {
      for c in 0..nw {
        // Output pixel centre relative to the centre, y pointing up, scaled
        // from the whole-pixel canvas back to the exact bounding box.
        let dx = (c as f64 + 0.5 - nw as f64 / 2.0) * bw / nw as f64;
        let dy = (nh as f64 / 2.0 - (r as f64 + 0.5)) * bh / nh as f64;
        // Turned back by the angle into the source image.
        let sx = dx * cos + dy * sin + wf / 2.0;
        let sy = -dx * sin + dy * cos + hf / 2.0;
        let (col, row) = (sx.floor(), (hf - sy).floor());
        if col >= 0.0 && row >= 0.0 && (col as usize) < w && (row as usize) < h
        {
          let base = (row as usize * w + col as usize) * ch;
          out.extend_from_slice(&data[base..base + ch]);
        } else {
          out.extend(std::iter::repeat_n(0.0, ch));
        }
      }
    }
    return Ok(Expr::Image {
      color_space: None,
      display_size: None,
      width: nw as u32,
      height: nh as u32,
      channels: *channels,
      data: Arc::new(out),
      image_type: *image_type,
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
    color_space: None,
    display_size: None,
    width: new_w,
    height: new_h,
    channels: *channels,
    data: Arc::new(new_data),
    image_type: *image_type,
  })
}

/// ImageResize[img, {w, h}] or ImageResize[img, w] - Resize to target dimensions
pub fn image_resize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let coerced;
  let args: &[Expr] = if !matches!(&args[0], Expr::Image { .. })
    && let Some(img) = coerce_graphics_to_image(&args[0])
  {
    let mut v = args.to_vec();
    v[0] = img;
    coerced = v;
    &coerced
  } else {
    args
  };

  if !matches!(&args[0], Expr::Image { .. }) {
    crate::emit_message(&format!(
      "ImageResize::imginv: Expecting an image or graphics instead of {}.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated("ImageResize", args));
  }

  // Trailing arguments are options; `Resampling -> "Nearest"` (or
  // "Constant") picks the nearest source pixel instead of interpolating.
  let mut nearest = false;
  for opt in &args[2..] {
    let Expr::Rule {
      pattern,
      replacement,
    } = opt
    else {
      return Ok(unevaluated("ImageResize", args));
    };
    if matches!(pattern.as_ref(), Expr::Identifier(n) | Expr::String(n) if n == "Resampling")
      && matches!(replacement.as_ref(), Expr::String(m) | Expr::Identifier(m) if m == "Nearest" || m == "Constant")
    {
      nearest = true;
    }
  }

  // `Scaled[s]` / `Scaled[{sx, sy}]` is a fraction of the image's own size:
  // restate it as the pixel dimensions it stands for, then resize as usual.
  let scaled;
  let args: &[Expr] = if let Expr::FunctionCall { name, args: sargs } = &args[1]
    && name == "Scaled"
    && sargs.len() == 1
    && let Expr::Image { width, height, .. } = &args[0]
    && let Some((sx, sy)) = match &sargs[0] {
      Expr::List(f) if f.len() == 2 => {
        expr_to_f64(&f[0]).ok().zip(expr_to_f64(&f[1]).ok())
      }
      f => expr_to_f64(f).ok().map(|v| (v, v)),
    }
    && sx > 0.0
    && sy > 0.0
  {
    let px = |frac: f64, side: u32| {
      Expr::Integer((frac * side as f64).round().max(1.0) as i128)
    };
    scaled = vec![
      args[0].clone(),
      Expr::List(vec![px(sx, *width), px(sy, *height)].into()),
    ];
    &scaled
  } else {
    args
  };

  // `{{wmax}, {hmax}}` fits the image inside a wmax x hmax box, keeping its
  // aspect ratio: restate it as the pixel dimensions it stands for.
  let boxed;
  let args: &[Expr] = if let Expr::List(dims) = &args[1]
    && dims.len() == 2
    && let Expr::Image { width, height, .. } = &args[0]
    && let (Expr::List(w), Expr::List(h)) = (&dims[0], &dims[1])
    && w.len() == 1
    && h.len() == 1
    && let Ok(wmax) = expr_to_f64(&w[0])
    && let Ok(hmax) = expr_to_f64(&h[0])
    && wmax > 0.0
    && hmax > 0.0
  {
    let k = (wmax / *width as f64).min(hmax / *height as f64);
    let px =
      |side: u32| Expr::Integer((k * side as f64).round().max(1.0) as i128);
    boxed = vec![
      args[0].clone(),
      Expr::List(vec![px(*width), px(*height)].into()),
    ];
    &boxed
  } else {
    args
  };

  // Validate the size specification against the form as written, so the message
  // names what the caller passed. A size that is present but not positive is
  // reported; a bare specification that is not a number at all is left alone
  // silently, both matching wolframscript. Automatic and All stand for "derive
  // this side", so they pass.
  {
    let placeholder = |e: &Expr| matches!(e, Expr::Identifier(s) | Expr::Constant(s) if s == "Automatic" || s == "All");
    let imgrssz = || {
      crate::emit_message(&format!(
        "ImageResize::imgrssz: The size {} is not a valid image size \
         specification.",
        crate::syntax::format_expr(&args[1], crate::syntax::ExprForm::Output)
      ));
      Ok(unevaluated("ImageResize", args))
    };
    match &args[1] {
      Expr::List(dims) if !dims.is_empty() && dims.len() <= 2 => {
        for dim in dims {
          if placeholder(dim) {
            continue;
          }
          match expr_to_f64(dim) {
            Ok(v) if v > 0.0 => {}
            // A list is reported either way — an element that is not a number
            // is as invalid as one that is not positive.
            _ => return imgrssz(),
          }
        }
      }
      other if placeholder(other) => {}
      other => match expr_to_f64(other) {
        Ok(v) if v > 0.0 => {}
        Ok(_) => return imgrssz(),
        // A bare non-numeric size may still become one later.
        Err(_) => return Ok(unevaluated("ImageResize", args)),
      },
    }
  }

  let Expr::Image {
    color_space: _,
    display_size: _,
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
  else {
    unreachable!()
  };
  let src_w = *width as usize;
  let src_h = *height as usize;
  let ch = *channels as usize;

  // Parse the target size.
  let is_automatic =
    |e: &Expr| matches!(e, Expr::Identifier(s) if s == "Automatic");
  let (new_w, new_h) = match &args[1] {
    Expr::List(dims) if dims.len() == 2 => {
      let auto_w = is_automatic(&dims[0]);
      let auto_h = is_automatic(&dims[1]);
      let w_opt = if auto_w {
        None
      } else {
        Some(expr_to_f64(&dims[0])? as u32)
      };
      let h_opt = if auto_h {
        None
      } else {
        Some(expr_to_f64(&dims[1])? as u32)
      };
      match (w_opt, h_opt) {
        (Some(w), Some(h)) => (w, h),
        (Some(w), None) => {
          let h = ((w as f64) * (src_h as f64) / (src_w as f64)).round() as u32;
          (w, h.max(1))
        }
        (None, Some(h)) => {
          let w = ((h as f64) * (src_w as f64) / (src_h as f64)).round() as u32;
          (w.max(1), h)
        }
        (None, None) => (*width, *height),
      }
    }
    Expr::List(dims) if dims.len() == 1 => {
      // {n}: cap the longer side at n, preserve aspect.
      let n = expr_to_f64(&dims[0])?;
      let (w, h) = if src_w >= src_h {
        let h = (n * src_h as f64 / src_w as f64).round().max(1.0) as u32;
        (n as u32, h)
      } else {
        let w = (n * src_w as f64 / src_h as f64).round().max(1.0) as u32;
        (w, n as u32)
      };
      (w, h)
    }
    other => {
      // A fractional size rounds, and never below a single pixel:
      // ImageResize[img, 0.5] is a 1x1 image, not an error.
      let w = (expr_to_f64(other)?.round().max(1.0)) as u32;
      let h = ((w as f64) * (src_h as f64) / (src_w as f64)).round() as u32;
      (w, h.max(1))
    }
  };

  let (new_w, new_h) = (new_w.max(1), new_h.max(1));

  // Bilinear interpolation on the f64 buffer (per channel).
  let mut new_data = vec![0.0_f64; (new_w as usize) * (new_h as usize) * ch];
  let scale_x = src_w as f64 / new_w as f64;
  let scale_y = src_h as f64 / new_h as f64;
  for ny in 0..new_h as usize {
    if nearest {
      let sy = ((((ny as f64) + 0.5) * scale_y).floor() as usize)
        .min(src_h.saturating_sub(1));
      for nx in 0..new_w as usize {
        let sx = ((((nx as f64) + 0.5) * scale_x).floor() as usize)
          .min(src_w.saturating_sub(1));
        for c in 0..ch {
          new_data[(ny * new_w as usize + nx) * ch + c] =
            data[(sy * src_w + sx) * ch + c];
        }
      }
      continue;
    }
    let sy_f = ((ny as f64) + 0.5) * scale_y - 0.5;
    let sy0 = sy_f.floor().max(0.0) as usize;
    let sy1 = (sy0 + 1).min(src_h.saturating_sub(1));
    let fy = (sy_f - sy0 as f64).clamp(0.0, 1.0);
    for nx in 0..new_w as usize {
      let sx_f = ((nx as f64) + 0.5) * scale_x - 0.5;
      let sx0 = sx_f.floor().max(0.0) as usize;
      let sx1 = (sx0 + 1).min(src_w.saturating_sub(1));
      let fx = (sx_f - sx0 as f64).clamp(0.0, 1.0);
      for c in 0..ch {
        let v00 = data[(sy0 * src_w + sx0) * ch + c];
        let v01 = data[(sy0 * src_w + sx1) * ch + c];
        let v10 = data[(sy1 * src_w + sx0) * ch + c];
        let v11 = data[(sy1 * src_w + sx1) * ch + c];
        let v0 = v00 * (1.0 - fx) + v01 * fx;
        let v1 = v10 * (1.0 - fx) + v11 * fx;
        let v = v0 * (1.0 - fy) + v1 * fy;
        new_data[(ny * new_w as usize + nx) * ch + c] = v;
      }
    }
  }

  Ok(Expr::Image {
    color_space: None,
    display_size: None,
    width: new_w,
    height: new_h,
    channels: *channels,
    data: Arc::new(new_data),
    image_type: *image_type,
  })
}

/// ImageCrop[img, {{x1,y1},{x2,y2}}] - Crop to region
/// How many pixels a cropping spec removes from the low side (left, or bottom
/// in Wolfram's bottom-left origin) when `remove` pixels have to go in total.
/// `low_share` is the fraction of the removal taken from the low side: 1 for
/// `Left`/`Bottom`, 0 for `Right`/`Top`, 1/2 for `Center`. Centering splits the
/// remainder with round-half-to-even, so cropping a 4-wide image to 3 keeps the
/// leftmost three while cropping it to 1 keeps the third pixel.
fn image_crop_low_margin(low_share: f64, remove: i64) -> i64 {
  (remove as f64 * low_share).round_ties_even() as i64
}

/// The per-axis share of the removed pixels that comes off the low side
/// (left / bottom), as `(horizontal, vertical)`. The third argument is either
/// side names (`Left`, `Right`, `Top`, `Bottom`, `Center`) or a numeric pair
/// `{x, y}` in `[-1, 1]` where `-1` is the same as `Left`/`Bottom` and `1` the
/// same as `Right`/`Top`.
fn image_crop_sides(spec: &Expr) -> Option<(f64, f64)> {
  let mut horizontal = 0.5;
  let mut vertical = 0.5;
  if let Expr::List(items) = spec
    && items.len() == 2
    && let (Some(x), Some(y)) = (
      crate::functions::math_ast::try_eval_to_f64(&items[0]),
      crate::functions::math_ast::try_eval_to_f64(&items[1]),
    )
  {
    let share = |t: f64| (1.0 - t.clamp(-1.0, 1.0)) / 2.0;
    return Some((share(x), share(y)));
  }
  let names: Vec<&Expr> = match spec {
    Expr::List(items) => items.iter().collect(),
    single => vec![single],
  };
  for name in names {
    let (Expr::Identifier(s) | Expr::String(s)) = name else {
      return None;
    };
    match s.as_str() {
      "Left" => horizontal = 1.0,
      "Right" => horizontal = 0.0,
      "Bottom" => vertical = 1.0,
      "Top" => vertical = 0.0,
      "Center" => {}
      other => {
        crate::emit_message(&format!(
          "ImageCrop::cspecsym: Cropping value {other} is not one of Center, Left, Right, Top or Bottom."
        ));
        return None;
      }
    }
  }
  Some((horizontal, vertical))
}

pub fn image_crop_ast(all_args: &[Expr]) -> Result<Expr, InterpreterError> {
  // A trailing `Padding -> value` option fills the pixels added when the
  // requested size exceeds the image.
  let mut args = all_args;
  let mut padding: Option<&Expr> = None;
  while let Some((
    Expr::Rule {
      pattern,
      replacement,
    },
    rest,
  )) = args.split_last()
  {
    if matches!(&**pattern, Expr::Identifier(n) | Expr::String(n) if n == "Padding")
    {
      padding.get_or_insert(&**replacement);
    }
    args = rest;
  }
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "ImageCrop expects 1, 2 or 3 arguments".into(),
    ));
  }

  let Expr::Image {
    color_space: _,
    display_size: _,
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
  else {
    return Err(InterpreterError::EvaluationError(
      "ImageCrop: first argument is not an Image".into(),
    ));
  };

  // ImageCrop[image, size] / ImageCrop[image, size, spec] — crop (or, when the
  // requested size is larger, pad) to an explicit width and height.
  if args.len() >= 2 {
    let echo = || Ok(unevaluated("ImageCrop", all_args));
    // Non-integer sizes (e.g. `{300, 300}/zoom`) are rounded to whole pixels.
    let pixels = |e: &Expr| {
      as_int(e).or_else(|| {
        crate::functions::math_ast::try_eval_to_f64(e)
          .filter(|v| v.is_finite())
          .map(|v| v.round() as i64)
      })
    };
    let (Some(target_w), Some(target_h)) = (match &args[1] {
      Expr::List(size) if size.len() == 2 => {
        (pixels(&size[0]), pixels(&size[1]))
      }
      other => (pixels(other), pixels(other)),
    }) else {
      return echo();
    };
    let (horizontal, vertical) = match args.get(2) {
      Some(spec) => match image_crop_sides(spec) {
        Some(sides) => sides,
        None => return echo(),
      },
      None => (0.5, 0.5),
    };

    let remove_w = *width as i64 - target_w;
    let remove_h = *height as i64 - target_h;
    let left = image_crop_low_margin(horizontal, remove_w);
    let bottom = image_crop_low_margin(vertical, remove_h);
    // The extents are removals, so they enter the padder negated.
    return Ok(
      image_pad_extents(
        &args[0],
        -left,
        -(remove_w - left),
        -bottom,
        -(remove_h - bottom),
        padding,
      )
      .unwrap_or_else(|| unevaluated("ImageCrop", all_args)),
    );
  }

  // Auto-crop: trim any uniform border that matches the (0, 0) pixel.
  let w = *width as usize;
  let h = *height as usize;
  let ch = *channels as usize;
  if w == 0 || h == 0 {
    return Ok(args[0].clone());
  }
  let corner: Vec<f64> = (0..ch).map(|c| data[c]).collect();
  // For Real32, allow one f32 ulp of tolerance against the corner.
  let tol = if matches!(image_type, ImageType::Real32) {
    f32::EPSILON as f64
  } else {
    0.0
  };
  let pixel_matches = |x: usize, y: usize| -> bool {
    let base = (y * w + x) * ch;
    (0..ch).all(|c| (data[base + c] - corner[c]).abs() <= tol)
  };
  let mut top = h;
  'top: for y in 0..h {
    for x in 0..w {
      if !pixel_matches(x, y) {
        top = y;
        break 'top;
      }
    }
  }
  if top == h {
    // The whole image matches the corner; return a 1×1 image.
    return Ok(Expr::Image {
      color_space: None,
      display_size: None,
      width: 1,
      height: 1,
      channels: *channels,
      data: Arc::new(corner),
      image_type: *image_type,
    });
  }
  let mut bottom = top;
  for y in (top..h).rev() {
    let mut found = false;
    for x in 0..w {
      if !pixel_matches(x, y) {
        bottom = y + 1;
        found = true;
        break;
      }
    }
    if found {
      break;
    }
  }
  let mut left = w;
  'left: for x in 0..w {
    for y in top..bottom {
      if !pixel_matches(x, y) {
        left = x;
        break 'left;
      }
    }
  }
  let mut right = left;
  for x in (left..w).rev() {
    let mut found = false;
    for y in top..bottom {
      if !pixel_matches(x, y) {
        right = x + 1;
        found = true;
        break;
      }
    }
    if found {
      break;
    }
  }

  let new_w = right - left;
  let new_h = bottom - top;
  let mut new_data = Vec::with_capacity(new_w * new_h * ch);
  for y in top..bottom {
    for x in left..right {
      let base = (y * w + x) * ch;
      for c in 0..ch {
        new_data.push(data[base + c]);
      }
    }
  }
  Ok(Expr::Image {
    color_space: None,
    display_size: None,
    width: new_w as u32,
    height: new_h as u32,
    channels: *channels,
    data: Arc::new(new_data),
    image_type: *image_type,
  })
}

/// The source pixel a padded position maps to under an edge-extension mode,
/// where `i` may run outside `0..len`. Returns `None` for a constant fill.
fn image_pad_source(mode: &str, i: i64, len: i64) -> Option<i64> {
  match mode {
    "Fixed" => Some(i.clamp(0, len - 1)),
    "Periodic" => Some(i.rem_euclid(len)),
    // Mirror about the edges without repeating them: a b c b a b c …
    "Reflected" if len > 1 => {
      let period = 2 * (len - 1);
      let m = i.rem_euclid(period);
      Some(if m > len - 1 { period - m } else { m })
    }
    "Reflected" => Some(0),
    _ => None,
  }
}

/// An integer padding extent (negative extents trim).
fn as_int(e: &Expr) -> Option<i64> {
  match e {
    Expr::Integer(n) => Some(*n as i64),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => as_int(operand).map(|n| -n),
    _ => None,
  }
}

/// Read one padding extent pair `{lo, hi}`, or a scalar standing for both.
fn image_pad_extent(spec: &Expr) -> Option<(i64, i64)> {
  // A fractional extent counts its whole pixels (26.6 pads 26, -1.5
  // trims 1).
  let whole = |e: &Expr| {
    as_int(e).or_else(|| {
      crate::functions::math_ast::try_eval_to_f64(e)
        .filter(|v| v.is_finite())
        .map(|v| v.trunc() as i64)
    })
  };
  match spec {
    Expr::List(pair) if pair.len() == 2 => {
      Some((whole(&pair[0])?, whole(&pair[1])?))
    }
    other => {
      let n = whole(other)?;
      Some((n, n))
    }
  }
}

/// ImagePad[img, m] / ImagePad[img, {{left, right}, {bottom, top}}] —
/// grow (or, for negative extents, trim) an image on each side. The optional
/// third argument is the fill: a constant value, or one of the edge-extension
/// modes `"Fixed"`, `"Periodic"` and `"Reflected"`. Note that the vertical
/// pair is `{bottom, top}` while `ImageData` lists rows top-first.
pub fn image_pad_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let echo = || Ok(unevaluated("ImagePad", args));
  if args.len() < 2 || args.len() > 3 {
    return echo();
  }
  if !matches!(&args[0], Expr::Image { .. }) {
    return echo();
  }

  // `m` pads every side; `{{left, right}, {bottom, top}}` names them.
  let ((left, right), (bottom, top)) = match &args[1] {
    Expr::List(sides)
      if sides.len() == 2
        && sides.iter().any(|s| matches!(s, Expr::List(_))) =>
    {
      match (image_pad_extent(&sides[0]), image_pad_extent(&sides[1])) {
        (Some(h), Some(v)) => (h, v),
        _ => return echo(),
      }
    }
    other => match image_pad_extent(other) {
      Some((a, b)) => ((a, b), (a, b)),
      None => return echo(),
    },
  };

  // A color fills an image only with as many channels as the image has (or
  // just one, a gray level); a single-channel image takes on the color's
  // channels: gray values become RGB and existing pixels opaque.
  let color_channels = match args.get(2) {
    Some(Expr::FunctionCall { name, args: c }) => {
      match (name.as_str(), c.len()) {
        ("GrayLevel", 1) => Some(1),
        ("GrayLevel", 2) => Some(2),
        ("RGBColor" | "Hue", 3) => Some(3),
        ("RGBColor" | "Hue", 4) => Some(4),
        _ => None,
      }
    }
    _ => None,
  };
  if let (Some(n), Expr::Image { channels: 1, .. }) = (color_channels, &args[0])
    && n > 1
  {
    let promoted = promote_gray_image(&args[0], n);
    let mut promoted_args = args.to_vec();
    promoted_args[0] = promoted;
    return image_pad_ast(&promoted_args);
  }
  if let (Some(padding), Expr::Image { channels, .. }) = (args.get(2), &args[0])
    && *channels > 1
    && let Some(n) = color_channels
    && n != 1
    && n != *channels as usize
  {
    crate::emit_message(&format!(
      "ImagePad::incpad: The specified padding value {} is not compatible \
         with a {}-channel image with Automatic color space.",
      crate::syntax::format_expr(padding, crate::syntax::ExprForm::Output),
      channels
    ));
    return echo();
  }
  if let Some(image) =
    image_pad_extents(&args[0], left, right, bottom, top, args.get(2))
  {
    Ok(image)
  } else {
    crate::emit_message(&format!(
      "ImagePad::padnull: Padding specification {} corresponds to an empty image.",
      crate::syntax::format_expr(&args[1], crate::syntax::ExprForm::Output)
    ));
    echo()
  }
}

/// A single-channel image with `channels` channels: the gray value in each
/// color channel and an opaque alpha.
fn promote_gray_image(image: &Expr, channels: usize) -> Expr {
  let Expr::Image {
    color_space,
    display_size,
    width,
    height,
    data,
    image_type,
    ..
  } = image
  else {
    return image.clone();
  };
  let colors = if channels >= 3 { 3 } else { 1 };
  let alpha = channels == 2 || channels == 4;
  let mut out = Vec::with_capacity(data.len() * channels);
  for &v in data.iter() {
    for _ in 0..colors {
      out.push(v);
    }
    if alpha {
      out.push(1.0);
    }
  }
  Expr::Image {
    color_space: *color_space,
    display_size: *display_size,
    width: *width,
    height: *height,
    channels: channels as u8,
    data: Arc::new(out),
    image_type: *image_type,
  }
}

/// Grow `image` by the given per-side extents (negative extents trim), filling
/// new pixels from `padding`: an edge-extension mode, a constant value or list
/// of per-channel values, or black when absent. Returns `None` when the result
/// would be empty or the fill cannot be read.
fn image_pad_extents(
  image: &Expr,
  left: i64,
  right: i64,
  bottom: i64,
  top: i64,
  padding: Option<&Expr>,
) -> Option<Expr> {
  let Expr::Image {
    width,
    height,
    channels,
    data,
    image_type,
    ..
  } = image
  else {
    return None;
  };
  let (w, h, ch) = (*width as i64, *height as i64, *channels as usize);
  let new_w = w + left + right;
  let new_h = h + bottom + top;
  if new_w <= 0 || new_h <= 0 {
    return None;
  }

  // The fill: an extension mode, or a constant (defaulting to black).
  let mode = match padding {
    Some(Expr::String(s) | Expr::Identifier(s))
      if image_pad_source(s, 0, 1).is_some() =>
    {
      Some(s.clone())
    }
    _ => None,
  };
  let fill: Vec<f64> = match (&mode, padding) {
    (Some(_), _) | (None, None) => vec![0.0; ch],
    (None, Some(value)) => match value {
      Expr::List(items) if items.len() == ch => {
        let mut v = Vec::with_capacity(ch);
        for item in items {
          v.push(crate::functions::math_ast::try_eval_to_f64(item)?);
        }
        v
      }
      other => {
        if let Some((r, g, b, alpha)) = color_directive_to_rgb(other) {
          // A color directive fills in the image's own channel layout: gray
          // images take the luminance, and an alpha channel the directive's
          // opacity (opaque when it names none).
          match ch {
            1 => vec![(r + g + b) / 3.0],
            2 => vec![(r + g + b) / 3.0, alpha.unwrap_or(1.0)],
            3 => vec![r, g, b],
            _ => {
              let mut v = vec![r, g, b, alpha.unwrap_or(1.0)];
              v.resize(ch, 1.0);
              v
            }
          }
        } else {
          vec![crate::functions::math_ast::try_eval_to_f64(other)?; ch]
        }
      }
    },
  };

  let mut new_data = Vec::with_capacity((new_w * new_h) as usize * ch);
  for row in 0..new_h {
    // Row 0 of the output is the top, so the top extent shifts the source.
    let sy = row - top;
    for col in 0..new_w {
      let sx = col - left;
      let src = match &mode {
        Some(m) => Some((
          image_pad_source(m, sx, w).unwrap_or(0),
          image_pad_source(m, sy, h).unwrap_or(0),
        )),
        None if (0..w).contains(&sx) && (0..h).contains(&sy) => Some((sx, sy)),
        None => None,
      };
      match src {
        Some((x, y)) => {
          let base = ((y * w + x) as usize) * ch;
          new_data.extend_from_slice(&data[base..base + ch]);
        }
        None => new_data.extend_from_slice(&fill),
      }
    }
  }

  Some(Expr::Image {
    color_space: None,
    display_size: None,
    width: new_w as u32,
    height: new_h as u32,
    channels: *channels,
    data: Arc::new(new_data),
    image_type: *image_type,
  })
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
              return Ok(unevaluated("ImageTrim", args));
            }
          };
          let (bx, by) = match &outer[1] {
            Expr::List(p) if p.len() == 2 => {
              (expr_to_f64(&p[0])?, expr_to_f64(&p[1])?)
            }
            _ => {
              return Ok(unevaluated("ImageTrim", args));
            }
          };
          (ax.min(bx), ay.min(by), ax.max(bx), ay.max(by))
        }
        _ => {
          return Ok(unevaluated("ImageTrim", args));
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

      let w_us = w as usize;
      let ch = *channels as usize;
      let mut new_data =
        Vec::with_capacity((crop_w as usize) * (crop_h as usize) * ch);
      for y in first_row..=last_row {
        for x in first_col..=last_col {
          let base = ((y as usize) * w_us + (x as usize)) * ch;
          for c in 0..ch {
            new_data.push(data[base + c]);
          }
        }
      }
      // Determine the image type from the first arm; this branch
      // already requires args[0] to be Expr::Image.
      let image_type = match &args[0] {
        Expr::Image { image_type, .. } => *image_type,
        _ => unreachable!(),
      };
      Ok(Expr::Image {
        color_space: None,
        display_size: None,
        width: crop_w,
        height: crop_h,
        channels: *channels,
        data: Arc::new(new_data),
        image_type,
      })
    }
    _ => Ok(unevaluated("ImageTrim", args)),
  }
}

// ─── Advanced functions (Phase 3) ──────────────────────────────────────────

/// EdgeDetect[image] / EdgeDetect[image, r] / EdgeDetect[image, r, t] —
/// mark the edge pixels of an image as a `"Bit"` image. This is the
/// Canny pipeline on top of `GradientFilter`'s derivative-of-Gaussian
/// gradient (shared through `plane_gradient`, so there is one gradient
/// in the codebase): thin the gradient magnitude to single-pixel ridges
/// by suppressing every pixel that a neighbour along the gradient
/// direction beats, then keep what clears the threshold. The default
/// threshold is `FindThreshold[GradientFilter[image, r]]`, the same
/// 256-bin cluster-variance split wolframscript uses; an explicit `t`
/// is compared against the gradient magnitude directly.
///
/// Trailing rules are options: `Method -> "Canny"` (or `Automatic`) and
/// `"StraightEdges" -> 0` describe what is computed here; any other
/// option value leaves the call unevaluated.
pub fn edge_detect_ast(all_args: &[Expr]) -> Result<Expr, InterpreterError> {
  let option_start = all_args
    .iter()
    .position(|a| matches!(a, Expr::Rule { .. } | Expr::RuleDelayed { .. }))
    .unwrap_or(all_args.len());
  let (args, opts) = all_args.split_at(option_start);
  if opts
    .iter()
    .any(|o| !matches!(o, Expr::Rule { .. } | Expr::RuleDelayed { .. }))
  {
    return Ok(unevaluated("EdgeDetect", all_args));
  }
  let is_str = |e: &Expr, v: &str| matches!(e, Expr::String(s) if s == v);
  let supported = opts.iter().all(|o| match o {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => match pattern.as_ref() {
      Expr::Identifier(k) if k == "Method" => {
        is_str(replacement, "Canny")
          || matches!(replacement.as_ref(), Expr::Identifier(a) if a == "Automatic")
      }
      p if is_str(p, "StraightEdges") => {
        crate::functions::math_ast::try_eval_to_f64(replacement) == Some(0.0)
          || matches!(replacement.as_ref(), Expr::Identifier(a) if a == "Automatic")
      }
      _ => false,
    },
    _ => false,
  });
  if !supported {
    return Ok(unevaluated("EdgeDetect", all_args));
  }
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "EdgeDetect expects 1 to 3 arguments".into(),
    ));
  }

  // The range is `GradientFilter`'s radius, specified exactly as `Blur`'s
  // is: one number, or `{r_rows, r_columns}` for a different range along
  // each axis. Unlike `Blur` it is not capped at half the image.
  let Some((rows, columns)) = blur_radius_pair("EdgeDetect", args) else {
    return Ok(unevaluated("EdgeDetect", args));
  };
  let Expr::Image {
    width,
    height,
    channels,
    data,
    ..
  } = &args[0]
  else {
    unreachable!("blur_radius_pair checked for an image")
  };
  let user_threshold = match args.get(2) {
    None => None,
    Some(spec) => {
      if let Some(value) = crate::functions::math_ast::try_eval_to_f64(spec) {
        Some(value)
      } else {
        crate::emit_message(&format!(
          "EdgeDetect::bdthr: Invalid threshold specification {}.",
          crate::syntax::format_expr(spec, crate::syntax::ExprForm::Output)
        ));
        return Ok(unevaluated("EdgeDetect", args));
      }
    }
  };

  let w = *width as usize;
  let h = *height as usize;
  let ch = *channels as usize;

  // Edges are found on the luminance of the image.
  let gray: Vec<f64> = (0..w * h)
    .map(|i| {
      if ch == 1 {
        data[i]
      } else {
        let base = i * ch;
        0.299 * data[base] + 0.587 * data[base + 1] + 0.114 * data[base + 2]
      }
    })
    .collect();

  let (dx, dy) = plane_gradient(
    &gray,
    w,
    h,
    gaussian_axis_from_radius(rows),
    gaussian_axis_from_radius(columns),
  );
  let magnitudes: Vec<f64> = (0..w * h).map(|i| dx[i].hypot(dy[i])).collect();

  let threshold =
    user_threshold.unwrap_or_else(|| cluster_variance_threshold(&magnitudes));

  // Non-maximum suppression thins each ridge to one pixel: keep a pixel
  // only if neither neighbour along the quantized gradient direction is
  // stronger.
  let clamp_y = |y: i32| y.clamp(0, h as i32 - 1) as usize;
  let clamp_x = |x: i32| x.clamp(0, w as i32 - 1) as usize;
  let mut result = vec![0.0_f64; w * h];
  for y in 0..h {
    for x in 0..w {
      let index = y * w + x;
      let magnitude = magnitudes[index];
      if magnitude <= 0.0 || magnitude < threshold {
        continue;
      }
      let (step_y, step_x) =
        quantize_gradient_direction(dy[index].atan2(dx[index]));
      let ahead =
        magnitudes[clamp_y(y as i32 + step_y) * w + clamp_x(x as i32 + step_x)];
      let behind =
        magnitudes[clamp_y(y as i32 - step_y) * w + clamp_x(x as i32 - step_x)];
      if magnitude >= ahead && magnitude >= behind {
        result[index] = 1.0;
      }
    }
  }

  Ok(Expr::Image {
    color_space: None,
    display_size: None,
    width: *width,
    height: *height,
    channels: 1,
    data: Arc::new(result),
    image_type: ImageType::Bit,
  })
}

/// The threshold `FindThreshold` picks for the gradient magnitudes, so
/// `EdgeDetect` keeps the same edges wolframscript does.
fn cluster_variance_threshold(values: &[f64]) -> f64 {
  match otsu_threshold(values, ImageType::Real64, false) {
    Some(t) => crate::functions::math_ast::try_eval_to_f64(&t).unwrap_or(0.0),
    None => values.first().map_or(0.0, |v| v.max(0.0)),
  }
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
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated("DominantColors", args));
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
      let num_pixels = w * h;

      // Single-channel (grayscale) image: cluster luminance values and
      // emit `GrayLevel[v]` per cluster centre.
      if ch == 1 {
        let mut grays: Vec<[f64; 1]> = Vec::with_capacity(num_pixels);
        for i in 0..num_pixels {
          grays.push([data[i]]);
        }
        let k = n.min(num_pixels);
        let centers = kmeans_1d(&grays, k, 20);
        let colors: Vec<Expr> = centers
          .iter()
          .map(|c| call1("GrayLevel", Expr::Real(c[0])))
          .collect();
        return Ok(Expr::List(colors.into()));
      }

      if ch < 3 {
        return Err(InterpreterError::EvaluationError(
          "DominantColors: requires an RGB or RGBA image".into(),
        ));
      }

      // Collect RGB pixels
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
        .map(|c| {
          call(
            "RGBColor",
            vec![Expr::Real(c[0]), Expr::Real(c[1]), Expr::Real(c[2])],
          )
        })
        .collect();

      Ok(Expr::List(colors.into()))
    }
    _ => unreachable!("non-Image first arg handled above"),
  }
}

/// 1-D k-means for grayscale `DominantColors`. Centres are sorted in
/// ascending value order so the output list is deterministic.
fn kmeans_1d(pixels: &[[f64; 1]], k: usize, max_iters: usize) -> Vec<[f64; 1]> {
  if pixels.is_empty() || k == 0 {
    return vec![];
  }
  let step = pixels.len().max(1) / k.max(1);
  let mut centers: Vec<[f64; 1]> = (0..k)
    .map(|i| pixels[(i * step).min(pixels.len() - 1)])
    .collect();
  let mut assignments = vec![0usize; pixels.len()];

  for _ in 0..max_iters {
    let mut changed = false;
    for (i, pixel) in pixels.iter().enumerate() {
      let mut best = 0;
      let mut best_dist = f64::MAX;
      for (j, center) in centers.iter().enumerate() {
        let dist = (pixel[0] - center[0]).powi(2);
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
    let mut sums = vec![[0.0_f64; 1]; k];
    let mut counts = vec![0usize; k];
    for (i, pixel) in pixels.iter().enumerate() {
      let c = assignments[i];
      sums[c][0] += pixel[0];
      counts[c] += 1;
    }
    for j in 0..k {
      if counts[j] > 0 {
        centers[j][0] = sums[j][0] / counts[j] as f64;
      }
    }
  }
  centers.sort_by(|a, b| {
    a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal)
  });
  centers
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

/// ImageApply[f, img] — apply f to each pixel. Grayscale images pass
/// the scalar pixel value as f's single argument. Multi-channel images
/// pass the channel list (as a List) as f's single argument; if f
/// returns a list, the output keeps that channel count, otherwise the
/// output is single-channel (grayscale).
///
/// ImageApply[f, img, Masking -> mask] restricts `f` to the pixels
/// selected by `mask`; every other pixel passes through unchanged.
pub fn image_apply_ast(
  args: &[Expr],
  eval_fn: &dyn Fn(&Expr) -> Result<Expr, InterpreterError>,
) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "ImageApply expects 2 or 3 arguments".into(),
    ));
  }

  let func = &args[0];
  // Any function: a pure function, a symbol, an InterpolatingFunction, ….
  let apply = |arg: Expr| -> Result<Expr, InterpreterError> {
    match func {
      Expr::Function { body } => {
        eval_fn(&crate::syntax::substitute_slots(body, &[arg]))
      }
      _ => crate::evaluator::function_application::apply_function_to_arg(
        func, &arg,
      ),
    }
  };

  let Expr::Image {
    color_space,
    display_size: _,
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[1]
  else {
    return Err(InterpreterError::EvaluationError(
      "ImageApply: second argument is not an Image".into(),
    ));
  };
  let ch = *channels as usize;
  let w = *width as usize;
  let h = *height as usize;
  let num_pixels = w * h;

  // Options: `Masking -> region` and `Interleaving -> False` (f sees each
  // channel value on its own instead of the list of a pixel's channels).
  let mut mask: Option<Vec<bool>> = None;
  let mut interleaving = true;
  for opt in &args[2..] {
    let Expr::Rule {
      pattern,
      replacement,
    } = opt
    else {
      return Err(InterpreterError::EvaluationError(
        "ImageApply: unsupported option".into(),
      ));
    };
    match pattern.as_ref() {
      Expr::Identifier(s) if s == "Masking" => {
        mask = resolve_mask(replacement, w, h)?;
      }
      Expr::Identifier(s) if s == "Interleaving" => {
        interleaving =
          !matches!(replacement.as_ref(), Expr::Identifier(v) if v == "False");
      }
      _ => {
        return Err(InterpreterError::EvaluationError(
          "ImageApply: unsupported option".into(),
        ));
      }
    }
  }
  // Without interleaving a pixel's channels are mapped one value at a time.
  let apply = |arg: Expr| -> Result<Expr, InterpreterError> {
    match &arg {
      Expr::List(values) if !interleaving => Ok(Expr::List(
        values
          .iter()
          .map(|v| apply(v.clone()))
          .collect::<Result<Vec<_>, _>>()?
          .into(),
      )),
      _ => apply(arg),
    }
  };
  let included = |i: usize| -> bool { mask.as_ref().is_none_or(|m| m[i]) };

  // For Real32 images, snap pixel values to their f32 representation
  // before passing them to f. wolframscript's image arithmetic is done
  // in f32 throughout; without the snap, a function like `#^2 &` would
  // square the f64 value and round differently on the way back into the
  // f32 buffer.
  let is_real32 = matches!(image_type, ImageType::Real32);
  let snap = |v: f64| -> f64 { if is_real32 { (v as f32) as f64 } else { v } };

  // A grayscale pixel is passed to `f` as a bare scalar; a multi-channel
  // pixel is passed as a List of its channel values. Either shape of `f`
  // may return a scalar (grayscale output) or a List (multi-channel
  // output) independent of the input shape — e.g. a grayscale mask image
  // colorized by a function returning `{r, g, b}` per pixel, as a
  // Demonstrations Project puzzle-reveal pattern does. So the output
  // channel count is probed from the first result rather than assumed
  // from the input.
  let pixel_input = |i: usize| -> Expr {
    if ch == 1 {
      return Expr::Real(snap(data[i]));
    }
    let base = i * ch;
    Expr::List(
      (0..ch)
        .map(|c| Expr::Real(snap(data[base + c])))
        .collect::<Vec<_>>()
        .into(),
    )
  };

  // Probe the first *included* pixel (pixel 0 when there's no mask) to
  // determine the output channel count. An image masked out everywhere
  // has nothing to probe, so it keeps the source channel count.
  let first_included = (0..num_pixels).find(|&i| included(i));
  let first_result =
    first_included.map(|i| apply(pixel_input(i))).transpose()?;
  let out_ch = match &first_result {
    Some(Expr::List(vs)) => vs.len(),
    Some(_) => 1,
    None => ch,
  };
  if first_result.is_some() && out_ch == 0 {
    return Err(InterpreterError::EvaluationError(
      "ImageApply: function returned an empty list".into(),
    ));
  }

  let push_result =
    |result: &Expr, dst: &mut Vec<f64>| -> Result<(), InterpreterError> {
      match result {
        Expr::List(vs) => {
          for v in vs {
            dst.push(expr_to_f64(v)?);
          }
        }
        other => dst.push(expr_to_f64(other)?),
      }
      Ok(())
    };

  let mut new_data: Vec<f64> = Vec::with_capacity(num_pixels * out_ch);
  for i in 0..num_pixels {
    if !included(i) {
      // Masked-out pixels pass through unchanged. If `f` changes the
      // channel count for included pixels, reuse the last source
      // channel to pad rather than fail on this rare combination.
      let base = i * ch;
      for c in 0..out_ch {
        new_data.push(data[base + c.min(ch - 1)]);
      }
      continue;
    }
    if first_included == Some(i) {
      push_result(first_result.as_ref().unwrap(), &mut new_data)?;
    } else {
      let result = apply(pixel_input(i))?;
      push_result(&result, &mut new_data)?;
    }
  }

  // The result stays in the source's colour space (an HSB image mapped
  // through `f` is still HSB) unless `f` changed the channel count.
  Ok(Expr::Image {
    color_space: if out_ch == ch { *color_space } else { None },
    display_size: None,
    width: *width,
    height: *height,
    channels: out_ch as u8,
    data: Arc::new(new_data),
    image_type: *image_type,
  })
}

/// Extract `(r, g, b, optional_alpha)` from a color directive (RGBColor
/// or GrayLevel). Returns None for anything else, including named
/// constants that haven't already been resolved to RGBColor.
pub(crate) fn color_directive_to_rgb(
  e: &Expr,
) -> Option<(f64, f64, f64, Option<f64>)> {
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
    // Hue[h] uses full saturation and brightness; Hue[h, s, b(, a)] is the
    // general HSB form.
    "Hue" if args.len() == 1 => {
      let (r, g, b) = hsv_to_rgb(to_f(&args[0])?, 1.0, 1.0);
      Some((r, g, b, None))
    }
    "Hue" if args.len() == 3 || args.len() == 4 => {
      let (r, g, b) =
        hsv_to_rgb(to_f(&args[0])?, to_f(&args[1])?, to_f(&args[2])?);
      let alpha = if args.len() == 4 {
        Some(to_f(&args[3])?)
      } else {
        None
      };
      Some((r, g, b, alpha))
    }
    // `LABColor[l, a, b(, alpha)]` (scaled by 1/100, D50) and
    // `XYZColor[x, y, z(, alpha)]` (D50), back to sRGB.
    "LABColor" | "XYZColor" if args.len() == 3 || args.len() == 4 => {
      let (p, q, t) = (to_f(&args[0])?, to_f(&args[1])?, to_f(&args[2])?);
      let (r, g, b) = if name == "LABColor" {
        lab_to_srgb(p, q, t)
      } else {
        xyz_d50_to_srgb(p, q, t)
      };
      let alpha = if args.len() == 4 {
        Some(to_f(&args[3])?)
      } else {
        None
      };
      Some((r, g, b, alpha))
    }
    "CMYKColor" if args.len() == 4 || args.len() == 5 => {
      let c = to_f(&args[0])?;
      let m = to_f(&args[1])?;
      let y = to_f(&args[2])?;
      let k = to_f(&args[3])?;
      let r = (1.0 - c) * (1.0 - k);
      let g = (1.0 - m) * (1.0 - k);
      let b = (1.0 - y) * (1.0 - k);
      let alpha = if args.len() == 5 {
        Some(to_f(&args[4])?)
      } else {
        None
      };
      Some((r, g, b, alpha))
    }
    _ => None,
  }
}

/// The pixels of an image as they are displayed: an HSB or CMYK image is
/// decoded to RGB (keeping an HSBA alpha channel), anything else is shown as
/// stored. Returns the channel count of the returned data.
pub fn rgb_pixels<'a>(
  color_space: Option<&str>,
  channels: u8,
  data: &'a [f64],
) -> (u8, Cow<'a, [f64]>) {
  let ch = channels as usize;
  let (is_cmyk, has_alpha) = match (color_space, ch) {
    (Some("HSB"), 3) => (false, false),
    (Some("HSB"), 4) => (false, true),
    (Some("CMYK"), 4) => (true, false),
    _ => return (channels, Cow::Borrowed(data)),
  };
  let mut rgb = Vec::with_capacity(data.len());
  for px in data.chunks_exact(ch) {
    let (r, g, b) = if is_cmyk {
      let k = 1.0 - px[3];
      ((1.0 - px[0]) * k, (1.0 - px[1]) * k, (1.0 - px[2]) * k)
    } else {
      hsv_to_rgb(px[0], px[1], px[2])
    };
    rgb.extend_from_slice(&[r, g, b]);
    if has_alpha {
      rgb.push(px[3]);
    }
  }
  (if has_alpha { 4 } else { 3 }, Cow::Owned(rgb))
}

/// HSB/HSV → RGB. `h`, `s`, `v` in [0, 1]; the hue wraps modulo 1.
fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (f64, f64, f64) {
  let h6 = h.rem_euclid(1.0) * 6.0;
  let i = h6.floor();
  let f = h6 - i;
  let p = v * (1.0 - s);
  let q = v * (1.0 - s * f);
  let t = v * (1.0 - s * (1.0 - f));
  match (i as i64) % 6 {
    0 => (v, t, p),
    1 => (q, v, p),
    2 => (p, v, t),
    3 => (p, q, v),
    4 => (t, p, v),
    _ => (v, p, q),
  }
}

/// RGB → HSB/HSV, each component in [0, 1].
fn rgb_to_hsv(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
  let max = r.max(g).max(b);
  let min = r.min(g).min(b);
  let delta = max - min;
  let v = max;
  let s = if max == 0.0 { 0.0 } else { delta / max };
  let h = if delta == 0.0 {
    0.0
  } else if max == r {
    (((g - b) / delta).rem_euclid(6.0)) / 6.0
  } else if max == g {
    ((b - r) / delta + 2.0) / 6.0
  } else {
    ((r - g) / delta + 4.0) / 6.0
  };
  (h, s, v)
}

/// RGB → CMYK, each component in [0, 1].
fn rgb_to_cmyk(r: f64, g: f64, b: f64) -> (f64, f64, f64, f64) {
  let k = 1.0 - r.max(g).max(b);
  if (1.0 - k).abs() < 1e-12 {
    (0.0, 0.0, 0.0, k)
  } else {
    (
      (1.0 - r - k) / (1.0 - k),
      (1.0 - g - k) / (1.0 - k),
      (1.0 - b - k) / (1.0 - k),
      k,
    )
  }
}

/// ColorConvert[img, "Grayscale"] - Convert between color spaces
/// The static name a converted image records as its colour space, or `None`
/// for one that is not named.
fn color_space_tag(space: &str) -> Option<&'static str> {
  Some(match space {
    "Grayscale" => "Grayscale",
    "RGB" => "RGB",
    "CMYK" => "CMYK",
    "HSB" | "Hue" => "HSB",
    "LAB" => "LAB",
    "LCH" => "LCH",
    "LUV" => "LUV",
    "XYZ" => "XYZ",
    _ => return None,
  })
}

/// The same image, recorded as being in `tag`'s colour space.
fn retag(image: &Expr, tag: Option<&'static str>) -> Expr {
  match image {
    Expr::Image {
      width,
      height,
      channels,
      data,
      image_type,
      ..
    } => Expr::Image {
      color_space: tag,
      display_size: None,
      width: *width,
      height: *height,
      channels: *channels,
      data: data.clone(),
      image_type: *image_type,
    },
    other => other.clone(),
  }
}

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
  // Wolfram accepts color-space names case-insensitively — older notebooks
  // commonly write "GrayScale" — so normalize to the canonical spelling.
  // The color-directive heads name their spaces too ("RGBColor" is RGB,
  // "GrayLevel" is Grayscale, …), as in wolframscript.
  let target_space = if target_space.eq_ignore_ascii_case("grayscale")
    || target_space == "GrayLevel"
  {
    "Grayscale"
  } else if target_space.eq_ignore_ascii_case("rgb")
    || target_space == "RGBColor"
  {
    "RGB"
  } else if target_space.eq_ignore_ascii_case("cmyk")
    || target_space == "CMYKColor"
  {
    "CMYK"
  } else if target_space.eq_ignore_ascii_case("hsb") {
    "HSB"
  } else if target_space.eq_ignore_ascii_case("hue") {
    "Hue"
  } else if target_space == "LABColor" {
    "LAB"
  } else if target_space == "XYZColor" {
    "XYZ"
  } else if target_space == "LCHColor" {
    "LCH"
  } else if target_space == "LUVColor" {
    "LUV"
  } else {
    target_space
  };

  // A list of colors converts each one.
  if let Expr::List(items) = &args[0]
    && !items.is_empty()
    && items.iter().all(|c| color_directive_to_rgb(c).is_some())
  {
    return items
      .iter()
      .map(|c| color_convert_ast(&[c.clone(), args[1].clone()]))
      .collect::<Result<Vec<_>, _>>()
      .map(|v| Expr::List(v.into()));
  }

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
        return Ok(call("GrayLevel", gargs));
      }
      "RGB" => {
        let mut rargs = vec![Expr::Real(r), Expr::Real(g), Expr::Real(b)];
        if let Some(a) = alpha {
          rargs.push(Expr::Real(a));
        }
        return Ok(call("RGBColor", rargs));
      }
      "CMYK" => {
        let (c, m, y, k) = rgb_to_cmyk(r, g, b);
        let mut cargs =
          vec![Expr::Real(c), Expr::Real(m), Expr::Real(y), Expr::Real(k)];
        if let Some(a) = alpha {
          cargs.push(Expr::Real(a));
        }
        return Ok(call("CMYKColor", cargs));
      }
      // "HSB" produces the Hue[h, s, b(, a)] directive.
      "HSB" | "Hue" => {
        let (h, s, v) = rgb_to_hsv(r, g, b);
        let mut hargs = vec![Expr::Real(h), Expr::Real(s), Expr::Real(v)];
        if let Some(a) = alpha {
          hargs.push(Expr::Real(a));
        }
        return Ok(call("Hue", hargs));
      }
      "LAB" | "XYZ" => {
        let (p, q, t) = if target_space == "LAB" {
          srgb_to_lab(r, g, b)
        } else {
          linear_rgb_to_xyz_d50(
            srgb_to_linear(r),
            srgb_to_linear(g),
            srgb_to_linear(b),
          )
        };
        let mut cargs = vec![Expr::Real(p), Expr::Real(q), Expr::Real(t)];
        if let Some(a) = alpha {
          cargs.push(Expr::Real(a));
        }
        let head = if target_space == "LAB" {
          "LABColor"
        } else {
          "XYZColor"
        };
        return Ok(call(head, cargs));
      }
      _ => {}
    }
  }
  if !matches!(
    target_space,
    "Grayscale"
      | "RGB"
      | "CMYK"
      | "HSB"
      | "Hue"
      | "LAB"
      | "XYZ"
      | "LCH"
      | "LUV"
  ) {
    crate::emit_message(&format!(
      "ColorConvert::imgcstype: {target_space} is an invalid color space specification."
    ));
    return Ok(unevaluated("ColorConvert", args));
  }

  // An HSB or CMYK source is first decoded to RGB, so every target below
  // starts from RGB (or gray) pixels; converting to its own space is a no-op.
  let decoded;
  let source = match &args[0] {
    Expr::Image {
      color_space: Some(space @ ("HSB" | "CMYK")),
      width,
      height,
      channels,
      data,
      image_type,
      ..
    } => {
      if color_space_tag(target_space) == Some(*space) {
        return Ok(args[0].clone());
      }
      match rgb_pixels(Some(space), *channels, data) {
        (out_ch, Cow::Owned(rgb)) => {
          decoded = Expr::Image {
            color_space: Some("RGB"),
            display_size: None,
            width: *width,
            height: *height,
            channels: out_ch,
            data: Arc::new(rgb),
            image_type: *image_type,
          };
          &decoded
        }
        _ => &args[0],
      }
    }
    other => other,
  };

  match source {
    Expr::Image {
      color_space: _,
      display_size: _,
      width,
      height,
      channels,
      data,
      image_type,
    } => {
      let w = *width as usize;
      let h = *height as usize;
      let ch = *channels as usize;
      // The converted image carries the space it was converted to.
      let tag = color_space_tag(target_space);

      match target_space {
        "Grayscale" => {
          if ch == 1 {
            return Ok(retag(source, tag)); // Already grayscale
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
            color_space: tag,
            display_size: None,
            width: *width,
            height: *height,
            channels: 1,
            data: Arc::new(new_data),
            image_type: *image_type,
          })
        }
        "RGB" => {
          if ch == 3 {
            return Ok(retag(source, tag)); // Already RGB
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
              color_space: tag,
              display_size: None,
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
              color_space: tag,
              display_size: None,
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
        // "HSB" and "CMYK" are per-pixel recodings of the RGB triple, so
        // they share `rgb_to_hsv` / `rgb_to_cmyk` with the color-directive
        // branch above. A grayscale source is read as the RGB gray it
        // stands for; an alpha channel rides along unchanged, matching
        // Wolfram's HSBA/CMYKA images.
        "HSB" | "Hue" | "CMYK" => {
          if ch != 1 && ch != 3 && ch != 4 {
            return Err(InterpreterError::EvaluationError(
              "ColorConvert: unsupported channel count".into(),
            ));
          }
          let has_alpha = ch == 4;
          let out_ch = if target_space == "CMYK" { 4 } else { 3 };
          let total = out_ch + usize::from(has_alpha);
          let mut new_data = Vec::with_capacity(w * h * total);
          for i in 0..(w * h) {
            let base = i * ch;
            let (r, g, b) = if ch == 1 {
              (data[base], data[base], data[base])
            } else {
              (data[base], data[base + 1], data[base + 2])
            };
            if target_space == "CMYK" {
              let (c, m, y, k) = rgb_to_cmyk(r, g, b);
              new_data.extend_from_slice(&[c, m, y, k]);
            } else {
              let (hue, s, v) = rgb_to_hsv(r, g, b);
              new_data.extend_from_slice(&[hue, s, v]);
            }
            if has_alpha {
              new_data.push(data[base + 3]);
            }
          }
          Ok(Expr::Image {
            color_space: tag,
            display_size: None,
            width: *width,
            height: *height,
            channels: total as u8,
            data: Arc::new(new_data),
            image_type: *image_type,
          })
        }
        _ => Err(InterpreterError::EvaluationError(format!(
          "ColorConvert: unsupported color space \"{target_space}\""
        ))),
      }
    }
    // Anything else (a symbol, a non-color expression) is reported and the
    // call stays unevaluated, as in wolframscript.
    other => {
      crate::emit_message(&format!(
        "ColorConvert::ccvinput: {} should be a valid image, a color directive, a list of machine-sized real numbers of length up to 5 or a list of such objects.",
        crate::syntax::expr_to_string(other)
      ));
      Ok(unevaluated("ColorConvert", args))
    }
  }
}

/// ImageCompose[img, overlay] / ImageCompose[img, {overlay, α}] —
/// overlay an image on top of img, centered. The output has the same
/// dimensions, channel count and image type as `img`. With an alpha
/// argument, the overlapping region is blended as (1 - α)*bg + α*ov;
/// without one the overlay replaces the background pixels.
/// A position spec of `ImageCompose` — `{x, y}` in image coordinates (origin
/// bottom-left), or the names `Left`/`Right`/`Center` and `Bottom`/`Top`
/// (alone for one axis, or in an `{x, y}` pair) — resolved against an image
/// of the given size.
fn image_compose_position(spec: &Expr, w: f64, h: f64) -> Option<(f64, f64)> {
  let axis = |e: &Expr, len: f64, low: &str, high: &str| -> Option<f64> {
    match e {
      Expr::Identifier(n) | Expr::String(n) => match n.as_str() {
        "Center" => Some(len / 2.0),
        n if n == low => Some(0.0),
        n if n == high => Some(len),
        _ => None,
      },
      other => crate::functions::math_ast::try_eval_to_f64(other),
    }
  };
  match spec {
    // Scaled[{sx, sy}]: fractions of the image's width and height.
    Expr::FunctionCall { name, args }
      if name == "Scaled" && args.len() == 1 =>
    {
      let Expr::List(p) = &args[0] else {
        return None;
      };
      if p.len() != 2 {
        return None;
      }
      let f = crate::functions::math_ast::try_eval_to_f64;
      Some((f(&p[0])? * w, f(&p[1])? * h))
    }
    Expr::List(p) if p.len() == 2 => Some((
      axis(&p[0], w, "Left", "Right")?,
      axis(&p[1], h, "Bottom", "Top")?,
    )),
    Expr::Identifier(n) | Expr::String(n) => match n.as_str() {
      "Center" => Some((w / 2.0, h / 2.0)),
      "Left" => Some((0.0, h / 2.0)),
      "Right" => Some((w, h / 2.0)),
      "Bottom" => Some((w / 2.0, 0.0)),
      "Top" => Some((w / 2.0, h)),
      _ => None,
    },
    _ => None,
  }
}

pub fn image_compose_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 5 {
    return Err(InterpreterError::EvaluationError(
      "ImageCompose expects 2 to 5 arguments".into(),
    ));
  }
  // ImageCompose[image, overlay, pos, opos, {a, b, c}]: the Porter–Duff
  // weights of the overlay-only, image-only and overlapping parts; the
  // default {1, 1, 1} draws the overlay over the image.
  let operator = match args.get(4) {
    None => (1.0, 1.0, 1.0),
    Some(Expr::List(w)) if w.len() == 3 => {
      let f = crate::functions::math_ast::try_eval_to_f64;
      match (f(&w[0]), f(&w[1]), f(&w[2])) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => return Ok(unevaluated("ImageCompose", args)),
      }
    }
    Some(_) => return Ok(unevaluated("ImageCompose", args)),
  };

  // Parse the overlay spec: either an Image, or {Image, α}.
  let (overlay_expr, alpha) = match &args[1] {
    Expr::List(items) if items.len() == 2 => {
      let a = expr_to_f64(&items[1])?;
      (&items[0], Some(a))
    }
    _ => (&args[1], None),
  };
  // A graphic on either side is rasterized first, as wolframscript does.
  let base_raster;
  let base: &Expr = match &args[0] {
    Expr::Image { .. } => &args[0],
    other => match coerce_graphics_to_image(other) {
      Some(img) => {
        base_raster = img;
        &base_raster
      }
      None => &args[0],
    },
  };
  let overlay_raster;
  let overlay_expr: &Expr = match overlay_expr {
    Expr::Image { .. } => overlay_expr,
    other => match coerce_graphics_to_image(other) {
      Some(img) => {
        overlay_raster = img;
        &overlay_raster
      }
      None => overlay_expr,
    },
  };
  if !matches!(base, Expr::Image { .. })
    || !matches!(overlay_expr, Expr::Image { .. })
  {
    let bad = if matches!(base, Expr::Image { .. }) {
      overlay_expr
    } else {
      base
    };
    crate::emit_message(&format!(
      "ImageCompose::imginv: Expecting an image or graphics instead of {}.",
      expr_to_string(bad)
    ));
    return Ok(unevaluated("ImageCompose", args));
  }

  let (
    Expr::Image {
      color_space: cs1,
      display_size: _,
      width: w1,
      height: h1,
      channels: ch1,
      data: data1,
      image_type: _,
    },
    Expr::Image {
      width: w2,
      height: h2,
      channels: ch2,
      data: data2,
      color_space: cs2,
      ..
    },
  ) = (base, overlay_expr)
  else {
    return Ok(unevaluated("ImageCompose", args));
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
  let (mut offset_x, mut offset_y) =
    (bw / 2 - ow / 2, (bh + 1) / 2 - (oh + 1) / 2);
  // ImageCompose[image, overlay, pos, opos] puts the overlay's `opos` point
  // (default: its center) on the image's `pos` point (default: its center).
  if args.len() >= 3 {
    let base_pos = image_compose_position(&args[2], bw as f64, bh as f64);
    let over_pos = match args.get(3) {
      Some(spec) => image_compose_position(spec, ow as f64, oh as f64),
      None => Some((ow as f64 / 2.0, oh as f64 / 2.0)),
    };
    let (Some((px, py)), Some((qx, qy))) = (base_pos, over_pos) else {
      return Ok(unevaluated("ImageCompose", args));
    };
    // Image y grows upward; the buffer's rows run top-down.
    offset_x = (px - qx).round() as i64;
    offset_y = (bh as f64 - (py - qy) - oh as f64).round() as i64;
  }

  // Porter–Duff compositing on the colour channels: with image and overlay
  // opacities αi, αo, the result has opacity a αo (1 - αi) + b αi (1 - αo)
  // + c αi αo, and the colours weighted the same way (the overlap showing
  // the overlay). A {image, α} overlay is faded by α. An image without an
  // alpha channel keeps none; its pixels are the premultiplied result.
  let base_alpha = has_alpha_channel(*ch1, *cs1);
  let over_alpha = has_alpha_channel(*ch2, *cs2);
  let bcolors = if base_alpha { bch - 1 } else { bch };
  let ocolors = if over_alpha { och - 1 } else { och };
  // An overlay with an alpha channel gives the result one too.
  let out_alpha = base_alpha || over_alpha;
  let out_ch = bcolors + usize::from(out_alpha);
  let (wa, wb, wc) = operator;
  let fade = alpha.unwrap_or(1.0);
  // Pixel values are single-precision; the arithmetic is done on them in
  // double precision and the result rounded back, as wolframscript does.
  let px = |v: f64| v as f32 as f64;
  let n = (bw * bh) as usize;
  let mut out: Vec<f64> = Vec::with_capacity(n * out_ch);
  for by in 0..bh {
    for bx in 0..bw {
      let base_idx = (by as usize * *w1 as usize + bx as usize) * bch;
      let (ox, oy) = (bx - offset_x, by - offset_y);
      let over_idx = ((0..ow).contains(&ox) && (0..oh).contains(&oy))
        .then(|| (oy as usize * *w2 as usize + ox as usize) * och);
      let ai = if base_alpha {
        px(data1[base_idx + bch - 1])
      } else {
        1.0
      };
      let ao = match over_idx {
        Some(i) => {
          (if over_alpha {
            px(data2[i + och - 1])
          } else {
            1.0
          }) * fade
        }
        None => 0.0,
      };
      let ar = wa * ao * (1.0 - ai) + wb * ai * (1.0 - ao) + wc * ai * ao;
      for c in 0..bcolors {
        let ci = px(data1[base_idx + c]);
        let co = match over_idx {
          Some(i) => px(data2[i + c.min(ocolors - 1)]),
          None => 0.0,
        };
        let premult = wa * co * ao * (1.0 - ai)
          + wb * ci * ai * (1.0 - ao)
          + wc * co * ai * ao;
        let value = if out_alpha {
          if ar > 0.0 { premult / ar } else { 0.0 }
        } else {
          premult
        };
        out.push(px(value));
      }
      if out_alpha {
        out.push(px(ar));
      }
    }
  }

  Ok(Expr::Image {
    color_space: if out_alpha {
      cs1.or(*cs2).or(match bcolors {
        1 => Some("Grayscale"),
        3 => Some("RGB"),
        _ => None,
      })
    } else {
      *cs1
    },
    display_size: None,
    width: *w1,
    height: *h1,
    channels: out_ch as u8,
    data: Arc::new(out),
    image_type: ImageType::Real32,
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
      "{name} expects exactly 2 arguments"
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
      color_space: _,
      display_size: _,
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
    // An image carrying an alpha channel combines with its opaque
    // counterpart (an RGBA rasterized graphic with an RGB photo, say) on
    // the shared color channels; a grayscale image broadcasts over a
    // color one.
    let alpha_pair = matches!((*ch1, *ch2), (2, 1) | (1, 2) | (4, 3) | (3, 4));
    let gray_pair = (*ch1 == 1) != (*ch2 == 1) && !alpha_pair;
    if ch1 != ch2 && !alpha_pair && !gray_pair {
      return Err(InterpreterError::EvaluationError(format!(
        "{name}: images must have the same number of channels"
      )));
    }
    let out_ch = if gray_pair {
      *ch1.max(ch2) as usize
    } else {
      *ch1.min(ch2) as usize
    };
    let (c1, c2) = (*ch1 as usize, *ch2 as usize);
    let (w1, h1, w2, h2) = (*w1 as i64, *h1 as i64, *w2 as i64, *h2 as i64);
    let is_r32 = matches!(t1, ImageType::Real32);
    // Images of different sizes: the result keeps the first image's size,
    // the second is centered over it, and pixels it does not cover keep
    // the first image's values.
    let (dx, dy) = ((w1 - w2).div_euclid(2), (h1 - h2).div_euclid(2));
    let mut new_data: Vec<f64> =
      Vec::with_capacity((w1 * h1) as usize * out_ch);
    for y in 0..h1 {
      let y2 = y - dy;
      for x in 0..w1 {
        let x2 = x - dx;
        let i1 = (y * w1 + x) as usize * c1;
        let covered = (0..h2).contains(&y2) && (0..w2).contains(&x2);
        let i2 = if covered {
          Some((y2 * w2 + x2) as usize * c2)
        } else {
          None
        };
        for k in 0..out_ch {
          let a = data1[i1 + if c1 == 1 { 0 } else { k }];
          new_data.push(match i2 {
            Some(i2) => {
              apply(a, data2[i2 + if c2 == 1 { 0 } else { k }], is_r32)
            }
            None => a,
          });
        }
      }
    }
    // Integer-typed images saturate at their range limits.
    let new_data =
      if matches!(t1, ImageType::Bit | ImageType::Byte | ImageType::Bit16) {
        new_data.into_iter().map(|v| v.clamp(0.0, 1.0)).collect()
      } else {
        new_data
      };
    return Ok(Expr::Image {
      color_space: None,
      display_size: None,
      width: w1 as u32,
      height: h1 as u32,
      channels: out_ch as u8,
      data: Arc::new(new_data),
      image_type: *t1,
    });
  }

  // (Image, {c1, c2, ...}) / ({c1, c2, ...}, Image) — one constant per
  // channel, e.g. the per-channel gain of a white balance.
  for (img_idx, list_idx) in [(0usize, 1usize), (1, 0)] {
    if let Expr::Image {
      width,
      height,
      channels,
      data,
      image_type,
      ..
    } = &args[img_idx]
      && let Expr::List(items) = &args[list_idx]
      && items.len() == *channels as usize
      && *channels > 1
    {
      let mut per_channel = Vec::with_capacity(items.len());
      for item in items {
        match crate::functions::math_ast::try_eval_to_f64(item) {
          Some(v) => per_channel.push(v),
          None => break,
        }
      }
      if per_channel.len() == items.len() {
        let is_r32 = matches!(image_type, ImageType::Real32);
        let ch = *channels as usize;
        let new_data: Vec<f64> = data
          .iter()
          .enumerate()
          .map(|(i, &v)| {
            let s = per_channel[i % ch];
            if img_idx == 0 {
              apply(v, s, is_r32)
            } else {
              apply(s, v, is_r32)
            }
          })
          .collect();
        return Ok(Expr::Image {
          color_space: None,
          display_size: None,
          width: *width,
          height: *height,
          channels: *channels,
          data: Arc::new(new_data),
          image_type: *image_type,
        });
      }
    }
  }

  // (Image, scalar) — apply `op(pixel, scalar)` to every pixel.
  if let Expr::Image {
    color_space: _,
    display_size: _,
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
    && let Some(s) = crate::functions::math_ast::try_eval_to_f64(&args[1])
  {
    let is_r32 = matches!(image_type, ImageType::Real32);
    let new_data: Vec<f64> =
      data.iter().map(|&v| apply(v, s, is_r32)).collect();
    return Ok(Expr::Image {
      color_space: None,
      display_size: None,
      width: *width,
      height: *height,
      channels: *channels,
      data: Arc::new(new_data),
      image_type: *image_type,
    });
  }
  // (scalar, Image) — apply `op(scalar, pixel)`.
  if let Expr::Image {
    color_space: _,
    display_size: _,
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[1]
    && let Some(s) = crate::functions::math_ast::try_eval_to_f64(&args[0])
  {
    let is_r32 = matches!(image_type, ImageType::Real32);
    let new_data: Vec<f64> =
      data.iter().map(|&v| apply(s, v, is_r32)).collect();
    return Ok(Expr::Image {
      color_space: None,
      display_size: None,
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
  let bad = if matches!(&args[0], Expr::Image { .. }) {
    &args[1]
  } else {
    &args[0]
  };
  crate::emit_message(&format!(
    "{}::imginv: Expecting an image or graphics instead of {}.",
    name,
    expr_to_string(bad)
  ));
  Ok(unevaluated(name, args))
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
      "{name} expects at least 2 arguments"
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

  // RandomImage[dist, …]: pixel values drawn from a distribution.
  if let Some(dist @ Expr::FunctionCall { name, .. }) = args.first()
    && name.ends_with("Distribution")
  {
    let (w, h) = if args.len() == 2 {
      let Some(dimensions) = positive_image_dimensions("RandomImage", &args[1])
      else {
        return Ok(unevaluated("RandomImage", args));
      };
      dimensions
    } else {
      (150u32, 150u32)
    };
    let len = (w as usize) * (h as usize);
    // RandomVariate's own complaints are not this call's: it reports bddist.
    crate::push_quiet();
    let sample = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "RandomVariate".to_string(),
      args: vec![dist.clone(), Expr::Integer(len as i128)].into(),
    });
    crate::pop_quiet();
    let data: Option<Vec<f64>> = match &sample {
      Ok(Expr::List(values)) if values.len() == len => {
        values.iter().map(try_eval_to_f64).collect()
      }
      _ => None,
    };
    let Some(data) = data else {
      crate::emit_message(&format!(
        "RandomImage::bddist: The specified random distribution {} should generate a real number or a list of real numbers.",
        expr_to_string(dist)
      ));
      return Ok(unevaluated("RandomImage", args));
    };
    return Ok(Expr::Image {
      color_space: None,
      display_size: None,
      width: w,
      height: h,
      channels: 1,
      data: Arc::new(data),
      image_type: ImageType::Real32,
    });
  }
  // RandomImage[] → 150x150, [0,1]
  // RandomImage[max] / RandomImage[{min, max}] → 150x150
  // RandomImage[…, {w, h}] → w×h
  let (min_val, max_val) = match args.first() {
    None => (0.0, 1.0),
    Some(Expr::List(range)) if range.len() == 2 => {
      (expr_to_f64(&range[0])?, expr_to_f64(&range[1])?)
    }
    Some(other) => (0.0, expr_to_f64(other)?),
  };
  // An empty range has nothing to sample; Wolfram reports the equivalent
  // uniform distribution as unusable and leaves the call unevaluated.
  // Negated on purpose: NaN bounds are not a usable range either.
  #[allow(clippy::neg_cmp_op_on_partial_ord)]
  if !(min_val < max_val) {
    // Echo the bounds as written, so an integer 0 is not reported as 0.
    let bound = |i: usize, fallback: f64| -> Expr {
      match args.first() {
        Some(Expr::List(range)) if range.len() == 2 => range[i].clone(),
        Some(other) if i == 1 => other.clone(),
        _ => {
          if fallback.fract() == 0.0 {
            Expr::Integer(fallback as i128)
          } else {
            Expr::Real(fallback)
          }
        }
      }
    };
    let range = Expr::List(vec![bound(0, min_val), bound(1, max_val)].into());
    crate::emit_message(&format!(
      "RandomImage::bddist: The specified random distribution UniformDistribution[{}] should generate a real number or a list of real numbers.",
      expr_to_string(&range)
    ));
    return Ok(unevaluated("RandomImage", args));
  }

  let (w, h) = if args.len() == 2 {
    let Some(dimensions) = positive_image_dimensions("RandomImage", &args[1])
    else {
      return Ok(unevaluated("RandomImage", args));
    };
    dimensions
  } else {
    (150u32, 150u32)
  };

  use rand::RngExt;
  let channels = 1u8; // Grayscale by default (matches Wolfram)
  let len = (w as usize) * (h as usize) * (channels as usize);
  let data: Vec<f64> = crate::with_rng(|rng| {
    (0..len)
      .map(|_| rng.random_range(min_val..max_val))
      .collect()
  });

  Ok(Expr::Image {
    color_space: None,
    display_size: None,
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
  Ok(bool_expr(result))
}

/// PixelValue[img, {x, y}] — pixel value at column x (1-indexed from the
/// left) and row y (1-indexed from the bottom). Out-of-bounds positions
/// return 0 (or a list of zeros for multi-channel images).
pub fn pixel_value_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let Expr::Image {
    color_space: _,
    display_size: _,
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
  else {
    crate::emit_message(&format!(
      "PixelValue::imginv: Expecting an image or graphics instead of {}.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated("PixelValue", args));
  };
  let Expr::List(pos) = &args[1] else {
    return Ok(unevaluated("PixelValue", args));
  };
  let coord = |e: &Expr| -> Option<i64> {
    match e {
      Expr::Integer(n) => Some(*n as i64),
      Expr::Real(f) => Some(f.round() as i64),
      _ => None,
    }
  };
  let ch = *channels as usize;
  let is_real32 = matches!(image_type, ImageType::Real32);
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
    return Ok(unevaluated("PixelValue", args));
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

/// TextRecognize[img, level?] — OCR stub. Real text recognition is not
/// implemented; this matches wolframscript's empty-output fallback when
/// no text is found in the image. For non-image input it emits the
/// TextRecognize::imgvinv warning and stays unevaluated.
pub fn text_recognize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if !matches!(&args[0], Expr::Image { .. }) && !is_valid_image3d(&args[0]) {
    crate::emit_message(&format!(
      "TextRecognize::imgvinv: Expecting an image, graphics or video instead of {}.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated("TextRecognize", args));
  }
  // Two-arg form with a structural level: `Line`, `Word`, `Character`,
  // `Block` — wolframscript returns an empty list when no text is
  // found. Anything else falls back to the empty-string scalar.
  if args.len() >= 2
    && let Expr::String(s) = &args[1]
    && matches!(s.as_str(), "Line" | "Word" | "Character" | "Block")
  {
    return Ok(Expr::List(Vec::new().into()));
  }
  Ok(Expr::String(String::new()))
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
  let unevaluated = || unevaluated("GradientFilter", args);
  if args.len() != 2 {
    return Ok(unevaluated());
  }

  // 1D list input with radius 1: use the original central-difference
  // path (matches wolframscript's GradientFilter[{...}, 1] output).
  if let (Expr::List(elems), Some(r)) = (
    &args[0],
    crate::functions::math_ast::try_eval_to_f64(&args[1])
      .filter(|v| *v >= 0.0)
      .map(|v| v as usize),
  ) && r == 1
    // Restrict to flat numeric lists (not nested-list matrices, which
    // need the 2D path further down).
    && !elems.iter().any(|e| matches!(e, Expr::List(_)))
  {
    let n = elems.len();
    let mut values: Vec<f64> = Vec::with_capacity(n);
    for e in elems {
      let Some(v) = crate::functions::math_ast::try_eval_to_f64(e) else {
        return Ok(unevaluated());
      };
      values.push(v);
    }
    if n == 0 {
      return Ok(Expr::List(Vec::new().into()));
    }
    if n == 1 {
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

  // Image input (and any other radius spec): use the Bessel-based
  // discrete-Gaussian gradient. Parse radius / sigma.
  let (radius, sigma) = match &args[1] {
    Expr::Integer(n) if *n >= 1 => (*n as usize, (*n as f64) / 2.0),
    Expr::Real(f) if *f >= 1.0 => (f.floor() as usize, *f / 2.0),
    Expr::List(items) if items.len() == 2 => {
      let r_int = match &items[0] {
        Expr::Integer(n) if *n >= 1 => *n as usize,
        Expr::Real(f) if *f >= 1.0 => f.floor() as usize,
        _ => return Ok(unevaluated()),
      };
      let sigma = match &items[1] {
        Expr::Integer(n) if *n > 0 => *n as f64,
        Expr::Real(f) if *f > 0.0 => *f,
        _ => return Ok(unevaluated()),
      };
      (r_int, sigma)
    }
    _ => return Ok(unevaluated()),
  };

  // Image input: separable 2D gradient.
  //   1. Build the 1D smooth kernel `T_k = exp(-σ²)·I_k(σ²)`
  //      (normalised, length 2·radius+1).
  //   2. Build the 1D derivative kernel `D[k] = -k·T[k] / Σ_j j²·T[j]`
  //      — chosen so applying D to a unit ramp recovers its slope.
  //   3. For each pixel compute
  //        ∂I/∂x = T ⊛_y D ⊛_x I,
  //        ∂I/∂y = D ⊛_y T ⊛_x I,
  //      then `|∇I| = √((∂I/∂x)² + (∂I/∂y)²)`.
  if let Expr::Image {
    color_space: _,
    display_size: _,
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
  {
    let w = *width as usize;
    let h = *height as usize;
    let ch = *channels as usize;
    if w == 0 || h == 0 {
      return Ok(args[0].clone());
    }
    let mut out: Vec<f64> = vec![0.0; data.len()];
    for c_idx in 0..ch {
      let plane: Vec<f64> = (0..w * h).map(|i| data[i * ch + c_idx]).collect();
      let axis = GaussianAxis { radius, sigma };
      let (dx, dy) = plane_gradient(&plane, w, h, axis, axis);
      for i in 0..w * h {
        out[i * ch + c_idx] = dx[i].hypot(dy[i]);
      }
    }
    return Ok(Expr::Image {
      color_space: None,
      display_size: None,
      width: *width,
      height: *height,
      channels: *channels,
      data: Arc::new(out),
      image_type: *image_type,
    });
  }

  Ok(unevaluated())
}

/// The two partial derivatives of one image plane, smoothed by the
/// discrete Gaussian of half-width `radius` and standard deviation
/// `sigma`: `dx = D ⊛_x (T ⊛_y plane)` and `dy = D ⊛_y (T ⊛_x plane)`,
/// where `T` is `gaussian_filter_kernel` and `D` its derivative. Shared
/// by `GradientFilter`, whose result is the magnitude, and by
/// `EdgeDetect`, which also needs the direction.
fn plane_gradient(
  plane: &[f64],
  w: usize,
  h: usize,
  vertical: GaussianAxis,
  horizontal: GaussianAxis,
) -> (Vec<f64>, Vec<f64>) {
  let smooth_rows = gaussian_filter_kernel(horizontal.radius, horizontal.sigma);
  let smooth_columns = gaussian_filter_kernel(vertical.radius, vertical.sigma);
  let derivative_rows = gradient_filter_derivative_kernel(&smooth_rows);
  let derivative_columns = gradient_filter_derivative_kernel(&smooth_columns);
  let along_rows = |source: &[f64], kernel: &[f64]| {
    let mut out = vec![0.0; w * h];
    for y in 0..h {
      let row: Vec<f64> = (0..w).map(|x| source[y * w + x]).collect();
      let filtered =
        crate::functions::math_ast::convolve_edge_padded(&row, kernel);
      out[y * w..y * w + w].copy_from_slice(&filtered);
    }
    out
  };
  let along_columns = |source: &[f64], kernel: &[f64]| {
    let mut out = vec![0.0; w * h];
    for x in 0..w {
      let column: Vec<f64> = (0..h).map(|y| source[y * w + x]).collect();
      let filtered =
        crate::functions::math_ast::convolve_edge_padded(&column, kernel);
      for y in 0..h {
        out[y * w + x] = filtered[y];
      }
    }
    out
  };
  (
    along_rows(&along_columns(plane, &smooth_columns), &derivative_rows),
    along_columns(&along_rows(plane, &smooth_rows), &derivative_columns),
  )
}

/// 1D Gaussian derivative kernel `D[k] = -k·T[k] / Σ_j j²·T[j]`. The
/// denominator normalises the kernel so that applying it to a unit
/// ramp (`{0, 1, 2, …}`) recovers slope 1 — wolframscript uses the
/// same normalisation.
fn gradient_filter_derivative_kernel(smooth: &[f64]) -> Vec<f64> {
  let len = smooth.len();
  let radius = (len.saturating_sub(1) / 2) as i64;
  let mut d: Vec<f64> = Vec::with_capacity(len);
  let mut denom: f64 = 0.0;
  for (i, &t_val) in smooth.iter().enumerate() {
    let k = i as i64 - radius;
    let kf = k as f64;
    d.push(-kf * t_val);
    denom += kf * kf * t_val;
  }
  if denom > 0.0 {
    for v in &mut d {
      *v /= denom;
    }
  }
  d
}

/// LaplacianFilter[image, r] — sum of the second derivatives of the
/// image smoothed by the discrete Gaussian of half-width `r` and
/// standard deviation `r/2`: `L ⊛_x (T ⊛_y I) + L ⊛_y (T ⊛_x I)`, where
/// `T` is `gaussian_filter_kernel` and `L[k] = T[k-1] - 2·T[k] + T[k+1]`
/// (so a unit parabola `k²` yields 2). Padding repeats the edge pixel.
pub fn laplacian_filter_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("LaplacianFilter", args);
  let Expr::Image {
    width,
    height,
    channels,
    data,
    image_type,
    ..
  } = &args[0]
  else {
    return Ok(unevaluated());
  };
  let axis = match expr_to_f64_opt(&args[1]) {
    Some(r) if r >= 1.0 => gaussian_axis_from_radius(r),
    _ => return Ok(unevaluated()),
  };
  let w = *width as usize;
  let h = *height as usize;
  let ch = *channels as usize;
  if w == 0 || h == 0 {
    return Ok(args[0].clone());
  }
  let smooth = gaussian_filter_kernel(axis.radius, axis.sigma);
  let wide = gaussian_filter_kernel(axis.radius + 1, axis.sigma);
  let mut second: Vec<f64> = (1..wide.len() - 1)
    .map(|i| wide[i - 1] - 2.0 * wide[i] + wide[i + 1])
    .collect();
  // Truncating the kernel must not leak a constant response, so the
  // centre tap absorbs the truncation error and the taps sum to zero.
  let centre = axis.radius;
  let others: f64 = second
    .iter()
    .enumerate()
    .filter(|(i, _)| *i != centre)
    .map(|(_, v)| v)
    .sum();
  second[centre] = -others;
  let convolve = crate::functions::math_ast::convolve_edge_padded;
  let along_rows = |source: &[f64], kernel: &[f64]| {
    let mut out = vec![0.0; w * h];
    for y in 0..h {
      let filtered = convolve(&source[y * w..(y + 1) * w], kernel);
      out[y * w..(y + 1) * w].copy_from_slice(&filtered);
    }
    out
  };
  let along_columns = |source: &[f64], kernel: &[f64]| {
    let mut out = vec![0.0; w * h];
    for x in 0..w {
      let column: Vec<f64> = (0..h).map(|y| source[y * w + x]).collect();
      let filtered = convolve(&column, kernel);
      for y in 0..h {
        out[y * w + x] = filtered[y];
      }
    }
    out
  };
  let mut out: Vec<f64> = vec![0.0; data.len()];
  for c_idx in 0..ch {
    let plane: Vec<f64> = (0..w * h).map(|i| data[i * ch + c_idx]).collect();
    let dxx = along_rows(&along_columns(&plane, &smooth), &second);
    let dyy = along_columns(&along_rows(&plane, &smooth), &second);
    for i in 0..w * h {
      out[i * ch + c_idx] = dxx[i] + dyy[i];
    }
  }
  Ok(Expr::Image {
    color_space: None,
    display_size: None,
    width: *width,
    height: *height,
    channels: *channels,
    data: Arc::new(out),
    image_type: *image_type,
  })
}

pub fn median_filter_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // A list argument reads its range like the other neighborhood filters:
  // the magnitude, and a whole number or nothing. An image keeps taking
  // any non-negative radius it is handed.
  if matches!(&args[0], Expr::List(_)) {
    let Some(radius) = neighborhood_radius(
      "MedianFilter",
      &args[1],
      NeighborhoodRange::Whole,
      true,
    ) else {
      return Ok(unevaluated("MedianFilter", args));
    };
    let mut whole = args.to_vec();
    whole[1] = Expr::Integer(radius as i128);
    return Ok(median_filter_body(&whole));
  }
  Ok(median_filter_body(args))
}

/// The body of `median_filter_ast`, reached with a range already
/// resolved for list data.
fn median_filter_body(args: &[Expr]) -> Expr {
  let r = crate::functions::math_ast::try_eval_to_f64(&args[1])
    .filter(|v| *v >= 0.0)
    .map(|v| v as usize);

  // MedianFilter[Image, r] — apply 2D median filter per channel on the
  // pixel buffer; result preserves precision and image type.
  if let (
    Expr::Image {
      color_space: _,
      display_size: _,
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
    return Expr::Image {
      color_space: None,
      display_size: None,
      width: *width,
      height: *height,
      channels: *channels,
      data: Arc::new(new_data),
      image_type: *image_type,
    };
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
          let med = crate::evaluator::evaluate_expr_to_expr(&call(
            "Median",
            vec![Expr::List(window.into())],
          ))
          .unwrap_or_else(|_| get(y, x));
          new_row.push(med);
        }
        rows.push(Expr::List(new_row.into()));
      }
      return Expr::List(rows.into());
    }
    // 1D path.
    let n = elems.len();
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
      let lo = i.saturating_sub(r);
      let hi = if i + r < n { i + r } else { n - 1 };
      let window: Vec<Expr> = elems[lo..=hi].to_vec();
      let med = crate::evaluator::evaluate_expr_to_expr(&call(
        "Median",
        vec![Expr::List(window.into())],
      ))
      .unwrap_or_else(|_| elems[i].clone());
      result.push(med);
    }
    return Expr::List(result.into());
  }

  let valid = matches!(&args[0], Expr::Image { .. } | Expr::List(_));
  if !valid {
    // wolframscript prints MedianFilter::arg1 to stdout for a non-array arg.
    crate::emit_message_to_stdout(&format!(
      "MedianFilter::arg1: The first argument {} should be a rectangular array, image or video.",
      expr_to_string(&args[0])
    ));
  }
  unevaluated("MedianFilter", args)
}

/// MeanFilter[arg, r] — replace every element by the mean of the values
/// in a neighborhood of radius `r` (window clipped at boundaries).
/// Supports flat 1D lists and rectangular 2D nested lists. A negative
/// integer radius is treated as its absolute value (matching
/// wolframscript). Non-integer radii emit MeanFilter::bdrad and the
/// expression is returned unevaluated.
pub fn mean_filter_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // MeanFilter[Image, r] — the mean of each clipped neighbourhood, per
  // channel, on the pixel buffer. Samples land back in the image's own
  // precision, which for the usual Real32 image is what rounds 4/9 to
  // 0.4444444477558136 the way wolframscript reports it.
  if let (
    Expr::Image {
      color_space,
      display_size: _,
      width,
      height,
      channels,
      data,
      image_type,
    },
    Some(r),
  ) = (
    &args[0],
    crate::functions::math_ast::try_eval_to_f64(&args[1])
      .filter(|v| v.fract() == 0.0)
      .map(|v| v.abs() as usize),
  ) {
    let (w, h, ch) = (*width as usize, *height as usize, *channels as usize);
    let mut new_data = vec![0.0; data.len()];
    for c_idx in 0..ch {
      for y in 0..h {
        for x in 0..w {
          let y0 = y.saturating_sub(r);
          let y1 = (y + r).min(h.saturating_sub(1));
          let x0 = x.saturating_sub(r);
          let x1 = (x + r).min(w.saturating_sub(1));
          let mut sum = 0.0;
          let mut count = 0.0;
          for yy in y0..=y1 {
            for xx in x0..=x1 {
              sum += data[(yy * w + xx) * ch + c_idx];
              count += 1.0;
            }
          }
          new_data[(y * w + x) * ch + c_idx] = sum / count;
        }
      }
    }
    return Ok(quantize_image(Expr::Image {
      color_space: *color_space,
      display_size: None,
      width: *width,
      height: *height,
      channels: *channels,
      data: Arc::new(new_data),
      image_type: *image_type,
    }));
  }
  Ok(aggregating_filter_ast(
    args,
    "Mean",
    "MeanFilter",
    NeighborhoodRange::Whole,
  ))
}

/// The same image with every sample stored at its own precision, so a
/// `Real32` image reports the values it can actually hold.
fn quantize_image(image: Expr) -> Expr {
  let Expr::Image {
    color_space,
    display_size: _,
    width,
    height,
    channels,
    data,
    image_type,
  } = &image
  else {
    return image;
  };
  let rounded: Vec<f64> = match image_type {
    ImageType::Real32 => data.iter().map(|v| *v as f32 as f64).collect(),
    _ => data.to_vec(),
  };
  Expr::Image {
    color_space: *color_space,
    display_size: None,
    width: *width,
    height: *height,
    channels: *channels,
    data: Arc::new(rounded),
    image_type: *image_type,
  }
}

/// `ImageFilter[f, image, r]` — `f` applied to the range-`r` neighbourhood of
/// every pixel, in each channel. Unlike the aggregating filters, the window is
/// always the full `(2r+1)` square: the image is extended by repeating its
/// edge samples rather than clipped, so a corner sees as many values as the
/// centre does.
pub fn image_filter_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let echo = || Ok(unevaluated("ImageFilter", args));
  if args.len() != 3 {
    return echo();
  }
  let Expr::Image {
    color_space,
    display_size: _,
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[1]
  else {
    return echo();
  };
  let Some(r) = crate::functions::math_ast::try_eval_to_f64(&args[2])
    .filter(|v| v.fract() == 0.0 && *v >= 0.0)
    .map(|v| v as usize)
  else {
    return echo();
  };
  let (w, h, ch) = (*width as usize, *height as usize, *channels as usize);
  if w == 0 || h == 0 {
    return echo();
  }
  // Samples are reported at the image's own precision, so that is what the
  // filter sees.
  let held: Vec<f64> = match image_type {
    ImageType::Real32 => data.iter().map(|v| *v as f32 as f64).collect(),
    _ => data.to_vec(),
  };
  let span = 2 * r + 1;
  let mut new_data = vec![0.0; held.len()];
  for c_idx in 0..ch {
    for y in 0..h {
      for x in 0..w {
        let mut rows = Vec::with_capacity(span);
        for dy in 0..span {
          // An index past an edge takes the edge sample.
          let yy = (y + dy).saturating_sub(r).min(h - 1);
          let mut row = Vec::with_capacity(span);
          for dx in 0..span {
            let xx = (x + dx).saturating_sub(r).min(w - 1);
            row.push(Expr::Real(held[(yy * w + xx) * ch + c_idx]));
          }
          rows.push(Expr::List(row.into()));
        }
        let applied = crate::evaluator::apply_function_to_arg(
          &args[0],
          &Expr::List(rows.into()),
        )?;
        // A filter that does not answer one number per pixel is not one this
        // builds an image from.
        let Some(value) = crate::functions::math_ast::try_eval_to_f64(&applied)
        else {
          return echo();
        };
        new_data[(y * w + x) * ch + c_idx] = value;
      }
    }
  }
  Ok(quantize_image(Expr::Image {
    color_space: *color_space,
    display_size: None,
    width: *width,
    height: *height,
    channels: *channels,
    data: Arc::new(new_data),
    image_type: *image_type,
  }))
}

/// `ImageDifference[a, b]` — the absolute difference of two images of the
/// same size, sample by sample.
pub fn image_difference_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let echo = || Ok(unevaluated("ImageDifference", args));
  if args.len() != 2 {
    return echo();
  }
  // Samples are stored at full precision and only reported at the image's
  // own, so the difference has to be taken between the values the image
  // actually holds.
  let parts = |e: &Expr| match e {
    Expr::Image {
      width,
      height,
      channels,
      data,
      image_type,
      ..
    } => {
      let held: Vec<f64> = match image_type {
        ImageType::Real32 => data.iter().map(|v| *v as f32 as f64).collect(),
        _ => data.to_vec(),
      };
      Some((*width, *height, *channels, held))
    }
    _ => None,
  };
  for arg in args {
    if parts(arg).is_none() {
      crate::emit_message(&format!(
        "ImageDifference::imginv: Expecting an image or graphics instead of {}.",
        expr_to_string(arg)
      ));
      return echo();
    }
  }
  let (w1, h1, c1, a) = parts(&args[0]).expect("checked above");
  let (w2, h2, c2, b) = parts(&args[1]).expect("checked above");
  if (w1, h1, c1) != (w2, h2, c2) {
    crate::emit_message(
      "ImageDifference::imginvd: Expecting images of the same size in the \
       input.",
    );
    return echo();
  }
  let data: Vec<f64> =
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).collect();
  // The difference of two images is reported at Real32 whatever they were
  // stored as.
  Ok(quantize_image(Expr::Image {
    color_space: None,
    display_size: None,
    width: w1,
    height: h1,
    channels: c1,
    data: Arc::new(data),
    image_type: ImageType::Real32,
  }))
}

/// StandardDeviationFilter[data, r] — like MeanFilter but each window is
/// summarised by its (sample) StandardDeviation. Results are exact when the
/// input is exact. A window of a single element has no sample standard
/// deviation, so a zero range — or data with only one element — reports
/// `StandardDeviationFilter::shlen` rather than letting `StandardDeviation`
/// complain about a length under its own name.
pub fn standard_deviation_filter_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if let Expr::List(elems) = &args[0]
    && let Some(radius) = neighborhood_radius(
      "StandardDeviationFilter",
      &args[1],
      NeighborhoodRange::Whole,
      false,
    )
    && (radius == 0 || elems.len() < 2)
  {
    crate::emit_message(
      "StandardDeviationFilter::shlen: Cannot compute the standard \
       deviation of one element.",
    );
    return Ok(unevaluated("StandardDeviationFilter", args));
  }
  Ok(aggregating_filter_ast(
    args,
    "StandardDeviation",
    "StandardDeviationFilter",
    NeighborhoodRange::Whole,
  ))
}

/// `aggregating_filter_ast` for callers outside this module.
pub fn aggregating_filter_public(
  args: &[Expr],
  agg: &str,
  filter_name: &str,
  range: NeighborhoodRange,
) -> Result<Expr, InterpreterError> {
  Ok(aggregating_filter_ast(args, agg, filter_name, range))
}

/// How a neighborhood filter reads its range argument. All of them take
/// the magnitude, so a negative range filters the same neighborhood as
/// its positive counterpart, but they differ on fractions: `MinFilter`,
/// `MaxFilter` and `CommonestFilter` round any real up, while
/// `MeanFilter`, `MedianFilter` and `StandardDeviationFilter` insist on a
/// whole number and report anything else.
#[derive(Clone, Copy, PartialEq)]
pub enum NeighborhoodRange {
  /// `Abs[r]`, and `<head>::bdrad` if that is not an integer.
  Whole,
  /// `Ceiling[Abs[r]]` for any real.
  RoundedUp,
}

/// The radius named by a filter's range argument. `None` for a range
/// that names none; pass `report` to emit `<head>::bdrad` for it, which
/// the caller defers until it knows the first argument was data at all.
pub fn neighborhood_radius(
  head: &str,
  spec: &Expr,
  range: NeighborhoodRange,
  report: bool,
) -> Option<usize> {
  let reject = || {
    if report {
      crate::emit_message(&format!(
        "{head}::bdrad: {} is not a valid neighborhood range specification.",
        expr_to_string(spec)
      ));
    }
    None
  };
  let Some(magnitude) =
    crate::functions::math_ast::try_eval_to_f64(spec).map(f64::abs)
  else {
    return reject();
  };
  match range {
    NeighborhoodRange::Whole if magnitude.fract() != 0.0 => reject(),
    NeighborhoodRange::Whole => Some(magnitude as usize),
    NeighborhoodRange::RoundedUp => Some(magnitude.ceil() as usize),
  }
}

/// Shared engine for windowed aggregating filters (MeanFilter,
/// StandardDeviationFilter, …): slide a truncated radius-`r` window over a 1D
/// list or 2D array and replace each element with `agg` of its window.
fn aggregating_filter_ast(
  args: &[Expr],
  agg: &str,
  filter_name: &str,
  range: NeighborhoodRange,
) -> Expr {
  let r = neighborhood_radius(filter_name, &args[1], range, false);

  if let (Expr::List(elems), Some(r)) = (&args[0], r) {
    // 2D path: every element is a List of the same (non-zero) length.
    if !elems.is_empty()
      && elems.iter().all(|row| {
        matches!(row, Expr::List(items) if items.len() == match &elems[0] {
          Expr::List(first) => first.len(),
          _ => 0,
        })
      })
      && matches!(&elems[0], Expr::List(items) if !items.is_empty())
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
          let mean = crate::evaluator::evaluate_expr_to_expr(&call(
            agg,
            vec![Expr::List(window.into())],
          ))
          .unwrap_or_else(|_| get(y, x));
          new_row.push(mean);
        }
        rows.push(Expr::List(new_row.into()));
      }
      return Expr::List(rows.into());
    }
    // 1D path.
    let n = elems.len();
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
      let lo = i.saturating_sub(r);
      let hi = if i + r < n { i + r } else { n - 1 };
      let window: Vec<Expr> = elems[lo..=hi].to_vec();
      let mean = crate::evaluator::evaluate_expr_to_expr(&call(
        agg,
        vec![Expr::List(window.into())],
      ))
      .unwrap_or_else(|_| elems[i].clone());
      result.push(mean);
    }
    return Expr::List(result.into());
  }

  // A range that named no radius has already been reported.
  if r.is_none() && matches!(&args[0], Expr::List(_)) {
    neighborhood_radius(filter_name, &args[1], range, true);
  } else if !matches!(&args[0], Expr::Image { .. } | Expr::List(_)) {
    crate::emit_message(&format!(
      "{filter_name}::arg1: The first argument {} should be a rectangular array, image or video.",
      expr_to_string(&args[0])
    ));
  }
  unevaluated(filter_name, args)
}

/// ImageCorrelate[img, kernel] — 2D correlation per channel: each output
/// pixel is the kernel's dot product with the neighborhood *in the same
/// orientation*, so the impulse response is the kernel reversed.
/// Everything but that orientation — boundary handling, kernel centering,
/// channels and image type — is shared with `ImageConvolve` through
/// [`spatial_filter_ast`].
pub fn image_correlate_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ImageCorrelate expects exactly 2 arguments".into(),
    ));
  }
  spatial_filter_ast("ImageCorrelate", args, &args[1])
}

/// ImageConvolve[img, kernel] — 2D convolution per channel with
/// replicated boundary. Convolution reflects the kernel through its
/// center before the dot product (so the impulse response *is* the
/// kernel, matching `ListConvolve` against `ListCorrelate`); the
/// filtering itself is [`spatial_filter_ast`], shared with
/// `ImageCorrelate`. A symmetric kernel — `GaussianMatrix`, `BoxMatrix`,
/// `DiskMatrix` — makes the two identical.
pub fn image_convolve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ImageConvolve expects exactly 2 arguments".into(),
    ));
  }
  // A kernel that isn't a rectangular numeric matrix can't be reflected;
  // hand it to the filter unchanged so it reports/echoes it the same way
  // ImageCorrelate would.
  let kernel = reverse_kernel(&args[1]).unwrap_or_else(|| args[1].clone());
  spatial_filter_ast("ImageConvolve", args, &kernel)
}

/// Reflect a rank-2 kernel through its center (reverse both the row order
/// and each row's column order). `None` for anything that is not a
/// rectangular matrix — the entries themselves are left alone, so a kernel
/// written with rationals (`BoxMatrix[1]/9`) reflects like any other and
/// non-numeric entries are still rejected downstream, by the filter.
fn reverse_kernel(kernel: &Expr) -> Option<Expr> {
  let Expr::List(rows) = kernel else {
    return None;
  };
  let Some(Expr::List(first)) = rows.first() else {
    return None;
  };
  let cols = first.len();
  let mut out: Vec<Expr> = Vec::with_capacity(rows.len());
  for row in rows.iter().rev() {
    let Expr::List(items) = row else {
      return None;
    };
    if items.len() != cols {
      return None;
    }
    out.push(Expr::List(items.iter().rev().cloned().collect()));
  }
  Some(Expr::List(out.into()))
}

/// Read a rectangular numeric matrix (nested list of lists of numbers)
/// into row-major order, shared by `ImageConvolve`/`ImageCorrelate`'s
/// kernel argument and `ImageDeconvolve`'s point-spread-function argument.
/// `None` for anything that is not a non-empty rectangular matrix.
fn parse_numeric_matrix(expr: &Expr) -> Option<(usize, usize, Vec<f64>)> {
  let Expr::List(rows) = expr else {
    return None;
  };
  if rows.is_empty() {
    return None;
  }
  let krows = rows.len();
  let Expr::List(first) = &rows[0] else {
    return None;
  };
  let kcols = first.len();
  if kcols == 0 {
    return None;
  }
  let mut matrix = Vec::with_capacity(krows * kcols);
  for row in rows {
    let Expr::List(items) = row else {
      return None;
    };
    if items.len() != kcols {
      return None;
    }
    for v in items {
      matrix.push(expr_to_f64(v).ok()?);
    }
  }
  Some((krows, kcols, matrix))
}

/// The value bound to `key -> …` among a call's trailing option arguments,
/// written as `Rule` or `RuleDelayed`.
fn option_value<'a>(opts: &'a [Expr], key: &str) -> Option<&'a Expr> {
  opts.iter().find_map(|o| match o {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } if matches!(pattern.as_ref(), Expr::Identifier(k) if k == key) => {
      Some(replacement.as_ref())
    }
    _ => None,
  })
}

/// Forward 2D DFT of a real `width × height` array (row-major), computed
/// as separable 1D DFTs (rows, then columns) with `FourierParameters ->
/// {1, 1}` (unscaled forward sum, `1/n` inverse) rather than `Fourier`'s
/// own `{0, 1}` default — this is the convention under which pointwise
/// multiplication in the transform domain is exactly circular convolution
/// in the spatial domain, which the regularized inverse filter below
/// relies on. `idft2d` is its exact inverse.
fn dft2d(width: usize, height: usize, re: &[f64]) -> Vec<(f64, f64)> {
  let mut data: Vec<(f64, f64)> = re.iter().map(|&v| (v, 0.0)).collect();
  for y in 0..height {
    let row = &data[y * width..(y + 1) * width];
    let transformed =
      crate::functions::math_ast::dft_core(row, 1.0, 1.0, false);
    data[y * width..(y + 1) * width].copy_from_slice(&transformed);
  }
  let mut col = vec![(0.0, 0.0); height];
  for x in 0..width {
    for (y, slot) in col.iter_mut().enumerate() {
      *slot = data[y * width + x];
    }
    let transformed =
      crate::functions::math_ast::dft_core(&col, 1.0, 1.0, false);
    for (y, value) in transformed.into_iter().enumerate() {
      data[y * width + x] = value;
    }
  }
  data
}

/// Inverse of [`dft2d`].
fn idft2d(width: usize, height: usize, data: &[(f64, f64)]) -> Vec<(f64, f64)> {
  let mut data = data.to_vec();
  for y in 0..height {
    let row = &data[y * width..(y + 1) * width];
    let transformed = crate::functions::math_ast::dft_core(row, 1.0, 1.0, true);
    data[y * width..(y + 1) * width].copy_from_slice(&transformed);
  }
  let mut col = vec![(0.0, 0.0); height];
  for x in 0..width {
    for (y, slot) in col.iter_mut().enumerate() {
      *slot = data[y * width + x];
    }
    let transformed =
      crate::functions::math_ast::dft_core(&col, 1.0, 1.0, true);
    for (y, value) in transformed.into_iter().enumerate() {
      data[y * width + x] = value;
    }
  }
  data
}

/// ImageDeconvolve[image, ker] / ImageDeconvolve[image, ker, opts] —
/// restores an image blurred by point-spread function `ker`, undoing what
/// `ImageConvolve[image, ker]` would have done.
///
/// Woxi implements the spectral method family — `Method -> "DampedLS"`
/// (the default), `"Tikhonov"` and `"Wiener"` — as one regularized inverse
/// filter in the frequency domain: `Fhat = conj(K)·G / (|K|² + λ)`, where
/// `G` and `K` are the 2D DFTs of the (padded) image and the
/// zero-centered PSF. `λ` is either the method's own numeric parameter
/// (`Method -> {"Tikhonov", λ}`) or, when that parameter is `Automatic`
/// or omitted, a small multiple of the kernel's peak spectral energy —
/// Woxi's own heuristic, since Mathematica's automatic-λ selection isn't
/// published. The iterative methods (`"TSVD"`, `"Hybrid"`,
/// `"RichardsonLucy"`, `"SteepestDescent"`, `"TotalVariation"`) are not
/// implemented and echo the call unevaluated.
///
/// `Padding` (default `"Reversed"`) controls how the image is extended by
/// half the kernel size before the transform, reusing the same
/// `"Fixed"`/`"Periodic"`/reflected boundary modes as `ImagePad` (Woxi maps
/// `"Reversed"` to the reflected mode); the result is cropped back to the
/// original dimensions.
pub fn image_deconvolve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "ImageDeconvolve expects at least 2 arguments".into(),
    ));
  }
  let Expr::Image {
    color_space,
    display_size: _,
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
  else {
    crate::emit_message(&format!(
      "ImageDeconvolve::imginv: Expecting an image or graphics instead of {}.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated("ImageDeconvolve", args));
  };
  let Some((krows, kcols, kernel)) = parse_numeric_matrix(&args[1]) else {
    return Ok(unevaluated("ImageDeconvolve", args));
  };

  let opts = &args[2..];
  let (method_name, method_param): (String, Option<f64>) =
    match option_value(opts, "Method") {
      Some(Expr::String(s)) => (s.clone(), None),
      Some(Expr::List(items)) if !items.is_empty() => {
        let name = match &items[0] {
          Expr::String(s) => s.clone(),
          _ => "DampedLS".to_string(),
        };
        let param = items
          .get(1)
          .and_then(crate::functions::math_ast::try_eval_to_f64)
          .filter(|p| *p > 0.0);
        (name, param)
      }
      _ => ("DampedLS".to_string(), None),
    };
  if !matches!(method_name.as_str(), "DampedLS" | "Tikhonov" | "Wiener") {
    // Iterative methods aren't implemented; echo the call rather than
    // fabricate a result.
    return Ok(unevaluated("ImageDeconvolve", args));
  }
  let padding_mode = match option_value(opts, "Padding") {
    Some(Expr::String(s)) if s == "Fixed" => "Fixed",
    Some(Expr::String(s)) if s == "Periodic" => "Periodic",
    _ => "Reflected", // "Reversed" (the default) and anything unrecognized.
  };

  let w = *width as usize;
  let h = *height as usize;
  let ch = *channels as usize;
  let cy = krows / 2;
  let cx = kcols / 2;
  let pad_y = cy.min(h.saturating_sub(1));
  let pad_x = cx.min(w.saturating_sub(1));
  let pw = w + 2 * pad_x;
  let ph = h + 2 * pad_y;

  // Embed the PSF into a pw×ph array with its center at (0, 0), wrapping
  // around — the standard placement for a circular-convolution kernel.
  let mut kernel_full = vec![0.0_f64; pw * ph];
  for ky in 0..krows {
    for kx in 0..kcols {
      let oy = (ky as isize - cy as isize).rem_euclid(ph as isize) as usize;
      let ox = (kx as isize - cx as isize).rem_euclid(pw as isize) as usize;
      kernel_full[oy * pw + ox] += kernel[ky * kcols + kx];
    }
  }
  let kernel_fft = dft2d(pw, ph, &kernel_full);
  let max_k2 = kernel_fft
    .iter()
    .map(|&(re, im)| re * re + im * im)
    .fold(0.0_f64, f64::max);
  let lambda = method_param.unwrap_or(max_k2 * 1e-3).max(1e-12);

  let mut out_data = vec![0.0_f64; data.len()];
  let mut padded = vec![0.0_f64; pw * ph];
  for c in 0..ch {
    for py in 0..ph {
      let sy =
        image_pad_source(padding_mode, py as i64 - pad_y as i64, h as i64)
          .unwrap_or(0) as usize;
      for px in 0..pw {
        let sx =
          image_pad_source(padding_mode, px as i64 - pad_x as i64, w as i64)
            .unwrap_or(0) as usize;
        padded[py * pw + px] = data[(sy * w + sx) * ch + c];
      }
    }
    let img_fft = dft2d(pw, ph, &padded);
    let mut restored_fft = vec![(0.0, 0.0); pw * ph];
    for i in 0..pw * ph {
      let (kr, ki) = kernel_fft[i];
      let (gr, gi) = img_fft[i];
      let denom = kr * kr + ki * ki + lambda;
      restored_fft[i] =
        ((kr * gr + ki * gi) / denom, (kr * gi - ki * gr) / denom);
    }
    let restored = idft2d(pw, ph, &restored_fft);
    for y in 0..h {
      for x in 0..w {
        let (re, _im) = restored[(y + pad_y) * pw + (x + pad_x)];
        out_data[(y * w + x) * ch + c] = re;
      }
    }
  }

  Ok(Expr::Image {
    color_space: *color_space,
    display_size: None,
    width: *width,
    height: *height,
    channels: *channels,
    data: Arc::new(out_data),
    image_type: *image_type,
  })
}

/// The kernel filtering shared by `ImageCorrelate` and `ImageConvolve`:
/// per channel, each output pixel is the dot product of `kernel` with the
/// surrounding neighborhood, with the boundary extended by replication.
/// The kernel's center sits at floor(rows/2), floor(cols/2). Channels and
/// image_type are preserved and output values are not clamped.
///
/// `head` and `args` are the *caller's* head and arguments, used for
/// messages and for echoing an unevaluated call; `kernel` is the matrix to
/// filter with, which convolution passes reflected (see
/// [`image_convolve_ast`]).
fn spatial_filter_ast(
  head: &str,
  args: &[Expr],
  kernel_expr: &Expr,
) -> Result<Expr, InterpreterError> {
  let Expr::Image {
    color_space: _,
    display_size: _,
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
  else {
    crate::emit_message(&format!(
      "{head}::imginv: Expecting an image or graphics instead of {}.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated(head, args));
  };

  // Parse the kernel: a 1D or 2D nested list of numbers.
  let Expr::List(rows) = kernel_expr else {
    return Ok(unevaluated(head, args));
  };
  if rows.is_empty() {
    return Ok(args[0].clone());
  }
  let krows = rows.len();
  let kcols = match &rows[0] {
    Expr::List(items) => items.len(),
    _ => 0,
  };
  if kcols == 0 {
    return Ok(args[0].clone());
  }
  let mut kernel = Vec::with_capacity(krows * kcols);
  for row in rows {
    let Expr::List(items) = row else {
      return Ok(unevaluated(head, args));
    };
    if items.len() != kcols {
      return Ok(unevaluated(head, args));
    }
    for v in items {
      kernel.push(expr_to_f64(v)?);
    }
  }
  let cy = krows / 2;
  let cx = kcols / 2;

  let w = *width as usize;
  let h = *height as usize;
  let ch = *channels as usize;
  let mut new_data = vec![0.0_f64; data.len()];
  for c_idx in 0..ch {
    for y in 0..h {
      for x in 0..w {
        let mut sum = 0.0;
        for ky in 0..krows {
          for kx in 0..kcols {
            let sy = (y as isize + ky as isize - cy as isize)
              .clamp(0, h as isize - 1) as usize;
            let sx = (x as isize + kx as isize - cx as isize)
              .clamp(0, w as isize - 1) as usize;
            sum += kernel[ky * kcols + kx] * data[(sy * w + sx) * ch + c_idx];
          }
        }
        new_data[(y * w + x) * ch + c_idx] = sum;
      }
    }
  }
  Ok(Expr::Image {
    color_space: None,
    display_size: None,
    width: *width,
    height: *height,
    channels: *channels,
    data: Arc::new(new_data),
    image_type: *image_type,
  })
}

/// GaussianFilter[arg, sigma] — stub. Real Gaussian filtering for
/// images/arrays/videos is not implemented yet; this stub matches
/// wolframscript's GaussianFilter::arg1 warning for inputs that are
/// neither image nor list.
pub fn gaussian_filter_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    let valid =
      matches!(args.first(), Some(Expr::Image { .. } | Expr::List(_)));
    if !valid && !args.is_empty() {
      crate::emit_message(&format!(
        "GaussianFilter::arg1: The first argument {} should be a rectangular array, image or video.",
        expr_to_string(&args[0])
      ));
    }
    return Ok(unevaluated("GaussianFilter", args));
  }
  // Parse the radius specification from the second argument. A plain
  // number `r` gives a kernel of half-width `Ceiling[r]` and standard
  // deviation `r/2`; the pair `{r, sigma}` sets the deviation
  // explicitly. Any other list shape is left for a caller to make sense
  // of, so it echoes rather than reporting a radius that may be valid.
  let axis = match &args[1] {
    Expr::List(items) if items.len() == 2 => {
      let (Some(radius), Some(sigma)) =
        (expr_to_f64_opt(&items[0]), expr_to_f64_opt(&items[1]))
      else {
        return Ok(unevaluated("GaussianFilter", args));
      };
      if sigma <= 0.0 {
        return Ok(unevaluated("GaussianFilter", args));
      }
      GaussianAxis {
        radius: if radius <= 0.0 {
          0
        } else {
          radius.ceil() as usize
        },
        sigma,
      }
    }
    Expr::List(_) => {
      return Ok(unevaluated("GaussianFilter", args));
    }
    other => {
      if let Some(radius) = expr_to_f64_opt(other) {
        gaussian_axis_from_radius(radius)
      } else {
        crate::emit_message(&format!(
          "GaussianFilter::bdrad: The radius specification {} must be a \
         non-complex number or a nonempty list of non-complex numbers.",
          crate::syntax::format_expr(other, crate::syntax::ExprForm::Output)
        ));
        return Ok(unevaluated("GaussianFilter", args));
      }
    }
  };
  let (radius, sigma) = (axis.radius, axis.sigma);

  let kernel = gaussian_filter_kernel(radius, sigma);

  // Scalar volume: the separable kernel runs along each of the three axes.
  if let Some((dims, vals, opts)) = image3d_scalar_volume(&args[0]) {
    let [d0, d1, d2] = dims;
    let strides = [d1 * d2, d2, 1];
    let mut cur = vals;
    for (axis, &len) in dims.iter().enumerate() {
      let stride = strides[axis];
      let mut next = cur.clone();
      for start in 0..cur.len() {
        // Visit each line once, from its first element along `axis`.
        if (start / stride) % len != 0 {
          continue;
        }
        let line: Vec<f64> =
          (0..len).map(|t| cur[start + t * stride]).collect();
        let out =
          crate::functions::math_ast::convolve_edge_padded(&line, &kernel);
        for (t, v) in out.into_iter().enumerate() {
          next[start + t * stride] = v;
        }
      }
      cur = next;
    }
    debug_assert_eq!(cur.len(), d0 * d1 * d2);
    return Ok(image3d_from_volume(dims, &cur, opts));
  }

  // List input (1D).
  if let Expr::List(items) = &args[0] {
    let data: Vec<f64> = items
      .iter()
      .filter_map(crate::functions::math_ast::try_eval_to_f64)
      .collect();
    if data.len() != items.len() {
      return Ok(unevaluated("GaussianFilter", args));
    }
    let result =
      crate::functions::math_ast::convolve_edge_padded(&data, &kernel);
    return Ok(Expr::List(result.into_iter().map(Expr::Real).collect()));
  }

  // Image input: the same radius applies along both axes. Unlike `Blur`,
  // the result is never quantized back onto an integer image's grid;
  // wolframscript hands back `"Real32"` pixels for every integer type.
  if let Some(mut filtered) = gaussian_filter_image(&args[0], axis, axis) {
    if let Expr::Image { image_type, .. } = &mut filtered
      && *image_type != ImageType::Real64
    {
      *image_type = ImageType::Real32;
    }
    return Ok(filtered);
  }

  crate::emit_message(&format!(
    "GaussianFilter::arg1: The first argument {} should be a rectangular array, image or video.",
    expr_to_string(&args[0])
  ));
  Ok(unevaluated("GaussianFilter", args))
}

/// Round a filtered image back onto the grid its type can store, as
/// `Blur` and `Sharpen` do. A `"Byte"` image holds whole values in
/// `0..255` and a `"Bit16"` image whole values in `0..65535`, so a filter
/// result is rounded and clipped to that range. A `"Bit"` image cannot
/// represent a blur at all, so it is promoted to `"Real32"` instead of
/// being flattened back to two levels. Real images are left alone.
fn requantize_filtered(mut image: Expr) -> Expr {
  let Expr::Image {
    data, image_type, ..
  } = &mut image
  else {
    return image;
  };
  let levels = match image_type {
    ImageType::Byte => 255.0,
    ImageType::Bit16 => 65535.0,
    ImageType::Bit => {
      *image_type = ImageType::Real32;
      return image;
    }
    ImageType::Real32 | ImageType::Real64 => {
      return image;
    }
  };
  *data = Arc::new(
    data
      .iter()
      .map(|value| (value * levels).round().clamp(0.0, levels) / levels)
      .collect(),
  );
  image
}

/// The half-width and standard deviation of a separable Gaussian filter
/// along one axis.
#[derive(Clone, Copy)]
struct GaussianAxis {
  radius: usize,
  sigma: f64,
}

/// Turn a `Blur`/`GaussianFilter` pixel radius into the kernel it names:
/// half-width `Ceiling[r]` and standard deviation `r/2`, as
/// wolframscript does. A radius of zero or less filters with nothing,
/// leaving the image untouched.
fn gaussian_axis_from_radius(radius: f64) -> GaussianAxis {
  if radius <= 0.0 {
    return GaussianAxis {
      radius: 0,
      sigma: 0.0,
    };
  }
  GaussianAxis {
    radius: radius.ceil() as usize,
    sigma: radius / 2.0,
  }
}

/// Apply a separable discrete-Gaussian filter to an image: `horizontal`
/// convolves each row, `vertical` each resulting column. Both passes
/// pad by repeating the edge pixel, matching wolframscript's default
/// `Padding -> "Fixed"`. Returns `None` for a non-image argument.
fn gaussian_filter_image(
  image: &Expr,
  vertical: GaussianAxis,
  horizontal: GaussianAxis,
) -> Option<Expr> {
  let Expr::Image {
    color_space: _,
    display_size: _,
    width,
    height,
    channels,
    data,
    image_type,
  } = image
  else {
    return None;
  };
  let w = *width as usize;
  let h = *height as usize;
  let ch = *channels as usize;
  if w == 0 || h == 0 {
    return Some(image.clone());
  }
  let row_kernel = gaussian_filter_kernel(horizontal.radius, horizontal.sigma);
  let column_kernel = gaussian_filter_kernel(vertical.radius, vertical.sigma);

  let mut out: Vec<f64> = vec![0.0; data.len()];
  for c_idx in 0..ch {
    let mut row_filtered: Vec<f64> = vec![0.0; w * h];
    for y in 0..h {
      let row: Vec<f64> =
        (0..w).map(|x| data[(y * w + x) * ch + c_idx]).collect();
      let filtered =
        crate::functions::math_ast::convolve_edge_padded(&row, &row_kernel);
      for x in 0..w {
        row_filtered[y * w + x] = filtered[x];
      }
    }
    for x in 0..w {
      let col: Vec<f64> = (0..h).map(|y| row_filtered[y * w + x]).collect();
      let filtered =
        crate::functions::math_ast::convolve_edge_padded(&col, &column_kernel);
      for y in 0..h {
        out[(y * w + x) * ch + c_idx] = filtered[y];
      }
    }
  }
  Some(Expr::Image {
    color_space: None,
    display_size: None,
    width: *width,
    height: *height,
    channels: *channels,
    data: Arc::new(out),
    image_type: *image_type,
  })
}

/// Build the 1D Gaussian-filter kernel of half-width `radius` and
/// standard deviation `sigma` using the Bessel-based discrete
/// Gaussian: `T_k(t) = exp(-t)·I_k(t)` with `t = sigma²`. The kernel is
/// emitted for `k ∈ [-radius, radius]` and normalised to sum to 1 —
/// this matches wolframscript's `GaussianMatrix[{radius, sigma}]`
/// values byte-for-byte (e.g. `{0.0994, 0.8012, 0.0994}` for r=1).
fn gaussian_filter_kernel(radius: usize, sigma: f64) -> Vec<f64> {
  let t = sigma * sigma;
  let len = 2 * radius + 1;
  let mut kernel: Vec<f64> = Vec::with_capacity(len);
  let exp_neg_t = (-t).exp();
  for k_signed in -(radius as i64)..=(radius as i64) {
    let k_abs = k_signed.unsigned_abs() as f64;
    let i_k = crate::functions::math_ast::bessel_i(k_abs, t);
    kernel.push(exp_neg_t * i_k);
  }
  let sum: f64 = kernel.iter().sum();
  if sum > 0.0 {
    for v in &mut kernel {
      *v /= sum;
    }
  }
  kernel
}

/// GaussianMatrix[r] / GaussianMatrix[{r, sigma}] — produce the
/// (2r+1)×(2r+1) discrete Gaussian matrix. The matrix is the outer
/// product of the 1D discrete-Gaussian kernel (`gaussian_filter_kernel`)
/// with itself, matching wolframscript byte-for-byte. With the integer
/// form `GaussianMatrix[r]`, the standard deviation defaults to `r/2`.
pub fn gaussian_matrix_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Resolve radius and sigma from the single argument.
  let (radius, sigma): (usize, f64) = match &args[0] {
    // GaussianMatrix[{r, sigma}]
    Expr::List(spec) if spec.len() == 2 => {
      let r = match expr_to_f64_opt(&spec[0]) {
        Some(v) if v >= 0.0 && v.fract() == 0.0 => v as usize,
        _ => return Ok(symbolic_gaussian_matrix(args)),
      };
      let s = match expr_to_f64_opt(&spec[1]) {
        Some(v) if v > 0.0 => v,
        _ => return Ok(symbolic_gaussian_matrix(args)),
      };
      (r, s)
    }
    // GaussianMatrix[r] with integer r; sigma defaults to r/2.
    other => match expr_to_f64_opt(other) {
      Some(v) if v >= 0.0 && v.fract() == 0.0 => (v as usize, v / 2.0),
      _ => return Ok(symbolic_gaussian_matrix(args)),
    },
  };

  if sigma <= 0.0 {
    return Ok(symbolic_gaussian_matrix(args));
  }

  // GaussianMatrix[r, {n1, n2}]: derivative orders along rows and columns;
  // a single order `n` is `{n, n}`.
  if let Some(orders) = args.get(1) {
    let order = |e: &Expr| match e {
      Expr::Integer(n) if *n >= 0 && *n <= u32::MAX as i128 => Some(*n as u32),
      _ => None,
    };
    let orders: Option<Vec<u32>> = match orders {
      Expr::List(ns) => ns.iter().map(order).collect(),
      single => order(single).map(|n| vec![n, n]),
    };
    return Ok(match orders.as_deref() {
      Some(&[n1, n2]) => gaussian_derivative_matrix(radius, sigma, n1, n2),
      // Orders for a higher-rank array are not implemented.
      Some(_) => symbolic_gaussian_matrix(args),
      None => {
        crate::emit_message(&format!(
          "GaussianMatrix::ilsmn: Single or list of non-negative \
           machine-sized integers expected at position 2 of {}.",
          expr_to_string(&symbolic_gaussian_matrix(args))
        ));
        symbolic_gaussian_matrix(args)
      }
    });
  }

  let kernel = gaussian_filter_kernel(radius, sigma);
  let rows: Vec<Expr> = kernel
    .iter()
    .map(|&a| {
      Expr::List(
        kernel
          .iter()
          .map(|&b| Expr::Real(a * b))
          .collect::<Vec<_>>()
          .into(),
      )
    })
    .collect::<Vec<_>>();
  Ok(Expr::List(rows.into()))
}

/// The 1-D factor of order `n` of a Gaussian derivative matrix: the
/// discrete Gaussian `T(k) = e^-t I_|k|(t)` (`t = sigma^2`) for `n = 0`,
/// normalized over the window, and otherwise the central difference applied
/// `n` times to the untruncated `T`,
/// `sum_i (-1)^i Binomial[n, i] T(k + n - 2 i) / 2^n`.
fn gaussian_derivative_factor(radius: usize, sigma: f64, n: u32) -> Vec<f64> {
  if n == 0 {
    return gaussian_filter_kernel(radius, sigma);
  }
  let t = sigma * sigma;
  let discrete = |k: i64| {
    (-t).exp()
      * crate::functions::math_ast::bessel_i(k.unsigned_abs() as f64, t)
  };
  let n = n as i64;
  let scale = 2f64.powi(n as i32);
  // The factor is even or odd with `n`: computed for `k >= 0` and mirrored,
  // so that symmetric entries agree to the last bit.
  let half: Vec<f64> = (0..=radius as i64)
    .map(|k| {
      // An odd factor vanishes at the center; the sum would leave rounding.
      if k == 0 && n % 2 == 1 {
        return 0.0;
      }
      let mut binomial = 1.0;
      let mut sum = 0.0;
      for i in 0..=n {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        sum += sign * binomial * discrete(k + n - 2 * i);
        binomial = binomial * (n - i) as f64 / (i + 1) as f64;
      }
      sum / scale
    })
    .collect();
  let mirror = if n % 2 == 0 { 1.0 } else { -1.0 };
  half[1..]
    .iter()
    .rev()
    .map(|v| mirror * v)
    .chain(half.iter().copied())
    .collect()
}

/// `GaussianMatrix[{r, sigma}, {n1, n2}]`, as wolframscript builds it: the
/// outer product of the row factor of order `n1` and the column factor of
/// order `n2` (`gaussian_derivative_factor`), less its mean when both
/// orders are even (an odd one already sums to zero), scaled so that
/// `sum M[i, j] i^n1 j^n2 = (-1)^(n1+n2) n1! n2!` over the offsets `i, j`
/// from the center. Derivatives of a constant vanish and those of the
/// matching monomial come out exactly.
fn gaussian_derivative_matrix(
  radius: usize,
  sigma: f64,
  n1: u32,
  n2: u32,
) -> Expr {
  let rows = gaussian_derivative_factor(radius, sigma, n1);
  let cols = gaussian_derivative_factor(radius, sigma, n2);
  let mut m: Vec<Vec<f64>> = rows
    .iter()
    .map(|a| cols.iter().map(|b| a * b).collect())
    .collect();
  if n1 + n2 > 0 {
    let size = (2 * radius + 1) as f64;
    let mean = if n1.is_multiple_of(2) && n2.is_multiple_of(2) {
      m.iter().flatten().sum::<f64>() / (size * size)
    } else {
      0.0
    };
    let offset = |i: usize| i as f64 - radius as f64;
    let mut moment = 0.0;
    for (i, row) in m.iter_mut().enumerate() {
      for (j, v) in row.iter_mut().enumerate() {
        *v -= mean;
        moment += *v * offset(i).powi(n1 as i32) * offset(j).powi(n2 as i32);
      }
    }
    let factorial = |n: u32| (1..=n).map(f64::from).product::<f64>();
    let sign = if (n1 + n2).is_multiple_of(2) {
      1.0
    } else {
      -1.0
    };
    let scale = sign * factorial(n1) * factorial(n2) / moment;
    for v in m.iter_mut().flatten() {
      *v *= scale;
    }
  }
  Expr::List(
    m.into_iter()
      .map(|row| Expr::List(row.into_iter().map(Expr::Real).collect()))
      .collect(),
  )
}

/// Best-effort numeric extraction shared by GaussianMatrix's argument
/// parsing. Returns `None` for non-numeric expressions so the caller can
/// keep the call symbolic.
fn expr_to_f64_opt(e: &Expr) -> Option<f64> {
  match e {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(r) => Some(*r),
    _ => None,
  }
}

fn symbolic_gaussian_matrix(args: &[Expr]) -> Expr {
  unevaluated("GaussianMatrix", args)
}

/// Colorize[matrix, opts] — colorize an integer-label matrix as an RGB
/// byte image. Background label 0 is black. With the default
/// `ColorFunction -> Automatic` every other label gets a distinct hue; with
/// an explicit `ColorFunction -> f`, `f` is applied to each label's place
/// in `[0, 1]`; `f` may be a function, a named `ColorData` gradient
/// string, or a color-function head such as `Hue`. Non-matrix /
/// non-image arguments emit `Colorize::invinput` and stay symbolic,
/// matching wolframscript.
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
    for row in rows {
      if let Expr::List(cols) = row {
        for c in cols {
          if let Expr::Integer(n) = c {
            labels.push(*n as i64);
          }
        }
      }
    }
    let max = labels.iter().copied().max().unwrap_or(0).max(1) as f64;
    let color_function = option_value(&args[1..], "ColorFunction")
      .filter(|f| !matches!(f, Expr::Identifier(n) if n == "Automatic"));
    // Each distinct label is colored once and reused for every pixel. Label
    // k of m takes the color at the middle (k - 1/2)/m of its share of
    // [0, 1]; label 0 is the black background. The result is a byte image
    // whose channel value v falls in byte ceil(256 v) - 1, the way
    // wolframscript quantizes it (0.125 is byte 31, 0.9 byte 230).
    let mut palette: std::collections::HashMap<i64, (f64, f64, f64)> =
      std::collections::HashMap::new();
    let mut data: Vec<f64> = Vec::with_capacity(labels.len() * 3);
    let to_byte = |c: f64| {
      if c <= 0.0 {
        0.0
      } else {
        ((c * 256.0).ceil() - 1.0).clamp(0.0, 255.0) / 255.0
      }
    };
    for v in &labels {
      let (r, g, b) = *palette.entry(*v).or_insert_with(|| {
        if *v == 0 {
          return (0.0, 0.0, 0.0);
        }
        let t = (*v as f64 - 0.5) / max;
        let (r, g, b) = colorize_label(*v, t, color_function);
        (to_byte(r), to_byte(g), to_byte(b))
      });
      data.extend([r, g, b]);
    }
    return Ok(Expr::Image {
      color_space: None,
      display_size: None,
      width,
      height,
      channels: 3,
      data: Arc::new(data),
      image_type: ImageType::Byte,
    });
  }
  // An image is colored through a lookup table, the way wolframscript
  // does it: a colour image is taken to grayscale, values are clipped to
  // [0, 1], and a pixel x takes the table entry i = min(floor(1024 x),
  // 1023) (for an integer image type its raw integer value). The table
  // runs up to imax, the largest entry used, and is filled in full, after
  // probing the function at 0, 1/2 and 1 twice; entry i is the
  // `ColorFunction` at i/imax (i/1023 with `ColorFunctionScaling ->
  // False`). The result is an RGB byte image.
  if let Expr::Image {
    channels,
    image_type,
    ..
  } = &args[0]
    && let Some(cf) = option_value(&args[1..], "ColorFunction")
      .filter(|f| !matches!(f, Expr::Identifier(n) if n == "Automatic"))
  {
    let gray = if *channels == 1 {
      args[0].clone()
    } else {
      crate::evaluator::evaluate_expr_to_expr(&call(
        "ColorConvert",
        vec![args[0].clone(), Expr::String("Grayscale".into())],
      ))?
    };
    let Expr::Image {
      width,
      height,
      channels: 1,
      data,
      ..
    } = &gray
    else {
      return Ok(unevaluated("Colorize", args));
    };
    let scaled = !matches!(
      option_value(&args[1..], "ColorFunctionScaling"),
      Some(Expr::Identifier(n)) if n == "False"
    );
    let integer_range = match image_type {
      ImageType::Bit => Some(1.0),
      ImageType::Byte => Some(255.0),
      ImageType::Bit16 => Some(65535.0),
      _ => None,
    };
    let index = |v: f64| match integer_range {
      Some(range) => (v.clamp(0.0, 1.0) * range).round() as usize,
      None => ((v.clamp(0.0, 1.0) * 1024.0).floor() as usize).min(1023),
    };
    let imax = data.iter().map(|&v| index(v)).max().unwrap_or(0);
    let to_byte = |c: f64| {
      if c <= 0.0 {
        0.0
      } else {
        ((c * 256.0).ceil() - 1.0).clamp(0.0, 255.0) / 255.0
      }
    };
    for _ in 0..2 {
      for t in [0.0, 0.5, 1.0] {
        colorize_label(0, t, Some(cf));
      }
    }
    let table: Vec<(f64, f64, f64)> = (0..=imax)
      .map(|i| {
        let t = if !scaled {
          i as f64 * (1.0 / 1023.0)
        } else if imax == 0 {
          0.0
        } else {
          i as f64 * (1.0 / imax as f64)
        };
        let (r, g, b) = colorize_label(0, t, Some(cf));
        (to_byte(r), to_byte(g), to_byte(b))
      })
      .collect();
    let mut out: Vec<f64> = Vec::with_capacity(data.len() * 3);
    for &v in data.iter() {
      let (r, g, b) = table[index(v)];
      out.extend([r, g, b]);
    }
    return Ok(Expr::Image {
      color_space: Some("RGB"),
      display_size: None,
      width: *width,
      height: *height,
      channels: 3,
      data: Arc::new(out),
      image_type: ImageType::Byte,
    });
  }
  let is_image = matches!(&args[0], Expr::Image { .. });
  if !is_image {
    crate::emit_message(&format!(
      "Colorize::invinput: Expecting an integer matrix or an image instead of {}.",
      expr_to_string(&args[0])
    ));
  }
  Ok(unevaluated("Colorize", args))
}

/// The RGB color of one `Colorize` label. `t` is the label rescaled to
/// `[0, 1]`; `color_function` is `None` for the automatic palette.
fn colorize_label(
  label: i64,
  t: f64,
  color_function: Option<&Expr>,
) -> (f64, f64, f64) {
  let Some(f) = color_function else {
    // Golden-ratio hue steps keep neighboring labels visually distinct.
    let hue = (label as f64 * 0.618_033_988_749_895).rem_euclid(1.0);
    return hsv_to_rgb(hue, 0.6, 0.9);
  };
  let color = match f {
    Expr::String(scheme) => {
      crate::functions::chart::sample_named_gradient(scheme, t)
        .map(|(r, g, b)| {
          call(
            "RGBColor",
            vec![Expr::Real(r), Expr::Real(g), Expr::Real(b)],
          )
        })
        .or_else(|| {
          let scheme_fn = crate::evaluator::evaluate_expr_to_expr(&call(
            "ColorData",
            vec![f.clone()],
          ))
          .ok()?;
          crate::evaluator::apply_function_to_arg(&scheme_fn, &Expr::Real(t))
            .ok()
        })
    }
    Expr::Identifier(head) => {
      crate::evaluator::evaluate_expr_to_expr(&call(head, vec![Expr::Real(t)]))
        .ok()
    }
    other => {
      crate::evaluator::apply_function_to_arg(other, &Expr::Real(t)).ok()
    }
  };
  color
    .and_then(|c| color_directive_to_rgb(&c))
    .map_or((t, t, t), |(r, g, b, _)| (r, g, b))
}

/// The positions of the 8-connected foreground neighbors of pixel `i`.
/// Everything beyond the border counts as background.
fn neighbors_of(
  alive: &[bool],
  w: usize,
  h: usize,
  i: usize,
) -> Vec<(usize, usize)> {
  let (y, x) = (i / w, i % w);
  let mut out = Vec::with_capacity(8);
  for dy in -1i64..=1 {
    for dx in -1i64..=1 {
      if dy == 0 && dx == 0 {
        continue;
      }
      let ny = y as i64 + dy;
      let nx = x as i64 + dx;
      if ny < 0 || nx < 0 || ny >= h as i64 || nx >= w as i64 {
        continue;
      }
      if alive[ny as usize * w + nx as usize] {
        out.push((ny as usize, nx as usize));
      }
    }
  }
  out
}

/// Whether pixel `i` is the tip of a branch: its foreground neighbors form
/// a single 8-connected clump, and at most one of them is edge-adjacent.
///
/// The clump condition is what makes the end of a diagonal staircase a tip
/// even though it has two neighbors; the 4-neighbor cap is what stops the
/// corner of a solid block — whose three neighbors are also one clump —
/// from counting as one. A pixel with no neighbors at all is an isolated
/// point rather than a tip, and survives.
fn is_branch_tip(alive: &[bool], w: usize, h: usize, i: usize) -> bool {
  if !alive[i] {
    return false;
  }
  let neighbors = neighbors_of(alive, w, h, i);
  if neighbors.is_empty() {
    return false;
  }
  let (y, x) = (i / w, i % w);
  let edge_adjacent = neighbors
    .iter()
    .filter(|(ny, nx)| ny.abs_diff(y) + nx.abs_diff(x) == 1)
    .count();
  if edge_adjacent > 1 {
    return false;
  }
  // One clump: grow from the first neighbor through 8-adjacency.
  let mut seen = vec![false; neighbors.len()];
  seen[0] = true;
  let mut stack = vec![0usize];
  let mut reached = 1;
  while let Some(a) = stack.pop() {
    for b in 0..neighbors.len() {
      if seen[b] {
        continue;
      }
      if neighbors[a].0.abs_diff(neighbors[b].0) <= 1
        && neighbors[a].1.abs_diff(neighbors[b].1) <= 1
      {
        seen[b] = true;
        reached += 1;
        stack.push(b);
      }
    }
  }
  reached == neighbors.len()
}

/// Label the 8-connected components of a foreground mask. The result has
/// one entry per pixel: `Some(label)` for a foreground pixel, `None` for a
/// background one. Labels are numbered from 0 in raster order of their
/// first pixel.
fn connected_components(
  alive: &[bool],
  w: usize,
  h: usize,
) -> Vec<Option<usize>> {
  let mut labels: Vec<Option<usize>> = vec![None; w * h];
  let mut next = 0usize;
  let mut stack: Vec<(usize, usize)> = Vec::new();
  for start in 0..w * h {
    if !alive[start] || labels[start].is_some() {
      continue;
    }
    let label = next;
    next += 1;
    labels[start] = Some(label);
    stack.push((start / w, start % w));
    while let Some((y, x)) = stack.pop() {
      for dy in -1i64..=1 {
        for dx in -1i64..=1 {
          let ny = y as i64 + dy;
          let nx = x as i64 + dx;
          if ny < 0 || nx < 0 || ny >= h as i64 || nx >= w as i64 {
            continue;
          }
          let i = ny as usize * w + nx as usize;
          if alive[i] && labels[i].is_none() {
            labels[i] = Some(label);
            stack.push((ny as usize, nx as usize));
          }
        }
      }
    }
  }
  labels
}

/// Pruning[img] / Pruning[img, n] — remove the outermost branches of the
/// thin objects in an image by setting their pixels to black.
///
/// One pass deletes every *endpoint*: a foreground pixel whose 8-connected
/// foreground neighbors are either a single pixel or two pixels that touch
/// each other (the tip of a staircase). A pixel with three or more
/// neighbors is a *branch point*.
///
/// `Pruning[img, n]` runs `n` such passes — `Infinity` runs them until
/// nothing more falls away — and prunes only the connected components that
/// contain a branch point, so a component that is one unbranched arc is a
/// shape rather than a branch and is left whole however long the pass count
/// is.
///
/// `Pruning[img]` is a different operation, not `n = 1`: it erodes from
/// both ends without that protection, so every thin arc is worn down to a
/// single pixel. wolframscript draws the same distinction between the two
/// forms.
///
/// A pixel with no foreground neighbor at all is an isolated point, not a
/// branch tip, and survives. Surviving pixels keep their original value,
/// so grayscale detail is preserved and only the pruned pixels go to 0;
/// each channel of a multichannel image is pruned on its own. Pixels
/// beyond the border count as background (`Padding -> 0`, the default).
pub fn pruning_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("Pruning", args);
  let Expr::Image {
    color_space,
    display_size: _,
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
  else {
    crate::emit_message(&format!(
      "Pruning::imginv: Expecting an image or graphics instead of {}.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated());
  };
  if args.len() > 2 {
    return Ok(unevaluated());
  }
  // The branch-length limit: a non-negative integer, or Infinity for
  // "until stable". Anything else is reported as `invniter` and left
  // unevaluated. With no limit given at all every arc is worn down to a
  // point, which is a different operation — see the doc comment.
  let passes: usize = match args.get(1) {
    None => usize::MAX,
    Some(Expr::Identifier(name)) if name == "Infinity" => usize::MAX,
    Some(Expr::Integer(n)) if *n >= 0 => *n as usize,
    Some(_) => {
      crate::emit_message(
        "Pruning::invniter: Expecting a non-negative integer value for the \
         number of iterations.",
      );
      return Ok(unevaluated());
    }
  };
  // Only the explicit-count form protects unbranched components; the bare
  // `Pruning[img]` wears every arc down to a single pixel.
  let protect_unbranched = args.len() > 1;

  let w = *width as usize;
  let h = *height as usize;
  let ch = *channels as usize;
  if w == 0 || h == 0 || ch == 0 || passes == 0 {
    return Ok(args[0].clone());
  }
  let mut new_data = data.as_ref().clone();
  for c_idx in 0..ch {
    // `alive[y * w + x]` tracks whether the pixel is still foreground.
    let mut alive: Vec<bool> = (0..w * h)
      .map(|i| new_data[i * ch + c_idx] != 0.0)
      .collect();
    // `Pruning[img, n]` only touches components that had a branch point to
    // begin with: a component that is one unbranched arc is a shape, not a
    // branch, and survives however many passes are asked for. The bare
    // `Pruning[img]` has no such protection.
    let prunable: Vec<bool> = if protect_unbranched {
      let labels = connected_components(&alive, w, h);
      let count = labels.iter().flatten().max().map_or(0, |l| l + 1);
      let mut branched = vec![false; count];
      for i in 0..w * h {
        if let Some(label) = labels[i]
          && neighbors_of(&alive, w, h, i).len() >= 3
        {
          branched[label] = true;
        }
      }
      (0..w * h)
        .map(|i| labels[i].is_some_and(|label| branched[label]))
        .collect()
    } else {
      vec![true; w * h]
    };

    for _ in 0..passes {
      let labels = connected_components(&alive, w, h);
      let component_count = labels.iter().flatten().max().map_or(0, |l| l + 1);
      let doomed_mask: Vec<bool> = (0..w * h)
        .map(|i| prunable[i] && is_branch_tip(&alive, w, h, i))
        .collect();
      let mut doomed: Vec<usize> =
        (0..w * h).filter(|&i| doomed_mask[i]).collect();

      // A component all of whose pixels are tips would vanish altogether.
      // `Pruning[img, n]` leaves such a component alone for good; the bare
      // form keeps its first pixel, so a thin arc wears down to a single
      // point rather than to nothing.
      let mut all_tips = vec![true; component_count];
      for i in 0..w * h {
        if let Some(label) = labels[i]
          && !doomed_mask[i]
        {
          all_tips[label] = false;
        }
      }
      let mut spared = vec![false; component_count];
      doomed.retain(|&i| {
        let Some(label) = labels[i] else { return true };
        if !all_tips[label] {
          return true;
        }
        if protect_unbranched {
          return false; // the whole component stays
        }
        if spared[label] {
          return true;
        }
        spared[label] = true;
        false
      });

      if doomed.is_empty() {
        break; // stable; further passes would change nothing
      }
      for &i in &doomed {
        alive[i] = false;
        new_data[i * ch + c_idx] = 0.0;
      }
    }
  }
  Ok(Expr::Image {
    color_space: *color_space,
    display_size: None,
    width: *width,
    height: *height,
    channels: *channels,
    data: Arc::new(new_data),
    image_type: *image_type,
  })
}

/// The pixels of an image as a gray plane `(width, height, values)` in
/// raster order (top row first); `None` for anything that is not an image.
fn image_gray_plane(e: &Expr) -> Option<(usize, usize, Vec<f64>)> {
  let rows = image_gray_matrix(e)?;
  let h = rows.len();
  let w = rows.first().map_or(0, Vec::len);
  Some((w, h, rows.into_iter().flatten().collect()))
}

/// The pixels around `(y, x)`, clockwise from the one above:
/// N, NE, E, SE, S, SW, W, NW. Anything beyond the border is `false`.
fn pixel_ring(
  mask: &[bool],
  w: usize,
  h: usize,
  y: usize,
  x: usize,
) -> [bool; 8] {
  const OFFSETS: [(i64, i64); 8] = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
  ];
  let mut ring = [false; 8];
  for (slot, (dy, dx)) in ring.iter_mut().zip(OFFSETS) {
    let ny = y as i64 + dy;
    let nx = x as i64 + dx;
    if ny >= 0 && nx >= 0 && ny < h as i64 && nx < w as i64 {
      *slot = mask[ny as usize * w + nx as usize];
    }
  }
  ring
}

/// How many separate runs of set pixels the ring contains.
fn ring_runs(ring: [bool; 8]) -> usize {
  (0..8).filter(|&i| !ring[i] && ring[(i + 1) % 8]).count()
}

/// Reduce a foreground mask to a one-pixel-wide skeleton (Zhang–Suen
/// thinning). Isolated pixels and the tips of thin lines survive.
fn thin_to_skeleton(fg: &mut [bool], w: usize, h: usize) {
  loop {
    let mut changed = false;
    for second_pass in [false, true] {
      let mut doomed = Vec::new();
      for y in 0..h {
        for x in 0..w {
          if !fg[y * w + x] {
            continue;
          }
          let ring = pixel_ring(fg, w, h, y, x);
          let neighbors = ring.iter().filter(|&&p| p).count();
          if !(2..=6).contains(&neighbors) || ring_runs(ring) != 1 {
            continue;
          }
          let (n, e, s, west) = (ring[0], ring[2], ring[4], ring[6]);
          let keep = if second_pass {
            n && west && (e || s)
          } else {
            e && s && (n || west)
          };
          if !keep {
            doomed.push(y * w + x);
          }
        }
      }
      changed |= !doomed.is_empty();
      for i in doomed {
        fg[i] = false;
      }
    }
    if !changed {
      break;
    }
  }
}

/// A key that orders `f64`s the way `f64::total_cmp` does.
fn total_order_key(v: f64) -> u64 {
  let bits = v.to_bits();
  if bits >> 63 == 1 {
    !bits
  } else {
    bits | (1 << 63)
  }
}

/// WatershedComponents[image] / WatershedComponents[image, markers] —
/// flood the image as a height field from its markers and label every
/// pixel with the number of the basin it drains into.
///
/// Without markers, each regional minimum (a connected plateau with no
/// lower neighbor) seeds a basin; the basins are numbered from 1 in raster
/// order of their first pixel. Explicit markers are a matrix of integers
/// the size of the image, where 0 means "not a marker". Pixels reached by
/// two different basins at once form the watershed lines and are labeled 0.
/// Pixels touch their eight neighbors.
pub fn watershed_components_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("WatershedComponents", args);
  if args.is_empty() || args.len() > 2 {
    return Ok(unevaluated());
  }
  let Some((w, h, gray)) = image_gray_plane(&args[0]) else {
    crate::emit_message(&format!(
      "WatershedComponents::imginv: Expecting an image or graphics instead \
       of {}.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated());
  };
  let n = w * h;
  // `labels`: 0 = unlabeled, -1 = watershed line, otherwise the basin.
  let mut labels = vec![0i64; n];
  match args.get(1) {
    Some(Expr::List(rows)) => {
      let parsed: Option<Vec<i64>> = (rows.len() == h)
        .then(|| {
          rows
            .iter()
            .map(|row| match row {
              Expr::List(cells) if cells.len() == w => cells
                .iter()
                .map(|c| match c {
                  Expr::Integer(v) if *v >= 0 => i64::try_from(*v).ok(),
                  _ => None,
                })
                .collect::<Option<Vec<i64>>>(),
              _ => None,
            })
            .collect::<Option<Vec<Vec<i64>>>>()
            .map(|r| r.into_iter().flatten().collect())
        })
        .flatten();
      match parsed {
        Some(markers) => labels = markers,
        None => return Ok(unevaluated()),
      }
    }
    Some(_) => return Ok(unevaluated()),
    None => {
      let mut next = 0i64;
      let mut seen = vec![false; n];
      for start in 0..n {
        if seen[start] {
          continue;
        }
        let level = gray[start];
        let mut plateau = vec![start];
        seen[start] = true;
        let mut is_minimum = true;
        let mut k = 0;
        while k < plateau.len() {
          let (y, x) = (plateau[k] / w, plateau[k] % w);
          k += 1;
          for dy in -1i64..=1 {
            for dx in -1i64..=1 {
              let ny = y as i64 + dy;
              let nx = x as i64 + dx;
              if (dy == 0 && dx == 0)
                || ny < 0
                || nx < 0
                || ny >= h as i64
                || nx >= w as i64
              {
                continue;
              }
              let j = ny as usize * w + nx as usize;
              if gray[j] < level {
                is_minimum = false;
              } else if gray[j] == level && !seen[j] {
                seen[j] = true;
                plateau.push(j);
              }
            }
          }
        }
        if is_minimum {
          next += 1;
          for &p in &plateau {
            labels[p] = next;
          }
        }
      }
    }
  }

  let mut queued = vec![false; n];
  let mut heap = std::collections::BinaryHeap::new();
  let mut seq = 0u64;
  for i in 0..n {
    if labels[i] > 0 {
      queued[i] = true;
      heap.push(std::cmp::Reverse((total_order_key(gray[i]), seq, i)));
      seq += 1;
    }
  }
  while let Some(std::cmp::Reverse((_, _, i))) = heap.pop() {
    let (y, x) = (i / w, i % w);
    let mut neighbors = Vec::with_capacity(8);
    for dy in -1i64..=1 {
      for dx in -1i64..=1 {
        let ny = y as i64 + dy;
        let nx = x as i64 + dx;
        if (dy == 0 && dx == 0)
          || ny < 0
          || nx < 0
          || ny >= h as i64
          || nx >= w as i64
        {
          continue;
        }
        neighbors.push(ny as usize * w + nx as usize);
      }
    }
    if labels[i] == 0 {
      let mut basin = 0i64;
      let mut contested = false;
      for &j in &neighbors {
        if labels[j] > 0 {
          if basin == 0 {
            basin = labels[j];
          } else if basin != labels[j] {
            contested = true;
          }
        }
      }
      if contested {
        labels[i] = -1;
        continue;
      }
      labels[i] = basin;
    }
    for &j in &neighbors {
      if !queued[j] && labels[j] == 0 {
        queued[j] = true;
        heap.push(std::cmp::Reverse((total_order_key(gray[j]), seq, j)));
        seq += 1;
      }
    }
  }

  Ok(Expr::List(
    (0..h)
      .map(|y| {
        Expr::List(
          (0..w)
            .map(|x| Expr::Integer(labels[y * w + x].max(0) as i128))
            .collect::<Vec<_>>()
            .into(),
        )
      })
      .collect::<Vec<_>>()
      .into(),
  ))
}

/// MorphologicalGraph[image] — the graph of the skeleton of the nonzero
/// pixels of an image. Its vertices are the skeleton's end points, branch
/// points and isolated points (touching ones merged into one vertex, placed
/// at their centroid, with `y` pointing up as in `ImageDimensions`), and its
/// edges are the skeleton arcs joining them. A closed loop without any
/// branch point becomes a single vertex. Vertices are numbered in raster
/// order of their first pixel. Trailing options are passed on to `Graph`.
pub fn morphological_graph_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("MorphologicalGraph", args);
  if args.is_empty() {
    return Ok(unevaluated());
  }
  let Some((w, h, gray)) = image_gray_plane(&args[0]) else {
    crate::emit_message(&format!(
      "MorphologicalGraph::imginv: Expecting an image or graphics instead \
       of {}.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated());
  };
  if !args[1..]
    .iter()
    .all(|a| matches!(a, Expr::Rule { .. } | Expr::RuleDelayed { .. }))
  {
    return Ok(unevaluated());
  }
  let mut skeleton: Vec<bool> = gray.iter().map(|&v| v != 0.0).collect();
  thin_to_skeleton(&mut skeleton, w, h);

  // Vertices sit at end points (one run of neighbors), branch points (three
  // or more runs) and isolated pixels.
  let is_vertex_pixel: Vec<bool> = (0..w * h)
    .map(|i| {
      skeleton[i] && {
        let ring = pixel_ring(&skeleton, w, h, i / w, i % w);
        ring_runs(ring) != 2
      }
    })
    .collect();
  let vertex_of = connected_components(&is_vertex_pixel, w, h);
  let mut vertex_count = vertex_of.iter().flatten().max().map_or(0, |l| l + 1);
  let mut sums = vec![(0.0f64, 0.0f64, 0usize); vertex_count];
  for (i, label) in vertex_of.iter().enumerate() {
    if let Some(v) = label {
      sums[*v].0 += (i % w) as f64 + 0.5;
      sums[*v].1 += h as f64 - ((i / w) as f64 + 0.5);
      sums[*v].2 += 1;
    }
  }
  let mut coordinates: Vec<(f64, f64)> = sums
    .iter()
    .map(|&(sx, sy, c)| (sx / c as f64, sy / c as f64))
    .collect();

  // The arcs are the chains of the remaining skeleton pixels; each joins the
  // vertices it touches.
  let on_arc: Vec<bool> = (0..w * h)
    .map(|i| skeleton[i] && !is_vertex_pixel[i])
    .collect();
  let arc_of = connected_components(&on_arc, w, h);
  let arc_count = arc_of.iter().flatten().max().map_or(0, |l| l + 1);
  let mut touching: Vec<std::collections::BTreeSet<usize>> =
    vec![std::collections::BTreeSet::default(); arc_count];
  let mut first_pixel = vec![usize::MAX; arc_count];
  for i in 0..w * h {
    let Some(arc) = arc_of[i] else { continue };
    first_pixel[arc] = first_pixel[arc].min(i);
    for dy in -1i64..=1 {
      for dx in -1i64..=1 {
        let ny = (i / w) as i64 + dy;
        let nx = (i % w) as i64 + dx;
        if ny < 0 || nx < 0 || ny >= h as i64 || nx >= w as i64 {
          continue;
        }
        if let Some(v) = vertex_of[ny as usize * w + nx as usize] {
          touching[arc].insert(v);
        }
      }
    }
  }
  let mut edges = std::collections::BTreeSet::new();
  for (arc, vertices) in touching.iter().enumerate() {
    if vertices.is_empty() {
      // A closed loop: its one vertex is the loop's first pixel.
      let i = first_pixel[arc];
      coordinates
        .push(((i % w) as f64 + 0.5, h as f64 - ((i / w) as f64 + 0.5)));
      vertex_count += 1;
      continue;
    }
    let vs: Vec<usize> = vertices.iter().copied().collect();
    for (a, &u) in vs.iter().enumerate() {
      for &v in &vs[a + 1..] {
        edges.insert((u, v));
      }
    }
  }

  let mut graph_args = vec![
    Expr::List(
      (1..=vertex_count as i128)
        .map(Expr::Integer)
        .collect::<Vec<_>>()
        .into(),
    ),
    Expr::List(
      edges
        .iter()
        .map(|&(u, v)| Expr::FunctionCall {
          name: "UndirectedEdge".to_string(),
          args: vec![
            Expr::Integer(u as i128 + 1),
            Expr::Integer(v as i128 + 1),
          ]
          .into(),
        })
        .collect::<Vec<_>>()
        .into(),
    ),
    Expr::Rule {
      pattern: Box::new(Expr::Identifier("VertexCoordinates".to_string())),
      replacement: Box::new(Expr::List(
        coordinates
          .iter()
          .map(|&(x, y)| Expr::List(vec![Expr::Real(x), Expr::Real(y)].into()))
          .collect::<Vec<_>>()
          .into(),
      )),
    },
  ];
  graph_args.extend(args[1..].iter().cloned());
  crate::evaluator::evaluate_function_call_ast("Graph", &graph_args)
}

/// The pixel levels FindThreshold bins: one value per pixel (the channel
/// mean of a multichannel one), with the image type.
fn threshold_levels(data: &[f64], channels: usize, ty: ImageType) -> Vec<f64> {
  data
    .chunks(channels.max(1))
    .map(|px| {
      let px = if channels >= 3 { &px[..3] } else { &px[..1] };
      if ty == ImageType::Real64 {
        px.iter().sum::<f64>() / px.len() as f64
      } else {
        let sum: f32 = px.iter().map(|&v| v as f32).sum();
        (sum / px.len() as f32) as f64
      }
    })
    .collect()
}

/// FindThreshold's default "Cluster" method: Otsu's between-class variance
/// maximization over a histogram. An integer image is binned on its raw
/// levels (256 for Byte, 65536 for Bit16); a real one in 256 bins spanning
/// its own range. The first bin that maximizes the variance closes the
/// lower class and is the threshold; when only two levels occur it is the
/// one midway between them, and a constant image gives 1.
pub fn find_threshold_value(
  data: &[f64],
  channels: usize,
  ty: ImageType,
) -> Expr {
  let levels = threshold_levels(data, channels, ty);
  otsu_threshold(&levels, ty, false).unwrap_or(Expr::Real(1.0))
}

/// Otsu's threshold of `levels` (see `find_threshold_value`), or None when
/// they are all equal. `unit_bins` bins real levels on [0, 1] instead of
/// their own range (MorphologicalBinarize's default).
fn otsu_threshold(
  levels: &[f64],
  ty: ImageType,
  unit_bins: bool,
) -> Option<Expr> {
  let (lo, hi) = levels
    .iter()
    .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
      (a.min(v), b.max(v))
    });
  if levels.is_empty() || lo >= hi {
    return None;
  }
  let max_level = match ty {
    ImageType::Bit => Some(1.0),
    ImageType::Byte => Some(255.0),
    ImageType::Bit16 => Some(65535.0),
    ImageType::Real32 | ImageType::Real64 => None,
  };
  let (lo, hi) = if unit_bins { (0.0, 1.0) } else { (lo, hi) };
  let (bins, width) = match max_level {
    Some(m) => (m as usize + 1, 1.0 / m),
    None => (256, (hi - lo) / 256.0),
  };
  let bin_of = |v: f64| match max_level {
    Some(m) => (v * m).round().clamp(0.0, m) as usize,
    None => (((v - lo) / width).max(0.0) as usize).min(bins - 1),
  };
  let value_of = |k: f64| match max_level {
    Some(_) => k * width,
    None => lo + k * width,
  };
  let mut counts = vec![0.0f64; bins];
  for &v in levels {
    counts[bin_of(v)] += 1.0;
  }
  let occupied: Vec<usize> = (0..bins).filter(|&k| counts[k] > 0.0).collect();
  if let [a, b] = occupied.as_slice() {
    if ty == ImageType::Bit {
      return Some(Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
      });
    }
    return Some(Expr::Real(value_of((*a + *b) as f64 / 2.0)));
  }
  let total = levels.len() as f64;
  let moment: f64 = (0..bins).map(|k| k as f64 * counts[k]).sum();
  let (mut below, mut moment_below) = (0.0, 0.0);
  let (mut best_variance, mut best_bin) = (-1.0f64, 0usize);
  for (k, &count) in counts.iter().enumerate() {
    below += count;
    moment_below += k as f64 * count;
    let above = total - below;
    if below == 0.0 || above == 0.0 {
      continue;
    }
    let difference = moment_below / below - (moment - moment_below) / above;
    let variance = below * above * difference * difference;
    if variance > best_variance {
      best_variance = variance;
      best_bin = k;
    }
  }
  Some(Expr::Real(value_of(best_bin as f64)))
}

/// FindThreshold[image] / FindThreshold[data] with the default method.
pub fn find_threshold_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  match &args[0] {
    Expr::Image {
      channels,
      data,
      image_type,
      ..
    } => Ok(find_threshold_value(data, *channels as usize, *image_type)),
    other => {
      let mut values = Vec::new();
      fn collect(e: &Expr, out: &mut Vec<f64>) -> bool {
        match e {
          Expr::List(items) => items.iter().all(|i| collect(i, out)),
          _ => crate::functions::math_ast::try_eval_to_f64(e)
            .map(|v| out.push(v))
            .is_some(),
        }
      }
      if matches!(other, Expr::List(_)) && collect(other, &mut values) {
        return Ok(find_threshold_value(&values, 1, ImageType::Real32));
      }
      Ok(unevaluated("FindThreshold", args))
    }
  }
}

/// MorphologicalBinarize[img, {t1, t2}] / [img, t] — hysteresis
/// binarization: pixels strictly above t2 seed regions that grow through
/// 8-connected pixels strictly above t1. A scalar t means {0.8 t, t},
/// a one-element list {t} means {t, t}. Multichannel pixels compare on
/// their channel MEAN (f32; unlike DistanceTransform's luminance).
/// Without thresholds, t is Otsu's threshold of the image binned on [0, 1]
/// (FindThreshold bins it on its own range).
pub fn morphological_binarize_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("MorphologicalBinarize", args);
  let Expr::Image {
    width,
    height,
    channels,
    ref data,
    image_type,
    ..
  } = args[0]
  else {
    crate::emit_message(&format!(
      "MorphologicalBinarize::imginv: Expecting an image or graphics instead of {}.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated());
  };
  let num = crate::functions::math_ast::try_eval_to_f64;
  let default_threshold;
  let spec = if let Some(spec) = args.get(1) {
    spec
  } else {
    let levels = threshold_levels(data, channels as usize, image_type);
    default_threshold =
      otsu_threshold(&levels, image_type, true).unwrap_or(Expr::Real(1.0));
    &default_threshold
  };
  let thresholds = match spec {
    Expr::List(ts) if ts.len() == 1 => num(&ts[0]).map(|t| (t, t)),
    Expr::List(ts) if ts.len() == 2 => num(&ts[0]).zip(num(&ts[1])),
    Expr::List(_) => None,
    e => num(e).map(|t| (0.8 * t, t)),
  };
  let Some((t1, t2)) = thresholds else {
    crate::emit_message(&format!(
      "MorphologicalBinarize::bdarg2: Invalid threshold specification {}.",
      expr_to_string(spec)
    ));
    return Ok(unevaluated());
  };

  let (w, h, ch) = (width as usize, height as usize, channels as usize);
  let n = w * h;
  let gray = |i: usize| -> f64 {
    let px = &data[i * ch..(i + 1) * ch];
    if ch >= 3 {
      let m = (px[0] as f32 + px[1] as f32 + px[2] as f32) / 3.0f32;
      m as f64
    } else {
      (px[0] as f32) as f64
    }
  };

  // Seed pixels flood through weak pixels with 8-connectivity.
  let mut out = vec![0.0f64; n];
  let mut stack: Vec<usize> = Vec::new();
  for i in 0..n {
    if gray(i) > t2 {
      out[i] = 1.0;
      stack.push(i);
    }
  }
  while let Some(i) = stack.pop() {
    let (x, y) = ((i % w) as isize, (i / w) as isize);
    for dy in -1..=1isize {
      for dx in -1..=1isize {
        let (nx, ny) = (x + dx, y + dy);
        if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
          continue;
        }
        let j = ny as usize * w + nx as usize;
        if out[j] == 0.0 && gray(j) > t1 {
          out[j] = 1.0;
          stack.push(j);
        }
      }
    }
  }

  Ok(Expr::Image {
    color_space: None,
    display_size: None,
    width,
    height,
    channels: 1,
    data: Arc::new(out),
    image_type: ImageType::Bit,
  })
}

/// ImageValue[img, {x, y}] / [img, {pos1, pos2, ...}] — bilinear sample
/// in the image coordinate system (x from the left edge, y up from the
/// bottom edge, pixel centers at half-integers) with zero padding
/// outside. Real32 images compute in f32, everything else in f64.
pub fn image_value_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("ImageValue", args);
  let Expr::Image {
    width,
    height,
    channels,
    ref data,
    image_type,
    ..
  } = args[0]
  else {
    crate::emit_message(&format!(
      "ImageValue::imginv: Expecting an image or graphics instead of {}.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated());
  };
  let (w, h, ch) = (width as usize, height as usize, channels as usize);

  let coord = |e: &Expr| -> Option<(f64, f64)> {
    let Expr::List(xy) = e else { return None };
    if xy.len() != 2 {
      return None;
    }
    let x = crate::functions::math_ast::try_eval_to_f64(&xy[0])?;
    let y = crate::functions::math_ast::try_eval_to_f64(&xy[1])?;
    (x.is_finite() && y.is_finite()).then_some((x, y))
  };

  // Distinguish a single point {x, y} from a list of points.
  enum Spec {
    One((f64, f64)),
    Many(Vec<(f64, f64)>),
  }
  let spec = (|| -> Option<Spec> {
    match &args[1] {
      Expr::List(items)
        if items.len() == 2 && !matches!(&items[0], Expr::List(_)) =>
      {
        Some(Spec::One(coord(&args[1])?))
      }
      Expr::List(items) => {
        let pts: Option<Vec<_>> = items.iter().map(coord).collect();
        Some(Spec::Many(pts?))
      }
      _ => None,
    }
  })();
  let Some(spec) = spec else {
    crate::emit_message(&format!(
      "ImageValue::imgrng: The specified argument {} should be an image, a graphics object or a list of coordinates.",
      expr_to_string(&args[1])
    ));
    return Ok(unevaluated());
  };

  // Pixel accessor by 0-based column and bottom-based row, 0 outside.
  let pixel = |col: i64, brow: i64, c: usize| -> f64 {
    if col < 0 || brow < 0 || col >= w as i64 || brow >= h as i64 {
      0.0
    } else {
      let row = h - 1 - brow as usize;
      data[(row * w + col as usize) * ch + c]
    }
  };
  // Tensor-product weighted sum in f64. Real32 images snap their pixel
  // values to f32 and round the final result to f32 (pinned against
  // wolframscript to the last bit; a nested-lerp form differs by 1 ULP).
  let real32 = image_type == ImageType::Real32;
  let sample = |x: f64, y: f64, c: usize| -> f64 {
    let gx = x - 0.5;
    let gy = y - 0.5;
    let (j0f, k0f) = (gx.floor(), gy.floor());
    let (fx, fy) = (gx - j0f, gy - k0f);
    let (j0, k0) = (j0f as i64, k0f as i64);
    let p = |dj: i64, dk: i64| {
      let v = pixel(j0 + dj, k0 + dk, c);
      if real32 { (v as f32) as f64 } else { v }
    };
    let v = (1.0 - fx) * (1.0 - fy) * p(0, 0)
      + fx * (1.0 - fy) * p(1, 0)
      + (1.0 - fx) * fy * p(0, 1)
      + fx * fy * p(1, 1);
    if real32 { (v as f32) as f64 } else { v }
  };
  let value_at = |x: f64, y: f64| -> Expr {
    if ch == 1 {
      Expr::Real(sample(x, y, 0))
    } else {
      Expr::List(
        (0..ch)
          .map(|c| Expr::Real(sample(x, y, c)))
          .collect::<Vec<_>>()
          .into(),
      )
    }
  };

  Ok(match spec {
    Spec::One((x, y)) => value_at(x, y),
    Spec::Many(pts) => Expr::List(
      pts
        .iter()
        .map(|&(x, y)| value_at(x, y))
        .collect::<Vec<_>>()
        .into(),
    ),
  })
}

/// Priority-flood morphological reconstruction by erosion over `mask`
/// (per channel plane): the result is the smallest R >= mask with
/// R <= marker that is stable under neighborhood erosion. Used by
/// FillingTransform; `corner_neighbors` selects 4- vs 8-connectivity.
fn reconstruct_by_erosion(
  mask: &[f64],
  marker: &[f64],
  w: usize,
  h: usize,
  corner_neighbors: bool,
) -> Vec<f64> {
  use std::cmp::Reverse;
  use std::collections::BinaryHeap;
  let mut r: Vec<f64> =
    mask.iter().zip(marker).map(|(&i, &m)| i.max(m)).collect();
  // Pixel values are non-negative, so the IEEE bit pattern orders them.
  let mut heap: BinaryHeap<(Reverse<u64>, usize)> =
    (0..r.len()).map(|i| (Reverse(r[i].to_bits()), i)).collect();
  let mut done = vec![false; r.len()];
  while let Some((Reverse(bits), i)) = heap.pop() {
    if done[i] || f64::from_bits(bits) > r[i] {
      continue;
    }
    done[i] = true;
    let (x, y) = (i % w, i / w);
    let mut relax =
      |nx: isize, ny: isize, heap: &mut BinaryHeap<(Reverse<u64>, usize)>| {
        if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
          return;
        }
        let j = ny as usize * w + nx as usize;
        let cand = mask[j].max(r[i]);
        if cand < r[j] {
          r[j] = cand;
          heap.push((Reverse(cand.to_bits()), j));
        }
      };
    let (xi, yi) = (x as isize, y as isize);
    relax(xi - 1, yi, &mut heap);
    relax(xi + 1, yi, &mut heap);
    relax(xi, yi - 1, &mut heap);
    relax(xi, yi + 1, &mut heap);
    if corner_neighbors {
      relax(xi - 1, yi - 1, &mut heap);
      relax(xi + 1, yi - 1, &mut heap);
      relax(xi - 1, yi + 1, &mut heap);
      relax(xi + 1, yi + 1, &mut heap);
    }
  }
  r
}

/// Marker plane for hole filling: the image itself on the border,
/// `interior` (a large value or mask + h) inside.
fn filling_marker(mask: &[f64], w: usize, h: usize, add: f64) -> Vec<f64> {
  (0..mask.len())
    .map(|i| {
      let (x, y) = (i % w, i / w);
      if x == 0 || y == 0 || x == w - 1 || y == h - 1 {
        mask[i]
      } else if add.is_infinite() {
        f64::INFINITY
      } else {
        mask[i] + add
      }
    })
    .collect()
}

/// FillingTransform[img] / [img, h] / [img, marker] — fill extended
/// minima (holes) per channel. Decoded from wolframscript probes:
/// - plain form: flood from the border with 4-connectivity;
/// - depth form: reconstruction with interior marker mask+h and
///   8-connectivity; Bit becomes Real32, Byte/Bit16 stay quantized;
/// - marker form: per-pixel I + (F - I) * m where F is the plain fill
///   and m is the largest (clamped) marker value in each 4-connected
///   basin; the result is Real64 for Real64 inputs and Real32 otherwise.
///   DeleteSmallComponents[m] / [m, n] replaces the positive integers of a label
///   matrix with 0 wherever their component's tally is `n` or fewer (default
///   n = 0, i.e. a no-op). Each distinct nonzero value is one component.
pub fn delete_small_components_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("DeleteSmallComponents", args));
  if args.is_empty() || args.len() > 2 {
    return unevaluated();
  }
  if matches!(&args[0], Expr::Image { .. }) {
    return delete_small_image_components(args);
  }
  let Expr::List(rows) = &args[0] else {
    return unevaluated();
  };
  // Parse the integer label matrix and tally each nonzero label.
  let mut mat: Vec<Vec<i128>> = Vec::with_capacity(rows.len());
  let mut counts: std::collections::HashMap<i128, usize> =
    std::collections::HashMap::new();
  for r in rows {
    let Expr::List(cells) = r else {
      return unevaluated();
    };
    let mut row = Vec::with_capacity(cells.len());
    for c in cells {
      let Some(v) = crate::functions::math_ast::try_eval_to_f64(c) else {
        return unevaluated();
      };
      if v.fract() != 0.0 {
        return unevaluated();
      }
      let iv = v as i128;
      if iv != 0 {
        *counts.entry(iv).or_insert(0) += 1;
      }
      row.push(iv);
    }
    mat.push(row);
  }
  // Threshold: components with `n` or fewer pixels are deleted (default 0).
  let n = match args.get(1) {
    None => 0usize,
    Some(e) => match crate::functions::math_ast::try_eval_to_f64(e) {
      Some(v) if v >= 0.0 && v.fract() == 0.0 => v as usize,
      _ => return unevaluated(),
    },
  };
  let out: Vec<Expr> = mat
    .into_iter()
    .map(|row| {
      Expr::List(
        row
          .into_iter()
          .map(|v| {
            let keep = v != 0 && counts.get(&v).copied().unwrap_or(0) > n;
            Expr::Integer(if keep { v } else { 0 })
          })
          .collect::<Vec<_>>()
          .into(),
      )
    })
    .collect();
  Ok(Expr::List(out.into()))
}

/// ComponentMeasurements[labelmatrix, property] measures each labelled
/// component of a label matrix (each distinct nonzero value is a component).
/// Returns a list of `label -> value` rules sorted by label. Supported
/// properties: "Count" (pixel count), "Area" (count as a real), and "Label"
/// (the label itself), and "LabelCount" (the number of components); a list of properties yields a tuple per component.
pub fn component_measurements_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("ComponentMeasurements", args));
  if args.len() != 2 {
    return unevaluated();
  }
  // An image is measured through its morphological components.
  if matches!(&args[0], Expr::Image { .. }) {
    let labels = crate::evaluator::evaluate_expr_to_expr(&call(
      "MorphologicalComponents",
      vec![args[0].clone()],
    ))?;
    if !matches!(labels, Expr::List(_)) {
      return unevaluated();
    }
    return component_measurements_ast(&[labels, args[1].clone()]);
  }
  let Expr::List(rows) = &args[0] else {
    return unevaluated();
  };
  // Parse the label array (a matrix, or a rank-3 array of voxels) and collect
  // cell counts per nonzero label, keyed by the label's integer value
  // (component labels are integers).
  let mut counts: std::collections::BTreeMap<i128, usize> =
    std::collections::BTreeMap::new();
  fn leaves<'a>(e: &'a Expr, depth: usize, out: &mut Vec<&'a Expr>) -> bool {
    match e {
      Expr::List(items) if depth > 0 => {
        items.iter().all(|i| leaves(i, depth - 1, out))
      }
      Expr::List(_) => false,
      other if depth == 0 => {
        out.push(other);
        true
      }
      _ => false,
    }
  }
  let rank = if matches!(rows.first(), Some(Expr::List(r)) if matches!(r.first(), Some(Expr::List(_))))
  {
    3
  } else {
    2
  };
  let mut cells = Vec::new();
  if !leaves(&args[0], rank, &mut cells) {
    return unevaluated();
  }
  for c in cells {
    let Some(v) = crate::functions::math_ast::try_eval_to_f64(c) else {
      return unevaluated();
    };
    if v == 0.0 {
      continue;
    }
    if v.fract() != 0.0 {
      return unevaluated(); // non-integer labels are not supported
    }
    *counts.entry(v as i128).or_insert(0) += 1;
  }

  // Which properties to report (a single property or a list of them).
  let props: Vec<String> = match &args[1] {
    Expr::String(s) => vec![s.clone()],
    Expr::List(items) => {
      let mut v = Vec::with_capacity(items.len());
      for it in items {
        match it {
          Expr::String(s) => v.push(s.clone()),
          _ => return unevaluated(),
        }
      }
      v
    }
    _ => return unevaluated(),
  };
  let single = matches!(&args[1], Expr::String(_));

  let label_count = counts.len();
  let measure = |prop: &str, label: i128, count: usize| -> Option<Expr> {
    match prop {
      "LabelCount" => Some(Expr::Integer(label_count as i128)),
      "Count" => Some(Expr::Integer(count as i128)),
      "Area" => Some(Expr::Real(count as f64)),
      // The radius of the disk with the same area.
      "EquivalentDiskRadius" => {
        Some(Expr::Real((count as f64 / std::f64::consts::PI).sqrt()))
      }
      "Label" => Some(Expr::Integer(label)),
      _ => None,
    }
  };

  let mut rules = Vec::with_capacity(counts.len());
  for (&label, &count) in &counts {
    let value = if single {
      match measure(&props[0], label, count) {
        Some(v) => v,
        None => return unevaluated(),
      }
    } else {
      let mut tuple = Vec::with_capacity(props.len());
      for p in &props {
        match measure(p, label, count) {
          Some(v) => tuple.push(v),
          None => return unevaluated(),
        }
      }
      Expr::List(tuple.into())
    };
    rules.push(rule_expr(Expr::Integer(label), value));
  }
  Ok(Expr::List(rules.into()))
}

/// Gray levels of an image as rows of pixels (top row first); colour
/// images use the channel mean. `None` for anything that is not an image.
fn image_gray_matrix(e: &Expr) -> Option<Vec<Vec<f64>>> {
  let Expr::Image {
    width,
    height,
    channels,
    data,
    ..
  } = e
  else {
    return None;
  };
  let (w, h, ch) = (*width as usize, *height as usize, *channels as usize);
  let gray = |i: usize| -> f64 {
    let px = &data[i * ch..(i + 1) * ch];
    if ch >= 3 {
      ((px[0] as f32 + px[1] as f32 + px[2] as f32) / 3.0f32) as f64
    } else {
      (px[0] as f32) as f64
    }
  };
  Some(
    (0..h)
      .map(|y| (0..w).map(|x| gray(y * w + x)).collect())
      .collect(),
  )
}

/// MorphologicalPerimeter[image] / [image, t] gives a binary image of the
/// foreground pixels (values above `t`, default 0) that touch a background
/// pixel horizontally or vertically.
pub fn morphological_perimeter_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("MorphologicalPerimeter", args);
  let (Expr::Image { width, height, .. }, Some(mat)) =
    (&args[0], image_gray_matrix(&args[0]))
  else {
    crate::emit_message(&format!(
      "MorphologicalPerimeter::imginv: Expecting an image or graphics instead of {}.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated());
  };
  let threshold = match args.get(1) {
    None => 0.0,
    Some(t) => match crate::functions::math_ast::try_eval_to_f64(t) {
      Some(v) => v,
      None => return Ok(unevaluated()),
    },
  };
  let (w, h) = (*width as usize, *height as usize);
  let fg = |y: i64, x: i64| -> Option<bool> {
    if y < 0 || x < 0 || y >= h as i64 || x >= w as i64 {
      None
    } else {
      Some(mat[y as usize][x as usize] > threshold)
    }
  };
  let mut out = vec![0.0f64; w * h];
  for y in 0..h as i64 {
    for x in 0..w as i64 {
      if fg(y, x) != Some(true) {
        continue;
      }
      let touches_background = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        .iter()
        .any(|&(dy, dx)| fg(y + dy, x + dx) == Some(false));
      if touches_background {
        out[y as usize * w + x as usize] = 1.0;
      }
    }
  }
  Ok(Expr::Image {
    color_space: None,
    display_size: None,
    width: *width,
    height: *height,
    channels: 1,
    data: Arc::new(out),
    image_type: ImageType::Bit,
  })
}

/// MorphologicalComponents[matrix] / [matrix, t] labels the connected
/// foreground components (values above the threshold `t`, default 0) with
/// consecutive integers assigned in raster order of first appearance.
/// Connectivity is 8-way by default; `CornerNeighbors -> False` selects 4-way.
/// `MorphologicalComponents` of a rank-3 array: voxels above the threshold
/// labelled by connected component in scan order, 26-connected (6 with
/// `CornerNeighbors -> False`). `None` for anything but a numeric rank-3
/// array.
fn morphological_components_3d(
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  let Expr::List(planes) = args.first()? else {
    return None;
  };
  let Expr::List(first_plane) = planes.first()? else {
    return None;
  };
  let Expr::List(first_row) = first_plane.first()? else {
    return None;
  };
  let dims = [planes.len(), first_plane.len(), first_row.len()];
  let mut values = Vec::with_capacity(dims.iter().product());
  for plane in planes {
    let Expr::List(rows) = plane else { return None };
    if rows.len() != dims[1] {
      return None;
    }
    for row in rows {
      let Expr::List(cells) = row else { return None };
      if cells.len() != dims[2] {
        return None;
      }
      for c in cells {
        values.push(crate::functions::math_ast::try_eval_to_f64(c)?);
      }
    }
  }
  let mut threshold = 0.0f64;
  let mut corner = true;
  for a in &args[1..] {
    match a {
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } if matches!(pattern.as_ref(), Expr::Identifier(s) if s == "CornerNeighbors") =>
      {
        corner =
          matches!(replacement.as_ref(), Expr::Identifier(s) if s == "True");
      }
      _ => threshold = crate::functions::math_ast::try_eval_to_f64(a)?,
    }
  }
  let offsets: Vec<[i64; 3]> = (-1..=1)
    .flat_map(|a| (-1..=1).flat_map(move |b| (-1..=1).map(move |c| [a, b, c])))
    .filter(|o| *o != [0, 0, 0])
    .filter(|o| corner || o.iter().filter(|v| **v != 0).count() == 1)
    .collect();
  let index = |p: [usize; 3]| (p[0] * dims[1] + p[1]) * dims[2] + p[2];
  let mut labels = vec![0i64; values.len()];
  let mut next = 1i64;
  for i in 0..dims[0] {
    for j in 0..dims[1] {
      for l in 0..dims[2] {
        let start = [i, j, l];
        if values[index(start)] <= threshold || labels[index(start)] != 0 {
          continue;
        }
        labels[index(start)] = next;
        let mut queue = std::collections::VecDeque::from([start]);
        while let Some(p) = queue.pop_front() {
          for o in &offsets {
            let q =
              [p[0] as i64 + o[0], p[1] as i64 + o[1], p[2] as i64 + o[2]];
            if q.iter().zip(&dims).any(|(v, d)| *v < 0 || *v >= *d as i64) {
              continue;
            }
            let q = [q[0] as usize, q[1] as usize, q[2] as usize];
            if values[index(q)] > threshold && labels[index(q)] == 0 {
              labels[index(q)] = next;
              queue.push_back(q);
            }
          }
        }
        next += 1;
      }
    }
  }
  Some(Ok(Expr::List(
    (0..dims[0])
      .map(|i| {
        Expr::List(
          (0..dims[1])
            .map(|j| {
              Expr::List(
                (0..dims[2])
                  .map(|l| Expr::Integer(labels[index([i, j, l])] as i128))
                  .collect::<Vec<_>>()
                  .into(),
              )
            })
            .collect::<Vec<_>>()
            .into(),
        )
      })
      .collect::<Vec<_>>()
      .into(),
  )))
}

pub fn morphological_components_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("MorphologicalComponents", args));
  if args.is_empty() {
    return unevaluated();
  }
  if let Some(result) = morphological_components_3d(args) {
    return result;
  }
  // Parse a rectangular numeric matrix, or take the gray levels of an image.
  let mut mat: Vec<Vec<f64>> = Vec::new();
  let mut ncols: Option<usize> = None;
  if let Some(gray) = image_gray_matrix(&args[0]) {
    ncols = gray.first().map(std::vec::Vec::len);
    mat = gray;
  } else {
    let Expr::List(rows) = &args[0] else {
      return unevaluated();
    };
    mat.reserve(rows.len());
    for r in rows {
      let Expr::List(cells) = r else {
        return unevaluated();
      };
      let mut row = Vec::with_capacity(cells.len());
      for c in cells {
        match crate::functions::math_ast::try_eval_to_f64(c) {
          Some(v) => row.push(v),
          None => return unevaluated(),
        }
      }
      match ncols {
        Some(nc) if nc != row.len() => return unevaluated(),
        _ => ncols = Some(row.len()),
      }
      mat.push(row);
    }
  }
  let h = mat.len();
  let w = ncols.unwrap_or(0);
  if h == 0 || w == 0 {
    return Ok(args[0].clone());
  }

  // Optional threshold (a bare number) and CornerNeighbors option.
  let mut threshold = 0.0f64;
  let mut corner = true;
  for a in &args[1..] {
    match a {
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } if matches!(pattern.as_ref(),
        Expr::Identifier(s) if s == "CornerNeighbors") =>
      {
        corner = matches!(replacement.as_ref(),
          Expr::Identifier(s) if s == "True");
      }
      _ => match crate::functions::math_ast::try_eval_to_f64(a) {
        Some(v) => threshold = v,
        None => return unevaluated(),
      },
    }
  }

  // 8- or 4-connected flood fill in raster order.
  let offsets: &[(i64, i64)] = if corner {
    &[
      (-1, -1),
      (-1, 0),
      (-1, 1),
      (0, -1),
      (0, 1),
      (1, -1),
      (1, 0),
      (1, 1),
    ]
  } else {
    &[(-1, 0), (0, -1), (0, 1), (1, 0)]
  };
  let fg = |y: usize, x: usize| mat[y][x] > threshold;
  let mut labels = vec![vec![0i64; w]; h];
  let mut next = 1i64;
  for y0 in 0..h {
    for x0 in 0..w {
      if !fg(y0, x0) || labels[y0][x0] != 0 {
        continue;
      }
      let lbl = next;
      next += 1;
      let mut queue = std::collections::VecDeque::new();
      labels[y0][x0] = lbl;
      queue.push_back((y0, x0));
      while let Some((cy, cx)) = queue.pop_front() {
        for &(dy, dx) in offsets {
          let ny = cy as i64 + dy;
          let nx = cx as i64 + dx;
          if ny < 0 || nx < 0 || ny >= h as i64 || nx >= w as i64 {
            continue;
          }
          let (ny, nx) = (ny as usize, nx as usize);
          if fg(ny, nx) && labels[ny][nx] == 0 {
            labels[ny][nx] = lbl;
            queue.push_back((ny, nx));
          }
        }
      }
    }
  }

  let out: Vec<Expr> = labels
    .into_iter()
    .map(|row| {
      Expr::List(
        row
          .into_iter()
          .map(|v| Expr::Integer(v as i128))
          .collect::<Vec<_>>()
          .into(),
      )
    })
    .collect();
  Ok(Expr::List(out.into()))
}

pub fn filling_transform_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("FillingTransform", args);
  let Expr::Image {
    width,
    height,
    channels,
    ref data,
    image_type,
    ..
  } = args[0]
  else {
    crate::emit_message(&format!(
      "FillingTransform::imginv: Expecting an image or graphics instead of {}.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated());
  };
  let (w, h, ch) = (width as usize, height as usize, channels as usize);
  let n = w * h;

  // Snap stored values to f32 unless the image genuinely holds f64,
  // mirroring wolframscript's Real32 storage of non-Real64 images.
  let snap = |v: f64, t: ImageType| -> f64 {
    if t == ImageType::Real64 {
      v
    } else {
      (v as f32) as f64
    }
  };

  enum Second {
    None,
    Depth(f64),
    Marker(Vec<f64>),
  }
  let second = if args.len() == 2 {
    match &args[1] {
      Expr::Image {
        width: mw,
        height: mh,
        channels: mc,
        data: mdata,
        image_type: mt,
        ..
      } => {
        if *mc != 1 {
          // Multichannel markers echo unevaluated without a message.
          return Ok(unevaluated());
        }
        // The largest clamped marker value per 4-connected basin scales
        // the fill; positions outside a smaller marker read as zero.
        let (mw, mh) = (*mw as usize, *mh as usize);
        let mut m = vec![0.0f64; n];
        for y in 0..h.min(mh) {
          for x in 0..w.min(mw) {
            m[y * w + x] = snap(mdata[y * mw + x], *mt).clamp(0.0, 1.0);
          }
        }
        Second::Marker(m)
      }
      e => match crate::functions::math_ast::try_eval_to_f64(e) {
        Some(v) if v >= 0.0 => Second::Depth(v),
        Some(_) => {
          crate::emit_message(&format!(
            "FillingTransform::invh: The height specification {} must be positive.",
            expr_to_string(e)
          ));
          return Ok(unevaluated());
        }
        None => {
          crate::emit_message(&format!(
            "FillingTransform::arg2: Expecting either a marker or depth specification as the second argument instead of {}.",
            expr_to_string(e)
          ));
          return Ok(unevaluated());
        }
      },
    }
  } else {
    Second::None
  };

  let corner = matches!(second, Second::Depth(_));
  let mut out = vec![0.0f64; n * ch];
  for c in 0..ch {
    let mask: Vec<f64> =
      (0..n).map(|i| snap(data[i * ch + c], image_type)).collect();
    let plane: Vec<f64> = match &second {
      Second::None => {
        let marker = filling_marker(&mask, w, h, f64::INFINITY);
        reconstruct_by_erosion(&mask, &marker, w, h, corner)
      }
      Second::Depth(d) => {
        let marker = filling_marker(&mask, w, h, *d);
        reconstruct_by_erosion(&mask, &marker, w, h, corner)
      }
      Second::Marker(m) => {
        let marker = filling_marker(&mask, w, h, f64::INFINITY);
        let f = reconstruct_by_erosion(&mask, &marker, w, h, corner);
        // Label 4-connected basins (F > mask) and take each basin's
        // largest marker value.
        let mut label = vec![usize::MAX; n];
        let mut basin_m: Vec<f64> = Vec::new();
        for start in 0..n {
          if label[start] != usize::MAX || f[start] <= mask[start] {
            continue;
          }
          let id = basin_m.len();
          basin_m.push(0.0);
          let mut stack = vec![start];
          label[start] = id;
          while let Some(i) = stack.pop() {
            basin_m[id] = basin_m[id].max(m[i]);
            let (x, y) = (i % w, i / w);
            for (nx, ny) in [
              (x.wrapping_sub(1), y),
              (x + 1, y),
              (x, y.wrapping_sub(1)),
              (x, y + 1),
            ] {
              if nx >= w || ny >= h {
                continue;
              }
              let j = ny * w + nx;
              if label[j] == usize::MAX && f[j] > mask[j] {
                label[j] = id;
                stack.push(j);
              }
            }
          }
        }
        (0..n)
          .map(|i| {
            if label[i] == usize::MAX {
              mask[i]
            } else {
              // Linear interpolation between the image and its plain
              // fill: F*m + I*(1-m), in f32 arithmetic unless Real64
              // (each operation rounds — this matches wolframscript
              // bit-exact where I + (F-I)*m does not).
              let mv = basin_m[label[i]];
              if image_type == ImageType::Real64 {
                let m = (mv as f32) as f64;
                f[i] * m + mask[i] * (1.0 - m)
              } else {
                let m = mv as f32;
                let r = f[i] as f32 * m + mask[i] as f32 * (1.0 - m);
                r as f64
              }
            }
          })
          .collect()
      }
    };
    for i in 0..n {
      out[i * ch + c] = plane[i];
    }
  }

  let out_type = match &second {
    Second::None => image_type,
    Second::Depth(_) => match image_type {
      ImageType::Bit => ImageType::Real32,
      t => t,
    },
    Second::Marker(_) => match image_type {
      ImageType::Real64 => ImageType::Real64,
      _ => ImageType::Real32,
    },
  };
  // Quantized types keep their quantization for the depth form.
  if matches!(second, Second::Depth(_)) {
    let scale = match out_type {
      ImageType::Byte => Some(255.0),
      ImageType::Bit16 => Some(65535.0),
      _ => None,
    };
    if let Some(s) = scale {
      for v in &mut out {
        *v = (*v * s).round() / s;
      }
    }
  }

  Ok(Expr::Image {
    color_space: None,
    display_size: None,
    width,
    height,
    channels,
    data: Arc::new(out),
    image_type: out_type,
  })
}

/// One-dimensional squared-distance transform (Felzenszwalb-Huttenlocher
/// lower-envelope-of-parabolas pass). `f` holds source costs, `d` results.
fn edt_pass_1d(f: &[f64], d: &mut [f64]) {
  let n = f.len();
  let mut v = vec![0usize; n];
  let mut z = vec![0.0f64; n + 1];
  let mut k = 0usize;
  z[0] = f64::NEG_INFINITY;
  z[1] = f64::INFINITY;
  for q in 1..n {
    loop {
      let vk = v[k];
      let s = ((f[q] + (q * q) as f64) - (f[vk] + (vk * vk) as f64))
        / (2.0 * (q - vk) as f64);
      if s <= z[k] {
        k -= 1;
      } else {
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f64::INFINITY;
        break;
      }
    }
  }
  k = 0;
  for (q, dq) in d.iter_mut().enumerate() {
    while z[k + 1] < q as f64 {
      k += 1;
    }
    let diff = q as f64 - v[k] as f64;
    *dq = diff * diff + f[v[k]];
  }
}

/// Marker cost for foreground pixels: far larger than any possible
/// squared pixel distance, small enough to stay exact through the passes.
const EDT_FAR: f64 = 1e18;

/// The metric a `DistanceTransform` measures distances in.
#[derive(Clone, Copy, PartialEq, Eq)]
enum EdtMetric {
  Euclidean,
  Manhattan,
  Chessboard,
}

/// Exact city-block (4-neighbour) or chessboard (8-neighbour) distance to
/// the nearest zero-cost pixel, by a forward and a backward sweep.
fn chamfer_distance(
  cost: &[f64],
  w: usize,
  h: usize,
  diagonal: bool,
) -> Vec<f64> {
  let mut d: Vec<f64> = cost
    .iter()
    .map(|&c| if c > 0.0 { f64::INFINITY } else { 0.0 })
    .collect();
  let relax = |d: &mut [f64], x: usize, y: usize, dx: isize, dy: isize| {
    let (nx, ny) = (x as isize + dx, y as isize + dy);
    if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
      let cand = d[ny as usize * w + nx as usize] + 1.0;
      if cand < d[y * w + x] {
        d[y * w + x] = cand;
      }
    }
  };
  for y in 0..h {
    for x in 0..w {
      relax(&mut d, x, y, -1, 0);
      relax(&mut d, x, y, 0, -1);
      if diagonal {
        relax(&mut d, x, y, -1, -1);
        relax(&mut d, x, y, 1, -1);
      }
    }
  }
  for y in (0..h).rev() {
    for x in (0..w).rev() {
      relax(&mut d, x, y, 1, 0);
      relax(&mut d, x, y, 0, 1);
      if diagonal {
        relax(&mut d, x, y, 1, 1);
        relax(&mut d, x, y, -1, 1);
      }
    }
  }
  d
}

/// DistanceTransform[img] / [img, t] — each foreground pixel becomes its
/// Euclidean distance to the nearest background pixel (the image border
/// does not count as background). Foreground is luminance strictly above
/// t (default 0), evaluated on f32-snapped pixel values. When the image
/// has no background pixel at all, wolframscript returns all-1 values.
/// Exact non-machine thresholds (e.g. 1/2) trigger image-dependent
/// garbage in wolframscript and are deliberately given sane numeric
/// semantics here instead.
pub fn distance_transform_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("DistanceTransform", args);
  // `DistanceFunction -> f` picks the metric; the remaining arguments are
  // the image and the optional threshold.
  let mut metric = EdtMetric::Euclidean;
  let mut positional: Vec<Expr> = Vec::new();
  for arg in args {
    if let Expr::Rule {
      pattern,
      replacement,
    } = arg
      && matches!(pattern.as_ref(), Expr::Identifier(n) if n == "DistanceFunction")
    {
      let value = crate::evaluator::evaluate_expr_to_expr(replacement)
        .unwrap_or_else(|_| replacement.as_ref().clone());
      metric = match &value {
        Expr::Identifier(n) if n == "EuclideanDistance" => EdtMetric::Euclidean,
        Expr::Identifier(n) if n == "ManhattanDistance" => EdtMetric::Manhattan,
        Expr::Identifier(n) if n == "ChessboardDistance" => {
          EdtMetric::Chessboard
        }
        _ => return Ok(unevaluated()),
      };
    } else {
      positional.push(arg.clone());
    }
  }
  if positional.is_empty() || positional.len() > 2 {
    return Ok(unevaluated());
  }
  let args: &[Expr] = &positional;
  // Graphics are rasterized, like wolframscript does for any image argument.
  let coerced;
  let args: &[Expr] = if !matches!(&args[0], Expr::Image { .. })
    && let Some(img) = coerce_graphics_to_image(&args[0])
  {
    let mut v = args.to_vec();
    v[0] = img;
    coerced = v;
    &coerced
  } else {
    args
  };
  let Expr::Image {
    width,
    height,
    channels,
    ref data,
    ..
  } = args[0]
  else {
    crate::emit_message(&format!(
      "DistanceTransform::imginv: Expecting an image or graphics instead of {}.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated());
  };
  let t = if args.len() == 2 {
    if let Some(v) = crate::functions::math_ast::try_eval_to_f64(&args[1]) {
      v
    } else {
      crate::emit_message(&format!(
        "DistanceTransform::rthres: The specified threshold value {} should represent a real number.",
        expr_to_string(&args[1])
      ));
      return Ok(unevaluated());
    }
  } else {
    0.0
  };

  let (w, h, ch) = (width as usize, height as usize, channels as usize);
  let n = w * h;
  // Foreground test on the f32-snapped luminance (channels 3/4 use the
  // 0.299/0.587/0.114 weights in f32 arithmetic; 2 channels are
  // gray+alpha). The threshold itself stays in f64.
  let luminance = |i: usize| -> f64 {
    let px = &data[i * ch..(i + 1) * ch];
    let l32 = if ch >= 3 {
      0.299f32 * px[0] as f32
        + 0.587f32 * px[1] as f32
        + 0.114f32 * px[2] as f32
    } else {
      px[0] as f32
    };
    l32 as f64
  };

  let mut cost: Vec<f64> = (0..n)
    .map(|i| if luminance(i) > t { EDT_FAR } else { 0.0 })
    .collect();
  if cost.iter().all(|&c| c > 0.0) {
    // No background pixel anywhere: wolframscript yields all-1 values.
    return Ok(Expr::Image {
      color_space: None,
      display_size: None,
      width,
      height,
      channels: 1,
      data: Arc::new(vec![1.0; n]),
      image_type: ImageType::Real32,
    });
  }

  if metric != EdtMetric::Euclidean {
    let dist = chamfer_distance(&cost, w, h, metric == EdtMetric::Chessboard);
    return Ok(Expr::Image {
      color_space: None,
      display_size: None,
      width,
      height,
      channels: 1,
      data: Arc::new(dist),
      image_type: ImageType::Real32,
    });
  }

  // Column pass then row pass over squared distances.
  let mut buf = vec![0.0f64; h.max(w)];
  let mut out = vec![0.0f64; n];
  for x in 0..w {
    let col: Vec<f64> = (0..h).map(|y| cost[y * w + x]).collect();
    edt_pass_1d(&col, &mut buf[..h]);
    for y in 0..h {
      out[y * w + x] = buf[y];
    }
  }
  for y in 0..h {
    let row: Vec<f64> = (0..w).map(|x| out[y * w + x]).collect();
    edt_pass_1d(&row, &mut buf[..w]);
    for x in 0..w {
      cost[y * w + x] = buf[x].sqrt();
    }
  }

  Ok(Expr::Image {
    color_space: None,
    display_size: None,
    width,
    height,
    channels: 1,
    data: Arc::new(cost),
    image_type: ImageType::Real32,
  })
}

/// Valid ColorCombine color-space names with their required channel count
/// (decoded from wolframscript: imgcstype for anything else).
const COLOR_COMBINE_SPACES: &[(&str, usize)] = &[
  ("Grayscale", 1),
  ("RGB", 3),
  ("HSB", 3),
  ("XYZ", 3),
  ("LAB", 3),
  ("LUV", 3),
  ("CMYK", 4),
];

/// ColorCombine[{img1, img2, ...}] / ColorCombine[imgs, colorspace] —
/// interleave the channels of the inputs into one multichannel image.
/// The colorspace argument only tags the result (no data conversion);
/// its validity is checked before the image list, and its channel count
/// must match the combined total. The result type is the highest input
/// type in the order Bit < Byte < Bit16 < Real32 < Real64.
pub fn color_combine_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("ColorCombine", args);

  let color_space: Option<(&'static str, usize)> = if args.len() == 2 {
    let found = match &args[1] {
      Expr::String(s) => COLOR_COMBINE_SPACES
        .iter()
        .find(|(name, _)| name == s)
        .copied(),
      _ => None,
    };
    if let Some(cs) = found {
      Some(cs)
    } else {
      let shown = match &args[1] {
        Expr::String(s) => s.clone(),
        e => expr_to_string(e),
      };
      crate::emit_message(&format!(
        "ColorCombine::imgcstype: {shown} is an invalid color space specification."
      ));
      return Ok(unevaluated());
    }
  } else {
    None
  };

  let invalid_list = || {
    crate::emit_message(&format!(
      "ColorCombine::ccbinput: {} should be a list of images with the same image dimensions.",
      expr_to_string(&args[0])
    ));
  };
  let Expr::List(items) = &args[0] else {
    invalid_list();
    return Ok(unevaluated());
  };
  let mut inputs: Vec<(u32, u32, usize, &std::sync::Arc<Vec<f64>>, ImageType)> =
    Vec::with_capacity(items.len());
  for item in items {
    let Expr::Image {
      width,
      height,
      channels,
      data,
      image_type,
      ..
    } = item
    else {
      invalid_list();
      return Ok(unevaluated());
    };
    inputs.push((*width, *height, *channels as usize, data, *image_type));
  }
  let Some(&(w, h, ..)) = inputs.first() else {
    invalid_list();
    return Ok(unevaluated());
  };
  if inputs.iter().any(|&(iw, ih, ..)| iw != w || ih != h) {
    invalid_list();
    return Ok(unevaluated());
  }

  let total: usize = inputs.iter().map(|&(.., ch, _, _)| ch).sum();
  // One channel more than the space needs is an alpha channel.
  if let Some((name, want)) = color_space
    && total != want
    && total != want + 1
  {
    crate::emit_message(&format!(
      "ColorCombine::imgcsmis: The specified color space {name} and the number of channels {total} are not compatible."
    ));
    return Ok(unevaluated());
  }

  let type_rank = |t: ImageType| match t {
    ImageType::Bit => 0,
    ImageType::Byte => 1,
    ImageType::Bit16 => 2,
    ImageType::Real32 => 3,
    ImageType::Real64 => 4,
  };
  let image_type = inputs
    .iter()
    .map(|&(.., t)| t)
    .max_by_key(|&t| type_rank(t))
    .unwrap();

  // Pixel data is stored normalized to [0, 1] regardless of type, so the
  // channels interleave without rescaling. Real32 inputs are snapped to
  // f32 so their values keep single precision when the result is Real64
  // (wolframscript stores Real32 images as f32).
  let n = (w as usize) * (h as usize);
  let mut data = Vec::with_capacity(n * total);
  for i in 0..n {
    for &(.., ch, src, ty) in &inputs {
      for v in &src[i * ch..(i + 1) * ch] {
        data.push(if ty == ImageType::Real32 {
          (*v as f32) as f64
        } else {
          *v
        });
      }
    }
  }

  Ok(Expr::Image {
    color_space: color_space.map(|(name, _)| name),
    display_size: None,
    width: w,
    height: h,
    channels: total as u8,
    data: Arc::new(data),
    image_type,
  })
}

/// ColorSeparate[img] — return one single-channel image per channel
/// of the input. Grayscale images pass through unchanged (as a list
/// of one). The output images preserve the input's width, height, and
/// image type.
pub fn color_separate_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let Expr::Image {
    color_space: _,
    display_size: _,
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
  else {
    crate::emit_message(&format!(
      "ColorSeparate::imginv: Expecting an image or graphics instead of {}.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated("ColorSeparate", args));
  };
  let ch = *channels as usize;
  let n = (*width as usize) * (*height as usize);
  let mut images = Vec::with_capacity(ch);
  for c_idx in 0..ch {
    let channel_data: Vec<f64> = (0..n).map(|i| data[i * ch + c_idx]).collect();
    images.push(Expr::Image {
      color_space: None,
      display_size: None,
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
      expr_to_string(&args[0])
    ));
  }
  Ok(unevaluated("ColorQuantize", args))
}

/// Threshold[data]            replaces values with |x| ≤ 10^-10 by zero.
/// Threshold[data, t]         uses t as the threshold.
/// Lists of arbitrary nesting depth are walked recursively. Real values
/// emit a Real 0.; everything else emits an Integer 0, matching
/// wolframscript. Non-numeric leaves trigger Threshold::nlist and the
/// call is returned unevaluated.
pub fn threshold_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("Threshold", args);
  let is_array_like = matches!(
    &args[0],
    Expr::Image { .. } | Expr::List(_) | Expr::FunctionCall { .. }
  );
  if !is_array_like {
    crate::emit_message(&format!(
      "Threshold::wlist: Argument {} should be one of rectangular array of any depth, image, sound or sampled sound list.",
      expr_to_string(&args[0])
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
  // Default threshold is 10^-10 with the "Hard" method. A second argument may
  // be a bare numeric threshold (Hard) or a method spec
  // {"Hard"|"Soft"|"PiecewiseGarrote", delta}. The "Firm" method and any
  // unknown spec are left unevaluated. The threshold value must be numeric.
  let (method, threshold): (ThreshMethod, Expr) = if args.len() == 2 {
    if let Expr::List(spec) = &args[1] {
      if spec.len() == 2
        && let Expr::String(name) = &spec[0]
        && crate::functions::math_ast::try_eval_to_f64(&spec[1]).is_some()
      {
        let m = match name.as_str() {
          "Hard" => ThreshMethod::Hard,
          "Firm" => ThreshMethod::Firm,
          "Soft" => ThreshMethod::Soft,
          "PiecewiseGarrote" => ThreshMethod::Garrote,
          "Hyperbola" => ThreshMethod::Hyperbola,
          "SmoothGarrote" => ThreshMethod::SmoothGarrote,
          _ => return Ok(unevaluated()),
        };
        (m, spec[1].clone())
      } else {
        return Ok(unevaluated());
      }
    } else if crate::functions::math_ast::try_eval_to_f64(&args[1]).is_some() {
      (ThreshMethod::Hard, args[1].clone())
    } else {
      return Ok(unevaluated());
    }
  } else {
    (
      ThreshMethod::Hard,
      call(
        "Rational",
        vec![Expr::Integer(1), Expr::Integer(10_000_000_000)],
      ),
    )
  };
  // When the data array contains any inexact value (Real/BigFloat), the whole
  // result is real-valued: every surviving leaf and every introduced zero is
  // converted to a machine Real, matching wolframscript. An all-exact array
  // (Integers/Rationals) keeps its exact leaves.
  //
  // The threshold also influences the result type for every method except bare
  // "Hard" (which just selects the original leaf and never combines it with the
  // threshold, so it stays exact). For the others — Firm included — wolframscript
  // promotes the whole array to machine reals when the threshold is inexact OR
  // an Integer; only a non-integer Rational threshold (e.g. 3/2) keeps the
  // result exact. (`4/2` reduces to the Integer 2 before reaching here.)
  fn contains_inexact(e: &Expr) -> bool {
    match e {
      Expr::Real(_) | Expr::BigFloat(_, _) => true,
      Expr::List(items) => items.iter().any(contains_inexact),
      _ => false,
    }
  }
  let threshold_promotes = !matches!(method, ThreshMethod::Hard)
    && (contains_inexact(&threshold) || matches!(threshold, Expr::Integer(_)));
  let promote_to_real = contains_inexact(&args[0]) || threshold_promotes;
  // Walk the array recursively. A non-list, non-numeric leaf triggers
  // the Threshold::nlist message and aborts.
  fn apply(
    data: &Expr,
    t: &Expr,
    promote: bool,
    method: ThreshMethod,
  ) -> Option<Expr> {
    match data {
      Expr::List(items) => {
        if items.is_empty() {
          // Empty list at the top level was already rejected; nested
          // empty lists round-trip unchanged.
          return Some(Expr::List(items.clone()));
        }
        let mut out = Vec::with_capacity(items.len());
        for item in items {
          out.push(apply(item, t, promote, method)?);
        }
        Some(Expr::List(out.into()))
      }
      x if crate::functions::math_ast::try_eval_to_f64(x).is_some() => {
        if promote {
          // Machine-real result: compute directly in f64 so the rounding
          // matches wolframscript's machine arithmetic exactly (a symbolic
          // reduction back to f64 can drift in the last ULP).
          let f = threshold_one_method_f64(x, t, method);
          // Normalize -0.0 → 0.0 so it prints as `0.` like wolframscript.
          Some(Expr::Real(if f == 0.0 { 0.0 } else { f }))
        } else {
          Some(threshold_one_method(x, t, method))
        }
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
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated());
  }
  if let Some(result) = apply(&args[0], &threshold, promote_to_real, method) {
    Ok(result)
  } else {
    crate::emit_message(&format!(
      "Threshold::nlist: Argument {} is not a nonempty list or rectangular array of numeric quantities.",
      expr_to_string(&args[0])
    ));
    Ok(unevaluated())
  }
}

/// Thresholding method for `Threshold[data, {method, delta}]`.
#[derive(Clone, Copy)]
enum ThreshMethod {
  /// Keep x when |x| > delta, else 0 (the default).
  Hard,
  /// `Firm[delta]` with a single cutoff is identical to `Hard` (the firm
  /// transition region only exists with a two-value spec, which Wolfram
  /// rejects here).
  Firm,
  /// Sign[x] * Max[|x| - delta, 0].
  Soft,
  /// Non-negative garrote: x - delta^2 / x when |x| > delta, else 0.
  Garrote,
  /// Garrote hyperbola: Sign[x] * Sqrt[x^2 - delta^2] when |x| > delta, else 0.
  Hyperbola,
  /// Smooth garrote: x^3 / (x^2 + delta^2) for all x.
  SmoothGarrote,
}

/// Apply a single thresholding method to one numeric leaf. "Hard"/"Firm"
/// preserve the leaf's exact/inexact kind via `threshold_one`. The remaining
/// methods build the defining expression and let the evaluator reduce it, so
/// exact inputs stay exact (e.g. `Threshold[{2}, {"Hyperbola", 3/2}]` →
/// `Sqrt[7]/2`) and inexact inputs reduce to a machine real — matching
/// wolframscript. The caller (`apply`) handles real-promotion of the array.
fn threshold_one_method(x: &Expr, t: &Expr, method: ThreshMethod) -> Expr {
  // Helper constructors for the symbolic forms.
  let pow = |b: Expr, e: i128| pow(b, Expr::Integer(e));
  let abs_x = call1("Abs", x.clone());
  let expr = match method {
    ThreshMethod::Hard | ThreshMethod::Firm => return threshold_one(x, t),
    // Sign[x] * Max[|x| - t, 0]
    ThreshMethod::Soft => times(vec![
      call1("Sign", x.clone()),
      call(
        "Max",
        vec![plus(vec![abs_x, neg(t.clone())]), Expr::Integer(0)],
      ),
    ]),
    // If[|x| > t, x - t^2/x, 0]
    ThreshMethod::Garrote => call(
      "If",
      vec![
        call("Greater", vec![abs_x, t.clone()]),
        plus(vec![
          x.clone(),
          times(vec![
            Expr::Integer(-1),
            pow(t.clone(), 2),
            pow(x.clone(), -1),
          ]),
        ]),
        Expr::Integer(0),
      ],
    ),
    // If[|x| > t, Sign[x] * Sqrt[x^2 - t^2], 0]
    ThreshMethod::Hyperbola => call(
      "If",
      vec![
        call("Greater", vec![abs_x, t.clone()]),
        times(vec![
          call1("Sign", x.clone()),
          call(
            "Sqrt",
            vec![plus(vec![
              pow(x.clone(), 2),
              times(vec![Expr::Integer(-1), pow(t.clone(), 2)]),
            ])],
          ),
        ]),
        Expr::Integer(0),
      ],
    ),
    // x^3 / (x^2 + t^2)
    ThreshMethod::SmoothGarrote => times(vec![
      pow(x.clone(), 3),
      pow(plus(vec![pow(x.clone(), 2), pow(t.clone(), 2)]), -1),
    ]),
  };
  crate::evaluator::evaluate_expr_to_expr(&expr).unwrap_or(expr)
}

/// Machine-real version of `threshold_one_method`, used when the result is
/// promoted to reals. Computing in f64 directly (rather than reducing the
/// symbolic form back to f64) matches wolframscript's machine arithmetic in
/// the last ULP.
fn threshold_one_method_f64(x: &Expr, t: &Expr, method: ThreshMethod) -> f64 {
  let xf = crate::functions::math_ast::try_eval_to_f64(x).unwrap_or(0.0);
  let tf = crate::functions::math_ast::try_eval_to_f64(t).unwrap_or(0.0);
  match method {
    ThreshMethod::Hard | ThreshMethod::Firm => {
      if xf.abs() > tf {
        xf
      } else {
        0.0
      }
    }
    ThreshMethod::Soft => xf.signum() * (xf.abs() - tf).max(0.0),
    ThreshMethod::Garrote => {
      if xf.abs() > tf {
        xf - tf * tf / xf
      } else {
        0.0
      }
    }
    ThreshMethod::Hyperbola => {
      if xf.abs() > tf {
        xf.signum() * (xf * xf - tf * tf).sqrt()
      } else {
        0.0
      }
    }
    ThreshMethod::SmoothGarrote => xf * xf * xf / (xf * xf + tf * tf),
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

/// One axis of an ImagePartition size spec.
enum PartitionAxisMode {
  /// Plain size `n`: top-left anchored grid, only complete blocks kept.
  Full(usize),
  /// `{n}` form: centered grid, partial edge blocks are kept (clipped).
  Clipped(usize),
}

/// A positive block size / offset component: numeric, floored, >= 1.
fn partition_positive_int(e: &Expr) -> Option<usize> {
  let v = crate::functions::math_ast::try_eval_to_f64(e)?;
  if !v.is_finite() {
    return None;
  }
  let f = v.floor();
  if f >= 1.0 && f <= u32::MAX as f64 {
    Some(f as usize)
  } else {
    None
  }
}

/// A block size / offset component along one axis of length `dim` pixels:
/// a plain number (floored, >= 1) or `Scaled[s]`, a fraction of `dim`
/// (rounded, >= 1).
fn partition_size_value(e: &Expr, dim: usize) -> Option<usize> {
  if let Expr::FunctionCall { name, args } = e
    && name == "Scaled"
    && args.len() == 1
  {
    let s = crate::functions::math_ast::try_eval_to_f64(&args[0])?;
    if !s.is_finite() {
      return None;
    }
    let px = (s * dim as f64).round();
    return if px >= 1.0 && px <= u32::MAX as f64 {
      Some(px as usize)
    } else {
      None
    };
  }
  partition_positive_int(e)
}

/// Parse one element of a two-element size spec: `n`, `{n}`, `Scaled[s]` or
/// `{Scaled[s]}`, against an axis of length `dim` pixels.
fn partition_size_elem(e: &Expr, dim: usize) -> Option<PartitionAxisMode> {
  match e {
    Expr::List(items) if items.len() == 1 => {
      partition_size_value(&items[0], dim).map(PartitionAxisMode::Clipped)
    }
    Expr::List(_) => None,
    _ => partition_size_value(e, dim).map(PartitionAxisMode::Full),
  }
}

/// Parse the full size spec (2nd argument) into per-axis (x, y) modes.
/// `width`/`height` are the image's pixel dimensions, needed to resolve a
/// `Scaled[s]` fraction into a pixel count on each axis.
fn partition_size_spec(
  spec: &Expr,
  width: usize,
  height: usize,
) -> Option<(PartitionAxisMode, PartitionAxisMode)> {
  match spec {
    Expr::List(items) if items.len() == 1 => {
      // {s} (and {{s}}) apply the clipped mode to both axes.
      let inner = match &items[0] {
        Expr::List(inner) if inner.len() == 1 => &inner[0],
        Expr::List(_) => return None,
        e => e,
      };
      let nx = partition_size_value(inner, width)?;
      let ny = partition_size_value(inner, height)?;
      Some((
        PartitionAxisMode::Clipped(nx),
        PartitionAxisMode::Clipped(ny),
      ))
    }
    Expr::List(items) if items.len() == 2 => Some((
      partition_size_elem(&items[0], width)?,
      partition_size_elem(&items[1], height)?,
    )),
    Expr::List(_) => None,
    e => {
      let nx = partition_size_value(e, width)?;
      let ny = partition_size_value(e, height)?;
      Some((PartitionAxisMode::Full(nx), PartitionAxisMode::Full(ny)))
    }
  }
}

/// Block positions along one axis as (start, length), top/left to
/// bottom/right, already clipped to the image extent.
fn partition_axis_blocks(
  dim: usize,
  mode: &PartitionAxisMode,
  step: usize,
) -> Vec<(usize, usize)> {
  match *mode {
    PartitionAxisMode::Full(w) => {
      let w = w.min(dim);
      let mut blocks = Vec::new();
      let mut x = 0;
      while x + w <= dim {
        blocks.push((x, w));
        x += step;
      }
      blocks
    }
    PartitionAxisMode::Clipped(w) => {
      // A grid of ceil(dim/step) blocks is centered on the image (odd
      // overhang goes to the leading edge); the kept blocks are every grid
      // position whose center lies within the closed image interval,
      // clipped to the image extent.
      let n = dim.div_ceil(step);
      let span = w + (n - 1) * step;
      let overhang = span.saturating_sub(dim) as i64;
      let (w_i, step_i) = (w as i64, step as i64);
      let mut x = -((overhang + 1) / 2);
      while 2 * (x - step_i) + w_i >= 0 {
        x -= step_i;
      }
      let mut blocks = Vec::new();
      while 2 * x + w_i <= 2 * dim as i64 {
        let start = x.max(0) as usize;
        let end = ((x + w_i) as usize).min(dim);
        if end > start {
          blocks.push((start, end - start));
        }
        x += step_i;
      }
      blocks
    }
  }
}

/// ImagePartition[img, s], [img, {w, h}], [img, sizes, offsets].
/// Size components are `n` (complete blocks only), `{n}` (centered grid
/// keeping clipped partial blocks), or either wrapped in `Scaled[s]` to give
/// a size that is a fraction `s` of that axis's pixel dimension; sizes and
/// offsets are floored, offsets clamped to >= 1. Decoded from wolframscript
/// probes.
pub fn image_partition_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("ImagePartition", args);
  let Expr::Image {
    color_space,
    display_size: _,
    width,
    height,
    channels,
    ref data,
    image_type,
  } = args[0]
  else {
    crate::emit_message(&format!(
      "ImagePartition::imginv: Expecting an image or graphics instead of {}.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated());
  };

  let Some((mode_x, mode_y)) =
    partition_size_spec(&args[1], width as usize, height as usize)
  else {
    crate::emit_message(&format!(
      "ImagePartition::arg2: {} is not a valid size specification for image partitions.",
      expr_to_string(&args[1])
    ));
    return Ok(unevaluated());
  };

  let (dx, dy) = if args.len() == 3 {
    let invalid = |shown: &str| {
      crate::emit_message(&format!(
        "ImagePartition::arg3: {shown} is not a positive number or a pair of positive numbers."
      ));
    };
    // Positivity is checked on the raw value; the effective step is
    // max(1, floor(d)). Invalid scalars are shown normalized to a pair.
    let step_of = |e: &Expr| -> Option<usize> {
      let v = crate::functions::math_ast::try_eval_to_f64(e)?;
      if v > 0.0 {
        Some((v.floor() as usize).max(1))
      } else {
        None
      }
    };
    match &args[2] {
      Expr::List(items) if items.len() == 2 => {
        if let (Some(a), Some(b)) = (step_of(&items[0]), step_of(&items[1])) {
          (a, b)
        } else {
          invalid(&expr_to_string(&args[2]));
          return Ok(unevaluated());
        }
      }
      Expr::List(_) => {
        invalid(&expr_to_string(&args[2]));
        return Ok(unevaluated());
      }
      e => {
        if let Some(d) = step_of(e) {
          (d, d)
        } else {
          let shown = expr_to_string(e);
          invalid(&format!("{{{shown}, {shown}}}"));
          return Ok(unevaluated());
        }
      }
    }
  } else {
    // Default offsets are the (floored) block sizes.
    let size = |m: &PartitionAxisMode| match *m {
      PartitionAxisMode::Full(n) | PartitionAxisMode::Clipped(n) => n,
    };
    (size(&mode_x), size(&mode_y))
  };

  let (w, h, ch) = (width as usize, height as usize, channels as usize);
  let cols = partition_axis_blocks(w, &mode_x, dx);
  let rows = partition_axis_blocks(h, &mode_y, dy);

  let grid: Vec<Expr> = rows
    .iter()
    .map(|&(y0, bh)| {
      Expr::List(
        cols
          .iter()
          .map(|&(x0, bw)| {
            let mut block = Vec::with_capacity(bw * bh * ch);
            for y in y0..y0 + bh {
              let base = (y * w + x0) * ch;
              block.extend_from_slice(&data[base..base + bw * ch]);
            }
            Expr::Image {
              color_space,
              display_size: None,
              width: bw as u32,
              height: bh as u32,
              channels,
              data: Arc::new(block),
              image_type,
            }
          })
          .collect::<Vec<_>>()
          .into(),
      )
    })
    .collect();
  Ok(Expr::List(grid.into()))
}

pub fn image_take_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "ImageTake expects 2 or 3 arguments".into(),
    ));
  }

  if let Expr::Image {
    color_space,
    display_size: _,
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
  {
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
      color_space: *color_space,
      display_size: None,
      width: new_w as u32,
      height: new_h as u32,
      channels: *channels,
      data: Arc::new(new_data),
      image_type: *image_type,
    })
  } else {
    crate::emit_message(&format!(
      "ImageTake::imginv: Expecting an image or graphics instead of {}.",
      expr_to_string(&args[0])
    ));
    Ok(unevaluated("ImageTake", args))
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
    let count = base + usize::from(r < remainder);
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
    color_space: _,
    display_size: _,
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
    op: BinaryOperator::Times,
    left,
    right,
  } = expr
  {
    if let Some(w) = crate::functions::math_ast::try_eval_to_f64(left)
      && let Expr::Image {
        color_space: _,
        display_size: _,
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
        color_space: _,
        display_size: _,
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
        color_space: _,
        display_size: _,
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
        color_space: _,
        display_size: _,
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
  for it in items {
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
    color_space: None,
    display_size: None,
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

  let Expr::List(list) = &args[0] else {
    return Err(InterpreterError::EvaluationError(
      "ImageCollage: first argument must be a list of images".into(),
    ));
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
    for cell in *row {
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
    color_space: None,
    display_size: None,
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
  let Expr::List(outer) = &args[0] else {
    return Err(InterpreterError::EvaluationError(
      "ImageAssemble: first argument must be a list".into(),
    ));
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
  let num_cols = grid.iter().map(std::vec::Vec::len).max().unwrap_or(0);
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
      for cell in row {
        if let Some((_, _, h)) = cell
          && *h != expected_h
        {
          crate::emit_message(
            "ImageAssemble::row: \
               Expecting images of the same height in one row.",
          );
          return Ok(unevaluated("ImageAssemble", args));
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
          return Ok(unevaluated("ImageAssemble", args));
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
      "Import: cannot decode image: {e}"
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
  if !crate::vfs::exists(path) {
    crate::emit_message(&format!(
      "Import::nffil: File {path} not found during Import."
    ));
    return Ok(fail_expr());
  }
  let Ok(img) = image::open(crate::vfs::resolve(path)) else {
    // Wolfram returns $Failed when the file cannot be opened
    return Ok(fail_expr());
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
        "Import: failed to run curl: {e}"
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
/// - **Atkinson Hyperlegible Mono** (OFL) as the monospace fallback; it
///   shares the same hyperlegible design language as its proportional
///   sibling. This is a static Regular instance of the variable font:
///   usvg ignores font variations and draws a variable font's default
///   instance, which for the Mono font is ExtraLight — too thin to read
///   at typeset-output sizes.
///
/// Anything with exotic glyphs outside the embedded fonts will fall
/// through to `load_system_fonts()`.
#[cfg(not(target_arch = "wasm32"))]
fn load_embedded_fonts(fontdb: &mut resvg::usvg::fontdb::Database) {
  fontdb.load_font_data(
    include_bytes!("../../resources/AtkinsonHyperlegibleMono-Regular.ttf")
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

/// Extract an SVG rendering of `expr` when it is a `Graphics`/`Graphics3D`
/// expression, evaluated or still a bare `FunctionCall`. Returns `None`
/// for anything else.
fn graphics_svg(expr: &Expr) -> Option<String> {
  match expr {
    Expr::Graphics { svg, .. } => Some(svg.clone()),
    Expr::FunctionCall { name, args }
      if name == "Graphics" || name == "Graphics3D" =>
    {
      match crate::functions::graphics::graphics_ast(args) {
        Ok(Expr::Graphics { ref svg, .. }) => Some(svg.clone()),
        _ => None,
      }
    }
    _ => None,
  }
}

/// Coerce a `Graphics`/`Graphics3D` expression into an `Image` by
/// rasterizing it, matching wolframscript's image-processing functions
/// (`ImageResize`, `ImageData`, `ImageAdjust`, ...), which transparently
/// accept graphics wherever an image is expected. Returns `None` for
/// anything that isn't a renderable graphics expression, or on the
/// wasm32 target where the SVG rasterizer is unavailable — callers fall
/// back to their existing `imginv` handling in that case.
pub(crate) fn coerce_graphics_to_image(expr: &Expr) -> Option<Expr> {
  #[cfg(target_arch = "wasm32")]
  {
    let _ = expr;
    None
  }
  #[cfg(not(target_arch = "wasm32"))]
  {
    rasterize_svg(&graphics_svg(expr)?, 96.0).ok()
  }
}

/// The `ImageSize -> {w, h}` option among `opts` when it names an explicit
/// positive integer pixel size (a scalar width leaves the height to the
/// graphic's aspect ratio, which only affects display).
#[cfg(not(target_arch = "wasm32"))]
fn pixel_size_option(opts: &[Expr]) -> Option<Expr> {
  opts.iter().find_map(|opt| match opt {
    Expr::Rule {
      pattern,
      replacement,
    } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "ImageSize") => {
      match replacement.as_ref() {
        Expr::List(dims)
          if dims.len() == 2
            && dims.iter().all(|d| matches!(d, Expr::Integer(n) if *n > 0)) =>
        {
          Some(replacement.as_ref().clone())
        }
        _ => None,
      }
    }
    _ => None,
  })
}

/// Rasterize[expr] or Rasterize[expr, ImageResolution -> n]
/// Converts a Graphics, Grid, or other visual expression to a raster image.
#[cfg(not(target_arch = "wasm32"))]
pub fn rasterize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Rasterize[expr, elem, opts]: the element of the rasterization to give.
  if let Some(Expr::String(elem)) = args.get(1) {
    let mut rest = vec![args[0].clone()];
    rest.extend(args[2..].iter().cloned());
    let image = || rasterize_ast(&rest);
    let eval = |e: Expr| crate::evaluator::evaluate_expr_to_expr(&e);
    return match elem.as_str() {
      "Image" => image(),
      // The colour channels as bytes.
      "Data" => eval(call(
        "ImageData",
        vec![
          call("RemoveAlphaChannel", vec![image()?]),
          Expr::String("Byte".into()),
        ],
      )),
      "RasterSize" => eval(call("ImageDimensions", vec![image()?])),
      // {width, height, baseline}: the extent in printer's points and the
      // baseline's distance from the top. Graphics sit on their centre
      // (BaselinePosition -> Axis); a line of text or a formula on its
      // ascent, three quarters of the line height.
      "BoundingBox" => {
        let Expr::Image { width, height, .. } = rasterize_ast(&rest)? else {
          return Ok(Expr::Identifier("$Failed".into()));
        };
        let graphic = matches!(&args[0], Expr::Graphics { .. })
          || matches!(&args[0], Expr::FunctionCall { name, .. }
            if name == "Graphics" || name == "Graphics3D");
        let baseline = if graphic {
          height.div_ceil(2)
        } else {
          (0.75 * height as f64).round() as u32
        };
        Ok(Expr::List(
          vec![
            Expr::Integer(width as i128),
            Expr::Integer(height as i128),
            Expr::Integer(baseline as i128),
          ]
          .into(),
        ))
      }
      _ => {
        crate::emit_message(&format!(
          "Rasterize::elmntavl: \"{elem}\" is not an available element. \
           Possible elements include \"BoundingBox\", \"Data\", \
           \"Graphics\", \"RasterSize\", \"Regions\", and \"Image\"."
        ));
        Ok(Expr::Identifier("$Failed".into()))
      }
    };
  }
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
  let svg_str = if let Some(svg) = graphics_svg(&args[0]) {
    svg
  } else {
    match &args[0] {
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
      // Anything else (strings, `Style`d text, formulas) renders the way
      // `ExportString[expr, "SVG"]` draws it.
      _ => crate::evaluator::dispatch::io_functions::expr_to_svg(&args[0]),
    }
  };

  // RasterSize -> w fixes the width (the height follows the aspect ratio);
  // RasterSize -> {w, h} fixes both.
  let raster_size = args[1..].iter().find_map(|opt| match opt {
    Expr::Rule {
      pattern,
      replacement,
    } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "RasterSize") => {
      match replacement.as_ref() {
        Expr::Integer(w) if *w > 0 => Some((*w as u32, None)),
        Expr::List(d) => match d.as_slice() {
          [Expr::Integer(w), Expr::Integer(h)] if *w > 0 && *h > 0 => {
            Some((*w as u32, Some(*h as u32)))
          }
          _ => None,
        },
        _ => None,
      }
    }
    _ => None,
  });
  rasterize_svg_to(&svg_str, dpi, raster_size)
}

/// Rasterize an SVG string to an Expr::Image at the given DPI.
#[cfg(not(target_arch = "wasm32"))]
pub fn rasterize_svg(
  svg_str: &str,
  dpi: f64,
) -> Result<Expr, InterpreterError> {
  rasterize_svg_to(svg_str, dpi, None)
}

/// [`rasterize_svg`] at an explicit pixel size: `(w, None)` scales the
/// drawing to `w` pixels wide keeping its aspect ratio, `(w, Some(h))`
/// stretches it to exactly `w` x `h` (Rasterize's `RasterSize`).
#[cfg(not(target_arch = "wasm32"))]
fn rasterize_svg_to(
  svg_str: &str,
  dpi: f64,
  raster_size: Option<(u32, Option<u32>)>,
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
  let opt = resvg::usvg::Options {
    fontdb: StdArc::new(fontdb),
    ..Default::default()
  };

  let tree = resvg::usvg::Tree::from_str(svg_str, &opt).map_err(|e| {
    InterpreterError::EvaluationError(format!(
      "Rasterize: SVG parse error: {e}"
    ))
  })?;

  // Compute output pixel dimensions scaled by DPI
  let svg_size = tree.size();
  let scale = dpi / 96.0; // usvg default DPI is 96
  let (svg_w, svg_h) = (svg_size.width() as f64, svg_size.height() as f64);
  let (pix_w, pix_h, scale_x, scale_y) = match raster_size {
    Some((w, Some(h))) => (w, h, w as f64 / svg_w, h as f64 / svg_h),
    Some((w, None)) => {
      let k = w as f64 / svg_w;
      (w, ((svg_h * k).round() as u32).max(1), k, k)
    }
    None => (
      (svg_w * scale).ceil() as u32,
      (svg_h * scale).ceil() as u32,
      scale,
      scale,
    ),
  };

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
    resvg::tiny_skia::Transform::from_scale(scale_x as f32, scale_y as f32);
  resvg::render(&tree, transform, &mut pixmap.as_mut());

  // The drawing sits on an opaque white background, so the alpha channel
  // carries nothing: like Wolfram's Rasterize the result is an RGB image.
  let rgba_data = pixmap.data();
  let data: Vec<f64> = rgba_data
    .as_chunks::<4>()
    .0
    .iter()
    .flat_map(|px| px[..3].iter().map(|&v| v as f64 / 255.0))
    .collect();

  Ok(Expr::Image {
    color_space: Some("RGB"),
    display_size: None,
    width: pix_w,
    height: pix_h,
    channels: 3,
    data: Arc::new(data),
    image_type: ImageType::Byte,
  })
}

/// Resolve a `Masking -> spec` option (as used by `ImageApply`) into a
/// per-pixel inclusion mask matching an image of `target_w` x
/// `target_h` pixels. Returns `None` for `All`/`None`/`Automatic`,
/// meaning "no restriction — apply everywhere".
///
/// `Image` masks are centered on the target the way wolframscript
/// centers an array/image region of interest; any channel with a
/// positive value marks the pixel included. `Graphics` masks are
/// rasterized directly to `target_w` x `target_h` and any pixel the
/// drawing touches (non-zero alpha) is included.
fn resolve_mask(
  mask_expr: &Expr,
  target_w: usize,
  target_h: usize,
) -> Result<Option<Vec<bool>>, InterpreterError> {
  match mask_expr {
    Expr::Identifier(s) if s == "All" || s == "None" || s == "Automatic" => {
      Ok(None)
    }
    Expr::Image {
      width,
      height,
      channels,
      data,
      ..
    } => {
      let mw = *width as usize;
      let mh = *height as usize;
      let mch = *channels as usize;
      let off_x = (target_w as i64 - mw as i64) / 2;
      let off_y = (target_h as i64 - mh as i64) / 2;
      let mut mask = vec![false; target_w * target_h];
      for my in 0..mh {
        for mx in 0..mw {
          let tx = mx as i64 + off_x;
          let ty = my as i64 + off_y;
          if tx < 0 || ty < 0 || tx >= target_w as i64 || ty >= target_h as i64
          {
            continue;
          }
          let base = (my * mw + mx) * mch;
          let positive = data[base..base + mch].iter().any(|&v| v > 0.0);
          if positive {
            mask[ty as usize * target_w + tx as usize] = true;
          }
        }
      }
      Ok(Some(mask))
    }
    Expr::Graphics { svg, .. } => {
      rasterize_mask_svg(svg, target_w as u32, target_h as u32).map(Some)
    }
    Expr::FunctionCall { name, .. }
      if name == "Graphics" || name == "Graphics3D" =>
    {
      match graphics_svg(mask_expr) {
        Some(svg) => {
          rasterize_mask_svg(&svg, target_w as u32, target_h as u32).map(Some)
        }
        None => Err(InterpreterError::EvaluationError(
          "ImageApply: could not render Masking graphics".into(),
        )),
      }
    }
    _ => Err(InterpreterError::EvaluationError(
      "ImageApply: unsupported Masking specification".into(),
    )),
  }
}

/// Rasterize a Graphics SVG to exactly `w` x `h` pixels on a transparent
/// canvas and report which pixels the drawing touched (alpha > 0),
/// matching wolframscript's "use positive raster values" rule for a
/// `Masking -> Graphics[...]` region of interest.
#[cfg(not(target_arch = "wasm32"))]
fn rasterize_mask_svg(
  svg_str: &str,
  w: u32,
  h: u32,
) -> Result<Vec<bool>, InterpreterError> {
  use std::sync::Arc as StdArc;

  let mut fontdb = resvg::usvg::fontdb::Database::new();
  fontdb.load_system_fonts();
  load_embedded_fonts(&mut fontdb);
  let opt = resvg::usvg::Options {
    fontdb: StdArc::new(fontdb),
    ..Default::default()
  };

  let tree = resvg::usvg::Tree::from_str(svg_str, &opt).map_err(|e| {
    InterpreterError::EvaluationError(format!(
      "ImageApply: mask SVG parse error: {e}"
    ))
  })?;

  let svg_size = tree.size();
  if svg_size.width() <= 0.0 || svg_size.height() <= 0.0 || w == 0 || h == 0 {
    return Err(InterpreterError::EvaluationError(
      "ImageApply: mask has zero size".into(),
    ));
  }

  let mut pixmap = resvg::tiny_skia::Pixmap::new(w, h).ok_or_else(|| {
    InterpreterError::EvaluationError(
      "ImageApply: failed to create mask pixel buffer".into(),
    )
  })?;
  // Left transparent (not filled): painted (non-zero alpha) pixels are
  // the mask region, unpainted pixels are excluded.
  let scale_x = w as f32 / svg_size.width();
  let scale_y = h as f32 / svg_size.height();
  let transform = resvg::tiny_skia::Transform::from_scale(scale_x, scale_y);
  resvg::render(&tree, transform, &mut pixmap.as_mut());

  let rgba_data = pixmap.data();
  let mut mask = vec![false; (w * h) as usize];
  for i in 0..(w * h) as usize {
    mask[i] = rgba_data[i * 4 + 3] > 0;
  }
  Ok(mask)
}

#[cfg(target_arch = "wasm32")]
fn rasterize_mask_svg(
  _svg_str: &str,
  _w: u32,
  _h: u32,
) -> Result<Vec<bool>, InterpreterError> {
  Err(InterpreterError::EvaluationError(
    "ImageApply: Masking with a Graphics region is unavailable on this \
     target"
      .into(),
  ))
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
      "Export: cannot save \"{path}\": {e}"
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

  let file = File::create(crate::vfs::resolve(path)).map_err(|e| {
    InterpreterError::EvaluationError(format!(
      "Export: cannot save \"{path}\": {e}"
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

/// The pixel dimensions named by a `ConstantImage`/`RandomImage` size
/// argument: one positive integer for a square, or `{width, height}`.
/// `None` after reporting `<head>::bddim`, which is what a dimension that
/// is not a positive integer earns — a zero one used to reach the PNG
/// encoder and abort.
fn positive_image_dimensions(head: &str, spec: &Expr) -> Option<(u32, u32)> {
  let positive = |e: &Expr| match e {
    Expr::Integer(n) if *n >= 1 => u32::try_from(*n).ok(),
    _ => None,
  };
  let dimensions = match spec {
    Expr::List(dims) if dims.len() == 2 => {
      match (positive(&dims[0]), positive(&dims[1])) {
        (Some(w), Some(h)) => Some((w, h)),
        _ => None,
      }
    }
    other => positive(other).map(|s| (s, s)),
  };
  if dimensions.is_none() {
    crate::emit_message(&format!(
      "{head}::bddim: The specified dimensions {} should be a positive \
       integer or a list of positive integers for every spatial dimension.",
      crate::syntax::format_expr(spec, crate::syntax::ExprForm::Output)
    ));
  }
  dimensions
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
    let Some(dimensions) = positive_image_dimensions("ConstantImage", &args[1])
    else {
      return Ok(unevaluated("ConstantImage", args));
    };
    dimensions
  } else {
    (150u32, 150u32)
  };

  // Determine the color value(s)
  let (channels, pixel): (u8, Vec<f64>) = resolve_color_value(&args[0])?;

  let len = (w as usize) * (h as usize) * (channels as usize);
  let data: Vec<f64> = pixel.iter().cycle().take(len).copied().collect();

  Ok(Expr::Image {
    color_space: None,
    display_size: None,
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
      if let Some(color_expr) = crate::evaluator::named_color_expr(name) {
        resolve_color_value(&color_expr)
      } else {
        Err(InterpreterError::EvaluationError(format!(
          "ConstantImage: unknown color \"{name}\""
        )))
      }
    }
    _ => Err(InterpreterError::EvaluationError(format!(
      "ConstantImage: unsupported value {}",
      expr_to_string(expr)
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

/// Linear-light sRGB → CIE XYZ under a D50 illuminant. The coefficients
/// match Wolfram's internal sRGB → D50 conversion (Bradford chromatic
/// adaptation), so `ColorConvert[Red, "XYZ"]` lines up to f64 precision.
const LINEAR_RGB_TO_XYZ_D50: [[f64; 3]; 3] = [
  [
    0.43602191242669813,
    0.38510884137388846,
    0.14308124062061284,
  ],
  [0.22247517260243593, 0.7169066111623497, 0.06061821623521406],
  [
    0.013928134067434789,
    0.09710156693979348,
    0.7141585835116003,
  ],
];

/// Convert linear-light sRGB `(r, g, b)` (each in `[0, 1]`) to CIE XYZ
/// under a D50 illuminant.
fn linear_rgb_to_xyz_d50(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
  mat3_apply(&LINEAR_RGB_TO_XYZ_D50, (r, g, b))
}

/// Multiply the column vector `v` by the 3×3 matrix `m`.
fn mat3_apply(m: &[[f64; 3]; 3], v: (f64, f64, f64)) -> (f64, f64, f64) {
  let (a, b, c) = v;
  (
    m[0][0] * a + m[0][1] * b + m[0][2] * c,
    m[1][0] * a + m[1][1] * b + m[1][2] * c,
    m[2][0] * a + m[2][1] * b + m[2][2] * c,
  )
}

/// Invert a 3×3 matrix by cofactors. Every matrix inverted here is a
/// fixed, well-conditioned color-space matrix, so a singular input can
/// only mean a typo in a constant — hence the panic rather than an
/// `Option` the callers would have to thread through.
fn mat3_inverse(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
  let c = |r: usize, k: usize| -> f64 {
    let rows: Vec<usize> = (0..3).filter(|&i| i != r).collect();
    let cols: Vec<usize> = (0..3).filter(|&i| i != k).collect();
    let minor = m[rows[0]][cols[0]] * m[rows[1]][cols[1]]
      - m[rows[0]][cols[1]] * m[rows[1]][cols[0]];
    if (r + k).is_multiple_of(2) {
      minor
    } else {
      -minor
    }
  };
  let det = m[0][0] * c(0, 0) + m[0][1] * c(0, 1) + m[0][2] * c(0, 2);
  assert!(det != 0.0, "singular color-space matrix");
  // The inverse is the transposed cofactor matrix over the determinant.
  let mut out = [[0.0; 3]; 3];
  for (r, row) in out.iter_mut().enumerate() {
    for (k, cell) in row.iter_mut().enumerate() {
      *cell = c(k, r) / det;
    }
  }
  out
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

/// CIE XYZ under a D50 illuminant → gamma-encoded sRGB, the inverse of
/// `linear_rgb_to_xyz_d50` followed by `srgb_to_linear`.
fn xyz_d50_to_srgb(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
  let (r, g, b) = mat3_apply(&mat3_inverse(&LINEAR_RGB_TO_XYZ_D50), (x, y, z));
  (linear_to_srgb(r), linear_to_srgb(g), linear_to_srgb(b))
}

/// Wolfram-style scaled LAB (each component divided by 100) → sRGB, the
/// inverse of [`srgb_to_lab`].
fn lab_to_srgb(l: f64, a: f64, b: f64) -> (f64, f64, f64) {
  // The inverse of `lab_f`: a cube above the cutoff, affine below.
  fn lab_f_inverse(t: f64) -> f64 {
    const DELTA: f64 = 6.0 / 29.0;
    if t > DELTA {
      t * t * t
    } else {
      3.0 * DELTA * DELTA * (t - 4.0 / 29.0)
    }
  }
  let fy = (100.0 * l + 16.0) / 116.0;
  let fx = fy + 100.0 * a / 500.0;
  let fz = fy - 100.0 * b / 200.0;
  // The D50 reference white of `srgb_to_lab`.
  let x = 0.9642119944211995 * lab_f_inverse(fx);
  let y = lab_f_inverse(fy);
  let z = 0.8251882845188289 * lab_f_inverse(fz);
  xyz_d50_to_srgb(x, y, z)
}

/// Resolve any color expression (named, RGBColor, GrayLevel, etc.) to
/// scaled LAB. Returns `None` if the expression is not a color.
fn color_to_lab(expr: &Expr) -> Option<(f64, f64, f64)> {
  let color = crate::functions::graphics::parse_color(expr)?;
  Some(srgb_to_lab(color.r, color.g, color.b))
}

/// Convert a linear-light sRGB component back to its gamma-encoded value —
/// the inverse of [`srgb_to_linear`].
fn linear_to_srgb(c: f64) -> f64 {
  if c <= 0.0031308 {
    c * 12.92
  } else {
    1.055 * c.powf(1.0 / 2.4) - 0.055
  }
}

/// The Bradford cone-response matrix, XYZ → LMS. Chromatic adaptation
/// scales each LMS component independently, which is what makes a
/// reference color map onto a target one without wrecking the rest of the
/// palette.
const XYZ_TO_LMS_BRADFORD: [[f64; 3]; 3] = [
  [0.8951, 0.2664, -0.1614],
  [-0.7502, 1.7135, 0.0367],
  [0.0389, -0.0685, 1.0296],
];

/// sRGB `(r, g, b)` in `[0, 1]` → Bradford LMS cone responses.
fn srgb_to_lms(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
  let xyz = linear_rgb_to_xyz_d50(
    srgb_to_linear(r),
    srgb_to_linear(g),
    srgb_to_linear(b),
  );
  mat3_apply(&XYZ_TO_LMS_BRADFORD, xyz)
}

/// ColorBalance[img, ref] / ColorBalance[img, ref -> target] /
/// ColorBalance[img, ref, target] — white balancing by von Kries
/// chromatic adaptation, Wolfram's default `Method -> "LMSScaling"`.
///
/// Every pixel is taken into Bradford LMS cone space, each of the three
/// cone responses is scaled by `target / ref` in that same space, and the
/// result comes back to sRGB clipped to `[0, 1]`.
///
/// The two cone responses being divided are those of the *chromaticities*
/// of the reference and the target — each color's XYZ normalized to
/// `Y = 1` — so the adaptation changes the color cast without changing how
/// bright the picture is. That is why `ColorBalance[img, Green]` sends
/// green not to white but to the gray of green's own luminance, and why a
/// neutral gray comes back magenta.
///
/// A single-channel image is taken up to RGB first, so a grayscale picture
/// comes back colored; an alpha channel rides along untouched. The
/// automatic `ColorBalance[img]` form, which has to *estimate* the
/// reference from the image, is not implemented and stays unevaluated.
pub fn color_balance_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("ColorBalance", args);
  let Expr::Image {
    color_space,
    display_size: _,
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
  else {
    crate::emit_message(&format!(
      "ColorBalance::imginv: Expecting an image or graphics instead of {}.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated());
  };

  // Resolve the reference → target pair from the three accepted spellings.
  let white = call(
    "RGBColor",
    vec![Expr::Integer(1), Expr::Integer(1), Expr::Integer(1)],
  );
  let (source_expr, target_expr) = match args.len() {
    2 => match &args[1] {
      Expr::Rule {
        pattern,
        replacement,
      } => (pattern.as_ref().clone(), replacement.as_ref().clone()),
      other => (other.clone(), white),
    },
    3 => (args[1].clone(), args[2].clone()),
    _ => return Ok(unevaluated()),
  };
  let (Some(source), Some(target)) = (
    crate::functions::graphics::parse_color(&source_expr),
    crate::functions::graphics::parse_color(&target_expr),
  ) else {
    return Ok(unevaluated());
  };

  let (Some((sl, sm, ss)), Some((tl, tm, ts))) = (
    chromaticity_lms(source.r, source.g, source.b),
    chromaticity_lms(target.r, target.g, target.b),
  ) else {
    // A reference with no luminance (black) has no chromaticity to adapt
    // away from, so there is nothing to do.
    return Ok(unevaluated());
  };
  if sl <= 0.0 || sm <= 0.0 || ss <= 0.0 {
    return Ok(unevaluated());
  }
  let gains = (tl / sl, tm / sm, ts / ss);

  // A grayscale picture has no color channels to rebalance, but the
  // balancing gives it some: wolframscript takes it up to RGB first.
  let ch = (*channels as usize).max(3);
  let pixels = *width as usize * *height as usize;
  let mut new_data = if (*channels as usize) < 3 {
    let mut expanded = Vec::with_capacity(pixels * 3);
    for i in 0..pixels {
      let gray = data[i * *channels as usize];
      expanded.extend_from_slice(&[gray, gray, gray]);
    }
    expanded
  } else {
    data.as_ref().clone()
  };
  let color_space = if (*channels as usize) < 3 {
    Some("RGB")
  } else {
    *color_space
  };

  let lms_to_xyz = mat3_inverse(&XYZ_TO_LMS_BRADFORD);
  let xyz_to_linear_rgb = mat3_inverse(&LINEAR_RGB_TO_XYZ_D50);
  for i in 0..pixels {
    let base = i * ch;
    let (l, m, s) =
      srgb_to_lms(new_data[base], new_data[base + 1], new_data[base + 2]);
    let adapted = (l * gains.0, m * gains.1, s * gains.2);
    let (r, g, b) =
      mat3_apply(&xyz_to_linear_rgb, mat3_apply(&lms_to_xyz, adapted));
    new_data[base] = linear_to_srgb(r).clamp(0.0, 1.0);
    new_data[base + 1] = linear_to_srgb(g).clamp(0.0, 1.0);
    new_data[base + 2] = linear_to_srgb(b).clamp(0.0, 1.0);
  }
  Ok(Expr::Image {
    color_space,
    display_size: None,
    width: *width,
    height: *height,
    channels: ch as u8,
    data: Arc::new(new_data),
    image_type: *image_type,
  })
}

/// The Bradford cone responses of a color's *chromaticity* — its XYZ
/// normalized to `Y = 1`. Dividing one of these by another is the gain
/// that changes a color cast without changing the brightness. `None` for
/// a color with no luminance at all.
fn chromaticity_lms(r: f64, g: f64, b: f64) -> Option<(f64, f64, f64)> {
  let (x, y, z) = linear_rgb_to_xyz_d50(
    srgb_to_linear(r),
    srgb_to_linear(g),
    srgb_to_linear(b),
  );
  if y <= 0.0 {
    return None;
  }
  Some(mat3_apply(&XYZ_TO_LMS_BRADFORD, (x / y, 1.0, z / y)))
}

/// ColorDistance[c1, c2] — Euclidean distance between two colors in
/// (Wolfram-scaled) CIE LAB space. The optional
/// `DistanceFunction -> f` rule applies a custom function to the LAB
/// triples instead of the default Euclidean norm.
/// ColorReplace[image, c -> r], ColorReplace[image, {c1 -> r1, …}],
/// ColorReplace[image, c] and a trailing distance `d` (default 0.1).
///
/// A pixel takes the replacement of the first rule whose color lies within
/// `d` of it. Distance is Euclidean in the image's own channels, with the
/// pixel read at single precision and the bound scaled by the channel
/// count — `0.1` lets an RGB pixel within `0.3` of the color through, a
/// grayscale one within `0.1`, as wolframscript decides it. The match is
/// strict, so `d = 0` replaces nothing. Replacement colors are stored as
/// bytes (`GrayLevel[0.5]` becomes `128/255`); a bare color makes the
/// matching pixels transparent instead. A grayscale image stays grayscale
/// when every replacement is a `GrayLevel`, and the result is a Byte image
/// for Bit and Byte input and a Real32 image otherwise.
pub fn color_replace_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let Expr::Image {
    width,
    height,
    channels,
    data,
    image_type,
    ..
  } = &args[0]
  else {
    crate::emit_message(&format!(
      "ColorReplace::imginv: Expecting an image or graphics instead of {}.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated("ColorReplace", args));
  };
  let src_ch = *channels as usize;
  // Images with an alpha channel are not handled.
  if src_ch != 1 && src_ch != 3 {
    return Ok(unevaluated("ColorReplace", args));
  }
  let distance = match args.get(2) {
    None => 0.1,
    Some(d) => match crate::functions::math_ast::try_eval_to_f64(d) {
      Some(v) if v >= 0.0 => v,
      _ => return Ok(unevaluated("ColorReplace", args)),
    },
  };
  let is_gray = |e: &Expr| matches!(e, Expr::FunctionCall { name, .. } if name == "GrayLevel");
  // (target, replacement); `None` replaces with transparency.
  let specs: Vec<(&Expr, Option<&Expr>)> = match &args[1] {
    Expr::Rule {
      pattern,
      replacement,
    } => vec![(pattern.as_ref(), Some(replacement.as_ref()))],
    Expr::List(items) => {
      let mut specs = Vec::with_capacity(items.len());
      for item in items {
        match item {
          Expr::Rule {
            pattern,
            replacement,
          } => specs.push((pattern.as_ref(), Some(replacement.as_ref()))),
          _ => return Ok(unevaluated("ColorReplace", args)),
        }
      }
      specs
    }
    color => vec![(color, None)],
  };
  let mut rules: Vec<(Vec<f64>, Option<crate::functions::graphics::Color>)> =
    Vec::with_capacity(specs.len());
  for (target, replacement) in &specs {
    let Some(t) = crate::functions::graphics::parse_color(target) else {
      return Ok(unevaluated("ColorReplace", args));
    };
    let target = if src_ch == 1 {
      vec![if is_gray(target) {
        t.r
      } else {
        0.299 * t.r + 0.587 * t.g + 0.114 * t.b
      }]
    } else {
      vec![t.r, t.g, t.b]
    };
    let replacement = match replacement {
      None => None,
      Some(r) => match crate::functions::graphics::parse_color(r) {
        Some(c) => Some(c),
        None => return Ok(unevaluated("ColorReplace", args)),
      },
    };
    rules.push((target, replacement));
  }
  let gray_out =
    src_ch == 1 && specs.iter().all(|(_, r)| r.is_none_or(is_gray));
  let alpha_out = rules.iter().any(|(_, r)| r.is_none_or(|c| c.a < 1.0));
  let color_ch = if gray_out { 1 } else { 3 };
  let out_ch = color_ch + usize::from(alpha_out);
  let byte = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() / 255.0;
  let bound = distance * src_ch as f64;
  let pixels = (*width as usize) * (*height as usize);
  let mut out = Vec::with_capacity(pixels * out_ch);
  for p in 0..pixels {
    let px: Vec<f64> = data[p * src_ch..(p + 1) * src_ch]
      .iter()
      .map(|v| *v as f32 as f64)
      .collect();
    let hit = rules.iter().find(|(target, _)| {
      px.iter()
        .zip(target)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
        < bound
    });
    let color: Vec<f64> = match hit {
      Some((_, Some(c))) if gray_out => vec![byte(c.r)],
      Some((_, Some(c))) => vec![byte(c.r), byte(c.g), byte(c.b)],
      _ if color_ch == src_ch => px.clone(),
      _ => vec![px[0]; 3],
    };
    out.extend(color);
    if alpha_out {
      out.push(match hit {
        Some((_, Some(c))) => byte(c.a),
        Some((_, None)) => 0.0,
        None => 1.0,
      });
    }
  }
  Ok(Expr::Image {
    color_space: None,
    display_size: None,
    width: *width,
    height: *height,
    channels: out_ch as u8,
    data: Arc::new(out),
    image_type: if matches!(image_type, ImageType::Bit | ImageType::Byte) {
      ImageType::Byte
    } else {
      ImageType::Real32
    },
  })
}

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
      expr_to_string(&args[0])
    ))
  })?;
  let lab2 = color_to_lab(&args[1]).ok_or_else(|| {
    InterpreterError::EvaluationError(format!(
      "ColorDistance: unsupported color {}",
      expr_to_string(&args[1])
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
  let avg_c = f64::midpoint(c1, c2);
  let avg_c7 = avg_c.powi(7);
  let g = 0.5 * (1.0 - (avg_c7 / (avg_c7 + 25f64.powi(7))).sqrt());

  let a1p = a1 * (1.0 + g);
  let a2p = a2 * (1.0 + g);
  let c1p = (a1p * a1p + b1 * b1).sqrt();
  let c2p = (a2p * a2p + b2 * b2).sqrt();
  let avg_cp = f64::midpoint(c1p, c2p);

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

  let avg_lp = f64::midpoint(l1, l2);
  let avg_hp_deg = if cprod == 0.0 {
    h1p + h2p
  } else if (h1p - h2p).abs() <= 180.0 {
    f64::midpoint(h1p, h2p)
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
  let t = if (164.0..345.0).contains(&h1) {
    0.56 + (0.2 * cos(h1 + 168.0)).abs()
  } else {
    0.36 + (0.4 * cos(h1 + 35.0)).abs()
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

/// CrossingDetect[array | image] / CrossingDetect[..., delta] — marks the
/// zero crossings: after treating values with |v| < delta as zero, an
/// element is 1 exactly when its value is positive and some 8-neighbor is
/// negative (wolframscript-verified, including that zeros neither mark
/// nor trigger). Arrays give a binary SparseArray; a single-channel image
/// gives a binary ("Bit") image.
pub fn crossing_detect_ast(
  all_args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("CrossingDetect", all_args));
  // CornerNeighbors -> False (or None) uses only the 4 edge neighbors.
  let mut corners = true;
  let mut n_args = all_args.len();
  while n_args > 1
    && let Some((Expr::Identifier(opt), value, _)) =
      crate::functions::expr_form::rule_parts(&all_args[n_args - 1])
  {
    if opt != "CornerNeighbors" {
      return unevaluated();
    }
    corners = match value {
      Expr::Identifier(v) if v == "True" || v == "Automatic" => true,
      Expr::Identifier(v) if v == "False" || v == "None" => false,
      _ => return unevaluated(),
    };
    n_args -= 1;
  }
  let args = &all_args[..n_args];
  if args.is_empty() || args.len() > 2 {
    return unevaluated();
  }
  let delta = match args.get(1) {
    None => 0.0,
    Some(d) => match crate::functions::math_ast::try_eval_to_f64(d) {
      Some(v) if v >= 0.0 => v,
      _ => return unevaluated(),
    },
  };
  let zeroed = |v: f64| if v.abs() < delta { 0.0 } else { v };
  let crossings = |grid: &Vec<Vec<f64>>| -> Vec<Vec<i128>> {
    let h = grid.len();
    let w = grid.first().map_or(0, std::vec::Vec::len);
    let mut out = vec![vec![0i128; w]; h];
    for r in 0..h {
      for c in 0..w {
        let v = zeroed(grid[r][c]);
        if v <= 0.0 {
          continue;
        }
        'search: for dr in -1i64..=1 {
          for dc in -1i64..=1 {
            if (dr == 0 && dc == 0) || (!corners && dr != 0 && dc != 0) {
              continue;
            }
            let (nr, nc) = (r as i64 + dr, c as i64 + dc);
            if nr < 0 || nc < 0 || nr >= h as i64 || nc >= w as i64 {
              continue;
            }
            if zeroed(grid[nr as usize][nc as usize]) < 0.0 {
              out[r][c] = 1;
              break 'search;
            }
          }
        }
      }
    }
    out
  };
  let num = |e: &Expr| crate::functions::math_ast::try_eval_to_f64(e);

  match &args[0] {
    // Single-channel image → binary image.
    Expr::Image {
      width,
      height,
      channels: 1,
      data,
      ..
    } => {
      let (w, h) = (*width as usize, *height as usize);
      let grid: Vec<Vec<f64>> =
        (0..h).map(|r| data[r * w..(r + 1) * w].to_vec()).collect();
      let marked = crossings(&grid);
      let out: Vec<f64> = marked
        .iter()
        .flat_map(|row| row.iter().map(|&b| b as f64))
        .collect();
      Ok(Expr::Image {
        color_space: None,
        display_size: None,
        width: *width,
        height: *height,
        channels: 1,
        data: std::sync::Arc::new(out),
        image_type: ImageType::Bit,
      })
    }
    // 1-D numeric list → binary SparseArray vector.
    Expr::List(items)
      if !items.is_empty() && items.iter().all(|e| num(e).is_some()) =>
    {
      let row: Vec<f64> = items.iter().map(|e| num(e).unwrap()).collect();
      let marked = crossings(&vec![row]);
      let dense = Expr::List(
        marked[0]
          .iter()
          .map(|&b| Expr::Integer(b))
          .collect::<Vec<_>>()
          .into(),
      );
      crate::evaluator::evaluate_expr_to_expr(&call1("SparseArray", dense))
    }
    // 2-D numeric matrix → binary SparseArray matrix. (In one dimension
    // the 8-neighborhood reduces to the two adjacent elements.)
    Expr::List(rows)
      if !rows.is_empty()
        && rows.iter().all(|r| matches!(r, Expr::List(_))) =>
    {
      let mut grid: Vec<Vec<f64>> = Vec::with_capacity(rows.len());
      let mut width: Option<usize> = None;
      for r in rows {
        let Expr::List(cells) = r else {
          return unevaluated();
        };
        if *width.get_or_insert(cells.len()) != cells.len() || cells.is_empty()
        {
          return unevaluated();
        }
        let mut vals = Vec::with_capacity(cells.len());
        for c in cells {
          match num(c) {
            Some(v) => vals.push(v),
            None => return unevaluated(),
          }
        }
        grid.push(vals);
      }
      let marked = crossings(&grid);
      let dense = Expr::List(
        marked
          .iter()
          .map(|row| {
            Expr::List(
              row
                .iter()
                .map(|&b| Expr::Integer(b))
                .collect::<Vec<_>>()
                .into(),
            )
          })
          .collect::<Vec<_>>()
          .into(),
      );
      crate::evaluator::evaluate_expr_to_expr(&call1("SparseArray", dense))
    }
    _ => unevaluated(),
  }
}

// ---------------------------------------------------------------------------
// ImageMeasurements
// ---------------------------------------------------------------------------

/// The measurements `ImageMeasurements` names, in the order it lists them.
const IMAGE_MEASUREMENT_PROPERTIES: [&str; 32] = [
  "AspectRatio",
  "Channels",
  "ColorSpace",
  "ContourHierarchy",
  "Contours",
  "DataRange",
  "DataType",
  "Dimensions",
  "Energy",
  "Entropy",
  "ImageDimensions",
  "IntensityCentroid",
  "IntensityMomentsMatrix",
  "Interleaving",
  "Max",
  "MaxIntensity",
  "Mean",
  "MeanIntensity",
  "Median",
  "MedianIntensity",
  "Min",
  "MinIntensity",
  "MinMax",
  "MinMaxIntensity",
  "PerimeterPositions",
  "SampleDepth",
  "Skew",
  "StandardDeviation",
  "StandardDeviationIntensity",
  "Total",
  "TotalIntensity",
  "Transparency",
];

/// The samples of one channel, in row order.
fn channel_samples(data: &[f64], channels: usize, channel: usize) -> Vec<f64> {
  data
    .iter()
    .skip(channel)
    .step_by(channels)
    .copied()
    .collect()
}

/// The median of a run of samples.
fn samples_median(samples: &[f64]) -> f64 {
  let mut sorted = samples.to_vec();
  sorted.sort_by(f64::total_cmp);
  let n = sorted.len();
  if n % 2 == 1 {
    sorted[n / 2]
  } else {
    f64::midpoint(sorted[n / 2 - 1], sorted[n / 2])
  }
}

/// The sample standard deviation of a run of samples.
fn samples_standard_deviation(samples: &[f64]) -> f64 {
  let n = samples.len();
  if n < 2 {
    return 0.0;
  }
  let mean = samples.iter().sum::<f64>() / n as f64;
  let sum: f64 = samples.iter().map(|v| (v - mean).powi(2)).sum();
  (sum / (n - 1) as f64).sqrt()
}

/// One measurement over each channel in turn, as a bare value for a
/// single-channel image and a list for any other.
fn per_channel(
  data: &[f64],
  channels: usize,
  measure: impl Fn(&[f64]) -> Expr,
) -> Expr {
  if channels == 1 {
    return measure(data);
  }
  Expr::List(
    (0..channels)
      .map(|c| measure(&channel_samples(data, channels, c)))
      .collect(),
  )
}

/// A single named measurement, or `None` when it is not one this handles.
fn image_measurement(
  width: u32,
  height: u32,
  channels: u8,
  data: &[f64],
  image_type: ImageType,
  color_space: Option<&'static str>,
  property: &str,
) -> Option<Expr> {
  let (w, h, ch) = (width as usize, height as usize, channels as usize);
  let real = |v: f64| Expr::Real(v);
  let count = data.len().max(1) as f64;
  // The histogram the information measures are taken over counts distinct
  // PIXELS, not distinct samples — a colour pixel is one observation however
  // many channels it carries.
  let pixels = (data.len() / ch.max(1)).max(1) as f64;
  let _ = count;
  let histogram = || -> Vec<f64> {
    let mut sorted: Vec<Vec<u64>> = data
      .chunks(ch.max(1))
      .map(|p| p.iter().map(|v| v.to_bits()).collect())
      .collect();
    sorted.sort();
    let mut counts = Vec::new();
    let mut run = 0.0;
    for i in 0..sorted.len() {
      run += 1.0;
      if i + 1 == sorted.len() || sorted[i + 1] != sorted[i] {
        counts.push(run / pixels);
        run = 0.0;
      }
    }
    counts
  };
  match property {
    "Mean" => Some(per_channel(data, ch, |s| {
      real(s.iter().sum::<f64>() / s.len().max(1) as f64)
    })),
    "Total" => Some(per_channel(data, ch, |s| real(s.iter().sum()))),
    "Min" => Some(per_channel(data, ch, |s| {
      real(s.iter().copied().fold(f64::INFINITY, f64::min))
    })),
    "Max" => Some(per_channel(data, ch, |s| {
      real(s.iter().copied().fold(f64::NEG_INFINITY, f64::max))
    })),
    "Median" => Some(per_channel(data, ch, |s| real(samples_median(s)))),
    // A single sample has no sample deviation. wolframscript reports that as
    // a one-element list for the plain measure and bare for the intensity
    // one, which is its own inconsistency rather than a rule.
    "StandardDeviation" if data.len() / ch.max(1) < 2 => {
      Some(Expr::List(vec![id_expr("Indeterminate"); ch].into()))
    }
    "StandardDeviation" => Some(per_channel(data, ch, |s| {
      real(samples_standard_deviation(s))
    })),
    "MinMax" => Some(per_channel(data, ch, |s| {
      Expr::List(
        vec![
          real(s.iter().copied().fold(f64::INFINITY, f64::min)),
          real(s.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
        ]
        .into(),
      )
    })),
    // The intensity measures read the grey level, which for a single-channel
    // image is the sample itself. A colour image would need its luma, which
    // is not handled here.
    "MeanIntensity"
    | "TotalIntensity"
    | "MinIntensity"
    | "MaxIntensity"
    | "MedianIntensity"
    | "StandardDeviationIntensity"
    | "MinMaxIntensity"
      if ch == 1 =>
    {
      let plain = image_measurement(
        width,
        height,
        channels,
        data,
        image_type,
        color_space,
        property.trim_end_matches("Intensity"),
      )?;
      // `MinMax` is a pair either way; a lone `Indeterminate` is not wrapped
      // here the way the plain measure wraps it.
      Some(match (&plain, property) {
        (Expr::List(items), "StandardDeviationIntensity")
          if items.len() == 1 =>
        {
          items[0].clone()
        }
        _ => plain,
      })
    }
    // Shannon entropy of the sample histogram, in nats.
    "Entropy" => Some(real(
      -histogram()
        .iter()
        .map(|p| if *p > 0.0 { p * p.ln() } else { 0.0 })
        .sum::<f64>(),
    )),
    // The histogram's own squared weight.
    "Energy" => Some(real(histogram().iter().map(|p| p * p).sum())),
    // The centre of mass of the intensities, over pixel centres measured from
    // the bottom left.
    "IntensityCentroid" if ch == 1 => {
      let total: f64 = data.iter().sum();
      if total == 0.0 {
        return None;
      }
      let mut cx = 0.0;
      let mut cy = 0.0;
      for (i, value) in data.iter().enumerate() {
        let (col, row) = (i % w, i / w);
        cx += (col as f64 + 0.5) * value;
        // Rows are stored from the top, and the axis runs from the bottom.
        cy += (h as f64 - row as f64 - 0.5) * value;
      }
      Some(Expr::List(vec![real(cx / total), real(cy / total)].into()))
    }
    "Channels" => Some(Expr::Integer(ch as i128)),
    "ColorSpace" => Some(match color_space {
      Some(name) => str_expr(name),
      None => id_expr("Automatic"),
    }),
    "ImageDimensions" => Some(Expr::List(
      vec![Expr::Integer(w as i128), Expr::Integer(h as i128)].into(),
    )),
    "Dimensions" => Some(Expr::List(
      if ch == 1 {
        vec![Expr::Integer(h as i128), Expr::Integer(w as i128)]
      } else {
        vec![
          Expr::Integer(h as i128),
          Expr::Integer(w as i128),
          Expr::Integer(ch as i128),
        ]
      }
      .into(),
    )),
    "AspectRatio" => crate::evaluator::evaluate_expr_to_expr(&div(
      Expr::Integer(h as i128),
      Expr::Integer(w as i128),
    ))
    .ok(),
    "DataType" => Some(Expr::String(
      match image_type {
        ImageType::Bit => "Bit",
        ImageType::Byte => "UnsignedInteger8",
        ImageType::Bit16 => "UnsignedInteger16",
        ImageType::Real32 => "Real32",
        ImageType::Real64 => "Real64",
      }
      .to_string(),
    )),
    "DataRange" => Some(Expr::List(vec![real(0.0), real(1.0)].into())),
    "Interleaving" => Some(Expr::Identifier(
      if ch == 1 { "None" } else { "True" }.to_string(),
    )),
    "SampleDepth" => Some(Expr::Integer(match image_type {
      ImageType::Bit => 1,
      ImageType::Byte => 8,
      ImageType::Bit16 => 16,
      ImageType::Real32 => 32,
      ImageType::Real64 => 64,
    })),
    "Transparency" => Some(bool_expr(false)),
    _ => None,
  }
}

/// `ImageMeasurements[image, property]` — a named measurement of an image, or
/// a list of them for a list of names.
pub fn image_measurements_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let echo = || Ok(unevaluated("ImageMeasurements", args));
  if args.len() < 2 || args.len() > 3 {
    let n = args.len();
    let noun = if n == 1 { "argument" } else { "arguments" };
    crate::emit_message(&format!(
      "ImageMeasurements::argtu: ImageMeasurements called with {n} {noun}; \
       2 or 3 arguments are expected."
    ));
    return echo();
  }
  let Expr::Image {
    width,
    height,
    channels,
    data,
    image_type,
    color_space,
    display_size: _,
  } = &args[0]
  else {
    return echo();
  };
  let one = |property: &Expr| -> Option<Result<Expr, InterpreterError>> {
    let Expr::String(name) = property else {
      return None;
    };
    if name == "Properties" {
      return Some(Ok(Expr::List(
        IMAGE_MEASUREMENT_PROPERTIES
          .iter()
          .map(|p| str_expr(p))
          .collect(),
      )));
    }
    if !IMAGE_MEASUREMENT_PROPERTIES.contains(&name.as_str()) {
      crate::emit_message(&format!(
        "ImageMeasurements::invprop: {name} is not a known image-measurement \
         property."
      ));
      return Some(echo());
    }
    image_measurement(
      *width,
      *height,
      *channels,
      data,
      *image_type,
      *color_space,
      name,
    )
    .map(Ok)
  };
  match &args[1] {
    Expr::List(names) => {
      let mut out = Vec::with_capacity(names.len());
      for name in names {
        match one(name) {
          Some(Ok(value)) => out.push(value),
          Some(Err(e)) => return Err(e),
          None => return echo(),
        }
      }
      Ok(Expr::List(out.into()))
    }
    property => match one(property) {
      Some(result) => result,
      None => echo(),
    },
  }
}

/// `{{a, b}, {c, d}}` of numbers → `[a, b, c, d]`.
fn range_pair_of_pairs(e: &Expr) -> Option<[f64; 4]> {
  let Expr::List(rows) = e else { return None };
  if rows.len() != 2 {
    return None;
  }
  let mut out = [0.0; 4];
  for (i, row) in rows.iter().enumerate() {
    let Expr::List(p) = row else { return None };
    if p.len() != 2 {
      return None;
    }
    out[2 * i] = crate::functions::math_ast::try_eval_to_f64(&p[0])?;
    out[2 * i + 1] = crate::functions::math_ast::try_eval_to_f64(&p[1])?;
  }
  Some(out)
}

/// A positive pixel count given as an integer or a real (rounded).
fn pixel_extent(e: &Expr) -> Option<usize> {
  let v = crate::functions::math_ast::try_eval_to_f64(e)?.round();
  (v >= 1.0 && v.is_finite()).then_some(v as usize)
}

/// ImageForwardTransformation[img, f, {w, h}] — map every point of `img`
/// through `f` and resample the result onto a `w × h` image.
///
/// `f` takes a `{x, y}` point (in `DataRange` coordinates, y pointing up)
/// and returns its new position. `DataRange` defaults to the pixel extent
/// of `img`; `PlotRange` (the part of the transformed plane that is shown)
/// defaults to the bounding box of the transformed image. The source is cut
/// into a lattice of cells whose corners are mapped through `f`; each cell
/// is rasterized as two triangles with the source position interpolated
/// across them, so a smooth `f` leaves no holes. `Background` fills the
/// uncovered pixels (black by default).
pub fn image_forward_transformation_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let echo = || Ok(unevaluated("ImageForwardTransformation", args));
  if args.len() < 2 {
    return echo();
  }
  let Expr::Image {
    width,
    height,
    channels,
    data,
    image_type,
    ..
  } = &args[0]
  else {
    return echo();
  };
  let (sw, sh, ch) = (*width as usize, *height as usize, *channels as usize);

  // Positional size, then options.
  let mut rest = &args[2..];
  let mut size: Option<(usize, usize)> = None;
  if let Some(first) = rest.first()
    && !matches!(first, Expr::Rule { .. } | Expr::RuleDelayed { .. })
  {
    size = match first {
      Expr::List(v) if v.len() == 2 => {
        match (pixel_extent(&v[0]), pixel_extent(&v[1])) {
          (Some(w), Some(h)) => Some((w, h)),
          _ => return echo(),
        }
      }
      // A lone width: the height follows the plot range's aspect ratio,
      // settled once that range is known.
      other => match pixel_extent(other) {
        Some(w) => Some((w, 0)),
        _ => return echo(),
      },
    };
    rest = &rest[1..];
  }
  let full_range = [0.0, sw as f64, 0.0, sh as f64];
  let data_range = option_value(rest, "DataRange")
    .and_then(range_pair_of_pairs)
    .unwrap_or(full_range);
  let plot_range =
    option_value(rest, "PlotRange").and_then(range_pair_of_pairs);
  let background: Vec<f64> =
    match option_value(rest, "Background").and_then(color_directive_to_rgb) {
      Some((r, g, b, a)) => match ch {
        1 => vec![(r + g + b) / 3.0],
        2 => vec![(r + g + b) / 3.0, a.unwrap_or(1.0)],
        3 => vec![r, g, b],
        _ => vec![r, g, b, a.unwrap_or(1.0)],
      },
      None => vec![0.0; ch],
    };

  // Lattice nodes: source pixel coordinates (u, v) → transformed (X, Y).
  let mut nx = sw.clamp(2, 400);
  let mut ny = sh.clamp(2, 400);
  if let Some((w, h)) = size {
    nx = nx.max(w.min(400));
    ny = ny.max(h.min(400));
  }
  type Node = (f64, f64, f64, f64);
  let mut nodes: Vec<Option<Node>> = Vec::with_capacity((nx + 1) * (ny + 1));
  for j in 0..=ny {
    let v = sh as f64 * j as f64 / ny as f64;
    for i in 0..=nx {
      let u = sw as f64 * i as f64 / nx as f64;
      let dx = data_range[0] + (data_range[1] - data_range[0]) * u / sw as f64;
      let dy = data_range[2]
        + (data_range[3] - data_range[2]) * (sh as f64 - v) / sh as f64;
      let point = Expr::List(vec![Expr::Real(dx), Expr::Real(dy)].into());
      let mapped = crate::evaluator::apply_function_to_arg(&args[1], &point)
        .ok()
        .and_then(|r| match &r {
          Expr::List(p) if p.len() == 2 => Some((
            crate::functions::math_ast::try_eval_to_f64(&p[0])?,
            crate::functions::math_ast::try_eval_to_f64(&p[1])?,
          )),
          _ => None,
        })
        .filter(|(x, y)| x.is_finite() && y.is_finite());
      nodes.push(mapped.map(|(x, y)| (x, y, u, v)));
    }
  }

  let range = plot_range.unwrap_or_else(|| {
    let (mut x0, mut x1) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut y0, mut y1) = (f64::INFINITY, f64::NEG_INFINITY);
    for (x, y, _, _) in nodes.iter().flatten() {
      x0 = x0.min(*x);
      x1 = x1.max(*x);
      y0 = y0.min(*y);
      y1 = y1.max(*y);
    }
    [x0, x1, y0, y1]
  });
  let (rw, rh) = (range[1] - range[0], range[3] - range[2]);
  if !(rw.is_finite() && rh.is_finite()) || rw <= 0.0 || rh <= 0.0 {
    return echo();
  }
  let (ow, oh) = match size {
    Some((w, 0)) => (w, ((w as f64 * rh / rw).round() as usize).max(1)),
    Some(s) => s,
    None => (sw, sh),
  };
  if ow.saturating_mul(oh) > 64_000_000 {
    return echo();
  }

  let mut out = Vec::with_capacity(ow * oh * ch);
  for _ in 0..ow * oh {
    out.extend_from_slice(&background);
  }

  // Source sample at continuous pixel position (u, v) (pixel centres at +½).
  let sample = |u: f64, v: f64, dst: &mut [f64]| {
    let fx = (u - 0.5).clamp(0.0, sw as f64 - 1.0);
    let fy = (v - 0.5).clamp(0.0, sh as f64 - 1.0);
    let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(sw - 1), (y0 + 1).min(sh - 1));
    let (tx, ty) = (fx - x0 as f64, fy - y0 as f64);
    for (c, d) in dst.iter_mut().enumerate() {
      let at = |x: usize, y: usize| data[(y * sw + x) * ch + c];
      *d = (at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx) * (1.0 - ty)
        + (at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx) * ty;
    }
  };

  // Output pixel coordinates of a node (continuous, y pointing down).
  let to_px = |x: f64, y: f64| {
    (
      (x - range[0]) / rw * ow as f64,
      (range[3] - y) / rh * oh as f64,
    )
  };
  let mut raster = |tri: [Node; 3]| {
    let p: Vec<(f64, f64)> = tri.iter().map(|n| to_px(n.0, n.1)).collect();
    let min_x = p.iter().map(|q| q.0).fold(f64::INFINITY, f64::min);
    let max_x = p.iter().map(|q| q.0).fold(f64::NEG_INFINITY, f64::max);
    let min_y = p.iter().map(|q| q.1).fold(f64::INFINITY, f64::min);
    let max_y = p.iter().map(|q| q.1).fold(f64::NEG_INFINITY, f64::max);
    if max_x < 0.0 || max_y < 0.0 || min_x > ow as f64 || min_y > oh as f64 {
      return;
    }
    let det = (p[1].1 - p[2].1) * (p[0].0 - p[2].0)
      + (p[2].0 - p[1].0) * (p[0].1 - p[2].1);
    if det.abs() < 1e-12 {
      return;
    }
    let ix0 = (min_x - 0.5).ceil().max(0.0) as usize;
    let ix1 = (max_x - 0.5).floor().min(ow as f64 - 1.0);
    let iy0 = (min_y - 0.5).ceil().max(0.0) as usize;
    let iy1 = (max_y - 0.5).floor().min(oh as f64 - 1.0);
    if ix1 < 0.0 || iy1 < 0.0 {
      return;
    }
    let eps = -1e-9;
    for iy in iy0..=(iy1 as usize) {
      for ix in ix0..=(ix1 as usize) {
        let (qx, qy) = (ix as f64 + 0.5, iy as f64 + 0.5);
        let l0 = ((p[1].1 - p[2].1) * (qx - p[2].0)
          + (p[2].0 - p[1].0) * (qy - p[2].1))
          / det;
        let l1 = ((p[2].1 - p[0].1) * (qx - p[2].0)
          + (p[0].0 - p[2].0) * (qy - p[2].1))
          / det;
        let l2 = 1.0 - l0 - l1;
        if l0 < eps || l1 < eps || l2 < eps {
          continue;
        }
        let u = l0 * tri[0].2 + l1 * tri[1].2 + l2 * tri[2].2;
        let v = l0 * tri[0].3 + l1 * tri[1].3 + l2 * tri[2].3;
        let base = (iy * ow + ix) * ch;
        sample(u, v, &mut out[base..base + ch]);
      }
    }
  };
  for j in 0..ny {
    for i in 0..nx {
      let at = |di: usize, dj: usize| nodes[(j + dj) * (nx + 1) + i + di];
      if let (Some(a), Some(b), Some(c), Some(d)) =
        (at(0, 0), at(1, 0), at(0, 1), at(1, 1))
      {
        raster([a, b, c]);
        raster([b, d, c]);
      }
    }
  }

  Ok(Expr::Image {
    color_space: None,
    display_size: None,
    width: ow as u32,
    height: oh as u32,
    channels: *channels,
    data: Arc::new(out),
    image_type: *image_type,
  })
}

/// ImageTransformation[img, f] — build an image whose every pixel takes the
/// colour `img` has at the point `f` maps that pixel's position to.
///
/// `f` receives an `{x, y}` point of the output (in `PlotRange` coordinates,
/// y pointing up) and returns a point of the input (in `DataRange`
/// coordinates); both ranges default to the pixel extent of `img`. The
/// result is `img`-sized unless a size is given. Positions that land
/// outside the input take the `Padding` value (black by default;
/// `"Fixed"` repeats the nearest edge pixel).
pub fn image_transformation_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let echo = || Ok(unevaluated("ImageTransformation", args));
  if args.len() < 2 {
    return echo();
  }
  let Expr::Image {
    width,
    height,
    channels,
    data,
    image_type,
    ..
  } = &args[0]
  else {
    return echo();
  };
  let (sw, sh, ch) = (*width as usize, *height as usize, *channels as usize);

  let mut rest = &args[2..];
  let mut size = (sw, sh);
  if let Some(first) = rest.first()
    && !matches!(first, Expr::Rule { .. } | Expr::RuleDelayed { .. })
  {
    size = match first {
      Expr::List(v) if v.len() == 2 => {
        match (pixel_extent(&v[0]), pixel_extent(&v[1])) {
          (Some(w), Some(h)) => (w, h),
          _ => return echo(),
        }
      }
      other => match pixel_extent(other) {
        Some(w) => (
          w,
          ((w as f64 * sh as f64 / sw as f64).round() as usize).max(1),
        ),
        _ => return echo(),
      },
    };
    rest = &rest[1..];
  }
  let (ow, oh) = size;
  if ow.saturating_mul(oh) > 4_000_000 {
    return echo();
  }
  let data_range = option_value(rest, "DataRange")
    .and_then(range_pair_of_pairs)
    .unwrap_or([0.0, sw as f64, 0.0, sh as f64]);
  let plot_range = option_value(rest, "PlotRange")
    .and_then(range_pair_of_pairs)
    .unwrap_or(data_range);
  let (dw, dh) = (data_range[1] - data_range[0], data_range[3] - data_range[2]);
  let (pw, ph) = (plot_range[1] - plot_range[0], plot_range[3] - plot_range[2]);
  if dw <= 0.0 || dh <= 0.0 || pw <= 0.0 || ph <= 0.0 {
    return echo();
  }

  // `None` clamps to the nearest edge pixel; otherwise the fill value.
  let padding: Option<Vec<f64>> = match option_value(rest, "Padding") {
    Some(Expr::String(s)) if s == "Fixed" => None,
    Some(value) => {
      Some(if let Some((r, g, b, a)) = color_directive_to_rgb(value) {
        match ch {
          1 => vec![(r + g + b) / 3.0],
          2 => vec![(r + g + b) / 3.0, a.unwrap_or(1.0)],
          3 => vec![r, g, b],
          _ => vec![r, g, b, a.unwrap_or(1.0)],
        }
      } else {
        let v =
          crate::functions::math_ast::try_eval_to_f64(value).unwrap_or(0.0);
        vec![v; ch]
      })
    }
    None => Some(vec![0.0; ch]),
  };

  let mut out = Vec::with_capacity(ow * oh * ch);
  for iy in 0..oh {
    let y = plot_range[3] - ph * (iy as f64 + 0.5) / oh as f64;
    for ix in 0..ow {
      let x = plot_range[0] + pw * (ix as f64 + 0.5) / ow as f64;
      let point = Expr::List(vec![Expr::Real(x), Expr::Real(y)].into());
      let source = crate::evaluator::apply_function_to_arg(&args[1], &point)
        .ok()
        .and_then(|r| match &r {
          Expr::List(p) if p.len() == 2 => Some((
            crate::functions::math_ast::try_eval_to_f64(&p[0])?,
            crate::functions::math_ast::try_eval_to_f64(&p[1])?,
          )),
          _ => None,
        })
        .filter(|(sx, sy)| sx.is_finite() && sy.is_finite());
      // Continuous source pixel position (pixel centres at +½, y down).
      let uv = source.map(|(sx, sy)| {
        (
          (sx - data_range[0]) / dw * sw as f64,
          (data_range[3] - sy) / dh * sh as f64,
        )
      });
      let inside = uv.filter(|(u, v)| {
        (0.0..=sw as f64).contains(u) && (0.0..=sh as f64).contains(v)
      });
      match (inside, uv, &padding) {
        (Some((u, v)), _, _) | (None, Some((u, v)), None) => {
          let fx = (u - 0.5).clamp(0.0, sw as f64 - 1.0);
          let fy = (v - 0.5).clamp(0.0, sh as f64 - 1.0);
          let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
          let (x1, y1) = ((x0 + 1).min(sw - 1), (y0 + 1).min(sh - 1));
          let (tx, ty) = (fx - x0 as f64, fy - y0 as f64);
          for c in 0..ch {
            let at = |x: usize, y: usize| data[(y * sw + x) * ch + c];
            out.push(
              (at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx) * (1.0 - ty)
                + (at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx) * ty,
            );
          }
        }
        (_, _, Some(fill)) => out.extend_from_slice(fill),
        _ => out.extend(std::iter::repeat_n(0.0, ch)),
      }
    }
  }

  Ok(Expr::Image {
    color_space: None,
    display_size: None,
    width: ow as u32,
    height: oh as u32,
    channels: *channels,
    data: Arc::new(out),
    image_type: *image_type,
  })
}

/// Bounding box `[xmin, xmax, ymin, ymax]` of the 2D points in a graphics
/// primitive tree (circle and disk radii included).
fn primitive_bounds(e: &Expr, bounds: &mut Option<[f64; 4]>) {
  fn grow(b: &mut Option<[f64; 4]>, x: f64, y: f64) {
    match b {
      Some(r) => {
        r[0] = r[0].min(x);
        r[1] = r[1].max(x);
        r[2] = r[2].min(y);
        r[3] = r[3].max(y);
      }
      None => *b = Some([x, x, y, y]),
    }
  }
  fn point_of(e: &Expr) -> Option<(f64, f64)> {
    match e {
      Expr::List(p) if p.len() == 2 => Some((
        crate::functions::math_ast::try_eval_to_f64(&p[0])?,
        crate::functions::math_ast::try_eval_to_f64(&p[1])?,
      )),
      _ => None,
    }
  }
  match e {
    Expr::List(items) => match point_of(e) {
      Some((x, y)) => grow(bounds, x, y),
      None => items.iter().for_each(|i| primitive_bounds(i, bounds)),
    },
    Expr::FunctionCall { name, args }
      if matches!(
        name.as_str(),
        "Disk" | "Circle" | "Rectangle" | "Line" | "Point" | "Polygon"
      ) =>
    {
      if let Some(first) = args.first() {
        let radius = if matches!(name.as_str(), "Disk" | "Circle") {
          match args.get(1) {
            Some(r) => crate::functions::math_ast::try_eval_to_f64(r),
            None => Some(1.0),
          }
          .unwrap_or(0.0)
        } else {
          0.0
        };
        if let Some((x, y)) = point_of(first) {
          grow(bounds, x - radius, y - radius);
          grow(bounds, x + radius, y + radius);
        } else if radius == 0.0 || !matches!(name.as_str(), "Disk" | "Circle") {
          primitive_bounds(first, bounds);
        } else {
          grow(bounds, -radius, -radius);
          grow(bounds, radius, radius);
        }
      }
      if matches!(name.as_str(), "Rectangle")
        && let Some(second) = args.get(1)
      {
        primitive_bounds(second, bounds);
      }
    }
    _ => {}
  }
}

/// Expand one `HighlightImage` annotation into plain graphics primitives:
/// `ImageMarker[pts, graphic]` places a copy of `graphic` (in image pixel
/// units) centred on each point; anything else is already a primitive.
fn highlight_primitives(mark: &Expr, out: &mut Vec<Expr>) {
  match mark {
    Expr::List(items) => {
      items.iter().for_each(|m| highlight_primitives(m, out));
    }
    Expr::FunctionCall { name, args } if name == "ImageMarker" => {
      let Some(points) = args.first() else { return };
      let Some(Expr::FunctionCall {
        name: gname,
        args: gargs,
      }) = args.get(1)
      else {
        return;
      };
      if gname != "Graphics" || gargs.is_empty() {
        return;
      }
      let prims = &gargs[0];
      let centre = if let Some(r) =
        option_value(&gargs[1..], "PlotRange").and_then(range_pair_of_pairs)
      {
        (f64::midpoint(r[0], r[1]), f64::midpoint(r[2], r[3]))
      } else {
        let mut b = None;
        primitive_bounds(prims, &mut b);
        b.map_or((0.0, 0.0), |r| {
          (f64::midpoint(r[0], r[1]), f64::midpoint(r[2], r[3]))
        })
      };
      let pts: Vec<&Expr> = match points {
        Expr::List(p)
          if p.len() == 2
            && p.iter().all(|c| {
              crate::functions::math_ast::try_eval_to_f64(c).is_some()
            }) =>
        {
          vec![points]
        }
        Expr::List(p) => p.iter().collect(),
        _ => return,
      };
      for pt in pts {
        let Expr::List(xy) = pt else { continue };
        if xy.len() != 2 {
          continue;
        }
        let (Some(x), Some(y)) = (
          crate::functions::math_ast::try_eval_to_f64(&xy[0]),
          crate::functions::math_ast::try_eval_to_f64(&xy[1]),
        ) else {
          continue;
        };
        out.push(call(
          "Translate",
          vec![
            prims.clone(),
            Expr::List(
              vec![Expr::Real(x - centre.0), Expr::Real(y - centre.1)].into(),
            ),
          ],
        ));
      }
    }
    other => out.push(other.clone()),
  }
}

/// HighlightImage[img, marks] — `img` with graphics `marks` drawn over it.
/// The marks live in image pixel coordinates (origin bottom-left); the result
/// is a new image the size of `img`. Display options such as `ImageSize` do
/// not change the pixels and are ignored.
pub fn highlight_image_ast(
  args: &[Expr],
  eval_fn: &dyn Fn(&Expr) -> Result<Expr, InterpreterError>,
) -> Result<Expr, InterpreterError> {
  let echo = || Ok(unevaluated("HighlightImage", args));
  let Some(Expr::Image { width, height, .. }) = args.first() else {
    return echo();
  };
  let (w, h) = (*width as f64, *height as f64);
  let mut prims = Vec::new();
  for mark in &args[1..] {
    if !matches!(mark, Expr::Rule { .. } | Expr::RuleDelayed { .. }) {
      highlight_primitives(mark, &mut prims);
    }
  }
  let call = |name: &str, args: Vec<Expr>| call(name, args);
  let pair =
    |a: f64, b: f64| Expr::List(vec![Expr::Real(a), Expr::Real(b)].into());
  let rule = |k: &str, v: Expr| rule_expr(id_expr(k), v);
  let mut content = vec![call(
    "Raster",
    vec![
      // `Raster` lists its rows bottom-first, `ImageData` top-first.
      call("Reverse", vec![call("ImageData", vec![args[0].clone()])]),
      Expr::List(vec![pair(0.0, 0.0), pair(w, h)].into()),
    ],
  )];
  content.extend(prims);
  let graphics = call(
    "Graphics",
    vec![
      Expr::List(content.into()),
      rule(
        "PlotRange",
        Expr::List(vec![pair(0.0, w), pair(0.0, h)].into()),
      ),
      rule("ImageSize", pair(w, h)),
      rule("PlotRangePadding", id_expr("None")),
      rule("ImagePadding", Expr::Integer(0)),
    ],
  );
  eval_fn(&call("Rasterize", vec![graphics]))
}

/// `Inpaint[image, mask]` — fills the pixels the mask marks (nonzero) from
/// their surroundings by diffusion: each masked pixel repeatedly becomes the
/// mean of its eight neighbors (Method -> "Diffusion", the heat-equation
/// fill) for `MaxIterations` sweeps (default 200). The mask is an image or
/// a matrix of the image's size.
pub fn inpaint_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let Some(Expr::Image {
    color_space,
    display_size,
    width,
    height,
    channels,
    data,
    image_type,
  }) = args.first()
  else {
    return Ok(unevaluated("Inpaint", args));
  };
  let (w, h, ch) = (*width as usize, *height as usize, *channels as usize);
  let mask: Vec<bool> = match args.get(1) {
    Some(Expr::Image {
      width: mw,
      height: mh,
      channels: mch,
      data: md,
      ..
    }) if *mw as usize == w && *mh as usize == h => {
      let mch = *mch as usize;
      (0..w * h)
        .map(|i| (0..mch).any(|c| md[i * mch + c] > 0.0))
        .collect()
    }
    Some(Expr::List(rows)) if rows.len() == h => {
      let mut m = Vec::with_capacity(w * h);
      for row in rows {
        let Expr::List(cells) = row else {
          return Ok(unevaluated("Inpaint", args));
        };
        if cells.len() != w {
          return Ok(unevaluated("Inpaint", args));
        }
        for c in cells {
          m.push(
            crate::functions::math_ast::try_eval_to_f64(c)
              .is_some_and(|v| v != 0.0),
          );
        }
      }
      m
    }
    _ => return Ok(unevaluated("Inpaint", args)),
  };
  let iterations = args[2..]
    .iter()
    .find_map(|o| match o {
      Expr::Rule { pattern, replacement }
        if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "MaxIterations") =>
      {
        match replacement.as_ref() {
          Expr::Integer(n) if *n >= 0 => Some(*n as usize),
          _ => None,
        }
      }
      _ => None,
    })
    .unwrap_or(200);
  let mut out: Vec<f64> = data.as_ref().clone();
  // Masked pixels start from the mean of the known ones, so the diffusion
  // has a neutral starting point.
  for c in 0..ch {
    let (sum, count) = (0..w * h)
      .filter(|&i| !mask[i])
      .fold((0.0, 0usize), |(s, n), i| (s + data[i * ch + c], n + 1));
    let mean = if count > 0 { sum / count as f64 } else { 0.0 };
    for i in 0..w * h {
      if mask[i] {
        out[i * ch + c] = mean;
      }
    }
  }
  let masked: Vec<usize> = (0..w * h).filter(|&i| mask[i]).collect();
  for _ in 0..iterations {
    let prev = out.clone();
    for &i in &masked {
      let (x, y) = (i % w, i / w);
      let mut neighbors = Vec::with_capacity(8);
      for dy in -1i64..=1 {
        for dx in -1i64..=1 {
          let (nx, ny) = (x as i64 + dx, y as i64 + dy);
          if (dx, dy) != (0, 0)
            && nx >= 0
            && ny >= 0
            && (nx as usize) < w
            && (ny as usize) < h
          {
            neighbors.push(ny as usize * w + nx as usize);
          }
        }
      }
      for c in 0..ch {
        let s: f64 = neighbors.iter().map(|&j| prev[j * ch + c]).sum();
        out[i * ch + c] = s / neighbors.len().max(1) as f64;
      }
    }
  }
  Ok(Expr::Image {
    color_space: *color_space,
    display_size: *display_size,
    width: *width,
    height: *height,
    channels: *channels,
    data: Arc::new(out),
    image_type: *image_type,
  })
}

/// ImageLines[image] / ImageLines[image, t] / ImageLines[image, t, MaxFeatures -> n]
/// — straight lines through the foreground pixels of `image`, found with a
/// Hough transform and returned as `{{x1, y1}, {x2, y2}}` pairs where the
/// ends lie on the image border (image coordinates, origin bottom-left),
/// strongest first. `t` is the minimum number of votes as a fraction of the
/// strongest line (default 0.5). `MaxFeatures -> n` keeps the n strongest.
pub fn image_lines_ast(all_args: &[Expr]) -> Result<Expr, InterpreterError> {
  let option_start = all_args
    .iter()
    .position(|a| matches!(a, Expr::Rule { .. } | Expr::RuleDelayed { .. }))
    .unwrap_or(all_args.len());
  let (args, opts) = all_args.split_at(option_start);
  let mut max_features: Option<usize> = None;
  for o in opts {
    match o {
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => match (pattern.as_ref(), replacement.as_ref()) {
        (Expr::Identifier(k), Expr::Integer(n)) if k == "MaxFeatures" => {
          max_features = Some((*n).max(0) as usize);
        }
        (Expr::Identifier(k), Expr::Identifier(a))
          if k == "MaxFeatures" && a == "All" => {}
        _ => return Ok(unevaluated("ImageLines", all_args)),
      },
      _ => return Ok(unevaluated("ImageLines", all_args)),
    }
  }
  let Some(Expr::Image {
    width,
    height,
    channels,
    data,
    ..
  }) = args.first()
  else {
    return Ok(unevaluated("ImageLines", all_args));
  };
  let min_fraction = match args.get(1) {
    None => 0.5,
    Some(t) => match crate::functions::math_ast::try_eval_to_f64(t) {
      Some(v) if (0.0..=1.0).contains(&v) => v,
      _ => return Ok(unevaluated("ImageLines", all_args)),
    },
  };
  if args.len() > 2 {
    return Ok(unevaluated("ImageLines", all_args));
  }

  let w = *width as usize;
  let h = *height as usize;
  let ch = *channels as usize;
  let (wf, hf) = (w as f64, h as f64);

  // Foreground pixels in image coordinates (pixel centres, y upwards).
  let mut points: Vec<(f64, f64)> = Vec::new();
  for row in 0..h {
    for col in 0..w {
      let i = row * w + col;
      let v = if ch >= 3 {
        let b = i * ch;
        0.299 * data[b] + 0.587 * data[b + 1] + 0.114 * data[b + 2]
      } else {
        data[i * ch]
      };
      if v > 0.5 {
        points.push((col as f64 + 0.5, hf - row as f64 - 0.5));
      }
    }
  }
  if points.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }

  const THETA_STEPS: usize = 180;
  let diagonal = wf.hypot(hf);
  let rho_bins = (2.0 * diagonal).ceil() as usize + 1;
  let trig: Vec<(f64, f64)> = (0..THETA_STEPS)
    .map(|k| {
      let t = std::f64::consts::PI * k as f64 / THETA_STEPS as f64;
      (t.cos(), t.sin())
    })
    .collect();
  let mut acc = vec![0u32; THETA_STEPS * rho_bins];
  for &(x, y) in &points {
    for (k, &(c, s)) in trig.iter().enumerate() {
      let rho = x * c + y * s + diagonal;
      acc[k * rho_bins + rho.round() as usize] += 1;
    }
  }

  // Local maxima of the accumulator, strongest first.
  let at = |k: i64, r: i64| -> u32 {
    if r < 0 || r >= rho_bins as i64 {
      return 0;
    }
    let kk = k.rem_euclid(THETA_STEPS as i64) as usize;
    // Wrapping theta by pi mirrors rho.
    let rr = if k < 0 || k >= THETA_STEPS as i64 {
      rho_bins as i64 - 1 - r
    } else {
      r
    };
    acc[kk * rho_bins + rr as usize]
  };
  let mut peaks: Vec<(u32, usize, usize)> = Vec::new();
  for k in 0..THETA_STEPS {
    for r in 0..rho_bins {
      let v = acc[k * rho_bins + r];
      if v == 0 {
        continue;
      }
      let is_peak = (-2i64..=2).all(|dk| {
        (-2i64..=2).all(|dr| {
          (dk == 0 && dr == 0) || {
            let n = at(k as i64 + dk, r as i64 + dr);
            n < v || (n == v && (dk, dr) > (0, 0))
          }
        })
      });
      if is_peak {
        peaks.push((v, k, r));
      }
    }
  }
  peaks.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
  let strongest = peaks.first().map_or(0, |p| p.0) as f64;
  let cutoff = (strongest * min_fraction).max(2.0);

  let mut lines = Vec::new();
  for &(votes, k, r) in &peaks {
    if (votes as f64) < cutoff || max_features.is_some_and(|n| lines.len() >= n)
    {
      break;
    }
    let (mut c, mut s) = trig[k];
    let mut rho = r as f64 - diagonal;
    // Refine the binned estimate with a total-least-squares fit through
    // the pixels lying within a pixel of it.
    for tolerance in [1.0, 0.6, 0.6] {
      let near: Vec<(f64, f64)> = points
        .iter()
        .copied()
        .filter(|&(x, y)| (x * c + y * s - rho).abs() <= tolerance)
        .collect();
      if near.len() >= 2 {
        let n = near.len() as f64;
        let mx = near.iter().map(|p| p.0).sum::<f64>() / n;
        let my = near.iter().map(|p| p.1).sum::<f64>() / n;
        let sxx = near.iter().map(|p| (p.0 - mx).powi(2)).sum::<f64>();
        let syy = near.iter().map(|p| (p.1 - my).powi(2)).sum::<f64>();
        let sxy = near.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f64>();
        // Normal direction of the principal axis' complement.
        let angle =
          0.5 * (2.0 * sxy).atan2(sxx - syy) + std::f64::consts::FRAC_PI_2;
        let (nc, ns) = (angle.cos(), angle.sin());
        // Keep the normal on the same side as the binned one.
        let sign = if nc * c + ns * s < 0.0 { -1.0 } else { 1.0 };
        c = sign * nc;
        s = sign * ns;
        rho = mx * c + my * s;
      }
    }
    // Intersect x c + y s = rho with the border of [0, w] x [0, h].
    let mut ends: Vec<(f64, f64)> = Vec::new();
    let mut add = |p: (f64, f64)| {
      if !ends
        .iter()
        .any(|q| (q.0 - p.0).abs() < 1e-9 && (q.1 - p.1).abs() < 1e-9)
      {
        ends.push(p);
      }
    };
    let eps = 1e-9;
    if s.abs() > 1e-12 {
      for x in [0.0, wf] {
        let y = (rho - x * c) / s;
        if (-eps..=hf + eps).contains(&y) {
          add((x, y.clamp(0.0, hf)));
        }
      }
    }
    if c.abs() > 1e-12 {
      for y in [0.0, hf] {
        let x = (rho - y * s) / c;
        if (-eps..=wf + eps).contains(&x) {
          add((x.clamp(0.0, wf), y));
        }
      }
    }
    if ends.len() < 2 {
      continue;
    }
    ends.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pt =
      |p: (f64, f64)| Expr::List(vec![Expr::Real(p.0), Expr::Real(p.1)].into());
    lines.push(Expr::List(
      vec![pt(ends[0]), pt(ends[ends.len() - 1])].into(),
    ));
  }
  Ok(Expr::List(lines.into()))
}

/// The number of colour channels of a colour space, `None` when the space
/// is unknown (an `Automatic` image has no alpha channel to tell apart).
fn color_space_channels(space: Option<&str>) -> Option<u8> {
  match space? {
    "Grayscale" => Some(1),
    "CMYK" => Some(4),
    _ => Some(3),
  }
}

/// Whether the image's last channel is an alpha channel: one past what its
/// (explicit) colour space needs.
fn has_alpha_channel(channels: u8, space: Option<&str>) -> bool {
  color_space_channels(space).is_some_and(|n| channels == n + 1)
}

/// AlphaChannel[image]: the opacity of each pixel as a one-channel image
/// (all ones for an image without an alpha channel).
pub fn alpha_channel_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let Expr::Image {
    width,
    height,
    channels,
    data,
    color_space,
    ..
  } = &args[0]
  else {
    return Ok(unevaluated("AlphaChannel", args));
  };
  let ch = *channels as usize;
  let n = (*width as usize) * (*height as usize);
  let alpha: Vec<f64> = if has_alpha_channel(*channels, *color_space) {
    (0..n).map(|i| data[i * ch + ch - 1]).collect()
  } else {
    vec![1.0; n]
  };
  Ok(Expr::Image {
    width: *width,
    height: *height,
    channels: 1,
    data: std::sync::Arc::new(alpha),
    image_type: crate::syntax::ImageType::Real32,
    color_space: None,
    display_size: None,
  })
}

/// RemoveAlphaChannel[image] drops the alpha channel;
/// RemoveAlphaChannel[image, color] first composites the image over a
/// background of that colour. An image without one is returned as is.
pub fn remove_alpha_channel_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let Expr::Image {
    width,
    height,
    channels,
    data,
    color_space,
    display_size,
    ..
  } = &args[0]
  else {
    return Ok(unevaluated("RemoveAlphaChannel", args));
  };
  if !has_alpha_channel(*channels, *color_space) {
    return Ok(args[0].clone());
  }
  let ch = *channels as usize;
  let colors = ch - 1;
  let background: Option<Vec<f64>> = match args.get(1) {
    None => None,
    Some(c) => {
      let Some(col) = crate::functions::graphics::parse_color(c) else {
        return Ok(unevaluated("RemoveAlphaChannel", args));
      };
      Some(if colors == 1 {
        vec![0.299 * col.r + 0.587 * col.g + 0.114 * col.b]
      } else {
        vec![col.r, col.g, col.b]
      })
    }
  };
  let n = (*width as usize) * (*height as usize);
  let mut out = Vec::with_capacity(n * colors);
  for i in 0..n {
    let px = &data[i * ch..(i + 1) * ch];
    let a = px[ch - 1];
    for k in 0..colors {
      out.push(match &background {
        Some(bg) => px[k] * a + bg.get(k).copied().unwrap_or(0.0) * (1.0 - a),
        None => px[k],
      });
    }
  }
  Ok(Expr::Image {
    width: *width,
    height: *height,
    channels: colors as u8,
    data: std::sync::Arc::new(out),
    image_type: crate::syntax::ImageType::Real32,
    color_space: *color_space,
    display_size: *display_size,
  })
}

/// SetAlphaChannel[image] gives the image an opaque alpha channel;
/// SetAlphaChannel[image, a] one with opacity `a` (a number, or a
/// grayscale image of the same size). An existing alpha channel is replaced.
pub fn set_alpha_channel_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let Expr::Image {
    width,
    height,
    channels,
    data,
    color_space,
    display_size,
    ..
  } = &args[0]
  else {
    return Ok(unevaluated("SetAlphaChannel", args));
  };
  let ch = *channels as usize;
  let colors = if has_alpha_channel(*channels, *color_space) {
    ch - 1
  } else {
    ch
  };
  let space = color_space.or(match colors {
    1 => Some("Grayscale"),
    3 => Some("RGB"),
    _ => None,
  });
  let n = (*width as usize) * (*height as usize);
  let alpha = |i: usize| -> Option<f64> {
    match args.get(1) {
      None => Some(1.0),
      // An alpha image of another size sits centred on the image, zero
      // outside it.
      Some(Expr::Image {
        width: aw,
        height: ah,
        channels: ac,
        data: ad,
        ..
      }) => {
        let (w, h) = (*width as i64, *height as i64);
        let (aw, ah) = (*aw as i64, *ah as i64);
        // Offsets from the bottom-left corner, rounded half to even.
        let half_even = |d: i64| (d as f64 / 2.0).round_ties_even() as i64;
        let bottom = half_even(h - ah);
        let top = h - ah - bottom;
        let left = half_even(w - aw);
        let (r, c) = ((i as i64) / w, (i as i64) % w);
        let (ar, acol) = (r - top, c - left);
        Some(if (0..ah).contains(&ar) && (0..aw).contains(&acol) {
          ad[((ar * aw + acol) as usize) * (*ac as usize)]
        } else {
          0.0
        })
      }
      Some(v) => {
        // A real alpha is stored in single precision.
        crate::functions::math_ast::try_eval_to_f64(v).map(|a| a as f32 as f64)
      }
    }
  };
  let mut out = Vec::with_capacity(n * (colors + 1));
  for i in 0..n {
    out.extend_from_slice(&data[i * ch..i * ch + colors]);
    match alpha(i) {
      Some(a) => out.push(a),
      None => return Ok(unevaluated("SetAlphaChannel", args)),
    }
  }
  Ok(Expr::Image {
    width: *width,
    height: *height,
    channels: (colors + 1) as u8,
    data: std::sync::Arc::new(out),
    image_type: crate::syntax::ImageType::Real32,
    color_space: space,
    display_size: *display_size,
  })
}

/// DeleteSmallComponents[image] / DeleteSmallComponents[image, n]: the
/// image with its small morphological components set to background — those
/// of at most `n` pixels, or by default those smaller than the mean
/// component size.
fn delete_small_image_components(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let held = || Ok(unevaluated("DeleteSmallComponents", args));
  if args.is_empty() || args.len() > 2 {
    return held();
  }
  let limit = match args.get(1) {
    None => None,
    Some(Expr::Integer(n)) if *n >= 0 => Some(*n as usize),
    Some(_) => return held(),
  };
  let labels_expr = crate::evaluator::evaluate_expr_to_expr(&call(
    "MorphologicalComponents",
    vec![args[0].clone()],
  ))?;
  let Expr::List(rows) = &labels_expr else {
    return held();
  };
  let labels: Vec<Vec<i128>> = match rows
    .iter()
    .map(|r| match r {
      Expr::List(cs) => cs
        .iter()
        .map(|c| match c {
          Expr::Integer(n) => Some(*n),
          _ => None,
        })
        .collect::<Option<Vec<_>>>(),
      _ => None,
    })
    .collect::<Option<Vec<_>>>()
  {
    Some(l) => l,
    None => return held(),
  };
  let mut counts: std::collections::BTreeMap<i128, usize> =
    std::collections::BTreeMap::new();
  for &l in labels.iter().flatten() {
    if l != 0 {
      *counts.entry(l).or_default() += 1;
    }
  }
  if counts.is_empty() {
    return Ok(args[0].clone());
  }
  let mean = counts.values().sum::<usize>() as f64 / counts.len() as f64;
  let removed = |l: i128| {
    l != 0
      && match limit {
        Some(n) => counts[&l] <= n,
        None => (counts[&l] as f64) < mean,
      }
  };
  match &args[0] {
    Expr::Image {
      color_space,
      display_size,
      width,
      height,
      channels,
      data,
      image_type,
    } => {
      let (w, ch) = (*width as usize, *channels as usize);
      let mut out = data.as_ref().clone();
      for (r, row) in labels.iter().enumerate() {
        for (c, &l) in row.iter().enumerate() {
          if removed(l) {
            let base = (r * w + c) * ch;
            for v in &mut out[base..base + ch] {
              *v = 0.0;
            }
          }
        }
      }
      Ok(Expr::Image {
        color_space: *color_space,
        display_size: *display_size,
        width: *width,
        height: *height,
        channels: *channels,
        data: Arc::new(out),
        image_type: *image_type,
      })
    }
    _ => held(),
  }
}
