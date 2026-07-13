#[allow(unused_imports)]
use super::*;

/// Colors of a Wolfram `ColorData` indexed scheme (`ColorData[n, "ColorList"]`).
///
/// Scheme 97 is the default plot-color palette; its exact RGB values are
/// reproduced here (verified against wolframscript's output). Other indexed
/// schemes are not yet tabulated.
fn indexed_color_list(n: i128) -> Option<&'static [(f64, f64, f64)]> {
  // ColorData[97, "ColorList"] — the modern default plot palette (10 colors).
  const SCHEME_97: [(f64, f64, f64); 10] = [
    (0.368417, 0.506779, 0.709798),
    (0.880722, 0.611041, 0.142051),
    (0.560181, 0.691569, 0.194885),
    (0.922526, 0.385626, 0.209179),
    (0.528488, 0.470624, 0.701351),
    (0.772079, 0.431554, 0.102387),
    (0.363898, 0.618501, 0.782349),
    (1.0, 0.75, 0.0),
    (0.647624, 0.37816, 0.614037),
    (0.571589, 0.586483, 0.0),
  ];
  match n {
    97 => Some(&SCHEME_97),
    _ => None,
  }
}

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

/// Resolve `Import`'s first argument to a path string: a plain string or a
/// `URL["…"]` wrapper (which wolframscript accepts interchangeably).
fn import_path_spec(expr: &Expr) -> Option<String> {
  match expr {
    Expr::String(s) => Some(s.clone()),
    Expr::FunctionCall { name, args } if name == "URL" && args.len() == 1 => {
      match &args[0] {
        Expr::String(s) => Some(s.clone()),
        _ => None,
      }
    }
    _ => None,
  }
}

/// Match Import's second argument against the row/column subset shape
/// `{"Data", rowspec}` or `{"Data", rowspec, colspec}` (specs being
/// integers, spans, or All). Returns the row spec and optional column spec.
fn import_data_spec_args(elem: &Expr) -> Option<(&Expr, Option<&Expr>)> {
  let Expr::List(items) = elem else {
    return None;
  };
  if !matches!(items.first(), Some(Expr::String(s)) if s == "Data") {
    return None;
  }
  let is_spec = crate::functions::csv_ast::is_position_spec;
  match items.len() {
    2 if is_spec(&items[1]) => Some((&items[1], None)),
    3 if is_spec(&items[1]) && is_spec(&items[2]) => {
      Some((&items[1], Some(&items[2])))
    }
    _ => None,
  }
}

/// Import an SVG file (local or URL) as a `Graphics` expression, matching
/// wolframscript's vector import (which prints as `-Graphics-` in the CLI
/// and renders as the SVG itself in visual hosts).
#[cfg(not(target_arch = "wasm32"))]
fn import_svg(path: &str, is_url: bool) -> Result<Expr, InterpreterError> {
  if !is_url && !std::path::Path::new(path).exists() {
    crate::emit_message(&format!(
      "Import::nffil: File {} not found during Import.",
      path
    ));
    return Ok(Expr::Identifier("$Failed".to_string()));
  }
  let svg = import_read_text(path, is_url)?;
  Ok(crate::graphics_result(svg))
}

/// Read the textual contents of a path that is either a local file or an
/// `http(s)://` URL. Used by formats (JSON/RawJSON) that parse text.
#[cfg(not(target_arch = "wasm32"))]
fn import_read_text(
  path: &str,
  is_url: bool,
) -> Result<String, InterpreterError> {
  if is_url {
    crate::functions::xlsx_ast::download_url(path)
      .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
  } else {
    std::fs::read_to_string(path).map_err(|e| {
      InterpreterError::EvaluationError(format!(
        "Import: cannot open \"{}\": {}",
        path, e
      ))
    })
  }
}

/// Import a file (local or URL) as JSON/RawJSON, parsing it into the same
/// nested structure that `ImportString[..., "JSON"|"RawJSON"]` produces.
#[cfg(not(target_arch = "wasm32"))]
fn import_json(
  path: &str,
  is_url: bool,
  raw: bool,
) -> Result<Expr, InterpreterError> {
  let content = import_read_text(path, is_url)?;
  Ok(match serde_json::from_str::<serde_json::Value>(&content) {
    Ok(value) => json_value_to_expr(&value, raw),
    Err(_) => Expr::Identifier("$Failed".to_string()),
  })
}

/// Import a Netpbm (PPM / PGM / PBM / PNM) file.
///
/// Matches wolframscript's behaviour for the two common failure modes:
/// emits `Import::nffil` when the path doesn't exist and `Import::fmterr`
/// when the file is present but its contents aren't a recognisable Netpbm
/// stream. Both cases return `$Failed`. The `image` crate isn't built
/// with the `pnm` feature, so even well-formed PPMs currently return
/// `$Failed` after the magic-byte check passes — but without the error
/// message, mirroring wolframscript's silent success on a valid file.
#[cfg(not(target_arch = "wasm32"))]
fn import_netpbm(path: &str) -> Result<Expr, InterpreterError> {
  if !std::path::Path::new(path).exists() {
    // wolframscript prints Import::nffil to stdout for a missing file.
    crate::emit_message_to_stdout(&format!(
      "Import::nffil: File {} not found during Import.",
      path
    ));
    return Ok(Expr::Identifier("$Failed".to_string()));
  }
  let bytes = match std::fs::read(path) {
    Ok(b) => b,
    Err(_) => {
      return Ok(Expr::Identifier("$Failed".to_string()));
    }
  };
  // A valid Netpbm stream starts with `P1`..`P6` followed by whitespace.
  let valid_magic = bytes.len() >= 3
    && bytes[0] == b'P'
    && matches!(bytes[1], b'1'..=b'7')
    && matches!(bytes[2], b' ' | b'\t' | b'\n' | b'\r');
  if !valid_magic {
    // wolframscript prints Import::fmterr to stdout for a malformed file.
    crate::emit_message_to_stdout(
      "Import::fmterr: Cannot import data as PPM format.",
    );
    return Ok(Expr::Identifier("$Failed".to_string()));
  }
  // Magic looks plausible but Woxi doesn't yet parse Netpbm pixel data.
  // Return $Failed silently — matches wolframscript on a valid file when
  // parsing succeeds (no message; wolframscript returns `Image[…]`).
  Ok(Expr::Identifier("$Failed".to_string()))
}

/// Import a host-registered virtual file (WASM). The browser host registers
/// file contents via `crate::wasm::set_virtual_file` (e.g. chat attachments);
/// all non-URL `Import` calls resolve against that store since there is no
/// local filesystem in the browser.
#[cfg(target_arch = "wasm32")]
fn import_virtual(
  path: &str,
  element: Option<&str>,
) -> Result<Expr, InterpreterError> {
  let Some(bytes) = crate::wasm::virtual_file(path) else {
    return Err(InterpreterError::EvaluationError(format!(
      "Import: cannot open \"{}\" in the browser. Only files attached to the conversation can be imported.",
      path
    )));
  };
  let ext = import_extension(path);

  if matches!(
    ext.as_str(),
    "jpg" | "jpeg" | "png" | "gif" | "bmp" | "tiff" | "tif" | "webp"
  ) {
    return crate::functions::image_ast::import_image_from_bytes(&bytes);
  }

  // CERN ROOT files: binary format, decoded straight from the bytes.
  if ext == "root" || matches!(element, Some("ROOT")) {
    return crate::functions::root_ast::root_import_bytes(&bytes);
  }

  let content = String::from_utf8(bytes).map_err(|_| {
    InterpreterError::EvaluationError(format!(
      "Import: \"{}\" is not valid UTF-8 text",
      path
    ))
  })?;

  // JSON via explicit format element or .json extension.
  if matches!(element, Some("JSON") | Some("RawJSON")) || ext == "json" {
    let raw = matches!(element, Some("RawJSON"));
    return Ok(match serde_json::from_str::<serde_json::Value>(&content) {
      Ok(value) => json_value_to_expr(&value, raw),
      Err(_) => Expr::Identifier("$Failed".to_string()),
    });
  }

  if ext == "svg" {
    return Ok(crate::graphics_result(content));
  }

  if ext == "csv" || ext == "tsv" {
    let rows: Vec<Vec<String>> = if ext == "tsv" {
      crate::functions::csv_ast::parse_tsv(&content)
    } else {
      crate::functions::csv_ast::parse_csv(&content)
    };
    return Ok(crate::functions::csv_ast::csv_import_element(
      &rows, element,
    ));
  }

  // Everything else is treated as plain text (matches the native txt path).
  match element {
    None => Ok(Expr::String(
      crate::functions::txt_ast::strip_trailing_newline(content),
    )),
    Some(elem) => crate::functions::txt_ast::import_element(&content, elem)
      .ok_or_else(|| {
        InterpreterError::EvaluationError(format!(
          "Import: unsupported element \"{}\" for \"{}\"",
          elem, path
        ))
      }),
  }
}

/// Convert a parsed JSON value to a Woxi expression, matching
/// `ImportString[…, "JSON"]`: arrays become lists, scalars/true/false/null map
/// to atoms. A JSON object becomes a list of `key -> value` rules for the
/// "JSON" format, or an `Association` for "RawJSON" (`raw == true`); both keep
/// the source key order.
fn json_value_to_expr(value: &serde_json::Value, raw: bool) -> Expr {
  use serde_json::Value;
  match value {
    Value::Null => Expr::Identifier("Null".to_string()),
    Value::Bool(true) => Expr::Identifier("True".to_string()),
    Value::Bool(false) => Expr::Identifier("False".to_string()),
    Value::Number(n) => {
      if let Some(i) = n.as_i64() {
        Expr::Integer(i as i128)
      } else if let Some(u) = n.as_u64() {
        Expr::Integer(u as i128)
      } else {
        Expr::Real(n.as_f64().unwrap_or(0.0))
      }
    }
    Value::String(s) => Expr::String(s.clone()),
    Value::Array(items) => {
      Expr::List(items.iter().map(|v| json_value_to_expr(v, raw)).collect())
    }
    Value::Object(map) => {
      if raw {
        Expr::Association(
          map
            .iter()
            .map(|(k, v)| (Expr::String(k.clone()), json_value_to_expr(v, raw)))
            .collect(),
        )
      } else {
        Expr::List(
          map
            .iter()
            .map(|(k, v)| Expr::Rule {
              pattern: Box::new(Expr::String(k.clone())),
              replacement: Box::new(json_value_to_expr(v, raw)),
            })
            .collect(),
        )
      }
    }
  }
}

pub fn dispatch_image_functions(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "CrossingDetect" => {
      return Some(crate::functions::image_ast::crossing_detect_ast(args));
    }
    "Image" if !args.is_empty() => {
      // Invalid image data shouldn't error — return unevaluated so wrapping
      // predicates like ImageQ can still classify it as False.
      let result = crate::functions::image_ast::image_constructor_ast(args);
      return Some(match result {
        Ok(expr) => Ok(expr),
        Err(_) => Ok(Expr::FunctionCall {
          name: "Image".to_string(),
          args: args.to_vec().into(),
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
    "PixelValuePositions" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::image_ast::pixel_value_positions_ast(
        args,
      ));
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
    "Thumbnail" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::image_ast::thumbnail_ast(args));
    }
    "ImageAdjust" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::image_ast::image_adjust_ast(args));
    }
    "ImageReflect" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::image_ast::image_reflect_ast(args));
    }
    "ImageRotate" if !args.is_empty() => {
      return Some(crate::functions::image_ast::image_rotate_ast(args));
    }
    "ImageResize" if args.len() >= 2 => {
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
    "ImagePartition" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::image_ast::image_partition_ast(args));
    }
    "Threshold" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::image_ast::threshold_ast(args));
    }
    "ColorQuantize" if args.len() == 2 => {
      return Some(crate::functions::image_ast::color_quantize_ast(args));
    }
    "ColorCombine" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::image_ast::color_combine_ast(args));
    }
    "ColorSeparate" if args.len() == 1 => {
      return Some(crate::functions::image_ast::color_separate_ast(args));
    }
    "Colorize" if !args.is_empty() => {
      return Some(crate::functions::image_ast::colorize_ast(args));
    }
    "GaussianFilter" if args.len() == 2 => {
      return Some(crate::functions::image_ast::gaussian_filter_ast(args));
    }
    "GaussianMatrix" if args.len() == 1 => {
      return Some(crate::functions::image_ast::gaussian_matrix_ast(args));
    }
    "ImageConvolve" if args.len() == 2 => {
      return Some(crate::functions::image_ast::image_convolve_ast(args));
    }
    "MedianFilter" if args.len() == 2 => {
      return Some(crate::functions::image_ast::median_filter_ast(args));
    }
    "MeanFilter" if args.len() == 2 => {
      return Some(crate::functions::image_ast::mean_filter_ast(args));
    }
    "StandardDeviationFilter" if args.len() == 2 => {
      return Some(crate::functions::image_ast::standard_deviation_filter_ast(
        args,
      ));
    }
    "GradientFilter" if args.len() == 2 => {
      return Some(crate::functions::image_ast::gradient_filter_ast(args));
    }
    "TextRecognize" if !args.is_empty() => {
      return Some(crate::functions::image_ast::text_recognize_ast(args));
    }
    "PixelValue" if args.len() == 2 => {
      return Some(crate::functions::image_ast::pixel_value_ast(args));
    }
    "BinaryImageQ" if args.len() == 1 => {
      return Some(crate::functions::image_ast::binary_image_q_ast(args));
    }
    "EdgeDetect" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::image_ast::edge_detect_ast(args));
    }
    "DominantColors" if !args.is_empty() => {
      // Dispatch every non-empty call: dominant_colors_ast itself checks
      // for non-Image first arg (emits DominantColors::imginv) before
      // validating arg count, matching wolframscript when an upstream
      // Import has returned $Failed.
      return Some(crate::functions::image_ast::dominant_colors_ast(args));
    }
    "ImageApply" if args.len() == 2 => {
      return Some(crate::functions::image_ast::image_apply_ast(
        args,
        &evaluate_expr_to_expr,
      ));
    }
    "ColorConvert" if args.len() == 2 => {
      let is_color_directive = matches!(
        &args[0],
        Expr::FunctionCall { name: n, .. }
          if n == "RGBColor" || n == "GrayLevel"
            || n == "Hue" || n == "CMYKColor"
      );
      if matches!(&args[0], Expr::Image { .. }) || is_color_directive {
        return Some(crate::functions::image_ast::color_convert_ast(args));
      }
    }
    "ColorDistance" if args.len() >= 2 => {
      return Some(crate::functions::image_ast::color_distance_ast(args));
    }
    // ColorData[]: list of available data categories.
    "ColorData" if args.is_empty() => {
      return Some(Ok(Expr::List(
        vec![
          Expr::String("Gradients".to_string()),
          Expr::String("Indexed".to_string()),
          Expr::String("Named".to_string()),
          Expr::String("Physical".to_string()),
        ]
        .into(),
      )));
    }
    // ColorData[n, "ColorList"]: the ordered list of colors in indexed
    // scheme n (e.g. `ColorData[97, "ColorList"]` — the default plot palette).
    "ColorData" if args.len() == 2 => {
      if let (Expr::Integer(n), Expr::String(prop)) = (&args[0], &args[1])
        && prop == "ColorList"
        && let Some(colors) = indexed_color_list(*n)
      {
        return Some(Ok(Expr::List(
          colors
            .iter()
            .map(|&(r, g, b)| Expr::FunctionCall {
              name: "RGBColor".to_string(),
              args: vec![Expr::Real(r), Expr::Real(g), Expr::Real(b)].into(),
            })
            .collect(),
        )));
      }
    }
    // ColorData["Gradients"]: list of named built-in color gradients.
    "ColorData" if args.len() == 1 => {
      if let Expr::String(s) = &args[0]
        && s == "Gradients"
      {
        let names = [
          "AlpineColors",
          "Aquamarine",
          "ArmyColors",
          "AtlanticColors",
          "AuroraColors",
          "AvocadoColors",
          "BeachColors",
          "BlueGreenYellow",
          "BrassTones",
          "BrightBands",
          "BrownCyanTones",
          "CandyColors",
          "CherryTones",
          "CMYKColors",
          "CoffeeTones",
          "DarkBands",
          "DarkRainbow",
          "DarkTerrain",
          "DeepSeaColors",
          "FallColors",
          "FruitPunchColors",
          "FuchsiaTones",
          "GrayTones",
          "GrayYellowTones",
          "GreenBrownTerrain",
          "GreenPinkTones",
          "IslandColors",
          "LakeColors",
          "LightTemperatureMap",
          "LightTerrain",
          "MintColors",
          "NeonColors",
          "Pastel",
          "PearlColors",
          "PigeonTones",
          "PlumColors",
          "Rainbow",
          "RedBlueTones",
          "RedGreenSplit",
          "RoseColors",
          "RustTones",
          "SandyTerrain",
          "SiennaTones",
          "SolarColors",
          "SouthwestColors",
          "StarryNightColors",
          "SunsetColors",
          "TemperatureMap",
          "ThermometerColors",
          "ValentineTones",
          "WatermelonColors",
        ];
        return Some(Ok(Expr::List(
          names
            .iter()
            .map(|n| Expr::String((*n).to_string()))
            .collect(),
        )));
      }
    }
    "ImageCompose" if args.len() == 2 => {
      return Some(crate::functions::image_ast::image_compose_ast(args));
    }
    "ImageAdd" if args.len() >= 2 => {
      return Some(crate::functions::image_ast::image_add_ast(args));
    }
    "ImageSubtract" if args.len() >= 2 => {
      return Some(crate::functions::image_ast::image_subtract_ast(args));
    }
    "ImageMultiply" if args.len() >= 2 => {
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
      let path = match import_path_spec(&args[0]) {
        Some(p) => p,
        None => {
          return Some(Ok(Expr::FunctionCall {
            name: "Import".to_string(),
            args: args.to_vec().into(),
          }));
        }
      };
      let is_url = path.starts_with("http://") || path.starts_with("https://");
      let ext = import_extension(&path);

      // In the browser every non-URL path is served from the
      // host-registered virtual file store.
      #[cfg(target_arch = "wasm32")]
      if !is_url {
        return Some(import_virtual(&path, None));
      }

      if ext == "svg" {
        #[cfg(not(target_arch = "wasm32"))]
        {
          return Some(import_svg(&path, is_url));
        }
        #[cfg(target_arch = "wasm32")]
        {
          return Some(Err(InterpreterError::EvaluationError(
            "Import: SVG import is not available in the browser".into(),
          )));
        }
      }

      if ext == "json" {
        #[cfg(not(target_arch = "wasm32"))]
        {
          return Some(import_json(&path, is_url, false));
        }
        #[cfg(target_arch = "wasm32")]
        {
          return Some(Err(InterpreterError::EvaluationError(
            "Import: JSON import is not available in the browser".into(),
          )));
        }
      }

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
          if ext == "root" {
            return Some(
              crate::functions::xlsx_ast::download_url(&path).and_then(
                |bytes| crate::functions::root_ast::root_import_bytes(&bytes),
              ),
            );
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
          "ppm" | "pgm" | "pbm" | "pnm" => {
            return Some(import_netpbm(&path));
          }
          "root" => {
            return Some(crate::functions::root_ast::root_import_file(&path));
          }
          "wav" | "wave" | "flac" | "mp3" | "ogg" | "oga" | "opus" | "m4a"
          | "aac" | "aif" | "aiff" => {
            return Some(crate::functions::audio_ast::import_audio_file(&path));
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
      let path = match import_path_spec(&args[0]) {
        Some(p) => p,
        None => {
          return Some(Ok(Expr::FunctionCall {
            name: "Import".to_string(),
            args: args.to_vec().into(),
          }));
        }
      };
      let is_url = path.starts_with("http://") || path.starts_with("https://");
      let ext = import_extension(&path);

      // In the browser every non-URL path is served from the
      // host-registered virtual file store.
      #[cfg(target_arch = "wasm32")]
      if !is_url {
        if let Expr::String(elem) = &args[1] {
          return Some(import_virtual(&path, Some(elem)));
        }
        // Row/column subsets: Import[f, {"Data", rowspec(, colspec)}].
        if matches!(ext.as_str(), "csv" | "tsv")
          && let Some((row_spec, col_spec)) = import_data_spec_args(&args[1])
        {
          let Some(bytes) = crate::wasm::virtual_file(&path) else {
            return Some(Err(InterpreterError::EvaluationError(format!(
              "Import: cannot open \"{}\" in the browser. Only files attached to the conversation can be imported.",
              path
            ))));
          };
          let content = match String::from_utf8(bytes) {
            Ok(c) => c,
            Err(_) => {
              return Some(Err(InterpreterError::EvaluationError(format!(
                "Import: \"{}\" is not valid UTF-8 text",
                path
              ))));
            }
          };
          let rows = if ext == "tsv" {
            crate::functions::csv_ast::parse_tsv(&content)
          } else {
            crate::functions::csv_ast::parse_csv(&content)
          };
          return Some(crate::functions::csv_ast::csv_import_data_spec(
            &rows, row_spec, col_spec,
          ));
        }
        // ROOT element paths resolve against the virtual file store.
        if let Expr::List(items) = &args[1] {
          let is_root_fmt =
            matches!(items.first(), Some(Expr::String(s)) if s == "ROOT");
          if is_root_fmt || ext == "root" {
            let elems: &[Expr] = if is_root_fmt { &items[1..] } else { items };
            let Some(bytes) = crate::wasm::virtual_file(&path) else {
              return Some(Err(InterpreterError::EvaluationError(format!(
                "Import: cannot open \"{}\" in the browser. Only files attached to the conversation can be imported.",
                path
              ))));
            };
            return Some(
              crate::functions::root_ast::root_import_bytes_element(
                &bytes, elems,
              ),
            );
          }
        }
        return Some(Ok(Expr::FunctionCall {
          name: "Import".to_string(),
          args: args.to_vec().into(),
        }));
      }

      if let Expr::String(fmt) = &args[1]
        && (fmt == "JSON" || fmt == "RawJSON")
      {
        #[cfg(not(target_arch = "wasm32"))]
        {
          return Some(import_json(&path, is_url, fmt == "RawJSON"));
        }
        #[cfg(target_arch = "wasm32")]
        {
          return Some(Err(InterpreterError::EvaluationError(
            "Import: JSON import is not available in the browser".into(),
          )));
        }
      }

      // Explicit "ROOT" format: parse as a CERN ROOT file regardless of
      // the extension.
      if let Expr::String(fmt) = &args[1]
        && fmt == "ROOT"
      {
        #[cfg(not(target_arch = "wasm32"))]
        {
          if is_url {
            return Some(
              crate::functions::xlsx_ast::download_url(&path).and_then(
                |bytes| crate::functions::root_ast::root_import_bytes(&bytes),
              ),
            );
          }
          return Some(crate::functions::root_ast::root_import_file(&path));
        }
        #[cfg(target_arch = "wasm32")]
        {
          return Some(Err(InterpreterError::EvaluationError(
            "Import: ROOT import from a path is not available in the browser"
              .into(),
          )));
        }
      }

      // ROOT element paths: Import[file, {"ROOT", "dir/obj", …}] descends
      // into the file; for a .root extension the "ROOT" marker is optional.
      if let Expr::List(items) = &args[1] {
        let is_root_fmt =
          matches!(items.first(), Some(Expr::String(s)) if s == "ROOT");
        if is_root_fmt || ext == "root" {
          #[cfg(not(target_arch = "wasm32"))]
          {
            let elems: &[Expr] = if is_root_fmt { &items[1..] } else { items };
            if is_url {
              return Some(
                crate::functions::xlsx_ast::download_url(&path).and_then(
                  |bytes| {
                    crate::functions::root_ast::root_import_bytes_element(
                      &bytes, elems,
                    )
                  },
                ),
              );
            }
            return Some(crate::functions::root_ast::root_import_file_element(
              &path, elems,
            ));
          }
          #[cfg(target_arch = "wasm32")]
          {
            return Some(Err(InterpreterError::EvaluationError(
              "Import: ROOT import from a path is not available in the browser"
                .into(),
            )));
          }
        }
      }

      if ext == "csv" {
        // Row/column subsets: Import[f, {"Data", rowspec(, colspec)}].
        #[cfg(not(target_arch = "wasm32"))]
        if !is_url
          && let Some((row_spec, col_spec)) = import_data_spec_args(&args[1])
        {
          return Some(crate::functions::csv_ast::csv_import_file_data_spec(
            &path, row_spec, col_spec,
          ));
        }
        let element = match &args[1] {
          Expr::String(e) => e.clone(),
          _ => {
            return Some(Ok(Expr::FunctionCall {
              name: "Import".to_string(),
              args: args.to_vec().into(),
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
              args: args.to_vec().into(),
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

      // Netpbm family: dispatch via either the file extension or an
      // explicit `"PPM" | "PGM" | "PBM" | "PNM"` format argument so that
      // `Import["file.ppm","PPM"]` works even when the extension was
      // omitted or mis-cased.
      let explicit_fmt = if let Expr::String(s) = &args[1] {
        Some(s.to_ascii_lowercase())
      } else {
        None
      };
      if matches!(ext.as_str(), "ppm" | "pgm" | "pbm" | "pnm")
        || matches!(
          explicit_fmt.as_deref(),
          Some("ppm") | Some("pgm") | Some("pbm") | Some("pnm")
        )
      {
        #[cfg(not(target_arch = "wasm32"))]
        {
          if is_url {
            // URL imports for netpbm aren't supported yet; fall through.
          } else {
            return Some(import_netpbm(&path));
          }
        }
        #[cfg(target_arch = "wasm32")]
        {
          return Some(Err(InterpreterError::EvaluationError(
            "Import: local file access is not available in the browser".into(),
          )));
        }
      }

      // Fall through for unsupported formats
      return Some(Ok(Expr::FunctionCall {
        name: "Import".to_string(),
        args: args.to_vec().into(),
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
            args: args.to_vec().into(),
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
              args: args.to_vec().into(),
            }));
          }
        }
      } else {
        "CSV"
      };

      // Plain-text format elements that don't depend on a parser.
      // `Elements` returns the static list of supported plain-text
      // element names so callers can probe what's available.
      if format == "Elements" {
        let names =
          ["Data", "Lines", "Plaintext", "String", "Summary", "Words"];
        return Some(Ok(Expr::List(
          names
            .iter()
            .map(|n| Expr::String((*n).to_string()))
            .collect(),
        )));
      }
      // `Lines` splits the input at every `\n`. A single trailing newline
      // is dropped (matches wolframscript: `"a\nb\n"` → `{"a","b"}`),
      // but interior blank lines are preserved.
      if format == "Lines" {
        let trimmed = content.strip_suffix('\n').unwrap_or(&content);
        let items: Vec<Expr> = if trimmed.is_empty() {
          Vec::new()
        } else {
          trimmed
            .split('\n')
            .map(|s| Expr::String(s.to_string()))
            .collect()
        };
        return Some(Ok(Expr::List(items.into())));
      }
      // `String` / `Plaintext` / `Text` return the input verbatim.
      if format == "String" || format == "Plaintext" || format == "Text" {
        return Some(Ok(Expr::String(content)));
      }
      // `JSON` parses the string into nested lists: a JSON array becomes a
      // List, a JSON object becomes a list of `key -> value` rules (string
      // keys, insertion order preserved), and scalars/true/false/null map to
      // the corresponding atoms. Invalid JSON yields $Failed.
      if format == "JSON" || format == "RawJSON" {
        let raw = format == "RawJSON";
        return Some(Ok(
          match serde_json::from_str::<serde_json::Value>(&content) {
            Ok(value) => json_value_to_expr(&value, raw),
            Err(_) => Expr::Identifier("$Failed".to_string()),
          },
        ));
      }
      // `TSV` is tab-separated; `Table` splits each line on runs of
      // whitespace. Both reuse the CSV element machinery (number auto-typing,
      // {{…}, …} shape) after splitting into string rows.
      if format == "TSV" || format == "Table" {
        let trimmed = content.strip_suffix('\n').unwrap_or(&content);
        let rows: Vec<Vec<String>> = if trimmed.is_empty() {
          Vec::new()
        } else {
          trimmed
            .split('\n')
            .map(|line| {
              if format == "TSV" {
                line.split('\t').map(|s| s.to_string()).collect()
              } else {
                line.split_whitespace().map(|s| s.to_string()).collect()
              }
            })
            .collect()
        };
        return Some(Ok(crate::functions::csv_ast::csv_import_element(
          &rows, None,
        )));
      }
      // `Words` splits on ASCII whitespace and drops empty fragments.
      if format == "Words" {
        let items: Vec<Expr> = content
          .split_whitespace()
          .map(|s| Expr::String(s.to_string()))
          .collect();
        return Some(Ok(Expr::List(items.into())));
      }

      if format != "CSV" {
        return Some(Ok(Expr::FunctionCall {
          name: "ImportString".to_string(),
          args: args.to_vec().into(),
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
