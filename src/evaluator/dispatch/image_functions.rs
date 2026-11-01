use super::*;

/// Wolfram's own definition of every indexed color scheme, one per line:
/// the scheme number, its `"Range"` (`{1, n, 1}` for a palette that cycles,
/// `{1, Infinity, 1}` for one that keeps generating colors), and its color
/// function — exactly what `ColorData[n]` holds in wolframscript.
static INDEXED_SCHEMES_TSV: &str =
  include_str!("../../../resources/indexed_color_schemes.tsv");

thread_local! {
  static INDEXED_SCHEMES: std::cell::RefCell<
    Option<std::collections::HashMap<i128, (Expr, Expr)>>,
  > = const { std::cell::RefCell::new(None) };
}

/// The `(range, color function)` of indexed scheme `n`, parsed on first use.
fn indexed_scheme(n: i128) -> Option<(Expr, Expr)> {
  INDEXED_SCHEMES.with(|cell| {
    let mut cache = cell.borrow_mut();
    let table = cache.get_or_insert_with(|| {
      INDEXED_SCHEMES_TSV
        .lines()
        .filter_map(|line| {
          let mut fields = line.splitn(3, '\t');
          let n = fields.next()?.parse::<i128>().ok()?;
          let range = crate::syntax::string_to_expr(fields.next()?).ok()?;
          let body = crate::syntax::string_to_expr(fields.next()?).ok()?;
          Some((n, (range, body)))
        })
        .collect()
    });
    table.get(&n).cloned()
  })
}

/// `ColorData[n]`: `ColorDataFunction[n, "Indexed", range, function]`.
fn indexed_color_function(n: i128) -> Option<Expr> {
  let (range, body) = indexed_scheme(n)?;
  Some(call(
    "ColorDataFunction",
    vec![Expr::Integer(n), str_expr("Indexed"), range, body],
  ))
}

/// `ColorData[n, k]`: the scheme's color function applied to `k`.
fn indexed_scheme_color(n: i128, k: &Expr) -> Option<Expr> {
  let (_, body) = indexed_scheme(n)?;
  crate::evaluator::function_application::apply_function_to_arg(&body, k).ok()
}

/// `ColorData[n, "ColorList"]`: one full cycle of a cycling palette, or the
/// first 15 colors of a generated one.
fn indexed_color_list(n: i128) -> Option<Expr> {
  let (range, _) = indexed_scheme(n)?;
  let len = match &range {
    Expr::List(items) => match items.get(1) {
      Some(Expr::Integer(len)) => *len,
      _ => 15,
    },
    _ => return None,
  };
  (1..=len)
    .map(|k| indexed_scheme_color(n, &Expr::Integer(k)))
    .collect::<Option<Vec<_>>>()
    .map(|colors| Expr::List(colors.into()))
}

/// The color function of a named gradient, echoed in wolframscript's
/// structured form
///   `ColorDataFunction[scheme, "Gradients", {0, 1}, Blend[scheme, #1] &]`.
/// Both `ColorData["scheme"]` and `ColorData["scheme", "ColorFunction"]`
/// give it; applying it blends the stored control points at the parameter
/// (see `apply_curried_call`).
fn gradient_color_function(scheme: &str) -> Expr {
  let blend = Expr::Function {
    body: Box::new(call("Blend", vec![str_expr(scheme), Expr::Slot(1)])),
  };
  call(
    "ColorDataFunction",
    vec![
      str_expr(scheme),
      str_expr("Gradients"),
      Expr::List(vec![Expr::Integer(0), Expr::Integer(1)].into()),
      blend,
    ],
  )
}

/// The color function of a reversed named gradient,
/// `ColorData[{scheme, "Reverse"}]`:
///   `ColorDataFunction[scheme, "Gradients", {0, 1},
///     Blend[scheme, 1 - #1] &]`.
fn reversed_gradient_color_function(scheme: &str) -> Expr {
  let blend = Expr::Function {
    body: Box::new(call(
      "Blend",
      vec![
        str_expr(scheme),
        call(
          "Plus",
          vec![
            Expr::Integer(1),
            call("Times", vec![Expr::Integer(-1), Expr::Slot(1)]),
          ],
        ),
      ],
    )),
  };
  call(
    "ColorDataFunction",
    vec![
      str_expr(scheme),
      str_expr("Gradients"),
      Expr::List(vec![Expr::Integer(0), Expr::Integer(1)].into()),
      blend,
    ],
  )
}

/// The color function of a named gradient stretched over a custom range,
/// `ColorData[{scheme, {min, max}}]`:
///   `ColorDataFunction[scheme, "Gradients", {min, max},
///     Blend[scheme, Rescale[#1, {min, max}]] &]`.
fn ranged_gradient_color_function(scheme: &str, range: &Expr) -> Expr {
  let blend = Expr::Function {
    body: Box::new(call(
      "Blend",
      vec![
        str_expr(scheme),
        call("Rescale", vec![Expr::Slot(1), range.clone()]),
      ],
    )),
  };
  call(
    "ColorDataFunction",
    vec![
      str_expr(scheme),
      str_expr("Gradients"),
      range.clone(),
      blend,
    ],
  )
}

/// A horizontal gradient strip — what `ColorData[name, "Image"]` returns,
/// shown as the swatch in gradient pickers (e.g. a Manipulate color-scheme
/// dropdown). Despite the property name, wolframscript hands back a
/// `Graphics[{Raster[…]}, …]` rather than an `Image`: a single row of 101
/// samples over the unit square, drawn at a 1/8 aspect ratio.
fn gradient_strip_image(controls: &[(f64, f64, f64)]) -> Expr {
  const SAMPLES: usize = 101;
  let row: Vec<Expr> = (0..SAMPLES)
    .map(|x| {
      let (r, g, b) = crate::functions::chart::gradient_color_at(
        controls,
        x as f64 / (SAMPLES - 1) as f64,
      );
      Expr::List(vec![Expr::Real(r), Expr::Real(g), Expr::Real(b)].into())
    })
    .collect();
  let unit_corner = |a: i128, b: i128| {
    Expr::List(vec![Expr::Integer(a), Expr::Integer(b)].into())
  };
  let raster = call(
    "Raster",
    vec![
      Expr::List(vec![Expr::List(row.into())].into()),
      Expr::List(vec![unit_corner(0, 0), unit_corner(1, 1)].into()),
    ],
  );
  let option = |name: &str, value: Expr| rule_expr(id_expr(name), value);
  call(
    "Graphics",
    vec![
      Expr::List(vec![raster].into()),
      Expr::List(
        vec![
          option("ImageSize", Expr::Integer(250)),
          option("ContentSelectable", bool_expr(false)),
          option("AspectRatio", div2(Expr::Integer(1), Expr::Integer(8))),
          option(
            "PlotRange",
            Expr::List(vec![unit_corner(0, 1), unit_corner(0, 1)].into()),
          ),
        ]
        .into(),
      ),
    ],
  )
}

/// Wolfram's "VisibleSpectrum" gradient: piecewise linear between these
/// (wavelength in nm, r, g, b) knots, black outside 380–750 nm. The knots
/// were read off wolframscript's `ColorData["VisibleSpectrum", w]`.
const VISIBLE_SPECTRUM_KNOTS: [(f64, f64, f64, f64); 25] = [
  (380.0, 0.0, 0.0, 0.0),
  (430.0, 0.4744513369704753, 0.0, 1.0),
  (440.0, 0.4812997627779147, 0.0, 1.0),
  (450.0, 0.3852118779771466, 0.0, 1.0),
  (455.0, 0.2837943951348853, 0.0, 1.0),
  (465.0, 0.0, 0.0, 1.0),
  (470.0, 0.0, 0.1763006526677175, 1.0),
  (475.0, 0.0, 0.36799242017191025, 1.0),
  (480.0, 0.0, 0.48449418286510887, 1.0),
  (495.0, 0.0, 0.8978593823560532, 0.8042310312204397),
  (510.0, 0.0, 0.982210818520001, 0.3620321472504943),
  (520.0, 0.0, 1.0, 0.0),
  (550.0, 0.0, 1.0, 0.0),
  (560.0, 0.5394416079798189, 1.0, 0.0),
  (565.0, 0.7626168031212038, 1.0, 0.0),
  (570.0, 0.9299689875333503, 1.0, 0.0),
  (573.0, 1.0, 0.9719495680768989, 0.0),
  (580.0, 1.0, 0.8912722682805861, 0.0),
  (600.0, 1.0, 0.44854054699533813, 0.0),
  (605.0, 1.0, 0.2777211925446866, 0.0),
  (610.0, 1.0, 0.0, 0.0),
  (640.0, 1.0, 0.0, 0.0),
  (645.0, 0.9580637484636693, 0.0, 0.0),
  (660.0, 0.6786210159793107, 0.0, 0.0),
  (750.0, 0.0, 0.0, 0.0),
];

fn visible_spectrum_rgb(w: f64) -> (f64, f64, f64) {
  for pair in VISIBLE_SPECTRUM_KNOTS.windows(2) {
    let ((w0, r0, g0, b0), (w1, r1, g1, b1)) = (pair[0], pair[1]);
    if (w0..=w1).contains(&w) {
      let at = |v0: f64, v1: f64| v0 + (v1 - v0) * (w - w0) / (w1 - w0);
      return (at(r0, r1), at(g0, g1), at(b0, b1));
    }
  }
  (0.0, 0.0, 0.0)
}

/// The color of a black body glowing at `kelvin`: the Planckian locus
/// chromaticity (Kim et al.'s cubic approximation, valid 1667–25000 K)
/// converted to sRGB and scaled so the brightest channel is 1.
fn black_body_rgb(kelvin: f64) -> (f64, f64, f64) {
  let t = kelvin.clamp(1667.0, 25000.0);
  let x = if t <= 4000.0 {
    -0.2661239e9 / t.powi(3) - 0.2343589e6 / t.powi(2)
      + 0.8776956e3 / t
      + 0.179910
  } else {
    -3.0258469e9 / t.powi(3)
      + 2.1070379e6 / t.powi(2)
      + 0.2226347e3 / t
      + 0.240390
  };
  let y = if t <= 2222.0 {
    -1.1063814 * x.powi(3) - 1.34811020 * x.powi(2) + 2.18555832 * x
      - 0.20219683
  } else if t <= 4000.0 {
    -0.9549476 * x.powi(3) - 1.37418593 * x.powi(2) + 2.09137015 * x
      - 0.16748867
  } else {
    3.0817580 * x.powi(3) - 5.87338670 * x.powi(2) + 3.75112997 * x - 0.37001483
  };
  let (big_x, big_z) = (x / y, (1.0 - x - y) / y);
  let linear = [
    3.2406 * big_x - 1.5372 + -0.4986 * big_z,
    -0.9689 * big_x + 1.8758 + 0.0415 * big_z,
    0.0557 * big_x - 0.2040 + 1.0570 * big_z,
  ];
  let peak = linear.iter().copied().fold(f64::MIN, f64::max);
  let encode = |v: f64| {
    let v = (v / peak).clamp(0.0, 1.0);
    if v <= 0.0031308 {
      12.92 * v
    } else {
      1.055 * v.powf(1.0 / 2.4) - 0.055
    }
  };
  (encode(linear[0]), encode(linear[1]), encode(linear[2]))
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

/// The file extension a Wolfram import *format* name stands for, or `None`
/// when the name is an element (`"Data"`, `"Elements"`, …) instead.
fn import_format_name(name: &str) -> Option<&'static str> {
  Some(match name {
    "CSV" => "csv",
    "TSV" => "tsv",
    "JSON" | "RawJSON" => "json",
    "SVG" => "svg",
    "PDF" => "pdf",
    "XLSX" => "xlsx",
    "XLS" => "xls",
    "ODS" => "ods",
    "ROOT" => "root",
    "PNG" => "png",
    "JPEG" => "jpg",
    "GIF" => "gif",
    "BMP" => "bmp",
    "TIFF" => "tiff",
    "PPM" => "ppm",
    "PGM" => "pgm",
    "PBM" => "pbm",
    "PNM" => "pnm",
    "Text" | "String" | "Plaintext" => "txt",
    "Graph6" => "g6",
    _ => return None,
  })
}

/// `Import` of a Graph6 file: its text read the way `ImportString[text,
/// {"Graph6", element}]` reads it. `spec` is Import's second argument, if
/// any — the format, an element, or `{"Graph6", element}`. `None` when the
/// file cannot be read, leaving the usual missing-file handling to run.
#[cfg(not(target_arch = "wasm32"))]
fn import_graph6_file(
  path: &str,
  spec: Option<&Expr>,
) -> Option<Result<Expr, InterpreterError>> {
  let content = std::fs::read_to_string(path).ok()?;
  let element = match spec {
    Some(Expr::String(s)) if s != "Graph6" => Some(Expr::String(s.clone())),
    Some(Expr::List(items)) if items.len() == 2 => Some(items[1].clone()),
    _ => None,
  };
  let mut format = vec![str_expr("Graph6")];
  format.extend(element);
  dispatch_image_functions(
    "ImportString",
    &[Expr::String(content), Expr::List(format.into())],
  )
}

/// The format an `Import` spec names, for the `"FMT"` and `{"FMT", elem…}`
/// shapes alike.
fn import_spec_format(spec: &Expr) -> Option<&'static str> {
  match spec {
    Expr::String(fmt) => import_format_name(fmt),
    Expr::List(items) => match items.first() {
      Some(Expr::String(fmt)) => import_format_name(fmt),
      _ => None,
    },
    _ => None,
  }
}

/// The file extension standing in for an explicit `Import` format, so a
/// command's output is dispatched the way the same bytes in a file would
/// be. Text is the fallback: it is what a pipe carries unless the caller
/// says otherwise.
#[cfg(not(target_arch = "wasm32"))]
fn import_format_extension(format: Option<&Expr>) -> &'static str {
  format.and_then(import_spec_format).unwrap_or("txt")
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
  if !is_url && !crate::vfs::exists(path) {
    crate::emit_message(&format!(
      "Import::nffil: File {path} not found during Import."
    ));
    return Ok(fail_expr());
  }
  let svg = import_read_text(path, is_url)?;
  Ok(crate::graphics_result(svg))
}

/// `Import[pdf, element]`: the pages as `Graphics` by default or for
/// `"Pages"`, one page for `{"Pages", n}`, the page count, or the text.
#[cfg(not(target_arch = "wasm32"))]
fn import_pdf_element(
  path: &str,
  element: Option<&Expr>,
) -> Result<Expr, InterpreterError> {
  if !crate::vfs::exists(path) {
    crate::emit_message(&format!(
      "Import::nffil: File {path} not found during Import."
    ));
    return Ok(fail_expr());
  }
  use crate::functions::pdf_import;
  let pages =
    |path: &str| pdf_import::import_pages(path).map(|p| Expr::List(p.into()));
  match element {
    None => pages(path),
    Some(Expr::String(e)) => match e.as_str() {
      "Pages" | "Graphics" => pages(path),
      "PageCount" => {
        pdf_import::page_count(path).map(|n| Expr::Integer(n as i128))
      }
      "Plaintext" | "Text" => pdf_import::plaintext(path).map(Expr::String),
      "Elements" => Ok(Expr::List(
        ["PageCount", "Pages", "Plaintext"]
          .iter()
          .map(|e| str_expr(e))
          .collect(),
      )),
      _ => {
        crate::emit_message(&format!(
          "Import::noelem: The Import element \"{e}\" is not present when importing as PDF."
        ));
        Ok(fail_expr())
      }
    },
    // {"Pages", n} — one page, counted from 1.
    Some(Expr::List(items))
      if items.len() == 2
        && matches!(&items[0], Expr::String(e) if e == "Pages")
        && matches!(&items[1], Expr::Integer(_)) =>
    {
      let Expr::Integer(n) = &items[1] else {
        unreachable!()
      };
      let all = pdf_import::import_pages(path)?;
      if let Some(page) = usize::try_from(*n - 1).ok().and_then(|i| all.get(i))
      {
        return Ok(page.clone());
      }
      crate::emit_message(&format!(
        "Import::noelem: The Import element {{\"Pages\", {n}}} is not present when importing as PDF."
      ));
      Ok(fail_expr())
    }
    Some(other) => {
      crate::emit_message(&format!(
        "Import::noelem: The Import element {} is not present when importing as PDF.",
        crate::syntax::expr_to_string(other)
      ));
      Ok(fail_expr())
    }
  }
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
    std::fs::read_to_string(crate::vfs::resolve(path)).map_err(|e| {
      InterpreterError::EvaluationError(format!(
        "Import: cannot open \"{path}\": {e}"
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
    Err(_) => fail_expr(),
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
fn import_netpbm(path: &str) -> Expr {
  if !crate::vfs::exists(path) {
    // wolframscript prints Import::nffil to stdout for a missing file.
    crate::emit_message_to_stdout(&format!(
      "Import::nffil: File {path} not found during Import."
    ));
    return fail_expr();
  }
  let Ok(bytes) = std::fs::read(crate::vfs::resolve(path)) else {
    return fail_expr();
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
    return fail_expr();
  }
  // Magic looks plausible but Woxi doesn't yet parse Netpbm pixel data.
  // Return $Failed silently — matches wolframscript on a valid file when
  // parsing succeeds (no message; wolframscript returns `Image[…]`).
  fail_expr()
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
      Err(_) => fail_expr(),
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
/// The number literals of a JSON document, in the order they are written.
/// String contents are skipped, so a digit inside a string is not a token.
fn json_number_tokens(text: &str) -> Vec<String> {
  let bytes: Vec<char> = text.chars().collect();
  let mut tokens = Vec::new();
  let mut i = 0;
  while i < bytes.len() {
    let c = bytes[i];
    if c == '"' {
      i += 1;
      while i < bytes.len() && bytes[i] != '"' {
        i += if bytes[i] == '\\' { 2 } else { 1 };
      }
      i += 1;
      continue;
    }
    let starts_number = c.is_ascii_digit()
      || (c == '-' && bytes.get(i + 1).is_some_and(char::is_ascii_digit));
    if starts_number {
      let start = i;
      i += 1;
      while i < bytes.len()
        && (bytes[i].is_ascii_digit()
          || matches!(bytes[i], '.' | 'e' | 'E')
          || (matches!(bytes[i], '+' | '-')
            && matches!(bytes[i - 1], 'e' | 'E')))
      {
        i += 1;
      }
      tokens.push(bytes[start..i].iter().collect());
      continue;
    }
    i += 1;
  }
  tokens
}

/// The exact value of a JSON number literal that carries no decimal point:
/// wolframscript reads `1e3` as the integer 1000 and `1e-3` as the rational
/// 1/1000. A literal with a decimal point stays an approximate real.
fn exact_json_number(token: &str) -> Option<Expr> {
  if token.contains('.') {
    return None;
  }
  let (mantissa, exponent) = match token.split_once(['e', 'E']) {
    Some((m, e)) => (m, e.parse::<i32>().ok()?),
    None => (token, 0),
  };
  let value: i128 = mantissa.parse().ok()?;
  if exponent >= 0 {
    let scale = 10i128.checked_pow(exponent as u32)?;
    return Some(Expr::Integer(value.checked_mul(scale)?));
  }
  let scale = 10i128.checked_pow(exponent.unsigned_abs())?;
  Some(crate::functions::math_ast::make_rational(value, scale))
}

/// Replace the approximate numbers that came from exponent-notation literals
/// with their exact values, walking the tree in the order the literals appear.
/// Anything unexpected (a token that does not match the parsed value) leaves
/// the whole document as serde read it.
fn exact_json_numbers(
  expr: &Expr,
  tokens: &[String],
  index: &mut usize,
) -> Option<Expr> {
  match expr {
    Expr::Integer(_) | Expr::Real(_) | Expr::BigInteger(_) => {
      let token = tokens.get(*index)?;
      *index += 1;
      let parsed: f64 = token.parse().ok()?;
      let current = match expr {
        Expr::Integer(i) => *i as f64,
        Expr::Real(f) => *f,
        _ => parsed,
      };
      if (parsed - current).abs() > f64::EPSILON * parsed.abs().max(1.0) {
        return None;
      }
      Some(exact_json_number(token).unwrap_or_else(|| expr.clone()))
    }
    Expr::List(items) => {
      let mut out = Vec::with_capacity(items.len());
      for item in items {
        out.push(exact_json_numbers(item, tokens, index)?);
      }
      Some(Expr::List(out.into()))
    }
    Expr::Association(pairs) => {
      let mut out = Vec::with_capacity(pairs.len());
      for (k, v) in pairs {
        out.push((k.clone(), exact_json_numbers(v, tokens, index)?));
      }
      Some(Expr::Association(out))
    }
    Expr::Rule {
      pattern,
      replacement,
    } => Some(Expr::Rule {
      pattern: pattern.clone(),
      replacement: Box::new(exact_json_numbers(replacement, tokens, index)?),
    }),
    other => Some(other.clone()),
  }
}

fn json_value_to_expr(value: &serde_json::Value, raw: bool) -> Expr {
  use serde_json::Value;
  match value {
    Value::Null => null_expr(),
    Value::Bool(true) => bool_expr(true),
    Value::Bool(false) => bool_expr(false),
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
            .map(|(k, v)| {
              rule_expr(Expr::String(k.clone()), json_value_to_expr(v, raw))
            })
            .collect(),
        )
      }
    }
  }
}

/// The argument positions at which an image function takes an image, for
/// the functions Wolfram also hands a graphic to (rasterizing it first).
fn image_argument_positions(name: &str, argc: usize) -> Vec<usize> {
  match name {
    "ImageApply" => vec![1],
    "ImageAdd" | "ImageSubtract" | "ImageMultiply" | "ImageDifference" => {
      (0..argc).collect()
    }
    "ImageCompose" => (0..argc.min(2)).collect(),
    "ImageData"
    | "Binarize"
    | "ImageAdjust"
    | "ColorQuantize"
    | "ImageRotate"
    | "ImageReflect"
    | "Blur"
    | "EdgeDetect"
    | "ImageLines"
    | "ImageDimensions"
    | "Sharpen"
    | "DominantColors"
    | "ColorNegate"
    | "ColorConvert"
    | "ImageCrop"
    | "ImageTake"
    | "ImagePad"
    | "ImageTrim"
    | "GaussianFilter"
    | "MedianFilter"
    | "Dilation"
    | "Erosion"
    | "Opening"
    | "Closing"
    | "ImageHistogram"
    | "ImageLevels"
    | "ImageMeasurements"
    | "ImageColorSpace"
    | "ImageChannels"
    | "ImageType"
    | "ColorSeparate"
    | "ImageEffect"
    | "ImageAspectRatio"
    | "Thinning"
    | "MorphologicalComponents"
    | "WatershedComponents"
    | "MorphologicalGraph"
    | "FillingTransform"
    | "DistanceTransform"
    | "ImageConvolve"
    | "ImageCorrelate"
    | "ImageFilter"
    | "ImageValue"
    | "PixelValue"
    | "ImageTransformation"
    | "ColorDetect"
    | "ImagePartition"
    | "ImageResize"
    | "RemoveAlphaChannel"
    | "SetAlphaChannel" => vec![0],
    _ => Vec::new(),
  }
}

/// `ColorData["VisibleSpectrum"]` / `ColorData["BlackBodySpectrum"]`: the
/// color function over the scheme's physical domain (wavelength in nm,
/// temperature in K).
fn physical_spectrum_color_function(scheme: &str) -> Expr {
  let domain = if scheme == "VisibleSpectrum" {
    (380, 750)
  } else {
    (1000, 12000)
  };
  let sample = Expr::Function {
    body: Box::new(call("ColorData", vec![str_expr(scheme), Expr::Slot(1)])),
  };
  call(
    "ColorDataFunction",
    vec![
      str_expr(scheme),
      str_expr("Gradients"),
      Expr::List(vec![Expr::Integer(domain.0), Expr::Integer(domain.1)].into()),
      sample,
    ],
  )
}

pub fn dispatch_image_functions(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  // SetAlphaChannel[graphic, alpha] rasterizes the graphic at the alpha
  // image's size.
  if name == "SetAlphaChannel"
    && args.len() == 2
    && let Expr::Image { width, height, .. } = &args[1]
    && (matches!(&args[0], Expr::Graphics { .. })
      || matches!(&args[0], Expr::FunctionCall { name, .. } if name == "Graphics"))
  {
    let size = Expr::List(
      vec![
        Expr::Integer(*width as i128),
        Expr::Integer(*height as i128),
      ]
      .into(),
    );
    let raster = crate::evaluator::evaluate_function_call_ast(
      "Rasterize",
      &[
        args[0].clone(),
        Expr::Rule {
          pattern: Box::new(Expr::Identifier("RasterSize".into())),
          replacement: Box::new(size),
        },
      ],
    );
    if let Ok(img @ Expr::Image { .. }) = raster {
      return dispatch_image_functions(name, &[img, args[1].clone()]);
    }
  }
  // A graphic where an image is expected is rasterized first, as in
  // Wolfram: ColorNegate[Graphics[…]] is an Image.
  let positions = image_argument_positions(name, args.len());
  if positions.iter().any(|&i| {
    args.get(i).is_some_and(|a| {
      matches!(a, Expr::Graphics { .. })
        || matches!(a, Expr::FunctionCall { name, .. } if name == "Graphics")
    })
  }) {
    let mut rasterized = args.to_vec();
    let mut changed = false;
    for &i in &positions {
      let is_graphic = rasterized.get(i).is_some_and(|a| {
        matches!(a, Expr::Graphics { .. })
          || matches!(a, Expr::FunctionCall { name, .. } if name == "Graphics")
      });
      if is_graphic
        && let Some(img) =
          crate::functions::image_ast::coerce_graphics_to_image(&rasterized[i])
      {
        rasterized[i] = img;
        changed = true;
      }
    }
    if changed {
      return dispatch_image_functions(name, &rasterized);
    }
  }
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
        Err(_) => Ok(unevaluated("Image", args)),
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
    "ImageMeasurements" => {
      return Some(crate::functions::image_ast::image_measurements_ast(args));
    }
    "ImageChannels" if args.len() == 1 => {
      return Some(crate::functions::image_ast::image_channels_ast(args));
    }
    "ImageType" if args.len() == 1 => {
      return Some(crate::functions::image_ast::image_type_ast(args));
    }
    "Image3DSlices" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::image_ast::image3d_slices_ast(args));
    }
    // ImageData[image, type, Interleaving -> False]: the multichannel data
    // channel by channel, {c, h, w}, rather than pixel by pixel.
    "ImageData"
      if (2..=3).contains(&args.len())
        && matches!(args.last(), Some(Expr::Rule { pattern, .. })
          if matches!(pattern.as_ref(), Expr::Identifier(o) if o == "Interleaving")) =>
    {
      let (pos, opts) = super::split_trailing_options(args);
      let interleaved = match opts.last().map(|(_, v)| v) {
        Some(Expr::Identifier(v)) if v == "True" => true,
        Some(Expr::Identifier(v)) if v == "False" => false,
        _ => return Some(Ok(unevaluated(name, args))),
      };
      let data = match crate::functions::image_ast::image_data_ast(pos) {
        Ok(d) => d,
        Err(e) => return Some(Err(e)),
      };
      let three_d = matches!(&data, Expr::List(rows)
        if matches!(rows.first(), Some(Expr::List(px))
          if matches!(px.first(), Some(Expr::List(_)))));
      if interleaved || !three_d {
        return Some(Ok(data));
      }
      return Some(crate::evaluator::evaluate_function_call_ast(
        "Transpose",
        &[
          data,
          Expr::List(
            vec![Expr::Integer(2), Expr::Integer(3), Expr::Integer(1)].into(),
          ),
        ],
      ));
    }
    "ImageData" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::image_ast::image_data_ast(args));
    }
    "PixelValuePositions" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::image_ast::pixel_value_positions_ast(
        args,
      ));
    }
    "AlphaChannel" if args.len() == 1 => {
      return Some(crate::functions::image_ast::alpha_channel_ast(args));
    }
    "RemoveAlphaChannel" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::image_ast::remove_alpha_channel_ast(args));
    }
    "SetAlphaChannel" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::image_ast::set_alpha_channel_ast(args));
    }
    "ImageColorSpace" if args.len() == 1 => {
      return Some(crate::functions::image_ast::image_color_space_ast(args));
    }
    "ColorNegate" if args.len() == 1 => {
      return Some(crate::functions::image_ast::color_negate_ast(args));
    }
    "Binarize" if !args.is_empty() && args.len() <= 3 => {
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
    "ImageEffect" if args.len() == 2 => {
      return Some(crate::functions::image_ast::image_effect_ast(args));
    }
    "ImageAdjust" if !args.is_empty() && args.len() <= 4 => {
      return Some(crate::functions::image_ast::image_adjust_ast(args));
    }
    "HistogramTransform" if !args.is_empty() => {
      return Some(crate::functions::image_ast::histogram_transform_ast(args));
    }
    "ImageHistogram" if !args.is_empty() => {
      return Some(crate::functions::image_ast::image_histogram_ast(args));
    }
    "ImageLevels" if !args.is_empty() => {
      return Some(crate::functions::image_ast::image_levels_ast(args));
    }
    "WatershedComponents" if !args.is_empty() => {
      return Some(crate::functions::image_ast::watershed_components_ast(args));
    }
    "MorphologicalGraph" if !args.is_empty() => {
      return Some(crate::functions::image_ast::morphological_graph_ast(args));
    }
    "Pruning" if !args.is_empty() => {
      return Some(crate::functions::image_ast::pruning_ast(args));
    }
    "ColorBalance" if !args.is_empty() => {
      return Some(crate::functions::image_ast::color_balance_ast(args));
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
    "ImageCrop" if !args.is_empty() => {
      return Some(crate::functions::image_ast::image_crop_ast(args));
    }
    "ImageForwardTransformation" if args.len() >= 2 => {
      return Some(
        crate::functions::image_ast::image_forward_transformation_ast(args),
      );
    }
    "ImageTransformation" if args.len() >= 2 => {
      return Some(crate::functions::image_ast::image_transformation_ast(args));
    }
    "HighlightImage" if args.len() >= 2 => {
      return Some(crate::functions::image_ast::highlight_image_ast(
        args,
        &evaluate_expr_to_expr,
      ));
    }
    "ImagePad" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::image_ast::image_pad_ast(args));
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
    "FindThreshold" if args.len() == 1 => {
      return Some(crate::functions::image_ast::find_threshold_ast(args));
    }
    "MorphologicalBinarize" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::image_ast::morphological_binarize_ast(
        args,
      ));
    }
    "MorphologicalComponents" if !args.is_empty() => {
      return Some(crate::functions::image_ast::morphological_components_ast(
        args,
      ));
    }
    "MorphologicalPerimeter" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::image_ast::morphological_perimeter_ast(
        args,
      ));
    }
    "ComponentMeasurements" if args.len() == 2 => {
      return Some(crate::functions::image_ast::component_measurements_ast(
        args,
      ));
    }
    "DeleteSmallComponents" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::image_ast::delete_small_components_ast(
        args,
      ));
    }
    "ImageValue" if args.len() == 2 => {
      return Some(crate::functions::image_ast::image_value_ast(args));
    }
    "FillingTransform" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::image_ast::filling_transform_ast(args));
    }
    "DistanceTransform" if (1..=3).contains(&args.len()) => {
      return Some(crate::functions::image_ast::distance_transform_ast(args));
    }
    "ColorCombine" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::image_ast::color_combine_ast(args));
    }
    "ColorSeparate" if args.len() == 1 => {
      return Some(crate::functions::image_ast::color_separate_ast(args));
    }
    // ColorSeparate[image, space]: the channels of the image in that space.
    "ColorSeparate"
      if args.len() == 2
        && matches!(&args[0], Expr::Image { .. })
        && matches!(&args[1], Expr::String(_)) =>
    {
      let converted = match crate::evaluator::evaluate_function_call_ast(
        "ColorConvert",
        args,
      ) {
        Ok(c) => c,
        Err(e) => return Some(Err(e)),
      };
      if matches!(converted, Expr::Image { .. }) {
        return Some(crate::functions::image_ast::color_separate_ast(&[
          converted,
        ]));
      }
    }
    "Colorize" if !args.is_empty() => {
      return Some(crate::functions::image_ast::colorize_ast(args));
    }
    "GaussianFilter" if args.len() == 2 => {
      return Some(crate::functions::image_ast::gaussian_filter_ast(args));
    }
    // `{{r}}` / `{{r}, σ}` ask for the one-dimensional kernel, which is
    // the margin of the matrix: the matrix is the kernel's outer product.
    "GaussianMatrix"
      if args.len() == 1
        && let Expr::List(spec) = &args[0]
        && let Some(Expr::List(radius)) = spec.first()
        && radius.len() == 1
        && spec.len() <= 2 =>
    {
      let mut flat = vec![radius[0].clone()];
      flat.extend(spec[1..].iter().cloned());
      let spec2d = if flat.len() == 1 {
        flat.remove(0)
      } else {
        Expr::List(flat.into())
      };
      let matrix = crate::functions::image_ast::gaussian_matrix_ast(&[spec2d]);
      return Some(matrix.and_then(|m| match &m {
        Expr::List(_) => {
          crate::evaluator::evaluate_function_call_ast("Total", &[m])
        }
        _ => Ok(crate::helpers::unevaluated("GaussianMatrix", args)),
      }));
    }
    "GaussianMatrix" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::image_ast::gaussian_matrix_ast(args));
    }
    "ImageConvolve" if args.len() == 2 => {
      return Some(crate::functions::image_ast::image_convolve_ast(args));
    }
    "ImageDeconvolve" if args.len() >= 2 => {
      return Some(crate::functions::image_ast::image_deconvolve_ast(args));
    }
    "ImageCorrelate" if args.len() == 2 => {
      return Some(crate::functions::image_ast::image_correlate_ast(args));
    }
    "MedianFilter" if args.len() == 2 => {
      return Some(crate::functions::image_ast::median_filter_ast(args));
    }
    "ImageFilter" if args.len() == 3 => {
      return Some(crate::functions::image_ast::image_filter_ast(args));
    }
    "ImageDifference" => {
      return Some(crate::functions::image_ast::image_difference_ast(args));
    }
    // The geometric mean over each neighborhood is the exponential of the
    // arithmetic mean of the logarithms, so it is MeanFilter's job.
    "Inpaint" if args.len() >= 2 => {
      return Some(crate::functions::image_ast::inpaint_ast(args));
    }
    "GeometricMeanFilter" if args.len() == 2 => {
      let src = match &args[0] {
        Expr::Image { .. } => {
          "Image[Exp[MeanFilter[Log[ImageData[#1]], #2]]] &"
        }
        // Windows of a vector go through GeometricMean so that exact
        // data stays in radical form instead of E^(Log[..]/n).
        Expr::List(items)
          if items.iter().all(|e| !matches!(e, Expr::List(_))) =>
        {
          "Table[GeometricMean[#1[[Max[i - #2, 1] ;; Min[i + #2, Length[#1]]]]], {i, Length[#1]}] &"
        }
        Expr::List(_) => {
          "Table[GeometricMean[Flatten[#1[[Max[i - #2, 1] ;; Min[i + #2, Length[#1]], Max[j - #2, 1] ;; Min[j + #2, Length[#1[[1]]]]]]]], {i, Length[#1]}, {j, Length[#1[[1]]]}] &"
        }
        _ => {
          return Some(Ok(crate::helpers::unevaluated(
            "GeometricMeanFilter",
            args,
          )));
        }
      };
      return Some(crate::interpret_to_expr(src).and_then(|f| {
        crate::evaluator::evaluate_expr_to_expr(&Expr::CurriedCall {
          func: Box::new(f),
          args: args.to_vec(),
        })
      }));
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
    "LaplacianFilter" if args.len() == 2 => {
      return Some(crate::functions::image_ast::laplacian_filter_ast(args));
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
    "EdgeDetect" if !args.is_empty() => {
      return Some(crate::functions::image_ast::edge_detect_ast(args));
    }
    "ImageLines" if !args.is_empty() => {
      return Some(crate::functions::image_ast::image_lines_ast(args));
    }
    "DominantColors" if !args.is_empty() => {
      // Dispatch every non-empty call: dominant_colors_ast itself checks
      // for non-Image first arg (emits DominantColors::imginv) before
      // validating arg count, matching wolframscript when an upstream
      // Import has returned $Failed.
      return Some(crate::functions::image_ast::dominant_colors_ast(args));
    }
    "ImageApply" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::image_ast::image_apply_ast(
        args,
        &evaluate_expr_to_expr,
      ));
    }
    "ColorConvert" if args.len() == 2 => {
      // Any color directive `color_convert_ast` can read.
      let is_color = |c: &Expr| {
        crate::functions::image_ast::color_directive_to_rgb(c).is_some()
      };
      let is_color_directive = is_color(&args[0]);
      let is_color_list = matches!(&args[0], Expr::List(items)
        if !items.is_empty() && items.iter().all(is_color));
      if matches!(&args[0], Expr::Image { .. })
        || is_color_directive
        || is_color_list
      {
        return Some(crate::functions::image_ast::color_convert_ast(args));
      }
    }
    "ColorReplace" if (2..=3).contains(&args.len()) => {
      return Some(crate::functions::image_ast::color_replace_ast(args));
    }
    "ColorDistance" if args.len() >= 2 => {
      return Some(crate::functions::image_ast::color_distance_ast(args));
    }
    // ColorData[]: list of available data categories.
    "ColorData" if args.is_empty() => {
      return Some(Ok(Expr::List(
        vec![
          str_expr("Gradients"),
          str_expr("Indexed"),
          str_expr("Named"),
          str_expr("Physical"),
        ]
        .into(),
      )));
    }
    // ColorData["Named"] / ColorData["Physical"]: the schemes of a class.
    "ColorData"
      if args.len() == 1
        && matches!(&args[0], Expr::String(c) if c == "Named" || c == "Physical") =>
    {
      let names: &[&str] = if matches!(&args[0], Expr::String(c) if c == "Named")
      {
        &[
          "Atoms",
          "Crayola",
          "GeologicAges",
          "HTML",
          "Legacy",
          "WebSafe",
        ]
      } else {
        &["BlackBodySpectrum", "HypsometricTints", "VisibleSpectrum"]
      };
      return Some(Ok(Expr::List(
        names.iter().map(|n| str_expr(n)).collect::<Vec<_>>().into(),
      )));
    }
    // ColorData[n, "ColorList"]: the ordered list of colors in indexed
    // scheme n (e.g. `ColorData[97, "ColorList"]` — the default plot palette).
    "ColorData" if args.len() == 2 => {
      // ColorData["HTML" | "Legacy", "ColorRules"]: every name -> colour,
      // in table order.
      if let (Expr::String(scheme), Expr::String(property)) =
        (&args[0], &args[1])
        && property == "ColorRules"
      {
        let names: Option<Vec<&str>> = match scheme.as_str() {
          "HTML" => Some(HTML_COLORS.iter().map(|c| c.0).collect()),
          "Legacy" => Some(LEGACY_COLORS.iter().map(|c| c.0).collect()),
          other => named_color_table(other)
            .map(|table| table.iter().map(|c| c.0).collect()),
        };
        if let Some(names) = names {
          let mut rules = Vec::with_capacity(names.len());
          for name in names {
            let color = match crate::evaluator::evaluate_function_call_ast(
              "ColorData",
              &[args[0].clone(), str_expr(name)],
            ) {
              Ok(c) => c,
              Err(e) => return Some(Err(e)),
            };
            rules.push(Expr::Rule {
              pattern: Box::new(str_expr(name)),
              replacement: Box::new(color),
            });
          }
          return Some(Ok(Expr::List(rules.into())));
        }
      }
      // ColorData["Crayola" | "WebSafe" | "GeologicAges", name].
      if let (Expr::String(scheme), Expr::String(color_name)) =
        (&args[0], &args[1])
        && let Some(table) = named_color_table(scheme)
        && let Some(&(_, r, g, b)) = table
          .iter()
          .find(|(n, ..)| n.eq_ignore_ascii_case(color_name))
      {
        return Some(Ok(call(
          "RGBColor",
          vec![Expr::Real(r), Expr::Real(g), Expr::Real(b)],
        )));
      }
      // ColorData["Legacy", name]: the old `Graphics`Colors`` palette, which
      // is a different table from the CSS one — see `LEGACY_COLORS`.
      if let (Expr::String(scheme), Expr::String(color_name)) =
        (&args[0], &args[1])
        && scheme == "Legacy"
        && let Some(color) = legacy_named_color(color_name)
      {
        return Some(Ok(color));
      }
      // ColorData["Atoms", element]: the colour an element is drawn in a
      // molecular model, looked up by symbol or atomic number.
      if let Expr::String(scheme) = &args[0]
        && scheme == "Atoms"
        && let Some((r, g, b)) = atom_color(&args[1])
      {
        return Some(Ok(call(
          "RGBColor",
          vec![Expr::Real(r), Expr::Real(g), Expr::Real(b)],
        )));
      }
      // ColorData["HTML", name]: the named CSS colour.
      if let (Expr::String(scheme), Expr::String(color_name)) =
        (&args[0], &args[1])
        && scheme == "HTML"
        && let Some((r, g, b)) = html_named_color(color_name)
      {
        // wolframscript keeps an exactly-zero channel exact: `Aqua` is
        // `RGBColor[0, 1., 1.]`, not `RGBColor[0., 1., 1.]`.
        let channel = |v: f64| {
          if v == 0.0 {
            Expr::Integer(0)
          } else {
            Expr::Real(v)
          }
        };
        return Some(Ok(call(
          "RGBColor",
          vec![channel(r), channel(g), channel(b)],
        )));
      }
      // ColorData["VisibleSpectrum" | "BlackBodySpectrum", "ColorFunction"]:
      // the same object the one-argument form gives.
      if let (Expr::String(scheme), Expr::String(p)) = (&args[0], &args[1])
        && p == "ColorFunction"
        && (scheme == "VisibleSpectrum" || scheme == "BlackBodySpectrum")
      {
        return Some(Ok(physical_spectrum_color_function(scheme)));
      }
      // ColorData["VisibleSpectrum", wavelength]: the color of light with
      // the given wavelength in nm (380–750 nm; black outside that range).
      if let Expr::String(scheme) = &args[0]
        && scheme == "VisibleSpectrum"
        && !matches!(&args[1], Expr::String(_))
        && let Some(w) = crate::functions::math_ast::try_eval_to_f64(&args[1])
      {
        let (r, g, b) = visible_spectrum_rgb(w);
        return Some(Ok(call(
          "RGBColor",
          vec![Expr::Real(r), Expr::Real(g), Expr::Real(b)],
        )));
      }
      // ColorData["BlackBodySpectrum", kelvin]: the color of a black body at
      // the given temperature.
      if let Expr::String(scheme) = &args[0]
        && scheme == "BlackBodySpectrum"
        && !matches!(&args[1], Expr::String(_))
        && let Some(t) = crate::functions::math_ast::try_eval_to_f64(&args[1])
      {
        let (r, g, b) = black_body_rgb(t);
        return Some(Ok(call(
          "RGBColor",
          vec![Expr::Real(r), Expr::Real(g), Expr::Real(b)],
        )));
      }
      // ColorData[name, t]: sample the named gradient at parameter t
      // (clamped to [0, 1]), and ColorData[name, "Image"]: a horizontal
      // gradient strip image (the swatch Wolfram shows in gradient
      // pickers).
      if let Expr::String(scheme) = &args[0]
        && let Some(controls) =
          crate::functions::chart::named_color_scheme(scheme)
      {
        if matches!(&args[1], Expr::String(p) if p == "Image") {
          return Some(Ok(gradient_strip_image(controls)));
        }
        // ColorData[name, "ColorFunction"]: the gradient's color function,
        // the same object `ColorData[name]` gives on its own. Demonstrations
        // reach for the property form when the result is applied straight
        // away, as in `ColorData[name, "ColorFunction"][t]`.
        if matches!(&args[1], Expr::String(p) if p == "ColorFunction") {
          return Some(Ok(gradient_color_function(scheme)));
        }
        // ColorData[name, "Range"]: the parameter interval the gradient is
        // defined over — always {0, 1} for the built-in gradients.
        if matches!(&args[1], Expr::String(p) if p == "Range") {
          return Some(Ok(Expr::List(
            vec![Expr::Integer(0), Expr::Integer(1)].into(),
          )));
        }
        if !matches!(&args[1], Expr::String(_))
          && let Some(t) = crate::functions::math_ast::try_eval_to_f64(&args[1])
        {
          let (r, g, b) =
            crate::functions::chart::gradient_color_at(controls, t);
          return Some(Ok(call(
            "RGBColor",
            vec![Expr::Real(r), Expr::Real(g), Expr::Real(b)],
          )));
        }
      }
      // ColorData[n, k]: the k-th color of indexed scheme n. A negative
      // index is not a color but an unknown property.
      if let Expr::Integer(n) = &args[0]
        && !matches!(&args[1], Expr::String(_))
        && crate::functions::math_ast::try_eval_to_f64(&args[1])
          .is_some_and(|k| k >= 0.0)
        && let Some(color) = indexed_scheme_color(*n, &args[1])
      {
        return Some(Ok(color));
      }
      if let (Expr::Integer(n), Expr::String(prop)) = (&args[0], &args[1]) {
        match prop.as_str() {
          "ColorList" => {
            if let Some(colors) = indexed_color_list(*n) {
              return Some(Ok(colors));
            }
          }
          "ColorFunction" => {
            if let Some(f) = indexed_color_function(*n) {
              return Some(Ok(f));
            }
          }
          "Range" => {
            if let Some((range, _)) = indexed_scheme(*n) {
              return Some(Ok(range));
            }
          }
          _ => {}
        }
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
          names.iter().map(|n| str_expr(n)).collect(),
        )));
      }
      // ColorData["WebSafe"]: applying it to a hex code (`"#99CC00"`) gives
      // the web-safe colour, via the two-argument form.
      if matches!(&args[0], Expr::String(s) if s == "WebSafe") {
        let sample = Expr::Function {
          body: Box::new(call(
            "ColorData",
            vec![str_expr("WebSafe"), Expr::Slot(1)],
          )),
        };
        return Some(Ok(call(
          "ColorDataFunction",
          vec![
            str_expr("WebSafe"),
            str_expr("Named"),
            Expr::List(vec![].into()),
            sample,
          ],
        )));
      }
      // ColorData["VisibleSpectrum"] (wavelength, 380–750 nm) and
      // ColorData["BlackBodySpectrum"] (temperature in K): the color
      // function of the scheme. Applying it (`ColorData["VisibleSpectrum"][w]`, the way
      // Demonstrations tint a plot by wavelength) delegates to the
      // two-argument form.
      if let Expr::String(scheme) = &args[0]
        && (scheme == "VisibleSpectrum" || scheme == "BlackBodySpectrum")
      {
        return Some(Ok(physical_spectrum_color_function(scheme)));
      }
      // ColorData["scheme"]: the gradient's color function, echoed in
      // wolframscript's structured form
      //   ColorDataFunction[scheme, "Gradients", {0, 1}, Blend[scheme, #1] &].
      // Applying it (ColorData["scheme"][t]) blends the stored control
      // points at t (see apply_curried_call).
      if let Expr::String(scheme) = &args[0]
        && crate::functions::chart::named_color_scheme(scheme).is_some()
      {
        return Some(Ok(gradient_color_function(scheme)));
      }
      // ColorData[{"scheme", {min, max}}]: the gradient over a custom range.
      if let Expr::List(items) = &args[0]
        && let [Expr::String(scheme), range @ Expr::List(bounds)] =
          items.as_slice()
        && bounds.len() == 2
        && crate::functions::chart::named_color_scheme(scheme).is_some()
      {
        return Some(Ok(ranged_gradient_color_function(scheme, range)));
      }
      // ColorData[{"scheme", "Reverse"}]: the gradient run backwards.
      if let Expr::List(items) = &args[0]
        && let [Expr::String(scheme), Expr::String(modifier)] = items.as_slice()
        && modifier == "Reverse"
        && crate::functions::chart::named_color_scheme(scheme).is_some()
      {
        return Some(Ok(reversed_gradient_color_function(scheme)));
      }
      // ColorData[n]: the indexed scheme's color function, echoed in the
      // same structured form as the named-gradient case above. Applying it
      // (ColorData[n][k] — the common way Demonstrations cycle a palette,
      // e.g. `ColorData[1][i]`) delegates to the already-supported
      // `ColorData[n, k]` two-argument form (see `apply_curried_call`).
      if let Expr::Integer(n) = &args[0]
        && let Some(f) = indexed_color_function(*n)
      {
        return Some(Ok(f));
      }
    }
    "ImageCompose" if (2..=5).contains(&args.len()) => {
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
    // RandomImage[spec, size, "type"]: the random image converted to the
    // given pixel type, as Image[img, "type"] does.
    "RandomImage" if args.len() == 3 && matches!(args[2], Expr::String(_)) => {
      let img = match crate::functions::image_ast::random_image_ast(&args[..2])
      {
        Ok(i) => i,
        Err(e) => return Some(Err(e)),
      };
      if !matches!(img, Expr::Image { .. }) {
        return Some(Ok(unevaluated(name, args)));
      }
      return Some(crate::evaluator::evaluate_function_call_ast(
        "Image",
        &[img, args[2].clone()],
      ));
    }
    "RandomImage" if args.len() <= 2 => {
      return Some(crate::functions::image_ast::random_image_ast(args));
    }
    // ConstantImage[color, size, type, opts]: the constant image, converted
    // by Image to the requested type and given its options.
    "ConstantImage"
      if args.len() >= 3
        || (args.len() == 2 && matches!(&args[1], Expr::Rule { .. })) =>
    {
      let (base, rest) = match &args[1] {
        Expr::Rule { .. } => (&args[..1], &args[1..]),
        _ => (&args[..2], &args[2..]),
      };
      let image = match crate::functions::image_ast::constant_image_ast(base) {
        Ok(img @ Expr::Image { .. }) => img,
        Ok(_) => return Some(Ok(unevaluated(name, args))),
        Err(e) => return Some(Err(e)),
      };
      let mut image_args = vec![image];
      image_args.extend(rest.iter().cloned());
      return Some(crate::evaluator::evaluate_function_call_ast(
        "Image",
        &image_args,
      ));
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
    // Import["!command"] / Import["!command", format] — run the command and
    // import what it writes to standard output. Its bytes are materialized
    // in a temporary file so every format below reads them exactly as it
    // reads a real file; the file carries the extension that stands for the
    // requested format (text when none is given).
    #[cfg(not(target_arch = "wasm32"))]
    "Import"
      if (1..=2).contains(&args.len())
        && import_path_spec(&args[0]).is_some_and(|p| p.starts_with('!')) =>
    {
      use crate::evaluator::dispatch::io_functions::{
        command_file_spec, run_command_capture_bytes,
      };
      let path = import_path_spec(&args[0]).unwrap_or_default();
      // A pipe carries no file name to guess a format from, so wolframscript
      // refuses to guess: `Import["!printf hi"]` is `$Failed`, and only
      // `Import["!printf hi", "Text"]` reads it.
      if args.len() == 1 {
        crate::emit_message_to_stdout(
          "Import::general: A format must be specified when importing \
           from a pipe.",
        );
        return Some(Ok(fail_expr()));
      }
      let command = command_file_spec(&path).unwrap_or_default();
      let failed = || {
        crate::emit_message(&format!("Import::nffil: Cannot open {path}."));
        Some(Ok(fail_expr()))
      };
      let Some(bytes) = run_command_capture_bytes(command) else {
        return failed();
      };
      let ext = import_format_extension(args.get(1));
      let Ok(temp) = crate::utils::create_temp_file(ext) else {
        return failed();
      };
      if std::fs::write(&temp, &bytes).is_err() {
        let _ = std::fs::remove_file(&temp);
        return failed();
      }
      let mut piped = args.to_vec();
      piped[0] = Expr::String(temp.to_string_lossy().into_owned());
      let result = dispatch_image_functions("Import", &piped);
      let _ = std::fs::remove_file(&temp);
      return result;
    }
    // Import[file, (elem,) opt -> value, …] — the options (CharacterEncoding
    // and the like) do not change what is read here; import the file as
    // the arguments before them say.
    "Import"
      if args.len() >= 2
        && matches!(
          args[args.len() - 1],
          Expr::Rule { .. } | Expr::RuleDelayed { .. }
        ) =>
    {
      let core: Vec<Expr> = args
        .iter()
        .take_while(|a| {
          !matches!(a, Expr::Rule { .. } | Expr::RuleDelayed { .. })
        })
        .cloned()
        .collect();
      if core.is_empty() || core.len() > 2 {
        return Some(Ok(unevaluated("Import", args)));
      }
      return dispatch_image_functions("Import", &core);
    }
    "Import" if args.len() == 1 => {
      let Some(path) = import_path_spec(&args[0]) else {
        return Some(Ok(unevaluated("Import", args)));
      };
      let is_url = path.starts_with("http://") || path.starts_with("https://");
      let ext = import_extension(&path);

      // In the browser every non-URL path is served from the
      // host-registered virtual file store.
      #[cfg(target_arch = "wasm32")]
      if !is_url {
        return Some(import_virtual(&path, None));
      }

      #[cfg(not(target_arch = "wasm32"))]
      if ext == "g6"
        && !is_url
        && let Some(result) = import_graph6_file(&path, None)
      {
        return Some(result);
      }

      if ext == "pdf" && !is_url {
        #[cfg(not(target_arch = "wasm32"))]
        {
          return Some(import_pdf_element(&path, None));
        }
        #[cfg(target_arch = "wasm32")]
        {
          return Some(Err(InterpreterError::EvaluationError(
            "Import: PDF import is not available in the browser".into(),
          )));
        }
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
              std::fs::read_to_string(crate::vfs::resolve(&path))
                .map(|s| {
                  Expr::String(
                    crate::functions::txt_ast::strip_trailing_newline(s),
                  )
                })
                .map_err(|e| {
                  InterpreterError::EvaluationError(format!(
                    "Import: cannot open \"{path}\": {e}"
                  ))
                }),
            );
          }
          "jpg" | "jpeg" | "png" | "gif" | "bmp" | "tiff" | "tif" => {
            return Some(crate::functions::image_ast::import_image(&path));
          }
          "ppm" | "pgm" | "pbm" | "pnm" => {
            return Some(Ok(import_netpbm(&path)));
          }
          "root" => {
            return Some(crate::functions::root_ast::root_import_file(&path));
          }
          // An XML document reads into the same symbolic form ImportString
          // gives.
          "xml" => {
            return Some(
              std::fs::read_to_string(crate::vfs::resolve(&path))
                .map_err(|e| {
                  InterpreterError::EvaluationError(format!(
                    "Import: cannot open \"{path}\": {e}"
                  ))
                })
                .map(|content| {
                  match crate::functions::xml_ast::parse_xml_document(&content)
                  {
                    Ok(document) => document,
                    Err(error) => {
                      let (line, character) =
                        crate::functions::xml_ast::line_and_character(
                          &content,
                          error.offset,
                        );
                      crate::emit_message(&match &error.unresolved_prefix {
                        Some(prefix) => format!(
                          "Import::nfprserr: prefix '{prefix}' can not be \
                           resolved to namespace URI at line: {line} \
                           character: {character} in input string."
                        ),
                        None => format!(
                          "Import::nfprserr: invalid document structure at \
                           line: {line} character: {character} in input \
                           string."
                        ),
                      });
                      fail_expr()
                    }
                  }
                }),
            );
          }
          "wav" | "wave" | "flac" | "mp3" | "ogg" | "oga" | "opus" | "m4a"
          | "aac" | "aif" | "aiff" => {
            return Some(crate::functions::audio_ast::import_audio_file(&path));
          }
          _ => {
            return Some(Err(InterpreterError::EvaluationError(format!(
              "Import: unsupported file format \"{ext}\""
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
      let Some(path) = import_path_spec(&args[0]) else {
        return Some(Ok(unevaluated("Import", args)));
      };
      let is_url = path.starts_with("http://") || path.starts_with("https://");
      let file_ext = import_extension(&path);
      // A format named outright wins over the one the file name suggests:
      // `Import["script.js", "Text"]` reads the file as text, whatever the
      // extension would otherwise have dispatched to.
      let ext = import_spec_format(&args[1])
        .map_or_else(|| file_ext.clone(), ToString::to_string);

      #[cfg(not(target_arch = "wasm32"))]
      if ext == "g6"
        && !is_url
        && let Some(result) = import_graph6_file(&path, Some(&args[1]))
      {
        return Some(result);
      }

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
        return Some(Ok(unevaluated("Import", args)));
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

      // A PDF's elements: "Pages" (the default), "PageCount", "Plaintext".
      #[cfg(not(target_arch = "wasm32"))]
      if (file_ext == "pdf" || ext == "pdf") && !is_url {
        let element: Option<Expr> = match &args[1] {
          Expr::List(items) if matches!(items.first(), Some(Expr::String(f)) if f == "PDF") => {
            match items.len() {
              1 => None,
              2 => Some(items[1].clone()),
              _ => Some(Expr::List(items[1..].to_vec().into())),
            }
          }
          Expr::String(f) if f == "PDF" => None,
          other => Some(other.clone()),
        };
        return Some(import_pdf_element(&path, element.as_ref()));
      }

      // Formats with handling of their own have claimed the call by now.
      // What is left is a second argument that names a format, an element,
      // or both — and naming the format the file already has says nothing
      // about *what* to extract, so `Import[f, "CSV"]` is `Import[f]` and
      // `Import[f, {"CSV", elem}]` is `Import[f, elem]`. Without this the
      // format name would travel on as an element name that no importer
      // recognizes.
      if import_spec_format(&args[1]) == Some(file_ext.as_str()) {
        let reduced: Vec<Expr> = match &args[1] {
          Expr::List(items) if items.len() > 1 => {
            let mut rest: Vec<Expr> = items.iter().skip(1).cloned().collect();
            let elem = if rest.len() == 1 {
              rest.remove(0)
            } else {
              Expr::List(rest.into())
            };
            vec![args[0].clone(), elem]
          }
          _ => vec![args[0].clone()],
        };
        return dispatch_image_functions("Import", &reduced);
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
            return Some(Ok(unevaluated("Import", args)));
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
            return Some(Ok(unevaluated("Import", args)));
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
            match std::fs::read_to_string(crate::vfs::resolve(&path)) {
              Ok(s) => s,
              Err(e) => {
                return Some(Err(InterpreterError::EvaluationError(format!(
                  "Import: cannot open \"{path}\": {e}"
                ))));
              }
            }
          };
          match crate::functions::txt_ast::import_element(&content, &element) {
            Some(expr) => return Some(Ok(expr)),
            None => {
              return Some(Err(InterpreterError::EvaluationError(format!(
                "Import: unsupported element \"{element}\" for txt file"
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
          Some("ppm" | "pgm" | "pbm" | "pnm")
        )
      {
        #[cfg(not(target_arch = "wasm32"))]
        {
          if is_url {
            // URL imports for netpbm aren't supported yet; fall through.
          } else {
            return Some(Ok(import_netpbm(&path)));
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
      return Some(Ok(unevaluated("Import", args)));
    }
    "ImportString" if !args.is_empty() => {
      // Trailing rules are options; anything else past the format is an
      // argument-count error, as wolframscript reports.
      let (positional, options): (Vec<&Expr>, Vec<&Expr>) =
        args.iter().partition(|a| {
          !matches!(a, Expr::Rule { .. } | Expr::RuleDelayed { .. })
        });
      if positional.len() > 2 {
        crate::emit_message(&format!(
          "ImportString::argt: ImportString called with {} arguments; 1 or 2 \
           arguments are expected.",
          args.len()
        ));
        return Some(Ok(unevaluated("ImportString", args)));
      }
      // `"Numeric" -> False` keeps the fields of a tabular format as strings.
      let numeric = !options.iter().any(|o| {
        matches!(o, Expr::Rule { pattern, replacement }
          if matches!(pattern.as_ref(), Expr::String(s) if s == "Numeric")
            && matches!(replacement.as_ref(), Expr::Identifier(v) if v == "False"))
      });
      let args: Vec<Expr> = positional.iter().map(|a| (*a).clone()).collect();
      let args = &args[..];
      let content = if let Expr::String(s) = &args[0] {
        s.clone()
      } else {
        // wolframscript emits an error when the first argument isn't a
        // string — match that message format.
        crate::emit_message(&format!(
          "ImportString::string: First argument {} is not a string.",
          expr_to_string(&args[0])
        ));
        return Some(Ok(unevaluated("ImportString", args)));
      };

      // ImportString[str] — default to CSV
      // ImportString[str, "CSV"] — explicit CSV
      // The format argument is either a name or a `{name, element}` list
      // naming the part of the document to read.
      let (format, requested_element): (&str, Option<String>) =
        if args.len() == 2 {
          match &args[1] {
            Expr::String(s) => (s.as_str(), None),
            Expr::List(spec) if !spec.is_empty() && spec.len() <= 2 => {
              let Expr::String(name) = &spec[0] else {
                return Some(Ok(unevaluated("ImportString", args)));
              };
              let element = match spec.get(1) {
                None => None,
                Some(Expr::String(e)) => Some(e.clone()),
                Some(_) => {
                  return Some(Ok(unevaluated("ImportString", args)));
                }
              };
              (name.as_str(), element)
            }
            _ => {
              return Some(Ok(unevaluated("ImportString", args)));
            }
          }
        } else {
          ("CSV", None)
        };
      // An element the format does not offer is reported and fails the import.
      let unknown_element = |element: &str, format: &str| {
        crate::emit_message(&format!(
          "Import::noelem: The Import element \"{element}\" is not present \
           when importing as {format}."
        ));
        Some(Ok(fail_expr()))
      };

      // `"Numeric" -> False` keeps every field as the string it was written
      // as, which is exactly the RawData element. An explicit element wins.
      let table_element = match (&requested_element, numeric) {
        (Some(element), _) => Some(element.as_str()),
        (None, false) => Some("RawData"),
        (None, true) => None,
      };

      // Plain-text format elements that don't depend on a parser.
      // `Elements` returns the static list of supported plain-text
      // element names so callers can probe what's available.
      if format == "Elements" {
        let names =
          ["Data", "Lines", "Plaintext", "String", "Summary", "Words"];
        return Some(Ok(Expr::List(
          names.iter().map(|n| str_expr(n)).collect(),
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
          trimmed.split('\n').map(str_expr).collect()
        };
        return Some(Ok(Expr::List(items.into())));
      }
      // `String` / `Plaintext` / `Text` return the input verbatim.
      if format == "String" || format == "Plaintext" || format == "Text" {
        return Some(Ok(Expr::String(content)));
      }
      // `List` reads one field per line, typed like a CSV field: a number
      // becomes a number, a quoted string loses its quotes, anything else
      // stays the text of the line.
      if format == "List" {
        let trimmed = content.strip_suffix('\n').unwrap_or(&content);
        let rows: Vec<Vec<String>> = if trimmed.is_empty() {
          Vec::new()
        } else {
          trimmed
            .split('\n')
            .map(|line| {
              // A quoted line loses its quotes, as a quoted CSV field does.
              let field = match line
                .strip_prefix('"')
                .and_then(|r| r.strip_suffix('"'))
              {
                Some(inner) if line.len() >= 2 => inner.replace("\"\"", "\""),
                _ => line.to_string(),
              };
              vec![field]
            })
            .collect()
        };
        let table =
          crate::functions::csv_ast::csv_import_element(&rows, table_element);
        return Some(Ok(match &table {
          Expr::List(items) => Expr::List(
            items
              .iter()
              .map(|row| match row {
                Expr::List(fields) if fields.len() == 1 => fields[0].clone(),
                other => other.clone(),
              })
              .collect(),
          ),
          other => other.clone(),
        }));
      }
      // `XML` parses the document into wolframscript's symbolic XML.
      if format == "XML" {
        return Some(Ok(
          match crate::functions::xml_ast::parse_xml_document(&content) {
            Ok(document) => match &requested_element {
              None => document,
              Some(element) => {
                match crate::functions::xml_ast::xml_import_element(
                  &document, element,
                ) {
                  Some(part) => part,
                  None => return unknown_element(element, "XML"),
                }
              }
            },
            Err(error) => {
              let (line, character) =
                crate::functions::xml_ast::line_and_character(
                  &content,
                  error.offset,
                );
              crate::emit_message(&match &error.unresolved_prefix {
                Some(prefix) => format!(
                  "Import::nfprserr: prefix '{prefix}' can not be resolved \
                   to namespace URI at line: {line} character: {character} \
                   in input string."
                ),
                None => format!(
                  "Import::nfprserr: invalid document structure at line: \
                   {line} character: {character} in input string."
                ),
              });
              fail_expr()
            }
          },
        ));
      }
      // `JSON` parses the string into nested lists: a JSON array becomes a
      // List, a JSON object becomes a list of `key -> value` rules (string
      // keys, insertion order preserved), and scalars/true/false/null map to
      // the corresponding atoms. Invalid JSON yields $Failed.
      if format == "JSON" || format == "RawJSON" {
        if content.trim().is_empty() {
          crate::emit_message("Import::jsonnullinput: Data in input is null.");
          return Some(Ok(fail_expr()));
        }
        let raw = format == "RawJSON";
        return Some(Ok(
          match serde_json::from_str::<serde_json::Value>(&content) {
            Ok(value) => {
              let parsed = json_value_to_expr(&value, raw);
              let tokens = json_number_tokens(&content);
              let mut index = 0;
              match exact_json_numbers(&parsed, &tokens, &mut index) {
                Some(exact) if index == tokens.len() => exact,
                _ => parsed,
              }
            }
            Err(_) => fail_expr(),
          },
        ));
      }
      // `TSV` is tab-separated; `Table` splits each line on runs of
      // whitespace. Both reuse the CSV element machinery (number auto-typing,
      // {{…}, …} shape) after splitting into string rows.
      if format == "TSV" || format == "Table" {
        // `Table` is a plain grid: it offers none of the column metadata the
        // CSV element list carries.
        if let Some(element) = &requested_element {
          let known = if format == "Table" {
            crate::functions::csv_ast::TABLE_ELEMENTS
              .contains(&element.as_str())
              || element == "Elements"
          } else {
            crate::functions::csv_ast::is_csv_element(element)
          };
          if !known {
            return unknown_element(element, format);
          }
          if format == "Table" && element == "Elements" {
            return Some(Ok(Expr::List(
              crate::functions::csv_ast::TABLE_ELEMENTS
                .iter()
                .map(|n| str_expr(n))
                .collect(),
            )));
          }
          // The remaining Table elements have no implementation of their own.
          if format == "Table"
            && !matches!(element.as_str(), "Data" | "Summary" | "Tabular")
          {
            return Some(Ok(unevaluated("ImportString", args)));
          }
        }
        // An empty TSV document has no rows to read at all.
        if format == "TSV" && content.is_empty() {
          crate::emit_message(
            "Import::fmterr: Cannot import data as TSV format.",
          );
          return Some(Ok(fail_expr()));
        }
        let trimmed = content.strip_suffix('\n').unwrap_or(&content);
        let rows: Vec<Vec<String>> = if trimmed.is_empty() {
          Vec::new()
        } else {
          trimmed
            .split('\n')
            .map(|line| {
              if format == "TSV" {
                line
                  .split('\t')
                  .map(std::string::ToString::to_string)
                  .collect()
              } else {
                line
                  .split_whitespace()
                  .map(std::string::ToString::to_string)
                  .collect()
              }
            })
            .collect()
        };
        return Some(Ok(crate::functions::csv_ast::csv_import_element(
          &rows,
          table_element,
        )));
      }
      // `Words` splits on ASCII whitespace and drops empty fragments.
      if format == "Words" {
        let items: Vec<Expr> =
          content.split_whitespace().map(str_expr).collect();
        return Some(Ok(Expr::List(items.into())));
      }

      // `Graph6` stores graphs as printable-ASCII adjacency bit vectors.
      if format == "Graph6" {
        const GRAPH6_ELEMENTS: &[&str] = &[
          "AdjacencyMatrix",
          "EdgeRules",
          "Graph",
          "Graphics",
          "GraphicsList",
          "GraphList",
          "Summary",
          "VertexCount",
          "VertexList",
        ];
        let element = requested_element.as_deref().unwrap_or("Graph");
        if element == "Elements" {
          return Some(Ok(Expr::List(
            GRAPH6_ELEMENTS.iter().map(|n| str_expr(n)).collect(),
          )));
        }
        if !GRAPH6_ELEMENTS.contains(&element) {
          return unknown_element(element, "Graph6");
        }
        let Some(mut graphs) = crate::functions::graph::decode_graph6(&content)
        else {
          crate::emit_message(
            "Import::fmterr: Cannot import data as Graph6 format.",
          );
          return Some(Ok(fail_expr()));
        };
        // Edges are listed in lexicographic order of their vertex pairs,
        // not in the column order the bit vector stores them in.
        for (_, edges) in &mut graphs {
          for edge in edges.iter_mut() {
            *edge = (edge.0.min(edge.1), edge.0.max(edge.1));
          }
          edges.sort_unstable();
        }
        let int = |v: usize| Expr::Integer(v as i128);
        let vertices = |n: usize| Expr::List((1..=n).map(int).collect());
        let graph = |n: usize, edges: &[(usize, usize)]| {
          crate::evaluator::evaluate_function_call_ast(
            "Graph",
            &[
              vertices(n),
              Expr::List(
                edges
                  .iter()
                  .map(|&(a, b)| call("UndirectedEdge", vec![int(a), int(b)]))
                  .collect(),
              ),
            ],
          )
        };
        let per_graph: Result<Vec<Expr>, InterpreterError> = graphs
          .iter()
          .map(|(n, edges)| match element {
            "EdgeRules" => Ok(Expr::List(
              edges
                .iter()
                .map(|&(a, b)| rule_expr(int(a), int(b)))
                .collect(),
            )),
            "VertexCount" => Ok(int(*n)),
            "VertexList" => Ok(vertices(*n)),
            "AdjacencyMatrix" => {
              let mut rows = vec![vec![Expr::Integer(0); *n]; *n];
              for &(a, b) in edges {
                rows[a - 1][b - 1] = Expr::Integer(1);
                rows[b - 1][a - 1] = Expr::Integer(1);
              }
              Ok(Expr::List(
                rows.into_iter().map(|r| Expr::List(r.into())).collect(),
              ))
            }
            _ => graph(*n, edges),
          })
          .collect();
        let per_graph = match per_graph {
          Ok(list) => list,
          Err(e) => return Some(Err(e)),
        };
        return Some(Ok(match element {
          // The drawings are not produced; leave the request as written.
          "Graphics" | "GraphicsList" | "Summary" => {
            unevaluated("ImportString", args)
          }
          // A single graph comes back on its own, several as a list.
          "Graph" if per_graph.len() == 1 => {
            per_graph.into_iter().next().unwrap()
          }
          _ => Expr::List(per_graph.into()),
        }));
      }

      if format != "CSV" {
        return Some(Ok(unevaluated("ImportString", args)));
      }

      if content.is_empty() {
        crate::emit_message(
          "Import::fmterr: Cannot import data as CSV format.",
        );
        return Some(Ok(fail_expr()));
      }
      if let Some(element) = &requested_element
        && !crate::functions::csv_ast::is_csv_element(element)
      {
        return unknown_element(element, "CSV");
      }
      let rows = crate::functions::csv_ast::parse_csv(&content);
      return Some(Ok(crate::functions::csv_ast::csv_import_element(
        &rows,
        table_element,
      )));
    }
    _ => {}
  }
  None
}

/// The `"HTML"` named-colour scheme: the CSS colour names with the exact
/// values wolframscript reports. A Demonstrations palette reaches for these
/// by name (`ColorData["HTML", "SlateBlue"]`), and the lookup ignores case
/// the way `ColorData` does.
#[rustfmt::skip]
const HTML_COLORS: &[(&str, f64, f64, f64)] = &[
  ("AliceBlue", 0.9411764705882353, 0.9725490196078431, 1.0),
  ("AntiqueWhite", 0.9803921568627451, 0.9215686274509803, 0.8431372549019608),
  ("Aqua", 0.0, 1.0, 1.0),
  ("Aquamarine", 0.4980392156862745, 1.0, 0.8313725490196078),
  ("Azure", 0.9411764705882353, 1.0, 1.0),
  ("Beige", 0.9607843137254902, 0.9607843137254902, 0.8627450980392157),
  ("Bisque", 1.0, 0.8941176470588235, 0.7686274509803921),
  ("Black", 0.0, 0.0, 0.0),
  ("BlanchedAlmond", 1.0, 0.9215686274509803, 0.803921568627451),
  ("Blue", 0.0, 0.0, 1.0),
  ("BlueViolet", 0.5411764705882353, 0.16862745098039217, 0.8862745098039215),
  ("Brown", 0.6470588235294118, 0.16470588235294117, 0.16470588235294117),
  ("BurlyWood", 0.8705882352941177, 0.7215686274509804, 0.5294117647058824),
  ("CadetBlue", 0.37254901960784315, 0.6196078431372549, 0.6274509803921569),
  ("Chartreuse", 0.4980392156862745, 1.0, 0.0),
  ("Chocolate", 0.8235294117647058, 0.4117647058823529, 0.11764705882352941),
  ("Coral", 1.0, 0.4980392156862745, 0.3137254901960784),
  ("CornflowerBlue", 0.39215686274509803, 0.5843137254901961, 0.9294117647058824),
  ("Cornsilk", 1.0, 0.9725490196078431, 0.8627450980392157),
  ("Crimson", 0.8627450980392157, 0.0784313725490196, 0.23529411764705882),
  ("Cyan", 0.0, 1.0, 1.0),
  ("DarkBlue", 0.0, 0.0, 0.5450980392156862),
  ("DarkCyan", 0.0, 0.5450980392156862, 0.5450980392156862),
  ("DarkGoldenrod", 0.7215686274509804, 0.5254901960784314, 0.043137254901960784),
  ("DarkGray", 0.6627450980392157, 0.6627450980392157, 0.6627450980392157),
  ("DarkGreen", 0.0, 0.39215686274509803, 0.0),
  ("DarkKhaki", 0.7411764705882353, 0.7176470588235294, 0.4196078431372549),
  ("DarkMagenta", 0.5450980392156862, 0.0, 0.5450980392156862),
  ("DarkOliveGreen", 0.3333333333333333, 0.4196078431372549, 0.1843137254901961),
  ("DarkOrange", 1.0, 0.5490196078431373, 0.0),
  ("DarkOrchid", 0.6, 0.19607843137254902, 0.8),
  ("DarkRed", 0.5450980392156862, 0.0, 0.0),
  ("DarkSalmon", 0.9137254901960784, 0.5882352941176471, 0.4784313725490196),
  ("DarkSeaGreen", 0.5607843137254902, 0.7372549019607844, 0.5607843137254902),
  ("DarkSlateBlue", 0.2823529411764706, 0.2392156862745098, 0.5450980392156862),
  ("DarkSlateGray", 0.1843137254901961, 0.30980392156862746, 0.30980392156862746),
  ("DarkTurquoise", 0.0, 0.807843137254902, 0.8196078431372549),
  ("DarkViolet", 0.580392156862745, 0.0, 0.8274509803921568),
  ("DeepPink", 1.0, 0.0784313725490196, 0.5764705882352941),
  ("DeepSkyBlue", 0.0, 0.7490196078431373, 1.0),
  ("DimGray", 0.4117647058823529, 0.4117647058823529, 0.4117647058823529),
  ("DodgerBlue", 0.11764705882352941, 0.5647058823529412, 1.0),
  ("FireBrick", 0.6980392156862745, 0.13333333333333333, 0.13333333333333333),
  ("FloralWhite", 1.0, 0.9803921568627451, 0.9411764705882353),
  ("ForestGreen", 0.13333333333333333, 0.5450980392156862, 0.13333333333333333),
  ("Fuchsia", 1.0, 0.0, 1.0),
  ("Gainsboro", 0.8627450980392157, 0.8627450980392157, 0.8627450980392157),
  ("GhostWhite", 0.9725490196078431, 0.9725490196078431, 1.0),
  ("Gold", 1.0, 0.8431372549019608, 0.0),
  ("Goldenrod", 0.8549019607843137, 0.6470588235294118, 0.12549019607843137),
  ("Gray", 0.5019607843137255, 0.5019607843137255, 0.5019607843137255),
  ("Green", 0.0, 0.5019607843137255, 0.0),
  ("GreenYellow", 0.6784313725490196, 1.0, 0.1843137254901961),
  ("Honeydew", 0.9411764705882353, 1.0, 0.9411764705882353),
  ("HotPink", 1.0, 0.4117647058823529, 0.7058823529411764),
  ("IndianRed", 0.803921568627451, 0.3607843137254902, 0.3607843137254902),
  ("Indigo", 0.29411764705882354, 0.0, 0.5098039215686274),
  ("Ivory", 1.0, 1.0, 0.9411764705882353),
  ("Khaki", 0.9411764705882353, 0.9019607843137255, 0.5490196078431373),
  ("Lavender", 0.9019607843137255, 0.9019607843137255, 0.9803921568627451),
  ("LavenderBlush", 1.0, 0.9411764705882353, 0.9607843137254902),
  ("LawnGreen", 0.48627450980392156, 0.9882352941176471, 0.0),
  ("LemonChiffon", 1.0, 0.9803921568627451, 0.803921568627451),
  ("LightBlue", 0.6784313725490196, 0.8470588235294118, 0.9019607843137255),
  ("LightCoral", 0.9411764705882353, 0.5019607843137255, 0.5019607843137255),
  ("LightCyan", 0.8784313725490196, 1.0, 1.0),
  ("LightGoldenrodYellow", 0.9803921568627451, 0.9803921568627451, 0.8235294117647058),
  ("LightGray", 0.8274509803921568, 0.8274509803921568, 0.8274509803921568),
  ("LightGreen", 0.5647058823529412, 0.9333333333333333, 0.5647058823529412),
  ("LightPink", 1.0, 0.7137254901960784, 0.7568627450980392),
  ("LightSalmon", 1.0, 0.6274509803921569, 0.4784313725490196),
  ("LightSeaGreen", 0.12549019607843137, 0.6980392156862745, 0.6666666666666666),
  ("LightSkyBlue", 0.5294117647058824, 0.807843137254902, 0.9803921568627451),
  ("LightSlateGray", 0.4666666666666667, 0.5333333333333333, 0.6),
  ("LightSteelBlue", 0.6901960784313725, 0.7686274509803921, 0.8705882352941177),
  ("LightYellow", 1.0, 1.0, 0.8784313725490196),
  ("Lime", 0.0, 1.0, 0.0),
  ("LimeGreen", 0.19607843137254902, 0.803921568627451, 0.19607843137254902),
  ("Linen", 0.9803921568627451, 0.9411764705882353, 0.9019607843137255),
  ("Magenta", 1.0, 0.0, 1.0),
  ("Maroon", 0.5019607843137255, 0.0, 0.0),
  ("MediumAquamarine", 0.4, 0.803921568627451, 0.6666666666666666),
  ("MediumBlue", 0.0, 0.0, 0.803921568627451),
  ("MediumOrchid", 0.7294117647058823, 0.3333333333333333, 0.8274509803921568),
  ("MediumPurple", 0.5764705882352941, 0.4392156862745098, 0.8588235294117647),
  ("MediumSeaGreen", 0.23529411764705882, 0.7019607843137254, 0.44313725490196076),
  ("MediumSlateBlue", 0.4823529411764706, 0.40784313725490196, 0.9333333333333333),
  ("MediumSpringGreen", 0.0, 0.9803921568627451, 0.6039215686274509),
  ("MediumTurquoise", 0.2823529411764706, 0.8196078431372549, 0.8),
  ("MediumVioletRed", 0.7803921568627451, 0.08235294117647059, 0.5215686274509804),
  ("MidnightBlue", 0.09803921568627451, 0.09803921568627451, 0.4392156862745098),
  ("MintCream", 0.9607843137254902, 1.0, 0.9803921568627451),
  ("MistyRose", 1.0, 0.8941176470588235, 0.8823529411764706),
  ("Moccasin", 1.0, 0.8941176470588235, 0.7098039215686275),
  ("NavajoWhite", 1.0, 0.8705882352941177, 0.6784313725490196),
  ("Navy", 0.0, 0.0, 0.5019607843137255),
  ("OldLace", 0.9921568627450981, 0.9607843137254902, 0.9019607843137255),
  ("Olive", 0.5019607843137255, 0.5019607843137255, 0.0),
  ("OliveDrab", 0.4196078431372549, 0.5568627450980392, 0.13725490196078433),
  ("Orange", 1.0, 0.6470588235294118, 0.0),
  ("OrangeRed", 1.0, 0.27058823529411763, 0.0),
  ("Orchid", 0.8549019607843137, 0.4392156862745098, 0.8392156862745098),
  ("PaleGoldenrod", 0.9333333333333333, 0.9098039215686274, 0.6666666666666666),
  ("PaleGreen", 0.596078431372549, 0.984313725490196, 0.596078431372549),
  ("PaleTurquoise", 0.6862745098039216, 0.9333333333333333, 0.9333333333333333),
  ("PaleVioletRed", 0.8588235294117647, 0.4392156862745098, 0.5764705882352941),
  ("PapayaWhip", 1.0, 0.9372549019607843, 0.8352941176470589),
  ("PeachPuff", 1.0, 0.8549019607843137, 0.7254901960784313),
  ("Peru", 0.803921568627451, 0.5215686274509804, 0.24705882352941178),
  ("Pink", 1.0, 0.7529411764705882, 0.796078431372549),
  ("Plum", 0.8666666666666667, 0.6274509803921569, 0.8666666666666667),
  ("PowderBlue", 0.6901960784313725, 0.8784313725490196, 0.9019607843137255),
  ("Purple", 0.5019607843137255, 0.0, 0.5019607843137255),
  ("Red", 1.0, 0.0, 0.0),
  ("RosyBrown", 0.7372549019607844, 0.5607843137254902, 0.5607843137254902),
  ("RoyalBlue", 0.2549019607843137, 0.4117647058823529, 0.8823529411764706),
  ("SaddleBrown", 0.5450980392156862, 0.27058823529411763, 0.07450980392156863),
  ("Salmon", 0.9803921568627451, 0.5019607843137255, 0.44705882352941173),
  ("SandyBrown", 0.9568627450980391, 0.6431372549019607, 0.3764705882352941),
  ("SeaGreen", 0.1803921568627451, 0.5450980392156862, 0.3411764705882353),
  ("Seashell", 1.0, 0.9607843137254902, 0.9333333333333333),
  ("Sienna", 0.6274509803921569, 0.32156862745098036, 0.1764705882352941),
  ("Silver", 0.7529411764705882, 0.7529411764705882, 0.7529411764705882),
  ("SkyBlue", 0.5294117647058824, 0.807843137254902, 0.9215686274509803),
  ("SlateBlue", 0.4156862745098039, 0.3529411764705882, 0.803921568627451),
  ("SlateGray", 0.4392156862745098, 0.5019607843137255, 0.5647058823529412),
  ("Snow", 1.0, 0.9803921568627451, 0.9803921568627451),
  ("SpringGreen", 0.0, 1.0, 0.4980392156862745),
  ("SteelBlue", 0.27450980392156865, 0.5098039215686274, 0.7058823529411764),
  ("Tan", 0.8235294117647058, 0.7058823529411764, 0.5490196078431373),
  ("Teal", 0.0, 0.5019607843137255, 0.5019607843137255),
  ("Thistle", 0.8470588235294118, 0.7490196078431373, 0.8470588235294118),
  ("Tomato", 1.0, 0.38823529411764707, 0.2784313725490196),
  ("Turquoise", 0.25098039215686274, 0.8784313725490196, 0.8156862745098039),
  ("Violet", 0.9333333333333333, 0.5098039215686274, 0.9333333333333333),
  ("Wheat", 0.9607843137254902, 0.8705882352941177, 0.7019607843137254),
  ("White", 1.0, 1.0, 1.0),
  ("WhiteSmoke", 0.9607843137254902, 0.9607843137254902, 0.9607843137254902),
  ("Yellow", 1.0, 1.0, 0.0),
  ("YellowGreen", 0.6039215686274509, 0.803921568627451, 0.19607843137254902),
];

/// The `"Crayola"` named-colour scheme, with the values wolframscript reports.
#[rustfmt::skip]
const CRAYOLA_COLORS: &[(&str, f64, f64, f64)] = &[
  ("Almond", 0.9372549019607843, 0.8705882352941177, 0.803921568627451),
  ("AntiqueBrass", 0.803921568627451, 0.5843137254901961, 0.4588235294117647),
  ("Apricot", 0.9921568627450981, 0.8509803921568627, 0.7098039215686275),
  ("Aquamarine", 0.47058823529411764, 0.8588235294117647, 0.8862745098039215),
  ("Asparagus", 0.5294117647058824, 0.6627450980392157, 0.4196078431372549),
  ("AtomicTangerine", 1.0, 0.6431372549019608, 0.4549019607843137),
  ("BananaMania", 0.9803921568627451, 0.9058823529411765, 0.7098039215686275),
  ("Beaver", 0.6235294117647059, 0.5058823529411764, 0.4392156862745098),
  ("Bittersweet", 0.9921568627450981, 0.48627450980392156, 0.43137254901960786),
  ("Black", 0.0, 0.0, 0.0),
  ("BlizzardBlue", 0.6745098039215687, 0.8980392156862745, 0.9333333333333333),
  ("Blue", 0.12156862745098039, 0.4588235294117647, 0.996078431372549),
  ("BlueBell", 0.6352941176470588, 0.6352941176470588, 0.8156862745098039),
  ("BlueGray", 0.4, 0.6, 0.8),
  ("BlueGreen", 0.050980392156862744, 0.596078431372549, 0.7294117647058823),
  ("BlueViolet", 0.45098039215686275, 0.4, 0.7411764705882353),
  ("Blush", 0.8705882352941177, 0.36470588235294116, 0.5137254901960784),
  ("BrickRed", 0.796078431372549, 0.2549019607843137, 0.32941176470588235),
  ("Brown", 0.7058823529411765, 0.403921568627451, 0.30196078431372547),
  ("BurntOrange", 1.0, 0.4980392156862745, 0.28627450980392155),
  ("BurntSienna", 0.9176470588235294, 0.49411764705882355, 0.36470588235294116),
  ("CadetBlue", 0.6901960784313725, 0.7176470588235294, 0.7764705882352941),
  ("Canary", 1.0, 1.0, 0.6),
  ("CaribbeanGreen", 0.10980392156862745, 0.8274509803921568, 0.6352941176470588),
  ("CarnationPink", 1.0, 0.6666666666666666, 0.8),
  ("Cerise", 0.8666666666666667, 0.26666666666666666, 0.5725490196078431),
  ("Cerulean", 0.11372549019607843, 0.6745098039215687, 0.8392156862745098),
  ("Chestnut", 0.7372549019607844, 0.36470588235294116, 0.34509803921568627),
  ("Copper", 0.8666666666666667, 0.5803921568627451, 0.4588235294117647),
  ("Cornflower", 0.6039215686274509, 0.807843137254902, 0.9215686274509803),
  ("CottonCandy", 1.0, 0.7372549019607844, 0.8509803921568627),
  ("Dandelion", 0.9921568627450981, 0.8588235294117647, 0.42745098039215684),
  ("Denim", 0.16862745098039217, 0.4235294117647059, 0.7686274509803922),
  ("DesertSand", 0.9372549019607843, 0.803921568627451, 0.7215686274509804),
  ("Eggplant", 0.43137254901960786, 0.3176470588235294, 0.3764705882352941),
  ("ElectricLime", 0.807843137254902, 1.0, 0.11372549019607843),
  ("Fern", 0.44313725490196076, 0.7372549019607844, 0.47058823529411764),
  ("ForestGreen", 0.42745098039215684, 0.6823529411764706, 0.5058823529411764),
  ("Fuchsia", 0.7647058823529411, 0.39215686274509803, 0.7725490196078432),
  ("FuzzyWuzzy", 0.8, 0.4, 0.4),
  ("Gold", 0.9058823529411765, 0.7764705882352941, 0.592156862745098),
  ("Goldenrod", 0.9882352941176471, 0.8509803921568627, 0.4588235294117647),
  ("GrannySmithApple", 0.6588235294117647, 0.8941176470588236, 0.6274509803921569),
  ("Gray", 0.5843137254901961, 0.5686274509803921, 0.5490196078431373),
  ("Green", 0.10980392156862745, 0.6745098039215687, 0.47058823529411764),
  ("GreenBlue", 0.06666666666666667, 0.39215686274509803, 0.7058823529411765),
  ("GreenYellow", 0.9411764705882353, 0.9098039215686274, 0.5686274509803921),
  ("HotMagenta", 0.9725490196078431, 0.09019607843137255, 0.24313725490196078),
  ("Inchworm", 0.6980392156862745, 0.9254901960784314, 0.36470588235294116),
  ("Indigo", 0.36470588235294116, 0.4627450980392157, 0.796078431372549),
  ("JazzberryJam", 0.792156862745098, 0.21568627450980393, 0.403921568627451),
  ("JungleGreen", 0.23137254901960785, 0.6901960784313725, 0.5607843137254902),
  ("LaserLemon", 0.996078431372549, 0.996078431372549, 0.13333333333333333),
  ("Lavender", 0.9882352941176471, 0.7058823529411765, 0.8352941176470589),
  ("LemonYellow", 1.0, 0.9568627450980393, 0.30980392156862746),
  ("MacaroniandCheese", 1.0, 0.7411764705882353, 0.5333333333333333),
  ("Magenta", 0.9647058823529412, 0.39215686274509803, 0.6862745098039216),
  ("MagicMint", 0.6666666666666666, 0.9411764705882353, 0.8196078431372549),
  ("Mahogany", 0.803921568627451, 0.2901960784313726, 0.2980392156862745),
  ("Maize", 0.9294117647058824, 0.8196078431372549, 0.611764705882353),
  ("Manatee", 0.592156862745098, 0.6039215686274509, 0.6666666666666666),
  ("MangoTango", 1.0, 0.5098039215686274, 0.2627450980392157),
  ("Maroon", 0.7843137254901961, 0.2196078431372549, 0.35294117647058826),
  ("Mauvelous", 0.9372549019607843, 0.596078431372549, 0.6666666666666666),
  ("Melon", 0.9921568627450981, 0.7372549019607844, 0.7058823529411765),
  ("MidnightBlue", 0.10196078431372549, 0.2823529411764706, 0.4627450980392157),
  ("MountainMeadow", 0.18823529411764706, 0.7294117647058823, 0.5607843137254902),
  ("Mulberry", 0.7725490196078432, 0.29411764705882354, 0.5490196078431373),
  ("NavyBlue", 0.09803921568627451, 0.4549019607843137, 0.8235294117647058),
  ("NeonCarrot", 1.0, 0.6392156862745098, 0.2627450980392157),
  ("OliveGreen", 0.7294117647058823, 0.7215686274509804, 0.4235294117647059),
  ("Orange", 1.0, 0.4588235294117647, 0.2196078431372549),
  ("OrangeRed", 1.0, 0.16862745098039217, 0.16862745098039217),
  ("OrangeYellow", 0.9725490196078431, 0.8352941176470589, 0.40784313725490196),
  ("Orchid", 0.9019607843137255, 0.6588235294117647, 0.8431372549019608),
  ("OuterSpace", 0.2549019607843137, 0.2901960784313726, 0.2980392156862745),
  ("OutrageousOrange", 1.0, 0.43137254901960786, 0.2901960784313726),
  ("PacificBlue", 0.10980392156862745, 0.6627450980392157, 0.788235294117647),
  ("Peach", 1.0, 0.8117647058823529, 0.6705882352941176),
  ("Periwinkle", 0.7725490196078432, 0.8156862745098039, 0.9019607843137255),
  ("PiggyPink", 0.9921568627450981, 0.8666666666666667, 0.9019607843137255),
  ("PineGreen", 0.08235294117647059, 0.5019607843137255, 0.47058823529411764),
  ("PinkFlamingo", 0.9882352941176471, 0.4549019607843137, 0.9921568627450981),
  ("PinkSherbert", 0.9686274509803922, 0.5607843137254902, 0.6549019607843137),
  ("Plum", 0.5568627450980392, 0.27058823529411763, 0.5215686274509804),
  ("Purple", 0.5725490196078431, 0.43137254901960786, 0.6823529411764706),
  ("PurpleHeart", 0.4549019607843137, 0.25882352941176473, 0.7843137254901961),
  ("PurpleMountainsMajesty", 0.615686274509804, 0.5058823529411764, 0.7294117647058823),
  ("PurplePizzazz", 1.0, 0.0, 0.8),
  ("RadicalRed", 1.0, 0.28627450980392155, 0.4235294117647059),
  ("RawSienna", 0.8392156862745098, 0.5411764705882353, 0.34901960784313724),
  ("RawUmber", 0.44313725490196076, 0.29411764705882354, 0.13725490196078433),
  ("RazzleDazzleRose", 1.0, 0.2823529411764706, 0.8156862745098039),
  ("Razzmatazz", 0.8901960784313725, 0.1450980392156863, 0.4196078431372549),
  ("Red", 0.9333333333333333, 0.12549019607843137, 0.30196078431372547),
  ("RedOrange", 1.0, 0.3254901960784314, 0.28627450980392155),
  ("RedViolet", 0.7529411764705882, 0.26666666666666666, 0.5607843137254902),
  ("RobinsEggBlue", 0.12156862745098039, 0.807843137254902, 0.796078431372549),
  ("RoyalPurple", 0.47058823529411764, 0.3176470588235294, 0.6627450980392157),
  ("Salmon", 1.0, 0.6078431372549019, 0.6666666666666666),
  ("Scarlet", 0.9882352941176471, 0.1568627450980392, 0.2784313725490196),
  ("ScreaminGreen", 0.4627450980392157, 1.0, 0.47843137254901963),
  ("SeaGreen", 0.6235294117647059, 0.8862745098039215, 0.7490196078431373),
  ("Sepia", 0.6470588235294118, 0.4117647058823529, 0.30980392156862746),
  ("Shadow", 0.5411764705882353, 0.4745098039215686, 0.36470588235294116),
  ("Shamrock", 0.27058823529411763, 0.807843137254902, 0.6352941176470588),
  ("ShockingPink", 0.984313725490196, 0.49411764705882355, 0.9921568627450981),
  ("Silver", 0.803921568627451, 0.7725490196078432, 0.7607843137254902),
  ("SkyBlue", 0.5019607843137255, 0.8549019607843137, 0.9215686274509803),
  ("SpringGreen", 0.9254901960784314, 0.9176470588235294, 0.7450980392156863),
  ("Sunglow", 1.0, 0.8117647058823529, 0.2823529411764706),
  ("SunsetOrange", 0.9921568627450981, 0.3686274509803922, 0.3254901960784314),
  ("Tan", 0.9803921568627451, 0.6549019607843137, 0.4235294117647059),
  ("TealBlue", 0.09411764705882353, 0.6549019607843137, 0.7098039215686275),
  ("Thistle", 0.9215686274509803, 0.7803921568627451, 0.8745098039215686),
  ("TickleMePink", 0.9882352941176471, 0.5372549019607843, 0.6745098039215687),
  ("Timberwolf", 0.8588235294117647, 0.8431372549019608, 0.8235294117647058),
  ("TropicalRainForest", 0.09019607843137255, 0.5019607843137255, 0.42745098039215684),
  ("Tumbleweed", 0.8705882352941177, 0.6666666666666666, 0.5333333333333333),
  ("TurquoiseBlue", 0.4666666666666667, 0.8666666666666667, 0.9058823529411765),
  ("UnmellowYellow", 1.0, 1.0, 0.4),
  ("Violet", 0.5725490196078431, 0.43137254901960786, 0.6823529411764706),
  ("VioletBlue", 0.19607843137254902, 0.2901960784313726, 0.6980392156862745),
  ("VioletRed", 0.9686274509803922, 0.3254901960784314, 0.5803921568627451),
  ("VividTangerine", 1.0, 0.6274509803921569, 0.5372549019607843),
  ("VividViolet", 0.5607843137254902, 0.3137254901960784, 0.615686274509804),
  ("White", 1.0, 1.0, 1.0),
  ("WildBlueYonder", 0.6352941176470588, 0.6784313725490196, 0.8156862745098039),
  ("WildStrawberry", 1.0, 0.2627450980392157, 0.6431372549019608),
  ("WildWatermelon", 0.9882352941176471, 0.4235294117647059, 0.5215686274509804),
  ("Wisteria", 0.803921568627451, 0.6431372549019608, 0.8705882352941177),
  ("Yellow", 0.9882352941176471, 0.9098039215686274, 0.5137254901960784),
  ("YellowGreen", 0.7725490196078432, 0.8901960784313725, 0.5176470588235295),
  ("YellowOrange", 1.0, 0.6823529411764706, 0.25882352941176473),
];

/// The `"WebSafe"` named-colour scheme: the 216 `#RRGGBB` web-safe colours, in wolframscript's order.
#[rustfmt::skip]
const WEBSAFE_COLORS: &[(&str, f64, f64, f64)] = &[
  ("#FFFFFF", 1.0, 1.0, 1.0),
  ("#FFFFCC", 1.0, 1.0, 0.8),
  ("#FFFF99", 1.0, 1.0, 0.6),
  ("#FFFF66", 1.0, 1.0, 0.4),
  ("#FFFF33", 1.0, 1.0, 0.2),
  ("#FFFF00", 1.0, 1.0, 0.0),
  ("#FFCCFF", 1.0, 0.8, 1.0),
  ("#FFCCCC", 1.0, 0.8, 0.8),
  ("#FFCC99", 1.0, 0.8, 0.6),
  ("#FFCC66", 1.0, 0.8, 0.4),
  ("#FFCC33", 1.0, 0.8, 0.2),
  ("#FFCC00", 1.0, 0.8, 0.0),
  ("#FF99FF", 1.0, 0.6, 1.0),
  ("#FF99CC", 1.0, 0.6, 0.8),
  ("#FF9999", 1.0, 0.6, 0.6),
  ("#FF9966", 1.0, 0.6, 0.4),
  ("#FF9933", 1.0, 0.6, 0.2),
  ("#FF9900", 1.0, 0.6, 0.0),
  ("#FF66FF", 1.0, 0.4, 1.0),
  ("#FF66CC", 1.0, 0.4, 0.8),
  ("#FF6699", 1.0, 0.4, 0.6),
  ("#FF6666", 1.0, 0.4, 0.4),
  ("#FF6633", 1.0, 0.4, 0.2),
  ("#FF6600", 1.0, 0.4, 0.0),
  ("#FF33FF", 1.0, 0.2, 1.0),
  ("#FF33CC", 1.0, 0.2, 0.8),
  ("#FF3399", 1.0, 0.2, 0.6),
  ("#FF3366", 1.0, 0.2, 0.4),
  ("#FF3333", 1.0, 0.2, 0.2),
  ("#FF3300", 1.0, 0.2, 0.0),
  ("#FF00FF", 1.0, 0.0, 1.0),
  ("#FF00CC", 1.0, 0.0, 0.8),
  ("#FF0099", 1.0, 0.0, 0.6),
  ("#FF0066", 1.0, 0.0, 0.4),
  ("#FF0033", 1.0, 0.0, 0.2),
  ("#FF0000", 1.0, 0.0, 0.0),
  ("#CCFFFF", 0.8, 1.0, 1.0),
  ("#CCFFCC", 0.8, 1.0, 0.8),
  ("#CCFF99", 0.8, 1.0, 0.6),
  ("#CCFF66", 0.8, 1.0, 0.4),
  ("#CCFF33", 0.8, 1.0, 0.2),
  ("#CCFF00", 0.8, 1.0, 0.0),
  ("#CCCCFF", 0.8, 0.8, 1.0),
  ("#CCCCCC", 0.8, 0.8, 0.8),
  ("#CCCC99", 0.8, 0.8, 0.6),
  ("#CCCC66", 0.8, 0.8, 0.4),
  ("#CCCC33", 0.8, 0.8, 0.2),
  ("#CCCC00", 0.8, 0.8, 0.0),
  ("#CC99FF", 0.8, 0.6, 1.0),
  ("#CC99CC", 0.8, 0.6, 0.8),
  ("#CC9999", 0.8, 0.6, 0.6),
  ("#CC9966", 0.8, 0.6, 0.4),
  ("#CC9933", 0.8, 0.6, 0.2),
  ("#CC9900", 0.8, 0.6, 0.0),
  ("#CC66FF", 0.8, 0.4, 1.0),
  ("#CC66CC", 0.8, 0.4, 0.8),
  ("#CC6699", 0.8, 0.4, 0.6),
  ("#CC6666", 0.8, 0.4, 0.4),
  ("#CC6633", 0.8, 0.4, 0.2),
  ("#CC6600", 0.8, 0.4, 0.0),
  ("#CC33FF", 0.8, 0.2, 1.0),
  ("#CC33CC", 0.8, 0.2, 0.8),
  ("#CC3399", 0.8, 0.2, 0.6),
  ("#CC3366", 0.8, 0.2, 0.4),
  ("#CC3333", 0.8, 0.2, 0.2),
  ("#CC3300", 0.8, 0.2, 0.0),
  ("#CC00FF", 0.8, 0.0, 1.0),
  ("#CC00CC", 0.8, 0.0, 0.8),
  ("#CC0099", 0.8, 0.0, 0.6),
  ("#CC0066", 0.8, 0.0, 0.4),
  ("#CC0033", 0.8, 0.0, 0.2),
  ("#CC0000", 0.8, 0.0, 0.0),
  ("#99FFFF", 0.6, 1.0, 1.0),
  ("#99FFCC", 0.6, 1.0, 0.8),
  ("#99FF99", 0.6, 1.0, 0.6),
  ("#99FF66", 0.6, 1.0, 0.4),
  ("#99FF33", 0.6, 1.0, 0.2),
  ("#99FF00", 0.6, 1.0, 0.0),
  ("#99CCFF", 0.6, 0.8, 1.0),
  ("#99CCCC", 0.6, 0.8, 0.8),
  ("#99CC99", 0.6, 0.8, 0.6),
  ("#99CC66", 0.6, 0.8, 0.4),
  ("#99CC33", 0.6, 0.8, 0.2),
  ("#99CC00", 0.6, 0.8, 0.0),
  ("#9999FF", 0.6, 0.6, 1.0),
  ("#9999CC", 0.6, 0.6, 0.8),
  ("#999999", 0.6, 0.6, 0.6),
  ("#999966", 0.6, 0.6, 0.4),
  ("#999933", 0.6, 0.6, 0.2),
  ("#999900", 0.6, 0.6, 0.0),
  ("#9966FF", 0.6, 0.4, 1.0),
  ("#9966CC", 0.6, 0.4, 0.8),
  ("#996699", 0.6, 0.4, 0.6),
  ("#996666", 0.6, 0.4, 0.4),
  ("#996633", 0.6, 0.4, 0.2),
  ("#996600", 0.6, 0.4, 0.0),
  ("#9933FF", 0.6, 0.2, 1.0),
  ("#9933CC", 0.6, 0.2, 0.8),
  ("#993399", 0.6, 0.2, 0.6),
  ("#993366", 0.6, 0.2, 0.4),
  ("#993333", 0.6, 0.2, 0.2),
  ("#993300", 0.6, 0.2, 0.0),
  ("#9900FF", 0.6, 0.0, 1.0),
  ("#9900CC", 0.6, 0.0, 0.8),
  ("#990099", 0.6, 0.0, 0.6),
  ("#990066", 0.6, 0.0, 0.4),
  ("#990033", 0.6, 0.0, 0.2),
  ("#990000", 0.6, 0.0, 0.0),
  ("#66FFFF", 0.4, 1.0, 1.0),
  ("#66FFCC", 0.4, 1.0, 0.8),
  ("#66FF99", 0.4, 1.0, 0.6),
  ("#66FF66", 0.4, 1.0, 0.4),
  ("#66FF33", 0.4, 1.0, 0.2),
  ("#66FF00", 0.4, 1.0, 0.0),
  ("#66CCFF", 0.4, 0.8, 1.0),
  ("#66CCCC", 0.4, 0.8, 0.8),
  ("#66CC99", 0.4, 0.8, 0.6),
  ("#66CC66", 0.4, 0.8, 0.4),
  ("#66CC33", 0.4, 0.8, 0.2),
  ("#66CC00", 0.4, 0.8, 0.0),
  ("#6699FF", 0.4, 0.6, 1.0),
  ("#6699CC", 0.4, 0.6, 0.8),
  ("#669999", 0.4, 0.6, 0.6),
  ("#669966", 0.4, 0.6, 0.4),
  ("#669933", 0.4, 0.6, 0.2),
  ("#669900", 0.4, 0.6, 0.0),
  ("#6666FF", 0.4, 0.4, 1.0),
  ("#6666CC", 0.4, 0.4, 0.8),
  ("#666699", 0.4, 0.4, 0.6),
  ("#666666", 0.4, 0.4, 0.4),
  ("#666633", 0.4, 0.4, 0.2),
  ("#666600", 0.4, 0.4, 0.0),
  ("#6633FF", 0.4, 0.2, 1.0),
  ("#6633CC", 0.4, 0.2, 0.8),
  ("#663399", 0.4, 0.2, 0.6),
  ("#663366", 0.4, 0.2, 0.4),
  ("#663333", 0.4, 0.2, 0.2),
  ("#663300", 0.4, 0.2, 0.0),
  ("#6600FF", 0.4, 0.0, 1.0),
  ("#6600CC", 0.4, 0.0, 0.8),
  ("#660099", 0.4, 0.0, 0.6),
  ("#660066", 0.4, 0.0, 0.4),
  ("#660033", 0.4, 0.0, 0.2),
  ("#660000", 0.4, 0.0, 0.0),
  ("#33FFFF", 0.2, 1.0, 1.0),
  ("#33FFCC", 0.2, 1.0, 0.8),
  ("#33FF99", 0.2, 1.0, 0.6),
  ("#33FF66", 0.2, 1.0, 0.4),
  ("#33FF33", 0.2, 1.0, 0.2),
  ("#33FF00", 0.2, 1.0, 0.0),
  ("#33CCFF", 0.2, 0.8, 1.0),
  ("#33CCCC", 0.2, 0.8, 0.8),
  ("#33CC99", 0.2, 0.8, 0.6),
  ("#33CC66", 0.2, 0.8, 0.4),
  ("#33CC33", 0.2, 0.8, 0.2),
  ("#33CC00", 0.2, 0.8, 0.0),
  ("#3399FF", 0.2, 0.6, 1.0),
  ("#3399CC", 0.2, 0.6, 0.8),
  ("#339999", 0.2, 0.6, 0.6),
  ("#339966", 0.2, 0.6, 0.4),
  ("#339933", 0.2, 0.6, 0.2),
  ("#339900", 0.2, 0.6, 0.0),
  ("#3366FF", 0.2, 0.4, 1.0),
  ("#3366CC", 0.2, 0.4, 0.8),
  ("#336699", 0.2, 0.4, 0.6),
  ("#336666", 0.2, 0.4, 0.4),
  ("#336633", 0.2, 0.4, 0.2),
  ("#336600", 0.2, 0.4, 0.0),
  ("#3333FF", 0.2, 0.2, 1.0),
  ("#3333CC", 0.2, 0.2, 0.8),
  ("#333399", 0.2, 0.2, 0.6),
  ("#333366", 0.2, 0.2, 0.4),
  ("#333333", 0.2, 0.2, 0.2),
  ("#333300", 0.2, 0.2, 0.0),
  ("#3300FF", 0.2, 0.0, 1.0),
  ("#3300CC", 0.2, 0.0, 0.8),
  ("#330099", 0.2, 0.0, 0.6),
  ("#330066", 0.2, 0.0, 0.4),
  ("#330033", 0.2, 0.0, 0.2),
  ("#330000", 0.2, 0.0, 0.0),
  ("#00FFFF", 0.0, 1.0, 1.0),
  ("#00FFCC", 0.0, 1.0, 0.8),
  ("#00FF99", 0.0, 1.0, 0.6),
  ("#00FF66", 0.0, 1.0, 0.4),
  ("#00FF33", 0.0, 1.0, 0.2),
  ("#00FF00", 0.0, 1.0, 0.0),
  ("#00CCFF", 0.0, 0.8, 1.0),
  ("#00CCCC", 0.0, 0.8, 0.8),
  ("#00CC99", 0.0, 0.8, 0.6),
  ("#00CC66", 0.0, 0.8, 0.4),
  ("#00CC33", 0.0, 0.8, 0.2),
  ("#00CC00", 0.0, 0.8, 0.0),
  ("#0099FF", 0.0, 0.6, 1.0),
  ("#0099CC", 0.0, 0.6, 0.8),
  ("#009999", 0.0, 0.6, 0.6),
  ("#009966", 0.0, 0.6, 0.4),
  ("#009933", 0.0, 0.6, 0.2),
  ("#009900", 0.0, 0.6, 0.0),
  ("#0066FF", 0.0, 0.4, 1.0),
  ("#0066CC", 0.0, 0.4, 0.8),
  ("#006699", 0.0, 0.4, 0.6),
  ("#006666", 0.0, 0.4, 0.4),
  ("#006633", 0.0, 0.4, 0.2),
  ("#006600", 0.0, 0.4, 0.0),
  ("#0033FF", 0.0, 0.2, 1.0),
  ("#0033CC", 0.0, 0.2, 0.8),
  ("#003399", 0.0, 0.2, 0.6),
  ("#003366", 0.0, 0.2, 0.4),
  ("#003333", 0.0, 0.2, 0.2),
  ("#003300", 0.0, 0.2, 0.0),
  ("#0000FF", 0.0, 0.0, 1.0),
  ("#0000CC", 0.0, 0.0, 0.8),
  ("#000099", 0.0, 0.0, 0.6),
  ("#000066", 0.0, 0.0, 0.4),
  ("#000033", 0.0, 0.0, 0.2),
  ("#000000", 0.0, 0.0, 0.0),
];

/// The `"GeologicAges"` named-colour scheme: the standard colours of the geologic time units, as wolframscript reports them (six digits).
#[rustfmt::skip]
const GEOLOGICAGES_COLORS: &[(&str, f64, f64, f64)] = &[
  ("Phanerozoic", 0.690196, 0.886275, 0.819608),
  ("Cenozoic", 1.0, 1.0, 0.0),
  ("Quaternary", 1.0, 1.0, 0.301961),
  ("Holocene", 1.0, 1.0, 0.701961),
  ("Pleistocene", 1.0, 0.921569, 0.384314),
  ("Neogene", 0.992157, 0.8, 0.541176),
  ("Pliocene", 0.996078, 0.921569, 0.67451),
  ("Miocene", 1.0, 0.870588, 0.0),
  ("Paleogene", 1.0, 0.701961, 0.0),
  ("Oligocene", 0.917647, 0.776471, 0.447059),
  ("Eocene", 0.917647, 0.678431, 0.262745),
  ("Paleocene", 0.921569, 0.576471, 0.00392157),
  ("Mesozoic", 0.498039, 0.678431, 0.317647),
  ("Cretaceous", 0.498039, 0.764706, 0.109804),
  ("UpperCretaceous", 0.870588, 0.945098, 0.592157),
  ("LowerCretaceous", 0.701961, 0.87451, 0.498039),
  ("Jurassic", 0.301961, 0.705882, 0.494118),
  ("UpperJurassic", 0.8, 0.921569, 0.772549),
  ("MiddleJurassic", 0.498039, 0.792157, 0.576471),
  ("LowerJurassic", 0.4, 0.752941, 0.572549),
  ("Triassic", 0.403922, 0.764706, 0.717647),
  ("UpperTriassic", 0.8, 0.92549, 0.882353),
  ("MiddleTriassic", 0.6, 0.843137, 0.745098),
  ("LowerTriassic", 0.403922, 0.701961, 0.623529),
  ("Paleozoic", 0.501961, 0.709804, 0.835294),
  ("Permian", 0.403922, 0.776471, 0.866667),
  ("Lopingian", 0.701961, 0.890196, 0.933333),
  ("Guadalupian", 0.6, 0.847059, 0.847059),
  ("Cisuralian", 0.501961, 0.807843, 0.788235),
  ("Carboniferous", 0.6, 0.741176, 0.854902),
  ("Pennsylvanian", 0.407843, 0.623529, 0.792157),
  ("Mississippian", 0.501961, 0.568627, 0.678431),
  ("Devonian", 0.6, 0.6, 0.788235),
  ("UpperDevonian", 0.796078, 0.741176, 0.862745),
  ("MiddleDevonian", 0.6, 0.513725, 0.745098),
  ("LowerDevonian", 0.501961, 0.490196, 0.729412),
  ("Silurian", 0.694118, 0.447059, 0.713725),
  ("Pridoli", 0.913725, 0.780392, 0.886275),
  ("Ludlow", 0.792157, 0.654902, 0.819608),
  ("Wenlock", 0.694118, 0.537255, 0.701961),
  ("Llandovery", 0.596078, 0.345098, 0.658824),
  ("Ordovician", 0.976471, 0.505882, 0.65098),
  ("UpperOrdovician", 0.984314, 0.705882, 0.741176),
  ("MiddleOrdovician", 0.980392, 0.603922, 0.694118),
  ("LowerOrdovician", 0.901961, 0.490196, 0.643137),
  ("Cambrian", 0.984314, 0.501961, 0.372549),
  ("UpperCambrian", 0.992157, 0.803922, 0.721569),
  ("MiddleCambrian", 0.909804, 0.682353, 0.592157),
  ("LowerCambrian", 0.905882, 0.486275, 0.447059),
  ("Precambrian", 0.698039, 0.52549, 0.32549),
  ("Proterozoic", 0.8, 0.847059, 0.568627),
  ("Neoproterozoic", 0.792157, 0.647059, 0.584314),
  ("Ediacaran", 0.917647, 0.847059, 0.737255),
  ("Cryogenian", 0.862745, 0.670588, 0.666667),
  ("Tonian", 0.796078, 0.643137, 0.423529),
  ("Mesoproterozoic", 0.866667, 0.760784, 0.533333),
  ("Stenian", 0.866667, 0.760784, 0.533333),
  ("Ectasian", 0.866667, 0.760784, 0.533333),
  ("Calymmian", 0.866667, 0.760784, 0.533333),
  ("Paleoproterozoic", 0.701961, 0.698039, 0.368627),
  ("Statherian", 0.701961, 0.698039, 0.368627),
  ("Orosirian", 0.701961, 0.698039, 0.368627),
  ("Rhyacian", 0.701961, 0.698039, 0.368627),
  ("Siderian", 0.701961, 0.698039, 0.368627),
  ("Archean", 0.6, 0.678431, 0.67451),
  ("Neoarchean", 0.796078, 0.803922, 0.784314),
  ("Mesoarchean", 0.698039, 0.709804, 0.686275),
  ("Paleoarchean", 0.6, 0.592157, 0.568627),
  ("Eoarchean", 0.501961, 0.564706, 0.564706),
  ("Hadean", 0.5, 0.5, 0.56),
];

/// The machine-number named-colour tables beyond HTML and Legacy.
fn named_color_table(
  scheme: &str,
) -> Option<&'static [(&'static str, f64, f64, f64)]> {
  match scheme {
    "Crayola" => Some(CRAYOLA_COLORS),
    "WebSafe" => Some(WEBSAFE_COLORS),
    "GeologicAges" => Some(GEOLOGICAGES_COLORS),
    _ => None,
  }
}

/// `ColorData["HTML", name]` → the named CSS colour as an `RGBColor`.
fn html_named_color(name: &str) -> Option<(f64, f64, f64)> {
  HTML_COLORS
    .iter()
    .find(|(n, ..)| n.eq_ignore_ascii_case(name))
    .map(|&(_, r, g, b)| (r, g, b))
}

/// The `"Legacy"` named-colour scheme. Despite the overlapping names it is
/// *not* the CSS table: it is the old `Graphics`Colors`` package's palette,
/// whose channels are stored to six digits and miss the exact `n/255` values
/// by ~1e-5 (`Wheat` is `0.960799`, not `245/255 = 0.960784`). It also carries
/// ~50 artists'-pigment names the CSS table has no entry for (`AlizarinCrimson`,
/// `TerreVerte`, …) and lacks ~50 CSS ones (`Tan`, `Teal`, `WhiteSmoke`, …),
/// so the two schemes need separate tables.
///
/// The trailing mask says which channels wolframscript keeps *exact*: bit 0 is
/// red, bit 1 green, bit 2 blue. It is per entry rather than per value —
/// `Orange` is `RGBColor[1, 0.5, 0]` while `Chartreuse` is
/// `RGBColor[0.498001, 1., 0.]` with the very same 1 and 0 inexact.
const LEGACY_COLORS: &[(&str, f64, f64, f64, u8)] = &[
  ("AliceBlue", 0.941206, 0.972503, 1.0, 0b000),
  ("AlizarinCrimson", 0.889996, 0.149998, 0.209998, 0b000),
  ("Antique", 0.980575, 0.92233, 0.844661, 0b000),
  ("Apricot", 1.0, 0.340007, 0.129994, 0b000),
  ("Aquamarine", 0.498001, 1.0, 0.831401, 0b000),
  ("AureolineYellow", 1.0, 0.659993, 0.140004, 0b000),
  ("Azure", 0.94174, 1.0, 1.0, 0b000),
  ("Banana", 0.889996, 0.810007, 0.340007, 0b000),
  ("Beige", 0.640004, 0.580004, 0.5, 0b000),
  ("Bisque", 1.0, 0.898053, 0.771844, 0b000),
  ("Black", 0.0, 0.0, 0.0, 0b111),
  ("BlanchedAlmond", 1.0, 0.921598, 0.803903, 0b000),
  ("Blue", 0.0, 0.0, 1.0, 0b111),
  ("BlueViolet", 0.5412, 0.1686, 0.886303, 0b000),
  ("Brick", 0.610004, 0.400006, 0.119999, 0b000),
  ("Brown", 0.6, 0.4, 0.2, 0b000),
  ("BrownMadder", 0.859996, 0.159993, 0.159993, 0b000),
  ("BrownOchre", 0.53, 0.260002, 0.119999, 0b000),
  ("Burlywood", 0.870602, 0.721595, 0.529405, 0b000),
  ("BurntSienna", 0.539994, 0.209998, 0.059999, 0b000),
  ("BurntUmber", 0.539994, 0.200003, 0.140004, 0b000),
  ("CadetBlue", 0.372494, 0.619602, 0.627506, 0b000),
  ("CadmiumLemon", 1.0, 0.889996, 0.009995, 0b000),
  ("CadmiumOrange", 1.0, 0.380001, 0.009995, 0b000),
  ("CadmiumYellow", 1.0, 0.599994, 0.069994, 0b000),
  ("Carrot", 0.930006, 0.569994, 0.129994, 0b000),
  ("Cerulean", 0.020005, 0.719993, 0.799997, 0b000),
  ("Chartreuse", 0.498001, 1.0, 0.0, 0b000),
  ("Chocolate", 0.823496, 0.411802, 0.117603, 0b000),
  ("ChromeOxideGreen", 0.400006, 0.5, 0.080004, 0b000),
  ("CinnabarGreen", 0.380001, 0.700003, 0.159993, 0b000),
  ("Cobalt", 0.239998, 0.350002, 0.670003, 0b000),
  ("CobaltGreen", 0.239998, 0.569994, 0.250008, 0b000),
  ("ColdGray", 0.5, 0.539994, 0.53, 0b000),
  ("Coral", 1.0, 0.498001, 0.3137, 0b000),
  ("CornflowerBlue", 0.392193, 0.584307, 0.929395, 0b000),
  ("Cornsilk", 1.0, 0.972503, 0.862697, 0b000),
  ("Cyan", 0.0, 1.0, 1.0, 0b111),
  ("CyanWhite", 0.878399, 1.0, 1.0, 0b000),
  ("DarkGoldenrod", 0.721595, 0.525498, 0.043107, 0b000),
  ("DarkGreen", 0.0, 0.392193, 0.0, 0b000),
  ("DarkKhaki", 0.741203, 0.717597, 0.419599, 0b000),
  ("DarkOliveGreen", 0.333293, 0.419599, 0.184301, 0b000),
  ("DarkOrange", 1.0, 0.548997, 0.0, 0b000),
  ("DarkOrchid", 0.599994, 0.196097, 0.799997, 0b000),
  ("DarkSeaGreen", 0.560793, 0.737297, 0.560793, 0b000),
  ("DarkSlateBlue", 0.282403, 0.239204, 0.545106, 0b000),
  ("DarkSlateGray", 0.184301, 0.309793, 0.309793, 0b000),
  ("DarkTurquoise", 0.0, 0.807794, 0.819605, 0b000),
  ("DarkViolet", 0.580401, 0.0, 0.827494, 0b000),
  ("DeepCadmiumRed", 0.889996, 0.089999, 0.050005, 0b000),
  ("DeepCobaltViolet", 0.569994, 0.129994, 0.619999, 0b000),
  ("DeepMadderLake", 0.889996, 0.179998, 0.189993, 0b000),
  ("DeepNaplesYellow", 1.0, 0.659993, 0.069994, 0b000),
  ("DeepOchre", 0.449995, 0.239998, 0.099994, 0b000),
  ("DeepPink", 1.0, 0.078402, 0.576495, 0b000),
  ("DeepSkyBlue", 0.0, 0.749, 1.0, 0b000),
  ("DimGray", 0.333333, 0.333333, 0.333333, 0b000),
  ("DodgerBlue", 0.117603, 0.564699, 1.0, 0b000),
  ("Eggshell", 0.990005, 0.900006, 0.790002, 0b000),
  ("EmeraldGreen", 0.0, 0.790002, 0.340007, 0b000),
  ("EnglishRed", 0.829997, 0.239998, 0.099994, 0b000),
  ("Firebrick", 0.698004, 0.133305, 0.133305, 0b000),
  ("Floral", 1.0, 0.980407, 0.941206, 0b000),
  ("ForestGreen", 0.133305, 0.545106, 0.133305, 0b000),
  ("Gainsboro", 0.862697, 0.862697, 0.862697, 0b000),
  ("GeraniumLake", 0.889996, 0.069994, 0.189993, 0b000),
  ("Ghost", 0.972503, 0.972503, 1.0, 0b000),
  ("Gold", 1.0, 0.843104, 0.0, 0b000),
  ("Goldenrod", 0.8549, 0.647099, 0.125507, 0b000),
  ("GoldOchre", 0.780007, 0.47, 0.149998, 0b000),
  ("Gray", 0.5, 0.5, 0.5, 0b000),
  ("Green", 0.0, 1.0, 0.0, 0b111),
  ("GreenishUmber", 1.0, 0.239998, 0.050005, 0b000),
  ("GreenYellow", 0.678396, 1.0, 0.184301, 0b000),
  ("Honeydew", 0.941206, 1.0, 0.941206, 0b000),
  ("HotPink", 1.0, 0.411802, 0.705893, 0b000),
  ("IndianRed", 0.689993, 0.089999, 0.119999, 0b000),
  ("Indigo", 0.03, 0.179998, 0.329997, 0b000),
  ("Ivory", 1.0, 1.0, 0.941206, 0b000),
  ("IvoryBlack", 0.159993, 0.140004, 0.129994, 0b000),
  ("Khaki", 0.941206, 0.902005, 0.548997, 0b000),
  ("LampBlack", 0.179998, 0.280007, 0.230003, 0b000),
  ("Lavender", 0.902005, 0.902005, 0.980407, 0b000),
  ("LavenderBlush", 1.0, 0.941206, 0.960799, 0b000),
  ("LawnGreen", 0.486297, 0.988205, 0.0, 0b000),
  ("LemonChiffon", 1.0, 0.980407, 0.803903, 0b000),
  ("LightBeige", 0.960799, 0.960799, 0.862697, 0b000),
  ("LightBlue", 0.678396, 0.847102, 0.902005, 0b000),
  ("LightCadmiumRed", 1.0, 0.009995, 0.050005, 0b000),
  ("LightCadmiumYellow", 1.0, 0.689993, 0.059999, 0b000),
  ("LightCoral", 0.941206, 0.501999, 0.501999, 0b000),
  ("LightGoldenrod", 0.933302, 0.866695, 0.509796, 0b000),
  ("LightGray", 0.666666, 0.666666, 0.666666, 0b000),
  ("LightPink", 1.0, 0.713706, 0.756905, 0b000),
  ("LightSalmon", 1.0, 0.627506, 0.478393, 0b000),
  ("LightSeaGreen", 0.125507, 0.698004, 0.666707, 0b000),
  ("LightSkyBlue", 0.529405, 0.807794, 0.980407, 0b000),
  ("LightSlateBlue", 0.517594, 0.439207, 1.0, 0b000),
  ("LightSlateGray", 0.466704, 0.533296, 0.599994, 0b000),
  ("LightSteelBlue", 0.690207, 0.768593, 0.870602, 0b000),
  ("LightViridian", 0.430006, 1.0, 0.440001, 0b000),
  ("LightYellow", 1.0, 1.0, 0.878399, 0b000),
  ("LimeGreen", 0.196097, 0.803903, 0.196097, 0b000),
  ("Linen", 0.980407, 0.941206, 0.902005, 0b000),
  ("Magenta", 1.0, 0.0, 1.0, 0b111),
  ("ManganeseBlue", 0.009995, 0.659993, 0.619999, 0b000),
  ("Maroon", 0.690207, 0.188192, 0.376507, 0b000),
  ("MarsOrange", 0.589999, 0.269997, 0.080004, 0b000),
  ("MarsYellow", 0.889996, 0.440001, 0.099994, 0b000),
  ("MediumAquamarine", 0.400006, 0.803903, 0.666707, 0b000),
  ("MediumBlue", 0.0, 0.0, 0.803903, 0b000),
  ("MediumOrchid", 0.729408, 0.333293, 0.827494, 0b000),
  ("MediumPurple", 0.576495, 0.439207, 0.858806, 0b000),
  ("MediumSeaGreen", 0.235298, 0.702002, 0.443098, 0b000),
  ("MediumSlateBlue", 0.482406, 0.407804, 0.933302, 0b000),
  ("MediumSpringGreen", 0.0, 0.980407, 0.6039, 0b000),
  ("MediumTurquoise", 0.282403, 0.819605, 0.799997, 0b000),
  ("MediumVioletRed", 0.780404, 0.0824, 0.521607, 0b000),
  ("Melon", 0.889996, 0.659993, 0.410001, 0b000),
  ("MidnightBlue", 0.097995, 0.097995, 0.439207, 0b000),
  ("Mint", 0.739998, 0.990005, 0.790002, 0b000),
  ("MintCream", 0.960799, 1.0, 0.980407, 0b000),
  ("MistyRose", 1.0, 0.894101, 0.882397, 0b000),
  ("Moccasin", 1.0, 0.894101, 0.709799, 0b000),
  ("Navajo", 1.0, 0.870602, 0.678396, 0b000),
  ("Navy", 0.0, 0.0, 0.501999, 0b000),
  ("NavyBlue", 0.0, 0.0, 0.501999, 0b000),
  ("Oak", 1.0, 0.490005, 0.250008, 0b000),
  ("OldLace", 0.992203, 0.960799, 0.902005, 0b000),
  ("Olive", 0.230003, 0.370006, 0.170003, 0b000),
  ("OliveDrab", 0.419599, 0.556902, 0.137303, 0b000),
  ("Orange", 1.0, 0.5, 0.0, 0b101),
  ("OrangeRed", 1.0, 0.270608, 0.0, 0b000),
  ("Orchid", 0.8549, 0.439207, 0.839198, 0b000),
  ("PaleGoldenrod", 0.933302, 0.909803, 0.666707, 0b000),
  ("PaleGreen", 0.596103, 0.984298, 0.596103, 0b000),
  ("PaleTurquoise", 0.6863, 0.933302, 0.933302, 0b000),
  ("PaleVioletRed", 0.858806, 0.439207, 0.576495, 0b000),
  ("PapayaWhip", 1.0, 0.9373, 0.835307, 0b000),
  ("Peach", 0.44, 0.26, 0.26, 0b000),
  ("PeachPuff", 1.0, 0.8549, 0.725501, 0b000),
  ("Peacock", 0.200003, 0.629994, 0.790002, 0b000),
  ("PermanentGreen", 0.039995, 0.790002, 0.170003, 0b000),
  ("PermanentRedViolet", 0.859996, 0.149998, 0.269997, 0b000),
  ("Peru", 0.803903, 0.521607, 0.247093, 0b000),
  ("Pink", 1.0, 0.5, 0.5, 0b001),
  ("Plum", 0.866695, 0.627506, 0.866695, 0b000),
  ("PowderBlue", 0.690207, 0.878399, 0.902005, 0b000),
  ("PrussianBlue", 0.18, 0.18, 0.31, 0b000),
  ("Purple", 0.5, 0.0, 0.5, 0b010),
  ("Raspberry", 0.53, 0.149998, 0.340007, 0b000),
  ("RawSienna", 0.780007, 0.380001, 0.080004, 0b000),
  ("RawUmber", 0.449995, 0.290002, 0.069994, 0b000),
  ("Red", 1.0, 0.0, 0.0, 0b111),
  ("RoseMadder", 0.889996, 0.209998, 0.219993, 0b000),
  ("RosyBrown", 0.737297, 0.560793, 0.560793, 0b000),
  ("RoyalBlue", 0.254906, 0.411802, 0.882397, 0b000),
  ("SaddleBrown", 0.545106, 0.270608, 0.074496, 0b000),
  ("Salmon", 0.980407, 0.501999, 0.447096, 0b000),
  ("SandyBrown", 0.956893, 0.643101, 0.376507, 0b000),
  ("SapGreen", 0.189993, 0.5, 0.080004, 0b000),
  ("SeaGreen", 0.180395, 0.545106, 0.341197, 0b000),
  ("Seashell", 1.0, 0.960799, 0.933302, 0b000),
  ("Sepia", 0.370006, 0.149998, 0.069994, 0b000),
  ("Sienna", 0.627506, 0.321604, 0.176504, 0b000),
  ("SkyBlue", 0.529405, 0.807794, 0.921598, 0b000),
  ("SlateBlue", 0.415693, 0.352901, 0.803903, 0b000),
  ("SlateGray", 0.439207, 0.501999, 0.564699, 0b000),
  ("Smoke", 0.960799, 0.960799, 0.960799, 0b000),
  ("Snow", 1.0, 0.980407, 0.980407, 0b000),
  ("SpringGreen", 0.0, 1.0, 0.498001, 0b000),
  ("SteelBlue", 0.274499, 0.509796, 0.705893, 0b000),
  ("TerreVerte", 0.219993, 0.370006, 0.059999, 0b000),
  ("Thistle", 0.847102, 0.749, 0.847102, 0b000),
  ("Titanium", 0.990005, 1.0, 0.940001, 0b000),
  ("Tomato", 1.0, 0.388195, 0.278405, 0b000),
  ("Turquoise", 0.250999, 0.878399, 0.815699, 0b000),
  ("TurquoiseBlue", 0.0, 0.780007, 0.550005, 0b000),
  ("Ultramarine", 0.069994, 0.039995, 0.559999, 0b000),
  ("UltramarineViolet", 0.359996, 0.140004, 0.430006, 0b000),
  ("VanDykeBrown", 0.370006, 0.149998, 0.020005, 0b000),
  ("VenetianRed", 0.829997, 0.099994, 0.119999, 0b000),
  ("Violet", 0.559999, 0.370006, 0.599994, 0b000),
  ("VioletRed", 0.815699, 0.125507, 0.564699, 0b000),
  ("WarmGray", 0.5, 0.5, 0.410001, 0b000),
  ("Wheat", 0.960799, 0.870602, 0.702002, 0b000),
  ("White", 1.0, 1.0, 1.0, 0b111),
  ("Yellow", 1.0, 1.0, 0.0, 0b111),
  ("YellowBrown", 0.86, 0.58, 0.44, 0b000),
  ("YellowGreen", 0.6039, 0.803903, 0.196097, 0b000),
  ("YellowOchre", 0.889996, 0.509995, 0.089999, 0b000),
  ("Zinc", 0.990005, 0.97, 1.0, 0b000),
];

/// Element symbols in atomic-number order with their `"Atoms"` colour
/// (the usual CPK/Jmol molecular-model palette), as 0xRRGGBB.
const ATOM_COLORS: [(&str, u32); 36] = [
  ("H", 0xFFFFFF),
  ("He", 0xD9FFFF),
  ("Li", 0xCC80FF),
  ("Be", 0xC2FF00),
  ("B", 0xFFB5B5),
  ("C", 0x909090),
  ("N", 0x3050F8),
  ("O", 0xFF0D0D),
  ("F", 0x90E050),
  ("Ne", 0xB3E3F5),
  ("Na", 0xAB5CF2),
  ("Mg", 0x8AFF00),
  ("Al", 0xBFA6A6),
  ("Si", 0xF0C8A0),
  ("P", 0xFF8000),
  ("S", 0xFFFF30),
  ("Cl", 0x1FF01F),
  ("Ar", 0x80D1E3),
  ("K", 0x8F40D4),
  ("Ca", 0x3DFF00),
  ("Sc", 0xE6E6E6),
  ("Ti", 0xBFC2C7),
  ("V", 0xA6A6AB),
  ("Cr", 0x8A99C7),
  ("Mn", 0x9C7AC7),
  ("Fe", 0xE06633),
  ("Co", 0xF090A0),
  ("Ni", 0x50D050),
  ("Cu", 0xC88033),
  ("Zn", 0x7D80B0),
  ("Ga", 0xC28F8F),
  ("Ge", 0x668F8F),
  ("As", 0xBD80E3),
  ("Se", 0xFFA100),
  ("Br", 0xA62929),
  ("Kr", 0x5CB8D1),
];

/// `ColorData["Atoms", element]` → the element's colour channels in 0–1, for
/// an element given by symbol (`"O"`) or atomic number (`8`).
fn atom_color(element: &Expr) -> Option<(f64, f64, f64)> {
  let rgb = match element {
    Expr::String(sym) => ATOM_COLORS.iter().find(|(s, _)| s == sym)?.1,
    Expr::Integer(z) => {
      ATOM_COLORS
        .get(usize::try_from(*z).ok()?.checked_sub(1)?)?
        .1
    }
    _ => return None,
  };
  let channel = |shift: u32| f64::from((rgb >> shift) & 0xFF) / 255.0;
  Some((channel(16), channel(8), channel(0)))
}

/// `ColorData["Legacy", name]` → the named legacy colour as an `RGBColor`,
/// with each channel already exact-or-inexact as wolframscript stores it.
fn legacy_named_color(name: &str) -> Option<Expr> {
  let &(_, r, g, b, exact) = LEGACY_COLORS
    .iter()
    .find(|(n, ..)| n.eq_ignore_ascii_case(name))?;
  let channel = |v: f64, bit: u8| {
    if exact & (1 << bit) != 0 {
      Expr::Integer(v as i128)
    } else {
      Expr::Real(v)
    }
  };
  Some(call(
    "RGBColor",
    vec![channel(r, 0), channel(g, 1), channel(b, 2)],
  ))
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
