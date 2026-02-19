//! Converts Graphics primitives into Mathematica GraphicsBox expression strings.
//!
//! This module produces the text that goes inside a `Cell[BoxData[...], "Output"]`
//! in a `.nb` file, so that Mathematica renders the graphics natively without
//! re-evaluation.

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use flate2::Compression;
use flate2::write::ZlibEncoder;
use std::io::Write;

/// Encode a list of `(f64, f64)` coordinate pairs as Mathematica `CompressedData["..."]`.
///
/// The binary format inside is Mathematica's internal serialization:
///   `!boR` header + `f` tag + count + `s` tag + "List" + nested `f`/`r` items
///
/// Then: `"1:" + base64(zlib(binary))`
pub fn compressed_point_list(points: &[(f64, f64)]) -> String {
  let mut buf: Vec<u8> = Vec::with_capacity(points.len() * 20 + 32);

  // Magic header
  buf.extend_from_slice(b"!boR");

  // Top-level: function with `points.len()` args, head = "List"
  write_func_header(&mut buf, "List", points.len() as u32);

  // Each element is a {x, y} pair = List[x, y]
  for &(x, y) in points {
    write_func_header(&mut buf, "List", 2);
    write_real(&mut buf, x);
    write_real(&mut buf, y);
  }

  compress_and_encode(&buf)
}

/// Encode a flat list of f64 values as CompressedData.
pub fn compressed_real_list(values: &[f64]) -> String {
  let mut buf: Vec<u8> = Vec::with_capacity(values.len() * 10 + 16);
  buf.extend_from_slice(b"!boR");
  write_func_header(&mut buf, "List", values.len() as u32);
  for &v in values {
    write_real(&mut buf, v);
  }
  compress_and_encode(&buf)
}

// ── Binary serialization helpers ──────────────────────────────────────

fn write_func_header(buf: &mut Vec<u8>, head: &str, arg_count: u32) {
  buf.push(b'f');
  buf.extend_from_slice(&arg_count.to_le_bytes());
  write_symbol(buf, head);
}

fn write_symbol(buf: &mut Vec<u8>, name: &str) {
  buf.push(b's');
  buf.extend_from_slice(&(name.len() as u32).to_le_bytes());
  buf.extend_from_slice(name.as_bytes());
}

fn write_real(buf: &mut Vec<u8>, v: f64) {
  buf.push(b'r');
  buf.extend_from_slice(&v.to_le_bytes());
}

#[allow(dead_code)]
fn write_integer(buf: &mut Vec<u8>, v: i32) {
  buf.push(b'i');
  buf.extend_from_slice(&v.to_le_bytes());
}

fn compress_and_encode(data: &[u8]) -> String {
  let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
  encoder.write_all(data).unwrap();
  let compressed = encoder.finish().unwrap();
  format!("1:{}", BASE64.encode(&compressed))
}

// ── GraphicsBox string builders ───────────────────────────────────────

/// Format an f64 for Mathematica output (no trailing zeros, no scientific notation for common values)
fn fmt_real(v: f64) -> String {
  if v == v.floor() && v.abs() < 1e15 {
    // Integer-valued: emit without decimal point for cleanliness
    format!("{}", v as i64)
  } else {
    format!("{}", v)
  }
}

/// Build the GraphicsBox expression string for a simple Graphics[] with primitives.
///
/// `directives_and_primitives` is the list of Mathematica box expressions
/// (e.g. `RGBColor[1, 0, 0]`, `DiskBox[{0, 0}]`) already formatted as strings.
pub fn graphics_box(elements: &[String]) -> String {
  if elements.is_empty() {
    return "GraphicsBox[{}]".to_string();
  }
  format!("GraphicsBox[{{\n  {}}}]", elements.join(", "))
}

/// Convert an RGB color to its Mathematica box form.
pub fn rgbcolor_box(r: f64, g: f64, b: f64) -> String {
  format!(
    "RGBColor[{}, {}, {}]",
    fmt_real(r),
    fmt_real(g),
    fmt_real(b)
  )
}

/// Convert opacity directive.
pub fn opacity_box(o: f64) -> String {
  format!("Opacity[{}]", fmt_real(o))
}

/// Convert AbsoluteThickness directive.
pub fn abs_thickness_box(t: f64) -> String {
  format!("AbsoluteThickness[{}]", fmt_real(t))
}

/// Convert PointSize directive.
pub fn point_size_box(s: f64) -> String {
  format!("PointSize[{}]", fmt_real(s))
}

/// DiskBox[{cx, cy}] or DiskBox[{cx, cy}, r]
pub fn disk_box(cx: f64, cy: f64, r: f64) -> String {
  if (r - 1.0).abs() < 1e-10 {
    format!("DiskBox[{{{}, {}}}]", fmt_real(cx), fmt_real(cy))
  } else {
    format!(
      "DiskBox[{{{}, {}}}, {}]",
      fmt_real(cx),
      fmt_real(cy),
      fmt_real(r)
    )
  }
}

/// DiskBox[{cx, cy}, r, {a1, a2}] for sectors
pub fn disk_sector_box(cx: f64, cy: f64, r: f64, a1: f64, a2: f64) -> String {
  format!(
    "DiskBox[{{{}, {}}}, {}, {{{}, {}}}]",
    fmt_real(cx),
    fmt_real(cy),
    fmt_real(r),
    fmt_real(a1),
    fmt_real(a2)
  )
}

/// CircleBox[{cx, cy}] or CircleBox[{cx, cy}, r]
pub fn circle_box(cx: f64, cy: f64, r: f64) -> String {
  if (r - 1.0).abs() < 1e-10 {
    format!("CircleBox[{{{}, {}}}]", fmt_real(cx), fmt_real(cy))
  } else {
    format!(
      "CircleBox[{{{}, {}}}, {}]",
      fmt_real(cx),
      fmt_real(cy),
      fmt_real(r)
    )
  }
}

/// RectangleBox[{xmin, ymin}, {xmax, ymax}]
pub fn rectangle_box(x_min: f64, y_min: f64, x_max: f64, y_max: f64) -> String {
  format!(
    "RectangleBox[{{{}, {}}}, {{{}, {}}}]",
    fmt_real(x_min),
    fmt_real(y_min),
    fmt_real(x_max),
    fmt_real(y_max)
  )
}

/// PointBox[{x, y}]
pub fn point_box(x: f64, y: f64) -> String {
  format!("PointBox[{{{}, {}}}]", fmt_real(x), fmt_real(y))
}

/// PointBox[{{x1,y1}, {x2,y2}, ...}]
pub fn point_box_multi(points: &[(f64, f64)]) -> String {
  let pts: Vec<String> = points
    .iter()
    .map(|(x, y)| format!("{{{}, {}}}", fmt_real(*x), fmt_real(*y)))
    .collect();
  format!("PointBox[{{{}}}]", pts.join(", "))
}

/// LineBox[CompressedData["..."]] for large coordinate lists,
/// or LineBox[{{x1,y1}, ...}] for small ones.
pub fn line_box(segments: &[Vec<(f64, f64)>]) -> Vec<String> {
  segments
    .iter()
    .map(|seg| {
      if seg.len() > 20 {
        format!(
          "LineBox[CompressedData[\"{}\"]]",
          compressed_point_list(seg)
        )
      } else {
        let pts: Vec<String> = seg
          .iter()
          .map(|(x, y)| format!("{{{}, {}}}", fmt_real(*x), fmt_real(*y)))
          .collect();
        format!("LineBox[{{{}}}]", pts.join(", "))
      }
    })
    .collect()
}

/// PolygonBox[{{x1,y1}, ...}]
pub fn polygon_box(points: &[(f64, f64)]) -> String {
  if points.len() > 20 {
    format!(
      "PolygonBox[CompressedData[\"{}\"]]",
      compressed_point_list(points)
    )
  } else {
    let pts: Vec<String> = points
      .iter()
      .map(|(x, y)| format!("{{{}, {}}}", fmt_real(*x), fmt_real(*y)))
      .collect();
    format!("PolygonBox[{{{}}}]", pts.join(", "))
  }
}

/// ArrowBox[{{x1,y1}, ...}]
pub fn arrow_box(points: &[(f64, f64)]) -> String {
  let pts: Vec<String> = points
    .iter()
    .map(|(x, y)| format!("{{{}, {}}}", fmt_real(*x), fmt_real(*y)))
    .collect();
  format!("ArrowBox[{{{}}}]", pts.join(", "))
}

/// InsetBox[text, {x, y}]
pub fn inset_box(text: &str, x: f64, y: f64) -> String {
  // Escape quotes in text
  let escaped = text.replace('\\', "\\\\").replace('"', "\\\"");
  format!(
    "InsetBox[\"{}\", {{{}, {}}}]",
    escaped,
    fmt_real(x),
    fmt_real(y)
  )
}

/// BezierCurveBox[{{x1,y1}, ...}]
pub fn bezier_curve_box(points: &[(f64, f64)]) -> String {
  let pts: Vec<String> = points
    .iter()
    .map(|(x, y)| format!("{{{}, {}}}", fmt_real(*x), fmt_real(*y)))
    .collect();
  format!("BezierCurveBox[{{{}}}]", pts.join(", "))
}

/// EdgeForm[...] directive
pub fn edge_form_box(color: Option<(f64, f64, f64)>) -> String {
  match color {
    Some((r, g, b)) => format!("EdgeForm[{}]", rgbcolor_box(r, g, b)),
    None => "EdgeForm[None]".to_string(),
  }
}

/// FaceForm[...] directive
pub fn face_form_box(color: Option<(f64, f64, f64)>) -> String {
  match color {
    Some((r, g, b)) => format!("FaceForm[{}]", rgbcolor_box(r, g, b)),
    None => "FaceForm[None]".to_string(),
  }
}

/// Dashing[{d1, d2, ...}]
pub fn dashing_box(dashes: &[f64]) -> String {
  let ds: Vec<String> = dashes.iter().map(|d| fmt_real(*d)).collect();
  format!("Dashing[{{{}}}]", ds.join(", "))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_compressed_roundtrip_simple() {
    // Verify the format is "1:" + base64
    let result = compressed_point_list(&[(0.0, 0.0), (1.0, 1.0)]);
    assert!(result.starts_with("1:"));
    // Should be decodable: strip "1:", base64 decode, zlib decompress
    let b64 = &result[2..];
    let compressed = BASE64.decode(b64).unwrap();
    let mut decoder = flate2::read::ZlibDecoder::new(&compressed[..]);
    let mut decompressed = Vec::new();
    std::io::Read::read_to_end(&mut decoder, &mut decompressed).unwrap();
    // Should start with !boR magic
    assert_eq!(&decompressed[..4], b"!boR");
  }

  #[test]
  fn test_disk_box_default_radius() {
    assert_eq!(disk_box(0.0, 0.0, 1.0), "DiskBox[{0, 0}]");
  }

  #[test]
  fn test_disk_box_custom_radius() {
    assert_eq!(disk_box(1.0, 2.0, 0.5), "DiskBox[{1, 2}, 0.5]");
  }

  #[test]
  fn test_line_box_small() {
    let segs = vec![vec![(0.0, 0.0), (1.0, 1.0)]];
    let result = line_box(&segs);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], "LineBox[{{0, 0}, {1, 1}}]");
  }

  #[test]
  fn test_line_box_large_uses_compressed() {
    let seg: Vec<(f64, f64)> =
      (0..30).map(|i| (i as f64, (i as f64).sin())).collect();
    let result = line_box(&[seg]);
    assert_eq!(result.len(), 1);
    assert!(result[0].starts_with("LineBox[CompressedData[\"1:"));
  }
}
