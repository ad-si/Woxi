//! `Import` of PDF files.
//!
//! Every page's content stream is interpreted into an SVG picture that
//! becomes a `Graphics`, the way wolframscript's `Import["file.pdf"]` gives
//! one `Graphics` per page. The interpreter covers the vector part of the
//! content model — paths, transforms, colours, clipping, transparency,
//! form and image XObjects — and draws text through SVG text elements
//! under the font family the PDF names, since embedded font programs are
//! not rasterised here.

use crate::InterpreterError;
use crate::syntax::Expr;
use lopdf::content::Content;
use lopdf::{Dictionary, Document, Object, ObjectId};
use std::collections::HashSet;
use std::fmt::Write as _;

/// The pages of the PDF at `path`, each as a `Graphics`.
pub fn import_pages(path: &str) -> Result<Vec<Expr>, InterpreterError> {
  let doc = load(path)?;
  Ok(
    page_svgs(&doc)
      .into_iter()
      .map(crate::graphics_result)
      .collect(),
  )
}

/// The number of pages in the PDF at `path`.
pub fn page_count(path: &str) -> Result<usize, InterpreterError> {
  Ok(load(path)?.get_pages().len())
}

/// The text of every page of the PDF at `path`, pages separated by a form
/// feed the way wolframscript's `"Plaintext"` element separates them.
pub fn plaintext(path: &str) -> Result<String, InterpreterError> {
  let doc = load(path)?;
  let pages: Vec<u32> = doc.get_pages().keys().copied().collect();
  let mut out = String::new();
  for (i, page) in pages.iter().enumerate() {
    if i > 0 {
      out.push('\u{c}');
    }
    let text = doc.extract_text(&[*page]).unwrap_or_default();
    out.push_str(text.trim_end());
  }
  Ok(out)
}

/// The SVG rendering of every page of a PDF given as bytes.
pub fn pages_svg_from_bytes(bytes: &[u8]) -> Result<Vec<String>, String> {
  let doc = Document::load_mem(bytes).map_err(|e| e.to_string())?;
  Ok(page_svgs(&doc))
}

fn load(path: &str) -> Result<Document, InterpreterError> {
  Document::load(crate::vfs::resolve(path)).map_err(|e| {
    InterpreterError::EvaluationError(format!(
      "Import: cannot read \"{path}\" as PDF: {e}"
    ))
  })
}

fn page_svgs(doc: &Document) -> Vec<String> {
  doc
    .get_pages()
    .values()
    .map(|&page_id| render_page(doc, page_id))
    .collect()
}

/// A 2-D affine matrix in PDF's `[a b c d e f]` layout.
type Matrix = [f64; 6];

const IDENTITY: Matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

/// `m × n`: apply `m` first, then `n` (PDF's convention for `cm`).
fn mul(m: &Matrix, n: &Matrix) -> Matrix {
  [
    m[0] * n[0] + m[1] * n[2],
    m[0] * n[1] + m[1] * n[3],
    m[2] * n[0] + m[3] * n[2],
    m[2] * n[1] + m[3] * n[3],
    m[4] * n[0] + m[5] * n[2] + n[4],
    m[4] * n[1] + m[5] * n[3] + n[5],
  ]
}

fn apply(m: &Matrix, x: f64, y: f64) -> (f64, f64) {
  (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])
}

/// The length scale of a matrix, for widths given in user space.
fn scale_of(m: &Matrix) -> f64 {
  (m[0] * m[3] - m[1] * m[2]).abs().sqrt()
}

fn fmt(v: f64) -> String {
  if !v.is_finite() {
    return "0".to_string();
  }
  let s = format!("{v:.3}");
  let s = s.trim_end_matches('0').trim_end_matches('.');
  if s.is_empty() || s == "-0" {
    "0".to_string()
  } else {
    s.to_string()
  }
}

fn matrix_attr(m: &Matrix) -> String {
  format!(
    "matrix({} {} {} {} {} {})",
    fmt(m[0]),
    fmt(m[1]),
    fmt(m[2]),
    fmt(m[3]),
    fmt(m[4]),
    fmt(m[5])
  )
}

/// A colour as an SVG paint, from PDF components by their count.
fn paint(components: &[f64]) -> String {
  let clamp = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
  match components {
    [g] => {
      let g = clamp(*g);
      format!("rgb({g},{g},{g})")
    }
    [r, g, b] => format!("rgb({},{},{})", clamp(*r), clamp(*g), clamp(*b)),
    [c, m, y, k] => {
      let ch = |v: f64| clamp((1.0 - v) * (1.0 - k));
      format!("rgb({},{},{})", ch(*c), ch(*m), ch(*y))
    }
    _ => "rgb(0,0,0)".to_string(),
  }
}

/// The font a text run is drawn with.
#[derive(Clone)]
struct Font {
  family: String,
  weight: &'static str,
  style: &'static str,
  /// Glyph advances by character code, in text-space units (1/1000 em),
  /// where the PDF lists them.
  first_char: u32,
  widths: Vec<f64>,
  /// A CID-keyed font takes two-byte codes.
  two_byte: bool,
  /// How its codes decode to text.
  encoding: Option<Dictionary>,
}

#[derive(Clone)]
struct State {
  ctm: Matrix,
  fill: String,
  stroke: String,
  fill_alpha: f64,
  stroke_alpha: f64,
  line_width: f64,
  line_cap: u8,
  line_join: u8,
  dash: Vec<f64>,
  clip: Option<usize>,
  font: Option<Font>,
  font_size: f64,
  char_spacing: f64,
  word_spacing: f64,
  hscale: f64,
  leading: f64,
  rise: f64,
  render_mode: i64,
}

impl State {
  fn new(ctm: Matrix) -> Self {
    Self {
      ctm,
      fill: "rgb(0,0,0)".to_string(),
      stroke: "rgb(0,0,0)".to_string(),
      fill_alpha: 1.0,
      stroke_alpha: 1.0,
      line_width: 1.0,
      line_cap: 0,
      line_join: 0,
      dash: Vec::new(),
      clip: None,
      font: None,
      font_size: 0.0,
      char_spacing: 0.0,
      word_spacing: 0.0,
      hscale: 1.0,
      leading: 0.0,
      rise: 0.0,
      render_mode: 0,
    }
  }
}

struct Renderer<'a> {
  doc: &'a Document,
  /// The picture, in PDF page coordinates (flipped by the outer group).
  body: String,
  defs: String,
  clip_count: usize,
  image_count: usize,
  /// Form XObjects currently being drawn, so a form that refers to itself
  /// does not recurse for ever.
  active_forms: HashSet<ObjectId>,
}

/// The state that path construction carries between operators.
struct PathBuilder {
  d: String,
  current: (f64, f64),
  start: (f64, f64),
  /// A `W`/`W*` seen since the last painting operator, with its rule.
  pending_clip: Option<&'static str>,
}

impl PathBuilder {
  fn new() -> Self {
    Self {
      d: String::new(),
      current: (0.0, 0.0),
      start: (0.0, 0.0),
      pending_clip: None,
    }
  }
}

/// Text object state: the text matrix and the line matrix.
struct TextState {
  tm: Matrix,
  tlm: Matrix,
}

fn number(obj: &Object) -> Option<f64> {
  match obj {
    Object::Integer(i) => Some(*i as f64),
    Object::Real(r) => Some(f64::from(*r)),
    _ => None,
  }
}

fn numbers(objs: &[Object]) -> Vec<f64> {
  objs.iter().filter_map(number).collect()
}

/// A page attribute, inherited from the page tree when the page itself
/// does not carry it.
fn inherited<'a>(
  doc: &'a Document,
  page: &'a Dictionary,
  key: &[u8],
) -> Option<&'a Object> {
  let mut node = page;
  for _ in 0..64 {
    if let Ok(value) = node.get(key) {
      return doc.dereference(value).ok().map(|(_, o)| o);
    }
    let parent = node.get(b"Parent").ok()?;
    let (_, parent) = doc.dereference(parent).ok()?;
    node = parent.as_dict().ok()?;
  }
  None
}

fn rect(obj: Option<&Object>) -> Option<[f64; 4]> {
  let items = obj?.as_array().ok()?;
  let v = numbers(items);
  if v.len() == 4 {
    Some([
      v[0].min(v[2]),
      v[1].min(v[3]),
      v[0].max(v[2]),
      v[1].max(v[3]),
    ])
  } else {
    None
  }
}

fn render_page(doc: &Document, page_id: ObjectId) -> String {
  let Ok(page) = doc.get_dictionary(page_id) else {
    return empty_svg(1.0, 1.0);
  };
  let media =
    rect(inherited(doc, page, b"MediaBox")).unwrap_or([0.0, 0.0, 612.0, 792.0]);
  // The visible part of the page is its CropBox, clipped to the MediaBox.
  let bbox = rect(inherited(doc, page, b"CropBox")).map_or(media, |c| {
    [
      c[0].max(media[0]),
      c[1].max(media[1]),
      c[2].min(media[2]),
      c[3].min(media[3]),
    ]
  });
  let width = (bbox[2] - bbox[0]).max(1.0);
  let height = (bbox[3] - bbox[1]).max(1.0);
  let rotate = inherited(doc, page, b"Rotate")
    .and_then(number)
    .map_or(0, |r| ((r as i64).rem_euclid(360) / 90) * 90);

  let resources = inherited(doc, page, b"Resources")
    .and_then(|r| r.as_dict().ok())
    .cloned()
    .unwrap_or_default();
  let content = doc.get_page_content(page_id);

  let mut renderer = Renderer {
    doc,
    body: String::new(),
    defs: String::new(),
    clip_count: 0,
    image_count: 0,
    active_forms: HashSet::new(),
  };
  renderer.run(&content, &resources, State::new(IDENTITY));

  // Page space is y-up; the outer group flips it and moves the box's
  // corner to the origin. A /Rotate turns the finished page.
  let (out_w, out_h, turn) = match rotate {
    90 => (
      height,
      width,
      format!("translate({} 0) rotate(90) ", fmt(height)),
    ),
    180 => (
      width,
      height,
      format!("translate({} {}) rotate(180) ", fmt(width), fmt(height)),
    ),
    270 => (
      height,
      width,
      format!("translate(0 {}) rotate(270) ", fmt(width)),
    ),
    _ => (width, height, String::new()),
  };
  let flip = format!("matrix(1 0 0 -1 {} {})", fmt(-bbox[0]), fmt(bbox[3]));
  let mut svg = String::new();
  let _ = writeln!(
    svg,
    "<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">",
    fmt(out_w),
    fmt(out_h),
    fmt(out_w),
    fmt(out_h)
  );
  if !renderer.defs.is_empty() {
    let _ = write!(svg, "<defs>\n{}</defs>\n", renderer.defs);
  }
  let _ = write!(
    svg,
    "<g transform=\"{turn}{flip}\">\n{}</g>\n</svg>\n",
    renderer.body
  );
  svg
}

fn empty_svg(w: f64, h: f64) -> String {
  format!(
    "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\"></svg>\n",
    fmt(w),
    fmt(h),
    fmt(w),
    fmt(h)
  )
}

impl Renderer<'_> {
  /// A named entry of a resource category (`/XObject`, `/Font`, …),
  /// dereferenced.
  fn resource(
    &self,
    resources: &Dictionary,
    category: &[u8],
    name: &[u8],
  ) -> Option<Object> {
    let cat = resources.get(category).ok()?;
    let (_, cat) = self.doc.dereference(cat).ok()?;
    let entry = cat.as_dict().ok()?.get(name).ok()?;
    let (_, entry) = self.doc.dereference(entry).ok()?;
    Some(entry.clone())
  }

  /// The number of components a colour space takes.
  fn colour_components(&self, resources: &Dictionary, space: &Object) -> usize {
    match space {
      Object::Name(n) => match n.as_slice() {
        b"DeviceGray" | b"G" | b"CalGray" => 1,
        b"DeviceRGB" | b"RGB" | b"CalRGB" | b"Lab" => 3,
        b"DeviceCMYK" | b"CMYK" => 4,
        b"Pattern" => 0,
        other => self
          .resource(resources, b"ColorSpace", other)
          .map_or(3, |cs| self.colour_components(resources, &cs)),
      },
      Object::Array(items) => {
        let family = items.first().and_then(|o| o.as_name().ok());
        match family {
          Some(b"ICCBased") => items
            .get(1)
            .and_then(|s| self.doc.dereference(s).ok())
            .and_then(|(_, s)| s.as_stream().ok())
            .and_then(|s| s.dict.get(b"N").ok())
            .and_then(number)
            .map_or(3, |n| n as usize),
          Some(b"CalRGB" | b"Lab") => 3,
          Some(b"CalGray" | b"Separation" | b"Indexed") => 1,
          Some(b"DeviceN") => items
            .get(1)
            .and_then(|n| n.as_array().ok())
            .map_or(1, Vec::len),
          Some(b"Pattern") => 0,
          Some(b"DeviceGray") => 1,
          Some(b"DeviceCMYK") => 4,
          _ => 3,
        }
      }
      _ => 3,
    }
  }

  /// Colour components as a paint, where a one-component value of a
  /// `Separation`/`Indexed` space is read as ink coverage.
  fn paint_for(components: &[f64], separation: bool) -> String {
    if separation && components.len() == 1 {
      return paint(&[1.0 - components[0]]);
    }
    paint(components)
  }

  fn clip_attr(state: &State) -> String {
    state
      .clip
      .map_or_else(String::new, |id| format!(" clip-path=\"url(#clip{id})\""))
  }

  /// Emit the current path with the given painting, and turn it into a
  /// clip when a `W` is pending.
  fn paint_path(
    &mut self,
    pb: &mut PathBuilder,
    state: &mut State,
    fill: bool,
    even_odd: bool,
    stroke: bool,
  ) {
    let d = std::mem::take(&mut pb.d);
    if !d.is_empty() && (fill || stroke) {
      let mut attrs = String::new();
      if fill {
        let _ = write!(attrs, " fill=\"{}\"", state.fill);
        if even_odd {
          attrs.push_str(" fill-rule=\"evenodd\"");
        }
        if state.fill_alpha < 1.0 {
          let _ = write!(attrs, " fill-opacity=\"{}\"", fmt(state.fill_alpha));
        }
      } else {
        attrs.push_str(" fill=\"none\"");
      }
      if stroke {
        // Width 0 asks for the thinnest line the device draws.
        let width = state.line_width * scale_of(&state.ctm);
        let width = if width <= 0.0 { 0.3 } else { width };
        let _ = write!(
          attrs,
          " stroke=\"{}\" stroke-width=\"{}\"",
          state.stroke,
          fmt(width)
        );
        if state.stroke_alpha < 1.0 {
          let _ =
            write!(attrs, " stroke-opacity=\"{}\"", fmt(state.stroke_alpha));
        }
        match state.line_cap {
          1 => attrs.push_str(" stroke-linecap=\"round\""),
          2 => attrs.push_str(" stroke-linecap=\"square\""),
          _ => {}
        }
        match state.line_join {
          1 => attrs.push_str(" stroke-linejoin=\"round\""),
          2 => attrs.push_str(" stroke-linejoin=\"bevel\""),
          _ => {}
        }
        if !state.dash.is_empty() && state.dash.iter().any(|d| *d > 0.0) {
          let scale = scale_of(&state.ctm);
          let dashes: Vec<String> =
            state.dash.iter().map(|d| fmt(d * scale)).collect();
          let _ = write!(attrs, " stroke-dasharray=\"{}\"", dashes.join(" "));
        }
      }
      let _ = writeln!(
        self.body,
        "<path d=\"{d}\"{attrs}{}/>",
        Self::clip_attr(state)
      );
    }
    if let Some(rule) = pb.pending_clip.take() {
      self.clip_count += 1;
      let id = self.clip_count;
      let parent = Self::clip_attr(state);
      let rule_attr = if rule == "evenodd" {
        " clip-rule=\"evenodd\""
      } else {
        ""
      };
      // An empty clip path clips everything away, as it does in PDF.
      let _ = writeln!(
        self.defs,
        "<clipPath id=\"clip{id}\"{parent}><path d=\"{d}\"{rule_attr}/></clipPath>"
      );
      state.clip = Some(id);
    }
  }

  fn run(&mut self, content: &[u8], resources: &Dictionary, initial: State) {
    let Ok(content) = Content::decode(content) else {
      return;
    };
    let mut state = initial;
    let mut stack: Vec<State> = Vec::new();
    let mut pb = PathBuilder::new();
    let mut text = TextState {
      tm: IDENTITY,
      tlm: IDENTITY,
    };
    // Whether the current fill/stroke colour space reads one component as
    // ink coverage rather than gray.
    let mut fill_sep = false;
    let mut stroke_sep = false;
    let mut fill_n = 1usize;
    let mut stroke_n = 1usize;

    for op in &content.operations {
      let o = op.operands.as_slice();
      match op.operator.as_str() {
        // Graphics state
        "q" => stack.push(state.clone()),
        "Q" => {
          if let Some(s) = stack.pop() {
            state = s;
          }
        }
        "cm" => {
          let v = numbers(o);
          if v.len() == 6 {
            state.ctm = mul(&[v[0], v[1], v[2], v[3], v[4], v[5]], &state.ctm);
          }
        }
        "w" => {
          if let Some(w) = o.first().and_then(number) {
            state.line_width = w;
          }
        }
        "J" => {
          if let Some(c) = o.first().and_then(number) {
            state.line_cap = c as u8;
          }
        }
        "j" => {
          if let Some(j) = o.first().and_then(number) {
            state.line_join = j as u8;
          }
        }
        "d" => {
          state.dash = o
            .first()
            .and_then(|a| a.as_array().ok())
            .map(|a| numbers(a))
            .unwrap_or_default();
        }
        "gs" => {
          if let Some(name) = o.first().and_then(|n| n.as_name().ok())
            && let Some(Object::Dictionary(gs)) =
              self.resource(resources, b"ExtGState", name)
          {
            if let Some(a) = gs.get(b"CA").ok().and_then(number) {
              state.stroke_alpha = a;
            }
            if let Some(a) = gs.get(b"ca").ok().and_then(number) {
              state.fill_alpha = a;
            }
            if let Some(w) = gs.get(b"LW").ok().and_then(number) {
              state.line_width = w;
            }
            if let Ok(Object::Array(d)) = gs.get(b"D")
              && let Some(Object::Array(dashes)) = d.first()
            {
              state.dash = numbers(dashes);
            }
          }
        }

        // Colour
        "g" | "G" => {
          let v = numbers(o);
          if v.len() == 1 {
            let p = paint(&v);
            if op.operator == "g" {
              state.fill = p;
              fill_sep = false;
              fill_n = 1;
            } else {
              state.stroke = p;
              stroke_sep = false;
              stroke_n = 1;
            }
          }
        }
        "rg" | "RG" => {
          let v = numbers(o);
          if v.len() == 3 {
            let p = paint(&v);
            if op.operator == "rg" {
              state.fill = p;
              fill_sep = false;
              fill_n = 3;
            } else {
              state.stroke = p;
              stroke_sep = false;
              stroke_n = 3;
            }
          }
        }
        "k" | "K" => {
          let v = numbers(o);
          if v.len() == 4 {
            let p = paint(&v);
            if op.operator == "k" {
              state.fill = p;
              fill_sep = false;
              fill_n = 4;
            } else {
              state.stroke = p;
              stroke_sep = false;
              stroke_n = 4;
            }
          }
        }
        "cs" | "CS" => {
          if let Some(space) = o.first() {
            let n = self.colour_components(resources, space);
            let sep = match space {
              Object::Name(name) => self
                .resource(resources, b"ColorSpace", name)
                .is_some_and(|cs| {
                  matches!(
                    cs.as_array()
                      .ok()
                      .and_then(|a| a.first())
                      .and_then(|f| f.as_name().ok()),
                    Some(b"Separation" | b"Indexed" | b"DeviceN")
                  )
                }),
              _ => false,
            };
            // A new colour space starts out black.
            let black = paint(&vec![0.0; n.max(1)]);
            if op.operator == "cs" {
              fill_n = n;
              fill_sep = sep;
              state.fill = if sep { paint(&[1.0]) } else { black };
            } else {
              stroke_n = n;
              stroke_sep = sep;
              state.stroke = if sep { paint(&[1.0]) } else { black };
            }
          }
        }
        "sc" | "scn" | "SC" | "SCN" => {
          let v = numbers(o);
          // A pattern paint (`/P1 scn`) has no components; it draws as
          // mid-gray so the shape it fills still shows.
          let is_fill = op.operator.starts_with('s');
          let (n, sep) = if is_fill {
            (fill_n, fill_sep)
          } else {
            (stroke_n, stroke_sep)
          };
          let p = if v.is_empty() {
            "rgb(128,128,128)".to_string()
          } else if v.len() == n || n == 0 {
            Self::paint_for(&v, sep)
          } else {
            paint(&v)
          };
          if is_fill {
            state.fill = p;
          } else {
            state.stroke = p;
          }
        }

        // Path construction — points go straight to page space.
        "m" => {
          let v = numbers(o);
          if v.len() == 2 {
            let p = apply(&state.ctm, v[0], v[1]);
            let _ = write!(pb.d, "M{} {}", fmt(p.0), fmt(p.1));
            pb.current = p;
            pb.start = p;
          }
        }
        "l" => {
          let v = numbers(o);
          if v.len() == 2 {
            let p = apply(&state.ctm, v[0], v[1]);
            let _ = write!(pb.d, "L{} {}", fmt(p.0), fmt(p.1));
            pb.current = p;
          }
        }
        "c" => {
          let v = numbers(o);
          if v.len() == 6 {
            let p1 = apply(&state.ctm, v[0], v[1]);
            let p2 = apply(&state.ctm, v[2], v[3]);
            let p3 = apply(&state.ctm, v[4], v[5]);
            let _ = write!(
              pb.d,
              "C{} {} {} {} {} {}",
              fmt(p1.0),
              fmt(p1.1),
              fmt(p2.0),
              fmt(p2.1),
              fmt(p3.0),
              fmt(p3.1)
            );
            pb.current = p3;
          }
        }
        "v" => {
          let v = numbers(o);
          if v.len() == 4 {
            let p1 = pb.current;
            let p2 = apply(&state.ctm, v[0], v[1]);
            let p3 = apply(&state.ctm, v[2], v[3]);
            let _ = write!(
              pb.d,
              "C{} {} {} {} {} {}",
              fmt(p1.0),
              fmt(p1.1),
              fmt(p2.0),
              fmt(p2.1),
              fmt(p3.0),
              fmt(p3.1)
            );
            pb.current = p3;
          }
        }
        "y" => {
          let v = numbers(o);
          if v.len() == 4 {
            let p1 = apply(&state.ctm, v[0], v[1]);
            let p3 = apply(&state.ctm, v[2], v[3]);
            let _ = write!(
              pb.d,
              "C{} {} {} {} {} {}",
              fmt(p1.0),
              fmt(p1.1),
              fmt(p3.0),
              fmt(p3.1),
              fmt(p3.0),
              fmt(p3.1)
            );
            pb.current = p3;
          }
        }
        "h" => {
          if !pb.d.is_empty() {
            pb.d.push('Z');
            pb.current = pb.start;
          }
        }
        "re" => {
          let v = numbers(o);
          if v.len() == 4 {
            let corners = [
              apply(&state.ctm, v[0], v[1]),
              apply(&state.ctm, v[0] + v[2], v[1]),
              apply(&state.ctm, v[0] + v[2], v[1] + v[3]),
              apply(&state.ctm, v[0], v[1] + v[3]),
            ];
            let _ = write!(
              pb.d,
              "M{} {}L{} {}L{} {}L{} {}Z",
              fmt(corners[0].0),
              fmt(corners[0].1),
              fmt(corners[1].0),
              fmt(corners[1].1),
              fmt(corners[2].0),
              fmt(corners[2].1),
              fmt(corners[3].0),
              fmt(corners[3].1)
            );
            pb.current = corners[0];
            pb.start = corners[0];
          }
        }

        // Path painting
        "S" => self.paint_path(&mut pb, &mut state, false, false, true),
        "s" => {
          if !pb.d.is_empty() {
            pb.d.push('Z');
          }
          self.paint_path(&mut pb, &mut state, false, false, true);
        }
        "f" | "F" => self.paint_path(&mut pb, &mut state, true, false, false),
        "f*" => self.paint_path(&mut pb, &mut state, true, true, false),
        "B" => self.paint_path(&mut pb, &mut state, true, false, true),
        "B*" => self.paint_path(&mut pb, &mut state, true, true, true),
        "b" => {
          if !pb.d.is_empty() {
            pb.d.push('Z');
          }
          self.paint_path(&mut pb, &mut state, true, false, true);
        }
        "b*" => {
          if !pb.d.is_empty() {
            pb.d.push('Z');
          }
          self.paint_path(&mut pb, &mut state, true, true, true);
        }
        "n" => self.paint_path(&mut pb, &mut state, false, false, false),
        "W" => pb.pending_clip = Some("nonzero"),
        "W*" => pb.pending_clip = Some("evenodd"),

        // XObjects
        "Do" => {
          if let Some(name) = o.first().and_then(|n| n.as_name().ok()) {
            self.draw_xobject(resources, name, &state);
          }
        }

        // Text
        "BT" => {
          text.tm = IDENTITY;
          text.tlm = IDENTITY;
        }
        "ET" => {}
        "Tc" => {
          if let Some(v) = o.first().and_then(number) {
            state.char_spacing = v;
          }
        }
        "Tw" => {
          if let Some(v) = o.first().and_then(number) {
            state.word_spacing = v;
          }
        }
        "Tz" => {
          if let Some(v) = o.first().and_then(number) {
            state.hscale = v / 100.0;
          }
        }
        "TL" => {
          if let Some(v) = o.first().and_then(number) {
            state.leading = v;
          }
        }
        "Ts" => {
          if let Some(v) = o.first().and_then(number) {
            state.rise = v;
          }
        }
        "Tr" => {
          if let Some(v) = o.first().and_then(number) {
            state.render_mode = v as i64;
          }
        }
        "Tf" => {
          if let Some(name) = o.first().and_then(|n| n.as_name().ok()) {
            state.font = self.font(resources, name);
          }
          if let Some(size) = o.get(1).and_then(number) {
            state.font_size = size;
          }
        }
        "Td" => {
          let v = numbers(o);
          if v.len() == 2 {
            text.tlm = mul(&[1.0, 0.0, 0.0, 1.0, v[0], v[1]], &text.tlm);
            text.tm = text.tlm;
          }
        }
        "TD" => {
          let v = numbers(o);
          if v.len() == 2 {
            state.leading = -v[1];
            text.tlm = mul(&[1.0, 0.0, 0.0, 1.0, v[0], v[1]], &text.tlm);
            text.tm = text.tlm;
          }
        }
        "Tm" => {
          let v = numbers(o);
          if v.len() == 6 {
            text.tlm = [v[0], v[1], v[2], v[3], v[4], v[5]];
            text.tm = text.tlm;
          }
        }
        "T*" => {
          text.tlm = mul(&[1.0, 0.0, 0.0, 1.0, 0.0, -state.leading], &text.tlm);
          text.tm = text.tlm;
        }
        "Tj" => {
          if let Some(Object::String(bytes, _)) = o.first() {
            self.show_text(bytes, &state, &mut text);
          }
        }
        "'" => {
          text.tlm = mul(&[1.0, 0.0, 0.0, 1.0, 0.0, -state.leading], &text.tlm);
          text.tm = text.tlm;
          if let Some(Object::String(bytes, _)) = o.first() {
            self.show_text(bytes, &state, &mut text);
          }
        }
        "\"" => {
          if let Some(aw) = o.first().and_then(number) {
            state.word_spacing = aw;
          }
          if let Some(ac) = o.get(1).and_then(number) {
            state.char_spacing = ac;
          }
          text.tlm = mul(&[1.0, 0.0, 0.0, 1.0, 0.0, -state.leading], &text.tlm);
          text.tm = text.tlm;
          if let Some(Object::String(bytes, _)) = o.get(2) {
            self.show_text(bytes, &state, &mut text);
          }
        }
        "TJ" => {
          if let Some(Object::Array(items)) = o.first() {
            for item in items {
              match item {
                Object::String(bytes, _) => {
                  self.show_text(bytes, &state, &mut text);
                }
                other => {
                  if let Some(adj) = number(other) {
                    let tx = -adj / 1000.0 * state.font_size * state.hscale;
                    text.tm = mul(&[1.0, 0.0, 0.0, 1.0, tx, 0.0], &text.tm);
                  }
                }
              }
            }
          }
        }

        // Shadings, inline images, marked content, type 3 glyph metrics
        // and compatibility sections draw nothing here.
        _ => {}
      }
    }
  }

  /// The font resource `name` names, with what is needed to place its text.
  fn font(&self, resources: &Dictionary, name: &[u8]) -> Option<Font> {
    let Object::Dictionary(dict) = self.resource(resources, b"Font", name)?
    else {
      return None;
    };
    let subtype = dict
      .get(b"Subtype")
      .ok()
      .and_then(|s| s.as_name().ok())
      .unwrap_or(b"");
    let two_byte = subtype == b"Type0";
    let base = dict
      .get(b"BaseFont")
      .ok()
      .and_then(|b| b.as_name().ok())
      .map(|b| String::from_utf8_lossy(b).into_owned())
      .unwrap_or_default();
    // `ABCDEF+Times-BoldItalic` → family `Times`, bold, italic.
    let name = base.split_once('+').map_or(base.as_str(), |(_, n)| n);
    let lower = name.to_ascii_lowercase();
    let weight = if lower.contains("bold")
      || lower.contains("black")
      || lower.contains("heavy")
    {
      "bold"
    } else {
      "normal"
    };
    let style = if lower.contains("italic") || lower.contains("oblique") {
      "italic"
    } else {
      "normal"
    };
    let stem = name.split(['-', ',']).next().unwrap_or(name).to_string();
    let family = if lower.starts_with("times")
      || lower.contains("serif") && !lower.contains("sans")
    {
      "'Times New Roman', Times, serif".to_string()
    } else if lower.starts_with("courier") || lower.contains("mono") {
      "'Courier New', Courier, monospace".to_string()
    } else if lower.starts_with("helvetica") || lower.starts_with("arial") {
      "Helvetica, Arial, sans-serif".to_string()
    } else if lower.starts_with("symbol") {
      "Symbol".to_string()
    } else {
      format!("'{stem}', sans-serif")
    };

    // Simple fonts list their glyph widths; a CID font's come from its
    // descendant's W array, which is read as a default here.
    let first_char = dict
      .get(b"FirstChar")
      .ok()
      .and_then(number)
      .map_or(0, |n| n as u32);
    let widths = dict
      .get(b"Widths")
      .ok()
      .and_then(|w| self.doc.dereference(w).ok())
      .and_then(|(_, w)| w.as_array().ok())
      .map(|w| {
        w.iter()
          .map(|x| {
            self
              .doc
              .dereference(x)
              .ok()
              .and_then(|(_, x)| number(x))
              .unwrap_or(0.0)
          })
          .collect()
      })
      .unwrap_or_default();
    Some(Font {
      family,
      weight,
      style,
      first_char,
      widths,
      two_byte,
      encoding: Some(dict),
    })
  }

  /// Draw a string at the text matrix and advance it.
  fn show_text(&mut self, bytes: &[u8], state: &State, text: &mut TextState) {
    let Some(font) = &state.font else {
      return;
    };
    let fs = state.font_size;
    // Decode the codes to text through the font's encoding; a font whose
    // encoding cannot be resolved shows its bytes as Latin-1.
    let decoded = font
      .encoding
      .as_ref()
      .and_then(|dict| dict.get_font_encoding(self.doc).ok())
      .and_then(|enc| enc.bytes_to_string(bytes).ok())
      .unwrap_or_else(|| bytes.iter().map(|&b| b as char).collect());

    // Where the run starts, in page space. The glyphs are drawn in SVG's
    // y-down text space, so the text matrix takes a flip.
    let trm = mul(
      &[fs * state.hscale, 0.0, 0.0, fs, 0.0, state.rise],
      &mul(&text.tm, &state.ctm),
    );
    let placed = mul(&[1.0, 0.0, 0.0, -1.0, 0.0, 0.0], &trm);
    if state.render_mode != 3
      && state.render_mode != 7
      && !decoded.trim().is_empty()
    {
      let fill = if state.render_mode == 1 || state.render_mode == 5 {
        "none".to_string()
      } else {
        state.fill.clone()
      };
      let mut attrs = format!(
        " font-family=\"{}\" font-size=\"1\" fill=\"{}\"",
        font.family, fill
      );
      if font.weight != "normal" {
        let _ = write!(attrs, " font-weight=\"{}\"", font.weight);
      }
      if font.style != "normal" {
        let _ = write!(attrs, " font-style=\"{}\"", font.style);
      }
      if state.fill_alpha < 1.0 {
        let _ = write!(attrs, " fill-opacity=\"{}\"", fmt(state.fill_alpha));
      }
      if matches!(state.render_mode, 1 | 2 | 5 | 6) {
        let _ = write!(
          attrs,
          " stroke=\"{}\" stroke-width=\"{}\"",
          state.stroke,
          fmt(state.line_width / fs.max(1e-6))
        );
      }
      let escaped = decoded
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");
      let _ = writeln!(
        self.body,
        "<text transform=\"{}\"{attrs} xml:space=\"preserve\"{}>{escaped}</text>",
        matrix_attr(&placed),
        Self::clip_attr(state)
      );
    }

    // Advance: each glyph's width, plus character and word spacing.
    let mut tx = 0.0;
    let codes: Vec<u32> = if font.two_byte {
      bytes
        .chunks(2)
        .map(|c| (u32::from(c[0]) << 8) | u32::from(*c.get(1).unwrap_or(&0)))
        .collect()
    } else {
      bytes.iter().map(|&b| u32::from(b)).collect()
    };
    for code in codes {
      let w = code
        .checked_sub(font.first_char)
        .and_then(|i| font.widths.get(i as usize))
        .copied()
        .unwrap_or(if font.two_byte { 1000.0 } else { 500.0 });
      let mut advance = w / 1000.0 * fs + state.char_spacing;
      if code == 32 && !font.two_byte {
        advance += state.word_spacing;
      }
      tx += advance * state.hscale;
    }
    text.tm = mul(&[1.0, 0.0, 0.0, 1.0, tx, 0.0], &text.tm);
  }

  /// Draw a form or image XObject at the current transform.
  fn draw_xobject(
    &mut self,
    resources: &Dictionary,
    name: &[u8],
    state: &State,
  ) {
    let Some(entry) = resources
      .get(b"XObject")
      .ok()
      .and_then(|x| self.doc.dereference(x).ok())
      .and_then(|(_, x)| x.as_dict().ok())
      .and_then(|x| x.get(name).ok())
    else {
      return;
    };
    let id = entry.as_reference().ok();
    let Ok((_, obj)) = self.doc.dereference(entry) else {
      return;
    };
    let Ok(stream) = obj.as_stream() else {
      return;
    };
    let subtype = stream
      .dict
      .get(b"Subtype")
      .ok()
      .and_then(|s| s.as_name().ok())
      .unwrap_or(b"");
    match subtype {
      b"Form" => {
        if let Some(id) = id {
          if self.active_forms.contains(&id) || self.active_forms.len() > 32 {
            return;
          }
          self.active_forms.insert(id);
        }
        let mut inner = state.clone();
        if let Ok(Object::Array(m)) = stream.dict.get(b"Matrix") {
          let v = numbers(m);
          if v.len() == 6 {
            inner.ctm = mul(&[v[0], v[1], v[2], v[3], v[4], v[5]], &inner.ctm);
          }
        }
        // The form is clipped to its bounding box.
        if let Some(b) = rect(stream.dict.get(b"BBox").ok()) {
          let corners = [
            apply(&inner.ctm, b[0], b[1]),
            apply(&inner.ctm, b[2], b[1]),
            apply(&inner.ctm, b[2], b[3]),
            apply(&inner.ctm, b[0], b[3]),
          ];
          self.clip_count += 1;
          let clip_id = self.clip_count;
          let parent = Self::clip_attr(&inner);
          let _ = writeln!(
            self.defs,
            "<clipPath id=\"clip{clip_id}\"{parent}><path d=\"M{} {}L{} {}L{} {}L{} {}Z\"/></clipPath>",
            fmt(corners[0].0),
            fmt(corners[0].1),
            fmt(corners[1].0),
            fmt(corners[1].1),
            fmt(corners[2].0),
            fmt(corners[2].1),
            fmt(corners[3].0),
            fmt(corners[3].1)
          );
          inner.clip = Some(clip_id);
        }
        let own_resources = stream
          .dict
          .get(b"Resources")
          .ok()
          .and_then(|r| self.doc.dereference(r).ok())
          .and_then(|(_, r)| r.as_dict().ok())
          .cloned();
        let content = stream
          .decompressed_content()
          .unwrap_or_else(|_| stream.content.clone());
        self.run(&content, own_resources.as_ref().unwrap_or(resources), inner);
        if let Some(id) = id {
          self.active_forms.remove(&id);
        }
      }
      b"Image" => {
        if let Some(href) = image_data_uri(stream) {
          self.image_count += 1;
          // The image fills the unit square of user space, its first row
          // at the top (y = 1).
          let placed = mul(&[1.0, 0.0, 0.0, -1.0, 0.0, 1.0], &state.ctm);
          let mut attrs = String::new();
          if state.fill_alpha < 1.0 {
            let _ = write!(attrs, " opacity=\"{}\"", fmt(state.fill_alpha));
          }
          let _ = writeln!(
            self.body,
            "<image x=\"0\" y=\"0\" width=\"1\" height=\"1\" preserveAspectRatio=\"none\" transform=\"{}\"{attrs}{} xlink:href=\"{href}\"/>",
            matrix_attr(&placed),
            Self::clip_attr(state)
          );
        }
      }
      _ => {}
    }
  }
}

/// An image XObject's pixels as a data URI: JPEG data passes through, and
/// 8-bit gray/RGB samples are packed into a PNG. Other encodings (JPX,
/// CCITT, JBIG2, deep or indexed samples) are not decoded.
fn image_data_uri(stream: &lopdf::Stream) -> Option<String> {
  use base64::Engine as _;
  let filters = stream.filters().unwrap_or_default();
  let width = stream.dict.get(b"Width").ok().and_then(number)? as u32;
  let height = stream.dict.get(b"Height").ok().and_then(number)? as u32;
  if filters.iter().any(|f| *f == b"DCTDecode") {
    let encoded =
      base64::engine::general_purpose::STANDARD.encode(&stream.content);
    return Some(format!("data:image/jpeg;base64,{encoded}"));
  }
  let bpc = stream
    .dict
    .get(b"BitsPerComponent")
    .ok()
    .and_then(number)
    .unwrap_or(8.0) as u32;
  if bpc != 8 {
    return None;
  }
  let data = stream.decompressed_content().ok()?;
  let space = stream
    .dict
    .get(b"ColorSpace")
    .ok()
    .and_then(|c| c.as_name().ok())
    .unwrap_or(b"DeviceRGB");
  let mut png = Vec::new();
  let cursor = &mut std::io::Cursor::new(&mut png);
  match space {
    b"DeviceGray" | b"CalGray" | b"G" => {
      let img = image::GrayImage::from_raw(width, height, data)?;
      img.write_to(cursor, image::ImageFormat::Png).ok()?;
    }
    b"DeviceRGB" | b"CalRGB" | b"RGB" => {
      let img = image::RgbImage::from_raw(width, height, data)?;
      img.write_to(cursor, image::ImageFormat::Png).ok()?;
    }
    _ => return None,
  }
  let encoded = base64::engine::general_purpose::STANDARD.encode(&png);
  Some(format!("data:image/png;base64,{encoded}"))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn matrices_compose_in_pdf_order() {
    let scale = [2.0, 0.0, 0.0, 2.0, 0.0, 0.0];
    let shift = [1.0, 0.0, 0.0, 1.0, 5.0, 7.0];
    // Scale first, then shift.
    let m = mul(&scale, &shift);
    assert_eq!(apply(&m, 1.0, 1.0), (7.0, 9.0));
    // Shift first, then scale.
    let m = mul(&shift, &scale);
    assert_eq!(apply(&m, 1.0, 1.0), (12.0, 16.0));
  }

  #[test]
  fn paints_by_component_count() {
    assert_eq!(paint(&[1.0]), "rgb(255,255,255)");
    assert_eq!(paint(&[1.0, 0.0, 0.0]), "rgb(255,0,0)");
    assert_eq!(paint(&[0.0, 0.0, 0.0, 1.0]), "rgb(0,0,0)");
  }

  #[test]
  fn numbers_format_compactly() {
    assert_eq!(fmt(1.0), "1");
    assert_eq!(fmt(-0.0004), "0");
    assert_eq!(fmt(12.3456), "12.346");
  }
}
