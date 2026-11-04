//! Graphics shown inside a list, `Row`, `Grid`, `Column` or `TableForm`.
//!
//! The Wolfram front end lays a picture out at a reduced size when it sits
//! inside another construct: one whose `ImageSize` is automatic is drawn at
//! half its standalone size in a list, `Grid`, `Column` or `TableForm`, and
//! at a quarter in a `Row` of several items without a separator. It is drawn
//! *at* that size — lines, points, tick marks and tick labels keep their
//! absolute sizes (a 10 pt tick label stays 10 pt) — rather than being drawn
//! full size and shrunk. A graphic with an explicit `ImageSize` keeps it.
//!
//! The multiplier in effect is a property of the display context, so it is
//! kept here while a container's items are being rendered and consulted by
//! the picture arms of `expr_to_svg`.

use crate::syntax::Expr;
use std::cell::Cell;

thread_local! {
  static IMAGE_SIZE_MULTIPLIER: Cell<f64> = const { Cell::new(1.0) };
}

/// The `ImageSize` multiplier for an automatically sized picture in a list,
/// `Grid`, `Column`, `TableForm` or `Row` with a separator.
pub const LIST_MULTIPLIER: f64 = 0.5;

/// The multiplier for a picture in a `Row` of several items without a
/// separator.
pub const ROW_MULTIPLIER: f64 = 0.25;

/// The multiplier pictures rendered now are subject to (1 outside any
/// container).
pub fn image_size_multiplier() -> f64 {
  IMAGE_SIZE_MULTIPLIER.with(Cell::get)
}

/// Runs `f` with `multiplier` in effect. The innermost container decides:
/// a `Row` inside a list draws its pictures at a quarter, not an eighth.
pub fn with_image_size_multiplier<R>(
  multiplier: f64,
  f: impl FnOnce() -> R,
) -> R {
  let previous = IMAGE_SIZE_MULTIPLIER.with(|m| m.replace(multiplier));
  let result = f();
  IMAGE_SIZE_MULTIPLIER.with(|m| m.set(previous));
  result
}

/// Whether `options` give an `ImageSize` other than `Automatic`. Nested
/// option lists (`Graphics[prims, {opts}]`) are looked through.
fn has_explicit_image_size(options: &[Expr]) -> bool {
  options.iter().any(|o| match o {
    Expr::List(items) => has_explicit_image_size(items),
    _ => matches!(
      image_size_rule_value(o),
      Some(value) if !matches!(value, Expr::Identifier(v) if v == "Automatic")
    ),
  })
}

/// The value of an `ImageSize -> v` (or `:>`) rule, as written — a delayed
/// value is not evaluated, it may have side effects.
fn image_size_rule_value(option: &Expr) -> Option<&Expr> {
  match option {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "ImageSize") => {
      Some(replacement)
    }
    _ => None,
  }
}

/// `options` without any `ImageSize` rule, followed by `ImageSize -> width`.
fn with_image_size(options: &[Expr], width: u32) -> Vec<Expr> {
  fn strip(options: &[Expr]) -> Vec<Expr> {
    options
      .iter()
      .filter(|o| image_size_rule_value(o).is_none())
      .map(|o| match o {
        Expr::List(items) => Expr::List(strip(items).into()),
        other => other.clone(),
      })
      .collect()
  }
  let mut out = strip(options);
  out.push(crate::helpers::rule_expr(
    crate::helpers::id_expr("ImageSize"),
    Expr::Integer(i128::from(width)),
  ));
  out
}

/// The width an automatically sized picture `natural_width` wide is drawn
/// at under the current multiplier, or `None` outside a container.
fn container_width(natural_width: f64) -> Option<u32> {
  let m = image_size_multiplier();
  (m < 1.0 && natural_width > 0.0)
    .then(|| (natural_width * m).round().max(1.0) as u32)
}

/// The arguments of a symbolic `Graphics[…]`/`Graphics3D[…]` to render it
/// with inside the current container: its own with `ImageSize` set to the
/// reduced automatic size (both default to 360 wide). `None` when no
/// container applies or the picture sets its own size.
pub fn symbolic_graphics_args(args: &[Expr]) -> Option<Vec<Expr>> {
  if args.is_empty() || has_explicit_image_size(&args[1..]) {
    return None;
  }
  let width = container_width(crate::functions::plot::DEFAULT_WIDTH as f64)?;
  let mut out = vec![args[0].clone()];
  out.extend(with_image_size(&args[1..], width));
  Some(out)
}

/// A rendered picture (`Expr::Graphics`) drawn again at the reduced size of
/// the current container, from the data it keeps: a plot's sampled series
/// (`source`), the plot call that made it (see [`remember_plot_call`]), or
/// the symbolic graphic it was built from (`structure`).
///
/// A redraw is only trusted when the same data, drawn at the picture's
/// own size, reproduces the picture exactly — otherwise the stored data
/// does not capture everything the picture shows and the caller keeps the
/// original rendering. `None` when no container applies, the picture has an
/// explicit `ImageSize`, or it cannot be redrawn faithfully.
pub fn resized_graphic_svg(expr: &Expr) -> Option<String> {
  let Expr::Graphics {
    svg,
    source,
    structure,
    ..
  } = expr
  else {
    return None;
  };
  let (natural_width, _) = crate::functions::graphics::svg_natural_size(svg)?;
  let width = container_width(natural_width)?;
  // What the picture holds (an `Inset` plot, say) is drawn at its own size.
  with_image_size_multiplier(1.0, || {
    redraw_at_width(svg, source.as_deref(), structure.as_deref(), width)
  })
}

fn redraw_at_width(
  svg: &str,
  source: Option<&crate::syntax::PlotSource>,
  structure: Option<&Expr>,
  width: u32,
) -> Option<String> {
  // An explicit size is kept, whichever record of it says so.
  if source.is_some_and(|s| has_explicit_image_size(&s.options))
    || matches!(structure, Some(Expr::FunctionCall { args, .. })
      if args.len() > 1 && has_explicit_image_size(&args[1..]))
  {
    return None;
  }
  source
    .and_then(|source| redraw_source(svg, source, width))
    .or_else(|| replot_at_width(svg, width))
    .or_else(|| structure.and_then(|s| redraw_structure(svg, s, width)))
}

/// A plot redrawn from its sampled series by the plot renderer.
fn redraw_source(
  svg: &str,
  source: &crate::syntax::PlotSource,
  width: u32,
) -> Option<String> {
  let draw = |width: Option<u32>| -> Option<String> {
    let mut source = source.clone();
    if let Some(w) = width {
      source.options = with_image_size(&source.options, w);
    }
    let svg =
      crate::functions::plot::render_merged_plot_source(&source).ok()?;
    Some(crate::functions::plot_padding::keep_text_in_view(&svg).into_owned())
  };
  (draw(None).as_deref() == Some(svg))
    .then(|| draw(Some(width)))
    .flatten()
}

/// A graphic redrawn from the symbolic `Graphics[…]`/`Graphics3D[…]` it
/// was built from.
fn redraw_structure(svg: &str, structure: &Expr, width: u32) -> Option<String> {
  let Expr::FunctionCall { name, args } = structure else {
    return None;
  };
  if args.is_empty() {
    return None;
  }
  let render = |args: &[Expr]| -> Option<String> {
    let captured = crate::captured_graphics_count();
    let result = match name.as_str() {
      "Graphics" => crate::functions::graphics::graphics_ast(args),
      "Graphics3D" => crate::functions::plot3d::graphics3d_ast(args),
      _ => return None,
    };
    crate::truncate_captured_graphics(captured);
    match &result {
      Ok(Expr::Graphics { svg, .. }) => Some(svg.clone()),
      _ => None,
    }
  };
  if render(args).as_deref() != Some(svg) {
    return None;
  }
  let mut resized = vec![args[0].clone()];
  resized.extend(with_image_size(&args[1..], width));
  render(&resized)
}

// ── Plot calls remembered for a redraw ──────────────────────────────────
//
// Not every plot keeps data its own renderer can be run on again (a
// `ContourPlot`'s filled bands, a `BarChart`, a 3D surface). Those are
// redrawn by evaluating the plot call once more with the reduced
// `ImageSize`. The call is remembered when the plot is made, keyed by the
// picture it made, with the values its free symbols had then (a `Table`
// iterator is gone by the time the list is displayed). A redraw is only
// used when evaluating the remembered call again reproduces the picture
// exactly, so a call whose meaning has changed since (a redefined function,
// a random sample) keeps its original rendering.

thread_local! {
  static PLOT_CALLS: std::cell::RefCell<std::collections::VecDeque<(u64, String, Vec<Expr>)>> =
    const { std::cell::RefCell::new(std::collections::VecDeque::new()) };
}

/// How many plot calls are remembered; the oldest are forgotten first.
const PLOT_CALL_MEMORY: usize = 64;

fn svg_key(svg: &str) -> u64 {
  use std::hash::{Hash, Hasher};
  let mut hasher = std::collections::hash_map::DefaultHasher::new();
  svg.hash(&mut hasher);
  hasher.finish()
}

/// The picture a plot call evaluated to — the graphic itself, or the one a
/// `Legended` wraps.
fn plot_svg(result: &Expr) -> Option<&str> {
  match result {
    Expr::Graphics { svg, .. } => Some(svg),
    Expr::FunctionCall { name, args }
      if name == "Legended" && !args.is_empty() =>
    {
      plot_svg(&args[0])
    }
    _ => None,
  }
}

/// Heads whose evaluation changes state, so a plot call holding one is
/// never evaluated again behind the user's back.
fn has_side_effects(expr: &Expr) -> bool {
  const EFFECTFUL: &[&str] = &[
    "Print",
    "Echo",
    "EchoFunction",
    "Set",
    "SetDelayed",
    "Unset",
    "Clear",
    "ClearAll",
    "AppendTo",
    "PrependTo",
    "AddTo",
    "SubtractFrom",
    "TimesBy",
    "DivideBy",
    "Increment",
    "Decrement",
    "PreIncrement",
    "PreDecrement",
    "Sow",
    "Export",
    "Put",
    "PutAppend",
    "Write",
    "WriteString",
    "Run",
    "RunProcess",
    "Pause",
    "Message",
    "Beep",
    "Speak",
    "EmitSound",
    "SeedRandom",
    "CreateFile",
    "DeleteFile",
  ];
  let head_effectful =
    |name: &str| EFFECTFUL.contains(&name) || name.starts_with("Random");
  match expr {
    Expr::FunctionCall { name, .. } if head_effectful(name) => true,
    Expr::Identifier(name) if head_effectful(name) => true,
    _ => crate::syntax::expr_children(expr)
      .into_iter()
      .any(has_side_effects),
  }
}

fn collect_symbols(expr: &Expr, out: &mut Vec<String>) {
  if let Expr::Identifier(name) = expr {
    if !out.contains(name) {
      out.push(name.clone());
    }
    return;
  }
  for child in crate::syntax::expr_children(expr) {
    collect_symbols(child, out);
  }
}

/// The plot arguments with every free symbol that has a value now replaced
/// by it, except the plot's own variables (the first element of each
/// iterator `{x, …}`), which the plot localizes. `None` when a value would
/// bring in a side effect.
fn with_current_values(args: &[Expr]) -> Option<Vec<Expr>> {
  let mut plot_vars: Vec<&str> = Vec::new();
  for arg in args.iter().skip(1) {
    if let Expr::List(items) = arg
      && let Some(Expr::Identifier(v)) = items.first()
    {
      plot_vars.push(v);
    }
  }
  let mut symbols = Vec::new();
  for arg in args {
    collect_symbols(arg, &mut symbols);
  }
  let mut bindings: Vec<(String, Expr)> = Vec::new();
  for name in symbols {
    if plot_vars.contains(&name.as_str())
      || crate::evaluator::functions::is_builtin_symbol(&name)
    {
      continue;
    }
    let symbol = Expr::Identifier(name.clone());
    let Ok(value) = crate::evaluator::evaluate_expr_to_expr(&symbol) else {
      continue;
    };
    if matches!(&value, Expr::Identifier(v) if *v == name) {
      continue;
    }
    if has_side_effects(&value) {
      return None;
    }
    bindings.push((name, value));
  }
  if bindings.is_empty() {
    return Some(args.to_vec());
  }
  let refs: Vec<(&str, &Expr)> =
    bindings.iter().map(|(n, v)| (n.as_str(), v)).collect();
  Some(
    args
      .iter()
      .map(|a| crate::syntax::substitute_variables(a, &refs))
      .collect(),
  )
}

/// Remembers the plot call `name[args]` that evaluated to `result`, so the
/// picture can be redrawn at a container's reduced size later. Calls with
/// an explicit `ImageSize` (kept in containers) or a side effect are not.
pub fn remember_plot_call(name: &str, args: &[Expr], result: &Expr) {
  let Some(svg) = plot_svg(result) else {
    return;
  };
  if args.is_empty()
    || has_explicit_image_size(&args[1..])
    || args.iter().any(has_side_effects)
  {
    return;
  }
  let Some(args) = with_current_values(args) else {
    return;
  };
  let key = svg_key(svg);
  PLOT_CALLS.with(|calls| {
    let mut calls = calls.borrow_mut();
    calls.retain(|(k, ..)| *k != key);
    if calls.len() >= PLOT_CALL_MEMORY {
      calls.pop_front();
    }
    calls.push_back((key, name.to_string(), args));
  });
}

fn recalled_plot_call(svg: &str) -> Option<(String, Vec<Expr>)> {
  let key = svg_key(svg);
  PLOT_CALLS.with(|calls| {
    calls
      .borrow()
      .iter()
      .rev()
      .find(|(k, ..)| *k == key)
      .map(|(_, name, args)| (name.clone(), args.clone()))
  })
}

/// Evaluates the plot call `name[args]` again, quietly and without leaving
/// its picture among the session's captured output, and returns the
/// picture it draws.
fn replot(name: &str, args: &[Expr]) -> Option<String> {
  let warnings = crate::snapshot_warnings();
  let captured = crate::captured_graphics_count();
  crate::push_quiet();
  let result = crate::evaluator::evaluate_expr_to_expr(&crate::helpers::call(
    name,
    args.to_vec(),
  ));
  crate::pop_quiet();
  crate::truncate_captured_graphics(captured);
  crate::restore_warnings(warnings);
  plot_svg(&result.ok()?).map(str::to_string)
}

/// The picture `svg` drawn again `width` wide by evaluating the plot call
/// that made it, when that call is remembered and still reproduces it.
fn replot_at_width(svg: &str, width: u32) -> Option<String> {
  let (name, args) = recalled_plot_call(svg)?;
  if replot(&name, &args).as_deref() != Some(svg) {
    return None;
  }
  let mut sized = args[..1].to_vec();
  sized.extend(with_image_size(&args[1..], width));
  replot(&name, &sized)
}

/// One element of a list laid out as a picture: a rendered cell, or a
/// nested list drawn with its own braces.
pub enum ListCell {
  Picture(String),
  List(Vec<ListCell>),
}

// Geometry of a list typeset around pictures, in points, as the front end
// sets it: the advance of a brace, of a comma with the space after it, the
// indent of a continuation line, the widest line before the list wraps,
// and the line pitch as a multiple of the line height.
const BRACE_ADVANCE: f64 = 7.80127;
const COMMA_ADVANCE: f64 = 12.48;
// The comma's own ink, which is all of it a line ending in one occupies.
const COMMA_INK: f64 = 5.2;
const CONTINUATION_INDENT: f64 = 8.3;
const MAX_LINE_WIDTH: f64 = 1008.5;
const LINE_SPACING: f64 = 1.25;
// How far the text baseline sits below the centre line the pictures are
// centred on (the math axis of the surrounding text).
const BASELINE_BELOW_AXIS: f64 = 2.86;

enum Atom {
  Open,
  Close,
  Comma,
  /// `base`: how far below its top the picture's baseline lies.
  Picture {
    svg: String,
    w: f64,
    h: f64,
    base: f64,
  },
}

/// The attribute a picture drawn with `BaselinePosition -> Axis` carries
/// on its `<svg>` element: its axis, in points below its top.
const BASELINE_ATTR: &str = "data-baseline";

/// Mark the picture `svg` as sitting on its line at `y` points below its
/// top (see [`BASELINE_ATTR`]).
pub fn set_baseline_attr(svg: &mut String, y: f64) {
  if let Some(pos) = svg.find("<svg ") {
    svg.insert_str(pos + 5, &format!("{BASELINE_ATTR}=\"{y:.2}\" "));
  }
}

/// The baseline a picture's `<svg>` element records (see
/// [`set_baseline_attr`]).
fn baseline_attr(svg: &str) -> Option<f64> {
  let head = &svg[..svg.find('>')?];
  let start =
    head.find(&format!("{BASELINE_ATTR}=\""))? + BASELINE_ATTR.len() + 2;
  let rest = &head[start..];
  rest[..rest.find('"')?].parse().ok()
}

impl Atom {
  fn width(&self) -> f64 {
    match self {
      Atom::Open | Atom::Close => BRACE_ADVANCE,
      Atom::Comma => COMMA_ADVANCE,
      Atom::Picture { w, .. } => *w,
    }
  }
  /// The extent above the line's axis (pictures centre on it unless they
  /// name their own baseline).
  fn above(&self) -> f64 {
    match self {
      Atom::Picture { base, .. } => *base,
      _ => 0.0,
    }
  }
  /// The extent below the line's axis.
  fn below(&self) -> f64 {
    match self {
      Atom::Picture { h, base, .. } => h - base,
      _ => 0.0,
    }
  }
}

fn push_atoms(cell: ListCell, out: &mut Vec<Atom>) -> Option<()> {
  match cell {
    ListCell::Picture(svg) => {
      let (w, h) = crate::functions::graphics::svg_natural_size(&svg)?;
      let base = baseline_attr(&svg)
        .filter(|b| b.is_finite() && (0.0..=h).contains(b))
        .unwrap_or(h / 2.0);
      out.push(Atom::Picture { svg, w, h, base });
    }
    ListCell::List(items) => {
      out.push(Atom::Open);
      let n = items.len();
      for (i, item) in items.into_iter().enumerate() {
        push_atoms(item, out)?;
        if i + 1 < n {
          out.push(Atom::Comma);
        }
      }
      out.push(Atom::Close);
    }
  }
  Some(())
}

/// Packs the units (top-level elements, each with its trailing comma) into
/// lines no wider than `limit`, filling each line before starting the
/// next. Returns the index of the first unit of each line.
fn pack_lines(widths: &[f64], limit: f64) -> Vec<usize> {
  let n = widths.len();
  let mut starts = vec![0];
  let mut line = BRACE_ADVANCE;
  for (i, w) in widths.iter().enumerate() {
    let line_start = *starts.last().expect("one line") == i;
    if !line_start && line + w + line_end_adjustment(i, n) > limit {
      starts.push(i);
      line = CONTINUATION_INDENT;
    }
    line += w;
  }
  starts
}

/// What a line ending with unit `i` of `n` occupies beyond the units'
/// advances: the closing brace after the last unit, and only the ink of
/// the comma (not the space after it) that ends any other line.
fn line_end_adjustment(i: usize, n: usize) -> f64 {
  if i + 1 == n {
    BRACE_ADVANCE
  } else {
    COMMA_INK - COMMA_ADVANCE
  }
}

/// The lines a list of `widths` units is set in: as few as fit in
/// [`MAX_LINE_WIDTH`], with the units spread over them as evenly as the
/// widest line allows (six equal pictures that need two lines go three and
/// three, seven go four and three).
fn balanced_lines(widths: &[f64]) -> Vec<usize> {
  let fewest = pack_lines(widths, MAX_LINE_WIDTH);
  if fewest.len() == 1 {
    return fewest;
  }
  // The narrowest line limit that still needs no more lines.
  let mut candidates: Vec<f64> = Vec::new();
  for a in 0..widths.len() {
    let mut w = if a == 0 {
      BRACE_ADVANCE
    } else {
      CONTINUATION_INDENT
    };
    for (b, unit) in widths.iter().enumerate().skip(a) {
      w += unit;
      let line = w + line_end_adjustment(b, widths.len());
      if line <= MAX_LINE_WIDTH {
        candidates.push(line);
      }
    }
  }
  candidates.sort_by(f64::total_cmp);
  candidates.dedup();
  let (mut lo, mut hi) = (0, candidates.len());
  while lo < hi {
    let mid = usize::midpoint(lo, hi);
    if pack_lines(widths, candidates[mid] + 1e-9).len() <= fewest.len() {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  candidates
    .get(lo)
    .map_or(fewest, |&limit| pack_lines(widths, limit + 1e-9))
}

/// Draws `{cell₁, cell₂, …}` the way a notebook shows a list holding
/// pictures: each picture at its own size, centred on the line's text
/// axis, separated by commas, nested lists with their own braces, and the
/// whole wrapped onto several lines when it is wider than the page.
pub fn list_layout_svg(cells: Vec<ListCell>) -> Option<String> {
  if cells.is_empty() {
    return None;
  }
  // The atoms of each top-level element, with the comma that follows it.
  let n = cells.len();
  let mut units: Vec<Vec<Atom>> = Vec::with_capacity(n);
  for (i, cell) in cells.into_iter().enumerate() {
    let mut atoms = Vec::new();
    push_atoms(cell, &mut atoms)?;
    if i + 1 < n {
      atoms.push(Atom::Comma);
    }
    units.push(atoms);
  }
  let widths: Vec<f64> = units
    .iter()
    .map(|u| u.iter().map(Atom::width).sum())
    .collect();
  let starts = balanced_lines(&widths);

  let mut lines: Vec<Vec<Atom>> = Vec::with_capacity(starts.len());
  let mut units = units.into_iter();
  for (l, &start) in starts.iter().enumerate() {
    let end = starts.get(l + 1).copied().unwrap_or(n);
    let mut line = Vec::new();
    if l == 0 {
      line.push(Atom::Open);
    }
    for _ in start..end {
      line.extend(units.next()?);
    }
    if l + 1 == starts.len() {
      line.push(Atom::Close);
    }
    lines.push(line);
  }

  let text_fill = crate::functions::graphics::theme().text_primary;
  let mut body = String::with_capacity(4096);
  let mut width: f64 = 0.0;
  let mut top = 0.0;
  let mut bottom: f64 = 0.0;
  for (l, line) in lines.iter().enumerate() {
    let above = line.iter().map(Atom::above).fold(0.0, f64::max);
    let below = line.iter().map(Atom::below).fold(0.0, f64::max);
    let line_h = above + below;
    let axis = top + above;
    let baseline = axis + BASELINE_BELOW_AXIS;
    let mut x = if l == 0 { 0.0 } else { CONTINUATION_INDENT };
    for atom in line {
      match atom {
        Atom::Picture { svg, w, h, base } => {
          body.push_str(&crate::functions::graphics::embed_svg_in_rect(
            svg,
            x,
            axis - base,
            *w,
            *h,
          ));
        }
        Atom::Open | Atom::Close => {
          // The front end stretches the brace a little around the line.
          let glyph = if matches!(atom, Atom::Open) { '{' } else { '}' };
          body.push_str(&format!(
            "<text x=\"0\" y=\"0\" font-family=\"Times, serif\" \
             font-size=\"16\" fill=\"{text_fill}\" \
             transform=\"translate({:.2},{:.2}) scale(1,1.35)\">{glyph}</text>\n",
            x + 0.6,
            baseline - 0.6,
          ));
        }
        Atom::Comma => {
          body.push_str(&format!(
            "<text x=\"{:.2}\" y=\"{baseline:.2}\" font-family=\"Times, serif\" \
             font-size=\"13\" fill=\"{text_fill}\">,</text>\n",
            x + 1.6,
          ));
        }
      }
      x += atom.width();
    }
    if matches!(line.last(), Some(Atom::Comma)) {
      x += COMMA_INK - COMMA_ADVANCE;
    }
    width = width.max(x);
    bottom = top + line_h;
    top += LINE_SPACING * line_h;
  }
  let (w, h) = ((width - 1e-6).ceil(), (bottom - 1e-6).ceil().max(1.0));
  Some(format!(
    "<svg width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\" \
     xmlns=\"http://www.w3.org/2000/svg\">\n{body}</svg>"
  ))
}
