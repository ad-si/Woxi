use plotters::prelude::{Color, *};

use super::*;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::graphics::{Color as GfxColor, parse_color};
use crate::functions::math_ast::{
  build_complex_float_expr, try_eval_to_f64, try_extract_complex_f64,
};
use crate::functions::plot::{
  DEFAULT_WIDTH, RESOLUTION_SCALE, evaluate_at_xy, generate_axes_only,
  parse_image_size, parse_iterator, plot_theme, rewrite_svg_header,
  substitute_var, svg_header,
};

const FIELD_GRID: usize = 100;
const VECTOR_GRID: usize = 15;

/// Default look of a contour line, used unless `ContourStyle` says otherwise.
/// The weight is scaled by `render_width / 1000` when it is emitted.
const CONTOUR_LINE_COLOR: &str = "#404040";
const CONTOUR_LINE_WEIGHT: f64 = 2.5;

/// Evaluate a boolean condition at (x, y) - returns true/false
fn evaluate_condition(
  body: &Expr,
  xvar: &str,
  yvar: &str,
  xval: f64,
  yval: f64,
) -> bool {
  let sub1 = substitute_var(body, xvar, &Expr::Real(xval));
  let sub2 = substitute_var(&sub1, yvar, &Expr::Real(yval));
  if let Ok(result) = evaluate_expr_to_expr(&sub2) {
    matches!(result, Expr::Identifier(ref s) if s == "True")
  } else {
    false
  }
}

/// Evaluate {vx, vy} at (x, y)
fn evaluate_vector(
  body: &Expr,
  xvar: &str,
  yvar: &str,
  xval: f64,
  yval: f64,
) -> Option<(f64, f64)> {
  let sub1 = substitute_var(body, xvar, &Expr::Real(xval));
  let sub2 = substitute_var(&sub1, yvar, &Expr::Real(yval));
  let result = evaluate_expr_to_expr(&sub2).ok()?;
  if let Expr::List(items) = &result
    && items.len() == 2
  {
    let vx = try_eval_to_f64(&items[0])?;
    let vy = try_eval_to_f64(&items[1])?;
    return Some((vx, vy));
  }
  None
}

/// Evaluate the scalar `body` at (x, y).
fn evaluate_scalar(
  body: &Expr,
  xvar: &str,
  yvar: &str,
  xval: f64,
  yval: f64,
) -> Option<f64> {
  let sub1 = substitute_var(body, xvar, &Expr::Real(xval));
  let sub2 = substitute_var(&sub1, yvar, &Expr::Real(yval));
  try_eval_to_f64(&evaluate_expr_to_expr(&sub2).ok()?)
}

/// Split a stream-density field into its vector part and, for the
/// `{{vx, vy}, s}` form, the scalar `s` that shades the background. The form
/// is recognised by a first element that evaluates to a two-component
/// vector at the corner of the plot range.
fn split_vector_and_density<'a>(
  spec: &'a Expr,
  xvar: &str,
  yvar: &str,
  x_val: f64,
  y_val: f64,
) -> (&'a Expr, Option<&'a Expr>) {
  if let Expr::List(items) = spec
    && items.len() == 2
    && evaluate_vector(&items[0], xvar, yvar, x_val, y_val).is_some()
  {
    return (&items[0], Some(&items[1]));
  }
  (spec, None)
}

/// Parse ImageSize from options
/// The `PlotLabel` of a field plot, rendered as SVG markup. Field plots draw
/// their own axes, so the label is injected above the plot area rather than
/// going through the line renderer's label pass.
fn parse_field_plot_label(
  args: &[Expr],
  start: usize,
) -> Option<(String, Option<f64>)> {
  args[start..].iter().find_map(|opt| {
    let (Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    }) = opt
    else {
      return None;
    };
    if !matches!(pattern.as_ref(), Expr::Identifier(n) if n == "PlotLabel") {
      return None;
    }
    let value = evaluate_expr_to_expr(replacement)
      .unwrap_or_else(|_| replacement.as_ref().clone());
    // A size on the outermost `Style` is the label's font size, which the
    // caller applies at render resolution; leaving it inside the markup
    // would emit it in unscaled units and draw the label ten times too
    // small.
    let (inner, size) = peel_label_font_size(&value);
    Some((crate::functions::graphics::expr_to_svg_markup(inner), size))
  })
}

/// The axes margins that leave room for a field plot's `PlotLabel`, or
/// `None` when there is no label to draw (the caller then keeps plotters'
/// default margins). Shared by every field plot that draws its own axes
/// with [`crate::functions::plot::generate_axes_only_opts`], so the top
/// margin, tick-label areas, and font-size fallback all move together.
fn field_plot_label_margins(
  plot_label: Option<&(String, Option<f64>)>,
) -> Option<crate::functions::plot::MarginOverrides> {
  plot_label.map(|(_, size)| crate::functions::plot::MarginOverrides {
    top_margin: ((size.unwrap_or(14.0) * 2.0).round() as u32)
      * RESOLUTION_SCALE,
    x_label_area: 40 * RESOLUTION_SCALE,
    y_label_area: 65 * RESOLUTION_SCALE,
    hide_x_ticks: false,
    hide_y_ticks: false,
  })
}

/// Split `Style[expr, …, n, …]` into its content and the font size `n`.
fn peel_label_font_size(value: &Expr) -> (&Expr, Option<f64>) {
  let Expr::FunctionCall { name, args } = value else {
    return (value, None);
  };
  if !crate::functions::graphics::is_style_wrapper(name) || args.is_empty() {
    return (value, None);
  }
  let size = args[1..].iter().find_map(|d| match d {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    _ => None,
  });
  match size {
    Some(_) => (&args[0], size),
    None => (value, None),
  }
}

/// Draw a field plot's `PlotLabel`, centred above its plot area.
fn inject_field_plot_label(
  svg: &mut String,
  area: &crate::functions::plot::PlotArea,
  (label, size): &(String, Option<f64>),
) {
  if label.trim().is_empty() {
    return;
  }
  let sf = f64::from(crate::functions::plot::RESOLUTION_SCALE);
  let font = size.unwrap_or(14.0) * sf;
  let cx = area.plot_x0 + area.plot_w / 2.0;
  let y = (area.plot_y0 - sf * 8.0).max(font);
  svg.push_str(&format!(
    "<text x=\"{cx:.1}\" y=\"{y:.1}\" text-anchor=\"middle\" \
     font-family=\"sans-serif\" font-size=\"{font:.0}\" fill=\"#333\">{label}</text>\n",
  ));
}

fn parse_field_options(args: &[Expr], start: usize) -> (u32, u32, bool) {
  let mut svg_width = DEFAULT_WIDTH;
  let mut svg_height = DEFAULT_WIDTH; // Square default for field plots
  let mut full_width = false;
  for opt in &args[start..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && matches!(pattern.as_ref(), Expr::Identifier(name) if name == "ImageSize")
      // Square default aspect: field plots have AspectRatio -> 1, so
      // ImageSize -> n means n x n.
      && let Some((w, h, fw)) =
        parse_image_size(replacement, DEFAULT_WIDTH, DEFAULT_WIDTH)
    {
      svg_width = w;
      svg_height = h;
      full_width = fw;
    }
  }
  (svg_width, svg_height, full_width)
}

/// `Epilog -> …` primitives for `VectorPlot`, drawn over the finished field
/// with `plot_epilog::render_epilog_svg` — the same mechanism `DensityPlot`/
/// `ContourPlot` already use via `DensityContourOptions`. Evaluated eagerly
/// so primitives that reference a Manipulate's current control values (e.g.
/// `Disk[place1]`) resolve to numeric coordinates.
fn parse_vector_plot_epilog(args: &[Expr], start: usize) -> Vec<Expr> {
  parse_overlay_primitives(args, start, "Epilog")
}

/// The primitives of an `Epilog`/`Prolog` option among `args[start..]`,
/// evaluated eagerly. Empty when the option is absent or `None`.
fn parse_overlay_primitives(
  args: &[Expr],
  start: usize,
  option: &str,
) -> Vec<Expr> {
  for opt in args.iter().skip(start) {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && matches!(pattern.as_ref(), Expr::Identifier(name) if name == option)
    {
      let val = evaluate_expr_to_expr(replacement)
        .unwrap_or_else(|_| (**replacement).clone());
      return match val {
        Expr::List(ref items) => items.to_vec(),
        Expr::Identifier(ref s) if s == "None" => Vec::new(),
        other => vec![other],
      };
    }
  }
  Vec::new()
}

/// Draw a `Prolog`'s primitives under a field plot: spliced in right after
/// the canvas background rect, so the axes, shading and contours that follow
/// paint over them.
fn splice_field_plot_prolog(
  svg: &mut String,
  area: &crate::functions::plot::PlotArea,
  prolog: &[Expr],
) {
  if prolog.is_empty() {
    return;
  }
  let prolog_area = crate::functions::plot_epilog::PlotArea {
    x0: area.plot_x0,
    y0: area.plot_y0,
    w: area.plot_w,
    h: area.plot_h,
    x_min: area.x_min,
    x_max: area.x_max,
    y_min: area.y_min,
    y_max: area.y_max,
    scale: f64::from(RESOLUTION_SCALE),
  };
  let overlay = crate::functions::plot_epilog::render_epilog_svg(
    prolog,
    &prolog_area,
    "prolog",
  );
  crate::functions::plot::splice_after_background(svg, &overlay);
}

/// Contour level specification from the Contours option.
enum ContourSpec {
  /// Contours -> n: n equally spaced levels
  Count(usize),
  /// Contours -> {v1, v2, ...}: explicit levels
  Levels(Vec<f64>),
  /// Contours -> Automatic: ~10 levels at "nice" round values
  Automatic,
}

/// Options shared by the density / contour plot family.
#[allow(clippy::struct_excessive_bools)]
struct DensityContourOptions {
  svg_width: u32,
  svg_height: u32,
  full_width: bool,
  color_function: Option<String>,
  /// `ColorFunction -> f` for an `f` that isn't a recognized named gradient
  /// (a pure function, a user-defined symbol, …): applied per-cell via the
  /// evaluator, same as `ArrayPlot`'s `color_function_expr`.
  color_function_expr: Option<Expr>,
  /// `ColorFunctionScaling -> False` passes each cell's raw value to
  /// `color_function_expr`; the default (`True`) passes the value rescaled
  /// to the 0..1 range of the sampled data.
  color_function_scaling: bool,
  contours: ContourSpec,
  contour_shading: bool,
  /// `Mesh -> n`: how many mesh lines each mesh function contributes.
  mesh: Option<usize>,
  /// `MeshFunctions -> {f1, …}`: the functions whose level curves the
  /// mesh lines are. Empty means the coordinates themselves.
  mesh_functions: Vec<Expr>,
  /// `ContourStyle -> {Thick, Blue}`: how the contour lines are drawn.
  /// `None` keeps the default thin dark-grey line.
  contour_style: Option<crate::functions::plot::SeriesStyle>,
  /// `ContourStyle -> {{Black, Thick}, {Red, Thick}}`: one style per
  /// contour function, cycled when there are more functions than styles.
  /// Empty unless every entry of the option is itself a list or Directive.
  contour_styles: Vec<Option<crate::functions::plot::SeriesStyle>>,
  /// `BoundaryStyle -> style`: draws the edge of the region a
  /// `RegionFunction` (or an undefined value) cuts out of the plot. `None`
  /// (the default, as in Wolfram) draws no boundary.
  boundary_style: Option<crate::functions::plot::SeriesStyle>,
  /// `FrameLabel -> {bottom, left}`: the labels along the frame edges.
  frame_labels: crate::functions::plot::FrameLabels,
  /// `FrameTicks`: whether the bottom (x) and left (y) edges keep their
  /// tick marks and labels.
  frame_ticks: (bool, bool),
  /// `Frame -> False` leaves out the box (and its ticks) around the plot.
  frame: bool,
  /// `Epilog -> {…}`: graphics primitives drawn over the finished plot,
  /// in data coordinates.
  epilog: Vec<Expr>,
  /// `Prolog -> {…}`: graphics primitives drawn under the plot.
  prolog: Vec<Expr>,
  /// `PlotLabel -> …`: the caption drawn above the plot, already typeset
  /// as SVG markup, alongside the font size its outer `Style` asked for.
  plot_label: Option<(String, Option<f64>)>,
  /// `RegionFunction -> f`: keep only grid cells where `f[x, y, z]` (the
  /// plotted value `z`) is `True`; other cells are left blank, matching
  /// `Plot3D`/`ContourPlot3D`'s `RegionFunction`.
  region_function: Option<Expr>,
  /// `ContourLabels -> True` (or a function): label each contour line with
  /// its level.
  contour_labels: bool,
  /// `PlotLegends -> Automatic`: a colour bar right of the plot with one
  /// swatch per shaded band, labelled at the contour levels.
  bar_legend: bool,
  /// `PlotRange -> {zmin, zmax}` (or the third part of a three-axis
  /// range): the values the colours span, instead of the sampled ones.
  value_range: Option<(f64, f64)>,
  /// `ClippingStyle`: how a cell outside `value_range` is drawn.
  clipping: Clipping,
}

/// How a density cell whose value lies outside an explicit `PlotRange` is
/// drawn, as `ClippingStyle` sets it.
#[derive(Clone, Copy)]
enum Clipping {
  /// `None` (Wolfram's default): the cell is left blank.
  Blank,
  /// `Automatic`: the colour of the end of the range it is beyond.
  Saturate,
  /// A colour (or `{below, above}` colours); `None` leaves that side blank.
  Colors(Option<(u8, u8, u8)>, Option<(u8, u8, u8)>),
}

impl Clipping {
  fn parse(value: &Expr) -> Self {
    let rgb = |e: &Expr| {
      parse_color(e).map(|c| {
        (
          (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
          (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
          (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
        )
      })
    };
    match value {
      Expr::Identifier(s) if s == "Automatic" => Clipping::Saturate,
      Expr::List(sides) if sides.len() == 2 => {
        Clipping::Colors(rgb(&sides[0]), rgb(&sides[1]))
      }
      other => match rgb(other) {
        Some(c) => Clipping::Colors(Some(c), Some(c)),
        None => Clipping::Blank,
      },
    }
  }
}

/// The value range of a `PlotRange -> {zmin, zmax}` (or `{{…}, {…},
/// {zmin, zmax}}`) given to a density or contour plot.
fn plot_value_range(value: &Expr) -> Option<(f64, f64)> {
  let pair = |e: &Expr| match e {
    Expr::List(ends) if ends.len() == 2 => {
      let lo = try_eval_to_f64(&ends[0])?;
      let hi = try_eval_to_f64(&ends[1])?;
      (lo < hi).then_some((lo, hi))
    }
    _ => None,
  };
  match value {
    Expr::List(parts) if parts.len() == 3 => pair(&parts[2]),
    other => pair(other),
  }
}

/// The per-contour styles of a `ContourStyle` value whose entries are all
/// lists or `Directive[…]`s (`{{Black, Thick}, {Red, Thick}}`,
/// `{Directive[Red, Dashed], Directive[Blue]}`); empty for a single
/// directive list such as `{Thick, Blue}`.
fn per_contour_styles(
  value: &Expr,
) -> Vec<Option<crate::functions::plot::SeriesStyle>> {
  let val = evaluate_expr_to_expr(value).unwrap_or_else(|_| value.clone());
  match &val {
    Expr::List(items)
      if !items.is_empty() && items.iter().all(|i| {
        matches!(i, Expr::List(_))
          || matches!(i, Expr::FunctionCall { name, .. } if name == "Directive")
      }) =>
    {
      items
        .iter()
        .map(crate::functions::plot::parse_style_directives)
        .collect()
    }
    _ => Vec::new(),
  }
}

impl DensityContourOptions {
  /// The style of the `index`-th contour function: its own entry of a
  /// per-contour `ContourStyle` (cycled), else the single shared style.
  fn style_for_contour(
    &self,
    index: usize,
  ) -> Option<&crate::functions::plot::SeriesStyle> {
    if self.contour_styles.is_empty() {
      self.contour_style.as_ref()
    } else {
      self.contour_styles[index % self.contour_styles.len()].as_ref()
    }
  }

  /// `(expr, scaling)` for `scaled_color`/`band_color`, when `ColorFunction`
  /// was given a pure function or symbol rather than a named gradient.
  fn cf_expr(&self) -> Option<(&Expr, bool)> {
    self
      .color_function_expr
      .as_ref()
      .map(|e| (e, self.color_function_scaling))
  }
}

/// Parse ImageSize, ColorFunction, Contours, ContourShading, Mesh,
/// MeshFunctions, ContourStyle, FrameLabel and Epilog options.
fn parse_density_contour_options(
  args: &[Expr],
  start: usize,
) -> DensityContourOptions {
  let (svg_width, mut svg_height, full_width) =
    parse_field_options(args, start);
  // `AspectRatio -> r` (height / width) reshapes the picture; an
  // `ImageSize -> {w, h}` pair already fixes both sides and wins.
  let explicit_size_pair = args[start..].iter().any(|opt| {
    matches!(opt, Expr::Rule { pattern, replacement }
      if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "ImageSize")
        && matches!(replacement.as_ref(), Expr::List(_)))
  });
  if !explicit_size_pair {
    for opt in &args[start..] {
      if let Expr::Rule {
        pattern,
        replacement,
      } = opt
        && matches!(pattern.as_ref(), Expr::Identifier(n) if n == "AspectRatio")
      {
        let value = evaluate_expr_to_expr(replacement)
          .unwrap_or_else(|_| replacement.as_ref().clone());
        if let Some(r) = try_eval_to_f64(&value).filter(|r| *r > 0.0) {
          svg_height = ((svg_width as f64 * r).round() as u32).max(1);
        }
      }
    }
  }
  let mut color_function = None;
  let mut contours = ContourSpec::Automatic;
  let mut contour_shading = true;
  let mut mesh: Option<usize> = None;
  let mut mesh_functions: Vec<Expr> = Vec::new();
  let mut contour_style = None;
  let mut contour_styles = Vec::new();
  let mut boundary_style = None;
  let mut frame_labels = crate::functions::plot::FrameLabels::default();
  let mut frame_ticks = (true, true);
  let mut frame = true;
  let mut epilog: Vec<Expr> = Vec::new();
  let mut region_function: Option<Expr> = None;
  let mut color_function_expr: Option<Expr> = None;
  let mut color_function_scaling = true;
  let mut contour_labels = false;
  let mut bar_legend = false;
  let mut value_range: Option<(f64, f64)> = None;
  let mut clipping = Clipping::Blank;
  for opt in &args[start..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && let Expr::Identifier(name) = pattern.as_ref()
    {
      // An option's value may still be the symbol it was written with
      // (`Contours -> con` inside a Manipulate), so resolve it first.
      let resolved = match replacement.as_ref() {
        v @ (Expr::Integer(_) | Expr::Real(_) | Expr::List(_)) => v.clone(),
        other => evaluate_expr_to_expr(other).unwrap_or_else(|_| other.clone()),
      };
      let replacement = &resolved;
      match name.as_str() {
        "ColorFunction" => {
          if let Some(s) = color_function_scheme_name(replacement) {
            color_function = Some(s);
          } else if !matches!(replacement, Expr::Identifier(v) if v == "Automatic" || v == "None")
          {
            color_function_expr = Some(replacement.clone());
          }
        }
        "ColorFunctionScaling" => {
          if let Expr::Identifier(v) = replacement
            && v == "False"
          {
            color_function_scaling = false;
          }
        }
        "Contours" => match replacement {
          Expr::Integer(n) if *n > 0 => {
            contours = ContourSpec::Count(*n as usize);
          }
          Expr::List(items) => {
            let levels: Vec<f64> = items
              .iter()
              .filter_map(|item| {
                let v = evaluate_expr_to_expr(item).unwrap_or(item.clone());
                try_eval_to_f64(&v)
              })
              .filter(|v| v.is_finite())
              .collect();
            if !levels.is_empty() {
              contours = ContourSpec::Levels(levels);
            }
          }
          _ => {}
        },
        // `ContourShading -> None` is the documented way to ask for
        // contour lines with nothing filled in between; `False` is the
        // older spelling.
        "ContourShading" => {
          if matches!(replacement, Expr::Identifier(v) if v == "False" || v == "None")
          {
            contour_shading = false;
          }
        }
        // `Mesh -> n` draws n mesh lines per mesh function; `Mesh -> None`
        // (the default) draws none.
        "Mesh" => match replacement {
          Expr::Integer(n) if *n >= 0 => mesh = Some(*n as usize),
          Expr::Identifier(v) if v == "None" || v == "False" => mesh = None,
          Expr::Identifier(v) if v == "All" || v == "Automatic" => {
            mesh = Some(10);
          }
          _ => {}
        },
        // `MeshFunctions -> {f1, f2, …}`: the mesh lines are level curves
        // of each function, the way the contours are of the plotted one.
        "MeshFunctions" => {
          match replacement {
            Expr::List(items) => {
              mesh_functions = items.iter().cloned().collect();
            }
            // A lone function stands for a list of one: `MeshFunctions ->
            // (f[#1, #2] &)`.
            f @ Expr::Function { .. } => mesh_functions = vec![f.clone()],
            _ => {}
          }
        }
        "ContourStyle" => {
          contour_style =
            crate::functions::plot::parse_style_directives(replacement);
          contour_styles = per_contour_styles(replacement);
        }
        "BoundaryStyle" => {
          boundary_style = match replacement {
            Expr::Identifier(v) if v == "None" || v == "Automatic" => None,
            other => crate::functions::plot::parse_style_directives(other),
          };
        }
        "FrameLabel" => {
          frame_labels = crate::functions::plot::parse_frame_label(replacement);
        }
        "Frame" => {
          frame = !matches!(replacement, Expr::Identifier(v) if v == "False");
        }
        "FrameTicks" => {
          frame_ticks =
            crate::functions::plot::frame_tick_axes_visible(replacement);
        }
        // Evaluate now so primitives inside (e.g. `Point[{a, b}]` marking a
        // Manipulate's current parameters) resolve to numeric coordinates
        // while the surrounding variable bindings are still live.
        "Epilog" => {
          let val = evaluate_expr_to_expr(replacement)
            .unwrap_or_else(|_| replacement.clone());
          epilog = match val {
            Expr::List(ref items) => items.to_vec(),
            Expr::Identifier(ref s) if s == "None" => Vec::new(),
            other => vec![other],
          };
        }
        "RegionFunction" => {
          region_function = Some(replacement.clone());
        }
        "ContourLabels" => {
          contour_labels = !matches!(replacement, Expr::Identifier(v)
            if v == "False" || v == "None");
        }
        "PlotRange" => value_range = plot_value_range(replacement),
        "ClippingStyle" => clipping = Clipping::parse(replacement),
        "PlotLegends" => {
          bar_legend = matches!(replacement, Expr::Identifier(v)
            if v == "Automatic");
        }
        _ => {}
      }
    }
  }
  if !frame {
    frame_ticks = (false, false);
  }
  DensityContourOptions {
    svg_width,
    svg_height,
    full_width,
    color_function,
    color_function_expr,
    color_function_scaling,
    contours,
    contour_shading,
    mesh,
    mesh_functions,
    contour_style,
    contour_styles,
    boundary_style,
    frame_labels,
    frame_ticks,
    frame,
    epilog,
    prolog: parse_overlay_primitives(args, start, "Prolog"),
    plot_label: parse_field_plot_label(args, start),
    region_function,
    contour_labels,
    bar_legend,
    value_range,
    clipping,
  }
}

/// Whether `(x, y, z)` satisfies `region`, a `RegionFunction ->
/// Function[{x, y, z}, …]` value: called positionally exactly like
/// `Plot3D`/`ContourPlot3D`'s region function, so it does not need the
/// plot's own x/y variable names substituted in first.
fn region_allows(region: &Expr, x: f64, y: f64, z: f64) -> bool {
  let call = Expr::CurriedCall {
    func: Box::new(region.clone()),
    args: vec![Expr::Real(x), Expr::Real(y), Expr::Real(z)],
  };
  matches!(
    evaluate_expr_to_expr(&call),
    Ok(Expr::Identifier(ref s)) if s == "True"
  )
}

impl DensityContourOptions {
  /// Whether a `FrameLabel` was given for the bottom or the left edge —
  /// the two that need room reserved outside the plotting area.
  fn has_outer_frame_labels(&self) -> bool {
    !self.frame_labels.bottom.is_empty() || !self.frame_labels.left.is_empty()
  }

  /// The stroke colour and weight for contour lines: whatever
  /// `ContourStyle` asked for, otherwise the default thin dark grey.
  ///
  /// `render_contour_lines_styled` scales its weight by
  /// `render_width / 1000`, so a thickness given in display pixels
  /// converts by the picture's own width.
  fn contour_stroke(&self) -> (String, f64) {
    self.contour_stroke_for(self.contour_style.as_ref())
  }

  fn contour_stroke_for(
    &self,
    style: Option<&crate::functions::plot::SeriesStyle>,
  ) -> (String, f64) {
    let Some(style) = style else {
      return (CONTOUR_LINE_COLOR.to_string(), CONTOUR_LINE_WEIGHT);
    };
    let color = style.color.map_or_else(
      || CONTOUR_LINE_COLOR.to_string(),
      super::graphics::Color::to_svg_rgb,
    );
    let weight = match style.thickness {
      Some(t) if self.svg_width > 0 => t * 1000.0 / self.svg_width as f64,
      _ => CONTOUR_LINE_WEIGHT,
    };
    (color, weight)
  }
}

/// Axes for a field plot, with room reserved outside the plotting area for
/// whatever `FrameLabel` asks to be written there.
fn field_plot_axes(
  x_range: (f64, f64),
  y_range: (f64, f64),
  opts: &DensityContourOptions,
) -> Result<crate::functions::plot::PlotArea, InterpreterError> {
  let needs_margins = opts.has_outer_frame_labels()
    || opts.plot_label.is_some()
    || opts.frame_ticks != (true, true);
  let margins = needs_margins.then_some({
    let top_margin = match &opts.plot_label {
      Some((_, size)) => {
        ((size.unwrap_or(14.0) * 2.0).round() as u32) * RESOLUTION_SCALE
      }
      None => 10 * RESOLUTION_SCALE,
    };
    // Without tick labels only a `FrameLabel` still needs room outside.
    let (x_ticks, y_ticks) = opts.frame_ticks;
    let outer = opts.has_outer_frame_labels();
    // On a small `ImageSize` the full tick-label areas would swallow the
    // plot (or even go negative), so they never take more than 40% of it.
    let x_label_area = if x_ticks || outer {
      ((40 + 24) * RESOLUTION_SCALE)
        .min(opts.svg_height * RESOLUTION_SCALE * 2 / 5)
    } else {
      0
    };
    let y_label_area = if y_ticks || outer {
      ((65 + 20) * RESOLUTION_SCALE)
        .min(opts.svg_width * RESOLUTION_SCALE * 2 / 5)
    } else {
      0
    };
    crate::functions::plot::MarginOverrides {
      top_margin,
      x_label_area,
      y_label_area,
      hide_x_ticks: !opts.frame_ticks.0,
      hide_y_ticks: !opts.frame_ticks.1,
    }
  });
  crate::functions::plot::generate_axes_only_opts(
    x_range,
    y_range,
    opts.svg_width,
    opts.svg_height,
    opts.full_width,
    None,
    margins.as_ref(),
  )
}

/// Draw a field plot's `Epilog` primitives and `FrameLabel` text over the
/// finished picture. Called with the closing `</svg>` already stripped, so
/// the fragments land on top of everything drawn before them.
fn push_field_plot_overlays(
  svg: &mut String,
  area: &crate::functions::plot::PlotArea,
  opts: &DensityContourOptions,
) {
  splice_field_plot_prolog(svg, area, &opts.prolog);
  if let Some(label) = &opts.plot_label {
    inject_field_plot_label(svg, area, label);
  }
  push_epilog_primitives(svg, area, &opts.epilog);
  push_frame_labels(svg, area, &opts.frame_labels);
}

/// Draw a field plot's `Epilog` primitives over everything drawn so far
/// (the closing `</svg>` must already be stripped from `svg`).
fn push_epilog_primitives(
  svg: &mut String,
  area: &crate::functions::plot::PlotArea,
  epilog: &[Expr],
) {
  if epilog.is_empty() {
    return;
  }
  let epilog_area = crate::functions::plot_epilog::PlotArea {
    x0: area.plot_x0,
    y0: area.plot_y0,
    w: area.plot_w,
    h: area.plot_h,
    x_min: area.x_min,
    x_max: area.x_max,
    y_min: area.y_min,
    y_max: area.y_max,
    scale: f64::from(RESOLUTION_SCALE),
  };
  svg.push_str(&crate::functions::plot_epilog::render_epilog_svg(
    epilog,
    &epilog_area,
    "epilog",
  ));
}

/// Where `ContourLabels -> True` writes each level: the middle of the
/// level's longest contour line, in the coordinates `marching_squares_segments`
/// was given. Levels whose lines are too short to carry a label are skipped.
fn contour_label_points(
  grid: &[Vec<f64>],
  levels: &[f64],
  x0: f64,
  y0: f64,
  cell_w: f64,
  cell_h: f64,
) -> Vec<(f64, (f64, f64))> {
  let mut points = Vec::new();
  for &level in levels {
    let segments =
      marching_squares_segments(grid, level, x0, y0, cell_w, cell_h);
    let chains = chain_segments(&segments);
    let Some(chain) = chains.iter().max_by_key(|c| c.len()) else {
      continue;
    };
    if chain.len() < 4 {
      continue;
    }
    points.push((level, chain[chain.len() / 2]));
  }
  points
}

/// `ContourLabels -> True`: write each contour's level on its longest
/// line, centred on the line with a light halo so it stays legible over
/// the shading.
fn push_contour_labels(
  svg: &mut String,
  grid: &[Vec<f64>],
  levels: &[f64],
  area: &crate::functions::plot::PlotArea,
) {
  let (.., label_fill, _) = plot_theme();
  let cell_w = area.plot_w / FIELD_GRID as f64;
  let cell_h = area.plot_h / FIELD_GRID as f64;
  let font = 11.0 * RESOLUTION_SCALE as f64;
  for (level, (x, y)) in contour_label_points(
    grid,
    levels,
    area.plot_x0,
    area.plot_y0,
    cell_w,
    cell_h,
  ) {
    svg.push_str(&format!(
      "<text x=\"{x:.1}\" y=\"{y:.1}\" text-anchor=\"middle\" dominant-baseline=\"central\" font-family=\"sans-serif\" font-size=\"{font:.0}\" fill=\"{label_fill}\" stroke=\"white\" stroke-width=\"{:.1}\" stroke-opacity=\"0.8\" paint-order=\"stroke\">{}</text>\n",
      3.0 * RESOLUTION_SCALE as f64,
      crate::functions::plot::format_tick(level)
    ));
  }
}

/// Extra display-pixel width a bar legend adds right of the plot.
const BAR_LEGEND_STRIP: u32 = 53;

/// `PlotLegends -> Automatic`: widen the picture and draw a vertical colour
/// bar with one swatch per contour band (lowest at the bottom), labelled at
/// the contour levels — Wolfram's `BarLegend`.
fn push_bar_legend(
  svg: &mut String,
  area: &crate::functions::plot::PlotArea,
  opts: &DensityContourOptions,
  levels: &[f64],
  (v_min, v_max): (f64, f64),
) {
  let sf = RESOLUTION_SCALE as f64;
  // Label every other level once there are too many to fit.
  let stride = levels.len().div_ceil(8).max(1);
  let labels: Vec<(f64, String)> = levels
    .iter()
    .step_by(stride)
    .map(|&l| (l, crate::functions::plot::format_tick(l)))
    .collect();
  // The strip must reach past the widest label; a tight right margin (an
  // explicit `ImagePadding`) would otherwise clip it at the canvas edge.
  let widest = labels
    .iter()
    .map(|(_, t)| t.chars().count())
    .max()
    .unwrap_or(0) as f64;
  let right_margin =
    (area.render_width as f64 - area.plot_x0 - area.plot_w) / sf;
  let needed = 27.0 + 13.0 + 6.0 + widest * 14.0 * 0.6 + 4.0;
  let strip =
    BAR_LEGEND_STRIP.max((needed - right_margin).ceil().max(0.0) as u32);
  let new_w = opts.svg_width + strip;
  let new_render_w = new_w * RESOLUTION_SCALE;
  let render_h = opts.svg_height * RESOLUTION_SCALE;
  // The background rectangle spans the old canvas; stretch it first.
  let old_bg = format!("width=\"{}\" height=\"{render_h}\"", area.render_width);
  let new_bg = format!("width=\"{new_render_w}\" height=\"{render_h}\"");
  *svg = svg.replacen(&old_bg, &new_bg, 1);
  rewrite_svg_header(
    svg,
    new_w,
    opts.svg_height,
    new_render_w,
    render_h,
    opts.full_width,
  );

  let (.., label_fill, _) = plot_theme();
  let bar_w = 13.0 * sf;
  let bar_h = area.plot_h * 0.625;
  let bar_x = area.plot_x0 + area.plot_w + 27.0 * sf;
  let bar_y = area.plot_y0 + (area.plot_h - bar_h) / 2.0;
  let span = v_max - v_min;
  let frac = |v: f64| {
    if span > 0.0 {
      ((v - v_min) / span).clamp(0.0, 1.0)
    } else {
      0.5
    }
  };
  let mut bounds = Vec::with_capacity(levels.len() + 2);
  bounds.push(v_min);
  bounds.extend_from_slice(levels);
  bounds.push(v_max);
  for band in 0..bounds.len() - 1 {
    let (r, g, b) = band_color(
      band,
      levels,
      v_min,
      v_max,
      opts.color_function.as_deref(),
      opts.cf_expr(),
    );
    let y_top = bar_y + bar_h * (1.0 - frac(bounds[band + 1]));
    let y_bot = bar_y + bar_h * (1.0 - frac(bounds[band]));
    svg.push_str(&format!(
      "<rect x=\"{bar_x:.1}\" y=\"{y_top:.1}\" width=\"{bar_w:.1}\" height=\"{:.1}\" fill=\"rgb({r},{g},{b})\"/>\n",
      y_bot - y_top
    ));
  }
  svg.push_str(&format!(
    "<rect x=\"{bar_x:.1}\" y=\"{bar_y:.1}\" width=\"{bar_w:.1}\" height=\"{bar_h:.1}\" fill=\"none\" stroke=\"rgb(110,110,110)\" stroke-width=\"{:.1}\"/>\n",
    0.8 * sf
  ));
  for (level, text) in &labels {
    svg.push_str(&format!(
      "<text x=\"{:.1}\" y=\"{:.1}\" font-family=\"sans-serif\" font-size=\"{:.0}\" fill=\"{label_fill}\" dominant-baseline=\"central\">{}</text>\n",
      bar_x + bar_w + 6.0 * sf,
      bar_y + bar_h * (1.0 - frac(*level)),
      sf * 14.0,
      text
    ));
  }
}

/// A standalone `BarLegend[{colors, {min, max}}]` (or `BarLegend[colors]`
/// over 0..1): a vertical continuous colour bar `bar_height` display
/// pixels tall, labelled at round values to its right. `colors` is a
/// gradient name or a colour function of the rescaled value.
pub(crate) fn bar_legend_svg(args: &[Expr], bar_height: f64) -> Option<String> {
  let (colors, (lo, hi)) = match args.first()? {
    Expr::List(spec) if spec.len() == 2 => match &spec[1] {
      Expr::List(range) if range.len() == 2 => (
        &spec[0],
        (try_eval_to_f64(&range[0])?, try_eval_to_f64(&range[1])?),
      ),
      _ => return None,
    },
    other => (other, (0.0, 1.0)),
  };
  if hi.partial_cmp(&lo) != Some(std::cmp::Ordering::Greater) {
    return None;
  }
  let name = color_function_scheme_name(colors);
  let color_at = |t: f64| match &name {
    Some(name) => apply_named_color_function(name, t),
    None => scaled_color(t, lo + t * (hi - lo), None, Some((colors, true))),
  };

  let (.., label_fill, _) = plot_theme();
  let (bar_w, pad) = (13.0, 4.0);
  let step = crate::functions::plot::nice_step(hi - lo, 5);
  let ticks: Vec<f64> = {
    let first = (lo / step - 1e-9).ceil() as i64;
    let last = (hi / step + 1e-9).floor() as i64;
    (first..=last).map(|k| k as f64 * step).collect()
  };
  let labels: Vec<String> = ticks
    .iter()
    .map(|&v| crate::functions::plot::format_tick_with_step(v, step))
    .collect();
  let label_w =
    labels.iter().map(|l| l.chars().count()).max().unwrap_or(1) as f64 * 7.5;
  let width = pad + bar_w + 6.0 + label_w + pad;
  let height = bar_height + 2.0 * pad + 8.0;
  let top = pad + 4.0;
  let mut svg = format!(
    "<svg width=\"{width:.0}\" height=\"{height:.0}\" viewBox=\"0 0 {width:.0} {height:.0}\" xmlns=\"http://www.w3.org/2000/svg\">\n"
  );
  // The bar, as thin horizontal bands from the low end (bottom) upward.
  const BANDS: usize = 64;
  let band_h = bar_height / BANDS as f64;
  for k in 0..BANDS {
    let (r, g, b) = color_at((k as f64 + 0.5) / BANDS as f64);
    let y = top + bar_height - (k + 1) as f64 * band_h;
    svg.push_str(&format!(
      "<rect x=\"{pad:.1}\" y=\"{y:.2}\" width=\"{bar_w:.1}\" height=\"{:.2}\" fill=\"rgb({r},{g},{b})\"/>\n",
      band_h + 0.3
    ));
  }
  svg.push_str(&format!(
    "<rect x=\"{pad:.1}\" y=\"{top:.1}\" width=\"{bar_w:.1}\" height=\"{bar_height:.1}\" fill=\"none\" stroke=\"rgb(110,110,110)\" stroke-width=\"0.8\"/>\n"
  ));
  for (v, label) in ticks.iter().zip(&labels) {
    let y = top + bar_height * (1.0 - (v - lo) / (hi - lo));
    svg.push_str(&format!(
      "<line x1=\"{:.1}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\" stroke=\"rgb(110,110,110)\" stroke-width=\"0.8\"/>\n",
      pad + bar_w - 3.0,
      pad + bar_w
    ));
    svg.push_str(&format!(
      "<text x=\"{:.1}\" y=\"{y:.1}\" font-family=\"sans-serif\" font-size=\"11\" fill=\"{label_fill}\" dominant-baseline=\"central\">{label}</text>\n",
      pad + bar_w + 6.0
    ));
  }
  svg.push_str("</svg>\n");
  Some(svg)
}

/// Write a plot's `FrameLabel` text along the frame edges it names.
fn push_frame_labels(
  svg: &mut String,
  area: &crate::functions::plot::PlotArea,
  frame_labels: &crate::functions::plot::FrameLabels,
) {
  if frame_labels.bottom.is_empty()
    && frame_labels.left.is_empty()
    && frame_labels.top.is_empty()
    && frame_labels.right.is_empty()
  {
    return;
  }
  let (.., label_fill, title_fill) = plot_theme();
  let mut plot_opts = crate::functions::plot::PlotOptions::default();
  let non_empty = |s: &String| (!s.is_empty()).then(|| s.clone());
  plot_opts.frame_label_bottom = non_empty(&frame_labels.bottom);
  plot_opts.frame_label_left = non_empty(&frame_labels.left);
  plot_opts.frame_label_top = non_empty(&frame_labels.top);
  plot_opts.frame_label_right = non_empty(&frame_labels.right);
  svg.push_str(&crate::functions::plot::plot_labels_svg(
    &plot_opts,
    (area.plot_x0, area.plot_y0, area.plot_w, area.plot_h),
    (area.x_min, area.x_max, area.y_min, area.y_max),
    0.0,
    RESOLUTION_SCALE as f64,
    label_fill,
    title_fill,
    f64::INFINITY,
  ));
}

/// Map value to a blue-white-red color
fn value_to_color(v: f64, v_min: f64, v_max: f64) -> (u8, u8, u8) {
  let range = v_max - v_min;
  let t = if range.abs() < f64::EPSILON {
    0.5
  } else {
    ((v - v_min) / range).clamp(0.0, 1.0)
  };

  if t < 0.5 {
    // Blue to white
    let s = t * 2.0;
    let r = (s * 255.0) as u8;
    let g = (s * 255.0) as u8;
    let b = 255;
    (r, g, b)
  } else {
    // White to red
    let s = (t - 0.5) * 2.0;
    let r = 255;
    let g = ((1.0 - s) * 255.0) as u8;
    let b = ((1.0 - s) * 255.0) as u8;
    (r, g, b)
  }
}

/// Default density / contour gradient: dark blue → azure → teal → green →
/// yellow, approximating Mathematica's default DensityPlot color scheme.
fn density_gradient(t: f64) -> (u8, u8, u8) {
  const STOPS: [(f64, f64, f64); 8] = [
    (53.0, 42.0, 160.0),
    (28.0, 83.0, 217.0),
    (22.0, 128.0, 224.0),
    (15.0, 168.0, 200.0),
    (62.0, 192.0, 152.0),
    (140.0, 199.0, 96.0),
    (219.0, 194.0, 67.0),
    (252.0, 232.0, 38.0),
  ];
  let t = if t.is_finite() {
    t.clamp(0.0, 1.0)
  } else {
    0.5
  };
  let x = t * (STOPS.len() - 1) as f64;
  let i = (x.floor() as usize).min(STOPS.len() - 2);
  let f = x - i as f64;
  let a = STOPS[i];
  let b = STOPS[i + 1];
  (
    (a.0 + (b.0 - a.0) * f).round() as u8,
    (a.1 + (b.1 - a.1) * f).round() as u8,
    (a.2 + (b.2 - a.2) * f).round() as u8,
  )
}

/// Map a scaled value t in [0,1] through the plot's color function:
/// a named gradient when ColorFunction -> "Name" was given, a pure function
/// or user-defined symbol applied per-cell when `color_function_expr` was
/// given instead, otherwise the default density gradient. `raw` is the
/// cell's unscaled value, passed to `color_function_expr` in place of `t`
/// when `ColorFunctionScaling -> False`.
fn scaled_color(
  t: f64,
  raw: f64,
  color_function: Option<&str>,
  color_function_expr: Option<(&Expr, bool)>,
) -> (u8, u8, u8) {
  if let Some(name) = color_function {
    return apply_named_color_function(name, t);
  }
  if let Some((cf_expr, scaling)) = color_function_expr {
    let arg = if scaling { t } else { raw };
    let call = Expr::CurriedCall {
      func: Box::new(cf_expr.clone()),
      args: vec![Expr::Real(arg)],
    };
    if let Some(c) = evaluate_expr_to_expr(&call)
      .ok()
      .and_then(|r| parse_color(&r))
    {
      return (
        (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
        (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
        (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
      );
    }
    let gray = ((1.0 - t) * 255.0).round() as u8;
    return (gray, gray, gray);
  }
  density_gradient(t)
}

/// Robust value range over a set of function samples — an IQR fence
/// excludes extreme outliers so a singularity in the sampled domain (e.g. a
/// pole in the plotted function) can't collapse the color scale and contour
/// levels onto a handful of pixels around it, leaving the rest of the plot a
/// single flat color. Mirrors `plot::sampled_y_range`'s approach for `Plot`'s
/// y-range, applied here to a 2D function's sampled values.
fn robust_value_range(values: &[f64]) -> (f64, f64) {
  let mut vs: Vec<f64> =
    values.iter().copied().filter(|v| v.is_finite()).collect();
  if vs.is_empty() {
    return (f64::INFINITY, f64::NEG_INFINITY);
  }
  vs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
  let n = vs.len();
  if n == 1 {
    return (vs[0], vs[0]);
  }

  let q1 = vs[n / 4];
  let q3 = vs[3 * n / 4];
  let iqr = q3 - q1;

  // If IQR is negligible, no outliers — use the full min/max.
  if iqr < 1e-10 {
    return (vs[0], vs[n - 1]);
  }

  let fence_lo = q1 - 3.0 * iqr;
  let fence_hi = q3 + 3.0 * iqr;

  let v_min = vs.iter().copied().find(|&v| v >= fence_lo).unwrap_or(vs[0]);
  let v_max = vs
    .iter()
    .rev()
    .copied()
    .find(|&v| v <= fence_hi)
    .unwrap_or(vs[n - 1]);

  (v_min, v_max)
}

/// Scale a value into [0,1] over [v_min, v_max] (0.5 for a flat range).
fn scale_value(v: f64, v_min: f64, v_max: f64) -> f64 {
  let range = v_max - v_min;
  if range.abs() < f64::EPSILON {
    0.5
  } else {
    ((v - v_min) / range).clamp(0.0, 1.0)
  }
}

/// Render a sampled scalar grid (grid[col][row], row 0 = bottom) as a bitmap
/// stretched over the plot area. One pixel per sample; the SVG viewer's
/// smooth image scaling interpolates between samples, which matches
/// Mathematica's smooth density shading (and avoids the hairline seams that
/// per-cell <rect> tiles produce). Non-finite samples become transparent.
fn embed_grid_image(
  svg: &mut String,
  grid: &[Vec<f64>],
  v_min: f64,
  v_max: f64,
  x0: f64,
  y0: f64,
  w: f64,
  h: f64,
  color_function: Option<&str>,
  color_function_expr: Option<(&Expr, bool)>,
) {
  if !v_min.is_finite() || !v_max.is_finite() {
    return;
  }
  let colors = cell_colors(
    grid,
    (v_min, v_max),
    color_function,
    color_function_expr,
    Clipping::Saturate,
  );
  embed_color_image(svg, &colors, x0, y0, w, h);
}

/// The colour of each grid cell (`grid[col][row]`, row 0 at the bottom) on
/// a colour scale spanning `range`; a cell without a value, or one outside
/// the range that `clipping` leaves blank, has none.
fn cell_colors(
  grid: &[Vec<f64>],
  (v_min, v_max): (f64, f64),
  color_function: Option<&str>,
  color_function_expr: Option<(&Expr, bool)>,
  clipping: Clipping,
) -> Vec<Vec<Option<(u8, u8, u8)>>> {
  let color = |v: f64| {
    scaled_color(
      scale_value(v, v_min, v_max),
      v,
      color_function,
      color_function_expr,
    )
  };
  grid
    .iter()
    .map(|col| {
      col
        .iter()
        .map(|&v| {
          if !v.is_finite() {
            return None;
          }
          let side = if v < v_min {
            Some(false)
          } else if v > v_max {
            Some(true)
          } else {
            None
          };
          match (side, clipping) {
            (None, _) | (Some(_), Clipping::Saturate) => Some(color(v)),
            (Some(_), Clipping::Blank) => None,
            (Some(above), Clipping::Colors(below_c, above_c)) => {
              if above {
                above_c
              } else {
                below_c
              }
            }
          }
        })
        .collect()
    })
    .collect()
}

/// Embed a grid of cell colours (`colors[col][row]`, row 0 at the bottom;
/// `None` transparent) as an image filling the given plot rectangle.
fn embed_color_image(
  svg: &mut String,
  colors: &[Vec<Option<(u8, u8, u8)>>],
  x0: f64,
  y0: f64,
  w: f64,
  h: f64,
) {
  use base64::Engine as _;
  let cols = colors.len();
  let rows = colors.first().map_or(0, std::vec::Vec::len);
  if cols == 0 || rows == 0 {
    return;
  }
  let mut img = image::RgbaImage::new(cols as u32, rows as u32);
  for (i, col) in colors.iter().enumerate() {
    for (j, cell) in col.iter().enumerate() {
      let px = match cell {
        Some((r, g, b)) => image::Rgba([*r, *g, *b, 255]),
        None => image::Rgba([0, 0, 0, 0]),
      };
      img.put_pixel(i as u32, (rows - 1 - j) as u32, px);
    }
  }
  let mut png = Vec::new();
  if image::DynamicImage::ImageRgba8(img)
    .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
    .is_err()
  {
    return;
  }
  let b64 = base64::engine::general_purpose::STANDARD.encode(&png);
  svg.push_str(&format!(
    "<image x=\"{x0:.1}\" y=\"{y0:.1}\" width=\"{w:.1}\" height=\"{h:.1}\" preserveAspectRatio=\"none\" href=\"data:image/png;base64,{b64}\"/>\n"
  ));
}

/// Bilinearly resample a node grid (grid[col][row], row 0 = bottom) to
/// out_cols x out_rows samples whose pixel centers span the same area, so
/// coarse data grids render as smooth shading instead of visible pixels.
fn resample_node_grid(
  grid: &[Vec<f64>],
  out_cols: usize,
  out_rows: usize,
) -> Vec<Vec<f64>> {
  let cols = grid.len();
  let rows = if cols > 0 { grid[0].len() } else { 0 };
  if cols < 2 || rows < 2 {
    return grid.to_vec();
  }
  let mut out = vec![vec![f64::NAN; out_rows]; out_cols];
  for (i, out_col) in out.iter_mut().enumerate() {
    let x = (i as f64 + 0.5) / out_cols as f64 * (cols - 1) as f64;
    let i0 = (x.floor() as usize).min(cols - 2);
    let fx = x - i0 as f64;
    for (j, out_v) in out_col.iter_mut().enumerate() {
      let y = (j as f64 + 0.5) / out_rows as f64 * (rows - 1) as f64;
      let j0 = (y.floor() as usize).min(rows - 2);
      let fy = y - j0 as f64;
      *out_v = grid[i0][j0] * (1.0 - fx) * (1.0 - fy)
        + grid[i0 + 1][j0] * fx * (1.0 - fy)
        + grid[i0][j0 + 1] * (1.0 - fx) * fy
        + grid[i0 + 1][j0 + 1] * fx * fy;
    }
  }
  out
}

/// Draw the frame box around the plot area (Mathematica draws density and
/// contour plots with Frame -> True).
fn push_frame(svg: &mut String, x0: f64, y0: f64, w: f64, h: f64) {
  let (_, dark_gray, _, _, _) = plot_theme();
  svg.push_str(&format!(
    "<rect x=\"{x0:.1}\" y=\"{y0:.1}\" width=\"{w:.1}\" height=\"{h:.1}\" fill=\"none\" stroke=\"rgb({},{},{})\" stroke-width=\"{}\"/>\n",
    dark_gray.0, dark_gray.1, dark_gray.2, RESOLUTION_SCALE
  ));
}

/// Pick ~target contour levels at "nice" round values strictly inside
/// (v_min, v_max), like Mathematica's Contours -> Automatic.
fn nice_contour_levels(v_min: f64, v_max: f64, target: usize) -> Vec<f64> {
  let range = v_max - v_min;
  if !range.is_finite() || range <= 0.0 {
    return vec![];
  }
  let raw = range / (target as f64 + 1.0);
  // libm for cross-platform bit-identical results (see channel_to_u8)
  let mag = libm::pow(10.0, libm::floor(libm::log10(raw)));
  let norm = raw / mag;
  let step = if norm <= 1.0 {
    1.0
  } else if norm <= 2.0 {
    2.0
  } else if norm <= 5.0 {
    5.0
  } else {
    10.0
  } * mag;
  let mut levels = Vec::new();
  let mut k = libm::floor(v_min / step) + 1.0;
  while k * step < v_max - 1e-9 * step {
    if k * step > v_min + 1e-9 * step {
      levels.push(k * step);
    }
    k += 1.0;
  }
  levels
}

/// Resolve the Contours option to a sorted list of levels inside the range.
fn resolve_contour_levels(
  spec: &ContourSpec,
  v_min: f64,
  v_max: f64,
) -> Vec<f64> {
  match spec {
    ContourSpec::Automatic => nice_contour_levels(v_min, v_max, 10),
    ContourSpec::Count(n) => {
      let step = (v_max - v_min) / (*n as f64 + 1.0);
      (1..=*n).map(|k| v_min + k as f64 * step).collect()
    }
    ContourSpec::Levels(levels) => {
      let mut ls: Vec<f64> = levels
        .iter()
        .copied()
        .filter(|v| *v > v_min && *v < v_max)
        .collect();
      ls.sort_by(|a, b| a.partial_cmp(b).unwrap());
      ls.dedup();
      ls
    }
  }
}

/// A polygon vertex carrying the interpolated field value.
type PtV = ((f64, f64), f64);

/// Clip a polygon against v >= level (keep_above) or v <= level, with
/// positions interpolated linearly along the edges (Sutherland–Hodgman).
fn clip_polygon(poly: &[PtV], level: f64, keep_above: bool) -> Vec<PtV> {
  let inside = |v: f64| if keep_above { v >= level } else { v <= level };
  let mut out = Vec::with_capacity(poly.len() + 2);
  for k in 0..poly.len() {
    let (pa, va) = poly[k];
    let (pb, vb) = poly[(k + 1) % poly.len()];
    if inside(va) {
      out.push((pa, va));
    }
    if inside(va) != inside(vb) {
      let t = (level - va) / (vb - va);
      let p = (pa.0 + (pb.0 - pa.0) * t, pa.1 + (pb.1 - pa.1) * t);
      out.push((p, level));
    }
  }
  out
}

/// Fill the regions between contour levels with flat band colors (isobands).
/// Cells fully inside one band are merged into horizontal run rectangles;
/// cells straddling a level are clipped into per-band polygons whose edges
/// follow the interpolated contour, so band boundaries are smooth.
/// grid is grid[col][row] with row 0 at the bottom.
#[allow(clippy::too_many_arguments)]
/// One shaded region of a contour plot, in *cell-index* space: `x` counts
/// sample columns and `y` sample rows, counting up. Both the SVG renderer
/// and the symbolic `structure` map these through their own affine
/// transform, so the two always agree on where a band lies.
struct BandRegion {
  /// Which band, as an index into the bounds (`v_min`, levels…, `v_max`).
  band: usize,
  /// Corners in order. A run of whole cells is its four corners.
  points: Vec<(f64, f64)>,
  /// Whether those corners are an axis-aligned rectangle (a run of cells
  /// no level crosses), which the SVG renderer draws as one `<rect>`.
  is_rect: bool,
}

/// The shaded bands of a contour plot: runs of cells that lie wholly in
/// one band, and the cells a level crosses clipped to each band it meets.
fn contour_band_regions(grid: &[Vec<f64>], levels: &[f64]) -> Vec<BandRegion> {
  let cols = grid.len();
  if cols < 2 {
    return Vec::new();
  }
  let rows = grid[0].len();
  if rows < 2 {
    return Vec::new();
  }
  let n_ci = cols - 1;
  let n_cj = rows - 1;

  // Index of the band containing value v.
  let band_of = |v: f64| -> usize {
    match levels.iter().position(|&l| v < l) {
      Some(b) => b,
      None => levels.len(),
    }
  };

  let mut regions: Vec<BandRegion> = Vec::new();
  for j in 0..n_cj {
    // Merge consecutive same-band cells in this row into one rectangle.
    let mut run: Option<(usize, usize, usize)> = None; // (band, i_start, i_end)
    let flush = |regions: &mut Vec<BandRegion>,
                 run: &mut Option<(usize, usize, usize)>| {
      if let Some((b, s, e)) = run.take() {
        let (x0, x1) = (s as f64, e as f64);
        let (y0, y1) = (j as f64, j as f64 + 1.0);
        regions.push(BandRegion {
          band: b,
          points: vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
          is_rect: true,
        });
      }
    };
    for i in 0..n_ci {
      let v00 = grid[i][j];
      let v10 = grid[i + 1][j];
      let v01 = grid[i][j + 1];
      let v11 = grid[i + 1][j + 1];
      if !v00.is_finite()
        || !v10.is_finite()
        || !v01.is_finite()
        || !v11.is_finite()
      {
        flush(&mut regions, &mut run);
        continue;
      }
      let cmin = v00.min(v10).min(v01).min(v11);
      let cmax = v00.max(v10).max(v01).max(v11);
      let b_lo = band_of(cmin);
      let b_hi = band_of(cmax);
      if b_lo == b_hi {
        run = match run {
          Some((b, s, e)) if b == b_lo && e == i => Some((b, s, i + 1)),
          other => {
            if other.is_some() {
              let mut prev = other;
              flush(&mut regions, &mut prev);
            }
            Some((b_lo, i, i + 1))
          }
        };
        continue;
      }
      flush(&mut regions, &mut run);
      let base: [PtV; 4] = [
        ((i as f64, j as f64), v00),
        ((i as f64 + 1.0, j as f64), v10),
        ((i as f64 + 1.0, j as f64 + 1.0), v11),
        ((i as f64, j as f64 + 1.0), v01),
      ];
      for b in b_lo..=b_hi {
        let mut poly: Vec<PtV> = base.to_vec();
        if b > 0 {
          poly = clip_polygon(&poly, levels[b - 1], true);
        }
        if b < levels.len() {
          poly = clip_polygon(&poly, levels[b], false);
        }
        if poly.len() < 3 {
          continue;
        }
        regions.push(BandRegion {
          band: b,
          points: poly.iter().map(|&(p, _)| p).collect(),
          is_rect: false,
        });
      }
    }
    flush(&mut regions, &mut run);
  }
  regions
}

/// The colour of band `b`, from the value at its middle.
fn band_color(
  band: usize,
  levels: &[f64],
  v_min: f64,
  v_max: f64,
  color_function: Option<&str>,
  color_function_expr: Option<(&Expr, bool)>,
) -> (u8, u8, u8) {
  let mut bounds = Vec::with_capacity(levels.len() + 2);
  bounds.push(v_min);
  bounds.extend_from_slice(levels);
  bounds.push(v_max);
  let mid = f64::midpoint(bounds[band], bounds[band + 1]);
  scaled_color(
    scale_value(mid, v_min, v_max),
    mid,
    color_function,
    color_function_expr,
  )
}

#[allow(clippy::too_many_arguments)]
fn render_contour_bands(
  svg: &mut String,
  grid: &[Vec<f64>],
  levels: &[f64],
  v_min: f64,
  v_max: f64,
  plot_x0: f64,
  plot_y0: f64,
  cell_w: f64,
  cell_h: f64,
  color_function: Option<&str>,
  color_function_expr: Option<(&Expr, bool)>,
) {
  let Some(rows) = grid.first().map(std::vec::Vec::len) else {
    return;
  };
  if rows < 2 {
    return;
  }
  let n_cj = (rows - 1) as f64;
  // A stroke in the fill color prevents antialiasing seams between
  // adjacent fills (1px at display size; band boundaries are covered by
  // the contour lines drawn on top).
  let seam = RESOLUTION_SCALE as f64;

  for region in contour_band_regions(grid, levels) {
    let (r, g, b) = band_color(
      region.band,
      levels,
      v_min,
      v_max,
      color_function,
      color_function_expr,
    );
    let fill = format!("rgb({r},{g},{b})");
    let device = |(fx, fy): (f64, f64)| {
      (plot_x0 + fx * cell_w, plot_y0 + (n_cj - fy) * cell_h)
    };
    if region.is_rect {
      let (sx, y1) = device(region.points[0]);
      let (ex, y0) = device(region.points[2]);
      svg.push_str(&format!(
        "<rect x=\"{sx:.1}\" y=\"{y0:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"{fill}\" stroke=\"{fill}\" stroke-width=\"{seam:.1}\"/>\n",
        ex - sx,
        y1 - y0
      ));
      continue;
    }
    let mut d = String::new();
    for (k, &pt) in region.points.iter().enumerate() {
      let (x, y) = device(pt);
      d.push_str(&format!("{}{x:.1} {y:.1}", if k == 0 { "M" } else { "L" }));
    }
    d.push('Z');
    svg.push_str(&format!(
      "<path d=\"{d}\" fill=\"{fill}\" stroke=\"{fill}\" stroke-width=\"{seam:.1}\" stroke-linejoin=\"round\"/>\n"
    ));
  }
}

/// The shaded bands as symbolic primitives in data coordinates, so a
/// `Show` that merges the plot with other graphics still draws the
/// shading rather than the bare contour lines.
fn contour_band_primitives(
  grid: &[Vec<f64>],
  levels: &[f64],
  v_min: f64,
  v_max: f64,
  x_min: f64,
  x_max: f64,
  y_min: f64,
  y_max: f64,
  color_function: Option<&str>,
  color_function_expr: Option<(&Expr, bool)>,
) -> Vec<Expr> {
  let cols = grid.len();
  let rows = grid.first().map_or(0, std::vec::Vec::len);
  if cols < 2 || rows < 2 {
    return Vec::new();
  }
  let step_x = (x_max - x_min) / (cols - 1) as f64;
  let step_y = (y_max - y_min) / (rows - 1) as f64;
  let mut items: Vec<Expr> = Vec::new();
  let mut last_color: Option<(u8, u8, u8)> = None;
  for region in contour_band_regions(grid, levels) {
    let color = band_color(
      region.band,
      levels,
      v_min,
      v_max,
      color_function,
      color_function_expr,
    );
    if last_color != Some(color) {
      let rgb = call(
        "RGBColor",
        vec![
          Expr::Real(color.0 as f64 / 255.0),
          Expr::Real(color.1 as f64 / 255.0),
          Expr::Real(color.2 as f64 / 255.0),
        ],
      );
      // Edged in its own colour, so neighbouring bands meet without an
      // antialiasing seam — the same trick the SVG renderer plays with a
      // stroke in the fill colour.
      items.push(call1("EdgeForm", rgb.clone()));
      items.push(rgb);
      last_color = Some(color);
    }
    let points: Vec<Expr> = region
      .points
      .iter()
      .map(|&(fx, fy)| {
        Expr::List(
          vec![
            Expr::Real(x_min + fx * step_x),
            Expr::Real(y_min + fy * step_y),
          ]
          .into(),
        )
      })
      .collect();
    items.push(call1("Polygon", Expr::List(points.into())));
  }
  items
}

/// Compute marching-squares contour segments for one level.
/// grid is grid[col][row] with row 0 at the bottom.
fn marching_squares_segments(
  grid: &[Vec<f64>],
  level: f64,
  plot_x0: f64,
  plot_y0: f64,
  cell_w: f64,
  cell_h: f64,
) -> Vec<((f64, f64), (f64, f64))> {
  let cols = grid.len();
  let rows = if cols > 0 { grid[0].len() } else { 0 };
  let mut segments = Vec::new();
  if cols < 2 || rows < 2 {
    return segments;
  }
  let n_cj = rows - 1;
  for i in 0..cols - 1 {
    for j in 0..n_cj {
      let v00 = grid[i][j];
      let v10 = grid[i + 1][j];
      let v01 = grid[i][j + 1];
      let v11 = grid[i + 1][j + 1];
      if !v00.is_finite()
        || !v10.is_finite()
        || !v01.is_finite()
        || !v11.is_finite()
      {
        continue;
      }

      let case = (v00 >= level) as u8
        | (((v10 >= level) as u8) << 1)
        | (((v01 >= level) as u8) << 2)
        | (((v11 >= level) as u8) << 3);

      if case == 0 || case == 15 {
        continue;
      }

      // Clamped: interpolation only runs on edges with a sign change, but
      // near-equal endpoint values could still push the result outside the
      // edge by floating-point noise.
      let lerp = |va: f64, vb: f64| -> f64 {
        if (vb - va).abs() < f64::EPSILON {
          0.5
        } else {
          ((level - va) / (vb - va)).clamp(0.0, 1.0)
        }
      };

      let sx = |fx: f64| -> f64 { plot_x0 + (i as f64 + fx) * cell_w };
      let sy =
        |fy: f64| -> f64 { plot_y0 + (n_cj as f64 - (j as f64 + fy)) * cell_h };

      let bottom = (sx(lerp(v00, v10)), sy(0.0));
      let top = (sx(lerp(v01, v11)), sy(1.0));
      let left = (sx(0.0), sy(lerp(v00, v01)));
      let right = (sx(1.0), sy(lerp(v10, v11)));

      // Bits: 1 = v00 (bottom-left), 2 = v10 (bottom-right),
      // 4 = v01 (top-left), 8 = v11 (top-right).
      match case {
        1 | 14 => segments.push((bottom, left)),
        2 | 13 => segments.push((bottom, right)),
        3 | 12 => segments.push((left, right)),
        4 | 11 => segments.push((left, top)),
        5 | 10 => segments.push((bottom, top)),
        6 => {
          // Saddle: v10 and v01 above the level
          segments.push((bottom, right));
          segments.push((left, top));
        }
        9 => {
          // Saddle: v00 and v11 above the level
          segments.push((bottom, left));
          segments.push((right, top));
        }
        7 | 8 => segments.push((right, top)),
        _ => {}
      }
    }
  }
  segments
}

/// Chain contour segments that share endpoints into polylines so lines
/// render with clean joins and the SVG stays compact.
fn chain_segments(
  segments: &[((f64, f64), (f64, f64))],
) -> Vec<Vec<(f64, f64)>> {
  chain_segments_scaled(segments, 16.0)
}

/// Like `chain_segments`, but with an explicit endpoint-matching resolution:
/// two endpoints snap together when equal at `1 / key_scale` granularity.
/// Screen-pixel callers use 16.0; data-coordinate callers must scale by the
/// plot range so the tolerance stays proportionally as fine.
fn chain_segments_scaled(
  segments: &[((f64, f64), (f64, f64))],
  key_scale: f64,
) -> Vec<Vec<(f64, f64)>> {
  use std::collections::HashMap;
  let key = |p: (f64, f64)| -> (i64, i64) {
    (
      (p.0 * key_scale).round() as i64,
      (p.1 * key_scale).round() as i64,
    )
  };
  // Zero-length segments occur when the contour passes exactly through a
  // grid node; neighboring cells provide the connecting geometry, so they
  // would only render as spurious dots.
  let segments: Vec<((f64, f64), (f64, f64))> = segments
    .iter()
    .copied()
    .filter(|(a, b)| key(*a) != key(*b))
    .collect();
  // Endpoint -> indices of segments touching it
  let mut by_point: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
  for (idx, (a, b)) in segments.iter().enumerate() {
    by_point.entry(key(*a)).or_default().push(idx);
    by_point.entry(key(*b)).or_default().push(idx);
  }
  let mut used = vec![false; segments.len()];
  let mut chains = Vec::new();
  for start in 0..segments.len() {
    if used[start] {
      continue;
    }
    used[start] = true;
    let (a, b) = segments[start];
    let mut chain = vec![a, b];
    // Extend at both ends
    for end in 0..2 {
      loop {
        let tip = if end == 0 {
          *chain.last().unwrap()
        } else {
          chain[0]
        };
        let Some(cands) = by_point.get(&key(tip)) else {
          break;
        };
        let Some(&next) = cands.iter().find(|&&idx| !used[idx]) else {
          break;
        };
        used[next] = true;
        let (na, nb) = segments[next];
        let far = if key(na) == key(tip) { nb } else { na };
        if end == 0 {
          chain.push(far);
        } else {
          chain.insert(0, far);
        }
      }
    }
    chains.push(chain);
  }
  chains
}

/// Draw contour lines for the given levels as thin dark polylines.
#[allow(clippy::too_many_arguments)]
/// One mesh function's samples over the plot grid, with the levels its
/// mesh lines are drawn at.
struct MeshGrid {
  grid: Vec<Vec<f64>>,
  levels: Vec<f64>,
}

/// Sample a mesh function over the plot's grid and pick `count` evenly
/// spaced levels across the values it takes — `MeshFunctions -> {10 #1 &}`
/// with `Mesh -> 11` gives eleven lines of constant x. `region_mask`, when
/// given, is the main plot's already `RegionFunction`-masked grid (NaN
/// outside the region): a cell excluded there is excluded here too, so mesh
/// lines don't spill past the region boundary the contours/bands stop at.
#[allow(clippy::too_many_arguments)]
fn sample_mesh_grid(
  f: &Expr,
  xvar: &str,
  yvar: &str,
  x_min: f64,
  x_max: f64,
  y_min: f64,
  y_max: f64,
  count: usize,
  region_mask: Option<&[Vec<f64>]>,
) -> Option<MeshGrid> {
  let n = FIELD_GRID + 1;
  let mut grid = vec![vec![f64::NAN; n]; n];
  let (mut v_min, mut v_max) = (f64::INFINITY, f64::NEG_INFINITY);
  for i in 0..n {
    let x = x_min + i as f64 / FIELD_GRID as f64 * (x_max - x_min);
    for j in 0..n {
      let y = y_min + j as f64 / FIELD_GRID as f64 * (y_max - y_min);
      let masked = region_mask
        .and_then(|m| m.get(i).and_then(|row| row.get(j)))
        .is_some_and(|v| !v.is_finite());
      if masked {
        continue;
      }
      // A mesh function is usually a pure function of the two
      // coordinates (`10 #1 &`); a plain expression in x and y works too.
      let v = mesh_function_value(f, xvar, yvar, x, y)?;
      if v.is_finite() {
        grid[i][j] = v;
        v_min = v_min.min(v);
        v_max = v_max.max(v);
      }
    }
  }
  if !v_min.is_finite() || !v_max.is_finite() || v_max <= v_min {
    return None;
  }
  // `Mesh -> n` draws n lines *between* the edges of the value range.
  let step = (v_max - v_min) / (count as f64 + 1.0);
  let levels = (1..=count).map(|k| v_min + k as f64 * step).collect();
  Some(MeshGrid { grid, levels })
}

/// The mesh functions a `Mesh`/`MeshFunctions` pair asks for, each sampled
/// over the plot's grid with the levels its lines are drawn at. The default
/// mesh functions are the coordinates themselves.
fn sample_mesh_grids(
  opts: &DensityContourOptions,
  xvar: &str,
  yvar: &str,
  (x_min, x_max): (f64, f64),
  (y_min, y_max): (f64, f64),
  region_mask: &[Vec<f64>],
) -> Vec<MeshGrid> {
  let Some(count) = opts.mesh.filter(|&c| c > 0) else {
    return Vec::new();
  };
  let functions: Vec<Expr> = if opts.mesh_functions.is_empty() {
    vec![
      Expr::Identifier(xvar.to_string()),
      Expr::Identifier(yvar.to_string()),
    ]
  } else {
    opts.mesh_functions.clone()
  };
  functions
    .iter()
    .filter_map(|f| {
      sample_mesh_grid(
        f,
        xvar,
        yvar,
        x_min,
        x_max,
        y_min,
        y_max,
        count,
        Some(region_mask),
      )
    })
    .collect()
}

/// The mesh lines as data-space `Line` primitives under a light grey
/// thin-line style, in their own list so the grey does not leak onto
/// whatever is drawn after them. Empty when there are no mesh lines.
fn mesh_structure_items(
  mesh_grids: &[MeshGrid],
  (x_min, x_max): (f64, f64),
  (y_min, y_max): (f64, f64),
) -> Option<Expr> {
  let span = (x_max - x_min).abs().max((y_max - y_min).abs());
  let key_scale = if span > 0.0 { 4096.0 / span } else { 16.0 };
  let step_x = (x_max - x_min) / FIELD_GRID as f64;
  let step_y = (y_max - y_min) / FIELD_GRID as f64;
  let mut mesh_items: Vec<Expr> = vec![
    call1("GrayLevel", Expr::Real(0.69)),
    call1("AbsoluteThickness", Expr::Real(0.4)),
  ];
  for mesh_grid in mesh_grids {
    for &level in &mesh_grid.levels {
      let segments = marching_squares_segments(
        &mesh_grid.grid,
        level,
        x_min,
        y_max,
        step_x,
        -step_y,
      );
      for chain in chain_segments_scaled(&segments, key_scale) {
        let points: Vec<Expr> = chain
          .iter()
          .map(|(x, y)| Expr::List(vec![Expr::Real(*x), Expr::Real(*y)].into()))
          .collect();
        mesh_items.push(call1("Line", Expr::List(points.into())));
      }
    }
  }
  (mesh_items.len() > 2).then(|| Expr::List(mesh_items.into()))
}

/// Evaluate a mesh function at one grid point, whether it is written as a
/// pure function of the coordinates or as an expression in them.
fn mesh_function_value(
  f: &Expr,
  xvar: &str,
  yvar: &str,
  x: f64,
  y: f64,
) -> Option<f64> {
  match f {
    Expr::Function { .. } => {
      let applied = Expr::CurriedCall {
        func: Box::new(f.clone()),
        args: vec![Expr::Real(x), Expr::Real(y)],
      };
      try_eval_to_f64(&evaluate_expr_to_expr(&applied).ok()?)
    }
    other => evaluate_at_xy(other, xvar, yvar, x, y),
  }
}

#[allow(clippy::too_many_arguments)]
fn render_contour_lines_styled(
  svg: &mut String,
  grid: &[Vec<f64>],
  levels: &[f64],
  plot_x0: f64,
  plot_y0: f64,
  cell_w: f64,
  cell_h: f64,
  render_width: u32,
  stroke: &str,
  weight: f64,
  dash: &str,
) {
  let stroke_w = render_width as f64 / 1000.0 * weight;
  for &level in levels {
    let segments =
      marching_squares_segments(grid, level, plot_x0, plot_y0, cell_w, cell_h);
    for chain in chain_segments(&segments) {
      let points: Vec<String> = chain
        .iter()
        .map(|(x, y)| format!("{x:.1},{y:.1}"))
        .collect();
      svg.push_str(&format!(
        "<polyline points=\"{}\" fill=\"none\" stroke=\"{stroke}\" stroke-width=\"{stroke_w:.1}\"{dash} stroke-linecap=\"round\" stroke-linejoin=\"round\"/>\n",
        points.join(" ")
      ));
    }
  }
}

/// DensityPlot[f, {x, xmin, xmax}, {y, ymin, ymax}]
/// The symbolic `Graphics[items, opts]` a `ContourPlot`/`DensityPlot`
/// carries alongside its rendering, so `Show` can merge its primitives with
/// other graphics. It keeps the call's options too, so the merged picture is
/// still drawn on the axes and range the plot asked for, and spells out the
/// plot's own defaults where `Graphics` differs: framed, and square.
/// Wrap a contour plot's primitives into the `GraphicsComplex[pts, …]`
/// Wolfram returns: every `Line` of coordinate pairs (at any list depth)
/// becomes a `Line` of 1-based indices into one shared point table, so
/// `Cases[plot, _GraphicsComplex, ∞]` and `pts[[First[line]]]` work as in
/// Wolfram and `Normal` recovers the coordinates.
fn contour_graphics_complex(items: &[Expr]) -> Expr {
  fn coords(e: &Expr) -> Option<(f64, f64)> {
    match e {
      Expr::List(p) if p.len() == 2 => match (&p[0], &p[1]) {
        (Expr::Real(x), Expr::Real(y)) => Some((*x, *y)),
        _ => None,
      },
      _ => None,
    }
  }
  fn index(
    e: &Expr,
    pts: &mut Vec<Expr>,
    seen: &mut std::collections::HashMap<(u64, u64), i128>,
  ) -> Expr {
    match e {
      Expr::List(items) => {
        Expr::List(items.iter().map(|i| index(i, pts, seen)).collect())
      }
      Expr::FunctionCall { name, args }
        if name == "Line"
          && args.len() == 1
          && let Expr::List(chain) = &args[0]
          && !chain.is_empty()
          && chain.iter().all(|p| coords(p).is_some()) =>
      {
        let ids: Vec<Expr> = chain
          .iter()
          .map(|p| {
            let (x, y) = coords(p).unwrap();
            let id =
              *seen.entry((x.to_bits(), y.to_bits())).or_insert_with(|| {
                pts.push(p.clone());
                pts.len() as i128
              });
            Expr::Integer(id)
          })
          .collect();
        call1("Line", Expr::List(ids.into()))
      }
      _ => e.clone(),
    }
  }
  let mut pts = Vec::new();
  let mut seen = std::collections::HashMap::new();
  let data: Vec<Expr> = items
    .iter()
    .map(|i| index(i, &mut pts, &mut seen))
    .collect();
  call(
    "GraphicsComplex",
    vec![Expr::List(pts.into()), Expr::List(data.into())],
  )
}

fn field_plot_structure(
  items: Vec<Expr>,
  args: &[Expr],
  (x_min, x_max): (f64, f64),
  (y_min, y_max): (f64, f64),
  pin_range: bool,
) -> Expr {
  fn name_of(o: &Expr) -> Option<&str> {
    crate::functions::graphics::option_name_value(o).map(|(n, _)| n)
  }
  let mut structure_args = vec![Expr::List(items.into())];
  let mut explicit = crate::functions::plot::explicit_options(args);
  // The plot's own `PlotRange` clips the *function values* — its x/y axes
  // are already fixed by the `{x,xmin,xmax}`/`{y,ymin,ymax}` iterators and
  // stay put regardless of it. But `Graphics`/`Show` read a `PlotRange` rule
  // as an axis-range spec, so forwarding the call's value verbatim would let
  // it crop the axes of the merged picture. `pin_range` replaces it with
  // the iterator ranges: `Show` then keeps the plotted domain as its axes,
  // and labels placed beyond it (`PlotRangeClipping -> False`) lie outside
  // the frame instead of stretching it.
  explicit.retain(|o| name_of(o) != Some("PlotRange"));
  let pair =
    |lo: f64, hi: f64| Expr::List(vec![Expr::Real(lo), Expr::Real(hi)].into());
  if pin_range {
    explicit.push(rule_expr(
      id_expr("PlotRange"),
      Expr::List(vec![pair(x_min, x_max), pair(y_min, y_max)].into()),
    ));
  }
  if !explicit.iter().any(|o| name_of(o) == Some("Frame")) {
    structure_args.push(rule_expr(id_expr("Frame"), bool_expr(true)));
  }
  if !explicit.iter().any(|o| name_of(o) == Some("AspectRatio")) {
    structure_args.push(rule_expr(id_expr("AspectRatio"), Expr::Integer(1)));
  }
  structure_args.extend(explicit);
  call("Graphics", structure_args)
}

pub fn density_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let (xvar, x_min, x_max) = parse_iterator(&args[1], "DensityPlot")?;
  let (yvar, y_min, y_max) = parse_iterator(&args[2], "DensityPlot")?;
  // `p = a x^2 + b x; RegionPlot[y > p, …]` — the head holds its arguments,
  // so a body that names its function indirectly has to be evaluated once
  // before the plot variables can be substituted into it.
  let resolved = crate::functions::plot::resolve_indirect_plot_body(
    &args[0],
    &[xvar.as_str(), yvar.as_str()],
  );
  let body = resolved.as_ref().unwrap_or(&args[0]);
  let opts = parse_density_contour_options(args, 3);

  // Sample grid
  let mut grid = vec![vec![f64::NAN; FIELD_GRID]; FIELD_GRID];
  let mut samples = Vec::with_capacity(FIELD_GRID * FIELD_GRID);

  for i in 0..FIELD_GRID {
    let x = x_min + (i as f64 + 0.5) / FIELD_GRID as f64 * (x_max - x_min);
    for j in 0..FIELD_GRID {
      let y = y_min + (j as f64 + 0.5) / FIELD_GRID as f64 * (y_max - y_min);
      if let Some(v) = evaluate_at_xy(body, &xvar, &yvar, x, y)
        && v.is_finite()
        && opts
          .region_function
          .as_ref()
          .is_none_or(|r| region_allows(r, x, y, v))
      {
        grid[i][j] = v;
        samples.push(v);
      }
    }
  }
  // An explicit `PlotRange -> {zmin, zmax}` fixes the values the colours
  // span; cells beyond it follow `ClippingStyle` (blank by default).
  // Otherwise the colours span the sampled values.
  let (v_min, v_max) = opts
    .value_range
    .unwrap_or_else(|| robust_value_range(&samples));
  let clipping = if opts.value_range.is_some() {
    opts.clipping
  } else {
    Clipping::Saturate
  };
  let colors = (v_min.is_finite() && v_max.is_finite()).then(|| {
    cell_colors(
      &grid,
      (v_min, v_max),
      opts.color_function.as_deref(),
      opts.cf_expr(),
      clipping,
    )
  });

  // Use plotters for axes, then overlay the density image
  let mut area = field_plot_axes((x_min, x_max), (y_min, y_max), &opts)?;

  // Taken out of `area` so the geometry stays available for the overlays
  // appended once the plot's own content is drawn.
  let mut svg = std::mem::take(&mut area.svg);
  // Remove closing </svg> to append custom elements
  if let Some(pos) = svg.rfind("</svg>") {
    svg.truncate(pos);
  }

  if let Some(colors) = &colors {
    embed_color_image(
      &mut svg,
      colors,
      area.plot_x0,
      area.plot_y0,
      area.plot_w,
      area.plot_h,
    );
  }
  // Mesh lines lie over the density, the level curves of each mesh
  // function (the coordinates themselves by default).
  let mesh_grids = sample_mesh_grids(
    &opts,
    &xvar,
    &yvar,
    (x_min, x_max),
    (y_min, y_max),
    &grid,
  );
  for mesh_grid in &mesh_grids {
    render_contour_lines_styled(
      &mut svg,
      &mesh_grid.grid,
      &mesh_grid.levels,
      area.plot_x0,
      area.plot_y0,
      area.plot_w / FIELD_GRID as f64,
      area.plot_h / FIELD_GRID as f64,
      area.render_width,
      "#b0b0b0",
      0.4,
      "",
    );
  }
  if opts.frame {
    push_frame(
      &mut svg,
      area.plot_x0,
      area.plot_y0,
      area.plot_w,
      area.plot_h,
    );
  }
  push_field_plot_overlays(&mut svg, &area, &opts);

  svg.push_str("</svg>");

  // The coloured cells as a `Raster` spanning the plotted domain (rows from
  // the bottom up), so `Show` can lay other graphics over the density
  // instead of setting the two pictures side by side. Cells without a
  // colour stay transparent.
  // A body with no numeric values (an equation, say) leaves the plot blank:
  // no raster, but the empty frame still merges under `Show`.
  let raster = colors.as_ref().map(|colors| {
    let channel = |c: u8| Expr::Real(c as f64 / 255.0);
    let rows: Vec<Expr> = (0..FIELD_GRID)
      .map(|j| {
        let row: Vec<Expr> = colors
          .iter()
          .map(|col| {
            let ((r, g, b), alpha) = match col[j] {
              Some(rgb) => (rgb, 1.0),
              None => ((0, 0, 0), 0.0),
            };
            Expr::List(
              vec![channel(r), channel(g), channel(b), Expr::Real(alpha)]
                .into(),
            )
          })
          .collect();
        Expr::List(row.into())
      })
      .collect();
    let corner =
      |x: f64, y: f64| Expr::List(vec![Expr::Real(x), Expr::Real(y)].into());
    call(
      "Raster",
      vec![
        Expr::List(rows.into()),
        Expr::List(vec![corner(x_min, y_min), corner(x_max, y_max)].into()),
      ],
    )
  });
  let mut structure_items: Vec<Expr> = raster.into_iter().collect();
  structure_items.extend(mesh_structure_items(
    &mesh_grids,
    (x_min, x_max),
    (y_min, y_max),
  ));
  let structure = field_plot_structure(
    structure_items,
    args,
    (x_min, x_max),
    (y_min, y_max),
    true,
  );
  let graphic = crate::plot_result_with_structure(svg, structure);
  if !opts.bar_legend {
    return Ok(graphic);
  }
  // `PlotLegends -> Automatic` gives `Legended[plot, Placed[BarLegend[…],
  // After]]`, as in Wolfram: a colour bar over the plotted value range,
  // which `Show` keeps when it merges the plot with other graphics.
  let colors = match (&opts.color_function, &opts.color_function_expr) {
    (_, Some(f)) => f.clone(),
    (Some(name), None) => Expr::String(name.clone()),
    (None, None) => str_expr("WL14DefaultDensityGradient"),
  };
  let bar = call1(
    "BarLegend",
    Expr::List(
      vec![
        colors,
        Expr::List(vec![Expr::Real(v_min), Expr::Real(v_max)].into()),
      ]
      .into(),
    ),
  );
  Ok(call(
    "Legended",
    vec![graphic, call("Placed", vec![bar, id_expr("After")])],
  ))
}

/// ContourPlot[f, {x, xmin, xmax}, {y, ymin, ymax}]
/// Uses marching squares to draw contour lines.
/// Rewrite an equation `lhs == rhs` into `lhs - rhs`, whose zero contour
/// is the equation's implicit curve. Returns `None` for non-equations.
pub(crate) fn equation_zero_body(e: &Expr) -> Option<Expr> {
  let (lhs, rhs) = match e {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && operators.len() == 1
      && operators[0] == ComparisonOp::Equal =>
    {
      (&operands[0], &operands[1])
    }
    Expr::FunctionCall { name, args } if name == "Equal" && args.len() == 2 => {
      (&args[0], &args[1])
    }
    _ => return None,
  };
  Some(minus2(lhs.clone(), rhs.clone()))
}

/// Bounds of a second plot iterator that depend on the first variable, as in
/// `ContourPlot[…, {x, 0, 20}, {y, -10, .99 x}]`.
struct DependentBounds {
  xvar: String,
  lo: Expr,
  hi: Expr,
}

impl DependentBounds {
  /// The `(min, max)` of the iterator at a given value of the first variable
  /// (NaN-free: an unevaluable bound leaves that side unbounded).
  fn at(&self, x: f64) -> (f64, f64) {
    let eval = |e: &Expr, default: f64| {
      let sub =
        crate::functions::plot::substitute_var(e, &self.xvar, &Expr::Real(x));
      evaluate_expr_to_expr(&sub)
        .ok()
        .and_then(|r| try_eval_to_f64(&r))
        .filter(|v| v.is_finite())
        .unwrap_or(default)
    };
    (
      eval(&self.lo, f64::NEG_INFINITY),
      eval(&self.hi, f64::INFINITY),
    )
  }
}

/// Like `parse_iterator`, but for an iterator whose bounds may mention the
/// preceding variable. The returned `(min, max)` is the bounding box of the
/// region over the first variable's range; the bounds themselves come back
/// so the sampler can mask what falls outside.
fn parse_dependent_iterator(
  spec: &Expr,
  label: &str,
  xvar: &str,
  x_min: f64,
  x_max: f64,
) -> Result<(String, f64, f64, Option<DependentBounds>), InterpreterError> {
  if let Ok((var, lo, hi)) = parse_iterator(spec, label) {
    return Ok((var, lo, hi, None));
  }
  let Expr::List(items) = spec else {
    return parse_iterator(spec, label).map(|(v, a, b)| (v, a, b, None));
  };
  if items.len() != 3 {
    return parse_iterator(spec, label).map(|(v, a, b)| (v, a, b, None));
  }
  let Expr::Identifier(var) = &items[0] else {
    return parse_iterator(spec, label).map(|(v, a, b)| (v, a, b, None));
  };
  let bounds = DependentBounds {
    xvar: xvar.to_string(),
    lo: items[1].clone(),
    hi: items[2].clone(),
  };
  let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
  for i in 0..=FIELD_GRID {
    let x = x_min + i as f64 / FIELD_GRID as f64 * (x_max - x_min);
    let (a, b) = bounds.at(x);
    if a.is_finite() && b.is_finite() && a <= b {
      lo = lo.min(a);
      hi = hi.max(b);
    }
  }
  if !lo.is_finite() || !hi.is_finite() {
    return parse_iterator(spec, label).map(|(v, a, b)| (v, a, b, None));
  }
  Ok((var.clone(), lo, hi, Some(bounds)))
}

/// The arguments of a held field plot (`ContourPlot`, `DensityPlot`,
/// `RegionPlot`) with each option's value evaluated, the way the plot
/// itself evaluates its options. The plot variables (`{x, …}` iterator
/// specs among `args[1..first_option]`) are localized meanwhile, so an
/// option may still mention them symbolically (`Exclusions -> {x == 0}`).
/// Without this, a style such as `ContourStyle -> styles[[Mod[n, 2, 1]]]`
/// in a `Table` over `n` was copied into the graphics unevaluated, and
/// failed once `n` was no longer bound. Delayed `:>` options stay held.
pub fn with_evaluated_options(
  args: &[Expr],
  first_option: usize,
) -> Result<Vec<Expr>, InterpreterError> {
  use crate::evaluator::symbol_values::{
    restore_symbol_values, take_symbol_values,
  };
  let first_option = first_option.min(args.len());
  let vars: Vec<String> = args[1..first_option]
    .iter()
    .filter_map(|spec| match spec {
      Expr::List(items) => match items.first() {
        Some(Expr::Identifier(var)) => Some(var.clone()),
        _ => None,
      },
      _ => None,
    })
    .collect();
  let saved: Vec<_> = vars.iter().map(|v| take_symbol_values(v)).collect();
  let mut out = args[..first_option].to_vec();
  let mut result = Ok(());
  for arg in &args[first_option..] {
    match arg {
      Expr::Rule {
        pattern,
        replacement,
      } if result.is_ok() => {
        match crate::evaluator::evaluate_expr_to_expr(replacement) {
          Ok(value) => out.push(Expr::Rule {
            pattern: pattern.clone(),
            replacement: Box::new(value),
          }),
          Err(e) => result = Err(e),
        }
      }
      other => out.push(other.clone()),
    }
  }
  for values in saved.into_iter().rev() {
    restore_symbol_values(values);
  }
  result.map(|()| out)
}

pub fn contour_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // A body that names its function indirectly (`f = x^2 + y^2;
  // ContourPlot[f, …]`) is evaluated once first — see
  // `resolve_indirect_plot_body`.
  let resolved = match (
    parse_iterator(&args[1], "ContourPlot"),
    parse_iterator(&args[2], "ContourPlot"),
  ) {
    (Ok((xvar, ..)), Ok((yvar, ..))) => {
      crate::functions::plot::resolve_indirect_plot_body(
        &args[0],
        &[xvar.as_str(), yvar.as_str()],
      )
    }
    _ => None,
  };
  let body = resolved.as_ref().unwrap_or(&args[0]);
  // `ContourPlot[lhs == rhs, …]` (or a list of equations) plots each
  // equation's implicit curve — the zero contour of `lhs - rhs` — with no
  // band shading, matching Wolfram.
  let eq_bodies: Option<Vec<Expr>> = match body {
    Expr::List(items)
      if !items.is_empty()
        && items.iter().all(|i| equation_zero_body(i).is_some()) =>
    {
      Some(items.iter().filter_map(equation_zero_body).collect())
    }
    _ => equation_zero_body(body).map(|b| vec![b]),
  };
  // `ContourPlot[{f, g}, …, Contours -> {c, …}]` draws the level curves of
  // each function; a curve `f == c` is the zero contour of `f - c`.
  let eq_bodies = eq_bodies.or_else(|| match body {
    Expr::List(items)
      if !items.is_empty()
        && items.iter().all(|i| !matches!(i, Expr::List(_))) =>
    {
      match parse_density_contour_options(args, 3).contours {
        ContourSpec::Levels(levels) if !levels.is_empty() => Some(
          items
            .iter()
            .flat_map(|f| {
              levels
                .iter()
                .map(move |&c| crate::helpers::minus2(f.clone(), Expr::Real(c)))
            })
            .collect(),
        ),
        _ => None,
      }
    }
    _ => None,
  });
  if let Some(bodies) = eq_bodies {
    return contour_plot_equations(&bodies, args);
  }
  let (xvar, x_min, x_max) = parse_iterator(&args[1], "ContourPlot")?;
  let (yvar, y_min, y_max, y_bounds) =
    parse_dependent_iterator(&args[2], "ContourPlot", &xvar, x_min, x_max)?;
  let opts = parse_density_contour_options(args, 3);

  let n = FIELD_GRID + 1;

  // Sample grid
  let mut grid = vec![vec![f64::NAN; n]; n];
  let mut samples = Vec::with_capacity(n * n);

  for i in 0..n {
    let x = x_min + i as f64 / FIELD_GRID as f64 * (x_max - x_min);
    let (row_lo, row_hi) = match &y_bounds {
      Some(b) => b.at(x),
      None => (f64::NEG_INFINITY, f64::INFINITY),
    };
    for j in 0..n {
      let y = y_min + j as f64 / FIELD_GRID as f64 * (y_max - y_min);
      if y < row_lo || y > row_hi {
        continue;
      }
      if let Some(v) = evaluate_at_xy(body, &xvar, &yvar, x, y)
        && v.is_finite()
        && opts
          .region_function
          .as_ref()
          .is_none_or(|r| region_allows(r, x, y, v))
      {
        grid[i][j] = v;
        samples.push(v);
      }
    }
  }
  let (v_min, v_max) = robust_value_range(&samples);

  if !v_min.is_finite() || !v_max.is_finite() {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }

  let levels = resolve_contour_levels(&opts.contours, v_min, v_max);

  // Sample each mesh function over the same grid and pick its levels.
  let mesh_grids = sample_mesh_grids(
    &opts,
    &xvar,
    &yvar,
    (x_min, x_max),
    (y_min, y_max),
    &grid,
  );

  // Data-space contour polylines for the symbolic `structure`, so
  // `ContourPlot[…][[1]]` yields primitives (with the ContourStyle
  // directives) that can be re-embedded in an outer `Graphics[…]` the way
  // Wolfram's GraphicsComplex form allows.
  let mut structure_items: Vec<Expr> = Vec::new();
  // The shaded bands go underneath, in their own list so their colours do
  // not leak onto the mesh and contour lines drawn over them.
  if opts.contour_shading {
    let bands = contour_band_primitives(
      &grid,
      &levels,
      v_min,
      v_max,
      x_min,
      x_max,
      y_min,
      y_max,
      opts.color_function.as_deref(),
      opts.cf_expr(),
    );
    if !bands.is_empty() {
      structure_items.push(Expr::List(bands.into()));
    }
  }
  // Mesh lines next, in their own list so their grey does not leak onto
  // the contours drawn after them.
  structure_items.extend(mesh_structure_items(
    &mesh_grids,
    (x_min, x_max),
    (y_min, y_max),
  ));
  structure_items.extend(contour_style_directives(args, 3));
  {
    let step_x = (x_max - x_min) / FIELD_GRID as f64;
    let step_y = (y_max - y_min) / FIELD_GRID as f64;
    // Endpoint snapping must stay proportional to the range (the pixel
    // callers use 16.0 on a ~400px canvas).
    let span = (x_max - x_min).abs().max((y_max - y_min).abs());
    let key_scale = if span > 0.0 { 4096.0 / span } else { 16.0 };
    for &level in &levels {
      // `plot_y0 = y_max` with a negated cell height turns the screen-space
      // (top-down) y mapping into the plain data-space one.
      let segments =
        marching_squares_segments(&grid, level, x_min, y_max, step_x, -step_y);
      for chain in chain_segments_scaled(&segments, key_scale) {
        let points: Vec<Expr> = chain
          .iter()
          .map(|(x, y)| Expr::List(vec![Expr::Real(*x), Expr::Real(*y)].into()))
          .collect();
        structure_items.push(call1("Line", Expr::List(points.into())));
      }
    }
    // The level labels travel with the plot, so a `Show` that layers
    // other graphics over it keeps them.
    if opts.contour_labels {
      let labels: Vec<Expr> =
        contour_label_points(&grid, &levels, x_min, y_max, step_x, -step_y)
          .into_iter()
          .map(|(level, (x, y))| {
            call(
              "Text",
              vec![
                Expr::String(crate::functions::plot::format_tick(level)),
                Expr::List(vec![Expr::Real(x), Expr::Real(y)].into()),
              ],
            )
          })
          .collect();
      if !labels.is_empty() {
        structure_items.push(Expr::List(labels.into()));
      }
    }
  }
  let structure = field_plot_structure(
    vec![contour_graphics_complex(&structure_items)],
    args,
    (x_min, x_max),
    (y_min, y_max),
    false,
  );

  // Use plotters for axes
  let mut area = field_plot_axes((x_min, x_max), (y_min, y_max), &opts)?;

  // Taken out of `area` so the geometry stays available for the overlays
  // appended once the plot's own content is drawn.
  let mut svg = std::mem::take(&mut area.svg);
  if let Some(pos) = svg.rfind("</svg>") {
    svg.truncate(pos);
  }

  let cell_w = area.plot_w / FIELD_GRID as f64;
  let cell_h = area.plot_h / FIELD_GRID as f64;

  if opts.contour_shading {
    render_contour_bands(
      &mut svg,
      &grid,
      &levels,
      v_min,
      v_max,
      area.plot_x0,
      area.plot_y0,
      cell_w,
      cell_h,
      opts.color_function.as_deref(),
      opts.cf_expr(),
    );
  }
  // Mesh lines sit under the contours: they are the level curves of the
  // mesh functions (the coordinates themselves by default), so the same
  // marching squares draw them.
  for mesh_grid in &mesh_grids {
    render_contour_lines_styled(
      &mut svg,
      &mesh_grid.grid,
      &mesh_grid.levels,
      area.plot_x0,
      area.plot_y0,
      cell_w,
      cell_h,
      area.render_width,
      "#b0b0b0",
      0.4,
      "",
    );
  }
  let (contour_color, contour_weight) = opts.contour_stroke();
  render_contour_lines_styled(
    &mut svg,
    &grid,
    &levels,
    area.plot_x0,
    area.plot_y0,
    cell_w,
    cell_h,
    area.render_width,
    &contour_color,
    contour_weight,
    "",
  );
  if opts.boundary_style.is_some() {
    // The region's edge is the level curve halfway between the cells that
    // have a value and those left blank.
    let defined: Vec<Vec<f64>> = grid
      .iter()
      .map(|col| {
        col
          .iter()
          .map(|v| if v.is_finite() { 1.0 } else { 0.0 })
          .collect()
      })
      .collect();
    let (color, weight) = opts.contour_stroke_for(opts.boundary_style.as_ref());
    let weight = if opts
      .boundary_style
      .as_ref()
      .is_some_and(|s| s.thickness.is_some())
    {
      weight
    } else {
      CONTOUR_LINE_WEIGHT * 2.0
    };
    render_contour_lines_styled(
      &mut svg,
      &defined,
      &[0.5],
      area.plot_x0,
      area.plot_y0,
      cell_w,
      cell_h,
      area.render_width,
      &color,
      weight,
      "",
    );
  }
  if opts.frame {
    push_frame(
      &mut svg,
      area.plot_x0,
      area.plot_y0,
      area.plot_w,
      area.plot_h,
    );
  }
  push_field_plot_overlays(&mut svg, &area, &opts);
  if opts.contour_labels {
    push_contour_labels(&mut svg, &grid, &levels, &area);
  }
  if opts.bar_legend && opts.contour_shading {
    push_bar_legend(&mut svg, &area, &opts, &levels, (v_min, v_max));
  }

  svg.push_str("</svg>");
  Ok(crate::plot_result_with_structure(svg, structure))
}

/// The `ContourStyle -> …` option items of a ContourPlot call (a single
/// directive or a directive list), for embedding into the plot's symbolic
/// `structure`. Empty when the option is absent.
fn contour_style_directives(args: &[Expr], opts_from: usize) -> Vec<Expr> {
  for arg in args.iter().skip(opts_from) {
    if let Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } = arg
      && matches!(pattern.as_ref(), Expr::Identifier(s) if s == "ContourStyle")
    {
      return match replacement.as_ref() {
        Expr::List(items) => items.iter().cloned().collect(),
        other => vec![other.clone()],
      };
    }
  }
  Vec::new()
}

/// The ` stroke-dasharray` attribute of a contour whose style carries
/// dashing (`Dashed`, `Dotted`, `Dashing[…]`); empty for a solid line.
fn contour_dash_attr(
  style: Option<&crate::functions::plot::SeriesStyle>,
  render_width: u32,
  svg_width: u32,
) -> String {
  let Some(dashes) = style.and_then(|s| s.dashing.as_ref()) else {
    return String::new();
  };
  let scale = render_width as f64 / svg_width.max(1) as f64;
  let lens: Vec<String> = dashes
    .iter()
    .map(|d| {
      let len =
        crate::functions::plot::dash_len(*d, render_width as f64, scale);
      format!("{len:.1}")
    })
    .collect();
  format!(" stroke-dasharray=\"{}\"", lens.join(","))
}

/// The equation form of ContourPlot: draw each body's zero contour as an
/// implicit curve. No bands are shaded — Wolfram renders equation contours
/// as bare curves on an unshaded background.
fn contour_plot_equations(
  bodies: &[Expr],
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let (xvar, x_min, x_max) = parse_iterator(&args[1], "ContourPlot")?;
  let (yvar, y_min, y_max) = parse_iterator(&args[2], "ContourPlot")?;
  let opts = parse_density_contour_options(args, 3);

  let n = FIELD_GRID + 1;
  let mut area = field_plot_axes((x_min, x_max), (y_min, y_max), &opts)?;

  // Taken out of `area` so the geometry stays available for the overlays
  // appended once the plot's own content is drawn.
  let mut svg = std::mem::take(&mut area.svg);
  if let Some(pos) = svg.rfind("</svg>") {
    svg.truncate(pos);
  }

  let cell_w = area.plot_w / FIELD_GRID as f64;
  let cell_h = area.plot_h / FIELD_GRID as f64;

  // Alongside the standalone SVG, collect every contour chain in *data*
  // coordinates as a line series, so `Show[{ContourPlot[…], Graphics[…]}]`
  // can merge the curve with other primitives. The chains are traced in a
  // scaled 0‥1000 space — `chain_segments` keys points on a 1/16 grid,
  // which would glue distinct points together in small data ranges.
  let mut series: Vec<crate::syntax::PlotSeriesData> = Vec::new();
  let mut series_per_body: Vec<std::ops::Range<usize>> = Vec::new();

  for (body_index, body) in bodies.iter().enumerate() {
    let style = opts.style_for_contour(body_index);
    let (contour_color, contour_weight) = opts.contour_stroke_for(style);
    let series_color =
      style.and_then(|s| s.color).map_or((0x40, 0x40, 0x40), |c| {
        (
          (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
          (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
          (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
        )
      });
    let series_thickness = style.and_then(|s| s.thickness);
    let dash = contour_dash_attr(style, area.render_width, opts.svg_width);
    let first_series = series.len();
    let mut grid = vec![vec![f64::NAN; n]; n];
    let mut any_finite = false;
    for (i, row) in grid.iter_mut().enumerate() {
      let x = x_min + i as f64 / FIELD_GRID as f64 * (x_max - x_min);
      for (j, cell) in row.iter_mut().enumerate() {
        let y = y_min + j as f64 / FIELD_GRID as f64 * (y_max - y_min);
        if let Some(v) = evaluate_at_xy(body, &xvar, &yvar, x, y)
          && v.is_finite()
          && opts
            .region_function
            .as_ref()
            .is_none_or(|r| region_allows(r, x, y, v))
        {
          *cell = v;
          any_finite = true;
        }
      }
    }
    if !any_finite {
      series_per_body.push(first_series..first_series);
      continue;
    }
    render_contour_lines_styled(
      &mut svg,
      &grid,
      &[0.0],
      area.plot_x0,
      area.plot_y0,
      cell_w,
      cell_h,
      area.render_width,
      &contour_color,
      contour_weight,
      &dash,
    );
    let scaled_cell = 1000.0 / FIELD_GRID as f64;
    let segments =
      marching_squares_segments(&grid, 0.0, 0.0, 0.0, scaled_cell, scaled_cell);
    for chain in chain_segments(&segments) {
      let points: Vec<(f64, f64)> = chain
        .iter()
        .map(|&(u, v)| {
          (
            x_min + u / 1000.0 * (x_max - x_min),
            y_min + v / 1000.0 * (y_max - y_min),
          )
        })
        .collect();
      if points.len() >= 2 {
        // The curve carries its `ContourStyle` into the series too, so a
        // `Show[{ContourPlot[…], …}]` re-render keeps the colour and weight
        // the standalone picture was drawn with.
        series.push(crate::syntax::PlotSeriesData {
          points,
          color: series_color,
          is_scatter: false,
          filling: crate::syntax::SeriesFilling::None,
          fill_color: None,
          fill_opacity: None,
          marker: None,
          thickness: series_thickness,
          point_size: None,
          dashing: None,
          opacity: None,
        });
      }
    }
    series_per_body.push(first_series..series.len());
  }

  if opts.frame {
    push_frame(
      &mut svg,
      area.plot_x0,
      area.plot_y0,
      area.plot_w,
      area.plot_h,
    );
  }
  push_field_plot_overlays(&mut svg, &area, &opts);

  svg.push_str("</svg>");
  // Alongside the plot source (for `Show`), remember the symbolic form of
  // the curves — the ContourStyle directives plus one `Line[…]` per chain —
  // so `ContourPlot[…][[1]]` yields primitives that can be re-embedded in
  // an outer `Graphics[…]` (the Demonstrations Tooltip-stripping idiom).
  let directives = contour_style_directives(args, 3);
  let mut structure_items: Vec<Expr> = if opts.contour_styles.is_empty() {
    directives.clone()
  } else {
    Vec::new()
  };
  for (body_index, range) in series_per_body.iter().enumerate() {
    // A per-contour style precedes just its own function's curves.
    if !opts.contour_styles.is_empty() {
      structure_items.push(directives[body_index % directives.len()].clone());
    }
    for s in &series[range.clone()] {
      let points: Vec<Expr> = s
        .points
        .iter()
        .map(|(x, y)| Expr::List(vec![Expr::Real(*x), Expr::Real(*y)].into()))
        .collect();
      structure_items.push(call1("Line", Expr::List(points.into())));
    }
  }
  // A contour plot is framed and has no axes unless asked otherwise, so a
  // `Show` of several of them (`Frame -> False` or not) must not grow the
  // axes that `Show` otherwise gives a plot.
  let mut source_options = crate::functions::plot::explicit_options(args);
  let has = |opts: &[Expr], n: &str| {
    opts.iter().any(|o| {
      crate::functions::graphics::option_name_value(o).map(|(name, _)| name)
        == Some(n)
    })
  };
  if !has(&source_options, "Axes") {
    source_options.push(rule_expr(id_expr("Axes"), bool_expr(false)));
  }
  if !has(&source_options, "Frame") {
    source_options.push(rule_expr(id_expr("Frame"), bool_expr(true)));
  }
  // `Graphics[prims, {opts}]`, as Wolfram's: `g[[2]]` is the option list
  // (`Graphics[g[[1]], g[[2]]]` rebuilds a plot).
  let structure = call(
    "Graphics",
    vec![
      Expr::List(vec![contour_graphics_complex(&structure_items)].into()),
      Expr::List(source_options.clone().into()),
    ],
  );
  let source = crate::syntax::PlotSource {
    series,
    x_range: (x_min, x_max),
    y_range: (y_min, y_max),
    image_size: (opts.svg_width, opts.svg_height),
    options: source_options,
    log_x: false,
    log_y: false,
  };
  let mut result = crate::graphics_result_with_source(svg, source);
  if let Expr::Graphics {
    structure: slot, ..
  } = &mut result
  {
    *slot = Some(Box::new(structure));
  }
  Ok(result)
}

/// Default `PlotPoints` of a `RegionPlot`: samples per axis of the grid the
/// region is traced on. Wolfram starts coarser and refines adaptively; a
/// finer uniform grid whose boundary crossings are then bisected to full
/// precision gets the same picture without the refinement machinery.
const REGION_PLOT_POINTS: usize = 100;

/// Options that only steer how `RegionPlot` samples and styles its regions;
/// everything else is a `Graphics` option and passes through to the result.
const REGION_PLOT_ONLY_OPTIONS: &[&str] = &[
  "PlotStyle",
  "BoundaryStyle",
  "PlotPoints",
  "MaxRecursion",
  "Mesh",
  "MeshFunctions",
  "MeshStyle",
  "MeshShading",
  "ColorFunction",
  "ColorFunctionScaling",
  "EvaluationMonitor",
  "PerformanceGoal",
  "PlotLegends",
  "PlotTheme",
  "WorkingPrecision",
  "TextureCoordinateFunction",
  "TextureCoordinateScaling",
];

/// RegionPlot[pred, {x, xmin, xmax}, {y, ymin, ymax}, opts…]
/// RegionPlot[{pred1, pred2, …}, …]
///
/// Each predicate's region is traced as exact boundary loops and drawn the
/// way Wolfram draws it: filled at 30% opacity in its `ColorData[97]` colour
/// (or its `PlotStyle`), with the boundary stroked in the same colour. The
/// picture is assembled as `Graphics[…]` primitives and rendered by the
/// shared Graphics renderer, so every `Graphics` option (`Axes`,
/// `AxesOrigin`, `GridLines`, `Frame`, `ImageSize`, …) applies, and
/// `Part`/`Show` see real geometry.
pub fn region_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let (xvar, x_min, x_max) = parse_iterator(&args[1], "RegionPlot")?;
  let (yvar, y_min, y_max) = parse_iterator(&args[2], "RegionPlot")?;
  // `p = a x^2 + b x; RegionPlot[y > p, …]` — the head holds its arguments,
  // so a body that names its function indirectly has to be evaluated once
  // before the plot variables can be substituted into it.
  let resolved = crate::functions::plot::resolve_indirect_plot_body(
    &args[0],
    &[xvar.as_str(), yvar.as_str()],
  );
  let body = resolved.as_ref().unwrap_or(&args[0]);
  let opts = &args[3..];
  let option = |key: &str| {
    opts.iter().rev().find_map(|opt| {
      let (name, value) = crate::functions::graphics::option_name_value(opt)?;
      (name == key).then(|| value.into_owned())
    })
  };

  let predicates: Vec<&Expr> = match body {
    Expr::List(items) => items.iter().collect(),
    other => vec![other],
  };
  let n = match &option("PlotPoints") {
    Some(Expr::Integer(k)) if *k >= 2 => *k as usize,
    Some(Expr::List(ks)) => match ks.first() {
      Some(Expr::Integer(k)) if *k >= 2 => *k as usize,
      _ => REGION_PLOT_POINTS,
    },
    _ => REGION_PLOT_POINTS,
  };
  // `PlotStyle -> s` styles every region, `PlotStyle -> {s1, s2, …}` one
  // region each (cycling).
  let plot_styles: Vec<Expr> = match &option("PlotStyle") {
    Some(Expr::List(items)) if predicates.len() > 1 => items.to_vec(),
    Some(Expr::Identifier(s)) if s == "Automatic" => Vec::new(),
    Some(style) => vec![style.clone()],
    None => Vec::new(),
  };
  // `BoundaryStyle -> None` drops the outline; any other explicit value
  // replaces the default stroke in the region's colour.
  let boundary_style = option("BoundaryStyle")
    .filter(|s| !matches!(s, Expr::Identifier(n) if n == "Automatic"));

  // `ColorFunction -> f` colours the region pointwise by `f[x, y]` (the
  // coordinates scaled to 0..1 over the plot range unless
  // `ColorFunctionScaling -> False`) instead of filling it with one flat tint.
  let color_function = option("ColorFunction").filter(
    |f| !matches!(f, Expr::Identifier(s) if s == "Automatic" || s == "None"),
  );
  let scaling = !matches!(
    option("ColorFunctionScaling"),
    Some(Expr::Identifier(ref s)) if s == "False"
  );

  let mut items: Vec<Expr> = Vec::new();
  for (k, pred) in predicates.iter().enumerate() {
    if let Some(cf) = &color_function {
      let cells = region_plot_color_cells(
        pred,
        cf,
        scaling,
        (&xvar, &yvar),
        (x_min, x_max),
        (y_min, y_max),
        n.min(REGION_PLOT_COLOR_CELLS),
      );
      if !cells.is_empty() {
        let mut group = vec![call0("EdgeForm")];
        group.extend(cells);
        items.push(Expr::List(group.into()));
      }
      continue;
    }
    let loops =
      region_plot_loops(pred, &xvar, &yvar, (x_min, x_max), (y_min, y_max), n);
    let polygons = region_plot_polygons(&loops);
    if polygons.is_empty() {
      continue;
    }
    let palette = crate::functions::plot::PLOT_COLORS;
    let style = (!plot_styles.is_empty())
      .then(|| plot_styles[k % plot_styles.len()].clone());
    // The default outline follows the region's colour: its `PlotStyle`
    // colour when it names one, else the palette's.
    let color = style
      .as_ref()
      .and_then(style_color)
      .unwrap_or_else(|| rgb_color(palette[k % palette.len()]));
    let mut face = vec![color.clone(), call1("Opacity", Expr::Real(0.3))];
    face.extend(style);
    let edge = match &boundary_style {
      Some(Expr::Identifier(s)) if s == "None" => call0("EdgeForm"),
      Some(style) => call1("EdgeForm", style.clone()),
      None => call1(
        "EdgeForm",
        Expr::List(
          vec![color, call1("AbsoluteThickness", Expr::Real(1.6))].into(),
        ),
      ),
    };
    let mut group = vec![edge, call("Directive", face)];
    group.extend(polygons);
    items.push(Expr::List(group.into()));
  }

  // Wolfram's RegionPlot defaults, for the options the caller left unset.
  let rule = |name: &str, value: Expr| rule_expr(id_expr(name), value);
  let range =
    |lo: f64, hi: f64| Expr::List(vec![Expr::Real(lo), Expr::Real(hi)].into());
  let scaled_pad = || {
    let side = call1("Scaled", Expr::Real(0.02));
    Expr::List(vec![side.clone(), side].into())
  };
  let defaults = vec![
    rule("AspectRatio", Expr::Integer(1)),
    rule("Frame", bool_expr(true)),
    rule(
      "PlotRange",
      Expr::List(vec![range(x_min, x_max), range(y_min, y_max)].into()),
    ),
    rule(
      "PlotRangePadding",
      Expr::List(vec![scaled_pad(), scaled_pad()].into()),
    ),
  ];
  let given: Vec<&str> = opts
    .iter()
    .filter_map(|opt| {
      crate::functions::graphics::option_name_value(opt).map(|(name, _)| name)
    })
    .collect();
  let mut graphics_args = vec![Expr::List(items.into())];
  graphics_args.extend(
    opts
      .iter()
      .filter(|opt| {
        !crate::functions::graphics::option_name_value(opt)
          .is_some_and(|(name, _)| REGION_PLOT_ONLY_OPTIONS.contains(&name))
      })
      .cloned(),
  );
  graphics_args.extend(defaults.into_iter().filter(|opt| {
    crate::functions::graphics::option_name_value(opt)
      .is_some_and(|(name, _)| !given.contains(&name))
  }));
  let mut result = crate::functions::graphics::graphics_ast(&graphics_args)?;
  // Keep the primitives on the rendering so `RegionPlot[…][[1]]` reaches
  // the region's geometry and `Show` can merge it with other layers.
  if let Expr::Graphics { structure, .. } = &mut result {
    *structure = Some(Box::new(crate::plot_option_list(call(
      "Graphics",
      graphics_args,
    ))));
  }
  Ok(result)
}

/// Cells per axis of a `ColorFunction`-coloured `RegionPlot`.
const REGION_PLOT_COLOR_CELLS: usize = 60;

/// The `{color, Rectangle[…]}` cells of a `ColorFunction`-coloured region:
/// every cell of an `n × n` grid whose centre satisfies `pred` is filled with
/// `cf[x, y]`, the coordinates scaled to 0..1 over the plot range when
/// `scaling` is set.
fn region_plot_color_cells(
  pred: &Expr,
  cf: &Expr,
  scaling: bool,
  (xvar, yvar): (&str, &str),
  (x_min, x_max): (f64, f64),
  (y_min, y_max): (f64, f64),
  n: usize,
) -> Vec<Expr> {
  let point =
    |x: f64, y: f64| Expr::List(vec![Expr::Real(x), Expr::Real(y)].into());
  let mut cells = Vec::new();
  for i in 0..n {
    let (fx0, fx1) = (i as f64 / n as f64, (i + 1) as f64 / n as f64);
    for j in 0..n {
      let (fy0, fy1) = (j as f64 / n as f64, (j + 1) as f64 / n as f64);
      let fx = f64::midpoint(fx0, fx1);
      let fy = f64::midpoint(fy0, fy1);
      let (cx, cy) =
        (x_min + fx * (x_max - x_min), y_min + fy * (y_max - y_min));
      if !evaluate_condition(pred, xvar, yvar, cx, cy) {
        continue;
      }
      let (ax, ay) = if scaling { (fx, fy) } else { (cx, cy) };
      let applied = Expr::CurriedCall {
        func: Box::new(cf.clone()),
        args: vec![Expr::Real(ax), Expr::Real(ay)],
      };
      let Some(color) = evaluate_expr_to_expr(&applied)
        .ok()
        .and_then(|r| parse_color(&r))
      else {
        continue;
      };
      let rect = call(
        "Rectangle",
        vec![
          point(x_min + fx0 * (x_max - x_min), y_min + fy0 * (y_max - y_min)),
          point(x_min + fx1 * (x_max - x_min), y_min + fy1 * (y_max - y_min)),
        ],
      );
      cells.push(Expr::List(
        vec![
          call(
            "RGBColor",
            vec![
              Expr::Real(color.r),
              Expr::Real(color.g),
              Expr::Real(color.b),
            ],
          ),
          rect,
        ]
        .into(),
      ));
    }
  }
  cells
}

/// The colour a `PlotStyle` entry names — the entry itself, or the last
/// colour inside a `Directive[…]`/list of directives — as `RGBColor[…]`.
fn style_color(style: &Expr) -> Option<Expr> {
  match style {
    Expr::List(items) => items.iter().rev().find_map(style_color),
    Expr::FunctionCall { name, args } if name == "Directive" => {
      args.iter().rev().find_map(style_color)
    }
    other => {
      let c = parse_color(other)?;
      Some(call(
        "RGBColor",
        vec![Expr::Real(c.r), Expr::Real(c.g), Expr::Real(c.b)],
      ))
    }
  }
}

/// Traces the boundary of the region where `pred` holds as closed loops in
/// data coordinates.
///
/// The predicate is sampled on an `(n + 1) × (n + 1)` grid padded with a
/// ring of "outside" samples that sit on the frame itself, so a region that
/// reaches the edge of the plot range is closed off along the frame instead
/// of leaving an open contour. Marching squares then finds the cell edges
/// the boundary crosses, and each crossing is bisected on the predicate
/// itself, so straight boundaries (linear inequalities) come out straight
/// rather than as a staircase of grid steps.
fn region_plot_loops(
  pred: &Expr,
  xvar: &str,
  yvar: &str,
  (x_min, x_max): (f64, f64),
  (y_min, y_max): (f64, f64),
  n: usize,
) -> Vec<Vec<(f64, f64)>> {
  use std::collections::HashMap;
  // The interior sample lines are nudged off the round values by a small,
  // different fraction of a cell per axis: a boundary through "nice" points
  // (`y > x/2 + 1` passes exactly through many grid nodes of a round range)
  // would otherwise put samples right on it, where the strict inequality is
  // false, pinching the region apart at each of them. The frame lines stay
  // exact so the region still closes along the plot range.
  let axis = |lo: f64, hi: f64, nudge: f64| -> Vec<f64> {
    (0..=n)
      .map(|i| {
        let t = if i == 0 || i == n {
          i as f64
        } else {
          i as f64 + nudge
        };
        lo + t / n as f64 * (hi - lo)
      })
      .collect()
  };
  let xs = axis(x_min, x_max, 0.0137);
  let ys = axis(y_min, y_max, 0.0291);
  let inside: Vec<Vec<bool>> = xs
    .iter()
    .map(|&x| {
      ys.iter()
        .map(|&y| evaluate_condition(pred, xvar, yvar, x, y))
        .collect()
    })
    .collect();
  if !inside.iter().flatten().any(|&b| b) {
    return Vec::new();
  }

  // Padded index `p` in `0..=n + 2` maps to grid index `p - 1`, clamped so
  // the pad ring shares the frame's coordinates; only real samples can be
  // inside.
  let m = n + 2;
  let grid_idx = |p: usize| p.saturating_sub(1).min(n);
  let is_in = |p: usize, q: usize| {
    (1..=n + 1).contains(&p) && (1..=n + 1).contains(&q) && inside[p - 1][q - 1]
  };
  let coord =
    |p: usize, q: usize| -> (f64, f64) { (xs[grid_idx(p)], ys[grid_idx(q)]) };

  let precision = (x_max - x_min)
    .abs()
    .max((y_max - y_min).abs())
    .max(f64::MIN_POSITIVE)
    * 1e-14;
  // The boundary point on the edge from padded vertex `a` to `b` (one
  // inside, one outside), shared by the two cells that edge borders.
  let mut crossings: HashMap<((usize, usize), (usize, usize)), (f64, f64)> =
    HashMap::new();
  let mut crossing = |a: (usize, usize), b: (usize, usize)| -> (f64, f64) {
    let key = if a <= b { (a, b) } else { (b, a) };
    if let Some(&pt) = crossings.get(&key) {
      return pt;
    }
    let (mut lo, mut hi) = if is_in(a.0, a.1) {
      (coord(a.0, a.1), coord(b.0, b.1))
    } else {
      (coord(b.0, b.1), coord(a.0, a.1))
    };
    // A pad edge has both ends on the frame: the boundary runs along it.
    if lo != hi {
      for _ in 0..60 {
        if (lo.0 - hi.0).abs().max((lo.1 - hi.1).abs()) <= precision {
          break;
        }
        let mid = (f64::midpoint(lo.0, hi.0), f64::midpoint(lo.1, hi.1));
        if evaluate_condition(pred, xvar, yvar, mid.0, mid.1) {
          lo = mid;
        } else {
          hi = mid;
        }
      }
    }
    // A boundary on an axis comes out as a tiny residue of the bisection;
    // snap it to the exact zero it stands for.
    let chop = |v: f64| if v.abs() <= precision { 0.0 } else { v };
    let pt = (
      chop(f64::midpoint(lo.0, hi.0)),
      chop(f64::midpoint(lo.1, hi.1)),
    );
    crossings.insert(key, pt);
    pt
  };

  let mut segments: Vec<((f64, f64), (f64, f64))> = Vec::new();
  for p in 0..m {
    for q in 0..m {
      let (bl, br, tl, tr) = ((p, q), (p + 1, q), (p, q + 1), (p + 1, q + 1));
      let case = is_in(bl.0, bl.1) as u8
        | ((is_in(br.0, br.1) as u8) << 1)
        | ((is_in(tl.0, tl.1) as u8) << 2)
        | ((is_in(tr.0, tr.1) as u8) << 3);
      if case == 0 || case == 15 {
        continue;
      }
      let mut seg = |e1: ((usize, usize), (usize, usize)),
                     e2: ((usize, usize), (usize, usize))| {
        let a = crossing(e1.0, e1.1);
        let b = crossing(e2.0, e2.1);
        segments.push((a, b));
      };
      let bottom = (bl, br);
      let top = (tl, tr);
      let left = (bl, tl);
      let right = (br, tr);
      match case {
        1 | 14 => seg(bottom, left),
        2 | 13 => seg(bottom, right),
        3 | 12 => seg(left, right),
        4 | 11 => seg(left, top),
        5 | 10 => seg(bottom, top),
        7 | 8 => seg(right, top),
        // Saddles: keep the two inside corners apart.
        6 => {
          seg(bottom, right);
          seg(left, top);
        }
        9 => {
          seg(bottom, left);
          seg(right, top);
        }
        _ => {}
      }
    }
  }

  let span = (x_max - x_min).abs().max((y_max - y_min).abs());
  let key_scale = if span > 0.0 { 1e9 / span } else { 1e9 };
  chain_segments_scaled(&segments, key_scale)
    .into_iter()
    .filter_map(|mut chain| {
      // Every loop closes (the pad ring guarantees it); drop the repeated
      // closing point and the collinear points of straight runs.
      if chain.len() > 1 && chain.first() == chain.last() {
        chain.pop();
      }
      let simplified = drop_collinear_points(&chain, span * 1e-9);
      let cell = span / n as f64;
      let sharpened = sharpen_corners(&simplified, cell);
      (sharpened.len() >= 3).then_some(sharpened)
    })
    .collect()
}

/// Restores the sharp corners marching squares cuts off. Where two straight
/// boundary pieces meet (`y > x/2 + 1 && y < 3x/2 - 1` at `{2, 2}`), the
/// corner falls inside a grid cell and the traced loop bevels it with one
/// to a few short edges. Such a short run between two long straight edges
/// is replaced by the point where those edges' lines meet, provided it lies
/// near the run. A curved boundary consists of short edges only, so it is
/// left alone.
fn sharpen_corners(ring: &[(f64, f64)], cell: f64) -> Vec<(f64, f64)> {
  let len = ring.len();
  if len < 5 || cell <= 0.0 {
    return ring.to_vec();
  }
  let dist = |a: (f64, f64), b: (f64, f64)| (a.0 - b.0).hypot(a.1 - b.1);
  // Edge `i` runs from `ring[i]` to `ring[i + 1]`.
  let edge_len = |i: usize| dist(ring[i % len], ring[(i + 1) % len]);
  let long = |i: usize| edge_len(i) > 3.0 * cell;
  let short = |i: usize| edge_len(i) < 2.0 * cell;
  let mut drop = vec![false; len];
  let mut replace: Vec<Option<(f64, f64)>> = vec![None; len];
  for start in 0..len {
    // `start` is the long edge before a run of 1–3 short edges.
    if !long(start) {
      continue;
    }
    let mut run = 0;
    while run < 3 && short(start + 1 + run) {
      run += 1;
    }
    let after = start + 1 + run;
    if run == 0 || !long(after) {
      continue;
    }
    let (a, b) = (ring[start % len], ring[(start + 1) % len]);
    let (c, d) = (ring[after % len], ring[(after + 1) % len]);
    let (r, q) = ((b.0 - a.0, b.1 - a.1), (d.0 - c.0, d.1 - c.1));
    let denom = r.0 * q.1 - r.1 * q.0;
    if denom.abs() < 1e-12 * dist(a, b) * dist(c, d) {
      continue;
    }
    let t = ((c.0 - a.0) * q.1 - (c.1 - a.1) * q.0) / denom;
    let corner = (a.0 + t * r.0, a.1 + t * r.1);
    if dist(corner, b) > 2.0 * cell || dist(corner, c) > 2.0 * cell {
      continue;
    }
    replace[(start + 1) % len] = Some(corner);
    for k in 2..=run {
      drop[(start + k) % len] = true;
    }
    drop[after % len] = true;
  }
  (0..len)
    .filter(|&i| !drop[i] || replace[i].is_some())
    .map(|i| replace[i].unwrap_or(ring[i]))
    .collect()
}

/// Removes the points of a closed loop that lie on the straight line
/// between their neighbours, so a straight boundary is a single edge.
fn drop_collinear_points(ring: &[(f64, f64)], tol: f64) -> Vec<(f64, f64)> {
  let mut pts = ring.to_vec();
  loop {
    let len = pts.len();
    if len < 4 {
      return pts;
    }
    let keep: Vec<bool> = (0..len)
      .map(|i| {
        let (a, b, c) = (pts[(i + len - 1) % len], pts[i], pts[(i + 1) % len]);
        let cross = (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
        let base = ((c.0 - a.0).powi(2) + (c.1 - a.1).powi(2)).sqrt();
        // Distance of `b` from the line `a`–`c`, and `b` between them.
        let dot = (b.0 - a.0) * (c.0 - a.0) + (b.1 - a.1) * (c.1 - a.1);
        !(base > 0.0
          && cross.abs() / base <= tol
          && dot >= 0.0
          && dot <= base * base)
      })
      .collect();
    if keep.iter().all(|&k| k) {
      return pts;
    }
    // Drop every other removable point per pass so two neighbouring
    // removals never both rely on each other.
    let mut removed_prev = false;
    let mut next = Vec::with_capacity(len);
    for (i, &pt) in pts.iter().enumerate() {
      if !keep[i] && !removed_prev {
        removed_prev = true;
      } else {
        next.push(pt);
        removed_prev = false;
      }
    }
    pts = next;
  }
}

/// Turns a region's boundary loops into `Polygon`s: a loop nested inside an
/// odd number of others is a hole of the innermost loop around it, and comes
/// out as `Polygon[outer -> {holes…}]`.
fn region_plot_polygons(loops: &[Vec<(f64, f64)>]) -> Vec<Expr> {
  fn contains(ring: &[(f64, f64)], (px, py): (f64, f64)) -> bool {
    let mut inside = false;
    let len = ring.len();
    for i in 0..len {
      let (x1, y1) = ring[i];
      let (x2, y2) = ring[(i + 1) % len];
      if (y1 > py) != (y2 > py) && px < x1 + (py - y1) / (y2 - y1) * (x2 - x1) {
        inside = !inside;
      }
    }
    inside
  }
  // A probe strictly inside the loop's own edge band: the midpoint of its
  // first edge never lies on another loop, unlike a shared frame vertex.
  let probe = |ring: &[(f64, f64)]| {
    let (a, b) = (ring[0], ring[1]);
    (f64::midpoint(a.0, b.0), f64::midpoint(a.1, b.1))
  };
  let parents: Vec<Vec<usize>> = loops
    .iter()
    .enumerate()
    .map(|(i, ring)| {
      let pt = probe(ring);
      (0..loops.len())
        .filter(|&j| j != i && contains(&loops[j], pt))
        .collect()
    })
    .collect();
  let pt_list = |ring: &[(f64, f64)]| {
    Expr::List(
      ring
        .iter()
        .map(|&(x, y)| Expr::List(vec![Expr::Real(x), Expr::Real(y)].into()))
        .collect(),
    )
  };
  let mut polygons = Vec::new();
  for (i, ring) in loops.iter().enumerate() {
    if parents[i].len() % 2 == 1 {
      continue;
    }
    let holes: Vec<Expr> = (0..loops.len())
      .filter(|&h| {
        parents[h].len() == parents[i].len() + 1 && parents[h].contains(&i)
      })
      .map(|h| pt_list(&loops[h]))
      .collect();
    polygons.push(if holes.is_empty() {
      call1("Polygon", pt_list(ring))
    } else {
      call1(
        "Polygon",
        rule_expr(pt_list(ring), Expr::List(holes.into())),
      )
    });
  }
  polygons
}

/// How `VectorPlot` draws its vectors, from `ColorFunction` and
/// `VectorStyle`.
#[derive(Default)]
struct VectorArrowStyle {
  /// `ColorFunction -> f`: `f` maps each vector's scaled magnitude to a color.
  color_function: Option<Expr>,
  /// `VectorStyle -> color`: one color for every vector.
  color: Option<(u8, u8, u8)>,
  /// `VectorStyle -> "Segment"`: plain segments without arrowheads.
  segments: bool,
}

/// The `ColorFunction` and `VectorStyle` options among `args[start..]`.
fn parse_vector_arrow_style(args: &[Expr], start: usize) -> VectorArrowStyle {
  let mut style = VectorArrowStyle::default();
  for opt in args.iter().skip(start) {
    let Expr::Rule {
      pattern,
      replacement,
    } = opt
    else {
      continue;
    };
    let Expr::Identifier(name) = pattern.as_ref() else {
      continue;
    };
    match name.as_str() {
      "ColorFunction" => {
        if !matches!(replacement.as_ref(), Expr::Identifier(s) if s == "Automatic" || s == "None")
        {
          style.color_function = Some((**replacement).clone());
        }
      }
      "VectorStyle" => {
        let directives: Vec<Expr> = match replacement.as_ref() {
          Expr::List(items) => items.to_vec(),
          other => vec![other.clone()],
        };
        for d in &directives {
          if matches!(d, Expr::String(s) if s == "Segment") {
            style.segments = true;
          } else if let Some(c) = crate::functions::graphics::parse_color(d) {
            style.color = Some((
              (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
              (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
              (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
            ));
          }
        }
      }
      _ => {}
    }
  }
  style
}

/// `VectorPoints -> n`: the number of sample points across each axis.
fn parse_vector_points(args: &[Expr], start: usize) -> Option<usize> {
  let value = option_rule(&args[start..], "VectorPoints")?;
  let n = try_eval_to_f64(&evaluate_expr_to_expr(value).ok()?)?;
  (n >= 2.0).then(|| n.round() as usize)
}

/// Draws arrows for a set of `(x, y, vx, vy, magnitude)` vectors onto `svg`,
/// scaled so the largest arrow spans about `cell_size * 0.4` pixels, and
/// returns the `{color, Arrow[...]}` primitives drawn (so callers can hand
/// them back as `Graphics[{…}]`, e.g. for `VectorPlot[…][[1]]`).
///
/// Each vector points along the direction it has in the plot's own data
/// coordinates, so a slope field stays tangent to the curves plotted over it
/// even when the two axes have different scales.
///
/// Shared by `VectorPlot` (vectors sampled from a symbolic field over a
/// regular grid) and `ListVectorPlot` (vectors supplied as literal data), so
/// both draw identically instead of maintaining two arrow renderers.
fn render_vector_arrows(
  vectors: &[(f64, f64, f64, f64, f64)],
  max_mag: f64,
  cell_size: f64,
  area: &crate::functions::plot::PlotArea,
  style: &VectorArrowStyle,
  svg: &mut String,
) -> Vec<Expr> {
  let mut primitives: Vec<Expr> = Vec::new();
  if max_mag <= 0.0 {
    return primitives;
  }

  let to_px = |x: f64, y: f64| -> (f64, f64) {
    let sx =
      area.plot_x0 + (x - area.x_min) / (area.x_max - area.x_min) * area.plot_w;
    let sy =
      area.plot_y0 + (area.y_max - y) / (area.y_max - area.y_min) * area.plot_h;
    (sx, sy)
  };

  // Pixels per unit of each axis: the vector's on-screen direction.
  let px_per_x = area.plot_w / (area.x_max - area.x_min);
  let px_per_y = area.plot_h / (area.y_max - area.y_min);
  let longest_px = vectors
    .iter()
    .map(|&(_, _, vx, vy, _)| (vx * px_per_x).hypot(vy * px_per_y))
    .fold(0.0_f64, f64::max);
  if longest_px <= 0.0 {
    return primitives;
  }
  let arrow_scale = cell_size * 0.4 / longest_px;
  let stroke_w = area.render_width as f64 / 1000.0 * 1.5;
  // The pixel-space arrow length maps back to a data-space length so the
  // symbolic arrows have the same footprint as the drawn ones.
  let min_mag = vectors.iter().map(|v| v.4).fold(f64::INFINITY, f64::min);
  for &(x, y, vx, vy, mag) in vectors {
    if mag < 1e-15 {
      continue;
    }

    let (sx, sy) = to_px(x, y);
    let dx = vx * px_per_x * arrow_scale;
    let dy = -vy * px_per_y * arrow_scale; // SVG y is flipped

    let ex = sx + dx;
    let ey = sy + dy;

    // Color: an explicit `VectorStyle` color, else `ColorFunction` of the
    // scaled magnitude, else a magnitude gradient.
    let t = (mag / max_mag).clamp(0.0, 1.0);
    let default_color = (
      (t * 200.0) as u8 + 50,
      ((1.0 - t) * 150.0) as u8 + 50,
      100_u8,
    );
    let (r, g, b) = style.color.unwrap_or_else(|| {
      style
        .color_function
        .as_ref()
        .and_then(|f| {
          let scaled = if max_mag > min_mag {
            (mag - min_mag) / (max_mag - min_mag)
          } else {
            0.0
          };
          let call = Expr::CurriedCall {
            func: Box::new(f.clone()),
            args: vec![Expr::Real(scaled)],
          };
          let c = crate::functions::graphics::parse_color(
            &evaluate_expr_to_expr(&call).ok()?,
          )?;
          Some((
            (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
            (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
            (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
          ))
        })
        .unwrap_or(default_color)
    });

    primitives.push(Expr::List(
      vec![
        rgb_color((r, g, b)),
        call1(
          "Arrow",
          Expr::List(
            vec![
              Expr::List(vec![Expr::Real(x), Expr::Real(y)].into()),
              Expr::List(
                vec![
                  Expr::Real(x + dx / px_per_x),
                  Expr::Real(y - dy / px_per_y),
                ]
                .into(),
              ),
            ]
            .into(),
          ),
        ),
      ]
      .into(),
    ));

    // Arrow shaft
    svg.push_str(&format!(
      "<line x1=\"{sx:.1}\" y1=\"{sy:.1}\" x2=\"{ex:.1}\" y2=\"{ey:.1}\" stroke=\"rgb({r},{g},{b})\" stroke-width=\"{stroke_w:.1}\"/>\n"
    ));

    // Arrowhead
    let len = (dx * dx + dy * dy).sqrt();
    if len > 2.0 && !style.segments {
      let ux = dx / len;
      let uy = dy / len;
      let head_len = len * 0.3;
      let head_w = head_len * 0.4;
      let px = -uy;
      let py = ux;
      let bx1 = ex - ux * head_len + px * head_w;
      let by1 = ey - uy * head_len + py * head_w;
      let bx2 = ex - ux * head_len - px * head_w;
      let by2 = ey - uy * head_len - py * head_w;
      svg.push_str(&format!(
        "<polygon points=\"{ex:.1},{ey:.1} {bx1:.1},{by1:.1} {bx2:.1},{by2:.1}\" fill=\"rgb({r},{g},{b})\"/>\n"
      ));
    }
  }

  primitives
}

/// VectorPlot[{vx, vy}, {x, xmin, xmax}, {y, ymin, ymax}]
pub fn vector_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let (xvar, x_min, x_max) = parse_iterator(&args[1], "VectorPlot")?;
  let (yvar, y_min, y_max) = parse_iterator(&args[2], "VectorPlot")?;
  // `p = a x^2 + b x; RegionPlot[y > p, …]` — the head holds its arguments,
  // so a body that names its function indirectly has to be evaluated once
  // before the plot variables can be substituted into it.
  let resolved = crate::functions::plot::resolve_indirect_plot_body(
    &args[0],
    &[xvar.as_str(), yvar.as_str()],
  );
  let body = resolved.as_ref().unwrap_or(&args[0]);
  let (svg_width, svg_height, full_width) = parse_field_options(args, 3);
  let epilog = parse_vector_plot_epilog(args, 3);
  let prolog = parse_overlay_primitives(args, 3, "Prolog");
  let plot_label = parse_field_plot_label(args, 3);

  let arrow_style = parse_vector_arrow_style(args, 3);
  // `VectorPoints -> n` samples n points across each axis.
  let grid = parse_vector_points(args, 3).map_or(VECTOR_GRID, |n| n - 1);
  let x_step = (x_max - x_min) / grid as f64;
  let y_step = (y_max - y_min) / grid as f64;

  // Sample vectors and find max magnitude
  let mut vectors = Vec::new();
  let mut max_mag = 0.0_f64;
  for i in 0..=grid {
    let x = x_min + i as f64 * x_step;
    for j in 0..=grid {
      let y = y_min + j as f64 * y_step;
      if let Some((vx, vy)) = evaluate_vector(body, &xvar, &yvar, x, y) {
        let mag = (vx * vx + vy * vy).sqrt();
        max_mag = max_mag.max(mag);
        vectors.push((x, y, vx, vy, mag));
      }
    }
  }

  // Use plotters for axes, reserving room above the frame for a PlotLabel.
  let margins = field_plot_label_margins(plot_label.as_ref());
  let mut area = crate::functions::plot::generate_axes_only_opts(
    (x_min, x_max),
    (y_min, y_max),
    svg_width,
    svg_height,
    full_width,
    None,
    margins.as_ref(),
  )?;

  let mut svg = std::mem::take(&mut area.svg);
  if let Some(pos) = svg.rfind("</svg>") {
    svg.truncate(pos);
  }
  splice_field_plot_prolog(&mut svg, &area, &prolog);
  if let Some(label) = &plot_label {
    inject_field_plot_label(&mut svg, &area, label);
  }

  let cell_size = (area.plot_w / grid as f64).min(area.plot_h / grid as f64);

  // Arrows in data coordinates, kept so `VectorPlot[…][[1]]` yields the
  // primitives a surrounding `Graphics[{…}]` can redraw.
  let primitives = render_vector_arrows(
    &vectors,
    max_mag,
    cell_size,
    &area,
    &arrow_style,
    &mut svg,
  );

  push_epilog_primitives(&mut svg, &area, &epilog);

  svg.push_str("</svg>");
  Ok(crate::plot_result_with_structure(
    svg,
    call1("Graphics", Expr::List(primitives.into())),
  ))
}

/// Reads a literal `{x, y}` point/vector pair, evaluating each coordinate.
fn as_xy_pair(expr: &Expr) -> Option<(f64, f64)> {
  let Expr::List(items) = expr else {
    return None;
  };
  if items.len() != 2 {
    return None;
  }
  let x = try_eval_to_f64(&items[0])?;
  let y = try_eval_to_f64(&items[1])?;
  Some((x, y))
}

/// ListVectorPlot[{{{x1, y1}, {vx1, vy1}}, {{x2, y2}, {vx2, vy2}}, ...}]
///
/// Draws arrows from literal `{position, vector}` data, rather than sampling
/// a symbolic field over a grid the way `VectorPlot` does. This is also the
/// modern name for the legacy `VectorFieldPlots\`ListVectorFieldPlot`, whose
/// calls are normalized to `ListVectorPlot` before dispatch (see
/// `evaluate_function_call_ast`) since the two take identical data.
pub fn list_vector_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let data = evaluate_expr_to_expr(&args[0])?;
  let Expr::List(rows) = &data else {
    return Err(InterpreterError::EvaluationError(
      "ListVectorPlot: first argument must be a list of {position, vector} pairs"
        .into(),
    ));
  };

  let mut vectors = Vec::new();
  let mut max_mag = 0.0_f64;
  let (mut x_min, mut x_max, mut y_min, mut y_max) = (
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::INFINITY,
    f64::NEG_INFINITY,
  );
  for row in rows {
    let Expr::List(pair) = row else { continue };
    if pair.len() != 2 {
      continue;
    }
    let Some((x, y)) = as_xy_pair(&pair[0]) else {
      continue;
    };
    let Some((vx, vy)) = as_xy_pair(&pair[1]) else {
      continue;
    };
    x_min = x_min.min(x);
    x_max = x_max.max(x);
    y_min = y_min.min(y);
    y_max = y_max.max(y);
    let mag = (vx * vx + vy * vy).sqrt();
    max_mag = max_mag.max(mag);
    vectors.push((x, y, vx, vy, mag));
  }

  if vectors.is_empty() || !x_min.is_finite() || !y_min.is_finite() {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }
  // A single row/column of data (or all-equal coordinates) has zero span;
  // pad it so the axes aren't degenerate.
  if (x_max - x_min).abs() < 1e-12 {
    x_min -= 1.0;
    x_max += 1.0;
  }
  if (y_max - y_min).abs() < 1e-12 {
    y_min -= 1.0;
    y_max += 1.0;
  }

  let (svg_width, svg_height, full_width) = parse_field_options(args, 1);
  let mut area = generate_axes_only(
    (x_min, x_max),
    (y_min, y_max),
    svg_width,
    svg_height,
    full_width,
  )?;
  let mut svg = std::mem::take(&mut area.svg);
  if let Some(pos) = svg.rfind("</svg>") {
    svg.truncate(pos);
  }

  // Data points aren't laid out on a regular grid, so the arrow scale is
  // based on the average spacing implied by the plot area and point count
  // instead of `VectorPlot`'s fixed sampling grid.
  let cell_size =
    (area.plot_w * area.plot_h / vectors.len().max(1) as f64).sqrt();
  let arrow_style = parse_vector_arrow_style(args, 1);
  let primitives = render_vector_arrows(
    &vectors,
    max_mag,
    cell_size,
    &area,
    &arrow_style,
    &mut svg,
  );

  svg.push_str("</svg>");
  Ok(crate::plot_result_with_structure(
    svg,
    call1("Graphics", Expr::List(primitives.into())),
  ))
}

/// StreamPlot[{vx, vy}, {x, xmin, xmax}, {y, ymin, ymax}]
/// Draws evenly spaced streamlines of the field (see `stream_primitives`).
pub fn stream_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let (xvar, x_min, x_max) = parse_iterator(&args[1], "StreamPlot")?;
  let (yvar, y_min, y_max) = parse_iterator(&args[2], "StreamPlot")?;
  // `p = a x^2 + b x; RegionPlot[y > p, …]` — the head holds its arguments,
  // so a body that names its function indirectly has to be evaluated once
  // before the plot variables can be substituted into it.
  let resolved = crate::functions::plot::resolve_indirect_plot_body(
    &args[0],
    &[xvar.as_str(), yvar.as_str()],
  );
  let body = resolved.as_ref().unwrap_or(&args[0]);
  let (svg_width, svg_height, full_width) = parse_field_options(args, 3);
  let plot_label = parse_field_plot_label(args, 3);
  let prolog = parse_overlay_primitives(args, 3, "Prolog");
  let epilog = parse_overlay_primitives(args, 3, "Epilog");

  // Use plotters for axes, reserving room above the frame for a PlotLabel.
  let margins = field_plot_label_margins(plot_label.as_ref());
  let area = crate::functions::plot::generate_axes_only_opts(
    (x_min, x_max),
    (y_min, y_max),
    svg_width,
    svg_height,
    full_width,
    None,
    margins.as_ref(),
  )?;

  let mut svg = area.svg.clone();
  if let Some(pos) = svg.rfind("</svg>") {
    svg.truncate(pos);
  }
  splice_field_plot_prolog(&mut svg, &area, &prolog);
  if let Some(label) = &plot_label {
    inject_field_plot_label(&mut svg, &area, label);
  }

  let primitives =
    stream_primitives(body, &xvar, &yvar, (x_min, x_max), (y_min, y_max), args);
  push_stream_svg(&mut svg, &primitives, &area);

  push_epilog_primitives(&mut svg, &area, &epilog);

  svg.push_str("</svg>");
  Ok(crate::plot_result_with_structure(
    svg,
    call1("Graphics", Expr::List(primitives.into())),
  ))
}

/// The value of option `name` among a plot's arguments.
fn option_rule<'a>(args: &'a [Expr], name: &str) -> Option<&'a Expr> {
  args.iter().find_map(|a| match a {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } if matches!(&**pattern, Expr::Identifier(n) if n == name) => {
      Some(&**replacement)
    }
    _ => None,
  })
}

/// `RGBColor[r, g, b]` from 8-bit channel values.
fn rgb_color((r, g, b): (u8, u8, u8)) -> Expr {
  call(
    "RGBColor",
    vec![
      Expr::Real(r as f64 / 255.0),
      Expr::Real(g as f64 / 255.0),
      Expr::Real(b as f64 / 255.0),
    ],
  )
}

/// The streamlines `StreamPlot` and `StreamDensityPlot` draw over the
/// domain, as primitives in data coordinates: evenly spaced streamlines of
/// the field `body`, each a run of arrows coloured by the flow speed, styled
/// by the plot's `StreamPoints`, `StreamColorFunction`, `StreamStyle` and
/// `RegionFunction` options among `args`.
fn stream_primitives(
  body: &Expr,
  xvar: &str,
  yvar: &str,
  (x_min, x_max): (f64, f64),
  (y_min, y_max): (f64, f64),
  args: &[Expr],
) -> Vec<Expr> {
  // `StreamStyle -> color` paints every streamline alike; the default
  // cycles through the plot colours.
  let stream_color = args.iter().skip(3).find_map(|a| match a {
    Expr::Rule {
      pattern,
      replacement,
    } if matches!(&**pattern, Expr::Identifier(n) if n == "StreamStyle") => {
      let style = crate::functions::plot::parse_style_directives(replacement)?;
      let c = style.color?;
      Some((
        (c.r * 255.0).round() as u8,
        (c.g * 255.0).round() as u8,
        (c.b * 255.0).round() as u8,
      ))
    }
    _ => None,
  });

  // `RegionFunction -> f` confines the streamlines: `f[x, y]` must hold at
  // every point drawn.
  let region = args.iter().skip(3).find_map(|a| match a {
    Expr::Rule {
      pattern,
      replacement,
    } if matches!(&**pattern, Expr::Identifier(n) if n == "RegionFunction") => {
      Some((**replacement).clone())
    }
    _ => None,
  });
  let in_region = |x: f64, y: f64| -> bool {
    region.as_ref().is_none_or(|r| {
      let call = Expr::CurriedCall {
        func: Box::new(r.clone()),
        args: vec![Expr::Real(x), Expr::Real(y)],
      };
      matches!(
        evaluate_expr_to_expr(&call),
        Ok(Expr::Identifier(ref s)) if s == "True"
      )
    })
  };

  // `StreamPoints` sets how densely the plot is filled, as the number of
  // streamlines across either axis. Wolfram's own spacing depends on the
  // field — about 29 lines across a uniform flow but 17 across the flow
  // around a cylinder — so the default sits between the two, and the named
  // and numeric settings keep Wolfram's proportions to it (`StreamPoints ->
  // n` grows as √n).
  let lines_across = match option_rule(args, "StreamPoints") {
    Some(Expr::Identifier(s)) if s == "Coarse" => 11.0,
    Some(Expr::Identifier(s)) if s == "Medium" => 16.0,
    Some(Expr::Identifier(s)) if s == "Fine" => 26.0,
    Some(e) => try_eval_to_f64(e)
      .filter(|n| *n >= 1.0)
      .map_or(22.0, |n| 2.2 * n.sqrt()),
    None => 22.0,
  };

  // The field, sampled once over the plot and interpolated between nodes;
  // a node outside the `RegionFunction` holds no flow.
  let field = crate::functions::streamlines::SampledField::sample(
    |x, y| {
      in_region(x, y)
        .then(|| evaluate_vector(body, xvar, yvar, x, y))
        .flatten()
    },
    (x_min, x_max),
    (y_min, y_max),
    64,
  );
  let lines =
    crate::functions::streamlines::even_streamlines(&field, 1.0 / lines_across);

  // Each streamline is drawn as a run of arrows, each 8% of the plot range
  // long with its head at the end (shorter at a streamline's end, with a
  // proportionally smaller head), coloured by the flow speed through it.
  const SEGMENT: f64 = 0.08;
  let normalized_len = |a: (f64, f64), b: (f64, f64)| {
    let du = (b.0 - a.0) / (x_max - x_min);
    let dv = (b.1 - a.1) / (y_max - y_min);
    (du * du + dv * dv).sqrt()
  };
  let mut segments: Vec<(Vec<(f64, f64)>, f64, f64)> = Vec::new();
  for line in &lines {
    let mut current = vec![line[0]];
    let mut length = 0.0;
    for pair in line.windows(2) {
      current.push(pair[1]);
      length += normalized_len(pair[0], pair[1]);
      if length >= SEGMENT {
        segments.push((
          std::mem::replace(&mut current, vec![pair[1]]),
          length,
          0.0,
        ));
        length = 0.0;
      }
    }
    if current.len() > 1 {
      segments.push((current, length, 0.0));
    }
  }
  for (points, _, speed) in &mut segments {
    let mid = points[points.len() / 2];
    *speed = field
      .at(mid.0, mid.1)
      .map_or(0.0, |(vx, vy)| (vx * vx + vy * vy).sqrt());
  }
  let (speed_min, speed_max) = segments
    .iter()
    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), (_, _, s)| {
      (lo.min(*s), hi.max(*s))
    });

  // By default the speed picks a colour on WL12DefaultVectorGradient, which
  // overrides a `StreamStyle` colour as in Wolfram. `StreamColorFunction ->
  // None` draws every streamline in the `StreamStyle` colour (or the first
  // plot colour); a gradient name or a function replaces the default one.
  let color_function = option_rule(args, "StreamColorFunction");
  let color_of = |speed: f64| -> (u8, u8, u8) {
    let t = if speed_max > speed_min {
      (speed - speed_min) / (speed_max - speed_min)
    } else {
      0.5
    };
    match color_function {
      Some(Expr::Identifier(s)) if s == "None" => {
        stream_color.unwrap_or((49, 122, 163))
      }
      Some(Expr::String(name)) => apply_named_color_function(name.as_str(), t),
      Some(Expr::Identifier(s)) if s == "Automatic" => {
        crate::functions::plot::wl_default_vector_gradient(t)
      }
      Some(f) => scaled_color(t, speed, None, Some((f, true))),
      None => crate::functions::plot::wl_default_vector_gradient(t),
    }
  };
  let primitives: Vec<Expr> = segments
    .into_iter()
    .map(|(points, length, speed)| {
      let head = 0.02 * (length / SEGMENT).min(1.0);
      let coords = Expr::List(
        points
          .into_iter()
          .map(|(x, y)| Expr::List(vec![Expr::Real(x), Expr::Real(y)].into()))
          .collect(),
      );
      Expr::List(
        vec![
          call1(
            "Arrowheads",
            Expr::List(
              vec![Expr::List(vec![Expr::Real(head), Expr::Real(1.0)].into())]
                .into(),
            ),
          ),
          rgb_color(color_of(speed)),
          call1("Arrow", coords),
        ]
        .into(),
      )
    })
    .collect();
  primitives
}

/// Draw `stream_primitives` over the plot area of a field plot.
fn push_stream_svg(
  svg: &mut String,
  primitives: &[Expr],
  area: &crate::functions::plot::PlotArea,
) {
  svg.push_str(&crate::functions::plot_epilog::render_epilog_svg(
    primitives,
    &crate::functions::plot_epilog::PlotArea {
      x0: area.plot_x0,
      y0: area.plot_y0,
      w: area.plot_w,
      h: area.plot_h,
      x_min: area.x_min,
      x_max: area.x_max,
      y_min: area.y_min,
      y_max: area.y_max,
      scale: f64::from(RESOLUTION_SCALE),
    },
    "streams",
  ));
}
/// StreamDensityPlot: StreamPlot overlaid on DensityPlot background
pub fn stream_density_plot_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let (xvar, x_min, x_max) = parse_iterator(&args[1], "StreamDensityPlot")?;
  let (yvar, y_min, y_max) = parse_iterator(&args[2], "StreamDensityPlot")?;
  // `p = a x^2 + b x; RegionPlot[y > p, …]` — the head holds its arguments,
  // so a body that names its function indirectly has to be evaluated once
  // before the plot variables can be substituted into it.
  let resolved = crate::functions::plot::resolve_indirect_plot_body(
    &args[0],
    &[xvar.as_str(), yvar.as_str()],
  );
  let spec = resolved.as_ref().unwrap_or(&args[0]);
  // `{{vx, vy}, s}` takes the streamlines from `{vx, vy}` and shades the
  // background by the scalar `s` instead of by the flow speed.
  let (body, density) =
    split_vector_and_density(spec, &xvar, &yvar, x_min, y_min);
  let (svg_width, svg_height, full_width) = parse_field_options(args, 3);
  let plot_label = parse_field_plot_label(args, 3);
  let prolog = parse_overlay_primitives(args, 3, "Prolog");
  let epilog = parse_overlay_primitives(args, 3, "Epilog");

  let grid_n = 60;

  // Compute magnitude field for density background
  let mut mag_grid = vec![vec![f64::NAN; grid_n]; grid_n];
  let mut mag_samples = Vec::with_capacity(grid_n * grid_n);

  for i in 0..grid_n {
    let x = x_min + (i as f64 + 0.5) / grid_n as f64 * (x_max - x_min);
    for j in 0..grid_n {
      let y = y_min + (j as f64 + 0.5) / grid_n as f64 * (y_max - y_min);
      let value = match density {
        Some(scalar) => evaluate_scalar(scalar, &xvar, &yvar, x, y),
        None => evaluate_vector(body, &xvar, &yvar, x, y)
          .map(|(vx, vy)| (vx * vx + vy * vy).sqrt()),
      };
      if let Some(value) = value
        && value.is_finite()
      {
        mag_grid[i][j] = value;
        mag_samples.push(value);
      }
    }
  }
  let (v_min, v_max) = robust_value_range(&mag_samples);

  // Use plotters for axes, reserving room above the frame for a PlotLabel.
  let margins = field_plot_label_margins(plot_label.as_ref());
  let area = crate::functions::plot::generate_axes_only_opts(
    (x_min, x_max),
    (y_min, y_max),
    svg_width,
    svg_height,
    full_width,
    None,
    margins.as_ref(),
  )?;

  let plot_x0 = area.plot_x0;
  let plot_y0 = area.plot_y0;
  let plot_w = area.plot_w;
  let plot_h = area.plot_h;

  let mut svg = area.svg.clone();
  if let Some(pos) = svg.rfind("</svg>") {
    svg.truncate(pos);
  }
  splice_field_plot_prolog(&mut svg, &area, &prolog);
  if let Some(label) = &plot_label {
    inject_field_plot_label(&mut svg, &area, label);
  }

  // Density background: the flow speed, on Wolfram's orange-to-yellow
  // vector density gradient unless a `ColorFunction` is given.
  let density_opts = parse_density_contour_options(args, 3);
  let (color_function, cf_expr) = match density_opts.cf_expr() {
    Some(cf) => (density_opts.color_function.as_deref(), Some(cf)),
    None => (
      Some(
        density_opts
          .color_function
          .as_deref()
          .unwrap_or("WL12DefaultVectorDensityGradient"),
      ),
      None,
    ),
  };
  embed_grid_image(
    &mut svg,
    &mag_grid,
    v_min,
    v_max,
    plot_x0,
    plot_y0,
    plot_w,
    plot_h,
    color_function,
    cf_expr,
  );

  // Streamlines, drawn as in StreamPlot.
  let primitives =
    stream_primitives(body, &xvar, &yvar, (x_min, x_max), (y_min, y_max), args);
  push_stream_svg(&mut svg, &primitives, &area);

  push_frame(&mut svg, plot_x0, plot_y0, plot_w, plot_h);
  push_epilog_primitives(&mut svg, &area, &epilog);

  svg.push_str("</svg>");
  Ok(crate::graphics_result(svg))
}

/// VectorDensityPlot[{vx, vy}, {x, xmin, xmax}, {y, ymin, ymax}]
/// The vector field's arrows over a density background of its magnitude.
/// `ColorFunction` shades the background, so the arrows keep the default
/// style.
pub fn vector_density_plot_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let (xvar, x_min, x_max) = parse_iterator(&args[1], "VectorDensityPlot")?;
  let (yvar, y_min, y_max) = parse_iterator(&args[2], "VectorDensityPlot")?;
  let resolved = crate::functions::plot::resolve_indirect_plot_body(
    &args[0],
    &[xvar.as_str(), yvar.as_str()],
  );
  let body = resolved.as_ref().unwrap_or(&args[0]);
  let (svg_width, svg_height, full_width) = parse_field_options(args, 3);
  let plot_label = parse_field_plot_label(args, 3);

  let grid_n = 60;
  let mut mag_grid = vec![vec![f64::NAN; grid_n]; grid_n];
  let mut mag_samples = Vec::with_capacity(grid_n * grid_n);
  for i in 0..grid_n {
    let x = x_min + (i as f64 + 0.5) / grid_n as f64 * (x_max - x_min);
    for j in 0..grid_n {
      let y = y_min + (j as f64 + 0.5) / grid_n as f64 * (y_max - y_min);
      if let Some((vx, vy)) = evaluate_vector(body, &xvar, &yvar, x, y) {
        let mag = vx.hypot(vy);
        if mag.is_finite() {
          mag_grid[i][j] = mag;
          mag_samples.push(mag);
        }
      }
    }
  }
  let (v_min, v_max) = robust_value_range(&mag_samples);

  let grid = parse_vector_points(args, 3).map_or(VECTOR_GRID, |n| n - 1);
  let x_step = (x_max - x_min) / grid as f64;
  let y_step = (y_max - y_min) / grid as f64;
  let mut vectors = Vec::new();
  let mut max_mag = 0.0_f64;
  for i in 0..=grid {
    let x = x_min + i as f64 * x_step;
    for j in 0..=grid {
      let y = y_min + j as f64 * y_step;
      if let Some((vx, vy)) = evaluate_vector(body, &xvar, &yvar, x, y) {
        let mag = vx.hypot(vy);
        max_mag = max_mag.max(mag);
        vectors.push((x, y, vx, vy, mag));
      }
    }
  }

  let margins = field_plot_label_margins(plot_label.as_ref());
  let mut area = crate::functions::plot::generate_axes_only_opts(
    (x_min, x_max),
    (y_min, y_max),
    svg_width,
    svg_height,
    full_width,
    None,
    margins.as_ref(),
  )?;
  let mut svg = std::mem::take(&mut area.svg);
  if let Some(pos) = svg.rfind("</svg>") {
    svg.truncate(pos);
  }
  if let Some(label) = &plot_label {
    inject_field_plot_label(&mut svg, &area, label);
  }

  let density_opts = parse_density_contour_options(args, 3);
  let (color_function, cf_expr) = match density_opts.cf_expr() {
    Some(cf) => (density_opts.color_function.as_deref(), Some(cf)),
    None => (
      Some(
        density_opts
          .color_function
          .as_deref()
          .unwrap_or("WL12DefaultVectorDensityGradient"),
      ),
      None,
    ),
  };
  embed_grid_image(
    &mut svg,
    &mag_grid,
    v_min,
    v_max,
    area.plot_x0,
    area.plot_y0,
    area.plot_w,
    area.plot_h,
    color_function,
    cf_expr,
  );

  let cell_size = (area.plot_w / grid as f64).min(area.plot_h / grid as f64);
  let arrow_style = VectorArrowStyle::default();
  let primitives = render_vector_arrows(
    &vectors,
    max_mag,
    cell_size,
    &area,
    &arrow_style,
    &mut svg,
  );
  push_frame(
    &mut svg,
    area.plot_x0,
    area.plot_y0,
    area.plot_w,
    area.plot_h,
  );

  svg.push_str("</svg>");
  Ok(crate::plot_result_with_structure(
    svg,
    call1("Graphics", Expr::List(primitives.into())),
  ))
}

/// ListDensityPlot[data] - density plot from data
/// Accepts either:
///   - a matrix {{z11, z12, ...}, {z21, z22, ...}, ...}
///   - a list of {x, y, z} triples {{x1,y1,z1}, {x2,y2,z2}, ...}
pub fn list_density_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let data = evaluate_expr_to_expr(&args[0])?;
  let rows = match &data {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "ListDensityPlot: first argument must be a list of data".into(),
      ));
    }
  };

  let opts = parse_density_contour_options(args, 1);
  let nearest = interpolation_order_is_zero(&args[1..]);
  let (grid, x_min, x_max, y_min, y_max, v_min, v_max, _n_rows, _n_cols) =
    parse_list_data_to_grid(rows, "ListDensityPlot", nearest);
  let ((x_min, x_max), (y_min, y_max)) =
    matrix_data_range(rows, &args[1..], (x_min, x_max), (y_min, y_max));

  if grid.is_empty() || !v_min.is_finite() || !v_max.is_finite() {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }

  let mut area = field_plot_axes((x_min, x_max), (y_min, y_max), &opts)?;

  // Taken out of `area` so the geometry stays available for the overlays
  // appended once the plot's own content is drawn.
  let mut svg = std::mem::take(&mut area.svg);
  if let Some(pos) = svg.rfind("</svg>") {
    svg.truncate(pos);
  }

  // Coarse data grids are upsampled in-process so the shading interpolates
  // smoothly between the data points instead of showing blocky pixels.
  // `InterpolationOrder -> 0` keeps the data's cells as flat blocks.
  let grid = if grid.len() < 64 || grid[0].len() < 64 {
    if nearest {
      nearest_node_grid(&grid, 256, 256)
    } else {
      resample_node_grid(&grid, 256, 256)
    }
  } else {
    grid
  };

  embed_grid_image(
    &mut svg,
    &grid,
    v_min,
    v_max,
    area.plot_x0,
    area.plot_y0,
    area.plot_w,
    area.plot_h,
    opts.color_function.as_deref(),
    opts.cf_expr(),
  );
  if opts.frame {
    push_frame(
      &mut svg,
      area.plot_x0,
      area.plot_y0,
      area.plot_w,
      area.plot_h,
    );
  }
  push_field_plot_overlays(&mut svg, &area, &opts);

  svg.push_str("</svg>");
  Ok(crate::graphics_result(svg))
}

/// Whether the options hold `InterpolationOrder -> 0`.
fn interpolation_order_is_zero(opts: &[Expr]) -> bool {
  opts.iter().any(|opt| {
    matches!(
      opt,
      Expr::Rule { pattern, replacement }
        if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "InterpolationOrder")
          && matches!(replacement.as_ref(), Expr::Integer(0))
    )
  })
}

/// Upsample a node grid by taking the nearest node, so every data cell
/// stays one flat block.
fn nearest_node_grid(
  grid: &[Vec<f64>],
  out_cols: usize,
  out_rows: usize,
) -> Vec<Vec<f64>> {
  let cols = grid.len();
  let rows = if cols > 0 { grid[0].len() } else { 0 };
  if cols < 2 || rows < 2 {
    return grid.to_vec();
  }
  (0..out_cols)
    .map(|i| {
      let x = (i as f64 + 0.5) / out_cols as f64 * (cols - 1) as f64;
      let i0 = (x.round() as usize).min(cols - 1);
      (0..out_rows)
        .map(|j| {
          let y = (j as f64 + 0.5) / out_rows as f64 * (rows - 1) as f64;
          grid[i0][(y.round() as usize).min(rows - 1)]
        })
        .collect()
    })
    .collect()
}

/// SmoothHistogram[data] / SmoothHistogram[{data1, …}, bw, dfun, opts] —
/// the line plot of each dataset's `SmoothKernelDistribution` (bandwidth
/// `bw`, Silverman's rule by default) as its PDF, or the "CDF", "SF" or "HF"
/// named by `dfun`, from about 3.35 bandwidths below the smallest datum to as
/// many above the largest (where Wolfram's own range ends, measured on
/// several datasets; its exact cut is where the estimate becomes negligible).
pub fn smooth_histogram_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unchanged = || Ok(unevaluated("SmoothHistogram", args));
  let (positional, options): (Vec<&Expr>, Vec<&Expr>) = args
    .iter()
    .partition(|a| !matches!(a, Expr::Rule { .. } | Expr::RuleDelayed { .. }));
  let Some(Expr::List(items)) = positional.first() else {
    return unchanged();
  };
  let numbers = |items: &[Expr]| -> Option<Vec<f64>> {
    items
      .iter()
      .map(crate::functions::math_ast::try_eval_to_f64)
      .collect()
  };
  let datasets: Vec<(Expr, Vec<f64>)> =
    if items.iter().all(|i| matches!(i, Expr::List(_))) {
      let mut sets = Vec::new();
      for item in items {
        let Expr::List(inner) = item else {
          unreachable!()
        };
        match numbers(inner) {
          Some(v) if v.len() >= 2 => sets.push((item.clone(), v)),
          _ => return unchanged(),
        }
      }
      sets
    } else {
      match numbers(items) {
        Some(v) if v.len() >= 2 => vec![(positional[0].clone(), v)],
        _ => return unchanged(),
      }
    };
  if datasets.is_empty() {
    return unchanged();
  }
  let bandwidth = match positional.get(1) {
    None => None,
    Some(Expr::Identifier(a)) if a == "Automatic" => None,
    Some(e) => match crate::functions::math_ast::try_eval_to_f64(e) {
      Some(h) if h > 0.0 => Some(h),
      _ => return unchanged(),
    },
  };
  let dfun = match positional.get(2) {
    None => "PDF",
    Some(Expr::String(f)) => match f.as_str() {
      "PDF" => "PDF",
      "CDF" => "CDF",
      "SF" => "SurvivalFunction",
      "HF" => "HazardFunction",
      _ => return unchanged(),
    },
    Some(_) => return unchanged(),
  };
  let var = Expr::Identifier("SmoothHistogram$x".to_string());
  let mut lo = f64::INFINITY;
  let mut hi = f64::NEG_INFINITY;
  let mut curves = Vec::with_capacity(datasets.len());
  for (data, values) in &datasets {
    let h = bandwidth.unwrap_or_else(|| {
      crate::functions::math_ast::silverman_bandwidth(values)
    });
    let (min, max) = values
      .iter()
      .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
        (a.min(v), b.max(v))
      });
    lo = lo.min(min - 3.35 * h);
    hi = hi.max(max + 3.35 * h);
    let mut dist_args = vec![data.clone()];
    if let Some(h) = bandwidth {
      dist_args.push(Expr::Real(h));
    }
    curves.push(call(
      dfun,
      vec![call("SmoothKernelDistribution", dist_args), var.clone()],
    ));
  }
  // Not an increasing range, NaN bounds included.
  if lo.partial_cmp(&hi) != Some(std::cmp::Ordering::Less) {
    return unchanged();
  }
  let body = if curves.len() == 1 {
    curves.pop().unwrap()
  } else {
    Expr::List(curves.into())
  };
  let mut plot_args = vec![
    body,
    Expr::List(vec![var, Expr::Real(lo), Expr::Real(hi)].into()),
  ];
  plot_args.extend(options.into_iter().cloned());
  if !plot_args.iter().any(|a| {
    crate::functions::graphics::option_name_value(a)
      .is_some_and(|(name, _)| name == "PlotRange")
  }) {
    plot_args.push(Expr::Rule {
      pattern: Box::new(Expr::Identifier("PlotRange".to_string())),
      replacement: Box::new(Expr::Identifier("All".to_string())),
    });
  }
  crate::evaluator::evaluate_function_call_ast("Plot", &plot_args)
}

/// SmoothDensityHistogram[{{x1, y1}, …}, bspec, "PDF", opts] — a density
/// plot of the data's Gaussian kernel density estimate. The bandwidth is
/// Silverman's rule in each coordinate unless `bspec` gives one bandwidth
/// or `{hx, hy}`; the estimate is shaded over the data range widened by
/// three bandwidths, through ListDensityPlot with the remaining options.
pub fn smooth_density_histogram_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  smooth_density_surface("SmoothDensityHistogram", args, list_density_plot_ast)
}

/// SmoothHistogram3D[{{x1, y1}, …}, bspec, "PDF", opts] — the same kernel
/// density estimate as `SmoothDensityHistogram`, as a surface through
/// ListPlot3D.
pub fn smooth_histogram_3d_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  smooth_density_surface("SmoothHistogram3D", args, |plot_args| {
    crate::evaluator::evaluate_function_call_ast("ListPlot3D", plot_args)
  })
}

/// The Gaussian kernel density estimate of 2D data sampled on a grid, as
/// `{x, y, density}` triples handed to `render` with the remaining options.
fn smooth_density_surface(
  head: &str,
  args: &[Expr],
  render: impl Fn(&[Expr]) -> Result<Expr, InterpreterError>,
) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: head.to_string(),
      args: args.to_vec().into(),
    })
  };
  let is_option =
    |e: &Expr| matches!(e, Expr::Rule { .. } | Expr::RuleDelayed { .. });
  let positional: Vec<&Expr> =
    args.iter().take_while(|a| !is_option(a)).collect();
  let options: Vec<Expr> =
    args.iter().filter(|a| is_option(a)).cloned().collect();
  if positional.is_empty() || positional.len() > 3 {
    return unevaluated();
  }
  if let Some(dfun) = positional.get(2)
    && !matches!(dfun, Expr::String(s) if s == "PDF")
  {
    return unevaluated();
  }
  let data = evaluate_expr_to_expr(positional[0])?;
  let Expr::List(points) = &data else {
    return unevaluated();
  };
  let mut xs = Vec::with_capacity(points.len());
  let mut ys = Vec::with_capacity(points.len());
  for p in points {
    let Expr::List(c) = p else {
      return unevaluated();
    };
    let (Some(x), Some(y)) = (
      c.first().and_then(try_eval_to_f64),
      c.get(1).and_then(try_eval_to_f64),
    ) else {
      return unevaluated();
    };
    if c.len() != 2 {
      return unevaluated();
    }
    xs.push(x);
    ys.push(y);
  }
  if xs.len() < 2 {
    return unevaluated();
  }
  let silverman = |v: &[f64]| {
    let h = crate::functions::math_ast::silverman_bandwidth(v);
    if h > 0.0 && h.is_finite() { h } else { 1.0 }
  };
  let (hx, hy) = match positional.get(1).map(|b| evaluate_expr_to_expr(b)) {
    None => (silverman(&xs), silverman(&ys)),
    Some(b) => match &b? {
      Expr::Identifier(a) if a == "Automatic" => {
        (silverman(&xs), silverman(&ys))
      }
      Expr::String(a) if a == "Silverman" => (silverman(&xs), silverman(&ys)),
      Expr::List(hs) if hs.len() == 2 => {
        match (try_eval_to_f64(&hs[0]), try_eval_to_f64(&hs[1])) {
          (Some(a), Some(b)) if a > 0.0 && b > 0.0 => (a, b),
          _ => return unevaluated(),
        }
      }
      other => match try_eval_to_f64(other) {
        Some(h) if h > 0.0 => (h, h),
        _ => return unevaluated(),
      },
    },
  };
  let range = |v: &[f64], h: f64| {
    let lo = v.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (lo - 3.0 * h, hi + 3.0 * h)
  };
  let ((x0, x1), (y0, y1)) = (range(&xs, hx), range(&ys, hy));
  const SAMPLES: usize = 40;
  let norm = 1.0 / (xs.len() as f64 * 2.0 * std::f64::consts::PI * hx * hy);
  let mut triples = Vec::with_capacity(SAMPLES * SAMPLES);
  for i in 0..SAMPLES {
    let y = y0 + (y1 - y0) * i as f64 / (SAMPLES - 1) as f64;
    for j in 0..SAMPLES {
      let x = x0 + (x1 - x0) * j as f64 / (SAMPLES - 1) as f64;
      let density: f64 = xs
        .iter()
        .zip(&ys)
        .map(|(xi, yi)| {
          let (u, v) = ((x - xi) / hx, (y - yi) / hy);
          (-0.5 * (u * u + v * v)).exp()
        })
        .sum::<f64>()
        * norm;
      triples.push(Expr::List(
        vec![Expr::Real(x), Expr::Real(y), Expr::Real(density)].into(),
      ));
    }
  }
  let mut plot_args = vec![Expr::List(triples.into())];
  plot_args.extend(options);
  render(&plot_args)
}

/// ReliefPlot[array] - shaded relief of a matrix of heights: each cell is
/// coloured by its height and darkened or lightened by how its slope
/// faces the light (`LightingAngle`). The first matrix row is at the top
/// unless `DataReversed -> True`; `DataRange` places the matrix.
pub fn relief_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let data = evaluate_expr_to_expr(&args[0])?;
  let rows = match &data {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ReliefPlot".to_string(),
        args: args.to_vec().into(),
      });
    }
  };
  let opts = parse_density_contour_options(args, 1);
  let (mut grid, _, _, _, _, v_min, v_max, n_rows, n_cols) =
    parse_matrix_to_grid(rows, "ReliefPlot");
  if grid.is_empty() || !v_min.is_finite() || !v_max.is_finite() {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }

  let mut data_range = ((1.0, n_cols as f64), (1.0, n_rows as f64));
  let mut reversed = false;
  let mut lighting = Some(std::f64::consts::FRAC_PI_4 * 3.0);
  for opt in &args[1..] {
    let Expr::Rule {
      pattern,
      replacement,
    } = opt
    else {
      continue;
    };
    let Expr::Identifier(name) = pattern.as_ref() else {
      continue;
    };
    let value = evaluate_expr_to_expr(replacement)
      .unwrap_or_else(|_| replacement.as_ref().clone());
    match (name.as_str(), &value) {
      ("DataRange", Expr::List(axes)) if axes.len() == 2 => {
        if let (Expr::List(xs), Expr::List(ys)) = (&axes[0], &axes[1])
          && xs.len() == 2
          && ys.len() == 2
          && let (Some(x0), Some(x1), Some(y0), Some(y1)) = (
            try_eval_to_f64(&xs[0]),
            try_eval_to_f64(&xs[1]),
            try_eval_to_f64(&ys[0]),
            try_eval_to_f64(&ys[1]),
          )
        {
          data_range = ((x0, x1), (y0, y1));
        }
      }
      ("DataReversed", Expr::Identifier(v)) => reversed = v == "True",
      ("LightingAngle", Expr::Identifier(v)) if v == "None" => lighting = None,
      ("LightingAngle", v) => {
        if let Some(a) = try_eval_to_f64(v) {
          lighting = Some(a);
        }
      }
      _ => {}
    }
  }
  // `parse_matrix_to_grid` puts the first matrix row at the bottom.
  if !reversed {
    for col in &mut grid {
      col.reverse();
    }
  }

  let mut area = field_plot_axes(data_range.0, data_range.1, &opts)?;
  let mut svg = std::mem::take(&mut area.svg);
  if let Some(pos) = svg.rfind("</svg>") {
    svg.truncate(pos);
  }

  let orig_cols = grid.len();
  let orig_rows = grid[0].len();
  let grid = if orig_cols < 256 || orig_rows < 256 {
    resample_node_grid(&grid, 256.max(orig_cols), 256.max(orig_rows))
  } else {
    grid
  };
  // Distance between neighbouring samples in units of original cells, so
  // slopes do not depend on how finely the grid was resampled.
  let step_x = (orig_cols.max(2) - 1) as f64 / grid.len() as f64;
  let step_y = (orig_rows.max(2) - 1) as f64 / grid[0].len() as f64;

  let mut colors = cell_colors(
    &grid,
    (v_min, v_max),
    opts.color_function.as_deref(),
    opts.cf_expr(),
    Clipping::Saturate,
  );
  let plain = opts.color_function.is_none() && opts.cf_expr().is_none();
  if let Some(azimuth) = lighting {
    let elevation = std::f64::consts::FRAC_PI_4;
    let light = (
      elevation.cos() * azimuth.cos(),
      elevation.cos() * azimuth.sin(),
      elevation.sin(),
    );
    let (nc, nr) = (grid.len(), grid[0].len());
    for i in 0..nc {
      for j in 0..nr {
        let at = |a: usize, b: usize| grid[a][b];
        let (i0, i1) = (i.saturating_sub(1), (i + 1).min(nc - 1));
        let (j0, j1) = (j.saturating_sub(1), (j + 1).min(nr - 1));
        let dzdx = (at(i1, j) - at(i0, j)) / ((i1 - i0).max(1) as f64 * step_x);
        let dzdy = (at(i, j1) - at(i, j0)) / ((j1 - j0).max(1) as f64 * step_y);
        if !(dzdx.is_finite() && dzdy.is_finite()) {
          continue;
        }
        let norm = (dzdx * dzdx + dzdy * dzdy + 1.0).sqrt();
        let lit = (-dzdx * light.0 - dzdy * light.1 + light.2) / norm;
        let factor = (0.45 + 0.55 * lit / light.2).clamp(0.0, 1.5);
        if let Some(c) = &mut colors[i][j] {
          let shade = |ch: u8| {
            let v = ch as f64;
            if factor <= 1.0 {
              v * factor
            } else {
              v + (255.0 - v) * (factor - 1.0)
            }
            .round()
            .clamp(0.0, 255.0) as u8
          };
          *c = (shade(c.0), shade(c.1), shade(c.2));
        }
      }
    }
  }
  if plain && lighting.is_none() {
    for col in &mut colors {
      for c in col.iter_mut().flatten() {
        *c = (235, 235, 235);
      }
    }
  }
  embed_color_image(
    &mut svg,
    &colors,
    area.plot_x0,
    area.plot_y0,
    area.plot_w,
    area.plot_h,
  );
  if opts.frame {
    push_frame(
      &mut svg,
      area.plot_x0,
      area.plot_y0,
      area.plot_w,
      area.plot_h,
    );
  }
  push_field_plot_overlays(&mut svg, &area, &opts);
  svg.push_str("</svg>");
  Ok(crate::graphics_result(svg))
}

/// Parse list data and return a grid of z-values with axis ranges.
/// Returns (grid, x_min, x_max, y_min, y_max, v_min, v_max, n_rows, n_cols).
/// Grid is stored as grid[col][row] for marching squares compatibility.
fn parse_list_data_to_grid(
  rows: &[Expr],
  func_name: &str,
  nearest: bool,
) -> (
  std::vec::Vec<std::vec::Vec<f64>>,
  f64,
  f64,
  f64,
  f64,
  f64,
  f64,
  usize,
  usize,
) {
  // Determine if input is {x,y,z} triples or a matrix
  let is_triples = is_triples_data(rows);

  if is_triples {
    parse_triples_to_grid(rows, func_name, nearest)
  } else {
    parse_matrix_to_grid(rows, func_name)
  }
}

/// `DataRange -> {{xmin, xmax}, {ymin, ymax}}` for matrix data: the
/// coordinates the first and last columns (x) and rows (y) sit at, instead
/// of the default of 1-based indices. Triples carry their own coordinates,
/// so they and absent or malformed options keep `default_x` / `default_y`.
fn matrix_data_range(
  rows: &[Expr],
  opts: &[Expr],
  default_x: (f64, f64),
  default_y: (f64, f64),
) -> ((f64, f64), (f64, f64)) {
  if is_triples_data(rows) {
    return (default_x, default_y);
  }
  for opt in opts {
    let Expr::Rule {
      pattern,
      replacement,
    } = opt
    else {
      continue;
    };
    if !matches!(pattern.as_ref(), Expr::Identifier(n) if n == "DataRange") {
      continue;
    }
    let val = evaluate_expr_to_expr(replacement)
      .unwrap_or_else(|_| (**replacement).clone());
    if let Expr::List(axes) = &val
      && axes.len() == 2
      && let (Expr::List(xs), Expr::List(ys)) = (&axes[0], &axes[1])
      && xs.len() == 2
      && ys.len() == 2
      && let (Some(x0), Some(x1), Some(y0), Some(y1)) = (
        try_eval_to_f64(&xs[0]),
        try_eval_to_f64(&xs[1]),
        try_eval_to_f64(&ys[0]),
        try_eval_to_f64(&ys[1]),
      )
    {
      return ((x0, x1), (y0, y1));
    }
  }
  (default_x, default_y)
}

/// Check if list data represents {x,y,z} triples
fn is_triples_data(rows: &[Expr]) -> bool {
  rows.iter().all(|r| {
    if let Expr::List(items) = r
      && items.len() == 3
    {
      return items.iter().all(|item| {
        let v = evaluate_expr_to_expr(item).unwrap_or(item.clone());
        try_eval_to_f64(&v).is_some()
      });
    }
    false
  }) && rows.len() > 1
    && {
      let first_row = if let Expr::List(items) = &rows[0] {
        items
          .iter()
          .filter_map(|item| {
            let v = evaluate_expr_to_expr(item).unwrap_or(item.clone());
            try_eval_to_f64(&v)
          })
          .collect::<Vec<_>>()
      } else {
        vec![]
      };
      rows.len() > 3 || {
        let mut xs = std::collections::HashSet::new();
        let mut ys = std::collections::HashSet::new();
        for r in rows {
          if let Expr::List(items) = r {
            if let Some(x) = try_eval_to_f64(
              &evaluate_expr_to_expr(&items[0]).unwrap_or(items[0].clone()),
            ) {
              xs.insert(x.to_bits());
            }
            if let Some(y) = try_eval_to_f64(
              &evaluate_expr_to_expr(&items[1]).unwrap_or(items[1].clone()),
            ) {
              ys.insert(y.to_bits());
            }
          }
        }
        xs.len() > 1 && ys.len() > 1 && first_row[0] != 1.0
      }
    }
}

/// Parse a matrix into a grid (grid[col][row] format)
fn parse_matrix_to_grid(
  rows: &[Expr],
  _func_name: &str,
) -> (
  std::vec::Vec<std::vec::Vec<f64>>,
  f64,
  f64,
  f64,
  f64,
  f64,
  f64,
  usize,
  usize,
) {
  let mut matrix: Vec<Vec<f64>> = Vec::new();
  let mut v_min = f64::INFINITY;
  let mut v_max = f64::NEG_INFINITY;

  for row in rows {
    if let Expr::List(items) = row {
      let vals: Vec<f64> = items
        .iter()
        .map(|item| {
          let v = evaluate_expr_to_expr(item).unwrap_or(item.clone());
          try_eval_to_f64(&v).unwrap_or(f64::NAN)
        })
        .collect();
      for &v in &vals {
        if v.is_finite() {
          v_min = v_min.min(v);
          v_max = v_max.max(v);
        }
      }
      matrix.push(vals);
    }
  }

  if matrix.is_empty() {
    return (vec![], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0);
  }

  let n_rows = matrix.len();
  let n_cols = matrix.iter().map(std::vec::Vec::len).max().unwrap_or(1);

  // Convert to grid[col][row] format (row 0 at the bottom). Matrix row i
  // sits at y = i + 1, i.e. the first data row is at the bottom, matching
  // Mathematica's ListDensityPlot / ListContourPlot orientation.
  let mut grid = vec![vec![f64::NAN; n_rows]; n_cols];
  for (i, row) in matrix.iter().enumerate() {
    for (j, &val) in row.iter().enumerate() {
      if j < n_cols {
        grid[j][i] = val;
      }
    }
  }

  (
    grid,
    1.0,
    n_cols as f64,
    1.0,
    n_rows as f64,
    v_min,
    v_max,
    n_rows,
    n_cols,
  )
}

/// Parse {x,y,z} triples into a grid using inverse distance weighting
fn parse_triples_to_grid(
  rows: &[Expr],
  _func_name: &str,
  nearest: bool,
) -> (
  std::vec::Vec<std::vec::Vec<f64>>,
  f64,
  f64,
  f64,
  f64,
  f64,
  f64,
  usize,
  usize,
) {
  let mut points: Vec<(f64, f64, f64)> = Vec::new();

  for row in rows {
    if let Expr::List(items) = row
      && items.len() == 3
    {
      let x = try_eval_to_f64(
        &evaluate_expr_to_expr(&items[0]).unwrap_or(items[0].clone()),
      );
      let y = try_eval_to_f64(
        &evaluate_expr_to_expr(&items[1]).unwrap_or(items[1].clone()),
      );
      let z = try_eval_to_f64(
        &evaluate_expr_to_expr(&items[2]).unwrap_or(items[2].clone()),
      );
      if let (Some(x), Some(y), Some(z)) = (x, y, z)
        && x.is_finite()
        && y.is_finite()
        && z.is_finite()
      {
        points.push((x, y, z));
      }
    }
  }

  if points.is_empty() {
    return (vec![], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0);
  }

  let x_min = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
  let x_max = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
  let y_min = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
  let y_max = points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
  let z_min = points.iter().map(|p| p.2).fold(f64::INFINITY, f64::min);
  let z_max = points.iter().map(|p| p.2).fold(f64::NEG_INFINITY, f64::max);

  // Nearest-sample shading needs a finer grid for sharp cell borders.
  let grid_n = if nearest { 256 } else { FIELD_GRID };
  let x_range = x_max - x_min;
  let y_range = y_max - y_min;

  // grid[col][row] with IDW interpolation, sampled at nodes so the
  // outermost samples land exactly on the data range bounds
  let mut grid = vec![vec![f64::NAN; grid_n]; grid_n];
  for i in 0..grid_n {
    let gx = x_min + i as f64 / (grid_n - 1) as f64 * x_range;
    for j in 0..grid_n {
      let gy = y_min + j as f64 / (grid_n - 1) as f64 * y_range;

      if nearest {
        let mut best = (f64::INFINITY, f64::NAN);
        for &(px, py, pz) in &points {
          let d = (gx - px).powi(2) + (gy - py).powi(2);
          if d < best.0 {
            best = (d, pz);
          }
        }
        grid[i][j] = best.1;
        continue;
      }

      let mut w_sum = 0.0;
      let mut z_sum = 0.0;
      let mut exact = None;

      for &(px, py, pz) in &points {
        let dx = gx - px;
        let dy = gy - py;
        let dist_sq = dx * dx + dy * dy;
        if dist_sq < 1e-20 {
          exact = Some(pz);
          break;
        }
        let w = 1.0 / dist_sq;
        w_sum += w;
        z_sum += w * pz;
      }

      grid[i][j] = exact.unwrap_or_else(|| z_sum / w_sum);
    }
  }

  (
    grid, x_min, x_max, y_min, y_max, z_min, z_max, grid_n, grid_n,
  )
}

/// ListContourPlot[data] - contour plot from data
/// Accepts same data formats as ListDensityPlot.
pub fn list_contour_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let data = evaluate_expr_to_expr(&args[0])?;
  let rows = match &data {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "ListContourPlot: first argument must be a list of data".into(),
      ));
    }
  };

  let opts = parse_density_contour_options(args, 1);
  let (grid, x_min, x_max, y_min, y_max, v_min, v_max, _n_rows, _n_cols) =
    parse_list_data_to_grid(rows, "ListContourPlot", false);
  let ((x_min, x_max), (y_min, y_max)) =
    matrix_data_range(rows, &args[1..], (x_min, x_max), (y_min, y_max));

  if grid.is_empty() || !v_min.is_finite() || !v_max.is_finite() {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }

  let mut area = field_plot_axes((x_min, x_max), (y_min, y_max), &opts)?;

  // Taken out of `area` so the geometry stays available for the overlays
  // appended once the plot's own content is drawn.
  let mut svg = std::mem::take(&mut area.svg);
  if let Some(pos) = svg.rfind("</svg>") {
    svg.truncate(pos);
  }

  // The grid holds node values: n columns of n rows span n-1 cells each way.
  let grid_cols = grid.len();
  let grid_rows = grid[0].len();
  let cell_w = area.plot_w / (grid_cols.max(2) - 1) as f64;
  let cell_h = area.plot_h / (grid_rows.max(2) - 1) as f64;

  let levels = resolve_contour_levels(&opts.contours, v_min, v_max);

  if opts.contour_shading {
    render_contour_bands(
      &mut svg,
      &grid,
      &levels,
      v_min,
      v_max,
      area.plot_x0,
      area.plot_y0,
      cell_w,
      cell_h,
      opts.color_function.as_deref(),
      opts.cf_expr(),
    );
  }
  let (contour_color, contour_weight) = opts.contour_stroke();
  render_contour_lines_styled(
    &mut svg,
    &grid,
    &levels,
    area.plot_x0,
    area.plot_y0,
    cell_w,
    cell_h,
    area.render_width,
    &contour_color,
    contour_weight,
    "",
  );
  if opts.frame {
    push_frame(
      &mut svg,
      area.plot_x0,
      area.plot_y0,
      area.plot_w,
      area.plot_h,
    );
  }
  push_field_plot_overlays(&mut svg, &area, &opts);

  svg.push_str("</svg>");
  Ok(crate::graphics_result(svg))
}

/// A cell in an ArrayPlot matrix: either a numeric value or an explicit color.
#[derive(Clone)]
enum ArrayCell {
  // The evaluated cell value, kept alongside its `f64` approximation so
  // `ColorRules` can match it exactly (e.g. `I`/`-I`, which have no
  // real-valued `f64` form and would otherwise collapse to `0.0`).
  Value(f64, Expr),
  Color(GfxColor),
}

/// The gradient scheme name of a `ColorFunction` option value: a bare
/// string (`"TemperatureMap"`), a `ColorData["TemperatureMap"]` call, or
/// the structured `ColorDataFunction["TemperatureMap", …]` form the call
/// evaluates to. Shared with `ListPlot3D` (`plot3d.rs`), which resolves its
/// own `ColorFunction` option the same way rather than re-parsing it.
pub(crate) fn color_function_scheme_name(val: &Expr) -> Option<String> {
  match val {
    // `"Thermometer"` is the short spelling Wolfram accepts for
    // `"ThermometerColors"` in a `ColorFunction` option.
    Expr::String(s) if s == "Thermometer" => Some("ThermometerColors".into()),
    Expr::String(s) => Some(s.clone()),
    // Only the plain gradient: a ranged or reversed `ColorDataFunction`
    // (`ColorData[{"Rainbow", "Reverse"}]`) carries a different blend and
    // is applied per value instead.
    Expr::FunctionCall { name, args }
      if name == "ColorData" && args.len() == 1 =>
    {
      match &args[0] {
        Expr::String(s) => Some(s.clone()),
        _ => None,
      }
    }
    Expr::FunctionCall { name, args }
      if name == "ColorDataFunction" && args.len() == 4 =>
    {
      match (&args[0], &args[2], &args[3]) {
        (Expr::String(s), Expr::List(r), Expr::Function { body })
          if matches!(r.as_slice(), [Expr::Integer(0), Expr::Integer(1)])
            && matches!(body.as_ref(), Expr::FunctionCall { name: b, args: ba }
              if b == "Blend" && matches!(ba.as_slice(), [_, Expr::Slot(1)])) =>
        {
          Some(s.clone())
        }
        _ => None,
      }
    }
    _ => None,
  }
}

/// Apply a named color function (gradient) to a normalized value t in [0,1].
/// Schemes with stored `ColorData` control points (see
/// `chart::sample_named_gradient`) interpolate those — matching
/// wolframscript exactly; the rest fall back to analytic approximations.
/// Shared with `ListPlot3D` (`plot3d.rs`), which colors its surface by the
/// same named gradients instead of duplicating this table.
pub(crate) fn apply_named_color_function(name: &str, t: f64) -> (u8, u8, u8) {
  let t = t.clamp(0.0, 1.0);
  if let Some((r, g, b)) =
    crate::functions::chart::sample_named_gradient(name, t)
  {
    return (
      (r * 255.0).round() as u8,
      (g * 255.0).round() as u8,
      (b * 255.0).round() as u8,
    );
  }
  match name {
    "WL12DefaultVectorGradient" => {
      crate::functions::plot::wl_default_vector_gradient(t)
    }
    "WL12DefaultVectorDensityGradient" => {
      crate::functions::plot::wl_default_vector_density_gradient(t)
    }
    // The gradient `DensityPlot` shades with when given no `ColorFunction`.
    "WL14DefaultDensityGradient" => density_gradient(t),
    "Rainbow" => {
      // Hue-based rainbow: red -> yellow -> green -> cyan -> blue -> violet
      let c = GfxColor::from_hue(t * 0.83, 1.0, 1.0);
      (
        (c.r * 255.0).round() as u8,
        (c.g * 255.0).round() as u8,
        (c.b * 255.0).round() as u8,
      )
    }
    "TemperatureMap" => {
      // Blue -> white -> red
      value_to_color(t, 0.0, 1.0)
    }
    "SunsetColors" => {
      // Dark blue -> red -> yellow -> white
      let (r, g, b) = if t < 0.33 {
        let s = t / 0.33;
        (0.1 + s * 0.7, 0.05 + s * 0.0, 0.3 + s * (-0.2))
      } else if t < 0.66 {
        let s = (t - 0.33) / 0.33;
        (0.8 + s * 0.2, 0.05 + s * 0.55, 0.1 + s * 0.0)
      } else {
        let s = (t - 0.66) / 0.34;
        (1.0, 0.6 + s * 0.4, 0.1 + s * 0.9)
      };
      (
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8,
      )
    }
    "ThermometerColors" => {
      // Blue -> cyan -> green -> yellow -> red
      let (r, g, b) = if t < 0.25 {
        let s = t / 0.25;
        (0.0, s, 1.0)
      } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        (0.0, 1.0, 1.0 - s)
      } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        (s, 1.0, 0.0)
      } else {
        let s = (t - 0.75) / 0.25;
        (1.0, 1.0 - s, 0.0)
      };
      (
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8,
      )
    }
    "GreenPinkTones" => {
      // Green -> white -> pink
      if t < 0.5 {
        let s = t * 2.0;
        (
          (s * 255.0).round() as u8,
          ((0.5 + 0.5 * s) * 255.0).round() as u8,
          (s * 255.0).round() as u8,
        )
      } else {
        let s = (t - 0.5) * 2.0;
        (
          255,
          ((1.0 - 0.5 * s) * 255.0).round() as u8,
          ((1.0 - 0.5 * s) * 255.0).round() as u8,
        )
      }
    }
    // Default: grayscale (same as "GrayTones")
    _ => {
      let gray = ((1.0 - t) * 255.0).round() as u8;
      (gray, gray, gray)
    }
  }
}

/// The rows `ArrayPlot` draws from `data`: SparseArrays (the whole array
/// or its rows) densified. `None` when `data` is not a list of lists.
fn array_plot_matrix(data: &Expr) -> Option<Vec<Expr>> {
  let densify = |e: &Expr| {
    crate::functions::list_helpers_ast::densify_sparse_array(e)
      .unwrap_or_else(|| e.clone())
  };
  let dense = densify(data);
  let Expr::List(rows) = &dense else {
    return None;
  };
  // A list of SparseArray rows (possibly of differing lengths) plots as the
  // ragged list of their dense forms.
  let rows: Vec<Expr> = rows.iter().map(densify).collect();
  rows
    .iter()
    .all(|r| matches!(r, Expr::List(_)))
    .then_some(rows)
}

/// `ArrayPlot::mat` for a first argument that is not a list of lists, which
/// leaves the call unevaluated. Checked outside the plot's quiet scope so
/// the message is seen.
pub fn array_plot_argument_error(args: &[Expr]) -> Option<Expr> {
  let data = args.first()?;
  if array_plot_matrix(data).is_some() {
    return None;
  }
  crate::emit_message(&format!(
    "ArrayPlot::mat: Argument {} at position 1 is not a list of lists.",
    crate::syntax::format_expr(data, crate::syntax::ExprForm::Output)
  ));
  Some(crate::helpers::unevaluated("ArrayPlot", args))
}

/// ArrayPlot[{{v11, ...}, ...}] - grayscale grid from matrix values
/// 0 is always white; the maximum value is black.
/// Supports options: ColorRules, Mesh, ColorFunction, ImageSize, FrameLabel,
/// Epilog.
/// Cells can also be explicit color directives (e.g. Pink, Red).
pub fn array_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let args = &crate::functions::graphics::splice_option_lists(args)[..];
  let data = evaluate_expr_to_expr(&args[0])?;
  let Some(rows) = array_plot_matrix(&data) else {
    return Ok(crate::helpers::unevaluated("ArrayPlot", args));
  };

  // Parse options
  // Kept as a rules list (rather than pre-resolved to `f64`) so a cell can
  // be matched by `Replace` semantics — exact values like `I`/`-I` (which
  // have no `f64` form) and patterns alike, not just real numbers.
  let mut color_rules: Option<Expr> = None;
  let mut mesh = false;
  // `MeshStyle` / `FrameStyle`: colors of the cell grid and of the border.
  let mut mesh_style_color: Option<RGBColor> = None;
  let mut frame_style_color: Option<RGBColor> = None;
  let mut color_function: Option<String> = None;
  // `ColorFunction -> f` for an `f` that isn't a recognized named gradient
  // (a pure function, a user-defined symbol, …): applied per-cell via the
  // evaluator, same as `RegionFunction` in `region_allows` above.
  let mut color_function_expr: Option<Expr> = None;
  // `ColorFunctionScaling -> False` passes each cell's raw value to
  // `color_function_expr`; the default (`True`) passes the value rescaled
  // to the 0..1 range of the matrix, matching `Options[ArrayPlot]`.
  let mut color_function_scaling = true;
  let mut frame_labels = crate::functions::plot::FrameLabels::default();
  // `Epilog -> {…}`: graphics primitives drawn over the finished grid, in
  // data coordinates where a cell at 1-indexed (row, col) spans
  // `{col - 1, col} x {n_rows - row, n_rows - row + 1}` — row 1 at the top,
  // column 1 at the left, matching `ArrayPlot`'s own `PlotRange`.
  let mut epilog: Vec<Expr> = Vec::new();
  // `DataRange -> {{xmin, xmax}, {ymin, ymax}}`: the coordinates the grid
  // spans (default: cell indices). It only places the grid in the `Raster`
  // that `First`/`Part` expose; the standalone picture is unaffected.
  let mut data_range: Option<((f64, f64), (f64, f64))> = None;
  // `DataReversed -> True`: first row at the bottom instead of the top.
  let mut data_reversed = false;
  // `AspectRatio -> r` (height / width) overrides the default of matching
  // the matrix shape.
  let mut aspect_ratio: Option<f64> = None;
  // `PlotRangePadding -> …`: extra room (in cell units) around the grid;
  // `None` by default, unlike other plots.
  let mut range_padding: Option<
    crate::functions::graphics::PlotRangePaddingSpec,
  > = None;
  // `FrameTicks -> {{left, right}, {bottom, top}}` with explicit
  // `{position, label}` ticks, and the `FrameTicksStyle` they default to.
  let mut frame_ticks = FrameTickLabels::default();
  let mut tick_style = TickStyle::default();

  for opt in &args[1..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && let Expr::Identifier(name) = pattern.as_ref()
    {
      match name.as_str() {
        "DataRange" => {
          let val = evaluate_expr_to_expr(replacement)
            .unwrap_or_else(|_| (**replacement).clone());
          if let Expr::List(axes) = &val
            && axes.len() == 2
            && let (Expr::List(xs), Expr::List(ys)) = (&axes[0], &axes[1])
            && xs.len() == 2
            && ys.len() == 2
            && let (Some(x0), Some(x1), Some(y0), Some(y1)) = (
              try_eval_to_f64(&xs[0]),
              try_eval_to_f64(&xs[1]),
              try_eval_to_f64(&ys[0]),
              try_eval_to_f64(&ys[1]),
            )
          {
            data_range = Some(((x0, x1), (y0, y1)));
          }
        }
        "AspectRatio" => {
          aspect_ratio = try_eval_to_f64(replacement).filter(|r| *r > 0.0);
        }
        "DataReversed" => {
          if let Expr::Identifier(v) = replacement.as_ref()
            && v == "True"
          {
            data_reversed = true;
          }
        }
        "ColorRules" => {
          color_rules = Some(
            evaluate_expr_to_expr(replacement)
              .unwrap_or_else(|_| replacement.as_ref().clone()),
          );
        }
        "Mesh" => match replacement.as_ref() {
          Expr::Identifier(v) if v == "True" || v == "All" => mesh = true,
          _ => {}
        },
        "MeshStyle" => {
          let val = evaluate_expr_to_expr(replacement)
            .unwrap_or_else(|_| (**replacement).clone());
          mesh_style_color = style_rgb(&val);
        }
        "FrameStyle" => {
          let val = evaluate_expr_to_expr(replacement)
            .unwrap_or_else(|_| (**replacement).clone());
          frame_style_color = style_rgb(&val);
        }
        "ColorFunction" => {
          if let Some(s) = color_function_scheme_name(replacement.as_ref()) {
            color_function = Some(s);
          } else if !matches!(replacement.as_ref(), Expr::Identifier(v) if v == "Automatic" || v == "None")
          {
            color_function_expr = Some(replacement.as_ref().clone());
          }
        }
        "ColorFunctionScaling" => {
          if let Expr::Identifier(v) = replacement.as_ref()
            && v == "False"
          {
            color_function_scaling = false;
          }
        }
        // An array plot is drawn inside a frame, so it takes the same
        // `FrameLabel` a framed plot does — on any of the four edges.
        "FrameLabel" => {
          frame_labels = crate::functions::plot::parse_frame_label(replacement);
        }
        "FrameTicks" => {
          let val = evaluate_expr_to_expr(replacement)
            .unwrap_or_else(|_| (**replacement).clone());
          frame_ticks = parse_frame_tick_labels(&val);
        }
        "FrameTicksStyle" => {
          let val = evaluate_expr_to_expr(replacement)
            .unwrap_or_else(|_| (**replacement).clone());
          tick_style = parse_tick_style(&val, TickStyle::default());
        }
        "PlotRangePadding" => {
          let val = evaluate_expr_to_expr(replacement)
            .unwrap_or_else(|_| (**replacement).clone());
          range_padding =
            Some(crate::functions::graphics::parse_plot_range_padding(&val));
        }
        "Epilog" => {
          let val = evaluate_expr_to_expr(replacement)
            .unwrap_or_else(|_| (**replacement).clone());
          epilog = match val {
            Expr::List(ref items) => items.to_vec(),
            Expr::Identifier(ref s) if s == "None" => Vec::new(),
            other => vec![other],
          };
        }
        _ => {}
      }
    }
  }

  // Parse matrix: each cell is either a numeric value or a color directive
  let mut matrix: Vec<Vec<ArrayCell>> = Vec::new();
  let mut v_min = f64::INFINITY;
  let mut v_max = f64::NEG_INFINITY;

  for row in &rows {
    if let Expr::List(items) = row {
      let cells: Vec<ArrayCell> = items
        .iter()
        .map(|item| {
          let v = evaluate_expr_to_expr(item).unwrap_or(item.clone());
          // Try as color first
          if let Some(color) = parse_color(&v) {
            ArrayCell::Color(color)
          } else {
            let f = try_eval_to_f64(&v).unwrap_or(0.0);
            ArrayCell::Value(f, v)
          }
        })
        .collect();
      for cell in &cells {
        if let ArrayCell::Value(v, _) = cell
          && v.is_finite()
        {
          v_min = v_min.min(*v);
          v_max = v_max.max(*v);
        }
      }
      matrix.push(cells);
    }
  }

  if matrix.is_empty() {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }

  let (svg_width, svg_height, full_width) = parse_field_options(args, 1);
  let n_rows = matrix.len();
  let n_cols = matrix.iter().map(std::vec::Vec::len).max().unwrap_or(1);

  // Padding in cell units; `Automatic` means none for an array plot.
  let ((pad_l, pad_r), (pad_b, pad_t)) = match &range_padding {
    Some(spec) => (
      spec.x.resolve(n_cols as f64, 0.0),
      spec.y.resolve(n_rows as f64, 0.0),
    ),
    None => ((0.0, 0.0), (0.0, 0.0)),
  };
  let total_cols = n_cols as f64 + pad_l + pad_r;
  let total_rows = n_rows as f64 + pad_b + pad_t;

  // Aspect ratio: match the matrix shape (wider if more cols, taller if more
  // rows), fitted inside the `ImageSize` box — an explicit `{w, h}` is a
  // bounding box, not the picture's own size.
  let aspect = aspect_ratio.unwrap_or(total_rows / total_cols);
  let (svg_width, svg_height) = {
    let (w, h) = (svg_width as f64, svg_height as f64);
    let (fit_w, fit_h) = if h / w <= aspect {
      (h / aspect, h)
    } else {
      (w, w * aspect)
    };
    ((fit_w.round() as u32).max(1), (fit_h.round() as u32).max(1))
  };

  let s = RESOLUTION_SCALE;
  let render_width = svg_width * s;
  let render_height = svg_height * s;

  let (bg_color, _, _, _, _) = plot_theme();
  let border_color = frame_style_color.unwrap_or(RGBColor(0x66, 0x66, 0x66));

  let mut cell_rgb: Vec<Vec<(u8, u8, u8)>> = Vec::with_capacity(n_rows);
  let mut buf = String::new();
  {
    let root = SVGBackend::with_string(&mut buf, (render_width, render_height))
      .into_drawing_area();
    root.fill(&bg_color).map_err(|e| {
      InterpreterError::EvaluationError(format!("ArrayPlot: {e}"))
    })?;

    let cell_w = render_width as f64 / total_cols;
    let cell_h = render_height as f64 / total_rows;
    let off_x = pad_l * cell_w;
    let off_y = pad_t * cell_h;

    for (i, row) in matrix.iter().enumerate() {
      let mut rgb_row = Vec::with_capacity(row.len());
      for (j, cell) in row.iter().enumerate() {
        let (r, g, b) = match cell {
          ArrayCell::Color(color) => (
            (color.r.clamp(0.0, 1.0) * 255.0).round() as u8,
            (color.g.clamp(0.0, 1.0) * 255.0).round() as u8,
            (color.b.clamp(0.0, 1.0) * 255.0).round() as u8,
          ),
          ArrayCell::Value(val, cell_expr) => {
            // Check ColorRules first — matched with the same `Replace`
            // semantics as `/.` (exact value or pattern), not `f64`
            // equality, so non-real cell values (`I`, `-I`, …) match too.
            let found = color_rules.as_ref().and_then(|rules| {
              crate::evaluator::pattern_matching::apply_replace_all_ast(
                cell_expr, rules,
              )
              .ok()
              .and_then(|replaced| parse_color(&replaced))
              .map(|rule_color| {
                (
                  (rule_color.r.clamp(0.0, 1.0) * 255.0).round() as u8,
                  (rule_color.g.clamp(0.0, 1.0) * 255.0).round() as u8,
                  (rule_color.b.clamp(0.0, 1.0) * 255.0).round() as u8,
                )
              })
            });
            found.unwrap_or_else(|| {
              let range = v_max - v_min;
              let t = if range.abs() < f64::EPSILON {
                0.5
              } else {
                ((*val - v_min) / range).clamp(0.0, 1.0)
              };
              if let Some(ref cf_name) = color_function {
                apply_named_color_function(cf_name, t)
              } else if let Some(ref cf_expr) = color_function_expr {
                // Raw (unscaled) values commonly come from exact integer
                // literals (e.g. packed 24-bit colors); passing them back
                // as a `Real` would strip that integer-ness and make
                // functions like `IntegerDigits` inside the color function
                // stay unevaluated. The scaled value has no such origin,
                // so it always becomes a `Real`.
                let arg_expr = if color_function_scaling {
                  Expr::Real(t)
                } else if val.fract() == 0.0 && val.abs() < 1e15 {
                  Expr::Integer(*val as i128)
                } else {
                  Expr::Real(*val)
                };
                let call = Expr::CurriedCall {
                  func: Box::new(cf_expr.clone()),
                  args: vec![arg_expr],
                };
                evaluate_expr_to_expr(&call)
                  .ok()
                  .and_then(|r| parse_color(&r))
                  .map_or_else(
                    || {
                      let gray = ((1.0 - t) * 255.0).round() as u8;
                      (gray, gray, gray)
                    },
                    |c| {
                      (
                        (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
                        (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
                        (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
                      )
                    },
                  )
              } else {
                let gray = ((1.0 - t) * 255.0).round() as u8;
                (gray, gray, gray)
              }
            })
          }
        };

        rgb_row.push((r, g, b));
        let vi = if data_reversed { n_rows - 1 - i } else { i };
        let x0 = (off_x + j as f64 * cell_w).round() as i32;
        let y0 = (off_y + vi as f64 * cell_h).round() as i32;
        let x1 = (off_x + (j + 1) as f64 * cell_w).round() as i32;
        let y1 = (off_y + (vi + 1) as f64 * cell_h).round() as i32;

        root
          .draw(&Rectangle::new(
            [(x0, y0), (x1, y1)],
            RGBColor(r, g, b).filled(),
          ))
          .map_err(|e| {
            InterpreterError::EvaluationError(format!("ArrayPlot: {e}"))
          })?;
      }
      cell_rgb.push(rgb_row);
    }

    // Draw mesh grid lines between cells
    if mesh {
      let mesh_color = mesh_style_color.unwrap_or(RGBColor(0x66, 0x66, 0x66));
      let line_w = (s as i32).max(1);
      // Vertical lines
      for j in 1..n_cols {
        let x = (off_x + j as f64 * cell_w).round() as i32;
        root
          .draw(&Rectangle::new(
            [
              (x - line_w / 2, off_y.round() as i32),
              (
                x + (line_w + 1) / 2,
                (off_y + n_rows as f64 * cell_h).round() as i32,
              ),
            ],
            mesh_color.filled(),
          ))
          .map_err(|e| {
            InterpreterError::EvaluationError(format!("ArrayPlot: {e}"))
          })?;
      }
      // Horizontal lines
      for i in 1..n_rows {
        let y = (off_y + i as f64 * cell_h).round() as i32;
        root
          .draw(&Rectangle::new(
            [
              (off_x.round() as i32, y - line_w / 2),
              (
                (off_x + n_cols as f64 * cell_w).round() as i32,
                y + (line_w + 1) / 2,
              ),
            ],
            mesh_color.filled(),
          ))
          .map_err(|e| {
            InterpreterError::EvaluationError(format!("ArrayPlot: {e}"))
          })?;
      }
    }

    // Draw border around the entire plot
    let bw = s as i32;
    let rw = render_width as i32;
    let rh = render_height as i32;
    for rect in [
      [(0, 0), (rw, bw)],       // top
      [(0, rh - bw), (rw, rh)], // bottom
      [(0, 0), (bw, rh)],       // left
      [(rw - bw, 0), (rw, rh)], // right
    ] {
      root
        .draw(&Rectangle::new(rect, border_color.filled()))
        .map_err(|e| {
          InterpreterError::EvaluationError(format!("ArrayPlot: {e}"))
        })?;
    }

    root.present().map_err(|e| {
      InterpreterError::EvaluationError(format!("ArrayPlot: {e}"))
    })?;
  }

  if !epilog.is_empty()
    && let Some(pos) = buf.rfind("</svg>")
  {
    buf.truncate(pos);
    let epilog_area = crate::functions::plot_epilog::PlotArea {
      x0: 0.0,
      y0: 0.0,
      w: render_width as f64,
      h: render_height as f64,
      x_min: -pad_l,
      x_max: n_cols as f64 + pad_r,
      y_min: -pad_b,
      y_max: n_rows as f64 + pad_t,
      scale: RESOLUTION_SCALE as f64,
    };
    buf.push_str(&crate::functions::plot_epilog::render_epilog_svg(
      &epilog,
      &epilog_area,
      "arrayplot",
    ));
    buf.push_str("</svg>");
  }

  rewrite_svg_header(
    &mut buf,
    svg_width,
    svg_height,
    render_width,
    render_height,
    full_width,
  );

  let (buf, svg_width, svg_height) = frame_ticked_svg(
    &buf,
    svg_width,
    svg_height,
    (n_cols, n_rows),
    &frame_ticks,
    &tick_style,
  );
  let buf = frame_labelled_svg(&buf, svg_width, svg_height, &frame_labels);
  let buf = match parse_field_plot_label(args, 1) {
    Some(label) => with_plot_label_above(&buf, &label),
    None => buf,
  };

  // The symbolic content, as `First`/`Part` see it: a `Raster` of the cell
  // colors with the first row at the top (`Raster` rows run bottom-up, so a
  // default plot lists them reversed) over the `DataRange` rectangle.
  // `DataRange` places the cell *centers*, so the raster reaches half a
  // cell beyond it (a single cell spans the whole range); without one the
  // cells sit on the integer grid {{0, 0}, {cols, rows}}.
  let rect = match data_range {
    Some(((x0, x1), (y0, y1))) => {
      let widen = |lo: f64, hi: f64, n: usize| {
        if n > 1 {
          let half = (hi - lo) / (n - 1) as f64 / 2.0;
          (lo - half, hi + half)
        } else {
          (lo, hi)
        }
      };
      let (x0, x1) = widen(x0, x1, n_cols);
      let (y0, y1) = widen(y0, y1, n_rows);
      Expr::List(
        vec![
          Expr::List(vec![Expr::Real(x0), Expr::Real(y0)].into()),
          Expr::List(vec![Expr::Real(x1), Expr::Real(y1)].into()),
        ]
        .into(),
      )
    }
    None => Expr::List(
      vec![
        Expr::List(vec![Expr::Integer(0), Expr::Integer(0)].into()),
        Expr::List(
          vec![Expr::Integer(n_cols as i128), Expr::Integer(n_rows as i128)]
            .into(),
        ),
      ]
      .into(),
    ),
  };
  let channel = |v: u8| Expr::Real(f64::from(v) / 255.0);
  let mut raster_rows: Vec<Expr> = cell_rgb
    .iter()
    .map(|row| {
      Expr::List(
        row
          .iter()
          .map(|&(r, g, b)| {
            Expr::List(vec![channel(r), channel(g), channel(b)].into())
          })
          .collect::<Vec<_>>()
          .into(),
      )
    })
    .collect();
  if !data_reversed {
    raster_rows.reverse();
  }
  let raster = call("Raster", vec![Expr::List(raster_rows.into()), rect]);
  // What `Show` redraws when it layers other graphics over the plot: the
  // mesh lines as primitives, and the explicit frame ticks as a framed
  // plot's options (ticks are in `ArrayPlot` coordinates — the center of
  // column `j` is `j`, of row `i` from the top is `i` — which sit half a
  // cell inside the raster's own coordinates).
  let mut prims = vec![raster];
  if mesh {
    let mesh_color = mesh_style_color.unwrap_or(RGBColor(0x66, 0x66, 0x66));
    let channel = |v: u8| Expr::Real(f64::from(v) / 255.0);
    prims.push(call(
      "RGBColor",
      vec![
        channel(mesh_color.0),
        channel(mesh_color.1),
        channel(mesh_color.2),
      ],
    ));
    let point = |x: usize, y: usize| {
      Expr::List(
        vec![Expr::Integer(x as i128), Expr::Integer(y as i128)].into(),
      )
    };
    let segment =
      |a: Expr, b: Expr| call1("Line", Expr::List(vec![a, b].into()));
    for j in 1..n_cols {
      prims.push(segment(point(j, 0), point(j, n_rows)));
    }
    for i in 1..n_rows {
      prims.push(segment(point(0, i), point(n_cols, i)));
    }
  }
  let content = if prims.len() == 1 {
    prims.remove(0)
  } else {
    Expr::List(prims.into())
  };
  let mut structure_args = vec![content];
  let ticks_present = [
    &frame_ticks.left,
    &frame_ticks.right,
    &frame_ticks.bottom,
    &frame_ticks.top,
  ]
  .iter()
  .any(|edge| !edge.is_empty());
  if ticks_present {
    let rule = |name: &str, value: Expr| Expr::Rule {
      pattern: Box::new(Expr::Identifier(name.to_string())),
      replacement: Box::new(value),
    };
    let edge = |ticks: &[FrameTickLabel], horizontal: bool| {
      if ticks.is_empty() {
        return Expr::Identifier("None".to_string());
      }
      Expr::List(
        ticks
          .iter()
          .map(|t| {
            let pos = if horizontal {
              t.pos - 0.5
            } else {
              n_rows as f64 - t.pos + 0.5
            };
            Expr::List(
              vec![Expr::Real(pos), Expr::String(t.text.clone())].into(),
            )
          })
          .collect::<Vec<_>>()
          .into(),
      )
    };
    structure_args.push(rule("Frame", Expr::Identifier("True".to_string())));
    structure_args.push(rule(
      "PlotRangePadding",
      Expr::Identifier("None".to_string()),
    ));
    structure_args.push(rule(
      "FrameTicks",
      Expr::List(
        vec![
          Expr::List(
            vec![
              edge(&frame_ticks.left, false),
              edge(&frame_ticks.right, false),
            ]
            .into(),
          ),
          Expr::List(
            vec![
              edge(&frame_ticks.bottom, true),
              edge(&frame_ticks.top, true),
            ]
            .into(),
          ),
        ]
        .into(),
      ),
    ));
  }
  // The sizing options travel with the symbolic form, so `Show[ArrayPlot[…],
  // …]` redraws the grid at the size and shape it was plotted with.
  for opt in &args[1..] {
    if let Expr::Rule { pattern, .. } = opt
      && matches!(pattern.as_ref(), Expr::Identifier(n)
        if n == "ImageSize" || n == "AspectRatio")
    {
      structure_args.push(opt.clone());
    }
  }
  let structure = call("Graphics", structure_args);
  Ok(crate::graphics_result_with_structure(buf, structure))
}

/// One explicit frame tick: a position in the plot's coordinates, its label
/// text, and the label's own `Style` overrides.
struct FrameTickLabel {
  pos: f64,
  text: String,
  style: TickStyle,
}

#[derive(Clone, Copy, Default)]
struct TickStyle {
  size: Option<f64>,
  color: Option<(u8, u8, u8)>,
}

#[derive(Default)]
struct FrameTickLabels {
  left: Vec<FrameTickLabel>,
  right: Vec<FrameTickLabel>,
  bottom: Vec<FrameTickLabel>,
  top: Vec<FrameTickLabel>,
}

/// Fold a number (font size) and color directives into `base`.
/// The color a style spec (`White`, `{Red, Thick}`, `Directive[…]`) sets.
fn style_rgb(value: &Expr) -> Option<RGBColor> {
  let items: Vec<&Expr> = match value {
    Expr::List(items) => items.iter().collect(),
    Expr::FunctionCall { name, args } if name == "Directive" => {
      args.iter().collect()
    }
    other => vec![other],
  };
  items.into_iter().find_map(|item| {
    let c = parse_color(item)?;
    let ch = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    Some(RGBColor(ch(c.r), ch(c.g), ch(c.b)))
  })
}

fn parse_tick_style(value: &Expr, base: TickStyle) -> TickStyle {
  let mut style = base;
  let items: Vec<&Expr> = match value {
    Expr::List(items) => items.iter().collect(),
    other => vec![other],
  };
  for item in items {
    if let Some(size) = try_eval_to_f64(item) {
      style.size = Some(size);
    } else if let Some(c) = parse_color(item) {
      style.color = Some((
        (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
        (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
        (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
      ));
    }
  }
  style
}

/// The explicit ticks of one frame edge: a list of `{pos, label, …}`.
/// `None` / `Automatic` / a bare list of positions draw no custom labels.
fn parse_edge_tick_labels(edge: &Expr) -> Vec<FrameTickLabel> {
  let Expr::List(ticks) = edge else {
    return Vec::new();
  };
  ticks
    .iter()
    .filter_map(|tick| {
      let Expr::List(parts) = tick else {
        return None;
      };
      if parts.len() < 2 {
        return None;
      }
      let pos = try_eval_to_f64(&parts[0])?;
      let (label, style) = match &parts[1] {
        Expr::FunctionCall { name, args }
          if name == "Style" && !args.is_empty() =>
        {
          let style = args[1..]
            .iter()
            .fold(TickStyle::default(), |st, a| parse_tick_style(a, st));
          (&args[0], style)
        }
        other => (other, TickStyle::default()),
      };
      let text = match label {
        Expr::String(s) => s.clone(),
        other => crate::syntax::expr_to_output(other),
      };
      Some(FrameTickLabel { pos, text, style })
    })
    .collect()
}

/// `FrameTicks -> {{left, right}, {bottom, top}}` (also `{bottom, left}`
/// pairs of plain tick lists apply to the bottom and left edges).
fn parse_frame_tick_labels(value: &Expr) -> FrameTickLabels {
  let mut out = FrameTickLabels::default();
  let Expr::List(items) = value else {
    return out;
  };
  if items.len() != 2 {
    return out;
  }
  // An edge pair holds tick lists (lists of lists) or `None`/`Automatic`.
  let pair = |e: &Expr| match e {
    Expr::List(p) if p.len() == 2 => Some((p[0].clone(), p[1].clone())),
    _ => None,
  };
  if let (Some((l, r)), Some((b, t))) = (pair(&items[0]), pair(&items[1])) {
    out.left = parse_edge_tick_labels(&l);
    out.right = parse_edge_tick_labels(&r);
    out.bottom = parse_edge_tick_labels(&b);
    out.top = parse_edge_tick_labels(&t);
  } else {
    out.bottom = parse_edge_tick_labels(&items[0]);
    out.left = parse_edge_tick_labels(&items[1]);
  }
  out
}

/// Draw explicit `FrameTicks` labels around a finished array plot. Positions
/// are in `ArrayPlot` coordinates: the center of the cell at column `j`
/// (resp. row `i`, counted from the top) is `j` (resp. `i`). The picture is
/// inset into a canvas widened by the label extents; returns the new size.
fn frame_ticked_svg(
  svg: &str,
  width: u32,
  height: u32,
  (n_cols, n_rows): (usize, usize),
  ticks: &FrameTickLabels,
  default_style: &TickStyle,
) -> (String, u32, u32) {
  let all = [&ticks.left, &ticks.right, &ticks.bottom, &ticks.top];
  if all.iter().all(|e| e.is_empty()) {
    return (svg.to_string(), width, height);
  }
  const DEFAULT_SIZE: f64 = 12.0;
  const GAP: f64 = 6.0;
  let size_of = |t: &FrameTickLabel| {
    t.style.size.or(default_style.size).unwrap_or(DEFAULT_SIZE)
  };
  let text_w =
    |t: &FrameTickLabel| 0.55 * size_of(t) * t.text.chars().count() as f64;
  let side = |edge: &[FrameTickLabel], horizontal: bool| -> f64 {
    edge
      .iter()
      .map(|t| {
        if horizontal {
          text_w(t)
        } else {
          size_of(t) * 1.2
        }
      })
      .fold(0.0, f64::max)
      + if edge.is_empty() { 0.0 } else { GAP }
  };
  let (ml, mr) = (side(&ticks.left, true), side(&ticks.right, true));
  let (mt, mb) = (side(&ticks.top, false), side(&ticks.bottom, false));
  let (w, h) = (width as f64, height as f64);
  let (total_w, total_h) = ((w + ml + mr).ceil(), (h + mt + mb).ceil());
  let fill_default = crate::functions::graphics::theme().text_primary;
  let label = |t: &FrameTickLabel, x: f64, y: f64, anchor: &str| {
    let fill = t
      .style
      .color
      .or(default_style.color)
      .map_or(fill_default.to_string(), |(r, g, b)| {
        format!("rgb({r},{g},{b})")
      });
    format!(
      "<text x=\"{x:.1}\" y=\"{y:.1}\" text-anchor=\"{anchor}\" \
       dominant-baseline=\"central\" font-family=\"sans-serif\" \
       font-size=\"{:.0}\" fill=\"{fill}\">{}</text>\n",
      size_of(t),
      crate::functions::graphics::svg_escape(&t.text)
    )
  };
  let mut out = format!(
    "<svg width=\"{total_w:.0}\" height=\"{total_h:.0}\" \
     viewBox=\"0 0 {total_w:.0} {total_h:.0}\" \
     xmlns=\"http://www.w3.org/2000/svg\">\n\
     <svg x=\"{ml:.1}\" y=\"{mt:.1}\" width=\"{w:.0}\" height=\"{h:.0}\" \
     overflow=\"visible\">\n{}\n</svg>\n",
    svg.trim()
  );
  let x_of = |pos: f64| ml + (pos - 0.5) / n_cols as f64 * w;
  let y_of = |pos: f64| mt + (pos - 0.5) / n_rows as f64 * h;
  for t in &ticks.left {
    out.push_str(&label(t, ml - GAP, y_of(t.pos), "end"));
  }
  for t in &ticks.right {
    out.push_str(&label(t, ml + w + GAP, y_of(t.pos), "start"));
  }
  for t in &ticks.bottom {
    out.push_str(&label(
      t,
      x_of(t.pos),
      mt + h + GAP + size_of(t) * 0.6,
      "middle",
    ));
  }
  for t in &ticks.top {
    out.push_str(&label(
      t,
      x_of(t.pos),
      mt - GAP - size_of(t) * 0.6,
      "middle",
    ));
  }
  out.push_str("</svg>\n");
  (out, total_w as u32, total_h as u32)
}

/// Set a `PlotLabel` caption above a finished picture: the picture keeps its
/// size and is inset into a canvas that is taller by one line of text.
fn with_plot_label_above(
  svg: &str,
  (label, size): &(String, Option<f64>),
) -> String {
  let attr = |name: &str| -> Option<f64> {
    let head = &svg[..svg.find('>')?];
    let start = head.find(&format!(" {name}=\""))? + name.len() + 3;
    let end = start + head[start..].find('"')?;
    head[start..end].parse().ok()
  };
  let (Some(w), Some(h)) = (attr("width"), attr("height")) else {
    return svg.to_string();
  };
  if label.trim().is_empty() {
    return svg.to_string();
  }
  let font = size.unwrap_or(14.0);
  let gutter = font * 1.8;
  let total_h = h + gutter;
  let fill = crate::functions::graphics::theme().text_primary;
  format!(
    "<svg width=\"{w:.0}\" height=\"{total_h:.0}\" \
     viewBox=\"0 0 {w:.0} {total_h:.0}\" xmlns=\"http://www.w3.org/2000/svg\">\n\
     <svg x=\"0\" y=\"{gutter:.1}\" width=\"{w:.0}\" height=\"{h:.0}\" \
     overflow=\"visible\">\n{}\n</svg>\n\
     <text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" \
     dominant-baseline=\"central\" font-family=\"sans-serif\" \
     font-size=\"{font:.0}\" fill=\"{fill}\" xml:space=\"preserve\">{label}</text>\n</svg>\n",
    svg.trim(),
    w / 2.0,
    gutter / 2.0,
  )
}

/// Put a plot's `FrameLabel` text around an already-rendered picture.
///
/// The picture keeps its size and is inset into a larger canvas with just
/// enough room on each labelled edge for one line of text. Left and right
/// labels are rotated to run along their edge, the way Wolfram sets them:
/// counter-clockwise on the left, clockwise on the right, so each reads
/// from the outside in. Returns `svg` unchanged when there is no label.
fn frame_labelled_svg(
  svg: &str,
  width: u32,
  height: u32,
  labels: &crate::functions::plot::FrameLabels,
) -> String {
  let edges = [&labels.bottom, &labels.left, &labels.top, &labels.right];
  if edges.iter().all(|e| e.is_empty()) {
    return svg.to_string();
  }
  let font_size = 14.0_f64;
  let gutter = font_size * 1.6;
  let pad = |present: bool| if present { gutter } else { 0.0 };
  let (ml, mr) = (pad(!labels.left.is_empty()), pad(!labels.right.is_empty()));
  let (mt, mb) = (pad(!labels.top.is_empty()), pad(!labels.bottom.is_empty()));
  let (w, h) = (width as f64, height as f64);
  let (total_w, total_h) = (w + ml + mr, h + mt + mb);
  let fill = crate::functions::graphics::theme().text_primary;
  let text = |x: f64, y: f64, rot: f64, label: &str| {
    let transform = if rot == 0.0 {
      String::new()
    } else {
      format!(" transform=\"rotate({rot},{x:.1},{y:.1})\"")
    };
    format!(
      "<text x=\"{x:.1}\" y=\"{y:.1}\" text-anchor=\"middle\" \
       dominant-baseline=\"central\" font-family=\"sans-serif\" \
       font-size=\"{font_size:.0}\" fill=\"{fill}\"{transform}>{}</text>\n",
      crate::functions::graphics::box_string_to_svg(label)
    )
  };
  let mut out = format!(
    "<svg width=\"{total_w:.0}\" height=\"{total_h:.0}\" \
     viewBox=\"0 0 {total_w:.0} {total_h:.0}\" \
     xmlns=\"http://www.w3.org/2000/svg\">\n"
  );
  // The picture itself, shifted into the space the labels leave.
  out.push_str(&format!(
    "<svg x=\"{ml:.1}\" y=\"{mt:.1}\" width=\"{w:.0}\" height=\"{h:.0}\" \
     overflow=\"visible\">\n"
  ));
  out.push_str(svg.trim());
  out.push_str("\n</svg>\n");
  let (cx, cy) = (ml + w / 2.0, mt + h / 2.0);
  if !labels.top.is_empty() {
    out.push_str(&text(cx, mt / 2.0, 0.0, &labels.top));
  }
  if !labels.bottom.is_empty() {
    out.push_str(&text(cx, mt + h + mb / 2.0, 0.0, &labels.bottom));
  }
  if !labels.left.is_empty() {
    out.push_str(&text(ml / 2.0, cy, -90.0, &labels.left));
  }
  if !labels.right.is_empty() {
    out.push_str(&text(ml + w + mr / 2.0, cy, 90.0, &labels.right));
  }
  out.push_str("</svg>\n");
  out
}

/// MatrixPlot[matrix] - like ArrayPlot with automatic color scaling
pub fn matrix_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Options that choose how cells are colored or captioned are handled by
  // the one canonical grid renderer, ArrayPlot.
  let styled = crate::functions::graphics::splice_option_lists(args)
    .iter()
    .skip(1)
    .any(|opt| {
      matches!(opt, Expr::Rule { pattern, .. }
        if matches!(pattern.as_ref(), Expr::Identifier(n)
          if matches!(n.as_str(),
            "ColorFunction" | "ColorRules" | "PlotLabel" | "Mesh")))
    });
  if styled {
    return array_plot_ast(args);
  }
  let data = evaluate_expr_to_expr(&args[0])?;
  let data = crate::functions::list_helpers_ast::densify_sparse_array(&data)
    .unwrap_or(data);
  let Expr::List(rows) = &data else {
    return Err(InterpreterError::EvaluationError(
      "MatrixPlot: first argument must be a matrix".into(),
    ));
  };

  let mut matrix: Vec<Vec<f64>> = Vec::new();
  // Entries that are themselves colors (`Pink`, `RGBColor[…]`) are drawn as
  // given and take no part in the automatic value scaling.
  let mut direct: Vec<Vec<Option<GfxColor>>> = Vec::new();
  let mut v_min = f64::INFINITY;
  let mut v_max = f64::NEG_INFINITY;

  for row in rows {
    if let Expr::List(items) = row {
      let mut vals = Vec::with_capacity(items.len());
      let mut colors = Vec::with_capacity(items.len());
      for item in items {
        let v = evaluate_expr_to_expr(item).unwrap_or(item.clone());
        let color = parse_color(&v);
        let val = if color.is_some() {
          0.0
        } else {
          try_eval_to_f64(&v).unwrap_or(0.0)
        };
        if color.is_none() && val.is_finite() {
          v_min = v_min.min(val);
          v_max = v_max.max(val);
        }
        vals.push(val);
        colors.push(color);
      }
      matrix.push(vals);
      direct.push(colors);
    }
  }

  if matrix.is_empty() {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }

  let (svg_width, svg_height, full_width) = parse_field_options(args, 1);
  let n_rows = matrix.len();
  let n_cols = matrix.iter().map(std::vec::Vec::len).max().unwrap_or(1);
  let w = svg_width as f64;
  let h = svg_height as f64;
  let cell_w = w / n_cols as f64;
  let cell_h = h / n_rows as f64;

  let mut svg = svg_header(svg_width, svg_height, full_width);

  for (i, row) in matrix.iter().enumerate() {
    for (j, &val) in row.iter().enumerate() {
      let fill = if let Some(c) = direct[i][j] {
        c.to_svg_rgb()
      } else {
        let (r, g, b) = value_to_color(val, v_min, v_max);
        format!("rgb({r},{g},{b})")
      };
      let sx = j as f64 * cell_w;
      let sy = i as f64 * cell_h;
      svg.push_str(&format!(
        "<rect x=\"{sx:.1}\" y=\"{sy:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"{fill}\" stroke=\"none\"/>\n",
        cell_w + 0.5, cell_h + 0.5
      ));
    }
  }

  svg.push_str("</svg>");
  Ok(crate::graphics_result(svg))
}

/// Parse a complex iterator: {z, zmin, zmax} where zmin/zmax are complex numbers.
/// Returns (var_name, re_min, re_max, im_min, im_max).
fn parse_complex_iterator(
  spec: &Expr,
  label: &str,
) -> Result<(String, f64, f64, f64, f64), InterpreterError> {
  match spec {
    Expr::List(items) if items.len() == 3 => {
      let var = match &items[0] {
        Expr::Identifier(name) => name.clone(),
        _ => {
          return Err(InterpreterError::EvaluationError(format!(
            "{label}: iterator variable must be a symbol"
          )));
        }
      };
      let zmin_expr = evaluate_expr_to_expr(&items[1])?;
      let zmax_expr = evaluate_expr_to_expr(&items[2])?;
      let (re_min, im_min) =
        try_extract_complex_f64(&zmin_expr).ok_or_else(|| {
          InterpreterError::EvaluationError(format!(
            "{label}: cannot evaluate iterator min to a complex number"
          ))
        })?;
      let (re_max, im_max) =
        try_extract_complex_f64(&zmax_expr).ok_or_else(|| {
          InterpreterError::EvaluationError(format!(
            "{label}: cannot evaluate iterator max to a complex number"
          ))
        })?;
      Ok((var, re_min, re_max, im_min, im_max))
    }
    _ => Err(InterpreterError::EvaluationError(format!(
      "{label}: iterator must be {{var, zmin, zmax}}"
    ))),
  }
}

/// Evaluate a complex function f(z) at the point (re, im).
/// Substitutes z with the complex value and extracts (re, im) of the result.
fn evaluate_complex_at(
  body: &Expr,
  zvar: &str,
  re: f64,
  im: f64,
) -> Option<(f64, f64)> {
  let z_val = build_complex_float_expr(re, im);
  let sub = substitute_var(body, zvar, &z_val);
  let result = evaluate_expr_to_expr(&sub).ok()?;
  try_extract_complex_f64(&result)
}

/// Convert complex value to domain coloring HSB color.
/// Hue is determined by Arg(z), brightness by Abs(z).
fn complex_to_rgb(re: f64, im: f64) -> (u8, u8, u8) {
  // libm (not the std methods backed by the platform's libm) so the result
  // is bit-identical across platforms; macOS and glibc disagree by ULPs,
  // which flips the 8-bit rounding below at .5 boundaries.
  let arg = libm::atan2(im, re); // -PI to PI
  let abs = (re * re + im * im).sqrt();

  // Hue: map argument from [-PI, PI] to [0, 1]
  let hue = (arg + std::f64::consts::PI) / (2.0 * std::f64::consts::PI);

  // Brightness: use a smooth function that maps [0, inf) to [0, 1)
  // Following Mathematica's approach: brightness varies with magnitude
  let brightness = 1.0 - 1.0 / (1.0 + libm::pow(abs, 0.3));

  // Saturation: high saturation, slightly reduced for very large/small magnitudes
  let saturation = 0.8 + 0.2 * (1.0 - (2.0 * brightness - 1.0).abs());

  hsb_to_rgb(hue, saturation, brightness)
}

/// Convert HSB (all in [0,1]) to RGB.
fn hsb_to_rgb(h: f64, s: f64, b: f64) -> (u8, u8, u8) {
  let h = h.fract();
  let h = if h < 0.0 { h + 1.0 } else { h };
  let hi = (h * 6.0).floor() as u32 % 6;
  let f = h * 6.0 - hi as f64;
  let p = b * (1.0 - s);
  let q = b * (1.0 - f * s);
  let t = b * (1.0 - (1.0 - f) * s);
  let (r, g, bl) = match hi {
    0 => (b, t, p),
    1 => (q, b, p),
    2 => (p, b, t),
    3 => (p, q, b),
    4 => (t, p, b),
    _ => (b, p, q),
  };
  (channel_to_u8(r), channel_to_u8(g), channel_to_u8(bl))
}

/// Quantize a [0,1] color channel to an 8-bit value.
///
/// The magnitude/argument feeding domain coloring is computed with libm's
/// `atan2`/`pow`, whose internal `mul_add` calls compile to a hardware FMA on
/// aarch64 but not on x86_64. That single-vs-double rounding shifts the result
/// by ~1e-13, which is invisible in the color itself but flips the 8-bit
/// rounding of a channel value sitting exactly on an `x.5` boundary — so the
/// same plot renders one LSB apart on macOS and on the Linux CI runner.
///
/// Snapping to a 1e-9 grid (10^4× the FP noise, far below any perceptible color
/// step) before rounding pulls both platforms onto the same side of the
/// boundary, making the SVG bit-identical everywhere.
fn channel_to_u8(c: f64) -> u8 {
  let v = c * 255.0;
  ((v * 1e9).round() / 1e9).round() as u8
}

/// ComplexPlot[f, {z, zmin, zmax}]
/// Domain coloring visualization of a complex function.
pub fn complex_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let body = &args[0];
  let (zvar, re_min, re_max, im_min, im_max) =
    parse_complex_iterator(&args[1], "ComplexPlot")?;
  let (svg_width, svg_height, full_width) = parse_field_options(args, 2);

  let grid_size = FIELD_GRID;
  let plot_label = parse_field_plot_label(args, 2);

  // Use plotters for axes (Re on x-axis, Im on y-axis), reserving room above
  // the frame for a PlotLabel.
  let margins = field_plot_label_margins(plot_label.as_ref());
  let mut area = crate::functions::plot::generate_axes_only_opts(
    (re_min, re_max),
    (im_min, im_max),
    svg_width,
    svg_height,
    full_width,
    None,
    margins.as_ref(),
  )?;

  let mut svg = std::mem::take(&mut area.svg);
  // Remove closing </svg> to append custom elements
  if let Some(pos) = svg.rfind("</svg>") {
    svg.truncate(pos);
  }
  if let Some(label) = &plot_label {
    inject_field_plot_label(&mut svg, &area, label);
  }

  // Render domain coloring as one embedded bitmap (a pixel per sample)
  // stretched over the plot area: no per-cell rect seams.
  use base64::Engine as _;
  let mut img = image::RgbaImage::new(grid_size as u32, grid_size as u32);
  for i in 0..grid_size {
    let re = re_min + (i as f64 + 0.5) / grid_size as f64 * (re_max - re_min);
    for j in 0..grid_size {
      let im = im_min + (j as f64 + 0.5) / grid_size as f64 * (im_max - im_min);
      let px = if let Some((fre, fim)) =
        evaluate_complex_at(body, &zvar, re, im)
        && fre.is_finite()
        && fim.is_finite()
      {
        let (r, g, b) = complex_to_rgb(fre, fim);
        image::Rgba([r, g, b, 255])
      } else {
        image::Rgba([0, 0, 0, 0])
      };
      // Flip y: higher im values at top
      img.put_pixel(i as u32, (grid_size - 1 - j) as u32, px);
    }
  }
  let mut png = Vec::new();
  if image::DynamicImage::ImageRgba8(img)
    .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
    .is_ok()
  {
    let b64 = base64::engine::general_purpose::STANDARD.encode(&png);
    svg.push_str(&format!(
      "<image x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" preserveAspectRatio=\"none\" href=\"data:image/png;base64,{b64}\"/>\n",
      area.plot_x0, area.plot_y0, area.plot_w, area.plot_h
    ));
  }
  push_frame(
    &mut svg,
    area.plot_x0,
    area.plot_y0,
    area.plot_w,
    area.plot_h,
  );

  svg.push_str("</svg>");
  Ok(crate::graphics_result(svg))
}

#[cfg(test)]
mod robust_range_tests {
  use super::robust_value_range;

  #[test]
  fn robust_value_range_ignores_a_pole() {
    // A handful of samples near a singularity (huge magnitude, like a
    // function blowing up at one grid point) shouldn't drag v_max out to
    // where the rest of the well-behaved samples all scale to the same
    // color. Mirrors the shape of a `ContourPlot`/`DensityPlot` grid sample
    // set for a function like `1/Sqrt[x^2 + y^2]`.
    let mut samples: Vec<f64> = (0..96).map(|i| i as f64 * 0.1).collect();
    samples.extend([1.0e6, 2.0e6, 5.0e6, 1.0e7]);

    let (v_min, v_max) = robust_value_range(&samples);

    assert!(v_min <= 0.0, "got v_min={v_min}");
    // The pole samples must be fenced off — the reported max should stay
    // close to the well-behaved bulk, not jump to the outliers.
    assert!(
      v_max < 100.0,
      "expected the pole samples to be excluded, got v_max={v_max}"
    );
  }

  #[test]
  fn robust_value_range_keeps_full_span_without_outliers() {
    // A smooth, well-behaved sample set (no huge gap between the bulk and
    // the extremes) should keep its literal min/max — nothing to fence off.
    let samples: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
    let (v_min, v_max) = robust_value_range(&samples);
    assert_eq!(v_min, 0.0);
    assert_eq!(v_max, 9.9);
  }

  #[test]
  fn robust_value_range_empty_is_non_finite() {
    let (v_min, v_max) = robust_value_range(&[]);
    assert!(!v_min.is_finite());
    assert!(!v_max.is_finite());
  }
}
