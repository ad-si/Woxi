use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::graphics::{Color, parse_color};
use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::plot::{
  BinSpec, DEFAULT_HEIGHT, DEFAULT_WIDTH, MarginOverrides, PLOT_COLORS,
  RESOLUTION_SCALE, generate_bar_svg, generate_histogram_svg, parse_image_size,
};
use crate::syntax::Expr;

/// Extract a flat list of f64 values from the first argument.
fn extract_values(arg: &Expr) -> Result<Vec<f64>, InterpreterError> {
  let data = evaluate_expr_to_expr(arg)?;
  match &data {
    Expr::List(items) => {
      let mut vals = Vec::with_capacity(items.len());
      for item in items {
        let v = evaluate_expr_to_expr(item).unwrap_or(item.clone());
        if let Some(f) = try_eval_to_f64(&v) {
          vals.push(f);
        }
      }
      Ok(vals)
    }
    _ => Err(InterpreterError::EvaluationError(
      "Chart: first argument must be a list".into(),
    )),
  }
}

/// Extract grouped values from the first argument.
/// Returns a list of groups, where each group is a list of f64 values.
/// - `{1, 2, 3}` → `[[1], [2], [3]]` (flat: each value is its own group)
/// - `{{1,2}, {3,4}}` → `[[1,2], [3,4]]` (grouped: sublists are groups)
fn extract_grouped_values(
  arg: &Expr,
) -> Result<Vec<Vec<f64>>, InterpreterError> {
  let data = evaluate_expr_to_expr(arg)?;
  let items = match &data {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Chart: first argument must be a list".into(),
      ));
    }
  };

  if items.is_empty() {
    return Ok(vec![]);
  }

  // Check if the first element is a list (grouped mode)
  let first_eval = evaluate_expr_to_expr(&items[0]).unwrap_or(items[0].clone());
  if matches!(&first_eval, Expr::List(_)) {
    // Grouped: each item is a sublist
    let mut groups = Vec::with_capacity(items.len());
    for item in items {
      let ev = evaluate_expr_to_expr(item).unwrap_or(item.clone());
      if let Expr::List(inner) = &ev {
        let mut vals = Vec::new();
        for v in inner {
          let vv = evaluate_expr_to_expr(v).unwrap_or(v.clone());
          if let Some(f) = try_eval_to_f64(&vv) {
            vals.push(f);
          }
        }
        groups.push(vals);
      }
    }
    Ok(groups)
  } else {
    // Flat: each value is its own group with one bar
    let mut groups = Vec::with_capacity(items.len());
    for item in items {
      let v = evaluate_expr_to_expr(item).unwrap_or(item.clone());
      if let Some(f) = try_eval_to_f64(&v) {
        groups.push(vec![f]);
      }
    }
    Ok(groups)
  }
}

/// A label with optional styling (bold, italic, color, font-size).
pub(crate) struct StyledLabel {
  pub text: String,
  pub bold: bool,
  pub italic: bool,
  pub color: Option<Color>,
  pub font_size: Option<f64>,
}

/// Parse a plain string or `Style["text", Bold, Italic, color, size]` into a `StyledLabel`.
pub(crate) fn parse_styled_label(expr: &Expr) -> Option<StyledLabel> {
  match expr {
    Expr::String(s) => Some(StyledLabel {
      text: s.clone(),
      bold: false,
      italic: false,
      color: None,
      font_size: None,
    }),
    Expr::Identifier(s) => Some(StyledLabel {
      text: s.clone(),
      bold: false,
      italic: false,
      color: None,
      font_size: None,
    }),
    Expr::FunctionCall { name, args }
      if name == "Style" && !args.is_empty() =>
    {
      let text = match &args[0] {
        Expr::String(s) => s.clone(),
        Expr::Identifier(s) => s.clone(),
        _ => return None,
      };
      let mut bold = false;
      let mut italic = false;
      let mut color = None;
      let mut font_size = None;
      for arg in &args[1..] {
        match arg {
          Expr::Identifier(s) if s == "Bold" => bold = true,
          Expr::Identifier(s) if s == "Italic" => italic = true,
          _ => {
            if let Some(c) = parse_color(arg) {
              color = Some(c);
            } else if let Some(f) = try_eval_to_f64(arg) {
              font_size = Some(f);
            }
          }
        }
      }
      Some(StyledLabel {
        text,
        bold,
        italic,
        color,
        font_size,
      })
    }
    _ => None,
  }
}

/// Position for chart labels placed with `Placed[labels, position]`.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum LabelPosition {
  Below,  // Below the bar at the x-axis (default)
  Above,  // Just above the top of the bar
  Center, // Vertically centered within the bar
}

/// Parsed chart options.
pub(crate) struct ChartOptions {
  pub svg_width: u32,
  pub svg_height: u32,
  pub full_width: bool,
  pub chart_labels: Vec<ChartLabel>,
  pub chart_label_position: LabelPosition,
  pub plot_label: Option<StyledLabel>,
  pub axes_label: Option<(String, String)>,
  pub chart_style: Vec<Color>,
}

/// A chart label with optional rotation angle (in radians).
#[derive(Clone)]
pub(crate) struct ChartLabel {
  pub text: String,
  /// Rotation angle in radians (0.0 = no rotation, negative = clockwise).
  pub rotation: f64,
}

impl ChartLabel {
  pub fn plain(text: String) -> Self {
    Self {
      text,
      rotation: 0.0,
    }
  }
}

/// Extract a string from an Expr (Identifier or String).
pub(crate) fn expr_to_label(e: &Expr) -> Option<String> {
  match e {
    Expr::String(s) => Some(s.clone()),
    Expr::Identifier(s) => Some(s.clone()),
    _ => None,
  }
}

/// Extract a chart label from an Expr, supporting Rotate[label, angle].
fn expr_to_chart_label(e: &Expr) -> Option<ChartLabel> {
  // Rotate["label", angle]
  if let Expr::FunctionCall { name, args } = e
    && name == "Rotate"
    && args.len() >= 2
  {
    let text = expr_to_label(&args[0])?;
    let angle = try_eval_to_f64(
      &evaluate_expr_to_expr(&args[1]).unwrap_or(args[1].clone()),
    )
    .unwrap_or(0.0);
    return Some(ChartLabel {
      text,
      rotation: angle,
    });
  }
  // Plain label
  expr_to_label(e).map(ChartLabel::plain)
}

/// Parse options from chart arguments.
fn parse_chart_options(args: &[Expr]) -> ChartOptions {
  let mut opts = ChartOptions {
    svg_width: DEFAULT_WIDTH,
    svg_height: DEFAULT_HEIGHT,
    full_width: false,
    chart_labels: Vec::new(),
    chart_label_position: LabelPosition::Below,
    plot_label: None,
    axes_label: None,
    chart_style: Vec::new(),
  };
  for opt in &args[1..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && let Expr::Identifier(name) = pattern.as_ref()
    {
      match name.as_str() {
        "ImageSize" => {
          if let Some((w, h, fw)) =
            parse_image_size(replacement, DEFAULT_WIDTH, DEFAULT_HEIGHT)
          {
            opts.svg_width = w;
            opts.svg_height = h;
            opts.full_width = fw;
          }
        }
        "ChartLabels" => {
          // Placed[{labels...}, position] — check raw expr first to avoid
          // evaluating Placed[] as an unknown function
          let raw = replacement.as_ref();
          if let Expr::FunctionCall { name, args } = raw
            && name == "Placed"
            && args.len() == 2
          {
            let pos_name = match &args[1] {
              Expr::Identifier(s) => s.as_str(),
              _ => "",
            };
            opts.chart_label_position = match pos_name {
              "Above" | "Top" => LabelPosition::Above,
              "Center" => LabelPosition::Center,
              // Below, Bottom, Right, Left, and any other value → default (Below)
              _ => LabelPosition::Below,
            };
            let labels_val =
              evaluate_expr_to_expr(&args[0]).unwrap_or(args[0].clone());
            if let Expr::List(items) = &labels_val {
              for item in items {
                if let Some(cl) = expr_to_chart_label(item) {
                  opts.chart_labels.push(cl);
                }
              }
            }
          } else {
            let val = evaluate_expr_to_expr(replacement)
              .unwrap_or(*replacement.clone());
            if let Expr::List(items) = &val {
              for item in items {
                if let Some(cl) = expr_to_chart_label(item) {
                  opts.chart_labels.push(cl);
                }
              }
            }
          }
        }
        "PlotLabel" => {
          let val =
            evaluate_expr_to_expr(replacement).unwrap_or(*replacement.clone());
          if let Some(sl) = parse_styled_label(&val) {
            opts.plot_label = Some(sl);
          }
        }
        "AxesLabel" | "FrameLabel" => {
          let val =
            evaluate_expr_to_expr(replacement).unwrap_or(*replacement.clone());
          if let Expr::List(items) = &val
            && items.len() >= 2
          {
            let x = expr_to_label(&items[0]).unwrap_or_default();
            let y = expr_to_label(&items[1]).unwrap_or_default();
            opts.axes_label = Some((x, y));
          }
        }
        "ChartStyle" => {
          let val =
            evaluate_expr_to_expr(replacement).unwrap_or(*replacement.clone());
          match &val {
            Expr::List(items) => {
              for item in items {
                if let Some(c) = parse_color(item) {
                  opts.chart_style.push(c);
                }
              }
            }
            _ => {
              if let Some(c) = parse_color(&val) {
                opts.chart_style.push(c);
              }
            }
          }
        }
        _ => {}
      }
    }
  }
  opts
}

fn svg_header(w: u32, h: u32, full_width: bool) -> String {
  let (bg, _, _, _, _) = crate::functions::plot::plot_theme();
  let bg_fill = format!("rgb({},{},{})", bg.0, bg.1, bg.2);
  if full_width {
    format!(
      "<svg width=\"100%\" viewBox=\"0 0 {w} {h}\" preserveAspectRatio=\"xMidYMid meet\" xmlns=\"http://www.w3.org/2000/svg\">\n\
       <rect width=\"{w}\" height=\"{h}\" fill=\"{bg_fill}\"/>\n"
    )
  } else {
    format!(
      "<svg width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\" preserveAspectRatio=\"xMidYMid meet\" xmlns=\"http://www.w3.org/2000/svg\">\n\
       <rect width=\"{w}\" height=\"{h}\" fill=\"{bg_fill}\"/>\n"
    )
  }
}

/// Format a numeric value for display in chart tooltips.
/// Integers (or values very close to integers) are shown without decimals.
fn format_chart_value(v: f64) -> String {
  if (v - v.round()).abs() < 1e-10 {
    format!("{}", v as i64)
  } else {
    format!("{v}")
  }
}

/// BarChart[{v1, v2, ...}] or BarChart[{{v1, v2}, {v3, v4}, ...}]
pub fn bar_chart_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let groups = extract_grouped_values(&args[0])?;
  if groups.is_empty() {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }
  let opts = parse_chart_options(args);

  let svg = generate_bar_svg(
    &groups,
    opts.svg_width,
    opts.svg_height,
    opts.full_width,
    &opts.chart_labels,
    opts.chart_label_position,
    opts.plot_label.as_ref(),
    opts
      .axes_label
      .as_ref()
      .map(|(x, y)| (x.as_str(), y.as_str())),
    &opts.chart_style,
  )?;
  Ok(crate::graphics_result(svg))
}

/// PieChart[{v1, v2, ...}]
pub fn pie_chart_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let values = extract_values(&args[0])?;
  if values.is_empty() {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }
  let opts = parse_chart_options(args);
  let (svg_width, svg_height, full_width) =
    (opts.svg_width, opts.svg_height, opts.full_width);

  let w = svg_width as f64;
  let h = svg_height as f64;
  let cx = w / 2.0;
  let cy = h / 2.0;
  let radius = (w.min(h) / 2.0) * 0.85;
  let total: f64 = values.iter().sum();
  if total <= 0.0 {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }

  let mut svg = svg_header(svg_width, svg_height, full_width);

  // Sort slices smallest-to-largest, drawn clockwise.
  // The smallest slice sits right above the negative x-axis,
  // with sizes increasing in the clockwise direction.
  let mut indexed: Vec<(usize, f64)> =
    values.iter().copied().enumerate().collect();
  indexed
    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

  // Start at π (negative x-axis) so the smallest slice sits right above it.
  let mut start_angle = std::f64::consts::PI;
  for (color_idx, &(_orig_idx, val)) in indexed.iter().enumerate() {
    let (r, g, b) = PLOT_COLORS[color_idx % PLOT_COLORS.len()];
    let sweep = 2.0 * std::f64::consts::PI * val / total;
    let end_angle = start_angle + sweep;

    let x1 = cx + radius * start_angle.cos();
    let y1 = cy + radius * start_angle.sin();
    let x2 = cx + radius * end_angle.cos();
    let y2 = cy + radius * end_angle.sin();
    let large_arc = if sweep > std::f64::consts::PI { 1 } else { 0 };

    let tooltip = format_chart_value(val);
    svg.push_str(&format!(
      "<path d=\"M{cx:.2},{cy:.2} L{x1:.2},{y1:.2} A{radius:.2},{radius:.2} 0 {large_arc},1 {x2:.2},{y2:.2} Z\" \
       fill=\"rgb({r},{g},{b})\" stroke=\"white\" stroke-width=\"1\"><title>{tooltip}</title></path>\n"
    ));

    start_angle = end_angle;
  }

  svg.push_str("</svg>");
  Ok(crate::graphics_result(svg))
}

/// BarChart3D[{v1, v2, ...}] or BarChart3D[{{v1, v2}, {v3, v4}, ...}]
/// Renders a 3D bar chart with each bar drawn as a colored cuboid.
pub fn bar_chart_3d_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use crate::functions::plot3d::{
    Point3D, Triangle, apply_lighting, depth, project, tessellate_cuboid,
    triangle_normal,
  };

  let groups = match extract_grouped_values(&args[0]) {
    Ok(g) => g,
    Err(_) => {
      // Return unevaluated for invalid (non-list) input, matching wolframscript.
      return Ok(Expr::FunctionCall {
        name: "BarChart3D".to_string(),
        args: args.to_vec(),
      });
    }
  };
  if groups.is_empty() {
    return Ok(crate::graphics3d_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }
  let opts = parse_chart_options(args);
  let (svg_width, svg_height, full_width) =
    (opts.svg_width, opts.svg_height, opts.full_width);

  // Determine value range across all bars
  let mut v_max = f64::NEG_INFINITY;
  let mut v_min = f64::INFINITY;
  for group in &groups {
    for &v in group {
      v_max = v_max.max(v);
      v_min = v_min.min(v);
    }
  }
  if !v_max.is_finite() {
    return Ok(crate::graphics3d_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }
  let z_lo = v_min.min(0.0);
  let z_hi = v_max.max(0.0);
  let z_range = if (z_hi - z_lo).abs() < 1e-15 {
    1.0
  } else {
    z_hi - z_lo
  };

  // Map a value to normalized z in [-Z_SCALE, Z_SCALE].
  // Z_SCALE matches the default BoxRatios used by plot3d.
  const Z_SCALE: f64 = 0.4;
  let nz = |v: f64| -> f64 { ((v - z_lo) / z_range) * 2.0 * Z_SCALE - Z_SCALE };
  let z_base = nz(0.0);

  // Layout: each group occupies a "slot" along x; bars within a group are
  // positioned side by side along y.
  let n_groups = groups.len();
  let max_bars_per_group = groups.iter().map(|g| g.len()).max().unwrap_or(1);

  // Normalized box spans x in [-1, 1]. Bars are centered in y with a
  // total depth proportional to the per-group x width so they look thin
  // rather than stretched far into the back.
  let group_step = 2.0 / n_groups as f64;
  let group_pad = group_step * 0.15;
  // Total y-extent used by bars (shared across all groups). Tying this
  // to group_step keeps each bar's footprint close to square, and the
  // 0.75 factor keeps bars noticeably thinner than their visible width.
  let total_y_span = group_step * 0.75;
  let y_origin = -total_y_span / 2.0;
  let bar_y_step = total_y_span / max_bars_per_group as f64;
  let bar_y_pad = bar_y_step * 0.15;

  let camera = chart_3d_camera();
  let mut all_triangles: Vec<Triangle> = Vec::new();

  for (gi, group) in groups.iter().enumerate() {
    let x_lo = -1.0 + gi as f64 * group_step + group_pad;
    let x_hi = -1.0 + (gi as f64 + 1.0) * group_step - group_pad;
    for (bi, &v) in group.iter().enumerate() {
      let y_lo = y_origin + bi as f64 * bar_y_step + bar_y_pad;
      let y_hi = y_origin + (bi as f64 + 1.0) * bar_y_step - bar_y_pad;
      let z_top = nz(v);
      // Build cuboid from base (z_base) to top (z_top).
      let (zmin, zmax) = if z_top >= z_base {
        (z_base, z_top)
      } else {
        (z_top, z_base)
      };
      let p_min = Point3D {
        x: x_lo,
        y: y_lo,
        z: zmin,
      };
      let p_max = Point3D {
        x: x_hi,
        y: y_hi,
        z: zmax,
      };

      // Choose color: ChartStyle overrides; otherwise cycle PLOT_COLORS by
      // group index (single-bar) or by bar index within group.
      let color_index = if max_bars_per_group > 1 { bi } else { gi };
      let base_color = if !opts.chart_style.is_empty() {
        let c = &opts.chart_style[color_index % opts.chart_style.len()];
        (
          (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
          (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
          (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
        )
      } else {
        PLOT_COLORS[color_index % PLOT_COLORS.len()]
      };

      for (v0, v1, v2) in tessellate_cuboid(&p_min, &p_max) {
        let normal = triangle_normal(v0, v1, v2);
        let color = apply_lighting(base_color, normal);
        let center = Point3D {
          x: (v0.x + v1.x + v2.x) / 3.0,
          y: (v0.y + v1.y + v2.y) / 3.0,
          z: (v0.z + v1.z + v2.z) / 3.0,
        };
        all_triangles.push(Triangle {
          projected: [
            project(v0, &camera),
            project(v1, &camera),
            project(v2, &camera),
          ],
          depth: depth(center, &camera),
          color,
          opacity: 1.0,
        });
      }
    }
  }

  // Painter's algorithm: draw farthest first.
  all_triangles.sort_by(|a, b| {
    b.depth
      .partial_cmp(&a.depth)
      .unwrap_or(std::cmp::Ordering::Equal)
  });

  let svg =
    render_3d_triangles(&all_triangles, svg_width, svg_height, full_width);
  Ok(crate::graphics3d_result(svg))
}

/// PieChart3D[{v1, v2, ...}]
/// Renders a 3D pie chart: a short cylinder split into colored wedges.
pub fn pie_chart_3d_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use crate::functions::plot3d::{
    Point3D, Triangle, apply_lighting, depth, project, triangle_normal,
  };

  let values = match extract_values(&args[0]) {
    Ok(v) => v,
    Err(_) => {
      return Ok(Expr::FunctionCall {
        name: "PieChart3D".to_string(),
        args: args.to_vec(),
      });
    }
  };
  if values.is_empty() {
    return Ok(crate::graphics3d_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }
  let total: f64 = values.iter().sum();
  if total <= 0.0 {
    return Ok(crate::graphics3d_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }
  let opts = parse_chart_options(args);
  let (svg_width, svg_height, full_width) =
    (opts.svg_width, opts.svg_height, opts.full_width);

  // Geometry constants in normalized world space.
  let radius = 0.95_f64;
  let half_thickness = 0.12_f64;
  let z_top = half_thickness;
  let z_bot = -half_thickness;

  // Sort slices smallest-to-largest, drawn clockwise — same convention as
  // the 2D PieChart.
  let mut indexed: Vec<(usize, f64)> =
    values.iter().copied().enumerate().collect();
  indexed
    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

  let camera = chart_3d_camera();
  let mut all_triangles: Vec<Triangle> = Vec::new();
  // Number of segments per full circle for tessellating arcs
  const SEG_PER_TURN: usize = 64;

  let mut start_angle = std::f64::consts::PI;
  for (color_idx, &(_orig_idx, val)) in indexed.iter().enumerate() {
    let sweep = 2.0 * std::f64::consts::PI * val / total;
    if sweep <= 0.0 {
      continue;
    }
    let end_angle = start_angle + sweep;

    let base_color = if !opts.chart_style.is_empty() {
      let c = &opts.chart_style[color_idx % opts.chart_style.len()];
      (
        (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
        (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
        (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
      )
    } else {
      PLOT_COLORS[color_idx % PLOT_COLORS.len()]
    };

    // Number of subdivisions for this slice (at least 1).
    let n_seg = ((sweep / (2.0 * std::f64::consts::PI) * SEG_PER_TURN as f64)
      .ceil() as usize)
      .max(1);
    let center_top = Point3D {
      x: 0.0,
      y: 0.0,
      z: z_top,
    };
    let center_bot = Point3D {
      x: 0.0,
      y: 0.0,
      z: z_bot,
    };

    let push_tri =
      |v0: Point3D, v1: Point3D, v2: Point3D, all: &mut Vec<Triangle>| {
        let normal = triangle_normal(v0, v1, v2);
        let color = apply_lighting(base_color, normal);
        let center = Point3D {
          x: (v0.x + v1.x + v2.x) / 3.0,
          y: (v0.y + v1.y + v2.y) / 3.0,
          z: (v0.z + v1.z + v2.z) / 3.0,
        };
        all.push(Triangle {
          projected: [
            project(v0, &camera),
            project(v1, &camera),
            project(v2, &camera),
          ],
          depth: depth(center, &camera),
          color,
          opacity: 1.0,
        });
      };

    // Top fan, bottom fan, and outer cylindrical surface.
    for s in 0..n_seg {
      let t0 = s as f64 / n_seg as f64;
      let t1 = (s + 1) as f64 / n_seg as f64;
      let a0 = start_angle + t0 * sweep;
      let a1 = start_angle + t1 * sweep;
      let (c0, s0) = (a0.cos(), a0.sin());
      let (c1, s1) = (a1.cos(), a1.sin());

      let p_top_0 = Point3D {
        x: radius * c0,
        y: radius * s0,
        z: z_top,
      };
      let p_top_1 = Point3D {
        x: radius * c1,
        y: radius * s1,
        z: z_top,
      };
      let p_bot_0 = Point3D {
        x: radius * c0,
        y: radius * s0,
        z: z_bot,
      };
      let p_bot_1 = Point3D {
        x: radius * c1,
        y: radius * s1,
        z: z_bot,
      };

      // Top face (fan from center)
      push_tri(center_top, p_top_0, p_top_1, &mut all_triangles);
      // Bottom face (reverse winding so the normal points down)
      push_tri(center_bot, p_bot_1, p_bot_0, &mut all_triangles);
      // Outer cylindrical wall: two triangles per quad
      push_tri(p_top_0, p_bot_0, p_bot_1, &mut all_triangles);
      push_tri(p_top_0, p_bot_1, p_top_1, &mut all_triangles);
    }

    // Two flat side walls (one for each radial edge of the slice).
    let (cs_lo, ss_lo) = (start_angle.cos(), start_angle.sin());
    let (cs_hi, ss_hi) = (end_angle.cos(), end_angle.sin());
    let edge0_top = Point3D {
      x: radius * cs_lo,
      y: radius * ss_lo,
      z: z_top,
    };
    let edge0_bot = Point3D {
      x: radius * cs_lo,
      y: radius * ss_lo,
      z: z_bot,
    };
    let edge1_top = Point3D {
      x: radius * cs_hi,
      y: radius * ss_hi,
      z: z_top,
    };
    let edge1_bot = Point3D {
      x: radius * cs_hi,
      y: radius * ss_hi,
      z: z_bot,
    };
    // Side at start_angle
    push_tri(center_top, center_bot, edge0_bot, &mut all_triangles);
    push_tri(center_top, edge0_bot, edge0_top, &mut all_triangles);
    // Side at end_angle (reverse winding for the opposing face)
    push_tri(center_top, edge1_top, edge1_bot, &mut all_triangles);
    push_tri(center_top, edge1_bot, center_bot, &mut all_triangles);

    start_angle = end_angle;
  }

  // Painter's algorithm
  all_triangles.sort_by(|a, b| {
    b.depth
      .partial_cmp(&a.depth)
      .unwrap_or(std::cmp::Ordering::Equal)
  });

  let svg =
    render_3d_triangles(&all_triangles, svg_width, svg_height, full_width);
  Ok(crate::graphics3d_result(svg))
}

/// Camera angle used by the 3D chart functions. Pulled back from the
/// default Plot3D viewpoint so bars/slices are viewed more from the front
/// and less from above, giving a flatter, more front-facing look.
/// `azimuth` closer to -π/2 ≈ -1.5708 places the viewer directly in front
/// (along -y); values closer to 0 move it toward the right (+x).
fn chart_3d_camera() -> crate::functions::plot3d::Camera {
  crate::functions::plot3d::Camera {
    azimuth: -1.35, // ~-77° — only ~13° offset from a direct front view
    elevation: 0.32, // ~18°
  }
}

/// Render a list of depth-sorted projected triangles into an SVG string.
fn render_3d_triangles(
  triangles: &[crate::functions::plot3d::Triangle],
  svg_width: u32,
  svg_height: u32,
  full_width: bool,
) -> String {
  // Find bounding box of all projected points
  let mut px_min = f64::INFINITY;
  let mut px_max = f64::NEG_INFINITY;
  let mut py_min = f64::INFINITY;
  let mut py_max = f64::NEG_INFINITY;
  for tri in triangles {
    for &(px, py) in &tri.projected {
      px_min = px_min.min(px);
      px_max = px_max.max(px);
      py_min = py_min.min(py);
      py_max = py_max.max(py);
    }
  }
  if !px_min.is_finite() {
    px_min = -1.0;
    px_max = 1.0;
    py_min = -1.0;
    py_max = 1.0;
  }
  let p_width = (px_max - px_min).max(1e-15);
  let p_height = (py_max - py_min).max(1e-15);

  let margin = 25.0;
  let draw_w = svg_width as f64 - 2.0 * margin;
  let draw_h = svg_height as f64 - 2.0 * margin;
  let scale = (draw_w / p_width).min(draw_h / p_height);
  let cx = margin + draw_w / 2.0;
  let cy = margin + draw_h / 2.0;
  let p_cx = (px_min + px_max) / 2.0;
  let p_cy = (py_min + py_max) / 2.0;

  let to_svg = |px: f64, py: f64| -> (f64, f64) {
    (cx + (px - p_cx) * scale, cy - (py - p_cy) * scale)
  };

  let mut svg = String::with_capacity(triangles.len() * 120 + 1000);
  if full_width {
    svg.push_str(&format!(
      "<svg width=\"100%\" viewBox=\"0 0 {svg_width} {svg_height}\" preserveAspectRatio=\"xMidYMid meet\" xmlns=\"http://www.w3.org/2000/svg\">\n"
    ));
  } else {
    svg.push_str(&format!(
      "<svg width=\"{svg_width}\" height=\"{svg_height}\" viewBox=\"0 0 {svg_width} {svg_height}\" xmlns=\"http://www.w3.org/2000/svg\">\n"
    ));
  }
  let (bg, _, _, _, _) = crate::functions::plot::plot_theme();
  svg.push_str(&format!(
    "<rect width=\"{svg_width}\" height=\"{svg_height}\" fill=\"rgb({},{},{})\"/>\n",
    bg.0, bg.1, bg.2
  ));

  for tri in triangles {
    let (x0, y0) = to_svg(tri.projected[0].0, tri.projected[0].1);
    let (x1, y1) = to_svg(tri.projected[1].0, tri.projected[1].1);
    let (x2, y2) = to_svg(tri.projected[2].0, tri.projected[2].1);
    let (r, g, b) = tri.color;
    svg.push_str(&format!(
      "<polygon points=\"{x0:.1},{y0:.1} {x1:.1},{y1:.1} {x2:.1},{y2:.1}\" fill=\"rgb({r},{g},{b})\" stroke=\"rgb({r},{g},{b})\" stroke-width=\"0.5\"/>\n"
    ));
  }

  svg.push_str("</svg>");
  svg
}

/// Histogram[{d1, d2, ...}] or Histogram[{d1, d2, ...}, nbins]
/// or Histogram[{d1, d2, ...}, {{e1, e2, ...}}]
pub fn histogram_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let values = extract_values(&args[0])?;
  if values.is_empty() {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }

  // Check if args[1] is a bin count, custom bin edges, or an option
  let bin_spec = if args.len() > 1 {
    let evaluated = evaluate_expr_to_expr(&args[1])?;
    // Check for {{e1, e2, ...}} — a list containing a single list of bin edges
    if let Expr::List(outer) = &evaluated {
      if outer.len() == 1 {
        if let Expr::List(inner) = &outer[0] {
          let edges: Vec<f64> =
            inner.iter().filter_map(try_eval_to_f64).collect();
          if edges.len() >= 2 {
            Some(BinSpec::Edges(edges))
          } else {
            None
          }
        } else {
          None
        }
      } else {
        None
      }
    } else if let Some(f) = try_eval_to_f64(&evaluated) {
      let n = f as usize;
      if n > 0 { Some(BinSpec::Count(n)) } else { None }
    } else {
      None
    }
  } else {
    None
  };

  let opts = parse_chart_options(args);
  let (svg_width, svg_height, full_width) =
    (opts.svg_width, opts.svg_height, opts.full_width);

  let svg = generate_histogram_svg(
    &values, bin_spec, svg_width, svg_height, full_width,
  )?;
  Ok(crate::graphics_result(svg))
}

/// Compute box-whisker statistics for a sorted dataset.
/// Uses Mathematica's quartile interpolation:
///   Q1 at position (n+2)/4, Q2 at (n+1)/2, Q3 at (3n+2)/4 (1-indexed).
fn box_stats(sorted: &[f64]) -> (f64, f64, f64, f64, f64) {
  let n = sorted.len() as f64;
  let interp = |pos: f64| -> f64 {
    let idx = pos - 1.0; // convert to 0-indexed
    let lo = idx.floor() as usize;
    let hi = (idx.ceil() as usize).min(sorted.len() - 1);
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
  };
  let q1 = interp((n + 2.0) / 4.0);
  let median = interp((n + 1.0) / 2.0);
  let q3 = interp((3.0 * n + 2.0) / 4.0);
  (sorted[0], q1, median, q3, sorted[sorted.len() - 1])
}

/// Parse the first argument into one or more datasets (Vec<Vec<f64>>).
/// Supports:
/// - `{d1, d2, ...}` → single dataset
/// - `{{d1, d2, ...}, {d3, d4, ...}}` → multiple datasets
fn parse_datasets(arg: &Expr) -> Result<Vec<Vec<f64>>, InterpreterError> {
  let data = evaluate_expr_to_expr(arg)?;
  let items = match &data {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "BoxWhiskerChart: first argument must be a list".into(),
      ));
    }
  };

  if items.is_empty() {
    return Ok(vec![]);
  }

  // Check if the first element is a list (multiple datasets)
  if let Expr::List(_) = &items[0] {
    let mut datasets = Vec::new();
    for item in items {
      if let Expr::List(inner) = item {
        let mut vals = Vec::new();
        for v in inner {
          let ev = evaluate_expr_to_expr(v).unwrap_or(v.clone());
          if let Some(f) = try_eval_to_f64(&ev) {
            vals.push(f);
          }
        }
        if !vals.is_empty() {
          datasets.push(vals);
        }
      }
    }
    Ok(datasets)
  } else {
    // Flat list: single dataset
    let mut vals = Vec::new();
    for item in items {
      let ev = evaluate_expr_to_expr(item).unwrap_or(item.clone());
      if let Some(f) = try_eval_to_f64(&ev) {
        vals.push(f);
      }
    }
    if vals.is_empty() {
      Ok(vec![])
    } else {
      Ok(vec![vals])
    }
  }
}

/// BoxWhiskerChart[{d1, d2, ...}] or BoxWhiskerChart[{{d1,...}, {d2,...}, ...}]
pub fn box_whisker_chart_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let mut datasets = parse_datasets(&args[0])?;
  if datasets.is_empty() {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }
  let opts = parse_chart_options(args);
  let (svg_width, svg_height, full_width) =
    (opts.svg_width, opts.svg_height, opts.full_width);

  // Sort each dataset and compute stats
  let mut all_stats = Vec::new();
  let mut global_min = f64::INFINITY;
  let mut global_max = f64::NEG_INFINITY;
  for ds in &mut datasets {
    ds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let (lo, q1, med, q3, hi) = box_stats(ds);
    global_min = global_min.min(lo);
    global_max = global_max.max(hi);
    all_stats.push((lo, q1, med, q3, hi));
  }

  let v_range = global_max - global_min;
  let v_pad = if v_range.abs() < f64::EPSILON {
    1.0
  } else {
    v_range * 0.08
  };
  let v_min = global_min - v_pad;
  let v_max = global_max + v_pad;

  // Use plotters for axes
  let n = datasets.len();

  // Calculate margins based on label presence
  let s = RESOLUTION_SCALE as i32;
  let sf = RESOLUTION_SCALE as f64;
  let has_chart_labels = !opts.chart_labels.is_empty();
  let has_x_axis_label =
    opts.axes_label.as_ref().is_some_and(|(x, _)| !x.is_empty());
  let has_plot_label = opts
    .plot_label
    .as_ref()
    .is_some_and(|sl| !sl.text.is_empty());

  let top_margin = if has_plot_label {
    25 * s as u32
  } else {
    10 * s as u32
  };
  let has_rotated_labels =
    opts.chart_labels.iter().any(|l| l.rotation.abs() > 0.01);
  let label_extra = if has_rotated_labels {
    30.0 * sf
  } else if has_chart_labels {
    15.0 * sf
  } else {
    0.0
  };
  let bottom_extra =
    label_extra + if has_x_axis_label { 16.0 * sf } else { 0.0 };
  let x_label_area = 8 * RESOLUTION_SCALE + bottom_extra as u32;
  let y_label_area = 40 * RESOLUTION_SCALE;

  let margin_overrides = MarginOverrides {
    top_margin,
    x_label_area,
    y_label_area,
  };

  // One tick mark per box (centered), no labels — matches Mathematica
  let x_tick_positions: Vec<f64> = (0..n).map(|i| i as f64 + 0.5).collect();
  let area = crate::functions::plot::generate_axes_only_opts(
    (0.0, n as f64),
    (v_min, v_max),
    svg_width,
    svg_height,
    full_width,
    Some(&x_tick_positions),
    Some(&margin_overrides),
  )?;

  let plot_x0 = area.plot_x0;
  let plot_y0 = area.plot_y0;
  let plot_w = area.plot_w;
  let plot_h = area.plot_h;

  let mut svg = area.svg;
  if let Some(pos) = svg.rfind("</svg>") {
    svg.truncate(pos);
  }

  let map_y =
    |v: f64| -> f64 { plot_y0 + (v_max - v) / (v_max - v_min) * plot_h };
  let slot_w = plot_w / n as f64;
  let stroke_w = (area.render_width as f64 / 1000.0 * 1.5).max(1.0);

  for (i, &(lo, q1, med, q3, hi)) in all_stats.iter().enumerate() {
    let (r, g, b) = if !opts.chart_style.is_empty() {
      let c = &opts.chart_style[i % opts.chart_style.len()];
      (
        (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
        (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
        (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
      )
    } else {
      PLOT_COLORS[0]
    };
    let cx = plot_x0 + (i as f64 + 0.5) * slot_w;
    let box_half_w = slot_w * 0.3;
    let cap_half_w = box_half_w * 0.5;

    let (_bg, _dg, _lg, whisker_fill, _tf) =
      crate::functions::plot::plot_theme();
    let whisker_color = whisker_fill;
    // Whisker line (vertical)
    svg.push_str(&format!(
      "<line x1=\"{cx:.1}\" y1=\"{:.1}\" x2=\"{cx:.1}\" y2=\"{:.1}\" stroke=\"{whisker_color}\" stroke-width=\"{stroke_w:.1}\"/>\n",
      map_y(hi), map_y(lo)
    ));
    // Top whisker cap
    svg.push_str(&format!(
      "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{whisker_color}\" stroke-width=\"{stroke_w:.1}\"/>\n",
      cx - cap_half_w, map_y(hi), cx + cap_half_w, map_y(hi)
    ));
    // Bottom whisker cap
    svg.push_str(&format!(
      "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{whisker_color}\" stroke-width=\"{stroke_w:.1}\"/>\n",
      cx - cap_half_w, map_y(lo), cx + cap_half_w, map_y(lo)
    ));
    // Box (Q1 to Q3)
    let box_top = map_y(q3);
    let box_bot = map_y(q1);
    let box_h = box_bot - box_top;
    svg.push_str(&format!(
      "<rect x=\"{:.1}\" y=\"{box_top:.1}\" width=\"{:.1}\" height=\"{box_h:.1}\" fill=\"rgb({r},{g},{b})\" stroke=\"{whisker_color}\" stroke-width=\"{stroke_w:.1}\"/>\n",
      cx - box_half_w, box_half_w * 2.0
    ));
    // Median line
    svg.push_str(&format!(
      "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"white\" stroke-width=\"{:.1}\"/>\n",
      cx - box_half_w, map_y(med), cx + box_half_w, map_y(med), stroke_w * 1.5
    ));
  }

  // Insert label SVG elements before </svg>
  let axis_y = plot_y0 + plot_h;
  let font_size = sf * 11.0;
  let title_font_size = sf * 13.0;
  let margin_left = 10.0 * sf;
  let mut labels_svg = String::new();

  let (_bg, _dg, _lg, chart_label_fill, chart_title_fill) =
    crate::functions::plot::plot_theme();

  // ChartLabels: centered below each box
  if has_chart_labels {
    for (i, label) in opts.chart_labels.iter().enumerate().take(n) {
      let cx = plot_x0 + (i as f64 + 0.5) * slot_w;
      // Mathematica Rotate is counterclockwise-positive; SVG is clockwise-positive
      let svg_rotation_deg = -label.rotation.to_degrees();
      let is_rotated = svg_rotation_deg.abs() > 0.01;
      if is_rotated {
        let char_width_estimate = font_size * 0.6;
        let half_text_w = label.text.len() as f64 * char_width_estimate / 2.0;
        let sin_a = svg_rotation_deg.to_radians().sin().abs();
        let offset = half_text_w * sin_a + font_size * 0.5;
        let ay = axis_y + offset;
        labels_svg.push_str(&format!(
          "<text x=\"{cx:.1}\" y=\"{ay:.1}\" text-anchor=\"middle\" \
           font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
           fill=\"{chart_label_fill}\" transform=\"rotate({svg_rotation_deg:.1},{cx:.1},{ay:.1})\">{}</text>\n",
          html_escape(&label.text)
        ));
      } else {
        let ly = axis_y + font_size * 1.5;
        labels_svg.push_str(&format!(
          "<text x=\"{cx:.1}\" y=\"{ly:.1}\" text-anchor=\"middle\" \
           font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
           fill=\"{chart_label_fill}\">{}</text>\n",
          html_escape(&label.text)
        ));
      }
    }
  }

  // AxesLabel / FrameLabel: x-axis label centered below, y-axis label rotated on left
  if let Some((x_label, y_label)) = &opts.axes_label {
    if !x_label.is_empty() {
      let cx = plot_x0 + plot_w / 2.0;
      let base_y = axis_y
        + if has_chart_labels {
          font_size * 1.5 + font_size * 1.3
        } else {
          font_size * 1.5
        };
      labels_svg.push_str(&format!(
        "<text x=\"{cx:.1}\" y=\"{base_y:.1}\" text-anchor=\"middle\" \
         font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
         fill=\"{chart_label_fill}\">{}</text>\n",
        html_escape(x_label)
      ));
    }
    if !y_label.is_empty() {
      let cy = plot_y0 + plot_h / 2.0;
      let lx = margin_left + font_size * 0.8;
      labels_svg.push_str(&format!(
        "<text x=\"{lx:.1}\" y=\"{cy:.1}\" text-anchor=\"middle\" \
         font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
         fill=\"{chart_label_fill}\" transform=\"rotate(-90,{lx:.1},{cy:.1})\">{}</text>\n",
        html_escape(y_label)
      ));
    }
  }

  // PlotLabel: centered above the chart
  if let Some(sl) = &opts.plot_label
    && !sl.text.is_empty()
  {
    let cx = plot_x0 + plot_w / 2.0;
    let ty = top_margin as f64 - title_font_size * 0.5;
    let fs = sl.font_size.map(|f| f * sf).unwrap_or(title_font_size);
    let fill = sl
      .color
      .as_ref()
      .map(|c| c.to_svg_rgb())
      .unwrap_or_else(|| chart_title_fill.to_string());
    let mut style_attrs = String::new();
    if sl.bold {
      style_attrs.push_str(" font-weight=\"bold\"");
    }
    if sl.italic {
      style_attrs.push_str(" font-style=\"italic\"");
    }
    labels_svg.push_str(&format!(
      "<text x=\"{cx:.1}\" y=\"{ty:.1}\" text-anchor=\"middle\" \
         font-family=\"sans-serif\" font-size=\"{fs:.0}\" \
         fill=\"{fill}\"{style_attrs}>{}</text>\n",
      html_escape(&sl.text)
    ));
  }

  svg.push_str(&labels_svg);
  svg.push_str("</svg>");
  Ok(crate::graphics_result(svg))
}

/// BubbleChart[{{x,y,z}, ...}]
pub fn bubble_chart_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let data = evaluate_expr_to_expr(&args[0])?;
  let items = match &data {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "BubbleChart: first argument must be a list of {x, y, z} triples"
          .into(),
      ));
    }
  };

  let mut triples = Vec::new();
  for item in items {
    if let Expr::List(inner) = item
      && inner.len() >= 3
    {
      let x = try_eval_to_f64(
        &evaluate_expr_to_expr(&inner[0]).unwrap_or(inner[0].clone()),
      );
      let y = try_eval_to_f64(
        &evaluate_expr_to_expr(&inner[1]).unwrap_or(inner[1].clone()),
      );
      let z = try_eval_to_f64(
        &evaluate_expr_to_expr(&inner[2]).unwrap_or(inner[2].clone()),
      );
      if let (Some(x), Some(y), Some(z)) = (x, y, z) {
        triples.push((x, y, z));
      }
    }
  }

  if triples.is_empty() {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }

  let opts = parse_chart_options(args);
  let (svg_width, svg_height, full_width) =
    (opts.svg_width, opts.svg_height, opts.full_width);
  let w = svg_width as f64;
  let h = svg_height as f64;
  let margin = 50.0;
  let plot_w = w - 2.0 * margin;
  let plot_h = h - 2.0 * margin;

  let x_min = triples.iter().map(|t| t.0).fold(f64::INFINITY, f64::min);
  let x_max = triples
    .iter()
    .map(|t| t.0)
    .fold(f64::NEG_INFINITY, f64::max);
  let y_min = triples.iter().map(|t| t.1).fold(f64::INFINITY, f64::min);
  let y_max = triples
    .iter()
    .map(|t| t.1)
    .fold(f64::NEG_INFINITY, f64::max);
  let z_max = triples.iter().map(|t| t.2.abs()).fold(0.0_f64, f64::max);
  let max_radius = 20.0;

  let x_range = if (x_max - x_min).abs() < f64::EPSILON {
    1.0
  } else {
    x_max - x_min
  };
  let y_range = if (y_max - y_min).abs() < f64::EPSILON {
    1.0
  } else {
    y_max - y_min
  };

  let mut svg = svg_header(svg_width, svg_height, full_width);

  for (i, &(x, y, z)) in triples.iter().enumerate() {
    let (r, g, b) = PLOT_COLORS[i % PLOT_COLORS.len()];
    let sx = margin + ((x - x_min) / x_range) * plot_w;
    let sy = margin + ((y_max - y) / y_range) * plot_h;
    let radius = if z_max > 0.0 {
      (z.abs() / z_max) * max_radius
    } else {
      5.0
    };
    let radius = radius.max(2.0);
    svg.push_str(&format!(
      "<circle cx=\"{sx:.1}\" cy=\"{sy:.1}\" r=\"{radius:.1}\" fill=\"rgb({r},{g},{b})\" fill-opacity=\"0.7\"/>\n"
    ));
  }

  svg.push_str("</svg>");
  Ok(crate::graphics_result(svg))
}

/// SectorChart[{{angle, radius}, ...}] - like PieChart but with variable radius
pub fn sector_chart_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let data = evaluate_expr_to_expr(&args[0])?;
  let items = match &data {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "SectorChart: first argument must be a list".into(),
      ));
    }
  };

  let mut sectors: Vec<(f64, f64)> = Vec::new();
  for item in items {
    if let Expr::List(pair) = item {
      if pair.len() >= 2 {
        let a = try_eval_to_f64(
          &evaluate_expr_to_expr(&pair[0]).unwrap_or(pair[0].clone()),
        );
        let r = try_eval_to_f64(
          &evaluate_expr_to_expr(&pair[1]).unwrap_or(pair[1].clone()),
        );
        if let (Some(a), Some(r)) = (a, r) {
          sectors.push((a, r));
        }
      }
    } else if let Some(v) =
      try_eval_to_f64(&evaluate_expr_to_expr(item).unwrap_or(item.clone()))
    {
      sectors.push((v, 1.0));
    }
  }

  if sectors.is_empty() {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }

  let opts = parse_chart_options(args);
  let (svg_width, svg_height, full_width) =
    (opts.svg_width, opts.svg_height, opts.full_width);
  let w = svg_width as f64;
  let h = svg_height as f64;
  let cx = w / 2.0;
  let cy = h / 2.0;
  let max_radius = (w.min(h) / 2.0) * 0.85;
  let total_angle: f64 = sectors.iter().map(|s| s.0).sum();
  let r_max = sectors.iter().map(|s| s.1).fold(0.0_f64, f64::max);
  let r_max = if r_max <= 0.0 { 1.0 } else { r_max };

  let mut svg = svg_header(svg_width, svg_height, full_width);

  let mut start_angle = -std::f64::consts::FRAC_PI_2;
  for (i, &(angle_val, radius_val)) in sectors.iter().enumerate() {
    let (r, g, b) = PLOT_COLORS[i % PLOT_COLORS.len()];
    let sweep = 2.0 * std::f64::consts::PI * angle_val / total_angle;
    let sector_r = (radius_val / r_max) * max_radius;
    let end_angle = start_angle + sweep;

    let x1 = cx + sector_r * start_angle.cos();
    let y1 = cy + sector_r * start_angle.sin();
    let x2 = cx + sector_r * end_angle.cos();
    let y2 = cy + sector_r * end_angle.sin();
    let large_arc = if sweep > std::f64::consts::PI { 1 } else { 0 };

    let tooltip = format!(
      "{{{}, {}}}",
      format_chart_value(angle_val),
      format_chart_value(radius_val)
    );
    svg.push_str(&format!(
      "<path d=\"M{cx:.2},{cy:.2} L{x1:.2},{y1:.2} A{sector_r:.2},{sector_r:.2} 0 {large_arc},1 {x2:.2},{y2:.2} Z\" \
       fill=\"rgb({r},{g},{b})\" stroke=\"white\" stroke-width=\"1\"><title>{tooltip}</title></path>\n"
    ));

    start_angle = end_angle;
  }

  svg.push_str("</svg>");
  Ok(crate::graphics_result(svg))
}

/// Convert a date expression to AbsoluteTime (seconds since 1900-01-01).
/// Handles: date lists ({y}, {y,m}, {y,m,d}, {y,m,d,h,min,sec}),
/// DateObject[{...}], and raw numeric AbsoluteTime values.
fn date_expr_to_absolute_time(expr: &Expr) -> Option<f64> {
  let evaluated = evaluate_expr_to_expr(expr).unwrap_or(expr.clone());

  // Try as a raw numeric value (already AbsoluteTime)
  if let Some(t) = try_eval_to_f64(&evaluated) {
    return Some(t);
  }

  // Try extracting date components from list or DateObject
  if let Some(components) =
    crate::functions::datetime_ast::extract_date_components(&evaluated)
  {
    if components.is_empty() {
      return None;
    }
    let year = components[0] as i64;
    let month = if components.len() > 1 {
      components[1] as i64
    } else {
      1
    };
    let day = if components.len() > 2 {
      components[2] as i64
    } else {
      1
    };
    let hour = if components.len() > 3 {
      components[3] as i64
    } else {
      0
    };
    let minute = if components.len() > 4 {
      components[4] as i64
    } else {
      0
    };
    let second = if components.len() > 5 {
      components[5]
    } else {
      0.0
    };
    return Some(crate::functions::datetime_ast::date_to_absolute_seconds(
      year, month, day, hour, minute, second,
    ));
  }

  None
}

/// DateListPlot[{{date, y}, ...}] - plots data with date-valued x-axis.
/// Dates can be date lists ({y,m,d}), DateObject[{...}], or AbsoluteTime values.
/// Multiple datasets are supported: DateListPlot[{data1, data2, ...}].
pub fn date_list_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let data = evaluate_expr_to_expr(&args[0])?;
  let items = match &data {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "DateListPlot: first argument must be a list".into(),
      ));
    }
  };

  // Detect whether this is multiple datasets or a single dataset.
  // Multiple datasets: {{pair1, pair2, ...}, {pair3, pair4, ...}}
  // Single dataset: {{date1, y1}, {date2, y2}, ...}
  let datasets: Vec<&[Expr]> = if !items.is_empty()
    && items.iter().all(|item| {
      if let Expr::List(inner) = item {
        // A dataset is a list of pairs (each pair is a list of length 2
        // where first element is a date)
        !inner.is_empty()
          && inner
            .iter()
            .all(|sub| matches!(sub, Expr::List(p) if p.len() >= 2))
      } else {
        false
      }
    }) {
    // Multiple datasets
    items
      .iter()
      .map(|item| {
        if let Expr::List(inner) = item {
          inner.as_slice()
        } else {
          unreachable!()
        }
      })
      .collect()
  } else {
    // Single dataset
    vec![items.as_slice()]
  };

  let mut all_series = Vec::new();
  for dataset in &datasets {
    let mut points = Vec::new();
    for item in *dataset {
      if let Expr::List(pair) = item
        && pair.len() >= 2
      {
        let x = date_expr_to_absolute_time(&pair[0]);
        let y = try_eval_to_f64(
          &evaluate_expr_to_expr(&pair[1]).unwrap_or(pair[1].clone()),
        );
        if let (Some(x), Some(y)) = (x, y) {
          points.push((x, y));
        }
      }
    }
    if !points.is_empty() {
      all_series.push(points);
    }
  }

  if all_series.is_empty() {
    // Return unevaluated (like Wolfram does for invalid data)
    return Ok(Expr::FunctionCall {
      name: "DateListPlot".to_string(),
      args: args.to_vec(),
    });
  }

  let chart_opts = parse_chart_options(args);

  // Compute ranges across all series
  let mut x_min = f64::INFINITY;
  let mut x_max = f64::NEG_INFINITY;
  let mut y_min = f64::INFINITY;
  let mut y_max = f64::NEG_INFINITY;
  for series in &all_series {
    for &(x, y) in series {
      x_min = x_min.min(x);
      x_max = x_max.max(x);
      y_min = y_min.min(y);
      y_max = y_max.max(y);
    }
  }
  let yp = (y_max - y_min) * 0.04;
  let yp = if yp.abs() < f64::EPSILON { 1.0 } else { yp };

  // Compute x range with padding
  let xp = (x_max - x_min) * 0.04;
  let xp = if xp.abs() < f64::EPSILON { 86400.0 } else { xp };
  let x_range_min = x_min - xp;
  let x_range_max = x_max + xp;

  // Compute nice date tick positions
  let date_ticks =
    crate::functions::plot::generate_date_ticks(x_range_min, x_range_max);

  // Use date_axis mode which suppresses x labels from plotters;
  // we inject our own date labels afterwards.
  let plot_opts = crate::functions::plot::PlotOptions {
    svg_width: chart_opts.svg_width,
    svg_height: chart_opts.svg_height,
    full_width: chart_opts.full_width,
    date_axis: true,
    ..Default::default()
  };

  let svg = crate::functions::plot::generate_svg_with_filling(
    &all_series,
    (x_range_min, x_range_max),
    (y_min - yp, y_max + yp),
    &plot_opts,
  )?;

  // Inject date tick labels into the SVG
  let svg =
    inject_date_labels(&svg, &date_ticks, x_range_min, x_range_max, &plot_opts);

  Ok(crate::graphics_result(svg))
}

/// Inject date tick labels and tick marks into an SVG generated by plotters.
/// Since plotters can't place ticks at exact date boundaries, we add them manually.
fn inject_date_labels(
  svg: &str,
  date_ticks: &[f64],
  x_min: f64,
  x_max: f64,
  opts: &crate::functions::plot::PlotOptions,
) -> String {
  use crate::functions::plot::{RESOLUTION_SCALE, format_date_tick};

  if date_ticks.is_empty() {
    return svg.to_string();
  }

  let sf = RESOLUTION_SCALE as f64;
  let render_width = opts.svg_width as f64 * sf;
  let render_height = opts.svg_height as f64 * sf;

  // Estimate plot area margins (matching generate_svg_with_options logic)
  // y_label_area = 65 * RESOLUTION_SCALE, margin_left = 10 * RESOLUTION_SCALE
  let margin_left = 10.0 * sf;
  let y_label_area = 65.0 * sf;
  let margin_right = 10.0 * sf;
  let margin_bottom = 10.0 * sf;
  let x_label_area = 40.0 * sf;

  let plot_left = margin_left + y_label_area;
  let plot_right = render_width - margin_right;
  let plot_bottom = render_height - margin_bottom - x_label_area;
  let plot_width = plot_right - plot_left;

  let font_size = sf * 18.0;
  let tick_len = 4.0 * sf;

  let mut extra = String::new();
  for &t in date_ticks {
    let frac = (t - x_min) / (x_max - x_min);
    let px = plot_left + frac * plot_width;
    let label = html_escape(&format_date_tick(t));

    // Tick mark
    extra.push_str(&format!(
      "<polyline fill=\"none\" opacity=\"1\" stroke=\"#666666\" \
       stroke-width=\"{}\" points=\"{:.0},{:.0} {:.0},{:.0}\"/>",
      RESOLUTION_SCALE,
      px,
      plot_bottom,
      px,
      plot_bottom + tick_len,
    ));

    // Label
    extra.push_str(&format!(
      "<text x=\"{:.0}\" y=\"{:.0}\" dy=\"0.76em\" text-anchor=\"middle\" \
       font-family=\"sans-serif\" font-size=\"{font_size}\" \
       opacity=\"1\" fill=\"#666666\">{label}</text>",
      px,
      plot_bottom + tick_len + 2.0,
    ));
  }

  // Insert before closing </svg> tag
  if let Some(pos) = svg.rfind("</svg>") {
    let mut result = svg[..pos].to_string();
    result.push_str(&extra);
    result.push_str("</svg>");
    result
  } else {
    svg.to_string()
  }
}

/// Escape special HTML characters in text content.
fn html_escape(s: &str) -> String {
  s.replace('&', "&amp;")
    .replace('<', "&lt;")
    .replace('>', "&gt;")
    .replace('"', "&quot;")
}

/// WordCloud[{"word1", "word2", ...}] or WordCloud[{1, 2, 3, ...}]
pub fn word_cloud_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use std::collections::HashMap;

  // Extract list of strings from first argument
  let data = evaluate_expr_to_expr(&args[0])?;
  let items = match &data {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "WordCloud: first argument must be a list".into(),
      ));
    }
  };

  let mut words: Vec<String> = Vec::new();
  for item in items {
    let ev = evaluate_expr_to_expr(item).unwrap_or(item.clone());
    match &ev {
      Expr::String(s) => words.push(s.clone()),
      _ => words.push(crate::syntax::expr_to_string(&ev)),
    }
  }

  if words.is_empty() {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }

  // Count word frequencies
  let mut freq: HashMap<String, usize> = HashMap::new();
  for w in &words {
    *freq.entry(w.clone()).or_insert(0) += 1;
  }

  // Sort by frequency (descending), then alphabetically for stability
  let mut sorted_words: Vec<(String, usize)> = freq.into_iter().collect();
  sorted_words.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

  // Parse options (ImageSize)
  let opts = parse_chart_options(args);
  let (svg_width, svg_height, full_width) =
    (opts.svg_width, opts.svg_height, opts.full_width);
  let w = svg_width as f64;
  let h = svg_height as f64;
  let cx = w / 2.0;
  let cy = h / 2.0;

  // Map frequencies to font sizes (linear scaling)
  let max_freq = sorted_words[0].1 as f64;
  let min_freq = sorted_words.last().unwrap().1 as f64;
  let min_font = 12.0_f64;
  let max_font = (h * 0.2).min(60.0);

  let font_size_for = |count: usize| -> f64 {
    if (max_freq - min_freq).abs() < f64::EPSILON {
      (min_font + max_font) / 2.0
    } else {
      min_font
        + (count as f64 - min_freq) / (max_freq - min_freq)
          * (max_font - min_font)
    }
  };

  // Placed word bounding boxes: (x_min, y_min, x_max, y_max)
  let mut placed: Vec<(f64, f64, f64, f64)> = Vec::new();

  struct PlacedWord {
    x: f64,
    y: f64,
    font_size: f64,
    text: String,
    color_idx: usize,
    rotated: bool,
  }
  let mut placed_words: Vec<PlacedWord> = Vec::new();

  let char_width_factor = 0.6;

  for (i, (word, count)) in sorted_words.iter().enumerate() {
    let font_size = font_size_for(*count);
    let rotated = i % 3 == 1; // every third word rotated

    // Estimate bounding box dimensions
    let text_w = word.len() as f64 * char_width_factor * font_size;
    let text_h = font_size;
    let (box_w, box_h) = if rotated {
      (text_h, text_w) // swap for rotation
    } else {
      (text_w, text_h)
    };

    // Try placing using Archimedean spiral from center
    let max_steps = 10000;
    let a = 1.5; // spiral spacing

    let mut placed_ok = false;
    for s in 0..max_steps {
      let t = s as f64 * 0.04;
      let x = cx + a * t * t.cos();
      let y = cy + a * t * t.sin() * 0.6; // compress vertically to fit aspect ratio

      // Bounding box centered at (x, y)
      let x_min = x - box_w / 2.0;
      let y_min = y - box_h / 2.0;
      let x_max = x + box_w / 2.0;
      let y_max = y + box_h / 2.0;

      // Check bounds
      if x_min < 2.0 || y_min < 2.0 || x_max > w - 2.0 || y_max > h - 2.0 {
        continue;
      }

      // Check collision with already placed words
      let collides = placed.iter().any(|&(px_min, py_min, px_max, py_max)| {
        x_min < px_max && x_max > px_min && y_min < py_max && y_max > py_min
      });

      if !collides {
        placed.push((x_min, y_min, x_max, y_max));
        placed_words.push(PlacedWord {
          x,
          y,
          font_size,
          text: word.clone(),
          color_idx: i,
          rotated,
        });
        placed_ok = true;
        break;
      }
    }

    if !placed_ok {
      continue; // skip words that can't be placed
    }
  }

  // Generate SVG
  let mut svg = svg_header(svg_width, svg_height, full_width);

  for pw in &placed_words {
    let (r, g, b) = PLOT_COLORS[pw.color_idx % PLOT_COLORS.len()];
    let transform = if pw.rotated {
      format!(
        " transform=\"translate({:.1},{:.1}) rotate(-90)\"",
        pw.x, pw.y
      )
    } else {
      format!(" transform=\"translate({:.1},{:.1})\"", pw.x, pw.y)
    };
    svg.push_str(&format!(
      "<text{transform} font-size=\"{:.1}\" fill=\"rgb({r},{g},{b})\" \
       font-family=\"sans-serif\" text-anchor=\"middle\" \
       dominant-baseline=\"central\">{}</text>\n",
      pw.font_size,
      html_escape(&pw.text)
    ));
  }

  svg.push_str("</svg>");
  Ok(crate::graphics_result(svg))
}
