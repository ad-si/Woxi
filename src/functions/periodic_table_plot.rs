use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::element_data::{
  ElementLayout, element_phase, elements_layout, resolve_atomic_number,
};
use crate::functions::plot::{DEFAULT_HEIGHT, DEFAULT_WIDTH, parse_image_size};
use crate::syntax::Expr;

/// Side length of a single element cell, in SVG grid units. The whole plot
/// is rendered in these units and scaled to the requested pixel size via the
/// SVG `viewBox`, so all layout math stays in simple integers.
const CELL: f64 = 100.0;
/// Inner gap between a cell's border and the coloured tile.
const PAD: f64 = 6.0;
/// Extra horizontal gap between groups 2 and 3 (columns >= 2 shift right
/// by this much) hosting the `*` / `**` series-insertion markers.
const GUTTER: f64 = 36.0;
/// Default on-screen width in pixels (matches the periodic table's 18-column
/// aspect ratio, not the generic plot default).
const DEFAULT_PTP_WIDTH: u32 = 540;

/// Grid position of a single element, in cell units. `(0, 0)` is the
/// top-left cell of the main table (Hydrogen's row / group 1).
struct Cell {
  col: f64,
  row: f64,
}

/// Map an element to its `(col, row)` position in the standard periodic
/// table layout. Main-group and d-block elements sit at `(group, period)`;
/// the f-block (lanthanides 57–70 and actinides 89–102) is pulled out into
/// two rows below the main table, aligned under group 3. Small `*` / `**`
/// markers in the gutter between groups 2 and 3 show where the detached
/// rows insert.
fn element_cell(elem: &ElementLayout) -> Cell {
  match elem.group {
    Some(g) => Cell {
      col: (g - 1) as f64,
      row: (elem.period - 1) as f64,
    },
    None => {
      // f-block. Lanthanides (57–70) and actinides (89–102) each occupy a
      // separate row below the main table, offset by half a row as a gap.
      let (base_z, extra_row) = if elem.atomic_number <= 71 {
        (57, 7.6)
      } else {
        (89, 8.6)
      };
      Cell {
        col: 2.0 + (elem.atomic_number - base_z) as f64,
        row: extra_row,
      }
    }
  }
}

/// Colour for an element tile based on its block, adapted to the theme.
fn block_color(block: &str, dark: bool) -> &'static str {
  if dark {
    match block {
      "s" => "#7a3b3b",
      "p" => "#3b5a7a",
      "d" => "#3b7a5a",
      "f" => "#6a4a7a",
      _ => "#555555",
    }
  } else {
    match block {
      "s" => "#f3c9c9",
      "p" => "#c9dcf3",
      "d" => "#c9f3d9",
      "f" => "#e2c9f3",
      _ => "#dddddd",
    }
  }
}

/// Tile colours for the `"Phase"` property plot: (gas, solid, liquid,
/// missing). The light palette matches wolframscript's legend swatches.
fn phase_palette(
  dark: bool,
) -> (&'static str, &'static str, &'static str, &'static str) {
  if dark {
    ("#3b5a7a", "#7a5f3b", "#4a7a3b", "#2a2a2a")
  } else {
    ("#7ebbdd", "#f6c06d", "#a2cc79", "#e6e6e6")
  }
}

/// Colour for an element tile in the `"Phase"` plot.
fn phase_color(phase: Option<&str>, dark: bool) -> &'static str {
  let (gas, solid, liquid, missing) = phase_palette(dark);
  match phase {
    Some("Gas") => gas,
    Some("Liquid") => liquid,
    Some("Solid") => solid,
    _ => missing,
  }
}

/// Theme-dependent colours: (background, faded tile, text, faded text, border).
fn theme_colors(
  dark: bool,
) -> (
  &'static str,
  &'static str,
  &'static str,
  &'static str,
  &'static str,
) {
  if dark {
    ("#1a1a1a", "#2a2a2a", "#e0e0e0", "#666666", "#444444")
  } else {
    ("#ffffff", "#f0f0f0", "#222222", "#bbbbbb", "#999999")
  }
}

/// Horizontal position of a cell's left edge: columns right of group 2
/// shift by the gutter that hosts the series-insertion markers.
fn cell_x(col: f64) -> f64 {
  col * CELL + if col >= 2.0 { GUTTER } else { 0.0 }
}

/// Draw `count` vertically stacked asterisks centred in the group 2/3
/// gutter, in row `row` — the insertion markers linking the gutter to the
/// detached lanthanide/actinide rows.
fn push_series_marker(body: &mut String, row: f64, count: usize, color: &str) {
  let cx = 2.0 * CELL + GUTTER / 2.0;
  let cy = (row + 0.5) * CELL;
  let step = 24.0;
  let top = cy - step * (count as f64 - 1.0) / 2.0;
  for i in 0..count {
    // The asterisk glyph's ink sits high in the em box; push the baseline
    // below the target centre so the mark looks centred.
    body.push_str(&format!(
      "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" \
       font-family=\"sans-serif\" font-size=\"40\" fill=\"{}\">*</text>\n",
      cx,
      top + i as f64 * step + 14.0,
      color,
    ));
  }
}

/// Draw the phase legend under the table: a "phase at STP" title pill
/// above a centred row of colour swatches labelled gas / solid / liquid /
/// — (phase not available).
fn push_phase_legend(body: &mut String, vb_w: f64, dark: bool) {
  let (gas, solid, liquid, missing) = phase_palette(dark);
  let (_, _, text, _, border) = theme_colors(dark);
  let pill_bg = if dark { "#333333" } else { "#ececec" };

  // Title pill, centred.
  let (pill_w, pill_h) = (300.0, 60.0);
  let cx = vb_w / 2.0;
  let pill_y = 9.85 * CELL;
  body.push_str(&format!(
    "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" \
     rx=\"12\" fill=\"{}\"/>\n",
    cx - pill_w / 2.0,
    pill_y,
    pill_w,
    pill_h,
    pill_bg,
  ));
  body.push_str(&format!(
    "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" \
     font-family=\"sans-serif\" font-size=\"36\" fill=\"{}\">phase at \
     STP</text>\n",
    cx,
    pill_y + 42.0,
    text,
  ));

  // Swatch row, laid out as one centred group.
  let entries: [(&str, &str); 4] = [
    (gas, "gas"),
    (solid, "solid"),
    (liquid, "liquid"),
    (missing, "\u{2014}"),
  ];
  const SWATCH: f64 = 44.0; // swatch side length
  const GAP: f64 = 14.0; // between a swatch and its label
  const SPACING: f64 = 60.0; // between entries
  const CHAR_W: f64 = 21.0; // approximate glyph width at font-size 40
  let label_w = |label: &str| {
    if label == "\u{2014}" {
      40.0 // the em dash is a full em wide
    } else {
      label.chars().count() as f64 * CHAR_W
    }
  };
  let total: f64 = entries
    .iter()
    .map(|(_, l)| SWATCH + GAP + label_w(l))
    .sum::<f64>()
    + SPACING * (entries.len() - 1) as f64;
  let mut x = (vb_w - total) / 2.0;
  let row_y = 10.7 * CELL;
  for (color, label) in entries {
    body.push_str(&format!(
      "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" \
       rx=\"6\" fill=\"{}\" stroke=\"{}\" stroke-width=\"1.5\"/>\n",
      x, row_y, SWATCH, SWATCH, color, border,
    ));
    body.push_str(&format!(
      "<text x=\"{:.1}\" y=\"{:.1}\" font-family=\"sans-serif\" \
       font-size=\"40\" fill=\"{}\">{}</text>\n",
      x + SWATCH + GAP,
      row_y + SWATCH / 2.0 + 14.0,
      text,
      label,
    ));
    x += SWATCH + GAP + label_w(label) + SPACING;
  }
}

/// Escape the small set of characters that matter inside SVG text nodes.
fn escape_xml(s: &str) -> String {
  s.replace('&', "&amp;")
    .replace('<', "&lt;")
    .replace('>', "&gt;")
}

/// What the (optional) first positional argument asks for.
enum HighlightSpec {
  /// No element specification: render the full table.
  Full,
  /// Highlight exactly these atomic numbers and fade the rest.
  Highlight(std::collections::HashSet<i128>),
  /// Color each tile by the element's phase at STP and attach a
  /// `SwatchLegend` (the `"Phase"` property specification).
  Phase,
  /// Not a form Woxi renders; echo the call unevaluated (any diagnostic
  /// message has already been emitted).
  Unevaluated,
}

/// Resolve an `Entity["Element", spec]` expression to its atomic number.
fn element_entity_atomic_number(expr: &Expr) -> Option<i128> {
  if let Expr::FunctionCall { name, args } = expr
    && name == "Entity"
    && args.len() == 2
    && matches!(&args[0], Expr::String(s) if s == "Element")
  {
    resolve_atomic_number(&args[1])
  } else {
    None
  }
}

/// Interpret the first positional argument as a highlight specification.
/// Like wolframscript, only `Entity["Element", …]` forms — single or in a
/// list — select elements. A bare string names an `EntityProperty`
/// (property coloring, which Woxi does not implement); a string that names
/// an ELEMENT is therefore never a valid property and draws the `elmntav`
/// message. Any other expression draws the `inpt` message. Lists with
/// non-entity members stay unevaluated without a message.
fn parse_highlights(first: Option<&Expr>) -> HighlightSpec {
  let Some(first) = first else {
    return HighlightSpec::Full;
  };
  match first {
    Expr::List(items) => {
      let mut set = std::collections::HashSet::new();
      for item in items.iter() {
        match element_entity_atomic_number(item) {
          Some(z) => {
            set.insert(z);
          }
          None => return HighlightSpec::Unevaluated,
        }
      }
      HighlightSpec::Highlight(set)
    }
    other => {
      if let Some(z) = element_entity_atomic_number(other) {
        return HighlightSpec::Highlight(std::iter::once(z).collect());
      }
      match other {
        Expr::String(s) => {
          if s == "Phase" {
            return HighlightSpec::Phase;
          }
          if resolve_atomic_number(other).is_some() {
            crate::emit_message_to_stdout(&format!(
              "PeriodicTablePlot::elmntav: \"{}\" is not an available \
               \"Element\" property.",
              s
            ));
          }
          HighlightSpec::Unevaluated
        }
        // EntityProperty["Element", "Phase"] is the entity form of the
        // "Phase" property specification.
        Expr::FunctionCall { name, args }
          if name == "EntityProperty"
            && args.len() == 2
            && matches!(&args[0], Expr::String(s) if s == "Element")
            && matches!(&args[1], Expr::String(s) if s == "Phase") =>
        {
          HighlightSpec::Phase
        }
        // An unresolvable entity, or property coloring Woxi does not
        // implement: leave unevaluated without a (wrong) message.
        Expr::FunctionCall { name, .. }
          if name == "Entity" || name == "EntityProperty" =>
        {
          HighlightSpec::Unevaluated
        }
        _ => {
          crate::emit_message_to_stdout(&format!(
            "PeriodicTablePlot::inpt: Input {} should be an \"Element\" \
             Entity or EntityProperty.",
            crate::syntax::expr_to_output(other)
          ));
          HighlightSpec::Unevaluated
        }
      }
    }
  }
}

/// PeriodicTablePlot[] / PeriodicTablePlot[{elems…}] / PeriodicTablePlot[elem]
/// with optional `ImageSize` option.
pub fn periodic_table_plot_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let mut svg_width = DEFAULT_PTP_WIDTH;
  let mut full_width = false;
  for opt in args {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && let Expr::Identifier(name) = pattern.as_ref()
      && name == "ImageSize"
      && let Some((w, _h, fw)) =
        parse_image_size(replacement, DEFAULT_WIDTH, DEFAULT_HEIGHT)
    {
      svg_width = w;
      full_width = fw;
    }
  }

  // The first non-Rule argument (if any) is the highlight specification;
  // evaluate it so e.g. ElementData["Fe"] resolves to its Entity form.
  let spec_idx = args.iter().position(|a| !matches!(a, Expr::Rule { .. }));
  let spec = spec_idx.map(|i| {
    evaluate_expr_to_expr(&args[i]).unwrap_or_else(|_| args[i].clone())
  });
  let mut phase_mode = false;
  let highlights = match parse_highlights(spec.as_ref()) {
    HighlightSpec::Full => None,
    HighlightSpec::Highlight(set) => Some(set),
    HighlightSpec::Phase => {
      phase_mode = true;
      None
    }
    HighlightSpec::Unevaluated => {
      // Echo the call with the evaluated specification, as wolframscript
      // does (e.g. PeriodicTablePlot[1 + 5] echoes PeriodicTablePlot[6]).
      let mut echo_args = args.to_vec();
      if let (Some(i), Some(spec)) = (spec_idx, spec) {
        echo_args[i] = spec;
      }
      return Ok(Expr::FunctionCall {
        name: "PeriodicTablePlot".to_string(),
        args: echo_args.into(),
      });
    }
  };
  let elements = elements_layout();
  let dark = crate::is_dark_mode();
  let (bg, faded_tile, text, faded_text, border) = theme_colors(dark);

  // Overall grid extent: 18 columns plus the group 2/3 gutter wide; the
  // actinide row sits at row 8.6, so the drawing reaches 9.6 rows tall.
  // The phase legend adds 1.8 rows below the table.
  let vb_w = 18.0 * CELL + GUTTER;
  let vb_h = if phase_mode { 11.4 * CELL } else { 9.6 * CELL };
  let svg_height = (svg_width as f64 * vb_h / vb_w).round().max(1.0) as u32;

  let mut body = String::new();
  body.push_str(&format!(
    "<rect width=\"{:.0}\" height=\"{:.0}\" fill=\"{}\"/>\n",
    vb_w, vb_h, bg
  ));

  for elem in &elements {
    let cell = element_cell(elem);
    let x = cell_x(cell.col);
    let y = cell.row * CELL;
    let highlighted = highlights
      .as_ref()
      .is_none_or(|h| h.contains(&elem.atomic_number));

    let (fill, num_fill, sym_fill) = if phase_mode {
      (
        phase_color(element_phase(elem.atomic_number), dark),
        text,
        text,
      )
    } else if highlighted {
      (block_color(elem.block, dark), text, text)
    } else {
      (faded_tile, faded_text, faded_text)
    };

    // Tile.
    body.push_str(&format!(
      "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" \
       rx=\"6\" fill=\"{}\" stroke=\"{}\" stroke-width=\"1.5\"/>\n",
      x + PAD,
      y + PAD,
      CELL - 2.0 * PAD,
      CELL - 2.0 * PAD,
      fill,
      border,
    ));
    // Atomic number, top-left.
    body.push_str(&format!(
      "<text x=\"{:.1}\" y=\"{:.1}\" font-family=\"sans-serif\" \
       font-size=\"20\" fill=\"{}\">{}</text>\n",
      x + PAD + 6.0,
      y + PAD + 20.0,
      num_fill,
      elem.atomic_number,
    ));
    // Element symbol, centred.
    body.push_str(&format!(
      "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" \
       font-family=\"sans-serif\" font-size=\"40\" font-weight=\"bold\" \
       fill=\"{}\">{}</text>\n",
      x + CELL / 2.0,
      y + CELL / 2.0 + 20.0,
      sym_fill,
      escape_xml(elem.abbreviation),
    ));
  }

  // Insertion markers: `*` in the group 2/3 gutter of period 6 and to
  // the left of the detached lanthanide row, `**` likewise for period 7
  // and the actinide row.
  push_series_marker(&mut body, 5.0, 1, text);
  push_series_marker(&mut body, 6.0, 2, text);
  push_series_marker(&mut body, 7.6, 1, text);
  push_series_marker(&mut body, 8.6, 2, text);

  if phase_mode {
    push_phase_legend(&mut body, vb_w, dark);
  }

  let width_attr = if full_width {
    "width=\"100%\"".to_string()
  } else {
    format!("width=\"{}\" height=\"{}\"", svg_width, svg_height)
  };
  let svg = format!(
    "<svg {} viewBox=\"0 0 {:.0} {:.0}\" \
     xmlns=\"http://www.w3.org/2000/svg\">\n{}</svg>",
    width_attr, vb_w, vb_h, body
  );

  let graphics = crate::graphics_result(svg);
  if phase_mode {
    return Ok(phase_legended(graphics));
  }
  Ok(graphics)
}

/// Wrap the phase-coloured table in `Legended[…, Placed[SwatchLegend[…],
/// Below]]`, the expression wolframscript returns for
/// `PeriodicTablePlot["Phase"]` (the swatch colours are wolframscript's
/// exact values).
fn phase_legended(graphics: Expr) -> Expr {
  fn call(name: &str, args: Vec<Expr>) -> Expr {
    Expr::FunctionCall {
      name: name.to_string(),
      args: args.into(),
    }
  }
  let rgb = |r: f64, g: f64, b: f64| {
    call(
      "RGBColor",
      vec![Expr::Real(r), Expr::Real(g), Expr::Real(b)],
    )
  };
  let colors = Expr::List(
    vec![
      rgb(0.493332, 0.733333, 0.866667),
      rgb(0.96666, 0.7513329, 0.4283329),
      rgb(0.636667, 0.799999, 0.473333),
      call("GrayLevel", vec![Expr::Real(0.9)]),
    ]
    .into(),
  );
  let labels = Expr::List(
    vec![
      Expr::String("gas".to_string()),
      Expr::String("solid".to_string()),
      Expr::String("liquid".to_string()),
      call("Missing", vec![Expr::String("NotAvailable".to_string())]),
    ]
    .into(),
  );
  let rule = |lhs: &str, rhs: Expr| Expr::Rule {
    pattern: Box::new(Expr::Identifier(lhs.to_string())),
    replacement: Box::new(rhs),
  };
  let legend = call(
    "SwatchLegend",
    vec![
      colors,
      labels,
      rule("LegendLayout", Expr::Identifier("Row".to_string())),
      rule(
        "LegendLabel",
        call(
          "EntityProperty",
          vec![
            Expr::String("Element".to_string()),
            Expr::String("Phase".to_string()),
          ],
        ),
      ),
    ],
  );
  call(
    "Legended",
    vec![
      graphics,
      call(
        "Placed",
        vec![legend, Expr::Identifier("Below".to_string())],
      ),
    ],
  )
}
