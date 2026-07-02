use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::element_data::{
  ElementLayout, elements_layout, resolve_atomic_number,
};
use crate::functions::plot::{DEFAULT_HEIGHT, DEFAULT_WIDTH, parse_image_size};
use crate::syntax::Expr;

/// Side length of a single element cell, in SVG grid units. The whole plot
/// is rendered in these units and scaled to the requested pixel size via the
/// SVG `viewBox`, so all layout math stays in simple integers.
const CELL: f64 = 100.0;
/// Inner gap between a cell's border and the coloured tile.
const PAD: f64 = 6.0;
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
/// two rows below the main table, aligned under group 3.
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
          if resolve_atomic_number(other).is_some() {
            crate::emit_message_to_stdout(&format!(
              "PeriodicTablePlot::elmntav: \"{}\" is not an available \
               \"Element\" property.",
              s
            ));
          }
          HighlightSpec::Unevaluated
        }
        // An unresolvable entity, or property coloring via EntityProperty:
        // leave unevaluated without a (wrong) message.
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
  let highlights = match parse_highlights(spec.as_ref()) {
    HighlightSpec::Full => None,
    HighlightSpec::Highlight(set) => Some(set),
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

  // Overall grid extent: 18 columns wide; the actinide row sits at row 8.6,
  // so the drawing reaches 9.6 rows tall.
  let vb_w = 18.0 * CELL;
  let vb_h = 9.6 * CELL;
  let svg_height = (svg_width as f64 * vb_h / vb_w).round().max(1.0) as u32;

  let mut body = String::new();
  body.push_str(&format!(
    "<rect width=\"{:.0}\" height=\"{:.0}\" fill=\"{}\"/>\n",
    vb_w, vb_h, bg
  ));

  for elem in &elements {
    let cell = element_cell(elem);
    let x = cell.col * CELL;
    let y = cell.row * CELL;
    let highlighted = highlights
      .as_ref()
      .is_none_or(|h| h.contains(&elem.atomic_number));

    let (fill, num_fill, sym_fill) = if highlighted {
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

  Ok(crate::graphics_result(svg))
}
