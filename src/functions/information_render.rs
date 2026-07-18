//! Graphical rendering for `Information[…]` results.
//!
//! Builds Grid-based SVGs for `InformationData` (a single symbol's info card)
//! and `InformationDataGrid` (the wildcard `?Plot*` view of matching names).
//! Both use the existing `grid_svg_with_gaps` infrastructure so theming and
//! layout stay consistent with `Grid[]` / `TableForm[]` output.

use crate::syntax::{Expr, bool_expr};

/// Build a `Style[content, Bold]` Expr.
fn bold_style(text: &str) -> Expr {
  Expr::FunctionCall {
    name: "Style".to_string(),
    args: vec![
      Expr::String(text.to_string()),
      Expr::Identifier("Bold".to_string()),
    ]
    .into(),
  }
}

/// Build a `Style[content, color]` Expr where `color` is a named color identifier.
fn colored_style(text: &str, color: &str) -> Expr {
  Expr::FunctionCall {
    name: "Style".to_string(),
    args: vec![
      Expr::String(text.to_string()),
      Expr::Identifier(color.to_string()),
    ]
    .into(),
  }
}

/// Build a `Rule[lhs, rhs]` Expr.
fn rule(lhs: Expr, rhs: Expr) -> Expr {
  Expr::Rule {
    pattern: Box::new(lhs),
    replacement: Box::new(rhs),
  }
}

/// Build a `List[…]` Expr.
fn list(items: Vec<Expr>) -> Expr {
  Expr::List(items.into())
}

/// One row of an `Information[…]` card. `value` is the display text;
/// `url`, when set, makes the cell a clickable hyperlink whose anchor target
/// is the URL while the rendered text remains `value`.
pub struct InfoField {
  pub label: String,
  pub value: String,
  pub url: Option<String>,
}

impl InfoField {
  pub fn text(label: impl Into<String>, value: impl Into<String>) -> Self {
    Self {
      label: label.into(),
      value: value.into(),
      url: None,
    }
  }

  pub fn link(
    label: impl Into<String>,
    value: impl Into<String>,
    url: impl Into<String>,
  ) -> Self {
    Self {
      label: label.into(),
      value: value.into(),
      url: Some(url.into()),
    }
  }
}

/// Render an InformationData card as SVG.
///
/// `title` is rendered as a bold header value (e.g. the symbol name);
/// `fields` is a list of (label, value[, url]) entries displayed in a
/// two-column table below the header. Entries with a `url` become clickable
/// hyperlinks via the grid renderer's `Hyperlink[…]` cell handling.
pub fn render_information_card_svg(
  title: &str,
  fields: &[InfoField],
) -> Option<String> {
  // Build a Grid with the title row spanning both columns, followed by
  // alternating field rows.
  let mut rows: Vec<Expr> = Vec::with_capacity(fields.len() + 1);

  // Header row: bold title (left) + a muted "Symbol" tag (right).
  rows.push(list(vec![
    Expr::FunctionCall {
      name: "Style".to_string(),
      args: vec![
        Expr::String(title.to_string()),
        Expr::Identifier("Bold".to_string()),
        Expr::FunctionCall {
          name: "Rule".to_string(),
          args: vec![
            Expr::Identifier("FontSize".to_string()),
            Expr::Integer(16),
          ]
          .into(),
        },
      ]
      .into(),
    },
    colored_style("Symbol", "Gray"),
  ]));

  for field in fields {
    let value_cell = match &field.url {
      Some(url) => Expr::FunctionCall {
        name: "Hyperlink".to_string(),
        args: vec![
          Expr::String(field.value.clone()),
          Expr::String(url.clone()),
        ]
        .into(),
      },
      None => Expr::String(field.value.clone()),
    };
    rows.push(list(vec![bold_style(&field.label), value_cell]));
  }

  let grid_data = list(rows);

  // Grid options: outer frame, left-aligned both columns with right-aligned
  // labels, modest spacing, alternating row backgrounds for readability.
  let frame_opt = rule(Expr::Identifier("Frame".to_string()), bool_expr(true));
  let alignment_opt = rule(
    Expr::Identifier("Alignment".to_string()),
    list(vec![list(vec![
      Expr::Identifier("Right".to_string()),
      Expr::Identifier("Left".to_string()),
    ])]),
  );
  let spacings_opt = rule(
    Expr::Identifier("Spacings".to_string()),
    list(vec![Expr::Integer(2), Expr::Real(0.6)]),
  );
  let dividers_opt = rule(
    Expr::Identifier("Dividers".to_string()),
    list(vec![
      Expr::Identifier("None".to_string()),
      list(vec![
        Expr::Integer(2),
        Expr::Identifier("LightGray".to_string()),
      ]),
    ]),
  );

  let args = vec![
    grid_data,
    frame_opt,
    alignment_opt,
    spacings_opt,
    dividers_opt,
  ];
  crate::functions::graphics::grid_svg_with_gaps(&args, &[]).ok()
}

/// Render an InformationDataGrid (wildcard query result) as SVG.
///
/// `groups` is a list of (context, [matching names]) pairs (one entry per
/// context, e.g. `("System`", ["Plot", "Plot3D", …])`). The names are wrapped
/// across multiple columns so that long lists stay readable.
pub fn render_information_grid_svg(
  groups: &[(String, Vec<String>)],
) -> Option<String> {
  let mut rows: Vec<Expr> = Vec::new();

  for (ctx, names) in groups {
    // Context label row (bold, single-cell row spanning the context line).
    rows.push(list(vec![bold_style(ctx)]));

    if names.is_empty() {
      rows.push(list(vec![colored_style("(no matching symbols)", "Gray")]));
    } else {
      // Wrap names into columns of ~4 per row for compact display.
      let columns_per_row = 4_usize;
      for chunk in names.chunks(columns_per_row) {
        let cells: Vec<Expr> =
          chunk.iter().map(|n| Expr::String(n.clone())).collect();
        rows.push(list(cells));
      }
    }
  }

  if rows.is_empty() {
    return None;
  }

  let grid_data = list(rows);

  let frame_opt = rule(Expr::Identifier("Frame".to_string()), bool_expr(true));
  let alignment_opt = rule(
    Expr::Identifier("Alignment".to_string()),
    Expr::Identifier("Left".to_string()),
  );
  let spacings_opt = rule(
    Expr::Identifier("Spacings".to_string()),
    list(vec![Expr::Integer(2), Expr::Real(0.6)]),
  );

  let args = vec![grid_data, frame_opt, alignment_opt, spacings_opt];
  crate::functions::graphics::grid_svg_with_gaps(&args, &[]).ok()
}
