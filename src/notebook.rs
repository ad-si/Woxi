//! Parser and serializer for Mathematica `.nb` notebook files.
//!
//! A `.nb` file is a plain-text file containing Wolfram Language
//! expressions that describe a notebook.  The top-level structure
//! looks like:
//!
//! ```text
//! Notebook[{
//!   Cell[CellGroupData[{
//!     Cell["Title text", "Title"],
//!     Cell[BoxData[...], "Input"],
//!     Cell[BoxData[...], "Output"]
//!   }, Open]],
//!   Cell["Some text", "Text"],
//!   ...
//! }]
//! ```
//!
//! This module provides a lightweight parser that extracts the cells
//! (with their style and content) and a serializer that writes them
//! back out.

use std::fmt;

/// A complete Mathematica notebook.
#[derive(Debug, Clone)]
pub struct Notebook {
  pub cells: Vec<CellEntry>,
}

/// An entry in the notebook – either a single cell or a group.
#[derive(Debug, Clone)]
pub enum CellEntry {
  Single(Cell),
  Group(CellGroup),
}

/// A cell group contains a list of cells (typically an input cell
/// followed by its output).
#[derive(Debug, Clone)]
pub struct CellGroup {
  pub cells: Vec<Cell>,
  pub open: bool,
}

/// A single notebook cell.
#[derive(Debug, Clone)]
pub struct Cell {
  pub style: CellStyle,
  pub content: String,
  /// For Chapter/Subchapter cells: whether the section is collapsed
  /// (hiding all cells below it until the next same-or-higher heading).
  /// Persisted as a `CellOpen -> False` option in the `.nb` file.
  pub collapsed: bool,
}

/// The style/type of a cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellStyle {
  Title,
  Subtitle,
  Chapter,
  Subchapter,
  Section,
  Subsection,
  Subsubsection,
  Text,
  Item,
  Subitem,
  Input,
  Output,
  Code,
  Print,
}

impl CellStyle {
  fn from_str(s: &str) -> Option<Self> {
    match s {
      "Title" => Some(Self::Title),
      "Subtitle" => Some(Self::Subtitle),
      "Chapter" => Some(Self::Chapter),
      "Subchapter" => Some(Self::Subchapter),
      "Section" => Some(Self::Section),
      "Subsection" => Some(Self::Subsection),
      "Subsubsection" => Some(Self::Subsubsection),
      "Text" => Some(Self::Text),
      "Item" => Some(Self::Item),
      "Subitem" => Some(Self::Subitem),
      "Input" => Some(Self::Input),
      "Output" => Some(Self::Output),
      "Code" => Some(Self::Code),
      "Print" => Some(Self::Print),
      _ => None,
    }
  }

  pub fn as_str(self) -> &'static str {
    match self {
      Self::Title => "Title",
      Self::Subtitle => "Subtitle",
      Self::Chapter => "Chapter",
      Self::Subchapter => "Subchapter",
      Self::Section => "Section",
      Self::Subsection => "Subsection",
      Self::Subsubsection => "Subsubsection",
      Self::Text => "Text",
      Self::Item => "Item",
      Self::Subitem => "Subitem",
      Self::Input => "Input",
      Self::Output => "Output",
      Self::Code => "Code",
      Self::Print => "Print",
    }
  }
}

impl fmt::Display for CellStyle {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(self.as_str())
  }
}

// ── Parsing ─────────────────────────────────────────────────────────

/// Parse a `.nb` file's contents into a `Notebook`.
pub fn parse_notebook(input: &str) -> Result<Notebook, String> {
  // The FrontEnd hard-wraps long lines when it writes a `.nb`, marking
  // each break with a trailing backslash. Undo that first: the break can
  // fall anywhere, including in the middle of a run of closing brackets
  // (`}]}]}\⏎]}]`), which leaves the box structure unparseable.
  let input = &strip_line_continuations(input);
  // Real .nb files have comment headers/footers around Notebook[{...}].
  // Find the start of the Notebook expression.
  let nb_start = input
    .find("Notebook[{")
    .ok_or("Expected Notebook[{...}] wrapper")?;
  let after_prefix = &input[nb_start + "Notebook[{".len()..];

  // Find the matching `}` that closes the cell list.
  let (cell_list, _remainder) = find_matching_brace(after_prefix)
    .map_err(|e| format!("Parsing Notebook cell list: {e}"))?;

  let cells = parse_cell_list(cell_list)?;
  Ok(Notebook { cells })
}

/// Remove the FrontEnd's physical line-wrap continuations: a backslash
/// immediately before a newline, plus the newline itself. Wolfram drops
/// them both between tokens and inside string literals (`"ab\⏎cd"` is
/// `"abcd"`), so the text they join is what the cell really contains.
///
/// A backslash that is itself escaped (`\\` at the end of a line) is a
/// literal backslash, not a continuation — only an odd-length run counts.
fn strip_line_continuations(input: &str) -> String {
  if !input.contains('\\') {
    return input.to_string();
  }
  let mut out = String::with_capacity(input.len());
  let mut backslashes = 0usize;
  let mut chars = input.chars().peekable();
  while let Some(c) = chars.next() {
    // `\r\n` counts as one newline.
    let is_newline = c == '\n' || (c == '\r' && chars.peek() == Some(&'\n'));
    if is_newline && backslashes % 2 == 1 {
      out.pop(); // the continuation backslash
      if c == '\r' {
        chars.next();
      }
      backslashes = 0;
      continue;
    }
    if c == '\\' {
      backslashes += 1;
    } else {
      backslashes = 0;
    }
    out.push(c);
  }
  out
}

/// Parse a comma-separated list of Cell[...] or
/// Cell[CellGroupData[...]] entries.
fn parse_cell_list(input: &str) -> Result<Vec<CellEntry>, String> {
  let mut entries = Vec::new();
  let items = split_top_level_commas(input);

  for item in items {
    let item = item.trim();
    if item.is_empty() {
      continue;
    }
    entries.push(parse_cell_entry(item)?);
  }

  Ok(entries)
}

/// Parse a single cell entry (Cell[...]).
fn parse_cell_entry(s: &str) -> Result<CellEntry, String> {
  let s = s.trim();
  if !s.starts_with("Cell[") {
    return Err(format!(
      "Expected Cell[...], got: {}",
      &s[..s.len().min(60)]
    ));
  }

  // Strip Cell[ ... ]
  let inner = &s[5..];
  let inner = inner
    .strip_suffix(']')
    .ok_or("Missing closing ] for Cell")?;

  // Check if it's a CellGroupData
  let inner_trimmed = inner.trim();
  if let Some(after_prefix) = inner_trimmed.strip_prefix("CellGroupData[") {
    // Find the matching `]` for `CellGroupData[` so we can ignore any
    // trailing options on the outer Cell (e.g. `ExpressionUUID -> "…"`).
    let (group_inner, _trailing_options) = find_matching_bracket(after_prefix)?;
    return parse_cell_group_body(group_inner);
  }

  // Regular cell: Cell["content", "Style"]
  let cell = parse_single_cell(inner_trimmed);
  Ok(CellEntry::Single(cell))
}

/// Parse the body of `CellGroupData[...]`, i.e. the content *between*
/// the brackets: `{cells...}, Open|Closed`. Any trailing items beyond
/// `Open`/`Closed` (e.g. positional options) are ignored.
fn parse_cell_group_body(s: &str) -> Result<CellEntry, String> {
  // Find the { ... } cell list and the Open/Closed flag.
  let rest = s.trim();
  let rest = rest
    .strip_prefix('{')
    .ok_or("Expected { after CellGroupData[")?;

  // We need to find the matching }
  let (cell_list_str, remainder) = find_matching_brace(rest)?;

  let mut cells = Vec::new();
  let cell_items = split_top_level_commas(cell_list_str);
  for item in cell_items {
    let item = item.trim();
    if item.is_empty() {
      continue;
    }
    match parse_cell_entry(item)? {
      CellEntry::Single(c) => cells.push(c),
      CellEntry::Group(g) => {
        // Flatten nested groups into cells
        cells.extend(g.cells);
      }
    }
  }

  let remainder = remainder.trim();
  let remainder = remainder.strip_prefix(',').unwrap_or(remainder);
  let open = !remainder.trim().starts_with("Closed");

  Ok(CellEntry::Group(CellGroup { cells, open }))
}

/// Parse the content of a single cell:
/// e.g. `"some text", "Title"` or `BoxData[...], "Input"`
fn parse_single_cell(s: &str) -> crate::notebook::Cell {
  // Split on the last comma at top level to get the style
  let parts = split_top_level_commas(s);
  if parts.len() < 2 {
    // Try to handle cells with just content
    let content = extract_string_content(s);
    return Cell {
      style: CellStyle::Text,
      content,
      collapsed: false,
    };
  }

  // The style string is the first part that parses as a valid style
  // after the content. Any remaining parts after that are options
  // (e.g. `CellOpen -> False`).
  let mut style_idx = None;
  for (i, part) in parts.iter().enumerate().skip(1) {
    let trimmed = part.trim().trim_matches('"').trim();
    if CellStyle::from_str(trimmed).is_some() {
      style_idx = Some(i);
      break;
    }
  }
  // Fallback for styles we don't model (e.g. `CodeText`, `MSG`): the style is
  // the first bare string literal after the content. Options always take the
  // form `name -> value` / `name :> value`, so a part with no rule arrow that
  // starts with `"` is the style name. Picking it (rather than the last part)
  // keeps the trailing options out of the extracted content.
  let style_idx = style_idx
    .or_else(|| {
      parts.iter().enumerate().skip(1).find_map(|(i, part)| {
        let t = part.trim();
        let is_option = t.contains("->") || t.contains(":>");
        (!is_option && t.starts_with('"')).then_some(i)
      })
    })
    .unwrap_or(parts.len() - 1);

  let style_str = parts[style_idx].trim();
  let style_str = style_str.trim_matches('"').trim();
  let style = CellStyle::from_str(style_str).unwrap_or(CellStyle::Text);

  // Parts before the style are the content.
  let content_parts = &parts[..style_idx];
  let raw_content = content_parts.join(",");
  let content = extract_cell_content(&raw_content);

  // Parts after the style are options.
  let mut collapsed = false;
  for opt in &parts[style_idx + 1..] {
    if is_cell_open_false(opt) {
      collapsed = true;
    }
  }

  Cell {
    style,
    content,
    collapsed,
  }
}

/// Does this option-expression set `CellOpen -> False`?
fn is_cell_open_false(s: &str) -> bool {
  let s: String = s.chars().filter(|c| !c.is_whitespace()).collect();
  s == "CellOpen->False"
}

/// Is there nothing left to draw once the characters that have no glyph are
/// taken out? The script-base placeholders (`\[InvisiblePrefixScriptBase]`)
/// are such characters, and a box built on one is a *prefix* script whose
/// base is empty.
fn draws_nothing(s: &str) -> bool {
  crate::syntax::substitute_private_use_glyphs(s)
    .trim()
    .is_empty()
}

/// The part specification inside a `⟦…⟧` group, or None when `s` is an
/// ordinary subscript. The double-bracket characters are what
/// `\[LeftDoubleBracket]` / `\[RightDoubleBracket]` unescape to.
fn part_spec_inside_double_brackets(s: &str) -> Option<&str> {
  let s = s.trim();
  // Wolfram writes `〚…〛` (U+301A/U+301B); the mathematical white square
  // brackets `⟦…⟧` are the same thing in notebooks written elsewhere.
  s.strip_prefix('\u{301A}')
    .and_then(|inner| inner.strip_suffix('\u{301B}'))
    .or_else(|| {
      s.strip_prefix('\u{27E6}')
        .and_then(|inner| inner.strip_suffix('\u{27E7}'))
    })
    .map(str::trim)
}

/// `base[[spec]]` when `base` is a single token, `Part[base, spec]` when it
/// is anything else — the function form needs no parentheses to keep its
/// precedence.
fn format_part_access(base: &str, spec: &str) -> String {
  let simple = !base.is_empty()
    && base
      .chars()
      .all(|c| c.is_alphanumeric() || matches!(c, '$' | '`' | '#' | '_'));
  if simple {
    format!("{base}[[{spec}]]")
  } else {
    format!("Part[{base}, {spec}]")
  }
}

/// Convert a bare box-expression source (`SubscriptBox[p, 0]`, as carried by
/// an inline `\!\(\*…\)` segment inside a string) into the evaluable
/// expression it typesets (`Subscript[p, 0]`), including the `RowBox[{…}]`
/// a flat run of source typesets to. Returns None when `s` is not one of the
/// recognised typeset box heads.
pub fn box_source_to_expression(s: &str) -> Option<String> {
  let s = s.trim();
  // `RowBox[{…}]` is the box form of a plain run of source, which is what
  // an expression with no 2-D structure typesets to.
  if let Some(inner) = s.strip_prefix("RowBox[") {
    let inner = inner.strip_suffix(']')?;
    return Some(extract_rowbox_content(inner));
  }
  extract_typeset_box(s)
}

/// Extract cell content from BoxData[...] or a quoted string.
fn extract_cell_content(s: &str) -> String {
  let s = s.trim();

  // Handle BoxData[RowBox[{...}]], BoxData["..."], or BoxData[{...}]
  if let Some(inner) = s.strip_prefix("BoxData[") {
    let inner = inner.strip_suffix(']').unwrap_or(inner);
    return extract_cell_content(inner);
  }

  // Handle RowBox[{"...", ...}]
  if let Some(inner) = s.strip_prefix("RowBox[") {
    let inner = inner.strip_suffix(']').unwrap_or(inner);
    return extract_rowbox_content(inner);
  }

  // Handle TextData[...] used by Text/Section/Subsection heading cells.
  // Forms: `TextData["str"]`, `TextData[{elem, elem, ...}]`, or
  // `TextData[ButtonBox[...]]` (a hyperlink). The Wolfram Demonstrations
  // templates embed the section label as the leading string, followed by an
  // inline `Cell[...]` "more info" opener button that carries no textual
  // content — see `extract_textdata`.
  if let Some(inner) = s.strip_prefix("TextData[") {
    let inner = inner.strip_suffix(']').unwrap_or(inner);
    return extract_textdata(inner);
  }

  // Handle 2-D typeset display boxes. These are emitted by the Wolfram
  // FrontEnd to render assignments like `r = (matrix)` or expressions
  // containing fractions / exponents in pretty-printed form; converting
  // them back to plain InputForm text keeps the cell evaluable.
  if let Some(result) = extract_typeset_box(s) {
    return result;
  }

  // Handle a bare list `{...}` (e.g. multi-statement Input cells written as
  // `BoxData[{ RowBox[...], "\n", RowBox[...], ... }]`). The list's items
  // follow the same conventions as a RowBox, so reuse that extractor.
  if s.starts_with('{') && s.ends_with('}') {
    return extract_rowbox_content(s);
  }

  // Handle quoted strings
  extract_string_content(s)
}

/// Is this argument a top-level option rule (`name -> value` or
/// `name :> value`)? Typeset box heads carry display options after their
/// positional arguments (e.g. `SuperscriptBox[a, b, MultilineFunction ->
/// None]`); they must not leak into the reconstructed InputForm text.
fn is_option_arg(s: &str) -> bool {
  let mut depth = 0i32;
  let mut in_string = false;
  let mut prev_backslash = false;
  let bytes = s.as_bytes();

  for (i, c) in s.char_indices() {
    if in_string {
      if c == '"' && !prev_backslash {
        in_string = false;
      }
      prev_backslash = c == '\\' && !prev_backslash;
      continue;
    }
    match c {
      '"' => in_string = true,
      '{' | '[' | '(' => depth += 1,
      '}' | ']' | ')' => depth -= 1,
      '-' | ':' if depth == 0 && bytes.get(i + 1) == Some(&b'>') => {
        return true;
      }
      _ => {}
    }
  }
  false
}

/// Number of prime marks if `s` is a superscript string consisting solely
/// of `\[Prime]` named characters (or `′`), the FrontEnd's typeset form of
/// `Derivative`: `SuperscriptBox["\[Theta]", "\[Prime]\[Prime]"]` is
/// `θ''`. Returns `None` for any other superscript.
fn prime_marks(s: &str) -> Option<usize> {
  let mut rest = s.trim().trim_matches('"');
  if rest.is_empty() {
    return None;
  }
  let mut count = 0;
  while !rest.is_empty() {
    if let Some(r) = rest.strip_prefix("\\[Prime]") {
      rest = r;
    } else {
      rest = rest.strip_prefix('\u{2032}')?;
    }
    count += 1;
  }
  Some(count)
}

/// The elements of a `TemplateBox`'s first argument, which holds the
/// template's slots. It is a list, written either as `{…}` (how a `.nb` file
/// spells it) or as `List[…]` (how `InputForm` writes a box escape).
fn template_box_parts(first_arg: &str) -> Vec<&str> {
  let first = first_arg.trim();
  let inner = first
    .strip_prefix('{')
    .and_then(|s| s.strip_suffix('}'))
    .or_else(|| {
      first
        .strip_prefix("List[")
        .and_then(|s| s.strip_suffix(']'))
    })
    .unwrap_or(first);
  split_top_level_commas(inner)
}

/// The `Row[…]` call one of the FrontEnd's row templates stands for.
///
/// `RowDefault` holds just the parts. The separator variants put the
/// separator first: a string one appears twice — as the text it draws and as
/// the literal it was written as — and only the literal reads back as the
/// separator's own expression.
fn row_template_source(tag: &str, parts: &[&str]) -> String {
  let mut parts = parts.iter().map(|p| box_part_source(p.trim()));
  let separator = match tag {
    "RowWithSeparators" => {
      parts.next();
      parts.next()
    }
    "RowWithSeparator" => parts.next(),
    _ => None,
  };
  let items = parts.collect::<Vec<_>>().join(", ");
  match separator {
    Some(sep) => format!("Row[{{{items}}}, {sep}]"),
    None => format!("Row[{{{items}}}]"),
  }
}

/// The plain text a display-only box shows, or `None` when the box is not
/// simple display text.
///
/// Styling wrappers (`StyleBox`, `TagBox`, …) are unwrapped down to the
/// string literal they decorate. Box expressions nest string literals one
/// level deeper than ordinary arguments — the FrontEnd writes the displayed
/// string `label` as `"\"label\""` — so every quoting layer is stripped.
fn display_text(s: &str) -> Option<String> {
  let s = s.trim();
  for head in ["StyleBox", "TagBox", "FormBox", "AdjustmentBox", "FrameBox"] {
    if let Some(rest) = s.strip_prefix(&format!("{head}[")) {
      let (inner, _) = find_matching_bracket(rest).ok()?;
      let first = split_top_level_commas(inner).into_iter().next()?;
      return display_text(first);
    }
  }
  if !(s.starts_with('"') && s.ends_with('"') && s.len() >= 2) {
    return None;
  }
  let mut text = s.to_string();
  while text.starts_with('"') && text.ends_with('"') && text.len() >= 2 {
    text = extract_string_content(&text);
  }
  Some(text)
}

/// Render a `CheckboxBox[value, {off, on}]` as a glyph.
///
/// `args` are the box's positional arguments. When `with_label` is set and
/// the `on` alternative is a string, it is appended as the checkbox's label
/// (`CheckboxBox[False, {False, "Mathematics"}]` → `☐ Mathematics`). Rows
/// that carry their own label text pass `false` so the label is not
/// repeated.
fn render_checkbox(args: &[String], with_label: bool) -> String {
  let value = args.first().map_or("", |a| a.trim());
  let mut checked = value == "True";
  let mut label = None;
  if let Some(alts) = args.get(1) {
    let alts = alts.trim();
    if let Some(inner) =
      alts.strip_prefix('{').and_then(|a| a.strip_suffix('}'))
    {
      let alt_parts = split_top_level_commas(inner);
      let off = alt_parts.first().map(|p| p.trim());
      if let Some(on) = alt_parts.get(1).map(|p| p.trim()) {
        // With degenerate alternatives ({False, False}) fall back to
        // the truthiness of the value itself.
        if off != Some(on) {
          checked = value == on;
        }
        if with_label
          && on.starts_with('"')
          && on.ends_with('"')
          && on.len() >= 2
        {
          label = Some(extract_string_content(on));
        }
      }
    }
  }
  let mark = if checked { "\u{2611}" } else { "\u{2610}" };
  match label {
    Some(l) => format!("{mark} {l}"),
    None => mark.to_string(),
  }
}

/// Recognise the typeset box heads that the FrontEnd uses to pretty-print
/// expressions (`FractionBox`, `SuperscriptBox`, `SqrtBox`, `TagBox`,
/// `GridBox`, …) and convert them back into evaluable InputForm text.
///
/// Returns `None` if `s` does not start with one of those heads.
fn extract_typeset_box(s: &str) -> Option<String> {
  fn split_args(head: &str, s: &str) -> Option<Vec<String>> {
    let prefix = format!("{head}[");
    let rest = s.strip_prefix(&prefix)?;
    let (inner, _) = find_matching_bracket(rest).ok()?;
    Some(
      split_top_level_commas(inner)
        .into_iter()
        .map(|p| p.trim().to_string())
        .collect(),
    )
  }
  // `TemplateBox` is used by the FrontEnd to pretty-print typeset values whose
  // underlying expression is encoded in the trailing identifier. The actuarial
  // example notebook uses
  //   TemplateBox[{"19400", RowBox[…], "US dollars", "\"USDollars\""},
  //               "QuantityPrefix"]
  // for currency literals — we map these back to `Quantity[number, "unit"]`
  // so the cell remains evaluable.
  if s.starts_with("TemplateBox[")
    && let Some(args) = split_args("TemplateBox", s)
    && args.len() >= 2
  {
    let tag = args.last().unwrap().trim();
    let tag = tag.trim_matches('"');
    let parts = template_box_parts(&args[0]);
    if tag == "QuantityPrefix" || tag == "Quantity" {
      // First positional element is a list `{number, displayed_unit, unit_name, unit_id_string}`.
      if parts.len() >= 4 {
        let number = extract_cell_content(parts[0].trim());
        // Element 4 is the canonical unit name string. It's written as
        // `"\"USDollars\""` in the box expression (an *inner* string literal),
        // so we strip every layer of surrounding quotes/backslashes to get
        // the bare name, then re-wrap in a single pair of quotes.
        let mut unit_name = parts[3].trim().to_string();
        while unit_name.len() >= 2
          && unit_name.starts_with('"')
          && unit_name.ends_with('"')
        {
          unit_name = unit_name[1..unit_name.len() - 1].to_string();
        }
        // Unescape any remaining `\"` pairs.
        let unit_name = unit_name.replace("\\\"", "");
        return Some(format!("Quantity[{number}, \"{unit_name}\"]"));
      }
    }
    let is_checkbox_row =
      parts.iter().any(|p| p.trim().starts_with("CheckboxBox["));
    // Demonstration metadata cells pair a checkbox with its caption in a
    // `RowDefault` row, so the generic "first element only" fallback below
    // would drop the caption and leave a bare glyph. Only rows whose
    // non-checkbox parts are plain display text are joined: the category
    // picker wraps its captions in collapsible `PaneSelectorBox` chrome
    // instead, which has no useful flat rendering, so those keep falling back
    // to the first element.
    if tag == "RowDefault" && is_checkbox_row {
      // Render every non-checkbox part as display text; bail out as soon as
      // one has no such rendering.
      let captions: Option<Vec<Option<String>>> = parts
        .iter()
        .map(|part| {
          let part = part.trim();
          if part.starts_with("CheckboxBox[") {
            Some(None)
          } else {
            display_text(part).map(Some)
          }
        })
        .collect();
      if let Some(captions) = captions
        && captions
          .iter()
          .any(|c| c.as_ref().is_some_and(|t| !t.trim().is_empty()))
      {
        // The row supplies the caption, so a checkbox contributes only its
        // glyph — its `on` alternative would just repeat the caption.
        let mut out = String::new();
        for (part, caption) in parts.iter().zip(&captions) {
          match caption {
            Some(text) => out.push_str(text),
            None => out.push_str(&render_checkbox(
              &split_args("CheckboxBox", part.trim()).unwrap_or_default(),
              false,
            )),
          }
        }
        return Some(out.trim().to_string());
      }
    }
    // The row templates are the box form of `Row` — the FrontEnd lays the
    // parts out side by side. Rebuilding the `Row[…]` call (rather than
    // gluing the parts' text together) is what lets a typeset `Row` read
    // back as the expression it was typeset from.
    if matches!(tag, "RowDefault" | "RowWithSeparator" | "RowWithSeparators")
      && !is_checkbox_row
    {
      return Some(row_template_source(tag, &parts));
    }
    // Fallback: first positional argument is the displayed value.
    if let Some(first_part) = parts.first() {
      return Some(extract_cell_content(first_part.trim()));
    }
  }
  // The FrontEnd typesets an inline `Image[…]` literal as
  //   GraphicsBox[TagBox[RasterBox[data, rect, range, opts…],
  //     BoxForm`ImageTag["Byte", ColorSpace -> "RGB", Interleaving -> True],
  //     Selectable -> False], BaseStyle -> "ImageGraphics", …]
  // (e.g. the embedded source image of a Wolfram Demonstration's
  // initialization cell). Rebuild the evaluable `Image[…]` constructor from
  // the box expression so such cells stay runnable.
  if s.starts_with("GraphicsBox[")
    && let Some(args) = split_args("GraphicsBox", s)
    && let Some(first) = args.first()
    && let Some(image) = extract_image_from_boxes(first)
  {
    return Some(image);
  }
  for head in [
    "FractionBox",
    "SuperscriptBox",
    "SubscriptBox",
    "SubsuperscriptBox",
    "OverscriptBox",
    "UnderscriptBox",
    "SqrtBox",
    "RadicalBox",
    "TagBox",
    "GridBox",
    "StyleBox",
    "InterpretationBox",
    "FormBox",
    "AdjustmentBox",
    "FrameBox",
    "CheckboxBox",
  ] {
    if !s.starts_with(head) {
      continue;
    }
    let mut args = split_args(head, s)?;
    // Drop trailing display options (`MultilineFunction -> None`, …) so the
    // positional-argument matches below see the box's real arity.
    args.retain(|a| !is_option_arg(a));
    let conv = |a: &str| extract_cell_content(a);
    // The `⎧` brace a `Piecewise` is typeset with is a `GridBox` holding the
    // `\[Piecewise]` character beside a `GridBox` of value/condition rows —
    // reading it as a plain nested list would hand a plot a list where it
    // expects a function.
    if head == "GridBox"
      && let Some(piecewise) =
        args.first().and_then(|rows| piecewise_from_grid(rows))
    {
      return Some(piecewise);
    }
    return Some(match head {
      // `FractionBox[a, b]` → `(a)/(b)`. The parentheses preserve operator
      // precedence around composite numerators / denominators.
      "FractionBox" if args.len() == 2 => {
        format!("({})/({})", conv(&args[0]), conv(&args[1]))
      }
      // `SuperscriptBox[f, "\[Prime]"]` is the typeset form of the
      // derivative `f'` — emit prime characters, not exponentiation.
      "SuperscriptBox"
        if args.len() == 2 && prime_marks(&args[1]).is_some() =>
      {
        format!(
          "{}{}",
          conv(&args[0]),
          "'".repeat(prime_marks(&args[1]).unwrap())
        )
      }
      // A script hung on `\[InvisiblePrefixScriptBase]` is a *prefix*
      // script — `\!\(\*SuperscriptBox[\(\[InvisiblePrefixScriptBase]\),
      // \(1\)]\)Σ` typesets as `¹Σ`. The base is empty, so the
      // exponentiation form would both fail to parse (`()^(1)`) and, once
      // evaluated, drop the script entirely (`x^1` is `x`). `Superscript`
      // stays unevaluated and keeps it.
      "SuperscriptBox" if args.len() == 2 && draws_nothing(&conv(&args[0])) => {
        format!("Superscript[\"\", {}]", conv(&args[1]))
      }
      // `SuperscriptBox[a, b]` → `(a)^(b)`.
      "SuperscriptBox" if args.len() == 2 => {
        format!("({})^({})", conv(&args[0]), conv(&args[1]))
      }
      // `SubscriptBox[a, b]` → `Subscript[a, b]` (Wolfram's evaluable form),
      // except when the subscript is a `\[LeftDoubleBracket]…\]` group: the
      // FrontEnd also accepts `Part` typeset as a bracketed subscript, and
      // `SubscriptBox["c", RowBox[{"⟦", "1", "⟧"}]]` means `c[[1]]`.
      "SubscriptBox" if args.len() == 2 => {
        let sub = conv(&args[1]);
        let base = conv(&args[0]);
        if let Some(spec) = part_spec_inside_double_brackets(&sub) {
          format_part_access(&base, spec)
        } else {
          // The subscript is usually a real index (`Subscript[p, 0]`
          // stays evaluable, keeping `p` and `0` bare), but a
          // typeset annotation like `\!\(\*SubscriptBox[\(0\),
          // \(+\)]\)` (the "0⁺" of a one-sided limit) has no meaning
          // as code on its own — `Subscript[0, +]` would not parse.
          // Quote such a subscript as a string literal instead, the
          // same fallback `OverscriptBox`/`UnderscriptBox` use for
          // their bare mark below.
          let sub = if crate::parse_to_expr(&sub).is_ok() {
            sub
          } else {
            format!("\"{}\"", escape_string(&sub))
          };
          // Prefix subscript, as above — keep an explicit empty string
          // so the result still parses.
          if draws_nothing(&base) {
            format!("Subscript[\"\", {sub}]")
          } else {
            format!("Subscript[{base}, {sub}]")
          }
        }
      }
      // `SubsuperscriptBox[a, b, c]` → `Subscript[a, b]^c`.
      "SubsuperscriptBox" if args.len() == 3 => {
        format!(
          "(Subscript[{}, {}])^({})",
          conv(&args[0]),
          conv(&args[1]),
          conv(&args[2])
        )
      }
      // `OverscriptBox[a, b]` / `UnderscriptBox[a, b]` → `Overscript[a, "b"]`
      // / `Underscript[a, "b"]` (Wolfram's evaluable forms, which — like
      // `Subscript` — stay symbolic rather than evaluating away). The mark
      // is a display annotation (a hat, bar, tilde, or a rate constant over
      // a reaction arrow), not code, so it is quoted as a string literal
      // rather than left bare: an unquoted `_` would otherwise parse as the
      // `Blank[]` pattern instead of the macron mark it typesets.
      "OverscriptBox" if args.len() == 2 => {
        format!(
          "Overscript[{}, \"{}\"]",
          conv(&args[0]),
          escape_string(&conv(&args[1]))
        )
      }
      "UnderscriptBox" if args.len() == 2 => {
        format!(
          "Underscript[{}, \"{}\"]",
          conv(&args[0]),
          escape_string(&conv(&args[1]))
        )
      }
      // `SqrtBox[a]` → `Sqrt[a]`.
      "SqrtBox" if args.len() == 1 => format!("Sqrt[{}]", conv(&args[0])),
      // `RadicalBox[a, n]` → `Surd[a, n]` (n-th root).
      "RadicalBox" if args.len() == 2 => {
        format!("Surd[{}, {}]", conv(&args[0]), conv(&args[1]))
      }
      // `TagBox[content, tag, opts...]` is a display annotation; the
      // evaluable value is just `content`.
      "TagBox" if !args.is_empty() => conv(&args[0]),
      // `StyleBox`, `FrameBox`, `AdjustmentBox`, `FormBox` similarly wrap
      // a displayed expression; recurse into the first arg.
      "StyleBox" | "FrameBox" | "AdjustmentBox" | "FormBox"
        if !args.is_empty() =>
      {
        conv(&args[0])
      }
      // `InterpretationBox[displayed, value]` stores both the typeset form
      // and the underlying expression that should be used for evaluation.
      // We want the second argument — with any display-form wrapper it
      // carries removed, since that only records how the boxes were laid
      // out.
      "InterpretationBox" if args.len() >= 2 => {
        strip_display_form_wrapper(&conv(&args[1]))
      }
      // `CheckboxBox[value, {off, on}]` (Demonstrations metadata cells) —
      // render a checkbox glyph, labelled with the `on` alternative when
      // it is a string (`CheckboxBox[False, {False, "Mathematics"}]` →
      // `☐ Mathematics`).
      "CheckboxBox" if !args.is_empty() => render_checkbox(&args, true),
      // `GridBox[{{r11, r12, …}, {r21, …}, …}, opts…]` → the raw rows as a
      // list literal. The rows themselves may contain box expressions, so
      // recurse into each cell.
      "GridBox" if !args.is_empty() => {
        // `args[0]` is the outer list of rows: `{{a, b}, {c, d}}`.
        let rows_text = args[0].trim();
        let rows_inner = rows_text
          .strip_prefix('{')
          .and_then(|s| s.strip_suffix('}'))
          .unwrap_or(rows_text);
        let rows: Vec<String> = split_top_level_commas(rows_inner)
          .into_iter()
          .map(|row| {
            let row = row.trim();
            let row_inner = row
              .strip_prefix('{')
              .and_then(|s| s.strip_suffix('}'))
              .unwrap_or(row);
            let cells: Vec<String> = split_top_level_commas(row_inner)
              .into_iter()
              .map(|cell| extract_cell_content(cell.trim()))
              .collect();
            format!("{{{}}}", cells.join(", "))
          })
          .collect();
        format!("{{{}}}", rows.join(", "))
      }
      _ => return None,
    });
  }
  None
}

/// Rebuild an evaluable `Image[data, "type", opts…]` from the typeset
/// raster boxes inside a `GraphicsBox` (see `extract_typeset_box`).
/// `s` is the GraphicsBox's first argument, expected to be
/// `TagBox[RasterBox[…], BoxForm`ImageTag[type, opts…], …]`.
/// Returns `None` when `s` is not an image raster (an ordinary graphic).
fn extract_image_from_boxes(s: &str) -> Option<String> {
  let rest = s.trim().strip_prefix("TagBox[")?;
  let (inner, _) = find_matching_bracket(rest).ok()?;
  let targs = split_top_level_commas(inner);
  let raster = targs.first()?.trim();
  let image_tag = targs
    .iter()
    .map(|t| t.trim())
    .find(|t| t.starts_with("BoxForm`ImageTag["))?;

  // RasterBox[data, {{x0, y0}, {x1, y1}}, range, opts…]
  let rest = raster.strip_prefix("RasterBox[")?;
  let (inner, _) = find_matching_bracket(rest).ok()?;
  let rargs = split_top_level_commas(inner);
  let data_raw = rargs.first()?.trim();
  // The data is either `CompressedData["…"]` (a base64 payload that the
  // FrontEnd wraps across many lines — strip that layout whitespace) or a
  // literal nested list.
  let data = if let Some(rest) = data_raw.strip_prefix("CompressedData[") {
    let (inner, _) = find_matching_bracket(rest).ok()?;
    let trimmed = inner.trim();
    let unquoted = trimmed
      .strip_prefix('"')
      .and_then(|t| t.strip_suffix('"'))
      .unwrap_or(trimmed);
    let payload: String =
      unquoted.chars().filter(|c| !c.is_whitespace()).collect();
    format!("CompressedData[\"{payload}\"]")
  } else {
    data_raw.to_string()
  };

  // `Raster` rows run bottom-to-top while `Image` rows run top-to-bottom.
  // Image typesetting compensates by flipping the bounding rectangle's y
  // axis (`{{0, h}, {w, 0}}`), which means the stored rows are already in
  // Image order. A raster with a normal-orientation rectangle stores its
  // rows bottom-up, so those need an explicit `Reverse` to become Image
  // rows.
  let rows_bottom_up = rargs
    .get(1)
    .and_then(|rect| {
      let rect = rect.trim();
      let inner = rect.strip_prefix('{')?.strip_suffix('}')?;
      let corners = split_top_level_commas(inner);
      let corner_y = |c: &str| -> Option<f64> {
        let inner = c.trim().strip_prefix('{')?.strip_suffix('}')?;
        let parts = split_top_level_commas(inner);
        parts.get(1)?.trim().parse::<f64>().ok()
      };
      let y0 = corner_y(corners.first()?)?;
      let y1 = corner_y(corners.get(1)?)?;
      Some(y0 < y1)
    })
    .unwrap_or(false);
  let data = if rows_bottom_up {
    format!("Reverse[{data}]")
  } else {
    data
  };

  // BoxForm`ImageTag["Byte", ColorSpace -> "RGB", Interleaving -> True]
  let rest = image_tag.strip_prefix("BoxForm`ImageTag[")?;
  let (inner, _) = find_matching_bracket(rest).ok()?;
  let iargs = split_top_level_commas(inner);
  let ty = extract_string_content(iargs.first()?.trim());
  let mut result = format!("Image[{data}, \"{ty}\"");
  for opt in &iargs[1..] {
    let opt = opt.trim();
    // Only pass through rule-shaped options (ColorSpace, Interleaving, …).
    if opt.contains("->") || opt.contains(":>") {
      result.push_str(", ");
      result.push_str(&normalize_whitespace(opt));
    }
  }
  result.push(']');
  Some(result)
}

/// Collapse all whitespace runs (including newlines from the FrontEnd's
/// line wrapping) into single spaces.
fn normalize_whitespace(s: &str) -> String {
  s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Positional arguments of `head[...]`, with trailing display options
/// (`MultilineFunction -> None`, …) dropped.
fn positional_box_args(head: &str, s: &str) -> Option<Vec<String>> {
  let rest = s.trim().strip_prefix(head)?.strip_prefix('[')?;
  let (inner, _) = find_matching_bracket(rest).ok()?;
  let mut args: Vec<String> = split_top_level_commas(inner)
    .into_iter()
    .map(|p| p.trim().to_string())
    .collect();
  args.retain(|a| !is_option_arg(a));
  Some(args)
}

/// An `InterpretationBox`'s meaning with a display-form wrapper
/// (`InputForm[expr]`, `TraditionalForm[expr]`, …) peeled off.
///
/// Pasting an `InputForm`-formatted result back into a notebook stores it as
/// `InterpretationBox[StyleBox[<boxes>, …], InputForm[<expr>], AutoDelete ->
/// True, …]`: the wrapper records *how* the boxes were formatted, while the
/// expression the cell stands for is `<expr>` itself — which is what
/// re-evaluating the cell gives back. Keeping the wrapper would leave the
/// value an inert one-element `InputForm[…]` object, so `Dimensions`, `Map`
/// and `Part` would each see a single opaque element instead of the array
/// inside it (a Demonstration whose coordinate table is pasted that way then
/// computes nothing at all).
fn strip_display_form_wrapper(s: &str) -> String {
  let trimmed = s.trim();
  for head in ["InputForm", "OutputForm", "StandardForm", "TraditionalForm"] {
    if !trimmed.starts_with(head) {
      continue;
    }
    if let Some(args) = positional_box_args(head, trimmed)
      && args.len() == 1
    {
      return args.into_iter().next().unwrap_or_default();
    }
  }
  trimmed.to_string()
}

/// The evaluable head of a typeset "big operator" glyph — `∑` → `Sum`,
/// `∏` → `Product`. The base argument of the box is either the named
/// character (`"\[Sum]"`, as written in a `.nb` file) or the Unicode
/// character itself.
fn big_operator_head(base: &str) -> Option<&'static str> {
  let base = base.trim();
  let base = base
    .strip_prefix('"')
    .and_then(|b| b.strip_suffix('"'))
    .unwrap_or(base)
    .trim();
  match base {
    r"\[Sum]" | "∑" => Some("Sum"),
    r"\[Product]" | "∏" => Some("Product"),
    _ => None,
  }
}

/// The items of a `{a, b, …}` list literal, unwrapped and trimmed.
fn braced_list_items(s: &str) -> Option<Vec<&str>> {
  let inner = s.trim().strip_prefix('{')?.strip_suffix('}')?;
  Some(
    split_top_level_commas(inner)
      .into_iter()
      .map(str::trim)
      .collect(),
  )
}

/// `Piecewise[{{v1, c1}, …}]` from the rows of the `GridBox` a piecewise
/// function is typeset as: one row holding the `\[Piecewise]` brace beside
/// an inner `GridBox` of value/condition pairs. The pair tagged
/// `PiecewiseDefault` is the value outside every condition — Wolfram writes
/// an unset default as `Null`, which is its own default (`0`) and so is
/// dropped rather than passed on.
fn piecewise_from_grid(rows_text: &str) -> Option<String> {
  let [row] = braced_list_items(rows_text)?[..] else {
    return None;
  };
  let [brace, grid] = braced_list_items(row)?[..] else {
    return None;
  };
  if brace.trim().trim_matches('"').trim() != r"\[Piecewise]" {
    return None;
  }
  let inner = positional_box_args("GridBox", grid)?;
  let mut pieces = Vec::new();
  let mut default = None;
  for pair in braced_list_items(inner.first()?)? {
    let [value, condition] = braced_list_items(pair)?[..] else {
      return None;
    };
    if condition.contains("PiecewiseDefault") {
      let value = extract_cell_content(value);
      if value.trim() != "Null" {
        default = Some(value);
      }
      continue;
    }
    pieces.push(format!(
      "{{{}, {}}}",
      extract_cell_content(value),
      extract_cell_content(condition)
    ));
  }
  let pieces = pieces.join(", ");
  Some(match default {
    Some(default) => format!("Piecewise[{{{pieces}}}, {default}]"),
    None => format!("Piecewise[{{{pieces}}}]"),
  })
}

/// Is this box the typeset integral sign?
fn is_integral_sign(s: &str) -> bool {
  let s = s.trim();
  let s = s
    .strip_prefix('"')
    .and_then(|b| b.strip_suffix('"'))
    .unwrap_or(s)
    .trim();
  s == r"\[Integral]" || s == "\u{222B}"
}

/// The limits of a typeset integral sign, when `s` is one: `None` for the
/// indefinite `∫`, `Some((lo, hi))` for the `SubsuperscriptBox["∫", lo, hi]`
/// a definite integral is written as. `None` for anything that is not an
/// integral sign at all.
fn integral_limits(s: &str) -> Option<Option<(String, String)>> {
  if let Some(args) =
    positional_box_args("SubsuperscriptBox", s).filter(|a| a.len() == 3)
  {
    return is_integral_sign(&args[0]).then(|| {
      Some((
        extract_cell_content(&args[1]),
        extract_cell_content(&args[2]),
      ))
    });
  }
  is_integral_sign(s).then_some(None)
}

/// The variable of a `RowBox[{"\[DifferentialD]", x}]` — the typeset `ⅆx`
/// that closes an integral body and names its integration variable. The
/// FrontEnd sometimes inserts an explicit space box between the glyph and
/// the variable (`RowBox[{"\[DifferentialD]", " ", x}]`), which is dropped
/// before checking the shape.
fn differential_variable(s: &str) -> Option<String> {
  let inner = s.trim().strip_prefix("RowBox[")?.strip_suffix(']')?;
  let inner = inner.trim().strip_prefix('{')?.strip_suffix('}')?;
  let mut args = split_top_level_commas(inner);
  if args.len() == 3 && args[1].trim().trim_matches('"').trim().is_empty() {
    args.remove(1);
  }
  if args.len() != 2 {
    return None;
  }
  let head = args[0].trim().trim_matches('"').trim();
  (head == r"\[DifferentialD]" || head == "\u{2146}" || head == "\u{F74C}")
    .then(|| extract_cell_content(args[1]))
}

/// Split the boxes that follow an integral sign into the integrand and the
/// variable of the `ⅆx` that closes them. The FrontEnd normally groups the
/// whole body in a row of its own, so a single `RowBox` is unwrapped first.
fn split_integral_body(parts: &[&str]) -> Option<(String, String)> {
  if let [only] = parts
    && let Some(inner) = only.trim().strip_prefix("RowBox[")
    && let Some(inner) = inner.strip_suffix(']')
    && let Some(inner) = inner.trim().strip_prefix('{')
    && let Some(inner) = inner.strip_suffix('}')
  {
    return split_integral_body(&split_top_level_commas(inner));
  }
  let (last, rest) = parts.split_last()?;
  let var = differential_variable(last)?;
  let integrand = extract_rowbox_content(&rest.join(","));
  (!integrand.trim().is_empty()).then_some((integrand, var))
}

/// Split an iterator underscript (`n=1`) at its top-level `=`, ignoring
/// the two-character operators that merely end in `=` (`==`, `<=`, …).
fn split_iterator_assignment(s: &str) -> Option<(&str, &str)> {
  let mut depth = 0i32;
  for (i, c) in s.char_indices() {
    match c {
      '[' | '{' | '(' => depth += 1,
      ']' | '}' | ')' => depth -= 1,
      '=' if depth == 0 => {
        let prev = s[..i].chars().next_back();
        let next = s[i + 1..].chars().next();
        if matches!(prev, Some('=' | '<' | '>' | '!' | ':' | '/' | '+' | '-'))
          || next == Some('=')
        {
          continue;
        }
        return Some((&s[..i], &s[i + 1..]));
      }
      _ => {}
    }
  }
  None
}

/// Recognise a typeset big-operator box (`UnderoverscriptBox["\[Sum]",
/// under, over]`, or the `UnderscriptBox` form with no upper limit) and
/// return its evaluable head together with the iterator specification.
///
/// These boxes stand *before* their body in the enclosing row, so the
/// caller supplies the body: `∑_(n=1)^m f` → `Sum[f, {n, 1, m}]`. The
/// iterator forms mirror what the FrontEnd's own parser produces —
/// `∑_i^m f` → `Sum[f, {i, m}]` and `∑_(n=1) f` → `Sum[f, n = 1]`.
fn big_operator_call(s: &str) -> Option<(&'static str, String)> {
  let args = positional_box_args("UnderoverscriptBox", s)
    .filter(|a| a.len() == 3)
    .or_else(|| {
      positional_box_args("UnderscriptBox", s).filter(|a| a.len() == 2)
    })?;
  let head = big_operator_head(&args[0])?;
  let under = extract_cell_content(&args[1]);
  let iterator = match args.get(2) {
    Some(over) => {
      let over = extract_cell_content(over);
      match split_iterator_assignment(&under) {
        Some((var, lower)) => {
          format!("{{{}, {}, {}}}", var.trim(), lower.trim(), over.trim())
        }
        None => format!("{{{}, {}}}", under.trim(), over.trim()),
      }
    }
    None => under.trim().to_string(),
  };
  Some((head, iterator))
}

/// The variables of a `SubscriptBox["\\[PartialD]", vars]` — the typeset
/// partial-derivative operator. The FrontEnd reads
/// `SubscriptBox["\\[PartialD]", x] f[x]` as `D[f[x], x]` and
/// `SubscriptBox["\\[PartialD]", RowBox[{"x", ",", "x"}]] f[x]` as
/// `D[f[x], x, x]`, so the operand is whatever follows in the row.
fn partial_derivative_vars(s: &str) -> Option<String> {
  let args = positional_box_args("SubscriptBox", s).filter(|a| a.len() == 2)?;
  let base = extract_cell_content(&args[0]);
  (base.trim() == "\u{2202}").then(|| extract_cell_content(&args[1]))
}

/// Extract text from a RowBox expression by concatenating string
/// elements.
fn extract_rowbox_content(s: &str) -> String {
  let s = s.trim();
  let s = s.strip_prefix('{').unwrap_or(s);
  let s = s.strip_suffix('}').unwrap_or(s);

  let parts = split_top_level_commas(s);
  let mut result = String::new();
  for (i, part) in parts.iter().enumerate() {
    let part = part.trim();
    // A big-operator box (`∑`, `∏`) is written before its body, and the
    // FrontEnd groups the operator with exactly its operand in a row of
    // their own — so everything after it here is the body.
    if i + 1 < parts.len()
      && let Some((head, iterator)) = big_operator_call(part)
    {
      let body = extract_rowbox_content(&parts[i + 1..].join(","));
      result.push_str(&format!("{head}[{body}, {iterator}]"));
      break;
    }
    // The integral sign also takes the rest of the row as its body; the `ⅆx`
    // that closes the body names the integration variable, and a
    // `SubsuperscriptBox` sign carries the limits of a definite integral.
    if i + 1 < parts.len()
      && let Some(limits) = integral_limits(part)
      && let Some((integrand, var)) = split_integral_body(&parts[i + 1..])
    {
      let iterator = match limits {
        Some((lo, hi)) => {
          format!("{{{}, {}, {}}}", var.trim(), lo.trim(), hi.trim())
        }
        None => var.trim().to_string(),
      };
      result.push_str(&format!("Integrate[{integrand}, {iterator}]"));
      break;
    }
    // The partial-derivative operator takes the rest of the row as its
    // operand, the same way a big operator does.
    if i + 1 < parts.len()
      && let Some(vars) = partial_derivative_vars(part)
    {
      let body = extract_rowbox_content(&parts[i + 1..].join(","));
      // Parenthesised: the operator is usually juxtaposed with a
      // coefficient (`u ∂ₓc`), and `uD[…]` would read as a call to a
      // symbol named `uD` rather than a product.
      result.push_str(&format!("(D[{body}, {vars}])"));
      break;
    }
    let piece = box_part_source(part);
    // A bare `#`/`##` and a following letter-initial piece are *separate*
    // sibling boxes here (implicit multiplication, e.g. `# Sin[Pi/u]`
    // typeset without a literal space token between them), but gluing
    // their text together verbatim would read back as named-slot syntax
    // (`#Sin` = `Slot["Sin"]`) instead. Insert a space to keep the
    // juxtaposition a product rather than change its meaning.
    if result.ends_with('#')
      && piece
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic())
    {
      result.push(' ');
    }
    result.push_str(&piece);
  }
  result
}

/// The cell source one element of a box row (or of a box template's slot
/// list) stands for.
fn box_part_source(part: &str) -> String {
  let part = part.trim();
  if part.starts_with('"') && part.ends_with('"') && part.len() >= 2 {
    let inner = &part[1..part.len() - 1];
    // A box element whose own text is quoted (`"\"…\""`) is a *string
    // literal* in the cell, not an operator token. Its named characters
    // are content, so `\[GreaterEqual]` stays `≥` rather than collapsing
    // to the ASCII operator `>=` the way a bare `"\[GreaterEqual]"`
    // element between two operands does.
    if inner.starts_with("\\\"") {
      string_literal_source(inner)
    } else {
      unescape_code_string(inner)
    }
  } else if part.starts_with("RowBox[") {
    extract_rowbox_content(&part[7..part.len().saturating_sub(1)])
  } else if let Some(converted) = extract_typeset_box(part) {
    converted
  } else {
    // For non-string tokens, include as-is
    part.to_string()
  }
}

/// Render the argument of a `TextData[...]` wrapper to plain text.
///
/// `TextData` holds a run of inline content — a bare string, a single box
/// (e.g. `ButtonBox`), or a `{...}` list mixing strings and boxes. We
/// concatenate the textual rendering of each element.
fn extract_textdata(s: &str) -> String {
  let s = s.trim();
  if s.starts_with('{') && s.ends_with('}') {
    let inner = &s[1..s.len() - 1];
    return split_top_level_commas(inner)
      .iter()
      .map(|p| render_text_element(p.trim()))
      .collect::<String>();
  }
  render_text_element(s)
}

/// Render one inline element of a `TextData` run to plain text.
fn render_text_element(s: &str) -> String {
  let s = s.trim();

  // Plain string literal (prose).
  if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
    return extract_string_content(s);
  }

  // `StyleBox["text", opts...]` and `ButtonBox["label", opts...]` (hyperlinks)
  // display their first argument; recurse into it so nested styling resolves.
  for head in ["StyleBox", "ButtonBox"] {
    if let Some(rest) = s.strip_prefix(&format!("{head}["))
      && let Ok((inner, _)) = find_matching_bracket(rest)
    {
      let args = split_top_level_commas(inner);
      if let Some(first) = args.first() {
        return render_text_element(first.trim());
      }
    }
  }

  // Inline `Cell[...]` elements inside a TextData run come in two kinds.
  // Styled inline content — `Cell[BoxData[FormBox[…]], "InlineMath"]` and
  // friends — carries real prose (math embedded in a sentence) and must be
  // rendered, otherwise the surrounding text is left with holes: formula
  // styles render as display text (`D_U`, `V²`, equation grids), code
  // styles as evaluable InputForm. Unstyled inline cells are the attached
  // "more info" opener buttons in Demonstrations templates
  // (PaneSelectorBox/TemplateBox chrome) — they carry no textual content,
  // so drop them.
  if let Some(rest) = s.strip_prefix("Cell[")
    && let Ok((inner, _)) = find_matching_bracket(rest)
  {
    let parts = split_top_level_commas(inner);
    let style = parts.iter().skip(1).find_map(|p| {
      // A part may start with a `\`-newline line continuation left over
      // from the .nb file's physical line wrapping — strip it before
      // matching the style string.
      let t = p.trim();
      let t = t.strip_prefix('\\').map_or(t, str::trim_start);
      let is_option = t.contains("->") || t.contains(":>");
      (!is_option && t.starts_with('"') && t.ends_with('"') && t.len() >= 2)
        .then(|| t[1..t.len() - 1].to_string())
    });
    return match style.as_deref() {
      Some("InlineMath" | "InlineFormula") => parts
        .first()
        .map(|c| render_boxes_text(c.trim()))
        .unwrap_or_default(),
      Some("InlineCell" | "InlineCode" | "InlineInput" | "InlineOutput") => {
        parts
          .first()
          .map(|c| extract_cell_content(c.trim()))
          .unwrap_or_default()
      }
      _ => String::new(),
    };
  }

  // Nested RowBox / typeset boxes.
  if s.starts_with("RowBox[") {
    return extract_rowbox_content(&s[7..s.len().saturating_sub(1)]);
  }
  if let Some(conv) = extract_typeset_box(s) {
    return conv;
  }

  s.to_string()
}

/// Whether display text stands on its own inside a flattened fraction, i.e.
/// binds as one unit without help. A single run of symbol characters does;
/// so does a run already wrapped in its own parentheses. Anything holding a
/// top-level operator or space is a compound and needs grouping.
fn reads_as_one_unit(s: &str) -> bool {
  let s = s.trim();
  if s.is_empty() {
    return true;
  }
  // A sign belongs to the atom it precedes: `-1` is as atomic as `1`.
  let body = s.strip_prefix(['-', '+', '\u{2212}']).unwrap_or(s);
  let mut depth = 0i32;
  let mut wrapped_whole = s.starts_with('(');
  for (i, c) in body.char_indices() {
    match c {
      '(' | '[' | '{' => depth += 1,
      ')' | ']' | '}' => {
        depth -= 1;
        // A closing bracket before the end means the leading `(` did not
        // wrap the whole string — `(a+b)/c` is not one unit.
        if depth == 0 && i + c.len_utf8() != body.len() {
          wrapped_whole = false;
        }
      }
      _ if depth == 0
        && (c.is_whitespace() || "+-*/^=<>,;|&\u{2212}".contains(c)) =>
      {
        return false;
      }
      _ => {}
    }
  }
  wrapped_whole || depth == 0
}

/// One side of a flattened fraction. A two-dimensional fraction states its
/// own grouping by being two-dimensional; flattened onto one line that
/// grouping has to come back as parentheses, or `FractionBox[μ/ρ, k/(C_p ρ)]`
/// reads back as `μ/ρ/k/(C_p ρ)` — a different quantity.
fn group_fraction_part(s: &str) -> String {
  if reads_as_one_unit(s) {
    s.trim().to_string()
  } else {
    format!("({})", s.trim())
  }
}

/// Render a box expression as *display* text for a prose (Text) cell —
/// the inline-formula counterpart of `extract_typeset_box`, preferring
/// readable notation over evaluable InputForm: `SubscriptBox["D", "U"]` →
/// `D_U`, `SuperscriptBox["V", "2"]` → `V²`, `FractionBox[a, b]` → `a/b`,
/// and `GridBox` rows on separate lines. Named characters resolve to
/// Unicode (`\[Del]` → `∇`), never to ASCII operator rewrites.
fn render_boxes_text(s: &str) -> String {
  let s = s.trim();

  // Plain string literal.
  if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
    return unescape_string(&s[1..s.len() - 1]);
  }

  // A bare `{…}` list of inline items.
  if s.starts_with('{') && s.ends_with('}') {
    return split_top_level_commas(&s[1..s.len() - 1])
      .iter()
      .map(|p| render_boxes_text(p.trim()))
      .collect::<String>();
  }

  // A `TextData[...]` run inside a box tree (e.g. one inline-math cell's
  // argument accidentally wraps another whole cell — an artifact of
  // copy/pasting one formula into another in the FrontEnd). Its content
  // follows the same prose conventions as a top-level `TextData[...]`, so
  // reuse that extractor rather than treating it as an opaque box.
  if let Some(inner) = s
    .strip_prefix("TextData[")
    .and_then(|r| r.strip_suffix(']'))
  {
    return extract_textdata(inner);
  }

  // A `Cell[...]` nested inside a box tree — the same accidental-nesting
  // artifact from the other side (a `FormBox` argument that is itself a
  // whole inline-math `Cell[...]`). `render_text_element` already knows
  // how to unwrap an inline-math/inline-formula cell's content.
  if s.starts_with("Cell[") {
    return render_text_element(s);
  }

  // Superscripts of digits (and a leading minus) read best as Unicode
  // superscript characters: `V²`, `∇²`, `10⁻³`.
  fn superscript_unicode(s: &str) -> Option<String> {
    s.chars()
      .map(|c| match c {
        '0' => Some('⁰'),
        '1' => Some('¹'),
        '2' => Some('²'),
        '3' => Some('³'),
        '4' => Some('⁴'),
        '5' => Some('⁵'),
        '6' => Some('⁶'),
        '7' => Some('⁷'),
        '8' => Some('⁸'),
        '9' => Some('⁹'),
        '-' => Some('⁻'),
        '+' => Some('⁺'),
        _ => None,
      })
      .collect()
  }

  for head in [
    "BoxData",
    "FormBox",
    "StyleBox",
    "TagBox",
    "AdjustmentBox",
    "FrameBox",
    "ButtonBox",
    "PaneBox",
    "ItemBox",
  ] {
    if let Some(args) = positional_box_args(head, s)
      && let Some(first) = args.first()
    {
      // These wrappers only style/annotate their first argument.
      return render_boxes_text(first);
    }
  }
  if let Some(args) = positional_box_args("InterpretationBox", s)
    && !args.is_empty()
  {
    // The displayed form is the first argument (the second is the value).
    return render_boxes_text(&args[0]);
  }
  if let Some(args) = positional_box_args("RowBox", s)
    && let Some(first) = args.first()
  {
    return render_boxes_text(first);
  }
  if let Some(args) = positional_box_args("FractionBox", s)
    && args.len() == 2
  {
    return format!(
      "{}/{}",
      group_fraction_part(&render_boxes_text(&args[0])),
      group_fraction_part(&render_boxes_text(&args[1]))
    );
  }
  if let Some(args) = positional_box_args("SubscriptBox", s)
    && args.len() == 2
  {
    let base = render_boxes_text(&args[0]);
    let sub = render_boxes_text(&args[1]);
    // A `⟦…⟧` subscript is a `Part` access, which reads as `c⟦1⟧` — no
    // underscore between the base and its brackets.
    if part_spec_inside_double_brackets(&sub).is_some() {
      return format!("{base}{sub}");
    }
    // An empty base carries the script alone (the FrontEnd draws nothing
    // where the base would go), so there is nothing for an underscore to
    // attach to: `SubscriptBox["", SubscriptBox["C", "p"]]` is `C_p`.
    if draws_nothing(&base) {
      return sub;
    }
    return format!("{base}_{sub}");
  }
  if let Some(args) = positional_box_args("SuperscriptBox", s)
    && args.len() == 2
  {
    // Prime-mark superscripts are the typeset derivative (`θ'`), not an
    // exponent.
    if let Some(n) = prime_marks(&args[1]) {
      return format!("{}{}", render_boxes_text(&args[0]), "'".repeat(n));
    }
    let base = render_boxes_text(&args[0]);
    let exp = render_boxes_text(&args[1]);
    return match superscript_unicode(&exp) {
      Some(sup) if !sup.is_empty() => format!("{base}{sup}"),
      _ => format!("{base}^{exp}"),
    };
  }
  if let Some(args) = positional_box_args("SubsuperscriptBox", s)
    && args.len() == 3
  {
    let base = render_boxes_text(&args[0]);
    let sub = render_boxes_text(&args[1]);
    let exp = render_boxes_text(&args[2]);
    return match superscript_unicode(&exp) {
      Some(sup) if !sup.is_empty() => format!("{base}_{sub}{sup}"),
      _ => format!("{base}_{sub}^{exp}"),
    };
  }
  if let Some(args) = positional_box_args("SqrtBox", s)
    && args.len() == 1
  {
    return format!("√{}", render_boxes_text(&args[0]));
  }
  // Limits under and over a base — in prose this is nearly always a big
  // operator: `∑_(n=1)^m`. Multi-token limits get parentheses so the
  // sum's range stays legible, and a big operator keeps a space to its
  // body, which follows it in the enclosing row.
  fn group_limit(limit: &str) -> String {
    if limit.chars().all(|c| c.is_alphanumeric() || c == '.') {
      limit.to_string()
    } else {
      format!("({limit})")
    }
  }
  if let Some(args) = positional_box_args("UnderoverscriptBox", s)
    .filter(|a| a.len() == 3)
    .or_else(|| {
      positional_box_args("UnderscriptBox", s).filter(|a| a.len() == 2)
    })
  {
    let mut out = format!(
      "{}_{}",
      render_boxes_text(&args[0]),
      group_limit(&render_boxes_text(&args[1]))
    );
    if let Some(over) = args.get(2) {
      out.push('^');
      out.push_str(&group_limit(&render_boxes_text(over)));
    }
    if big_operator_head(&args[0]).is_some() {
      out.push(' ');
    }
    return out;
  }
  // A script above the base with nothing below it — a rate constant over a
  // reaction arrow (`⟶^(k₂ᵃ)`), or a hat/bar over a variable.
  if let Some(args) =
    positional_box_args("OverscriptBox", s).filter(|a| a.len() == 2)
  {
    let base = render_boxes_text(&args[0]);
    let over = render_boxes_text(&args[1]);
    // A diacritic (combining or spacing accent) sits directly on the base
    // rather than reading as an exponent: `OverscriptBox["x", "^"]` → `x̂`.
    if let Some(combining) = combining_accent(&over) {
      return format!("{base}{combining}");
    }
    let mut out = format!("{base}^{}", group_limit(&over));
    if big_operator_head(&args[0]).is_some() {
      out.push(' ');
    }
    return out;
  }
  if let Some(args) = positional_box_args("GridBox", s)
    && let Some(rows_text) = args.first()
  {
    // Rows on separate lines, columns separated by two spaces.
    let rows_inner = rows_text
      .strip_prefix('{')
      .and_then(|r| r.strip_suffix('}'))
      .unwrap_or(rows_text);
    return split_top_level_commas(rows_inner)
      .iter()
      .map(|row| {
        let row = row.trim();
        let row_inner = row
          .strip_prefix('{')
          .and_then(|r| r.strip_suffix('}'))
          .unwrap_or(row);
        split_top_level_commas(row_inner)
          .iter()
          .map(|cell| render_boxes_text(cell.trim()))
          .collect::<Vec<_>>()
          .join("  ")
      })
      .collect::<Vec<_>>()
      .join("\n");
  }

  // Anything else falls back to the evaluable-InputForm extractor.
  extract_cell_content(s)
}

/// The Unicode combining mark for an accent placed over a base by an
/// `OverscriptBox` — `OverHat[x]`, `OverBar[x]`, `OverVector[x]` and friends
/// all typeset that way. `None` for anything that reads as a script rather
/// than a diacritic (a rate constant over a reaction arrow, say).
pub(crate) fn combining_accent(over: &str) -> Option<&'static str> {
  match over.trim() {
    "^" | "\\[Hat]" | "\u{F759}" => Some("\u{0302}"),
    "~" | "\\[Tilde]" | "\u{223C}" => Some("\u{0303}"),
    "." => Some("\u{0307}"),
    ".." => Some("\u{0308}"),
    "_" | "\\[Macron]" | "\u{00AF}" => Some("\u{0304}"),
    "\\[RightVector]" | "\u{21C0}" => Some("\u{20D7}"),
    _ => None,
  }
}

/// Map Wolfram named operator characters to their InputForm ASCII
/// equivalents for display in code cells. The Wolfram FrontEnd encodes
/// operators like `->` as `\[Rule]` (private-use U+F522), which has no
/// glyph in normal fonts. Returning ASCII keeps the cell editable.
///
/// Display-only "invisible" characters — `\[NoBreak]`, `\[InvisibleSpace]`,
/// `\[InvisibleTimes]`, `\[InvisibleComma]`, `\[ImplicitPlus]` — exist purely
/// to influence pretty-printing in the FrontEnd. They have no meaning at
/// evaluation time, so we strip them when reconstructing InputForm text
/// from box expressions to avoid the parser tripping over the leftover
/// `\[…]` tokens.
fn named_char_to_code_op(name: &str) -> Option<&'static str> {
  Some(match name {
    "Rule" => "->",
    "RuleDelayed" => ":>",
    "DirectedEdge" => "\\[DirectedEdge]",
    "UndirectedEdge" => "\\[UndirectedEdge]",
    "Distributed" => "\\[Distributed]",
    "Conditioned" => "\\[Conditioned]",
    // \[Equal] is the typeset name for the `==` comparison operator. The
    // default Wolfram→Unicode mapping is U+003D (`=`), which is `Set`
    // (assignment) at evaluation time — definitely not what the box
    // expression means. Force the operator form here so cells like
    //   RowBox[{"prob2", "\[Equal]", RowBox[{"3", "prob4"}]}]
    // parse as a predicate, not an assignment.
    "Equal" => "==",
    // `\[Prime]` is the typeset derivative mark (`f\[Prime]` = `f'`).
    // The Unicode prime (U+2032) is not a valid InputForm operator, so
    // map it to the ASCII apostrophe to keep the cell evaluable.
    "Prime" => "'",
    "NotEqual" => "!=",
    "LessEqual" => "<=",
    "GreaterEqual" => ">=",
    "And" => "&&",
    "Or" => "||",
    "Cross" => "\\[Cross]",
    "NoBreak"
    | "InvisibleSpace"
    | "InvisibleComma"
    | "ImplicitPlus"
    | "AutoSpace"
    | "ZeroWidthSpace"
    | "NonBreakingSpace"
    | "InvisiblePrefixScriptBase"
    | "InvisiblePostfixScriptBase"
    | "Null"
    | "SpanFromLeft"
    | "SpanFromAbove"
    | "SpanFromBoth"
    | "RawEscape"
    | "RawBackspace" => "",
    // The FrontEnd's own newline (U+F3A3) separates statements in a typeset
    // cell; the reconstructed code needs a plain line break.
    "IndentingNewLine" => "\n",
    // Typographic spacing characters separate tokens in typeset code
    // (e.g. `"/.", "\[VeryThinSpace]", "sol"`). Emit a plain ASCII space
    // so the reconstructed code carries no invisible Unicode.
    "ThinSpace"
    | "VeryThinSpace"
    | "MediumSpace"
    | "ThickSpace"
    | "NegativeThinSpace"
    | "NegativeVeryThinSpace"
    | "NegativeMediumSpace"
    | "NegativeThickSpace" => " ",
    // `\[InvisibleTimes]` represents implicit multiplication (e.g. `2 x`
    // is encoded as `2 \[InvisibleTimes] x` in box form). Map it to an
    // explicit `*` so the resulting InputForm parses as multiplication
    // rather than producing two adjacent tokens that mean nothing.
    "InvisibleTimes" => "*",
    _ => return None,
  })
}

/// Extract a plain string value, handling escaped quotes.
fn extract_string_content(s: &str) -> String {
  let s = s.trim();
  if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
    let inner = &s[1..s.len() - 1];
    let result = unescape_string(inner);
    // When the raw string used \<...\> delimiters (multi-line text cells),
    // the line-continuation newlines adjacent to the delimiters produce
    // spurious leading/trailing newlines – trim them.
    if inner.starts_with("\\<") || inner.ends_with("\\>") {
      result.trim_matches('\n').to_string()
    } else {
      result
    }
  } else {
    s.to_string()
  }
}

/// The cell source for a box element that is itself a quoted string.
///
/// Unescaping such an element yields the string's *value*; any quote inside
/// that value has to be escaped again for the result to read back as a single
/// literal. A `Plot` legend written as an inline cell — `"\!\(\*Cell[\"f[x]\",
/// ExpressionUUID->\"…\"]\)"` — otherwise ends its string at the first inner
/// quote and leaves the rest of the option as stray tokens.
fn string_literal_source(inner: &str) -> String {
  let value = unescape_string(inner);
  match value.strip_prefix('"').and_then(|v| v.strip_suffix('"')) {
    Some(body) if body.contains('"') => {
      format!("\"{}\"", escape_unescaped_quotes(body))
    }
    _ => value,
  }
}

/// Escape every `"` in `body` that isn't already preceded by a backslash.
///
/// A body produced by [`unescape_string`] can already contain a *literal*
/// `\"` pair — e.g. a doubly nested cell string whose `\\\"` raw box text
/// decodes to a real backslash followed by a real quote, which is already
/// valid escaped-quote source. Blindly escaping every `"` (regardless of
/// what precedes it) would double the backslash and corrupt that source;
/// walking the string and skipping over existing `\`-escaped pairs avoids
/// that.
fn escape_unescaped_quotes(body: &str) -> String {
  let mut result = String::with_capacity(body.len());
  let mut chars = body.chars();
  while let Some(c) = chars.next() {
    if c == '\\' {
      result.push(c);
      if let Some(next) = chars.next() {
        result.push(next);
      }
    } else if c == '"' {
      result.push_str("\\\"");
    } else {
      result.push(c);
    }
  }
  result
}

/// Unescape Wolfram-style string escapes.
fn unescape_string(s: &str) -> String {
  unescape_string_inner(s, false)
}

/// Unescape a string from a code cell (BoxData/RowBox), preferring
/// ASCII operator forms (e.g. `\[Rule]` → `->`) for editability.
fn unescape_code_string(s: &str) -> String {
  unescape_string_inner(s, true)
}

fn unescape_string_inner(s: &str, code: bool) -> String {
  let mut result = String::with_capacity(s.len());
  let mut chars = s.chars();
  while let Some(c) = chars.next() {
    if c == '\\' {
      match chars.next() {
        Some('n') => result.push('\n'),
        Some('t') => result.push('\t'),
        Some('\\') => result.push('\\'),
        Some('"') => result.push('"'),
        Some('<') => {
          // \< is a Wolfram string delimiter in box expressions – skip
        }
        Some('>') => {
          // \> is a Wolfram string delimiter in box expressions – skip
        }
        Some('[') => {
          // Wolfram named character like \[Alpha] / \[CloseCurlyQuote].
          // Translate to Unicode when known; otherwise keep \[Name].
          let mut name = String::new();
          for ch in chars.by_ref() {
            if ch == ']' {
              break;
            }
            name.push(ch);
          }
          if code && let Some(op) = named_char_to_code_op(&name) {
            result.push_str(op);
            continue;
          }
          // Prose display: `\[Cross]` canonically maps to a private-use
          // codepoint (U+F4A0) with no glyph in normal fonts; the visible
          // multiplication sign is what a text cell means (`40 × 40`), which
          // is a narrower reading than the vector-product ⨯ every other
          // private-use glyph substitution settles on.
          if !code && name == "Cross" {
            result.push('\u{00D7}');
            continue;
          }
          // The non-printing raw control characters set no type: a
          // Demonstration's caption opens its inline formula with a
          // `\[RawEscape]`, which must leave nothing behind rather than
          // print its own name or a control byte.
          if !code && matches!(name.as_str(), "RawEscape" | "RawBackspace") {
            continue;
          }
          match crate::syntax::named_char_to_unicode(&name) {
            // Prose is text to be *drawn*, so the private-use code points
            // Wolfram stores give way to the glyphs a normal font has.
            Some(uni) if !code => result
              .push_str(&crate::syntax::substitute_private_use_glyphs(uni)),
            Some(uni) => result.push_str(uni),
            None => result.push_str(&format!("\\[{name}]")),
          }
        }
        Some('\n') => {
          // `\` at end of line is a Wolfram line continuation: drop the
          // backslash AND the newline, joining the lines into one.
        }
        Some(other) => {
          result.push('\\');
          result.push(other);
        }
        None => result.push('\\'),
      }
    } else {
      result.push(c);
    }
  }
  result
}

/// Extract the `Initialization :> ( … )` code from a saved FrontEnd
/// dynamic-widget dump (the `DynamicModuleBox[…]` text stored in the Output
/// cell of an evaluated `Manipulate[…]`).
///
/// `Manipulate[…, SaveDefinitions -> True]` embeds every definition its body
/// depends on in this Initialization, so running the returned code makes a
/// freshly opened notebook's widget work before any other cell has been
/// evaluated. The saved code is rewritten into plain session-level input:
/// `$CellContext`` prefixes are dropped and the FrontEnd's line-continuation
/// markers (a `\` at end of line in box text) are removed.
pub fn extract_saved_initialization(box_dump: &str) -> Option<String> {
  let mut search_from = 0;
  while let Some(rel) = box_dump[search_from..].find("Initialization") {
    let after_kw = search_from + rel + "Initialization".len();
    let rest = box_dump[after_kw..].trim_start();
    if let Some(rest) = rest.strip_prefix(":>") {
      let rest = rest.trim_start();
      if let Some(body) = rest.strip_prefix('(')
        && let Some(inner) = matching_paren_prefix(body)
      {
        let cleaned = inner.replace("\\\n", "").replace("$CellContext`", "");
        let cleaned = cleaned.trim();
        if !cleaned.is_empty() && cleaned != "None" {
          return Some(cleaned.to_string());
        }
      }
    }
    search_from = after_kw;
  }
  None
}

/// The distinct symbol names Wolfram qualified with `` $CellContext` `` in a
/// saved `Initialization :> ( … )` block — i.e. the names
/// [`extract_saved_initialization`] strips down to bare identifiers.
///
/// `` $CellContext` `` is the DynamicModule's own private context: a
/// Demonstration's `SaveDefinitions -> True` helper (say a `Midpoint` the
/// author wrote before Wolfram ever shipped a built-in of that name) lives
/// there specifically so it can never collide with a same-named symbol
/// anywhere else — including one a *later* Wolfram Language version adds to
/// `System\``. Once the prefix is stripped for evaluation, that protection
/// is gone: the bare name is looked up exactly where the real built-in
/// lives, and redefining a Protected symbol fails. Unprotecting these names
/// first (the caller's job) restores the isolation the prefix used to give.
pub fn saved_initialization_context_symbols(box_dump: &str) -> Vec<String> {
  const PREFIX: &str = "$CellContext`";
  let mut names: Vec<String> = Vec::new();
  let mut search_from = 0;
  while let Some(rel) = box_dump[search_from..].find("Initialization") {
    let after_kw = search_from + rel + "Initialization".len();
    let rest = box_dump[after_kw..].trim_start();
    if let Some(rest) = rest.strip_prefix(":>")
      && let Some(body) = rest.trim_start().strip_prefix('(')
      && let Some(inner) = matching_paren_prefix(body)
    {
      let mut scan_from = 0;
      while let Some(rel2) = inner[scan_from..].find(PREFIX) {
        let start = scan_from + rel2 + PREFIX.len();
        let end = inner[start..]
          .find(|c: char| !(c.is_alphanumeric() || c == '$'))
          .map_or(inner.len(), |i| start + i);
        let name = &inner[start..end];
        if !name.is_empty() && !names.iter().any(|n| n == name) {
          names.push(name.to_string());
        }
        scan_from = end.max(start + 1);
      }
    }
    search_from = after_kw;
  }
  names
}

/// The prefix of `s` up to (excluding) the `)` matching an already-consumed
/// `(`. Skips over string literals, where parentheses are just text.
fn matching_paren_prefix(s: &str) -> Option<&str> {
  let mut depth = 1i32;
  let mut in_string = false;
  let mut prev_backslash = false;

  for (i, c) in s.char_indices() {
    if in_string {
      if c == '"' && !prev_backslash {
        in_string = false;
      }
      prev_backslash = c == '\\' && !prev_backslash;
      continue;
    }
    match c {
      '"' => in_string = true,
      '(' => depth += 1,
      ')' => {
        depth -= 1;
        if depth == 0 {
          return Some(&s[..i]);
        }
      }
      _ => {}
    }
  }
  None
}

/// Find the matching `}` for content that starts right after `{`.
/// Returns (content_inside_braces, remainder_after_brace).
fn find_matching_brace(s: &str) -> Result<(&str, &str), String> {
  let mut depth = 1i32;
  let mut in_string = false;
  let mut prev_backslash = false;

  for (i, c) in s.char_indices() {
    if in_string {
      if c == '"' && !prev_backslash {
        in_string = false;
      }
      prev_backslash = c == '\\' && !prev_backslash;
      continue;
    }

    match c {
      '"' => in_string = true,
      '{' => depth += 1,
      '}' => {
        depth -= 1;
        if depth == 0 {
          return Ok((&s[..i], &s[i + 1..]));
        }
      }
      _ => {}
    }
  }

  Err("Unmatched opening brace".to_string())
}

/// Find the matching `]` for content that starts right after `[`.
/// Returns (content_inside_brackets, remainder_after_bracket).
fn find_matching_bracket(s: &str) -> Result<(&str, &str), String> {
  let mut depth = 1i32;
  let mut in_string = false;
  let mut prev_backslash = false;

  for (i, c) in s.char_indices() {
    if in_string {
      if c == '"' && !prev_backslash {
        in_string = false;
      }
      prev_backslash = c == '\\' && !prev_backslash;
      continue;
    }

    match c {
      '"' => in_string = true,
      '[' => depth += 1,
      ']' => {
        depth -= 1;
        if depth == 0 {
          return Ok((&s[..i], &s[i + 1..]));
        }
      }
      _ => {}
    }
  }

  Err("Unmatched opening bracket".to_string())
}

/// Split a string on commas at the top level (not inside brackets,
/// braces, parentheses, or strings).
fn split_top_level_commas(s: &str) -> Vec<&str> {
  let mut parts = Vec::new();
  let mut depth = 0i32;
  let mut in_string = false;
  let mut prev_backslash = false;
  let mut start = 0;

  for (i, c) in s.char_indices() {
    if in_string {
      if c == '"' && !prev_backslash {
        in_string = false;
      }
      prev_backslash = c == '\\' && !prev_backslash;
      continue;
    }

    match c {
      '"' => in_string = true,
      '{' | '[' | '(' => depth += 1,
      '}' | ']' | ')' => depth -= 1,
      ',' if depth == 0 => {
        parts.push(&s[start..i]);
        start = i + 1;
      }
      _ => {}
    }
  }

  if start < s.len() {
    parts.push(&s[start..]);
  }

  parts
}

// ── Serialization ───────────────────────────────────────────────────

/// Escape a string for Wolfram Language output.
fn escape_string(s: &str) -> String {
  let mut result = String::with_capacity(s.len() + 8);
  for c in s.chars() {
    match c {
      '"' => result.push_str("\\\""),
      '\\' => result.push_str("\\\\"),
      '\n' => result.push_str("\\n"),
      '\t' => result.push_str("\\t"),
      _ => result.push(c),
    }
  }
  result
}

impl fmt::Display for Notebook {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    writeln!(f, "Notebook[{{")?;
    for (i, entry) in self.cells.iter().enumerate() {
      if i > 0 {
        writeln!(f, ",")?;
      }
      write!(f, "{entry}")?;
    }
    writeln!(f)?;
    write!(f, "}}]")
  }
}

impl fmt::Display for CellEntry {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Single(cell) => write!(f, "{cell}"),
      Self::Group(group) => write!(f, "{group}"),
    }
  }
}

impl fmt::Display for CellGroup {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "Cell[CellGroupData[{{")?;
    for (i, cell) in self.cells.iter().enumerate() {
      if i > 0 {
        write!(f, ",")?;
      }
      writeln!(f)?;
      write!(f, "{cell}")?;
    }
    writeln!(f)?;
    write!(f, "}}, {}]]", if self.open { "Open" } else { "Closed" })
  }
}

impl fmt::Display for Cell {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    // Options that apply to any cell style.
    let options = if self.collapsed
      && matches!(self.style, CellStyle::Chapter | CellStyle::Subchapter)
    {
      ", CellOpen -> False"
    } else {
      ""
    };

    match self.style {
      CellStyle::Input | CellStyle::Code => {
        // For input cells, wrap content in BoxData
        let lines: Vec<&str> = self.content.lines().collect();
        if lines.len() <= 1 {
          write!(
            f,
            "Cell[BoxData[\"{}\"], \"{}\"{}]",
            escape_string(&self.content),
            self.style,
            options
          )
        } else {
          // Multi-line: use RowBox with \n separators
          write!(f, "Cell[BoxData[RowBox[{{")?;
          for (i, line) in lines.iter().enumerate() {
            if i > 0 {
              write!(f, ", \"\\n\", ")?;
            }
            write!(f, "\"{}\"", escape_string(line))?;
          }
          write!(f, "}}]], \"{}\"{}]", self.style, options)
        }
      }
      CellStyle::Output | CellStyle::Print => {
        write!(
          f,
          "Cell[BoxData[\"{}\"], \"{}\"{}]",
          escape_string(&self.content),
          self.style,
          options
        )
      }
      _ => {
        // Text-style cells: Cell["content", "Style"]
        write!(
          f,
          "Cell[\"{}\", \"{}\"{}]",
          escape_string(&self.content),
          self.style,
          options
        )
      }
    }
  }
}

// ── Convenience constructors ────────────────────────────────────────

impl Default for Notebook {
  fn default() -> Self {
    Self::new()
  }
}

impl Notebook {
  /// Create an empty notebook.
  pub fn new() -> Self {
    Self { cells: Vec::new() }
  }

  /// Add a single cell at the end.
  pub fn push_cell(&mut self, cell: Cell) {
    self.cells.push(CellEntry::Single(cell));
  }

  /// Add a cell group (e.g. input + output pair).
  #[allow(dead_code)]
  pub fn push_group(&mut self, cells: Vec<Cell>) {
    self
      .cells
      .push(CellEntry::Group(CellGroup { cells, open: true }));
  }

  /// Flatten all cells into a flat ordered list with their group
  /// index.
  pub fn flat_cells(&self) -> Vec<(usize, &Cell)> {
    let mut result = Vec::new();
    for (group_idx, entry) in self.cells.iter().enumerate() {
      match entry {
        CellEntry::Single(cell) => {
          result.push((group_idx, cell));
        }
        CellEntry::Group(group) => {
          for cell in &group.cells {
            result.push((group_idx, cell));
          }
        }
      }
    }
    result
  }
}

impl Cell {
  pub fn new(style: CellStyle, content: impl Into<String>) -> Self {
    Self {
      style,
      content: content.into(),
      collapsed: false,
    }
  }
}

// ── Export formats ──────────────────────────────────────────────

/// Escape a string for JSON output.
fn escape_json(s: &str) -> String {
  let mut result = String::with_capacity(s.len() + 8);
  for c in s.chars() {
    match c {
      '"' => result.push_str("\\\""),
      '\\' => result.push_str("\\\\"),
      '\n' => result.push_str("\\n"),
      '\r' => result.push_str("\\r"),
      '\t' => result.push_str("\\t"),
      c if (c as u32) < 0x20 => {
        result.push_str(&format!("\\u{:04x}", c as u32));
      }
      _ => result.push(c),
    }
  }
  result
}

/// Format a string as a JSON array of source lines (Jupyter convention).
fn json_source_lines(content: &str) -> String {
  if content.is_empty() {
    return "[\"\"]".to_string();
  }
  let lines: Vec<&str> = content.split('\n').collect();
  let mut parts = Vec::new();
  for (i, line) in lines.iter().enumerate() {
    if i < lines.len() - 1 {
      parts.push(format!("\"{}\\n\"", escape_json(line)));
    } else {
      parts.push(format!("\"{}\"", escape_json(line)));
    }
  }
  format!("[{}]", parts.join(", "))
}

/// Convert a cell to its Markdown heading representation.
fn heading_markdown(cell: &Cell) -> String {
  match cell.style {
    CellStyle::Title => format!("# {}", cell.content),
    CellStyle::Subtitle => format!("*{}*", cell.content),
    CellStyle::Chapter => format!("## {}", cell.content),
    CellStyle::Subchapter => format!("### {}", cell.content),
    CellStyle::Section => format!("#### {}", cell.content),
    CellStyle::Subsection => format!("##### {}", cell.content),
    CellStyle::Subsubsection => {
      format!("###### {}", cell.content)
    }
    CellStyle::Item => format!("- {}", cell.content),
    CellStyle::Subitem => format!("  - {}", cell.content),
    _ => cell.content.clone(),
  }
}

/// Escape special LaTeX characters in text.
fn escape_latex(s: &str) -> String {
  let mut result = String::with_capacity(s.len() + 8);
  for c in s.chars() {
    match c {
      '#' => result.push_str("\\#"),
      '$' => result.push_str("\\$"),
      '%' => result.push_str("\\%"),
      '&' => result.push_str("\\&"),
      '_' => result.push_str("\\_"),
      '{' => result.push_str("\\{"),
      '}' => result.push_str("\\}"),
      '~' => result.push_str("\\textasciitilde{}"),
      '^' => result.push_str("\\textasciicircum{}"),
      '\\' => result.push_str("\\textbackslash{}"),
      _ => result.push(c),
    }
  }
  result
}

fn jupyter_markdown_cell(source: &str) -> String {
  let mut out = String::new();
  out.push_str("    {\n");
  out.push_str("      \"cell_type\": \"markdown\",\n");
  out.push_str("      \"metadata\": {},\n");
  out.push_str(&format!(
    "      \"source\": {}\n",
    json_source_lines(source)
  ));
  out.push_str("    }");
  out
}

fn jupyter_code_cell(
  source: &str,
  outputs: &[&Cell],
  exec_count: u32,
) -> String {
  let mut out = String::new();
  out.push_str("    {\n");
  out.push_str("      \"cell_type\": \"code\",\n");
  out.push_str(&format!("      \"execution_count\": {exec_count},\n"));
  out.push_str("      \"metadata\": {},\n");
  out.push_str(&format!(
    "      \"source\": {},\n",
    json_source_lines(source)
  ));

  if outputs.is_empty() {
    out.push_str("      \"outputs\": []\n");
  } else {
    out.push_str("      \"outputs\": [\n");
    let mut output_parts = Vec::new();
    for cell in outputs {
      let mut o = String::new();
      if cell.style == CellStyle::Print {
        o.push_str("        {\n");
        o.push_str("          \"output_type\": \"stream\",\n");
        o.push_str("          \"name\": \"stdout\",\n");
        o.push_str(&format!(
          "          \"text\": {}\n",
          json_source_lines(&cell.content)
        ));
        o.push_str("        }");
      } else {
        o.push_str("        {\n");
        o.push_str("          \"output_type\": \"execute_result\",\n");
        o.push_str(&format!("          \"execution_count\": {exec_count},\n"));
        o.push_str("          \"data\": {\n");
        o.push_str(&format!(
          "            \"text/plain\": {}\n",
          json_source_lines(&cell.content)
        ));
        o.push_str("          },\n");
        o.push_str("          \"metadata\": {}\n");
        o.push_str("        }");
      }
      output_parts.push(o);
    }
    out.push_str(&output_parts.join(",\n"));
    out.push('\n');
    out.push_str("      ]\n");
  }

  out.push_str("    }");
  out
}

impl Notebook {
  /// Export as Markdown.
  pub fn to_markdown(&self) -> String {
    let mut out = String::new();
    for (_, cell) in self.flat_cells() {
      match cell.style {
        CellStyle::Title => {
          out.push_str(&format!("# {}\n\n", cell.content));
        }
        CellStyle::Subtitle => {
          out.push_str(&format!("*{}*\n\n", cell.content));
        }
        CellStyle::Chapter => {
          out.push_str(&format!("## {}\n\n", cell.content));
        }
        CellStyle::Subchapter => {
          out.push_str(&format!("### {}\n\n", cell.content));
        }
        CellStyle::Section => {
          out.push_str(&format!("#### {}\n\n", cell.content));
        }
        CellStyle::Subsection => {
          out.push_str(&format!("##### {}\n\n", cell.content));
        }
        CellStyle::Subsubsection => {
          out.push_str(&format!("###### {}\n\n", cell.content));
        }
        CellStyle::Text => {
          out.push_str(&format!("{}\n\n", cell.content));
        }
        CellStyle::Item => {
          out.push_str(&format!("- {}\n\n", cell.content));
        }
        CellStyle::Subitem => {
          out.push_str(&format!("  - {}\n\n", cell.content));
        }
        CellStyle::Input | CellStyle::Code => {
          out.push_str(&format!("```wolfram\n{}\n```\n\n", cell.content));
        }
        CellStyle::Output | CellStyle::Print => {
          out.push_str(&format!("```\n{}\n```\n\n", cell.content));
        }
      }
    }
    out.trim_end().to_string()
  }

  /// Export as LaTeX.
  pub fn to_latex(&self) -> String {
    let flat = self.flat_cells();
    let mut out = String::new();

    out.push_str("\\documentclass{article}\n");
    out.push_str("\\usepackage[utf8]{inputenc}\n\n");

    // Extract first title for \title{} / \maketitle
    let has_title = flat.iter().any(|(_, c)| c.style == CellStyle::Title);
    if let Some((_, cell)) =
      flat.iter().find(|(_, c)| c.style == CellStyle::Title)
    {
      out.push_str(&format!("\\title{{{}}}\n", escape_latex(&cell.content)));
      out.push_str("\\date{}\n");
    }

    out.push_str("\n\\begin{document}\n\n");

    if has_title {
      out.push_str("\\maketitle\n\n");
    }

    let mut first_title_skipped = false;
    for (_, cell) in &flat {
      match cell.style {
        CellStyle::Title => {
          if !first_title_skipped {
            first_title_skipped = true;
            continue;
          }
          out.push_str(&format!(
            "\\section*{{{}}}\n\n",
            escape_latex(&cell.content)
          ));
        }
        CellStyle::Subtitle => {
          out.push_str(&format!(
            "\\begin{{center}}\n\\large \\textit{{{}}}\n\\end{{center}}\n\n",
            escape_latex(&cell.content)
          ));
        }
        CellStyle::Chapter => {
          out.push_str(&format!(
            "\\chapter*{{{}}}\n\n",
            escape_latex(&cell.content)
          ));
        }
        CellStyle::Subchapter => {
          out.push_str(&format!(
            "\\section*{{{}}}\n\n",
            escape_latex(&cell.content)
          ));
        }
        CellStyle::Section => {
          out.push_str(&format!(
            "\\section{{{}}}\n\n",
            escape_latex(&cell.content)
          ));
        }
        CellStyle::Subsection => {
          out.push_str(&format!(
            "\\subsection{{{}}}\n\n",
            escape_latex(&cell.content)
          ));
        }
        CellStyle::Subsubsection => {
          out.push_str(&format!(
            "\\subsubsection{{{}}}\n\n",
            escape_latex(&cell.content)
          ));
        }
        CellStyle::Text => {
          out.push_str(&escape_latex(&cell.content));
          out.push_str("\n\n");
        }
        CellStyle::Item => {
          out.push_str(&format!(
            "\\begin{{itemize}}\n\\item {}\n\\end{{itemize}}\n\n",
            escape_latex(&cell.content)
          ));
        }
        CellStyle::Subitem => {
          out.push_str(&format!(
            "\\begin{{itemize}}\n\\item \\begin{{itemize}}\n\\item {}\n\\end{{itemize}}\n\\end{{itemize}}\n\n",
            escape_latex(&cell.content)
          ));
        }
        CellStyle::Input | CellStyle::Code => {
          out.push_str("\\begin{verbatim}\n");
          out.push_str(&cell.content);
          out.push_str("\n\\end{verbatim}\n\n");
        }
        CellStyle::Output | CellStyle::Print => {
          out.push_str("\\begin{verbatim}\n");
          out.push_str(&cell.content);
          out.push_str("\n\\end{verbatim}\n\n");
        }
      }
    }

    out.push_str("\\end{document}\n");
    out
  }

  /// Export as Typst.
  pub fn to_typst(&self) -> String {
    let mut out = String::new();
    for (_, cell) in self.flat_cells() {
      match cell.style {
        CellStyle::Title => {
          out.push_str(&format!("= {}\n\n", cell.content));
        }
        CellStyle::Subtitle => {
          out.push_str(&format!("_{}_\n\n", cell.content));
        }
        CellStyle::Chapter => {
          out.push_str(&format!("== {}\n\n", cell.content));
        }
        CellStyle::Subchapter => {
          out.push_str(&format!("=== {}\n\n", cell.content));
        }
        CellStyle::Section => {
          out.push_str(&format!("==== {}\n\n", cell.content));
        }
        CellStyle::Subsection => {
          out.push_str(&format!("===== {}\n\n", cell.content));
        }
        CellStyle::Subsubsection => {
          out.push_str(&format!("====== {}\n\n", cell.content));
        }
        CellStyle::Text => {
          out.push_str(&format!("{}\n\n", cell.content));
        }
        CellStyle::Item => {
          out.push_str(&format!("- {}\n\n", cell.content));
        }
        CellStyle::Subitem => {
          out.push_str(&format!("  - {}\n\n", cell.content));
        }
        CellStyle::Input | CellStyle::Code => {
          out.push_str(&format!("```wl\n{}\n```\n\n", cell.content));
        }
        CellStyle::Output | CellStyle::Print => {
          out.push_str(&format!("```\n{}\n```\n\n", cell.content));
        }
      }
    }
    out.trim_end().to_string()
  }

  /// Export as Jupyter Notebook (JSON).
  pub fn to_jupyter(&self) -> String {
    let mut cells_json = Vec::new();
    let mut exec_count = 1u32;

    for entry in &self.cells {
      match entry {
        CellEntry::Single(cell) => match cell.style {
          CellStyle::Input | CellStyle::Code => {
            cells_json.push(jupyter_code_cell(&cell.content, &[], exec_count));
            exec_count += 1;
          }
          CellStyle::Output | CellStyle::Print => {
            cells_json.push(jupyter_markdown_cell(&format!(
              "```\n{}\n```",
              cell.content
            )));
          }
          _ => {
            cells_json.push(jupyter_markdown_cell(&heading_markdown(cell)));
          }
        },
        CellEntry::Group(group) => {
          let Some(first) = group.cells.first() else {
            continue;
          };
          if first.style == CellStyle::Input || first.style == CellStyle::Code {
            let outputs: Vec<&Cell> = group.cells[1..]
              .iter()
              .filter(|c| {
                c.style == CellStyle::Output || c.style == CellStyle::Print
              })
              .collect();
            cells_json.push(jupyter_code_cell(
              &first.content,
              &outputs,
              exec_count,
            ));
            exec_count += 1;
          } else {
            for cell in &group.cells {
              match cell.style {
                CellStyle::Input | CellStyle::Code => {
                  cells_json.push(jupyter_code_cell(
                    &cell.content,
                    &[],
                    exec_count,
                  ));
                  exec_count += 1;
                }
                _ => {
                  cells_json
                    .push(jupyter_markdown_cell(&heading_markdown(cell)));
                }
              }
            }
          }
        }
      }
    }

    let mut out = String::new();
    out.push_str("{\n");
    out.push_str("  \"nbformat\": 4,\n");
    out.push_str("  \"nbformat_minor\": 5,\n");
    out.push_str("  \"metadata\": {\n");
    out.push_str("    \"kernelspec\": {\n");
    out.push_str("      \"display_name\": \"Wolfram Language\",\n");
    out.push_str("      \"language\": \"wolfram\",\n");
    out.push_str("      \"name\": \"wolfram\"\n");
    out.push_str("    }\n");
    out.push_str("  },\n");
    out.push_str("  \"cells\": [\n");
    out.push_str(&cells_json.join(",\n"));
    out.push('\n');
    out.push_str("  ]\n");
    out.push_str("}\n");
    out
  }
}

// ── Stored-output rendering ─────────────────────────────────────────────
//
// Notebooks saved by Mathematica can carry pre-rendered results that Woxi
// cannot regenerate: `RasterBox[CompressedData["…"]]` snapshot images and
// `CheckboxBox[…]` grids (the Demonstrations submission templates). These
// helpers decode such stored Output cells into something displayable.

/// Decode the `RasterBox[CompressedData["…"]]` image embedded in a stored
/// Output cell into an SVG that embeds the pixels as a PNG data URI.
/// Returns `None` when the content holds no decodable raster.
///
/// The compressed payload is Mathematica's binary serialization of
/// `RawArray["UnsignedInteger8", pixels]`: `!boR` magic, an `f`unction
/// header naming `RawArray`, a `S`tring type tag, and a `b`yte-array tag
/// with rank, dimensions (`{height, width, channels}` or
/// `{height, width}`), and the raw samples.
pub fn stored_output_image_svg(content: &str) -> Option<String> {
  let start = content.find("RasterBox[CompressedData[\"")?;
  let rest = &content[start + "RasterBox[CompressedData[\"".len()..];
  let end = rest.find('"')?;
  let b64: String = rest[..end]
    .chars()
    .filter(|c| !c.is_whitespace() && *c != '\\')
    .collect();
  let b64 = b64.strip_prefix("1:")?;

  use base64::Engine;
  let compressed =
    base64::engine::general_purpose::STANDARD.decode(b64).ok()?;
  let mut raw = Vec::new();
  std::io::Read::read_to_end(
    &mut flate2::read::ZlibDecoder::new(&compressed[..]),
    &mut raw,
  )
  .ok()?;

  let (height, width, channels, pixels) = parse_raw_array_u8(&raw)?;

  // Encode as PNG (top row first, matching Image/Rasterize order).
  let mut png = Vec::new();
  let color = match channels {
    1 => image::ExtendedColorType::L8,
    3 => image::ExtendedColorType::Rgb8,
    4 => image::ExtendedColorType::Rgba8,
    _ => return None,
  };
  use image::ImageEncoder;
  image::codecs::png::PngEncoder::new(&mut png)
    .write_image(pixels, width, height, color)
    .ok()?;
  let png_b64 = base64::engine::general_purpose::STANDARD.encode(&png);

  // Display at the notebook's typical pane width; the viewBox keeps the
  // native resolution so zooming stays sharp.
  let max_display = 450.0_f64;
  let scale = (max_display / width as f64).min(1.0);
  let dw = (width as f64 * scale).round() as u32;
  let dh = (height as f64 * scale).round() as u32;
  Some(format!(
    "<svg width=\"{dw}\" height=\"{dh}\" viewBox=\"0 0 {width} {height}\" \
     xmlns=\"http://www.w3.org/2000/svg\" \
     xmlns:xlink=\"http://www.w3.org/1999/xlink\">\
     <image width=\"{width}\" height=\"{height}\" \
     xlink:href=\"data:image/png;base64,{png_b64}\"/></svg>"
  ))
}

/// Parse Mathematica's serialized `RawArray["UnsignedInteger8", …]`:
/// returns `(height, width, channels, samples)`. Rank-2 arrays are
/// grayscale (`channels = 1`).
fn parse_raw_array_u8(raw: &[u8]) -> Option<(u32, u32, u8, &[u8])> {
  let mut pos = 0usize;
  let take = |pos: &mut usize, n: usize| -> Option<&[u8]> {
    let s = raw.get(*pos..*pos + n)?;
    *pos += n;
    Some(s)
  };
  let read_u32 = |pos: &mut usize| -> Option<u32> {
    take(pos, 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()))
  };

  if take(&mut pos, 4)? != b"!boR" {
    return None;
  }
  // f <argc> s <len> "RawArray"
  if take(&mut pos, 1)? != b"f" {
    return None;
  }
  let _argc = read_u32(&mut pos)?;
  if take(&mut pos, 1)? != b"s" {
    return None;
  }
  let name_len = read_u32(&mut pos)? as usize;
  if take(&mut pos, name_len)? != b"RawArray" {
    return None;
  }
  // S <len> "UnsignedInteger8"
  if take(&mut pos, 1)? != b"S" {
    return None;
  }
  let ty_len = read_u32(&mut pos)? as usize;
  if take(&mut pos, ty_len)? != b"UnsignedInteger8" {
    return None;
  }
  // b <rank> <dim…> <samples>
  if take(&mut pos, 1)? != b"b" {
    return None;
  }
  let rank = read_u32(&mut pos)? as usize;
  if !(2..=3).contains(&rank) {
    return None;
  }
  let mut dims = [0u32; 3];
  for d in dims.iter_mut().take(rank) {
    *d = read_u32(&mut pos)?;
  }
  let (height, width, channels) = if rank == 2 {
    (dims[0], dims[1], 1u8)
  } else {
    (dims[0], dims[1], u8::try_from(dims[2]).ok()?)
  };
  let expected = height as usize * width as usize * channels as usize;
  let pixels = raw.get(pos..pos + expected)?;
  Some((height, width, channels, pixels))
}

/// Collect the checkbox entries of an already-extracted grid, i.e. the
/// nested list of `☐ label` / `☑ label` strings `extract_cell_content`
/// leaves behind for a `GridBox` of checkboxes. Entries are appended in
/// source order as `[ ] label` / `[x] label`; anything that is not a
/// checkbox glyph is skipped (the trailing `""` padding cells of a ragged
/// category grid, for instance).
fn collect_checkbox_glyphs(s: &str, out: &mut Vec<String>) {
  let t = s.trim();
  if let Some(inner) = t.strip_prefix('{').and_then(|r| r.strip_suffix('}')) {
    let mut grew = false;
    for part in split_top_level_commas(inner) {
      let part = part.trim();
      let is_entry = part.starts_with('{')
        || part.starts_with('\u{2610}')
        || part.starts_with('\u{2611}');
      if !is_entry {
        // The labels are unquoted, so one containing a comma ("Systems,
        // Models & Methods") was split by the enclosing list — stitch the
        // fragment back onto the entry it came from.
        if grew && let Some(last) = out.last_mut() {
          last.push_str(", ");
          last.push_str(part.trim_matches('"').trim());
        }
        continue;
      }
      let before = out.len();
      collect_checkbox_glyphs(part, out);
      grew = out.len() > before;
    }
    return;
  }
  let t = t.trim_matches('"').trim();
  if let Some(label) = t.strip_prefix('\u{2611}') {
    out.push(format!("[x] {}", label.trim()).trim_end().to_string());
  } else if let Some(label) = t.strip_prefix('\u{2610}') {
    out.push(format!("[ ] {}", label.trim()).trim_end().to_string());
  }
}

/// Render a stored `CheckboxBox[…]` output (the Demonstrations category /
/// compatibility pickers) as plain text: one `[x] label` / `[ ] label`
/// entry per checkbox, in source order. Accepts both the raw box text and
/// the glyph grid `extract_cell_content` produces from it. Returns `None`
/// when the content holds no checkbox.
pub fn stored_output_checkbox_text(content: &str) -> Option<String> {
  if !content.contains("CheckboxBox[") {
    let mut lines = Vec::new();
    collect_checkbox_glyphs(content, &mut lines);
    return if lines.is_empty() {
      None
    } else {
      Some(lines.join("\n"))
    };
  }
  let mut lines = Vec::new();
  let mut rest = content;
  while let Some(idx) = rest.find("CheckboxBox[") {
    rest = &rest[idx + "CheckboxBox[".len()..];
    let (inner, tail) = find_matching_bracket(rest).ok()?;
    let args = split_top_level_commas(inner);
    let checked = args.first().map(|a| a.trim()) == Some("True");
    // The label is the "on" value when it is a string (`{False, "Math"}`);
    // a plain `{False, True}` checkbox has no label.
    let label = args
      .get(1)
      .map(|a| a.trim())
      .and_then(|vals| {
        let vals = vals.strip_prefix('{')?.strip_suffix('}')?;
        let parts = split_top_level_commas(vals);
        let on = parts.get(1)?.trim();
        if on.starts_with('"') && on.ends_with('"') && on.len() >= 2 {
          Some(extract_string_content(on))
        } else {
          None
        }
      })
      .unwrap_or_default();
    let mark = if checked { "[x]" } else { "[ ]" };
    if label.is_empty() {
      lines.push(mark.to_string());
    } else {
      lines.push(format!("{mark} {label}"));
    }
    rest = tail;
  }
  if lines.is_empty() {
    None
  } else {
    Some(lines.join("\n"))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_parse_simple_notebook() {
    let nb = r#"Notebook[{
Cell["Hello World", "Title"],
Cell["Some explanation", "Text"],
Cell[BoxData["1 + 1"], "Input"]
}]"#;

    let parsed = parse_notebook(nb).unwrap();
    assert_eq!(parsed.cells.len(), 3);

    match &parsed.cells[0] {
      CellEntry::Single(cell) => {
        assert_eq!(cell.style, CellStyle::Title);
        assert_eq!(cell.content, "Hello World");
      }
      CellEntry::Group(_) => panic!("Expected single cell"),
    }

    match &parsed.cells[1] {
      CellEntry::Single(cell) => {
        assert_eq!(cell.style, CellStyle::Text);
        assert_eq!(cell.content, "Some explanation");
      }
      CellEntry::Group(_) => panic!("Expected single cell"),
    }

    match &parsed.cells[2] {
      CellEntry::Single(cell) => {
        assert_eq!(cell.style, CellStyle::Input);
        assert_eq!(cell.content, "1 + 1");
      }
      CellEntry::Group(_) => panic!("Expected single cell"),
    }
  }

  #[test]
  fn test_parse_cell_group() {
    let nb = r#"Notebook[{
Cell[CellGroupData[{
Cell[BoxData["2 + 3"], "Input"],
Cell[BoxData["5"], "Output"]
}, Open]]
}]"#;

    let parsed = parse_notebook(nb).unwrap();
    assert_eq!(parsed.cells.len(), 1);

    match &parsed.cells[0] {
      CellEntry::Group(group) => {
        assert!(group.open);
        assert_eq!(group.cells.len(), 2);
        assert_eq!(group.cells[0].style, CellStyle::Input);
        assert_eq!(group.cells[0].content, "2 + 3");
        assert_eq!(group.cells[1].style, CellStyle::Output);
        assert_eq!(group.cells[1].content, "5");
      }
      CellEntry::Single(_) => panic!("Expected cell group"),
    }
  }

  #[test]
  fn test_roundtrip() {
    let mut nb = Notebook::new();
    nb.push_cell(Cell::new(CellStyle::Title, "My Notebook"));
    nb.push_group(vec![
      Cell::new(CellStyle::Input, "1 + 1"),
      Cell::new(CellStyle::Output, "2"),
    ]);
    nb.push_cell(Cell::new(CellStyle::Text, "Some text"));

    let serialized = nb.to_string();
    let reparsed = parse_notebook(&serialized).unwrap();

    assert_eq!(reparsed.cells.len(), 3);
  }

  #[test]
  fn test_escape_roundtrip() {
    let original = r#"He said "hello" and x\y"#;
    let escaped = escape_string(original);
    let unescaped = unescape_string(&escaped);
    assert_eq!(unescaped, original);
  }

  #[test]
  fn test_split_top_level_commas() {
    let s = r#""a", "b", Cell[1, 2], "c""#;
    let parts = split_top_level_commas(s);
    assert_eq!(parts.len(), 4);
    assert_eq!(parts[0].trim(), "\"a\"");
    assert_eq!(parts[1].trim(), "\"b\"");
    assert_eq!(parts[2].trim(), "Cell[1, 2]");
    assert_eq!(parts[3].trim(), "\"c\"");
  }

  #[test]
  fn test_export_markdown() {
    let mut nb = Notebook::new();
    nb.push_cell(Cell::new(CellStyle::Title, "My Notebook"));
    nb.push_cell(Cell::new(CellStyle::Text, "Some text"));
    nb.push_cell(Cell::new(CellStyle::Section, "Introduction"));
    nb.push_group(vec![
      Cell::new(CellStyle::Input, "1 + 1"),
      Cell::new(CellStyle::Output, "2"),
    ]);

    let md = nb.to_markdown();
    assert!(md.contains("# My Notebook"));
    assert!(md.contains("Some text"));
    assert!(md.contains("#### Introduction"));
    assert!(md.contains("```wolfram\n1 + 1\n```"));
    assert!(md.contains("```\n2\n```"));
  }

  #[test]
  fn test_export_latex() {
    let mut nb = Notebook::new();
    nb.push_cell(Cell::new(CellStyle::Title, "My Notebook"));
    nb.push_cell(Cell::new(CellStyle::Section, "Introduction"));
    nb.push_cell(Cell::new(CellStyle::Text, "Some text"));
    nb.push_group(vec![
      Cell::new(CellStyle::Input, "1 + 1"),
      Cell::new(CellStyle::Output, "2"),
    ]);

    let tex = nb.to_latex();
    assert!(tex.contains("\\documentclass{article}"));
    assert!(tex.contains("\\title{My Notebook}"));
    assert!(tex.contains("\\maketitle"));
    assert!(tex.contains("\\section{Introduction}"));
    assert!(tex.contains("Some text"));
    assert!(tex.contains("\\begin{verbatim}\n1 + 1\n\\end{verbatim}"));
    assert!(tex.contains("\\begin{verbatim}\n2\n\\end{verbatim}"));
  }

  #[test]
  fn test_export_latex_special_chars() {
    let mut nb = Notebook::new();
    nb.push_cell(Cell::new(CellStyle::Text, "Price is $10 & 50% off"));

    let tex = nb.to_latex();
    assert!(tex.contains("\\$"));
    assert!(tex.contains("\\&"));
    assert!(tex.contains("\\%"));
  }

  #[test]
  fn test_export_typst() {
    let mut nb = Notebook::new();
    nb.push_cell(Cell::new(CellStyle::Title, "My Notebook"));
    nb.push_cell(Cell::new(CellStyle::Section, "Introduction"));
    nb.push_cell(Cell::new(CellStyle::Text, "Some text"));
    nb.push_group(vec![
      Cell::new(CellStyle::Input, "1 + 1"),
      Cell::new(CellStyle::Output, "2"),
    ]);

    let typ = nb.to_typst();
    assert!(typ.contains("= My Notebook"));
    assert!(typ.contains("==== Introduction"));
    assert!(typ.contains("Some text"));
    assert!(typ.contains("```wl\n1 + 1\n```"));
    assert!(typ.contains("```\n2\n```"));
  }

  #[test]
  fn test_export_jupyter() {
    let mut nb = Notebook::new();
    nb.push_cell(Cell::new(CellStyle::Title, "My Notebook"));
    nb.push_group(vec![
      Cell::new(CellStyle::Input, "1 + 1"),
      Cell::new(CellStyle::Output, "2"),
    ]);

    let ipynb = nb.to_jupyter();
    assert!(ipynb.contains("\"nbformat\": 4"));
    assert!(ipynb.contains("\"cell_type\": \"markdown\""));
    assert!(ipynb.contains("\"cell_type\": \"code\""));
    assert!(ipynb.contains("\"execute_result\""));
    assert!(ipynb.contains("# My Notebook"));
    assert!(ipynb.contains("1 + 1"));
  }

  #[test]
  fn test_export_jupyter_print_output() {
    let mut nb = Notebook::new();
    nb.push_group(vec![
      Cell::new(CellStyle::Input, "Print[42]"),
      Cell::new(CellStyle::Print, "42"),
    ]);

    let ipynb = nb.to_jupyter();
    assert!(ipynb.contains("\"output_type\": \"stream\""));
    assert!(ipynb.contains("\"name\": \"stdout\""));
  }

  #[test]
  fn test_export_markdown_all_heading_levels() {
    let mut nb = Notebook::new();
    nb.push_cell(Cell::new(CellStyle::Title, "T"));
    nb.push_cell(Cell::new(CellStyle::Subtitle, "ST"));
    nb.push_cell(Cell::new(CellStyle::Chapter, "C"));
    nb.push_cell(Cell::new(CellStyle::Subchapter, "SC"));
    nb.push_cell(Cell::new(CellStyle::Section, "S"));
    nb.push_cell(Cell::new(CellStyle::Subsection, "SS"));
    nb.push_cell(Cell::new(CellStyle::Subsubsection, "SSS"));

    let md = nb.to_markdown();
    assert!(md.contains("# T"));
    assert!(md.contains("*ST*"));
    assert!(md.contains("## C"));
    assert!(md.contains("### SC"));
    assert!(md.contains("#### S"));
    assert!(md.contains("##### SS"));
    assert!(md.contains("###### SSS"));
  }

  #[test]
  fn test_parse_new_cell_types() {
    let nb = r#"Notebook[{
Cell["A chapter", "Chapter"],
Cell["A subchapter", "Subchapter"],
Cell["An item", "Item"],
Cell["A subitem", "Subitem"]
}]"#;

    let parsed = parse_notebook(nb).unwrap();
    assert_eq!(parsed.cells.len(), 4);

    let styles: Vec<CellStyle> = parsed
      .cells
      .iter()
      .filter_map(|e| match e {
        CellEntry::Single(c) => Some(c.style),
        CellEntry::Group(_) => None,
      })
      .collect();
    assert_eq!(
      styles,
      vec![
        CellStyle::Chapter,
        CellStyle::Subchapter,
        CellStyle::Item,
        CellStyle::Subitem,
      ]
    );
  }

  #[test]
  fn test_roundtrip_new_cell_types() {
    let mut nb = Notebook::new();
    nb.push_cell(Cell::new(CellStyle::Chapter, "Chapter 1"));
    nb.push_cell(Cell::new(CellStyle::Subchapter, "Subchapter 1.1"));
    nb.push_cell(Cell::new(CellStyle::Item, "First item"));
    nb.push_cell(Cell::new(CellStyle::Subitem, "Nested item"));

    let serialized = nb.to_string();
    assert!(serialized.contains("\"Chapter\""));
    assert!(serialized.contains("\"Subchapter\""));
    assert!(serialized.contains("\"Item\""));
    assert!(serialized.contains("\"Subitem\""));

    let reparsed = parse_notebook(&serialized).unwrap();
    assert_eq!(reparsed.cells.len(), 4);
  }

  #[test]
  fn test_textdata_inline_math_cell_renders_content() {
    // Inline `Cell[…, "InlineMath"]` elements inside a TextData run carry
    // real prose (math embedded in a sentence) and must be rendered, not
    // dropped like the Demonstrations "more info" chrome buttons.
    let nb = r#"Notebook[{
Cell[TextData[{
 "To find ",
 Cell[BoxData[
  FormBox[
   RowBox[{"P", "(",
    RowBox[{"X", "\[LessEqual]", "x"}], ")"}], TraditionalForm]],
  "InlineMath",ExpressionUUID->"c04c6311-9407-4855-8351-984bf610bb65"],
 " with mean ",
 Cell[BoxData[
  FormBox["\[Mu]", TraditionalForm]], "InlineMath",ExpressionUUID->
  "2769c287-5751-4749-947c-fcdd1da9d653"],
 "."
}], "Text"]
}]"#;
    let parsed = parse_notebook(nb).unwrap();
    match &parsed.cells[0] {
      CellEntry::Single(cell) => {
        assert_eq!(cell.style, CellStyle::Text);
        // Display text keeps the typeset relation sign (`≤`, not the
        // InputForm `<=`) — this is prose, not code.
        assert_eq!(cell.content, "To find P(X\u{2264}x) with mean \u{03bc}.");
      }
      CellEntry::Group(_) => panic!("Expected single cell"),
    }
  }

  #[test]
  fn test_textdata_chrome_button_cell_still_dropped() {
    // Unstyled inline cells (the Demonstrations "more info" opener
    // buttons) carry no textual content and stay dropped.
    let nb = r#"Notebook[{
Cell[TextData[{
 "Caption",
 Cell[BoxData[
  PaneSelectorBox[{True->
   TemplateBox[{"CaptionCells"},
    "MoreInfoOpenerButtonTemplate"]}, Dynamic[
    CurrentValue[
     EvaluationNotebook[], {TaggingRules, "ResourceCreateNotebook"}]],
   ImageSize->Automatic]],ExpressionUUID->
  "4c32c08b-d967-45c6-8920-0c21a5734cd7"]
}], "Section"]
}]"#;
    let parsed = parse_notebook(nb).unwrap();
    match &parsed.cells[0] {
      CellEntry::Single(cell) => {
        assert_eq!(cell.style, CellStyle::Section);
        assert_eq!(cell.content, "Caption");
      }
      CellEntry::Group(_) => panic!("Expected single cell"),
    }
  }

  #[test]
  fn test_export_markdown_items() {
    let mut nb = Notebook::new();
    nb.push_cell(Cell::new(CellStyle::Item, "First"));
    nb.push_cell(Cell::new(CellStyle::Subitem, "Nested"));

    let md = nb.to_markdown();
    assert!(md.contains("- First"));
    assert!(md.contains("  - Nested"));
  }

  #[test]
  fn test_escape_json() {
    assert_eq!(escape_json("hello"), "hello");
    assert_eq!(escape_json("he\"llo"), "he\\\"llo");
    assert_eq!(escape_json("a\\b"), "a\\\\b");
    assert_eq!(escape_json("a\nb"), "a\\nb");
    assert_eq!(escape_json("a\tb"), "a\\tb");
  }

  #[test]
  fn test_collapsed_chapter_serializes_cell_open_false() {
    let mut nb = Notebook::new();
    let mut chapter = Cell::new(CellStyle::Chapter, "Intro");
    chapter.collapsed = true;
    nb.push_cell(chapter);
    nb.push_cell(Cell::new(CellStyle::Subchapter, "Details"));

    let serialized = nb.to_string();
    assert!(
      serialized.contains("\"Chapter\", CellOpen -> False"),
      "expected CellOpen -> False on collapsed chapter, got: {serialized}"
    );
    // Non-collapsed subchapter must NOT have the option.
    assert!(!serialized.contains("\"Subchapter\", CellOpen -> False"));
  }

  #[test]
  fn test_collapsed_flag_roundtrips() {
    let mut nb = Notebook::new();
    let mut chapter = Cell::new(CellStyle::Chapter, "Chapter 1");
    chapter.collapsed = true;
    nb.push_cell(chapter);
    let mut subchapter = Cell::new(CellStyle::Subchapter, "Sub 1.1");
    subchapter.collapsed = true;
    nb.push_cell(subchapter);
    nb.push_cell(Cell::new(CellStyle::Chapter, "Chapter 2"));

    let serialized = nb.to_string();
    let reparsed = parse_notebook(&serialized).unwrap();
    assert_eq!(reparsed.cells.len(), 3);

    let collapsed_states: Vec<bool> = reparsed
      .cells
      .iter()
      .filter_map(|e| match e {
        CellEntry::Single(c) => Some(c.collapsed),
        CellEntry::Group(_) => None,
      })
      .collect();
    assert_eq!(collapsed_states, vec![true, true, false]);
  }

  #[test]
  fn test_non_heading_cells_do_not_emit_cell_open() {
    // Even if the flag is true on a non-heading cell, we don't emit
    // the option — collapse only applies to Chapter/Subchapter.
    let mut nb = Notebook::new();
    let mut text = Cell::new(CellStyle::Text, "hi");
    text.collapsed = true;
    nb.push_cell(text);

    let serialized = nb.to_string();
    assert!(!serialized.contains("CellOpen"));
  }

  #[test]
  fn test_parse_cell_with_cell_open_option() {
    let nb = r#"Notebook[{
Cell["Chapter 1", "Chapter", CellOpen -> False],
Cell["Chapter 2", "Chapter"]
}]"#;

    let parsed = parse_notebook(nb).unwrap();
    assert_eq!(parsed.cells.len(), 2);

    match &parsed.cells[0] {
      CellEntry::Single(cell) => {
        assert_eq!(cell.style, CellStyle::Chapter);
        assert_eq!(cell.content, "Chapter 1");
        assert!(cell.collapsed);
      }
      CellEntry::Group(_) => panic!("Expected single cell"),
    }
    match &parsed.cells[1] {
      CellEntry::Single(cell) => {
        assert!(!cell.collapsed);
      }
      CellEntry::Group(_) => panic!("Expected single cell"),
    }
  }

  #[test]
  fn test_json_source_lines() {
    assert_eq!(json_source_lines(""), "[\"\"]");
    assert_eq!(json_source_lines("hello"), "[\"hello\"]");
    assert_eq!(json_source_lines("a\nb"), "[\"a\\n\", \"b\"]");
  }

  #[test]
  fn test_parse_real_hello_world_nb() {
    let contents = std::fs::read_to_string(concat!(
      env!("CARGO_MANIFEST_DIR"),
      "/tests/notebooks/hello_world.nb"
    ))
    .unwrap();
    let nb = parse_notebook(&contents).unwrap();
    assert_eq!(nb.cells.len(), 1);
    match &nb.cells[0] {
      CellEntry::Group(group) => {
        assert!(group.open);
        assert_eq!(group.cells.len(), 2);
        assert_eq!(group.cells[0].style, CellStyle::Input);
        // Content should be the reconstructed expression
        assert!(
          group.cells[0].content.contains("StringJoin"),
          "Expected Input cell to contain 'StringJoin', got: {:?}",
          group.cells[0].content
        );
        assert_eq!(group.cells[1].style, CellStyle::Output);
        assert!(
          group.cells[1].content.contains("Hello World!"),
          "Expected Output cell to contain 'Hello World!', got: {:?}",
          group.cells[1].content
        );
      }
      CellEntry::Single(_) => panic!("Expected a cell group"),
    }
  }

  #[test]
  fn test_parse_real_syntax_nb() {
    let contents = std::fs::read_to_string(concat!(
      env!("CARGO_MANIFEST_DIR"),
      "/tests/notebooks/syntax.nb"
    ))
    .unwrap();
    let nb = parse_notebook(&contents).unwrap();
    // syntax.nb has 8 cell groups (each input/output pair)
    assert_eq!(nb.cells.len(), 8);
    for entry in &nb.cells {
      match entry {
        CellEntry::Group(group) => {
          assert!(group.open);
          assert_eq!(group.cells.len(), 2);
          assert_eq!(group.cells[0].style, CellStyle::Input);
          assert_eq!(group.cells[1].style, CellStyle::Output);
        }
        CellEntry::Single(_) => {
          panic!("Expected all entries to be cell groups")
        }
      }
    }
  }

  #[test]
  fn test_parse_real_layout_typography_nb() {
    let contents = std::fs::read_to_string(concat!(
      env!("CARGO_MANIFEST_DIR"),
      "/tests/notebooks/layout_typography.nb"
    ))
    .unwrap();
    let nb = parse_notebook(&contents).unwrap();
    // Should have one top-level group containing all the heading cells
    assert!(!nb.cells.is_empty(), "Expected at least one cell entry");
    let flat = nb.flat_cells();
    let styles: Vec<CellStyle> = flat.iter().map(|(_, c)| c.style).collect();
    assert!(styles.contains(&CellStyle::Title), "Expected a Title cell");
    assert!(
      styles.contains(&CellStyle::Subtitle),
      "Expected a Subtitle cell"
    );
    assert!(
      styles.contains(&CellStyle::Chapter),
      "Expected Chapter cells"
    );
    assert!(
      styles.contains(&CellStyle::Section),
      "Expected Section cells"
    );
    assert!(
      styles.contains(&CellStyle::Subsection),
      "Expected Subsection cells"
    );
    assert!(
      styles.contains(&CellStyle::Subsubsection),
      "Expected Subsubsection cells"
    );
    assert!(styles.contains(&CellStyle::Text), "Expected a Text cell");
  }

  #[test]
  fn test_parse_real_understanding_2d_translation_nb() {
    // A trimmed Wolfram Demonstrations template notebook (the shape of
    // downloaded Demonstration .nb files): section headers carrying inline
    // more-info opener cells, a Manipulate input with its stored
    // DynamicModuleBox widget dump, and snapshot raster outputs.
    let contents = std::fs::read_to_string(concat!(
      env!("CARGO_MANIFEST_DIR"),
      "/tests/notebooks/understanding_2d_translation.nb"
    ))
    .unwrap();
    let nb = parse_notebook(&contents).unwrap();
    let flat = nb.flat_cells();
    let styles: Vec<CellStyle> = flat.iter().map(|(_, c)| c.style).collect();
    assert_eq!(
      styles,
      vec![
        CellStyle::Title,
        CellStyle::Section, // "Caption" header (inline opener cell dropped)
        CellStyle::Text,
        CellStyle::Text, // CodeText falls back to Text
        CellStyle::Input,
        CellStyle::Output, // DynamicModuleBox widget dump
        CellStyle::Section,
        CellStyle::Output, // snapshot raster
      ]
    );

    // The section header keeps only its label; the attached more-info
    // opener cell contributes no text.
    assert_eq!(flat[1].1.content, "Caption");

    // Styled CodeText prose flattens its StyleBox runs into plain text.
    assert_eq!(
      flat[3].1.content,
      "If you provide initialization code, include a SaveDefinitions->True \
       option in the Manipulate."
    );

    // The Manipulate input reconstructs to evaluable InputForm: named
    // operator characters become ASCII and \[DoubleDownArrow] becomes ⇓.
    let input = &flat[4].1.content;
    assert!(input.starts_with("Manipulate["), "got: {input}");
    assert!(
      input.contains("PlotRange->{{-2,2}, {-2,2}}"),
      "got: {input}"
    );
    assert!(input.contains("Style[ \"\u{21D3}\", 25]"), "got: {input}");
    assert!(
      input.contains(
        "{{rsource, True, Tooltip[\"source\",\"Show source object\"]}, \
         {True, False}}"
      ),
      "got: {input}"
    );

    // The stored widget dump unwraps TagBox/StyleBox down to the
    // DynamicModuleBox (which the Studio replaces with a live widget).
    assert!(
      flat[5].1.content.starts_with("DynamicModuleBox["),
      "got: {}",
      &flat[5].1.content[..60.min(flat[5].1.content.len())]
    );
  }

  #[test]
  fn test_parse_real_demonstration_nb() {
    // Reduced Wolfram Demonstrations "definition notebook" (the
    // ColorRelationships template): deeply nested cell groups, Section
    // headers carrying inline MoreInfo opener cells, an Input +
    // DynamicModuleBox-dump Output pair, a RasterBox snapshot Output,
    // and Item keyword cells.
    let contents = std::fs::read_to_string(concat!(
      env!("CARGO_MANIFEST_DIR"),
      "/tests/notebooks/demonstration.nb"
    ))
    .unwrap();
    let nb = parse_notebook(&contents).unwrap();
    let flat = nb.flat_cells();
    let cells: Vec<&Cell> = flat.iter().map(|(_, c)| *c).collect();

    assert_eq!(cells[0].style, CellStyle::Title);
    assert_eq!(cells[0].content, "Color Relationships");

    // Section headers render just their label — the trailing inline
    // `Cell[BoxData[PaneSelectorBox[…]]]` opener button carries no text.
    assert_eq!(cells[1].style, CellStyle::Section);
    assert_eq!(cells[1].content, "Initialization Code");

    // The initialization Input cell reconstructs evaluable InputForm.
    assert_eq!(cells[2].style, CellStyle::Input);
    assert_eq!(
      cells[2].content,
      "swatch[clr_]:=Graphics[{clr,Rectangle[]}]"
    );

    assert_eq!(cells[3].style, CellStyle::Section);
    assert_eq!(cells[3].content, "Manipulate");

    // The Manipulate input keeps ASCII `->` (from `\[Rule]`) and its
    // stored output is recognizable as a FrontEnd widget dump (TagBox/
    // StyleBox wrappers unwrap to the DynamicModuleBox).
    assert_eq!(cells[4].style, CellStyle::Input);
    assert!(cells[4].content.starts_with("Manipulate["));
    assert!(cells[4].content.contains("SaveDefinitions->True"));
    assert_eq!(cells[5].style, CellStyle::Output);
    assert!(
      cells[5]
        .content
        .trim_start()
        .starts_with("DynamicModuleBox[")
    );

    // Snapshot outputs and keyword items parse with their styles. The
    // `PaneBox[GraphicsBox[TagBox[RasterBox[…]]]]` box dump stays raw —
    // `stored_output_image_svg` decodes it for display.
    assert!(
      cells.iter().any(
        |c| c.style == CellStyle::Output && c.content.contains("RasterBox")
      ),
      "expected a RasterBox snapshot output"
    );
    let items: Vec<&str> = cells
      .iter()
      .filter(|c| c.style == CellStyle::Item)
      .map(|c| c.content.as_str())
      .collect();
    assert_eq!(items, vec!["hue", "color wheel"]);
  }

  #[test]
  fn test_unescape_wolfram_string_delimiters() {
    // \< and \> are Wolfram string delimiters in box expressions
    assert_eq!(unescape_string(r#"\<"Hello"\>"#), r#""Hello""#);
    assert_eq!(unescape_string(r#"\<Hello World!\>"#), "Hello World!");
  }

  #[test]
  fn test_string_literal_source_preserves_already_escaped_quotes() {
    // A tooltip string containing an embedded quoted phrase (e.g. "Pareto
    // superior") round-trips through the front end as `\<...\\\"...\\\"...\>`:
    // the \\\" pairs already decode to a valid `\"` escape and must not be
    // escaped a second time into `\\"`.
    let inner = r#"\"\<a \\\"quoted\\\" phrase\>\""#;
    assert_eq!(string_literal_source(inner), r#""a \"quoted\" phrase""#);
  }

  #[test]
  fn test_unescape_line_continuation() {
    // `\` at end of line is a Wolfram line continuation: both the
    // backslash and the newline are dropped, joining the lines.
    assert_eq!(unescape_string("hello\\\nworld"), "helloworld");
    // Combined with \< and \>
    assert_eq!(unescape_string("\\<\\\nSome text.\\\n\\>"), "Some text.");
  }

  #[test]
  fn test_unescape_named_curly_quote() {
    // \[CloseCurlyQuote] should render as the typographic apostrophe.
    assert_eq!(
      unescape_string("doesn\\[CloseCurlyQuote]t"),
      "doesn\u{2019}t"
    );
  }

  #[test]
  fn test_unescape_named_indenting_newline() {
    // \[IndentingNewLine] should render as a real newline.
    assert_eq!(unescape_string("a\\[IndentingNewLine]b"), "a\nb");
  }

  #[test]
  fn test_extract_cell_content_boxdata_list() {
    // Multi-statement Input cells use BoxData[{ RowBox, "\n", RowBox, ... }].
    let s = r#"BoxData[{
 RowBox[{"a", "=", "1"}], "\n",
 RowBox[{"b", "=", "2"}]
}]"#;
    assert_eq!(extract_cell_content(s), "a=1\nb=2");
  }

  #[test]
  fn test_extract_cell_content_rule_operator() {
    // \[Rule] inside BoxData/RowBox should render as ASCII `->` so the
    // cell stays editable. The Wolfram private-use codepoint (U+F522)
    // has no glyph in normal fonts and would appear blank.
    let s = r#"BoxData[RowBox[{"ImageSize", "\[Rule]", "350"}]]"#;
    assert_eq!(extract_cell_content(s), "ImageSize->350");
  }

  #[test]
  fn test_extract_cell_content_equal_operator() {
    // `\[Equal]` is the typeset name for the `==` comparison operator. The
    // default Wolfram→Unicode mapping is U+003D (`=`), which is `Set`
    // (assignment) at evaluation time — definitely not what the box
    // expression means inside e.g. `SolveValues[lhs \[Equal] rhs, m]`.
    let s = r#"BoxData[RowBox[{"a", "\[Equal]", "b"}]]"#;
    assert_eq!(extract_cell_content(s), "a==b");
  }

  #[test]
  fn test_extract_cell_content_inequality_operators() {
    // Same concern for the related comparison operators.
    let s = r#"BoxData[RowBox[{"a", "\[NotEqual]", "b"}]]"#;
    assert_eq!(extract_cell_content(s), "a!=b");
    let s = r#"BoxData[RowBox[{"a", "\[LessEqual]", "b"}]]"#;
    assert_eq!(extract_cell_content(s), "a<=b");
    let s = r#"BoxData[RowBox[{"a", "\[GreaterEqual]", "b"}]]"#;
    assert_eq!(extract_cell_content(s), "a>=b");
  }

  /// The FrontEnd also typesets `Part` as a bracketed subscript, which is
  /// how a Demonstrations cell stores `c[[1]]`. Regression: it came back as
  /// `Subscript[c, ⟦1⟧]`, which does not parse.
  #[test]
  fn test_subscript_box_with_double_brackets_is_part() {
    let s = r#"BoxData[SubscriptBox["c", RowBox[{"\[LeftDoubleBracket]", "1", "\[RightDoubleBracket]"}]]]"#;
    assert_eq!(extract_cell_content(s), "c[[1]]");
    // Several indices, and a non-token base that needs the function form.
    let s = r#"BoxData[SubscriptBox["c", RowBox[{"\[LeftDoubleBracket]", RowBox[{"1", ",", "2"}], "\[RightDoubleBracket]"}]]]"#;
    assert_eq!(extract_cell_content(s), "c[[1,2]]");
    let s = r#"BoxData[SubscriptBox[RowBox[{"a", "+", "b"}], RowBox[{"\[LeftDoubleBracket]", "1", "\[RightDoubleBracket]"}]]]"#;
    assert_eq!(extract_cell_content(s), "Part[a+b, 1]");
    // An ordinary subscript is still `Subscript`.
    let s = r#"BoxData[SubscriptBox["c", "1"]]"#;
    assert_eq!(extract_cell_content(s), "Subscript[c, 1]");
  }

  /// A subscript can also be a bare display glyph rather than a real index
  /// — a Demonstrations control label typeset "0" with a subscript "+" (the
  /// one-sided-limit notation "0⁺") as `SubscriptBox["0", "+"]`. Regression:
  /// it came back as `Subscript[0, +]`, which does not parse (`+` alone is
  /// not a complete expression), so the reconstructed source fell back to
  /// showing the raw box syntax instead of typesetting anything.
  #[test]
  fn test_subscript_with_bare_operator_glyph_is_quoted() {
    let s = r#"BoxData[SubscriptBox["0", "+"]]"#;
    assert_eq!(extract_cell_content(s), "Subscript[0, \"+\"]");
    // A real indexed variable keeps its identifier/number subscript bare.
    let s = r#"BoxData[SubscriptBox["p", "0"]]"#;
    assert_eq!(extract_cell_content(s), "Subscript[p, 0]");
    // A compound index expression still parses as one, so it stays bare.
    let s = r#"BoxData[SubscriptBox["x", RowBox[{"i", "+", "1"}]]]"#;
    assert_eq!(extract_cell_content(s), "Subscript[x, i+1]");
  }

  /// `OverscriptBox`/`UnderscriptBox` become the evaluable `Overscript`/
  /// `Underscript` forms — Wolfram's own typesetting heads, which (like
  /// `Subscript`) stay symbolic rather than evaluating away. Regression: an
  /// antiquark label (`OverscriptBox["u", "_"]`, from a physics
  /// Demonstration's quark-content picker) came back with its mark left
  /// bare, which parses as the `Blank[]` pattern rather than the macron
  /// mark it typesets.
  #[test]
  fn test_overscript_box_becomes_evaluable_overscript() {
    let s = r#"BoxData[OverscriptBox["u", "_"]]"#;
    assert_eq!(extract_cell_content(s), "Overscript[u, \"_\"]");
    let s = r#"BoxData[OverscriptBox["x", "^"]]"#;
    assert_eq!(extract_cell_content(s), "Overscript[x, \"^\"]");
    let s = r#"BoxData[UnderscriptBox["x", "k"]]"#;
    assert_eq!(extract_cell_content(s), "Underscript[x, \"k\"]");
  }

  /// `SubscriptBox["\\[PartialD]", vars]` is the typeset partial-derivative
  /// operator, which takes the expression *after* it as its operand. The
  /// FrontEnd reads it as `D[body, vars]`, as
  /// `ToExpression[…, StandardForm, Hold]` confirms; a bare
  /// `Subscript[\u{2202}, x]` does not parse in either engine.
  #[test]
  fn test_partial_derivative_box_is_a_derivative() {
    let s = r#"BoxData[RowBox[{SubscriptBox["\[PartialD]", "x"], RowBox[{"f", "[", "x", "]"}]}]]"#;
    assert_eq!(extract_cell_content(s), "(D[f[x], x])");
    // Several variables: a repeated one is a higher-order derivative.
    let s = r#"BoxData[RowBox[{SubscriptBox["\[PartialD]", RowBox[{"x", ",", "x"}]], RowBox[{"c", "[", RowBox[{"x", ",", "t"}], "]"}]}]]"#;
    assert_eq!(extract_cell_content(s), "(D[c[x,t], x,x])");
    // The operator is usually juxtaposed with a coefficient, so the call is
    // parenthesised — `u D[…]` must stay a product, not become a symbol
    // named `uD`.
    let s = r#"BoxData[RowBox[{"u", RowBox[{SubscriptBox["\[PartialD]", "x"], RowBox[{"c", "[", RowBox[{"x", ",", "t"}], "]"}]}]}]]"#;
    assert_eq!(extract_cell_content(s), "u(D[c[x,t], x])");
    // An ordinary subscript is untouched.
    let s = r#"BoxData[RowBox[{SubscriptBox["a", "x"], "b"}]]"#;
    assert_eq!(extract_cell_content(s), "Subscript[a, x]b");
  }

  /// A named character inside a *string literal* is content, so it stays
  /// Unicode; only a bare operator token between operands collapses to its
  /// ASCII form. Regression: a Demonstrations label
  /// `"\!\(\*SubscriptBox[\(H\), \(0\)]\): p \[GreaterEqual] …"` came back
  /// with `>=` in the middle of the sentence.
  #[test]
  fn test_named_character_inside_string_literal_stays_unicode() {
    let s =
      r#"BoxData[RowBox[{"f", "[", "\"\<p \[GreaterEqual] q\>\"", "]"}]]"#;
    assert_eq!(extract_cell_content(s), "f[\"p ≥ q\"]");
    // The same character as an operator token still becomes `>=`.
    let s = r#"BoxData[RowBox[{"a", "\[GreaterEqual]", "b"}]]"#;
    assert_eq!(extract_cell_content(s), "a>=b");
  }

  /// `\[CenterDot]` is the dot product a Demonstration's prose writes its
  /// matrix formulas with (`T' = a · T · a^T`). It was the one named
  /// character the operator lexer knew but the character table did not, so a
  /// Text cell printed the escape `\[CenterDot]` verbatim mid-sentence.
  #[test]
  fn test_center_dot_named_character_renders_as_middle_dot() {
    let s = r#"TextData["T'=a \[CenterDot] T \[CenterDot] a"]"#;
    assert_eq!(extract_cell_content(s), "T'=a \u{00B7} T \u{00B7} a");
  }

  #[test]
  fn test_extract_cell_content_template_box_quantity() {
    // `TemplateBox[{value, displayed, name, unit_id}, "QuantityPrefix"]` is
    // how the FrontEnd typesets currency literals like `$5000`. We unpack
    // these back to `Quantity[5000, "USDollars"]` so the cell stays evaluable.
    let s = r#"BoxData[TemplateBox[{"5000", RowBox[{FormBox["\"$\"", TraditionalForm], "\[VeryThinSpace]"}], "US dollars", "\"USDollars\""}, "QuantityPrefix"]]"#;
    assert_eq!(extract_cell_content(s), "Quantity[5000, \"USDollars\"]");
  }

  #[test]
  fn test_extract_cell_content_rule_delayed_operator() {
    let s = r#"BoxData[RowBox[{"x", "\[RuleDelayed]", "y"}]]"#;
    assert_eq!(extract_cell_content(s), "x:>y");
  }

  #[test]
  fn test_extract_cell_content_rule_with_nested_part() {
    // Regression: from polfosol primeturn.nb — `ImageSize -> Part[…] 35`.
    // Without the ASCII mapping the `\[Rule]` displayed as a missing
    // glyph and the user saw `ImageSizePart` instead of `ImageSize->Part`.
    let s = r#"BoxData[RowBox[{"ImageSize", "\[Rule]",
       RowBox[{
        RowBox[{"Part", "[",
         RowBox[{"r", ",", "1", ",", "1"}], "]"}],
        "35"}]}]]"#;
    assert_eq!(extract_cell_content(s), "ImageSize->Part[r,1,1]35");
  }

  #[test]
  fn test_extract_image_raster_literal() {
    // An inline `Image[…]` literal is typeset as GraphicsBox[TagBox[
    // RasterBox[…], BoxForm`ImageTag[…]]]. The flipped bounding rectangle
    // ({{0, h}, {w, 0}}) means the raster rows are already in Image order.
    let s = r#"GraphicsBox[
       TagBox[RasterBox[{{{1, 2, 3}, {4, 5, 6}}}, {{0, 1}, {2, 0}}, {0, 255},
         ColorFunction->RGBColor],
        BoxForm`ImageTag["Byte", ColorSpace -> "RGB", Interleaving -> True],
        Selectable->False],
       BaseStyle->"ImageGraphics",
       ImageSize->Automatic,
       ImageSizeRaw->{2, 1},
       PlotRange->{{0, 2}, {0, 1}}]"#;
    assert_eq!(
      extract_cell_content(s),
      "Image[{{{1, 2, 3}, {4, 5, 6}}}, \"Byte\", ColorSpace -> \"RGB\", \
       Interleaving -> True]"
    );
  }

  #[test]
  fn test_extract_image_raster_compressed_data() {
    // CompressedData payloads are wrapped across lines by the FrontEnd; the
    // layout whitespace must be stripped so the base64 payload survives.
    let s = "GraphicsBox[\n       TagBox[RasterBox[CompressedData[\"\n1:eJxTTMoP\nSmNiYGAA\nAAtLAe0=\n         \"], {{0, 2}, {2, 0}}, {0, 255}],\n        BoxForm`ImageTag[\"Byte\", ColorSpace -> \"GrayLevel\"],\n        Selectable->False],\n       BaseStyle->\"ImageGraphics\"]";
    assert_eq!(
      extract_cell_content(s),
      "Image[CompressedData[\"1:eJxTTMoPSmNiYGAAAAtLAe0=\"], \"Byte\", \
       ColorSpace -> \"GrayLevel\"]"
    );
  }

  #[test]
  fn test_extract_image_raster_bottom_up_rows() {
    // A normal-orientation bounding rectangle ({{0, 0}, {w, h}}) stores the
    // raster rows bottom-up, so they must be reversed into Image order.
    let s = r#"GraphicsBox[
       TagBox[RasterBox[{{0, 1}, {2, 3}}, {{0, 0}, {2, 2}}, {0, 255}],
        BoxForm`ImageTag["Byte", ColorSpace -> "GrayLevel"],
        Selectable->False],
       BaseStyle->"ImageGraphics"]"#;
    assert_eq!(
      extract_cell_content(s),
      "Image[Reverse[{{0, 1}, {2, 3}}], \"Byte\", ColorSpace -> \"GrayLevel\"]"
    );
  }

  #[test]
  fn test_extract_image_raster_inside_assignment() {
    // The Demonstrations init cell embeds the image literal inside an
    // assignment: `image = ColorConvert[Image[…], "GrayScale"];`.
    let s = r#"BoxData[
 RowBox[{" ",
  RowBox[{
   RowBox[{"image", "=",
    RowBox[{"ColorConvert", "[",
     RowBox[{
      GraphicsBox[
       TagBox[RasterBox[{{{1, 2, 3}}}, {{0, 1}, {1, 0}}, {0, 255}],
        BoxForm`ImageTag["Byte", ColorSpace -> "RGB", Interleaving -> True],
        Selectable->False],
       BaseStyle->"ImageGraphics"], ",", "\"\<GrayScale\>\""}], "]"}]}],
   ";"}]}]]"#;
    assert_eq!(
      extract_cell_content(s),
      " image=ColorConvert[Image[{{{1, 2, 3}}}, \"Byte\", \
       ColorSpace -> \"RGB\", Interleaving -> True],\"GrayScale\"];"
    );
  }

  #[test]
  fn test_extract_plain_graphicsbox_not_treated_as_image() {
    // A GraphicsBox without the RasterBox/ImageTag pair is not an image
    // literal and must not be rewritten.
    let s = "GraphicsBox[DiskBox[{0, 0}]]";
    assert_eq!(extract_cell_content(s), "GraphicsBox[DiskBox[{0, 0}]]");
  }

  #[test]
  fn test_extract_textdata_section_label() {
    // Section/heading cells in Wolfram Demonstrations wrap the label in
    // `TextData[{"Label", Cell[BoxData[...]]}]`, where the trailing inline
    // `Cell` is an attached "more info" opener button that carries no text.
    // We render just the label.
    let s = r#"TextData[{
 "Manipulate",
 Cell[BoxData[
  PaneSelectorBox[{True->
   TemplateBox[{"ManipulateGroup"},
    "MoreInfoOpenerButtonTemplate"]}, Dynamic[x]]]]
}]"#;
    assert_eq!(extract_cell_content(s), "Manipulate");
  }

  #[test]
  fn test_extract_textdata_inline_math() {
    // Inline math cells (`Cell[BoxData[FormBox[…, TraditionalForm]],
    // "InlineMath"]`) contribute their typeset content to the prose —
    // dropping them loses words ("The triangle ABC" became "The triangle").
    let s = r#"TextData[{
 "The triangle ",
 Cell[BoxData[
  FormBox["ABC", TraditionalForm]], "InlineMath",ExpressionUUID->
  "4b134b90-4d52-4df9-bcc7-264d63b666b9"],
 " is limited to the range ",
 Cell[BoxData[
  FormBox[
   RowBox[{"[",
    RowBox[{
     RowBox[{"-", "9"}], ",", "9"}], "]"}], TraditionalForm]], "InlineMath",
  ExpressionUUID->"4a1e2e70-0973-4b3c-994b-25f0dad1ea7d"],
 ". Drag the point ",
 Cell[BoxData[
  FormBox[
   StyleBox["A",
    FontSlant->"Plain"], TraditionalForm]], "InlineMath",ExpressionUUID->
  "d0cf31da-6a61-4979-98a7-c37abad7eb18"],
 "."
}]"#;
    assert_eq!(
      extract_cell_content(s),
      "The triangle ABC is limited to the range [-9,9]. Drag the point A."
    );
  }

  #[test]
  fn test_stored_output_raster_snapshot_decodes_to_svg() {
    // Build the exact serialization Mathematica writes for a stored
    // Rasterize result: `RawArray["UnsignedInteger8", pixels]` with a
    // rank-3 {height, width, 3} byte array, zlib-compressed and
    // base64-encoded behind a "1:" prefix.
    let (h, w) = (2u32, 3u32);
    let mut raw: Vec<u8> = Vec::new();
    raw.extend_from_slice(b"!boR");
    raw.push(b'f');
    raw.extend_from_slice(&2u32.to_le_bytes());
    raw.push(b's');
    raw.extend_from_slice(&8u32.to_le_bytes());
    raw.extend_from_slice(b"RawArray");
    raw.push(b'S');
    raw.extend_from_slice(&16u32.to_le_bytes());
    raw.extend_from_slice(b"UnsignedInteger8");
    raw.push(b'b');
    raw.extend_from_slice(&3u32.to_le_bytes());
    raw.extend_from_slice(&h.to_le_bytes());
    raw.extend_from_slice(&w.to_le_bytes());
    raw.extend_from_slice(&3u32.to_le_bytes());
    raw.extend_from_slice(&[128u8; 2 * 3 * 3]);

    use base64::Engine;
    use std::io::Write;
    let mut enc = flate2::write::ZlibEncoder::new(
      Vec::new(),
      flate2::Compression::default(),
    );
    enc.write_all(&raw).unwrap();
    let b64 =
      base64::engine::general_purpose::STANDARD.encode(enc.finish().unwrap());
    let content = format!(
      "PaneBox[\n  GraphicsBox[\n   TagBox[RasterBox[CompressedData[\"\n\
       1:{b64}\n    \"], {{{{0, 0}}, {{1, 1}}}}]]]]"
    );

    let svg = stored_output_image_svg(&content).expect("decodes");
    assert!(svg.contains("viewBox=\"0 0 3 2\""), "{svg}");
    assert!(svg.contains("data:image/png;base64,"), "{svg}");
    // Non-raster outputs decode to nothing.
    assert!(stored_output_image_svg("{1, 2, 3}").is_none());
  }

  #[test]
  fn test_stored_output_checkbox_grid_renders_as_text() {
    let content = r#"{{{{CheckboxBox[False, {False, "Mathematics"}]}, {CheckboxBox[True, {False, "Life Sciences"}]}}, {{CheckboxBox[False, {False, True}]}}}}"#;
    assert_eq!(
      stored_output_checkbox_text(content).unwrap(),
      "[ ] Mathematics\n[x] Life Sciences\n[ ]"
    );
    assert!(stored_output_checkbox_text("{1, 2}").is_none());
  }

  #[test]
  fn test_stored_output_checkbox_text_reads_extracted_glyph_grid() {
    // `parse_notebook` already turns the checkbox boxes into glyphs, so the
    // Studio only ever sees the extracted grid — it must render from that
    // form too, rather than falling back to showing the raw braces.
    let content = "{{{{\u{2610} Mathematics}, {\u{2611} Life Sciences}}, \
       {{\u{2610} Systems, Models & Methods}, {\"\"}}}}";
    assert_eq!(
      stored_output_checkbox_text(content).unwrap(),
      "[ ] Mathematics\n[x] Life Sciences\n[ ] Systems, Models & Methods"
    );
    // A lone unlabelled checkbox (the compatibility pickers) still renders.
    assert_eq!(
      stored_output_checkbox_text("{{{{\u{2611}}}}}").unwrap(),
      "[x]"
    );
    assert!(stored_output_checkbox_text("{{1, 2}, {3, 4}}").is_none());
  }

  #[test]
  fn test_extract_cell_content_row_default_checkbox_caption() {
    // Demonstrations compatibility pickers put the caption next to the
    // checkbox in a `RowDefault` row, and leave the checkbox's own
    // alternatives degenerate — the caption must not be dropped.
    assert_eq!(
      extract_cell_content(
        r#"BoxData[TemplateBox[{CheckboxBox[True, {False, False}], "\" \"", StyleBox["\"Supported in cloud\"", FontSize -> 12, StripOnInput -> False]}, "RowDefault"]]"#
      ),
      "\u{2611} Supported in cloud"
    );
    // When the row *and* the checkbox carry the label, it appears once.
    assert_eq!(
      extract_cell_content(
        r#"BoxData[TemplateBox[{CheckboxBox[False, {False, "Triangles"}], "\" \"", StyleBox["\"Triangles\"", FontSize -> 12]}, "RowDefault"]]"#
      ),
      "\u{2610} Triangles"
    );
    // A row whose caption is wrapped in collapsible chrome has no flat
    // rendering; fall back to the leading element (the labelled checkbox).
    assert_eq!(
      extract_cell_content(
        r#"BoxData[TemplateBox[{CheckboxBox[False, {False, "Mathematics"}], "\" \"", StyleBox[DynamicModuleBox[{Typeset`var$$ = False}, PaneSelectorBox[{False -> "\"Mathematics\""}, Dynamic[Typeset`var$$]]]]}, "RowDefault"]]"#
      ),
      "\u{2610} Mathematics"
    );
  }

  #[test]
  fn test_extract_textdata_mixed_styled_prose() {
    // A `TextData` run mixing plain strings and `StyleBox` spans concatenates
    // to the plain prose.
    let s = r#"TextData[{
 "If you provide initialization code, include a ",
 StyleBox["SaveDefinitions->True", "MRbig"],
 " option in the ",
 StyleBox["Manipulate", "MRbig"],
 "."
}]"#;
    assert_eq!(
      extract_cell_content(s),
      "If you provide initialization code, include a SaveDefinitions->True \
       option in the Manipulate."
    );
  }

  #[test]
  fn test_extract_textdata_inline_math_cell() {
    // Inline `Cell[BoxData[…], "InlineMath"]` elements hold real formulas
    // (unlike the "more info" opener buttons) and must render as text.
    let s = r#"TextData[{
 "Given a point F (the focus) and a line ",
 Cell[BoxData[
  FormBox["d", TraditionalForm]], "InlineMath",ExpressionUUID->
  "23d75367-0ed4-44ff-ba06-4d9bdc71d1e9"],
 " (the directrix)."
}]"#;
    assert_eq!(
      extract_cell_content(s),
      "Given a point F (the focus) and a line d (the directrix)."
    );
  }

  #[test]
  fn test_extract_cell_content_prime_derivative() {
    // `SuperscriptBox[f, "\[Prime]"]` is the FrontEnd's typeset form of a
    // derivative; the trailing `MultilineFunction -> None` display option
    // must be ignored. From the trebuchet Demonstration notebook.
    let s = r#"BoxData[RowBox[{
      SuperscriptBox["\[Theta]", "\[Prime]\[Prime]",
       MultilineFunction->None], "[", "t", "]"}]]"#;
    assert_eq!(extract_cell_content(s), "θ''[t]");
    let s = r#"BoxData[RowBox[{
      SuperscriptBox["\[Phi]", "\[Prime]",
       MultilineFunction->None], "[", "0", "]"}]]"#;
    assert_eq!(extract_cell_content(s), "ϕ'[0]");
  }

  /// `\[RawEscape]` names the ASCII escape character. It sets no type, so a
  /// caption that opens an inline formula with one reads as the formula
  /// alone. Regression: the name was printed verbatim into the cell.
  #[test]
  fn test_extract_cell_content_raw_escape_sets_no_type() {
    let s = r#"TextData[{
 "\[RawEscape]",
 Cell[BoxData[FormBox[RowBox[{"E", "(", "R", ")"}], TraditionalForm]],
  "InlineMath"],
 "."
}]"#;
    assert_eq!(extract_cell_content(s), "E(R).");
  }

  /// A script hung on `\[InvisiblePrefixScriptBase]` is a *prefix* script:
  /// the FrontEnd writes the term symbol `¹Σ` as a superscript on that
  /// invisible placeholder. Exponentiation would be wrong twice over — the
  /// empty base leaves `()^(1)`, which does not parse, and `x^1` evaluates
  /// away the script — so the box becomes a `Superscript`, which does not
  /// evaluate. From the "Bohr's Model for the Hydrogen Molecule"
  /// Demonstration's state picker.
  #[test]
  fn test_extract_cell_content_invisible_prefix_script_base() {
    assert_eq!(
      extract_cell_content(
        r#"BoxData[SuperscriptBox["\[InvisiblePrefixScriptBase]", "1"]]"#
      ),
      "Superscript[\"\", 1]"
    );
    assert_eq!(
      extract_cell_content(
        r#"BoxData[SubscriptBox["\[InvisiblePrefixScriptBase]", "u"]]"#
      ),
      "Subscript[\"\", u]"
    );
    // The placeholder carries no glyph of its own wherever it turns up.
    assert_eq!(
      extract_cell_content(
        r#"BoxData[RowBox[{"\[InvisiblePrefixScriptBase]", "x"}]]"#
      ),
      "x"
    );
  }

  #[test]
  fn test_extract_cell_content_superscript_with_option() {
    // A plain power whose box carries a display option must still convert.
    let s = r#"BoxData[SuperscriptBox["x", "2", MultilineFunction->None]]"#;
    assert_eq!(extract_cell_content(s), "(x)^(2)");
  }

  #[test]
  fn test_extract_cell_content_squared_derivative() {
    // Nested: (φ'[t])^2 — a derivative RowBox inside a power box.
    let s = r#"BoxData[SuperscriptBox[
      RowBox[{
       SuperscriptBox["\[Phi]", "\[Prime]",
        MultilineFunction->None], "[", "t", "]"}], "2"]]"#;
    assert_eq!(extract_cell_content(s), "(ϕ'[t])^(2)");
  }

  #[test]
  fn test_extract_cell_content_slot_before_function_call() {
    // `#` and a following function-call box are *separate* sibling
    // elements of the row (implicit multiplication — e.g. a Manipulate
    // body scaling a value by `# Cos[x]` where the FrontEnd's box form
    // omits a literal space token between them). Gluing their text
    // together verbatim would read back as the named-slot syntax `#Cos`
    // (`Slot["Cos"]`) instead of `Slot[1] * Cos[x]`, which then throws at
    // evaluation time because the argument isn't an Association.
    let s = r##"BoxData[RowBox[{"#", RowBox[{"Cos", "[", "x", "]"}]}]]"##;
    let content = extract_cell_content(s);
    assert_eq!(content, "# Cos[x]");
    assert_eq!(
      crate::interpret(&format!("({content})&[3]")).unwrap(),
      "3*Cos[x]"
    );
  }

  #[test]
  fn test_extract_cell_content_checkbox() {
    // Demonstrations metadata cells hold checkbox grids; render glyphs.
    assert_eq!(
      extract_cell_content(
        r#"BoxData[CheckboxBox[False, {False, "Mathematics"}]]"#
      ),
      "\u{2610} Mathematics"
    );
    assert_eq!(
      extract_cell_content(
        r#"BoxData[CheckboxBox["Physics", {False, "Physics"}]]"#
      ),
      "\u{2611} Physics"
    );
    assert_eq!(
      extract_cell_content(r#"BoxData[CheckboxBox[True, {False, True}]]"#),
      "\u{2611}"
    );
    // Degenerate alternatives: fall back to the value's truthiness.
    assert_eq!(
      extract_cell_content(r#"BoxData[CheckboxBox[False, {False, False}]]"#),
      "\u{2610}"
    );
  }

  #[test]
  fn test_extract_textdata_leading_inline_math() {
    // An inline math cell at the *start* of a prose run (the sentence's
    // subject, from the trebuchet notebook) must render too.
    let s = r#"TextData[{
 Cell[BoxData[
  FormBox[
   RowBox[{
    RowBox[{"\[Theta]", "(", "t", ")"}], ",",
    RowBox[{"\[Phi]", "(", "t", ")"}]}], TraditionalForm]], "InlineMath",
  ExpressionUUID->"430c8958-3ee9-4d09-9a56-2cb3400dbb47"],
 " are the angles"
}]"#;
    assert_eq!(extract_cell_content(s), "θ(t),ϕ(t) are the angles");
  }

  #[test]
  fn test_extract_textdata_inline_math_subscripts() {
    // `SubscriptBox["m", "1"]` renders as display text in prose.
    let s = r#"TextData[{
 Cell[BoxData[
  FormBox[
   RowBox[{
    SubscriptBox["m", "1"], ",",
    SubscriptBox["m", "2"]}], TraditionalForm]], "InlineMath",ExpressionUUID->
  "dfd0ccb6-6747-48f8-9a58-eb78bd009bbc"],
 " are the weights"
}]"#;
    assert_eq!(extract_cell_content(s), "m_1,m_2 are the weights");
  }

  #[test]
  fn test_extract_textdata_inline_math_formula() {
    // A boxed formula inside prose converts to display text (Unicode
    // superscripts, not InputForm parentheses) — this is a sentence.
    let s = r#"TextData[{
 "One form of the equation of a parabola is ",
 Cell[BoxData[
  FormBox[
   RowBox[{
    SuperscriptBox["y", "2"], "=",
    RowBox[{"2", "p", " ", "x"}]}], TraditionalForm]], "InlineMath",
  ExpressionUUID->"9efcaa42-0dc4-43ec-a2b4-2e1f21a58f50"],
 "."
}]"#;
    assert_eq!(
      extract_cell_content(s),
      "One form of the equation of a parabola is y\u{00b2}=2p x."
    );
  }

  #[test]
  fn test_extract_textdata_display_formula() {
    // A whole Text cell holding one display formula (the trebuchet
    // notebook's Lagrangian) must not come back empty; derivative prime
    // marks and their `MultilineFunction -> None` display option render
    // as `θ'`.
    let s = r#"TextData[Cell[BoxData[
 FormBox[
  RowBox[{"\[ScriptCapitalL]", "=",
   RowBox[{
    StyleBox[
     FractionBox["1", "2"],
     FontFamily->"Times New Roman",
     FontSize->12], " ",
    SuperscriptBox[
     RowBox[{
      SuperscriptBox["\[Theta]", "\[Prime]",
       MultilineFunction->None], "(", "t", ")"}], "2"]}]}],
  TraditionalForm]], "InlineMath",ExpressionUUID->"x"]]"#;
    assert_eq!(extract_cell_content(s), "\u{2112}=1/2 θ'(t)\u{00B2}");
  }

  #[test]
  fn test_extract_textdata_button_hyperlink() {
    // `TextData[ButtonBox["label", ...]]` is a hyperlink; render its label.
    let s = r#"TextData[ButtonBox["Ellipse",
 BaseStyle->"Hyperlink",
 ButtonData->{URL["http://mathworld.wolfram.com/Ellipse.html"], None}]]"#;
    assert_eq!(extract_cell_content(s), "Ellipse");
  }

  #[test]
  fn test_parse_cell_unrecognized_style_keeps_content_clean() {
    // `CodeText` is a style we don't model. The style-detection fallback must
    // still identify it as the style (not fold the trailing options into the
    // content) so the extracted content is just the cell body.
    let s = r#"TextData[{"hello ", StyleBox["world", "MRbig"]}], "CodeText",
 Editable->False,
 CellID->687519280,ExpressionUUID->"39c481cb-843d-42b3-8aef-160cab90e699""#;
    let cell = parse_single_cell(s);
    // Unmodelled styles fall back to Text for rendering.
    assert_eq!(cell.style, CellStyle::Text);
    assert_eq!(cell.content, "hello world");
  }

  #[test]
  fn test_extract_section_with_curly_quote() {
    // Section cell from a real .nb file with \<…\> wrapping, line
    // continuations, and \[CloseCurlyQuote] for the apostrophe.
    let s = r#""\<\
This code doesn\[CloseCurlyQuote]t work anymore due to changes in twitter\
\[CloseCurlyQuote]s API\
\>""#;
    assert_eq!(
      extract_cell_content(s),
      "This code doesn\u{2019}t work anymore due to changes in twitter\u{2019}s API"
    );
  }

  #[test]
  fn test_parse_cell_group_with_trailing_options() {
    // Real .nb files emit `Cell[CellGroupData[{...}, Open], ExpressionUUID -> "..."]`.
    // The outer Cell has options after the CellGroupData that must be ignored.
    let nb = r#"Notebook[{
Cell[CellGroupData[{
Cell[BoxData["1 + 1"], "Input", ExpressionUUID -> "aaa"],
Cell[BoxData["2"], "Output", ExpressionUUID -> "bbb"]
}, Open], ExpressionUUID -> "ccc"]
}]"#;

    let parsed = parse_notebook(nb).unwrap();
    assert_eq!(parsed.cells.len(), 1);
    match &parsed.cells[0] {
      CellEntry::Group(group) => {
        assert!(group.open);
        assert_eq!(group.cells.len(), 2);
        assert_eq!(group.cells[0].style, CellStyle::Input);
        assert_eq!(group.cells[1].style, CellStyle::Output);
      }
      CellEntry::Single(_) => panic!("Expected cell group"),
    }
  }

  /// `∫ (1/y) ⅆy == ∫ x ⅆx + c`, the way the "Separable Differential
  /// Equations" chapter of *Introduction to Calculus* writes it. The
  /// integral sign takes the rest of its row as the body, and the `ⅆy`
  /// closing that body names the integration variable.
  #[test]
  fn indefinite_integral_boxes_become_integrate() {
    let nb = r#"Notebook[{
Cell[BoxData[
 RowBox[{
  RowBox[{"\[Integral]",
   RowBox[{
    RowBox[{"(", RowBox[{"1", "/", "y"}], ")"}],
    RowBox[{"\[DifferentialD]", "y"}]}]}], "\[Equal]",
  RowBox[{
   RowBox[{"\[Integral]",
    RowBox[{"x", RowBox[{"\[DifferentialD]", "x"}]}]}], "+", "c"}]}]], "Input"]
}]"#;
    let parsed = parse_notebook(nb).unwrap();
    let CellEntry::Single(cell) = &parsed.cells[0] else {
      panic!("expected a single cell");
    };
    assert_eq!(
      cell.content, "Integrate[(1/y), y]==Integrate[x, x]+c",
      "the ⅆ-closed integral body must name the integration variable"
    );
  }

  /// A definite integral writes its limits on the sign itself:
  /// `∫_0^2 (x^2 + 1) ⅆx` is `Integrate[x^2 + 1, {x, 0, 2}]`.
  #[test]
  fn definite_integral_boxes_carry_their_limits() {
    let nb = r#"Notebook[{
Cell[BoxData[
 RowBox[{
  SubsuperscriptBox["\[Integral]", "0", "2"],
  RowBox[{
   RowBox[{"(", RowBox[{SuperscriptBox["x", "2"], "+", "1"}], ")"}],
   RowBox[{"\[DifferentialD]", "x"}]}]}]], "Input"]
}]"#;
    let parsed = parse_notebook(nb).unwrap();
    let CellEntry::Single(cell) = &parsed.cells[0] else {
      panic!("expected a single cell");
    };
    assert_eq!(cell.content, "Integrate[((x)^(2)+1), {x, 0, 2}]");
  }

  /// Some notebooks typeset the `ⅆx` that closes an integral body with an
  /// explicit space box between the glyph and the variable
  /// (`RowBox[{"\[DifferentialD]", " ", "x"}]`) rather than gluing them
  /// directly (`RowBox[{"\[DifferentialD]", "x"}]`, covered above). The
  /// integration variable must still be recovered either way.
  #[test]
  fn definite_integral_with_spaced_differential_becomes_integrate() {
    let nb = r#"Notebook[{
Cell[BoxData[
 RowBox[{
  SubsuperscriptBox["\[Integral]", RowBox[{"-", "1"}], "1"],
  RowBox[{"g", " ",
   RowBox[{"\[DifferentialD]", " ", "t"}]}]}]], "Input"]
}]"#;
    let parsed = parse_notebook(nb).unwrap();
    let CellEntry::Single(cell) = &parsed.cells[0] else {
      panic!("expected a single cell");
    };
    assert_eq!(cell.content, "Integrate[g , {t, -1, 1}]");
  }

  /// Without an integral sign the differential is content of its own:
  /// `ⅆarea == 2 π r ⅆr` is an equation between differentials, so the
  /// `ⅆ` has to survive into the cell's source.
  #[test]
  fn bare_differential_survives_the_conversion() {
    let nb = r#"Notebook[{
Cell[BoxData[
 RowBox[{
  RowBox[{"\[DifferentialD]", "area"}], "==",
  RowBox[{"2", "\[Pi]", " ", "r", " ",
   RowBox[{"\[DifferentialD]", "r"}]}]}]], "Input"]
}]"#;
    let parsed = parse_notebook(nb).unwrap();
    let CellEntry::Single(cell) = &parsed.cells[0] else {
      panic!("expected a single cell");
    };
    assert!(
      cell.content.matches('\u{F74C}').count() == 2,
      "both differentials must survive: {}",
      cell.content
    );
  }

  /// The `⎧` brace of a typeset `Piecewise` must come back as a
  /// `Piecewise[…]` call — `RevolutionPlot3D` of a nested list has no
  /// function to sample.
  #[test]
  fn piecewise_brace_becomes_a_piecewise_call() {
    let nb = r#"Notebook[{
Cell[BoxData[
 RowBox[{"RevolutionPlot3D", "[",
  RowBox[{
   TagBox[GridBox[{
      {"\[Piecewise]", GridBox[{
         {RowBox[{"f", "[", RowBox[{"x", "+", "6"}], "]"}],
          RowBox[{RowBox[{"-", "6"}], "<=", "x", "<=", RowBox[{"-", "2"}]}]},
         {"Null", TagBox["True", "PiecewiseDefault", AutoDelete->True]}
        }, Selectable->True]}
     }, Selectable->True], "Piecewise", DeleteWithContents->True], ",",
   RowBox[{"{", RowBox[{"x", ",", RowBox[{"-", "6"}], ",", "6"}], "}"}]}],
  "]"}]], "Input"]
}]"#;
    let parsed = parse_notebook(nb).unwrap();
    let CellEntry::Single(cell) = &parsed.cells[0] else {
      panic!("expected a single cell");
    };
    assert_eq!(
      cell.content,
      "RevolutionPlot3D[Piecewise[{{f[x+6], -6<=x<=-2}}],{x,-6,6}]"
    );
  }

  /// A `Plot` legend written as an inline cell carries quotes inside its
  /// string; they have to stay escaped or the option reads as stray tokens.
  #[test]
  fn quotes_inside_a_string_literal_stay_escaped() {
    let nb = r#"Notebook[{
Cell[BoxData[
 RowBox[{"Plot", "[",
  RowBox[{"x", ",",
   RowBox[{"{", RowBox[{"x", ",", "0", ",", "1"}], "}"}], ",",
   RowBox[{"PlotLegends", "\[Rule]",
    RowBox[{"{", "\"\<\!\(\*Cell[\"f[x]\",ExpressionUUID->\"abc\"]\)\>\"", "}"}]}]}],
  "]"}]], "Input"]
}]"#;
    let parsed = parse_notebook(nb).unwrap();
    let CellEntry::Single(cell) = &parsed.cells[0] else {
      panic!("expected a single cell");
    };
    assert!(
      cell
        .content
        .contains(r#"{"\!\(\*Cell[\"f[x]\",ExpressionUUID->\"abc\"]\)"}"#),
      "the legend must stay one string literal: {}",
      cell.content
    );
  }

  #[test]
  fn test_parse_layout_typography_no_backslashes() {
    let contents = std::fs::read_to_string(concat!(
      env!("CARGO_MANIFEST_DIR"),
      "/tests/notebooks/layout_typography.nb"
    ))
    .unwrap();
    let nb = parse_notebook(&contents).unwrap();
    let flat = nb.flat_cells();

    // Find the Subtitle cell
    let subtitle = flat
      .iter()
      .find(|(_, c)| c.style == CellStyle::Subtitle)
      .map(|(_, c)| c)
      .expect("Expected a Subtitle cell");
    assert_eq!(
      subtitle.content,
      "Showcasing all layout and typography features of Wolfram notebooks."
    );
    assert!(
      !subtitle.content.contains('\\'),
      "Subtitle should not contain backslashes, got: {:?}",
      subtitle.content
    );

    // Find the Text cell
    let text_cell = flat
      .iter()
      .find(|(_, c)| c.style == CellStyle::Text)
      .map(|(_, c)| c)
      .expect("Expected a Text cell");
    assert!(
      !text_cell.content.contains('\\'),
      "Text cell should not contain backslashes, got: {:?}",
      text_cell.content
    );
  }

  #[test]
  fn test_graphicsbox_raster_input_cell() {
    // An image pasted into an Input cell is stored as
    // `GraphicsBox[TagBox[RasterBox[data, rect, range, opts], …], opts]`
    // (e.g. `slika = <photo>;` in Demonstration notebooks). It must come
    // back as an evaluable `Image[…]` literal.
    let nb = r#"Notebook[{
Cell[BoxData[
 RowBox[{
  RowBox[{"slika", "=",
   GraphicsBox[
    TagBox[RasterBox[CompressedData["
1:eJxTTMoPSmNiYGAo5gASQYnljkVFiZXBAkBOaF5xZnpeaopnXklqemqRRRJIGQ
yf4GL4DwC5VA4w
      "], {{0, 2}, {2, 0}}, {0, 255},
      ColorFunction->RGBColor],
     BoxForm`ImageTag["Byte", ColorSpace -> "RGB", Interleaving -> True],
     Selectable->False],
    BaseStyle->"ImageGraphics",
    ImageSizeRaw->{2, 2},
    PlotRange->{{0, 2}, {0, 2}}]}], ";"}]], "Input"]
}]"#;

    let parsed = parse_notebook(nb).unwrap();
    let flat = parsed.flat_cells();
    assert_eq!(flat.len(), 1);
    let cell = flat[0].1;
    assert_eq!(cell.style, CellStyle::Input);
    assert!(
      cell.content.starts_with("slika=Image[CompressedData["),
      "got: {}",
      &cell.content[..cell.content.len().min(80)]
    );
    assert!(
      cell
        .content
        .contains("\"Byte\", ColorSpace -> \"RGB\", Interleaving -> True]"),
      "got: {}",
      cell.content
    );
    assert!(
      !cell.content.contains("GraphicsBox")
        && !cell.content.contains("TagBox")
        && !cell.content.contains("RasterBox"),
      "box heads must be converted, got: {}",
      cell.content
    );
    assert!(cell.content.trim_end().ends_with(';'));
  }

  #[test]
  fn test_extract_saved_initialization() {
    // The Output dump of `Manipulate[…, SaveDefinitions -> True]`:
    // Deinitialization (whose name contains "initialization") must be
    // skipped, `$CellContext`` prefixes dropped, and the FrontEnd's
    // line-continuation `\` markers removed.
    let dump = "DynamicModuleBox[{$CellContext`k$$ = 0}, \
      DynamicBox[…],\n\
      Deinitialization:>None,\n\
      DynamicModuleValues:>{},\n\
      Initialization:>({$CellContext`a = 1, $CellContext`b = \\\n\
      {2, 3}}; Typeset`initDone$$ = True),\n\
      SynchronousInitialization->True]";
    let init = extract_saved_initialization(dump).unwrap();
    assert_eq!(init, "{a = 1, b = {2, 3}}; Typeset`initDone$$ = True");
  }

  #[test]
  fn test_saved_initialization_context_symbols() {
    let dump = "DynamicModuleBox[{$CellContext`k$$ = 0}, \
      DynamicBox[…],\n\
      Initialization:>($CellContext`Midpoint[\
      Pattern[$CellContext`a, Blank[]], Pattern[$CellContext`b, Blank[]]] := \
      ($CellContext`a + $CellContext`b)/2; \
      Typeset`initDone$$ = True)]";
    let mut names = saved_initialization_context_symbols(dump);
    names.sort();
    // `a` and `b` are pattern variable names, `Midpoint` is the helper
    // being defined; each appears once even though `Midpoint` and `a`/`b`
    // each occur more than once in the dump.
    assert_eq!(names, vec!["Midpoint", "a", "b"]);
  }

  #[test]
  fn test_saved_initialization_context_symbols_none() {
    assert_eq!(
      saved_initialization_context_symbols(
        "DynamicModuleBox[{}, DynamicBox[…]]"
      ),
      Vec::<String>::new()
    );
  }

  #[test]
  fn test_extract_saved_initialization_absent() {
    assert_eq!(
      extract_saved_initialization("DynamicModuleBox[{}, DynamicBox[…]]"),
      None
    );
    assert_eq!(
      extract_saved_initialization(
        "DynamicModuleBox[{}, Deinitialization:>None]"
      ),
      None
    );
  }

  #[test]
  fn test_inline_math_cells_render_in_text() {
    // `Cell[BoxData[…], "InlineMath"]` runs inside TextData carry the
    // prose's symbols (the Wolfram Demonstrations template); they must
    // render as display text, not vanish.
    let nb = r#"Notebook[{
Cell[TextData[{
 "Here ",
 Cell[BoxData[
  FormBox[
   SubscriptBox["D", "U"], TraditionalForm]], "InlineMath",ExpressionUUID->
  "332957a9-d341-4570-bd7c-95b02f59c357"],
 " and ",
 Cell[BoxData[
  FormBox[
   SuperscriptBox["V", "2"], TraditionalForm]], "InlineMath",ExpressionUUID->
  "f34a7ed8-3e3c-46ff-b104-0705ae5e03bb"],
 " appear."
}], "Text"]
}]"#;
    let parsed = parse_notebook(nb).unwrap();
    match &parsed.cells[0] {
      CellEntry::Single(cell) => {
        assert_eq!(cell.style, CellStyle::Text);
        assert_eq!(cell.content, "Here D_U and V\u{00b2} appear.");
      }
      CellEntry::Group(_) => panic!("Expected single cell"),
    }
  }

  #[test]
  fn test_inline_math_grid_of_equations() {
    // A TextData whose sole element is an inline-math cell holding a
    // GridBox renders each row on its own line.
    let nb = r#"Notebook[{
Cell[TextData[Cell[BoxData[
 FormBox[
  RowBox[{"{", GridBox[{
     {
      RowBox[{"U", " ", "\[RightArrow]", " ", "P"}]},
     {
      RowBox[{"V", " ", "\[RightArrow]", " ", "Q"}]}
    }]}],
  TraditionalForm]], \
"InlineMath",ExpressionUUID->"97d25ac6-5009-432e-bb32-5ac8fe1a68f7"]], "Text"]
}]"#;
    let parsed = parse_notebook(nb).unwrap();
    match &parsed.cells[0] {
      CellEntry::Single(cell) => {
        assert_eq!(cell.content, "{U \u{2192} P\nV \u{2192} Q");
      }
      CellEntry::Group(_) => panic!("Expected single cell"),
    }
  }

  #[test]
  fn test_doubly_nested_inline_math_cell_renders_its_formula() {
    // The FrontEnd occasionally saves an inline-math cell whose `FormBox`
    // argument is itself a whole `Cell[TextData[Cell[BoxData[...]]]]` —
    // an artifact of pasting one inline formula's cell into another's box
    // slot. The nested cell must still resolve to its formula text
    // instead of leaking the raw box source into the surrounding prose.
    let nb = r#"Notebook[{
Cell[TextData[{
 "the potential is ",
 Cell[BoxData[
  FormBox[Cell[TextData[Cell[BoxData[
    FormBox[
     RowBox[{
      RowBox[{"V", "(", "x", ")"}], "=", "0"}], TraditionalForm]],
    "InlineMath",ExpressionUUID->"11111111-1111-1111-1111-111111111111"]],
    "InlineMath",ExpressionUUID->"22222222-2222-2222-2222-222222222222"],
   TraditionalForm]], "InlineMath",ExpressionUUID->
  "33333333-3333-3333-3333-333333333333"],
 " elsewhere."
}], "Text"]
}]"#;
    let parsed = parse_notebook(nb).unwrap();
    match &parsed.cells[0] {
      CellEntry::Single(cell) => {
        assert_eq!(cell.content, "the potential is V(x)=0 elsewhere.");
      }
      CellEntry::Group(_) => panic!("Expected single cell"),
    }
  }

  #[test]
  fn test_inline_cross_renders_as_multiplication_sign() {
    // `\[Cross]` canonically maps to a glyphless private-use codepoint;
    // prose shows the multiplication sign (`40 × 40`).
    let nb = r#"Notebook[{
Cell[TextData[{
 "a field size of ",
 Cell[BoxData[
  FormBox[
   RowBox[{"40", "\[Cross]", "40"}], TraditionalForm]], "InlineMath",
  ExpressionUUID->"6e1aba5b-f538-4ff2-b2b5-68f7aa88e1fd"]
}], "Text"]
}]"#;
    let parsed = parse_notebook(nb).unwrap();
    match &parsed.cells[0] {
      CellEntry::Single(cell) => {
        assert_eq!(cell.content, "a field size of 40\u{00d7}40");
      }
      CellEntry::Group(_) => panic!("Expected single cell"),
    }
  }

  /// The prose text a single inline-math cell renders to.
  fn inline_math_text(boxes: &str) -> String {
    let nb = format!(
      "Notebook[{{\nCell[TextData[Cell[BoxData[\n FormBox[{boxes},\n  \
       TraditionalForm]], \"InlineMath\"]], \"Text\"]\n}}]"
    );
    let parsed = parse_notebook(&nb).unwrap();
    match &parsed.cells[0] {
      CellEntry::Single(cell) => cell.content.clone(),
      CellEntry::Group(_) => panic!("Expected single cell"),
    }
  }

  #[test]
  fn test_fraction_of_fractions_keeps_its_grouping() {
    // A stacked fraction says how it groups by being two-dimensional. Once
    // flattened onto one line, `FractionBox[a/b, c/d]` without parentheses
    // would read as `a/b/c/d` — a different quantity.
    assert_eq!(
      inline_math_text(
        r#"FractionBox[RowBox[{"a", "/", "b"}], RowBox[{"c", "/", "d"}]]"#
      ),
      "(a/b)/(c/d)"
    );
  }

  #[test]
  fn test_fraction_of_sums_keeps_its_grouping() {
    assert_eq!(
      inline_math_text(
        r#"FractionBox[RowBox[{"a", "-", "b"}], RowBox[{"c", "-", "d"}]]"#
      ),
      "(a-b)/(c-d)"
    );
  }

  #[test]
  fn test_fraction_of_products_keeps_its_grouping() {
    // Juxtaposition binds no tighter than division in linear notation, so a
    // product denominator needs the parentheses just as much as a sum does.
    assert_eq!(
      inline_math_text(
        r#"FractionBox[RowBox[{"u", " ", "v"}], RowBox[{"g", " ", "h"}]]"#
      ),
      "(u v)/(g h)"
    );
  }

  #[test]
  fn test_fraction_of_single_symbols_stays_unparenthesized() {
    // Atoms already bind as one unit; parenthesizing them is only noise.
    assert_eq!(inline_math_text(r#"FractionBox["x", "H"]"#), "x/H");
    assert_eq!(
      inline_math_text(r#"FractionBox["1", SubscriptBox["C", "p"]]"#),
      "1/C_p"
    );
  }

  #[test]
  fn test_fraction_side_already_parenthesized_is_not_wrapped_twice() {
    assert_eq!(
      inline_math_text(
        r#"FractionBox["1", RowBox[{"(", RowBox[{"a", "+", "b"}], ")"}]]"#
      ),
      "1/(a+b)"
    );
  }

  #[test]
  fn test_fraction_of_negative_number_stays_unparenthesized() {
    // A sign belongs to the atom it precedes.
    assert_eq!(inline_math_text(r#"FractionBox["1", "-2"]"#), "1/-2");
  }

  #[test]
  fn test_subscript_on_empty_base_drops_the_underscore() {
    // `SubscriptBox["", …]` draws the script alone — there is no base for an
    // underscore to attach to, so `_C_p` would invent a symbol.
    assert_eq!(
      inline_math_text(r#"SubscriptBox["", SubscriptBox["C", "p"]]"#),
      "C_p"
    );
  }

  #[test]
  fn test_non_inline_math_cells_stay_dropped() {
    // Inline cells that are NOT inline formulas (the Demonstrations
    // "more info" opener buttons) still carry no textual content.
    let nb = r#"Notebook[{
Cell[TextData[{
 "Caption",
 Cell[BoxData[
  PaneSelectorBox[{True->"x"}, Dynamic[y]]],ExpressionUUID->
  "7d8221a0-ff0d-462f-ac85-47023eae4458"]
}], "Section"]
}]"#;
    let parsed = parse_notebook(nb).unwrap();
    match &parsed.cells[0] {
      CellEntry::Single(cell) => {
        assert_eq!(cell.style, CellStyle::Section);
        assert_eq!(cell.content, "Caption");
      }
      CellEntry::Group(_) => panic!("Expected single cell"),
    }
  }

  /// A typeset big operator stands *before* its body in the enclosing
  /// row, so the whole rest of that row is the summand — the same way
  /// the FrontEnd's own parser reads these boxes.
  #[test]
  fn test_big_operator_boxes_become_sum_and_product() {
    let s = r#"BoxData[RowBox[{
  UnderoverscriptBox["\[Sum]", RowBox[{"n", "=", "1"}], "m"],
  FractionBox[RowBox[{"Sin", "[", "t", "]"}], SuperscriptBox["n", "2"]]}]]"#;
    assert_eq!(
      extract_cell_content(s),
      "Sum[(Sin[t])/((n)^(2)), {n, 1, m}]"
    );
    // `∏` is the same shape, and the operator may be preceded by factors
    // that are *not* part of the body (the FrontEnd groups the operator
    // and its operand in a row of their own).
    let s = r#"BoxData[RowBox[{"0.4", RowBox[{
  UnderoverscriptBox["\[Product]", RowBox[{"k", "=", "2"}], "n"], "k"}]}]]"#;
    assert_eq!(extract_cell_content(s), "0.4Product[k, {k, 2, n}]");
    // Without a lower-limit assignment the two limits are the iterator
    // itself (`∑_i^m f` → `Sum[f, {i, m}]`), and an `UnderscriptBox` has
    // no upper limit at all (`∑_(n=1) f` → `Sum[f, n = 1]`).
    let s = r#"BoxData[RowBox[{UnderoverscriptBox["\[Sum]", "i", "m"], "f"}]]"#;
    assert_eq!(extract_cell_content(s), "Sum[f, {i, m}]");
    let s = r#"BoxData[RowBox[{UnderscriptBox["\[Sum]", RowBox[{"n", "=", "1"}]], "f"}]]"#;
    assert_eq!(extract_cell_content(s), "Sum[f, n=1]");
    // A non-operator base keeps its box (nothing to evaluate here).
    let s = r#"BoxData[RowBox[{UnderoverscriptBox["x", "a", "b"], "f"}]]"#;
    assert!(!extract_cell_content(s).starts_with("Sum["));
  }

  /// In prose the same box is display text, not code: `∑_(n=1)^m …`.
  #[test]
  fn test_big_operator_boxes_render_as_display_text() {
    let nb = r#"Notebook[{
Cell[TextData[{
 "Sums like ",
 Cell[BoxData[
  FormBox[
   RowBox[{
    UnderoverscriptBox["\[Sum]", RowBox[{"n", "=", "1"}], "m"],
    FractionBox[RowBox[{"sin", "[", "t", "]"}], SuperscriptBox["n", "2"]]}],
   TraditionalForm]], "InlineMath"],
 " are curves."
}], "Text"]
}]"#;
    let parsed = parse_notebook(nb).unwrap();
    match &parsed.cells[0] {
      CellEntry::Single(cell) => {
        assert_eq!(cell.content, "Sums like ∑_(n=1)^m sin[t]/n² are curves.");
      }
      CellEntry::Group(_) => panic!("Expected single cell"),
    }
  }

  /// A reaction scheme sets its rate constants over the arrows: an
  /// `OverscriptBox` when only one constant is written, an
  /// `UnderoverscriptBox` when the reverse rate is written under it. Both
  /// have to reach the prose as text, and so do the long arrows they sit
  /// on — a missing case used to leave the raw box source in the cell.
  #[test]
  fn test_overscript_boxes_render_as_display_text() {
    let nb = r#"Notebook[{
Cell[TextData[{
 Cell[BoxData[
  FormBox[
   RowBox[{"X",
    UnderoverscriptBox["\[DoubleLongLeftRightArrow]",
     SubsuperscriptBox["k", "1", "d"],
     SubsuperscriptBox["k", "1", "a"]], "Y",
    FormBox[
     OverscriptBox["\[LongRightArrow]",
      SubsuperscriptBox["k", "2", "a"]],
     TraditionalForm], "Z"}],
   TraditionalForm]], "InlineMath"],
 " is the scheme."
}], "Text"]
}]"#;
    let parsed = parse_notebook(nb).unwrap();
    match &parsed.cells[0] {
      CellEntry::Single(cell) => {
        assert_eq!(
          cell.content,
          "X⟺_(k_1^d)^(k_1^a)Y⟶^(k_2^a)Z is the scheme."
        );
      }
      CellEntry::Group(_) => panic!("Expected single cell"),
    }
  }

  /// An accent over a base is a diacritic rather than a script: `OverHat`,
  /// `OverBar` and `OverVector` all typeset as an `OverscriptBox` and read
  /// as the accented letter.
  #[test]
  fn test_overscript_accents_render_as_diacritics() {
    let nb = r#"Notebook[{
Cell[TextData[{
 Cell[BoxData[
  FormBox[
   RowBox[{OverscriptBox["x", "^"], "+", OverscriptBox["y", "_"], "+",
    OverscriptBox["z", "\[RightVector]"]}],
   TraditionalForm]], "InlineMath"]
}], "Text"]
}]"#;
    let parsed = parse_notebook(nb).unwrap();
    match &parsed.cells[0] {
      CellEntry::Single(cell) => {
        assert_eq!(cell.content, "x\u{0302}+y\u{0304}+z\u{20D7}");
      }
      CellEntry::Group(_) => panic!("Expected single cell"),
    }
  }

  /// Pasting an `InputForm`-formatted result back into a cell stores it as
  /// `InterpretationBox[StyleBox[…], InputForm[expr], …]`. The wrapper is
  /// only a record of how the boxes were laid out — the cell stands for
  /// `expr`, which is what re-evaluating it gives back. Keeping the wrapper
  /// would leave an inert one-element `InputForm[…]` object, so `Dimensions`
  /// and `Map` would see one opaque element instead of the array in it (the
  /// shape a Demonstration's coordinate table is written in).
  #[test]
  fn test_interpretation_box_drops_input_form_wrapper() {
    let nb = r#"Notebook[{
Cell[BoxData[
 RowBox[{"data", "=",
  InterpretationBox[
   StyleBox[
    RowBox[{"{", RowBox[{"1", ",", "2"}], "}"}],
    ShowStringCharacters->True,
    NumberMarks->True],
   InputForm[{{1, 2}, {3, 4}}],
   AutoDelete->True,
   Editable->True]}]], "Input"]
}]"#;
    let parsed = parse_notebook(nb).unwrap();
    match &parsed.cells[0] {
      CellEntry::Single(cell) => {
        assert_eq!(cell.content, "data={{1, 2}, {3, 4}}");
      }
      CellEntry::Group(_) => panic!("Expected single cell"),
    }
  }

  /// An `InterpretationBox` whose meaning is an ordinary expression keeps
  /// it verbatim — only display-form wrappers are peeled off.
  #[test]
  fn test_interpretation_box_keeps_plain_meaning() {
    let nb = r#"Notebook[{
Cell[BoxData[
 InterpretationBox[
  StyleBox["x", ShowStringCharacters->False],
  Quantity[3, "Meters"]]], "Input"]
}]"#;
    let parsed = parse_notebook(nb).unwrap();
    match &parsed.cells[0] {
      CellEntry::Single(cell) => {
        assert_eq!(cell.content, "Quantity[3, \"Meters\"]");
      }
      CellEntry::Group(_) => panic!("Expected single cell"),
    }
  }

  /// The FrontEnd hard-wraps long lines with a trailing backslash, and
  /// the break can land anywhere — including inside a run of closing
  /// brackets, which used to leave the tail of the box expression in the
  /// cell as raw text.
  #[test]
  fn test_line_wrap_continuations_are_rejoined() {
    let nb = "Notebook[{\n\
Cell[BoxData[\n\
 RowBox[{\"f\", \"[\", RowBox[{\"a\", \",\", \"b\"}\\\n], \"]\"}]], \"Input\"]\n\
}]";
    let parsed = parse_notebook(nb).unwrap();
    match &parsed.cells[0] {
      CellEntry::Single(cell) => assert_eq!(cell.content, "f[a,b]"),
      CellEntry::Group(_) => panic!("Expected single cell"),
    }
    // Inside a string literal the continuation is dropped too, joining
    // the two halves — but an *escaped* backslash at the end of a line
    // is a literal backslash, not a continuation.
    assert_eq!(strip_line_continuations("ab\\\ncd"), "abcd");
    assert_eq!(strip_line_continuations("ab\\\\\ncd"), "ab\\\\\ncd");
    assert_eq!(strip_line_continuations("ab\\\r\ncd"), "abcd");
    assert_eq!(strip_line_continuations("plain\ntext"), "plain\ntext");
  }
}
