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
      // Kernel messages ("During evaluation of In[1]:=") are stdout-like
      // output attached to the input that raised them, not prose.
      "Print" | "Message" => Some(Self::Print),
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
    return extract_row(s, true);
  }

  // Handle quoted strings
  extract_string_content(s)
}

/// Converts an `InterpretationBox[boxes, value]`'s `value` argument, which
/// (per `extract_cell_content`'s doc comment above) is written as ordinary
/// source rather than a box row. A bare list literal in that position must
/// keep its items comma-separated: routing it through `extract_cell_content`
/// instead would hit that function's "BoxData statement list" handling,
/// which *concatenates* top-level items the way `RowBox[{"a", "+",
/// "b"}]`'s children are joined into `a+b` — gluing two list elements
/// together with no separator reads back as implicit multiplication
/// (`{1, 2} {3, 4}` is `Times[{1, 2}, {3, 4}]`, not a 2-element list of
/// lists), silently turning stored data into a `Thread`ed product one
/// evaluation removed from the value the notebook actually saved.
fn convert_interpretation_meaning(s: &str) -> String {
  let trimmed = s.trim();
  if let Some(inner) =
    trimmed.strip_prefix('{').and_then(|s| s.strip_suffix('}'))
  {
    let items: Vec<String> = split_top_level_commas(inner)
      .into_iter()
      .map(|item| convert_interpretation_meaning(item.trim()))
      .collect();
    return format!("{{{}}}", items.join(", "));
  }
  extract_cell_content(trimmed)
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

/// Head Mathematica's box parser gives a `SuperscriptBox` whose script is
/// the bare operator character `op` (`"*"` → `SuperStar`, ...).
fn super_operator_head(op: &str) -> Option<&'static str> {
  match op.trim_matches('"') {
    "*" => Some("SuperStar"),
    "\u{2020}" => Some("SuperDagger"),
    // The postfix operator characters typeset as a superscript:
    // `SuperscriptBox["A", "\[Transpose]"]` is `Transpose[A]`.
    "\\[Transpose]" | "\u{F3C7}" => Some("Transpose"),
    "\\[Conjugate]" | "\u{F3C8}" => Some("Conjugate"),
    "\\[ConjugateTranspose]"
    | "\u{F3C9}"
    | "\\[HermitianConjugate]"
    | "\u{F3CE}" => Some("ConjugateTranspose"),
    _ => None,
  }
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

/// The order list `["n1", "n2", ...]` if `s` is `TagBox[order, Derivative]`
/// — the FrontEnd's typeset form of a mixed or higher-order partial
/// derivative's order tuple, used as a `SuperscriptBox`'s script whenever
/// `Derivative[n1, n2, ...][f]` can't be shown with prime marks (more than
/// one argument, or a single argument past the order primes are used for):
/// `SuperscriptBox["p", TagBox[RowBox[{"(", RowBox[{"1", ",", "0"}], ")"}],
/// Derivative]]` typesets `Derivative[1, 0][p]` as `p^(1,0)`. A
/// single-argument high order has no top-level comma at all (`p^(4)` is
/// `TagBox["(4)", Derivative]`). Returns `None` for any other superscript,
/// so callers fall back to the ordinary `(base)^(script)` reading.
fn derivative_tag_orders(s: &str) -> Option<Vec<String>> {
  let args = positional_box_args("TagBox", s.trim())?;
  let [content, tag] = &args[..] else {
    return None;
  };
  if tag.trim() != "Derivative" {
    return None;
  }
  let rendered = extract_cell_content(content);
  let inner = rendered.trim().strip_prefix('(')?.strip_suffix(')')?.trim();
  if inner.is_empty() {
    return None;
  }
  let orders: Vec<String> = split_top_level_commas(inner)
    .into_iter()
    .map(|o| o.trim().to_string())
    .collect();
  if orders.iter().any(std::string::String::is_empty) {
    None
  } else {
    Some(orders)
  }
}

/// The elements of a `TemplateBox`'s first argument, which holds the
/// template's slots. It is a list, written either as `{…}` (how a `.nb` file
/// spells it) or as `List[…]` (how `InputForm` writes a box escape).
/// The source a `TemplateBox[<|…|>, tag, InterpretationFunction ->
/// (Slot["key"] &)]` stands for: the association's `"key"` entry, and of an
/// `InterpretationBox[display, expr, …]` there its `expr`.
fn template_box_interpretation(args: &[String]) -> Option<String> {
  let assoc = args.first()?.trim();
  let inner = assoc.strip_prefix("<|")?.strip_suffix("|>")?;
  let function = args.iter().find_map(|a| {
    let (name, value) = a.split_once("->")?;
    (name.trim() == "InterpretationFunction").then_some(value.trim())
  })?;
  let key = function
    .split_once("Slot[\"")
    .and_then(|(_, rest)| rest.split_once("\"]"))
    .map(|(key, _)| key.to_string())?;
  let value = split_top_level_commas(inner)
    .into_iter()
    .find_map(|entry| {
      let (name, value) = entry.split_once("->")?;
      (name.trim().trim_matches('"') == key).then_some(value.trim())
    })?;
  if let Some(rest) = value.strip_prefix("InterpretationBox[")
    && let Ok((body, _)) = find_matching_bracket(rest)
    && let Some(expr) = split_top_level_commas(body).get(1)
  {
    return Some(strip_display_form_wrapper(expr.trim()));
  }
  Some(extract_cell_content(value))
}

/// The visible text of the HTML page held by an `EmbeddedHTML["…"]` box
/// part. Markup, scripts and styles are dropped, common entities are decoded
/// and an `<img>` is shown as `[Image: alt-or-file-name]`, since the page
/// itself cannot be rendered.
fn embedded_html_text(part: &str) -> Option<String> {
  let rest = &part[part.find("EmbeddedHTML[")? + "EmbeddedHTML[".len()..];
  let (body, _) = find_matching_bracket(rest).ok()?;
  let literal = body.trim().strip_prefix('"')?.strip_suffix('"')?;
  let html = unescape_string(&literal.replace("\\\n", ""));

  let mut out = String::new();
  let mut i = 0;
  while i < html.len() {
    let Some(lt) = html[i..].find('<') else {
      out.push_str(&html[i..]);
      break;
    };
    out.push_str(&html[i..i + lt]);
    i += lt;
    let Some(gt) = html[i..].find('>') else { break };
    let tag = &html[i + 1..i + gt];
    let name = tag
      .trim_start_matches('/')
      .split(|c: char| c.is_whitespace() || c == '/')
      .next()
      .unwrap_or("")
      .to_ascii_lowercase();
    i += gt + 1;
    match name.as_str() {
      "script" | "style" if !tag.starts_with('/') => {
        let close = format!("</{name}");
        i = html[i..]
          .to_ascii_lowercase()
          .find(&close)
          .and_then(|c| html[i + c..].find('>').map(|g| i + c + g + 1))
          .unwrap_or(html.len());
      }
      "img" => {
        let attr = |key: &str| {
          let at = tag.find(&format!("{key}=\""))? + key.len() + 2;
          let end = tag[at..].find('"')?;
          Some(tag[at..at + end].trim().to_string())
        };
        let label = attr("alt").filter(|a| !a.is_empty()).or_else(|| {
          attr("src").map(|s| s.rsplit('/').next().unwrap_or("").to_string())
        });
        out.push_str(&format!("[Image: {}]\n", label.unwrap_or_default()));
      }
      "br" | "p" | "div" | "li" | "tr" | "h1" | "h2" | "h3" => out.push('\n'),
      _ => {}
    }
  }
  let out = out
    .replace("&nbsp;", " ")
    .replace("&lt;", "<")
    .replace("&gt;", ">")
    .replace("&quot;", "\"")
    .replace("&amp;", "&");
  let lines: Vec<&str> = out
    .lines()
    .map(str::trim)
    .filter(|l| !l.is_empty())
    .collect();
  Some(lines.join("\n"))
}

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
    // A template over an association whose `InterpretationFunction` picks
    // one of its entries (`Slot["interp"] &`) — the FrontEnd's icon of a
    // `Root[…]` or of a long number — stands for that entry, which is
    // usually an `InterpretationBox[display, expr]`.
    if let Some(value) = template_box_interpretation(&args) {
      return Some(value);
    }
    let tag = args.last().unwrap().trim();
    let tag = tag.trim_matches('"');
    let parts = template_box_parts(&args[0]);
    if tag == "QuantityPrefix" || tag == "Quantity" {
      // First positional element is a list `{number, displayed_unit, unit_name, unit_id_string}`.
      if parts.len() >= 4 {
        let number = extract_cell_content(parts[0].trim());
        let unit_arg = parts[3].trim();
        // A compound unit (e.g. `Newtons Meters^2 / Kilograms^2`) is encoded
        // as nested typeset boxes (`FractionBox`, `RowBox`, `SuperscriptBox`,
        // …) rather than a single string literal — recognisable by the
        // presence of a box head's `[`. Recursing through the general
        // box-to-InputForm converter rebuilds it as a proper string-arithmetic
        // unit expression (`"Newtons" ("Meters"^2/"Kilograms"^2)`); treating
        // it as a bare string (the simple-unit path below) would instead
        // stringify the box source itself.
        if unit_arg.contains('[') {
          let unit_expr = extract_cell_content(unit_arg);
          return Some(format!("Quantity[{number}, {unit_expr}]"));
        }
        // Element 4 is the canonical unit name string. It's written as
        // `"\"USDollars\""` in the box expression (an *inner* string literal),
        // so we strip every layer of surrounding quotes/backslashes to get
        // the bare name, then re-wrap in a single pair of quotes.
        let mut unit_name = unit_arg.to_string();
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
    // A Demonstration's cloud-compatible snapshot slideshow is stored as
    // `TemplateBox[{InterpretationBox[Null, EmbeddedHTML["<html…>"]], …},
    // "EmbeddedHTML"]`. Show the page's visible text rather than its markup.
    if tag == "EmbeddedHTML"
      && let Some(text) = parts.first().and_then(|p| embedded_html_text(p))
    {
      return Some(text);
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
  {
    if let Some(mut image) = extract_image_from_boxes(first) {
      // The box form keeps the image's display size as a `GraphicsBox`
      // option, which the `Image[…]` constructor takes as `ImageSize`.
      if let Some(size) = args[1..]
        .iter()
        .filter_map(|a| a.trim().strip_prefix("ImageSize"))
        .find_map(|r| r.trim_start().strip_prefix("->"))
        .map(normalize_whitespace)
        .filter(|size| size != "Automatic")
      {
        image.pop();
        image.push_str(&format!(", ImageSize -> {size}]"));
      }
      return Some(image);
    }
    // A typeset `Graph` is a `NamespaceBox["NetworkGraphics", …]` whose
    // dynamic module keeps the graph itself as
    // `Typeset\`graph = HoldComplete[Graph[…]]`: that is the value.
    if first
      .trim_start()
      .starts_with("NamespaceBox[\"NetworkGraphics\"")
      && let Some(i) = first.find("Typeset`graph = HoldComplete[")
      && let Ok((graph, _)) = find_matching_bracket(
        &first[i + "Typeset`graph = HoldComplete[".len()..],
      )
    {
      return Some(graph.trim().to_string());
    }
    // Not a raster snapshot — an inline `Graphics[…]` literal built from
    // vector primitives (`PointBox`, `FilledCurveBox`, …), e.g. a
    // precomputed point set a Demonstration's initialization code bakes in
    // rather than recomputing on every evaluation. Rebuild the evaluable
    // `Graphics[…]` expression the same way, so the cell stays runnable
    // instead of embedding raw, un-evaluable box syntax.
    return Some(box_source_to_graphics_expr(s));
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
    "ButtonBox",
    "CheckboxBox",
  ] {
    if !s.starts_with(head) {
      continue;
    }
    let mut args = split_args(head, s)?;
    // Drop trailing display options (`MultilineFunction -> None`, …) so the
    // positional-argument matches below see the box's real arity.
    args.retain(|a| !is_option_arg(a));
    // A bare string operand (`FractionBox["x", X
    // symbol name, so it reads as code: the private-use letter keeps its
    // code point instead of becoming the Unicode glyph a prose cell draws.
    let conv = |a: &str| {
      let a = a.trim();
      if a.len() >= 2
        && a.starts_with('"')
        && a.ends_with('"')
        && !a[1..a.len() - 1].contains("\\\"")
      {
        unescape_code_string(&a[1..a.len() - 1])
      } else {
        extract_cell_content(a)
      }
    };
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
      // `FractionBox[a, b]` → `((a)/(b))`. The parentheses preserve operator
      // precedence around composite numerators / denominators, and the outer
      // pair keeps the quotient one operand of a tighter-binding neighbour
      // (`Ceiling /@ FractionBox[l, d]` is `Ceiling /@ (l/d)`).
      "FractionBox" if args.len() == 2 => {
        format!("(({})/({}))", conv(&args[0]), conv(&args[1]))
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
      // A bare `*` or `\[Dagger]` script is the typeset form of `SuperStar`
      // and `SuperDagger`
      // (`SuperscriptBox["A", "*"]` is `SuperStar[A]`), which are
      // evaluable heads users can attach definitions to.
      "SuperscriptBox"
        if args.len() == 2
          && super_operator_head(&conv(&args[1])).is_some() =>
      {
        format!(
          "{}[{}]",
          super_operator_head(&conv(&args[1])).unwrap(),
          conv(&args[0])
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
      // `SuperscriptBox[f, TagBox[order, Derivative]]` is the typeset form
      // of a mixed or higher-order partial derivative — emit
      // `Derivative[n1, n2, ...][f]`, not exponentiation. Must be checked
      // before the generic "doesn't parse as an expression" fallback below,
      // since the bare order tuple (`(1, 0)`) is not valid code on its own.
      "SuperscriptBox"
        if args.len() == 2 && derivative_tag_orders(&args[1]).is_some() =>
      {
        format!(
          "Derivative[{}][{}]",
          derivative_tag_orders(&args[1]).unwrap().join(", "),
          conv(&args[0])
        )
      }
      // A script that isn't a valid expression on its own — e.g. a bare
      // `-` used only to typeset an ion's charge, like the "H" of "H⁻"
      // (`SuperscriptBox["H", "-"]`) — can't become `(a)^(b)`: `("H")^(-)`
      // dangles the minus sign with no operand and fails to parse.
      // Mathematica's own box-to-expression conversion falls back to the
      // display wrapper `Superscript` whenever a script doesn't parse as
      // an expression; mirror that here (same fallback `SubscriptBox`
      // uses below for its subscript).
      "SuperscriptBox"
        if args.len() == 2
          && crate::parse_to_expr(&conv(&args[1])).is_err() =>
      {
        format!(
          "Superscript[{}, \"{}\"]",
          conv(&args[0]),
          escape_string(&conv(&args[1]))
        )
      }
      // `SuperscriptBox[a, b]` → `(a)^(b)`.
      // The outer pair keeps the power one operand of a tighter-binding
      // neighbour: `f /@ SuperscriptBox[rx, n]` is `f /@ (rx^n)`.
      "SuperscriptBox" if args.len() == 2 => {
        format!("(({})^({}))", conv(&args[0]), conv(&args[1]))
      }
      // `SubscriptBox[a, b]` → `Subscript[a, b]` (Wolfram's evaluable form),
      // except when the subscript is a `\[LeftDoubleBracket]…\]` group: the
      // FrontEnd also accepts `Part` typeset as a bracketed subscript, and
      // `SubscriptBox["c", RowBox[{"⟦", "1", "⟧"}]]` means `c[[1]]`.
      "SubscriptBox" if args.len() == 2 => {
        let sub = conv(&args[1]);
        let base = conv(&args[0]);
        // `Part` typed with ASCII brackets as the subscript, `c` over
        // `[[1]]`, is `c[[1]]` too.
        let ascii_part = sub
          .trim()
          .strip_prefix("[[")
          .and_then(|r| r.strip_suffix("]]"))
          .map(str::trim);
        if let Some(spec) =
          part_spec_inside_double_brackets(&sub).or(ascii_part)
        {
          format_part_access(&base, spec)
        } else if split_top_level_commas(&sub).len() > 1
          && split_top_level_commas(&sub)
            .iter()
            .all(|p| crate::parse_to_expr(p.trim()).is_ok())
        {
          // `X` over `1,i` is `Subscript[X, 1, i]`: one index per entry.
          let base = if crate::parse_to_expr(&base).is_ok() {
            base
          } else {
            format!("\"{}\"", escape_string(&base))
          };
          format!("Subscript[{base}, {sub}]")
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
          // A base that is bare punctuation (the lone `)` of a typeset
          // `C(CH₃)₄`) is likewise display text, not code.
          let base = if crate::parse_to_expr(&base).is_ok() {
            base
          } else {
            format!("\"{}\"", escape_string(&base))
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
      // A superscript that isn't a valid expression on its own (the bare
      // `+` of a typeset "sweep₍p₎⁺") can't become `(Subscript[a, b])^(+)`;
      // fall back to the `Subsuperscript` display wrapper with the script
      // quoted, as `SuperscriptBox` does above.
      "SubsuperscriptBox"
        if args.len() == 3
          && crate::parse_to_expr(&conv(&args[2])).is_err() =>
      {
        let sub = conv(&args[1]);
        let sub = if crate::parse_to_expr(&sub).is_ok() {
          sub
        } else {
          format!("\"{}\"", escape_string(&sub))
        };
        format!(
          "Subsuperscript[{}, {}, \"{}\"]",
          conv(&args[0]),
          sub,
          escape_string(&conv(&args[2]))
        )
      }
      // `SubsuperscriptBox[a, b, c]` → `Subscript[a, b]^c`.
      "SubsuperscriptBox" if args.len() == 3 => {
        format!(
          "((Subscript[{}, {}])^({}))",
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
      // `RadicalBox[a, n]` → `a^(1/n)`. This is the principal power, not
      // `Surd[a, n]`, so the index may be any number (e.g. a symbolic or
      // real Weibull shape parameter), not only an integer.
      "RadicalBox" if args.len() == 2 => {
        format!("Power[{}, Power[{}, -1]]", conv(&args[0]), conv(&args[1]))
      }
      // `TagBox[content, tag, opts...]` is a display annotation; the
      // evaluable value is just `content`.
      "TagBox" if !args.is_empty() => conv(&args[0]),
      // `StyleBox[op, "OperatorCharacter"]` colors one operator character
      // of a larger expression (e.g. a `Which` guard's `&&`/`==`) without
      // an enclosing `RowBox` grouping it with its operands — the FrontEnd
      // uses this specifically to syntax-highlight code, never for prose.
      // Recursing through the generic `conv` (`extract_cell_content`)
      // below would reconstruct a bare `"\[Equal]"` argument through the
      // *prose* unescaper (the same fallback a `Cell[…, "Text"]` cell's
      // plain string content goes through), which maps `\[Equal]` to the
      // single display glyph `=` it looks like typeset — silently turning
      // an equality test into an assignment (`Set`). Only a bare named-
      // character token (not a real string that merely starts with one)
      // takes this path, so an actual quoted string styled this way still
      // renders as prose.
      //
      // The FrontEnd also colors operators with option-only wrappers
      // (`StyleBox["\[Equal]", FontColor -> GrayLevel[0]]`, as in a
      // Demonstration's code cells), which need the same treatment.
      "StyleBox"
        if !args.is_empty() && stylebox_bare_operator(&args[0]).is_some() =>
      {
        stylebox_bare_operator(&args[0]).unwrap().to_string()
      }
      // `StyleBox`, `FrameBox`, `AdjustmentBox`, `FormBox` similarly wrap
      // a displayed expression; recurse into the first arg.
      // `ButtonBox["SeedRandom", BaseStyle -> "Link", …]` is a documentation
      // hyperlink pasted into code; it displays (and evaluates as) its label.
      "StyleBox" | "FrameBox" | "AdjustmentBox" | "FormBox" | "ButtonBox"
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
        let raw_meaning = args[1].trim();
        // `value` is written as ordinary source (not a box row), so a bare
        // string here is already a String *literal* — e.g. a large
        // `Compress`ed blob the FrontEnd iconizes for display but stores
        // faithfully as `drawfig = "1:eJzs…"`. `extract_cell_content`'s
        // usual handling of a top-level quoted string is for *display*
        // text (`Cell["…", "Text"]`) and strips the quotes; doing that here
        // would turn the string into a bareword identifier and make the
        // cell fail to parse.
        if raw_meaning.starts_with('"')
          && raw_meaning.ends_with('"')
          && raw_meaning.len() >= 2
        {
          let inner = &raw_meaning[1..raw_meaning.len() - 1];
          format!("\"{}\"", escape_string(&unescape_string(inner)))
        } else if let Some(items) = braced_list_items(raw_meaning) {
          // Likewise a braced list is a list *value* (an iconized
          // `{Graphics[…], …}`), not a row of boxes to concatenate — which
          // would fuse the items into a product.
          let items: Vec<String> = items
            .into_iter()
            .map(|item| {
              extract_cell_content(&format!("InterpretationBox[x, {item}]"))
            })
            .collect();
          format!("{{{}}}", items.join(", "))
        } else {
          strip_display_form_wrapper(&convert_interpretation_meaning(
            raw_meaning,
          ))
        }
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
            // A quoted cell is code like a row element (`"\[ScriptM]"`
            // names the same symbol it does inside a `RowBox`), not prose
            // whose private-use letters give way to visible glyphs.
            let cells: Vec<String> = split_top_level_commas(row_inner)
              .into_iter()
              .map(|cell| {
                let cell = cell.trim();
                if cell.starts_with('"') {
                  box_part_source(cell)
                } else {
                  extract_cell_content(cell)
                }
              })
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
  piecewise_from_grid_box(grid)
}

/// `Piecewise[{{v1, c1}, …}]` from a `GridBox[{{v1, c1}, …}, …]` of
/// value/condition rows — the piece of a typeset `Piecewise` that sits
/// beside the `\[Piecewise]` brace, whether that pair is wrapped in an
/// outer `GridBox` row (`piecewise_from_grid`, above) or written directly
/// in a `RowBox` alongside the brace character.
fn piecewise_from_grid_box(grid: &str) -> Option<String> {
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

/// The two entries of a two-row, one-column `GridBox[{{top}, {bottom}}]`;
/// `None` for any other grid shape.
fn stacked_pair_of_grid_box(grid: &str) -> Option<(String, String)> {
  let inner = positional_box_args("GridBox", grid)?;
  let rows = braced_list_items(inner.first()?)?;
  let [top, bottom] = rows[..] else {
    return None;
  };
  let [top] = braced_list_items(top)?[..] else {
    return None;
  };
  let [bottom] = braced_list_items(bottom)?[..] else {
    return None;
  };
  Some((top.to_string(), bottom.to_string()))
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
  if args.len() == 3 && {
    let mid = args[1].trim().trim_matches('"').trim();
    mid.is_empty() || mid == r"\[InvisibleSpace]" || mid == "\u{2063}"
  } {
    args.remove(1);
  }
  if args.len() != 2 {
    return None;
  }
  let head = args[0].trim().trim_matches('"').trim();
  (head == r"\[DifferentialD]" || head == "\u{2146}" || head == "\u{F74C}")
    .then(|| extract_cell_content(args[1]))
}

/// `Integrate[…]` for an integral sign with limits `limits` followed by the
/// boxes `parts`. Nested signs (`∫_0^π ∫_a^b ρ ⅆρ ⅆθ`) all take their body
/// from the innermost row, whose trailing differentials pair with the signs
/// innermost-first; the iterators are listed outermost-first, as Wolfram's
/// own parser does: `Integrate[ρ, {θ, 0, π}, {ρ, a, b}]`.
fn nested_integral(
  limits: Option<(String, String)>,
  parts: &[&str],
) -> Option<String> {
  fn unwrap_row(parts: &[&str]) -> Vec<String> {
    if let [only] = parts
      && let Some(inner) = only.trim().strip_prefix("RowBox[")
      && let Some(inner) = inner.strip_suffix(']')
      && let Some(inner) = inner.trim().strip_prefix('{')
      && let Some(inner) = inner.strip_suffix('}')
    {
      let inner: Vec<&str> = split_top_level_commas(inner);
      return unwrap_row(&inner);
    }
    parts.iter().map(std::string::ToString::to_string).collect()
  }
  let mut signs = vec![limits];
  let mut body = unwrap_row(parts);
  while body.len() > 1
    && let Some(inner_limits) = integral_limits(&body[0])
  {
    signs.push(inner_limits);
    let rest: Vec<&str> = body[1..].iter().map(String::as_str).collect();
    body = unwrap_row(&rest);
  }
  // Trailing differentials, popped from the end: the outermost sign's
  // first, matching `signs`.
  let mut vars: Vec<String> = Vec::new();
  while vars.len() < signs.len()
    && let Some(last) = body.last()
    && let Some(var) = differential_variable(last)
  {
    vars.push(var);
    body.pop();
  }
  if vars.len() != signs.len() {
    return None;
  }
  let integrand = extract_rowbox_content(&body.join(","));
  if integrand.trim().is_empty() {
    return None;
  }
  let iterators: Vec<String> = signs
    .iter()
    .zip(vars.iter())
    .map(|(limits, var)| match limits {
      Some((lo, hi)) => {
        format!("{{{}, {}, {}}}", var.trim(), lo.trim(), hi.trim())
      }
      None => var.trim().to_string(),
    })
    .collect();
  Some(format!("Integrate[{integrand}, {}]", iterators.join(", ")))
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
  // The operator may be wrapped in a `StyleBox` (`StyleBox[…, ScriptLevel
  // -> 0]` keeps the limits full-size inside a fraction).
  let s = positional_box_args("StyleBox", s.trim())
    .filter(|a| !a.is_empty())
    .map_or_else(|| s.to_string(), |a| a[0].trim().to_string());
  let s = s.as_str();
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

/// The head and variables of a typeset `∇` operator:
/// `SubscriptBox["\\[Del]", vars]` is the gradient and
/// `SubsuperscriptBox["\\[Del]", vars, "2"]` the Laplacian.
fn del_operator(s: &str) -> Option<(&'static str, String)> {
  let is_del = |b: &str| {
    let b = extract_cell_content(b);
    let b = b.trim();
    b == "\u{2207}" || b == r"\[Del]"
  };
  if let Some(args) =
    positional_box_args("SubsuperscriptBox", s.trim()).filter(|a| a.len() == 3)
  {
    return (is_del(&args[0]) && extract_cell_content(&args[2]).trim() == "2")
      .then(|| ("Laplacian", extract_cell_content(&args[1])));
  }
  let args =
    positional_box_args("SubscriptBox", s.trim()).filter(|a| a.len() == 2)?;
  is_del(&args[0]).then(|| ("Grad", extract_cell_content(&args[1])))
}

/// The variables of a `SubscriptBox["\\[PartialD]", vars]` — the typeset
/// partial-derivative operator. The FrontEnd reads
/// `SubscriptBox["\\[PartialD]", x] f[x]` as `D[f[x], x]` and
/// `SubscriptBox["\\[PartialD]", RowBox[{"x", ",", "x"}]] f[x]` as
/// `D[f[x], x, x]`, so the operand is whatever follows in the row.
fn partial_derivative_vars(s: &str) -> Option<String> {
  // The FrontEnd's syntax coloring may wrap the operator in a `StyleBox`
  // (`StyleBox[SubscriptBox["\[PartialD]", "x"], FontColor -> …]`).
  let s = positional_box_args("StyleBox", s.trim())
    .filter(|a| !a.is_empty())
    .map_or_else(|| s.to_string(), |a| a[0].trim().to_string());
  let args =
    positional_box_args("SubscriptBox", &s).filter(|a| a.len() == 2)?;
  let base = extract_cell_content(&args[0]);
  (base.trim() == "\u{2202}").then(|| extract_cell_content(&args[1]))
}

/// Append `piece` to `result`, inserting the operator implicit
/// multiplication needs when the two would otherwise fuse into one longer
/// identifier — e.g. a preceding `2 \[Pi]` (rendered as the bare letter
/// `π`) directly followed by `Integrate[…]` must not read back as the
/// single undefined symbol `πIntegrate`. Mirrors the juxtaposition rule
/// applied to ordinary sibling box pieces below: a trailing run of word
/// characters that contains a letter, followed by a piece starting with a
/// letter or a digit, needs a `*` between them (`RowBox[{"d", "400"}]` is
/// `d*400`, not the symbol `d400`), as does a number followed by a number
/// (`RowBox[{"2", "400"}]`); a `#` prefix needs a space instead, for
/// named-slot syntax. A number followed by a letter (`2x`) already reads
/// as a product.
/// Whether two adjacent row parts are the pieces of one word whose middle a
/// `StyleBox` restyles: a bare word token next to a `StyleBox` around a bare
/// word token, with no space token between them.
fn styling_splits_word(prev: &str, part: &str) -> bool {
  let word = |s: &str| -> Option<String> {
    let t = s.trim().strip_prefix('"')?.strip_suffix('"')?;
    (!t.is_empty() && t.chars().all(char::is_alphanumeric))
      .then(|| t.to_string())
  };
  let styled = |s: &str| {
    positional_box_args("StyleBox", s)
      .and_then(|args| args.first().and_then(|a| word(a)))
  };
  matches!(
    (word(prev), styled(part), styled(prev), word(part)),
    (Some(_), Some(_), _, _) | (_, _, Some(_), Some(_))
  )
}

fn push_juxtaposed(result: &mut String, piece: &str) {
  let piece_starts_alpha =
    piece.chars().next().is_some_and(char::is_alphabetic);
  let piece_starts_digit =
    piece.chars().next().is_some_and(|c| c.is_ascii_digit());
  let trailing_run: Vec<char> = result
    .chars()
    .rev()
    .take_while(|c| c.is_alphanumeric() || *c == '_')
    .collect();
  let trailing_run_has_letter = trailing_run.iter().any(|c| c.is_alphabetic());
  let trailing_run_is_number =
    !trailing_run.is_empty() && trailing_run.iter().all(char::is_ascii_digit);
  // Two operator tokens side by side would fuse into a different one:
  // `RowBox[{RowBox[{"view", "!"}], "\[Equal]", …}]` is `(view!) == …`,
  // which glued reads `view!== …` and does not parse, and `x`, `-`, `-1`
  // would read as a decrement.
  const OPERATOR_CHARS: &[char] = &[
    '!', '=', '+', '-', '*', '/', '<', '>', '&', '|', '.', ':', '^', '@', '~',
    '?', '%',
  ];
  if result.ends_with(OPERATOR_CHARS) && piece.starts_with(OPERATOR_CHARS)
    || result.ends_with('#') && piece_starts_alpha
  {
    result.push(' ');
  } else if result.ends_with("/.")
    && piece.chars().next().is_some_and(|c| c.is_ascii_digit())
  {
    // `x /. 0 -> y`: glued as `x/.0->y` the tokenizer reads `/` then the
    // real number `.0` instead of the `/.` (ReplaceAll) operator.
    result.push(' ');
  } else if trailing_run_has_letter
    && (piece_starts_alpha || piece_starts_digit)
    || trailing_run_is_number && piece_starts_digit
  {
    result.push('*');
  }
  result.push_str(piece);
}

/// Does `s` (plain, already-flattened WL source) carry a bracket-depth-0
/// assignment operator (`=`, `:=`, `+=`, `-=`, `*=`, `/=`, `^=`) — as
/// opposed to a comparison (`==`, `<=`, `>=`, `!=`) or one nested inside a
/// call/list/group? Used to decide whether a juxtaposed `RowBox` factor
/// needs re-parenthesizing — see the call site in [`extract_rowbox_content`].
fn has_top_level_assignment(s: &str) -> bool {
  let chars: Vec<char> = s.chars().collect();
  let mut depth = 0i32;
  let mut in_string = false;
  let mut prev_backslash = false;
  let mut i = 0;
  while i < chars.len() {
    let c = chars[i];
    if in_string {
      if c == '"' && !prev_backslash {
        in_string = false;
      }
      prev_backslash = c == '\\' && !prev_backslash;
      i += 1;
      continue;
    }
    match c {
      '"' => in_string = true,
      '{' | '[' | '(' => depth += 1,
      '}' | ']' | ')' => depth -= 1,
      '=' if depth == 0 => {
        // `=!=` (UnsameQ): not an assignment.
        if chars.get(i + 1) == Some(&'!') && chars.get(i + 2) == Some(&'=') {
          i += 3;
          continue;
        }
        // `==` / `===` (Equal / SameQ): not an assignment. `===` is three
        // consecutive `=` characters — consume all of them so the trailing
        // one isn't mistaken for a fresh (assignment) `=`.
        if chars.get(i + 1) == Some(&'=') {
          i += 2;
          while chars.get(i) == Some(&'=') {
            i += 1;
          }
          continue;
        }
        // `<=`, `>=`, `!=`: comparisons, not assignments.
        if i > 0 && matches!(chars[i - 1], '<' | '>' | '!') {
          i += 1;
          continue;
        }
        // A bare `=` (Set), or the second character of `:=`, `+=`, `-=`,
        // `*=`, `/=`, `^=` (SetDelayed / the AddTo family) — all bind
        // looser than implicit multiplication.
        return true;
      }
      _ => {}
    }
    i += 1;
  }
  false
}

/// Is `part` (one raw element of a `RowBox`'s child list, as split by
/// [`split_top_level_commas`]) a token that carries no meaning of its own —
/// display-only 2-D layout whitespace (`" "`, `\[IndentingNewLine]`)? A
/// literal `"\n"` is deliberately *not* included: that's the distinct
/// multi-statement-list convention (`BoxData[{stmt, "\n", stmt, …}]`, see
/// [`extract_cell_content`]) marking a hard boundary between independent
/// statements, not in-expression formatting.
fn is_soft_whitespace_box_token(part: &str) -> bool {
  let t = part.trim();
  t == "\" \"" || t == "\"\\[IndentingNewLine]\""
}

/// Did a real (non-whitespace) sibling *expression* already appear before
/// `parts[i]` in this row, with nothing but soft layout whitespace between
/// them — i.e. would placing `parts[i]` here read as implicit
/// multiplication against it? Walks backward from `i`, skipping
/// [`is_soft_whitespace_box_token`] entries; a literal `"\n"`
/// multi-statement separator stops the walk (`false`) since it marks the
/// start of a new, independent statement — see the call site in
/// [`extract_rowbox_content`].
///
/// A bracket-opener or list/statement separator (`[`, `(`, `{`, `,`, `;`)
/// also stops the walk (`false`): it already establishes that `parts[i]`
/// begins a fresh argument, element, or statement, so there is no
/// preceding *value* for it to be juxtaposed against. Treating one of
/// these as "a real predecessor" wrongly parenthesizes a function's whole
/// `body, iterator` argument list (e.g. `Table[stmt; stmt; …, {i, 0, n}]`,
/// found immediately after `Table["["`) as one factor — trapping the
/// argument-separating comma inside the added parens and producing
/// unparseable text; the same misfire wraps a whole comma-separated
/// `Module` local-variable list, or a `With[{v = 1}, body]`'s body sitting
/// right after the binding list's own `,`, in one invalid `(a, b, …)`
/// group instead of leaving each item alone.
fn has_real_predecessor(
  parts: &[String],
  i: usize,
  newline_ends_statement: bool,
) -> bool {
  for part in parts[..i].iter().rev() {
    let t = part.trim();
    // Inside a row a line break is layout, not a statement boundary
    if t == "\"\\n\"" && !newline_ends_statement {
      continue;
    }
    // A literal `"\n"` multi-statement separator, a list/argument comma or
    // statement `;`, or an opening delimiter starting the enclosing group
    // are all structural punctuation, not a value the current part could be
    // juxtaposed (implicitly multiplied) against — e.g. the second binding
    // of a `With[{a = …, b = …}, …]` list follows a `","`, not an
    // expression, so it is never "juxtaposed" onto the first; likewise an
    // argument list's own opening `"["` precedes its contents, not a value
    // to multiply them against — e.g. `f[a = 1; g[a], h]`
    // (`RowBox[{"f", "[", RowBox[{assignment-box, ",", "h"}], "]"}]`) needs
    // no parentheses at all, since the comma already ends that argument and
    // reparsing the flattened text recovers the original grouping as-is.
    if matches!(
      t,
      "\"\\n\"" | "\",\"" | "\";\"" | "\"{\"" | "\"(\"" | "\"[\""
    ) {
      return false;
    }
    // A comment, `(* … *)` (a bare token or a RowBox around one), is no
    // value either: whatever stands before it decides.
    if is_soft_whitespace_box_token(t)
      || t.starts_with("RowBox[{\"(*\"")
      || t.starts_with("\"(*")
    {
      continue;
    }
    if is_bare_char(t, '[')
      || is_bare_char(t, '(')
      || is_bare_char(t, '{')
      || is_bare_char(t, ',')
      || is_bare_char(t, ';')
    {
      return false;
    }
    return true;
  }
  false
}

/// Collapse a `\[LeftDoubleBracket] … \[RightDoubleBracket]` pair back into
/// `Part` syntax before the row is joined.
///
/// `a[[i]]` typesets as a *flat* row — `RowBox[{"a", "\[LeftDoubleBracket]",
/// i, "\[RightDoubleBracket]"}]` — not as a `SubscriptBox` (that form exists
/// too, and is handled separately in `extract_typeset_box`, but is the rarer
/// one in practice). Left unhandled, the bracket names would fall through
/// to their literal Unicode glyphs (`⟦`/`⟧`), which Woxi's parser does not
/// accept as `Part` syntax, breaking every cell that indexes a list this
/// way — extremely common in Demonstrations-style code.
///
/// Runs before the main row-scanning loop so chained accesses (`a[[i]][[j]]`,
/// itself a flat run of two such pairs) fold left-to-right: each merge's
/// plain-text result becomes the `base` of the next pair to its right.
fn merge_double_bracket_parts(parts: &[&str]) -> Vec<String> {
  let mut out: Vec<String> =
    parts.iter().map(|p| p.trim().to_string()).collect();
  while let Some(k) = (1..out.len().saturating_sub(2)).find(|&k| {
    is_bare_named_char(&out[k], "LeftDoubleBracket")
      && is_bare_named_char(&out[k + 2], "RightDoubleBracket")
  }) {
    // Layout whitespace between the base and `[[` (a line break after a
    // closing `]`) doesn't detach the index from its base.
    let mut b = k - 1;
    while b > 0 && is_soft_whitespace_box_token(&out[b]) {
      b -= 1;
    }
    let base = box_part_source(&out[b]);
    let spec = box_part_source(&out[k + 1]);
    out.splice(b..=k + 2, [format_part_access(&base, &spec)]);
  }
  out
}

/// Extract text from a RowBox expression by concatenating string
/// elements.
fn extract_rowbox_content(s: &str) -> String {
  extract_row(s, false)
}

/// The text of a box row. `newline_ends_statement` is set only for a
/// cell's top-level statement list (`BoxData[{stmt, "\n", stmt}]`), where
/// a `"\n"` separates statements; inside a `RowBox` it is layout, and two
/// operands either side of it multiply — `RowBox[{"0", "\n",
/// RowBox[{"b", "=", "1"}]}]` is `0*(b = 1)` in wolframscript.
fn extract_row(s: &str, newline_ends_statement: bool) -> String {
  let s = s.trim();
  let s = s.strip_prefix('{').unwrap_or(s);
  let s = s.strip_suffix('}').unwrap_or(s);

  let parts = merge_double_bracket_parts(&split_top_level_commas(s));
  let mut result = String::new();
  let mut i = 0;
  while i < parts.len() {
    let part = parts[i].trim();
    // A big-operator box (`∑`, `∏`) is written before its body, and the
    // FrontEnd groups the operator with exactly its operand in a row of
    // their own — so everything after it here is the body.
    if i + 1 < parts.len()
      && let Some((head, iterator)) = big_operator_call(part)
    {
      let body = extract_rowbox_content(&parts[i + 1..].join(","));
      push_juxtaposed(&mut result, &format!("{head}[{body}, {iterator}]"));
      break;
    }
    // The integral sign also takes the rest of the row as its body; the `ⅆx`
    // that closes the body names the integration variable, and a
    // `SubsuperscriptBox` sign carries the limits of a definite integral.
    if i + 1 < parts.len()
      && let Some(limits) = integral_limits(part)
      && let Some(integral) = nested_integral(
        limits,
        &parts[i + 1..]
          .iter()
          .map(String::as_str)
          .collect::<Vec<_>>(),
      )
    {
      push_juxtaposed(&mut result, &integral);
      break;
    }
    // `∇` with a subscript of variables is the typeset gradient,
    // `Grad[u, vars]`, and with a superscript 2 the Laplacian,
    // `Laplacian[u, vars]`; the operand is the box that follows (as in
    // Wolfram's parser: `∇²u == 0` is `Laplacian[u, …] == 0`).
    if i + 1 < parts.len()
      && let Some((head, vars)) = del_operator(part)
    {
      let operand = extract_cell_content(parts[i + 1].trim());
      push_juxtaposed(&mut result, &format!("{head}[{operand}, {vars}]"));
      i += 2;
      continue;
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
    // A `\[Piecewise]` brace immediately followed by a `GridBox` of
    // value/condition rows is the typeset form of `Piecewise[…]` written
    // directly in a row (`f[z_] = \[Piecewise]GridBox[{…}]`), rather than
    // wrapped in an outer `GridBox` as `piecewise_from_grid` expects.
    if i + 1 < parts.len()
      && is_bare_named_char(part, "Piecewise")
      && parts[i + 1].trim().starts_with("GridBox")
      && let Some(piecewise) = piecewise_from_grid_box(parts[i + 1].trim())
    {
      push_juxtaposed(&mut result, &piecewise);
      i += 2;
      continue;
    }
    // `\[LeftBracketingBar] body \[RightBracketingBar]` is the typeset form
    // of `Abs[body]` — the only thing that pair of glyphs ever brackets.
    if i + 2 < parts.len()
      && is_bare_named_char(part, "LeftBracketingBar")
      && is_bare_named_char(parts[i + 2].trim(), "RightBracketingBar")
    {
      push_juxtaposed(
        &mut result,
        &format!("Abs[{}]", box_part_source(parts[i + 1].trim())),
      );
      i += 3;
      continue;
    }
    // A symbol immediately followed by a parenthesised, comma-separated
    // argument list — `RowBox[{"\[Psi]", "(", RowBox[{"x", ",", "0"}],
    // ")"}]` — is the FrontEnd's TraditionalForm-input rendering of function
    // application (textbook notation `f(x, y)` for `f[x, y]`), which the
    // Demonstrations Project uses for a function's typeset label. A bare
    // comma is never valid inside ordinary parentheses (that's a grouping,
    // not an argument list), so this shape is unambiguous — unlike a
    // single-argument `f(x)`, which is already valid, equivalent
    // multiplication syntax and is left alone.
    if i + 3 < parts.len()
      && is_bare_char(parts[i + 1].trim(), '(')
      && is_bare_char(parts[i + 3].trim(), ')')
      && let Some(args) = paren_call_arglist(parts[i + 2].trim())
    {
      result.push_str(&box_part_source(part));
      result.push('[');
      result.push_str(&args.join(", "));
      result.push(']');
      i += 4;
      continue;
    }
    // TraditionalForm writes the elementary functions as lowercase
    // textbook names applied with parentheses — `exp(x)`, `sin(x)`,
    // `ln(x)` — where even a single argument is a call, not a product of
    // the symbols `exp` and `x`.
    if i + 3 < parts.len()
      && is_bare_char(parts[i + 1].trim(), '(')
      && is_bare_char(parts[i + 3].trim(), ')')
      && let Some(head) = traditional_function_name(part.trim())
    {
      let inner = parts[i + 2].trim();
      let args = paren_call_arglist(inner)
        .map_or_else(|| box_part_source(inner), |a| a.join(", "));
      push_juxtaposed(&mut result, &format!("{head}[{args}]"));
      i += 4;
      continue;
    }
    let piece = box_part_source(part);
    // Two sibling boxes juxtaposed with no operator between them (no
    // `\[InvisibleTimes]`, no literal `" "` part) mean implicit
    // multiplication — e.g. `RowBox[{"a", RowBox[{"x", "^", "3"}]}]` for
    // `a*x^3`. Gluing their text together verbatim would instead read back
    // as one longer identifier: a trailing symbol swallows a following
    // letter into a different name (`ax^3` parses as the single identifier
    // `ax` to the third power, not `a` times `x^3`), and for a bare `#` it
    // reads as named-slot syntax (`#Sin` = `Slot["Sin"]`). A trailing plain
    // number is unambiguous either way (`2Product[…]` already reads back as
    // `2*Product[…]`, since digits can't extend into letters), so only a
    // trailing run that itself contains a letter is at risk.
    //
    // A nested `RowBox` juxtaposed this way onto a preceding factor is a
    // *self-contained* box-tree node — 2-D layout encodes grouping through
    // nesting itself, so the FrontEnd never needs a visible `(…)` around it
    // even when its own top level carries an assignment (a Demonstrations
    // idiom: two Module-body statements stacked on separate lines with a
    // missing `;` between them, e.g. `While[…] \n\n vecLEN = Length[…]`,
    // which the box tree still stores as `RowBox[{While[…]-box, …,
    // vecLEN=…-box}]` — the assignment box is its own node regardless).
    // Flattening that box to bare text and reparsing it loses the boundary
    // nesting gave it: implicit multiplication binds *tighter* than `=`, so
    // the reparsed text captures the wrong left-hand side (`Set[Times[
    // While[…], vecLEN], Length[…]]`, which fails outright — `Times` is
    // Protected) instead of `Times[While[…], Set[vecLEN, Length[…]]]`.
    // Parenthesizing a nested box's own assignment restores the original
    // grouping; the extra parens are always harmless for a factor that
    // didn't need them.
    // Never wrap a piece that carries its own bracket-depth-0 comma: it
    // is not a single juxtaposed statement but a whole `body, iterator`
    // (or similar) argument list — e.g. `Table`'s bracket content, whose
    // separating comma must stay outside any added parens. Parenthesizing
    // it would trap that comma inside a bare `(…)`, which is not valid
    // Wolfram syntax and fails to reparse.
    let piece = if part.trim().starts_with("RowBox[")
      && has_real_predecessor(&parts, i, newline_ends_statement)
      && has_top_level_assignment(&piece)
      && split_top_level_commas(&piece).len() == 1
    {
      // The piece may already carry its own statement-ending `;` (a
      // Demonstrations idiom: `var = value;` written as its own row,
      // stacked on the next statement with only blank lines — no visible
      // operator — between them). That `;` has to land *outside* the
      // added parens, not inside: `(var = value;)` is a one-statement
      // `CompoundExpression` whose trailing `;` makes its *own* value
      // `Null`, and that `Null` then becomes a real factor when this
      // parenthesized group is juxtaposed against whatever follows —
      // corrupting the *next* statement's own assignment the same way
      // this wrapping exists to prevent (`Null*nextVar = …` binds `=` to
      // the whole product, failing with `Set::write` on the Protected
      // `Times` head instead of ever assigning `nextVar`). Keeping the
      // `;` outside makes this factor its own complete, self-terminated
      // statement instead: `(var = value);`, unable to juxtapose with
      // anything after it at all.
      //
      // A trailing `;` is not always a statement terminator, though: it is
      // also the second character of the `;;` `Span` operator (`rng =
      // 2;;` is `Set[rng, Span[2, All]]`, not `Set[rng, 2]` followed by an
      // empty statement). Blindly stripping one `;` off `"rng=2;;"` would
      // leave `"rng=2;"` — a *different*, silently wrong expression
      // (`Set[rng, 2]`) once the remaining `;` is reinterpreted as its own
      // terminator. Counting the run of trailing `;` tells them apart: an
      // *even* run (0, 2, 4, …) is entirely `Span` syntax with no real
      // terminator to relocate, so the piece is wrapped whole; an *odd*
      // run has exactly one genuine terminator on top of a (possibly
      // empty) `Span`, so only that one is moved outside, leaving the rest
      // of the run — still even — attached to the `Span` it belongs to.
      let trailing_semicolons =
        piece.chars().rev().take_while(|&c| c == ';').count();
      if trailing_semicolons % 2 == 1 {
        let stripped = &piece[..piece.len() - 1];
        format!("({stripped});")
      } else {
        format!("({piece})")
      }
    } else if part.trim().starts_with("RowBox[")
      && piece
        .trim_end()
        .chars()
        .rev()
        .take_while(|&c| c == ';')
        .count()
        % 2
        == 1
      && parts[i + 1..]
        .iter()
        .find(|p| !is_soft_whitespace_box_token(p))
        .is_some_and(|next| {
          // The next token is an operator applied to this whole row.
          let t = next.trim().trim_matches('"');
          next.trim().starts_with('"')
            && t.starts_with([
              '/', '@', '+', '*', '^', '.', '~', '=', '>', '<', '|', '&', '?',
              ':',
            ])
        })
    {
      // A nested row that ends in a statement `;` and is followed by an
      // operator (`RowBox[{RowBox[{x = …, ";"}], "/.", rules}]`) is the
      // operand `(x = …;)`: the box tree groups it, flat text would not.
      format!("({piece})")
    } else if part.starts_with("RowBox[")
      && i + 1 < parts.len()
      && piece.ends_with('&')
      && !piece.ends_with("&&")
      && split_top_level_commas(&piece).len() == 1
    {
      // A nested row that ends in the postfix `&` is a complete pure
      // function. `&` binds looser than every infix operator, so spliced
      // into the parent row as bare text it would swallow whatever follows
      // (`RowBox[{RowBox[{f, "&"}], "/@", x}]` is `(f &) /@ x`, not
      // `f /@ x &`). A row carrying a top-level comma is a whole argument
      // list (`Select[list, f &]`) and must not be parenthesized.
      format!("({piece})")
    } else {
      piece
    };
    // A nested row that already ends in its own statement-ending `;`
    // followed by a sibling bare `";"` is two terminators — a statement
    // plus an empty one (`CompoundExpression[stmt, Null]`, then `;`) —
    // not the `;;` `Span` operator. Glued together as text the two would
    // read back as `Span`, so keep them apart.
    if piece == ";"
      && result.ends_with(';')
      && i > 0
      && parts[i - 1].trim().starts_with("RowBox[")
    {
      result.push(' ');
    }
    // A word split by styling part of it (`RowBox[{"Ke", StyleBox["t1",
    // FontColor -> …]}]`) is still one word to the FrontEnd: `Ket1`.
    if i > 0 && styling_splits_word(&parts[i - 1], part) {
      result.push_str(&piece);
      i += 1;
      continue;
    }
    push_juxtaposed(&mut result, &piece);
    i += 1;
  }
  result
}

/// Is `part` the quoted single character `"ch"` (a bare display token, not
/// part of a wider string of text)?
fn is_bare_char(part: &str, ch: char) -> bool {
  let mut chars = part.chars();
  matches!(
    (chars.next(), chars.next(), chars.next(), chars.next()),
    (Some('"'), Some(c), Some('"'), None) if c == ch
  )
}

/// Is `part` the quoted named character `"\[name]"` on its own?
fn is_bare_named_char(part: &str, name: &str) -> bool {
  part == format!("\"\\[{name}]\"")
}

/// The comma-separated arguments of a parenthesised call's box, if `s` (the
/// box between a function-application `"("` and `")"`) is a `RowBox` whose
/// own top-level elements are joined by at least one bare `","` — i.e. it
/// really is an argument list, not a single parenthesised expression.
fn paren_call_arglist(s: &str) -> Option<Vec<String>> {
  let inner = s.strip_prefix("RowBox[")?;
  let inner = &inner[..inner.len().saturating_sub(1)];
  let inner = inner.trim().strip_prefix('{')?.strip_suffix('}')?;
  let subparts = split_top_level_commas(inner);
  if !subparts.iter().any(|p| is_bare_char(p.trim(), ',')) {
    return None;
  }
  let mut args = Vec::new();
  let mut current = String::new();
  for p in &subparts {
    let p = p.trim();
    if is_bare_char(p, ',') {
      args.push(std::mem::take(&mut current));
    } else {
      current.push_str(&box_part_source(p));
    }
  }
  args.push(current);
  Some(args)
}

/// The Wolfram Language head a TraditionalForm lowercase function name
/// (`"exp"`, `"ln"`, …, as a bare quoted box string) stands for.
fn traditional_function_name(part: &str) -> Option<&'static str> {
  let name = part.strip_prefix('"')?.strip_suffix('"')?;
  Some(match name {
    "exp" => "Exp",
    "ln" => "Log",
    "sin" => "Sin",
    "cos" => "Cos",
    "tan" => "Tan",
    "cot" => "Cot",
    "sec" => "Sec",
    "csc" => "Csc",
    "sinh" => "Sinh",
    "cosh" => "Cosh",
    "tanh" => "Tanh",
    "coth" => "Coth",
    "sech" => "Sech",
    "csch" => "Csch",
    _ => return None,
  })
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
    } else if inner.starts_with("\\<") && inner.ends_with("\\>") {
      // `\<…\>` is the linear-syntax spelling of a string literal's
      // quotes (`"\<surface 1\>"` is the string `"surface 1"`), so the
      // element is a string literal just like `"\"…\""`.
      format!(
        "\"{}\"",
        escape_unescaped_quotes(&unescape_code_string(inner))
      )
    } else {
      unescape_code_string(inner)
    }
  } else if part.starts_with("RowBox[") {
    extract_rowbox_content(&part[7..part.len().saturating_sub(1)])
  } else if part.starts_with("Cell[") {
    // An inline cell embedded in a row (a display equation stored as
    // `RowBox[{Cell[…, "InlineMath"], ","}]`) stands for its typeset
    // content, not for its own `Cell[…]` source.
    render_text_element(part)
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
        .map(|c| render_inline_math_content(c.trim()))
        .unwrap_or_default(),
      Some("InlineCell" | "InlineCode" | "InlineInput" | "InlineOutput") => {
        parts
          .first()
          .map(|c| extract_cell_content(c.trim()))
          .unwrap_or_default()
      }
      // An unstyled wrapper around another cell (`Cell[BoxData[Cell[…]]]`)
      // or around typeset math (`Cell[BoxData[FormBox[…]]]`, as produced for
      // short inline formulas like `k = 3`) is just the FrontEnd's nesting,
      // not chrome: render what it holds.
      None => parts
        .first()
        .filter(|c| {
          let c = c.trim();
          c.strip_prefix("BoxData[").is_some_and(|r| {
            let r = r.trim_start();
            r.starts_with("Cell[") || r.starts_with("FormBox[")
          }) || c
            .strip_prefix("TextData[")
            .is_some_and(|r| r.trim_start().starts_with("Cell["))
        })
        .map(|c| render_inline_math_content(c.trim()))
        .unwrap_or_default(),
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

/// Render the content of an inline math cell. Usually that is plain box
/// source, but the FrontEnd also nests cells inside each other
/// (`Cell[BoxData[Cell[TextData[Cell[BoxData[FormBox[…]], "InlineMath"]],
/// "InlineMath"]], "InlineMath"]`) when a formula is pasted into a formula;
/// unwrap those layers so the innermost typeset content still renders.
fn render_inline_math_content(s: &str) -> String {
  let s = s.trim();
  for head in ["BoxData", "TextData"] {
    if let Some(rest) = s.strip_prefix(&format!("{head}["))
      && let Ok((inner, after)) = find_matching_bracket(rest)
      && after.trim().is_empty()
    {
      let inner = inner.trim();
      let nested = inner.starts_with("Cell[")
        || inner.starts_with("TextData[")
        || (head == "TextData" && inner.starts_with('{'));
      if nested {
        return if head == "TextData" {
          extract_textdata(inner)
        } else {
          render_inline_math_content(inner)
        };
      }
    }
  }
  if s.starts_with("Cell[") {
    return render_text_element(s);
  }
  render_boxes_text(s)
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

/// Convert one `\(…\)` linear-syntax group's inner text (delimiters already
/// stripped) into ordinary box source. A `\*` marks box code in its own
/// right (`\*SuperscriptBox[…]`), at the start of the group or part-way
/// through it (`-\*SuperscriptBox[…]`) — drop the marker and keep converting;
/// plain runs around it are display atoms, which box source spells as quoted
/// strings, and several pieces form a `RowBox`.
fn linear_syntax_group_to_box_source(inner: &str) -> String {
  let chars: Vec<char> = inner.chars().collect();
  // Split into plain-text runs and `\*`-introduced box-code runs. A code
  // run is a head name followed by its balanced `[…]` argument list, and
  // may sit anywhere in the group (`-\*SuperscriptBox[…]`, `(t/\*Sub…])`).
  let mut pieces: Vec<String> = Vec::new();
  let mut text = String::new();
  let flush_text = |text: &mut String, pieces: &mut Vec<String>| {
    if !text.is_empty() {
      pieces.push(format!(
        "\"{}\"",
        linear_syntax_code_to_box_source(text).replace('"', "\\\"")
      ));
      text.clear();
    }
  };
  let mut i = 0;
  while i < chars.len() {
    if chars[i] == '\\' && chars.get(i + 1) == Some(&'*') {
      flush_text(&mut text, &mut pieces);
      let mut j = i + 2;
      // Head name (or a bare `\(…\)` group right after the marker).
      while j < chars.len() && (chars[j].is_alphanumeric() || chars[j] == '$') {
        j += 1;
      }
      if chars.get(j) == Some(&'[') {
        let mut depth = 0;
        while j < chars.len() {
          match chars[j] {
            '[' => depth += 1,
            ']' => {
              depth -= 1;
              if depth == 0 {
                j += 1;
                break;
              }
            }
            _ => {}
          }
          j += 1;
        }
      } else if chars.get(j) == Some(&'\\') && chars.get(j + 1) == Some(&'(') {
        let mut depth = 0;
        while j < chars.len() {
          if chars[j] == '\\' && chars.get(j + 1) == Some(&'(') {
            depth += 1;
            j += 2;
          } else if chars[j] == '\\' && chars.get(j + 1) == Some(&')') {
            depth -= 1;
            j += 2;
            if depth == 0 {
              break;
            }
          } else {
            j += 1;
          }
        }
      }
      let code: String = chars[i + 2..j].iter().collect();
      pieces.push(linear_syntax_code_to_box_source(&code));
      i = j;
    } else if chars[i] == '\\' && chars.get(i + 1) == Some(&'(') {
      // A nested `\(…\)` group (`\(\(12\)\( \)\)`) is one box piece in its
      // own right; quoting it into a text run would print the markers.
      flush_text(&mut text, &mut pieces);
      let mut depth = 1;
      let mut j = i + 2;
      while j < chars.len() && depth > 0 {
        if chars[j] == '\\' && chars.get(j + 1) == Some(&'(') {
          depth += 1;
          j += 2;
        } else if chars[j] == '\\' && chars.get(j + 1) == Some(&')') {
          depth -= 1;
          j += 2;
        } else {
          j += 1;
        }
      }
      let end = if depth == 0 { j - 2 } else { j };
      let nested: String = chars[i + 2..end].iter().collect();
      pieces.push(linear_syntax_group_to_box_source(&nested));
      i = j;
    } else {
      text.push(chars[i]);
      i += 1;
    }
  }
  flush_text(&mut text, &mut pieces);
  match pieces.len() {
    0 => "\"\"".to_string(),
    1 => pieces.remove(0),
    _ => format!("RowBox[{{{}}}]", pieces.join(", ")),
  }
}

/// Replace every `\(…\)` linear-syntax group inside box code with its box
/// source spelling (see [`linear_syntax_group_to_box_source`]), leaving real
/// box syntax (`Head[`, `,`, `]`, bare digits, …) untouched.
fn linear_syntax_code_to_box_source(s: &str) -> String {
  let chars: Vec<char> = s.chars().collect();
  let mut out = String::new();
  let mut i = 0;
  while i < chars.len() {
    if chars[i] == '\\' && chars.get(i + 1) == Some(&'(') {
      let mut depth = 1;
      let mut j = i + 2;
      while j < chars.len() && depth > 0 {
        if chars[j] == '\\' && chars.get(j + 1) == Some(&'(') {
          depth += 1;
          j += 2;
        } else if chars[j] == '\\' && chars.get(j + 1) == Some(&')') {
          depth -= 1;
          j += 2;
        } else {
          j += 1;
        }
      }
      let inner: String = chars[i + 2..j.saturating_sub(2)].iter().collect();
      out.push_str(&linear_syntax_group_to_box_source(&inner));
      i = j;
    } else {
      out.push(chars[i]);
      i += 1;
    }
  }
  out
}

/// Render the FrontEnd's "linear syntax" box escapes — `\!\(\*SuperscriptBox[
/// \(X\), \(2\)]\)` — embedded in ordinary prose text as the same display
/// text a real box tree would produce, leaving surrounding plain characters
/// untouched. A computed `Row[…]` built from `ToString[…, StandardForm]`
/// pieces (a Wolfram Demonstrations template pattern) stores its typeset
/// arguments this way.
fn render_linear_syntax_escapes(s: &str) -> String {
  let chars: Vec<char> = s.chars().collect();
  let mut out = String::new();
  let mut i = 0;
  while i < chars.len() {
    if chars[i] == '\\'
      && chars.get(i + 1) == Some(&'!')
      && chars.get(i + 2) == Some(&'\\')
      && chars.get(i + 3) == Some(&'(')
    {
      let mut depth = 1;
      let mut j = i + 4;
      while j < chars.len() && depth > 0 {
        if chars[j] == '\\' && chars.get(j + 1) == Some(&'(') {
          depth += 1;
          j += 2;
        } else if chars[j] == '\\' && chars.get(j + 1) == Some(&')') {
          depth -= 1;
          j += 2;
        } else {
          j += 1;
        }
      }
      let inner: String = chars[i + 4..j.saturating_sub(2)].iter().collect();
      let box_source = linear_syntax_group_to_box_source(&inner);
      out.push_str(&render_boxes_text(&box_source));
      i = j;
    } else {
      out.push(chars[i]);
      i += 1;
    }
  }
  out
}

/// Parenthesize a sub/superscript that is more than one plain token so the
/// flat prose rendering stays unambiguous: an exponent `-b e^(-c t)` must
/// not read as `e^-b e^-c t`. Single words, numbers and lone symbols
/// (`*`, `†`) stay bare, and an already parenthesized script is kept as is.
fn group_script(script: &str) -> String {
  // Padding spaces inside a script (`SubscriptBox["a", "1 "]`) draw
  // nothing and must not force parentheses around a one-token script.
  let script = script.trim();
  let encloses_all = script.starts_with('(') && {
    let mut depth = 0usize;
    let mut closes_at_end = false;
    for (i, c) in script.char_indices() {
      match c {
        '(' => depth += 1,
        ')' => {
          depth -= 1;
          if depth == 0 {
            closes_at_end = i + 1 == script.len();
            break;
          }
        }
        _ => {}
      }
    }
    closes_at_end
  };
  if encloses_all
    || script.chars().count() <= 1
    || script.chars().all(|c| c.is_alphanumeric() || c == '.')
  {
    script.to_string()
  } else {
    format!("({script})")
  }
}

/// Render a box expression as *display* text for a prose (Text) cell —
/// the inline-formula counterpart of `extract_typeset_box`, preferring
/// readable notation over evaluable InputForm: `SubscriptBox["D", "U"]` →
/// `D_U`, `SuperscriptBox["V", "2"]` → `V²`, `FractionBox[a, b]` → `a/b`,
/// and `GridBox` rows on separate lines. Named characters resolve to
/// Unicode (`\[Del]` → `∇`), never to ASCII operator rewrites.
/// `0.2\`` / `1.5\`20` — a number token carrying a precision or accuracy
/// mark. The mark is input syntax only; the notebook shows just the number.
fn strip_number_precision_mark(text: &str) -> Option<String> {
  let (number, mark) = text.split_once('`')?;
  let is_number = !number.is_empty()
    && number.chars().next().is_some_and(|c| c.is_ascii_digit())
    && number.chars().all(|c| c.is_ascii_digit() || c == '.');
  let mark = mark.strip_prefix('`').unwrap_or(mark);
  let is_mark = mark.chars().all(|c| c.is_ascii_digit() || c == '.');
  (is_number && is_mark).then(|| number.to_string())
}

pub(crate) fn render_boxes_text(s: &str) -> String {
  let s = s.trim();

  // Plain string literal.
  if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
    let text =
      render_linear_syntax_escapes(&unescape_string(&s[1..s.len() - 1]));
    return strip_number_precision_mark(&text).unwrap_or(text);
  }

  // A bare `{…}` list of inline items.
  if s.starts_with('{') && s.ends_with('}') {
    let items: Vec<&str> = split_top_level_commas(&s[1..s.len() - 1])
      .into_iter()
      .map(str::trim)
      .collect();
    let mut out = String::new();
    let mut i = 0;
    while i < items.len() {
      // `( GridBox[{{n}, {k}}] )` is the stacked binomial coefficient. On
      // one line of prose the grid would break across lines, so read it
      // as `C(n, k)`.
      if i + 2 < items.len()
        && is_bare_char(items[i], '(')
        && is_bare_char(items[i + 2], ')')
        && let Some((top, bottom)) = stacked_pair_of_grid_box(items[i + 1])
      {
        out.push_str(&format!(
          "C({}, {})",
          render_boxes_text(&top),
          render_boxes_text(&bottom)
        ));
        i += 3;
        continue;
      }
      out.push_str(&render_boxes_text(items[i]));
      i += 1;
    }
    return out;
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
    return format!("{base}_{}", group_script(&sub));
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
      _ => format!("{base}^{}", group_script(&exp)),
    };
  }
  if let Some(args) = positional_box_args("SubsuperscriptBox", s)
    && args.len() == 3
  {
    let base = render_boxes_text(&args[0]);
    let sub = render_boxes_text(&args[1]);
    let exp = render_boxes_text(&args[2]);
    return match superscript_unicode(&exp) {
      Some(sup) if !sup.is_empty() => {
        format!("{base}_{}{sup}", group_script(&sub))
      }
      _ => format!("{base}_{}^{}", group_script(&sub), group_script(&exp)),
    };
  }
  if let Some(args) = positional_box_args("SqrtBox", s)
    && args.len() == 1
  {
    let radicand = render_boxes_text(&args[0]);
    // A multi-token radicand needs parentheses to stay unambiguous in
    // linear text: `√(a+b)`, but `√x`.
    return if radicand.chars().all(|c| c.is_alphanumeric() || c == '.') {
      format!("√{radicand}")
    } else {
      format!("√({radicand})")
    };
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
  // `TemplateBox[{…}, "RowDefault"]` (and its separator variants) is the box
  // form of a typeset `Row[…]` — the FrontEnd lays the parts out side by
  // side. Unlike `extract_typeset_box`'s handling of the same tag (which
  // rebuilds evaluable `Row[…]` source), prose wants the parts' *display*
  // text joined directly, so a computed `Row[…]` of typeset pieces (e.g. a
  // Wolfram Demonstrations caption built from `ToString[…, StandardForm]`
  // strings) reads the same as any other inline formula.
  if let Some(args) = positional_box_args("TemplateBox", s)
    && args.len() >= 2
  {
    let tag = args.last().unwrap().trim().trim_matches('"');
    if matches!(tag, "RowDefault" | "RowWithSeparator" | "RowWithSeparators") {
      // A string-typed slot is written doubly quoted (the FrontEnd's usual
      // convention for a `Row[…]` element that is itself a string, e.g.
      // `"\"2 \""` for the string `"2 "`) — `display_text` peels off both
      // layers. A slot holding a real box tree (`SuperscriptBox[…]`, …)
      // isn't a string at all, so it renders directly instead.
      let render_part = |p: &str| {
        let p = p.trim();
        match display_text(p) {
          Some(text) => render_linear_syntax_escapes(&text),
          None => render_boxes_text(p),
        }
      };
      let parts = template_box_parts(&args[0]);
      let mut iter = parts.iter();
      let separator = match tag {
        "RowWithSeparators" => {
          iter.next();
          iter.next()
        }
        "RowWithSeparator" => iter.next(),
        _ => None,
      };
      let sep_text = separator.map_or_else(String::new, |p| render_part(p));
      return iter
        .map(|p| render_part(p))
        .collect::<Vec<_>>()
        .join(&sep_text);
    }
  }

  // A `GraphicsBox[…]`/`Graphics3DBox[…]` embedded directly in prose — e.g.
  // a diagram inside a Demonstration's Details text (`Cell[TextData[
  // Cell[BoxData[FormBox[GraphicsBox[…], TraditionalForm]], "InlineMath"]],
  // "Text"]`) — is display-only, not code: falling back to the
  // evaluable-InputForm extractor would flood the paragraph with the
  // reconstructed `Graphics[…]` source (hundreds to thousands of characters
  // of box coordinates). Wolfram's own plain-text form of a graphic is
  // `-Graphics-` (`-Graphics3D-` for a 3-D scene, `-Image-` for a raster);
  // use the same placeholder here rather than rendering nothing, so the
  // surrounding sentence still reads.
  if let Some(args) = positional_box_args("GraphicsBox", s) {
    return if args
      .first()
      .is_some_and(|a| extract_image_from_boxes(a).is_some())
    {
      "-Image-".to_string()
    } else {
      "-Graphics-".to_string()
    };
  }
  if positional_box_args("Graphics3DBox", s).is_some() {
    return "-Graphics3D-".to_string();
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

/// The InputForm operator a `StyleBox[…, "OperatorCharacter"]`'s first
/// argument stands for, if that argument is *only* a bare named-character
/// escape (`"\[Equal]"`) and nothing else. A real string that merely
/// starts with the same text (`"\[Equal] sign"`, say) is not this — the
/// whole argument has to be exactly the one escape, quotes included.
fn stylebox_bare_operator(s: &str) -> Option<&'static str> {
  let s = s.trim();
  let inner = s.strip_prefix('"')?.strip_suffix('"')?;
  let name = inner.strip_prefix("\\[")?.strip_suffix(']')?;
  named_char_to_code_op(name)
}

/// Map Wolfram named operator characters to their InputForm ASCII
/// equivalents for display in code cells. The Wolfram FrontEnd encodes
/// operators like `->` as `\[Rule]` (private-use U+F522), which has no
/// glyph in normal fonts. Returning ASCII keeps the cell editable.
///
/// Display-only "invisible" characters — `\[NoBreak]`, `\[InvisibleSpace]`,
/// `\[InvisibleTimes]`, `\[ImplicitPlus]` — exist purely
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
    // `\[LeftAssociation]`/`\[RightAssociation]` are the typeset forms of
    // the association delimiters `<|` and `|>`.
    "LeftAssociation" => "<|",
    "RightAssociation" => "|>",
    "NotEqual" => "!=",
    "LessEqual" => "<=",
    "GreaterEqual" => ">=",
    "And" => "&&",
    "Or" => "||",
    "Cross" => "\\[Cross]",
    // `expr\[Transpose]`/`expr\[ConjugateTranspose]` are the FrontEnd's
    // postfix notation for `Transpose[expr]`/`ConjugateTranspose[expr]`.
    // Their private-use codepoints (U+F3C7, U+F3C9) have no glyph in
    // normal fonts and would render as blank/tofu boxes; the grammar
    // accepts the readable `\[Name]` escape as the same postfix operator.
    "Transpose" => "\\[Transpose]",
    "Conjugate" => "\\[Conjugate]",
    "ConjugateTranspose" => "\\[ConjugateTranspose]",
    "NoBreak"
    | "InvisibleSpace"
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
    // `\[InvisibleComma]` is a real comma that merely isn't drawn: it
    // separates the indices of `SubscriptBox["k", RowBox[{"2",
    // "\[InvisibleComma]", "1"}]]`, i.e. `Subscript[k, 2, 1]`.
    "InvisibleComma" => ",",
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
/// Decode escapes in saved *code* (a dumped variable value): segments
/// outside string literals are unescaped like Input-cell source, while a
/// string literal is kept exactly as written — its `\"` and `\\` are the
/// literal's own escapes, and decoding them would end the string early
/// (`"\"JH …\" - Q "` read back as `""*JH*…`).
fn unescape_code_outside_strings(s: &str) -> String {
  let mut out = String::with_capacity(s.len());
  let mut segment = String::new();
  let mut chars = s.chars().peekable();
  while let Some(c) = chars.next() {
    if c == '"' {
      out.push_str(&unescape_code_string(&segment));
      segment.clear();
      out.push('"');
      while let Some(d) = chars.next() {
        if d == '\\' {
          match chars.next() {
            // A named character inside the literal draws as the glyph the
            // cell's own source shows for it (`"air \[ShortRightArrow]
            // disk"` reads `"air → disk"` there). Left as the escape, it
            // would parse to Wolfram's private-use code point and never
            // equal the source's string, so a control restored from the
            // dump would match none of the body's `=== "air → disk"` tests.
            Some('[') => {
              let mut name = String::new();
              for ch in chars.by_ref() {
                if ch == ']' {
                  break;
                }
                name.push(ch);
              }
              match crate::syntax::named_char_to_unicode(&name) {
                Some(uni) => out
                  .push_str(&crate::syntax::substitute_private_use_glyphs(uni)),
                None => out.push_str(&format!("\\[{name}]")),
              }
            }
            Some(e) => {
              out.push(d);
              out.push(e);
            }
            None => out.push(d),
          }
        } else {
          out.push(d);
          if d == '"' {
            break;
          }
        }
      }
    } else {
      segment.push(c);
    }
  }
  out.push_str(&unescape_code_string(&segment));
  out
}

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
        Some('.') => {
          // `\.HH` is Wolfram's 2-hex-digit escape for a Latin-1 code
          // point (e.g. `\.aa` is "ª", FEMININE ORDINAL INDICATOR) — the
          // same family as `\:XXXX` below, just narrower. Authors' names
          // in a Demonstration's "Contributed By" section are a common
          // source: accented/special Latin-1 letters outside the
          // `\[Name]` table get written this way.
          let hex: String = chars.by_ref().take(2).collect();
          if let Ok(code) = u32::from_str_radix(&hex, 16)
            && let Some(ch) = char::from_u32(code)
          {
            // `\.00` (character 0) is an invisible spacer some authors
            // leave after a number (`RowBox[{"1", "\.00"}]`); other
            // control codes (`\.08`, …) turn up the same way. None of them
            // is valid in Wolfram source, so they must vanish rather than
            // break the cell's code.
            if !ch.is_control() || matches!(ch, '\t' | '\n' | '\r') {
              result.push(ch);
            }
          } else {
            result.push('\\');
            result.push('.');
            result.push_str(&hex);
          }
        }
        Some(':') => {
          // `\:XXXX` is Wolfram's ASCII-safe hex escape for an arbitrary
          // character. Box source uses it for characters that have no
          // `\[Name]`, e.g. the pair of invisible bracket markers a
          // FrontEnd wraps a `GridBox` matrix literal in
          // (`RowBox[{"(", "\:f3a2", GridBox[…], "\:f3a2", ")"}]`).
          // Substitute the same safe textual stand-in a `\[Name]` escape
          // would draw as: the raw private-use code point isn't valid
          // Wolfram Language source (unlike the handful of private-use
          // operators, e.g. Rule's U+F522, that the parser does accept
          // literally), so an unmapped invisible marker must become
          // nothing rather than get inserted into the cell's code.
          let hex: String = chars.by_ref().take(4).collect();
          if let Some(ch) =
            u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32)
          {
            result.push_str(&crate::syntax::substitute_private_use_glyphs(
              &ch.to_string(),
            ));
          } else {
            result.push('\\');
            result.push(':');
            result.push_str(&hex);
          }
        }
        Some('|') => {
          // `\|HHHHHH` is Wolfram's 6-hex-digit escape for a code point
          // outside the Basic Multilingual Plane — the same family as
          // `\:XXXX` above, just wide enough for e.g. an Egyptian
          // hieroglyph (Unicode Plane 1). A Demonstration's data literals
          // are a common source: characters with no `\[Name]` and no
          // 4-hex-digit representation get written this way.
          let hex: String = chars.by_ref().take(6).collect();
          if let Ok(code) = u32::from_str_radix(&hex, 16)
            && let Some(ch) = char::from_u32(code)
          {
            result.push(ch);
          } else {
            result.push('\\');
            result.push('|');
            result.push_str(&hex);
          }
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
        Some(' ') => {
          // Same linear-syntax family as `\<`/`\>` above: inside `\!\(…\)`
          // embedded box syntax a bare space separates tokens rather than
          // standing for itself, so the FrontEnd escapes an actual space
          // character as `\ ` (e.g. a Manipulate control label padding a
          // subscript's digits). Unescapes to one plain space.
          result.push(' ');
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

/// Give the symbols a saved Manipulate dump shares by bare name with a
/// control variable their own spelling. The FrontEnd writes the control `n`
/// as `` $CellContext`n$$ `` but a helper definition's `n` (a pattern
/// variable, a stored rule `p[n_, 1] -> n`) as `` $CellContext`n ``: two
/// distinct symbols. Stripping the `$$` suffix and the context would merge
/// them, so a rule stored by the Initialization would read the control's
/// current value wherever it is re-evaluated. The bare ones get a `$ctx`
/// suffix instead, which no stripping step removes.
fn separate_control_namesakes(dump: &str) -> String {
  const PREFIX: &str = "$CellContext`";
  let symbol_at = |from: usize| {
    let end = dump[from..]
      .find(|c: char| !(is_symbol_char(c) || c == '$'))
      .map_or(dump.len(), |i| from + i);
    &dump[from..end]
  };
  let mut controls: Vec<&str> = Vec::new();
  let mut at = 0;
  while let Some(rel) = dump[at..].find(PREFIX) {
    let start = at + rel + PREFIX.len();
    let name = symbol_at(start);
    if let Some(base) = name.strip_suffix("$$") {
      controls.push(base);
    }
    at = start + name.len().max(1);
  }
  let mut out = String::with_capacity(dump.len());
  let mut at = 0;
  while let Some(rel) = dump[at..].find(PREFIX) {
    let start = at + rel + PREFIX.len();
    let name = symbol_at(start);
    out.push_str(&dump[at..start]);
    out.push_str(name);
    if !name.ends_with("$$") && controls.contains(&name) {
      out.push_str("$ctx");
    }
    at = start + name.len();
  }
  out.push_str(&dump[at..]);
  out
}

/// Drop the contexts a saved Manipulate dump qualifies its symbols with:
/// `` $CellContext` `` (the DynamicModule's private context) and
/// `` DemonstrationsTools` `` (where the Demonstrations authoring notebooks
/// create their control variables). The definitions in the same dump use the
/// bare names, so keeping the prefix would make `` DemonstrationsTools`x ``
/// a different symbol from the `x` those definitions refer to.
fn strip_saved_contexts(s: &str) -> String {
  strip_numbered_cell_contexts(&s.replace("$CellContext`", ""))
    .replace("DemonstrationsTools`", "")
}

/// Drop the `` Cell$$2138` `` style qualifiers a notebook saved with
/// `CellContext -> Cell` writes on a cell's symbols: the same cell's
/// definitions also appear with bare names, so, like `` $CellContext` ``,
/// the qualifier has to go for the two spellings to be one symbol.
fn strip_numbered_cell_contexts(s: &str) -> String {
  const PREFIX: &str = "Cell$$";
  let mut out = String::with_capacity(s.len());
  let mut rest = s;
  while let Some(at) = rest.find(PREFIX) {
    let after = &rest[at + PREFIX.len()..];
    let digits = after.bytes().take_while(u8::is_ascii_digit).count();
    // Only `Cell$$<digits>`` as a whole symbol prefix, not inside a name
    let preceded_by_name_char = rest[..at]
      .chars()
      .next_back()
      .is_some_and(|c| c.is_alphanumeric() || c == '$' || c == '`');
    if digits > 0 && after[digits..].starts_with('`') && !preceded_by_name_char
    {
      out.push_str(&rest[..at]);
      rest = &after[digits + 1..];
    } else {
      out.push_str(&rest[..at + PREFIX.len()]);
      rest = after;
    }
  }
  out.push_str(rest);
  out
}

/// Drop the `` $CellContext` `` qualifier from the source of an Input cell.
/// A `Manipulate[…, SaveDefinitions -> True]` saved by the FrontEnd carries
/// an `Initialization` whose helper definitions spell the symbols its
/// control variables share with the body as `` $CellContext`name `` while
/// the pattern heads of the same definition stay bare, so left alone the
/// body would refer to symbols the heads never bind.
pub fn strip_cell_context_prefix(code: &str) -> String {
  strip_numbered_cell_contexts(&code.replace("$CellContext`", ""))
}

/// Net open-bracket depth of Wolfram source: `(`, `[` and `{` open, their
/// closers close; string literals and `(* … *)` comments are skipped.
fn open_bracket_depth(code: &str) -> i32 {
  let chars: Vec<char> = code.chars().collect();
  let mut depth = 0;
  let mut comment_depth = 0;
  let mut in_string = false;
  let mut i = 0;
  while i < chars.len() {
    let c = chars[i];
    let next = chars.get(i + 1).copied();
    if in_string {
      match c {
        '\\' => i += 1,
        '"' => in_string = false,
        _ => {}
      }
    } else if comment_depth > 0 {
      if c == '*' && next == Some(')') {
        comment_depth -= 1;
        i += 1;
      } else if c == '(' && next == Some('*') {
        comment_depth += 1;
        i += 1;
      }
    } else if c == '(' && next == Some('*') {
      comment_depth += 1;
      i += 1;
    } else {
      match c {
        '"' => in_string = true,
        '(' | '[' | '{' => depth += 1,
        ')' | ']' | '}' => depth -= 1,
        _ => {}
      }
    }
    i += 1;
  }
  depth
}

/// Join notebook input cells whose source is left open (unclosed brackets)
/// with the cells that follow, until the expression is complete. The
/// FrontEnd continues an incomplete input across cells, so authors can
/// split one long definition over several cells; evaluating each cell on
/// its own would fail to parse every part.
pub fn join_continuation_cells<S: AsRef<str>>(cells: &[S]) -> Vec<String> {
  let mut joined = Vec::new();
  let mut pending: Option<String> = None;
  for cell in cells {
    let cell = cell.as_ref();
    let combined = match pending.take() {
      Some(prev) => format!("{prev}\n{cell}"),
      None => cell.to_string(),
    };
    if open_bracket_depth(&combined) > 0 {
      pending = Some(combined);
    } else {
      joined.push(combined);
    }
  }
  // A trailing cell that never closes is kept as-is so the caller reports
  // the syntax error.
  joined.extend(pending);
  joined
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
/// markers (a `\` at end of line in box text) are removed, and so is the
/// `$$` suffix of the DynamicModule's control variables.
pub fn extract_saved_initialization(box_dump: &str) -> Option<String> {
  let box_dump = &separate_control_namesakes(box_dump);
  let mut search_from = 0;
  while let Some(rel) = box_dump[search_from..].find("Initialization") {
    let after_kw = search_from + rel + "Initialization".len();
    let rest = box_dump[after_kw..].trim_start();
    if let Some(rest) = rest.strip_prefix(":>") {
      let rest = rest.trim_start();
      if let Some(body) = rest.strip_prefix('(')
        && let Some(inner) = matching_paren_prefix(body)
      {
        // The control variables lose their `$$` suffix here exactly as in
        // the reconstructed body and specifications, or a delayed helper
        // (`take := Take[data, m$$]`) would read a symbol no control sets.
        let cleaned = strip_uniquification_suffix(&strip_saved_contexts(
          &inner.replace("\\\n", ""),
        ));
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
          .find(|c: char| !(is_symbol_char(c) || c == '$'))
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

/// The value of `"<key>" :> <value>` inside a FrontEnd box dump, up to the
/// first top-level comma (or the end of the enclosing bracket/brace/paren).
/// Skips over string literals, where brackets are just text.
fn extract_arrow_value<'a>(box_dump: &'a str, key: &str) -> Option<&'a str> {
  let marker = format!("\"{key}\"");
  let rel = box_dump.find(&marker)?;
  let rest = box_dump[rel + marker.len()..].trim_start();
  let rest = rest.strip_prefix(":>")?.trim_start();

  let mut depth = 0i32;
  let mut in_string = false;
  let mut prev_backslash = false;
  for (i, c) in rest.char_indices() {
    if in_string {
      if c == '"' && !prev_backslash {
        in_string = false;
      }
      prev_backslash = c == '\\' && !prev_backslash;
      continue;
    }
    match c {
      '"' => in_string = true,
      '(' | '[' | '{' => depth += 1,
      ')' | ']' | '}' => {
        if depth == 0 {
          return Some(rest[..i].trim_end());
        }
        depth -= 1;
      }
      ',' if depth == 0 => return Some(rest[..i].trim_end()),
      _ => {}
    }
  }
  let trimmed = rest.trim_end();
  if trimmed.is_empty() {
    None
  } else {
    Some(trimmed)
  }
}

/// Can `c` appear inside a symbol name? Besides Unicode letters and digits
/// this covers the private-use-area letters that a decoded `\[ScriptM]`
/// (and the rest of Wolfram's script / Gothic / double-struck alphabets)
/// turns into.
fn is_symbol_char(c: char) -> bool {
  c.is_alphanumeric() || crate::syntax::is_symbol_letter(c)
}

/// Whether `prefix` ends in a complete `\[Name]` named-character escape
/// (e.g. the `\[Alpha]` of a variable `\[Alpha]$$`).
fn closes_named_character(prefix: &[char]) -> bool {
  if prefix.last() != Some(&']') {
    return false;
  }
  let mut j = prefix.len() - 1;
  while j > 0 && prefix[j - 1].is_alphanumeric() {
    j -= 1;
  }
  j >= 2 && prefix[j - 1] == '[' && prefix[j - 2] == '\\'
}

/// Drop the DynamicModule's `$$` uniquification suffix from every symbol in
/// `src` (`` p1$$ `` → `` p1 ``), so a control variable reads — and matches
/// the plain names [`extract_saved_manipulate_variables`] reports — as the
/// author wrote it. Only a `$$` that ends a symbol is removed; a symbol
/// like `` x$1$$ `` keeps its inner `$`.
fn strip_uniquification_suffix(src: &str) -> String {
  let chars: Vec<char> = src.chars().collect();
  let mut out = String::with_capacity(src.len());
  let mut i = 0;
  while i < chars.len() {
    let ends_symbol = chars[i] == '$'
      && chars.get(i + 1) == Some(&'$')
      && i > 0
      && (is_symbol_char(chars[i - 1])
        || chars[i - 1] == '$'
        || closes_named_character(&chars[..i]))
      && !chars
        .get(i + 2)
        .is_some_and(|c| is_symbol_char(*c) || *c == '$' || *c == '`');
    if ends_symbol {
      i += 2;
    } else {
      out.push(chars[i]);
      i += 1;
    }
  }
  out
}

/// Rebuild a plain `Manipulate[body, spec, …]` source expression from a
/// saved FrontEnd dynamic-widget dump (the `DynamicModuleBox[…]` text
/// stored in the Output cell of an evaluated `Manipulate[…]`) when there is
/// no Input cell holding the original source to re-evaluate — e.g. a
/// notebook downloaded straight from a Wolfram Demonstrations Project share
/// link, which carries only the compiled `Manipulate\`ManipulateBoxes[…]`
/// widget and no editable Wolfram Language source.
///
/// The dump's `"Body" :> …` and `"Specifications" :> {…}` entries carry
/// exactly the two pieces a `Manipulate[…]` call needs; `` $CellContext` ``
/// prefixes and the FrontEnd's line-continuation `\` markers are stripped
/// the same way [`extract_saved_initialization`] strips them, so the result
/// evaluates as ordinary session-level input.
pub fn reconstruct_manipulate_from_box_dump(box_dump: &str) -> Option<String> {
  let box_dump = &separate_control_namesakes(box_dump);
  let clean = |s: &str| {
    strip_uniquification_suffix(&strip_saved_contexts(s).replace("\\\n", ""))
  };
  let body = clean(extract_arrow_value(box_dump, "Body")?);
  let specs = clean(extract_arrow_value(box_dump, "Specifications")?);
  let specs = specs.trim();
  let specs_inner = specs.strip_prefix('{')?.strip_suffix('}')?.trim();
  // The compiled `"Options" :> {…}` clause carries the widget-level caption
  // (`FrameLabel -> …`) shown with the output; it is the only option the
  // reconstruction keeps.
  let frame_label = extract_arrow_value(box_dump, "Options")
    .map(clean)
    .and_then(|opts| {
      let inner = opts
        .trim()
        .strip_prefix('{')?
        .strip_suffix('}')?
        .to_string();
      split_top_level_commas(&inner)
        .into_iter()
        .map(|part| part.trim().to_string())
        .find(|part| {
          part
            .strip_prefix("FrameLabel")
            .is_some_and(|r| r.trim_start().starts_with("->"))
        })
    });
  let mut args = vec![body];
  if !specs_inner.is_empty() {
    args.push(specs_inner.to_string());
  }
  args.extend(frame_label);
  Some(format!("Manipulate[{}]", args.join(", ")))
}

/// The live session values a saved FrontEnd dynamic-widget dump's
/// `"Variables" :> { $CellContext\`var$$ = value, … }` clause records, as
/// `(name, value)` pairs with the `` $CellContext` `` prefix and the
/// DynamicModule's own `$$` uniquification suffix stripped from each name —
/// so `"ctrl"`, not `` $CellContext`ctrl$$ `` — and any `\[Name]` character
/// escape decoded to Unicode in both name and value, matching how a
/// notebook's own Input-cell source is decoded, so the name is ready to
/// match a control by its plain source-level variable name.
///
/// A notebook saved from the desktop FrontEnd keeps both the original
/// `Manipulate[…]` source (in its Input cell) and this dump (in the
/// following Output cell) — and the two can disagree: the source's own
/// spec still carries whatever default the author *wrote*, while this dump
/// carries whatever the widget's sliders were actually sitting at the
/// moment the file was last saved. Wolfram's FrontEnd opens the notebook
/// showing that saved state, not the source's default, so a caller
/// rebuilding the widget from the Input cell (see
/// `instantiate_stored_manipulate` in woxi-studio) needs this to match.
///
/// The true live state is the outer `DynamicModuleBox[{ $CellContext\`var$$
/// = value, … }, …]` initializer list, so that is tried first. The inner
/// compiled `Manipulate\`ManipulateBoxes[…]`'s own `"Variables" :> {…}`
/// clause is only a fallback for a dump with no `DynamicModuleBox` wrapper
/// (e.g. one reconstructed for `reconstruct_manipulate_from_box_dump`):
/// that clause records whatever the controls held when the box was last
/// *compiled*, not necessarily the live value — a slider moved afterward
/// updates the outer list without recompiling the inner boxes, so the two
/// can disagree (regression: a Demonstration whose saved widget had a
/// slider left off its default reopened at the default instead).
pub fn extract_saved_manipulate_variables(
  box_dump: &str,
) -> Vec<(String, String)> {
  let from_module = extract_dynamic_module_variables(box_dump);
  if !from_module.is_empty() {
    return resolve_saved_variable_references(&from_module);
  }
  let Some(raw) = extract_arrow_value(box_dump, "Variables") else {
    return Vec::new();
  };
  let Some(inner) = raw
    .trim()
    .strip_prefix('{')
    .and_then(|s| s.strip_suffix('}'))
  else {
    return Vec::new();
  };
  let pairs: Vec<(String, String)> = split_top_level_commas(inner)
    .into_iter()
    .filter_map(|part| {
      let part = part.trim();
      let eq = part.find('=')?;
      let (lhs, rhs) = (part[..eq].trim(), part[eq + 1..].trim());
      let name = strip_saved_contexts(lhs);
      let name = name.strip_suffix("$$").unwrap_or(&name);
      let name = unescape_code_string(name);
      if name.is_empty() || rhs.is_empty() {
        return None;
      }
      // A saved default that is itself a bare symbol (e.g. a SetterBar
      // choice like `abc`, as opposed to a string or number) is dumped
      // context-qualified as `$CellContext`abc` — strip it here too, the
      // same as every other occurrence of the prefix, so the restored
      // value matches the plain symbol the live source's spec defines
      // (`scale1[abc]`), not a distinct `$CellContext`abc` that resolves
      // to nothing.
      let value = collapse_dump_line_breaks(&strip_saved_contexts(rhs));
      Some((name, unescape_code_outside_strings(&value)))
    })
    .collect();
  resolve_saved_variable_references(&pairs)
}

/// Replace a saved value that is just another saved variable's `name$$`
/// (a control initialised from its default, e.g. `` $CellContext`n$$ =
/// $CellContext`nDef$$ ``) by that variable's own value, so the control
/// opens on the default instead of an unbound `nDef$$` symbol. Chains are
/// followed; a reference to an unknown or cyclic variable is left as is.
fn resolve_saved_variable_references(
  pairs: &[(String, String)],
) -> Vec<(String, String)> {
  let lookup = |name: &str| {
    pairs
      .iter()
      .find(|(n, _)| n == name)
      .map(|(_, v)| v.clone())
  };
  pairs
    .iter()
    .map(|(name, value)| {
      let mut value = value.clone();
      for _ in 0..pairs.len() {
        let next = substitute_saved_references(&value, &lookup);
        if next == value {
          break;
        }
        value = next;
      }
      (name.clone(), value)
    })
    .collect()
}

/// Replace every `name$$` token in `value` whose `name` the saved variable
/// list defines with that variable's value — bare when the whole value is
/// the reference (`n$$ = nDef$$`), parenthesized when it sits inside a
/// larger expression (`t$$ = 0.75/f$$`) so operator precedence survives.
/// Tokens naming no saved variable are left untouched, and a reference
/// inside a holding head (`Dynamic[… on$$ = False …]`) is not a value but
/// the variable itself, so it becomes the plain name (`on`).
fn substitute_saved_references(
  value: &str,
  lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
  const HOLDING: [&str; 7] = [
    "Dynamic",
    "Hold",
    "HoldForm",
    "HoldComplete",
    "Unevaluated",
    "Function",
    "RuleDelayed",
  ];
  let chars: Vec<char> = value.chars().collect();
  let is_word = |c: char| c.is_alphanumeric();
  let mut out = String::new();
  // The heads of the brackets open at this point, and the word just read.
  let mut heads: Vec<String> = Vec::new();
  let mut last_word = String::new();
  let mut i = 0;
  while i < chars.len() {
    if chars[i] == '"' {
      // Copy a string literal through untouched.
      out.push('"');
      i += 1;
      while i < chars.len() && chars[i] != '"' {
        if chars[i] == '\\' && i + 1 < chars.len() {
          out.push(chars[i]);
          i += 1;
        }
        out.push(chars[i]);
        i += 1;
      }
      if i < chars.len() {
        out.push('"');
        i += 1;
      }
      last_word.clear();
      continue;
    }
    if !is_word(chars[i]) || (i > 0 && is_word(chars[i - 1])) {
      match chars[i] {
        '[' => heads.push(std::mem::take(&mut last_word)),
        ']' => {
          heads.pop();
        }
        c if !c.is_whitespace() => last_word.clear(),
        _ => {}
      }
      out.push(chars[i]);
      i += 1;
      continue;
    }
    let start = i;
    while i < chars.len() && is_word(chars[i]) {
      i += 1;
    }
    let word: String = chars[start..i].iter().collect();
    last_word.clone_from(&word);
    let has_suffix =
      chars.get(i) == Some(&'$') && chars.get(i + 1) == Some(&'$');
    let held = heads.iter().any(|h| HOLDING.contains(&h.as_str()));
    match has_suffix.then(|| lookup(&word)).flatten() {
      Some(_) if held => {
        i += 2;
        out.push_str(&word);
      }
      Some(resolved) => {
        i += 2;
        let whole = start == 0 && i == chars.len();
        let atomic = resolved
          .chars()
          .all(|c| c.is_alphanumeric() || c == '.' || c == '_');
        if whole || atomic {
          out.push_str(&resolved);
        } else {
          out.push_str(&format!("({resolved})"));
        }
      }
      None => out.push_str(&word),
    }
  }
  out
}

/// The `(name, value)` pairs assigned directly in a `DynamicModuleBox[{…},
/// …]` dump's own leading variable-initializer list — the DynamicModule's
/// live state, as opposed to the inner compiled box structure's stale
/// snapshot (see `extract_saved_manipulate_variables`). Skips `Typeset\`…`
/// bookkeeping entries (no `$CellContext\`` prefix) and the linking/shadow
/// variables a `ControllerVariables` control pairs with a real one (e.g.
/// `` $CellContext`Gas$79129$$ ``): those keep a digit run between two
/// single `$`s, so stripping the trailing `` $$ `` leaves a residual `$` in
/// the name, which is the filter.
fn extract_dynamic_module_variables(box_dump: &str) -> Vec<(String, String)> {
  let Some(after) = box_dump
    .find("DynamicModuleBox[")
    .map(|i| &box_dump[i + "DynamicModuleBox[".len()..])
  else {
    return Vec::new();
  };
  let Some(after_brace) = after.trim_start().strip_prefix('{') else {
    return Vec::new();
  };
  let Ok((inner, _rest)) = find_matching_brace(after_brace) else {
    return Vec::new();
  };
  split_top_level_commas(inner)
    .into_iter()
    .filter_map(|part| {
      let part = part.trim();
      let eq = part.find('=')?;
      let (lhs, rhs) = (part[..eq].trim(), part[eq + 1..].trim());
      let name = lhs.strip_prefix("$CellContext`")?;
      let name = name.strip_suffix("$$")?;
      if name.is_empty() || name.contains('$') || rhs.is_empty() {
        return None;
      }
      // See the matching strip in `extract_saved_manipulate_variables`: a
      // bare-symbol default value is context-qualified too, not just the
      // variable name.
      let value = collapse_dump_line_breaks(&strip_saved_contexts(rhs));
      Some((
        unescape_code_string(name),
        unescape_code_outside_strings(&value),
      ))
    })
    .collect()
}

/// The front end wraps long box dumps at arbitrary spaces
/// (`Rational[23, 90] \n    Pi`). A line break inside a saved value is
/// just that wrapping — read as a statement separator it would truncate the
/// value to its last line — so fold it back into a single space (implicit
/// multiplication). Values holding string literals are left untouched.
fn collapse_dump_line_breaks(value: &str) -> String {
  if value.contains('"') || !value.contains('\n') {
    return value.to_string();
  }
  value.split_whitespace().collect::<Vec<_>>().join(" ")
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

  // Brackets inside a comment (`(* If[a … *)`) do not count.
  let mut comment_depth = 0usize;
  let bytes = s.as_bytes();
  for (i, c) in s.char_indices() {
    if in_string {
      if c == '"' && !prev_backslash {
        in_string = false;
      }
      prev_backslash = c == '\\' && !prev_backslash;
      continue;
    }
    if c == '(' && bytes.get(i + 1) == Some(&b'*') {
      comment_depth += 1;
      continue;
    }
    if comment_depth > 0 {
      if c == ')' && i > 0 && bytes[i - 1] == b'*' {
        comment_depth -= 1;
      }
      continue;
    }

    match c {
      '"' => in_string = true,
      '{' | '[' | '(' => depth += 1,
      '}' | ']' | ')' => depth -= 1,
      // An association's `<|` … `|>` nests like a bracket pair.
      '<' if bytes.get(i + 1) == Some(&b'|') => depth += 1,
      '|' if bytes.get(i + 1) == Some(&b'>') => depth -= 1,
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

/// Decode a bare packed array of bytes: `!boR` + `b` + rank + dims + raw
/// `UnsignedInteger8` samples, with no `RawArray["UnsignedInteger8", …]`
/// function-call wrapper. This is the shape a Demonstration's
/// Initialization Code cell embeds for an inline bitmap assigned straight
/// to a symbol (`icon = Graphics[RasterBox[CompressedData["…"], …]]`), as
/// opposed to `parse_raw_array_u8`'s wrapped shape for a stored Output
/// snapshot. Demands the samples account for every remaining byte, so a
/// payload that merely starts the same way as this format is rejected
/// instead of silently truncated.
fn parse_packed_byte_array(raw: &[u8]) -> Option<(u32, u32, u8, &[u8])> {
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
  if pos + expected != raw.len() {
    return None;
  }
  Some((height, width, channels, pixels))
}

/// Decode a `CompressedData["1:…"]` payload as pixel data — either the
/// `RawArray["UnsignedInteger8", …]` shape `parse_raw_array_u8` reads or
/// the bare packed-byte-array shape `parse_packed_byte_array` reads — and
/// format it as the nested Wolfram Language list literal `Raster`/
/// `RasterBox` expects: one list of rows, each a list of per-pixel
/// grayscale values or `{r, g, b}`/`{r, g, b, a}` lists. Returns `None`
/// when the payload decodes to something else (e.g. the packed-real-array
/// coordinate format `decode_compressed_real_array` reads instead).
fn decode_compressed_raster_as_wl_list(payload: &str) -> Option<String> {
  let b64: String = payload.chars().filter(|c| !c.is_whitespace()).collect();
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

  let (height, width, channels, pixels) =
    parse_raw_array_u8(&raw).or_else(|| parse_packed_byte_array(&raw))?;
  Some(packed_raster_array_to_wl_list(
    height, width, channels, pixels,
  ))
}

/// Format a decoded pixel array (see `parse_raw_array_u8`) as the nested
/// Wolfram Language list literal it represents.
fn packed_raster_array_to_wl_list(
  height: u32,
  width: u32,
  channels: u8,
  pixels: &[u8],
) -> String {
  let channels = channels as usize;
  let width = width as usize;
  let mut rows = Vec::with_capacity(height as usize);
  for r in 0..height as usize {
    let mut cols = Vec::with_capacity(width);
    for c in 0..width {
      let base = (r * width + c) * channels;
      let px = &pixels[base..base + channels];
      if channels == 1 {
        cols.push(px[0].to_string());
      } else {
        let joined =
          px.iter().map(u8::to_string).collect::<Vec<_>>().join(", ");
        cols.push(format!("{{{joined}}}"));
      }
    }
    rows.push(format!("{{{}}}", cols.join(", ")));
  }
  format!("{{{}}}", rows.join(", "))
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

/// Decode Mathematica's serialized packed array of machine reals found
/// inside a `CompressedData["1:…"]` literal — the format the FrontEnd uses
/// for the point/curve coordinates of a Demonstrations snapshot graphic
/// (as opposed to `RawArray["UnsignedInteger8", …]`, the pixel format
/// `parse_raw_array_u8` decodes). `payload` is the text between the
/// quotes. Returns the array's dimensions together with its values in
/// row-major order.
fn decode_compressed_real_array(
  payload: &str,
) -> Option<(Vec<usize>, Vec<f64>)> {
  let b64: String = payload.chars().filter(|c| !c.is_whitespace()).collect();
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
  // A packed array of machine reals is a single tag byte (`e`), the rank,
  // that many dimensions, then the raw `f64` data — no type-name string,
  // unlike the `RawArray["UnsignedInteger8", …]` pixel format above.
  if take(&mut pos, 1)? != b"e" {
    return None;
  }
  let rank = read_u32(&mut pos)? as usize;
  if rank == 0 || rank > 4 {
    return None;
  }
  let mut dims = Vec::with_capacity(rank);
  let mut total = 1usize;
  for _ in 0..rank {
    let d = read_u32(&mut pos)? as usize;
    total = total.checked_mul(d)?;
    dims.push(d);
  }
  let bytes = take(&mut pos, total * 8)?;
  let values = bytes
    .as_chunks::<8>()
    .0
    .iter()
    .map(|c| f64::from_le_bytes(*c))
    .collect();
  Some((dims, values))
}

/// Format a decoded packed real array (see `decode_compressed_real_array`)
/// as the nested Wolfram Language list literal it represents, e.g. `[25,
/// 2]` values format as `{{x1, y1}, {x2, y2}, …}`.
fn packed_real_array_to_wl_list(dims: &[usize], values: &[f64]) -> String {
  fn build(dims: &[usize], values: &[f64], offset: &mut usize) -> String {
    let Some((&first, rest)) = dims.split_first() else {
      let v = values[*offset];
      *offset += 1;
      return format!("{v}");
    };
    let items: Vec<String> =
      (0..first).map(|_| build(rest, values, offset)).collect();
    format!("{{{}}}", items.join(", "))
  }
  let mut offset = 0;
  build(dims, values, &mut offset)
}

/// Rewrite a `.nb` box dump's vector-drawing heads (`PointBox`, `LineBox`,
/// `DiskBox`, `RectangleBox`, `FilledCurveBox`, `GraphicsBox`, …) back to
/// the `Graphics` primitives they typeset (`Point`, `Line`, `Disk`,
/// `Rectangle`, `Polygon`, `Graphics`), and inlines any `CompressedData[…]`
/// coordinate blob as a literal list, so the result parses and evaluates
/// as an ordinary Wolfram Language expression. Display-only wrappers
/// (`TagBox`, a `StyleBox`'s box-metadata options like `StripOnInput`) are
/// dropped; the directives a `StyleBox` carries (`RGBColor`, `Thickness`,
/// …) are kept in front of the primitives they apply to, the way
/// `Graphics` itself represents a styled sublist. Anything not recognized
/// (already-valid syntax like `RGBColor[…]`, bare numbers, option rules)
/// is recursed into unchanged, in case it nests a `CompressedData` blob.
fn box_source_to_graphics_expr(s: &str) -> String {
  let s = s.trim();

  if let Some(rest) = s.strip_prefix("CompressedData[\"") {
    let Some(end) = rest.find('"') else {
      return "Null".to_string();
    };
    let payload = &rest[..end];
    // A `RasterBox`'s CompressedData carries pixel bytes
    // (`RawArray["UnsignedInteger8", …]`), not the packed-real-array
    // format `decode_compressed_real_array` reads; try that shape first
    // so an Initialization Code cell that assigns a bitmap to a symbol
    // (a common Demonstrations pattern for custom icons) reconstructs a
    // real pixel grid instead of losing the picture to `Null`.
    if let Some(list) = decode_compressed_raster_as_wl_list(payload) {
      return list;
    }
    return decode_compressed_real_array(payload).map_or_else(
      || "Null".to_string(),
      |(dims, values)| packed_real_array_to_wl_list(&dims, &values),
    );
  }

  if let Some(inner) = s.strip_prefix('{').and_then(|r| r.strip_suffix('}')) {
    let items: Vec<String> = split_top_level_commas(inner)
      .into_iter()
      .map(|item| box_source_to_graphics_expr(item.trim()))
      .collect();
    return format!("{{{}}}", items.join(", "));
  }

  let box_call = |head: &str| -> Option<Vec<String>> {
    let rest = s.strip_prefix(head)?.strip_prefix('[')?;
    let (inner, _) = find_matching_bracket(rest).ok()?;
    Some(
      split_top_level_commas(inner)
        .into_iter()
        .map(|a| a.trim().to_string())
        .collect(),
    )
  };

  if let Some(args) = box_call("StyleBox")
    && !args.is_empty()
  {
    let body = box_source_to_graphics_expr(&args[0]);
    let dirs: Vec<String> = args[1..]
      .iter()
      .filter(|a| !is_option_arg(a))
      .map(|a| box_source_to_graphics_expr(a))
      .collect();
    return if dirs.is_empty() {
      body
    } else {
      format!("{{{}, {}}}", dirs.join(", "), body)
    };
  }

  if let Some(args) = box_call("TagBox")
    && let Some(first) = args.first()
  {
    return box_source_to_graphics_expr(first);
  }

  if let Some(args) = box_call("GraphicsBox")
    && let Some(first) = args.first()
  {
    let body = box_source_to_graphics_expr(first);
    let rest_args: Vec<String> = args[1..]
      .iter()
      .map(|a| box_source_to_graphics_expr(a))
      .collect();
    return if rest_args.is_empty() {
      format!("Graphics[{body}]")
    } else {
      format!("Graphics[{body}, {}]", rest_args.join(", "))
    };
  }

  if let Some(args) = box_call("FilledCurveBox")
    && args.len() >= 2
  {
    // The first argument's tags describe how the point run is split into
    // line/Bezier segments; approximating every segment as a straight
    // line keeps this a general decoder rather than one tuned to a
    // specific curve, at the cost of corner-cutting through Bezier
    // control points instead of curving smoothly between them. The
    // points themselves — the second argument — already trace the
    // boundary in order, so a straight-edged `Polygon` through them is a
    // reasonable approximation of the filled curve.
    let points = box_source_to_graphics_expr(&args[1]);
    return format!("Polygon[{points}]");
  }

  // `JoinedCurveBox` is the unfilled counterpart of `FilledCurveBox`
  // (an open or closed multi-segment curve, e.g. a Demonstration's
  // toolbar/hint glyph icon): same `{tags}, {points}, CurveClosed -> …`
  // shape, but stroked rather than filled. Reuse the same
  // straight-line approximation and rename to the evaluable `JoinedCurve`
  // primitive `graphics_ast` already understands, wrapping the point run
  // in `Line[…]` the way `JoinedCurve[{Line[…], …}]` expects its pieces.
  if let Some(args) = box_call("JoinedCurveBox")
    && args.len() >= 2
  {
    let points = box_source_to_graphics_expr(&args[1]);
    return format!("JoinedCurve[{{Line[{points}]}}]");
  }

  // `NCache[exact, approximate]` keeps the exact value a numeric box was
  // typeset from; the exact one is the value.
  if let Some(args) = box_call("NCache")
    && let Some(first) = args.first()
  {
    return box_source_to_graphics_expr(first);
  }

  for (box_head, prim_head) in [
    ("PointBox", "Point"),
    ("LineBox", "Line"),
    ("DiskBox", "Disk"),
    ("RectangleBox", "Rectangle"),
    ("RasterBox", "Raster"),
    ("PolygonBox", "Polygon"),
    ("CircleBox", "Circle"),
    ("ArrowBox", "Arrow"),
  ] {
    if let Some(args) = box_call(box_head) {
      let converted: Vec<String> = args
        .iter()
        .map(|a| box_source_to_graphics_expr(a))
        .collect();
      return format!("{prim_head}[{}]", converted.join(", "));
    }
  }

  // Anything else (`RGBColor[…]`, `Hue[…]`, `Thickness[…]`, option rules,
  // bare numbers/strings, …) is already valid Wolfram Language syntax;
  // recurse into its arguments only in case a `CompressedData` happens to
  // be nested inside one, rather than special-casing every directive head.
  if let Some(open) = s.find('[')
    && s[..open].chars().all(|c| c.is_alphanumeric() || c == '`')
    && let Ok((inner, tail)) = find_matching_bracket(&s[open + 1..])
    && tail.trim().is_empty()
  {
    let head = &s[..open];
    let converted: Vec<String> = split_top_level_commas(inner)
      .into_iter()
      .map(|a| box_source_to_graphics_expr(a.trim()))
      .collect();
    return format!("{head}[{}]", converted.join(", "));
  }

  s.to_string()
}

/// Render a stored `GraphicsBox[…]` output that draws directly with vector
/// primitives — `PointBox`, `LineBox`, `FilledCurveBox`, `DiskBox`,
/// `RectangleBox` — rather than a `RasterBox` snapshot; this is the box
/// form Wolfram's FrontEnd saves for a Demonstration's geometric
/// constructions (Voronoi/Delaunay diagrams, tilings, lattice-point
/// figures, …), which `stored_output_image_svg` does not cover. Rebuilds
/// an evaluable `Graphics[…]` expression from the box dump (see
/// `box_source_to_graphics_expr`) and renders it by evaluating that
/// expression — the same path `ExportString[…, "SVG"]` uses — so the
/// result matches the app's usual graphics styling. Returns `None` when
/// `content` holds no vector `GraphicsBox`, or the reconstructed source
/// fails to parse or evaluate (rather than showing a mangled picture).
pub fn stored_output_vector_graphics_svg(content: &str) -> Option<String> {
  if content.contains("RasterBox[") {
    return None;
  }
  let start = content.find("GraphicsBox[")?;
  let source = box_source_to_graphics_expr(&content[start..]);
  if !source.starts_with("Graphics[") {
    return None;
  }
  let expr = crate::parse_to_expr(&source).ok()?;
  let crate::syntax::Expr::FunctionCall { name, args } = &expr else {
    return None;
  };
  if name != "Graphics" {
    return None;
  }
  match crate::functions::graphics::graphics_ast(args).ok()? {
    crate::syntax::Expr::Graphics { ref svg, .. } => Some(svg.clone()),
    _ => None,
  }
}

#[cfg(test)]
mod tests {
  #[test]
  fn embedded_html_cell_shows_visible_text() {
    let nb = parse_notebook(
      "Notebook[{Cell[BoxData[TemplateBox[{InterpretationBox[Null, EmbeddedHTML[\"<html><head><style>p{}</style></head><body><p>Hi &amp; bye</p><img src=\\\"http://x/a/Snap-1.png\\\" /><script>f();</script></body></html>\"]], TagBox[\"x\", Short]}, \"EmbeddedHTML\"]], \"Text\"]}]",
    )
    .unwrap();
    let CellEntry::Single(cell) = &nb.cells[0] else {
      panic!("expected a single cell")
    };
    assert_eq!(cell.content, "Hi & bye\n[Image: Snap-1.png]");
  }

  #[test]
  fn strip_cell_context_prefix_drops_numbered_cell_contexts() {
    assert_eq!(
      strip_cell_context_prefix(
        "Cell$$2138`f[1] := Cell$$2138`dfunc[x]; a$Cell$$2`b; Cell$$`c; Cell$$12x"
      ),
      "f[1] := dfunc[x]; a$Cell$$2`b; Cell$$`c; Cell$$12x"
    );
  }

  #[test]
  fn strip_cell_context_prefix_drops_only_the_qualifier() {
    assert_eq!(
      strip_cell_context_prefix("f[a_] := $CellContext`a + Global`b"),
      "f[a_] := a + Global`b"
    );
  }

  #[test]
  fn test_join_continuation_cells() {
    let cells = ["f[x_] := Module[{a = 1},", "a + x]", "g = 2;"];
    assert_eq!(
      join_continuation_cells(&cells),
      vec!["f[x_] := Module[{a = 1},\na + x]", "g = 2;"]
    );
    // Brackets in strings and comments do not count.
    let cells = ["s = \"[\";", "(* { *) t = 1;"];
    assert_eq!(
      join_continuation_cells(&cells),
      vec!["s = \"[\";", "(* { *) t = 1;"]
    );
    // An unterminated trailing cell is kept for error reporting.
    assert_eq!(join_continuation_cells(&["f[", "x"]), vec!["f[\nx"]);
  }

  use super::*;

  /// The FrontEnd's icon for a `Root[…]` is a `TemplateBox` over an
  /// association whose `InterpretationFunction` picks the `"interp"` entry.
  #[test]
  fn root_icon_template_box_reads_back_as_root() {
    let boxes = r#"RowBox[{"Sqrt", "[", TemplateBox[<|"icon" -> "Root", "small" -> InterpretationBox[StyleBox["1.47", ShowStringCharacters -> False], 1.4655712318767679`15.95, Editable -> False], "approx" -> 1.465571231876768, "interp" -> InterpretationBox["", Root[-1 - #^2 + #^3& , 1, 0]], "head" -> "Root", "degree" -> "3"|>, "NumericalApproximationInterp", InterpretationFunction->(Slot["interp"]& )], "]"}]"#;
    assert_eq!(
      extract_cell_content(boxes).replace(' ', ""),
      "Sqrt[Root[-1-#^2+#^3&,1,0]]"
    );
  }

  /// Regression: a control variable named with a private-use-area letter
  /// (`\[ScriptM]`, already decoded to U+F6BE in the saved box dump) lost
  /// its `$$` suffix strip, so the saved value never matched the control.
  #[test]
  fn uniquification_suffix_is_stripped_after_private_use_letters() {
    let script_m = "\u{F6BE}";
    let src = format!("f[{script_m}$$, g$$, h$$x]");
    assert_eq!(
      strip_uniquification_suffix(&src),
      format!("f[{script_m}, g, h$$x]")
    );
  }

  /// Regression: a multi-statement `Table` body — several `stmt;`-joined
  /// assignments ending in a result expression, each wrapped in its own
  /// `RowBox` the way the FrontEnd stores a stack of Input-cell lines —
  /// was flattened with the assignment-protecting parens (see
  /// `has_real_predecessor`) wrongly applied to the *whole* `body,
  /// iterator` row instead of individual statements, trapping the
  /// argument-separating comma inside them: `Table[stmt;(stmt);(stmt);
  /// result,{i,0,n})]` instead of `Table[stmt;(stmt);(stmt);result,{i,0,
  /// n}]`. The corrupted text failed to reparse.
  #[test]
  fn test_extract_cell_content_table_body_with_stacked_statements() {
    let s = r#"BoxData[RowBox[{RowBox[{"total", "=", RowBox[{"Table", "[", "\[IndentingNewLine]", RowBox[{RowBox[{RowBox[{"p", "=", "1"}], ";", "\[IndentingNewLine]", RowBox[{"q", "=", "2"}], ";", "\[IndentingNewLine]", RowBox[{"r", "=", RowBox[{"p", "+", "q"}]}], ";", "\[IndentingNewLine]", RowBox[{"{", "r", "}"}]}], "\[IndentingNewLine]", ",", RowBox[{"{", RowBox[{"i", ",", "1", ",", "3"}], "}"}]}], "]"}]}], ";"}]]"#;
    let content = extract_cell_content(s);
    assert_eq!(content, "total=Table[\np=1;\nq=2;\nr=p+q;\n{r}\n,{i,1,3}];");
    crate::clear_state();
    crate::interpret(&content).unwrap();
    assert_eq!(crate::interpret("total").unwrap(), "{{3}, {3}, {3}}");
  }

  /// Regression: a `StyleBox[…, "OperatorCharacter"]`-wrapped `\[Equal]` —
  /// the FrontEnd colors a condition's comparison operators this way —
  /// reconstructed as a single `=` (`Set`, assignment) instead of `==`
  /// (`Equal`), because the wrapper recursed through `extract_cell_content`
  /// (via the generic "StyleBox wraps its displayed argument" case) using
  /// the prose unescaper instead of the code-operator one `box_part_source`
  /// already used for a bare (unwrapped) `"\[Equal]"` RowBox element. A
  /// silently corrupted equality test is worse than a parse failure: the
  /// cell still evaluates, just as the wrong code (e.g. a `Which` guard
  /// that always assigns rather than compares).
  #[test]
  fn test_extract_cell_content_stylebox_wrapped_equal() {
    let s = r#"BoxData[RowBox[{"o1", StyleBox["\[Equal]", "OperatorCharacter"], "\"yes\""}]]"#;
    assert_eq!(extract_cell_content(s), r#"o1=="yes""#);
  }

  /// Regression: the FrontEnd's syntax coloring also wraps operators in
  /// option-only `StyleBox`es (`StyleBox["\[Equal]", FontColor -> …]`), which
  /// reconstructed as `=` (assignment) like the `"OperatorCharacter"` form,
  /// and wraps a partial-derivative operator the same way
  /// (`StyleBox[SubscriptBox["\[PartialD]", "t"], …] f[t]`), which then
  /// read as a product with a `Subscript` instead of `D[f[t], t]`.
  #[test]
  fn test_extract_cell_content_option_stylebox_equal_and_derivative() {
    let s = r#"BoxData[RowBox[{"a", StyleBox["\[Equal]", FontColor -> GrayLevel[0]], "1"}]]"#;
    assert_eq!(extract_cell_content(s), "a==1");
    let s = r#"BoxData[RowBox[{RowBox[{"b", " ", RowBox[{StyleBox[SubscriptBox["\[PartialD]", "t"], FontColor -> GrayLevel[0]], RowBox[{"f", "[", "t", "]"}]}]}], StyleBox["\[Equal]", FontColor -> GrayLevel[0]], "2"}]]"#;
    let code = extract_cell_content(s);
    assert!(code.contains("D[f[t], t]") && code.contains("=="), "{code}");
  }

  /// Regression: an earlier fix for the `StyleBox` case above routed
  /// *every* quoted-string leaf `extract_cell_content` sees through the
  /// code-operator unescaper, not just an operator-tagged `StyleBox`. That
  /// broke ordinary prose: `parse_single_cell` sends a plain `Cell["…",
  /// "Text"]`'s content through this same function regardless of style, so
  /// a `\[Equal]` written mid-sentence (a Demonstration's "for a \[Equal] b
  /// use ...") reconstructed as the code operator `==` instead of the
  /// single `=` glyph it typesets as. The fix must be scoped to the
  /// `"OperatorCharacter"`-tagged `StyleBox` case only; every other quoted
  /// string, including this one despite starting with the same escape,
  /// keeps going through the prose unescaper.
  #[test]
  fn test_extract_cell_content_text_cell_named_char_stays_prose() {
    let s = r#""a \[Equal] b""#;
    assert_eq!(extract_cell_content(s), "a = b");
  }

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
  fn test_textdata_inline_math_groups_compound_scripts() {
    // A nested exponent must not flatten into `e^-b e^-c t`: compound
    // scripts are parenthesized, simple ones stay bare.
    let nb = r#"Notebook[{
Cell[TextData[{
 "Model ",
 Cell[BoxData[
  FormBox[
   RowBox[{
    SubscriptBox["y", "G"], "=", 
    SubscriptBox["y", RowBox[{"i", "+", "1"}]], " ", 
    SuperscriptBox["e", 
     RowBox[{
      RowBox[{"-", "b"}], " ", 
      SuperscriptBox["e", 
       RowBox[{
        RowBox[{"-", "c"}], " ", "t"}]]}]], "+", 
    SuperscriptBox["x", "n"], "+", 
    SuperscriptBox["x", "2"]}], TraditionalForm]], "InlineMath"],
 " done"
}], "Text"]
}]"#;
    let parsed = parse_notebook(nb).unwrap();
    let flat = parsed.flat_cells();
    assert_eq!(
      flat[0].1.content,
      "Model y_G=y_(i+1) e^(-b e^(-c t))+x^n+x² done"
    );
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
  fn test_message_cell_parses_as_print_output() {
    // A kernel message cell (`"Message", "MSG"`) follows the output it came
    // from; it must attach to the input as output, not become a Text cell.
    let nb = parse_notebook(
      "Notebook[{\nCell[\"1/0\", \"Input\"],\n\
       Cell[\"Power::infy: Infinite expression 1/0 encountered.\", \
       \"Message\", \"MSG\",\n CellLabel->\"During evaluation of In[1]:=\"]\n}]",
    )
    .unwrap();
    let flat = nb.flat_cells();
    assert_eq!(flat.len(), 2);
    assert_eq!(flat[1].1.style, CellStyle::Print);
    assert_eq!(
      flat[1].1.content,
      "Power::infy: Infinite expression 1/0 encountered."
    );
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
  fn test_box_leaf_with_string_delimiters_is_a_string_literal() {
    // A Manipulate control label written `"\<surface 1\>"` in a RowBox is
    // the string "surface 1", not the bare tokens `surface 1`.
    let nb = parse_notebook(
      r#"Notebook[{Cell[BoxData[RowBox[{"f", "[", RowBox[{RowBox[{"{", RowBox[{"k", ",", "1", ",", "\"\<surface 1\>\""}], "}"}], ",", RowBox[{"Appearance", "\[Rule]", "\"\<Labeled\>\""}]}], "]"}]], "Input"]}]"#,
    )
    .unwrap();
    let flat = nb.flat_cells();
    assert_eq!(
      flat[0].1.content,
      r#"f[{k,1,"surface 1"},Appearance->"Labeled"]"#
    );
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
  fn test_unescape_literal_space() {
    // `\ ` (backslash immediately followed by a space) is part of the same
    // linear-syntax escape family as `\<`/`\>` above: inside `\!\(…\)`
    // embedded box syntax a bare space separates tokens rather than
    // standing for itself, so the FrontEnd escapes an actual space
    // character as `\ ` — e.g. a Manipulate control label padding a
    // subscript's digits for alignment. It unescapes to one plain space.
    assert_eq!(unescape_string(r"a\ b"), "a b");
    assert_eq!(unescape_string(r"2\ \ \ "), "2   ");
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
  fn test_unescape_two_hex_digit_escape() {
    // `\.HH` is Wolfram's 2-hex-digit Latin-1 escape, e.g. in a
    // Demonstration author's name written as `M\.aa` for "Mª".
    assert_eq!(unescape_string("M\\.aa"), "M\u{AA}");
    // `\.00` is character 0, an invisible spacer: it must not leave a NUL
    // byte behind in the cell's code.
    assert_eq!(unescape_string("1\\.00"), "1");
    // Other control codes (e.g. backspace) are invisible spacers too.
    assert_eq!(unescape_string("0\\.08"), "0");
    assert_eq!(unescape_code_string("{0, 0\\.08, 1}"), "{0, 0, 1}");
    assert_eq!(unescape_code_string("caf\\.e9"), "caf\u{E9}");
    // An incomplete/invalid escape is left untouched rather than eating
    // following characters.
    assert_eq!(unescape_string("a\\.zzb"), "a\\.zzb");
  }

  #[test]
  fn saved_value_keeps_string_literal_escapes() {
    // A saved Manipulate value holding a quote: the literal's own `\"`
    // escapes stay, so it still reads back as one string.
    let saved = r#""\"JH JW\" - QYJWH ""#;
    assert_eq!(unescape_code_outside_strings(saved), saved);
    assert_eq!(
      unescape_code_outside_strings(r#"{\[Alpha], "a\\b"}"#),
      "{\u{3B1}, \"a\\\\b\"}"
    );
  }

  #[test]
  fn test_unescape_six_hex_digit_escape() {
    // `\|HHHHHH` is Wolfram's 6-hex-digit escape for a code point outside
    // the Basic Multilingual Plane, e.g. an Egyptian hieroglyph a
    // Demonstration's data literal writes as `\|013362`.
    assert_eq!(unescape_string("\\|013362"), "\u{13362}");
    assert_eq!(unescape_code_string("\\<\\|0133fa\\>"), "\u{133fa}");
    // An incomplete/invalid escape is left untouched rather than eating
    // following characters.
    assert_eq!(unescape_string("a\\|zzzzzzb"), "a\\|zzzzzzb");
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

  #[test]
  fn test_extract_cell_content_transpose_postfix_operator() {
    // `expr\[Transpose]` is the FrontEnd's postfix notation for
    // `Transpose[expr]` (e.g. `{a, b}\[Transpose]`). Its private-use
    // codepoint (U+F3C7) has no glyph in normal fonts and would render as
    // a blank/tofu box, so keep the readable `\[Transpose]` escape name —
    // our grammar accepts that literal text as the same postfix operator.
    let s = r#"BoxData[RowBox[{RowBox[{"{", "a", "}"}], "\[Transpose]"}]]"#;
    assert_eq!(extract_cell_content(s), "{a}\\[Transpose]");
  }

  #[test]
  fn test_extract_cell_content_conjugate_transpose_postfix_operator() {
    let s =
      r#"BoxData[RowBox[{RowBox[{"{", "a", "}"}], "\[ConjugateTranspose]"}]]"#;
    assert_eq!(extract_cell_content(s), "{a}\\[ConjugateTranspose]");
  }

  #[test]
  fn test_extract_cell_content_conjugate_postfix_operator() {
    let s = r#"BoxData[RowBox[{"z", "\[Conjugate]"}]]"#;
    assert_eq!(extract_cell_content(s), "z\\[Conjugate]");
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

  /// `\[InvisibleComma]` is a comma that isn't drawn. Regression: it was
  /// dropped, so the indices of `k` with subscript `2,2` fused into the
  /// product `2*2` and became `Subscript[k, 4]`.
  #[test]
  fn test_invisible_comma_separates_subscript_indices() {
    let s =
      r#"BoxData[SubscriptBox["k", RowBox[{"2", "\[InvisibleComma]", "1"}]]]"#;
    assert_eq!(extract_cell_content(s), "Subscript[k, 2,1]");
    let s = r#"BoxData[RowBox[{"f", "[", RowBox[{"1", "\[InvisibleComma]", "2"}], "]"}]]"#;
    assert_eq!(extract_cell_content(s), "f[1,2]");
  }

  /// `a[[i]]` normally typesets as a *flat* row — `"a"`,
  /// `"\[LeftDoubleBracket]"`, `i`, `"\[RightDoubleBracket]"` — not as the
  /// `SubscriptBox` form above (that one is rarer in practice, though the
  /// FrontEnd accepts it too). Regression: the bracket names fell through to
  /// their raw Unicode glyphs (`⟦`/`⟧`), which Woxi's parser does not accept
  /// as `Part` syntax, so a downloaded Demonstration notebook's Input cells
  /// — which lean on this typesetting heavily — failed to evaluate.
  #[test]
  fn test_flat_row_double_brackets_is_part() {
    let s = r#"BoxData[RowBox[{"c", "\[LeftDoubleBracket]", "1", "\[RightDoubleBracket]"}]]"#;
    assert_eq!(extract_cell_content(s), "c[[1]]");
    // A compound index expression, wrapped in its own RowBox.
    let s = r#"BoxData[RowBox[{"list", "\[LeftDoubleBracket]", RowBox[{"i", "+", "1"}], "\[RightDoubleBracket]"}]]"#;
    assert_eq!(extract_cell_content(s), "list[[i+1]]");
    // A non-token base still needs the function form.
    let s = r#"BoxData[RowBox[{RowBox[{"a", "+", "b"}], "\[LeftDoubleBracket]", "1", "\[RightDoubleBracket]"}]]"#;
    assert_eq!(extract_cell_content(s), "Part[a+b, 1]");
    // Chained accesses fold left-to-right: `a[[i]][[j]]`.
    let s = r#"BoxData[RowBox[{"a", "\[LeftDoubleBracket]", "i", "\[RightDoubleBracket]", "\[LeftDoubleBracket]", "j", "\[RightDoubleBracket]"}]]"#;
    assert_eq!(extract_cell_content(s), "Part[a[[i]], j]");
    // The pair sits inside a larger row (an assignment), as it does in real
    // Demonstrations code (`list = listi[[RandomInteger[{1, n}]]]`).
    let s = r#"BoxData[RowBox[{"list", "=", RowBox[{"listi", "\[LeftDoubleBracket]", RowBox[{"RandomInteger", "[", "n", "]"}], "\[RightDoubleBracket]"}]}]]"#;
    assert_eq!(extract_cell_content(s), "list=listi[[RandomInteger[n]]]");
    // A line break between the base and `[[` must not turn the index into
    // an implicit product with a stray `Null`.
    let s = r#"BoxData[RowBox[{RowBox[{"f", "[", "x", "]"}], "\[IndentingNewLine]", "\[LeftDoubleBracket]", "1", "\[RightDoubleBracket]"}]]"#;
    assert_eq!(extract_cell_content(s), "Part[f[x], 1]");
  }

  /// A bare string operand of a `FractionBox` is a symbol name, so
  /// `\[ScriptCapitalD]` keeps its private-use code point (U+F773) like it
  /// does inside a `RowBox`; the Unicode glyph U+1D49F would be a different
  /// symbol. Regression: Studio's chemical transport Demonstration mixed
  /// both spellings of the dispersion coefficient.
  #[test]
  fn test_fraction_box_symbol_operand_keeps_private_use_code_point() {
    let s = r#"BoxData[RowBox[{FractionBox[RowBox[{"v", " ", "x"}], "\[ScriptCapitalD]"], "+", "\[ScriptCapitalD]"}]]"#;
    let out = extract_cell_content(s);
    assert_eq!(out.matches('\u{F773}').count(), 2, "{out:?}");
    assert!(!out.contains('\u{1D49F}'), "{out:?}");
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

  /// Separate box tokens never fuse: a symbol then a number, or two
  /// numbers, are a product (`Round[d 400]` is `d*400`, not the symbol
  /// `d400`).
  #[test]
  fn test_juxtaposed_tokens_with_digits_multiply() {
    let s = r#"BoxData[RowBox[{"Round", "[", RowBox[{"d", "400"}], "]"}]]"#;
    assert_eq!(extract_cell_content(s), "Round[d*400]");
    let s = r#"BoxData[RowBox[{"2", "400"}]]"#;
    assert_eq!(extract_cell_content(s), "2*400");
    let s = r#"BoxData[RowBox[{"2", "x"}]]"#;
    assert_eq!(extract_cell_content(s), "2x");
  }

  /// Nested definite integrals take their differentials innermost-first
  /// and list their iterators outermost-first, as Wolfram's parser does.
  #[test]
  fn test_nested_definite_integrals() {
    let s = r#"BoxData[RowBox[{SubsuperscriptBox["\[Integral]", "0", "\[Pi]"], RowBox[{SubsuperscriptBox["\[Integral]", "a", "b"], RowBox[{"\[Rho]", RowBox[{"\[DifferentialD]", "\[Rho]"}], RowBox[{"\[DifferentialD]", "\[Theta]"}]}]}]}]]"#;
    assert_eq!(
      extract_cell_content(s),
      "Integrate[\u{3C1}, {\u{3B8}, 0, \u{3C0}}, {\u{3C1}, a, b}]"
    );
  }

  /// `∇` with variables is the gradient, and with a superscript 2 the
  /// Laplacian, of the box that follows.
  #[test]
  fn test_del_operators() {
    let s = r#"BoxData[RowBox[{RowBox[{SubsuperscriptBox["\[Del]", RowBox[{"{", RowBox[{"x", ",", "y"}], "}"}], "2"], RowBox[{"u", "[", RowBox[{"x", ",", "y"}], "]"}]}], "\[Equal]", "0"}]]"#;
    assert_eq!(extract_cell_content(s), "Laplacian[u[x,y], {x,y}]==0");
    let s = r#"BoxData[RowBox[{SubscriptBox["\[Del]", RowBox[{"{", RowBox[{"x", ",", "y"}], "}"}]], "f"}]]"#;
    assert_eq!(extract_cell_content(s), "Grad[f, {x,y}]");
  }

  /// `\[LeftBracketingBar] body \[RightBracketingBar]` is the typeset form
  /// of `Abs[body]` — the FrontEnd's only use for that glyph pair.
  #[test]
  fn test_bracketing_bars_become_abs() {
    let s = r#"BoxData[RowBox[{"\[LeftBracketingBar]", "x", "\[RightBracketingBar]"}]]"#;
    assert_eq!(extract_cell_content(s), "Abs[x]");
    // The body can be a compound expression.
    let s = r#"BoxData[RowBox[{"\[LeftBracketingBar]", RowBox[{"x", "+", "y"}], "\[RightBracketingBar]"}]]"#;
    assert_eq!(extract_cell_content(s), "Abs[x+y]");
    // Juxtaposed with the rest of a row, like any other factor.
    let s = r#"BoxData[RowBox[{"2", RowBox[{"\[LeftBracketingBar]", "x", "\[RightBracketingBar]"}]}]]"#;
    assert_eq!(extract_cell_content(s), "2Abs[x]");
  }

  /// Restyling part of a word does not split it: the FrontEnd reads
  /// `RowBox[{"Ke", StyleBox["t1", …]}]` as the one symbol `Ket1`.
  #[test]
  fn test_style_box_inside_word_keeps_word() {
    let s = r#"BoxData[RowBox[{"Module", "[", RowBox[{RowBox[{"{", RowBox[{StyleBox[RowBox[{"Ke", StyleBox["t1", FontColor->GrayLevel[0]]}]], ",", "Ket2"}], "}"}], ",", "1"}], "]"}]]"#;
    assert_eq!(extract_cell_content(s), "Module[{Ket1,Ket2},1]");
  }

  /// A two-row, one-column `GridBox` in bare parentheses reads back as the
  /// column matrix `{{n}, {k}}`, as `ToExpression` gives it — not as
  /// `Binomial[n, k]`, which would break code that builds a column vector
  /// this way (`g = (GridBox[{{a}, {b}}]); LinearSolve[m, g]`).
  #[test]
  fn test_stacked_grid_in_parens_is_column_matrix() {
    let s = r#"BoxData[RowBox[{"(", GridBox[{{"a"}, {"b"}}], ")"}]]"#;
    assert_eq!(extract_cell_content(s), "({{a}, {b}})");
    let s =
      r#"BoxData[RowBox[{"2", RowBox[{"(", GridBox[{{"a"}, {"b"}}], ")"}]}]]"#;
    assert_eq!(extract_cell_content(s), "2({{a}, {b}})");
  }

  /// A symbol immediately followed by a parenthesised, comma-separated
  /// argument list (`RowBox[{"\[Psi]", "(", RowBox[{"x", ",", "0"}],
  /// ")"}]`) is the FrontEnd's TraditionalForm-input rendering of function
  /// application — textbook notation `f(x, y)` for `f[x, y]` — since a bare
  /// comma is never valid grouping inside ordinary parentheses. Regression:
  /// a Wolfram Demonstration's plot label (`Style[TraditionalForm[Abs[ψ[x,
  /// 0]]^2], 16]`, typeset with `ψ(x, 0)` in place of `ψ[x, 0]`) came back
  /// as `ψ(x,0)`, which is not valid syntax and failed to parse at all.
  #[test]
  fn test_traditional_form_paren_call_becomes_function_application() {
    let s = r#"BoxData[RowBox[{"\[Psi]", "(", RowBox[{"x", ",", "0"}], ")"}]]"#;
    assert_eq!(extract_cell_content(s), "ψ[x, 0]");
    // Three or more arguments.
    let s =
      r#"BoxData[RowBox[{"f", "(", RowBox[{"x", ",", "y", ",", "z"}], ")"}]]"#;
    assert_eq!(extract_cell_content(s), "f[x, y, z]");
    // An argument can itself be a compound expression.
    let s = r#"BoxData[RowBox[{"\[Psi]", "(", RowBox[{"x", ",", RowBox[{"y", "+", "1"}]}], ")"}]]"#;
    assert_eq!(extract_cell_content(s), "ψ[x, y+1]");
    // A single argument has no comma to disambiguate it from ordinary
    // parenthesised multiplication, so it is left as valid, equivalent
    // `f(x)` syntax rather than reinterpreted as a call.
    let s = r#"BoxData[RowBox[{"f", "(", "x", ")"}]]"#;
    assert_eq!(extract_cell_content(s), "f(x)");
  }

  /// TraditionalForm writes `Exp[x]` as `exp(x)` — a single-argument call
  /// that must not be read back as the product `exp*x`. Regression found
  /// via a Wolfram Demonstration whose Initialization defined its rate
  /// constants as `100 exp(-4000/T)`.
  #[test]
  fn test_traditional_form_single_argument_function_call() {
    let s = r#"BoxData[RowBox[{"100", " ", RowBox[{"exp", "(", RowBox[{"-", FractionBox["4000", "T"]}], ")"}]}]]"#;
    assert_eq!(extract_cell_content(s), "100 Exp[-((4000)/(T))]");
    let s = r#"BoxData[RowBox[{"ln", "(", "x", ")"}]]"#;
    assert_eq!(extract_cell_content(s), "Log[x]");
    let s = r#"BoxData[RowBox[{"sin", "(", "x", ")"}]]"#;
    assert_eq!(extract_cell_content(s), "Sin[x]");
  }

  /// `RadicalBox[a, n]` is the power `a^(1/n)`, not `Surd[a, n]`: the index
  /// may be a real or symbolic number (a Weibull shape parameter), for which
  /// `Surd` rejects the second argument. Regression found via a Wolfram
  /// Demonstration fitting lifetime data to a Weibull model.
  #[test]
  fn test_radical_box_is_a_power_not_surd() {
    let s = r#"BoxData[RadicalBox["x", "b"]]"#;
    assert_eq!(extract_cell_content(s), "Power[x, Power[b, -1]]");
    let s = r#"BoxData[RadicalBox["0.5", "1.5"]]"#;
    assert_eq!(extract_cell_content(s), "Power[0.5, Power[1.5, -1]]");
  }

  /// A parenthesised group (literal `"("`/`")"` box tokens, as the FrontEnd
  /// writes them to force grouping) that is itself followed by a property
  /// call — `(expr)["Prop"]` — must keep its parens when reconstructed as
  /// InputForm text, or the trailing `[...]` rebinds to the group's last
  /// operand instead of the whole group, changing precedence (`a /.
  /// sol[[1,1]]["Domain"]` applies `/.` to a *different* expression than
  /// `(a /. sol[[1,1]])["Domain"]`). Regression found via a Wolfram
  /// Demonstration whose `NDSolve` result was queried with `(r /.
  /// sol[[1,1]])["Domain"]`.
  #[test]
  fn test_parenthesised_group_before_property_call_keeps_parens() {
    let s = r#"BoxData[RowBox[{
      RowBox[{"(",
        RowBox[{"a", "/.",
          RowBox[{"sol", "[", RowBox[{"[", RowBox[{"1", ",", "1"}], "]"}], "]"}]}],
      ")"}],
      "[", "\"\<Domain\>\"", "]"
    }]]"#;
    assert_eq!(extract_cell_content(s), "(a/.sol[[1,1]])[\"Domain\"]");
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

  /// `\[UpEquilibrium]` (⥮) and `\[ReverseUpEquilibrium]` (⥯) are the
  /// same-spin-pair glyphs a Demonstrations electron-configuration diagram
  /// draws its spin arrows with. They were missing from the named-character
  /// table, so a labelled cell fell back to printing the literal escape
  /// `\[UpEquilibrium]` instead of the glyph.
  #[test]
  fn test_up_equilibrium_named_character_renders_as_unicode() {
    let s = r#"BoxData[RowBox[{"f", "[", "\"\<\[UpEquilibrium]\>\"", "]"}]]"#;
    assert_eq!(extract_cell_content(s), "f[\"\u{296E}\"]");
    let s =
      r#"BoxData[RowBox[{"f", "[", "\"\<\[ReverseUpEquilibrium]\>\"", "]"}]]"#;
    assert_eq!(extract_cell_content(s), "f[\"\u{296F}\"]");
  }

  /// A caption's inline math routinely states set membership
  /// (`\[Element] \[DoubleStruckCapitalR]`, `…N`, `…Z`, …). Unlike the
  /// sibling private-use *script* letters (`\[ScriptCapitalD]` → 𝒟), the
  /// private-use *double-struck* letters had no glyph substitution at all,
  /// so a Text cell printed nothing visible for them. `R`, `N` and `Z` are
  /// three of the seven capitals Unicode gives their own Letterlike
  /// Symbols code point rather than a Mathematical Alphanumeric Symbols
  /// slot; `a` exercises the plain contiguous lowercase block.
  #[test]
  fn test_double_struck_named_characters_render_as_unicode() {
    let s = r#"TextData["\[DoubleStruckCapitalR] \[DoubleStruckCapitalN] \[DoubleStruckCapitalZ] \[DoubleStruckA]"]"#;
    assert_eq!(
      extract_cell_content(s),
      "\u{211D} \u{2115} \u{2124} \u{1D552}"
    );
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
  fn test_extract_cell_content_template_box_compound_quantity() {
    // A compound unit (e.g. `Quantity[9.8, "Meters"/"Seconds"^2]`) is
    // typeset with the unit slot holding real boxes (`FractionBox`,
    // `SuperscriptBox`) instead of a single string literal like the simple
    // `"USDollars"` case above. Regression test for a bug where the boxes
    // were stringified verbatim (`Quantity[9.8, "FractionBox[...]"]`)
    // instead of being recursively converted back to InputForm text.
    let s = r#"BoxData[TemplateBox[{"9.8", RowBox[{"\"m\"", "/", SuperscriptBox["\"s\"", "2"]}], "meters per second squared", FractionBox["\"Meters\"", SuperscriptBox["\"Seconds\"", "2"]]}, "Quantity"]]"#;
    assert_eq!(
      extract_cell_content(s),
      "Quantity[9.8, ((\"Meters\")/(((\"Seconds\")^(2))))]"
    );
  }

  #[test]
  fn test_extract_cell_content_fraction_box_is_one_operand() {
    // `Ceiling /@ FractionBox[l, d]` maps over the quotient; without the
    // outer parentheses it became `(Ceiling /@ l) / d`.
    let s = r#"BoxData[RowBox[{"Ceiling", "/@", FractionBox[RowBox[{"{", RowBox[{"i", ",", "j"}], "}"}], SqrtBox["s"]]}]]"#;
    let out = extract_cell_content(s);
    assert!(out.starts_with("Ceiling/@(("), "{out}");
    assert!(out.ends_with(")/(Sqrt[s]))"), "{out}");
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
  fn test_extract_image_raster_keeps_image_size() {
    // Regression: the `ImageSize` option of the typeset `GraphicsBox` was
    // dropped, so a restored image showed at its pixel size instead of the
    // size the notebook displayed it at.
    let s = "GraphicsBox[\n       TagBox[RasterBox[CompressedData[\"\n1:eJxTTMoP\nSmNiYGAA\nAAtLAe0=\n         \"], {{0, 2}, {2, 0}}, {0, 255}],\n        BoxForm`ImageTag[\"Byte\", ColorSpace -> \"GrayLevel\"],\n        Selectable->False],\n       BaseStyle->\"ImageGraphics\",\n       ImageSize->{281.5, Automatic}]";
    assert_eq!(
      extract_cell_content(s),
      "Image[CompressedData[\"1:eJxTTMoPSmNiYGAAAAtLAe0=\"], \"Byte\", \
       ColorSpace -> \"GrayLevel\", ImageSize -> {281.5, Automatic}]"
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
    // literal — it typesets a vector graphic instead, and is rewritten to
    // the `Graphics[…]` expression it draws so the cell stays runnable
    // (rather than left as un-evaluable box syntax).
    let s = "GraphicsBox[DiskBox[{0, 0}]]";
    assert_eq!(extract_cell_content(s), "Graphics[Disk[{0, 0}]]");
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
  fn test_extract_textdata_unstyled_inline_formula_cell() {
    // Short inline formulas are written as unstyled `Cell[BoxData[FormBox[…]]]`
    // elements (no "InlineMath" style); they must still contribute their
    // content instead of leaving holes in the sentence.
    let s = r#"TextData[{
 "Rules with ",
 Cell[BoxData[
  FormBox[
   RowBox[{"k", "=", "3"}], TraditionalForm]],ExpressionUUID->
  "8e2c93cc-7427-4e4c-804d-674333ce53be"],
 " colors and radius ",
 Cell[BoxData[
  FormBox[
   RowBox[{"r", "=", "1"}], TraditionalForm]],ExpressionUUID->
  "701fcd97-4426-1943-9c77-f63aa139034d"],
 " are counted."
}]"#;
    assert_eq!(
      extract_cell_content(s),
      "Rules with k=3 colors and radius r=1 are counted."
    );
  }

  #[test]
  fn test_extract_display_formula_wrapping_inline_math_cell() {
    // A Text cell whose BoxData is a display `FormBox[RowBox[{Cell[…,
    // "InlineMath"], ","}]]`: the equation sits in an inline cell followed by
    // trailing punctuation. The cell must render as its typeset content
    // instead of leaking its raw `Cell[TextData[…` source into the prose.
    let s = r#"BoxData[
 FormBox[
  RowBox[{Cell[TextData[Cell[BoxData[
    FormBox[
     RowBox[{
      RowBox[{"Q", "(", "u", ")"}], "=", 
      FractionBox["a", "b"]}], TraditionalForm]],
    ExpressionUUID->"11111111-1111-1111-1111-111111111111"]], "InlineMath",
    ExpressionUUID->"22222222-2222-2222-2222-222222222222"], ","}], 
  TraditionalForm]]"#;
    let out = extract_cell_content(s);
    assert!(!out.contains("Cell["), "raw cell source leaked: {out}");
    assert!(!out.contains("TextData"), "raw cell source leaked: {out}");
    assert!(out.contains("Q(u)"), "{out}");
    assert!(out.trim_end().ends_with(','), "{out}");
  }

  #[test]
  fn test_extract_display_formula_unstyled_textdata_wrapper() {
    // The FrontEnd also wraps the inline formula in an *unstyled*
    // `Cell[TextData[Cell[…, "InlineMath"]]]` before the trailing
    // punctuation; that wrapper must not swallow the formula.
    let s = r#"BoxData[
 FormBox[
  RowBox[{Cell[TextData[Cell[BoxData[
    FormBox[
     RowBox[{
      SubscriptBox["X", "tot"], "=", 
      SubscriptBox["X", 
       StyleBox["S", FontSlant->"Plain"]]}], TraditionalForm]], "InlineMath",
    ExpressionUUID->"11111111-1111-1111-1111-111111111111"]],ExpressionUUID->
    "22222222-2222-2222-2222-222222222222"], "."}], TraditionalForm]]"#;
    let out = extract_cell_content(s);
    assert!(!out.contains("Cell["), "raw cell source leaked: {out}");
    assert!(out.contains("X_tot"), "{out}");
    assert!(out.contains("X_S"), "{out}");
    assert!(out.trim_end().ends_with('.'), "{out}");
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

  /// Build the exact serialization Mathematica writes for a packed array
  /// of machine reals — `!boR` + tag `e` + rank + dims + raw `f64` data,
  /// zlib-compressed and base64-encoded behind a `1:` prefix — the same
  /// format `test_stored_output_raster_snapshot_decodes_to_svg` builds for
  /// the sibling `RawArray["UnsignedInteger8", …]` pixel format.
  fn make_compressed_real_array(dims: &[u32], values: &[f64]) -> String {
    let mut raw: Vec<u8> = Vec::new();
    raw.extend_from_slice(b"!boR");
    raw.push(b'e');
    raw.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for d in dims {
      raw.extend_from_slice(&d.to_le_bytes());
    }
    for v in values {
      raw.extend_from_slice(&v.to_le_bytes());
    }

    use base64::Engine;
    use std::io::Write;
    let mut enc = flate2::write::ZlibEncoder::new(
      Vec::new(),
      flate2::Compression::default(),
    );
    enc.write_all(&raw).unwrap();
    let b64 =
      base64::engine::general_purpose::STANDARD.encode(enc.finish().unwrap());
    format!("1:{b64}")
  }

  #[test]
  fn test_decode_compressed_real_array_round_trips() {
    let payload =
      make_compressed_real_array(&[3, 2], &[0.0, 0.0, 1.5, 2.25, -3.0, 4.0]);
    let (dims, values) =
      decode_compressed_real_array(&payload).expect("decodes");
    assert_eq!(dims, vec![3, 2]);
    assert_eq!(values, vec![0.0, 0.0, 1.5, 2.25, -3.0, 4.0]);
    assert_eq!(
      packed_real_array_to_wl_list(&dims, &values),
      "{{0, 0}, {1.5, 2.25}, {-3, 4}}"
    );
  }

  #[test]
  fn test_box_source_to_graphics_expr_renames_primitive_boxes() {
    assert_eq!(
      box_source_to_graphics_expr("PointBox[{{0, 0}, {1, 1}}]"),
      "Point[{{0, 0}, {1, 1}}]"
    );
    assert_eq!(
      box_source_to_graphics_expr("LineBox[{{0, 0}, {1, 1}}]"),
      "Line[{{0, 0}, {1, 1}}]"
    );
    assert_eq!(
      box_source_to_graphics_expr("DiskBox[{0, 0}, 1]"),
      "Disk[{0, 0}, 1]"
    );
    assert_eq!(
      box_source_to_graphics_expr("RectangleBox[{0, 0}, {1, 1}]"),
      "Rectangle[{0, 0}, {1, 1}]"
    );
  }

  #[test]
  fn test_box_source_to_graphics_expr_flattens_style_box() {
    assert_eq!(
      box_source_to_graphics_expr(
        "StyleBox[{PointBox[{0, 0}]}, RGBColor[1, 0, 0], StripOnInput -> False]"
      ),
      "{RGBColor[1, 0, 0], {Point[{0, 0}]}}"
    );
  }

  #[test]
  fn test_box_source_to_graphics_expr_unwraps_tag_box() {
    assert_eq!(
      box_source_to_graphics_expr("TagBox[PointBox[{0, 0}], Whatever]"),
      "Point[{0, 0}]"
    );
  }

  #[test]
  fn test_box_source_to_graphics_expr_filled_curve_to_polygon() {
    // The tag structure (first argument) only distinguishes line/Bezier
    // segments; it is dropped, and the point run (second argument) becomes
    // a straight-edged `Polygon`.
    assert_eq!(
      box_source_to_graphics_expr(
        "FilledCurveBox[{{{1, 2, 3}}}, {{{0, 0}, {1, 0}, {1, 1}}}, \
         CurveClosed -> {1}]"
      ),
      "Polygon[{{{0, 0}, {1, 0}, {1, 1}}}]"
    );
  }

  #[test]
  fn test_box_source_to_graphics_expr_joined_curve_to_joined_curve() {
    // `JoinedCurveBox` is the stroked (unfilled) sibling of
    // `FilledCurveBox`, saved for a Demonstration's cached curve/icon
    // output; it must rename to the evaluable `JoinedCurve` primitive
    // (approximating every segment as a straight line, the same
    // simplification `FilledCurveBox` already makes) instead of being
    // left as an unrecognized head that `graphics_ast` silently drops.
    assert_eq!(
      box_source_to_graphics_expr(
        "JoinedCurveBox[{{{1, 2, 3}}}, {{{0, 0}, {1, 0}, {1, 1}}}, \
         CurveClosed -> {1}]"
      ),
      "JoinedCurve[{Line[{{{0, 0}, {1, 0}, {1, 1}}}]}]"
    );
  }

  #[test]
  fn test_stored_output_vector_graphics_renders_joined_curve_box() {
    // A stored `GraphicsBox[JoinedCurveBox[…]]` output (the box form the
    // FrontEnd saves for a cached curve, distinct from the already-handled
    // `FilledCurveBox`) must still render as an SVG picture rather than
    // silently producing no primitives.
    let content = r#"Cell[BoxData[
 GraphicsBox[JoinedCurveBox[{{{1, 2, 3}}}, {{{0, 0}, {1, 0}, {1, 1}}}]],
 "Output"]"#;
    let svg = stored_output_vector_graphics_svg(content)
      .expect("a JoinedCurveBox output must render as an SVG");
    assert!(svg.contains("<svg"), "{svg}");
    assert!(
      svg.contains("<path") || svg.contains("<polyline"),
      "expected a stroked path/polyline for the joined curve: {svg}"
    );
  }

  #[test]
  fn test_box_source_to_graphics_expr_inlines_compressed_data() {
    let payload = make_compressed_real_array(&[2, 2], &[0.0, 0.0, 1.0, 1.0]);
    let source = format!(
      "FilledCurveBox[{{{{1, 2, 2}}}}, {{CompressedData[\"{payload}\"]}}]"
    );
    assert_eq!(
      box_source_to_graphics_expr(&source),
      "Polygon[{{{0, 0}, {1, 1}}}]"
    );
  }

  /// Build the exact serialization Mathematica writes for a `RasterBox`'s
  /// pixel data (`!boR`, then `f`/`RawArray`/`S`/`UnsignedInteger8`, then
  /// `b`, rank, dims and the raw byte samples), zlib-compressed and
  /// base64-encoded behind a `1:` prefix (the format
  /// `test_stored_output_raster_snapshot_decodes_to_svg` also builds).
  fn make_compressed_raster_array(
    height: u32,
    width: u32,
    channels: u32,
    pixels: &[u8],
  ) -> String {
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
    let rank = if channels == 1 { 2u32 } else { 3u32 };
    raw.extend_from_slice(&rank.to_le_bytes());
    raw.extend_from_slice(&height.to_le_bytes());
    raw.extend_from_slice(&width.to_le_bytes());
    if channels != 1 {
      raw.extend_from_slice(&channels.to_le_bytes());
    }
    raw.extend_from_slice(pixels);

    use base64::Engine;
    use std::io::Write;
    let mut enc = flate2::write::ZlibEncoder::new(
      Vec::new(),
      flate2::Compression::default(),
    );
    enc.write_all(&raw).unwrap();
    let b64 =
      base64::engine::general_purpose::STANDARD.encode(enc.finish().unwrap());
    format!("1:{b64}")
  }

  #[test]
  fn test_decode_compressed_raster_as_wl_list_decodes_grayscale_and_rgb() {
    // A 1x2 grayscale image.
    let gray_payload = make_compressed_raster_array(1, 2, 1, &[10, 200]);
    assert_eq!(
      decode_compressed_raster_as_wl_list(&gray_payload),
      Some("{{10, 200}}".to_string())
    );

    // A 2x1 RGB image: one red pixel row, one green pixel row.
    let rgb_payload =
      make_compressed_raster_array(2, 1, 3, &[255, 0, 0, 0, 255, 0]);
    assert_eq!(
      decode_compressed_raster_as_wl_list(&rgb_payload),
      Some("{{{255, 0, 0}}, {{0, 255, 0}}}".to_string())
    );

    // The packed-real-array format is not the raster format.
    let real_payload = make_compressed_real_array(&[2], &[1.0, 2.0]);
    assert_eq!(decode_compressed_raster_as_wl_list(&real_payload), None);
  }

  /// A `RasterBox` reads as the `Raster` it typesets, and its
  /// `CompressedData` carries pixel bytes, not the packed-real-array
  /// coordinate format — an Initialization Code cell
  /// that assigns a bitmap icon to a symbol (a common Demonstrations
  /// pattern for custom pictures used inside a `Manipulate`) must
  /// reconstruct the actual pixel grid rather than losing the picture to
  /// `Null`, which previously left every use of that symbol blank.
  #[test]
  fn test_box_source_to_graphics_expr_inlines_raster_pixel_data() {
    let payload = make_compressed_raster_array(1, 2, 3, &[1, 2, 3, 4, 5, 6]);
    let source = format!(
      "RasterBox[CompressedData[\"{payload}\"], {{{{0, 0}}, {{2, 1}}}}, \
       {{0, 255}}]"
    );
    assert_eq!(
      box_source_to_graphics_expr(&source),
      "Raster[{{{1, 2, 3}, {4, 5, 6}}}, {{0, 0}, {2, 1}}, {0, 255}]"
    );
    // `NCache[exact, approximate]` reads as its exact value.
    assert_eq!(
      box_source_to_graphics_expr(
        "GraphicsBox[RasterBox[NCache[{{1, 2}}, {{1., 2.}}]]]"
      ),
      "Graphics[Raster[{{1, 2}}]]"
    );
  }

  /// Build the serialization Mathematica writes for a *bare* packed byte
  /// array — `!boR` + `b` + rank + dims + raw samples, with no
  /// `RawArray["UnsignedInteger8", …]` function-call wrapper — the shape
  /// a Demonstration's Initialization Code cell embeds for an inline
  /// bitmap assigned straight to a symbol, as opposed to
  /// `make_compressed_raster_array`'s wrapped shape for a stored Output
  /// snapshot.
  fn make_compressed_packed_byte_array(
    height: u32,
    width: u32,
    channels: u32,
    pixels: &[u8],
  ) -> String {
    let mut raw: Vec<u8> = Vec::new();
    raw.extend_from_slice(b"!boR");
    raw.push(b'b');
    let rank = if channels == 1 { 2u32 } else { 3u32 };
    raw.extend_from_slice(&rank.to_le_bytes());
    raw.extend_from_slice(&height.to_le_bytes());
    raw.extend_from_slice(&width.to_le_bytes());
    if channels != 1 {
      raw.extend_from_slice(&channels.to_le_bytes());
    }
    raw.extend_from_slice(pixels);

    use base64::Engine;
    use std::io::Write;
    let mut enc = flate2::write::ZlibEncoder::new(
      Vec::new(),
      flate2::Compression::default(),
    );
    enc.write_all(&raw).unwrap();
    let b64 =
      base64::engine::general_purpose::STANDARD.encode(enc.finish().unwrap());
    format!("1:{b64}")
  }

  /// The bare packed-byte-array format (no `RawArray` wrapper) is the one
  /// Wolfram actually writes for a picture assigned inline in a
  /// Demonstration's Initialization Code cell — verified by decoding the
  /// real `CompressedData` payload from a downloaded Wolfram
  /// Demonstrations Project notebook and finding it used this shape
  /// rather than `parse_raw_array_u8`'s `RawArray`-wrapped one.
  #[test]
  fn test_decode_compressed_raster_as_wl_list_decodes_bare_packed_bytes() {
    let payload =
      make_compressed_packed_byte_array(1, 2, 3, &[7, 8, 9, 10, 11, 12]);
    assert_eq!(
      decode_compressed_raster_as_wl_list(&payload),
      Some("{{{7, 8, 9}, {10, 11, 12}}}".to_string())
    );

    let gray_payload = make_compressed_packed_byte_array(2, 1, 1, &[3, 250]);
    assert_eq!(
      decode_compressed_raster_as_wl_list(&gray_payload),
      Some("{{3}, {250}}".to_string())
    );
  }

  #[test]
  fn test_text_cell_inline_diagram_renders_as_graphics_placeholder() {
    // A Demonstration's Details text sometimes embeds a small illustrative
    // diagram inline in a sentence via `Cell[BoxData[FormBox[GraphicsBox[
    // …], TraditionalForm]], "InlineMath"]` (e.g. arrows and labels
    // sketching a setup) rather than a formula. Left unconverted, the Text
    // cell would show the reconstructed `Graphics[…]` source — hundreds of
    // characters of box coordinates — inline in the prose. It must instead
    // fall back to Wolfram's own plain-text placeholder for a graphic,
    // `-Graphics-`, the same way `Print[Graphics[…]]` renders as text.
    let text_data = r#"{"setup: ", Cell[BoxData[
 FormBox[
  GraphicsBox[{RGBColor[1, 0, 0], PointBox[{0, 0}],
    InsetBox["label", {1, 1}]}], TraditionalForm]], "InlineMath",
  ExpressionUUID->"00000000-0000-0000-0000-000000000000"], "."}"#;
    let rendered = extract_textdata(text_data);
    assert_eq!(rendered, "setup: -Graphics-.");
  }

  #[test]
  fn test_text_cell_inline_raster_renders_as_image_placeholder() {
    // Same idea, but the inline box is a raster (`RasterBox` with an
    // `ImageTag`) rather than vector primitives — Wolfram's plain-text
    // placeholder for that case is `-Image-`, not `-Graphics-`.
    let payload = make_compressed_packed_byte_array(1, 1, 1, &[128]);
    let raster = format!(
      "RasterBox[CompressedData[\"{payload}\"], {{{{0, 1}}, {{1, 0}}}}, {{0, 255}}]"
    );
    let tag_box = format!("TagBox[{raster}, BoxForm`ImageTag[\"Byte\"]]");
    let graphics_box = format!("GraphicsBox[{tag_box}]");
    let cell = format!("Cell[BoxData[{graphics_box}], \"InlineMath\"]");
    let text_data = format!("{{\"photo: \", {cell}}}");
    let rendered = extract_textdata(&text_data);
    assert_eq!(rendered, "photo: -Image-");
  }

  /// Same idea as `test_text_cell_inline_diagram_renders_as_graphics_placeholder`,
  /// but the inline box is a `Graphics3DBox[…]` — e.g. a 3-D sketch of a
  /// solid embedded inline in a Demonstration's Details text. Regression:
  /// only `GraphicsBox` was recognized, so a `Graphics3DBox` fell through to
  /// the evaluable-InputForm extractor and flooded the paragraph with the
  /// reconstructed `Graphics3D[…]` source instead of Wolfram's own
  /// plain-text placeholder, `-Graphics3D-`.
  #[test]
  fn test_text_cell_inline_3d_diagram_renders_as_graphics3d_placeholder() {
    let text_data = r#"{"solid: ", Cell[BoxData[
 FormBox[
  Graphics3DBox[{RGBColor[0, 0, 1], Point3DBox[{0, 0, 0}]}],
  TraditionalForm]], "InlineMath",
  ExpressionUUID->"00000000-0000-0000-0000-000000000001"], "."}"#;
    let rendered = extract_textdata(text_data);
    assert_eq!(rendered, "solid: -Graphics3D-.");
  }

  #[test]
  fn test_stored_output_vector_graphics_svg_renders_points() {
    let content = "Cell[BoxData[GraphicsBox[{RGBColor[1, 0, 0], \
                    PointBox[{0, 0}]}]], \"Output\"]";
    let svg = stored_output_vector_graphics_svg(content).expect("renders");
    assert!(svg.contains("<svg"), "{svg}");
    assert!(svg.contains("<circle"), "{svg}");
  }

  #[test]
  fn test_stored_output_vector_graphics_svg_skips_raster_and_plain_content() {
    // A raster snapshot is `stored_output_image_svg`'s job, not this one's.
    let raster = "GraphicsBox[TagBox[RasterBox[CompressedData[\"1:x\"], \
                  {{0, 1}, {1, 0}}, {0, 255}], BoxForm`ImageTag[\"Byte\"]]]";
    assert!(stored_output_vector_graphics_svg(raster).is_none());
    assert!(stored_output_vector_graphics_svg("{1, 2, 3}").is_none());
  }

  #[test]
  fn test_extract_cell_content_inline_vector_graphics_literal_stays_runnable() {
    // A Demonstration's initialization code sometimes bakes in a
    // precomputed picture (e.g. a lattice of points) as an inline
    // `Graphics[…]` literal rather than recomputing it on every
    // evaluation; the FrontEnd typesets that literal as a `GraphicsBox[…]`
    // of vector primitives, exactly like the one this regression guards
    // (a real Demonstration used `pts = GraphicsBox[…]` this way, with
    // `FilledCurveBox`/`PointBox` primitives holding `CompressedData`
    // coordinates — reproduced here with synthetic data, not the original
    // notebook's). Left unconverted, the reconstructed Input cell would
    // embed that raw box syntax as code, which fails to evaluate; this
    // must come back as ordinary, runnable `Graphics[…]` instead.
    let payload =
      make_compressed_real_array(&[3, 2], &[0.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
    let s = format!(
      "RowBox[{{\"pts\", \"=\", GraphicsBox[{{StyleBox[{{FilledCurveBox[{{{{1, \
       2, 3}}}}, {{CompressedData[\"{payload}\"]}}]}}, RGBColor[1, 0, 0]], \
       PointBox[{{0, 0}}]}}]}}]"
    );
    let result = extract_cell_content(&s);
    assert!(!result.contains("FilledCurveBox"), "{result}");
    assert!(!result.contains("CompressedData"), "{result}");
    assert!(result.contains("Graphics["), "{result}");
    assert!(result.contains("Polygon["), "{result}");
    assert!(result.contains("Point["), "{result}");
    // The reconstructed source must actually parse as Wolfram Language.
    assert!(crate::parse_to_expr(&result).is_ok(), "{result}");
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

  /// A stacked binomial coefficient inside prose math must stay on one
  /// line instead of spilling its grid rows across a line break.
  #[test]
  fn test_textdata_inline_math_stacked_binomial_stays_on_one_line() {
    let s = r#"TextData[Cell[BoxData[
 FormBox[
  RowBox[{
   RowBox[{"(", GridBox[{
      {"n"},
      {"k"}
     }], ")"}],
   RowBox[{"a", "+", "b"}]}], TraditionalForm]], "InlineMath"]]"#;
    assert_eq!(extract_cell_content(s), "C(n, k)a+b");
    let s = r#"TextData[{"so ", Cell[BoxData[
 FormBox[RowBox[{"(", GridBox[{{SubscriptBox["m", "1"]}, {"2"}}], ")"}],
  TraditionalForm]], "InlineMath"], " follows"}]"#;
    assert_eq!(extract_cell_content(s), "so C(m_1, 2) follows");
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

  #[test]
  fn test_extract_cell_content_super_star_and_friends() {
    // `SuperscriptBox["A", "*"]` is `SuperStar[A]`, not a string-scripted
    // `Superscript`, so definitions and replacements on it work.
    let s = r#"BoxData[RowBox[{SuperscriptBox["A", "*"], "=", "3"}]]"#;
    assert_eq!(extract_cell_content(s), "SuperStar[A]=3");
    let s = r#"BoxData[SuperscriptBox["M", "\[Dagger]"]]"#;
    assert_eq!(extract_cell_content(s), "SuperDagger[M]");
  }

  #[test]
  fn test_extract_cell_content_postfix_operator_superscripts() {
    // `SuperscriptBox["A", "\[ConjugateTranspose]"]` is the typeset form of
    // `ConjugateTranspose[A]` (NumericalRangeForSomeComplexUpper…).
    for (op, head) in [
      ("\\[Transpose]", "Transpose"),
      ("\\[Conjugate]", "Conjugate"),
      ("\\[ConjugateTranspose]", "ConjugateTranspose"),
      ("\\[HermitianConjugate]", "ConjugateTranspose"),
    ] {
      let s = format!(r#"BoxData[SuperscriptBox["A", "{op}"]]"#);
      assert_eq!(extract_cell_content(&s), format!("{head}[A]"));
    }
  }

  /// A mixed or higher-order partial derivative — more than one argument,
  /// or a single argument past the order primes are used for — types as
  /// `SuperscriptBox[f, TagBox[<parenthesized order tuple>, Derivative]]`,
  /// not the plain-prime form the previous test covers. This must convert
  /// to `Derivative[n1, n2, ...][f]`, not to `(f)^((1,0))`, which would
  /// neither parse nor mean the same thing. Regression: a Picard-Fuchs
  /// derivation whose helper functions pattern-match on
  /// `Derivative[1, 0][p][a, t]`-shaped terms never matched anything,
  /// because every occurrence typeset this way decoded to
  /// `Superscript[p, "(1,0)"][a, t]` instead.
  #[test]
  fn test_extract_cell_content_mixed_partial_derivative() {
    let s = r#"BoxData[RowBox[{
      SuperscriptBox["p",
       TagBox[RowBox[{"(", RowBox[{"1", ",", "0"}], ")"}], Derivative],
       MultilineFunction->None], "[",
      RowBox[{"a", ",", "t"}], "]"}]]"#;
    assert_eq!(extract_cell_content(s), "Derivative[1, 0][p][a,t]");

    // A single-argument function differentiated to an order past where
    // primes are used still carries the same `TagBox[…, Derivative]`
    // wrapper, just with one bare number instead of a comma list.
    let s = r#"BoxData[RowBox[{
      SuperscriptBox["f", TagBox["(4)", Derivative], MultilineFunction->None],
      "[", "x", "]"}]]"#;
    assert_eq!(extract_cell_content(s), "Derivative[4][f][x]");
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

  /// A power typeset as a superscript is one operand of its neighbours:
  /// `f /@ rx^n` maps over the power, not the power of a map
  /// (InversiveGeometryVIISpiderEyes).
  #[test]
  fn superscript_is_one_operand() {
    let s = r#"BoxData[RowBox[{"f", "/@", SuperscriptBox["rx", "n"]}]]"#;
    assert_eq!(extract_cell_content(s), "f/@((rx)^(n))");
  }

  /// `SubscriptBox["X", "1,i"]` is `Subscript[X, 1, i]`, and a subscript
  /// typed as ASCII `[[1]]` is `Part` (DynamicsOfAContinuousDistillation…,
  /// TaxIncidence).
  #[test]
  fn subscript_index_lists_and_ascii_parts() {
    let s = r#"BoxData[SubscriptBox["X", RowBox[{"1", ",", "i"}]]]"#;
    assert_eq!(extract_cell_content(s), "Subscript[X, 1,i]");
    let s = r#"BoxData[SubscriptBox["x", RowBox[{"[", RowBox[{"[", "1", "]"}], "]"}]]]"#;
    assert_eq!(extract_cell_content(s), "x[[1]]");
  }

  #[test]
  fn comment_before_arguments_is_not_a_factor() {
    let s = r#"BoxData[RowBox[{"f", "[", "\[IndentingNewLine]", RowBox[{"(*", "c", "*)"}], "\[IndentingNewLine]", RowBox[{RowBox[{RowBox[{"a", "=", "1"}], ";", RowBox[{"(*", RowBox[{"g", "["}], "*)"}], "a"}], ",", "b"}], "]"}]]"#;
    let code = extract_cell_content(s);
    assert!(crate::parse_to_expr(&code).is_ok(), "{code}");
    assert!(!code.contains("(a=1"), "{code}");
  }

  /// A typeset Graph literal is the Graph kept in its NetworkGraphics
  /// module (GraphProducts).
  #[test]
  fn network_graphics_box_is_its_graph() {
    let s = r#"BoxData[GraphicsBox[NamespaceBox["NetworkGraphics", DynamicModuleBox[{Typeset`graph = HoldComplete[Graph[{1, 2}, {Null, SparseArray[Automatic, {2, 2}, 0, {1, {{0, 1, 2}, {{2}, {1}}}, Pattern}]}, {VertexCoordinates -> {{1, 0}, {2, 0}}}]], Typeset`boxes}, {}]]]]"#;
    assert_eq!(
      extract_cell_content(s),
      "Graph[{1, 2}, {Null, SparseArray[Automatic, {2, 2}, 0, {1, {{0, 1, 2}, {{2}, {1}}}, Pattern}]}, {VertexCoordinates -> {{1, 0}, {2, 0}}}]"
    );
  }

  /// A comment between a call's bracket and its arguments neither makes
  /// the arguments a product nor hides a top-level comma with the
  /// brackets it contains (PolyhedralPlayground).
  #[test]
  fn comments_inside_argument_lists() {
    assert_eq!(
      split_top_level_commas("a=1;(*g[*)a, b"),
      vec!["a=1;(*g[*)a", " b"]
    );
  }

  #[test]
  fn test_extract_cell_content_superscript_with_option() {
    // A plain power whose box carries a display option must still convert.
    let s = r#"BoxData[SuperscriptBox["x", "2", MultilineFunction->None]]"#;
    assert_eq!(extract_cell_content(s), "((x)^(2))");
  }

  /// `SuperscriptBox["H", "-"]` typesets the hydride ion "H⁻": a bare `-`
  /// with no operand, used purely for display. `(H)^(-)` would dangle the
  /// minus sign and fail to parse — Mathematica itself falls back to the
  /// unevaluated `Superscript` wrapper whenever a script isn't a valid
  /// expression on its own. From the "Variational Calculations on the
  /// Helium Isoelectronic Series" Demonstration's element/ion table.
  #[test]
  fn test_extract_cell_content_superscript_bare_minus() {
    let s = r#"BoxData[SuperscriptBox["\"\<H\>\"", "-"]]"#;
    assert_eq!(extract_cell_content(s), "Superscript[\"H\", \"-\"]");
  }

  #[test]
  fn test_extract_cell_content_squared_derivative() {
    // Nested: (φ'[t])^2 — a derivative RowBox inside a power box.
    let s = r#"BoxData[SuperscriptBox[
      RowBox[{
       SuperscriptBox["\[Phi]", "\[Prime]",
        MultilineFunction->None], "[", "t", "]"}], "2"]]"#;
    assert_eq!(extract_cell_content(s), "((ϕ'[t])^(2))");
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
  fn test_extract_cell_content_symbol_before_power_juxtaposition() {
    // A coefficient symbol directly followed by a power box with no
    // operator between them — `RowBox[{"a", RowBox[{"x", "^", "3"}]}]` — is
    // the FrontEnd's typeset form of `a*x^3` (implicit multiplication with
    // no `\[InvisibleTimes]` token). Gluing the two pieces' text together
    // verbatim reads back as the single identifier `ax` to the third power
    // instead of `a` times `x^3`.
    let s = r#"BoxData[RowBox[{"a", RowBox[{"x", "^", "3"}]}]]"#;
    let content = extract_cell_content(s);
    assert_eq!(content, "a*x^3");
    assert_eq!(
      crate::interpret(&format!("({content}) /. {{a -> 2, x -> 3}}")).unwrap(),
      "54"
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
  fn test_extract_textdata_multi_token_subscripts_are_grouped() {
    // A subscript made of several tokens reads as one script, so
    // `R_{a+b}` is `R_(a+b)` and not `R_a+b`; single tokens stay bare.
    let s = r#"TextData[{
 Cell[BoxData[
  FormBox[
   RowBox[{
    SubscriptBox["R", RowBox[{"a", "+", "b"}]], "=",
    SubscriptBox["R", RowBox[{"c", ",", "d"}]], "+",
    SubsuperscriptBox["T", RowBox[{"x", "+", "y"}], "4"], "+",
    SubscriptBox["R", "e"]}], TraditionalForm]], "InlineMath"],
 " done"
}]"#;
    assert_eq!(
      extract_cell_content(s),
      "R_(a+b)=R_(c,d)+T_(x+y)\u{2074}+R_e done"
    );
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
  fn test_extract_textdata_nested_inline_math_cells() {
    // A formula pasted into a formula leaves inline cells nested inside
    // each other (`Cell[BoxData[Cell[TextData[Cell[BoxData[…]]]]]]`). The
    // innermost typeset content must still render, not vanish and leave
    // the surrounding prose with just its trailing punctuation.
    let s = r#"TextData[{
 Cell[BoxData[Cell[TextData[Cell[BoxData[
   FormBox[
    RowBox[{
     UnderoverscriptBox["\[Sum]",
      RowBox[{"j", "=", "1"}], "m"],
     FractionBox[
      RowBox[{"cos", "(", RowBox[{"3", " ", "j", " ", "x"}], ")"}], "j"]}],
    TraditionalForm]], "InlineMath",ExpressionUUID->"a"]], "InlineMath",
   ExpressionUUID->"b"]],ExpressionUUID->"c"],
 ","
}]"#;
    assert_eq!(extract_cell_content(s), "\u{2211}_(j=1)^m cos(3 j x)/j,");
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
    assert_eq!(cell.content, "Integrate[(((x)^(2))+1), {x, 0, 2}]");
  }

  /// A coefficient ending in a named-character letter (`2 \[Pi]`, typeset
  /// as the bare glyph `π`) directly followed by the integral sign must
  /// not fuse into the single undefined symbol `πIntegrate` once the sign
  /// is replaced by the `Integrate[...]` call — `π` and `Integrate` need
  /// an inserted `*` between them, the same as any other juxtaposition
  /// that would otherwise merge two identifiers.
  #[test]
  fn integral_after_named_character_coefficient_keeps_them_apart() {
    let nb = r#"Notebook[{
Cell[BoxData[
 RowBox[{"2", "\[Pi]",
  SubsuperscriptBox["\[Integral]", "0", "h"],
  RowBox[{"x", RowBox[{"\[DifferentialD]", "x"}]}]}]], "Input"]
}]"#;
    let parsed = parse_notebook(nb).unwrap();
    let CellEntry::Single(cell) = &parsed.cells[0] else {
      panic!("expected a single cell");
    };
    assert_eq!(cell.content, "2\u{03c0}*Integrate[x, {x, 0, h}]");
    // Re-parsing the recovered source must evaluate the integral, not call
    // an undefined `πIntegrate` symbol.
    assert_eq!(
      crate::interpret(&format!("{} /. h -> 1", cell.content)).unwrap(),
      "Pi"
    );
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
      "RevolutionPlot3D[Piecewise[{{f[x+6], -6<=x<= -2}}],{x,-6,6}]"
    );
  }

  /// A nested row ending in the postfix `&` is a complete pure function: in
  /// the parent row `(f &) /@ list` must keep its grouping, since `&` binds
  /// looser than `/@` and would otherwise swallow `/@ list`.
  #[test]
  fn nested_pure_function_row_keeps_its_grouping() {
    let nb = r##"Notebook[{
Cell[BoxData[
 RowBox[{RowBox[{RowBox[{"f", "[", "#", "]"}], "&"}], "/@",
  RowBox[{"{", RowBox[{"1", ",", "2"}], "}"}]}]], "Input"]
}]"##;
    let parsed = parse_notebook(nb).unwrap();
    let CellEntry::Single(cell) = &parsed.cells[0] else {
      panic!("expected a single cell");
    };
    assert_eq!(cell.content, "(f[#]&)/@{1,2}");
  }

  /// A call whose argument-list row ends in a pure function
  /// (`Select[list, f &]`) must keep its comma-separated arguments: the row
  /// is an argument list, not a nested pure function to parenthesize, and
  /// `Select[(list, f &)]` is not valid syntax.
  #[test]
  fn call_with_trailing_pure_function_argument_is_not_parenthesized() {
    let nb = r##"Notebook[{
Cell[BoxData[
 RowBox[{"Select", "[",
  RowBox[{RowBox[{"p", "[", RowBox[{"[", "1", "]"}], "]"}], ",",
   RowBox[{RowBox[{"MemberQ", "[", RowBox[{"#", ",", "x"}], "]"}], "&"}]}],
  "]"}]], "Input"]
}]"##;
    let parsed = parse_notebook(nb).unwrap();
    let CellEntry::Single(cell) = &parsed.cells[0] else {
      panic!("expected a single cell");
    };
    assert_eq!(cell.content, "Select[p[[1]],MemberQ[#,x]&]");
  }

  /// A `Piecewise` typeset directly in a `RowBox` — the `\[Piecewise]`
  /// brace immediately followed by its `GridBox` of value/condition rows,
  /// with no outer wrapping `GridBox`/`TagBox` (the shape produced by an
  /// immediate `=` assignment, e.g. Wolfram's "Tax Rates and Tax Revenue"
  /// Demonstration's `totalTax[…] = \[Piecewise]…`) — must also come back
  /// as a `Piecewise[…]` call, not the raw grid rows as a bare nested list.
  #[test]
  fn piecewise_brace_in_a_bare_rowbox_becomes_a_piecewise_call() {
    let nb = r#"Notebook[{
Cell[BoxData[
 RowBox[{
  RowBox[{
   RowBox[{"f", "[", "z_", "]"}], "=",
   RowBox[{"\[Piecewise]", GridBox[{
      {"a", RowBox[{"z", ">", "0"}]},
      {"b", TagBox["True", "PiecewiseDefault", AutoDelete->True]}
     }]}]}], ";"}]], "Input"]
}]"#;
    let parsed = parse_notebook(nb).unwrap();
    let CellEntry::Single(cell) = &parsed.cells[0] else {
      panic!("expected a single cell");
    };
    assert_eq!(cell.content, "f[z_]=Piecewise[{{a, z>0}}, b];");
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
  fn test_saved_dump_drops_demonstrations_tools_context() {
    let dump = "DynamicModuleBox[{$CellContext`k$$ = 0}, \
      DynamicBox[…],\n\
      Initialization:>(f[DemonstrationsTools`x_] := \
      DemonstrationsTools`x^2; $CellContext`a = f[3])]";
    let init = extract_saved_initialization(dump).unwrap();
    assert_eq!(init, "f[x_] := x^2; a = f[3]");
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
    assert_eq!(init, "{a = 1, b = {2, 3}}; Typeset`initDone = True");
  }

  #[test]
  fn test_extract_saved_initialization_strips_control_suffix() {
    // A delayed helper reading a control (`take := Take[data, m$$]`) must
    // see the same plain name the reconstructed controls carry.
    let dump = "DynamicModuleBox[{$CellContext`m$$ = 3}, DynamicBox[…],\n\
      Initialization:>($CellContext`take := Take[$CellContext`data, \
      $CellContext`m$$]; $CellContext`x$7$$ = 1)]";
    let init = extract_saved_initialization(dump).unwrap();
    assert_eq!(init, "take := Take[data, m]; x$7 = 1");
  }

  #[test]
  fn test_extract_saved_initialization_keeps_control_namesakes_apart() {
    // `$CellContext`k` (a helper's own symbol) and the control
    // `$CellContext`k$$` are different symbols; a rule stored by the
    // Initialization must not read the control's value once it is set.
    let dump = "DynamicModuleBox[{$CellContext`k$$ = 3}, DynamicBox[…],\n\
      Initialization:>($CellContext`r = {p[Pattern[$CellContext`k, Blank[]]] \
      -> $CellContext`k}; $CellContext`s = $CellContext`k$$)]";
    let init = extract_saved_initialization(dump).unwrap();
    assert_eq!(init, "r = {p[Pattern[k$ctx, Blank[]]] -> k$ctx}; s = k");
    crate::interpret("k = 3").unwrap();
    assert_eq!(
      crate::interpret("r = {p[k$ctx_] -> k$ctx}; r").unwrap(),
      "{p[k$ctx_] -> k$ctx}"
    );
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
  fn test_extract_saved_manipulate_variables_prefers_outer_module_state() {
    // The outer `DynamicModuleBox[{…}, …]` initializer list is the
    // DynamicModule's live state (updated the instant a slider moves); the
    // inner compiled `ManipulateBoxes[…]`'s own `"Variables" :> {…}` clause
    // is only whatever was true when the box structure was last compiled,
    // and the two can disagree once a control moves without a recompile.
    // Regression: only the (stale) inner clause was read, so a Demonstration
    // saved with a control off its declared default reopened at the
    // default instead of where it was actually left.
    let dump = "DynamicModuleBox[{$CellContext`n$$ = 33.}, \
      DynamicBox[Manipulate`ManipulateBoxes[\n\
      1, StandardForm, \n\
      \"Variables\" :> {$CellContext`n$$ = 10}, \n\
      \"Body\" :> $CellContext`n$$, \n\
      \"Specifications\" :> {{$CellContext`n$$, 1, 100}}]]]";
    assert_eq!(
      extract_saved_manipulate_variables(dump),
      vec![("n".to_string(), "33.".to_string())]
    );
  }

  #[test]
  fn test_extract_saved_manipulate_variables_keeps_held_references() {
    // A reference to another variable is replaced by its saved value only
    // where it is evaluated; inside `Dynamic[…]` it is the variable itself.
    // Regression: `on$$ = False` inside a saved Dynamic became
    // `False = False` (Set::wrsym) in HowDoorbellsWork.
    let dump = "DynamicModuleBox[{$CellContext`x$$ = 0.7, \
      $CellContext`t$$ = 0.75/$CellContext`x$$, \
      $CellContext`obj$$ = {Dynamic[If[$CellContext`x$$ < 0.69, 1, \
      $CellContext`on$$ = False; \"x$$\"]], $CellContext`x$$}, \
      $CellContext`on$$ = False}, DynamicBox[…]]";
    assert_eq!(
      extract_saved_manipulate_variables(dump),
      vec![
        ("x".to_string(), "0.7".to_string()),
        ("t".to_string(), "0.75/0.7".to_string()),
        (
          "obj".to_string(),
          "{Dynamic[If[x < 0.69, 1, on = False; \"x$$\"]], 0.7}".to_string()
        ),
        ("on".to_string(), "False".to_string()),
      ]
    );
  }

  #[test]
  fn test_extract_saved_manipulate_variables_decodes_named_characters_in_strings()
   {
    // A saved string choice written with a named character
    // (`\[ShortRightArrow]`) must read back as the glyph the cell's own
    // source shows for it, or the restored control equals none of the
    // body's string comparisons and the widget renders nothing.
    // Other escapes inside the literal stay untouched.
    let dump = "DynamicModuleBox[{$CellContext`mode$$ = \
      \"a \\[ShortRightArrow] b\", $CellContext`t$$ = \
      \"q\\\"\\[NoSuchName]\"}, DynamicBox[…]]";
    assert_eq!(
      extract_saved_manipulate_variables(dump),
      vec![
        ("mode".to_string(), "\"a → b\"".to_string()),
        ("t".to_string(), "\"q\\\"\\[NoSuchName]\"".to_string()),
      ]
    );
  }

  #[test]
  fn test_extract_saved_manipulate_variables_joins_wrapped_values() {
    // The front end wraps long dumps mid-value (`Rational[23, 90] \n Pi`);
    // the line break must not split the value into two statements.
    let dump = "DynamicModuleBox[{$CellContext`a$$ = Rational[23, 90] \n    Pi, \
      $CellContext`s$$ = \"x\ny\"}, DynamicBox[…]]";
    assert_eq!(
      extract_saved_manipulate_variables(dump),
      vec![
        ("a".to_string(), "Rational[23, 90] Pi".to_string()),
        ("s".to_string(), "\"x\ny\"".to_string()),
      ]
    );
  }

  #[test]
  fn test_extract_saved_manipulate_variables_resolves_default_references() {
    // A control initialised from its own default variable, which is listed
    // later (and a chain through two defaults), must open on that default
    // rather than an unbound `nDef$$` symbol.
    let dump = "DynamicModuleBox[{$CellContext`n$$ = $CellContext`nDef$$, \
      $CellContext`nDef$$ = 2, $CellContext`k$$ = $CellContext`n$$, \
      $CellContext`z$$ = $CellContext`missing$$}, DynamicBox[…]]";
    assert_eq!(
      extract_saved_manipulate_variables(dump),
      vec![
        ("n".to_string(), "2".to_string()),
        ("nDef".to_string(), "2".to_string()),
        ("k".to_string(), "2".to_string()),
        ("z".to_string(), "missing$$".to_string()),
      ]
    );
  }

  #[test]
  fn test_extract_saved_manipulate_variables_resolves_embedded_references() {
    // A default computed from another control (`t = 0.75/f`) must open on
    // the computed value, not on an unbound `f$$` symbol.
    let dump = "DynamicModuleBox[{$CellContext`f$$ = 0.3, \
      $CellContext`t$$ = 0.75/$CellContext`f$$, \
      $CellContext`u$$ = 2 $CellContext`t$$ + $CellContext`gone$$}, \
      DynamicBox[…]]";
    assert_eq!(
      extract_saved_manipulate_variables(dump),
      vec![
        ("f".to_string(), "0.3".to_string()),
        ("t".to_string(), "0.75/0.3".to_string()),
        ("u".to_string(), "2 (0.75/0.3) + gone$$".to_string()),
      ]
    );
  }

  #[test]
  fn test_extract_saved_manipulate_variables_skips_controller_shadow_vars() {
    // `ControllerVariables` pairs a real control with a hidden
    // `$CellContext`name$digits$$` linking variable in the same outer
    // initializer list; only the real `$CellContext`name$$` entry (no
    // digit run before the closing `$$`) is a control's saved value.
    let dump = "DynamicModuleBox[{$CellContext`n$$ = 5, \
      Typeset`show$$ = True, $CellContext`n$79129$$ = 0}, \
      DynamicBox[…]]";
    assert_eq!(
      extract_saved_manipulate_variables(dump),
      vec![("n".to_string(), "5".to_string())]
    );
  }

  #[test]
  fn test_extract_saved_manipulate_variables_falls_back_without_module_box() {
    // A dump with no `DynamicModuleBox` wrapper at all (e.g. a
    // hand-assembled or otherwise unusual dump) still recovers values from
    // the inner `"Variables" :> {…}` clause.
    let dump = "\"Variables\" :> {$CellContext`a$$ = 7}";
    assert_eq!(
      extract_saved_manipulate_variables(dump),
      vec![("a".to_string(), "7".to_string())]
    );
  }

  #[test]
  fn test_extract_saved_manipulate_variables_strips_context_from_symbol_value()
  {
    // A saved default that is itself a bare symbol (a SetterBar choice
    // like `abc`, as opposed to a string or number) is dumped
    // context-qualified on *both* sides of the `=`: `$CellContext`style$$ =
    // $CellContext`abc`. Regression: only the name (`style`) was stripped
    // of the prefix; the value stayed `$CellContext`abc`, a symbol
    // distinct from the plain `abc` the live source's own spec (and any
    // helper function keyed on it, e.g. `lookup[abc] = …`) uses — so the
    // restored control could never match its own choice list.
    let dump = "DynamicModuleBox[{$CellContext`style$$ = $CellContext`abc}, \
      DynamicBox[…]]";
    assert_eq!(
      extract_saved_manipulate_variables(dump),
      vec![("style".to_string(), "abc".to_string())]
    );
  }

  #[test]
  fn test_reconstruct_manipulate_from_box_dump() {
    // A minimal stand-in for the `Manipulate`ManipulateBoxes[…]` structure
    // Wolfram embeds in a saved Output cell: enough surrounding box syntax
    // to exercise the extraction, with unrelated keys interleaved the way
    // the real dump orders "Variables"/"ControllerVariables" before "Body".
    let dump = "DynamicModuleBox[{$CellContext`a$$ = 2}, \
      DynamicBox[Manipulate`ManipulateBoxes[\n\
      1, StandardForm, \n\
      \"Variables\" :> {$CellContext`a$$ = 2}, \n\
      \"Body\" :> Plot[$CellContext`a$$*Sin[$CellContext`x], {$CellContext`x, 0, \
      2 Pi}], \n\
      \"Specifications\" :> {{{$CellContext`a$$, 2, \"amplitude\"}, 1, 5}}, \n\
      \"Options\" :> {}],\n\
      DynamicModuleValues:>{}]]";
    let src = reconstruct_manipulate_from_box_dump(dump).unwrap();
    assert_eq!(
      src,
      "Manipulate[Plot[a*Sin[x], {x, 0, 2 Pi}], {{a, 2, \"amplitude\"}, 1, 5}]"
    );
  }

  #[test]
  fn test_reconstruct_manipulate_from_box_dump_multiple_specs() {
    let dump = "DynamicModuleBox[{}, DynamicBox[Manipulate`ManipulateBoxes[\n\
      1, StandardForm, \n\
      \"Body\" :> $CellContext`f[$CellContext`n$$, $CellContext`m$$], \n\
      \"Specifications\" :> {{{$CellContext`n$$, 1, \"n\"}, 1, 10}, \
      {{$CellContext`m$$, 1, \"m\"}, {1, 2, 3}}}]]]";
    let src = reconstruct_manipulate_from_box_dump(dump).unwrap();
    assert_eq!(
      src,
      "Manipulate[f[n, m], {{n, 1, \"n\"}, 1, 10}, {{m, 1, \"m\"}, {1, 2, 3}}]"
    );
  }

  #[test]
  fn test_reconstruct_manipulate_from_box_dump_strips_uniquification_suffix() {
    // Regression: control names kept the `$$` suffix, so they showed as
    // `p1$$` and never matched the plain names of the saved variable values.
    let dump = "DynamicModuleBox[{}, DynamicBox[Manipulate`ManipulateBoxes[\n\
      1, StandardForm, \n\
      \"Body\" :> Graphics[Line[{$CellContext`p1$$, $CellContext`q$$}]], \n\
      \"Specifications\" :> {{{$CellContext`p1$$, {0, 1}}, {-1, -1}, {1, 1}, \
      ControlType -> None}}]]]";
    let src = reconstruct_manipulate_from_box_dump(dump).unwrap();
    assert_eq!(
      src,
      "Manipulate[Graphics[Line[{p1, q}]], {{p1, {0, 1}}, {-1, -1}, {1, 1}, ControlType -> None}]"
    );
  }

  #[test]
  fn test_reconstruct_manipulate_from_box_dump_absent() {
    assert_eq!(
      reconstruct_manipulate_from_box_dump(
        "DynamicModuleBox[{}, DynamicBox[…]]"
      ),
      None
    );
  }

  #[test]
  fn test_render_boxes_text_drops_number_precision_marks() {
    assert_eq!(
      render_boxes_text(r#"RowBox[{"0.2`", " ", "mm"}]"#),
      "0.2 mm"
    );
    assert_eq!(render_boxes_text(r#""1.5`20""#), "1.5");
    assert_eq!(render_boxes_text(r#""3``2""#), "3");
    // Not a number followed by a mark: left untouched.
    assert_eq!(render_boxes_text(r#""x`""#), "x`");
  }

  #[test]
  fn test_render_boxes_text_sqrt_parenthesizes_compound_radicand() {
    assert_eq!(render_boxes_text(r#"SqrtBox["x"]"#), "√x");
    assert_eq!(
      render_boxes_text(r#"SqrtBox[RowBox[{"a", "+", "b"}]]"#),
      "√(a+b)"
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
  fn test_linear_syntax_box_code_inside_group_text() {
    // Box code introduced by `\*` part-way through a `\(…\)` group, with
    // plain text before and after it, must not leak its source as text.
    let s = r"A: \!\(\*SubscriptBox[\(p\), \(asym\)]\) \!\(\*SuperscriptBox[\(e\), \(-\*SuperscriptBox[\((t/\*SubscriptBox[\(t\), \(c\)])\), \(m\)]\)]\)";
    let out = render_linear_syntax_escapes(s);
    assert!(!out.contains("Box"), "{out}");
    assert!(!out.contains('\\'), "{out}");
    assert!(out.starts_with("A: p_asym e"), "{out}");
  }

  #[test]
  fn test_linear_syntax_nested_group_script() {
    // A script written as nested groups (`\(\(12\)\( \)\)`, what the
    // FrontEnd saves for a padded index) must render like a plain script.
    let nested = render_linear_syntax_escapes(
      r"\!\(\*SubscriptBox[\(\[Sigma]\), \(\(12\)\( \)\)]\)= ",
    );
    let plain = render_linear_syntax_escapes(
      r"\!\(\*SubscriptBox[\(\[Sigma]\), \(12\)]\)= ",
    );
    assert!(!nested.contains('\\') && !nested.contains('"'), "{nested}");
    assert_eq!(nested.trim(), plain.trim());
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
  fn test_inline_math_template_box_row_with_linear_syntax_strings() {
    // A computed `Row[…]` built from `ToString[…, StandardForm]` pieces (a
    // Wolfram Demonstrations caption pattern) is stored as
    // `TemplateBox[{…}, "RowDefault"]` whose string slots carry the
    // FrontEnd's "linear syntax" box escapes (`\!\(\*SuperscriptBox[\(A\),
    // \(2\)]\)`) instead of real nested boxes. Both the template row and the
    // embedded escapes must render as ordinary display text, not leak as
    // literal `Row[...]` source or raw backslash-escape sequences.
    let nb = r#"Notebook[{
Cell[TextData[Cell[BoxData[
 FormBox[
  TemplateBox[{
    "\"2 \"", "\"(\"",
     "\"\\!\\(\\*SuperscriptBox[\\(A\\), \\(2\\)]\\)\"", "\"+\"",
     "\"\\!\\(\\*SuperscriptBox[\\(B\\), \\(2\\)]\\)\"", "\")\"", "\" = \"",
     SuperscriptBox["C", "2"]},
    "RowDefault"], TraditionalForm]], "InlineMath",ExpressionUUID->
  "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"]], "Text"]
}]"#;
    let parsed = parse_notebook(nb).unwrap();
    match &parsed.cells[0] {
      CellEntry::Single(cell) => {
        assert_eq!(cell.content, "2 (A\u{00b2}+B\u{00b2}) = C\u{00b2}");
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
  fn test_padded_script_needs_no_parentheses() {
    assert_eq!(inline_math_text(r#"SubscriptBox["a", "1 "]"#), "a_1");
    assert_eq!(inline_math_text(r#"SuperscriptBox["x", "n "]"#), "x^n");
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
      "Sum[((Sin[t])/(((n)^(2)))), {n, 1, m}]"
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
    // A `StyleBox` around the operator (`ScriptLevel -> 0`) is transparent.
    let s = r#"BoxData[RowBox[{StyleBox[UnderoverscriptBox["\[Sum]", RowBox[{"j", "=", "1"}], "2"], ScriptLevel->0], StyleBox[" ", ScriptLevel->0], RowBox[{"(", RowBox[{"t", "[", "j", "]"}], ")"}]}]]"#;
    assert_eq!(extract_cell_content(s), "Sum[ (t[j]), {j, 1, 2}]");
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

  /// An iconized list (`Iconize[{a, b, c}]`) is stored as an
  /// `InterpretationBox` whose meaning is the plain `{a, b, c}` list value.
  /// It must stay a list rather than being read as a row of boxes, which
  /// would fuse the items into a product.
  #[test]
  fn test_interpretation_box_list_meaning_stays_a_list() {
    let nb = r#"Notebook[{
Cell[BoxData[
 RowBox[{"Length", "[",
  InterpretationBox[
   DynamicModuleBox[{Typeset`open = False}, "placeholder"],
   {Graphics[{Circle[{0, 0}, 1]}], {1, 2}, "s"}], "]"}]], "Input"]
}]"#;
    let parsed = parse_notebook(nb).unwrap();
    match &parsed.cells[0] {
      CellEntry::Single(cell) => {
        assert_eq!(
          cell.content,
          "Length[{Graphics[{Circle[{0, 0}, 1]}], {1, 2}, \"s\"}]"
        );
      }
      CellEntry::Group(_) => panic!("Expected single cell"),
    }
  }

  /// A large `Compress`ed string the FrontEnd iconizes for display (shown
  /// as a "String length: … / Byte count: …" placeholder box) is still
  /// stored faithfully as the `InterpretationBox`'s meaning. Since that
  /// meaning is written as ordinary source rather than a box row, the bare
  /// string must round-trip as a quoted literal — the Demonstrations
  /// Project's `SaveDefinitions->True` Manipulates commonly hide their data
  /// this way (`drawfig = "1:eJzs…"`, later read back with
  /// `Uncompress[drawfig]`).
  #[test]
  fn test_interpretation_box_string_meaning_keeps_quotes() {
    let nb = r#"Notebook[{
Cell[BoxData[
 RowBox[{"drawfig", "=", " ",
  InterpretationBox[
   DynamicModuleBox[{Typeset`open = False}, "placeholder"],
   "1:eJzsdata=="]}]], "Input"]
}]"#;
    let parsed = parse_notebook(nb).unwrap();
    match &parsed.cells[0] {
      CellEntry::Single(cell) => {
        assert_eq!(cell.content, "drawfig= \"1:eJzsdata==\"");
      }
      CellEntry::Group(_) => panic!("Expected single cell"),
    }
  }

  /// `RowBox[{x, "/.", "0", "\[Rule]", y}]` is a replacement of the integer
  /// `0`; glued as `x/.0->y` it tokenized as division by the real `.0`.
  #[test]
  fn test_replace_all_followed_by_digit_keeps_operator() {
    let nb = r#"Notebook[{
Cell[BoxData[
 RowBox[{"{", RowBox[{"0", ",", "1"}], "}", "/.", "0", "\[Rule]", "\"\""}]], "Input"]
}]"#;
    let parsed = parse_notebook(nb).unwrap();
    match &parsed.cells[0] {
      CellEntry::Single(cell) => {
        assert!(cell.content.contains("/. 0"), "{}", cell.content);
        assert_eq!(crate::interpret(&cell.content).unwrap(), "{, 1}");
      }
      CellEntry::Group(_) => panic!("Expected single cell"),
    }
  }

  /// An `InterpretationBox` whose meaning is a bare list literal (not
  /// wrapped in `InputForm[…]`) must keep its elements as separate list
  /// items. This is the `IconizedObject` display Wolfram uses for a large
  /// stored array — e.g. a Demonstration's `SaveDefinitions->True` cache of
  /// a table built from mismatched-length rows. Routing the value through
  /// `extract_cell_content`'s "BoxData statement list" handling used to
  /// concatenate the two elements with no separator, so `{{1, 2, 3}, {4,
  /// 5}}` read back as `{1, 2, 3} {4, 5}` — implicit multiplication of two
  /// lists — and evaluating it raised `Thread::tdlen` for the length
  /// mismatch instead of yielding the original 2-element list.
  #[test]
  fn test_interpretation_box_list_meaning_keeps_list_structure() {
    let nb = r#"Notebook[{
Cell[BoxData[
 RowBox[{"table", "=",
  InterpretationBox[
   StyleBox["\"data\"", ShowStringCharacters->False],
   {{1, 2, 3}, {4, 5}}]}]], "Input"]
}]"#;
    let parsed = parse_notebook(nb).unwrap();
    let content = match &parsed.cells[0] {
      CellEntry::Single(cell) => cell.content.clone(),
      CellEntry::Group(_) => panic!("Expected single cell"),
    };
    assert_eq!(content, "table={{1, 2, 3}, {4, 5}}");
    assert_eq!(
      crate::interpret(&format!("{content};Length[table]")).unwrap(),
      "2"
    );
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

  #[test]
  fn test_gridbox_cell_script_letter_names_the_row_symbol() {
    // A script letter standing alone in a grid cell is the same symbol as
    // in a row: both must keep Wolfram's private-use code point, or the
    // `With` below substitutes into one occurrence and not the other.
    let nb = r#"Notebook[{
Cell[BoxData[RowBox[{"With", "[", RowBox[{RowBox[{"{", RowBox[{"\[ScriptM]", "=", "2"}], "}"}], ",", RowBox[{"Transpose", "[", GridBox[{{RowBox[{"1", "-", "\[ScriptM]"}], "\[ScriptM]"}}], "]"}]}], "]"}]], "Input"]
}]"#;
    let parsed = parse_notebook(nb).unwrap();
    match &parsed.cells[0] {
      CellEntry::Single(cell) => {
        assert_eq!(
          cell.content,
          "With[{\u{f6be}=2},Transpose[{{1-\u{f6be}, \u{f6be}}}]]"
        );
        assert_eq!(crate::interpret(&cell.content).unwrap(), "{{-1}, {2}}");
      }
      CellEntry::Group(_) => panic!("Expected single cell"),
    }
  }

  #[test]
  fn test_input_cell_matrix_wrapped_in_invisible_gridbox_brackets() {
    // The FrontEnd wraps a matrix literal that's an argument to a
    // function call in a pair of invisible `\:f3a2` bracket markers
    // alongside the visible `(` `)` — this is how a Wolfram Demonstrations
    // Project notebook's Initialization code typesets e.g.
    // `arrowHead = {Line[({{1, 2}, {3, 4}})]}` (independently written, not
    // copied from any specific Demonstration). The markers carry no
    // lexical meaning and must not leak into the reconstructed source.
    let nb = r#"Notebook[{
Cell[BoxData[RowBox[{"arrowHead", "=", RowBox[{"{", RowBox[{"Line", "[", RowBox[{"(", "\:f3a2", GridBox[{{"1", "2"}, {"3", "4"}}], "\:f3a2", ")"}], "]"}], "}"}]}]], "Input"]
}]"#;
    let parsed = parse_notebook(nb).unwrap();
    match &parsed.cells[0] {
      CellEntry::Single(cell) => {
        assert_eq!(cell.style, CellStyle::Input);
        assert_eq!(cell.content, "arrowHead={Line[({{1, 2}, {3, 4}})]}");
        assert!(
          !cell.content.contains('\u{f3a2}'),
          "the invisible bracket marker must not appear in the reconstructed \
           source: {:?}",
          cell.content
        );
        crate::interpret(&cell.content).expect(
          "the reconstructed source must be valid, evaluable Wolfram Language",
        );
      }
      CellEntry::Group(_) => panic!("Expected single cell"),
    }
  }

  #[test]
  fn top_level_assignment_is_detected_past_bracket_depth() {
    assert!(has_top_level_assignment("total=Length[list]"));
    assert!(has_top_level_assignment("total:=Length[list]"));
    assert!(has_top_level_assignment("total+=1"));
    // A comparison, not an assignment.
    assert!(!has_top_level_assignment("total==Length[list]"));
    assert!(!has_top_level_assignment("total<=3"));
    assert!(!has_top_level_assignment("total>=3"));
    assert!(!has_top_level_assignment("total!=3"));
    // An `=` nested inside a call/list is not at depth 0.
    assert!(!has_top_level_assignment("f[total=1]"));
    assert!(!has_top_level_assignment("{a==b}"));
    assert!(!has_top_level_assignment("Length[list]"));
  }

  #[test]
  fn juxtaposed_assignment_after_loop_keeps_its_own_grouping() {
    // As part of a scheduled QA routine, Woxi Studio was tested against a
    // randomly sampled Wolfram Demonstration notebook whose Initialization
    // code had two Module-body statements — a loop, then an assignment —
    // stacked on separate lines with the semicolon between them missing,
    // so the box tree holds only whitespace (`\[IndentingNewLine]`)
    // between the two `RowBox` siblings: `RowBox[{loop-box, …,
    // assignment-box}]`. Real Wolfram's box tree keeps the assignment as
    // its own self-contained node regardless — 2-D grouping is structural
    // (nesting), not textual — so it evaluates as `Times[loop, Set[var,
    // value]]`. Flattening the boxes to bare text without restoring that
    // grouping let the `=` escape its own box and bind to the *outer*
    // juxtaposition instead: the reparsed text read as `Set[Times[loop,
    // var], value]`, which fails outright (`Times` is Protected), so the
    // variable was never actually assigned — corrupting every later
    // statement that depended on it. This is a self-authored,
    // construct-equivalent example (an invented `Do`/`Length` loop, not
    // the specific Demonstration's code or data, which is copyrighted).
    let boxes = r#"RowBox[{RowBox[{"Do", "[", RowBox[{RowBox[{"n", "++"}], ",", RowBox[{"{", "3", "}"}]}], "]"}], "\[IndentingNewLine]", "\[IndentingNewLine]", RowBox[{"total", "=", RowBox[{"Length", "[", "list", "]"}]}]}]"#;
    let expr_src =
      box_source_to_expression(boxes).expect("box source must convert");
    assert!(
      expr_src.contains("(total=Length[list])"),
      "the assignment must stay parenthesized as its own factor, got: \
       {expr_src:?}"
    );

    let result =
      crate::interpret(&format!("n=0; list={{1,2,3}}; {expr_src}; total"))
        .expect("the reconstructed source must evaluate without error");
    assert_eq!(
      result, "3",
      "the assignment must actually run (not fail as Set::write on a \
       corrupted `Times[…]` target), got: {result:?} from {expr_src:?}"
    );
  }

  #[test]
  fn line_break_inside_a_row_multiplies_its_neighbours() {
    // Inside a row a "\n" is layout, so `0` and the assignment row after it
    // multiply, with the assignment kept as its own factor: wolframscript
    // reads `RowBox[{"0", "\n", RowBox[{"b", "=", …}]}]` as `0*(b = …)`
    // (SecondOrderReactionWithDiffusionInALiquidFilm relies on the
    // assignment running when the surrounding definition is used).
    let boxes = r#"RowBox[{RowBox[{"e", "[", "i_", "]"}], ":=", RowBox[{"x", "\[Equal]", RowBox[{"0", "\n", RowBox[{"b", "=", RowBox[{"a", "\[Equal]", "1"}]}]}]}]}]"#;
    let expr_src =
      box_source_to_expression(boxes).expect("box source must convert");
    let result = crate::interpret(&format!("{expr_src}; {{e[1], b}}")).unwrap();
    assert_eq!(result, "{x == 0, a == 1}", "from {expr_src:?}");
  }

  #[test]
  fn juxtaposed_semicolon_terminated_stmt_does_not_corrupt_next_assignment() {
    // As part of a scheduled QA routine, Woxi Studio was tested against a
    // randomly sampled Wolfram Demonstration notebook ("Neat Alternating
    // Tilings of the Plane") whose Initialization Code assigns several
    // module-level variables, each on its own line and each with its own
    // trailing `;`, separated only by blank lines (two stacked
    // `\[IndentingNewLine]` tokens, no literal operator) — the same
    // "2-D layout is the only grouping" idiom as the test above, except
    // here the *juxtaposed* statement is itself already `;`-terminated,
    // e.g. `RowBox[{RowBox[{"is0", "=", …}], ";"}]`. The existing fix
    // wraps such a statement in parens to keep its own `=` from binding to
    // the *outer* juxtaposition — but it wrapped the trailing `;` along
    // with it (`(is0 = …;)`), making the *whole parenthesized group's own
    // value* `Null` (a bare `;` inside `(…)` always adds an implicit
    // trailing `Null`). That `Null` was still one un-terminated factor of
    // an implicit product, free to keep juxtaposing forward — so the
    // *next* blank-line-separated statement's `=` bound to `Null * that
    // statement's own left-hand side` instead of to the statement alone,
    // failing with `Set::write` on the Protected `Times` head and never
    // assigning it. This is a self-authored, construct-equivalent example
    // (invented variable names/values, not the specific Demonstration's
    // code or data, which is copyrighted).
    let boxes = r#"RowBox[{RowBox[{RowBox[{"a", "=", "1"}], ";"}], "\[IndentingNewLine]", "\[IndentingNewLine]", RowBox[{RowBox[{"b", "=", "2"}], ";"}], "\[IndentingNewLine]", "\[IndentingNewLine]", RowBox[{"total", "=", "b"}]}]"#;
    let expr_src =
      box_source_to_expression(boxes).expect("box source must convert");
    assert!(
      expr_src.contains("(b=2);"),
      "the semicolon-terminated statement's own `;` must stay outside the \
       added parens, got: {expr_src:?}"
    );

    let result = crate::interpret(&expr_src)
      .expect("the reconstructed source must evaluate without error");
    assert_eq!(
      result, "2",
      "the final assignment must actually run (not fail as Set::write on \
       a corrupted `Times[…]` target carrying a stray `Null` factor from \
       the previous statement), got: {result:?} from {expr_src:?}"
    );
  }

  #[test]
  fn typeset_association_delimiters_convert_to_ascii() {
    // The FrontEnd encodes `<|a -> 1|>` as `\[LeftAssociation]` …
    // `\[RightAssociation]` (private-use U+F113/U+F114) inside RowBox.
    let boxes = r#"RowBox[{"assoc", "=", RowBox[{"\[LeftAssociation]", RowBox[{RowBox[{"\"k\"", "\[Rule]", "1"}], ",", RowBox[{"\"m\"", "\[Rule]", RowBox[{"\[LeftAssociation]", RowBox[{"\"n\"", "\[Rule]", "2"}], "\[RightAssociation]"}]}]}], "\[RightAssociation]"}]}]"#;
    let expr_src =
      box_source_to_expression(boxes).expect("box source must convert");
    assert!(expr_src.contains("<|"), "got: {expr_src:?}");
    let result = crate::interpret(&format!("{expr_src}; assoc[\"m\"][\"n\"]"))
      .expect("the reconstructed source must evaluate without error");
    assert_eq!(result, "2");
  }

  #[test]
  fn wrapped_statement_ending_in_an_open_span_keeps_both_semicolons() {
    // Regression for a review finding on the fix above: a trailing `;` is
    // not always a statement terminator that needs relocating outside the
    // added parens — it can also be the second character of the `;;`
    // `Span` operator. `rng = 2;;` (`Set[rng, Span[2, All]]`) juxtaposed
    // against a preceding statement via the same "blank lines, no visible
    // operator" idiom flattens to the piece `"rng=2;;"`. Treating its last
    // `;` as a lone terminator (stripping exactly one) would leave
    // `"rng=2;"` wrapped as `(rng=2;)` — a *different*, silently wrong
    // expression (`Set[rng, 2]`, plain `2` instead of `Span[2, All]`) once
    // the remaining `;` is read as that piece's own terminator. Since both
    // `;` here belong to `Span`, neither should move: the whole piece
    // wraps unchanged, `(rng=2;;)`. This is a self-authored,
    // construct-equivalent example (an invented `Do`/`Span` assignment,
    // not the specific Demonstration's code or data, which is
    // copyrighted).
    let boxes = r#"RowBox[{RowBox[{"Do", "[", RowBox[{RowBox[{"n", "++"}], ",", RowBox[{"{", "3", "}"}]}], "]"}], "\[IndentingNewLine]", "\[IndentingNewLine]", RowBox[{"rng", "=", RowBox[{"2", ";", ";"}]}]}]"#;
    let expr_src =
      box_source_to_expression(boxes).expect("box source must convert");
    assert!(
      expr_src.contains("(rng=2;;)"),
      "both semicolons of the open `Span` must stay together inside the \
       added parens, got: {expr_src:?}"
    );

    let result = crate::interpret(&format!("n=0; {expr_src}; Head[rng]"))
      .expect("the reconstructed source must evaluate without error");
    assert_eq!(
      result, "Span",
      "rng must stay a `Span[2, All]`, not get silently truncated to the \
       plain integer `2`, got: {result:?} from {expr_src:?}"
    );
  }

  #[test]
  fn terminated_nested_row_followed_by_bare_semicolon_is_not_span() {
    // A statement box that already carries its own `;` (`RowBox[{stmt,
    // ";"}]`), wrapped in an outer row with one more bare `";"`, is a
    // statement followed by an empty one — not the `;;` `Span` operator
    // that the two glued semicolons would read back as. Otherwise the
    // assigned value becomes `Span[value, All]`.
    let boxes = r#"RowBox[{RowBox[{RowBox[{"vals", "=", RowBox[{"{", RowBox[{"1", ",", "2", ",", "3"}], "}"}]}], ";"}], ";"}]"#;
    let expr_src =
      box_source_to_expression(boxes).expect("box source must convert");
    assert!(
      !expr_src.contains(";;"),
      "the two semicolons must stay apart, got: {expr_src:?}"
    );
    let result = crate::interpret(&format!("{expr_src}\nHead[vals]"))
      .expect("the reconstructed source must evaluate without error");
    assert_eq!(result, "List", "got: {result:?} from {expr_src:?}");
  }

  #[test]
  fn assignment_argument_before_a_sibling_arg_is_not_over_wrapped() {
    // As part of a scheduled QA routine, Woxi Studio was tested against a
    // randomly sampled Wolfram Demonstration notebook ("Intersecting Lines
    // in All Possible Ways") whose `Manipulate` body is `points =
    // Table[…]; LocatorPane[…]` — an assignment followed by a semicolon,
    // both inside the *first* comma-separated argument of an enclosing
    // call, with a sibling argument (the control specs) following. The
    // `has_real_predecessor` check that decides whether a nested `RowBox`
    // needs re-parenthesizing (see
    // `juxtaposed_assignment_after_loop_keeps_its_own_grouping` above)
    // treated the "[" opening the call's own argument list as a "real"
    // preceding sibling expression, so the whole comma-separated argument
    // row — not just the assignment's own factor — was wrapped: reparsing
    // `f[points = Table[…]; body, {ctrl, …}]` as
    // `f[(points = Table[…]; body, {ctrl, …})]` puts a bare comma directly
    // inside a parenthesized group, which is a parse error. This is a
    // self-authored, construct-equivalent example (an invented `f`/`g`
    // call, not the specific Demonstration's code, which is copyrighted).
    let boxes = r#"RowBox[{"f", "[", RowBox[{RowBox[{RowBox[{"a", "=", "1"}], ";", RowBox[{"g", "[", "a", "]"}]}], ",", RowBox[{"{", RowBox[{"b", ",", "1", ",", "2"}], "}"}]}], "]"}]"#;
    let expr_src =
      box_source_to_expression(boxes).expect("box source must convert");
    assert!(
      !expr_src.contains('('),
      "a comma-delimited argument needs no extra parentheses around its \
       own assignment, got: {expr_src:?}"
    );

    let result =
      crate::interpret(&format!("g[x_]:=x+1; f[x_,y_]:={{x,y}}; {expr_src}"))
        .expect(
          "the reconstructed source must parse and evaluate without error",
        );
    assert_eq!(
      result, "{2, {b, 1, 2}}",
      "the assignment must run as part of the first argument and the \
       second argument must stay independent, got: {result:?} from \
       {expr_src:?}"
    );
  }

  #[test]
  fn same_q_and_unsame_q_are_not_top_level_assignments() {
    // `===` (SameQ) and `=!=` (UnsameQ) are three-character comparison
    // operators, not assignments. The two-character `==` check consumed
    // only the first pair of `=` in `===`, leaving its third `=` to be
    // misread as a fresh (assignment) `=`; `=!=` was not recognized at all
    // (its leading `=` is followed by `!`, not `=`), so it fell straight
    // through to the "bare assignment" case.
    assert!(!has_top_level_assignment("a===b"));
    assert!(!has_top_level_assignment("a=!=b"));
    assert!(!has_top_level_assignment("Last[l]===First[l]"));
    assert!(!has_top_level_assignment("Round[sum]=!=0"));
    // A real assignment following one of these must still be detected.
    assert!(has_top_level_assignment("(a===b);total=1"));
  }

  #[test]
  fn same_q_argument_list_keeps_its_grouping() {
    // As part of a scheduled QA routine, Woxi Studio was tested against a
    // randomly sampled Wolfram Demonstration notebook ("Color Blindness")
    // whose Manipulate Initialization defines a point-in-polygon helper
    // that closes an open path with `If[Last[l]===First[l], l,
    // Append[l, l[[1]]]]`. Flattening the box tree misdetected the `===`
    // as a top-level assignment (see `same_q_and_unsame_q_are_not_top_
    // level_assignments`) and wrapped the whole three-argument `If[…]` row
    // in a stray `(…)`, turning it into `If[(a===b,l,Append[…])]` — invalid
    // syntax, since parentheses cannot hold a comma-separated list — which
    // failed to parse at all. This is a self-authored, construct-equivalent
    // example (an invented list-closing helper, not the Demonstration's
    // own code or data, which is copyrighted).
    let boxes = r#"RowBox[{"If", "[", RowBox[{RowBox[{RowBox[{"Last", "[", "l", "]"}], "===", RowBox[{"First", "[", "l", "]"}]}], ",", "l", ",", RowBox[{"Append", "[", RowBox[{"l", ",", RowBox[{"l", "[", RowBox[{"[", "1", "]"}], "]"}]}], "]"}]}], "]"}]"#;
    let expr_src =
      box_source_to_expression(boxes).expect("box source must convert");
    assert!(
      !expr_src.contains('('),
      "a plain If[a===b, x, y] needs no extra parens, got: {expr_src:?}"
    );

    let closed =
      crate::interpret(&format!("l={{1,2,3,1}}; {expr_src}")).unwrap();
    let open = crate::interpret(&format!("l={{1,2,3}}; {expr_src}")).unwrap();
    assert_eq!(closed, "{1, 2, 3, 1}", "from {expr_src:?}");
    assert_eq!(open, "{1, 2, 3, 1}", "from {expr_src:?}");
  }

  #[test]
  fn unsame_q_condition_keeps_its_grouping() {
    // The same `===`/`=!=` misdetection also broke a bare `x=!=y` condition
    // used directly as a function argument (the same Demonstration's
    // winding-number test ends in `Round[sum]=!=0`), wrapping it in a
    // stray `(…)` around the whole surrounding argument list.
    let boxes = r#"RowBox[{"Not", "[", RowBox[{RowBox[{"Round", "[", "sum", "]"}], "=!=", "0"}], "]"}]"#;
    let expr_src =
      box_source_to_expression(boxes).expect("box source must convert");
    assert!(!expr_src.contains('('), "got: {expr_src:?}");
    assert_eq!(
      crate::interpret(&format!("sum=0.4; {expr_src}")).unwrap(),
      "True",
      "from {expr_src:?}"
    );
  }

  #[test]
  fn comma_separated_binding_does_not_gain_stray_parens() {
    // As part of the same QA pass, the Demonstration's Manipulate also
    // binds two variables in a single `With[{a = …, b = …}, body]`, each
    // assignment box a sibling of the other separated by a `","` display
    // token. `has_real_predecessor` treated that comma (and the `"{"`
    // opening the binding list) as "a real expression already precedes
    // this", the same condition `juxtaposed_assignment_after_loop_keeps_
    // its_own_grouping` guards against — so the second binding's `RowBox`
    // was wrapped in a stray `(…)`, and the enclosing `{…}` list (now
    // holding what the flattener saw as one comma-holding parenthesized
    // item) was wrapped in another, producing invalid `{(a = …,(b =
    // …))}` syntax that failed to parse. This is a self-authored,
    // construct-equivalent example, not the Demonstration's own code.
    let boxes = r#"RowBox[{"With", "[", RowBox[{RowBox[{"{", RowBox[{RowBox[{"a", "=", "1"}], ",", "\[IndentingNewLine]", RowBox[{"b", "=", "2"}]}], "}"}], ",", RowBox[{"a", "+", "b"}]}], "]"}]"#;
    let expr_src =
      box_source_to_expression(boxes).expect("box source must convert");
    assert!(
      !expr_src.contains('('),
      "a plain {{a=1,b=2}} binding list needs no extra parens, got: {expr_src:?}"
    );
    assert_eq!(
      crate::interpret(&expr_src).unwrap(),
      "3",
      "from {expr_src:?}"
    );
  }

  #[test]
  fn module_variable_list_assignments_are_not_wrapped_together() {
    // As part of the same scheduled QA routine, a much larger Wolfram
    // Demonstration notebook's `Module[{var1 = val1, var2 = val2, ...}, …]`
    // local-variable list (many `var = val` items, each a comma-separated
    // sibling `RowBox` — the ordinary box shape for such a list) reconstructed
    // as one invalid `(var1 = val1, var2 = val2, ...)` group: every item after
    // the first got parenthesized *together* with everything before it,
    // because `has_real_predecessor` treated the preceding bare `","` (an
    // explicit separator, not implicit multiplication) as a "real" sibling
    // needing juxtaposition protection. The resulting `(a, b)` — a comma
    // inside plain parens, never valid Wolfram syntax outside a list or call
    // — made the whole cell unparseable. This is a self-authored,
    // construct-equivalent example (an invented three-variable `Module`, not
    // the specific Demonstration's code, which is copyrighted).
    let boxes = r#"RowBox[{"Module", "[", RowBox[{RowBox[{"{", RowBox[{RowBox[{"a", "=", "1"}], ",", RowBox[{"b", "=", "2"}], ",", "c"}], "}"}], ",", RowBox[{"a", "+", "b"}]}], "]"}]"#;
    let expr_src =
      box_source_to_expression(boxes).expect("box source must convert");
    assert_eq!(
      expr_src, "Module[{a=1,b=2,c},a+b]",
      "each list item must stay its own factor, with no extra grouping \
       parens around any of them, got: {expr_src:?}"
    );
    let result = crate::interpret(&expr_src)
      .expect("the reconstructed source must evaluate without error");
    assert_eq!(result, "3");

    // The same bug hit a `RowBox` sibling directly inside a function-call
    // bracket too — e.g. `With[{v = 1}, body]`'s `body`, which sits right
    // after the argument list's own `,` — since a bare `"["`/`"{"`/`"("`
    // opening delimiter was *also* treated as a "real" predecessor.
    let with_boxes = r#"RowBox[{"With", "[", RowBox[{RowBox[{"{", RowBox[{"v", "=", "1"}], "}"}], ",", RowBox[{"v", "+", "1"}]}], "]"}]"#;
    let with_src =
      box_source_to_expression(with_boxes).expect("box source must convert");
    assert_eq!(with_src, "With[{v=1},v+1]", "got: {with_src:?}");
    assert_eq!(crate::interpret(&with_src).unwrap(), "2");
  }

  #[test]
  fn reconstructed_manipulate_strips_suffix_from_named_character_variables() {
    // A control whose variable is a Greek letter is dumped as
    // `\[Alpha]$$`; the uniquification suffix used to survive because the
    // character before `$$` is the escape's closing bracket, leaving the
    // control labelled `α$$`.
    let dump = "DynamicModuleBox[{}, DynamicBox[Manipulate`ManipulateBoxes[1, \
      StandardForm, \"Body\" :> $CellContext`\\[Alpha]$$ + $CellContext`k$$, \
      \"Specifications\" :> {{{$CellContext`\\[Alpha]$$, {0.2, 0}}, {-1, -1}, \
      {1, 1}}, {$CellContext`k$$, 1, 5}}]]]";
    let src = reconstruct_manipulate_from_box_dump(dump).unwrap();
    assert!(!src.contains("$$"), "got: {src}");
    assert!(src.contains("\\[Alpha]"), "got: {src}");
  }
  /// Regression: a documentation hyperlink pasted into code
  /// (`ButtonBox["Name", BaseStyle -> "Link", ButtonData -> "paclet:…"]`)
  /// leaked its raw box syntax into the cell source, so the call head was
  /// unparseable. It must read as its label.
  #[test]
  fn buttonbox_hyperlink_inside_code_row_reads_as_label() {
    let boxes = r#"RowBox[{RowBox[{ButtonBox["Sqrt",
       BaseStyle->"Link",
       ButtonData->"paclet:ref/Sqrt"], "[", "16", "]"}], ";", "x"}]"#;
    let src = box_source_to_expression(boxes).expect("box source converts");
    assert!(!src.contains("ButtonBox"), "got: {src:?}");
    assert_eq!(crate::interpret(&src).unwrap(), "x");
  }

  /// `ⅆx` written with an `\[InvisibleSpace]` between the glyph and the
  /// variable still closes the integral body, so the cell stays parseable
  /// instead of leaving a bare `Subscript[∫, 0]`.
  #[test]
  fn integral_differential_with_invisible_space_names_its_variable() {
    let nb = r#"Notebook[{
Cell[BoxData[
 RowBox[{
  SubsuperscriptBox["\[Integral]", "0", "1"],
  RowBox[{
   RowBox[{"(", RowBox[{"2", "-", "x"}], ")"}],
   RowBox[{"\[DifferentialD]", "\[InvisibleSpace]", "x"}]}]}]], "Input"]
}]"#;
    let parsed = parse_notebook(nb).unwrap();
    let CellEntry::Single(cell) = &parsed.cells[0] else {
      panic!("expected a single cell");
    };
    assert_eq!(cell.content, "Integrate[(2-x), {x, 0, 1}]");
  }

  /// A row that ends in a statement `;` and is the operand of a following
  /// operator keeps its grouping: `(s = expr;) /. para`.
  #[test]
  fn semicolon_terminated_row_before_an_operator_is_parenthesized() {
    let s = r#"BoxData[RowBox[{RowBox[{RowBox[{"s", "=", "x"}], ";"}], "/.", "para"}]]"#;
    assert_eq!(extract_cell_content(s), "(s=x;)/.para");
  }

  /// Operator tokens from neighbouring boxes are kept apart so they do not
  /// read back as one longer operator (Demonstration CoulombWavefunctions:
  /// `(view!) == "complex"` became the unparsable `view!=="complex"`).
  #[test]
  fn neighbouring_operator_tokens_do_not_fuse() {
    let s = r#"BoxData[RowBox[{RowBox[{"view", "!"}], "\[Equal]", "\"\<complex\>\""}]]"#;
    let text = extract_cell_content(s);
    assert_eq!(text, "view! ==\"complex\"");
    let s = r#"BoxData[RowBox[{"x", "-", RowBox[{"-", "1"}]}]]"#;
    assert_eq!(extract_cell_content(s), "x- -1");
  }
}
