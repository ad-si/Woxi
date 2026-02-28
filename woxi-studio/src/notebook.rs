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
}

/// The style/type of a cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellStyle {
  Title,
  Subtitle,
  Section,
  Subsection,
  Subsubsection,
  Text,
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
      "Section" => Some(Self::Section),
      "Subsection" => Some(Self::Subsection),
      "Subsubsection" => Some(Self::Subsubsection),
      "Text" => Some(Self::Text),
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
      Self::Section => "Section",
      Self::Subsection => "Subsection",
      Self::Subsubsection => "Subsubsection",
      Self::Text => "Text",
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
  let trimmed = input.trim();

  // The file should start with Notebook[{
  let inner = strip_wrapper(trimmed, "Notebook")
    .ok_or("Expected Notebook[{...}] wrapper")?;

  let cells = parse_cell_list(inner)?;
  Ok(Notebook { cells })
}

/// Strip `Name[{ ... }]` and return the inner content.
fn strip_wrapper<'a>(
  s: &'a str,
  name: &str,
) -> Option<&'a str> {
  let s = s.trim();
  let rest = s.strip_prefix(name)?;
  let rest = rest.trim();
  let rest = rest.strip_prefix('[')?;
  let rest = rest.trim();
  let rest = rest.strip_prefix('{')?;

  // Find the matching closing `}]` from the end.
  let rest_end = rest.trim_end();
  let rest_end = rest_end.strip_suffix(']')?;
  let rest_end = rest_end.trim_end();
  let rest_end = rest_end.strip_suffix('}')?;
  Some(rest_end)
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
  if inner_trimmed.starts_with("CellGroupData[") {
    return parse_cell_group(inner_trimmed);
  }

  // Regular cell: Cell["content", "Style"]
  let cell = parse_single_cell(inner_trimmed)?;
  Ok(CellEntry::Single(cell))
}

/// Parse CellGroupData[{cells...}, Open|Closed]
fn parse_cell_group(s: &str) -> Result<CellEntry, String> {
  let rest = s
    .strip_prefix("CellGroupData[")
    .ok_or("Expected CellGroupData[")?;
  let rest = rest
    .strip_suffix(']')
    .ok_or("Missing closing ] for CellGroupData")?;

  // Find the { ... } cell list and the Open/Closed flag.
  let rest = rest.trim();
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
fn parse_single_cell(s: &str) -> Result<Cell, String> {
  // Split on the last comma at top level to get the style
  let parts = split_top_level_commas(s);
  if parts.len() < 2 {
    // Try to handle cells with just content
    let content = extract_string_content(s);
    return Ok(Cell {
      style: CellStyle::Text,
      content,
    });
  }

  let style_str = parts.last().unwrap().trim();
  let style_str = style_str
    .trim_matches('"')
    .trim();
  let style =
    CellStyle::from_str(style_str).unwrap_or(CellStyle::Text);

  // Join all parts except the last as the content
  let content_parts = &parts[..parts.len() - 1];
  let raw_content = content_parts.join(",");
  let content = extract_cell_content(&raw_content);

  Ok(Cell { style, content })
}

/// Extract cell content from BoxData[...] or a quoted string.
fn extract_cell_content(s: &str) -> String {
  let s = s.trim();

  // Handle BoxData[RowBox[{...}]]  or BoxData["..."]
  if s.starts_with("BoxData[") {
    let inner = &s[8..];
    let inner = inner
      .strip_suffix(']')
      .unwrap_or(inner);
    return extract_cell_content(inner);
  }

  // Handle RowBox[{"...", ...}]
  if s.starts_with("RowBox[") {
    let inner = &s[7..];
    let inner = inner
      .strip_suffix(']')
      .unwrap_or(inner);
    return extract_rowbox_content(inner);
  }

  // Handle quoted strings
  extract_string_content(s)
}

/// Extract text from a RowBox expression by concatenating string
/// elements.
fn extract_rowbox_content(s: &str) -> String {
  let s = s.trim();
  let s = s.strip_prefix('{').unwrap_or(s);
  let s = s.strip_suffix('}').unwrap_or(s);

  let parts = split_top_level_commas(s);
  let mut result = String::new();
  for part in parts {
    let part = part.trim();
    if part.starts_with('"') && part.ends_with('"') && part.len() >= 2
    {
      result.push_str(&unescape_string(&part[1..part.len() - 1]));
    } else if part.starts_with("RowBox[") {
      result.push_str(&extract_rowbox_content(
        &part[7..part.len().saturating_sub(1)],
      ));
    } else if part == "\"\\n\"" || part == "\"\\[NewLine]\"" {
      result.push('\n');
    } else {
      // For non-string tokens, include as-is
      result.push_str(part);
    }
  }
  result
}

/// Extract a plain string value, handling escaped quotes.
fn extract_string_content(s: &str) -> String {
  let s = s.trim();
  if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
    unescape_string(&s[1..s.len() - 1])
  } else {
    s.to_string()
  }
}

/// Unescape Wolfram-style string escapes.
fn unescape_string(s: &str) -> String {
  let mut result = String::with_capacity(s.len());
  let mut chars = s.chars();
  while let Some(c) = chars.next() {
    if c == '\\' {
      match chars.next() {
        Some('n') => result.push('\n'),
        Some('t') => result.push('\t'),
        Some('\\') => result.push('\\'),
        Some('"') => result.push('"'),
        Some('[') => {
          // Wolfram special character like \[Alpha]
          let mut name = String::new();
          for ch in chars.by_ref() {
            if ch == ']' {
              break;
            }
            name.push(ch);
          }
          // Keep as \[Name] for now
          result.push_str(&format!("\\[{name}]"));
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

/// Find the matching `}` for content that starts right after `{`.
/// Returns (content_inside_braces, remainder_after_brace).
fn find_matching_brace(
  s: &str,
) -> Result<(&str, &str), String> {
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
      CellEntry::Single(cell) => write!(f, "{cell}"),
      CellEntry::Group(group) => write!(f, "{group}"),
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
    write!(
      f,
      "}}, {}]]",
      if self.open { "Open" } else { "Closed" }
    )
  }
}

impl fmt::Display for Cell {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self.style {
      CellStyle::Input | CellStyle::Code => {
        // For input cells, wrap content in BoxData
        let lines: Vec<&str> =
          self.content.lines().collect();
        if lines.len() <= 1 {
          write!(
            f,
            "Cell[BoxData[\"{}\"], \"{}\"]",
            escape_string(&self.content),
            self.style
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
          write!(f, "}}]], \"{}\"]", self.style)
        }
      }
      CellStyle::Output | CellStyle::Print => {
        write!(
          f,
          "Cell[BoxData[\"{}\"], \"{}\"]",
          escape_string(&self.content),
          self.style
        )
      }
      _ => {
        // Text-style cells: Cell["content", "Style"]
        write!(
          f,
          "Cell[\"{}\", \"{}\"]",
          escape_string(&self.content),
          self.style
        )
      }
    }
  }
}

// ── Convenience constructors ────────────────────────────────────────

impl Notebook {
  /// Create an empty notebook.
  pub fn new() -> Self {
    Notebook { cells: Vec::new() }
  }

  /// Add a single cell at the end.
  pub fn push_cell(&mut self, cell: Cell) {
    self.cells.push(CellEntry::Single(cell));
  }

  /// Add a cell group (e.g. input + output pair).
  #[allow(dead_code)]
  pub fn push_group(&mut self, cells: Vec<Cell>) {
    self.cells.push(CellEntry::Group(CellGroup {
      cells,
      open: true,
    }));
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
    Cell {
      style,
      content: content.into(),
    }
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
      _ => panic!("Expected single cell"),
    }

    match &parsed.cells[1] {
      CellEntry::Single(cell) => {
        assert_eq!(cell.style, CellStyle::Text);
        assert_eq!(cell.content, "Some explanation");
      }
      _ => panic!("Expected single cell"),
    }

    match &parsed.cells[2] {
      CellEntry::Single(cell) => {
        assert_eq!(cell.style, CellStyle::Input);
        assert_eq!(cell.content, "1 + 1");
      }
      _ => panic!("Expected single cell"),
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
      _ => panic!("Expected cell group"),
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
}
