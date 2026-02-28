//! Simple syntax highlighter for Wolfram Language.

use iced::advanced::text::highlighter::{self, Highlighter};
use iced::{Color, Font, Theme};
use std::ops::Range;

#[derive(Debug, Clone, PartialEq)]
pub struct WolframSettings {
  pub enabled: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum WolframHighlight {
  Comment,
  String,
  Number,
  Function,
  Keyword,
  Operator,
  Pattern,
  Normal,
}

pub struct WolframHighlighter {
  enabled: bool,
  current_line: usize,
}

impl Highlighter for WolframHighlighter {
  type Settings = WolframSettings;
  type Highlight = WolframHighlight;
  type Iterator<'a> = std::vec::IntoIter<(Range<usize>, WolframHighlight)>;

  fn new(settings: &Self::Settings) -> Self {
    WolframHighlighter {
      enabled: settings.enabled,
      current_line: 0,
    }
  }

  fn update(&mut self, new_settings: &Self::Settings) {
    self.enabled = new_settings.enabled;
  }

  fn change_line(&mut self, line: usize) {
    if self.current_line > line {
      self.current_line = line;
    }
  }

  fn highlight_line(&mut self, line: &str) -> Self::Iterator<'_> {
    self.current_line += 1;
    if !self.enabled || line.is_empty() {
      return Vec::new().into_iter();
    }
    tokenize_line(line).into_iter()
  }

  fn current_line(&self) -> usize {
    self.current_line
  }
}

pub fn format_highlight(
  highlight: &WolframHighlight,
  theme: &Theme,
) -> highlighter::Format<Font> {
  let is_dark = !matches!(theme, Theme::Light);

  let color = match highlight {
    WolframHighlight::Normal => {
      return highlighter::Format::default();
    }
    WolframHighlight::Comment => {
      if is_dark {
        Color::from_rgb(0.42, 0.45, 0.49)
      } else {
        Color::from_rgb(0.42, 0.45, 0.49)
      }
    }
    WolframHighlight::String => {
      if is_dark {
        Color::from_rgb(0.60, 0.76, 0.47)
      } else {
        Color::from_rgb(0.31, 0.63, 0.31)
      }
    }
    WolframHighlight::Number => {
      if is_dark {
        Color::from_rgb(0.82, 0.60, 0.40)
      } else {
        Color::from_rgb(0.60, 0.41, 0.00)
      }
    }
    WolframHighlight::Function => {
      if is_dark {
        Color::from_rgb(0.38, 0.69, 0.94)
      } else {
        Color::from_rgb(0.25, 0.47, 0.95)
      }
    }
    WolframHighlight::Keyword => {
      if is_dark {
        Color::from_rgb(0.78, 0.47, 0.87)
      } else {
        Color::from_rgb(0.65, 0.15, 0.64)
      }
    }
    WolframHighlight::Operator => {
      if is_dark {
        Color::from_rgb(0.34, 0.71, 0.76)
      } else {
        Color::from_rgb(0.00, 0.52, 0.74)
      }
    }
    WolframHighlight::Pattern => {
      if is_dark {
        Color::from_rgb(0.88, 0.42, 0.46)
      } else {
        Color::from_rgb(0.89, 0.34, 0.29)
      }
    }
  };

  highlighter::Format {
    color: Some(color),
    font: None,
  }
}

fn is_keyword(word: &str) -> bool {
  matches!(
    word,
    "True"
      | "False"
      | "None"
      | "Null"
      | "All"
      | "Infinity"
      | "Pi"
      | "E"
      | "I"
      | "Return"
      | "Break"
      | "Continue"
      | "Throw"
      | "Catch"
      | "Module"
      | "Block"
      | "With"
      | "If"
      | "Which"
      | "Switch"
      | "Do"
      | "For"
      | "While"
      | "Table"
      | "Function"
      | "Set"
      | "SetDelayed"
      | "Rule"
      | "RuleDelayed"
      | "Map"
      | "Apply"
      | "Select"
      | "CompoundExpression"
  )
}

fn tokenize_line(line: &str) -> Vec<(Range<usize>, WolframHighlight)> {
  let mut tokens = Vec::new();
  let bytes = line.as_bytes();
  let mut i = 0;

  while i < bytes.len() {
    // Skip non-ASCII
    if bytes[i] >= 128 {
      i += 1;
      while i < bytes.len() && bytes[i] & 0xC0 == 0x80 {
        i += 1;
      }
      continue;
    }

    let start = i;
    let c = bytes[i] as char;

    // Whitespace
    if c.is_ascii_whitespace() {
      i += 1;
      continue;
    }

    // Comment: (* ... *)
    if c == '(' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
      i += 2;
      loop {
        if i + 1 >= bytes.len() {
          i = bytes.len();
          break;
        }
        if bytes[i] == b'*' && bytes[i + 1] == b')' {
          i += 2;
          break;
        }
        i += 1;
      }
      tokens.push((start..i, WolframHighlight::Comment));
      continue;
    }

    // String: "..."
    if c == '"' {
      i += 1;
      while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
          i += 2;
          continue;
        }
        if bytes[i] == b'"' {
          i += 1;
          break;
        }
        i += 1;
      }
      tokens.push((start..i, WolframHighlight::String));
      continue;
    }

    // Number
    if c.is_ascii_digit() {
      while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
        i += 1;
      }
      tokens.push((start..i, WolframHighlight::Number));
      continue;
    }

    // Identifier
    if c.is_ascii_alphabetic() || c == '$' {
      while i < bytes.len()
        && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'$')
      {
        i += 1;
      }
      let word = &line[start..i];
      let next_is_bracket = i < bytes.len() && bytes[i] == b'[';

      let highlight = if is_keyword(word) {
        WolframHighlight::Keyword
      } else if next_is_bracket || bytes[start].is_ascii_uppercase() {
        WolframHighlight::Function
      } else {
        WolframHighlight::Normal
      };

      tokens.push((start..i, highlight));
      continue;
    }

    // Pattern: _, __, ___, _Head
    if c == '_' {
      while i < bytes.len() && bytes[i] == b'_' {
        i += 1;
      }
      while i < bytes.len()
        && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'$')
      {
        i += 1;
      }
      tokens.push((start..i, WolframHighlight::Pattern));
      continue;
    }

    // Slot: #, ##, #1
    if c == '#' {
      i += 1;
      while i < bytes.len() && (bytes[i] == b'#' || bytes[i].is_ascii_digit()) {
        i += 1;
      }
      tokens.push((start..i, WolframHighlight::Pattern));
      continue;
    }

    // Operators
    if "+-*/^@~!<>=&|;:,.?%".contains(c) {
      i += 1;
      while i < bytes.len() && "=>&|:->".contains(bytes[i] as char) {
        i += 1;
      }
      tokens.push((start..i, WolframHighlight::Operator));
      continue;
    }

    // Everything else (brackets, etc.)
    i += 1;
  }

  tokens
}
