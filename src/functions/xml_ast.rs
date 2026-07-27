//! XML import and export: the symbolic XML wolframscript reads a document
//! into, `XMLObject["Document"][prolog, root, epilog]` with elements written
//! as `XMLElement[name, {attributes}, {children}]`.

use crate::syntax::Expr;

/// A parse failure, carrying the byte offset it happened at so the caller can
/// report the line and character wolframscript names.
pub struct XmlError {
  pub offset: usize,
}

/// `(line, character)` of a byte offset, both 1-based.
pub fn line_and_character(text: &str, offset: usize) -> (usize, usize) {
  let upto = &text[..offset.min(text.len())];
  let line = upto.matches('\n').count() + 1;
  let character = match upto.rfind('\n') {
    Some(i) => upto[i + 1..].chars().count() + 1,
    None => upto.chars().count() + 1,
  };
  (line, character)
}

struct Parser<'a> {
  text: &'a [u8],
  source: &'a str,
  pos: usize,
}

/// Decode the predefined XML entities and numeric character references.
fn decode_entities(text: &str) -> String {
  let mut out = String::with_capacity(text.len());
  let mut rest = text;
  while let Some(i) = rest.find('&') {
    out.push_str(&rest[..i]);
    let after = &rest[i + 1..];
    let Some(end) = after.find(';') else {
      out.push('&');
      rest = after;
      continue;
    };
    let name = &after[..end];
    let decoded = match name {
      "amp" => Some('&'.to_string()),
      "lt" => Some('<'.to_string()),
      "gt" => Some('>'.to_string()),
      "quot" => Some('"'.to_string()),
      "apos" => Some('\''.to_string()),
      _ => name
        .strip_prefix('#')
        .and_then(|digits| match digits.strip_prefix(['x', 'X']) {
          Some(hex) => u32::from_str_radix(hex, 16).ok(),
          None => digits.parse::<u32>().ok(),
        })
        .and_then(char::from_u32)
        .map(|c| c.to_string()),
    };
    match decoded {
      Some(text) => {
        out.push_str(&text);
        rest = &after[end + 1..];
      }
      // An unknown entity is left as written.
      None => {
        out.push('&');
        rest = after;
      }
    }
  }
  out.push_str(rest);
  out
}

/// Escape the characters that cannot stand for themselves in XML text.
fn escape_text(text: &str) -> String {
  text
    .replace('&', "&amp;")
    .replace('<', "&lt;")
    .replace('>', "&gt;")
}

impl<'a> Parser<'a> {
  fn new(source: &'a str) -> Self {
    Parser {
      text: source.as_bytes(),
      source,
      pos: 0,
    }
  }

  fn error<T>(&self) -> Result<T, XmlError> {
    Err(XmlError { offset: self.pos })
  }

  fn peek(&self) -> Option<u8> {
    self.text.get(self.pos).copied()
  }

  fn starts_with(&self, s: &str) -> bool {
    self.source[self.pos.min(self.source.len())..].starts_with(s)
  }

  fn skip_whitespace(&mut self) {
    while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
      self.pos += 1;
    }
  }

  /// Consume `<!-- … -->`, `<![CDATA[ … ]]>` or `<!DOCTYPE …>`, returning the
  /// CDATA text (the others carry nothing).
  fn skip_to(&mut self, end: &str) -> Result<usize, XmlError> {
    match self.source[self.pos..].find(end) {
      Some(i) => {
        let start = self.pos;
        self.pos += i + end.len();
        Ok(start)
      }
      None => self.error(),
    }
  }

  /// An XML name: letters, digits, `_`, `-`, `.`, `:`.
  fn parse_name(&mut self) -> Result<String, XmlError> {
    let start = self.pos;
    while matches!(self.peek(), Some(c)
      if c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-' | b'.' | b':'))
    {
      self.pos += 1;
    }
    if self.pos == start {
      return self.error();
    }
    Ok(self.source[start..self.pos].to_string())
  }

  /// `name="value"` pairs up to the end of the start tag.
  fn parse_attributes(&mut self) -> Result<Vec<Expr>, XmlError> {
    let mut attributes = Vec::new();
    loop {
      self.skip_whitespace();
      match self.peek() {
        Some(b'>') | Some(b'/') | Some(b'?') | None => return Ok(attributes),
        _ => {}
      }
      let name = self.parse_name()?;
      self.skip_whitespace();
      if self.peek() != Some(b'=') {
        return self.error();
      }
      self.pos += 1;
      self.skip_whitespace();
      let quote = match self.peek() {
        Some(q @ (b'"' | b'\'')) => q,
        _ => return self.error(),
      };
      self.pos += 1;
      let start = self.pos;
      while self.peek().is_some_and(|c| c != quote) {
        self.pos += 1;
      }
      if self.peek().is_none() {
        return self.error();
      }
      let value = decode_entities(&self.source[start..self.pos]);
      self.pos += 1;
      attributes.push(Expr::Rule {
        pattern: Box::new(Expr::String(name)),
        replacement: Box::new(Expr::String(value)),
      });
    }
  }

  /// A whole element, starting at its `<`.
  fn parse_element(&mut self) -> Result<Expr, XmlError> {
    if self.peek() != Some(b'<') {
      return self.error();
    }
    self.pos += 1;
    let name = self.parse_name()?;
    let attributes = self.parse_attributes()?;
    self.skip_whitespace();
    if self.starts_with("/>") {
      self.pos += 2;
      return Ok(xml_element(&name, attributes, Vec::new()));
    }
    if self.peek() != Some(b'>') {
      return self.error();
    }
    self.pos += 1;
    let mut children: Vec<Expr> = Vec::new();
    loop {
      if self.pos >= self.text.len() {
        return self.error();
      }
      if self.starts_with("</") {
        self.pos += 2;
        let closing = self.parse_name()?;
        self.skip_whitespace();
        if closing != name || self.peek() != Some(b'>') {
          return self.error();
        }
        self.pos += 1;
        return Ok(xml_element(&name, attributes, children));
      }
      if self.starts_with("<!--") {
        // Comments carry no content of their own.
        self.pos += 4;
        self.skip_to("-->")?;
        continue;
      }
      if self.starts_with("<![CDATA[") {
        self.pos += 9;
        let start = self.pos;
        let end = self.skip_to("]]>")?;
        let _ = end;
        let text = &self.source[start..self.pos - 3];
        if !text.is_empty() {
          children.push(Expr::String(text.to_string()));
        }
        continue;
      }
      if self.peek() == Some(b'<') {
        children.push(self.parse_element()?);
        continue;
      }
      // Text up to the next tag. A run of pure whitespace between elements is
      // layout, not content, and is dropped.
      let start = self.pos;
      while self.peek().is_some_and(|c| c != b'<') {
        self.pos += 1;
      }
      let text = &self.source[start..self.pos];
      if !text.trim().is_empty() {
        children.push(Expr::String(decode_entities(text)));
      }
    }
  }

  /// The `<?xml … ?>` declaration, comments and doctype ahead of the root.
  fn parse_prolog(&mut self) -> Result<Vec<Expr>, XmlError> {
    let mut prolog = Vec::new();
    loop {
      self.skip_whitespace();
      if self.starts_with("<?xml") {
        self.pos += 5;
        let attributes = self.parse_attributes()?;
        self.skip_whitespace();
        if !self.starts_with("?>") {
          return self.error();
        }
        self.pos += 2;
        // The declaration keeps its pseudo-attributes, capitalized as
        // wolframscript writes them.
        let renamed: Vec<Expr> = attributes
          .iter()
          .map(|a| match a {
            Expr::Rule {
              pattern,
              replacement,
            } => Expr::Rule {
              pattern: Box::new(match pattern.as_ref() {
                Expr::String(s) => Expr::String(capitalize(s)),
                other => other.clone(),
              }),
              replacement: replacement.clone(),
            },
            other => other.clone(),
          })
          .collect();
        prolog.push(Expr::CurriedCall {
          func: Box::new(Expr::FunctionCall {
            name: "XMLObject".to_string(),
            args: vec![Expr::String("Declaration".to_string())].into(),
          }),
          args: renamed,
        });
        continue;
      }
      if self.starts_with("<!--") {
        self.pos += 4;
        self.skip_to("-->")?;
        continue;
      }
      if self.starts_with("<!") || self.starts_with("<?") {
        self.skip_to(">")?;
        continue;
      }
      return Ok(prolog);
    }
  }
}

fn capitalize(s: &str) -> String {
  let mut chars = s.chars();
  match chars.next() {
    Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    None => String::new(),
  }
}

fn xml_element(name: &str, attributes: Vec<Expr>, children: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: "XMLElement".to_string(),
    args: vec![
      Expr::String(name.to_string()),
      Expr::List(attributes.into()),
      Expr::List(children.into()),
    ]
    .into(),
  }
}

/// Parse a whole XML document into wolframscript's symbolic form.
pub fn parse_xml_document(source: &str) -> Result<Expr, XmlError> {
  let mut parser = Parser::new(source);
  let prolog = parser.parse_prolog()?;
  let root = parser.parse_element()?;
  // Trailing comments and whitespace are the epilog; wolframscript reports it
  // empty for the documents it can round-trip.
  let epilog: Vec<Expr> = Vec::new();
  parser.skip_whitespace();
  while parser.starts_with("<!--") {
    parser.pos += 4;
    parser.skip_to("-->")?;
    parser.skip_whitespace();
  }
  if parser.pos != parser.text.len() {
    return parser.error();
  }
  Ok(Expr::CurriedCall {
    func: Box::new(Expr::FunctionCall {
      name: "XMLObject".to_string(),
      args: vec![Expr::String("Document".to_string())].into(),
    }),
    args: vec![Expr::List(prolog.into()), root, Expr::List(epilog.into())],
  })
}

/// Serialize symbolic XML back to text. Accepts an `XMLElement`, a whole
/// `XMLObject["Document"][…]`, or a bare string (character data).
pub fn xml_to_string(expr: &Expr) -> Option<String> {
  match expr {
    Expr::String(s) => Some(escape_text(s)),
    Expr::FunctionCall { name, args }
      if name == "XMLElement" && args.len() == 3 =>
    {
      let Expr::String(tag) = &args[0] else {
        return None;
      };
      let mut out = format!("<{tag}");
      if let Expr::List(attributes) = &args[1] {
        for a in attributes.iter() {
          let (Expr::Rule {
            pattern,
            replacement,
          }
          | Expr::RuleDelayed {
            pattern,
            replacement,
          }) = a
          else {
            return None;
          };
          let (Expr::String(key), Expr::String(value)) =
            (pattern.as_ref(), replacement.as_ref())
          else {
            return None;
          };
          out.push_str(&format!(" {key}=\"{}\"", escape_text(value)));
        }
      }
      let children = match &args[2] {
        Expr::List(items) => items,
        _ => return None,
      };
      if children.is_empty() {
        out.push_str(&format!("></{tag}>"));
        return Some(out);
      }
      out.push('>');
      for child in children.iter() {
        out.push_str(&xml_to_string(child)?);
      }
      out.push_str(&format!("</{tag}>"));
      Some(out)
    }
    // XMLObject["Document"][prolog, root, epilog]
    Expr::CurriedCall { func, args } => {
      let Expr::FunctionCall { name, args: head } = func.as_ref() else {
        return None;
      };
      if name != "XMLObject" || head.len() != 1 {
        return None;
      }
      match &head[0] {
        Expr::String(kind) if kind == "Document" && args.len() == 3 => {
          xml_to_string(&args[1])
        }
        _ => None,
      }
    }
    _ => None,
  }
}
