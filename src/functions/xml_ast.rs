//! XML import and export: the symbolic XML wolframscript reads a document
//! into, `XMLObject["Document"][prolog, root, epilog]` with elements written
//! as `XMLElement[name, {attributes}, {children}]`.

#[allow(unused_imports)]
use super::*;

/// A parse failure, carrying the byte offset it happened at so the caller can
/// report the line and character wolframscript names.
pub struct XmlError {
  pub offset: usize,
  /// Set when the failure is an XML namespace prefix with no declaration.
  pub unresolved_prefix: Option<String>,
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

/// The namespace URI of the `xmlns` attributes themselves.
const XMLNS_URI: &str = "http://www.w3.org/2000/xmlns/";

struct Parser<'a> {
  text: &'a [u8],
  source: &'a str,
  pos: usize,
  /// Prefix → URI, one frame per open element.
  namespaces: Vec<Vec<(String, String)>>,
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
      namespaces: Vec::new(),
    }
  }

  fn error<T>(&self) -> Result<T, XmlError> {
    Err(XmlError {
      offset: self.pos,
      unresolved_prefix: None,
    })
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
  fn parse_attributes(&mut self) -> Result<Vec<(String, String)>, XmlError> {
    let mut attributes: Vec<(String, String)> = Vec::new();
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
      attributes.push((name, value));
    }
  }

  /// A whole element, starting at its `<`.
  fn parse_element(&mut self) -> Result<Expr, XmlError> {
    if self.peek() != Some(b'<') {
      return self.error();
    }
    self.pos += 1;
    let name = self.parse_name()?;
    let raw_attributes = self.parse_attributes()?;
    self.skip_whitespace();
    // The xmlns declarations on this tag are in scope for its own name and
    // attributes as well as for everything inside it.
    self.namespaces.push(
      raw_attributes
        .iter()
        .filter_map(|(key, value)| match key.split_once(':') {
          Some(("xmlns", prefix)) => Some((prefix.to_string(), value.clone())),
          _ => None,
        })
        .collect(),
    );
    // The tag is consumed before its names are resolved, so an unresolvable
    // prefix is reported at the end of the tag — the position wolframscript
    // names.
    let empty_element = if self.starts_with("/>") {
      self.pos += 2;
      true
    } else if self.peek() == Some(b'>') {
      self.pos += 1;
      false
    } else {
      self.namespaces.pop();
      return self.error();
    };
    let qualified = self
      .qualified_name(&name, false)
      .and_then(|n| Ok((n, self.qualified_attributes(&raw_attributes)?)));
    let (element_name, attributes) = match qualified {
      Ok(pair) => pair,
      Err(e) => {
        self.namespaces.pop();
        return Err(e);
      }
    };
    if empty_element {
      self.namespaces.pop();
      return Ok(xml_element(element_name, attributes, Vec::new()));
    }
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
          self.namespaces.pop();
          return self.error();
        }
        self.pos += 1;
        self.namespaces.pop();
        return Ok(xml_element(element_name, attributes, children));
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

  /// The URI a prefix is bound to by the innermost declaration in scope.
  fn namespace_uri(&self, prefix: &str) -> Option<&str> {
    self
      .namespaces
      .iter()
      .rev()
      .flat_map(|frame| frame.iter().rev())
      .find(|(p, _)| p == prefix)
      .map(|(_, uri)| uri.as_str())
  }

  /// An element or attribute name as wolframscript writes it: a prefixed name
  /// becomes `{namespace-uri, local-name}`, an unprefixed one stays a plain
  /// string — even under a default `xmlns`, which only wolframscript's
  /// attribute list records.
  fn qualified_name(
    &self,
    name: &str,
    is_attribute: bool,
  ) -> Result<Expr, XmlError> {
    let Some((prefix, local)) = name.split_once(':') else {
      return Ok(Expr::String(name.to_string()));
    };
    if is_attribute && prefix == "xmlns" {
      return Ok(namespaced_name(XMLNS_URI, local));
    }
    match self.namespace_uri(prefix) {
      Some(uri) => Ok(namespaced_name(uri, local)),
      None => Err(XmlError {
        offset: self.pos,
        unresolved_prefix: Some(prefix.to_string()),
      }),
    }
  }

  /// The attribute rules of a start tag, `xmlns` declarations included: they
  /// are written under the namespace of the `xmlns` attributes themselves.
  fn qualified_attributes(
    &self,
    raw: &[(String, String)],
  ) -> Result<Vec<Expr>, XmlError> {
    let mut attributes = Vec::with_capacity(raw.len());
    for (name, value) in raw {
      let key = if name == "xmlns" {
        namespaced_name(XMLNS_URI, "xmlns")
      } else {
        self.qualified_name(name, true)?
      };
      attributes.push(Expr::Rule {
        pattern: Box::new(key),
        replacement: Box::new(Expr::String(value.clone())),
      });
    }
    Ok(attributes)
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
          .map(|(name, value)| Expr::Rule {
            pattern: Box::new(Expr::String(capitalize(name))),
            replacement: Box::new(Expr::String(value.clone())),
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

/// `{namespace-uri, local-name}`, the name form wolframscript gives a
/// prefixed element or attribute.
fn namespaced_name(uri: &str, local: &str) -> Expr {
  Expr::List(
    vec![
      Expr::String(uri.to_string()),
      Expr::String(local.to_string()),
    ]
    .into(),
  )
}

fn xml_element(name: Expr, attributes: Vec<Expr>, children: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: "XMLElement".to_string(),
    args: vec![
      name,
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

/// Escape the characters that cannot stand for themselves inside a
/// single-quoted attribute value. wolframscript escapes both quote characters
/// regardless of the delimiter it picked.
fn escape_attribute(text: &str) -> String {
  escape_text(text)
    .replace('"', "&quot;")
    .replace('\'', "&apos;")
}

/// An element or attribute name: either a plain string or wolframscript's
/// `{namespace-uri, local-name}` pair. A pair is written with the prefix the
/// enclosing `xmlns:` declarations bind to that URI, and with the URI itself
/// as the prefix when nothing declares it.
fn xml_name(expr: &Expr, prefixes: &[(String, String)]) -> Option<String> {
  match expr {
    Expr::String(s) => Some(s.clone()),
    Expr::List(parts) if parts.len() == 2 => {
      let (Expr::String(uri), Expr::String(local)) = (&parts[0], &parts[1])
      else {
        return None;
      };
      // An xmlns declaration names itself: {xmlns-uri, "xmlns"} is the
      // default-namespace attribute, any other local name a prefix binding.
      if uri == XMLNS_URI {
        return Some(if local == "xmlns" {
          local.clone()
        } else {
          format!("xmlns:{local}")
        });
      }
      let prefix = prefixes
        .iter()
        .rev()
        .find(|(_, u)| u == uri)
        .map(|(p, _)| p.as_str())
        .unwrap_or(uri);
      Some(format!("{prefix}:{local}"))
    }
    _ => None,
  }
}

/// The prefix bindings an element's own attribute list declares.
fn declared_prefixes(attributes: &Expr) -> Vec<(String, String)> {
  let Expr::List(items) = attributes else {
    return Vec::new();
  };
  let mut out = Vec::new();
  for a in items.iter() {
    let (Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    }) = a
    else {
      continue;
    };
    let (Expr::List(name), Expr::String(uri)) =
      (pattern.as_ref(), replacement.as_ref())
    else {
      continue;
    };
    if name.len() == 2
      && let (Expr::String(ns), Expr::String(prefix)) = (&name[0], &name[1])
      && ns == XMLNS_URI
      && prefix != "xmlns"
    {
      out.push((prefix.clone(), uri.clone()));
    }
  }
  out
}

/// The head arguments of an `XMLObject["kind"][…]` expression.
fn xml_object<'a>(expr: &'a Expr, kind: &str) -> Option<&'a [Expr]> {
  let Expr::CurriedCall { func, args } = expr else {
    return None;
  };
  let Expr::FunctionCall { name, args: head } = func.as_ref() else {
    return None;
  };
  if name != "XMLObject" || head.len() != 1 {
    return None;
  }
  match &head[0] {
    Expr::String(k) if k == kind => Some(args),
    _ => None,
  }
}

/// Serialize the `XMLObject["Declaration"][…]` entries of a document prolog.
/// wolframscript prints `<?xml version='1.0' encoding='UTF-8'?>` followed by a
/// line break, with the option names lower-cased.
fn prolog_to_string(prolog: &Expr) -> Option<String> {
  let Expr::List(items) = prolog else {
    return None;
  };
  let mut out = String::new();
  for item in items.iter() {
    if let Some(options) = xml_object(item, "Declaration") {
      out.push_str("<?xml");
      for option in options {
        let (Expr::Rule {
          pattern,
          replacement,
        }
        | Expr::RuleDelayed {
          pattern,
          replacement,
        }) = option
        else {
          return None;
        };
        let (Expr::String(key), Expr::String(value)) =
          (pattern.as_ref(), replacement.as_ref())
        else {
          return None;
        };
        out.push_str(&format!(
          " {}='{}'",
          key.to_lowercase(),
          escape_attribute(value)
        ));
      }
      out.push_str("?>\n");
    } else {
      out.push_str(&xml_to_string_at(item, 0, &[])?);
    }
  }
  Some(out)
}

/// Serialize symbolic XML back to text. Accepts an `XMLElement`, a whole
/// `XMLObject["Document"][…]`, or a bare string (character data).
pub fn xml_to_string(expr: &Expr) -> Option<String> {
  xml_to_string_at(expr, 0, &[])
}

/// Serialize `expr` as the content of an element nested `depth` levels deep,
/// with `prefixes` the `xmlns:` bindings in scope. An element whose children
/// are all elements is laid out one child per line, indented by one space per
/// level; anything else (in particular mixed content) stays on a single line —
/// the layout wolframscript writes.
fn xml_to_string_at(
  expr: &Expr,
  depth: usize,
  prefixes: &[(String, String)],
) -> Option<String> {
  match expr {
    Expr::String(s) => Some(escape_text(s)),
    Expr::FunctionCall { name, args }
      if name == "XMLElement" && args.len() == 3 =>
    {
      let mut prefixes = prefixes.to_vec();
      prefixes.extend(declared_prefixes(&args[1]));
      let prefixes = prefixes.as_slice();
      let tag = xml_name(&args[0], prefixes)?;
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
          let key = xml_name(pattern, prefixes)?;
          let Expr::String(value) = replacement.as_ref() else {
            return None;
          };
          out.push_str(&format!(" {key}='{}'", escape_attribute(value)));
        }
      }
      let children = match &args[2] {
        Expr::List(items) => items,
        _ => return None,
      };
      if children.is_empty() {
        out.push_str(" />");
        return Some(out);
      }
      let all_elements = children.iter().all(|c| {
        matches!(c, Expr::FunctionCall { name, args }
          if name == "XMLElement" && args.len() == 3)
      });
      out.push('>');
      if all_elements {
        for child in children.iter() {
          out.push('\n');
          out.push_str(&" ".repeat(depth + 1));
          out.push_str(&xml_to_string_at(child, depth + 1, prefixes)?);
        }
        out.push('\n');
        out.push_str(&" ".repeat(depth));
      } else {
        for child in children.iter() {
          out.push_str(&xml_to_string_at(child, depth, prefixes)?);
        }
      }
      out.push_str(&format!("</{tag}>"));
      Some(out)
    }
    Expr::CurriedCall { .. } => {
      // XMLObject["Document"][prolog, root, epilog]
      if let Some(parts) = xml_object(expr, "Document")
        && parts.len() == 3
      {
        let mut out = prolog_to_string(&parts[0])?;
        out.push_str(&xml_to_string_at(&parts[1], depth, prefixes)?);
        return Some(out);
      }
      // XMLObject["Comment"]["text"] — wolframscript breaks the line after a
      // comment even in otherwise inline content.
      if let Some(parts) = xml_object(expr, "Comment")
        && parts.len() == 1
        && let Expr::String(text) = &parts[0]
      {
        return Some(format!("<!--{text}-->\n"));
      }
      None
    }
    _ => None,
  }
}

/// The import elements wolframscript offers for an XML document.
pub const XML_ELEMENTS: [&str; 9] = [
  "CDATA",
  "Comments",
  "EmbeddedDTD",
  "Plaintext",
  "Summary",
  "Tags",
  "Tree",
  "XMLElement",
  "XMLObject",
];

/// The root element of a parsed document.
fn document_root(document: &Expr) -> Option<&Expr> {
  match document {
    Expr::CurriedCall { args, .. } if args.len() == 3 => Some(&args[1]),
    _ => None,
  }
}

/// Every tag name in the document, in document order.
fn collect_tags(expr: &Expr, out: &mut Vec<String>) {
  if let Expr::FunctionCall { name, args } = expr
    && name == "XMLElement"
    && args.len() == 3
  {
    match &args[0] {
      Expr::String(tag) => out.push(tag.clone()),
      // A namespaced name is written `{uri, local}`; the tag is its local part.
      Expr::List(parts) if parts.len() == 2 => {
        if let Expr::String(local) = &parts[1] {
          out.push(local.clone());
        }
      }
      _ => {}
    }
    if let Expr::List(children) = &args[2] {
      for child in children.iter() {
        collect_tags(child, out);
      }
    }
  }
}

/// Every run of character data in the document, in document order.
fn collect_text(expr: &Expr, out: &mut Vec<String>) {
  match expr {
    Expr::String(text) => out.push(text.clone()),
    Expr::FunctionCall { name, args }
      if name == "XMLElement" && args.len() == 3 =>
    {
      if let Expr::List(children) = &args[2] {
        for child in children.iter() {
          collect_text(child, out);
        }
      }
    }
    _ => {}
  }
}

/// One import element of a parsed XML document. `None` means the element is
/// not one wolframscript knows.
pub fn xml_import_element(document: &Expr, element: &str) -> Option<Expr> {
  let root = document_root(document);
  Some(match element {
    "XMLObject" => document.clone(),
    "XMLElement" => Expr::List(vec![root?.clone()].into()),
    "Tags" => {
      let mut tags = Vec::new();
      collect_tags(root?, &mut tags);
      tags.sort();
      tags.dedup();
      Expr::List(tags.into_iter().map(Expr::String).collect())
    }
    "CDATA" => {
      let mut text = Vec::new();
      collect_text(root?, &mut text);
      Expr::List(text.into_iter().map(Expr::String).collect())
    }
    "Plaintext" => {
      let mut text = Vec::new();
      collect_text(root?, &mut text);
      Expr::String(text.join("\n"))
    }
    // A comment carries no content, so wolframscript reports none.
    "Comments" | "EmbeddedDTD" => Expr::List(Vec::new().into()),
    "Elements" => Expr::List(
      XML_ELEMENTS
        .iter()
        .map(|n| Expr::String((*n).to_string()))
        .collect(),
    ),
    _ => return None,
  })
}
