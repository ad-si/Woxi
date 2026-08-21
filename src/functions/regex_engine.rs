//! The regular-expression engine behind `RegularExpression`.
//!
//! The Wolfram Language's `RegularExpression` is PCRE, which has look-around
//! (`(?=…)`, `(?!…)`, `(?<=…)`, `(?<!…)`) and backreferences. Rust's `regex`
//! crate deliberately has neither — it guarantees linear time, and those
//! constructs cost that guarantee. `fancy-regex` adds them by backtracking
//! over `regex` for the parts it can.
//!
//! So Woxi uses both: every pattern is offered to `regex` first, and only one
//! it refuses falls through to `fancy-regex`. A pattern that does not use the
//! extra constructs therefore behaves — and performs — exactly as it did
//! before this module existed, and one that does now works instead of
//! failing to compile.
//!
//! [`WoxiRegex`] exposes the slice of the `regex` API that the string
//! functions use, with the same shapes (`Match::start`/`end`/`as_str`,
//! `Captures::get`), so call sites read the same whichever engine backs them.

/// One matched span of a subject string.
#[derive(Debug, Clone, Copy)]
pub struct Match<'t> {
  text: &'t str,
  start: usize,
  end: usize,
}

impl<'t> Match<'t> {
  /// Byte offset of the first character of the match.
  pub fn start(&self) -> usize {
    self.start
  }

  /// Byte offset just past the last character of the match.
  pub fn end(&self) -> usize {
    self.end
  }

  /// The matched text.
  pub fn as_str(&self) -> &'t str {
    &self.text[self.start..self.end]
  }

  /// Length of the matched text in bytes.
  pub fn len(&self) -> usize {
    self.end - self.start
  }

  /// Whether the match is empty.
  pub fn is_empty(&self) -> bool {
    self.start == self.end
  }

  /// The match's span, for callers that want the range itself.
  pub fn range(&self) -> std::ops::Range<usize> {
    self.start..self.end
  }
}

/// The groups one match captured. Group 0 is the whole match; a group that
/// took part in no alternative is `None`, exactly as in the `regex` crate.
#[derive(Debug, Clone)]
pub struct Captures<'t> {
  text: &'t str,
  groups: Vec<Option<(usize, usize)>>,
  names: Names,
}

impl<'t> Captures<'t> {
  /// The `i`th group, or `None` when the group did not participate.
  pub fn get(&self, i: usize) -> Option<Match<'t>> {
    self
      .groups
      .get(i)
      .copied()
      .flatten()
      .map(|(start, end)| Match {
        text: self.text,
        start,
        end,
      })
  }

  /// The group the pattern named `name`, or `None` when there is no such
  /// group or it did not participate in the match.
  pub fn name(&self, name: &str) -> Option<Match<'t>> {
    let index = self.names.iter().position(|n| n.as_deref() == Some(name))?;
    self.get(index)
  }

  /// How many groups the pattern has, counting the whole match.
  pub fn len(&self) -> usize {
    self.groups.len()
  }

  /// Whether there are no groups at all (never true for a real match, which
  /// always has group 0 — present so `len` does not stand alone).
  pub fn is_empty(&self) -> bool {
    self.groups.is_empty()
  }
}

/// The capture-group names of one pattern, in group order, shared by every
/// `Captures` it produces rather than copied into each.
type Names = std::sync::Arc<Vec<Option<String>>>;

/// A compiled regular expression, backed by whichever engine can take it.
#[derive(Debug, Clone)]
pub struct WoxiRegex {
  engine: Engine,
  names: Names,
}

/// Which engine compiled a pattern.
#[derive(Debug, Clone)]
enum Engine {
  /// The linear-time engine, which handles all but the extra constructs.
  Plain(regex::Regex),
  /// The backtracking engine, for look-around and backreferences.
  Fancy(Box<fancy_regex::Regex>),
}

/// Why a pattern would not compile. Carries the message the engine that
/// rejected it produced, so the user sees the real diagnosis.
#[derive(Debug)]
pub struct Error(String);

impl std::fmt::Display for Error {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str(&self.0)
  }
}

impl std::error::Error for Error {}

impl WoxiRegex {
  /// Compile `pat`, preferring the linear-time engine.
  ///
  /// When that engine refuses the pattern the backtracking one gets a turn.
  /// If it refuses too, the reported error is the *first* engine's: a plain
  /// mistake like an unclosed group should read as such rather than as a
  /// complaint about a construct the user never wrote.
  pub fn new(pat: &str) -> Result<Self, Error> {
    let engine = match regex::Regex::new(pat) {
      Ok(re) => Engine::Plain(re),
      Err(plain_err) => match fancy_regex::Regex::new(pat) {
        Ok(re) => Engine::Fancy(Box::new(re)),
        Err(_) => return Err(Error(plain_err.to_string())),
      },
    };
    let names: Vec<Option<String>> = match &engine {
      Engine::Plain(re) => {
        re.capture_names().map(|n| n.map(str::to_string)).collect()
      }
      Engine::Fancy(re) => {
        re.capture_names().map(|n| n.map(str::to_string)).collect()
      }
    };
    Ok(Self {
      engine,
      names: std::sync::Arc::new(names),
    })
  }

  /// The pattern this was compiled from.
  pub fn as_str(&self) -> &str {
    match &self.engine {
      Engine::Plain(re) => re.as_str(),
      Engine::Fancy(re) => re.as_str(),
    }
  }

  /// Whether the pattern matches anywhere in `text`.
  pub fn is_match(&self, text: &str) -> bool {
    match &self.engine {
      Engine::Plain(re) => re.is_match(text),
      // A backtracking run can hit its step limit; treat that as "no match"
      // rather than propagating an error through every call site.
      Engine::Fancy(re) => re.is_match(text).unwrap_or(false),
    }
  }

  /// The leftmost match in `text`, if any.
  pub fn find<'t>(&self, text: &'t str) -> Option<Match<'t>> {
    self.find_at(text, 0)
  }

  /// The leftmost match starting at or after byte offset `start`.
  ///
  /// Look-around still sees the whole subject: searching from an offset
  /// asks where the match begins, it does not cut the string short.
  pub fn find_at<'t>(&self, text: &'t str, start: usize) -> Option<Match<'t>> {
    if start > text.len() {
      return None;
    }
    match &self.engine {
      Engine::Plain(re) => re.find_at(text, start).map(|m| Match {
        text,
        start: m.start(),
        end: m.end(),
      }),
      Engine::Fancy(re) => {
        re.find_from_pos(text, start).ok().flatten().map(|m| Match {
          text,
          start: m.start(),
          end: m.end(),
        })
      }
    }
  }

  /// Every non-overlapping match, left to right.
  pub fn find_iter<'t>(&self, text: &'t str) -> Vec<Match<'t>> {
    let mut found = Vec::new();
    let mut at = 0usize;
    while let Some(m) = self.find_at(text, at) {
      // An empty match would otherwise spin on the same offset forever.
      at = if m.end() > m.start() {
        m.end()
      } else {
        next_char_boundary(text, m.end())
      };
      found.push(m);
      if at > text.len() {
        break;
      }
    }
    found
  }

  /// The groups of the leftmost match in `text`.
  pub fn captures<'t>(&self, text: &'t str) -> Option<Captures<'t>> {
    self.captures_at(text, 0)
  }

  /// The groups of the leftmost match starting at or after `start`.
  pub fn captures_at<'t>(
    &self,
    text: &'t str,
    start: usize,
  ) -> Option<Captures<'t>> {
    if start > text.len() {
      return None;
    }
    match &self.engine {
      Engine::Plain(re) => re.captures_at(text, start).map(|caps| Captures {
        text,
        groups: (0..caps.len())
          .map(|i| caps.get(i).map(|m| (m.start(), m.end())))
          .collect(),
        names: self.names.clone(),
      }),
      Engine::Fancy(re) => {
        let caps = re.captures_from_pos(text, start).ok().flatten()?;
        Some(Captures {
          text,
          groups: (0..caps.len())
            .map(|i| caps.get(i).map(|m| (m.start(), m.end())))
            .collect(),
          names: self.names.clone(),
        })
      }
    }
  }

  /// The names of the capture groups, in order, `None` for unnamed ones.
  pub fn capture_names(&self) -> &[Option<String>] {
    &self.names
  }

  /// `text` split around each match, keeping every piece between them.
  pub fn split<'t>(&self, text: &'t str) -> Vec<&'t str> {
    self.splitn(text, usize::MAX)
  }

  /// `text` split around matches, into at most `limit` pieces.
  pub fn splitn<'t>(&self, text: &'t str, limit: usize) -> Vec<&'t str> {
    let mut pieces = Vec::new();
    if limit == 0 {
      return pieces;
    }
    let mut last = 0usize;
    let mut at = 0usize;
    while pieces.len() + 1 < limit {
      let Some(m) = self.find_at(text, at) else {
        break;
      };
      // A zero-width match is a split point like any other — that is how
      // `StringSplit[…, StartOfLine]` cuts a string at boundaries rather
      // than at characters — but the search must still move on past it, or
      // it would keep finding the same one.
      at = if m.is_empty() {
        next_char_boundary(text, m.end())
      } else {
        m.end()
      };
      pieces.push(&text[last..m.start()]);
      last = m.end();
      if at > text.len() {
        break;
      }
    }
    pieces.push(&text[last..]);
    pieces
  }
}

/// The next character boundary at or after `i`, so advancing past a
/// zero-width match never lands inside a multi-byte character.
fn next_char_boundary(text: &str, i: usize) -> usize {
  let mut next = i + 1;
  while next <= text.len() && !text.is_char_boundary(next) {
    next += 1;
  }
  next
}
