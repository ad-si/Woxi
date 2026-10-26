//! `ExampleData[{"Text", …}]`: generated stand-ins for public texts.
//!
//! Each text is generated from the dictionary (English) or from syllables
//! over the letters of its language, with the original's word count, line
//! and paragraph structure and average sentence length. `"Title"`,
//! `"Author"`, `"Language"` are the recorded metadata; the words are not
//! the original's.

use super::images::cached;
use super::rng::Rng;
use super::words::{Lexicon, capitalize, prose};

use super::{Collection, Entry};
use crate::functions::*;

pub struct Texts;

/// `@Counts`: {characters, words, lines, empty lines, sentences,
/// formatted characters, paragraph breaks}.
fn counts(entry: &Entry) -> [usize; 7] {
  let mut out = [0usize; 7];
  if let Some(Expr::List(ref c)) = entry.stat("@Counts") {
    for (i, v) in c.iter().enumerate().take(7) {
      if let Expr::Integer(n) = v {
        out[i] = *n as usize;
      }
    }
  }
  out
}

fn lexicon(entry: &Entry) -> Lexicon {
  let language = match entry.exact("Language") {
    Some(Expr::String(ref l)) => l.clone(),
    _ => "English".to_string(),
  };
  let letters: Vec<char> = match entry.stat("@Alphabet") {
    Some(Expr::String(ref s)) => s.chars().collect(),
    // A large script: {count, {lowest code point, highest}}.
    Some(Expr::List(ref l)) => match l.get(1) {
      Some(Expr::List(r)) => match (r.first(), r.get(1)) {
        (Some(Expr::Integer(lo)), Some(Expr::Integer(hi))) => {
          // Characters of the script's main block: Hangul syllables or
          // CJK ideographs.
          let (lo, hi) =
            if language == "Korean" || (*hi >= 0xac00 && *hi < 0xd7b0) {
              (0xac00, (*hi as u32).min(0xd7a3))
            } else {
              ((*lo as u32).max(0x4e00), (*hi as u32).clamp(0x4e01, 0x9fff))
            };
          let mut rng = entry.rng("alphabet");
          (0..800)
            .filter_map(|_| {
              char::from_u32(lo + rng.below((hi - lo) as usize) as u32)
            })
            .collect()
        }
        _ => Vec::new(),
      },
      _ => Vec::new(),
    },
    _ => Vec::new(),
  };
  // Space-separated unless words are very long on average (Chinese and
  // Japanese count whole phrases as words).
  let [chars, words, ..] = counts(entry);
  Lexicon::for_language(&language, &letters, chars < 12 * words.max(1))
}

/// The generated lines of the text ("" between paragraphs).
fn lines(entry: &Entry) -> Vec<String> {
  let [_, words, line_count, empty, sentences, ..] = counts(entry);
  let lex = lexicon(entry);
  let mut rng = entry.rng("text");
  let text_lines = line_count.saturating_sub(empty).max(1);
  let mean_sentence = words as f64 / sentences.max(1) as f64;
  let [chars, ..] = counts(entry);
  let body: String;
  let unspaced: Vec<String>;
  let tokens: Vec<&str> = if lex.spaced() {
    body = prose(&lex, &mut rng, words.max(1), mean_sentence.clamp(4.0, 60.0));
    body.split(' ').collect()
  } else {
    // Unspaced scripts: one sentence of the typical length per line.
    let per_line =
      (chars as f64 / text_lines as f64 / 1.5).round().max(1.0) as usize;
    // Words beyond one per line are phrases separated by spaces.
    let extra = words.saturating_sub(text_lines);
    unspaced = (0..text_lines)
      .map(|i| {
        let phrases =
          1 + extra / text_lines + usize::from(i < extra % text_lines);
        (0..phrases)
          .map(|_| {
            super::words::sentence(&lex, &mut rng, (per_line / phrases).max(1))
          })
          .collect::<Vec<_>>()
          .join(" ")
      })
      .collect();
    unspaced.iter().map(String::as_str).collect()
  };
  // Words per line: evenly spread with jitter, summing to the total.
  let mut sizes = vec![tokens.len() / text_lines; text_lines];
  for s in sizes.iter_mut().take(tokens.len() % text_lines) {
    *s += 1;
  }
  jitter(&mut sizes, &mut rng);
  // Where the empty lines go: the recorded number of paragraphs (cells of
  // the notebook form) means that many - 1 breaks, sharing the empty lines
  // (a break can be several empty lines long).
  let paragraphs = match entry.stat("@Cells") {
    Some(Expr::List(ref styles)) => styles
      .iter()
      .map(|t| match t {
        Expr::List(p) => match p.get(1) {
          Some(Expr::Integer(n)) => *n as usize,
          _ => 0,
        },
        _ => 0,
      })
      .sum::<usize>(),
    _ => empty + 1,
  };
  let gaps = paragraphs
    .saturating_sub(1)
    .min(empty)
    .min(text_lines.saturating_sub(1));
  let mut breaks = vec![0usize; text_lines];
  let mut placed = 0;
  while placed < gaps {
    let i = 1 + rng.below(text_lines - 1);
    if breaks[i] == 0 {
      breaks[i] = 1;
      placed += 1;
    }
  }
  let break_rows: Vec<usize> =
    (0..text_lines).filter(|&i| breaks[i] > 0).collect();
  for _ in gaps..empty {
    if let Some(&i) = break_rows.get(rng.below(break_rows.len().max(1))) {
      breaks[i] += 1;
    }
  }
  let mut out = Vec::with_capacity(line_count);
  let mut k = 0;
  for (i, &n) in sizes.iter().enumerate() {
    for _ in 0..breaks[i] {
      out.push(String::new());
    }
    let sep = if lex.spaced() { " " } else { "" };
    let mut line = tokens[k..(k + n).min(tokens.len())].join(sep);
    // A new paragraph starts a sentence.
    if breaks[i] > 0 || i == 0 {
      line = capitalize(&line);
    }
    out.push(line);
    k += n;
  }
  while out.len() < line_count {
    out.push(String::new());
  }
  out
}

/// Move words between neighbouring lines so line lengths vary like real
/// text without changing the total.
fn jitter(sizes: &mut [usize], rng: &mut Rng) {
  for i in 0..sizes.len().saturating_sub(1) {
    let movable = sizes[i] / 4;
    if movable > 0 {
      let d = rng.below(movable + 1);
      if rng.chance(0.5) {
        sizes[i] -= d;
        sizes[i + 1] += d;
      } else if sizes[i + 1] > d {
        sizes[i + 1] -= d;
        sizes[i] += d;
      }
    }
  }
}

fn joined(lines: &[String]) -> String {
  lines
    .iter()
    .filter(|l| !l.is_empty())
    .cloned()
    .collect::<Vec<_>>()
    .join(" ")
}

/// Paragraphs: runs of non-empty lines.
fn paragraphs(lines: &[String]) -> Vec<String> {
  lines
    .split(std::string::String::is_empty)
    .filter(|p| !p.is_empty())
    .map(|p| p.join(" "))
    .collect()
}

impl Collection for Texts {
  fn value(&self, entry: &Entry, property: Option<&str>) -> Option<Expr> {
    let key = |p: &str| format!("Text/{}/{p}", entry.name);
    let get_lines = || lines(entry);
    Some(match property? {
      "Lines" => cached(key("Lines"), || super::string_list(&get_lines())),
      "String" => cached(key("String"), || str_expr(&joined(&get_lines()))),
      "FormattedText" => {
        cached(key("FormattedText"), || str_expr(&get_lines().join("\n")))
      }
      "Words" => cached(key("Words"), || {
        let s = joined(&get_lines());
        super::string_list(&s.split_whitespace().collect::<Vec<_>>())
      }),
      "NotebookExpression" => cached(key("NotebookExpression"), || {
        let cells: Vec<Expr> = paragraphs(&get_lines())
          .iter()
          .map(|p| call("Cell", vec![str_expr(p), str_expr("Text")]))
          .collect();
        call("Notebook", vec![Expr::List(cells.into())])
      }),
      _ => return None,
    })
  }
}
