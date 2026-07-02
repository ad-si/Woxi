//! Graphical (SVG) rendering for `QuestionObject[…]`.
//!
//! The Wolfram front end displays a `QuestionObject` as an interactive
//! question panel: the prompt text, an answer area and a Submit button.
//! When the embedded `AssessmentFunction` carries an explicit list of answer
//! choices the answer area is a set of radio buttons (multiple choice);
//! otherwise it is a free-form input field. Woxi draws the same panel as a
//! static SVG so the object has a graphical form in `ExportString[…, "SVG"]`,
//! raster/PDF export and the Playground.

use crate::functions::graphics::{MONO_ADVANCE, svg_escape, theme};
use crate::syntax::{Expr, ExprForm, format_expr};

const FONT_SIZE: f64 = 14.0;
const LINE_H: f64 = 20.0;
const PAD: f64 = 14.0;
/// Maximum characters per question line before wrapping.
const WRAP_COLS: usize = 56;
/// Height of one multiple-choice row.
const CHOICE_ROW_H: f64 = 24.0;
/// Horizontal offset of the choice label from the radio button.
const CHOICE_LABEL_X: f64 = 24.0;
const INPUT_W: f64 = 240.0;
const INPUT_H: f64 = 26.0;
const BUTTON_H: f64 = 28.0;
/// The Wolfram interface blue used for the Submit button.
const BUTTON_FILL: &str = "#024D88";

/// Monospace advance width of one character at `FONT_SIZE`.
fn char_w() -> f64 {
  FONT_SIZE * MONO_ADVANCE
}

/// Display text for a prompt or answer choice.
fn display_text(expr: &Expr) -> String {
  match expr {
    Expr::String(s) => s.clone(),
    other => format_expr(other, ExprForm::Output),
  }
}

/// Greedy word wrap at `WRAP_COLS` columns; overlong words are hard-split.
fn wrap_text(text: &str) -> Vec<String> {
  let mut lines: Vec<String> = Vec::new();
  let mut current = String::new();
  for word in text.split_whitespace() {
    let mut word = word;
    // Hard-split words longer than a full line.
    while word.chars().count() > WRAP_COLS {
      if !current.is_empty() {
        lines.push(std::mem::take(&mut current));
      }
      let split: String = word.chars().take(WRAP_COLS).collect();
      lines.push(split.clone());
      word = &word[split.len()..];
    }
    let needed = if current.is_empty() {
      word.chars().count()
    } else {
      current.chars().count() + 1 + word.chars().count()
    };
    if needed > WRAP_COLS && !current.is_empty() {
      lines.push(std::mem::take(&mut current));
    }
    if !current.is_empty() {
      current.push(' ');
    }
    current.push_str(word);
  }
  if !current.is_empty() {
    lines.push(current);
  }
  if lines.is_empty() {
    lines.push(String::new());
  }
  lines
}

/// The answer area of the panel: explicit choices (radio buttons) or a
/// free-form input field.
enum AnswerArea {
  Choices(Vec<String>),
  InputField,
}

/// Extract the answer area from the assessment argument. An
/// `AssessmentFunction[spec]` wrapper is unwrapped first; a list spec yields
/// one choice per entry (the left-hand side of `answer -> grade` rules, or
/// the bare answer value itself), any other spec yields an input field.
fn answer_area(assess: &Expr) -> AnswerArea {
  let spec = match assess {
    Expr::FunctionCall { name, args }
      if name == "AssessmentFunction" && args.len() == 1 =>
    {
      &args[0]
    }
    other => other,
  };
  match spec {
    Expr::List(items) if !items.is_empty() => AnswerArea::Choices(
      items
        .iter()
        .map(|e| match e {
          Expr::Rule { pattern, .. } => display_text(pattern),
          other => display_text(other),
        })
        .collect(),
    ),
    _ => AnswerArea::InputField,
  }
}

/// Render `QuestionObject[q, assess]` (or `QuestionObject[assess]`) as a
/// question-panel SVG. Returns `None` when `expr` is not a well-formed
/// `QuestionObject`.
pub fn question_object_to_svg(expr: &Expr) -> Option<String> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "QuestionObject" || args.is_empty() || args.len() > 2 {
    return None;
  }
  let (question, assess) = match args.len() {
    2 => (Some(&args[0]), &args[1]),
    _ => (None, &args[0]),
  };

  let ch = char_w();
  let t = theme();
  let question_lines: Vec<String> = question
    .map(|q| wrap_text(&display_text(q)))
    .unwrap_or_default();
  let area = answer_area(assess);

  // ── Measure ────────────────────────────────────────────────────────────
  let question_w = question_lines
    .iter()
    .map(|l| l.chars().count())
    .max()
    .unwrap_or(0) as f64
    * ch;
  let area_w = match &area {
    AnswerArea::Choices(labels) => labels
      .iter()
      .map(|l| CHOICE_LABEL_X + l.chars().count() as f64 * ch)
      .fold(0.0, f64::max),
    AnswerArea::InputField => INPUT_W,
  };
  let button_w = 6.0 * ch + 36.0; // "Submit" plus horizontal padding
  let content_w = question_w.max(area_w).max(button_w).max(240.0);
  let width = content_w + 2.0 * PAD;

  let question_h = question_lines.len() as f64 * LINE_H;
  let area_h = match &area {
    AnswerArea::Choices(labels) => labels.len() as f64 * CHOICE_ROW_H,
    AnswerArea::InputField => INPUT_H,
  };
  let question_gap = if question_lines.is_empty() { 0.0 } else { 10.0 };
  let height = PAD + question_h + question_gap + area_h + 14.0 + BUTTON_H + PAD;

  // ── Draw ───────────────────────────────────────────────────────────────
  let mut svg = format!(
    "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width:.0}\" \
     height=\"{height:.0}\" viewBox=\"0 0 {width:.0} {height:.0}\" \
     font-family=\"monospace\" font-size=\"{FONT_SIZE:.0}\">\n"
  );
  // Panel frame
  svg.push_str(&format!(
    "<rect x=\"0.5\" y=\"0.5\" width=\"{:.1}\" height=\"{:.1}\" rx=\"3\" \
     fill=\"none\" stroke=\"{}\"/>\n",
    width - 1.0,
    height - 1.0,
    t.framed_border
  ));

  let mut y = PAD;
  // Question text
  for line in &question_lines {
    svg.push_str(&format!(
      "<text x=\"{PAD:.1}\" y=\"{:.1}\" fill=\"{}\" \
       xml:space=\"preserve\">{}</text>\n",
      y + FONT_SIZE * 0.8,
      t.text_primary,
      svg_escape(line)
    ));
    y += LINE_H;
  }
  y += question_gap;

  // Answer area
  match &area {
    AnswerArea::Choices(labels) => {
      for label in labels {
        let cy = y + CHOICE_ROW_H / 2.0;
        svg.push_str(&format!(
          "<circle cx=\"{:.1}\" cy=\"{cy:.1}\" r=\"6\" fill=\"none\" \
           stroke=\"{}\" stroke-width=\"1.2\"/>\n",
          PAD + 7.0,
          t.text_secondary
        ));
        svg.push_str(&format!(
          "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"{}\" \
           xml:space=\"preserve\">{}</text>\n",
          PAD + CHOICE_LABEL_X,
          cy + FONT_SIZE * 0.3,
          t.text_primary,
          svg_escape(label)
        ));
        y += CHOICE_ROW_H;
      }
    }
    AnswerArea::InputField => {
      svg.push_str(&format!(
        "<rect x=\"{PAD:.1}\" y=\"{y:.1}\" width=\"{INPUT_W:.1}\" \
         height=\"{INPUT_H:.1}\" rx=\"2\" fill=\"none\" stroke=\"{}\"/>\n",
        t.table_border_strong
      ));
      y += INPUT_H;
    }
  }
  y += 14.0;

  // Submit button
  svg.push_str(&format!(
    "<rect x=\"{PAD:.1}\" y=\"{y:.1}\" width=\"{button_w:.1}\" \
     height=\"{BUTTON_H:.1}\" rx=\"4\" fill=\"{BUTTON_FILL}\"/>\n"
  ));
  svg.push_str(&format!(
    "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"white\" \
     text-anchor=\"middle\">Submit</text>\n",
    PAD + button_w / 2.0,
    y + BUTTON_H / 2.0 + FONT_SIZE * 0.3
  ));

  svg.push_str("</svg>");
  Some(svg)
}
