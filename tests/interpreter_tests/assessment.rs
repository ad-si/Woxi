//! Tests for the quiz/assessment primitives: `QuestionObject`,
//! `AssessmentFunction`, and `AssessmentResultObject`.

use super::*;

// ─── Symbolic objects stay symbolic ─────────────────────────────────────────

#[test]
fn assessment_function_stays_symbolic() {
  assert_eq!(
    interpret("AssessmentFunction[{True}]").unwrap(),
    "AssessmentFunction[{True}]"
  );
}

#[test]
fn question_object_stays_symbolic() {
  assert_eq!(
    interpret(
      r#"QuestionObject["Is a whale a mammal?", AssessmentFunction[{True}]]"#
    )
    .unwrap(),
    "QuestionObject[Is a whale a mammal?, AssessmentFunction[{True}]]"
  );
}

#[test]
fn question_object_head() {
  assert_eq!(
    interpret(r#"Head[QuestionObject["q", AssessmentFunction[{True}]]]"#)
      .unwrap(),
    "QuestionObject"
  );
}

#[test]
fn assessment_function_head() {
  assert_eq!(
    interpret("Head[AssessmentFunction[{True}]]").unwrap(),
    "AssessmentFunction"
  );
}

#[test]
fn assessment_objects_emit_no_unimplemented_warning() {
  clear_state();
  let _ = interpret("AssessmentFunction[{True}]").unwrap();
  let warnings = woxi::get_captured_warnings();
  assert!(
    !warnings.iter().any(|w| w.contains("not yet implemented")),
    "unexpected unimplemented warning: {:?}",
    warnings
  );
}

// ─── Grading with a rule-based answer key ────────────────────────────────────

#[test]
fn assessment_grades_correct_answer() {
  assert_eq!(
    interpret(
      "AssessmentFunction[{714 -> False, 755 -> True, 868 -> False}][755]"
    )
    .unwrap(),
    "AssessmentResultObject[<|AnswerCorrect -> True, Score -> 1|>]"
  );
}

#[test]
fn assessment_grades_incorrect_answer() {
  assert_eq!(
    interpret(
      "AssessmentFunction[{714 -> False, 755 -> True, 868 -> False}][714]"
    )
    .unwrap(),
    "AssessmentResultObject[<|AnswerCorrect -> False, Score -> 0|>]"
  );
}

#[test]
fn assessment_grades_unlisted_answer_as_incorrect() {
  assert_eq!(
    interpret(
      "AssessmentFunction[{714 -> False, 755 -> True, 868 -> False}][999]"
    )
    .unwrap(),
    "AssessmentResultObject[<|AnswerCorrect -> False, Score -> 0|>]"
  );
}

// ─── Grading with a plain-value answer key ───────────────────────────────────

#[test]
fn assessment_plain_value_key_matches() {
  assert_eq!(
    interpret("AssessmentFunction[{1, 2, 3}][2]").unwrap(),
    "AssessmentResultObject[<|AnswerCorrect -> True, Score -> 1|>]"
  );
}

#[test]
fn assessment_plain_value_key_no_match() {
  assert_eq!(
    interpret("AssessmentFunction[{1, 2, 3}][5]").unwrap(),
    "AssessmentResultObject[<|AnswerCorrect -> False, Score -> 0|>]"
  );
}

#[test]
fn assessment_true_false_question() {
  assert_eq!(
    interpret("AssessmentFunction[{True}][True]").unwrap(),
    "AssessmentResultObject[<|AnswerCorrect -> True, Score -> 1|>]"
  );
  assert_eq!(
    interpret("AssessmentFunction[{True}][False]").unwrap(),
    "AssessmentResultObject[<|AnswerCorrect -> False, Score -> 0|>]"
  );
}

// ─── Numeric scores ──────────────────────────────────────────────────────────

#[test]
fn assessment_positive_score_is_correct() {
  assert_eq!(
    interpret(r#"AssessmentFunction[{"cat" -> 10, "dog" -> 0}]["cat"]"#)
      .unwrap(),
    "AssessmentResultObject[<|AnswerCorrect -> True, Score -> 10|>]"
  );
}

#[test]
fn assessment_zero_score_is_incorrect() {
  assert_eq!(
    interpret(r#"AssessmentFunction[{"cat" -> 10, "dog" -> 0}]["dog"]"#)
      .unwrap(),
    "AssessmentResultObject[<|AnswerCorrect -> False, Score -> 0|>]"
  );
}

// ─── AssessmentResultObject property access ──────────────────────────────────

#[test]
fn result_object_score_property() {
  assert_eq!(
    interpret(
      "AssessmentFunction[{714 -> False, 755 -> True}][755][\"Score\"]"
    )
    .unwrap(),
    "1"
  );
}

#[test]
fn result_object_answer_correct_property() {
  assert_eq!(
    interpret(
      "AssessmentFunction[{714 -> False, 755 -> True}][755][\"AnswerCorrect\"]"
    )
    .unwrap(),
    "True"
  );
}

#[test]
fn result_object_all_property() {
  assert_eq!(
    interpret(r#"AssessmentFunction[{"cat" -> 10}]["cat"][All]"#).unwrap(),
    "<|AnswerCorrect -> True, Score -> 10|>"
  );
}

// ─── QuestionObject delegates grading to its assessment ──────────────────────

#[test]
fn question_object_grades_through_assessment() {
  assert_eq!(
    interpret(
      r#"QuestionObject["Is a whale a mammal?", AssessmentFunction[{True}]][True]"#
    )
    .unwrap(),
    "AssessmentResultObject[<|AnswerCorrect -> True, Score -> 1|>]"
  );
}

#[test]
fn question_object_incorrect_answer() {
  assert_eq!(
    interpret(
      r#"QuestionObject["Is a whale a mammal?", AssessmentFunction[{True}]][False]"#
    )
    .unwrap(),
    "AssessmentResultObject[<|AnswerCorrect -> False, Score -> 0|>]"
  );
}

#[test]
fn question_object_single_argument_form() {
  // QuestionObject[assess] derives the question from the assessment alone.
  assert_eq!(
    interpret(
      "QuestionObject[AssessmentFunction[{42 -> True}]][42][\"AnswerCorrect\"]"
    )
    .unwrap(),
    "True"
  );
}

#[test]
fn question_object_property_chain() {
  assert_eq!(
    interpret(
      r#"QuestionObject["q", AssessmentFunction[{True}]][True]["AnswerCorrect"]"#
    )
    .unwrap(),
    "True"
  );
}

// ─── Graphical (SVG) rendering ──────────────────────────────────────────────

#[test]
fn question_object_svg_multiple_choice_panel() {
  // A rule-list answer key renders as a question panel with one radio
  // button per choice and a Submit button.
  let svg = interpret(
    r#"ExportString[QuestionObject["How many career home runs did Hank Aaron hit?", AssessmentFunction[{714 -> False, 755 -> True, 868 -> False}]], "SVG"]"#,
  )
  .unwrap();
  assert!(svg.starts_with("<svg"), "expected an SVG, got: {svg}");
  assert!(svg.contains("How many career home runs did Hank Aaron hit?"));
  for choice in ["714", "755", "868"] {
    assert!(
      svg.contains(&format!(">{choice}</text>")),
      "missing choice {choice}: {svg}"
    );
  }
  assert_eq!(svg.matches("<circle").count(), 3, "one radio per choice");
  assert!(
    svg.contains(">Submit</text>"),
    "missing Submit button: {svg}"
  );
  // The panel must not be the typeset expression dump.
  assert!(
    !svg.contains("QuestionObject"),
    "raw expression leaked: {svg}"
  );
  assert!(!svg.contains("AssessmentFunction"));
}

#[test]
fn question_object_svg_bare_spec_choices() {
  // The assessment may be a bare answer-key list without the
  // AssessmentFunction wrapper.
  let svg = interpret(
    r#"ExportString[QuestionObject["Pick a primary color.", {"Red", "Blue"}], "SVG"]"#,
  )
  .unwrap();
  assert_eq!(svg.matches("<circle").count(), 2);
  assert!(svg.contains(">Red</text>"), "got: {svg}");
  assert!(svg.contains(">Blue</text>"), "got: {svg}");
}

#[test]
fn question_object_svg_input_field_for_non_list_spec() {
  // A non-list answer key has no explicit choices: the answer area is a
  // free-form input field (no radio buttons).
  let svg = interpret(
    r#"ExportString[QuestionObject["What is 2+2?", AssessmentFunction[4]], "SVG"]"#,
  )
  .unwrap();
  assert!(svg.contains("What is 2+2?"), "got: {svg}");
  assert!(!svg.contains("<circle"), "no radio buttons expected: {svg}");
  assert!(svg.contains(">Submit</text>"), "got: {svg}");
  // Two rects besides the frame: the input field and the Submit button.
  assert_eq!(svg.matches("<rect").count(), 3, "got: {svg}");
}

#[test]
fn question_object_svg_wraps_long_question() {
  let svg = interpret(
    r#"ExportString[QuestionObject["Which of the following statements about the career of the baseball player Henry Louis Aaron is accurate and correct?", AssessmentFunction[{1 -> True}]], "SVG"]"#,
  )
  .unwrap();
  // The 122-character question must wrap onto multiple text lines.
  let question_lines = svg
    .lines()
    .filter(|l| {
      l.contains("<text") && !l.contains("Submit") && !l.contains(">1</text>")
    })
    .count();
  assert!(question_lines >= 2, "expected wrapped question, got: {svg}");
}

#[test]
fn question_object_playground_output_is_panel_svg() {
  // In the playground the bare QuestionObject result is displayed as the
  // question panel, not as typeset expression text.
  clear_state();
  interpret(
    r#"QuestionObject["Is a whale a mammal?", AssessmentFunction[{True}]]"#,
  )
  .unwrap();
  let svg = woxi::get_captured_output_svg().expect("output SVG captured");
  assert!(svg.contains("Is a whale a mammal?"), "got: {svg}");
  assert!(svg.contains(">Submit</text>"), "got: {svg}");
  assert!(!svg.contains("QuestionObject"), "got: {svg}");
}
