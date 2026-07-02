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
