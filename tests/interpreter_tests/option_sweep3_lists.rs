// Argument forms of list and structure functions found missing by the
// reference-page parameter sweep (round 3).

use super::*;

mod projection_onto_span {
  use super::*;

  #[test]
  fn onto_a_plane() {
    assert_eq!(
      interpret("Projection[{1, 2, 3}, {{1, 1, 0}, {0, 1, 0}}]").unwrap(),
      "{1, 2, 0}"
    );
  }

  #[test]
  fn machine_vectors() {
    assert_eq!(
      interpret("Projection[{1., 2., 3.}, {{1., 1., 0.}, {0., 1., 0.}}]")
        .unwrap(),
      "{1., 2., 0.}"
    );
  }

  #[test]
  fn dependent_vectors_span_a_line() {
    assert_eq!(
      interpret("Projection[{1, 2, 3}, {{1, 1, 0}, {2, 2, 0}}]").unwrap(),
      "{3/2, 3/2, 0}"
    );
    assert_eq!(
      interpret("Projection[{1, 2, 3}, IdentityMatrix[3]]").unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn complex_vectors_use_the_hermitian_product() {
    assert_eq!(
      interpret("Projection[{3 + I, 2, 2 - I}, {{2, 4, 5 I}, {I, 2, 3}}]")
        .unwrap(),
      "{555/397 + (477*I)/397, 1102/397 - (330*I)/397, \
       562/397 - (389*I)/397}"
    );
    // The residual is orthogonal to every spanning vector.
    assert_eq!(
      interpret(
        "u = {3 + I, 2, 2 - I}; vs = {{2, 4, 5 I}, {I, 2, 3}}; \
         Conjugate[vs] . (u - Projection[u, vs])"
      )
      .unwrap(),
      "{0, 0}"
    );
  }
}

mod insert_along_key_path {
  use super::*;

  #[test]
  fn into_an_association_in_a_list() {
    assert_eq!(
      interpret(
        "Insert[{<|a -> 1, b -> 2|>, <|a -> 2, b -> 4|>}, x -> Infinity, \
         {1, Key[a]}]"
      )
      .unwrap(),
      "{<|x -> Infinity, a -> 1, b -> 2|>, <|a -> 2, b -> 4|>}"
    );
  }

  #[test]
  fn through_nested_associations() {
    assert_eq!(
      interpret(
        "Insert[<|k -> <|a -> 1, b -> 2|>|>, x -> 5, {Key[k], Key[b]}]"
      )
      .unwrap(),
      "<|k -> <|a -> 1, x -> 5, b -> 2|>|>"
    );
    assert_eq!(
      interpret("Insert[<|\"k\" -> {1, 2}|>, x, {\"k\", 2}]").unwrap(),
      "<|k -> {1, x, 2}|>"
    );
  }

  #[test]
  fn a_missing_key_leaves_the_call_unevaluated() {
    assert_eq!(
      interpret("Insert[{<|a -> 1|>}, x -> 1, {1, Key[c]}]").unwrap(),
      "Insert[{<|a -> 1|>}, x -> 1, {1, Key[c]}]"
    );
  }
}

mod random_choice_from_span {
  use super::*;

  #[test]
  fn span_of_integers() {
    assert_eq!(
      interpret(
        "r = RandomChoice[-10 ;; 10, 10]; \
         {Length[r], SubsetQ[Range[-10, 10], r]}"
      )
      .unwrap(),
      "{10, True}"
    );
  }

  #[test]
  fn span_with_a_step() {
    assert_eq!(
      interpret(
        "r = RandomChoice[-10 ;; 10 ;; 2, 500]; \
         {Length[r], SubsetQ[Range[-10, 10, 2], r]}"
      )
      .unwrap(),
      "{500, True}"
    );
    assert_eq!(
      interpret("MemberQ[{1, 2, 3}, RandomChoice[1 ;; 3]]").unwrap(),
      "True"
    );
  }
}

mod substitution_system_step_specs {
  use super::*;

  #[test]
  fn step_range() {
    assert_eq!(
      interpret(
        "SubstitutionSystem[{\"A\" -> \"AB\", \"B\" -> \"A\"}, \"A\", {3, 5}]"
      )
      .unwrap(),
      "{ABAAB, ABAABABA, ABAABABAABAAB}"
    );
  }

  #[test]
  fn step_range_with_increment() {
    assert_eq!(
      interpret(
        "SubstitutionSystem[{\"A\" -> \"AB\", \"B\" -> \"A\"}, \"A\", {1, 5, 2}]"
      )
      .unwrap(),
      "{AB, ABAAB, ABAABABAABAAB}"
    );
    assert_eq!(
      interpret("SubstitutionSystem[{1 -> {1, 0}, 0 -> {0, 1}}, {1}, {2, 3}]")
        .unwrap(),
      "{{1, 0, 0, 1}, {1, 0, 0, 1, 0, 1, 1, 0}}"
    );
  }
}
