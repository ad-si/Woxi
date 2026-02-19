use super::*;

mod dataset_ast {
  use super::*;

  #[test]
  fn dataset_single_association() {
    assert_eq!(
      interpret("Dataset[<|\"Name\" -> \"John\", \"Age\" -> 30, \"City\" -> \"NYC\"|>]").unwrap(),
      "Dataset[<|Name -> John, Age -> 30, City -> NYC|>, TypeSystem`Struct[{Name, Age, City}, {TypeSystem`Atom[String], TypeSystem`Atom[Integer], TypeSystem`Atom[String]}], <||>]"
    );
  }

  #[test]
  fn dataset_list_of_associations() {
    assert_eq!(
      interpret("Dataset[{<|\"Name\" -> \"John\", \"Age\" -> 30|>, <|\"Name\" -> \"Jane\", \"Age\" -> 28|>}]").unwrap(),
      "Dataset[{<|Name -> John, Age -> 30|>, <|Name -> Jane, Age -> 28|>}, TypeSystem`Vector[TypeSystem`Struct[{Name, Age}, {TypeSystem`Atom[String], TypeSystem`Atom[Integer]}], 2], <||>]"
    );
  }

  #[test]
  fn dataset_plain_list() {
    assert_eq!(
      interpret("Dataset[{1, 2, 3}]").unwrap(),
      "Dataset[{1, 2, 3}, TypeSystem`Vector[TypeSystem`Atom[Integer], 3], <||>]"
    );
  }

  #[test]
  fn dataset_atom() {
    assert_eq!(
      interpret("Dataset[42]").unwrap(),
      "Dataset[42, TypeSystem`Atom[Integer], <||>]"
    );
  }

  #[test]
  fn dataset_string_list() {
    assert_eq!(
      interpret("Dataset[{\"a\", \"b\"}]").unwrap(),
      "Dataset[{a, b}, TypeSystem`Vector[TypeSystem`Atom[String], 2], <||>]"
    );
  }

  #[test]
  fn dataset_real_list() {
    assert_eq!(
      interpret("Dataset[{1.5, 2.3}]").unwrap(),
      "Dataset[{1.5, 2.3}, TypeSystem`Vector[TypeSystem`Atom[Real], 2], <||>]"
    );
  }

  #[test]
  fn dataset_mixed_types_tuple() {
    assert_eq!(
      interpret("Dataset[{1, \"a\"}]").unwrap(),
      "Dataset[{1, a}, TypeSystem`Tuple[{TypeSystem`Atom[Integer], TypeSystem`Atom[String]}], <||>]"
    );
  }

  #[test]
  fn dataset_boolean_list() {
    assert_eq!(
      interpret("Dataset[{True, False}]").unwrap(),
      "Dataset[{True, False}, TypeSystem`Vector[TypeSystem`Atom[TypeSystem`Boolean], 2], <||>]"
    );
  }

  #[test]
  fn dataset_nested_lists() {
    assert_eq!(
      interpret("Dataset[{{1, 2}, {3, 4}}]").unwrap(),
      "Dataset[{{1, 2}, {3, 4}}, TypeSystem`Vector[TypeSystem`Vector[TypeSystem`Atom[Integer], 2], 2], <||>]"
    );
  }

  #[test]
  fn dataset_homogeneous_assoc() {
    assert_eq!(
      interpret("Dataset[<|\"a\" -> 1, \"b\" -> 2|>]").unwrap(),
      "Dataset[<|a -> 1, b -> 2|>, TypeSystem`Assoc[TypeSystem`Atom[TypeSystem`Enumeration[a, b]], TypeSystem`Atom[Integer], 2], <||>]"
    );
  }

  #[test]
  fn dataset_assoc_with_list_values() {
    assert_eq!(
      interpret("Dataset[<|\"a\" -> {1, 2}, \"b\" -> {3, 4}|>]").unwrap(),
      "Dataset[<|a -> {1, 2}, b -> {3, 4}|>, TypeSystem`Assoc[TypeSystem`Atom[TypeSystem`Enumeration[a, b]], TypeSystem`Vector[TypeSystem`Atom[Integer], 2], 2], <||>]"
    );
  }

  #[test]
  fn dataset_nested_associations() {
    assert_eq!(
      interpret("Dataset[<|\"a\" -> <|\"x\" -> 1|>, \"b\" -> <|\"x\" -> 2|>|>]").unwrap(),
      "Dataset[<|a -> <|x -> 1|>, b -> <|x -> 2|>|>, TypeSystem`Assoc[TypeSystem`Atom[String], TypeSystem`Struct[{x}, {TypeSystem`Atom[Integer]}], 2], <||>]"
    );
  }

  #[test]
  fn dataset_with_variable() {
    assert_eq!(
      interpret("assoc = <|\"Name\" -> \"John\", \"Age\" -> 30, \"City\" -> \"NYC\"|>; Dataset[assoc]").unwrap(),
      "Dataset[<|Name -> John, Age -> 30, City -> NYC|>, TypeSystem`Struct[{Name, Age, City}, {TypeSystem`Atom[String], TypeSystem`Atom[Integer], TypeSystem`Atom[String]}], <||>]"
    );
  }

  #[test]
  fn dataset_list_variable() {
    assert_eq!(
      interpret("data = {<|\"Name\" -> \"John\", \"Age\" -> 30|>, <|\"Name\" -> \"Jane\", \"Age\" -> 28|>}; Dataset[data]").unwrap(),
      "Dataset[{<|Name -> John, Age -> 30|>, <|Name -> Jane, Age -> 28|>}, TypeSystem`Vector[TypeSystem`Struct[{Name, Age}, {TypeSystem`Atom[String], TypeSystem`Atom[Integer]}], 2], <||>]"
    );
  }

  #[test]
  fn normal_dataset() {
    assert_eq!(
      interpret("Normal[Dataset[<|\"Name\" -> \"John\", \"Age\" -> 30|>]]").unwrap(),
      "<|Name -> John, Age -> 30|>"
    );
  }

  #[test]
  fn head_dataset() {
    assert_eq!(
      interpret("Head[Dataset[<|\"Name\" -> \"John\"|>]]").unwrap(),
      "Dataset"
    );
  }

  #[test]
  fn dataset_integer_keys() {
    assert_eq!(
      interpret("Dataset[<|1 -> \"x\", 2 -> \"y\"|>]").unwrap(),
      "Dataset[<|1 -> x, 2 -> y|>, TypeSystem`Assoc[TypeSystem`Atom[Integer], TypeSystem`Atom[String], 2], <||>]"
    );
  }

  #[test]
  fn dataset_mixed_assoc_values() {
    assert_eq!(
      interpret("Dataset[<|\"a\" -> 1, \"b\" -> \"x\"|>]").unwrap(),
      "Dataset[<|a -> 1, b -> x|>, TypeSystem`Struct[{a, b}, {TypeSystem`Atom[Integer], TypeSystem`Atom[String]}], <||>]"
    );
  }

  #[test]
  fn dataset_already_typed() {
    // Dataset with 3 args should pass through
    assert_eq!(
      interpret("Dataset[{1, 2}, foo, bar]").unwrap(),
      "Dataset[{1, 2}, foo, bar]"
    );
  }
}
