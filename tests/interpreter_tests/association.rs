use super::*;

mod association_ast {
  use super::*;

  #[test]
  fn keys_basic() {
    assert_eq!(
      interpret("Keys[<|\"a\" -> 1, \"b\" -> 2|>]").unwrap(),
      "{a, b}"
    );
  }

  #[test]
  fn values_basic() {
    assert_eq!(
      interpret("Values[<|\"a\" -> 1, \"b\" -> 2|>]").unwrap(),
      "{1, 2}"
    );
  }

  #[test]
  fn keys_with_variable() {
    assert_eq!(
      interpret("h = <|\"x\" -> 10, \"y\" -> 20|>; Keys[h]").unwrap(),
      "{x, y}"
    );
  }

  #[test]
  fn values_with_variable() {
    assert_eq!(
      interpret("h = <|\"x\" -> 10, \"y\" -> 20|>; Values[h]").unwrap(),
      "{10, 20}"
    );
  }

  #[test]
  fn key_exists_q_true() {
    assert_eq!(
      interpret("h = <|\"a\" -> 1|>; KeyExistsQ[h, \"a\"]").unwrap(),
      "True"
    );
  }

  #[test]
  fn key_exists_q_false() {
    assert_eq!(
      interpret("h = <|\"a\" -> 1|>; KeyExistsQ[h, \"b\"]").unwrap(),
      "False"
    );
  }

  #[test]
  fn key_drop_from() {
    assert_eq!(
      interpret("KeyDropFrom[<|\"a\" -> 1, \"b\" -> 2|>, \"a\"]").unwrap(),
      "<|b -> 2|>"
    );
  }

  #[test]
  fn part_extraction() {
    assert_eq!(
      interpret("h = <|\"Green\" -> 2, \"Red\" -> 1|>; h[[\"Green\"]]")
        .unwrap(),
      "2"
    );
  }

  #[test]
  fn map_over_association() {
    assert_eq!(
      interpret("Map[#^2&, <|\"a\" -> 2, \"b\" -> 3|>]").unwrap(),
      "<|a -> 4, b -> 9|>"
    );
  }

  #[test]
  fn nested_association_flattens_and_later_key_overrides() {
    assert_eq!(
      interpret("<|a -> x, b -> y, <|a -> z, d -> t|>|>").unwrap(),
      "<|a -> z, b -> y, d -> t|>"
    );
  }

  #[test]
  fn association_rule_to_nested_association_keeps_as_value() {
    assert_eq!(
      interpret(
        "Association[a -> x, b -> y, c -> Association[d -> t, Association[e -> u]]]"
      )
      .unwrap(),
      "<|a -> x, b -> y, c -> <|d -> t, e -> u|>|>"
    );
  }
}

mod association_part_assignment {
  use super::*;

  #[test]
  fn association_update_existing_key() {
    let result = interpret(
      r#"myHash = <|"A" -> 1, "B" -> 2|>; myHash[["A"]] = 5; myHash"#,
    )
    .unwrap();
    assert_eq!(result, "<|A -> 5, B -> 2|>");
  }

  #[test]
  fn association_add_new_key() {
    let result =
      interpret(r#"myHash = <|"A" -> 1|>; myHash[["B"]] = 2; myHash"#).unwrap();
    assert_eq!(result, "<|A -> 1, B -> 2|>");
  }

  #[test]
  fn association_assign_by_integer_position() {
    // a[[n]] = v selects the n-th value by position (not a new "n" key).
    let result =
      interpret(r#"a = <|"x" -> 1, "y" -> 2|>; a[[1]] = 9; a"#).unwrap();
    assert_eq!(result, "<|x -> 9, y -> 2|>");
  }

  #[test]
  fn association_assign_by_negative_position() {
    let result =
      interpret(r#"a = <|"x" -> 1, "y" -> 2|>; a[[-1]] = 9; a"#).unwrap();
    assert_eq!(result, "<|x -> 1, y -> 9|>");
  }

  #[test]
  fn association_assign_deep_key_path() {
    // Deep Part assignment descending through a nested association by key.
    let result =
      interpret(r#"a = <|"x" -> <|"n" -> 5|>|>; a[["x", "n"]] = 7; a"#)
        .unwrap();
    assert_eq!(result, "<|x -> <|n -> 7|>|>");
  }

  #[test]
  fn association_assign_nested_in_list() {
    // Regression for the `deepcopy` Rosetta task: Part assignment that
    // descends a list, into an association value, then into a nested list.
    let result =
      interpret(r#"a = {<|"k" -> {10, 20}|>}; a[[1, 1, 2]] = 99; a"#).unwrap();
    assert_eq!(result, "{<|k -> {10, 99}|>}");
  }
}

mod association_literal_access {
  use super::*;

  #[test]
  fn literal_single_bracket_access() {
    assert_eq!(
      interpret(r#"<|"a" -> 1, "b" -> 2, "c" -> 3|>["a"]"#).unwrap(),
      "1"
    );
  }

  #[test]
  fn literal_single_bracket_access_middle_key() {
    assert_eq!(
      interpret(r#"<|"a" -> 1, "b" -> 2, "c" -> 3|>["b"]"#).unwrap(),
      "2"
    );
  }

  #[test]
  fn literal_single_bracket_missing_key() {
    assert_eq!(
      interpret(r#"<|"a" -> 1, "b" -> 2|>["d"]"#).unwrap(),
      "Missing[KeyAbsent, d]"
    );
  }

  #[test]
  fn literal_double_bracket_access() {
    assert_eq!(
      interpret(r#"<|"a" -> 1, "b" -> 2, "c" -> 3|>[["a"]]"#).unwrap(),
      "1"
    );
  }

  #[test]
  fn literal_nested_access() {
    assert_eq!(
      interpret(r#"<|"a" -> 1, "b" -> <|"x" -> 10|>|>["b"]["x"]"#).unwrap(),
      "10"
    );
  }

  #[test]
  fn literal_integer_key_access() {
    assert_eq!(
      interpret(r#"<|1 -> "one", 2 -> "two"|>[1]"#).unwrap(),
      "one"
    );
  }

  #[test]
  fn integer_position_access() {
    assert_eq!(
      interpret(r#"<|"a" -> 1, "b" -> 2, "c" -> 3|>[[2]]"#).unwrap(),
      "2"
    );
  }

  #[test]
  fn negative_position_access() {
    assert_eq!(
      interpret(r#"<|"a" -> 1, "b" -> 2, "c" -> 3|>[[-1]]"#).unwrap(),
      "3"
    );
  }
}

mod association_nested_access {
  use super::*;

  #[test]
  fn nested_access_two_levels() {
    let result = interpret(
      r#"assoc = <|"outer" -> <|"inner" -> 8|>|>; assoc["outer", "inner"]"#,
    )
    .unwrap();
    assert_eq!(result, "8");
  }

  #[test]
  fn single_key_access() {
    let result = interpret(r#"assoc = <|"a" -> 1|>; assoc["a"]"#).unwrap();
    assert_eq!(result, "1");
  }
}

mod association_thread {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("AssociationThread[{a, b}, {1, 2}]").unwrap(),
      "<|a -> 1, b -> 2|>"
    );
  }
}

mod association_map {
  use super::*;

  #[test]
  fn reverse_on_association() {
    assert_eq!(
      interpret("AssociationMap[Reverse, <|a -> 1, b -> 2|>]").unwrap(),
      "<|1 -> a, 2 -> b|>"
    );
  }

  #[test]
  fn symbolic_function_on_association() {
    assert_eq!(
      interpret("AssociationMap[f, <|x -> 10, y -> 20|>]").unwrap(),
      "Association[f[x -> 10], f[y -> 20]]"
    );
  }

  #[test]
  fn on_list() {
    assert_eq!(
      interpret("AssociationMap[StringLength, {\"cat\", \"horse\", \"ox\"}]")
        .unwrap(),
      "<|cat -> 3, horse -> 5, ox -> 2|>"
    );
  }

  // Operator form: AssociationMap[f][list].
  #[test]
  fn operator_form() {
    assert_eq!(
      interpret("AssociationMap[f][{1, 2, 3}]").unwrap(),
      "<|1 -> f[1], 2 -> f[2], 3 -> f[3]|>"
    );
  }

  #[test]
  fn operator_stays_inert() {
    assert_eq!(interpret("AssociationMap[f]").unwrap(), "AssociationMap[f]");
  }
}

mod merge {
  use super::*;

  #[test]
  fn merge_with_total() {
    assert_eq!(
      interpret("Merge[{<|a -> 1|>, <|a -> 2, b -> 3|>}, Total]").unwrap(),
      "<|a -> 3, b -> 3|>"
    );
  }

  #[test]
  fn merge_list_of_rules() {
    // wolframscript accepts a flat list of rules, grouping by key.
    assert_eq!(
      interpret("Merge[{a -> 1, a -> 2, b -> 3}, Total]").unwrap(),
      "<|a -> 3, b -> 3|>"
    );
  }

  #[test]
  fn merge_list_of_rule_lists() {
    assert_eq!(
      interpret("Merge[{{a -> 1, b -> 2}, {a -> 10}}, Total]").unwrap(),
      "<|a -> 11, b -> 2|>"
    );
  }

  #[test]
  fn merge_single_rule() {
    assert_eq!(interpret("Merge[{a -> 1}, Total]").unwrap(), "<|a -> 1|>");
  }
}

mod key_map {
  use super::*;

  #[test]
  fn key_map_basic() {
    assert_eq!(
      interpret("KeyMap[f, <|a -> 1, b -> 2|>]").unwrap(),
      "<|f[a] -> 1, f[b] -> 2|>"
    );
  }

  // Operator form: KeyMap[f][assoc].
  #[test]
  fn key_map_operator_form() {
    assert_eq!(
      interpret("KeyMap[f][<|a -> 1, b -> 2|>]").unwrap(),
      "<|f[a] -> 1, f[b] -> 2|>"
    );
  }

  #[test]
  fn key_map_operator_stays_inert() {
    assert_eq!(interpret("KeyMap[f]").unwrap(), "KeyMap[f]");
  }
}

mod key_select {
  use super::*;

  #[test]
  fn key_select_even() {
    assert_eq!(
      interpret("KeySelect[<|1 -> a, 2 -> b, 3 -> c|>, EvenQ]").unwrap(),
      "<|2 -> b|>"
    );
  }

  #[test]
  fn key_select_string_starts_q() {
    assert_eq!(
        interpret(
          "KeySelect[<|\"apple\" -> 1, \"ax\" -> 2, \"banana\" -> 3|>, StringStartsQ[\"a\"]]"
        )
        .unwrap(),
        "<|apple -> 1, ax -> 2|>"
      );
  }

  #[test]
  fn key_select_string_ends_q() {
    assert_eq!(
        interpret(
          "KeySelect[<|\"hello\" -> 1, \"world\" -> 2, \"foo\" -> 3|>, StringEndsQ[\"o\"]]"
        )
        .unwrap(),
        "<|hello -> 1, foo -> 3|>"
      );
  }

  #[test]
  fn key_select_anonymous_function() {
    assert_eq!(
      interpret("KeySelect[<|-3 -> x, 0 -> y, 5 -> z|>, # > 0 &]").unwrap(),
      "<|5 -> z|>"
    );
  }

  // Operator form: KeySelect[crit][assoc].
  #[test]
  fn key_select_operator_form() {
    assert_eq!(
      interpret("KeySelect[EvenQ][<|1 -> x, 2 -> y, 3 -> z|>]").unwrap(),
      "<|2 -> y|>"
    );
    assert_eq!(
      interpret("KeySelect[# =!= b &][<|a -> 1, b -> 2, c -> 3|>]").unwrap(),
      "<|a -> 1, c -> 3|>"
    );
  }

  #[test]
  fn key_select_operator_stays_inert() {
    assert_eq!(interpret("KeySelect[EvenQ]").unwrap(), "KeySelect[EvenQ]");
  }
}

mod key_take {
  use super::*;

  #[test]
  fn key_take_basic() {
    assert_eq!(
      interpret("KeyTake[<|a -> 1, b -> 2, c -> 3|>, {a, c}]").unwrap(),
      "<|a -> 1, c -> 3|>"
    );
  }

  // Operator form: KeyTake[keys][assoc].
  #[test]
  fn key_take_operator_form() {
    assert_eq!(
      interpret("KeyTake[{a, c}][<|a -> 1, b -> 2, c -> 3|>]").unwrap(),
      "<|a -> 1, c -> 3|>"
    );
  }

  #[test]
  fn key_take_operator_stays_inert() {
    assert_eq!(interpret("KeyTake[{a, c}]").unwrap(), "KeyTake[{a, c}]");
  }
}

mod key_drop {
  use super::*;

  #[test]
  fn key_drop_basic() {
    assert_eq!(
      interpret("KeyDrop[<|a -> 1, b -> 2, c -> 3|>, {a}]").unwrap(),
      "<|b -> 2, c -> 3|>"
    );
  }

  // Operator form: KeyDrop[keys][assoc], including a bare single key.
  #[test]
  fn key_drop_operator_form() {
    assert_eq!(
      interpret("KeyDrop[{b}][<|a -> 1, b -> 2, c -> 3|>]").unwrap(),
      "<|a -> 1, c -> 3|>"
    );
    assert_eq!(
      interpret("KeyDrop[b][<|a -> 1, b -> 2, c -> 3|>]").unwrap(),
      "<|a -> 1, c -> 3|>"
    );
  }

  #[test]
  fn key_drop_operator_stays_inert() {
    assert_eq!(interpret("KeyDrop[b]").unwrap(), "KeyDrop[b]");
  }
}

mod key_sort {
  use super::*;

  #[test]
  fn string_keys() {
    assert_eq!(
      interpret("KeySort[<|c -> 3, a -> 1, b -> 2|>]").unwrap(),
      "<|a -> 1, b -> 2, c -> 3|>"
    );
  }

  #[test]
  fn integer_keys() {
    assert_eq!(
      interpret("KeySort[<|3 -> c, 1 -> a, 2 -> b|>]").unwrap(),
      "<|1 -> a, 2 -> b, 3 -> c|>"
    );
  }

  #[test]
  fn already_sorted() {
    assert_eq!(
      interpret("KeySort[<|a -> 1, b -> 2|>]").unwrap(),
      "<|a -> 1, b -> 2|>"
    );
  }
}

mod key_value_map {
  use super::*;

  #[test]
  fn named_function() {
    assert_eq!(
      interpret("KeyValueMap[f, <|a -> 1, b -> 2|>]").unwrap(),
      "{f[a, 1], f[b, 2]}"
    );
  }

  #[test]
  fn list_function() {
    assert_eq!(
      interpret("KeyValueMap[List, <|a -> 1, b -> 2|>]").unwrap(),
      "{{a, 1}, {b, 2}}"
    );
  }

  #[test]
  fn anonymous_function() {
    assert_eq!(
      interpret("KeyValueMap[#2 &, <|x -> 10, y -> 20|>]").unwrap(),
      "{10, 20}"
    );
  }

  // Operator form: KeyValueMap[f][assoc].
  #[test]
  fn operator_form() {
    assert_eq!(
      interpret("KeyValueMap[f][<|a -> 1, b -> 2|>]").unwrap(),
      "{f[a, 1], f[b, 2]}"
    );
  }
}

mod reverse_rule {
  use super::*;

  #[test]
  fn reverse_rule() {
    assert_eq!(interpret("Reverse[a -> b]").unwrap(), "b -> a");
  }

  #[test]
  fn reverse_rule_numeric() {
    assert_eq!(interpret("Reverse[1 -> 2]").unwrap(), "2 -> 1");
  }
}

mod reverse_association {
  use super::*;

  #[test]
  fn reverse_association() {
    assert_eq!(
      interpret("Reverse[<|a -> 1, b -> 2, c -> 3|>]").unwrap(),
      "<|c -> 3, b -> 2, a -> 1|>"
    );
  }

  #[test]
  fn reverse_empty_association() {
    assert_eq!(interpret("Reverse[<||>]").unwrap(), "<||>");
  }

  #[test]
  fn reverse_single_element_association() {
    assert_eq!(interpret("Reverse[<|a -> 1|>]").unwrap(), "<|a -> 1|>");
  }
}

mod sort_association {
  use super::*;

  #[test]
  fn sort_association_by_values() {
    assert_eq!(
      interpret("Sort[<|c -> 3, a -> 1, b -> 2|>]").unwrap(),
      "<|a -> 1, b -> 2, c -> 3|>"
    );
  }

  #[test]
  fn sort_empty_association() {
    assert_eq!(interpret("Sort[<||>]").unwrap(), "<||>");
  }

  #[test]
  fn sort_by_association() {
    assert_eq!(
      interpret("SortBy[<|c -> 3, a -> 1, b -> 2|>, Identity]").unwrap(),
      "<|a -> 1, b -> 2, c -> 3|>"
    );
  }

  #[test]
  fn sort_by_association_descending() {
    assert_eq!(
      interpret("SortBy[<|c -> 3, a -> 1, b -> 2|>, Minus]").unwrap(),
      "<|c -> 3, b -> 2, a -> 1|>"
    );
  }
}

mod value_q {
  use super::*;

  #[test]
  fn value_q_defined() {
    assert_eq!(interpret("x = 5; ValueQ[x]").unwrap(), "True");
  }

  #[test]
  fn value_q_undefined() {
    assert_eq!(interpret("ValueQ[undefined]").unwrap(), "False");
  }

  #[test]
  fn value_q_cleared() {
    assert_eq!(interpret("x = 5; ClearAll[x]; ValueQ[x]").unwrap(), "False");
  }

  #[test]
  fn value_q_downvalue_match() {
    assert_eq!(
      interpret("Foo[x_Integer] := Mod[x, 2]; ValueQ[Foo[8]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn value_q_downvalue_nonmatching_args() {
    // wolframscript returns True whenever the head has any DownValues,
    // regardless of whether the specific argument matches a pattern.
    assert_eq!(
      interpret("Foo[x_Integer] := Mod[x, 2]; ValueQ[Foo[{a, b}]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn value_q_undefined_head() {
    assert_eq!(interpret("ValueQ[Foo[1]]").unwrap(), "False");
  }

  #[test]
  fn value_q_specific_downvalue() {
    assert_eq!(
      interpret("Foo[5] = \"five\"; ValueQ[Foo[5]]").unwrap(),
      "True"
    );
  }
}

mod clear {
  use super::*;

  #[test]
  fn clear_variable() {
    assert_eq!(interpret("x = 5; Clear[x]; x").unwrap(), "x");
  }

  #[test]
  fn clear_function_definition() {
    assert_eq!(interpret("f[x_] := x^2; Clear[f]; f[3]").unwrap(), "f[3]");
  }

  #[test]
  fn clear_multiple() {
    assert_eq!(
      interpret("a = 1; b = 2; Clear[a, b]; {a, b}").unwrap(),
      "{a, b}"
    );
  }

  #[test]
  fn clear_preserves_others() {
    assert_eq!(
      interpret("a = 1; b = 2; Clear[a]; {a, b}").unwrap(),
      "{a, 2}"
    );
  }

  #[test]
  fn clear_preserves_attributes() {
    assert_eq!(
      interpret("ClearAll[g]; SetAttributes[g, Flat]; Clear[g]; Attributes[g]")
        .unwrap(),
      "{Flat}"
    );
  }

  #[test]
  fn clear_returns_null() {
    assert_eq!(interpret("x = 5; Clear[x]").unwrap(), "\0");
  }
}

mod clear_all {
  use super::*;

  #[test]
  fn clear_variable() {
    assert_eq!(interpret("x = 5; ClearAll[x]; x").unwrap(), "x");
  }

  #[test]
  fn clear_multiple() {
    assert_eq!(
      interpret("a = 1; b = 2; ClearAll[a, b]; {a, b}").unwrap(),
      "{a, b}"
    );
  }

  #[test]
  fn clear_preserves_others() {
    assert_eq!(
      interpret("a = 1; b = 2; ClearAll[a]; {a, b}").unwrap(),
      "{a, 2}"
    );
  }
}

mod multiline_association {
  use super::*;

  #[test]
  fn basic_multiline() {
    clear_state();
    assert_eq!(
      interpret("a = <|\n \"x\" -> 1,\n \"y\" -> 2\n|>\na").unwrap(),
      "<|x -> 1, y -> 2|>"
    );
  }

  #[test]
  fn multiline_with_print() {
    clear_state();
    let res = interpret_with_stdout(
      "coinWeights = <|\n \"1c\" -> 230,\n \"2c\" -> 306\n|>;\nPrint[coinWeights]",
    )
    .unwrap();
    assert_eq!(res.stdout.trim(), "<|1c -> 230, 2c -> 306|>");
    assert_eq!(res.result, "\0");
  }

  #[test]
  fn multiline_part_extraction() {
    clear_state();
    assert_eq!(
      interpret(
        "a = <|\n \"x\" -> 10,\n \"y\" -> 20,\n \"z\" -> 30\n|>\na[[\"y\"]]"
      )
      .unwrap(),
      "20"
    );
  }

  #[test]
  fn multiline_keys_values() {
    clear_state();
    assert_eq!(
      interpret("a = <|\n \"a\" -> 1,\n \"b\" -> 2\n|>\nKeys[a]").unwrap(),
      "{a, b}"
    );
  }

  #[test]
  fn nested_multiline() {
    clear_state();
    assert_eq!(
      interpret(
        "a = <|\"x\" -> <|\n \"nested\" -> 42\n|>|>\na[[\"x\"]][[\"nested\"]]"
      )
      .unwrap(),
      "42"
    );
  }

  #[test]
  fn multiline_many_entries() {
    clear_state();
    assert_eq!(
      interpret(
        "d = <|\n \"a\" -> 1,\n \"b\" -> 2,\n \"c\" -> 3,\n \"d\" -> 4,\n \"e\" -> 5\n|>\nLength[Keys[d]]"
      )
      .unwrap(),
      "5"
    );
  }
}

mod key {
  use super::*;

  #[test]
  fn basic_string_key() {
    assert_eq!(
      interpret("Key[\"a\"][<|\"a\" -> 1, \"b\" -> 2|>]").unwrap(),
      "1"
    );
  }

  #[test]
  fn integer_key() {
    assert_eq!(
      interpret("Key[2][<|1 -> \"a\", 2 -> \"b\"|>]").unwrap(),
      "b"
    );
  }

  #[test]
  fn missing_key() {
    assert_eq!(
      interpret("Key[\"missing\"][<|\"a\" -> 1|>]").unwrap(),
      "Missing[KeyAbsent, missing]"
    );
  }

  #[test]
  fn symbol_key() {
    assert_eq!(interpret("Key[x][<|x -> 42, y -> 99|>]").unwrap(), "42");
  }

  #[test]
  fn unevaluated_without_arg() {
    assert_eq!(interpret("Key[\"a\"]").unwrap(), "Key[a]");
  }

  #[test]
  fn non_association_arg() {
    assert_eq!(
      interpret("Key[\"a\"][{1, 2, 3}]").unwrap(),
      "Key[a][{1, 2, 3}]"
    );
  }

  #[test]
  fn last_key_in_association() {
    assert_eq!(
      interpret("Key[\"c\"][<|\"a\" -> 1, \"b\" -> 2, \"c\" -> 3|>]").unwrap(),
      "3"
    );
  }

  #[test]
  fn with_variable() {
    assert_eq!(
      interpret("assoc = <|\"x\" -> 10, \"y\" -> 20|>; Key[\"y\"][assoc]")
        .unwrap(),
      "20"
    );
  }
}

mod key_union {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("KeyUnion[{<|a -> 1, b -> 2|>, <|b -> 3, c -> 4|>}]").unwrap(),
      "{<|a -> 1, b -> 2, c -> Missing[KeyAbsent, c]|>, <|a -> Missing[KeyAbsent, a], b -> 3, c -> 4|>}"
    );
  }

  #[test]
  fn disjoint_keys() {
    assert_eq!(
      interpret("KeyUnion[{<|a -> 1|>, <|b -> 2|>}]").unwrap(),
      "{<|a -> 1, b -> Missing[KeyAbsent, b]|>, <|a -> Missing[KeyAbsent, a], b -> 2|>}"
    );
  }

  #[test]
  fn identical_keys() {
    assert_eq!(
      interpret("KeyUnion[{<|a -> 1|>, <|a -> 2|>}]").unwrap(),
      "{<|a -> 1|>, <|a -> 2|>}"
    );
  }
}

mod key_intersection {
  use super::*;

  #[test]
  fn keeps_only_common_keys() {
    // Each association is restricted to the keys common to all, with its own
    // values.
    assert_eq!(
      interpret("KeyIntersection[{<|a -> 1, b -> 2|>, <|b -> 3, c -> 4|>}]")
        .unwrap(),
      "{<|b -> 2|>, <|b -> 3|>}"
    );
  }

  #[test]
  fn single_association_unchanged() {
    assert_eq!(
      interpret("KeyIntersection[{<|a -> 1, b -> 2|>}]").unwrap(),
      "{<|a -> 1, b -> 2|>}"
    );
  }

  #[test]
  fn no_common_keys_gives_empty_associations() {
    assert_eq!(
      interpret("KeyIntersection[{<|a -> 1|>, <|b -> 2|>}]").unwrap(),
      "{<||>, <||>}"
    );
  }

  #[test]
  fn common_keys_ordered_by_first_association() {
    // The key order follows the first association and is applied to all.
    assert_eq!(
      interpret("KeyIntersection[{<|b -> 1, a -> 2|>, <|a -> 3, b -> 4|>}]")
        .unwrap(),
      "{<|b -> 1, a -> 2|>, <|b -> 4, a -> 3|>}"
    );
  }

  #[test]
  fn non_list_argument_stays_unevaluated() {
    assert_eq!(
      interpret("KeyIntersection[5]").unwrap(),
      "KeyIntersection[5]"
    );
  }
}

mod key_complement {
  use super::*;

  #[test]
  fn first_keys_absent_from_the_rest() {
    assert_eq!(
      interpret(
        "KeyComplement[{<|a -> 1, b -> 2, c -> 5|>, <|b -> 3|>, <|c -> 4|>}]"
      )
      .unwrap(),
      "<|a -> 1|>"
    );
  }

  #[test]
  fn single_association_unchanged() {
    assert_eq!(
      interpret("KeyComplement[{<|a -> 1, b -> 2|>}]").unwrap(),
      "<|a -> 1, b -> 2|>"
    );
  }

  #[test]
  fn all_keys_removed_gives_empty_association() {
    assert_eq!(
      interpret("KeyComplement[{<|a -> 1|>, <|a -> 9|>}]").unwrap(),
      "<||>"
    );
  }

  #[test]
  fn non_list_argument_stays_unevaluated() {
    assert_eq!(interpret("KeyComplement[5]").unwrap(), "KeyComplement[5]");
  }
}

mod association_list_operations {
  use super::*;

  #[test]
  fn append_to_association() {
    assert_eq!(
      interpret("Append[<|a -> 1, b -> 2|>, c -> 3]").unwrap(),
      "<|a -> 1, b -> 2, c -> 3|>"
    );
  }

  #[test]
  fn append_to_empty_association() {
    assert_eq!(interpret("Append[<||>, a -> 1]").unwrap(), "<|a -> 1|>");
  }

  #[test]
  fn prepend_to_association() {
    assert_eq!(
      interpret("Prepend[<|a -> 1, b -> 2|>, c -> 3]").unwrap(),
      "<|c -> 3, a -> 1, b -> 2|>"
    );
  }

  #[test]
  fn append_duplicate_key() {
    // Appending with existing key removes old entry, adds new at end
    assert_eq!(
      interpret("Append[<|a -> 1, b -> 2|>, a -> 10]").unwrap(),
      "<|b -> 2, a -> 10|>"
    );
  }

  #[test]
  fn prepend_duplicate_key() {
    // Prepending with existing key removes old entry, adds new at front
    assert_eq!(
      interpret("Prepend[<|a -> 1, b -> 2|>, a -> 10]").unwrap(),
      "<|a -> 10, b -> 2|>"
    );
  }

  #[test]
  fn join_two_associations() {
    assert_eq!(
      interpret("Join[<|a -> 1|>, <|b -> 2|>]").unwrap(),
      "<|a -> 1, b -> 2|>"
    );
  }

  #[test]
  fn join_three_associations() {
    assert_eq!(
      interpret("Join[<|a -> 1|>, <|b -> 2|>, <|c -> 3|>]").unwrap(),
      "<|a -> 1, b -> 2, c -> 3|>"
    );
  }

  #[test]
  fn join_associations_duplicate_keys() {
    // Later values should override earlier ones for the same key
    assert_eq!(
      interpret("Join[<|a -> 1, b -> 2|>, <|b -> 3, c -> 4|>]").unwrap(),
      "<|a -> 1, b -> 3, c -> 4|>"
    );
    assert_eq!(
      interpret("Join[<|a -> 1, b -> 2|>, <|a -> 10|>]").unwrap(),
      "<|a -> 10, b -> 2|>"
    );
  }

  #[test]
  fn select_from_association() {
    assert_eq!(
      interpret("Select[<|a -> 1, b -> 2, c -> 3|>, (# > 1) &]").unwrap(),
      "<|b -> 2, c -> 3|>"
    );
  }

  #[test]
  fn select_none_from_association() {
    assert_eq!(
      interpret("Select[<|a -> 1, b -> 2|>, (# > 10) &]").unwrap(),
      "<||>"
    );
  }

  #[test]
  fn select_with_limit_from_association() {
    assert_eq!(
      interpret("Select[<|a -> 1, b -> 2, c -> 3|>, (# > 0) &, 2]").unwrap(),
      "<|a -> 1, b -> 2|>"
    );
  }

  #[test]
  fn association_constructor_from_list() {
    assert_eq!(
      interpret("Association[{a -> 1, b -> 2}]").unwrap(),
      "<|a -> 1, b -> 2|>"
    );
  }

  #[test]
  fn association_constructor_from_rules() {
    assert_eq!(
      interpret("Association[a -> 1, b -> 2]").unwrap(),
      "<|a -> 1, b -> 2|>"
    );
  }

  #[test]
  fn association_constructor_empty_list() {
    assert_eq!(interpret("Association[{}]").unwrap(), "<||>");
  }

  #[test]
  fn association_constructor_from_evaluated_argument() {
    // Association evaluates its argument to normalize it: a Table/Map that
    // produces a list of rules builds the association.
    assert_eq!(
      interpret("Association[Table[i -> i^2, {i, 3}]]").unwrap(),
      "<|1 -> 1, 2 -> 4, 3 -> 9|>"
    );
    assert_eq!(
      interpret("Association[Map[# -> #^2 &, {1, 2, 3}]]").unwrap(),
      "<|1 -> 1, 2 -> 4, 3 -> 9|>"
    );
  }

  #[test]
  fn association_constructor_keeps_held_form_when_not_rules() {
    // If the evaluated argument is not a valid association structure, the
    // original held form is preserved (HoldAllComplete).
    assert_eq!(
      interpret("Association[Range[3]]").unwrap(),
      "Association[Range[3]]"
    );
    assert_eq!(interpret("Association[x]").unwrap(), "Association[x]");
  }

  #[test]
  fn association_splices_nested_associations() {
    // Inner Association arguments have their pairs spliced into the outer one,
    // matching Wolfram: Association[a -> 1, Association[b -> 2]] -> <|a -> 1, b -> 2|>.
    assert_eq!(
      interpret("Association[a -> 1, Association[b -> 2, c -> 3], d -> 4]")
        .unwrap(),
      "<|a -> 1, b -> 2, c -> 3, d -> 4|>"
    );
  }

  #[test]
  fn association_with_nested_value_preserves_inner() {
    // A nested association as a rule's *value* is preserved as a value
    // (not spliced).
    assert_eq!(
      interpret(
        "Association[a -> x, b -> y, c -> Association[d -> t, Association[e -> u]]]"
      )
      .unwrap(),
      "<|a -> x, b -> y, c -> <|d -> t, e -> u|>|>"
    );
  }

  #[test]
  fn association_literal_accepts_rule_delayed() {
    // `:>` is valid inside <|...|>; AssociationQ recognizes it as an association.
    assert_eq!(
      interpret("AssociationQ[<|a -> 1, b :> 2|>]").unwrap(),
      "True"
    );
  }

  #[test]
  fn association_literal_non_rule_items_parse() {
    // <|a, b|> parses as Association[a, b] (bare-symbol args, not rules);
    // AssociationQ returns False because it isn't a well-formed association.
    assert_eq!(interpret("AssociationQ[<|a, b|>]").unwrap(), "False");
  }

  #[test]
  fn take_from_association() {
    assert_eq!(
      interpret("Take[<|a -> 1, b -> 2, c -> 3|>, 2]").unwrap(),
      "<|a -> 1, b -> 2|>"
    );
  }

  #[test]
  fn take_negative_from_association() {
    assert_eq!(
      interpret("Take[<|a -> 1, b -> 2, c -> 3|>, -1]").unwrap(),
      "<|c -> 3|>"
    );
  }

  #[test]
  fn first_of_association() {
    assert_eq!(interpret("First[<|a -> 1, b -> 2, c -> 3|>]").unwrap(), "1");
  }

  #[test]
  fn last_of_association() {
    assert_eq!(interpret("Last[<|a -> 1, b -> 2, c -> 3|>]").unwrap(), "3");
  }

  #[test]
  fn drop_from_association() {
    assert_eq!(
      interpret("Drop[<|a -> 1, b -> 2, c -> 3|>, 1]").unwrap(),
      "<|b -> 2, c -> 3|>"
    );
  }

  #[test]
  fn drop_negative_from_association() {
    assert_eq!(
      interpret("Drop[<|a -> 1, b -> 2, c -> 3|>, -1]").unwrap(),
      "<|a -> 1, b -> 2|>"
    );
  }
}

mod member_q_association {
  use super::*;

  #[test]
  fn member_q_value_present() {
    assert_eq!(interpret("MemberQ[<|a -> 1, b -> 2|>, 2]").unwrap(), "True");
  }

  #[test]
  fn member_q_key_not_checked() {
    assert_eq!(
      interpret("MemberQ[<|a -> 1, b -> 2|>, a]").unwrap(),
      "False"
    );
  }

  #[test]
  fn member_q_value_absent() {
    assert_eq!(
      interpret("MemberQ[<|a -> 1, b -> 2|>, 3]").unwrap(),
      "False"
    );
  }
}

mod count_association {
  use super::*;

  #[test]
  fn count_matching_values() {
    assert_eq!(
      interpret("Count[<|a -> 1, b -> 2, c -> 1|>, 1]").unwrap(),
      "2"
    );
  }

  #[test]
  fn count_no_matches() {
    assert_eq!(interpret("Count[<|a -> 1, b -> 2|>, 3]").unwrap(), "0");
  }
}

mod aggregate_association {
  use super::*;

  #[test]
  fn total_association() {
    assert_eq!(interpret("Total[<|a -> 1, b -> 2, c -> 3|>]").unwrap(), "6");
  }

  #[test]
  fn mean_association() {
    assert_eq!(interpret("Mean[<|a -> 1, b -> 2, c -> 3|>]").unwrap(), "2");
  }

  #[test]
  fn min_association() {
    assert_eq!(interpret("Min[<|a -> 1, b -> 2, c -> 3|>]").unwrap(), "1");
  }

  #[test]
  fn max_association() {
    assert_eq!(interpret("Max[<|a -> 1, b -> 2, c -> 3|>]").unwrap(), "3");
  }
}

mod catenate_association {
  use super::*;

  #[test]
  fn catenate_associations() {
    assert_eq!(
      interpret("Catenate[{<|a -> 1|>, <|b -> 2|>}]").unwrap(),
      "{1, 2}"
    );
  }

  #[test]
  fn catenate_mixed_lists_and_associations() {
    assert_eq!(
      interpret("Catenate[{{1, 2}, <|a -> 3|>}]").unwrap(),
      "{1, 2, 3}"
    );
  }
}

mod apply_association {
  use super::*;

  #[test]
  fn apply_plus_to_association() {
    assert_eq!(
      interpret("Apply[Plus, <|a -> 1, b -> 2, c -> 3|>]").unwrap(),
      "6"
    );
  }

  #[test]
  fn apply_f_to_association() {
    assert_eq!(
      interpret("Apply[f, <|a -> 1, b -> 2|>]").unwrap(),
      "f[1, 2]"
    );
  }
}

mod association_thread_rule_form {
  use super::*;

  #[test]
  fn association_thread_rule_form() {
    assert_eq!(
      interpret("AssociationThread[{a, b, c} -> {1, 2, 3}]").unwrap(),
      "<|a -> 1, b -> 2, c -> 3|>"
    );
  }
}

mod lookup {
  use super::*;

  #[test]
  fn lookup_threads_over_list_of_associations() {
    assert_eq!(
      interpret("Lookup[{<|a -> 1, b -> 2|>, <|a -> 3, c -> 4|>}, a]").unwrap(),
      "{1, 3}"
    );
  }

  #[test]
  fn lookup_threads_with_default() {
    assert_eq!(
      interpret(r#"Lookup[{<|a -> 1|>, <|b -> 2|>}, a, "miss"]"#).unwrap(),
      "{1, miss}"
    );
  }

  #[test]
  fn lookup_list_of_keys() {
    assert_eq!(
      interpret("Lookup[<|a -> 1, b -> 2, c -> 3|>, {a, b}]").unwrap(),
      "{1, 2}"
    );
  }

  #[test]
  fn lookup_list_of_keys_with_missing() {
    assert_eq!(
      interpret("Lookup[<|a -> 1, b -> 2, c -> 3|>, {b, c, d}]").unwrap(),
      "{2, 3, Missing[KeyAbsent, d]}"
    );
  }

  #[test]
  fn lookup_list_of_keys_with_default() {
    assert_eq!(
      interpret(r#"Lookup[<|a -> 1, b -> 2, c -> 3|>, {a, d}, "missing"]"#)
        .unwrap(),
      "{1, missing}"
    );
  }

  #[test]
  fn lookup_single_key_still_works() {
    assert_eq!(interpret("Lookup[<|a -> 1, b -> 2|>, a]").unwrap(), "1");
  }

  #[test]
  fn lookup_single_key_missing() {
    assert_eq!(
      interpret("Lookup[<|a -> 1, b -> 2|>, c]").unwrap(),
      "Missing[KeyAbsent, c]"
    );
  }

  #[test]
  fn lookup_single_key_with_default() {
    assert_eq!(
      interpret(r#"Lookup[<|a -> 1, b -> 2|>, c, "default"]"#).unwrap(),
      "default"
    );
  }

  // Operator form: Lookup[key][assoc].
  #[test]
  fn lookup_operator_form() {
    assert_eq!(interpret("Lookup[a][<|a -> 1, b -> 2|>]").unwrap(), "1");
    assert_eq!(
      interpret("Lookup[c][<|a -> 1, b -> 2|>]").unwrap(),
      "Missing[KeyAbsent, c]"
    );
  }

  #[test]
  fn lookup_operator_form_key_list() {
    assert_eq!(
      interpret("Lookup[{a, c}][<|a -> 1, b -> 2|>]").unwrap(),
      "{1, Missing[KeyAbsent, c]}"
    );
  }

  #[test]
  fn lookup_operator_stays_inert() {
    assert_eq!(interpret("Lookup[a]").unwrap(), "Lookup[a]");
  }
}

mod cases {
  use super::super::case_helpers::assert_case;

  #[test]
  fn association() {
    assert_case(
      r#"Head[<|a -> x, b -> y, c -> z|>]; <|a -> x, b -> y|>; Association[{a -> x^2, b -> y}]"#,
      r#"<|a -> x^2, b -> y|>"#,
    );
  }
  #[test]
  fn association_literal() {
    assert_case(
      r#"Head[<|a -> x, b -> y, c -> z|>]; <|a -> x, b -> y|>; Association[{a -> x^2, b -> y}]; <|a -> x, b -> y, <|a -> z, d -> t|>|>"#,
      r#"<|a -> z, b -> y, d -> t|>"#,
    );
  }
  #[test]
  fn association_q_1() {
    assert_case(r#"AssociationQ[<|a -> 1, b :> 2|>]"#, r#"True"#);
  }
  #[test]
  fn association_q_2() {
    assert_case(
      r#"AssociationQ[<|a -> 1, b :> 2|>]; AssociationQ[<|a, b|>]"#,
      r#"False"#,
    );
  }
  #[test]
  fn keys_1() {
    assert_case(r#"Keys[<|a -> x, b -> y|>]"#, r#"{a, b}"#);
  }
  #[test]
  fn keys_2() {
    assert_case(
      r#"Keys[<|a -> x, b -> y|>]; Keys[{a -> x, b -> y}]"#,
      r#"{a, b}"#,
    );
  }
  #[test]
  fn keys_3() {
    assert_case(
      r#"Keys[<|a -> x, b -> y|>]; Keys[{a -> x, b -> y}]; Keys[{<|a -> x, b -> y|>, {w -> z, {}}}]"#,
      r#"{{a, b}, {w, {}}}"#,
    );
  }
  #[test]
  fn keys_4() {
    assert_case(
      r#"Keys[<|a -> x, b -> y|>]; Keys[{a -> x, b -> y}]; Keys[{<|a -> x, b -> y|>, {w -> z, {}}}]; Keys[{c -> z, b -> y, a -> x}]"#,
      r#"{c, b, a}"#,
    );
  }
  #[test]
  fn values_1() {
    assert_case(r#"Values[<|a -> x, b -> y|>]"#, r#"{x, y}"#);
  }
  #[test]
  fn values_2() {
    assert_case(
      r#"Values[<|a -> x, b -> y|>]; Values[{a -> x, b -> y}]"#,
      r#"{x, y}"#,
    );
  }
  #[test]
  fn values_3() {
    assert_case(
      r#"Values[<|a -> x, b -> y|>]; Values[{a -> x, b -> y}]; Values[{<|a -> x, b -> y|>, {c -> z, {}}}]"#,
      r#"{{x, y}, {z, {}}}"#,
    );
  }
  #[test]
  fn values_4() {
    assert_case(
      r#"Values[<|a -> x, b -> y|>]; Values[{a -> x, b -> y}]; Values[{<|a -> x, b -> y|>, {c -> z, {}}}]; Values[{c -> z, b -> y, a -> x}]"#,
      r#"{z, y, x}"#,
    );
  }
}

mod query {
  use super::*;

  #[test]
  fn descending_operators() {
    // All maps the rest of the spec over elements
    assert_eq!(
      interpret(
        "Query[All, \"a\"][{<|\"a\" -> 1, \"b\" -> 2|>, <|\"a\" -> 3, \"b\" -> 4|>}]"
      )
      .unwrap(),
      "{1, 3}"
    );
    // Keys and parts descend
    assert_eq!(
      interpret("Query[\"a\"][<|\"a\" -> 1, \"b\" -> 2|>]").unwrap(),
      "1"
    );
    assert_eq!(interpret("Query[2][{10, 20, 30}]").unwrap(), "20");
    assert_eq!(
      interpret("Query[\"a\", 2][<|\"a\" -> {10, 20}|>]").unwrap(),
      "20"
    );
    assert_eq!(
      interpret(
        "Query[All, \"a\", 2][{<|\"a\" -> {1, 2}|>, <|\"a\" -> {3, 4}|>}]"
      )
      .unwrap(),
      "{2, 4}"
    );
    // List operators filter/sort/take at their level, then the rest
    // maps over survivors
    assert_eq!(
      interpret("Query[Select[#[\"a\"] > 1 &], \"a\"][{<|\"a\" -> 1|>, <|\"a\" -> 3|>, <|\"a\" -> 5|>}]").unwrap(),
      "{3, 5}"
    );
    assert_eq!(
      interpret(
        "Query[SortBy[#[\"a\"] &], \"a\"][{<|\"a\" -> 3|>, <|\"a\" -> 1|>}]"
      )
      .unwrap(),
      "{1, 3}"
    );
    assert_eq!(
      interpret("Query[TakeLargest[2]][{5, 8, 6, 7}]").unwrap(),
      "{8, 7}"
    );
  }

  #[test]
  fn ascending_operators() {
    // Functions apply on the way up, after deeper levels
    assert_eq!(
      interpret("Query[Total, \"a\"][{<|\"a\" -> 1|>, <|\"a\" -> 3|>}]")
        .unwrap(),
      "4"
    );
    assert_eq!(
      interpret("Query[Mean, \"a\"][{<|\"a\" -> 1|>, <|\"a\" -> 3|>}]")
        .unwrap(),
      "2"
    );
    assert_eq!(interpret("Query[Total][{1, 2, 3}]").unwrap(), "6");
    assert_eq!(
      interpret("Query[All, Total][{{1, 2}, {3, 4}}]").unwrap(),
      "{3, 7}"
    );
    assert_eq!(
      interpret("Query[All, All, f][{{1, 2}, {3, 4}}]").unwrap(),
      "{{f[1], f[2]}, {f[3], f[4]}}"
    );
  }

  #[test]
  fn identity_and_missing() {
    assert_eq!(interpret("Query[All][{1, 2, 3}]").unwrap(), "{1, 2, 3}");
    assert_eq!(
      interpret("Query[\"x\"][<|\"a\" -> 1|>]").unwrap(),
      "Missing[KeyAbsent, x]"
    );
  }
}

mod part_on_associations {
  use super::*;

  #[test]
  fn part_string_key_lookup() {
    assert_eq!(interpret(r#"<|"a" -> 1, "b" -> 2|>[["b"]]"#).unwrap(), "2");
  }

  #[test]
  fn part_string_key_absent_returns_missing() {
    // wolframscript returns Missing[KeyAbsent, key] silently (no message)
    assert_eq!(
      interpret(r#"<|"b" -> 1|>[["a"]]"#).unwrap(),
      "Missing[KeyAbsent, a]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(msgs.is_empty(), "expected no messages, got {:?}", msgs);
  }

  #[test]
  fn part_string_key_does_not_match_symbol_key() {
    assert_eq!(
      interpret(r#"<|a -> 1|>[["a"]]"#).unwrap(),
      "Missing[KeyAbsent, a]"
    );
  }

  #[test]
  fn part_symbol_index_emits_pkspec1() {
    // A bare symbol is not a valid association part spec — pkspec1 +
    // unevaluated (it does NOT match a symbol key; that needs Key[...])
    assert_eq!(interpret("<|x -> 1|>[[x]]").unwrap(), "<|x -> 1|>[[x]]");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "Part::pkspec1: The expression x cannot be used as a part specification."
      )),
      "expected Part::pkspec1 message, got {:?}",
      msgs
    );
  }

  #[test]
  fn part_key_wrapper_matches_any_key() {
    assert_eq!(interpret("<|x -> 1|>[[Key[x]]]").unwrap(), "1");
    assert_eq!(interpret(r#"<|"a" -> 1|>[[Key["a"]]]"#).unwrap(), "1");
  }

  #[test]
  fn part_key_wrapper_absent_returns_missing() {
    // The Missing token carries the whole Key[...] wrapper
    assert_eq!(
      interpret(r#"<|"b" -> 1|>[[Key["a"]]]"#).unwrap(),
      "Missing[KeyAbsent, Key[a]]"
    );
  }

  #[test]
  fn part_span_returns_sub_association() {
    assert_eq!(
      interpret(r#"<|"a" -> 1, "b" -> 2, "c" -> 3|>[[1 ;; 2]]"#).unwrap(),
      "<|a -> 1, b -> 2|>"
    );
    // Negative step reverses the entries
    assert_eq!(
      interpret(r#"<|"a" -> 1, "b" -> 2, "c" -> 3|>[[3 ;; 1 ;; -1]]"#).unwrap(),
      "<|c -> 3, b -> 2, a -> 1|>"
    );
  }

  #[test]
  fn part_list_index_returns_sub_association() {
    assert_eq!(
      interpret(r#"<|"a" -> 1, "b" -> 2, "c" -> 3|>[[{1, 3}]]"#).unwrap(),
      "<|a -> 1, c -> 3|>"
    );
    assert_eq!(
      interpret(r#"<|"a" -> 1, "b" -> 2, "c" -> 3|>[[{"a", "c"}]]"#).unwrap(),
      "<|a -> 1, c -> 3|>"
    );
  }

  #[test]
  fn part_list_index_absent_key_keeps_missing_entry() {
    assert_eq!(
      interpret(r#"<|"a" -> 1, "b" -> 2|>[[{"a", "zz"}]]"#).unwrap(),
      "<|a -> 1, zz -> Missing[KeyAbsent, zz]|>"
    );
  }

  #[test]
  fn part_positional_out_of_bounds_emits_partw() {
    assert_eq!(
      interpret(r#"<|"a" -> 1|>[[5]]"#).unwrap(),
      "<|a -> 1|>[[5]]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(
        |m| m.contains("Part::partw: Part 5 of <|a -> 1|> does not exist.")
      ),
      "expected Part::partw message, got {:?}",
      msgs
    );
  }

  #[test]
  fn part_negative_index_counts_from_end() {
    assert_eq!(
      interpret(r#"<|"a" -> 1, "b" -> 2, "c" -> 3|>[[-1]]"#).unwrap(),
      "3"
    );
  }
}

mod invalid_subject_messages {
  use super::*;

  #[test]
  fn invrl_family_emits_message_and_returns_unevaluated() {
    // Regression: Keys/Values/Lookup/KeyTake/KeyDrop/KeySort raised
    // hard errors for invalid subjects
    for (input, func) in [
      ("Keys[x]", "Keys"),
      ("Values[x]", "Values"),
      ("KeySort[x]", "KeySort"),
    ] {
      assert_eq!(interpret(input).unwrap(), input);
      let msgs = woxi::get_captured_messages_raw();
      let expected = format!(
        "{}::invrl: The argument x is not a valid Association or a list of rules.",
        func
      );
      assert!(
        msgs.iter().any(|m| m.contains(&expected)),
        "expected {:?}, got {:?}",
        expected,
        msgs
      );
    }
    assert_eq!(interpret("Lookup[x, a]").unwrap(), "Lookup[x, a]");
    assert_eq!(interpret("KeyTake[x, {a}]").unwrap(), "KeyTake[x, {a}]");
    assert_eq!(interpret("KeyDrop[x, {a}]").unwrap(), "KeyDrop[x, {a}]");
    // Non-List heads are invalid subjects too
    assert_eq!(
      interpret("Keys[f[a -> 1, b -> 2]]").unwrap(),
      "Keys[f[a -> 1, b -> 2]]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "Keys::invrl: The argument f[a -> 1, b -> 2] is not a valid Association or a list of rules."
      )),
      "expected invrl message for general head, got {:?}",
      msgs
    );
  }

  #[test]
  fn function_specific_tags() {
    // Each function has its own message tag and wording
    assert_eq!(interpret("KeyValueMap[f, x]").unwrap(), "KeyValueMap[f, x]");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "KeyValueMap::invak: The argument x is not a valid Association."
      )),
      "expected invak message, got {:?}",
      msgs
    );
    assert_eq!(interpret("KeyMap[f, x]").unwrap(), "KeyMap[f, x]");
    assert_eq!(interpret("KeySelect[x, f]").unwrap(), "KeySelect[x, f]");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "KeySelect::invru: The argument x is not a valid Association or a list of rules."
      )),
      "expected invru message, got {:?}",
      msgs
    );
    assert_eq!(
      interpret("AssociationMap[f, x]").unwrap(),
      "AssociationMap[f, x]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "AssociationMap::invrp: The argument x is not a valid Association or a list."
      )),
      "expected invrp message, got {:?}",
      msgs
    );
    assert_eq!(interpret("Merge[x, Total]").unwrap(), "Merge[x, Total]");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "Merge::list1: The argument x is not a valid list of Associations or rules or lists of rules."
      )),
      "expected list1 message, got {:?}",
      msgs
    );
    assert_eq!(interpret("KeyUnion[x]").unwrap(), "KeyUnion[x]");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "KeyUnion::invar: The argument x is not a valid list of Associations or rules."
      )),
      "expected invar message, got {:?}",
      msgs
    );
  }

  #[test]
  fn key_exists_q_answers_false_after_message() {
    // Regression: hard error; wolframscript answers False
    assert_eq!(interpret("KeyExistsQ[x, a]").unwrap(), "False");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "KeyExistsQ::invrl: The argument x is not a valid Association or a list of rules."
      )),
      "expected invrl message, got {:?}",
      msgs
    );
    // Lists of rules are valid subjects (regression: hard error)
    assert_eq!(
      interpret("KeyExistsQ[{a -> 1, b -> 2}, a]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("KeyExistsQ[{a -> 1, b -> 2}, c]").unwrap(),
      "False"
    );
  }

  #[test]
  fn key_drop_from_validates_its_variable() {
    // Undefined symbol → blnoval; non-symbol → rvalue
    assert_eq!(interpret("KeyDropFrom[u, a]").unwrap(), "KeyDropFrom[u, a]");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "KeyDropFrom::blnoval: The symbol u at position 1 should have an immediate value defined."
      )),
      "expected blnoval message, got {:?}",
      msgs
    );
    assert_eq!(interpret("KeyDropFrom[5, a]").unwrap(), "KeyDropFrom[5, a]");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "KeyDropFrom::rvalue: 5 is not a variable with a value, so its value cannot be changed."
      )),
      "expected rvalue message, got {:?}",
      msgs
    );
    // The mutating happy path still works
    assert_eq!(
      interpret("m = <|a -> 1, b -> 2|>; KeyDropFrom[m, a]; m").unwrap(),
      "<|b -> 2|>"
    );
  }
}
