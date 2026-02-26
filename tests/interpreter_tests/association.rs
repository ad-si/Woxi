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
    assert_eq!(interpret("x = 5; Clear[x]").unwrap(), "Null");
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
    assert_eq!(res.result, "Null");
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
