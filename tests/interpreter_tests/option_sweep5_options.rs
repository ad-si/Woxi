// Options whose names appeared nowhere in Woxi's source, found by the
// reference-page parameter sweep (round 5). Expected values are the ones
// the Wolfram reference pages show, or follow from the definitions.

use super::*;

/// `interpret(code)` must evaluate to `True`.
fn assert_true(code: &str) {
  assert_eq!(interpret(code).unwrap(), "True", "{code}");
}

mod url_query_options {
  use super::*;

  #[test]
  fn url_query_decode_sorts_queries() {
    assert_eq!(
      interpret(r#"URLQueryDecode["z=1&a=2&c=4", "SortQueries" -> True]"#)
        .unwrap(),
      "{a -> 2, c -> 4, z -> 1}"
    );
    assert_eq!(
      interpret(r#"URLQueryDecode["z=1&a=2&c=4"]"#).unwrap(),
      "{z -> 1, a -> 2, c -> 4}"
    );
  }

  #[test]
  fn url_query_decode_removes_empty_queries() {
    assert_true(
      r#"URLQueryDecode["z=1&a&c=4&z=", "RemoveEmptyQueries" -> True] === {"z" -> "1", "c" -> "4"}"#,
    );
    assert_true(
      r#"URLQueryDecode["z=1&a&c=4&z="] === {"z" -> "1", "a" -> "", "c" -> "4", "z" -> ""}"#,
    );
  }

  #[test]
  fn url_query_decode_rejects_unknown_options() {
    assert_eq!(
      interpret(r#"URLQueryDecode["a=1", Foo -> 2]"#).unwrap(),
      "URLQueryDecode[a=1, Foo -> 2]"
    );
  }

  #[test]
  fn url_build_merges_existing_query() {
    assert_eq!(
      interpret(
        r#"URLBuild["https://example.com?x=2", {"c" -> 4, "z" -> 1, "a" -> 2}]"#
      )
      .unwrap(),
      "https://example.com/?x=2&c=4&z=1&a=2"
    );
    assert_eq!(
      interpret(
        r#"URLBuild["https://example.com/?a=2", <|"n" -> {1, 2, 3}|>]"#
      )
      .unwrap(),
      "https://example.com/?a=2&n=1&n=2&n=3"
    );
    assert_eq!(
      interpret(
        r#"URLBuild[URL["https://www.example.com/?i=1"], <|"x" -> 2|>]"#
      )
      .unwrap(),
      "https://www.example.com/?i=1&x=2"
    );
  }

  #[test]
  fn url_build_sorts_queries() {
    assert_eq!(
      interpret(
        r#"URLBuild["https://example.com?x=2", {"c" -> 4, "z" -> 1, "a" -> 2}, "SortQueries" -> True]"#
      )
      .unwrap(),
      "https://example.com/?a=2&c=4&x=2&z=1"
    );
  }

  #[test]
  fn url_build_removes_empty_queries() {
    assert_eq!(
      interpret(
        r#"URLBuild["https://example.com?x=2", {"y" -> None, "z" -> 3}, "RemoveEmptyQueries" -> True]"#
      )
      .unwrap(),
      "https://example.com/?x=2&z=3"
    );
    assert_eq!(
      interpret(
        r#"URLBuild["https://example.com?x=2", {"y" -> None, "z" -> 3}]"#
      )
      .unwrap(),
      "https://example.com/?x=2&y&z=3"
    );
  }

  #[test]
  fn url_build_adds_root_path_before_query() {
    assert_eq!(
      interpret(r#"URLBuild["https://example.com", <|"n" -> {1, 2, 3}|>]"#)
        .unwrap(),
      "https://example.com/?n=1&n=2&n=3"
    );
    assert_eq!(
      interpret(
        r#"URLBuild[{"https://www.google.com", "search"}, {"q" -> "Mathematica"}]"#
      )
      .unwrap(),
      "https://www.google.com/search?q=Mathematica"
    );
    assert_eq!(
      interpret(
        r#"URLBuild["https://example.com", {"value" -> 1, "value" -> 2}]"#
      )
      .unwrap(),
      "https://example.com/?value=1&value=2"
    );
  }

  #[test]
  fn url_build_association_joins_path_after_authority() {
    assert_eq!(
      interpret(
        r#"URLBuild[<|"Scheme" -> "https", "User" -> "user:info", "Domain" -> "www.domain.com", "Port" -> "80", "Path" -> {"path", "to", "resource"}, "Query" -> {"param1" -> "val2", "param2" -> "val2"}, "Fragment" -> "fragment"|>]"#
      )
      .unwrap(),
      "https://user:info@www.domain.com:80/path/to/resource?param1=val2&param2=val2#fragment"
    );
    assert_eq!(
      interpret(r#"URLBuild[<|"Path" -> "path/to/dir", "Query" -> "x=3"|>]"#)
        .unwrap(),
      "path/to/dir?x=3"
    );
  }

  #[test]
  fn url_query_encode_missing_values_and_lists() {
    assert_eq!(
      interpret(
        r#"URLQueryEncode[<|"user" -> Missing, "number" -> 3, "other" -> None|>]"#
      )
      .unwrap(),
      "user&number=3&other"
    );
    assert_eq!(
      interpret(
        r#"URLQueryEncode[<|"number" -> {1, 2, 3}, "name" -> {"jack", "john", "bob"}|>]"#
      )
      .unwrap(),
      "number=1&number=2&number=3&name=jack&name=john&name=bob"
    );
    assert_eq!(
      interpret(
        r#"URLQueryEncode[{"a" -> 2, "c" -> 4, "z" -> None}, "RemoveEmptyQueries" -> True]"#
      )
      .unwrap(),
      "a=2&c=4"
    );
    assert_eq!(
      interpret(r#"URLQueryEncode[{"url" -> URL["http://www.wolfram.com"]}]"#)
        .unwrap(),
      "url=http%3A%2F%2Fwww.wolfram.com"
    );
  }
}

mod join_across {
  use super::*;

  const AUTHORS: &str = r#"{<|"Author" -> "Mark Twain", "YearOfBirth" -> 1835|>, <|"Author" -> "Jules Verne", "YearOfBirth" -> 1828|>}"#;
  const BOOKS: &str = r#"{<|"Author" -> "Mark Twain", "Title" -> "Tom Sawyer"|>, <|"Author" -> "Mark Twain", "Title" -> "Huckleberry Finn"|>, <|"Author" -> "Emilio Salgari", "Title" -> "Sandokan"|>}"#;

  #[test]
  fn inner_join_is_the_default() {
    assert_true(&format!(
      r#"JoinAcross[{AUTHORS}, {BOOKS}, "Author"] === {{<|"Author" -> "Mark Twain", "YearOfBirth" -> 1835, "Title" -> "Tom Sawyer"|>, <|"Author" -> "Mark Twain", "YearOfBirth" -> 1835, "Title" -> "Huckleberry Finn"|>}}"#
    ));
    assert_true(&format!(
      r#"JoinAcross[{AUTHORS}, {BOOKS}, "Author", "Inner"] === JoinAcross[{AUTHORS}, {BOOKS}, "Author"]"#
    ));
  }

  #[test]
  fn left_join_pads_unmatched_rows() {
    assert_true(&format!(
      r#"JoinAcross[{AUTHORS}, {BOOKS}, "Author", "Left"] === {{<|"Author" -> "Mark Twain", "YearOfBirth" -> 1835, "Title" -> "Tom Sawyer"|>, <|"Author" -> "Mark Twain", "YearOfBirth" -> 1835, "Title" -> "Huckleberry Finn"|>, <|"Author" -> "Jules Verne", "YearOfBirth" -> 1828, "Title" -> Missing["Unmatched"]|>}}"#
    ));
  }

  #[test]
  fn colliding_keys_take_the_left_value_by_default() {
    assert_eq!(
      interpret(
        "JoinAcross[{<|a -> 1, b -> X|>}, {<|a -> 1, b -> Y|>}, Key[a], \"Inner\"]"
      )
      .unwrap(),
      "{<|a -> 1, b -> X|>}"
    );
  }

  #[test]
  fn key_collision_function_right() {
    assert_eq!(
      interpret(
        "JoinAcross[{<|a -> 1, b -> X|>}, {<|a -> 1, b -> Y|>}, Key[a], \"Inner\", KeyCollisionFunction -> Right]"
      )
      .unwrap(),
      "{<|a -> 1, b -> Y|>}"
    );
  }

  #[test]
  fn key_collision_function_generates_new_keys() {
    assert_eq!(
      interpret(
        "JoinAcross[{<|a -> 1, b -> X|>}, {<|a -> 1, b -> Y|>}, Key[a], \"Inner\", KeyCollisionFunction -> Function[x, {left[x], right[x]}]]"
      )
      .unwrap(),
      "{<|a -> 1, left[b] -> X, right[b] -> Y|>}"
    );
  }

  #[test]
  fn key_collision_function_none_drops_both() {
    assert_eq!(
      interpret(
        "JoinAcross[{<|a -> 1, b -> X|>}, {<|a -> 1, b -> Y|>}, Key[a], \"Inner\", KeyCollisionFunction -> None]"
      )
      .unwrap(),
      "{<|a -> 1|>}"
    );
  }

  #[test]
  fn keys_joined_across_different_names_still_collide() {
    let base = "JoinAcross[{<|a -> 1, b -> 2|>}, {<|a -> 2, b -> 1|>}, {Key[a] -> Key[b], Key[b] -> Key[a]}, \"Left\"";
    assert_eq!(
      interpret(&format!(
        "{base}, KeyCollisionFunction -> Function[x, {{left[x], right[x]}}]]"
      ))
      .unwrap(),
      "{<|left[a] -> 1, right[a] -> 2, left[b] -> 2, right[b] -> 1|>}"
    );
    assert_eq!(
      interpret(&format!("{base}]")).unwrap(),
      "{<|a -> 1, b -> 2|>}"
    );
    assert_eq!(
      interpret(&format!("{base}, KeyCollisionFunction -> Left]")).unwrap(),
      "{<|a -> 1, b -> 2|>}"
    );
    assert_eq!(
      interpret(&format!("{base}, KeyCollisionFunction -> Right]")).unwrap(),
      "{<|a -> 2, b -> 1|>}"
    );
  }

  #[test]
  fn join_across_a_key_rule() {
    assert_eq!(
      interpret(
        r#"JoinAcross[{<|"a" -> 1, "b" -> 2|>}, {<|"c" -> 1, "d" -> 3|>}, "a" -> "c"]"#
      )
      .unwrap(),
      "{<|a -> 1, b -> 2, c -> 1, d -> 3|>}"
    );
  }

  #[test]
  fn unknown_option_stays_unevaluated() {
    assert_eq!(
      interpret(r#"JoinAcross[{<|"a" -> 1|>}, {<|"a" -> 1|>}, "a", Foo -> 1]"#)
        .unwrap(),
      "JoinAcross[{<|a -> 1|>}, {<|a -> 1|>}, a, Foo -> 1]"
    );
  }
}

mod text_string_formats {
  use super::*;

  #[test]
  fn boolean_strings() {
    assert_eq!(
      interpret("TextString[{True, False, True, True}]").unwrap(),
      "{True, False, True, True}"
    );
    assert_eq!(
      interpret(
        r#"TextString[{True, False, True, True}, BooleanStrings -> {"+", "-"}]"#
      )
      .unwrap(),
      "{+, -, +, +}"
    );
    assert_eq!(
      interpret(
        r#"TextString[{True, False, True, True}, BooleanStrings -> {"1", "0"}]"#
      )
      .unwrap(),
      "{1, 0, 1, 1}"
    );
  }

  #[test]
  fn association_format() {
    assert_eq!(
      interpret(
        r#"TextString[<|"first" -> 1, "second" -> 2, "third" -> 3|>, AssociationFormat -> {"[ ", "; ", " ]", " = "}]"#
      )
      .unwrap(),
      "[ first = 1; second = 2; third = 3 ]"
    );
    assert_eq!(
      interpret(
        r#"TextString[<|a -> 1, b -> 2|>, AssociationFormat -> {"{", ", ", "}", ":"}]"#
      )
      .unwrap(),
      "{a:1, b:2}"
    );
    assert_eq!(
      interpret(
        r#"TextString[<|"first" -> 1, "second" -> 2|>, AssociationFormat -> Automatic]"#
      )
      .unwrap(),
      "<|first -> 1, second -> 2|>"
    );
  }

  #[test]
  fn list_format() {
    assert_eq!(
      interpret(r#"TextString[Range[10], ListFormat -> {"[ ", "; ", " ]"}]"#)
        .unwrap(),
      "[ 1; 2; 3; 4; 5; 6; 7; 8; 9; 10 ]"
    );
    assert_eq!(
      interpret(r#"TextString[{1, 2, 3, 4}, ListFormat -> {"[", ", ", "]"}]"#)
        .unwrap(),
      "[1, 2, 3, 4]"
    );
    assert_eq!(
      interpret("TextString[Range[10], ListFormat -> Automatic]").unwrap(),
      "{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}"
    );
  }

  #[test]
  fn elided_forms() {
    assert_eq!(
      interpret("TextString[{f[10], g[20], h[30]}, ElidedForms -> {f, h}]")
        .unwrap(),
      "{f[\u{2026}], g[20], h[\u{2026}]}"
    );
    assert_eq!(
      interpret("TextString[{f[x], g[x]}, ElidedForms -> f]").unwrap(),
      "{f[\u{2026}], g[x]}"
    );
    assert_eq!(
      interpret("TextString[{f[x], g[x]}, ElidedForms -> All]").unwrap(),
      "{f[\u{2026}], g[\u{2026}]}"
    );
    assert_eq!(
      interpret("TextString[{f[10], g[20], h[30]}]").unwrap(),
      "{f[10], g[20], h[30]}"
    );
  }

  #[test]
  fn graphics_are_elided_by_default() {
    assert_eq!(
      interpret("TextString[Graphics[Disk[]]]").unwrap(),
      "Graphics[\u{2026}]"
    );
  }

  #[test]
  fn missing_string() {
    assert_eq!(
      interpret("TextString[{1, 2, Missing[]}]").unwrap(),
      "{1, 2, }"
    );
    assert_eq!(
      interpret(r#"TextString[{1, 2, Missing[]}, MissingString -> "?"]"#)
        .unwrap(),
      "{1, 2, ?}"
    );
    assert_eq!(
      interpret(
        r#"TextString[{1, 2, Missing["NotAvailable"]}, MissingString -> Automatic]"#
      )
      .unwrap(),
      "{1, 2, NotAvailable}"
    );
  }

  #[test]
  fn unknown_option_stays_unevaluated() {
    assert_eq!(
      interpret("TextString[{1, 2}, Foo -> 1]").unwrap(),
      "TextString[{1, 2}, Foo -> 1]"
    );
  }
}

mod read_separators {
  use super::*;

  #[test]
  fn null_records() {
    assert_true(
      r#"ReadList[StringToStream["a\nb\n\nc\n\n\nd"], Record] === {"a", "b", "c", "d"}"#,
    );
    assert_true(
      r#"ReadList[StringToStream["a\nb\n\nc\n\n\nd"], Record, NullRecords -> True] === {"a", "b", "", "c", "", "", "d"}"#,
    );
  }

  #[test]
  fn word_separators_and_null_words() {
    assert_true(
      r#"ReadList[StringToStream["first:second::fourth:::seventh"], Word, WordSeparators -> {":"}] === {"first", "second", "fourth", "seventh"}"#,
    );
    assert_true(
      r#"ReadList[StringToStream["first:second::fourth:::seventh"], Word, WordSeparators -> {":"}, NullWords -> True] === {"first", "second", "", "fourth", "", "", "seventh"}"#,
    );
  }

  #[test]
  fn no_word_separators_reads_records() {
    assert_true(
      r#"ReadList[StringToStream["cat dog bird fish goat"], Word] === {"cat", "dog", "bird", "fish", "goat"}"#,
    );
    assert_true(
      r#"ReadList[StringToStream["cat dog bird fish goat"], Word, WordSeparators -> None] === {"cat dog bird fish goat"}"#,
    );
    assert_true(
      r#"ReadList[StringToStream["cat dog bird fish goat"], Record] === {"cat dog bird fish goat"}"#,
    );
  }

  #[test]
  fn record_lists() {
    assert_eq!(
      interpret(
        r#"ReadList[StringToStream["11.1   22.2    33.3\n44.4   55.5    66.6"], Number]"#
      )
      .unwrap(),
      "{11.1, 22.2, 33.3, 44.4, 55.5, 66.6}"
    );
    assert_eq!(
      interpret(
        r#"ReadList[StringToStream["11.1   22.2    33.3\n44.4   55.5    66.6"], Number, RecordLists -> True]"#
      )
      .unwrap(),
      "{{11.1, 22.2, 33.3}, {44.4, 55.5, 66.6}}"
    );
  }

  #[test]
  fn token_words() {
    assert_true(
      r#"ReadList[StringToStream["22*a*b+56*c+13*a*d"], Word] === {"22*a*b+56*c+13*a*d"}"#,
    );
    assert_true(
      r#"ReadList[StringToStream["22*a*b+56*c+13*a*d"], Word, TokenWords -> {"+", "*"}] === {"22", "*", "a", "*", "b", "+", "56", "*", "c", "+", "13", "*", "a", "*", "d"}"#,
    );
  }

  #[test]
  fn read_takes_the_same_options() {
    assert_true(
      r#"s = StringToStream["a:b:c"]; {Read[s, Word, WordSeparators -> {":"}], Read[s, Word, WordSeparators -> {":"}]} === {"a", "b"}"#,
    );
    assert_true(
      r#"s = StringToStream["x\ny"]; {Read[s, Record], Read[s, Record], Read[s, Record]} === {"x", "y", EndOfFile}"#,
    );
  }

  #[test]
  fn unknown_option_stays_unevaluated() {
    assert_eq!(
      interpret(r#"ReadList[StringToStream["a b"], Word, Foo -> 1]"#).unwrap(),
      "ReadList[InputStream[String, 3], Word, Foo -> 1]"
    );
  }
}

mod find_options {
  use super::*;

  #[test]
  fn anchored_search() {
    assert_true(
      r#"FindList[StringToStream["abcdef\ndefghi"], "def", AnchoredSearch -> True] === {"defghi"}"#,
    );
    assert_true(
      r#"FindList[StringToStream["abcdef\ndefghi"], "def", AnchoredSearch -> False] === {"abcdef", "defghi"}"#,
    );
    assert_true(
      r#"Find[StringToStream["abcdef\ndefghi"], "def", AnchoredSearch -> True] === "defghi""#,
    );
  }

  #[test]
  fn word_search() {
    assert_true(
      r#"FindList[StringToStream["abc def\ndefghi"], "def", WordSearch -> True] === {"abc def"}"#,
    );
    assert_true(
      r#"FindList[StringToStream["abc:def\ndefghi"], "def", WordSearch -> True, WordSeparators -> {":"}] === {"abc:def"}"#,
    );
  }

  #[test]
  fn ignore_case() {
    assert_true(
      r#"FindList[StringToStream["abc DEF\ndefghi"], "def", IgnoreCase -> True] === {"abc DEF", "defghi"}"#,
    );
  }

  #[test]
  fn unknown_option_stays_unevaluated() {
    assert_eq!(
      interpret(r#"FindList[StringToStream["abc"], "a", Foo -> 1]"#).unwrap(),
      "FindList[InputStream[String, 3], a, Foo -> 1]"
    );
  }
}

mod byte_ordering {
  use super::*;

  fn write_then(name: &str, write: &str, read: &str) -> String {
    let path = temp_file(name);
    interpret(&format!(
      "f = \"{path}\"; s = OpenWrite[f, BinaryFormat -> True]; {write}; Close[s]; {read}"
    ))
    .unwrap()
  }

  #[test]
  fn reverse_byte_ordering_reads_the_bytes_backwards() {
    assert_eq!(
      write_then(
        "woxi_sweep5_bo1.bin",
        "BinaryWrite[s, 12345, \"Integer32\"]",
        "r = OpenRead[f, BinaryFormat -> True]; a = BinaryRead[r, \"Integer32\", ByteOrdering -> $ByteOrdering]; Close[r]; r = OpenRead[f, BinaryFormat -> True]; b = BinaryRead[r, \"Integer32\", ByteOrdering -> -$ByteOrdering]; Close[r]; {a, b, FromDigits[Reverse[IntegerDigits[b, 256]], 256]}",
      ),
      "{12345, 959447040, 12345}"
    );
  }

  #[test]
  fn big_endian_write_and_read() {
    assert_eq!(
      write_then(
        "woxi_sweep5_bo2.bin",
        "BinaryWrite[s, {258, -2}, \"Integer16\", ByteOrdering -> 1]",
        "{BinaryReadList[f], BinaryReadList[f, \"Integer16\", ByteOrdering -> 1], BinaryReadList[f, \"UnsignedInteger16\", ByteOrdering -> 1], BinaryReadList[f, \"Integer16\"]}",
      ),
      "{{1, 2, 255, 254}, {258, -2}, {258, 65534}, {513, -257}}"
    );
  }

  #[test]
  fn reals() {
    assert_eq!(
      write_then(
        "woxi_sweep5_bo3.bin",
        "BinaryWrite[s, {1.5, -0.25}, \"Real64\"]",
        "{BinaryReadList[f, \"Real64\"], BinaryReadList[f, \"Real64\", ByteOrdering -> -1]}",
      ),
      "{{1.5, -0.25}, {1.5, -0.25}}"
    );
  }

  #[test]
  fn unknown_option_stays_unevaluated() {
    let path = temp_file("woxi_sweep5_bo4.bin");
    assert_eq!(
      interpret(&format!("BinaryReadList[\"{path}\", \"Byte\", Foo -> 1]"))
        .unwrap(),
      format!("BinaryReadList[{path}, Byte, Foo -> 1]")
    );
  }
}

mod overwrite_target {
  use super::*;

  #[test]
  fn copy_file() {
    let a = temp_file("woxi_sweep5_ow_a.txt");
    let b = temp_file("woxi_sweep5_ow_b.txt");
    assert_eq!(
      interpret(&format!(
        "a = \"{a}\"; b = \"{b}\"; Quiet[DeleteFile[{{a, b}}]]; Export[a, \"x\", \"Text\"]; Export[b, \"y\", \"Text\"]; {{Quiet[CopyFile[a, b]], Quiet[CopyFile[a, b, OverwriteTarget -> False]], Import[b, \"Text\"], CopyFile[a, b, OverwriteTarget -> True] === b, Import[b, \"Text\"]}}"
      ))
      .unwrap(),
      "{$Failed, $Failed, y, True, x}"
    );
  }

  #[test]
  fn rename_file() {
    let a = temp_file("woxi_sweep5_rn_a.txt");
    let b = temp_file("woxi_sweep5_rn_b.txt");
    assert_eq!(
      interpret(&format!(
        "a = \"{a}\"; b = \"{b}\"; Export[a, \"x\", \"Text\"]; Export[b, \"y\", \"Text\"]; {{Quiet[RenameFile[a, b]], Import[b, \"Text\"], RenameFile[a, b, OverwriteTarget -> True] === b, Import[b, \"Text\"], FileExistsQ[a]}}"
      ))
      .unwrap(),
      "{$Failed, y, True, x, False}"
    );
  }

  #[test]
  fn unknown_option_stays_unevaluated() {
    assert_eq!(
      interpret("CopyFile[\"a\", \"b\", Foo -> 1]").unwrap(),
      "CopyFile[a, b, Foo -> 1]"
    );
  }
}

mod maximum_flow_capacities {
  use super::*;

  #[test]
  fn edge_capacity_of_the_graph() {
    assert_eq!(
      interpret(
        "FindMaximumFlow[Graph[{1 -> 2, 2 -> 3, 1 -> 3}, EdgeCapacity -> {2, 3, 4}], 1, 3]"
      )
      .unwrap(),
      "6"
    );
  }

  #[test]
  fn edge_capacity_option() {
    assert_eq!(
      interpret(
        "FindMaximumFlow[Graph[{1 -> 2, 2 -> 3, 1 -> 3}], 1, 3, EdgeCapacity -> {2, 3, 4}]"
      )
      .unwrap(),
      "6"
    );
    assert_eq!(
      interpret(
        "FindMaximumFlow[Graph[{1 -> 2, 2 -> 3, 1 -> 3}], 1, 3, EdgeCapacity -> {0.5, 1, 0.25}]"
      )
      .unwrap(),
      "0.75"
    );
  }

  #[test]
  fn unit_capacities_by_default() {
    assert_eq!(
      interpret("FindMaximumFlow[Graph[{1 -> 2, 2 -> 3, 1 -> 3}], 1, 3]")
        .unwrap(),
      "2"
    );
    assert_eq!(
      interpret("FindMaximumFlow[{1 -> 2, 2 -> 3, 3 -> 1, 3 -> 4}, 1, 3]")
        .unwrap(),
      "1"
    );
  }

  #[test]
  fn undirected_edges_carry_flow_both_ways() {
    assert_eq!(
      interpret(
        "FindMaximumFlow[Graph[{1 <-> 2, 2 <-> 3, 3 <-> 1}, EdgeCapacity -> {2, 3, 4}], 1, 3]"
      )
      .unwrap(),
      "6"
    );
  }

  #[test]
  fn vertex_capacity() {
    assert_eq!(
      interpret(
        "FindMaximumFlow[Graph[{1 -> 2, 2 -> 3, 1 -> 3}], 1, 3, EdgeCapacity -> {2, 3, 4}, VertexCapacity -> {10, 1, 10}]"
      )
      .unwrap(),
      "5"
    );
    assert_eq!(
      interpret(
        "FindMaximumFlow[Graph[{1 -> 2, 2 -> 3, 1 -> 3}, VertexCapacity -> {10, 1, 10}], 1, 3, EdgeCapacity -> {2, 3, 4}]"
      )
      .unwrap(),
      "5"
    );
  }

  #[test]
  fn unknown_option_stays_unevaluated() {
    assert_true(
      "Head[FindMaximumFlow[Graph[{1 -> 2}], 1, 2, Foo -> 1]] === FindMaximumFlow",
    );
  }
}

mod hypothesis_test_options {
  use super::*;

  const DATA: &str = "{1.2, -0.5, 0.3, 2.1, -1.4, 0.8, 0.1, -0.2, 1.7, -0.9}";

  #[test]
  fn alternative_hypothesis_gives_one_sided_p_values() {
    assert_true(&format!(
      "p = LocationTest[{DATA}, 0, \"PValue\"]; g = LocationTest[{DATA}, 0, \"PValue\", AlternativeHypothesis -> \"Greater\"]; l = LocationTest[{DATA}, 0, \"PValue\", AlternativeHypothesis -> \"Less\"]; Abs[g - p/2] < 10^-12 && Abs[l - (1 - p/2)] < 10^-12"
    ));
    assert_true(&format!(
      "LocationTest[{DATA}, 0, \"PValue\", AlternativeHypothesis -> \"Unequal\"] == LocationTest[{DATA}, 0, \"PValue\"]"
    ));
  }

  #[test]
  fn significance_level_is_the_rejection_cutoff() {
    // The p-value of this sample is about 0.394.
    assert_eq!(
      interpret(&format!(
        "{{LocationTest[{DATA}, 0, \"ShortTestConclusion\"], LocationTest[{DATA}, 0, \"ShortTestConclusion\", SignificanceLevel -> 0.5]}}"
      ))
      .unwrap(),
      "{Do not reject, Reject}"
    );
    // The p-value does not depend on the level.
    assert_true(&format!(
      "LocationTest[{DATA}, 0, \"PValue\", SignificanceLevel -> 0.5] == LocationTest[{DATA}, 0, \"PValue\"]"
    ));
  }

  #[test]
  fn distribution_fit_test_significance_level() {
    // The p-value against NormalDistribution[0, 1] is about 0.61.
    assert_eq!(
      interpret(&format!(
        "{{DistributionFitTest[{DATA}, NormalDistribution[0, 1], \"ShortTestConclusion\"], DistributionFitTest[{DATA}, NormalDistribution[0, 1], \"ShortTestConclusion\", SignificanceLevel -> 0.9], DistributionFitTest[{DATA}, NormalDistribution[3, 1], \"ShortTestConclusion\"]}}"
      ))
      .unwrap(),
      "{Do not reject, Reject, Reject}"
    );
  }

  #[test]
  fn pearson_chi_square_test_significance_level() {
    assert_eq!(
      interpret(&format!(
        "{{PearsonChiSquareTest[{DATA}, NormalDistribution[0, 1], \"ShortTestConclusion\"], PearsonChiSquareTest[{DATA}, NormalDistribution[0, 1], \"ShortTestConclusion\", SignificanceLevel -> 0.9]}}"
      ))
      .unwrap(),
      "{Do not reject, Reject}"
    );
    assert_true(&format!(
      "PearsonChiSquareTest[{DATA}, NormalDistribution[0, 1], \"PValue\", SignificanceLevel -> 0.9] == PearsonChiSquareTest[{DATA}, NormalDistribution[0, 1]]"
    ));
  }

  #[test]
  fn invalid_settings_stay_unevaluated() {
    assert_true(&format!(
      "Head[LocationTest[{DATA}, 0, \"PValue\", AlternativeHypothesis -> \"Bigger\"]] === LocationTest"
    ));
    assert_true(&format!(
      "Head[DistributionFitTest[{DATA}, NormalDistribution[0, 1], Foo -> 2]] === DistributionFitTest"
    ));
  }
}

mod numeric_sum_options {
  use super::*;

  #[test]
  fn nsum_terms() {
    assert_true(
      "Abs[NSum[1/((k - 20)^2 + 1), {k, 0, Infinity}, NSumTerms -> 30] - 3.10462] < 10^-5",
    );
  }

  #[test]
  fn nproduct_factors() {
    assert_true(
      "Abs[NProduct[1 + 1/(1 + (k - 20)^2), {k, 0, Infinity}, NProductFactors -> 30] - 12.9013] < 10^-4",
    );
  }

  #[test]
  fn iterator_steps() {
    assert_true(
      "Abs[NProduct[1 + 1/2^i, {i, 0, Infinity, 2}] - 2.71182] < 10^-5",
    );
    assert_true("Abs[NSum[1/2^i, {i, 0, Infinity, 2}] - 4/3] < 10^-10");
    assert_true("NSum[1/2^i, {i, 0, 10, 2}] == N[Sum[1/2^i, {i, 0, 10, 2}]]");
    assert_eq!(interpret("NProduct[i, {i, 1, 9, 2}]").unwrap(), "945.");
  }
}

mod gradient_option {
  use super::*;

  #[test]
  fn supplied_gradient_gives_the_same_optimum() {
    assert_eq!(
      interpret(
        "FindMinimum[(x - 1)^2 + (y - 2)^2, {x, y}, Gradient -> {2 (x - 1), 2 (y - 2)}]"
      )
      .unwrap(),
      "{0., {x -> 1., y -> 2.}}"
    );
    assert_eq!(
      interpret("FindMaximum[-(x - 1)^2, {x, 0}, Gradient :> {-2 (x - 1)}]")
        .unwrap(),
      "{0., {x -> 1.}}"
    );
    assert_eq!(
      interpret("FindArgMin[(x - 3)^2, {x, 0}, Gradient -> {2 (x - 3)}]")
        .unwrap(),
      "{3.}"
    );
    assert_eq!(
      interpret("FindMinValue[(x - 3)^2 + 1, {x, 0}, Gradient -> {2 (x - 3)}]")
        .unwrap(),
      "1."
    );
  }
}
