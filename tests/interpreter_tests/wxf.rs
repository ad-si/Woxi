use super::*;

// BinarySerialize / BinaryDeserialize — the WXF binary format. Every byte
// sequence below is verified against wolframscript.

mod binary_serialize {
  use super::*;

  #[test]
  fn returns_byte_array() {
    assert_eq!(
      interpret("BinarySerialize[{1, 2, 3}]").unwrap(),
      "ByteArray[<16>]"
    );
  }

  #[test]
  fn integers_choose_the_smallest_width() {
    // int8
    assert_eq!(
      interpret("Normal[BinarySerialize[42]]").unwrap(),
      "{56, 58, 67, 42}"
    );
    assert_eq!(
      interpret("Normal[BinarySerialize[-1]]").unwrap(),
      "{56, 58, 67, 255}"
    );
    // int16
    assert_eq!(
      interpret("Normal[BinarySerialize[200]]").unwrap(),
      "{56, 58, 106, 200, 0}"
    );
    assert_eq!(
      interpret("Normal[BinarySerialize[-129]]").unwrap(),
      "{56, 58, 106, 127, 255}"
    );
    assert_eq!(
      interpret("Normal[BinarySerialize[1000]]").unwrap(),
      "{56, 58, 106, 232, 3}"
    );
    // int32
    assert_eq!(
      interpret("Normal[BinarySerialize[100000]]").unwrap(),
      "{56, 58, 105, 160, 134, 1, 0}"
    );
    assert_eq!(
      interpret("Normal[BinarySerialize[32768]]").unwrap(),
      "{56, 58, 105, 0, 128, 0, 0}"
    );
    // int64
    assert_eq!(
      interpret("Normal[BinarySerialize[10^12]]").unwrap(),
      "{56, 58, 76, 0, 16, 165, 212, 232, 0, 0, 0}"
    );
    assert_eq!(
      interpret("Normal[BinarySerialize[2^31]]").unwrap(),
      "{56, 58, 76, 0, 0, 0, 128, 0, 0, 0, 0}"
    );
    assert_eq!(
      interpret("Normal[BinarySerialize[2^63 - 1]]").unwrap(),
      "{56, 58, 76, 255, 255, 255, 255, 255, 255, 255, 127}"
    );
  }

  #[test]
  fn big_integers_serialize_as_decimal_strings() {
    assert_eq!(
      interpret("Normal[BinarySerialize[10^30]]").unwrap(),
      "{56, 58, 73, 31, 49, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48}"
    );
    assert_eq!(
      interpret("Normal[BinarySerialize[-10^30]]").unwrap(),
      "{56, 58, 73, 32, 45, 49, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48}"
    );
  }

  #[test]
  fn reals_and_strings() {
    assert_eq!(
      interpret("Normal[BinarySerialize[3.14]]").unwrap(),
      "{56, 58, 114, 31, 133, 235, 81, 184, 30, 9, 64}"
    );
    assert_eq!(
      interpret("Normal[BinarySerialize[-2.5]]").unwrap(),
      "{56, 58, 114, 0, 0, 0, 0, 0, 0, 4, 192}"
    );
    assert_eq!(
      interpret("Normal[BinarySerialize[\"hello\"]]").unwrap(),
      "{56, 58, 83, 5, 104, 101, 108, 108, 111}"
    );
  }

  #[test]
  fn symbols_get_contexts() {
    // User symbols carry the Global` context.
    assert_eq!(
      interpret("Normal[BinarySerialize[x]]").unwrap(),
      "{56, 58, 115, 8, 71, 108, 111, 98, 97, 108, 96, 120}"
    );
    // System symbols are written bare.
    assert_eq!(
      interpret("Normal[BinarySerialize[Pi]]").unwrap(),
      "{56, 58, 115, 2, 80, 105}"
    );
    assert_eq!(
      interpret("Normal[BinarySerialize[True]]").unwrap(),
      "{56, 58, 115, 4, 84, 114, 117, 101}"
    );
  }

  #[test]
  fn lists_and_functions() {
    assert_eq!(
      interpret("Normal[BinarySerialize[{1, 2, 3}]]").unwrap(),
      "{56, 58, 102, 3, 115, 4, 76, 105, 115, 116, 67, 1, 67, 2, 67, 3}"
    );
    assert_eq!(
      interpret("Normal[BinarySerialize[{}]]").unwrap(),
      "{56, 58, 102, 0, 115, 4, 76, 105, 115, 116}"
    );
    assert_eq!(
      interpret("Normal[BinarySerialize[f[1, \"a\"]]]").unwrap(),
      "{56, 58, 102, 2, 115, 8, 71, 108, 111, 98, 97, 108, 96, 102, 67, 1, 83, 1, 97}"
    );
    assert_eq!(
      interpret("Normal[BinarySerialize[{1.5, 2.5}]]").unwrap(),
      "{56, 58, 102, 2, 115, 4, 76, 105, 115, 116, 114, 0, 0, 0, 0, 0, 0, 248, 63, 114, 0, 0, 0, 0, 0, 0, 4, 64}"
    );
  }

  #[test]
  fn fullform_heads_for_operator_expressions() {
    // 1/3 == Rational[1, 3]
    assert_eq!(
      interpret("Normal[BinarySerialize[1/3]]").unwrap(),
      "{56, 58, 102, 2, 115, 8, 82, 97, 116, 105, 111, 110, 97, 108, 67, 1, 67, 3}"
    );
    assert_eq!(
      interpret("Normal[BinarySerialize[-1/3]]").unwrap(),
      "{56, 58, 102, 2, 115, 8, 82, 97, 116, 105, 111, 110, 97, 108, 67, 255, 67, 3}"
    );
    // x -> y == Rule[x, y]
    assert_eq!(
      interpret("Normal[BinarySerialize[x -> y]]").unwrap(),
      "{56, 58, 102, 2, 115, 4, 82, 117, 108, 101, 115, 8, 71, 108, 111, 98, 97, 108, 96, 120, 115, 8, 71, 108, 111, 98, 97, 108, 96, 121}"
    );
    // x/y == Times[x, Power[y, -1]]
    assert_eq!(
      interpret("Normal[BinarySerialize[x/y]]").unwrap(),
      "{56, 58, 102, 2, 115, 5, 84, 105, 109, 101, 115, 115, 8, 71, 108, 111, 98, 97, 108, 96, 120, 102, 2, 115, 5, 80, 111, 119, 101, 114, 115, 8, 71, 108, 111, 98, 97, 108, 96, 121, 67, 255}"
    );
    // Sqrt[2] == Power[2, Rational[1, 2]]
    assert_eq!(
      interpret("Normal[BinarySerialize[Sqrt[2]]]").unwrap(),
      "{56, 58, 102, 2, 115, 5, 80, 111, 119, 101, 114, 67, 2, 102, 2, 115, 8, 82, 97, 116, 105, 111, 110, 97, 108, 67, 1, 67, 2}"
    );
  }

  #[test]
  fn associations_and_byte_arrays() {
    assert_eq!(
      interpret("Normal[BinarySerialize[<|\"a\" -> 1, \"b\" -> 2|>]]").unwrap(),
      "{56, 58, 65, 2, 45, 83, 1, 97, 67, 1, 45, 83, 1, 98, 67, 2}"
    );
    assert_eq!(
      interpret("Normal[BinarySerialize[ByteArray[{1, 2, 3}]]]").unwrap(),
      "{56, 58, 66, 3, 1, 2, 3}"
    );
  }

  #[test]
  fn numeric_complex_values() {
    // 2 + 3 I == Complex[2, 3]
    assert_eq!(
      interpret("Normal[BinarySerialize[2 + 3*I]]").unwrap(),
      "{56, 58, 102, 2, 115, 7, 67, 111, 109, 112, 108, 101, 120, 67, 2, 67, 3}"
    );
    assert_eq!(
      interpret("Normal[BinarySerialize[I]]").unwrap(),
      "{56, 58, 102, 2, 115, 7, 67, 111, 109, 112, 108, 101, 120, 67, 0, 67, 1}"
    );
    assert_eq!(
      interpret("Normal[BinarySerialize[-I]]").unwrap(),
      "{56, 58, 102, 2, 115, 7, 67, 111, 109, 112, 108, 101, 120, 67, 0, 67, 255}"
    );
    assert_eq!(
      interpret("Normal[BinarySerialize[1/2 + 3/4*I]]").unwrap(),
      "{56, 58, 102, 2, 115, 7, 67, 111, 109, 112, 108, 101, 120, 102, 2, 115, 8, 82, 97, 116, 105, 111, 110, 97, 108, 67, 1, 67, 2, 102, 2, 115, 8, 82, 97, 116, 105, 111, 110, 97, 108, 67, 3, 67, 4}"
    );
    // Machine precision is contagious: 2.5 I == Complex[0., 2.5].
    assert_eq!(
      interpret("Normal[BinarySerialize[2.5*I]]").unwrap(),
      "{56, 58, 102, 2, 115, 7, 67, 111, 109, 112, 108, 101, 120, 114, 0, 0, 0, 0, 0, 0, 0, 0, 114, 0, 0, 0, 0, 0, 0, 4, 64}"
    );
    // I as a Power base still serializes (Power[Complex[0, 1], x]).
    assert_eq!(
      interpret("Normal[BinarySerialize[I^x]]").unwrap(),
      "{56, 58, 102, 2, 115, 5, 80, 111, 119, 101, 114, 102, 2, 115, 7, 67, 111, 109, 112, 108, 101, 120, 67, 0, 67, 1, 115, 8, 71, 108, 111, 98, 97, 108, 96, 120}"
    );
  }

  #[test]
  fn symbolic_complex_sum_serializes_complex_atom_first() {
    // wolframscript's canonical Plus order places the numeric Complex atom
    // ahead of the symbolic term (Plus[Complex[0, 3], x]); Woxi keeps it last
    // internally but reorders it on serialization so the bytes match WS
    // exactly (verified against `Normal[BinarySerialize[x + 3*I]]`).
    assert_eq!(
      interpret("Normal[BinarySerialize[x + 3*I]]").unwrap(),
      "{56, 58, 102, 2, 115, 4, 80, 108, 117, 115, 102, 2, 115, 7, 67, 111, \
       109, 112, 108, 101, 120, 67, 0, 67, 3, 115, 8, 71, 108, 111, 98, 97, \
       108, 96, 120}"
    );
    // And it round-trips back to the original complex sum.
    assert_eq!(
      interpret("BinaryDeserialize[BinarySerialize[x + 3*I]]").unwrap(),
      "x + 3*I"
    );
  }

  #[test]
  fn symbolic_complex_products_stay_unevaluated() {
    // A product folds its numeric factor INTO the Complex atom in
    // wolframscript (Times[Complex[0, 3], x]) whereas Woxi keeps them
    // separate (Times[3, Complex[0, 1], x]), so it bails out instead of
    // producing wrong bytes.
    assert_eq!(
      interpret("BinarySerialize[3*I*x]").unwrap(),
      "BinarySerialize[(3*I)*x]"
    );
  }
}

mod binary_deserialize {
  use super::*;

  #[test]
  fn round_trips() {
    assert_eq!(
      interpret("BinaryDeserialize[BinarySerialize[{1, 2, 3}]]").unwrap(),
      "{1, 2, 3}"
    );
    assert_eq!(
      interpret("BinaryDeserialize[BinarySerialize[f[x, 2.5, \"s\", 1/3]]]")
        .unwrap(),
      "f[x, 2.5, s, 1/3]"
    );
    assert_eq!(
      interpret("BinaryDeserialize[BinarySerialize[<|\"a\" -> 1|>]]").unwrap(),
      "<|a -> 1|>"
    );
    assert_eq!(
      interpret("BinaryDeserialize[BinarySerialize[2 + 3*I]]").unwrap(),
      "2 + 3*I"
    );
    assert_eq!(
      interpret("BinaryDeserialize[BinarySerialize[ByteArray[{1, 2, 3}]]]")
        .unwrap(),
      "ByteArray[<3>]"
    );
    assert_eq!(
      interpret(
        "BinaryDeserialize[BinarySerialize[{\"a\" -> 1, 2.5, {1, 2}}]]"
      )
      .unwrap(),
      "{a -> 1, 2.5, {1, 2}}"
    );
  }

  #[test]
  fn reads_raw_bytes() {
    assert_eq!(
      interpret("BinaryDeserialize[ByteArray[{56, 58, 67, 42}]]").unwrap(),
      "42"
    );
  }

  #[test]
  fn reads_packed_arrays() {
    // wolframscript packs Range[3] as a rank-1 int8 packed array.
    assert_eq!(
      interpret(
        "BinaryDeserialize[ByteArray[{56, 58, 193, 0, 1, 3, 1, 2, 3}]]"
      )
      .unwrap(),
      "{1, 2, 3}"
    );
    // Rank-1 real64 packed array (N[Range[2]]).
    assert_eq!(
      interpret(
        "BinaryDeserialize[ByteArray[{56, 58, 193, 35, 1, 2, 0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 0, 64}]]"
      )
      .unwrap(),
      "{1., 2.}"
    );
  }

  #[test]
  fn corrupt_data_fails() {
    clear_state();
    let result =
      interpret_with_stdout("BinaryDeserialize[ByteArray[{1, 2, 3}]]").unwrap();
    assert_eq!(result.result, "$Failed");
    assert!(result.warnings[0].contains(
      "BinaryDeserialize::corrupt: Serialized data ByteArray[<3>] is \
       corrupt and does not represent an expression."
    ));
  }
}
