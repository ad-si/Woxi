// More argument forms found missing by the reference-page parameter sweep.

use super::*;

mod to_expression_boxes {
  use super::*;

  #[test]
  fn boxes_read_as_their_input() {
    for (boxes, expected) in [
      ("SqrtBox[\"x\"]", "Sqrt[x]"),
      ("SuperscriptBox[\"x\", \"3\"]", "x^3"),
      ("SubscriptBox[\"x\", \"0\"]", "Subscript[x, 0]"),
      ("OverscriptBox[\"x\", \"1\"]", "Overscript[x, 1]"),
      ("UnderscriptBox[\"x\", \"0\"]", "Underscript[x, 0]"),
      (
        "UnderoverscriptBox[\"x\", \"0\", \"1\"]",
        "Underoverscript[x, 0, 1]",
      ),
      ("FractionBox[\"1\", \"2\"]", "1/2"),
      ("RowBox[{\"1\", \"+\", \"x\"}]", "1 + x"),
    ] {
      assert_eq!(
        interpret(&format!("ToExpression[{boxes}, StandardForm]")).unwrap(),
        expected,
        "{boxes}"
      );
    }
  }

  #[test]
  fn held_parse_of_nested_boxes() {
    assert_eq!(
      interpret(
        "ToExpression[RowBox[{SuperscriptBox[\"x\", \"2\"], \"+\", \"1\"}], \
         StandardForm, Hold]"
      )
      .unwrap(),
      "Hold[x^2 + 1]"
    );
  }
}

mod even_step_functions {
  use super::*;

  #[test]
  fn heaviside_pi_and_lambda_are_even() {
    assert_eq!(interpret("HeavisidePi[-x]").unwrap(), "HeavisidePi[x]");
    assert_eq!(
      interpret("HeavisideLambda[-2 x]").unwrap(),
      "HeavisideLambda[2*x]"
    );
  }
}

mod fold_pair_list {
  use super::*;

  #[test]
  fn elements_of_any_head() {
    assert_eq!(
      interpret("FoldPairList[TakeDrop, {a, b, c, d, e, f}, h[1, 3, 2]]")
        .unwrap(),
      "{{a}, {b, c, d}, {e, f}}"
    );
  }

  #[test]
  fn throw_inside_the_function_is_caught() {
    assert_eq!(
      interpret(
        "Catch[FoldPairList[{If[#1 > 1000, Throw[#1]], #1 #2} &, 1, Range[10]]]"
      )
      .unwrap(),
      "5040"
    );
  }
}

mod gaussian_prime_power {
  use super::*;

  #[test]
  fn prime_power_q_over_the_gaussian_integers() {
    assert_eq!(
      interpret("PrimePowerQ[5, GaussianIntegers -> True]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("PrimePowerQ[9, GaussianIntegers -> True]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("PrimePowerQ[2, GaussianIntegers -> True]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("PrimePowerQ[6, GaussianIntegers -> True]").unwrap(),
      "False"
    );
  }
}

mod any_head_collections {
  use super::*;

  #[test]
  fn count_distinct_of_held_arguments() {
    assert_eq!(
      interpret("CountDistinct[Hold[1 + 2, 4 - 1, 1 + 2, 2 + 1]]").unwrap(),
      "3"
    );
  }

  #[test]
  fn numerical_sort_of_any_head_and_associations() {
    assert_eq!(
      interpret("NumericalSort[data[0, 1, Infinity, -Infinity]]").unwrap(),
      "data[-Infinity, 0, 1, Infinity]"
    );
    assert_eq!(
      interpret(
        "NumericalSort[<|a -> 0, b -> 1, c -> Infinity, d -> -Infinity|>]"
      )
      .unwrap(),
      "<|d -> -Infinity, a -> 0, b -> 1, c -> Infinity|>"
    );
    assert_eq!(
      interpret("NumericalSort[{3, -Infinity, Pi, Infinity}]").unwrap(),
      "{-Infinity, 3, Pi, Infinity}"
    );
  }
}

mod string_part_spans {
  use super::*;

  #[test]
  fn spans_pick_characters() {
    assert_eq!(
      interpret("StringPart[\"abcdefghijklm\", 1 ;; 6]").unwrap(),
      "{a, b, c, d, e, f}"
    );
    assert_eq!(
      interpret("StringPart[\"abcdefghijklm\", 1 ;; -1 ;; 2]").unwrap(),
      "{a, c, e, g, i, k, m}"
    );
    assert_eq!(
      interpret("StringPart[\"abcdefghijklm\", -1 ;; 1 ;; -2]").unwrap(),
      "{m, k, i, g, e, c, a}"
    );
  }
}

mod small_forms {
  use super::*;

  #[test]
  fn file_name_parts_of_a_file_object() {
    assert_eq!(
      interpret("FileExtension[File[\"file.txt\"]]").unwrap(),
      "txt"
    );
    assert_eq!(
      interpret("FileBaseName[File[\"a/file.txt\"]]").unwrap(),
      "file"
    );
  }

  #[test]
  fn bit_not_is_an_involution() {
    assert_eq!(interpret("BitNot[BitNot[x]]").unwrap(), "x");
  }

  #[test]
  fn dirichlet_character_threads() {
    assert_eq!(
      interpret("DirichletCharacter[3, 2, {1, 2, 3, 4, 5}]").unwrap(),
      "{1, -1, 0, 1, -1}"
    );
  }

  #[test]
  fn upsample_padding() {
    assert_eq!(
      interpret("Upsample[{1, 2, 3}, 3, 2, -1]").unwrap(),
      "{-1, 1, -1, -1, 2, -1, -1, 3, -1}"
    );
  }

  #[test]
  fn symbolic_power_range() {
    assert_eq!(
      interpret("PowerRange[a, a^4, a]").unwrap(),
      "{a, a^2, a^3, a^4}"
    );
    assert_eq!(
      interpret("PowerRange[1, x^2, Sqrt[x]]").unwrap(),
      "{1, Sqrt[x], x, x^(3/2), x^2}"
    );
  }
}

mod finite_differences {
  use super::*;

  #[test]
  fn difference_delta_in_several_variables() {
    assert_eq!(
      interpret("DifferenceDelta[f[i, j], i, j]").unwrap(),
      "f[i, j] - f[i, 1 + j] - f[1 + i, j] + f[1 + i, 1 + j]"
    );
  }

  #[test]
  fn difference_quotient_orders_and_variables() {
    assert_eq!(
      interpret("DifferenceQuotient[f[x], {x, 2, h}]").unwrap(),
      "(f[x] - 2*f[h + x] + f[2*h + x])/h^2"
    );
    assert_eq!(
      interpret("DifferenceQuotient[f[x, y], {x, r}, {y, s}]").unwrap(),
      "(f[x, y] - f[x, s + y] - f[r + x, y] + f[r + x, s + y])/(r*s)"
    );
    assert_eq!(
      interpret("DifferenceQuotient[f[x], {x, h}, {y, k}]").unwrap(),
      "0"
    );
  }
}
