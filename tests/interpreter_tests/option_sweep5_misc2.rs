// Long-tail argument forms found missing by the reference-page parameter
// sweep (round 5, second batch). Expected values are the ones the Wolfram
// reference pages show, or follow from the definitions.

use super::*;

mod gaussian_integers {
  use super::*;

  #[test]
  fn moebius_mu() {
    assert_eq!(interpret("MoebiusMu[5 + 6 I]").unwrap(), "-1");
    // 2 + 2 I = -I (1 + I)^3 has a repeated prime factor.
    assert_eq!(interpret("MoebiusMu[2 + 2 I]").unwrap(), "0");
    // 3 + I = (1 + I)(2 - I): two primes.
    assert_eq!(interpret("MoebiusMu[3 + I]").unwrap(), "1");
  }

  #[test]
  fn modular_inverse() {
    assert_eq!(
      interpret("ModularInverse[3 - I, 5 + 2 I]").unwrap(),
      "-1 - I"
    );
    assert_eq!(
      interpret("Mod[(3 - I) ModularInverse[3 - I, 5 + 2 I], 5 + 2 I]")
        .unwrap(),
      "1"
    );
  }

  #[test]
  fn square_free_q() {
    assert_eq!(interpret("SquareFreeQ[3 + 2 I]").unwrap(), "True");
    assert_eq!(interpret("SquareFreeQ[2 + 2 I]").unwrap(), "False");
  }
}

mod square_free_q_multivariate {
  use super::*;

  #[test]
  fn polynomials_in_several_variables() {
    assert_eq!(interpret("SquareFreeQ[x^3 - x^2 y]").unwrap(), "False");
    assert_eq!(interpret("SquareFreeQ[x y + 1]").unwrap(), "True");
    assert_eq!(
      interpret("SquareFreeQ[(x - y)^2 (x + y)]").unwrap(),
      "False"
    );
  }
}

mod special_values {
  use super::*;

  #[test]
  fn arc_tanh_of_complex_infinity() {
    assert_eq!(
      interpret("ArcTanh[ComplexInfinity]").unwrap(),
      "Indeterminate"
    );
  }
}

mod permutation_cycles_of_cycles {
  use super::*;

  #[test]
  fn cycles_input() {
    assert_eq!(
      interpret("PermutationCycles[Cycles[{{1, 3, 5}, {2, 4, 6}}]]").unwrap(),
      "Cycles[{{1, 3, 5}, {2, 4, 6}}]"
    );
    assert_eq!(
      interpret("PermutationCycles[Cycles[{{1, 5, 6}, {10, 7}}], Identity]")
        .unwrap(),
      "{{1, 5, 6}, {2}, {3}, {4}, {7, 10}, {8}, {9}}"
    );
  }
}

mod digit_count_length {
  use super::*;

  #[test]
  fn last_len_digits() {
    assert_eq!(interpret("DigitCount[122000, 10, 0, 9]").unwrap(), "6");
    assert_eq!(
      interpret("DigitCount[242442422, 3, {1, 2}, 12]").unwrap(),
      "{4, 4}"
    );
    assert_eq!(
      interpret(
        "DigitCount[242442422, 3, 0, 20] == Count[PadLeft[IntegerDigits[242442422, 3], 20], 0]"
      )
      .unwrap(),
      "True"
    );
    // Fewer digits than the number has: only the last ones count.
    assert_eq!(interpret("DigitCount[12345, 10, 1, 3]").unwrap(), "0");
  }
}

mod effective_interest_schedule {
  use super::*;

  #[test]
  fn compound_average_growth_rate() {
    assert_eq!(
      interpret("EffectiveInterest[{r1, r2, r3, r4}]").unwrap(),
      "-1 + ((1 + r1)*(1 + r2)*(1 + r3)*(1 + r4))^(1/4)"
    );
    assert_eq!(
      interpret(
        "Abs[EffectiveInterest[{.05, .065, .04, .07, .085}] - 0.0618842] < 10^-6"
      )
      .unwrap(),
      "True"
    );
  }
}

mod horner_form_variable_list {
  use super::*;

  #[test]
  fn nested_in_each_variable() {
    assert_eq!(
      interpret("HornerForm[x y + 2 x^2 y + 2 x y^2 + 4 x^2 y^2, {x, y}]")
        .unwrap(),
      "x*(y*(1 + 2*y) + x*y*(2 + 4*y))"
    );
    assert_eq!(
      interpret("HornerForm[1 + 2 x + 3 x^2, {x}]").unwrap(),
      "1 + x*(2 + 3*x)"
    );
  }
}

mod integer_reverse_mixed_radix {
  use super::*;

  #[test]
  fn mixed_radix() {
    assert_eq!(
      interpret("IntegerReverse[1234, MixedRadix[{7, 11}]]").unwrap(),
      "170"
    );
  }
}

mod absolute_time_strings {
  use super::*;

  #[test]
  fn date_strings() {
    assert_eq!(
      interpret("AbsoluteTime[\"2022-12-1\"]").unwrap(),
      "3878841600"
    );
  }
}

mod distance_matrix_forms {
  use super::*;

  #[test]
  fn two_lists() {
    assert_eq!(
      interpret("DistanceMatrix[{1, 2, 3, 4}, {5, 6, 7}]").unwrap(),
      "{{4, 5, 6}, {3, 4, 5}, {2, 3, 4}, {1, 2, 3}}"
    );
    assert_eq!(
      interpret(
        "Max[Abs[DistanceMatrix[{{1.5, 4.3}, {2.1, 5.6}, {6.2, 4.5}}, {{1.6, 4.2}, {2.3, 5.4}}] - {{0.141421, 1.36015}, {1.48661, 0.282843}, {4.60977, 4.0025}}]] < 10^-5"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn default_distance_by_data() {
    // Strings and nominal sequences: EditDistance.
    assert_eq!(
      interpret("DistanceMatrix[{\"abcd\", \"bcde\", \"xyz\"}]").unwrap(),
      "{{0, 2, 4}, {2, 0, 4}, {4, 4, 0}}"
    );
    assert_eq!(
      interpret(
        "DistanceMatrix[{{\"a\", \"b\", \"c\", \"d\"}, {\"b\", \"c\", \"d\", \"e\"}, {\"x\", \"y\", \"z\"}}]"
      )
      .unwrap(),
      "{{0, 2, 4}, {2, 0, 4}, {4, 4, 0}}"
    );
    // Nominal vectors: HammingDistance.
    assert_eq!(
      interpret(
        "DistanceMatrix[{{\"A\", \"Sugar\"}, {\"A\", \"Salt\"}, {\"B\", \"Sugar\"}, {\"B\", \"Salt\"}}]"
      )
      .unwrap(),
      "{{0, 1, 1, 2}, {1, 0, 2, 1}, {1, 2, 0, 1}, {2, 1, 1, 0}}"
    );
    // Numerical sequences of different lengths: WarpingDistance.
    assert_eq!(
      interpret(
        "Max[Abs[DistanceMatrix[{{1.1, 1.2, 1.3, 1.4, 1.5}, {1.2, 1.3}, {1.4, 1.5, 1.6}}] - {{0, 0.4, 0.7}, {0.4, 0, 0.7}, {0.7, 0.7, 0}}]] < 10^-12"
      )
      .unwrap(),
      "True"
    );
  }
}

mod bell_y_matrix {
  use super::*;

  #[test]
  fn generalized_bell_polynomial() {
    assert_eq!(
      interpret("BellY[Partition[Range[25], 5]]").unwrap(),
      "68327950346100"
    );
    assert_eq!(
      interpret("Table[BellY[ConstantArray[1, {n, 2}]], {n, 10}]").unwrap(),
      "{1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975}"
    );
    assert_eq!(
      interpret("Table[BellY[ConstantArray[{1, z}, n]], {n, 5}]").unwrap(),
      "{z, z + z^2, z + 3*z^2 + z^3, z + 7*z^2 + 6*z^3 + z^4, z + 15*z^2 + 25*z^3 + 10*z^4 + z^5}"
    );
    assert_eq!(
      interpret("Table[BellY[ConstantArray[1, {n, n}]], {n, 1, 7}]").unwrap(),
      "{1, 2, 12, 154, 3455, 120196, 5995892}"
    );
    assert_eq!(
      interpret("BellY[Table[{1, Cumulant[k]}, {k, 3}]]").unwrap(),
      "Cumulant[1]^3 + 3*Cumulant[1]*Cumulant[2] + Cumulant[3]"
    );
  }

  #[test]
  fn partial_polynomials_combine() {
    assert_eq!(
      interpret(
        "n = 4; Sum[C[k] BellY[n, k, Array[x, {n, 2}]], {k, n}] == BellY[Join[Transpose[{Table[C[k], {k, n}]}], Array[x, {n, 2}], 2]] // Expand"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn numeric_partial_polynomials_evaluate() {
    assert_eq!(interpret("BellY[4, 2, {1, 1, 1}]").unwrap(), "7");
  }
}

mod planar_angle_full_exterior {
  use super::*;

  #[test]
  fn full_exterior() {
    assert_eq!(
      interpret("PlanarAngle[{{1, 1}, {0, 0}, {1, 0}}, \"FullExterior\"]")
        .unwrap(),
      "(7*Pi)/4"
    );
  }
}

mod fourier_dct_forms {
  use super::*;

  #[test]
  fn complex_data() {
    assert_eq!(
      interpret(
        "Max[Abs[FourierDCT[{1, 2 I, 3, 4 I}] - {2. + 3. I, -0.112085 - 1.46508 I, -0.707107 + 0.707107 I, 1.57716 - 1.68925 I}]] < 10^-5"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn named_type() {
    assert_eq!(
      interpret("Round[FourierDCT[{0, 0, 1, 0, 0}, \"IV\"], 0.000001]")
        .unwrap(),
      "{0.447214, -0.447214, -0.447214, 0.447214, 0.447214}"
    );
  }

  #[test]
  fn arrays_are_transformed_along_every_axis() {
    assert_eq!(
      interpret(
        "m = {{1, 2, 5}, {3, 4, 0}}; Chop[FourierDCT[FourierDCT[m], 3] - m]"
      )
      .unwrap(),
      "{{0, 0, 0}, {0, 0, 0}}"
    );
    assert_eq!(
      interpret(
        "m = {{1., 2.}, {3., 4.}}; Chop[FourierDCT[m] - Transpose[FourierDCT /@ Transpose[FourierDCT /@ m]]]"
      )
      .unwrap(),
      "{{0, 0}, {0, 0}}"
    );
  }
}

mod arbitrary_precision_inverses {
  use super::*;

  #[test]
  fn inverse_erfc() {
    assert_eq!(
      interpret(
        "Abs[N[InverseErfc[33/100], 50] - 0.68880252811655645040250472890525783544948992349371`50] < 10^-48"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("Precision[N[InverseErfc[33/100], 50]]").unwrap(),
      "50."
    );
    assert_eq!(
      interpret("Abs[Erf[N[InverseErf[1/3], 30]] - 1/3] < 10^-28").unwrap(),
      "True"
    );
  }

  #[test]
  fn inverse_gudermannian() {
    assert_eq!(
      interpret(
        "Abs[N[InverseGudermannian[4/3], 50] - 2.1261760907822693919362362711721561854503251223380`50] < 10^-48"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("Precision[N[InverseGudermannian[4/3], 50]]").unwrap(),
      "50."
    );
    assert_eq!(
      interpret(
        "Abs[N[InverseGudermannian[I Pi + 5]] - (-0.0829127 + 1.54624 I)] < 10^-6"
      )
      .unwrap(),
      "True"
    );
    // Beyond Pi/2 a real argument has a complex value.
    assert_eq!(
      interpret(
        "Abs[(InverseGudermannian[E] // N) - (0.436551 + 3.14159 I)] < 10^-5"
      )
      .unwrap(),
      "True"
    );
  }
}

mod n_with_accuracy {
  use super::*;

  #[test]
  fn infinite_precision_goal() {
    assert_eq!(
      interpret("Accuracy[N[Exp[100], {Infinity, -10}]]").unwrap(),
      "-10."
    );
    assert_eq!(
      interpret("Round[Accuracy[N[Pi^10, {Infinity, 5}]], 0.001]").unwrap(),
      "5."
    );
    assert_eq!(
      interpret("Round[N[Pi^10, {Infinity, 5}], 0.0001] == 93648.0475")
        .unwrap(),
      "True"
    );
  }
}
