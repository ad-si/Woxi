// Wavelet argument forms found missing by the reference-page parameter
// sweep (round 5). Expected values are from the Wolfram reference pages.

use super::*;

/// The coefficient arrays of `dwd` (in rule order) agree with `expected`
/// (the 6-digit values printed on the reference page).
fn assert_coefs(dwd: &str, expected: &str) {
  let code = format!(
    "Max[Abs[Flatten[Normal[{dwd}][[All, 2]]] - Flatten[{expected}]]] < 10^-4"
  );
  assert_eq!(
    interpret(&code).unwrap(),
    "True",
    "{dwd} gave {}",
    interpret(&format!("Normal[{dwd}]")).unwrap()
  );
}

mod default_wavelet {
  use super::*;

  #[test]
  fn lifting_1d() {
    assert_eq!(
      interpret(
        "Normal[LiftingWaveletTransform[{1, 2, 3, 4, 5, 6, 7, 8}]][[All, 1]]"
      )
      .unwrap(),
      "{{0}, {1}, {0, 0}, {0, 1}, {0, 0, 0}, {0, 0, 1}}"
    );
    assert_coefs(
      "LiftingWaveletTransform[{56, 40, 8, 24, 48, 48, 40, 16}]",
      "{{67.8823, 22.6274, 67.8823, 39.598}, {-11.3137, 11.3137, 0., -16.9706}, \
       {64., 76.}, {-32., -20.}, {98.9949}, {8.48528}}",
    );
    assert_coefs(
      "LiftingWaveletTransform[Range[8]]",
      "{{2.12132, 4.94975, 7.77817, 10.6066}, {0.707107, 0.707107, 0.707107, \
       0.707107}, {5., 13.}, {2., 2.}, {12.7279}, {5.65685}}",
    );
  }

  #[test]
  fn lifting_automatic_matches_haar() {
    assert_eq!(
      interpret(
        "Normal[LiftingWaveletTransform[Range[8], Automatic]] == \
         Normal[LiftingWaveletTransform[Range[8], HaarWavelet[]]]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn stationary_1d() {
    assert_coefs(
      "StationaryWaveletTransform[{56, 40, 8, 24, 48, 48, 40, 16}]",
      "{{36., 48., 24., 16., 36., 48., 44., 28.}, {20., -8., -16., 8., 12., \
       0., -4., -12.}, {40., 38., 30., 32., 30., 32., 40., 38.}, {-4., 10., \
       -6., -16., 6., 16., 4., -10.}, {35., 35., 35., 35., 35., 35., 35., \
       35.}, {5., 3., -5., -3., -5., -3., 5., 3.}}",
    );
    assert_coefs(
      "StationaryWaveletTransform[Range[4]]",
      "{{2.5, 1.5, 2.5, 3.5}, {-1.5, 0.5, 0.5, 0.5}, {2.5, 2.5, 2.5, 2.5}, \
       {0., -1., 0., 1.}}",
    );
  }

  #[test]
  fn packet_1d() {
    assert_eq!(
      interpret(
        "Normal[DiscreteWaveletPacketTransform[{1, 2, 3, 4, 5, 6, 7, 8}]][[All, 1]]"
      )
      .unwrap(),
      "{{0}, {1}, {0, 0}, {0, 1}, {1, 0}, {1, 1}, {0, 0, 0}, {0, 0, 1}, \
       {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}}"
    );
    assert_coefs(
      "DiscreteWaveletPacketTransform[Range[8], Automatic, 2]",
      "{{2.12132, 4.94975, 7.77817, 10.6066}, {-0.707107, -0.707107, \
       -0.707107, -0.707107}, {5., 13.}, {-2., -2.}, {-1., -1.}, {0., 0.}}",
    );
  }

  #[test]
  fn two_dimensional() {
    for f in [
      "LiftingWaveletTransform",
      "StationaryWaveletTransform",
      "DiscreteWaveletPacketTransform",
    ] {
      assert_eq!(
        interpret(&format!("{f}[{{{{1, 2}}, {{3, 4}}}}][\"Wavelet\"]"))
          .unwrap(),
        "HaarWavelet[]",
        "{f}"
      );
      assert_eq!(
        interpret(&format!(
          "Chop[InverseWaveletTransform[{f}[{{{{1, 2}}, {{3, 4}}}}]] - {{{{1, 2}}, {{3, 4}}}}]"
        ))
        .unwrap(),
        "{{0, 0}, {0, 0}}",
        "{f}"
      );
    }
  }
}

mod lifting_exact_data {
  use super::*;

  #[test]
  fn exact_numeric_data_is_transformed_numerically() {
    // Data from the "Use a higher refinement level" reference example.
    assert_eq!(
      interpret(
        "LiftingWaveletTransform[Table[Sin[x^2], {x, 0, 10, 10/511}], \
         DaubechiesWavelet[4], 3][\"Dimensions\"]"
      )
      .unwrap(),
      "{{0} -> {256}, {1} -> {256}, {0, 0} -> {128}, {0, 1} -> {128}, \
       {0, 0, 0} -> {64}, {0, 0, 1} -> {64}}"
    );
    assert_eq!(
      interpret(
        "Normal[LiftingWaveletTransform[{Sqrt[2], 1, 3, Pi}]] == \
         Normal[LiftingWaveletTransform[N[{Sqrt[2], 1, 3, Pi}]]]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn integer_data_with_daubechies() {
    assert_eq!(
      interpret(
        "Chop[InverseWaveletTransform[LiftingWaveletTransform[Range[16], \
         DaubechiesWavelet[4], 3]] - Range[16]]"
      )
      .unwrap(),
      "{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}"
    );
  }

  #[test]
  fn working_precision_infinity_stays_exact() {
    assert_eq!(
      interpret(
        "Simplify[Normal[LiftingWaveletTransform[{a, b, c, d}, \
         WorkingPrecision -> Infinity]][[3]]]"
      )
      .unwrap(),
      "{0, 0} -> {(a + b + c + d)/2}"
    );
  }
}

mod simplify_rules {
  use super::*;

  #[test]
  fn simplify_threads_over_rules() {
    assert_eq!(interpret("Simplify[x -> (a + a)/2]").unwrap(), "x -> a");
    assert_eq!(
      interpret("Simplify[{x -> {(a + a)/2}, y -> Sqrt[2] (b/Sqrt[2])}]")
        .unwrap(),
      "{x -> {a}, y -> b}"
    );
  }
}

mod threshold {
  use super::*;

  const SURE_DEF: &str = "surethreshold[w_List] := Block[{n = Length[w], sw, s, id}, \
     sw = Sort[w^2]; s = Accumulate[sw] + Range[n - 1, 0, -1] sw; \
     id = First[Ordering[n - 2 Range[n] + s]]; Sqrt[sw[[id]]]]; \
     dwd = LiftingWaveletTransform[Table[2 - 5 t + 5 Exp[-500 (t - 0.5)^2] + \
     0.1 Sin[37000. t^2], {t, 0, 1, 1/1023}], SymletWavelet[2], 5];";

  #[test]
  fn sure_matches_reference_formula() {
    // The reference page computes the "SURE" threshold with this function
    // applied to the finest detail coefficients; every level uses it.
    assert_eq!(
      interpret(&format!(
        "{SURE_DEF} d = surethreshold[First[dwd[{{1}}, \"Values\"]]]; \
         WaveletThreshold[dwd, \"SURE\"][\"ThresholdValues\"] == \
         Thread[{{{{1}}, {{0, 1}}, {{0, 0, 1}}, {{0, 0, 0, 1}}, \
         {{0, 0, 0, 0, 1}}}} -> d]"
      ))
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn soft_sure() {
    // {"Soft", "SURE"} soft-thresholds with the SURE value, which is the
    // same as an explicit {"Soft", delta}.
    assert_eq!(
      interpret(&format!(
        "{SURE_DEF} d = surethreshold[First[dwd[{{1}}, \"Values\"]]]; \
         Normal[WaveletThreshold[dwd, {{\"Soft\", \"SURE\"}}]] == \
         Normal[WaveletThreshold[dwd, {{\"Soft\", d}}]]"
      ))
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(&format!(
        "{SURE_DEF} Normal[WaveletThreshold[dwd, \"SUREShrink\"]] == \
         Normal[WaveletThreshold[dwd, {{\"Soft\", \"SURE\"}}]]"
      ))
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn smooth_garrote() {
    // x^(2n+1) / (x^(2n) + delta^(2n)) on the detail coefficients.
    assert_eq!(
      interpret(
        "dwd = DiscreteWaveletTransform[{1., -3., 2., 0.5}, HaarWavelet[], 1]; \
         sgx = First[dwd[{1}, \"Values\"]]; \
         Max[Abs[First[WaveletThreshold[dwd, {\"SmoothGarrote\", 1, 2}][{1}, \
         \"Values\"]] - sgx^5/(sgx^4 + 1)]] < 10^-12"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "Head[WaveletThreshold[DiscreteWaveletTransform[Range[16] + \
         0.1 Sin[Range[16]^2]], \"SmoothGarrote\"]]"
      )
      .unwrap(),
      "DiscreteWaveletData"
    );
  }
}

mod filter_properties {
  use super::*;

  #[test]
  fn orthogonal() {
    assert_eq!(
      interpret(
        "WaveletFilterCoefficients[DaubechiesWavelet[2], \"Properties\"]"
      )
      .unwrap(),
      "{PrimalLowpass, PrimalHighpass, LiftingFilter, AllLiftingFilter, \
       BestLiftingFilter}"
    );
  }

  #[test]
  fn biorthogonal() {
    assert_eq!(
      interpret(
        "WaveletFilterCoefficients[BiorthogonalSplineWavelet[2, 2], \"Properties\"]"
      )
      .unwrap(),
      "{PrimalLowpass, PrimalHighpass, DualLowpass, DualHighpass, \
       LiftingFilter, AllLiftingFilter, BestLiftingFilter}"
    );
  }
}

mod battle_lemarie_plot {
  use super::*;

  #[test]
  fn plot_psi_and_phi() {
    for f in ["WaveletPsi", "WaveletPhi"] {
      assert_eq!(
        interpret(&format!(
          "Head[Plot[{f}[BattleLemarieWavelet[3, 10], x], {{x, -5, 5}}]]"
        ))
        .unwrap(),
        "Graphics",
        "{f}"
      );
    }
  }

  #[test]
  fn phi_is_symmetric() {
    assert_eq!(
      interpret(
        "Abs[WaveletPhi[BattleLemarieWavelet[3, 10], 0.7] - \
         WaveletPhi[BattleLemarieWavelet[3, 10], -0.7]] < 10^-6"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn filter_matches_reference() {
    assert_eq!(
      interpret(
        "Max[Abs[WaveletFilterCoefficients[BattleLemarieWavelet[3, 5]][[All, 2]] - \
         {0.0297468, 0.0226846, -0.0778079, -0.035498, 0.30683, 0.541736, \
         0.30683, -0.035498, -0.0778079, 0.0226846, 0.0297468}]] < 10^-6"
      )
      .unwrap(),
      "True"
    );
  }
}
