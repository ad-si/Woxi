use super::*;

mod wavelets {
  use super::*;

  // -------------------------------------------------------------------------
  // Wavelet family heads (symbolic constructor objects)
  // -------------------------------------------------------------------------

  #[test]
  fn family_heads_stay_unevaluated() {
    assert_eq!(interpret("HaarWavelet[]").unwrap(), "HaarWavelet[]");
    assert_eq!(
      interpret("DaubechiesWavelet[4]").unwrap(),
      "DaubechiesWavelet[4]"
    );
    assert_eq!(interpret("SymletWavelet[]").unwrap(), "SymletWavelet[]");
    assert_eq!(
      interpret("MexicanHatWavelet[2]").unwrap(),
      "MexicanHatWavelet[2]"
    );
    assert_eq!(interpret("CDFWavelet[]").unwrap(), "CDFWavelet[]");
  }

  // -------------------------------------------------------------------------
  // WaveletFilterCoefficients
  // -------------------------------------------------------------------------

  #[test]
  fn haar_primal_lowpass() {
    // Machine reals by default; exact fractions only on WorkingPrecision.
    assert_eq!(
      interpret("WaveletFilterCoefficients[HaarWavelet[]]").unwrap(),
      "{{0, 0.5}, {1, 0.5}}"
    );
    assert_eq!(
      interpret(
        "WaveletFilterCoefficients[HaarWavelet[], WorkingPrecision -> Infinity]"
      )
      .unwrap(),
      "{{0, 1/2}, {1, 1/2}}"
    );
  }

  #[test]
  fn haar_primal_highpass() {
    assert_eq!(
      interpret("WaveletFilterCoefficients[HaarWavelet[], \"PrimalHighpass\"]")
        .unwrap(),
      "{{0, 0.5}, {1, -0.5}}"
    );
  }

  #[test]
  fn daubechies2_exact() {
    assert_eq!(
      interpret(
        "WaveletFilterCoefficients[DaubechiesWavelet[2], \"PrimalLowpass\", WorkingPrecision -> Infinity]"
      )
      .unwrap(),
      "{{0, (1 + Sqrt[3])/8}, {1, (3 + Sqrt[3])/8}, {2, (3 - Sqrt[3])/8}, {3, (1 - Sqrt[3])/8}}"
    );
  }

  #[test]
  fn daubechies_lowpass_sums_to_one() {
    assert_eq!(
      interpret(
        "Abs[Total[Last /@ WaveletFilterCoefficients[DaubechiesWavelet[7]]] - 1] < 1*^-12"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn daubechies_orthogonality() {
    // Sum h_k h_{k+2} == 0 (shifted orthogonality, sum-1 normalization)
    assert_eq!(
      interpret(
        "h = Last /@ WaveletFilterCoefficients[DaubechiesWavelet[4]]; Abs[Total[Drop[h, 2]*Drop[h, -2]]] < 1*^-12"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn highpass_sums_to_zero() {
    assert_eq!(
      interpret(
        "Abs[Total[Last /@ WaveletFilterCoefficients[SymletWavelet[5], \"PrimalHighpass\"]]] < 1*^-10"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn biorthogonal_spline_primal_is_long_filter() {
    // Wolfram labels the longer complementary filter "Primal" and the shorter
    // B-spline filter "Dual"; machine reals by default.
    assert_eq!(
      interpret(
        "WaveletFilterCoefficients[BiorthogonalSplineWavelet[2, 2], \"PrimalLowpass\"]"
      )
      .unwrap(),
      "{{-2, -0.125}, {-1, 0.25}, {0, 0.75}, {1, 0.25}, {2, -0.125}}"
    );
    assert_eq!(
      interpret(
        "WaveletFilterCoefficients[BiorthogonalSplineWavelet[2, 2], \"DualLowpass\"]"
      )
      .unwrap(),
      "{{-1, 0.25}, {0, 0.5}, {1, 0.25}}"
    );
  }

  #[test]
  fn biorthogonal_spline_exact_rationals() {
    // WorkingPrecision -> Infinity gives the exact dyadic rationals.
    assert_eq!(
      interpret(
        "WaveletFilterCoefficients[BiorthogonalSplineWavelet[2, 2], \"PrimalLowpass\", WorkingPrecision -> Infinity]"
      )
      .unwrap(),
      "{{-2, -1/8}, {-1, 1/4}, {0, 3/4}, {1, 1/4}, {2, -1/8}}"
    );
    assert_eq!(
      interpret(
        "WaveletFilterCoefficients[BiorthogonalSplineWavelet[2, 2], \"DualLowpass\", WorkingPrecision -> Infinity]"
      )
      .unwrap(),
      "{{-1, 1/4}, {0, 1/2}, {1, 1/4}}"
    );
    // The analyzing highpass filters are reported with flipped sign.
    assert_eq!(
      interpret(
        "WaveletFilterCoefficients[BiorthogonalSplineWavelet[2, 2], \"PrimalHighpass\", WorkingPrecision -> Infinity]"
      )
      .unwrap(),
      "{{0, -1/4}, {1, 1/2}, {2, -1/4}}"
    );
    assert_eq!(
      interpret(
        "WaveletFilterCoefficients[BiorthogonalSplineWavelet[2, 2], \"DualHighpass\", WorkingPrecision -> Infinity]"
      )
      .unwrap(),
      "{{-1, -1/8}, {0, -1/4}, {1, 3/4}, {2, -1/4}, {3, -1/8}}"
    );
  }

  #[test]
  fn reverse_biorthogonal_swaps_primal_and_dual() {
    assert_eq!(
      interpret(
        "WaveletFilterCoefficients[ReverseBiorthogonalSplineWavelet[2, 2], \"DualLowpass\"] === WaveletFilterCoefficients[BiorthogonalSplineWavelet[2, 2], \"PrimalLowpass\"]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn cdf53_default_is_machine_exact_on_infinity() {
    assert_eq!(
      interpret("WaveletFilterCoefficients[CDFWavelet[\"5/3\"]]").unwrap(),
      "{{-1, 0.25}, {0, 0.5}, {1, 0.25}}"
    );
    assert_eq!(
      interpret(
        "WaveletFilterCoefficients[CDFWavelet[\"5/3\"], WorkingPrecision -> Infinity]"
      )
      .unwrap(),
      "{{-1, 1/4}, {0, 1/2}, {1, 1/4}}"
    );
  }

  #[test]
  fn cdf97_matches_jpeg2000_analysis_filter() {
    // The 9-tap analysis (dual) lowpass center coefficient.
    assert_eq!(
      interpret(
        "h = WaveletFilterCoefficients[CDFWavelet[], \"DualLowpass\"]; {Length[h], Abs[h[[5, 2]] - 0.6029490182363579] < 1*^-10}"
      )
      .unwrap(),
      "{9, True}"
    );
  }

  #[test]
  fn shannon_filter_exact_values() {
    // Machine reals by default.
    assert_eq!(
      interpret("f = WaveletFilterCoefficients[ShannonWavelet[2]]; f[[3]]")
        .unwrap(),
      "{0, 0.5}"
    );
    assert_eq!(
      interpret(
        "f = WaveletFilterCoefficients[ShannonWavelet[2]]; {f[[2]], f[[4]]}"
      )
      .unwrap(),
      "{{-1, 0.3183098861837907}, {1, 0.3183098861837907}}"
    );
    // Exact closed form on WorkingPrecision -> Infinity.
    assert_eq!(
      interpret(
        "f = WaveletFilterCoefficients[ShannonWavelet[2], WorkingPrecision -> Infinity]; {f[[2]], f[[3]], f[[4]]}"
      )
      .unwrap(),
      "{{-1, Pi^(-1)}, {0, 1/2}, {1, Pi^(-1)}}"
    );
  }

  #[test]
  fn meyer_filter_nearly_sums_to_one() {
    assert_eq!(
      interpret(
        "Abs[Total[Last /@ WaveletFilterCoefficients[MeyerWavelet[3]]] - 1] < 0.01"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn battle_lemarie_orthonormality() {
    // Sum h^2 == 1/2 for orthogonal families (sum-1 normalization).
    assert_eq!(
      interpret(
        "h = Last /@ WaveletFilterCoefficients[BattleLemarieWavelet[1]]; Abs[Total[h^2] - 1/2] < 1*^-4"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn symlet_matches_reference_table() {
    assert_eq!(
      interpret(
        "h = WaveletFilterCoefficients[SymletWavelet[4]]; Abs[h[[1, 2]] - 0.0322231006040427/Sqrt[2]] < 1*^-12"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn coiflet_matches_reference_table() {
    assert_eq!(
      interpret(
        "h = WaveletFilterCoefficients[CoifletWavelet[1]]; {Length[h], Abs[h[[3, 2]] - 0.8525720202116004/Sqrt[2]] < 1*^-12}"
      )
      .unwrap(),
      "{6, True}"
    );
  }

  #[test]
  fn filter_spec_list_gives_list_of_results() {
    assert_eq!(
      interpret(
        "WaveletFilterCoefficients[HaarWavelet[], {\"PrimalLowpass\", \"PrimalHighpass\"}]"
      )
      .unwrap(),
      "{{{0, 0.5}, {1, 0.5}}, {{0, 0.5}, {1, -0.5}}}"
    );
  }

  #[test]
  fn lifting_filter_gives_lifting_filter_data() {
    assert_eq!(
      interpret("WaveletFilterCoefficients[HaarWavelet[], \"LiftingFilter\"]")
        .unwrap(),
      "LiftingFilterData[HaarWavelet[]]"
    );
    assert_eq!(
      interpret(
        "WaveletFilterCoefficients[HaarWavelet[], \"LiftingFilter\"][\"PrimalLowpass\"]"
      )
      .unwrap(),
      "{{0, 0.5}, {1, 0.5}}"
    );
  }

  #[test]
  fn invalid_wavelet_stays_unevaluated() {
    assert_eq!(
      interpret("WaveletFilterCoefficients[SymletWavelet[25]]").unwrap(),
      "WaveletFilterCoefficients[SymletWavelet[25]]"
    );
  }

  // -------------------------------------------------------------------------
  // DiscreteWaveletTransform / DiscreteWaveletData
  // -------------------------------------------------------------------------

  #[test]
  fn dwt_haar_level1_coefficients() {
    assert_eq!(
      interpret(
        "DiscreteWaveletTransform[{1, 2, 3, 4}, HaarWavelet[], 1][All]"
      )
      .unwrap(),
      "{{0} -> {2.121320343559643, 4.949747468305833}, {1} -> {-0.7071067811865476, -0.7071067811865476}}"
    );
  }

  #[test]
  fn dwt_default_refinement() {
    // floor(log2(4) + 1/2) == 2
    assert_eq!(
      interpret("DiscreteWaveletTransform[{1, 2, 3, 4}][\"Refinement\"]")
        .unwrap(),
      "2"
    );
    // floor(log2(8) + 1/2) == 3
    assert_eq!(
      interpret("DiscreteWaveletTransform[Range[8]][\"Refinement\"]").unwrap(),
      "3"
    );
  }

  #[test]
  fn dwt_default_wavelet_is_haar() {
    assert_eq!(
      interpret("DiscreteWaveletTransform[{1, 2, 3, 4}][\"Wavelet\"]").unwrap(),
      "HaarWavelet[]"
    );
  }

  #[test]
  fn dwt_basis_index() {
    assert_eq!(
      interpret("DiscreteWaveletTransform[{1, 2, 3, 4}][\"BasisIndex\"]")
        .unwrap(),
      "{{1}, {0, 1}, {0, 0}}"
    );
  }

  #[test]
  fn dwd_wind_access_and_values_form() {
    assert_eq!(
      interpret("DiscreteWaveletTransform[{1, 2, 3, 4}][{0, 1}]").unwrap(),
      "{{0, 1} -> {-2.}}"
    );
    assert_eq!(
      interpret("DiscreteWaveletTransform[{1, 2, 3, 4}][{0, 1}, \"Values\"]")
        .unwrap(),
      "{{-2.}}"
    );
  }

  #[test]
  fn dwd_pattern_access() {
    assert_eq!(
      interpret("DiscreteWaveletTransform[Range[8]][{0, _}] // Length")
        .unwrap(),
      "2"
    );
  }

  #[test]
  fn dwd_multiple_wind_access() {
    assert_eq!(
      interpret("DiscreteWaveletTransform[{1, 2, 3, 4}][{{0}, {1}}] // Length")
        .unwrap(),
      "2"
    );
  }

  #[test]
  fn normal_gives_rules() {
    assert_eq!(
      interpret("Normal[DiscreteWaveletTransform[{1, 2, 3, 4}]] // Length")
        .unwrap(),
      "4"
    );
  }

  #[test]
  fn dwt_energy_conservation_orthogonal() {
    assert_eq!(
      interpret(
        "data = Range[16] // N; dwd = DiscreteWaveletTransform[data, DaubechiesWavelet[3]]; Abs[Norm[data] - Norm[Flatten[Last /@ dwd[Automatic]]]] < 1*^-9"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn dwt_2d_subbands() {
    assert_eq!(
      interpret("DiscreteWaveletTransform[{{1, 2}, {3, 4}}][All]").unwrap(),
      "{{0} -> {{5.000000000000001}}, {1} -> {{-1.0000000000000002}}, {2} -> {{-2.}}, {3} -> {{0.}}}"
    );
  }

  #[test]
  fn dwt_rejects_invalid_data() {
    assert_eq!(
      interpret("DiscreteWaveletTransform[\"abc\"]").unwrap(),
      "DiscreteWaveletTransform[abc]"
    );
  }

  // -------------------------------------------------------------------------
  // InverseWaveletTransform
  // -------------------------------------------------------------------------

  #[test]
  fn inverse_recovers_data_haar() {
    assert_eq!(
      interpret(
        "d = InverseWaveletTransform[DiscreteWaveletTransform[{1., 2., 3., 4.}]]; Max[Abs[d - {1, 2, 3, 4}]] < 1*^-12"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn inverse_recovers_data_many_families() {
    for wave in [
      "DaubechiesWavelet[2]",
      "DaubechiesWavelet[5]",
      "SymletWavelet[4]",
      "SymletWavelet[8]",
      "CoifletWavelet[1]",
      "CoifletWavelet[3]",
      "BiorthogonalSplineWavelet[2, 2]",
      "BiorthogonalSplineWavelet[4, 2]",
      "ReverseBiorthogonalSplineWavelet[2, 2]",
      "CDFWavelet[]",
      "CDFWavelet[\"5/3\"]",
    ] {
      let code = format!(
        "d = InverseWaveletTransform[DiscreteWaveletTransform[Range[16] // N, {wave}]]; Max[Abs[d - Range[16]]] < 1*^-8"
      );
      assert_eq!(interpret(&code).unwrap(), "True", "family: {wave}");
    }
  }

  #[test]
  fn inverse_recovers_odd_length_data() {
    assert_eq!(
      interpret(
        "d = InverseWaveletTransform[DiscreteWaveletTransform[Range[9] // N, DaubechiesWavelet[3]]]; Max[Abs[d - Range[9]]] < 1*^-9"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn inverse_recovers_2d_data() {
    assert_eq!(
      interpret(
        "m = Table[N[i^2 + j], {i, 6}, {j, 6}]; rec = InverseWaveletTransform[DiscreteWaveletTransform[m, DaubechiesWavelet[2]]]; Max[Abs[Flatten[rec - m]]] < 1*^-9"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn inverse_with_reflected_padding() {
    assert_eq!(
      interpret(
        "d = InverseWaveletTransform[DiscreteWaveletTransform[Range[10] // N, DaubechiesWavelet[2], 2, Padding -> \"Reflected\"]]; Max[Abs[d - Range[10]]] < 1*^-9"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn inverse_symbolic_exact() {
    assert_eq!(
      interpret(
        "Simplify[InverseWaveletTransform[DiscreteWaveletTransform[{a, b, c, d}, HaarWavelet[]]]]"
      )
      .unwrap(),
      "{a, b, c, d}"
    );
  }

  #[test]
  fn inverse_partial_returns_smaller_dwd() {
    assert_eq!(
      interpret(
        "pd = InverseWaveletTransform[DiscreteWaveletTransform[Range[8]], Automatic, 1]; {Head[pd], pd[\"Refinement\"]}"
      )
      .unwrap(),
      "{DiscreteWaveletData, 2}"
    );
  }

  #[test]
  fn inverse_with_wind_uses_only_those_coefficients() {
    // Reconstruction from the coarsest coefficients of constant data
    // reproduces the constant; detail-only reconstruction is zero.
    assert_eq!(
      interpret(
        "dwd = DiscreteWaveletTransform[{5., 5., 5., 5.}]; Chop[InverseWaveletTransform[dwd, Automatic, {1}]]"
      )
      .unwrap(),
      "{0, 0, 0, 0}"
    );
    assert_eq!(
      interpret(
        "dwd = DiscreteWaveletTransform[{5., 5., 5., 5.}]; Max[Abs[InverseWaveletTransform[dwd, Automatic, {0, 0}] - 5]] < 1*^-12"
      )
      .unwrap(),
      "True"
    );
  }

  // -------------------------------------------------------------------------
  // Stationary, packet, and lifting transforms
  // -------------------------------------------------------------------------

  #[test]
  fn swt_coefficients_same_length_as_data() {
    assert_eq!(
      interpret(
        "swd = StationaryWaveletTransform[{1, 2, 3, 4}, HaarWavelet[], 1]; Length /@ (Last /@ swd[All])"
      )
      .unwrap(),
      "{4, 4}"
    );
  }

  #[test]
  fn swt_level1_values() {
    // The stationary transform convolves with sum-1 filters (no Sqrt[2]).
    assert_eq!(
      interpret(
        "StationaryWaveletTransform[{1, 2, 3, 4}, HaarWavelet[], 1][{0}, \"Values\"]"
      )
      .unwrap(),
      "{{2.5, 1.5, 2.5, 3.5}}"
    );
    assert_eq!(
      interpret(
        "StationaryWaveletTransform[{1, 2, 3, 4}, HaarWavelet[], 1][{1}, \"Values\"]"
      )
      .unwrap(),
      "{{-1.5, 0.5, 0.5, 0.5}}"
    );
  }

  #[test]
  fn swt_level2_atrous_dilation() {
    // Level 2 convolves the level-1 approximation with the filter dilated by 2.
    assert_eq!(
      interpret(
        "StationaryWaveletTransform[{1, 2, 3, 4, 5, 6, 7, 8}, HaarWavelet[], 2][{0, 0}, \"Values\"]"
      )
      .unwrap(),
      "{{5.5, 4.5, 3.5, 2.5, 3.5, 4.5, 5.5, 6.5}}"
    );
  }

  #[test]
  fn swt_exact_symbolic() {
    assert_eq!(
      interpret(
        "Normal[StationaryWaveletTransform[{a, b, c, d}, HaarWavelet[], 1, WorkingPrecision -> Infinity]]"
      )
      .unwrap(),
      "{{0} -> {a/2 + d/2, a/2 + b/2, b/2 + c/2, c/2 + d/2}, {1} -> {a/2 - d/2, -1/2*a + b/2, -1/2*b + c/2, -1/2*c + d/2}}"
    );
    assert_eq!(
      interpret(
        "Simplify[InverseWaveletTransform[StationaryWaveletTransform[{a, b, c, d}, HaarWavelet[], 1, WorkingPrecision -> Infinity]]]"
      )
      .unwrap(),
      "{a, b, c, d}"
    );
  }

  #[test]
  fn swt_inverse_recovers_data() {
    assert_eq!(
      interpret(
        "d = InverseWaveletTransform[StationaryWaveletTransform[Range[12] // N, DaubechiesWavelet[2]]]; Max[Abs[d - Range[12]]] < 1*^-9"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn packet_transform_full_tree() {
    assert_eq!(
      interpret(
        "DiscreteWaveletPacketTransform[{0, 0, 1, 0}, HaarWavelet[], 2][\"WaveletIndex\"]"
      )
      .unwrap(),
      "{{0}, {1}, {0, 0}, {0, 1}, {1, 0}, {1, 1}}"
    );
  }

  #[test]
  fn packet_default_refinement_capped_at_4() {
    assert_eq!(
      interpret(
        "DiscreteWaveletPacketTransform[Range[64] // N][\"Refinement\"]"
      )
      .unwrap(),
      "4"
    );
  }

  #[test]
  fn packet_basis_is_deepest_level() {
    assert_eq!(
      interpret(
        "DiscreteWaveletPacketTransform[{1, 2, 3, 4}, HaarWavelet[], 2][\"BasisIndex\"]"
      )
      .unwrap(),
      "{{0, 0}, {0, 1}, {1, 0}, {1, 1}}"
    );
  }

  #[test]
  fn packet_inverse_recovers_data() {
    assert_eq!(
      interpret(
        "d = InverseWaveletTransform[DiscreteWaveletPacketTransform[Range[8] // N, SymletWavelet[3]]]; Max[Abs[d - Range[8]]] < 1*^-9"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn stationary_packet_inverse_recovers_data() {
    assert_eq!(
      interpret(
        "d = InverseWaveletTransform[StationaryWaveletPacketTransform[Range[8] // N, HaarWavelet[], 2]]; Max[Abs[d - Range[8]]] < 1*^-9"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn lifting_transform_exact_symbolic() {
    assert_eq!(
      interpret(
        "Normal[LiftingWaveletTransform[{1, 1, 3, 1}, HaarWavelet[], 1, WorkingPrecision -> Infinity]]"
      )
      .unwrap(),
      "{{0} -> {Sqrt[2], 2*Sqrt[2]}, {1} -> {0, -Sqrt[2]}}"
    );
  }

  #[test]
  fn lifting_transform_symbolic_data() {
    assert_eq!(
      interpret(
        "LiftingWaveletTransform[{a, b}, HaarWavelet[], 1][{1}, \"Values\"]"
      )
      .unwrap(),
      "{{Sqrt[2]*(-1/2*a + b/2)}}"
    );
  }

  #[test]
  fn lifting_default_refinement_from_factorization() {
    // 2-adic valuation of 100 is 2.
    assert_eq!(
      interpret("LiftingWaveletTransform[Range[100] // N][\"Refinement\"]")
        .unwrap(),
      "2"
    );
  }

  #[test]
  fn lifting_inverse_recovers_data() {
    assert_eq!(
      interpret(
        "d = InverseWaveletTransform[LiftingWaveletTransform[Range[8] // N, SymletWavelet[4]]]; Max[Abs[d - Range[8]]] < 1*^-9"
      )
      .unwrap(),
      "True"
    );
  }

  // -------------------------------------------------------------------------
  // WaveletThreshold / WaveletMapIndexed / WaveletBestBasis
  // -------------------------------------------------------------------------

  #[test]
  fn threshold_soft_explicit_delta() {
    assert_eq!(
      interpret(
        "WaveletThreshold[DiscreteWaveletTransform[{1., 5., 2., 8.}], {\"Soft\", 1.}][{1}, \"Values\"]"
      )
      .unwrap(),
      "{{-1.8284271247461903, -3.2426406871192857}}"
    );
  }

  #[test]
  fn threshold_hard_explicit_delta() {
    assert_eq!(
      interpret(
        "WaveletThreshold[DiscreteWaveletTransform[{1., 5., 2., 8.}], {\"Hard\", 3.}][{1}, \"Values\"]"
      )
      .unwrap(),
      "{{0., -4.242640687119286}}"
    );
  }

  #[test]
  fn threshold_returns_dwd_and_records_values() {
    assert_eq!(
      interpret(
        "t = WaveletThreshold[DiscreteWaveletTransform[Range[8] // N]]; {Head[t], Length[t[\"ThresholdValues\"]]}"
      )
      .unwrap(),
      "{DiscreteWaveletData, 3}"
    );
  }

  #[test]
  fn threshold_keeps_coarse_coefficients() {
    assert_eq!(
      interpret(
        "dwd = DiscreteWaveletTransform[Range[4] // N]; t = WaveletThreshold[dwd, {\"Hard\", 100.}]; t[{0, 0}, \"Values\"] === dwd[{0, 0}, \"Values\"]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn threshold_largest_coefficients() {
    assert_eq!(
      interpret(
        "t = WaveletThreshold[DiscreteWaveletTransform[{9., 1., 1., 1., 8., 1., 1., 1.}, HaarWavelet[], 1], {\"LargestCoefficients\", 1}]; Count[Flatten[Last /@ t[All]], 0. | 0]"
      )
      .unwrap(),
      "3"
    );
  }

  #[test]
  fn map_indexed_applies_to_all_coefficients() {
    assert_eq!(
      interpret(
        "WaveletMapIndexed[#1*0 &, DiscreteWaveletTransform[{1, 2, 3, 4}]][All]"
      )
      .unwrap(),
      "{{0} -> {0., 0.}, {1} -> {0., 0.}, {0, 0} -> {0.}, {0, 1} -> {0.}}"
    );
  }

  #[test]
  fn map_indexed_receives_wavelet_index() {
    assert_eq!(
      interpret(
        "WaveletMapIndexed[(#2 &), DiscreteWaveletTransform[{1, 2, 3, 4}, HaarWavelet[], 1]][{1}]"
      )
      .unwrap(),
      "{{1} -> {1}}"
    );
  }

  #[test]
  fn map_indexed_with_wind_spec() {
    assert_eq!(
      interpret(
        "dwd = DiscreteWaveletTransform[{1., 2., 3., 4.}, HaarWavelet[], 1]; m = WaveletMapIndexed[#1*0 &, dwd, {1}]; {m[{0}, \"Values\"] === dwd[{0}, \"Values\"], m[{1}, \"Values\"]}"
      )
      .unwrap(),
      "{True, {{0., 0.}}}"
    );
  }

  #[test]
  fn best_basis_returns_packet_dwd_with_basis() {
    assert_eq!(
      interpret(
        "b = WaveletBestBasis[DiscreteWaveletPacketTransform[{1, 1, 2, 3, 2, 2, 1, 1}]]; Head[b]"
      )
      .unwrap(),
      "DiscreteWaveletData"
    );
  }

  #[test]
  fn best_basis_constant_data_concentrates_energy() {
    // For constant data the energy keeps concentrating into the coarse
    // chain, so the entropy-minimal basis follows it to the deepest level.
    assert_eq!(
      interpret(
        "WaveletBestBasis[DiscreteWaveletPacketTransform[Table[1., {8}]]][\"BasisIndex\"]"
      )
      .unwrap(),
      "{{0, 0, 0}, {0, 0, 1}, {0, 1}, {1}}"
    );
  }

  #[test]
  fn best_basis_requires_packet_transform() {
    assert_eq!(
      interpret(
        "WaveletBestBasis[DiscreteWaveletTransform[{1, 2, 3, 4}]] // Head"
      )
      .unwrap(),
      "WaveletBestBasis"
    );
  }

  #[test]
  fn best_basis_inverse_still_recovers_data() {
    assert_eq!(
      interpret(
        "b = WaveletBestBasis[DiscreteWaveletPacketTransform[Range[8] // N]]; d = InverseWaveletTransform[b]; Max[Abs[d - Range[8]]] < 1*^-9"
      )
      .unwrap(),
      "True"
    );
  }

  // -------------------------------------------------------------------------
  // Continuous wavelet transform
  // -------------------------------------------------------------------------

  #[test]
  fn cwt_defaults() {
    assert_eq!(
      interpret(
        "cwd = ContinuousWaveletTransform[Range[32] // N]; {cwd[\"Octaves\"], cwd[\"Voices\"], cwd[\"Wavelet\"]}"
      )
      .unwrap(),
      "{4, 4, MexicanHatWavelet[1]}"
    );
  }

  #[test]
  fn cwt_coefficient_arrays_match_data_length() {
    assert_eq!(
      interpret(
        "cwd = ContinuousWaveletTransform[Range[16] // N]; Union[Length /@ (Last /@ cwd[All])]"
      )
      .unwrap(),
      "{16}"
    );
  }

  #[test]
  fn cwt_octave_voice_spec() {
    assert_eq!(
      interpret(
        "ContinuousWaveletTransform[Range[16] // N, MorletWavelet[], {2, 3}][\"WaveletIndex\"] // Length"
      )
      .unwrap(),
      "6"
    );
  }

  #[test]
  fn inverse_cwt_recovers_data() {
    assert_eq!(
      interpret(
        "data = Table[Sin[2 Pi k/16.], {k, 0, 31}]; rec = InverseContinuousWaveletTransform[ContinuousWaveletTransform[data]]; Max[Abs[rec - data]] < 0.01"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn cwt_scales_property() {
    assert_eq!(
      interpret(
        "s = ContinuousWaveletTransform[Range[8] // N][\"Scales\"]; ({1, 1} /. s) == 2*2^(1/4)"
      )
      .unwrap(),
      "True"
    );
  }

  // -------------------------------------------------------------------------
  // WaveletPhi / WaveletPsi
  // -------------------------------------------------------------------------

  #[test]
  fn haar_phi_is_piecewise() {
    assert_eq!(
      interpret("WaveletPhi[HaarWavelet[], x]").unwrap(),
      "Piecewise[{{1, Inequality[0, LessEqual, x, Less, 1]}}, 0]"
    );
  }

  #[test]
  fn haar_psi_values() {
    assert_eq!(
      interpret(
        "{WaveletPsi[HaarWavelet[], 0.25], WaveletPsi[HaarWavelet[], 0.75]}"
      )
      .unwrap(),
      "{1, -1}"
    );
  }

  #[test]
  fn mexican_hat_psi_formula() {
    assert_eq!(
      interpret("WaveletPsi[MexicanHatWavelet[1], t]").unwrap(),
      "(2*E^(-1/2*t^2)*(1 - t^2))/(Sqrt[3]*Pi^(1/4))"
    );
  }

  #[test]
  fn mexican_hat_equals_dgaussian2() {
    // MexicanHatWavelet[] is DGaussianWavelet[2]; compare numerically since
    // the two symbolic forms differ only in Pi^(1/4) vs Sqrt[Sqrt[Pi]].
    assert_eq!(
      interpret(
        "Abs[WaveletPsi[MexicanHatWavelet[1], 0.7] - WaveletPsi[DGaussianWavelet[2], 0.7]] < 1*^-12"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn morlet_psi_at_zero() {
    assert_eq!(
      interpret(
        "Abs[WaveletPsi[MorletWavelet[], 0.] - 0.7511250525754707] < 1*^-10"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn shannon_phi_is_sinc() {
    assert_eq!(
      interpret("WaveletPhi[ShannonWavelet[], x]").unwrap(),
      "Sinc[Pi*x]"
    );
  }

  #[test]
  fn daubechies_phi_cascade_value() {
    // phi_db2(1) == (1 + Sqrt[3])/2
    assert_eq!(
      interpret(
        "Abs[WaveletPhi[DaubechiesWavelet[2], 1.] - (1 + Sqrt[3])/2] < 1*^-4"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn phi_symbolic_argument_stays_unevaluated_for_cascade_families() {
    assert_eq!(
      interpret("WaveletPhi[DaubechiesWavelet[2], x] // Head").unwrap(),
      "WaveletPhi"
    );
  }

  #[test]
  fn gabor_psi_is_complex() {
    assert_eq!(
      interpret("Head[WaveletPsi[GaborWavelet[], 0.5]]").unwrap(),
      "Complex"
    );
  }

  #[test]
  fn meyer_phi_at_zero_near_peak() {
    // The Meyer scaling function peaks at the origin with phi(0) ~ 1.05 and
    // decays away from it.
    assert_eq!(
      interpret(
        "{Abs[WaveletPhi[MeyerWavelet[3], 0.] - 1.05] < 0.01, Abs[WaveletPhi[MeyerWavelet[3], 8.]] < 0.01}"
      )
      .unwrap(),
      "{True, True}"
    );
  }

  // -------------------------------------------------------------------------
  // Visualization
  // -------------------------------------------------------------------------

  #[test]
  fn wavelet_list_plot_returns_graphics() {
    assert_eq!(
      interpret(
        "Head[WaveletListPlot[DiscreteWaveletTransform[Table[Sin[k/3.], {k, 16}]]]]"
      )
      .unwrap(),
      "Graphics"
    );
  }

  #[test]
  fn wavelet_scalogram_returns_graphics() {
    assert_eq!(
      interpret(
        "Head[WaveletScalogram[DiscreteWaveletTransform[Table[Sin[k/3.], {k, 16}]]]]"
      )
      .unwrap(),
      "Graphics"
    );
    assert_eq!(
      interpret(
        "Head[WaveletScalogram[ContinuousWaveletTransform[Table[Sin[k/3.], {k, 16}]]]]"
      )
      .unwrap(),
      "Graphics"
    );
  }

  #[test]
  fn wavelet_matrix_plot_returns_graphics() {
    assert_eq!(
      interpret(
        "Head[WaveletMatrixPlot[DiscreteWaveletTransform[Table[N[i + j], {i, 8}, {j, 8}]]]]"
      )
      .unwrap(),
      "Graphics"
    );
  }

  #[test]
  fn wavelet_image_plot_returns_image() {
    assert_eq!(
      interpret(
        "Head[WaveletImagePlot[DiscreteWaveletTransform[Table[N[Mod[i j, 5]], {i, 8}, {j, 8}]]]]"
      )
      .unwrap(),
      "Image"
    );
  }

  #[test]
  fn wavelet_matrix_plot_requires_2d_data() {
    assert_eq!(
      interpret(
        "WaveletMatrixPlot[DiscreteWaveletTransform[{1, 2, 3, 4}]] // Head"
      )
      .unwrap(),
      "WaveletMatrixPlot"
    );
  }
}
