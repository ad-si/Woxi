// Performance fixes found by the reference-page parameter sweep (round 5):
// examples that timed out or crashed in Woxi while Wolfram returns promptly.
// The sizes here are small enough to keep the suite fast but still take the
// fixed code paths.

use super::*;

mod machine_real_statistics {
  use super::*;

  /// Deterministic machine-real data that is not evenly spaced.
  const DATA: &str = "d = N[Table[Sin[k^2] + k/7, {k, 1, 41}]]";

  #[test]
  fn central_moment_matches_the_general_formula_bit_for_bit() {
    clear_state();
    assert_eq!(
      interpret(&format!(
        "{DATA}; m = Mean[d]; \
         Table[CentralMoment[d, r] - Divide[Plus @@ ((d - m)^r), 41], {{r, 1, 4}}]"
      ))
      .unwrap(),
      "{0., 0., 0., 0.}"
    );
  }

  #[test]
  fn moment_is_the_mean_of_the_powers() {
    clear_state();
    assert_eq!(
      interpret(&format!("{DATA}; Moment[d, 3] - Mean[d^3]")).unwrap(),
      "0."
    );
  }

  #[test]
  fn mean_deviation_matches_the_general_formula() {
    clear_state();
    assert_eq!(
      interpret(&format!(
        "{DATA}; MeanDeviation[d] - (Plus @@ Abs[d - Mean[d]])/41"
      ))
      .unwrap(),
      "0."
    );
  }

  #[test]
  fn covariance_matches_the_general_formula() {
    clear_state();
    assert_eq!(
      interpret(&format!(
        "{DATA}; e = Reverse[d]; \
         Covariance[d, e] - (Plus @@ ((d - Mean[d]) (e - Mean[e])))/40"
      ))
      .unwrap(),
      "0."
    );
  }

  #[test]
  fn cumulant_of_reals_is_real() {
    clear_state();
    assert_eq!(
      interpret("Cumulant[{1., 2., 4., 8.}, 2]").unwrap(),
      "7.1875"
    );
    assert_eq!(
      interpret("Cumulant[{1., 2., 4., 8.}, 3]").unwrap(),
      "12.65625"
    );
  }

  #[test]
  fn skewness_and_kurtosis_of_reals() {
    clear_state();
    assert_eq!(
      interpret("Skewness[{1., 2., 4., 8.}]").unwrap(),
      interpret("N[Skewness[{1, 2, 4, 8}]]").unwrap()
    );
    assert_eq!(
      interpret("Kurtosis[{1., 2., 4., 8.}]").unwrap(),
      interpret("N[Kurtosis[{1, 2, 4, 8}]]").unwrap()
    );
  }

  #[test]
  fn geometric_mean_of_many_samples_does_not_underflow() {
    clear_state();
    // The running product of 10^5 numbers in (0, 1) underflows to 0.
    let r = interpret("GeometricMean[N[Table[1/E, {10^5}]]]").unwrap();
    assert!(r.starts_with("0.3678794411"), "{r}");
    assert_eq!(interpret("GeometricMean[{4., 9}]").unwrap(), "6.");
  }

  #[test]
  fn big_samples_finish() {
    clear_state();
    assert_eq!(
      interpret(
        "d = RandomReal[1, 10^5]; \
         {NumberQ[GeometricMean[d]], NumberQ[CentralMoment[d, 2]], \
          NumberQ[Moment[d, 2]], NumberQ[Skewness[d]], NumberQ[Kurtosis[d]], \
          NumberQ[Cumulant[d, 3]], NumberQ[MeanDeviation[d]], \
          NumberQ[Covariance[d, Reverse[d]]], Length[Quartiles[d]], \
          NumberQ[InterquartileRange[d]], NumberQ[QuartileDeviation[d]], \
          NumberQ[QuartileSkewness[d]], NumberQ[BiweightMidvariance[d]]}"
      )
      .unwrap(),
      "{True, True, True, True, True, True, True, True, 3, True, True, True, True}"
    );
  }

  #[test]
  fn quartiles_of_reals() {
    clear_state();
    assert_eq!(
      interpret("Quartiles[{5., 1., 3., 2., 4.}]").unwrap(),
      "{1.75, 3., 4.25}"
    );
  }

  #[test]
  fn biweight_midvariance_of_reals() {
    clear_state();
    assert_eq!(
      interpret(
        "Abs[BiweightMidvariance[{1., 2., 3., 4., 5.}] - \
         BiweightMidvariance[{1, 2, 3, 4, 5}]] < 10^-12"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("BiweightMidvariance[{2., 2., 2.}]").unwrap(),
      "Indeterminate"
    );
  }
}

mod commonest {
  use super::*;

  #[test]
  fn large_list() {
    clear_state();
    assert_eq!(
      interpret("Commonest[Join[Table[1, {10^5}], Table[3, {10^3}], {7, 8}]]")
        .unwrap(),
      "{1}"
    );
    assert_eq!(
      interpret(
        "Commonest[Join[Table[1, {10^5}], Table[3, {10^3}], {7, 8}], 2]"
      )
      .unwrap(),
      "{1, 3}"
    );
  }

  #[test]
  fn many_distinct_elements() {
    clear_state();
    assert_eq!(
      interpret("Commonest[Append[Range[20000], 777]]").unwrap(),
      "{777}"
    );
  }
}

mod moving_average {
  use super::*;

  #[test]
  fn machine_reals() {
    clear_state();
    assert_eq!(
      interpret("MovingAverage[{1.5, 2.25, 3.0, 4.5, 7.}, 2]").unwrap(),
      "{1.875, 2.625, 3.75, 5.75}"
    );
    assert_eq!(
      interpret("MovingAverage[{1.5, 2.25, 3.0}, 1]").unwrap(),
      "{1.5, 2.25, 3.}"
    );
  }

  #[test]
  fn matches_the_window_sum() {
    clear_state();
    assert_eq!(
      interpret(
        "d = N[Table[Sin[k^2], {k, 30}]]; \
         MovingAverage[d, 7] - Table[(Plus @@ d[[i ;; i + 6]])/7, {i, 24}] // Union"
      )
      .unwrap(),
      "{0.}"
    );
  }

  #[test]
  fn large_data() {
    clear_state();
    assert_eq!(
      interpret(
        "data = ConstantArray[1., 10^4] + RandomReal[{-0.1, 0.1}, 10^4]; \
         Length[MovingAverage[data, 100]]"
      )
      .unwrap(),
      "9901"
    );
  }
}

mod max_memory_used {
  use super::*;

  #[test]
  fn measures_a_held_expression() {
    clear_state();
    assert_eq!(
      interpret("Positive[MaxMemoryUsed[Range[10^4]]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("MemberQ[Attributes[MaxMemoryUsed], HoldAll]").unwrap(),
      "True"
    );
  }
}

mod identity_matrix_sparse {
  use super::*;

  #[test]
  fn huge_sparse_identity() {
    clear_state();
    assert_eq!(
      interpret("SparseArrayQ[IdentityMatrix[10^6, SparseArray]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("Dimensions[IdentityMatrix[10^6, SparseArray]]").unwrap(),
      "{1000000, 1000000}"
    );
  }

  #[test]
  fn same_form_as_converting_the_dense_identity() {
    clear_state();
    assert_eq!(
      interpret(
        "IdentityMatrix[3, SparseArray] === SparseArray[IdentityMatrix[3]]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("Normal[IdentityMatrix[{2, 3}, SparseArray]]").unwrap(),
      "{{1, 0, 0}, {0, 1, 0}}"
    );
    assert_eq!(
      interpret("Normal[IdentityMatrix[{3, 2}, SparseArray]]").unwrap(),
      "{{1, 0}, {0, 1}, {0, 0}}"
    );
  }
}

mod prime_counting {
  use super::*;

  #[test]
  fn prime_pi_above_the_sieve_threshold() {
    clear_state();
    assert_eq!(interpret("PrimePi[10^7]").unwrap(), "664579");
    assert_eq!(interpret("PrimePi[123456789]").unwrap(), "7027260");
    assert_eq!(interpret("PrimePi[10^10]").unwrap(), "455052511");
  }

  #[test]
  fn nth_prime_by_counting() {
    clear_state();
    assert_eq!(interpret("Prime[10^8]").unwrap(), "2038074743");
    // Around a sieve boundary: the last prime below 10^8 and the next.
    assert_eq!(interpret("Prime[5761455]").unwrap(), "99999989");
    assert_eq!(interpret("Prime[5761456]").unwrap(), "100000007");
    assert_eq!(interpret("Prime[10^9]").unwrap(), "22801763489");
  }

  #[test]
  fn prime_and_prime_pi_are_inverse() {
    clear_state();
    assert_eq!(interpret("PrimePi[Prime[12345678]]").unwrap(), "12345678");
  }
}

mod expand_power {
  use super::*;

  #[test]
  fn multinomial_power() {
    clear_state();
    assert_eq!(
      interpret("Expand[(2 x - y/3 + 1)^3]").unwrap(),
      "1 + 6*x + 12*x^2 + 8*x^3 - y - 4*x*y - 4*x^2*y + y^2/3 + (2*x*y^2)/3 - y^3/27"
    );
    assert_eq!(
      interpret(
        "Expand[(x + y + z)^5] === Expand[(x + y + z)^2 (x + y + z)^3]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn high_powers() {
    clear_state();
    assert_eq!(interpret("Length[Expand[(x + y)^2000]]").unwrap(), "2001");
    assert_eq!(
      interpret("Length[Expand[(x + y + z)^100]]").unwrap(),
      "5151"
    );
  }

  #[test]
  fn colliding_terms() {
    clear_state();
    assert_eq!(
      interpret("Expand[(1 + x + x^2)^3]").unwrap(),
      "1 + 3*x + 6*x^2 + 7*x^3 + 6*x^4 + 3*x^5 + x^6"
    );
  }
}

mod integer_roots {
  use super::*;

  #[test]
  fn sparse_high_degree() {
    clear_state();
    assert_eq!(
      interpret("Solve[x^12345 - 2 x^777 + 1 == 0, x, Integers]").unwrap(),
      "{{x -> 1}}"
    );
    assert_eq!(
      interpret("Reduce[x^1000 - 2 x^777 + 1 == 0, x, Integers]").unwrap(),
      "x == 1"
    );
  }

  #[test]
  fn multiplicities_and_zero() {
    clear_state();
    assert_eq!(
      interpret("Solve[(x - 1)^3 x^2 == 0, x, Integers]").unwrap(),
      "{{x -> 0}, {x -> 0}, {x -> 1}, {x -> 1}, {x -> 1}}"
    );
    assert_eq!(
      interpret("Reduce[(x - 1)^3 x^2 == 0, x, Integers]").unwrap(),
      "x == 0 || x == 1"
    );
    assert_eq!(
      interpret("Solve[(x - 1)^2 (x + 2) == 0, x, Integers]").unwrap(),
      "{{x -> -2}, {x -> 1}, {x -> 1}}"
    );
  }

  #[test]
  fn no_integer_roots() {
    clear_state();
    assert_eq!(interpret("Solve[x^7 + 5 == 0, x, Integers]").unwrap(), "{}");
    assert_eq!(
      interpret("Reduce[x^7 + 5 == 0, x, Integers]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("Solve[2 x^7 - 3 x + 1 == 1/2, x, Integers]").unwrap(),
      "{}"
    );
  }

  #[test]
  fn several_roots() {
    clear_state();
    assert_eq!(
      interpret("Reduce[x^3 - 6 x^2 + 11 x - 6 == 0, x, Integers]").unwrap(),
      "x == 1 || x == 2 || x == 3"
    );
    assert_eq!(
      interpret("Solve[x^2 == 4, x, Integers]").unwrap(),
      "{{x -> -2}, {x -> 2}}"
    );
  }
}

mod time_constrained_pause {
  use super::*;

  #[test]
  fn check_abort_without_propagation_continues() {
    clear_state();
    let start = std::time::Instant::now();
    let r = interpret_with_stdout(
      "TimeConstrained[CheckAbort[Pause[1]; Echo[\"first\"], Echo[\"second\"], \
       PropagateAborts -> False]; Echo[\"finished CheckAbort\"], .1]",
    )
    .unwrap();
    assert_eq!(r.result, "finished CheckAbort");
    assert_eq!(r.stdout, ">> second\n>> finished CheckAbort\n");
    assert!(start.elapsed().as_secs_f64() < 0.9);
  }

  #[test]
  fn timeout_aborts_propagate_by_default() {
    clear_state();
    let r = interpret_with_stdout(
      "TimeConstrained[CheckAbort[Pause[1], Echo[\"check\"], \
       PropagateAborts -> Automatic]; Echo[\"past CheckAbort\"], .1]",
    )
    .unwrap();
    assert_eq!(r.result, "$Aborted");
    assert_eq!(r.stdout, ">> check\n");
  }

  #[test]
  fn user_aborts_do_not_propagate_by_default() {
    clear_state();
    let r = interpret_with_stdout(
      "CheckAbort[Abort[], Echo[\"check\"], PropagateAborts -> Automatic]; \
       Echo[\"past CheckAbort\"]",
    )
    .unwrap();
    assert_eq!(r.result, "past CheckAbort");
  }

  #[test]
  fn pause_is_interrupted() {
    clear_state();
    let start = std::time::Instant::now();
    assert_eq!(
      interpret("TimeConstrained[Pause[3], 0.1, \"late\"]").unwrap(),
      "late"
    );
    assert_eq!(
      interpret("TimeConstrained[TimeConstrained[Pause[3], 1, \"inner\"], 0.1, \"outer\"]")
        .unwrap(),
      "outer"
    );
    assert!(start.elapsed().as_secs_f64() < 2.0);
    assert_eq!(
      interpret("TimeConstrained[Pause[0.01]; 5, 1]").unwrap(),
      "5"
    );
  }
}

mod nminimize_vector_equality {
  use super::*;

  #[test]
  fn threads_the_equation() {
    clear_state();
    assert_eq!(
      interpret("NMinimize[{x - y, {x + y, x - y} == {1/2, 1}}, {x, y}]")
        .unwrap(),
      "{1., {x -> 0.75, y -> -0.25}}"
    );
  }
}

mod factor_integer_cyclotomic {
  use super::*;

  #[test]
  fn mersenne_numbers() {
    clear_state();
    assert_eq!(
      interpret("Table[Length[FactorInteger[2^n - 1]], {n, 50, 500, 50}]")
        .unwrap(),
      "{7, 12, 13, 18, 11, 25, 20, 24, 24, 22}"
    );
  }

  #[test]
  fn factorizations_are_complete_and_prime() {
    clear_state();
    assert_eq!(
      interpret(
        "Union[Table[With[{f = FactorInteger[m]}, \
         Times @@ Power @@@ f == m && AllTrue[f[[All, 1]], PrimeQ]], \
         {m, {2^250 - 1, 2^500 - 1, 3^120 - 1, 3^125 + 1, 10^72 + 1}}]]"
      )
      .unwrap(),
      "{True}"
    );
  }

  #[test]
  fn repeated_factors_merge() {
    clear_state();
    // The prime 2 divides several cyclotomic pieces of 3^90 - 1.
    assert_eq!(
      interpret("FactorInteger[3^90 - 1] === FactorInteger[Times @@ Power @@@ FactorInteger[3^90 - 1]]")
        .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("FactorInteger[2^64 - 1]").unwrap(),
      "{{3, 1}, {5, 1}, {17, 1}, {257, 1}, {641, 1}, {65537, 1}, {6700417, 1}}"
    );
  }
}
