use super::*;

// ─── Construction & Normalization ───────────────────────────────────────────

#[test]
fn interval_single_span() {
  assert_eq!(interpret("Interval[{1, 3}]").unwrap(), "Interval[{1, 3}]");
}

#[test]
fn interval_auto_sort_endpoints() {
  assert_eq!(interpret("Interval[{5, 1}]").unwrap(), "Interval[{1, 5}]");
}

#[test]
fn interval_merge_overlapping() {
  assert_eq!(
    interpret("Interval[{1, 3}, {2, 4}]").unwrap(),
    "Interval[{1, 4}]"
  );
}

#[test]
fn interval_merge_adjacent() {
  assert_eq!(
    interpret("Interval[{1, 3}, {3, 5}]").unwrap(),
    "Interval[{1, 5}]"
  );
}

#[test]
fn interval_no_merge_disjoint() {
  assert_eq!(
    interpret("Interval[{1, 2}, {4, 5}]").unwrap(),
    "Interval[{1, 2}, {4, 5}]"
  );
}

#[test]
fn interval_empty() {
  assert_eq!(interpret("Interval[]").unwrap(), "Interval[]");
}

#[test]
fn interval_multiple_merge() {
  assert_eq!(
    interpret("Interval[{1, 3}, {5, 7}, {2, 6}]").unwrap(),
    "Interval[{1, 7}]"
  );
}

#[test]
fn interval_rational_endpoints() {
  assert_eq!(
    interpret("Interval[{1/3, 2/3}]").unwrap(),
    "Interval[{1/3, 2/3}]"
  );
}

#[test]
fn interval_infinity_endpoint() {
  assert_eq!(
    interpret("Interval[{-Infinity, 5}]").unwrap(),
    "Interval[{-Infinity, 5}]"
  );
}

// ─── Arithmetic: Plus ───────────────────────────────────────────────────────

#[test]
fn interval_plus_two_intervals() {
  assert_eq!(
    interpret("Interval[{1, 2}] + Interval[{3, 4}]").unwrap(),
    "Interval[{4, 6}]"
  );
}

#[test]
fn interval_plus_scalar() {
  assert_eq!(
    interpret("Interval[{1, 2}] + 3").unwrap(),
    "Interval[{4, 5}]"
  );
}

#[test]
fn interval_plus_multi_span() {
  assert_eq!(
    interpret("Interval[{1,2}] + Interval[{3,5},{7,9}]").unwrap(),
    "Interval[{4, 7}, {8, 11}]"
  );
}

// ─── Arithmetic: Times ──────────────────────────────────────────────────────

#[test]
fn interval_times_two_intervals() {
  assert_eq!(
    interpret("Interval[{-2, 1}] * Interval[{3, 4}]").unwrap(),
    "Interval[{-8, 4}]"
  );
}

#[test]
fn interval_times_scalar() {
  assert_eq!(
    interpret("2 * Interval[{1, 3}]").unwrap(),
    "Interval[{2, 6}]"
  );
}

#[test]
fn interval_times_zero() {
  assert_eq!(
    interpret("0 * Interval[{1, 3}]").unwrap(),
    "Interval[{0, 0}]"
  );
}

#[test]
fn interval_times_multi_span() {
  assert_eq!(
    interpret("Interval[{1,2}] * Interval[{3,5},{7,9}]").unwrap(),
    "Interval[{3, 18}]"
  );
}

// ─── Arithmetic: Divide ─────────────────────────────────────────────────────

#[test]
fn interval_divide_scalar_by_interval() {
  assert_eq!(
    interpret("1 / Interval[{2, 4}]").unwrap(),
    "Interval[{1/4, 1/2}]"
  );
}

#[test]
fn interval_reciprocal() {
  assert_eq!(
    interpret("Interval[{1, 2}]^(-1)").unwrap(),
    "Interval[{1/2, 1}]"
  );
}

// ─── Arithmetic: Power ──────────────────────────────────────────────────────

#[test]
fn interval_power_even_spans_zero() {
  assert_eq!(
    interpret("Interval[{-1, 2}]^2").unwrap(),
    "Interval[{0, 4}]"
  );
}

#[test]
fn interval_power_even_negative() {
  assert_eq!(
    interpret("Interval[{-3, -1}]^2").unwrap(),
    "Interval[{1, 9}]"
  );
}

#[test]
fn interval_power_even_positive() {
  assert_eq!(interpret("Interval[{2, 3}]^2").unwrap(), "Interval[{4, 9}]");
}

#[test]
fn interval_power_odd() {
  assert_eq!(
    interpret("Interval[{2, 3}]^3").unwrap(),
    "Interval[{8, 27}]"
  );
}

#[test]
fn interval_power_odd_negative() {
  assert_eq!(
    interpret("Interval[{-2, 1}]^3").unwrap(),
    "Interval[{-8, 1}]"
  );
}

// ─── Min / Max ──────────────────────────────────────────────────────────────

#[test]
fn interval_min() {
  assert_eq!(interpret("Min[Interval[{3, 5}]]").unwrap(), "3");
}

#[test]
fn interval_max() {
  assert_eq!(interpret("Max[Interval[{1, 3}, {5, 7}]]").unwrap(), "7");
}

#[test]
fn interval_min_multi_span() {
  assert_eq!(interpret("Min[Interval[{3, 5}, {1, 2}]]").unwrap(), "1");
}

// ─── IntervalUnion ──────────────────────────────────────────────────────────

#[test]
fn interval_union_disjoint() {
  assert_eq!(
    interpret("IntervalUnion[Interval[{1, 3}], Interval[{5, 7}]]").unwrap(),
    "Interval[{1, 3}, {5, 7}]"
  );
}

#[test]
fn interval_union_overlapping() {
  assert_eq!(
    interpret("IntervalUnion[Interval[{1, 4}], Interval[{3, 7}]]").unwrap(),
    "Interval[{1, 7}]"
  );
}

// ─── IntervalIntersection ───────────────────────────────────────────────────

#[test]
fn interval_intersection_overlapping() {
  assert_eq!(
    interpret("IntervalIntersection[Interval[{1, 3}], Interval[{2, 5}]]")
      .unwrap(),
    "Interval[{2, 3}]"
  );
}

#[test]
fn interval_intersection_disjoint() {
  assert_eq!(
    interpret("IntervalIntersection[Interval[{1, 2}], Interval[{4, 5}]]")
      .unwrap(),
    "Interval[]"
  );
}

// ─── IntervalMemberQ ────────────────────────────────────────────────────────

#[test]
fn interval_member_q_point_true() {
  assert_eq!(
    interpret("IntervalMemberQ[Interval[{1, 5}], 3]").unwrap(),
    "True"
  );
}

#[test]
fn interval_member_q_point_false() {
  assert_eq!(
    interpret("IntervalMemberQ[Interval[{1, 5}], 6]").unwrap(),
    "False"
  );
}

#[test]
fn interval_member_q_sub_interval_true() {
  assert_eq!(
    interpret("IntervalMemberQ[Interval[{1, 5}], Interval[{2, 3}]]").unwrap(),
    "True"
  );
}

#[test]
fn interval_member_q_multi_span() {
  assert_eq!(
    interpret("IntervalMemberQ[Interval[{1, 3}, {5, 7}], 4]").unwrap(),
    "False"
  );
}

#[test]
fn interval_member_q_multi_span_true() {
  assert_eq!(
    interpret("IntervalMemberQ[Interval[{1, 3}, {5, 7}], 6]").unwrap(),
    "True"
  );
}

#[test]
fn interval_member_q_list_threads() {
  // A list of test points threads, returning one Boolean per point.
  assert_eq!(
    interpret("IntervalMemberQ[Interval[{1, 5}], {2, 7}]").unwrap(),
    "{True, False}"
  );
  assert_eq!(
    interpret("IntervalMemberQ[Interval[{1, 5}], {0, 3, 6}]").unwrap(),
    "{False, True, False}"
  );
  assert_eq!(
    interpret("IntervalMemberQ[Interval[{1, 3}, {5, 7}], {2, 4, 6}]").unwrap(),
    "{True, False, True}"
  );
  assert_eq!(
    interpret("IntervalMemberQ[Interval[{1, 5}], {}]").unwrap(),
    "{}"
  );
}

// ─── Comparisons ────────────────────────────────────────────────────────────

#[test]
fn interval_less_disjoint_true() {
  assert_eq!(
    interpret("Less[Interval[{1, 2}], Interval[{5, 6}]]").unwrap(),
    "True"
  );
}

#[test]
fn interval_less_overlapping_unevaluated() {
  let result = interpret("Less[Interval[{1, 5}], Interval[{3, 7}]]").unwrap();
  assert!(result.contains("Less"));
}

#[test]
fn interval_greater_scalar() {
  assert_eq!(interpret("Greater[Interval[{5, 6}], 3]").unwrap(), "True");
}

#[test]
fn interval_less_equal_true() {
  assert_eq!(
    interpret("LessEqual[Interval[{1, 2}], Interval[{2, 3}]]").unwrap(),
    "True"
  );
}

#[test]
fn interval_greater_equal_true() {
  assert_eq!(
    interpret("GreaterEqual[Interval[{5, 6}], Interval[{3, 4}]]").unwrap(),
    "True"
  );
}

// ─── CenteredInterval intersection/union ────────────────────────────────

#[test]
fn centered_interval_real_passthrough() {
  // CenteredInterval[c, r] with real c, r is just the inert wrapper:
  // Woxi must not emit the "not yet implemented" warning anymore.
  assert_eq!(
    interpret("CenteredInterval[5, 1]").unwrap(),
    "CenteredInterval[5, 1]"
  );
}

#[test]
fn centered_interval_intersection_real() {
  // Real centered intervals: [4, 6] ∩ [5, 9] = [5, 6] → centre 11/2, radius 1/2.
  assert_eq!(
    interpret(
      "IntervalIntersection[CenteredInterval[5, 1], CenteredInterval[7, 2]]"
    )
    .unwrap(),
    "CenteredInterval[11/2, 1/2]"
  );
}

#[test]
fn centered_interval_union_real() {
  // Smallest centred interval containing both: span [4, 9] → centre 13/2, radius 5/2.
  assert_eq!(
    interpret("IntervalUnion[CenteredInterval[5, 1], CenteredInterval[7, 2]]")
      .unwrap(),
    "CenteredInterval[13/2, 5/2]"
  );
}

#[test]
fn centered_interval_complex_intersection_audit_case() {
  // The audit case: rectangular centred intervals in the complex plane.
  //   B₁: c=2+3i, r=1+i  → Re∈[1, 3], Im∈[2, 4]
  //   B₂: c=1+i,  r=2+2i → Re∈[-1, 3], Im∈[-1, 3]
  // Intersection: Re∈[1, 3], Im∈[2, 3] → centre 2 + 5/2·i, radii (1, 1/2).
  assert_eq!(
    interpret(
      "IntervalIntersection[CenteredInterval[2 + 3*I, 1 + I], \
       CenteredInterval[1 + I, 2 + 2*I]]"
    )
    .unwrap(),
    "CenteredInterval[2 + (5*I)/2, 1 + I/2]"
  );
}

#[test]
fn centered_interval_complex_union_audit_case() {
  // Smallest axis-aligned box covering both:
  //   Re∈[-1, 3], Im∈[-1, 4] → centre 1 + 3/2·i, radii (2, 5/2).
  assert_eq!(
    interpret(
      "IntervalUnion[CenteredInterval[2 + 3*I, 1 + I], \
       CenteredInterval[1 + I, 2 + 2*I]]"
    )
    .unwrap(),
    "CenteredInterval[1 + (3*I)/2, 2 + (5*I)/2]"
  );
}

#[test]
fn centered_interval_intersection_disjoint_returns_empty() {
  // Disjoint real intervals → empty Interval[].
  assert_eq!(
    interpret(
      "IntervalIntersection[CenteredInterval[0, 1], CenteredInterval[10, 1]]"
    )
    .unwrap(),
    "Interval[]"
  );
}

// ─── Sign predicates on intervals ────────────────────────────────────────────

#[test]
fn positive_interval_all_positive() {
  assert_eq!(interpret("Positive[Interval[{1, 5}]]").unwrap(), "True");
  assert_eq!(interpret("Negative[Interval[{1, 5}]]").unwrap(), "False");
  assert_eq!(interpret("NonNegative[Interval[{1, 5}]]").unwrap(), "True");
  assert_eq!(interpret("NonPositive[Interval[{1, 5}]]").unwrap(), "False");
}

#[test]
fn sign_interval_all_negative() {
  assert_eq!(interpret("Positive[Interval[{-5, -1}]]").unwrap(), "False");
  assert_eq!(interpret("Negative[Interval[{-5, -1}]]").unwrap(), "True");
  assert_eq!(
    interpret("NonPositive[Interval[{-5, -1}]]").unwrap(),
    "True"
  );
}

#[test]
fn sign_interval_touching_zero() {
  // [0,5]: not strictly positive, but non-negative.
  assert_eq!(interpret("Positive[Interval[{0, 5}]]").unwrap(), "False");
  assert_eq!(interpret("NonNegative[Interval[{0, 5}]]").unwrap(), "True");
  // [-5,0]: not strictly negative, but non-positive.
  assert_eq!(interpret("NonPositive[Interval[{-5, 0}]]").unwrap(), "True");
}

#[test]
fn sign_interval_straddling_zero_is_indeterminate() {
  assert_eq!(
    interpret("Positive[Interval[{-1, 5}]]").unwrap(),
    "Positive[Interval[{-1, 5}]]"
  );
  assert_eq!(
    interpret("Negative[Interval[{-1, 5}]]").unwrap(),
    "Negative[Interval[{-1, 5}]]"
  );
}

// ─── Abs on intervals ────────────────────────────────────────────────────────

#[test]
fn abs_interval_straddling_zero() {
  // Contains 0 → lower bound becomes 0.
  assert_eq!(
    interpret("Abs[Interval[{-2, 3}]]").unwrap(),
    "Interval[{0, 3}]"
  );
  assert_eq!(
    interpret("Abs[Interval[{-3, 1}]]").unwrap(),
    "Interval[{0, 3}]"
  );
}

#[test]
fn abs_interval_one_sided() {
  assert_eq!(
    interpret("Abs[Interval[{2, 5}]]").unwrap(),
    "Interval[{2, 5}]"
  );
  // All negative: endpoints flip.
  assert_eq!(
    interpret("Abs[Interval[{-5, -2}]]").unwrap(),
    "Interval[{2, 5}]"
  );
}

#[test]
fn abs_interval_multiple_segments_merge() {
  assert_eq!(
    interpret("Abs[Interval[{1, 2}, {4, 5}]]").unwrap(),
    "Interval[{1, 2}, {4, 5}]"
  );
  // Symmetric segments collapse onto one another.
  assert_eq!(
    interpret("Abs[Interval[{-5, -2}, {2, 5}]]").unwrap(),
    "Interval[{2, 5}]"
  );
}

// ─── Floor / Ceiling / Round on intervals ────────────────────────────────────

#[test]
fn floor_ceiling_round_interval() {
  // Monotonic non-decreasing functions map the span endpoints.
  assert_eq!(
    interpret("Round[Interval[{1.2, 3.8}]]").unwrap(),
    "Interval[{1, 4}]"
  );
  assert_eq!(
    interpret("Floor[Interval[{1.2, 3.8}]]").unwrap(),
    "Interval[{1, 3}]"
  );
  assert_eq!(
    interpret("Ceiling[Interval[{1.2, 3.8}]]").unwrap(),
    "Interval[{2, 4}]"
  );
}

#[test]
fn floor_interval_negative_and_multi_span() {
  assert_eq!(
    interpret("Floor[Interval[{-2, 2}]]").unwrap(),
    "Interval[{-2, 2}]"
  );
  assert_eq!(
    interpret("Round[Interval[{1, 3}, {5, 7}]]").unwrap(),
    "Interval[{1, 3}, {5, 7}]"
  );
}

// ─── Monotonic elementary functions over intervals ──────────────────────────

#[test]
fn sqrt_interval() {
  assert_eq!(
    interpret("Sqrt[Interval[{4, 9}]]").unwrap(),
    "Interval[{2, 3}]"
  );
  assert_eq!(
    interpret("Sqrt[Interval[{0, 16}]]").unwrap(),
    "Interval[{0, 4}]"
  );
  // Multiple spans map independently.
  assert_eq!(
    interpret("Sqrt[Interval[{1, 4}, {9, 16}]]").unwrap(),
    "Interval[{1, 2}, {3, 4}]"
  );
}

#[test]
fn exp_interval() {
  assert_eq!(
    interpret("Exp[Interval[{0, 1}]]").unwrap(),
    "Interval[{1, E}]"
  );
  assert_eq!(
    interpret("Exp[Interval[{-1, 1}]]").unwrap(),
    "Interval[{E^(-1), E}]"
  );
}

#[test]
fn log_interval() {
  assert_eq!(
    interpret("Log[Interval[{1, E}]]").unwrap(),
    "Interval[{0, 1}]"
  );
  // Symbolic endpoint is preserved.
  assert_eq!(
    interpret("Log[Interval[{1, 100}]]").unwrap(),
    "Interval[{0, Log[100]}]"
  );
}

// Out-of-domain endpoints (a complex image) leave the call unevaluated.
#[test]
fn sqrt_interval_out_of_domain_unevaluated() {
  assert_eq!(
    interpret("Sqrt[Interval[{-1, 4}]]").unwrap(),
    "Sqrt[Interval[{-1, 4}]]"
  );
}

#[test]
fn hyperbolic_interval() {
  assert_eq!(
    interpret("Sinh[Interval[{0, 1}]]").unwrap(),
    "Interval[{0, Sinh[1]}]"
  );
  assert_eq!(
    interpret("Tanh[Interval[{0, 1}]]").unwrap(),
    "Interval[{0, Tanh[1]}]"
  );
  assert_eq!(
    interpret("ArcSinh[Interval[{0, 1}]]").unwrap(),
    "Interval[{0, ArcSinh[1]}]"
  );
  assert_eq!(
    interpret("ArcTanh[Interval[{0, 1/2}]]").unwrap(),
    "Interval[{0, ArcTanh[1/2]}]"
  );
}

#[test]
fn inverse_trig_interval_increasing() {
  assert_eq!(
    interpret("ArcTan[Interval[{0, 1}]]").unwrap(),
    "Interval[{0, Pi/4}]"
  );
  assert_eq!(
    interpret("ArcSin[Interval[{0, 1/2}]]").unwrap(),
    "Interval[{0, Pi/6}]"
  );
  assert_eq!(
    interpret("ArcSin[Interval[{-1, 1}]]").unwrap(),
    "Interval[{-1/2*Pi, Pi/2}]"
  );
}

// ArcCos is decreasing, so the mapped endpoints come back swapped and sorted.
#[test]
fn arccos_interval_decreasing() {
  assert_eq!(
    interpret("ArcCos[Interval[{0, 1/2}]]").unwrap(),
    "Interval[{Pi/3, Pi/2}]"
  );
}

// Out-of-domain endpoints leave the call unevaluated.
#[test]
fn arctanh_interval_out_of_domain_unevaluated() {
  assert_eq!(
    interpret("ArcTanh[Interval[{0, 2}]]").unwrap(),
    "ArcTanh[Interval[{0, 2}]]"
  );
}

// Interval ^ Interval: for non-negative bases the image is the min/max over
// the four corner values. Verified against wolframscript.
#[test]
fn interval_power_interval_basic() {
  assert_eq!(
    interpret("Interval[{2, 3}]^Interval[{2, 3}]").unwrap(),
    "Interval[{4, 27}]"
  );
}

#[test]
fn interval_power_interval_base_straddles_one() {
  assert_eq!(
    interpret("Interval[{1/2, 2}]^Interval[{2, 3}]").unwrap(),
    "Interval[{1/8, 8}]"
  );
}

#[test]
fn interval_power_interval_negative_exponent_span() {
  assert_eq!(
    interpret("Interval[{2, 3}]^Interval[{-1, 1}]").unwrap(),
    "Interval[{1/3, 3}]"
  );
  assert_eq!(
    interpret("Interval[{3, 5}]^Interval[{-2, -1}]").unwrap(),
    "Interval[{1/25, 1/3}]"
  );
}

#[test]
fn interval_power_interval_base_contains_zero() {
  // 0^positive = 0, so the lower corner is 0.
  assert_eq!(
    interpret("Interval[{0, 2}]^Interval[{2, 3}]").unwrap(),
    "Interval[{0, 8}]"
  );
}

#[test]
fn interval_power_interval_negative_base_unevaluated() {
  // A base spanning negative values is left unevaluated (wolframscript).
  assert_eq!(
    interpret("Interval[{-2, 3}]^Interval[{2, 3}]").unwrap(),
    "Interval[{-2, 3}]^Interval[{2, 3}]"
  );
}

// ─── Sin/Cos of intervals (range over the span) ─────────────────────────────

#[test]
fn sin_interval_monotonic_segment() {
  assert_eq!(
    interpret("Sin[Interval[{Pi/6, Pi/3}]]").unwrap(),
    "Interval[{1/2, Sqrt[3]/2}]"
  );
}

#[test]
fn sin_interval_includes_maximum() {
  // Pi/2 (where Sin = 1) lies inside, so the upper bound is 1.
  assert_eq!(
    interpret("Sin[Interval[{0, Pi}]]").unwrap(),
    "Interval[{0, 1}]"
  );
}

#[test]
fn sin_interval_includes_both_extrema() {
  assert_eq!(
    interpret("Sin[Interval[{0, 2 Pi}]]").unwrap(),
    "Interval[{-1, 1}]"
  );
  assert_eq!(
    interpret("Sin[Interval[{-Pi/2, Pi/2}]]").unwrap(),
    "Interval[{-1, 1}]"
  );
}

#[test]
fn sin_interval_descending_segment() {
  assert_eq!(
    interpret("Sin[Interval[{Pi, 2 Pi}]]").unwrap(),
    "Interval[{-1, 0}]"
  );
}

#[test]
fn cos_interval_ranges() {
  assert_eq!(
    interpret("Cos[Interval[{0, Pi}]]").unwrap(),
    "Interval[{-1, 1}]"
  );
  assert_eq!(
    interpret("Cos[Interval[{0, Pi/2}]]").unwrap(),
    "Interval[{0, 1}]"
  );
  assert_eq!(
    interpret("Cos[Interval[{Pi/3, 2 Pi/3}]]").unwrap(),
    "Interval[{-1/2, 1/2}]"
  );
}

// ─── MinMax of an interval ────────────────────────────────────────────────────

#[test]
fn min_max_interval() {
  // The overall extent of the interval.
  assert_eq!(interpret("MinMax[Interval[{2, 7}]]").unwrap(), "{2, 7}");
  assert_eq!(interpret("MinMax[Interval[{-3, 4}]]").unwrap(), "{-3, 4}");
  // A multi-segment interval spans its outermost endpoints.
  assert_eq!(
    interpret("MinMax[Interval[{1, 2}, {5, 8}]]").unwrap(),
    "{1, 8}"
  );
  // The optional expansion argument widens the result.
  assert_eq!(interpret("MinMax[Interval[{2, 7}], 1]").unwrap(), "{1, 8}");
  // Symbolic bounds stay unevaluated.
  assert_eq!(
    interpret("MinMax[Interval[{a, b}]]").unwrap(),
    "MinMax[Interval[{a, b}]]"
  );
}
