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
