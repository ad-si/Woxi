// Elementary functions applied to `Interval`, `Between` with intervals and
// the `IntervalMemberQ` operator form, found missing by the reference-page
// parameter sweep (round 4).

use super::*;

fn check(input: &str, expected: &str) {
  assert_eq!(interpret(input).unwrap(), expected, "input: {input}");
}

mod unbounded_trig {
  use super::*;

  #[test]
  fn sin_cos_cover_full_range() {
    check("Sin[Interval[{-Infinity, Infinity}]]", "Interval[{-1, 1}]");
    check("Cos[Interval[{0, Infinity}]]", "Interval[{-1, 1}]");
    check("Sin[Interval[{-Infinity, 0}]]", "Interval[{-1, 1}]");
  }

  #[test]
  fn tan_cot_cover_all_reals() {
    check(
      "Tan[Interval[{0, Infinity}]]",
      "Interval[{-Infinity, Infinity}]",
    );
    check(
      "Cot[Interval[{-Infinity, Infinity}]]",
      "Interval[{-Infinity, Infinity}]",
    );
  }

  #[test]
  fn sec_csc_cover_both_branches() {
    check(
      "Sec[Interval[{-Infinity, Infinity}]]",
      "Interval[{-Infinity, -1}, {1, Infinity}]",
    );
    check(
      "Csc[Interval[{1, Infinity}]]",
      "Interval[{-Infinity, -1}, {1, Infinity}]",
    );
  }
}

mod degree_trig {
  use super::*;

  #[test]
  fn sin_cos_degrees() {
    check(
      "SinDegrees[Interval[{-Infinity, Infinity}]]",
      "Interval[{-1, 1}]",
    );
    check("SinDegrees[Interval[{0, 90}]]", "Interval[{0, 1}]");
    check("SinDegrees[Interval[{30, 150}]]", "Interval[{1/2, 1}]");
    check("CosDegrees[Interval[{60, 180}]]", "Interval[{-1, 1/2}]");
    check("CosDegrees[Interval[{0, 720}]]", "Interval[{-1, 1}]");
  }

  #[test]
  fn tan_cot_degrees() {
    check("TanDegrees[Interval[{0, 45}]]", "Interval[{0, 1}]");
    check(
      "TanDegrees[Interval[{45, 135}]]",
      "Interval[{-Infinity, -1}, {1, Infinity}]",
    );
    check("CotDegrees[Interval[{30, 45}]]", "Interval[{1, Sqrt[3]}]");
  }

  #[test]
  fn sec_csc_degrees() {
    check("SecDegrees[Interval[{0, 60}]]", "Interval[{1, 2}]");
    check(
      "SecDegrees[Interval[{60, 120}]]",
      "Interval[{-Infinity, -2}, {2, Infinity}]",
    );
    check("CscDegrees[Interval[{30, 150}]]", "Interval[{1, 2}]");
  }

  #[test]
  fn inverse_degrees() {
    check("ArcSinDegrees[Interval[{0, 1/2}]]", "Interval[{0, 30}]");
    check("ArcCosDegrees[Interval[{0, 1/2}]]", "Interval[{60, 90}]");
    check("ArcTanDegrees[Interval[{-1, 1}]]", "Interval[{-45, 45}]");
  }
}

mod arccot {
  use super::*;

  #[test]
  fn decreasing_branch() {
    check("ArcCot[Interval[{1, Sqrt[3]}]]", "Interval[{Pi/6, Pi/4}]");
    check(
      "ArcCotDegrees[Interval[{1, Sqrt[3]}]]",
      "Interval[{30, 45}]",
    );
    check("ArcCot[Interval[{1, Infinity}]]", "Interval[{0, Pi/4}]");
    check("ArcCot[Interval[{0, 1}]]", "Interval[{Pi/4, Pi/2}]");
  }

  #[test]
  fn endpoints_in_arctan_form() {
    check(
      "ArcCot[Interval[{2, 3}]]",
      "Interval[{ArcTan[1/3], ArcTan[1/2]}]",
    );
  }

  #[test]
  fn span_across_the_jump_stays_unevaluated() {
    check("ArcCot[Interval[{-1, 1}]]", "ArcCot[Interval[{-1, 1}]]");
  }
}

mod piecewise_monotone {
  use super::*;

  #[test]
  fn sign() {
    check("Sign[Interval[{1, 3}]]", "Interval[{1, 1}]");
    check("Sign[Interval[{-1, 3}]]", "Interval[{-1, 1}]");
    check("Sign[Interval[{a, b}]]", "Sign[Interval[{a, b}]]");
  }

  #[test]
  fn unit_step_and_integer_part() {
    check("UnitStep[Interval[{-1, 2}]]", "Interval[{0, 1}]");
    check("IntegerPart[Interval[{-1, 3/2}]]", "Interval[{-1, 1}]");
  }

  #[test]
  fn real_abs() {
    check("RealAbs[Interval[{-5, 3}]]", "Interval[{0, 5}]");
  }

  #[test]
  fn re_and_conjugate_of_real_interval() {
    check("Re[Interval[{-1, 1}]]", "Interval[{-1, 1}]");
    check("Conjugate[Interval[{-5, 5}]]", "Interval[{-5, 5}]");
    check("Re[Interval[{a, b}]]", "Re[Interval[{a, b}]]");
  }
}

mod between_interval {
  use super::*;

  #[test]
  fn point_and_subinterval() {
    check("Between[3, Interval[{2, 5}]]", "True");
    check("Between[7, Interval[{0, 10}]]", "True");
    check("Between[Interval[{0, 1}], Interval[{2, 5}]]", "False");
    check("Between[Interval[{2, 4}], Interval[{2, 5}]]", "True");
    check("Between[Interval[{3, 4}], Interval[{2, 5}]]", "True");
  }

  #[test]
  fn operator_form() {
    check("Between[Interval[{2, 5}]][3]", "True");
    check("Between[{1, 5}][3]", "True");
    check("Select[{1, 4, 7}, Between[{2, 6}]]", "{4}");
  }

  #[test]
  fn symbolic_stays_unevaluated() {
    check(
      "Between[x, Interval[{a, b}]]",
      "Between[x, Interval[{a, b}]]",
    );
  }
}

mod interval_member_q_operator {
  use super::*;

  #[test]
  fn applies_to_point() {
    check("IntervalMemberQ[Interval[{3, 4}]][Pi]", "True");
    check("IntervalMemberQ[Interval[{3, 4}]][5]", "False");
    check(
      "IntervalMemberQ[Interval[{3, 4}]]",
      "IntervalMemberQ[Interval[{3, 4}]]",
    );
  }
}
