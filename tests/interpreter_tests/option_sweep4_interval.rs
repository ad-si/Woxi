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
}

mod reciprocal_inverses {
  use super::*;

  #[test]
  fn circular_in_reciprocal_form() {
    check("ArcSec[Interval[{2, 5}]]", "Interval[{Pi/3, ArcCos[1/5]}]");
    check("ArcCsc[Interval[{1, 5}]]", "Interval[{ArcSin[1/5], Pi/2}]");
    check(
      "ArcSecDegrees[Interval[{2, 5}]]",
      "Interval[{60, ArcCosDegrees[1/5]}]",
    );
    check(
      "ArcCscDegrees[Interval[{1, 5}]]",
      "Interval[{ArcSinDegrees[1/5], 90}]",
    );
  }

  #[test]
  fn hyperbolic_in_reciprocal_form() {
    check("ArcCoth[Interval[{1, Coth[5]}]]", "Interval[{5, Infinity}]");
    check(
      "ArcCsch[Interval[{-4, Csch[2]}]]",
      "Interval[{-Infinity, -ArcSinh[1/4]}, {2, Infinity}]",
    );
  }

  #[test]
  fn out_of_domain_stays_unevaluated() {
    check("ArcSec[Interval[{1/2, 2}]]", "ArcSec[Interval[{1/2, 2}]]");
    // Regression: a complex endpoint image (`ArcCos[2]`) used to slip in.
    check("ArcCos[Interval[{1/2, 2}]]", "ArcCos[Interval[{1/2, 2}]]");
  }
}

mod inverse_hyperbolic_round_trip {
  use super::*;

  #[test]
  fn exact_real_argument_folds() {
    check("ArcSinh[Sinh[2]]", "2");
    check("ArcTanh[Tanh[5]]", "5");
    check("ArcTanh[Tanh[-1/3]]", "-1/3");
    check("ArcTanh[Tanh[x]]", "ArcTanh[Tanh[x]]");
  }
}

mod piecewise_monotone {
  use super::*;

  #[test]
  fn sign() {
    check("Sign[Interval[{1, 3}]]", "1");
    check("Sign[Interval[{-3, -1}, {-1/2, -1/4}]]", "-1");
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

  #[test]
  fn im_and_arg_of_real_interval() {
    check("Im[Interval[{-1, 1}]]", "0");
    check("Arg[Interval[{1, 3}]]", "Interval[{0, 0}]");
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
  fn unsorted_interval() {
    check("Between[3, Interval[{5, 2}]]", "True");
  }

  #[test]
  fn centered_interval_range() {
    check("Between[Interval[{3, 4}], CenteredInterval[5, 3]]", "True");
    check(
      "Between[CenteredInterval[4, 1], CenteredInterval[5, 3]]",
      "True",
    );
    check("Between[4, CenteredInterval[5, 2]]", "True");
    check("Between[1, CenteredInterval[5, 2]]", "False");
  }

  #[test]
  fn symbolic_gives_either_order() {
    check("Between[x, Interval[{a, b}]]", "a <= x <= b || b <= x <= a");
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

mod mixed_centered_interval {
  use super::*;

  #[test]
  fn member_q_boxes() {
    check(
      "IntervalMemberQ[CenteredInterval[2 + 3I, 1 + I], E + Pi I]",
      "True",
    );
    check(
      "IntervalMemberQ[CenteredInterval[0, 1 + I], \
       CenteredInterval[1/2, 1/4 + I/4]]",
      "True",
    );
    check(
      "IntervalMemberQ[CenteredInterval[1 + I, 2 + 3I], \
       Interval[{0, 1}, {2, 3}]]",
      "True",
    );
  }

  #[test]
  fn intersection_takes_real_segment() {
    check(
      "IntervalIntersection[Interval[{-2, -1/2}, {1/2, 2}], \
       CenteredInterval[0, 1 + I]]",
      "Interval[{-1, -1/2}, {1/2, 1}]",
    );
  }

  #[test]
  fn union_with_real_centered_interval() {
    check(
      "IntervalUnion[Interval[{-2, -1/2}, {1, 2}], CenteredInterval[0, 1/2]]",
      "Interval[{-2, 1/2}, {1, 2}]",
    );
  }
}

mod doc_trig_examples {
  use super::*;

  #[test]
  fn exact_spans() {
    check("Sin[Interval[{-Pi/6, Pi/6}]]", "Interval[{-1/2, 1/2}]");
    check("SinDegrees[Interval[{-30, 30}]]", "Interval[{-1/2, 1/2}]");
    check("CosDegrees[Interval[{90, 60}]]", "Interval[{0, 1/2}]");
    check("TanDegrees[Interval[{0, 30}]]", "Interval[{0, 1/Sqrt[3]}]");
    check("SecDegrees[Interval[{30, 60}]]", "Interval[{2/Sqrt[3], 2}]");
    check("CscDegrees[Interval[{30, 60}]]", "Interval[{2/Sqrt[3], 2}]");
    check(
      "CotDegrees[Interval[{-45, 45}]]",
      "Interval[{-Infinity, -1}, {1, Infinity}]",
    );
  }
}
