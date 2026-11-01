// Argument forms of special functions and calculus constructs found missing by
// the reference-page parameter sweep (round 3).

use super::*;

/// `expr` evaluates to a machine number within `tol` of zero.
fn assert_near_zero(expr: &str, tol: f64) {
  let out = interpret(&format!("Abs[N[{expr}]]")).unwrap();
  let v: f64 = out
    .replace("*^", "e")
    .trim_end_matches('.')
    .parse()
    .unwrap_or_else(|_| panic!("{expr} gave {out}"));
  assert!(v < tol, "{expr} gave |.| = {out}");
}

mod associated_anger_weber {
  use super::*;

  #[test]
  fn symbolic_stays_unevaluated() {
    assert_eq!(
      interpret("AngerJ[\\[Nu], \\[Mu], x]").unwrap(),
      "AngerJ[ν, μ, x]"
    );
    assert_eq!(interpret("WeberE[n, m, x]").unwrap(), "WeberE[n, m, x]");
    assert_eq!(
      interpret("AngerJ[1/2, 1/3, 2]").unwrap(),
      "AngerJ[1/2, 1/3, 2]"
    );
  }

  #[test]
  fn zero_mu_is_the_plain_function() {
    assert_near_zero("AngerJ[2, 0, 1.5] - BesselJ[2, 1.5]", 1e-14);
    assert_near_zero("AngerJ[0.3, 0, 2.5] - AngerJ[0.3, 2.5]", 1e-12);
    assert_near_zero("WeberE[0.3, 0, 2.5] - WeberE[0.3, 2.5]", 1e-12);
  }

  #[test]
  fn mu_one_is_the_derivative_of_the_plain_function() {
    // d/dz AngerJ[ν, 0, z] = WeberE[ν, 1, z]/2 and
    // d/dz AngerJ[ν, z] = (AngerJ[ν - 1, z] - AngerJ[ν + 1, z])/2.
    assert_near_zero(
      "WeberE[0.5, 1, 1.5] - (AngerJ[-0.5, 1.5] - AngerJ[1.5, 1.5])",
      1e-12,
    );
  }

  #[test]
  fn singular_weight_at_the_endpoints() {
    // AngerJ[0, μ, 0] = 2^μ Gamma[(μ + 1)/2] / (Sqrt[Pi] Gamma[μ/2 + 1]).
    for mu in ["-0.5", "-0.9", "-0.99", "0.25", "2.5"] {
      assert_near_zero(
        &format!(
          "AngerJ[0, {mu}, 0] - 2^{mu} Gamma[({mu} + 1)/2]/\
           (Sqrt[Pi] Gamma[{mu}/2 + 1])"
        ),
        1e-12,
      );
    }
    // The odd integrand Sin[ν θ] vanishes for ν = 0 at z = 0.
    assert_near_zero("WeberE[0, -0.5, 0.]", 1e-14);
  }

  #[test]
  fn listable_in_every_argument() {
    assert_eq!(
      interpret("Length[AngerJ[0.5, {0, 1, 2}, 1.]]").unwrap(),
      "3"
    );
  }
}

mod circle_through_center {
  use super::*;

  #[test]
  fn given_center_and_radius() {
    assert_eq!(
      interpret(
        "CircleThrough[{{0, 0}, {1, 0}, {0, 1}}, {1/2, 1/2}, 1/Sqrt[2]]"
      )
      .unwrap(),
      "Circle[{1/2, 1/2}, 1/Sqrt[2]]"
    );
  }

  #[test]
  fn given_center_only() {
    assert_eq!(
      interpret("CircleThrough[{{0, 0}, {1, 0}, {0, 1}}, {1/2, 1/2}]").unwrap(),
      "Circle[{1/2, 1/2}, 1/Sqrt[2]]"
    );
    assert_eq!(
      interpret("CircleThrough[{{3, 4}}, {0, 0}]").unwrap(),
      "Circle[{0, 0}, 5]"
    );
  }
}
