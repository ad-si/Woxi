// Options, argument forms and panics left over from the reference-page
// parameter sweep (round 3).

use super::*;

mod abort_protect {
  use super::*;

  #[test]
  fn abort_waits_for_the_protected_evaluation() {
    let r = interpret_with_stdout("AbortProtect[Abort[]; Print[a]]").unwrap();
    assert_eq!(r.stdout, "a\n");
    assert_eq!(r.result, "$Aborted");
  }

  #[test]
  fn without_an_abort_the_body_value_is_returned() {
    assert_eq!(interpret("AbortProtect[1 + 2]").unwrap(), "3");
  }

  #[test]
  fn nested_check_abort_catches_the_abort() {
    assert_eq!(
      interpret("AbortProtect[CheckAbort[Abort[], caught]]").unwrap(),
      "caught"
    );
  }

  #[test]
  fn outer_check_abort_sees_the_deferred_abort() {
    let r = interpret_with_stdout(
      "CheckAbort[AbortProtect[Abort[]; Print[b]; 1], h]",
    )
    .unwrap();
    assert_eq!(r.stdout, "b\n");
    assert_eq!(r.result, "h");
  }

  #[test]
  fn nested_protects_defer_to_the_outermost() {
    let r = interpret_with_stdout(
      "AbortProtect[AbortProtect[Abort[]; Print[i]]; Print[o]]",
    )
    .unwrap();
    assert_eq!(r.stdout, "i\no\n");
    assert_eq!(r.result, "$Aborted");
  }

  #[test]
  fn abort_after_the_protect_is_not_deferred() {
    let r =
      interpret_with_stdout("AbortProtect[1]; Abort[]; Print[c]").unwrap();
    assert_eq!(r.stdout, "");
    assert_eq!(r.result, "$Aborted");
  }
}

mod gaussian_power_mod {
  use super::*;

  #[test]
  fn gaussian_base() {
    // (2 + I)^2 = 3 + 4 I ≡ I (mod 3)
    assert_eq!(interpret("PowerMod[2 + I, 2, 3]").unwrap(), "I");
    assert_eq!(interpret("PowerMod[2 + I, 0, 3]").unwrap(), "1");
  }

  #[test]
  fn agrees_with_mod_of_the_power() {
    for (a, b, m) in [
      ("2 + I", "5", "1 + 3*I"),
      ("2 + I", "100", "7 + 2*I"),
      ("3", "4", "2 + I"),
      ("-1 + 4*I", "17", "5"),
      ("3*I", "9", "4 - I"),
    ] {
      assert_eq!(
        interpret(&format!("PowerMod[{a}, {b}, {m}] == Mod[({a})^{b}, {m}]"))
          .unwrap(),
        "True",
        "PowerMod[{a}, {b}, {m}]"
      );
    }
  }

  #[test]
  fn integer_arguments_are_unchanged() {
    assert_eq!(interpret("PowerMod[2, 10, 7]").unwrap(), "2");
    assert_eq!(interpret("PowerMod[3, -1, 7]").unwrap(), "5");
  }
}

mod identity_matrix_target_structure {
  use super::*;

  #[test]
  fn dense_and_sparse() {
    assert_eq!(
      interpret("IdentityMatrix[2, TargetStructure -> \"Dense\"]").unwrap(),
      "{{1, 0}, {0, 1}}"
    );
    assert_eq!(
      interpret(
        "IdentityMatrix[3, TargetStructure -> \"Sparse\"] === \
         IdentityMatrix[3, SparseArray]"
      )
      .unwrap(),
      "True"
    );
  }
}

mod rule_form_fit_data {
  use super::*;

  #[test]
  fn linear_model_fit_rule_lists() {
    let rows = interpret(
      "LinearModelFit[{{1, 2}, {2, 3}, {3, 5}}, x, x][\"BestFitParameters\"]",
    )
    .unwrap();
    assert_eq!(
      interpret(
        "LinearModelFit[{1 -> 2, 2 -> 3, 3 -> 5}, x, x][\"BestFitParameters\"]"
      )
      .unwrap(),
      rows
    );
    assert_eq!(
      interpret(
        "LinearModelFit[{1, 2, 3} -> {2, 3, 5}, x, x][\"BestFitParameters\"]"
      )
      .unwrap(),
      rows
    );
  }

  #[test]
  fn linear_model_fit_vector_inputs() {
    assert_eq!(
      interpret(
        "LinearModelFit[{{1, 2} -> 3, {2, 1} -> 4, {3, 3} -> 8, {0, 1} -> 1}, \
         {x, y}, {x, y}][\"BestFitParameters\"] == \
         LinearModelFit[{{1, 2, 3}, {2, 1, 4}, {3, 3, 8}, {0, 1, 1}}, \
         {x, y}, {x, y}][\"BestFitParameters\"]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn nonlinear_model_fit_rule_list() {
    assert_eq!(
      interpret(
        "NonlinearModelFit[{1 -> 2.1, 2 -> 3.9, 3 -> 8.2}, a Exp[b x], \
         {a, b}, x][\"BestFitParameters\"] == \
         NonlinearModelFit[{{1, 2.1}, {2, 3.9}, {3, 8.2}}, a Exp[b x], \
         {a, b}, x][\"BestFitParameters\"]"
      )
      .unwrap(),
      "True"
    );
  }
}

mod spherical_bessel_negative_argument {
  use super::*;

  #[test]
  fn y_is_odd_for_even_order() {
    // y_n(-x) = (-1)^(n+1) y_n(x)
    assert_eq!(
      interpret("SphericalBesselY[2, -5.] == -SphericalBesselY[2, 5.]")
        .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("SphericalBesselY[1, -2.] == SphericalBesselY[1, 2.]").unwrap(),
      "True"
    );
  }

  #[test]
  fn hankel_h1() {
    assert_eq!(
      interpret("SphericalHankelH1[2, -5.]").unwrap(),
      "0.13473121008512523 - 0.16499545760110443*I"
    );
  }
}

mod big_integer_matrix_decompositions {
  use super::*;

  #[test]
  fn smith_decomposition_of_a_large_matrix() {
    assert_eq!(
      interpret(
        "SeedRandom[1]; m = RandomInteger[{-100, 100}, {8, 8}]; \
         {u, s, v} = SmithDecomposition[m]; \
         {u . m . v == s, Abs[Det[u]], Abs[Det[v]], \
         DiagonalMatrix[Diagonal[s]] == s, Times @@ Diagonal[s] == Abs[Det[m]]}"
      )
      .unwrap(),
      "{True, 1, 1, True, True}"
    );
  }

  #[test]
  fn hermite_decomposition_of_a_large_matrix() {
    assert_eq!(
      interpret(
        "SeedRandom[2]; m = RandomInteger[{-100, 100}, {7, 9}]; \
         {u, h} = HermiteDecomposition[m]; \
         {u . m == h, Abs[Det[u]], UpperTriangularize[h] == h}"
      )
      .unwrap(),
      "{True, 1, True}"
    );
  }

  #[test]
  fn small_decompositions_are_unchanged() {
    assert_eq!(
      interpret("HermiteDecomposition[{{2, 3, 5}, {7, 11, 13}}]").unwrap(),
      "{{{11, -3}, {-7, 2}}, {{1, 0, 16}, {0, 1, -9}}}"
    );
  }
}

mod big_resultants {
  use super::*;

  #[test]
  fn resultant_beyond_i128() {
    // Checked against an exact rational elimination of the Sylvester matrix.
    assert_eq!(
      interpret(
        "Resultant[Sum[(k + 3) x^k, {k, 0, 15}], \
         Sum[(2 k - 7) x^k, {k, 0, 14}], x]"
      )
      .unwrap(),
      "-3045675241403636842837082934891"
    );
  }

  #[test]
  fn discriminant_of_a_degree_12_polynomial() {
    assert_eq!(
      interpret(
        "p = Sum[(-1)^k (k + 2) x^k, {k, 0, 12}]; \
         Discriminant[p, x] == Resultant[p, D[p, x], x]/Coefficient[p, x, 12]"
      )
      .unwrap(),
      "True"
    );
  }
}
