// Argument forms and options found missing by the reference-page
// parameter sweep (every documented example run through Woxi).

use super::*;

mod head_wrapper {
  use super::*;

  #[test]
  fn head_with_wrapper() {
    assert_eq!(interpret("Head[f[x], g]").unwrap(), "g[f]");
    assert_eq!(
      interpret("Head[Unevaluated[InverseFunction[Sin][1]], Hold]").unwrap(),
      "Hold[InverseFunction[Sin]]"
    );
  }
}

mod thread_positions {
  use super::*;

  const E: &str = "f[{a, b}, {r, s}, {u, v}, {x, y}]";

  #[test]
  fn all_and_none() {
    assert_eq!(
      interpret(&format!("Thread[{E}, List, All]")).unwrap(),
      "{f[a, r, u, x], f[b, s, v, y]}"
    );
    assert_eq!(
      interpret(&format!("Thread[{E}, List, None]")).unwrap(),
      "f[{a, b}, {r, s}, {u, v}, {x, y}]"
    );
  }

  #[test]
  fn negative_count_threads_over_the_last_positions() {
    assert_eq!(
      interpret(&format!("Thread[{E}, List, -2]")).unwrap(),
      "{f[{a, b}, {r, s}, u, x], f[{a, b}, {r, s}, v, y]}"
    );
  }

  #[test]
  fn range_with_step() {
    assert_eq!(
      interpret(&format!("Thread[{E}, List, {{1, -1, 2}}]")).unwrap(),
      "{f[a, {r, s}, u, {x, y}], f[b, {r, s}, v, {x, y}]}"
    );
  }

  #[test]
  fn positions_beyond_the_arguments() {
    let r =
      interpret_with_stdout("Thread[f[{a, b}, {r, s}], List, 3]").unwrap();
    assert_eq!(r.result, "Thread[f[{a, b}, {r, s}], List, 3]");
    assert_eq!(
      r.warnings,
      vec![
        "Thread::tpos: Cannot thread over positions 1 through 3 in f[{a, b}, {r, s}]."
      ]
    );
  }
}

mod partition_upto_offset {
  use super::*;

  #[test]
  fn upto_with_offset() {
    assert_eq!(
      interpret("Partition[{a, b, c, d, e, f, g}, UpTo[3], 1]").unwrap(),
      "{{a, b, c}, {b, c, d}, {c, d, e}, {d, e, f}, {e, f, g}, {f, g}, {g}}"
    );
    assert_eq!(
      interpret("Partition[{a, b, c, d, e, f, g}, UpTo[2], 3]").unwrap(),
      "{{a, b}, {d, e}, {g}}"
    );
  }
}

mod heads_option_with_levels {
  use super::*;

  #[test]
  fn map_with_levelspec_and_heads() {
    assert_eq!(
      interpret("Map[f, {{{{a}}}}, 2, Heads -> True]").unwrap(),
      "f[List][f[f[List][f[{{a}}]]]]"
    );
    assert_eq!(
      interpret("Map[f, {a, {b}}, {2}, Heads -> True]").unwrap(),
      "{a, f[List][f[b]]}"
    );
    assert_eq!(
      interpret("Map[f, {a, {b}}, 2, Heads -> False]").unwrap(),
      "{f[a], f[{f[b]}]}"
    );
  }

  #[test]
  fn apply_with_levelspec_and_heads() {
    assert_eq!(
      interpret("Apply[f, p[x][q[y]], {1}, Heads -> True]").unwrap(),
      "f[x][f[y]]"
    );
    assert_eq!(
      interpret("Apply[f, p[x][q[y]], {1}, Heads -> False]").unwrap(),
      "p[x][f[y]]"
    );
    assert_eq!(
      interpret("Apply[f, p[x][q[y][r[z]]], {0, 1}, Heads -> True]").unwrap(),
      "f[f[r[z]]]"
    );
  }
}

mod do_iterators {
  use super::*;

  #[test]
  fn symbolic_bounds_with_a_numeric_span() {
    let r = interpret_with_stdout("Do[Print[n], {n, x, x + 3y, y}]").unwrap();
    assert_eq!(r.stdout, "x\nx + y\nx + 2*y\nx + 3*y\n");
    let r = interpret_with_stdout("Do[Print[n], {n, x, x + 2}]").unwrap();
    assert_eq!(r.stdout, "x\n1 + x\n2 + x\n");
  }

  #[test]
  fn infinite_upper_bound_runs_until_return() {
    assert_eq!(
      interpret("Do[If[k^2 > 100, Return[k]], {k, Infinity}]").unwrap(),
      "11"
    );
    assert_eq!(
      interpret("Do[If[k > 3, Return[k]], {k, 1/2, Infinity, 1/2}]").unwrap(),
      "7/2"
    );
  }
}

mod fold_operator_form {
  use super::*;

  #[test]
  fn with_initial_value() {
    assert_eq!(
      interpret("Fold[f][x, {a, b, c, d}]").unwrap(),
      "f[f[f[f[x, a], b], c], d]"
    );
    assert_eq!(
      interpret("FoldList[f][x, {a, b}]").unwrap(),
      "{x, f[x, a], f[f[x, a], b]}"
    );
  }
}

mod from_digits_roman {
  use super::*;

  #[test]
  fn roman_numerals() {
    assert_eq!(interpret("FromDigits[\"XVII\", \"Roman\"]").unwrap(), "17");
    assert_eq!(
      interpret("FromDigits[\"MCMXCIV\", \"Roman\"]").unwrap(),
      "1994"
    );
  }
}

mod matrix_power_vector {
  use super::*;

  #[test]
  fn applied_to_a_vector() {
    assert_eq!(
      interpret("MatrixPower[{{1, 1}, {1, 0}}, 4, {1, 0}]").unwrap(),
      "{5, 3}"
    );
    assert_eq!(
      interpret("MatrixPower[{{a, b}, {c, d}}, n, {1, 0}]").unwrap(),
      "MatrixPower[{{a, b}, {c, d}}, n, {1, 0}]"
    );
  }
}

mod infinite_arguments {
  use super::*;

  #[test]
  fn inverse_trig_at_infinity() {
    assert_eq!(
      interpret(
        "{ArcSin[Infinity], ArcSin[-Infinity], ArcCos[Infinity], \
         ArcCos[-Infinity], ArcSin[ComplexInfinity], ArcCos[ComplexInfinity]}"
      )
      .unwrap(),
      "{DirectedInfinity[-I], DirectedInfinity[I], DirectedInfinity[I], \
       DirectedInfinity[-I], ComplexInfinity, ComplexInfinity}"
    );
    assert_eq!(
      interpret(
        "{ArcSec[Infinity], ArcSec[ComplexInfinity], ArcCot[ComplexInfinity], \
         ArcCoth[ComplexInfinity], ArcCsch[ComplexInfinity]}"
      )
      .unwrap(),
      "{Pi/2, Pi/2, 0, 0, 0}"
    );
    assert_eq!(
      interpret("{ArcTanh[Infinity], ArcTanh[-Infinity]}").unwrap(),
      "{(-1/2*I)*Pi, I/2*Pi}"
    );
  }

  #[test]
  fn complex_components_at_infinity() {
    assert_eq!(
      interpret(
        "{Arg[Infinity], Arg[-Infinity], Re[ComplexInfinity], \
         Im[ComplexInfinity], Conjugate[ComplexInfinity]}"
      )
      .unwrap(),
      "{0, Pi, Indeterminate, Indeterminate, ComplexInfinity}"
    );
  }

  #[test]
  fn two_argument_arctan_with_an_infinite_coordinate() {
    assert_eq!(
      interpret(
        "{ArcTan[Infinity, 1], ArcTan[-1, Infinity], ArcTan[-Infinity, 2], \
         ArcTan[-Infinity, -2], ArcTan[3, -Infinity]}"
      )
      .unwrap(),
      "{0, Pi/2, Pi, -Pi, -1/2*Pi}"
    );
    assert_eq!(
      interpret("ArcTan[Infinity, y]").unwrap(),
      "ArcTan[Infinity, y]"
    );
  }
}

mod kronecker_delta_numeric {
  use super::*;

  #[test]
  fn nonzero_numeric_quantities() {
    assert_eq!(
      interpret(
        "{KroneckerDelta[I], KroneckerDelta[Pi], KroneckerDelta[1 + I], \
         KroneckerDelta[0], KroneckerDelta[x]}"
      )
      .unwrap(),
      "{0, 0, 0, 1, KroneckerDelta[x]}"
    );
  }
}

mod modulus_option {
  use super::*;

  #[test]
  fn coefficient_modulo() {
    assert_eq!(
      interpret("Coefficient[(x + 1)^3, x, 2, Modulus -> 2]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Coefficient[(x + 1)^3, x, 1, Modulus -> 3]").unwrap(),
      "0"
    );
  }

  #[test]
  fn coefficient_list_modulo() {
    assert_eq!(
      interpret("CoefficientList[(x + 1)^5, x, Modulus -> 2]").unwrap(),
      "{1, 1, 0, 0, 1, 1}"
    );
    // A leading coefficient that vanishes modulo m drops out.
    assert_eq!(
      interpret("CoefficientList[(2 x + 1)^2, x, Modulus -> 4]").unwrap(),
      "{1}"
    );
  }
}

mod target_structure {
  use super::*;

  #[test]
  fn diagonal_matrix_dense_and_sparse() {
    assert_eq!(
      interpret("DiagonalMatrix[{1, 2, 3}, TargetStructure -> \"Dense\"]")
        .unwrap(),
      "{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}"
    );
    assert_eq!(
      interpret(
        "Normal[DiagonalMatrix[{1, 2}, 1, TargetStructure -> \"Sparse\"]]"
      )
      .unwrap(),
      "{{0, 1, 0}, {0, 0, 2}, {0, 0, 0}}"
    );
    assert_eq!(
      interpret("Head[DiagonalMatrix[{1, 2}, TargetStructure -> \"Sparse\"]]")
        .unwrap(),
      "SparseArray"
    );
  }
}

mod random_big_integers {
  use super::*;

  #[test]
  fn bounds_beyond_machine_integers() {
    assert_eq!(
      interpret(
        "SeedRandom[3]; r = RandomInteger[10^50]; \
         {IntegerQ[r], 0 <= r <= 10^50}"
      )
      .unwrap(),
      "{True, True}"
    );
    assert_eq!(
      interpret(
        "SeedRandom[3]; m = RandomInteger[{-10^40, 10^40}, {2, 3}]; \
         {Dimensions[m], And @@ Flatten[Map[-10^40 <= # <= 10^40 &, m, {2}]]}"
      )
      .unwrap(),
      "{{2, 3}, True}"
    );
  }
}

mod random_color_templates {
  use super::*;

  #[test]
  fn blanks_are_filled() {
    assert_eq!(
      interpret(
        "c = RandomColor[RGBColor[1, _, _], 4]; \
         {Length[c], Union[c[[All, 1]]], And @@ Map[0 <= # <= 1 &, \
         Flatten[c[[All, 2 ;; 3]] /. RGBColor -> List]]}"
      )
      .unwrap(),
      "{4, {1}, True}"
    );
    assert_eq!(
      interpret("Dimensions[RandomColor[Hue[_, 1, 1], {2, 3}]]").unwrap(),
      "{2, 3}"
    );
  }
}

mod inner_general_heads {
  use super::*;

  #[test]
  fn same_head_operands() {
    assert_eq!(interpret("Inner[s, f[1], f[2], t]").unwrap(), "t[s[1, 2]]");
    assert_eq!(
      interpret("Inner[f, g[a, b], g[c, d], h]").unwrap(),
      "h[f[a, c], f[b, d]]"
    );
  }
}

mod conjugate_transpose_permutation {
  use super::*;

  #[test]
  fn with_level_permutation() {
    assert_eq!(
      interpret("ConjugateTranspose[{{{1 + I, 2}, {3, 4 I}}}, {1, 3, 2}]")
        .unwrap(),
      "{{{1 - I, 3}, {2, -4*I}}}"
    );
  }
}

mod fit_design_matrix {
  use super::*;

  #[test]
  fn least_squares_coefficients() {
    assert_eq!(
      interpret("Fit[{{{1, 1}, {1, 2}, {1, 3}}, {1, 2, 2}}]").unwrap(),
      "{2/3, 1/2}"
    );
  }
}

mod dsolve_one_argument {
  use super::*;

  #[test]
  fn unknowns_read_from_the_equations() {
    assert_eq!(
      interpret("DSolve[y'[x] == y[x]]").unwrap(),
      "{{y[x] -> E^x*C[1]}}"
    );
    assert_eq!(
      interpret("DSolve[{y'[x] == z[x], z'[x] == -y[x]}]").unwrap(),
      "{{y[x] -> C[1]*Cos[x] + C[2]*Sin[x], z[x] -> C[2]*Cos[x] - C[1]*Sin[x]}}"
    );
    assert_eq!(
      interpret("DSolveValue[y''[x] + y[x] == 0]").unwrap(),
      "C[1]*Cos[x] + C[2]*Sin[x]"
    );
  }
}

mod nsolve_domain_and_precision {
  use super::*;

  #[test]
  fn precision_after_the_domain() {
    assert_eq!(
      interpret("NSolve[x^5 - x + 2, x, Reals, 30]").unwrap(),
      interpret("NSolve[x^5 - x + 2, x, Reals]").unwrap()
    );
  }
}

mod to_boxes_form {
  use super::*;

  #[test]
  fn standard_form_argument() {
    assert_eq!(
      interpret("ToBoxes[x^2, StandardForm]").unwrap(),
      "SuperscriptBox[x, 2]"
    );
  }
}

mod date_plus_one_argument {
  use super::*;

  #[test]
  fn days_after_today() {
    assert_eq!(
      interpret("DatePlus[3] == DatePlus[Today, 3]").unwrap(),
      "True"
    );
  }
}

mod nielsen_polylog {
  use super::*;

  #[test]
  fn reductions() {
    assert_eq!(interpret("PolyLog[n, 1, z]").unwrap(), "PolyLog[1 + n, z]");
    assert_eq!(interpret("PolyLog[2, 1, z]").unwrap(), "PolyLog[3, z]");
    assert_eq!(interpret("PolyLog[0, 2, z]").unwrap(), "Log[1 - z]^2/2");
    assert_eq!(interpret("PolyLog[1, 3, z]").unwrap(), "PolyLog[1, 3, z]");
  }

  #[test]
  fn derivative() {
    assert_eq!(
      interpret("D[PolyLog[n, p, z], z]").unwrap(),
      "PolyLog[-1 + n, p, z]/z"
    );
  }
}

mod theta_constants {
  use super::*;

  #[test]
  fn two_argument_theta() {
    assert_eq!(
      interpret("EllipticTheta[3, 0.5] == EllipticTheta[3, 0, 0.5]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("EllipticTheta[a, q]").unwrap(),
      "EllipticTheta[a, q]"
    );
  }
}

mod legendre_type {
  use super::*;

  #[test]
  fn type_one_is_the_default() {
    assert_eq!(
      interpret("LegendreP[2, 1, 1, z]").unwrap(),
      interpret("LegendreP[2, 1, z]").unwrap()
    );
  }
}

mod ordering_functions_returning_order_values {
  use super::*;

  #[test]
  fn minus_one_puts_the_second_element_first() {
    assert_eq!(interpret("Sort[{3, 1, 2}, Order]").unwrap(), "{1, 2, 3}");
    assert_eq!(
      interpret("Sort[{3, 1, 2}, NumericalOrder]").unwrap(),
      "{1, 2, 3}"
    );
    assert_eq!(
      interpret("Ordering[{3, 1, 2}, All, Order]").unwrap(),
      "{2, 3, 1}"
    );
    assert_eq!(interpret("Sort[{c, a, b}, Less]").unwrap(), "{c, a, b}");
  }
}

mod multivariate_step_functions {
  use super::*;

  #[test]
  fn products_of_the_one_variable_functions() {
    assert_eq!(interpret("UnitBox[1, Pi, 5.3]").unwrap(), "0");
    assert_eq!(interpret("UnitBox[0, 1/4]").unwrap(), "1");
    assert_eq!(interpret("UnitTriangle[1/2, 1/4]").unwrap(), "3/8");
    assert_eq!(interpret("HeavisideLambda[0, 1/2]").unwrap(), "1/2");
    assert_eq!(interpret("UnitBox[x, 0]").unwrap(), "UnitBox[x, 0]");
  }

  #[test]
  fn exact_numeric_constants() {
    assert_eq!(
      interpret(
        "{UnitBox[Pi], UnitBox[Sqrt[2]/4], UnitTriangle[Pi], \
         UnitTriangle[Pi/10], HeavisidePi[E]}"
      )
      .unwrap(),
      "{0, 1, 0, 1 - Pi/10, 0}"
    );
  }
}

mod divisor_sigma_general_order {
  use super::*;

  #[test]
  fn symbolic_and_rational_orders() {
    assert_eq!(
      interpret("DivisorSigma[k, 30]").unwrap(),
      "1 + 2^k + 3^k + 5^k + 6^k + 10^k + 15^k + 30^k"
    );
    assert_eq!(
      interpret("DivisorSigma[1/2, 12]").unwrap(),
      "3 + Sqrt[2] + 3*Sqrt[3] + Sqrt[6]"
    );
    assert_eq!(
      interpret("DivisorSigma[k, n]").unwrap(),
      "DivisorSigma[k, n]"
    );
  }
}

mod operator_forms {
  use super::*;

  #[test]
  fn key_exists_q() {
    assert_eq!(
      interpret("KeyExistsQ[a][<|a -> x, b -> y|>]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("KeyExistsQ[\"a\"] /@ {\"a\" -> 1, \"b\" -> 2}").unwrap(),
      "{True, False}"
    );
  }

  #[test]
  fn replace_list() {
    assert_eq!(
      interpret("ReplaceList[{x__, y__} -> {{x}, {y}}][{a, b, c}]").unwrap(),
      "{{{a}, {b, c}}, {{a, b}, {c}}}"
    );
  }

  #[test]
  fn string_replace_part() {
    assert_eq!(
      interpret("StringReplacePart[\"ABCDEFGH\", {2, 5}][\"abcdefghijk\"]")
        .unwrap(),
      "aABCDEFGHfghijk"
    );
  }
}

mod extremal_by_with_ordering_function {
  use super::*;

  #[test]
  fn ordering_function_as_fourth_argument() {
    assert_eq!(
      interpret("MaximalBy[{3, -5, 2}, Abs, 2, Less]").unwrap(),
      "{-5, 3}"
    );
    assert_eq!(
      interpret(
        "MinimalBy[{DateObject[{2024, 9, 12}], DateObject[{2022, 9, 12}], \
         DateObject[{2023, 9, 12}]}, Identity, 1, NumericalOrder]"
      )
      .unwrap(),
      "{DateObject[{2022, 9, 12}, Day]}"
    );
    assert_eq!(
      interpret("TakeSmallestBy[{5, 1, 3}, Identity, 2, Order]").unwrap(),
      "{1, 3}"
    );
    assert_eq!(
      interpret("TakeLargestBy[{5, 1, 3}, Identity, 2, Order]").unwrap(),
      "{5, 3}"
    );
  }
}

mod inverse_erf_two_arguments {
  use super::*;

  #[test]
  fn generalized_inverse() {
    assert_eq!(
      interpret("InverseErf[0.4, 0.2] - InverseErf[0.2 + Erf[0.4]]").unwrap(),
      "0."
    );
    assert_eq!(interpret("InverseErf[a, Erf[a, z]]").unwrap(), "z");
    assert_eq!(interpret("InverseErf[0, s]").unwrap(), "InverseErf[s]");
  }
}

mod square_free_q_variable {
  use super::*;

  #[test]
  fn with_respect_to_a_variable() {
    assert_eq!(interpret("SquareFreeQ[x y^2, x]").unwrap(), "True");
    assert_eq!(interpret("SquareFreeQ[x^2 y, x]").unwrap(), "False");
    assert_eq!(
      interpret("SquareFreeQ[(x + 1)^2 (x - 1), x]").unwrap(),
      "False"
    );
  }
}

mod power_range_one_argument {
  use super::*;

  #[test]
  fn powers_of_ten() {
    assert_eq!(
      interpret("PowerRange[10^4]").unwrap(),
      "{1, 10, 100, 1000, 10000}"
    );
  }
}

mod parent_directory_levels {
  use super::*;

  #[test]
  fn several_levels_up() {
    assert_eq!(
      interpret("ParentDirectory[\"/a/b/c/d\", 2]").unwrap(),
      "/a/b"
    );
    assert_eq!(
      interpret("ParentDirectory[File[\"/a/b/c\"], 2]").unwrap(),
      "File[/a]"
    );
  }
}

mod uncompress_wrapper {
  use super::*;

  #[test]
  fn wrapped_before_evaluation() {
    assert_eq!(
      interpret("Uncompress[Compress[Unevaluated[1 + 1]], Hold]").unwrap(),
      "Hold[1 + 1]"
    );
    assert_eq!(
      interpret("Uncompress[Compress[Unevaluated[1 + 1]]]").unwrap(),
      "2"
    );
  }
}

mod group_support_and_orbits {
  use super::*;

  #[test]
  fn support_of_the_generators() {
    assert_eq!(
      interpret(
        "GroupSupport[PermutationGroup[{Cycles[{{1, 4}}], Cycles[{{2, 3, 7}}]}]]"
      )
      .unwrap(),
      "{1, 2, 3, 4, 7}"
    );
    assert_eq!(
      interpret(
        "GroupOrbits[PermutationGroup[{Cycles[{{1, 4}}], Cycles[{{2, 3, 7}}]}]]"
      )
      .unwrap(),
      "{{1, 4}, {2, 3, 7}}"
    );
  }
}
