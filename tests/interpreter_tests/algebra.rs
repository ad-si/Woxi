use super::*;

mod polynomial_q {
  use super::*;

  #[test]
  fn basic_polynomial() {
    assert_eq!(interpret("PolynomialQ[x^2 + 1, x]").unwrap(), "True");
    assert_eq!(
      interpret("PolynomialQ[3*x^3 + 2*x + 1, x]").unwrap(),
      "True"
    );
  }

  #[test]
  fn constant_is_polynomial() {
    assert_eq!(interpret("PolynomialQ[5, x]").unwrap(), "True");
  }

  #[test]
  fn variable_is_polynomial() {
    assert_eq!(interpret("PolynomialQ[x, x]").unwrap(), "True");
  }

  #[test]
  fn non_polynomial() {
    assert_eq!(interpret("PolynomialQ[Sin[x], x]").unwrap(), "False");
    assert_eq!(interpret("PolynomialQ[1/x, x]").unwrap(), "False");
  }

  #[test]
  fn multivariate() {
    assert_eq!(interpret("PolynomialQ[x^2 + y, x]").unwrap(), "True");
  }

  #[test]
  fn multivariate_list_of_vars() {
    // PolynomialQ with a list of variables
    assert_eq!(interpret("PolynomialQ[x + y^2, {x, y}]").unwrap(), "True");
    assert_eq!(
      interpret("PolynomialQ[x^2 + 2*x*y + y^2, {x, y}]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("PolynomialQ[Sin[x] + y, {x, y}]").unwrap(),
      "False"
    );
  }

  #[test]
  fn subexpression_as_variable() {
    // f[a] treated as an atomic variable — f[a] + f[a]^2 is a polynomial in f[a]
    assert_eq!(
      interpret("PolynomialQ[f[a] + f[a]^2, f[a]]").unwrap(),
      "True"
    );
    // Subexpression inside a list of variables
    assert_eq!(
      interpret("PolynomialQ[f[a] + g[b]^2, {f[a], g[b]}]").unwrap(),
      "True"
    );
    // Not a polynomial: 1/f[a] has negative power of f[a]
    assert_eq!(interpret("PolynomialQ[1/f[a], f[a]]").unwrap(), "False");
  }
}

mod exponent {
  use super::*;

  #[test]
  fn basic_exponent() {
    assert_eq!(interpret("Exponent[x^3 + x, x]").unwrap(), "3");
    assert_eq!(interpret("Exponent[x^2 + 3*x + 2, x]").unwrap(), "2");
  }

  // A bare number in the form slot is not a variable that ever appears, so
  // Exponent is 0 — it must not raise an error. (Integers, reals, rationals,
  // and 0 all behave the same.)
  #[test]
  fn number_form_gives_zero() {
    assert_eq!(interpret("Exponent[x^2, 5]").unwrap(), "0");
    assert_eq!(interpret("Exponent[x^2 + 3 x, 5]").unwrap(), "0");
    assert_eq!(interpret("Exponent[x^2, 2.5]").unwrap(), "0");
    assert_eq!(interpret("Exponent[x^2, 1/2]").unwrap(), "0");
    assert_eq!(interpret("Exponent[x^2, 0]").unwrap(), "0");
  }

  // A compound form (a sum, a product, or a function application) is treated
  // as an atomic unit: Exponent reports the highest power of that whole
  // subexpression, without expanding it.
  #[test]
  fn compound_form_as_unit() {
    assert_eq!(interpret("Exponent[(x + 1)^2, x + 1]").unwrap(), "2");
    assert_eq!(interpret("Exponent[(x + 1)^2 y, x + 1]").unwrap(), "2");
    assert_eq!(interpret("Exponent[(x + 1)^2 + 1, x + 1]").unwrap(), "2");
    assert_eq!(interpret("Exponent[Sin[x]^3, Sin[x]]").unwrap(), "3");
    assert_eq!(
      interpret("Exponent[Sin[x]^2 + Sin[x], Sin[x]]").unwrap(),
      "2"
    );
    // A form that never appears gives 0.
    assert_eq!(interpret("Exponent[x^2 y, 2 x]").unwrap(), "0");
  }

  #[test]
  fn bigint_coefficient_is_constant() {
    // Regression: is_constant_wrt did not recognize BigInteger, so a term with
    // a coefficient beyond i128 was treated as variable-dependent. Exponent
    // returned the coefficient (or stayed unevaluated) instead of the degree.
    assert_eq!(
      interpret(
        "Exponent[100000000000000000000000000000000000000000 x^2 + 5 x + 7, x]"
      )
      .unwrap(),
      "2"
    );
    assert_eq!(
      interpret("Exponent[100000000000000000000000000000000000000000, x]")
        .unwrap(),
      "0"
    );
  }

  // Exponent reduces a SeriesData to its polynomial first, so it reports the
  // series truncation degree (and Min reports the lowest order).
  #[test]
  fn from_series_data() {
    assert_eq!(
      interpret("Exponent[Series[Exp[x], {x, 0, 5}], x]").unwrap(),
      "5"
    );
    assert_eq!(
      interpret("Exponent[Series[Sin[x], {x, 0, 7}], x]").unwrap(),
      "7"
    );
    // Sin starts at x^1, so Min gives 1.
    assert_eq!(
      interpret("Exponent[Series[Sin[x], {x, 0, 7}], x, Min]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Exponent[SeriesData[x, 0, {1, 2, 3}, 0, 3, 1], x]").unwrap(),
      "2"
    );
  }

  #[test]
  fn constant_exponent() {
    assert_eq!(interpret("Exponent[5, x]").unwrap(), "0");
  }

  #[test]
  fn linear_exponent() {
    assert_eq!(interpret("Exponent[3*x + 1, x]").unwrap(), "1");
  }

  #[test]
  fn rational_exponent() {
    assert_eq!(interpret("Exponent[b*x^(3/2), x]").unwrap(), "3/2");
    assert_eq!(interpret("Exponent[x^(1/2) + x^(5/2), x]").unwrap(), "5/2");
    assert_eq!(interpret("Exponent[x^(1/3) + x^(2/3), x]").unwrap(), "2/3");
  }

  #[test]
  fn rational_exponent_min() {
    assert_eq!(
      interpret("Exponent[x^(1/2) + x^(5/2), x, Min]").unwrap(),
      "1/2"
    );
  }

  #[test]
  fn exponent_list() {
    assert_eq!(interpret("Exponent[-4x, x, List]").unwrap(), "{1}");
    assert_eq!(
      interpret("Exponent[x^3 + 2x^2 - 5x + 1, x, List]").unwrap(),
      "{0, 1, 2, 3}"
    );
    assert_eq!(interpret("Exponent[x^2 + 1, x, List]").unwrap(), "{0, 2}");
    assert_eq!(interpret("Exponent[5, x, List]").unwrap(), "{0}");
  }

  #[test]
  fn exponent_list_in_map() {
    assert_eq!(
      interpret(
        "u = -4x; Map[Function[Coefficient[u,x,#]*x^#], Exponent[u,x,List]]"
      )
      .unwrap(),
      "{-4*x}"
    );
  }

  // Symbolic exponents should produce a Max[...] expression rather than
  // staying unevaluated. Regression for mathics algebra.py:1320.
  #[test]
  fn symbolic_exponent_yields_max() {
    assert_eq!(
      interpret("Exponent[x^(n + 1) + Sqrt[x] + 1, x]").unwrap(),
      "Max[1/2, 1 + n]"
    );
    assert_eq!(interpret("Exponent[x^n + x^2, x]").unwrap(), "Max[2, n]");
  }
}

mod coefficient {
  use super::*;

  #[test]
  fn quadratic_coefficients() {
    assert_eq!(interpret("Coefficient[x^2 + 3*x + 2, x, 2]").unwrap(), "1");
    assert_eq!(interpret("Coefficient[x^2 + 3*x + 2, x, 1]").unwrap(), "3");
    assert_eq!(interpret("Coefficient[x^2 + 3*x + 2, x, 0]").unwrap(), "2");
  }

  // A bare number in the form slot is not a valid variable: emit
  // Coefficient::ivar and stay unevaluated, rather than treating the number
  // as an absent monomial and returning 0.
  #[test]
  fn number_form_emits_ivar() {
    for (input, call, bad) in [
      ("Coefficient[x^2, 5]", "Coefficient[x^2, 5]", "5"),
      ("Coefficient[x^2, 2.5]", "Coefficient[x^2, 2.5]", "2.5"),
      ("Coefficient[x^2, 0]", "Coefficient[x^2, 0]", "0"),
      ("Coefficient[x^2, 5, 2]", "Coefficient[x^2, 5, 2]", "5"),
    ] {
      clear_state();
      assert_eq!(interpret(input).unwrap(), call, "for {input}");
      let expected =
        format!("Coefficient::ivar: {bad} is not a valid variable.");
      let msgs = woxi::get_captured_messages_raw();
      assert!(
        msgs.iter().any(|m| m.contains(&expected)),
        "expected {expected:?} for {input}, got {msgs:?}"
      );
    }
  }

  // A rational form is also invalid; only the unevaluated result is checked
  // here because wolframscript renders the message in 2D.
  #[test]
  fn rational_form_unevaluated() {
    assert_eq!(
      interpret("Coefficient[x^2, 1/2]").unwrap(),
      "Coefficient[x^2, 1/2]"
    );
  }

  // Symbolic monomial forms (a power, a product, a sum) remain valid.
  #[test]
  fn symbolic_form_unaffected() {
    assert_eq!(interpret("Coefficient[x^2, x^2]").unwrap(), "1");
    assert_eq!(interpret("Coefficient[3 x y, x y]").unwrap(), "3");
    assert_eq!(interpret("Coefficient[x^2, 2 y]").unwrap(), "0");
    assert_eq!(interpret("Coefficient[x^2, x + 1]").unwrap(), "0");
  }

  // CoefficientList shares the ivar rule: a bare number in the variable slot
  // emits CoefficientList::ivar and stays unevaluated.
  #[test]
  fn coefficient_list_number_form_emits_ivar() {
    for (input, call, bad) in [
      ("CoefficientList[x^2, 5]", "CoefficientList[x^2, 5]", "5"),
      ("CoefficientList[x^2, 0]", "CoefficientList[x^2, 0]", "0"),
      (
        "CoefficientList[x^2, 2.5]",
        "CoefficientList[x^2, 2.5]",
        "2.5",
      ),
    ] {
      clear_state();
      assert_eq!(interpret(input).unwrap(), call, "for {input}");
      let expected =
        format!("CoefficientList::ivar: {bad} is not a valid variable.");
      let msgs = woxi::get_captured_messages_raw();
      assert!(
        msgs.iter().any(|m| m.contains(&expected)),
        "expected {expected:?} for {input}, got {msgs:?}"
      );
    }
  }

  // A rational is also invalid; only the result is checked because
  // wolframscript renders the message in 2D.
  #[test]
  fn coefficient_list_rational_form_unevaluated() {
    assert_eq!(
      interpret("CoefficientList[x^2, 1/2]").unwrap(),
      "CoefficientList[x^2, 1/2]"
    );
  }

  // A genuine variable is unaffected.
  #[test]
  fn coefficient_list_symbol_unaffected() {
    assert_eq!(
      interpret("CoefficientList[1 + 2 x + 3 x^2, x]").unwrap(),
      "{1, 2, 3}"
    );
  }

  // Coefficient extracts from a SeriesData by reducing it to its polynomial
  // first (Normal): the x^k coefficient, or 0 past the truncation order.
  #[test]
  fn from_series_data() {
    assert_eq!(
      interpret("Coefficient[Series[Exp[x], {x, 0, 5}], x, 3]").unwrap(),
      "1/6"
    );
    assert_eq!(
      interpret("Coefficient[Series[Sin[x], {x, 0, 7}], x, 5]").unwrap(),
      "1/120"
    );
    assert_eq!(
      interpret("Coefficient[Series[1/(1 - x), {x, 0, 5}], x, 0]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret(
        "Coefficient[SeriesData[x, 0, {1, 1, 1/2, 1/6}, 0, 4, 1], x, 2]"
      )
      .unwrap(),
      "1/2"
    );
    // Past the truncation order the coefficient is 0.
    assert_eq!(
      interpret("Coefficient[Series[Exp[x], {x, 0, 5}], x, 10]").unwrap(),
      "0"
    );
  }

  #[test]
  fn default_power_is_one() {
    assert_eq!(interpret("Coefficient[x^2 + 3*x + 2, x]").unwrap(), "3");
  }

  #[test]
  fn symbolic_coefficients() {
    assert_eq!(
      interpret("Coefficient[a*x^2 + b*x + c, x, 2]").unwrap(),
      "a"
    );
    assert_eq!(
      interpret("Coefficient[a*x^2 + b*x + c, x, 1]").unwrap(),
      "b"
    );
    assert_eq!(
      interpret("Coefficient[a*x^2 + b*x + c, x, 0]").unwrap(),
      "c"
    );
  }

  #[test]
  fn zero_coefficient() {
    assert_eq!(interpret("Coefficient[x^2 + 1, x, 1]").unwrap(), "0");
  }

  #[test]
  fn bigint_coefficient_extraction() {
    // Regression: Coefficient returned 0 (or a mis-attributed power) when the
    // coefficient exceeded i128, because term_var_power_and_coeff did not
    // recognize BigInteger as a constant factor.
    assert_eq!(
      interpret(
        "Coefficient[88888888888888888888888888888888888888888 x^3 + 5 x^2, x, 3]"
      )
      .unwrap(),
      "88888888888888888888888888888888888888888"
    );
    // The central binomial coefficient of (1 + x)^200 is a 60-digit BigInteger.
    assert_eq!(
      interpret("Coefficient[Expand[(1 + x)^200], x, 100]").unwrap(),
      interpret("Binomial[200, 100]").unwrap()
    );
  }

  #[test]
  fn monomial_second_argument() {
    // Coefficient[expr, x^n] should mean "coefficient of x^n" — the
    // same result as Coefficient[expr, x, n].
    assert_eq!(interpret("Coefficient[(x + 1)^5, x^3]").unwrap(), "10");
    assert_eq!(interpret("Coefficient[3 x^2 + 5 x, x^2]").unwrap(), "3");
    assert_eq!(interpret("Coefficient[(x + y)^4, x^2]").unwrap(), "6*y^2");
  }

  #[test]
  fn multivariate_monomial_second_argument() {
    // Coefficient[expr, x^a * y^b] extracts the coefficient of that
    // exact monomial across multiple variables.
    assert_eq!(
      interpret("Coefficient[(x + y)^4, (x^2) * (y^2)]").unwrap(),
      "6"
    );
    assert_eq!(
      interpret("Coefficient[(x + 3 y)^5, x * y^4]").unwrap(),
      "405"
    );
  }

  #[test]
  fn non_polynomial_factor() {
    // x*Cos[x+3] is linear in x; Cos[x+3] is the coefficient
    assert_eq!(
      interpret("Coefficient[x*Cos[x + 3] + 6*y, x]").unwrap(),
      "Cos[3 + x]"
    );
  }

  // `Coefficient[expr, var, 0]` where expr doesn't mention var should
  // return expr unchanged — wolframscript preserves the user's form, so we
  // skip the pre-expand step. Regression for mathics algebra.py:1316.
  #[test]
  fn degree_zero_in_unmentioned_variable_preserves_form() {
    assert_eq!(
      interpret("Coefficient[(x + 2)^3 + (x + 3)^2, y, 0]").unwrap(),
      "(2 + x)^3 + (3 + x)^2"
    );
  }

  #[test]
  fn rational_expression_coefficient() {
    // Coefficient[(x+2)/(y-3) + (x+3)/(y-2), x] extracts coefficients of x.
    // Matches wolframscript.
    assert_eq!(
      interpret("Coefficient[(x + 2)/(y - 3) + (x + 3)/(y - 2), x]").unwrap(),
      "(-3 + y)^(-1) + (-2 + y)^(-1)"
    );
  }

  // Wolfram's `Coefficient[expr, form]` does *literal* factor-multiset
  // matching on `form`. The form's numeric coefficient is just another
  // factor that has to appear as-is in a term — `6*x` does not contain
  // `2*x` as a literal factor (6 ≠ 2), so the coefficient is 0, not 3.
  #[test]
  fn composite_form_with_numeric_factor() {
    assert_eq!(interpret("Coefficient[6*x, 2*x]").unwrap(), "0");
    assert_eq!(interpret("Coefficient[2*x, 2*x]").unwrap(), "1");
    assert_eq!(interpret("Coefficient[a*(2*x), 2*x]").unwrap(), "a");
    assert_eq!(interpret("Coefficient[2*x*y + 4*x, 2*x]").unwrap(), "y");
    assert_eq!(interpret("Coefficient[2*x*y + 2*x, 2*x]").unwrap(), "1 + y");
    assert_eq!(interpret("Coefficient[6*x + 2*x*y, 2*x]").unwrap(), "y");
  }

  // Sign matters in literal factor matching: `-2` and `2` are distinct
  // factors, so `-2 x` does not contribute to `Coefficient[..., 2 x]`.
  #[test]
  fn composite_form_negative_numeric() {
    assert_eq!(interpret("Coefficient[-2*x*y + 2*x, 2*x]").unwrap(), "1");
    assert_eq!(interpret("Coefficient[2*x*y - 2*x, 2*x]").unwrap(), "y");
    assert_eq!(interpret("Coefficient[-2*x*y - 2*x, 2*x]").unwrap(), "0");
  }

  // 3-arg form with composite second argument: for n = 1 it reduces to
  // the 2-arg form. Matches wolframscript.
  #[test]
  fn composite_form_three_arg_power_one() {
    assert_eq!(interpret("Coefficient[2*a*x + b*y, 2*x, 1]").unwrap(), "a");
  }
}

mod expand {
  use super::*;

  #[test]
  fn simple_product() {
    assert_eq!(
      interpret("Expand[(x + 1)*(x + 2)]").unwrap(),
      "2 + 3*x + x^2"
    );
  }

  #[test]
  fn square() {
    assert_eq!(interpret("Expand[(x + 1)^2]").unwrap(), "1 + 2*x + x^2");
  }

  #[test]
  fn cube() {
    assert_eq!(
      interpret("Expand[(x + 1)^3]").unwrap(),
      "1 + 3*x + 3*x^2 + x^3"
    );
  }

  #[test]
  fn distribute() {
    assert_eq!(interpret("Expand[x*(x + 1)]").unwrap(), "x + x^2");
  }

  #[test]
  fn already_expanded() {
    assert_eq!(interpret("Expand[x^2 + 3*x + 2]").unwrap(), "2 + 3*x + x^2");
  }

  // Regression: a positive unit-numerator rational coefficient renders as a
  // division (`x/4`), not `x*1/4`, even when Expand leaves the rational as the
  // trailing factor of the Times. `|n| > 1` keeps the `(n*X)/d` form.
  #[test]
  fn unit_rational_coefficient_display() {
    assert_eq!(interpret("Expand[x/4]").unwrap(), "x/4");
    assert_eq!(interpret("Expand[Pi/4]").unwrap(), "Pi/4");
    assert_eq!(interpret("Expand[Pi/3]").unwrap(), "Pi/3");
    assert_eq!(interpret("Expand[2 x/4]").unwrap(), "x/2");
    assert_eq!(interpret("Expand[3 Pi/4]").unwrap(), "(3*Pi)/4");
  }

  #[test]
  fn constant() {
    assert_eq!(interpret("Expand[5]").unwrap(), "5");
  }

  #[test]
  fn difference_of_squares() {
    assert_eq!(interpret("Expand[(x + 2)*(x - 2)]").unwrap(), "-4 + x^2");
  }

  #[test]
  fn multivariate_two_vars() {
    assert_eq!(interpret("Expand[(x + y)^2]").unwrap(), "x^2 + 2*x*y + y^2");
  }

  #[test]
  fn multivariate_four_vars() {
    assert_eq!(
      interpret("Expand[(a + b)*(c + d)]").unwrap(),
      "a*c + b*c + a*d + b*d"
    );
  }

  #[test]
  fn multivariate_with_constant() {
    assert_eq!(
      interpret("Expand[(x + y + 1)^2]").unwrap(),
      "1 + 2*x + x^2 + 2*y + 2*x*y + y^2"
    );
  }

  #[test]
  fn expand_inside_equal() {
    assert_eq!(
      interpret("Expand[(x - 1)*(x^2 + 1) == 0]").unwrap(),
      "-1 + x - x^2 + x^3 == 0"
    );
  }

  #[test]
  fn expand_inside_comparison() {
    assert_eq!(
      interpret("Expand[(a + b)^2 == (c + d)^2]").unwrap(),
      "a^2 + 2*a*b + b^2 == c^2 + 2*c*d + d^2"
    );
  }

  #[test]
  fn expand_inside_inequality() {
    assert_eq!(
      interpret("Expand[(x + 1)^2 > x]").unwrap(),
      "1 + 2*x + x^2 > x"
    );
  }

  #[test]
  fn inequality_numeric_evaluation() {
    assert_eq!(
      interpret("Inequality[1, Less, 3, Less, 5]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("Inequality[1, Less, 0, Less, 5]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("Inequality[1, Less, x, Less, 5] /. x -> 3").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("Inequality[1, Less, x, Less, 5] /. x -> 0").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("Inequality[1, LessEqual, 1, LessEqual, 5]").unwrap(),
      "True"
    );
  }
}

mod simplify {
  use super::*;

  #[test]
  fn combine_like_terms() {
    assert_eq!(interpret("Simplify[x + x]").unwrap(), "2*x");
  }

  // A standalone `c*Log[n]` (positive integers) folds to `Log[n^c]` when that
  // is simpler under wolframscript's digit-aware complexity measure. Large
  // powers (many digits) and symbolic bases/coefficients stay factored.
  #[test]
  fn folds_standalone_integer_log() {
    assert_eq!(interpret("Simplify[2*Log[2]]").unwrap(), "Log[4]");
    assert_eq!(interpret("Simplify[2*Log[3]]").unwrap(), "Log[9]");
    assert_eq!(interpret("Simplify[3*Log[3]]").unwrap(), "Log[27]");
    assert_eq!(interpret("Simplify[2*Log[100]]").unwrap(), "Log[10000]");
    // Right at the digit-count threshold for base 2: 2^13 folds, 2^14 does not.
    assert_eq!(interpret("Simplify[13*Log[2]]").unwrap(), "Log[8192]");
    assert_eq!(interpret("Simplify[14*Log[2]]").unwrap(), "14*Log[2]");
    // A large power stays factored (Log[1048576] is more complex).
    assert_eq!(interpret("Simplify[20*Log[2]]").unwrap(), "20*Log[2]");
    // 3^7 = 2187 has too many digits to be worth folding.
    assert_eq!(interpret("Simplify[7*Log[3]]").unwrap(), "7*Log[3]");
    // A symbolic base or coefficient is never folded.
    assert_eq!(interpret("Simplify[2*Log[x]]").unwrap(), "2*Log[x]");
    assert_eq!(interpret("Simplify[a*Log[2]]").unwrap(), "a*Log[2]");
    // A single Log is left as-is.
    assert_eq!(interpret("Simplify[Log[2]]").unwrap(), "Log[2]");
  }

  // Integer-base logs in a sum merge into a single Log, choosing between the
  // fully-merged `Log[prod n^c]` and the coefficient-GCD-factored `g*Log[…]`
  // forms by wolframscript's digit-aware complexity measure.
  #[test]
  fn merges_integer_logs_in_sum() {
    // Basic products and quotients.
    assert_eq!(interpret("Simplify[Log[2] + Log[3]]").unwrap(), "Log[6]");
    assert_eq!(interpret("Simplify[Log[6] - Log[2]]").unwrap(), "Log[3]");
    assert_eq!(
      interpret("Simplify[Log[2] + Log[3] + Log[5]]").unwrap(),
      "Log[30]"
    );
    // Coefficients: fully merged when the product stays small.
    assert_eq!(
      interpret("Simplify[3*Log[2] + 2*Log[3]]").unwrap(),
      "Log[72]"
    );
    assert_eq!(
      interpret("Simplify[5*Log[2] - Log[6]]").unwrap(),
      "Log[16/3]"
    );
    // Large shared coefficient: GCD-factored form wins over the huge product.
    assert_eq!(
      interpret("Simplify[20*Log[2] + 20*Log[3]]").unwrap(),
      "20*Log[6]"
    );
    assert_eq!(
      interpret("Simplify[4*Log[2] + 4*Log[3]]").unwrap(),
      "4*Log[6]"
    );
    // Right at the threshold between fully-merged and GCD-factored.
    assert_eq!(
      interpret("Simplify[3*Log[2] + 3*Log[3]]").unwrap(),
      "Log[216]"
    );
    // A negative result flips to `-Log[...]`.
    assert_eq!(interpret("Simplify[Log[2] - Log[3]]").unwrap(), "-Log[3/2]");
    assert_eq!(interpret("Simplify[-2*Log[2]]").unwrap(), "-Log[4]");
    // Logs cancel to zero.
    assert_eq!(interpret("Simplify[2*Log[2] - 2*Log[2]]").unwrap(), "0");
    // Non-log terms are kept alongside the merged Log.
    assert_eq!(interpret("Simplify[2*Log[2] + 1]").unwrap(), "1 + Log[4]");
    assert_eq!(interpret("Simplify[x + 2*Log[2]]").unwrap(), "x + Log[4]");
    // Symbolic bases/coefficients are never merged (would need a > 0 etc.).
    assert_eq!(
      interpret("Simplify[Log[a] + Log[b]]").unwrap(),
      "Log[a] + Log[b]"
    );
    assert_eq!(
      interpret("Simplify[a*Log[2] + b*Log[3]]").unwrap(),
      "a*Log[2] + b*Log[3]"
    );
  }

  // Simplify must keep the canonical radical form for a numeric base:
  // `2*Sqrt[2]` (= 2^1 * 2^(1/2)) stays factored rather than merging into the
  // less-standard `2^(3/2)`. Symbolic bases still merge (`x*Sqrt[x]`).
  #[test]
  fn keeps_numeric_radical_factored() {
    assert_eq!(interpret("Simplify[2*Sqrt[2]]").unwrap(), "2*Sqrt[2]");
    assert_eq!(interpret("Simplify[3*Sqrt[3]]").unwrap(), "3*Sqrt[3]");
    assert_eq!(interpret("Simplify[5*5^(1/2)]").unwrap(), "5*Sqrt[5]");
    assert_eq!(interpret("Simplify[2*2^(1/3)]").unwrap(), "2*2^(1/3)");
    // A bare radical power extracts its integer factor.
    assert_eq!(interpret("Simplify[2^(3/2)]").unwrap(), "2*Sqrt[2]");
    assert_eq!(interpret("Simplify[Sqrt[8]]").unwrap(), "2*Sqrt[2]");
  }

  #[test]
  fn symbolic_radical_still_merges() {
    assert_eq!(interpret("Simplify[x*Sqrt[x]]").unwrap(), "x^(3/2)");
  }

  #[test]
  fn combine_like_terms_implicit_coefficient() {
    // `x + 2 x` should collapse to `3*x` without needing Simplify.
    assert_eq!(interpret("x + 2 x").unwrap(), "3*x");
  }

  #[test]
  fn distribute_sign_over_parens() {
    // `a - (5 + a + 2 b) + 3 a q` — the negated parenthesis is distributed
    // and like terms collapse, leaving `-5 - 2*b + 3*a*q`.
    assert_eq!(
      interpret("a - (5+ a+ 2 b) + 3 a q").unwrap(),
      "-5 - 2*b + 3*a*q"
    );
  }

  #[test]
  fn pythagorean_identity_not_auto_simplified() {
    // Sin[1]^2 + Cos[1]^2 - 1 is not collapsed to 0 without Simplify —
    // the expression remains symbolic with a literal -1 (matches wolframscript).
    assert_eq!(
      interpret("1/(Sin[1]^2+Cos[1]^2-1)").unwrap(),
      "(-1 + Cos[1]^2 + Sin[1]^2)^(-1)"
    );
  }

  #[test]
  fn combine_powers() {
    assert_eq!(interpret("Simplify[x*x]").unwrap(), "x^2");
  }

  #[test]
  fn cancel_division() {
    assert_eq!(interpret("Simplify[(x^2 - 1)/(x - 1)]").unwrap(), "1 + x");
  }

  #[test]
  fn trivial() {
    assert_eq!(interpret("Simplify[5]").unwrap(), "5");
    assert_eq!(interpret("Simplify[x]").unwrap(), "x");
  }

  /// Regression test for issue #93: Simplify must handle canonical
  /// Times[Power[]] form identically to Divide form
  #[test]
  fn simplify_canonical_division_form() {
    assert_eq!(interpret("Simplify[(x^2 - 1)/(x - 1)]").unwrap(), "1 + x");
    assert_eq!(
      interpret("Simplify[(x^2 - 1) * (x - 1)^-1]").unwrap(),
      "1 + x"
    );
  }

  #[test]
  fn simplify_combines_conjugate_partial_fractions() {
    // Sum of complex-conjugate partial fractions must combine and factor the
    // resulting denominator: 1/2*(1/(1-I x)+1/(1+I x)) -> 1/(1+x^2).
    assert_eq!(
      interpret("Simplify[1/2 * (1/(1 - I x) + 1/(1 + I x))]").unwrap(),
      "(1 + x^2)^(-1)"
    );
    assert_eq!(
      interpret("Simplify[1/(1 - I x) + 1/(1 + I x)]").unwrap(),
      "2/(1 + x^2)"
    );
  }

  #[test]
  fn simplify_scalar_times_sum_of_fractions() {
    // `k * (sum of fractions)` must distribute and combine for any scalar k
    // (numeric, rational, or symbolic), not stay as an uncombined product.
    assert_eq!(
      interpret("Simplify[3 (1/(a - I x) + 1/(a + I x))]").unwrap(),
      "(6*a)/(a^2 + x^2)"
    );
    assert_eq!(
      interpret("Simplify[1/2 (1/(a - I x) + 1/(a + I x))]").unwrap(),
      "a/(a^2 + x^2)"
    );
    assert_eq!(
      interpret("Simplify[c (1/(a - I x) + 1/(a + I x))]").unwrap(),
      "(2*a*c)/(a^2 + x^2)"
    );
  }

  #[test]
  fn simplify_complex_conjugate_difference_sign() {
    // Together produces the correct `-2 I x` numerator; the Factor sign bug
    // used to flip it to `+2 I x` during Simplify.
    assert_eq!(
      interpret("Simplify[1/(1 + I x) - 1/(1 - I x)]").unwrap(),
      "((-2*I)*x)/(1 + x^2)"
    );
  }

  #[test]
  fn simplify_rational_equality_proves_true() {
    // Equation simplification must combine rational functions over a common
    // denominator, not just Expand: both sides reduce to 1/(1+x^2).
    assert_eq!(
      interpret("Simplify[1/2 * (1/(1 - I x) + 1/(1 + I x)) == 1/(1 + x*x)]")
        .unwrap(),
      "True"
    );
    // A genuinely-unequal pair stays an unevaluated comparison.
    assert_eq!(
      interpret("Simplify[1/(1 + x) == 1/(1 + x^2)]").unwrap(),
      "(1 + x)^(-1) == (1 + x^2)^(-1)"
    );
  }

  #[test]
  fn pythagorean_identity() {
    assert_eq!(interpret("Simplify[Sin[x]^2 + Cos[x]^2]").unwrap(), "1");
  }

  #[test]
  fn pythagorean_with_coefficient() {
    assert_eq!(interpret("Simplify[2*Sin[x]^2 + 2*Cos[x]^2]").unwrap(), "2");
  }

  #[test]
  fn pythagorean_with_extra_terms() {
    assert_eq!(interpret("Simplify[Sin[y]^2 + Cos[y]^2 + 1]").unwrap(), "2");
  }

  // The hyperbolic Pythagorean identity Cosh[x]^2 - Sinh[x]^2 = 1.
  #[test]
  fn hyperbolic_pythagorean_identity() {
    assert_eq!(interpret("Simplify[Cosh[x]^2 - Sinh[x]^2]").unwrap(), "1");
    assert_eq!(
      interpret("FullSimplify[Cosh[x]^2 - Sinh[x]^2]").unwrap(),
      "1"
    );
    // Sinh^2 - Cosh^2 = -1 (the Cosh coefficient wins).
    assert_eq!(interpret("Simplify[Sinh[x]^2 - Cosh[x]^2]").unwrap(), "-1");
  }

  #[test]
  fn hyperbolic_pythagorean_with_coefficient() {
    assert_eq!(
      interpret("Simplify[3 Cosh[x]^2 - 3 Sinh[x]^2]").unwrap(),
      "3"
    );
    assert_eq!(
      interpret("Simplify[2 Cosh[x]^2 - 2 Sinh[x]^2 + 5]").unwrap(),
      "7"
    );
  }

  #[test]
  fn pythagorean_induced_singularity() {
    // Simplify[1/(Sin[1]^2 + Cos[1]^2 - 1)] cancels to 1/0 → ComplexInfinity.
    // Regression for mathics test_structure.py:37 (test_numericq) ensuring
    // Simplify collapses exposed `0^(-1)` rather than leaving it raw.
    assert_eq!(
      interpret("Simplify[1/(Sin[1]^2 + Cos[1]^2 - 1)]").unwrap(),
      "ComplexInfinity"
    );
  }

  #[test]
  fn combine_like_denominator_fractions() {
    assert_eq!(interpret("Simplify[a/x + b/x]").unwrap(), "(a + b)/x");
  }

  #[test]
  fn combine_like_denominator_with_extra() {
    assert_eq!(
      interpret("Simplify[a/x + b/x + c/y]").unwrap(),
      "(a + b)/x + c/y"
    );
  }

  #[test]
  fn combine_fractions_different_denominators() {
    assert_eq!(
      interpret(
        "Simplify[k*q/(2*a^4*(1 + s)^(3/2)) + k*q*(1 + s)^(9/4)/(2*a^4)]"
      )
      .unwrap(),
      "(k*q*(1 + (1 + s)^(15/4)))/(2*a^4*(1 + s)^(3/2))"
    );
  }

  #[test]
  fn trig_polynomial_power_reduction() {
    // Simplify[D[Sin[x]^10, {x, 4}]] should use double-angle forms
    assert_eq!(
      interpret("Simplify[D[Sin[x]^10, {x, 4}]]").unwrap(),
      "10*(141 + 238*Cos[2*x] + 125*Cos[4*x])*Sin[x]^6"
    );
  }

  #[test]
  fn trig_polynomial_simple() {
    assert_eq!(
      interpret("Simplify[3*Cos[x]^2*Sin[x]^2 + Sin[x]^4]").unwrap(),
      "(2 + Cos[2*x])*Sin[x]^2"
    );
  }

  #[test]
  fn trig_polynomial_cos_dominant() {
    assert_eq!(
      interpret("Simplify[2*Cos[x]^4 - Cos[x]^2*Sin[x]^2]").unwrap(),
      "(Cos[x]^2*(1 + 3*Cos[2*x]))/2"
    );
  }

  #[test]
  fn equation_algebraically_equal() {
    // x^2 - y^2 == (x+y)(x-y) is always True
    assert_eq!(
      interpret("Simplify[x^2 - y^2 == (x + y)(x - y)]").unwrap(),
      "True"
    );
  }

  #[test]
  fn equation_expansion_equal() {
    assert_eq!(
      interpret("Simplify[(a + b)^2 == a^2 + 2 a b + b^2]").unwrap(),
      "True"
    );
  }

  #[test]
  fn equation_constant_difference_false() {
    // x^2 == x^2 + 1 is always False
    assert_eq!(interpret("Simplify[x^2 == x^2 + 1]").unwrap(), "False");
  }

  #[test]
  fn equation_symbolic_stays_unevaluated() {
    assert_eq!(interpret("Simplify[x == y]").unwrap(), "x == y");
  }
}

mod factor {
  use super::*;

  #[test]
  fn quadratic() {
    assert_eq!(
      interpret("Factor[x^2 + 3*x + 2]").unwrap(),
      "(1 + x)*(2 + x)"
    );
  }

  #[test]
  fn difference_of_squares() {
    assert_eq!(interpret("Factor[x^2 - 4]").unwrap(), "(-2 + x)*(2 + x)");
  }

  #[test]
  fn with_common_factor() {
    assert_eq!(
      interpret("Factor[2*x^2 + 6*x + 4]").unwrap(),
      "2*(1 + x)*(2 + x)"
    );
  }

  #[test]
  fn irreducible() {
    assert_eq!(interpret("Factor[x^2 + 1]").unwrap(), "1 + x^2");
  }

  // Regression: non-monic quadratics with rational (non-integer) roots must
  // factor via the rational-root theorem, ordering factors by leading
  // coefficient like wolframscript.
  #[test]
  fn non_monic_quadratic_rational_roots() {
    assert_eq!(
      interpret("Factor[6*x^2 + 11*x + 3]").unwrap(),
      "(3 + 2*x)*(1 + 3*x)"
    );
    assert_eq!(
      interpret("Factor[4*x^2 - 9]").unwrap(),
      "(-3 + 2*x)*(3 + 2*x)"
    );
    assert_eq!(
      interpret("Factor[6*x^2 - x - 2]").unwrap(),
      "(1 + 2*x)*(-2 + 3*x)"
    );
    assert_eq!(
      interpret("Factor[10*x^2 + 11*x + 3]").unwrap(),
      "(1 + 2*x)*(3 + 5*x)"
    );
    assert_eq!(
      interpret("Factor[12*x^2 + 7*x + 1]").unwrap(),
      "(1 + 3*x)*(1 + 4*x)"
    );
  }

  // Regression: a non-monic cubic that splits into three linear factors.
  #[test]
  fn non_monic_cubic_rational_roots() {
    assert_eq!(
      interpret("Factor[6*x^3 + 11*x^2 + 6*x + 1]").unwrap(),
      "(1 + x)*(1 + 2*x)*(1 + 3*x)"
    );
  }

  // Regression: numeric content on top of a non-monic factorization.
  #[test]
  fn non_monic_with_numeric_content() {
    assert_eq!(
      interpret("Factor[12*x^2 + 22*x + 6]").unwrap(),
      "2*(3 + 2*x)*(1 + 3*x)"
    );
  }

  // Regression: a leading minus over a multi-factor product is wrapped by
  // wolframscript as -((..)*(..)) and -(1 + x)^2, while a single irreducible
  // factor is returned expanded.
  #[test]
  fn negative_leading_sign_factorizations() {
    assert_eq!(
      interpret("Factor[-x^2 - 5*x - 6]").unwrap(),
      "-((2 + x)*(3 + x))"
    );
    assert_eq!(interpret("Factor[-(x + 1)^2]").unwrap(), "-(1 + x)^2");
    assert_eq!(interpret("Factor[-x - 2]").unwrap(), "-2 - x");
    assert_eq!(interpret("Factor[-x^2 - 1]").unwrap(), "-1 - x^2");
    assert_eq!(
      interpret("Factor[-6*x^2 - 11*x - 3]").unwrap(),
      "-((3 + 2*x)*(1 + 3*x))"
    );
  }

  #[test]
  fn cubic() {
    assert_eq!(
      interpret("Factor[x^3 - 1]").unwrap(),
      "(-1 + x)*(1 + x + x^2)"
    );
  }

  #[test]
  fn linear() {
    assert_eq!(interpret("Factor[2*x + 4]").unwrap(), "2*(2 + x)");
  }

  #[test]
  fn cyclotomic_x6_minus_1() {
    assert_eq!(
      interpret("Factor[x^6 - 1]").unwrap(),
      "(-1 + x)*(1 + x)*(1 - x + x^2)*(1 + x + x^2)"
    );
  }

  #[test]
  fn cyclotomic_x12_minus_1() {
    assert_eq!(
      interpret("Factor[x^12 - 1]").unwrap(),
      "(-1 + x)*(1 + x)*(1 + x^2)*(1 - x + x^2)*(1 + x + x^2)*(1 - x^2 + x^4)"
    );
  }

  #[test]
  fn cyclotomic_x100_minus_1() {
    assert_eq!(
      interpret("Factor[x^100 - 1]").unwrap(),
      "(-1 + x)*(1 + x)*(1 + x^2)*(1 - x + x^2 - x^3 + x^4)*(1 + x + x^2 + x^3 + x^4)*(1 - x^2 + x^4 - x^6 + x^8)*(1 - x^5 + x^10 - x^15 + x^20)*(1 + x^5 + x^10 + x^15 + x^20)*(1 - x^10 + x^20 - x^30 + x^40)"
    );
  }

  #[test]
  fn irreducible_x4_plus_1() {
    assert_eq!(interpret("Factor[x^4 + 1]").unwrap(), "1 + x^4");
  }

  #[test]
  fn cyclotomic_x4_minus_1() {
    assert_eq!(
      interpret("Factor[x^4 - 1]").unwrap(),
      "(-1 + x)*(1 + x)*(1 + x^2)"
    );
  }

  #[test]
  fn repeated_root_squared() {
    assert_eq!(interpret("Factor[x^2 + 2*x + 1]").unwrap(), "(1 + x)^2");
  }

  #[test]
  fn repeated_root_cubed() {
    assert_eq!(
      interpret("Factor[x^3 + 3*x^2 + 3*x + 1]").unwrap(),
      "(1 + x)^3"
    );
  }

  #[test]
  fn factor_threads_over_list() {
    // Factor is Listable — threads over list arguments.
    assert_eq!(
      interpret("Factor[{x + x^2, 2 x + 2 y + 2}]").unwrap(),
      "{x*(1 + x), 2*(1 + x + y)}"
    );
  }

  #[test]
  fn factor_threads_over_equation() {
    // Factor on an equation factors each side separately, matching
    // wolframscript. Regression for mathics algebra.py:1393.
    assert_eq!(
      interpret("x^2 - x == 0 // Factor").unwrap(),
      "(-1 + x)*x == 0"
    );
  }
}

mod factor_multivariate {
  use super::*;

  #[test]
  fn bivariate_perfect_square() {
    assert_eq!(interpret("Factor[x^2 + 2*x*y + y^2]").unwrap(), "(x + y)^2");
  }

  #[test]
  fn bivariate_difference_of_squares() {
    assert_eq!(interpret("Factor[x^2 - y^2]").unwrap(), "(x - y)*(x + y)");
  }

  #[test]
  fn common_variable_factor() {
    assert_eq!(interpret("Factor[x^2*y + x*y^2]").unwrap(), "x*y*(x + y)");
  }

  #[test]
  fn negative_monomial_keeps_sign() {
    // A single monomial has no nontrivial factorization; the multivariate
    // content-extraction used to drop the leading sign (`-2 a x` → `2 a x`).
    assert_eq!(interpret("Factor[-2 a x]").unwrap(), "-2*a*x");
    assert_eq!(interpret("Factor[-6 x y]").unwrap(), "-6*x*y");
  }

  #[test]
  fn imaginary_monomial_keeps_sign() {
    // The imaginary unit parses as a symbol, so `-2 I x` was a two-"variable"
    // monomial that hit the same sign-dropping bug, yielding `(2 I) x`.
    assert_eq!(interpret("Factor[-2 I x]").unwrap(), "(-2*I)*x");
    assert_eq!(interpret("Factor[-I x]").unwrap(), "-I*x");
    assert_eq!(interpret("Factor[-3 I x^2]").unwrap(), "(-3*I)*x^2");
  }

  #[test]
  fn laurent_product_keeps_denominator() {
    // Expanding `a (1/b + 1/c)` yields the Laurent form `a/b + a/c`, which the
    // polynomial factorer mishandled by dropping `1/(b c)` and returning the
    // wrong value `a (b + c)`. Together first, then factor num/den.
    assert_eq!(
      interpret("Factor[a (1/b + 1/c)]").unwrap(),
      "(a*(b + c))/(b*c)"
    );
    // Simplify keeps the leaf-smaller uncombined form (matching wolframscript);
    // the point is that it no longer returns the *wrong* value `a (b + c)`.
    assert_eq!(
      interpret("Simplify[a (1/b + 1/c)]").unwrap(),
      "a*(b^(-1) + c^(-1))"
    );
  }

  #[test]
  fn perfect_cube() {
    assert_eq!(
      interpret("Factor[x^3 + 3*x^2*y + 3*x*y^2 + y^3]").unwrap(),
      "(x + y)^3"
    );
  }

  #[test]
  fn target_expression() {
    assert_eq!(
      interpret("Factor[Expand[Expand[(x + y)^2 + 9(2 + x)(x + y)]^3]]")
        .unwrap(),
      "(x + y)^3*(18 + 10*x + y)^3"
    );
  }

  #[test]
  fn irreducible_sum_of_squares() {
    assert_eq!(interpret("Factor[x^2 + y^2]").unwrap(), "x^2 + y^2");
  }

  #[test]
  fn with_numeric_gcd() {
    assert_eq!(
      interpret("Factor[6*x^2 + 12*x*y + 6*y^2]").unwrap(),
      "6*(x + y)^2"
    );
  }

  // Factor's internal grouping previously used a single-variable-first
  // sort that doesn't match Wolfram's canonical Times order, so e.g.
  // `Factor[x*a == x*b + x*c]` would emit `x*(b + c)` instead of
  // `(b + c)*x`. The result now flows through `times_ast` so factors are
  // canonically ordered.
  #[test]
  fn equation_canonical_factor_order() {
    assert_eq!(
      interpret("Factor[x a == x b + x c]").unwrap(),
      "a*x == (b + c)*x"
    );
  }

  // Regression: Factor[x^10 - y^10] used to time out (Kronecker
  // substitution produced a degree-110 sparse polynomial that
  // factor_integer_poly tried every cyclotomic divisor against). The
  // homogeneous-binomial fast path emits the cyclotomic decomposition
  // directly.
  #[test]
  fn x10_minus_y10_homogeneous() {
    assert_eq!(
      interpret("Factor[x^10 - y^10]").unwrap(),
      "(x - y)*(x + y)*(x^4 - x^3*y + x^2*y^2 - x*y^3 + y^4)*(x^4 + x^3*y + x^2*y^2 + x*y^3 + y^4)"
    );
  }

  #[test]
  fn x6_minus_y6_homogeneous() {
    assert_eq!(
      interpret("Factor[x^6 - y^6]").unwrap(),
      "(x - y)*(x + y)*(x^2 - x*y + y^2)*(x^2 + x*y + y^2)"
    );
  }

  #[test]
  fn x4_minus_y4_homogeneous() {
    assert_eq!(
      interpret("Factor[x^4 - y^4]").unwrap(),
      "(x - y)*(x + y)*(x^2 + y^2)"
    );
  }

  #[test]
  fn x8_minus_y8_homogeneous() {
    assert_eq!(
      interpret("Factor[x^8 - y^8]").unwrap(),
      "(x - y)*(x + y)*(x^2 + y^2)*(x^4 + y^4)"
    );
  }

  #[test]
  fn x12_minus_y12_homogeneous() {
    assert_eq!(
      interpret("Factor[x^12 - y^12]").unwrap(),
      "(x - y)*(x + y)*(x^2 + y^2)*(x^2 - x*y + y^2)*(x^2 + x*y + y^2)*(x^4 - x^2*y^2 + y^4)"
    );
  }

  // x^n + y^n should still factor for composite n (e.g. n = 6).
  #[test]
  fn x6_plus_y6_homogeneous() {
    assert_eq!(
      interpret("Factor[x^6 + y^6]").unwrap(),
      "(x^2 + y^2)*(x^4 - x^2*y^2 + y^4)"
    );
  }
}

mod factor_list {
  use super::*;

  #[test]
  fn cubic() {
    assert_eq!(
      interpret("FactorList[x^3 - 1]").unwrap(),
      "{{1, 1}, {-1 + x, 1}, {1 + x + x^2, 1}}"
    );
  }

  #[test]
  fn with_numeric_coefficient() {
    assert_eq!(
      interpret("FactorList[2*x^2 + 4*x + 2]").unwrap(),
      "{{2, 1}, {1 + x, 2}}"
    );
  }

  #[test]
  fn irreducible() {
    assert_eq!(
      interpret("FactorList[x^2 + 1]").unwrap(),
      "{{1, 1}, {1 + x^2, 1}}"
    );
  }

  #[test]
  fn constant() {
    assert_eq!(interpret("FactorList[6]").unwrap(), "{{6, 1}}");
  }

  #[test]
  fn quartic() {
    assert_eq!(
      interpret("FactorList[x^4 - 1]").unwrap(),
      "{{1, 1}, {-1 + x, 1}, {1 + x, 1}, {1 + x^2, 1}}"
    );
  }

  #[test]
  fn repeated_root() {
    assert_eq!(
      interpret("FactorList[x^3 + 3*x^2 + 3*x + 1]").unwrap(),
      "{{1, 1}, {1 + x, 3}}"
    );
  }

  #[test]
  fn quadratic() {
    assert_eq!(
      interpret("FactorList[x^2 + 3*x + 2]").unwrap(),
      "{{1, 1}, {1 + x, 1}, {2 + x, 1}}"
    );
  }
}

mod cancel {
  use super::*;

  #[test]
  fn cancel_simple() {
    assert_eq!(interpret("Cancel[(x^2 - 1)/(x - 1)]").unwrap(), "1 + x");
  }

  #[test]
  fn cancel_cubic() {
    assert_eq!(interpret("Cancel[(x^3 - x)/(x^2 - 1)]").unwrap(), "x");
  }

  #[test]
  fn cancel_symbolic_common_factor() {
    assert_eq!(interpret("Cancel[(a*b)/(a*c)]").unwrap(), "b/c");
  }

  #[test]
  fn cancel_symbolic_powers() {
    assert_eq!(interpret("Cancel[(a^2*b)/(a*b^2)]").unwrap(), "a/b");
  }

  #[test]
  fn cancel_numeric_content() {
    assert_eq!(interpret("Cancel[(2*x)/(4*x)]").unwrap(), "1/2");
  }

  #[test]
  fn cancel_mixed_symbolic_and_poly() {
    assert_eq!(interpret("Cancel[(a*b*x)/(a*c*x^2)]").unwrap(), "b/(c*x)");
  }

  #[test]
  fn cancel_quadratic() {
    assert_eq!(
      interpret("Cancel[(x^2 + 2*x + 1)/(x + 1)]").unwrap(),
      "1 + x"
    );
  }

  /// Regression test for issue #93: Cancel must handle canonical
  /// Times[Power[]] form identically to Divide form
  #[test]
  fn cancel_canonical_times_power_form() {
    // Both forms must produce the same result
    assert_eq!(interpret("Cancel[(x^2 - 1)/(x - 1)]").unwrap(), "1 + x");
    assert_eq!(
      interpret("Cancel[(x^2 - 1) * (x - 1)^-1]").unwrap(),
      "1 + x"
    );
  }
}

mod expand_modulus {
  use super::*;

  #[test]
  fn modulus_3() {
    assert_eq!(
      interpret("Expand[(1 + a)^12, Modulus -> 3]").unwrap(),
      "1 + a^3 + a^9 + a^12"
    );
  }

  #[test]
  fn modulus_4() {
    assert_eq!(
      interpret("Expand[(1 + a)^12, Modulus -> 4]").unwrap(),
      "1 + 2*a^2 + 3*a^4 + 3*a^8 + 2*a^10 + a^12"
    );
  }

  #[test]
  fn modulus_simple() {
    assert_eq!(
      interpret("Expand[(1 + a)^3, Modulus -> 3]").unwrap(),
      "1 + a^3"
    );
  }

  #[test]
  fn trig_sin_sum() {
    assert_eq!(
      interpret("Expand[Sin[x + y], Trig -> True]").unwrap(),
      "Cos[y]*Sin[x] + Cos[x]*Sin[y]"
    );
  }

  #[test]
  fn trig_tanh_sum() {
    assert_eq!(
      interpret("Expand[Tanh[x + y], Trig -> True]").unwrap(),
      "(Cosh[y]*Sinh[x])/(Cosh[x]*Cosh[y] + Sinh[x]*Sinh[y]) + (Cosh[x]*Sinh[y])/(Cosh[x]*Cosh[y] + Sinh[x]*Sinh[y])"
    );
  }
}

mod expand_all {
  use super::*;

  #[test]
  fn expand_all_basic() {
    assert_eq!(
      interpret("ExpandAll[x*(x + 1)^2]").unwrap(),
      "x + 2*x^2 + x^3"
    );
  }

  #[test]
  fn expand_all_expands_denominator() {
    // ExpandAll expands both the numerator and the denominator. Regression
    // for mathics algebra.py:1229.
    assert_eq!(
      interpret("ExpandAll[(a + b) ^ 2 / (c + d)^2]").unwrap(),
      "a^2/(c^2 + 2*c*d + d^2) + (2*a*b)/(c^2 + 2*c*d + d^2) \
       + b^2/(c^2 + 2*c*d + d^2)"
    );
  }

  #[test]
  fn expand_all_with_modulus_reduces_denominator() {
    // ExpandAll now accepts a Modulus option (like Expand) and applies the
    // reduction to coefficients in both numerator and denominator
    // subexpressions. Here `3*x^2*y` and `3*x*y^2` drop out of `(x+y)^3`
    // mod 3, leaving just `x^3 + y^3`. wolframscript keeps the resulting
    // fraction together (rather than distributing each numerator term over
    // the denominator) when a Modulus is supplied.
    assert_eq!(
      interpret("ExpandAll[(1 + a) ^ 6 / (x + y)^3, Modulus -> 3]").unwrap(),
      "(1 + 2*a^3 + a^6)/(x^3 + y^3)"
    );
  }
}

mod collect_tests {
  use super::*;

  #[test]
  fn collect_basic() {
    assert_eq!(interpret("Collect[x*y + x*z, x]").unwrap(), "x*(y + z)");
  }

  // Collecting with respect to a bare number is a no-op: the expression is
  // returned unchanged, rather than staying unevaluated.
  #[test]
  fn collect_by_number_returns_expression() {
    assert_eq!(interpret("Collect[x^2, 5]").unwrap(), "x^2");
    assert_eq!(interpret("Collect[a x + b x, 5]").unwrap(), "a*x + b*x");
    assert_eq!(interpret("Collect[x^2, 2.5]").unwrap(), "x^2");
    assert_eq!(interpret("Collect[x^2 + x, 0]").unwrap(), "x + x^2");
  }

  #[test]
  fn collect_constant_after_collect_variable() {
    // A bare-symbol constant term that sorts after the collect variable must
    // be placed last, matching wolframscript's canonical order, e.g.
    // `Collect[x^3 + y + x, x]` → `x + x^3 + y` (not `y + x + x^3`).
    assert_eq!(interpret("Collect[x^3 + y + x, x]").unwrap(), "x + x^3 + y");
    assert_eq!(
      interpret("Collect[x^2 + 2 x^2 + y, x]").unwrap(),
      "3*x^2 + y"
    );
    assert_eq!(
      interpret("Collect[x^3 + y + x^2, x]").unwrap(),
      "x^2 + x^3 + y"
    );
    // A constant that sorts before the collect variable still leads.
    assert_eq!(interpret("Collect[z + x + x^2, x]").unwrap(), "x + x^2 + z");
    assert_eq!(interpret("Collect[x^2 + a, x]").unwrap(), "a + x^2");
  }

  #[test]
  fn collect_compound_target_function() {
    // Collect accepts a compound target like q[x] as the variable.
    assert_eq!(
      interpret("Collect[q[x] + q[x] q[y], q[x]]").unwrap(),
      "q[x]*(1 + q[y])"
    );
  }

  #[test]
  fn collect_compound_target_with_constant() {
    // When the compound target only appears linearly with no other copies,
    // Collect returns the sum unchanged (with canonical term ordering).
    assert_eq!(
      interpret("Collect[q[0, x] q[0, y] + 1, q[0, x]]").unwrap(),
      "1 + q[0, x]*q[0, y]"
    );
  }

  #[test]
  fn collect_symbolic_coefficients() {
    // Coefficients with variables before the collect variable go first
    assert_eq!(
      interpret("Collect[a*x^2 + b*x + c*x^2 + d*x, x]").unwrap(),
      "(b + d)*x + (a + c)*x^2"
    );
  }

  #[test]
  fn collect_mixed_coefficients() {
    // Coefficient (2+y) has y > x alphabetically, so x goes first
    assert_eq!(
      interpret("Collect[x*y + 2*x + 3*y + 6, x]").unwrap(),
      "6 + 3*y + x*(2 + y)"
    );
  }

  #[test]
  fn collect_with_constant_term() {
    assert_eq!(
      interpret("Collect[a*x^2 + b*x^2 + c*x + d*x + e, x]").unwrap(),
      "e + (c + d)*x + (a + b)*x^2"
    );
  }

  #[test]
  fn collect_product_coefficient_canonical_ordering() {
    // When the coefficient is a product (not a sum), factors should be
    // flattened and sorted in canonical order (alphabetical).
    assert_eq!(
      interpret("Collect[a x^2 + b x^2 y + c x y, x]").unwrap(),
      "c*x*y + x^2*(a + b*y)"
    );
  }

  #[test]
  fn collect_with_head() {
    assert_eq!(
      interpret("Collect[a x + b x + c, x, h]").unwrap(),
      "x*h[a + b] + h[c]"
    );
  }

  #[test]
  fn collect_two_variables_shared_factor_ascending_powers() {
    // Regression: when collecting by {x, y} and every grouped term shares
    // the y factor, terms should be ordered by ascending power of x and
    // each term's Plus coefficient should appear *before* the monomial
    // factors (Plus * x^k * y), matching wolframscript exactly.
    assert_eq!(
      interpret(
        "Collect[a^2 y + 2 a b y + b^2 y + 2 a x y + 2 b x y + x^2 y + c^2 x^2 y + 2 c d x^2 y + d^2 x^2 y, {x, y}]"
      )
      .unwrap(),
      "(a^2 + 2*a*b + b^2)*y + (2*a + 2*b)*x*y + (1 + c^2 + 2*c*d + d^2)*x^2*y"
    );
  }
}

mod together {
  use super::*;

  #[test]
  fn together_basic() {
    assert_eq!(interpret("Together[1/x + 1/y]").unwrap(), "(x + y)/(x*y)");
  }

  // Together cancels a common polynomial factor between numerator and a bare
  // (unfactored) polynomial denominator, like wolframscript: (x^2+x)/(x^2-1)
  // shares (x+1), reducing to x/(x-1).
  #[test]
  fn together_cancels_polynomial_gcd() {
    assert_eq!(
      interpret("Together[(x^2 + x)/(x^2 - 1)]").unwrap(),
      "x/(-1 + x)"
    );
    assert_eq!(
      interpret("Together[x^2/(x^2 - 1) + x/(x^2 - 1)]").unwrap(),
      "x/(-1 + x)"
    );
    assert_eq!(
      interpret("Together[(x^2 - 4)/(x^2 - x - 2)]").unwrap(),
      "(2 + x)/(1 + x)"
    );
    // No common factor: the fraction is left combined but uncancelled.
    assert_eq!(
      interpret("Together[(2 x)/(x^2 - 1)]").unwrap(),
      "(2*x)/(-1 + x^2)"
    );
    // Factored denominators keep their factored form (no spurious expansion).
    assert_eq!(
      interpret("Together[1/(x - 1) + 1/(x + 1)]").unwrap(),
      "(2*x)/((-1 + x)*(1 + x))"
    );
  }

  #[test]
  fn together_symbolic_fractions() {
    assert_eq!(
      interpret("Together[a/b + c/d]").unwrap(),
      "(b*c + a*d)/(b*d)"
    );
  }

  #[test]
  fn together_scalar_times_sum_of_fractions() {
    // A sum of fractions wrapped in a scalar product must still be combined;
    // preprocessing previously left `Times[scalar, Plus[...]]` uncombined.
    assert_eq!(interpret("Together[x (1/x + 1/y)]").unwrap(), "(x + y)/y");
    assert_eq!(
      interpret("Together[3 (1/(a - I x) + 1/(a + I x))]").unwrap(),
      "(6*a)/(a^2 + x^2)"
    );
  }

  #[test]
  fn together_subtracted_fractions() {
    assert_eq!(
      interpret("Together[1/(x-1) - 1/(x+1)]").unwrap(),
      "2/((-1 + x)*(1 + x))"
    );
  }

  #[test]
  fn together_added_fractions_with_binomial_denominators() {
    assert_eq!(
      interpret("Together[1/(x-1) + 1/(x+1)]").unwrap(),
      "(2*x)/((-1 + x)*(1 + x))"
    );
  }

  // Together cancels the numerator/denominator GCD when the fraction reduces
  // to a polynomial.
  #[test]
  fn together_cancels_to_polynomial() {
    assert_eq!(interpret("Together[(x^2 - 1)/(x - 1)]").unwrap(), "1 + x");
    assert_eq!(
      interpret("Together[(x^3 - 1)/(x - 1)]").unwrap(),
      "1 + x + x^2"
    );
    assert_eq!(
      interpret("Together[(x^2 + 2 x + 1)/(x + 1)]").unwrap(),
      "1 + x"
    );
    // Two variables.
    assert_eq!(interpret("Together[(x^2 - y^2)/(x - y)]").unwrap(), "x + y");
  }

  // When the reduced polynomial carries numeric content, it is factored out
  // (FactorTerms behavior), matching wolframscript.
  #[test]
  fn together_factors_numeric_content_after_cancel() {
    assert_eq!(
      interpret("Together[(6 x^2 - 6)/(3 x - 3)]").unwrap(),
      "2*(1 + x)"
    );
    assert_eq!(interpret("Together[(2 x + 2)/(x + 1)]").unwrap(), "2");
  }
}

mod apart {
  use super::*;

  #[test]
  fn apart_basic() {
    assert_eq!(
      interpret("Apart[1/(x^2 - 1)]").unwrap(),
      "1/(2*(-1 + x)) - 1/(2*(1 + x))"
    );
  }

  #[test]
  fn apart_x2_plus_1_over_x3_minus_x() {
    assert_eq!(
      interpret("Apart[(x^2 + 1)/(x^3 - x)]").unwrap(),
      "(-1 + x)^(-1) - x^(-1) + (1 + x)^(-1)"
    );
  }

  #[test]
  fn apart_two_linear_factors() {
    assert_eq!(
      interpret("Apart[1/((x - 1)*(x - 2))]").unwrap(),
      "(-2 + x)^(-1) - (-1 + x)^(-1)"
    );
  }

  #[test]
  fn apart_three_linear_factors() {
    assert_eq!(
      interpret("Apart[1/((x-1)*(x-2)*(x-3))]").unwrap(),
      "1/(2*(-3 + x)) - (-2 + x)^(-1) + 1/(2*(-1 + x))"
    );
  }

  #[test]
  fn apart_on_non_rational_is_noop() {
    // Apart on an expression without a denominator should return it unchanged.
    assert_eq!(
      interpret("Apart[Sin[1 / (x ^ 2 - y ^ 2)]]").unwrap(),
      "Sin[(x^2 - y^2)^(-1)]"
    );
  }

  #[test]
  fn apart_on_equation_is_noop() {
    // Apart on a non-numeric expression without a denominator returns it
    // unchanged. Wolframscript -code prints OutputForm, which strips the
    // quotes around held strings inside comparisons (e.g. `a == A`).
    assert_eq!(interpret("Apart[a == \"A\"]").unwrap(), "a == A");
  }

  // A factor whose residue is zero must not produce a spurious `0/(...)`
  // term: (x + 1)/(x^2 + x) = (x + 1)/(x (x + 1)) = 1/x.
  #[test]
  fn apart_drops_zero_residue_term() {
    assert_eq!(interpret("Apart[(x + 1)/(x^2 + x)]").unwrap(), "x^(-1)");
  }

  // A removable factor (the numerator cancels one root) leaves only the
  // surviving partial fraction: (x - 3)/((x - 3)(x + 1)) = 1/(x + 1).
  #[test]
  fn apart_cancelling_factor() {
    assert_eq!(
      interpret("Apart[(x - 3)/(x^2 - 2 x - 3)]").unwrap(),
      "(1 + x)^(-1)"
    );
  }

  #[test]
  fn apart_numerator_x_over_two_factors() {
    assert_eq!(
      interpret("Apart[x/((x - 1) (x - 2))]").unwrap(),
      "2/(-2 + x) - (-1 + x)^(-1)"
    );
  }

  // A constant polynomial part is spliced in as a flat sum (no spurious
  // parentheses around the partial-fraction part).
  #[test]
  fn apart_with_constant_quotient_is_flat() {
    assert_eq!(
      interpret("Apart[(x^2 + 1)/(x^2 - 1)]").unwrap(),
      "1 + (-1 + x)^(-1) - (1 + x)^(-1)"
    );
  }

  // Repeated (squared) denominator factors: each root of multiplicity m
  // contributes terms 1/(x-r), …, 1/(x-r)^m, highest power first.
  #[test]
  fn apart_squared_factor() {
    assert_eq!(
      interpret("Apart[1/(x^2 (x + 1))]").unwrap(),
      "x^(-2) - x^(-1) + (1 + x)^(-1)"
    );
  }

  #[test]
  fn apart_repeated_linear_factor_only() {
    assert_eq!(interpret("Apart[1/(x + 2)^2]").unwrap(), "(2 + x)^(-2)");
    assert_eq!(interpret("Apart[3/(x + 2)^2]").unwrap(), "3/(2 + x)^2");
    assert_eq!(
      interpret("Apart[(2 x + 5)/(x + 2)^2]").unwrap(),
      "(2 + x)^(-2) + 2/(2 + x)"
    );
  }

  // Irreducible quadratic factors: constant numerators over the linear
  // factors and linear numerators `B x + C` over the quadratics, matching
  // wolframscript's canonical output exactly.
  #[test]
  fn apart_irreducible_quadratic_factors() {
    // Two distinct quadratics; numerators collapse to constants by symmetry.
    assert_eq!(
      interpret("Apart[1/((x^2 + 1)(x^2 + 4))]").unwrap(),
      "1/(3*(1 + x^2)) - 1/(3*(4 + x^2))"
    );
    // Coefficient 1 renders with the `^(-1)` reciprocal form.
    assert_eq!(
      interpret("Apart[1/((x^2 + 1)(x^2 + 2))]").unwrap(),
      "(1 + x^2)^(-1) - (2 + x^2)^(-1)"
    );
    // Linear numerator over the quadratic factor; linear factor term first.
    assert_eq!(
      interpret("Apart[1/((x + 1)(x^2 + 1))]").unwrap(),
      "1/(2*(1 + x)) + (1 - x)/(2*(1 + x^2))"
    );
    assert_eq!(
      interpret("Apart[(x + 1)/((x^2 + 1)(x - 1))]").unwrap(),
      "(-1 + x)^(-1) - x/(1 + x^2)"
    );
    // Numerator with an x term over each quadratic.
    assert_eq!(
      interpret("Apart[x/((x^2 + 1)(x^2 + 4))]").unwrap(),
      "x/(3*(1 + x^2)) - x/(3*(4 + x^2))"
    );
    // Quadratic with a middle term, plus a linear factor.
    assert_eq!(
      interpret("Apart[(2 x + 1)/((x^2 + x + 1)(x - 5))]").unwrap(),
      "11/(31*(-5 + x)) + (-4 - 11*x)/(31*(1 + x + x^2))"
    );
    // The quartic factors into two conjugate quadratics automatically.
    assert_eq!(
      interpret("Apart[1/(x^4 + x^2 + 1)]").unwrap(),
      "(1 - x)/(2*(1 - x + x^2)) + (1 + x)/(2*(1 + x + x^2))"
    );
    // Repeated quadratic factor times a linear factor (powers k = 1, 2).
    assert_eq!(
      interpret("Apart[1/((x^2 + 1)^2 (x - 1))]").unwrap(),
      "1/(4*(-1 + x)) + (-1 - x)/(2*(1 + x^2)^2) + (-1 - x)/(4*(1 + x^2))"
    );
    // Mixed linear + quadratic.
    assert_eq!(
      interpret("Apart[1/(x (x^2 + 1))]").unwrap(),
      "x^(-1) - x/(1 + x^2)"
    );
  }

  #[test]
  fn apart_squared_factor_with_simple_factor() {
    assert_eq!(
      interpret("Apart[1/((x - 1)^2 (x + 1))]").unwrap(),
      "1/(2*(-1 + x)^2) - 1/(4*(-1 + x)) + 1/(4*(1 + x))"
    );
  }

  #[test]
  fn apart_cubic_repeated_factor() {
    assert_eq!(
      interpret("Apart[1/((x - 1)^3 (x + 2))]").unwrap(),
      "1/(3*(-1 + x)^3) - 1/(9*(-1 + x)^2) + 1/(27*(-1 + x)) - 1/(27*(2 + x))"
    );
  }

  #[test]
  fn apart_pure_power_denominator() {
    assert_eq!(interpret("Apart[1/x^3]").unwrap(), "x^(-3)");
    assert_eq!(interpret("Apart[5/(x - 1)^2]").unwrap(), "5/(-1 + x)^2");
  }

  #[test]
  fn apart_two_repeated_factors() {
    assert_eq!(
      interpret("Apart[1/(x^2 (x + 1)^2)]").unwrap(),
      "x^(-2) - 2/x + (1 + x)^(-2) + 2/(1 + x)"
    );
  }
}

mod switch {
  use super::*;

  #[test]
  fn basic_match() {
    assert_eq!(interpret("Switch[2, 1, a, 2, b, 3, c]").unwrap(), "b");
  }

  #[test]
  fn first_match() {
    assert_eq!(interpret("Switch[1, 1, a, 2, b, 3, c]").unwrap(), "a");
  }

  #[test]
  fn no_match_returns_unevaluated() {
    assert_eq!(
      interpret("Switch[4, 1, a, 2, b, 3, c]").unwrap(),
      "Switch[4, 1, a, 2, b, 3, c]"
    );
  }

  #[test]
  fn symbolic_target_no_match_returns_unevaluated() {
    assert_eq!(
      interpret("Switch[p, a, 1, b, 2]").unwrap(),
      "Switch[p, a, 1, b, 2]"
    );
  }

  #[test]
  fn wildcard_match() {
    assert_eq!(interpret("Switch[4, 1, a, _, c]").unwrap(), "c");
  }

  #[test]
  fn evaluated_expression() {
    assert_eq!(interpret("Switch[1 + 1, 1, a, 2, b, 3, c]").unwrap(), "b");
  }

  #[test]
  fn head_constrained_blank() {
    // _Head pattern matches expressions with matching head
    assert_eq!(
      interpret("Switch[C[1], _C, matched, _, other]").unwrap(),
      "matched"
    );
  }

  #[test]
  fn head_constrained_blank_no_match() {
    // _Head pattern doesn't match different head
    assert_eq!(
      interpret("Switch[D[1], _C, matched, _, other]").unwrap(),
      "other"
    );
  }

  #[test]
  fn head_constrained_blank_integer() {
    assert_eq!(
      interpret("Switch[42, _Integer, num, _String, str]").unwrap(),
      "num"
    );
  }

  #[test]
  fn named_pattern() {
    // x_ matches anything — Switch does NOT bind pattern variables
    assert_eq!(interpret("Switch[42, x_, x + 1]").unwrap(), "1 + x");
  }
}

mod piecewise {
  use super::*;

  #[test]
  fn first_true() {
    assert_eq!(interpret("Piecewise[{{1, True}}]").unwrap(), "1");
  }

  #[test]
  fn second_true() {
    assert_eq!(
      interpret("Piecewise[{{1, False}, {2, True}}]").unwrap(),
      "2"
    );
  }

  #[test]
  fn default_value() {
    assert_eq!(
      interpret("Piecewise[{{1, False}, {2, False}}, 42]").unwrap(),
      "42"
    );
  }

  #[test]
  fn no_match_default_zero() {
    assert_eq!(interpret("Piecewise[{{1, False}}]").unwrap(), "0");
  }

  #[test]
  fn with_conditions() {
    clear_state();
    assert_eq!(
      interpret("x = 5; Piecewise[{{1, x < 0}, {2, x >= 0}}]").unwrap(),
      "2"
    );
  }

  #[test]
  fn prefix_apply() {
    // Piecewise @ {{...}} should work the same as Piecewise[{{...}}]
    assert_eq!(
      interpret("Piecewise @ {{1, True}, {2, False}}").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Piecewise @ {{1, False}, {2, True}}").unwrap(),
      "2"
    );
  }

  #[test]
  fn prefix_apply_with_conditions() {
    clear_state();
    assert_eq!(
      interpret("x = 3; Piecewise @ {{1, x < 0}, {2, x >= 0}}").unwrap(),
      "2"
    );
  }

  #[test]
  fn with_chained_inequality() {
    clear_state();
    assert_eq!(
      interpret("x = 0.5; Piecewise[{{1, 0 <= x < 1}, {2, 1 <= x < 2}}]")
        .unwrap(),
      "1"
    );
    assert_eq!(
      interpret("x = 1.5; Piecewise[{{1, 0 <= x < 1}, {2, 1 <= x < 2}}]")
        .unwrap(),
      "2"
    );
  }

  #[test]
  fn via_table_prefix_apply() {
    assert_eq!(
      interpret("With[{l = {1, 2, 1}}, Piecewise @ Table[{(-1)^i, Accumulate[Prepend[l, 0]][[i]] <= t < Accumulate[Prepend[l, 0]][[i + 1]]}, {i, 1, Length[l]}] /. t -> 0.5]").unwrap(),
      "-1"
    );
    assert_eq!(
      interpret("With[{l = {1, 2, 1}}, Piecewise @ Table[{(-1)^i, Accumulate[Prepend[l, 0]][[i]] <= t < Accumulate[Prepend[l, 0]][[i + 1]]}, {i, 1, Length[l]}] /. t -> 1.5]").unwrap(),
      "1"
    );
  }

  #[test]
  fn variable_holding_pairs_evaluates_to_inner_value() {
    // Regression: `Piecewise[x]` where `x` is bound to a literal list of
    // pairs erroneously errored out because the head check ran before the
    // arg was evaluated. With the eval-the-arg-first change, the variable
    // resolves to its bound List first.
    assert_eq!(
      interpret("x = {{1, True}, {2, False}}; Piecewise[x]").unwrap(),
      "1"
    );
  }
}

mod match_q {
  use super::*;

  #[test]
  fn head_matching() {
    assert_eq!(interpret("MatchQ[{1, 2, 3}, _List]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[42, _Integer]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[3.14, _Real]").unwrap(), "True");
    assert_eq!(interpret(r#"MatchQ["hello", _String]"#).unwrap(), "True");
  }

  #[test]
  fn head_mismatch() {
    assert_eq!(interpret("MatchQ[1, _String]").unwrap(), "False");
    assert_eq!(interpret("MatchQ[1, _List]").unwrap(), "False");
    assert_eq!(interpret(r#"MatchQ["x", _Integer]"#).unwrap(), "False");
  }

  #[test]
  fn blank_matches_anything() {
    assert_eq!(interpret("MatchQ[42, _]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[{1, 2}, _]").unwrap(), "True");
    assert_eq!(interpret(r#"MatchQ["x", _]"#).unwrap(), "True");
  }

  #[test]
  fn literal_matching() {
    assert_eq!(interpret("MatchQ[42, 42]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[42, 43]").unwrap(), "False");
    assert_eq!(interpret("MatchQ[x, x]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[x, y]").unwrap(), "False");
  }

  #[test]
  fn operator_form() {
    assert_eq!(interpret("MatchQ[_Integer][123]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[_String][123]").unwrap(), "False");
  }

  #[test]
  fn repeated_pattern_variable_must_match() {
    // Regression: matches_pattern_ast ignored Pattern names, so both `a_`
    // positions in {a_, b_, a_} would match independently.
    assert_eq!(
      interpret("MatchQ[{1, 2, 3}, {a_, b_, a_}]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("MatchQ[{1, 2, 1}, {a_, b_, a_}]").unwrap(),
      "True"
    );
  }

  #[test]
  fn repeated_pattern_variable_in_function_args() {
    // Pattern variable `x_` used twice must match the same value
    assert_eq!(interpret("MatchQ[f[1, 1], f[x_, x_]]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[f[1, 2], f[x_, x_]]").unwrap(), "False");
  }
}

mod replace_all_after_operators {
  use super::*;

  #[test]
  fn replace_all_after_plus() {
    assert_eq!(interpret("x + y /. x -> 1").unwrap(), "1 + y");
  }

  #[test]
  fn replace_all_after_times() {
    assert_eq!(interpret("x * y /. x -> 2").unwrap(), "2*y");
  }

  #[test]
  fn replace_all_after_power() {
    assert_eq!(interpret("x^2 + y /. x -> 3").unwrap(), "9 + y");
  }

  #[test]
  fn replace_all_multiple_operators() {
    assert_eq!(interpret("x + y + z /. {x -> 1, y -> 2}").unwrap(), "3 + z");
  }

  #[test]
  fn replace_all_times_multiple_vars() {
    assert_eq!(interpret("x * y * z /. {x -> 2, y -> 3}").unwrap(), "6*z");
  }

  #[test]
  fn replace_all_with_implicit_times() {
    assert_eq!(interpret("2 x + 3 y /. {x -> 1, y -> 2}").unwrap(), "8");
  }

  #[test]
  fn replace_repeated_after_plus() {
    assert_eq!(interpret("x + y //. x -> 1").unwrap(), "1 + y");
  }

  #[test]
  fn replace_repeated_after_times() {
    assert_eq!(interpret("x * y //. x -> 2").unwrap(), "2*y");
  }

  #[test]
  fn replace_all_after_comparison() {
    assert_eq!(interpret("x > y /. x -> 3").unwrap(), "3 > y");
  }

  #[test]
  fn replace_all_all_vars_replaced() {
    assert_eq!(interpret("x + y /. {x -> 10, y -> 20}").unwrap(), "30");
  }

  #[test]
  fn replace_all_list_of_rules_simultaneous() {
    // Rules should be applied simultaneously, not sequentially
    // 2->1 and 1->0 should NOT chain (2 becomes 1, not 0)
    assert_eq!(
      interpret("{1,1,2,2,2,2} /. {2 -> 1, 1 -> 0}").unwrap(),
      "{0, 0, 1, 1, 1, 1}"
    );
  }

  #[test]
  fn replace_all_list_of_rules_first_match_wins() {
    // x->y should match, then y->z should NOT apply to the result
    assert_eq!(interpret("x /. {x -> y, y -> z}").unwrap(), "y");
  }

  #[test]
  fn replace_all_list_of_rules_no_match() {
    // No rule matches, original expression returned
    assert_eq!(
      interpret("{3, 4, 5} /. {1 -> a, 2 -> b}").unwrap(),
      "{3, 4, 5}"
    );
  }

  #[test]
  fn replace_all_list_of_rules_partial_match() {
    // Only some elements match
    assert_eq!(
      interpret("{a, b, c} /. {a -> x, b -> y}").unwrap(),
      "{x, y, c}"
    );
  }

  #[test]
  fn replace_all_list_of_rules_swap() {
    // Swap a and b simultaneously
    assert_eq!(
      interpret("{a, b, a, b} /. {a -> b, b -> a}").unwrap(),
      "{b, a, b, a}"
    );
  }
}

mod replace_all_expression_rhs {
  use super::*;

  #[test]
  fn rule_delayed_with_division() {
    assert_eq!(
      interpret("{0, 0, 1, 1, 1, 1} /. {any_Integer :> any / 2}").unwrap(),
      "{0, 0, 1/2, 1/2, 1/2, 1/2}"
    );
  }

  #[test]
  fn rule_delayed_with_addition() {
    assert_eq!(
      interpret("{1, 2, 3} /. {x_ :> x + 10}").unwrap(),
      "{11, 12, 13}"
    );
  }

  #[test]
  fn rule_with_expression_rhs() {
    assert_eq!(
      interpret("{1, 2, 3} /. x_Integer -> x + 10").unwrap(),
      "{11, 12, 13}"
    );
  }

  #[test]
  fn rule_delayed_with_power() {
    assert_eq!(interpret("5 /. x_Integer :> x^2").unwrap(), "25");
  }
}

mod replace_all_head_constraint {
  use super::*;

  #[test]
  fn integer_head_matches() {
    assert_eq!(
      interpret("{0, 0, 1, 1, 1, 1} /. {any_Integer :> any / 2}").unwrap(),
      "{0, 0, 1/2, 1/2, 1/2, 1/2}"
    );
  }

  #[test]
  fn integer_head_skips_non_integers() {
    assert_eq!(
      interpret(r#"{1, 2.5, "hello", x} /. a_Integer :> a + 100"#).unwrap(),
      "{101, 2.5, hello, x}"
    );
  }

  #[test]
  fn string_head_matches() {
    assert_eq!(
      interpret(r#"{1, "hi", "bye"} /. s_String :> StringLength[s]"#).unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn real_head_matches() {
    assert_eq!(
      interpret("{1, 2.5, 3.0} /. x_Real :> x * 10").unwrap(),
      "{1, 25., 30.}"
    );
  }

  #[test]
  fn head_no_match_returns_unchanged() {
    assert_eq!(
      interpret(r#""hello" /. x_Integer :> x^2"#).unwrap(),
      "hello"
    );
  }

  #[test]
  fn multiple_head_rules() {
    assert_eq!(
      interpret("{1, 2.5, 3} /. {a_Integer :> a + 100, b_Real :> b * 10}")
        .unwrap(),
      "{101, 25., 103}"
    );
  }
}

mod replace_all_conditional_multi_rules {
  use super::*;

  #[test]
  fn conditional_pattern_single_rule() {
    assert_eq!(interpret("27 /. n_ /; OddQ[n] :> 3 n + 1").unwrap(), "82");
  }

  #[test]
  fn conditional_pattern_multi_rules_scalar() {
    // Multi-rule ReplaceAll with conditional patterns on a scalar
    assert_eq!(
      interpret("6 /. {n_ /; EvenQ[n] :> n/2, n_ /; OddQ[n] :> 3 n + 1}")
        .unwrap(),
      "3"
    );
  }

  #[test]
  fn conditional_pattern_multi_rules_list() {
    assert_eq!(
      interpret("{27, 6} /. {n_ /; EvenQ[n] :> n/2, n_ /; OddQ[n] :> 3 n + 1}")
        .unwrap(),
      "{82, 3}"
    );
  }

  #[test]
  fn nested_list_multi_rules() {
    // Multi-rule ReplaceAll should recurse into nested lists
    assert_eq!(
      interpret("{C[1], {X, 4, Y, C[1]}} /. {X -> a, Y -> b}").unwrap(),
      "{C[1], {a, 4, b, C[1]}}"
    );
  }

  #[test]
  fn nested_list_swap() {
    assert_eq!(
      interpret("{{a, b}, {c, d}} /. {a -> 1, d -> 4}").unwrap(),
      "{{1, b}, {c, 4}}"
    );
  }

  // Regression: `expr /. rules-tree` should recurse into *every* level of
  // List in the rules argument, not just when each top-level element is a
  // flat rule list. Matches wolframscript's tree-of-rule-lists semantics.
  #[test]
  fn nested_rule_tree_three_levels() {
    assert_eq!(
      interpret("{a, b} /. {{{a->x, b->y}, {a->w, b->z}}, {a->u, b->v}}")
        .unwrap(),
      "{{{x, y}, {w, z}}, {u, v}}"
    );
  }

  #[test]
  fn nested_rule_tree_two_levels() {
    assert_eq!(
      interpret("{a, b} /. {{a->x, b->y}, {a->w, b->z}}").unwrap(),
      "{{x, y}, {w, z}}"
    );
  }
}

mod replace_all_variable_rhs {
  use super::*;

  #[test]
  fn variable_holding_rules() {
    clear_state();
    assert_eq!(
      interpret("r = {x -> 1, y -> 2}; {x, y, z} /. r").unwrap(),
      "{1, 2, z}"
    );
  }

  #[test]
  fn variable_holding_conditional_rules() {
    clear_state();
    assert_eq!(
      interpret("r = {x_ /; EvenQ[x] :> x/2}; Map[# /. r &, {4, 7}]").unwrap(),
      "{2, 7}"
    );
  }

  #[test]
  fn variable_in_anonymous_function() {
    clear_state();
    assert_eq!(
      interpret("r = {a -> b, b -> c}; Nest[# /. r &, a, 2]").unwrap(),
      "c"
    );
  }
}

mod replace_repeated {
  use super::*;

  #[test]
  fn replace_repeated_applies_multiple_times() {
    assert_eq!(interpret("f[f[f[f[2]]]] //. f[2] -> 2").unwrap(), "2");
  }

  #[test]
  fn replace_repeated_simple() {
    assert_eq!(interpret("f[f[2]] //. f[2] -> 2").unwrap(), "2");
  }
}

mod replace_repeated_operator_form {
  use super::*;

  #[test]
  fn operator_form_works() {
    // ReplaceRepeated[rule][expr] should work like expr //. rule
    let result =
      interpret("ReplaceRepeated[f[2] -> 2][f[f[f[f[2]]]]]").unwrap();
    assert_eq!(result, "2");
  }

  #[test]
  fn infix_form_works() {
    let result = interpret("f[f[f[2]]] //. f[2] -> 2").unwrap();
    assert_eq!(result, "2");
  }
}

mod symbolic_equal {
  use super::*;

  #[test]
  fn numeric_equal() {
    assert_eq!(interpret("1 == 1").unwrap(), "True");
    assert_eq!(interpret("1 == 2").unwrap(), "False");
  }

  #[test]
  fn identical_symbols() {
    assert_eq!(interpret("x == x").unwrap(), "True");
  }

  #[test]
  fn reciprocal_product_equal() {
    // Power[Times[...], -1] should distribute and match 1/(...) form
    assert_eq!(interpret("1/(a*b) == (a*b)^-1").unwrap(), "True");
    assert_eq!(interpret("1/(x*y) == (x*y)^-1").unwrap(), "True");
    assert_eq!(
      interpret("1/(Sqrt[x]*(a + b*x)) == (Sqrt[x]*(a + b*x))^-1").unwrap(),
      "True"
    );
  }

  #[test]
  fn different_symbols_stay_symbolic() {
    assert_eq!(interpret("x == y").unwrap(), "x == y");
  }

  #[test]
  fn symbolic_expression_vs_number() {
    assert_eq!(interpret("x + 1 == 0").unwrap(), "1 + x == 0");
  }

  #[test]
  fn same_q_always_evaluates() {
    assert_eq!(interpret("x === x").unwrap(), "True");
    assert_eq!(interpret("x === y").unwrap(), "False");
  }

  #[test]
  fn unequal_symbolic() {
    assert_eq!(interpret("x != x").unwrap(), "False");
    assert_eq!(interpret("x != y").unwrap(), "x != y");
  }

  #[test]
  fn unsame_q_always_evaluates() {
    assert_eq!(interpret("x =!= x").unwrap(), "False");
    assert_eq!(interpret("x =!= y").unwrap(), "True");
  }

  // SameQ/UnsameQ compare deeply nested expressions without overflowing the
  // stack. Regression: comparing Nest[f, x, 5000] used to crash with a
  // stack overflow because the formatter (expr_to_string) was not stack-safe.
  #[test]
  fn same_q_deeply_nested_no_overflow() {
    // Depth 600 is comfortably past the old ~500-deep stack-overflow point
    // while staying cheap enough to run under the parallel test harness.
    assert_eq!(
      interpret("UnsameQ[Nest[f, x, 600], Nest[f, x, 601]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("SameQ[Nest[f, x, 600], Nest[f, x, 600]]").unwrap(),
      "True"
    );
  }
}

mod solve {
  use super::*;

  #[test]
  fn linear_equation() {
    assert_eq!(interpret("Solve[x - 5 == 0, x]").unwrap(), "{{x -> 5}}");
    assert_eq!(interpret("Solve[2*x + 6 == 0, x]").unwrap(), "{{x -> -3}}");
    assert_eq!(interpret("Solve[3*x + 9 == 0, x]").unwrap(), "{{x -> -3}}");
  }

  #[test]
  fn quadratic_integer_roots() {
    assert_eq!(
      interpret("Solve[x^2 + 3x - 10 == 0, x]").unwrap(),
      "{{x -> -5}, {x -> 2}}"
    );
  }

  #[test]
  fn one_argument_auto_detects_variable() {
    // Single-variable: the variable is inferred from the equation.
    assert_eq!(interpret("Solve[2 x == 6]").unwrap(), "{{x -> 3}}");
    assert_eq!(
      interpret("Solve[x^2 - 5 x + 6 == 0]").unwrap(),
      "{{x -> 2}, {x -> 3}}"
    );
    // Determined system: solve for all variables.
    assert_eq!(
      interpret("Solve[{x + y == 3, x - y == 1}]").unwrap(),
      "{{x -> 2, y -> 1}}"
    );
    // A trivially-true condition yields the empty solution {{}}.
    assert_eq!(interpret("Solve[x == x]").unwrap(), "{{}}");
  }

  #[test]
  fn one_argument_underdetermined_stays_unevaluated() {
    // An underdetermined system uses a non-obvious variable-selection
    // heuristic in wolframscript, so Woxi leaves it unevaluated rather than
    // guessing the wrong variable.
    assert_eq!(interpret("Solve[x + y == 3]").unwrap(), "Solve[x + y == 3]");
  }

  #[test]
  fn quadratic_symmetric() {
    assert_eq!(
      interpret("Solve[x^2 - 4 == 0, x]").unwrap(),
      "{{x -> -2}, {x -> 2}}"
    );
  }

  #[test]
  fn quadratic_with_leading_coeff() {
    assert_eq!(
      interpret("Solve[2*x^2 - 8 == 0, x]").unwrap(),
      "{{x -> -2}, {x -> 2}}"
    );
  }

  #[test]
  fn quadratic_repeated_root() {
    assert_eq!(
      interpret("Solve[x^2 + 2*x + 1 == 0, x]").unwrap(),
      "{{x -> -1}, {x -> -1}}"
    );
  }

  #[test]
  fn quadratic_complex_roots() {
    assert_eq!(
      interpret("Solve[x^2 + 1 == 0, x]").unwrap(),
      "{{x -> -I}, {x -> I}}"
    );
  }

  #[test]
  fn quadratic_irrational_roots() {
    assert_eq!(
      interpret("Solve[x^2 - 5 == 0, x]").unwrap(),
      "{{x -> -Sqrt[5]}, {x -> Sqrt[5]}}"
    );
  }

  #[test]
  fn quadratic_golden_ratio() {
    assert_eq!(
      interpret("Solve[x^2 + x - 1 == 0, x]").unwrap(),
      "{{x -> (-1 - Sqrt[5])/2}, {x -> (-1 + Sqrt[5])/2}}"
    );
  }

  #[test]
  fn denominator_of_rational_function_roots() {
    // Solve[Denominator[f[x]] == 0, x] for f[x] = 4x/(x^2 + 3x + 5).
    // Matches wolframscript; mathics docstring displays as "-3/2 +/- I/2 Sqrt[11]".
    assert_eq!(
      interpret(
        "f[x_] := 4 x / (x^2 + 3 x + 5); Solve[Denominator[f[x]] == 0, x]"
      )
      .unwrap(),
      "{{x -> (-3 - I*Sqrt[11])/2}, {x -> (-3 + I*Sqrt[11])/2}}"
    );
  }

  #[test]
  fn quadratic_general() {
    assert_eq!(
      interpret("Solve[x^2 - 2*x - 3 == 0, x]").unwrap(),
      "{{x -> -1}, {x -> 3}}"
    );
  }

  #[test]
  fn trivial_x_equals_zero() {
    assert_eq!(interpret("Solve[x == 0, x]").unwrap(), "{{x -> 0}}");
  }

  #[test]
  fn tautology() {
    assert_eq!(interpret("Solve[x == x, x]").unwrap(), "{{}}");
  }

  #[test]
  fn contradiction() {
    assert_eq!(interpret("Solve[2 == 3, x]").unwrap(), "{}");
  }

  #[test]
  fn rational_solution() {
    assert_eq!(interpret("Solve[2*x - 1 == 0, x]").unwrap(), "{{x -> 1/2}}");
  }

  #[test]
  fn constant_variable_rejected() {
    assert_eq!(
      interpret("Solve[x + E == 0, E]").unwrap(),
      "Solve[E + x == 0, E]"
    );
  }

  #[test]
  fn fractional_power_equation() {
    // Physics: find extrema of E-field on axis of ring charge
    // Common constant factors (2*k*q) are factored out before the quadratic formula
    assert_eq!(
      interpret(
        "Solve[(2*k*q*(a^2 + x^2)^(3/2) - 6*k*q*x^2*(a^2 + x^2)^(1/2))/(a^2 + x^2)^3 == 0, x]"
      )
      .unwrap(),
      "{{x -> -(a/Sqrt[2])}, {x -> a/Sqrt[2]}}"
    );
  }

  #[test]
  fn symbolic_quadratic_simple() {
    assert_eq!(
      interpret("Solve[a^2 - 2*x^2 == 0, x]").unwrap(),
      "{{x -> -(a/Sqrt[2])}, {x -> a/Sqrt[2]}}"
    );
  }

  #[test]
  fn symbolic_quadratic_general() {
    assert_eq!(
      interpret("Solve[a*x^2 + b*x + c == 0, x]").unwrap(),
      "{{x -> (-b - Sqrt[b^2 - 4*a*c])/(2*a)}, {x -> (-b + Sqrt[b^2 - 4*a*c])/(2*a)}}"
    );
  }

  #[test]
  fn quartic_factor_based() {
    assert_eq!(
      interpret("Solve[x^4 - x == 0, x]").unwrap(),
      "{{x -> 0}, {x -> 1}, {x -> -(-1)^(1/3)}, {x -> (-1)^(2/3)}}"
    );
  }

  #[test]
  fn cubic_factor_based() {
    assert_eq!(
      interpret("Solve[x^3 - 1 == 0, x]").unwrap(),
      "{{x -> 1}, {x -> -(-1)^(1/3)}, {x -> (-1)^(2/3)}}"
    );
  }

  #[test]
  fn cubic_integer_roots_sorted() {
    // Solutions must be sorted ascending, matching Wolfram Language
    assert_eq!(
      interpret("Solve[x^3 - 6 x^2 + 11 x - 6 == 0, x]").unwrap(),
      "{{x -> 1}, {x -> 2}, {x -> 3}}"
    );
  }

  #[test]
  fn quartic_integer_roots_sorted() {
    // Solutions must be sorted ascending, matching Wolfram Language
    assert_eq!(
      interpret("Solve[x^4 - 10 x^2 + 9 == 0, x]").unwrap(),
      "{{x -> -3}, {x -> -1}, {x -> 1}, {x -> 3}}"
    );
  }

  #[test]
  fn cyclotomic_phi3() {
    assert_eq!(
      interpret("Solve[x^2 + x + 1 == 0, x]").unwrap(),
      "{{x -> -(-1)^(1/3)}, {x -> (-1)^(2/3)}}"
    );
  }

  #[test]
  fn cyclotomic_phi6() {
    assert_eq!(
      interpret("Solve[x^2 - x + 1 == 0, x]").unwrap(),
      "{{x -> (-1)^(1/3)}, {x -> -(-1)^(2/3)}}"
    );
  }

  #[test]
  fn nonlinear_system() {
    assert_eq!(
      interpret("Solve[{3 x ^ 2 - 3 y == 0, 3 y ^ 2 - 3 x == 0}, {x, y}]")
        .unwrap(),
      "{{x -> 0, y -> 0}, {x -> 1, y -> 1}, {x -> -(-1)^(1/3), y -> (-1)^(2/3)}, {x -> (-1)^(2/3), y -> -(-1)^(1/3)}}"
    );
  }

  #[test]
  fn linear_system_with_and_in_list() {
    // Solve[{eq1 && eq2}, {x, y}] should behave like
    // Solve[{eq1, eq2}, {x, y}] — the conjunction inside the list is
    // flattened into individual equations.
    assert_eq!(
      interpret(
        "{x, y} /. Solve[{2 x + y == 12 && x + 4 y == 34}, {x, y}] // First"
      )
      .unwrap(),
      "{2, 8}"
    );
  }

  #[test]
  fn underdetermined_quadratic_cubic() {
    // Single equation with two variables: solve for lowest-degree variable
    assert_eq!(
      interpret("Solve[x^2 - y^3 == 1, {x, y}]").unwrap(),
      "{{x -> -Sqrt[1 + y^3]}, {x -> Sqrt[1 + y^3]}}"
    );
  }

  #[test]
  fn underdetermined_quintic_quadratic() {
    // Prefers y (degree 2) over x (degree 5)
    assert_eq!(
      interpret("Solve[x^5 - y^2 == 1, {x, y}]").unwrap(),
      "{{y -> -Sqrt[-1 + x^5]}, {y -> Sqrt[-1 + x^5]}}"
    );
  }

  #[test]
  fn solve_pure_cubic_symbolic() {
    assert_eq!(
      interpret("Solve[y^3 == a, y]").unwrap(),
      "{{y -> a^(1/3)}, {y -> -((-1)^(1/3)*a^(1/3))}, {y -> (-1)^(2/3)*a^(1/3)}}"
    );
  }

  #[test]
  fn solve_pure_quintic_symbolic() {
    assert_eq!(
      interpret("Solve[y^5 == a, y]").unwrap(),
      "{{y -> a^(1/5)}, {y -> -((-1)^(1/5)*a^(1/5))}, {y -> (-1)^(2/5)*a^(1/5)}, {y -> -((-1)^(3/5)*a^(1/5))}, {y -> (-1)^(4/5)*a^(1/5)}}"
    );
  }

  // Regression: a negative leading coefficient (e.g. the `c - x^2` factors of
  // x^4 - 4) must still yield simplified, correctly ordered roots — not
  // `-(-Sqrt[2])` / `I*(-Sqrt[2])` and not a flipped order.
  #[test]
  fn solve_negative_leading_coefficient() {
    // Irrational real roots.
    assert_eq!(
      interpret("Solve[2 - x^2 == 0, x]").unwrap(),
      "{{x -> -Sqrt[2]}, {x -> Sqrt[2]}}"
    );
    assert_eq!(
      interpret("Solve[3 - x^2 == 0, x]").unwrap(),
      "{{x -> -Sqrt[3]}, {x -> Sqrt[3]}}"
    );
    // Complex roots stay simplified.
    assert_eq!(
      interpret("Solve[-2 - x^2 == 0, x]").unwrap(),
      "{{x -> -I*Sqrt[2]}, {x -> I*Sqrt[2]}}"
    );
    // Perfect-square discriminant: smaller (more negative) root first.
    assert_eq!(
      interpret("Solve[6 - x - x^2 == 0, x]").unwrap(),
      "{{x -> -3}, {x -> 2}}"
    );
    // The audit case: real-domain roots of a biquadratic factor product.
    assert_eq!(
      interpret("SolveValues[(x^2 + 2)*(x^2 - 2) == 0, x, Reals]").unwrap(),
      "{-Sqrt[2], Sqrt[2]}"
    );
  }

  #[test]
  fn solve_sqrt_equation() {
    assert_eq!(interpret("Solve[Sqrt[x] == 3, x]").unwrap(), "{{x -> 9}}");
  }

  #[test]
  fn solve_sqrt_nested() {
    assert_eq!(
      interpret("Solve[Sqrt[x + 1] == 4, x]").unwrap(),
      "{{x -> 15}}"
    );
  }

  // Abs[f(x)] == c → f == c ∪ f == -c.
  #[test]
  fn solve_abs_basic() {
    assert_eq!(
      interpret("Solve[Abs[x] == 3, x]").unwrap(),
      "{{x -> -3}, {x -> 3}}"
    );
    assert_eq!(interpret("Solve[Abs[x] == 0, x]").unwrap(), "{{x -> 0}}");
    assert_eq!(interpret("Solve[Abs[x] == -1, x]").unwrap(), "{}");
  }

  #[test]
  fn solve_abs_shifted_and_scaled() {
    assert_eq!(
      interpret("Solve[Abs[x - 2] == 5, x]").unwrap(),
      "{{x -> -3}, {x -> 7}}"
    );
    assert_eq!(
      interpret("Solve[Abs[2 x] == 4, x]").unwrap(),
      "{{x -> -2}, {x -> 2}}"
    );
    assert_eq!(
      interpret("Solve[3 Abs[x] == 12, x]").unwrap(),
      "{{x -> -4}, {x -> 4}}"
    );
  }

  #[test]
  fn solve_abs_symbolic_rhs() {
    assert_eq!(
      interpret("Solve[Abs[x] == a, x]").unwrap(),
      "{{x -> -a}, {x -> a}}"
    );
  }

  #[test]
  fn solve_log_equation() {
    assert_eq!(interpret("Solve[Log[x] == 2, x]").unwrap(), "{{x -> E^2}}");
  }

  #[test]
  fn solve_exp_equation() {
    // Matches wolframscript: returns the full complex solution with
    // ConditionalExpression, covering all integer branches.
    assert_eq!(
      interpret("Solve[Exp[x] == 1, x]").unwrap(),
      "{{x -> ConditionalExpression[(2*I)*Pi*C[1], Element[C[1], Integers]]}}"
    );
  }

  #[test]
  fn solve_log_with_linear_inner() {
    // Matches wolframscript's preferred form: (-1 + E^3)/2 over
    // -((1 - E^3)/2).
    assert_eq!(
      interpret("Solve[Log[2*x + 1] == 3, x]").unwrap(),
      "{{x -> (-1 + E^3)/2}}"
    );
  }

  // wolframscript's `Solve` orders solutions lexicographically by
  // (real, imag) — `-I` and `I` slot between `0` and `1` because they
  // share real part 0. (`Root` uses a different rule that floats every
  // real to the head of the whole list.)
  #[test]
  fn solve_orders_complex_roots_by_real_part() {
    assert_eq!(
      interpret("Solve[x^5 == x, x]").unwrap(),
      "{{x -> -1}, {x -> 0}, {x -> -I}, {x -> I}, {x -> 1}}"
    );
  }

  #[test]
  fn solve_orders_pure_complex_roots() {
    // `x^4 == 1`: real ±1 split around the unit-circle complex pair.
    assert_eq!(
      interpret("Solve[x^4 == 1, x]").unwrap(),
      "{{x -> -1}, {x -> -I}, {x -> I}, {x -> 1}}"
    );
  }
}

mod rsolve {
  use super::*;

  #[test]
  fn constant_coeff_second_order() {
    assert_eq!(
      interpret("RSolve[{a[n + 2] == a[n], a[0] == 1, a[1] == 4}, a, n]")
        .unwrap(),
      "{{a -> Function[{n}, (5 - 3*(-1)^n)/2]}}"
    );
  }

  // Repeated characteristic root with initial conditions: the basis is
  // r^n, n r^n, so the linear system uses n^mult * r^n columns.
  #[test]
  fn repeated_root_double_two() {
    assert_eq!(
      interpret(
        "RSolve[{a[n] == 4 a[n-1] - 4 a[n-2], a[1] == 2, a[2] == 8}, a[n], n]"
      )
      .unwrap(),
      "{{a[n] -> 2^n*n}}"
    );
  }

  #[test]
  fn repeated_root_double_three() {
    assert_eq!(
      interpret(
        "RSolve[{a[n] == 6 a[n-1] - 9 a[n-2], a[1] == 3, a[2] == 18}, a[n], n]"
      )
      .unwrap(),
      "{{a[n] -> 3^n*n}}"
    );
  }

  // Repeated root r = 1 (1^n = 1) yields a polynomial in n.
  #[test]
  fn repeated_root_unity_linear() {
    assert_eq!(
      interpret(
        "RSolve[{a[n] == 2 a[n-1] - a[n-2], a[0] == 1, a[1] == 3}, a[n], n]"
      )
      .unwrap(),
      "{{a[n] -> 1 + 2*n}}"
    );
    assert_eq!(
      interpret(
        "RSolve[{a[n] == 2 a[n-1] - a[n-2], a[1] == 1, a[2] == 2}, a[n], n]"
      )
      .unwrap(),
      "{{a[n] -> n}}"
    );
  }

  // First-order arithmetic progression a[n] == a[n-1] + d (constant step d).
  // General solution d*n + C[1]; with one initial condition, the specific
  // value v + d*(n - k0).
  #[test]
  fn arithmetic_progression_general() {
    assert_eq!(
      interpret("RSolve[a[n] == a[n-1] + 1, a[n], n]").unwrap(),
      "{{a[n] -> n + C[1]}}"
    );
    assert_eq!(
      interpret("RSolve[a[n] == a[n-1] + 2, a[n], n]").unwrap(),
      "{{a[n] -> 2*n + C[1]}}"
    );
    assert_eq!(
      interpret("RSolve[a[n] == a[n-1] - 1, a[n], n]").unwrap(),
      "{{a[n] -> -n + C[1]}}"
    );
    // Symbolic step, and the forward-shift spelling a[n+1] == a[n] + 3.
    assert_eq!(
      interpret("RSolve[a[n] == a[n-1] + k, a[n], n]").unwrap(),
      "{{a[n] -> k*n + C[1]}}"
    );
    assert_eq!(
      interpret("RSolve[a[n+1] == a[n] + 3, a[n], n]").unwrap(),
      "{{a[n] -> 3*n + C[1]}}"
    );
  }

  #[test]
  fn arithmetic_progression_with_initial_condition() {
    assert_eq!(
      interpret("RSolve[{a[n] == a[n-1] + 1, a[0] == 0}, a[n], n]").unwrap(),
      "{{a[n] -> n}}"
    );
    assert_eq!(
      interpret("RSolve[{a[n] == a[n-1] + 2, a[1] == 3}, a[n], n]").unwrap(),
      "{{a[n] -> 1 + 2*n}}"
    );
    assert_eq!(
      interpret("RSolve[{a[n] == a[n-1] + 5, a[2] == 10}, a[n], n]").unwrap(),
      "{{a[n] -> 5*n}}"
    );
    assert_eq!(
      interpret("RSolve[{a[n] == a[n-1] + d, a[0] == c}, a[n], n]").unwrap(),
      "{{a[n] -> c + d*n}}"
    );
  }

  // Equations may be joined with `&&` instead of a list; the conjunction is
  // flattened and solved identically, matching wolframscript.
  #[test]
  fn conditions_joined_with_and() {
    assert_eq!(
      interpret("RSolve[a[n] == a[n-1] + 1 && a[1] == 1, a[n], n]").unwrap(),
      "{{a[n] -> n}}"
    );
    assert_eq!(
      interpret("RSolve[a[n] == 2 a[n-1] && a[0] == 1, a[n], n]").unwrap(),
      "{{a[n] -> 2^n}}"
    );
    // Three conjuncts: recurrence plus two initial conditions.
    assert_eq!(
      interpret(
        "RSolve[a[n] == 4 a[n-1] - 4 a[n-2] && a[1] == 2 && a[2] == 8, a[n], n]"
      )
      .unwrap(),
      "{{a[n] -> 2^n*n}}"
    );
    // RSolveValue accepts the conjunction form too (it delegates to RSolve).
    assert_eq!(
      interpret("RSolveValue[a[n] == 2 a[n-1] && a[0] == 1, a[n], n]").unwrap(),
      "2^n"
    );
  }

  // An index-dependent forcing term (a[n-1] + n) or a coefficient other than 1
  // (3 a[n-1] + 1) is not an arithmetic progression and stays unevaluated,
  // matching the deeper cases Woxi does not yet close-form.
  #[test]
  fn non_arithmetic_forcing_stays_unevaluated() {
    assert_eq!(
      interpret("RSolve[a[n] == a[n-1] + n, a[n], n]").unwrap(),
      "RSolve[a[n] == n + a[-1 + n], a[n], n]"
    );
    assert_eq!(
      interpret("RSolve[a[n] == 3 a[n-1] + 1, a[n], n]").unwrap(),
      "RSolve[a[n] == 1 + 3*a[-1 + n], a[n], n]"
    );
  }
}

mod full_simplify {
  use super::*;

  #[test]
  fn algebraic_factoring() {
    assert_eq!(
      interpret("FullSimplify[x^2 + 2*x + 1]").unwrap(),
      "(1 + x)^2"
    );
  }

  #[test]
  fn trig_identity() {
    assert_eq!(interpret("FullSimplify[Sin[x]^2 + Cos[x]^2]").unwrap(), "1");
  }

  // Denesting nested radicals: Sqrt[a + b Sqrt[c]] -> Sqrt[d] +/- Sqrt[e]
  // when a^2 - b^2 c is a perfect square.
  #[test]
  fn denest_two_surds() {
    assert_eq!(
      interpret("FullSimplify[Sqrt[5 + 2 Sqrt[6]]]").unwrap(),
      "Sqrt[2] + Sqrt[3]"
    );
  }

  #[test]
  fn denest_integer_plus_surd() {
    assert_eq!(
      interpret("FullSimplify[Sqrt[3 + 2 Sqrt[2]]]").unwrap(),
      "1 + Sqrt[2]"
    );
  }

  #[test]
  fn denest_with_coefficient_four() {
    assert_eq!(
      interpret("FullSimplify[Sqrt[7 + 4 Sqrt[3]]]").unwrap(),
      "2 + Sqrt[3]"
    );
  }

  #[test]
  fn denest_minus_sign() {
    assert_eq!(
      interpret("FullSimplify[Sqrt[7 - 2 Sqrt[10]]]").unwrap(),
      "-Sqrt[2] + Sqrt[5]"
    );
  }

  #[test]
  fn denest_sum_combines() {
    assert_eq!(
      interpret("FullSimplify[Sqrt[5 + 2 Sqrt[6]] + Sqrt[5 - 2 Sqrt[6]]]")
        .unwrap(),
      "2*Sqrt[3]"
    );
  }

  // Non-denestable radical (a^2 - b^2 c not a perfect square) stays nested.
  #[test]
  fn non_denestable_stays_nested() {
    assert_eq!(
      interpret("FullSimplify[Sqrt[5 + 2 Sqrt[5]]]").unwrap(),
      "Sqrt[5 + 2*Sqrt[5]]"
    );
  }

  // Plain Simplify must NOT denest (only FullSimplify does).
  #[test]
  fn simplify_does_not_denest() {
    assert_eq!(
      interpret("Simplify[Sqrt[3 + 2 Sqrt[2]]]").unwrap(),
      "Sqrt[3 + 2*Sqrt[2]]"
    );
  }

  // Gamma[a]/Gamma[b], a - b = k a positive integer: the rising-factorial
  // product. Leaf-count gated, so k <= 3 reduces but k = 4 keeps the ratio.
  #[test]
  fn gamma_ratio_rising_factorial() {
    assert_eq!(
      interpret("FullSimplify[Gamma[n + 1]/Gamma[n]]").unwrap(),
      "n"
    );
    assert_eq!(
      interpret("FullSimplify[Gamma[n + 2]/Gamma[n]]").unwrap(),
      "n*(1 + n)"
    );
    assert_eq!(
      interpret("FullSimplify[Gamma[n + 3]/Gamma[n]]").unwrap(),
      "n*(1 + n)*(2 + n)"
    );
    // a - b = 1 with a shifted denominator.
    assert_eq!(
      interpret("FullSimplify[Gamma[2 n]/Gamma[2 n - 1]]").unwrap(),
      "-1 + 2*n"
    );
    assert_eq!(
      interpret("FullSimplify[Gamma[x + 3]/Gamma[x + 1]]").unwrap(),
      "(1 + x)*(2 + x)"
    );
  }

  // Factorial ratios reduce like Gamma ratios (n! = Gamma[n+1]): n!/(n-k)! ->
  // rising-factorial product for small k.
  #[test]
  fn factorial_ratio_rising_factorial() {
    assert_eq!(interpret("FullSimplify[n! / (n - 1)!]").unwrap(), "n");
    assert_eq!(
      interpret("FullSimplify[n! / (n - 2)!]").unwrap(),
      "(-1 + n)*n"
    );
    assert_eq!(interpret("FullSimplify[(n + 1)! / n!]").unwrap(), "1 + n");
    assert_eq!(
      interpret("FullSimplify[(n + 2)! / n!]").unwrap(),
      "(1 + n)*(2 + n)"
    );
    // k = 3 with an all-binomial product (longer than the ratio) still
    // reduces, matching wolframscript.
    assert_eq!(
      interpret("FullSimplify[(n + 3)! / n!]").unwrap(),
      "(1 + n)*(2 + n)*(3 + n)"
    );
    // k = 4: left unevaluated.
    assert_eq!(
      interpret("FullSimplify[n! / (n - 4)!]").unwrap(),
      "n!/(-4 + n)!"
    );
  }

  // z Gamma[z] -> Gamma[z+1] (the Gamma recurrence), but only when the result
  // is no more complex than the input (matching wolframscript's LeafCount
  // comparison).
  #[test]
  fn gamma_factor_absorption() {
    assert_eq!(
      interpret("FullSimplify[x Gamma[x]]").unwrap(),
      "Gamma[1 + x]"
    );
    assert_eq!(
      interpret("FullSimplify[n Gamma[n]]").unwrap(),
      "Gamma[1 + n]"
    );
    assert_eq!(
      interpret("FullSimplify[(x + 1) Gamma[x + 1]]").unwrap(),
      "Gamma[2 + x]"
    );
    assert_eq!(
      interpret("FullSimplify[Sin[x] Gamma[Sin[x]]]").unwrap(),
      "Gamma[1 + Sin[x]]"
    );
    // A numeric coefficient makes the absorbed form longer, so it is kept.
    assert_eq!(
      interpret("FullSimplify[2 x Gamma[x]]").unwrap(),
      "2*x*Gamma[x]"
    );
    // No matching factor: unchanged.
    assert_eq!(interpret("FullSimplify[x Gamma[y]]").unwrap(), "x*Gamma[y]");
  }

  #[test]
  fn gamma_ratio_not_reduced() {
    // k = 4: the product is longer than the ratio, so the ratio is kept.
    assert_eq!(
      interpret("FullSimplify[Gamma[n + 4]/Gamma[n]]").unwrap(),
      "Gamma[4 + n]/Gamma[n]"
    );
    // Non-integer difference, different symbols, and plain Simplify: unchanged.
    assert_eq!(
      interpret("FullSimplify[Gamma[n + 1/2]/Gamma[n]]").unwrap(),
      "Gamma[1/2 + n]/Gamma[n]"
    );
    assert_eq!(
      interpret("Simplify[Gamma[n + 1]/Gamma[n]]").unwrap(),
      "Gamma[1 + n]/Gamma[n]"
    );
  }

  // ArcSin[u] + ArcCos[u] -> Pi/2 (and the ArcSec/ArcCsc pair). A
  // FullSimplify-only identity, applied only to a bare two-term sum.
  #[test]
  fn complementary_inverse_trig() {
    assert_eq!(
      interpret("FullSimplify[ArcSin[x] + ArcCos[x]]").unwrap(),
      "Pi/2"
    );
    assert_eq!(
      interpret("FullSimplify[ArcSec[x] + ArcCsc[x]]").unwrap(),
      "Pi/2"
    );
    assert_eq!(
      interpret("FullSimplify[ArcSin[2 x] + ArcCos[2 x]]").unwrap(),
      "Pi/2"
    );
    assert_eq!(
      interpret("FullSimplify[2 ArcSin[x] + 2 ArcCos[x]]").unwrap(),
      "Pi"
    );
  }

  // Must NOT reduce: ArcTan + ArcCot is +-Pi/2 by sign; an extra term blocks
  // it; and plain Simplify never applies it.
  #[test]
  fn complementary_inverse_trig_no_false_positive() {
    assert_eq!(
      interpret("FullSimplify[ArcTan[x] + ArcCot[x]]").unwrap(),
      "ArcCot[x] + ArcTan[x]"
    );
    assert_eq!(
      interpret("FullSimplify[ArcSin[x] + ArcCos[x] + z]").unwrap(),
      "z + ArcCos[x] + ArcSin[x]"
    );
    assert_eq!(
      interpret("Simplify[ArcSin[x] + ArcCos[x]]").unwrap(),
      "ArcCos[x] + ArcSin[x]"
    );
  }

  #[test]
  fn trig_ratio_cot() {
    assert_eq!(interpret("Simplify[Cos[x]/Sin[x]]").unwrap(), "Cot[x]");
  }

  #[test]
  fn trig_ratio_tan() {
    assert_eq!(interpret("Simplify[Sin[x]/Cos[x]]").unwrap(), "Tan[x]");
  }

  #[test]
  fn trig_ratio_with_arg() {
    assert_eq!(
      interpret("Simplify[Sin[2*x]/Cos[2*x]]").unwrap(),
      "Tan[2*x]"
    );
  }

  #[test]
  fn trig_with_symbolic_coefficients() {
    assert_eq!(
      interpret(
        "FullSimplify[{a^2*((-1 + Sin[theta])^2 + Cos[theta]^2), a^2*((1 + Sin[theta])^2 + Cos[theta]^2)}]"
      )
      .unwrap(),
      "{-2*a^2*(-1 + Sin[theta]), 2*a^2*(1 + Sin[theta])}"
    );
  }

  #[test]
  fn numeric_factoring() {
    assert_eq!(interpret("FullSimplify[3*x + 6]").unwrap(), "3*(2 + x)");
  }

  #[test]
  fn trivial() {
    assert_eq!(interpret("FullSimplify[5]").unwrap(), "5");
    assert_eq!(interpret("FullSimplify[x]").unwrap(), "x");
  }

  #[test]
  fn combine_like_denominator_fractions() {
    assert_eq!(interpret("FullSimplify[a/x + b/x]").unwrap(), "(a + b)/x");
  }

  #[test]
  fn abs_quotient() {
    // Abs[a]/Abs[b] → Abs[a/b] with expansion
    assert_eq!(
      interpret("FullSimplify[Abs[1 + x^3]/Abs[x]]").unwrap(),
      "Abs[x^(-1) + x^2]"
    );
  }

  #[test]
  fn abs_product() {
    // Abs[a]*Abs[b] → Abs[a*b]
    assert_eq!(
      interpret("FullSimplify[Abs[x]*Abs[y]]").unwrap(),
      "Abs[x*y]"
    );
  }

  #[test]
  fn combine_fractions_different_denominators() {
    assert_eq!(
      interpret(
        "FullSimplify[k*q/(2*a^4*(1 + s)^(3/2)) + k*q*(1 + s)^(9/4)/(2*a^4)]"
      )
      .unwrap(),
      "(k*q*(1 + (1 + s)^(15/4)))/(2*a^4*(1 + s)^(3/2))"
    );
  }

  // Regression: FullSimplify should partially factor sums whose terms split
  // into variable-disjoint groups. `1 + c^2 + 2 c d + d^2` has a constant
  // `1` plus a group connected by c,d that factors to `(c+d)^2`.
  #[test]
  fn partial_factor_disjoint_groups() {
    assert_eq!(
      interpret("FullSimplify[1 + c^2 + 2 c d + d^2]").unwrap(),
      "1 + (c + d)^2"
    );
  }

  // Regression: FullSimplify should recursively re-simplify inside a factored
  // product. After pulling `y` out of the sum, the remaining polynomial in x
  // should be collected by x with each coefficient factored in turn.
  #[test]
  fn nested_factor_after_common_factor() {
    assert_eq!(
      interpret(
        "FullSimplify[(a^2 + 2 a b + b^2) y + (2 a + 2 b) x y + (1 + c^2 + 2 c d + d^2) x^2 y]"
      )
      .unwrap(),
      "((a + b)^2 + 2*(a + b)*x + (1 + (c + d)^2)*x^2)*y"
    );
  }

  // Regression: FullSimplify should collect a multi-variable polynomial by a
  // chosen variable and factor each collected coefficient.
  #[test]
  fn collect_and_factor_coefficients() {
    assert_eq!(
      interpret(
        "FullSimplify[a^2 + 2 a b + b^2 + 2 a x + 2 b x + x^2 + c^2 x^2 + 2 c d x^2 + d^2 x^2]"
      )
      .unwrap(),
      "(a + b)^2 + 2*(a + b)*x + (1 + (c + d)^2)*x^2"
    );
  }

  #[test]
  fn simplify_conditional_expression_passthrough() {
    // Simplify[ConditionalExpression[1, a > 0]] leaves the conditional
    // intact — matches wolframscript. (Mathics returns Undefined, a
    // different design choice.)
    assert_eq!(
      interpret("Simplify[ConditionalExpression[1, a > 0]]").unwrap(),
      "ConditionalExpression[1, a > 0]"
    );
  }
}

mod simplify_assumptions {
  use super::*;

  #[test]
  fn simplify_with_assumptions_option() {
    assert_eq!(
      interpret("Simplify[x + x, Assumptions -> x > 0]").unwrap(),
      "2*x"
    );
  }

  #[test]
  fn full_simplify_with_assumptions_option() {
    assert_eq!(
      interpret("FullSimplify[x + x, Assumptions -> x > 0]").unwrap(),
      "2*x"
    );
  }

  // Regression: Simplify should accept a direct assumption (not only
  // `Assumptions -> val`), and `Assuming[...]` should propagate to a nested
  // `Simplify[...]` call via `$Assumptions`.
  #[test]
  fn simplify_with_direct_assumption() {
    assert_eq!(interpret("Simplify[Sqrt[x^2], x > 0]").unwrap(), "x");
  }

  #[test]
  fn simplify_power_one_half_with_assumption() {
    // (x^2)^(1/2) is the unevaluated Power form of Sqrt[x^2].
    assert_eq!(interpret("Simplify[(x^2)^(1/2), x > 0]").unwrap(), "x");
  }

  // Trigonometric functions at integer multiples of Pi, under an integer
  // assumption: Sin[k Pi] = 0, Tan[k Pi] = 0, Cos[k Pi] = (-1)^k. The
  // coefficient may be any ordering/arity (n Pi, Pi n, 2 Pi n, (2n+1) Pi).
  #[test]
  fn simplify_trig_at_integer_multiples_of_pi() {
    assert_eq!(
      interpret("Simplify[Sin[Pi n], Element[n, Integers]]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("Simplify[Sin[2 Pi n], Element[n, Integers]]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("Simplify[Tan[Pi n], Element[n, Integers]]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("Simplify[Cos[Pi n], Element[n, Integers]]").unwrap(),
      "(-1)^n"
    );
    assert_eq!(
      interpret("Simplify[Cos[2 Pi n], Element[n, Integers]]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Simplify[Cos[(2 n + 1) Pi], Element[n, Integers]]").unwrap(),
      "-1"
    );
  }

  // (-1)^k collapses to ±1 when the parity of the integer exponent is known.
  #[test]
  fn simplify_neg_one_integer_power() {
    assert_eq!(
      interpret("Simplify[(-1)^(2 n), Element[n, Integers]]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Simplify[(-1)^(2 n + 1), Element[n, Integers]]").unwrap(),
      "-1"
    );
    // Parity unknown: stays symbolic.
    assert_eq!(
      interpret("Simplify[(-1)^n, Element[n, Integers]]").unwrap(),
      "(-1)^n"
    );
  }

  #[test]
  fn assuming_propagates_to_simplify() {
    assert_eq!(
      interpret("Assuming[x > 0, Simplify[Sqrt[x^2]]]").unwrap(),
      "x"
    );
  }

  #[test]
  fn assuming_combines_with_inner_simplify_assumption() {
    // Outer Assuming and inner direct assumption should combine via And.
    assert_eq!(
      interpret("Assuming[x > 0, Simplify[Sqrt[x^2] + Sqrt[y^2], y > 0]]")
        .unwrap(),
      "x + y"
    );
  }

  #[test]
  fn assuming_propagates_to_simplify_multi_var() {
    assert_eq!(
      interpret("Assuming[x > 0 && y > 0, Simplify[Sqrt[x^2] + Sqrt[y^2]]]")
        .unwrap(),
      "x + y"
    );
  }

  // Regression (mathics test_assumptions.py:22): `Assuming[var == value,
  // Integrate[...]]` substitutes `var → value` in the Integrate body
  // before evaluating, so the definite integral specialises to the
  // concrete numeric result.
  #[test]
  fn assuming_eq_one_integrate_x_n() {
    assert_eq!(
      interpret("Assuming[n == 1, Integrate[x^n, {x, 0, 1}]]").unwrap(),
      "1/2"
    );
  }

  #[test]
  fn assuming_eq_two_integrate_x_n() {
    assert_eq!(
      interpret("Assuming[n == 2, Integrate[x^n, {x, 0, 1}]]").unwrap(),
      "1/3"
    );
  }

  // Substitution must only kick in when the body has an Integrate /
  // Sum / Product / Limit subexpression. A bare `x^n` keeps its
  // symbolic form (matching wolframscript).
  #[test]
  fn assuming_eq_does_not_substitute_into_bare_power() {
    assert_eq!(interpret("Assuming[n == 1, x^n]").unwrap(), "x^n");
  }
}

// Regression: Simplify should collapse nested continued-fraction-like
// expressions by combining inner fractions, not leave them in the
// `1 + (1 + x^(-1))^(-1)` form.
mod simplify_continued_fractions {
  use super::*;

  #[test]
  fn single_level_nested_inverse() {
    // 1/(1 + 1/x) → x/(1 + x)
    assert_eq!(interpret("Simplify[1/(1 + 1/x)]").unwrap(), "x/(1 + x)");
  }

  #[test]
  fn plus_with_single_level_nested_inverse() {
    // 1 + 1/(1 + 1/x) → 1 + x/(1 + x)
    assert_eq!(
      interpret("Simplify[1 + 1/(1 + 1/x)]").unwrap(),
      "1 + x/(1 + x)"
    );
  }

  #[test]
  fn two_level_nested_inverse() {
    // 1/(1 + 1/(1 + 1/x)) → (1 + x)/(1 + 2 x)
    assert_eq!(
      interpret("Simplify[1/(1 + 1/(1 + 1/x))]").unwrap(),
      "(1 + x)/(1 + 2*x)"
    );
  }

  #[test]
  fn plus_with_two_level_nested_inverse() {
    // 1 + 1/(1 + 1/(1 + 1/x)) → (2 + 3 x)/(1 + 2 x)
    assert_eq!(
      interpret("Simplify[1 + 1/(1 + 1/(1 + 1/x))]").unwrap(),
      "(2 + 3*x)/(1 + 2*x)"
    );
  }

  #[test]
  fn nested_inverse_with_symbolic_coefficients() {
    // a / (b + c/d) → a d / (c + b d)
    assert_eq!(
      interpret("Simplify[a/(b + c/d)]").unwrap(),
      "(a*d)/(c + b*d)"
    );
  }

  #[test]
  fn together_continued_fraction() {
    // Same input — Together alone should also combine fully.
    assert_eq!(
      interpret("Together[1 + 1/(1 + 1/(1 + 1/x))]").unwrap(),
      "(2 + 3*x)/(1 + 2*x)"
    );
  }
}

mod roots {
  use super::*;

  #[test]
  fn roots_linear() {
    assert_eq!(interpret("Roots[x == 5, x]").unwrap(), "x == 5");
  }

  #[test]
  fn roots_quadratic_integer() {
    let result = interpret("Roots[x^2 - 4 == 0, x]").unwrap();
    assert_eq!(result, "x == 2 || x == -2");
  }

  #[test]
  fn roots_quadratic_symbolic() {
    let result = interpret("Roots[a*x^2 + b*x + c == 0, x]").unwrap();
    assert!(result.contains("||"), "Should have two roots: {}", result);
    assert!(result.contains("Sqrt"), "Should have Sqrt: {}", result);
  }

  #[test]
  fn roots_no_solution() {
    // x^2 + 1 == 0 has complex roots
    let result = interpret("Roots[x^2 + 1 == 0, x]").unwrap();
    assert!(
      result.contains("||"),
      "Should return complex roots: {}",
      result
    );
  }

  #[test]
  fn roots_quadratic_repeated() {
    // Double root: x == 1 repeated twice (matching Wolfram behavior)
    let result = interpret("Roots[x^2 - 2*x + 1 == 0, x]").unwrap();
    assert_eq!(result, "x == 1 || x == 1");
  }

  // For a general polynomial (roots that are not negatives of each other),
  // wolframscript lists the roots in ascending order, matching Solve.
  #[test]
  fn roots_general_ascending() {
    assert_eq!(
      interpret("Roots[x^2 - 5*x + 6 == 0, x]").unwrap(),
      "x == 2 || x == 3"
    );
    assert_eq!(
      interpret("Roots[x^2 + x - 6 == 0, x]").unwrap(),
      "x == -3 || x == 2"
    );
    assert_eq!(
      interpret("Roots[x^3 - 6*x^2 + 11*x - 6 == 0, x]").unwrap(),
      "x == 1 || x == 2 || x == 3"
    );
    assert_eq!(
      interpret("Roots[x^4 - 5*x^2 + 4 == 0, x]").unwrap(),
      "x == -2 || x == -1 || x == 1 || x == 2"
    );
  }

  // A pure quadratic x^2 == c has roots ±r summing to zero; wolframscript
  // lists the principal `+` root first (3 || -3, Sqrt[2] || -Sqrt[2], I || -I).
  #[test]
  fn roots_pure_quadratic_positive_first() {
    assert_eq!(
      interpret("Roots[x^2 - 9 == 0, x]").unwrap(),
      "x == 3 || x == -3"
    );
    assert_eq!(
      interpret("Roots[x^2 - 2 == 0, x]").unwrap(),
      "x == Sqrt[2] || x == -Sqrt[2]"
    );
    assert_eq!(
      interpret("Roots[x^2 + 1 == 0, x]").unwrap(),
      "x == I || x == -I"
    );
  }

  // Complex conjugate roots from a quadratic with a linear term keep Solve's
  // order (the `1 - 2 I` branch first).
  #[test]
  fn roots_complex_conjugates() {
    assert_eq!(
      interpret("Roots[x^2 - 2*x + 5 == 0, x]").unwrap(),
      "x == 1 - 2*I || x == 1 + 2*I"
    );
  }
}

mod nroots {
  use super::*;

  // Helper to extract numeric value (Re or Im) from a "k.kkk" or "k.kkk*I" or
  // "k.kkk + k.kkk*I" string. Returns (re, im).
  fn parse_root(s: &str) -> (f64, f64) {
    let s = s.trim();
    if let Some(stripped) = s.strip_suffix("*I") {
      if let Some(idx) = stripped.rfind(" + ") {
        return (
          stripped[..idx].parse().unwrap(),
          stripped[idx + 3..].parse().unwrap(),
        );
      } else if let Some(idx) = stripped.rfind(" - ") {
        return (
          stripped[..idx].parse().unwrap(),
          -stripped[idx + 3..].parse::<f64>().unwrap(),
        );
      } else {
        return (0.0, stripped.parse().unwrap());
      }
    }
    (s.parse().unwrap(), 0.0)
  }

  #[test]
  fn nroots_linear() {
    assert_eq!(interpret("NRoots[x - 2 == 0, x]").unwrap(), "x == 2.");
  }

  #[test]
  fn nroots_quadratic_real() {
    // x^2 - 2 == 0 → ±Sqrt[2]
    assert_eq!(
      interpret("NRoots[x^2 - 2 == 0, x]").unwrap(),
      "x == -1.4142135623730951 || x == 1.4142135623730951"
    );
  }

  #[test]
  fn nroots_quadratic_imag() {
    // x^2 + 1 == 0 → ±I
    assert_eq!(
      interpret("NRoots[x^2 + 1 == 0, x]").unwrap(),
      "x == 0. - 1.*I || x == 0. + 1.*I"
    );
  }

  #[test]
  fn nroots_cubic_audit_case() {
    // Audit case: 1 + 2x + 3x^2 + 4x^3 = 0
    let result = interpret("NRoots[1 + 2*x + 3*x^2 + 4*x^3 == 0, x]").unwrap();
    let parts: Vec<&str> = result.split(" || ").collect();
    assert_eq!(parts.len(), 3);
    let roots: Vec<(f64, f64)> = parts
      .iter()
      .map(|p| {
        let s = p.strip_prefix("x == ").unwrap();
        parse_root(s)
      })
      .collect();
    // Real root then two complex conjugates, sorted by (re, im).
    let expected = [
      (-0.605829586188268_f64, 0.0_f64),
      (-0.07208520690586598, -0.6383267351483765),
      (-0.07208520690586598, 0.6383267351483765),
    ];
    for (i, (er, ei)) in expected.iter().enumerate() {
      assert!((roots[i].0 - er).abs() < 1e-9, "Re mismatch at {}", i);
      assert!((roots[i].1 - ei).abs() < 1e-9, "Im mismatch at {}", i);
    }
  }

  #[test]
  fn nroots_cubic_unity() {
    // x^3 - 1 == 0: complex roots first (real -0.5), then real root 1.
    let result = interpret("NRoots[x^3 - 1 == 0, x]").unwrap();
    let parts: Vec<&str> = result.split(" || ").collect();
    assert_eq!(parts.len(), 3);
    let roots: Vec<(f64, f64)> = parts
      .iter()
      .map(|p| {
        let s = p.strip_prefix("x == ").unwrap();
        parse_root(s)
      })
      .collect();
    let expected = [
      (-0.5, -0.8660254037844386),
      (-0.5, 0.8660254037844386),
      (1.0, 0.0),
    ];
    for (i, (er, ei)) in expected.iter().enumerate() {
      assert!((roots[i].0 - er).abs() < 1e-9, "Re mismatch at {}", i);
      assert!((roots[i].1 - ei).abs() < 1e-9, "Im mismatch at {}", i);
    }
  }
}

mod eliminate {
  use super::*;

  #[test]
  fn eliminate_single_variable_linear() {
    // Eliminate y from {x == 2 + y, y == z}
    let result = interpret("Eliminate[{x == 2 + y, y == z}, y]").unwrap();
    assert_eq!(result, "2 + z == x");
  }

  #[test]
  fn eliminate_single_from_two_linear() {
    // Eliminate a from {x == a + b, y == a - b}
    let result = interpret("Eliminate[{x == a + b, y == a - b}, a]").unwrap();
    assert_eq!(result, "x - y == 2*b");
  }

  // Solving an equation with a fractional coefficient for the eliminated
  // variable simplifies the quotient (t = 2*x), instead of leaving a nested
  // fraction like -(x/-1/2).
  #[test]
  fn eliminate_fractional_coefficient() {
    let result = interpret("Eliminate[{x == t/2, y == t}, t]").unwrap();
    assert_eq!(result, "y == 2*x");
  }

  #[test]
  fn eliminate_to_constant() {
    // Eliminate y from {x + y == 3, x - y == 1}
    let result = interpret("Eliminate[{x + y == 3, x - y == 1}, y]").unwrap();
    assert_eq!(result, "x == 2");
  }

  #[test]
  fn eliminate_with_product() {
    // Eliminate x from {a == x + y, b == x*y}
    let result = interpret("Eliminate[{a == x + y, b == x*y}, x]").unwrap();
    assert_eq!(result, "-b + a*y - y^2 == 0");
  }

  #[test]
  fn eliminate_variable_not_found() {
    // If variable doesn't appear, equation is returned unchanged
    let result = interpret("Eliminate[{x == 2}, y]").unwrap();
    assert_eq!(result, "x == 2");
  }

  #[test]
  fn eliminate_single_equation() {
    // With a single equation and the variable in it, eliminating gives True
    let result = interpret("Eliminate[{x == 2}, x]").unwrap();
    assert_eq!(result, "True");
  }
}

mod to_rules {
  use super::*;

  #[test]
  fn to_rules_single_equation() {
    // x == 5 → {x -> 5}
    assert_eq!(interpret("ToRules[x == 5]").unwrap(), "{x -> 5}");
  }

  #[test]
  fn to_rules_or_conditions() {
    // x == -2 || x == 2 → Sequence[{x -> -2}, {x -> 2}] (displays as "{x -> -2}{x -> 2}")
    assert_eq!(
      interpret("ToRules[x == -2 || x == 2]").unwrap(),
      "{x -> -2}{x -> 2}"
    );
  }

  #[test]
  fn to_rules_from_roots() {
    // Convert Roots output to Solve-style rules
    assert_eq!(
      interpret("ToRules[Roots[x^2 - 4 == 0, x]]").unwrap(),
      "{x -> 2}{x -> -2}"
    );
  }

  #[test]
  fn to_rules_and_conditions() {
    // x == 1 && y == 2 → {x -> 1, y -> 2}
    assert_eq!(
      interpret("ToRules[x == 1 && y == 2]").unwrap(),
      "{x -> 1, y -> 2}"
    );
  }

  #[test]
  fn to_rules_true() {
    assert_eq!(interpret("ToRules[True]").unwrap(), "{}");
  }

  #[test]
  fn to_rules_false() {
    // ToRules[False] returns Sequence[] (empty, matches Wolfram behavior)
    // When wrapped in ToString[(ToRules[False]), InputForm] → "InputForm"
    assert_eq!(interpret("ToRules[False]").unwrap(), "");
  }
}

mod reduce {
  use super::*;

  // ── Trivial cases ──

  #[test]
  fn reduce_true() {
    assert_eq!(interpret("Reduce[True, x]").unwrap(), "True");
  }

  // Reduce of a trig equation over a bounded interval gives the concrete
  // in-range roots as a disjunction (or a single equation / False).
  #[test]
  fn trig_bounded_sin_zero() {
    assert_eq!(
      interpret("Reduce[Sin[x] == 0 && 0 < x < 2 Pi, x]").unwrap(),
      "x == Pi"
    );
  }

  #[test]
  fn trig_bounded_cos_half() {
    assert_eq!(
      interpret("Reduce[Cos[x] == 1/2 && 0 < x < 2 Pi, x]").unwrap(),
      "x == Pi/3 || x == (5*Pi)/3"
    );
  }

  #[test]
  fn trig_bounded_tan_one() {
    assert_eq!(
      interpret("Reduce[Tan[x] == 1 && 0 < x < 2 Pi, x]").unwrap(),
      "x == Pi/4 || x == (5*Pi)/4"
    );
  }

  #[test]
  fn trig_bounded_no_solution() {
    assert_eq!(
      interpret("Reduce[Cos[x] == 5 && 0 < x < 2 Pi, x]").unwrap(),
      "False"
    );
  }

  // Without a bounding constraint the general periodic family is kept.
  #[test]
  fn trig_unbounded_keeps_family() {
    assert_eq!(
      interpret("Reduce[Cos[x] == 1/2, x]").unwrap(),
      "x == ConditionalExpression[-1/3*Pi + 2*Pi*C[1], \
       Element[C[1], Integers]] || x == ConditionalExpression[Pi/3 + \
       2*Pi*C[1], Element[C[1], Integers]]"
    );
  }

  #[test]
  fn reduce_exists_quadratic_linear_audit_case() {
    // Audit case: find conditions on `a` such that some (x, y) satisfies
    // x² + a·y² ≤ 1 ∧ x − y ≥ 2. Lagrange max of (x − y) over the
    // ellipse-or-strip is `sqrt(1 + 1/a)` for a > 0 and unbounded for
    // a ≤ 0, so the system is satisfiable iff `a <= 1/3`.
    assert_eq!(
      interpret("Reduce[Exists[{x, y}, x^2 + a*y^2 <= 1 && x - y >= 2], a]")
        .unwrap(),
      "a <= 1/3"
    );
  }

  #[test]
  fn reduce_exists_quadratic_linear_unit_circle_lower_bound() {
    // Same shape but tighter linear bound: x − y ≥ 1 (instead of 2).
    // 1/a ≥ 1²/1 - 1 = 0, so a > 0 always works and a ≤ 0 trivially
    // works as well — the system is satisfiable for every real `a`.
    assert_eq!(
      interpret("Reduce[Exists[{x, y}, x^2 + a*y^2 <= 1 && x - y >= 1], a]")
        .unwrap(),
      "True"
    );
  }

  #[test]
  fn reduce_false() {
    assert_eq!(interpret("Reduce[False, x]").unwrap(), "False");
  }

  // A chained two-sided numeric bound reduces to the canonical Inequality
  // form, just like the equivalent And-of-inequalities.
  #[test]
  fn reduce_chained_inequality_numeric() {
    assert_eq!(
      interpret("Reduce[0 < x < 5, x]").unwrap(),
      "Inequality[0, Less, x, Less, 5]"
    );
    assert_eq!(
      interpret("Reduce[1 <= x <= 3, x]").unwrap(),
      "Inequality[1, LessEqual, x, LessEqual, 3]"
    );
    // Mixed strictness parses as an Inequality[...] and must also reduce.
    assert_eq!(
      interpret("Reduce[0 < x <= 10, x]").unwrap(),
      "Inequality[0, Less, x, LessEqual, 10]"
    );
    assert_eq!(
      interpret("Reduce[-2 <= x < 7, x]").unwrap(),
      "Inequality[-2, LessEqual, x, Less, 7]"
    );
  }

  #[test]
  fn reduce_chained_inequality_empty() {
    // An impossible two-sided bound reduces to False.
    assert_eq!(interpret("Reduce[5 < x < 1, x]").unwrap(), "False");
  }

  // Reduce[Abs[f] op c, x, Reals] splits the absolute value into the
  // corresponding real intervals (previously these returned False/garbage).
  #[test]
  fn reduce_abs_inequality_reals() {
    assert_eq!(
      interpret("Reduce[Abs[x] < 3, x, Reals]").unwrap(),
      "Inequality[-3, Less, x, Less, 3]"
    );
    assert_eq!(
      interpret("Reduce[Abs[x] <= 3, x, Reals]").unwrap(),
      "Inequality[-3, LessEqual, x, LessEqual, 3]"
    );
    assert_eq!(
      interpret("Reduce[Abs[x] > 2, x, Reals]").unwrap(),
      "x < -2 || x > 2"
    );
    assert_eq!(
      interpret("Reduce[Abs[x] >= 2, x, Reals]").unwrap(),
      "x <= -2 || x >= 2"
    );
    // Shifted argument: -1 < x < 3.
    assert_eq!(
      interpret("Reduce[Abs[x - 1] < 2, x, Reals]").unwrap(),
      "Inequality[-1, Less, x, Less, 3]"
    );
    // A constant coefficient (Abs[2 x] evaluates to 2 Abs[x]) divides through.
    assert_eq!(
      interpret("Reduce[Abs[2 x] < 6, x, Reals]").unwrap(),
      "Inequality[-3, Less, x, Less, 3]"
    );
    // The bound flips with a negative coefficient.
    assert_eq!(
      interpret("Reduce[-2 Abs[x] > -6, x, Reals]").unwrap(),
      "Inequality[-3, Less, x, Less, 3]"
    );
    // Quadratic argument is solved on the bare polynomial.
    assert_eq!(
      interpret("Reduce[Abs[x^2 - 1] < 3, x, Reals]").unwrap(),
      "Inequality[-2, Less, x, Less, 2]"
    );
  }

  #[test]
  fn reduce_abs_inequality_boundary_reals() {
    // c <= 0 boundaries.
    assert_eq!(interpret("Reduce[Abs[x] < 0, x, Reals]").unwrap(), "False");
    assert_eq!(
      interpret("Reduce[Abs[x] <= 0, x, Reals]").unwrap(),
      "x == 0"
    );
    assert_eq!(
      interpret("Reduce[Abs[x] > 0, x, Reals]").unwrap(),
      "x < 0 || x > 0"
    );
    assert_eq!(interpret("Reduce[Abs[x] >= 0, x, Reals]").unwrap(), "True");
    assert_eq!(interpret("Reduce[Abs[x] > -1, x, Reals]").unwrap(), "True");
  }

  #[test]
  fn reduce_abs_not_equal_reals() {
    assert_eq!(
      interpret("Reduce[Abs[x] != 2, x, Reals]").unwrap(),
      "x < -2 || Inequality[-2, Less, x, Less, 2] || x > 2"
    );
    assert_eq!(
      interpret("Reduce[Abs[x] != 0, x, Reals]").unwrap(),
      "x < 0 || x > 0"
    );
    // c < 0: every real satisfies it.
    assert_eq!(interpret("Reduce[Abs[x] != -1, x, Reals]").unwrap(), "True");
  }

  // Reduce[..., Modulus -> n] enumerates solutions in Z/nZ.
  #[test]
  fn reduce_modulus_one_var_quadratic() {
    assert_eq!(
      interpret("Reduce[x^2 == 1, x, Modulus -> 5]").unwrap(),
      "x == 1 || x == 4"
    );
  }

  #[test]
  fn reduce_modulus_one_var_quadratic_mod4() {
    assert_eq!(
      interpret("Reduce[x^2 == 1, x, Modulus -> 4]").unwrap(),
      "x == 1 || x == 3"
    );
  }

  // Two-variable polynomial mod 4 (the audit case).
  #[test]
  fn reduce_modulus_two_vars() {
    assert_eq!(
      interpret("Reduce[x^5 == y^4 + x*y + 1, {x, y}, Modulus -> 4]").unwrap(),
      "(x == 1 && y == 0) || (x == 1 && y == 3) || (x == 2 && y == 1) || \
       (x == 2 && y == 3) || (x == 3 && y == 2) || (x == 3 && y == 3)"
    );
  }

  // No solutions returns False.
  #[test]
  fn reduce_modulus_no_solutions() {
    assert_eq!(
      interpret("Reduce[x^2 == 2, x, Modulus -> 4]").unwrap(),
      "False"
    );
  }

  // ── Linear equations ──

  #[test]
  fn linear_equation() {
    assert_eq!(interpret("Reduce[2*x + 3 == 7, x]").unwrap(), "x == 2");
  }

  #[test]
  fn linear_equation_negative() {
    assert_eq!(interpret("Reduce[x - 5 == 0, x]").unwrap(), "x == 5");
  }

  #[test]
  fn trivial_equation() {
    assert_eq!(interpret("Reduce[x == 5, x]").unwrap(), "x == 5");
  }

  // ── Quadratic equations ──

  #[test]
  fn quadratic_integer_roots() {
    assert_eq!(
      interpret("Reduce[x^2 - 4 == 0, x]").unwrap(),
      "x == -2 || x == 2"
    );
  }

  #[test]
  fn quadratic_two_roots() {
    assert_eq!(
      interpret("Reduce[x^2 + x - 6 == 0, x]").unwrap(),
      "x == -3 || x == 2"
    );
  }

  #[test]
  fn quadratic_repeated_root() {
    assert_eq!(
      interpret("Reduce[x^2 + 2*x + 1 == 0, x]").unwrap(),
      "x == -1"
    );
  }

  #[test]
  fn quadratic_irrational_roots() {
    assert_eq!(
      interpret("Reduce[x^2 - 3 == 0, x]").unwrap(),
      "x == -Sqrt[3] || x == Sqrt[3]"
    );
  }

  #[test]
  fn quadratic_complex_roots() {
    assert_eq!(
      interpret("Reduce[x^2 + 1 == 0, x]").unwrap(),
      "x == -I || x == I"
    );
  }

  // ── Higher degree equations (via factoring) ──

  #[test]
  fn cubic_equation() {
    assert_eq!(
      interpret("Reduce[x^3 - 3*x^2 + 2*x == 0, x]").unwrap(),
      "x == 0 || x == 1 || x == 2"
    );
  }

  #[test]
  fn cubic_factored() {
    assert_eq!(
      interpret("Reduce[(x - 1)*(x - 2)*(x - 3) == 0, x]").unwrap(),
      "x == 1 || x == 2 || x == 3"
    );
  }

  #[test]
  fn quartic_factored() {
    assert_eq!(
      interpret("Reduce[x*(x - 1)*(x - 2)*(x - 3) == 0, x]").unwrap(),
      "x == 0 || x == 1 || x == 2 || x == 3"
    );
  }

  // ── Domain filtering ──

  #[test]
  fn complex_roots_over_reals() {
    assert_eq!(
      interpret("Reduce[x^2 + 1 == 0, x, Reals]").unwrap(),
      "False"
    );
  }

  #[test]
  fn complex_roots_over_integers() {
    assert_eq!(
      interpret("Reduce[x^2 + 1 == 0, x, Integers]").unwrap(),
      "False"
    );
  }

  #[test]
  fn real_roots_over_reals() {
    assert_eq!(
      interpret("Reduce[x^2 - 1 == 0, x, Reals]").unwrap(),
      "x == -1 || x == 1"
    );
  }

  // ── Or (disjunction) ──

  #[test]
  fn reduce_or() {
    assert_eq!(
      interpret("Reduce[x == 1 || x == 2, x]").unwrap(),
      "x == 1 || x == 2"
    );
  }

  // ── Simple inequalities ──

  #[test]
  fn simple_inequality_gt() {
    assert_eq!(interpret("Reduce[x > 0, x]").unwrap(), "x > 0");
  }

  // ── Quadratic inequalities ──

  #[test]
  fn quadratic_inequality_less() {
    assert_eq!(
      interpret("Reduce[x^2 < 4, x]").unwrap(),
      "Inequality[-2, Less, x, Less, 2]"
    );
  }

  #[test]
  fn quadratic_inequality_greater() {
    assert_eq!(
      interpret("Reduce[x^2 - 1 > 0, x]").unwrap(),
      "x < -1 || x > 1"
    );
  }

  #[test]
  fn quadratic_inequality_geq() {
    assert_eq!(
      interpret("Reduce[x^2 - 1 >= 0, x]").unwrap(),
      "x <= -1 || x >= 1"
    );
  }

  #[test]
  fn factored_inequality_less() {
    assert_eq!(
      interpret("Reduce[(x - 1)*(x + 2) < 0, x]").unwrap(),
      "Inequality[-2, Less, x, Less, 1]"
    );
  }

  #[test]
  fn factored_inequality_leq() {
    assert_eq!(
      interpret("Reduce[(x - 1)*(x + 2) <= 0, x]").unwrap(),
      "Inequality[-2, LessEqual, x, LessEqual, 1]"
    );
  }

  #[test]
  fn factored_inequality_greater() {
    assert_eq!(
      interpret("Reduce[(x - 1)*(x + 2) > 0, x]").unwrap(),
      "x < -2 || x > 1"
    );
  }

  // ── Always true / always false inequalities ──

  #[test]
  fn always_true_inequality() {
    assert_eq!(
      interpret("Reduce[x^2 + 1 > 0, x]").unwrap(),
      "Element[x, Reals]"
    );
  }

  #[test]
  fn always_true_with_reals_domain() {
    assert_eq!(interpret("Reduce[x^2 + 1 > 0, x, Reals]").unwrap(), "True");
  }

  #[test]
  fn always_false_inequality() {
    assert_eq!(interpret("Reduce[x^2 + 1 < 0, x]").unwrap(), "False");
  }

  // ── And (conjunction) ──

  #[test]
  fn equation_with_inequality_constraint() {
    assert_eq!(interpret("Reduce[x^2 == 9 && x > 0, x]").unwrap(), "x == 3");
  }

  #[test]
  fn combined_inequalities() {
    assert_eq!(
      interpret("Reduce[x > 0 && x < 5 && x > 3, x]").unwrap(),
      "Inequality[3, Less, x, Less, 5]"
    );
  }

  #[test]
  fn combined_two_bounds() {
    assert_eq!(
      interpret("Reduce[x > 2 && x < 10, x]").unwrap(),
      "Inequality[2, Less, x, Less, 10]"
    );
  }

  #[test]
  fn mixed_equation_inequality() {
    assert_eq!(
      interpret("Reduce[x^2 <= 4 && x > 0, x]").unwrap(),
      "Inequality[0, Less, x, LessEqual, 2]"
    );
  }

  // Over the integers, a bounded two-sided interval enumerates the contained
  // integers (respecting strict vs non-strict bounds); without a domain it
  // stays an Inequality.
  #[test]
  fn integers_bounded_interval_strict() {
    assert_eq!(
      interpret("Reduce[x > 2 && x < 5, x, Integers]").unwrap(),
      "x == 3 || x == 4"
    );
  }

  #[test]
  fn integers_bounded_interval_inclusive() {
    assert_eq!(
      interpret("Reduce[0 <= x <= 3, x, Integers]").unwrap(),
      "x == 0 || x == 1 || x == 2 || x == 3"
    );
  }

  #[test]
  fn integers_bounded_interval_mixed_strictness() {
    assert_eq!(
      interpret("Reduce[1 < x <= 4, x, Integers]").unwrap(),
      "x == 2 || x == 3 || x == 4"
    );
    assert_eq!(
      interpret("Reduce[x >= 2 && x < 6, x, Integers]").unwrap(),
      "x == 2 || x == 3 || x == 4 || x == 5"
    );
  }

  #[test]
  fn integers_bounded_interval_empty() {
    assert_eq!(
      interpret("Reduce[2 < x < 3, x, Integers]").unwrap(),
      "False"
    );
  }

  #[test]
  fn integers_bounded_interval_single() {
    assert_eq!(
      interpret("Reduce[2 <= x <= 2, x, Integers]").unwrap(),
      "x == 2"
    );
  }

  #[test]
  fn integers_bounded_interval_no_domain_stays_inequality() {
    // Without the Integers domain the interval form is retained.
    assert_eq!(
      interpret("Reduce[x > 2 && x < 5, x]").unwrap(),
      "Inequality[2, Less, x, Less, 5]"
    );
  }

  // A single one-sided bound over the integers is tightened to the nearest
  // admissible integer and paired with the domain-membership conjunct, matching
  // wolframscript: Reduce[x > 2, x, Integers] -> Element[x, Integers] && x >= 3.
  #[test]
  fn integers_one_sided_strict_lower() {
    assert_eq!(
      interpret("Reduce[x > 2, x, Integers]").unwrap(),
      "Element[x, Integers] && x >= 3"
    );
  }

  #[test]
  fn integers_one_sided_inclusive_lower() {
    assert_eq!(
      interpret("Reduce[x >= 2, x, Integers]").unwrap(),
      "Element[x, Integers] && x >= 2"
    );
  }

  #[test]
  fn integers_one_sided_strict_upper() {
    assert_eq!(
      interpret("Reduce[x < 5, x, Integers]").unwrap(),
      "Element[x, Integers] && x <= 4"
    );
  }

  #[test]
  fn integers_one_sided_inclusive_upper() {
    assert_eq!(
      interpret("Reduce[x <= 5, x, Integers]").unwrap(),
      "Element[x, Integers] && x <= 5"
    );
  }

  #[test]
  fn integers_one_sided_negative_bound() {
    assert_eq!(
      interpret("Reduce[x < -3, x, Integers]").unwrap(),
      "Element[x, Integers] && x <= -4"
    );
  }

  // A non-integer rational bound is rounded inward to the nearest integer; an
  // integer coefficient on the variable is divided out first.
  #[test]
  fn integers_one_sided_rational_bound() {
    assert_eq!(
      interpret("Reduce[x > 5/2, x, Integers]").unwrap(),
      "Element[x, Integers] && x >= 3"
    );
    assert_eq!(
      interpret("Reduce[2 x > 5, x, Integers]").unwrap(),
      "Element[x, Integers] && x >= 3"
    );
  }

  // Without the Integers domain a one-sided bound is left as the bare
  // comparison.
  #[test]
  fn integers_one_sided_no_domain_stays_comparison() {
    assert_eq!(interpret("Reduce[x > 2, x]").unwrap(), "x > 2");
  }

  // ── Reduce InputForm: chained inequalities use Inequality[] head ──

  #[test]
  fn reduce_quadratic_inequality_input_form() {
    assert_eq!(
      interpret("ToString[Reduce[x^2 < 4, x], InputForm]").unwrap(),
      "Inequality[-2, Less, x, Less, 2]"
    );
  }

  #[test]
  fn reduce_factored_inequality_input_form() {
    assert_eq!(
      interpret("ToString[Reduce[(x - 1)*(x + 2) < 0, x], InputForm]").unwrap(),
      "Inequality[-2, Less, x, Less, 1]"
    );
  }

  #[test]
  fn reduce_factored_inequality_leq_input_form() {
    assert_eq!(
      interpret("ToString[Reduce[(x - 1)*(x + 2) <= 0, x], InputForm]")
        .unwrap(),
      "Inequality[-2, LessEqual, x, LessEqual, 1]"
    );
  }

  #[test]
  fn reduce_combined_inequalities_input_form() {
    assert_eq!(
      interpret("ToString[Reduce[x > 0 && x < 5 && x > 3, x], InputForm]")
        .unwrap(),
      "Inequality[3, Less, x, Less, 5]"
    );
  }

  #[test]
  fn reduce_two_bounds_input_form() {
    assert_eq!(
      interpret("ToString[Reduce[x > 2 && x < 10, x], InputForm]").unwrap(),
      "Inequality[2, Less, x, Less, 10]"
    );
  }

  // ── Multi-variable systems ──

  #[test]
  fn two_variable_linear_system() {
    assert_eq!(
      interpret("Reduce[x + y == 5 && x - y == 1, {x, y}]").unwrap(),
      "x == 3 && y == 2"
    );
  }

  #[test]
  fn two_variable_list_input() {
    assert_eq!(
      interpret("Reduce[{x + y == 5, x - y == 1}, {x, y}]").unwrap(),
      "x == 3 && y == 2"
    );
  }

  // ── Multi-variable nonlinear ──

  #[test]
  fn reduce_two_var_nonlinear() {
    assert_eq!(
      interpret("Reduce[x^2 - y^3 == 1, {x, y}]").unwrap(),
      "y == (-1 + x^2)^(1/3) || y == -((-1)^(1/3)*(-1 + x^2)^(1/3)) || y == (-1)^(2/3)*(-1 + x^2)^(1/3)"
    );
  }

  #[test]
  fn reduce_two_var_solve_for_lower_degree() {
    assert_eq!(
      interpret("Reduce[x^5 - y^2 == 1, {x, y}]").unwrap(),
      "y == -Sqrt[-1 + x^5] || y == Sqrt[-1 + x^5]"
    );
  }

  // ── Cubic with expanded polynomial ──

  #[test]
  fn cubic_expanded() {
    assert_eq!(
      interpret("Reduce[x^3 - 6 x^2 + 11 x - 6 == 0, x]").unwrap(),
      "x == 1 || x == 2 || x == 3"
    );
  }

  #[test]
  fn quadratic_ineq_with_sign_constraint_negative() {
    // x^2 > 4 && x < 0  →  x < -2
    assert_eq!(interpret("Reduce[x^2 > 4 && x < 0, x]").unwrap(), "x < -2");
  }

  #[test]
  fn quadratic_ineq_with_sign_constraint_positive() {
    // x^2 > 4 && x > 0  →  x > 2
    assert_eq!(interpret("Reduce[x^2 > 4 && x > 0, x]").unwrap(), "x > 2");
  }

  #[test]
  fn quadratic_ineq_with_range_constraint() {
    // x^2 > 4 && x > 0 && x < 10  →  2 < x < 10
    assert_eq!(
      interpret("Reduce[x^2 > 4 && x > 0 && x < 10, x]").unwrap(),
      "Inequality[2, Less, x, Less, 10]"
    );
  }

  // ── Inverse trig (degrees) ──

  #[test]
  fn arc_cos_degrees_greater_than_60() {
    // arccos(x) > 60° iff -1 <= x < cos(60°) = 1/2.
    assert_eq!(
      interpret("Reduce[ArcCosDegrees[x] > 60, x]").unwrap(),
      "Inequality[-1, LessEqual, x, Less, 1/2]"
    );
  }

  #[test]
  fn arc_sin_degrees_greater_than_60() {
    // arcsin(x) > 60° iff sin(60°) = Sqrt[3]/2 < x <= 1.
    assert_eq!(
      interpret("Reduce[ArcSinDegrees[x] > 60, x]").unwrap(),
      "Inequality[Sqrt[3]/2, Less, x, LessEqual, 1]"
    );
  }

  #[test]
  fn arc_tan_degrees_greater_than_60() {
    // arctan(x) > 60° iff x > tan(60°) = Sqrt[3].
    assert_eq!(
      interpret("Reduce[ArcTanDegrees[x] > 60, x]").unwrap(),
      "x > Sqrt[3]"
    );
  }

  #[test]
  fn arc_cot_degrees_greater_than_60() {
    // arccot(x) > 60° iff 0 <= x < cot(60°) = 1/Sqrt[3].
    assert_eq!(
      interpret("Reduce[ArcCotDegrees[x] > 60, x]").unwrap(),
      "Inequality[0, LessEqual, x, Less, 1/Sqrt[3]]"
    );
  }

  #[test]
  fn arc_csc_degrees_greater_than_60() {
    // arccsc(x) > 60° iff 1 <= x < csc(60°) = 2/Sqrt[3].
    assert_eq!(
      interpret("Reduce[ArcCscDegrees[x] > 60, x]").unwrap(),
      "Inequality[1, LessEqual, x, Less, 2/Sqrt[3]]"
    );
  }

  #[test]
  fn arc_sec_degrees_greater_than_60() {
    // arcsec(x) > 60° iff x > 2 || x <= -1.
    assert_eq!(
      interpret("Reduce[ArcSecDegrees[x] > 60, x]").unwrap(),
      "x > 2 || x <= -1"
    );
  }
}

mod nsolve {
  use super::*;

  #[test]
  fn linear_equation() {
    assert_eq!(interpret("NSolve[x - 5 == 0, x]").unwrap(), "{{x -> 5.}}");
  }

  #[test]
  fn quadratic_integer_roots() {
    assert_eq!(
      interpret("NSolve[x^2 - 4 == 0, x]").unwrap(),
      "{{x -> -2.}, {x -> 2.}}"
    );
  }

  #[test]
  fn quadratic_irrational_roots() {
    assert_eq!(
      interpret("NSolve[x^2 + x - 1 == 0, x]").unwrap(),
      "{{x -> -1.618033988749895}, {x -> 0.6180339887498948}}"
    );
  }

  #[test]
  fn quadratic_complex_roots() {
    assert_eq!(
      interpret("NSolve[x^2 + 1 == 0, x]").unwrap(),
      "{{x -> 0. - 1.*I}, {x -> 0. + 1.*I}}"
    );
  }

  #[test]
  fn cubic_roots() {
    assert_eq!(
      interpret("NSolve[x^3 - 3*x^2 + 2*x == 0, x]").unwrap(),
      "{{x -> 0.}, {x -> 1.}, {x -> 2.}}"
    );
  }

  #[test]
  fn roots_ordered_by_real_then_imaginary_part() {
    // Regression: roots must be ordered by ascending real part, then ascending
    // imaginary part (matching wolframscript) rather than inheriting symbolic
    // Solve's order. Previously the real cube root sorted first.
    assert_eq!(
      interpret("NSolve[x^3 - 2 == 0, x]").unwrap(),
      "{{x -> -0.6299605249474366 - 1.0911236359717214*I}, \
       {x -> -0.6299605249474366 + 1.0911236359717214*I}, \
       {x -> 1.2599210498948732}}"
    );
    // Real root between two complex pairs stays correctly placed.
    assert_eq!(
      interpret("NSolve[x^4 - 1 == 0, x]").unwrap(),
      "{{x -> -1.}, {x -> 0. - 1.*I}, {x -> 0. + 1.*I}, {x -> 1.}}"
    );
  }

  #[test]
  fn with_user_defined_function() {
    assert_eq!(
      interpret("f[x_] := x^2 + x + 1; NSolve[f[b] - 2 == 0, b]").unwrap(),
      "{{b -> -1.618033988749895}, {b -> 0.6180339887498948}}"
    );
  }

  #[test]
  fn rational_solution() {
    assert_eq!(
      interpret("NSolve[2*x - 1 == 0, x]").unwrap(),
      "{{x -> 0.5}}"
    );
  }

  #[test]
  fn cube_roots_of_unity_fully_numericized() {
    // Regression: Solve returns the complex roots of x^3 == 1 as radical
    // forms (1, -(-1)^(1/3), (-1)^(2/3)). NSolve previously left the
    // Times[-1, Power[-1, 1/3]] root symbolic; every root must numericize.
    // No symbolic Power may leak into the result.
    assert_eq!(
      interpret("FreeQ[NSolve[x^3 == 1, x], Power]").unwrap(),
      "True"
    );
    // Rounding away the last-digit float noise exposes the exact structure
    // and ordering (ascending real part, then imaginary), matching
    // wolframscript.
    assert_eq!(
      interpret("Round[x /. NSolve[x^3 == 1, x], 1/1000]").unwrap(),
      "{-1/2 - (433*I)/500, -1/2 + (433*I)/500, 1}"
    );
  }
}

mod find_root {
  use super::*;

  #[test]
  fn polynomial_root() {
    assert_eq!(
      interpret("FindRoot[x^2 - 2, {x, 1}]").unwrap(),
      "{x -> 1.4142135623730951}"
    );
  }

  #[test]
  fn polynomial_root_negative_start() {
    assert_eq!(
      interpret("FindRoot[x^2 - 2, {x, -1}]").unwrap(),
      "{x -> -1.4142135623730951}"
    );
  }

  #[test]
  fn equation_form() {
    assert_eq!(
      interpret("FindRoot[Cos[x] == x, {x, 0}]").unwrap(),
      "{x -> 0.7390851332151607}"
    );
  }

  #[test]
  fn transcendental() {
    assert_eq!(
      interpret("FindRoot[Sin[x] + Exp[x], {x, 0}]").unwrap(),
      "{x -> -0.5885327439818611}"
    );
  }

  #[test]
  fn cubic() {
    assert_eq!(
      interpret("FindRoot[x^3 - x - 1, {x, 1}]").unwrap(),
      "{x -> 1.324717957244746}"
    );
  }

  #[test]
  fn exponential() {
    assert_eq!(
      interpret("FindRoot[Exp[x] - 2, {x, 0}]").unwrap(),
      "{x -> 0.6931471805599453}"
    );
  }

  #[test]
  fn trivial() {
    assert_eq!(interpret("FindRoot[x, {x, 5}]").unwrap(), "{x -> 0.}");
  }

  #[test]
  fn quadratic_larger_start() {
    assert_eq!(
      interpret("FindRoot[x^2 - 10^5 x + 1 == 0, {x, 10^6}]").unwrap(),
      "{x -> 99999.99999000001}"
    );
  }

  #[test]
  fn bessel_j_root() {
    // FindRoot should work with BesselJ using numerical derivatives
    let result = interpret("FindRoot[BesselJ[0,x], {x,10.5}]").unwrap();
    assert!(
      result.starts_with("{x -> 18.07106396"),
      "Expected root near 18.071..., got: {}",
      result
    );
  }

  #[test]
  fn undefined_function_returns_unevaluated() {
    // Matches wolframscript: if the function can't be evaluated numerically
    // (e.g. f[x] undefined), FindRoot emits FindRoot::nlnum and returns
    // the expression unevaluated rather than erroring out.
    assert_eq!(
      interpret("FindRoot[f[x] == 0, {x, 0}]").unwrap(),
      "FindRoot[f[x] == 0, {x, 0}]"
    );
  }
  #[test]
  fn find_root_complex_starting_point() {
    // Complex starting points must drive Newton iteration in C, not abort
    // with "starting point must be numeric". For x^2+x+1 starting at -I
    // the iteration converges to the lower root -1/2 - sqrt(3)/2 i.
    let result = interpret("FindRoot[x^2 + x + 1, {x, -I}]").unwrap();
    // wolframscript yields {x -> -0.5 - 0.8660254037844386*I};
    // accept any complex value within a small tolerance of that root.
    assert!(
      result.contains("-0.5") && result.contains("0.866"),
      "Expected complex root near -0.5 - 0.866*I, got: {}",
      result
    );
  }
}

mod replace {
  use super::*;

  #[test]
  fn simple_match() {
    assert_eq!(interpret("Replace[x, x -> 2]").unwrap(), "2");
  }

  #[test]
  fn operator_form() {
    // Replace[rules][expr] is the curried form — expr goes first when
    // flattened (unlike Map/Apply, where the list comes first).
    assert_eq!(interpret("Replace[{x_ -> x + 1}][10]").unwrap(), "11");
    assert_eq!(interpret("Replace[{x_ -> x^2}][y]").unwrap(), "y^2");
  }

  #[test]
  fn operator_form_replace_all() {
    assert_eq!(
      interpret("ReplaceAll[{x -> 1, y -> 2}][x + y]").unwrap(),
      "3"
    );
  }

  #[test]
  fn with_rule_list() {
    assert_eq!(interpret("Replace[x, {x -> 2}]").unwrap(), "2");
  }

  #[test]
  fn no_subexpression_match() {
    assert_eq!(interpret("Replace[1 + x, {x -> 2}]").unwrap(), "1 + x");
  }

  #[test]
  fn multiple_rule_sets() {
    assert_eq!(
      interpret("Replace[x, {{x -> 1}, {x -> 2}}]").unwrap(),
      "{1, 2}"
    );
  }

  #[test]
  fn first_matching_rule() {
    assert_eq!(interpret("Replace[x, {x -> 10, y -> 20}]").unwrap(), "10");
  }

  #[test]
  fn pattern_match() {
    assert_eq!(interpret("Replace[42, n_Integer -> n + 1]").unwrap(), "43");
  }

  #[test]
  fn negative_levelspec_leaves() {
    // {-1} matches all atoms (Depth = 1).
    assert_eq!(
      interpret("Replace[f[1, g[2, h[3]]], _Integer -> 0, {-1}]").unwrap(),
      "f[0, g[0, h[0]]]"
    );
  }

  #[test]
  fn negative_levelspec_leaves_in_list() {
    assert_eq!(
      interpret("Replace[{1, {2, {3, 4}}}, _Integer -> 0, {-1}]").unwrap(),
      "{0, {0, {0, 0}}}"
    );
  }

  #[test]
  fn negative_levelspec_subtree_depth_two() {
    // {-2} matches subtrees with Depth = 2 (e.g. h[3] in this expression).
    assert_eq!(
      interpret("Replace[f[1, g[2, h[3]]], h[_] -> X, {-2}]").unwrap(),
      "f[1, g[2, X]]"
    );
  }

  #[test]
  fn negative_levelspec_no_match_at_wrong_depth() {
    // {-3} of f[1,g[2,h[3]]] only matches the subtree g[2,h[3]] (Depth = 3).
    assert_eq!(
      interpret("Replace[f[1, g[2, h[3]]], _ -> X, {-3}]").unwrap(),
      "f[1, X]"
    );
  }
}

mod distribute {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("Distribute[f[a + b, c]]").unwrap(),
      "f[a, c] + f[b, c]"
    );
  }

  #[test]
  fn both_sums() {
    assert_eq!(
      interpret("Distribute[f[a + b, c + d]]").unwrap(),
      "f[a, c] + f[a, d] + f[b, c] + f[b, d]"
    );
  }

  #[test]
  fn times_over_plus() {
    assert_eq!(
      interpret("Distribute[(a + b)(c + d)]").unwrap(),
      "a*c + b*c + a*d + b*d"
    );
  }

  #[test]
  fn three_terms() {
    assert_eq!(
      interpret("Distribute[f[a + b + c, d]]").unwrap(),
      "f[a, d] + f[b, d] + f[c, d]"
    );
  }

  #[test]
  fn three_args() {
    let result = interpret("Distribute[f[a + b, c + d, e + g]]").unwrap();
    assert!(result.contains("f[a, c, e]"));
    assert!(result.contains("f[b, d, g]"));
  }

  #[test]
  fn no_distribution_needed() {
    assert_eq!(interpret("Distribute[f[a, b]]").unwrap(), "f[a, b]");
  }

  #[test]
  fn with_head_restriction() {
    assert_eq!(
      interpret("Distribute[(a + b)(c + d), Plus, Times]").unwrap(),
      "a*c + b*c + a*d + b*d"
    );
  }

  #[test]
  fn atom_input() {
    assert_eq!(interpret("Distribute[x]").unwrap(), "x");
  }

  #[test]
  fn distribute_over_list() {
    assert_eq!(
      interpret("Distribute[f[{a, b}, {c, d}], List]").unwrap(),
      "{f[a, c], f[a, d], f[b, c], f[b, d]}"
    );
  }

  #[test]
  fn distribute_nested_lists() {
    assert_eq!(
      interpret("Distribute[{{1, 2}, {3, 4}}, List]").unwrap(),
      "{{1, 3}, {1, 4}, {2, 3}, {2, 4}}"
    );
  }
}

mod polynomial_remainder {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("PolynomialRemainder[x^3 + 2x + 1, x^2 + 1, x]").unwrap(),
      "1 + x"
    );
  }

  #[test]
  fn exact_division() {
    assert_eq!(
      interpret("PolynomialRemainder[x (x^2 + 1), x^2 + 1, x]").unwrap(),
      "0"
    );
  }

  #[test]
  fn symbolic_coefficients() {
    assert_eq!(
      interpret("PolynomialRemainder[a x^2 + b x + c, x + 1, x]").unwrap(),
      "a - b + c"
    );
  }

  #[test]
  fn lower_degree_dividend() {
    assert_eq!(
      interpret("PolynomialRemainder[x + 1, x^2, x]").unwrap(),
      "1 + x"
    );
  }

  #[test]
  fn quotient_basic() {
    assert_eq!(
      interpret("PolynomialQuotient[x^3 + 2x + 1, x^2 + 1, x]").unwrap(),
      "x"
    );
  }

  #[test]
  fn quotient_and_remainder_consistency() {
    // p = q * quotient + remainder
    let q =
      interpret("PolynomialQuotient[x^4 + 3x^2 + x, x^2 + 1, x]").unwrap();
    let r =
      interpret("PolynomialRemainder[x^4 + 3x^2 + x, x^2 + 1, x]").unwrap();
    // q should be x^2 + 2, r should be x - 2
    assert_eq!(q, "2 + x^2");
    assert_eq!(r, "-2 + x");
  }

  // The Modulus option performs the division over the field GF(p).
  #[test]
  fn modulus_option() {
    // Over GF(2), (x^2-1)/(x-1) = x+1 exactly.
    assert_eq!(
      interpret("PolynomialQuotient[x^2 - 1, x - 1, x, Modulus -> 2]").unwrap(),
      "1 + x"
    );
    assert_eq!(
      interpret("PolynomialRemainder[x^2 + 1, x + 1, x, Modulus -> 2]")
        .unwrap(),
      "0"
    );
    // (x^3+1)/(x+1) = x^2 - x + 1 == x^2 + x + 1 over GF(2).
    assert_eq!(
      interpret("PolynomialQuotient[x^3 + 1, x + 1, x, Modulus -> 2]").unwrap(),
      "1 + x + x^2"
    );
    // A nonzero modular remainder.
    assert_eq!(
      interpret("PolynomialRemainder[x^3 + x + 1, x^2 + 1, x, Modulus -> 2]")
        .unwrap(),
      "1"
    );
    // Over GF(5): (x^2+3x+2)/(x+1) = x+2.
    assert_eq!(
      interpret("PolynomialQuotient[x^2 + 3x + 2, x + 1, x, Modulus -> 5]")
        .unwrap(),
      "2 + x"
    );
    // PolynomialQuotientRemainder returns the {quotient, remainder} pair.
    assert_eq!(
      interpret("PolynomialQuotientRemainder[x^2 - 1, x - 1, x, Modulus -> 2]")
        .unwrap(),
      "{1 + x, 0}"
    );
  }
}

mod polynomial_lcm {
  use super::*;

  // PolynomialLCM[a, b] = (a / gcd) * b, displayed as an unexpanded product
  // matching Wolfram's factored form rather than the expanded polynomial.
  #[test]
  fn factored_two_factors() {
    assert_eq!(
      interpret("PolynomialLCM[x^2 - 1, x - 1]").unwrap(),
      "(-1 + x)*(1 + x)"
    );
  }

  #[test]
  fn factored_with_repeated_root() {
    assert_eq!(
      interpret("PolynomialLCM[x^2 - 1, x^2 + 2 x + 1]").unwrap(),
      "(-1 + x)*(1 + 2*x + x^2)"
    );
  }

  #[test]
  fn factored_distinct_quadratics() {
    assert_eq!(
      interpret("PolynomialLCM[x^2 + 3 x + 2, x^2 + 4 x + 3]").unwrap(),
      "(2 + x)*(3 + 4*x + x^2)"
    );
  }

  #[test]
  fn numeric_coefficients() {
    assert_eq!(interpret("PolynomialLCM[2 x, 3 x]").unwrap(), "6*x");
  }

  #[test]
  fn integer_arguments() {
    assert_eq!(interpret("PolynomialLCM[6, 4]").unwrap(), "12");
  }

  // When one polynomial divides the other the quotient is 1, so the LCM is the
  // multiple itself (a single, expanded factor).
  #[test]
  fn divisible_pair_single_factor() {
    assert_eq!(
      interpret("PolynomialLCM[x - 1, x^2 - 1]").unwrap(),
      "-1 + x^2"
    );
  }

  #[test]
  fn three_arguments() {
    assert_eq!(
      interpret("PolynomialLCM[x^2 - 1, x - 1, x + 1]").unwrap(),
      "(-1 + x)*(1 + x)"
    );
  }
}

mod polynomial_extended_gcd {
  use super::*;

  // PolynomialExtendedGCD[p, q, x] returns {g, {s, t}} with s*p + t*q == g.
  #[test]
  fn basic() {
    assert_eq!(
      interpret("PolynomialExtendedGCD[x^2 - 1, x^3 - 1, x]").unwrap(),
      "{-1 + x, {-x, 1}}"
    );
  }

  // With a Modulus the extended GCD is computed over the field GF(p), with a
  // monic GCD; s*p + t*q == g (mod p).
  #[test]
  fn modulus_option() {
    assert_eq!(
      interpret("PolynomialExtendedGCD[x^2 - 1, x - 1, x, Modulus -> 2]")
        .unwrap(),
      "{1 + x, {0, 1}}"
    );
    assert_eq!(
      interpret("PolynomialExtendedGCD[x^3 + x, x^2 + 1, x, Modulus -> 2]")
        .unwrap(),
      "{1 + x^2, {0, 1}}"
    );
    assert_eq!(
      interpret("PolynomialExtendedGCD[x^2 + 1, x + 2, x, Modulus -> 5]")
        .unwrap(),
      "{2 + x, {0, 1}}"
    );
    // Nontrivial Bezout coefficients over GF(3).
    assert_eq!(
      interpret("PolynomialExtendedGCD[x^4 - 1, x^2 + x, x, Modulus -> 3]")
        .unwrap(),
      "{1 + x, {2, 1 + 2*x + x^2}}"
    );
  }
}

mod polynomial_reduce {
  use super::*;

  #[test]
  fn single_divisor() {
    // x^2 + 1 = (x - 1)(x + 1) + 2.
    assert_eq!(
      interpret("PolynomialReduce[x^2 + 1, {x + 1}, x]").unwrap(),
      "{{-1 + x}, 2}"
    );
  }

  #[test]
  fn two_divisors_exact() {
    // x^3 + 2x + 1 = x(x^2 + 1) + 1(x + 1).
    assert_eq!(
      interpret("PolynomialReduce[x^3 + 2 x + 1, {x^2 + 1, x + 1}, x]")
        .unwrap(),
      "{{x, 1}, 0}"
    );
  }

  #[test]
  fn geometric_quotient() {
    assert_eq!(
      interpret("PolynomialReduce[x^3, {x - 1}, x]").unwrap(),
      "{{1 + x + x^2}, 1}"
    );
  }

  #[test]
  fn rational_coefficients() {
    assert_eq!(
      interpret("PolynomialReduce[x^2, {3 x + 1}, x]").unwrap(),
      "{{-1/9 + x/3}, 1/9}"
    );
  }

  #[test]
  fn divisor_degree_exceeds_dividend() {
    // No reduction possible; the whole polynomial is the remainder.
    assert_eq!(
      interpret("PolynomialReduce[x^2 + 1, {x^3 + 1}, x]").unwrap(),
      "{{0}, 1 + x^2}"
    );
  }

  #[test]
  fn constant_dividend() {
    assert_eq!(
      interpret("PolynomialReduce[5, {x + 1}, x]").unwrap(),
      "{{0}, 5}"
    );
  }

  #[test]
  fn variable_as_single_element_list() {
    assert_eq!(
      interpret("PolynomialReduce[x^2 + 1, {x + 1}, {x}]").unwrap(),
      "{{-1 + x}, 2}"
    );
  }

  // Multivariate division (lexicographic order over the given variables).
  #[test]
  fn multivariate_single_divisor() {
    assert_eq!(
      interpret("PolynomialReduce[x^2 + y^2, {x - y}, {x, y}]").unwrap(),
      "{{x + y}, 2*y^2}"
    );
    assert_eq!(
      interpret("PolynomialReduce[x^3 + y^3, {x + y}, {x, y}]").unwrap(),
      "{{x^2 - x*y + y^2}, 0}"
    );
  }

  #[test]
  fn multivariate_two_divisors() {
    assert_eq!(
      interpret("PolynomialReduce[x^2 + y^2, {x + y, x - y}, {x, y}]").unwrap(),
      "{{x - y, 0}, 2*y^2}"
    );
    assert_eq!(
      interpret("PolynomialReduce[x^2 y^2, {x^2 - y, x y - 1}, {x, y}]")
        .unwrap(),
      "{{y^2, 0}, y^3}"
    );
  }

  #[test]
  fn multivariate_rational_coefficient_quotient() {
    assert_eq!(
      interpret("PolynomialReduce[2 x^2, {3 x}, {x, y}]").unwrap(),
      "{{(2*x)/3}, 0}"
    );
  }

  #[test]
  fn multivariate_nonzero_remainder() {
    assert_eq!(
      interpret("PolynomialReduce[x^2 y + x y^2, {x y - 1}, {x, y}]").unwrap(),
      "{{x + y}, x + y}"
    );
  }
}

mod solve_expression_target {
  use super::*;

  #[test]
  fn solve_for_function_call() {
    assert_eq!(
      interpret("Solve[f[x + y] == 3, f[x + y]]").unwrap(),
      "{{f[x + y] -> 3}}"
    );
  }
}

mod solve_with_domain {
  use super::*;

  #[test]
  fn reals_no_complex() {
    assert_eq!(interpret("Solve[x^2 == -1, x, Reals]").unwrap(), "{}");
  }

  #[test]
  fn reals_with_solutions() {
    assert_eq!(
      interpret("Solve[x^2 == 1, x, Reals]").unwrap(),
      "{{x -> -1}, {x -> 1}}"
    );
  }

  #[test]
  fn integers_filters_non_integer() {
    assert_eq!(
      interpret("Solve[-4 - 4 x + x^4 + x^5 == 0, x, Integers]").unwrap(),
      "{{x -> -1}}"
    );
  }

  #[test]
  fn integers_no_solutions() {
    assert_eq!(interpret("Solve[x^4 == 4, x, Integers]").unwrap(), "{}");
  }

  #[test]
  fn integers_bounded_linear_two_vars_unique() {
    assert_eq!(
      interpret(
        "Solve[{15 n + 17 m == 200, n >= 0, m >= 0}, {n, m}, Integers]"
      )
      .unwrap(),
      "{{n -> 2, m -> 10}}"
    );
  }

  #[test]
  fn integers_bounded_linear_two_vars_multi() {
    assert_eq!(
      interpret("Solve[{x + y == 5, x >= 0, y >= 0}, {x, y}, Integers]")
        .unwrap(),
      "{{x -> 0, y -> 5}, {x -> 1, y -> 4}, {x -> 2, y -> 3}, {x -> 3, y -> 2}, {x -> 4, y -> 1}, {x -> 5, y -> 0}}"
    );
  }

  #[test]
  fn integers_bounded_linear_with_upper_bounds() {
    assert_eq!(
      interpret(
        "Solve[{x + y == 10, x >= 0, y >= 0, x <= 5, y <= 5}, {x, y}, Integers]"
      )
      .unwrap(),
      "{{x -> 5, y -> 5}}"
    );
  }

  // Strict `>` excludes the boundary integer (x > 0 means x >= 1), unlike the
  // non-strict `>= 0` case above which includes 0.
  #[test]
  fn integers_bounded_strict_lower() {
    assert_eq!(
      interpret("Solve[{x + y == 5, x > 0, y > 0}, {x, y}, Integers]").unwrap(),
      "{{x -> 1, y -> 4}, {x -> 2, y -> 3}, {x -> 3, y -> 2}, {x -> 4, y -> 1}}"
    );
  }

  #[test]
  fn integers_bounded_strict_lower_shifted() {
    // x > 2 and y > 2 means both >= 3.
    assert_eq!(
      interpret("Solve[{x + y == 10, x > 2, y > 2}, {x, y}, Integers]")
        .unwrap(),
      "{{x -> 3, y -> 7}, {x -> 4, y -> 6}, {x -> 5, y -> 5}, {x -> 6, y -> 4}, {x -> 7, y -> 3}}"
    );
  }

  #[test]
  fn integers_bounded_mixed_strict_nonstrict() {
    // x > 0 (>= 1) but y >= 0 (includes 0).
    assert_eq!(
      interpret("Solve[{x + y == 5, x > 0, y >= 0}, {x, y}, Integers]")
        .unwrap(),
      "{{x -> 1, y -> 4}, {x -> 2, y -> 3}, {x -> 3, y -> 2}, {x -> 4, y -> 1}, {x -> 5, y -> 0}}"
    );
  }
}

// Solving a periodic (trig) equation over a bounded interval specializes the
// general ConditionalExpression family to the concrete solutions in range.
mod solve_periodic_bounded {
  use super::*;

  #[test]
  fn sin_open_interval() {
    assert_eq!(
      interpret("Solve[Sin[x] == 0 && 0 < x < 2 Pi, x]").unwrap(),
      "{{x -> Pi}}"
    );
  }

  // A closed interval includes the endpoints, sorted ascending.
  #[test]
  fn sin_closed_interval() {
    assert_eq!(
      interpret("Solve[Sin[x] == 0 && 0 <= x <= 2 Pi, x]").unwrap(),
      "{{x -> 0}, {x -> Pi}, {x -> 2*Pi}}"
    );
  }

  #[test]
  fn cos_open_interval() {
    assert_eq!(
      interpret("Solve[Cos[x] == 0 && 0 < x < 2 Pi, x]").unwrap(),
      "{{x -> Pi/2}, {x -> (3*Pi)/2}}"
    );
  }

  #[test]
  fn sin_symmetric_interval() {
    assert_eq!(
      interpret("Solve[Sin[x] == 0 && -Pi < x < Pi, x]").unwrap(),
      "{{x -> 0}}"
    );
  }

  // Without a bounding constraint the general periodic family is kept.
  #[test]
  fn unbounded_keeps_general_family() {
    assert_eq!(
      interpret("Solve[Sin[x] == 0, x]").unwrap(),
      "{{x -> ConditionalExpression[2*Pi*C[1], Element[C[1], Integers]]}, \
       {x -> ConditionalExpression[Pi + 2*Pi*C[1], Element[C[1], Integers]]}}"
    );
  }

  // A non-special right-hand side uses the inverse-trig family, so a bounded
  // interval still yields concrete solutions.
  #[test]
  fn cos_half_bounded() {
    assert_eq!(
      interpret("Solve[Cos[x] == 1/2 && 0 < x < 2 Pi, x]").unwrap(),
      "{{x -> Pi/3}, {x -> (5*Pi)/3}}"
    );
  }

  #[test]
  fn sin_half_bounded() {
    assert_eq!(
      interpret("Solve[Sin[x] == 1/2 && 0 < x < 2 Pi, x]").unwrap(),
      "{{x -> Pi/6}, {x -> (5*Pi)/6}}"
    );
  }

  // Tan has period Pi, so a 2-Pi interval gives two solutions.
  #[test]
  fn tan_one_bounded() {
    assert_eq!(
      interpret("Solve[Tan[x] == 1 && 0 < x < 2 Pi, x]").unwrap(),
      "{{x -> Pi/4}, {x -> (5*Pi)/4}}"
    );
  }

  // An exact irrational right-hand side also resolves.
  #[test]
  fn sin_sqrt2_over_2_bounded() {
    assert_eq!(
      interpret("Solve[Sin[x] == Sqrt[2]/2 && 0 < x < 2 Pi, x]").unwrap(),
      "{{x -> Pi/4}, {x -> (3*Pi)/4}}"
    );
  }

  // The unbounded general family matches wolframscript's two branches.
  #[test]
  fn cos_half_unbounded_family() {
    assert_eq!(
      interpret("Solve[Cos[x] == 1/2, x]").unwrap(),
      "{{x -> ConditionalExpression[-1/3*Pi + 2*Pi*C[1], \
       Element[C[1], Integers]]}, {x -> ConditionalExpression[Pi/3 + 2*Pi*C[1], \
       Element[C[1], Integers]]}}"
    );
  }
}

mod solve_always {
  use super::*;

  #[test]
  fn linear_single_variable() {
    assert_eq!(
      interpret("SolveAlways[a*x + b == 0, x]").unwrap(),
      "{{a -> 0, b -> 0}}"
    );
  }

  #[test]
  fn quadratic_with_offsets() {
    assert_eq!(
      interpret("SolveAlways[(a - 2)*x^2 + (b + 1)*x + c == 0, x]").unwrap(),
      "{{a -> 2, b -> -1, c -> 0}}"
    );
  }

  #[test]
  fn matching_polynomial() {
    assert_eq!(
      interpret("SolveAlways[a*x^2 + b*x + c == 3*x^2 - 5*x + 7, x]").unwrap(),
      "{{a -> 3, b -> -5, c -> 7}}"
    );
  }

  #[test]
  fn trivially_true() {
    assert_eq!(interpret("SolveAlways[0 == 0, x]").unwrap(), "{{}}");
  }

  #[test]
  fn impossible_equation() {
    assert_eq!(interpret("SolveAlways[x + 1 == 0, x]").unwrap(), "{}");
  }

  #[test]
  fn no_parameters() {
    assert_eq!(
      interpret("SolveAlways[3*x^2 + 5*x + 7 == 0, x]").unwrap(),
      "{}"
    );
  }

  #[test]
  fn multivariate() {
    assert_eq!(
      interpret("SolveAlways[(a - 2)*x + (b + 1)*y + c == 0, {x, y}]").unwrap(),
      "{{a -> 2, b -> -1, c -> 0}}"
    );
  }

  #[test]
  fn multivariate_all_zero() {
    assert_eq!(
      interpret("SolveAlways[a*x + b*y + c == 0, {x, y}]").unwrap(),
      "{{a -> 0, b -> 0, c -> 0}}"
    );
  }

  #[test]
  fn list_form_single_var() {
    assert_eq!(
      interpret("SolveAlways[a*x + b == 0, {x}]").unwrap(),
      "{{a -> 0, b -> 0}}"
    );
  }

  #[test]
  fn quadratic_cross_terms() {
    assert_eq!(
      interpret("SolveAlways[a*x^2 + b*x*y + c*y^2 == 0, {x, y}]").unwrap(),
      "{{a -> 0, b -> 0, c -> 0}}"
    );
  }
}

mod factor_terms {
  use super::*;

  #[test]
  fn basic_integer_gcd() {
    assert_eq!(
      interpret("FactorTerms[3 + 6 x + 3 x^2]").unwrap(),
      "3*(1 + 2*x + x^2)"
    );
  }

  #[test]
  fn factor_from_all_terms() {
    assert_eq!(
      interpret("FactorTerms[6 x + 9 x^2]").unwrap(),
      "3*(2*x + 3*x^2)"
    );
  }

  #[test]
  fn simple_factoring() {
    assert_eq!(interpret("FactorTerms[5 + 10 x]").unwrap(), "5*(1 + 2*x)");
  }

  #[test]
  fn gcd_one_no_change() {
    assert_eq!(interpret("FactorTerms[x + x^2]").unwrap(), "x + x^2");
  }

  #[test]
  fn single_term() {
    assert_eq!(interpret("FactorTerms[7 x]").unwrap(), "7*x");
  }

  #[test]
  fn constant() {
    assert_eq!(interpret("FactorTerms[42]").unwrap(), "42");
  }

  #[test]
  fn zero() {
    assert_eq!(interpret("FactorTerms[0]").unwrap(), "0");
  }

  #[test]
  fn variable_only() {
    assert_eq!(interpret("FactorTerms[x]").unwrap(), "x");
  }

  #[test]
  fn negative_gcd() {
    assert_eq!(
      interpret("FactorTerms[-6*x - 9*x^2]").unwrap(),
      "-3*(2*x + 3*x^2)"
    );
  }

  #[test]
  fn negative_gcd_with_constant() {
    assert_eq!(interpret("FactorTerms[-3 - 6*x]").unwrap(), "-3*(1 + 2*x)");
  }

  #[test]
  fn symbolic_coefficients() {
    assert_eq!(
      interpret("FactorTerms[2 a + 4 b + 6 c]").unwrap(),
      "2*(a + 2*b + 3*c)"
    );
  }

  #[test]
  fn no_common_factor() {
    assert_eq!(interpret("FactorTerms[a + b]").unwrap(), "a + b");
  }

  #[test]
  fn rational_coefficients() {
    assert_eq!(
      interpret("FactorTerms[2/3 + (4/3)*x]").unwrap(),
      "(2*(1 + 2*x))/3"
    );
  }

  #[test]
  fn with_variable_argument() {
    assert_eq!(
      interpret("FactorTerms[3 + 3 a + 6 a x + 6 x + 12 a x^2 + 12 x^2, x]")
        .unwrap(),
      "3*(1 + a)*(1 + 2*x + 4*x^2)"
    );
  }

  #[test]
  fn threads_over_list() {
    assert_eq!(
      interpret("FactorTerms[{6 + 12 x, 4 + 8 x}]").unwrap(),
      "{6*(1 + 2*x), 4*(1 + 2*x)}"
    );
  }
}

mod cyclotomic {
  use super::*;

  #[test]
  fn phi_1() {
    assert_eq!(interpret("Cyclotomic[1, x]").unwrap(), "-1 + x");
  }

  #[test]
  fn phi_2() {
    assert_eq!(interpret("Cyclotomic[2, x]").unwrap(), "1 + x");
  }

  #[test]
  fn phi_3() {
    assert_eq!(interpret("Cyclotomic[3, x]").unwrap(), "1 + x + x^2");
  }

  #[test]
  fn phi_4() {
    assert_eq!(interpret("Cyclotomic[4, x]").unwrap(), "1 + x^2");
  }

  #[test]
  fn phi_5() {
    assert_eq!(
      interpret("Cyclotomic[5, x]").unwrap(),
      "1 + x + x^2 + x^3 + x^4"
    );
  }

  #[test]
  fn phi_6() {
    assert_eq!(interpret("Cyclotomic[6, x]").unwrap(), "1 - x + x^2");
  }

  #[test]
  fn phi_8() {
    assert_eq!(interpret("Cyclotomic[8, x]").unwrap(), "1 + x^4");
  }

  #[test]
  fn phi_12() {
    assert_eq!(interpret("Cyclotomic[12, x]").unwrap(), "1 - x^2 + x^4");
  }

  #[test]
  fn phi_0() {
    assert_eq!(interpret("Cyclotomic[0, x]").unwrap(), "1");
  }

  #[test]
  fn numeric_evaluation() {
    assert_eq!(interpret("Cyclotomic[1, 3]").unwrap(), "2");
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[Cyclotomic]").unwrap(),
      "{Listable, Protected}"
    );
  }
}

mod expand_numerator {
  use super::*;

  #[test]
  fn basic_square() {
    assert_eq!(
      interpret("ExpandNumerator[(1 + x)^2/(1 + y)^2]").unwrap(),
      "(1 + 2*x + x^2)/(1 + y)^2"
    );
  }

  #[test]
  fn cubic_numerator() {
    assert_eq!(
      interpret("ExpandNumerator[(a + b)^3/(c + d)]").unwrap(),
      "(a^3 + 3*a^2*b + 3*a*b^2 + b^3)/(c + d)"
    );
  }

  #[test]
  fn no_fraction() {
    assert_eq!(interpret("ExpandNumerator[x + 1]").unwrap(), "1 + x");
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[ExpandNumerator]").unwrap(),
      "{Protected}"
    );
  }
}

mod expand_denominator {
  use super::*;

  #[test]
  fn basic_square() {
    assert_eq!(
      interpret("ExpandDenominator[(1 + x)^2/(1 + y)^2]").unwrap(),
      "(1 + x)^2/(1 + 2*y + y^2)"
    );
  }

  #[test]
  fn cubic_denominator() {
    assert_eq!(
      interpret("ExpandDenominator[(a + b)/(c + d)^3]").unwrap(),
      "(a + b)/(c^3 + 3*c^2*d + 3*c*d^2 + d^3)"
    );
  }

  #[test]
  fn no_denominator() {
    assert_eq!(interpret("ExpandDenominator[x + 1]").unwrap(), "1 + x");
  }

  #[test]
  fn simple_fraction() {
    assert_eq!(
      interpret("ExpandDenominator[x/(y+z)]").unwrap(),
      "x/(y + z)"
    );
  }

  #[test]
  fn sum_of_fractions() {
    assert_eq!(
      interpret("ExpandDenominator[a/(1+x)^2 + b/(1+y)^2]").unwrap(),
      "a/(1 + 2*x + x^2) + b/(1 + 2*y + y^2)"
    );
  }

  #[test]
  fn multi_factor_denominator() {
    // A denominator that's a product of two sums must be fully distributed,
    // not expanded factor-by-factor. Regression for mathics algebra.py:1288
    // (ExpandDenominator[(a+b)^2 / ((c+d)^2 (e+f))]).
    assert_eq!(
      interpret("ExpandDenominator[(a + b)^2 / ((c + d)^2 (e + f))]").unwrap(),
      "(a + b)^2/(c^2*e + 2*c*d*e + d^2*e + c^2*f + 2*c*d*f + d^2*f)"
    );
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[ExpandDenominator]").unwrap(),
      "{Protected}"
    );
  }
}

mod power_expand {
  use super::*;

  #[test]
  fn product_power() {
    assert_eq!(interpret("PowerExpand[(a*b)^s]").unwrap(), "a^s*b^s");
  }

  #[test]
  fn product_power_three_factors() {
    assert_eq!(interpret("PowerExpand[(a*b*c)^s]").unwrap(), "a^s*b^s*c^s");
  }

  #[test]
  fn nested_power() {
    assert_eq!(interpret("PowerExpand[(a^r)^s]").unwrap(), "a^(r*s)");
  }

  #[test]
  fn quotient_power() {
    assert_eq!(interpret("PowerExpand[(x/y)^n]").unwrap(), "x^n/y^n");
  }

  #[test]
  fn integer_exponent() {
    assert_eq!(interpret("PowerExpand[(x*y)^2]").unwrap(), "x^2*y^2");
  }

  #[test]
  fn numeric_factor_power() {
    assert_eq!(interpret("PowerExpand[(2*x)^n]").unwrap(), "2^n*x^n");
  }

  #[test]
  fn compound_powers_in_product() {
    assert_eq!(
      interpret("PowerExpand[(x^a*y^b)^c]").unwrap(),
      "x^(a*c)*y^(b*c)"
    );
  }

  #[test]
  fn sqrt_product() {
    assert_eq!(
      interpret("PowerExpand[Sqrt[a*b]]").unwrap(),
      "Sqrt[a]*Sqrt[b]"
    );
  }

  #[test]
  fn sqrt_power_product() {
    assert_eq!(interpret("PowerExpand[Sqrt[x^2*y^4]]").unwrap(), "x*y^2");
  }

  #[test]
  fn sqrt_squared() {
    assert_eq!(interpret("PowerExpand[Sqrt[x^2]]").unwrap(), "x");
  }

  #[test]
  fn fractional_power() {
    assert_eq!(interpret("PowerExpand[(x^2)^(1/3)]").unwrap(), "x^(2/3)");
  }

  #[test]
  fn sqrt_three_factors() {
    assert_eq!(
      interpret("PowerExpand[(a*b*c)^(1/2)]").unwrap(),
      "Sqrt[a]*Sqrt[b]*Sqrt[c]"
    );
  }

  #[test]
  fn half_power_compound() {
    assert_eq!(
      interpret("PowerExpand[(x^2*y^3)^(1/2)]").unwrap(),
      "x*y^(3/2)"
    );
  }

  #[test]
  fn log_product() {
    assert_eq!(
      interpret("PowerExpand[Log[a*b]]").unwrap(),
      "Log[a] + Log[b]"
    );
  }

  #[test]
  fn log_product_three() {
    assert_eq!(
      interpret("PowerExpand[Log[a*b*c]]").unwrap(),
      "Log[a] + Log[b] + Log[c]"
    );
  }

  #[test]
  fn log_power() {
    assert_eq!(interpret("PowerExpand[Log[a^b]]").unwrap(), "b*Log[a]");
  }

  #[test]
  fn log_quotient() {
    assert_eq!(
      interpret("PowerExpand[Log[a/b]]").unwrap(),
      "Log[a] - Log[b]"
    );
  }

  #[test]
  fn log_sqrt() {
    assert_eq!(interpret("PowerExpand[Log[Sqrt[x]]]").unwrap(), "Log[x]/2");
  }

  #[test]
  fn log_e_power() {
    assert_eq!(interpret("PowerExpand[Log[E^x]]").unwrap(), "x");
  }

  #[test]
  fn log_compound_product_powers() {
    assert_eq!(
      interpret("PowerExpand[Log[x^2*y^3]]").unwrap(),
      "2*Log[x] + 3*Log[y]"
    );
  }

  #[test]
  fn log_symbolic_powers() {
    assert_eq!(
      interpret("PowerExpand[Log[x^a*y^b*z^c]]").unwrap(),
      "a*Log[x] + b*Log[y] + c*Log[z]"
    );
  }

  #[test]
  fn exp_log_identity() {
    assert_eq!(interpret("PowerExpand[Exp[x*Log[y]]]").unwrap(), "y^x");
  }

  #[test]
  fn sum_passthrough() {
    assert_eq!(interpret("PowerExpand[(a+b)^n]").unwrap(), "(a + b)^n");
  }

  #[test]
  fn log_sum_passthrough() {
    assert_eq!(interpret("PowerExpand[Log[a+b]]").unwrap(), "Log[a + b]");
  }

  #[test]
  fn atom_passthrough() {
    assert_eq!(interpret("PowerExpand[x]").unwrap(), "x");
    assert_eq!(interpret("PowerExpand[5]").unwrap(), "5");
  }

  #[test]
  fn additive_passthrough() {
    assert_eq!(interpret("PowerExpand[x+y]").unwrap(), "x + y");
  }

  #[test]
  fn thread_over_list() {
    assert_eq!(
      interpret("PowerExpand[{(a*b)^s, Log[a*b], Sqrt[x*y]}]").unwrap(),
      "{a^s*b^s, Log[a] + Log[b], Sqrt[x]*Sqrt[y]}"
    );
  }

  #[test]
  fn nested_in_function() {
    assert_eq!(
      interpret("PowerExpand[Sin[(a*b)^s]]").unwrap(),
      "Sin[a^s*b^s]"
    );
  }

  #[test]
  fn sum_of_powers() {
    assert_eq!(
      interpret("PowerExpand[(x+y)^n + (a*b)^s]").unwrap(),
      "a^s*b^s + (x + y)^n"
    );
  }

  #[test]
  fn with_assumptions() {
    // Second argument (Assumptions) is accepted
    assert_eq!(
      interpret("PowerExpand[(a*b)^s, Assumptions -> a > 0]").unwrap(),
      "a^s*b^s"
    );
  }
}

mod resultant {
  use super::*;

  #[test]
  fn basic_integer() {
    assert_eq!(interpret("Resultant[x^2 + 1, x^3 - 1, x]").unwrap(), "2");
  }

  #[test]
  fn common_root() {
    // x^2 - 1 and x^3 - 1 share x=1 as a root
    assert_eq!(interpret("Resultant[x^2 - 1, x^3 - 1, x]").unwrap(), "0");
  }

  #[test]
  fn linear_polynomials() {
    assert_eq!(interpret("Resultant[2*x + 3, 4*x - 1, x]").unwrap(), "-14");
  }

  #[test]
  fn symbolic_linear() {
    assert_eq!(
      interpret("Resultant[a*x + b, c*x + d, x]").unwrap(),
      "-(b*c) + a*d"
    );
  }

  #[test]
  fn symbolic_quadratic_expanded() {
    assert_eq!(
      interpret("Expand[Resultant[x^2 + a*x + b, x^2 + c*x + d, x]]").unwrap(),
      "b^2 - a*b*c + b*c^2 + a^2*d - 2*b*d - a*c*d + d^2"
    );
  }

  #[test]
  fn quadratic_common_root() {
    // x^2 - 5x + 6 = (x-2)(x-3), x^2 - 3x + 2 = (x-1)(x-2), share x=2
    assert_eq!(
      interpret("Resultant[x^2 - 5*x + 6, x^2 - 3*x + 2, x]").unwrap(),
      "0"
    );
  }

  #[test]
  fn zero_polynomial() {
    assert_eq!(interpret("Resultant[x, x^2, x]").unwrap(), "0");
  }

  #[test]
  fn constant_and_polynomial() {
    assert_eq!(interpret("Resultant[3, x + 1, x]").unwrap(), "3");
  }

  #[test]
  fn symbolic_stays_unevaluated() {
    assert_eq!(interpret("Resultant[f, g, x]").unwrap(), "1");
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[Resultant]").unwrap(),
      "{Listable, Protected}"
    );
  }

  // A Modulus option reduces the resultant's coefficients modulo p.
  #[test]
  fn modulus_option() {
    // Res(x^2-1, x-1) = 0, reduced mod 2.
    assert_eq!(
      interpret("Resultant[x^2 - 1, x - 1, x, Modulus -> 2]").unwrap(),
      "0"
    );
    // Res(x^2+1, x+1) = 2, mod 5 stays 2.
    assert_eq!(
      interpret("Resultant[x^2 + 1, x + 1, x, Modulus -> 5]").unwrap(),
      "2"
    );
    // Symbolic coefficients: Res(x^2+a, x+b) = a + b^2 (mod 3).
    assert_eq!(
      interpret("Resultant[x^2 + a, x + b, x, Modulus -> 3]").unwrap(),
      "a + b^2"
    );
    // A leading coefficient divisible by p: Res over Z is 3, reduced to 1.
    assert_eq!(
      interpret("Resultant[2x^2 + 1, x + 1, x, Modulus -> 2]").unwrap(),
      "1"
    );
  }
}

mod discriminant {
  use super::*;

  // The Modulus option reduces the discriminant's coefficients modulo p.
  #[test]
  fn modulus_option() {
    // Disc(x^2+b x+c) = b^2 - 4c == b^2 over GF(2).
    assert_eq!(
      interpret("Discriminant[x^2 + b x + c, x, Modulus -> 2]").unwrap(),
      "b^2"
    );
    // Disc(x^3+p x+q) = -4 p^3 - 27 q^2 == 2 p^3 over GF(3).
    assert_eq!(
      interpret("Discriminant[x^3 + p x + q, x, Modulus -> 3]").unwrap(),
      "2*p^3"
    );
    // Disc(a x^2+b x+c) = b^2 - 4 a c == b^2 + 2 a c over GF(3).
    assert_eq!(
      interpret("Discriminant[a x^2 + b x + c, x, Modulus -> 3]").unwrap(),
      "b^2 + 2*a*c"
    );
    // The unmodulated discriminant is unchanged.
    assert_eq!(
      interpret("Discriminant[x^2 + b x + c, x]").unwrap(),
      "b^2 - 4*c"
    );
  }
}

mod factor_square_free {
  use super::*;

  #[test]
  fn basic_repeated_factor() {
    assert_eq!(
      interpret("FactorSquareFree[x^5 - x^4 - x + 1]").unwrap(),
      "(-1 + x)^2*(1 + x + x^2 + x^3)"
    );
  }

  #[test]
  fn with_x_factor() {
    assert_eq!(
      interpret("FactorSquareFree[x^4 - 2*x^3 + x^2]").unwrap(),
      "(-1 + x)^2*x^2"
    );
  }

  #[test]
  fn with_integer_content() {
    assert_eq!(
      interpret("FactorSquareFree[12*x^3 + 36*x^2 + 36*x + 12]").unwrap(),
      "12*(1 + x)^3"
    );
  }

  #[test]
  fn square_free_unchanged() {
    assert_eq!(interpret("FactorSquareFree[x^6 - 1]").unwrap(), "-1 + x^6");
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[FactorSquareFree]").unwrap(),
      "{Listable, Protected}"
    );
  }
}

mod factor_terms_list {
  use super::*;

  #[test]
  fn common_integer_factor() {
    assert_eq!(
      interpret("FactorTermsList[6*x^2 - 12*x + 6]").unwrap(),
      "{6, 1 - 2*x + x^2}"
    );
  }

  #[test]
  fn no_common_factor() {
    assert_eq!(
      interpret("FactorTermsList[x^2 + 2*x + 1]").unwrap(),
      "{1, 1 + 2*x + x^2}"
    );
  }

  #[test]
  fn negative_leading() {
    assert_eq!(
      interpret("FactorTermsList[-3*x^2 + 6*x - 9]").unwrap(),
      "{-3, 3 - 2*x + x^2}"
    );
  }

  #[test]
  fn constant_input() {
    assert_eq!(interpret("FactorTermsList[5]").unwrap(), "{5, 1}");
  }

  #[test]
  fn no_numeric_content() {
    assert_eq!(
      interpret("FactorTermsList[x^3 + x]").unwrap(),
      "{1, x + x^3}"
    );
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[FactorTermsList]").unwrap(),
      "{Protected}"
    );
  }

  // Two-arg form always returns a 3-element list {numeric_content,
  // non-var-part, var-part}, even when the polynomial coefficients aren't
  // integers (regression for mathics Factor doctest).
  #[test]
  fn two_arg_symbol_independent_of_var() {
    assert_eq!(interpret("FactorTermsList[f, x]").unwrap(), "{1, f, 1}");
  }

  #[test]
  fn two_arg_scaled_symbol_independent_of_var() {
    assert_eq!(interpret("FactorTermsList[3*f, x]").unwrap(), "{3, f, 1}");
  }

  #[test]
  fn two_arg_var_independent_pure_number() {
    // Pure numeric inputs still collapse to the 2-element form.
    assert_eq!(interpret("FactorTermsList[4, x]").unwrap(), "{4, 1}");
  }
}

mod refine {
  use super::*;

  #[test]
  fn sqrt_x_squared_positive() {
    assert_eq!(interpret("Refine[Sqrt[x^2], x > 0]").unwrap(), "x");
  }

  // Max/Min collapse when the assumption orders their arguments.
  #[test]
  fn max_min_ordered() {
    assert_eq!(interpret("Refine[Max[a, b], a > b]").unwrap(), "a");
    assert_eq!(interpret("Refine[Max[a, b], a < b]").unwrap(), "b");
    assert_eq!(interpret("Refine[Max[a, b], a >= b]").unwrap(), "a");
    assert_eq!(interpret("Refine[Min[a, b], a < b]").unwrap(), "a");
    assert_eq!(interpret("Refine[Min[a, b], a > b]").unwrap(), "b");
    // Reversed assumption order is handled symmetrically.
    assert_eq!(interpret("Refine[Max[a, b], b < a]").unwrap(), "a");
    // Simplify accepts assumptions the same way.
    assert_eq!(interpret("Simplify[Max[a, b], a > b]").unwrap(), "a");
    // x > 0 orders Max[x, 0] (which canonicalises to Max[0, x]).
    assert_eq!(interpret("Refine[Max[x, 0], x > 0]").unwrap(), "x");
    assert_eq!(interpret("Refine[Min[x, 0], x > 0]").unwrap(), "0");
    // No ordering known -> unchanged.
    assert_eq!(interpret("Refine[Max[a, b], a == b]").unwrap(), "Max[a, b]");
  }

  // A finite factor with a known sign collapses `factor * Infinity` to the
  // correctly-signed infinity (matching wolframscript).
  #[test]
  fn signed_factor_times_infinity() {
    assert_eq!(interpret("Refine[a*Infinity, a > 0]").unwrap(), "Infinity");
    assert_eq!(interpret("Refine[a*Infinity, a < 0]").unwrap(), "-Infinity");
    assert_eq!(
      interpret("Refine[a*(-Infinity), a > 0]").unwrap(),
      "-Infinity"
    );
    assert_eq!(
      interpret("Refine[a*(-Infinity), a < 0]").unwrap(),
      "Infinity"
    );
    // A strict lower bound above zero is also positive.
    assert_eq!(interpret("Refine[a*Infinity, a > 5]").unwrap(), "Infinity");
    // A positive numeric coefficient does not change the sign.
    assert_eq!(
      interpret("Refine[2 a*Infinity, a > 0]").unwrap(),
      "Infinity"
    );
    // Product of two positive factors stays positive.
    assert_eq!(
      interpret("Refine[a b*Infinity, a > 0 && b > 0]").unwrap(),
      "Infinity"
    );
    // Simplify accepts the assumption the same way.
    assert_eq!(
      interpret("Simplify[a*Infinity, a > 0]").unwrap(),
      "Infinity"
    );
    // Assuming propagates the assumption to an inner Simplify.
    assert_eq!(
      interpret("Assuming[x > 0, Simplify[x*Infinity]]").unwrap(),
      "Infinity"
    );
    // `a >= 0` is not decidable (0 * Infinity is Indeterminate): unchanged.
    assert_eq!(
      interpret("Refine[a*Infinity, a >= 0]").unwrap(),
      "a*Infinity"
    );
  }

  #[test]
  fn sqrt_y_squared_positive() {
    assert_eq!(interpret("Refine[Sqrt[y^2], y > 0]").unwrap(), "y");
  }

  #[test]
  fn abs_x_positive() {
    assert_eq!(interpret("Refine[Abs[x], x > 0]").unwrap(), "x");
  }

  #[test]
  fn abs_y_positive() {
    assert_eq!(interpret("Refine[Abs[y], y > 0]").unwrap(), "y");
  }

  #[test]
  fn assumptions_option_form() {
    // Refine[expr, Assumptions -> cond] behaves like Refine[expr, cond].
    assert_eq!(
      interpret("Refine[Abs[x], Assumptions -> x > 0]").unwrap(),
      "x"
    );
    assert_eq!(
      interpret("Refine[Sqrt[x^2], Assumptions -> x > 0]").unwrap(),
      "x"
    );
    assert_eq!(
      interpret("Refine[Sign[x], Assumptions -> x > 0]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Refine[Floor[x], Assumptions -> Element[x, Integers]]")
        .unwrap(),
      "x"
    );
  }

  #[test]
  fn sqrt_x_squared_no_assumption() {
    // Without assumptions, Sqrt[x^2] should stay as Sqrt[x^2]
    assert_eq!(interpret("Sqrt[x^2]").unwrap(), "Sqrt[x^2]");
  }

  #[test]
  fn sqrt_integer_squared() {
    // Sqrt[4] = 2, Sqrt[9] = 3 (exact integers, no assumptions needed)
    assert_eq!(interpret("Sqrt[4]").unwrap(), "2");
    assert_eq!(interpret("Sqrt[9]").unwrap(), "3");
  }

  #[test]
  fn sqrt_known_non_negative_squared() {
    // Sqrt[(positive_constant)^2] should simplify without Refine
    assert_eq!(interpret("Sqrt[Pi^2]").unwrap(), "Pi");
  }

  #[test]
  fn refine_nested_expression() {
    assert_eq!(
      interpret("Refine[Sqrt[x^2] + Sqrt[y^2], x > 0 && y > 0]").unwrap(),
      "x + y"
    );
  }

  #[test]
  fn refine_no_simplification_needed() {
    // Expression that doesn't benefit from assumptions
    assert_eq!(interpret("Refine[x + 1, x > 0]").unwrap(), "1 + x");
  }

  #[test]
  fn sqrt_x_squared_x_gt_positive() {
    // x > 2 implies x > 0, so Sqrt[x^2] → x
    assert_eq!(interpret("Refine[Sqrt[x^2], x > 2]").unwrap(), "x");
  }

  #[test]
  fn sqrt_x_squared_x_ge_positive() {
    // x >= 5 implies x > 0, so Sqrt[x^2] → x
    assert_eq!(interpret("Refine[Sqrt[x^2], x >= 5]").unwrap(), "x");
  }

  #[test]
  fn sqrt_x_squared_positive_lt_x() {
    // 3 < x implies x > 0, so Sqrt[x^2] → x
    assert_eq!(interpret("Refine[Sqrt[x^2], 3 < x]").unwrap(), "x");
  }

  #[test]
  fn sqrt_x_squared_positive_le_x() {
    // 1 <= x implies x > 0, so Sqrt[x^2] → x
    assert_eq!(interpret("Refine[Sqrt[x^2], 1 <= x]").unwrap(), "x");
  }

  #[test]
  fn abs_x_gt_positive() {
    // x > 7 implies x > 0, so Abs[x] → x
    assert_eq!(interpret("Refine[Abs[x], x > 7]").unwrap(), "x");
  }

  // --- Single argument ---

  #[test]
  fn single_arg_symbol() {
    assert_eq!(interpret("Refine[x]").unwrap(), "x");
  }

  #[test]
  fn single_arg_numeric() {
    assert_eq!(interpret("Refine[Abs[2]]").unwrap(), "2");
  }

  // --- Negative variable assumptions ---

  #[test]
  fn abs_x_negative() {
    assert_eq!(interpret("Refine[Abs[x], x < 0]").unwrap(), "-x");
  }

  #[test]
  fn sqrt_x_squared_negative() {
    assert_eq!(interpret("Refine[Sqrt[x^2], x < 0]").unwrap(), "-x");
  }

  // --- General (x^n)^(1/m) simplification ---

  #[test]
  fn cube_root_x_cubed_positive() {
    assert_eq!(interpret("Refine[(x^3)^(1/3), x >= 0]").unwrap(), "x");
  }

  #[test]
  fn fifth_root_x_fifth_positive() {
    assert_eq!(interpret("Refine[(x^5)^(1/5), x >= 0]").unwrap(), "x");
  }

  #[test]
  fn fourth_root_x_fourth_positive() {
    assert_eq!(interpret("Refine[(x^4)^(1/4), x >= 0]").unwrap(), "x");
  }

  #[test]
  fn fourth_root_x_fourth_negative() {
    // x^4 is always positive, (x^4)^(1/4) = |x| = -x when x < 0
    assert_eq!(interpret("Refine[(x^4)^(1/4), x < 0]").unwrap(), "-x");
  }

  #[test]
  fn sixth_power_cube_root_positive() {
    // (x^6)^(1/3) = x^2 when x >= 0
    assert_eq!(interpret("Refine[(x^6)^(1/3), x >= 0]").unwrap(), "x^2");
  }

  #[test]
  fn sixth_power_cube_root_negative() {
    // (x^6)^(1/3) = x^2 when x < 0 (x^6 always positive, result is |x|^2 = x^2)
    assert_eq!(interpret("Refine[(x^6)^(1/3), x < 0]").unwrap(), "x^2");
  }

  // --- Sign function ---

  #[test]
  fn sign_x_positive() {
    assert_eq!(interpret("Refine[Sign[x], x > 0]").unwrap(), "1");
  }

  #[test]
  fn sign_x_negative() {
    assert_eq!(interpret("Refine[Sign[x], x < 0]").unwrap(), "-1");
  }

  #[test]
  fn sign_x_gt_5() {
    // x > 5 implies positive
    assert_eq!(interpret("Refine[Sign[x], x > 5]").unwrap(), "1");
  }

  // --- Arg function ---

  #[test]
  fn arg_x_positive() {
    assert_eq!(interpret("Refine[Arg[x], x > 0]").unwrap(), "0");
  }

  #[test]
  fn arg_x_negative() {
    assert_eq!(interpret("Refine[Arg[x], x < 0]").unwrap(), "Pi");
  }

  // --- Re/Im with Element assumptions ---

  #[test]
  fn re_x_real() {
    assert_eq!(interpret("Refine[Re[x], Element[x, Reals]]").unwrap(), "x");
  }

  #[test]
  fn im_x_real() {
    assert_eq!(interpret("Refine[Im[x], Element[x, Reals]]").unwrap(), "0");
  }

  // --- Abs[u]^(even) → u^(even) for real u ---

  #[test]
  fn abs_squared_real() {
    assert_eq!(
      interpret("Refine[Abs[x]^2, Element[x, Reals]]").unwrap(),
      "x^2"
    );
    assert_eq!(
      interpret("Simplify[Abs[x]^2, Element[x, Reals]]").unwrap(),
      "x^2"
    );
  }

  #[test]
  fn abs_higher_even_power_real() {
    assert_eq!(
      interpret("Refine[Abs[x]^4, Element[x, Reals]]").unwrap(),
      "x^4"
    );
    assert_eq!(
      interpret("Refine[Abs[x]^6, Element[x, Reals]]").unwrap(),
      "x^6"
    );
  }

  // Odd powers of Abs stay, even with a real assumption.
  #[test]
  fn abs_odd_power_real_unchanged() {
    assert_eq!(
      interpret("Simplify[Abs[x]^3, Element[x, Reals]]").unwrap(),
      "Abs[x]^3"
    );
  }

  // The Abs argument may itself be a real expression.
  #[test]
  fn abs_squared_real_expression() {
    assert_eq!(
      interpret("Simplify[Abs[x + 1]^2, Element[x, Reals]]").unwrap(),
      "(1 + x)^2"
    );
    assert_eq!(
      interpret("Simplify[Abs[2 x]^2, Element[x, Reals]]").unwrap(),
      "4*x^2"
    );
  }

  // Without any real assumption the form is preserved (x may be complex).
  #[test]
  fn abs_squared_no_assumption_unchanged() {
    assert_eq!(interpret("Simplify[Abs[x]^2]").unwrap(), "Abs[x]^2");
  }

  // --- Floor/Ceiling with Element[x, Integers] ---

  #[test]
  fn floor_x_integer() {
    assert_eq!(
      interpret("Refine[Floor[x], Element[x, Integers]]").unwrap(),
      "x"
    );
  }

  #[test]
  fn ceiling_x_integer() {
    assert_eq!(
      interpret("Refine[Ceiling[x], Element[x, Integers]]").unwrap(),
      "x"
    );
  }

  // --- Inequality simplification under assumptions ---

  #[test]
  fn x_gt_0_given_x_gt_1() {
    assert_eq!(interpret("Refine[x > 0, x > 1]").unwrap(), "True");
  }

  #[test]
  fn x_lt_0_given_x_gt_1() {
    assert_eq!(interpret("Refine[x < 0, x > 1]").unwrap(), "False");
  }

  #[test]
  fn x_geq_0_given_x_gt_0() {
    assert_eq!(interpret("Refine[x >= 0, x > 0]").unwrap(), "True");
  }

  #[test]
  fn same_inequality() {
    assert_eq!(interpret("Refine[x > 0, x > 0]").unwrap(), "True");
  }

  // --- Compound assumptions ---

  #[test]
  fn abs_sum_compound() {
    assert_eq!(
      interpret("Refine[Abs[x] + Abs[y], x > 0 && y > 0]").unwrap(),
      "x + y"
    );
  }

  #[test]
  fn abs_product_positive() {
    assert_eq!(
      interpret("Refine[Abs[x*y], x > 0 && y > 0]").unwrap(),
      "x*y"
    );
  }

  // --- Positive var implied by Element and inequality ---

  #[test]
  fn abs_x_reals_nonneg() {
    assert_eq!(
      interpret("Refine[Abs[x], Element[x, Reals] && x >= 0]").unwrap(),
      "x"
    );
  }

  // --- Element with Alternatives pattern ---

  #[test]
  fn element_alternatives_reals() {
    assert_eq!(
      interpret("Refine[Re[a + b I], Element[a | b, Reals]]").unwrap(),
      "a"
    );
  }

  #[test]
  fn element_alternatives_integers() {
    assert_eq!(
      interpret("Refine[Floor[x], Element[x, Integers]]").unwrap(),
      "x"
    );
  }

  // --- Sqrt[x^2] with x ∈ Reals → Abs[x] ---

  #[test]
  fn sqrt_x_squared_real() {
    assert_eq!(
      interpret("Refine[Sqrt[x^2], Element[x, Reals]]").unwrap(),
      "Abs[x]"
    );
  }

  // --- Power rules ---

  #[test]
  fn power_of_power_bounded_exp() {
    assert_eq!(interpret("Refine[(a^b)^c, -1 < b < 1]").unwrap(), "a^(b*c)");
  }

  #[test]
  fn combine_power_product_positive_bases() {
    assert_eq!(
      interpret("Refine[a^p b^p, a > 0 && b > 0]").unwrap(),
      "(a*b)^p"
    );
  }

  // --- Log simplifications ---

  #[test]
  fn log_negative_var() {
    assert_eq!(
      interpret("Refine[Log[x], x < 0]").unwrap(),
      "I*Pi + Log[-x]"
    );
  }

  #[test]
  fn log_power_bounded_exp() {
    assert_eq!(
      interpret("Refine[Log[x^p], -1 < p < 1]").unwrap(),
      "p*Log[x]"
    );
  }

  // Log[E^y] -> y when y is known real (the exponent's principal log).
  #[test]
  fn log_exp_real_var() {
    assert_eq!(
      interpret("Refine[Log[E^x], Element[x, Reals]]").unwrap(),
      "x"
    );
  }

  #[test]
  fn log_exp_positive_var() {
    assert_eq!(interpret("Refine[Log[E^x], x > 0]").unwrap(), "x");
  }

  #[test]
  fn log_exp_real_linear_exponent() {
    assert_eq!(
      interpret("Refine[Log[E^(2 x)], Element[x, Reals]]").unwrap(),
      "2*x"
    );
    assert_eq!(
      interpret("Refine[Log[Exp[x]], Element[x, Reals]]").unwrap(),
      "x"
    );
  }

  #[test]
  fn log_exp_unknown_var_unchanged() {
    // No realness assumption: the 2*Pi*I branch ambiguity remains, so the
    // logarithm must not be simplified.
    assert_eq!(interpret("Refine[Log[E^x], True]").unwrap(), "Log[E^x]");
    assert_eq!(
      interpret("Refine[Log[E^x], Element[x, Complexes]]").unwrap(),
      "Log[E^x]"
    );
  }

  // --- Conjugate of a real-valued expression ---

  // Conjugate[x] -> x when x is known real (Element[x, Reals], x > 0, ...).
  #[test]
  fn conjugate_real_var() {
    assert_eq!(
      interpret("Refine[Conjugate[x], Element[x, Reals]]").unwrap(),
      "x"
    );
    assert_eq!(interpret("Refine[Conjugate[x], x > 0]").unwrap(), "x");
  }

  #[test]
  fn conjugate_real_compound() {
    assert_eq!(
      interpret(
        "Refine[Conjugate[x + y], Element[x, Reals] && Element[y, Reals]]"
      )
      .unwrap(),
      "x + y"
    );
    assert_eq!(
      interpret("Refine[Conjugate[2 x], Element[x, Reals]]").unwrap(),
      "2*x"
    );
    assert_eq!(
      interpret("Refine[Conjugate[x^2], Element[x, Reals]]").unwrap(),
      "x^2"
    );
  }

  #[test]
  fn conjugate_unknown_or_imaginary_unchanged() {
    // No realness assumption: Conjugate stays put.
    assert_eq!(
      interpret("Refine[Conjugate[x], Element[x, Complexes]]").unwrap(),
      "Conjugate[x]"
    );
    // I*x with x real is imaginary, so Conjugate flips its sign rather than
    // vanishing.
    assert_eq!(
      interpret("Refine[Conjugate[I x], Element[x, Reals]]").unwrap(),
      "-I*x"
    );
  }

  // --- Sign predicates under assumptions ---

  #[test]
  fn positive_predicate() {
    assert_eq!(interpret("Refine[Positive[x], x > 0]").unwrap(), "True");
    assert_eq!(interpret("Refine[Positive[x], x < 0]").unwrap(), "False");
    assert_eq!(interpret("Refine[Positive[x], x <= 0]").unwrap(), "False");
    assert_eq!(interpret("Refine[Positive[x + 1], x > 0]").unwrap(), "True");
    assert_eq!(interpret("Refine[Positive[2 x], x > 0]").unwrap(), "True");
  }

  #[test]
  fn negative_predicate() {
    assert_eq!(interpret("Refine[Negative[x], x < 0]").unwrap(), "True");
    assert_eq!(interpret("Refine[Negative[x], x > 0]").unwrap(), "False");
    assert_eq!(interpret("Refine[Negative[-x], x > 0]").unwrap(), "True");
  }

  #[test]
  fn nonnegative_predicate() {
    assert_eq!(interpret("Refine[NonNegative[x], x >= 0]").unwrap(), "True");
    assert_eq!(interpret("Refine[NonNegative[x], x < 0]").unwrap(), "False");
    assert_eq!(
      interpret("Refine[NonNegative[x^2], Element[x, Reals]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn nonpositive_predicate() {
    assert_eq!(interpret("Refine[NonPositive[x], x < 0]").unwrap(), "True");
    assert_eq!(interpret("Refine[NonPositive[x], x <= 0]").unwrap(), "True");
  }

  // x <= 0 means non-positive, NOT strictly negative: these stay unevaluated
  // (x could be 0), and Sqrt[x^2] still collapses to -x.
  #[test]
  fn nonpositive_assumption_is_not_strict() {
    assert_eq!(
      interpret("Refine[Negative[x], x <= 0]").unwrap(),
      "Negative[x]"
    );
    assert_eq!(
      interpret("Refine[NonNegative[x], x <= 0]").unwrap(),
      "NonNegative[x]"
    );
    assert_eq!(interpret("Refine[Sqrt[x^2], x <= 0]").unwrap(), "-x");
    assert_eq!(interpret("Refine[Abs[x], x <= 0]").unwrap(), "-x");
  }

  // --- Trig with integer multiples of Pi ---

  #[test]
  fn sin_k_pi_integer() {
    assert_eq!(
      interpret("Refine[Sin[k Pi], Element[k, Integers]]").unwrap(),
      "0"
    );
  }

  #[test]
  fn cos_x_plus_k_pi_integer() {
    assert_eq!(
      interpret("Refine[Cos[x + k Pi], Element[k, Integers]]").unwrap(),
      "(-1)^k*Cos[x]"
    );
  }

  // --- ArcTan[Tan[x]] ---

  #[test]
  fn arctan_tan_in_range() {
    assert_eq!(
      interpret("Refine[ArcTan[Tan[x]], -Pi/2 < Re[x] < Pi/2]").unwrap(),
      "x"
    );
  }

  // --- Algebraic comparisons ---

  #[test]
  fn equation_by_substitution() {
    assert_eq!(
      interpret("Refine[a^2 - b^2 + 1 == 0, a + b == 0]").unwrap(),
      "False"
    );
  }

  #[test]
  fn quadratic_form_nonneg() {
    assert_eq!(
      interpret("Refine[a^2 - a b + b^2 >= 0, Element[a | b, Reals]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn sign_positive_definite() {
    assert_eq!(
      interpret("Refine[Sign[x^2 - x y + y^2 + 1], Element[x | y, Reals]]")
        .unwrap(),
      "1"
    );
  }

  // --- Element membership ---

  #[test]
  fn element_real_positive_division() {
    assert_eq!(
      interpret(
        "Refine[Element[(2 x + x^p)/(x Gamma[x + 2]), Reals], x > 0 && p > 0]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn element_integer_floor_power() {
    assert_eq!(
      interpret("Refine[Element[2 k^3 Floor[x]^k, Integers], Element[k, Integers] && k > 0 && Element[x, Reals]]").unwrap(),
      "True"
    );
  }

  // --- Floor/Ceiling with compound expressions ---

  #[test]
  fn floor_integer_linear() {
    assert_eq!(
      interpret("Refine[Floor[2 a + 1], Element[a, Integers]]").unwrap(),
      "1 + 2*a"
    );
  }

  #[test]
  fn ceiling_bounded_var() {
    assert_eq!(interpret("Refine[Ceiling[x], 2 < x <= 3]").unwrap(), "3");
  }

  // --- FractionalPart and Mod ---

  #[test]
  fn fractional_part_negative_with_mod() {
    assert_eq!(
      interpret("Refine[FractionalPart[a], a < 0 && Mod[a, 1] == 1/3]")
        .unwrap(),
      "-2/3"
    );
  }

  #[test]
  fn mod_from_integer_element() {
    assert_eq!(
      interpret("Refine[Mod[a, 4], Element[(a + 3)/4, Integers]]").unwrap(),
      "1"
    );
  }

  // --- Assuming + Refine ---

  #[test]
  fn assuming_refine_compound_comparison() {
    assert_eq!(
      interpret("Assuming[x >= 0 && y < 0, Refine[x - y > 0]]").unwrap(),
      "True"
    );
  }

  // --- Nonnegative variable handling ---

  #[test]
  fn sqrt_x_squared_nonneg() {
    assert_eq!(interpret("Refine[Sqrt[x^2], x >= 0]").unwrap(), "x");
  }

  #[test]
  fn inequality_false_under_sum_of_squares_constraint() {
    // (x-1)^2 + (y-2)^2 >= 2 when x^2 + y^2 <= 1, so < 3/2 is False
    assert_eq!(
      interpret("Refine[(x - 1)^2 + (y - 2)^2 < 3/2, x^2 + y^2 <= 1]").unwrap(),
      "False"
    );
  }
}

mod simplify_solve_verification {
  use super::*;

  #[test]
  fn simplify_quadratic_formula_substitution() {
    // Substituting quadratic formula roots back into polynomial should give 0
    assert_eq!(
      interpret(
        "sol = Solve[a x^2 + b x + c == 0, x]; Simplify[a x^2 + b x + c /. sol]"
      )
      .unwrap(),
      "{0, 0}"
    );
  }

  #[test]
  fn simplify_threads_over_list() {
    assert_eq!(interpret("Simplify[{1 + 1, 2 + 3}]").unwrap(), "{2, 5}");
  }

  #[test]
  fn together_fraction_power() {
    // Together should correctly handle (sum/product)^n terms
    assert_eq!(
      interpret("Together[a*(x/a)^2 + x]").unwrap(),
      "(a*x + x^2)/a"
    );
  }
}

mod expand_fraction_power {
  use super::*;

  #[test]
  fn expand_fraction_squared() {
    // (x+y)^2/z^2 expanded, displaying negative exponents as fractions
    assert_eq!(
      interpret("Expand[((x + y)/z)^2]").unwrap(),
      "x^2/z^2 + (2*x*y)/z^2 + y^2/z^2"
    );
  }

  #[test]
  fn expand_product_power() {
    assert_eq!(interpret("Expand[(2*a)^2]").unwrap(), "4*a^2");
  }

  #[test]
  fn expand_product_power_three() {
    assert_eq!(interpret("Expand[(3*x)^3]").unwrap(), "27*x^3");
  }

  #[test]
  fn expand_high_power_bigint_coefficients() {
    // Regression: Expand[(1 + x)^200] panicked with i128 overflow once the
    // binomial coefficients exceeded i128 (~p >= 132), then produced
    // uncombined/wrong terms. It must promote to BigInteger and stay exact.
    // 201 distinct powers, and the result equals the binomial sum.
    assert_eq!(interpret("Length[Expand[(1 + x)^200]]").unwrap(), "201");
    assert_eq!(
      interpret(
        "Expand[(1 + x)^200] - Sum[Binomial[200, k] x^k, {k, 0, 200}] // Expand"
      )
      .unwrap(),
      "0"
    );
    // Small integer cases are unchanged.
    assert_eq!(
      interpret("Expand[(1 + x)^4]").unwrap(),
      "1 + 4*x + 6*x^2 + 4*x^3 + x^4"
    );
  }

  #[test]
  fn expand_rational_coefficient_folds_term_numerics() {
    // Regression: a rational leading coefficient times a binomial power left
    // each cross-term with an uncombined numeric product (e.g. 15*2 instead
    // of 30). The per-term numeric fold now matches wolframscript.
    assert_eq!(
      interpret("Expand[15 (-1 + x)^2 / 4]").unwrap(),
      "15/4 - (15*x)/2 + (15*x^2)/4"
    );
    assert_eq!(
      interpret("Expand[15 (-1 + x)^2 / 2]").unwrap(),
      "15/2 - 15*x + (15*x^2)/2"
    );
    // A single foldable cross term collapses cleanly.
    assert_eq!(interpret("Expand[(x/2 + 1)^2]").unwrap(), "1 + x + x^2/4");
    assert_eq!(
      interpret("Expand[6 (x + 1/2)^2]").unwrap(),
      "3/2 + 6*x + 6*x^2"
    );
    // Integer-coefficient and multivariate expansions are unchanged.
    assert_eq!(
      interpret("Expand[15 (-1 + x)^2]").unwrap(),
      "15 - 30*x + 15*x^2"
    );
    assert_eq!(
      interpret("Expand[(a + b + c)^2]").unwrap(),
      "a^2 + 2*a*b + b^2 + 2*a*c + 2*b*c + c^2"
    );
  }

  #[test]
  fn expand_rational_coefficient_inputform_subtracts() {
    // Regression: a Plus term with a negative Rational/Real coefficient
    // (`Times[Rational[-15, 2], x]`) rendered in InputForm as an addition of a
    // negative coefficient (`+ (-15*x)/2`) instead of a subtraction. The
    // BinaryOp::Times branch of the Plus InputForm renderer only pulled the
    // sign out for negative Integer coefficients; now it handles negative
    // Rational and Real coefficients too, matching wolframscript.
    assert_eq!(
      interpret("ToString[Expand[15 (-1 + x)^2 / 4], InputForm]").unwrap(),
      "15/4 - (15*x)/2 + (15*x^2)/4"
    );
    assert_eq!(
      interpret("ToString[3/4 + (-1/2) x, InputForm]").unwrap(),
      "3/4 - x/2"
    );
    assert_eq!(
      interpret("ToString[Expand[1.5 (x - 1)], InputForm]").unwrap(),
      "-1.5 + 1.5*x"
    );
  }
}

mod root {
  use super::*;

  #[test]
  fn sqrt_2_first_root() {
    assert_eq!(interpret("Root[#^2 - 2 &, 1]").unwrap(), "-Sqrt[2]");
  }

  #[test]
  fn sqrt_2_second_root() {
    assert_eq!(interpret("Root[#^2 - 2 &, 2]").unwrap(), "Sqrt[2]");
  }

  #[test]
  fn linear() {
    assert_eq!(interpret("Root[# &, 1]").unwrap(), "0");
  }

  #[test]
  fn quadratic_integer_roots() {
    assert_eq!(interpret("Root[#^2 - 3*# + 2 &, 1]").unwrap(), "1");
    assert_eq!(interpret("Root[#^2 - 3*# + 2 &, 2]").unwrap(), "2");
  }

  #[test]
  fn complex_roots() {
    assert_eq!(interpret("Root[#^2 + 1 &, 1]").unwrap(), "-I");
    assert_eq!(interpret("Root[#^2 + 1 &, 2]").unwrap(), "I");
  }

  #[test]
  fn fourth_roots_of_unity_minus_one() {
    // x^4 - 1: roots are -1, 1, -I, I
    assert_eq!(interpret("Root[#^4 - 1 &, 1]").unwrap(), "-1");
    assert_eq!(interpret("Root[#^4 - 1 &, 2]").unwrap(), "1");
    assert_eq!(interpret("Root[#^4 - 1 &, 3]").unwrap(), "-I");
    assert_eq!(interpret("Root[#^4 - 1 &, 4]").unwrap(), "I");
  }

  #[test]
  fn cubic_with_real_root() {
    // x^3 - 1: real root is 1
    assert_eq!(interpret("Root[#^3 - 1 &, 1]").unwrap(), "1");
  }

  #[test]
  fn numerical_value() {
    let result = interpret("N[Root[#^2 - 2 &, 1]]").unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!(
      (val + std::f64::consts::SQRT_2).abs() < 1e-10,
      "Expected -1.414..., got {}",
      val
    );
  }

  // N of a higher-degree Root (no radical form) numerically finds the k-th
  // root: real roots first in increasing order.
  #[test]
  fn numerical_value_quintic_real_root() {
    let result = interpret("N[Root[#^5 - # - 1 &, 1]]").unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!(
      (val - 1.1673039782614187).abs() < 1e-10,
      "Expected 1.1673..., got {}",
      val
    );
  }

  // A quartic with four real roots is returned in increasing order.
  #[test]
  fn numerical_value_quartic_all_real() {
    assert_eq!(
      interpret("Table[N[Root[#^4 - 5 #^2 + 4 &, k]], {k, 1, 4}]").unwrap(),
      "{-2., -1., 1., 2.}"
    );
  }

  // Complex roots follow the reals, ordered by increasing real then
  // imaginary part. Re/Im are checked to avoid last-digit float noise.
  #[test]
  fn numerical_value_complex_root() {
    let re = interpret("Re[N[Root[#^3 - 2 &, 2]]]")
      .unwrap()
      .parse::<f64>()
      .unwrap();
    let im = interpret("Im[N[Root[#^3 - 2 &, 2]]]")
      .unwrap()
      .parse::<f64>()
      .unwrap();
    assert!((re - (-0.6299605249474366)).abs() < 1e-10);
    assert!((im - (-1.0911236359717214)).abs() < 1e-10);
  }

  #[test]
  fn out_of_range_error() {
    assert!(interpret("Root[#^2 - 1 &, 3]").is_err());
  }
}

mod root_sum {
  use super::*;

  // RootSum is not implemented; the call stays symbolic. Woxi's output
  // matches wolframscript byte-for-byte, including the formatting of
  // nested pure functions.
  #[test]
  fn irreducible_cyclotomic_stays_symbolic() {
    assert_eq!(
      interpret("RootSum[1+#+#^2+#^3+#^4 &, Log[x + #] &]").unwrap(),
      "RootSum[1 + #1 + #1^2 + #1^3 + #1^4 & , Log[x + #1] & ]"
    );
  }

  // For a polynomial f with numeric coefficients and a polynomial form, the
  // sum over the roots is a symmetric function obtainable from Newton's power
  // sums — an exact rational that needs no explicit roots (e.g. it works even
  // for the unsolvable quintic below).
  #[test]
  fn sum_of_roots() {
    // Sum of the roots of x^2 - 2 is 0 (no x term).
    assert_eq!(interpret("RootSum[#^2 - 2 &, # &]").unwrap(), "0");
    assert_eq!(interpret("RootSum[#^3 - 2 &, # &]").unwrap(), "0");
    assert_eq!(interpret("RootSum[#^4 - 1 &, # &]").unwrap(), "0");
  }

  #[test]
  fn sum_of_powers_of_roots() {
    // Roots 1, 2: sum of squares = 5, sum of cubes = 35.
    assert_eq!(interpret("RootSum[#^2 - 3 # + 2 &, #^2 &]").unwrap(), "5");
    assert_eq!(interpret("RootSum[#^2 - 5 # + 6 &, #^3 &]").unwrap(), "35");
    // Sum of squares of the roots of x^2 - 2 is 2 + 2 = 4.
    assert_eq!(interpret("RootSum[#^2 - 2 &, #^2 &]").unwrap(), "4");
  }

  #[test]
  fn unsolvable_cubic_and_quintic_power_sums() {
    // x^3 - x - 1 (no closed-form roots): sum of squares = e1^2 - 2 e2 = 2.
    assert_eq!(interpret("RootSum[#^3 - # - 1 &, #^2 &]").unwrap(), "2");
    assert_eq!(interpret("RootSum[#^3 + # + 1 &, #^2 &]").unwrap(), "-2");
    // x^5 - x - 1 (unsolvable by radicals): sum of squares = 0.
    assert_eq!(interpret("RootSum[#^5 - # - 1 &, #^2 &]").unwrap(), "0");
  }

  #[test]
  fn non_monic_and_affine_form() {
    // 2 x^2 - 8 has roots ±2; sum of squares = 8.
    assert_eq!(interpret("RootSum[2 #^2 - 8 &, #^2 &]").unwrap(), "8");
    // form 3 #^2 + 1: 3*(sum of squares) + 1*(root count) = 3*4 + 2 = 14.
    assert_eq!(interpret("RootSum[#^2 - 2 &, 3 #^2 + 1 &]").unwrap(), "14");
    // Constant form sums to constant * (number of roots).
    assert_eq!(interpret("RootSum[#^2 - 2 &, 5 &]").unwrap(), "10");
    // Linear polynomial: single root 3, squared = 9.
    assert_eq!(interpret("RootSum[# - 3 &, #^2 &]").unwrap(), "9");
  }
}

mod polynomial_mod {
  use super::*;

  #[test]
  fn basic_cubic() {
    assert_eq!(
      interpret("PolynomialMod[x^3 + 2x + 1, 3]").unwrap(),
      "1 + 2*x + x^3"
    );
  }

  #[test]
  fn all_coefficients_zero() {
    assert_eq!(interpret("PolynomialMod[6x^2 + 9x + 12, 3]").unwrap(), "0");
  }

  #[test]
  fn mod_one() {
    assert_eq!(interpret("PolynomialMod[x + y + z, 1]").unwrap(), "0");
  }

  #[test]
  fn constant_polynomial() {
    assert_eq!(interpret("PolynomialMod[7, 3]").unwrap(), "1");
  }

  #[test]
  fn zero_polynomial() {
    assert_eq!(interpret("PolynomialMod[0, 5]").unwrap(), "0");
  }

  #[test]
  fn with_large_coefficients() {
    assert_eq!(
      interpret("PolynomialMod[5x^2 + 3x + 7, 4]").unwrap(),
      "3 + 3*x + x^2"
    );
  }

  #[test]
  fn multivariate() {
    assert_eq!(interpret("PolynomialMod[3x + 4y + 5, 3]").unwrap(), "2 + y");
  }

  #[test]
  fn symbolic_modulus_unevaluated() {
    assert_eq!(interpret("PolynomialMod[x^2, m]").unwrap(), "x^2");
  }
}

mod interpolating_polynomial {
  use super::*;

  #[test]
  fn quadratic_explicit_points() {
    // Through (1,1),(2,4),(3,9) → x^2
    assert_eq!(
      interpret("Expand[InterpolatingPolynomial[{{1,1},{2,4},{3,9}}, x]]")
        .unwrap(),
      "x^2"
    );
  }

  #[test]
  fn quadratic_implicit_points() {
    // {1,4,9} at x=1,2,3 → x^2
    assert_eq!(
      interpret("Expand[InterpolatingPolynomial[{1,4,9}, x]]").unwrap(),
      "x^2"
    );
  }

  #[test]
  fn linear_two_points() {
    // Through (0,0),(1,1) → x
    assert_eq!(
      interpret("InterpolatingPolynomial[{{0,0},{1,1}}, x]").unwrap(),
      "x"
    );
  }

  #[test]
  fn constant_one_point() {
    // Single point → constant
    assert_eq!(
      interpret("InterpolatingPolynomial[{{5,3}}, x]").unwrap(),
      "3"
    );
  }

  #[test]
  fn cubic_values() {
    // {0,1,8,27} at x=1,2,3,4 should give (x-1)^3
    let result =
      interpret("Expand[InterpolatingPolynomial[{0, 1, 8, 27}, x]]").unwrap();
    assert_eq!(result, "-1 + 3*x - 3*x^2 + x^3");
  }

  #[test]
  fn newton_form_structure() {
    // InterpolatingPolynomial[{{1,1},{2,4},{3,9}}, x]
    // Newton form: 1 + (x-1)*(1 + 3*(x-2)) = 1 + (x-1)*(1 + 3x - 6) = 1 + (-1+x)*(3*(-1+x) - 2)
    let _result =
      interpret("InterpolatingPolynomial[{{1,1},{2,4},{3,9}}, x]").unwrap();
    // Just verify it evaluates to correct values
    let at1 =
      interpret("InterpolatingPolynomial[{{1,1},{2,4},{3,9}}, x] /. x -> 1")
        .unwrap();
    let at2 =
      interpret("InterpolatingPolynomial[{{1,1},{2,4},{3,9}}, x] /. x -> 2")
        .unwrap();
    let at3 =
      interpret("InterpolatingPolynomial[{{1,1},{2,4},{3,9}}, x] /. x -> 3")
        .unwrap();
    assert_eq!(at1, "1");
    assert_eq!(at2, "4");
    assert_eq!(at3, "9");
  }

  #[test]
  fn linear_three_collinear() {
    // Through (0,0),(1,2),(2,4) → 2x
    assert_eq!(
      interpret("Expand[InterpolatingPolynomial[{{0,0},{1,2},{2,4}}, x]]")
        .unwrap(),
      "2*x"
    );
  }

  #[test]
  fn non_list_unevaluated() {
    assert_eq!(
      interpret("InterpolatingPolynomial[5, x]").unwrap(),
      "InterpolatingPolynomial[5, x]"
    );
  }
}

mod minimal_polynomial {
  use super::*;

  #[test]
  fn integer() {
    assert_eq!(interpret("MinimalPolynomial[3, x]").unwrap(), "-3 + x");
  }

  #[test]
  fn rational() {
    assert_eq!(interpret("MinimalPolynomial[1/3, x]").unwrap(), "-1 + 3*x");
  }

  #[test]
  fn sqrt_2() {
    assert_eq!(
      interpret("MinimalPolynomial[Sqrt[2], x]").unwrap(),
      "-2 + x^2"
    );
  }

  #[test]
  fn sqrt_3() {
    assert_eq!(
      interpret("MinimalPolynomial[Sqrt[3], x]").unwrap(),
      "-3 + x^2"
    );
  }

  #[test]
  fn cube_root_2() {
    assert_eq!(
      interpret("MinimalPolynomial[2^(1/3), x]").unwrap(),
      "-2 + x^3"
    );
  }

  #[test]
  fn golden_ratio() {
    assert_eq!(
      interpret("MinimalPolynomial[GoldenRatio, x]").unwrap(),
      "-1 - x + x^2"
    );
  }

  #[test]
  fn imaginary_unit() {
    assert_eq!(interpret("MinimalPolynomial[I, x]").unwrap(), "1 + x^2");
  }

  #[test]
  fn sum_of_square_roots() {
    assert_eq!(
      interpret("MinimalPolynomial[Sqrt[2] + Sqrt[3], x]").unwrap(),
      "1 - 10*x^2 + x^4"
    );
  }

  #[test]
  fn scaled_sqrt() {
    assert_eq!(
      interpret("MinimalPolynomial[2*Sqrt[3], x]").unwrap(),
      "-12 + x^2"
    );
  }

  #[test]
  fn one_plus_sqrt_2() {
    assert_eq!(
      interpret("MinimalPolynomial[1 + Sqrt[2], x]").unwrap(),
      "-1 - 2*x + x^2"
    );
  }

  // A quotient that stays a Divide BinaryOp (e.g. Cos[Pi/5] → (1+Sqrt[5])/4)
  // is rewritten as a product a * b^-1, so its minimal polynomial is found.
  #[test]
  fn divide_form() {
    assert_eq!(
      interpret("MinimalPolynomial[Cos[Pi/5], x]").unwrap(),
      "-1 - 2*x + 4*x^2"
    );
    assert_eq!(
      interpret("MinimalPolynomial[Cos[2*Pi/5], x]").unwrap(),
      "-1 + 2*x + 4*x^2"
    );
    assert_eq!(
      interpret("MinimalPolynomial[(3 + Sqrt[2])/7, x]").unwrap(),
      "1 - 6*x + 7*x^2"
    );
  }

  // The reciprocal of an algebraic number (e.g. Cos[Pi/4] = Sqrt[2]^-1) has the
  // coefficient-reversed minimal polynomial of the base.
  #[test]
  fn reciprocal_of_algebraic() {
    assert_eq!(
      interpret("MinimalPolynomial[Cos[Pi/4], x]").unwrap(),
      "-1 + 2*x^2"
    );
    assert_eq!(
      interpret("MinimalPolynomial[1/2^(1/3), x]").unwrap(),
      "-1 + 2*x^3"
    );
    assert_eq!(
      interpret("MinimalPolynomial[1/(1 + Sqrt[2]), x]").unwrap(),
      "-1 + 2*x + x^2"
    );
  }

  // A pure-imaginary radical I*Sqrt[n] has the irreducible degree-2 minimal
  // polynomial x^2 + n. Previously the resultant construction returned the
  // non-minimal perfect power (x^2 + n)^2 because no real numeric value was
  // available to pick the irreducible factor.
  #[test]
  fn imaginary_square_root() {
    assert_eq!(
      interpret("MinimalPolynomial[Sqrt[-2], x]").unwrap(),
      "2 + x^2"
    );
    assert_eq!(
      interpret("MinimalPolynomial[I*Sqrt[3], x]").unwrap(),
      "3 + x^2"
    );
    assert_eq!(
      interpret("MinimalPolynomial[Sqrt[-8], x]").unwrap(),
      "8 + x^2"
    );
  }

  #[test]
  fn wrong_arg_count() {
    // Single argument returns unevaluated or error
    assert!(
      interpret("MinimalPolynomial[Sqrt[2]]").is_err()
        || interpret("MinimalPolynomial[Sqrt[2]]")
          .unwrap()
          .contains("MinimalPolynomial")
    );
  }

  #[test]
  fn negative_sqrt() {
    assert_eq!(
      interpret("MinimalPolynomial[-Sqrt[2], x]").unwrap(),
      "-2 + x^2"
    );
  }
}

mod algebraic_integer_q {
  use super::*;

  // Ordinary integers (including 0 and negatives) are algebraic integers.
  #[test]
  fn integers() {
    assert_eq!(interpret("AlgebraicIntegerQ[5]").unwrap(), "True");
    assert_eq!(interpret("AlgebraicIntegerQ[0]").unwrap(), "True");
    assert_eq!(interpret("AlgebraicIntegerQ[-7]").unwrap(), "True");
  }

  // Non-integer rationals are not algebraic integers.
  #[test]
  fn rationals() {
    assert_eq!(interpret("AlgebraicIntegerQ[1/2]").unwrap(), "False");
    assert_eq!(interpret("AlgebraicIntegerQ[2/3]").unwrap(), "False");
  }

  // Radicals and other algebraic numbers with a monic minimal polynomial.
  #[test]
  fn algebraic_integers() {
    assert_eq!(interpret("AlgebraicIntegerQ[Sqrt[2]]").unwrap(), "True");
    assert_eq!(interpret("AlgebraicIntegerQ[I]").unwrap(), "True");
    assert_eq!(interpret("AlgebraicIntegerQ[GoldenRatio]").unwrap(), "True");
    assert_eq!(
      interpret("AlgebraicIntegerQ[(1 + Sqrt[5])/2]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AlgebraicIntegerQ[Sqrt[2] + Sqrt[3]]").unwrap(),
      "True"
    );
    assert_eq!(interpret("AlgebraicIntegerQ[2^(1/3)]").unwrap(), "True");
    assert_eq!(interpret("AlgebraicIntegerQ[Sqrt[-3]]").unwrap(), "True");
  }

  // Algebraic numbers whose minimal polynomial is not monic.
  #[test]
  fn non_algebraic_integers() {
    assert_eq!(interpret("AlgebraicIntegerQ[Sqrt[2]/2]").unwrap(), "False");
    assert_eq!(interpret("AlgebraicIntegerQ[Sqrt[2]/3]").unwrap(), "False");
  }

  // Transcendentals, free symbols, and inexact numbers are not algebraic
  // integers.
  #[test]
  fn non_algebraic() {
    assert_eq!(interpret("AlgebraicIntegerQ[Pi]").unwrap(), "False");
    assert_eq!(interpret("AlgebraicIntegerQ[x]").unwrap(), "False");
    assert_eq!(interpret("AlgebraicIntegerQ[3.5]").unwrap(), "False");
    assert_eq!(interpret("AlgebraicIntegerQ[2.0]").unwrap(), "False");
  }
}

mod find_instance {
  use super::*;

  #[test]
  fn simple_equation() {
    assert_eq!(
      interpret("FindInstance[x^2 == 4, x]").unwrap(),
      "{{x -> 2}}"
    );
  }

  #[test]
  fn multiple_solutions() {
    assert_eq!(
      interpret("FindInstance[x^2 == 4, x, 3]").unwrap(),
      "{{x -> -2}, {x -> 2}}"
    );
  }

  #[test]
  fn quadratic() {
    assert_eq!(
      interpret("FindInstance[x^2 - 5 x + 6 == 0, x]").unwrap(),
      "{{x -> 3}}"
    );
  }

  #[test]
  fn two_variable_equation() {
    assert_eq!(
      interpret("FindInstance[x^2 + y^2 == 1, {x, y}]").unwrap(),
      "{{x -> -1, y -> 0}}"
    );
  }

  #[test]
  fn integer_domain_inequality() {
    assert_eq!(
      interpret("FindInstance[x > 3 && x < 5, x, Integers]").unwrap(),
      "{{x -> 4}}"
    );
  }

  #[test]
  fn no_solution() {
    assert_eq!(
      interpret("FindInstance[x^2 == -1, x, Reals]").unwrap(),
      "{}"
    );
  }

  #[test]
  fn single_var_in_list() {
    assert_eq!(
      interpret("FindInstance[x^2 == 9, {x}]").unwrap(),
      "{{x -> 3}}"
    );
  }
}

mod find_sequence_function {
  use super::*;

  #[test]
  fn linear_sequence() {
    assert_eq!(
      interpret("FindSequenceFunction[{1, 2, 3, 4, 5}, n]").unwrap(),
      "n"
    );
  }

  #[test]
  fn squares() {
    assert_eq!(
      interpret("FindSequenceFunction[{1, 4, 9, 16, 25}, n]").unwrap(),
      "n^2"
    );
  }

  #[test]
  fn operator_form_applied() {
    // One-argument operator form: FindSequenceFunction[seq][k] is the k-th term.
    assert_eq!(
      interpret("FindSequenceFunction[{1, 4, 9, 16, 25}][6]").unwrap(),
      "36"
    );
  }

  #[test]
  fn operator_form_mapped() {
    // Operator form composes with Map (`/@`).
    assert_eq!(
      interpret("FindSequenceFunction[{1, 2, 4, 7, 11}] /@ Range[6]").unwrap(),
      "{1, 2, 4, 7, 11, 16}"
    );
  }

  #[test]
  fn cubes() {
    assert_eq!(
      interpret("FindSequenceFunction[{1, 8, 27, 64, 125}, n]").unwrap(),
      "n^3"
    );
  }

  #[test]
  fn constant() {
    assert_eq!(
      interpret("FindSequenceFunction[{5, 5, 5, 5}, n]").unwrap(),
      "5"
    );
  }

  #[test]
  fn powers_of_2() {
    assert_eq!(
      interpret("FindSequenceFunction[{2, 4, 8, 16, 32}, n]").unwrap(),
      "2^n"
    );
  }

  #[test]
  fn powers_of_3() {
    assert_eq!(
      interpret("FindSequenceFunction[{3, 9, 27, 81, 243}, n]").unwrap(),
      "3^n"
    );
  }

  #[test]
  fn factorial() {
    assert_eq!(
      interpret("FindSequenceFunction[{1, 2, 6, 24, 120}, n]").unwrap(),
      "n!"
    );
  }

  #[test]
  fn triangular_numbers() {
    // (n*(1+n))/2 expanded form
    assert_eq!(
      interpret("FindSequenceFunction[{1, 3, 6, 10, 15}, n]").unwrap(),
      "n/2 + n^2/2"
    );
  }

  #[test]
  fn formula_is_correct() {
    // Verify the found formula gives correct values by substitution
    assert_eq!(
      interpret("FindSequenceFunction[{1, 4, 9, 16, 25}, n] /. n -> 6")
        .unwrap(),
      "36"
    );
  }
}

mod horner_form {
  use super::*;

  #[test]
  fn basic_univariate() {
    assert_eq!(
      interpret("HornerForm[11 x^3 - 4 x^2 + 7 x + 2]").unwrap(),
      "2 + x*(7 + x*(-4 + 11*x))"
    );
  }

  #[test]
  fn explicit_variable() {
    assert_eq!(
      interpret("HornerForm[a + b x + c x^2, x]").unwrap(),
      "a + x*(b + c*x)"
    );
  }

  #[test]
  fn constant() {
    assert_eq!(interpret("HornerForm[5]").unwrap(), "5");
  }

  #[test]
  fn zero() {
    assert_eq!(interpret("HornerForm[0]").unwrap(), "0");
  }

  #[test]
  fn single_variable() {
    assert_eq!(interpret("HornerForm[x]").unwrap(), "x");
  }

  #[test]
  fn monomial() {
    assert_eq!(interpret("HornerForm[x^3]").unwrap(), "x^3");
  }

  #[test]
  fn linear_polynomial() {
    assert_eq!(interpret("HornerForm[x + 2]").unwrap(), "2 + x");
  }

  #[test]
  fn non_polynomial() {
    assert_eq!(interpret("HornerForm[Sin[x]]").unwrap(), "Sin[x]");
  }

  #[test]
  fn degree_four() {
    assert_eq!(
      interpret("HornerForm[x^4 + 2 x^3 - x + 5]").unwrap(),
      "5 + x*(-1 + x^2*(2 + x))"
    );
  }

  #[test]
  fn univariate_in_a() {
    assert_eq!(
      interpret("HornerForm[a^2 + 3 a + 1]").unwrap(),
      "1 + a*(3 + a)"
    );
  }

  #[test]
  fn multivariate_explicit_x() {
    assert_eq!(
      interpret("HornerForm[x^2 + 2 x y + y^2, x]").unwrap(),
      "y^2 + x*(x + 2*y)"
    );
  }

  #[test]
  fn multivariate_explicit_y() {
    assert_eq!(
      interpret("HornerForm[x^2 + 2 x y + y^2, y]").unwrap(),
      "x^2 + y*(2*x + y)"
    );
  }

  #[test]
  fn unrelated_variable() {
    assert_eq!(
      interpret("HornerForm[3 x^2 + 2 x + 1, y]").unwrap(),
      "1 + 2*x + 3*x^2"
    );
  }

  #[test]
  fn rational_function() {
    let result =
      interpret("HornerForm[(11 x^3 - 4 x^2 + 7 x + 2)/(x^2 - 3 x + 1)]")
        .unwrap();
    assert_eq!(result, "(2 + x*(7 + x*(-4 + 11*x)))/(1 + (-3 + x)*x)");
  }

  #[test]
  fn all_numeric_coefficients() {
    assert_eq!(
      interpret("HornerForm[1 + 2 x + 3 x^2 + 4 x^3]").unwrap(),
      "1 + x*(2 + x*(3 + 4*x))"
    );
  }

  #[test]
  fn quadratic() {
    assert_eq!(
      interpret("HornerForm[x^2 + 5 x + 6]").unwrap(),
      "6 + x*(5 + x)"
    );
  }
}

mod function_expand {
  use super::*;

  #[test]
  fn pochhammer() {
    assert_eq!(
      interpret("FunctionExpand[Pochhammer[a, n]]").unwrap(),
      "Gamma[a + n]/Gamma[a]"
    );
  }

  #[test]
  fn beta() {
    assert_eq!(
      interpret("FunctionExpand[Beta[a, b]]").unwrap(),
      "(Gamma[a]*Gamma[b])/Gamma[a + b]"
    );
  }

  #[test]
  fn binomial_n_2() {
    assert_eq!(
      interpret("FunctionExpand[Binomial[n, 2]]").unwrap(),
      "((-1 + n)*n)/2"
    );
  }

  #[test]
  fn haversine() {
    assert_eq!(
      interpret("FunctionExpand[Haversine[x]]").unwrap(),
      "(1 - Cos[x])/2"
    );
  }

  #[test]
  fn inverse_haversine() {
    assert_eq!(
      interpret("FunctionExpand[InverseHaversine[x]]").unwrap(),
      "2*ArcSin[Sqrt[x]]"
    );
  }

  #[test]
  fn inverse_haversine_complex_numeric() {
    // 2 * ArcSin[Sqrt[z]] for complex z must be correctly-rounded to match
    // wolframscript bit-for-bit (regression for mathics
    // numbers/trig.py:723).
    assert_eq!(
      interpret("InverseHaversine[1 + 2.5 I]").unwrap(),
      "1.764589463349829 + 2.3309746530493123*I"
    );
  }

  #[test]
  fn sinc() {
    assert_eq!(interpret("FunctionExpand[Sinc[x]]").unwrap(), "Sin[x]/x");
  }

  // LogisticSigmoid[x] -> 1/(1 + E^(-x)).
  #[test]
  fn logistic_sigmoid() {
    assert_eq!(
      interpret("FunctionExpand[LogisticSigmoid[x]]").unwrap(),
      "(1 + E^(-x))^(-1)"
    );
    assert_eq!(
      interpret("FunctionExpand[LogisticSigmoid[2 x]]").unwrap(),
      "(1 + E^(-2*x))^(-1)"
    );
    assert_eq!(
      interpret("FunctionExpand[LogisticSigmoid[a + b]]").unwrap(),
      "(1 + E^(-a - b))^(-1)"
    );
  }

  #[test]
  fn chebyshev_t() {
    assert_eq!(
      interpret("FunctionExpand[ChebyshevT[n, x]]").unwrap(),
      "Cos[n*ArcCos[x]]"
    );
  }

  #[test]
  fn chebyshev_u() {
    assert_eq!(
      interpret("FunctionExpand[ChebyshevU[n, x]]").unwrap(),
      "Sin[(1 + n)*ArcCos[x]]/(Sqrt[1 - x]*Sqrt[1 + x])"
    );
  }

  #[test]
  fn fibonacci() {
    assert_eq!(
      interpret("FunctionExpand[Fibonacci[n]]").unwrap(),
      "(((1 + Sqrt[5])/2)^n - (2/(1 + Sqrt[5]))^n*Cos[n*Pi])/Sqrt[5]"
    );
  }

  #[test]
  fn lucas_l() {
    assert_eq!(
      interpret("FunctionExpand[LucasL[n]]").unwrap(),
      "((1 + Sqrt[5])/2)^n + (2/(1 + Sqrt[5]))^n*Cos[n*Pi]"
    );
  }

  #[test]
  fn gamma_half() {
    assert_eq!(interpret("FunctionExpand[Gamma[1/2]]").unwrap(), "Sqrt[Pi]");
  }

  // Factorial[n] (n!) expands to the Gamma function.
  #[test]
  fn factorial() {
    assert_eq!(
      interpret("FunctionExpand[Factorial[n]]").unwrap(),
      "Gamma[1 + n]"
    );
    assert_eq!(interpret("FunctionExpand[n!]").unwrap(), "Gamma[1 + n]");
    // A concrete factorial still evaluates numerically.
    assert_eq!(interpret("FunctionExpand[Factorial[5]]").unwrap(), "120");
  }

  // Binomial with a symbolic second argument expands to the Gamma form.
  #[test]
  fn binomial_symbolic_k() {
    assert_eq!(
      interpret("FunctionExpand[Binomial[n, k]]").unwrap(),
      "Gamma[1 + n]/(Gamma[1 + k]*Gamma[1 - k + n])"
    );
  }

  #[test]
  fn catalan_number() {
    assert_eq!(
      interpret("FunctionExpand[CatalanNumber[n]]").unwrap(),
      "(2^(2*n)*Gamma[1/2 + n])/(Sqrt[Pi]*Gamma[2 + n])"
    );
    // A concrete value still evaluates.
    assert_eq!(interpret("FunctionExpand[CatalanNumber[3]]").unwrap(), "5");
  }

  #[test]
  fn subfactorial() {
    assert_eq!(
      interpret("FunctionExpand[Subfactorial[n]]").unwrap(),
      "Gamma[1 + n, -1]/E"
    );
  }

  #[test]
  fn passthrough() {
    // Functions without expansion rules pass through
    assert_eq!(interpret("FunctionExpand[Sin[x]]").unwrap(), "Sin[x]");
  }
}

mod to_radicals {
  use super::*;

  #[test]
  fn quadratic() {
    assert_eq!(
      interpret("ToRadicals[Root[#^2 - 3 &, 1]]").unwrap(),
      "-Sqrt[3]"
    );
    assert_eq!(
      interpret("ToRadicals[Root[#^2 - 3 &, 2]]").unwrap(),
      "Sqrt[3]"
    );
  }

  #[test]
  fn quadratic_with_linear() {
    assert_eq!(
      interpret("ToRadicals[Root[#^2 + 3# + 1 &, 1]]").unwrap(),
      "(-3 - Sqrt[5])/2"
    );
    assert_eq!(
      interpret("ToRadicals[Root[#^2 + 3# + 1 &, 2]]").unwrap(),
      "(-3 + Sqrt[5])/2"
    );
  }

  #[test]
  fn cubic_pure() {
    assert_eq!(
      interpret("ToRadicals[Root[#^3 - 2 &, 1]]").unwrap(),
      "2^(1/3)"
    );
  }

  #[test]
  fn quartic_pure() {
    assert_eq!(
      interpret("ToRadicals[Root[#^4 - 2 &, 1]]").unwrap(),
      "-2^(1/4)"
    );
    assert_eq!(
      interpret("ToRadicals[Root[#^4 - 2 &, 2]]").unwrap(),
      "2^(1/4)"
    );
  }

  #[test]
  fn quintic_pure() {
    assert_eq!(
      interpret("ToRadicals[Root[#^5 - 2 &, 1]]").unwrap(),
      "2^(1/5)"
    );
  }

  #[test]
  fn sixth_root() {
    assert_eq!(
      interpret("ToRadicals[Root[#^6 - 2 &, 1]]").unwrap(),
      "-2^(1/6)"
    );
  }
}

mod cases {
  use super::super::case_helpers::assert_case;

  #[test]
  fn maximize() {
    assert_case(r#"Maximize[-2 x^2 - 3 x + 5, x]"#, r#"{49/8, {x -> -3/4}}"#);
  }
  #[test]
  fn minimize() {
    assert_case(r#"Minimize[2 x^2 - 3 x + 5, x]"#, r#"{31/8, {x -> 3/4}}"#);
  }
  #[test]
  fn arg_max_unconstrained() {
    // ArgMax[f, x] returns the bare argument maximizing f (scalar for a
    // single variable). Matches wolframscript `2`.
    assert_case(r#"ArgMax[-(x^2) + 4 x + 1, x]"#, r#"2"#);
  }
  #[test]
  fn arg_min_unconstrained() {
    assert_case(r#"ArgMin[x^2 + 2 x + 5, x]"#, r#"-1"#);
  }
  #[test]
  fn arg_max_box_constraint() {
    assert_case(r#"ArgMax[{x^2, -1 <= x <= 3}, x]"#, r#"3"#);
  }
  #[test]
  fn arg_max_interval_objective() {
    assert_case(r#"ArgMax[{x (10 - x), 0 <= x <= 10}, x]"#, r#"5"#);
  }
  #[test]
  fn arg_max_disk_multivar() {
    // Multiple variables yield a list of the optimizing arguments.
    assert_case(
      r#"ArgMax[{x + y, x^2 + y^2 <= 1}, {x, y}]"#,
      r#"{1/Sqrt[2], 1/Sqrt[2]}"#,
    );
  }
  #[test]
  fn arg_min_equality_multivar() {
    assert_case(
      r#"ArgMin[{x^2 + y^2, x + y == 1}, {x, y}]"#,
      r#"{1/2, 1/2}"#,
    );
  }
  #[test]
  fn min_value_univariate() {
    assert_case(r#"MinValue[2 x^2 - 3 x + 5, x]"#, r#"31/8"#);
  }
  #[test]
  fn min_value_multivariate() {
    assert_case(r#"MinValue[(x y - 3)^2 + 1, {x, y}]"#, r#"1"#);
    assert_case(r#"MinValue[x^2 + y^2 - 2 x, {x, y}]"#, r#"-1"#);
  }
  #[test]
  fn min_value_disk_constraint() {
    assert_case(
      r#"MinValue[{x - 2 y, x^2 + y^2 <= 1}, {x, y}]"#,
      r#"-Sqrt[5]"#,
    );
  }
  #[test]
  fn max_value_univariate() {
    assert_case(r#"MaxValue[-x^2 + 4 x, x]"#, r#"4"#);
  }
  #[test]
  fn max_value_disk_constraint() {
    assert_case(
      r#"MaxValue[{x + 2 y, x^2 + y^2 <= 1}, {x, y}]"#,
      r#"Sqrt[5]"#,
    );
  }
  #[test]
  fn max_value_equality_constraint() {
    assert_case(r#"MaxValue[{x y, x + y == 4}, {x, y}]"#, r#"4"#);
  }
  #[test]
  fn max_value_unbounded_no_message() {
    // Maximize emits Maximize::natt here; MaxValue must return Infinity
    // without any message (matches wolframscript). Check[] detects a
    // stray message and would return the fallback string instead.
    assert_case(r#"MaxValue[x^2, x]"#, r#"Infinity"#);
    assert_case(r#"Check[MaxValue[x^2, x], "msg emitted"]"#, r#"Infinity"#);
  }
  #[test]
  fn apart_1() {
    assert_case(
      r#"Apart[1 / (x^2 + 5x + 6)]"#,
      r#"(2 + x)^(-1) - (3 + x)^(-1)"#,
    );
  }
  #[test]
  fn apart_2() {
    assert_case(
      r#"Apart[1 / (x^2 + 5x + 6)]; Apart[1 / (x^2 - y^2), x]"#,
      r#"1/(2*(x - y)*y) - 1/(2*y*(x + y))"#,
    );
  }
  #[test]
  fn apart_3() {
    assert_case(
      r#"Apart[1 / (x^2 + 5x + 6)]; Apart[1 / (x^2 - y^2), x]; Apart[1 / (x^2 - y^2), y]"#,
      r#"-1/2*1/(x*(-x + y)) + 1/(2*x*(x + y))"#,
    );
  }
  #[test]
  fn apart_4() {
    assert_case(
      r#"Apart[1 / (x^2 + 5x + 6)]; Apart[1 / (x^2 - y^2), x]; Apart[1 / (x^2 - y^2), y]; Apart[{1 / (x^2 + 5x + 6)}]"#,
      r#"{(2 + x)^(-1) - (3 + x)^(-1)}"#,
    );
  }
  #[test]
  fn sin() {
    assert_case(
      r#"Apart[1 / (x^2 + 5x + 6)]; Apart[1 / (x^2 - y^2), x]; Apart[1 / (x^2 - y^2), y]; Apart[{1 / (x^2 + 5x + 6)}]; Sin[1 / (x ^ 2 - y ^ 2)] // Apart"#,
      r#"Sin[(x^2 - y^2)^(-1)]"#,
    );
  }
  #[test]
  fn equal_1() {
    assert_case(
      r#"Apart[1 / (x^2 + 5x + 6)]; Apart[1 / (x^2 - y^2), x]; Apart[1 / (x^2 - y^2), y]; Apart[{1 / (x^2 + 5x + 6)}]; Sin[1 / (x ^ 2 - y ^ 2)] // Apart; a == "A" // Apart // InputForm"#,
      r#"InputForm[a == "A"]"#,
    );
  }
  #[test]
  fn cancel_1() {
    assert_case(r#"Cancel[x / x ^ 2]"#, r#"x^(-1)"#);
  }
  #[test]
  fn cancel_2() {
    assert_case(
      r#"Cancel[x / x ^ 2]; Cancel[x / x ^ 2 + y / y ^ 2]"#,
      r#"x^(-1) + y^(-1)"#,
    );
  }
  #[test]
  fn cancel_3() {
    assert_case(
      r#"Cancel[x / x ^ 2]; Cancel[x / x ^ 2 + y / y ^ 2]; Cancel[f[x] / x + x * f[x] / x ^ 2]"#,
      r#"(2*f[x])/x"#,
    );
  }
  #[test]
  fn equal_2() {
    assert_case(
      r#"Cancel[x / x ^ 2]; Cancel[x / x ^ 2 + y / y ^ 2]; Cancel[f[x] / x + x * f[x] / x ^ 2]; a == "A" // Cancel // InputForm"#,
      r#"InputForm[a == "A"]"#,
    );
  }
  #[test]
  fn coefficient_1() {
    assert_case(r#"Coefficient[(x + y)^4, (x^2) * (y^2)]"#, r#"6"#);
  }
  #[test]
  fn coefficient_2() {
    assert_case(
      r#"Coefficient[(x + y)^4, (x^2) * (y^2)]; Coefficient[a x^2 + b y^3 + c x + d y + 5, x]"#,
      r#"c"#,
    );
  }
  #[test]
  fn coefficient_3() {
    assert_case(
      r#"Coefficient[(x + y)^4, (x^2) * (y^2)]; Coefficient[a x^2 + b y^3 + c x + d y + 5, x]; Coefficient[(x + 3 y)^5, x]"#,
      r#"405*y^4"#,
    );
  }
  #[test]
  fn coefficient_4() {
    assert_case(
      r#"Coefficient[(x + y)^4, (x^2) * (y^2)]; Coefficient[a x^2 + b y^3 + c x + d y + 5, x]; Coefficient[(x + 3 y)^5, x]; Coefficient[(x + 3 y)^5, x * y^4]"#,
      r#"405"#,
    );
  }
  #[test]
  fn coefficient_5() {
    assert_case(
      r#"Coefficient[(x + y)^4, (x^2) * (y^2)]; Coefficient[a x^2 + b y^3 + c x + d y + 5, x]; Coefficient[(x + 3 y)^5, x]; Coefficient[(x + 3 y)^5, x * y^4]; Coefficient[(x + 2)/(y - 3) + (x + 3)/(y - 2), x]"#,
      r#"(-3 + y)^(-1) + (-2 + y)^(-1)"#,
    );
  }
  #[test]
  fn coefficient_6() {
    assert_case(
      r#"Coefficient[(x + y)^4, (x^2) * (y^2)]; Coefficient[a x^2 + b y^3 + c x + d y + 5, x]; Coefficient[(x + 3 y)^5, x]; Coefficient[(x + 3 y)^5, x * y^4]; Coefficient[(x + 2)/(y - 3) + (x + 3)/(y - 2), x]; Coefficient[x*Cos[x + 3] + 6*y, x]"#,
      r#"Cos[3 + x]"#,
    );
  }
  #[test]
  fn coefficient_7() {
    assert_case(
      r#"Coefficient[(x + y)^4, (x^2) * (y^2)]; Coefficient[a x^2 + b y^3 + c x + d y + 5, x]; Coefficient[(x + 3 y)^5, x]; Coefficient[(x + 3 y)^5, x * y^4]; Coefficient[(x + 2)/(y - 3) + (x + 3)/(y - 2), x]; Coefficient[x*Cos[x + 3] + 6*y, x]; Coefficient[(x + 1)^3, x, 2]"#,
      r#"3"#,
    );
  }
  #[test]
  fn coefficient_8() {
    assert_case(
      r#"Coefficient[(x + y)^4, (x^2) * (y^2)]; Coefficient[a x^2 + b y^3 + c x + d y + 5, x]; Coefficient[(x + 3 y)^5, x]; Coefficient[(x + 3 y)^5, x * y^4]; Coefficient[(x + 2)/(y - 3) + (x + 3)/(y - 2), x]; Coefficient[x*Cos[x + 3] + 6*y, x]; Coefficient[(x + 1)^3, x, 2]; Coefficient[a x^2 + b y^3 + c x + d y + 5, y, 3]"#,
      r#"b"#,
    );
  }
  #[test]
  fn coefficient_9() {
    assert_case(
      r#"Coefficient[(x + y)^4, (x^2) * (y^2)]; Coefficient[a x^2 + b y^3 + c x + d y + 5, x]; Coefficient[(x + 3 y)^5, x]; Coefficient[(x + 3 y)^5, x * y^4]; Coefficient[(x + 2)/(y - 3) + (x + 3)/(y - 2), x]; Coefficient[x*Cos[x + 3] + 6*y, x]; Coefficient[(x + 1)^3, x, 2]; Coefficient[a x^2 + b y^3 + c x + d y + 5, y, 3]; Coefficient[(x + 2)^3 + (x + 3)^2, x, 0]"#,
      r#"17"#,
    );
  }
  #[test]
  fn coefficient_10() {
    assert_case(
      r#"Coefficient[(x + y)^4, (x^2) * (y^2)]; Coefficient[a x^2 + b y^3 + c x + d y + 5, x]; Coefficient[(x + 3 y)^5, x]; Coefficient[(x + 3 y)^5, x * y^4]; Coefficient[(x + 2)/(y - 3) + (x + 3)/(y - 2), x]; Coefficient[x*Cos[x + 3] + 6*y, x]; Coefficient[(x + 1)^3, x, 2]; Coefficient[a x^2 + b y^3 + c x + d y + 5, y, 3]; Coefficient[(x + 2)^3 + (x + 3)^2, x, 0]; Coefficient[(x + 2)^3 + (x + 3)^2, y, 0]"#,
      r#"(2 + x) ^ 3 + (3 + x) ^ 2"#,
    );
  }
  #[test]
  fn coefficient_11() {
    assert_case(
      r#"Coefficient[(x + y)^4, (x^2) * (y^2)]; Coefficient[a x^2 + b y^3 + c x + d y + 5, x]; Coefficient[(x + 3 y)^5, x]; Coefficient[(x + 3 y)^5, x * y^4]; Coefficient[(x + 2)/(y - 3) + (x + 3)/(y - 2), x]; Coefficient[x*Cos[x + 3] + 6*y, x]; Coefficient[(x + 1)^3, x, 2]; Coefficient[a x^2 + b y^3 + c x + d y + 5, y, 3]; Coefficient[(x + 2)^3 + (x + 3)^2, x, 0]; Coefficient[(x + 2)^3 + (x + 3)^2, y, 0]; Coefficient[a x^2 + b y^3 + c x + d y + 5, x, 0]"#,
      r#"5 + d*y + b*y^3"#,
    );
  }
  #[test]
  fn coefficient_list_1() {
    assert_case(
      r#"CoefficientList[(x + 3)^5, x]"#,
      r#"{243, 405, 270, 90, 15, 1}"#,
    );
  }
  #[test]
  fn coefficient_list_bigint_coefficient() {
    // Regression: a coefficient beyond i128 made max_power/is_constant_wrt
    // treat the term as variable-dependent, so CoefficientList stayed
    // unevaluated. It must list the BigInteger coefficient.
    assert_case(
      r#"CoefficientList[100000000000000000000000000000000000000000 x^2 + 5 x + 7, x]"#,
      r#"{7, 5, 100000000000000000000000000000000000000000}"#,
    );
    // The leading coefficient of (2 x + 3)^60 is 2^60.
    assert_case(
      r#"Last[CoefficientList[(2 x + 3)^60, x]]"#,
      r#"1152921504606846976"#,
    );
  }
  #[test]
  fn coefficient_list_2() {
    assert_case(
      r#"CoefficientList[(x + 3)^5, x]; CoefficientList[(x + y)^4, x]"#,
      r#"{y^4, 4*y^3, 6*y^2, 4*y, 1}"#,
    );
  }
  #[test]
  fn coefficient_list_3() {
    assert_case(
      r#"CoefficientList[(x + 3)^5, x]; CoefficientList[(x + y)^4, x]; CoefficientList[a x^2 + b y^3 + c x + d y + 5, x]"#,
      r#"{5 + d*y + b*y^3, c, a}"#,
    );
  }
  #[test]
  fn coefficient_list_4() {
    assert_case(
      r#"CoefficientList[(x + 3)^5, x]; CoefficientList[(x + y)^4, x]; CoefficientList[a x^2 + b y^3 + c x + d y + 5, x]; CoefficientList[(x + 2)/(y - 3) + x/(y - 2), x]"#,
      r#"{2/(-3 + y), (-3 + y)^(-1) + (-2 + y)^(-1)}"#,
    );
  }
  #[test]
  fn coefficient_list_5() {
    assert_case(
      r#"CoefficientList[(x + 3)^5, x]; CoefficientList[(x + y)^4, x]; CoefficientList[a x^2 + b y^3 + c x + d y + 5, x]; CoefficientList[(x + 2)/(y - 3) + x/(y - 2), x]; CoefficientList[(x + y)^3, z]"#,
      r#"{(x + y) ^ 3}"#,
    );
  }
  #[test]
  fn coefficient_list_6() {
    assert_case(
      r#"CoefficientList[(x + 3)^5, x]; CoefficientList[(x + y)^4, x]; CoefficientList[a x^2 + b y^3 + c x + d y + 5, x]; CoefficientList[(x + 2)/(y - 3) + x/(y - 2), x]; CoefficientList[(x + y)^3, z]; CoefficientList[a x^2 + b y^3 + c x + d y + 5, {x, y}]"#,
      r#"{{5, d, 0, b}, {c, 0, 0, 0}, {a, 0, 0, 0}}"#,
    );
  }
  #[test]
  fn coefficient_list_7() {
    assert_case(
      r#"CoefficientList[(x + 3)^5, x]; CoefficientList[(x + y)^4, x]; CoefficientList[a x^2 + b y^3 + c x + d y + 5, x]; CoefficientList[(x + 2)/(y - 3) + x/(y - 2), x]; CoefficientList[(x + y)^3, z]; CoefficientList[a x^2 + b y^3 + c x + d y + 5, {x, y}]; CoefficientList[(x - 2 y + 3 z)^3, {x, y, z}]"#,
      r#"{{{0, 0, 0, 27}, {0, 0, -54, 0}, {0, 36, 0, 0}, {-8, 0, 0, 0}}, {{0, 0, 27, 0}, {0, -36, 0, 0}, {12, 0, 0, 0}, {0, 0, 0, 0}}, {{0, 9, 0, 0}, {-6, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}, {{1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}}"#,
    );
  }
  #[test]
  fn collect_1() {
    assert_case(r#"Collect[(x+y)^3, y]"#, r#"x^3 + 3*x^2*y + 3*x*y^2 + y^3"#);
  }
  #[test]
  fn collect_2() {
    assert_case(
      r#"Collect[(x+y)^3, y]; Collect[2 Sin[x z] (x+2 y^2 + Sin[y] x), y]"#,
      r#"4*y^2*Sin[x*z] + 2*(x + x*Sin[y])*Sin[x*z]"#,
    );
  }
  #[test]
  fn collect_3() {
    assert_case(
      r#"Collect[(x+y)^3, y]; Collect[2 Sin[x z] (x+2 y^2 + Sin[y] x), y]; Collect[3 x y+2 Sin[x z] (x+2 y^2 + x) + (x+y)^3, y]"#,
      r#"x^3 + (3*x + 3*x^2)*y + y^3 + 4*x*Sin[x*z] + y^2*(3*x + 4*Sin[x*z])"#,
    );
  }
  #[test]
  fn collect_4() {
    assert_case(
      r#"Collect[(x+y)^3, y]; Collect[2 Sin[x z] (x+2 y^2 + Sin[y] x), y]; Collect[3 x y+2 Sin[x z] (x+2 y^2 + x) + (x+y)^3, y]; Collect[3 x y+2 Sin[x z] (x+2 y^2 + x) + (x+y)^3, {x,y}]"#,
      r#"x^3 + 3*x^2*y + y^3 + 4*y^2*Sin[x*z] + x*(3*y + 3*y^2 + 4*Sin[x*z])"#,
    );
  }
  #[test]
  fn collect_5() {
    assert_case(
      r#"Collect[(x+y)^3, y]; Collect[2 Sin[x z] (x+2 y^2 + Sin[y] x), y]; Collect[3 x y+2 Sin[x z] (x+2 y^2 + x) + (x+y)^3, y]; Collect[3 x y+2 Sin[x z] (x+2 y^2 + x) + (x+y)^3, {x,y}]; Collect[3 x y+2 Sin[x z] (x+2 y^2 + x) + (x+y)^3, {x,y}, h]"#,
      r#"x^3*h[1] + y^3*h[1] + x^2*y*h[3] + y^2*h[4*Sin[x*z]] + x*(y*h[3] + y^2*h[3] + h[4*Sin[x*z]])"#,
    );
  }
  #[test]
  fn collect_6() {
    assert_case(
      r#"Collect[(x+y)^3, y]; Collect[2 Sin[x z] (x+2 y^2 + Sin[y] x), y]; Collect[3 x y+2 Sin[x z] (x+2 y^2 + x) + (x+y)^3, y]; Collect[3 x y+2 Sin[x z] (x+2 y^2 + x) + (x+y)^3, {x,y}]; Collect[3 x y+2 Sin[x z] (x+2 y^2 + x) + (x+y)^3, {x,y}, h]; Collect[(1 + a + x)^3, x]"#,
      r#"1 + 3*a + 3*a^2 + a^3 + (3 + 6*a + 3*a^2)*x + (3 + 3*a)*x^2 + x^3"#,
    );
  }
  #[test]
  fn collect_7() {
    assert_case(
      r#"Collect[(x+y)^3, y]; Collect[2 Sin[x z] (x+2 y^2 + Sin[y] x), y]; Collect[3 x y+2 Sin[x z] (x+2 y^2 + x) + (x+y)^3, y]; Collect[3 x y+2 Sin[x z] (x+2 y^2 + x) + (x+y)^3, {x,y}]; Collect[3 x y+2 Sin[x z] (x+2 y^2 + x) + (x+y)^3, {x,y}, h]; Collect[(1 + a + x)^3, x]; Collect[a x + b y + c x + d y, y]"#,
      r#"a*x + c*x + (b + d)*y"#,
    );
  }
  #[test]
  fn collect_8() {
    assert_case(
      r#"Collect[(x+y)^3, y]; Collect[2 Sin[x z] (x+2 y^2 + Sin[y] x), y]; Collect[3 x y+2 Sin[x z] (x+2 y^2 + x) + (x+y)^3, y]; Collect[3 x y+2 Sin[x z] (x+2 y^2 + x) + (x+y)^3, {x,y}]; Collect[3 x y+2 Sin[x z] (x+2 y^2 + x) + (x+y)^3, {x,y}, h]; Collect[(1 + a + x)^3, x]; Collect[a x + b y + c x + d y, y]; Collect[(1 + a + x)^3, x, Simplify]"#,
      r#"(1 + a)^3 + 3*(1 + a)^2*x + 3*(1 + a)*x^2 + x^3"#,
    );
  }
  #[test]
  fn denominator_1() {
    assert_case(r#"Denominator[2 / 3]"#, r#"3"#);
  }
  #[test]
  fn denominator_2() {
    assert_case(r#"Denominator[2 / 3]; Denominator[a / b]"#, r#"b"#);
  }
  #[test]
  fn denominator_3() {
    assert_case(
      r#"Denominator[2 / 3]; Denominator[a / b]; Denominator[a + b]"#,
      r#"1"#,
    );
  }
  #[test]
  fn denominator_4() {
    assert_case(
      r#"Denominator[2 / 3]; Denominator[a / b]; Denominator[a + b]; Denominator[a x^n y^-m]"#,
      r#"y ^ m"#,
    );
  }
  #[test]
  fn denominator_5() {
    assert_case(
      r#"Denominator[2 / 3]; Denominator[a / b]; Denominator[a + b]; Denominator[a x^n y^-m]; Denominator[Sin[x]^a (Sin[x] - 2)^-b]"#,
      r#"(-2 + Sin[x]) ^ b"#,
    );
  }
  #[test]
  fn denominator_6() {
    assert_case(
      r#"Denominator[2 / 3]; Denominator[a / b]; Denominator[a + b]; Denominator[a x^n y^-m]; Denominator[Sin[x]^a (Sin[x] - 2)^-b]; Denominator[3/7 + I/11]"#,
      r#"77"#,
    );
  }
  #[test]
  fn denominator_7() {
    assert_case(
      r#"Denominator[2 / 3]; Denominator[a / b]; Denominator[a + b]; Denominator[a x^n y^-m]; Denominator[Sin[x]^a (Sin[x] - 2)^-b]; Denominator[3/7 + I/11]; Denominator[{1, 2, 3, 4, 5, 6}/3]"#,
      r#"{3, 3, 1, 3, 3, 1}"#,
    );
  }
  #[test]
  fn denominator_8() {
    assert_case(
      r#"Denominator[2 / 3]; Denominator[a / b]; Denominator[a + b]; Denominator[a x^n y^-m]; Denominator[Sin[x]^a (Sin[x] - 2)^-b]; Denominator[3/7 + I/11]; Denominator[{1, 2, 3, 4, 5, 6}/3]; Denominator[{Sin[x], Cos[x], Tan[x], Csc[x], Sec[x], Cot[x]}, Trig -> True]"#,
      r#"{1, 1, Cos[x], Sin[x], Cos[x], Sin[x]}"#,
    );
  }
  #[test]
  fn denominator_9() {
    assert_case(
      r#"Denominator[2 / 3]; Denominator[a / b]; Denominator[a + b]; Denominator[a x^n y^-m]; Denominator[Sin[x]^a (Sin[x] - 2)^-b]; Denominator[3/7 + I/11]; Denominator[{1, 2, 3, 4, 5, 6}/3]; Denominator[{Sin[x], Cos[x], Tan[x], Csc[x], Sec[x], Cot[x]}, Trig -> True]; Denominator[{Sinh[x], Cosh[x], Tanh[x], Csch[x] , Sech[x], Coth[x]}, Trig -> True]"#,
      r#"{1, 1, Cosh[x], Sinh[x], Cosh[x], Sinh[x]}"#,
    );
  }
  // A bare negative power (Power[base, -n] not inside a Times) is a pure
  // denominator: Denominator[x^-2] = x^2 and Numerator[x^-2] = 1. Previously
  // the standalone-Power case fell through to Denominator -> 1.
  #[test]
  fn denominator_bare_negative_power() {
    assert_case(r#"Denominator[x^-1]"#, r#"x"#);
    assert_case(r#"Denominator[x^-2]"#, r#"x^2"#);
    assert_case(r#"Denominator[(x + 1)^-2]"#, r#"(1 + x)^2"#);
    // A positive power still has denominator 1.
    assert_case(r#"Denominator[x^2]"#, r#"1"#);
  }
  #[test]
  fn numerator_bare_negative_power() {
    assert_case(r#"Numerator[x^-1]"#, r#"1"#);
    assert_case(r#"Numerator[x^-2]"#, r#"1"#);
    assert_case(r#"Numerator[a^-3]"#, r#"1"#);
    // A positive power is its own numerator.
    assert_case(r#"Numerator[x^2]"#, r#"x^2"#);
  }
  // A negative RATIONAL exponent (x^(-1/2) = 1/Sqrt[x]) is also a denominator
  // power. Previously negate_if_negative did not recognise Rational[-n, d].
  #[test]
  fn denominator_negative_rational_power() {
    assert_case(r#"Denominator[x^(-1/2)]"#, r#"Sqrt[x]"#);
    assert_case(r#"Denominator[x^(-3/2)]"#, r#"x^(3/2)"#);
    assert_case(r#"Denominator[1/Sqrt[2]]"#, r#"Sqrt[2]"#);
    // A positive fractional exponent has denominator 1.
    assert_case(r#"Denominator[x^(1/2)]"#, r#"1"#);
  }
  #[test]
  fn numerator_negative_rational_power() {
    assert_case(r#"Numerator[x^(-1/2)]"#, r#"1"#);
    assert_case(r#"Numerator[x^(-3/2)]"#, r#"1"#);
  }
  #[test]
  fn expand_1() {
    assert_case(r#"Expand[(x + y) ^ 3]"#, r#"x^3 + 3*x^2*y + 3*x*y^2 + y^3"#);
  }
  #[test]
  fn expand_2() {
    assert_case(
      r#"Expand[(x + y) ^ 3]; Expand[(a + b) (a + c + d)]"#,
      r#"a^2 + a*b + a*c + b*c + a*d + b*d"#,
    );
  }
  #[test]
  fn expand_3() {
    assert_case(
      r#"Expand[(x + y) ^ 3]; Expand[(a + b) (a + c + d)]; Expand[(a + b) (a + c + d) (e + f) + e a a]"#,
      r#"2*a^2*e + a*b*e + a*c*e + b*c*e + a*d*e + b*d*e + a^2*f + a*b*f + a*c*f + b*c*f + a*d*f + b*d*f"#,
    );
  }
  #[test]
  fn expand_4() {
    assert_case(
      r#"Expand[(x + y) ^ 3]; Expand[(a + b) (a + c + d)]; Expand[(a + b) (a + c + d) (e + f) + e a a]; Expand[(a + b) ^ 2 * (c + d)]"#,
      r#"a^2*c + 2*a*b*c + b^2*c + a^2*d + 2*a*b*d + b^2*d"#,
    );
  }
  #[test]
  fn expand_5() {
    assert_case(
      r#"Expand[(x + y) ^ 3]; Expand[(a + b) (a + c + d)]; Expand[(a + b) (a + c + d) (e + f) + e a a]; Expand[(a + b) ^ 2 * (c + d)]; Expand[(x + y) ^ 2 + x y]"#,
      r#"x^2 + 3*x*y + y^2"#,
    );
  }
  #[test]
  fn expand_6() {
    assert_case(
      r#"Expand[(x + y) ^ 3]; Expand[(a + b) (a + c + d)]; Expand[(a + b) (a + c + d) (e + f) + e a a]; Expand[(a + b) ^ 2 * (c + d)]; Expand[(x + y) ^ 2 + x y]; Expand[((a + b) (c + d)) ^ 2 + b (1 + a)]"#,
      r#"b + a*b + a^2*c^2 + 2*a*b*c^2 + b^2*c^2 + 2*a^2*c*d + 4*a*b*c*d + 2*b^2*c*d + a^2*d^2 + 2*a*b*d^2 + b^2*d^2"#,
    );
  }
  #[test]
  fn expand_7() {
    assert_case(
      r#"Expand[(x + y) ^ 3]; Expand[(a + b) (a + c + d)]; Expand[(a + b) (a + c + d) (e + f) + e a a]; Expand[(a + b) ^ 2 * (c + d)]; Expand[(x + y) ^ 2 + x y]; Expand[((a + b) (c + d)) ^ 2 + b (1 + a)]; Expand[{4 (x + y), 2 (x + y) -> 4 (x + y)}]"#,
      r#"{4*x + 4*y, 2*x + 2*y -> 4*x + 4*y}"#,
    );
  }
  #[test]
  fn expand_8() {
    assert_case(
      r#"Expand[(x + y) ^ 3]; Expand[(a + b) (a + c + d)]; Expand[(a + b) (a + c + d) (e + f) + e a a]; Expand[(a + b) ^ 2 * (c + d)]; Expand[(x + y) ^ 2 + x y]; Expand[((a + b) (c + d)) ^ 2 + b (1 + a)]; Expand[{4 (x + y), 2 (x + y) -> 4 (x + y)}]; Expand[Sin[x + y], Trig -> True]"#,
      r#"Cos[y]*Sin[x] + Cos[x]*Sin[y]"#,
    );
  }
  #[test]
  fn expand_9() {
    assert_case(
      r#"Expand[(x + y) ^ 3]; Expand[(a + b) (a + c + d)]; Expand[(a + b) (a + c + d) (e + f) + e a a]; Expand[(a + b) ^ 2 * (c + d)]; Expand[(x + y) ^ 2 + x y]; Expand[((a + b) (c + d)) ^ 2 + b (1 + a)]; Expand[{4 (x + y), 2 (x + y) -> 4 (x + y)}]; Expand[Sin[x + y], Trig -> True]; Expand[Tanh[x + y], Trig -> True]"#,
      r#"(Cosh[y]*Sinh[x])/(Cosh[x]*Cosh[y] + Sinh[x]*Sinh[y]) + (Cosh[x]*Sinh[y])/(Cosh[x]*Cosh[y] + Sinh[x]*Sinh[y])"#,
    );
  }
  #[test]
  fn expand_10() {
    assert_case(
      r#"Expand[(x + y) ^ 3]; Expand[(a + b) (a + c + d)]; Expand[(a + b) (a + c + d) (e + f) + e a a]; Expand[(a + b) ^ 2 * (c + d)]; Expand[(x + y) ^ 2 + x y]; Expand[((a + b) (c + d)) ^ 2 + b (1 + a)]; Expand[{4 (x + y), 2 (x + y) -> 4 (x + y)}]; Expand[Sin[x + y], Trig -> True]; Expand[Tanh[x + y], Trig -> True]; Expand[Sin[x (1 + y)]]"#,
      r#"Sin[x*(1 + y)]"#,
    );
  }
  #[test]
  fn expand_11() {
    assert_case(
      r#"Expand[(x + y) ^ 3]; Expand[(a + b) (a + c + d)]; Expand[(a + b) (a + c + d) (e + f) + e a a]; Expand[(a + b) ^ 2 * (c + d)]; Expand[(x + y) ^ 2 + x y]; Expand[((a + b) (c + d)) ^ 2 + b (1 + a)]; Expand[{4 (x + y), 2 (x + y) -> 4 (x + y)}]; Expand[Sin[x + y], Trig -> True]; Expand[Tanh[x + y], Trig -> True]; Expand[Sin[x (1 + y)]]; Expand[(x+a)^2+(y+a)^2+(x+y)(x+a), y]"#,
      r#"a^2 + x*(a + x) + (a + x)^2 + 2*a*y + (a + x)*y + y^2"#,
    );
  }
  #[test]
  fn expand_12() {
    assert_case(
      r#"Expand[(x + y) ^ 3]; Expand[(a + b) (a + c + d)]; Expand[(a + b) (a + c + d) (e + f) + e a a]; Expand[(a + b) ^ 2 * (c + d)]; Expand[(x + y) ^ 2 + x y]; Expand[((a + b) (c + d)) ^ 2 + b (1 + a)]; Expand[{4 (x + y), 2 (x + y) -> 4 (x + y)}]; Expand[Sin[x + y], Trig -> True]; Expand[Tanh[x + y], Trig -> True]; Expand[Sin[x (1 + y)]]; Expand[(x+a)^2+(y+a)^2+(x+y)(x+a), y]; Expand[(1 + a)^12, Modulus -> 3]"#,
      r#"1 + a ^ 3 + a ^ 9 + a ^ 12"#,
    );
  }
  #[test]
  fn expand_13() {
    assert_case(
      r#"Expand[(x + y) ^ 3]; Expand[(a + b) (a + c + d)]; Expand[(a + b) (a + c + d) (e + f) + e a a]; Expand[(a + b) ^ 2 * (c + d)]; Expand[(x + y) ^ 2 + x y]; Expand[((a + b) (c + d)) ^ 2 + b (1 + a)]; Expand[{4 (x + y), 2 (x + y) -> 4 (x + y)}]; Expand[Sin[x + y], Trig -> True]; Expand[Tanh[x + y], Trig -> True]; Expand[Sin[x (1 + y)]]; Expand[(x+a)^2+(y+a)^2+(x+y)(x+a), y]; Expand[(1 + a)^12, Modulus -> 3]; Expand[(1 + a)^12, Modulus -> 4]"#,
      r#"1 + 2*a^2 + 3*a^4 + 3*a^8 + 2*a^10 + a^12"#,
    );
  }
  #[test]
  fn exponent_1() {
    assert_case(r#"Exponent[5 x^2 - 3 x + 7, x]"#, r#"2"#);
  }
  #[test]
  fn exponent_2() {
    assert_case(
      r#"Exponent[5 x^2 - 3 x + 7, x]; Exponent[(x^3 + 1)^2 + 1, x]"#,
      r#"6"#,
    );
  }
  #[test]
  fn exponent_3() {
    assert_case(
      r#"Exponent[5 x^2 - 3 x + 7, x]; Exponent[(x^3 + 1)^2 + 1, x]; Exponent[x^(n + 1) + Sqrt[x] + 1, x]"#,
      r#"Max[1 / 2, 1 + n]"#,
    );
  }
  #[test]
  fn exponent_4() {
    assert_case(
      r#"Exponent[5 x^2 - 3 x + 7, x]; Exponent[(x^3 + 1)^2 + 1, x]; Exponent[x^(n + 1) + Sqrt[x] + 1, x]; Exponent[x / y, y]"#,
      r#"-1"#,
    );
  }
  #[test]
  fn exponent_5() {
    assert_case(
      r#"Exponent[5 x^2 - 3 x + 7, x]; Exponent[(x^3 + 1)^2 + 1, x]; Exponent[x^(n + 1) + Sqrt[x] + 1, x]; Exponent[x / y, y]; Exponent[(x^2 + 1)^3 - 1, x, Min]"#,
      r#"2"#,
    );
  }
  #[test]
  fn exponent_6() {
    assert_case(
      r#"Exponent[5 x^2 - 3 x + 7, x]; Exponent[(x^3 + 1)^2 + 1, x]; Exponent[x^(n + 1) + Sqrt[x] + 1, x]; Exponent[x / y, y]; Exponent[(x^2 + 1)^3 - 1, x, Min]; Exponent[0, x]"#,
      r#"-Infinity"#,
    );
  }
  #[test]
  fn exponent_7() {
    assert_case(
      r#"Exponent[5 x^2 - 3 x + 7, x]; Exponent[(x^3 + 1)^2 + 1, x]; Exponent[x^(n + 1) + Sqrt[x] + 1, x]; Exponent[x / y, y]; Exponent[(x^2 + 1)^3 - 1, x, Min]; Exponent[0, x]; Exponent[1, x]"#,
      r#"0"#,
    );
  }
  #[test]
  fn factor_1() {
    assert_case(r#"Factor[x ^ 2 + 2 x + 1]"#, r#"(1 + x) ^ 2"#);
  }
  #[test]
  fn factor_2() {
    assert_case(
      r#"Factor[x ^ 2 + 2 x + 1]; Factor[1 / (x^2+2x+1) + 1 / (x^4+2x^2+1)]"#,
      r#"(2 + 2*x + 3*x^2 + x^4)/((1 + x)^2*(1 + x^2)^2)"#,
    );
  }
  #[test]
  fn factor_3() {
    assert_case(
      r#"Factor[x ^ 2 + 2 x + 1]; Factor[1 / (x^2+2x+1) + 1 / (x^4+2x^2+1)]; Factor[x a == x b + x c]"#,
      r#"a*x == (b + c)*x"#,
    );
  }
  #[test]
  fn factor_4() {
    assert_case(
      r#"Factor[x ^ 2 + 2 x + 1]; Factor[1 / (x^2+2x+1) + 1 / (x^4+2x^2+1)]; Factor[x a == x b + x c]; Factor[{x + x^2, 2 x + 2 y + 2}]"#,
      r#"{x*(1 + x), 2*(1 + x + y)}"#,
    );
  }
  #[test]
  fn factor_5() {
    assert_case(
      r#"Factor[x ^ 2 + 2 x + 1]; Factor[1 / (x^2+2x+1) + 1 / (x^4+2x^2+1)]; Factor[x a == x b + x c]; Factor[{x + x^2, 2 x + 2 y + 2}]; Factor[x ^ 3 + 3 x ^ 2 y + 3 x y ^ 2 + y ^ 3]"#,
      r#"(x + y) ^ 3"#,
    );
  }
  #[test]
  fn equal_3() {
    assert_case(
      r#"Factor[x ^ 2 + 2 x + 1]; Factor[1 / (x^2+2x+1) + 1 / (x^4+2x^2+1)]; Factor[x a == x b + x c]; Factor[{x + x^2, 2 x + 2 y + 2}]; Factor[x ^ 3 + 3 x ^ 2 y + 3 x y ^ 2 + y ^ 3]; x^2 - x == 0 // Factor"#,
      r#"(-1 + x)*x == 0"#,
    );
  }
  #[test]
  fn equal_4() {
    assert_case(
      r#"Factor[x ^ 2 + 2 x + 1]; Factor[1 / (x^2+2x+1) + 1 / (x^4+2x^2+1)]; Factor[x a == x b + x c]; Factor[{x + x^2, 2 x + 2 y + 2}]; Factor[x ^ 3 + 3 x ^ 2 y + 3 x y ^ 2 + y ^ 3]; x^2 - x == 0 // Factor; a == "A" // Factor // InputForm"#,
      r#"InputForm[a == "A"]"#,
    );
  }
  #[test]
  fn simplify_1() {
    assert_case(r#"Simplify[2*Sin[x]^2 + 2*Cos[x]^2]"#, r#"2"#);
  }
  #[test]
  fn simplify_2() {
    assert_case(r#"Simplify[2*Sin[x]^2 + 2*Cos[x]^2]; Simplify[x]"#, r#"x"#);
  }
  #[test]
  fn simplify_3() {
    assert_case(
      r#"Simplify[2*Sin[x]^2 + 2*Cos[x]^2]; Simplify[x]; Simplify[f[x]]"#,
      r#"f[x]"#,
    );
  }
  #[test]
  fn simplify_4() {
    assert_case(
      r#"Simplify[2*Sin[x]^2 + 2*Cos[x]^2]; Simplify[x]; Simplify[f[x]]; $Assumptions={a <= 0}; Simplify[ConditionalExpression[1, a > 0]]"#,
      r#"Undefined"#,
    );
  }
  #[test]
  fn simplify_5() {
    assert_case(
      r#"Simplify[2*Sin[x]^2 + 2*Cos[x]^2]; Simplify[x]; Simplify[f[x]]; $Assumptions={a <= 0}; Simplify[ConditionalExpression[1, a > 0]]; Simplify[ConditionalExpression[1, a > 0] ConditionalExpression[1, b > 0], { b > 0 }]"#,
      r#"Undefined"#,
    );
  }
  #[test]
  fn simplify_6() {
    assert_case(
      r#"Simplify[2*Sin[x]^2 + 2*Cos[x]^2]; Simplify[x]; Simplify[f[x]]; $Assumptions={a <= 0}; Simplify[ConditionalExpression[1, a > 0]]; Simplify[ConditionalExpression[1, a > 0] ConditionalExpression[1, b > 0], { b > 0 }]; Simplify[ConditionalExpression[1, a > 0] ConditionalExpression[1, b > 0], Assumptions -> { b > 0 }]"#,
      r#"ConditionalExpression[1, a > 0]"#,
    );
  }
  #[test]
  fn simplify_7() {
    assert_case(
      r#"Simplify[2*Sin[x]^2 + 2*Cos[x]^2]; Simplify[x]; Simplify[f[x]]; $Assumptions={a <= 0}; Simplify[ConditionalExpression[1, a > 0]]; Simplify[ConditionalExpression[1, a > 0] ConditionalExpression[1, b > 0], { b > 0 }]; Simplify[ConditionalExpression[1, a > 0] ConditionalExpression[1, b > 0], Assumptions -> { b > 0 }]; Simplify[ConditionalExpression[1, a > 0] ConditionalExpression[1, b > 0], {a>0},Assumptions -> { b > 0 }]"#,
      r#"1"#,
    );
  }
  #[test]
  fn simplify_8() {
    assert_case(
      r#"Simplify[2*Sin[x]^2 + 2*Cos[x]^2]; Simplify[x]; Simplify[f[x]]; $Assumptions={a <= 0}; Simplify[ConditionalExpression[1, a > 0]]; Simplify[ConditionalExpression[1, a > 0] ConditionalExpression[1, b > 0], { b > 0 }]; Simplify[ConditionalExpression[1, a > 0] ConditionalExpression[1, b > 0], Assumptions -> { b > 0 }]; Simplify[ConditionalExpression[1, a > 0] ConditionalExpression[1, b > 0], {a>0},Assumptions -> { b > 0 }]; $Assumptions={}; Simplify[20 Log[2]]"#,
      r#"20*Log[2]"#,
    );
  }
  #[test]
  fn simplify_9() {
    assert_case(
      r#"Simplify[2*Sin[x]^2 + 2*Cos[x]^2]; Simplify[x]; Simplify[f[x]]; $Assumptions={a <= 0}; Simplify[ConditionalExpression[1, a > 0]]; Simplify[ConditionalExpression[1, a > 0] ConditionalExpression[1, b > 0], { b > 0 }]; Simplify[ConditionalExpression[1, a > 0] ConditionalExpression[1, b > 0], Assumptions -> { b > 0 }]; Simplify[ConditionalExpression[1, a > 0] ConditionalExpression[1, b > 0], {a>0},Assumptions -> { b > 0 }]; $Assumptions={}; Simplify[20 Log[2]]; Simplify[20 Log[2], ComplexityFunction->LeafCount]"#,
      r#"Log[1048576]"#,
    );
  }
  #[test]
  fn full_simplify() {
    assert_case(r#"FullSimplify[2*Sin[x]^2 + 2*Cos[x]^2]"#, r#"2"#);
  }
  #[test]
  fn minimal_polynomial_1() {
    assert_case(r#"MinimalPolynomial[7, x]"#, r#"-7 + x"#);
  }
  #[test]
  fn minimal_polynomial_2() {
    assert_case(
      r#"MinimalPolynomial[7, x]; MinimalPolynomial[Sqrt[2] + Sqrt[3], x]"#,
      r#"1 - 10*x^2 + x^4"#,
    );
  }
  #[test]
  fn minimal_polynomial_3() {
    assert_case(
      r#"MinimalPolynomial[7, x]; MinimalPolynomial[Sqrt[2] + Sqrt[3], x]; MinimalPolynomial[Sqrt[1 + Sqrt[3]], x]"#,
      r#"-2 - 2*x^2 + x^4"#,
    );
  }
  #[test]
  fn minimal_polynomial_4() {
    assert_case(
      r#"MinimalPolynomial[7, x]; MinimalPolynomial[Sqrt[2] + Sqrt[3], x]; MinimalPolynomial[Sqrt[1 + Sqrt[3]], x]; MinimalPolynomial[Sqrt[I + Sqrt[6]], x]"#,
      r#"49 - 10*x^4 + x^8"#,
    );
  }
  #[test]
  fn numerator_1() {
    assert_case(r#"Numerator[2 / 3]"#, r#"2"#);
  }
  #[test]
  fn numerator_2() {
    assert_case(r#"Numerator[2 / 3]; Numerator[a / b]"#, r#"a"#);
  }
  #[test]
  fn numerator_3() {
    assert_case(
      r#"Numerator[2 / 3]; Numerator[a / b]; Numerator[(x - 1) (x - 2)/(x - 3)^2]"#,
      r#"(-2 + x)*(-1 + x)"#,
    );
  }
  #[test]
  fn numerator_4() {
    assert_case(
      r#"Numerator[2 / 3]; Numerator[a / b]; Numerator[(x - 1) (x - 2)/(x - 3)^2]; Numerator[3/7 + I/11]"#,
      r#"33 + 7*I"#,
    );
  }
  #[test]
  fn numerator_5() {
    assert_case(
      r#"Numerator[2 / 3]; Numerator[a / b]; Numerator[(x - 1) (x - 2)/(x - 3)^2]; Numerator[3/7 + I/11]; Numerator[{Sin[x], Cos[x], Tan[x], Csc[x], Sec[x], Cot[x]}, Trig -> True]"#,
      r#"{Sin[x], Cos[x], Sin[x], 1, 1, Cos[x]}"#,
    );
  }
  #[test]
  fn numerator_6() {
    assert_case(
      r#"Numerator[2 / 3]; Numerator[a / b]; Numerator[(x - 1) (x - 2)/(x - 3)^2]; Numerator[3/7 + I/11]; Numerator[{Sin[x], Cos[x], Tan[x], Csc[x], Sec[x], Cot[x]}, Trig -> True]; Numerator[{Sinh[x], Cosh[x], Tanh[x], Csch[x], Sech[x], Coth[x]}, Trig -> True]"#,
      r#"{Sinh[x], Cosh[x], Sinh[x], 1, 1, Cosh[x]}"#,
    );
  }
  #[test]
  fn set_1() {
    assert_case(
      r#"Numerator[2 / 3]; Numerator[a / b]; Numerator[(x - 1) (x - 2)/(x - 3)^2]; Numerator[3/7 + I/11]; Numerator[{Sin[x], Cos[x], Tan[x], Csc[x], Sec[x], Cot[x]}, Trig -> True]; Numerator[{Sinh[x], Cosh[x], Tanh[x], Csch[x], Sech[x], Coth[x]}, Trig -> True]; expr = 5/7 (x - 1)^2/(x - 2)^3 a^b c^-d"#,
      r#"(5*a^b*(-1 + x)^2)/(7*c^d*(-2 + x)^3)"#,
    );
  }
  #[test]
  fn set_2() {
    assert_case(
      r#"Numerator[2 / 3]; Numerator[a / b]; Numerator[(x - 1) (x - 2)/(x - 3)^2]; Numerator[3/7 + I/11]; Numerator[{Sin[x], Cos[x], Tan[x], Csc[x], Sec[x], Cot[x]}, Trig -> True]; Numerator[{Sinh[x], Cosh[x], Tanh[x], Csch[x], Sech[x], Coth[x]}, Trig -> True]; expr = 5/7 (x - 1)^2/(x - 2)^3 a^b c^-d; num = Numerator[expr]"#,
      r#"5*a^b*(-1 + x)^2"#,
    );
  }
  #[test]
  fn set_3() {
    assert_case(
      r#"Numerator[2 / 3]; Numerator[a / b]; Numerator[(x - 1) (x - 2)/(x - 3)^2]; Numerator[3/7 + I/11]; Numerator[{Sin[x], Cos[x], Tan[x], Csc[x], Sec[x], Cot[x]}, Trig -> True]; Numerator[{Sinh[x], Cosh[x], Tanh[x], Csch[x], Sech[x], Coth[x]}, Trig -> True]; expr = 5/7 (x - 1)^2/(x - 2)^3 a^b c^-d; num = Numerator[expr]; den = Denominator[expr]"#,
      r#"7*c^d*(-2 + x)^3"#,
    );
  }
  #[test]
  fn equal_5() {
    assert_case(
      r#"Numerator[2 / 3]; Numerator[a / b]; Numerator[(x - 1) (x - 2)/(x - 3)^2]; Numerator[3/7 + I/11]; Numerator[{Sin[x], Cos[x], Tan[x], Csc[x], Sec[x], Cot[x]}, Trig -> True]; Numerator[{Sinh[x], Cosh[x], Tanh[x], Csch[x], Sech[x], Coth[x]}, Trig -> True]; expr = 5/7 (x - 1)^2/(x - 2)^3 a^b c^-d; num = Numerator[expr]; den = Denominator[expr]; expr === num / den"#,
      r#"True"#,
    );
  }
  #[test]
  fn polynomial_q_1() {
    assert_case(r#"PolynomialQ[x^2]"#, r#"True"#);
  }
  #[test]
  fn polynomial_q_2() {
    assert_case(r#"PolynomialQ[x^2]; PolynomialQ[2]"#, r#"True"#);
  }
  #[test]
  fn polynomial_q_3() {
    assert_case(
      r#"PolynomialQ[x^2]; PolynomialQ[2]; PolynomialQ[x^2 + x/y]"#,
      r#"False"#,
    );
  }
  #[test]
  fn polynomial_q_4() {
    assert_case(
      r#"PolynomialQ[x^2]; PolynomialQ[2]; PolynomialQ[x^2 + x/y]; PolynomialQ[x^3 - 2 x/y + 3xz, x]"#,
      r#"True"#,
    );
  }
  #[test]
  fn polynomial_q_5() {
    assert_case(
      r#"PolynomialQ[x^2]; PolynomialQ[2]; PolynomialQ[x^2 + x/y]; PolynomialQ[x^3 - 2 x/y + 3xz, x]; PolynomialQ[x^3 - 2 x/y^2 + 3xz, y]"#,
      r#"False"#,
    );
  }
  #[test]
  fn polynomial_q_6() {
    assert_case(
      r#"PolynomialQ[x^2]; PolynomialQ[2]; PolynomialQ[x^2 + x/y]; PolynomialQ[x^3 - 2 x/y + 3xz, x]; PolynomialQ[x^3 - 2 x/y^2 + 3xz, y]; PolynomialQ[f[a] + f[a]^2, f[a]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn polynomial_q_7() {
    assert_case(
      r#"PolynomialQ[x^2]; PolynomialQ[2]; PolynomialQ[x^2 + x/y]; PolynomialQ[x^3 - 2 x/y + 3xz, x]; PolynomialQ[x^3 - 2 x/y^2 + 3xz, y]; PolynomialQ[f[a] + f[a]^2, f[a]]; PolynomialQ[x^2 + axy^2 - bSin[c], {x, y}]"#,
      r#"True"#,
    );
  }
  #[test]
  fn polynomial_q_8() {
    assert_case(
      r#"PolynomialQ[x^2]; PolynomialQ[2]; PolynomialQ[x^2 + x/y]; PolynomialQ[x^3 - 2 x/y + 3xz, x]; PolynomialQ[x^3 - 2 x/y^2 + 3xz, y]; PolynomialQ[f[a] + f[a]^2, f[a]]; PolynomialQ[x^2 + axy^2 - bSin[c], {x, y}]; PolynomialQ[x^2 + axy^2 - bSin[c], {a, b, c}]"#,
      r#"False"#,
    );
  }
  #[test]
  fn together_1() {
    assert_case(r#"Together[a / c + b / c]"#, r#"(a + b) / c"#);
  }
  #[test]
  fn together_2() {
    assert_case(
      r#"Together[a / c + b / c]; Together[{x / (y+1) + x / (y+1)^2}]"#,
      r#"{(x*(2 + y))/(1 + y)^2}"#,
    );
  }
  #[test]
  fn together_3() {
    assert_case(
      r#"Together[a / c + b / c]; Together[{x / (y+1) + x / (y+1)^2}]; Together[f[a / c + b / c]]"#,
      r#"f[a / c + b / c]"#,
    );
  }
  #[test]
  fn find_maximum_1() {
    assert_case(r#"FindMaximum[-(x-3)^2+2., {x, 1}]"#, r#"{2., {x -> 3.}}"#);
  }
  #[test]
  fn find_maximum_2() {
    assert_case(
      r#"FindMaximum[-(x-3)^2+2., {x, 1}]; FindMaximum[-10*^-30 *(x-3)^2+2., {x, 1}]"#,
      r#"{2., {x -> 3.}}"#,
    );
  }
  #[test]
  fn find_maximum_3() {
    assert_case(
      r#"FindMaximum[-(x-3)^2+2., {x, 1}]; FindMaximum[-10*^-30 *(x-3)^2+2., {x, 1}]; FindMaximum[Sin[x], {x, 1}]"#,
      r#"{1., {x -> 1.5707963267948957}}"#,
    );
  }
  #[test]
  fn find_maximum_accepts_options() {
    // Trailing options (Method, MaxIterations, ...) must not abort the
    // call. Wolfram accepts the 3-arg form; Woxi previously rejected it
    // with FindMaximum::argrx.
    assert_case(
      r#"FindMaximum[-(x-3)^2+2., {x, 1}, MaxIterations->2]"#,
      r#"{2., {x -> 3.}}"#,
    );
    assert_case(
      r#"FindMaximum[Sin[x], {x, 1}, Method->"Newton"]"#,
      r#"{1., {x -> 1.5707963267948957}}"#,
    );
  }
  #[test]
  fn find_minimum_1() {
    assert_case(r#"FindMinimum[(x-3)^2+2., {x, 1}]"#, r#"{2., {x -> 3.}}"#);
  }
  #[test]
  fn find_minimum_2() {
    assert_case(
      r#"FindMinimum[(x-3)^2+2., {x, 1}]; FindMinimum[10*^-30 *(x-3)^2+2., {x, 1}]"#,
      r#"{2., {x -> 3.}}"#,
    );
  }
  #[test]
  fn find_minimum_3() {
    assert_case(
      r#"FindMinimum[(x-3)^2+2., {x, 1}]; FindMinimum[10*^-30 *(x-3)^2+2., {x, 1}]; FindMinimum[Sin[x], {x, 1}]"#,
      r#"{-1., {x -> -1.5707963267955243}}"#,
    );
  }
  #[test]
  fn find_root_1() {
    assert_case(
      r#"FindRoot[Cos[x], {x, 1}]"#,
      r#"{x -> 1.5707963267948966}"#,
    );
  }
  #[test]
  fn find_root_2() {
    assert_case(
      r#"FindRoot[Cos[x], {x, 1}]; FindRoot[Sin[x] + Exp[x],{x, 0}]"#,
      r#"{x -> -0.5885327439818611}"#,
    );
  }
  #[test]
  fn find_root_3() {
    assert_case(
      r#"FindRoot[Cos[x], {x, 1}]; FindRoot[Sin[x] + Exp[x],{x, 0}]; FindRoot[Sin[x] + Exp[x] == Pi,{x, 0}]"#,
      r#"{x -> 0.8668152399114581}"#,
    );
  }
  #[test]
  fn find_root_4() {
    assert_case(
      r#"FindRoot[Cos[x], {x, 1}]; FindRoot[Sin[x] + Exp[x],{x, 0}]; FindRoot[Sin[x] + Exp[x] == Pi,{x, 0}]; x = "I am the result!"; FindRoot[Tan[x] + Sin[x] == Pi, {x, 1}]"#,
      r#"{"I am the result!" -> 1.1491129543142686}"#,
    );
  }
  #[test]
  fn solve_1() {
    assert_case(
      r#"Solve[-4 - 4 x + x^4 + x^5 == 0, x, Integers]"#,
      r#"{{x -> -1}}"#,
    );
  }
  #[test]
  fn solve_2() {
    assert_case(
      r#"Solve[-4 - 4 x + x^4 + x^5 == 0, x, Integers]; Solve[x^4 == 4, x, Integers]"#,
      r#"{}"#,
    );
  }
  #[test]
  fn solve_3() {
    assert_case(r#"Solve[x^3 == 1, x, Reals]"#, r#"{{x -> 1}}"#);
  }
  #[test]
  fn root_1() {
    assert_case(r#"Root[#1 ^ 2 - 1&, 1]"#, r#"-1"#);
  }
  #[test]
  fn root_2() {
    assert_case(r#"Root[#1 ^ 2 - 1&, 1]; Root[#1 ^ 2 - 1&, 2]"#, r#"1"#);
  }
  #[test]
  fn root_3() {
    assert_case(
      r#"Root[#1 ^ 2 - 1&, 1]; Root[#1 ^ 2 - 1&, 2]; Root[#1 ^ 5 + 2 #1 + 1&, 2]"#,
      r#"Root[1 + 2*#1 + #1^5 & , 2, 0]"#,
    );
  }

  #[test]
  fn root_three_argument_form_accepted() {
    // wolframscript prints Root with an explicit 0 (exact) tag. Calling
    // Root[f, k, 0] directly should not emit `Root::argrx` and should
    // come back unchanged (it is already in canonical form).
    assert_case(
      r#"Root[1 + 2*#1 + #1^5 & , 1, 0]"#,
      r#"Root[1 + 2*#1 + #1^5 & , 1, 0]"#,
    );
    assert_case(
      r#"Root[1 + 2*#1 + #1^5 & , 3, 0]"#,
      r#"Root[1 + 2*#1 + #1^5 & , 3, 0]"#,
    );
  }

  #[test]
  fn solve_unsolvable_quintic_returns_root_list() {
    // The audit's Root diff case. wolframscript returns five Root
    // expressions for an irreducible quintic with no radical solution.
    assert_case(
      r#"Solve[x^5 + 2*x + 1 == 0, x]"#,
      r#"{{x -> Root[1 + 2*#1 + #1^5 & , 1, 0]}, {x -> Root[1 + 2*#1 + #1^5 & , 2, 0]}, {x -> Root[1 + 2*#1 + #1^5 & , 3, 0]}, {x -> Root[1 + 2*#1 + #1^5 & , 4, 0]}, {x -> Root[1 + 2*#1 + #1^5 & , 5, 0]}}"#,
    );
  }
  #[test]
  fn solve_4() {
    assert_case(r#"Solve[x ^ 2 - 3 x == 4, x]"#, r#"{{x -> -1}, {x -> 4}}"#);
  }
  #[test]
  fn solve_5() {
    assert_case(
      r#"Solve[x ^ 2 - 3 x == 4, x]; Solve[4 y - 8 == 0, y]"#,
      r#"{{y -> 2}}"#,
    );
  }
  #[test]
  fn equal_6() {
    assert_case(
      r#"Solve[x ^ 2 - 3 x == 4, x]; Solve[4 y - 8 == 0, y]; sol = Solve[2 x^2 - 10 x - 12 == 0, x]"#,
      r#"{{x -> -1}, {x -> 6}}"#,
    );
  }
  #[test]
  fn divide() {
    assert_case(
      r#"Solve[x ^ 2 - 3 x == 4, x]; Solve[4 y - 8 == 0, y]; sol = Solve[2 x^2 - 10 x - 12 == 0, x]; x /. sol"#,
      r#"{-1, 6}"#,
    );
  }
  #[test]
  fn solve_6() {
    assert_case(
      r#"Solve[x ^ 2 - 3 x == 4, x]; Solve[4 y - 8 == 0, y]; sol = Solve[2 x^2 - 10 x - 12 == 0, x]; x /. sol; Solve[x + 1 == x, x]"#,
      r#"{}"#,
    );
  }
  #[test]
  fn solve_7() {
    assert_case(
      r#"Solve[x ^ 2 - 3 x == 4, x]; Solve[4 y - 8 == 0, y]; sol = Solve[2 x^2 - 10 x - 12 == 0, x]; x /. sol; Solve[x + 1 == x, x]; Solve[x ^ 2 == x ^ 2, x]"#,
      r#"{{}}"#,
    );
  }
  #[test]
  fn solve_8() {
    assert_case(
      r#"Solve[x ^ 2 - 3 x == 4, x]; Solve[4 y - 8 == 0, y]; sol = Solve[2 x^2 - 10 x - 12 == 0, x]; x /. sol; Solve[x + 1 == x, x]; Solve[x ^ 2 == x ^ 2, x]; Solve[x / (x ^ 2 + 1) == 1, x]"#,
      r#"{{x -> (-1)^(1/3)}, {x -> -(-1)^(2/3)}}"#,
    );
  }
  #[test]
  fn solve_9() {
    assert_case(
      r#"Solve[x ^ 2 - 3 x == 4, x]; Solve[4 y - 8 == 0, y]; sol = Solve[2 x^2 - 10 x - 12 == 0, x]; x /. sol; Solve[x + 1 == x, x]; Solve[x ^ 2 == x ^ 2, x]; Solve[x / (x ^ 2 + 1) == 1, x]; Solve[(x^2 + 3 x + 2)/(4 x - 2) == 0, x]"#,
      r#"{{x -> -2}, {x -> -1}}"#,
    );
  }
  #[test]
  fn solve_10() {
    assert_case(
      r#"Solve[x ^ 2 - 3 x == 4, x]; Solve[4 y - 8 == 0, y]; sol = Solve[2 x^2 - 10 x - 12 == 0, x]; x /. sol; Solve[x + 1 == x, x]; Solve[x ^ 2 == x ^ 2, x]; Solve[x / (x ^ 2 + 1) == 1, x]; Solve[(x^2 + 3 x + 2)/(4 x - 2) == 0, x]; Solve[Cos[x] == 0, x]"#,
      r#"{{x -> ConditionalExpression[-1/2*Pi + 2*Pi*C[1], Element[C[1], Integers]]}, {x -> ConditionalExpression[Pi/2 + 2*Pi*C[1], Element[C[1], Integers]]}}"#,
    );
  }
  #[test]
  fn solve_11() {
    assert_case(
      r#"Solve[x ^ 2 - 3 x == 4, x]; Solve[4 y - 8 == 0, y]; sol = Solve[2 x^2 - 10 x - 12 == 0, x]; x /. sol; Solve[x + 1 == x, x]; Solve[x ^ 2 == x ^ 2, x]; Solve[x / (x ^ 2 + 1) == 1, x]; Solve[(x^2 + 3 x + 2)/(4 x - 2) == 0, x]; Solve[Cos[x] == 0, x]; Solve[f[x + y] == 3, f[x + y]]"#,
      r#"{{f[x + y] -> 3}}"#,
    );
  }
  #[test]
  fn simplify_10() {
    assert_case(
      r#"LeastSquares[{{1, 2}, {2, 3}, {5, 6}}, {1, 5, 3}]; Simplify[LeastSquares[{{1, 2}, {2, 3}, {5, 6}}, {1, x, 3}]]"#,
      r#"{(-4*(-3 + 2*x))/13, (-4 + 7*x)/13}"#,
    );
  }
  #[test]
  fn least_squares() {
    assert_case(
      r#"LeastSquares[{{1, 2}, {2, 3}, {5, 6}}, {1, 5, 3}]; Simplify[LeastSquares[{{1, 2}, {2, 3}, {5, 6}}, {1, x, 3}]]; LeastSquares[{{1, 1, 1}, {1, 1, 2}}, {1, 3}]"#,
      r#"{-1/2, -1/2, 2}"#,
    );
  }
  #[test]
  fn simplify_11() {
    assert_case(
      r#"Simplify[Gamma[z] - (z - 1)!]"#,
      r#"-(-1 + z)! + Gamma[z]"#,
    );
  }
  #[test]
  fn gamma_1() {
    assert_case(r#"Simplify[Gamma[z] - (z - 1)!]; Gamma[8]"#, r#"5040"#);
  }
  #[test]
  fn gamma_2() {
    assert_case(
      r#"Simplify[Gamma[z] - (z - 1)!]; Gamma[8]; Gamma[1/2]"#,
      r#"Sqrt[Pi]"#,
    );
  }
  #[test]
  fn gamma_3() {
    assert_case(
      r#"Simplify[Gamma[z] - (z - 1)!]; Gamma[8]; Gamma[1/2]; Gamma[123.78]"#,
      r#"4.210777742909557*^204"#,
    );
  }
  #[test]
  fn gamma_4() {
    assert_case(
      r#"Simplify[Gamma[z] - (z - 1)!]; Gamma[8]; Gamma[1/2]; Gamma[123.78]; Gamma[1. + I]"#,
      r#"0.49801566811835557 - 0.15494982830181037*I"#,
    );
  }
  #[test]
  fn gamma_5() {
    assert_case(
      r#"Simplify[Gamma[z] - (z - 1)!]; Gamma[8]; Gamma[1/2]; Gamma[123.78]; Gamma[1. + I]; Gamma[1, x]"#,
      r#"E ^ (-x)"#,
    );
  }
  #[test]
  fn gamma_6() {
    assert_case(
      r#"Simplify[Gamma[z] - (z - 1)!]; Gamma[8]; Gamma[1/2]; Gamma[123.78]; Gamma[1. + I]; Gamma[1, x]; Gamma[0, x]"#,
      r#"Gamma[0, x]"#,
    );
  }
  #[test]
  fn boolean_q() {
    assert_case(
      r#"BooleanQ["string"]; BooleanQ[Together[x/y + y/x]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn max() {
    assert_case(
      r#"BooleanQ["string"]; BooleanQ[Together[x/y + y/x]]; Max[x]"#,
      r#"x"#,
    );
  }
  #[test]
  fn min() {
    assert_case(
      r#"BooleanQ["string"]; BooleanQ[Together[x/y + y/x]]; Max[x]; Min[x]"#,
      r#"x"#,
    );
  }
  #[test]
  fn unequal_1() {
    assert_case(
      r#"BooleanQ["string"]; BooleanQ[Together[x/y + y/x]]; Max[x]; Min[x]; Pi != N[Pi]"#,
      r#"False"#,
    );
  }
  #[test]
  fn unequal_2() {
    assert_case(
      r#"BooleanQ["string"]; BooleanQ[Together[x/y + y/x]]; Max[x]; Min[x]; Pi != N[Pi]; a_ != b_"#,
      r#"(a_) != (b_)"#,
    );
  }
  #[test]
  fn unequal_3() {
    assert_case(
      r#"BooleanQ["string"]; BooleanQ[Together[x/y + y/x]]; Max[x]; Min[x]; Pi != N[Pi]; a_ != b_; Clear[a, b];a != a != a"#,
      r#"False"#,
    );
  }
  #[test]
  fn string_literal() {
    assert_case(
      r#"BooleanQ["string"]; BooleanQ[Together[x/y + y/x]]; Max[x]; Min[x]; Pi != N[Pi]; a_ != b_; Clear[a, b];a != a != a; "abc" != "def" != "abc""#,
      r#"False"#,
    );
  }
  #[test]
  fn unequal_4() {
    assert_case(
      r#"BooleanQ["string"]; BooleanQ[Together[x/y + y/x]]; Max[x]; Min[x]; Pi != N[Pi]; a_ != b_; Clear[a, b];a != a != a; "abc" != "def" != "abc"; a != b != a"#,
      r#"a != b != a"#,
    );
  }
  #[test]
  fn solve_12() {
    assert_case(r#"Solve[x^2 +1 == 0, x]"#, r#"{{x -> -I}, {x -> I}}"#);
  }
  #[test]
  fn solve_13() {
    assert_case(
      r#"Solve[x^2 +1 == 0, x]; Solve[x^5==x,x]"#,
      r#"{{x -> -1}, {x -> 0}, {x -> -I}, {x -> I}, {x -> 1}}"#,
    );
  }
  #[test]
  fn apart_5() {
    assert_case(
      r#"Attributes[f] = {HoldAll}; Apart[f[x + x]]"#,
      r#"f[x + x]"#,
    );
  }
  #[test]
  fn apart_6() {
    assert_case(
      r#"Attributes[f] = {HoldAll}; Apart[f[x + x]]; Attributes[f] = {}; Apart[f[x + x]]"#,
      r#"f[2*x]"#,
    );
  }
}

mod irreducible_polynomial_q {
  use super::*;

  #[test]
  fn irreducible_univariate() {
    // Irreducible over the rationals.
    assert_eq!(
      interpret("IrreduciblePolynomialQ[x^2 + 1]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("IrreduciblePolynomialQ[x^4 + 1]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("IrreduciblePolynomialQ[x^2 + x + 1]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("IrreduciblePolynomialQ[x^6 + x^3 + 1]").unwrap(),
      "True"
    );
    // Linear polynomials are irreducible.
    assert_eq!(interpret("IrreduciblePolynomialQ[x]").unwrap(), "True");
    assert_eq!(interpret("IrreduciblePolynomialQ[x + 1]").unwrap(), "True");
  }

  #[test]
  fn reducible_univariate() {
    assert_eq!(
      interpret("IrreduciblePolynomialQ[x^2 - 1]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("IrreduciblePolynomialQ[x^3 - x]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("IrreduciblePolynomialQ[x^4 - 1]").unwrap(),
      "False"
    );
    // Repeated factor (x+1)^2 — not irreducible.
    assert_eq!(
      interpret("IrreduciblePolynomialQ[x^2 + 2 x + 1]").unwrap(),
      "False"
    );
  }

  #[test]
  fn constant_content_is_ignored() {
    // 2 (x^2 + 1) — the numeric content does not count as a factor.
    assert_eq!(
      interpret("IrreduciblePolynomialQ[2 x^2 + 2]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("IrreduciblePolynomialQ[3 x + 6]").unwrap(),
      "True"
    );
    // Rational coefficients are allowed.
    assert_eq!(
      interpret("IrreduciblePolynomialQ[x^2/4 + 1]").unwrap(),
      "True"
    );
  }

  #[test]
  fn multivariate() {
    assert_eq!(
      interpret("IrreduciblePolynomialQ[x^2 + y^2]").unwrap(),
      "True"
    );
    assert_eq!(interpret("IrreduciblePolynomialQ[x*y]").unwrap(), "False");
  }

  #[test]
  fn constants_are_not_irreducible() {
    assert_eq!(interpret("IrreduciblePolynomialQ[5]").unwrap(), "False");
    assert_eq!(interpret("IrreduciblePolynomialQ[6]").unwrap(), "False");
    assert_eq!(interpret("IrreduciblePolynomialQ[0]").unwrap(), "False");
    assert_eq!(interpret("IrreduciblePolynomialQ[1]").unwrap(), "False");
    assert_eq!(interpret("IrreduciblePolynomialQ[Pi]").unwrap(), "False");
    assert_eq!(interpret("IrreduciblePolynomialQ[E]").unwrap(), "False");
  }
}

mod inequality_display {
  use super::*;

  #[test]
  fn mixed_strictness_keeps_head() {
    // Regression: mixed-strictness Inequality used to print as the
    // chained 1 < x <= 10; wolframscript keeps the head in script mode
    assert_eq!(
      interpret("1 < x <= 10").unwrap(),
      "Inequality[1, Less, x, LessEqual, 10]"
    );
    assert_eq!(
      interpret("Inequality[1, Less, x, Less, 5]").unwrap(),
      "Inequality[1, Less, x, Less, 5]"
    );
  }

  #[test]
  fn numeric_inequalities_still_evaluate() {
    assert_eq!(interpret("1 < 3 <= 10").unwrap(), "True");
    assert_eq!(
      interpret("Inequality[1, Less, x, LessEqual, 10] /. x -> 0").unwrap(),
      "False"
    );
  }

  #[test]
  fn function_range_inequality_form() {
    assert_eq!(
      interpret("FunctionRange[1/(1 + x^2), x, y]").unwrap(),
      "Inequality[0, Less, y, LessEqual, 1]"
    );
  }
}

mod sinusoid_extremum_values {
  use super::super::case_helpers::assert_case;

  #[test]
  fn plain_sin_cos() {
    assert_case(r#"MinValue[Sin[x], x]"#, r#"-1"#);
    assert_case(r#"MaxValue[Sin[x], x]"#, r#"1"#);
  }

  #[test]
  fn scaled_and_shifted() {
    assert_case(r#"MinValue[2 Sin[x] + 1, x]"#, r#"-1"#);
    assert_case(r#"MaxValue[3 Cos[x] - 2, x]"#, r#"1"#);
    // Inner argument doesn't affect the range
    assert_case(r#"MinValue[2 Sin[3 x + 1] + 5, x]"#, r#"3"#);
    assert_case(r#"MaxValue[2 Sin[3 x + 1] + 5, x]"#, r#"7"#);
  }

  #[test]
  fn negative_and_rational_amplitudes() {
    assert_case(r#"MinValue[-4 Cos[2 x], x]"#, r#"-4"#);
    assert_case(r#"MinValue[Sin[x]/3, x]"#, r#"-1/3"#);
  }
}

mod groebner_basis {
  use super::*;

  #[test]
  fn linear_system() {
    assert_eq!(
      interpret("GroebnerBasis[{x + y, x - y}, {x, y}]").unwrap(),
      "{y, x}"
    );
  }

  #[test]
  fn circle_and_line() {
    assert_eq!(
      interpret("GroebnerBasis[{x^2 + y^2 - 1, x - y}, {x, y}]").unwrap(),
      "{-1 + 2*y^2, x - y}"
    );
  }

  #[test]
  fn hyperbola_and_circle() {
    assert_eq!(
      interpret("GroebnerBasis[{x y - 1, x^2 + y^2 - 4}, {x, y}]").unwrap(),
      "{1 - 4*y^2 + y^4, x - 4*y + y^3}"
    );
  }

  #[test]
  fn cyclic_three() {
    assert_eq!(
      interpret(
        "GroebnerBasis[{x + y + z, x y + y z + z x, x y z - 1}, {x, y, z}]"
      )
      .unwrap(),
      "{-1 + z^3, y^2 + y*z + z^2, x + y + z}"
    );
  }

  #[test]
  fn normalization() {
    // Content is divided out
    assert_eq!(
      interpret("GroebnerBasis[{2 x + 2 y}, {x, y}]").unwrap(),
      "{x + y}"
    );
    assert_eq!(
      interpret("GroebnerBasis[{x^2 - 1}, {x}]").unwrap(),
      "{-1 + x^2}"
    );
  }

  #[test]
  fn inconsistent_system_is_unit_ideal() {
    assert_eq!(interpret("GroebnerBasis[{x, x + 1}, {x}]").unwrap(), "{1}");
  }

  #[test]
  fn unsupported_stays_unevaluated() {
    assert_eq!(
      interpret("GroebnerBasis[{Sin[x]}, {x}]").unwrap(),
      "GroebnerBasis[{Sin[x]}, {x}]"
    );
  }
}

mod resolve {
  use super::*;

  #[test]
  fn exists_decisions() {
    assert_eq!(
      interpret("Resolve[Exists[x, x^2 == 4], Reals]").unwrap(),
      "True"
    );
    // Solutions are complex, so nothing exists over the reals
    assert_eq!(
      interpret("Resolve[Exists[x, x^2 == -1], Reals]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("Resolve[Exists[x, x > 0 && x < 1]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn forall_decisions() {
    assert_eq!(
      interpret("Resolve[ForAll[x, x^2 >= 0], Reals]").unwrap(),
      "True"
    );
    // x = 0 violates the strict inequality
    assert_eq!(
      interpret("Resolve[ForAll[x, x^2 > 0], Reals]").unwrap(),
      "False"
    );
  }

  #[test]
  fn parametric_conditions() {
    assert_eq!(
      interpret("Resolve[Exists[x, x^2 == c], Reals]").unwrap(),
      "c >= 0"
    );
    assert_eq!(
      interpret("Resolve[ForAll[x, x^2 + c > 0], Reals]").unwrap(),
      "c > 0"
    );
  }

  // Multiple bound variables, additively separable polynomial conditions.
  #[test]
  fn exists_multivar() {
    // The unit disc is non-empty.
    assert_eq!(
      interpret("Resolve[Exists[{x, y}, x^2 + y^2 < 1]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("Resolve[Exists[{x, y}, x^2 + y^2 < 1], Reals]").unwrap(),
      "True"
    );
    // x^2 + y^2 >= 0, so it can never be < -1.
    assert_eq!(
      interpret("Resolve[Exists[{x, y}, x^2 + y^2 < -1]]").unwrap(),
      "False"
    );
    // Over the reals the sum of squares can't equal -1 ...
    assert_eq!(
      interpret("Resolve[Exists[{x, y}, x^2 + y^2 == -1], Reals]").unwrap(),
      "False"
    );
    // ... but over the (default) complexes it can.
    assert_eq!(
      interpret("Resolve[Exists[{x, y}, x^2 + y^2 == -1]]").unwrap(),
      "True"
    );
    // An odd power makes the range all of R.
    assert_eq!(
      interpret("Resolve[Exists[{x, y}, x^2 - y^2 == 5], Reals]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("Resolve[Exists[{x, y, z}, x^2 + y^2 + z^2 < 1], Reals]")
        .unwrap(),
      "True"
    );
    // Single-element variable lists behave like the scalar form.
    assert_eq!(interpret("Resolve[Exists[{x}, x^2 < 1]]").unwrap(), "True");
  }

  #[test]
  fn forall_multivar() {
    assert_eq!(
      interpret("Resolve[ForAll[{x, y}, x^2 + y^2 >= 0], Reals]").unwrap(),
      "True"
    );
    // x = y = 0 violates the strict inequality.
    assert_eq!(
      interpret("Resolve[ForAll[{x, y}, x^2 + y^2 > 0], Reals]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("Resolve[ForAll[{x, y}, x^2 + y^2 + 1 > 0], Reals]").unwrap(),
      "True"
    );
    // x^2 - y^2 ranges over all reals, so it is not always positive.
    assert_eq!(
      interpret("Resolve[ForAll[{x, y}, x^2 - y^2 > 0], Reals]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("Resolve[ForAll[{x, y}, x^2 + y^2 != -1], Reals]").unwrap(),
      "True"
    );
  }
}

mod trig_factor {
  use super::*;

  #[test]
  fn sin_plus_minus_cos() {
    // Pi/4 leads when the variable sorts after "Pi"
    assert_eq!(
      interpret("TrigFactor[Sin[x] + Cos[x]]").unwrap(),
      "Sqrt[2]*Sin[Pi/4 + x]"
    );
    assert_eq!(
      interpret("TrigFactor[Sin[x] - Cos[x]]").unwrap(),
      "-(Sqrt[2]*Sin[Pi/4 - x])"
    );
    assert_eq!(
      interpret("TrigFactor[Cos[x] - Sin[x]]").unwrap(),
      "Sqrt[2]*Sin[Pi/4 - x]"
    );
    // The variable leads when it sorts before "Pi"
    assert_eq!(
      interpret("TrigFactor[Sin[a] + Cos[a]]").unwrap(),
      "Sqrt[2]*Sin[a + Pi/4]"
    );
    assert_eq!(
      interpret("TrigFactor[Sin[a] - Cos[a]]").unwrap(),
      "Sqrt[2]*Sin[a - Pi/4]"
    );
    assert_eq!(
      interpret("TrigFactor[Cos[a] - Sin[a]]").unwrap(),
      "-(Sqrt[2]*Sin[a - Pi/4])"
    );
    // Composite arguments
    assert_eq!(
      interpret("TrigFactor[Sin[a + b] + Cos[a + b]]").unwrap(),
      "Sqrt[2]*Sin[a + b + Pi/4]"
    );
    assert_eq!(
      interpret("TrigFactor[Sin[a + b] - Cos[a + b]]").unwrap(),
      "Sqrt[2]*Sin[a + b - Pi/4]"
    );
    assert_eq!(
      interpret("TrigFactor[Cos[a + b] - Sin[a + b]]").unwrap(),
      "-(Sqrt[2]*Sin[a + b - Pi/4])"
    );
  }

  #[test]
  fn one_plus_minus_trig_half_angle_squares() {
    assert_eq!(interpret("TrigFactor[1 + Cos[x]]").unwrap(), "2*Cos[x/2]^2");
    assert_eq!(interpret("TrigFactor[1 - Cos[x]]").unwrap(), "2*Sin[x/2]^2");
    assert_eq!(
      interpret("TrigFactor[1 + Sin[x]]").unwrap(),
      "2*Sin[Pi/4 + x/2]^2"
    );
    assert_eq!(
      interpret("TrigFactor[1 - Sin[x]]").unwrap(),
      "2*Sin[Pi/4 - x/2]^2"
    );
    assert_eq!(
      interpret("TrigFactor[1 - Sin[a]]").unwrap(),
      "2*Sin[a/2 - Pi/4]^2"
    );
    // Double angles halve exactly instead of printing (2*a)/2
    assert_eq!(interpret("TrigFactor[1 + Cos[2 a]]").unwrap(), "2*Cos[a]^2");
    assert_eq!(interpret("TrigFactor[1 - Cos[2 a]]").unwrap(), "2*Sin[a]^2");
    assert_eq!(
      interpret("TrigFactor[1 + Sin[2 a]]").unwrap(),
      "2*Sin[a + Pi/4]^2"
    );
    assert_eq!(
      interpret("TrigFactor[1 - Sin[2 a]]").unwrap(),
      "2*Sin[a - Pi/4]^2"
    );
  }

  #[test]
  fn double_angles() {
    assert_eq!(
      interpret("TrigFactor[Sin[2 x]]").unwrap(),
      "2*Cos[x]*Sin[x]"
    );
    assert_eq!(
      interpret("TrigFactor[Sin[2 t]]").unwrap(),
      "2*Cos[t]*Sin[t]"
    );
    assert_eq!(
      interpret("TrigFactor[Cos[2 x]]").unwrap(),
      "2*Sin[Pi/4 - x]*Sin[Pi/4 + x]"
    );
    assert_eq!(
      interpret("TrigFactor[Cos[2 a]]").unwrap(),
      "-2*Sin[a - Pi/4]*Sin[a + Pi/4]"
    );
  }

  #[test]
  fn difference_of_squares() {
    assert_eq!(
      interpret("TrigFactor[Sin[x]^2 - Cos[x]^2]").unwrap(),
      "-2*Sin[Pi/4 - x]*Sin[Pi/4 + x]"
    );
    assert_eq!(
      interpret("TrigFactor[Cos[x]^2 - Sin[x]^2]").unwrap(),
      "2*Sin[Pi/4 - x]*Sin[Pi/4 + x]"
    );
    assert_eq!(
      interpret("TrigFactor[Sin[a]^2 - Cos[a]^2]").unwrap(),
      "2*Sin[a - Pi/4]*Sin[a + Pi/4]"
    );
    assert_eq!(
      interpret("TrigFactor[Sin[2 x]^2 - Cos[2 x]^2]").unwrap(),
      "-2*Sin[Pi/4 - 2*x]*Sin[Pi/4 + 2*x]"
    );
  }

  #[test]
  fn sin_sum_to_product() {
    // Sin[p] +- Sin[q] sum-to-product for distinct atomic arguments.
    assert_eq!(
      interpret("TrigFactor[Sin[x] + Sin[y]]").unwrap(),
      "2*Cos[x/2 - y/2]*Sin[x/2 + y/2]"
    );
    assert_eq!(
      interpret("TrigFactor[Sin[x] - Sin[y]]").unwrap(),
      "2*Cos[x/2 + y/2]*Sin[x/2 - y/2]"
    );
    assert_eq!(
      interpret("TrigFactor[Sin[a] + Sin[b]]").unwrap(),
      "2*Cos[a/2 - b/2]*Sin[a/2 + b/2]"
    );
    assert_eq!(
      interpret("TrigFactor[Sin[a] - Sin[b]]").unwrap(),
      "2*Cos[a/2 + b/2]*Sin[a/2 - b/2]"
    );
    // Both terms negative: overall sign folds into the leading -2.
    assert_eq!(
      interpret("TrigFactor[-Sin[x] - Sin[y]]").unwrap(),
      "-2*Cos[x/2 - y/2]*Sin[x/2 + y/2]"
    );
    // Reversed difference normalizes to x-before-y with a pulled sign.
    assert_eq!(
      interpret("TrigFactor[Sin[y] - Sin[x]]").unwrap(),
      "-2*Cos[x/2 + y/2]*Sin[x/2 - y/2]"
    );
  }

  #[test]
  fn passthrough_when_nothing_factors() {
    assert_eq!(interpret("TrigFactor[Sin[x]]").unwrap(), "Sin[x]");
    assert_eq!(interpret("TrigFactor[x + 1]").unwrap(), "1 + x");
    // Integer-multiple arguments factor further in Wolfram, so the
    // single-step sum-to-product rule deliberately leaves them alone.
    assert_eq!(
      interpret("TrigFactor[Sin[2 x] + Sin[4 x]]").unwrap(),
      "Sin[2*x] + Sin[4*x]"
    );
  }
}

mod subresultants {
  use super::*;

  #[test]
  fn integer_chains() {
    // First element is the resultant; 0 there signals a common root
    assert_eq!(
      interpret("Subresultants[x^2 - 1, x^3 - 1, x]").unwrap(),
      "{0, 1, 1}"
    );
    assert_eq!(
      interpret("Subresultants[x^4 + x^2 + 1, x^2 + 1, x]").unwrap(),
      "{1, 0, 1}"
    );
    assert_eq!(
      interpret("Subresultants[x^3 - 2 x + 1, x^2 - 1, x]").unwrap(),
      "{0, -1, 1}"
    );
    // gcd of degree 1: s_0 = 0, s_1 != 0
    assert_eq!(
      interpret("Subresultants[x^2 - 4, x^2 - 5 x + 6, x]").unwrap(),
      "{0, -5, 1}"
    );
  }

  #[test]
  fn argument_order_changes_sign() {
    assert_eq!(interpret("Subresultants[x - 2, x^3, x]").unwrap(), "{8, 1}");
    assert_eq!(
      interpret("Subresultants[x^3, x - 2, x]").unwrap(),
      "{-8, 1}"
    );
  }

  #[test]
  fn symbolic_coefficients() {
    assert_eq!(
      interpret("Subresultants[x^2 + a, x + b, x]").unwrap(),
      "{a + b^2, 1}"
    );
    // Classic discriminant pair (p, p')
    assert_eq!(
      interpret("Subresultants[a x^2 + b x + c, 2 a x + b, x]").unwrap(),
      "{-(a*b^2) + 4*a^2*c, 2*a}"
    );
  }

  #[test]
  fn degenerate_inputs() {
    // Constant polynomial: only the resultant c^deg remains
    assert_eq!(interpret("Subresultants[3, x^2 + 1, x]").unwrap(), "{9}");
    // Two constants: empty Sylvester block determinant
    assert_eq!(interpret("Subresultants[3, 5, x]").unwrap(), "{1}");
    // Zero polynomial has no subresultant chain
    assert_eq!(interpret("Subresultants[x^2 + 1, 0, x]").unwrap(), "{}");
    // Identical polynomials: everything but the trivial entry vanishes
    assert_eq!(
      interpret("Subresultants[x^2 + 1, x^2 + 1, x]").unwrap(),
      "{0, 0, 1}"
    );
  }
}

mod count_roots {
  use super::*;

  #[test]
  fn all_reals_simple() {
    assert_eq!(interpret("CountRoots[x^2 - 1, x]").unwrap(), "2");
    assert_eq!(interpret("CountRoots[x^2 + 1, x]").unwrap(), "0");
    assert_eq!(interpret("CountRoots[x^3 - x, x]").unwrap(), "3");
    assert_eq!(interpret("CountRoots[x^4 + 1, x]").unwrap(), "0");
    assert_eq!(interpret("CountRoots[x^6 - 1, x]").unwrap(), "2");
  }

  #[test]
  fn counts_with_multiplicity() {
    assert_eq!(interpret("CountRoots[(x - 2)^3, x]").unwrap(), "3");
    assert_eq!(interpret("CountRoots[(x^2 - 2)^2, x]").unwrap(), "4");
    assert_eq!(
      interpret("CountRoots[(x - 1)^2 (x - 2)^3 (x - 3), x]").unwrap(),
      "6"
    );
    // Triple root at the origin.
    assert_eq!(interpret("CountRoots[x^3, x]").unwrap(), "3");
  }

  #[test]
  fn closed_interval_includes_endpoints() {
    let p = "(x - 1) (x - 2) (x - 3) (x - 4)";
    assert_eq!(
      interpret(&format!("CountRoots[{p}, {{x, 0, 5}}]")).unwrap(),
      "4"
    );
    assert_eq!(
      interpret(&format!("CountRoots[{p}, {{x, 2, 4}}]")).unwrap(),
      "3"
    );
    assert_eq!(
      interpret(&format!("CountRoots[{p}, {{x, 2, 3}}]")).unwrap(),
      "2"
    );
  }

  #[test]
  fn interval_with_irrational_roots() {
    // Sqrt[2] ~ 1.414 lies in [0, 2] but not [0, 1].
    assert_eq!(interpret("CountRoots[x^2 - 2, {x, 0, 1}]").unwrap(), "0");
    assert_eq!(interpret("CountRoots[x^2 - 2, {x, 0, 2}]").unwrap(), "1");
    assert_eq!(interpret("CountRoots[x^2 - 2, x]").unwrap(), "2");
  }

  #[test]
  fn rational_roots_and_bounds() {
    assert_eq!(
      interpret("CountRoots[(x - 1/2) (x - 3/2), x]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("CountRoots[(x - 1/2) (x - 3/2), {x, 0, 1}]").unwrap(),
      "1"
    );
  }

  #[test]
  fn infinite_bounds() {
    assert_eq!(
      interpret(
        "CountRoots[(x - 1) (x - 2) (x - 3) (x - 4), {x, -Infinity, Infinity}]"
      )
      .unwrap(),
      "4"
    );
    assert_eq!(
      interpret("CountRoots[(x + 5) (x - 5), {x, -Infinity, 0}]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("CountRoots[(x + 5) (x - 5), {x, 0, Infinity}]").unwrap(),
      "1"
    );
  }

  #[test]
  fn multiplicity_inside_interval() {
    // Double root at 1, triple at 2, simple at 3; on [1, 2]: 2 + 3 = 5.
    assert_eq!(
      interpret("CountRoots[(x - 1)^2 (x - 2)^3 (x - 3), {x, 1, 2}]").unwrap(),
      "5"
    );
  }

  #[test]
  fn constants_and_linear() {
    assert_eq!(interpret("CountRoots[5, x]").unwrap(), "0");
    assert_eq!(interpret("CountRoots[x, x]").unwrap(), "1");
  }

  #[test]
  fn non_polynomial_stays_unevaluated() {
    assert_eq!(
      interpret("CountRoots[Sin[x], x]").unwrap(),
      "CountRoots[Sin[x], x]"
    );
  }
}

mod arctan_two_arg {
  use super::*;

  // ArcTan[x, y] reduces to the quadrant-adjusted ArcTan[y/x], kept symbolic
  // when it doesn't simplify further.
  #[test]
  fn first_quadrant() {
    assert_eq!(interpret("ArcTan[3, 4]").unwrap(), "ArcTan[4/3]");
    assert_eq!(interpret("ArcTan[2, 6]").unwrap(), "ArcTan[3]");
    assert_eq!(interpret("ArcTan[1/2, 3/4]").unwrap(), "ArcTan[3/2]");
  }

  #[test]
  fn other_quadrants() {
    assert_eq!(interpret("ArcTan[3, -4]").unwrap(), "-ArcTan[4/3]");
    assert_eq!(interpret("ArcTan[-3, 4]").unwrap(), "Pi - ArcTan[4/3]");
    assert_eq!(interpret("ArcTan[-3, -4]").unwrap(), "-Pi + ArcTan[4/3]");
    assert_eq!(interpret("ArcTan[-1/2, 3/4]").unwrap(), "Pi - ArcTan[3/2]");
  }

  #[test]
  fn axes_and_special_angles() {
    assert_eq!(interpret("ArcTan[0, 5]").unwrap(), "Pi/2");
    assert_eq!(interpret("ArcTan[5, 0]").unwrap(), "0");
    assert_eq!(interpret("ArcTan[-5, 0]").unwrap(), "Pi");
    assert_eq!(interpret("ArcTan[1, 1]").unwrap(), "Pi/4");
    assert_eq!(interpret("ArcTan[1, Sqrt[3]]").unwrap(), "Pi/3");
  }

  // Single-argument ArcTan is odd: negative integers/rationals factor the sign.
  #[test]
  fn single_arg_odd_function() {
    assert_eq!(interpret("ArcTan[-2]").unwrap(), "-ArcTan[2]");
    assert_eq!(interpret("ArcTan[-4/3]").unwrap(), "-ArcTan[4/3]");
    assert_eq!(interpret("ArcTan[-1]").unwrap(), "-1/4*Pi");
  }
}

mod arccot_exact {
  use super::*;

  // ArcCot keeps exact arguments symbolic unless they reduce to a closed
  // form; it must not numericize an exact integer like ArcCot[2].
  #[test]
  fn integer_arguments_stay_symbolic() {
    assert_eq!(interpret("ArcCot[2]").unwrap(), "ArcCot[2]");
    assert_eq!(interpret("ArcCot[3]").unwrap(), "ArcCot[3]");
  }

  #[test]
  fn closed_form_values() {
    assert_eq!(interpret("ArcCot[0]").unwrap(), "Pi/2");
    assert_eq!(interpret("ArcCot[1]").unwrap(), "Pi/4");
    assert_eq!(interpret("ArcCot[Sqrt[3]]").unwrap(), "Pi/6");
    assert_eq!(interpret("ArcCot[1/Sqrt[3]]").unwrap(), "Pi/3");
  }

  #[test]
  fn limits_at_infinity() {
    assert_eq!(interpret("ArcCot[Infinity]").unwrap(), "0");
    assert_eq!(interpret("ArcCot[-Infinity]").unwrap(), "0");
  }

  // Odd function: the sign factors out.
  #[test]
  fn odd_function() {
    assert_eq!(interpret("ArcCot[-1]").unwrap(), "-1/4*Pi");
    assert_eq!(interpret("ArcCot[-2]").unwrap(), "-ArcCot[2]");
    assert_eq!(interpret("ArcCot[-3]").unwrap(), "-ArcCot[3]");
  }

  // Inexact arguments still evaluate numerically. The exact last ULP of the
  // result depends on the platform's libm (macOS/wolframscript give
  // 0.46364760900080615; glibc on Linux CI differs by one ULP), so compare
  // numerically rather than by exact string — matching how the other
  // floating-point tests in this suite assert (see `rms_reals`).
  #[test]
  fn real_argument_is_numeric() {
    let result = interpret("ArcCot[2.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.46364760900080615).abs() < 1e-12);
  }
}

mod arccsch_arccoth_exact {
  use super::*;

  // ArcCsch keeps exact arguments symbolic (it had numericized ArcCsch[2]).
  #[test]
  fn arccsch_exact_symbolic() {
    assert_eq!(interpret("ArcCsch[2]").unwrap(), "ArcCsch[2]");
    assert_eq!(interpret("ArcCsch[1]").unwrap(), "ArcCsch[1]");
  }

  #[test]
  fn arccsch_special_and_odd() {
    assert_eq!(interpret("ArcCsch[0]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("ArcCsch[Infinity]").unwrap(), "0");
    assert_eq!(interpret("ArcCsch[-Infinity]").unwrap(), "0");
    assert_eq!(interpret("ArcCsch[-2]").unwrap(), "-ArcCsch[2]");
    assert_eq!(interpret("ArcCsch[-1/2]").unwrap(), "-ArcCsch[1/2]");
  }

  #[test]
  fn arccsch_real_numeric() {
    assert_eq!(interpret("ArcCsch[2.0]").unwrap(), "0.48121182505960347");
  }

  // ArcCoth gains the odd-function negation and ±Infinity limits.
  #[test]
  fn arccoth_odd_and_infinity() {
    assert_eq!(interpret("ArcCoth[-2]").unwrap(), "-ArcCoth[2]");
    assert_eq!(interpret("ArcCoth[-1/2]").unwrap(), "-ArcCoth[1/2]");
    assert_eq!(interpret("ArcCoth[Infinity]").unwrap(), "0");
    assert_eq!(interpret("ArcCoth[-Infinity]").unwrap(), "0");
  }

  #[test]
  fn arccoth_existing_values_unchanged() {
    assert_eq!(interpret("ArcCoth[1]").unwrap(), "Infinity");
    assert_eq!(interpret("ArcCoth[-1]").unwrap(), "-Infinity");
    assert_eq!(interpret("ArcCoth[2]").unwrap(), "ArcCoth[2]");
  }
}

#[cfg(test)]
mod log_power_exact {
  use woxi::interpret;

  // Log[Sqrt[n]] = Log[n]/2 for positive integers (verified against wolframscript)
  #[test]
  fn log_sqrt_integer() {
    assert_eq!(interpret("Log[Sqrt[2]]").unwrap(), "Log[2]/2");
    assert_eq!(interpret("Log[Sqrt[15]]").unwrap(), "Log[15]/2");
    assert_eq!(interpret("Log[Sqrt[7]]").unwrap(), "Log[7]/2");
  }

  // Log[n^(p/q)] = (p/q) Log[n] for fractional exponents
  #[test]
  fn log_integer_fractional_power() {
    assert_eq!(interpret("Log[3^(2/5)]").unwrap(), "(2*Log[3])/5");
    assert_eq!(interpret("Log[2^(1/3)]").unwrap(), "Log[2]/3");
    assert_eq!(interpret("Log[6^(2/3)]").unwrap(), "(2*Log[6])/3");
  }

  // Negative fractional exponent
  #[test]
  fn log_integer_negative_power() {
    assert_eq!(interpret("Log[5^(-1/2)]").unwrap(), "-1/2*Log[5]");
  }

  // Positive real constant bases: Pi, EulerGamma, GoldenRatio, Catalan
  #[test]
  fn log_constant_power() {
    assert_eq!(interpret("Log[Pi^(1/2)]").unwrap(), "Log[Pi]/2");
    assert_eq!(interpret("Log[Pi^(3/2)]").unwrap(), "(3*Log[Pi])/2");
    assert_eq!(
      interpret("Log[EulerGamma^(1/2)]").unwrap(),
      "Log[EulerGamma]/2"
    );
    assert_eq!(
      interpret("Log[GoldenRatio^(1/2)]").unwrap(),
      "Log[GoldenRatio]/2"
    );
    assert_eq!(interpret("Log[Catalan^(1/3)]").unwrap(), "Log[Catalan]/3");
  }

  // LogGamma[1/2] = Log[Sqrt[Pi]] must now simplify to Log[Pi]/2
  #[test]
  fn log_gamma_half() {
    assert_eq!(interpret("LogGamma[1/2]").unwrap(), "Log[Pi]/2");
  }

  // Symbolic bases must NOT simplify (sign unknown); E base handled separately;
  // integer base with |exp|>1 stays a product; perfect powers reduce first.
  #[test]
  fn log_power_passthrough_and_special() {
    assert_eq!(interpret("Log[Sqrt[x]]").unwrap(), "Log[Sqrt[x]]");
    assert_eq!(interpret("Log[E^(1/2)]").unwrap(), "1/2");
    assert_eq!(interpret("Log[2^(3/2)]").unwrap(), "Log[2*Sqrt[2]]");
    assert_eq!(interpret("Log[8^(1/3)]").unwrap(), "Log[2]");
  }

  // Numeric value is unchanged by the symbolic simplification
  #[test]
  fn log_sqrt_numeric() {
    assert_eq!(interpret("N[Log[Sqrt[2]]]").unwrap(), "0.34657359027997264");
  }
}

// Sqrt[a]/Sqrt[b] = Sqrt[a/b] for positive numeric/constant radicands, matching
// wolframscript. Free symbols stay split; a non-radical numerator stays as-is.
mod sqrt_ratio_combination {
  use super::*;

  #[test]
  fn integer_over_constant() {
    assert_eq!(interpret("Sqrt[2]/Sqrt[Pi]").unwrap(), "Sqrt[2/Pi]");
    assert_eq!(interpret("Sqrt[2]/Sqrt[E]").unwrap(), "Sqrt[2/E]");
  }

  #[test]
  fn constant_over_integer() {
    assert_eq!(interpret("Sqrt[Pi]/Sqrt[2]").unwrap(), "Sqrt[Pi/2]");
  }

  #[test]
  fn constant_over_constant() {
    assert_eq!(interpret("Sqrt[Pi]/Sqrt[E]").unwrap(), "Sqrt[Pi/E]");
  }

  #[test]
  fn with_outer_coefficient() {
    assert_eq!(interpret("2*Sqrt[2]/Sqrt[Pi]").unwrap(), "2*Sqrt[2/Pi]");
  }

  #[test]
  fn product_radicands() {
    assert_eq!(interpret("Sqrt[2*Pi]/Sqrt[3]").unwrap(), "Sqrt[(2*Pi)/3]");
    assert_eq!(interpret("Sqrt[2]/Sqrt[3*Pi]").unwrap(), "Sqrt[2/(3*Pi)]");
  }

  // The merged radicand must be canonicalised: Sqrt[2/Pi]/Sqrt[5] has a
  // numerator base of 2/Pi, and dividing it by 5 must reduce to 2/(5*Pi)
  // rather than leaving the unreduced (2*Pi^(-1))/5 inside the Sqrt.
  #[test]
  fn reciprocal_radicand_is_normalized() {
    assert_eq!(interpret("Sqrt[2/Pi]/Sqrt[5]").unwrap(), "Sqrt[2/(5*Pi)]");
    assert_eq!(interpret("Sqrt[2/Pi]/Sqrt[3]").unwrap(), "Sqrt[2/(3*Pi)]");
    assert_eq!(
      interpret("2*Sqrt[2/Pi]*1/Sqrt[1 + 2^2]").unwrap(),
      "2*Sqrt[2/(5*Pi)]"
    );
  }

  #[test]
  fn integer_over_integer_unchanged() {
    assert_eq!(interpret("Sqrt[3]/Sqrt[5]").unwrap(), "Sqrt[3/5]");
    assert_eq!(interpret("Sqrt[6]/Sqrt[2]").unwrap(), "Sqrt[3]");
  }

  #[test]
  fn free_symbols_stay_split() {
    assert_eq!(interpret("Sqrt[x]/Sqrt[y]").unwrap(), "Sqrt[x]/Sqrt[y]");
    assert_eq!(interpret("Sqrt[2]/Sqrt[a]").unwrap(), "Sqrt[2]/Sqrt[a]");
  }

  #[test]
  fn non_radical_numerator_stays() {
    // Numerator 2 is not a square root, so wolframscript keeps 2/Sqrt[Pi].
    assert_eq!(interpret("2/Sqrt[Pi]").unwrap(), "2/Sqrt[Pi]");
  }

  #[test]
  fn besselj_half_integer() {
    assert_eq!(
      interpret("BesselJ[1/2, x]").unwrap(),
      "(Sqrt[2/Pi]*Sin[x])/Sqrt[x]"
    );
    assert_eq!(
      interpret("BesselJ[-1/2, x]").unwrap(),
      "(Sqrt[2/Pi]*Cos[x])/Sqrt[x]"
    );
  }
}
