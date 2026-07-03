use super::*;

// Regression tests for fixes that, together, let memoized recursive programs
// (e.g. the Egyptian-fraction RosettaCode task) run correctly and in linear
// time. Each group targets a distinct bug uncovered while activating that
// script.
mod large_number_and_memoization {
  use super::*;

  // `f[x_] := f[x] = …` (the memoization idiom) must dispatch in O(1) and
  // return the stored value, rather than accumulating O(n) DownValues that
  // are scanned linearly (which made the whole recursion O(n²)).
  mod memoization {
    use super::*;

    #[test]
    fn returns_stored_value() {
      assert_eq!(interpret("f[n_] := f[n] = n^2; f[7]").unwrap(), "49");
    }

    #[test]
    fn fibonacci_with_base_cases() {
      assert_eq!(
        interpret(
          "fib[0] = 0; fib[1] = 1; \
           fib[n_] := fib[n] = fib[n - 1] + fib[n - 2]; fib[30]"
        )
        .unwrap(),
        "832040"
      );
    }

    #[test]
    fn redefining_a_literal_overrides_it() {
      assert_eq!(interpret("g[3] = 10; g[3] = 99; g[3]").unwrap(), "99");
    }

    #[test]
    fn clear_resets_memoized_values() {
      assert_eq!(interpret("h[2] = 5; Clear[h]; h[2]").unwrap(), "h[2]");
    }

    #[test]
    fn multi_argument_memoization() {
      assert_eq!(
        interpret("a[i_, j_] := a[i, j] = i*10 + j; a[2, 3]").unwrap(),
        "23"
      );
    }

    // DownValues must still report memoized values (reconstructed from the
    // cache) alongside the pattern rule.
    #[test]
    fn downvalues_reports_memoized_entries() {
      let out = interpret("k[n_] := k[n] = n + 1; k[5]; Length[DownValues[k]]")
        .unwrap();
      // One pattern rule + one memoized value for k[5].
      assert_eq!(out, "2");
    }
  }

  // Plus of rationals whose numerator/denominator exceed i128 must still
  // combine into a single reduced rational (previously left unevaluated).
  mod big_rational_addition {
    use super::*;

    #[test]
    fn small_plus_big_denominator() {
      assert_eq!(
        interpret("1/3 + 1/170141183460469231731687303715884105729").unwrap(),
        "56713727820156410577229101238628035244/\
         170141183460469231731687303715884105729"
      );
    }

    #[test]
    fn sum_of_unit_fractions_with_huge_denominators() {
      // A valid Egyptian-fraction expansion of 44/53 sums back to 44/53.
      assert_eq!(
        interpret(
          "Total[{1/2, 1/4, 1/13, 1/307, 1/120871, 1/20453597227, \
           1/697249399186783218655, \
           1/1458470173998990524806872692984177836808420}]"
        )
        .unwrap(),
        "44/53"
      );
    }

    #[test]
    fn integer_plus_big_rational() {
      assert_eq!(
        interpret("5 + 1/100000000000000000000000000000000000000").unwrap(),
        "500000000000000000000000000000000000001/\
         100000000000000000000000000000000000000"
      );
    }
  }

  // Floor/Ceiling of exact rationals must be computed with BigInt arithmetic,
  // never via f64 (which overflows to ±inf beyond ~1.8e308 and then saturates
  // `as i128` to i128::MAX).
  mod exact_floor_ceiling {
    use super::*;

    #[test]
    fn ceiling_beyond_f64_range() {
      assert_eq!(
        interpret("Ceiling[10^400 / 3]").unwrap(),
        format!("3{}4", "3".repeat(398))
      );
    }

    #[test]
    fn floor_negative_beyond_f64_range() {
      assert_eq!(
        interpret("Floor[-(10^400)/3]").unwrap(),
        format!("-3{}4", "3".repeat(398))
      );
    }

    #[test]
    fn small_rationals_unchanged() {
      assert_eq!(
        interpret(
          "{Ceiling[7/2], Floor[7/2], Ceiling[-7/2], Floor[-7/2], \
           Ceiling[5], Floor[-5]}"
        )
        .unwrap(),
        "{4, 3, -3, -4, 5, -5}"
      );
    }
  }

  // Sort / MaximalBy / MinimalBy must order BigIntegers exactly, not by an
  // f64 projection that collapses everything beyond f64 range to inf.
  mod large_integer_ordering {
    use super::*;

    #[test]
    fn sort_orders_huge_integers() {
      assert_eq!(
        interpret(
          "Sort[{10^320, 10^310, 10^330}] === {10^310, 10^320, 10^330}"
        )
        .unwrap(),
        "True"
      );
    }

    #[test]
    fn maximal_by_picks_largest_huge_key() {
      assert_eq!(
        interpret(
          "MaximalBy[{{1/10^310}, {1/10^330}, {1/10^320}}, \
           Denominator@*Last] === {{1/10^330}}"
        )
        .unwrap(),
        "True"
      );
    }

    #[test]
    fn minimal_by_picks_smallest_huge_key() {
      assert_eq!(
        interpret(
          "MinimalBy[{{1/10^310}, {1/10^330}, {1/10^320}}, \
           Denominator@*Last] === {{1/10^310}}"
        )
        .unwrap(),
        "True"
      );
    }
  }

  // A Composition (`@*`) used as the key/element function must be applied,
  // not have the argument appended into the composition.
  mod composition_application {
    use super::*;

    #[test]
    fn map_applies_composition() {
      assert_eq!(
        interpret("Map[Length@*Union, {{1, 1, 2}, {3}}]").unwrap(),
        "{2, 1}"
      );
    }

    #[test]
    fn maximal_by_applies_composition() {
      assert_eq!(
        interpret("MaximalBy[{{1}, {1/2, 1/3}, {1/2}}, Length@*Union]")
          .unwrap(),
        "{{1/2, 1/3}}"
      );
    }
  }

  // Regression for https://github.com/ad-si/Woxi/issues/179:
  // `a + -n x` must fold to `a - n x` even when the coefficient `n`
  // overflows i128 and is stored as a BigInteger. The sign-extraction
  // logic previously only handled `Expr::Integer` coefficients, so a big
  // integer coefficient wrongly printed as `a + -n x`.
  mod plus_negative_bigint_coefficient {
    use super::*;

    #[test]
    fn folds_to_minus() {
      // 2^129 = 680564733841876926926749214863536422912 overflows i128.
      assert_eq!(
        interpret("1 + -2^129 x").unwrap(),
        "1 - 680564733841876926926749214863536422912*x"
      );
      // Still within i128 (2^125), same result shape.
      assert_eq!(
        interpret("1 + -2^125 x").unwrap(),
        "1 - 42535295865117307932921825928971026432*x"
      );
      // Leading symbol, trailing big negative term.
      assert_eq!(
        interpret("x - 2^129 y").unwrap(),
        "x - 680564733841876926926749214863536422912*y"
      );
      // Multiple non-numeric factors after the big coefficient.
      assert_eq!(
        interpret("a + -2^129 b c").unwrap(),
        "a - 680564733841876926926749214863536422912*b*c"
      );
      // InputForm path (expr_to_input_form) folds the sign too.
      assert_eq!(
        interpret("ToString[1 + -2^129 x, InputForm]").unwrap(),
        "1 - 680564733841876926926749214863536422912*x"
      );
    }
  }
}
