// Options and argument forms found missing by the second reference-page
// parameter sweep.

use super::*;

mod map_apply_heads {
  use super::*;

  #[test]
  fn heads_true_also_applies_to_the_head() {
    assert_eq!(
      interpret("MapApply[f, p[x][q[y]], Heads -> True]").unwrap(),
      "f[x][f[y]]"
    );
    assert_eq!(
      interpret("MapApply[f, p[x][q[y]], Heads -> False]").unwrap(),
      "p[x][f[y]]"
    );
  }
}

mod seed_random_method {
  use super::*;

  #[test]
  fn method_alone_reseeds() {
    assert_eq!(
      interpret("SeedRandom[Method -> \"MersenneTwister\"] === Null").unwrap(),
      "True"
    );
  }
}

mod parallel_options {
  use super::*;

  #[test]
  fn parallel_table_ignores_scheduling_options() {
    assert_eq!(
      interpret("ParallelTable[local[i], {i, 4}, DistributedContexts -> None]")
        .unwrap(),
      "{local[1], local[2], local[3], local[4]}"
    );
    assert_eq!(
      interpret("ParallelTable[i^2, {i, 3}, Method -> \"FinestGrained\"]")
        .unwrap(),
      "{1, 4, 9}"
    );
  }

  #[test]
  fn parallelize_ignores_scheduling_options() {
    assert_eq!(
      interpret(
        "Parallelize[Table[local[i], {i, 3}], Method -> \"CoarsestGrained\"]"
      )
      .unwrap(),
      "{local[1], local[2], local[3]}"
    );
  }
}

mod url_character_encoding {
  use super::*;

  #[test]
  fn decode_macintosh_roman() {
    assert_eq!(
      interpret(
        "URLDecode[\"E%9Atv%9As+Lor%87nd+University\", \
         CharacterEncoding -> \"MacintoshRoman\"]"
      )
      .unwrap(),
      "Eötvös Loránd University"
    );
  }

  #[test]
  fn query_encode_macintosh_roman() {
    assert_eq!(
      interpret(
        "URLQueryEncode[{\"university\" -> \"Eötvös Loránd University\"}, \
         CharacterEncoding -> \"MacintoshRoman\"]"
      )
      .unwrap(),
      "university=E%9Atv%9As+Lor%87nd+University"
    );
  }

  #[test]
  fn build_encodes_the_query() {
    assert_eq!(
      interpret(
        "URLBuild[\"https://university.com\", {\"university\" -> \
         \"Eötvös Loránd University\"}, CharacterEncoding -> \"MacintoshRoman\"]"
      )
      .unwrap(),
      "https://university.com/?university=E%9Atv%9As+Lor%87nd+University"
    );
    assert_eq!(
      interpret("URLBuild[\"https://x.com\", {\"q\" -> \"a b&c\"}]").unwrap(),
      "https://x.com/?q=a+b%26c"
    );
  }

  #[test]
  fn decode_keeps_an_encoded_plus() {
    assert_eq!(interpret("URLDecode[\"a%2Bb+c\"]").unwrap(), "a+b c");
    assert_eq!(interpret("URLDecode[URLEncode[\"1+1\"]]").unwrap(), "1+1");
  }

  #[test]
  fn build_keeps_its_options_when_unevaluated() {
    assert_eq!(
      interpret("URLBuild[123, CharacterEncoding -> \"ISOLatin1\"]").unwrap(),
      "URLBuild[123, CharacterEncoding -> ISOLatin1]"
    );
  }

  #[test]
  fn encode_latin1() {
    assert_eq!(
      interpret("URLEncode[\"Eötvös\", CharacterEncoding -> \"ISOLatin1\"]")
        .unwrap(),
      "E%F6tv%F6s"
    );
  }
}

mod first_position_and_case_heads {
  use super::*;

  #[test]
  fn first_position_heads_false_skips_heads() {
    assert_eq!(interpret("FirstPosition[p[x][q[y]], p]").unwrap(), "{0, 0}");
    assert_eq!(
      interpret("FirstPosition[p[x][q[y]], p, Heads -> False]").unwrap(),
      "Missing[NotFound]"
    );
    assert_eq!(
      interpret("FirstPosition[p[x][q[y]], p, d, Heads -> False]").unwrap(),
      "d"
    );
  }

  #[test]
  fn first_case_heads_true_searches_heads() {
    assert_eq!(
      interpret(
        "FirstCase[f[g][h], _Symbol, None, {0, Infinity}, Heads -> True]"
      )
      .unwrap(),
      "f"
    );
    assert_eq!(
      interpret("FirstCase[{1, x}, _Symbol, Heads -> True]").unwrap(),
      "List"
    );
  }

  #[test]
  fn first_case_searches_any_expression() {
    assert_eq!(interpret("FirstCase[f[a, b], _Symbol]").unwrap(), "a");
    assert_eq!(interpret("FirstCase[f[g][h], _Symbol]").unwrap(), "h");
    assert_eq!(
      interpret("FirstCase[f[1, 2], _Symbol]").unwrap(),
      "Missing[NotFound]"
    );
    assert_eq!(interpret("FirstCase[x, _]").unwrap(), "Missing[NotFound]");
  }
}

mod replace_dispatch_level {
  use super::*;

  #[test]
  fn dispatch_table_at_a_level() {
    assert_eq!(
      interpret("Replace[{1, 2}, Dispatch[{1 -> a, 3 -> b}], 1]").unwrap(),
      "{a, 2}"
    );
    assert_eq!(
      interpret("Replace[{1, 2}, Dispatch[{1 -> a, 3 -> b}], {1}]").unwrap(),
      "{a, 2}"
    );
  }
}

mod inverse_trig_complex {
  use super::*;

  #[test]
  fn reciprocal_functions_at_machine_complex_points() {
    // ArcCot[z] is ArcTan[1/z], and likewise for the hyperbolic ones.
    for (inverse, forward) in [
      ("ArcCot", "ArcTan"),
      ("ArcCoth", "ArcTanh"),
      ("ArcSech", "ArcCosh"),
      ("ArcCsch", "ArcSinh"),
    ] {
      assert_eq!(
        interpret(&format!("{inverse}[1.5 + 0.3 I]")).unwrap(),
        interpret(&format!("{forward}[1/(1.5 + 0.3 I)]")).unwrap(),
      );
    }
    assert_eq!(interpret("Head[ArcCot[2. - I]]").unwrap(), "Complex");
  }
}

mod check_abort_propagate {
  use super::*;

  #[test]
  fn propagate_aborts_true_reaborts() {
    assert_eq!(
      interpret("CheckAbort[Abort[], 7, PropagateAborts -> True]; 8").unwrap(),
      "$Aborted"
    );
  }

  #[test]
  fn propagate_aborts_false_or_automatic_absorbs() {
    assert_eq!(
      interpret("CheckAbort[Abort[], 7, PropagateAborts -> False]").unwrap(),
      "7"
    );
    assert_eq!(
      interpret("CheckAbort[Abort[], 7, PropagateAborts -> Automatic]; 8")
        .unwrap(),
      "8"
    );
  }
}

mod algorithm_options {
  use super::*;

  #[test]
  fn resultant_method() {
    assert_eq!(
      interpret("Resultant[x^2 - 1, x - 2, x, Method -> \"Modular\"]").unwrap(),
      "3"
    );
    assert_eq!(
      interpret("Resultant[x^2 - a, x - b, x, Method -> \"BezoutMatrix\"]")
        .unwrap(),
      interpret("Resultant[x^2 - a, x - b, x]").unwrap()
    );
  }

  #[test]
  fn eigensystem_method() {
    assert_eq!(
      interpret("Eigenvalues[{{2, 0}, {0, 3}}, Method -> \"Arnoldi\"]")
        .unwrap(),
      "{3, 2}"
    );
    assert_eq!(
      interpret("Eigensystem[{{2, 0}, {0, 3}}, 1, Method -> \"Arnoldi\"]")
        .unwrap(),
      "{{3}, {{0, 1}}}"
    );
  }

  #[test]
  fn least_squares_tolerance() {
    assert_eq!(
      interpret(
        "LeastSquares[{{1, 2}, {3, 4}, {5, 6}}, {1, 2, 3}, Tolerance -> 0]"
      )
      .unwrap(),
      "{0, 1/2}"
    );
  }

  #[test]
  fn antihermitian_tolerance_on_exact_matrix() {
    assert_eq!(
      interpret("AntihermitianMatrixQ[{{I, 1}, {-1, 0}}, Tolerance -> 10^-12]")
        .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AntisymmetricMatrixQ[{{0, 1}, {-1, 0}}, Tolerance -> 10^-12]")
        .unwrap(),
      "True"
    );
  }

  #[test]
  fn region_measure_goal_options() {
    assert_eq!(interpret("Area[Disk[], AccuracyGoal -> 8]").unwrap(), "Pi");
    assert_eq!(
      interpret("Volume[Ball[], PrecisionGoal -> 5]").unwrap(),
      "(4*Pi)/3"
    );
  }
}

mod overflow_regressions {
  use super::*;

  #[test]
  fn polygamma_at_large_integers() {
    assert_eq!(
      interpret("PolyGamma[100]").unwrap(),
      "360968703235711654233892612988250163157207/\
       69720375229712477164533808935312303556800 - EulerGamma"
    );
    assert_eq!(
      interpret("PolyGamma[1, 100] == Pi^2/6 - HarmonicNumber[99, 2]").unwrap(),
      "True"
    );
  }

  #[test]
  fn divisor_sigma_of_a_large_number() {
    assert_eq!(
      interpret("DivisorSigma[2, 10^30]").unwrap(),
      "1388888888888888888587721618754026546109040824596043713112451"
    );
    assert_eq!(interpret("DivisorSigma[-2, 12]").unwrap(), "35/24");
  }

  #[test]
  fn ramanujan_tau_of_a_large_argument() {
    assert_eq!(
      interpret("Table[RamanujanTau[n], {n, 10}]").unwrap(),
      "{1, -24, 252, -1472, 4830, -6048, -16744, 84480, -113643, -115920}"
    );
    // τ is multiplicative: 72675 = 3^2 5^2 17 19.
    assert_eq!(
      interpret(
        "RamanujanTau[72675] == \
         RamanujanTau[9] RamanujanTau[25] RamanujanTau[17] RamanujanTau[19]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn boolean_function_with_more_than_seven_variables() {
    assert_eq!(
      interpret("Head[BooleanFunction[0, 10]]").unwrap(),
      "BooleanFunction"
    );
  }

  #[test]
  fn random_complex_beyond_machine_range() {
    assert_eq!(
      interpret("Head[RandomComplex[10^1000 + 10^1000 I]]").unwrap(),
      "RandomComplex"
    );
  }

  #[test]
  fn many_channel_image() {
    assert_eq!(
      interpret(
        "ImageDimensions[Image[RandomReal[1, {8, 12, 3}], Interleaving -> False]]"
      )
      .unwrap(),
      "{3, 12}"
    );
  }
}

mod linear_recurrence_backward {
  use super::*;

  #[test]
  fn negative_indices_run_the_recurrence_backwards() {
    assert_eq!(
      interpret("LinearRecurrence[{1, 1}, {1, 1}, {-5, 5}]").unwrap(),
      "{5, -3, 2, -1, 1, 0, 1, 1, 2, 3, 5}"
    );
    assert_eq!(
      interpret("LinearRecurrence[{1, 1}, {1, 1}, {0}]").unwrap(),
      "{0}"
    );
    assert_eq!(
      interpret("LinearRecurrence[{2, 3}, {1, 1}, {-2, 3}]").unwrap(),
      "{-13/27, 5/9, -1/3, 1, 1, 5}"
    );
  }
}

mod series_negative_order {
  use super::*;

  #[test]
  fn principal_part_only() {
    assert_eq!(
      interpret("Series[1/Sin[x]^10, {x, 0, -5}]").unwrap(),
      "SeriesData[x, 0, {1, 0, 5/3, 0, 13/9}, -10, -4, 1]"
    );
    assert_eq!(
      interpret("Series[Csc[x], {x, 0, -1}]").unwrap(),
      "SeriesData[x, 0, {1}, -1, 0, 1]"
    );
    assert_eq!(
      interpret("Series[1/x^10 + 1/x^3, {x, 0, -5}]").unwrap(),
      "SeriesData[x, 0, {1}, -10, -4, 1]"
    );
  }

  #[test]
  fn no_term_left_is_the_order_term_at_the_valuation() {
    assert_eq!(
      interpret("Series[1/x^3, {x, 0, -5}]").unwrap(),
      "SeriesData[x, 0, {}, -3, -3, 1]"
    );
    assert_eq!(
      interpret("Series[Exp[x], {x, 0, -1}]").unwrap(),
      "SeriesData[x, 0, {}, 0, 0, 1]"
    );
  }

  #[test]
  fn deep_pole() {
    assert_eq!(
      interpret("Series[1/x^10, {x, 0, 0}]").unwrap(),
      "SeriesData[x, 0, {1}, -10, 1, 1]"
    );
  }
}

mod throw_through_map {
  use super::*;

  #[test]
  fn throw_applied_by_map_is_caught() {
    assert_eq!(interpret("Catch[Throw /@ {a, b, c}]").unwrap(), "a");
    assert_eq!(interpret("Catch[Map[Throw, {1, 2}]]").unwrap(), "1");
  }
}

mod drop_any_head {
  use super::*;

  #[test]
  fn drop_at_two_levels_of_a_product() {
    assert_eq!(
      interpret("Drop[(a + b + c) (t + u + v) (x + y + z), 2, 2]").unwrap(),
      "z"
    );
    assert_eq!(
      interpret("Drop[f[g[a, b], h[c, d]], 1, 1]").unwrap(),
      "f[h[d]]"
    );
  }
}

mod insert_association_key {
  use super::*;

  #[test]
  fn string_position_names_a_key() {
    assert_eq!(
      interpret("Insert[<|\"a\" -> 1, \"b\" -> 2|>, \"c\" -> 3, \"b\"]")
        .unwrap(),
      "<|a -> 1, c -> 3, b -> 2|>"
    );
  }
}

mod gaussian_factor {
  use super::*;

  #[test]
  fn factor_over_the_gaussian_integers() {
    assert_eq!(
      interpret("Factor[1 + x^2, GaussianIntegers -> True]").unwrap(),
      "(-I + x)*(I + x)"
    );
    assert_eq!(
      interpret("Factor[1 + x^2, Extension -> I]").unwrap(),
      "(-I + x)*(I + x)"
    );
    assert_eq!(
      interpret("FactorList[1 + x^2, GaussianIntegers -> True]").unwrap(),
      "{{1, 1}, {-I + x, 1}, {I + x, 1}}"
    );
  }

  #[test]
  fn irreducible_factors_stay() {
    // x^2 + 2 has no Gaussian factors.
    assert_eq!(
      interpret("Factor[x^2 + 2, GaussianIntegers -> True]").unwrap(),
      "2 + x^2"
    );
  }

  #[test]
  fn factors_multiply_back() {
    assert_eq!(
      interpret(
        "Expand[Factor[x^4 + 10 x^2 + 9, GaussianIntegers -> True]] == \
         x^4 + 10 x^2 + 9"
      )
      .unwrap(),
      "True"
    );
  }
}

mod bitwise_symbolic {
  use super::*;

  #[test]
  fn integers_combine_and_symbols_reduce() {
    assert_eq!(interpret("BitAnd[3, 5, x]").unwrap(), "BitAnd[1, x]");
    assert_eq!(interpret("BitAnd[1, 2, x]").unwrap(), "0");
    assert_eq!(interpret("BitAnd[x, y, y, x]").unwrap(), "BitAnd[x, y]");
    assert_eq!(interpret("BitXor[x, y, y, x]").unwrap(), "0");
    assert_eq!(interpret("BitXor[1, 2, x]").unwrap(), "BitXor[3, x]");
    assert_eq!(interpret("BitOr[0, x]").unwrap(), "x");
    assert_eq!(interpret("BitOr[-1, x]").unwrap(), "-1");
    assert_eq!(interpret("BitAnd[12, 10]").unwrap(), "8");
  }
}

mod misc_argument_forms {
  use super::*;

  #[test]
  fn ordered_q_any_head() {
    assert_eq!(interpret("OrderedQ[f[b, a, c]]").unwrap(), "False");
    assert_eq!(interpret("OrderedQ[f[a, b, c]]").unwrap(), "True");
    assert_eq!(
      interpret("OrderedQ[<|1 -> 10, 2 -> 5, 3 -> 0|>]").unwrap(),
      "False"
    );
  }

  #[test]
  fn extremal_by_up_to() {
    assert_eq!(
      interpret("MaximalBy[{{a, 1}, {b, 1}, {b, 3}}, First, UpTo[4]]").unwrap(),
      interpret("MaximalBy[{{a, 1}, {b, 1}, {b, 3}}, First, 3]").unwrap()
    );
    assert_eq!(
      interpret("MinimalBy[{{a, 1}, {b, 1}, {b, 3}}, Last, UpTo[1]]").unwrap(),
      interpret("MinimalBy[{{a, 1}, {b, 1}, {b, 3}}, Last, 1]").unwrap()
    );
  }

  #[test]
  fn next_prime_of_a_rational() {
    assert_eq!(interpret("NextPrime[7/2]").unwrap(), "5");
    assert_eq!(interpret("NextPrime[7/2, -1]").unwrap(), "3");
    assert_eq!(interpret("NextPrime[7/2, 2]").unwrap(), "7");
  }

  #[test]
  fn real_abs_of_infinity() {
    assert_eq!(interpret("RealAbs[Infinity]").unwrap(), "Infinity");
    assert_eq!(interpret("RealAbs[-Infinity]").unwrap(), "Infinity");
  }

  #[test]
  fn delete_missing_levels() {
    assert_eq!(
      interpret(
        "DeleteMissing[{{a, b, c}, {d, Missing[], {e, Missing[]}}}, 2]"
      )
      .unwrap(),
      "{{a, b, c}, {d, {e, Missing[]}}}"
    );
    assert_eq!(
      interpret(
        "DeleteMissing[{{a, b, c}, {d, Missing[], {e, Missing[]}}}, Infinity]"
      )
      .unwrap(),
      "{{a, b, c}, {d, {e}}}"
    );
    assert_eq!(
      interpret(
        "DeleteMissing[{{a, 1}, {b, 2}, {c, Missing[]}, {d, {Missing[]}}}, 1, 1]"
      )
      .unwrap(),
      "{{a, 1}, {b, 2}, {d, {Missing[]}}}"
    );
  }

  #[test]
  fn first_position_operator_form() {
    assert_eq!(
      interpret("FirstPosition[_Integer][{a, b, 1, 2}]").unwrap(),
      "{3}"
    );
  }

  #[test]
  fn divisible_exact_numbers() {
    assert_eq!(interpret("Divisible[2 Pi, Pi/2]").unwrap(), "True");
    assert_eq!(interpret("Divisible[Sqrt[6], Sqrt[2]]").unwrap(), "False");
  }

  #[test]
  fn integer_partitions_last_ones() {
    assert_eq!(
      interpret("IntegerPartitions[6, All, All, -3]").unwrap(),
      "{{2, 2, 1, 1}, {2, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1}}"
    );
  }

  #[test]
  fn waves_thread_over_lists() {
    assert_eq!(
      interpret("SquareWave[{0.4, 1.2, 3.6}]").unwrap(),
      "{1, 1, -1}"
    );
    assert_eq!(
      interpret("SquareWave[{0, 5}, {0.4, 1.2, 3.6}]").unwrap(),
      "{5, 5, 0}"
    );
    assert_eq!(
      interpret("SquareWave[{{1/4, -1/5}, {0, 1/5}}]").unwrap(),
      "{{1, -1}, {1, 1}}"
    );
  }
}

mod more_parallel_options {
  use super::*;

  #[test]
  fn progress_reporting_and_contexts_are_ignored() {
    assert_eq!(
      interpret("ParallelMap[f, {1, 2}, ProgressReporting -> False]").unwrap(),
      "{f[1], f[2]}"
    );
    assert_eq!(
      interpret("ParallelSelect[{1, 2, 3}, OddQ, DistributedContexts -> None]")
        .unwrap(),
      "{1, 3}"
    );
  }
}

mod url_query_options {
  use super::*;

  #[test]
  fn sort_and_remove_empty_queries() {
    assert_eq!(
      interpret(
        "URLQueryEncode[{\"b\" -> \"2\", \"a\" -> \"1\"}, SortQueries -> True]"
      )
      .unwrap(),
      "a=1&b=2"
    );
    assert_eq!(
      interpret(
        "URLQueryEncode[{\"a\" -> \"\", \"b\" -> \"1\"}, RemoveEmptyQueries -> True]"
      )
      .unwrap(),
      "b=1"
    );
  }
}
