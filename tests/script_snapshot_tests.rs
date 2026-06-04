use std::fs;
use std::path::Path;
use woxi::{
  clear_state, interpret_with_stdout, set_script_command_line, without_shebang,
};

/// Run a script and snapshot its Print[] output.
///
/// By default, runs via the woxi interpreter.
/// Set WOXI_USE_WOLFRAM=true to run via `wolframscript -file` instead,
/// validating that wolframscript produces the same output:
///
///   WOXI_USE_WOLFRAM=true cargo test script_
fn run_script_snapshot(name: &str) {
  run_script_snapshot_with_args(name, &[]);
}

fn run_script_snapshot_with_args(name: &str, args: &[&str]) {
  let path = Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("tests/scripts")
    .join(name);
  let use_wolfram = std::env::var("WOXI_USE_WOLFRAM").as_deref() == Ok("true");

  let stdout = if use_wolfram {
    let mut cmd = std::process::Command::new("wolframscript");
    cmd.arg("-file").arg(&path);
    for arg in args {
      cmd.arg(arg);
    }
    let output = cmd.output().unwrap_or_else(|e| {
      panic!("Failed to run wolframscript on {}: {}", name, e)
    });

    assert!(
      output.status.success(),
      "wolframscript failed on {}: {}",
      name,
      String::from_utf8_lossy(&output.stderr)
    );

    String::from_utf8_lossy(&output.stdout).into_owned()
  } else {
    clear_state();

    // Set $ScriptCommandLine: first element is the script path, rest are args
    let mut cmd_line = vec![path.to_string_lossy().to_string()];
    cmd_line.extend(args.iter().map(|s| s.to_string()));
    set_script_command_line(&cmd_line);

    let content = fs::read_to_string(&path).unwrap();
    let code = without_shebang(&content);

    let result = interpret_with_stdout(&code)
      .unwrap_or_else(|e| panic!("Script {} failed: {}", name, e));

    result.stdout
  };

  insta::assert_snapshot!(name, stdout);
}

macro_rules! script_test {
  ($test_name:ident, $file:expr) => {
    #[test]
    fn $test_name() {
      run_script_snapshot($file);
    }
  };
}

script_test!(script_99_bottles_of_beer, "99_bottles_of_beer.wls");
script_test!(script_abc_problem, "abc_problem.wls");
script_test!(script_fizzbuzz_1, "fizzbuzz_1.wls");
script_test!(script_fizzbuzz_2, "fizzbuzz_2.wls");
script_test!(script_fizzbuzz_3, "fizzbuzz_3.wls");
script_test!(script_fizzbuzz_4, "fizzbuzz_4.wls");
script_test!(script_fizzbuzz_5, "fizzbuzz_5.wls");
script_test!(script_hello_world, "hello_world.wls");
script_test!(script_n_queens_problem_1, "n-queens_problem_1.wls");
script_test!(script_n_queens_problem_2, "n-queens_problem_2.wls");
script_test!(script_lindenmayer, "lindenmayer.wls");
script_test!(script_fibonacci_sequence, "fibonacci_sequence.wls");
script_test!(script_least_common_multiple, "least_common_multiple.wls");
script_test!(script_leap_year, "leap_year.wls");
script_test!(script_identity_matrix, "identity_matrix.wls");
script_test!(script_dot_product, "dot_product.wls");
script_test!(script_copy_a_string, "copy_a_string.wls");
script_test!(
  script_permutations_with_repetitions,
  "permutations_with_repetitions.wls"
);
script_test!(script_reverse_a_string, "reverse_a_string.wls");
script_test!(
  script_increment_a_numerical_string,
  "increment_a_numerical_string.wls"
);
script_test!(script_unicode_variable_names, "unicode_variable_names.wls");
script_test!(script_catamorphism, "catamorphism.wls");
script_test!(
  script_terminal_control_display_an_extended_character,
  "terminal_control_display_an_extended_character.wls"
);

script_test!(script_array_length, "array_length.wls");
script_test!(script_sort_an_integer_array, "sort_an_integer_array.wls");
script_test!(script_test_integerness, "test_integerness.wls");
script_test!(script_hello_world_text, "hello_world_text.wls");
script_test!(
  script_loops_for_with_a_specified_step,
  "loops_for_with_a_specified_step.wls"
);
script_test!(script_loops_while, "loops_while.wls");
script_test!(
  script_loop_over_multiple_arrays_simultaneously,
  "loop_over_multiple_arrays_simultaneously.wls"
);
script_test!(script_loops_for, "loops_for.wls");
script_test!(script_loops_downward_for, "loops_downward_for.wls");
script_test!(
  script_non_decimal_radices_output,
  "non-decimal_radices_output.wls"
);
script_test!(script_exponentiation_order, "exponentiation_order.wls");
script_test!(script_here_document, "here_document.wls");
script_test!(script_loops_continue, "loops_continue.wls");
script_test!(script_string_concatenation, "string_concatenation.wls");
script_test!(
  script_singly_linked_list_element_insertion,
  "singly-linked_list_element_insertion.wls"
);
script_test!(
  script_singly_linked_list_element_definition,
  "singly-linked_list_element_definition.wls"
);
script_test!(
  script_chinese_remainder_theorem,
  "chinese_remainder_theorem.wls"
);
script_test!(
  script_evaluate_binomial_coefficients,
  "evaluate_binomial_coefficients.wls"
);
script_test!(script_array_concatenation, "array_concatenation.wls");
script_test!(script_power_set, "power_set.wls");
script_test!(script_character_codes, "character_codes.wls");
script_test!(script_binary_digits, "binary_digits.wls");
script_test!(
  script_singly_linked_list_traversal,
  "singly-linked_list_traversal.wls"
);
script_test!(script_string_case, "string_case.wls");
script_test!(
  script_determine_if_a_string_is_numeric,
  "determine_if_a_string_is_numeric.wls"
);
script_test!(script_leonardo_numbers, "leonardo_numbers.wls");
script_test!(script_return_multiple_values, "return_multiple_values.wls");
script_test!(script_palindrome_detection, "palindrome_detection.wls");
script_test!(
  script_averages_root_mean_square,
  "averages_root_mean_square.wls"
);
script_test!(
  script_non_decimal_radices_convert,
  "non-decimal_radices_convert.wls"
);
script_test!(script_factors_of_an_integer, "factors_of_an_integer.wls");
script_test!(script_arrays, "arrays.wls");
script_test!(script_map_range, "map_range.wls");
script_test!(script_multisplit, "multisplit.wls");
script_test!(script_flatten_a_list, "flatten_a_list.wls");
script_test!(script_tokenize_a_string, "tokenize_a_string.wls");
script_test!(script_repeat, "repeat.wls");
script_test!(script_loops_do_while, "loops_do-while.wls");
script_test!(script_averages_mode, "averages_mode.wls");
script_test!(script_string_append, "string_append.wls");
script_test!(
  script_generate_lower_case_ascii_alphabet,
  "generate_lower_case_ascii_alphabet.wls"
);
script_test!(script_string_length, "string_length.wls");
script_test!(script_vector_products, "vector_products.wls");
script_test!(
  script_strip_whitespace,
  "strip_whitespace_from_a_string_top_and_tail.wls"
);
script_test!(script_function_definition, "function_definition.wls");
script_test!(
  script_formatted_numeric_output,
  "formatted_numeric_output.wls"
);
script_test!(
  script_sum_digits_of_an_integer,
  "sum_digits_of_an_integer.wls"
);
script_test!(script_inverted_syntax, "inverted_syntax.wls");
script_test!(
  script_longest_common_substring,
  "longest_common_substring.wls"
);
script_test!(script_substring, "substring.wls");
script_test!(script_string_comparison, "string_comparison.wls");
script_test!(script_string_matching, "string_matching.wls");
script_test!(
  script_sum_and_product_of_an_array,
  "sum_and_product_of_an_array.wls"
);
script_test!(
  script_find_first_and_last_set_bit,
  "find_first_and_last_set_bit_of_a_long_integer.wls"
);
script_test!(
  script_trigonometric_functions,
  "trigonometric_functions.wls"
);
script_test!(script_substring_top_and_tail, "substring_top_and_tail.wls");
script_test!(
  script_string_interpolation,
  "string_interpolation_(included).wls"
);
script_test!(script_josephus_problem, "josephus_problem.wls");
script_test!(script_loops_n_plus_one_half, "loops_n_plus_one_half.wls");
script_test!(
  script_partial_function_application,
  "partial_function_application.wls"
);
script_test!(
  script_non_decimal_radices_input,
  "non-decimal_radices_input.wls"
);
script_test!(script_levenshtein_distance, "levenshtein_distance.wls");
script_test!(script_100_doors, "100_doors.wls");
script_test!(script_table_form, "table_form.wls");
script_test!(script_euro_coins_100g, "euro_coins_100g.wls");
script_test!(script_word_wrap, "word_wrap.wls");
script_test!(script_y_combinator, "y_combinator.wls");
script_test!(script_yin_and_yang, "yin_and_yang.wls");
script_test!(
  script_zeckendorf_number_representation,
  "zeckendorf_number_representation.wls"
);

script_test!(script_visualize_a_tree, "visualize_a_tree.wls");
script_test!(script_voronoi_diagram_index, "voronoi_diagram_index.wls");

script_test!(script_entity_store, "entity_store.wls");

script_test!(script_sign_predicates, "sign-predicates.wls");
script_test!(script_euro_coins_100g_2, "euro_coins_100g_2.wls");

script_test!(script_anonymous_recursion, "anonymous_recursion.wls");
script_test!(
  script_averages_pythagorean_means,
  "averages_pythagorean_means.wls"
);
script_test!(script_count_in_factors, "count_in_factors.wls");
script_test!(script_extend_your_language, "extend_your_language.wls");
script_test!(script_jortsort, "jortsort.wls");
script_test!(
  script_move_to_front_algorithm,
  "move-to-front_algorithm.wls"
);
script_test!(script_negative_base_numbers, "negative_base_numbers.wls");
script_test!(
  script_order_disjoint_list_items,
  "order_disjoint_list_items.wls"
);
script_test!(script_pernicious_numbers, "pernicious_numbers.wls");
script_test!(script_population_count, "population_count.wls");
script_test!(script_priority_queue, "priority_queue.wls");
script_test!(
  script_respond_to_an_unknown_method_call,
  "respond_to_an_unknown_method_call.wls"
);
script_test!(
  script_solve_the_no_connection_puzzle,
  "solve_the_no_connection_puzzle.wls"
);
script_test!(
  script_short_circuit_evaluation,
  "short-circuit_evaluation.wls"
);
script_test!(
  script_terminal_control_ringing_the_terminal_bell,
  "terminal_control_ringing_the_terminal_bell.wls"
);
script_test!(script_top_rank_per_group, "top_rank_per_group.wls");
script_test!(
  script_the_twelve_days_of_christmas,
  "the_twelve_days_of_christmas.wls"
);

script_test!(
  script_electron_deflection_tube,
  "electron-deflection-tube.wls"
);
script_test!(script_euro_coins_100g_3, "euro_coins_100g_3.wls");
script_test!(script_euro_coins_100g_4, "euro_coins_100g_4.wls");
script_test!(script_rolldice, "rolldice.wls");
script_test!(script_spring_damper_system, "spring_damper_system.wl");

// === Auto-promoted RosettaCode scripts ===
script_test!(
  script_abundant_deficient_and_perfect_number_classifications,
  "abundant__deficient_and_perfect_number_classifications.wls"
);
script_test!(script_active_object, "active_object.wls");
script_test!(
  script_add_a_variable_to_a_class_instance_at_runtime,
  "add_a_variable_to_a_class_instance_at_runtime.wls"
);
script_test!(script_almost_prime, "almost_prime.wls");
script_test!(script_amb, "amb.wls");
script_test!(script_amicable_pairs, "amicable_pairs.wls");
script_test!(script_arithmetic_evaluation, "arithmetic_evaluation.wls");
script_test!(
  script_arithmetic_geometric_mean,
  "arithmetic-geometric_mean.wls"
);
script_test!(script_assertions, "assertions.wls");
script_test!(
  script_associative_array_creation,
  "associative_array_creation.wls"
);
script_test!(
  script_averages_arithmetic_mean,
  "averages_arithmetic_mean.wls"
);
script_test!(script_averages_mean_angle, "averages_mean_angle.wls");
script_test!(
  script_averages_mean_time_of_day,
  "averages_mean_time_of_day.wls"
);
script_test!(script_averages_median, "averages_median.wls");
script_test!(
  script_averages_simple_moving_average,
  "averages_simple_moving_average.wls"
);
script_test!(script_babbage_problem, "babbage_problem.wls");
script_test!(script_balanced_brackets, "balanced_brackets.wls");
script_test!(script_base64_decode_data, "base64_decode_data.wls");
script_test!(script_benfords_law, "benfords_law.wls");
script_test!(script_binary_search, "binary_search.wls");
script_test!(script_bitmap_read_a_ppm_file, "bitmap_read_a_ppm_file.wls");
script_test!(
  script_bitmap_write_a_ppm_file,
  "bitmap_write_a_ppm_file.wls"
);
script_test!(script_box_the_compass, "box_the_compass.wls");
script_test!(script_caesar_cipher, "caesar_cipher.wls");
script_test!(script_calendar, "calendar.wls");
script_test!(script_call_a_function, "call_a_function.wls");
script_test!(
  script_carmichael_3_strong_pseudoprimes,
  "carmichael_3_strong_pseudoprimes.wls"
);
script_test!(
  script_cartesian_product_of_two_or_more_lists,
  "cartesian_product_of_two_or_more_lists.wls"
);
script_test!(script_casting_out_nines, "casting_out_nines.wls");
script_test!(
  script_circles_of_given_radius_through_two_points,
  "circles_of_given_radius_through_two_points.wls"
);
script_test!(script_closest_pair_problem, "closest-pair_problem.wls");
script_test!(script_combinations, "combinations.wls");
script_test!(script_comma_quibbling, "comma_quibbling.wls");
script_test!(script_command_line_arguments, "command-line_arguments.wls");
script_test!(script_comments, "comments.wls");
script_test!(script_continued_fraction, "continued_fraction.wls");
script_test!(
  script_continued_fraction_arithmetic_construct_from_rational_number,
  "continued_fraction_arithmetic_construct_from_rational_number.wls"
);
script_test!(
  script_convert_seconds_to_compound_duration,
  "convert_seconds_to_compound_duration.wls"
);
script_test!(
  script_count_occurrences_of_a_substring,
  "count_occurrences_of_a_substring.wls"
);
script_test!(script_count_the_coins, "count_the_coins.wls");
script_test!(script_crc_32, "crc-32.wls");
script_test!(
  script_create_a_two_dimensional_array_at_runtime,
  "create_a_two-dimensional_array_at_runtime.wls"
);
script_test!(
  script_create_an_object_native_demonstration,
  "create_an_object_native_demonstration.wls"
);
script_test!(
  script_csv_to_html_translation,
  "csv_to_html_translation.wls"
);
script_test!(
  script_cumulative_standard_deviation,
  "cumulative_standard_deviation.wls"
);
script_test!(script_date_format, "date_format.wls");
script_test!(script_date_manipulation, "date_manipulation.wls");
script_test!(script_day_of_the_week, "day_of_the_week.wls");
script_test!(script_deconvolution_1d, "deconvolution_1d.wls");
script_test!(script_department_numbers, "department_numbers.wls");
// `detect_division_by_zero.wls` (Check[2/0, Print["division by 0"], Power::infy])
// is intentionally not snapshot-tested: wolframscript prints the `Power::infy`
// message to stdout in script mode (so it would land in the snapshot), whereas
// Woxi's script mode routes diagnostic messages to stderr. The computational
// behaviour matches (Check returns Print[...] which writes "division by 0");
// the divergence is purely in message I/O routing.
script_test!(
  script_determine_if_only_one_instance_is_running,
  "determine_if_only_one_instance_is_running.wls"
);
script_test!(script_digital_root, "digital_root.wls");
script_test!(
  script_dutch_national_flag_problem,
  "dutch_national_flag_problem.wls"
);
script_test!(script_emirp_primes, "emirp_primes.wls");
script_test!(script_empty_string, "empty_string.wls");
script_test!(script_entropy, "entropy.wls");
script_test!(script_environment_variables, "environment_variables.wls");
script_test!(script_equilibrium_index, "equilibrium_index.wls");
script_test!(
  script_ethiopian_multiplication,
  "ethiopian_multiplication.wls"
);
script_test!(
  script_exceptions_catch_an_exception_thrown_in_a_nested_call,
  "exceptions_catch_an_exception_thrown_in_a_nested_call.wls"
);
script_test!(
  script_execute_a_markov_algorithm,
  "execute_a_markov_algorithm.wls"
);
script_test!(
  script_exponentiation_operator,
  "exponentiation_operator.wls"
);
script_test!(
  script_extensible_prime_generator,
  "extensible_prime_generator.wls"
);
script_test!(
  script_extreme_floating_point_values,
  "extreme_floating_point_values.wls"
);
script_test!(script_factorial, "factorial.wls");
script_test!(script_fast_fourier_transform, "fast_fourier_transform.wls");
script_test!(script_fasta_format, "fasta_format.wls");
script_test!(
  script_fibonacci_n_step_number_sequences,
  "fibonacci_n-step_number_sequences.wls"
);
script_test!(script_fibonacci_word, "fibonacci_word.wls");
script_test!(script_filter, "filter.wls");
script_test!(
  script_find_first_and_last_set_bit_of_a_long_integer_2,
  "find_first_and_last_set_bit_of_a_long_integer_2.wls"
);
script_test!(script_first_class_functions, "first-class_functions.wls");
script_test!(
  script_first_class_functions_use_numbers_analogously,
  "first-class_functions_use_numbers_analogously.wls"
);
script_test!(script_formal_power_series, "formal_power_series.wls");
script_test!(
  script_formatted_numeric_output_2,
  "formatted_numeric_output_2.wls"
);
script_test!(script_function_definition_2, "function_definition_2.wls");
script_test!(script_function_frequency, "function_frequency.wls");
script_test!(script_gaussian_elimination, "gaussian_elimination.wls");
script_test!(script_general_fizzbuzz, "general_fizzbuzz.wls");
script_test!(script_generic_swap, "generic_swap.wls");
script_test!(script_gray_code, "gray_code.wls");
script_test!(script_grayscale_image, "grayscale_image.wls");
script_test!(
  script_greatest_element_of_a_list,
  "greatest_element_of_a_list.wls"
);
script_test!(script_hamming_numbers, "hamming_numbers.wls");
script_test!(script_happy_numbers, "happy_numbers.wls");
script_test!(
  script_harshad_or_niven_series,
  "harshad_or_niven_series.wls"
);
script_test!(script_hash_join, "hash_join.wls");
script_test!(script_haversine_formula, "haversine_formula.wls");
script_test!(
  script_hello_world_line_printer,
  "hello_world_line_printer.wls"
);
script_test!(
  script_hickerson_series_of_almost_integers,
  "hickerson_series_of_almost_integers.wls"
);
script_test!(script_higher_order_functions, "higher-order_functions.wls");
script_test!(
  script_hofstadter_figure_figure_sequences,
  "hofstadter_figure-figure_sequences.wls"
);
script_test!(script_host_introspection, "host_introspection.wls");
script_test!(script_huffman_coding, "huffman_coding.wls");
script_test!(script_iban, "iban.wls");
// `include_a_file.wls` (Get["myfile.m"] on a missing file) is intentionally
// not snapshot-tested for the same reason as `detect_division_by_zero` above:
// wolframscript prints the `Get::noopen` message to stdout in script mode (so
// it lands in the snapshot), whereas Woxi's library path (interpret_with_stdout)
// routes diagnostic messages to stderr/warnings, leaving stdout empty. The
// computational behaviour matches (Get returns $Failed); the divergence is
// purely in message I/O routing, so the two snapshot-validation engines
// (Woxi vs WOXI_USE_WOLFRAM) can never agree on a single snapshot.
script_test!(script_integer_overflow, "integer_overflow.wls");
script_test!(script_jensens_device, "jensens_device.wls");
script_test!(script_josephus_problem_2, "josephus_problem_2.wls");
script_test!(script_json, "json.wls");
script_test!(
  script_knapsack_problem_continuous,
  "knapsack_problem_continuous.wls"
);
script_test!(
  script_knapsack_problem_unbounded,
  "knapsack_problem_unbounded.wls"
);
script_test!(script_knuth_shuffle, "knuth_shuffle.wls");
script_test!(script_kronecker_product, "kronecker_product.wls");
script_test!(
  script_largest_int_from_concatenated_ints,
  "largest_int_from_concatenated_ints.wls"
);
script_test!(
  script_levenshtein_distance_alignment,
  "levenshtein_distance_alignment.wls"
);
script_test!(
  script_longest_common_substring_2,
  "longest_common_substring_2.wls"
);
script_test!(
  script_longest_increasing_subsequence,
  "longest_increasing_subsequence.wls"
);
script_test!(script_look_and_say_sequence, "look-and-say_sequence.wls");
script_test!(
  script_loops_n_plus_one_half_2,
  "loops_n_plus_one_half_2.wls"
);
script_test!(script_lu_decomposition, "lu_decomposition.wls");
script_test!(script_make_directory_path, "make_directory_path.wls");
script_test!(
  script_matrix_exponentiation_operator,
  "matrix-exponentiation_operator.wls"
);
script_test!(script_matrix_multiplication, "matrix_multiplication.wls");
script_test!(script_matrix_transposition, "matrix_transposition.wls");
script_test!(
  script_maximum_triangle_path_sum,
  "maximum_triangle_path_sum.wls"
);
script_test!(script_mcnuggets_problem, "mcnuggets_problem.wls");
script_test!(script_median_filter, "median_filter.wls");
script_test!(script_menu, "menu.wls");
script_test!(script_metaprogramming, "metaprogramming.wls");
script_test!(script_middle_three_digits, "middle_three_digits.wls");
script_test!(
  script_mind_boggling_card_trick,
  "mind_boggling_card_trick.wls"
);
script_test!(script_modular_inverse, "modular_inverse.wls");
script_test!(script_modulinos, "modulinos.wls");
script_test!(script_monty_hall_problem, "monty_hall_problem.wls");
script_test!(script_multifactorial, "multifactorial.wls");
script_test!(
  script_multiple_distinct_objects,
  "multiple_distinct_objects.wls"
);
script_test!(script_multiplication_tables, "multiplication_tables.wls");
script_test!(script_multiplicative_order, "multiplicative_order.wls");
script_test!(script_mutual_recursion, "mutual_recursion.wls");
script_test!(script_n24_game_solve, "24_game_solve.wls");
script_test!(
  script_narcissistic_decimal_number,
  "narcissistic_decimal_number.wls"
);
script_test!(script_nested_function, "nested_function.wls");
script_test!(
  script_non_continuous_subsequences,
  "non-continuous_subsequences.wls"
);
script_test!(script_nth, "nth.wls");
script_test!(script_numerical_integration, "numerical_integration.wls");
script_test!(
  script_numerical_integration_gauss_legendre_quadrature,
  "numerical_integration_gauss-legendre_quadrature.wls"
);
script_test!(
  script_one_of_n_lines_in_a_file,
  "one_of_n_lines_in_a_file.wls"
);
script_test!(
  script_order_two_numerical_lists,
  "order_two_numerical_lists.wls"
);
script_test!(script_ordered_partitions, "ordered_partitions.wls");
script_test!(script_pangram_checker, "pangram_checker.wls");
script_test!(script_parallel_calculations, "parallel_calculations.wls");
script_test!(
  script_partial_function_application_2,
  "partial_function_application_2.wls"
);
script_test!(
  script_pascals_triangle_puzzle,
  "pascals_triangle_puzzle.wls"
);
script_test!(script_password_generator, "password_generator.wls");
script_test!(script_perfect_numbers, "perfect_numbers.wls");
script_test!(script_perfect_shuffle, "perfect_shuffle.wls");
script_test!(script_permutations, "permutations.wls");
script_test!(script_phrase_reversals, "phrase_reversals.wls");
script_test!(
  script_polynomial_long_division,
  "polynomial_long_division.wls"
);
script_test!(script_polynomial_regression, "polynomial_regression.wls");
script_test!(script_price_fraction, "price_fraction.wls");
script_test!(
  script_primality_by_trial_division,
  "primality_by_trial_division.wls"
);
script_test!(script_problem_of_apollonius, "problem_of_apollonius.wls");
script_test!(script_program_name, "program_name.wls");
script_test!(script_program_termination, "program_termination.wls");
script_test!(script_proper_divisors, "proper_divisors.wls");
script_test!(script_pythagorean_triples, "pythagorean_triples.wls");
script_test!(script_queue_definition, "queue_definition.wls");
script_test!(script_ramseys_theorem, "ramseys_theorem.wls");
script_test!(
  script_random_number_generator_device,
  "random_number_generator_(device).wls"
);
script_test!(script_range_expansion, "range_expansion.wls");
script_test!(script_range_extraction, "range_extraction.wls");
script_test!(
  script_read_a_file_line_by_line,
  "read_a_file_line_by_line.wls"
);
script_test!(
  script_reduced_row_echelon_form,
  "reduced_row_echelon_form.wls"
);
script_test!(script_regular_expressions, "regular_expressions.wls");
script_test!(
  script_remove_duplicate_elements,
  "remove_duplicate_elements.wls"
);
script_test!(
  script_remove_lines_from_a_file,
  "remove_lines_from_a_file.wls"
);
script_test!(script_rep_string, "rep-string.wls");
script_test!(script_repeat_a_string, "repeat_a_string.wls");
script_test!(script_roman_numerals_encode, "roman_numerals_encode.wls");
script_test!(
  script_roots_of_a_quadratic_function,
  "roots_of_a_quadratic_function.wls"
);
script_test!(script_roots_of_unity, "roots_of_unity.wls");
script_test!(script_search_a_list, "search_a_list.wls");
script_test!(
  script_search_a_list_of_records,
  "search_a_list_of_records.wls"
);
script_test!(script_secure_temporary_file, "secure_temporary_file.wls");
script_test!(
  script_self_describing_numbers,
  "self-describing_numbers.wls"
);
script_test!(script_set, "set.wls");
script_test!(script_set_consolidation, "set_consolidation.wls");
script_test!(
  script_seven_sided_dice_from_five_sided_dice,
  "seven-sided_dice_from_five-sided_dice.wls"
);
script_test!(script_shell_one_liner, "shell_one-liner.wls");
script_test!(
  script_shoelace_formula_for_polygonal_area,
  "shoelace_formula_for_polygonal_area.wls"
);
script_test!(script_show_the_epoch, "show_the_epoch.wls");
script_test!(script_sierpinski_carpet, "sierpinski_carpet.wls");
script_test!(script_sierpinski_pentagon, "sierpinski_pentagon.wls");
script_test!(script_sleep, "sleep.wls");
script_test!(script_smith_numbers, "smith_numbers.wls");
script_test!(
  script_sort_an_array_of_composite_structures,
  "sort_an_array_of_composite_structures.wls"
);
script_test!(script_sort_stability, "sort_stability.wls");
script_test!(script_sort_three_variables, "sort_three_variables.wls");
script_test!(
  script_sort_using_a_custom_comparator,
  "sort_using_a_custom_comparator.wls"
);
script_test!(
  script_sorting_algorithms_bead_sort,
  "sorting_algorithms_bead_sort.wls"
);
script_test!(
  script_sorting_algorithms_bubble_sort,
  "sorting_algorithms_bubble_sort.wls"
);
script_test!(
  script_sorting_algorithms_cocktail_sort,
  "sorting_algorithms_cocktail_sort.wls"
);
script_test!(
  script_sorting_algorithms_comb_sort,
  "sorting_algorithms_comb_sort.wls"
);
script_test!(
  script_sorting_algorithms_counting_sort,
  "sorting_algorithms_counting_sort.wls"
);
script_test!(
  script_sorting_algorithms_gnome_sort,
  "sorting_algorithms_gnome_sort.wls"
);
script_test!(
  script_sorting_algorithms_heapsort,
  "sorting_algorithms_heapsort.wls"
);
script_test!(
  script_sorting_algorithms_insertion_sort,
  "sorting_algorithms_insertion_sort.wls"
);
script_test!(
  script_sorting_algorithms_permutation_sort,
  "sorting_algorithms_permutation_sort.wls"
);
script_test!(
  script_sorting_algorithms_quicksort,
  "sorting_algorithms_quicksort.wls"
);
script_test!(
  script_sorting_algorithms_shell_sort,
  "sorting_algorithms_shell_sort.wls"
);
script_test!(
  script_sorting_algorithms_stooge_sort,
  "sorting_algorithms_stooge_sort.wls"
);
script_test!(
  script_sorting_algorithms_strand_sort,
  "sorting_algorithms_strand_sort.wls"
);
script_test!(script_soundex, "soundex.wls");
script_test!(script_sparkline_in_unicode, "sparkline_in_unicode.wls");
script_test!(script_spiral_matrix, "spiral_matrix.wls");
script_test!(
  script_split_a_character_string_based_on_change_of_character,
  "split_a_character_string_based_on_change_of_character.wls"
);
script_test!(
  script_sql_based_authentication,
  "sql-based_authentication.wls"
);
script_test!(script_stair_climbing_puzzle, "stair-climbing_puzzle.wls");
script_test!(script_stem_and_leaf_plot, "stem-and-leaf_plot.wls");
script_test!(script_string_comparison_2, "string_comparison_2.wls");
script_test!(script_string_matching_2, "string_matching_2.wls");
script_test!(script_string_prepend, "string_prepend.wls");
script_test!(
  script_strip_comments_from_a_string,
  "strip_comments_from_a_string.wls"
);
script_test!(
  script_strip_control_codes_and_extended_characters_from_a_string,
  "strip_control_codes_and_extended_characters_from_a_string.wls"
);
script_test!(
  script_strip_whitespace_from_a_string_top_and_tail_2,
  "strip_whitespace_from_a_string_top_and_tail_2.wls"
);
script_test!(script_substring_2, "substring_2.wls");
script_test!(
  script_substring_top_and_tail_2,
  "substring_top_and_tail_2.wls"
);
script_test!(script_subtractive_generator, "subtractive_generator.wls");
script_test!(script_sudoku, "sudoku.wls");
script_test!(
  script_sum_and_product_of_an_array_2,
  "sum_and_product_of_an_array_2.wls"
);
script_test!(
  script_sum_digits_of_an_integer_2,
  "sum_digits_of_an_integer_2.wls"
);
script_test!(script_sum_of_a_series, "sum_of_a_series.wls");
script_test!(script_temperature_conversion, "temperature_conversion.wls");
script_test!(
  script_terminal_control_clear_the_screen,
  "terminal_control_clear_the_screen.wls"
);
script_test!(
  script_terminal_control_cursor_positioning,
  "terminal_control_cursor_positioning.wls"
);
script_test!(
  script_terminal_control_dimensions,
  "terminal_control_dimensions.wls"
);
script_test!(
  script_terminal_control_hiding_the_cursor,
  "terminal_control_hiding_the_cursor.wls"
);
script_test!(
  script_terminal_control_inverse_video,
  "terminal_control_inverse_video.wls"
);
script_test!(
  script_terminal_control_preserve_screen,
  "terminal_control_preserve_screen.wls"
);
script_test!(
  script_trigonometric_functions_2,
  "trigonometric_functions_2.wls"
);
script_test!(script_truncatable_primes, "truncatable_primes.wls");
script_test!(script_truncate_a_file, "truncate_a_file.wls");
script_test!(script_truth_table, "truth_table.wls");
script_test!(
  script_unbias_a_random_generator,
  "unbias_a_random_generator.wls"
);
script_test!(script_unix_ls, "unix_ls.wls");
script_test!(
  script_utf_8_encode_and_decode,
  "utf-8_encode_and_decode.wls"
);
script_test!(
  script_variable_length_quantity,
  "variable-length_quantity.wls"
);
script_test!(script_variable_size_get, "variable_size_get.wls");
script_test!(
  script_verify_distribution_uniformity_chi_squared_test,
  "verify_distribution_uniformity_chi-squared_test.wls"
);
script_test!(
  script_verify_distribution_uniformity_naive,
  "verify_distribution_uniformity_naive.wls"
);
script_test!(script_vigen_c3_a8re_cipher, "vigen%C3%A8re_cipher.wls");
script_test!(
  script_walk_a_directory_non_recursively,
  "walk_a_directory_non-recursively.wls"
);
script_test!(
  script_walk_a_directory_recursively,
  "walk_a_directory_recursively.wls"
);
script_test!(script_write_entire_file, "write_entire_file.wls");
script_test!(
  script_write_float_arrays_to_a_text_file,
  "write_float_arrays_to_a_text_file.wls"
);

// === Slow-but-matching RosettaCode scripts (60s timeout safe) ===
script_test!(script_aks_test_for_primes, "aks_test_for_primes.wls");
script_test!(
  script_arithmetic_geometric_mean_calculate_pi,
  "arithmetic-geometric_mean_calculate_pi.wls"
);
script_test!(script_barnsley_fern, "barnsley_fern.wls");
script_test!(script_fractran, "fractran.wls");
script_test!(
  script_largest_number_divisible_by_its_digits,
  "largest_number_divisible_by_its_digits.wls"
);
script_test!(script_list_comprehensions, "list_comprehensions.wls");
script_test!(script_munching_squares, "munching_squares.wls");
script_test!(
  script_n9_billion_names_of_god_the_integer,
  "9_billion_names_of_god_the_integer.wls"
);
script_test!(
  script_partition_an_integer_x_into_n_primes,
  "partition_an_integer_x_into_n_primes.wls"
);
script_test!(script_probabilistic_choice, "probabilistic_choice.wls");
script_test!(script_square_free_integers, "square-free_integers.wls");
script_test!(
  script_sum_multiples_of_3_and_5,
  "sum_multiples_of_3_and_5.wls"
);
script_test!(script_sum_to_100, "sum_to_100.wls");

// === Additional matching scripts (post-bugfixes) ===
script_test!(script_archimedean_spiral, "archimedean_spiral.wls");
script_test!(
  script_bitmap_bezier_curves_cubic,
  "bitmap_bezier_curves_cubic.wls"
);
script_test!(
  script_bitmap_bezier_curves_quadratic,
  "bitmap_bezier_curves_quadratic.wls"
);
script_test!(script_bitmap_flood_fill, "bitmap_flood_fill.wls");
script_test!(
  script_bitmap_ppm_conversion_through_a_pipe,
  "bitmap_ppm_conversion_through_a_pipe.wls"
);
script_test!(
  script_catalan_numbers_pascals_triangle,
  "catalan_numbers_pascals_triangle.wls"
);
script_test!(script_chaos_game, "chaos_game.wls");
script_test!(
  script_color_of_a_screen_pixel,
  "color_of_a_screen_pixel.wls"
);
script_test!(script_colour_bars_display, "colour_bars_display.wls");
script_test!(
  script_colour_pinstripe_display,
  "colour_pinstripe_display.wls"
);
script_test!(
  script_constrained_random_points_on_a_circle,
  "constrained_random_points_on_a_circle.wls"
);
script_test!(script_death_star, "death_star.wls");
script_test!(script_dragon_curve, "dragon_curve.wls");
script_test!(script_draw_a_cuboid, "draw_a_cuboid.wls");
script_test!(script_draw_a_sphere, "draw_a_sphere.wls");
script_test!(
  script_elementary_cellular_automaton,
  "elementary_cellular_automaton.wls"
);
script_test!(script_evolutionary_algorithm, "evolutionary_algorithm.wls");
script_test!(script_fibonacci_word_fractal, "fibonacci_word_fractal.wls");
script_test!(
  script_magic_squares_of_odd_order,
  "magic_squares_of_odd_order.wls"
);
script_test!(script_morse_code, "morse_code.wls");
script_test!(script_munchausen_numbers, "munchausen_numbers.wls");
script_test!(script_pentagram, "pentagram.wls");
script_test!(script_pinstripe_display, "pinstripe_display.wls");
script_test!(script_plot_coordinate_pairs, "plot_coordinate_pairs.wls");
script_test!(
  script_reverse_words_in_a_string,
  "reverse_words_in_a_string.wls"
);
script_test!(script_semiprime, "semiprime.wls");
script_test!(
  script_sorting_algorithms_radix_sort,
  "sorting_algorithms_radix_sort.wls"
);
script_test!(
  script_sorting_algorithms_selection_sort,
  "sorting_algorithms_selection_sort.wls"
);
script_test!(script_superellipse, "superellipse.wls");
script_test!(
  script_van_der_corput_sequence,
  "van_der_corput_sequence.wls"
);
script_test!(
  script_assigning_values_to_an_array,
  "assigning_values_to_an_array.wls"
);
script_test!(
  script_sequence_of_non_squares,
  "sequence_of_non-squares.wls"
);
script_test!(script_accumulator_factory, "accumulator_factory.wls");
script_test!(script_balanced_ternary, "balanced_ternary.wls");
script_test!(script_cramers_rule, "cramers_rule.wls");
script_test!(
  script_eulers_sum_of_powers_conjecture,
  "eulers_sum_of_powers_conjecture.wls"
);
script_test!(
  script_gui_enabling_disabling_of_controls,
  "gui_enabling_disabling_of_controls.wls"
);
script_test!(script_koch_curve, "koch_curve.wls");
script_test!(script_langtons_ant, "langtons_ant.wls");
script_test!(script_left_factorials, "left_factorials.wls");
script_test!(
  script_longest_string_challenge,
  "longest_string_challenge.wls"
);
script_test!(script_resistor_mesh, "resistor_mesh.wls");
script_test!(script_roots_of_a_function, "roots_of_a_function.wls");
script_test!(script_run_length_encoding, "run-length_encoding.wls");
script_test!(script_xml_dom_serialization, "xml_dom_serialization.wls");
script_test!(script_xml_output, "xml_output.wls");
script_test!(script_zebra_puzzle, "zebra_puzzle.wls");

// === Additional promoted RosettaCode scripts ===
script_test!(script_quine, "quine.wls");
script_test!(script_ackermann_function, "ackermann_function.wls");
script_test!(script_arithmetic_rational, "arithmetic_rational.wls");
script_test!(
  script_catmull_clark_subdivision_surface,
  "catmull%E2%80%93clark_subdivision_surface.wls"
);
script_test!(script_executable_library, "executable_library.wls");
script_test!(script_forward_difference, "forward_difference.wls");
script_test!(script_function_composition, "function_composition.wls");
script_test!(
  script_greatest_subsequential_sum,
  "greatest_subsequential_sum.wls"
);
script_test!(script_hofstadter_q_sequence, "hofstadter_q_sequence.wls");
script_test!(script_knights_tour, "knights_tour.wls");
script_test!(script_long_multiplication, "long_multiplication.wls");
script_test!(script_maze_generation, "maze_generation.wls");
script_test!(script_md5_implementation, "md5_implementation.wls");
script_test!(
  script_miller_rabin_primality_test,
  "miller-rabin_primality_test.wls"
);
script_test!(
  script_real_constants_and_functions,
  "real_constants_and_functions.wls"
);
script_test!(script_sum_of_squares, "sum_of_squares.wls");
script_test!(script_tree_traversal, "tree_traversal.wls");
script_test!(script_url_encoding, "url_encoding.wls");
script_test!(script_vampire_number, "vampire_number.wls");
script_test!(script_zig_zag_matrix, "zig-zag_matrix.wls");

// === Additional promoted _tasks_ RosettaCode scripts ===
script_test!(
  script_calculating_the_value_of_e,
  "calculating_the_value_of_e.wls"
);
script_test!(script_hailstone_sequence, "hailstone_sequence.wls");
script_test!(
  script_interactive_programming,
  "interactive_programming.wls"
);
script_test!(script_mac_vendor_lookup, "mac_vendor_lookup.wls");
script_test!(script_md5, "md5.wls");
script_test!(script_monte_carlo_methods, "monte_carlo_methods.wls");
script_test!(script_primorial_numbers, "primorial_numbers.wls");
script_test!(script_sierpinski_triangle, "sierpinski_triangle.wls");
script_test!(script_time_a_function, "time_a_function.wls");
script_test!(script_variadic_function, "variadic_function.wls");

// === Scripts unlocked by multi-line operator continuation fix ===
script_test!(script_permutation_test, "permutation_test.wls");
script_test!(script_url_decoding, "url_decoding.wls");

// === RosettaCode _tasks_* first-block scripts ===
script_test!(script_n15_puzzle_game, "15_puzzle_game.wls");
script_test!(script_n24_game, "24_game.wls");
script_test!(script_animate_a_pendulum, "animate_a_pendulum.wls");
script_test!(
  script_arbitrary_precision_integers_included,
  "arbitrary-precision_integers_(included).wls"
);
script_test!(script_arithmetic_integer, "arithmetic_integer.wls");
script_test!(script_bitmap, "bitmap.wls");
script_test!(script_cholesky_decomposition, "cholesky_decomposition.wls");
script_test!(script_deconvolution_2d_, "deconvolution_2d_.wls");
script_test!(
  script_distributed_programming,
  "distributed_programming.wls"
);
script_test!(script_draw_a_clock, "draw_a_clock.wls");
script_test!(script_draw_a_rotating_cube, "draw_a_rotating_cube.wls");
script_test!(script_farey_sequence, "farey_sequence.wls");
script_test!(
  script_get_system_command_output,
  "get_system_command_output.wls"
);
script_test!(
  script_hello_world_standard_error,
  "hello_world_standard_error.wls"
);
script_test!(script_hilbert_curve, "hilbert_curve.wls");
script_test!(script_honeycombs, "honeycombs.wls");
script_test!(script_https, "https.wls");
script_test!(
  script_https_client_authenticated,
  "https_client-authenticated.wls"
);
script_test!(script_image_noise, "image_noise.wls");
script_test!(script_integer_comparison, "integer_comparison.wls");
script_test!(script_joystick_position, "joystick_position.wls");
script_test!(script_julia_set, "julia_set.wls");
script_test!(
  script_longest_common_subsequence,
  "longest_common_subsequence.wls"
);
script_test!(script_maze_solving, "maze_solving.wls");
script_test!(script_musical_scale, "musical_scale.wls");
script_test!(
  script_pascal_matrix_generation,
  "pascal_matrix_generation.wls"
);
script_test!(script_record_sound, "record_sound.wls");
script_test!(
  script_retrieve_and_search_chat_history,
  "retrieve_and_search_chat_history.wls"
);
script_test!(script_roman_numerals_decode, "roman_numerals_decode.wls");
script_test!(script_send_email, "send_email.wls");
script_test!(
  script_sutherland_hodgman_polygon_clipping,
  "sutherland-hodgman_polygon_clipping.wls"
);
script_test!(script_topological_sort, "topological_sort.wls");
script_test!(script_total_circles_area, "total_circles_area.wls");
script_test!(script_url_parser, "url_parser.wls");

// === Scripts unlocked by ValueQ-on-FunctionCall fix ===
script_test!(script_abstract_type, "abstract_type.wls");

// === Scripts unlocked by Do[i, {.., n/2}] rational iterator bounds +
//     trailing-`!`-as-prefix-Not multi-line continuation +
//     AppendTo[x[[i]], v] Part-target mutation ===
script_test!(script_lychrel_numbers, "lychrel_numbers.wls");
script_test!(script_sieve_of_eratosthenes, "sieve_of_eratosthenes.wls");
script_test!(
  script_digital_root_multiplicative_digital_root,
  "digital_root_multiplicative_digital_root.wls"
);

// === Multi-iterator Do propagates Break/Return through all levels ===
script_test!(script_loops_nested, "loops_nested.wls");
// === Implicit-multiplication of factorials: `a! b!` parses as
//     `Times[Factorial[a], Factorial[b]]` ===
script_test!(script_catalan_numbers, "catalan_numbers.wls");
script_test!(script_modular_exponentiation, "modular_exponentiation.wls");
script_test!(
  script_horners_rule_for_polynomial_evaluation,
  "horners_rule_for_polynomial_evaluation.wls"
);
script_test!(script_kaprekar_numbers, "kaprekar_numbers.wls");

script_test!(
  script_convert_decimal_number_to_rational,
  "convert_decimal_number_to_rational.wls"
);

script_test!(
  script_terminal_control_unicode_output,
  "terminal_control_unicode_output.wls"
);

// === Unlocked by Part-assignment descending into Associations ===
script_test!(script_deepcopy, "deepcopy.wls");

// === Unlocked by script-mode Column rendering as `Column[{…}]` ===
script_test!(script_twelve_statements, "twelve_statements.wls");

// === Unlocked by the FindSequenceFunction operator form (`f[seq][k]`) ===
script_test!(script_floyds_triangle, "floyds_triangle.wls");
// === Unlocked by WriteString[$Output, …] writing to captured stdout ===
script_test!(
  script_hello_world_newline_omission,
  "hello_world_newline_omission.wls"
);
// === Unlocked by the DayRange weekday filter + DatePlus multi-unit fix
//     (uses a warm-up shim to swallow wolframscript's CalendarData init Print) ===
script_test!(
  script_find_the_last_sunday_of_each_month,
  "find_the_last_sunday_of_each_month.wls"
);
// === Unlocked by RealDigits padding exact values with 0 (not Indeterminate) ===
script_test!(
  script_decimal_floating_point_number_to_binary,
  "decimal_floating_point_number_to_binary.wls"
);
// === Unlocked by named-character rendering (\[Union]/\[Intersection]/\[Minus]) ===
script_test!(script_set_of_real_numbers, "set_of_real_numbers.wls");
// === Curated to the Permutations/Select solution (avoids Reduce over Integers) ===
script_test!(
  script_dinesmans_multiple_dwelling_problem,
  "dinesmans_multiple-dwelling_problem.wls"
);
// === Unlocked by string patterns matching across newlines (dotall) ===
script_test!(script_strip_block_comments, "strip_block_comments.wls");
// === Unlocked by Map of a named Function + Part-result application a[[i]][] ===
script_test!(script_closures_value_capture, "closures_value_capture.wls");
// === Unlocked by pure-function `?test` conditions firing in DownValue dispatch ===
script_test!(script_sedols, "sedols.wls");
// === Unlocked by FindShortestPath (Dijkstra on a weighted Graph) ===
script_test!(script_dijkstras_algorithm, "dijkstras_algorithm.wls");
// === Unlocked by empty Span parts (a[[n;;]] / a[[;;-2]] beyond length → {}) ===
script_test!(
  script_parsing_shunting_yard_algorithm,
  "parsing_shunting-yard_algorithm.wls"
);

// === RosettaCode tasks curated from the _tasks_ source dumps into a single
//     runnable solution (output verified identical to wolframscript) ===
script_test!(script_bitwise_operations, "bitwise_operations.wls");
script_test!(script_compound_data_type, "compound_data_type.wls");
script_test!(script_conjugate_transpose, "conjugate_transpose.wls");
script_test!(script_literals_integer, "literals_integer.wls");
script_test!(script_sort_disjoint_sublist, "sort_disjoint_sublist.wls");
script_test!(
  script_sorting_algorithms_merge_sort,
  "sorting_algorithms_merge_sort.wls"
);

// === Unlocked by StringJoin[] -> "" (empty StringJoin @@ {} folds cleanly) ===
script_test!(
  script_old_lady_swallowed_a_fly,
  "old_lady_swallowed_a_fly.wls"
);

script_test!(
  script_case_sensitivity_of_identifiers,
  "case-sensitivity_of_identifiers.wls"
);

script_test!(
  script_strip_a_set_of_characters_from_a_string,
  "strip_a_set_of_characters_from_a_string.wls"
);

script_test!(
  script_find_the_missing_permutation,
  "find_the_missing_permutation.wls"
);

script_test!(script_lucas_lehmer_test, "lucas-lehmer_test.wls");

script_test!(script_named_parameters, "named_parameters.wls");

script_test!(
  script_combinations_with_repetitions,
  "combinations_with_repetitions.wls"
);

script_test!(script_enumerations, "enumerations.wls");

script_test!(script_optional_parameters, "optional_parameters.wls");

script_test!(script_qr_decomposition, "qr_decomposition.wls");

script_test!(script_arithmetic_complex, "arithmetic_complex.wls");

script_test!(script_collections, "collections.wls");

script_test!(script_ludic_numbers, "ludic_numbers.wls");

script_test!(script_undefined_values, "undefined_values.wls");

// === Unlocked by Set parser/storage fixes ===
// - `lhs = body & [args]`: bracket args now belong INSIDE the assignment's RHS
//   instead of being wrapped around the entire Set node.
// - `lhs = "a" -> "n"`: stored Raw values like `"a" -> "n"` are no longer
//   misread back as a single quoted string literal.
script_test!(script_rot_13, "rot-13.wls");

script_test!(
  script_find_common_directory_path,
  "find_common_directory_path.wls"
);

script_test!(script_infinity, "infinity.wls");

script_test!(script_exceptions, "exceptions.wls");

script_test!(script_bernoulli_numbers, "bernoulli_numbers.wls");

script_test!(script_topswops, "topswops.wls");

// === Manually-fixed RC scripts (source had pre-existing bugs) ===
// Each of these had a real bug in the RC task source — undefined symbol,
// protected-name override, stray glyph, missing `Push` head — that errored
// even under wolframscript. Promoted versions correct the typo while
// preserving the original algorithm.
script_test!(script_taxicab_numbers, "taxicab_numbers.wls");
script_test!(
  script_compare_a_list_of_strings,
  "compare_a_list_of_strings.wls"
);
script_test!(
  script_apply_a_callback_to_an_array,
  "apply_a_callback_to_an_array.wls"
);
script_test!(script_null_object, "null_object.wls");
script_test!(script_documentation, "documentation.wls");
script_test!(script_symmetric_difference, "symmetric_difference.wls");
script_test!(script_hash_from_two_arrays, "hash_from_two_arrays.wls");
script_test!(script_stack, "stack.wls");
script_test!(script_five_weekends, "five_weekends.wls");
script_test!(
  script_determinant_and_permanent,
  "determinant_and_permanent.wls"
);
script_test!(
  script_element_wise_operations,
  "element-wise_operations.wls"
);
script_test!(script_runtime_evaluation, "runtime_evaluation.wls");
script_test!(
  script_runtime_evaluation_in_an_environment,
  "runtime_evaluation_in_an_environment.wls"
);
script_test!(script_logical_operations, "logical_operations.wls");

script_test!(script_prime_decomposition, "prime_decomposition.wls");

script_test!(
  script_linear_congruential_generator,
  "linear_congruential_generator.wls"
);

script_test!(
  script_luhn_test_of_credit_card_numbers,
  "luhn_test_of_credit_card_numbers.wls"
);

#[test]
fn script_cli_args() {
  run_script_snapshot_with_args("cli_args.wls", &["5"]);
}
