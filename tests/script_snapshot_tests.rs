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

#[test]
fn script_cli_args() {
  run_script_snapshot_with_args("cli_args.wls", &["5"]);
}
