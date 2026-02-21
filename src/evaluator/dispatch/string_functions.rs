#[allow(unused_imports)]
use super::*;

pub fn dispatch_string_functions(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "StringLength" if args.len() == 1 => {
      return Some(crate::functions::string_ast::string_length_ast(args));
    }
    "StringTake" if args.len() == 2 => {
      return Some(crate::functions::string_ast::string_take_ast(args));
    }
    "StringDrop" if args.len() == 2 => {
      return Some(crate::functions::string_ast::string_drop_ast(args));
    }
    "Compress" if args.len() == 1 => {
      return Some(crate::functions::string_ast::compress_ast(args));
    }
    "Uncompress" if args.len() == 1 => {
      return Some(crate::functions::string_ast::uncompress_ast(args));
    }
    "StringJoin" => {
      return Some(crate::functions::string_ast::string_join_ast(args));
    }
    "StringSplit" if !args.is_empty() => {
      return Some(crate::functions::string_ast::string_split_ast(args));
    }
    "StringStartsQ" if args.len() == 2 => {
      return Some(crate::functions::string_ast::string_starts_q_ast(args));
    }
    "StringEndsQ" if args.len() == 2 => {
      return Some(crate::functions::string_ast::string_ends_q_ast(args));
    }
    "StringContainsQ" if args.len() == 2 => {
      return Some(crate::functions::string_ast::string_contains_q_ast(args));
    }
    "StringReplace" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::string_ast::string_replace_ast(args));
    }
    "ToUpperCase" if args.len() == 1 => {
      return Some(crate::functions::string_ast::to_upper_case_ast(args));
    }
    "ToLowerCase" if args.len() == 1 => {
      return Some(crate::functions::string_ast::to_lower_case_ast(args));
    }
    "Characters" if args.len() == 1 => {
      return Some(crate::functions::string_ast::characters_ast(args));
    }
    "StringRiffle" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::string_ast::string_riffle_ast(args));
    }
    "StringPosition" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::string_ast::string_position_ast(args));
    }
    "StringMatchQ" if args.len() == 2 => {
      return Some(crate::functions::string_ast::string_match_q_ast(args));
    }
    "StringReverse" if args.len() == 1 => {
      return Some(crate::functions::string_ast::string_reverse_ast(args));
    }
    "StringRepeat" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::string_ast::string_repeat_ast(args));
    }
    "StringTrim" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::string_ast::string_trim_ast(args));
    }
    "StringCases" if args.len() == 2 => {
      return Some(crate::functions::string_ast::string_cases_ast(args));
    }
    "ToString" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::string_ast::to_string_ast(args));
    }
    "ToExpression" if args.len() == 1 => {
      return Some(crate::functions::string_ast::to_expression_ast(args));
    }
    "StringPadLeft" if args.len() >= 2 && args.len() <= 3 => {
      return Some(crate::functions::string_ast::string_pad_left_ast(args));
    }
    "StringPadRight" if args.len() >= 2 && args.len() <= 3 => {
      return Some(crate::functions::string_ast::string_pad_right_ast(args));
    }
    "EditDistance" if args.len() == 2 => {
      return Some(crate::functions::string_ast::edit_distance_ast(args));
    }
    "LongestCommonSubsequence" if args.len() == 2 => {
      return Some(
        crate::functions::string_ast::longest_common_subsequence_ast(args),
      );
    }
    "StringCount" if args.len() == 2 => {
      return Some(crate::functions::string_ast::string_count_ast(args));
    }
    "StringFreeQ" if args.len() == 2 => {
      return Some(crate::functions::string_ast::string_free_q_ast(args));
    }
    "ToCharacterCode" if args.len() == 1 => {
      return Some(crate::functions::string_ast::to_character_code_ast(args));
    }
    "FromCharacterCode" if args.len() == 1 => {
      return Some(crate::functions::string_ast::from_character_code_ast(args));
    }
    "CharacterRange" if args.len() == 2 => {
      return Some(crate::functions::string_ast::character_range_ast(args));
    }
    "IntegerString" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::string_ast::integer_string_ast(args));
    }
    "Alphabet" if args.is_empty() => {
      return Some(crate::functions::string_ast::alphabet_ast(args));
    }
    "LetterQ" if args.len() == 1 => {
      return Some(crate::functions::string_ast::letter_q_ast(args));
    }
    "UpperCaseQ" if args.len() == 1 => {
      return Some(crate::functions::string_ast::upper_case_q_ast(args));
    }
    "LowerCaseQ" if args.len() == 1 => {
      return Some(crate::functions::string_ast::lower_case_q_ast(args));
    }
    "DigitQ" if args.len() == 1 => {
      return Some(crate::functions::string_ast::digit_q_ast(args));
    }
    "StringInsert" if args.len() == 3 => {
      return Some(crate::functions::string_ast::string_insert_ast(args));
    }
    "StringDelete" if args.len() == 2 => {
      return Some(crate::functions::string_ast::string_delete_ast(args));
    }
    "Capitalize" if args.len() == 1 => {
      return Some(crate::functions::string_ast::capitalize_ast(args));
    }
    "Decapitalize" if args.len() == 1 => {
      return Some(crate::functions::string_ast::decapitalize_ast(args));
    }
    "StringPart" if args.len() == 2 => {
      return Some(crate::functions::string_ast::string_part_ast(args));
    }
    "StringTakeDrop" if args.len() == 2 => {
      return Some(crate::functions::string_ast::string_take_drop_ast(args));
    }
    "HammingDistance" if args.len() == 2 => {
      return Some(crate::functions::string_ast::hamming_distance_ast(args));
    }
    "CharacterCounts" if args.len() == 1 => {
      return Some(crate::functions::string_ast::character_counts_ast(args));
    }
    "RemoveDiacritics" if args.len() == 1 => {
      return Some(crate::functions::string_ast::remove_diacritics_ast(args));
    }
    "StringRotateLeft" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::string_ast::string_rotate_left_ast(args));
    }
    "StringRotateRight" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::string_ast::string_rotate_right_ast(args));
    }
    "AlphabeticSort" if args.len() == 1 => {
      return Some(crate::functions::string_ast::alphabetic_sort_ast(args));
    }
    "Hash" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::string_ast::hash_ast(args));
    }
    _ => {}
  }
  None
}
