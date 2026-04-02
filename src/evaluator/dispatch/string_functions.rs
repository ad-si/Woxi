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
    "TemplateApply" if args.len() == 2 => {
      return Some(crate::functions::string_ast::template_apply_ast(args));
    }
    "StringSplit" if !args.is_empty() => {
      return Some(crate::functions::string_ast::string_split_ast(args));
    }
    "StringStartsQ" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::string_ast::string_starts_q_ast(args));
    }
    "StringEndsQ" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::string_ast::string_ends_q_ast(args));
    }
    "StringContainsQ" if args.len() == 2 || args.len() == 3 => {
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
    "StringMatchQ" if args.len() == 2 || args.len() == 3 => {
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
    "ToString" | "TextString" if args.len() == 1 || args.len() == 2 => {
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
    "SequenceAlignment" if args.len() == 2 => {
      return Some(crate::functions::string_ast::sequence_alignment_ast(args));
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
    "FromLetterNumber" if args.len() == 1 => {
      return Some(crate::functions::string_ast::from_letter_number_ast(args));
    }
    "LetterNumber" if args.len() == 1 => {
      return Some(crate::functions::string_ast::letter_number_ast(args));
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
    "DictionaryWordQ" if args.len() == 1 => {
      return Some(crate::functions::string_ast::dictionary_word_q_ast(args));
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
    "StringPartition" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::string_ast::string_partition_ast(args));
    }
    "Hash" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::string_ast::hash_ast(args));
    }
    "SyntaxQ" if args.len() == 1 => {
      if let Expr::String(s) = &args[0] {
        let is_valid = crate::parse(s).is_ok();
        return Some(Ok(Expr::Identifier(
          if is_valid { "True" } else { "False" }.to_string(),
        )));
      }
    }
    "LetterCounts" if args.len() == 1 => {
      if let Expr::String(s) = &args[0] {
        // Count letters (alphabetic only), sorted by count descending then last position descending
        let mut counts: Vec<(char, i128, usize)> = Vec::new();
        for (pos, ch) in s.chars().enumerate() {
          if ch.is_alphabetic() {
            if let Some(entry) = counts.iter_mut().find(|(c, _, _)| *c == ch) {
              entry.1 += 1;
              entry.2 = pos;
            } else {
              counts.push((ch, 1, pos));
            }
          }
        }
        // Sort by count descending, then by last position descending
        counts.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)));
        let pairs: Vec<(Expr, Expr)> = counts
          .into_iter()
          .map(|(ch, count, _)| {
            (Expr::String(ch.to_string()), Expr::Integer(count))
          })
          .collect();
        return Some(Ok(Expr::Association(pairs)));
      }
    }
    "TextWords" if args.len() == 1 => {
      if let Expr::String(s) = &args[0] {
        // Split into words, stripping punctuation from each word
        let words: Vec<Expr> = s
          .split_whitespace()
          .map(|w| {
            let trimmed: String =
              w.chars().filter(|c| c.is_alphanumeric()).collect();
            trimmed
          })
          .filter(|w| !w.is_empty())
          .map(|w| Expr::String(w))
          .collect();
        return Some(Ok(Expr::List(words)));
      }
    }
    "WordCounts" if args.len() == 1 => {
      if let Expr::String(s) = &args[0] {
        let mut counts: Vec<(String, i128, usize)> = Vec::new();
        for (pos, word) in s.split_whitespace().enumerate() {
          if let Some(entry) = counts.iter_mut().find(|(w, _, _)| w == word) {
            entry.1 += 1;
            entry.2 = pos;
          } else {
            counts.push((word.to_string(), 1, pos));
          }
        }
        // Sort by count descending, then by last position descending
        counts.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)));
        let pairs: Vec<(Expr, Expr)> = counts
          .into_iter()
          .map(|(word, count, _)| (Expr::String(word), Expr::Integer(count)))
          .collect();
        return Some(Ok(Expr::Association(pairs)));
      }
    }
    "NumericalSort" if args.len() == 1 => {
      if let Expr::List(ref elems) = args[0] {
        let mut items: Vec<Expr> = elems.clone();
        items.sort_by(|a, b| {
          // NumericalSort sorts by numerical value for numbers,
          // and lexicographically for strings (not natural sort).
          let sa = crate::syntax::expr_to_string(a);
          let sb = crate::syntax::expr_to_string(b);
          sa.cmp(&sb)
        });
        return Some(Ok(Expr::List(items)));
      }
    }
    "StringReplaceList" if args.len() == 2 => {
      // StringReplaceList["string", "pattern" -> "replacement"]
      // Returns list of strings, each with one occurrence replaced
      if let Expr::String(s) = &args[0]
        && let Expr::Rule {
          pattern,
          replacement,
        } = &args[1]
        && let (Expr::String(pat), Expr::String(rep)) =
          (pattern.as_ref(), replacement.as_ref())
      {
        let mut results = Vec::new();
        let pat_len = pat.len();
        if pat_len > 0 {
          let s_bytes = s.as_bytes();
          let pat_bytes = pat.as_bytes();
          for i in 0..=s.len().saturating_sub(pat_len) {
            if &s_bytes[i..i + pat_len] == pat_bytes {
              let mut result = String::new();
              result.push_str(&s[..i]);
              result.push_str(rep);
              result.push_str(&s[i + pat_len..]);
              results.push(Expr::String(result));
            }
          }
        }
        return Some(Ok(Expr::List(results)));
      }
    }
    // WordCount["string"] — count words in a string
    "WordCount" if args.len() == 1 => {
      if let Expr::String(s) = &args[0] {
        let count = s.split_whitespace().count();
        return Some(Ok(Expr::Integer(count as i128)));
      }
    }
    "URLEncode" if args.len() == 1 => {
      return Some(crate::functions::string_ast::url_encode_ast(args));
    }
    "URLDecode" if args.len() == 1 => {
      return Some(crate::functions::string_ast::url_decode_ast(args));
    }
    "StringToByteArray" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::string_ast::string_to_byte_array_ast(
        args,
      ));
    }
    "ByteArrayToString" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::string_ast::byte_array_to_string_ast(
        args,
      ));
    }
    _ => {}
  }
  None
}
