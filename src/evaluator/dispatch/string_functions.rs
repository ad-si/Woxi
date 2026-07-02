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
    "CompressedData" => {
      return Some(crate::functions::string_ast::compressed_data_ast(args));
    }
    "StringJoin" => {
      return Some(crate::functions::string_ast::string_join_ast(args));
    }
    "TemplateApply" if args.len() == 2 => {
      return Some(crate::functions::string_ast::template_apply_ast(args));
    }
    "StringExtract" if args.len() >= 2 => {
      return Some(crate::functions::string_ast::string_extract_ast(args));
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
    "StringReplace" if (2..=4).contains(&args.len()) => {
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
    "StringRiffle" if !args.is_empty() => {
      return Some(crate::functions::string_ast::string_riffle_ast(args));
    }
    "StringPosition" if args.len() >= 2 => {
      // Wolfram emits StringPosition::strse and returns the call
      // unevaluated whenever the first argument isn't a String or list
      // of Strings (e.g. an unbound symbol in a script). Mirror that
      // here so we don't silently coerce identifiers to their name and
      // return `{}`.
      fn is_valid_string_arg(e: &Expr) -> bool {
        match e {
          Expr::String(_) => true,
          Expr::List(items) => {
            items.iter().all(|it| matches!(it, Expr::String(_)))
          }
          _ => false,
        }
      }
      if !is_valid_string_arg(&args[0]) {
        let arg_strs: Vec<String> =
          args.iter().map(crate::syntax::expr_to_output).collect();
        crate::emit_message(&format!(
          "StringPosition::strse: A string or list of strings is expected at position 1 in StringPosition[{}].",
          arg_strs.join(", "),
        ));
        return Some(Ok(Expr::FunctionCall {
          name: "StringPosition".to_string(),
          args: args.to_vec().into(),
        }));
      }
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
    "StringCases" if args.len() >= 2 => {
      return Some(crate::functions::string_ast::string_cases_ast(args));
    }
    "ToString" | "TextString" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::string_ast::to_string_ast(args));
    }
    // Display wrapper: stays unevaluated in script-mode echo (rendering
    // happens through ToString), without the not-yet-implemented warning
    "PaddedForm" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "PaddedForm".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "ToExpression" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::string_ast::to_expression_ast(args));
    }
    "StringPadLeft" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::string_ast::string_pad_left_ast(args));
    }
    "StringPadRight" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::string_ast::string_pad_right_ast(args));
    }
    "InsertLinebreaks" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::string_ast::insert_linebreaks_ast(args));
    }
    "EditDistance" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::string_ast::edit_distance_ast(args));
    }
    "DamerauLevenshteinDistance" if args.len() == 2 || args.len() == 3 => {
      return Some(
        crate::functions::string_ast::damerau_levenshtein_distance_ast(args),
      );
    }
    "NeedlemanWunschSimilarity" if args.len() == 2 => {
      return Some(
        crate::functions::string_ast::needleman_wunsch_similarity_ast(args),
      );
    }
    "SmithWatermanSimilarity" if args.len() == 2 => {
      return Some(
        crate::functions::string_ast::smith_waterman_similarity_ast(args),
      );
    }
    "SequenceAlignment" if args.len() == 2 => {
      return Some(crate::functions::string_ast::sequence_alignment_ast(args));
    }
    "LongestCommonSubsequence" if args.len() == 2 => {
      return Some(
        crate::functions::string_ast::longest_common_subsequence_ast(args),
      );
    }
    "LongestCommonSequence" if args.len() == 2 => {
      return Some(crate::functions::string_ast::longest_common_sequence_ast(
        args,
      ));
    }
    "LongestCommonSubsequencePositions" if args.len() == 2 => {
      return Some(
        crate::functions::string_ast::longest_common_subsequence_positions_ast(
          args,
        ),
      );
    }
    "StringCount" if args.len() >= 2 => {
      return Some(crate::functions::string_ast::string_count_ast(args));
    }
    "StringFreeQ" if args.len() == 1 => {
      // Operator form: StringFreeQ[pattern] is a curried predicate that will
      // be applied to a string via the outer call (see function_application.rs).
      return Some(Ok(Expr::FunctionCall {
        name: "StringFreeQ".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "StringFreeQ" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::string_ast::string_free_q_ast(args));
    }
    "ToCharacterCode" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::string_ast::to_character_code_ast(args));
    }
    "FromCharacterCode" if args.len() == 1 || args.len() == 2 => {
      // FromCharacterCode accepts an optional CharacterEncoding string as a
      // second argument. The codepoints themselves are already Unicode, so
      // the encoding is informational for ASCII-compatible encodings like
      // "ISO8859-1" and "UTF-8" — we simply pass through and let the core
      // routine build the string from codepoints.
      // Validate args[0] up front — wolframscript emits
      // FromCharacterCode::intnm and returns the unevaluated form for any
      // non-Integer / non-list-of-Integers input. Mirror that so script
      // sequences using `%` (which resolves to Out[0] when nothing has
      // been printed) stay alive instead of aborting with a hard error.
      fn is_valid_codepoint_arg(e: &Expr) -> bool {
        match e {
          Expr::Integer(_) | Expr::BigInteger(_) => true,
          Expr::List(items) => items.iter().all(|it| {
            matches!(it, Expr::Integer(_) | Expr::BigInteger(_))
              || matches!(it, Expr::List(inner)
                if inner.iter().all(|x| matches!(x, Expr::Integer(_) | Expr::BigInteger(_))))
          }),
          _ => false,
        }
      }
      if !is_valid_codepoint_arg(&args[0]) {
        crate::emit_message(&format!(
          "FromCharacterCode::intnm: Non-negative machine-sized integer expected at position 1 in FromCharacterCode[{}].",
          crate::syntax::expr_to_string(&args[0]),
        ));
        return Some(Ok(Expr::FunctionCall {
          name: "FromCharacterCode".to_string(),
          args: args.to_vec().into(),
        }));
      }
      // With a "UTF8"/"UTF-8" encoding the integers are UTF-8 *bytes*, not
      // code points: decode the byte sequence into characters. (Other
      // ASCII-compatible encodings pass through unchanged below.)
      if args.len() == 2
        && let Expr::String(enc) = &args[1]
        && (enc == "UTF8" || enc == "UTF-8")
      {
        let bytes: Option<Vec<u8>> = match &args[0] {
          Expr::Integer(n) if (0..=255).contains(n) => Some(vec![*n as u8]),
          Expr::List(items) => items
            .iter()
            .map(|it| match it {
              Expr::Integer(n) if (0..=255).contains(n) => Some(*n as u8),
              _ => None,
            })
            .collect(),
          _ => None,
        };
        if let Some(bytes) = bytes {
          match String::from_utf8(bytes) {
            Ok(s) => return Some(Ok(Expr::String(s))),
            Err(_) => {
              // Invalid byte sequence: wolframscript warns, then falls back
              // to interpreting the integers as code points.
              crate::emit_message(&format!(
                "$CharacterEncoding::utf8: The byte sequence {} could not be \
                 interpreted as a character in the UTF-8 character encoding.",
                crate::syntax::format_expr(
                  &args[0],
                  crate::syntax::ExprForm::Output
                )
              ));
            }
          }
        }
      }
      return Some(crate::functions::string_ast::from_character_code_ast(
        &args[0..1],
      ));
    }
    "CharacterRange" if args.len() == 2 => {
      return Some(crate::functions::string_ast::character_range_ast(args));
    }
    "IntegerString" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::string_ast::integer_string_ast(args));
    }
    "Alphabet" if args.len() <= 1 => {
      return Some(crate::functions::string_ast::alphabet_ast(args));
    }
    "FromLetterNumber" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::string_ast::from_letter_number_ast(args));
    }
    "LetterNumber" if args.len() == 1 || args.len() == 2 => {
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
    "PrintableASCIIQ" if args.len() == 1 => {
      return Some(crate::functions::string_ast::printable_ascii_q_ast(args));
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
    "StringReplacePart" if args.len() == 3 => {
      return Some(crate::functions::string_ast::string_replace_part_ast(args));
    }
    "StringDelete" if args.len() >= 2 => {
      return Some(crate::functions::string_ast::string_delete_ast(args));
    }
    "Capitalize" if args.len() == 1 || args.len() == 2 => {
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
    "HammingDistance" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::string_ast::hamming_distance_ast(args));
    }
    "CharacterCounts" if args.len() == 1 => {
      return Some(crate::functions::string_ast::character_counts_ast(args));
    }
    "CharacterCounts" if args.len() == 2 => {
      return Some(crate::functions::string_ast::character_counts_ngram_ast(
        args,
      ));
    }
    "RemoveDiacritics" if args.len() == 1 => {
      return Some(crate::functions::string_ast::remove_diacritics_ast(args));
    }
    // The 2-argument target-script forms are not implemented yet; they
    // fall through and stay unevaluated.
    "Transliterate" if args.len() == 1 => {
      return Some(crate::functions::transliterate_ast::transliterate_ast(
        args,
      ));
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
      return Some(crate::functions::string_ast::letter_counts_ast(args));
    }
    "LetterCounts" if args.len() == 2 => {
      return Some(crate::functions::string_ast::letter_counts_ngram_ast(args));
    }
    "TextSentences" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::string_ast::text_sentences_ast(args));
    }
    "TextWords" if args.len() == 1 || args.len() == 2 => {
      if let Expr::String(s) = &args[0] {
        // Tokenize like WordCounts: surrounding punctuation is trimmed but
        // internal hyphens/apostrophes are kept ("YT-1300", "don't").
        let mut words = crate::functions::string_ast::text_word_tokens(s);
        // TextWords[s, n] returns the first n words (n must be positive).
        if args.len() == 2 {
          match &args[1] {
            Expr::Integer(n) if *n >= 1 => {
              words.truncate(*n as usize);
            }
            _ => {
              return Some(Ok(Expr::FunctionCall {
                name: "TextWords".to_string(),
                args: args.to_vec().into(),
              }));
            }
          }
        }
        let words: Vec<Expr> = words.into_iter().map(Expr::String).collect();
        return Some(Ok(Expr::List(words.into())));
      }
    }
    "WordCounts" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::string_ast::word_counts_ast(args));
    }
    "WordFrequency" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::string_ast::word_frequency_ast(args));
    }
    "NumericalSort" if args.len() == 1 => {
      if let Expr::List(ref elems) = args[0] {
        // The real value of an expression, evaluating constants and exact
        // forms (Pi, Sqrt[2], 1/3, …) via N; None for non-numeric atoms.
        fn num_val(e: &Expr) -> Option<f64> {
          if let Some(v) = crate::functions::math_ast::expr_to_f64(e) {
            return Some(v);
          }
          let n =
            crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
              name: "N".to_string(),
              args: vec![e.clone()].into(),
            })
            .ok()?;
          crate::functions::math_ast::expr_to_f64(&n)
        }
        // Compare by numeric value; vectors compare component-wise. Non-numeric
        // elements (strings, free symbols) fall back to canonical order.
        fn cmp(a: &Expr, b: &Expr) -> std::cmp::Ordering {
          use std::cmp::Ordering;
          if let (Expr::List(la), Expr::List(lb)) = (a, b) {
            for (x, y) in la.iter().zip(lb.iter()) {
              let c = cmp(x, y);
              if c != Ordering::Equal {
                return c;
              }
            }
            return la.len().cmp(&lb.len());
          }
          match (num_val(a), num_val(b)) {
            (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(Ordering::Equal),
            _ => crate::syntax::expr_to_string(a)
              .cmp(&crate::syntax::expr_to_string(b)),
          }
        }
        let mut items: Vec<Expr> = elems.to_vec();
        items.sort_by(cmp);
        return Some(Ok(Expr::List(items.into())));
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
        return Some(Ok(Expr::List(results.into())));
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
    "BaseEncode" if args.len() == 1 => {
      return Some(crate::functions::string_ast::base_encode_ast(args));
    }
    "BaseDecode" if args.len() == 1 => {
      return Some(crate::functions::string_ast::base_decode_ast(args));
    }
    _ => {}
  }
  None
}
