// String and text options / argument forms found missing by the
// reference-page parameter sweep (round 3).

use super::*;

mod alphabetic_sort {
  use super::*;

  #[test]
  fn lower_case_sorts_before_upper_case() {
    assert_eq!(
      interpret("AlphabeticSort[{\"b\", \"A\", \"a\", \"B\"}]").unwrap(),
      "{a, A, b, B}"
    );
  }

  #[test]
  fn swedish_letters_sort_after_z() {
    assert_eq!(
      interpret(
        "AlphabeticSort[{\"Encyclopædia\", \"Encyclopedia\", \
         \"Encyclopaedia\"}, \"Swedish\"]"
      )
      .unwrap(),
      "{Encyclopaedia, Encyclopedia, Encyclopædia}"
    );
    assert_eq!(
      interpret(
        "AlphabeticSort[{\"äpple\", \"zebra\", \"apa\", \"öl\", \"ål\"}, \
         \"Swedish\"]"
      )
      .unwrap(),
      "{apa, zebra, ål, äpple, öl}"
    );
  }

  #[test]
  fn spanish_enye_is_a_letter_of_its_own() {
    assert_eq!(
      interpret("AlphabeticSort[{\"oso\", \"ñu\", \"nube\"}, \"Spanish\"]")
        .unwrap(),
      "{nube, ñu, oso}"
    );
  }

  #[test]
  fn non_list_stays_unevaluated() {
    assert_eq!(
      interpret("AlphabeticSort[undefinedSymbol]").unwrap(),
      "AlphabeticSort[undefinedSymbol]"
    );
  }
}

mod longest_ordered_sequence {
  use super::*;

  #[test]
  fn ignore_case_compares_in_lower_case() {
    assert_eq!(
      interpret(
        "LongestOrderedSequence[\"WolframLanguage\", IgnoreCase -> True] === \
         LongestOrderedSequence[ToLowerCase[\"WolframLanguage\"]]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn ignore_case_keeps_the_original_characters() {
    // Folded "dabc" -> "abc", taken from the string as written.
    assert_eq!(
      interpret("LongestOrderedSequence[\"dAbC\", IgnoreCase -> True]")
        .unwrap(),
      "AbC"
    );
    assert_eq!(
      interpret("LongestOrderedSequence[\"dAbC\", IgnoreCase -> False]")
        .unwrap(),
      interpret("LongestOrderedSequence[\"dAbC\"]").unwrap()
    );
  }
}

mod palindrome_q {
  use super::*;

  #[test]
  fn diacritics_depend_on_the_language() {
    assert_eq!(
      interpret(
        "PalindromeQ[\"añana\", IgnoreDiacritics -> True, Language -> \
         \"English\"]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "PalindromeQ[\"añana\", IgnoreDiacritics -> True, Language -> \
         \"Spanish\"]"
      )
      .unwrap(),
      "False"
    );
  }

  #[test]
  fn language_alone_changes_nothing() {
    assert_eq!(
      interpret("PalindromeQ[\"añana\", Language -> \"English\"]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret(
        "PalindromeQ[\"Anna\", IgnoreCase -> True, Language -> \"Spanish\"]"
      )
      .unwrap(),
      "True"
    );
  }
}

mod letter_counts {
  use super::*;

  #[test]
  fn extra_characters_count_as_letters() {
    assert_eq!(
      interpret("LetterCounts[\"TK-421\", 1, {\"1\", \"2\"}]").unwrap(),
      "<|1 -> 1, 2 -> 1, K -> 1, T -> 1|>"
    );
  }

  #[test]
  fn extra_characters_join_ngrams() {
    assert_eq!(
      interpret("LetterCounts[\"TK-421\", 2, {\"1\", \"2\"}]").unwrap(),
      "<|TK -> 1, 21 -> 1|>"
    );
  }
}

mod template_slot {
  use super::*;

  #[test]
  fn non_list_argument_is_slot_one() {
    assert_eq!(
      interpret("TemplateApply[TemplateSlot[1], <|\"age\" -> 32|>]").unwrap(),
      "<|age -> 32|>"
    );
    assert_eq!(
      interpret("TemplateApply[TemplateSlot[1], {1}]").unwrap(),
      "1"
    );
  }

  #[test]
  fn named_slot_searches_every_association() {
    assert_eq!(
      interpret(
        "TemplateApply[TemplateSlot[\"name\"], {<|\"age\" -> 32|>, \
         <|\"name\" -> \"Paul\"|>}]"
      )
      .unwrap(),
      "Paul"
    );
  }

  #[test]
  fn filled_template_is_evaluated() {
    assert_eq!(
      interpret(
        "TemplateApply[Mean[{TemplateSlot[\"me\"], 50, 30}], <|\"me\" -> 100|>]"
      )
      .unwrap(),
      "60"
    );
  }
}

mod template_if {
  use super::*;

  #[test]
  fn condition_is_held_until_applied() {
    assert_eq!(
      interpret(
        "t = TemplateIf[StringQ[TemplateSlot[1]], TemplateSlot[1], \
         \"please insert a string\"]; {TemplateApply[t, 23], \
         TemplateApply[t, \"hello\"]}"
      )
      .unwrap(),
      "{please insert a string, hello}"
    );
  }

  #[test]
  fn insertion_function_formats_the_slots() {
    let t = "t = TemplateIf[MatchQ[TemplateSlot[\"planet\"], _String], \
             TemplateSlot[\"planet\"], \"Please insert a planet\", \
             InsertionFunction -> ToUpperCase];";
    assert_eq!(
      interpret(&format!(
        "{t} TemplateApply[t, <|\"planet\" -> \"Jupiter\"|>]"
      ))
      .unwrap(),
      "JUPITER"
    );
  }
}
