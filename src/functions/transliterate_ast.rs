//! AST-native implementation of `Transliterate`.
//!
//! `Transliterate[s]` converts text in other writing scripts to plain
//! Latin/ASCII. wolframscript implements this with ICU's
//! "Any-Latin; Latin-ASCII" transform chain, so the per-script rules here
//! follow the corresponding CLDR transform data:
//!
//! - Greek: ICU Greek-Latin classical scheme (β→b, η→e, φ→ph, χ→ch,
//!   ψ→ps, θ→th, γ-nasal, υ→u after α/ε/ο, rough breathing→h)
//! - Cyrillic: ISO 9 / GOST 7.79 System A, folded to ASCII (ь→', ъ→")
//! - Hiragana/Katakana: Hepburn romanization (し→shi, っ gemination, ん')
//! - Hangul: Revised Romanization transliteration (letter-by-letter)
//! - Everything else: Latin-ASCII folding (é→e, ß→ss, æ→ae, curly quotes)
//!
//! Scripts without rules here (Han, Arabic, Hebrew, Indic, …) and the
//! 2-argument target-script forms are not supported yet; unsupported
//! inputs are passed through / left unevaluated.

use crate::InterpreterError;
use crate::syntax::Expr;

/// Transliterate[s] or Transliterate[{s1, s2, ...}]
pub fn transliterate_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Transliterate expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::String(s) => Ok(Expr::String(transliterate_to_ascii(s))),
    Expr::List(items)
      if items.iter().all(|it| matches!(it, Expr::String(_))) =>
    {
      let out: Vec<Expr> = items
        .iter()
        .map(|it| match it {
          Expr::String(s) => Expr::String(transliterate_to_ascii(s)),
          _ => unreachable!(),
        })
        .collect();
      Ok(Expr::List(out.into()))
    }
    // Non-string arguments stay unevaluated, like in wolframscript.
    _ => Ok(Expr::FunctionCall {
      name: "Transliterate".to_string(),
      args: args.to_vec().into(),
    }),
  }
}

/// Transliterate a string to plain Latin/ASCII.
fn transliterate_to_ascii(input: &str) -> String {
  use unicode_normalization::UnicodeNormalization;
  // Compose first so decomposed kana voicing marks, decomposed Hangul
  // jamo, and combining accents arrive as single code points.
  let chars: Vec<char> = input.nfc().collect();
  let mut out = String::with_capacity(chars.len() * 2);
  let mut i = 0;
  while i < chars.len() {
    let c = chars[i];
    if is_kana(c) {
      i = emit_kana(&chars, i, &mut out);
    } else if is_greek(c) {
      i = emit_greek(&chars, i, &mut out);
    } else if let Some(lat) = cyrillic_to_latin(c) {
      out.push_str(lat);
      i += 1;
    } else if let Some(s) = hangul_syllable(c) {
      out.push_str(&s);
      i += 1;
    } else {
      fold_latin_ascii(c, &mut out);
      i += 1;
    }
  }
  out
}

// --------------------------------------------------------------------------
// Latin-ASCII folding
// --------------------------------------------------------------------------

/// Check if a character is a combining diacritical mark (the marks that
/// ICU's Latin-ASCII rule set removes).
fn is_combining_diacritic(c: char) -> bool {
  let cp = c as u32;
  (0x0300..=0x036F).contains(&cp) // Combining Diacritical Marks
    || (0x1AB0..=0x1AFF).contains(&cp) // ... Extended
    || (0x1DC0..=0x1DFF).contains(&cp) // ... Supplement
    || (0x20D0..=0x20FF).contains(&cp) // ... for Symbols
    || (0xFE20..=0xFE2F).contains(&cp) // Combining Half Marks
}

/// Fold a single character to ASCII like ICU's Latin-ASCII transform:
/// special-cased letters without decompositions, common punctuation, and
/// removal of combining diacritics after NFD decomposition.
fn fold_latin_ascii(c: char, out: &mut String) {
  use unicode_normalization::UnicodeNormalization;
  match c {
    'ß' => out.push_str("ss"),
    'ẞ' => out.push_str("SS"),
    'æ' => out.push_str("ae"),
    'Æ' => out.push_str("AE"),
    'œ' => out.push_str("oe"),
    'Œ' => out.push_str("OE"),
    'ø' => out.push('o'),
    'Ø' => out.push('O'),
    'đ' => out.push('d'),
    'Đ' => out.push('D'),
    'ð' => out.push('d'),
    'Ð' => out.push('D'),
    'þ' => out.push_str("th"),
    'Þ' => out.push_str("Th"),
    'ł' => out.push('l'),
    'Ł' => out.push('L'),
    'ħ' => out.push('h'),
    'Ħ' => out.push('H'),
    'ı' => out.push('i'),
    // Modifier primes (produced by ISO 9 for soft/hard signs)
    '\u{02B9}' => out.push('\''),
    '\u{02BA}' => out.push('"'),
    // Curly quotes, dashes, ellipsis, non-breaking space
    '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => out.push('\''),
    '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => out.push('"'),
    '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}'
    | '\u{2015}' => out.push('-'),
    '\u{2026}' => out.push_str("..."),
    '\u{00A0}' => out.push(' '),
    _ => {
      if c.is_ascii() {
        out.push(c);
      } else {
        for d in std::iter::once(c).nfd() {
          if !is_combining_diacritic(d) {
            out.push(d);
          }
        }
      }
    }
  }
}

// --------------------------------------------------------------------------
// Greek (ISO 843, folded to ASCII)
// --------------------------------------------------------------------------

fn is_greek(c: char) -> bool {
  matches!(c as u32, 0x0370..=0x03FF | 0x1F00..=0x1FFF)
}

/// Strip diacritics from a Greek letter and return its lowercase base,
/// whether the original was uppercase, and whether it carried a rough
/// breathing mark (U+0314), which transliterates as an `h`.
fn greek_base(c: char) -> Option<(char, bool, bool)> {
  use unicode_normalization::UnicodeNormalization;
  let mut marks = std::iter::once(c).nfd();
  let b = marks.next()?;
  if !is_greek(b) {
    return None;
  }
  let rough = marks.any(|m| m == '\u{0314}');
  let upper = b.is_uppercase();
  let low = b.to_lowercase().next()?;
  Some((low, upper, rough))
}

/// Last code point of a character's NFD expansion (the base letter if
/// unaccented, else its final combining mark).
fn last_nfd_codepoint(c: char) -> Option<char> {
  use unicode_normalization::UnicodeNormalization;
  std::iter::once(c).nfd().last()
}

/// Is the character at `idx` (if any) a lowercase letter? Used for the
/// casing of Greek digraphs (Θα → THA but Θε…ός → Theós).
fn next_is_lowercase(chars: &[char], idx: usize) -> bool {
  chars.get(idx).is_some_and(|c| c.is_lowercase())
}

fn emit_greek(chars: &[char], i: usize, out: &mut String) -> usize {
  let c = chars[i];
  // Greek punctuation
  if c == '\u{037E}' {
    out.push('?'); // Greek question mark
    return i + 1;
  }
  if c == '\u{0387}' {
    out.push(';'); // ano teleia
    return i + 1;
  }
  let Some((low, upper, rough)) = greek_base(c) else {
    out.push(c);
    return i + 1;
  };
  // γ before another velar (γ, κ, ξ, χ) is nasal: γγ → ng
  let nasal = low == 'γ'
    && chars
      .get(i + 1)
      .and_then(|&n| greek_base(n))
      .is_some_and(|(n, _, _)| matches!(n, 'γ' | 'κ' | 'ξ' | 'χ'));
  // υ in the diphthongs αυ, ευ, ου is transliterated u (else y). ICU
  // matches the context on NFD text, so an accent on the PRECEDING vowel
  // blocks the rule (άυ → ay) while an accent on υ itself does not
  // (ού → ou): check the last NFD code point of the previous character.
  let diphthong = low == 'υ'
    && i > 0
    && last_nfd_codepoint(chars[i - 1])
      .and_then(|p| p.to_lowercase().next())
      .is_some_and(|p| matches!(p, 'α' | 'ε' | 'ο'));
  let s: &str = match low {
    'α' => "a",
    'β' | 'ϐ' => "b",
    'γ' => {
      if nasal {
        "n"
      } else {
        "g"
      }
    }
    'δ' => "d",
    'ε' => "e",
    'ζ' => "z",
    'η' => "e",
    'θ' | 'ϑ' => "th",
    'ι' => "i",
    'κ' | 'ϰ' => "k",
    'λ' => "l",
    'μ' => "m",
    'ν' => "n",
    'ξ' => "x",
    'ο' => "o",
    'π' | 'ϖ' => "p",
    'ρ' | 'ϱ' => "r",
    'σ' | 'ς' | 'ϲ' => "s",
    'τ' => "t",
    'υ' => {
      if diphthong {
        "u"
      } else {
        "y"
      }
    }
    'φ' | 'ϕ' => "ph",
    'χ' => "ch",
    'ψ' => "ps",
    'ω' => "o",
    _ => {
      // Archaic letters etc. — pass through
      out.push(c);
      return i + 1;
    }
  };
  // Rough breathing adds an h: before vowels (ἁ → ha), after ρ (ῥ → rh)
  let s: String = if rough {
    if low == 'ρ' {
      "rh".to_string()
    } else {
      format!("h{s}")
    }
  } else {
    s.to_string()
  };
  if upper {
    if s.len() == 1 || next_is_lowercase(chars, i + 1) {
      // Single letter, or digraph followed by lowercase: capitalize
      let mut it = s.chars();
      let first = it.next().unwrap();
      out.extend(first.to_uppercase());
      out.push_str(it.as_str());
    } else {
      // Digraph in an all-caps context: ΘΑ → THA
      out.push_str(&s.to_uppercase());
    }
  } else {
    out.push_str(&s);
  }
  i + 1
}

// --------------------------------------------------------------------------
// Cyrillic (ISO 9 / GOST 7.79 System A, folded to ASCII)
// --------------------------------------------------------------------------

fn cyrillic_to_latin(c: char) -> Option<&'static str> {
  Some(match c {
    'а' => "a",
    'б' => "b",
    'в' => "v",
    'г' => "g",
    'д' => "d",
    'е' => "e",
    'ё' => "e",
    'ж' => "z",
    'з' => "z",
    'и' => "i",
    'й' => "j",
    'к' => "k",
    'л' => "l",
    'м' => "m",
    'н' => "n",
    'о' => "o",
    'п' => "p",
    'р' => "r",
    'с' => "s",
    'т' => "t",
    'у' => "u",
    'ф' => "f",
    'х' => "h",
    'ц' => "c",
    'ч' => "c",
    'ш' => "s",
    'щ' => "s",
    'ъ' => "\"",
    'ы' => "y",
    'ь' => "'",
    'э' => "e",
    'ю' => "u",
    'я' => "a",
    'А' => "A",
    'Б' => "B",
    'В' => "V",
    'Г' => "G",
    'Д' => "D",
    'Е' => "E",
    'Ё' => "E",
    'Ж' => "Z",
    'З' => "Z",
    'И' => "I",
    'Й' => "J",
    'К' => "K",
    'Л' => "L",
    'М' => "M",
    'Н' => "N",
    'О' => "O",
    'П' => "P",
    'Р' => "R",
    'С' => "S",
    'Т' => "T",
    'У' => "U",
    'Ф' => "F",
    'Х' => "H",
    'Ц' => "C",
    'Ч' => "C",
    'Ш' => "S",
    'Щ' => "S",
    'Ъ' => "\"",
    'Ы' => "Y",
    'Ь' => "'",
    'Э' => "E",
    'Ю' => "U",
    'Я' => "A",
    // Ukrainian, Belarusian, Serbian/Macedonian, historic letters
    'є' => "e",
    'Є' => "E",
    'і' => "i",
    'І' => "I",
    'ї' => "i",
    'Ї' => "I",
    'ґ' => "g",
    'Ґ' => "G",
    'ѓ' => "g",
    'Ѓ' => "G",
    'ќ' => "k",
    'Ќ' => "K",
    'ў' => "u",
    'Ў' => "U",
    'ђ' => "d",
    'Ђ' => "D",
    'ћ' => "c",
    'Ћ' => "C",
    'љ' => "l",
    'Љ' => "L",
    'њ' => "n",
    'Њ' => "N",
    'џ' => "d",
    'Џ' => "D",
    'ѕ' => "z",
    'Ѕ' => "Z",
    'ј' => "j",
    'Ј' => "J",
    'ѐ' => "e",
    'Ѐ' => "E",
    'ѝ' => "i",
    'Ѝ' => "I",
    'ѣ' => "e",
    'Ѣ' => "E",
    'ѳ' => "f",
    'Ѳ' => "F",
    'ѵ' => "y",
    'Ѵ' => "Y",
    _ => return None,
  })
}

// --------------------------------------------------------------------------
// Japanese kana (Hepburn romanization)
// --------------------------------------------------------------------------

fn is_kana(c: char) -> bool {
  matches!(c as u32, 0x3041..=0x3096 | 0x30A1..=0x30FA | 0x30FC)
}

/// Map katakana to the corresponding hiragana so both blocks share one
/// romanization table. The prolonged-sound mark and the rare va-row
/// katakana (which have no hiragana equivalents) pass through unchanged.
fn kana_base(c: char) -> char {
  match c as u32 {
    0x30A1..=0x30F6 => char::from_u32(c as u32 - 0x60).unwrap(),
    _ => c,
  }
}

fn kana_romaji(c: char) -> Option<&'static str> {
  Some(match c {
    'ぁ' | 'あ' => "a",
    'ぃ' | 'い' => "i",
    'ぅ' | 'う' => "u",
    'ぇ' | 'え' => "e",
    'ぉ' | 'お' => "o",
    'か' | 'ゕ' => "ka",
    'き' => "ki",
    'く' => "ku",
    'け' | 'ゖ' => "ke",
    'こ' => "ko",
    'が' => "ga",
    'ぎ' => "gi",
    'ぐ' => "gu",
    'げ' => "ge",
    'ご' => "go",
    'さ' => "sa",
    'し' => "shi",
    'す' => "su",
    'せ' => "se",
    'そ' => "so",
    'ざ' => "za",
    'じ' => "ji",
    'ず' => "zu",
    'ぜ' => "ze",
    'ぞ' => "zo",
    'た' => "ta",
    'ち' => "chi",
    'つ' => "tsu",
    'て' => "te",
    'と' => "to",
    'だ' => "da",
    'ぢ' => "ji",
    'づ' => "zu",
    'で' => "de",
    'ど' => "do",
    'な' => "na",
    'に' => "ni",
    'ぬ' => "nu",
    'ね' => "ne",
    'の' => "no",
    'は' => "ha",
    'ひ' => "hi",
    'ふ' => "fu",
    'へ' => "he",
    'ほ' => "ho",
    'ば' => "ba",
    'び' => "bi",
    'ぶ' => "bu",
    'べ' => "be",
    'ぼ' => "bo",
    'ぱ' => "pa",
    'ぴ' => "pi",
    'ぷ' => "pu",
    'ぺ' => "pe",
    'ぽ' => "po",
    'ま' => "ma",
    'み' => "mi",
    'む' => "mu",
    'め' => "me",
    'も' => "mo",
    'や' | 'ゃ' => "ya",
    'ゆ' | 'ゅ' => "yu",
    'よ' | 'ょ' => "yo",
    'ら' => "ra",
    'り' => "ri",
    'る' => "ru",
    'れ' => "re",
    'ろ' => "ro",
    'わ' | 'ゎ' => "wa",
    'ゐ' => "wi",
    'ゑ' => "we",
    'を' => "wo",
    'ん' => "n",
    'ゔ' => "vu",
    'ヷ' => "va",
    'ヸ' => "vi",
    'ヹ' => "ve",
    'ヺ' => "vo",
    _ => return None,
  })
}

/// Romanize the kana mora starting at `i`, contracting yōon digraphs
/// (きゃ → kya, しゃ → sha). Returns the romaji and how many characters
/// were consumed.
fn mora_at(chars: &[char], i: usize) -> Option<(String, usize)> {
  if i >= chars.len() || !is_kana(chars[i]) {
    return None;
  }
  let romaji = kana_romaji(kana_base(chars[i]))?;
  if i + 1 < chars.len() {
    let next = kana_base(chars[i + 1]);
    if matches!(next, 'ゃ' | 'ゅ' | 'ょ')
      && romaji.ends_with('i')
      && romaji != "i"
    {
      let stem = &romaji[..romaji.len() - 1];
      let v = match next {
        'ゃ' => "ya",
        'ゅ' => "yu",
        _ => "yo",
      };
      let combined = if matches!(stem, "sh" | "ch" | "j") {
        // sha, chu, jo — the y is absorbed
        format!("{}{}", stem, &v[1..])
      } else {
        format!("{}{}", stem, v)
      };
      return Some((combined, 2));
    }
  }
  Some((romaji.to_string(), 1))
}

fn emit_kana(chars: &[char], i: usize, out: &mut String) -> usize {
  match kana_base(chars[i]) {
    // Sokuon: geminate the following consonant (っち → tchi in Hepburn)
    'っ' => {
      if let Some((next, _)) = mora_at(chars, i + 1) {
        let first = next.chars().next().unwrap();
        if !"aiueo".contains(first) {
          if next.starts_with("ch") {
            out.push('t');
          } else {
            out.push(first);
          }
        }
      }
      i + 1
    }
    // Moraic n: apostrophe before vowels and y (しんいち → shin'ichi)
    'ん' => {
      out.push('n');
      if let Some((next, _)) = mora_at(chars, i + 1) {
        let first = next.chars().next().unwrap();
        if "aiueoy".contains(first) {
          out.push('\'');
        }
      }
      i + 1
    }
    // Prolonged sound mark lengthens the previous vowel; the macron it
    // would produce is folded away in ASCII output
    'ー' => i + 1,
    _ => {
      if let Some((romaji, used)) = mora_at(chars, i) {
        out.push_str(&romaji);
        i + used
      } else {
        out.push(chars[i]);
        i + 1
      }
    }
  }
}

// --------------------------------------------------------------------------
// Hangul (Revised Romanization, transliteration variant)
// --------------------------------------------------------------------------

fn hangul_syllable(c: char) -> Option<String> {
  let cp = c as u32;
  if !(0xAC00..=0xD7A3).contains(&cp) {
    return None;
  }
  const LEADS: [&str; 19] = [
    "g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj",
    "ch", "k", "t", "p", "h",
  ];
  const VOWELS: [&str; 21] = [
    "a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe",
    "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i",
  ];
  const TAILS: [&str; 28] = [
    "", "g", "kk", "gs", "n", "nj", "nh", "d", "l", "lg", "lm", "lb", "ls",
    "lt", "lp", "lh", "m", "b", "bs", "s", "ss", "ng", "j", "ch", "k", "t",
    "p", "h",
  ];
  let s = cp - 0xAC00;
  let lead = (s / 588) as usize;
  let vowel = ((s % 588) / 28) as usize;
  let tail = (s % 28) as usize;
  Some(format!("{}{}{}", LEADS[lead], VOWELS[vowel], TAILS[tail]))
}
