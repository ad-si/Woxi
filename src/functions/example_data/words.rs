//! Words and sentences for generated text.
//!
//! English prose mixes frequent function words (with a Zipf-like bias)
//! and a curated list of common content words. Other languages get pseudo-words built from syllables over
//! the letters that language uses (as recorded for the original text).

use super::rng::Rng;

/// Common, neutral English content words. The generated prose draws
/// from these rather than from the full dictionary, which reads better and
/// keeps rare or offensive words out of the example texts.
const CONTENT_WORDS: &[&str] = &[
  "house",
  "water",
  "river",
  "mountain",
  "garden",
  "window",
  "door",
  "road",
  "city",
  "village",
  "field",
  "forest",
  "tree",
  "flower",
  "stone",
  "sea",
  "ship",
  "boat",
  "island",
  "shore",
  "harbor",
  "bridge",
  "tower",
  "castle",
  "wall",
  "gate",
  "hill",
  "valley",
  "lake",
  "stream",
  "rain",
  "wind",
  "cloud",
  "sky",
  "sun",
  "moon",
  "star",
  "night",
  "morning",
  "evening",
  "summer",
  "winter",
  "spring",
  "autumn",
  "season",
  "year",
  "month",
  "week",
  "hour",
  "moment",
  "light",
  "shadow",
  "fire",
  "smoke",
  "snow",
  "ice",
  "storm",
  "thunder",
  "voice",
  "song",
  "music",
  "word",
  "letter",
  "book",
  "page",
  "story",
  "name",
  "land",
  "earth",
  "world",
  "country",
  "nation",
  "people",
  "family",
  "friend",
  "brother",
  "sister",
  "mother",
  "father",
  "child",
  "king",
  "queen",
  "prince",
  "lord",
  "lady",
  "servant",
  "soldier",
  "captain",
  "sailor",
  "farmer",
  "teacher",
  "doctor",
  "student",
  "writer",
  "painter",
  "traveler",
  "stranger",
  "neighbor",
  "guest",
  "crowd",
  "army",
  "council",
  "court",
  "law",
  "order",
  "peace",
  "war",
  "battle",
  "victory",
  "honor",
  "glory",
  "truth",
  "reason",
  "mind",
  "heart",
  "soul",
  "spirit",
  "hope",
  "fear",
  "joy",
  "sorrow",
  "anger",
  "love",
  "faith",
  "courage",
  "wisdom",
  "memory",
  "dream",
  "thought",
  "idea",
  "question",
  "answer",
  "purpose",
  "power",
  "strength",
  "freedom",
  "justice",
  "duty",
  "labor",
  "work",
  "trade",
  "money",
  "price",
  "market",
  "bread",
  "wine",
  "milk",
  "fruit",
  "apple",
  "grain",
  "corn",
  "salt",
  "horse",
  "dog",
  "bird",
  "fish",
  "sheep",
  "cattle",
  "wolf",
  "lion",
  "eagle",
  "bee",
  "table",
  "chair",
  "bed",
  "room",
  "hall",
  "roof",
  "floor",
  "lamp",
  "candle",
  "clock",
  "bell",
  "coat",
  "hat",
  "shoe",
  "ring",
  "sword",
  "shield",
  "arrow",
  "wheel",
  "cart",
  "path",
  "journey",
  "voyage",
  "distance",
  "corner",
  "edge",
  "center",
  "side",
  "top",
  "bottom",
  "line",
  "circle",
  "point",
  "number",
  "part",
  "piece",
  "half",
  "whole",
  "kind",
  "sort",
  "manner",
  "way",
  "means",
  "end",
  "beginning",
  "middle",
  "cause",
  "effect",
  "change",
  "chance",
  "danger",
  "death",
  "life",
  "birth",
  "health",
  "body",
  "hand",
  "face",
  "eye",
  "head",
  "arm",
  "foot",
  "hair",
  "blood",
  "bone",
  "skin",
  "breath",
  "sleep",
  "rest",
  "silence",
  "noise",
  "color",
  "gold",
  "silver",
  "iron",
  "wood",
  "glass",
  "paper",
  "cloth",
  "thread",
  "rope",
  "chain",
  "key",
  "box",
  "bag",
  "basket",
  "cup",
  "plate",
  "bowl",
  "bottle",
  "message",
  "news",
  "report",
  "history",
  "science",
  "art",
  "nature",
  "form",
  "figure",
  "pattern",
  "measure",
  "weight",
  "speak",
  "tell",
  "say",
  "ask",
  "call",
  "hear",
  "listen",
  "see",
  "watch",
  "look",
  "find",
  "seek",
  "keep",
  "hold",
  "give",
  "take",
  "bring",
  "carry",
  "send",
  "leave",
  "arrive",
  "return",
  "follow",
  "lead",
  "walk",
  "run",
  "ride",
  "sail",
  "climb",
  "fall",
  "rise",
  "stand",
  "sit",
  "lie",
  "wait",
  "stay",
  "live",
  "grow",
  "build",
  "make",
  "break",
  "open",
  "close",
  "begin",
  "finish",
  "learn",
  "teach",
  "know",
  "understand",
  "remember",
  "forget",
  "believe",
  "think",
  "wonder",
  "feel",
  "hate",
  "wish",
  "want",
  "need",
  "use",
  "try",
  "help",
  "serve",
  "rule",
  "govern",
  "judge",
  "agree",
  "promise",
  "refuse",
  "accept",
  "allow",
  "forbid",
  "write",
  "read",
  "sing",
  "play",
  "dance",
  "laugh",
  "smile",
  "weep",
  "cry",
  "pray",
  "fight",
  "win",
  "lose",
  "gather",
  "spread",
  "count",
  "buy",
  "sell",
  "pay",
  "owe",
  "borrow",
  "lend",
  "travel",
  "wander",
  "wake",
  "burn",
  "shine",
  "glow",
  "fade",
  "turn",
  "move",
  "pass",
  "cross",
  "reach",
  "touch",
  "meet",
  "join",
  "divide",
  "share",
  "choose",
  "decide",
  "appear",
  "vanish",
  "belong",
  "seem",
  "become",
  "remain",
  "happen",
  "increase",
  "reduce",
  "gain",
  "spend",
  "save",
  "good",
  "great",
  "small",
  "large",
  "little",
  "long",
  "short",
  "high",
  "low",
  "deep",
  "wide",
  "narrow",
  "old",
  "young",
  "new",
  "ancient",
  "early",
  "late",
  "quick",
  "slow",
  "strong",
  "weak",
  "bright",
  "dark",
  "heavy",
  "warm",
  "cold",
  "hot",
  "cool",
  "soft",
  "hard",
  "rough",
  "smooth",
  "clear",
  "plain",
  "simple",
  "rich",
  "poor",
  "full",
  "empty",
  "free",
  "broken",
  "quiet",
  "loud",
  "gentle",
  "fierce",
  "wild",
  "calm",
  "happy",
  "glad",
  "sad",
  "proud",
  "humble",
  "wise",
  "noble",
  "brave",
  "true",
  "false",
  "fair",
  "green",
  "blue",
  "red",
  "white",
  "black",
  "golden",
  "distant",
  "near",
  "strange",
  "common",
  "rare",
  "certain",
  "sudden",
  "final",
  "first",
  "last",
  "next",
  "secret",
  "public",
  "private",
  "silent",
  "sweet",
  "bitter",
  "fresh",
  "pure",
  "natural",
  "human",
  "divine",
  "mortal",
  "faithful",
  "careful",
  "patient",
  "eager",
  "honest",
  "equal",
  "general",
  "often",
  "never",
  "always",
  "seldom",
  "soon",
  "again",
  "still",
  "almost",
  "perhaps",
  "indeed",
  "rather",
  "quite",
  "slowly",
  "quickly",
  "gently",
  "together",
  "alone",
  "already",
  "forward",
  "away",
  "beyond",
  "within",
  "without",
  "above",
  "below",
  "across",
  "along",
  "around",
  "beside",
];

/// The most frequent English function words, used often enough that the
/// generated text has a realistic word-frequency profile.
const FUNCTION_WORDS: &[&str] = &[
  "the", "of", "and", "to", "a", "in", "that", "is", "was", "he", "for", "it",
  "with", "as", "his", "on", "be", "at", "by", "i", "this", "had", "not",
  "are", "but", "from", "or", "have", "an", "they", "which", "one", "you",
  "were", "her", "all", "she", "there", "would", "their", "we", "him", "been",
  "has", "when", "who", "will", "more", "no", "if", "out", "so", "said",
  "what", "up", "its", "about", "into", "than", "them", "can", "only", "other",
  "new", "some", "could", "time", "these", "two", "may", "then", "do", "first",
  "any", "my", "now", "such", "like", "our", "over", "man", "me", "even",
  "most", "made", "after", "also", "did", "many", "before", "must", "through",
  "back", "years", "where", "much", "your", "way", "well", "down", "should",
  "because", "each", "just", "those", "people", "how", "too", "little",
  "state", "good", "very", "make", "world", "still", "own", "see", "men",
  "work", "long", "get", "here", "between", "both", "life", "being", "under",
  "never", "day", "same", "another", "know", "while", "last", "might", "us",
  "great", "old", "year", "off", "come", "since", "against", "go", "came",
  "right", "used", "take", "three",
];

/// A source of words in one language.
pub enum Lexicon {
  English,
  /// Pseudo-words over an alphabet: vowels and consonants (in frequency
  /// order), or a list of whole characters for logographic scripts.
  Syllabic {
    vowels: Vec<char>,
    consonants: Vec<char>,
  },
  Logographic {
    characters: Vec<char>,
    spaced: bool,
  },
}

impl Lexicon {
  /// A lexicon for text whose letters, most frequent first, are `letters`
  /// (as recorded in the metadata); `language` picks the strategy.
  pub fn for_language(
    language: &str,
    letters: &[char],
    spaced: bool,
  ) -> Lexicon {
    if language == "English" && !letters.iter().any(|c| (*c as u32) > 0x24f) {
      return Lexicon::English;
    }
    let alphabetic: Vec<char> = letters
      .iter()
      .copied()
      .filter(|c| c.is_alphabetic())
      .collect();
    let logographic =
      alphabetic.iter().filter(|c| (**c as u32) >= 0x2e80).count();
    if logographic * 2 > alphabetic.len().max(1) {
      return Lexicon::Logographic {
        characters: alphabetic
          .into_iter()
          .filter(|c| (*c as u32) >= 0x2e80)
          .collect(),
        spaced,
      };
    }
    let is_vowel = |c: &char| {
      "aeiouyáéíóúàèìòùâêîôûäëïöüåæøœāēīōūαεηιουωаеёиоуыэюяıə"
        .contains(c.to_lowercase().next().unwrap_or(*c))
    };
    let lower: Vec<char> = alphabetic
      .iter()
      .copied()
      .filter(|c| c.is_lowercase())
      .collect();
    let pool = if lower.is_empty() { alphabetic } else { lower };
    let vowels: Vec<char> = pool.iter().copied().filter(is_vowel).collect();
    let consonants: Vec<char> =
      pool.iter().copied().filter(|c| !is_vowel(c)).collect();
    if vowels.is_empty() || consonants.is_empty() {
      return Lexicon::Logographic {
        characters: pool,
        spaced,
      };
    }
    Lexicon::Syllabic { vowels, consonants }
  }

  /// One word.
  pub fn word(&self, rng: &mut Rng) -> String {
    match self {
      Lexicon::English => {
        if rng.chance(0.45) {
          // Zipf-ish: the first function words are the most frequent.
          let i =
            ((rng.uniform().powi(3)) * FUNCTION_WORDS.len() as f64) as usize;
          let w = FUNCTION_WORDS[i.min(FUNCTION_WORDS.len() - 1)];
          return if w == "i" {
            "I".to_string()
          } else {
            w.to_string()
          };
        }
        rng.pick(CONTENT_WORDS).to_string()
      }
      Lexicon::Syllabic { vowels, consonants } => {
        let syllables = 1 + (rng.uniform().powi(2) * 3.5) as usize;
        let mut w = String::new();
        for _ in 0..syllables {
          if rng.chance(0.85) {
            w.push(*biased(rng, consonants));
          }
          w.push(*biased(rng, vowels));
          if rng.chance(0.25) {
            w.push(*biased(rng, consonants));
          }
        }
        w
      }
      Lexicon::Logographic { characters, spaced } => {
        let n = if *spaced {
          2 + rng.below(3)
        } else {
          1 + rng.below(2)
        };
        (0..n).map(|_| *biased(rng, characters)).collect()
      }
    }
  }

  /// Whether words are separated by spaces.
  pub fn spaced(&self) -> bool {
    match self {
      Lexicon::Logographic { spaced, .. } => *spaced,
      _ => true,
    }
  }
}

/// An index biased towards the start of a frequency-ordered list.
fn biased<'a, T>(rng: &mut Rng, items: &'a [T]) -> &'a T {
  let i = (rng.uniform().powf(1.8) * items.len() as f64) as usize;
  &items[i.min(items.len() - 1)]
}

/// Capitalize the first letter.
pub fn capitalize(w: &str) -> String {
  let mut c = w.chars();
  match c.next() {
    Some(f) => f.to_uppercase().chain(c).collect(),
    None => String::new(),
  }
}

/// A sentence of about `words` words, capitalized and punctuated.
pub fn sentence(lexicon: &Lexicon, rng: &mut Rng, words: usize) -> String {
  let n = words.max(1);
  let mut out = String::new();
  for i in 0..n {
    let mut w = lexicon.word(rng);
    if i == 0 {
      w = capitalize(&w);
    }
    if i > 0 && lexicon.spaced() {
      out.push(' ');
    }
    out.push_str(&w);
    if i + 1 < n && n > 6 && rng.chance(0.07) {
      out.push(if lexicon.spaced() { ',' } else { '、' });
    }
  }
  out.push(if !lexicon.spaced() {
    '。'
  } else if rng.chance(0.06) {
    '?'
  } else if rng.chance(0.03) {
    '!'
  } else {
    '.'
  });
  out
}

/// `count` words of running text forming sentences of a typical length.
pub fn prose(
  lexicon: &Lexicon,
  rng: &mut Rng,
  count: usize,
  mean_sentence: f64,
) -> String {
  let mut out = String::new();
  let mut left = count;
  while left > 0 {
    let len = ((mean_sentence + rng.normal() * mean_sentence * 0.4).round()
      as usize)
      .clamp(3, 80)
      .min(left);
    if !out.is_empty() && lexicon.spaced() {
      out.push(' ');
    }
    out.push_str(&sentence(lexicon, rng, len));
    left -= len;
  }
  out
}
