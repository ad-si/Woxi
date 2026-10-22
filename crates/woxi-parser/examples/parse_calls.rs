//! Report how many pest rule calls it takes to parse a file:
//! `cargo run --release -p woxi-parser --example parse_calls -- FILE`.
//! Useful for finding grammar constructs that backtrack exponentially.
use pest::Parser;
use woxi_parser::{Rule, WolframParser};

fn parses_within(input: &str, limit: usize) -> bool {
  pest::set_call_limit(std::num::NonZeroUsize::new(limit));
  WolframParser::parse(Rule::Program, input).is_ok()
}

fn main() {
  let path = std::env::args().nth(1).expect("usage: parse_calls FILE");
  let input = std::fs::read_to_string(path).expect("readable file");
  let (mut lo, mut hi) = (1usize, 1usize << 20);
  while !parses_within(&input, hi) {
    if hi > 1 << 36 {
      pest::set_call_limit(None);
      match WolframParser::parse(Rule::Program, &input) {
        Err(e) => println!("does not parse: {e}"),
        Ok(_) => println!("parses only without a call limit"),
      }
      return;
    }
    lo = hi;
    hi *= 2;
  }
  while hi - lo > hi / 100 {
    let mid = lo + (hi - lo) / 2;
    if parses_within(&input, mid) {
      hi = mid
    } else {
      lo = mid
    }
  }
  println!(
    "{} calls ({} bytes, {} per byte)",
    hi,
    input.len(),
    hi / input.len().max(1)
  );
}
