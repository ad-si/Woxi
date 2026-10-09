//! The Wolfram Language grammar (`wolfram.pest`) and the parser `pest_derive`
//! generates from it.
//!
//! This lives in its own crate purely for compile times: the derive macro
//! regenerates the whole parser on every compilation of the crate that
//! contains it, which inside the ~570k-line `woxi` crate cost several seconds
//! of macro expansion on every incremental rebuild (twice, for the library
//! and its unit tests). Here it only reruns when the grammar changes.
//! `woxi` re-exports both items, so they remain `woxi::{Rule, WolframParser}`.

use pest::Parser;

#[derive(pest_derive::Parser)]
#[grammar = "wolfram.pest"]
pub struct WolframParser;

impl WolframParser {
  pub fn parse_wolfram(
    input: &str,
  ) -> Result<pest::iterators::Pairs<'_, Rule>, Box<pest::error::Error<Rule>>>
  {
    // Deeply nested invalid input (e.g. `f[f[f[...` without closing
    // brackets) makes pest backtrack exponentially, so rejecting it would
    // take hours. The call limit turns that into a "call limit reached"
    // parse error. The worst script in tests/scripts needs ~5,700 calls
    // per byte, so 20,000 per byte (plus a base allowance covering the
    // fixed overhead on tiny inputs) leaves ample headroom for legitimate
    // code of any size, while short pathological inputs are rejected in
    // well under a second even in debug builds on slow CI hardware.
    let limit =
      1_000_000_usize.saturating_add(input.len().saturating_mul(20_000));
    pest::set_call_limit(std::num::NonZeroUsize::new(limit));
    Self::parse(Rule::Program, input).map_err(Box::new)
  }
}
