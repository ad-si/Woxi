use crate::syntax::{
  BinaryOperator, ComparisonOp, Expr, UnaryOperator, expr_to_string,
};
use crate::{
  ENV, InterpreterError, PART_DEPTH, StoredValue, format_real_result,
  format_result, interpret,
};

use std::collections::HashSet;
use std::sync::LazyLock;

/// Set of known Wolfram Language function names (from functions.csv)
/// that are NOT yet implemented in Woxi.
static KNOWN_WOLFRAM_FUNCTIONS: LazyLock<HashSet<&'static str>> =
  LazyLock::new(|| {
    let csv = include_str!("../../functions.csv");
    csv
      .lines()
      .skip(1) // skip header
      .filter_map(|line| {
        let mut parts = line.splitn(3, ',');
        let name = parts.next()?.trim();
        let _desc = parts.next()?;
        let status = parts.next().unwrap_or("").trim();
        // Only include functions that are NOT implemented (not ✅)
        if !status.starts_with("✅") && !name.is_empty() && name != "-----" {
          Some(name)
        } else {
          None
        }
      })
      .collect()
  });

/// Check if a function name is a known Wolfram Language function
/// that hasn't been implemented yet.
pub fn is_known_wolfram_function(name: &str) -> bool {
  KNOWN_WOLFRAM_FUNCTIONS.contains(name)
}

mod assignment;
mod attributes;
mod binary_ops;
mod core_eval;
mod dispatch;
mod function_application;
mod listable;
mod part_extraction;
mod pattern_functions;
mod pattern_matching;
mod scoping;
mod string_replace;
mod type_helpers;

pub use assignment::*;
pub use attributes::*;
pub use binary_ops::*;
pub use core_eval::*;
pub use dispatch::*;
pub use function_application::*;
pub use listable::*;
pub use part_extraction::*;
pub use pattern_functions::*;
pub use pattern_matching::*;
pub use scoping::*;
pub use string_replace::*;
pub use type_helpers::*;
