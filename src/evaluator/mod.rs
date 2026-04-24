use crate::syntax::{
  BinaryOperator, ComparisonOp, Expr, UnaryOperator, expr_to_string,
};
use crate::{ENV, InterpreterError, PART_DEPTH, StoredValue, interpret};

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

// functions.csv format: name,description,implementation_status,effect_level,version,rank
static FUNCTIONS_CSV: &str = include_str!("../../functions.csv");

/// Set of known Wolfram Language function names (from functions.csv)
/// that are NOT yet implemented in Woxi.
static KNOWN_WOLFRAM_FUNCTIONS: LazyLock<HashSet<&'static str>> =
  LazyLock::new(|| {
    FUNCTIONS_CSV
      .lines()
      .skip(1)
      .filter_map(|line| {
        let fields: Vec<&str> = line.splitn(4, ',').collect();
        let name = fields.first()?.trim();
        let status = fields.get(2).unwrap_or(&"").trim();
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

/// Information about a built-in Wolfram Language function.
pub struct BuiltinFunctionInfo {
  pub description: &'static str,
  pub effect_level: &'static str,
}

/// Registry of all function info from functions.csv.
static BUILTIN_FUNCTION_INFO: LazyLock<
  HashMap<&'static str, BuiltinFunctionInfo>,
> = LazyLock::new(|| {
  FUNCTIONS_CSV
    .lines()
    .skip(1)
    .filter_map(|line| {
      let fields: Vec<&str> = line.splitn(5, ',').collect();
      let name = fields.first()?.trim();
      let description = fields.get(1).unwrap_or(&"").trim();
      let effect_level = fields.get(3).unwrap_or(&"").trim();
      if name.is_empty() || name == "-----" {
        return None;
      }
      if description.is_empty() && effect_level.is_empty() {
        return None;
      }
      Some((
        name,
        BuiltinFunctionInfo {
          description,
          effect_level,
        },
      ))
    })
    .collect()
});

/// Look up built-in function info by name.
pub fn get_builtin_function_info(
  name: &str,
) -> Option<&'static BuiltinFunctionInfo> {
  BUILTIN_FUNCTION_INFO.get(name)
}

/// Get all built-in function names from functions.csv.
pub fn get_builtin_function_names() -> Vec<&'static str> {
  let mut names: Vec<&str> = BUILTIN_FUNCTION_INFO.keys().copied().collect();
  names.sort();
  names
}

/// Get every Wolfram Language function name that appears in
/// `functions.csv` — whether or not Woxi has implemented it.
pub fn known_wolfram_function_names() -> Vec<&'static str> {
  let mut names: Vec<&str> = KNOWN_WOLFRAM_FUNCTIONS.iter().copied().collect();
  names.sort();
  names
}

pub(crate) mod assignment;
mod attributes;
mod binary_ops;
mod core_eval;
pub mod dispatch;
mod function_application;
mod listable;
mod part_extraction;
mod pattern_functions;
pub(crate) mod pattern_matching;
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
