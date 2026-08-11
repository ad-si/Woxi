use crate::helpers::{binop, bool_expr, div2, minus2, pow2};
use crate::syntax::{
  BinaryOperator, ComparisonOp, Expr, UnaryOperator, expr_to_string,
  unevaluated,
};
use crate::{ENV, InterpreterError, PART_DEPTH, StoredValue, interpret};

pub(crate) mod assignment;
mod attributes;
mod binary_ops;
mod core_eval;
pub mod dispatch;
pub(crate) mod function_application;
pub mod functions;
mod listable;
pub(crate) mod part_extraction;
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
pub use functions::*;
pub use listable::*;
pub use part_extraction::*;
pub use pattern_functions::*;
pub use pattern_matching::*;
pub use scoping::*;
pub use string_replace::*;
pub use type_helpers::*;
