//! AST-based list helper functions.
//!
//! These functions work directly with `Expr` AST nodes, avoiding the string
//! round-trips and re-parsing that the original `list_helpers.rs` functions use.

use crate::InterpreterError;
use crate::syntax::Expr;

mod aggregation;
mod combinatorics;
mod construction;
mod element_access;
mod filtering;
mod functional;
mod mapping;
mod properties;
mod restructuring;
mod set_operations;
mod sorting;
mod summation;
mod utilities;

pub use aggregation::*;
pub use combinatorics::*;
pub use construction::*;
pub use element_access::*;
pub use filtering::*;
pub use functional::*;
pub use mapping::*;
pub use properties::*;
pub use restructuring::*;
pub use set_operations::*;
pub use sorting::*;
pub use summation::*;
pub use utilities::*;
