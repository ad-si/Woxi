use crate::InterpreterError;
use crate::helpers::bool_expr;
use crate::syntax::{Expr, unevaluated};

pub mod data;
pub mod edit;
pub mod filters;
pub mod measure;
pub mod spectral;

pub use data::*;
