//! Wavelet analysis: wavelet families, filter coefficients, discrete,
//! stationary, packet, lifting, and continuous wavelet transforms, plus
//! coefficient manipulation and visualization.

use crate::InterpreterError;
use crate::syntax::{Expr, expr_to_string, unevaluated};

pub mod continuous;
pub mod data;
pub mod filters;
pub mod phipsi;
pub mod plots;
pub mod tables;
pub mod transforms;
