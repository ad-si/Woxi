//! AST-native math functions.
//!
//! These functions work directly with `Expr` AST nodes.

mod numeric_utils;
mod arithmetic;
mod elementary;
mod trigonometric;
mod special_functions;
mod orthogonal_polynomials;
mod number_theory;
mod digits;
mod complex;
mod statistics;
mod numerical;
mod random;

pub use numeric_utils::*;
pub use arithmetic::*;
pub use elementary::*;
pub use trigonometric::*;
pub use special_functions::*;
pub use orthogonal_polynomials::*;
pub use number_theory::*;
pub use digits::*;
pub use complex::*;
pub use statistics::*;
pub use numerical::*;
pub use random::*;
