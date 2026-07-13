//! AST-native math functions.
//!
//! These functions work directly with `Expr` AST nodes.

mod airy;
mod arithmetic;
mod bessel;
mod carlson;
pub mod complex;
mod coordinate_arrays;
mod digits;
mod distributions;
mod elementary;
mod elliptic;
pub(crate) mod gamma;
mod hypergeometric;
pub(crate) mod integrals;
mod jacobi;
mod mathieu;
mod misc_special;
mod number_theory;
mod numeric_utils;
pub(crate) mod numerical;
mod orthogonal_polynomials;
mod polylog;
mod random;
mod statistics;
mod triangles;
mod trigonometric;
mod weierstrass;
mod zeta_functions;

pub use airy::*;
pub use arithmetic::*;
pub use bessel::*;
pub use carlson::*;
pub use complex::*;
pub use coordinate_arrays::*;
pub use digits::*;
pub use distributions::*;
pub use elementary::*;
pub use elliptic::*;
pub use gamma::*;
pub use hypergeometric::*;
pub use integrals::*;
pub use jacobi::*;
pub use mathieu::*;
pub use misc_special::*;
pub use number_theory::*;
pub use numeric_utils::*;
pub use numerical::*;
pub use orthogonal_polynomials::*;
pub use polylog::*;
pub use random::*;
pub use statistics::*;
pub use triangles::*;
pub use trigonometric::*;
pub use weierstrass::*;
pub use zeta_functions::*;

/// Public accessor for the process time-slice distribution (used by
/// SliceDistribution).
pub fn distributions_slice(
  proc_name: &str,
  dargs: &[crate::syntax::Expr],
  t: &crate::syntax::Expr,
) -> Option<crate::syntax::Expr> {
  distributions::process_slice_distribution(proc_name, dargs, t)
}

/// Public accessors for the process correlation closed forms.
pub fn statistics_process_correlation(
  proc: &crate::syntax::Expr,
  t1: &crate::syntax::Expr,
  t2: &crate::syntax::Expr,
) -> Option<crate::syntax::Expr> {
  statistics::process_correlation(proc, t1, t2)
}

pub fn statistics_process_absolute_correlation(
  proc: &crate::syntax::Expr,
  t1: &crate::syntax::Expr,
  t2: &crate::syntax::Expr,
) -> Option<crate::syntax::Expr> {
  statistics::process_absolute_correlation(proc, t1, t2)
}
