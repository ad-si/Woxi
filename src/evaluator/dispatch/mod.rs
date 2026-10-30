use super::*;

pub mod arg_count;
mod association_functions;
mod audio_functions;
mod boolean_functions;
pub(crate) mod calculus_functions;
pub mod complex_and_special;
mod datetime_functions;
pub(crate) mod evaluate_functions;
mod evaluation_control;
mod image_functions;
mod interval_functions;
pub(crate) mod io_functions;
mod linear_algebra_functions;
pub(crate) mod list_operations;
mod math_functions;
mod music_functions;
mod plotting;
mod polynomial_functions;
pub mod predicate_functions;
mod quantity_functions;
mod string_functions;
mod structural;
mod timeseries_functions;
mod wavelet_functions;

pub use association_functions::*;
pub use boolean_functions::*;
pub use calculus_functions::*;
pub use complex_and_special::*;
pub use datetime_functions::*;
pub(crate) use evaluate_functions::*;
pub use evaluation_control::*;
pub use image_functions::*;
pub use interval_functions::*;
pub use io_functions::*;
pub use linear_algebra_functions::*;
pub use list_operations::*;
pub use math_functions::*;
pub use plotting::*;
pub use polynomial_functions::*;
pub use predicate_functions::*;
pub use quantity_functions::*;
pub use string_functions::*;
pub use structural::*;

/// Split a call's arguments into the positional ones and the trailing
/// `name -> value` options (symbol or string names), in their given order.
pub(crate) fn split_trailing_options(
  args: &[Expr],
) -> (&[Expr], Vec<(String, Expr)>) {
  let mut split = args.len();
  let mut opts = Vec::new();
  while split > 0 {
    let (lhs, rhs) = match &args[split - 1] {
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => (pattern.as_ref(), replacement.as_ref()),
      _ => break,
    };
    let name = match lhs {
      Expr::Identifier(s) | Expr::String(s) => s.clone(),
      _ => break,
    };
    opts.push((name, rhs.clone()));
    split -= 1;
  }
  opts.reverse();
  (&args[..split], opts)
}
