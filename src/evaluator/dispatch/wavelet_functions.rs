//! Dispatch for wavelet analysis heads: filter coefficients, the discrete,
//! stationary, packet, lifting, and continuous wavelet transforms, their
//! inverses, coefficient manipulation (WaveletThreshold, WaveletMapIndexed,
//! WaveletBestBasis), scaling/wavelet functions (WaveletPhi/WaveletPsi),
//! and the wavelet plot functions. Also intercepts `Normal` applied to a
//! DiscreteWaveletData or ContinuousWaveletData object.

use crate::InterpreterError;
use crate::functions::wavelet_ast as wa;
use crate::functions::wavelet_ast::transforms::TransformKind;
use crate::syntax::Expr;

pub(super) fn dispatch_wavelet_functions(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "WaveletFilterCoefficients" => {
      Some(wa::wavelet_filter_coefficients_ast(args))
    }
    "DiscreteWaveletTransform" => {
      Some(wa::data::wavelet_transform_ast(TransformKind::Dwt, args))
    }
    "StationaryWaveletTransform" => {
      Some(wa::data::wavelet_transform_ast(TransformKind::Swt, args))
    }
    "DiscreteWaveletPacketTransform" => {
      Some(wa::data::wavelet_transform_ast(TransformKind::Dwpt, args))
    }
    "StationaryWaveletPacketTransform" => {
      Some(wa::data::wavelet_transform_ast(TransformKind::Swpt, args))
    }
    "LiftingWaveletTransform" => {
      Some(wa::data::wavelet_transform_ast(TransformKind::Lwt, args))
    }
    "InverseWaveletTransform" => {
      Some(wa::data::inverse_wavelet_transform_ast(args))
    }
    "WaveletThreshold" => Some(wa::data::wavelet_threshold_ast(args)),
    "WaveletMapIndexed" => Some(wa::data::wavelet_map_indexed_ast(args)),
    "WaveletBestBasis" => Some(wa::data::wavelet_best_basis_ast(args)),
    "ContinuousWaveletTransform" => {
      Some(wa::continuous::continuous_wavelet_transform_ast(args))
    }
    "InverseContinuousWaveletTransform" => Some(
      wa::continuous::inverse_continuous_wavelet_transform_ast(args),
    ),
    "WaveletPhi" => Some(wa::phipsi::wavelet_phi_ast(args)),
    "WaveletPsi" => Some(wa::phipsi::wavelet_psi_ast(args)),
    "WaveletScalogram" => Some(wa::plots::wavelet_scalogram_ast(args)),
    "WaveletListPlot" => Some(wa::plots::wavelet_list_plot_ast(args)),
    "WaveletMatrixPlot" => Some(wa::plots::wavelet_matrix_plot_ast(args)),
    "WaveletImagePlot" => Some(wa::plots::wavelet_image_plot_ast(args)),
    // Normal[dwd] / Normal[cwd] gives the list of coefficient rules.
    "Normal" if args.len() == 1 => {
      if wa::data::Dwd::from_expr(&args[0]).is_some() {
        Some(crate::evaluator::function_application::apply_curried_call(
          &args[0],
          &[Expr::Identifier("All".to_string())],
        ))
      } else if wa::continuous::Cwd::from_expr(&args[0]).is_some() {
        Some(crate::evaluator::function_application::apply_curried_call(
          &args[0],
          &[Expr::Identifier("All".to_string())],
        ))
      } else {
        None
      }
    }
    _ => None,
  }
}
