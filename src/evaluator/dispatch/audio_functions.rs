//! Dispatch for the audio-processing heads: editing (AudioAmplify,
//! AudioTrim, AudioJoin, AudioPitchShift), analysis (AudioMeasurements,
//! AudioLocalMeasurements, AudioIntervals), the short-time Fourier
//! transform (ShortTimeFourier), the spectral plots (Spectrogram,
//! Periodogram, Cepstrogram), the noise-removal filters (WienerFilter,
//! TotalVariationFilter), and the Audio paths of shared functions
//! (Mean/Median/Variance/Quantile, MeanFilter, LowpassFilter).

use crate::InterpreterError;
use crate::functions::audio_ast as audio;
use crate::syntax::Expr;

pub fn dispatch_audio_functions(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "AudioAmplify" => Some(audio::edit::audio_amplify_ast(args)),
    "AudioTrim" => Some(audio::edit::audio_trim_ast(args)),
    "AudioJoin" => Some(audio::edit::audio_join_ast(args)),
    "AudioPitchShift" => Some(audio::edit::audio_pitch_shift_ast(args)),
    "AudioMeasurements" => Some(audio::measure::audio_measurements_ast(args)),
    "AudioLocalMeasurements" => {
      Some(audio::measure::audio_local_measurements_ast(args))
    }
    "AudioIntervals" => Some(audio::measure::audio_intervals_ast(args)),
    "ShortTimeFourier" => Some(audio::spectral::short_time_fourier_ast(args)),
    "Spectrogram" => Some(audio::spectral::spectrogram_ast(args)),
    "Cepstrogram" => Some(audio::spectral::cepstrogram_ast(args)),
    "Periodogram" => Some(audio::spectral::periodogram_ast(args)),
    "WienerFilter" => Some(audio::filters::wiener_filter_ast(args)),
    "TotalVariationFilter" => {
      Some(audio::filters::total_variation_filter_ast(args))
    }
    // Shared heads: only claimed when the first argument is an audio
    // object; otherwise the regular list/image dispatch continues.
    "MeanFilter" => audio::filters::mean_filter_audio_ast(args),
    "LowpassFilter" => audio::filters::lowpass_filter_audio_ast(args),
    "Mean" | "Median" | "Variance" | "Quantile" => {
      audio::measure::audio_stat_ast(name, args)
    }
    // AudioCapture[] records from a microphone; this environment has no
    // audio input device, so it fails like wolframscript does headless.
    "AudioCapture" if args.is_empty() => {
      Some(Ok(Expr::Identifier("$Failed".to_string())))
    }
    // WebAudioSearch requires the paid web audio search service; without
    // service credentials it fails.
    "WebAudioSearch" if !args.is_empty() => {
      Some(Ok(Expr::Identifier("$Failed".to_string())))
    }
    _ => None,
  }
}
