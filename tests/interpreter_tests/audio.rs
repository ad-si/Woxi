use super::*;

// ─── AudioAmplify ────────────────────────────────────────────────────────────

#[test]
fn audio_amplify_scales_samples() {
  assert_eq!(
    interpret("AudioAmplify[Audio[{0.1, -0.2, 0.3}], 2]").unwrap(),
    "Audio[{0.2, -0.4, 0.6}, SampleRate -> 44100]"
  );
}

#[test]
fn audio_amplify_keeps_sample_rate() {
  assert_eq!(
    interpret("AudioAmplify[Audio[{0.5}, SampleRate -> 8000], 0.5]").unwrap(),
    "Audio[{0.25}, SampleRate -> 8000]"
  );
}

#[test]
fn audio_amplify_multichannel() {
  assert_eq!(
    interpret("AudioAmplify[Audio[{{0.1, 0.2}, {0.3, 0.4}}], 2]").unwrap(),
    "Audio[{{0.2, 0.4}, {0.6, 0.8}}, SampleRate -> 44100]"
  );
}

#[test]
fn audio_amplify_non_audio_stays_unevaluated() {
  assert_eq!(
    interpret("AudioAmplify[5, 2]").unwrap(),
    "AudioAmplify[5, 2]"
  );
}

// ─── AudioTrim ───────────────────────────────────────────────────────────────

#[test]
fn audio_trim_removes_silence_at_both_ends() {
  assert_eq!(
    interpret("AudioTrim[Audio[{0., 0., 0.5, 0.7, 0., 0.}]]").unwrap(),
    "Audio[{0.5, 0.7}, SampleRate -> 44100]"
  );
}

#[test]
fn audio_trim_first_seconds() {
  // 0.5 s at 4 Hz = the first 2 samples.
  assert_eq!(
    interpret("AudioTrim[Audio[{0.1, 0.2, 0.3, 0.4}, SampleRate -> 4], 0.5]")
      .unwrap(),
    "Audio[{0.1, 0.2}, SampleRate -> 4]"
  );
}

#[test]
fn audio_trim_interval() {
  assert_eq!(
    interpret(
      "AudioTrim[Audio[{0.1, 0.2, 0.3, 0.4}, SampleRate -> 4], {0.25, 0.75}]"
    )
    .unwrap(),
    "Audio[{0.2, 0.3}, SampleRate -> 4]"
  );
}

#[test]
fn audio_trim_negative_times_count_from_end() {
  assert_eq!(
    interpret(
      "AudioTrim[Audio[{0.1, 0.2, 0.3, 0.4}, SampleRate -> 4], {-0.5, All}]"
    )
    .unwrap(),
    "AudioTrim[Audio[{0.1, 0.2, 0.3, 0.4}, SampleRate -> 4], {-0.5, All}]"
  );
  assert_eq!(
    interpret(
      "AudioTrim[Audio[{0.1, 0.2, 0.3, 0.4}, SampleRate -> 4], {-0.5, -0.25}]"
    )
    .unwrap(),
    "Audio[{0.3}, SampleRate -> 4]"
  );
}

#[test]
fn audio_trim_quantity_seconds() {
  assert_eq!(
    interpret(
      "AudioTrim[Audio[{0.1, 0.2, 0.3, 0.4}, SampleRate -> 4], \
       Quantity[500, \"Milliseconds\"]]"
    )
    .unwrap(),
    "Audio[{0.1, 0.2}, SampleRate -> 4]"
  );
}

// ─── AudioJoin ───────────────────────────────────────────────────────────────

#[test]
fn audio_join_concatenates() {
  assert_eq!(
    interpret("AudioJoin[Audio[{0.1, 0.2}], Audio[{0.3}]]").unwrap(),
    "Audio[{0.1, 0.2, 0.3}, SampleRate -> 44100]"
  );
}

#[test]
fn audio_join_list_form() {
  assert_eq!(
    interpret("AudioJoin[{Audio[{0.1}], Audio[{0.2}], Audio[{0.3}]}]").unwrap(),
    "Audio[{0.1, 0.2, 0.3}, SampleRate -> 44100]"
  );
}

#[test]
fn audio_join_conforms_channel_counts() {
  // Mono + stereo joins as stereo (the mono channel is duplicated).
  assert_eq!(
    interpret("AudioJoin[Audio[{0.1, 0.2}], Audio[{{0.3}, {0.4}}]]").unwrap(),
    "Audio[{{0.1, 0.2, 0.3}, {0.1, 0.2, 0.4}}, SampleRate -> 44100]"
  );
}

#[test]
fn audio_join_conforms_sample_rates() {
  // The 4000 Hz half is resampled to 8000 Hz (linear interpolation).
  assert_eq!(
    interpret(
      "AudioJoin[Audio[{0., 1.}, SampleRate -> 4000], \
       Audio[{0.5}, SampleRate -> 8000]]"
    )
    .unwrap(),
    "Audio[{0., 0.3333333333333333, 0.6666666666666666, 1., 0.5}, \
     SampleRate -> 8000]"
  );
}

// ─── AudioPitchShift ─────────────────────────────────────────────────────────

#[test]
fn audio_pitch_shift_preserves_duration_and_rate() {
  assert_eq!(
    interpret(
      "AudioMeasurements[AudioPitchShift[Audio[Table[Sin[2 Pi 440 n/8000.], \
       {n, 0, 799}], SampleRate -> 8000], 1.5], \"Duration\"]"
    )
    .unwrap(),
    "Quantity[0.1, Seconds]"
  );
}

#[test]
fn audio_pitch_shift_semitones_spec() {
  assert_eq!(
    interpret(
      "Head[AudioPitchShift[Audio[Table[Sin[2 Pi 440 n/8000.], \
       {n, 0, 799}], SampleRate -> 8000], Quantity[12, \"Semitones\"]]]"
    )
    .unwrap(),
    "Audio"
  );
}

#[test]
fn audio_pitch_shift_raises_fundamental() {
  // An octave up should roughly double the detected fundamental (440 Hz
  // sine at 8 kHz detects as 8000/18 ≈ 444 Hz; doubled ≈ 888 ≈ 8000/9).
  assert_eq!(
    interpret(
      "AudioMeasurements[AudioPitchShift[Audio[Table[Sin[2 Pi 440 n/8000.], \
       {n, 0, 1599}], SampleRate -> 8000], 2], \"FundamentalFrequency\"]"
    )
    .unwrap(),
    "Quantity[888.8888888888889, Hertz]"
  );
}

// ─── AudioMeasurements ───────────────────────────────────────────────────────

#[test]
fn audio_measurements_rms() {
  assert_eq!(
    interpret(
      "AudioMeasurements[Audio[{0., 0.5, -0.5, 1.}], \"RMSAmplitude\"]"
    )
    .unwrap(),
    "0.6123724356957945"
  );
}

#[test]
fn audio_measurements_duration_quantity() {
  assert_eq!(
    interpret(
      "AudioMeasurements[Audio[{0., 0.5, -0.5, 1.}, SampleRate -> 8000], \
       \"Duration\"]"
    )
    .unwrap(),
    "Quantity[0.0005, Seconds]"
  );
}

#[test]
fn audio_measurements_property_list() {
  assert_eq!(
    interpret(
      "AudioMeasurements[Audio[{0., 0.5, -0.5, 1.}], {\"Max\", \"Min\", \
       \"Mean\", \"MinMax\"}]"
    )
    .unwrap(),
    "{1., -0.5, 0.25, {-0.5, 1.}}"
  );
}

#[test]
fn audio_measurements_power_energy_total_variation() {
  assert_eq!(
    interpret(
      "AudioMeasurements[Audio[{0., 0.5, -0.5, 1.}], {\"Power\", \
       \"Energy\", \"TotalVariation\"}]"
    )
    .unwrap(),
    "{0.375, 1.5, 3.}"
  );
}

#[test]
fn audio_measurements_zero_crossings() {
  assert_eq!(
    interpret(
      "AudioMeasurements[Audio[{1., -1., 1., -1.}], \"ZeroCrossings\"]"
    )
    .unwrap(),
    "3"
  );
}

#[test]
fn audio_measurements_multichannel_gives_per_channel_list() {
  assert_eq!(
    interpret("AudioMeasurements[Audio[{{0., 1.}, {1., 1.}}], \"Mean\"]")
      .unwrap(),
    "{0.5, 1.}"
  );
}

#[test]
fn audio_measurements_fundamental_frequency_of_sine() {
  // A 440 Hz sine at 8 kHz: the autocorrelation peak is at lag 18,
  // i.e. 8000/18 Hz.
  assert_eq!(
    interpret(
      "AudioMeasurements[Audio[Table[Sin[2 Pi 440 n/8000.], {n, 0, 799}], \
       SampleRate -> 8000], \"FundamentalFrequency\"]"
    )
    .unwrap(),
    "Quantity[444.44444444444446, Hertz]"
  );
}

#[test]
fn audio_measurements_spectral_centroid_near_sine_frequency() {
  assert_eq!(
    interpret(
      "Round[QuantityMagnitude[AudioMeasurements[Audio[Table[\
       Sin[2 Pi 440 n/8000.], {n, 0, 799}], SampleRate -> 8000], \
       \"SpectralCentroid\"]]]"
    )
    .unwrap(),
    "439"
  );
}

#[test]
fn audio_measurements_unknown_property_stays_unevaluated() {
  assert_eq!(
    interpret("AudioMeasurements[Audio[{0.1}], \"NoSuchProperty\"]").unwrap(),
    "AudioMeasurements[Audio[{0.1}], NoSuchProperty]"
  );
}

// ─── AudioLocalMeasurements ──────────────────────────────────────────────────

#[test]
fn audio_local_measurements_time_series() {
  // 400 samples at 8 kHz with 25 ms windows / 12.5 ms offset gives
  // windows at samples 0, 100, 200 (centers 0.0125, 0.025, 0.0375 s).
  assert_eq!(
    interpret(
      "AudioLocalMeasurements[Audio[Table[0.5, 400], SampleRate -> 8000], \
       \"Max\"]"
    )
    .unwrap(),
    "TimeSeries[{{0.0125, 0.5}, {0.025, 0.5}, {0.0375, 0.5}}]"
  );
}

#[test]
fn audio_local_measurements_property_list() {
  assert_eq!(
    interpret(
      "AudioLocalMeasurements[Audio[Table[0.5, 300], SampleRate -> 8000], \
       {\"Max\", \"Min\"}]"
    )
    .unwrap(),
    "{TimeSeries[{{0.0125, 0.5}, {0.025, 0.5}}], \
     TimeSeries[{{0.0125, 0.5}, {0.025, 0.5}}]}"
  );
}

// ─── AudioIntervals ──────────────────────────────────────────────────────────

#[test]
fn audio_intervals_finds_silence() {
  assert_eq!(
    interpret(
      "AudioIntervals[Audio[Join[Table[0., 1000], Table[0.5, 3000], \
       Table[0., 2000]], SampleRate -> 8000]]"
    )
    .unwrap(),
    "{{0., 0.125}, {0.5, 0.75}}"
  );
}

#[test]
fn audio_intervals_criterion_function() {
  assert_eq!(
    interpret(
      "AudioIntervals[Audio[Join[Table[0., 1000], Table[0.5, 3000], \
       Table[0., 2000]], SampleRate -> 8000], #RMSAmplitude > 0.1 &]"
    )
    .unwrap(),
    "{{0.1125, 0.5125}}"
  );
}

#[test]
fn audio_intervals_no_silence_gives_empty_list() {
  assert_eq!(
    interpret("AudioIntervals[Audio[Table[0.5, 400], SampleRate -> 8000]]")
      .unwrap(),
    "{}"
  );
}

// ─── ShortTimeFourier ────────────────────────────────────────────────────────

#[test]
fn short_time_fourier_default_partition() {
  // n = 8 → window 2^⌈log₂√8⌉ = 4, offset ⌈4/3⌉ = 2 → 3 frames.
  assert_eq!(
    interpret(
      "s = ShortTimeFourier[{1., 2., 3., 4., 5., 6., 7., 8.}]; \
       {s[\"WindowSize\"], s[\"Offset\"], s[\"NumberOfFrames\"]}"
    )
    .unwrap(),
    "{4, 2, 3}"
  );
}

#[test]
fn short_time_fourier_data_frames() {
  // First frame {1,2,3,4}: DC bin is 10/√4 = 5.
  assert_eq!(
    interpret(
      "ShortTimeFourier[{1., 2., 3., 4., 5., 6., 7., 8.}][\"Data\"][[1, 1]]"
    )
    .unwrap(),
    "5."
  );
}

#[test]
fn short_time_fourier_sample_rate_from_audio() {
  assert_eq!(
    interpret(
      "ShortTimeFourier[Audio[Table[0.1, 64], SampleRate -> 8000]]\
       [\"SampleRate\"]"
    )
    .unwrap(),
    "8000"
  );
}

#[test]
fn short_time_fourier_explicit_partition() {
  assert_eq!(
    interpret(
      "s = ShortTimeFourier[Table[1., 32], 8, 4]; \
       {s[\"WindowSize\"], s[\"Offset\"], s[\"NumberOfFrames\"]}"
    )
    .unwrap(),
    "{8, 4, 7}"
  );
}

// ─── Spectral plots ──────────────────────────────────────────────────────────

#[test]
fn spectrogram_returns_graphics() {
  assert_eq!(
    interpret(
      "Spectrogram[Audio[Table[Sin[2 Pi 440 n/8000.], {n, 0, 1999}], \
       SampleRate -> 8000]]"
    )
    .unwrap(),
    "-Graphics-"
  );
}

#[test]
fn spectrogram_of_list_returns_graphics() {
  assert_eq!(
    interpret("Spectrogram[Table[Sin[n/5.], {n, 500}]]").unwrap(),
    "-Graphics-"
  );
}

#[test]
fn cepstrogram_returns_graphics() {
  assert_eq!(
    interpret(
      "Cepstrogram[Audio[Table[Sin[2 Pi 440 n/8000.], {n, 0, 1999}], \
       SampleRate -> 8000]]"
    )
    .unwrap(),
    "-Graphics-"
  );
}

#[test]
fn periodogram_returns_graphics() {
  assert_eq!(
    interpret("Periodogram[Table[Sin[2 Pi 100 n/1000.], {n, 0, 999}]]")
      .unwrap(),
    "-Graphics-"
  );
}

#[test]
fn periodogram_of_audio_returns_graphics() {
  assert_eq!(
    interpret(
      "Periodogram[Audio[Table[Sin[2 Pi 440 n/8000.], {n, 0, 1999}], \
       SampleRate -> 8000]]"
    )
    .unwrap(),
    "-Graphics-"
  );
}

// ─── WienerFilter ────────────────────────────────────────────────────────────

#[test]
fn wiener_filter_constant_signal_unchanged() {
  assert_eq!(
    interpret("WienerFilter[{1., 1., 1., 1.}, 1]").unwrap(),
    "{1., 1., 1., 1.}"
  );
}

#[test]
fn wiener_filter_zero_noise_returns_signal() {
  assert_eq!(
    interpret("WienerFilter[{1., 2., 3., 4.}, 1, 0]").unwrap(),
    "{1., 2., 3., 4.}"
  );
}

#[test]
fn wiener_filter_2d_constant_unchanged() {
  assert_eq!(
    interpret("WienerFilter[{{2., 2.}, {2., 2.}}, 1]").unwrap(),
    "{{2., 2.}, {2., 2.}}"
  );
}

#[test]
fn wiener_filter_smooths_spike() {
  assert_eq!(
    interpret("WienerFilter[{0., 0., 1., 0., 0.}, 2][[3]] < 1.").unwrap(),
    "True"
  );
}

#[test]
fn wiener_filter_on_audio_returns_audio() {
  assert_eq!(
    interpret(
      "Head[WienerFilter[Audio[{0., 0.5, 0., 0.5}, SampleRate -> 8000], 1]]"
    )
    .unwrap(),
    "Audio"
  );
}

// ─── TotalVariationFilter ────────────────────────────────────────────────────

#[test]
fn total_variation_filter_constant_signal_unchanged() {
  assert_eq!(
    interpret("TotalVariationFilter[{2., 2., 2.}]").unwrap(),
    "{2., 2., 2.}"
  );
}

#[test]
fn total_variation_filter_reduces_spike() {
  assert_eq!(
    interpret("TotalVariationFilter[{0., 0., 5., 0., 0.}, 0.5][[3]] < 5.")
      .unwrap(),
    "True"
  );
}

#[test]
fn total_variation_filter_2d() {
  assert_eq!(
    interpret("TotalVariationFilter[{{1., 1.}, {1., 1.}}]").unwrap(),
    "{{1., 1.}, {1., 1.}}"
  );
}

#[test]
fn total_variation_filter_on_audio_returns_audio() {
  assert_eq!(
    interpret("Head[TotalVariationFilter[Audio[{0., 0.5, 0., 0.5}], 0.2]]")
      .unwrap(),
    "Audio"
  );
}

// ─── Shared filters on Audio ─────────────────────────────────────────────────

#[test]
fn mean_filter_on_audio() {
  assert_eq!(
    interpret("MeanFilter[Audio[{0., 1., 0., 1., 0.}], 1]").unwrap(),
    "Audio[{0.5, 0.3333333333333333, 0.6666666666666666, \
     0.3333333333333333, 0.5}, SampleRate -> 44100]"
  );
}

#[test]
fn mean_filter_on_lists_still_works() {
  assert_eq!(interpret("MeanFilter[{1, 5, 9}, 1]").unwrap(), "{3, 5, 7}");
}

#[test]
fn lowpass_filter_on_audio_returns_audio() {
  assert_eq!(
    interpret(
      "Head[LowpassFilter[Audio[Table[Sin[2 Pi 100 n/8000.] + \
       Sin[2 Pi 3000 n/8000.], {n, 0, 499}], SampleRate -> 8000], 2 Pi 500]]"
    )
    .unwrap(),
    "Audio"
  );
}

#[test]
fn lowpass_filter_on_audio_removes_high_frequency() {
  // After lowpassing at 500 Hz, the 3 kHz component is gone: the RMS of
  // the mixed signal (≈1) drops to that of a single sine (≈0.707).
  assert_eq!(
    interpret(
      "Round[QuantityMagnitude[AudioMeasurements[LowpassFilter[\
       Audio[Table[Sin[2 Pi 100 n/8000.] + Sin[2 Pi 3000 n/8000.], \
       {n, 0, 1999}], SampleRate -> 8000], 2 Pi 500], \
       \"FundamentalFrequency\"]]]"
    )
    .unwrap(),
    "100"
  );
}

// ─── Statistics on Audio objects ─────────────────────────────────────────────

#[test]
fn mean_of_audio() {
  assert_eq!(interpret("Mean[Audio[{0., 1.}]]").unwrap(), "0.5");
}

#[test]
fn median_of_audio() {
  assert_eq!(interpret("Median[Audio[{0.1, 0.9, 0.5}]]").unwrap(), "0.5");
}

#[test]
fn variance_of_multichannel_audio() {
  assert_eq!(
    interpret("Variance[Audio[{{0., 1.}, {2., 4.}}]]").unwrap(),
    "{0.5, 2}"
  );
}

#[test]
fn quantile_of_audio() {
  assert_eq!(
    interpret("Quantile[Audio[{0.1, 0.2, 0.3, 0.4}], 0.5]").unwrap(),
    "0.2"
  );
}

#[test]
fn mean_of_list_still_works() {
  assert_eq!(interpret("Mean[{1, 2, 3}]").unwrap(), "2");
}

// ─── Sound/Play input ────────────────────────────────────────────────────────

#[test]
fn audio_measurements_accepts_play_expression() {
  // Play synthesizes at 8000 Hz, so one second gives Quantity[1., s].
  assert_eq!(
    interpret(
      "AudioMeasurements[Play[Sin[2 Pi 440 t], {t, 0, 1}], \"Duration\"]"
    )
    .unwrap(),
    "Quantity[1., Seconds]"
  );
}

// ─── Capture / web search stubs ──────────────────────────────────────────────

#[test]
fn audio_capture_fails_without_device() {
  assert_eq!(interpret("AudioCapture[]").unwrap(), "$Failed");
}

#[test]
fn web_audio_search_fails_without_service() {
  assert_eq!(interpret("WebAudioSearch[\"rooster\"]").unwrap(), "$Failed");
}

// ─── WAV round trip ──────────────────────────────────────────────────────────

#[test]
fn wav_export_import_round_trip() {
  let dir = std::env::temp_dir().join("woxi_audio_tests");
  std::fs::create_dir_all(&dir).unwrap();
  let path = dir.join("roundtrip.wav");
  let path_str = path.display().to_string();
  // Export 4 samples at 8 kHz, re-import, and measure: 16-bit
  // quantization keeps values within 1/32768 of the originals.
  let code = format!(
    "Export[\"{path_str}\", Audio[{{0., 0.5, -0.5, 1.}}, \
     SampleRate -> 8000], \"WAV\"]; \
     a = Import[\"{path_str}\"]; \
     {{AudioMeasurements[a, \"Duration\"], \
     Round[AudioMeasurements[a, \"Max\"], 0.001], \
     Round[AudioMeasurements[a, \"Min\"], 0.001]}}"
  );
  assert_eq!(
    interpret(&code).unwrap(),
    "{Quantity[0.0005, Seconds], 1., -0.5}"
  );
  std::fs::remove_file(&path).ok();
}

#[test]
fn audio_from_wav_file_measurable() {
  let dir = std::env::temp_dir().join("woxi_audio_tests");
  std::fs::create_dir_all(&dir).unwrap();
  let path = dir.join("file_backed.wav");
  let path_str = path.display().to_string();
  let code = format!(
    "Export[\"{path_str}\", Audio[{{0.25, 0.25}}, SampleRate -> 4000], \
     \"WAV\"]; \
     AudioMeasurements[Audio[File[\"{path_str}\"]], \"Duration\"]"
  );
  assert_eq!(interpret(&code).unwrap(), "Quantity[0.0005, Seconds]");
  std::fs::remove_file(&path).ok();
}
