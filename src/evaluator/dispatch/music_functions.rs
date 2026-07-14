//! Dispatch for the Wolfram Language 15.0 ComputationalMusic functions.
//!
//! `MusicObjectQ` and `MusicPitch` carry real behavior; the remaining music
//! *objects* (MusicNote, MusicChord, …) are left as canonical symbolic
//! expressions by returning `None` here, matching how `Sound`/`SoundNote`
//! already behave.

#[allow(unused_imports)]
use super::*;
use crate::syntax::unevaluated;

pub(super) fn dispatch_music_functions(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "MusicObjectQ" if args.len() == 1 => {
      Some(Ok(crate::functions::music_ast::music_object_q(args)))
    }
    "MusicPitch" => crate::functions::music_ast::music_pitch(args).map(Ok),
    // MusicNote[pitch[, duration]] / MusicDuration[number] / MusicChord[name] /
    // MusicTimeSignature[n, d] / MusicRest[[duration]] canonicalize to their
    // WL 15 association forms; other arities/argument shapes are left as
    // canonical symbolic expressions.
    "MusicNote" if args.len() == 1 || args.len() == 2 => {
      crate::functions::music_ast::music_note(args).map(Ok)
    }
    "MusicDuration" if args.len() == 1 => {
      crate::functions::music_ast::music_duration(args).map(Ok)
    }
    "MusicChord" if args.len() == 1 => {
      crate::functions::music_ast::music_chord(args).map(Ok)
    }
    "MusicTimeSignature" if args.len() == 2 => {
      crate::functions::music_ast::music_time_signature(args).map(Ok)
    }
    "MusicRest" if args.len() <= 1 => {
      crate::functions::music_ast::music_rest(args).map(Ok)
    }
    // MusicMeasure[{events…}[, MusicTimeSignature[…]]] resolves its rhythm
    // against the meter (defaulting to common time); MusicVoice/MusicScore pack
    // their measures/voices into the WL 15 association forms.
    "MusicMeasure" if args.len() <= 2 => {
      crate::functions::music_ast::music_measure(args).map(Ok)
    }
    "MusicVoice" if args.len() <= 1 => {
      crate::functions::music_ast::music_voice(args).map(Ok)
    }
    "MusicScore" if args.len() <= 1 => {
      crate::functions::music_ast::music_score(args).map(Ok)
    }
    // MusicScale's second argument must be a property association; any other
    // second argument emits MusicScale::passc and stays unevaluated.
    "MusicScale" => crate::functions::music_ast::music_scale(args).map(Ok),
    // MusicPlot[obj] draws the object as staff notation, returning a Graphics
    // (which renders as the SVG in visual hosts and as `-Graphics-` in the CLI,
    // just like Plot). An invalid MusicScale (rejected by MusicScale::passc)
    // is not a valid music object: it emits MusicPlot::music and stays
    // unevaluated. Other non-renderable arguments are left symbolic.
    "MusicPlot" if args.len() == 1 => {
      if crate::functions::music_ast::is_invalid_music_scale(&args[0]) {
        crate::functions::music_ast::emit_music_plot_message(&args[0]);
        return Some(Ok(unevaluated("MusicPlot", args)));
      }
      crate::functions::music_render::music_to_svg(&args[0])
        .map(|svg| Ok(crate::graphics_result(svg)))
    }
    _ => None,
  }
}
