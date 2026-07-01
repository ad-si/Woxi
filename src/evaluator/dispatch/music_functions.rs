//! Dispatch for the Wolfram Language 15.0 ComputationalMusic functions.
//!
//! `MusicObjectQ` and `MusicPitch` carry real behavior; the remaining music
//! *objects* (MusicNote, MusicChord, …) are left as canonical symbolic
//! expressions by returning `None` here, matching how `Sound`/`SoundNote`
//! already behave.

#[allow(unused_imports)]
use super::*;

pub fn dispatch_music_functions(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "MusicObjectQ" if args.len() == 1 => {
      Some(Ok(crate::functions::music_ast::music_object_q(args)))
    }
    "MusicPitch" => crate::functions::music_ast::music_pitch(args).map(Ok),
    // MusicNote[pitch, duration] / MusicDuration[number] / MusicChord[name]
    // canonicalize to their WL 15 association forms; other arities/argument
    // shapes are left as canonical symbolic expressions.
    "MusicNote" if args.len() == 2 => {
      crate::functions::music_ast::music_note(args).map(Ok)
    }
    "MusicDuration" if args.len() == 1 => {
      crate::functions::music_ast::music_duration(args).map(Ok)
    }
    "MusicChord" if args.len() == 1 => {
      crate::functions::music_ast::music_chord(args).map(Ok)
    }
    // MusicPlot[obj] draws the object as staff notation, returning a Graphics
    // (which renders as the SVG in visual hosts and as `-Graphics-` in the CLI,
    // just like Plot). Non-renderable arguments are left symbolic.
    "MusicPlot" if args.len() == 1 => {
      crate::functions::music_render::music_to_svg(&args[0])
        .map(|svg| Ok(crate::graphics_result(svg)))
    }
    _ => None,
  }
}
