//! Piano-roll display of the Wolfram Language 15.0 ComputationalMusic objects.
//!
//! `MusicPlot[obj]` draws `obj` as a piano roll — one rounded bar per note,
//! placed at its onset (in whole notes) and MIDI pitch — built as the same
//! `Graphics` expression the Wolfram Language produces and rendered by the
//! regular graphics pipeline. A `MusicScore` is displayed as a summary panel:
//! play/stop buttons, a piano-roll thumbnail of its voices, and its duration
//! and time signature.
//!
//! `MusicNotation` is a Woxi extension (not yet part of the Wolfram Language)
//! that switches either display to staff notation:
//! `MusicScore[{…}, MusicNotation -> "SheetMusic"]` and
//! `MusicPlot[obj, MusicNotation -> "SheetMusic"]`.

#[allow(unused_imports)]
use super::*;
use crate::functions::graphics::theme;
use crate::functions::music_render::{
  TimedEvent, VoiceTimeline, score_voices, voice_timeline,
};

/// How a music object is displayed (the value of the `MusicNotation` option).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Notation {
  /// `Automatic` or `"PianoRoll"`: bars on a time × pitch grid.
  PianoRoll,
  /// `"SheetMusic"`: staff notation.
  SheetMusic,
}

/// Parse a `MusicNotation` option value, or `None` for an unknown value.
fn parse_notation(value: &Expr) -> Option<Notation> {
  match value {
    Expr::Identifier(s) if s == "Automatic" => Some(Notation::PianoRoll),
    Expr::String(s) if s == "PianoRoll" => Some(Notation::PianoRoll),
    Expr::String(s) if s == "SheetMusic" => Some(Notation::SheetMusic),
    _ => None,
  }
}

/// The `MusicNotation` stored in a resolved `MusicScore[<|…|>]`, if any.
pub fn stored_notation(expr: &Expr) -> Option<Notation> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "MusicScore" {
    return None;
  }
  let Some(Expr::Association(pairs)) = args.first() else {
    return None;
  };
  pairs.iter().find_map(|(k, v)| match k {
    Expr::String(n) if n == "MusicNotation" => parse_notation(v),
    _ => None,
  })
}

/// The colors of successive voices (the Wolfram Language's music palette).
const VOICE_COLORS: [(f64, f64, f64); 8] = [
  (0.24, 0.6, 0.8),
  (0.95, 0.627, 0.1425),
  (0.455, 0.7, 0.21),
  (0.922526, 0.385626, 0.209179),
  (0.578, 0.51, 0.85),
  (0.772079, 0.431554, 0.102387),
  (0.4, 0.64, 1.0),
  (1.0, 0.75, 0.0),
];

/// Half the height of a note bar, in semitones.
const BAR_HALF_HEIGHT: f64 = 0.4;

/// A music object laid out in time, voice by voice.
struct Roll {
  voices: Vec<VoiceTimeline>,
  /// Measure boundaries in whole notes: `0`, then the end of every measure.
  measure_lines: Vec<f64>,
  /// Total length in whole notes.
  length: f64,
  /// Lowest and highest MIDI number sounded.
  pitch_range: (i128, i128),
}

impl Roll {
  fn measure_count(&self) -> usize {
    self.measure_lines.len().saturating_sub(1)
  }

  /// The time signature of the first voice that states one (common time by
  /// default).
  fn time_signature(&self) -> (u32, u32) {
    self
      .voices
      .iter()
      .find_map(|v| v.time_signature)
      .unwrap_or((4, 4))
  }
}

/// The voices of a plottable music object: a score's voices, a voice or
/// measure itself, or a lone note/chord placed in a (padded) measure.
fn plot_voices(expr: &Expr) -> Option<Vec<Expr>> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  match name.as_str() {
    "MusicScore" => score_voices(args),
    "MusicVoice" | "MusicMeasure" => Some(vec![expr.clone()]),
    "MusicNote" | "MusicChord" => Some(vec![
      crate::functions::music_ast::music_measure(&[Expr::List(
        vec![expr.clone()].into(),
      )])
      .unwrap_or_else(|| expr.clone()),
    ]),
    _ => None,
  }
}

/// Lay out a music object for a piano roll, or `None` when it sounds no note.
fn roll_of(expr: &Expr) -> Option<Roll> {
  let voices: Vec<VoiceTimeline> =
    plot_voices(expr)?.iter().map(voice_timeline).collect();
  let pitches = voices
    .iter()
    .flat_map(|v| v.events.iter().flat_map(|e| e.midis.iter().copied()));
  let pitch_range = pitches.fold(None, |acc: Option<(i128, i128)>, m| {
    Some(acc.map_or((m, m), |(lo, hi)| (lo.min(m), hi.max(m))))
  })?;
  let length = voices
    .iter()
    .filter_map(|v| v.events.last().map(|e| e.onset + e.len))
    .fold(0.0, f64::max);
  // The measure grid is that of the voice with the most barlines.
  let barlines = voices
    .iter()
    .map(|v| &v.barlines)
    .max_by_key(|b| b.len())
    .cloned()
    .unwrap_or_default();
  let mut measure_lines = vec![0.0];
  measure_lines.extend(barlines.into_iter().filter(|&x| x > 0.0));
  if measure_lines.len() == 1 {
    measure_lines.push(length);
  }
  Some(Roll {
    voices,
    measure_lines,
    length,
    pitch_range,
  })
}

// ── MusicPlot ────────────────────────────────────────────────────────────────

/// An exact rational for a time position: rhythmic values are rationals with
/// small denominators, recovered from their `f64` form by continued fractions.
fn exact(x: f64) -> Expr {
  let (mut h0, mut h1, mut k0, mut k1) = (0_i128, 1_i128, 1_i128, 0_i128);
  let mut v = x;
  for _ in 0..32 {
    let a = v.floor();
    let ai = a as i128;
    (h0, h1) = (h1, ai * h1 + h0);
    (k0, k1) = (k1, ai * k1 + k0);
    if (h1 as f64 / k1 as f64 - x).abs() < 1e-9 || k1 > 1 << 20 {
      break;
    }
    v = 1.0 / (v - a);
  }
  crate::functions::math_ast::make_rational(h1, k1)
}

fn rgb(c: (f64, f64, f64)) -> Vec<Expr> {
  vec![Expr::Real(c.0), Expr::Real(c.1), Expr::Real(c.2)]
}

fn rule(name: &str, value: Expr) -> Expr {
  Expr::Rule {
    pattern: Box::new(Expr::Identifier(name.to_string())),
    replacement: Box::new(value),
  }
}

fn list(items: Vec<Expr>) -> Expr {
  Expr::List(items.into())
}

/// The primitives of one voice: a rounded rectangle per sounding tone and an
/// invisible line spanning each rest (so rests still count towards the plot
/// range), styled in the voice's color.
fn voice_primitives(voice: &VoiceTimeline, color: (f64, f64, f64)) -> Expr {
  let pitches: Vec<i128> =
    voice.events.iter().flat_map(|e| e.midis.clone()).collect();
  let rest_y = match (pitches.iter().min(), pitches.iter().max()) {
    (Some(lo), Some(hi)) => (lo + hi) as f64 / 2.0,
    _ => 60.0,
  };
  let mut prims = Vec::new();
  for TimedEvent { onset, len, midis } in &voice.events {
    let (x0, x1) = (exact(*onset), exact(onset + len));
    if midis.is_empty() {
      prims.push(list(vec![
        call1("Opacity", Expr::Integer(0)),
        call1(
          "Line",
          list(vec![
            list(vec![x0, Expr::Real(rest_y)]),
            list(vec![x1, Expr::Real(rest_y)]),
          ]),
        ),
      ]));
      continue;
    }
    for &m in midis {
      prims.push(call(
        "Rectangle",
        vec![
          list(vec![x0.clone(), Expr::Real(m as f64 - BAR_HALF_HEIGHT)]),
          list(vec![x1.clone(), Expr::Real(m as f64 + BAR_HALF_HEIGHT)]),
          rule(
            "RoundingRadius",
            list(vec![Expr::Real(0.2), Expr::Real(0.1)]),
          ),
        ],
      ));
    }
  }
  let mut face = rgb(color);
  face.push(Expr::Real(0.8));
  list(vec![
    call(
      "Directive",
      vec![
        call1("EdgeForm", call("RGBColor", rgb(color))),
        call1("FaceForm", call("RGBColor", face)),
      ],
    ),
    list(prims),
  ])
}

/// The integer points of `FindDivisions[{a, b}, n]`.
fn find_divisions(a: i128, b: i128, n: i128) -> Vec<i128> {
  let expr = call(
    "FindDivisions",
    vec![
      list(vec![Expr::Integer(a), Expr::Integer(b)]),
      Expr::Integer(n),
    ],
  );
  match &crate::evaluator::evaluate_expr_to_expr(&expr) {
    Ok(Expr::List(items)) => items
      .iter()
      .filter_map(|d| match d {
        Expr::Integer(k) => Some(*k),
        _ => None,
      })
      .collect(),
    _ => Vec::new(),
  }
}

/// The time-axis ticks: every measure start labelled with its measure number,
/// plus unlabelled quarter-note ticks (and eighth-note ticks for at most three
/// measures). Beyond six measures only a selection of measure numbers is
/// labelled.
fn time_ticks(roll: &Roll) -> Vec<Expr> {
  let tick = |x: f64, label: Expr, len: f64| {
    list(vec![
      exact(x),
      label,
      list(vec![Expr::Real(len), Expr::Integer(0)]),
    ])
  };
  let measures = roll.measure_count();
  let lines = &roll.measure_lines;
  if measures > 6 {
    let wanted = find_divisions(2, measures as i128 + 1, 10);
    return (1..=measures)
      .filter(|i| wanted.contains(&(*i as i128 + 1)))
      .map(|i| tick(lines[i], Expr::Integer(i as i128 + 1), 0.01))
      .collect();
  }
  let mut ticks = Vec::new();
  for i in 0..measures {
    let (start, end) = (lines[i], lines[i + 1]);
    ticks.push(tick(start, Expr::Integer(i as i128 + 1), 0.01));
    let mut steps = vec![0.25];
    if measures <= 3 {
      steps.push(0.125);
    }
    for step in steps {
      let mut x = start + step;
      while x <= end - step + 1e-9 {
        ticks.push(tick(x, Expr::String(String::new()), 0.005));
        x += step;
      }
    }
  }
  ticks.push(tick(
    lines[measures],
    Expr::Integer(measures as i128 + 1),
    0.01,
  ));
  ticks
}

/// The pitch-axis ticks: every C and G labelled with its name and octave.
fn pitch_ticks(lo: f64, hi: f64) -> Vec<Expr> {
  let mut ticks = Vec::new();
  let first = (lo / 12.0).floor() as i128 - 1;
  let last = (hi / 12.0).ceil() as i128 + 1;
  for k in first..=last {
    for (offset, letter) in [(0, "C"), (7, "G")] {
      let m = 12 * k + offset;
      if (lo..=hi).contains(&(m as f64)) {
        ticks.push(list(vec![
          Expr::Integer(m),
          Expr::String(format!("{letter}{}", k - 1)),
        ]));
      }
    }
  }
  ticks
}

/// The measure grid lines: every measure boundary, thinned out to at most
/// about eight lines for long pieces.
fn grid_lines(roll: &Roll) -> Vec<Expr> {
  let lines = &roll.measure_lines;
  let step = if lines.len() > 8 {
    lines.len().div_ceil(8)
  } else {
    1
  };
  lines.iter().step_by(step).map(|&x| exact(x)).collect()
}

/// `MusicPlot[obj, opts…]`: a piano roll of `obj` (or, with
/// `MusicNotation -> "SheetMusic"`, its staff notation). Other options are
/// passed on to the resulting `Graphics`. A music object that sounds no note
/// emits `MusicPlot::music` and stays unevaluated; a non-music argument is left
/// unevaluated.
pub fn music_plot(args: &[Expr]) -> Option<Result<Expr, InterpreterError>> {
  let (obj, opts) = args.split_first()?;
  let opts = crate::functions::music_ast::music_option_rules(opts)?;
  let mut notation = stored_notation(obj);
  let mut graphics_opts = Vec::new();
  for (name, value) in opts {
    if name == "MusicNotation" {
      notation = parse_notation(&value).or(notation);
    } else {
      graphics_opts.push(rule(&name, value));
    }
  }
  let is_music = matches!(obj, Expr::FunctionCall { name, .. }
    if crate::functions::music_ast::MUSIC_OBJECT_HEADS.contains(&name.as_str()));
  let invalid = || {
    crate::functions::music_ast::emit_music_plot_message(obj);
    Some(Ok(unevaluated("MusicPlot", args)))
  };

  if notation == Some(Notation::SheetMusic) {
    if crate::functions::music_ast::is_invalid_music_scale(obj) {
      return invalid();
    }
    return crate::functions::music_render::music_staff_svg(obj)
      .map(|svg| Ok(crate::graphics_result(svg)));
  }

  let Some(roll) = roll_of(obj) else {
    return if is_music { invalid() } else { None };
  };
  let content = list(
    roll
      .voices
      .iter()
      .enumerate()
      .rev()
      .map(|(i, v)| voice_primitives(v, VOICE_COLORS[i % VOICE_COLORS.len()]))
      .collect(),
  );

  // A narrow pitch range is shown in a fixed seven-semitone window around its
  // centre; a wider one spans the notes, the time axis sitting just below.
  let (lo, hi) = (roll.pitch_range.0 as f64, roll.pitch_range.1 as f64);
  let (origin_y, y_range, tick_range) = if hi - lo < 7.0 {
    let c = f64::midpoint(lo, hi);
    (
      c - 3.5,
      list(vec![Expr::Real(c - 3.5), Expr::Real(c + 3.5)]),
      (c - 3.5, c + 3.5),
    )
  } else {
    let (y_min, y_max) = (lo - BAR_HALF_HEIGHT, hi + BAR_HALF_HEIGHT);
    let origin = y_min - (y_max - y_min) / 11.0 + roll.length / 50.0;
    (
      origin,
      Expr::Identifier("Full".to_string()),
      (origin, y_max + 1.0),
    )
  };

  let mut graphics_args = vec![content];
  graphics_args.extend(graphics_opts);
  graphics_args.extend([
    rule(
      "AxesOrigin",
      list(vec![
        Expr::Identifier("Automatic".to_string()),
        Expr::Real(origin_y),
      ]),
    ),
    rule(
      "AspectRatio",
      crate::functions::math_ast::make_rational(1, 4),
    ),
    rule("Axes", bool_expr(true)),
    rule(
      "GridLines",
      list(vec![
        list(grid_lines(&roll)),
        Expr::Identifier("None".to_string()),
      ]),
    ),
    rule(
      "PlotRange",
      list(vec![Expr::Identifier("Full".to_string()), y_range]),
    ),
    rule(
      "Ticks",
      list(vec![
        list(time_ticks(&roll)),
        list(pitch_ticks(tick_range.0, tick_range.1)),
      ]),
    ),
  ]);
  // Like `Graphics[…]` itself, the plot stays symbolic during evaluation (so
  // `Show`, `Cases`, `Options`, … see its primitives) and is rendered at the
  // output stage.
  Some(Ok(call("Graphics", graphics_args)))
}

// ── Playback ─────────────────────────────────────────────────────────────────

/// Sample rate of the synthesized score audio (Hz).
const SYNTH_RATE: u32 = 22050;

/// The tempo of a score in quarter notes per minute: its `MusicTempo` (a
/// number, `Quantity[n, …]` or `MusicTempo[n]`), 120 by default.
fn score_tempo(score: &Expr) -> f64 {
  fn bpm(v: &Expr) -> Option<f64> {
    match v {
      Expr::FunctionCall { name, args }
        if (name == "Quantity" || name == "MusicTempo") && !args.is_empty() =>
      {
        bpm(&args[0])
      }
      other => crate::functions::math_ast::try_eval_to_f64(other),
    }
  }
  let Expr::FunctionCall { args, .. } = score else {
    return 120.0;
  };
  let Some(Expr::Association(pairs)) = args.first() else {
    return 120.0;
  };
  pairs
    .iter()
    .find_map(|(k, v)| match k {
      Expr::String(n) if n == "MusicTempo" => bpm(v),
      _ => None,
    })
    .filter(|t| *t > 0.0)
    .unwrap_or(120.0)
}

/// Synthesize a music object to mono samples: every tone a few decaying
/// harmonics (a soft, piano-like timbre) shaped by a short attack and release,
/// all voices mixed and normalized.
fn synthesize(roll: &Roll, tempo: f64) -> Vec<f64> {
  let rate = SYNTH_RATE as f64;
  let whole_secs = 4.0 * 60.0 / tempo;
  const RELEASE: f64 = 0.04;
  let total = ((roll.length * whole_secs + RELEASE) * rate).ceil() as usize;
  let mut out = vec![0.0; total];
  for voice in &roll.voices {
    for event in &voice.events {
      let start = (event.onset * whole_secs * rate) as usize;
      let secs = event.len * whole_secs;
      let n = ((secs + RELEASE) * rate) as usize;
      for &m in &event.midis {
        let freq = 440.0 * 2f64.powf((m - 69) as f64 / 12.0);
        for i in 0..n.min(total.saturating_sub(start)) {
          let t = i as f64 / rate;
          let attack = (t / 0.01).min(1.0);
          let release = if t > secs {
            (1.0 - (t - secs) / RELEASE).max(0.0)
          } else {
            1.0
          };
          let phase = 2.0 * std::f64::consts::PI * freq * t;
          let tone = phase.sin() * (-3.0 * t).exp()
            + 0.4 * (2.0 * phase).sin() * (-5.0 * t).exp()
            + 0.15 * (3.0 * phase).sin() * (-7.0 * t).exp();
          out[start + i] += attack * release * tone;
        }
      }
    }
  }
  let peak = out.iter().fold(0.0_f64, |a, s| a.max(s.abs()));
  if peak > 0.0 {
    for s in &mut out {
      *s *= 0.8 / peak;
    }
  }
  out
}

/// The audio a displayed `MusicScore` plays, as a base64 WAV — `None` when
/// the score sounds no note. Hosts wire it to the play and stop buttons
/// ([`PLAY_BUTTON`], [`STOP_BUTTON`]) in front of the score, whether it is
/// shown as its summary panel or as sheet music.
pub fn score_audio(score: &Expr) -> Option<String> {
  use base64::Engine;
  if !matches!(score, Expr::FunctionCall { name, .. } if name == "MusicScore") {
    return None;
  }
  let roll = roll_of(score)?;
  let samples = synthesize(&roll, score_tempo(score));
  let wav = crate::functions::sound::samples_to_mono_wav(&samples, SYNTH_RATE);
  Some(base64::engine::general_purpose::STANDARD.encode(&wav))
}

// ── MusicScore display ───────────────────────────────────────────────────────

/// The play/pause glyph markers of the play button: the triangle is shown and
/// the two pause bars hidden until [`score_svg_playing`] swaps them.
const PLAY_GLYPH: &str = "class=\"music-play\"";
const PAUSE_GLYPH_HIDDEN: &str = "class=\"music-pause\" display=\"none\"";

/// The play and stop buttons in front of a displayed score: a circle around
/// a triangle / rounded square each, grouped as `music-button`s (with a
/// `data-action`) so a host can make them clickable. The play button also
/// carries a hidden pause glyph (two bars, `music-pause`) that a host shows
/// in place of the triangle while the score plays, making it a pause button.
fn playback_buttons() -> String {
  let th = theme();
  let ((bx, play_y), (_, stop_y)) = (PLAY_BUTTON, STOP_BUTTON);
  let button = |action: &str, cy: f64, glyph: String| {
    format!(
      "<g class=\"music-button\" data-action=\"{action}\" \
       style=\"cursor:pointer\"><circle cx=\"{bx}\" cy=\"{cy}\" \
       r=\"{BUTTON_RADIUS}\" fill=\"#fff\" fill-opacity=\"0\" \
       stroke=\"{stroke}\" stroke-width=\"1.3\"/>{glyph}</g>",
      stroke = th.text_muted,
    )
  };
  let play = button(
    "play",
    play_y,
    format!(
      "<path {PLAY_GLYPH} d=\"M {x0:.2} {y0:.2} L {x1:.2} \
       {play_y:.2} L {x0:.2} {y1:.2} Z\" fill=\"{BUTTON_BLUE}\"/>\
       <g {PAUSE_GLYPH_HIDDEN} fill=\"{BUTTON_BLUE}\">\
       <rect x=\"{px0:.2}\" y=\"{py:.2}\" width=\"3.5\" height=\"11\" \
       rx=\"1\"/><rect x=\"{px1:.2}\" y=\"{py:.2}\" width=\"3.5\" \
       height=\"11\" rx=\"1\"/></g>",
      x0 = bx - 3.5,
      x1 = bx + 6.5,
      y0 = play_y - 6.5,
      y1 = play_y + 6.5,
      px0 = bx - 4.75,
      px1 = bx + 1.25,
      py = play_y - 5.5,
    ),
  );
  let stop = button(
    "stop",
    stop_y,
    format!(
      "<rect class=\"music-stop\" x=\"{sx:.2}\" y=\"{sy:.2}\" \
       width=\"11\" height=\"11\" rx=\"1.5\" fill=\"{BUTTON_BLUE}\"/>",
      sx = bx - 5.5,
      sy = stop_y - 5.5,
    ),
  );
  play + &stop
}

/// A displayed score's SVG as it looks while the score plays: the play
/// button shows its pause glyph instead of the triangle. For hosts that
/// cannot toggle the glyphs in place (e.g. ones that rasterize the SVG).
pub fn score_svg_playing(svg: &str) -> String {
  svg
    .replace(PLAY_GLYPH, &format!("{PLAY_GLYPH} display=\"none\""))
    .replace(PAUSE_GLYPH_HIDDEN, "class=\"music-pause\"")
}

/// A `MusicScore` shown as sheet music (`MusicNotation -> "SheetMusic"`): its
/// staff notation with the play and stop buttons in front of it, separated
/// by a divider. `None` when the score carries no notation.
pub fn score_sheet_music_svg(score: &Expr) -> Option<String> {
  use crate::functions::music_render::{
    music_staff_svg, nest_svg_at, svg_attr,
  };
  let staff = music_staff_svg(score)?;
  let (staff_w, staff_h) =
    (svg_attr(&staff, "width")?, svg_attr(&staff, "height")?);
  // Tall enough for both buttons; shorter notation is centred vertically.
  let height = staff_h.max(STOP_BUTTON.1 + BUTTON_RADIUS + 8.0);
  let staff_x = BUTTON_COLUMN_W + 8.0;
  let width = staff_x + staff_w;
  let mut buttons = String::new();
  if roll_of(score).is_some() {
    buttons = playback_buttons();
  }
  Some(format!(
    "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" \
     height=\"{height}\" viewBox=\"0 0 {width} {height}\">{buttons}\
     <line x1=\"{BUTTON_COLUMN_W}\" y1=\"0\" x2=\"{BUTTON_COLUMN_W}\" \
     y2=\"{height}\" stroke=\"{border}\"/>{staff}</svg>",
    border = theme().framed_border,
    staff = nest_svg_at(&staff, staff_x, (height - staff_h) / 2.0),
  ))
}

// ── MusicScore summary panel ─────────────────────────────────────────────────

/// Panel geometry, in SVG user units.
pub const PANEL_W: f64 = 440.0;
const ROLL_H: f64 = 124.0;
const FOOTER_LINE_H: f64 = 22.0;
const BUTTON_COLUMN_W: f64 = 46.0;
/// Centres of the play and stop buttons in front of a displayed score, for
/// hosts that make them clickable (see [`score_audio`]).
pub const PLAY_BUTTON: (f64, f64) =
  (BUTTON_COLUMN_W / 2.0, ROLL_H / 2.0 - 19.0);
pub const STOP_BUTTON: (f64, f64) =
  (BUTTON_COLUMN_W / 2.0, ROLL_H / 2.0 + 19.0);
/// Radius of the play and stop buttons.
pub const BUTTON_RADIUS: f64 = 11.0;
/// Fill of the play/stop button glyphs.
const BUTTON_BLUE: &str = "#3b7fc4";

fn svg_rgb((r, g, b): (f64, f64, f64)) -> String {
  let c = |v: f64| (v * 255.0).round() as u8;
  format!("rgb({},{},{})", c(r), c(g), c(b))
}

/// The summary panel of a `MusicScore`: play and stop buttons, a piano-roll
/// thumbnail with one color per voice, and a footer stating the score's length
/// in measures and its time signature. `None` when the score sounds no note.
pub fn score_panel_svg(score: &Expr) -> Option<String> {
  let roll = roll_of(score)?;
  let th = theme();
  let footer_h = 2.0 * FOOTER_LINE_H + 8.0;
  let height = ROLL_H + footer_h;
  let mut svg = format!(
    "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{PANEL_W}\" \
     height=\"{height}\" viewBox=\"0 0 {PANEL_W} {height}\">"
  );

  // Frame, footer band, and the divider right of the buttons.
  svg.push_str(&format!(
    "<clipPath id=\"music-panel\"><rect x=\"0.5\" y=\"0.5\" width=\"{w}\" \
     height=\"{h}\" rx=\"6\"/></clipPath>\
     <g clip-path=\"url(#music-panel)\">\
     <rect x=\"0\" y=\"{ROLL_H}\" width=\"{PANEL_W}\" height=\"{footer_h}\" \
     fill=\"{bg}\"/></g>\
     <rect x=\"0.5\" y=\"0.5\" width=\"{w}\" height=\"{h}\" rx=\"6\" \
     fill=\"none\" stroke=\"{border}\"/>\
     <line x1=\"0.5\" y1=\"{ROLL_H}\" x2=\"{x2}\" y2=\"{ROLL_H}\" \
     stroke=\"{border}\"/>\
     <line x1=\"{BUTTON_COLUMN_W}\" y1=\"0.5\" x2=\"{BUTTON_COLUMN_W}\" \
     y2=\"{ROLL_H}\" stroke=\"{border}\"/>",
    w = PANEL_W - 1.0,
    h = height - 1.0,
    x2 = PANEL_W - 0.5,
    bg = th.table_header_bg,
    border = th.framed_border,
  ));

  svg.push_str(&playback_buttons());

  // Piano-roll thumbnail: time runs across, pitch upwards; each note is a
  // thin bar with a small gap before the next.
  let (x_left, x_right) = (BUTTON_COLUMN_W + 14.0, PANEL_W - 14.0);
  let (y_top, y_bottom) = (18.0, ROLL_H - 18.0);
  let (lo, hi) = roll.pitch_range;
  let span = ((hi - lo) as f64).max(12.0);
  let mid = (lo + hi) as f64 / 2.0;
  let length = roll.length.max(f64::EPSILON);
  let px = |t: f64| x_left + t / length * (x_right - x_left);
  let py =
    |m: f64| y_bottom - (m - (mid - span / 2.0)) / span * (y_bottom - y_top);
  let bar_h = ((y_bottom - y_top) / span * 0.8).clamp(2.0, 4.0);
  for (i, voice) in roll.voices.iter().enumerate() {
    let color = svg_rgb(VOICE_COLORS[i % VOICE_COLORS.len()]);
    for event in &voice.events {
      let (x0, x1) = (px(event.onset) + 2.0, px(event.onset + event.len) - 2.0);
      let w = (x1 - x0).max(1.0);
      for &m in &event.midis {
        svg.push_str(&format!(
          "<rect class=\"music-note\" x=\"{x0:.2}\" y=\"{y:.2}\" \
           width=\"{w:.2}\" height=\"{bar_h:.2}\" fill=\"{color}\"/>",
          y = py(m as f64) - bar_h / 2.0,
        ));
      }
    }
  }

  // Footer text.
  let (num, den) = roll.time_signature();
  let lines = [
    format!("Duration: {} measures", roll.measure_count()),
    format!("Time Signature: {num}/{den}"),
  ];
  for (i, text) in lines.iter().enumerate() {
    svg.push_str(&format!(
      "<text x=\"12\" y=\"{y:.1}\" font-family=\"sans-serif\" \
       font-size=\"15\" fill=\"{}\">{text}</text>",
      th.text_secondary,
      y = ROLL_H + 4.0 + (i as f64 + 1.0) * FOOTER_LINE_H - 5.0,
    ));
  }
  svg.push_str("</svg>");
  Some(svg)
}
