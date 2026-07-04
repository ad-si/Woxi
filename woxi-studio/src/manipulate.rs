//! Woxi Studio helpers for rendering interactive `Manipulate[…]` cells.
//!
//! A held `Manipulate` expression is parsed into a `ManipulateState` that
//! stores the body (as InputForm source), the per-variable controls, the
//! current value of each control, and the currently rendered output
//! (SVG or text). On each slider/picklist change the state is updated
//! in place and the body is re-evaluated inside a `Block[{…}, body]` so
//! that free control variables are substituted.

use iced::widget::svg;
use woxi::functions::graphics::{
  DisplayNode, LabelRun, ManipulateControl, ManipulateSpec,
  apply_manipulate_mutations, build_manipulate_display, extract_control_spec,
  extract_list_animate_spec, extract_manipulate_spec,
};
use woxi::syntax::Expr;

/// Runtime state for a single control inside a Manipulate cell.
#[derive(Debug, Clone)]
pub enum ControlState {
  Continuous {
    name: String,
    label: String,
    /// The label split into styled runs, for rich-text (italic) rendering.
    label_runs: Vec<LabelRun>,
    min: f64,
    max: f64,
    /// Step size used by the slider. When the spec doesn't specify one
    /// we pick `(max - min) / 100`.
    step: f64,
    current: f64,
  },
  Discrete {
    name: String,
    label: String,
    label_runs: Vec<LabelRun>,
    /// Each entry is an InputForm fragment ready for substitution into the
    /// variable binding. For a rule-form choice `value -> "label"` this is the
    /// value (left side), not the whole rule.
    values: Vec<String>,
    /// The label shown for each choice, parallel to `values`. Equals `values`
    /// for plain choices; the rule's right side for rule-form choices.
    value_labels: Vec<String>,
    current_index: usize,
  },
  /// A 2D slider binding its variable to a `{x, y}` pair.
  Slider2D {
    name: String,
    label: String,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    x: f64,
    y: f64,
  },
  /// An interval slider binding its variable to a `{low, high}` pair.
  IntervalSlider {
    name: String,
    label: String,
    min: f64,
    max: f64,
    step: f64,
    low: f64,
    high: f64,
  },
}

impl ControlState {
  pub fn name(&self) -> &str {
    match self {
      ControlState::Continuous { name, .. } => name,
      ControlState::Discrete { name, .. } => name,
      ControlState::Slider2D { name, .. } => name,
      ControlState::IntervalSlider { name, .. } => name,
    }
  }

  /// InputForm fragment for the *current* value, for use inside a
  /// `Block[{name = <value>}, …]` binding.
  pub fn current_code(&self) -> String {
    match self {
      ControlState::Continuous { current, .. } => format_f64(*current),
      ControlState::Discrete {
        values,
        current_index,
        ..
      } => values
        .get(*current_index)
        .cloned()
        .unwrap_or_else(|| "Null".to_string()),
      ControlState::Slider2D { x, y, .. } => {
        format!("{{{}, {}}}", format_f64(*x), format_f64(*y))
      }
      ControlState::IntervalSlider { low, high, .. } => {
        format!("{{{}, {}}}", format_f64(*low), format_f64(*high))
      }
    }
  }
}

/// Full state for a Manipulate cell: the held body plus its rendered
/// output.
#[derive(Debug, Clone)]
pub struct ManipulateState {
  pub body: String,
  /// Initialization code from `Initialization :> …`. Prepended to every
  /// re-evaluation so that helper definitions introduced here remain in
  /// scope regardless of the slider state.
  pub initialization: Option<String>,
  pub controls: Vec<ControlState>,
  /// Mutable `ControlType -> None` state variables, `(name, current value as
  /// InputForm)`. Passed live in the binding set so interactive displays can
  /// rewrite them.
  pub state: Vec<(String, String)>,
  /// Extra display expressions (InputForm), e.g. a `Dynamic[Panel[Grid[…]]]`
  /// of checkboxes, rebuilt into `display_trees` on every re-evaluation.
  pub displays: Vec<String>,
  /// The rendered widget tree for each display element.
  pub display_trees: Vec<DisplayNode>,
  pub graphics_handle: Option<svg::Handle>,
  pub text_output: Option<String>,
  pub error: Option<String>,
  /// Per-control `Enabled` condition (InputForm code), parallel to `controls`.
  /// `None` means the control has no condition and is always enabled.
  control_enabled: Vec<Option<String>>,
  /// Whether each control is currently interactive, recomputed on every
  /// re-evaluation from `control_enabled` against the live bindings. Parallel
  /// to `controls`; a control indexes in with its position.
  pub control_is_enabled: Vec<bool>,
  /// Generation of the most recent control change. Bumped on every
  /// slider/picklist move so the throttled re-evaluation can tell whether a
  /// newer change has superseded a queued one.
  reeval_pending: u64,
  /// Generation last actually re-evaluated. When it lags `reeval_pending`
  /// there is fresh input waiting to be rendered.
  reeval_applied: u64,
  /// Whether a debounce timer is already in flight. While set, further
  /// control changes only bump `reeval_pending` instead of arming a second
  /// timer — this is what coalesces a burst of slider events into a single
  /// re-evaluation, mirroring the Playground's inflight/pending pipeline.
  reeval_scheduled: bool,
}

impl ManipulateState {
  /// Build a `ManipulateState` from an evaluated expression. Returns
  /// `None` if `expr` is not a well-formed Manipulate (in which case
  /// the caller should fall back to the normal text/graphics path).
  pub fn from_expr(expr: &Expr) -> Option<Self> {
    // Manipulate/Animate, a standalone Control, or a ListAnimate frame list
    // all back an interactive widget. (Native auto-play for Animate/ListAnimate
    // is a follow-up; here they render as an interactive frame slider.)
    let spec = extract_manipulate_spec(expr)
      .or_else(|| extract_control_spec(expr))
      .or_else(|| extract_list_animate_spec(expr))?;
    let controls = controls_from_spec(&spec);
    // Line each control up with its `Enabled` condition (if any) by name.
    let control_enabled: Vec<Option<String>> = controls
      .iter()
      .map(|c| {
        spec
          .control_enabled
          .iter()
          .find(|(n, _)| n == c.name())
          .map(|(_, cond)| cond.clone())
      })
      .collect();
    let control_is_enabled = vec![true; controls.len()];
    let mut state = ManipulateState {
      body: spec.body_code,
      initialization: spec.initialization,
      controls,
      state: spec.state,
      displays: spec.displays,
      display_trees: Vec::new(),
      graphics_handle: None,
      text_output: None,
      error: None,
      control_enabled,
      control_is_enabled,
      reeval_pending: 0,
      reeval_applied: 0,
      reeval_scheduled: false,
    };
    state.reevaluate();
    Some(state)
  }

  /// The full binding set (visible controls + mutable state) used to
  /// re-evaluate the body and render the display elements.
  fn bindings(&self) -> Vec<(String, String)> {
    let mut b: Vec<(String, String)> = self
      .controls
      .iter()
      .map(|c| (c.name().to_string(), c.current_code()))
      .collect();
    b.extend(self.state.iter().cloned());
    b
  }

  /// Apply an interactive checkbox write-back (e.g. `data[[3, 5]] = 1`),
  /// update the affected state variable, and re-render.
  pub fn apply_display_mutation(&mut self, mutation: &str) {
    let updated =
      apply_manipulate_mutations(&self.bindings(), &[mutation.to_string()]);
    for (name, value) in updated {
      match self.state.iter_mut().find(|(n, _)| *n == name) {
        Some(slot) => slot.1 = value,
        None => self.state.push((name, value)),
      }
    }
    self.reevaluate();
  }

  /// Register a control change and report whether the caller must arm a
  /// throttle timer. Re-evaluating the body on *every* slider mouse-move tick
  /// blocks the UI thread and makes the graphic stutter/flicker while
  /// dragging. Instead we mark the change here and only re-evaluate once the
  /// timer fires (see [`run_scheduled_reeval`]), coalescing the whole burst
  /// into a single render. Returns `true` when no timer is pending yet and the
  /// caller should spawn one.
  ///
  /// [`run_scheduled_reeval`]: Self::run_scheduled_reeval
  pub fn request_reeval(&mut self) -> bool {
    self.reeval_pending = self.reeval_pending.wrapping_add(1);
    if self.reeval_scheduled {
      false
    } else {
      self.reeval_scheduled = true;
      true
    }
  }

  /// Run a throttled re-evaluation when the debounce timer fires. Clears the
  /// pending-timer flag and re-evaluates only if a control change is still
  /// waiting to be rendered, so intermediate slider positions dropped during a
  /// fast drag never trigger a wasted (and UI-blocking) evaluation.
  pub fn run_scheduled_reeval(&mut self) {
    self.reeval_scheduled = false;
    if self.reeval_applied != self.reeval_pending {
      self.reeval_applied = self.reeval_pending;
      self.reevaluate();
    }
  }

  /// Re-run the body with the current control bindings and update the
  /// cached SVG / text output. Called on every slider change.
  pub fn reevaluate(&mut self) {
    let bindings = self.bindings();
    // Prepend `Initialization :> …` code so helper definitions made there
    // (e.g. `d[t_] := …`) are in scope while the body evaluates. The init
    // is a CompoundExpression — join with `;` so both run in one call.
    let code = match self.initialization.as_deref() {
      Some(init) => format!("{init}; {}", self.body),
      None => self.body.clone(),
    };

    // Install the bindings as globals once so a large `data` matrix is parsed
    // a single time, then evaluate the body, rebuild the display elements, and
    // resolve each control's `Enabled` condition against those same globals
    // (empty local bindings → no matrix re-embed).
    let displays = self.displays.clone();
    let control_enabled = self.control_enabled.clone();
    let (render, display_trees, enabled) =
      woxi::with_scoped_globals(&bindings, || {
        let trees: Vec<_> = displays
          .iter()
          .map(|d| build_manipulate_display(d, &[]))
          .collect();
        let enabled: Vec<bool> = control_enabled
          .iter()
          .map(|c| match c {
            Some(cond) => {
              woxi::functions::graphics::manipulate_condition_enabled(cond)
            }
            None => true,
          })
          .collect();
        (woxi::interpret_with_stdout(&code), trees, enabled)
      });
    self.display_trees = display_trees;
    self.control_is_enabled = enabled;

    // Double-buffer the render: build the new SVG handle in a local and only
    // swap the cached field once the replacement is ready, rather than nulling
    // it out before the (potentially slow) re-evaluation. This keeps the old
    // graphic on screen right up until the new one takes its place. A result
    // that genuinely produces no output still blanks the frame — the old
    // rendering is only preserved by being replaced, never by an absent one.
    //
    // The graphic is rendered by the iced `svg` widget (see the view layer),
    // not a pre-rasterized bitmap: iced's raster-image pipeline uploads any
    // texture larger than 2 MiB asynchronously on a worker thread, leaving the
    // image blank for a frame or two whenever the (always-unique) handle id
    // changes — that async upload gap is what made the graphic flicker while
    // dragging. The `svg` widget uploads synchronously in the same frame, so
    // the new graphic is drawn the instant it replaces the old one.
    match render {
      Ok(result) => {
        let cleaned = if result.graphics.is_some() || result.result == "\0" {
          String::new()
        } else {
          result
            .result
            .replace("-Graphics-", "")
            .replace("-Graphics3D-", "")
            .replace("-Image-", "")
            .trim()
            .to_string()
        };

        if let Some(svg) = result.graphics {
          let handle = svg::Handle::from_memory(svg.as_bytes().to_vec());
          self.graphics_handle = Some(handle);
          self.text_output = None;
          self.error = None;
        } else {
          // No graphic: either a textual result or genuinely empty output.
          // Blank the graphic and show the text (empty text => blank cell).
          self.graphics_handle = None;
          self.text_output = if cleaned.is_empty() {
            None
          } else {
            Some(cleaned)
          };
          self.error = None;
        }
      }
      Err(e) => {
        // Surface the evaluation error. The render path shows the error in
        // place of the graphic, so drop the cached rendering here.
        self.graphics_handle = None;
        self.text_output = None;
        self.error = Some(format!("{e}"));
      }
    }
  }
}

fn controls_from_spec(spec: &ManipulateSpec) -> Vec<ControlState> {
  spec
    .controls
    .iter()
    .map(|c| match c {
      ManipulateControl::Continuous {
        name,
        label,
        label_runs,
        min,
        max,
        step,
        initial,
      } => {
        let step = step.unwrap_or_else(|| {
          let span = (*max - *min).abs();
          if span > 0.0 { span / 100.0 } else { 1.0 }
        });
        ControlState::Continuous {
          name: name.clone(),
          label: label.clone(),
          label_runs: label_runs.clone(),
          min: *min,
          max: *max,
          step,
          current: *initial,
        }
      }
      ManipulateControl::Discrete {
        name,
        label,
        label_runs,
        values,
        value_labels,
        initial_index,
      } => ControlState::Discrete {
        name: name.clone(),
        label: label.clone(),
        label_runs: label_runs.clone(),
        values: values.clone(),
        value_labels: value_labels.clone(),
        current_index: *initial_index,
      },
      ManipulateControl::Slider2D {
        name,
        label,
        x_min,
        x_max,
        y_min,
        y_max,
        x_initial,
        y_initial,
      } => ControlState::Slider2D {
        name: name.clone(),
        label: label.clone(),
        x_min: *x_min,
        x_max: *x_max,
        y_min: *y_min,
        y_max: *y_max,
        x: *x_initial,
        y: *y_initial,
      },
      ManipulateControl::IntervalSlider {
        name,
        label,
        min,
        max,
        step,
        low_initial,
        high_initial,
      } => {
        let step = step.unwrap_or_else(|| {
          let span = (*max - *min).abs();
          if span > 0.0 { span / 100.0 } else { 1.0 }
        });
        ControlState::IntervalSlider {
          name: name.clone(),
          label: label.clone(),
          min: *min,
          max: *max,
          step,
          low: *low_initial,
          high: *high_initial,
        }
      }
    })
    .collect()
}

/// Format a f64 as Wolfram InputForm. Integers are rendered without a
/// decimal point so that e.g. `n = 10` substitutes as an Integer.
fn format_f64(v: f64) -> String {
  if v.is_finite() && v.fract() == 0.0 && v.abs() < 1e15 {
    format!("{}", v as i64)
  } else {
    format!("{}", v)
  }
}
