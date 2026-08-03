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
  apply_manipulate_mutations, build_manipulate_display, extract_animator_spec,
  extract_click_pane_spec, extract_control_spec, extract_list_animate_spec,
  extract_locator_pane_spec, extract_manipulate_spec,
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
    /// A rendered SVG icon per choice for rule labels that are graphics
    /// (`"+" -> myIcon[2]`), parallel to `values`. `None` = text label.
    value_label_svgs: Vec<Option<svg::Handle>>,
    current_index: usize,
    /// `ControlType -> PopupMenu`: always render a dropdown, even when the
    /// choice count is small enough for a SetterBar.
    popup: bool,
    /// `ControlType -> Slider`: render a slider stepping through the
    /// choices by index rather than a setter bar or dropdown.
    slider: bool,
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
    /// Write-back function (from `Locator[Dynamic[var, cb], …]` promotion):
    /// candidate values pass through `(cb)[{x, y}]` — which typically
    /// rounds, clips, or rejects them — before the variable is read back.
    write_callback: Option<String>,
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
  /// A `Locator` control binding its variable to a list of 2D points
  /// (e.g. draggable polygon vertices). Rendered as one X/Y slider pair
  /// per point; `auto_create` additionally offers add/remove buttons.
  Locator {
    name: String,
    label: String,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    points: Vec<(f64, f64)>,
    auto_create: bool,
  },
  /// A `ControlType -> Trigger` control: a play/pause pair sweeping its
  /// variable from `min` towards `max` in `step` increments while the
  /// widget is playing. `max` may be infinite (`{time, 0, Infinity, 1}`),
  /// in which case the sweep never wraps.
  Trigger {
    name: String,
    label: String,
    label_runs: Vec<LabelRun>,
    min: f64,
    max: f64,
    step: f64,
    current: f64,
  },
  /// A `Button[label, action]` row. Pressing it runs `action` (InputForm)
  /// against the live bindings — e.g. `time = 0; {U, V} = {Uinit, Vinit}`
  /// resets a simulation. Binds no variable.
  Button {
    label: String,
    label_runs: Vec<LabelRun>,
    action: String,
  },
  /// A static heading row between controls (a string or `Style[…]`
  /// Manipulate argument). Binds no variable.
  Heading {
    label: String,
    label_runs: Vec<LabelRun>,
  },
  /// A `Delimiter` argument: a horizontal separator row. Binds no variable.
  Divider,
}

impl ControlState {
  pub fn name(&self) -> &str {
    match self {
      ControlState::Continuous { name, .. } => name,
      ControlState::Discrete { name, .. } => name,
      ControlState::Slider2D { name, .. } => name,
      ControlState::IntervalSlider { name, .. } => name,
      ControlState::Trigger { name, .. } => name,
      ControlState::Locator { name, .. } => name,
      ControlState::Button { .. }
      | ControlState::Heading { .. }
      | ControlState::Divider => "",
    }
  }

  /// Whether this row binds a variable (annotation/button rows don't).
  pub fn binds_variable(&self) -> bool {
    !matches!(
      self,
      ControlState::Button { .. }
        | ControlState::Heading { .. }
        | ControlState::Divider
    )
  }

  /// InputForm fragment for the *current* value, for use inside a
  /// `Block[{name = <value>}, …]` binding.
  pub fn current_code(&self) -> String {
    match self {
      ControlState::Continuous { current, .. }
      | ControlState::Trigger { current, .. } => format_f64(*current),
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
      // Locator positions are machine reals (dragging produces fractional
      // coordinates); delegate so integral values keep their trailing dot.
      ControlState::Locator { points, .. } => {
        woxi::functions::graphics::format_point_list_input(points)
      }
      // Annotation and button rows bind no variable; never substituted.
      ControlState::Button { .. }
      | ControlState::Heading { .. }
      | ControlState::Divider => "Null".to_string(),
    }
  }

  /// Update this control's current value from an InputForm fragment — the
  /// read-back of a button action's write to the bound variable (e.g.
  /// `time = 0` rewinds the trigger it drives).
  fn set_current_from_code(&mut self, code: &str) {
    use woxi::syntax::Expr;
    fn as_f64(e: &Expr) -> Option<f64> {
      match e {
        Expr::Integer(n) => Some(*n as f64),
        Expr::Real(r) => Some(*r),
        _ => None,
      }
    }
    fn as_pair(e: &Expr) -> Option<(f64, f64)> {
      match e {
        Expr::List(items) if items.len() == 2 => {
          Some((as_f64(&items[0])?, as_f64(&items[1])?))
        }
        _ => None,
      }
    }
    let Ok(expr) = woxi::interpret_to_expr(code) else {
      return;
    };
    match self {
      ControlState::Continuous { current, .. }
      | ControlState::Trigger { current, .. } => {
        if let Some(v) = as_f64(&expr) {
          *current = v;
        }
      }
      ControlState::Discrete {
        values,
        current_index,
        ..
      } => {
        let form = woxi::syntax::expr_to_input_form(&expr);
        if let Some(i) = values.iter().position(|v| *v == form) {
          *current_index = i;
        }
      }
      ControlState::Slider2D { x, y, .. } => {
        if let Some((a, b)) = as_pair(&expr) {
          *x = a;
          *y = b;
        }
      }
      ControlState::IntervalSlider { low, high, .. } => {
        if let Some((a, b)) = as_pair(&expr) {
          *low = a;
          *high = b;
        }
      }
      ControlState::Locator { points, .. } => {
        if let Expr::List(items) = &expr
          && let Some(new_points) =
            items.iter().map(as_pair).collect::<Option<Vec<_>>>()
        {
          *points = new_points;
        }
      }
      ControlState::Button { .. }
      | ControlState::Heading { .. }
      | ControlState::Divider => {}
    }
  }
}

/// Full state for a Manipulate cell: the held body plus its rendered
/// output.
#[derive(Debug, Clone)]
pub struct ManipulateState {
  pub body: String,
  /// Initialization code from `Initialization :> …`. Run once (in
  /// [`from_expr`]) before the first render; its definitions persist in
  /// the interpreter's global environment across re-evaluations.
  ///
  /// [`from_expr`]: Self::from_expr
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
  /// Whether this widget auto-plays (from `Animate[…]` / `ListAnimate[…]` /
  /// `Animator[…]`): a timer advances the first continuous control while
  /// `playing` is set, and the view shows a play/pause toggle.
  pub animated: bool,
  /// Whether the animation is currently running. Starts `true` for animated
  /// widgets (Wolfram's default `AnimationRunning -> True`).
  pub playing: bool,
  /// `Appearance -> None`: hide the control rows (the animation just runs);
  /// the play/pause toggle stays visible for animated widgets.
  pub appearance_none: bool,
  /// `TrackedSymbols :> {…}`: the variables whose change re-runs the body.
  /// A control bound to any other variable still moves, but the rendering
  /// waits for a tracked variable to change. `None` tracks everything.
  tracked_symbols: Option<Vec<String>>,
  /// The variable a `ControlType -> Trigger`/`Animator` spec animates.
  /// `advance_animation` targets this control instead of defaulting to the
  /// first continuous one.
  animation_var: Option<String>,
  /// Continuous-control bounds that reference other control variables, as
  /// `(control name, min code, max code)`. Re-resolved against the live
  /// bindings on every re-evaluation so a slider range can follow another
  /// control (Kepler's time sliders are bounded by the orbital period `P`).
  dynamic_bounds: Vec<(String, Option<String>, Option<String>)>,
  /// Per-control choice lists that follow another control's variable
  /// (InputForm code), re-resolved on every change the way
  /// `dynamic_bounds` are.
  dynamic_values: Vec<(String, String)>,
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
    // Manipulate/Animate, a standalone Control/Animator, a ListAnimate frame
    // list, or a LocatorPane/ClickPane all back an interactive widget. The
    // animated ones (Animate/ListAnimate/Animator) auto-play: the app's
    // animation-tick subscription advances them while `playing` is set.
    let spec = extract_manipulate_spec(expr)
      .or_else(|| extract_control_spec(expr))
      .or_else(|| extract_list_animate_spec(expr))
      .or_else(|| extract_animator_spec(expr))
      .or_else(|| extract_locator_pane_spec(expr))
      .or_else(|| extract_click_pane_spec(expr))?;
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
      animated: spec.animated,
      // Auto-play immediately (Wolfram's default AnimationRunning -> True)
      // unless the spec was built paused (`AnimationRunning -> False`).
      playing: spec.animated && spec.animation_running,
      appearance_none: spec.appearance_none,
      tracked_symbols: spec.tracked_symbols,
      animation_var: spec.animation_var,
      dynamic_bounds: spec.dynamic_bounds,
      dynamic_values: spec.dynamic_values,
      control_enabled,
      control_is_enabled,
      reeval_pending: 0,
      reeval_applied: 0,
      reeval_scheduled: false,
    };
    // Run the `Initialization :> …` code ONCE, before the first render.
    // Re-running it on every re-evaluation would reset any state the body
    // mutates (e.g. the U/V concentration fields of a simulation stepping
    // itself forward), freezing the simulation at its first step. Helper
    // definitions persist in the interpreter's global environment, so a
    // single run keeps them in scope for every later re-evaluation.
    if let Some(init) = state.initialization.as_deref() {
      let _ = woxi::interpret_with_stdout(init);
    }
    state.reevaluate();
    Some(state)
  }

  /// Whether any control row is a `Trigger` (which carries its own
  /// play/pause toggle, replacing the widget-level one).
  pub fn has_trigger(&self) -> bool {
    self
      .controls
      .iter()
      .any(|c| matches!(c, ControlState::Trigger { .. }))
  }

  /// Run a `Button[…]` control's action: global side effects persist, and
  /// writes to bound control variables (e.g. `time = 0`) move the
  /// corresponding widgets. Re-renders afterwards.
  pub fn apply_button_action(&mut self, action: &str) {
    let updated = woxi::functions::graphics::apply_manipulate_button_action(
      &self.bindings(),
      action,
    );
    for (name, value) in updated {
      if let Some(slot) = self.state.iter_mut().find(|(n, _)| *n == name) {
        slot.1 = value;
        continue;
      }
      for ctrl in &mut self.controls {
        if ctrl.name() == name {
          ctrl.set_current_from_code(&value);
        }
      }
    }
    self.reevaluate();
  }

  /// The full binding set (visible controls + mutable state) used to
  /// re-evaluate the body and render the display elements.
  fn bindings(&self) -> Vec<(String, String)> {
    let mut b: Vec<(String, String)> = self
      .controls
      .iter()
      .filter(|c| c.binds_variable())
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

  /// Apply a 2D-slider change to axis 0 (x) or 1 (y). A control promoted
  /// from an in-body `Locator[Dynamic[var, cb], …]` routes the candidate
  /// point through `cb` first — evaluated against the *previous* bindings —
  /// and takes whatever value the callback actually stored (rounded,
  /// clamped, or unchanged when the callback rejects the move), exactly as
  /// Wolfram would.
  pub fn slider2d_change(&mut self, ctrl_idx: usize, axis: u8, value: f64) {
    let (candidate, callback, name) = {
      let Some(ControlState::Slider2D {
        name,
        x,
        y,
        write_callback,
        ..
      }) = self.controls.get(ctrl_idx)
      else {
        return;
      };
      let candidate = if axis == 0 { (value, *y) } else { (*x, value) };
      (candidate, write_callback.clone(), name.clone())
    };
    let accepted = match callback {
      Some(cb) => {
        // Bindings still hold the previous point, so the callback's
        // validation (e.g. a degenerate-triangle check) sees the old state.
        let bindings = self.bindings();
        let value_code = format!(
          "{{{}, {}}}",
          format_f64(candidate.0),
          format_f64(candidate.1)
        );
        woxi::functions::graphics::apply_manipulate_callback(
          &bindings,
          &cb,
          &value_code,
          &name,
        )
        .and_then(|v| woxi::functions::graphics::parse_manipulate_point(&v))
        .unwrap_or(candidate)
      }
      None => candidate,
    };
    if let Some(ControlState::Slider2D { x, y, .. }) =
      self.controls.get_mut(ctrl_idx)
    {
      *x = accepted.0;
      *y = accepted.1;
    }
  }

  /// Whether moving the control at `ctrl_idx` re-runs the body. With
  /// `TrackedSymbols :> {…}` only the listed variables do: Wolfram leaves
  /// the rendering as it is until one of them changes, so a control outside
  /// the list moves without re-rendering (and cannot show the body a
  /// half-updated set of values).
  fn control_is_tracked(&self, ctrl_idx: usize) -> bool {
    let Some(tracked) = &self.tracked_symbols else {
      return true;
    };
    match self.controls.get(ctrl_idx) {
      // A row that binds no variable (a button, a heading) is not a
      // variable change; its own handler decides whether to re-render.
      Some(control) if control.binds_variable() => {
        tracked.iter().any(|n| n == control.name())
      }
      _ => true,
    }
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
  pub fn request_reeval(&mut self, ctrl_idx: usize) -> bool {
    if !self.control_is_tracked(ctrl_idx) {
      return false;
    }
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

  /// Advance the animated control by one step, wrapping back to the start
  /// once it passes the end, then re-render. Called from the app's
  /// animation-tick subscription while `playing` is set. The target is the
  /// `ControlType -> Trigger`/`Animator` variable when the spec named one
  /// (a slider row for a finite sweep, a dedicated `Trigger` row for an
  /// infinite one — which never wraps), else the first continuous control.
  pub fn advance_animation(&mut self) {
    let target = self.animation_var.clone();
    let ctrl = self.controls.iter_mut().find(|c| match &target {
      Some(name) => matches!(
        c,
        ControlState::Continuous { name: n, .. }
        | ControlState::Trigger { name: n, .. } if n == name
      ),
      None => matches!(
        c,
        ControlState::Continuous { .. } | ControlState::Trigger { .. }
      ),
    });
    match ctrl {
      Some(ControlState::Continuous {
        min,
        max,
        step,
        current,
        ..
      })
      | Some(ControlState::Trigger {
        min,
        max,
        step,
        current,
        ..
      }) => {
        let mut v = *current + *step;
        // Loop back to the start once we step past the end (small epsilon
        // so floating-point drift doesn't skip the final frame). An
        // infinite end (`{time, 0, Infinity, 1}`) never wraps.
        if max.is_finite() && v > *max + *step * 1e-6 {
          v = *min;
        }
        *current = v;
      }
      _ => return,
    }
    self.reevaluate();
  }

  /// Re-run the body with the current control bindings and update the
  /// cached SVG / text output. Called on every slider change.
  ///
  /// The `Initialization :> …` code deliberately does NOT re-run here: it
  /// ran once in [`from_expr`], and its helper definitions persist in the
  /// interpreter's global environment. Re-running it per frame would reset
  /// any state the body itself mutates (a simulation stepping its fields
  /// forward would stay frozen on its first step).
  ///
  /// [`from_expr`]: Self::from_expr
  pub fn reevaluate(&mut self) {
    self.reevaluate_inner(true);
  }

  /// The body of [`reevaluate`]. `allow_retry` guards the single re-run
  /// that a re-resolved choice list can trigger: when the new list drops
  /// the selected value, the output just rendered was for a value the
  /// control no longer offers, so it is rendered once more for the value
  /// the control settled on. The re-run cannot cascade — it resolves the
  /// same lists against bindings that already satisfy them.
  ///
  /// [`reevaluate`]: Self::reevaluate
  fn reevaluate_inner(&mut self, allow_retry: bool) {
    let bindings = self.bindings();
    let code = self.body.clone();

    // Install the bindings as globals once so a large `data` matrix is parsed
    // a single time, then evaluate the body, rebuild the display elements, and
    // resolve each control's `Enabled` condition against those same globals
    // (empty local bindings → no matrix re-embed).
    let displays = self.displays.clone();
    let control_enabled = self.control_enabled.clone();
    let dynamic_bounds = self.dynamic_bounds.clone();
    let dynamic_values = self.dynamic_values.clone();
    let (render, display_trees, enabled, resolved_bounds, resolved_values) =
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
        // Re-resolve bounds that follow another control's variable (e.g. a
        // time slider capped by the orbital period) against these bindings.
        let resolved: Vec<(String, Option<f64>, Option<f64>)> = dynamic_bounds
          .iter()
          .map(|(name, min_code, max_code)| {
            let eval = |c: &Option<String>| {
              c.as_deref()
                .and_then(woxi::functions::graphics::manipulate_eval_bound_code)
            };
            (name.clone(), eval(min_code), eval(max_code))
          })
          .collect();
        // Likewise for choice lists built from another control's variable
        // (a level setter offering fewer levels in 3D).
        let values: Vec<(String, _)> = dynamic_values
          .iter()
          .filter_map(|(name, values_code)| {
            woxi::functions::graphics::manipulate_eval_values_code(values_code)
              .map(|cols| (name.clone(), cols))
          })
          .collect();
        (
          woxi::interpret_with_stdout(&code),
          trees,
          enabled,
          resolved,
          values,
        )
      });
    self.display_trees = display_trees;
    self.control_is_enabled = enabled;
    self.apply_dynamic_bounds(&resolved_bounds);
    // A re-resolved choice list may drop the value the body was just
    // rendered for; render again for the value the control settled on.
    if self.apply_dynamic_values(&resolved_values) && allow_retry {
      self.reevaluate_inner(false);
      return;
    }

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

  /// Move each named continuous control to its freshly resolved bounds and
  /// clamp its value inside them, so a slider range follows the control it
  /// references (dragging Kepler's period `P` widens the time sliders).
  fn apply_dynamic_bounds(
    &mut self,
    resolved: &[(String, Option<f64>, Option<f64>)],
  ) {
    for (name, new_min, new_max) in resolved {
      let Some(ControlState::Continuous {
        min, max, current, ..
      }) = self.controls.iter_mut().find(
        |c| matches!(c, ControlState::Continuous { name: n, .. } if n == name),
      )
      else {
        continue;
      };
      if let Some(v) = new_min {
        *min = *v;
      }
      if let Some(v) = new_max {
        *max = *v;
      }
      if *max < *min {
        std::mem::swap(min, max);
      }
      *current = current.clamp(*min, *max);
    }
  }

  /// Replace each named discrete control's choices with the freshly
  /// resolved list, so a setter bar follows the control it references (a
  /// level setter drops from six choices to three once the 3D view is on).
  /// The selected value is kept when it survives into the new list;
  /// otherwise the selection clamps to the last remaining choice, which is
  /// what Wolfram shows when the current level falls off the end.
  fn apply_dynamic_values(
    &mut self,
    resolved: &[(String, (Vec<String>, Vec<String>, Vec<Option<String>>))],
  ) -> bool {
    let mut selection_moved = false;
    for (name, (new_values, new_labels, new_svgs)) in resolved {
      let Some(ControlState::Discrete {
        values,
        value_labels,
        value_label_svgs,
        current_index,
        ..
      }) = self.controls.iter_mut().find(
        |c| matches!(c, ControlState::Discrete { name: n, .. } if n == name),
      )
      else {
        continue;
      };
      if values == new_values {
        continue;
      }
      let selected = values.get(*current_index).cloned();
      values.clone_from(new_values);
      value_labels.clone_from(new_labels);
      *value_label_svgs = new_svgs
        .iter()
        .map(|s| {
          s.as_ref()
            .map(|svg| svg::Handle::from_memory(svg.as_bytes().to_vec()))
        })
        .collect();
      let kept = selected.and_then(|v| values.iter().position(|nv| *nv == v));
      *current_index = kept.unwrap_or_else(|| values.len().saturating_sub(1));
      selection_moved |= kept.is_none();
    }
    selection_moved
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
        value_label_svgs,
        initial_index,
        popup,
        slider,
      } => ControlState::Discrete {
        name: name.clone(),
        label: label.clone(),
        label_runs: label_runs.clone(),
        values: values.clone(),
        value_labels: value_labels.clone(),
        value_label_svgs: value_label_svgs
          .iter()
          .map(|s| {
            s.as_ref()
              .map(|svg| svg::Handle::from_memory(svg.as_bytes().to_vec()))
          })
          .collect(),
        current_index: *initial_index,
        popup: *popup,
        slider: *slider,
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
        write_callback,
      } => ControlState::Slider2D {
        name: name.clone(),
        label: label.clone(),
        x_min: *x_min,
        x_max: *x_max,
        y_min: *y_min,
        y_max: *y_max,
        x: *x_initial,
        y: *y_initial,
        write_callback: write_callback.clone(),
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
      ManipulateControl::Locator {
        name,
        points,
        x_min,
        x_max,
        y_min,
        y_max,
        auto_create,
        label,
      } => ControlState::Locator {
        name: name.clone(),
        label: label.clone(),
        x_min: *x_min,
        x_max: *x_max,
        y_min: *y_min,
        y_max: *y_max,
        points: points.clone(),
        auto_create: *auto_create,
      },
      ManipulateControl::Trigger {
        name,
        min,
        max,
        step,
        initial,
        label,
        label_runs,
        ..
      } => ControlState::Trigger {
        name: name.clone(),
        label: label.clone(),
        label_runs: label_runs.clone(),
        min: *min,
        max: *max,
        step: *step,
        current: *initial,
      },
      ManipulateControl::Button {
        label,
        label_runs,
        action,
      } => ControlState::Button {
        label: label.clone(),
        label_runs: label_runs.clone(),
        action: action.clone(),
      },
      ManipulateControl::Heading { label, label_runs } => {
        ControlState::Heading {
          label: label.clone(),
          label_runs: label_runs.clone(),
        }
      }
      ManipulateControl::Divider => ControlState::Divider,
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
