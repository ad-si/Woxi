//! Woxi Studio helpers for rendering interactive `Manipulate[…]` cells.
//!
//! A held `Manipulate` expression is parsed into a `ManipulateState` that
//! stores the body (as InputForm source), the per-variable controls, the
//! current value of each control, and the currently rendered output
//! (SVG or text). On each slider/picklist change the state is updated
//! in place and the body is re-evaluated inside a `Block[{…}, body]` so
//! that free control variables are substituted.

use iced::widget::svg;
pub use woxi::functions::graphics::{ControlOrientation, ControlPlacement};
use woxi::functions::graphics::{
  DisplayNode, LabelRun, ManipulateControl, ManipulateSpec,
  apply_manipulate_mutations, build_manipulate_display, extract_animator_spec,
  extract_click_pane_spec, extract_control_spec, extract_list_animate_spec,
  extract_locator_pane_spec, extract_manipulate_spec, read_manipulate_state,
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
    /// Whether the variable is machine-real by construction (see
    /// `woxi::functions::graphics::ManipulateControl::Continuous::is_real`).
    /// A round current value (`0`, `20`) still substitutes as the real `0.`
    /// rather than the exact integer, matching Wolfram — which matters for
    /// a caption that formats it with `NumberForm[…, {n, f}]`.
    is_real: bool,
    /// `ControlType -> VerticalSlider`: rendered with `vertical_slider`
    /// instead of the default horizontal `slider`.
    orientation: ControlOrientation,
    /// The range runs along the imaginary axis; `min`/`max`/`current` hold
    /// imaginary coefficients and the variable binds `Complex[0, current]`.
    imaginary: bool,
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
    /// Each choice's display label as styled runs, parallel to `values` — a
    /// rule label's `Style[…, color]`/`Bold`/italic directive (e.g. a
    /// multi-way selector's `1 -> Style["top", Blue]`) rendered here instead
    /// of only in `value_labels`' plain text.
    value_label_runs: Vec<Vec<LabelRun>>,
    current_index: usize,
    /// The true current value's InputForm, kept only while it matches none
    /// of `values` (mirrors `ManipulateControl::Discrete::initial_overflow`
    /// in `woxi::functions::graphics`): several disjoint SetterBar rows can
    /// share one variable, and this row's own choices needn't include
    /// whatever value another row just picked. `current_code` and
    /// `set_current_from_code` prefer this over `values[current_index]`
    /// whenever it is set, so the row still hands back the right value even
    /// though none of its own buttons apply.
    overflow: Option<String>,
    /// `ControlType -> PopupMenu`: always render a dropdown, even when the
    /// choice count is small enough for a SetterBar.
    popup: bool,
    /// `ControlType -> SetterBar`: always render the row of buttons, even
    /// when there are more choices than the automatic split would put in a
    /// bar.
    setter_bar: bool,
    /// `ControlType -> Slider`: render a slider stepping through the
    /// choices by index rather than a setter bar or dropdown.
    slider: bool,
    /// `Appearance -> "Vertical"`: stack the SetterBar/RadioButtonBar
    /// choices in a column instead of a row. Ignored by the dropdown and
    /// index-slider layouts.
    vertical: bool,
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
    /// The variable follows the dragged points through a callback instead of
    /// holding them (see `woxi::functions::graphics::LocatorDerived`).
    derived: Option<woxi::functions::graphics::LocatorDerived>,
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
  /// A `ControlType -> ColorSlider`/`ColorSetter` control: a hue-gradient
  /// bar whose handle position (`0..1`) maps to `Hue[position]` at full
  /// saturation/value. `ColorSetter` renders and behaves identically —
  /// Wolfram draws it as the same widget, just click/tap-to-set — so both
  /// spec forms produce this one control rather than a duplicate.
  Color {
    name: String,
    label: String,
    label_runs: Vec<LabelRun>,
    /// The bound variable's current value, as InputForm. Starts as the
    /// spec's literal initial color (kept exactly even when it isn't on
    /// the hue gradient at all, e.g. `RGBColor[0, 0, 0]`); dragging the
    /// slider overwrites it with `Hue[<position>]`.
    current: String,
    /// The slider handle's on-screen position in `0.0..=1.0` (the hue
    /// fraction it was last dragged to). Purely cosmetic — defaults to
    /// `0.0`, since the spec's initial color is usually not on the
    /// gradient at all, without affecting `current`.
    position: f64,
  },
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
      ControlState::Color { name, .. } => name,
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
      ControlState::Continuous {
        current,
        is_real,
        imaginary,
        ..
      } => woxi::functions::graphics::continuous_binding_code(
        *current, *is_real, *imaginary,
      ),
      ControlState::Trigger { current, .. } => format_f64(*current),
      ControlState::Discrete {
        values,
        current_index,
        overflow,
        ..
      } => overflow.clone().unwrap_or_else(|| {
        values
          .get(*current_index)
          .cloned()
          .unwrap_or_else(|| "Null".to_string())
      }),
      ControlState::Slider2D { x, y, .. } => {
        format!("{{{}, {}}}", format_f64(*x), format_f64(*y))
      }
      ControlState::IntervalSlider { low, high, .. } => {
        format!("{{{}, {}}}", format_f64(*low), format_f64(*high))
      }
      // Locator positions are machine reals (dragging produces fractional
      // coordinates); delegate so integral values keep their trailing dot.
      ControlState::Locator {
        name,
        points,
        derived,
        ..
      } => woxi::functions::graphics::locator_binding_code(
        name,
        points,
        derived.as_ref(),
      ),
      ControlState::Color { current, .. } => current.clone(),
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
    // Exact rationals (`Rational[4, 5]`, as a saved dump records a slider's
    // `4/5`) and constants count as numbers too, not just machine literals.
    fn as_f64(e: &Expr) -> Option<f64> {
      woxi::functions::math_ast::try_eval_to_f64(e)
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
      ControlState::Continuous {
        current,
        is_real,
        step,
        min,
        max,
        ..
      } => {
        if let Some(v) = as_f64(&expr) {
          *current = v;
          // A machine real assigned to a slider whose spec was all exact
          // integers (`{c, -2, 2}`) shows Wolfram's slider is continuous:
          // an integer step could never have produced that value.
          if matches!(expr, Expr::Real(_)) && !*is_real {
            *is_real = true;
            let span = (*max - *min).abs();
            if span > 0.0 {
              *step = span / 100.0;
            }
          }
        }
      }
      ControlState::Trigger { current, .. } => {
        if let Some(v) = as_f64(&expr) {
          *current = v;
        }
      }
      ControlState::Discrete {
        values,
        current_index,
        overflow,
        ..
      } => {
        // Source form, like the choices: a `TraditionalForm[…]` value has
        // to stay that, not turn into its box rendering.
        let form = woxi::syntax::expr_to_source_form(&expr);
        if let Some(i) = values.iter().position(|v| *v == form) {
          *current_index = i;
          *overflow = None;
        } else {
          *overflow = Some(form);
        }
      }
      ControlState::Slider2D {
        x,
        y,
        x_min,
        x_max,
        y_min,
        y_max,
        ..
      } => {
        if let Some((a, b)) = as_pair(&expr) {
          *x = a;
          *y = b;
          // A locator can be dragged anywhere in the picture, so a restored
          // position outside the range derived from the initial value must
          // stay reachable instead of being clamped away.
          *x_min = x_min.min(a);
          *x_max = x_max.max(a);
          *y_min = y_min.min(b);
          *y_max = y_max.max(b);
        }
      }
      ControlState::IntervalSlider { low, high, .. } => {
        if let Some((a, b)) = as_pair(&expr) {
          *low = a;
          *high = b;
        }
      }
      // A derived variable's value is not the point list, so a write to it
      // cannot be mapped back onto the draggable points.
      ControlState::Locator {
        derived: Some(_), ..
      } => {}
      ControlState::Locator { points, .. } => {
        if let Expr::List(items) = &expr
          && let Some(new_points) =
            items.iter().map(as_pair).collect::<Option<Vec<_>>>()
        {
          *points = new_points;
        }
      }
      ControlState::Color { current, .. } => {
        *current = woxi::syntax::expr_to_input_form(&expr);
      }
      ControlState::Button { .. }
      | ControlState::Heading { .. }
      | ControlState::Divider => {}
    }
  }
}

/// A discrete control's freshly resolved choice list, as
/// `(control name, (values, labels, rendered labels, styled-label runs))`.
type ResolvedChoices = (
  String,
  (
    Vec<String>,
    Vec<String>,
    Vec<Option<String>>,
    Vec<Vec<LabelRun>>,
  ),
);

/// A pressable `Button` cell inside a rendered picture: its rectangle in the
/// picture's pixel coordinates and the held action to run when it is pressed.
#[derive(Debug, Clone, PartialEq)]
pub struct ButtonRegion {
  pub x: f32,
  pub y: f32,
  pub width: f32,
  pub height: f32,
  pub action: String,
}

/// Find every pressable `Button` cell the renderer marked in `svg` (see
/// `woxi::functions::graphics::BUTTON_HIT_ID_PREFIX`). The marker's absolute
/// position is read from the parsed tree, so it is right however deeply the
/// layout nested the cell's picture.
pub fn button_regions_of(svg: &str) -> Vec<ButtonRegion> {
  use resvg::usvg;
  use woxi::functions::graphics::{
    BUTTON_HIT_ID_PREFIX, button_action_from_hit_id,
  };
  if !svg.contains(BUTTON_HIT_ID_PREFIX) {
    return Vec::new();
  }
  let Ok(tree) = usvg::Tree::from_str(svg, &usvg::Options::default()) else {
    return Vec::new();
  };
  fn walk(group: &usvg::Group, out: &mut Vec<ButtonRegion>) {
    for node in group.children() {
      match node {
        usvg::Node::Group(g) => {
          // An `EventHandler`-wrapped shape: the group's own box is the
          // clickable area.
          if let Some(action) = button_action_from_hit_id(g.id()) {
            let r = g.abs_bounding_box();
            out.push(ButtonRegion {
              x: r.x(),
              y: r.y(),
              width: r.width(),
              height: r.height(),
              action,
            });
          } else {
            walk(g, out)
          }
        }
        usvg::Node::Path(p) => {
          if let Some(action) = button_action_from_hit_id(p.id()) {
            let r = p.abs_bounding_box();
            out.push(ButtonRegion {
              x: r.x(),
              y: r.y(),
              width: r.width(),
              height: r.height(),
              action,
            });
          }
        }
        _ => {}
      }
    }
  }
  let mut regions = Vec::new();
  walk(tree.root(), &mut regions);
  regions
}

/// A `ClickPane` drawn in a rendered picture: its rectangle in the picture's
/// pixel coordinates and the coordinate system a click inside it is reported
/// in.
#[derive(Debug, Clone, PartialEq)]
pub struct ClickRegion {
  pub x: f32,
  pub y: f32,
  pub width: f32,
  pub height: f32,
  pub hit: woxi::functions::graphics::ClickPaneHit,
}

impl ClickRegion {
  /// The code applying the pane's handler to a click at pixel `(px, py)` of
  /// the picture (the y axis points up in the pane's coordinates).
  pub fn click_action(&self, px: f32, py: f32) -> String {
    let h = &self.hit;
    let fx = f64::from((px - self.x) / self.width).clamp(0.0, 1.0);
    let fy = f64::from((py - self.y) / self.height).clamp(0.0, 1.0);
    let x = h.x_min + fx * (h.x_max - h.x_min);
    let y = h.y_max - fy * (h.y_max - h.y_min);
    format!("({})[{{{x}, {y}}}]", h.handler)
  }
}

/// Find every `ClickPane` the renderer marked in `svg` (see
/// `woxi::functions::graphics::CLICK_PANE_HIT_ID_PREFIX`).
pub fn click_regions_of(svg: &str) -> Vec<ClickRegion> {
  use resvg::usvg;
  use woxi::functions::graphics::{
    CLICK_PANE_HIT_ID_PREFIX, click_pane_from_hit_id,
  };
  if !svg.contains(CLICK_PANE_HIT_ID_PREFIX) {
    return Vec::new();
  }
  let Ok(tree) = usvg::Tree::from_str(svg, &usvg::Options::default()) else {
    return Vec::new();
  };
  fn walk(group: &usvg::Group, out: &mut Vec<ClickRegion>) {
    for node in group.children() {
      match node {
        usvg::Node::Group(g) => walk(g, out),
        usvg::Node::Path(p) => {
          if let Some(hit) = click_pane_from_hit_id(p.id()) {
            let r = p.abs_bounding_box();
            out.push(ClickRegion {
              x: r.x(),
              y: r.y(),
              width: r.width(),
              height: r.height(),
              hit,
            });
          }
        }
        _ => {}
      }
    }
  }
  let mut regions = Vec::new();
  walk(tree.root(), &mut regions);
  regions
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
  /// Pressable `Button` cells drawn inside the rendered picture (a board
  /// game's clickable squares in a `GraphicsGrid`), in the picture's pixel
  /// coordinates. The view overlays a real clickable area on each.
  pub button_regions: Vec<ButtonRegion>,
  /// `ClickPane`s drawn in the picture; a click inside one runs its handler
  /// on the click position.
  pub click_regions: Vec<ClickRegion>,
  /// Last cursor position over the picture, in its pixel coordinates.
  pub pointer: (f32, f32),
  pub text_output: Option<String>,
  pub error: Option<String>,
  /// Audio produced by the last pressed button's action (`Button[…,
  /// EmitSound[…]]`), waiting for the host to play it.
  pub emitted_sound: Option<woxi::AudioOutput>,
  /// Audio the body itself evaluates to (`Manipulate[Sound[…], …]`), shown
  /// as a play button under the controls.
  pub body_sound: Option<woxi::AudioOutput>,
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
  /// Locals hoisted out of a `DynamicModule` around the body.
  module_locals: Vec<String>,
  /// Whether the last evaluation changed one of `module_locals`: the body's
  /// own `Dynamic` rewrote it, so Wolfram would refresh it again right away.
  self_refreshing: bool,
  /// `ControlPlacement -> …`: which side of the output the control panel
  /// sits on. `Top` (Wolfram's default) stacks the controls above it.
  pub control_placement: ControlPlacement,
  /// `TrackedSymbols :> {…}`: the variables whose change re-runs the body.
  /// A control bound to any other variable still moves, but the rendering
  /// waits for a tracked variable to change. `None` tracks everything.
  tracked_symbols: Option<Vec<String>>,
  /// The variable a `ControlType -> Trigger`/`Animator` spec animates.
  /// `advance_animation` targets this control instead of defaulting to the
  /// first continuous one.
  animation_var: Option<String>,
  /// Per-control `TrackingFunction -> f` callback (InputForm code), as
  /// `(control name, function code)`. Run against the control's new value
  /// whenever the user moves it (see [`apply_tracking`]), so a Demonstration
  /// can reset a companion control the way Wolfram does.
  ///
  /// [`apply_tracking`]: Self::apply_tracking
  tracking: Vec<(String, String)>,
  /// Continuous-control bounds that reference other control variables, as
  /// `(control name, min code, max code)`. Re-resolved against the live
  /// bindings on every re-evaluation so a slider range can follow another
  /// control (Kepler's time sliders are bounded by the orbital period `P`).
  dynamic_bounds: Vec<woxi::functions::graphics::DynamicBound>,
  /// Per-control choice lists that follow another control's variable
  /// (InputForm code), re-resolved on every change the way
  /// `dynamic_bounds` are.
  dynamic_values: Vec<(String, String)>,
  /// Per-control captions that follow another control's variable (InputForm
  /// code), re-typeset on every change like `dynamic_values` are.
  dynamic_labels: Vec<(String, String)>,
  /// `(control name, reset code)` for a `LocatorPane` variable promoted from
  /// a body-local `var = expr;` assignment (see
  /// `woxi::functions::graphics::ManipulateSpec::dynamic_locator_defaults`).
  /// Unlike `dynamic_bounds`/`dynamic_values`, this is *not* re-resolved on
  /// every re-evaluation — only when [`pending_locator_reset`] marks it, i.e.
  /// when some *other* control changed since the last re-evaluation. Wolfram
  /// never reruns this statement for a change to the Locator's own dragged
  /// value (it isn't a declared control, so nothing tracks it), only for an
  /// actual control, which re-evaluates the whole body from scratch.
  ///
  /// [`pending_locator_reset`]: Self::pending_locator_reset
  dynamic_locator_defaults: Vec<(String, String)>,
  /// Parallel to `dynamic_locator_defaults`: whether that entry's default
  /// needs replaying at the next re-evaluation. Set by [`request_reeval`]
  /// for every entry other than the one (if any) whose own control just
  /// changed, cleared once [`reevaluate_inner`] replays it.
  ///
  /// [`request_reeval`]: Self::request_reeval
  /// [`reevaluate_inner`]: Self::reevaluate_inner
  pending_locator_reset: Vec<bool>,
  /// Per-control `Enabled` condition (InputForm code), parallel to `controls`.
  /// `None` means the control has no condition and is always enabled.
  control_enabled: Vec<Option<String>>,
  /// Whether each control is currently interactive, recomputed on every
  /// re-evaluation from `control_enabled` against the live bindings. Parallel
  /// to `controls`; a control indexes in with its position.
  pub control_is_enabled: Vec<bool>,
  /// Per-control visibility condition (InputForm code), parallel to
  /// `controls`. A control that belongs to a `PaneSelector` pane is on
  /// screen only while the selector holds that pane's value. `None` means
  /// the control belongs to no pane and is always shown.
  control_visible: Vec<Option<String>>,
  /// `(control name, parent variable, 1-based index)` for every control
  /// that drives one `Part` of a shared list variable rather than the
  /// whole variable — the widgets `Evaluate[Sequence @@ Table[…]]`
  /// generates for a bank of per-element sliders (see
  /// `woxi::functions::graphics::ParsedControl::ListElement`). `bindings`
  /// groups these by `parent` and reassembles them into one list binding
  /// instead of colliding same-named ones.
  list_elements: Vec<(String, String, usize)>,
  /// `Bookmarks -> {"name" :> assignment, …}`: a menu of named presets,
  /// `(label, assignment code)`. Selecting one runs [`apply_bookmark`],
  /// which is [`apply_button_action`] under a different name — the
  /// assignment code is arbitrary code run against the live bindings, same
  /// as a `Button`'s action.
  ///
  /// [`apply_bookmark`]: Self::apply_bookmark
  /// [`apply_button_action`]: Self::apply_button_action
  pub bookmarks: Vec<(String, String)>,
  /// Whether each control is currently on screen, recomputed on every
  /// re-evaluation from `control_visible` against the live bindings.
  pub control_is_visible: Vec<bool>,
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
  /// The body's raw result from the most recent re-evaluation (the error
  /// kept as its message), which [`check_render`] reports instead of
  /// evaluating the body once more.
  ///
  /// [`check_render`]: Self::check_render
  last_render: Option<Result<woxi::InterpretResult, String>>,
}

thread_local! {
  /// Set while [`ManipulateState::from_expr_unrendered`] builds a widget.
  static UNRENDERED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
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
    // Line each control up with its `Enabled` condition (if any) by name. A
    // `Button`'s own condition travels on the `ManipulateControl` itself
    // instead: it binds no variable, so `c.name()` is always `""` and a
    // name-keyed lookup into `spec.control_enabled` could never find it (and
    // would collide across buttons that each carry their own condition).
    let control_enabled: Vec<Option<String>> = controls
      .iter()
      .zip(spec.controls.iter())
      .map(|(c, raw)| match raw {
        ManipulateControl::Button { enabled, .. } => enabled.clone(),
        _ => spec
          .control_enabled
          .iter()
          .find(|(n, _)| n == c.name())
          .map(|(_, cond)| cond.clone()),
      })
      .collect();
    let control_is_enabled = vec![true; controls.len()];
    // …and with the pane it belongs to, for a `PaneSelector` control panel.
    // (a static text row has no variable, so it is filed under its label).
    let control_visible: Vec<Option<String>> = spec
      .controls
      .iter()
      .map(|raw| {
        let key = raw.visibility_key();
        spec
          .control_visible
          .iter()
          .find(|(n, _)| *n == key)
          .map(|(_, cond)| cond.clone())
      })
      .collect();
    let control_is_visible = vec![true; controls.len()];
    let mut state = ManipulateState {
      body: spec.body_code,
      initialization: spec.initialization,
      controls,
      state: spec.state,
      displays: spec.displays,
      display_trees: Vec::new(),
      graphics_handle: None,
      button_regions: Vec::new(),
      click_regions: Vec::new(),
      pointer: (0.0, 0.0),
      text_output: None,
      error: None,
      emitted_sound: None,
      body_sound: None,
      animated: spec.animated,
      // Auto-play immediately (Wolfram's default AnimationRunning -> True)
      // unless the spec was built paused (`AnimationRunning -> False`).
      playing: spec.animated && spec.animation_running,
      appearance_none: spec.appearance_none,
      control_placement: spec.control_placement,
      tracked_symbols: spec.tracked_symbols,
      animation_var: spec.animation_var,
      module_locals: spec.module_locals,
      self_refreshing: false,
      tracking: spec.tracking,
      dynamic_bounds: spec.dynamic_bounds,
      dynamic_values: spec.dynamic_values,
      dynamic_labels: spec.dynamic_labels,
      pending_locator_reset: vec![false; spec.dynamic_locator_defaults.len()],
      dynamic_locator_defaults: spec.dynamic_locator_defaults,
      control_enabled,
      control_is_enabled,
      control_visible,
      list_elements: spec.list_elements,
      bookmarks: spec.bookmarks,
      control_is_visible,
      reeval_pending: 0,
      reeval_applied: 0,
      reeval_scheduled: false,
      last_render: None,
    };
    // Run the `Initialization :> …` code ONCE, before the first render.
    // Re-running it on every re-evaluation would reset any state the body
    // mutates (e.g. the U/V concentration fields of a simulation stepping
    // itself forward), freezing the simulation at its first step. Helper
    // definitions persist in the interpreter's global environment, so a
    // single run keeps them in scope for every later re-evaluation.
    if let Some(init) = state.initialization.clone() {
      // The controls already hold their initial values when Initialization
      // runs (a DynamicModule's variables are set before its
      // Initialization), so code that reads them — `SUITS[[i]]` for a
      // `{SUITS, …, ControlType -> None}` list — sees them.
      let names: Vec<String> =
        state.state.iter().map(|(n, _)| n.clone()).collect();
      // A control name the code uses as a pattern variable
      // (`rate[t_] = D[dist[t], t]`) is a different, local `t` there, so it
      // is left unbound (Wolfram's DynamicModule renames only the free
      // occurrences).
      let pattern_names = woxi::syntax::string_to_expr(&init)
        .map(|e| woxi::evaluator::scoping::pattern_variable_names(&e))
        .unwrap_or_default();
      let bindings: Vec<(String, String)> = state
        .bindings()
        .into_iter()
        .filter(|(n, _)| !pattern_names.contains(n))
        .collect();
      // Only what the Initialization itself assigns is kept: a hidden
      // variable whose initializer depends on a control (a `DynamicModule`
      // local sized by one) stays code, so it follows saved control
      // values applied later.
      let seeded = woxi::with_scoped_globals(&bindings, || {
        let before = woxi::functions::graphics::read_manipulate_state(&names);
        let _ = woxi::interpret_with_stdout(&init);
        let after = woxi::functions::graphics::read_manipulate_state(&names);
        after
          .into_iter()
          .filter(|entry| !before.contains(entry))
          .collect::<Vec<_>>()
      });
      // A hidden `ControlType -> None` variable the Initialization just
      // seeded (e.g. `Initialization :> (v = RandomReal[…])` filling the
      // noise list a Demonstration's body perturbs) must keep that value,
      // not the spec's placeholder default (`v`'s declared domain, often
      // `{}`) — `reevaluate` below installs `state.state` as scoped
      // globals, which would otherwise immediately overwrite what
      // Initialization just wrote with the stale placeholder.
      for (name, value) in seeded {
        if let Some(slot) = state.state.iter_mut().find(|(n, _)| *n == name) {
          slot.1 = value;
        }
      }
    }
    // A display `Checkbox[Dynamic[var]]` on a never-set variable behaves as
    // unchecked (Wolfram stores the off value on first display); the body may
    // already branch on it.
    let known: Vec<String> = state
      .state
      .iter()
      .map(|(n, _)| n.clone())
      .chain(state.controls.iter().map(|c| c.name().to_string()))
      .collect();
    let defaults = woxi::functions::graphics::unset_checkbox_defaults(
      &state.displays,
      &known,
    );
    state.state.extend(defaults);
    // Rows sharing one variable start from the first row's initial value
    // (Wolfram stores a single `var$$ = <first initial>`), not the last's.
    for i in 0..state.controls.len() {
      let first = state.controls[..i]
        .iter()
        .all(|c| c.name() != state.controls[i].name());
      if first {
        state.sync_named_siblings(i);
      }
    }
    if !UNRENDERED.with(std::cell::Cell::get) {
      state.reevaluate();
    }
    Some(state)
  }

  /// [`from_expr`] without the first render, for a caller that renders
  /// once it has applied the notebook's saved control values (rendering the
  /// source defaults first would only be thrown away).
  ///
  /// [`from_expr`]: Self::from_expr
  pub fn from_expr_unrendered(expr: &Expr) -> Option<Self> {
    UNRENDERED.with(|u| u.set(true));
    let state = Self::from_expr(expr);
    UNRENDERED.with(|u| u.set(false));
    state
  }

  /// Apply saved session values — e.g. a Wolfram Demonstration's cached
  /// `"Variables" :> {…}` dump, see
  /// [`woxi::notebook::extract_saved_manipulate_variables`] — onto the
  /// matching controls by name, then re-render.
  ///
  /// A notebook saved from the desktop FrontEnd keeps both the original
  /// `Manipulate[…]` source and this dump, and the two can disagree: the
  /// source's own spec still carries whatever default the author *wrote*,
  /// while the dump carries whatever the widget's sliders were actually
  /// sitting at when the file was last saved. Without this, a widget
  /// rebuilt straight from the source (as `from_expr` does) would always
  /// open showing the authored default instead of Wolfram's own saved
  /// state. A name with no matching control (FrontEnd-only bookkeeping, or
  /// a control this build doesn't recognize) is silently skipped.
  pub fn apply_saved_variables(&mut self, saved: &[(String, String)]) {
    if saved.is_empty() {
      return;
    }
    for (name, code) in saved {
      // A widget rebuilt from a bare box dump keeps the DynamicModule's
      // `$$` uniquification suffix on its control names, while the saved
      // variable names arrive with it already stripped.
      let suffixed = format!("{name}$$");
      if let Some(control) = self
        .controls
        .iter_mut()
        .find(|c| c.name() == name || c.name() == suffixed)
      {
        control.set_current_from_code(code);
      } else if let Some(slot) = self
        .state
        .iter_mut()
        .find(|(n, _)| n == name || *n == suffixed)
      {
        // A hidden `ControlType -> None` variable (a cached puzzle, a
        // generated picture) reopens on what it held at save time too: the
        // helper code that would recompute it may be unavailable here.
        // Pictures freeze to a marker, the same as after any other update.
        let value = woxi::with_scoped_globals(&[], || {
          let expr = woxi::interpret_to_expr(code).ok()?;
          Some(
            woxi::freeze_graphic(&slot.0, &expr)
              .unwrap_or_else(|| woxi::syntax::expr_to_source_form(&expr)),
          )
        });
        if let Some(value) = value {
          slot.1 = value;
        }
      }
    }
    self.reevaluate();
  }

  /// Run the `idx`-th `Bookmarks` preset's assignment code — the same
  /// mechanism as [`apply_button_action`], just addressed by position in
  /// `bookmarks` instead of by an already-known action string. Does nothing
  /// if `idx` is out of range.
  ///
  /// [`apply_button_action`]: Self::apply_button_action
  pub fn apply_bookmark(&mut self, idx: usize) {
    if let Some((_, action)) = self.bookmarks.get(idx) {
      self.apply_button_action(&action.clone());
    }
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
  /// Run the handler of the picture's `ClickPane` for a click at the last
  /// pointer position, when that position lies inside the pane.
  pub fn apply_pane_click(&mut self) {
    let (px, py) = self.pointer;
    let action = self
      .click_regions
      .iter()
      .rev()
      .find(|r| {
        px >= r.x && px <= r.x + r.width && py >= r.y && py <= r.y + r.height
      })
      .map(|r| r.click_action(px, py));
    if let Some(action) = action {
      self.apply_button_action(&action);
    }
  }

  pub fn apply_button_action(&mut self, action: &str) {
    woxi::clear_captured_sound();
    let updated = woxi::functions::graphics::apply_manipulate_button_action(
      &self.bindings(),
      action,
    );
    self.emitted_sound = woxi::get_captured_sound();
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

  /// Copy the control at `ctrl_idx`'s just-changed value into every other
  /// control sharing its variable name. Wolfram lets more than one widget
  /// row bind the same Manipulate variable — e.g. a coarse and a fine
  /// `SetterBar` preset row for one count, as in "Polypath Iterations" —
  /// and moving either must move the other, since both are just different
  /// views onto the same underlying value.
  fn sync_named_siblings(&mut self, ctrl_idx: usize) {
    let Some(control) = self.controls.get(ctrl_idx) else {
      return;
    };
    let name = control.name().to_string();
    if name.is_empty() {
      return;
    }
    let code = control.current_code();
    for (i, ctrl) in self.controls.iter_mut().enumerate() {
      if i != ctrl_idx && ctrl.name() == name {
        ctrl.set_current_from_code(&code);
      }
    }
  }

  /// Run the control at `ctrl_idx`'s `TrackingFunction -> f` (if it has
  /// one) against its *new* current value, folding back whatever `f`
  /// assigns (e.g. a companion control it resets) into state/controls.
  /// Called right after a control's value is updated, before the caller
  /// requests a re-evaluation, so the tracking function's writes are
  /// already part of the binding set the re-render uses. Also keeps any
  /// sibling row bound to the same variable in sync (see
  /// `sync_named_siblings`), regardless of whether a tracking function
  /// is present.
  pub fn apply_tracking(&mut self, ctrl_idx: usize) {
    self.sync_named_siblings(ctrl_idx);
    let Some(control) = self.controls.get(ctrl_idx) else {
      return;
    };
    let Some((_, code)) =
      self.tracking.iter().find(|(n, _)| n == control.name())
    else {
      return;
    };
    let action = format!("({code})[{}]", control.current_code());
    let updated = woxi::functions::graphics::apply_manipulate_button_action(
      &self.bindings(),
      &action,
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
  }

  /// The full binding set (visible controls + mutable state) used to
  /// re-evaluate the body and render the display elements.
  ///
  /// A control that drives one `Part` of a shared list variable (see
  /// `list_elements`) is reported under a synthesized, internal-only name
  /// — several such controls sharing the real variable's name would
  /// collide as duplicate bindings. Those are grouped by parent variable
  /// here and reassembled into one list literal, with each control's
  /// current value substituted at its own (1-based) index; an index no
  /// control covers keeps `0` as a placeholder.
  ///
  /// Two disjoint ordinary control rows may also share one variable name
  /// (see `sync_named_siblings`), and mutable state can track a name that
  /// also has its own visible control; either way the name must appear at
  /// most once in the result. `reevaluate_inner` installs these as
  /// globals, which tolerates a repeated name by just taking the last
  /// write, but the button/tracking/mutation paths splice this list
  /// straight into a `Block[{…}, …]` local-variable specification, and
  /// Wolfram's `Block` rejects a spec that names the same local twice
  /// (`Block::dup`) — so an un-deduplicated list there silently drops the
  /// action instead of running it.
  fn bindings(&self) -> Vec<(String, String)> {
    let push_or_update = |b: &mut Vec<(String, String)>,
                          name: String,
                          code: String| {
      match b.iter_mut().find(|(n, _)| *n == name) {
        Some(slot) => slot.1 = code,
        None => b.push((name, code)),
      }
    };
    let mut b: Vec<(String, String)> = Vec::new();
    let mut list_groups: Vec<(String, Vec<(usize, String)>)> = Vec::new();
    for c in self.controls.iter().filter(|c| c.binds_variable()) {
      match self
        .list_elements
        .iter()
        .find(|(synth, ..)| synth == c.name())
      {
        Some((_, parent, index)) => {
          let group = match list_groups.iter_mut().find(|(p, _)| p == parent) {
            Some(g) => g,
            None => {
              list_groups.push((parent.clone(), Vec::new()));
              list_groups.last_mut().unwrap()
            }
          };
          group.1.push((*index, c.current_code()));
        }
        None => push_or_update(&mut b, c.name().to_string(), c.current_code()),
      }
    }
    for (parent, mut items) in list_groups {
      items.sort_by_key(|(index, _)| *index);
      let len = items.iter().map(|(index, _)| *index).max().unwrap_or(0);
      let mut parts = vec!["0".to_string(); len];
      for (index, value) in items {
        parts[index - 1] = value;
      }
      push_or_update(&mut b, parent, format!("{{{}}}", parts.join(", ")));
    }
    for (name, code) in self.state.iter().cloned() {
      push_or_update(&mut b, name, code);
    }
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

  /// Move a ColorSlider/ColorSetter handle to a new hue `position`
  /// (`0.0..=1.0`): updates the cosmetic handle position and rewrites the
  /// bound variable to `Hue[position]` — the InputForm Wolfram's own color
  /// control would bind after a drag (full saturation/value). No-op for
  /// any other control kind at `ctrl_idx`.
  pub fn color_change(&mut self, ctrl_idx: usize, position: f64) {
    if let Some(ControlState::Color {
      current,
      position: pos,
      ..
    }) = self.controls.get_mut(ctrl_idx)
    {
      *pos = position;
      *current = format!("Hue[{}]", format_f64(position));
    }
  }

  /// Select `choice_label` on the `Discrete` control at `ctrl_idx` (a
  /// SetterBar/RadioButtonBar/PopupMenu/checkbox click sends its choice's
  /// display label). Returns `true` if the control was found and the label
  /// matched one of its choices.
  ///
  /// Also clears a stale `overflow` (see its doc comment): the row's own
  /// choice was just picked, so the control can no longer be showing a value
  /// from outside its own domain. Without this, a control whose *initial*
  /// value happens to fall outside its own choice list (e.g. a
  /// `ControlType -> None` action variable defaulting to `0` with buttons
  /// for `1`..`4`, a common Demonstrations idiom for a button-driven state
  /// machine) would set `overflow` once at build time and then keep
  /// `current_code`/rendering preferring that same stale value forever —
  /// every later click on the row's own buttons would appear to do nothing.
  pub fn select_discrete(
    &mut self,
    ctrl_idx: usize,
    choice_label: &str,
  ) -> bool {
    let Some(ControlState::Discrete {
      value_labels,
      current_index,
      overflow,
      ..
    }) = self.controls.get_mut(ctrl_idx)
    else {
      return false;
    };
    let Some(idx) = value_labels.iter().position(|v| v == choice_label) else {
      return false;
    };
    *current_index = idx;
    *overflow = None;
    true
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
    // A change to any control *other* than a promoted `LocatorPane`
    // variable's own row means Wolfram would re-run the whole body —
    // replaying that variable's reset expression too. A change to its own
    // row (a drag) never does: it isn't a declared control, so nothing
    // reruns on it, which is the entire point of promoting it instead of
    // just re-evaluating the body's own `var = expr` statement in place.
    let changed_name = self.controls.get(ctrl_idx).map(ControlState::name);
    for (j, (name, _)) in self.dynamic_locator_defaults.iter().enumerate() {
      if changed_name != Some(name.as_str()) {
        self.pending_locator_reset[j] = true;
      }
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

  /// Whether the app's timer should drive this widget: it plays an
  /// animation, or its body keeps rewriting its own `DynamicModule` state.
  pub fn wants_tick(&self) -> bool {
    (self.animated && self.playing) || self.self_refreshing
  }

  /// One timer step: advance the animation, or refresh a self-updating body.
  pub fn tick(&mut self) {
    if self.animated && self.playing {
      self.advance_animation();
    } else if self.self_refreshing {
      self.reevaluate();
    }
  }

  /// Advance the animated control by one step, wrapping back to the start
  /// once it passes the end, then re-render. Called from the app's
  /// animation-tick subscription while `playing` is set. The target is the
  /// `ControlType -> Trigger`/`Animator` variable when the spec named one
  /// (always a dedicated `Trigger` row, which wraps for a finite sweep and
  /// runs forever for an infinite one), else the first continuous control.
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

  /// Evaluate the body once for the current control values and return the
  /// raw interpreter result, including the warnings that [`reevaluate`]
  /// discards. Used by the headless `--check` mode.
  ///
  /// [`reevaluate`]: Self::reevaluate
  pub fn check_render(
    &self,
  ) -> Result<woxi::InterpretResult, woxi::InterpreterError> {
    // The widget was just rendered for exactly these bindings.
    if let Some(last) = &self.last_render {
      return last
        .clone()
        .map_err(woxi::InterpreterError::EvaluationError);
    }
    woxi::with_scoped_globals(&self.bindings(), || {
      woxi::interpret_with_stdout(&self.body)
    })
  }

  /// Replay each promoted `LocatorPane` variable's reset expression that
  /// [`request_reeval`] marked pending — i.e. whose default depends on a
  /// control that changed since the last re-evaluation — against the
  /// current bindings, and move that control to the freshly computed value.
  /// A no-op when nothing is pending (the common case: a drag on the
  /// Locator's own row, or no such variable at all).
  ///
  /// [`request_reeval`]: Self::request_reeval
  fn apply_pending_locator_resets(&mut self) {
    if !self.pending_locator_reset.iter().any(|p| *p) {
      return;
    }
    let bindings = self.bindings();
    for j in 0..self.dynamic_locator_defaults.len() {
      if !self.pending_locator_reset[j] {
        continue;
      }
      self.pending_locator_reset[j] = false;
      let (name, code) = self.dynamic_locator_defaults[j].clone();
      let full_code =
        woxi::functions::graphics::manipulate_block_code(&code, &bindings);
      let Ok(expr) = woxi::interpret_to_expr(&full_code) else {
        continue;
      };
      let value_code = woxi::syntax::expr_to_input_form(&expr);
      for ctrl in &mut self.controls {
        if ctrl.name() == name {
          ctrl.set_current_from_code(&value_code);
        }
      }
    }
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
    self.apply_pending_locator_resets();
    let bindings = self.bindings();
    let code = self.body.clone();

    // Install the bindings as globals once so a large `data` matrix is parsed
    // a single time, then evaluate the body, rebuild the display elements, and
    // resolve each control's `Enabled` condition against those same globals
    // (empty local bindings → no matrix re-embed).
    let displays = self.displays.clone();
    let control_enabled = self.control_enabled.clone();
    let control_visible = self.control_visible.clone();
    let dynamic_bounds = self.dynamic_bounds.clone();
    let dynamic_values = self.dynamic_values.clone();
    let dynamic_labels = self.dynamic_labels.clone();
    // Every bound name (visible controls + hidden state), so a body that
    // reassigns one of its own controls — a common "preset" idiom, e.g.
    // `If[given > 0, {a, b, c} = presets[[given]]]` — moves that control's
    // widget too, not just the value used to render this frame.
    let binding_names: Vec<String> =
      bindings.iter().map(|(n, _)| n.clone()).collect();
    let (
      (render, raw_render),
      updated_bindings,
      display_trees,
      enabled,
      visible,
      resolved_bounds,
      resolved_values,
      resolved_labels,
    ) = woxi::with_scoped_globals(&bindings, || {
      // The body runs *before* the captions are built: a Manipulate body
      // may assign to the widget's own variables (`{v, e} = ve[[n]]`),
      // and a `Dynamic[…]` caption showing them must display what this
      // frame computed, not the previous frame's values.
      trace_evaluation(&woxi::functions::graphics::manipulate_block_code(
        &code, &bindings,
      ));
      let mut render = woxi::interpret_with_stdout(&code);
      let raw_render = match &render {
        Ok(r) => Ok(r.clone()),
        Err(e) => Err(e.to_string()),
      };
      // A `Pane[Grid[…], size]` body of plain text prints as source text in
      // the terminal; the widget shows the table the pane wraps instead.
      if matches!(&render, Ok(r) if r.graphics.is_none())
        && ["Pane[", "Panel[", "Framed["]
          .iter()
          .any(|w| code.trim_start().starts_with(w))
        && let Ok(table) =
          woxi::interpret_with_stdout(&format!("Text[{}]", code.trim()))
        && table.graphics.is_some()
      {
        render = Ok(table);
      }
      // A `Pane[text, size, …]` body only sizes and pads its text (the
      // terminal prints it as the `Pane[…]` call); the widget shows the
      // text the pane wraps.
      if matches!(&render, Ok(r) if r.graphics.is_none()
        && r.result.starts_with("Pane["))
        && code.trim_start().starts_with("Pane[")
        && let Ok(content) =
          woxi::interpret_with_stdout(&format!("First[{}]", code.trim()))
        && !content.result.starts_with("Pane[")
      {
        render = Ok(content);
      }
      let updated_bindings = read_manipulate_state(&binding_names);
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
      // Which `PaneSelector` pane is on screen, resolved the same way.
      let visible: Vec<bool> = control_visible
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
      let resolved: Vec<ResolvedBound> = dynamic_bounds
        .iter()
        .map(|(name, min_code, max_code, step_code)| {
          let eval = |c: &Option<String>| {
            c.as_deref()
              .and_then(woxi::functions::graphics::manipulate_eval_bound_code)
          };
          (
            name.clone(),
            eval(min_code),
            eval(max_code),
            eval(step_code),
          )
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
      // And captions that name another control's variable (a slider
      // called x₂ in one mode and z₀ in another).
      let labels: Vec<(String, _)> = dynamic_labels
        .iter()
        .filter_map(|(name, code)| {
          woxi::functions::graphics::manipulate_eval_label_code(code)
            .map(|runs| (name.clone(), runs))
        })
        .collect();
      (
        (render, raw_render),
        updated_bindings,
        trees,
        enabled,
        visible,
        resolved,
        values,
        labels,
      )
    });
    self.last_render = Some(raw_render);
    // Keep the body's writes to the widget's own variables, so the next
    // frame (and any caption) builds on them — whether that variable is
    // hidden state or a visible control's own slider/picker.
    self.self_refreshing = false;
    for (name, value) in updated_bindings {
      if let Some(slot) = self.state.iter_mut().find(|(n, _)| *n == name) {
        if slot.1 != value && self.module_locals.contains(&name) {
          self.self_refreshing = true;
        }
        slot.1 = value.clone();
      }
      // Every row bound to this name (a body may reassign a variable that
      // has more than one widget row, e.g. a shared SetterBar pair) moves
      // together — not just the first. A variable can also be tracked in
      // `self.state` at the same time it has its own visible control (a
      // Demonstration redeclaring its default via a separate
      // `{{var, default}, None}` spec alongside the row that already
      // controls it) — sync both, so the widget's own selection never
      // drifts from the value the body actually bound.
      for ctrl in &mut self.controls {
        if ctrl.name() == name {
          ctrl.set_current_from_code(&value);
        }
      }
    }
    self.display_trees = display_trees;
    self.control_is_enabled = enabled;
    self.control_is_visible = visible;
    self.apply_dynamic_bounds(&resolved_bounds);
    self.apply_dynamic_labels(resolved_labels);
    // A re-resolved choice list may drop the value the body was just
    // rendered for; render again for the value the control settled on.
    if self.apply_dynamic_values(&resolved_values) && allow_retry {
      self.reevaluate_inner(false);
      return;
    }
    // A `Dynamic` late in the body may set a module variable an earlier one
    // reads (`r = NSolve[…]` after a `Dynamic[… /. r]`); the front end
    // re-renders those once it changes. When that first pass warned, render
    // again with the updated variables, as the front end settles.
    if allow_retry
      && self.self_refreshing
      && matches!(&render, Ok(r) if !r.warnings.is_empty())
    {
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
    self.body_sound = render.as_ref().ok().and_then(|r| r.sound.clone());
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
        // The player stands in for the `-Sound-` placeholder.
        let cleaned = if self.body_sound.is_some() && cleaned == "-Sound-" {
          String::new()
        } else {
          cleaned
        };

        if let Some(svg) = result.graphics {
          let handle = svg::Handle::from_memory(svg.as_bytes().to_vec());
          self.button_regions = button_regions_of(&svg);
          self.click_regions = click_regions_of(&svg);
          self.graphics_handle = Some(handle);
          self.text_output = None;
          self.error = None;
        } else {
          // No graphic: either a textual result or genuinely empty output.
          // Blank the graphic and show the text (empty text => blank cell).
          self.graphics_handle = None;
          self.button_regions.clear();
          self.click_regions.clear();
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
        self.button_regions.clear();
        self.click_regions.clear();
        self.text_output = None;
        self.error = Some(format!("{e}"));
      }
    }
  }

  /// Move each named continuous control to its freshly resolved bounds and
  /// clamp its value inside them, so a slider range follows the control it
  /// references (dragging Kepler's period `P` widens the time sliders).
  fn apply_dynamic_bounds(&mut self, resolved: &[ResolvedBound]) {
    for (name, new_min, new_max, new_step) in resolved {
      // A dynamic bound may belong to either a Continuous slider or a
      // Trigger (a projectile's flight Trigger capped by another control's
      // computed range, e.g. `Dynamic[range[angle, speed]]`, follows that
      // control the same way a Continuous slider's dynamic bound does).
      if let Some(v) = new_step.filter(|v| *v != 0.0)
        && let Some(ControlState::Continuous { step, .. }) =
          self.controls.iter_mut().find(|c| {
            matches!(c, ControlState::Continuous { name: n, .. } if n == name)
          })
      {
        *step = v.abs();
      }
      let (min, max, current) = match self.controls.iter_mut().find(|c| {
        matches!(
          c,
          ControlState::Continuous { name: n, .. }
          | ControlState::Trigger { name: n, .. } if n == name
        )
      }) {
        Some(ControlState::Continuous {
          min, max, current, ..
        })
        | Some(ControlState::Trigger {
          min, max, current, ..
        }) => (min, max, current),
        _ => continue,
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

  /// Retitle each named control with its freshly typeset caption.
  fn apply_dynamic_labels(&mut self, resolved: Vec<(String, Vec<LabelRun>)>) {
    for (name, runs) in resolved {
      let Some(
        ControlState::Continuous {
          label, label_runs, ..
        }
        | ControlState::Discrete {
          label, label_runs, ..
        },
      ) = self.controls.iter_mut().find(|c| c.name() == name)
      else {
        continue;
      };
      *label = woxi::functions::graphics::manipulate_label_text(&runs);
      *label_runs = runs;
    }
  }

  /// Replace each named discrete control's choices with the freshly
  /// resolved list, so a setter bar follows the control it references (a
  /// level setter drops from six choices to three once the 3D view is on).
  /// The selected value is kept when it survives into the new list;
  /// otherwise the selection clamps to the last remaining choice, which is
  /// what Wolfram shows when the current level falls off the end.
  fn apply_dynamic_values(&mut self, resolved: &[ResolvedChoices]) -> bool {
    let mut selection_moved = false;
    for (name, (new_values, new_labels, new_svgs, new_runs)) in resolved {
      let Some(ControlState::Discrete {
        values,
        value_labels,
        value_label_svgs,
        value_label_runs,
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
      value_label_runs.clone_from(new_runs);
      let kept = selected.and_then(|v| values.iter().position(|nv| *nv == v));
      *current_index = kept.unwrap_or_else(|| values.len().saturating_sub(1));
      selection_moved |= kept.is_none();
    }
    selection_moved
  }
}

/// A dynamic bound re-resolved to numbers:
/// `(control name, min, max, step)`.
type ResolvedBound = (String, Option<f64>, Option<f64>, Option<f64>);

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
        is_real,
        orientation,
        imaginary,
      } => {
        // Wolfram treats a negative step like its magnitude, and a zero
        // step like an omitted one.
        let step =
          step.map(f64::abs).filter(|s| *s > 0.0).unwrap_or_else(|| {
            let span = (*max - *min).abs();
            let integral = min.fract() == 0.0 && max.fract() == 0.0;
            if !*is_real && integral && span >= 100.0 {
              // A wide all-exact-integer range (one whose continuous
              // granularity span/100 would reach a whole unit) makes an
              // integer slider; an exact non-integer bound (`Pi`, `1/2`)
              // keeps it continuous, and so do narrow integer ranges like
              // `{t, 0, 1}` or `{r, 0, 2}`, which a saved Manipulate shows
              // stepping through fractional values (r = 1.01).
              1.0
            } else if span > 0.0 {
              span / 100.0
            } else {
              1.0
            }
          });
        ControlState::Continuous {
          name: name.clone(),
          label: label.clone(),
          label_runs: label_runs.clone(),
          min: *min,
          max: *max,
          step,
          current: *initial,
          is_real: *is_real,
          orientation: *orientation,
          imaginary: *imaginary,
        }
      }
      ManipulateControl::Discrete {
        name,
        label,
        label_runs,
        values,
        value_labels,
        value_label_svgs,
        value_label_runs,
        initial_index,
        initial_overflow,
        popup,
        setter_bar,
        slider,
        vertical,
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
        value_label_runs: value_label_runs.clone(),
        current_index: *initial_index,
        overflow: initial_overflow.clone(),
        popup: *popup,
        setter_bar: *setter_bar,
        slider: *slider,
        vertical: *vertical,
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
        derived,
      } => ControlState::Locator {
        name: name.clone(),
        label: label.clone(),
        x_min: *x_min,
        x_max: *x_max,
        y_min: *y_min,
        y_max: *y_max,
        points: points.clone(),
        auto_create: *auto_create,
        derived: derived.clone(),
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
        enabled: _,
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
      ManipulateControl::Color {
        name,
        initial,
        label,
        label_runs,
      } => ControlState::Color {
        name: name.clone(),
        label: label.clone(),
        label_runs: label_runs.clone(),
        current: initial.clone(),
        position: 0.0,
      },
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

#[cfg(test)]
mod tests {
  /// A `Dynamic` that sets a module variable after an earlier `Dynamic`
  /// reads it (PolynomialRootsInTheComplexPlane): the front end re-renders
  /// once the variable changes, so the settled render carries no warnings
  /// from the read of the not-yet-set variable.
  #[test]
  fn later_dynamic_assignment_settles_the_earlier_reads() {
    let code = r#"Manipulate[DynamicModule[{f, r},
      Column[{Graphics[{Dynamic@Point[{Re[x], Im[x]} /. r /. (_Re | _Im) -> Sequence[]]},
          PlotRange -> 10],
        Dynamic[f = Function[a, a^2 - n]; r = Chop[NSolve[f[x] == 0, x]]; " "]}]],
      {{n, 4}, 1, 9}]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    let render = state.check_render().expect("render");
    assert!(
      render.warnings.is_empty(),
      "settled render must not warn: {:?}",
      render.warnings
    );
  }

  /// A conformal-map explorer: a textured square translated by a locator,
  /// shown beside its warp under a map chosen from a popup of pure
  /// functions labelled with `TraditionalForm`.
  #[test]
  fn conformal_warp_explorer_renders() {
    let code = r#"Manipulate[
      With[{g = Graphics[Translate[{Texture[ImageData[img]],
            Polygon[{{-2, -2}, {2, -2}, {2, 2}, {-2, 2}},
              VertexTextureCoordinates -> {{0, 0}, {1, 0}, {1, 1}, {0, 1}}]},
            pt], PlotRange -> {{-4, 4}, {-4, 4}},
          GridLines -> (Table[i, {i, Ceiling[#1], Floor[#2], 0.5}] &),
          Prolog -> {Opacity[0.2], LightGray, Scaled /@ Rectangle[{0, 0}, {1, 1}]},
          ImageSize -> 200]},
        Row[{g, Image[ImageTransformation[Image[g],
          {Re@f[#[[1]] + I #[[2]]], Im@f[#[[1]] + I #[[2]]]} &,
          DataRange -> {{-Pi, Pi}, {-Pi, Pi}}, PlotRange -> {{-Pi, Pi}, {-Pi, Pi}}],
          ImageSize -> 200]}]],
      {{img, Image[{{0, 1, 0, 1}, {1, 0, 1, 0}, {0, 1, 0, 1}, {1, 0, 1, 0}}]}, ControlType -> None},
      Spacer[10],
      Control[{{f, 1/# &}, (#[[1]] -> TraditionalForm[#[[2]]] &) /@
        {(#/I &) -> I z, (1/# &) -> z^-1, (#^2 &) -> Sqrt[z],
         (Exp[#] &) -> Log[z], (Sin[#] &) -> ArcSin[z]}, ControlType -> PopupMenu}],
      {{pt, {0, 0}}, {-Pi, -Pi}, {Pi, Pi}, ControlType -> Locator},
      TrackedSymbols :> {img, pt, f}]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    assert_eq!(state.error, None);
    assert!(
      state
        .controls
        .iter()
        .any(|c| matches!(c, ControlState::Slider2D { .. })),
      "the point locator must be exposed as a 2D pad"
    );
    assert!(
      state
        .controls
        .iter()
        .any(|c| matches!(c, ControlState::Discrete { .. })),
      "the function popup must be exposed"
    );
    assert!(
      state.graphics_handle.is_some(),
      "warp view must render, got text {:?}",
      state.text_output
    );
  }

  /// A contour-line explorer: a photo picked from a setter bar of images,
  /// contour levels set by a custom `Dynamic[slider[##]] &` control built on
  /// a `LocatorPane`, a color-scheme popup that is disabled while
  /// "contours only" is on, and a body interpolating the rotated pixel data.
  #[test]
  fn contour_explorer_with_custom_locator_slider_renders() {
    let code = r#"Manipulate[
      Module[{dens, im = ImageRotate[photos[[i]], -Pi/2]},
        dens = ListInterpolation[ImageData[im]];
        Framed[ContourPlot[dens[x, y], {x, 1, 4}, {y, 1, 4},
          PlotPoints -> ControlActive[20, 40], Contours -> levels,
          ContourStyle -> Black,
          ContourShading -> If[cnt, None, Automatic],
          ColorFunction -> scheme, Frame -> False,
          AspectRatio -> Automatic, ImageSize -> 350], FrameMargins -> 0]],
      Style["select an image", Bold],
      {{i, 2, ""}, {1 -> Image[{{0, 1, 0, 1}, {1, 0, 1, 0}, {0, 1, 0, 1}, {1, 0, 1, 0}}],
        2 -> Image[{{0, 0.3, 0.6, 1}, {0.2, 0.4, 0.7, 0.9}, {0.1, 0.5, 0.8, 1}, {0, 0.2, 0.5, 0.7}}]}},
      Delimiter,
      Row[{Style["move the ", Bold], Style["\[FilledDiamond]", Red, 14]}],
      {{levels, {1/7, 2/7, 3/7}, ""}, Dynamic[slider[##]] &},
      Delimiter,
      {{scheme, "SunsetColors", ""}, {"Rainbow", "SunsetColors", "Pastel"},
        Enabled :> Dynamic[If[cnt, False, True]]},
      {{cnt, False, ""}, {True, False}},
      Initialization :> (
        photos = {
          Image[{{0, 1, 0, 1}, {1, 0, 1, 0}, {0, 1, 0, 1}, {1, 0, 1, 0}}],
          Image[{{0, 0.3, 0.6, 1}, {0.2, 0.4, 0.7, 0.9}, {0.1, 0.5, 0.8, 1}, {0, 0.2, 0.5, 0.7}}]};
        slider[Dynamic[Pattern[locs, Blank[]]]] := DynamicModule[
          {pts = Map[{#, 0} &, Range[1/7, 1 - 1/7, 1/7]]},
          LocatorPane[
            Dynamic[pts, Function[{val, expr},
              locs = Part[val, All, 1]; expr = Map[{#, 0} &, locs], HoldAll]],
            Dynamic[Graphics[{}, Frame -> True, PlotRange -> {{0, 1}, {-1, 1}},
              AspectRatio -> 0.1, ImageSize -> 200]],
            {{0, 0}, {1, 0}}, Appearance -> Style["\[FilledDiamond]", Red, 14],
            LocatorAutoCreate -> {1, 16}]])
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    assert_eq!(state.error, None);
    assert!(
      state.graphics_handle.is_some(),
      "contour plot must render, got text {:?}",
      state.text_output
    );
    assert!(
      state
        .controls
        .iter()
        .any(|c| matches!(c, ControlState::Locator { .. })),
      "the custom slider must expose a locator control"
    );
    // The variable holds the markers' x positions (the callback's result),
    // not the {x, y} points themselves, and follows a drag.
    let idx = state
      .controls
      .iter()
      .position(|c| matches!(c, ControlState::Locator { .. }))
      .unwrap();
    let before = state.controls[idx].current_code();
    assert!(
      !before.contains("{{"),
      "levels must be plain numbers: {before}"
    );
    let mut state = state;
    if let ControlState::Locator { points, .. } = &mut state.controls[idx] {
      points[0].0 = 0.05;
    }
    let after = state.controls[idx].current_code();
    assert_ne!(before, after);
    assert!(after.starts_with("{0.05,"), "got {after}");
  }

  /// The same contour explorer with the custom slider written the way the
  /// Demonstrations Project does: `Dynamic[slider[##],
  /// SynchronousUpdating -> False] &` as the control, and a definition
  /// `slider[Dynamic@locs_] := …` using the prefix `Dynamic@` form.
  #[test]
  fn custom_locator_slider_with_dynamic_options_is_a_locator() {
    let code = r#"Manipulate[
      Framed[ListLinePlot[levels, ImageSize -> 100], FrameMargins -> 0],
      Style["move the locators", Bold],
      {{levels, {1/7, 2/7, 3/7}, ""},
        Dynamic[slider[##], SynchronousUpdating -> False] &},
      Initialization :> (
        slider[Dynamic@locs_] := DynamicModule[
          {pts = Map[{#, 0} &, Range[1/7, 1 - 1/7, 1/7]]},
          LocatorPane[
            Dynamic[pts, Function[{val, expr},
              locs = val[[All, 1]]; expr = {#1, 0} & /@ locs, HoldAll]],
            Dynamic[Graphics[{}, Frame -> True, PlotRange -> {{0, 1}, {-1, 1}},
              AspectRatio -> .1, ImageSize -> 200]],
            {{0, 0}, {1, 0}}, Appearance -> Style["\[FilledDiamond]", Red, 14],
            LocatorAutoCreate -> {1, 16}]])
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    assert_eq!(state.error, None);
    assert!(
      state
        .controls
        .iter()
        .any(|c| matches!(c, ControlState::Locator { .. })),
      "the custom slider must expose a locator control, got {:?}",
      state.controls
    );
  }

  /// A complex wave-equation explorer: the body solves a periodic nonlinear
  /// Schrödinger-type PDE with `NDSolve` and draws `Abs[u]` either as a
  /// density plot or a surface, chosen with a setter bar. The solution is
  /// complex-valued, so every control change re-solves and re-plots.
  #[test]
  fn complex_pde_explorer_renders_both_plot_types() {
    let code = r#"Manipulate[
      If[flat, DensityPlot, Plot3D][
        Evaluate[Abs[w[s, y]] /. Quiet[NDSolve[{
          I D[w[s, y], s] + 0.5 D[w[s, y], y, y] + Abs[w[s, y]]^2 w[s, y] == 0,
          w[0, y] == Exp[-(y - sep)^2] + Exp[-(y + sep)^2],
          w[s, -half] == w[s, half]},
          w, {s, 0, 2}, {y, -half, half}]]],
        {y, -half, half}, {s, 0, 2},
        Mesh -> mesh, PlotRange -> Full, PlotPoints -> 8,
        MaxRecursion -> ControlActive[0, 1], ImageSize -> {300, 200}],
      {{sep, 1., "separation"}, 0, 3, Appearance -> "Labeled"},
      {{half, 4., "range"}, 2, 8, Appearance -> "Labeled"},
      {{mesh, Automatic, "show mesh"}, {False, Automatic}, ControlType -> Checkbox},
      {{flat, True, "plot type"}, {True -> "2D", False -> "3D"}, ControlType -> SetterBar},
      ControlPlacement -> Top
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let mut state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    assert_eq!(state.error, None);
    assert!(
      state.graphics_handle.is_some(),
      "density plot must render, got text {:?}",
      state.text_output
    );
    let flat = state
      .controls
      .iter()
      .position(|c| c.name() == "flat")
      .expect("plot type control");
    assert!(state.select_discrete(flat, "3D"));
    assert_eq!(state.error, None);
    assert!(
      state.graphics_handle.is_some(),
      "surface plot must render, got text {:?}",
      state.text_output
    );
  }

  /// A hidden `ControlType -> None` variable is bound when Initialization
  /// runs, also when a control's spec depends on what Initialization sets
  /// (`Control[{{n, top/4}, 2^Range[4, k]}]` with `top = 2^k` there).
  #[test]
  fn initialization_sees_hidden_control_used_by_control_specs() {
    let code = r#"Manipulate[
      Length[data] + n,
      Row[{Control[{{n, top/4}, 2^Range[4, k], ControlType -> Setter}]}],
      {{k, 6}, ControlType -> None},
      Initialization :> (top = 2^k; data = Range[top])]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let mut state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    state.reevaluate();
    assert_eq!(state.error, None);
    assert_eq!(state.text_output.as_deref(), Some("80"));
  }

  /// An `Epilog -> (list /. rule)` option is parenthesised in the source; the
  /// body is printed back to text for every re-evaluation, and without the
  /// parentheses the option became `(Epilog -> list) /. rule`, so the
  /// annotations vanished from the plot.
  #[test]
  fn epilog_with_replacement_survives_reevaluation() {
    let code = r#"Manipulate[
      Plot[a x, {x, 0, 1},
        Epilog -> ({Red, Line[{{0, 0}, {t, a t}}], Point[{t, a t}]} /. a -> aa),
        ImageSize -> 200],
      {{aa, 2, "slope"}, 1, 3, 1}, {{t, 0.5, "t"}, 0, 1, 0.1},
      Initialization :> (a = 5)]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let mut state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    state.reevaluate();
    assert_eq!(state.error, None);
    assert!(
      state.body.contains("Epilog -> ({"),
      "the replacement must stay parenthesised, got {}",
      state.body
    );
  }

  /// A slider whose bounds and initial value are exact integers and that
  /// has no explicit step is an integer slider (step 1), not span/100.
  /// A button that stores a *list* of pictures in a state variable (a puzzle
  /// and its solution, as in "Polygonal Sudoku") must keep them as pictures:
  /// flattening the list to source text turned `wr[[1]]` into a `Part` error.
  #[test]
  fn button_action_storing_list_of_graphics_keeps_pictures() {
    let code = r#"Manipulate[
      If[new, "choose!", If[!alt, wr[[1]], wr[[2]]]],
      Row[{
        Control@{{new, True}, ControlType -> None},
        Button["new", wr = {Show[Graphics[Disk[]]], Show[Graphics[Circle[]]]}; new = False],
        Control@{{alt, False}, {True, False}}
      }],
      {{wr, Graphics[Disk[]]}, ControlType -> None}
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let mut state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    state.apply_button_action(
      "wr = {Show[Graphics[Disk[]]], Show[Graphics[Circle[]]]}; new = False",
    );
    assert_eq!(state.error, None);
    assert!(
      state.graphics_handle.is_some(),
      "wr[[1]] must render a picture, got text {:?}",
      state.text_output
    );
  }

  /// A pad of note buttons generated with `Map` whose action is
  /// `EmitSound[Sound[SoundNote[…]]]` (a "name that pitch" quiz shape):
  /// pressing one must hand its synthesized audio to the host instead of
  /// printing a "not yet implemented" warning, and a button without a sound
  /// must clear any stale audio.
  #[test]
  fn button_action_emitting_sound_exposes_audio() {
    let code = r#"Manipulate[
      Row[(Button[#, EmitSound[Sound[SoundNote[#, len, "Flute"]]]] &) /@ {0, 4, 7}, " "],
      {{len, 1}, 1, 3, 1},
      Button["silent", flag = 1]
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let mut state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    assert_eq!(state.error, None, "build must not report an error");
    state.apply_button_action("EmitSound[Sound[SoundNote[4, len, \"Flute\"]]]");
    assert!(
      state.emitted_sound.take().is_some(),
      "pressing a note button should expose playable audio"
    );
    state.apply_button_action("flag = 1");
    assert!(
      state.emitted_sound.is_none(),
      "a silent button must not replay stale audio"
    );
  }

  /// A body that evaluates to a `Sound` (optionally wrapped in `Magnify`)
  /// exposes playable audio that follows the controls, instead of a bare
  /// `-Sound-` placeholder.
  #[test]
  fn sound_body_exposes_playable_audio() {
    for body in [
      r#"Sound[SoundNote[{0, 4, 7} + k, 0.2, "Flute"]]"#,
      r#"Magnify[Sound[SoundNote[{0, 4, 7} + k, 0.2, "Flute"]], 2]"#,
    ] {
      let code = format!("Manipulate[{body}, {{k, 0, 11, 1}}]");
      let expr = woxi::interpret_to_expr(&code).expect("parse Manipulate expr");
      let state =
        ManipulateState::from_expr(&expr).expect("build Manipulate widget");
      assert_eq!(state.error, None, "build must not report an error");
      assert!(state.body_sound.is_some(), "no audio for {body}");
      assert_eq!(state.text_output, None, "placeholder shown for {body}");
    }
  }

  /// Regression: `Item[Button[…], Alignment -> Center]` in a Manipulate's
  /// control list is a pressable button, not a static annotation row that
  /// shows the button's source text as a heading.
  #[test]
  fn item_wrapped_button_is_a_button_control() {
    let code = r#"Manipulate[
      Range[n],
      {{n, 2}, 1, 4, 1},
      Item[Button["Reset", n = 1, ImageSize -> 300], Alignment -> Center]
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    assert_eq!(state.error, None, "build must not report an error");
    assert!(
      state.controls.iter().any(|c| matches!(
        c,
        ControlState::Button { label, action, .. }
          if label == "Reset" && action == "n = 1"
      )),
      "expected a Reset button control, got {:?}",
      state.controls
    );
    assert!(
      !state
        .controls
        .iter()
        .any(|c| matches!(c, ControlState::Heading { .. })),
      "the button must not be frozen into a heading"
    );
  }

  /// `Button[Dynamic[caption], ImageSize -> n]` has an option where the
  /// action would be: it is an inert caption that follows the live
  /// bindings, not a pressable button whose "action" is `ImageSize -> n`.
  #[test]
  fn actionless_button_with_dynamic_label_is_a_live_display() {
    let code = r#"Manipulate[
      Graphics[{}],
      {{step, 1}, ControlType -> None},
      Button[Dynamic@Switch[step, 1, "first", 2, "second"], ImageSize -> 240]
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    assert_eq!(state.error, None, "build must not report an error");
    assert!(
      !state
        .controls
        .iter()
        .any(|c| matches!(c, ControlState::Button { .. })),
      "no pressable button expected, got {:?}",
      state.controls
    );
    assert_eq!(
      state.displays.len(),
      1,
      "the caption must be one live display, got {:?}",
      state.displays
    );
    assert!(state.displays[0].contains("Switch"), "{:?}", state.displays);
  }

  /// A key-number Manipulate: a `Pane` around a `Grid` with `SpanFromLeft`,
  /// a `Setter` control whose choices are a `Dynamic[…]` list of numbers
  /// computed from a helper function defined in `Initialization`, a note
  /// name written as inline `\!\(\*…\)` box notation, and a `Play` cell next
  /// to the `Plot`.
  #[test]
  fn pane_grid_span_from_left_with_dynamic_setter_and_play() {
    let code = r#"Manipulate[
      Pane[Grid[{
        {Text@Style[TraditionalForm[y == Sin[2 Pi f[n] t]], 20], SpanFromLeft},
        {Text@Style[TraditionalForm[note == "\!\(\*SuperscriptBox[\(A\), \(#\)]\)"], 20],
         Text@Style[TraditionalForm[frequency == f[n] " Hz"], 20]},
        {Plot[Sin[f[n] 2 Pi t], {t, 0, w}, ImageSize -> {250, 175}],
         Play[Sin[f[n] 2 Pi t], {t, 0, 1}]}},
        ItemSize -> {Automatic, {4, 4, 10}}], {525, 250}],
      {{w, 0.05, "width"}, Dynamic[{0.05, 1/f[n], 2/f[n]}], ControlType -> Setter},
      {{n, 49, "key"}, 1, 88, 1, ControlType -> PopupMenu},
      Initialization :> (f[m_] := 440. 2^(Quotient[m, 12] - 4 + (Mod[m, 12] - 1)/12))]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let mut state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    assert_eq!(state.error, None);
    assert!(
      state.graphics_handle.is_some(),
      "the body must draw a picture"
    );
    state.reevaluate();
    assert_eq!(state.error, None);
  }

  /// A body that is `Text@Style[TraditionalForm[Column[…]]]` of held
  /// `Integrate` calls draws the typeset stack, not the `Column(…)` call
  /// as text.
  #[test]
  fn traditional_form_column_body_draws_a_picture() {
    let code = r#"Manipulate[
      Text@Style[TraditionalForm[Column[{
        HoldForm[Integrate[Sin[x]^n, x]],
        Row[{" = ", HoldForm[Integrate[Cos[x]^n, x]]}]}, ItemSize -> 28]], 18],
      {{n, 3}, 1, 9, 2}]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    assert_eq!(state.error, None);
    assert!(
      state.graphics_handle.is_some(),
      "the body must draw a picture, got text {:?}",
      state.text_output
    );
  }

  /// A choice list `Dynamic[Range[top]]` that follows a free variable the
  /// body itself assigns (not a control): switching the mode makes the body
  /// assign a different `top`, and the offered choices must follow it.
  #[test]
  fn choice_list_follows_variable_assigned_by_body() {
    let code = r#"Manipulate[
      If[mode == "short", top = 3; Range[top], top = 5; Range[top]],
      {{mode, "short"}, {"short", "long"}},
      {{k, 1}, Dynamic[Range[top]], ControlType -> SetterBar}
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let mut state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    let count_of = |state: &ManipulateState| {
      state
        .controls
        .iter()
        .find_map(|c| match c {
          ControlState::Discrete { name, values, .. } if name == "k" => {
            Some(values.len())
          }
          _ => None,
        })
        .expect("k control")
    };
    assert_eq!(count_of(&state), 3);
    assert!(state.select_discrete(0, "long"));
    state.reevaluate();
    assert_eq!(count_of(&state), 5);
    assert!(state.select_discrete(0, "short"));
    state.reevaluate();
    assert_eq!(count_of(&state), 3);
  }

  /// A sound-player display built with `Audio`Waveform` evaluates to an
  /// Audio object (shown as `-Audio-`), not the raw unevaluated call.
  #[test]
  fn audio_waveform_display_evaluates_to_audio_object() {
    let code = r#"Manipulate[
      Plot[Sin[x t], {x, 0, 6}],
      {{t, 0.2}, 0.1, 1, 0.1},
      Dynamic[Audio`Waveform[Audio`Square, 440., t, Overtones -> 5, SampleRate -> 2^16]]
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    assert_eq!(state.error, None);
    assert!(
      state.display_trees.iter().any(
        |n| matches!(n, DisplayNode::Static { text, .. } if text == "-Audio-")
      ),
      "{:?}",
      state.display_trees
    );
  }

  #[test]
  fn exact_integer_slider_without_step_steps_by_one() {
    let expr = woxi::interpret_to_expr(
      "Manipulate[{n, x}, {{n, 160}, 1, 160}, {{x, 0.5}, 0, 1}]",
    )
    .expect("parse Manipulate expr");
    let state = ManipulateState::from_expr(&expr).expect("build widget");
    let step_of = |name: &str| {
      state
        .controls
        .iter()
        .find_map(|c| match c {
          ControlState::Continuous { name: n, step, .. } if n == name => {
            Some(*step)
          }
          _ => None,
        })
        .expect("continuous control")
    };
    assert_eq!(step_of("n"), 1.0);
    assert_eq!(step_of("x"), 0.01);
  }

  /// Regression: a narrow exact-integer range such as `{r, 0, 2}` or
  /// `{pr, 1, 10}` is a continuous slider (a saved Steiner-chain
  /// Manipulate records `r = 1.01`, `pr = 3.26`), not an integer one.
  #[test]
  fn narrow_exact_integer_range_slider_is_continuous() {
    let expr = woxi::interpret_to_expr(
      "Manipulate[{r, p}, {{r, 1, \"radius\"}, 0, 2}, {{p, 3, \"range\"}, 1, 10}]",
    )
    .expect("parse Manipulate expr");
    let state = ManipulateState::from_expr(&expr).expect("build widget");
    for c in &state.controls {
      if let ControlState::Continuous {
        name,
        step,
        min,
        max,
        ..
      } = c
      {
        assert!(
          (*step - (*max - *min) / 100.0).abs() < 1e-12,
          "{name} should be continuous, got step {step}"
        );
      }
    }
  }

  /// A unit range with exact integer bounds (`{t, 0, 1}`) is a continuous
  /// fraction slider, not a two-position integer one.
  #[test]
  fn exact_unit_range_slider_is_continuous() {
    let expr = woxi::interpret_to_expr(
      "Manipulate[{t, u}, {{t, 0, \"open\"}, 0, 1}, {u, -3, -2}]",
    )
    .expect("parse Manipulate expr");
    let state = ManipulateState::from_expr(&expr).expect("build widget");
    for c in &state.controls {
      if let ControlState::Continuous { name, step, .. } = c {
        assert_eq!(*step, 0.01, "{name} should be continuous");
      }
    }
  }

  /// A body that calls a helper returning an `Animate[…]` (defined in the
  /// Manipulate's own `Initialization`) draws the animation's first frame
  /// rather than an empty picture.
  #[test]
  fn body_returning_animate_draws_first_frame() {
    woxi::clear_state();
    let expr = woxi::interpret_to_expr(
      "Manipulate[frameOf[r], {{r, 2, \"radius\"}, 1, 3}, \
       Initialization :> (frameOf[q_] := Animate[Graphics[{Red, Disk[{T, 0}, q]}], {T, 0, 1}])]",
    )
    .expect("parse Manipulate expr");
    let state = ManipulateState::from_expr(&expr).expect("build widget");
    assert!(state.graphics_handle.is_some());
    let render =
      woxi::with_scoped_globals(&[("r".to_string(), "2".to_string())], || {
        woxi::interpret_with_stdout(&state.body)
      })
      .expect("evaluate body");
    let svg = render.graphics.expect("body draws a graphic");
    assert!(svg.contains("<ellipse"), "{svg}");
  }

  /// A body that is a `ListAnimate` of precomputed frames (here 3D ones,
  /// built by a helper) draws the first frame, not the last one evaluated.
  #[test]
  fn body_list_animate_draws_first_frame() {
    woxi::clear_state();
    let expr = woxi::interpret_to_expr(
      "Manipulate[ListAnimate[frames[h]], {{h, 1}, 1, 3}, \
       Initialization :> (frames[q_] := Table[Graphics3D[{Polygon[{{0, 0, 0}, \
       {4, 0, 0}, {4, 4, k q}}]}, Boxed -> False], {k, 1, 3}])]",
    )
    .expect("parse Manipulate expr");
    let state = ManipulateState::from_expr(&expr).expect("build widget");
    let render =
      woxi::with_scoped_globals(&[("h".to_string(), "1".to_string())], || {
        woxi::interpret_with_stdout(&state.body)
      })
      .expect("evaluate body");
    let first = woxi::interpret(
      "ExportString[Graphics3D[{Polygon[{{0, 0, 0}, {4, 0, 0}, {4, 4, 1}}]}, \
       Boxed -> False], \"SVG\"]",
    )
    .expect("render first frame");
    assert_eq!(render.graphics.expect("body draws a graphic"), first);
  }

  /// A body that is a sized, padded `Pane` around styled text shows the
  /// text itself rather than the `Pane[…, {400, 80}, ImageMargins -> 20]`
  /// call it prints as in a terminal.
  #[test]
  fn pane_around_text_body_shows_the_text() {
    woxi::clear_state();
    let expr = woxi::interpret_to_expr(
      "Manipulate[Pane[Style[w <> \"!\", Bold, 16], {400, 80}, \
       ImageMargins -> 20], {{w, \"hello\", \"word\"}, {\"hello\", \"bye\"}, \
       ControlType -> PopupMenu}]",
    )
    .expect("parse Manipulate expr");
    let state = ManipulateState::from_expr(&expr).expect("build widget");
    assert_eq!(state.text_output.as_deref(), Some("hello!"));
  }

  /// Words joined from `Style[word, color, size]` through
  /// `ToString[…, StandardForm]` are drawn at their own sizes and colors,
  /// not as the `DisplayForm[…]` source of each box.
  #[test]
  fn styled_word_paragraph_draws_each_word_at_its_size() {
    woxi::clear_state();
    let expr = woxi::interpret_to_expr(
      "Manipulate[Text[Style[StringJoin[Riffle[ToString[#, StandardForm] & /@ \
       MapThread[Style[##, Bold] &, {{\"alpha\", \"beta\", \"gamma\"}, \
       {Red, Blue, Gray}, {n, 2 n, 3 n}}], \" \"]], LineIndent -> 0]], \
       {{n, 10}, 8, 12, 1}]",
    )
    .expect("parse Manipulate expr");
    let state = ManipulateState::from_expr(&expr).expect("build widget");
    assert!(state.text_output.is_none(), "{:?}", state.text_output);
    let render =
      woxi::with_scoped_globals(&[("n".to_string(), "10".to_string())], || {
        woxi::interpret_with_stdout(&state.body)
      })
      .expect("evaluate body");
    let svg = render.graphics.expect("body draws a graphic");
    for (word, size) in [("alpha", 10), ("beta", 20), ("gamma", 30)] {
      assert!(
        svg.contains(&format!("font-size=\"{size}.0\"")),
        "{word} at {size}: {svg}"
      );
      assert!(svg.contains(&format!(">{word}</text>")), "{svg}");
    }
    assert!(!svg.contains("DisplayForm"), "{svg}");
  }

  /// An exact but non-integer bound (`Pi`) makes the slider continuous,
  /// even though no bound was written as a machine real.
  #[test]
  fn exact_non_integer_bound_slider_is_continuous() {
    let expr = woxi::interpret_to_expr(
      "Manipulate[{r, q}, {{r, 0, \"rotate\"}, 0, Pi}, {q, 0, 5/2}]",
    )
    .expect("parse Manipulate expr");
    let state = ManipulateState::from_expr(&expr).expect("build widget");
    let step_of = |name: &str| {
      state
        .controls
        .iter()
        .find_map(|c| match c {
          ControlState::Continuous { name: n, step, .. } if n == name => {
            Some(*step)
          }
          _ => None,
        })
        .expect("continuous control")
    };
    assert!((step_of("r") - std::f64::consts::PI / 100.0).abs() < 1e-12);
    assert!((step_of("q") - 0.025).abs() < 1e-12);
  }

  /// A compiled widget dump lists each module variable as a hidden
  /// `{var, ControlType -> None}` entry before the visible control for the
  /// same variable; the hidden initial value must not shadow the control.
  #[test]
  fn hidden_state_entry_does_not_shadow_visible_control() {
    let code = r#"Manipulate[
      {a, b},
      {{a, 1}, ControlType -> None},
      {{a, 2, "a"}, {1 -> "one", 2 -> "two", 3 -> "three"},
        ControlType -> PopupMenu},
      {{b, 0.5}, 0, 1}
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let mut state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    assert!(
      state.state.iter().all(|(n, _)| n != "a"),
      "control-bound variable must not stay in hidden state: {:?}",
      state.state
    );
    let idx = state
      .controls
      .iter()
      .position(|c| c.name() == "a")
      .expect("popup control for a");
    assert!(state.select_discrete(idx, "three"));
    let bindings = state.bindings();
    assert_eq!(
      bindings
        .iter()
        .find(|(n, _)| n == "a")
        .map(|(_, v)| v.as_str()),
      Some("3")
    );
  }

  /// A saved hidden state value can wrap a rendered graphic in a helper the
  /// notebook defines later (`pts = helper[Graphics[…]]`). Freezing it must
  /// keep the symbolic source rather than the output-only `-Graphics-`
  /// placeholder, which cannot be parsed back when the body re-evaluates.
  #[test]
  fn hidden_state_with_graphic_keeps_symbolic_source() {
    let code = r#"Manipulate[
      Graphics[Circle[{0, 0}, n]],
      {{pts, helper[Graphics[Circle[{0, 0}, 1]]]}, ControlType -> None},
      {n, 1, 3, 1}
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    let value = state
      .state
      .iter()
      .find(|(n, _)| n == "pts")
      .map(|(_, v)| v.clone())
      .expect("pts state variable");
    assert!(!value.contains("-Graphics-"), "got {value}");
    assert!(value.starts_with("helper[Graphics["), "got {value}");
  }

  /// A saved `TraditionalForm[…]` menu value selects that entry: it is
  /// compared in source form like the choices, not as its box rendering
  /// (NewtonCotesQuadrature bound `f = FormBox[…]`).
  #[test]
  fn saved_traditional_form_value_selects_its_entry() {
    let code = r#"Manipulate[f,
      {{f, TraditionalForm[x^2]}, {TraditionalForm[x^2], TraditionalForm[x^3]}}
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let mut state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    state.apply_saved_variables(&[(
      "f".to_string(),
      "TraditionalForm[x^3]".to_string(),
    )]);
    let selected = state.controls.iter().find_map(|c| match c {
      ControlState::Discrete {
        current_index,
        overflow,
        ..
      } => Some((*current_index, overflow.clone())),
      _ => None,
    });
    assert_eq!(selected, Some((1, None)));
  }

  /// A hidden `ControlType -> None` variable whose default needs helper code
  /// the Studio cannot run reopens on the value saved in the notebook's
  /// widget dump, not on the failed recomputation.
  #[test]
  fn saved_value_restores_hidden_state_variable() {
    let code = r#"Manipulate[
      If[flag, pics[[1]], pics[[2]]],
      {{pics, unavailableHelper[3]}, ControlType -> None},
      {{flag, True}, {True, False}}
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let mut state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    assert!(state.graphics_handle.is_none());
    state.apply_saved_variables(&[(
      "pics".to_string(),
      "{Graphics[Circle[{0, 0}, 1]], Graphics[Disk[{0, 0}, 2]]}".to_string(),
    )]);
    assert_eq!(state.error, None);
    assert!(state.graphics_handle.is_some(), "saved picture is shown");
  }

  /// A saved locator position outside the range derived from its initial
  /// value stays reachable: the 2D slider's range grows to include it.
  #[test]
  fn saved_locator_position_outside_range_widens_slider() {
    let code = r#"Manipulate[
      Graphics[Point[pt]],
      {{pt, {1, 1}}, Locator}
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let mut state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    state.apply_saved_variables(&[("pt".to_string(), "{-30, 50}".to_string())]);
    let Some(ControlState::Slider2D {
      x,
      y,
      x_min,
      x_max,
      y_min,
      y_max,
      ..
    }) = state.controls.first()
    else {
      panic!("expected a 2D slider");
    };
    assert_eq!((*x, *y), (-30.0, 50.0));
    assert!(*x_min <= -30.0 && *x_max >= 1.0);
    assert!(*y_min <= 0.0 && *y_max >= 50.0);
  }

  /// A body that caches a rendered plot in hidden state (recomputing it only
  /// when a control changed) must still find the picture on the next frame:
  /// the state holds a `-Graphics-` placeholder that cannot be parsed back.
  #[test]
  fn cached_graphic_in_hidden_state_survives_reevaluation() {
    let code = r#"Manipulate[
      If[seen != n, seen = n; cache = Plot[x, {x, 0, n}]];
      Column[{cache, "caption"}],
      {{seen, 0}, ControlType -> None},
      {{cache, 0}, ControlType -> None},
      {n, 1, 3, 1}
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let mut state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    assert!(
      state.graphics_handle.is_some(),
      "first frame draws the plot"
    );
    // Nothing changed, so the body reuses `cache` from the saved state.
    state.reevaluate();
    assert_eq!(state.error, None);
    assert_eq!(
      state.text_output, None,
      "the cached plot must not print as a symbol name"
    );
    assert!(
      state.graphics_handle.is_some(),
      "second frame keeps the plot"
    );
  }

  /// Two SetterBar rows over disjoint choices can share one variable; the
  /// variable starts at the *first* row's initial value (Wolfram keeps one
  /// `var$$ = <first initial>`), so the second row shows nothing selected.
  #[test]
  fn shared_variable_rows_start_from_first_initial_value() {
    let code = r#"Manipulate[
      n,
      {{n, 10, "low"}, {10 -> "a", 20 -> "b"}, ControlType -> SetterBar},
      {{n, 30, "high"}, {30 -> "c", 40 -> "d"}, ControlType -> SetterBar}
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    let value = state
      .bindings()
      .into_iter()
      .find(|(n, _)| n == "n")
      .map(|(_, v)| v);
    assert_eq!(value.as_deref(), Some("10"));
  }

  use super::*;

  /// A Demonstration's setter-bar choices are typeset box labels
  /// (`DisplayForm[SubsuperscriptBox["C", 3, "+"]]`). The box heads must fold
  /// into Unicode scripts like `Subscript`/`Superscript` do, rather than
  /// showing the raw `SubsuperscriptBox[C, 3, +]` source.
  #[test]
  fn setter_labels_typeset_subsuperscript_boxes() {
    let expr = woxi::interpret_to_expr(
      "Manipulate[
        d,
        {{d, 1, \"\"}, {
          1 -> DisplayForm[SubscriptBox[\"A\", \"2\"]],
          2 -> DisplayForm[SubsuperscriptBox[\"C\", \"3\", \"+\"]],
          3 -> DisplayForm[SuperscriptBox[\"B\", \"2\"]]
        }, ControlType -> SetterBar}
      ]",
    )
    .expect("parse Manipulate expr");
    let state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    let labels = state
      .controls
      .iter()
      .find_map(|c| match c {
        ControlState::Discrete { value_labels, .. } => {
          Some(value_labels.clone())
        }
        _ => None,
      })
      .expect("setter control");
    assert_eq!(labels, vec!["A\u{2082}", "C\u{2083}\u{207A}", "B\u{00B2}"]);
  }

  /// A slider label typed as a subscripted `π` reads back as the symbol
  /// `Pi`; it must still be drawn as the glyph, not the word "Pi".
  #[test]
  fn control_label_typesets_pi_subscript_as_glyph() {
    let code = r#"Manipulate[p, {{p, 0.5, "\!\(\*SubscriptBox[\(π\), \(1\)]\)"}, 0, 1, 0.1}]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    let dump = format!("{:?}", state.controls);
    assert!(dump.contains("π"), "{dump}");
    assert!(!dump.contains("Pi"), "{dump}");
  }

  /// A slider over a variable written with a private-use named character
  /// (`\\[DoubleStruckCapitalC]`, U+F7A6) is labelled with the glyph a
  /// normal font has (`ℂ`), not the blank box of the raw code point.
  #[test]
  fn default_control_label_draws_private_use_symbol_as_glyph() {
    let code = "Manipulate[\\[DoubleStruckCapitalC] + \\[DoubleStruckCapitalR], \
      {\\[DoubleStruckCapitalC], 1, 5}, {\\[DoubleStruckCapitalR], 0, 1}]";
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    let dump = format!("{:?}", state.controls);
    assert!(dump.contains("label: \"ℂ\""), "{dump}");
    assert!(dump.contains("label: \"ℝ\""), "{dump}");
    assert!(!dump.contains("label: \"\u{f7a6}\""), "{dump}");
  }

  /// A choice label whose box code typesets a lone closing parenthesis
  /// (`\!\(\*SubscriptBox[\()\), \(4\)]\)`, as in a chemical formula
  /// `C(CH3)4`) must fold into a subscript like any other script, not leak
  /// the raw `SubscriptBox[), 4]` source.
  #[test]
  fn setter_labels_keep_lone_paren_box_atoms() {
    let expr = woxi::interpret_to_expr(
      r#"Manipulate[
        d,
        {{d, 1, ""}, {
          1 -> "C(\!\(\*SubscriptBox[\(CH\), \(3\)]\)\!\(\*SubscriptBox[\()\), \(4\)]\)"
        }, ControlType -> SetterBar}
      ]"#,
    )
    .expect("parse Manipulate expr");
    let state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    let labels = state
      .controls
      .iter()
      .find_map(|c| match c {
        ControlState::Discrete { value_labels, .. } => {
          Some(value_labels.clone())
        }
        _ => None,
      })
      .expect("setter control");
    assert_eq!(labels.len(), 1);
    assert!(!labels[0].contains("SubscriptBox"), "{}", labels[0]);
    assert!(labels[0].starts_with("C(CH\u{2083}"), "{}", labels[0]);
  }

  /// Checked a randomly-sampled Wolfram Demonstrations Project notebook
  /// ("Selective Resizing of Images") whose paired sliders each bound the
  /// other through a `Dynamic[…]` limit and one of them counts *down*:
  /// `{{hi, 114, "R"}, Dynamic[w - lo], 25, -1}`. The reversed pair
  /// (dynamic start above the fixed end) is sorted at parse time, which used
  /// to leave the dynamic code attached to the lower end — so the first
  /// re-resolution pushed the slider's minimum to the dynamic value, flipped
  /// the range and clamped the authored start (114) to 208.
  #[test]
  fn reversed_slider_keeps_dynamic_bound_on_its_own_end() {
    let expr = woxi::interpret_to_expr(
      "Manipulate[
        {w, lo, hi},
        {{lo, 92}, 25, Dynamic[w - hi], 1},
        {{hi, 114}, Dynamic[w - lo], 25, -1},
        {{w, 300}, None}
      ]",
    )
    .expect("parse Manipulate expr");
    let state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    let current = |name: &str| {
      state
        .controls
        .iter()
        .find(|c| c.name() == name)
        .expect("control")
        .current_code()
    };
    assert_eq!(current("lo"), "92");
    assert_eq!(current("hi"), "114");
  }

  /// A Demonstration declares a variable's visible control inside a
  /// `Row[{Control[…], …}]` and afterwards lists the bare
  /// `{var, ControlType -> None}` (no initial value, no domain) just to
  /// declare it. That bare entry must not reset the control's value to
  /// `Null`; the variable keeps the control's own default.
  #[test]
  fn bare_hidden_spec_after_control_keeps_control_default() {
    let code = r#"Manipulate[
      {count, tag},
      Row[{Control[{{count, 4, "count"}, 2, 9, 1,
        ControlType -> PopupMenu}], Spacer[10]}],
      {{tag, 1}, ControlType -> None},
      {count, ControlType -> None}
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    let ctrl = state
      .controls
      .iter()
      .find(|c| c.name() == "count")
      .expect("count control");
    assert_eq!(ctrl.current_code(), "4");
    assert!(
      state.state.iter().all(|(n, _)| n != "count"),
      "bare hidden spec must not become state: {:?}",
      state.state
    );
  }

  /// An interval control whose bounds are wrapped in `QuantityMagnitude`
  /// (a plain number is its own magnitude) must still be a real
  /// `IntervalSlider`; it used to vanish because the bounds stayed symbolic.
  #[test]
  fn interval_slider_with_quantity_magnitude_bounds() {
    let code = r#"Manipulate[
      {lam},
      Row[{Control[{{lam, {0.2, 0.3}, "volume"}, QuantityMagnitude[0.1],
        QuantityMagnitude[1.0] - 0.05, ControlType -> IntervalSlider}]}]
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    assert!(
      state.controls.iter().any(|c| matches!(
        c,
        ControlState::IntervalSlider { min, max, .. }
          if (*min - 0.1).abs() < 1e-9 && (*max - 0.95).abs() < 1e-9
      )),
      "expected an IntervalSlider over 0.1..0.95: {:?}",
      state.controls.iter().map(|c| c.name()).collect::<Vec<_>>()
    );
  }

  /// A slider whose whole range is one `Dynamic[Switch[…]]` handing back
  /// both ends as a spliced `Apply[Sequence, {lo, hi}]` pair must be a real
  /// slider over the active branch's range that follows the other control
  /// (it used to vanish, with its variable frozen at the initial value). A
  /// `HoldForm[m]` caption names the symbol rather than printing the call.
  #[test]
  fn single_dynamic_range_slider_follows_selector() {
    let code = r#"Manipulate[
      Plot[a x, {x, -1, 1}],
      {{mode, small}, {small, large}},
      {{pos, 1, "p"}, Dynamic[Switch[mode,
        small, Apply[Sequence, {-2, 3}],
        large, Apply[Sequence, {-20, 30}]]], ControlType -> LabeledSlider},
      FrameLabel -> Row[{HoldForm[a], " caption"}]
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let mut state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    assert_eq!(state.error, None);
    let range = |state: &ManipulateState| {
      state.controls.iter().find_map(|c| match c {
        ControlState::Continuous { name, min, max, .. } if name == "pos" => {
          Some((*min, *max))
        }
        _ => None,
      })
    };
    assert_eq!(range(&state), Some((-2.0, 3.0)));
    assert!(
      matches!(
        state.display_trees.first(),
        Some(DisplayNode::Row(parts))
          if matches!(parts.first(), Some(DisplayNode::Static { text, .. }) if text == "a")
      ),
      "{:?}",
      state.display_trees
    );
    let selector = state
      .controls
      .iter()
      .position(|c| c.name() == "mode")
      .expect("mode control");
    let large = match &state.controls[selector] {
      ControlState::Discrete { value_labels, .. } => value_labels[1].clone(),
      other => panic!("expected a discrete control, got {other:?}"),
    };
    assert!(state.select_discrete(selector, &large));
    state.reevaluate();
    assert_eq!(range(&state), Some((-20.0, 30.0)));
  }

  /// A control row followed by an empty string (`Row[{Control[…], …}] ""`,
  /// which the front end writes to keep a trailing blank) is the product of
  /// the row and `""`. Its controls must still become real, bound controls
  /// instead of the whole product being treated as an inert display.
  #[test]
  fn control_row_times_empty_string_still_yields_controls() {
    let code = r#"Manipulate[
      If[flag, {r, k}, {k, r}],
      Column[{
        Row[{Control[{{flag, True, ""},
          {True -> "first", False -> "second"}, Setter}], Spacer[8],
          Control[{{r, 3}, 1, 9, 1}]}] "",
        Grid[{{"k", Spacer[4], Control[{{k, 2, "k"}, 0, 5, 1}]}}]
      }]
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    let names: Vec<String> = state
      .controls
      .iter()
      .map(|c| c.name().to_string())
      .collect();
    for n in ["flag", "r", "k"] {
      assert!(names.iter().any(|x| x == n), "missing {n}: {names:?}");
    }
  }

  /// A row of `Control[{{var, default, "label"}, Checkbox}]` cells inside a
  /// `Grid` (with `Spacer` separators and a trailing empty cell left by a
  /// dangling comma) must produce one real checkbox control per variable,
  /// not static displays of their current values.
  #[test]
  fn checkbox_controls_in_a_grid_row_become_real_controls() {
    let code = r#"Manipulate[
      If[a || b, {t, c}, {}],
      Delimiter,
      Grid[{{
        Control[{{a, True, "first"}, Checkbox}],
        Spacer[15],
        Control[{{b, False, "second"}, Checkbox}],
        Spacer[15],
        Control[{{c, False, "third"}, Checkbox}],
      }}],
      {{t, 1, "turn"}, 0, 2, Enabled -> Dynamic[(a || b)]}
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    let names: Vec<String> = state
      .controls
      .iter()
      .map(|c| c.name().to_string())
      .collect();
    for n in ["a", "b", "c", "t"] {
      assert!(names.iter().any(|x| x == n), "missing {n}: {names:?}");
    }
  }

  /// A `Framed[Column[…]]` / `Framed[Grid[…]]` control panel holds real
  /// controls, including `ControlType -> Checkbox` cells and sliders written
  /// as `Slider[Dynamic[v], {min, max, step}, …]&` builders.
  #[test]
  fn framed_panel_holds_checkbox_and_slider_builder_controls() {
    let code = r#"Manipulate[
      {flag, ang, size},
      Framed[Grid[{
        {Style["options", Bold], SpanFromLeft},
        {Control[{{flag, True, "flag"}, {True, False}, ControlType -> Checkbox}]}}]],
      Framed[Column[{
        Control[{{ang, 0, "angle"},
          Slider[Dynamic[ang], {0, 359, 1}, Appearance -> "Labeled"] &}],
        Control@{{size, 5, "size"},
          Slider[Dynamic[size], {1, 15, 2}, ImageSize -> 80] &}}]]
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    for n in ["flag", "ang", "size"] {
      assert!(state.controls.iter().any(|c| c.name() == n), "missing {n}");
    }
    let bounds = state.controls.iter().find_map(|c| match c {
      ControlState::Continuous {
        name,
        min,
        max,
        step,
        ..
      } if name == "size" => Some((*min, *max, *step)),
      _ => None,
    });
    assert_eq!(bounds, Some((1.0, 15.0, 2.0)));
  }

  /// A Demonstrations idiom declares a control's own default
  /// (`Control[{{n, 2, "n"}, …}]`) and then redeclares the same variable's
  /// *actual* starting value via a separate, hidden `{{n, 3}, None}` spec —
  /// so the variable ends up tracked both by its visible control and by
  /// `ManipulateState::state`. The widget's own selection must still track
  /// whatever the body actually binds `n` to (3, from the override spec, as
  /// narrowed by `If[n < from, n = 3]`), not drift to the last entry of its
  /// choice list the way a stale `current_index` did before the two tracks
  /// were kept in sync.
  #[test]
  fn discrete_control_syncs_with_shadowing_state_override() {
    let expr = woxi::interpret_to_expr(
      "Manipulate[
        from = Switch[gr, 1, 2, 2, 3];
        upto = Switch[gr, 1, 4, 2, 4];
        If[n > upto, n = 2];
        If[n < from, n = 3];
        {n, gr},
        Control[{{n, 2, \"n\"}, Dynamic[Range[from, upto]], Setter}],
        {{from, 2}, None},
        {{upto, 5}, None},
        {{n, 3}, None},
        {{gr, 2}, None}
      ]",
    )
    .expect("parse Manipulate expr");
    let state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");

    let n_ctrl = state
      .controls
      .iter()
      .find(|c| c.name() == "n")
      .expect("n control");
    let n_state = state
      .state
      .iter()
      .find(|(name, _)| name == "n")
      .map(|(_, v)| v.as_str());

    // The widget's own selection and the shadowing state entry must agree
    // with each other, and both with the value the body actually bound.
    assert_eq!(n_ctrl.current_code(), "3");
    assert_eq!(n_state, Some("3"));
  }

  /// As part of a scheduled QA routine, Woxi Studio was tested against a
  /// randomly sampled Wolfram Demonstration notebook ("Chip Stack Game")
  /// whose action row is `Row[{Button["have a go", …, Enabled ->
  /// Dynamic[gameinplay]], Button["restart", …]}]` — the first button
  /// disables itself once a round-limit variable flips to `False`, the
  /// second has no `Enabled` option at all. This is a self-authored,
  /// construct-equivalent example (a generic step counter, not the
  /// specific Demonstration's names, values or wording, which are
  /// copyrighted).
  ///
  /// Regression: `ManipulateControl::Button` had no field for its own
  /// `Enabled` option, and the two call sites that build one
  /// (`Button[label, action, opts…]` and the JSON widget serializer) both
  /// dropped every option past `action` on the floor — so the value never
  /// existed to look up in the first place. Even if it had, a `Button`
  /// binds no variable (`ControlState::name()` returns `""` for one), so
  /// the ordinary name-keyed `spec.control_enabled` lookup used for every
  /// other control type could never find it, and two buttons with two
  /// different conditions would collide on that one empty-string key.
  /// The button stayed permanently clickable, so a "have a go"-style
  /// button never grays out once its own game-over condition holds,
  /// letting the underlying state run past the point the Demonstration
  /// means for it to stop.
  #[test]
  fn button_with_own_enabled_condition_disables_independently_of_sibling() {
    let code = r#"Manipulate[
      Graphics[{}],
      {{steps, 0}, ControlType -> None},
      {{active, True}, {True, False}, ControlType -> None},
      Row[{
        Button["step",
          {
            steps = steps + 1;
            If[steps >= 2, active = False]
          },
          Enabled -> Dynamic[active]
        ],
        Button["reset", {steps = 0; active = True}]
      }]
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let mut state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");

    let button_indices: Vec<usize> = state
      .controls
      .iter()
      .enumerate()
      .filter_map(|(i, c)| {
        matches!(c, ControlState::Button { .. }).then_some(i)
      })
      .collect();
    assert_eq!(
      button_indices.len(),
      2,
      "both Row-grouped buttons must become their own Button controls, \
       got: {:?}",
      state.controls
    );
    let (step_idx, reset_idx) = (button_indices[0], button_indices[1]);

    assert!(
      state.control_is_enabled[step_idx],
      "the gated button starts enabled while `active` is still True"
    );
    assert!(
      state.control_is_enabled[reset_idx],
      "a button with no Enabled option must always be enabled"
    );

    // Press "step" until the action flips `active` to False.
    for _ in 0..2 {
      let ControlState::Button { action, .. } = &state.controls[step_idx]
      else {
        panic!("expected a Button control");
      };
      let action = action.clone();
      state.apply_button_action(&action);
    }

    assert!(
      !state.control_is_enabled[step_idx],
      "the gated button must disable itself once its own Enabled \
       condition goes False, got: {:?}",
      state.control_is_enabled
    );
    assert!(
      state.control_is_enabled[reset_idx],
      "the ungated sibling button must stay enabled regardless of the \
       other button's condition — the two must not collide on a shared \
       empty-string lookup key"
    );

    // "reset" has no Enabled option, so it must still fire and clear the
    // other button's condition again.
    let ControlState::Button { action, .. } = &state.controls[reset_idx] else {
      panic!("expected a Button control");
    };
    let action = action.clone();
    state.apply_button_action(&action);
    assert!(
      state.control_is_enabled[step_idx],
      "resetting `active` back to True must re-enable the gated button"
    );
  }

  /// Checked a randomly-sampled Wolfram Demonstrations Project notebook
  /// ("Mixing and Infection in a Two-Group SIS Model") whose control panel
  /// lays its two sliders out as `Row[{Style[…], Spacer[…],
  /// Column[{Control[…], Control[…]}]}]` instead of bare top-level specs,
  /// and whose `Initialization`-defined helper calls `NSolve[{eqns…, 0 <
  /// v1 < 1, 0 < v2 < 1}, Reals]` — the domain-only, *no explicit variable
  /// list* form of `NSolve`/`Solve` that auto-detects the unknowns from the
  /// equations. Independently written, not copied from any specific
  /// Demonstration: different helper name, equations and preset values
  /// throughout.
  ///
  /// Regression coverage for the `NSolve`/`Solve` domain-only form itself
  /// (see the focused core-level tests in
  /// `tests/interpreter_tests/algebra.rs`), plus `LineLegend`/
  /// `LegendFunction`, `Darker`, `Bookmarks`, and `ControlPlacement ->
  /// Top`, exercised together the way the Demonstration combines them.
  #[test]
  fn nested_row_column_layout_with_domain_only_nsolve_helper() {
    let code = r#"Manipulate[
      Module[{p1s, p2s},
        {p1s, p2s} = Transpose[
          Table[
            If[a b > 1 || frac < 1 - (1 - a b)/(a + b - 2 a b),
              EquilibriumFractions[a, b, frac],
              {0, 0}
            ],
            {frac, 2/100, 1, 2/100}
          ]
        ];
        ListPlot[{p2s, (p1s + p2s)/2, p1s},
          DataRange -> {2, 100},
          PlotStyle -> {Red, Darker[Green], Blue},
          PlotLegends -> LineLegend[{"p2", "avg", "p1"},
            LegendFunction -> (Framed[#, RoundingRadius -> 5] &)],
          FrameLabel -> {"mixing (%)", None},
          Frame -> True,
          PlotLabel -> "equilibrium fractions",
          ImageSize -> {520, 400}
        ]
      ],
      Row[{
        Style["rate ", Bold],
        Spacer[20],
        Column[{
          Control[{{a, 0.5, "a"}, 0.01, 5, Appearance -> "Labeled"}],
          Control[{{b, 2, "b"}, 1.01, 5, Appearance -> "Labeled"}]
        }]
      }],
      ControlPlacement -> Top,
      SaveDefinitions -> True,
      Bookmarks -> {
        "preset1" :> {a = 0.25, b = 2},
        "preset2" :> {a = 1, b = 4}
      },
      Initialization :> (
        EquilibriumFractions[a_, b_, frac_] := Module[{v1, v2},
          First[
            {v1, v2} /. NSolve[
              {
                (1 - v1) a (frac v2 + (1 - frac) v1) - v1 == 0,
                (1 - v2) b (frac v1 + (1 - frac) v2) - v2 == 0,
                0 < v1 < 1, 0 < v2 < 1
              },
              Reals
            ]
          ]
        ]
      )
    ]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let state = ManipulateState::from_expr(&expr)
      .expect("the nested Row/Column layout should build a ManipulateState");

    assert_eq!(
      state.error, None,
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert_eq!(
      state.control_placement,
      ControlPlacement::Top,
      "ControlPlacement -> Top must survive into the widget state"
    );

    // The Style["rate ", Bold] heading and the two sliders pulled out of
    // the nested Row/Column layout must all show up as controls.
    let names: Vec<&str> = state.controls.iter().map(|c| c.name()).collect();
    assert_eq!(names, ["", "a", "b"]);

    assert!(
      state.graphics_handle.is_some(),
      "the equilibrium-fraction plot should render"
    );

    // A `Solve`/`NSolve` call that failed to auto-detect its variables
    // under the domain-only two-argument form prints Wolfram-style
    // `Tag::name:` error messages instead of solving, and the body ends up
    // unable to plot real numbers — re-run the body directly and check its
    // SVG output carries none of those markers, nor a bare unevaluated
    // `NSolve[…]`/`Solve[…]` call.
    let bindings: Vec<(String, String)> = state
      .controls
      .iter()
      .filter(|c| c.binds_variable())
      .map(|c| (c.name().to_string(), c.current_code()))
      .chain(state.state.iter().cloned())
      .collect();
    let svg = woxi::with_scoped_globals(&bindings, || {
      woxi::interpret_with_stdout(&state.body)
    })
    .expect("body evaluates")
    .graphics
    .expect("the equilibrium-fraction plot should render");
    for marker in [
      "::svars", "::reps", "::nmtx", "::shape", "NSolve[", "Solve[",
    ] {
      assert!(
        !svg.contains(marker),
        "the domain-only NSolve inside the Manipulate must actually solve, \
         not leak an error/unevaluated call into the plot ({marker:?} \
         found): {svg}"
      );
    }
  }

  /// A Demonstration whose "zoom" checkbox is written `Tooltip[Control[{{z,
  /// False, ""}, {True, False}}], "hint"]`: the tooltip only adds a hover
  /// hint, so `z` must be a real control the body can read. Before, the
  /// whole `Tooltip[…]` was treated as a display, `z` stayed unbound, and
  /// the `If[z, …]` in the body never evaluated.
  #[test]
  fn tooltip_wrapped_control_binds_its_variable() {
    let code = r#"Manipulate[
      Module[{m},
        If[zoomed, m = 10 k, m = k];
        Column[{Text[m]}]
      ],
      {{k, 2, "k"}, 1, 5, 1},
      Row[{"zoom ",
        Tooltip[Control[{{zoomed, False, ""}, {True, False}}], "zoom"]}]
    ]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let state = ManipulateState::from_expr(&expr)
      .expect("the Tooltip-wrapped control should build a ManipulateState");
    assert_eq!(state.error, None, "body must evaluate cleanly");
    assert!(
      state.controls.iter().any(|c| c.name() == "zoomed"),
      "Tooltip[Control[…]] must contribute a `zoomed` control"
    );
  }

  /// A control column whose entries are `If[cond, Identity, Invisible]@
  /// Control[…]` — controls shown only in one mode. They must be real,
  /// bound controls (the body reads `w`), each visible only in its mode.
  /// Before, the whole `If[…]@Control[…]` was treated as a display, so `w`
  /// stayed unbound and the body drew with a symbolic value.
  #[test]
  fn conditionally_invisible_controls_in_a_column_are_real_controls() {
    let code = r#"Manipulate[
      If[mode, Text[w], Text[v]],
      {mode, True, ControlType -> None},
      Dynamic[Column[{
        If[mode, Identity, Invisible]@Control[{{v, 2, "v"}, 0, 5, 1}],
        If[!mode, Identity, Invisible]@Control[{{w, 3, "w"}, 0, 5, 1}]}]]
    ]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let state = ManipulateState::from_expr(&expr)
      .expect("the conditional controls should build a ManipulateState");
    assert_eq!(state.error, None, "body must evaluate cleanly");
    let names: Vec<_> = state.controls.iter().map(|c| c.name()).collect();
    assert!(
      names.contains(&"v") && names.contains(&"w"),
      "both conditional controls must be real controls, got {names:?}"
    );
    let vis = |n: &str| {
      let i = names.iter().position(|x| *x == n).unwrap();
      state.control_is_visible[i]
    };
    assert!(vis("v"), "`v` is shown while mode is True");
    assert!(!vis("w"), "`w` is hidden while mode is True");
  }

  /// A `Dynamic@Column` control area whose second row is a nested
  /// `If[mode == 1, Control[…], If[mode == 2, d = 2.5; Control[…],
  /// d = 1; Control[…]]]`: each mode brings its own slider, and the
  /// branches' assignments only seed the variable. All three sliders are
  /// real controls (`d` is declared by two branches and registered once),
  /// shown only in their own mode — before, the whole `If` was drawn as
  /// inert `Control[…]` text and `p` stayed unbound.
  #[test]
  fn nested_if_with_compound_branches_holds_real_controls() {
    let code = r#"Manipulate[
      If[mode == 1, Text[p], Text[d]],
      Dynamic@Column[{
        Row[{Control[{{mode, 1, ""}, {1 -> "a", 2 -> "b", 3 -> "c"}, ControlType -> RadioButtonBar}], Spacer[40]}],
        If[mode == 1, Control[{{p, 0.5, "directrix"}, 0.4, 1, 0.1}],
          If[mode == 2, d = 2.5; Control[{{d, 3.5, "sum"}, 2.33, 3.6}],
            d = 1; Control[{{d, 1, "difference"}, 0.7, 1.16}]]]}]
    ]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let state = ManipulateState::from_expr(&expr)
      .expect("the conditional controls should build a ManipulateState");
    assert_eq!(state.error, None, "body must evaluate cleanly");
    let names: Vec<_> = state.controls.iter().map(|c| c.name()).collect();
    assert!(
      names.contains(&"p") && names.contains(&"d") && names.contains(&"mode"),
      "every branch's control must be a real control, got {names:?}"
    );
    let vis = |n: &str| {
      let i = names.iter().position(|x| *x == n).unwrap();
      state.control_is_visible[i]
    };
    assert!(vis("p"), "`p` is shown in mode 1");
    assert!(!vis("d"), "`d` is hidden in mode 1");
  }

  /// A `TabView` whose panes are `Grid`s with `SetterBar[Dynamic[var],
  /// choices] /. labels` cells (a key picker showing note names for numeric
  /// choices) and a `Sound[…]` row. The bars are lifted into the control
  /// panel with their labels, each shown only on its own tab, and the
  /// selected tab's grid is drawn as the picture instead of leaving the
  /// symbolic `TabView[…]` text.
  #[test]
  fn tabview_grid_panes_with_body_setter_bars() {
    let code = r#"Manipulate[
      TabView[{
        {"Major", "Major" -> Text@Grid[{
          {SetterBar[Dynamic[ka], {0, 7, 2}] /. names, "x"},
          {Sound[{SoundNote[ka, 0.2], SoundNote[ka + 4, 0.2]}], "y"}}]},
        {"Minor", "Minor" -> Grid[{
          {SetterBar[Dynamic[kb], {9, 4}] /. names}}]}
      }, Dynamic[tab]],
      {{tab, "Major"}, {"Major", "Minor"}, ControlType -> None},
      {ka, 0, ControlType -> None},
      {kb, 9, ControlType -> None},
      Initialization :> (names = {0 -> "C", 7 -> "G", 2 -> "D", 9 -> "A", 4 -> "E"})
    ]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let state = ManipulateState::from_expr(&expr)
      .expect("the tabbed setter-bar layout should build a ManipulateState");
    assert_eq!(state.error, None, "{:?}", state.error);

    let names: Vec<&str> = state.controls.iter().map(|c| c.name()).collect();
    assert_eq!(names, ["tab", "ka", "kb"]);
    let labels = |name: &str| {
      let c = state.controls.iter().find(|c| c.name() == name).unwrap();
      format!("{c:?}")
    };
    assert!(
      labels("ka").contains("\"C\", \"G\", \"D\""),
      "setter bar choices must carry their note-name labels: {}",
      labels("ka")
    );
    assert!(
      !state.body.contains("SetterBar"),
      "lifted setter bars must not stay in the body: {}",
      state.body
    );
    assert!(
      state.graphics_handle.is_some(),
      "the selected tab's grid must render as a picture, got text {:?}",
      state.text_output
    );
  }

  /// Checked a randomly-sampled Wolfram Demonstrations Project notebook
  /// ("Newton's Polynomial Solver") whose control panel is written as
  /// `Text@Grid[{{header, header}, {Control[…], Control[…]}, …}]` — a
  /// custom control table rather than bare top-level specs or a plain
  /// `Row`/`Column` grouping. Independently written, not copied from any
  /// specific Demonstration: different variables, bounds and headers
  /// throughout.
  ///
  /// Regression coverage for `Text[…]` wrapping a `Grid[…]` of inline
  /// `Control[…]` cells: `Text[…]` is not itself a `Row`/`Column`/`Grid`
  /// layout container, so without unwrapping it first, the whole
  /// `Text[Grid[…]]` argument used to be swallowed whole by the
  /// static-heading path (`is_manipulate_annotation_head`), stringifying
  /// every slider into inert label text instead of building a real
  /// control for it.
  #[test]
  fn text_grid_control_table_layout() {
    let code = r#"Manipulate[
      Plot[p*x^2 + q*x, {x, -5, 5}],
      Text@Grid[{
        {"coefficient", "value"},
        {Control[{{p, 1, "p"}, -3, 3, 0.5, Appearance -> "Labeled"}],
         Control[{{q, 0, "q"}, -3, 3, 0.5, Appearance -> "Labeled"}]}
      }]
    ]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let state = ManipulateState::from_expr(&expr)
      .expect("the Text[Grid[…]] control table should build a ManipulateState");

    assert_eq!(
      state.error, None,
      "body must evaluate cleanly: {:?}",
      state.error
    );

    // The two header strings become heading rows, and the two `Control[…]`
    // cells become real sliders bound to `p` and `q` — not a single
    // flattened text heading with no bound variables at all.
    let names: Vec<&str> = state.controls.iter().map(|c| c.name()).collect();
    assert_eq!(names, ["", "", "p", "q"]);
    assert!(
      state.controls[2].binds_variable() && state.controls[3].binds_variable(),
      "both Grid-embedded Control[…] cells must become real, bound sliders: \
       {:?}",
      state.controls
    );

    assert!(state.graphics_handle.is_some(), "the plot should render");
  }

  /// Regression coverage for a hidden-state `Manipulate` whose whole control
  /// panel is `Text[Row[{Grid[{{label, Slider[Dynamic[v], …]}}]}]]`: the
  /// `Text[…]` wrapper used to turn the display into a static picture, so the
  /// slider could not be dragged. The display tree must keep a live slider.
  #[test]
  fn text_wrapped_row_grid_keeps_live_slider() {
    let code = r#"Manipulate[
      Plot[g*x, {x, 0, 1}],
      Text[Row[{Grid[{{"gain", Spacer[5], Slider[Dynamic[g], {1, 5}]}}]}]],
      {{g, 2}, None}
    ]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let state = ManipulateState::from_expr(&expr)
      .expect("the Text[Row[Grid[…]]] panel should build a ManipulateState");

    fn has_slider(node: &DisplayNode) -> bool {
      match node {
        DisplayNode::Slider { .. } => true,
        DisplayNode::Row(c) | DisplayNode::Column(c) => {
          c.iter().any(has_slider)
        }
        DisplayNode::Grid(rows) => rows.iter().flatten().any(has_slider),
        DisplayNode::Panel(inner) => has_slider(inner),
        _ => false,
      }
    }
    assert!(
      state.display_trees.iter().any(has_slider),
      "Text-wrapped Row/Grid must keep its slider live: {:?}",
      state.display_trees
    );
  }

  /// Checked a randomly-sampled Wolfram Demonstrations Project notebook
  /// ("Crystal Lattices in Reciprocal Space") whose control panel is two
  /// collapsible `OpenerView[{label, content}]` sections: one of preset
  /// buttons assigning several variables at once, one holding a `Grid` of
  /// `Control[…]` sliders. Independently written: different variables and
  /// helper names.
  ///
  /// Regression coverage for `OpenerView` not being flattened like
  /// `Column`: its controls were shown as inert text, so the variables they
  /// declare stayed symbolic and the body rendered degenerate output.
  #[test]
  fn opener_view_sections_hold_real_controls() {
    let code = r#"Manipulate[
      Graphics[Line[{{0, 0}, {a, b}}], PlotRange -> 2],
      OpenerView[{"presets",
        Row[{Button["unit", {a, b} = {1, 1}], Button["flat", {a, b} = {1, 0}]}]}],
      OpenerView[{"manual",
        Grid[{{"a", Control[{{a, 1, ""}, -1, 1, 1}]},
              {"b", Control[{{b, 0, ""}, -1, 1, 1}]}}]}]
    ]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let state = ManipulateState::from_expr(&expr)
      .expect("OpenerView control sections should build a ManipulateState");
    assert_eq!(state.error, None, "{:?}", state.error);
    let bound: Vec<&str> = state
      .controls
      .iter()
      .filter(|c| c.binds_variable())
      .map(|c| c.name())
      .collect();
    assert_eq!(bound, ["a", "b"]);
    assert!(
      state
        .controls
        .iter()
        .any(|c| matches!(c, ControlState::Button { .. })),
      "preset buttons must become real buttons: {:?}",
      state.controls
    );
    assert!(state.graphics_handle.is_some(), "the plot should render");
  }

  /// Checked a randomly-sampled Wolfram Demonstrations Project notebook
  /// ("Vector Transformations and Eigenvectors of 2×2 Matrix") whose body
  /// builds its matrix and vector through a `Function[…][t]`/`ToExpression`
  /// indirection rather than referencing the control variables directly,
  /// and whose caption panel labels a determinant/inverse *formula* (not an
  /// actual matrix) via `MatrixForm[StringForm["(``×``)-(``×``)", Style[a,
  /// FontColor -> Red], …]]` — coloring each digit to match the slider that
  /// produced it. Independently written, not copied from any specific
  /// Demonstration: different helper name, layout and preset values
  /// throughout.
  ///
  /// Regression coverage for two bugs this uncovered in the shared
  /// typeset/SVG box pipeline (`expr_to_box_form` in
  /// `evaluator/dispatch/complex_and_special.rs`, used by both
  /// `ExportString[…, "SVG"]` and this Manipulate re-render path):
  ///
  /// 1. `MatrixForm[StringForm[…]]` — `MatrixForm` wrapping a *formula*,
  ///    not a list — fell through to the generic `FunctionCall` box
  ///    renderer, which printed the literal `MatrixForm[StringForm["…",
  ///    …]]` call (every token, including the raw template string, as its
  ///    own separate box) instead of the substituted formula. Wolfram
  ///    substitutes `StringForm` wherever it is typeset, and `MatrixForm`
  ///    on a non-list argument just displays that argument plainly.
  /// 2. `Style[value, FontColor -> color]` (or a bare color directive) was
  ///    silently discarded by `expr_to_box_form`, so every colored digit in
  ///    the formula rendered in the default text color instead of matching
  ///    its slider.
  #[test]
  fn nested_function_to_expression_matrix_with_colored_formula_captions() {
    let code = r#"Manipulate[
      Module[{m, u, w},
        m = Function[t, ToExpression[{{p, q}, {r, s}}]][t];
        u = Function[t, ToExpression[{u1, u2}]][t];
        w = m . u;
        Text@Column[{
          Framed[Labeled[
            MatrixForm[StringForm["(``\[Times]``)-(``\[Times]``)",
              Style[p, FontColor -> Red], Style[s, FontColor -> Green],
              Style[q, FontColor -> Brown], Style[r, FontColor -> Orange]]],
            "determinant formula", Top]],
          Framed[TableForm[
            Table[{Eigensystem[m][[1,n]], MatrixForm[Eigensystem[m][[2,n]]]},
              {n, 1, Length[Eigensystem[m][[1]]]}],
            TableHeadings -> {None, {"eigenvalue", "eigenvector"}}]],
          Graphics[{{Black, Arrow[{{0,0}, u}]}, {tint, Arrow[{{0,0}, w}]}},
            Axes -> True, ImageSize -> {150,150}]
        }]
      ],
      Grid[{
        {Control@{{p, 1, Style["p", Italic]}, -10, 10, 1, ImageSize -> Tiny},
         Control@{{q, 2, Style["q", Italic]}, -10, 10, 1, ImageSize -> Tiny}},
        {Control@{{r, 3, Style["r", Italic]}, -10, 10, 1, ImageSize -> Tiny},
         Control@{{s, 4, Style["s", Italic]}, -10, 10, 1, ImageSize -> Tiny}}
      }],
      {{u1, 5, Style["u1", Italic]}, -10, 10, 1, ImageSize -> Tiny},
      {{u2, 6, Style["u2", Italic]}, -10, 10, 1, ImageSize -> Tiny},
      {{tint, Red}, Red},
      ControlPlacement -> Left
    ]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let state = ManipulateState::from_expr(&expr)
      .expect("the Function/ToExpression body should build a ManipulateState");
    assert_eq!(
      state.error, None,
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the Text@Column body (formula caption + eigen-table + vector plot) \
       should render"
    );

    // p=1, q=2, r=3, s=4 (the sliders' defaults): re-render the body and
    // check the determinant formula substituted correctly and in color,
    // rather than dumping `MatrixForm[StringForm[…]]`'s literal source.
    let bindings: Vec<(String, String)> = state
      .controls
      .iter()
      .filter(|c| c.binds_variable())
      .map(|c| (c.name().to_string(), c.current_code()))
      .chain(state.state.iter().cloned())
      .collect();
    let svg = woxi::with_scoped_globals(&bindings, || {
      woxi::interpret_with_stdout(&state.body)
    })
    .expect("body evaluates")
    .graphics
    .expect("the Column body should render as a graphic");
    for marker in ["MatrixForm[", "StringForm[", "``"] {
      assert!(
        !svg.contains(marker),
        "the determinant formula must substitute, not leak {marker:?} from \
         the un-evaluated call: {svg}"
      );
    }
    assert!(
      svg.contains("(1") && svg.contains(")-(") && svg.contains("2"),
      "the substituted determinant formula '(1×4)-(2×3)' must appear: {svg}"
    );
    for color in [
      "rgb(255,0,0)",
      "rgb(0,255,0)",
      "rgb(153,102,51)",
      "rgb(255,128,0)",
    ] {
      assert!(
        svg.contains(color),
        "each `Style[…, FontColor -> …]` digit must keep its color \
         ({color} missing): {svg}"
      );
    }
  }

  /// Checked a randomly-sampled Wolfram Demonstrations Project notebook
  /// ("Rotating Wheel Illusion"), whose body is `Pane[Rotate[Graphics[…],
  /// 2 Pi t], …]` — the whole rendered picture wrapped in `Rotate` (as
  /// opposed to one shape rotated *inside* a `Graphics[{…}]` scene) so
  /// that dragging the `t` slider spins the entire wheel. Independently
  /// written, not copied from any specific Demonstration: different shape,
  /// helper names and preset values throughout.
  ///
  /// Regression coverage for a display-pipeline gap: nothing in the
  /// visual-mode rendering pipeline recognized `Rotate[graphic, angle]` as
  /// a *picture*, so the body evaluated to an unevaluated `Rotate[…]` call
  /// whose textual echo showed `Rotate[, <angle>]` (the `-Graphics-`
  /// placeholder stripped to nothing) and no graphic ever appeared —
  /// `ManipulateState::graphics_handle` stayed `None` for every value of
  /// `t`. See `rotate_graphic_svg` (`evaluator/dispatch/io_functions.rs`)
  /// and `render_rotate_if_needed` (`lib.rs`).
  #[test]
  fn rotate_wrapping_a_whole_graphic_renders_and_spins() {
    let code = r#"Manipulate[
      Pane[
        Rotate[
          Graphics[{Table[
            Rotate[Disk[{1, 0}, 0.2], 2 Pi k/n, {0, 0}],
            {k, 1, n}]}, PlotRange -> 1.5],
          2 Pi t],
        ImageSize -> {200, 200}, Alignment -> Center],
      {{n, 5, "petals"}, 3, 10, 1},
      {{t, 0, "spin"}, 0, 1, 0.01}
    ]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let state = ManipulateState::from_expr(&expr)
      .expect("the Rotate[Graphics[…], …] body should build a ManipulateState");
    assert_eq!(
      state.error, None,
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the whole-picture Rotate must render as a graphic, not fall back to \
       its textual echo: text={:?}",
      state.text_output
    );

    // Re-render at a non-zero `t` (a spun frame) and check the wheel
    // actually turned: five petals sit at 2π k/n + 2π t, not just 2π k/n.
    let bindings: Vec<(String, String)> = vec![
      ("n".to_string(), "5".to_string()),
      ("t".to_string(), "0.1".to_string()),
    ];
    let svg = woxi::with_scoped_globals(&bindings, || {
      woxi::interpret_with_stdout(&state.body)
    })
    .expect("body evaluates")
    .graphics
    .expect("the spun body should still render as a graphic");
    assert_eq!(
      svg.matches("<circle").count() + svg.matches("<ellipse").count(),
      5,
      "all five petals must still draw once spun: {svg}"
    );
  }

  /// Checked a randomly-sampled Wolfram Demonstrations Project notebook
  /// ("Nonlinear Wave Equations"), whose body is `If[twoD, DensityPlot,
  /// Plot3D][u[…] /. NDSolve[…], …]` with a `SetterBar` toggling the plot
  /// type and the space domain's two ends tied together by a periodic
  /// boundary (`u[t, x_min] == u[t, x_max]`) rather than fixed separately
  /// — independently reproduced here with a different equation, domain
  /// and control layout (a cubic nonlinearity in place of the
  /// Demonstration's own choice, and no equation-selector control).
  ///
  /// Regression coverage for `NDSolve`'s hyperbolic (second-order-in-time)
  /// solver accepting a periodic boundary at all (see the focused
  /// core-level tests alongside
  /// `pde_hyperbolic_wave_equation_with_periodic_boundary_matches_standing_wave`
  /// in `tests/interpreter_tests/calculus.rs`): a periodic equation used
  /// to be misread as an ordinary one-sided Dirichlet condition whose
  /// "value" was the other end's own unevaluated call, silently leaving
  /// the solve one boundary equation short — `NDSolve` returned
  /// unevaluated with no error message, and the widget's plot never
  /// rendered at all with the Demonstration's own default control values.
  #[test]
  fn nonlinear_wave_pde_with_periodic_boundary_and_setter_bar_plot_type() {
    let code = r#"Manipulate[
      If[twoD, DensityPlot, Plot3D][
        Evaluate[
          u[t, x] /. Quiet[NDSolve[
            {D[u[t, x], t, t] ==
               D[u[t, x], {x, 2}] - u[t, x] + u[t, x]^3/6,
             u[0, x] == amp*Cos[Pi x],
             Derivative[1, 0][u][0, x] == 0,
             u[t, -1] == u[t, 1]},
            u, {t, 0, tmax}, {x, -1, 1},
            Method -> {"MethodOfLines",
              "SpatialDiscretization" -> {"TensorProductGrid",
                "DifferenceOrder" -> "Pseudospectral",
                "MinStepSize" -> 0.2}}]]
        ],
        {x, -1, 1}, {t, 0, tmax},
        PlotRange -> All, ImageSize -> {400, 300}
      ],
      {{tmax, 1, "duration"}, 0.5, 2},
      {{amp, 1, "amplitude"}, 0.5, 2},
      {{twoD, True, "plot type"}, {True -> "2D", False -> "3D"},
        ControlType -> SetterBar}
    ]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let state = ManipulateState::from_expr(&expr)
      .expect("the periodic-boundary PDE Manipulate should build a state");

    assert_eq!(
      state.error, None,
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the density plot should render with the widget's default control \
       values"
    );

    let names: Vec<&str> = state.controls.iter().map(|c| c.name()).collect();
    assert_eq!(names, ["tmax", "amp", "twoD"]);
  }

  /// Checked a randomly-sampled Wolfram Demonstrations Project notebook
  /// ("Box-Counting Algorithm of the Hénon Map") whose dimension-estimation
  /// display mode fits a `LinearModelFit` to log-log data and captions the
  /// plot with the fitted slope and its uncertainty via
  /// `lm[{"ParameterTableEntries"}][[1,2,1]]` / `[[1,2,2]]` — a
  /// `FittedModel` property Woxi did not implement, aborting the widget
  /// with `FittedModel: unknown property "ParameterTableEntries"` the
  /// moment that mode was selected (see the regression tests alongside
  /// `LinearModelFit` in `tests/interpreter_tests/linear_algebra.rs` for the
  /// property itself). Independently written: different data, variable
  /// names and caption text throughout.
  #[test]
  fn manipulate_body_captions_fit_slope_with_parameter_table_entries() {
    let code = r#"Manipulate[
      Module[{lm, slopetext},
        lm = LinearModelFit[
          Table[{k, m*k + (-1)^k*0.2}, {k, 1, 8}], k, k];
        slopetext = Row[{"slope = ",
          SetPrecision[lm[{"ParameterTableEntries"}][[1, 2, 1]], 3],
          " +/- ",
          SetPrecision[lm[{"ParameterTableEntries"}][[1, 2, 2]], 3]}];
        Graphics[Text[slopetext]]
      ],
      {{m, 2}, 1, 5}
    ]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let state = ManipulateState::from_expr(&expr).expect("state should build");

    assert_eq!(
      state.error, None,
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the captioned graphic should render with the widget's default \
       control values"
    );
  }

  /// Checked a randomly-sampled Wolfram Demonstrations Project notebook
  /// whose body is `Module[{…}, Text@Column[{RegionPlot[…], Row[{…,
  /// var = Round[NIntegrate[Boole[region], {…}, {…}]], …}]}]]` driven by
  /// two `Control@{{var, default, "label"}, lo, hi, step, ImageSize ->
  /// Tiny, Appearance -> "Labeled"}` sliders laid out with `Spacer` — an
  /// area/volume readout computed live from a double integral of a
  /// region indicator. Independently reproduced here with a different
  /// region (an off-center disk instead of a fixed circle through its
  /// own center), different helper/variable names and a single slider —
  /// not copied from any specific Demonstration.
  ///
  /// Regression coverage for the whole `Boole`-in-a-double-`NIntegrate`
  /// path itself (see the focused core-level tests
  /// `nintegrate_iterated_boole_region_area_is_fast_and_exact` and
  /// `nintegrate_iterated_boole_off_center_region_is_fast_and_exact` in
  /// `tests/interpreter_tests/calculus.rs`): building this widget used to
  /// mean evaluating its body once with the default control values, which
  /// ran headfirst into that same hang — the whole Studio looked frozen
  /// before it ever showed a first frame.
  #[test]
  fn region_area_readout_from_double_nintegrate_with_boole_indicator() {
    let code = r#"Manipulate[
      Module[{filled},
        Text@Column[{
          RegionPlot[p^2 + (q - 3)^2 < 4 && q < cutoff, {p, -2, 2}, {q, 1, 5},
            Mesh -> None, PlotPoints -> 40, Axes -> None, Frame -> False,
            ImageSize -> 350, Epilog -> Circle[{0, 3}, 2]],
          Row[{"filled area: ",
            filled = Round[
              NIntegrate[Boole[p^2 + (q - 3)^2 < 4 && q < cutoff],
                {p, -2, 2}, {q, 1, 5}], 0.01]}]
        }, Alignment -> Center]
      ],
      Control@{{cutoff, 3, "cutoff level"}, 1, 5, 0.01,
        ImageSize -> Tiny, Appearance -> "Labeled"}
    ]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let start = std::time::Instant::now();
    let state = ManipulateState::from_expr(&expr)
      .expect("the region-area Manipulate should build a state");
    // Generous bound: this also renders the RegionPlot, not just the
    // NIntegrate, and only needs to catch a regression back to the
    // multi-minute hang the fix addresses, not enforce a tight SLA.
    assert!(
      start.elapsed().as_secs() < 20,
      "building the widget runs the body once at its default control \
       values — that must not hang on the double NIntegrate"
    );

    assert_eq!(
      state.error, None,
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the region plot should render with the widget's default control \
       values"
    );

    let names: Vec<&str> = state.controls.iter().map(|c| c.name()).collect();
    assert_eq!(names, ["cutoff"]);
  }

  /// A sampled Demonstration titles a control with `Row[{"title",
  /// Spacer[n]}]` and pins its readouts with left-aligned `Text`. The
  /// spacer is layout, so the control heading must not spell out the
  /// `Spacer[n]` head, and the left-aligned readout must be anchored at
  /// its left edge rather than centred on a guessed width.
  #[test]
  fn heading_row_with_spacer_and_left_aligned_readout() {
    let code = r#"Manipulate[
      Graphics[{Text[Row[{Spacer[30], "level = ", k}], {-1, 0}, {-1, 0}]},
        PlotRange -> {{-1, 1}, {-1, 1}}, ImageSize -> {200, 14}],
      Row[{"gain", Spacer[40]}],
      {{k, 3}, 1, 9, 1}
    ]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let state = ManipulateState::from_expr(&expr).expect("state should build");
    assert_eq!(state.error, None);
    for c in &state.controls {
      if let ControlState::Heading { label, .. } = c {
        assert!(!label.contains("Spacer"), "leaked head in {label:?}");
      }
    }
    assert!(state.graphics_handle.is_some());
  }

  /// End-to-end regression for a Wolfram Demonstrations Project idiom found
  /// opening a randomly-sampled Demonstration: a `Module[…]`
  /// unconditionally builds a couple of decorative `GraphicsComplex[
  /// PolyhedronData[name, "VertexCoordinates"], Line[PolyhedronData[name,
  /// "EdgeIndices"]]]` overlays and only shows each one when its own
  /// checkbox is on (`If[show, overlay, {}]`), with the checkboxes laid out
  /// via `Control@{{…}, {True, False}}` inside a `Grid` — one row pairing
  /// two numeric `Control@` sliders, the other a `Row[{Control@, Spacer[10],
  /// Control@}]` holding both checkboxes — plus `TrackedSymbols`.
  /// Independently reproduced here with a different base shape and
  /// different overlay roles, not copied from the source Demonstration.
  ///
  /// Regression coverage for `PolyhedronData["GreatStellatedDodecahedron" |
  /// "MathematicaSpikey", …]`: before those two entities existed, each
  /// `PolyhedronData[…]` call there stayed unevaluated (emitting a
  /// `PolyhedronData::notent` message every time the body re-ran), so
  /// toggling either checkbox re-rendered around a broken, unevaluated
  /// `GraphicsComplex` instead of the star polyhedron.
  #[test]
  fn expression_cell_control_rows_become_live_checkbox_displays() {
    // A saved Demonstration stores hand-built control rows wrapped in
    // `ExpressionCell[…]`; the wrapper must be transparent: no
    // `Manipulate::vsform` message and real checkboxes, not static text.
    let code = r#"Manipulate[
      Graphics[{If[big, Disk[{0, 0}, 1], Disk[{0, 0}, 0.5]], If[tag, Text["x", {0, 0}], {}]}],
      ExpressionCell[Row[{"  ", Grid[{{Checkbox[Dynamic[big]], "large", Checkbox[Dynamic[tag]], "tagged"}}]}]],
      {{big, True}, ControlType -> None},
      {{tag, False}, ControlType -> None}
    ]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let state = ManipulateState::from_expr(&expr).expect("state should build");

    // The `Checkbox[Dynamic[var]]` cells become real checkbox controls (one
    // per variable) rather than a static display, and nothing is left of the
    // `ExpressionCell` wrapper.
    assert_eq!(state.error, None);
    let names: Vec<&str> = state
      .controls
      .iter()
      .map(|c| c.name())
      .filter(|n| !n.is_empty())
      .collect();
    assert_eq!(names, ["big", "tag"]);
    assert!(
      state
        .displays
        .iter()
        .all(|d| !d.starts_with("ExpressionCell")),
      "wrapper must be stripped: {:?}",
      state.displays
    );
  }

  /// `RadioButton[Dynamic[{a, b}], {True, False}]` in a hand-built control
  /// panel is a live radio button: selected while the list matches, and a
  /// click assigns the whole list, so a group of them selects one of several
  /// exclusive flags.
  #[test]
  fn dynamic_list_radio_buttons_select_one_flag() {
    use woxi::functions::graphics::DisplayNode;

    let code = r#"Manipulate[
      Graphics[Text[If[a, "A", "B"]]],
      ExpressionCell[Grid[{
        {RadioButton[Dynamic[{a, b}], {True, False}], "a"},
        {RadioButton[Dynamic[{a, b}], {False, True}], "b"}}]],
      {{a, True}, ControlType -> None},
      {{b, False}, ControlType -> None}
    ]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let mut state =
      ManipulateState::from_expr(&expr).expect("state should build");
    assert_eq!(state.error, None);

    fn radios(node: &DisplayNode, out: &mut Vec<(String, bool, String)>) {
      match node {
        DisplayNode::Checkbox {
          target: Some(t),
          checked,
          on,
          ..
        } => out.push((t.clone(), *checked, on.clone())),
        DisplayNode::Grid(rows) => {
          rows.iter().flatten().for_each(|c| radios(c, out))
        }
        DisplayNode::Row(cs) | DisplayNode::Column(cs) => {
          cs.iter().for_each(|c| radios(c, out))
        }
        DisplayNode::Panel(c) => radios(c, out),
        _ => {}
      }
    }
    let collect = |state: &ManipulateState| {
      let mut out = Vec::new();
      state.display_trees.iter().for_each(|t| radios(t, &mut out));
      out
    };
    let before = collect(&state);
    assert_eq!(
      before.iter().map(|r| r.1).collect::<Vec<_>>(),
      [true, false],
      "{before:?}"
    );
    assert_eq!(before[0].0, "{a, b}");

    state.apply_display_mutation(&format!("{} = {}", before[1].0, before[1].2));
    assert_eq!(state.error, None);
    let after = collect(&state);
    assert_eq!(
      after.iter().map(|r| r.1).collect::<Vec<_>>(),
      [false, true],
      "{after:?}"
    );
  }

  /// A `Dynamic[…]` caption that reads a control (`Row[{"x", " " <>
  /// unit[lookup[sel, …]]}]`) was evaluated once while the widget was being
  /// built, before `sel` held its choice. `lookup` then returned `None` and
  /// `StringJoin` printed a bogus `StringJoin::string` message.
  #[test]
  fn dynamic_caption_is_not_evaluated_before_controls_are_bound() {
    let code = r#"Manipulate[
      Graphics[Disk[]],
      Dynamic[Row[{"unit", " " <> captionUnit[captionPick[captionSel]]}]],
      {{captionSel, "b"}, {"a", "b"}}
    ]"#;
    woxi::interpret(
      "captionPick[x_] := If[MemberQ[{\"a\", \"b\"}, x], x, None]",
    )
    .unwrap();
    // Records every evaluation that happens with the control still unbound.
    woxi::interpret("captionUnit[None] := (captionUnboundSeen = True; \"\")")
      .unwrap();
    woxi::interpret("captionUnit[x_String] := \"[\" <> x <> \"]\"").unwrap();
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let state = ManipulateState::from_expr(&expr).expect("state should build");
    assert_eq!(state.error, None);
    assert_eq!(
      woxi::interpret("ValueQ[captionUnboundSeen]").unwrap(),
      "False",
      "the caption was evaluated before its control was bound"
    );
  }

  /// A built-in-only caption `Dynamic[Last[minmax]]` over a hidden variable
  /// a `Module` body sets was evaluated while the widget was being searched
  /// for helper-made `LocatorPane`s — before the body had run — printing
  /// `Last::normal` for the still unbound variable.
  #[test]
  fn builtin_dynamic_caption_waits_for_the_body() {
    woxi::interpret("captionHeads = {}").unwrap();
    let code = r#"Manipulate[
      Module[{w = x}, captionMinMax = {w, w + 1}; w],
      Column[{Style[Row[{"side ", Dynamic[x],
          Button["-", x = x - 1, Appearance -> "Palette"],
          Control@{{x, 1, ""}, 0, 5, 1}}], 14],
        Style[Row[{"range: ",
          Dynamic[If[(AppendTo[captionHeads, Head[captionMinMax]]; True),
            Last[captionMinMax]]],
          Control[{{flag, False, "flag"}, {False, True}}]}], 14]},
        Alignment -> Center],
      {flag, False, ControlType -> None},
      {captionMinMax, {20, 28}, ControlType -> None}
    ]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let state = ManipulateState::from_expr(&expr).expect("state should build");
    assert_eq!(state.error, None);
    assert_eq!(
      woxi::interpret("DeleteDuplicates[captionHeads]").unwrap(),
      "{List}",
      "the caption was evaluated before the body set its variable"
    );
  }

  /// A caption `Dynamic[msg]` whose variable holds `Dynamic[a] Dynamic[b]`
  /// (text times a formatted number) shows both pieces, typeset side by
  /// side, instead of the raw `Dynamic[…]*Dynamic[…]` source.
  #[test]
  fn dynamic_caption_releases_dynamics_nested_in_a_variable() {
    let code = r#"Manipulate[
      (msg = Dynamic[lead[k]] Dynamic[PaddedForm[Round[k/3., 0.01], 4]];
        Graphics[Disk[]]),
      Row[{"= ", Style[Dynamic[msg], 14]}],
      {{k, 1}, 1, 3, 1}
    ]"#;
    woxi::interpret("lead[n_] := \"v \"").unwrap();
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let state = ManipulateState::from_expr(&expr).expect("state should build");
    assert_eq!(state.error, None);
    let text = format!("{:?}", state.display_trees);
    assert!(!text.contains("Dynamic["), "unreleased Dynamic: {text}");
    assert!(text.contains("v "), "missing text part: {text}");
    assert!(text.contains("0.33"), "missing number part: {text}");
  }

  #[test]
  fn demonstration_polyhedrondata_overlay_checkboxes_toggle_live() {
    let code = r#"Manipulate[
      Module[{base, star, spikes},
        base = GraphicsComplex[PolyhedronData["Icosahedron", "VertexCoordinates"],
          Polygon[PolyhedronData["Icosahedron", "FaceIndices"]]];
        star = GraphicsComplex[
          PolyhedronData["GreatStellatedDodecahedron", "VertexCoordinates"],
          Line[PolyhedronData["GreatStellatedDodecahedron", "EdgeIndices"]]];
        spikes = GraphicsComplex[
          PolyhedronData["MathematicaSpikey", "VertexCoordinates"],
          Line[PolyhedronData["MathematicaSpikey", "EdgeIndices"]]];
        Graphics3D[{base, If[showStar, Scale[star, grow], {}],
          If[showSpikes, Scale[spikes, shrink], {}]}, Boxed -> False]
      ],
      Grid[{
        {Control@{{grow, 1.2, "star size"}, 1, 2, 0.1, ImageSize -> Small},
         Control@{{shrink, 0.3, "spike size"}, 0.1, 0.5, 0.05,
           ImageSize -> Small}},
        {Row[{
           Control@{{showStar, False, "show star"}, {True, False}},
           Spacer[10],
           Control@{{showSpikes, False, "show spikes"}, {True, False}}
         }]}
      }],
      TrackedSymbols :> {grow, shrink, showStar, showSpikes}
    ]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let mut state =
      ManipulateState::from_expr(&expr).expect("state should build");

    assert_eq!(
      state.error, None,
      "body must evaluate cleanly at its default (both checkboxes off) \
       control values: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the base icosahedron must render"
    );

    let show_star_idx = state
      .controls
      .iter()
      .position(|c| c.name() == "showStar")
      .expect("showStar checkbox control");
    let show_spikes_idx = state
      .controls
      .iter()
      .position(|c| c.name() == "showSpikes")
      .expect("showSpikes checkbox control");

    assert!(state.select_discrete(show_star_idx, "True"));
    state.apply_tracking(show_star_idx);
    state.reevaluate();
    assert_eq!(
      state.error, None,
      "toggling on the great stellated dodecahedron overlay must not error: \
       {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the great stellated dodecahedron overlay must render"
    );

    assert!(state.select_discrete(show_spikes_idx, "True"));
    state.apply_tracking(show_spikes_idx);
    state.reevaluate();
    assert_eq!(
      state.error, None,
      "toggling on the Mathematica Spikey overlay must not error: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the Mathematica Spikey overlay must render"
    );

    let slider_names: Vec<&str> = state
      .controls
      .iter()
      .filter_map(|c| match c {
        ControlState::Continuous { name, .. } => Some(name.as_str()),
        _ => None,
      })
      .collect();
    assert_eq!(slider_names, ["grow", "shrink"]);
  }

  /// Checked a randomly-sampled Wolfram Demonstrations Project notebook
  /// ("Properties of Chemical Elements") whose `PopupMenu` control builds
  /// its choice list with `(If[# =!= Delimiter, # -> ElementData[1, #,
  /// "Description"], #]&) /@ {…}` — a `Map` over a list of property names
  /// interspersed with bare `Delimiter`s that group them into sections,
  /// each non-`Delimiter` entry turned into a `value -> label` rule via
  /// `ElementData`'s little-used third ("annotation") argument.
  /// Independently written, not copied from the Demonstration: different
  /// properties, a different default choice, and a differently laid out
  /// body.
  ///
  /// Regression coverage for two gaps that Demonstration's control panel
  /// hit and nothing else exercised: `ElementData[element, property,
  /// "Description"]` (previously always unevaluated — only the two- and
  /// one-argument forms were implemented, so every label rendered as the
  /// literal, unevaluated `ElementData[…]` call) and a bare `Delimiter`
  /// inside a choice list becoming a fourth, selectable "Delimiter" entry
  /// instead of the section separator it draws as in wolframscript (see
  /// `discrete_choice_columns` in `src/functions/graphics.rs`).
  #[test]
  fn popup_menu_choices_built_from_a_map_with_delimiters_and_element_descriptions()
   {
    let expr = woxi::interpret_to_expr(
      "Manipulate[
        ListPlot[Table[{z, ElementData[z, prop]}, {z, 1, 10}],
          PlotLabel -> ElementData[1, prop, \"Description\"]],
        {{prop, \"AtomicNumber\", \"property\"},
         (If[# =!= Delimiter, # -> ElementData[1, #, \"Description\"], #]&) /@
           {\"AtomicNumber\", \"AtomicRadius\", Delimiter, \"Density\"}}
      ]",
    )
    .expect("parse Manipulate expr");
    let state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");

    assert_eq!(
      state.error, None,
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the default (AtomicNumber) plot must draw"
    );

    let prop_ctrl = state
      .controls
      .iter()
      .find(|c| c.name() == "prop")
      .expect("prop control");
    match prop_ctrl {
      ControlState::Discrete {
        values,
        value_labels,
        ..
      } => {
        assert_eq!(
          values.len(),
          3,
          "the bare Delimiter must not become a fourth, selectable choice: {values:?}"
        );
        assert_eq!(
          value_labels,
          &vec![
            "atomic number".to_string(),
            "atomic radius".to_string(),
            // wolframscript's own description of the legacy `"Density"`
            // property (see `PROPERTY_DESCRIPTIONS` in
            // `src/functions/element_data.rs`).
            "mass density".to_string(),
          ]
        );
      }
      other => panic!("expected a Discrete PopupMenu control, got {other:?}"),
    }
  }

  /// A Manipulate whose body applies a plot head chosen by a `PopupMenu`
  /// (`kind[field @@ {x, y}, …]`) to a field picked from a menu of pure
  /// functions labelled with `TraditionalForm` rules; the choice list also
  /// carries a `FieldSize` option. `VectorDensityPlot` used to be missing,
  /// so selecting it left the body unevaluated.
  #[test]
  fn popup_menu_of_plot_heads_and_pure_function_fields_draws_vector_density() {
    let expr = woxi::interpret_to_expr(
      "Manipulate[
        kind[Apply[field, {x, y}], {x, -2, 2}, {y, -2, 2},
          ColorFunction -> palette, ImageSize -> {200, 200}],
        {{field, {#2, -#}&, \"field\"},
         {({#2, -#}&) -> TraditionalForm[{y, -x}],
          ({Sin[#], Cos[#2]}&) -> TraditionalForm[{Sin[x], Cos[y]}]},
         PopupMenu, FieldSize -> {Automatic, 2.5}},
        {{kind, VectorDensityPlot, \"plot type\"},
         {VectorPlot -> \"vector\", VectorDensityPlot -> \"vector density\"}},
        {{palette, \"SunsetColors\", \"palette\"},
         {\"SunsetColors\", \"LakeColors\"}}
      ]",
    )
    .expect("parse Manipulate expr");
    let state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    assert_eq!(state.error, None, "body must evaluate: {:?}", state.error);
    assert!(
      state.graphics_handle.is_some(),
      "the default VectorDensityPlot must draw"
    );
    assert_eq!(state.controls.len(), 3);
  }

  /// A Manipulate output saved by the FrontEnd names its controls with the
  /// `DynamicModule` `$$` suffix and no explicit label ("Ecosystem
  /// Dynamics"); the slider must read `amp`, not `amp$$`, while the variable
  /// keeps its real name.
  #[test]
  fn unlabeled_control_label_drops_dynamic_module_suffix() {
    let code = "Manipulate[Plot[amp$$ Sin[x], {x, 0, 6}], \
      {{amp$$, 1}, 0, 2, 0.1}, {{k$$, 2, \"freq\"}, 1, 4, 1}]";
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let state = ManipulateState::from_expr(&expr).expect("widget should build");
    let dump = |name: &str| {
      let c = state.controls.iter().find(|c| c.name() == name).unwrap();
      format!("{c:?}")
    };
    assert!(
      dump("amp$$").contains("label: \"amp\""),
      "{}",
      dump("amp$$")
    );
    assert!(dump("k$$").contains("label: \"freq\""), "{}", dump("k$$"));
  }

  /// A slider caption built with `StringJoin`/`ToString` shows the text it
  /// produces, not the unevaluated call.
  #[test]
  fn control_label_evaluates_string_join() {
    let code = r#"Manipulate[Plot[a x, {x, 0, 1}],
      {{a, 0, "pairs (0 to " <> ToString[z] <> ")"}, 0, z, 2},
      Initialization :> (z = 5)]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let state = ManipulateState::from_expr(&expr).expect("widget should build");
    let dump = format!("{:?}", state.controls);
    assert!(dump.contains("label: \"pairs (0 to 5)\""), "{dump}");
    assert!(!dump.contains("StringJoin"), "{dump}");
  }

  /// A slider caption that indexes a notebook list (`names[[2]]`) shows the
  /// element, not the source of the `Part` call.
  #[test]
  fn control_label_indexing_a_notebook_list_shows_the_element() {
    let code = r#"Manipulate[Plot[a x + b, {x, 0, 1}],
      {{a, 1, geneNames[[1]]}, 0, 2, 0.5},
      {{b, 1, geneNames[[2]]}, 0, 2, 0.5},
      Initialization :> (geneNames = {"AOAH", "CBLL1"})]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let state = ManipulateState::from_expr(&expr).expect("widget should build");
    let dump = format!("{:?}", state.controls);
    assert!(dump.contains("label: \"AOAH\""), "{dump}");
    assert!(dump.contains("label: \"CBLL1\""), "{dump}");
    assert!(!dump.contains("geneNames"), "{dump}");
  }

  /// A 2D slider captioned with a `GraphicsRow` of text and a small picture
  /// shows the text; the picture and the layout head's source drop out.
  #[test]
  fn control_label_graphics_row_keeps_text_only() {
    let code = r#"Manipulate[v, {{v, {1, 2}, GraphicsRow[{Style["vector (" <> ToString[v[[1]]] <> ")", 10], Framed[Graphics[Circle[]]]}]}, {-5, -5}, {5, 5}}]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let state = ManipulateState::from_expr(&expr).expect("widget should build");
    let dump = format!("{:?}", state.controls[0]);
    assert!(!dump.contains("GraphicsRow"), "{dump}");
    assert!(!dump.contains("Graphics"), "{dump}");
    assert!(dump.contains("vector ("), "{dump}");
  }

  /// A heading `Row[{"parameter ", Style[a, Italic], Spacer[10], }]` ends in
  /// a trailing comma, i.e. a `Null` item; it is drawn as nothing, so the
  /// heading reads "parameter a" rather than "parameter a Null".
  #[test]
  fn heading_row_with_trailing_comma_drops_null() {
    let code = r#"Manipulate[Plot[a x, {x, 0, 1}],
      Row[{"parameter ", Style["a", Italic], Spacer[10], }],
      {a, 1, 2}]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let state = ManipulateState::from_expr(&expr).expect("widget should build");
    let dump = format!("{:?}", state.controls);
    assert!(dump.contains("parameter"), "{dump}");
    assert!(!dump.contains("Null"), "{dump}");
  }

  /// A slider label whose subscript holds a negative index
  /// (`\!\(\*SubscriptBox[\(c\), \(2, \(-2\)\)]\)`) nests a second
  /// `\(…\)` group inside the first; it must typeset as a subscript like
  /// the flat `\(0, 0\)` form, not leak the raw `SubscriptBox[c, 2, -2]`.
  #[test]
  fn control_label_typesets_subscript_with_negative_index() {
    let code = r#"Manipulate[c5, {{c5, 0, "\!\(\*SubscriptBox[\(c\), \(2, \(-2\)\)]\)"}, 0, 1, 0.01}]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let state = ManipulateState::from_expr(&expr).expect("widget should build");
    let dump = format!("{:?}", state.controls[0]);
    assert!(!dump.contains("SubscriptBox"), "{dump}");
    assert!(dump.contains("-2"), "{dump}");
  }

  /// A subscript saved as nested groups with a padding space
  /// (`\(\(12\)\( \)\)`, how the FrontEnd keeps `σ12 `) typesets as a
  /// real subscript, the same as the unpadded `\(12\)`, not as `σ_12`.
  #[test]
  fn control_label_typesets_padded_subscript_index() {
    let code = r#"Manipulate[s, {{s, 0, "\!\(\*SubscriptBox[\(σ\), \(\(12\)\( \)\)]\)= "}, 0, 1, 0.1}]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let state = ManipulateState::from_expr(&expr).expect("widget should build");
    let dump = format!("{:?}", state.controls[0]);
    assert!(dump.contains("label: \"σ₁₂= \""), "{dump}");
  }

  /// A caption can apply a compound head to an argument —
  /// `TraditionalForm[Subscript[f, a][x]]` — which must typeset as the head's
  /// script followed by `(x)`, not leak the raw `Subscript[f, a][x]` source.
  #[test]
  fn control_label_typesets_compound_head_call() {
    let code = r#"Manipulate[Plot[a x, {x, 0, 1}],
      {{a, 1, Row[{"show ", TraditionalForm[Subscript[g, 1][x]], " here"}]},
        {True, False}}]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let state = ManipulateState::from_expr(&expr).expect("widget should build");
    let c = state.controls.iter().find(|c| c.name() == "a").unwrap();
    let dump = format!("{c:?}");
    assert!(dump.contains("label: \"show g₁(x) here\""), "{dump}");
  }

  /// A caption that writes a sample mean as `Subscript[OverBar[X], 1]` must
  /// typeset the bar over the letter, not leak the raw `OverBar[X]` source.
  #[test]
  fn control_label_typesets_overbar() {
    let code = r#"Manipulate[x,
      {{x, 1, Row[{"mean, ", Subscript[OverBar[Style["X", Italic]], 1]}]}, 0, 2}]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let state = ManipulateState::from_expr(&expr).expect("widget should build");
    let dump = format!("{:?}", state.controls);
    assert!(!dump.contains("OverBar"), "{dump}");
    assert!(dump.contains("label: \"mean, X\\u{304}₁\""), "{dump}");
  }

  /// A control caption written `Dynamic[If[mode == 0, x₂, z₀]]` follows the
  /// mode control: it is retypeset on every change rather than frozen to
  /// the source text of the `If`.
  #[test]
  fn dynamic_control_caption_follows_another_control() {
    let code = r#"Manipulate[Plot[a x, {x, 0, 1}],
      {{mode, 0, ""}, {0 -> "one", 1 -> "two"}},
      {{a, 1, Dynamic[If[mode == 0, Subscript[Style["x", Italic], "2"],
        Subscript[Style["z", Italic], "0"]]]}, 0, 2}]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let mut state =
      ManipulateState::from_expr(&expr).expect("widget should build");
    let label = |state: &ManipulateState| {
      let c = state.controls.iter().find(|c| c.name() == "a").unwrap();
      format!("{c:?}")
    };
    assert!(label(&state).contains("label: \"x₂\""), "{}", label(&state));
    let idx = state
      .controls
      .iter()
      .position(|c| c.name() == "mode")
      .unwrap();
    assert!(state.select_discrete(idx, "two"));
    state.reevaluate();
    assert!(label(&state).contains("label: \"z₀\""), "{}", label(&state));
  }

  /// A control written `{{flag, flag, "label"}, {False, True}}` names its
  /// own unassigned variable as the initial value. That is no value at
  /// all: the control starts on its first choice, so the body sees `False`
  /// rather than the bare symbol `flag` (which `Which`/`If` cannot decide).
  #[test]
  fn self_named_initial_value_starts_on_the_first_choice() {
    let code = r#"Manipulate[
      Graphics[{If[flag, Disk[], Circle[]]}],
      {{r, 1, "radius"}, 0, 2},
      {{flag, flag, "toggle"}, {False, True}}
    ]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let state = ManipulateState::from_expr(&expr).expect("state should build");
    let flag = state
      .controls
      .iter()
      .find(|c| c.name() == "flag")
      .expect("flag control");
    assert_eq!(flag.current_code(), "False");
    assert_eq!(state.error, None);
    assert!(state.graphics_handle.is_some());
  }

  /// A quiz Demonstration builds its answer choices from an improper
  /// integral whose upper limit is a Module local assigned `Infinity`; the
  /// stored `Infinity` must stay exact so the choices are exact too.
  #[test]
  fn body_popup_choices_from_improper_integral_stay_exact() {
    let code = r#"Manipulate[
      Module[{par, b, ans, choices},
        par = {1, Infinity}; b = par[[2]];
        ans = Integrate[x^(-2), {x, par[[1]], b}];
        choices = {ans, ans + 1, ans - 1/2};
        Column[{"Calculate", PopupMenu[Dynamic[an], choices]}]],
      {an, -2, ControlType -> None}]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let state = ManipulateState::from_expr(&expr).expect("state should build");
    let an = state
      .controls
      .iter()
      .find(|c| c.name() == "an")
      .expect("popup control");
    match an {
      ControlState::Discrete { value_labels, .. } => assert_eq!(
        value_labels,
        &vec!["1".to_string(), "2".to_string(), "1/2".to_string()]
      ),
      other => panic!("expected a Discrete popup, got {other:?}"),
    }
  }

  /// `Control[{reset, None}]` states neither a domain nor an initial value,
  /// only the bare control-type marker; the variable starts at `0`, so a
  /// body seeding the random generator from it renders.
  #[test]
  fn bare_none_control_starts_at_zero() {
    let code = r#"Manipulate[
      SeedRandom[reset]; Row[{reset, RandomInteger[{1, 5}] > 0}],
      {reset, None}]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let state = ManipulateState::from_expr(&expr).expect("state should build");
    assert_eq!(state.error, None);
    assert!(
      state.state.iter().any(|(n, v)| n == "reset" && v == "0"),
      "state: {:?}",
      state.state
    );
  }

  /// A two-player board game laid out as a `GraphicsGrid` of
  /// A `ClickPane` as the whole Manipulate body (a board painted by
  /// clicking it): the picture must offer a click region whose coordinates
  /// follow the graphic's `PlotRange`, and a click must run the handler on
  /// the clicked position and redraw.
  #[test]
  fn click_pane_body_runs_its_handler_on_the_clicked_position() {
    let code = r#"Manipulate[
      ClickPane[
        Graphics[{Table[{If[tiles[[i]] == 1, Red, Blue], Rectangle[{i - 1, 0}, {i, 1}]}, {i, 4}]},
          PlotRange -> {{0, 4}, {0, 1}}, ImageSize -> {400, 100}],
        (tiles[[Floor[#[[1]]] + 1]] = paint) &],
      {{paint, 2}, {1, 2}, ControlType -> SetterBar},
      {{tiles, {1, 1, 1, 1}}, ControlType -> None}
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("should parse");
    let mut state =
      ManipulateState::from_expr(&expr).expect("should build a widget");
    assert!(state.error.is_none(), "body failed: {:?}", state.error);
    assert_eq!(state.click_regions.len(), 1);
    let region = state.click_regions[0].clone();
    assert_eq!((region.width, region.height), (400.0, 100.0));
    // The third tile spans x in [200, 300) pixels.
    state.pointer = (250.0, 50.0);
    state.apply_pane_click();
    assert!(state.error.is_none(), "click failed: {:?}", state.error);
    let tiles = state
      .state
      .iter()
      .find(|(n, _)| n == "tiles")
      .map(|(_, v)| v.clone())
      .unwrap_or_default();
    assert_eq!(tiles, "{1, 1, 2, 1}");
    // A click's y axis points up: the top edge is the range's maximum.
    assert!(region.click_action(0.0, 0.0).contains(", 1}]"));
    assert!(region.click_action(0.0, 100.0).contains(", 0}]"));
  }

  /// `Button[Graphics[…], action]` cells (the shape of a tic-tac-toe style
  /// Demonstration): every enabled cell must be a pressable region of the
  /// picture, pressing one runs that cell's own action on the widget state,
  /// and a cell whose `Enabled` condition turns false stops being pressable.
  #[test]
  fn event_handler_shapes_in_graphics_are_clickable_regions() {
    let code = r#"Manipulate[
      Graphics[
        Table[
          With[{n = n},
            EventHandler[
              Dynamic[{shade[[n]], Rectangle[{n, 0}, {n + 1, 1}]}],
              {"MouseClicked" :> (shade[[n]] = Red)}]],
          {n, 4}],
        ImageSize -> 200],
      {{shade, Table[White, {4}]}, ControlType -> None}
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("should parse");
    let mut state =
      ManipulateState::from_expr(&expr).expect("should build a widget");
    assert!(state.error.is_none(), "body failed: {:?}", state.error);
    assert_eq!(state.button_regions.len(), 4, "one region per shape");
    let mut xs: Vec<i32> =
      state.button_regions.iter().map(|r| r.x as i32).collect();
    xs.sort();
    xs.dedup();
    assert_eq!(xs.len(), 4, "shapes must not share a region: {xs:?}");

    // Clicking a shape runs its own handler.
    let region = state
      .button_regions
      .iter()
      .find(|r| r.action.contains("shade[[3]]"))
      .expect("handler of the third shape")
      .clone();
    state.apply_button_action(&region.action);
    let shade = &state
      .state
      .iter()
      .find(|(name, _)| name == "shade")
      .expect("shade state")
      .1;
    assert!(shade.contains("RGBColor[1, 0, 0]"), "{shade}");
  }

  #[test]
  fn graphics_grid_button_cells_are_pressable_regions() {
    let code = r#"Manipulate[
      Module[{board},
        board = GraphicsGrid[
          Table[
            With[{i = i, j = j},
              Button[
                Graphics[{Dynamic@cols[[Mod[cells[[i, j]], 3, 1]]], Disk[{0, 0}, 1]}],
                cells[[i, j]] = Mod[cells[[i, j]] + turn, 3, 1]; turn = 3 - turn; used[[i, j]] = 1,
                Background -> LightGray,
                Enabled -> Dynamic[used[[i, j]] == 0]]],
            {i, 3}, {j, 3}],
          Spacings -> 0, Frame -> True, ImageSize -> {150, 150}];
        Row[{board}]],
      {{cells, Table[1, {3}, {3}]}, ControlType -> None},
      {{used, Table[0, {3}, {3}]}, ControlType -> None},
      {{turn, 1}, ControlType -> None},
      Initialization :> (cols = {White, Red, Blue})
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("should parse");
    let mut state =
      ManipulateState::from_expr(&expr).expect("should build a widget");
    assert!(state.error.is_none(), "body failed: {:?}", state.error);
    assert_eq!(state.button_regions.len(), 9, "one region per cell");
    // Regions are distinct pixel rectangles inside the picture.
    for region in &state.button_regions {
      assert!(region.width > 1.0 && region.height > 1.0, "{region:?}");
    }
    let mut origins: Vec<(i32, i32)> = state
      .button_regions
      .iter()
      .map(|r| (r.x as i32, r.y as i32))
      .collect();
    origins.sort();
    origins.dedup();
    assert_eq!(origins.len(), 9, "cells must not overlap: {origins:?}");

    // Pressing the first cell claims it for player 1 and passes the turn.
    let action = state.button_regions[0].action.clone();
    assert!(action.contains("cells[[1,1]]"), "{action}");
    state.apply_button_action(&action);
    assert!(state.error.is_none(), "button failed: {:?}", state.error);
    let value = |name: &str| {
      state
        .state
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, v)| v.clone())
        .unwrap_or_default()
    };
    assert_eq!(value("turn"), "2");
    assert!(
      value("cells").starts_with("{{2, 1, 1}"),
      "{}",
      value("cells")
    );
    // That cell is now disabled, so it no longer offers a region.
    assert_eq!(state.button_regions.len(), 8);
    assert!(
      state
        .button_regions
        .iter()
        .all(|r| !r.action.contains("[[1,1]]")),
      "the used cell must not stay pressable"
    );
  }

  /// As part of a scheduled QA routine, Woxi Studio was tested against a
  /// randomly sampled Wolfram Demonstration notebook (a hydrogen-atom
  /// quantum-number explorer) whose second setter row lists
  /// `Range[0, n - 1]` for a quantum number that depends on the first row.
  /// This is a self-authored, construct-equivalent example (a generic
  /// pair of dependent counters, not the Demonstration's names or code).
  ///
  /// Coverage: a plain (non-`Dynamic`) dependent choice list is rebuilt
  /// when the row it depends on changes.
  #[test]
  fn plain_range_choices_follow_the_row_they_depend_on() {
    let expr = woxi::interpret_to_expr(
      "Manipulate[
        If[b > a - 1, b = a - 1];
        {a, b},
        {{a, 1, Style[\"a\", Italic]}, Range[4], Setter},
        {{b, 0, Style[\"b\", Italic]}, Range[0, a - 1], Setter}
      ]",
    )
    .expect("parse Manipulate expr");
    let mut state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    let labels = |s: &ManipulateState| -> Vec<String> {
      match s.controls.iter().find(|c| c.name() == "b") {
        Some(ControlState::Discrete { value_labels, .. }) => {
          value_labels.clone()
        }
        _ => vec![],
      }
    };
    assert_eq!(labels(&state), vec!["0"]);
    let a_idx = state.controls.iter().position(|c| c.name() == "a").unwrap();
    assert!(state.select_discrete(a_idx, "4"));
    state.reevaluate();
    assert_eq!(labels(&state), vec!["0", "1", "2", "3"]);
  }
  /// A slider whose upper bound is `Dynamic[var]` of a hidden state variable
  /// that is still 0 when the spec is parsed (`{{n, 1}, 1, Dynamic[total],
  /// 1}` next to `{{total, 0}, ControlType -> None}`) is not a reversed
  /// slider: it keeps its static lower bound and opens up to the variable's
  /// live value once the body has set it.
  #[test]
  fn slider_with_unsettled_dynamic_max_keeps_static_min() {
    let expr = woxi::interpret_to_expr(
      "Manipulate[(total = 5; n),
        {{n, 1}, 1, Dynamic[total], 1},
        {{total, 0}, ControlType -> None}
      ]",
    )
    .expect("parse Manipulate expr");
    let state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    let bounds = state.controls.iter().find_map(|c| match c {
      ControlState::Continuous {
        name,
        min,
        max,
        current,
        ..
      } if name == "n" => Some((*min, *max, *current)),
      _ => None,
    });
    assert_eq!(bounds, Some((1.0, 5.0, 1.0)));
  }
  /// A slider bound that wraps `Dynamic` *inside* a larger expression
  /// (`k*If[Dynamic[big], 1, 100]`) still makes a control, and follows the
  /// other control when that one changes.
  #[test]
  fn slider_bound_with_nested_dynamic_follows_other_control() {
    let expr = woxi::interpret_to_expr(
      "Manipulate[{big, k},
        {{big, False}, {False, True}},
        {{k, 0}, 0, 19*If[Dynamic[big], 1, 100], If[Dynamic[big], 1, 100]}
      ]",
    )
    .expect("parse Manipulate expr");
    let mut state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    let bounds = |s: &ManipulateState| {
      s.controls.iter().find_map(|c| match c {
        ControlState::Continuous {
          name,
          min,
          max,
          step,
          ..
        } if name == "k" => Some((*min, *max, *step)),
        _ => None,
      })
    };
    assert_eq!(bounds(&state), Some((0.0, 1900.0, 100.0)));
    let big_idx = state
      .controls
      .iter()
      .position(|c| c.name() == "big")
      .unwrap();
    assert!(state.select_discrete(big_idx, "True"));
    state.reevaluate();
    assert_eq!(bounds(&state), Some((0.0, 19.0, 1.0)));
  }

  /// A message-typing Demonstration: a `Column` of a reset `Button`, a
  /// `GraphicsGrid` of character `Button`s, `Dynamic` read-outs of the typed
  /// string and a conditional `Plot`, driven by a hidden list variable.
  #[test]
  fn typed_message_modulation_widget_renders() {
    let code = r#"Manipulate[
      Text@Column[{
        Button["reset", msg = {}, ImageSize -> {200, 20}],
        GraphicsGrid[
          Partition[
            Map[Button[#, AppendTo[msg, #]] &, CharacterRange["a", "p"]], 4],
          Frame -> True, Spacings -> 2, ImageSize -> {300, 100}],
        Dynamic["Typed: " <> StringJoin[msg]],
        Dynamic[Grid[{msg, Flatten@IntegerString[ToCharacterCode[msg], 2, 7]},
          Frame -> All]],
        Dynamic@If[Length[msg] > 0,
          Plot[
            With[{k = Ceiling[t]},
              If[kind == "AM",
                (Flatten[IntegerDigits[ToCharacterCode[msg], 2, 7]][[k]] + .5)*
                  Sin[2 Pi*freq*t],
                Sin[(Flatten[IntegerDigits[ToCharacterCode[msg], 2, 7]][[k]] + .5)*
                  2 Pi*freq*t]]],
            {t, 0, 7*Length[msg]}, ImageSize -> {300, 100}],
          ""]},
        Frame -> True, Alignment -> Center],
      Row[{Control[{{freq, 1}, 0, 10, ImageSize -> Tiny}], Spacer[20],
        Control[{kind, {"AM", "FM"}}]}],
      Control[{{msg, {"A"}}, ControlType -> None}],
      ContentSize -> {400, 300}]"#;
    let expr = woxi::interpret_to_expr(code).expect("should parse");
    let state = ManipulateState::from_expr(&expr).expect("should build");
    assert_eq!(state.error, None);
    assert!(state.graphics_handle.is_some(), "no graphics output");
    assert_eq!(state.button_regions.len(), 16, "one region per key");
    let mut state = state;
    let action = state.button_regions[2].action.clone();
    state.apply_button_action(&action);
    assert_eq!(state.error, None);
    let msg = state
      .state
      .iter()
      .find(|(n, _)| n == "msg")
      .map(|(_, v)| v.clone());
    assert_eq!(msg.as_deref(), Some("{\"A\", \"c\"}"));
  }
}

#[cfg(test)]
mod scoped_button_tests {
  use super::*;

  /// Regression: a `Button` under `Module`/`With`/`If` in a Manipulate body
  /// was lifted out into the persistent display list, where the local
  /// variables it uses (`pair`, `n`) were unbound and every `If` branch was
  /// shown at once, leaving `Nothing` behind in the body. It must stay in
  /// the body, drawn as plates (also when it sits in a `Row` built by
  /// `Map` inside a `Grid`).
  #[test]
  fn buttons_under_scoping_and_conditionals_stay_in_the_body() {
    let code = r#"Manipulate[
      Module[{half},
        half[k_] := Quotient[k, 2];
        With[{n = half[m]},
          Grid[{
            {"header", SpanFromLeft},
            If[n > 2, {Button["big", Print[n]], SpanFromLeft},
              {Button["small", Print[n]], SpanFromLeft}],
            {Row[Map[Button[Grid[{{#}}], Print[#, n]] &, Range[n]]],
              SpanFromLeft}}]]],
      {{m, 8}, 0, 12, 1}]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let state = ManipulateState::from_expr(&expr).expect("build widget");
    assert_eq!(state.error, None);
    assert!(
      state.displays.is_empty(),
      "buttons must not be lifted out of their scope: {:?}",
      state.displays
    );
    assert!(state.graphics_handle.is_some(), "the body should render");
  }
}

#[cfg(test)]
mod textcell_tests {
  use super::*;

  /// A `TextCell[Row[{Style[…], …}]]` argument is a static caption line
  /// between the controls (a Demonstration's "Drag the locators" hint), not
  /// a malformed variable specification: it becomes a heading carrying the
  /// caption text and the real controls after it still build.
  #[test]
  fn text_cell_caption_is_a_heading_not_a_variable_spec() {
    let code = r#"Manipulate[
      Plot[a*x, {x, 0, 1}],
      TextCell[Row[{Style["Drag the slider ", Bold], Style["now", Red]}]],
      {{a, 1}, 0, 2}
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let state = ManipulateState::from_expr(&expr).expect("build widget");
    assert_eq!(state.error, None);
    let names: Vec<&str> = state.controls.iter().map(|c| c.name()).collect();
    assert_eq!(names, ["", "a"]);
    assert!(state.graphics_handle.is_some(), "the plot should render");
    assert!(!state.controls[0].binds_variable());
  }

  /// As part of a scheduled QA routine, Woxi Studio was tested against a
  /// randomly sampled Wolfram Demonstration notebook (a container-pouring
  /// puzzle). This is a self-authored, construct-equivalent example: a
  /// graphic whose `Inset[Button[…]]` actions swap list-of-pairs state with
  /// a tuple `Part` assignment and bump a hidden counter, with an empty
  /// argument inside the primitive list.
  ///
  /// Coverage: tuple `Part` assignment from a user helper, `++` on a hidden
  /// control, and `Inset`ed buttons as pressable regions.
  #[test]
  fn inset_buttons_with_tuple_part_assignment_update_hidden_state() {
    let code = r#"Manipulate[
      Deploy@Framed@Graphics[{
        Rectangle[{0, 0}, {3, s[[1, 1]]}],,
        Rectangle[{6, 0}, {9, s[[2, 1]]}],
        Inset[Button[" > ", {s[[1]], s[[2]]} = mv[{s[[1]], s[[2]]}, 5]; n++], {4.5, 2.5}],
        Inset[Button["RESET", s = {{1, 0}, {2, 0}}; n = 0], {0, -4}],
        Text[ToString[n] <> " steps", {10, -4}]},
        PlotRange -> {{-1, 12}, {-5, 10}}, ImageSize -> 300],
      {{s, {{3, 0}, {1, 0}}}, {-1, -5}, {16, 10}, None},
      {{n, 0}, None},
      SaveDefinitions -> True,
      Initialization :> (mv[{{a_, ta_}, {b_, tb_}}, cb_] :=
        If[a >= cb - b, {{a - cb + b, ta}, {cb, tb}}, {{0, ta}, {a + b, tb}}])
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("should parse");
    let mut state =
      ManipulateState::from_expr(&expr).expect("should build a widget");
    assert!(state.error.is_none(), "body failed: {:?}", state.error);
    assert!(state.graphics_handle.is_some(), "the graphic should render");
    assert_eq!(state.button_regions.len(), 2, "both Inset buttons");
    let value = |state: &ManipulateState, name: &str| {
      state
        .state
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, v)| v.clone())
        .unwrap_or_default()
    };
    let action = state.button_regions[0].action.clone();
    state.apply_button_action(&action);
    assert!(state.error.is_none(), "button failed: {:?}", state.error);
    assert_eq!(value(&state, "n"), "1");
    assert_eq!(value(&state, "s"), "{{0, 0}, {4, 0}}");
    let reset = state.button_regions[1].action.clone();
    state.apply_button_action(&reset);
    assert_eq!(value(&state, "n"), "0");
    assert_eq!(value(&state, "s"), "{{1, 0}, {2, 0}}");
  }

  /// A `Row` carrying an option inside a `Module` body must survive the
  /// body-source round trip with its `Row` head, or its items print as a
  /// scrambled product of strings in a plot label.
  #[test]
  fn module_body_keeps_row_with_options_in_plot_label() {
    let code = r#"Manipulate[
      Module[{P = 1, opt1},
        opt1 = Sequence[Axes -> False,
          PlotLabel -> Style[Row[{"a", " = ", P, " bar", Spacer@50, Subscript[Style["f", Italic], "B"], Superscript[" P", If[z < 1, "s", ""]], NumberForm[z, {2, 2}]}, ImageMargins -> 1], 17]];
        Plot[x, {x, 0, 1}, Evaluate[opt1]]],
      {z, 0, 1}
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    assert!(
      state.body.contains("Row[{"),
      "body lost its Row: {}",
      state.body
    );
  }

  /// A rotating shaded solid driven by a bare colour spec
  /// (`{c, RGBColor[…], ImageSize -> Tiny}`), an opacity slider, an
  /// edge checkbox and a `Trigger` spinning it about the z axis — the
  /// Demonstrations idiom for a turntable view of a polyhedron.
  /// The colour spec names no domain, so it is the control's *initial
  /// value*: it must build a colour swatch, not get baked into the body as
  /// a constant (which silently dropped the control, leaving the panel one
  /// row short of what Wolfram draws).
  #[test]
  fn rotating_solid_with_bare_color_spec_builds_a_color_control() {
    let code = r#"Manipulate[
      Graphics3D[
        Rotate[{shade, Specularity[White, 10], Opacity[opacity],
          EdgeForm[If[edge, Thickness[0.001], None]],
          Polygon /@ faces}, spin, {0, 0, 1}],
        Lighting -> "Neutral", Boxed -> False, ImageSize -> {200, 180}],
      {shade, RGBColor[0.3, 0.4, 0.8], ImageSize -> Tiny},
      {{opacity, 0.5}, 0.2, 0.9, 0.1, ImageSize -> Tiny},
      Delimiter,
      {edge, {False, True}, ControlType -> Checkbox},
      Delimiter,
      {spin, 0, 2 Pi, Pi/90, ControlType -> Trigger},
      ControlPlacement -> Left,
      Initialization :> (faces = {
        {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
        {{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
        {{1, 0, 0}, {0, -1, 0}, {0, 0, 1}},
        {{-1, 0, 0}, {0, 1, 0}, {0, 0, -1}}})
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    assert_eq!(state.error, None);
    assert!(
      state.graphics_handle.is_some(),
      "the solid must render, got text {:?}",
      state.text_output
    );
    assert_eq!(state.control_placement, ControlPlacement::Left);
    match &state.controls[..] {
      [
        ControlState::Color {
          name: colour,
          current,
          ..
        },
        ControlState::Continuous { name: alpha, .. },
        ControlState::Divider,
        ControlState::Discrete { name: flag, .. },
        ControlState::Divider,
        ControlState::Trigger {
          name: angle,
          min,
          max,
          ..
        },
      ] => {
        assert_eq!(colour, "shade");
        assert_eq!(current, "RGBColor[0.3, 0.4, 0.8]");
        assert_eq!(alpha, "opacity");
        assert_eq!(flag, "edge");
        assert_eq!(angle, "spin");
        assert_eq!(*min, 0.0);
        assert!((*max - 2.0 * std::f64::consts::PI).abs() < 1e-12);
      }
      other => panic!("unexpected control panel: {other:?}"),
    }
    // Picking a new colour re-renders the body with it bound.
    let mut state = state;
    state.color_change(0, 0.75);
    assert_eq!(state.controls[0].current_code(), "Hue[0.75]");
    state.reevaluate();
    assert_eq!(state.error, None);
    assert!(state.graphics_handle.is_some());
  }

  /// Frame-history browser: a button appends to a list held in a hidden
  /// state variable, a "frame index" slider's upper bound is
  /// `Dynamic[Length[frames]]`, and a `ProgressIndicator` display follows a
  /// counter variable with a `Dynamic` upper limit. Generic, self-authored
  /// construct (not taken from any particular Demonstration).
  ///
  /// Regression: the `Dynamic[total]` limit inside the progress indicator
  /// stayed unevaluated, so the bar printed as `ProgressIndicator[…]`
  /// source text instead of a filled track.
  #[test]
  fn button_grows_dynamic_slider_bound_and_progress_indicator_follows() {
    let code = r#"Manipulate[
      Graphics[{Disk[{0, 0}, frames[[idx]]]}],
      Dynamic[ProgressIndicator[done, {0, Dynamic[total]}, ImageSize -> 160]],
      Button["append",
        frames = Append[frames, Last[frames] + 1];
        idx = Length[frames]; done = 3; total = 6;
        ImageSize -> Automatic, Method -> "Queued"],
      {{idx, 1, ""}, 1, Dynamic[Length[frames]], 1, Appearance -> "Labeled"},
      {{done, 0}, ControlType -> None},
      {{total, 10}, ControlType -> None},
      {{frames, {1}}, ControlType -> None}
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let mut state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    assert_eq!(state.error, None);
    let progress_svg =
      |state: &ManipulateState| match state.display_trees.first() {
        Some(DisplayNode::Static { svg, .. }) => svg.clone(),
        other => panic!("expected a static progress display, got {other:?}"),
      };
    let before = progress_svg(&state).expect("progress bar drawn as SVG");
    let action = state
      .controls
      .iter()
      .find_map(|c| match c {
        ControlState::Button { action, .. } => Some(action.clone()),
        _ => None,
      })
      .expect("button");
    state.apply_button_action(&action);
    state.apply_button_action(&action);
    // Done = 3 of total = 6: half of the track is filled, unlike the empty
    // bar shown before the button ran.
    let after = progress_svg(&state).expect("progress bar drawn as SVG");
    assert_ne!(before, after);
    assert!(after.contains("width=\"74.5\""), "{after}");
    let (min, max, current) = state
      .controls
      .iter()
      .find_map(|c| match c {
        ControlState::Continuous {
          name,
          min,
          max,
          current,
          ..
        } if name == "idx" => Some((*min, *max, *current)),
        _ => None,
      })
      .expect("idx slider");
    // Two appends: the Dynamic upper bound follows `Length[frames]`.
    assert_eq!((min, max, current), (1.0, 3.0, 3.0));
  }

  /// Checked a randomly-sampled Wolfram Demonstrations Project notebook
  /// (a lease calculator) whose `Pane`d body is a user-defined function
  /// returning `Style[Grid[{{label, StringForm["$``", NumberForm[x,
  /// {Infinity, 2}, DigitBlock -> 3]]}, …}, Alignment -> "Left"], "Label",
  /// 14]`, with the sign prepended inside the `StringForm` template.
  /// Independently written: a different (loan) domain, names and labels.
  #[test]
  fn pane_body_with_stringform_numberform_digit_block_grid() {
    let code = r#"Manipulate[
      Pane[loanSummary[price, rate, years], ImageSize -> {400, 80},
        Alignment -> {Left, Center}],
      {{price, 250000, "price ($)"}, 50000, 900000, 5000,
        Appearance -> "Labeled"},
      {{rate, 0.05, "rate"}, 0.01, 0.1, 0.005, Appearance -> "Labeled"},
      {{years, 30, "years"}, 5, 40, 1, Appearance -> "Labeled"},
      SaveDefinitions -> True,
      Initialization :> (loanSummary[p_, r_, n_] := Module[{m, tot},
        m = p r/12/(1 - (1 + r/12)^(-12 n));
        tot = m*12*n - 2*p;
        Style[Grid[{
          {"monthly", StringForm["$``",
            NumberForm[m, {Infinity, 2}, DigitBlock -> 3]]},
          {"net", StringForm[If[tot < 0, "-", ""] <> "$``",
            NumberForm[If[tot < 0, -1, 1] tot, {Infinity, 2},
              DigitBlock -> 3]]},
          {"years", n}}, Alignment -> "Left"], "Label", 14]])
    ]"#;
    let expr =
      woxi::interpret_to_expr(code).expect("Manipulate should parse and hold");
    let state = ManipulateState::from_expr(&expr).expect("state should build");

    assert_eq!(
      state.error, None,
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the Grid readout should render with the default control values"
    );
  }

  /// A random polygon whose vertices live in a hidden `ControlType -> None`
  /// list the body regenerates from a seed slider and drags through a
  /// `LocatorPane`: the list gets exactly one draggable control (the pane's),
  /// never a second one from its hidden declaration.
  #[test]
  fn hidden_point_list_dragged_through_locator_pane_has_one_locator() {
    let code = r#"Manipulate[
      SeedRandom[sd];
      If[sd == oldsd, {}, pts = RandomReal[{0, 1}, {Length[pts], 2}]; oldsd = sd];
      If[cnt == oldcnt, oldcnt = Length[pts]; cnt = Length[pts],
        pts = PadRight[pts, {cnt, 2}, RandomReal[]]; oldcnt = cnt];
      LocatorPane[Dynamic[pts],
        Dynamic[ListLinePlot[pts, AspectRatio -> 1, Axes -> False] /. Line -> Polygon],
        LocatorAutoCreate -> True],
      {{cnt, 6, "points"}, 3, 20, 1},
      {{sd, 0, "random seed"}, 0, 1000, 1},
      {{oldsd, 0}, ControlType -> None},
      {{oldcnt, 6}, ControlType -> None},
      {{pts, RandomReal[{0, 1}, {6, 2}]}, ControlType -> None}
    ]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let state =
      ManipulateState::from_expr(&expr).expect("build Manipulate widget");
    assert_eq!(state.error, None);
    let locators = state
      .controls
      .iter()
      .filter(|c| matches!(c, ControlState::Locator { .. }))
      .count();
    assert_eq!(locators, 1, "controls: {:?}", state.controls);
  }
}

/// With `WOXI_STUDIO_TRACE` set, print every piece of code the notebook
/// loader evaluates to stderr before running it, so a load that crashes or
/// hangs can be replayed as a plain script with `woxi run`.
pub fn trace_evaluation(code: &str) {
  if std::env::var_os("WOXI_STUDIO_TRACE").is_some() {
    eprintln!("{};\n", code.trim_end().trim_end_matches(';'));
  }
}

#[cfg(test)]
mod mapped_layout_tests {
  use super::*;

  /// As part of a scheduled QA routine, Woxi Studio was tested against a
  /// randomly sampled Wolfram Demonstration notebook (a crystal-structure
  /// viewer). Its control panel is `Column[(Row[#, Spacer[3]] &) /@ {rows}]`
  /// — a pure function mapped over literal rows — which used to build no
  /// widget at all. This is a self-authored, construct-equivalent example.
  #[test]
  fn column_of_mapped_rows_exposes_its_controls() {
    let code = r#"Manipulate[
      Graphics[{c, Circle[{0, 0}, r]}],
      Column[(Row[#, Spacer[3]] &) /@ {
        {"radius", Control[{{r, 1, "r"}, 0, 2, 0.1}]},
        {Style["colour", Bold]},
        {Control[{{c, Red, ""}, Red, ImageSize -> Tiny}]}}],
      ControlPlacement -> Left, SaveDefinitions -> True]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let state = ManipulateState::from_expr(&expr).expect("build widget");
    assert_eq!(state.error, None);
    let names: Vec<&str> = state.controls.iter().map(|c| c.name()).collect();
    assert!(names.contains(&"r") && names.contains(&"c"), "{names:?}");
    assert!(state.graphics_handle.is_some(), "the graphic should render");
  }

  /// A choice label written as `StringForm["1``", Style[1/2, Tiny]]` shows the
  /// filled template, not the call's source.
  #[test]
  fn choice_label_fills_string_form_slots() {
    let code = r#"Manipulate[n,
      {{n, 5, "n"}, {5 -> StringForm["``", Style[1/2, Tiny]],
        9 -> StringForm["1``", Style[1/2, Tiny]]}}]"#;
    let expr = woxi::interpret_to_expr(code).expect("parse Manipulate expr");
    let state = ManipulateState::from_expr(&expr).expect("build widget");
    let labels = state
      .controls
      .iter()
      .find_map(|c| match c {
        ControlState::Discrete { value_labels, .. } => {
          Some(value_labels.clone())
        }
        _ => None,
      })
      .expect("discrete control");
    assert_eq!(labels, vec!["1/2", "11/2"]);
  }

  /// `SpanFromLeft` in a display `Grid` is layout, not a cell to draw.
  #[test]
  fn display_grid_span_marker_draws_nothing() {
    let node = build_manipulate_display(
      r#"Grid[{{"performance", SpanFromLeft}, {"quality", "x"}}]"#,
      &[],
    );
    assert!(!format!("{node:?}").contains("SpanFromLeft"), "{node:?}");
  }
}
