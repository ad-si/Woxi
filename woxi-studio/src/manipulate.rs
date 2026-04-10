//! Woxi Studio helpers for rendering interactive `Manipulate[…]` cells.
//!
//! A held `Manipulate` expression is parsed into a `ManipulateState` that
//! stores the body (as InputForm source), the per-variable controls, the
//! current value of each control, and the currently rendered output
//! (SVG or text). On each slider/picklist change the state is updated
//! in place and the body is re-evaluated inside a `Block[{…}, body]` so
//! that free control variables are substituted.

use std::sync::Arc;

use iced::widget::image;
use iced::widget::svg;
use woxi::functions::graphics::{
  ManipulateControl, ManipulateSpec, extract_manipulate_spec,
  manipulate_block_code,
};
use woxi::syntax::Expr;

use crate::rasterize_svg;

/// Runtime state for a single control inside a Manipulate cell.
#[derive(Debug, Clone)]
pub enum ControlState {
  Continuous {
    name: String,
    label: String,
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
    /// Each entry is an InputForm fragment ready for substitution.
    values: Vec<String>,
    current_index: usize,
  },
}

impl ControlState {
  pub fn name(&self) -> &str {
    match self {
      ControlState::Continuous { name, .. } => name,
      ControlState::Discrete { name, .. } => name,
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
  pub graphics_svg: Option<String>,
  pub graphics_handle: Option<svg::Handle>,
  pub graphics_image: Option<(image::Handle, u32, u32)>,
  pub text_output: Option<String>,
  pub error: Option<String>,
}

impl ManipulateState {
  /// Build a `ManipulateState` from an evaluated expression. Returns
  /// `None` if `expr` is not a well-formed Manipulate (in which case
  /// the caller should fall back to the normal text/graphics path).
  pub fn from_expr(
    expr: &Expr,
    scale_factor: f32,
    fontdb: &Arc<resvg::usvg::fontdb::Database>,
  ) -> Option<Self> {
    let spec = extract_manipulate_spec(expr)?;
    let controls = controls_from_spec(&spec);
    let mut state = ManipulateState {
      body: spec.body_code,
      initialization: spec.initialization,
      controls,
      graphics_svg: None,
      graphics_handle: None,
      graphics_image: None,
      text_output: None,
      error: None,
    };
    state.reevaluate(scale_factor, fontdb);
    Some(state)
  }

  /// Re-run the body with the current control bindings and update the
  /// cached SVG / text output. Called on every slider change.
  pub fn reevaluate(
    &mut self,
    scale_factor: f32,
    fontdb: &Arc<resvg::usvg::fontdb::Database>,
  ) {
    let bindings: Vec<(String, String)> = self
      .controls
      .iter()
      .map(|c| (c.name().to_string(), c.current_code()))
      .collect();
    let block = manipulate_block_code(&self.body, &bindings);
    // Prepend `Initialization :> …` code so helper definitions made there
    // (e.g. `d[t_] := …`) are in scope while the body evaluates. The init
    // is a CompoundExpression — join with `;` so both run in one call.
    let code = match self.initialization.as_deref() {
      Some(init) => format!("{init}; {block}"),
      None => block,
    };

    // Clear output state before re-evaluating so a partial result
    // doesn't linger when the new evaluation produces less output.
    self.graphics_svg = None;
    self.graphics_handle = None;
    self.graphics_image = None;
    self.text_output = None;
    self.error = None;

    match woxi::interpret_with_stdout(&code) {
      Ok(result) => {
        if let Some(svg) = result.graphics {
          self.graphics_handle =
            Some(svg::Handle::from_memory(svg.as_bytes().to_vec()));
          self.graphics_image = rasterize_svg(&svg, scale_factor, fontdb);
          self.graphics_svg = Some(svg);
        } else if result.result != "\0" {
          let cleaned = result
            .result
            .replace("-Graphics-", "")
            .replace("-Graphics3D-", "")
            .replace("-Image-", "");
          let cleaned = cleaned.trim();
          if !cleaned.is_empty() {
            self.text_output = Some(cleaned.to_string());
          }
        }
      }
      Err(e) => {
        self.error = Some(format!("{e}"));
      }
    }
  }

  /// Re-rasterize the current SVG at a new scale factor. Called when
  /// the window DPI changes (mirrors the cell-level rasterize flow).
  pub fn rerasterize(
    &mut self,
    scale_factor: f32,
    fontdb: &Arc<resvg::usvg::fontdb::Database>,
  ) {
    self.graphics_image = self
      .graphics_svg
      .as_ref()
      .and_then(|s| rasterize_svg(s, scale_factor, fontdb));
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
          min: *min,
          max: *max,
          step,
          current: *initial,
        }
      }
      ManipulateControl::Discrete {
        name,
        label,
        values,
        initial_index,
      } => ControlState::Discrete {
        name: name.clone(),
        label: label.clone(),
        values: values.clone(),
        current_index: *initial_index,
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
