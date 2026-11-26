//! Run every Input cell of a notebook through the exact pipeline Woxi
//! Studio uses for interactive cells (evaluate, detect a Manipulate,
//! build the widget state, re-evaluate the body once) and dump the
//! resulting widget structure. This makes "does this notebook's
//! Manipulate work in the Studio?" checkable without launching the GUI.
//!
//! Usage: cargo run --example dump_manipulate -- path/to/notebook.nb [svg-out-dir]
//!
//! When `svg-out-dir` is given, each widget's rendered SVG is written to
//! `<svg-out-dir>/widget-<n>.svg` so the visual output can be inspected.
//! Set `OVERRIDE="x=0.5,y=1"` to render the SVG with those control values
//! in place of the initial ones (e.g. to compare against a stored snapshot).
//! Set `MOVE="x=0.5,y=1"` to move the controls themselves to those values
//! and re-evaluate, so the side displays are dumped at them as well.
//! Set `CLICKS="1.5,2.5;9.5,2.5"` to click the picture's `ClickPane` at
//! those pane coordinates in turn; the variables (and, with `svg-out-dir`,
//! the picture as `widget-<n>-click-<k>.svg`) are dumped after each click.
//! Set `TICKS=40` to press play and advance the animation that many ticks;
//! the control values (and, with `svg-out-dir`, the picture as
//! `widget-<n>-ticks.svg`) are dumped afterwards.
//! Set `PRESS_PICTURE=k` to press the k-th `Button` drawn inside the
//! picture; the variables are dumped afterwards.
//! Set `PRESS_BUTTONS=1` to press every control-panel button in turn, or
//! `PRESS_BUTTONS="Solve,Reset"` to press only the buttons with those labels;
//! the variables are dumped after each press.

use std::path::PathBuf;

// The full ControlState API is only partially exercised here; the unused
// remainder is fine for a diagnostic example.
#[allow(dead_code)]
#[path = "../src/manipulate.rs"]
mod manipulate;

use woxi::notebook::{CellEntry, CellStyle, parse_notebook};

/// Mirror the Studio: names the dump kept in the private `$CellContext``
/// may shadow built-ins, so unprotect them before running the saved
/// `Initialization`.
fn unprotect_context_symbols(box_dump: &str) {
  let names = woxi::notebook::saved_initialization_context_symbols(box_dump);
  if !names.is_empty() {
    let _ = woxi::interpret(&format!("Unprotect[{}]", names.join(", ")));
  }
}

fn main() {
  let path: PathBuf = std::env::args()
    .nth(1)
    .expect("notebook path required")
    .into();
  let svg_out_dir: Option<PathBuf> = std::env::args().nth(2).map(Into::into);
  let src = std::fs::read_to_string(&path).expect("read file");
  let nb = parse_notebook(&src).expect("parse notebook");

  let mut all_cells = Vec::new();
  for entry in &nb.cells {
    match entry {
      CellEntry::Single(c) => all_cells.push(c.clone()),
      CellEntry::Group(g) => {
        for c in &g.cells {
          all_cells.push(c.clone());
        }
      }
    }
  }

  // A notebook downloaded straight from the Demonstrations Project holds
  // only the compiled widget dump in an Output cell. The Studio rebuilds a
  // `Manipulate[…]` source from it (see
  // `instantiate_manipulate_from_box_dump`); mirror that by inserting the
  // reconstructed source as a synthetic Input cell ahead of such a dump.
  let mut with_sources = Vec::new();
  for (idx, cell) in all_cells.iter().enumerate() {
    let follows_source = idx > 0
      && matches!(all_cells[idx - 1].style, CellStyle::Input | CellStyle::Code);
    if cell.style == CellStyle::Output
      && !follows_source
      && let Some(code) =
        woxi::notebook::reconstruct_manipulate_from_box_dump(&cell.content)
    {
      let mut synthetic = cell.clone();
      synthetic.style = CellStyle::Input;
      synthetic.content = code;
      with_sources.push(synthetic);
    }
    with_sources.push(cell.clone());
  }
  let all_cells = with_sources;

  let mut widget_count = 0;
  for (idx, cell) in all_cells.iter().enumerate() {
    // A standalone stored widget dump (no Input cell, e.g. a notebook
    // downloaded from the Demonstrations Project) is rebuilt into a plain
    // `Manipulate[…]` source first, as the Studio does when opening it.
    let follows_source = idx > 0
      && matches!(all_cells[idx - 1].style, CellStyle::Input | CellStyle::Code);
    let reconstructed = if cell.style == CellStyle::Output && !follows_source {
      woxi::notebook::reconstruct_manipulate_from_box_dump(&cell.content)
    } else {
      None
    };
    if reconstructed.is_none()
      && !matches!(cell.style, CellStyle::Input | CellStyle::Code)
    {
      continue;
    }
    // Definitions saved in the dump's own Initialization run first.
    if reconstructed.is_some()
      && let Some(init) =
        woxi::notebook::extract_saved_initialization(&cell.content)
    {
      unprotect_context_symbols(&cell.content);
      let _ = woxi::interpret(&init);
    }
    let code = woxi::notebook::strip_cell_context_prefix(
      reconstructed.as_deref().unwrap_or(cell.content.trim()),
    );
    // Like the Studio, unprotect the following dump's `$CellContext``
    // names before the source runs, even when the source's own
    // `Initialization` is what defines them.
    if let Some(next) = all_cells.get(idx + 1)
      && next.style == CellStyle::Output
    {
      unprotect_context_symbols(&next.content);
    }
    for stmt in woxi::split_into_statements(&code) {
      // Evaluate for side effects (definitions) exactly like the studio.
      let eval = woxi::interpret_with_stdout(&stmt);
      let Ok(expr) = woxi::interpret_to_expr(&stmt) else {
        continue;
      };
      // `Manipulate[…, SaveDefinitions -> True]` stores the definitions its
      // body depends on in the following Output cell's widget dump, because
      // the Manipulate's own `Initialization` option is absent. The Studio
      // runs the recovered copy before instantiating, or a Demonstration's
      // helper functions look undefined — but only when the live source
      // truly has no `Initialization` of its own: Wolfram embeds this same
      // box-dump shape for an ordinary `Initialization :> …` Manipulate too,
      // and running that FullForm copy a second time (ahead of the live one
      // `ManipulateState::from_expr` runs) can leave a helper function bound
      // to a broken duplicate rule.
      let has_own_initialization =
        woxi::functions::graphics::manipulate_has_own_initialization(&expr);
      if !has_own_initialization
        && let Some(next) = all_cells.get(idx + 1)
        && next.style == CellStyle::Output
        && let Some(init) =
          woxi::notebook::extract_saved_initialization(&next.content)
      {
        let _ = woxi::interpret(&init);
      }
      let Some(state) = manipulate::ManipulateState::from_expr(&expr) else {
        if let Ok(res) = &eval
          && res.result.starts_with("Manipulate[")
        {
          println!("!! Manipulate did NOT build a widget:");
          let snippet: String = stmt.chars().take(160).collect();
          println!("   {snippet}");
        }
        continue;
      };
      let mut state = state;
      if let Some(next) = all_cells.get(idx + 1)
        && next.style == CellStyle::Output
      {
        state.apply_saved_variables(
          &woxi::notebook::extract_saved_manipulate_variables(&next.content),
        );
      }
      // `MOVE="x=0.5,y=1"` moves the controls themselves and re-evaluates,
      // so the side displays (not just the picture) show those values.
      if let Ok(mv) = std::env::var("MOVE") {
        let saved: Vec<(String, String)> = split_top_level_commas(&mv)
          .iter()
          .filter_map(|kv| kv.split_once('='))
          .map(|(k, v)| (k.to_string(), v.to_string()))
          .collect();
        state.apply_saved_variables(&saved);
      }
      widget_count += 1;
      println!("=== Interactive widget #{widget_count} ===");
      println!("animated: {}", state.animated);
      println!("controls:");
      for (i, c) in state.controls.iter().enumerate() {
        // A control belonging to a `PaneSelector` pane the selector is not
        // currently showing is built but left off the panel.
        let hidden = !state.control_is_visible.get(i).copied().unwrap_or(true);
        let mark = if hidden { " [hidden]" } else { "" };
        println!("  {c:?}{mark}");
        // Only a control placed apart from the rest of the panel says so.
        let side = state.placement_of(i);
        if side != state.control_placement {
          println!("    placement: {side:?}");
        }
        if let Some(icon) = state.label_svg(c.name()) {
          println!("    label icon: {}x{}", icon.width, icon.height);
        }
        if let Some(Some((r, g, b))) = state.control_face.get(i) {
          println!("    face: rgb({r:.3}, {g:.3}, {b:.3})");
        }
      }
      if !state.state.is_empty() {
        println!("state vars:");
        for (n, v) in &state.state {
          let snippet: String = v.chars().take(100).collect();
          println!("  {n} = {snippet}");
        }
      }
      if !state.displays.is_empty() {
        println!("displays:");
        for d in &state.displays {
          let snippet: String = d.chars().take(120).collect();
          println!("  {snippet}");
        }
        println!("display trees:");
        for t in &state.display_trees {
          let dbg = format!("{t:?}");
          let snippet: String = dbg
            .chars()
            .take(
              std::env::var("DUMP_TREE_CHARS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(300),
            )
            .collect();
          println!("  {snippet}");
        }
      }
      println!("body:");
      for line in state.body.lines() {
        println!("  {line}");
      }
      println!(
        "render: graphics={} sound={} typeset={} text={:?} error={:?}",
        state.graphics_handle.is_some(),
        state.body_sound.is_some(),
        state.typeset_output.is_some(),
        state.text_output,
        state.error
      );
      // The typeset choice labels (popup/setter icons) of each discrete
      // control, so their rendering can be inspected too.
      if let Some(dir) = &svg_out_dir {
        for (ci, control) in state.controls.iter().enumerate() {
          let manipulate::ControlState::Discrete {
            value_label_svgs, ..
          } = control
          else {
            continue;
          };
          for (vi, handle) in value_label_svgs.iter().enumerate() {
            if let Some(handle) = handle
              && let iced::advanced::svg::Data::Bytes(bytes) = handle.data()
            {
              std::fs::create_dir_all(dir).expect("create svg out dir");
              let out = dir.join(format!(
                "widget-{widget_count}-control-{}-choice-{}.svg",
                ci + 1,
                vi + 1
              ));
              std::fs::write(&out, bytes).expect("write svg");
              println!("wrote {}", out.display());
            }
          }
        }
      }
      // A typeset textual result is drawn as its own SVG.
      if let Some(dir) = &svg_out_dir
        && let Some(svg) = &state.typeset_output
      {
        let out = dir.join(format!("widget-{widget_count}-typeset.svg"));
        std::fs::write(&out, svg).expect("write typeset SVG");
        println!("wrote {}", out.display());
      }
      if let Some(dir) = &svg_out_dir
        && state.graphics_handle.is_some()
      {
        // Re-render through the same bindings the widget uses so the SVG
        // bytes can be captured (the iced handle doesn't expose them).
        let mut bindings: Vec<(String, String)> = state
          .controls
          .iter()
          .filter(|c| c.binds_variable())
          .map(|c| (c.name().to_string(), c.current_code()))
          .collect();
        bindings.extend(state.state.iter().cloned());
        if let Ok(ov) = std::env::var("OVERRIDE") {
          for kv in split_top_level_commas(&ov) {
            if let Some((k, v)) = kv.split_once('=') {
              bindings.retain(|(n, _)| n != k);
              bindings.push((k.to_string(), v.to_string()));
            }
          }
        }
        // Like the widget's own re-evaluation, the Initialization is not
        // run again: it already ran once (with the control names it uses as
        // pattern variables left unbound) when the state was built.
        let render = woxi::with_scoped_globals(&bindings, || {
          woxi::interpret_with_stdout(&state.display_code(&state.body))
        });
        // Lay the output out in the `ContentSize` area, as the Studio does.
        let render = render.map(|res| {
          state
            .content_width
            .and_then(|w| manipulate::flow_to_content_width(&res, w))
            .unwrap_or(res)
        });
        let render = render.map(|res| {
          state
            .content_height
            .and_then(|h| manipulate::fit_to_content_height(&res, h))
            .unwrap_or(res)
        });
        if let Ok(res) = render
          && let Some(svg) = res.graphics
        {
          std::fs::create_dir_all(dir).expect("create svg out dir");
          let out = dir.join(format!("widget-{widget_count}.svg"));
          std::fs::write(&out, svg).expect("write svg");
          println!("wrote {}", out.display());
        }
      }
      if let Ok(clicks) = std::env::var("CLICKS") {
        click_pane(&mut state, &clicks, widget_count, svg_out_dir.as_deref());
      }
      if let Some(ticks) = std::env::var("TICKS")
        .ok()
        .and_then(|t| t.parse::<usize>().ok())
      {
        if !state.playing {
          state.toggle_play();
        }
        let started = std::time::Instant::now();
        for _ in 0..ticks {
          state.advance_animation();
        }
        println!(
          "after {ticks} ticks ({:.2?}): graphics={} text={:?} error={:?}",
          started.elapsed(),
          state.graphics_handle.is_some(),
          state.text_output,
          state.error
        );
        for c in state.controls.iter().filter(|c| c.binds_variable()) {
          println!("  {} = {}", c.name(), c.current_code());
        }
        if let Some(dir) = &svg_out_dir
          && let Some(handle) = &state.graphics_handle
          && let iced::advanced::svg::Data::Bytes(bytes) = handle.data()
        {
          std::fs::create_dir_all(dir).expect("create svg out dir");
          let out = dir.join(format!("widget-{widget_count}-ticks.svg"));
          std::fs::write(&out, bytes).expect("write svg");
          println!("wrote {}", out.display());
        }
      }
      if let Some(k) = std::env::var("PRESS_PICTURE")
        .ok()
        .and_then(|k| k.parse::<usize>().ok())
        && let Some(region) = state.button_regions.get(k)
      {
        let action = region.action.clone();
        println!("picture button {k}: {action}");
        state.apply_button_action(&action);
        println!("error={:?}", state.error);
        for (n, v) in &state.state {
          println!("  {n} = {v}");
        }
      }
      if std::env::var("PRESS_BUTTONS").is_ok() {
        let mut st = state;
        let actions: Vec<String> = st
          .controls
          .iter()
          .filter_map(|c| match c {
            manipulate::ControlState::Button { action, label, .. } => {
              let want = std::env::var("PRESS_BUTTONS").unwrap_or_default();
              (want.is_empty()
                || want == "1"
                || want.split(',').any(|w| w == label))
              .then(|| action.clone())
            }
            _ => None,
          })
          .collect();
        for a in actions {
          st.apply_button_action(&a);
          println!(
            "after `{a}`: graphics={} text={:?} error={:?}",
            st.graphics_handle.is_some(),
            st.text_output,
            st.error
          );
          for (n, v) in &st.state {
            println!("  {n} = {v}");
          }
        }
      }
      println!();
    }
  }
  println!("Interactive widgets built: {widget_count}");
}

/// Click the widget's first `ClickPane` at each `;`-separated `x,y` pane
/// coordinate of `clicks`, dumping the variables after every click.
fn click_pane(
  state: &mut manipulate::ManipulateState,
  clicks: &str,
  widget: usize,
  svg_out_dir: Option<&std::path::Path>,
) {
  for (k, click) in clicks.split(';').enumerate() {
    let Some(region) = state.click_regions.first().cloned() else {
      println!("!! no ClickPane to click");
      return;
    };
    let Some((gx, gy)) = click.split_once(',').and_then(|(x, y)| {
      Some((x.trim().parse::<f64>().ok()?, y.trim().parse::<f64>().ok()?))
    }) else {
      continue;
    };
    let h = &region.hit;
    let fx = ((gx - h.x_min) / (h.x_max - h.x_min)) as f32;
    let fy = ((h.y_max - gy) / (h.y_max - h.y_min)) as f32;
    state.pointer =
      (region.x + fx * region.width, region.y + fy * region.height);
    state.apply_pane_click();
    println!(
      "after click {} at {{{gx}, {gy}}}: error={:?}",
      k + 1,
      state.error
    );
    for (n, v) in &state.state {
      let snippet: String = v.chars().take(200).collect();
      println!("  {n} = {snippet}");
    }
    if let (Some(dir), Some(svg)) = (svg_out_dir, state.current_svg()) {
      let out = dir.join(format!("widget-{widget}-click-{}.svg", k + 1));
      std::fs::write(&out, svg).expect("write svg");
      println!("wrote {}", out.display());
    }
  }
}

/// Split an `OVERRIDE` list on the commas that are not nested inside
/// brackets, so list values such as `step={1,2,3}` stay whole.
fn split_top_level_commas(s: &str) -> Vec<&str> {
  let mut parts = Vec::new();
  let mut depth = 0i32;
  let mut start = 0;
  for (i, ch) in s.char_indices() {
    match ch {
      '{' | '[' | '(' => depth += 1,
      '}' | ']' | ')' => depth -= 1,
      ',' if depth == 0 => {
        parts.push(&s[start..i]);
        start = i + 1;
      }
      _ => {}
    }
  }
  parts.push(&s[start..]);
  parts
}
