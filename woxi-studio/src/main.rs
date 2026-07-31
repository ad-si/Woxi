mod cell_type_dropdown;
mod highlighter;
#[cfg(target_os = "macos")]
mod macos_open;
mod manipulate;

use woxi::notebook;

use iced::keyboard;
use iced::overlay::menu;
use iced::widget::operation::focus;
use iced::widget::{
  Column, Row, button, checkbox, column, container, image, mouse_area, opaque,
  pick_list, rich_text, row, rule, scrollable, slider, space, stack, svg, text,
  text_editor,
};
use iced::{
  Background, Border, Center, Color, Element, Fill, Font, Subscription, Task,
  Theme,
};

use notebook::{Cell, CellEntry, CellGroup, CellStyle, Notebook};
use std::path::PathBuf;
use std::sync::Arc;

fn main() -> iced::Result {
  #[cfg(target_os = "macos")]
  macos_open::register();

  iced::application(WoxiStudio::new, WoxiStudio::update, WoxiStudio::view)
    .title(|state: &WoxiStudio| match &state.file_path {
      Some(path) => {
        let name = path
          .file_name()
          .map(|n| n.to_string_lossy().into_owned())
          .unwrap_or_else(|| path.display().to_string());
        format!("Woxi Studio | {name}")
      }
      None => String::from("Woxi Studio"),
    })
    .subscription(WoxiStudio::subscription)
    .theme(|state: &WoxiStudio| state.theme.clone())
    .default_font(Font::MONOSPACE)
    .exit_on_close_request(false)
    .run()
}

// ── Application State ───────────────────────────────────────────────

struct WoxiStudio {
  /// Path to the currently opened .nb file, if any.
  file_path: Option<PathBuf>,
  /// The in-memory notebook model.
  notebook: Notebook,
  /// Per-cell editor state.
  cell_editors: Vec<CellEditor>,
  /// Which cell is currently focused (index into cell_editors).
  focused_cell: Option<usize>,
  /// Whether there are unsaved changes.
  is_dirty: bool,
  /// Whether a file operation is in progress.
  is_loading: bool,
  /// Status bar message.
  status: String,
  /// Application theme.
  theme: Theme,
  /// User's theme choice (Auto / Light / Dark).
  theme_choice: ThemeChoice,
  /// Which cell has its type menu open (if any).
  cell_type_menu_open: Option<usize>,
  /// Which add-cell divider is focused (index = cell above the divider).
  focused_divider: Option<usize>,
  /// Style to use for new cells.
  new_cell_style: CellStyle,
  /// Whether preview mode is active (hides gutter, borders, etc).
  preview_mode: bool,
  /// Display scale factor for HiDPI rasterization.
  scale_factor: f32,
  /// Font database for SVG text rendering (loaded once at startup).
  fontdb: Arc<resvg::usvg::fontdb::Database>,
  /// Index of the cell whose graphic is shown in the fullscreen modal.
  graphics_modal_cell: Option<usize>,
  /// Index of the cell whose graphic context menu is open (right-click menu).
  graphics_context_menu_cell: Option<usize>,
  /// Position (in window logical coords) where the context menu should appear.
  graphics_context_menu_pos: iced::Point,
  /// Latest known cursor position, tracked via global mouse events so we can
  /// place the right-click menu at the cursor.
  cursor_position: iced::Point,
  /// Whether the table of contents sidebar is visible.
  show_toc: bool,
  /// Current window width in logical pixels.
  window_width: f32,
  /// Which cell's gutter area is currently hovered (for showing drag handle).
  hovered_gutter: Option<usize>,
  /// Cell index currently being dragged for reordering.
  dragging_cell: Option<usize>,
  /// Drop target index (the cell index before which the dragged cell will be inserted).
  drop_target: Option<usize>,
  /// In-progress audio playback, if any. Tracks the external player
  /// process so it can be paused/resumed and so the play button of the
  /// owning cell can show a pause icon while audio is playing.
  playback: Option<Playback>,
  /// When the last animation advance finished. Ticks generated before this
  /// instant piled up in the runtime's message queue while that (blocking)
  /// advance ran; they are dropped instead of processed — see
  /// [`animation_tick_is_fresh`].
  last_anim_advance: Option<std::time::Instant>,
}

/// A running (or paused) external audio-player process tied to a cell.
struct Playback {
  /// Index of the cell whose audio is playing.
  cell: usize,
  /// The spawned player process (afplay / powershell / paplay …).
  child: std::process::Child,
  /// Whether playback is currently paused (process is SIGSTOP'd).
  paused: bool,
}

impl Drop for Playback {
  /// Kill the player when playback state is discarded so audio stops and
  /// no (possibly SIGSTOP'd) process outlives the app. Both calls are
  /// no-ops if the process already exited and was reaped.
  fn drop(&mut self) {
    let _ = self.child.kill();
    let _ = self.child.wait();
  }
}

/// Editor state for a single cell.
struct CellEditor {
  content: text_editor::Content,
  style: CellStyle,
  /// Cached output from evaluating this cell (raw text).
  output: Option<String>,
  /// Captured Print output.
  stdout: Option<String>,
  /// SVG data from Graphics/Plot evaluation.
  graphics_svg: Option<String>,
  /// Cached svg handle, built once per evaluation to avoid per-frame
  /// allocation and hashing during scroll.
  graphics_handle: Option<svg::Handle>,
  /// Pre-rasterized image of the SVG (avoids resvg parse on scroll).
  graphics_image: Option<(iced::widget::image::Handle, u32, u32)>,
  /// A display-only stored rendering from the .nb file (e.g. a
  /// Demonstrations snapshot decoded from `RasterBox[CompressedData[…]]`).
  /// The cell shows only the graphic; its box text stays in `content` so
  /// saving round-trips it.
  stored_graphic: bool,
  /// Typeset SVG renderings of the result outputs — the same SVGs the
  /// Playground shows — one per result statement that produced one. This is how
  /// number/superscript/fraction formatting is reused instead of being
  /// re-implemented for Studio. Empty when results display as plain text
  /// (trivial literals, or a notebook loaded from disk that hasn't been
  /// re-evaluated).
  output_svgs: Vec<String>,
  /// Pre-rasterized images of `output_svgs`, rebuilt on scale change.
  output_images: Vec<(iced::widget::image::Handle, u32, u32)>,
  /// Dark-mode flag in effect when `output_svgs` were generated. When it no
  /// longer matches the current theme the baked text color would clash with the
  /// background, so the view falls back to the theme-aware text output until the
  /// cell is re-evaluated.
  output_dark: bool,
  /// Whether every result statement in the cell produced a typeset SVG. Only
  /// then does the view render the SVG images (in place of the text output);
  /// otherwise — text-only results, mixed cells, or a rasterization failure —
  /// it shows the plain-text output so nothing is dropped.
  output_all_svg: bool,
  /// Playable audio from Play/Sound synthesis or an Audio object (file-backed
  /// or from sample data), if any. When present the cell renders a graphical
  /// audio player.
  sound: Option<woxi::AudioOutput>,
  /// Warning messages from evaluation (e.g. unimplemented functions).
  warnings: Vec<String>,
  /// Undo stack: previous text snapshots.
  undo_stack: Vec<String>,
  /// Redo stack: snapshots restored via undo.
  redo_stack: Vec<String>,
  /// Whether the input has changed since the last evaluation.
  output_stale: bool,
  /// For Chapter/Subchapter cells: whether the section is collapsed,
  /// hiding all cells below it until the next same-or-higher heading.
  is_collapsed: bool,
  /// Interactive Manipulate widget state, if the last evaluation
  /// produced a well-formed `Manipulate[…]` expression. When present,
  /// the cell renders sliders / pick lists instead of the plain echo.
  manipulate_state: Option<manipulate::ManipulateState>,
  /// `(label, uri)` pairs for `Hyperlink[…]` results. When non-empty,
  /// the cell renders clickable link buttons instead of (or alongside)
  /// the plain text echo.
  hyperlinks: Vec<(String, String)>,
  /// Selectable text_editor content for the output text.
  output_content: text_editor::Content,
  /// Selectable text_editor content for stdout (Print output).
  stdout_content: text_editor::Content,
}

// ── Messages ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[allow(dead_code)]
enum Message {
  // File operations
  NewNotebook,
  OpenFile,
  /// Tick on macOS to drain any pending Apple Event file-open requests.
  #[cfg(target_os = "macos")]
  PollPendingOpens,
  FileOpened(Result<(PathBuf, Arc<String>), FileError>),
  SaveFile,
  SaveFileAs,
  FileSaved(Result<PathBuf, FileError>),
  FileSavedThenClose(iced::window::Id, Result<PathBuf, FileError>),

  // Export
  ExportAs(ExportFormat),
  FileExported(Result<PathBuf, FileError>),

  // Cell editing
  CellAction(usize, text_editor::Action),
  /// Action on a read-only output editor (selection only, edits discarded).
  OutputAction(usize, text_editor::Action),
  /// Action on a read-only stdout editor (selection only, edits discarded).
  StdoutAction(usize, text_editor::Action),
  WrapSelection(usize, char, char),
  Undo(usize),
  Redo(usize),
  IndentLines(usize),
  UnindentLines(usize),
  ToggleComment(usize),
  CellStyleChanged(usize, CellStyle),
  FocusCell(usize),
  ScrollCellsToEnd,

  // Cell management
  AddCellBelow(usize),
  AddCellAbove(usize),
  DeleteCell(usize),
  MoveCellUp(usize),
  MoveCellDown(usize),

  // Evaluation
  EvaluateCell(usize),
  EvaluateAll,

  /// Toggle playback of the given cell's audio (from Play[…] / Sound[…] /
  /// Audio[…]): start playing, pause, or resume.
  PlaySound(usize),
  /// Periodic poll of the external audio player so the pause button
  /// reverts to a play button when playback finishes on its own.
  PlaybackTick,

  // Settings
  ThemeChanged(ThemeChoice),
  NewCellStyleChanged(CellStyle),

  // Cell type menu
  ToggleCellTypeMenu(usize),

  // Gutter hover (for showing drag handle)
  GutterEnter(usize),
  GutterExit(usize),

  // Cell drag-and-drop reordering
  DragStart(usize),
  DragOverCell(usize),
  DragEnd,

  // Collapse/expand Chapter or Subchapter
  ToggleCollapse(usize),

  // Preview mode
  TogglePreview,

  // Table of contents sidebar
  ToggleToc,
  ScrollToCell(usize),

  // Window
  CloseRequested(iced::window::Id),
  CloseConfirmed(iced::window::Id, rfd::MessageDialogResult),

  // Cell navigation
  FocusDividerBelow(usize),
  FocusDividerAbove(usize),

  // Keyboard
  KeyPressed(keyboard::Key, keyboard::Modifiers),

  // Display
  ScaleFactorChanged(f32),
  WindowResized(iced::Size),

  // Graphics modal
  OpenGraphicsModal(usize),
  CloseGraphicsModal,

  // Graphics right-click context menu
  ShowGraphicsContextMenu(usize),
  CloseGraphicsContextMenu,
  SaveGraphicAs(usize),
  GraphicSaved(Result<PathBuf, FileError>),
  CursorMoved(iced::Point),

  // Manipulate interactive widgets
  ManipulateContinuousChanged(usize, usize, f64),
  ManipulateDiscreteChanged(usize, usize, String),
  /// (cell_idx, ctrl_idx, axis 0=x/1=y, value)
  ManipulateSlider2DChanged(usize, usize, u8, f64),
  /// (cell_idx, ctrl_idx, endpoint 0=low/1=high, value)
  ManipulateIntervalChanged(usize, usize, u8, f64),
  /// One coordinate of a Locator point moved.
  /// (cell_idx, ctrl_idx, point_idx, axis 0=x/1=y, value)
  ManipulateLocatorChanged(usize, usize, usize, u8, f64),
  /// Add a point to a `LocatorAutoCreate` locator (at the range centre).
  /// (cell_idx, ctrl_idx)
  ManipulateLocatorAdded(usize, usize),
  /// Remove a point from a `LocatorAutoCreate` locator.
  /// (cell_idx, ctrl_idx, point_idx)
  ManipulateLocatorRemoved(usize, usize, usize),
  /// A checkbox in a Manipulate display element was toggled.
  /// (cell_idx, write-back assignment, e.g. `data[[3, 5]] = 1`)
  ManipulateDisplayToggled(usize, String),
  /// The throttle timer for a Manipulate cell fired; re-evaluate the body
  /// with the latest control values if any change is still pending.
  /// (cell_idx)
  ManipulateReeval(usize),
  /// Periodic tick advancing every playing Animate/ListAnimate widget by
  /// one animation step. Carries the instant the tick was generated so
  /// stale (backlogged) ticks can be dropped instead of each triggering a
  /// full re-evaluation.
  ManipulateAnimationTick(std::time::Instant),
  /// Toggle play/pause of an animated (Animate/ListAnimate) widget.
  ManipulateTogglePlay(usize),
  /// A `Button[…]` control row was pressed; run its action code.
  /// (cell_idx, ctrl_idx)
  ManipulateButtonPressed(usize, usize),
  /// Swallow an interaction with a disabled control (its `Enabled` condition
  /// is currently `False`) without changing any state.
  Noop,

  // Hyperlink: open the given URL in the user's default browser.
  OpenHyperlink(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThemeChoice {
  Auto,
  Light,
  Dark,
}

impl std::fmt::Display for ThemeChoice {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      ThemeChoice::Auto => write!(f, "Auto"),
      ThemeChoice::Light => write!(f, "Light"),
      ThemeChoice::Dark => write!(f, "Dark"),
    }
  }
}

impl ThemeChoice {
  const ALL: &'static [ThemeChoice] =
    &[ThemeChoice::Auto, ThemeChoice::Light, ThemeChoice::Dark];
}

/// Detect the OS theme, falling back to Dark.
fn detect_system_theme() -> Theme {
  match dark_light::detect() {
    Ok(dark_light::Mode::Light) => Theme::Light,
    _ => Theme::Dark,
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExportFormat {
  MathematicaNotebook,
  JupyterNotebook,
  Markdown,
  LaTeX,
  Typst,
  Pdf,
}

impl std::fmt::Display for ExportFormat {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      ExportFormat::MathematicaNotebook => {
        write!(f, "Mathematica Notebook")
      }
      ExportFormat::JupyterNotebook => {
        write!(f, "Jupyter Notebook")
      }
      ExportFormat::Markdown => write!(f, "Markdown"),
      ExportFormat::LaTeX => write!(f, "LaTeX"),
      ExportFormat::Typst => write!(f, "Typst"),
      ExportFormat::Pdf => write!(f, "PDF"),
    }
  }
}

impl ExportFormat {
  const ALL: &'static [ExportFormat] = &[
    ExportFormat::MathematicaNotebook,
    ExportFormat::JupyterNotebook,
    ExportFormat::Markdown,
    ExportFormat::LaTeX,
    ExportFormat::Typst,
    ExportFormat::Pdf,
  ];
}

// ── Application Logic ───────────────────────────────────────────────

impl WoxiStudio {
  fn new() -> (Self, Task<Message>) {
    let mut notebook = Notebook::new();
    notebook.push_cell(Cell::new(CellStyle::Title, "Untitled Notebook"));
    notebook.push_cell(Cell::new(CellStyle::Input, ""));

    let cell_editors = Self::editors_from_notebook(&notebook);

    let task = if let Some(path) = parse_cli_file_arg() {
      Task::perform(open_file_path(path), Message::FileOpened)
    } else if !std::env::args().any(|a| a == "--new")
      && let Some(path) = load_last_file_path()
    {
      Task::perform(open_file_path(path), Message::FileOpened)
    } else {
      Task::none()
    };

    (
      Self {
        file_path: None,
        notebook,
        cell_editors,
        focused_cell: Some(1),
        is_dirty: false,
        is_loading: false,
        status: String::from("Ready"),
        theme: detect_system_theme(),
        theme_choice: ThemeChoice::Auto,
        cell_type_menu_open: None,
        focused_divider: None,
        new_cell_style: CellStyle::Input,
        preview_mode: false,
        scale_factor: 1.0,
        fontdb: {
          let mut db = resvg::usvg::fontdb::Database::new();
          // Load the same embedded fallbacks used by the command-line
          // rasterizer/PDF exporter so that in-UI graphics render with a
          // consistent typeface even on systems with no system fonts.
          db.load_font_data(
            include_bytes!(
              "../../resources/AtkinsonHyperlegibleMono-VariableFont_wght.ttf"
            )
            .to_vec(),
          );
          db.load_font_data(
            include_bytes!(
              "../../resources/AtkinsonHyperlegibleNext-VariableFont_wght.ttf"
            )
            .to_vec(),
          );
          db.set_monospace_family("Atkinson Hyperlegible Mono");
          db.set_sans_serif_family("Atkinson Hyperlegible Next");
          db.set_serif_family("Atkinson Hyperlegible Next");
          db.set_cursive_family("Atkinson Hyperlegible Next");
          db.set_fantasy_family("Atkinson Hyperlegible Next");
          db.load_system_fonts();
          Arc::new(db)
        },
        graphics_modal_cell: None,
        graphics_context_menu_cell: None,
        graphics_context_menu_pos: iced::Point::ORIGIN,
        cursor_position: iced::Point::ORIGIN,
        show_toc: false,
        window_width: 1024.0,
        hovered_gutter: None,
        dragging_cell: None,
        drop_target: None,
        playback: None,
        last_anim_advance: None,
      },
      task,
    )
  }

  /// Kill any in-progress external audio player and clear playback state
  /// (Playback's Drop impl kills the process).
  fn stop_playback(&mut self) {
    self.playback = None;
  }

  /// Whether the given cell's audio is currently playing (not paused).
  fn is_playing(&self, idx: usize) -> bool {
    self
      .playback
      .as_ref()
      .is_some_and(|p| p.cell == idx && !p.paused)
  }

  /// Build editor state from a notebook.
  /// Output/Print cells within a group are attached to the
  /// preceding Input/Code cell rather than shown separately.
  fn editors_from_notebook(notebook: &Notebook) -> Vec<CellEditor> {
    let mut editors = Vec::new();
    // Input cells seen so far that have not been evaluated yet. A stored
    // interactive widget (Manipulate saved with `SaveDefinitions -> True`)
    // depends on helper functions defined in earlier Input cells — the
    // Demonstrations "Initialization Code" section. Mathematica embeds
    // those definitions in the saved DynamicModuleBox dump; Woxi drops the
    // dump and re-instantiates from the input, so the earlier cells must
    // run first for the widget's body to evaluate.
    let mut pending_init: Vec<String> = Vec::new();
    let mut state_cleared = false;

    for entry in &notebook.cells {
      match entry {
        CellEntry::Single(cell) => {
          // A standalone stored Output/Print renders as a graphic when it
          // decodes to something displayable (snapshot rasters, checkbox
          // grids); anything else keeps an ordinary editor below, so the
          // cell still survives a load/save round-trip either way.
          if matches!(cell.style, CellStyle::Output | CellStyle::Print)
            && let Some(editor) = stored_output_editor(cell)
          {
            editors.push(editor);
            continue;
          }
          if matches!(cell.style, CellStyle::Input | CellStyle::Code) {
            pending_init.push(cell.content.clone());
          }
          editors.push(CellEditor {
            content: text_editor::Content::with_text(&cell.content),
            style: cell.style,
            output: None,
            stdout: None,
            graphics_svg: None,
            graphics_handle: None,
            graphics_image: None,
            output_svgs: Vec::new(),
            output_images: Vec::new(),
            output_dark: false,
            output_all_svg: false,
            sound: None,
            warnings: Vec::new(),
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            output_stale: false,
            is_collapsed: cell.collapsed,
            manipulate_state: None,
            hyperlinks: Vec::new(),
            stored_graphic: false,
            output_content: text_editor::Content::new(),
            stdout_content: text_editor::Content::new(),
          });
        }
        CellEntry::Group(group) => {
          let cells = &group.cells;
          let mut i = 0;
          while i < cells.len() {
            let cell = &cells[i];
            if matches!(cell.style, CellStyle::Input | CellStyle::Code) {
              // Collect following Output/Print cells
              let mut output = None;
              let mut stdout = None;
              let mut j = i + 1;
              while j < cells.len()
                && matches!(
                  cells[j].style,
                  CellStyle::Output | CellStyle::Print
                )
              {
                match cells[j].style {
                  CellStyle::Output => {
                    output = Some(cells[j].content.clone());
                  }
                  CellStyle::Print => {
                    stdout = Some(cells[j].content.clone());
                  }
                  _ => {}
                }
                j += 1;
              }
              // A stored FrontEnd widget dump (`DynamicModuleBox[…]`, the
              // saved form of a live Manipulate) is meaningless as text —
              // drop it and instantiate the interactive widget from the
              // input instead, so the notebook opens with its Manipulate
              // live (as Mathematica would show it).
              let is_widget_dump =
                output.as_deref().is_some_and(is_dynamic_box_dump);
              let manipulate_state = if is_widget_dump {
                // Leftover state from a previously opened notebook must not
                // leak into the widget; start the first instantiation from
                // a clean slate.
                if !state_cleared {
                  woxi::clear_state();
                  state_cleared = true;
                }
                // Run the earlier Input cells (helper definitions) before
                // the widget's body first evaluates; the dump's saved
                // Initialization (SaveDefinitions -> True) then runs on
                // top of them inside instantiate_stored_manipulate.
                evaluate_pending_initialization(&mut pending_init);
                instantiate_stored_manipulate(
                  &cell.content,
                  output.as_deref().unwrap_or(""),
                )
              } else {
                pending_init.push(cell.content.clone());
                None
              };
              let output_content = match &output {
                Some(_) if is_widget_dump => text_editor::Content::new(),
                Some(s) => {
                  let d = s
                    .replace("-Graphics-", "")
                    .replace("-Graphics3D-", "")
                    .replace("-Image-", "");
                  let d = d.trim();
                  if d.is_empty() {
                    text_editor::Content::new()
                  } else {
                    text_editor::Content::with_text(d)
                  }
                }
                None => text_editor::Content::new(),
              };
              let stdout_content = match &stdout {
                Some(s) => text_editor::Content::with_text(s),
                None => text_editor::Content::new(),
              };
              editors.push(CellEditor {
                content: text_editor::Content::with_text(&cell.content),
                style: cell.style,
                output: if is_widget_dump { None } else { output },
                stdout,
                graphics_svg: None,
                graphics_handle: None,
                graphics_image: None,
                output_svgs: Vec::new(),
                output_images: Vec::new(),
                output_dark: false,
                output_all_svg: false,
                sound: None,
                warnings: Vec::new(),
                undo_stack: Vec::new(),
                redo_stack: Vec::new(),
                output_stale: false,
                is_collapsed: false,
                manipulate_state,
                hyperlinks: Vec::new(),
                stored_graphic: false,
                output_content,
                stdout_content,
              });
              i = j;
            } else if matches!(cell.style, CellStyle::Output | CellStyle::Print)
              && let Some(editor) = stored_output_editor(cell)
            {
              // A standalone stored output (no preceding Input) renders as
              // a graphic when it decodes to something displayable — e.g.
              // the Demonstrations "Snapshots" rasters.
              editors.push(editor);
              i += 1;
            } else {
              // Any other cell — including a standalone Output/Print that
              // doesn't decode to a displayable graphic — keeps its own
              // editor so it survives a load/save round-trip.
              editors.push(CellEditor {
                content: text_editor::Content::with_text(&cell.content),
                style: cell.style,
                output: None,
                stdout: None,
                graphics_svg: None,
                graphics_handle: None,
                graphics_image: None,
                output_svgs: Vec::new(),
                output_images: Vec::new(),
                output_dark: false,
                output_all_svg: false,
                sound: None,
                warnings: Vec::new(),
                undo_stack: Vec::new(),
                redo_stack: Vec::new(),
                output_stale: false,
                is_collapsed: false,
                manipulate_state: None,
                hyperlinks: Vec::new(),
                stored_graphic: false,
                output_content: text_editor::Content::new(),
                stdout_content: text_editor::Content::new(),
              });
              i += 1;
            }
          }
        }
      }
    }

    editors
  }

  /// Synchronize the notebook model from the editor state.
  fn sync_notebook_from_editors(&mut self) {
    let mut cells = Vec::new();
    let mut i = 0;
    while i < self.cell_editors.len() {
      let editor = &self.cell_editors[i];
      let content = editor.content.text().trim_end().to_string();
      let mut cell = Cell::new(editor.style, content);
      cell.collapsed = editor.is_collapsed;

      // Group input cells with their output
      if editor.style == CellStyle::Input
        && let Some(ref output) = editor.output
      {
        let output_cell = Cell::new(CellStyle::Output, output.clone());
        cells.push(CellEntry::Group(CellGroup {
          cells: vec![cell, output_cell],
          open: true,
        }));
        i += 1;
        continue;
      }

      cells.push(CellEntry::Single(cell));
      i += 1;
    }

    self.notebook.cells = cells;
  }

  fn update(&mut self, message: Message) -> Task<Message> {
    match message {
      Message::CloseRequested(id) => {
        if !self.is_dirty {
          return iced::window::close(id);
        }
        Task::perform(
          async {
            rfd::AsyncMessageDialog::new()
              .set_title("Unsaved Changes")
              .set_description(
                "You have unsaved changes. Do you want to save before closing?",
              )
              .set_buttons(rfd::MessageButtons::OkCancelCustom(
                "Save".to_string(),
                "Don't Save".to_string(),
              ))
              .show()
              .await
          },
          move |result| Message::CloseConfirmed(id, result),
        )
      }

      Message::CloseConfirmed(id, result) => match result {
        rfd::MessageDialogResult::Custom(label) if label == "Don't Save" => {
          iced::window::close(id)
        }
        rfd::MessageDialogResult::Custom(label) if label == "Save" => {
          self.sync_notebook_from_editors();
          let content = self.notebook.to_string();
          let path = self.file_path.clone();
          self.is_loading = true;
          self.status = String::from("Saving...");
          Task::perform(save_file(path, content), move |result| {
            Message::FileSavedThenClose(id, result)
          })
        }
        _ => Task::none(),
      },

      Message::FileSavedThenClose(id, result) => {
        self.is_loading = false;
        match result {
          Ok(path) => {
            self.status = format!("Saved: {}", path.display());
            save_last_file_path(&path);
            self.file_path = Some(path);
            self.is_dirty = false;
            iced::window::close(id)
          }
          Err(FileError::DialogClosed) => {
            self.status = String::from("Save cancelled");
            Task::none()
          }
          Err(FileError::IoError(e)) => {
            self.status = format!("Error saving: {e:?}");
            Task::none()
          }
        }
      }

      Message::NewNotebook => {
        if let Ok(exe) = std::env::current_exe() {
          let _ = std::process::Command::new(exe)
            .arg("--new")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();
        }
        Task::none()
      }

      Message::OpenFile => {
        if self.is_loading {
          return Task::none();
        }
        self.is_loading = true;
        self.status = String::from("Opening file...");
        Task::perform(open_file(), Message::FileOpened)
      }

      #[cfg(target_os = "macos")]
      Message::PollPendingOpens => {
        if self.is_loading {
          return Task::none();
        }
        let Some(path) = macos_open::take_pending().pop() else {
          return Task::none();
        };
        self.is_loading = true;
        self.status = format!("Opening: {}", path.display());
        Task::perform(open_file_path(path), Message::FileOpened)
      }

      Message::FileOpened(result) => {
        self.is_loading = false;
        match result {
          Ok((path, contents)) => match notebook::parse_notebook(&contents) {
            Ok(nb) => {
              self.stop_playback();
              self.status = format!("Opened: {}", path.display());
              save_last_file_path(&path);
              // Install the notebook's environment (directory, file name,
              // theme) before building the editors: loading may instantiate
              // stored Manipulate widgets, which evaluate their bodies.
              if let Some(dir) = path.parent() {
                woxi::set_notebook_directory(Some(
                  dir.to_string_lossy().into_owned(),
                ));
              }
              woxi::set_system_variable(
                "$InputFileName",
                &format!("\"{}\"", path.to_string_lossy()),
              );
              woxi::set_dark_mode(!matches!(self.theme, Theme::Light));
              self.cell_editors = Self::editors_from_notebook(&nb);
              // Stored graphics (snapshots decoded from the .nb) rasterize
              // at load; the scale-change handler re-rasterizes them like
              // any evaluated graphic.
              for editor in &mut self.cell_editors {
                if editor.stored_graphic
                  && let Some(ref svg) = editor.graphics_svg
                {
                  editor.graphics_image =
                    rasterize_svg(svg, self.scale_factor, &self.fontdb);
                }
              }
              self.notebook = nb;
              self.file_path = Some(path);
              self.is_dirty = false;
              self.show_toc = self
                .cell_editors
                .iter()
                .any(|e| heading_level(e.style).is_some());
              self.focused_cell = if self.cell_editors.is_empty() {
                None
              } else {
                Some(0)
              };
            }
            Err(e) => {
              self.status = format!("Parse error: {e}");
            }
          },
          Err(FileError::DialogClosed) => {
            self.status = String::from("Open cancelled");
          }
          Err(FileError::IoError(e)) => {
            self.status = format!("Error opening file: {e:?}");
          }
        }
        Task::none()
      }

      Message::SaveFile => {
        if self.is_loading {
          return Task::none();
        }
        self.sync_notebook_from_editors();
        let content = self.notebook.to_string();
        self.is_loading = true;
        self.status = String::from("Saving...");
        Task::perform(
          save_file(self.file_path.clone(), content),
          Message::FileSaved,
        )
      }

      Message::SaveFileAs => {
        if self.is_loading {
          return Task::none();
        }
        self.sync_notebook_from_editors();
        let content = self.notebook.to_string();
        self.is_loading = true;
        self.status = String::from("Saving as...");
        Task::perform(save_file(None, content), Message::FileSaved)
      }

      Message::FileSaved(result) => {
        self.is_loading = false;
        match result {
          Ok(path) => {
            self.status = format!("Saved: {}", path.display());
            save_last_file_path(&path);
            self.file_path = Some(path);
            self.is_dirty = false;
          }
          Err(FileError::DialogClosed) => {
            self.status = String::from("Save cancelled");
          }
          Err(FileError::IoError(e)) => {
            self.status = format!("Error saving: {e:?}");
          }
        }
        Task::none()
      }

      Message::ExportAs(format) => {
        if self.is_loading {
          return Task::none();
        }
        self.sync_notebook_from_editors();
        if format == ExportFormat::Pdf {
          let default_path =
            self.file_path.as_ref().map(|p| p.with_extension("pdf"));
          let cells: Vec<PdfCell> = self
            .cell_editors
            .iter()
            .map(|editor| PdfCell {
              style: editor.style,
              text: editor.content.text(),
              output: editor.output.clone(),
              stdout: editor.stdout.clone(),
              graphics_svg: editor.graphics_svg.clone(),
            })
            .collect();
          self.is_loading = true;
          self.status = String::from("Exporting as PDF...");
          Task::perform(export_pdf(default_path, cells), Message::FileExported)
        } else {
          let (content, filter_name, extension) = match format {
            ExportFormat::MathematicaNotebook => (
              self.notebook.to_string(),
              String::from("Mathematica Notebook"),
              String::from("nb"),
            ),
            ExportFormat::JupyterNotebook => (
              self.notebook.to_jupyter(),
              String::from("Jupyter Notebook"),
              String::from("ipynb"),
            ),
            ExportFormat::Markdown => (
              self.notebook.to_markdown(),
              String::from("Markdown"),
              String::from("md"),
            ),
            ExportFormat::LaTeX => (
              self.notebook.to_latex(),
              String::from("LaTeX"),
              String::from("tex"),
            ),
            ExportFormat::Typst => (
              self.notebook.to_typst(),
              String::from("Typst"),
              String::from("typ"),
            ),
            ExportFormat::Pdf => unreachable!(),
          };
          self.is_loading = true;
          self.status = format!("Exporting as {format}...");
          Task::perform(
            export_file(filter_name, extension, content),
            Message::FileExported,
          )
        }
      }

      Message::FileExported(result) => {
        self.is_loading = false;
        match result {
          Ok(path) => {
            self.status = format!("Exported: {}", path.display());
          }
          Err(FileError::DialogClosed) => {
            self.status = String::from("Export cancelled");
          }
          Err(FileError::IoError(e)) => {
            self.status = format!("Error exporting: {e:?}");
          }
        }
        Task::none()
      }

      Message::CellAction(idx, action) => {
        if idx < self.cell_editors.len() {
          self.focused_cell = Some(idx);
          self.focused_divider = None;
          let is_edit = action.is_edit();
          if is_edit {
            // Snapshot current text for undo
            let snap = self.cell_editors[idx].content.text();
            self.cell_editors[idx].undo_stack.push(snap);
            self.cell_editors[idx].redo_stack.clear();
          }
          self.cell_editors[idx].content.perform(action);
          if is_edit {
            self.is_dirty = true;
            self.cell_editors[idx].output_stale = true;
          }
        }
        Task::none()
      }

      Message::OutputAction(idx, action) => {
        if idx < self.cell_editors.len() && !action.is_edit() {
          self.cell_editors[idx].output_content.perform(action);
        }
        Task::none()
      }

      Message::StdoutAction(idx, action) => {
        if idx < self.cell_editors.len() && !action.is_edit() {
          self.cell_editors[idx].stdout_content.perform(action);
        }
        Task::none()
      }

      Message::WrapSelection(idx, open, close) => {
        if idx < self.cell_editors.len()
          && let Some(sel) = self.cell_editors[idx].content.selection()
        {
          // Snapshot for undo
          let snap = self.cell_editors[idx].content.text();
          self.cell_editors[idx].undo_stack.push(snap);
          self.cell_editors[idx].redo_stack.clear();
          // Insert open char (replaces the selection)
          self.cell_editors[idx]
            .content
            .perform(text_editor::Action::Edit(text_editor::Edit::Insert(
              open,
            )));
          // Insert the original selected text back
          for c in sel.chars() {
            self.cell_editors[idx]
              .content
              .perform(text_editor::Action::Edit(text_editor::Edit::Insert(c)));
          }
          // Insert close char
          self.cell_editors[idx]
            .content
            .perform(text_editor::Action::Edit(text_editor::Edit::Insert(
              close,
            )));
          self.is_dirty = true;
          self.cell_editors[idx].output_stale = true;
        }
        Task::none()
      }

      Message::Undo(idx) => {
        if idx < self.cell_editors.len()
          && let Some(prev) = self.cell_editors[idx].undo_stack.pop()
        {
          let current = self.cell_editors[idx].content.text();
          self.cell_editors[idx].redo_stack.push(current);
          self.cell_editors[idx].content =
            text_editor::Content::with_text(&prev);
          self.is_dirty = true;
          self.cell_editors[idx].output_stale = true;
        }
        Task::none()
      }

      Message::Redo(idx) => {
        if idx < self.cell_editors.len()
          && let Some(next) = self.cell_editors[idx].redo_stack.pop()
        {
          let current = self.cell_editors[idx].content.text();
          self.cell_editors[idx].undo_stack.push(current);
          self.cell_editors[idx].content =
            text_editor::Content::with_text(&next);
          self.is_dirty = true;
          self.cell_editors[idx].output_stale = true;
        }
        Task::none()
      }

      Message::IndentLines(idx) => {
        if idx < self.cell_editors.len() {
          let snap = self.cell_editors[idx].content.text();
          let cursor = self.cell_editors[idx].content.cursor().position;
          let selection = self.cell_editors[idx].content.selection();

          if let Some(sel_text) = selection {
            let lines: Vec<&str> = snap.lines().collect();
            let (start_line, end_line) =
              selection_line_range(cursor.line, &sel_text, lines.len());
            let (anchor, cursor_end) = selection_endpoints(
              cursor.line,
              cursor.column,
              &sel_text,
              &lines,
            );

            let new_text: String = lines
              .iter()
              .enumerate()
              .map(|(i, line)| {
                if i >= start_line && i <= end_line {
                  format!("  {line}")
                } else {
                  line.to_string()
                }
              })
              .collect::<Vec<_>>()
              .join("\n");
            let new_text = preserve_trailing_newline(&snap, new_text);
            self.cell_editors[idx].undo_stack.push(snap);
            self.cell_editors[idx].redo_stack.clear();
            self.cell_editors[idx].content =
              text_editor::Content::with_text(&new_text);
            // Restore selection with columns shifted by 2
            restore_selection(
              &mut self.cell_editors[idx].content,
              (anchor.0, anchor.1 + 2),
              (cursor_end.0, cursor_end.1 + 2),
            );
            self.is_dirty = true;
            self.cell_editors[idx].output_stale = true;
          } else {
            // No selection: insert 2 spaces at cursor position
            self.cell_editors[idx].undo_stack.push(snap);
            self.cell_editors[idx].redo_stack.clear();
            self.cell_editors[idx]
              .content
              .perform(text_editor::Action::Edit(text_editor::Edit::Insert(
                ' ',
              )));
            self.cell_editors[idx]
              .content
              .perform(text_editor::Action::Edit(text_editor::Edit::Insert(
                ' ',
              )));
            self.is_dirty = true;
            self.cell_editors[idx].output_stale = true;
          }
        }
        Task::none()
      }

      Message::UnindentLines(idx) => {
        if idx < self.cell_editors.len() {
          let snap = self.cell_editors[idx].content.text();
          let cursor = self.cell_editors[idx].content.cursor().position;
          let selection = self.cell_editors[idx].content.selection();
          let has_selection = selection.is_some();

          let lines: Vec<&str> = snap.lines().collect();
          let (start_line, end_line) = if let Some(sel_text) = &selection {
            selection_line_range(cursor.line, sel_text, lines.len())
          } else {
            (cursor.line, cursor.line)
          };

          // Compute how many spaces each line will lose
          let removed: Vec<usize> = lines
            .iter()
            .enumerate()
            .map(|(i, line)| {
              if i >= start_line && i <= end_line {
                if line.starts_with("  ") {
                  2
                } else if line.starts_with(' ') {
                  1
                } else {
                  0
                }
              } else {
                0
              }
            })
            .collect();

          let (anchor, cursor_end) = if let Some(sel_text) = &selection {
            selection_endpoints(cursor.line, cursor.column, sel_text, &lines)
          } else {
            ((cursor.line, cursor.column), (cursor.line, cursor.column))
          };

          let new_text: String = lines
            .iter()
            .enumerate()
            .map(|(i, line)| line[removed[i]..].to_string())
            .collect::<Vec<_>>()
            .join("\n");
          let new_text = preserve_trailing_newline(&snap, new_text);

          if new_text != snap {
            self.cell_editors[idx].undo_stack.push(snap);
            self.cell_editors[idx].redo_stack.clear();
            self.cell_editors[idx].content =
              text_editor::Content::with_text(&new_text);
            if has_selection {
              restore_selection(
                &mut self.cell_editors[idx].content,
                (anchor.0, anchor.1.saturating_sub(removed[anchor.0])),
                (
                  cursor_end.0,
                  cursor_end.1.saturating_sub(removed[cursor_end.0]),
                ),
              );
            }
            self.is_dirty = true;
            self.cell_editors[idx].output_stale = true;
          }
        }
        Task::none()
      }

      Message::ToggleComment(idx) => {
        if idx < self.cell_editors.len() {
          let snap = self.cell_editors[idx].content.text();
          let cursor = self.cell_editors[idx].content.cursor().position;
          let selection = self.cell_editors[idx].content.selection();

          if let Some(sel_text) = selection {
            let lines: Vec<&str> = snap.lines().collect();
            let (anchor, cursor_end) = selection_endpoints(
              cursor.line,
              cursor.column,
              &sel_text,
              &lines,
            );
            let sel_newlines = sel_text.chars().filter(|c| *c == '\n').count();

            if sel_newlines > 0 {
              // Multi-line selection: comment/uncomment whole lines
              let (start_line, end_line) =
                selection_line_range(cursor.line, &sel_text, lines.len());

              let all_commented = (start_line..=end_line).all(|i| {
                let trimmed = lines[i].trim();
                trimmed.starts_with("(*") && trimmed.ends_with("*)")
              });

              let new_text: String = lines
                .iter()
                .enumerate()
                .map(|(i, line)| {
                  if i >= start_line && i <= end_line {
                    if all_commented {
                      let trimmed = line.trim();
                      let leading_ws =
                        &line[..line.len() - line.trim_start().len()];
                      let inner = trimmed.strip_prefix("(*").unwrap_or(trimmed);
                      let inner = inner.strip_prefix(' ').unwrap_or(inner);
                      let inner = inner.strip_suffix("*)").unwrap_or(inner);
                      let inner = inner.strip_suffix(' ').unwrap_or(inner);
                      format!("{leading_ws}{inner}")
                    } else {
                      let leading_ws =
                        &line[..line.len() - line.trim_start().len()];
                      let content = line.trim_start();
                      format!("{leading_ws}(* {content} *)")
                    }
                  } else {
                    line.to_string()
                  }
                })
                .collect::<Vec<_>>()
                .join("\n");
              let new_text = preserve_trailing_newline(&snap, new_text);

              if new_text != snap {
                let old_anchor_len = lines[anchor.0].len();
                let old_cursor_len = lines[cursor_end.0].len();
                self.cell_editors[idx].undo_stack.push(snap);
                self.cell_editors[idx].redo_stack.clear();
                self.cell_editors[idx].content =
                  text_editor::Content::with_text(&new_text);
                let new_lines: Vec<&str> = new_text.lines().collect();
                let anchor_shift = new_lines
                  .get(anchor.0)
                  .map(|l| l.len() as isize - old_anchor_len as isize)
                  .unwrap_or(0);
                let cursor_shift = new_lines
                  .get(cursor_end.0)
                  .map(|l| l.len() as isize - old_cursor_len as isize)
                  .unwrap_or(0);
                restore_selection(
                  &mut self.cell_editors[idx].content,
                  (
                    anchor.0,
                    (anchor.1 as isize + anchor_shift).max(0) as usize,
                  ),
                  (
                    cursor_end.0,
                    (cursor_end.1 as isize + cursor_shift).max(0) as usize,
                  ),
                );
                self.is_dirty = true;
                self.cell_editors[idx].output_stale = true;
              }
            } else {
              // Single-line selection: wrap/unwrap only the selected text
              let (start, end) = if anchor.1 <= cursor_end.1 {
                (anchor, cursor_end)
              } else {
                (cursor_end, anchor)
              };

              let is_commented =
                sel_text.starts_with("(* ") && sel_text.ends_with(" *)");

              // Compute byte offset of selection start
              let mut byte_offset = 0;
              for (i, line) in lines.iter().enumerate() {
                if i == start.0 {
                  byte_offset += start.1;
                  break;
                }
                byte_offset += line.len() + 1;
              }

              let new_text = if is_commented {
                let before = &snap[..byte_offset];
                let after = &snap[byte_offset + sel_text.len()..];
                let inner = &sel_text[3..sel_text.len() - 3];
                format!("{before}{inner}{after}")
              } else {
                let before = &snap[..byte_offset];
                let after = &snap[byte_offset + sel_text.len()..];
                format!("{before}(* {sel_text} *){after}")
              };

              self.cell_editors[idx].undo_stack.push(snap);
              self.cell_editors[idx].redo_stack.clear();
              self.cell_editors[idx].content =
                text_editor::Content::with_text(&new_text);

              let new_end_col = (end.1 as isize
                + if is_commented { -6 } else { 6 })
              .max(0) as usize;
              for _ in 0..end.0 {
                self.cell_editors[idx].content.perform(
                  text_editor::Action::Move(text_editor::Motion::Down),
                );
              }
              self.cell_editors[idx]
                .content
                .perform(text_editor::Action::Move(text_editor::Motion::Home));
              for _ in 0..new_end_col {
                self.cell_editors[idx].content.perform(
                  text_editor::Action::Move(text_editor::Motion::Right),
                );
              }
              self.is_dirty = true;
              self.cell_editors[idx].output_stale = true;
            }
          } else {
            // No selection: toggle comment on the current line.
            // `str::lines()` drops a trailing empty line, so the cursor can
            // legitimately be on a line past the end of `lines`. Treat any
            // such position as an empty line rather than panicking.
            let mut lines: Vec<String> =
              snap.lines().map(|s| s.to_string()).collect();
            while lines.len() <= cursor.line {
              lines.push(String::new());
            }
            let (new_line, col_shift) =
              toggle_line_comment(&lines[cursor.line]);

            let new_text: String = lines
              .iter()
              .enumerate()
              .map(|(i, l)| {
                if i == cursor.line {
                  new_line.clone()
                } else {
                  l.to_string()
                }
              })
              .collect::<Vec<_>>()
              .join("\n");
            let new_text = preserve_trailing_newline(&snap, new_text);

            if new_text != snap {
              self.cell_editors[idx].undo_stack.push(snap);
              self.cell_editors[idx].redo_stack.clear();
              self.cell_editors[idx].content =
                text_editor::Content::with_text(&new_text);
              let new_col =
                (cursor.column as isize + col_shift).max(0) as usize;
              for _ in 0..cursor.line {
                self.cell_editors[idx].content.perform(
                  text_editor::Action::Move(text_editor::Motion::Down),
                );
              }
              self.cell_editors[idx]
                .content
                .perform(text_editor::Action::Move(text_editor::Motion::Home));
              for _ in 0..new_col {
                self.cell_editors[idx].content.perform(
                  text_editor::Action::Move(text_editor::Motion::Right),
                );
              }
              self.is_dirty = true;
              self.cell_editors[idx].output_stale = true;
            }
          }
        }
        Task::none()
      }

      Message::CellStyleChanged(idx, style) => {
        if idx < self.cell_editors.len() {
          self.cell_editors[idx].style = style;
          if style != CellStyle::Input {
            self.cell_editors[idx].output = None;
            self.cell_editors[idx].output_content = text_editor::Content::new();
            self.cell_editors[idx].stdout = None;
            self.cell_editors[idx].stdout_content = text_editor::Content::new();
            self.cell_editors[idx].graphics_svg = None;
            self.cell_editors[idx].graphics_handle = None;
            self.cell_editors[idx].graphics_image = None;
            self.cell_editors[idx].output_svgs.clear();
            self.cell_editors[idx].output_images.clear();
            self.cell_editors[idx].output_all_svg = false;
            self.cell_editors[idx].sound = None;
            self.cell_editors[idx].hyperlinks.clear();
            self.cell_editors[idx].warnings.clear();
            self.cell_editors[idx].output_stale = false;
          }
          self.is_dirty = true;
        }
        self.cell_type_menu_open = None;
        Task::none()
      }

      Message::ToggleCollapse(idx) => {
        if idx < self.cell_editors.len() {
          self.cell_editors[idx].is_collapsed =
            !self.cell_editors[idx].is_collapsed;
        }
        Task::none()
      }

      Message::ToggleCellTypeMenu(idx) => {
        if self.cell_type_menu_open == Some(idx) {
          self.cell_type_menu_open = None;
        } else {
          self.cell_type_menu_open = Some(idx);
        }
        Task::none()
      }

      Message::GutterEnter(idx) => {
        if self.dragging_cell.is_some() {
          // During drag, entering a cell updates the drop target
          return self.update(Message::DragOverCell(idx));
        }
        self.hovered_gutter = Some(idx);
        Task::none()
      }

      Message::GutterExit(idx) => {
        if self.hovered_gutter == Some(idx) && self.dragging_cell.is_none() {
          self.hovered_gutter = None;
        }
        Task::none()
      }

      Message::DragStart(idx) => {
        self.dragging_cell = Some(idx);
        self.drop_target = None;
        Task::none()
      }

      Message::DragOverCell(idx) => {
        if let Some(src) = self.dragging_cell {
          // Dropping at src or src+1 would leave the cell in the same place
          if idx != src && idx != src + 1 {
            self.drop_target = Some(idx);
          } else {
            self.drop_target = None;
          }
        }
        Task::none()
      }

      Message::DragEnd => {
        if let (Some(src), Some(dst)) = (self.dragging_cell, self.drop_target)
          && src != dst
          && dst <= self.cell_editors.len()
        {
          let cell = self.cell_editors.remove(src);
          let insert_at = if dst > src { dst - 1 } else { dst };
          let insert_at = insert_at.min(self.cell_editors.len());
          self.cell_editors.insert(insert_at, cell);
          self.focused_cell = Some(insert_at);
          self.is_dirty = true;
          if let Some(p) = &mut self.playback {
            if p.cell == src {
              p.cell = insert_at;
            } else {
              if p.cell > src {
                p.cell -= 1;
              }
              if p.cell >= insert_at {
                p.cell += 1;
              }
            }
          }
        }
        self.dragging_cell = None;
        self.drop_target = None;
        self.hovered_gutter = None;
        Task::none()
      }

      Message::TogglePreview => {
        self.preview_mode = !self.preview_mode;
        Task::none()
      }

      Message::ToggleToc => {
        self.show_toc = !self.show_toc;
        Task::none()
      }

      Message::ScrollToCell(idx) => {
        if idx < self.cell_editors.len() {
          self.focused_cell = Some(idx);
          self.focused_divider = None;
          self.cell_type_menu_open = None;
          let scroll_task = scroll_cell_into_view(
            iced::widget::Id::from("cells-scroll"),
            iced::widget::Id::from(format!("cell-{idx}")),
          );
          let focus_task = focus(iced::widget::Id::from(format!("cell-{idx}")));
          return Task::batch([scroll_task, focus_task]);
        }
        Task::none()
      }

      Message::OpenGraphicsModal(idx) => {
        if idx < self.cell_editors.len()
          && self.cell_editors[idx].graphics_svg.is_some()
        {
          self.graphics_modal_cell = Some(idx);
        }
        Task::none()
      }

      Message::CloseGraphicsModal => {
        self.graphics_modal_cell = None;
        self.graphics_context_menu_cell = None;
        Task::none()
      }

      Message::ShowGraphicsContextMenu(idx) => {
        if idx < self.cell_editors.len()
          && self.cell_editors[idx].graphics_svg.is_some()
        {
          self.graphics_context_menu_cell = Some(idx);
          self.graphics_context_menu_pos = self.cursor_position;
        }
        Task::none()
      }

      Message::CloseGraphicsContextMenu => {
        self.graphics_context_menu_cell = None;
        Task::none()
      }

      Message::CursorMoved(pos) => {
        self.cursor_position = pos;
        Task::none()
      }

      Message::SaveGraphicAs(idx) => {
        self.graphics_context_menu_cell = None;
        if idx >= self.cell_editors.len() {
          return Task::none();
        }
        let Some(svg_data) = self.cell_editors[idx].graphics_svg.clone() else {
          return Task::none();
        };
        let default_dir = self
          .file_path
          .as_ref()
          .and_then(|p| p.parent().map(|d| d.to_path_buf()));
        let fontdb = self.fontdb.clone();
        Task::perform(
          save_graphic(svg_data, default_dir, fontdb),
          Message::GraphicSaved,
        )
      }

      Message::GraphicSaved(result) => {
        match result {
          Ok(path) => {
            self.status = format!("Saved graphic: {}", path.display());
          }
          Err(FileError::DialogClosed) => {
            self.status = String::from("Save cancelled");
          }
          Err(FileError::IoError(e)) => {
            self.status = format!("Error saving graphic: {e:?}");
          }
        }
        Task::none()
      }

      Message::WindowResized(size) => {
        self.window_width = size.width;
        Task::none()
      }

      Message::ScaleFactorChanged(scale) => {
        if (scale - self.scale_factor).abs() > f32::EPSILON {
          self.scale_factor = scale;
          // Re-rasterize all existing SVGs at the new scale
          for editor in &mut self.cell_editors {
            editor.graphics_image = editor
              .graphics_svg
              .as_ref()
              .and_then(|s| rasterize_svg(s, scale, &self.fontdb));
            editor.output_images = editor
              .output_svgs
              .iter()
              .filter_map(|s| rasterize_svg(s, scale, &self.fontdb))
              .collect();
            // Manipulate graphics are drawn by the `svg` widget, which
            // rescales for DPI on its own — no manual re-rasterization needed.
          }
        }
        Task::none()
      }

      Message::ManipulateContinuousChanged(cell_idx, ctrl_idx, value) => {
        if let Some(editor) = self.cell_editors.get_mut(cell_idx)
          && let Some(state) = editor.manipulate_state.as_mut()
          && let Some(control) = state.controls.get_mut(ctrl_idx)
          && let manipulate::ControlState::Continuous { current, .. } = control
        {
          *current = value;
          if state.request_reeval() {
            return manipulate_reeval_task(cell_idx);
          }
        }
        Task::none()
      }

      Message::ManipulateDiscreteChanged(cell_idx, ctrl_idx, choice) => {
        if let Some(editor) = self.cell_editors.get_mut(cell_idx)
          && let Some(state) = editor.manipulate_state.as_mut()
          && let Some(control) = state.controls.get_mut(ctrl_idx)
          && let manipulate::ControlState::Discrete {
            value_labels,
            current_index,
            ..
          } = control
          && let Some(idx) = value_labels.iter().position(|v| *v == choice)
        {
          *current_index = idx;
          if state.request_reeval() {
            return manipulate_reeval_task(cell_idx);
          }
        }
        Task::none()
      }

      Message::ManipulateSlider2DChanged(cell_idx, ctrl_idx, axis, value) => {
        if let Some(editor) = self.cell_editors.get_mut(cell_idx)
          && let Some(state) = editor.manipulate_state.as_mut()
        {
          // Routes through the control's write-back callback (if any), so
          // e.g. Locator-promoted controls round/validate the candidate.
          state.slider2d_change(ctrl_idx, axis, value);
          if state.request_reeval() {
            return manipulate_reeval_task(cell_idx);
          }
        }
        Task::none()
      }

      Message::ManipulateIntervalChanged(
        cell_idx,
        ctrl_idx,
        endpoint,
        value,
      ) => {
        if let Some(editor) = self.cell_editors.get_mut(cell_idx)
          && let Some(state) = editor.manipulate_state.as_mut()
          && let Some(control) = state.controls.get_mut(ctrl_idx)
          && let manipulate::ControlState::IntervalSlider { low, high, .. } =
            control
        {
          // Keep the interval ordered: the low thumb can't pass the high
          // thumb and vice versa.
          if endpoint == 0 {
            *low = value.min(*high);
          } else {
            *high = value.max(*low);
          }
          if state.request_reeval() {
            return manipulate_reeval_task(cell_idx);
          }
        }
        Task::none()
      }

      Message::ManipulateLocatorChanged(
        cell_idx,
        ctrl_idx,
        point_idx,
        axis,
        value,
      ) => {
        if let Some(editor) = self.cell_editors.get_mut(cell_idx)
          && let Some(state) = editor.manipulate_state.as_mut()
          && let Some(control) = state.controls.get_mut(ctrl_idx)
          && let manipulate::ControlState::Locator { points, .. } = control
          && let Some(point) = points.get_mut(point_idx)
        {
          if axis == 0 {
            point.0 = value;
          } else {
            point.1 = value;
          }
          if state.request_reeval() {
            return manipulate_reeval_task(cell_idx);
          }
        }
        Task::none()
      }

      Message::ManipulateLocatorAdded(cell_idx, ctrl_idx) => {
        if let Some(editor) = self.cell_editors.get_mut(cell_idx)
          && let Some(state) = editor.manipulate_state.as_mut()
          && let Some(control) = state.controls.get_mut(ctrl_idx)
          && let manipulate::ControlState::Locator {
            points,
            x_min,
            x_max,
            y_min,
            y_max,
            ..
          } = control
        {
          // New points appear at the range centre, ready to drag.
          points.push(((*x_min + *x_max) / 2.0, (*y_min + *y_max) / 2.0));
          if state.request_reeval() {
            return manipulate_reeval_task(cell_idx);
          }
        }
        Task::none()
      }

      Message::ManipulateLocatorRemoved(cell_idx, ctrl_idx, point_idx) => {
        if let Some(editor) = self.cell_editors.get_mut(cell_idx)
          && let Some(state) = editor.manipulate_state.as_mut()
          && let Some(control) = state.controls.get_mut(ctrl_idx)
          && let manipulate::ControlState::Locator { points, .. } = control
          && point_idx < points.len()
        {
          points.remove(point_idx);
          if state.request_reeval() {
            return manipulate_reeval_task(cell_idx);
          }
        }
        Task::none()
      }

      Message::ManipulateDisplayToggled(cell_idx, mutation) => {
        if let Some(editor) = self.cell_editors.get_mut(cell_idx)
          && let Some(state) = editor.manipulate_state.as_mut()
        {
          state.apply_display_mutation(&mutation);
        }
        Task::none()
      }

      Message::ManipulateReeval(cell_idx) => {
        if let Some(editor) = self.cell_editors.get_mut(cell_idx)
          && let Some(state) = editor.manipulate_state.as_mut()
        {
          state.run_scheduled_reeval();
        }
        Task::none()
      }

      Message::ManipulateAnimationTick(tick_at) => {
        // Advancing re-evaluates every playing widget synchronously, which
        // can take longer than the tick interval. The timer keeps producing
        // ticks into the runtime's queue meanwhile, so without this check the
        // backlog grows every frame and each stale tick triggers another full
        // re-evaluation — the animation gets slower every cycle until the app
        // freezes. Only a tick generated *after* the previous advance
        // finished advances the animation; backlogged ones are dropped.
        if animation_tick_is_fresh(tick_at, self.last_anim_advance) {
          for editor in &mut self.cell_editors {
            if let Some(state) = editor.manipulate_state.as_mut()
              && state.animated
              && state.playing
            {
              state.advance_animation();
            }
          }
          self.last_anim_advance = Some(std::time::Instant::now());
        }
        Task::none()
      }

      Message::ManipulateTogglePlay(cell_idx) => {
        if let Some(editor) = self.cell_editors.get_mut(cell_idx)
          && let Some(state) = editor.manipulate_state.as_mut()
        {
          state.playing = !state.playing;
        }
        Task::none()
      }

      Message::ManipulateButtonPressed(cell_idx, ctrl_idx) => {
        if let Some(editor) = self.cell_editors.get_mut(cell_idx)
          && let Some(state) = editor.manipulate_state.as_mut()
          && let Some(manipulate::ControlState::Button { action, .. }) =
            state.controls.get(ctrl_idx)
        {
          let action = action.clone();
          state.apply_button_action(&action);
        }
        Task::none()
      }

      Message::Noop => Task::none(),

      Message::OpenHyperlink(url) => {
        open_url(&url);
        Task::none()
      }

      Message::FocusCell(idx) => {
        if idx < self.cell_editors.len() {
          self.focused_cell = Some(idx);
          self.focused_divider = None;
          self.cell_type_menu_open = None;
          return focus(iced::widget::Id::from(format!("cell-{idx}")));
        }
        self.focused_divider = None;
        self.cell_type_menu_open = None;
        Task::none()
      }

      Message::ScrollCellsToEnd => iced::widget::operation::snap_to_end(
        iced::widget::Id::from("cells-scroll"),
      ),

      Message::FocusDividerBelow(idx) => {
        if idx < self.cell_editors.len() {
          self.focused_divider = Some(idx);
          self.focused_cell = None;
        }
        Task::none()
      }

      Message::FocusDividerAbove(idx) => {
        if idx > 0 {
          self.focused_divider = Some(idx - 1);
          self.focused_cell = None;
        }
        Task::none()
      }

      Message::AddCellBelow(idx) => {
        let insert_at = (idx + 1).min(self.cell_editors.len());
        self.cell_editors.insert(
          insert_at,
          CellEditor {
            content: text_editor::Content::new(),
            style: self.new_cell_style,
            output: None,
            stdout: None,
            graphics_svg: None,
            graphics_handle: None,
            graphics_image: None,
            output_svgs: Vec::new(),
            output_images: Vec::new(),
            output_dark: false,
            output_all_svg: false,
            sound: None,
            warnings: Vec::new(),
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            output_stale: false,
            is_collapsed: false,
            manipulate_state: None,
            hyperlinks: Vec::new(),
            stored_graphic: false,
            output_content: text_editor::Content::new(),
            stdout_content: text_editor::Content::new(),
          },
        );
        self.focused_cell = Some(insert_at);
        self.focused_divider = None;
        self.is_dirty = true;
        if let Some(p) = &mut self.playback
          && p.cell >= insert_at
        {
          p.cell += 1;
        }
        focus(iced::widget::Id::from(format!("cell-{insert_at}")))
      }

      Message::AddCellAbove(idx) => {
        let insert_at = idx.min(self.cell_editors.len());
        self.cell_editors.insert(
          insert_at,
          CellEditor {
            content: text_editor::Content::new(),
            style: self.new_cell_style,
            output: None,
            stdout: None,
            graphics_svg: None,
            graphics_handle: None,
            graphics_image: None,
            output_svgs: Vec::new(),
            output_images: Vec::new(),
            output_dark: false,
            output_all_svg: false,
            sound: None,
            warnings: Vec::new(),
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            output_stale: false,
            is_collapsed: false,
            manipulate_state: None,
            hyperlinks: Vec::new(),
            stored_graphic: false,
            output_content: text_editor::Content::new(),
            stdout_content: text_editor::Content::new(),
          },
        );
        self.focused_cell = Some(insert_at);
        self.focused_divider = None;
        self.is_dirty = true;
        if let Some(p) = &mut self.playback
          && p.cell >= insert_at
        {
          p.cell += 1;
        }
        focus(iced::widget::Id::from(format!("cell-{insert_at}")))
      }

      Message::DeleteCell(idx) => {
        if self.cell_editors.len() > 1 && idx < self.cell_editors.len() {
          self.cell_editors.remove(idx);
          self.is_dirty = true;
          if let Some(ref mut focused) = self.focused_cell
            && *focused >= self.cell_editors.len()
          {
            *focused = self.cell_editors.len() - 1;
          }
          match self.playback.as_mut() {
            Some(p) if p.cell == idx => self.stop_playback(),
            Some(p) if p.cell > idx => p.cell -= 1,
            _ => {}
          }
        }
        Task::none()
      }

      Message::MoveCellUp(idx) => {
        if idx > 0 && idx < self.cell_editors.len() {
          self.cell_editors.swap(idx, idx - 1);
          self.focused_cell = Some(idx - 1);
          self.is_dirty = true;
          if let Some(p) = &mut self.playback {
            if p.cell == idx {
              p.cell = idx - 1;
            } else if p.cell == idx - 1 {
              p.cell = idx;
            }
          }
        }
        Task::none()
      }

      Message::MoveCellDown(idx) => {
        if idx + 1 < self.cell_editors.len() {
          self.cell_editors.swap(idx, idx + 1);
          self.focused_cell = Some(idx + 1);
          self.is_dirty = true;
          if let Some(p) = &mut self.playback {
            if p.cell == idx {
              p.cell = idx + 1;
            } else if p.cell == idx + 1 {
              p.cell = idx;
            }
          }
        }
        Task::none()
      }

      Message::EvaluateCell(idx) => {
        if idx < self.cell_editors.len()
          && matches!(
            self.cell_editors[idx].style,
            CellStyle::Input | CellStyle::Code
          )
        {
          let code = self.cell_editors[idx].content.text().trim().to_string();
          if !code.is_empty() {
            // The cell's audio output is about to be replaced — stop any
            // playback of the old audio.
            if self.playback.as_ref().is_some_and(|p| p.cell == idx) {
              self.stop_playback();
            }
            // Clear state and silently re-evaluate all preceding cells
            // so their side effects (variable assignments, function
            // definitions, etc.) are available in the current cell.
            woxi::clear_state();
            for prev in 0..idx {
              if matches!(
                self.cell_editors[prev].style,
                CellStyle::Input | CellStyle::Code
              ) {
                let prev_code =
                  self.cell_editors[prev].content.text().trim().to_string();
                if !prev_code.is_empty() {
                  let _ = woxi::interpret_with_stdout(&prev_code);
                }
              }
            }
            let is_dark = !matches!(self.theme, Theme::Light);
            evaluate_cell_statements(
              &mut self.cell_editors[idx],
              &code,
              is_dark,
              self.scale_factor,
              &self.fontdb,
            );
            self.status = format!("Evaluated cell {} successfully", idx + 1);
          }
        }
        Task::none()
      }

      Message::PlaySound(idx) => {
        // If this cell's audio is already playing (or paused), toggle
        // pause/resume instead of starting playback again.
        let toggling = self.playback.as_mut().is_some_and(|p| {
          p.cell == idx && matches!(p.child.try_wait(), Ok(None))
        });
        if toggling {
          #[cfg(unix)]
          {
            let playback = self.playback.as_mut().unwrap();
            let signal = if playback.paused { "-CONT" } else { "-STOP" };
            match signal_playback(&playback.child, signal) {
              Ok(()) => {
                playback.paused = !playback.paused;
                self.status = if playback.paused {
                  String::from("Sound paused")
                } else {
                  String::from("Playing sound…")
                };
              }
              Err(e) => self.status = format!("Could not pause sound: {e}"),
            }
          }
          #[cfg(not(unix))]
          {
            // No way to pause an external player here without extra
            // dependencies — stop instead; play restarts from the top.
            self.stop_playback();
            self.status = String::from("Sound stopped");
          }
          return Task::none();
        }

        self.stop_playback();
        if let Some(editor) = self.cell_editors.get(idx)
          && let Some(audio) = editor.sound.clone()
        {
          match play_audio(&audio) {
            Ok(child) => {
              self.playback = Some(Playback {
                cell: idx,
                child,
                paused: false,
              });
              self.status = String::from("Playing sound…");
            }
            Err(e) => self.status = format!("Could not play sound: {e}"),
          }
        }
        Task::none()
      }

      Message::PlaybackTick => {
        // Revert the pause button to a play button once the external
        // player exits (playback finished on its own or was killed).
        if self
          .playback
          .as_mut()
          .is_some_and(|p| !matches!(p.child.try_wait(), Ok(None)))
        {
          self.playback = None;
        }
        Task::none()
      }

      Message::EvaluateAll => {
        self.stop_playback();
        woxi::clear_state();
        for idx in 0..self.cell_editors.len() {
          if matches!(
            self.cell_editors[idx].style,
            CellStyle::Input | CellStyle::Code
          ) {
            let code = self.cell_editors[idx].content.text().trim().to_string();
            if !code.is_empty() {
              let is_dark = !matches!(self.theme, Theme::Light);
              evaluate_cell_statements(
                &mut self.cell_editors[idx],
                &code,
                is_dark,
                self.scale_factor,
                &self.fontdb,
              );
            }
          }
        }
        self.status = String::from("All cells evaluated");
        Task::none()
      }

      Message::ThemeChanged(choice) => {
        self.theme_choice = choice;
        self.theme = match choice {
          ThemeChoice::Auto => detect_system_theme(),
          ThemeChoice::Light => Theme::Light,
          ThemeChoice::Dark => Theme::Dark,
        };
        Task::none()
      }

      Message::NewCellStyleChanged(style) => {
        self.new_cell_style = style;
        Task::none()
      }

      Message::KeyPressed(key, modifiers) => {
        // Escape closes the graphics context menu, then the modal
        if matches!(
          key.as_ref(),
          keyboard::Key::Named(keyboard::key::Named::Escape)
        ) {
          if self.graphics_context_menu_cell.is_some() {
            self.graphics_context_menu_cell = None;
            return Task::none();
          }
          if self.graphics_modal_cell.is_some() {
            self.graphics_modal_cell = None;
            return Task::none();
          }
        }

        if modifiers.command() {
          match key.as_ref() {
            keyboard::Key::Character("s") => {
              if modifiers.shift() {
                return self.update(Message::SaveFileAs);
              }
              return self.update(Message::SaveFile);
            }
            keyboard::Key::Character("o") => {
              return self.update(Message::OpenFile);
            }
            keyboard::Key::Character("n") => {
              return self.update(Message::NewNotebook);
            }
            _ => {}
          }
        }

        // Ctrl+D: delete forward
        if modifiers.control()
          && let keyboard::Key::Character("d") = key.as_ref()
        {
          if let Some(idx) = self.focused_cell {
            self.cell_editors[idx]
              .content
              .perform(text_editor::Action::Edit(text_editor::Edit::Delete));
            self.is_dirty = true;
          }
          return Task::none();
        }

        // Ctrl+A: move cursor to start of line
        if modifiers.control()
          && let keyboard::Key::Character("a") = key.as_ref()
        {
          if let Some(idx) = self.focused_cell {
            self.cell_editors[idx]
              .content
              .perform(text_editor::Action::Move(text_editor::Motion::Home));
          }
          return Task::none();
        }

        // Ctrl+E: move cursor to end of line
        if modifiers.control()
          && let keyboard::Key::Character("e") = key.as_ref()
        {
          if let Some(idx) = self.focused_cell {
            self.cell_editors[idx]
              .content
              .perform(text_editor::Action::Move(text_editor::Motion::End));
          }
          return Task::none();
        }

        // Ctrl+W: delete previous word
        if modifiers.control()
          && let keyboard::Key::Character("w") = key.as_ref()
        {
          if let Some(idx) = self.focused_cell {
            self.cell_editors[idx].content.perform(
              text_editor::Action::Select(text_editor::Motion::WordLeft),
            );
            self.cell_editors[idx]
              .content
              .perform(text_editor::Action::Edit(text_editor::Edit::Backspace));
            self.is_dirty = true;
          }
          return Task::none();
        }

        // Divider navigation (when a "+" divider is focused)
        if let Some(div_idx) = self.focused_divider {
          let no_mods =
            !modifiers.shift() && !modifiers.command() && !modifiers.control();
          if no_mods {
            match key.as_ref() {
              keyboard::Key::Named(keyboard::key::Named::ArrowDown) => {
                let next_cell = div_idx + 1;
                if next_cell < self.cell_editors.len() {
                  self.focused_divider = None;
                  self.focused_cell = Some(next_cell);
                  return focus(iced::widget::Id::from(format!(
                    "cell-{next_cell}"
                  )));
                }
              }
              keyboard::Key::Named(keyboard::key::Named::ArrowUp) => {
                self.focused_divider = None;
                self.focused_cell = Some(div_idx);
                return focus(iced::widget::Id::from(format!(
                  "cell-{div_idx}"
                )));
              }
              keyboard::Key::Named(keyboard::key::Named::Enter) => {
                self.focused_divider = None;
                return self.update(Message::AddCellBelow(div_idx));
              }
              _ => {}
            }
          }
        }

        Task::none()
      }
    }
  }

  fn subscription(&self) -> Subscription<Message> {
    // Use event::listen_with instead of keyboard::on_key_press
    // because on_key_press only fires for Status::Ignored events,
    // which means it never fires when a text_editor has focus.
    let events = iced::event::listen_with(handle_event);
    let mut subs = vec![events];
    // While audio is playing, poll the player process so the pause
    // button reverts to a play button when playback finishes. A paused
    // (SIGSTOP'd) process cannot exit, so no polling is needed then.
    if self.playback.as_ref().is_some_and(|p| !p.paused) {
      subs.push(
        iced::time::every(std::time::Duration::from_millis(200))
          .map(|_| Message::PlaybackTick),
      );
    }
    // While any Animate/ListAnimate widget is playing, tick its animation
    // forward. One shared timer drives all playing widgets.
    if self.cell_editors.iter().any(|e| {
      e.manipulate_state
        .as_ref()
        .is_some_and(|s| s.animated && s.playing)
    }) {
      subs.push(
        iced::time::every(std::time::Duration::from_millis(ANIM_INTERVAL_MS))
          .map(Message::ManipulateAnimationTick),
      );
    }
    #[cfg(target_os = "macos")]
    subs.push(
      iced::time::every(std::time::Duration::from_millis(150))
        .map(|_| Message::PollPendingOpens),
    );
    Subscription::batch(subs)
  }

  fn view(&self) -> Element<'_, Message> {
    // ── Toolbar ──
    let eval_all_svg =
      svg::Handle::from_memory(PLAY_CIRCLE_SVG.as_bytes().to_vec());
    let toolbar = row![
      button(
        svg::Svg::new(eval_all_svg)
          .width(24)
          .height(24)
          .style(eval_all_icon_style),
      )
      .on_press(Message::EvaluateAll)
      .padding([2, 6])
      .style(trash_button_style),
      text(" | ").size(11),
      button(text("New").size(11))
        .on_press(Message::NewNotebook)
        .padding([3, 8])
        .style(muted_button_style),
      button(text("Open").size(11))
        .on_press_maybe((!self.is_loading).then_some(Message::OpenFile))
        .padding([3, 8])
        .style(muted_button_style),
      button(text("Save").size(11))
        .on_press_maybe(self.is_dirty.then_some(Message::SaveFile))
        .padding([3, 8])
        .style(muted_button_style),
      button(text("Save As").size(11))
        .on_press(Message::SaveFileAs)
        .padding([3, 8])
        .style(muted_button_style),
      pick_list(ExportFormat::ALL, None::<ExportFormat>, Message::ExportAs,)
        .placeholder("Export")
        .text_size(11)
        .padding([3, 8])
        .style(export_button_style)
        .menu_style(dropdown_menu_style),
      text(" | ").size(11),
      pick_list(
        ThemeChoice::ALL,
        Some(self.theme_choice),
        Message::ThemeChanged,
      )
      .text_size(11)
      .padding([3, 8])
      .style(dropdown_style)
      .menu_style(dropdown_menu_style),
      space::horizontal(),
      button(
        svg::Svg::new(svg::Handle::from_memory(ICON_TOC.as_bytes().to_vec(),))
          .width(16)
          .height(16)
          .style(gutter_icon_style),
      )
      .on_press(Message::ToggleToc)
      .padding([3, 6])
      .style(trash_button_style),
      button(
        svg::Svg::new(svg::Handle::from_memory(
          if self.preview_mode {
            ICON_EYE_OFF
          } else {
            ICON_EYE
          }
          .as_bytes()
          .to_vec(),
        ))
        .width(16)
        .height(16)
        .style(gutter_icon_style),
      )
      .on_press(Message::TogglePreview)
      .padding([3, 6])
      .style(trash_button_style),
    ]
    .spacing(4)
    .padding(6)
    .align_y(Center);

    // ── Cell editors ──
    let cells: Element<'_, Message> = if self.cell_editors.is_empty() {
      container(text("Empty notebook. Click '+' to add a cell.").size(13))
        .center_x(Fill)
        .padding(40)
        .into()
    } else {
      let mut col = Column::new().spacing(0).width(Fill);

      let hidden = self.compute_hidden_cells();

      let is_dragging = self.dragging_cell.is_some();

      if !self.preview_mode {
        // Add cell divider above the first cell
        let divider = self.view_add_cell_divider_above(0);
        if is_dragging {
          col = col.push(
            mouse_area(divider)
              .on_enter(Message::DragOverCell(0))
              .interaction(iced::mouse::Interaction::Grabbing),
          );
        } else {
          col = col.push(divider);
        }
      }

      let mut visible_count = 0usize;
      for (idx, editor) in self.cell_editors.iter().enumerate() {
        if hidden[idx] {
          continue;
        }
        // Add cell divider between cells
        if !self.preview_mode && visible_count > 0 {
          let divider = self.view_add_cell_divider(idx.saturating_sub(1));
          if is_dragging {
            col = col.push(
              mouse_area(divider)
                .on_enter(Message::DragOverCell(idx))
                .interaction(iced::mouse::Interaction::Grabbing),
            );
          } else {
            col = col.push(divider);
          }
        }

        // Drop indicator above this cell
        if self.drop_target == Some(idx) {
          col = col.push(
            container(rule::horizontal(2).style(drop_indicator_style))
              .padding([0, 20]),
          );
        }

        let is_focused = self.focused_cell == Some(idx);
        let cell_el = self.view_cell(idx, editor, is_focused);

        // During drag, wrap each cell in a mouse_area to detect hover
        let cell_el: Element<'_, Message> = if is_dragging {
          let is_being_dragged = self.dragging_cell == Some(idx);
          let inner: Element<'_, Message> = if is_being_dragged {
            container(cell_el).style(dragged_cell_style).into()
          } else {
            cell_el
          };
          mouse_area(inner)
            .on_enter(Message::DragOverCell(idx))
            .interaction(iced::mouse::Interaction::Grabbing)
            .into()
        } else {
          cell_el
        };

        col = col.push(cell_el);
        visible_count += 1;
      }

      let cell_count = self.cell_editors.len();

      if !self.preview_mode {
        // Final add-cell divider after last cell
        let divider = self.view_add_cell_divider(cell_count.saturating_sub(1));
        if is_dragging {
          col = col.push(
            mouse_area(divider)
              .on_enter(Message::DragOverCell(cell_count))
              .interaction(iced::mouse::Interaction::Grabbing),
          );
        } else {
          col = col.push(divider);
        }
      }

      // Drop indicator after the last cell
      if self.drop_target == Some(cell_count) {
        col = col.push(
          container(rule::horizontal(2).style(drop_indicator_style))
            .padding([0, 20]),
        );
      }

      // Bottom padding so the last element isn't clipped by the status bar
      col = col.push(space::Space::new().height(32));

      scrollable(container(col.max_width(800)).center_x(Fill).padding(
        iced::Padding {
          top: 0.0,
          right: 14.0,
          bottom: 0.0,
          left: 0.0,
        },
      ))
      .id(iced::widget::Id::from("cells-scroll"))
      .height(Fill)
      .into()
    };

    // ── Status bar ──
    let file_label = match &self.file_path {
      Some(p) => {
        let s = p.display().to_string();
        if s.len() > 60 {
          format!("...{}", &s[s.len() - 50..])
        } else {
          s
        }
      }
      None => String::from("Untitled"),
    };

    let dirty_marker = if self.is_dirty { " [modified]" } else { "" };

    let status_bar = row![
      text(format!("{file_label}{dirty_marker}")).size(11),
      text("  |  ").size(11),
      text(&self.status).size(11),
      text("  |  ").size(11),
      text(format!("{} cells", self.cell_editors.len())).size(11),
    ]
    .spacing(4)
    .padding([3, 8]);

    // ── Table of contents sidebar ──
    let content_area: Element<'_, Message> = if self.show_toc {
      let mut toc_col = Column::new().spacing(0).padding([8, 8]);

      let hidden = self.compute_hidden_cells();
      // Track the widest entry to size the panel dynamically.
      let mut max_entry_width: f32 = 0.0;
      for (idx, editor) in self.cell_editors.iter().enumerate() {
        if hidden[idx] {
          continue;
        }
        if let Some(level) = heading_level(editor.style) {
          let label = editor.content.text();
          let label = label.trim();
          let label = if label.is_empty() {
            format!("(empty {})", editor.style)
          } else if label.chars().count() > 60 {
            format!("{}…", label.chars().take(59).collect::<String>())
          } else {
            label.to_string()
          };
          let indent = (level as u16) * 12;
          let font_size = match level {
            0 => 13.0,
            1 => 12.0,
            _ => 11.0,
          };
          // Estimate entry width: left pad + text + right pad.
          // Average character width ≈ 0.48 × font_size for proportional fonts.
          let char_width = font_size * 0.48;
          let entry_width =
            (8 + indent) as f32 + label.len() as f32 * char_width + 8.0;
          if entry_width > max_entry_width {
            max_entry_width = entry_width;
          }
          toc_col = toc_col.push(
            button(text(label).size(font_size).font(Font::DEFAULT))
              .on_press(Message::ScrollToCell(idx))
              .padding(iced::Padding {
                top: 2.0,
                right: 8.0,
                bottom: 2.0,
                left: (8 + indent) as f32,
              })
              .width(Fill)
              .style(toc_entry_style),
          );
        }
      }

      // Size to fit content (with outer padding), but shrink when
      // the window is narrow (at most 30% of window width).
      let content_width = max_entry_width + 16.0;
      let window_cap = self.window_width * 0.3;
      let toc_width = content_width.min(window_cap).max(160.0);
      let toc_panel = container(scrollable(toc_col).height(Fill))
        .width(toc_width)
        .height(Fill)
        .style(toc_panel_style);

      row![toc_panel, rule::vertical(1).style(separator_style), cells,]
        .height(Fill)
        .into()
    } else {
      cells
    };

    // ── Layout ──
    let main_view: Element<'_, Message> = column![
      toolbar,
      rule::horizontal(1).style(separator_style),
      content_area,
      status_bar,
    ]
    .spacing(0)
    .into();

    // ── Graphics modal overlay ──
    // Always use stack! so the widget tree structure stays the same
    // when the modal opens/closes, preserving scroll position.
    let modal_layer: Element<'_, Message> =
      if let Some(modal_idx) = self.graphics_modal_cell {
        let editor = &self.cell_editors[modal_idx];

        let graphic: Element<'_, Message> =
          if let Some((ref img_handle, _w, _h)) = editor.graphics_image {
            image(img_handle.clone())
              .width(iced::Length::Shrink)
              .height(iced::Length::Shrink)
              .content_fit(iced::ContentFit::Contain)
              .into()
          } else if let Some(ref handle) = editor.graphics_handle {
            svg::Svg::new(handle.clone())
              .width(iced::Length::Shrink)
              .height(iced::Length::Shrink)
              .into()
          } else {
            text("No graphic").into()
          };

        let graphic_clickable: Element<'_, Message> = mouse_area(graphic)
          .on_right_press(Message::ShowGraphicsContextMenu(modal_idx))
          .into();

        let close_btn = button(text("Close").size(13))
          .on_press(Message::CloseGraphicsModal)
          .padding([6, 16])
          .style(muted_button_style);

        let modal_content = container(
          column![graphic_clickable, close_btn]
            .spacing(12)
            .align_x(Center),
        )
        .center(Fill)
        .padding(40);

        mouse_area(
          container(opaque(modal_content))
            .width(Fill)
            .height(Fill)
            .style(graphics_modal_backdrop_style),
        )
        .on_press(Message::CloseGraphicsModal)
        .into()
      } else {
        column![].into()
      };

    // ── Graphics context menu overlay ──
    // Positioned at the cursor location captured when the right-click fired.
    // A transparent full-window mouse_area catches outside-clicks to dismiss.
    let context_menu_layer: Element<'_, Message> =
      if let Some(menu_idx) = self.graphics_context_menu_cell {
        let save_btn = button(text("Save Graphic As…").size(13))
          .on_press(Message::SaveGraphicAs(menu_idx))
          .padding([6, 14])
          .style(context_menu_item_style);

        let menu = container(column![save_btn].spacing(2))
          .padding(4)
          .style(context_menu_style);

        let pos = self.graphics_context_menu_pos;
        let x = pos.x.max(0.0);
        let y = pos.y.max(0.0);
        let positioned = column![
          space::vertical().height(iced::Length::Fixed(y)),
          row![
            space::horizontal().width(iced::Length::Fixed(x)),
            opaque(menu),
          ],
        ];

        mouse_area(container(positioned).width(Fill).height(Fill))
          .on_press(Message::CloseGraphicsContextMenu)
          .on_right_press(Message::CloseGraphicsContextMenu)
          .into()
      } else {
        column![].into()
      };

    stack![main_view, modal_layer, context_menu_layer].into()
  }

  /// Compute which cells are hidden due to a collapsed Chapter or
  /// Subchapter above them. A collapsed heading hides all following
  /// cells until the next heading at the same level or higher.
  fn compute_hidden_cells(&self) -> Vec<bool> {
    let states: Vec<(CellStyle, bool)> = self
      .cell_editors
      .iter()
      .map(|e| (e.style, e.is_collapsed))
      .collect();
    compute_hidden_cells_from_states(&states)
  }

  /// Small "+" divider above a cell (inserts before it).
  fn view_add_cell_divider_above(&self, idx: usize) -> Element<'_, Message> {
    container(
      button(text("+").size(10))
        .on_press(Message::AddCellAbove(idx))
        .padding([0, 8])
        .style(add_cell_button_style),
    )
    .center_x(Fill)
    .padding([2, 0])
    .into()
  }

  /// Small "+" divider between cells.
  fn view_add_cell_divider(&self, idx: usize) -> Element<'_, Message> {
    let is_focused = self.focused_divider == Some(idx);
    let style_fn = if is_focused {
      focused_add_cell_button_style
    } else {
      add_cell_button_style
    };
    container(
      button(text("+").size(10))
        .on_press(Message::AddCellBelow(idx))
        .padding([0, 8])
        .style(style_fn),
    )
    .center_x(Fill)
    .padding([1, 0])
    .into()
  }

  fn view_cell<'a>(
    &'a self,
    idx: usize,
    editor: &'a CellEditor,
    _is_focused: bool,
  ) -> Element<'a, Message> {
    let is_input =
      editor.style == CellStyle::Input || editor.style == CellStyle::Code;
    let is_dark = !matches!(self.theme, Theme::Light);

    // ── Left gutter: style picker + delete ──
    let mut gutter = Column::new().spacing(2).width(iced::Length::Shrink);

    if !self.preview_mode {
      // Cell type: icon button with overlay dropdown
      gutter = gutter.push(cell_type_dropdown::cell_type_dropdown(
        editor.style,
        self.cell_type_menu_open == Some(idx),
        CELL_STYLES,
        Message::ToggleCellTypeMenu(idx),
        move |s| Message::CellStyleChanged(idx, s),
      ));

      // Drag handle: visible only when hovering the gutter area
      let show_handle =
        self.hovered_gutter == Some(idx) && self.dragging_cell.is_none();
      let drag_handle: Element<'a, Message> = if show_handle {
        let grip_svg = svg::Handle::from_memory(ICON_GRIP.as_bytes().to_vec());
        mouse_area(
          container(
            svg::Svg::new(grip_svg)
              .width(14)
              .height(14)
              .style(trash_icon_style),
          )
          .padding([4, 4])
          .style(drag_handle_container_style),
        )
        .on_press(Message::DragStart(idx))
        .interaction(iced::mouse::Interaction::Grab)
        .into()
      } else {
        // Invisible spacer to keep layout stable
        space::Space::new().width(22).into()
      };

      gutter = gutter.push(drag_handle);
      // Fill remaining gutter height so hover zone extends below
      gutter = gutter.push(space::Space::new().width(22).height(Fill));
    }

    let gutter: Element<'a, Message> = if !self.preview_mode {
      mouse_area(gutter)
        .on_enter(Message::GutterEnter(idx))
        .on_exit(Message::GutterExit(idx))
        .into()
    } else {
      gutter.into()
    };

    // ── Text editor ──
    let font_size = match editor.style {
      CellStyle::Title => 20.0,
      CellStyle::Subtitle => 16.0,
      CellStyle::Chapter => 18.0,
      CellStyle::Subchapter => 16.0,
      CellStyle::Section => 15.0,
      CellStyle::Subsection => 14.0,
      CellStyle::Subsubsection => 13.0,
      CellStyle::Item | CellStyle::Subitem => 13.0,
      _ => 13.0,
    };

    let cell_font = match editor.style {
      CellStyle::Title
      | CellStyle::Subtitle
      | CellStyle::Chapter
      | CellStyle::Subchapter => Font {
        weight: iced::font::Weight::Bold,
        ..Font::MONOSPACE
      },
      _ => Font::MONOSPACE,
    };

    let cell_style = editor.style;
    let in_preview = self.preview_mode;
    let has_output = editor.stdout.is_some()
      || editor.graphics_svg.is_some()
      || editor.output.as_ref().is_some_and(|o| {
        let d = o
          .replace("-Graphics-", "")
          .replace("-Graphics3D-", "")
          .replace("-Image-", "");
        !d.trim().is_empty()
      });
    // Display-only stored graphics (Demonstrations snapshots) show their
    // output section without an editor row.
    let is_grouped =
      (is_input && has_output || editor.stored_graphic) && !in_preview;
    let cursor_pos = editor.content.cursor().position;
    let cursor_line = cursor_pos.line;
    let cursor_column = cursor_pos.column;
    let line_count = editor.content.line_count();
    let at_last_line = cursor_line >= line_count.saturating_sub(1);
    let at_first_line = cursor_line == 0;
    let cell_count = self.cell_editors.len();
    let has_selection = editor.content.selection().is_some();
    let cursor_at_line_start = {
      let text = editor.content.text();
      text.lines().nth(cursor_line).is_none_or(|line| {
        line[..cursor_column.min(line.len())]
          .chars()
          .all(|c| c.is_whitespace())
      })
    };
    let cell_editor = text_editor(&editor.content)
      .id(iced::widget::Id::from(format!("cell-{idx}")))
      .on_action(move |action| Message::CellAction(idx, action))
      .key_binding(move |key_press| {
        let text_editor::KeyPress {
          key,
          modifiers,
          status,
          ..
        } = &key_press;
        // Only handle custom bindings when this editor is focused;
        // iced dispatches key events to ALL text_editors in the tree.
        if !matches!(status, text_editor::Status::Focused { .. }) {
          return text_editor::Binding::from_key_press(key_press);
        }
        // Ctrl+A / Ctrl+E: Emacs-style Home / End.
        // Must be checked before the `command()` block because on Linux
        // control() == command(), so Ctrl+A would otherwise become SelectAll.
        if modifiers.control() {
          match key.as_ref() {
            keyboard::Key::Character("a") => {
              return Some(text_editor::Binding::Move(
                text_editor::Motion::Home,
              ));
            }
            keyboard::Key::Character("e") => {
              return Some(text_editor::Binding::Move(
                text_editor::Motion::End,
              ));
            }
            _ => {}
          }
        }
        if modifiers.command() {
          match key.as_ref() {
            keyboard::Key::Character("z") if modifiers.shift() => {
              return Some(text_editor::Binding::Custom(Message::Redo(idx)));
            }
            keyboard::Key::Character("z") => {
              return Some(text_editor::Binding::Custom(Message::Undo(idx)));
            }
            keyboard::Key::Character("/") => {
              return Some(text_editor::Binding::Custom(
                Message::ToggleComment(idx),
              ));
            }
            // Let Cmd+V/C/X/A pass through to iced's default handling
            // (paste, copy, cut, select-all).
            keyboard::Key::Character("v" | "c" | "x" | "a") => {
              return text_editor::Binding::from_key_press(key_press);
            }
            // Suppress character insertion for other Cmd shortcuts
            // (e.g. Cmd+S, Cmd+O, Cmd+N) — these are handled by the
            // global event handler and must not insert text.
            keyboard::Key::Character(_) => {
              return Some(text_editor::Binding::Sequence(vec![]));
            }
            _ => {}
          }
        }
        // Shift+Enter: evaluate the cell and move to the cell below.
        // If there is no cell below, insert a new one. Handled here
        // (before the text editor processes the key) so no stray newline
        // is inserted into the cell's content.
        if modifiers.shift()
          && let keyboard::Key::Named(keyboard::key::Named::Enter) =
            key.as_ref()
        {
          let is_last = idx + 1 >= cell_count;
          let mut bindings =
            vec![text_editor::Binding::Custom(Message::EvaluateCell(idx))];
          if is_last {
            bindings
              .push(text_editor::Binding::Custom(Message::AddCellBelow(idx)));
            bindings
              .push(text_editor::Binding::Custom(Message::ScrollCellsToEnd));
          } else {
            bindings
              .push(text_editor::Binding::Custom(Message::FocusCell(idx + 1)));
          }
          return Some(text_editor::Binding::Sequence(bindings));
        }
        // Tab / Shift+Tab indentation
        if let keyboard::Key::Named(keyboard::key::Named::Tab) = key.as_ref() {
          if modifiers.shift() {
            return Some(text_editor::Binding::Custom(Message::UnindentLines(
              idx,
            )));
          } else if has_selection || cursor_at_line_start {
            return Some(text_editor::Binding::Custom(Message::IndentLines(
              idx,
            )));
          } else {
            // Tab not at beginning of line with no selection: do nothing
            return Some(text_editor::Binding::Sequence(vec![]));
          }
        }
        // Arrow key navigation between cells
        let no_mods =
          !modifiers.shift() && !modifiers.command() && !modifiers.control();
        if no_mods {
          if let keyboard::Key::Named(keyboard::key::Named::ArrowDown) =
            key.as_ref()
            && at_last_line
            && idx < cell_count.saturating_sub(1)
          {
            return Some(text_editor::Binding::Sequence(vec![
              text_editor::Binding::Unfocus,
              text_editor::Binding::Custom(Message::FocusDividerBelow(idx)),
            ]));
          }
          if let keyboard::Key::Named(keyboard::key::Named::ArrowUp) =
            key.as_ref()
            && at_first_line
            && idx > 0
          {
            return Some(text_editor::Binding::Sequence(vec![
              text_editor::Binding::Unfocus,
              text_editor::Binding::Custom(Message::FocusDividerAbove(idx)),
            ]));
          }
        }
        // Wrap selection with matching brackets/quotes
        if has_selection && let Some(ref text) = key_press.text {
          let pair = match text.as_ref() {
            "{" => Some(('{', '}')),
            "[" => Some(('[', ']')),
            "\"" => Some(('"', '"')),
            "'" => Some(('\'', '\'')),
            "(" => Some(('(', ')')),
            _ => None,
          };
          if let Some((open, close)) = pair {
            return Some(text_editor::Binding::Custom(Message::WrapSelection(
              idx, open, close,
            )));
          }
        }
        text_editor::Binding::from_key_press(key_press)
      })
      .font(cell_font)
      .height(iced::Length::Shrink)
      .padding(6)
      .size(font_size)
      .style(move |theme, status| {
        if in_preview {
          preview_editor_style(theme, status, cell_style)
        } else {
          cell_editor_style(theme, status, cell_style)
        }
      })
      .highlight_with::<highlighter::WolframHighlighter>(
        highlighter::WolframSettings { enabled: is_input },
        highlighter::format_highlight,
      );

    // ── Content column: editor + outputs ──

    let mut content_col = Column::new().spacing(0).width(Fill);
    if !editor.stored_graphic {
      content_col = content_col.push(cell_editor);
    }

    let stale = editor.output_stale;
    let stale_opacity = if stale { 0.35 } else { 1.0 };

    if is_grouped {
      // Small gap between input and output
      content_col = content_col.push(container(text("")).height(4).width(Fill));
      // Build output section with gray background
      let mut output_col = Column::new().spacing(0).width(Fill);

      // Warnings (e.g. unimplemented functions)
      if !editor.warnings.is_empty() {
        let warning_text = editor.warnings.join("\n");
        let warning_color = Color::from_rgba(0.85, 0.55, 0.10, stale_opacity);
        let warning_display = container(
          text(warning_text)
            .size(12)
            .font(Font::MONOSPACE)
            .color(warning_color),
        )
        .padding(6)
        .width(Fill);

        output_col = output_col.push(warning_display);
      }

      // Stdout (Print output)
      if editor.stdout.is_some() {
        let stdout_editor = text_editor(&editor.stdout_content)
          .on_action(move |action| Message::StdoutAction(idx, action))
          .font(Font::MONOSPACE)
          .height(iced::Length::Shrink)
          .padding(6)
          .size(12)
          .style(move |theme, status| {
            output_editor_style(theme, status, stale)
          });
        output_col = output_col.push(stdout_editor);
      }

      // Graphics rendering (pre-rasterized image, falls back to SVG)
      // Double-click opens a fullscreen modal for detailed inspection.
      // Right-click opens a context menu (Save Graphic As).
      if let Some((ref img_handle, w, h)) = editor.graphics_image {
        let mut img_widget = image(img_handle.clone())
          .width(iced::Length::Fixed(w as f32))
          .height(iced::Length::Fixed(h as f32));
        if stale {
          img_widget = img_widget.opacity(0.3_f32);
        }
        let clickable = mouse_area(container(img_widget).padding(4))
          .on_double_click(Message::OpenGraphicsModal(idx))
          .on_right_press(Message::ShowGraphicsContextMenu(idx));
        output_col = output_col.push(clickable);
      } else if let Some(ref handle) = editor.graphics_handle {
        let mut svg_widget =
          svg::Svg::new(handle.clone()).width(iced::Length::Shrink);
        if stale {
          svg_widget = svg_widget.opacity(0.3_f32);
        }
        let clickable = mouse_area(container(svg_widget).padding(4))
          .on_double_click(Message::OpenGraphicsModal(idx))
          .on_right_press(Message::ShowGraphicsContextMenu(idx));
        output_col = output_col.push(clickable);
      }

      // Interactive Manipulate widget
      if let Some(ref state) = editor.manipulate_state {
        output_col =
          output_col.push(render_manipulate_widget(idx, state, stale));
      }

      // Graphical audio player (Play[…] / Sound[…] / Audio[…] results)
      if let Some(ref audio) = editor.sound {
        output_col = output_col.push(render_audio_player(
          idx,
          audio,
          self.is_playing(idx),
        ));
      }

      // Hyperlink buttons (clickable, blue, opens URL on press)
      for (label, uri) in &editor.hyperlinks {
        output_col = output_col.push(render_hyperlink(label, uri, stale));
      }

      // Result output: the typeset SVG (same rendering the Playground shows)
      // when every result produced one and the baked colors still match the
      // theme, otherwise the selectable plain text (filtering graphics
      // placeholders).
      if editor.output_all_svg
        && editor.output_dark == is_dark
        && !editor.output_images.is_empty()
      {
        output_col =
          output_col.push(output_images_element(&editor.output_images, stale));
      } else if editor.output.is_some()
        && !editor.output_content.text().trim().is_empty()
      {
        let output_editor = text_editor(&editor.output_content)
          .on_action(move |action| Message::OutputAction(idx, action))
          .font(Font::MONOSPACE)
          .height(iced::Length::Shrink)
          .padding(6)
          .size(12)
          .style(move |theme, status| {
            output_editor_style(theme, status, stale)
          });
        output_col = output_col.push(output_editor);
      }

      content_col = content_col
        .push(container(output_col).width(Fill).style(output_area_style));
    } else {
      // Non-grouped: show outputs inline without special styling

      // Warnings
      if !editor.warnings.is_empty() {
        let warning_text = editor.warnings.join("\n");
        let warning_color = Color::from_rgba(0.85, 0.55, 0.10, stale_opacity);
        let warning_display = container(
          text(warning_text)
            .size(12)
            .font(Font::MONOSPACE)
            .color(warning_color),
        )
        .padding(6)
        .width(Fill);

        content_col = content_col.push(warning_display);
      }

      if editor.stdout.is_some() {
        let stdout_editor = text_editor(&editor.stdout_content)
          .on_action(move |action| Message::StdoutAction(idx, action))
          .font(Font::MONOSPACE)
          .height(iced::Length::Shrink)
          .padding(6)
          .size(12)
          .style(move |theme, status| {
            output_editor_style(theme, status, stale)
          });
        content_col = content_col.push(stdout_editor);
      }

      if let Some((ref img_handle, w, h)) = editor.graphics_image {
        let mut img_widget = image(img_handle.clone())
          .width(iced::Length::Fixed(w as f32))
          .height(iced::Length::Fixed(h as f32));
        if stale {
          img_widget = img_widget.opacity(0.3_f32);
        }
        let clickable = mouse_area(container(img_widget).padding(4))
          .on_double_click(Message::OpenGraphicsModal(idx))
          .on_right_press(Message::ShowGraphicsContextMenu(idx));
        content_col = content_col.push(clickable);
      } else if let Some(ref handle) = editor.graphics_handle {
        let mut svg_widget =
          svg::Svg::new(handle.clone()).width(iced::Length::Shrink);
        if stale {
          svg_widget = svg_widget.opacity(0.3_f32);
        }
        let clickable = mouse_area(container(svg_widget).padding(4))
          .on_double_click(Message::OpenGraphicsModal(idx))
          .on_right_press(Message::ShowGraphicsContextMenu(idx));
        content_col = content_col.push(clickable);
      }

      // Interactive Manipulate widget
      if let Some(ref state) = editor.manipulate_state {
        content_col =
          content_col.push(render_manipulate_widget(idx, state, stale));
      }

      // Graphical audio player (Play[…] / Sound[…] / Audio[…] results)
      if let Some(ref audio) = editor.sound {
        content_col = content_col.push(render_audio_player(
          idx,
          audio,
          self.is_playing(idx),
        ));
      }

      // Hyperlink buttons
      for (label, uri) in &editor.hyperlinks {
        content_col = content_col.push(render_hyperlink(label, uri, stale));
      }

      if editor.output_all_svg
        && editor.output_dark == is_dark
        && !editor.output_images.is_empty()
      {
        content_col =
          content_col.push(output_images_element(&editor.output_images, stale));
      } else if editor.output.is_some()
        && !editor.output_content.text().trim().is_empty()
      {
        let output_editor = text_editor(&editor.output_content)
          .on_action(move |action| Message::OutputAction(idx, action))
          .font(Font::MONOSPACE)
          .height(iced::Length::Shrink)
          .padding(6)
          .size(12)
          .style(move |theme, status| {
            output_editor_style(theme, status, stale)
          });
        content_col = content_col.push(output_editor);
      }
    }

    let content_el: Element<'a, Message> = content_col.into();

    // ── Right side: play button + trash ──
    let right_side: Element<'a, Message> = if !self.preview_mode {
      let trash_svg =
        svg::Handle::from_memory(TRASH_ICON_SVG.as_bytes().to_vec());
      let trash_btn = button(
        svg::Svg::new(trash_svg)
          .width(14)
          .height(14)
          .style(trash_icon_style),
      )
      .on_press_maybe(
        (self.cell_editors.len() > 1).then_some(Message::DeleteCell(idx)),
      )
      .padding([2, 4])
      .style(trash_button_style);

      let mut right_col = Column::new().spacing(2).padding(iced::Padding {
        top: 0.0,
        right: 0.0,
        bottom: 0.0,
        left: 4.0,
      });
      if is_input {
        let play_svg =
          svg::Handle::from_memory(PLAY_CIRCLE_SVG.as_bytes().to_vec());
        right_col = right_col.push(
          button(
            svg::Svg::new(play_svg)
              .width(14)
              .height(14)
              .style(trash_icon_style),
          )
          .on_press(Message::EvaluateCell(idx))
          .padding([2, 4])
          .style(trash_button_style),
        );
      }
      right_col = right_col.push(trash_btn);
      right_col.into()
    } else {
      text("").into()
    };

    // ── Collapse chevron (Chapter / Subchapter only) ──
    // Reserve a fixed-width slot at the very left of every cell so
    // the cell type dropdown and all downstream columns stay aligned
    // across cells. For Chapter/Subchapter the slot holds a clickable
    // chevron; for other cells it's empty.
    const CHEVRON_SLOT_WIDTH: f32 = 20.0;
    let is_collapsible =
      matches!(editor.style, CellStyle::Chapter | CellStyle::Subchapter);
    let chevron_el: Element<'a, Message> =
      if is_collapsible && !self.preview_mode {
        let chevron_svg = svg::Handle::from_memory(
          if editor.is_collapsed {
            ICON_CHEVRON_RIGHT
          } else {
            ICON_CHEVRON_DOWN
          }
          .as_bytes()
          .to_vec(),
        );
        container(
          button(
            svg::Svg::new(chevron_svg)
              .width(14)
              .height(14)
              .style(trash_icon_style),
          )
          .on_press(Message::ToggleCollapse(idx))
          .padding([2, 2])
          .style(trash_button_style),
        )
        .width(iced::Length::Fixed(CHEVRON_SLOT_WIDTH))
        .align_x(Center)
        .into()
      } else {
        // Empty spacer so cells without a chevron still align visually
        // with ones that have one.
        container(text(""))
          .width(iced::Length::Fixed(CHEVRON_SLOT_WIDTH))
          .into()
      };

    let cell_row = row![chevron_el, gutter, content_el, right_side]
      .spacing(0)
      .padding([1, 2]);

    container(cell_row).width(Fill).into()
  }
}

// ── Indent/unindent helpers ─────────────────────────────────────────

/// Given the cursor line, selected text, and total line count,
/// determine which lines are covered by the selection.
fn selection_line_range(
  cursor_line: usize,
  sel_text: &str,
  line_count: usize,
) -> (usize, usize) {
  let sel_lines = sel_text.chars().filter(|c| *c == '\n').count() + 1;
  // Cursor could be at either end of the selection
  let a = cursor_line.saturating_sub(sel_lines - 1);
  let b = cursor_line;
  let alt_end = cursor_line + sel_lines - 1;
  if alt_end < line_count && a == cursor_line {
    (cursor_line, alt_end)
  } else {
    (a, b)
  }
}

/// Derive both endpoints of a selection: (anchor, cursor) as (line, col).
/// The cursor position is known; the anchor is derived from the selected text.
fn selection_endpoints(
  cursor_line: usize,
  cursor_col: usize,
  sel_text: &str,
  lines: &[&str],
) -> ((usize, usize), (usize, usize)) {
  let sel_newlines = sel_text.chars().filter(|c| *c == '\n').count();

  if sel_newlines == 0 {
    // Single-line selection
    // Try forward: anchor before cursor
    let anchor_col = cursor_col.saturating_sub(sel_text.len());
    let candidate = &lines[cursor_line][anchor_col
      ..anchor_col + sel_text.len().min(lines[cursor_line].len() - anchor_col)];
    if candidate == sel_text {
      return ((cursor_line, anchor_col), (cursor_line, cursor_col));
    }
    // Backward: anchor after cursor
    let anchor_col = cursor_col + sel_text.len();
    return ((cursor_line, anchor_col), (cursor_line, cursor_col));
  }

  let sel_lines_vec: Vec<&str> = sel_text.split('\n').collect();

  // Try forward selection: cursor is at end, anchor is above
  let anchor_line = cursor_line.saturating_sub(sel_newlines);
  if anchor_line + sel_newlines == cursor_line {
    let first_sel_line = sel_lines_vec[0];
    if let Some(line) = lines.get(anchor_line)
      && line.ends_with(first_sel_line)
    {
      let anchor_col = line.len() - first_sel_line.len();
      return ((anchor_line, anchor_col), (cursor_line, cursor_col));
    }
  }

  // Backward selection: cursor is at start, anchor is below
  let anchor_line = cursor_line + sel_newlines;
  if anchor_line < lines.len() {
    let last_sel_line = sel_lines_vec[sel_lines_vec.len() - 1];
    let anchor_col = last_sel_line.len();
    return ((anchor_line, anchor_col), (cursor_line, cursor_col));
  }

  // Fallback
  ((cursor_line, cursor_col), (cursor_line, cursor_col))
}

/// After replacing editor content, restore a selection from
/// `anchor` (line, col) to `cursor_pos` (line, col).
/// Cursor starts at (0,0) after Content::with_text.
fn restore_selection(
  content: &mut text_editor::Content,
  anchor: (usize, usize),
  cursor_pos: (usize, usize),
) {
  // Move to anchor position first
  for _ in 0..anchor.0 {
    content.perform(text_editor::Action::Move(text_editor::Motion::Down));
  }
  content.perform(text_editor::Action::Move(text_editor::Motion::Home));
  for _ in 0..anchor.1 {
    content.perform(text_editor::Action::Move(text_editor::Motion::Right));
  }

  // Now select from anchor to cursor_pos
  if cursor_pos.0 > anchor.0 {
    for _ in anchor.0..cursor_pos.0 {
      content.perform(text_editor::Action::Select(text_editor::Motion::Down));
    }
    // After moving down, we need to go to the right column on the target line
    // Select::Down keeps the column, so go to Home first then right
    content.perform(text_editor::Action::Select(text_editor::Motion::Home));
    for _ in 0..cursor_pos.1 {
      content.perform(text_editor::Action::Select(text_editor::Motion::Right));
    }
  } else if cursor_pos.0 < anchor.0 {
    for _ in cursor_pos.0..anchor.0 {
      content.perform(text_editor::Action::Select(text_editor::Motion::Up));
    }
    content.perform(text_editor::Action::Select(text_editor::Motion::Home));
    for _ in 0..cursor_pos.1 {
      content.perform(text_editor::Action::Select(text_editor::Motion::Right));
    }
  } else {
    // Same line
    if cursor_pos.1 > anchor.1 {
      for _ in anchor.1..cursor_pos.1 {
        content
          .perform(text_editor::Action::Select(text_editor::Motion::Right));
      }
    } else if cursor_pos.1 < anchor.1 {
      for _ in cursor_pos.1..anchor.1 {
        content.perform(text_editor::Action::Select(text_editor::Motion::Left));
      }
    }
  }
}

/// Preserve trailing newline if the original text had one.
fn preserve_trailing_newline(original: &str, new_text: String) -> String {
  if original.ends_with('\n') && !new_text.ends_with('\n') {
    new_text + "\n"
  } else {
    new_text
  }
}

/// Toggle `(* ... *)` commenting on a single line, preserving leading
/// whitespace. Returns `(new_line, col_shift)` — `col_shift` is the
/// signed column adjustment to apply to a caret that was on this line.
fn toggle_line_comment(line: &str) -> (String, isize) {
  let trimmed = line.trim();
  let leading_ws = &line[..line.len() - line.trim_start().len()];
  if trimmed.starts_with("(*") && trimmed.ends_with("*)") {
    let inner = trimmed.strip_prefix("(*").unwrap_or(trimmed);
    let inner = inner.strip_prefix(' ').unwrap_or(inner);
    let inner = inner.strip_suffix("*)").unwrap_or(inner);
    let inner = inner.strip_suffix(' ').unwrap_or(inner);
    let removed =
      line.len() as isize - leading_ws.len() as isize - inner.len() as isize;
    (format!("{leading_ws}{inner}"), -removed)
  } else {
    (format!("{leading_ws}(* {trimmed} *)"), 3isize)
  }
}

// ── Event handling ──────────────────────────────────────────────────

fn handle_event(
  event: iced::Event,
  status: iced::event::Status,
  _id: iced::window::Id,
) -> Option<Message> {
  // Global mouse release ends any cell drag in progress
  if let iced::Event::Mouse(iced::mouse::Event::ButtonReleased(
    iced::mouse::Button::Left,
  )) = &event
  {
    return Some(Message::DragEnd);
  }

  // Track cursor position so we can place the right-click context menu.
  if let iced::Event::Mouse(iced::mouse::Event::CursorMoved { position }) =
    &event
  {
    return Some(Message::CursorMoved(*position));
  }

  if let iced::Event::Window(iced::window::Event::CloseRequested) = &event {
    return Some(Message::CloseRequested(_id));
  }

  if let iced::Event::Window(iced::window::Event::Rescaled(scale)) = &event {
    return Some(Message::ScaleFactorChanged(*scale));
  }

  if let iced::Event::Window(iced::window::Event::Resized(size)) = &event {
    return Some(Message::WindowResized(*size));
  }

  if let iced::Event::Keyboard(keyboard::Event::KeyPressed {
    key,
    modifiers,
    ..
  }) = event
  {
    // Escape key (always forwarded for modal close)
    if matches!(
      key.as_ref(),
      keyboard::Key::Named(keyboard::key::Named::Escape)
    ) {
      return Some(Message::KeyPressed(key, modifiers));
    }

    // When no widget captured the event (e.g. divider is focused),
    // handle arrow keys and Enter for navigation.
    if matches!(status, iced::event::Status::Ignored) {
      let no_mods =
        !modifiers.shift() && !modifiers.command() && !modifiers.control();
      if no_mods
        && let keyboard::Key::Named(
          keyboard::key::Named::ArrowDown
          | keyboard::key::Named::ArrowUp
          | keyboard::key::Named::Enter,
        ) = key.as_ref()
      {
        return Some(Message::KeyPressed(key, modifiers));
      }
    }

    // Ctrl shortcuts for text editing
    if modifiers.control() {
      match key.as_ref() {
        keyboard::Key::Character("d") | keyboard::Key::Character("w") => {
          return Some(Message::KeyPressed(key, modifiers));
        }
        _ => {}
      }
    }

    // Cmd/Ctrl shortcuts
    if modifiers.command() {
      match key.as_ref() {
        keyboard::Key::Character("s")
        | keyboard::Key::Character("o")
        | keyboard::Key::Character("n") => {
          return Some(Message::KeyPressed(key, modifiers));
        }
        _ => {}
      }
    }
  }
  None
}

// ── SVG rasterization ──────────────────────────────────────────────

/// Rasterize an SVG string to an RGBA bitmap at the given scale factor.
/// Returns the image handle together with the *logical* (1×) width and height.
/// Build the interactive widget for a Manipulate cell: one row of
/// controls (sliders or pick lists) followed by the current rendering.
/// Build the caption widget shown next to a Manipulate control. Renders the
/// label's styled runs as rich text so `Style[…, Italic]` shows as an italic
/// glyph (e.g. an italic `t`, or the italic `m` of `m₁`). Falls back to the
/// plain `label`, then the variable `name`, when there are no runs.
fn manipulate_label_widget<'a>(
  runs: &[woxi::functions::graphics::LabelRun],
  label: &str,
  name: &str,
  width: f32,
  enabled: bool,
) -> Element<'a, Message> {
  const SIZE: f32 = 12.0;
  // Match the family the upright runs inherit (the app default is
  // MONOSPACE) so an italic run doesn't jump to a different typeface.
  let italic = Font {
    style: iced::font::Style::Italic,
    ..Font::MONOSPACE
  };
  // A disabled control's label is dimmed to match the greyed-out widget.
  let color = move |theme: &Theme| {
    if enabled {
      text::Style::default()
    } else {
      text::Style {
        color: Some(theme.extended_palette().background.strong.color),
      }
    }
  };

  if runs.is_empty() {
    let fallback = if label.is_empty() { name } else { label };
    return text(fallback.to_string())
      .size(SIZE)
      .width(iced::Length::Fixed(width))
      .style(color)
      .into();
  }

  // rich_text spans carry a fixed color, so a disabled label uses a muted grey
  // (theme-agnostic) rather than the theme-derived color used above.
  let muted = Color::from_rgb(0.55, 0.55, 0.58);
  let spans: Vec<text::Span<'a, ()>> = runs
    .iter()
    .map(|r| {
      let mut s = iced::widget::span(r.text.clone());
      if r.italic {
        s = s.font(italic);
      }
      if !enabled {
        s = s.color(muted);
      }
      s
    })
    .collect();
  rich_text(spans)
    .size(SIZE)
    .width(iced::Length::Fixed(width))
    .into()
}

/// Approximate rendered width (px) of a Manipulate label at the caption font
/// size, from its character count. Used to size the shared label column to
/// the widest label so it sits snug against the sliders instead of leaving a
/// fixed 140px gutter.
fn manipulate_label_char_count(ctrl: &manipulate::ControlState) -> usize {
  let (label, name) = match ctrl {
    manipulate::ControlState::Continuous { label, name, .. }
    | manipulate::ControlState::Discrete { label, name, .. }
    | manipulate::ControlState::Slider2D { label, name, .. }
    | manipulate::ControlState::IntervalSlider { label, name, .. }
    | manipulate::ControlState::Trigger { label, name, .. }
    | manipulate::ControlState::Locator { label, name, .. } => (label, name),
    // Heading/divider rows span the full row instead of sitting in the
    // label column, so they don't widen it; a button carries its label
    // inside the button itself.
    manipulate::ControlState::Button { .. }
    | manipulate::ControlState::Heading { .. }
    | manipulate::ControlState::Divider => return 0,
  };
  let text = if label.is_empty() { name } else { label };
  text.chars().count()
}

/// Throttle window for Manipulate re-evaluation. A slider drag emits a burst of
/// change messages; coalescing them behind this short delay keeps the (blocking)
/// body evaluation off every mouse-move tick, so the graphic updates smoothly
/// instead of flickering while dragging. The control value itself still updates
/// immediately, so the thumb and value label track the cursor without lag.
const MANIPULATE_THROTTLE_MS: u64 = 16;

/// Auto-playing widgets (Animate / ListAnimate) advance their animation
/// control one step every ANIM_INTERVAL_MS. At ~60ms the default 100-step
/// continuous range sweeps in ~6s, matching Wolfram's leisurely Animate
/// (and the Playground's pace).
const ANIM_INTERVAL_MS: u64 = 60;

/// Maximum number of choices rendered as a segmented SetterBar (a row of
/// toggle buttons). Discrete controls with more settings fall back to a
/// dropdown so the control row can't grow unbounded.
const SETTER_BAR_MAX_CHOICES: usize = 6;

/// Whether an animation tick generated at `tick_at` should advance the
/// animation, given when the previous advance finished. The animation timer
/// keeps producing ticks while a (blocking, possibly slower-than-interval)
/// advance runs, so ticks queue up behind it in the runtime's message queue.
/// A tick generated before the previous advance finished is that backlog —
/// processing it would re-evaluate every playing widget again, making the
/// backlog grow without bound. Only a tick fresher than the last advance
/// counts; stale ones are dropped.
fn animation_tick_is_fresh(
  tick_at: std::time::Instant,
  last_advance: Option<std::time::Instant>,
) -> bool {
  last_advance.is_none_or(|done| tick_at >= done)
}

/// Spawn the debounce timer that drives a throttled Manipulate re-evaluation.
/// When it fires, `ManipulateReeval` re-evaluates the body with the latest
/// control values (see `ManipulateState::run_scheduled_reeval`).
fn manipulate_reeval_task(cell_idx: usize) -> Task<Message> {
  Task::perform(
    tokio::time::sleep(std::time::Duration::from_millis(
      MANIPULATE_THROTTLE_MS,
    )),
    move |()| Message::ManipulateReeval(cell_idx),
  )
}

fn render_manipulate_widget<'a>(
  cell_idx: usize,
  state: &'a manipulate::ManipulateState,
  stale: bool,
) -> Element<'a, Message> {
  let mut controls_col = Column::new().spacing(6).width(Fill);
  // `Appearance -> None` hides the control rows entirely — the animation
  // just runs. An animated widget keeps its play/pause toggle below.
  let show_controls = !state.appearance_none;
  // Size the label column to the widest label so it sits snug against the
  // sliders. ~7.3px per character at the 12px caption font (monospace),
  // plus a little trailing padding; clamped so a single-glyph label still
  // reads and a very long one can't swallow the slider.
  let max_label_chars = state
    .controls
    .iter()
    .map(manipulate_label_char_count)
    .max()
    .unwrap_or(0);
  let label_col_width = (max_label_chars as f32 * 7.3 + 6.0).clamp(20.0, 220.0);
  let visible_controls: &[manipulate::ControlState] =
    if show_controls { &state.controls } else { &[] };
  for (ctrl_idx, ctrl) in visible_controls.iter().enumerate() {
    // A control whose `Enabled` condition currently evaluates to `False` is
    // greyed out and swallows interaction (see `Message::Noop`).
    let enabled = state
      .control_is_enabled
      .get(ctrl_idx)
      .copied()
      .unwrap_or(true);
    match ctrl {
      manipulate::ControlState::Continuous {
        name,
        label,
        label_runs,
        min,
        max,
        step,
        current,
      } => {
        let label_widget = manipulate_label_widget(
          label_runs,
          label,
          name,
          label_col_width,
          enabled,
        );
        let mut s = slider(*min..=*max, *current, move |v| {
          if enabled {
            Message::ManipulateContinuousChanged(cell_idx, ctrl_idx, v)
          } else {
            Message::Noop
          }
        })
        .step(*step)
        .width(Fill);
        if !enabled {
          s = s.style(disabled_slider_style);
        }
        let value_widget = text(format_manipulate_number(*current))
          .size(11)
          .font(Font::MONOSPACE)
          .width(iced::Length::Fixed(64.0));
        let control_row = row![label_widget, s, value_widget]
          .align_y(Center)
          .spacing(8);
        controls_col = controls_col.push(control_row);
      }
      manipulate::ControlState::Discrete {
        name,
        label,
        label_runs,
        values,
        value_labels,
        value_label_svgs,
        current_index,
        popup,
      } => {
        let label_widget = manipulate_label_widget(
          label_runs,
          label,
          name,
          label_col_width,
          enabled,
        );
        // A boolean domain — `{v, {True, False}}` in either order — renders
        // as a checkbox, matching the Wolfram FrontEnd (which shows a
        // checkbox rather than a two-button setter for True/False).
        let bool_values: &[String] = values;
        let is_bool_domain = !*popup
          && bool_values.len() == 2
          && bool_values.iter().any(|v| v == "True")
          && bool_values.iter().any(|v| v == "False");
        if is_bool_domain {
          let checked =
            bool_values.get(*current_index).is_some_and(|v| v == "True");
          // Toggling selects the other entry; the update handler maps the
          // sent display label back to its index.
          let other_label = value_labels
            .iter()
            .zip(bool_values.iter())
            .find(|(_, v)| (*v == "True") != checked)
            .map(|(l, _)| l.clone());
          let mut cb = checkbox(checked);
          if enabled && let Some(target) = other_label {
            cb = cb.on_toggle(move |_| {
              Message::ManipulateDiscreteChanged(
                cell_idx,
                ctrl_idx,
                target.clone(),
              )
            });
          }
          let control_row = row![label_widget, cb].align_y(Center).spacing(8);
          controls_col = controls_col.push(control_row);
          continue;
        }
        let count = value_labels.len();
        // A small enumerated set renders as a segmented SetterBar (a row of
        // adjacent toggle buttons with the active choice highlighted), matching
        // Wolfram's SetterBar; a larger set — or an explicit
        // `ControlType -> PopupMenu` — renders a dropdown so the row can't
        // grow unbounded. The button labels are the display labels
        // (rule right-hand sides); pressing one sends its label, which the
        // update handler maps back to an index. A disabled control drops its
        // press handlers so it can't be changed.
        let control: Element<Message> =
          if count <= SETTER_BAR_MAX_CHOICES && !*popup {
            let mut bar = Row::new().spacing(0).align_y(Center);
            for (i, choice_label) in value_labels.iter().enumerate() {
              let is_selected = i == *current_index;
              let choice = choice_label.clone();
              // A choice whose rule label is a graphic (`"+" -> myIcon[2]`)
              // shows the rendered icon; text choices show their label.
              let btn_content: Element<Message> =
                match value_label_svgs.get(i).and_then(|s| s.as_ref()) {
                  Some(icon) => svg::Svg::new(icon.clone())
                    .width(iced::Length::Fixed(24.0))
                    .height(iced::Length::Fixed(14.0))
                    .into(),
                  None => text(choice_label.clone()).size(12).into(),
                };
              let mut btn = button(btn_content).padding([3, 10]).style(
                move |theme: &Theme, status| {
                  setter_button_style(
                    theme,
                    status,
                    is_selected,
                    i,
                    count,
                    enabled,
                  )
                },
              );
              if enabled {
                btn = btn.on_press(Message::ManipulateDiscreteChanged(
                  cell_idx, ctrl_idx, choice,
                ));
              }
              bar = bar.push(btn);
            }
            bar.into()
          } else {
            let selected = value_labels.get(*current_index).cloned();
            let on_select = move |choice: String| {
              if enabled {
                Message::ManipulateDiscreteChanged(cell_idx, ctrl_idx, choice)
              } else {
                Message::Noop
              }
            };
            pick_list(value_labels.clone(), selected, on_select)
              .width(iced::Length::Shrink)
              .into()
          };
        let control_row =
          row![label_widget, control].align_y(Center).spacing(8);
        controls_col = controls_col.push(control_row);
      }
      manipulate::ControlState::Slider2D {
        name,
        label,
        x_min,
        x_max,
        y_min,
        y_max,
        x,
        y,
        ..
      } => {
        // Rendered as two linked sliders (X and Y) driving the 2-vector.
        let x_span = (*x_max - *x_min).abs();
        let y_span = (*y_max - *y_min).abs();
        let x_step = if x_span > 0.0 { x_span / 100.0 } else { 1.0 };
        let y_step = if y_span > 0.0 { y_span / 100.0 } else { 1.0 };
        let mut x_slider = slider(*x_min..=*x_max, *x, move |v| {
          if enabled {
            Message::ManipulateSlider2DChanged(cell_idx, ctrl_idx, 0, v)
          } else {
            Message::Noop
          }
        })
        .step(x_step)
        .width(Fill);
        let mut y_slider = slider(*y_min..=*y_max, *y, move |v| {
          if enabled {
            Message::ManipulateSlider2DChanged(cell_idx, ctrl_idx, 1, v)
          } else {
            Message::Noop
          }
        })
        .step(y_step)
        .width(Fill);
        if !enabled {
          x_slider = x_slider.style(disabled_slider_style);
          y_slider = y_slider.style(disabled_slider_style);
        }
        let value_widget = text(format!(
          "{{{}, {}}}",
          format_manipulate_number(*x),
          format_manipulate_number(*y)
        ))
        .size(11)
        .font(Font::MONOSPACE)
        .width(iced::Length::Fixed(120.0));
        // Empty runs → plain label; shares label_col_width so 2D-slider rows
        // align with the other controls.
        let label_widget =
          manipulate_label_widget(&[], label, name, label_col_width, enabled);
        let control_row = row![
          label_widget,
          column![x_slider, y_slider].spacing(4),
          value_widget
        ]
        .align_y(Center)
        .spacing(8);
        controls_col = controls_col.push(control_row);
      }
      manipulate::ControlState::IntervalSlider {
        name,
        label,
        min,
        max,
        step,
        low,
        high,
      } => {
        // Rendered as two linked sliders (low and high endpoints).
        let mut low_slider = slider(*min..=*max, *low, move |v| {
          if enabled {
            Message::ManipulateIntervalChanged(cell_idx, ctrl_idx, 0, v)
          } else {
            Message::Noop
          }
        })
        .step(*step)
        .width(Fill);
        let mut high_slider = slider(*min..=*max, *high, move |v| {
          if enabled {
            Message::ManipulateIntervalChanged(cell_idx, ctrl_idx, 1, v)
          } else {
            Message::Noop
          }
        })
        .step(*step)
        .width(Fill);
        if !enabled {
          low_slider = low_slider.style(disabled_slider_style);
          high_slider = high_slider.style(disabled_slider_style);
        }
        let value_widget = text(format!(
          "{{{}, {}}}",
          format_manipulate_number(*low),
          format_manipulate_number(*high)
        ))
        .size(11)
        .font(Font::MONOSPACE)
        .width(iced::Length::Fixed(120.0));
        // Empty runs → plain label; shares label_col_width so interval-slider
        // rows align with the other controls.
        let label_widget =
          manipulate_label_widget(&[], label, name, label_col_width, enabled);
        let control_row = row![
          label_widget,
          column![low_slider, high_slider].spacing(4),
          value_widget
        ]
        .align_y(Center)
        .spacing(8);
        controls_col = controls_col.push(control_row);
      }
      manipulate::ControlState::Locator {
        name,
        label,
        x_min,
        x_max,
        y_min,
        y_max,
        points,
        auto_create,
      } => {
        // A list of draggable 2D points (e.g. polygon vertices): one X/Y
        // slider pair per point, plus add/remove buttons when the spec
        // allows `LocatorAutoCreate`.
        let x_span = (*x_max - *x_min).abs();
        let y_span = (*y_max - *y_min).abs();
        let x_step = if x_span > 0.0 { x_span / 100.0 } else { 1.0 };
        let y_step = if y_span > 0.0 { y_span / 100.0 } else { 1.0 };
        let label_widget =
          manipulate_label_widget(&[], label, name, label_col_width, enabled);
        let mut points_col = Column::new().spacing(4);
        for (point_idx, (x, y)) in points.iter().enumerate() {
          let mut x_slider = slider(*x_min..=*x_max, *x, move |v| {
            if enabled {
              Message::ManipulateLocatorChanged(
                cell_idx, ctrl_idx, point_idx, 0, v,
              )
            } else {
              Message::Noop
            }
          })
          .step(x_step)
          .width(Fill);
          let mut y_slider = slider(*y_min..=*y_max, *y, move |v| {
            if enabled {
              Message::ManipulateLocatorChanged(
                cell_idx, ctrl_idx, point_idx, 1, v,
              )
            } else {
              Message::Noop
            }
          })
          .step(y_step)
          .width(Fill);
          if !enabled {
            x_slider = x_slider.style(disabled_slider_style);
            y_slider = y_slider.style(disabled_slider_style);
          }
          let value_widget = text(format!(
            "{{{}, {}}}",
            format_manipulate_number(*x),
            format_manipulate_number(*y)
          ))
          .size(11)
          .font(Font::MONOSPACE)
          .width(iced::Length::Fixed(96.0));
          let mut point_row = row![
            text(format!("{}", point_idx + 1))
              .size(11)
              .font(Font::MONOSPACE),
            column![x_slider, y_slider].spacing(2),
            value_widget
          ]
          .align_y(Center)
          .spacing(8);
          if *auto_create {
            let mut remove_btn = button(text("−").size(11)).padding([1, 6]);
            if enabled {
              remove_btn =
                remove_btn.on_press(Message::ManipulateLocatorRemoved(
                  cell_idx, ctrl_idx, point_idx,
                ));
            }
            point_row = point_row.push(remove_btn);
          }
          points_col = points_col.push(point_row);
        }
        if *auto_create {
          let mut add_btn = button(text("+ point").size(11)).padding([1, 6]);
          if enabled {
            add_btn = add_btn
              .on_press(Message::ManipulateLocatorAdded(cell_idx, ctrl_idx));
          }
          points_col = points_col.push(row![add_btn]);
        }
        let control_row =
          row![label_widget, points_col].align_y(Center).spacing(8);
        controls_col = controls_col.push(control_row);
      }
      manipulate::ControlState::Trigger {
        name,
        label,
        label_runs,
        current,
        ..
      } => {
        // A Trigger control: its own play/pause toggle plus a live readout
        // of the swept variable (Wolfram's TriggerButton/PauseButton pair).
        let label_widget = manipulate_label_widget(
          label_runs,
          label,
          name,
          label_col_width,
          enabled,
        );
        let symbol = if state.playing { "❚❚" } else { "▶" };
        let play_btn =
          button(text(symbol).size(11))
            .padding([3, 10])
            .on_press(if enabled {
              Message::ManipulateTogglePlay(cell_idx)
            } else {
              Message::Noop
            });
        let value_widget = text(format_manipulate_number(*current))
          .size(11)
          .font(Font::MONOSPACE)
          .width(iced::Length::Fixed(64.0));
        let control_row = row![label_widget, play_btn, value_widget]
          .align_y(Center)
          .spacing(8);
        controls_col = controls_col.push(control_row);
      }
      manipulate::ControlState::Button {
        label, label_runs, ..
      } => {
        // A pressable action row (`Button["reset", …]`).
        let spans: Vec<iced::widget::text::Span<Message>> = label_runs
          .iter()
          .map(|run| {
            let mut font = Font::MONOSPACE;
            if run.italic {
              font.style = iced::font::Style::Italic;
            }
            iced::widget::span(run.text.clone()).font(font)
          })
          .collect();
        let btn_label: Element<Message> = if spans.is_empty() {
          text(label.clone()).size(11).font(Font::MONOSPACE).into()
        } else {
          rich_text(spans).size(11).into()
        };
        let btn = button(btn_label).padding([3, 10]).on_press(if enabled {
          Message::ManipulateButtonPressed(cell_idx, ctrl_idx)
        } else {
          Message::Noop
        });
        controls_col = controls_col.push(row![btn].align_y(Center));
      }
      manipulate::ControlState::Heading { label, label_runs } => {
        // A static heading row (a string or `Style[…]` Manipulate argument,
        // e.g. "signal 1"). Rendered bold across the full row.
        let spans: Vec<iced::widget::text::Span<Message>> = label_runs
          .iter()
          .map(|run| {
            let mut font = Font::MONOSPACE;
            font.weight = iced::font::Weight::Bold;
            if run.italic {
              font.style = iced::font::Style::Italic;
            }
            iced::widget::span(run.text.clone()).font(font)
          })
          .collect();
        let heading: Element<Message> = if spans.is_empty() {
          let mut font = Font::MONOSPACE;
          font.weight = iced::font::Weight::Bold;
          text(label.clone()).size(12).font(font).into()
        } else {
          rich_text(spans).size(12).into()
        };
        controls_col = controls_col.push(heading);
      }
      manipulate::ControlState::Divider => {
        // A `Delimiter` argument: a horizontal separator between control
        // groups.
        controls_col = controls_col.push(rule::horizontal(1));
      }
    }
  }

  // An animated widget (Animate / ListAnimate / Animator) gets a play/pause
  // toggle that starts in the playing state (Wolfram's default
  // AnimationRunning -> True). It stays visible under Appearance -> None so
  // the animation can still be paused. A Trigger control renders its own
  // toggle in its row, so the widget-level one would be redundant.
  if state.animated && !(show_controls && state.has_trigger()) {
    let symbol = if state.playing { "❚❚" } else { "▶" };
    let play_btn = button(text(symbol).size(11))
      .padding([3, 10])
      .on_press(Message::ManipulateTogglePlay(cell_idx));
    controls_col = controls_col.push(row![play_btn].align_y(Center));
  }

  let mut output_col = Column::new().spacing(0).width(Fill);
  if let Some(ref err) = state.error {
    let color =
      Color::from_rgba(0.85, 0.25, 0.25, if stale { 0.4 } else { 1.0 });
    output_col = output_col.push(
      container(
        text(err.clone())
          .size(12)
          .font(Font::MONOSPACE)
          .color(color),
      )
      .padding(4)
      .width(Fill),
    );
  } else if let Some(ref handle) = state.graphics_handle {
    // Render via the iced `svg` widget (not a pre-rasterized bitmap): its
    // vector cache uploads synchronously, so each re-evaluation's new handle
    // is drawn the same frame instead of flashing blank through iced's async
    // raster-upload path. See `ManipulateState::reevaluate`.
    let mut svg_widget =
      svg::Svg::new(handle.clone()).width(iced::Length::Shrink);
    if stale {
      svg_widget = svg_widget.opacity(0.3_f32);
    }
    output_col = output_col.push(container(svg_widget).padding(4));
  } else if let Some(ref txt) = state.text_output {
    let mut output_text = text(txt.clone()).size(12).font(Font::MONOSPACE);
    if stale {
      output_text = output_text.color(Color::from_rgba(0.5, 0.5, 0.5, 0.5));
    }
    output_col = output_col.push(container(output_text).padding(6).width(Fill));
  }

  // Extra display elements (e.g. a Checkbox grid) sit above the rendered
  // body output; each interactive checkbox emits a write-back on toggle.
  let mut widget_col = column![controls_col].spacing(6);
  for tree in &state.display_trees {
    widget_col = widget_col.push(render_display_node(cell_idx, tree));
  }
  widget_col = widget_col.push(output_col);

  container(widget_col).padding(6).width(Fill).into()
}

/// Recursively render a Manipulate display-element widget tree into iced.
/// Interactive checkboxes emit `ManipulateDisplayToggled` with the write-back
/// assignment (`<target> = <on|off>`) to apply on toggle.
fn render_display_node<'a>(
  cell_idx: usize,
  node: &woxi::functions::graphics::DisplayNode,
) -> Element<'a, Message> {
  use woxi::functions::graphics::DisplayNode;
  match node {
    DisplayNode::Panel(child) => {
      container(render_display_node(cell_idx, child))
        .padding(6)
        .style(container::rounded_box)
        .into()
    }
    DisplayNode::Grid(rows) => {
      let mut col = Column::new().spacing(2);
      for row_cells in rows {
        let mut r = Row::new().spacing(2).align_y(Center);
        for cell in row_cells {
          r = r.push(render_display_node(cell_idx, cell));
        }
        col = col.push(r);
      }
      col.into()
    }
    DisplayNode::Column(children) => {
      let mut col = Column::new().spacing(4);
      for c in children {
        col = col.push(render_display_node(cell_idx, c));
      }
      col.into()
    }
    DisplayNode::Row(children) => {
      let mut r = Row::new().spacing(4).align_y(Center);
      for c in children {
        r = r.push(render_display_node(cell_idx, c));
      }
      r.into()
    }
    DisplayNode::Checkbox {
      target,
      checked,
      on,
      off,
    } => {
      let cb = checkbox(*checked);
      match target {
        Some(t) => {
          let assignment =
            format!("{} = {}", t, if *checked { off } else { on });
          cb.on_toggle(move |_| {
            Message::ManipulateDisplayToggled(cell_idx, assignment.clone())
          })
          .into()
        }
        // Non-interactive checkbox: rendered but not clickable.
        None => cb.into(),
      }
    }
    DisplayNode::Toggler {
      label,
      mutation,
      selected,
    } => {
      // One choice of a TogglerBar: a toggle button whose press adds or
      // removes its value from the bound list variable.
      let selected = *selected;
      let mutation = mutation.clone();
      button(render_display_node(cell_idx, label))
        .padding([2, 6])
        .style(move |theme: &iced::Theme, status| {
          let mut style = if selected {
            button::primary(theme, status)
          } else {
            button::secondary(theme, status)
          };
          style.border.radius = 4.0.into();
          style
        })
        .on_press(Message::ManipulateDisplayToggled(cell_idx, mutation))
        .into()
    }
    DisplayNode::Static {
      svg: svg_src,
      text: txt,
    } => {
      if let Some(svg_str) = svg_src {
        let handle = svg::Handle::from_memory(svg_str.clone().into_bytes());
        svg::Svg::new(handle).width(iced::Length::Shrink).into()
      } else {
        text(txt.clone()).size(12).font(Font::MONOSPACE).into()
      }
    }
  }
}

/// Build a clickable hyperlink button: blue label, transparent
/// background, opens `uri` in the default browser on press. Stale
/// state dims the button to match other output widgets.
fn render_hyperlink<'a>(
  label: &str,
  uri: &str,
  stale: bool,
) -> Element<'a, Message> {
  let alpha = if stale { 0.4 } else { 1.0 };
  let link_color = Color::from_rgba(0.10, 0.45, 0.91, alpha);
  let label_text = text(label.to_string())
    .size(13)
    .color(link_color)
    .font(Font::MONOSPACE);
  button(label_text)
    .on_press(Message::OpenHyperlink(uri.to_string()))
    .padding([2, 6])
    .style(move |_theme, status| hyperlink_button_style(status, alpha))
    .into()
}

/// Render the graphical audio player shown for cells whose result is
/// playable audio (Play[…] / Sound[…] / Audio[…]): a play/pause toggle
/// button next to the audio's label (the source file name for file-backed
/// Audio objects). While the cell's audio is playing the button shows a
/// pause icon; pressing it pauses playback and reverts it to a play icon.
fn render_audio_player<'a>(
  idx: usize,
  audio: &woxi::AudioOutput,
  is_playing: bool,
) -> Element<'a, Message> {
  let icon = if is_playing { "⏸" } else { "▶" };
  let play = button(text(icon).size(14))
    .on_press(Message::PlaySound(idx))
    .padding([4, 10]);
  let label = audio.label.clone().unwrap_or_else(|| String::from("Sound"));
  let mut info = column![text(label).size(13).font(Font::MONOSPACE)];
  if audio.base64.is_empty() {
    // File-backed audio whose bytes could not be read — keep the player
    // chrome and explain why pressing play will not work.
    info = info.push(text("audio file could not be read").size(11));
  }
  let player = row![play, info].spacing(10).align_y(Center);
  container(player)
    .padding(8)
    .style(audio_player_style)
    .into()
}

/// Style the audio player card: a subtly bordered rounded container so the
/// player reads as one widget rather than a lone button.
fn audio_player_style(theme: &Theme) -> container::Style {
  let is_dark = !matches!(theme, Theme::Light);
  let (bg, border) = if is_dark {
    (
      Color::from_rgb(0.14, 0.14, 0.16),
      Color::from_rgb(0.30, 0.30, 0.34),
    )
  } else {
    (
      Color::from_rgb(0.96, 0.96, 0.97),
      Color::from_rgb(0.80, 0.80, 0.83),
    )
  };
  container::Style {
    background: Some(Background::Color(bg)),
    border: Border {
      color: border,
      width: 1.0,
      radius: 6.0.into(),
    },
    ..container::Style::default()
  }
}

/// Style the hyperlink button: borderless, transparent background,
/// subtle hover/press tint that doesn't overpower the link color.
fn hyperlink_button_style(status: button::Status, alpha: f32) -> button::Style {
  let bg = match status {
    button::Status::Hovered => Some(Background::Color(Color::from_rgba(
      0.10,
      0.45,
      0.91,
      0.10 * alpha,
    ))),
    button::Status::Pressed => Some(Background::Color(Color::from_rgba(
      0.10,
      0.45,
      0.91,
      0.18 * alpha,
    ))),
    _ => None,
  };
  button::Style {
    background: bg,
    text_color: Color::from_rgba(0.10, 0.45, 0.91, alpha),
    border: iced::Border::default(),
    shadow: Default::default(),
    snap: false,
  }
}

/// Format a slider value for the inline readout. Integers render
/// without a trailing zero, fractional values get 3 decimal digits of
/// precision (with trailing zeros trimmed).
fn format_manipulate_number(v: f64) -> String {
  if !v.is_finite() {
    return format!("{v}");
  }
  if v.fract() == 0.0 && v.abs() < 1e15 {
    return format!("{}", v as i64);
  }
  let s = format!("{:.3}", v);
  // Trim trailing zeros and a lone decimal point.
  let trimmed = s.trim_end_matches('0').trim_end_matches('.');
  if trimmed.is_empty() {
    "0".to_string()
  } else {
    trimmed.to_string()
  }
}

/// Build a display-only editor for a stored Output cell the interpreter
/// cannot regenerate: a Demonstrations snapshot (`RasterBox[
/// CompressedData["…"]]`) shows as a graphic, a `CheckboxBox[…]` grid as
/// read-only text lines. The original box text stays in `content` so
/// saving round-trips it. `None` for outputs with no displayable form.
fn stored_output_editor(cell: &Cell) -> Option<CellEditor> {
  let svg =
    woxi::notebook::stored_output_image_svg(&cell.content).or_else(|| {
      woxi::notebook::stored_output_checkbox_text(&cell.content)
        .map(|text| plain_text_svg(&text))
    })?;
  // The svg handle is a fallback for when rasterization fails; the normal
  // path shows the pre-rasterized image.
  let handle = svg::Handle::from_memory(svg.clone().into_bytes());
  Some(CellEditor {
    content: text_editor::Content::with_text(&cell.content),
    style: cell.style,
    output: None,
    stdout: None,
    graphics_svg: Some(svg),
    graphics_handle: Some(handle),
    graphics_image: None,
    output_svgs: Vec::new(),
    output_images: Vec::new(),
    output_dark: false,
    output_all_svg: false,
    sound: None,
    warnings: Vec::new(),
    undo_stack: Vec::new(),
    redo_stack: Vec::new(),
    output_stale: false,
    is_collapsed: cell.collapsed,
    manipulate_state: None,
    hyperlinks: Vec::new(),
    stored_graphic: true,
    output_content: text_editor::Content::new(),
    stdout_content: text_editor::Content::new(),
  })
}

/// A minimal SVG rendering of plain text lines (used for stored outputs
/// like checkbox grids), on a white card so it reads the same in both
/// themes.
fn plain_text_svg(text: &str) -> String {
  let lines: Vec<&str> = text.lines().collect();
  let char_w = 8.4_f64;
  let line_h = 20.0_f64;
  let width = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0) as f64
    * char_w
    + 16.0;
  let height = lines.len() as f64 * line_h + 12.0;
  let mut svg = format!(
    "<svg width=\"{width:.0}\" height=\"{height:.0}\" viewBox=\"0 0 \
     {width:.0} {height:.0}\" xmlns=\"http://www.w3.org/2000/svg\">\n\
     <rect width=\"{width:.0}\" height=\"{height:.0}\" fill=\"white\"/>\n"
  );
  for (i, line) in lines.iter().enumerate() {
    let escaped = line
      .replace('&', "&amp;")
      .replace('<', "&lt;")
      .replace('>', "&gt;");
    let y = 6.0 + i as f64 * line_h + 14.0;
    svg.push_str(&format!(
      "<text x=\"8\" y=\"{y:.0}\" font-family=\"monospace\" \
       font-size=\"14\" fill=\"#333\">{escaped}</text>\n"
    ));
  }
  svg.push_str("</svg>");
  svg
}

fn rasterize_svg(
  svg_str: &str,
  scale_factor: f32,
  fontdb: &Arc<resvg::usvg::fontdb::Database>,
) -> Option<(iced::widget::image::Handle, u32, u32)> {
  let opts = resvg::usvg::Options {
    fontdb: fontdb.clone(),
    ..Default::default()
  };
  let tree = resvg::usvg::Tree::from_str(svg_str, &opts).ok()?;
  let size = tree.size();
  let logical_w = size.width().ceil() as u32;
  let logical_h = size.height().ceil() as u32;
  if logical_w == 0 || logical_h == 0 {
    return None;
  }
  let physical_w = (logical_w as f32 * scale_factor).ceil() as u32;
  let physical_h = (logical_h as f32 * scale_factor).ceil() as u32;
  let mut pixmap = tiny_skia::Pixmap::new(physical_w, physical_h)?;
  let transform = tiny_skia::Transform::from_scale(scale_factor, scale_factor);
  resvg::render(&tree, transform, &mut pixmap.as_mut());
  let handle = iced::widget::image::Handle::from_rgba(
    physical_w,
    physical_h,
    pixmap.take(),
  );
  Some((handle, logical_w, logical_h))
}

/// Build the result-output element from pre-rasterized typeset SVG images,
/// stacked one per result statement. Each image is displayed at its logical
/// (unscaled) size so it stays crisp on HiDPI, matching the graphics output.
fn output_images_element<'a>(
  images: &'a [(iced::widget::image::Handle, u32, u32)],
  stale: bool,
) -> Element<'a, Message> {
  let mut col = Column::new().spacing(2);
  for (handle, w, h) in images {
    let mut img = image(handle.clone())
      .width(iced::Length::Fixed(*w as f32))
      .height(iced::Length::Fixed(*h as f32));
    if stale {
      img = img.opacity(0.3_f32);
    }
    col = col.push(container(img).padding(6));
  }
  col.into()
}

// ── Cell evaluation ─────────────────────────────────────────────────

/// File extension for the temp file holding decoded audio, derived from the
/// audio's MIME type so the system player recognizes the format.
fn audio_file_extension(mime: &str) -> &'static str {
  match mime {
    "audio/wav" => "wav",
    "audio/flac" => "flac",
    "audio/mpeg" => "mp3",
    "audio/ogg" => "ogg",
    "audio/mp4" => "m4a",
    "audio/aac" => "aac",
    "audio/aiff" => "aiff",
    _ => "bin",
  }
}

/// Send a signal (e.g. "-STOP" to pause, "-CONT" to resume) to the audio
/// player process via the standard `kill` utility, avoiding any extra
/// dependencies.
#[cfg(unix)]
fn signal_playback(
  child: &std::process::Child,
  signal: &str,
) -> Result<(), String> {
  let status = std::process::Command::new("kill")
    .arg(signal)
    .arg(child.id().to_string())
    .status()
    .map_err(|e| e.to_string())?;
  if status.success() {
    Ok(())
  } else {
    Err(format!("kill {signal} exited with {status}"))
  }
}

/// Decode base64 audio and play it through the operating system's audio
/// player. The bytes are written to a temp file and a platform-appropriate
/// player is spawned (non-blocking). Returns the player process handle so
/// playback can be paused, resumed, or stopped; an error string on failure.
fn play_audio(
  audio: &woxi::AudioOutput,
) -> Result<std::process::Child, String> {
  use base64::Engine;
  if audio.base64.is_empty() {
    return Err(match &audio.label {
      Some(label) => format!("audio file could not be read: {label}"),
      None => String::from("no audio data available"),
    });
  }
  let bytes = base64::engine::general_purpose::STANDARD
    .decode(&audio.base64)
    .map_err(|e| e.to_string())?;

  let mut path = std::env::temp_dir();
  path.push(format!(
    "woxi-studio-sound.{}",
    audio_file_extension(&audio.mime)
  ));
  std::fs::write(&path, &bytes).map_err(|e| e.to_string())?;

  #[cfg(target_os = "macos")]
  let result = std::process::Command::new("afplay").arg(&path).spawn();

  #[cfg(target_os = "windows")]
  let result = if audio.mime == "audio/wav" {
    std::process::Command::new("powershell")
      .args([
        "-NoProfile",
        "-Command",
        &format!(
          "(New-Object Media.SoundPlayer '{}').PlaySync()",
          path.display()
        ),
      ])
      .spawn()
  } else {
    // Media.SoundPlayer only decodes WAV — hand other formats to the
    // system's default audio player.
    std::process::Command::new("cmd")
      .args(["/C", "start", "", &path.display().to_string()])
      .spawn()
  };

  #[cfg(all(unix, not(target_os = "macos")))]
  let result = std::process::Command::new("paplay")
    .arg(&path)
    .spawn()
    .or_else(|_| std::process::Command::new("aplay").arg(&path).spawn())
    .or_else(|_| std::process::Command::new("xdg-open").arg(&path).spawn());

  result.map_err(|e| e.to_string())
}

/// Evaluate all statements in a cell and collect their results.
/// When a cell contains multiple newline-separated expressions,
/// each expression's output is included (matching Mathematica behavior).
/// Whether a stored Output cell holds a FrontEnd dynamic-widget dump — the
/// `DynamicModuleBox[…]` box form Mathematica saves for a live Manipulate.
/// Such text is meaningless outside the Wolfram FrontEnd.
fn is_dynamic_box_dump(output: &str) -> bool {
  let t = output.trim_start();
  t.starts_with("DynamicModuleBox[")
    || t.starts_with("TagBox[DynamicModuleBox[")
    || t.starts_with("DynamicBox[")
}

/// Evaluate (and drain) the Input-cell code accumulated ahead of a stored
/// interactive widget, so helper definitions from the notebook's
/// initialization cells (e.g. the Wolfram Demonstrations "Initialization
/// Code" section) are in scope when the widget's body evaluates. Results
/// and errors are discarded — the cells keep their stored outputs until
/// the user explicitly evaluates them.
fn evaluate_pending_initialization(pending: &mut Vec<String>) {
  for code in pending.drain(..) {
    for stmt in woxi::split_into_statements(&code) {
      let _ = woxi::interpret_with_stdout(&stmt);
    }
  }
}

/// Rebuild the interactive widget for a loaded Input cell whose stored
/// output was a dynamic-widget dump. Only a cell that is exactly one held
/// interactive expression (`Manipulate[…]`, `Animate[…]`, …) is
/// instantiated on load; anything else (definitions, side effects) waits
/// for an explicit evaluation.
fn instantiate_stored_manipulate(
  code: &str,
  stored_output: &str,
) -> Option<manipulate::ManipulateState> {
  let statements = woxi::split_into_statements(code);
  if statements.len() != 1 {
    return None;
  }
  // `Manipulate[…, SaveDefinitions -> True]` embeds the definitions its
  // body depends on in the stored output's Initialization. Run them once
  // (Wolfram's SynchronousInitialization) before instantiating, so the
  // widget works right when the notebook opens — before any of the
  // notebook's definition cells have been evaluated.
  if let Some(init) =
    woxi::notebook::extract_saved_initialization(stored_output)
  {
    let _ = woxi::interpret(&init);
  }
  let expr = woxi::interpret_to_expr(&statements[0]).ok()?;
  manipulate::ManipulateState::from_expr(&expr)
}

fn evaluate_cell_statements(
  editor: &mut CellEditor,
  code: &str,
  is_dark: bool,
  scale_factor: f32,
  fontdb: &Arc<resvg::usvg::fontdb::Database>,
) {
  // Render result SVGs (and any graphics) with theme-appropriate colors so the
  // typeset output reads against the current background.
  woxi::set_dark_mode(is_dark);

  let statements = woxi::split_into_statements(code);

  let mut outputs: Vec<String> = Vec::new();
  // Typeset SVG per result statement (the same rendering the Playground shows),
  // plus a count of result statements so the view can prefer the SVGs only when
  // every result produced one.
  let mut output_svgs: Vec<String> = Vec::new();
  let mut result_count = 0usize;
  let mut all_stdout = String::new();
  let mut last_graphics: Option<String> = None;
  let mut last_sound: Option<woxi::AudioOutput> = None;
  let mut all_warnings: Vec<String> = Vec::new();
  let mut had_error = false;
  // Track a Manipulate that appears as the final statement's result, so
  // we can render it as an interactive widget instead of a plain echo.
  let mut last_manipulate: Option<(String, manipulate::ManipulateState)> = None;
  let mut hyperlinks: Vec<(String, String)> = Vec::new();

  for stmt in &statements {
    match woxi::interpret_with_stdout(stmt) {
      Ok(result) => {
        if !result.stdout.is_empty() {
          all_stdout.push_str(&result.stdout);
        }
        all_warnings.extend(result.warnings);

        // Detect a top-level Manipulate[…] result by re-parsing the
        // statement to inspect the held Expr. Each new Manipulate in
        // the cell replaces any previous one so only the final
        // statement's interactive widget is shown.
        if result.result != "\0"
          && let Ok(expr) = woxi::interpret_to_expr(stmt)
          && let Some(state) = manipulate::ManipulateState::from_expr(&expr)
        {
          last_manipulate = Some((result.result.clone(), state));
          // Skip adding to outputs / graphics — the interactive widget
          // subsumes both the text echo and any placeholder graphics.
          last_graphics = None;
          continue;
        }

        // Detect a top-level Hyperlink[…] result so the cell can
        // render a clickable link button instead of plain text.
        if result.result != "\0"
          && let Ok(expr) = woxi::interpret_to_expr(stmt)
          && let Some((label, uri)) = extract_hyperlink(&expr)
        {
          hyperlinks.push((label, uri));
          continue;
        }

        if let Some(svg) = result.graphics
          && result.result != "\0"
        {
          last_graphics = Some(svg);
        }

        if let Some(audio) = result.sound
          && result.result != "\0"
        {
          last_sound = Some(audio);
        }

        if result.result != "\0" {
          result_count += 1;
          if let Some(svg) = result.output_svg {
            output_svgs.push(svg);
          }
          outputs.push(result.result);
        }
      }
      Err(woxi::InterpreterError::EmptyInput) => {}
      Err(e) => {
        result_count += 1;
        outputs.push(format!("Error: {e}"));
        had_error = true;
      }
    }
  }

  editor.output = if outputs.is_empty() {
    None
  } else {
    // Show notebook OutputForm: truncate arbitrary-precision reals to their
    // precision and drop the backtick marker (`N[Pi, 3]` → `3.14`). The CLI
    // keeps the full backtick InputForm; this is a display-layer transform.
    Some(woxi::truncate_precision_reals(&outputs.join("\n")))
  };
  editor.output_content = match &editor.output {
    Some(s) => {
      let display = s
        .replace("-Graphics-", "")
        .replace("-Graphics3D-", "")
        .replace("-Image-", "")
        .replace("-Sound-", "")
        .replace("-Audio-", "");
      let display = display.trim();
      if display.is_empty() {
        text_editor::Content::new()
      } else {
        text_editor::Content::with_text(display)
      }
    }
    None => text_editor::Content::new(),
  };
  editor.stdout = if all_stdout.is_empty() {
    None
  } else {
    Some(all_stdout)
  };
  editor.stdout_content = match &editor.stdout {
    Some(s) => text_editor::Content::with_text(s),
    None => text_editor::Content::new(),
  };
  // Typeset SVG output: rasterize each result SVG. The view uses these images
  // (instead of the plain text) only when every result produced one and the
  // theme still matches — see `output_all_svg` / `output_dark`.
  editor.output_dark = is_dark;
  editor.output_images = output_svgs
    .iter()
    .filter_map(|s| rasterize_svg(s, scale_factor, fontdb))
    .collect();
  // Require one image per result; a rasterization failure falls the whole cell
  // back to text so no result is silently dropped.
  editor.output_all_svg = result_count > 0
    && output_svgs.len() == result_count
    && editor.output_images.len() == output_svgs.len();
  editor.output_svgs = output_svgs;
  editor.sound = last_sound;
  editor.graphics_svg = last_graphics;
  editor.graphics_handle = editor
    .graphics_svg
    .as_ref()
    .map(|s| svg::Handle::from_memory(s.as_bytes().to_vec()));
  editor.graphics_image = editor
    .graphics_svg
    .as_ref()
    .and_then(|s| rasterize_svg(s, scale_factor, fontdb));
  editor.manipulate_state = last_manipulate.map(|(_, state)| state);
  editor.hyperlinks = hyperlinks;
  editor.warnings = all_warnings;
  editor.output_stale = false;
  let _ = had_error;
}

/// Extract `(label, uri)` from a top-level `Hyperlink[…]` expression.
/// Both `Hyperlink[uri]` and `Hyperlink[label, uri]` are accepted, with
/// the URI required to be a literal string. Returns `None` for any
/// other shape.
fn extract_hyperlink(expr: &woxi::syntax::Expr) -> Option<(String, String)> {
  let woxi::syntax::Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "Hyperlink" {
    return None;
  }
  match args.as_ref() {
    [woxi::syntax::Expr::String(uri)] => Some((uri.clone(), uri.clone())),
    [label, woxi::syntax::Expr::String(uri)] => {
      let label_str = match label {
        woxi::syntax::Expr::String(s) => s.clone(),
        other => woxi::syntax::expr_to_string(other),
      };
      Some((label_str, uri.clone()))
    }
    _ => None,
  }
}

/// Open `url` in the user's default browser. The command varies per
/// platform; failures are silently ignored (the worst case is a no-op
/// click, which is acceptable for a UI affordance).
fn open_url(url: &str) {
  #[cfg(target_os = "macos")]
  let cmd = "open";
  #[cfg(target_os = "linux")]
  let cmd = "xdg-open";
  #[cfg(target_os = "windows")]
  let cmd = "start";
  let _ = std::process::Command::new(cmd).arg(url).spawn();
}

// ── Custom styles ───────────────────────────────────────────────────

fn toc_panel_style(theme: &Theme) -> container::Style {
  let is_dark = !matches!(theme, Theme::Light);
  container::Style {
    background: Some(Background::Color(if is_dark {
      Color::from_rgb(0.12, 0.12, 0.14)
    } else {
      Color::from_rgb(0.95, 0.95, 0.96)
    })),
    ..Default::default()
  }
}

fn toc_entry_style(theme: &Theme, status: button::Status) -> button::Style {
  let is_dark = !matches!(theme, Theme::Light);
  let text_color = if is_dark {
    Color::from_rgb(0.78, 0.78, 0.82)
  } else {
    Color::from_rgb(0.15, 0.15, 0.20)
  };
  match status {
    button::Status::Hovered | button::Status::Pressed => button::Style {
      background: Some(Background::Color(if is_dark {
        Color::from_rgb(0.22, 0.22, 0.26)
      } else {
        Color::from_rgb(0.88, 0.88, 0.92)
      })),
      text_color,
      border: Border::default().rounded(4),
      ..Default::default()
    },
    _ => button::Style {
      background: None,
      text_color,
      border: Border::default().rounded(4),
      ..Default::default()
    },
  }
}

fn separator_style(theme: &Theme) -> rule::Style {
  let is_dark = !matches!(theme, Theme::Light);
  rule::Style {
    color: if is_dark {
      Color::from_rgb(0.22, 0.22, 0.25)
    } else {
      Color::from_rgb(0.82, 0.82, 0.82)
    },
    radius: 0.0.into(),
    fill_mode: rule::FillMode::Full,
    snap: true,
  }
}

fn editor_style(
  theme: &Theme,
  status: text_editor::Status,
) -> text_editor::Style {
  let mut style = text_editor::default(theme, status);
  style.border.radius = 6.0.into();
  let is_dark = !matches!(theme, Theme::Light);
  if is_dark {
    style.border.color = Color::from_rgb(0.22, 0.22, 0.25);
    style.background = Background::Color(Color::from_rgb(0.20, 0.20, 0.23));
    if matches!(status, text_editor::Status::Focused { .. }) {
      style.border.color = Color::from_rgb(0.30, 0.30, 0.38);
    }
  } else {
    style.background = Background::Color(Color::from_rgb(0.98, 0.98, 0.99));
    style.border.color = Color::from_rgb(0.82, 0.82, 0.85);
    if matches!(status, text_editor::Status::Focused { .. }) {
      style.border.color = Color::from_rgb(0.55, 0.55, 0.65);
    }
  }
  style
}

fn cell_editor_style(
  theme: &Theme,
  status: text_editor::Status,
  cell_style: CellStyle,
) -> text_editor::Style {
  let mut style = editor_style(theme, status);
  let is_dark = !matches!(theme, Theme::Light);
  let is_heading = matches!(
    cell_style,
    CellStyle::Title
      | CellStyle::Subtitle
      | CellStyle::Chapter
      | CellStyle::Subchapter
      | CellStyle::Section
      | CellStyle::Subsection
      | CellStyle::Subsubsection
      | CellStyle::Text
      | CellStyle::Item
      | CellStyle::Subitem
  );
  if is_heading {
    let bg = if is_dark {
      Color::from_rgb(0.12, 0.12, 0.14)
    } else {
      Color::WHITE
    };
    style.background = Background::Color(bg);
    style.border = Border {
      color: Color::TRANSPARENT,
      width: 0.0,
      radius: 0.0.into(),
    };
  }
  match cell_style {
    CellStyle::Title => {
      style.value = if is_dark {
        Color::from_rgb(0.92, 0.45, 0.28)
      } else {
        Color::from_rgb(0.78, 0.30, 0.15)
      };
    }
    CellStyle::Subtitle => {
      style.value = if is_dark {
        Color::from_rgb(0.90, 0.60, 0.25)
      } else {
        Color::from_rgb(0.75, 0.48, 0.10)
      };
    }
    _ => {}
  }
  style
}

fn graphics_modal_backdrop_style(_theme: &Theme) -> container::Style {
  container::Style {
    background: Some(Background::Color(Color::from_rgba(0.0, 0.0, 0.0, 0.75))),
    ..container::Style::default()
  }
}

fn context_menu_style(theme: &Theme) -> container::Style {
  let is_dark = !matches!(theme, Theme::Light);
  let (bg, border) = if is_dark {
    (
      Color::from_rgb(0.20, 0.22, 0.28),
      Color::from_rgba(1.0, 1.0, 1.0, 0.15),
    )
  } else {
    (Color::WHITE, Color::from_rgba(0.0, 0.0, 0.0, 0.15))
  };
  container::Style {
    background: Some(Background::Color(bg)),
    border: Border {
      color: border,
      width: 1.0,
      radius: 6.0.into(),
    },
    ..container::Style::default()
  }
}

/// Style one segment of a Manipulate SetterBar (segmented toggle group).
/// The selected segment is filled blue with white text; the others are a
/// neutral surface with a hairline border. `index`/`count` decide which
/// outer corners are rounded so the row reads as a single pill.
fn setter_button_style(
  theme: &Theme,
  status: button::Status,
  selected: bool,
  index: usize,
  count: usize,
  enabled: bool,
) -> button::Style {
  use iced::border::Radius;
  let is_dark = !matches!(theme, Theme::Light);
  // A disabled bar keeps the selected segment marked but drained of accent so
  // it reads as inactive; hover has no effect since it takes no input.
  let accent = if enabled {
    Color::from_rgb(0.26, 0.52, 0.96)
  } else {
    Color::from_rgba(0.26, 0.52, 0.96, 0.4)
  };
  let accent_hover = Color::from_rgb(0.30, 0.56, 0.98);

  // Round only the outer corners of the first and last segment.
  let r = 6.0;
  let first = index == 0;
  let last = index + 1 == count;
  let radius = Radius {
    top_left: if first { r } else { 0.0 },
    bottom_left: if first { r } else { 0.0 },
    top_right: if last { r } else { 0.0 },
    bottom_right: if last { r } else { 0.0 },
  };

  let (idle_bg, idle_text, border_color) = if is_dark {
    (
      Color::from_rgb(0.20, 0.21, 0.24),
      Color::from_rgb(0.85, 0.87, 0.92),
      Color::from_rgba(1.0, 1.0, 1.0, 0.18),
    )
  } else {
    (
      Color::from_rgb(0.97, 0.97, 0.98),
      Color::from_rgb(0.15, 0.15, 0.18),
      Color::from_rgba(0.0, 0.0, 0.0, 0.18),
    )
  };

  let hovered = matches!(status, button::Status::Hovered);
  let (background, text_color) = if selected {
    (if hovered { accent_hover } else { accent }, Color::WHITE)
  } else if hovered {
    let hb = if is_dark {
      Color::from_rgb(0.26, 0.27, 0.31)
    } else {
      Color::from_rgb(0.92, 0.93, 0.95)
    };
    (hb, idle_text)
  } else {
    (idle_bg, idle_text)
  };

  button::Style {
    background: Some(Background::Color(background)),
    text_color,
    border: Border {
      color: if selected { accent } else { border_color },
      width: 1.0,
      radius,
    },
    ..button::Style::default()
  }
}

/// Greyed style for a Manipulate slider whose control is currently disabled
/// (its `Enabled` condition is `False`): the rail and handle drop to muted
/// surface colors so the widget reads as inactive.
fn disabled_slider_style(
  theme: &Theme,
  _status: iced::widget::slider::Status,
) -> iced::widget::slider::Style {
  use iced::widget::slider::{Handle, HandleShape, Rail, Style};
  let palette = theme.extended_palette();
  let muted = palette.background.strong.color;
  Style {
    rail: Rail {
      backgrounds: (muted.into(), palette.background.weak.color.into()),
      width: 4.0,
      border: Border {
        radius: 2.0.into(),
        width: 0.0,
        color: Color::TRANSPARENT,
      },
    },
    handle: Handle {
      shape: HandleShape::Circle { radius: 7.0 },
      background: muted.into(),
      border_color: Color::TRANSPARENT,
      border_width: 0.0,
    },
  }
}

fn context_menu_item_style(
  theme: &Theme,
  status: button::Status,
) -> button::Style {
  let is_dark = !matches!(theme, Theme::Light);
  let (text_color, hover_bg) = if is_dark {
    (
      Color::from_rgb(0.88, 0.90, 0.95),
      Color::from_rgba(1.0, 1.0, 1.0, 0.08),
    )
  } else {
    (
      Color::from_rgb(0.10, 0.10, 0.10),
      Color::from_rgba(0.0, 0.0, 0.0, 0.06),
    )
  };
  let bg = match status {
    button::Status::Hovered | button::Status::Pressed => Some(hover_bg),
    _ => None,
  };
  button::Style {
    background: bg.map(Background::Color),
    text_color,
    border: Border {
      radius: 4.0.into(),
      ..Border::default()
    },
    ..button::Style::default()
  }
}

/// Style for read-only output text editors (selectable but not editable).
fn output_editor_style(
  theme: &Theme,
  _status: text_editor::Status,
  stale: bool,
) -> text_editor::Style {
  let is_dark = !matches!(theme, Theme::Light);
  let bg = if is_dark {
    Color::from_rgb(0.14, 0.14, 0.16)
  } else {
    Color::from_rgb(0.97, 0.97, 0.98)
  };
  let value = if stale {
    Color::from_rgba(0.5, 0.5, 0.5, 0.5)
  } else if is_dark {
    Color::from_rgb(0.85, 0.85, 0.88)
  } else {
    Color::from_rgb(0.15, 0.15, 0.15)
  };
  text_editor::Style {
    background: Background::Color(bg),
    border: Border {
      color: Color::TRANSPARENT,
      width: 0.0,
      radius: 0.0.into(),
    },
    placeholder: Color::TRANSPARENT,
    value,
    selection: if is_dark {
      Color::from_rgba(0.3, 0.5, 0.8, 0.3)
    } else {
      Color::from_rgba(0.3, 0.5, 0.8, 0.2)
    },
  }
}

fn output_area_style(_theme: &Theme) -> container::Style {
  container::Style {
    background: None,
    border: Border {
      color: Color::TRANSPARENT,
      width: 0.0,
      radius: 6.0.into(),
    },
    ..container::Style::default()
  }
}

fn preview_editor_style(
  theme: &Theme,
  _status: text_editor::Status,
  cell_style: CellStyle,
) -> text_editor::Style {
  let is_dark = !matches!(theme, Theme::Light);
  let bg = if is_dark {
    Color::from_rgb(0.12, 0.12, 0.14)
  } else {
    Color::WHITE
  };
  let mut style = text_editor::Style {
    background: Background::Color(bg),
    border: Border {
      color: Color::TRANSPARENT,
      width: 0.0,
      radius: 0.0.into(),
    },
    placeholder: Color::TRANSPARENT,
    value: if is_dark {
      Color::from_rgb(0.85, 0.85, 0.88)
    } else {
      Color::from_rgb(0.15, 0.15, 0.15)
    },
    selection: if is_dark {
      Color::from_rgba(0.3, 0.5, 0.8, 0.3)
    } else {
      Color::from_rgba(0.3, 0.5, 0.8, 0.2)
    },
  };
  match cell_style {
    CellStyle::Title => {
      style.value = if is_dark {
        Color::from_rgb(0.92, 0.45, 0.28)
      } else {
        Color::from_rgb(0.78, 0.30, 0.15)
      };
    }
    CellStyle::Subtitle => {
      style.value = if is_dark {
        Color::from_rgb(0.90, 0.60, 0.25)
      } else {
        Color::from_rgb(0.75, 0.48, 0.10)
      };
    }
    _ => {}
  }
  style
}

fn muted_button_style(theme: &Theme, status: button::Status) -> button::Style {
  let mut style = button::primary(theme, status);
  let is_dark = !matches!(theme, Theme::Light);
  if is_dark {
    style.background = Some(Background::Color(match status {
      button::Status::Active => Color::from_rgb(0.18, 0.26, 0.40),
      button::Status::Hovered => Color::from_rgb(0.22, 0.32, 0.48),
      button::Status::Pressed => Color::from_rgb(0.15, 0.22, 0.35),
      button::Status::Disabled => Color::from_rgb(0.14, 0.16, 0.22),
    }));
    style.text_color = Color::from_rgb(0.78, 0.82, 0.90);
  }
  style
}

fn trash_button_style(theme: &Theme, status: button::Status) -> button::Style {
  let mut style = button::text(theme, status);
  // Only show background on hover
  match status {
    button::Status::Hovered | button::Status::Pressed => {
      let is_dark = !matches!(theme, Theme::Light);
      style.background = Some(Background::Color(if is_dark {
        Color::from_rgba(1.0, 1.0, 1.0, 0.08)
      } else {
        Color::from_rgba(0.0, 0.0, 0.0, 0.06)
      }));
    }
    _ => {
      style.background = None;
    }
  }
  style
}

fn drag_handle_container_style(theme: &Theme) -> container::Style {
  let is_dark = !matches!(theme, Theme::Light);
  container::Style {
    background: Some(Background::Color(if is_dark {
      Color::from_rgba(1.0, 1.0, 1.0, 0.06)
    } else {
      Color::from_rgba(0.0, 0.0, 0.0, 0.04)
    })),
    border: Border {
      radius: 4.0.into(),
      ..Border::default()
    },
    ..container::Style::default()
  }
}

fn drop_indicator_style(theme: &Theme) -> rule::Style {
  let is_dark = !matches!(theme, Theme::Light);
  rule::Style {
    color: if is_dark {
      Color::from_rgb(0.35, 0.55, 0.95)
    } else {
      Color::from_rgb(0.25, 0.45, 0.85)
    },
    radius: 2.0.into(),
    fill_mode: rule::FillMode::Full,
    snap: true,
  }
}

fn dragged_cell_style(theme: &Theme) -> container::Style {
  let is_dark = !matches!(theme, Theme::Light);
  container::Style {
    background: Some(Background::Color(if is_dark {
      Color::from_rgba(1.0, 1.0, 1.0, 0.04)
    } else {
      Color::from_rgba(0.0, 0.0, 0.0, 0.03)
    })),
    border: Border {
      color: if is_dark {
        Color::from_rgba(0.35, 0.55, 0.95, 0.4)
      } else {
        Color::from_rgba(0.25, 0.45, 0.85, 0.3)
      },
      width: 1.0,
      radius: 4.0.into(),
    },
    ..container::Style::default()
  }
}

fn add_cell_button_style(
  theme: &Theme,
  status: button::Status,
) -> button::Style {
  let is_dark = !matches!(theme, Theme::Light);
  let text_color = match status {
    button::Status::Hovered | button::Status::Pressed => {
      if is_dark {
        Color::from_rgb(0.7, 0.7, 0.7)
      } else {
        Color::from_rgb(0.3, 0.3, 0.3)
      }
    }
    _ => {
      if is_dark {
        Color::from_rgb(0.45, 0.45, 0.45)
      } else {
        Color::from_rgb(0.6, 0.6, 0.6)
      }
    }
  };
  let background = match status {
    button::Status::Hovered | button::Status::Pressed => {
      Some(Background::Color(if is_dark {
        Color::from_rgba(1.0, 1.0, 1.0, 0.06)
      } else {
        Color::from_rgba(0.0, 0.0, 0.0, 0.04)
      }))
    }
    _ => None,
  };
  button::Style {
    text_color,
    background,
    border: Border {
      radius: 4.0.into(),
      ..Border::default()
    },
    ..button::text(theme, status)
  }
}

fn focused_add_cell_button_style(
  theme: &Theme,
  _status: button::Status,
) -> button::Style {
  let is_dark = !matches!(theme, Theme::Light);
  let text_color = if is_dark {
    Color::from_rgb(0.85, 0.85, 0.85)
  } else {
    Color::from_rgb(0.2, 0.2, 0.2)
  };
  let bg = if is_dark {
    Color::from_rgba(0.4, 0.6, 1.0, 0.2)
  } else {
    Color::from_rgba(0.2, 0.4, 0.8, 0.12)
  };
  let border_color = if is_dark {
    Color::from_rgba(0.4, 0.6, 1.0, 0.5)
  } else {
    Color::from_rgba(0.2, 0.4, 0.8, 0.4)
  };
  button::Style {
    text_color,
    background: Some(Background::Color(bg)),
    border: Border {
      radius: 4.0.into(),
      width: 1.0,
      color: border_color,
    },
    ..button::text(theme, _status)
  }
}

fn eval_all_icon_style(theme: &Theme, _status: svg::Status) -> svg::Style {
  let is_dark = !matches!(theme, Theme::Light);
  svg::Style {
    color: Some(if is_dark {
      Color::from_rgb(0.45, 0.78, 0.45)
    } else {
      Color::from_rgb(0.20, 0.55, 0.20)
    }),
  }
}

fn trash_icon_style(theme: &Theme, _status: svg::Status) -> svg::Style {
  let is_dark = !matches!(theme, Theme::Light);
  svg::Style {
    color: Some(if is_dark {
      Color::from_rgb(0.65, 0.65, 0.70)
    } else {
      Color::from_rgb(0.40, 0.40, 0.45)
    }),
  }
}

fn gutter_icon_style(theme: &Theme, _status: svg::Status) -> svg::Style {
  let is_dark = !matches!(theme, Theme::Light);
  svg::Style {
    color: Some(if is_dark {
      Color::from_rgb(0.65, 0.70, 0.78)
    } else {
      Color::from_rgb(0.35, 0.35, 0.40)
    }),
  }
}

fn export_button_style(
  theme: &Theme,
  status: pick_list::Status,
) -> pick_list::Style {
  let palette = theme.extended_palette();
  let bg = match status {
    pick_list::Status::Hovered | pick_list::Status::Opened { .. } => {
      palette.primary.strong.color
    }
    _ => palette.primary.base.color,
  };
  let text_color = Color::WHITE;
  pick_list::Style {
    text_color,
    placeholder_color: text_color,
    handle_color: text_color,
    background: Background::Color(bg),
    border: Border {
      color: bg,
      width: 1.0,
      radius: 2.0.into(),
    },
  }
}

fn dropdown_style(
  theme: &Theme,
  status: pick_list::Status,
) -> pick_list::Style {
  let mut style = pick_list::default(theme, status);
  style.border.radius = 6.0.into();
  let is_dark = !matches!(theme, Theme::Light);
  if is_dark {
    style.background = Background::Color(Color::from_rgb(0.14, 0.14, 0.16));
    style.border.color = Color::from_rgb(0.22, 0.22, 0.25);
    if matches!(
      status,
      pick_list::Status::Hovered | pick_list::Status::Opened { .. }
    ) {
      style.border.color = Color::from_rgb(0.30, 0.30, 0.38);
    }
  }
  style
}

fn dropdown_menu_style(theme: &Theme) -> menu::Style {
  let mut style = menu::default(theme);
  let is_dark = !matches!(theme, Theme::Light);
  if is_dark {
    style.background = Background::Color(Color::from_rgb(0.14, 0.14, 0.16));
    style.border.color = Color::from_rgb(0.22, 0.22, 0.25);
  }
  style
}

const TRASH_ICON_SVG: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M3 6h18"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>"#;

const PLAY_CIRCLE_SVG: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 9.003a1 1 0 0 1 1.517-.859l4.997 2.997a1 1 0 0 1 0 1.718l-4.997 2.997A1 1 0 0 1 9 14.996z"/><circle cx="12" cy="12" r="10"/></svg>"#;

const ICON_EYE: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0"/><circle cx="12" cy="12" r="3"/></svg>"#;

const ICON_EYE_OFF: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.733 5.076a10.744 10.744 0 0 1 11.205 6.575 1 1 0 0 1 0 .696 10.747 10.747 0 0 1-1.444 2.49"/><path d="M14.084 14.158a3 3 0 0 1-4.242-4.242"/><path d="M17.479 17.499a10.75 10.75 0 0 1-15.417-5.151 1 1 0 0 1 0-.696 10.75 10.75 0 0 1 4.446-5.143"/><path d="m2 2 20 20"/></svg>"#;

const ICON_TOC: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 12H3"/><path d="M16 18H3"/><path d="M16 6H3"/><path d="M21 12h.01"/><path d="M21 18h.01"/><path d="M21 6h.01"/></svg>"#;

const ICON_CHEVRON_DOWN: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>"#;
const ICON_CHEVRON_RIGHT: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m9 18 6-6-6-6"/></svg>"#;

const ICON_GRIP: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="9" cy="12" r="1"/><circle cx="9" cy="5" r="1"/><circle cx="9" cy="19" r="1"/><circle cx="15" cy="12" r="1"/><circle cx="15" cy="5" r="1"/><circle cx="15" cy="19" r="1"/></svg>"#;

// ── CellStyle display/picklist support ──────────────────────────────

/// Compute a boolean vector marking cells that should be hidden due
/// to a collapsed Chapter/Subchapter above them. A collapsed heading
/// hides all following cells until the next heading at the same
/// level or higher. The collapsed heading itself remains visible.
fn compute_hidden_cells_from_states(states: &[(CellStyle, bool)]) -> Vec<bool> {
  let mut hidden = vec![false; states.len()];
  // Stack of (heading index, heading level) for currently-active
  // collapsed regions.
  let mut stack: Vec<(usize, u8)> = Vec::new();
  for (i, &(style, is_collapsed)) in states.iter().enumerate() {
    // A new heading breaks out of any collapses at equal-or-lower
    // level (i.e. same-level or higher-priority heading).
    if let Some(level) = heading_level(style) {
      while let Some(&(_, top_level)) = stack.last() {
        if level <= top_level {
          stack.pop();
        } else {
          break;
        }
      }
    }
    if !stack.is_empty() {
      hidden[i] = true;
    }
    // If this cell is a collapsed Chapter/Subchapter, activate a
    // collapse region for subsequent cells.
    if is_collapsed
      && matches!(style, CellStyle::Chapter | CellStyle::Subchapter)
      && let Some(level) = heading_level(style)
    {
      stack.push((i, level));
    }
  }
  hidden
}

/// Creates a [`Task`] that scrolls the scrollable with `scrollable_id` so
/// that the focusable widget with `target_id` is at the top of the viewport.
///
/// Phase 1: traverse the widget tree and record the scrollable's viewport
/// bounds/translation and the target widget's screen bounds.
/// Phase 2 (chained via `Outcome::Chain`): `scroll_to` with the computed
/// absolute offset.
fn scroll_cell_into_view(
  scrollable_id: iced::widget::Id,
  target_id: iced::widget::Id,
) -> Task<Message> {
  use iced::advanced::widget::Operation;
  use iced::advanced::widget::operation;
  use iced::widget::operation::AbsoluteOffset;

  struct FindTarget {
    scrollable_id: iced::widget::Id,
    target_id: iced::widget::Id,
    scrollable_bounds_y: Option<f32>,
    target_bounds_y: Option<f32>,
  }

  impl Operation for FindTarget {
    fn traverse(&mut self, operate: &mut dyn FnMut(&mut dyn Operation)) {
      operate(self);
    }

    fn scrollable(
      &mut self,
      id: Option<&iced::widget::Id>,
      bounds: iced::Rectangle,
      _content_bounds: iced::Rectangle,
      _translation: iced::Vector,
      _state: &mut dyn operation::Scrollable,
    ) {
      if id == Some(&self.scrollable_id) {
        self.scrollable_bounds_y = Some(bounds.y);
      }
    }

    fn focusable(
      &mut self,
      id: Option<&iced::widget::Id>,
      bounds: iced::Rectangle,
      _state: &mut dyn operation::Focusable,
    ) {
      if id == Some(&self.target_id) {
        self.target_bounds_y = Some(bounds.y);
      }
    }

    fn finish(&self) -> operation::Outcome<()> {
      if let (Some(scroll_y), Some(target_y)) =
        (self.scrollable_bounds_y, self.target_bounds_y)
      {
        // Inside operate(), child bounds are in content-space
        // (not translated by scroll offset). So the content offset
        // of the target is simply its Y minus the scrollable's Y.
        let desired_offset = target_y - scroll_y;
        let id = self.scrollable_id.clone();
        operation::Outcome::Chain(Box::new(operation::scrollable::scroll_to(
          id,
          AbsoluteOffset {
            x: None,
            y: Some(desired_offset),
          },
        )))
      } else {
        operation::Outcome::None
      }
    }
  }

  iced::advanced::widget::operate(FindTarget {
    scrollable_id,
    target_id,
    scrollable_bounds_y: None,
    target_bounds_y: None,
  })
  .discard()
}

/// Heading level used for collapse/expand scoping. Lower numbers are
/// higher-level (Title is 0). Returns `None` for non-heading cells.
fn heading_level(style: CellStyle) -> Option<u8> {
  match style {
    CellStyle::Title => Some(0),
    CellStyle::Subtitle => Some(1),
    CellStyle::Chapter => Some(2),
    CellStyle::Subchapter => Some(3),
    CellStyle::Section => Some(4),
    CellStyle::Subsection => Some(5),
    CellStyle::Subsubsection => Some(6),
    _ => None,
  }
}

const CELL_STYLES: &[CellStyle] = &[
  CellStyle::Title,
  CellStyle::Subtitle,
  CellStyle::Chapter,
  CellStyle::Subchapter,
  CellStyle::Section,
  CellStyle::Subsection,
  CellStyle::Subsubsection,
  CellStyle::Text,
  CellStyle::Item,
  CellStyle::Subitem,
  CellStyle::Input,
  CellStyle::Output,
  CellStyle::Code,
];

// ── CLI argument parsing ─────────────────────────────────────────────

/// Returns the first positional (non-flag) argument as a file path, if any.
/// Relative paths are resolved against the current working directory.
fn parse_cli_file_arg() -> Option<PathBuf> {
  let arg = std::env::args().skip(1).find(|a| !a.starts_with('-'))?;
  let path = PathBuf::from(arg);
  // `woxi::utils::canonicalize` rather than `std::fs`'s so the window
  // title shows `C:\dir\notebook.nb` on Windows, not `\\?\C:\dir\…`.
  Some(woxi::utils::canonicalize(&path).unwrap_or(path))
}

// ── State persistence ────────────────────────────────────────────────

fn state_dir() -> Option<PathBuf> {
  let home = std::env::var("HOME").ok()?;
  Some(PathBuf::from(home).join(".config").join("woxi-studio"))
}

fn save_last_file_path(path: &std::path::Path) {
  if let Some(dir) = state_dir() {
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(dir.join("last_file"), path.display().to_string());
  }
}

fn load_last_file_path() -> Option<PathBuf> {
  let dir = state_dir()?;
  let content = std::fs::read_to_string(dir.join("last_file")).ok()?;
  let path = PathBuf::from(content.trim());
  if path.exists() { Some(path) } else { None }
}

// ── File I/O ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum FileError {
  DialogClosed,
  IoError(std::io::ErrorKind),
}

async fn open_file_path(
  path: PathBuf,
) -> Result<(PathBuf, Arc<String>), FileError> {
  let contents = tokio::fs::read_to_string(&path)
    .await
    .map(Arc::new)
    .map_err(|e| FileError::IoError(e.kind()))?;
  Ok((path, contents))
}

async fn open_file() -> Result<(PathBuf, Arc<String>), FileError> {
  let handle = rfd::AsyncFileDialog::new()
    .set_title("Open Notebook")
    .add_filter("Mathematica Notebook", &["nb"])
    .add_filter("All Files", &["*"])
    .pick_file()
    .await
    .ok_or(FileError::DialogClosed)?;

  let path = handle.path().to_owned();
  let contents = tokio::fs::read_to_string(&path)
    .await
    .map(Arc::new)
    .map_err(|e| FileError::IoError(e.kind()))?;

  Ok((path, contents))
}

async fn save_file(
  path: Option<PathBuf>,
  contents: String,
) -> Result<PathBuf, FileError> {
  let path = if let Some(path) = path {
    path
  } else {
    rfd::AsyncFileDialog::new()
      .set_title("Save Notebook")
      .add_filter("Mathematica Notebook", &["nb"])
      .save_file()
      .await
      .map(|h| h.path().to_owned())
      .ok_or(FileError::DialogClosed)?
  };

  tokio::fs::write(&path, &contents)
    .await
    .map_err(|e| FileError::IoError(e.kind()))?;

  Ok(path)
}

async fn export_file(
  filter_name: String,
  extension: String,
  contents: String,
) -> Result<PathBuf, FileError> {
  let path = rfd::AsyncFileDialog::new()
    .set_title("Export Notebook")
    .add_filter(&filter_name, &[&extension])
    .save_file()
    .await
    .map(|h| h.path().to_owned())
    .ok_or(FileError::DialogClosed)?;

  tokio::fs::write(&path, &contents)
    .await
    .map_err(|e| FileError::IoError(e.kind()))?;

  Ok(path)
}

/// Save a graphic (originally produced as SVG) to disk in the format
/// implied by the chosen file extension. Supports SVG, PNG, and PDF.
async fn save_graphic(
  svg_data: String,
  default_dir: Option<PathBuf>,
  fontdb: Arc<resvg::usvg::fontdb::Database>,
) -> Result<PathBuf, FileError> {
  let mut dialog = rfd::AsyncFileDialog::new()
    .set_title("Save Graphic As")
    .set_file_name("graphic.svg")
    .add_filter("SVG", &["svg"])
    .add_filter("PNG", &["png"])
    .add_filter("PDF", &["pdf"]);
  if let Some(dir) = default_dir {
    dialog = dialog.set_directory(dir);
  }
  let path = dialog
    .save_file()
    .await
    .map(|h| h.path().to_owned())
    .ok_or(FileError::DialogClosed)?;

  let ext = path
    .extension()
    .and_then(|e| e.to_str())
    .map(|s| s.to_ascii_lowercase())
    .unwrap_or_else(|| String::from("svg"));

  match ext.as_str() {
    "png" => {
      let png_bytes = encode_svg_as_png(&svg_data, &fontdb)
        .ok_or(FileError::IoError(std::io::ErrorKind::InvalidData))?;
      tokio::fs::write(&path, &png_bytes)
        .await
        .map_err(|e| FileError::IoError(e.kind()))?;
    }
    "pdf" => {
      let pdf_bytes = encode_svg_as_pdf(&svg_data)
        .map_err(|_| FileError::IoError(std::io::ErrorKind::InvalidData))?;
      tokio::fs::write(&path, &pdf_bytes)
        .await
        .map_err(|e| FileError::IoError(e.kind()))?;
    }
    _ => {
      tokio::fs::write(&path, svg_data.as_bytes())
        .await
        .map_err(|e| FileError::IoError(e.kind()))?;
    }
  }

  Ok(path)
}

/// Rasterize an SVG string to a PNG byte buffer at 2× scale.
fn encode_svg_as_png(
  svg_str: &str,
  fontdb: &Arc<resvg::usvg::fontdb::Database>,
) -> Option<Vec<u8>> {
  let opts = resvg::usvg::Options {
    fontdb: fontdb.clone(),
    ..Default::default()
  };
  let tree = resvg::usvg::Tree::from_str(svg_str, &opts).ok()?;
  let size = tree.size();
  let scale: f32 = 2.0;
  let w = (size.width() * scale).ceil() as u32;
  let h = (size.height() * scale).ceil() as u32;
  if w == 0 || h == 0 {
    return None;
  }
  let mut pixmap = tiny_skia::Pixmap::new(w, h)?;
  resvg::render(
    &tree,
    tiny_skia::Transform::from_scale(scale, scale),
    &mut pixmap.as_mut(),
  );
  pixmap.encode_png().ok()
}

/// Convert an SVG string to a PDF byte buffer via svg2pdf.
fn encode_svg_as_pdf(svg_str: &str) -> Result<Vec<u8>, ()> {
  let mut fontdb = svg2pdf::usvg::fontdb::Database::new();
  fontdb.load_font_data(
    include_bytes!(
      "../../resources/AtkinsonHyperlegibleMono-VariableFont_wght.ttf"
    )
    .to_vec(),
  );
  fontdb.load_font_data(
    include_bytes!(
      "../../resources/AtkinsonHyperlegibleNext-VariableFont_wght.ttf"
    )
    .to_vec(),
  );
  fontdb.set_monospace_family("Atkinson Hyperlegible Mono");
  fontdb.set_sans_serif_family("Atkinson Hyperlegible Next");
  fontdb.set_serif_family("Atkinson Hyperlegible Next");
  fontdb.load_system_fonts();

  let opt = svg2pdf::usvg::Options {
    fontdb: std::sync::Arc::new(fontdb),
    ..Default::default()
  };

  let tree = svg2pdf::usvg::Tree::from_str(svg_str, &opt).map_err(|_| ())?;
  svg2pdf::to_pdf(
    &tree,
    svg2pdf::ConversionOptions::default(),
    svg2pdf::PageOptions::default(),
  )
  .map_err(|_| ())
}

/// Data extracted from cell editors for PDF export.
struct PdfCell {
  style: CellStyle,
  text: String,
  output: Option<String>,
  stdout: Option<String>,
  graphics_svg: Option<String>,
}

async fn export_pdf(
  default_path: Option<PathBuf>,
  cells: Vec<PdfCell>,
) -> Result<PathBuf, FileError> {
  use std::fmt::Write;
  use std::sync::Arc as StdArc;

  let mut dialog = rfd::AsyncFileDialog::new()
    .set_title("Export as PDF")
    .add_filter("PDF", &["pdf"]);
  if let Some(ref p) = default_path {
    if let Some(dir) = p.parent() {
      dialog = dialog.set_directory(dir);
    }
    if let Some(name) = p.file_name() {
      dialog = dialog.set_file_name(name.to_string_lossy().as_ref());
    }
  }
  let path = dialog
    .save_file()
    .await
    .map(|h| h.path().to_owned())
    .ok_or(FileError::DialogClosed)?;

  let page_width: f64 = 595.0;
  let margin: f64 = 40.0;
  let content_width = page_width - 2.0 * margin;

  let mut elements = String::new();
  let mut y: f64 = margin;

  for cell in &cells {
    let trimmed = cell.text.trim();
    if trimmed.is_empty()
      && cell.graphics_svg.is_none()
      && cell.output.is_none()
      && cell.stdout.is_none()
    {
      continue;
    }

    match cell.style {
      CellStyle::Title => {
        y += 8.0;
        write_text_lines(
          &mut elements,
          &mut y,
          trimmed,
          margin,
          (24.0, "bold", "sans-serif"),
          "#000",
          30.0,
        );
        y += 12.0;
      }
      CellStyle::Subtitle => {
        write_text_lines(
          &mut elements,
          &mut y,
          trimmed,
          margin,
          (16.0, "normal", "sans-serif"),
          "#555",
          22.0,
        );
        y += 8.0;
      }
      CellStyle::Chapter => {
        y += 8.0;
        write_text_lines(
          &mut elements,
          &mut y,
          trimmed,
          margin,
          (20.0, "bold", "sans-serif"),
          "#000",
          26.0,
        );
        y += 10.0;
      }
      CellStyle::Subchapter => {
        y += 6.0;
        write_text_lines(
          &mut elements,
          &mut y,
          trimmed,
          margin,
          (17.0, "bold", "sans-serif"),
          "#000",
          22.0,
        );
        y += 8.0;
      }
      CellStyle::Section => {
        y += 6.0;
        write_text_lines(
          &mut elements,
          &mut y,
          trimmed,
          margin,
          (18.0, "bold", "sans-serif"),
          "#000",
          24.0,
        );
        y += 8.0;
      }
      CellStyle::Subsection => {
        y += 4.0;
        write_text_lines(
          &mut elements,
          &mut y,
          trimmed,
          margin,
          (15.0, "bold", "sans-serif"),
          "#000",
          20.0,
        );
        y += 6.0;
      }
      CellStyle::Subsubsection => {
        y += 2.0;
        write_text_lines(
          &mut elements,
          &mut y,
          trimmed,
          margin,
          (13.0, "bold", "sans-serif"),
          "#000",
          18.0,
        );
        y += 4.0;
      }
      CellStyle::Text => {
        let wrapped = word_wrap(trimmed, 80);
        write_text_lines(
          &mut elements,
          &mut y,
          &wrapped,
          margin,
          (12.0, "normal", "serif"),
          "#000",
          16.0,
        );
        y += 8.0;
      }
      CellStyle::Item => {
        let wrapped = word_wrap(trimmed, 78);
        write_text_lines(
          &mut elements,
          &mut y,
          &format!("• {wrapped}"),
          margin + 8.0,
          (12.0, "normal", "serif"),
          "#000",
          16.0,
        );
        y += 4.0;
      }
      CellStyle::Subitem => {
        let wrapped = word_wrap(trimmed, 76);
        write_text_lines(
          &mut elements,
          &mut y,
          &format!("◦ {wrapped}"),
          margin + 20.0,
          (12.0, "normal", "serif"),
          "#000",
          16.0,
        );
        y += 4.0;
      }
      CellStyle::Input | CellStyle::Code => {
        let lines: Vec<&str> = cell.text.lines().collect();
        let block_h = lines.len() as f64 * 14.0 + 12.0;
        let _ = write!(
          elements,
          r##"<rect x="{}" y="{}" width="{}" height="{}" fill="#f5f5f5" rx="3"/>"##,
          margin - 4.0,
          y - 2.0,
          content_width + 8.0,
          block_h,
        );
        y += 10.0;
        let x = margin;
        let (font_size, font_weight, font_family) =
          (11.0, "normal", "Atkinson Hyperlegible Mono, monospace");
        let fill = "#333";
        for line in &lines {
          let _ = write!(
            elements,
            r##"<text x="{x}" y="{y}" font-size="{font_size}" font-weight="{font_weight}" font-family="{font_family}" fill="{fill}">{}</text>"##,
            escape_xml(line),
          );
          y += 14.0;
        }
        y += 6.0;
      }
      CellStyle::Output | CellStyle::Print => {
        let cleaned = trimmed
          .replace("-Graphics-", "")
          .replace("-Graphics3D-", "")
          .replace("-Image-", "");
        let cleaned = cleaned.trim();
        if !cleaned.is_empty() {
          write_text_lines(
            &mut elements,
            &mut y,
            cleaned,
            margin,
            (11.0, "normal", "Atkinson Hyperlegible Mono, monospace"),
            "#666",
            14.0,
          );
          y += 4.0;
        }
      }
    }

    // Render output/graphics after Input/Code cells
    if cell.style == CellStyle::Input || cell.style == CellStyle::Code {
      if let Some(ref stdout) = cell.stdout {
        let s = stdout.trim();
        if !s.is_empty() {
          write_text_lines(
            &mut elements,
            &mut y,
            s,
            margin,
            (11.0, "normal", "Atkinson Hyperlegible Mono, monospace"),
            "#888",
            14.0,
          );
          y += 4.0;
        }
      }

      if let Some(ref svg_data) = cell.graphics_svg
        && let Some((svg_w, svg_h)) = parse_svg_dimensions(svg_data)
      {
        let scale = (content_width / svg_w).min(1.0);
        let rendered_w = svg_w * scale;
        let rendered_h = svg_h * scale;
        let _ = write!(
          elements,
          r#"<svg x="{margin}" y="{y}" width="{rendered_w}" height="{rendered_h}" viewBox="0 0 {svg_w} {svg_h}">"#,
        );
        elements.push_str(strip_svg_wrapper(svg_data));
        elements.push_str("</svg>");
        y += rendered_h + 8.0;
      }

      if let Some(ref output) = cell.output {
        let s = output
          .replace("-Graphics-", "")
          .replace("-Graphics3D-", "")
          .replace("-Image-", "");
        let s = s.trim();
        if !s.is_empty() {
          write_text_lines(
            &mut elements,
            &mut y,
            s,
            margin,
            (11.0, "normal", "Atkinson Hyperlegible Mono, monospace"),
            "#666",
            14.0,
          );
          y += 4.0;
        }
      }
    }
  }

  y += margin;

  let svg_doc = format!(
    r#"<svg xmlns="http://www.w3.org/2000/svg" width="{page_width}" height="{y}" viewBox="0 0 {page_width} {y}">{elements}</svg>"#,
  );

  // Convert SVG to PDF via svg2pdf
  let mut fontdb = svg2pdf::usvg::fontdb::Database::new();
  fontdb.load_font_data(
    include_bytes!(
      "../../resources/AtkinsonHyperlegibleMono-VariableFont_wght.ttf"
    )
    .to_vec(),
  );
  fontdb.load_font_data(
    include_bytes!(
      "../../resources/AtkinsonHyperlegibleNext-VariableFont_wght.ttf"
    )
    .to_vec(),
  );
  fontdb.set_monospace_family("Atkinson Hyperlegible Mono");
  fontdb.set_sans_serif_family("Atkinson Hyperlegible Next");
  fontdb.set_serif_family("Atkinson Hyperlegible Next");
  fontdb.set_cursive_family("Atkinson Hyperlegible Next");
  fontdb.set_fantasy_family("Atkinson Hyperlegible Next");
  fontdb.load_system_fonts();

  let opt = svg2pdf::usvg::Options {
    fontdb: StdArc::new(fontdb),
    ..Default::default()
  };

  let tree = svg2pdf::usvg::Tree::from_str(&svg_doc, &opt)
    .map_err(|_| FileError::IoError(std::io::ErrorKind::InvalidData))?;

  let pdf_bytes = svg2pdf::to_pdf(
    &tree,
    svg2pdf::ConversionOptions::default(),
    svg2pdf::PageOptions::default(),
  )
  .map_err(|_| FileError::IoError(std::io::ErrorKind::Other))?;

  tokio::fs::write(&path, &pdf_bytes)
    .await
    .map_err(|e| FileError::IoError(e.kind()))?;

  Ok(path)
}

/// Escape XML special characters for SVG text content.
fn escape_xml(s: &str) -> String {
  s.replace('&', "&amp;")
    .replace('<', "&lt;")
    .replace('>', "&gt;")
    .replace('"', "&quot;")
    .replace('\'', "&apos;")
}

/// Write multi-line text as SVG `<text>` elements, one per line.
fn write_text_lines(
  out: &mut String,
  y: &mut f64,
  text: &str,
  x: f64,
  font: (f64, &str, &str),
  fill: &str,
  line_height: f64,
) {
  use std::fmt::Write;
  let (font_size, font_weight, font_family) = font;
  for line in text.lines() {
    let _ = write!(
      out,
      r#"<text x="{x}" y="{y}" font-size="{font_size}" font-weight="{font_weight}" font-family="{font_family}" fill="{fill}">{}</text>"#,
      escape_xml(line),
    );
    *y += line_height;
  }
}

/// Wrap text at word boundaries to approximately `max_chars` per line.
fn word_wrap(text: &str, max_chars: usize) -> String {
  let mut result = String::new();
  for line in text.lines() {
    if line.len() <= max_chars {
      result.push_str(line);
      result.push('\n');
      continue;
    }
    let mut col = 0;
    for word in line.split_whitespace() {
      if col > 0 && col + 1 + word.len() > max_chars {
        result.push('\n');
        col = 0;
      }
      if col > 0 {
        result.push(' ');
        col += 1;
      }
      result.push_str(word);
      col += word.len();
    }
    result.push('\n');
  }
  result
}

/// Extract width and height from an SVG root element.
fn parse_svg_dimensions(svg: &str) -> Option<(f64, f64)> {
  // Try width="..." height="..." attributes first
  let w = parse_svg_attr(svg, "width")?;
  let h = parse_svg_attr(svg, "height")?;
  Some((w, h))
}

fn parse_svg_attr(svg: &str, attr: &str) -> Option<f64> {
  let tag_end = svg.find('>')?;
  let tag = &svg[..tag_end];
  let pattern = format!("{attr}=\"");
  let start = tag.find(&pattern)? + pattern.len();
  let end = start + tag[start..].find('"')?;
  tag[start..end].trim_end_matches("px").parse().ok()
}

/// Strip the outer `<svg ...>` and `</svg>` wrapper, returning inner content.
fn strip_svg_wrapper(svg: &str) -> &str {
  let inner_start = svg.find('>').map(|i| i + 1).unwrap_or(0);
  let inner_end = svg.rfind("</svg>").unwrap_or(svg.len());
  &svg[inner_start..inner_end]
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn collapsed_chapter_hides_following_until_next_chapter() {
    let states = &[
      (CellStyle::Chapter, true), // collapsed
      (CellStyle::Text, false),
      (CellStyle::Section, false),
      (CellStyle::Input, false),
      (CellStyle::Chapter, false), // new chapter: stops the collapse
      (CellStyle::Text, false),
    ];
    let hidden = compute_hidden_cells_from_states(states);
    assert_eq!(hidden, vec![false, true, true, true, false, false]);
  }

  #[test]
  fn collapsed_chapter_hides_until_title_or_subtitle() {
    // A Subtitle has a *higher* level than Chapter (smaller number),
    // so it also breaks the collapse region.
    let states = &[
      (CellStyle::Chapter, true),
      (CellStyle::Text, false),
      (CellStyle::Subtitle, false), // breaks collapse
      (CellStyle::Text, false),
    ];
    let hidden = compute_hidden_cells_from_states(states);
    assert_eq!(hidden, vec![false, true, false, false]);
  }

  #[test]
  fn collapsed_subchapter_only_hides_within_subchapter() {
    let states = &[
      (CellStyle::Chapter, false),
      (CellStyle::Subchapter, true), // collapsed
      (CellStyle::Section, false),
      (CellStyle::Text, false),
      (CellStyle::Subchapter, false), // new subchapter: stops
      (CellStyle::Text, false),
    ];
    let hidden = compute_hidden_cells_from_states(states);
    assert_eq!(hidden, vec![false, false, true, true, false, false]);
  }

  #[test]
  fn nested_collapse_both_collapsed() {
    let states = &[
      (CellStyle::Chapter, true),
      (CellStyle::Subchapter, true),
      (CellStyle::Text, false),
    ];
    // Both are collapsed. The outer Chapter hides every following
    // cell until another Chapter (or higher), so the Subchapter
    // itself is hidden (and therefore its collapse region is moot).
    let hidden = compute_hidden_cells_from_states(states);
    assert_eq!(hidden, vec![false, true, true]);
  }

  #[test]
  fn no_collapse_when_nothing_collapsed() {
    let states = &[
      (CellStyle::Chapter, false),
      (CellStyle::Text, false),
      (CellStyle::Subchapter, false),
      (CellStyle::Item, false),
      (CellStyle::Subitem, false),
    ];
    let hidden = compute_hidden_cells_from_states(states);
    assert_eq!(hidden, vec![false; 5]);
  }

  #[test]
  fn item_cells_are_not_collapsible() {
    // An Item cell is not a heading, so even if marked collapsed
    // (which the UI prevents), it does not hide cells.
    let states = &[(CellStyle::Item, true), (CellStyle::Text, false)];
    let hidden = compute_hidden_cells_from_states(states);
    assert_eq!(hidden, vec![false, false]);
  }

  #[test]
  fn manipulate_reeval_coalesces_burst() {
    // A burst of slider changes must arm exactly one throttle timer and then
    // re-evaluate a single time — this is what stops the per-tick blocking
    // eval that made the graphic flicker while dragging.
    let expr = woxi::interpret_to_expr("Manipulate[x, {x, 0, 10}]").unwrap();
    let mut state = manipulate::ManipulateState::from_expr(&expr).unwrap();

    // First change schedules the timer; the rest only accumulate.
    assert!(state.request_reeval(), "first change should arm the timer");
    assert!(!state.request_reeval(), "second change must not re-arm");
    assert!(!state.request_reeval(), "third change must not re-arm");

    // Timer fires: the pending changes render and the flag clears, so the
    // next change arms a fresh timer.
    state.run_scheduled_reeval();
    assert!(
      state.request_reeval(),
      "a change after the timer fired should arm a new timer"
    );

    // A timer that fires with nothing new pending is a no-op that still
    // clears the flag (so a later change can re-arm).
    state.run_scheduled_reeval();
    state.run_scheduled_reeval();
    assert!(state.request_reeval(), "flag must clear on an empty fire");
  }

  #[test]
  fn animate_widget_starts_playing_and_wraps() {
    // An Animate widget auto-plays from its initial value and its animation
    // tick advances the continuous control by one step, wrapping back to
    // the start once it passes the end.
    let expr = woxi::interpret_to_expr("Animate[x, {x, 0, 1, 0.5}]").unwrap();
    let mut state = manipulate::ManipulateState::from_expr(&expr).unwrap();
    assert!(state.animated, "Animate must mark the widget animated");
    assert!(state.playing, "AnimationRunning -> True is the default");
    let current = |s: &manipulate::ManipulateState| match &s.controls[0] {
      manipulate::ControlState::Continuous { current, .. } => *current,
      _ => panic!("expected continuous control"),
    };
    assert_eq!(current(&state), 0.0);
    state.advance_animation();
    assert_eq!(current(&state), 0.5);
    state.advance_animation();
    assert_eq!(current(&state), 1.0);
    state.advance_animation();
    assert_eq!(
      current(&state),
      0.0,
      "animation must loop back to the start"
    );
  }

  #[test]
  fn kepler_trigger_and_period_bounded_sliders() {
    // The "Kepler's Second Law" Demonstration pattern: time sliders bounded
    // by the orbital-period control P, plus a Trigger animating t. The
    // widget starts paused, the P-referencing ranges follow P as it moves,
    // and the animation tick targets the Trigger's variable.
    let expr = woxi::interpret_to_expr(
      "Manipulate[{t, dt}, \
       {{t, 0, \"time\"}, 0, P, .01}, \
       {{P, 20, \"period\"}, .1, 50, .01}, \
       {{dt, 5, \"span\"}, .1, P, .01}, \
       {{t, 0, \"animate\"}, 0, P, .01, ControlType -> Trigger}]",
    )
    .unwrap();
    let mut state = manipulate::ManipulateState::from_expr(&expr).unwrap();
    assert!(state.animated, "the Trigger makes the widget animatable");
    assert!(!state.playing, "a Trigger widget starts paused");
    assert_eq!(state.controls.len(), 3, "the Trigger adds no second t row");

    let bounds =
      |s: &manipulate::ManipulateState, i: usize| match &s.controls[i] {
        manipulate::ControlState::Continuous { min, max, .. } => (*min, *max),
        other => panic!("expected continuous control, got {other:?}"),
      };
    let current =
      |s: &manipulate::ManipulateState, i: usize| match &s.controls[i] {
        manipulate::ControlState::Continuous { current, .. } => *current,
        other => panic!("expected continuous control, got {other:?}"),
      };
    // Bounds resolved against P's initial value 20.
    assert_eq!(bounds(&state, 0), (0.0, 20.0));
    assert_eq!(bounds(&state, 2), (0.1, 20.0));

    // Dragging P to 40 widens both dependent ranges on the next render.
    match &mut state.controls[1] {
      manipulate::ControlState::Continuous { current, .. } => *current = 40.0,
      other => panic!("expected continuous control, got {other:?}"),
    }
    state.reevaluate();
    assert_eq!(bounds(&state, 0), (0.0, 40.0));
    assert_eq!(bounds(&state, 2), (0.1, 40.0));

    // Shrinking P to 1 clamps dt (currently 5) into the new range.
    match &mut state.controls[1] {
      manipulate::ControlState::Continuous { current, .. } => *current = 1.0,
      other => panic!("expected continuous control, got {other:?}"),
    }
    state.reevaluate();
    assert_eq!(bounds(&state, 2), (0.1, 1.0));
    assert_eq!(current(&state, 2), 1.0, "dt must clamp to the new max");

    // The animation targets the Trigger's variable t.
    state.advance_animation();
    assert!(
      (current(&state, 0) - 0.01).abs() < 1e-12,
      "the tick must advance t by its step"
    );
  }

  #[test]
  fn locator_manipulate_builds_a_draggable_widget() {
    // The "Center of Mass of a Polygon" Demonstration pattern: a Locator
    // bound to a point list drives the polygon, with icon-labelled
    // SetterBar choices. The widget must expose the points as a live
    // Locator control, render the graphic, and re-render after a point
    // moves or is added/removed (the LocatorAutoCreate interactions).
    woxi::interpret(
      "myIcon[n_] := Graphics[Line[{#, -#}] & /@ \
       ({Cos[#], Sin[#]} & /@ (n Range[0, 3] Pi/4)), \
       ImageSize -> {24, 12}];",
    )
    .unwrap();
    let expr = woxi::interpret_to_expr(
      "Manipulate[Graphics[{Polygon[pts]}, PlotRange -> {{0, 10}, {0, 10}}], \
       {{pts, 1.0 {{2, 2}, {8, 2}, {8, 8}}}, {0, 0}, {10, 10}, Locator, \
       LocatorAutoCreate -> True}, \
       {crosshairs, {\"none\", \"+\" -> myIcon[2]}, ControlType -> SetterBar}]",
    )
    .unwrap();
    let mut state = manipulate::ManipulateState::from_expr(&expr).unwrap();
    assert!(
      state.graphics_handle.is_some(),
      "initial frame should render the polygon: {:?}",
      state.error
    );
    match &state.controls[0] {
      manipulate::ControlState::Locator {
        points,
        auto_create,
        ..
      } => {
        assert_eq!(points, &[(2.0, 2.0), (8.0, 2.0), (8.0, 8.0)]);
        assert!(auto_create);
      }
      other => panic!("expected a locator control, got {other:?}"),
    }
    assert_eq!(
      state.controls[0].current_code(),
      "{{2., 2.}, {8., 2.}, {8., 8.}}",
      "locator points bind as machine reals"
    );
    // The icon-labelled SetterBar choice carries its rendered SVG.
    match &state.controls[1] {
      manipulate::ControlState::Discrete {
        value_labels,
        value_label_svgs,
        ..
      } => {
        assert_eq!(value_labels, &["none".to_string(), "+".to_string()]);
        assert!(value_label_svgs[0].is_none());
        assert!(value_label_svgs[1].is_some());
      }
      other => panic!("expected a discrete control, got {other:?}"),
    }
    // Drag a vertex, then add and remove a point — each re-render keeps
    // producing a graphic.
    if let manipulate::ControlState::Locator { points, .. } =
      &mut state.controls[0]
    {
      points[0] = (3.5, 2.5);
      points.push((5.0, 5.0));
    }
    state.reevaluate();
    assert!(state.graphics_handle.is_some(), "re-render after drag/add");
    if let manipulate::ControlState::Locator { points, .. } =
      &mut state.controls[0]
    {
      points.remove(3);
      points.remove(2);
      points.remove(1);
      points.remove(0);
    }
    state.reevaluate();
    // With every point removed the binding is the empty list; the body
    // must still evaluate (the Demonstration shows its "add some points"
    // message) rather than error out.
    assert!(state.error.is_none(), "empty point list must not error");
  }

  #[test]
  fn stale_animation_ticks_are_dropped() {
    // While a blocking animation advance runs, the timer keeps queueing
    // ticks. Ticks generated before the advance finished are backlog and
    // must be dropped — processing them would re-evaluate again and make
    // the backlog grow every cycle until the app freezes (the showcase.nb
    // spirograph regression).
    let t0 = std::time::Instant::now();
    let t1 = t0 + std::time::Duration::from_millis(60);
    let t2 = t0 + std::time::Duration::from_millis(120);

    // No advance yet: any tick is fresh.
    assert!(animation_tick_is_fresh(t0, None));

    // Last advance finished at t1. A tick generated earlier (t0) piled up
    // in the queue while that advance ran — stale, dropped.
    assert!(!animation_tick_is_fresh(t0, Some(t1)));

    // A tick generated at/after the finish instant is fresh.
    assert!(animation_tick_is_fresh(t1, Some(t1)));
    assert!(animation_tick_is_fresh(t2, Some(t1)));
  }

  #[test]
  fn animate_appearance_none_is_captured() {
    // `Appearance -> None` hides the control rows in the widget view.
    let expr =
      woxi::interpret_to_expr("Animate[x, {x, 0, 1}, Appearance -> None]")
        .unwrap();
    let state = manipulate::ManipulateState::from_expr(&expr).unwrap();
    assert!(state.appearance_none);
    // A plain Manipulate keeps its controls visible.
    let expr = woxi::interpret_to_expr("Manipulate[x, {x, 0, 1}]").unwrap();
    let state = manipulate::ManipulateState::from_expr(&expr).unwrap();
    assert!(!state.appearance_none);
    assert!(!state.animated && !state.playing);
  }

  #[test]
  fn animation_running_false_builds_widget_paused() {
    // `AnimationRunning -> False` keeps the play/pause toggle but starts
    // the widget paused until the user presses play.
    let expr = woxi::interpret_to_expr(
      "Animate[x, {x, 0, 1}, AnimationRunning -> False]",
    )
    .unwrap();
    let state = manipulate::ManipulateState::from_expr(&expr).unwrap();
    assert!(state.animated);
    assert!(!state.playing);
  }

  #[test]
  fn dynamic_box_dump_is_recognized() {
    // The saved box form of a live Manipulate (what Mathematica writes
    // into the Output cell) must be recognized so Studio hides the dump
    // and re-instantiates the widget instead.
    assert!(is_dynamic_box_dump(
      "DynamicModuleBox[{$CellContext`x$$ = 1}, …]"
    ));
    assert!(is_dynamic_box_dump("TagBox[DynamicModuleBox[{…}, …], …]"));
    assert!(is_dynamic_box_dump("DynamicBox[…]"));
    // Ordinary outputs are untouched.
    assert!(!is_dynamic_box_dump("42"));
    assert!(!is_dynamic_box_dump("{1, 2, 3}"));
    assert!(!is_dynamic_box_dump("GraphicsBox[…]"));
  }

  #[test]
  fn stored_manipulate_is_instantiated_on_load() {
    let state =
      instantiate_stored_manipulate("Manipulate[x^2, {x, 0, 10}]", "").unwrap();
    assert_eq!(state.controls.len(), 1);
    // A cell with side effects (multiple statements) is never auto-run.
    assert!(
      instantiate_stored_manipulate("y = 1;\nManipulate[x y, {x, 0, 10}]", "")
        .is_none()
    );
    // A non-Manipulate cell yields no widget.
    assert!(instantiate_stored_manipulate("1 + 1", "").is_none());
  }

  #[test]
  fn standalone_output_cells_get_editors() {
    // Output cells that do not follow an Input (e.g. the saved snapshot
    // images of a Demonstration notebook) must become editors; skipping
    // them would silently drop them from the file on the next save.
    let nb = woxi::notebook::parse_notebook(
      r#"Notebook[{
Cell[CellGroupData[{
Cell["Snapshots", "Section"],
Cell[BoxData["snapshot content"], "Output"]
}, Open]],
Cell[BoxData["standalone output"], "Output"]
}]"#,
    )
    .unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);
    assert_eq!(editors.len(), 3);
    assert_eq!(editors[1].style, CellStyle::Output);
    assert_eq!(editors[1].content.text().trim(), "snapshot content");
    assert_eq!(editors[2].style, CellStyle::Output);
    assert_eq!(editors[2].content.text().trim(), "standalone output");
  }

  #[test]
  fn stored_manipulate_runs_saved_initialization() {
    // `SaveDefinitions -> True` embeds the definitions the body needs in
    // the stored output's Initialization. Instantiating on load must run
    // them so the widget works before any other cell is evaluated.
    let dump = "DynamicModuleBox[{$CellContext`x$$ = 0}, \
      DynamicBox[…],\n\
      Deinitialization:>None,\n\
      Initialization:>({$CellContext`savedInitOffset = 41}; \
      Typeset`initDone$$ = True),\n\
      SynchronousInitialization->True]";
    let state = instantiate_stored_manipulate(
      "Manipulate[x + savedInitOffset, {x, 0, 10}]",
      dump,
    )
    .unwrap();
    assert_eq!(state.text_output.as_deref(), Some("41"));
  }

  /// A stored Manipulate whose body calls helpers from earlier Input
  /// cells (the Demonstrations "Initialization Code" section) must open
  /// live: `editors_from_notebook` replays the preceding inputs before
  /// instantiating the widget.
  #[test]
  fn stored_manipulate_replays_initialization_cells_on_load() {
    let nb_src = r#"Notebook[{
Cell[BoxData["initPlot[z_] := Plot[Sin[z x], {x, 0, 5}]"], "Input"],
Cell[CellGroupData[{
Cell[BoxData["Manipulate[initPlot[a], {a, 1, 3}]"], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`a$$ = 1}, \"…\"]"], "Output"]
}, Open]]
}]"#;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);
    let widget = editors
      .iter()
      .find_map(|e| e.manipulate_state.as_ref())
      .expect("the stored Manipulate must instantiate on load");
    assert!(
      widget.error.is_none(),
      "body must evaluate cleanly: {:?}",
      widget.error
    );
    assert!(
      widget.graphics_handle.is_some(),
      "initPlot from the initialization cell must be in scope, \
       so the first render produces the plot"
    );
  }

  #[test]
  fn two_circular_windows_notebook_opens_with_its_widget() {
    // End-to-end regression for the "Two Circular Windows" Demonstration.
    // Its cell stores `c[[1]]` as a bracketed subscript box, and the body
    // builds every polygon vertex from `Re`/`Im` of a complex exponential
    // whose exponent is written `k Pi I/25`.
    let nb_src = r##"Notebook[{
Cell[BoxData[
 RowBox[{"Manipulate", "[",
  RowBox[{
   RowBox[{"Graphics", "[",
    RowBox[{
     RowBox[{"Polygon", "[",
      RowBox[{
       RowBox[{
        RowBox[{"{",
         RowBox[{
          RowBox[{"Re", "[", "#", "]"}], ",",
          RowBox[{"Im", "[", "#", "]"}]}], "}"}], "&"}], "/@",
       RowBox[{"Table", "[",
        RowBox[{
         RowBox[{
          SubscriptBox["c",
           RowBox[{
           "\[LeftDoubleBracket]", "1", "\[RightDoubleBracket]"}]], "+",
          RowBox[{
           SuperscriptBox["\[ExponentialE]",
            RowBox[{"k", " ", "\[Pi]", " ",
             RowBox[{"\[ImaginaryI]", "/", "25"}]}]], "/", "10"}]}], ",",
         RowBox[{"{", RowBox[{"k", ",", "50"}], "}"}]}], "]"}]}], "]"}], ",",
     RowBox[{"PlotRange", "\[Rule]", "1"}]}], "]"}], ",",
   RowBox[{"{",
    RowBox[{
     RowBox[{"{", RowBox[{"c", ",", RowBox[{"{", RowBox[{"0", ",", "0"}], "}"}]}], "}"}], ",",
     RowBox[{"{", RowBox[{RowBox[{"-", "3"}], ",", RowBox[{"-", "2"}]}], "}"}], ",",
     RowBox[{"{", RowBox[{"4", ",", "2"}], "}"}], ",", "Locator"}], "}"}]}], "]"}]], "Input"]
}]"##;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);

    // The bracketed subscript is a `Part` access, so the cell is valid code.
    let code = editors[0].content.text();
    assert!(
      code.contains("c[[1]]"),
      "the bracketed subscript must parse as Part: {code}"
    );

    let widget = instantiate_stored_manipulate(&code, "")
      .expect("the Manipulate must build a widget");
    assert!(
      widget.error.is_none(),
      "body must evaluate cleanly: {:?}",
      widget.error
    );
    assert!(
      widget.graphics_handle.is_some(),
      "the polygon must render, which needs Re/Im of E^(k Pi I/25) to reduce"
    );
    match &widget.controls[..] {
      [manipulate::ControlState::Slider2D { name, x, y, .. }] => {
        assert_eq!(name, "c");
        assert_eq!((*x, *y), (0.0, 0.0));
      }
      other => panic!("expected one locator control, got {other:?}"),
    }
  }

  #[test]
  fn lorentz_oscillator_manipulate_labels_its_frequency_axis() {
    // End-to-end regression for the "Lorentz Oscillator Model for Optical
    // Constants" Demonstration: ten sliders separated by blank annotation
    // rows, plotting a complex refractive index over a frequency axis that
    // runs to 6*10^15. Those tick labels used to be spelled out in full.
    woxi::interpret(
      "n2[x_, n_, c1_, a1_, b1_] := \
       ((1 + n*10^27*(1.60217662*10^-19)^2 / \
        (8.85418782*10^-12*9.10938356*10^-31) * \
        (c1/(a1*a1*10^30 - x*x - I*b1*x*10^15)))^(1/2))",
    )
    .expect("the Lorentz helper must define");
    assert_eq!(
      woxi::interpret("Round[Re[n2[10^15, 1, 0.5, 1, 0.1]], 1/10^9]").unwrap(),
      "1455353331/500000000"
    );

    let code = "Manipulate[\
      Plot[Re[n2[x, n, c1, a1, b1]], {x, 0.01, 6*10^15}, \
        PlotLabel -> \"refractive index vs. frequency\"], \
      {{n, 1, \"number of electrons\"}, 1, 100, 1, \
        ImageSize -> Tiny, Appearance -> \"Labeled\"}, \"\", \
      {{c1, 0.5, \"oscillator strength 1\"}, 0, 1, .001, \
        ImageSize -> Tiny, Appearance -> \"Labeled\"}, \
      {{a1, 1, \"resonant frequency 1\"}, 0, 10, .01, \
        ImageSize -> Tiny, Appearance -> \"Labeled\"}, \
      {{b1, .1, \"damping factor 1\"}, 0, 10, .001, \
        ImageSize -> Tiny, Appearance -> \"Labeled\"}, \
      SaveDefinitions -> True]";
    let state = instantiate_stored_manipulate(code, "")
      .expect("the Lorentz Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(state.graphics_handle.is_some(), "the curve must render");

    // The `""` argument is a blank annotation row between the control
    // groups, not a control.
    let kinds: Vec<&str> = state
      .controls
      .iter()
      .map(|c| match c {
        manipulate::ControlState::Continuous { .. } => "continuous",
        manipulate::ControlState::Heading { .. } => "heading",
        other => panic!("unexpected control {other:?}"),
      })
      .collect();
    assert_eq!(
      kinds,
      vec![
        "continuous",
        "heading",
        "continuous",
        "continuous",
        "continuous"
      ]
    );
  }

  #[test]
  fn orthocenter_manipulate_uses_a_radical_helper() {
    // End-to-end regression for "The Sum of the Squares of the Distances
    // from the Vertices to the Orthocenter". Its initialization writes
    // Heron's formula with the `\[Sqrt]` prefix operator and pipes the
    // result of an applied anonymous function through `/.`; neither parsed,
    // so the whole cell was a syntax error and no widget appeared.
    woxi::interpret(
      "AreaOfTriangle[p_, q_, r_] := \
       (\\[Sqrt](#(# - EuclideanDistance[p, q])(# - EuclideanDistance[q, r])\
       (# - EuclideanDistance[p, r])) &[\
        (1)/(2) (EuclideanDistance[p, q] + EuclideanDistance[q, r] + \
          EuclideanDistance[p, r])] \
        /. Complex[a_, b_] /; Max[Abs@a, Abs@b] < (10)^(-4) -> 0)",
    )
    .expect("the Heron helper must define");
    assert_eq!(
      woxi::interpret(
        "AreaOfTriangle[{-0.745, -0.64}, {-0.083, 1.}, {0.98875, 0.25}]"
      )
      .unwrap(),
      "1.1270849999999994"
    );

    let code = "Manipulate[\
      Graphics[{Line[{pt1, pt2, pt3, pt1}], \
        Text[AreaOfTriangle[pt1, pt2, pt3], {0, 0}]}, \
        PlotRange -> 1.2], \
      {{pt1, {-0.745, -0.64}}, {-1, -1}, {1, 1}, Locator}, \
      {{pt2, {-0.083, 1.}}, {-1, -1}, {1, 1}, Locator}, \
      {{pt3, {0.98875, 0.25}}, {-1, -1}, {1, 1}, Locator}, \
      SaveDefinitions -> True]";
    let state = instantiate_stored_manipulate(code, "")
      .expect("the orthocenter Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(state.graphics_handle.is_some(), "the triangle must render");
    assert_eq!(state.controls.len(), 3);
    assert!(
      state
        .controls
        .iter()
        .all(|c| matches!(c, manipulate::ControlState::Slider2D { .. }))
    );
  }

  #[test]
  fn quicksort_manipulate_builds_all_four_controls() {
    // End-to-end regression for the "Quicksort versus Selection Sort"
    // Demonstration. Its fourth control is a custom one — `{{li, init, ""},
    // Button[…] &}` — which used to be unrecognised, and an unrecognised
    // control makes the whole widget fail to build.
    let code = "Manipulate[\
      Row[{t, v, k, Length[li]}], \
      {{t, \"quicksort\", \"sorting method\"}, \
        {\"quicksort\", \"selection sort\"}}, \
      {{v, \"bars\", \"view\"}, {\"bars\", \"squares\"}}, \
      {{k, 0, \"step\"}, 0, If[t === \"quicksort\", 12, 50], 1, \
        Appearance -> \"Labeled\"}, \
      {{li, {3, 1, 2}, \"\"}, \
        Button[\"Generate new list\", k = 0; li = {5, 4, 6}] &}, \
      SaveDefinitions -> True]";
    let state = instantiate_stored_manipulate(code, "")
      .expect("the quicksort Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );

    let kinds: Vec<&str> = state
      .controls
      .iter()
      .map(|c| match c {
        manipulate::ControlState::Continuous { .. } => "continuous",
        manipulate::ControlState::Discrete { .. } => "discrete",
        manipulate::ControlState::Button { .. } => "button",
        other => panic!("unexpected control {other:?}"),
      })
      .collect();
    assert_eq!(kinds, vec!["discrete", "discrete", "continuous", "button"]);
    match &state.controls[3] {
      manipulate::ControlState::Button { label, action, .. } => {
        assert_eq!(label, "Generate new list");
        assert_eq!(action, "k = 0; li = {5, 4, 6}");
      }
      other => panic!("expected the reset button, got {other:?}"),
    }
  }

  #[test]
  fn sphericon_nets_manipulate_draws_every_face() {
    // End-to-end regression for the "Nets for Polyhedral Approximations of
    // the Sphericon" Demonstration. Its controls sit in a `Row` of
    // `Control@…` entries, one of which is a colour picker, and the net is
    // a single `Polygon` holding a *list* of triangles.
    woxi::interpret(
      "pir2D[n_] := With[{fi = 2 ArcSin[Sin[Pi/(2 n)]/Sqrt[2]]}, \
       {Table[{1, i + 1, i + 2}, {i, 1, n}], \
        Join[{{0, 0}}, Table[Sqrt[2] {Cos[(j - 1) fi + (Pi - n fi)/2], \
          Sin[(j - 1) fi + (Pi - n fi)/2]}, {j, 1, n + 2}]]}]",
    )
    .expect("the net helper must define");

    let code = "Manipulate[\
      Graphics[{EdgeForm[Black], col, \
        Polygon[Map[pir2D[n][[2, #]] &, pir2D[n][[1]]]]}, \
        ImageSize -> 380], \
      {{n, 3, \"division\"}, 2, 20, 1, Appearance -> \"Labeled\"}, \
      Row[{Control@{{nt, 1, \"\"}, {1 -> \"net\", 2 -> \"surface\"}}, \
        Spacer[20], \
        Control@{{col, Red, \"color\"}, Red}}], \
      SaveDefinitions -> True]";
    let state = instantiate_stored_manipulate(code, "")
      .expect("the sphericon Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(state.graphics_handle.is_some(), "the net must render");

    // The division slider and the net/surface setter both survive the
    // colour control they share a `Row` with.
    let names: Vec<&str> = state
      .controls
      .iter()
      .map(|c| match c {
        manipulate::ControlState::Continuous { name, .. }
        | manipulate::ControlState::Discrete { name, .. } => name.as_str(),
        other => panic!("unexpected control {other:?}"),
      })
      .collect();
    assert_eq!(names, vec!["n", "nt"]);
  }

  #[test]
  fn mandelbrot_set_print_manipulate_renders_its_surface() {
    // End-to-end regression for the "Mandelbrot Set Print" Demonstration:
    // the body iterates a `Compile`d function whose step count is declared
    // `_Integer`, and feeds it an exact rational sample grid declared
    // `_Real`. With every argument coerced to a Real the `Nest` count
    // arrived as `6.` and the whole body failed with "ListPlot3D: first
    // argument must be a list of data".
    let code = "Manipulate[\
      ListPlot3D[\
        Partition[Norm /@ Transpose@mandelbrotnest[\
          Transpose@Flatten[\
            Table[{x, y} + xy, {y, -sca, sca, N[sca/aa]}, \
              {x, -sca, sca, N[sca/aa]}], 1], st], 2 aa + 1], \
        Boxed -> False, Axes -> False, Mesh -> None, \
        PlotRange -> {Automatic, Automatic, {0, 2}}], \
      {{sca, 5/4, \"size of view\"}, .01, 3/2}, \
      {{aa, 3, \"resolution\"}, 1, 25, 1}, \
      {{st, 6, \"steps\"}, 0, 15, 1}, \
      {{xy, {-2/3, 0}, \"center\"}, {-3/2, -1}, {1/2, 1}, \
        ControlPlacement -> Left}, \
      SaveDefinitions -> True]";
    // The initialization cell the Demonstration keeps above the Manipulate.
    woxi::interpret(
      "mandelbrotnest = Compile[{{g, _Real, 2}, {s, _Integer, 0}}, \
       Module[{a3, a4}, \
        Nest[(a3 = #[[1]]^2; a4 = #[[2]]^2; \
          Clip[{a3 - a4 + g[[1]], 2. *#[[1]]*#[[2]] + g[[2]]}, {-4, 4}]) &, \
          g, s]]]",
    )
    .expect("the compiled helper must define");

    let mut state = instantiate_stored_manipulate(code, "")
      .expect("the Mandelbrot Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(state.graphics_handle.is_some(), "the surface must render");
    assert_eq!(state.controls.len(), 4);

    // Raising the step count re-renders without error.
    if let manipulate::ControlState::Continuous { current, .. } =
      &mut state.controls[2]
    {
      *current = 10.0;
    }
    state.reevaluate();
    assert!(state.error.is_none(), "re-render failed: {:?}", state.error);
    assert!(state.graphics_handle.is_some());
  }

  #[test]
  fn gravestone_notebook_loads_its_compressed_texture() {
    // End-to-end regression for the "Gravestone from Transformation of
    // Bilinski Dodecahedron 2" Demonstration. Its texture cell is a
    // `GraphicsBox[TagBox[RasterBox[CompressedData[…]]]]`, which the parser
    // turns back into `Image[CompressedData[…]]`; the payload decompresses
    // to a `RawArray`, and without that head the cell died with
    // `Image::imgarray` and the texture was never defined.
    let nb_src = r##"Notebook[{
Cell[BoxData[
 RowBox[{"tex", "=",
  RowBox[{"Image", "[",
   RowBox[{"CompressedData", "[", "\"1:eJxeJzzK81NLcpMdiwqSqyMrq6uNjI11VEw0FEwqNVRqDbQUYDwa0E8EBciAuKBJWCytToKSqF5xZnpeakpnnklqempRRZKsQDh2xrG\"", "]"}], "]"}]}]], "Input"],
Cell[BoxData["ImageDimensions[tex]"], "Input"]
}]"##;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);
    let code = editors[0].content.text();
    assert!(
      code.starts_with("tex=Image[CompressedData["),
      "unexpected cell: {code}"
    );
    // The whole notebook evaluates without the texture cell erroring, and
    // the image is the 2x2 RGB one the payload holds.
    let mut out = String::new();
    for e in &editors {
      let r = woxi::interpret_with_stdout(&e.content.text())
        .expect("cell must evaluate");
      assert!(
        r.warnings.is_empty(),
        "cell emitted messages: {:?}",
        r.warnings
      );
      out = r.result;
    }
    assert_eq!(out, "{2, 2}");
  }

  #[test]
  fn word_problem_about_boats_manipulate_tracks_its_slider() {
    // End-to-end regression for the "A Word Problem about Boats"
    // Demonstration: the label and the point style are `:>` options whose
    // right-hand side is a `Which` over the slider, and the frame is drawn
    // without ticks.
    let code = "Manipulate[\
      With[{boat1 = {0, 1}, boat2 = {1800, 3}}, \
       Show[\
        Graphics[{Dashed, Gray, \
          Line[{{0, 1}, {Min[1800 11/7, Part[boat1 + {7/11 t, 0}, 1]], 1}}], \
          Line[{{1800, 3}, {Max[0, Part[boat2 - {t, 0}, 1]], 3}}]}], \
        ListPlot[{Tooltip[boat1 + If[t <= 1800 11/7, {7/11 t, 0}, \
            {3600 - 7/11 t, .2}], \"Boat A\"], \
          Tooltip[boat2 - If[t <= 1800, {t, 0}, {3600 - t, .2}], \
            \"Boat B\"]}, \
          PlotStyle :> {PointSize[.02], \
            Which[t == 1100, {PointSize[.04], Red}, True, Blue]}], \
        FrameTicks -> False, Axes -> False, Frame -> True, \
        PlotRange -> {{-19, 1820}, {0, 4}}, AspectRatio -> 1/2, \
        PlotLabel :> Which[t == 1100, \"Boats meet\", True, \"\"], \
        ImageSize -> 500]], \
      {{t, 0, \"time\"}, 0, 3300, 25}, AutorunSequencing -> {{1, 30}}]";
    let mut state = instantiate_stored_manipulate(code, "")
      .expect("the boats Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the initial render must produce the graphic"
    );
    match &state.controls[..] {
      [
        manipulate::ControlState::Continuous {
          name,
          label,
          current,
          min,
          max,
          step,
          ..
        },
      ] => {
        assert_eq!((name.as_str(), label.as_str()), ("t", "time"));
        assert_eq!((*current, *min, *max, *step), (0.0, 0.0, 3300.0, 25.0));
      }
      other => panic!("expected one time slider, got {other:?}"),
    }

    // Moving the slider to the meeting time re-renders without error.
    if let manipulate::ControlState::Continuous { current, .. } =
      &mut state.controls[0]
    {
      *current = 1100.0;
    }
    state.reevaluate();
    assert!(state.error.is_none(), "re-render failed: {:?}", state.error);
    assert!(state.graphics_handle.is_some());
  }

  #[test]
  fn power_of_a_test_manipulate_typesets_its_labels() {
    // End-to-end regression for the "Power of a Test about a Binomial
    // Parameter" Demonstration: the labels are strings carrying inline
    // linear-syntax boxes (`\!\(\*SubscriptBox[\(p\), \(0\)]\)`), and the
    // hypothesis setter labels each choice with a two-line `Column`.
    let code = "Manipulate[\
      GraphicsRow[{\
        Plot[Switch[test, 1, \
            Sum[PDF[BinomialDistribution[n, p], k], {k, 0, b}], 2, \
            Sum[PDF[BinomialDistribution[n, p], k], {k, b, n}]], \
          {p, If[test == 1, 0, p0], If[test == 1, p0, 1]}, \
          AxesOrigin -> {0, 0}, ImageSize -> {275, 300}, \
          PlotLabel -> Text[Row[{\"\\[Alpha]\", \" = K(\", \
            Subscript[p, 0], \") = \", \
            Sum[PDF[BinomialDistribution[n, p0], k], {k, 0, b}]}]]], \
        Graphics[{Thick, Line[{{0, 0}, {n, 0}}], \
          {Thick, Red, Line[{{b, -1}, {b, 1}}]}}, \
          AspectRatio -> 1/2, Axes -> {True, False}, \
          PlotRange -> {{0, n}, {-2, 1}}]}], \
      {{test, 1, \"test type\"}, \
       {1 -> Column[{\"\\!\\(\\*SubscriptBox[\\(H\\), \\(a\\)]\\): \
p < \\!\\(\\*SubscriptBox[\\(p\\), \\(0\\)]\\)\", \
          \"\\!\\(\\*SubscriptBox[\\(H\\), \\(0\\)]\\): \
p \\[GreaterEqual] \\!\\(\\*SubscriptBox[\\(p\\), \\(0\\)]\\)\"}], \
        2 -> Column[{\"\\!\\(\\*SubscriptBox[\\(H\\), \\(a\\)]\\): \
p > \\!\\(\\*SubscriptBox[\\(p\\), \\(0\\)]\\)\", \
          \"\\!\\(\\*SubscriptBox[\\(H\\), \\(0\\)]\\): \
p \\[LessEqual] \\!\\(\\*SubscriptBox[\\(p\\), \\(0\\)]\\)\"}]}, \
       ImageSize -> Tiny}, \
      {{n, 10, \"number of trials n\"}, 5, 25, 1, \
       Appearance -> \"Labeled\"}, \
      {{p0, .5, \"value to test against \
\\!\\(\\*SubscriptBox[\\(p\\), \\(0\\)]\\)\"}, .01, .99, \
       Appearance -> \"Labeled\"}, \
      {{b, 4, \"critical region boundary\"}, 0, n, 1, \
       Appearance -> \"Labeled\"}, \
      TrackedSymbols -> Manipulate]";
    let state = instantiate_stored_manipulate(code, "")
      .expect("the binomial-power Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the initial render must produce the two-panel graphic"
    );

    // The `\!\(\*SubscriptBox[…]\)` in the slider label typesets, and the
    // upper bound of `b` follows the initial `n`.
    let labels: Vec<&str> = state
      .controls
      .iter()
      .map(|c| match c {
        manipulate::ControlState::Continuous { label, .. }
        | manipulate::ControlState::Discrete { label, .. } => label.as_str(),
        other => panic!("unexpected control {other:?}"),
      })
      .collect();
    assert_eq!(
      labels,
      vec![
        "test type",
        "number of trials n",
        "value to test against p₀",
        "critical region boundary",
      ]
    );
    match &state.controls[3] {
      manipulate::ControlState::Continuous {
        current, min, max, ..
      } => assert_eq!((*current, *min, *max), (4.0, 0.0, 10.0)),
      other => panic!("expected a continuous slider, got {other:?}"),
    }

    // Each hypothesis choice is a two-line text label — not an icon, and
    // not the InputForm of its own `Column[…]`. `\[GreaterEqual]` inside
    // the string stays `≥`, and `Subscript[H, a]` folds to `Hₐ`.
    match &state.controls[0] {
      manipulate::ControlState::Discrete {
        values,
        value_labels,
        value_label_svgs,
        ..
      } => {
        assert_eq!(values, &vec!["1".to_string(), "2".to_string()]);
        assert_eq!(
          value_labels,
          &vec![
            "Hₐ: p < p₀\nH₀: p ≥ p₀".to_string(),
            "Hₐ: p > p₀\nH₀: p ≤ p₀".to_string(),
          ]
        );
        assert!(
          value_label_svgs.iter().all(Option::is_none),
          "a text column is not a graphical icon"
        );
      }
      other => panic!("expected the hypothesis setter, got {other:?}"),
    }

    // Switching to the upper-tail test re-renders without error.
    let mut state = state;
    if let manipulate::ControlState::Discrete { current_index, .. } =
      &mut state.controls[0]
    {
      *current_index = 1;
    }
    state.reevaluate();
    assert!(state.error.is_none(), "re-render failed: {:?}", state.error);
    assert!(state.graphics_handle.is_some());
  }

  #[test]
  fn sliding_the_roots_of_cubics_manipulate_builds_widget() {
    // End-to-end regression for the "Sliding the Roots of Cubics"
    // Demonstration: three `Appearance -> "Labeled"` sliders drive an
    // `NSolve` over a symbolic cubic, and the roots are plotted in the
    // complex plane with the polynomial itself as the plot label.
    let code = "Manipulate[\
      With[{func = x^3 + nice[a] x^2 + nice[b] x + nice[c]}, \
       Module[{rts}, \
        rts = NSolve[func == 0, x]; \
        ListPlot[{Re[x], Im[x]} /. rts, PlotStyle -> PointSize[0.03], \
         PlotRange -> {{-6, 6}, {-6, 6}}, ImageSize -> {400, 400}, \
         AspectRatio -> 1, PlotLabel -> func]]], \
      {{a, 0, Row[{\"coefficient of \", Superscript[\"x\", 2]}]}, -4, 4, .1, \
       Appearance -> \"Labeled\"}, \
      {{b, 0, \"coefficient of x\"}, -10, 10, .1, \
       Appearance -> \"Labeled\"}, \
      {{c, -1, \"constant term\"}, -15, 15, .1, \
       Appearance -> \"Labeled\"}, \
      Initialization :> (nice[d_] := \
        If[Abs[d - Round[d]] < .001, Round[d], d]), \
      SaveDefinitions -> True]";
    let mut state = instantiate_stored_manipulate(code, "")
      .expect("the cubic-roots Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the initial render must produce the root plot"
    );

    // Three labeled continuous sliders, in notebook order, with the
    // typeset `Row[…]`/`Superscript[…]` label flattened to text.
    let labels: Vec<&str> = state
      .controls
      .iter()
      .map(|c| match c {
        manipulate::ControlState::Continuous { label, .. } => label.as_str(),
        other => panic!("expected a continuous slider, got {other:?}"),
      })
      .collect();
    assert_eq!(
      labels,
      vec!["coefficient of x²", "coefficient of x", "constant term"]
    );
    match &state.controls[2] {
      manipulate::ControlState::Continuous {
        current, min, max, ..
      } => {
        assert_eq!((*current, *min, *max), (-1.0, -15.0, 15.0));
      }
      other => panic!("expected a continuous slider, got {other:?}"),
    }

    // Moving a slider re-solves the cubic without error.
    if let manipulate::ControlState::Continuous { current, .. } =
      &mut state.controls[1]
    {
      *current = 2.0;
    }
    state.reevaluate();
    assert!(state.error.is_none(), "re-render failed: {:?}", state.error);
    assert!(state.graphics_handle.is_some());
  }

  #[test]
  fn oscilloscope_manipulate_builds_full_widget() {
    // End-to-end regression for the "Oscilloscope with Two Signal Inputs"
    // Demonstration: the loaded Input cell must build a live widget with
    // the animation slider, per-signal headings, popup menus, a divider,
    // and a rendered plot.
    let code = "Manipulate[\n\
      Animate[Plot[{Subscript[r, 1] Subscript[signal, 1][Subscript[β, 1] \
      ω+ϕ+Subscript[α, 1]],Subscript[r, 2] Subscript[signal, 2][Subscript[\
      β, 2] ω+ϕ+Subscript[α, 2]]},{ω,0,10},ExclusionsStyle->Automatic,\
      PlotRange->{{0,10},{3,-3}}],{ϕ,0,Infinity},AnimationRunning->False],\n\
      Style[\"signal 1\",Bold,Medium],\n\
      {{Subscript[signal, 1], SquareWave,\"\"},{Sin->\"sine\",SquareWave->\
      \"square wave\",SawtoothWave->\"sawtooth wave\",TriangleWave->\
      \"triangle wave\"},ControlType->PopupMenu},\n\
      {{Subscript[α, 1],0,\"phase lag\"},0,2π, ImageSize-> Tiny},\n\
      {{Subscript[r, 1],1,\"amplitude\"},0,3,ImageSize-> Tiny},\n\
      {{Subscript[β, 1],1,\"frequency\"},0,5,ImageSize-> Tiny},\n\
      Delimiter,\n\
      Style[\"signal 2\",Bold,Medium],\n\
      {{Subscript[signal, 2], Sin,\"\"},{Sin->\"sine\",SquareWave->\
      \"square wave\",SawtoothWave->\"sawtooth wave\",TriangleWave->\
      \"triangle wave\"},ControlType->PopupMenu},\n\
      {{Subscript[α, 2],0,\"phase lag\"},0,2π,ImageSize-> Tiny},\n\
      {{Subscript[r, 2],1,\"amplitude\"},0,3,ImageSize-> Tiny},\n\
      {{Subscript[β, 2],1,\"frequency\"},0,5,ImageSize-> Tiny}\n\
      ]";
    let state = instantiate_stored_manipulate(code, "")
      .expect("oscilloscope Manipulate must build a widget");
    assert!(
      state.animated,
      "nested Animate body must animate the widget"
    );
    assert!(!state.playing, "AnimationRunning -> False starts paused");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "initial rendering must produce the plot"
    );
    // Row layout: animation slider, then heading + popup + 3 sliders per
    // signal, separated by a divider.
    let kinds: Vec<&str> = state
      .controls
      .iter()
      .map(|c| match c {
        manipulate::ControlState::Continuous { .. } => "continuous",
        manipulate::ControlState::Discrete { .. } => "discrete",
        manipulate::ControlState::Heading { .. } => "heading",
        manipulate::ControlState::Divider => "divider",
        _ => "other",
      })
      .collect();
    assert_eq!(
      kinds,
      vec![
        "continuous",
        "heading",
        "discrete",
        "continuous",
        "continuous",
        "continuous",
        "divider",
        "heading",
        "discrete",
        "continuous",
        "continuous",
        "continuous",
      ]
    );
    // The popup menus keep their dropdown form and rule-form labels.
    match &state.controls[2] {
      manipulate::ControlState::Discrete {
        value_labels,
        current_index,
        popup,
        ..
      } => {
        assert!(popup);
        assert_eq!(value_labels[*current_index], "square wave");
      }
      other => panic!("expected popup menu, got {other:?}"),
    }
    // Changing a signal re-renders without error (regression: symbol-valued
    // popup bindings must substitute as function heads).
    let mut state = state;
    if let manipulate::ControlState::Discrete { current_index, .. } =
      &mut state.controls[2]
    {
      *current_index = 3; // triangle wave
    }
    state.reevaluate();
    assert!(state.error.is_none(), "re-render failed: {:?}", state.error);
    assert!(state.graphics_handle.is_some());
  }

  #[test]
  fn in_body_locator_and_togglerbar_build_interactive_widget() {
    // The "Triangle Calculator" Demonstration pattern: every variable is
    // `ControlType -> None`, the points are driven by `Locator[Dynamic[…]]`
    // markers inside the body's Graphics, and a `TogglerBar[Dynamic[…]]`
    // in the output column switches display layers. The widget must expose
    // the points as controls (with the Dynamic's write-back callback) and
    // the TogglerBar as an interactive display row.
    let code = "Manipulate[\
      Column[{\
        Graphics[{Dynamic[{Line[{ptA, ptB}]}], \
          Locator[Dynamic[ptA, (If[valuesOK[{#, ptB}], ptA = #] &)[\
            Clip[Round[#], {-8, 8}]] &], \
            Graphics[{Disk[{0, 0}, 1]}, ImageSize -> 20]], \
          Locator[Dynamic[ptB, (If[valuesOK[{ptA, #}], ptB = #] &)[\
            Clip[Round[#], {-8, 8}]] &]]}, \
          PlotRange -> {{-9, 9}, {-9, 9}}], \
        TogglerBar[Dynamic[switches], {1 -> \"one\", 2 -> \"two\"}]}], \
      {{ptA, {7, -1}}, {-9, -9}, {9, 9}, ControlType -> None}, \
      {{ptB, {-5, -5}}, {-9, -9}, {9, 9}, ControlType -> None}, \
      {{switches, {1, 2}}, ControlType -> None}, \
      Initialization :> (valuesOK[pts_] := pts[[1]] =!= pts[[2]])]";
    let mut state = instantiate_stored_manipulate(code, "")
      .expect("in-body locator Manipulate must build a widget");
    assert!(state.error.is_none(), "body must render: {:?}", state.error);
    assert!(state.graphics_handle.is_some());

    // The two locator-driven points are visible 2D-slider controls with
    // their write-back callbacks; `switches` stays hidden mutable state.
    let sliders: Vec<&manipulate::ControlState> = state
      .controls
      .iter()
      .filter(|c| matches!(c, manipulate::ControlState::Slider2D { .. }))
      .collect();
    assert_eq!(sliders.len(), 2, "ptA and ptB must be promoted");
    assert_eq!(state.state.len(), 1);
    assert_eq!(state.state[0].0, "switches");

    // The TogglerBar is an interactive display row with both choices
    // selected initially.
    let togglers = collect_togglers(&state.display_trees);
    assert_eq!(togglers.len(), 2);
    assert!(togglers.iter().all(|(_, selected)| *selected));

    // A slider move routes through the callback: the candidate is rounded
    // to the lattice…
    state.slider2d_change(0, 0, 3.4);
    match &state.controls[0] {
      manipulate::ControlState::Slider2D { x, y, .. } => {
        assert_eq!((*x, *y), (3.0, -1.0));
      }
      other => panic!("expected Slider2D, got {other:?}"),
    }
    // …and a move that lands both points on the same spot is rejected by
    // the notebook's valuesOK check (ptB stays put).
    state.slider2d_change(1, 0, 3.0);
    state.slider2d_change(1, 1, -1.0);
    match &state.controls[1] {
      manipulate::ControlState::Slider2D { x, y, .. } => {
        assert_eq!((*x, *y), (3.0, -5.0), "degenerate move must be rejected");
      }
      other => panic!("expected Slider2D, got {other:?}"),
    }
    state.reevaluate();
    assert!(state.error.is_none(), "re-render failed: {:?}", state.error);

    // Clicking a toggler removes its value from the list; the rebuilt
    // display shows it unselected.
    let mutation = collect_togglers(&state.display_trees)[0].0.clone();
    state.apply_display_mutation(&mutation);
    assert_eq!(state.state[0].1, "{2}");
    let togglers = collect_togglers(&state.display_trees);
    assert_eq!(
      togglers.iter().map(|(_, s)| *s).collect::<Vec<_>>(),
      vec![false, true]
    );
  }

  /// Collect `(mutation, selected)` of every Toggler in a display tree.
  fn collect_togglers(
    trees: &[woxi::functions::graphics::DisplayNode],
  ) -> Vec<(String, bool)> {
    use woxi::functions::graphics::DisplayNode;
    fn walk(node: &DisplayNode, out: &mut Vec<(String, bool)>) {
      match node {
        DisplayNode::Toggler {
          mutation, selected, ..
        } => out.push((mutation.clone(), *selected)),
        DisplayNode::Panel(child) => walk(child, out),
        DisplayNode::Grid(rows) => {
          for row in rows {
            for cell in row {
              walk(cell, out);
            }
          }
        }
        DisplayNode::Column(children) | DisplayNode::Row(children) => {
          for c in children {
            walk(c, out);
          }
        }
        DisplayNode::Checkbox { .. } | DisplayNode::Static { .. } => {}
      }
    }
    let mut out = Vec::new();
    for t in trees {
      walk(t, &mut out);
    }
    out
  }

  #[test]
  fn gray_scott_manipulate_steps_and_resets() {
    // End-to-end regression for the "Gray-Scott Reaction-Diffusion"
    // Demonstration: Row-grouped SetterBars, a reset Button, a Trigger
    // sweeping `time`, a Dynamic body that mutates the simulation state,
    // and a run-ONCE Initialization (re-running it per frame would freeze
    // the simulation on its first step).
    let code = "Manipulate[\n\
      (Dynamic[U[[ss]]=U[[ss]]+0.125;\n\
      ArrayPlot[U[[ss]],PlotLabel->Row[{\"time steps \",time}],\
      ColorFunction->ColorData[\"TemperatureMap\"],ImageSize->100],\
      time,TrackedSymbols:>{time,ss}]),\n\
      Row[{Control[{{ss,1,\"field size\"},{1->\"4\",2->\"6\"},SetterBar,\
      Enabled->(time===0)}],Spacer[20]}],\n\
      Row[{Button[Style[\"reset\",11],Refresh[time,None];time=0;U=Uinit;,\
      ImageSize->Medium],Spacer[20],\
      Control[{{time,0,\"run/stop simulation\"},0,Infinity,1,\
      ControlType->Trigger,AnimationRunning->False}]}],\n\
      Initialization:>(\
      Uinit={Partition[Range[16],4]/16.,Partition[Range[36],6]/36.};\
      U=Uinit;)]";
    let mut state = instantiate_stored_manipulate(code, "")
      .expect("Gray-Scott Manipulate must build a widget");
    let kinds: Vec<&str> = state
      .controls
      .iter()
      .map(|c| match c {
        manipulate::ControlState::Discrete { .. } => "discrete",
        manipulate::ControlState::Button { .. } => "button",
        manipulate::ControlState::Trigger { .. } => "trigger",
        _ => "other",
      })
      .collect();
    assert_eq!(kinds, vec!["discrete", "button", "trigger"]);
    assert!(state.has_trigger());
    assert!(state.animated, "a Trigger control animates the widget");
    assert!(!state.playing, "AnimationRunning -> False starts paused");
    assert!(state.error.is_none(), "body error: {:?}", state.error);
    assert!(state.graphics_handle.is_some(), "must render the ArrayPlot");
    // Initialization ran once; the initial render stepped the field once:
    // 1/16 + 0.125 = 0.1875.
    assert_eq!(woxi::interpret("U[[1, 1, 1]]").unwrap(), "0.1875");
    // With time = 0 the SetterBar is enabled.
    assert!(state.control_is_enabled[0]);

    // One animation tick: the trigger advances to 1 and the body steps
    // the simulation again (NOT back to the first step — Initialization
    // must not re-run).
    state.advance_animation();
    match &state.controls[2] {
      manipulate::ControlState::Trigger { current, .. } => {
        assert_eq!(*current, 1.0);
      }
      other => panic!("expected trigger, got {other:?}"),
    }
    assert_eq!(woxi::interpret("U[[1, 1, 1]]").unwrap(), "0.3125");
    // Once running (time != 0), the Enabled condition greys the SetterBar.
    assert!(!state.control_is_enabled[0]);

    // Pressing the reset button rewinds the trigger and restores the
    // fields; the re-render steps once from the restored state.
    let action = match &state.controls[1] {
      manipulate::ControlState::Button { action, .. } => action.clone(),
      other => panic!("expected button, got {other:?}"),
    };
    state.apply_button_action(&action);
    match &state.controls[2] {
      manipulate::ControlState::Trigger { current, .. } => {
        assert_eq!(*current, 0.0, "reset must rewind the trigger");
      }
      other => panic!("expected trigger, got {other:?}"),
    }
    assert_eq!(woxi::interpret("U[[1, 1, 1]]").unwrap(), "0.1875");
    assert!(
      state.control_is_enabled[0],
      "reset re-enables the SetterBar"
    );
    assert!(state.error.is_none());
    assert!(state.graphics_handle.is_some());
  }

  #[test]
  fn toggle_line_comment_wraps_plain_line() {
    let (new_line, shift) = toggle_line_comment("foo");
    assert_eq!(new_line, "(* foo *)");
    assert_eq!(shift, 3);
  }

  #[test]
  fn toggle_line_comment_unwraps_commented_line() {
    let (new_line, shift) = toggle_line_comment("(* foo *)");
    assert_eq!(new_line, "foo");
    // 6 characters removed: "(* " (3) + " *)" (3).
    assert_eq!(shift, -6);
  }

  #[test]
  fn toggle_line_comment_preserves_leading_whitespace() {
    let (new_line, shift) = toggle_line_comment("  foo");
    assert_eq!(new_line, "  (* foo *)");
    assert_eq!(shift, 3);
  }

  #[test]
  fn toggle_line_comment_on_empty_line_does_not_panic() {
    // Regression: toggling a comment on an empty line used to index past
    // the end of `snap.lines()` and crash woxi-studio.
    let (new_line, shift) = toggle_line_comment("");
    assert_eq!(new_line, "(*  *)");
    assert_eq!(shift, 3);
  }

  #[test]
  fn toggle_line_comment_on_whitespace_only_line() {
    let (new_line, shift) = toggle_line_comment("   ");
    assert_eq!(new_line, "   (*  *)");
    assert_eq!(shift, 3);
  }

  // ── Hyperlink extraction ──

  #[test]
  fn extract_hyperlink_two_args() {
    let expr = woxi::syntax::Expr::FunctionCall {
      name: "Hyperlink".to_string(),
      args: vec![
        woxi::syntax::Expr::String("Woxi".to_string()),
        woxi::syntax::Expr::String("https://woxi.ad-si.com".to_string()),
      ]
      .into(),
    };
    assert_eq!(
      extract_hyperlink(&expr),
      Some(("Woxi".to_string(), "https://woxi.ad-si.com".to_string()))
    );
  }

  #[test]
  fn extract_hyperlink_single_arg_uses_uri_as_label() {
    let expr = woxi::syntax::Expr::FunctionCall {
      name: "Hyperlink".to_string(),
      args: vec![woxi::syntax::Expr::String(
        "https://woxi.ad-si.com".to_string(),
      )]
      .into(),
    };
    assert_eq!(
      extract_hyperlink(&expr),
      Some((
        "https://woxi.ad-si.com".to_string(),
        "https://woxi.ad-si.com".to_string()
      ))
    );
  }

  #[test]
  fn extract_hyperlink_non_string_uri_rejected() {
    let expr = woxi::syntax::Expr::FunctionCall {
      name: "Hyperlink".to_string(),
      args: vec![
        woxi::syntax::Expr::String("label".to_string()),
        woxi::syntax::Expr::Identifier("someVar".to_string()),
      ]
      .into(),
    };
    assert_eq!(extract_hyperlink(&expr), None);
  }

  #[test]
  fn extract_hyperlink_other_function_rejected() {
    let expr = woxi::syntax::Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        woxi::syntax::Expr::Integer(1),
        woxi::syntax::Expr::Integer(2),
      ]
      .into(),
    };
    assert_eq!(extract_hyperlink(&expr), None);
  }

  #[test]
  fn extract_hyperlink_zero_args_rejected() {
    let expr = woxi::syntax::Expr::FunctionCall {
      name: "Hyperlink".to_string(),
      args: vec![].into(),
    };
    assert_eq!(extract_hyperlink(&expr), None);
  }

  #[test]
  fn label_char_count_uses_visible_glyphs_and_falls_back_to_name() {
    // A short styled label counts its rendered glyphs (m₁ = 2), while an
    // empty label falls back to the variable name. The widest of these
    // drives the shared label-column width, so a row of single-glyph
    // labels no longer reserves the old fixed 140px gutter.
    let m1 = manipulate::ControlState::Continuous {
      name: "m1".to_string(),
      label: "m\u{2081}".to_string(),
      label_runs: vec![],
      min: 0.0,
      max: 1.0,
      step: 0.1,
      current: 0.0,
    };
    let empty = manipulate::ControlState::Continuous {
      name: "theta".to_string(),
      label: String::new(),
      label_runs: vec![],
      min: 0.0,
      max: 1.0,
      step: 0.1,
      current: 0.0,
    };
    assert_eq!(manipulate_label_char_count(&m1), 2);
    assert_eq!(manipulate_label_char_count(&empty), 5); // "theta"
  }

  // ── Result-output SVG rendering ──

  /// A blank Input-cell editor for exercising `evaluate_cell_statements`.
  fn blank_editor() -> CellEditor {
    CellEditor {
      content: text_editor::Content::new(),
      style: CellStyle::Input,
      output: None,
      stdout: None,
      graphics_svg: None,
      graphics_handle: None,
      graphics_image: None,
      output_svgs: Vec::new(),
      output_images: Vec::new(),
      output_dark: false,
      output_all_svg: false,
      sound: None,
      warnings: Vec::new(),
      undo_stack: Vec::new(),
      redo_stack: Vec::new(),
      output_stale: false,
      is_collapsed: false,
      manipulate_state: None,
      hyperlinks: Vec::new(),
      stored_graphic: false,
      output_content: text_editor::Content::new(),
      stdout_content: text_editor::Content::new(),
    }
  }

  #[test]
  fn scientific_real_result_renders_as_svg_image() {
    // A computed scientific real (`10.^10`) is shown as the typeset SVG image
    // (reusing the Playground rendering) rather than the plain `1.*^10` text.
    let fontdb = Arc::new(resvg::usvg::fontdb::Database::new());
    let mut editor = blank_editor();
    evaluate_cell_statements(&mut editor, "10.^10", false, 1.0, &fontdb);
    assert!(editor.output_all_svg, "result should render via SVG image");
    assert_eq!(editor.output_images.len(), 1);
    // The SVG typesets the exponent as a superscript, not the raw `*^`.
    assert_eq!(editor.output_svgs.len(), 1);
    assert!(!editor.output_svgs[0].contains("*^"));
    assert!(editor.output_svgs[0].contains('\u{00d7}'));
    // The raw text is still kept (for saving to the notebook).
    assert_eq!(editor.output.as_deref(), Some("1.\u{00d7}10^10"));
  }

  #[test]
  fn string_result_stays_plain_text() {
    // A bare string result has no typeset SVG (matching the Playground), so the
    // cell falls back to the selectable plain-text output.
    let fontdb = Arc::new(resvg::usvg::fontdb::Database::new());
    let mut editor = blank_editor();
    evaluate_cell_statements(&mut editor, "\"hello\"", false, 1.0, &fontdb);
    assert!(!editor.output_all_svg, "string should stay text");
    assert!(editor.output_images.is_empty());
    assert_eq!(editor.output.as_deref(), Some("hello"));
  }

  #[test]
  fn list_of_large_literals_groups_digits_as_svg() {
    // A list literal now renders as a typeset SVG so its numbers get digit
    // grouping (`{10000, 20000}` → `{10 000, 20 000}`).
    let fontdb = Arc::new(resvg::usvg::fontdb::Database::new());
    let mut editor = blank_editor();
    evaluate_cell_statements(
      &mut editor,
      "{10000, 20000}",
      false,
      1.0,
      &fontdb,
    );
    assert!(editor.output_all_svg, "list literal should render via SVG");
    assert_eq!(editor.output_images.len(), 1);
    assert!(editor.output_svgs[0].contains(">10<"));
    assert!(!editor.output_svgs[0].contains(">10000<"));
  }

  #[test]
  fn output_dark_flag_tracks_eval_theme() {
    // The dark-mode flag records the theme at evaluation time so the view can
    // fall back to text when the theme later changes.
    let fontdb = Arc::new(resvg::usvg::fontdb::Database::new());
    let mut editor = blank_editor();
    evaluate_cell_statements(&mut editor, "10.^10", true, 1.0, &fontdb);
    assert!(editor.output_dark);
  }

  /// The full widget of the "Trigonometric Sums as Parametric Curves"
  /// Demonstration: a setter bar over the four examples, six labeled
  /// sliders (two of them labeled with a typeset subscript) and a
  /// True/False setter, all drawn from definitions that sum to a
  /// symbolic upper limit.
  #[test]
  fn trigonometric_sums_manipulate_builds_all_controls() {
    for def in [
      "x[t_,m_,1]:=Sum[(Sin[(n)^(2) t])/((n)^(2)), {n, 1, m}]",
      "y[t_,m_,1]:=Sum[(Cos[(n)^(2) t])/((n)^(2)), {n, 1, m}]",
      "x[t_,m_,2]:=Sum[((-1))^(n)(Sin[((-1))^(n) (n)^(2) t])/((n)^(2)), {n, 1, m}]",
      "y[t_,m_,2]:=Sum[((-1))^(n)(Cos[((-1))^(n) (n)^(2) t])/((n)^(2)), {n, 1, m}]",
      "x[t_,m_,3]:=0.4Sum[(Sin[((-1))^(n) n t])/(n), {n, 1, m}]",
      "y[t_,m_,3]:=0.4Sum[(Cos[((-1))^(n) n t])/(n), {n, 1, m}]",
      "x[t_,m_,4]:=0.4Sum[(Sin[ n t])/(n), {n, 1, m}]",
      "y[t_,m_,4]:=0.4Sum[(Cos[ n t])/(n), {n, 1, m}]",
      "p[  m_,example_, k_, optionen___ ] := ParametricPlot[ Evaluate[ {x[t, m,example], y[t, m,example]} ],{t, 0,2 Pi}, PlotPoints -> k , AspectRatio -> Automatic, Axes -> False, ImageSize -> {400, 400}, optionen ]",
    ] {
      woxi::interpret(def).unwrap();
    }
    let code = "Manipulate[p[  m,ex, k, PlotRange -> {{xmin, xmax}, {ymin, ymax}} , Frame -> frame ],\n\
{{ex,1,\"example\"},{1,2,3,4}},\n\
{{m,51,\"sum index\"},1,100,1, Appearance -> \"Labeled\",ImageSize->Tiny},\n\
{{k,100,\"sample points\"},15,150,1, Appearance -> \"Labeled\",ImageSize->Tiny},\n\
{{xmin,-1.3,\"\\!\\(\\*SubscriptBox[\\(x\\), \\(min\\)]\\)\"},-1.3,1,.01,Appearance -> \"Labeled\",ImageSize->Tiny},\n\
{{xmax,1.3,\"\\!\\(\\*SubscriptBox[\\(x\\), \\(max\\)]\\)\"},0,1.3,.01,Appearance -> \"Labeled\",ImageSize->Tiny},\n\
{{ymin,-1.3,\"\\!\\(\\*SubscriptBox[\\(y\\), \\(min\\)]\\)\"},-1.3,1,.01,Appearance -> \"Labeled\",ImageSize->Tiny},\n\
{{ymax,1.7,\"\\!\\(\\*SubscriptBox[\\(y\\), \\(max\\)]\\)\"},-0.6,2.1,.01,Appearance -> \"Labeled\",ImageSize->Tiny},\n\
{{frame, False, \"frame\"}, {True, False}},\n\
TrackedSymbols->Manipulate,\n ControlPlacement -> Left,SaveDefinitions->True,AutorunSequencing->{1,4,5,6,7}]";
    let state = instantiate_stored_manipulate(code, "")
      .expect("the Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the initial render must produce the parametric curve"
    );
    let labels: Vec<&str> = state
      .controls
      .iter()
      .map(|c| match c {
        manipulate::ControlState::Continuous { label, .. } => label.as_str(),
        manipulate::ControlState::Discrete { label, .. } => label.as_str(),
        other => panic!("unexpected control: {other:?}"),
      })
      .collect();
    assert_eq!(
      labels,
      [
        "example",
        "sum index",
        "sample points",
        "xₘᵢₙ",
        "xₘₐₓ",
        "yₘᵢₙ",
        "yₘₐₓ",
        "frame",
      ]
    );
    match (&state.controls[0], &state.controls[1]) {
      (
        manipulate::ControlState::Discrete {
          values,
          current_index,
          ..
        },
        manipulate::ControlState::Continuous {
          min, max, current, ..
        },
      ) => {
        assert_eq!(values, &["1", "2", "3", "4"]);
        assert_eq!(*current_index, 0);
        assert_eq!((*min, *max, *current), (1.0, 100.0, 51.0));
      }
      other => panic!("unexpected leading controls: {other:?}"),
    }
  }

  /// End-to-end regression for the "Trigonometric Sums as Parametric
  /// Curves" Demonstration. Its initialization cells write the partial
  /// sums as typeset `∑` boxes, which have to come back as
  /// `Sum[…, {n, 1, m}]`, and its plotting helper is declared with a
  /// space before the closing bracket (`p[ m_, k_, opts___ ]`).
  #[test]
  fn trigonometric_sums_notebook_opens_with_its_widget() {
    let nb_src = r##"Notebook[{
Cell[BoxData[{
 RowBox[{
  RowBox[{
   RowBox[{"x", "[", RowBox[{"t_", ",", "m_"}], "]"}], ":=",
   RowBox[{
    UnderoverscriptBox["\[Sum]", RowBox[{"n", "=", "1"}], "m"],
    FractionBox[
     RowBox[{"Sin", "[", RowBox[{SuperscriptBox["n", "2"], " ", "t"}], "]"}],
     SuperscriptBox["n", "2"]]}]}], ";"}], "\[IndentingNewLine]",
 RowBox[{
  RowBox[{
   RowBox[{"y", "[", RowBox[{"t_", ",", "m_"}], "]"}], ":=",
   RowBox[{
    UnderoverscriptBox["\[Sum]", RowBox[{"n", "=", "1"}], "m"],
    FractionBox[
     RowBox[{"Cos", "[", RowBox[{SuperscriptBox["n", "2"], " ", "t"}], "]"}],
     SuperscriptBox["n", "2"]]}]}], ";"}]}], "Input"],
Cell[BoxData[
 RowBox[{
  RowBox[{
   RowBox[{"p", "[", "  ",
    RowBox[{"m_", ",", " ", "k_", ",", " ", "optionen___"}], " ", "]"}], " ",
   ":=",
   RowBox[{"ParametricPlot", "[",
    RowBox[{
     RowBox[{"Evaluate", "[",
      RowBox[{"{",
       RowBox[{
        RowBox[{"x", "[", RowBox[{"t", ",", " ", "m"}], "]"}], ",",
        RowBox[{"y", "[", RowBox[{"t", ",", " ", "m"}], "]"}]}], "}"}], "]"}],
     ",", RowBox[{"{", RowBox[{"t", ",", "0", ",", RowBox[{"2", "\[Pi]"}]}], "}"}],
     ",", RowBox[{"PlotPoints", "\[Rule]", "k"}],
     ",", RowBox[{"AspectRatio", "\[Rule]", "Automatic"}],
     ",", RowBox[{"Axes", "\[Rule]", "False"}], ",", "optionen"}], "]"}]}],
  ";"}]], "Input"],
Cell[CellGroupData[{
Cell[BoxData[
 RowBox[{"Manipulate", "[",
  RowBox[{
   RowBox[{"p", "[",
    RowBox[{"m", ",", "k", ",",
     RowBox[{"Frame", "\[Rule]", "frame"}]}], "]"}], ",",
   RowBox[{"{",
    RowBox[{
     RowBox[{"{", RowBox[{"m", ",", "5", ",", "\"\<sum index\>\""}], "}"}],
     ",", "1", ",", "100", ",", "1"}], "}"}], ",",
   RowBox[{"{",
    RowBox[{
     RowBox[{"{", RowBox[{"k", ",", "40", ",", "\"\<sample points\>\""}], "}"}],
     ",", "15", ",", "150", ",", "1"}], "}"}], ",",
   RowBox[{"{",
    RowBox[{
     RowBox[{"{", RowBox[{"frame", ",", "False", ",", "\"\<frame\>\""}], "}"}],
     ",", RowBox[{"{", RowBox[{"True", ",", "False"}], "}"}]}], "}"}], ",",
   RowBox[{"SaveDefinitions", "\[Rule]", "True"}]}], "]"}]], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`m$$ = 5}, \"…\"]"], "Output"]
}, Open]]
}]"##;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);

    // The typeset sums come back as evaluable `Sum[…]` calls.
    let definitions = editors[0].content.text();
    assert!(
      definitions.contains("Sum[(Sin[(n)^(2) t])/((n)^(2)), {n, 1, m}]"),
      "the ∑ box must become a Sum: {definitions}"
    );

    let widget = editors
      .iter()
      .find_map(|e| e.manipulate_state.as_ref())
      .expect("the Manipulate must instantiate on load");
    assert!(
      widget.error.is_none(),
      "body must evaluate cleanly: {:?}",
      widget.error
    );
    assert!(
      widget.graphics_handle.is_some(),
      "the parametric curve must render, which needs both the ∑ definitions \
       and the `p[ m_, … ]` helper (declared with a trailing space) to bind"
    );
    match &widget.controls[..] {
      [
        manipulate::ControlState::Continuous {
          name: m,
          label: m_label,
          current: m_now,
          ..
        },
        manipulate::ControlState::Continuous { name: k, .. },
        manipulate::ControlState::Discrete {
          name: frame,
          values,
          ..
        },
      ] => {
        assert_eq!(
          (m.as_str(), m_label.as_str(), *m_now),
          ("m", "sum index", 5.0)
        );
        assert_eq!(k, "k");
        assert_eq!(frame, "frame");
        assert_eq!(values, &["True".to_string(), "False".to_string()]);
      }
      other => panic!("unexpected controls: {other:?}"),
    }
  }
  /// End-to-end regression for the "Some Irreptiles of Order Greater
  /// than 20" Demonstration. Its data table leaves gaps (`{d1, d2, ,
  /// d4}` — an omitted element is `Null`), the FrontEnd hard-wrapped the
  /// long box expressions mid-bracket, the labels carry `\[Hyphen]`, and
  /// the tiles are drawn with `EdgeForm[If[outline, Thin, None]]`.
  #[test]
  fn irreptiles_notebook_opens_with_its_widget() {
    let nb_src = "Notebook[{\n".to_string()
      + r##"Cell[BoxData[
 RowBox[{RowBox[{"data", "=",
   RowBox[{"{", RowBox[{"7", ",", ",", "9"}], "}"}]}], ";"}]], "Input"],
Cell[CellGroupData[{
Cell[BoxData[
 RowBox[{"Manipulate", "[",
  RowBox[{
   RowBox[{"Graphics", "[",
    RowBox[{
     RowBox[{"{",
      RowBox[{
       RowBox[{"EdgeForm", "[",
        RowBox[{"If", "[",
         RowBox[{"outline", ",", "Thin", ",", "None"}], "]"}], "]"}], ",",
       "Orange", ",",
       RowBox[{"Polygon", "[",
        RowBox[{"{",
         RowBox[{
          RowBox[{"{", RowBox[{"0", ",", "0"}], "}"}], ",",
          RowBox[{"{",
           RowBox[{
            RowBox[{"data", "[",
             RowBox[{"[", "1", "]"}], "]"}], ",", "0"}\
], "}"}], ",",
          RowBox[{"{", RowBox[{"1", ",", "1"}], "}"}]}], "}"}], "]"}], ",",
       RowBox[{"Text", "[",
        RowBox[{"\"\<irrep\[Hyphen]21\>\"", ",",
         RowBox[{"{", RowBox[{"1", ",", "1"}], "}"}]}], "]"}]}], "}"}], ",",
     RowBox[{"ImageSize", "\[Rule]", "200"}]}], "]"}], ",",
   RowBox[{"{",
    RowBox[{
     RowBox[{"{", RowBox[{"outline", ",", "True", ",", "\"\<outline\>\""}],
      "}"}], ",",
     RowBox[{"{", RowBox[{"True", ",", "False"}], "}"}]}], "}"}]}], "]"}]], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`outline$$ = True}, \"…\"]"], "Output"]
}, Open]]
}]"##;
    let nb = woxi::notebook::parse_notebook(&nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);

    // The gap in the data table is `Null`, so the list still has three
    // elements and `data[[1]]` is the 7.
    let data_cell = editors[0].content.text();
    assert_eq!(data_cell.trim(), "data={7,,9};");
    assert_eq!(woxi::interpret("Length[{7,,9}]").unwrap(), "3");
    // The wrapped box rejoined, so the Part access survived the line
    // break, and the label's named character resolved.
    let manipulate_cell = editors[1].content.text();
    assert!(
      manipulate_cell.contains("data[[1]]"),
      "the wrapped Part box must rejoin: {manipulate_cell}"
    );
    assert!(
      manipulate_cell.contains("irrep\u{2010}21"),
      "\\[Hyphen] must resolve to its character: {manipulate_cell}"
    );

    let widget = editors
      .iter()
      .find_map(|e| e.manipulate_state.as_ref())
      .expect("the stored Manipulate must instantiate on load");
    assert!(
      widget.error.is_none(),
      "body must evaluate cleanly: {:?}",
      widget.error
    );
    assert!(
      widget.graphics_handle.is_some(),
      "the tile must render, which needs `data` from the preceding cell"
    );
    match &widget.controls[..] {
      [
        manipulate::ControlState::Discrete {
          name,
          label,
          values,
          ..
        },
      ] => {
        assert_eq!((name.as_str(), label.as_str()), ("outline", "outline"));
        assert_eq!(values, &["True".to_string(), "False".to_string()]);
      }
      other => panic!("unexpected controls: {other:?}"),
    }
  }

  /// End-to-end regression for the "An Expanding Structure Based on the
  /// Diamond Lattice" Demonstration: a `Graphics3D` scene assembled from
  /// `PolyhedronData` face lists, with a `RadioButton` control over a
  /// numeric range.
  #[test]
  fn diamond_lattice_notebook_opens_with_its_widget() {
    let nb_src = r##"Notebook[{
Cell[CellGroupData[{
Cell[BoxData[
 RowBox[{"Manipulate", "[",
  RowBox[{
   RowBox[{"Graphics3D", "[",
    RowBox[{
     RowBox[{"{",
      RowBox[{
       RowBox[{"RGBColor", "[",
        RowBox[{"1", ",", "0.5", ",", "0.5"}], "]"}], ",",
       RowBox[{"Scale", "[",
        RowBox[{
         RowBox[{"GraphicsComplex", "[",
          RowBox[{
           RowBox[{"PolyhedronData", "[",
            RowBox[{"\"\<Octahedron\>\"", ",", "\"\<VertexCoordinates\>\""}],
            "]"}], ",",
           RowBox[{"Polygon", "[",
            RowBox[{"PolyhedronData", "[",
             RowBox[{"\"\<Octahedron\>\"", ",", "\"\<FaceIndices\>\""}],
             "]"}], "]"}]}], "]"}], ",",
         RowBox[{"exp", " ",
          RowBox[{"{", RowBox[{"1", ",", "1", ",", "1"}], "}"}]}], ",",
         RowBox[{"{", RowBox[{"0", ",", "0", ",", "0"}], "}"}]}], "]"}]}],
      "}"}], ",",
     RowBox[{"Boxed", "\[Rule]", "False"}]}], "]"}], ",",
   RowBox[{"{",
    RowBox[{
     RowBox[{"{", RowBox[{"n", ",", "1", ",", "\"\<frequency\>\""}], "}"}],
     ",", "1", ",", "2", ",", "1", ",", "RadioButton"}], "}"}], ",",
   RowBox[{"{",
    RowBox[{
     RowBox[{"{", RowBox[{"exp", ",", "1.8", ",", "\"\<expand\>\""}], "}"}],
     ",", "1", ",", "2.6"}], "}"}]}], "]"}]], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`n$$ = 1}, \"…\"]"], "Output"]
}, Open]]
}]"##;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);
    let widget = editors
      .iter()
      .find_map(|e| e.manipulate_state.as_ref())
      .expect("the stored Manipulate must instantiate on load");
    assert!(
      widget.error.is_none(),
      "body must evaluate cleanly: {:?}",
      widget.error
    );
    assert!(
      widget.graphics_handle.is_some(),
      "the octahedron must render, which needs PolyhedronData's face list"
    );
    match &widget.controls[..] {
      [
        manipulate::ControlState::Discrete {
          name,
          label,
          values,
          ..
        },
        manipulate::ControlState::Continuous {
          name: exp, current, ..
        },
      ] => {
        // `RadioButton` over `1, 2, 1` is a choice between 1 and 2, not a
        // slider over the range.
        assert_eq!((name.as_str(), label.as_str()), ("n", "frequency"));
        assert_eq!(values, &["1".to_string(), "2".to_string()]);
        assert_eq!((exp.as_str(), *current), ("exp", 1.8));
      }
      other => panic!("unexpected controls: {other:?}"),
    }
  }

  /// End-to-end regression for the "Constant Price Elasticity of Demand"
  /// Demonstration: a `Grid` of two `Show[Plot[…], Graphics[…]]` panels,
  /// each drawn with the plot's own `PlotRange`, `PlotLabel` and
  /// `AxesLabel`.
  #[test]
  fn price_elasticity_notebook_opens_with_its_widget() {
    let nb_src = r##"Notebook[{
Cell[CellGroupData[{
Cell[BoxData[
 RowBox[{"Manipulate", "[",
  RowBox[{
   RowBox[{"Grid", "[",
    RowBox[{"{",
     RowBox[{"{",
      RowBox[{"Show", "[",
       RowBox[{
        RowBox[{"Plot", "[",
         RowBox[{
          RowBox[{"A", " ",
           SuperscriptBox["x", RowBox[{"1", "/", "\[Epsilon]"}]]}], ",",
          RowBox[{"{", RowBox[{"x", ",", "0", ",", "10"}], "}"}], ",",
          RowBox[{"PlotRange", "\[Rule]",
           RowBox[{"{",
            RowBox[{
             RowBox[{"{", RowBox[{"0", ",", "10"}], "}"}], ",",
             RowBox[{"{", RowBox[{"0", ",", "10"}], "}"}]}], "}"}]}], ",",
          RowBox[{"AxesLabel", "\[Rule]",
           RowBox[{"{",
            RowBox[{
             RowBox[{"Style", "[",
              RowBox[{"\"\<Q\>\"", ",", "Blue", ",", "Italic"}], "]"}], ",",
             RowBox[{"Style", "[",
              RowBox[{"\"\<P\>\"", ",", "Blue", ",", "Italic"}], "]"}]}],
            "}"}]}], ",",
          RowBox[{"PlotLabel", "\[Rule]", "\"\<Demand\>\""}]}], "]"}], ",",
        RowBox[{"Graphics", "[",
         RowBox[{"{",
          RowBox[{"PointSize", "[", "0.03", "]"}], ",",
          RowBox[{"Point", "[",
           RowBox[{"{", RowBox[{"2.5", ",", "price"}], "}"}], "]"}], "}"}],
         "]"}]}], "]"}], "}"}], "}"}], "]"}], ",",
   RowBox[{"{",
    RowBox[{
     RowBox[{"{", RowBox[{"A", ",", "5", ",", "\"\<A\>\""}], "}"}], ",",
     "0.2", ",", "20"}], "}"}], ",",
   RowBox[{"{",
    RowBox[{
     RowBox[{"{", RowBox[{"price", ",", "2", ",", "\"\<p\>\""}], "}"}], ",",
     "0.5", ",", "10"}], "}"}]}], "]"}]], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`A$$ = 5}, \"…\"]"], "Output"]
}, Open]]
}]"##;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);
    let widget = editors
      .iter()
      .find_map(|e| e.manipulate_state.as_ref())
      .expect("the stored Manipulate must instantiate on load");
    assert!(
      widget.error.is_none(),
      "body must evaluate cleanly: {:?}",
      widget.error
    );
    assert!(
      widget.graphics_handle.is_some(),
      "the Grid of plots must render as a picture, not as `-Graphics-` text"
    );
    match &widget.controls[..] {
      [
        manipulate::ControlState::Continuous {
          name: a,
          current: a_now,
          ..
        },
        manipulate::ControlState::Continuous {
          name: p,
          current: p_now,
          ..
        },
      ] => {
        assert_eq!((a.as_str(), *a_now), ("A", 5.0));
        assert_eq!((p.as_str(), *p_now), ("price", 2.0));
      }
      other => panic!("unexpected controls: {other:?}"),
    }
  }

  /// End-to-end regression for the "A Converging Geometric Series"
  /// Demonstration: a `Grid` with a `NumberForm` caption above a row of
  /// two pictures, the first assembled from rectangles `Sow`n in a loop.
  #[test]
  fn geometric_series_notebook_opens_with_its_widget() {
    let nb_src = r##"Notebook[{
Cell[CellGroupData[{
Cell[BoxData[
 RowBox[{"Manipulate", "[",
  RowBox[{
   RowBox[{"Grid", "[",
    RowBox[{"{",
     RowBox[{
      RowBox[{"{",
       RowBox[{"Text", "@",
        RowBox[{"Row", "[",
         RowBox[{"{",
          RowBox[{"\"\<area = \>\"", ",",
           RowBox[{"NumberForm", "[",
            RowBox[{
             RowBox[{"N", "[",
              RowBox[{"Sum", "[",
               RowBox[{
                SuperscriptBox[
                 RowBox[{"(", RowBox[{"1", "/", "2"}], ")"}], "t"], ",",
                RowBox[{"{", RowBox[{"t", ",", "1", ",", "n"}], "}"}]}],
               "]"}], "]"}], ",",
             RowBox[{"{", RowBox[{"7", ",", "9"}], "}"}]}], "]"}]}], "}"}],
         "]"}]}], "}"}], ",",
      RowBox[{"{",
       RowBox[{"Graphics", "[",
        RowBox[{
         RowBox[{"{",
          RowBox[{
           RowBox[{"EdgeForm", "[", "Black", "]"}], ",",
           RowBox[{"Hue", "[", "0.3", "]"}], ",",
           RowBox[{"Rectangle", "[",
            RowBox[{
             RowBox[{"{", RowBox[{"0", ",", "0"}], "}"}], ",",
             RowBox[{"{", RowBox[{"0.5", ",", "1"}], "}"}]}], "]"}]}], "}"}],
         ",", RowBox[{"Axes", "\[Rule]", "True"}], ",",
         RowBox[{"Ticks", "\[Rule]",
          RowBox[{"{",
           RowBox[{
            RowBox[{"{", RowBox[{"0", ",", "1"}], "}"}], ",",
            RowBox[{"{", "1", "}"}]}], "}"}]}], ",",
         RowBox[{"ImageSize", "\[Rule]", "200"}]}], "]"}], "}"}]}], "}"}],
    "]"}], ",",
   RowBox[{"{",
    RowBox[{
     RowBox[{"{", RowBox[{"n", ",", "3", ",", "\"\<n\>\""}], "}"}], ",", "1",
     ",", "25", ",", "1"}], "}"}]}], "]"}]], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`n$$ = 3}, \"…\"]"], "Output"]
}, Open]]
}]"##;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);
    let widget = editors
      .iter()
      .find_map(|e| e.manipulate_state.as_ref())
      .expect("the stored Manipulate must instantiate on load");
    assert!(
      widget.error.is_none(),
      "body must evaluate cleanly: {:?}",
      widget.error
    );
    assert!(
      widget.graphics_handle.is_some(),
      "the Grid must render as a picture: its caption and its rectangle"
    );
    match &widget.controls[..] {
      [
        manipulate::ControlState::Continuous {
          name,
          current,
          min,
          max,
          ..
        },
      ] => {
        assert_eq!((name.as_str(), *current), ("n", 3.0));
        assert_eq!((*min, *max), (1.0, 25.0));
      }
      other => panic!("unexpected controls: {other:?}"),
    }
  }

  /// End-to-end regression for "The Mayan Calendar" Demonstration: a
  /// wheel of `Disk` sectors whose teeth carry `Inset` pictures, with the
  /// day name written as a `Row` separated by a `Spacer`.
  #[test]
  fn mayan_calendar_notebook_opens_with_its_widget() {
    let nb_src = r##"Notebook[{
Cell[CellGroupData[{
Cell[BoxData[
 RowBox[{"Manipulate", "[",
  RowBox[{
   RowBox[{"Graphics", "[",
    RowBox[{
     RowBox[{"{",
      RowBox[{
       RowBox[{"Disk", "[",
        RowBox[{
         RowBox[{"{", RowBox[{"0", ",", "0"}], "}"}], ",", "1", ",",
         RowBox[{"{", RowBox[{"0", ",", "\[Pi]"}], "}"}]}], "]"}], ",",
       RowBox[{"Inset", "[",
        RowBox[{
         RowBox[{"Graphics", "[",
          RowBox[{"{",
           RowBox[{"Black", ",", RowBox[{"Disk", "[", "]"}]}], "}"}], "]"}],
         ",", RowBox[{"{", RowBox[{"0.5", ",", "0"}], "}"}], ",", "Automatic",
         ",", RowBox[{"{", RowBox[{"0.3", ",", "0.15"}], "}"}], ",",
         RowBox[{"{",
          RowBox[{"Automatic", ",", RowBox[{"{", RowBox[{"1", ",", "0"}], "}"}]}],
          "}"}]}], "]"}], ",",
       RowBox[{"Text", "[",
        RowBox[{
         RowBox[{"Style", "[",
          RowBox[{
           RowBox[{"Row", "[",
            RowBox[{
             RowBox[{"{",
              RowBox[{RowBox[{"Mod", "[", RowBox[{"day", ",", "13", ",", "1"}], "]"}],
               ",", "\"\<Imix\>\""}], "}"}], ",",
             RowBox[{"Spacer", "[", "1", "]"}]}], "]"}], ",", "16", ",", "Red"}],
          "]"}], ",", RowBox[{"{", RowBox[{"0", ",", "0"}], "}"}]}], "]"}]}],
      "}"}], ",",
     RowBox[{"ImageSize", "\[Rule]", RowBox[{"{", RowBox[{"300", ",", "200"}], "}"}]}],
     ",", RowBox[{"Background", "\[Rule]", "LightBlue"}]}], "]"}], ",",
   RowBox[{"{",
    RowBox[{
     RowBox[{"{", RowBox[{"day", ",", "1", ",", "\"\<day\>\""}], "}"}], ",",
     "1", ",", "260", ",", "1"}], "}"}]}], "]"}]], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`day$$ = 1}, \"…\"]"], "Output"]
}, Open]]
}]"##;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);
    // The cell reads back as evaluable code, spacer and all.
    let code = editors[0].content.text();
    assert!(code.contains("Spacer[1]"), "{code}");
    let widget = editors
      .iter()
      .find_map(|e| e.manipulate_state.as_ref())
      .expect("the stored Manipulate must instantiate on load");
    assert!(
      widget.error.is_none(),
      "body must evaluate cleanly: {:?}",
      widget.error
    );
    assert!(
      widget.graphics_handle.is_some(),
      "the wheel must render, with the inset picture on its tooth"
    );
    match &widget.controls[..] {
      [
        manipulate::ControlState::Continuous {
          name, label, max, ..
        },
      ] => {
        assert_eq!((name.as_str(), label.as_str()), ("day", "day"));
        assert_eq!(*max, 260.0);
      }
      other => panic!("unexpected controls: {other:?}"),
    }
  }

  /// End-to-end regression for "Inscribed Angles That Intercept the Same
  /// Arc": a `DynamicModule` whose body computes into locals and ends in a
  /// `Grid` of a readout layout above the drawing, driven by `Locator`
  /// controls.
  #[test]
  fn inscribed_angles_notebook_opens_with_its_widget() {
    let nb_src = r##"Notebook[{
Cell[CellGroupData[{
Cell[BoxData[
 RowBox[{"Manipulate", "[",
  RowBox[{
   RowBox[{"DynamicModule", "[",
    RowBox[{
     RowBox[{"{", "an", "}"}], ",",
     RowBox[{
      RowBox[{"an", "=",
       RowBox[{"Norm", "[", RowBox[{"pa", "-", "ac"}], "]"}]}], ";",
      RowBox[{"Grid", "[",
       RowBox[{"{",
        RowBox[{
         RowBox[{"{",
          RowBox[{"Column", "[",
           RowBox[{"{",
            RowBox[{
             RowBox[{"Grid", "[",
              RowBox[{"{",
               RowBox[{"{", RowBox[{"\"\<d\>\"", ",", "an"}], "}"}], "}"}],
              "]"}]}], "}"}], "]"}], "}"}], ",",
         RowBox[{"{",
          RowBox[{"Graphics", "[",
           RowBox[{
            RowBox[{"{",
             RowBox[{
              RowBox[{"Circle", "[", "]"}], ",",
              RowBox[{"Line", "[",
               RowBox[{"{", RowBox[{"pa", ",", "ac"}], "}"}], "]"}]}], "}"}],
            ",",
            RowBox[{"ImageSize", "\[Rule]", "200"}]}], "]"}], "}"}]}], "}"}],
       "]"}]}]}], "]"}], ",",
   RowBox[{"{",
    RowBox[{
     RowBox[{"{",
      RowBox[{"pa", ",",
       RowBox[{"{", RowBox[{RowBox[{"-", "0.6"}], ",", "0.8"}], "}"}]}], "}"}],
     ",", RowBox[{"{", RowBox[{RowBox[{"-", "1"}], ",", RowBox[{"-", "1"}]}], "}"}],
     ",", RowBox[{"{", RowBox[{"1", ",", "1"}], "}"}], ",", "Locator", ",",
     RowBox[{"Appearance", "\[Rule]", "None"}]}], "}"}], ",",
   RowBox[{"{",
    RowBox[{
     RowBox[{"{",
      RowBox[{"ac", ",",
       RowBox[{"{", RowBox[{"0.82", ",", RowBox[{"-", "0.57"}]}], "}"}]}], "}"}],
     ",", RowBox[{"{", RowBox[{RowBox[{"-", "1"}], ",", RowBox[{"-", "1"}]}], "}"}],
     ",", RowBox[{"{", RowBox[{"1", ",", "1"}], "}"}], ",", "Locator", ",",
     RowBox[{"Appearance", "\[Rule]", "None"}]}], "}"}]}], "]"}]], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`pa$$ = {-0.6, 0.8}}, \"…\"]"], "Output"]
}, Open]]
}]"##;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);
    let widget = editors
      .iter()
      .find_map(|e| e.manipulate_state.as_ref())
      .expect("the stored Manipulate must instantiate on load");
    assert!(
      widget.error.is_none(),
      "body must evaluate cleanly: {:?}",
      widget.error
    );
    assert!(
      widget.graphics_handle.is_some(),
      "the DynamicModule must display its Grid, not its own expression text"
    );
    // Both locators become draggable 2-D controls at their initial points.
    match &widget.controls[..] {
      [
        manipulate::ControlState::Slider2D {
          name: pa,
          x: pax,
          y: pay,
          ..
        },
        manipulate::ControlState::Slider2D {
          name: ac, x: acx, ..
        },
      ] => {
        assert_eq!((pa.as_str(), *pax, *pay), ("pa", -0.6, 0.8));
        assert_eq!((ac.as_str(), *acx), ("ac", 0.82));
      }
      other => panic!("unexpected controls: {other:?}"),
    }
  }

  /// End-to-end regression for "A Procedure to Compute the Digit Sequence
  /// of a Square Root": a `Grid` whose caption typesets a radical and a
  /// binary `BaseForm`, over a pair of `ArrayPlot`s.
  #[test]
  fn digit_sequence_notebook_opens_with_its_widget() {
    let nb_src = r##"Notebook[{
Cell[BoxData[
 RowBox[{
  RowBox[{"digitPlot", "[", RowBox[{"num_", ",", "steps_"}], "]"}], ":=",
  RowBox[{"Grid", "[",
   RowBox[{"{",
    RowBox[{
     RowBox[{"{",
      RowBox[{"Text", "@",
       RowBox[{"Style", "[",
        RowBox[{RowBox[{"Sqrt", "[", "num", "]"}], ",", "18", ",", "Bold"}],
        "]"}]}], "}"}], ",",
     RowBox[{"{",
      RowBox[{"Text", "@",
       RowBox[{"BaseForm", "[",
        RowBox[{
         RowBox[{"N", "[",
          RowBox[{RowBox[{"Sqrt", "[", "num", "]"}], ",", "20"}], "]"}], ",",
         "2"}], "]"}]}], "}"}], ",",
     RowBox[{"{",
      RowBox[{"ArrayPlot", "[",
       RowBox[{
        RowBox[{"Table", "[",
         RowBox[{
          RowBox[{"Mod", "[", RowBox[{RowBox[{"i", " ", "j"}], ",", "2"}], "]"}],
          ",", RowBox[{"{", RowBox[{"i", ",", "steps"}], "}"}], ",",
          RowBox[{"{", RowBox[{"j", ",", "steps"}], "}"}]}], "]"}], ",",
        RowBox[{"ImageSize", "\[Rule]",
         RowBox[{"{", RowBox[{"120", ",", "120"}], "}"}]}]}], "]"}], "}"}]}],
    "}"}], "]"}]}]], "Input"],
Cell[CellGroupData[{
Cell[BoxData[
 RowBox[{"Manipulate", "[",
  RowBox[{
   RowBox[{"digitPlot", "[", RowBox[{"num", ",", "steps"}], "]"}], ",",
   RowBox[{"{",
    RowBox[{
     RowBox[{"{",
      RowBox[{"num", ",", "2", ",", "\"\<square root of:\>\""}], "}"}], ",",
     RowBox[{"{", RowBox[{"2", ",", "3", ",", "5"}], "}"}]}], "}"}], ",",
   RowBox[{"{",
    RowBox[{RowBox[{"{", RowBox[{"steps", ",", "20"}], "}"}], ",", "10", ",",
     "40", ",", "1"}], "}"}]}], "]"}]], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`num$$ = 2}, \"…\"]"], "Output"]
}, Open]]
}]"##;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);
    let widget = editors
      .iter()
      .find_map(|e| e.manipulate_state.as_ref())
      .expect("the stored Manipulate must instantiate on load");
    assert!(
      widget.error.is_none(),
      "body must evaluate cleanly: {:?}",
      widget.error
    );
    assert!(
      widget.graphics_handle.is_some(),
      "the caption and both plots must draw"
    );
    match &widget.controls[..] {
      [
        manipulate::ControlState::Discrete {
          name,
          label,
          values,
          ..
        },
        manipulate::ControlState::Continuous {
          name: steps,
          current,
          ..
        },
      ] => {
        assert_eq!((name.as_str(), label.as_str()), ("num", "square root of:"));
        assert_eq!(
          values,
          &["2".to_string(), "3".to_string(), "5".to_string()]
        );
        assert_eq!((steps.as_str(), *current), ("steps", 20.0));
      }
      other => panic!("unexpected controls: {other:?}"),
    }
  }

  /// End-to-end regression for "A Solution of Euler's Type for an Exact
  /// Differential Equation": a `Show` of a meshed `ContourPlot` under the
  /// gradient arrows, driven by a locator the reader may add points to.
  #[test]
  fn exact_differential_notebook_opens_with_its_widget() {
    let nb_src = r##"Notebook[{
Cell[CellGroupData[{
Cell[BoxData[
 RowBox[{"Manipulate", "[",
  RowBox[{
   RowBox[{"Show", "[",
    RowBox[{
     RowBox[{"ContourPlot", "[",
      RowBox[{
       RowBox[{
        SuperscriptBox["x", "2"], "+",
        RowBox[{"0.9", " ", SuperscriptBox["y", "2"]}]}], ",",
       RowBox[{"{", RowBox[{"x", ",", RowBox[{"-", "1"}], ",", "1"}], "}"}],
       ",", RowBox[{"{", RowBox[{"y", ",", RowBox[{"-", "1"}], ",", "1"}], "}"}],
       ",", RowBox[{"Axes", "\[Rule]", "True"}], ",",
       RowBox[{"ContourShading", "\[Rule]", "None"}], ",",
       RowBox[{"Contours", "\[Rule]", "con"}], ",",
       RowBox[{"Mesh", "\[Rule]", "step"}], ",",
       RowBox[{"MeshFunctions", "\[Rule]",
        RowBox[{"{",
         RowBox[{
          RowBox[{RowBox[{"10", " ", "#1"}], "&"}], ",",
          RowBox[{RowBox[{"10", " ", "#2"}], "&"}]}], "}"}]}]}], "]"}], ",",
     RowBox[{"Graphics", "[",
      RowBox[{"{",
       RowBox[{"Arrow", "[",
        RowBox[{"{",
         RowBox[{
          RowBox[{"Last", "[", "pts", "]"}], ",",
          RowBox[{
           RowBox[{"Last", "[", "pts", "]"}], "+",
           RowBox[{"{", RowBox[{"0.3", ",", "0.2"}], "}"}]}]}], "}"}], "]"}],
       "}"}], "]"}], ",",
     RowBox[{"PlotRange", "\[Rule]", "1"}]}], "]"}], ",",
   RowBox[{"{",
    RowBox[{
     RowBox[{"{",
      RowBox[{"pts", ",",
       RowBox[{"{", RowBox[{"{", RowBox[{"0", ",", "0"}], "}"}], "}"}]}], "}"}],
     ",", RowBox[{"{", RowBox[{RowBox[{"-", "1"}], ",", RowBox[{"-", "1"}]}], "}"}],
     ",", RowBox[{"{", RowBox[{"1", ",", "1"}], "}"}], ",", "Locator", ",",
     RowBox[{"LocatorAutoCreate", "\[Rule]",
      RowBox[{"{", RowBox[{"1", ",", "\[Infinity]"}], "}"}]}]}], "}"}], ",",
   RowBox[{"{",
    RowBox[{RowBox[{"{", RowBox[{"con", ",", "1", ",", "\"\<contours\>\""}], "}"}],
     ",", "0", ",", "30", ",", "1"}], "}"}], ",",
   RowBox[{"{",
    RowBox[{RowBox[{"{", RowBox[{"step", ",", "11", ",", "\"\<step\>\""}], "}"}],
     ",", "1", ",", "11", ",", "2"}], "}"}]}], "]"}]], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`con$$ = 1}, \"…\"]"], "Output"]
}, Open]]
}]"##;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);
    let widget = editors
      .iter()
      .find_map(|e| e.manipulate_state.as_ref())
      .expect("the stored Manipulate must instantiate on load");
    assert!(
      widget.error.is_none(),
      "body must evaluate cleanly: {:?}",
      widget.error
    );
    assert!(
      widget.graphics_handle.is_some(),
      "the contours, the mesh and the arrow must all draw"
    );
    match &widget.controls[..] {
      [
        manipulate::ControlState::Locator {
          name,
          points,
          auto_create,
          ..
        },
        manipulate::ControlState::Continuous {
          name: con,
          current: con_now,
          ..
        },
        manipulate::ControlState::Continuous {
          name: step,
          current: step_now,
          ..
        },
      ] => {
        assert_eq!(
          (name.as_str(), points.as_slice()),
          ("pts", &[(0.0, 0.0)][..])
        );
        assert!(auto_create, "LocatorAutoCreate -> {{1, ∞}} allows adding");
        assert_eq!((con.as_str(), *con_now), ("con", 1.0));
        assert_eq!((step.as_str(), *step_now), ("step", 11.0));
      }
      other => panic!("unexpected controls: {other:?}"),
    }
  }

  /// End-to-end regression for the "Balanced Ternary Notation"
  /// Demonstration: a balance beam whose label is a `Section`-styled Row
  /// of balanced-ternary digits, the negative ones written as an
  /// underscored 1.
  #[test]
  fn balanced_ternary_notebook_opens_with_its_widget() {
    let nb_src = r##"Notebook[{
Cell[BoxData[
 RowBox[{
  RowBox[{"btDigits", "[", "n_", "]"}], ":=",
  RowBox[{"Row", "[",
   RowBox[{
    RowBox[{"IntegerDigits", "[",
     RowBox[{"n", ",", "3"}], "]"}], "/.",
    RowBox[{"{",
     RowBox[{"2", "\[Rule]",
      RowBox[{"UnderscriptBox", "[", RowBox[{"\"\<1\>\"", ",", "\"\<_\>\""}],
       "]"}]}], "}"}]}], "]"}]}]], "Input"],
Cell[CellGroupData[{
Cell[BoxData[
 RowBox[{"Manipulate", "[",
  RowBox[{
   RowBox[{"Graphics", "[",
    RowBox[{
     RowBox[{"{",
      RowBox[{"Circle", "[",
       RowBox[{RowBox[{"{", RowBox[{"0", ",", "0"}], "}"}], ",", "1"}], "]"}],
      "}"}], ",",
     RowBox[{"ImageSize", "\[Rule]",
      RowBox[{"{", RowBox[{"200", ",", "120"}], "}"}]}], ",",
     RowBox[{"PlotLabel", "\[Rule]",
      RowBox[{"Style", "[",
       RowBox[{
        RowBox[{"Row", "[",
         RowBox[{"{",
          RowBox[{"n", ",", "\"\< = \>\"", ",",
           RowBox[{"btDigits", "[", "n", "]"}]}], "}"}], "]"}], ",",
        "\"\<Section\>\""}], "]"}]}]}], "]"}], ",",
   RowBox[{"{",
    RowBox[{RowBox[{"{", RowBox[{"n", ",", "61", ",", "\"\<weight\>\""}], "}"}],
     ",", "1", ",", "121", ",", "1"}], "}"}]}], "]"}]], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`n$$ = 61}, \"…\"]"], "Output"]
}, Open]]
}]"##;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);
    let widget = editors
      .iter()
      .find_map(|e| e.manipulate_state.as_ref())
      .expect("the stored Manipulate must instantiate on load");
    assert!(
      widget.error.is_none(),
      "body must evaluate cleanly: {:?}",
      widget.error
    );
    assert!(
      widget.graphics_handle.is_some(),
      "the graphic and its typeset label must draw"
    );
    match &widget.controls[..] {
      [
        manipulate::ControlState::Continuous {
          name,
          label,
          current,
          max,
          ..
        },
      ] => {
        assert_eq!((name.as_str(), label.as_str()), ("n", "weight"));
        assert_eq!((*current, *max), (61.0, 121.0));
      }
      other => panic!("unexpected controls: {other:?}"),
    }
  }

  /// End-to-end regression for the "Force to Overcome Vacuum Pull"
  /// Demonstration: a `Column` of a diagram over two `Show[Plot[…],
  /// Graphics[…]]` panels, each captioned with a styled `FrameLabel`.
  #[test]
  fn vacuum_pull_notebook_opens_with_its_widget() {
    let nb_src = r##"Notebook[{
Cell[CellGroupData[{
Cell[BoxData[
 RowBox[{"Manipulate", "[",
  RowBox[{
   RowBox[{"Column", "[",
    RowBox[{"{",
     RowBox[{
      RowBox[{"Graphics", "[",
       RowBox[{
        RowBox[{"{", RowBox[{"Circle", "[",
          RowBox[{RowBox[{"{", RowBox[{"0", ",", "0"}], "}"}], ",", "d"}],
          "]"}], "}"}], ",",
        RowBox[{"ImageSize", "\[Rule]",
         RowBox[{"{", RowBox[{"200", ",", "120"}], "}"}]}]}], "]"}], ",",
      RowBox[{"Show", "[",
       RowBox[{
        RowBox[{"Plot", "[",
         RowBox[{
          RowBox[{"f", "[", RowBox[{"x", ",", "p"}], "]"}], ",",
          RowBox[{"{", RowBox[{"x", ",", "0.", ",", "60."}], "}"}], ",",
          RowBox[{"Frame", "\[Rule]", "True"}], ",",
          RowBox[{"GridLines", "\[Rule]", "Automatic"}], ",",
          RowBox[{"FrameLabel", "\[Rule]",
           RowBox[{"{",
            RowBox[{
             RowBox[{"{",
              RowBox[{
               RowBox[{"Style", "[",
                RowBox[{"\"\<force (kN)\>\"", ",", "12"}], "]"}], ",",
               "None"}], "}"}], ",",
             RowBox[{"{",
              RowBox[{
               RowBox[{"Style", "[",
                RowBox[{"\"\<diameter (cm)\>\"", ",", "12"}], "]"}], ",",
               "None"}], "}"}]}], "}"}]}], ",",
          RowBox[{"ImageSize", "\[Rule]",
           RowBox[{"{", RowBox[{"330", ",", "160"}], "}"}]}]}], "]"}], ",",
        RowBox[{"Graphics", "[",
         RowBox[{"{",
          RowBox[{
           RowBox[{"PointSize", "[", "0.04", "]"}], ",",
           RowBox[{"Point", "[",
            RowBox[{"{",
             RowBox[{"d", ",", RowBox[{"f", "[", RowBox[{"d", ",", "p"}], "]"}]}],
             "}"}], "]"}]}], "}"}], "]"}]}], "]"}]}], "}"}], "]"}], ",",
   RowBox[{"{",
    RowBox[{
     RowBox[{"{", RowBox[{"d", ",", "30.", ",", "\"\<diameter\>\""}], "}"}],
     ",", "1.", ",", "60."}], "}"}], ",",
   RowBox[{"{",
    RowBox[{
     RowBox[{"{", RowBox[{"p", ",", "20.", ",", "\"\<pressure\>\""}], "}"}],
     ",", "0.", ",", "100."}], "}"}], ",",
   RowBox[{"Initialization", "\[RuleDelayed]",
    RowBox[{"(",
     RowBox[{
      RowBox[{"f", "[",
       RowBox[{"dd_", ",", "pp_"}], "]"}], ":=",
      RowBox[{
       RowBox[{"(", RowBox[{"100.", "-", "pp"}], ")"}], "*",
       RowBox[{"Pi", "/", "8."}], "*",
       SuperscriptBox[
        RowBox[{"(", RowBox[{"dd", "/", "100."}], ")"}], "2"]}]}], ")"}]}]}],
  "]"}]], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`d$$ = 30.}, \"…\"]"], "Output"]
}, Open]]
}]"##;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);
    let widget = editors
      .iter()
      .find_map(|e| e.manipulate_state.as_ref())
      .expect("the stored Manipulate must instantiate on load");
    assert!(
      widget.error.is_none(),
      "body must evaluate cleanly: {:?}",
      widget.error
    );
    assert!(
      widget.graphics_handle.is_some(),
      "the column of diagram and captioned plot must draw"
    );
    match &widget.controls[..] {
      [
        manipulate::ControlState::Continuous {
          name: d,
          current: d_now,
          ..
        },
        manipulate::ControlState::Continuous {
          name: p,
          current: p_now,
          ..
        },
      ] => {
        assert_eq!((d.as_str(), *d_now), ("d", 30.0));
        assert_eq!((p.as_str(), *p_now), ("p", 20.0));
      }
      other => panic!("unexpected controls: {other:?}"),
    }
  }

  /// End-to-end regression for the "Goldbach Conjecture" Demonstration: a
  /// `Column` of the decompositions over a `ListPlot` whose axes are
  /// labelled with explicit ticks.
  #[test]
  fn goldbach_notebook_opens_with_its_widget() {
    let nb_src = r##"Notebook[{
Cell[CellGroupData[{
Cell[BoxData[
 RowBox[{"Manipulate", "[",
  RowBox[{
   RowBox[{"Column", "[",
    RowBox[{
     RowBox[{"{",
      RowBox[{
       RowBox[{"Text", "@",
        RowBox[{"Style", "[", RowBox[{"\"\<counts\>\"", ",", "Bold"}], "]"}]}],
       ",",
       RowBox[{"ListPlot", "[",
        RowBox[{
         RowBox[{"Table", "[",
          RowBox[{
           RowBox[{"Length", "[",
            RowBox[{"Select", "[",
             RowBox[{
              RowBox[{"Range", "[", RowBox[{"2", ",", "k"}], "]"}], ",",
              RowBox[{
               RowBox[{
                RowBox[{"PrimeQ", "[", "#", "]"}], "&&",
                RowBox[{"PrimeQ", "[", RowBox[{"k", "-", "#"}], "]"}]}], "&"}]}],
             "]"}], "]"}], ",",
           RowBox[{"{", RowBox[{"k", ",", "4", ",", "m", ",", "2"}], "}"}]}],
          "]"}], ",",
         RowBox[{"PlotStyle", "\[Rule]",
          RowBox[{"{",
           RowBox[{
            RowBox[{"PointSize", "[", "0.04", "]"}], ",",
            RowBox[{"RGBColor", "[",
             RowBox[{"1", ",", "0.47", ",", "0"}], "]"}]}], "}"}]}], ",",
         RowBox[{"PlotRange", "\[Rule]", "All"}], ",",
         RowBox[{"Ticks", "\[Rule]",
          RowBox[{"{",
           RowBox[{
            RowBox[{"Transpose", "[",
             RowBox[{"{",
              RowBox[{
               RowBox[{"Range", "[", "12", "]"}], ",",
               RowBox[{"2", "+",
                RowBox[{"2", " ", RowBox[{"Range", "[", "12", "]"}]}]}]}], "}"}],
             "]"}], ",", RowBox[{"Range", "[", "6", "]"}]}], "}"}]}], ",",
         RowBox[{"ImageSize", "\[Rule]",
          RowBox[{"{", RowBox[{"370", ",", "280"}], "}"}]}]}], "]"}]}], "}"}],
     ",", RowBox[{"Alignment", "\[Rule]", "Center"}]}], "]"}], ",",
   RowBox[{"{",
    RowBox[{
     RowBox[{"{", RowBox[{"m", ",", "26", ",", "\"\<maximum total\>\""}], "}"}],
     ",", "4", ",", "60", ",", "2"}], "}"}]}], "]"}]], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`m$$ = 26}, \"…\"]"], "Output"]
}, Open]]
}]"##;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);
    let widget = editors
      .iter()
      .find_map(|e| e.manipulate_state.as_ref())
      .expect("the stored Manipulate must instantiate on load");
    assert!(
      widget.error.is_none(),
      "body must evaluate cleanly: {:?}",
      widget.error
    );
    assert!(
      widget.graphics_handle.is_some(),
      "the heading and the plot beneath it must both draw"
    );
    match &widget.controls[..] {
      [
        manipulate::ControlState::Continuous {
          name,
          label,
          current,
          step,
          ..
        },
      ] => {
        assert_eq!((name.as_str(), label.as_str()), ("m", "maximum total"));
        assert_eq!((*current, *step), (26.0, 2.0));
      }
      other => panic!("unexpected controls: {other:?}"),
    }
  }

  /// End-to-end regression for the "Stochastic Model of Microbial Injury and
  /// Mortality" Demonstration. Its `Manipulate` mutates a table in place
  /// through `\[LeftDoubleBracket]…\[RightDoubleBracket]` part
  /// specifications inside a compound `If` body, and draws a `Show` of two
  /// framed plots — the notebook stores no output cell, so the widget is
  /// built from the input the way evaluating the cell does.
  #[test]
  fn microbial_injury_notebook_builds_its_widget() {
    let nb_src = r##"Notebook[{
Cell[BoxData[
 RowBox[{"Manipulate", "[",
  RowBox[{
   RowBox[{"Module", "[",
    RowBox[{
     RowBox[{"{",
      RowBox[{"tns", ",", "p1", ",", "p2"}], "}"}], ",",
     RowBox[{
      RowBox[{"tns", "=",
       RowBox[{"Table", "[",
        RowBox[{
         RowBox[{"{", RowBox[{"i", ",", "n0"}], "}"}], ",",
         RowBox[{"{", RowBox[{"i", ",", "1", ",", "5"}], "}"}]}], "]"}]}], ";",
      RowBox[{"Do", "[",
       RowBox[{
        RowBox[{"If", "[",
         RowBox[{"True", ",",
          RowBox[{
           RowBox[{
            RowBox[{"tns", "\[LeftDoubleBracket]",
             RowBox[{"i", ",", "2"}], "\[RightDoubleBracket]"}], "--"}], ";",
           RowBox[{"q", "=", "i"}]}]}], "]"}], ",",
        RowBox[{"{", RowBox[{"i", ",", "2", ",", "5"}], "}"}]}], "]"}], ";",
      RowBox[{"p1", "=",
       RowBox[{"Plot", "[",
        RowBox[{
         RowBox[{"Exp", "[",
          RowBox[{"-", RowBox[{"t", "/", "10"}]}], "]"}], ",",
         RowBox[{"{", RowBox[{"t", ",", "0", ",", "20"}], "}"}], ",",
         RowBox[{"PlotStyle", "\[Rule]",
          RowBox[{"{", RowBox[{"Thick", ",", "Green"}], "}"}]}], ",",
         RowBox[{"Frame", "\[Rule]", "True"}], ",",
         RowBox[{"FrameLabel", "\[Rule]",
          RowBox[{"{",
           RowBox[{
            RowBox[{"{",
             RowBox[{
              RowBox[{"Subscript", "[",
               RowBox[{"\"\<P\>\"", ",", "\"\<inj\>\""}], "]"}], ",",
              "\"\<\>\""}], "}"}], ",",
            RowBox[{"{", RowBox[{"\"\<t\>\"", ",", "\"\<curve\>\""}], "}"}]}],
           "}"}]}], ",",
         RowBox[{"ImagePadding", "\[Rule]",
          RowBox[{"{",
           RowBox[{
            RowBox[{"{", RowBox[{"45", ",", "10"}], "}"}], ",",
            RowBox[{"{", RowBox[{"45", ",", "20"}], "}"}]}], "}"}]}], ",",
         RowBox[{"ImageSize", "\[Rule]",
          RowBox[{"{", RowBox[{"280", ",", "148"}], "}"}]}]}], "]"}]}], ";",
      RowBox[{"p2", "=",
       RowBox[{"ListPlot", "[",
        RowBox[{"tns", ",",
         RowBox[{"Joined", "\[Rule]", "True"}], ",",
         RowBox[{"PlotStyle", "\[Rule]", "Red"}]}], "]"}]}], ";",
      RowBox[{"Show", "[", RowBox[{"p1", ",", "p2"}], "]"}]}]}], "]"}], ",",
   RowBox[{"{",
    RowBox[{
     RowBox[{"{",
      RowBox[{"n0", ",", "100", ",", "\"\<initial count\>\""}], "}"}], ",",
     "10", ",", "200", ",", "10"}], "}"}]}], "]"}]], "Input"]
}]"##;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);
    let code = editors
      .iter()
      .map(|e| e.content.text())
      .find(|t| t.starts_with("Manipulate["))
      .expect("the Manipulate cell must load");
    let widget = instantiate_stored_manipulate(&code, "")
      .expect("the Manipulate must instantiate");
    assert!(
      widget.error.is_none(),
      "body must evaluate cleanly: {:?}",
      widget.error
    );
    assert!(
      widget.graphics_handle.is_some(),
      "the merged plot must draw"
    );
    match &widget.controls[..] {
      [
        manipulate::ControlState::Continuous {
          name,
          label,
          current,
          step,
          ..
        },
      ] => {
        assert_eq!((name.as_str(), label.as_str()), ("n0", "initial count"));
        assert_eq!((*current, *step), (100.0, 10.0));
      }
      other => panic!("unexpected controls: {other:?}"),
    }
  }

  /// End-to-end regression for the "Sampling a Digital Signal"
  /// Demonstration: a `Grid` of two `ListPlot`s with `Filling -> Axis` and
  /// `ImagePadding`, driven by an `Initialization` block. The notebook stores
  /// no output cell, so the widget is built from the input the way evaluating
  /// the cell does.
  #[test]
  fn sampling_a_digital_signal_notebook_builds_its_widget() {
    let nb_src = r##"Notebook[{
Cell[CellGroupData[{
Cell[BoxData[
 RowBox[{"Manipulate", "[", "\[IndentingNewLine]", 
  RowBox[{
   RowBox[{
    RowBox[{"If", "[", 
     RowBox[{
      RowBox[{"k", ">", 
       RowBox[{"50", " ", "L"}]}], ",", 
      RowBox[{"k", "=", 
       RowBox[{"50", " ", "L"}]}]}], "]"}], ";", "\[IndentingNewLine]", 
    RowBox[{"Grid", "[", "\[IndentingNewLine]", 
     RowBox[{"{", "\[IndentingNewLine]", 
      RowBox[{
       RowBox[{"{", 
        RowBox[{"ListPlot", "[", 
         RowBox[{"tmp", ",", 
          RowBox[{"Filling", " ", "\[Rule]", " ", "Axis"}], ",", 
          RowBox[{"PlotMarkers", "\[Rule]", "Automatic"}], ",", 
          RowBox[{"ImagePadding", " ", "\[Rule]", " ", "20"}], ",", 
          RowBox[{"ImageSize", " ", "\[Rule]", " ", 
           RowBox[{"{", 
            RowBox[{"400", ",", "200"}], "}"}]}]}], "]"}], "}"}], ",", 
       RowBox[{"{", 
        RowBox[{"ListPlot", "[", 
         RowBox[{
          RowBox[{"If", "[", 
           RowBox[{
            RowBox[{"sampler", " ", "\[Equal]", " ", "\"\<up\>\""}], ",", 
            RowBox[{"f", "[", "L", "]"}], ",", 
            RowBox[{"f1", "[", "L", "]"}]}], "]"}], ",", 
          RowBox[{"Filling", " ", "\[Rule]", " ", "Axis"}], ",", 
          RowBox[{"PlotRange", " ", "\[Rule]", " ", 
           RowBox[{"{", 
            RowBox[{
             RowBox[{"{", 
              RowBox[{"0", ",", "k"}], "}"}], ",", 
             RowBox[{"{", 
              RowBox[{
               RowBox[{"-", "1"}], ",", "1"}], "}"}]}], "}"}]}], ",", 
          RowBox[{"PlotMarkers", "\[Rule]", "Automatic"}], ",", 
          RowBox[{"ImagePadding", " ", "\[Rule]", " ", "20"}], ",", 
          RowBox[{"ImageSize", " ", "\[Rule]", " ", 
           RowBox[{"{", 
            RowBox[{"400", ",", "200"}], "}"}]}]}], "]"}], "}"}]}], 
      "\[IndentingNewLine]", "}"}], "\[IndentingNewLine]", "]"}]}], ",", 
   "\[IndentingNewLine]", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"L", ",", "2", ",", "\"\<sampling by integer factor L\>\""}], 
      "}"}], ",", "2", ",", "10", ",", "1", ",", 
     RowBox[{"Appearance", "\[Rule]", "\"\<Labeled\>\""}]}], "}"}], ",", 
   "\[IndentingNewLine]", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"k", ",", "50", ",", "\"\<bottom plot range\>\""}], "}"}], ",", 
     "1", ",", 
     RowBox[{"50", " ", "L"}], ",", "1", ",", 
     RowBox[{"Appearance", "\[Rule]", "\"\<Labeled\>\""}]}], "}"}], ",", 
   "\[IndentingNewLine]", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"sampler", ",", "\"\<up\>\""}], "}"}], ",", 
     RowBox[{"{", 
      RowBox[{"\"\<up\>\"", ",", "\"\<down\>\""}], "}"}]}], "}"}], ",", 
   "\[IndentingNewLine]", 
   RowBox[{"TrackedSymbols", "\[Rule]", 
    RowBox[{"{", 
     RowBox[{"L", ",", "k", ",", "sampler"}], "}"}]}], ",", 
   "\[IndentingNewLine]", 
   RowBox[{"Initialization", " ", "\[RuleDelayed]", " ", 
    RowBox[{"{", "\[IndentingNewLine]", 
     RowBox[{
      RowBox[{"tmp", " ", "=", " ", 
       RowBox[{
        RowBox[{"Table", "[", 
         RowBox[{
          RowBox[{"Sin", "[", "x", "]"}], ",", 
          RowBox[{"{", 
           RowBox[{"x", ",", "0", ",", " ", 
            RowBox[{"720", " ", "Degree"}], ",", " ", 
            RowBox[{"15", " ", "Degree"}]}], "}"}]}], "]"}], "//", "N"}]}], 
      ";", "\[IndentingNewLine]", 
      RowBox[{
       RowBox[{"f", "[", "L1_", "]"}], ":=", 
       RowBox[{"Flatten", "[", 
        RowBox[{"Riffle", "[", 
         RowBox[{"tmp", ",", 
          RowBox[{"{", 
           RowBox[{"ConstantArray", "[", 
            RowBox[{"0", ",", 
             RowBox[{"L1", "-", "1"}]}], "]"}], "}"}]}], "]"}], "]"}]}], ";", 
      "\[IndentingNewLine]", 
      RowBox[{
       RowBox[{"f1", "[", "L2_", "]"}], " ", ":=", " ", 
       RowBox[{"tmp", "[", 
        RowBox[{"[", 
         RowBox[{"1", ";;", " ", ";;", "L2"}], "]"}], "]"}]}], ";"}], 
     "\[IndentingNewLine]", "}"}]}]}], "\[IndentingNewLine]", "]"}]], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`L$$ = 2, $CellContext`k$$ = 50, \
$CellContext`sampler$$ = \"up\"}, \"\[Ellipsis]\"]"], "Output"]
}, Open]]
}]"##;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);
    let widget = editors
      .iter()
      .find_map(|e| e.manipulate_state.as_ref())
      .expect("the Manipulate cell must instantiate on load");
    assert!(
      widget.error.is_none(),
      "body must evaluate cleanly: {:?}",
      widget.error
    );
    assert!(
      widget.graphics_handle.is_some(),
      "both stacked plots must draw"
    );
    match &widget.controls[..] {
      [
        manipulate::ControlState::Continuous {
          name: l,
          label: l_label,
          current: l_now,
          ..
        },
        manipulate::ControlState::Continuous {
          name: k,
          current: k_now,
          max: k_max,
          ..
        },
        manipulate::ControlState::Discrete {
          name: sampler,
          values,
          ..
        },
      ] => {
        assert_eq!(
          (l.as_str(), l_label.as_str(), *l_now),
          ("L", "sampling by integer factor L", 2.0)
        );
        // `50 L` sizes the second slider from the first control's value.
        assert_eq!((k.as_str(), *k_now, *k_max), ("k", 50.0, 100.0));
        assert_eq!(sampler.as_str(), "sampler");
        assert_eq!(values.len(), 2);
      }
      other => panic!("unexpected controls: {other:?}"),
    }
  }

  /// End-to-end regression for "The Price of a Call Option on Electrical
  /// Power": four `Initialization Code` cells define the option formula in
  /// Unicode notation (`\[ExponentialE]`, `\[Sigma]`, `\[ScriptCapitalN]`),
  /// and the `Manipulate` plots it with an `AxesLabel` and a dashed `Epilog`.
  #[test]
  fn call_option_notebook_builds_its_widget() {
    let nb_src = r##"Notebook[{
Cell[BoxData["d1[S_,K_,\[Sigma]_,\[Alpha]_,f_,T_,t_]:=With[{s=\[ExponentialE]^(f[#1])&},(\[ExponentialE]^(-\[Alpha] (T-t)) Log[S/s[t]]+(\[Sigma]^2 (1-\[ExponentialE]^(-2 \[Alpha] (T-t))))/(4 \[Alpha])+Log[s[T]/K])/Sqrt[(\[Sigma]^2 (1-\[ExponentialE]^(-2 \[Alpha] (T-t))))/(4 \[Alpha])]]"], "Input"],
Cell[BoxData["d2[S_,K_,\[Sigma]_,\[Alpha]_,f_,T_,t_]:=With[{s=\[ExponentialE]^(f[#1])&},(\[ExponentialE]^(-\[Alpha] (T-t)) Log[S/s[t]]+Log[s[T]/K])/Sqrt[(\[Sigma]^2 (1-\[ExponentialE]^(-2 \[Alpha] (T-t))))/(4 \[Alpha])]]"], "Input"],
Cell[BoxData["\[ScriptCapitalN][z_]:=(1+Erf[z/Sqrt[2]])/2"], "Input"],
Cell[BoxData["callValue[S_,K_,\[Sigma]_,\[Alpha]_,f_,T_,t_,r_]:=With[{s=\[ExponentialE]^(f[#1])&},\[ExponentialE]^(-r (T-t)) (s[T] (S/s[t])^(\[ExponentialE]^(-\[Alpha] (T-t))) Exp[\[Sigma]^2/(4 \[Alpha]) (1-Exp[-2 \[Alpha] (T-t)])] \[ScriptCapitalN][d1[S,K,\[Sigma],\[Alpha],f,T,t]]-K \[ScriptCapitalN][d2[S,K,\[Sigma],\[Alpha],f,T,t]])]"], "Input"],
Cell[CellGroupData[{
Cell[BoxData[
 RowBox[{"Manipulate", "[", "\[IndentingNewLine]", 
  RowBox[{
   RowBox[{"Module", "[", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"f", "=", 
       RowBox[{
        RowBox[{"a", "+", 
         RowBox[{"b", " ", 
          RowBox[{"Cos", "[", 
           RowBox[{"#", " ", "730", 
            RowBox[{"Pi", "/", "7"}]}], " ", "]"}]}]}], "&"}]}], "}"}], ",", 
     "\[IndentingNewLine]", 
     RowBox[{"Plot", "[", 
      RowBox[{
       RowBox[{"callValue", "[", 
        RowBox[{
        "s", ",", "1", ",", "\[Sigma]", ",", "\[Alpha]", ",", "f", ",", "1", 
         ",", "t", ",", "r"}], "]"}], ",", 
       RowBox[{"{", 
        RowBox[{"s", ",", "0.5", ",", "1.5"}], "}"}], ",", 
       RowBox[{"PlotRange", "\[Rule]", 
        RowBox[{"{", 
         RowBox[{
          RowBox[{"{", 
           RowBox[{"0.5", ",", "1.5"}], "}"}], ",", 
          RowBox[{"{", 
           RowBox[{"0", ",", "v"}], "}"}]}], "}"}]}], ",", 
       RowBox[{"AxesLabel", "\[Rule]", 
        RowBox[{"{", 
         RowBox[{
         "\"\<electrical power spot price\>\"", ",", "\"\<call value\>\""}], 
         "}"}]}], ",", 
       RowBox[{"Epilog", "\[Rule]", 
        RowBox[{"{", 
         RowBox[{
          RowBox[{"Dashing", "[", 
           RowBox[{"{", "0.01", "}"}], "]"}], ",", 
          RowBox[{"Line", "[", 
           RowBox[{"{", 
            RowBox[{
             RowBox[{"{", 
              RowBox[{"1", ",", "0"}], "}"}], ",", 
             RowBox[{"{", 
              RowBox[{"1", ",", "v"}], "}"}]}], "}"}], "]"}]}], "}"}]}], ",", 
       
       RowBox[{"ImageSize", "\[Rule]", 
        RowBox[{"{", 
         RowBox[{"500", ",", "300"}], "}"}]}]}], "]"}]}], "]"}], ",", 
   "\[IndentingNewLine]", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"\[Sigma]", ",", "0.7", ",", "\"\<volatility\>\""}], "}"}], ",",
      "0.01", ",", "1", ",", ".01", ",", 
     RowBox[{"Appearance", "\[Rule]", "\"\<Labeled\>\""}]}], "}"}], ",", 
   "\[IndentingNewLine]", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"\[Alpha]", ",", "1", ",", "\"\<rate of mean reversion\>\""}], 
      "}"}], ",", "0.01", ",", "2", ",", ".01", ",", 
     RowBox[{"Appearance", "\[Rule]", "\"\<Labeled\>\""}]}], "}"}], ",", 
   "\[IndentingNewLine]", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"t", ",", ".5", ",", "\"\<time\>\""}], "}"}], ",", "0", ",", 
     RowBox[{"364", "/", "365"}], ",", ".01", ",", 
     RowBox[{"Appearance", "\[Rule]", "\"\<Labeled\>\""}]}], "}"}], ",", 
   "\[IndentingNewLine]", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"r", ",", "0", ",", "\"\<interest rate\>\""}], "}"}], ",", "0", 
     ",", "0.2", ",", ".01", ",", 
     RowBox[{"Appearance", "\[Rule]", "\"\<Labeled\>\""}]}], "}"}], ",", 
   "\[IndentingNewLine]", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"a", ",", "0.3", ",", "\"\<constant seasonal component\>\""}], 
      "}"}], ",", "0", ",", "1", ",", ".01", ",", 
     RowBox[{"Appearance", "\[Rule]", "\"\<Labeled\>\""}]}], "}"}], ",", 
   "\[IndentingNewLine]", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"b", ",", "0.5", ",", "\"\<periodic seasonal component\>\""}], 
      "}"}], ",", "0", ",", "1", ",", ".01", ",", 
     RowBox[{"Appearance", "\[Rule]", "\"\<Labeled\>\""}]}], "}"}], ",", 
   "\[IndentingNewLine]", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"v", ",", "2", ",", "\"\<vertical range\>\""}], "}"}], ",", 
     "0.3", ",", "3", ",", ".01", ",", 
     RowBox[{"Appearance", "\[Rule]", "\"\<Labeled\>\""}]}], "}"}], ",", 
   "\[IndentingNewLine]", 
   RowBox[{"AutorunSequencing", "\[Rule]", 
    RowBox[{"{", 
     RowBox[{"1", ",", "3", ",", "5", ",", "7"}], "}"}]}], ",", 
   "\[IndentingNewLine]", 
   RowBox[{"SaveDefinitions", "\[Rule]", "True"}]}], "]"}]], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`\[Sigma]$$ = 0.7}, \"\[Ellipsis]\"]"], "Output"]
}, Open]]
}]"##;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);
    let widget = editors
      .iter()
      .find_map(|e| e.manipulate_state.as_ref())
      .expect("the Manipulate cell must instantiate on load");
    assert!(
      widget.error.is_none(),
      "the option formula must evaluate: {:?}",
      widget.error
    );
    assert!(widget.graphics_handle.is_some(), "the curve must draw");
    let names: Vec<&str> = widget
      .controls
      .iter()
      .map(|c| match c {
        manipulate::ControlState::Continuous { name, .. } => name.as_str(),
        other => panic!("unexpected control: {other:?}"),
      })
      .collect();
    assert_eq!(
      names,
      ["\u{3c3}", "\u{3b1}", "t", "r", "a", "b", "v"],
      "one slider per parameter, Greek names included"
    );
  }

  /// End-to-end regression for "Merging Schools of Fish": the swarms are
  /// built by assigning to a *list* of downvalue patterns
  /// (`{vectorField1[t_], vectorField2[t_]} = Table[…]`), then mapped through
  /// `@@@`, `Transpose` and matrix dot products into 120 translucent
  /// polygons. The notebook's 700-point fish outline is replaced here by a
  /// triangle; everything else is its own code.
  #[test]
  fn merging_schools_of_fish_notebook_builds_its_widget() {
    let nb_src = r##"Notebook[{
Cell[BoxData["fish[mp_, size_] := Polygon[(mp + 6 size (# - {0.5, 0.5})) & /@ {{0.7, 0.5}, {0.3, 0.55}, {0.3, 0.45}}]"], "Input"],
Cell[BoxData["{vectorField1[t_], vectorField2[t_]} = \nTable[\nModule[{φ = Sum[RandomReal[{-1, 1}] Sin[j t +2Pi RandomReal[]], {j, 6}]},\n              {{{Cos[φ], Sin[φ]}, {-Sin[φ], Cos[φ]}},\n              Table[ 2  Sum[RandomReal[{-1, 1}] Sin[j t ], {j, 6}], {2}]}\n            ],{2}];"], "Input"],
Cell[BoxData["{internalRotationField1[t_], internalRotationField2[t_]} = \nModule[{φ},\nTable[φ = Sum[RandomReal[{-1, 1}] Sin[j t +2Pi RandomReal[]], {j, 6}];\n            {{Cos[φ], Sin[φ]}, {-Sin[φ], Cos[φ]}}, {#}]]& /@ {60, 60};"], "Input"],
Cell[BoxData["{fishSwarm1Initial, fishSwarm2Initial} = \n               {RandomReal[{-1, 1}, {60, 2}], RandomReal[{-1, 1}, {60, 2}]} ;"], "Input"],
Cell[BoxData["r0=0.1;"], "Input"],
Cell[BoxData["swarm1Colors = Hue/@RandomReal[{-0.05, 0.05}, {60}];\nswarm2Colors = Hue/@RandomReal[0.7 + {-0.05, 0.05}, {60}];"], "Input"],
Cell[BoxData["swarm1Sizes = r0 RandomReal[1 + {-0.2, 0.2}, {60}];\nswarm2Sizes = r0 RandomReal[1 + {-0.2, 0.2}, {60}];"], "Input"],
Cell[BoxData["fishSwarm1[t_] :=\nModule[{ℛ, 𝒯,ℛis},\n             {ℛ, 𝒯}=vectorField1[t];\n             ℛis = internalRotationField1[t];\n             {Opacity[0.4+ 0.6 ArcTan[2t]/(Pi/2)],#}& /@\n             Transpose[{swarm1Colors, fish[#2, #1]& @@@\n              Transpose[{(1 + 4 Exp[-t])swarm1Sizes, (ℛ.#+ 0 𝒯)& /@\n            (#1.#2&@@@ Transpose[{ℛis, fishSwarm1Initial}])}]}]\n           \n         ]"], "Input"],
Cell[BoxData["fishSwarm2[t_] :=\nModule[{ℛ, 𝒯,ℛis},\n             {ℛ, 𝒯}=vectorField2[t];\n             ℛis = internalRotationField2[t];\n             {Opacity[0.4+ 0.6 ArcTan[2t]/(Pi/2)],#}& /@\n             Transpose[{swarm2Colors, fish[#2, #1]& @@@\n              Transpose[{(1 + 4 Exp[-t])swarm2Sizes, (ℛ.#+ 𝒯)& /@\n            (#1.#2&@@@ Transpose[{ℛis, fishSwarm2Initial}])}]}]\n   ]"], "Input"],
Cell[CellGroupData[{
Cell[BoxData["Manipulate[\nGraphics[\nSeedRandom[1];\nRandomSample[ Join[fishSwarm1[t], fishSwarm2[t]]], PlotRange -> 4,ImageSize->{450,450}],\n{{t,0,\"time\"}, 0, 2},\nSaveDefinitions -> True]"], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`t$$ = 0}, \"\\[Ellipsis]\"]"], "Output"]
}, Open]]
}]"##;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);
    let widget = editors
      .iter()
      .find_map(|e| e.manipulate_state.as_ref())
      .expect("the Manipulate cell must instantiate on load");
    assert!(
      widget.error.is_none(),
      "the swarm definitions must evaluate: {:?}",
      widget.error
    );
    assert!(widget.graphics_handle.is_some(), "the swarms must draw");
    match &widget.controls[..] {
      [
        manipulate::ControlState::Continuous {
          name,
          label,
          current,
          min,
          max,
          ..
        },
      ] => {
        assert_eq!((name.as_str(), label.as_str()), ("t", "time"));
        assert_eq!((*current, *min, *max), (0.0, 0.0, 2.0));
      }
      other => panic!("unexpected controls: {other:?}"),
    }
    // Both swarms reach the canvas: 60 fish each, drawn as polygons.
    let svg = woxi::interpret_with_stdout(&format!("t = 0;\n{}", widget.body))
      .expect("the body must render")
      .graphics
      .expect("the body must produce a graphic");
    assert_eq!(svg.matches("<polygon").count(), 120, "{svg}");
  }

  /// End-to-end regression for "Non Placet Net of a Dodecahedron". Its
  /// coordinate tables are written with `*^` exponents on precision-tagged
  /// reals (`-1.1102230246251565`*^-16`), and its four sliders fold the net
  /// with the four-argument `Rotate[g, theta, axis, point]`.
  #[test]
  fn dodecahedron_net_notebook_builds_its_widget() {
    let nb_src = r##"Notebook[{
Cell[BoxData["grafikaN={Line[{{1.`,0.`},{1.3090169943749475`,0.9510565162951536`},{0.5`,1.538841768587627`},{-0.3090169943749475`,0.9510565162951539`},{-1.1102230246251565`*^-16,2.220446049250313`*^-16},{1.`,0.`}}],Line[{{1.`,0.`},{1.8090169943749475`,-0.5877852522924732`},{2.618033988749895`,-1.1102230246251565`*^-16},{2.3090169943749475`,0.9510565162951534`},{1.3090169943749475`,0.9510565162951535`},{1.`,0.`}}],Line[{{2.618033988749895`,-1.1102230246251565`*^-16},{3.618033988749895`,0.`},{3.9270509831248424`,0.9510565162951534`},{3.118033988749895`,1.5388417685876263`},{2.3090169943749475`,0.9510565162951536`},{2.618033988749895`,-1.1102230246251565`*^-16}}],Line[{{3.118033988749895`,1.5388417685876263`},{2.809016994374948`,2.4898982848827806`},{1.8090169943749475`,2.4898982848827806`},{1.4999999999999998`,1.538841768587627`},{2.309016994374947`,0.9510565162951532`},{3.118033988749895`,1.5388417685876263`}}],Line[{{3.118033988749895`,1.5388417685876263`},{4.118033988749896`,1.5388417685876252`},{4.427050983124844`,2.4898982848827798`},{3.6180339887498967`,3.077683537175253`},{2.809016994374948`,2.489898284882781`},{3.118033988749895`,1.5388417685876263`}}],Line[{{3.6180339887498967`,3.077683537175253`},{3.30901699437495`,4.028740053470408`},{2.30901699437495`,4.028740053470409`},{2.000000000000001`,3.077683537175256`},{2.8090169943749475`,2.489898284882781`},{3.6180339887498967`,3.077683537175253`}}],Line[{{0.5`,1.538841768587627`},{0.19098300562505233`,2.4898982848827806`},{-0.8090169943749477`,2.4898982848827815`},{-1.1180339887498951`,1.538841768587627`},{-0.3090169943749477`,0.9510565162951539`},{0.5`,1.538841768587627`}}],Line[{{-0.8090169943749477`,2.4898982848827815`},{-1.6180339887498958`,3.077683537175255`},{-2.427050983124844`,2.4898982848827815`},{-2.118033988749896`,1.5388417685876261`},{-1.118033988749895`,1.538841768587627`},{-0.8090169943749477`,2.4898982848827815`}}],Line[{{-2.427050983124844`,2.4898982848827815`},{-3.4270509831248464`,2.4898982848827815`},{-3.7360679774997942`,1.5388417685876252`},{-2.9270509831248446`,0.9510565162951512`},{-2.118033988749896`,1.5388417685876261`},{-2.427050983124844`,2.4898982848827815`}}],Line[{{-2.9270509831248446`,0.9510565162951512`},{-2.618033988749896`,-4.884981308350689`*^-15},{-1.6180339887498936`,-4.440892098500626`*^-15},{-1.3090169943749452`,0.951056516295152`},{-2.1180339887498953`,1.5388417685876261`},{-2.9270509831248446`,0.9510565162951512`}}],Line[{{-2.9270509831248446`,0.9510565162951512`},{-3.927050983124847`,0.9510565162951501`},{-4.236067977499795`,-5.773159728050814`*^-15},{-3.4270509831248455`,-0.5877852522924805`},{-2.6180339887498953`,-4.440892098500626`*^-15},{-2.9270509831248446`,0.9510565162951512`}}],Line[{{-3.4270509831248455`,-0.5877852522924805`},{-3.1180339887498967`,-1.5388417685876368`},{-2.118033988749893`,-1.5388417685876368`},{-1.8090169943749457`,-0.5877852522924796`},{-2.618033988749896`,-5.329070518200751`*^-15},{-3.4270509831248455`,-0.5877852522924805`}}]}/.{{x_,y_}->{x,y,0},Line->Polygon};"], "Input"],
Cell[BoxData["koordinateN={{0,0},{1,0},{1.8090169943749475`,-0.5877852522924732`},{2.618033988749895`,-1.1102230246251565`*^-16},{3.618033988749895`,0.`},{3.9270509831248424`,0.9510565162951534`},{3.118033988749895`,1.5388417685876263`},{4.118033988749896`,1.5388417685876252`},{4.427050983124844`,2.4898982848827798`},{3.6180339887498967`,3.077683537175253`},{3.30901699437495`,4.028740053470408`},{2.30901699437495`,4.028740053470409`},{2.000000000000001`,3.077683537175256`},{2.809016994374948`,2.4898982848827806`},{1.8090169943749475`,2.4898982848827806`},{1.4999999999999998`,1.538841768587627`},{2.3090169943749475`,0.9510565162951534`},{1.3090169943749475`,0.9510565162951536`},{0.5`,1.538841768587627`},{0.19098300562505233`,2.4898982848827806`},{-0.8090169943749477`,2.4898982848827815`},{-1.6180339887498958`,3.077683537175255`},{-2.427050983124844`,2.4898982848827815`},{-3.4270509831248464`,2.4898982848827815`},{-3.7360679774997942`,1.5388417685876252`},{-2.9270509831248446`,0.9510565162951512`},{-3.927050983124847`,0.9510565162951501`},{-4.236067977499795`,-5.773159728050814`*^-15},{-3.4270509831248455`,-0.5877852522924805`},{-3.1180339887498967`,-1.5388417685876368`},{-2.118033988749893`,-1.5388417685876368`},{-1.8090169943749457`,-0.5877852522924796`},{-2.618033988749896`,-4.884981308350689`*^-15},{-1.6180339887498936`,-4.440892098500626`*^-15},{-1.3090169943749452`,0.951056516295152`},{-2.118033988749896`,1.5388417685876261`},{-1.1180339887498951`,1.538841768587627`},{-0.3090169943749475`,0.9510565162951539`}}/.{x_,y_}->{x,y,0};"], "Input"],
Cell[BoxData["fi=ArcCos[1/Sqrt[5]]//N;"], "Input"],
Cell[CellGroupData[{
Cell[BoxData["Manipulate[Graphics3D[{Rotate[{grafikaN[[7]],Rotate[{grafikaN[[8]],Rotate[{grafikaN[[9]],Rotate[{grafikaN[[10]],Rotate[{grafikaN[[11]],Rotate[grafikaN[[12]],k3 fi,koordinateN[[33]]-koordinateN[[29]],koordinateN[[29]]]},k3 fi,koordinateN[[33]]-koordinateN[[26]],koordinateN[[26]]]},-k3 fi,koordinateN[[26]]-koordinateN[[36]],koordinateN[[36]]]},k3 fi,koordinateN[[36]]-koordinateN[[23]],koordinateN[[23]]]},k4 fi,koordinateN[[21]]-koordinateN[[37]],koordinateN[[37]]]},k5 fi,koordinateN[[38]]-koordinateN[[10]],koordinateN[[10]]],grafikaN[[1]],Rotate[{grafikaN[[2]],Rotate[{grafikaN[[3]],Rotate[{grafikaN[[4]],Rotate[{grafikaN[[5]],Rotate[grafikaN[[6]],k2 fi,koordinateN[[14]]-koordinateN[[10]],koordinateN[[10]]]},k2 fi,koordinateN[[14]]-koordinateN[[7]],koordinateN[[7]]]},k2 fi,koordinateN[[17]]-koordinateN[[7]],koordinateN[[7]]]},k2 fi,koordinateN[[17]]-koordinateN[[4]],koordinateN[[4]]]},-k2 fi,koordinateN[[2]]-koordinateN[[18]],koordinateN[[18]]]},ImageSize->{450,450},ViewAngle->20 Degree,SphericalRegion->True,Boxed->False,ViewPoint->0.5{1,1,6},PlotRange->5],\n{{k2,0,\"fold 1\"},-1,1,ImageSize->Tiny},\n{{k3,0,\"fold 2\"},-1,1,ImageSize->Tiny},\n{{k4,0,\"fold 3\"},-1,1,ImageSize->Tiny},\n{{k5,0,\"fold 4\"},-1,1,ImageSize->Tiny},ControlPlacement->Left,SaveDefinitions->True]"], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`k2$$ = 0}, \"\\[Ellipsis]\"]"], "Output"]
}, Open]]
}]"##;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);
    let widget = editors
      .iter()
      .find_map(|e| e.manipulate_state.as_ref())
      .expect("the Manipulate cell must instantiate on load");
    assert!(
      widget.error.is_none(),
      "the coordinate tables must load: {:?}",
      widget.error
    );
    assert!(widget.graphics_handle.is_some(), "the net must draw");
    let folds: Vec<&str> = widget
      .controls
      .iter()
      .map(|c| match c {
        manipulate::ControlState::Continuous { label, .. } => label.as_str(),
        other => panic!("unexpected control: {other:?}"),
      })
      .collect();
    assert_eq!(folds, ["fold 1", "fold 2", "fold 3", "fold 4"]);

    // Folding actually moves the flaps: the twelve pentagons of the flat net
    // are drawn somewhere else once a fold is applied.
    let render = |k: &str| {
      woxi::interpret_with_stdout(&format!(
        "k2 = {k}; k3 = {k}; k4 = {k}; k5 = {k};\n{}",
        widget.body
      ))
      .expect("the body must render")
      .graphics
      .expect("the body must produce a graphic")
    };
    let flat = render("0");
    let folded = render("0.4");
    assert!(flat.contains("<polygon"), "{flat}");
    assert_ne!(flat, folded, "the sliders must fold the net");
  }

  /// End-to-end regression for "Deciding Rain-Affected Cricket Matches: The
  /// Duckworth-Lewis Method". Its controls live inside a `TabView`, and its
  /// scoreboard is a `Grid` of `StyleForm` cells with `SpanFromLeft` spans
  /// on a dark background. (The notebook's 18 KB of resource tables are
  /// trimmed here; the structure is its own.)
  #[test]
  fn duckworth_lewis_notebook_builds_its_widget() {
    let nb_src = r##"Notebook[{
Cell[BoxData["runRate[s_, o_] := If[o == 0, 0., N[Round[s/o, 0.01]]];"], "Input"],
Cell[CellGroupData[{
Cell[BoxData["Manipulate[\nGrid[{\n{StyleForm[\"This is a \"<>ToString[totalOvers]<>\" over match\",FontWeight->Bold,FontSize->16,FontColor->GrayLevel[1]],SpanFromLeft},\n{StyleForm[\"Team 1: \",FontWeight->Bold,FontSize->16,FontColor->GrayLevel[1]],StyleForm[ToString[team1Score]<>\" for \"<>ToString[team1Wickets],FontWeight->Bold,FontSize->16,FontColor->GrayLevel[1]]},\n{StyleForm[\"RR: \",FontWeight->Bold,FontSize->16,FontColor->GrayLevel[1]],StyleForm[ToString[runRate[team1Score,team1Overs]],FontWeight->Bold,FontSize->16,FontColor->GrayLevel[1]]}\n},Background->Black,Alignment->Left],\nColumn[{TabView[{\nStyle[\"The Match\",Bold]->Grid[{{Control[{{totalOvers,50,\"Overs per innings:\"},0,50,1,Appearance->\"Labeled\"}],SpanFromLeft},{Control[{{team1Score,0,\"Team 1:\"},0,400,1,Appearance->\"Labeled\"}],Control[{{team1Wickets,0,\"for\"},0,10,1,Appearance->\"Labeled\"}]}}],\nStyle[\"The Interruption\",Bold]->Column[{Control[{{team1Overs,25,\"Overs:\"},0,50,1,Appearance->\"Labeled\"}]}]}],\nControl[{{display,1,\"Display\"},{1->\"Scoreboard\",2->\"Resources\"}}]}],\nSaveDefinitions->True]"], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`totalOvers$$ = 50}, \"\\[Ellipsis]\"]"], "Output"]
}, Open]]
}]"##;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);
    let widget = editors
      .iter()
      .find_map(|e| e.manipulate_state.as_ref())
      .expect("the Manipulate cell must instantiate on load");
    assert!(
      widget.error.is_none(),
      "the scoreboard must evaluate: {:?}",
      widget.error
    );
    assert!(widget.graphics_handle.is_some(), "the scoreboard must draw");
    // Every tab's controls are found, not just the one outside the TabView.
    let names: Vec<&str> = widget
      .controls
      .iter()
      .map(|c| match c {
        manipulate::ControlState::Continuous { name, .. } => name.as_str(),
        manipulate::ControlState::Discrete { name, .. } => name.as_str(),
        other => panic!("unexpected control: {other:?}"),
      })
      .collect();
    assert_eq!(
      names,
      [
        "totalOvers",
        "team1Score",
        "team1Wickets",
        "team1Overs",
        "display"
      ]
    );

    let svg = woxi::interpret_with_stdout(&format!(
      "totalOvers = 50; team1Score = 120; team1Wickets = 3; \
       team1Overs = 25; display = 1;\n{}",
      widget.body
    ))
    .expect("the body must render")
    .graphics
    .expect("the body must produce a graphic");
    // The StyleForm cells render their content, not their own source, and
    // the SpanFromLeft placeholder draws nothing.
    assert!(!svg.contains("StyleForm"), "{svg}");
    assert!(!svg.contains("SpanFromLeft"), "{svg}");
    assert!(svg.contains("This is a 50 over match"), "{svg}");
    assert!(svg.contains(">120 for 3</text>"), "{svg}");
    assert!(
      svg.contains(">4.8</text>"),
      "the run rate must compute: {svg}"
    );
  }

  /// "Miscible Displacement of Oil in Heterogenous Porous Media" writes its
  /// transport equation with the typeset partial-derivative operator
  /// (`SubscriptBox["\\[PartialD]", …]`), which the notebook reader turns
  /// into `D[…]`. Before that the cell did not parse at all, so none of its
  /// controls were found.
  ///
  /// The body itself still needs a finite-element PDE solver — `NDSolve`
  /// handles ODEs only and `NeumannValue` is unimplemented — so this checks
  /// the cell loads and its controls are built, not that it draws.
  #[test]
  fn miscible_displacement_notebook_reads_its_partial_derivatives() {
    let nb_src = r##"Notebook[{
Cell[BoxData[
 RowBox[{"Manipulate", "[", 
  RowBox[{
   RowBox[{"Module", "[", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{
       RowBox[{"L", "=", "1"}], ",", 
       RowBox[{"c0", "=", "0"}], ",", 
       RowBox[{"s0", "=", "0"}], ",", 
       RowBox[{"cin", "=", "1"}], ",", "sol", ",", "Ints", ",", "ptime", ",", 
       "pspace"}], "}"}], ",", "\[IndentingNewLine]", "\[IndentingNewLine]", 
     RowBox[{
      RowBox[{"sol", "=", 
       RowBox[{"NDSolve", "[", 
        RowBox[{
         RowBox[{"{", "\[IndentingNewLine]", 
          RowBox[{
           RowBox[{
            RowBox[{
             RowBox[{"\[ScriptCapitalD]", 
              RowBox[{
               SubscriptBox["\[PartialD]", 
                RowBox[{"x", ",", "x"}]], 
               RowBox[{"c", "[", 
                RowBox[{"x", ",", "t"}], "]"}]}]}], "-", 
             RowBox[{"u", 
              RowBox[{
               SubscriptBox["\[PartialD]", "x"], 
               RowBox[{"c", "[", 
                RowBox[{"x", ",", "t"}], "]"}]}]}], "-", 
             RowBox[{
              RowBox[{"(", 
               RowBox[{"1", "-", "f"}], ")"}], 
              RowBox[{
               SubscriptBox["\[PartialD]", "t"], 
               RowBox[{"c", "[", 
                RowBox[{"x", ",", "t"}], "]"}]}]}], "-", 
             RowBox[{"f", 
              RowBox[{
               SubscriptBox["\[PartialD]", "t"], 
               RowBox[{"s", "[", 
                RowBox[{"x", ",", "t"}], "]"}]}]}]}], "\[Equal]", 
            "\[IndentingNewLine]", 
            RowBox[{
             RowBox[{"DirichletCondition", "[", 
              RowBox[{
               RowBox[{
                RowBox[{"c", "[", 
                 RowBox[{"x", ",", "t"}], "]"}], "\[Equal]", 
                RowBox[{
                 RowBox[{"(", 
                  RowBox[{"1", "-", 
                   RowBox[{"Exp", "[", 
                    RowBox[{
                    RowBox[{"-", "1000"}], "t"}], "]"}]}], ")"}], "cin"}]}], 
               ",", 
               RowBox[{"x", "\[Equal]", "0"}]}], "]"}], "+", 
             RowBox[{"NeumannValue", "[", 
              RowBox[{"0", ",", 
               RowBox[{"x", "\[Equal]", "L"}]}], "]"}]}]}], ",", 
           "\[IndentingNewLine]", 
           RowBox[{
            RowBox[{"c", "[", 
             RowBox[{"x", ",", "0"}], "]"}], "\[Equal]", "c0"}], ",", 
           "\[IndentingNewLine]", "\[IndentingNewLine]", 
           RowBox[{
            RowBox[{"f", 
             RowBox[{
              SubscriptBox["\[PartialD]", "t"], 
              RowBox[{"s", "[", 
               RowBox[{"x", ",", "t"}], "]"}]}]}], "\[Equal]", 
            RowBox[{"k", 
             RowBox[{"(", 
              RowBox[{
               RowBox[{"c", "[", 
                RowBox[{"x", ",", "t"}], "]"}], "-", 
               RowBox[{"s", "[", 
                RowBox[{"x", ",", "t"}], "]"}]}], ")"}]}]}], ",", 
           "\[IndentingNewLine]", 
           RowBox[{
            RowBox[{"s", "[", 
             RowBox[{"x", ",", "0"}], "]"}], "\[Equal]", "s0"}]}], 
          "\[IndentingNewLine]", "}"}], ",", "\[IndentingNewLine]", 
         RowBox[{"{", 
          RowBox[{"c", ",", "s"}], "}"}], ",", 
         RowBox[{"{", 
          RowBox[{"x", ",", "0", ",", "L"}], "}"}], ",", 
         RowBox[{"{", 
          RowBox[{"t", ",", "0", ",", "21"}], "}"}], ",", 
         RowBox[{"Method", "\[Rule]", 
          RowBox[{"{", 
           RowBox[{"\"\<MethodOfLines\>\"", ",", 
            RowBox[{
            "\"\<SpatialDiscretization\>\"", "\[Rule]", 
             "\"\<FiniteElement\>\""}]}], "}"}]}]}], "]"}]}], ";", 
      "\[IndentingNewLine]", "\[IndentingNewLine]", 
      RowBox[{"Ints", "=", 
       RowBox[{"Last", "[", 
        RowBox[{
         RowBox[{"(", 
          RowBox[{"1", "/", "L"}], ")"}], 
         RowBox[{"Quiet", "@", 
          RowBox[{"NIntegrate", "[", 
           RowBox[{
            RowBox[{
             RowBox[{"s", "[", 
              RowBox[{"x", ",", "time"}], "]"}], "/.", "sol"}], ",", 
            RowBox[{"{", 
             RowBox[{"x", ",", 
              SuperscriptBox["10", 
               RowBox[{"-", "3"}]], ",", "L"}], "}"}]}], "]"}]}]}], "]"}]}], 
      ";", "\[IndentingNewLine]", " ", "\[IndentingNewLine]", 
      RowBox[{"ptime", "=", 
       RowBox[{"Plot", "[", 
        RowBox[{
         RowBox[{
          RowBox[{"c", "[", 
           RowBox[{"L", ",", "t"}], "]"}], "/.", "sol"}], ",", 
         RowBox[{"{", 
          RowBox[{"t", ",", "0", ",", "time"}], "}"}], ",", 
         RowBox[{"PlotStyle", "\[Rule]", 
          RowBox[{"{", 
           RowBox[{"Black", ",", "Thick"}], "}"}]}], ",", 
         RowBox[{"PlotRange", "\[Rule]", 
          RowBox[{"{", 
           RowBox[{
            RowBox[{"{", 
             RowBox[{"0", ",", "time"}], "}"}], ",", 
            RowBox[{"{", 
             RowBox[{
              RowBox[{"-", "0.001"}], ",", "1.001"}], "}"}]}], "}"}]}], ",", 
         RowBox[{"Frame", "\[Rule]", "True"}], ",", 
         RowBox[{"LabelStyle", "\[Rule]", 
          RowBox[{"{", 
           RowBox[{"17", ",", "Black"}], "}"}]}], ",", 
         RowBox[{"FrameLabel", "\[Rule]", 
          RowBox[{"{", 
           RowBox[{
           "\"\<time\>\"", ",", "\"\<fraction of solvent in effluent\>\""}], 
           "}"}]}], ",", 
         RowBox[{"ImageSize", "\[Rule]", 
          RowBox[{"1.1", 
           RowBox[{"{", 
            RowBox[{"550", ",", "350"}], "}"}]}]}]}], "]"}]}], ";", 
      "\[IndentingNewLine]", "\[IndentingNewLine]", 
      RowBox[{"pspace", "=", 
       RowBox[{"Plot", "[", "\[IndentingNewLine]", 
        RowBox[{
         RowBox[{
          RowBox[{"1", "-", 
           RowBox[{"s", "[", 
            RowBox[{"x", ",", "time"}], "]"}]}], "/.", "sol"}], ",", 
         RowBox[{"{", 
          RowBox[{"x", ",", "0", ",", "L"}], "}"}], ",", 
         RowBox[{"PlotStyle", "\[Rule]", "Black"}], ",", 
         RowBox[{"PlotRange", "\[Rule]", 
          RowBox[{"{", 
           RowBox[{
            RowBox[{"{", 
             RowBox[{"0", ",", " ", "1"}], "}"}], ",", 
            RowBox[{"{", 
             RowBox[{
              RowBox[{"-", ".1"}], ",", "1.001"}], "}"}]}], "}"}]}], ",", 
         RowBox[{"Frame", "\[Rule]", "True"}], ",", 
         RowBox[{"LabelStyle", "\[Rule]", 
          RowBox[{"{", 
           RowBox[{"17", ",", "Black"}], "}"}]}], ",", 
         RowBox[{"ImageSize", "\[Rule]", 
          RowBox[{"1.1", 
           RowBox[{"{", 
            RowBox[{"550", ",", "350"}], "}"}]}]}], ",", 
         RowBox[{"FrameLabel", "\[Rule]", 
          RowBox[{"{", 
           RowBox[{
            RowBox[{"Row", "[", 
             RowBox[{"{", 
              RowBox[{"\"\<distance = \>\"", ",", " ", 
               RowBox[{"Style", "[", 
                RowBox[{"\"\<x\>\"", ",", "Italic"}], "]"}], ",", "\"\</\>\"",
                ",", 
               RowBox[{"Style", "[", 
                RowBox[{"\"\<L\>\"", ",", "Italic"}], "]"}]}], "}"}], "]"}], 
            ",", "\"\<fraction of oil in dead space\>\""}], "}"}]}], ",", 
         RowBox[{"Filling", "\[Rule]", "Bottom"}], ",", "\[IndentingNewLine]", 
         RowBox[{"Epilog", "\[Rule]", 
          RowBox[{"Style", "[", 
           RowBox[{
            RowBox[{"Text", "[", 
             RowBox[{
              RowBox[{"Row", "[", 
               RowBox[{"{", 
                RowBox[{
                "\"\<fraction of total oil in dead space = \>\"", " ", ",", 
                 RowBox[{"Chop", "[", 
                  RowBox[{
                   RowBox[{"NumberForm", " ", "[", 
                    RowBox[{
                    RowBox[{"1", "-", "Ints"}], ",", "3"}], "]"}], ",", 
                   SuperscriptBox["10", 
                    RowBox[{"-", "3"}]]}], "]"}]}], "}"}], "]"}], ",", 
              RowBox[{"{", " ", 
               RowBox[{"0.5", ",", 
                RowBox[{"-", "0.05"}]}], "}"}]}], "]"}], ",", "Black", ",", 
            "17"}], "]"}]}]}], "]"}]}], ";", "\[IndentingNewLine]", 
      "\[IndentingNewLine]", 
      RowBox[{"Which", "[", 
       RowBox[{
        RowBox[{"g", "\[Equal]", "1"}], ",", "ptime", ",", 
        RowBox[{"g", "==", "2"}], ",", "pspace"}], "]"}]}]}], "]"}], ",", 
   "\[IndentingNewLine]", 
   RowBox[{"Grid", "[", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{
       RowBox[{"{", 
        RowBox[{"Control", "@", 
         RowBox[{"{", 
          RowBox[{
           RowBox[{"{", " ", 
            RowBox[{"g", ",", "1", ",", "\"\<\>\""}], "}"}], ",", 
           RowBox[{"{", 
            RowBox[{
             RowBox[{"1", "\[Rule]", "\"\<time plot\>\""}], ",", 
             RowBox[{"2", "\[Rule]", "\"\<space plot\>\""}]}], 
            "\[IndentingNewLine]", "}"}], ",", "PopupMenu"}], "}"}]}], "}"}], 
       ",", "\[IndentingNewLine]", 
       RowBox[{"{", 
        RowBox[{
         RowBox[{"Control", "@", 
          RowBox[{"{", 
           RowBox[{
            RowBox[{"{", 
             RowBox[{"time", ",", "6.0", ",", "\"\<time\>\""}], "}"}], ",", 
            "0.1", ",", "20", ",", "0.1", ",", 
            RowBox[{"Appearance", "\[Rule]", "\"\<Labeled\>\""}], ",", 
            RowBox[{"ImageSize", "\[Rule]", "Tiny"}]}], "}"}]}], ",", 
         "\[IndentingNewLine]", 
         RowBox[{"Control", "@", 
          RowBox[{"{", 
           RowBox[{
            RowBox[{"{", 
             RowBox[{"k", ",", "0.05", ",", "\"\<rate constant\>\""}], "}"}], 
            ",", "0.005", ",", "5.0", ",", "0.001", ",", 
            RowBox[{"Appearance", "\[Rule]", "\"\<Labeled\>\""}], ",", 
            RowBox[{"ImageSize", "\[Rule]", "Tiny"}]}], "}"}]}]}], "}"}], ",",
        "\[IndentingNewLine]", 
       RowBox[{"{", 
        RowBox[{
         RowBox[{"Control", "@", 
          RowBox[{"{", 
           RowBox[{
            RowBox[{"{", 
             RowBox[{
             "\[ScriptCapitalD]", ",", "0.05", ",", "\"\<diffusivity\>\""}], 
             "}"}], ",", "0.0005", ",", "0.1", ",", "0.0001", ",", 
            RowBox[{"ImageSize", "\[Rule]", "Tiny"}], ",", 
            RowBox[{"Appearance", "\[Rule]", "\"\<Labeled\>\""}]}], "}"}]}], 
         ",", "\[IndentingNewLine]", 
         RowBox[{"Control", "@", 
          RowBox[{"{", 
           RowBox[{
            RowBox[{"{", 
             RowBox[{"f", ",", "0.25", ",", "\"\<dead space fraction\>\""}], 
             "}"}], ",", "0.01", ",", "0.50", ",", "0.01", ",", 
            RowBox[{"ImageSize", "\[Rule]", "Tiny"}], ",", 
            RowBox[{"Appearance", "\[Rule]", "\"\<Labeled\>\""}]}], "}"}]}], 
         ",", 
         RowBox[{"Control", "@", 
          RowBox[{"{", 
           RowBox[{
            RowBox[{"{", 
             RowBox[{"u", ",", "0.5", ",", "\"\<interstitial velocity\>\""}], 
             "}"}], ",", "0.01", ",", "1.0", ",", "0.01", ",", 
            RowBox[{"ImageSize", "\[Rule]", "Tiny"}], ",", 
            RowBox[{"Appearance", "\[Rule]", "\"\<Labeled\>\""}]}], "}"}]}]}],
         "}"}]}], "}"}], ",", 
     RowBox[{"Alignment", "\[Rule]", "Left"}]}], "]"}], ",", 
   RowBox[{"ControlPlacement", "\[Rule]", "Top"}]}], "]"}]], "Input"]
}]"##;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);
    let cell = editors
      .iter()
      .map(|e| e.content.text())
      .find(|t| t.starts_with("Manipulate["))
      .expect("the Manipulate cell must load");
    // The operator reads as a derivative of the expression that follows it,
    // parenthesised so its coefficient stays a product.
    assert!(
      cell.contains("\u{1d49f}(D[c[x,t], x,x])-u(D[c[x,t], x])"),
      "partial derivatives must read as D[…]: {cell}"
    );
    assert!(!cell.contains('\u{2202}'), "no bare operator left: {cell}");

    let widget = instantiate_stored_manipulate(&cell, "")
      .expect("the Manipulate must instantiate");
    let names: Vec<&str> = widget
      .controls
      .iter()
      .map(|c| match c {
        manipulate::ControlState::Continuous { name, .. } => name.as_str(),
        manipulate::ControlState::Discrete { name, .. } => name.as_str(),
        other => panic!("unexpected control: {other:?}"),
      })
      .collect();
    assert_eq!(names, ["g", "time", "k", "\u{1d49f}", "f", "u"]);
  }
}
