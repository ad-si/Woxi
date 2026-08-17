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
  /// A `Button[…]` inside a Manipulate display was pressed: run its held
  /// action against the widget's bindings.
  ManipulateDisplayAction(usize, String),
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
              // Collect following Output/Print cells. Source notebooks
              // downloaded from the Wolfram Demonstrations Project
              // sometimes carry a stray *Input*-styled cell between the
              // real source and its cached Output — a leftover evaluation
              // snapshot whose content is itself a raw FrontEnd widget
              // dump (`DynamicModuleBox[…]`), never meaningful as code.
              // Treat such a cell the same as a stored Output so it is
              // absorbed here instead of rendered as a broken, empty
              // "code" cell of its own.
              let mut output = None;
              let mut stdout = None;
              let mut j = i + 1;
              while j < cells.len()
                && (matches!(
                  cells[j].style,
                  CellStyle::Output | CellStyle::Print
                ) || (cells[j].style == CellStyle::Input
                  && is_dynamic_box_dump(&cells[j].content)))
              {
                match cells[j].style {
                  CellStyle::Output | CellStyle::Input => {
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
          state.apply_tracking(ctrl_idx);
          if state.request_reeval(ctrl_idx) {
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
          state.apply_tracking(ctrl_idx);
          if state.request_reeval(ctrl_idx) {
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
          state.apply_tracking(ctrl_idx);
          if state.request_reeval(ctrl_idx) {
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
          state.apply_tracking(ctrl_idx);
          if state.request_reeval(ctrl_idx) {
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
          state.apply_tracking(ctrl_idx);
          if state.request_reeval(ctrl_idx) {
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
          state.apply_tracking(ctrl_idx);
          if state.request_reeval(ctrl_idx) {
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
          state.apply_tracking(ctrl_idx);
          if state.request_reeval(ctrl_idx) {
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

      Message::ManipulateDisplayAction(cell_idx, action) => {
        if let Some(editor) = self.cell_editors.get_mut(cell_idx)
          && let Some(state) = editor.manipulate_state.as_mut()
        {
          state.apply_button_action(&action);
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
/// plain `label` when there are no runs.
///
/// A control that gives no label of its own already carries its variable name
/// as the label (Wolfram captions `{k, 0, 1}` with "k"), so an empty label is
/// never a missing one: it is the explicit `""` a Demonstration writes to
/// suppress the caption, and stays blank.
fn manipulate_label_widget<'a>(
  runs: &[woxi::functions::graphics::LabelRun],
  label: &str,
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
    return text(label.to_string())
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
  let label = match ctrl {
    manipulate::ControlState::Continuous { label, .. }
    | manipulate::ControlState::Discrete { label, .. }
    | manipulate::ControlState::Slider2D { label, .. }
    | manipulate::ControlState::IntervalSlider { label, .. }
    | manipulate::ControlState::Trigger { label, .. }
    | manipulate::ControlState::Locator { label, .. } => label,
    // Heading/divider rows span the full row instead of sitting in the
    // label column, so they don't widen it; a button carries its label
    // inside the button itself.
    manipulate::ControlState::Button { .. }
    | manipulate::ControlState::Heading { .. }
    | manipulate::ControlState::Divider => return 0,
  };
  // An explicitly empty label (`{{fig, 1, ""}}`) claims no width — see
  // `manipulate_label_widget` for why an empty label is never a missing one.
  label.chars().count()
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
/// toggle buttons) as long as the whole row still fits
/// ([`SETTER_BAR_MAX_ROW_CHARS`]). Wolfram picks a SetterBar for up to five
/// choices even when each one is a phrase ("battle of the sexes",
/// "right triangle"); past that the labels have to be short.
const SETTER_BAR_MAX_CHOICES: usize = 5;

/// How wide, in characters summed over every label, a SetterBar of at most
/// [`SETTER_BAR_MAX_CHOICES`] choices may be. Past this the row of buttons no
/// longer fits the control panel and Wolfram falls back to a PopupMenu. The
/// sampled Demonstrations bracket it: five figure names totalling 55
/// characters are a bar, five error descriptions totalling 72 are a dropdown.
const SETTER_BAR_MAX_ROW_CHARS: usize = 64;

/// Maximum number of *compact* choices — every label at most
/// [`SETTER_BAR_COMPACT_LABEL_CHARS`] wide, i.e. numbers or single letters —
/// still rendered as a SetterBar. A run of short buttons stays readable well
/// past five, but not without bound.
const SETTER_BAR_MAX_COMPACT_CHOICES: usize = 10;

/// A choice label this short (in characters) keeps its button narrow enough
/// to sit in a long SetterBar.
const SETTER_BAR_COMPACT_LABEL_CHARS: usize = 3;

/// Whether a discrete control's choices render as a segmented SetterBar (a row
/// of toggle buttons) rather than a dropdown.
///
/// Wolfram's `Manipulate` picks between `SetterBar` and `PopupMenu` on its own
/// whenever the spec doesn't say (`ControlType -> …` forces the choice, and is
/// carried separately as `popup`). Sampling the Demonstrations that leave it
/// automatic, the split follows the choice count first and the width of the
/// row only within that count — never the total width alone:
///
/// | choices | labels                              | width | Wolfram    |
/// |---------|-------------------------------------|-------|------------|
/// | 4       | `4, 20, 100, 500`                   |     9 | SetterBar  |
/// | 4       | `prisoners dilemma`, …              |    57 | SetterBar  |
/// | 5       | `2, 3, 4, 5, 6`                     |     5 | SetterBar  |
/// | 5       | `quadrilateral` … `right triangle`  |    55 | SetterBar  |
/// | 8       | `-3` … `4`                          |    16 | SetterBar  |
/// | 5       | `u(y)`, `error in approximating …`  |    72 | PopupMenu  |
/// | 6       | `triangle` … `octagon`              |    44 | PopupMenu  |
/// | 17      | `Hue`, `BlueGreenYellow`, …         |   183 | PopupMenu  |
/// | 33      | `-3` … `29`                         |    75 | PopupMenu  |
///
/// So five phrases stay a bar while six single words become a dropdown, even
/// though the six are the narrower row — the count decides first, and short
/// labels buy a longer bar.
///
/// A choice whose label is a rendered icon counts as compact: it draws at a
/// fixed 24px, narrower than a three-character button, so it costs the row
/// nothing.
fn renders_as_setter_bar(
  value_labels: &[String],
  value_label_svgs: &[Option<svg::Handle>],
) -> bool {
  let is_icon = |i: usize| value_label_svgs.get(i).is_some_and(Option::is_some);
  let count = value_labels.len();
  if count <= SETTER_BAR_MAX_CHOICES {
    let row_chars: usize = value_labels
      .iter()
      .enumerate()
      .filter(|(i, _)| !is_icon(*i))
      .map(|(_, label)| label.chars().count())
      .sum();
    return row_chars <= SETTER_BAR_MAX_ROW_CHARS;
  }
  count <= SETTER_BAR_MAX_COMPACT_CHOICES
    && value_labels.iter().enumerate().all(|(i, label)| {
      is_icon(i) || label.chars().count() <= SETTER_BAR_COMPACT_LABEL_CHARS
    })
}

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
  // reads and a very long one can't swallow the slider. Only the rows
  // actually on screen count — a `PaneSelector` pane that is not showing
  // must not reserve room for its labels.
  let max_label_chars = state
    .controls
    .iter()
    .enumerate()
    .filter(|(i, _)| state.control_is_visible.get(*i).copied().unwrap_or(true))
    .map(|(_, c)| manipulate_label_char_count(c))
    .max()
    .unwrap_or(0);
  let label_col_width = (max_label_chars as f32 * 7.3 + 6.0).clamp(20.0, 220.0);
  let visible_controls: &[manipulate::ControlState] =
    if show_controls { &state.controls } else { &[] };
  for (ctrl_idx, ctrl) in visible_controls.iter().enumerate() {
    // A control belonging to a `PaneSelector` pane the selector is not
    // showing is left out of the panel entirely, the way Wolfram swaps one
    // pane's controls for another's.
    if !state
      .control_is_visible
      .get(ctrl_idx)
      .copied()
      .unwrap_or(true)
    {
      continue;
    }
    // A control whose `Enabled` condition currently evaluates to `False` is
    // greyed out and swallows interaction (see `Message::Noop`).
    let enabled = state
      .control_is_enabled
      .get(ctrl_idx)
      .copied()
      .unwrap_or(true);
    match ctrl {
      manipulate::ControlState::Continuous {
        name: _,
        label,
        label_runs,
        min,
        max,
        step,
        current,
        ..
      } => {
        let label_widget =
          manipulate_label_widget(label_runs, label, label_col_width, enabled);
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
        name: _,
        label,
        label_runs,
        values,
        value_labels,
        value_label_svgs,
        current_index,
        popup,
        setter_bar: force_setter_bar,
        slider: as_slider,
      } => {
        let label_widget =
          manipulate_label_widget(label_runs, label, label_col_width, enabled);
        // `ControlType -> Slider` over a discrete domain: a slider that
        // steps through the choices by index, which is how Wolfram draws a
        // twenty-entry colour-scheme picker. Dragging sends the choice at
        // the new index, so the update handler needs no special case.
        if *as_slider && value_labels.len() > 1 {
          let last = value_labels.len() - 1;
          let choices = value_labels.clone();
          let mut s =
            slider(0..=last as u32, *current_index as u32, move |i| {
              match choices.get(i as usize) {
                Some(choice) if enabled => Message::ManipulateDiscreteChanged(
                  cell_idx,
                  ctrl_idx,
                  choice.clone(),
                ),
                _ => Message::Noop,
              }
            })
            .step(1u32)
            .width(Fill);
          if !enabled {
            s = s.style(disabled_slider_style);
          }
          let control_row = row![label_widget, s].align_y(Center).spacing(8);
          controls_col = controls_col.push(control_row);
          continue;
        }
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
        // A compact enumerated set renders as a segmented SetterBar (a row of
        // adjacent toggle buttons with the active choice highlighted), matching
        // Wolfram's SetterBar; a wider one — see `renders_as_setter_bar` — or
        // an explicit `ControlType -> PopupMenu` renders a dropdown so the row
        // can't grow unbounded. `renders_as_setter_bar` only decides for a
        // spec that stays silent: an explicit `ControlType -> SetterBar` gets
        // its bar however long the choice list is. The button labels are the
        // display labels (rule right-hand sides); pressing one sends its
        // label, which the update handler maps back to an index. A disabled
        // control drops its press handlers so it can't be changed.
        let control: Element<Message> = if *force_setter_bar
          || (renders_as_setter_bar(value_labels, value_label_svgs) && !*popup)
        {
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
        name: _,
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
          manipulate_label_widget(&[], label, label_col_width, enabled);
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
        name: _,
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
          manipulate_label_widget(&[], label, label_col_width, enabled);
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
        name: _,
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
          manipulate_label_widget(&[], label, label_col_width, enabled);
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
        name: _,
        label,
        label_runs,
        current,
        ..
      } => {
        // A Trigger control: its own play/pause toggle plus a live readout
        // of the swept variable (Wolfram's TriggerButton/PauseButton pair).
        let label_widget =
          manipulate_label_widget(label_runs, label, label_col_width, enabled);
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
    DisplayNode::Button { label, action } => {
      // A caption button (`Button["→", n++]`): pressing it runs the held
      // action against the widget's live bindings and re-renders.
      button(render_display_node(cell_idx, label))
        .padding([2, 10])
        .on_press(Message::ManipulateDisplayAction(cell_idx, action.clone()))
        .into()
    }
    DisplayNode::Spacer { width } => {
      space::Space::new().width(*width as f32).into()
    }
    DisplayNode::Text { runs } => {
      let spans: Vec<iced::widget::text::Span<Message>> = runs
        .iter()
        .map(|run| {
          let mut font = Font::MONOSPACE;
          if run.italic {
            font.style = iced::font::Style::Italic;
          }
          if run.bold {
            font.weight = iced::font::Weight::Bold;
          }
          let mut span = iced::widget::span(run.text.clone()).font(font);
          if let Some((r, g, b)) = run.color {
            span = span.color(Color::from_rgb(r, g, b));
          }
          span
        })
        .collect();
      if spans.is_empty() {
        text("").size(12).into()
      } else {
        rich_text(spans).size(12).into()
      }
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
    assert!(state.request_reeval(0), "first change should arm the timer");
    assert!(!state.request_reeval(0), "second change must not re-arm");
    assert!(!state.request_reeval(0), "third change must not re-arm");

    // Timer fires: the pending changes render and the flag clears, so the
    // next change arms a fresh timer.
    state.run_scheduled_reeval();
    assert!(
      state.request_reeval(0),
      "a change after the timer fired should arm a new timer"
    );

    // A timer that fires with nothing new pending is a no-op that still
    // clears the flag (so a later change can re-arm).
    state.run_scheduled_reeval();
    state.run_scheduled_reeval();
    assert!(state.request_reeval(0), "flag must clear on an empty fire");
  }

  /// A discrete control's `TrackingFunction -> f` resets a companion
  /// control when the picker changes — the Demonstrations idiom of
  /// rewinding a step/time slider whenever the selected mode changes.
  /// Independently written, not copied from any specific Demonstration.
  #[test]
  fn manipulate_tracking_function_resets_companion_control() {
    let expr = woxi::interpret_to_expr(
      "Manipulate[If[mode == 1, step^2, step + 1], \
       {{mode, 1, \"\"}, {1 -> \"square\", 2 -> \"increment\"}, \
        TrackingFunction -> (mode = #; step = 0; &)}, \
       {{step, 3, \"step\"}, 0, 5, 1}]",
    )
    .unwrap();
    let mut state = manipulate::ManipulateState::from_expr(&expr).unwrap();
    let mode_idx = state
      .controls
      .iter()
      .position(|c| c.name() == "mode")
      .unwrap();
    let step_idx = state
      .controls
      .iter()
      .position(|c| c.name() == "step")
      .unwrap();

    // Move `step` away from its initial value so the reset is observable.
    if let manipulate::ControlState::Continuous { current, .. } =
      &mut state.controls[step_idx]
    {
      *current = 4.0;
    }

    // Pick the other choice in the `mode` popup.
    if let manipulate::ControlState::Discrete { current_index, .. } =
      &mut state.controls[mode_idx]
    {
      *current_index = 1;
    }
    state.apply_tracking(mode_idx);

    let manipulate::ControlState::Continuous { current, .. } =
      &state.controls[step_idx]
    else {
      panic!("expected step to remain a continuous slider");
    };
    assert_eq!(*current, 0.0, "TrackingFunction should reset step to 0");
  }

  /// A control panel written as a `Grid` — the Demonstrations layout for a
  /// widget whose rows are not all one control wide — builds exactly the
  /// control rows the grid names. Its `SpanFromLeft` cell markers used to
  /// survive as display elements, so the widget grew a row of literal
  /// `SpanFromLeft` text under the sliders.
  #[test]
  fn manipulate_grid_panel_has_no_span_marker_rows() {
    let expr = woxi::interpret_to_expr(
      "Manipulate[Plot[Sin[a x], {x, 0, b}], \
       Grid[{{Control[{{a, 1, \"rate\"}, 1, 5}], SpanFromLeft}, \
       {\"extent:\", Control[{{b, 6}, 1, 10}], SpanFromLeft}}]]",
    )
    .unwrap();
    let state = manipulate::ManipulateState::from_expr(&expr).unwrap();
    let names: Vec<&str> = state.controls.iter().map(|c| c.name()).collect();
    assert_eq!(names, vec!["a", "", "b"], "heading row binds no variable");
    assert!(
      state.displays.is_empty(),
      "span markers are layout, not displays: {:?}",
      state.displays
    );
  }

  /// A physics-style Demonstration shape: a single labeled slider driving a
  /// multi-statement body that stacks several precomputed `Show`/`Plot`
  /// panels into a `GraphicsRow`, with a large `Initialization :>` block
  /// defining helper graphics and pattern-matched functions before the
  /// widget ever renders. This mirrors the general construct category used
  /// by many single-parameter Wolfram Demonstrations Project notebooks
  /// (independently written, not copied from any specific one).
  #[test]
  fn manipulate_multi_statement_body_with_initialization_builds_one_slider() {
    let expr = woxi::interpret_to_expr(
      "Manipulate[\
       line5 = Graphics[{Thick, Blue, Line[{{0, k}, {0.7, k}}]}]; \
       p1 = Plot[barrier[k, x], {x, 0, 1}, PlotStyle -> {Thick, Blue}]; \
       GraphicsRow[{Show[pot, p1, AspectRatio -> 1], Show[eplot, line5]}, \
       ImageSize -> {575, 450}], \
       {{k, 0.02, \"wavenumber\"}, 0.02, 0.85}, \
       TrackedSymbols :> {k}, \
       Initialization :> (\
       v0 = 1; alpha = 10; \
       pot = Show[Graphics[{Opacity[0.15], Rectangle[{1, 0}, {2, v0}]}], Axes -> True]; \
       eplot = Plot[Sin[x], {x, 0, 1}]; \
       barrier[kk_, x_] := Sin[alpha Sqrt[kk] x];\
       )]",
    )
    .expect("Manipulate should parse and hold");
    let state = manipulate::ManipulateState::from_expr(&expr)
      .expect("a single labeled slider should build a ManipulateState");

    assert_eq!(state.controls.len(), 1, "exactly one slider control");
    let manipulate::ControlState::Continuous {
      name,
      label,
      min,
      max,
      current,
      ..
    } = &state.controls[0]
    else {
      panic!("expected a continuous slider for `k`");
    };
    assert_eq!(name, "k");
    assert_eq!(label, "wavenumber");
    assert_eq!(*min, 0.02);
    assert_eq!(*max, 0.85);
    assert_eq!(*current, 0.02);

    assert!(
      state.body.contains("GraphicsRow") && state.body.contains("Show"),
      "body should keep the multi-statement GraphicsRow/Show chain: {}",
      state.body
    );
    let init = state
      .initialization
      .as_deref()
      .expect("Initialization :> block should be captured");
    assert!(
      init.contains("barrier")
        && init.contains("pot")
        && init.contains("eplot"),
      "initialization should keep every helper definition: {init}"
    );
    assert!(state.error.is_none(), "unexpected error: {:?}", state.error);
  }

  /// A `PaneSelector` control panel — a Demonstration whose modes each need
  /// different controls, as the closest-packing one does — shows only the
  /// pane the selector is on. The controls of the other panes are built (so
  /// their variables stay bound for the body) but left off the panel, and a
  /// pane with no controls at all leaves no row behind.
  #[test]
  fn manipulate_pane_selector_shows_one_panel_at_a_time() {
    let expr = woxi::interpret_to_expr(
      "Manipulate[{q, a, b}, Control[{{q, 2}, {1 -> \"one\", 2 -> \"two\", \
       3 -> \"three\"}, Setter}], \
       PaneSelector[{1 -> Control[{{a, 5}, 0, 10}], \
       2 -> Column[{Control[{{a, 5}, 0, 10}], Control[{{b, 1}, 0, 2}]}], \
       3 -> \" \"}, q]]",
    )
    .unwrap();
    let mut state = manipulate::ManipulateState::from_expr(&expr).unwrap();
    let names: Vec<&str> = state.controls.iter().map(|c| c.name()).collect();
    assert_eq!(
      names,
      vec!["q", "a", "b"],
      "the placeholder pane must not add a row"
    );
    // The selector starts on pane 2, which shows both of its controls.
    assert_eq!(state.control_is_visible, vec![true, true, true]);

    // Switching the selector swaps the panel: pane 1 offers only `a`.
    let select = |state: &mut manipulate::ManipulateState, idx: usize| {
      if let manipulate::ControlState::Discrete { current_index, .. } =
        &mut state.controls[0]
      {
        *current_index = idx;
      }
      state.reevaluate();
    };
    select(&mut state, 0);
    assert_eq!(state.control_is_visible, vec![true, true, false]);
    // Pane 3 is the placeholder: the selector is the only row left.
    select(&mut state, 2);
    assert_eq!(state.control_is_visible, vec![true, false, false]);
  }

  /// A `PaneSelector` keyed by strings (`"a" -> …`, `"b" -> …`) rather than
  /// integers — the Demonstrations idiom for a mode picker whose Setter
  /// labels double as the pane keys. String comparison must be exact (a
  /// pane keyed `"a"` is not shown when the selector reads the *symbol* `a`
  /// or an unrelated string), so this exercises the same visibility gating
  /// as `manipulate_pane_selector_shows_one_panel_at_a_time` with a
  /// different key type rather than assuming it falls out for free.
  #[test]
  fn manipulate_pane_selector_switches_on_string_keys() {
    let expr = woxi::interpret_to_expr(
      "Manipulate[{mode, a, b}, \
       Control[{{mode, \"first\"}, {\"first\", \"second\"}, Setter}], \
       PaneSelector[{\"first\" -> Control[{{a, 5}, 0, 10}], \
       \"second\" -> Control[{{b, 1}, 0, 2}]}, mode]]",
    )
    .unwrap();
    let mut state = manipulate::ManipulateState::from_expr(&expr).unwrap();
    let names: Vec<&str> = state.controls.iter().map(|c| c.name()).collect();
    assert_eq!(names, vec!["mode", "a", "b"]);
    // The selector starts on "first", so only `a`'s row shows.
    assert_eq!(state.control_is_visible, vec![true, true, false]);

    // Switching the Setter to "second" swaps the panel to `b`.
    if let manipulate::ControlState::Discrete { current_index, .. } =
      &mut state.controls[0]
    {
      *current_index = 1;
    }
    state.reevaluate();
    assert_eq!(state.control_is_visible, vec![true, false, true]);
  }

  #[test]
  fn manipulate_untracked_control_does_not_reeval() {
    // `TrackedSymbols :> {b}`: moving `a` changes its value but must not
    // re-run the body — Wolfram leaves the rendering as it is until a
    // tracked variable changes. Descartes's Rule of Signs relies on this:
    // its degree setter would otherwise feed the body a polynomial degree
    // that the (still stale) coefficient list cannot be dotted with.
    let expr = woxi::interpret_to_expr(
      "Manipulate[a + b, {a, 0, 10}, {b, 0, 10}, TrackedSymbols :> {b}]",
    )
    .unwrap();
    let mut state = manipulate::ManipulateState::from_expr(&expr).unwrap();
    assert_eq!(state.controls[0].name(), "a");
    assert_eq!(state.controls[1].name(), "b");
    assert!(!state.request_reeval(0), "untracked `a` must not re-render");
    assert!(state.request_reeval(1), "tracked `b` must re-render");

    // Without the option every control is tracked, as before.
    let expr =
      woxi::interpret_to_expr("Manipulate[a + b, {a, 0, 10}, {b, 0, 10}]")
        .unwrap();
    let mut state = manipulate::ManipulateState::from_expr(&expr).unwrap();
    assert!(state.request_reeval(0), "default tracks every control");
  }

  #[test]
  fn manipulate_body_assignment_moves_the_written_controls_sliders() {
    // A "preset" idiom several Demonstrations use: picking one control
    // (`preset`) makes the body assign new values straight into other
    // controls' own variables (`{a, b} = presets[[preset]]`). Wolfram
    // treats that assignment as moving those controls' widgets, not just
    // feeding the current render — so after the assignment runs, `a` and
    // `b`'s sliders must show the assigned values, not their declared
    // defaults.
    let expr = woxi::interpret_to_expr(
      "Manipulate[If[preset == 1, {a, b} = {10, 20}]; a + b, \
       {{preset, 1, \"\"}, {0, 1}}, {{a, 3}, 0, 100}, {{b, 4}, 0, 100}]",
    )
    .unwrap();
    let state = manipulate::ManipulateState::from_expr(&expr).unwrap();
    assert_eq!(state.text_output.as_deref(), Some("30"));
    let manipulate::ControlState::Continuous { name, current, .. } =
      &state.controls[1]
    else {
      panic!("expected a continuous slider for `a`");
    };
    assert_eq!(name, "a");
    assert_eq!(*current, 10.0, "slider `a` must follow the assignment");
    let manipulate::ControlState::Continuous { name, current, .. } =
      &state.controls[2]
    else {
      panic!("expected a continuous slider for `b`");
    };
    assert_eq!(name, "b");
    assert_eq!(*current, 20.0, "slider `b` must follow the assignment");
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
  fn sole_finite_trigger_builds_a_dedicated_trigger_row() {
    // A Demonstrations "play once over a fixed duration" control: unlike
    // Kepler's Trigger (a *second* spec for a variable that already has a
    // plain slider, so it only steals the animation without a row of its
    // own), this Trigger is the *only* spec for its variable and its sweep
    // end is finite — `AppearanceElements` even asks for player buttons
    // instead of a thumb. That must still build a dedicated `Trigger` row
    // (play/pause + step buttons), not fall back to an ordinary slider.
    let expr = woxi::interpret_to_expr(
      "Manipulate[Rotate[Square[], angle Degree], \
       {{angle, 0, \"spin\"}, 0, 360, 1, Trigger, \
       AppearanceElements -> {\"PlayPauseButton\", \"ResetButton\"}}]",
    )
    .unwrap();
    let state = manipulate::ManipulateState::from_expr(&expr).unwrap();
    assert!(
      state.animated,
      "a Trigger control makes the widget animatable"
    );
    assert!(!state.playing, "a Trigger sits paused until pressed");
    assert_eq!(state.controls.len(), 1);
    match &state.controls[0] {
      manipulate::ControlState::Trigger {
        name,
        min,
        max,
        step,
        current,
        ..
      } => {
        assert_eq!(name, "angle");
        assert_eq!(*min, 0.0);
        assert_eq!(*max, 360.0);
        assert_eq!(*step, 1.0);
        assert_eq!(*current, 0.0);
      }
      other => panic!("expected a dedicated Trigger row, got {other:?}"),
    }
  }

  #[test]
  fn sole_finite_trigger_animation_wraps_at_its_end() {
    // The dedicated Trigger row must still wrap back to its start once the
    // sweep passes `max`, exactly like a plain finite slider would — a
    // finite Trigger is a bounded loop, not an indefinite run.
    let expr = woxi::interpret_to_expr(
      "Manipulate[disk = Disk[{0, 0}, r], \
       {{r, 3, \"grow\"}, 1, 3, 1, Trigger}]",
    )
    .unwrap();
    let mut state = manipulate::ManipulateState::from_expr(&expr).unwrap();
    let current = |s: &manipulate::ManipulateState| match &s.controls[0] {
      manipulate::ControlState::Trigger { current, .. } => *current,
      other => panic!("expected a Trigger control, got {other:?}"),
    };
    assert_eq!(current(&state), 3.0, "starts at its explicit initial value");
    state.advance_animation();
    assert_eq!(current(&state), 1.0, "steps past max wrap back to min");
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

  /// A phase-portrait Demonstration — sliders for a nonlinear oscillator's
  /// parameters driving a `Module` that solves several initial conditions
  /// with `NDSolve`, then overlays the vector field (`StreamPlot`) with the
  /// solution trajectories (a multi-curve `ParametricPlot[Evaluate[Table[…
  /// /. sol]]]`, each curve pairing a position with its `InterpolatingFunction`
  /// derivative). This pattern used to make every slider drag stall for
  /// seconds: `f'[t]` on an `NDSolve`-produced `InterpolatingFunction` built
  /// and simplified a symbolic Lagrange polynomial through the general
  /// evaluator on every sampled point instead of computing the local
  /// derivative directly in machine arithmetic. Regression coverage for
  /// that fix lives with `InterpolatingFunction` in
  /// `tests/interpreter_tests/calculus.rs`; this test guards the widget
  /// built from it keeps working end to end, including after a slider
  /// changes and the body re-runs.
  #[test]
  fn phase_portrait_manipulate_builds_and_redraws() {
    let expr = woxi::interpret_to_expr(
      "Manipulate[\
        Module[{esol}, \
          esol = Table[NDSolve[{x''[t] + delta x'[t] + alpha x[t] + beta x[t]^3 == 0, x[0] == x0, x'[0] == 0}, x, {t, 0, tmax}][[1]], {x0, -2, 2, 1}]; \
          Show[ \
            StreamPlot[{y, -delta y - alpha x - beta x^3}, {x, -2.5, 2.5}, {y, -2.5, 2.5}], \
            ParametricPlot[Evaluate[Table[{x[t], x'[t]} /. esol[[i]], {i, Length[esol]}]], {t, 0, tmax}] \
          ] \
        ], \
        {{delta, 0.2, \"damping\"}, 0, 1}, \
        {{alpha, 1, \"stiffness\"}, -2, 2}, \
        {{beta, 0.5, \"nonlinearity\"}, -2, 2}, \
        {{tmax, 20, \"time\"}, 5, 60} \
      ]",
    )
    .unwrap();
    let mut state = manipulate::ManipulateState::from_expr(&expr)
      .expect("Manipulate builds a widget");
    assert_eq!(
      state.controls.iter().map(|c| c.name()).collect::<Vec<_>>(),
      vec!["delta", "alpha", "beta", "tmax"]
    );
    assert!(
      state.error.is_none(),
      "manipulate body errored: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "expected a rendered graphic, got text_output={:?}",
      state.text_output
    );

    // Dragging the `alpha` slider (stiffness) must re-solve and re-render
    // without error, matching a user interacting with the widget.
    let alpha_idx = state
      .controls
      .iter()
      .position(|c| c.name() == "alpha")
      .unwrap();
    if let manipulate::ControlState::Continuous { current, .. } =
      &mut state.controls[alpha_idx]
    {
      *current = -1.5;
    }
    state.request_reeval(alpha_idx);
    state.run_scheduled_reeval();
    assert!(
      state.error.is_none(),
      "manipulate body errored after slider drag: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "expected a rendered graphic after slider drag"
    );
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

  /// A published Demonstration lays its panel out itself — the controls
  /// arrive wrapped in `Control[…]` inside a `Column[…]` alongside a
  /// `Button[…]` — and writes every non-ASCII character as a `\:HHHH`
  /// escape. Three things used to go wrong on such a notebook:
  ///
  /// - the escapes stayed literal in the held expression, so a glyph picker
  ///   offered `\:03b1` … `\:03bc` instead of α … μ;
  /// - the body `Style[Column[…], size, Hue[…]]` fell back to the plain
  ///   text echo of the column instead of drawing it at the asked-for size
  ///   and colour;
  /// - an explicit `ControlType -> SetterBar` was ignored once the choice
  ///   list grew past what the automatic split puts in a bar.
  #[test]
  fn demonstration_panel_with_escaped_glyphs_opens_live() {
    let nb_src = r##"Notebook[{
Cell[CellGroupData[{
Cell[BoxData["Manipulate[
 Style[Column[{glyph,
    If[names, Identity, Invisible]@Switch[glyph,
      \"\\:03b1\", \"alpha\", \"\\:03b2\", \"beta\", \"\\:03b3\", \"gamma\", glyph, \"\"],
    \"\"}, Alignment -> Center], size, Hue[tone]],
 Column[{
   Control[{{glyph, \"pick a letter\"},
     {\"\\:03b1\", \"\\:03b2\", \"\\:03b3\", \"\\:03b4\", \"\\:03b5\", \"\\:03b6\",
      \"\\:03b7\", \"\\:03b8\", \"\\:03b9\", \"\\:03ba\", \"\\:03bb\", \"\\:03bc\"},
     ControlType -> SetterBar}],
   Row[{Button[\"say it\", spoken = glyph]}],
   Control[{{size, 40}, 20, 80, Appearance -> \"Labeled\"}],
   Control[{tone, 0, 1}],
   Control[{{names, False}, {True, False}}]}],
 ContentSize -> {400, 200}, Alignment -> Center]"], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`glyph$$ = \"pick a letter\"}, \"…\"]"], "Output"]
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
    // The styled column is drawn, not echoed as `Column[{…}, Alignment -> …]`.
    assert!(
      widget.graphics_handle.is_some() && widget.text_output.is_none(),
      "the styled column must draw: {:?}",
      widget.text_output
    );
    match &widget.controls[..] {
      [
        manipulate::ControlState::Discrete {
          name: glyph,
          values,
          value_labels,
          setter_bar,
          popup,
          ..
        },
        manipulate::ControlState::Button { label, action, .. },
        manipulate::ControlState::Continuous {
          name: size,
          min: size_min,
          max: size_max,
          current: size_now,
          ..
        },
        manipulate::ControlState::Continuous { name: tone, .. },
        manipulate::ControlState::Discrete {
          name: names,
          values: name_values,
          ..
        },
      ] => {
        assert_eq!(glyph, "glyph");
        // The `\:HHHH` escapes expand, in the bound values and the labels.
        assert_eq!(value_labels[..3], ["α", "β", "γ"]);
        assert_eq!(values[0], "\"α\"");
        // Twelve choices are past the automatic bar/dropdown split, so only
        // the explicit `ControlType -> SetterBar` keeps this a bar.
        assert!(*setter_bar && !*popup);
        assert!(
          !renders_as_setter_bar(value_labels, &[]),
          "the choice list must be long enough that only the explicit \
           ControlType keeps it a bar"
        );
        assert_eq!(label, "say it");
        assert_eq!(action, "spoken = glyph");
        assert_eq!(
          (size.as_str(), *size_min, *size_max, *size_now),
          ("size", 20.0, 80.0, 40.0)
        );
        assert_eq!(tone, "tone");
        assert_eq!(names, "names");
        assert_eq!(name_values, &["True".to_string(), "False".to_string()]);
      }
      other => panic!("unexpected controls: {other:?}"),
    }
  }

  /// A Demonstration whose `Initialization :> (Get["HypothesisTesting`"];)`
  /// loads the legacy `Statistics`HypothesisTests`` compatibility package
  /// and whose body extracts a named property with
  /// `TwoSidedPValue /. MeanTest[data, mu, TwoSided -> True]` must open
  /// live: the context-qualified `HypothesisTesting`MeanTest` call has to
  /// evaluate to a proper rule list (not merely echo unevaluated) so the
  /// `ReplaceAll` actually extracts a p-value for the plot.
  #[test]
  fn demonstration_with_legacy_hypothesis_testing_mean_test_opens_live() {
    let nb_src = r#"Notebook[{
Cell[CellGroupData[{
Cell[BoxData["Manipulate[
 Module[{p, x, repCount},
  SeedRandom[seed];
  repCount = {50, 100}[[reps]];
  p = Quiet@Table[
    x = RandomReal[NormalDistribution[0, 1], n];
    HypothesisTesting`TwoSidedPValue /. HypothesisTesting`MeanTest[x, 0, HypothesisTesting`TwoSided -> True],
    {repCount}];
  If[graph == 1, Histogram[p, 10, \"Probability\"], ListPlot[Sort[p]]]],
 {{n, 20, \"sample size\"}, 10, 50, 10, Appearance -> \"Labeled\"},
 {{seed, 1, \"seed\"}, 1, 100, 1, Appearance -> \"Labeled\"},
 {{reps, 1, \"reps\"}, {1 -> \"50\", 2 -> \"100\"}},
 {{graph, 1, \"graph\"}, {1 -> \"histogram\", 2 -> \"scatter\"}},
 TrackedSymbols :> {n, seed, reps, graph},
 SynchronousUpdating -> False,
 Initialization :> (Get[\"HypothesisTesting`\"];)]"], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`n$$ = 20}, \"…\"]"], "Output"]
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
      widget.graphics_handle.is_some() && widget.text_output.is_none(),
      "the histogram of extracted p-values must draw, not echo the \
       unevaluated MeanTest/ReplaceAll: {:?}",
      widget.text_output
    );
    assert_eq!(widget.controls.len(), 4);
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

  /// Wolfram Demonstrations Project "source" notebooks (the authoring
  /// copy downloaded from demonstrations.wolfram.com, as opposed to the
  /// deployed cloud embed) sometimes carry a stray *Input*-styled cell
  /// between the real `Manipulate[…]` source and its cached Output — a
  /// leftover evaluation snapshot whose content is itself a raw
  /// FrontEnd widget dump (`DynamicModuleBox[…]`), never meaningful as
  /// code. That extra cell must be absorbed rather than rendered as its
  /// own broken, empty "code" cell, and must not be mistaken for the
  /// real source when the live widget is instantiated.
  #[test]
  fn stray_input_styled_widget_dump_between_source_and_output_is_absorbed() {
    let nb_src = r#"Notebook[{
Cell[CellGroupData[{
Cell[BoxData["Manipulate[x^2, {x, 1, 10}]"], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`x$$ = 5}, \"…\"]"], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`x$$ = 1}, \"…\"]"], "Output"]
}, Open]]
}]"#;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);
    // The stray widget-dump Input cell is absorbed into the real source
    // cell's editor, not rendered as its own (broken, empty) entry.
    assert_eq!(editors.len(), 1);
    let widget = editors[0]
      .manipulate_state
      .as_ref()
      .expect("the stored Manipulate must instantiate on load");
    assert!(
      widget.error.is_none(),
      "body must evaluate cleanly: {:?}",
      widget.error
    );
  }

  /// A stored Manipulate whose body composes `Tooltip`-wrapped series
  /// (one built with `Table`, one with `NestList`/`Partition`) into a
  /// `ListLinePlot` with a `PlotLabel` assembled from `ToString`/
  /// `NumberForm` string concatenation, driven by three
  /// `Appearance -> "Labeled"` sliders — the shape of a typical Wolfram
  /// Demonstrations finance/growth calculator. This must open live with
  /// all three sliders recognized and the plot drawn.
  #[test]
  fn demonstration_labeled_sliders_drive_multi_series_line_plot() {
    let nb_src = r##"Notebook[{
Cell[BoxData["growthSeries[base_, pct_, periods_] := N[Table[base (1 + pct/100)^k, {k, 1, periods}]]"], "Input"],
Cell[CellGroupData[{
Cell[BoxData["Manipulate[
 ListLinePlot[
   {Tooltip[growthSeries[base, pct, periods], \"compounded\"],
    Tooltip[Last /@ Partition[NestList[# + base pct/100 &, base, periods], 3], \"linear\"]},
   PlotLabel -> \"base $\" <> ToString[NumberForm[base, {6, 0}]] <> \" at \" <>
     ToString[NumberForm[pct, {4, 2}]] <> \"%\",
   Frame -> {True, True, False, False},
   FrameLabel -> {\"period\", \"value\"},
   ImageSize -> {400, 300},
   AxesOrigin -> {0, 0}],
 {{base, 1000, \"base amount\"}, 200, 5000, 100, Appearance -> \"Labeled\"},
 {{pct, 5, \"growth percent\"}, 1, 20, 0.5, Appearance -> \"Labeled\"},
 {{periods, 10, \"periods\"}, 2, 30, 1, Appearance -> \"Labeled\"},
 SaveDefinitions -> True]"], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`base$$ = 1000, $CellContext`pct$$ = 5, $CellContext`periods$$ = 10}, \"…\"]"], "Output"]
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
      "the multi-series ListLinePlot must draw"
    );
    match &widget.controls[..] {
      [
        manipulate::ControlState::Continuous {
          name: base,
          label: base_label,
          min: base_min,
          max: base_max,
          current: base_now,
          ..
        },
        manipulate::ControlState::Continuous {
          name: pct,
          label: pct_label,
          min: pct_min,
          max: pct_max,
          current: pct_now,
          ..
        },
        manipulate::ControlState::Continuous {
          name: periods,
          label: periods_label,
          min: periods_min,
          max: periods_max,
          current: periods_now,
          ..
        },
      ] => {
        assert_eq!(
          (
            base.as_str(),
            base_label.as_str(),
            *base_min,
            *base_max,
            *base_now
          ),
          ("base", "base amount", 200.0, 5000.0, 1000.0)
        );
        assert_eq!(
          (
            pct.as_str(),
            pct_label.as_str(),
            *pct_min,
            *pct_max,
            *pct_now
          ),
          ("pct", "growth percent", 1.0, 20.0, 5.0)
        );
        assert_eq!(
          (
            periods.as_str(),
            periods_label.as_str(),
            *periods_min,
            *periods_max,
            *periods_now
          ),
          ("periods", "periods", 2.0, 30.0, 10.0)
        );
      }
      other => panic!("unexpected controls: {other:?}"),
    }
  }

  #[test]
  fn demonstration_compatibility_checkboxes_render_as_a_card() {
    // The metadata cells at the end of every Demonstration submission
    // notebook pair a checkbox with a caption in a `RowDefault` row. They
    // must open as a rendered checkbox card rather than as the raw nested
    // braces the box extraction leaves behind.
    let nb_src = r#"Notebook[{
Cell[BoxData[TagBox[GridBox[{{TagBox[GridBox[{{TemplateBox[{CheckboxBox[True, {False, False}], "\" \"", StyleBox["\"Supported in cloud\"", FontSize -> 12]}, "RowDefault"]}}], "Column"]}}], "Grid"]], "Output"]
}]"#;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let cell = match &nb.cells[0] {
      CellEntry::Single(c) => c.clone(),
      _ => panic!("expected a single cell"),
    };
    assert_eq!(cell.content, "{{{{\u{2611} Supported in cloud}}}}");
    let editor = stored_output_editor(&cell)
      .expect("a checkbox grid must render as a stored graphic");
    assert!(editor.stored_graphic);
    let svg = editor.graphics_svg.expect("the card is an SVG");
    assert!(
      svg.contains("Supported in cloud"),
      "the caption must survive into the card: {svg}"
    );
    assert!(
      svg.contains("[x]"),
      "the checkbox must show as ticked: {svg}"
    );
  }

  #[test]
  fn hinged_dissection_notebook_opens_with_its_widget() {
    // End-to-end regression for the shape of Demonstration that animates a
    // hinged dissection: an initialization cell defines the pieces and a
    // `helper[k_, opts___]` wrapper that forwards its options to
    // `Graphics`, and the Manipulate swings each piece with `Rotate` about
    // its own hinge, driven by a labelled slider.
    let nb_src = r#"Notebook[{
Cell[BoxData["squareA = Polygon[{{0, 0}, {1, 0}, {1, 1}, {0, 1}}];\nsquareB = Polygon[{{1, 0}, {2, 0}, {2, 1}, {1, 1}}];\nswing[k_, opts___] := Graphics[{RGBColor[1, 0, 0], squareA, RGBColor[0, 0, 1], Rotate[squareB, k Pi/2, {1, 0}]}, opts]"], "Input"],
Cell[CellGroupData[{
Cell[BoxData["Manipulate[swing[k, PlotRange -> {{-1.5, 2.5}, {-1.5, 2.5}}, ImageSize -> {300, 300}], {{k, 0, \"swing\"}, 0, 1}, SaveDefinitions -> True]"], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`k$$ = 0.}, \"…\"]"], "Output"]
}, Open]]
}]"#;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);
    let mut widget = editors
      .into_iter()
      .find_map(|e| e.manipulate_state)
      .expect("the stored Manipulate must instantiate on load");
    assert!(
      widget.error.is_none(),
      "body must evaluate cleanly: {:?}",
      widget.error
    );

    // `SaveDefinitions -> True` is a Manipulate option, not a control.
    match &widget.controls[..] {
      [
        manipulate::ControlState::Continuous {
          name,
          label,
          min,
          max,
          current,
          ..
        },
      ] => {
        assert_eq!(name, "k");
        assert_eq!(label, "swing");
        assert_eq!((*min, *max, *current), (0.0, 1.0, 0.0));
      }
      other => panic!("expected one labelled slider, got {other:?}"),
    }

    // The iced handle doesn't expose its bytes, so re-render the body
    // through the widget's own bindings to inspect the SVG.
    let render = |w: &manipulate::ManipulateState| {
      let bindings: Vec<(String, String)> = w
        .controls
        .iter()
        .filter(|c| c.binds_variable())
        .map(|c| (c.name().to_string(), c.current_code()))
        .collect();
      let code = match w.initialization.as_deref() {
        Some(init) => format!("{init}; {}", w.body),
        None => w.body.clone(),
      };
      woxi::with_scoped_globals(&bindings, || {
        woxi::interpret_with_stdout(&code)
      })
      .expect("body evaluates")
      .graphics
      .expect("the pieces must render")
    };

    // The helper's `opts___` reaches Graphics, so ImageSize is honoured.
    assert!(widget.graphics_handle.is_some());
    let unswung = render(&widget);
    assert!(
      unswung.contains("width=\"300\"") && unswung.contains("height=\"300\""),
      "ImageSize must pass through opts___: {unswung}"
    );

    // Swinging the hinge through a quarter turn moves the blue square: its
    // far edge rotates from x = 2 up to y = 1 above the hinge at {1, 0}.
    match &mut widget.controls[0] {
      manipulate::ControlState::Continuous { current, .. } => *current = 1.0,
      other => panic!("expected continuous control, got {other:?}"),
    }
    widget.reevaluate();
    assert!(widget.error.is_none());
    assert!(widget.graphics_handle.is_some());
    assert_ne!(
      unswung,
      render(&widget),
      "Rotate about the hinge must change the rendered geometry"
    );
  }

  /// End-to-end regression for the shape of dissection Demonstration that
  /// reassembles one figure into several others: a `Switch` over a figure
  /// picker chooses which rearrangement a single "move" slider drives. The
  /// picker names its choices but suppresses its own caption with `""`, and
  /// its five phrase-long labels sit in a SetterBar.
  #[test]
  fn dissection_figure_picker_notebook_opens_with_its_widget() {
    let nb_src = r#"Notebook[{
Cell[BoxData["lower = {RGBColor[1, 0, 0], Polygon[{{0, 0}, {1, 0}, {1, 1}}]};\nupper = {RGBColor[0, 0, 1], Polygon[{{0, 0}, {1, 1}, {0, 1}}]};\ncorners = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};"], "Input"],
Cell[CellGroupData[{
Cell[BoxData["Manipulate[Graphics[{EdgeForm[Black], Switch[fig, 1, {lower, upper}, 2, {lower, Translate[upper, k {1, 0}]}, 3, {lower, Rotate[upper, -k Pi/2, corners[[3]]]}, 4, {Translate[lower, k (corners[[4]] - corners[[2]])], upper}, 5, {lower, Rotate[{Translate[upper, k {1, 0}]}, k Pi, corners[[2]]]}]}, PlotRange -> {{-1.5, 2.5}, {-1.5, 2.5}}, ImageSize -> {300, 300}], {{fig, 1, \"\"}, {1 -> \"quadrilateral\", 2 -> \"Greek cross\", 3 -> \"rhomboid\", 4 -> \"rectangle\", 5 -> \"right triangle\"}}, {{k, 0, \"move\"}, 0, 1}, ControlPlacement -> Top, SaveDefinitions -> True]"], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`fig$$ = 1, $CellContext`k$$ = 0}, DynamicBox[…], Initialization:>({$CellContext`lower = {\n RGBColor[1, 0, 0], \n Polygon[{{0, 0}, {1, 0}, {1, 1}}]}, $CellContext`upper = {\n RGBColor[0, 0, 1], \n Polygon[{{0, 0}, {1, 1}, {0, 1}}]}, $CellContext`corners = {{0, 0}, {1, 0}, {1, 1}, {0, 1}}}; Typeset`initDone$$ = True)]"], "Output"]
}, Open]]
}]"#;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);
    let mut widget = editors
      .into_iter()
      .find_map(|e| e.manipulate_state)
      .expect("the stored Manipulate must instantiate on load");
    assert!(
      widget.error.is_none(),
      "body must evaluate cleanly: {:?}",
      widget.error
    );

    // `ControlPlacement` and `SaveDefinitions` are Manipulate options, not
    // controls, so the panel is the picker followed by the slider.
    match &widget.controls[..] {
      [
        manipulate::ControlState::Discrete {
          name,
          label,
          values,
          value_labels,
          current_index,
          popup,
          ..
        },
        manipulate::ControlState::Continuous {
          name: k_name,
          label: k_label,
          min,
          max,
          current,
          ..
        },
      ] => {
        assert_eq!(name, "fig");
        // `{{fig, 1, ""}}` suppresses the caption: an explicitly empty label
        // stays empty instead of falling back to the variable name.
        assert_eq!(label, "");
        assert_eq!(values, &["1", "2", "3", "4", "5"]);
        assert_eq!(
          value_labels,
          &[
            "quadrilateral",
            "Greek cross",
            "rhomboid",
            "rectangle",
            "right triangle",
          ]
        );
        assert_eq!(*current_index, 0);
        assert!(
          !*popup && renders_as_setter_bar(value_labels, &[]),
          "Wolfram shows the five figure names as a row of buttons"
        );
        assert_eq!((k_name.as_str(), k_label.as_str()), ("k", "move"));
        assert_eq!((*min, *max, *current), (0.0, 1.0, 0.0));
      }
      other => panic!("expected a figure picker and a slider, got {other:?}"),
    }

    // The suppressed caption claims no room in the shared label column, so
    // the setter bar starts where the slider's "move" caption ends.
    assert_eq!(manipulate_label_char_count(&widget.controls[0]), 0);
    assert_eq!(manipulate_label_char_count(&widget.controls[1]), 4);

    // The iced handle doesn't expose its bytes, so re-render the body
    // through the widget's own bindings to inspect the SVG.
    let render = |w: &manipulate::ManipulateState| {
      let bindings: Vec<(String, String)> = w
        .controls
        .iter()
        .filter(|c| c.binds_variable())
        .map(|c| (c.name().to_string(), c.current_code()))
        .collect();
      let code = match w.initialization.as_deref() {
        Some(init) => format!("{init}; {}", w.body),
        None => w.body.clone(),
      };
      woxi::with_scoped_globals(&bindings, || {
        woxi::interpret_with_stdout(&code)
      })
      .expect("body evaluates")
      .graphics
      .expect("the pieces must render")
    };

    assert!(widget.graphics_handle.is_some());
    let assembled = render(&widget);
    assert!(
      assembled.contains("width=\"300\"")
        && assembled.contains("height=\"300\""),
      "ImageSize must reach Graphics: {assembled}"
    );

    // Every branch starts from the same assembled figure, so the picker
    // alone changes nothing while the slider sits at zero.
    for index in 1..5 {
      match &mut widget.controls[0] {
        manipulate::ControlState::Discrete { current_index, .. } => {
          *current_index = index
        }
        other => panic!("expected the figure picker, got {other:?}"),
      }
      widget.reevaluate();
      assert!(widget.error.is_none(), "figure {} errored", index + 1);
      assert_eq!(
        assembled,
        render(&widget),
        "at move = 0 every figure must show the undissected shape"
      );
    }

    // Sliding "move" to the end takes each branch apart differently, so no
    // two of the five rearrangements render alike.
    let mut moved: Vec<String> = Vec::new();
    for index in 0..5 {
      match &mut widget.controls[0] {
        manipulate::ControlState::Discrete { current_index, .. } => {
          *current_index = index
        }
        other => panic!("expected the figure picker, got {other:?}"),
      }
      match &mut widget.controls[1] {
        manipulate::ControlState::Continuous { current, .. } => *current = 1.0,
        other => panic!("expected the move slider, got {other:?}"),
      }
      widget.reevaluate();
      assert!(widget.error.is_none(), "figure {} errored", index + 1);
      assert!(widget.graphics_handle.is_some());
      moved.push(render(&widget));
    }
    // Branch 1 leaves the pieces alone whatever `move` reads; the other four
    // each move them somewhere else.
    assert_eq!(moved[0], assembled, "the first figure never comes apart");
    for i in 1..5 {
      assert_ne!(moved[i], assembled, "figure {} must move its pieces", i + 1);
      for j in (i + 1)..5 {
        assert_ne!(
          moved[i],
          moved[j],
          "figures {} and {} coincide",
          i + 1,
          j + 1
        );
      }
    }
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
  fn grid_laid_out_controls_take_the_manipulate_level_control_type() {
    // End-to-end for the Demonstrations control-panel shape: the controls sit
    // in a `Grid[…]` and the panel picks their type once, for all of them.
    // Both would otherwise fall to their automatic type — a SetterBar, which
    // for choices this long is unreadably wide — and the body's range guard
    // has to keep the tabulated curve inside its own data range.
    let nb_src = r##"Notebook[{
Cell[BoxData[
 RowBox[{
  RowBox[{"curve", "=", RowBox[{"Interpolation", "[",
   RowBox[{
    RowBox[{"{", RowBox[{
      RowBox[{"{", RowBox[{"0", ",", "0"}], "}"}], ",",
      RowBox[{"{", RowBox[{"1", ",", "2"}], "}"}], ",",
      RowBox[{"{", RowBox[{"2", ",", "0"}], "}"}]}], "}"}], ",",
    RowBox[{"InterpolationOrder", "\[Rule]", "1"}]}], "]"}]}], ";",
  RowBox[{RowBox[{"guarded", "[", "w_", "]"}], ":=",
   RowBox[{"Piecewise", "[",
    RowBox[{
     RowBox[{"{", RowBox[{"{",
       RowBox[{RowBox[{"curve", "[", "w", "]"}], ",",
        RowBox[{"0", "\[LessEqual]", "w", "\[LessEqual]", "2"}]}], "}"}], "}"}],
     ",", "0"}], "]"}]}]}]], "Input"],
Cell[BoxData[
 RowBox[{"Manipulate", "[",
  RowBox[{
   RowBox[{"Plot", "[",
    RowBox[{
     RowBox[{"scale", " ",
      RowBox[{"guarded", "[", RowBox[{"t", "+", "shift"}], "]"}]}], ",",
     RowBox[{"{", RowBox[{"t", ",", "0", ",", "2"}], "}"}]}], "]"}], ",",
   RowBox[{"Grid", "[",
    RowBox[{"{", RowBox[{"{",
      RowBox[{
       RowBox[{"Control", "[",
        RowBox[{"{",
         RowBox[{
          RowBox[{"{", RowBox[{"scale", ",", "1", ",", "\"\<vertical scale\>\""}], "}"}],
          ",", RowBox[{"{", RowBox[{"1", ",", "2", ",", "3"}], "}"}]}], "}"}], "]"}],
       ",",
       RowBox[{"Control", "[",
        RowBox[{"{",
         RowBox[{
          RowBox[{"{", RowBox[{"shift", ",", "0", ",", "\"\<horizontal shift\>\""}], "}"}],
          ",", RowBox[{"Range", "[", RowBox[{"0", ",", "1", ",", "0.5"}], "]"}]}], "}"}], "]"}]}],
      "}"}], "}"}], "]"}], ",",
   RowBox[{"ControlType", "\[Rule]", "PopupMenu"}], ",",
   RowBox[{"SaveDefinitions", "\[Rule]", "True"}]}], "]"}]], "Input"]
}]"##;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);
    let definitions = editors[0].content.text();
    woxi::interpret(&definitions).expect("the definitions must evaluate");

    let widget = instantiate_stored_manipulate(&editors[1].content.text(), "")
      .expect("the Manipulate must build a widget");
    assert!(
      widget.error.is_none(),
      "body must evaluate cleanly: {:?}",
      widget.error
    );
    assert!(
      widget.graphics_handle.is_some(),
      "the guarded plot must render"
    );
    match &widget.controls[..] {
      [
        manipulate::ControlState::Discrete {
          name: first,
          popup: first_popup,
          ..
        },
        manipulate::ControlState::Discrete {
          name: second,
          popup: second_popup,
          ..
        },
      ] => {
        assert_eq!(first, "scale");
        assert_eq!(second, "shift");
        assert!(
          *first_popup && *second_popup,
          "the panel's ControlType must reach both controls"
        );
      }
      other => panic!("expected two discrete controls, got {other:?}"),
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
  fn locator_manipulate_lays_out_a_paned_readout_over_a_picture() {
    // End-to-end regression for the plane-geometry Demonstrations that stack
    // a numeric readout over the drawing: the body is a one-column `Grid`
    // whose first cell is a `Pane[Style[Grid[…], …], ImageSize -> …]` and
    // whose second is the `Graphics`. The pane's cell used to print the
    // `Pane[…]` call as source text, which both hid the readout and stretched
    // the widget to the width of that source.
    let code = "Manipulate[\
      Grid[{{Pane[Style[Grid[{{\"d\", \"\\[TildeTilde]\", \
          NumberForm[EuclideanDistance[pt1, pt2], {5, 2}]}}, \
          Dividers -> {{True, False, False, True}, All}, \
          Spacings -> {{1 -> 1.3, 2 -> 0.3, 3 -> 0.3, 4 -> 1.3}, Automatic}], \
          12, \"Label\"], \
        ImageSize -> {200, 40}, Alignment -> {Center, Center}]}, \
       {Graphics[{Line[{pt1, pt2}]}, ImageSize -> {200, 160}, \
          PlotRange -> {{-1, 1}, {-1, 1}}]}}, \
       ItemSize -> {Automatic, {2, 5}}, Alignment -> {Center, Top}], \
      {{pt1, {-0.5, -0.5}}, {-1, -1}, {1, 1}, Locator}, \
      {{pt2, {0.5, 0.5}}, {-1, -1}, {1, 1}, Locator}]";
    let state = instantiate_stored_manipulate(code, "")
      .expect("the paned-readout Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(state.graphics_handle.is_some(), "the body must render");
    assert_eq!(state.controls.len(), 2);
    assert!(
      state
        .controls
        .iter()
        .all(|c| matches!(c, manipulate::ControlState::Slider2D { .. }))
    );

    // Re-render through the widget's own bindings to inspect the SVG the
    // handle was built from: the readout shows its numbers, ruled into a
    // closed box, and the picture is drawn below it.
    let bindings: Vec<(String, String)> = state
      .controls
      .iter()
      .filter(|c| c.binds_variable())
      .map(|c| (c.name().to_string(), c.current_code()))
      .collect();
    let svg = woxi::with_scoped_globals(&bindings, || {
      woxi::interpret_with_stdout(&state.body)
    })
    .expect("the body must evaluate")
    .graphics
    .expect("the body must render a graphic");
    assert!(!svg.contains("Pane["), "the wrapper must not print: {svg}");
    assert!(svg.contains(">1.41<"), "the readout value: {svg}");
    assert!(svg.contains("<polyline"), "the drawn segment: {svg}");
    // `Dividers -> {…, All}` closes the readout top and bottom.
    let horizontal = svg
      .lines()
      .filter(|l| l.starts_with("<line "))
      .filter(|l| {
        let v = |n: &str| {
          l.split(&format!("{n}=\""))
            .nth(1)
            .map(|t| t.split('"').next().unwrap_or_default().to_string())
        };
        v("y1").is_some() && v("y1") == v("y2")
      })
      .count();
    assert_eq!(horizontal, 2, "readout ruled top and bottom: {svg}");
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

  /// The shape a whole family of solid-geometry Demonstrations is built
  /// on: sliders stacked in a `Column` of `Control@…` entries beside the
  /// picture, checkboxes toggling extra surfaces, and a `Graphics3D` whose
  /// contents are unbounded — a line drawn to the edge of the box and a
  /// plane filling its cross section of it — with a caption that reads off
  /// the numbers at a fixed width.
  ///
  /// Four things had to be fixed together for it to come out: the
  /// unbounded primitives drew nothing at all in 3D, `Sphere` with a list
  /// of centres collapsed to one sphere at the origin, `((a ⨯ b) ⨯ a)`
  /// parsed as the three-argument `Cross[a, b, a]` (an error for 3D
  /// vectors), and `NumberForm` left the numbers of a symbolic expression
  /// alone while a negative real coefficient printed as `+ -2. y`.
  #[test]
  fn skew_line_manipulate_draws_its_unbounded_geometry() {
    let code = "Manipulate[\
      Module[{u, w, n}, \
        u = {u1, u2, 1}; w = {1, w2, w3}; \
        n = ((u \\[Cross] w) \\[Cross] u); \
        Column[{\
          Text@Row[{\"n = \", NumberForm[N[n . {x, y, z}], {4, 3}]}], \
          Graphics3D[{Thick, Blue, InfiniteLine[{{0, 0, 0}, u}], \
            Black, Sphere[{{0, 0, 0}, u}, 0.2], \
            Opacity[0.4], \
            If[plane, {Green, InfinitePlane[{0, 0, 0}, {u, n}]}]}, \
            PlotRange -> 4, ImageSize -> {320, 300}]}, \
          Alignment -> Center]], \
      Column[{\
        Control@{{u1, 2, Subscript[Style[\"u\", Italic], 1]}, -4, 4, 0.01, \
          Appearance -> \"Labeled\"}, \
        Control@{{u2, 1, Subscript[Style[\"u\", Italic], 2]}, -4, 4, 0.01, \
          Appearance -> \"Labeled\"}, \
        \" \", \
        Control@{{w2, 3, Subscript[Style[\"w\", Italic], 2]}, -4, 4, 0.01, \
          Appearance -> \"Labeled\"}, \
        Control@{{w3, -1, Subscript[Style[\"w\", Italic], 3]}, -4, 4, 0.01, \
          Appearance -> \"Labeled\"}, \
        \" \", \
        Control@{{plane, False, \"show the plane\"}, {False, True}}}], \
      ControlPlacement -> Left]";
    let state = instantiate_stored_manipulate(code, "")
      .expect("the Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(state.graphics_handle.is_some(), "the scene must render");

    // Four sliders and the checkbox survive the `Column` they are laid out
    // in; the two `" "` spacers become headings.
    let kinds: Vec<&str> = state
      .controls
      .iter()
      .map(|c| match c {
        manipulate::ControlState::Continuous { .. } => "continuous",
        manipulate::ControlState::Discrete { .. } => "discrete",
        manipulate::ControlState::Heading { .. } => "heading",
        other => panic!("unexpected control {other:?}"),
      })
      .collect();
    assert_eq!(
      kinds,
      vec![
        "continuous",
        "continuous",
        "heading",
        "continuous",
        "continuous",
        "heading",
        "discrete"
      ]
    );

    let render = |plane: &str| -> String {
      woxi::interpret_with_stdout(&format!(
        "u1 = 2; u2 = 1; w2 = 3; w3 = -1; plane = {plane};\n{}",
        state.body
      ))
      .expect("the body must render")
      .graphics
      .expect("the body must produce a graphic")
    };

    // `n` is `(u ⨯ w) ⨯ u` = {-2, 14, -10} for those settings, so the
    // caption reads the plane's equation off at three decimals — with the
    // negative coefficients written as subtractions.
    let closed = render("False");
    assert!(
      closed.contains("-2.000 x + 14.000 y - 10.000 z"),
      "the caption must format every coefficient: {closed:.2000}"
    );

    // The blue line reaches the edge of the box: without the unbounded
    // primitives the picture held nothing but the marker spheres.
    assert!(
      closed.contains("stroke=\"rgb(0,0,255)\""),
      "the infinite line must be drawn"
    );
    // Both marker spheres are there — the list of centres is a sphere
    // each, not one stray sphere at the origin. Each is tessellated the
    // same way, so the scene holds twice one sphere's triangles.
    let one_sphere = woxi::interpret_with_stdout(
      "ExportString[Graphics3D[{Black, Sphere[{0, 0, 0}, 0.2]}, \
       PlotRange -> 4, ImageSize -> {320, 300}], \"SVG\"]",
    )
    .expect("the reference must render")
    .result;
    let black_faces = |svg: &str| svg.matches("fill=\"rgb(0,0,0)\"").count();
    assert_eq!(
      black_faces(&closed),
      2 * black_faces(&one_sphere),
      "both centres must get a sphere"
    );

    // Turning the checkbox on adds the green plane, clipped to the box.
    let open = render("True");
    let greens = |svg: &str| {
      svg
        .split("<polygon")
        .skip(1)
        .filter(|t| {
          t.split_once("fill=\"rgb(")
            .and_then(|(_, r)| r.split_once(')'))
            .is_some_and(|(c, _)| {
              let v: Vec<u32> =
                c.split(',').filter_map(|n| n.parse().ok()).collect();
              v.len() == 3 && v[1] > v[0] && v[1] > v[2]
            })
        })
        .count()
    };
    assert_eq!(greens(&closed), 0, "the plane is hidden until asked for");
    assert!(greens(&open) > 0, "the plane must fill its cross section");
  }

  #[test]
  fn modular_power_graph_manipulate_draws_its_own_edges() {
    // The shape the modular-arithmetic Demonstrations share: a `GraphPlot`
    // inside a `Pane`, laid out by name with `Method ->
    // "CircularEmbedding"` and drawn edge by edge with an
    // `EdgeShapeFunction`, plus a second slider whose maximum follows the
    // first. Without the shape function the plot came out as the default
    // grey arrows instead of the greyed-out dashed lines.
    let code = "Manipulate[\
      edges = Rule @@@ Flatten[Table[{a, b}, {a, k - 1}, {b, k - 1}], 1]; \
      marked = Rule @@@ Transpose[{Range[k], Mod[Range[k]^p, k, 1]}]; \
      Pane[GraphPlot[edges, Method -> \"CircularEmbedding\", \
        DirectedEdges -> True, \
        EdgeShapeFunction -> (If[MemberQ[List @@@ marked, #2], \
          {Blue, Arrow[#1]}, {LightGray, Dashed, Line[#1]}] &)], 380], \
      {{k, 7, \"modulus\"}, 2, 24, 1, Appearance -> \"Labeled\"}, \
      {{p, 2, \"power\"}, 0, k, 1, Appearance -> \"Labeled\"}, \
      TrackedSymbols -> Manipulate, AutorunSequencing -> {2}]";
    let mut state = instantiate_stored_manipulate(code, "")
      .expect("the modular-power Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(state.graphics_handle.is_some(), "the graph must render");

    let bounds =
      |s: &manipulate::ManipulateState, i: usize| match &s.controls[i] {
        manipulate::ControlState::Continuous { min, max, .. } => (*min, *max),
        other => panic!("expected continuous control, got {other:?}"),
      };
    assert_eq!(bounds(&state, 0), (2.0, 24.0));
    assert_eq!(
      bounds(&state, 1),
      (0.0, 7.0),
      "the power slider stops at the modulus"
    );

    // Widening the modulus widens the power slider with it.
    match &mut state.controls[0] {
      manipulate::ControlState::Continuous { current, .. } => *current = 12.0,
      other => panic!("expected continuous control, got {other:?}"),
    }
    state.reevaluate();
    assert!(state.error.is_none(), "still clean: {:?}", state.error);
    assert_eq!(bounds(&state, 1), (0.0, 12.0));

    // The edges the shape function draws are plain dashed lines, so the
    // picture carries no arrow heads at all.
    let svg = woxi::interpret(
      "ExportString[\
       GraphPlot[Rule @@@ Flatten[Table[{a, b}, {a, 6}, {b, 6}], 1], \
         Method -> \"CircularEmbedding\", DirectedEdges -> True, \
         EdgeShapeFunction -> (If[MemberQ[{{1, 2}}, #2], \
           {Blue, Arrow[#1]}, {LightGray, Dashed, Line[#1]}] &)], \"SVG\"]",
    )
    .expect("the plot must export");
    assert!(
      !svg.contains("<polygon"),
      "every edge is drawn as a dashed line: {svg}"
    );
  }

  /// End-to-end regression for the shape the "Recursive Exercises"
  /// Demonstrations share: a recursive family of nested `Disk`s whose
  /// level setter offers fewer levels once the 3D view is on, a colour
  /// index picked by a slider over a twenty-entry list, and two controls
  /// greyed out while the 3D view is showing.
  ///
  /// Both control shapes were wrong before: the level setter kept all six
  /// choices in 3D (its list was resolved once, at build time), and the
  /// twenty-entry colour list became a dropdown because nothing carried
  /// the `ControlType -> Slider` request through to the widget.
  #[test]
  fn nested_disks_manipulate_follows_its_dependent_controls() {
    woxi::interpret(
      "ring[c_, r_] := {White, Disk[c, r]}\n\
       nest[a_, b_, 1, _] := ring[{(a + b)/2, 0}, (b - a)/2]\n\
       nest[a_, b_, k_, sch_] := {ring[{(a + b)/2, 0}, (b - a)/2], \
         nest[a, (a + b)/2, k - 1, sch], nest[(a + b)/2, b, k - 1, sch], \
         ColorData[sch, \"ColorList\"][[9 - k]], \
         Disk[{(a + b)/2, (b - a)/3}, (b - a)/6], \
         Disk[{(a + b)/2, -(b - a)/3}, (b - a)/6]}",
    )
    .expect("the recursive helpers must define");

    let code = "Manipulate[\
      shapes = {ring[{0, 0}, 1], nest[-1., 1., lvl, sch]}; \
      edge = If[wide, 0.01, 0.002]; \
      If[solid, \
        Graphics3D[{EdgeForm[Thickness[edge]], Opacity[0.5], \
          (shapes /. Disk[{px_, py_}, pr_] :> Sphere[{px, py, 0}, pr])}, \
         Boxed -> False, ViewPoint -> {0, 0, 1}], \
        Graphics[{EdgeForm[Thickness[edge]], shapes}, PlotRange -> All]], \
      {{lvl, 2, \"level\"}, Range[1, If[solid, 3, 6], 1], \
        ControlType -> Setter}, \
      {{solid, False, \"3D version\"}, {True, False}, Enabled -> (lvl < 4)}, \
      {{wide, False, \"thick\"}, {True, False}, Enabled -> !solid}, \
      {{sch, 1, \"disks color\"}, Range[20], ControlType -> Slider, \
        Enabled -> !solid}, \
      SaveDefinitions -> True, TrackedSymbols -> Manipulate]";
    let mut state = instantiate_stored_manipulate(code, "")
      .expect("the nested-disks Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(state.graphics_handle.is_some(), "the disks must render");

    let discrete =
      |state: &manipulate::ManipulateState, idx: usize| match &state.controls
        [idx]
      {
        manipulate::ControlState::Discrete {
          values,
          current_index,
          slider,
          ..
        } => (values.clone(), *current_index, *slider),
        other => panic!("expected a discrete control, got {other:?}"),
      };

    // Flat: six levels, and the colour picker asks for a slider rather
    // than the dropdown twenty choices would otherwise get.
    assert_eq!(discrete(&state, 0).0.len(), 6);
    assert_eq!(discrete(&state, 0).1, 1, "level starts at 2");
    let (colours, colour_index, colour_slider) = discrete(&state, 3);
    assert_eq!(colours.len(), 20);
    assert_eq!(colour_index, 0);
    assert!(colour_slider, "ControlType -> Slider must reach the widget");
    assert_eq!(
      state.control_is_enabled,
      vec![true, true, true, true],
      "nothing is greyed out while the flat view is showing"
    );

    // Switching to 3D narrows the level setter to three choices, keeps the
    // selected level (2 is still offered), and greys out the two controls
    // whose `Enabled` conditions read the 3D flag.
    if let manipulate::ControlState::Discrete { current_index, .. } =
      &mut state.controls[1]
    {
      *current_index = 0; // True
    }
    state.reevaluate();
    assert!(state.error.is_none(), "3D render failed: {:?}", state.error);
    assert!(state.graphics_handle.is_some(), "the spheres must render");
    assert_eq!(discrete(&state, 0).0, vec!["1", "2", "3"]);
    assert_eq!(discrete(&state, 0).1, 1, "level 2 survives the narrowing");
    assert_eq!(
      state.control_is_enabled,
      vec![true, true, false, false],
      "thickness and colour are disabled in 3D"
    );

    // Switching back restores all six levels.
    if let manipulate::ControlState::Discrete { current_index, .. } =
      &mut state.controls[1]
    {
      *current_index = 1; // False
    }
    state.reevaluate();
    assert!(
      state.error.is_none(),
      "flat re-render failed: {:?}",
      state.error
    );
    assert_eq!(discrete(&state, 0).0.len(), 6);
  }

  /// Probe: a single continuous time slider drives a `Which` of several
  /// `Graphics3D` scenes assembled from a helper (built on
  /// `RevolutionPlot3D[…][[1]]`) and combined with axis+pivot `Rotate`.
  /// This shape mirrors a class of "morphing surface" Demonstrations.
  #[test]
  fn probe_time_sliced_rotation_scene() {
    woxi::interpret(
      "band[a1_, a2_, z1_] := {RevolutionPlot3D[{Sin[u] + 2, Cos[u]}, \
         {u, a1*Pi, a2*Pi}, {z, 0, z1*Pi}, Mesh -> None][[1]]}",
    )
    .expect("the helper must define");

    let code = "Manipulate[\
      piece = band[0, 2, 1]; \
      Graphics3D[{\
        Which[\
          s <= 1, band[0, 2, s], \
          1 < s <= 2, Rotate[piece, (s - 1)*Pi, {0, 0, 1}, {-1, 0, 0}], \
          True, Rotate[Rotate[piece, Pi, {0, 0, 1}, {-1, 0, 0}], \
            (s - 2)*Pi/2, {1, 0, 0}]\
        ]\
      }, Boxed -> False], \
      {{s, 0, \"time\"}, 0, 3}, \
      SaveDefinitions -> True, TrackedSymbols -> Manipulate]";
    let mut state = instantiate_stored_manipulate(code, "")
      .expect("the time-sliced rotation Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly at s=0: {:?}",
      state.error
    );
    assert!(state.graphics_handle.is_some(), "the scene must render");

    for probe in [0.5_f64, 1.5, 2.5] {
      if let manipulate::ControlState::Continuous { current, .. } =
        &mut state.controls[0]
      {
        *current = probe;
      } else {
        panic!("expected a continuous control");
      }
      state.reevaluate();
      assert!(
        state.error.is_none(),
        "body must evaluate cleanly at s={probe}: {:?}",
        state.error
      );
      assert!(
        state.graphics_handle.is_some(),
        "the scene must render at s={probe}"
      );
    }
  }

  /// A selected choice that the narrowed list no longer offers falls back
  /// to the last one it does, and the body is rendered again for it — so
  /// the graphic on screen always matches the control below it.
  #[test]
  fn narrowed_choice_list_clamps_a_dropped_selection() {
    let code = "Manipulate[k, \
      {{k, 5, \"k\"}, Range[1, If[few, 2, 5], 1], ControlType -> Setter}, \
      {{few, False, \"few\"}, {True, False}}]";
    let mut state =
      instantiate_stored_manipulate(code, "").expect("the widget must build");
    assert_eq!(state.text_output.as_deref(), Some("5"));

    if let manipulate::ControlState::Discrete { current_index, .. } =
      &mut state.controls[1]
    {
      *current_index = 0; // few = True
    }
    state.reevaluate();
    match &state.controls[0] {
      manipulate::ControlState::Discrete {
        values,
        current_index,
        ..
      } => {
        assert_eq!(values, &["1", "2"]);
        assert_eq!(*current_index, 1, "5 is gone, so the last choice wins");
      }
      other => panic!("expected a discrete control, got {other:?}"),
    }
    assert_eq!(
      state.text_output.as_deref(),
      Some("2"),
      "the output follows the value the control settled on"
    );
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
  fn setter_row_manipulate_renders_its_traditionalform_body() {
    // End-to-end regression for the Demonstrations layout that pairs a
    // `Setter` control and a `Button` inside a `Row[…]` with hidden
    // `ControlType -> None` state, and whose body is
    // `Text@Pane[Column[{TraditionalForm[…], Grid[…], Plot[…]}]]`.
    // Every piece used to break the widget: `Setter` was read as a bound
    // (so the whole Manipulate fell back to a text echo), the hidden
    // variable bound its choice *list* instead of its first choice, and
    // the `Text@Pane` body rendered only the plot buried inside it.
    let code = "Manipulate[\
        Text@Pane[Column[{\
          TraditionalForm[Row[{Style[\"q\", Italic], \"(\", x, \")\"}] == \
            Take[powers, deg + 1] . Reverse[coeffs]], \
          Grid[{Reverse[Take[powers, deg + 1]], coeffs}, Frame -> All], \
          Plot[Take[powers, deg + 1] . Reverse[coeffs], {x, -2, 2}, \
            ImageSize -> {200, Automatic}]}, Alignment -> Center], \
          ImageSize -> {400, 260}], \
        {coeffs, {{1, 1, 1}}, ControlType -> None}, \
        Row[{Control@{{deg, 2, \"degree\"}, Range[1, 4], Setter}, \
          Spacer[10], \
          Button[\"reset\", coeffs = {1, 1, 1}], \
          Spacer[10], \
          Control@{{scale, 1, \"scale\"}, Range[3]}}], \
        TrackedSymbols :> {scale}, \
        Initialization :> (powers = Table[x^i, {i, 0, 6}];)]";
    let state = instantiate_stored_manipulate(code, "")
      .expect("the setter-row Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the initial render must produce the column graphic"
    );

    // A setter, a button and a second setter, in display order.
    let kinds: Vec<&str> = state
      .controls
      .iter()
      .map(|c| match c {
        manipulate::ControlState::Discrete { label, .. } => label.as_str(),
        manipulate::ControlState::Button { label, .. } => label.as_str(),
        other => panic!("unexpected control {other:?}"),
      })
      .collect();
    assert_eq!(kinds, vec!["degree", "reset", "scale"]);
    match &state.controls[0] {
      manipulate::ControlState::Discrete {
        values,
        current_index,
        ..
      } => {
        assert_eq!(values, &vec!["1", "2", "3", "4"]);
        assert_eq!(*current_index, 1, "initial degree 2");
      }
      other => panic!("expected the degree setter, got {other:?}"),
    }
    // The hidden variable starts at the first choice of its domain.
    assert_eq!(
      state.state,
      vec![("coeffs".to_string(), "{1, 1, 1}".to_string())]
    );

    // `TrackedSymbols :> {scale}`: the degree setter moves without
    // re-rendering, the scale setter re-renders.
    let mut state = state;
    assert!(!state.request_reeval(0), "degree is not tracked");
    assert!(state.request_reeval(2), "scale is tracked");
    state.run_scheduled_reeval();
    assert!(state.error.is_none(), "re-render failed: {:?}", state.error);
    assert!(state.graphics_handle.is_some());

    // The button writes the coefficient list back and re-renders.
    let action = state
      .controls
      .iter()
      .find_map(|c| match c {
        manipulate::ControlState::Button { action, .. } => Some(action.clone()),
        _ => None,
      })
      .expect("a button row");
    state.apply_button_action(&action);
    assert!(state.error.is_none(), "button failed: {:?}", state.error);
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
  fn parametric_curves_manipulate_lays_out_its_plots() {
    // End-to-end regression for "Parametric Curves in 2D": four plots are
    // `Inset` into one picture, and the `t` slider is bounded by symbols
    // the body assigns before anything else. The widget did not build at
    // all, and once it did the insets drew as the words "-Graphics-".
    let code = "Manipulate[\n\
      tmin = 0; tmax = 2 Pi;\n\
      Column[{\n\
        Item[Text@TraditionalForm@Framed[Style[Row[{\"x = \", r[t][[1]]}], 20], \
          FrameStyle -> Red], Alignment -> {Center, Top}],\n\
        Graphics[{\n\
          Inset[Show[ParametricPlot[r[t], {t, tmin, tmax}, \
             PlotStyle -> Lighter[Gray, 0.3]], \
            Graphics[{PointSize[Medium], Point[r[tm]]}], \
            ImageSize -> {216, 216}, PlotRange -> {{-1, 1}, {-1, 1}}], \
           {-80, 60}],\n\
          Inset[Plot[r[t][[2]], {t, tmin, tmax}, PlotStyle -> Purple, \
            ImageSize -> {Automatic, 216}], {60, 60}]}, \
         PlotRange -> {{-150, 150}, {-100, 120}}, ImageSize -> 550, \
         Axes -> False]}, Spacings -> -0.5],\n\
      {{tm, 0, Style[\"t\", Italic]}, tmin, tmax, ControlPlacement -> Top, \
       Appearance -> \"Labeled\"},\n\
      {{r, {Cos[3 #], Sin[#]} &, \"\"}, \
       {({Cos[3 #], Sin[#]} &) -> \"a\", ({Cos[#], Sin[#]} &) -> \"b\"}, \
       ControlType -> SetterBar},\n\
      TrackedSymbols -> Manipulate]";
    let mut state = instantiate_stored_manipulate(code, "")
      .expect("the parametric-curves Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(state.graphics_handle.is_some(), "the inset plots must draw");

    // The `t` slider takes its bounds from the body, and the curve picker
    // offers both choices.
    match &state.controls[..] {
      [
        manipulate::ControlState::Continuous {
          label, min, max, ..
        },
        manipulate::ControlState::Discrete { values, .. },
      ] => {
        assert_eq!(label, "t");
        assert_eq!(*min, 0.0);
        assert!((*max - std::f64::consts::TAU).abs() < 1e-9);
        assert_eq!(values.len(), 2, "one choice per curve");
      }
      other => panic!("expected a slider and a picker, got {other:?}"),
    }

    // Tracing the curve re-renders.
    if let manipulate::ControlState::Continuous { current, .. } =
      &mut state.controls[0]
    {
      *current = 2.0;
    }
    state.reevaluate();
    assert!(state.error.is_none(), "re-render failed: {:?}", state.error);
    assert!(state.graphics_handle.is_some());
  }

  #[test]
  fn argand_diagram_manipulate_draws_its_locator_pane() {
    // End-to-end regression for "Argand Diagram": a locator on the complex
    // plane is driven by modulus and argument sliders, so its position is
    // computed first and handed to a two-argument `Dynamic`, and the
    // picture that follows it is `Dynamic`-wrapped. The pane drew nothing
    // but a strip of its own source.
    let code = "Manipulate[LocatorPane[\
      pt = {r Cos[θ], r Sin[θ]}; \
      Dynamic[pt, (r = Norm[#]; θ = ArcCos[#[[1]]/r];) &], \
      Dynamic@Graphics[{Circle[], \
        {RGBColor[.24, .39, .77], \
         Line[r {{0, 0}, {Cos[θ], 0}, {Cos[θ], Sin[θ]}, {0, 0}}], \
         RGBColor[1, .47, 0], Dashing[{.02}], Circle[{0, 0}, r]}, \
        Text[Style[TraditionalForm[\
          PaddedForm[Chop[r (Cos[θ] + I Sin[θ])], {4, 3}]], 14], \
         {1.2, 1.2}, {0, 0}, Background -> White]}, \
       ImageSize -> {500, 400}, PlotRange -> {2 {-1, 1}, 2 {-1, 1}}], \
      Appearance -> Graphics[{PointSize[.03], RGBColor[1, .47, 0], \
        Point[{0, 0}]}]], \
      {{pt, {.5, .5}}, {-1, -1}, {1, 1}, ControlType -> None}, \
      {{θ, π/4., \"argument\"}, 0., 2 π, Appearance -> \"Labeled\"}, \
      {{r, .5, \"modulus\"}, 0., Norm[{2, 2}], \
       Appearance -> \"Labeled\"}]";
    let mut state = instantiate_stored_manipulate(code, "")
      .expect("the Argand Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the locator pane must draw the diagram"
    );

    // The argument and modulus sliders (the locator itself is
    // `ControlType -> None`, driven by them).
    let sliders: Vec<(&str, f64, f64)> = state
      .controls
      .iter()
      .map(|c| match c {
        manipulate::ControlState::Continuous {
          label,
          current,
          max,
          ..
        } => (label.as_str(), *current, *max),
        other => panic!("expected a continuous slider, got {other:?}"),
      })
      .collect();
    assert_eq!(sliders[0].0, "argument");
    assert!((sliders[0].2 - std::f64::consts::TAU).abs() < 1e-9);
    assert_eq!(sliders[1].0, "modulus");
    assert_eq!(sliders[1].1, 0.5);

    // Turning the argument all the way round re-renders.
    if let manipulate::ControlState::Continuous { current, .. } =
      &mut state.controls[0]
    {
      *current = std::f64::consts::PI;
    }
    state.reevaluate();
    assert!(state.error.is_none(), "re-render failed: {:?}", state.error);
    assert!(state.graphics_handle.is_some());
  }

  #[test]
  fn freese_dissection_manipulate_moves_its_pieces() {
    // End-to-end regression for "3D Freese's Dissection of a Regular
    // Dodecagon into Two Squares": a slider slides and turns prisms cut
    // from a dodecagon. The pieces ask for `EdgeForm[]`, and one of them is
    // a concave nine-gon whose naive fan triangulation spiked outside its
    // own outline.
    woxi::interpret(
      "ngon[n_] := Table[{Re[Exp[I 2 Pi i/n]], Im[Exp[I 2 Pi i/n]]}, \
         {i, 0, n}] // N; \
       prizma4[vec_][poli_] := Module[{n = Length[poli]}, \
         {Join[{Reverse[Range[n]]}, {Range[n + 1, 2 n]}, \
            Table[{i, Mod[i + 1, n, 1], Mod[i + 1, n, 1] + n, i + n}, \
              {i, 1, n}]], \
          Join[poli, Table[poli[[i]] + vec, {i, 1, n}]]}]; \
       prizmB[vec_][tocke_] := \
         With[{toc = (tocke /. {x_, y_} -> {x, y, 0})}, \
           With[{tr = prizma4[vec][toc]}, Map[tr[[2, #]] &, tr[[1]]]]]; \
       dod0[aa_] := {aa[[1]], aa[[2]], 0}; \
       pts = ngon[12];",
    )
    .unwrap();
    let code = "Manipulate[\
      Graphics3D[{EdgeForm[], \
        {Brown, Polygon[prizmB[{0, 0, 0.1}][\
          {pts[[11]], pts[[12]], pts[[1]], pts[[2]], pts[[3]]}]]}, \
        Translate[{Green, Polygon[prizmB[{0, 0, 0.1}][\
          {pts[[4]], pts[[5]], pts[[6]], pts[[7]], pts[[8]]}]]}, \
         k1 dod0@(pts[[1]] - pts[[4]])]}, \
       PlotRange -> All, ImageSize -> {450, 450}, Boxed -> False], \
      {{k1, 0, \"move\"}, 0, 1, 0.001, Appearance -> \"Labeled\"}, \
      {{sph, False, \"show photo\"}, {True, False}}, \
      SaveDefinitions -> True]";
    let mut state = instantiate_stored_manipulate(code, "")
      .expect("the dissection Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the initial render must draw the assembled dodecagon"
    );

    // The move slider and the photo toggle.
    match &state.controls[..] {
      [
        manipulate::ControlState::Continuous {
          label,
          current,
          min,
          max,
          ..
        },
        manipulate::ControlState::Discrete {
          label: photo_label, ..
        },
      ] => {
        assert_eq!(label, "move");
        assert_eq!((*current, *min, *max), (0.0, 0.0, 1.0));
        assert_eq!(photo_label, "show photo");
      }
      other => panic!("expected a slider and a toggle, got {other:?}"),
    }

    // Sliding the pieces all the way apart re-renders.
    if let manipulate::ControlState::Continuous { current, .. } =
      &mut state.controls[0]
    {
      *current = 1.0;
    }
    state.reevaluate();
    assert!(state.error.is_none(), "re-render failed: {:?}", state.error);
    assert!(state.graphics_handle.is_some());
  }

  #[test]
  fn points_connection_manipulate_draws_its_click_pane() {
    // End-to-end regression for "The 6x6 Points Connection Problem": the
    // puzzle is played by clicking a grid of points, so the whole widget
    // body is a `ClickPane`. It built its controls but drew nothing — a
    // pane rendered as the textual echo of its own call.
    woxi::interpret(
      "lx = 6; ly = 6; rx = Range[lx]; ry = Range[ly]; dd = 1; \
       grid = Tuples[{dd rx, dd ry}]; \
       dist[{x1_, y1_}, {x2_, y2_}] := \
         Sqrt[(x1 - x2)^2 + (y1 - y2)^2];",
    )
    .unwrap();
    let code = "Manipulate[\n\
      lp = Length[pts];\n\
      ClickPane[\n\
       Graphics[{Thickness[.02], Black, PointSize[pointsize], \
         Point[Tuples[{dd rx, dd ry}]], \
         Line[{{0, -2}, {8.1, -2}}], \
         Table[Line[{{i, -2.5}, {i, -2}}], {i, 0, 8}], \
         Table[Text[i, {i, -2.7}], {i, 0, 8}], \
         Blue, Table[Point[pts[[i]]], {i, lp}], \
         If[lp > 1, Table[Line[{pts[[i]], pts[[i + 1]]}], {i, lp - 1}], {}]}, \
        ImageSize -> 350], \
       (pts = AppendTo[pts, #]) &], \n\
      {{ch, 1, \"challenge\"}, \
       {1 -> \"any path\", 2 -> \"no point twice\"}, \
       ControlType -> PopupMenu}, \
      {{pointsize, .03, \"pointsize\"}, 0, .1, ControlType -> Slider}, \
      {{pts, {}}, ControlType -> None}]";
    let mut state = instantiate_stored_manipulate(code, "")
      .expect("the points-connection Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the click pane must draw the grid of points"
    );

    // The challenge picker and the point-size slider.
    match &state.controls[..] {
      [
        manipulate::ControlState::Discrete {
          label,
          value_labels,
          ..
        },
        manipulate::ControlState::Continuous {
          label: size_label,
          current,
          min,
          max,
          ..
        },
      ] => {
        assert_eq!(label, "challenge");
        assert_eq!(
          value_labels,
          &["any path".to_string(), "no point twice".to_string()]
        );
        assert_eq!(size_label, "pointsize");
        assert_eq!((*current, *min, *max), (0.03, 0.0, 0.1));
      }
      other => panic!("expected a picker and a slider, got {other:?}"),
    }

    // Growing the points re-renders the pane.
    if let manipulate::ControlState::Continuous { current, .. } =
      &mut state.controls[1]
    {
      *current = 0.08;
    }
    state.reevaluate();
    assert!(state.error.is_none(), "re-render failed: {:?}", state.error);
    assert!(state.graphics_handle.is_some());
  }

  #[test]
  fn manipulate_row_with_dynamic_shows_live_value_not_frozen_text() {
    // Regression for "Water Colors Puzzle": a Demonstration's live move
    // counter is often written directly among the control-panel arguments as
    // `Row[{Style["moves: "], Dynamic[moves]}]`. `Row`/`Column`/`Style` are
    // normally treated as a static caption row (Wolfram's
    // `ThisIsNotAControl`), but one that embeds a `Dynamic[…]` must track
    // that variable's live value every frame instead of freezing into a
    // `Heading` that echoes the bare `Dynamic[moves]` source as literal text.
    let code = "Manipulate[\n\
      moves = moves + go;\n\
      go,\n\
      {{go, 0}, ControlType -> None},\n\
      Row[{Style[\"moves: \"], Dynamic[moves]}],\n\
      {{moves, 0}, ControlType -> None}\n\
      ]";
    let mut state =
      instantiate_stored_manipulate(code, "").expect("must build a widget");
    assert!(state.error.is_none(), "body must evaluate cleanly");

    // The Dynamic row must not show up as a frozen `Heading` echoing the
    // `Dynamic[moves]` source.
    assert!(
      !state
        .controls
        .iter()
        .any(|c| matches!(c, manipulate::ControlState::Heading { .. })),
      "the Row[{{…, Dynamic[moves]}}] must not freeze into a Heading, got {:?}",
      state.controls
    );

    // It renders live instead, starting at `moves`'s initial value (0).
    fn row_text(node: &woxi::functions::graphics::DisplayNode) -> String {
      use woxi::functions::graphics::DisplayNode;
      match node {
        DisplayNode::Row(children) | DisplayNode::Column(children) => {
          children.iter().map(row_text).collect()
        }
        DisplayNode::Text { runs } => {
          runs.iter().map(|r| r.text.as_str()).collect()
        }
        DisplayNode::Static { text, .. } => text.clone(),
        _ => String::new(),
      }
    }
    let moves_row = state
      .display_trees
      .iter()
      .find(|t| row_text(t).contains("moves:"))
      .unwrap_or_else(|| {
        panic!(
          "no display tree has the moves row: {:?}",
          state.display_trees
        )
      });
    assert_eq!(row_text(moves_row), "moves: 0");

    // Bumping `go` and re-rendering must move the live counter, not the
    // frozen text a `Heading` would have kept forever.
    if let Some(slot) = state.state.iter_mut().find(|(n, _)| n == "go") {
      slot.1 = "1".to_string();
    }
    state.reevaluate();
    let moves_row = state
      .display_trees
      .iter()
      .find(|t| row_text(t).contains("moves:"))
      .unwrap_or_else(|| {
        panic!(
          "no display tree has the moves row: {:?}",
          state.display_trees
        )
      });
    assert_eq!(row_text(moves_row), "moves: 1");
  }

  #[test]
  fn solar_panel_manipulate_folds_its_array() {
    // End-to-end regression for "Solar Panel of NASA's Phoenix Mars
    // Lander": two sliders fold a fan of triangular panels around a shaft.
    // Every panel is a flat `Polygon`, and the outline between them is what
    // makes the array read as separate panels rather than one disc.
    let code = "Manipulate[\n\
      z = Sin[Pi/n];\n\
      vO = {0, 0, 0}; vA = {0, -1, 0};\n\
      vB = {Sin[a 2 Pi/n], -Cos[a 2 Pi/n], 0};\n\
      vC = {Sin[a 2 Pi/n]/2, (-1 - Cos[a 2 Pi/n])/2, -z + a z};\n\
      OAC = Polygon[{vO, vA, vC}]; OCB = Polygon[{vO, vC, vB}];\n\
      full = {Specularity[0.7], RGBColor[1, 0.8, 0.2], \
        Table[Rotate[{OAC, OCB}, i a 2 Pi/n, {0, 0, 1}], {i, n}]};\n\
      shaft = Cylinder[{{0, 0, 0.1}, {0, 0, -1}}, .06];\n\
      hub = {RGBColor[0.4, 0.8, 0.9], \
        Cylinder[{{0, 0, 0.06}, {0, 0, -0.04}}, 0.1]};\n\
      Graphics3D[{full, shaft, hub}, ImageSize -> {380, 380}, \
        SphericalRegion -> True, Boxed -> False, \
        PlotRange -> {{-1, 1}, {-1, 1}, {-1, .2}}], \
      {{a, .9, \"deploy\"}, 0.02, 1}, \
      {{n, 12, \"number of segments\"}, 3, 20, 1}, \
      TrackedSymbols -> Manipulate]";
    let mut state = instantiate_stored_manipulate(code, "")
      .expect("the solar-panel Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the initial render must draw the panel array"
    );

    // The deployment angle and the panel count.
    let sliders: Vec<(&str, f64, f64, f64, f64)> = state
      .controls
      .iter()
      .map(|c| match c {
        manipulate::ControlState::Continuous {
          label,
          current,
          min,
          max,
          step,
          ..
        } => (label.as_str(), *current, *min, *max, *step),
        other => panic!("expected a continuous slider, got {other:?}"),
      })
      .collect();
    assert_eq!(sliders[0].0, "deploy");
    assert_eq!((sliders[0].1, sliders[0].2, sliders[0].3), (0.9, 0.02, 1.0));
    assert_eq!(sliders[1].0, "number of segments");
    assert_eq!(
      (sliders[1].1, sliders[1].2, sliders[1].3, sliders[1].4),
      (12.0, 3.0, 20.0, 1.0)
    );

    // Folding the array right down to three segments still renders.
    if let manipulate::ControlState::Continuous { current, .. } =
      &mut state.controls[0]
    {
      *current = 0.02;
    }
    if let manipulate::ControlState::Continuous { current, .. } =
      &mut state.controls[1]
    {
      *current = 3.0;
    }
    state.reevaluate();
    assert!(state.error.is_none(), "re-render failed: {:?}", state.error);
    assert!(state.graphics_handle.is_some());
  }

  #[test]
  fn dijkstra_manipulate_builds_its_button_bar() {
    // End-to-end regression for "Dijkstra's Algorithm": the graph is stepped
    // through by buttons, and the starting vertex is picked from a
    // `ButtonBar` whose buttons are built by a `Table` over labels the
    // `Initialization` option defines. The bar contributed no controls at
    // all, so there was no way to choose a starting vertex.
    let code = "Manipulate[\
      Column[{Text@Style[Row[{\"starting at vertex \", \
         Style[vertNames[[initialVertex]], Italic]}], Bold, 20], \
        Graphics[{Point /@ points, \
          Text[Style[Row[{\"(\", vertLabels[[1]], \")\"}], 12], {1, 1}]}, \
         ImageSize -> 200]}, Alignment -> Center], \
      Button[\"start over\", vertLabels = Table[Infinity, {4}]], \
      Delimiter, \"initial vertex\", \
      ButtonBar[Flatten[Table[{With[{k = k}, \
        vertNames[[k]] :> {initialVertex = k; \
          vertLabels = ReplacePart[Table[Infinity, {4}], k -> 0]}]}, \
        {k, 1, 4}]], ImageSize -> 25], \
      {{initialVertex, 1}, ControlType -> None}, \
      {{vertLabels, {0, Infinity, Infinity, Infinity}}, \
        ControlType -> None}, \
      Initialization :> (vertNames = {\"a\", \"b\", \"c\", \"d\"}; \
        points = {{1, 1}, {2, 1}, {1, 2}, {2, 2}};), \
      ControlPlacement -> Left]";
    let mut state = instantiate_stored_manipulate(code, "")
      .expect("the Dijkstra Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the initial render must draw the graph"
    );

    // The "start over" button, the heading, then one button per vertex out
    // of the ButtonBar.
    let labels: Vec<String> = state
      .controls
      .iter()
      .map(|c| match c {
        manipulate::ControlState::Button { label, .. } => label.clone(),
        manipulate::ControlState::Heading { label, .. } => {
          format!("#{label}")
        }
        manipulate::ControlState::Divider => "|".to_string(),
        other => panic!("unexpected control {other:?}"),
      })
      .collect();
    assert_eq!(
      labels,
      vec!["start over", "|", "#initial vertex", "a", "b", "c", "d"]
    );

    // Pressing a vertex button runs its action and re-renders.
    let vertex_c = state
      .controls
      .iter()
      .find_map(|c| match c {
        manipulate::ControlState::Button { label, action, .. }
          if label == "c" =>
        {
          Some(action.clone())
        }
        _ => None,
      })
      .expect("a button for vertex c");
    state.apply_button_action(&vertex_c);
    assert!(state.error.is_none(), "re-render failed: {:?}", state.error);
    assert!(state.graphics_handle.is_some());
  }

  #[test]
  fn vertex_distance_manipulate_builds_six_locators() {
    // End-to-end regression for "A Vertex Distance Relation for Two
    // Triangles": six draggable vertices drive two triangles, and a grid
    // above them shows both sides of the identity being illustrated. Its
    // header is a typeset linear-syntax label and the two sides are
    // multiplied with `×` — the named character, which used to be a parse
    // error that killed the whole cell.
    let code = "Manipulate[Module[{ABC, DEF}, \
      ABC = Abs[Det[Append[#, 1] & /@ {AA, BB, CC}]/2]; \
      DEF = Abs[Det[Append[#, 1] & /@ {DD, EE, FF}]/2]; \
      Column[{\
        Text@Grid[{{\"\\!\\(TraditionalForm\\`16\\\\ \
          \\*SubscriptBox[\\(S\\), \\(ABC\\)]\\)\", \"formula\"}, \
          {16 × ABC × DEF, 16 × ABC × DEF}}, Frame -> All, \
          ItemSize -> {8}], \
        Graphics[{Style[Triangle[{AA, BB, CC}], Opacity[.2], \
            EdgeForm[{Thick, Blue}]], \
          Style[Triangle[{DD, EE, FF}], Opacity[.2], \
            EdgeForm[{Thick, Red}]], \
          Text[Style[\"A\", 20, \"Label\"], AA + {0, .3}]}, \
          ImageSize -> 1.125 {450, 375}, PlotRange -> 4.5]}, \
        Alignment -> {Center, Top}]], \
      {{AA, {2, -1}}, {-4, -4}, {4, 4}, Locator}, \
      {{BB, {1, 2}}, {-4, -4}, {4, 4}, Locator}, \
      {{CC, {0, 0}}, {-4, -4}, {4, 4}, Locator}, \
      {{DD, {3.955, 1.}}, {-4, -4}, {4, 4}, Locator}, \
      {{EE, {-2.98, -2.56}}, {-4, -4}, {4, 4}, Locator}, \
      {{FF, {-1.545, 3.9}}, {-4, -4}, {4, 4}, Locator}, \
      TrackedSymbols :> {AA, BB, CC, DD, EE, FF}\
      (*, SynchronousUpdating -> False*)]";
    let mut state = instantiate_stored_manipulate(code, "")
      .expect("the vertex-distance Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the initial render must draw both triangles"
    );

    // Six draggable vertices, each a 2D locator over the same square.
    let locators: Vec<(&str, f64, f64)> = state
      .controls
      .iter()
      .map(|c| match c {
        manipulate::ControlState::Slider2D { name, x, y, .. } => {
          (name.as_str(), *x, *y)
        }
        other => panic!("expected a locator, got {other:?}"),
      })
      .collect();
    assert_eq!(
      locators,
      vec![
        ("AA", 2.0, -1.0),
        ("BB", 1.0, 2.0),
        ("CC", 0.0, 0.0),
        ("DD", 3.955, 1.0),
        ("EE", -2.98, -2.56),
        ("FF", -1.545, 3.9),
      ]
    );

    // Dragging a vertex re-solves the areas and re-renders.
    if let manipulate::ControlState::Slider2D { x, y, .. } =
      &mut state.controls[0]
    {
      *x = -1.5;
      *y = 3.0;
    }
    state.reevaluate();
    assert!(state.error.is_none(), "re-render failed: {:?}", state.error);
    assert!(state.graphics_handle.is_some());
  }

  #[test]
  fn constraint_tiling_manipulate_switches_net_and_solid() {
    // End-to-end regression for "Constraint Tiling on a Truncated
    // Icosahedron": a setter bar picks one of six constraint sets and a
    // second one switches between the flat net and the solid. The solid's
    // vertex coordinates are `Root[…]` objects — algebraic numbers with no
    // radical form — and every face that touched one used to be dropped.
    woxi::interpret(
      "hexFace[c_] := {GrayLevel[c], EdgeForm[Thick], \
         Polygon[Table[{Cos[i Pi/3], Sin[i Pi/3]}, {i, 6}]]}; \
       solid[c_] := {EdgeForm[Thick], Darker[Darker[Red]], \
         Polygon[{{0, 0, Root[1 - 20 #^2 + 80 #^4 &, 1]}, {1, 0, 0}, \
           {1, 1, 0}}], GrayLevel[c], \
         Polygon[{{0, 0, 0}, {1, 0, 0}, {1, 1, 1}}]}; \
       display[n_, tf_] := If[tf === 0, \
         Column[{Graphics[hexFace[n/6], ImageSize -> {200, 200}], \
           Row[{Graphics[hexFace[0], ImageSize -> {50, 50}]}]}, Center], \
         Column[{Graphics3D[{Glow[], Specularity[Black], solid[n/6]}, \
           Lighting -> \"Neutral\", ImageSize -> {200, 200}, \
           Boxed -> False], \
           Row[{Graphics[hexFace[0], ImageSize -> {50, 50}]}]}, Center]];",
    )
    .unwrap();
    let code = "Manipulate[display[code, tf], \
      {{code, 6, \"constraint set\"}, 1, 6, 1, ControlType -> SetterBar}, \
      {{tf, 1, \"\"}, {0 -> \"net\", 1 -> \"polyhedron\"}, \
       ControlType -> SetterBar}, \
      SynchronousUpdating -> False, SaveDefinitions -> True, \
      AutorunSequencing -> {1, 2}]";
    let mut state = instantiate_stored_manipulate(code, "")
      .expect("the constraint-tiling Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the initial render must draw the solid"
    );

    // Two setter bars: the six constraint sets, then the net/solid switch
    // whose labels come from its `value -> label` rules.
    match &state.controls[..] {
      [
        manipulate::ControlState::Discrete {
          label,
          values,
          current_index,
          ..
        },
        manipulate::ControlState::Discrete {
          value_labels,
          current_index: view_index,
          ..
        },
      ] => {
        assert_eq!(label, "constraint set");
        assert_eq!(values.len(), 6, "one choice per constraint set");
        assert_eq!(*current_index, 5, "the sixth set is selected");
        assert_eq!(
          value_labels,
          &["net".to_string(), "polyhedron".to_string()]
        );
        assert_eq!(*view_index, 1, "the solid is shown first");
      }
      other => panic!("expected two setter bars, got {other:?}"),
    }

    // Switching to the flat net re-renders.
    if let manipulate::ControlState::Discrete { current_index, .. } =
      &mut state.controls[1]
    {
      *current_index = 0;
    }
    state.reevaluate();
    assert!(state.error.is_none(), "re-render failed: {:?}", state.error);
    assert!(state.graphics_handle.is_some());
  }

  #[test]
  fn integer_grid_triangle_manipulate_draws_its_locators() {
    // End-to-end regression for "Area of a Triangle on an Integer Grid": a
    // seeded random triangle is shown in a `LocatorPane` whose three
    // vertices carry lettered markers, with the coordinates beside it (a
    // `Dynamic` label) and the area below. Everything the widget draws sits
    // inside display wrappers — `Column[Pane[Labeled[LocatorPane[…]]]]` —
    // which used to print as a line of source text instead.
    woxi::interpret(
      "names1[fonts1_] := Style[#, fonts1] & /@ CharacterRange[\"A\", \"U\"]; \
       names1A[fonts1_] := Table[Graphics[{{RGBColor[0.501961, 0, 0.25098], \
         PointSize -> 0.02, Point[{0, 0}]}, {RGBColor[0, 0.501961, 1], \
         Text[names1[fonts1][[i]], {0, 0}, {-1, -1}]}}], {i, 1, 20}]; \
       RandomKsublist[set_, k_] := Module[{sub = {}, rest = set, new}, \
         Do[new = rest[[RandomInteger[{1, Length[rest]}]]]; \
            sub = AppendTo[sub, new]; rest = Complement[rest, {new}], {k}]; \
         sub]; \
       set2[n_] := Flatten[Table[{i, j}, {i, 0, n - 1}, {j, 0, n - 1}], 1];",
    )
    .unwrap();
    let code = "Manipulate[\n\
      SeedRandom[seed];\n\
      With[{st = RandomKsublist[set2[9], 3]},\n\
       DynamicModule[{nn = st},\n\
        Column[{\n\
         Pane[Labeled[\n\
           LocatorPane[Dynamic[nn],\n\
            Graphics[{Line[Dynamic@{nn[[1]], nn[[2]], nn[[3]], nn[[1]]}]},\n\
             Axes -> True, PlotRange -> {{-1, 10}, {-1, 10}},\n\
             GridLines -> If[!help, None, {Range[-10, 10], Range[-10, 10]}],\n\
             GridLinesStyle -> Directive[RGBColor[1, 0.72549, 0.72549], Thin],\n\
             ImageSize -> {450, 430}],\n\
            {{-9, -9}, {9, 9}, {1, 1}}, Appearance -> names1A[14]],\n\
           Dynamic[Text@Grid[Table[{names1[14][[i]], \"=\", \
             StringReplace[ToString@nn[[i]], {\"{\" -> \"(\", \"}\" -> \")\"}]}, \
             {i, 1, 3}]]], Right], {550, 450}],\n\
         Text@Style[#, 14] &@Row[{Spacer[50], \"area of triangle ABC = \", \
           FullSimplify@(Dynamic@Abs[Det[{(nn[[2]] - nn[[1]])/2, \
             nn[[3]] - nn[[1]]}]]), \".\"}]}]]],\n\
      Row[{Control@{{help, False, \"show grid lines\"}, {False, True}}, \
        Spacer[20], \
        Control@{{seed, 1, \"random seed\"}, 1, 100000, 1, \
          Appearance -> \"Labeled\"}}],\n\
      SaveDefinitions -> True, AutorunSequencing -> {1, 2}]";
    let mut state = instantiate_stored_manipulate(code, "")
      .expect("the integer-grid Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the initial render must draw the triangle and its locators"
    );

    // Both controls come out of the `Row[…]` they are wrapped in: the
    // grid-lines checkbox and the random-seed slider.
    match &state.controls[..] {
      [
        manipulate::ControlState::Discrete { label, values, .. },
        manipulate::ControlState::Continuous {
          label: seed_label,
          min,
          max,
          current,
          ..
        },
      ] => {
        assert_eq!(label, "show grid lines");
        assert_eq!(values, &["False".to_string(), "True".to_string()]);
        assert_eq!(seed_label, "random seed");
        assert_eq!((*current, *min, *max), (1.0, 1.0, 100000.0));
      }
      other => panic!("expected a checkbox and a slider, got {other:?}"),
    }

    // Turning the grid lines on and reseeding both re-render.
    if let manipulate::ControlState::Discrete { current_index, .. } =
      &mut state.controls[0]
    {
      *current_index = 1;
    }
    if let manipulate::ControlState::Continuous { current, .. } =
      &mut state.controls[1]
    {
      *current = 42.0;
    }
    state.reevaluate();
    assert!(state.error.is_none(), "re-render failed: {:?}", state.error);
    assert!(state.graphics_handle.is_some());
  }

  #[test]
  fn diatomic_molecule_manipulate_builds_six_sliders() {
    // End-to-end regression for "The Six Degrees of Freedom of a Diatomic
    // Molecule": three translations, two rotations and the bond length drive
    // a `Graphics3D` of two spheres joined by a zigzag spring. The `Table`
    // that draws the spring reuses `x` as its own iterator, shadowing the
    // Manipulate variable of the same name that positions the molecule.
    let code = "Manipulate[\
      Graphics3D[\
       Translate[\
        Rotate[Rotate[{Blue, Thickness[.003], \
          Line[Table[{x r, 0, .25 (-1)^IntegerPart[4 x]}, {x, -1, 1, .25}]], \
          Green, Sphere[-{1 r, 0, 0}, .35], Sphere[{1 r, 0, 0}, .35]}, \
         θ, {0, 0, 1}], ϕ, {0, 1, 0}], {x, y, z}], \
       PlotRange -> {{-2.5, 7.5}, {-3, 7.5}, {-2, 7.5}}, \
       SphericalRegion -> True, ImageSize -> {400, 400}], \
      {x, 0, 5, ImageSize -> Tiny}, \
      {y, 0, 5, ImageSize -> Tiny}, \
      {z, 0, 5, ImageSize -> Tiny}, \
      {θ, 0, 2π, ImageSize -> Tiny}, \
      {ϕ, 0, 2π, ImageSize -> Tiny}, \
      {r, 0.6, 2, ImageSize -> Tiny}, \
      ControlPlacement -> Left]";
    let mut state = instantiate_stored_manipulate(code, "")
      .expect("the diatomic-molecule Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the initial render must produce the molecule"
    );

    // Six sliders: the three positions, the two Euler angles and the bond
    // length, each starting at the low end of its range.
    let sliders: Vec<(&str, f64, f64, f64)> = state
      .controls
      .iter()
      .map(|c| match c {
        manipulate::ControlState::Continuous {
          name,
          current,
          min,
          max,
          ..
        } => (name.as_str(), *current, *min, *max),
        other => panic!("expected a continuous slider, got {other:?}"),
      })
      .collect();
    let tau = std::f64::consts::TAU;
    assert_eq!(
      sliders,
      vec![
        ("x", 0.0, 0.0, 5.0),
        ("y", 0.0, 0.0, 5.0),
        ("z", 0.0, 0.0, 5.0),
        ("θ", 0.0, 0.0, tau),
        ("ϕ", 0.0, 0.0, tau),
        ("r", 0.6, 0.6, 2.0),
      ]
    );

    // Translating and rotating the molecule re-renders it.
    for (i, value) in [(0usize, 3.0), (3, 0.5), (5, 1.2)] {
      if let manipulate::ControlState::Continuous { current, .. } =
        &mut state.controls[i]
      {
        *current = value;
      }
    }
    state.reevaluate();
    assert!(state.error.is_none(), "re-render failed: {:?}", state.error);
    assert!(state.graphics_handle.is_some());
  }

  #[test]
  fn axial_dispersion_manipulate_builds_widget() {
    // End-to-end regression for the "Response of a Reactor with Axial
    // Dispersion to a Pulse Input Tracer (E-Curve)" Demonstration: a
    // ten-cell finite-difference discretization whose unknowns are
    // `Subscript[c, i]`, driven by a tracer pulse only 10^-6 wide, with the
    // package load in `Initialization` that the notebook asks for.
    //
    // The integration domain is the plotted `{t, 0, 5}` rather than the
    // notebook's `{t, 0, 20}` — solving four times as far only makes the
    // test slower, and the E-curve has decayed to nothing by t = 5 anyway.
    // The numbers themselves are checked in the interpreter's NDSolve tests.
    let code = "Manipulate[\n\
      Module[{sol, plt, dz, n, t, c, eq, Z},\n\
       dz = 1/Subscript[n, Z]; Subscript[n, Z] = 10;\n\
       Subscript[c, 0][t_] := 0.01 If[0 <= t <= 10^-6, 10^6, 0];\n\
       eq[1] = D[Subscript[c, 1][t], {t, 1}] == \
         (Subscript[c, 0][t] - Subscript[c, 1][t])/dz \
         - (Subscript[c, 1][t] - Subscript[c, 2][t])/(dz^2 Pe);\n\
       Table[eq[i] = D[Subscript[c, i][t], {t, 1}] == \
         (Subscript[c, i - 1][t] - Subscript[c, i][t])/dz \
         + (Subscript[c, i - 1][t] - 2 Subscript[c, i][t] \
         + Subscript[c, i + 1][t])/(dz^2 Pe), \
         {i, 2, Subscript[n, Z] - 1}];\n\
       eq[Subscript[n, Z]] = D[Subscript[c, Subscript[n, Z]][t], {t, 1}] == \
         (Subscript[c, Subscript[n, Z] - 1][t] \
         - Subscript[c, Subscript[n, Z]][t])/dz \
         + (Subscript[c, Subscript[n, Z] - 1][t] \
         - Subscript[c, Subscript[n, Z]][t])/(dz^2 Pe);\n\
       sol = NDSolve[Join[Table[eq[i], {i, 1, Subscript[n, Z]}], \
         Table[Subscript[c, i][0] == 0, {i, 1, Subscript[n, Z]}]], \
         Table[Subscript[c, i], {i, 1, Subscript[n, Z]}], {t, 0, 5}, \
         Method -> DifferentialEquations`NDSolveUtilities`StiffnessSwitching];\n\
       plt = Plot[First[Subscript[c, Subscript[n, Z]][t] /. sol], {t, 0, 5}, \
         PlotRange -> {0, 0.02}, PlotStyle -> Thickness[0.01], \
         Frame -> True, ImageSize -> {500, 400}, \
         FrameLabel -> {\"time\", \"concentration\"}]],\n\
      {{Pe, 50, \"Péclet number\"}, 0.5, 100, 0.5, \
       Appearance -> \"Labeled\"},\n\
      SynchronousUpdating -> False,\n\
      TrackedSymbols :> {Pe},\n\
      Initialization :> \
       (Get[\"DifferentialEquations`NDSolveUtilities`\"];)]";
    let mut state = instantiate_stored_manipulate(code, "")
      .expect("the axial-dispersion Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the initial render must produce the E-curve plot"
    );

    // One labeled Péclet-number slider.
    match &state.controls[..] {
      [
        manipulate::ControlState::Continuous {
          label,
          current,
          min,
          max,
          ..
        },
      ] => {
        assert_eq!(label, "Péclet number");
        assert_eq!((*current, *min, *max), (50.0, 0.5, 100.0));
      }
      other => panic!("expected one continuous slider, got {other:?}"),
    }

    // A smaller Péclet number means more dispersion; the system re-solves.
    if let manipulate::ControlState::Continuous { current, .. } =
      &mut state.controls[0]
    {
      *current = 10.0;
    }
    state.reevaluate();
    assert!(state.error.is_none(), "re-render failed: {:?}", state.error);
    assert!(state.graphics_handle.is_some());
  }

  #[test]
  fn orthic_triangle_manipulate_builds_widget() {
    // End-to-end regression for the "Three Concyclic Sets of Points
    // Associated with the Orthic Triangle" Demonstration: three Locator
    // controls drive a Module whose whole geometry is computed inside an
    // `If[nonDegenerate, …]` and then drawn by an `If[nonDegenerate,
    // Unevaluated[Sequence[…]], {}]` inside the Graphics primitive list.
    //
    // Two things used to break it: the `Unevaluated[Sequence[…]]` returned by
    // the `If` was not spliced (so the primitives referenced unbound Module
    // locals), and the round-off stripping rule `Complex[a_, b_] /; Abs[b] <
    // 10^-4 :> a` never matched, so an intersection came back complex.
    let code = "Manipulate[\n\
      Module[{ok, ah, bh, ch, icd, iab, h},\n\
       ok = area[pt1, pt2, pt3] > 10^-4;\n\
       If[ok,\n\
        ah = foot[pt1, pt2, pt3];\n\
        bh = foot[pt2, pt1, pt3];\n\
        ch = foot[pt3, pt1, pt2];\n\
        icd = incircle[ah, bh, ch];\n\
        iab = If[icd =!= {}, cut[ah, bh, Sequence @@ icd], {}];\n\
        h = meet[pt1, ah, pt2, bh]];\n\
       Graphics[{PointSize[.02], Thickness[.01], RGBColor[1, .26, 0],\n\
        Line[{pt1, pt2, pt3, pt1}],\n\
        If[ok, Unevaluated[Sequence[\n\
          RGBColor[.79, .71, .26],\n\
          Line /@ Transpose[{{pt1, pt2, pt3}, {ah, bh, ch}}],\n\
          RGBColor[1, .71, 0], Line[{ah, bh, ch, ah}],\n\
          RGBColor[.45, .7, .55], If[icd =!= {}, Circle @@ icd, {}],\n\
          RGBColor[.48, .11, .56],\n\
          Point[DeleteCases[{h, ah, bh, ch, iab}, {}]],\n\
          Black, Text[Style[\"H\", 20], h + {0, .4}]]], {}]},\n\
        ImageSize -> {450, 400}, PlotRange -> {{-4.1, 4.1}, {-4.1, 4.5}}]],\n\
      {{pt1, {-3.74, -3.61}}, {-4, -4}, {4, 4}, Locator},\n\
      {{pt2, {-0.62, 3.87}}, {-4, -4}, {4, 4}, Locator},\n\
      {{pt3, {4., -0.42}}, {-4, -4}, {4, 4}, Locator},\n\
      Initialization :> (\n\
       area[p_, q_, r_] := \
         Sqrt[#(# - EuclideanDistance[p, q]) (# - EuclideanDistance[q, r]) \
           (# - EuclideanDistance[p, r])] &[\
         (EuclideanDistance[p, q] + EuclideanDistance[q, r] \
           + EuclideanDistance[p, r])/2] \
         /. Complex[a_, b_] /; Max[Abs@a, Abs@b] < 10^-4 -> 0;\n\
       meet[p1_, p2_, p3_, p4_] := Block[{x, y},\n\
        {x, y} /. NSolve[{(y - p1[[2]]) (p2[[1]] - p1[[1]]) == \
           (p2[[2]] - p1[[2]]) (x - p1[[1]]),\n\
          (y - p3[[2]]) (p4[[1]] - p3[[1]]) == \
           (p4[[2]] - p3[[2]]) (x - p3[[1]])}, {x, y}][[1]]];\n\
       foot[pt_, pt1_, pt2_] := Block[{x, y},\n\
        {x, y} /. NSolve[{(pt2[[1]] - pt1[[1]]) (y - pt1[[2]]) == \
           (pt2[[2]] - pt1[[2]]) (x - pt1[[1]]),\n\
          (pt2[[2]] - pt1[[2]]) (y - pt[[2]]) == \
           (pt1[[1]] - pt2[[1]]) (x - pt[[1]])}, {x, y}][[1]]];\n\
       incircle[pA_, pB_, pC_] := Module[{a, b, c, s},\n\
        a = EuclideanDistance[pB, pC]; b = EuclideanDistance[pA, pC];\n\
        c = EuclideanDistance[pA, pB]; s = (a + b + c)/2;\n\
        {(a pA + b pB + c pC)/(a + b + c), \
          Sqrt[(s - a) (s - b) (s - c)/s]}];\n\
       cut[p1_, p2_, cen_, rad_] := Block[{x, y},\n\
        {x, y} /. Quiet[NSolve[{(y - p1[[2]]) (p2[[1]] - p1[[1]]) == \
           (p2[[2]] - p1[[2]]) (x - p1[[1]]),\n\
          (x - cen[[1]])^2 + (y - cen[[2]])^2 == rad^2}, {x, y}] \
         /. {Complex[a_, b_] /; Abs[b] < 10^-4 :> a}][[1]]];)]";
    let mut state = instantiate_stored_manipulate(code, "")
      .expect("the orthic-triangle Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the initial render must produce the figure"
    );

    // Three 2D locators, each ranged over the {-4, -4}..{4, 4} square.
    match &state.controls[..] {
      [
        manipulate::ControlState::Slider2D {
          name: n1, x: x1, ..
        },
        manipulate::ControlState::Slider2D { name: n2, .. },
        manipulate::ControlState::Slider2D {
          name: n3,
          x_min,
          x_max,
          ..
        },
      ] => {
        assert_eq!(
          (n1.as_str(), n2.as_str(), n3.as_str()),
          ("pt1", "pt2", "pt3")
        );
        assert!((*x1 - -3.74).abs() < 1e-9);
        assert_eq!((*x_min, *x_max), (-4.0, 4.0));
      }
      other => panic!("expected three 2D locators, got {other:?}"),
    }

    // Dragging one vertex re-solves the whole construction.
    if let manipulate::ControlState::Slider2D { x, y, .. } =
      &mut state.controls[0]
    {
      *x = -3.0;
      *y = -2.0;
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
  fn compass_construction_manipulate_builds_widget() {
    // End-to-end regression for a Demonstration that constructs with
    // compasses alone: the initialization crosses two circles through
    // `Solve`, a setter bar steps through the construction, and the body
    // picks that step's graphics out of the list the function returns.
    let code = "Manipulate[\
      Graphics[steps[t, r][[step]], PlotRange -> 2, ImageSize -> {300, 300}], \
      {{t, Pi/2, \"turn\"}, 0, 2 Pi}, \
      {{r, 0.7, \"reach\"}, 0.5, 1.2, Appearance -> \"Labeled\"}, \
      Control[{{step, 2, \"step\"}, {1, 2}}], \
      TrackedSymbols :> {t, r, step}, \
      Initialization :> (steps[t_, r_] := \
        Module[{p, sol, x, y, hits}, \
          p = {Cos[t], Sin[t]}; \
          sol = Quiet[Solve[x^2 + y^2 == 1 && \
            (x - p[[1]])^2 + (y - p[[2]])^2 == r^2, {x, y}]]; \
          hits = {{x, y} /. sol[[1]], {x, y} /. sol[[2]]}; \
          {{Circle[], Circle[p, r]}, \
           {Circle[], Circle[p, r], PointSize[0.03], \
            Point[hits[[1]]], Point[hits[[2]]]}}]), \
      SaveDefinitions -> True]";
    let mut state = instantiate_stored_manipulate(code, "")
      .expect("the compass-construction Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the initial render must draw the construction"
    );

    // Two sliders and the setter bar the construction steps through.
    match &state.controls[..] {
      [
        manipulate::ControlState::Continuous { label: turn, .. },
        manipulate::ControlState::Continuous { label: reach, .. },
        manipulate::ControlState::Discrete {
          values,
          current_index,
          ..
        },
      ] => {
        assert_eq!((turn.as_str(), reach.as_str()), ("turn", "reach"));
        assert_eq!(values, &vec!["1".to_string(), "2".to_string()]);
        assert_eq!(*current_index, 1);
      }
      other => panic!("unexpected controls: {other:?}"),
    }

    // Turning the point re-crosses the circles rather than leaving the
    // intersections unsolved, which would draw no points at all.
    if let manipulate::ControlState::Continuous { current, .. } =
      &mut state.controls[0]
    {
      *current = 2.5;
    }
    state.reevaluate();
    assert!(state.error.is_none(), "re-render failed: {:?}", state.error);
    assert!(state.graphics_handle.is_some());
  }

  #[test]
  fn sphere_cooling_manipulate_builds_widget() {
    // End-to-end regression for a "transient conduction in a sphere"
    // Demonstration: a `Do` loop walks `FindRoot` along the branches of the
    // sphere's eigenvalue equation `1 - x Cot[x] == Bi` to build the root
    // list a `Subscript`-indexed helper then looks up, and the resulting
    // truncated Fourier-Bessel series is plotted against two `Labeled`
    // sliders laid out in a `Row` with a `Spacer` between them.
    let code = "Manipulate[\
      Module[{roots, guess = 1, count = 8}, \
        roots = {}; \
        Do[\
          AppendTo[roots, w /. FindRoot[1 - w Cot[w] == bi, {w, guess}]]; \
          guess = guess + Pi, \
          {n, 1, count}\
        ]; \
        Subscript[eig, k_] := roots[[k]]; \
        Plot[\
          Sum[\
            (4 (Sin[Subscript[eig, k]] - Subscript[eig, k] Cos[Subscript[eig, k]])) / \
              (2 Subscript[eig, k] - Sin[2 Subscript[eig, k]]) * \
              Exp[-Subscript[eig, k]^2 fo], \
            {k, 1, count}\
          ], \
          {fo, 0, fomax}, \
          PlotRange -> {{0, fomax}, {0, 1.05}}, \
          PlotStyle -> {Blue, Thick}, \
          Frame -> True, \
          FrameLabel -> {\"Fourier number\", \"center temperature ratio\"}, \
          GridLines -> Automatic, \
          ImageSize -> {500, 320}\
        ]\
      ], \
      Row[{\
        Control[{{fomax, 0.3, \"Fourier number range\"}, 0.1, 1.0, 0.05, \
          Appearance -> \"Labeled\", ImageSize -> Small}], \
        Spacer[40], \
        Control[{{bi, 5, \"Biot number\"}, 0.5, 20, 1, \
          Appearance -> \"Labeled\", ImageSize -> Small}]\
      }], \
      ControlPlacement -> Top, \
      TrackedSymbols :> {fomax, bi}\
    ]";
    let mut state = instantiate_stored_manipulate(code, "")
      .expect("the sphere-cooling Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the initial render must draw the decay curve"
    );

    // Two labeled continuous sliders, in notebook order, laid out via Row.
    let labels: Vec<(&str, f64, f64)> = state
      .controls
      .iter()
      .map(|c| match c {
        manipulate::ControlState::Continuous {
          label, min, max, ..
        } => (label.as_str(), *min, *max),
        other => panic!("expected a continuous slider, got {other:?}"),
      })
      .collect();
    assert_eq!(
      labels,
      vec![
        ("Fourier number range", 0.1, 1.0),
        ("Biot number", 0.5, 20.0),
      ]
    );

    // Raising the Biot number re-solves the eigenvalue equation and
    // re-renders without error.
    if let manipulate::ControlState::Continuous { current, .. } =
      &mut state.controls[1]
    {
      *current = 15.0;
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

  /// Collect `(action, label)` of every Button in a display tree.
  fn collect_display_buttons(
    trees: &[woxi::functions::graphics::DisplayNode],
  ) -> Vec<(String, String)> {
    use woxi::functions::graphics::DisplayNode;
    fn label_of(node: &DisplayNode) -> String {
      match node {
        DisplayNode::Text { runs } => {
          runs.iter().map(|r| r.text.as_str()).collect()
        }
        DisplayNode::Static { text, .. } => text.clone(),
        _ => String::new(),
      }
    }
    fn walk(node: &DisplayNode, out: &mut Vec<(String, String)>) {
      match node {
        DisplayNode::Button { label, action } => {
          out.push((action.clone(), label_of(label)))
        }
        DisplayNode::Panel(child)
        | DisplayNode::Toggler { label: child, .. } => walk(child, out),
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
        _ => {}
      }
    }
    let mut out = Vec::new();
    for t in trees {
      walk(t, &mut out);
    }
    out
  }

  /// All the text a display tree shows, in reading order.
  fn display_text(trees: &[woxi::functions::graphics::DisplayNode]) -> String {
    use woxi::functions::graphics::DisplayNode;
    fn walk(node: &DisplayNode, out: &mut String) {
      match node {
        DisplayNode::Text { runs } => {
          for r in runs {
            out.push_str(&r.text);
          }
        }
        DisplayNode::Static { text, .. } => out.push_str(text),
        DisplayNode::Panel(child)
        | DisplayNode::Toggler { label: child, .. } => walk(child, out),
        DisplayNode::Button { label, .. } => walk(label, out),
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
        _ => {}
      }
    }
    let mut out = String::new();
    for t in trees {
      walk(t, &mut out);
    }
    out
  }

  /// A Demonstration built the way the Wolfram Demonstrations Project
  /// builds a browse-a-dataset widget: `Initialization` tabulates the
  /// datasets, the body highlights the current one, and a `Dynamic[…]`
  /// caption of buttons — not a slider — steps through them.
  const BROWSE_NETWORKS: &str = "Manipulate[\
    net = names[[k]]; {nv, ne} = sizes[[k]]; \
    g = ExampleData[{\"NetworkGraph\", net}]; \
    HighlightGraph[g, VertexList[g][[1 ;; 3]]], \
    {{k, 1}, {1, 2}, ControlType -> None}, \
    {{net, \"\"}, ControlType -> None}, \
    {{nv, 0}, ControlType -> None}, \
    {{ne, 0}, ControlType -> None}, \
    Dynamic[Row[{Button[\"prev\", If[k == 1, k = 2, k--]], Spacer[10], \
    Button[\"next\", If[k == 2, k = 1, k++]], Spacer[10], \
    \"showing \", Style[net, Bold, Red], \" with \", nv, \" vertices and \", \
    ne, \" edges\"}]], \
    Initialization :> (names = {\"ZacharysKarateClub\", \
    \"DolphinSocialNetwork\"}; \
    sizes = Map[(u = ExampleData[{\"NetworkGraph\", #}]; \
    {VertexCount[u], EdgeCount[u]}) &, names];)]";

  #[test]
  fn demonstration_browse_widget_renders_and_steps() {
    let expr = woxi::interpret_to_expr(BROWSE_NETWORKS).unwrap();
    let mut state = manipulate::ManipulateState::from_expr(&expr).unwrap();

    // The body renders the highlighted graph.
    assert!(state.error.is_none(), "render failed: {:?}", state.error);
    assert!(
      state.graphics_handle.is_some(),
      "the graph should render; text output was {:?}",
      state.text_output
    );

    // The caption reports what the body just computed, not the initial
    // placeholder values.
    let caption = display_text(&state.display_trees);
    assert!(
      caption.contains("ZacharysKarateClub")
        && caption.contains("34")
        && caption.contains("78"),
      "caption should describe the first network: {caption}"
    );

    // Two caption buttons, each carrying its held action.
    let buttons = collect_display_buttons(&state.display_trees);
    assert_eq!(
      buttons,
      vec![
        ("If[k == 1, k = 2, k--]".to_string(), "prev".to_string()),
        ("If[k == 2, k = 1, k++]".to_string(), "next".to_string()),
      ]
    );

    // Pressing "next" steps to the second dataset and re-renders.
    let next = buttons[1].0.clone();
    state.apply_button_action(&next);
    assert!(state.error.is_none(), "re-render failed: {:?}", state.error);
    assert!(state.graphics_handle.is_some());
    let caption = display_text(&state.display_trees);
    assert!(
      caption.contains("DolphinSocialNetwork")
        && caption.contains("62")
        && caption.contains("159"),
      "caption should follow the button press: {caption}"
    );

    // Pressing "next" again wraps back round to the first dataset.
    state.apply_button_action(&next);
    let caption = display_text(&state.display_trees);
    assert!(
      caption.contains("ZacharysKarateClub"),
      "the last dataset should wrap round: {caption}"
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
        DisplayNode::Button { label, .. } => walk(label, out),
        DisplayNode::Checkbox { .. }
        | DisplayNode::Spacer { .. }
        | DisplayNode::Text { .. }
        | DisplayNode::Static { .. } => {}
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
  fn label_char_count_uses_visible_glyphs() {
    // A short styled label counts its rendered glyphs (m₁ = 2), while a
    // suppressed one (`{{θ, 0, ""}, …}`) counts nothing — an unlabelled
    // control already carries its variable name as the label, so an empty
    // label is the author's explicit "no caption". The widest of these
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
      is_real: false,
    };
    let empty = manipulate::ControlState::Continuous {
      name: "theta".to_string(),
      label: String::new(),
      label_runs: vec![],
      min: 0.0,
      max: 1.0,
      step: 0.1,
      current: 0.0,
      is_real: false,
    };
    assert_eq!(manipulate_label_char_count(&m1), 2);
    assert_eq!(manipulate_label_char_count(&empty), 0);
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

  /// The widget shape a plane-geometry Demonstration uses: a `Text @
  /// Column` that stacks the curve's polar equation and its logarithmic
  /// derivative as `TraditionalForm` labels above a `Show` of a
  /// `PolarPlot` and the construction lines, driven by a labeled slider
  /// and a True/False setter under `TrackedSymbols :> {…}`.
  ///
  /// The equations are what makes this shape demanding: `r'` holds as
  /// `Derivative[1][r]` and `1/Cos[t]^2` as `Power[Cos[t], -2]`, neither
  /// of which a front end shows literally.
  #[test]
  fn polar_curve_with_derivative_labels_builds_its_widget() {
    let code = "Manipulate[\n\
Module[{p = Exp[a/2] {Cos[a], Sin[a]}, n},\n\
  n = Normalize[{-Tan[a], 1}];\n\
  Text @ Column[{\n\
    TraditionalForm[r == 1/Cos[t/2]^2],\n\
    TraditionalForm[r' == D[1/Cos[t/2]^2, t]],\n\
    TraditionalForm[r'/r == Simplify[D[1/Cos[t/2]^2, t]/(1/Cos[t/2]^2)]],\n\
    Show[\n\
      PolarPlot[Exp[t/2], {t, 0, 2 Pi}, PlotStyle -> {Blue, Thickness[0.004]}],\n\
      Graphics[{\n\
        If[marks, {Text[\"P\", p, {0, -2 Sign@Last@p}]}, {}],\n\
        Thickness[0.004],\n\
        {Red, Line[{{0, 0}, p}]},\n\
        {Darker@Green, Line[{p, p + len n}]}}],\n\
      ImageSize -> {320, 220}, PlotRange -> {{-8, 8}, {-4, 12}}]},\n\
    Alignment -> Center]],\n\
{{a, Pi/2, \"\\[Theta]\"}, 0, 2 Pi, Appearance -> \"Labeled\"},\n\
{{len, 2, \"length of normal\"}, 0., 3, Appearance -> \"Labeled\"},\n\
{marks, {True, False}},\n\
TrackedSymbols :> {a, len, marks}]";
    let mut state = instantiate_stored_manipulate(code, "")
      .expect("the Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the labeled column and its polar plot must render"
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
    assert_eq!(labels, ["\u{03B8}", "length of normal", "marks"]);
    match &state.controls[0] {
      manipulate::ControlState::Continuous {
        min, max, current, ..
      } => {
        assert!(*min == 0.0 && *max > 6.2, "θ spans a full turn");
        assert!((*current - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
      }
      other => panic!("θ must be a slider: {other:?}"),
    }
    // Dragging the angle past the construction's starting point has to
    // redraw rather than error out.
    if let manipulate::ControlState::Continuous { current, .. } =
      &mut state.controls[0]
    {
      *current = 4.0;
    }
    state.reevaluate();
    assert!(state.error.is_none(), "re-render failed: {:?}", state.error);
    assert!(state.graphics_handle.is_some());
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
    // parenthesised so its coefficient stays a product. The diffusivity is
    // the symbol `\[ScriptCapitalD]`, which is Wolfram's private-use script
    // capital D (U+F773).
    assert!(
      cell.contains("\u{F773}(D[c[x,t], x,x])-u(D[c[x,t], x])"),
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
    assert_eq!(names, ["g", "time", "k", "\u{F773}", "f", "u"]);
  }

  /// End-to-end regression for "Plot a Quadratic Inequality": the region is
  /// named indirectly through a `DynamicModule` local (`p = a x^2 + b x +
  /// c`), which `RegionPlot` — holding its arguments — could not resolve, so
  /// nothing was shaded.
  #[test]
  fn quadratic_inequality_notebook_shades_its_region() {
    let nb_src = r##"Notebook[{
Cell[CellGroupData[{
Cell[BoxData[
 RowBox[{"Manipulate", "[", 
  RowBox[{
   RowBox[{"DynamicModule", "[", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"p", "=", 
       RowBox[{
        RowBox[{"a", " ", 
         RowBox[{"x", "^", "2"}]}], "+", 
        RowBox[{"b", " ", "x"}], "+", "c"}]}], "}"}], ",", 
     "\[IndentingNewLine]", 
     RowBox[{"RegionPlot", "[", 
      RowBox[{
       RowBox[{"If", "[", 
        RowBox[{
         RowBox[{"ineq", "\[Equal]", "\"\<>\>\""}], ",", 
         RowBox[{"y", ">", "p"}], ",", 
         RowBox[{"y", "<", "p"}]}], "]"}], ",", 
       RowBox[{"{", 
        RowBox[{"x", ",", 
         RowBox[{"-", "10"}], ",", "10"}], "}"}], ",", 
       RowBox[{"{", 
        RowBox[{"y", ",", 
         RowBox[{"-", "10"}], ",", "10"}], "}"}], ",", 
       RowBox[{"PlotLabel", "\[Rule]", 
        RowBox[{"Style", "[", 
         RowBox[{
          RowBox[{"Row", "[", 
           RowBox[{"{", 
            RowBox[{
             RowBox[{"Style", "[", 
              RowBox[{"\"\<y\>\"", ",", "Italic"}], "]"}], ",", 
             RowBox[{"If", "[", 
              RowBox[{
               RowBox[{"ineq", "\[Equal]", "\"\<>\>\""}], ",", "\"\< > \>\"", 
               ",", "\"\< < \>\""}], "]"}], ",", 
             RowBox[{"TraditionalForm", "[", "p", "]"}], ",", "\"\<\\n\>\""}],
             "}"}], "]"}], ",", "14"}], "]"}]}], ",", " ", 
       RowBox[{"ImageSize", "\[Rule]", 
        RowBox[{"{", 
         RowBox[{"480", ",", " ", "360"}], "}"}]}], ",", 
       RowBox[{"ImageMargins", "\[Rule]", 
        RowBox[{"{", 
         RowBox[{
          RowBox[{"{", 
           RowBox[{"10", ",", "10"}], "}"}], ",", 
          RowBox[{"{", 
           RowBox[{"0", ",", "0"}], "}"}]}], "}"}]}]}], "]"}]}], "]"}], ",", 
   "\[IndentingNewLine]", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"a", ",", 
       RowBox[{"-", "5"}], ",", "\"\<quadratic coefficient\>\""}], "}"}], ",", 
     RowBox[{"-", "5"}], ",", "5", ",", 
     RowBox[{"Appearance", "\[Rule]", "\"\<Labeled\>\""}]}], "}"}], ",", 
   "\[IndentingNewLine]", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"b", ",", 
       RowBox[{"-", "5"}], ",", "\"\<linear coefficient\>\""}], "}"}], ",", 
     RowBox[{"-", "5"}], ",", "5", ",", 
     RowBox[{"Appearance", "\[Rule]", "\"\<Labeled\>\""}]}], "}"}], ",", 
   "\[IndentingNewLine]", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"c", ",", 
       RowBox[{"-", "5"}], ",", "\"\<constant term\>\""}], "}"}], ",", 
     RowBox[{"-", "5"}], ",", "5", ",", 
     RowBox[{"Appearance", "\[Rule]", "\"\<Labeled\>\""}]}], "}"}], ",", 
   "\[IndentingNewLine]", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"ineq", ",", "\"\<>\>\"", ",", "\"\<choose inequality\>\""}], 
      "}"}], ",", 
     RowBox[{"{", 
      RowBox[{"\"\<>\>\"", ",", "\"\<<\>\""}], "}"}]}], "}"}]}], 
  "]"}]], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`a$$ = -5}, \"\\[Ellipsis]\"]"], "Output"]
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
      "the region must evaluate: {:?}",
      widget.error
    );
    let names: Vec<&str> = widget
      .controls
      .iter()
      .map(|c| match c {
        manipulate::ControlState::Continuous { name, .. } => name.as_str(),
        manipulate::ControlState::Discrete { name, .. } => name.as_str(),
        other => panic!("unexpected control: {other:?}"),
      })
      .collect();
    assert_eq!(names, ["a", "b", "c", "ineq"]);

    let render = |ineq: &str| {
      woxi::interpret_with_stdout(&format!(
        "a = -5; b = -5; c = -5; ineq = \"{ineq}\";\n{}",
        widget.body
      ))
      .expect("the body must render")
      .graphics
      .expect("the body must produce a graphic")
    };
    let above = render(">");
    // The region is shaded, not an empty frame.
    assert!(above.matches("<rect").count() > 100, "region not shaded");
    // Flipping the inequality shades the complement.
    assert_ne!(above, render("<"), "the inequality control must matter");
  }

  /// End-to-end regression for the "Dedekind Cut" Demonstration: circles at
  /// every distinct rational `p/q < 1` for `p, q` up to a bound, coloured by
  /// which side of the cut they fall on, over a disk at the cut radius.
  ///
  /// It already worked; this pins it. The rational set, the cut radius and
  /// the rendered geometry all match wolframscript (checked at four control
  /// settings, agreeing within 0.5% of the frame width).
  #[test]
  fn dedekind_cut_notebook_draws_its_circles() {
    let nb_src = r##"Notebook[{
Cell[CellGroupData[{
Cell[BoxData[
 RowBox[{"Manipulate", "[", 
  RowBox[{
   RowBox[{"With", "[", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"dat", "=", 
       RowBox[{"Union", "[", 
        RowBox[{"Select", "[", 
         RowBox[{
          RowBox[{"Flatten", "[", 
           RowBox[{"Outer", "[", 
            RowBox[{
             RowBox[{
              RowBox[{"Rational", "[", 
               RowBox[{"#1", ",", "#2"}], "]"}], "&"}], ",", 
             RowBox[{"Range", "[", "ra", "]"}], ",", 
             RowBox[{"Range", "[", "ra", "]"}]}], "]"}], "]"}], ",", 
          RowBox[{
           RowBox[{"#", "<", "1"}], "&"}]}], "]"}], "]"}]}], "}"}], ",", 
     RowBox[{"With", "[", 
      RowBox[{
       RowBox[{"{", 
        RowBox[{"cut", "=", 
         RowBox[{"If", "[", 
          RowBox[{"racu", ",", 
           RowBox[{"dat", "[", 
            RowBox[{"[", 
             RowBox[{"1", "+", 
              RowBox[{"Round", "[", 
               RowBox[{"acut", 
                RowBox[{"(", 
                 RowBox[{
                  RowBox[{"Length", "[", "dat", "]"}], "-", "1"}], ")"}]}], 
               "]"}]}], "]"}], "]"}], ",", 
           RowBox[{
            RowBox[{"(", 
             RowBox[{
              RowBox[{"2", "^", 
               RowBox[{"(", 
                RowBox[{"1", "/", "2"}], ")"}]}], "/", "2"}], ")"}], 
            "acut"}]}], "]"}]}], "}"}], ",", 
       RowBox[{"Graphics", "[", 
        RowBox[{
         RowBox[{"{", 
          RowBox[{
           RowBox[{"If", "[", 
            RowBox[{"fli", ",", "White", ",", 
             RowBox[{"RGBColor", "[", 
              RowBox[{"1", ",", ".71", ",", "0"}], "]"}]}], "]"}], ",", 
           RowBox[{"Disk", "[", 
            RowBox[{
             RowBox[{"{", 
              RowBox[{"0", ",", "0"}], "}"}], ",", "cut"}], "]"}], ",", 
           RowBox[{
            RowBox[{
             RowBox[{"{", 
              RowBox[{
               RowBox[{"If", "[", 
                RowBox[{
                 RowBox[{"#", ">", "cut"}], ",", 
                 RowBox[{"RGBColor", "[", 
                  RowBox[{".67", ",", ".75", ",", ".15"}], "]"}], ",", 
                 RowBox[{"If", "[", 
                  RowBox[{
                   RowBox[{"cut", "\[Equal]", "#"}], ",", "Red", ",", 
                   RowBox[{"RGBColor", "[", 
                    RowBox[{".12", ",", ".61", ",", ".78"}], "]"}]}], "]"}]}],
                 "]"}], ",", 
               RowBox[{"Circle", "[", 
                RowBox[{
                 RowBox[{"{", 
                  RowBox[{"0", ",", "0"}], "}"}], ",", "#"}], "]"}]}], "}"}], 
             "&"}], "/@", "dat"}]}], "}"}], ",", 
         RowBox[{"PlotRange", "\[Rule]", "1"}], ",", 
         RowBox[{"Background", "\[Rule]", 
          RowBox[{"If", "[", 
           RowBox[{"fli", ",", 
            RowBox[{"RGBColor", "[", 
             RowBox[{"1", ",", ".71", ",", "0"}], "]"}], ",", "White"}], 
           "]"}]}], ",", 
         RowBox[{"ImageSize", "\[Rule]", 
          RowBox[{"{", 
           RowBox[{"350", ",", "350"}], "}"}]}]}], "]"}]}], "]"}]}], "]"}], 
   ",", "\[IndentingNewLine]", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"acut", ",", ".5", ",", "\"\<cut\>\""}], "}"}], ",", "0", ",", 
     "1", ",", 
     RowBox[{"Appearance", "\[Rule]", "\"\<Labeled\>\""}]}], "}"}], ",", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"racu", ",", "False", ",", "\"\<kind of cut\>\""}], "}"}], ",", 
     
     RowBox[{"{", 
      RowBox[{
       RowBox[{"False", "\[Rule]", "\"\<irrational\>\""}], ",", 
       RowBox[{"True", "\[Rule]", "\"\<rational\>\""}]}], "}"}]}], "}"}], ",",
    "\[IndentingNewLine]", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"fli", ",", "False", ",", "\"\<mark radii\>\""}], "}"}], ",", 
     RowBox[{"{", 
      RowBox[{
       RowBox[{"False", "\[Rule]", "\"\<smaller than cut\>\""}], ",", 
       RowBox[{"True", "\[Rule]", "\"\<larger than cut\>\""}]}], "}"}]}], 
    "}"}], ",", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{
      "ra", ",", "6", ",", "\"\<range of numerator and denominator\>\""}], 
      "}"}], ",", "2", ",", "16"}], "}"}], ",", 
   RowBox[{"Alignment", "\[Rule]", "Center"}], ",", 
   RowBox[{"AutorunSequencing", "\[Rule]", 
    RowBox[{"{", 
     RowBox[{"2", ",", "3", ",", "4"}], "}"}]}]}], "]"}]], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`acut$$ = 0.5}, \"\\[Ellipsis]\"]"], "Output"]
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
      "the cut must evaluate: {:?}",
      widget.error
    );
    assert!(widget.graphics_handle.is_some(), "the circles must draw");
    let names: Vec<&str> = widget
      .controls
      .iter()
      .map(|c| match c {
        manipulate::ControlState::Continuous { name, .. } => name.as_str(),
        manipulate::ControlState::Discrete { name, .. } => name.as_str(),
        other => panic!("unexpected control: {other:?}"),
      })
      .collect();
    assert_eq!(names, ["acut", "racu", "fli", "ra"]);

    let render = |racu: &str, fli: &str, ra: u32| {
      woxi::interpret_with_stdout(&format!(
        "acut = 0.5; racu = {racu}; fli = {fli}; ra = {ra};\n{}",
        widget.body
      ))
      .expect("the body must render")
      .graphics
      .expect("the body must produce a graphic")
    };
    // Eleven rationals below 1 with numerator and denominator up to 6, each
    // a circle, plus the disk at the cut.
    let plain = render("False", "False", 6);
    assert_eq!(plain.matches("<ellipse").count(), 12, "{plain}");
    // A rational cut lands on one of them, which turns red.
    assert!(!plain.contains("rgb(255,0,0)"), "{plain}");
    assert!(render("True", "False", 6).contains("rgb(255,0,0)"));
    // `mark radii` swaps the background and the disk.
    assert!(render("False", "True", 6).contains("fill=\"rgb(255,181,0)\""));
  }

  /// End-to-end regression for "Binomial Probability Distribution": a
  /// stem plot of the binomial PDF, titled and with both axes labelled.
  /// The unjoined `ListPlot` path drew no labels at all.
  #[test]
  fn binomial_distribution_notebook_labels_its_plot() {
    let nb_src = r##"Notebook[{
Cell[CellGroupData[{
Cell[BoxData[
 RowBox[{"Manipulate", "[", "\[IndentingNewLine]", 
  RowBox[{
   RowBox[{"ListPlot", "[", 
    RowBox[{
     RowBox[{"Table", "[", 
      RowBox[{
       RowBox[{"{", 
        RowBox[{"k", ",", 
         RowBox[{"PDF", "[", 
          RowBox[{
           RowBox[{"BinomialDistribution", "[", 
            RowBox[{"NumTrials", ",", "ProbSuccess"}], "]"}], ",", "k"}], 
          "]"}]}], "}"}], ",", 
       RowBox[{"{", 
        RowBox[{"k", ",", "0", ",", "NumTrials"}], "}"}]}], "]"}], ",", 
     RowBox[{"AxesOrigin", "\[Rule]", 
      RowBox[{"{", 
       RowBox[{"0", ",", "0"}], "}"}]}], ",", 
     RowBox[{"Filling", "\[Rule]", "Axis"}], ",", 
     RowBox[{"AxesLabel", "\[Rule]", 
      RowBox[{"{", 
       RowBox[{"x", ",", "\"\<\!\(\*SubscriptBox[\(f\), \(X\)]\)(x)\>\""}], 
       "}"}]}], ",", 
     RowBox[{"PlotRange", "\[Rule]", 
      RowBox[{"{", 
       RowBox[{"0", ",", "1"}], "}"}]}], ",", 
     RowBox[{"ImageSize", "\[Rule]", 
      RowBox[{"{", 
       RowBox[{"500", ",", "300"}], "}"}]}], ",", " ", 
     RowBox[{"PlotStyle", "\[Rule]", 
      RowBox[{"{", 
       RowBox[{"Blue", ",", 
        RowBox[{"PointSize", "[", "Medium", "]"}]}], "}"}]}], ",", " ", 
     RowBox[{"PlotLabel", "\[Rule]", 
      RowBox[{"Style", "[", 
       RowBox[{
       "\"\<probability associated with each value of the random \
variable\>\"", ",", "Purple", ",", "Bold", ",", "12"}], "]"}]}]}], "]"}], ",", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"NumTrials", ",", "10", ",", "\"\<number of trials\>\""}], 
      "}"}], ",", "1", ",", " ", "50", ",", "1", ",", 
     RowBox[{"Appearance", "\[Rule]", "\"\<Labeled\>\""}]}], "}"}], ",", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{
      "ProbSuccess", ",", ".25", ",", "\"\<probability of success\>\""}], 
      "}"}], ",", "0.01", ",", "1", ",", 
     RowBox[{"Appearance", "\[Rule]", "\"\<Labeled\>\""}]}], "}"}]}], 
  "]"}]], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`NumTrials$$ = 10}, \"\\[Ellipsis]\"]"], "Output"]
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
      "the distribution must evaluate: {:?}",
      widget.error
    );
    assert!(widget.graphics_handle.is_some(), "the stems must draw");
    let names: Vec<&str> = widget
      .controls
      .iter()
      .map(|c| match c {
        manipulate::ControlState::Continuous { name, .. } => name.as_str(),
        other => panic!("unexpected control: {other:?}"),
      })
      .collect();
    assert_eq!(names, ["NumTrials", "ProbSuccess"]);

    let svg = woxi::interpret_with_stdout(&format!(
      "NumTrials = 10; ProbSuccess = 0.25;\n{}",
      widget.body
    ))
    .expect("the body must render")
    .graphics
    .expect("the body must produce a graphic");
    // One point per k in 0..NumTrials.
    assert_eq!(svg.matches("<circle").count(), 11, "{svg}");
    // The title and both axis labels are drawn; the y label keeps the
    // subscript its inline box spells out.
    assert!(
      svg.contains("probability associated with each value"),
      "{svg}"
    );
    assert!(svg.contains(">x</text>"), "{svg}");
    assert!(
      svg.contains("baseline-shift=\"sub\""),
      "the y label's subscript must typeset: {svg}"
    );
  }

  /// End-to-end regression for "Dynamics of a Spring-Pendulum System". Four
  /// of its parameters are held by `{{x, v}, None}` controls, its motion
  /// comes from an `NDSolveValue` whose equations include an algebraic
  /// constraint tying the string lengths together, and its spring is a
  /// plotted sine curve reused as a shape through `First[Plot[…]]`.
  /// (The notebook's energy gauge and its two extra panels are trimmed
  /// here; the mechanics are its own.)
  #[test]
  fn spring_pendulum_notebook_solves_and_draws() {
    let nb_src = r##"Notebook[{
Cell[BoxData["coil[pt1:{x1_,y1_},pt2:{x2_,y2_},n_,a_,th_]:=Module[{s,al},s=EuclideanDistance[pt1,pt2];al=ArcTan@@(pt1-pt2);Translate[Rotate[Scale[First[Plot[a Sin[n t],{t,-Pi,Pi},PlotPoints->30,Axes->False]],{s/2/Pi,1}],al,{0,0}],pt1+-s/2{Cos[al],Sin[al]}]]"], "Input"],
Cell[CellGroupData[{
Cell[BoxData["Manipulate[\nModule[{g=9.81,sol,thsol,ysol,lsol},\nsol=NDSolveValue[{k s0+g m Cos[th[t]]+(-d+L) m (th'[t])^2+y[t] (-k+m (th'[t])^2)==m y''[t],\ng Sin[th[t]]+2. y'[t] th'[t]+l[t] th''[t]==0,\nl[t]+d-y[t]==L,\nth[0]==th0,th'[0]==0,y[0]==y0,y'[0]==0},{th,y,l},{t,0,tMax}];\n{thsol,ysol,lsol}=sol;\nGraphics[{AbsoluteThickness[2],\nLine[{lsol[time]{Sin[thsol[time]],-Cos[thsol[time]]},{0,0},{-1,0},{-1,ysol[time]}}],\nGray,Disk[lsol[time]{Sin[thsol[time]],-Cos[thsol[time]]},.1 m^(1/3)],\nBlue,coil[{-1.,-2},{-1,ysol[time]},12,.05,.35]},\nPlotRange->{{-1.5,1.5},{-2.05,.05}},ImageSize->300]],\n{{tMax,20},None},{{L,3.25,\"total string length\"},None},{{d,1,\"pivot spacing\"},None},\n{{s0,-1.125,\"spring stop\"},None},\n{{m,1,\"mass\"},.25,2.5,.001,Appearance->\"Labeled\"},\n{{th0,1.,\"initial angle\"},-1.25,1.25,.001,Appearance->\"Labeled\"},\n{{y0,-1.,\"spring start\"},-1.125,-.75,.001,Appearance->\"Labeled\"},\n{{k,150,\"spring constant\"},100,200,.1,Appearance->\"Labeled\"},\n{{time,0,\"time\"},0,tMax,.05,Appearance->\"Labeled\"},\nSaveDefinitions->True]"], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`m$$ = 1}, \"\\[Ellipsis]\"]"], "Output"]
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
      "the constrained system must solve: {:?}",
      widget.error
    );
    assert!(widget.graphics_handle.is_some(), "the pendulum must draw");
    // The four `None`-domain parameters are bound but get no widget row.
    let names: Vec<&str> = widget
      .controls
      .iter()
      .map(|c| match c {
        manipulate::ControlState::Continuous { name, .. } => name.as_str(),
        other => panic!("unexpected control: {other:?}"),
      })
      .collect();
    assert_eq!(names, ["m", "th0", "y0", "k", "time"]);
    let hidden: Vec<&str> =
      widget.state.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(hidden, ["tMax", "L", "d", "s0"]);

    let render = |time: &str| {
      let bindings: String = widget
        .state
        .iter()
        .map(|(n, v)| format!("{n} = {v};\n"))
        .collect();
      woxi::interpret_with_stdout(&format!(
        "{bindings}m = 1; th0 = 1.; y0 = -1.; k = 150; time = {time};\n{}",
        widget.body
      ))
      .expect("the body must render")
      .graphics
      .expect("the body must produce a graphic")
    };
    let at0 = render("0");
    // The string, the bob and the coil spring are all drawn.
    assert!(at0.contains("<polyline") || at0.contains("<path"), "{at0}");
    assert!(at0.contains("<ellipse") || at0.contains("<circle"), "{at0}");
    // The system evolves: the picture at a later time differs.
    assert_ne!(at0, render("3"), "the time control must move the pendulum");
  }

  /// End-to-end regression for a nonlinear-heat-conduction-style
  /// Demonstration built on orthogonal collocation: one node of an indexed
  /// family of functions is pinned to a fixed boundary value with
  /// `Y[n][t_] := 1` *after* the other equations already captured a
  /// literal, still-unresolved `Y[n][t]` call inside them (`n` bound to the
  /// Manipulate control's numeric value). That pattern used to be slow
  /// enough to look like a hang: the boundary reference sat outside
  /// `NDSolve`'s own dependent variables, so the whole right-hand side fell
  /// back to per-step symbolic evaluation for every one of the fixed grid's
  /// 1000 steps.
  #[test]
  fn pinned_boundary_node_notebook_solves_without_hanging() {
    let nb_src = r##"Notebook[{
Cell[CellGroupData[{
Cell[BoxData["Manipulate[Module[{Y,eq,sol},eq=Y[1]'[t]==Y[n][t]-Y[1][t];Y[n][t_]:=1;sol=NDSolve[{eq,Y[1][0]==0},Y[1],{t,0,2}];Plot[Y[1][t]/.sol[[1]],{t,0,2}]],{{n,4,\"boundary index\"},3,6,1}]"], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`n$$ = 4}, \"\\[Ellipsis]\"]"], "Output"]
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
      "the pinned-boundary system must solve: {:?}",
      widget.error
    );
    assert!(widget.graphics_handle.is_some(), "the curve must draw");
    match &widget.controls[..] {
      [manipulate::ControlState::Continuous { name, current, .. }] => {
        assert_eq!((name.as_str(), *current), ("n", 4.0));
      }
      other => panic!("unexpected controls: {other:?}"),
    }
  }

  /// End-to-end regression for "Nets for Regular Spherical Models": its
  /// body is a `Pane` around either a 3D model or a 2D net template, and
  /// the export path used to write the whole thing out as source rather
  /// than unwrapping the Pane.
  #[test]
  fn spherical_nets_notebook_draws_through_its_pane() {
    let nb_src = r##"Notebook[{
Cell[BoxData["ort[v_]:=Normalize[v];\ngreatcirclearc[a_,b_]:=Module[{au=ort[a//N],n1,n2,t,alpha,lis,alp},n1=Cross[a,b];\nn2=ort[Cross[n1,a]];alpha=ArcCos[ort[a].ort[b]];alp=IntegerPart[45.*alpha];lis=Table[Cos[t/45.] au+Sin[t/45.] n2,{t,0,alp}];Join[lis,{Cos[alpha] au+Sin[alpha] n2}]];\nClosedLine[a_]:=Line[Append[a,First[a]]];\nRot[om_,al_][x_]:=(x.om)om+(x-(x.om)om)Cos[al]+Cross[om,x]Sin[al];\noposite[r1_,r2_]:=Norm[r1+r2]<0.001;\neliminateoposite[z_List]:=Module[{result={First[z]}},Do[notop=True;Do[If[oposite[z[[i]],result[[j]]],notop=False],{j,1,Length[result]}];If[notop,AppendTo[result,z[[i]]]],{i,2,Length[z]}];result];\neverag[z_List]:=Apply[Plus,z]/Length[z];\nsred[ver_,faces_]:=Map[everag,Map[ver[[#]]&,faces]];\nmidleedge[ver_,edges_]:=Map[everag,Map[ver[[#]]&,edges]];\nedg2=PolyhedronData[\"Dodecahedron\",\"EdgeIndices\"];\nfac2=PolyhedronData[\"Dodecahedron\",\"FaceIndices\"];\nve2=PolyhedronData[\"Dodecahedron\",\"VertexCoordinates\"]//N;\nmidleed2=Map[everag,Map[ve2[[#]]&,edg2]];\nmidleedge2=eliminateoposite[midleed2];\nsr2=sred[ve2,fac2];\nsred2=eliminateoposite[sr2];\nver2=eliminateoposite[ve2];\nort2=Normalize[Cross[ve2[[1]],sr2[[1]]]];\nikozanumer=Join[{Rot[{0,0,1},0]},Flatten[Table[Rot[Normalize[ver2[[i]]]//N,j 2 Pi/3//N],{j,1,2},{i,1,Length[ver2]}]],Flatten[Table[Rot[Normalize[sred2[[i]]]//N,j 2 Pi/5//N],{j,1,4},{i,1,Length[sred2]}]],Flatten[Table[Rot[Normalize[midleedge2[[i]]]//N,j  Pi//N],{j,1,1},{i,1,Length[midleedge2]}]]];\ndodecahedron=With[{solid=GraphicsComplex[PolyhedronData[\"Dodecahedron\",\"VertexCoordinates\"],Polygon[PolyhedronData[\"Dodecahedron\",\"FaceIndices\"]]]},{solid[[2,1]],Map[Normalize,solid[[1]]//N]}];\nmoeb1=Join[greatcirclearc[dodecahedron[[2,1]],Normalize[sr2[[1]]]],greatcirclearc[Normalize[sr2[[1]]],Normalize[midleed2[[1]]]],greatcirclearc[Normalize[midleed2[[1]]],dodecahedron[[2,1]]]];\nmoeb1SM={With[{up=greatcirclearc[dodecahedron[[2,1]],Normalize[sr2[[1]]]]},Join[Reverse[up],0.8 up]],With[{up=greatcirclearc[Normalize[sr2[[1]]],Normalize[midleed2[[1]]]]},Join[Reverse[up],0.8 up]],With[{up=greatcirclearc[Normalize[midleed2[[1]]],dodecahedron[[2,1]]]},Join[Reverse[up],0.8 up]]};\nhiI=ArcCos[dodecahedron[[2,1]].Normalize[sr2[[1]]]];\nfiI=ArcCos[Normalize[sr2[[1]]].Normalize[midleed2[[1]]]];\npsiI=ArcCos[Normalize[midleed2[[1]]].dodecahedron[[2,1]]];\ngrd2b=Table[Map[ikozanumer[[i]],moeb1SM,{2}],{i,1,Length[ikozanumer]}];\ntetrahedron=With[{solid=GraphicsComplex[PolyhedronData[\"Tetrahedron\",\"VertexCoordinates\"],Polygon[PolyhedronData[\"Tetrahedron\",\"FaceIndices\"]]]},{solid[[2,1]],Map[Normalize,solid[[1]]//N]}];\nmoebt={With[{up=greatcirclearc[Normalize[(tetrahedron[[2,2]]+tetrahedron[[2,3]]+tetrahedron[[2,4]])/3],tetrahedron[[2,2]]]},Join[Reverse[up],0.8 up]],With[{up=greatcirclearc[tetrahedron[[2,2]],Normalize[(tetrahedron[[2,2]]+tetrahedron[[2,3]])/2]]},Join[Reverse[up],0.8 up]],With[{up=greatcirclearc[Normalize[(tetrahedron[[2,2]]+tetrahedron[[2,3]])/2],Normalize[(tetrahedron[[2,2]]+tetrahedron[[2,3]]+tetrahedron[[2,4]])/3]]},Join[Reverse[up],0.8 up]]};\npsi3=ArcCos[tetrahedron[[2,2]].Normalize[(tetrahedron[[2,2]]+tetrahedron[[2,3]])/2]];\nfi3=ArcCos[Normalize[(tetrahedron[[2,2]]+tetrahedron[[2,3]])/2].Normalize[(tetrahedron[[2,2]]+tetrahedron[[2,3]]+tetrahedron[[2,4]])/3]];\nhi3=ArcCos[Normalize[(tetrahedron[[2,2]]+tetrahedron[[2,3]]+tetrahedron[[2,4]])/3].tetrahedron[[2,2]]];\nedg3=PolyhedronData[\"Tetrahedron\",\"EdgeIndices\"];\nfac3=PolyhedronData[\"Tetrahedron\",\"FaceIndices\"];\nve3=PolyhedronData[\"Tetrahedron\",\"VertexCoordinates\"]//N;\nmidleed3=Map[everag,Map[ve3[[#]]&,edg3]];\nmidleedge3=eliminateoposite[midleed3];\nsr3=sred[ve3,fac3];\nsred3=eliminateoposite[sr3];\nver3=eliminateoposite[ve3];\ntetrahedralnumer=Join[{Rot[{0,0,1},0]},Flatten[Table[Rot[Normalize[ver3[[i]]]//N,j 2 Pi/3//N],{j,1,2},{i,1,Length[ver3]}]],Flatten[Table[Rot[Normalize[midleedge3[[i]]]//N,j  Pi//N],{j,1,1},{i,1,Length[midleedge3]}]]];\ngrdt=Table[Map[tetrahedralnumer[[i]],moebt,{2}],{i,1,Length[tetrahedralnumer]}];\nthird[n_]:={Cos[Pi/n],Sin[Pi/n],0};\nmoebD[n_]:=Join[greatcirclearc[{0,0,1},{1,0,0}],greatcirclearc[{1,0,0},third[n]],greatcirclearc[third[n],{0,0,1}]];\ncube=With[{solid=GraphicsComplex[PolyhedronData[\"Cube\",\"VertexCoordinates\"],Polygon[PolyhedronData[\"Cube\",\"FaceIndices\"]]]},{solid[[2,1]],Map[Normalize,solid[[1]]//N]}];\nedg1=PolyhedronData[\"Cube\",\"EdgeIndices\"];\nfac1=PolyhedronData[\"Cube\",\"FaceIndices\"];\nve1=PolyhedronData[\"Cube\",\"VertexCoordinates\"]//N;\nmidleed1=Map[everag,Map[ve1[[#]]&,edg1]];\nmidleedge1=eliminateoposite[midleed1];\nsr1=sred[ve1,fac1];\nsred1=eliminateoposite[sr1];\nver1=eliminateoposite[ve1];\noctahedralnumer=Join[{Rot[{0,0,1},0]},Flatten[Table[Rot[Normalize[ver1[[i]]]//N,j 2 Pi/3//N],{j,1,2},{i,1,Length[ver1]}]],Flatten[Table[Rot[Normalize[sred1[[i]]]//N,j 2 Pi/4//N],{j,1,3},{i,1,Length[sred1]}]],Flatten[Table[Rot[Normalize[midleedge1[[i]]]//N,j  Pi//N],{j,1,1},{i,1,Length[midleedge1]}]]];\nmoebc={With[{up=greatcirclearc[cube[[2,2]],Normalize[cube[[2,2]]+cube[[2,8]]]]},Join[Reverse[up],0.8 up]],With[{up=greatcirclearc[Normalize[cube[[2,2]]+cube[[2,8]]],Normalize[cube[[2,2]]+cube[[2,4]]]]},Join[Reverse[up],0.8 up]],With[{up=greatcirclearc[Normalize[cube[[2,2]]+cube[[2,4]]],cube[[2,2]]]},Join[Reverse[up],0.8 up]]};\ngrdc=Table[Map[octahedralnumer[[i]],moebc,{2}],{i,1,Length[octahedralnumer]}];\ngrdcs={grdt,grdc,grd2b};\nhi1=ArcCos[Normalize[cube[[2,2]]].Normalize[cube[[2,2]]+cube[[2,8]]]];\nfi1=ArcCos[Normalize[cube[[2,2]]+cube[[2,8]]].Normalize[cube[[2,2]]+cube[[2,4]]]];\npsi1=ArcCos[Normalize[cube[[2,2]]+cube[[2,4]]].Normalize[cube[[2,2]]]];\ngraf31=Graphics[{With[{gr1={Circle[{0,0},1,{0,hi1+fi1+psi1}],Circle[{0,0},0.8,{0,hi1+fi1+psi1}],Line[{{0.8,0},{1,0}}],Line[{0.8{Cos[hi1],Sin[hi1]},{Cos[hi1],Sin[hi1]}}],Line[{0.8{Cos[hi1+fi1],Sin[hi1+fi1]},{Cos[hi1+fi1],Sin[hi1+fi1]}}],Line[{0.8{Cos[hi1+fi1+psi1],Sin[hi1+fi1+psi1]},{Cos[hi1+fi1+psi1],Sin[hi1+fi1+psi1]}}]}},{Rotate[gr1,(Pi-(hi1+fi1+psi1))/2,{0,0}],Translate[Rotate[gr1,Pi+(Pi-(hi1+fi1+psi1))/2,{0,0}],{0.8,1.}]}]},PlotRange->{{-1.,1.77},{-.1,1.1}}];\ngraf32=Graphics[{With[{gr1={Circle[{0,0},1,{0,4hi1}],Circle[{0,0},0.8,{0,4hi1}],Line[{{0.8,0},{1,0}}],Table[Line[{0.8{Cos[i hi1],Sin[i hi1]},{Cos[i hi1],Sin[i hi1]}}],{i,0,4}]}},{Rotate[gr1,(Pi-(4 hi1))/2,{0,0}],Translate[Rotate[gr1,Pi+(Pi-4(hi1))/2,{0,0}],{0.8,0.4}]}]},PlotRange->{{-1.1,1.88},{-.7.9,1.1}}];\ngraf33=Graphics[{With[{gr1={Circle[{0,0},1,{0,6fi1}],Circle[{0,0},0.8,{0,6fi1}],Line[{{0.8,0},{1,0}}],Table[Line[{0.8{Cos[i 2fi1],Sin[i 2fi1]},{Cos[i 2fi1],Sin[i 2fi1]}}],{i,0,3}]}},{Rotate[gr1,(Pi-(6 fi1))/2,{0,0}],Translate[Rotate[gr1,Pi+(Pi-6(fi1))/2,{0,0}],{0.6,0.}]}]},PlotRange->{{-1.1,1.7},{-1.1,1.1}}];\ngraf34=Graphics[{With[{gr1={Circle[{0,0},1,{0,8psi1}],Circle[{0,0},0.8,{0,8psi1}],Line[{{0.8,0},{1,0}}],Table[Line[{0.8{Cos[i 2 psi1],Sin[i 2psi1]},{Cos[i 2psi1],Sin[i 2psi1]}}],{i,0,4}]}},{Rotate[gr1,(Pi-(8 psi1))/2,{0,0}],Translate[Rotate[gr1,Pi+(Pi-8(psi1))/2,{0,0}],{0.7,0.}]}]},PlotRange->{{-1.1,1.8},{-1.1,1.1}}];\ngraf11=Graphics[With[{gr1={Circle[{0,0},1,{0,hi3+fi3+psi3}],Circle[{0,0},0.8,{0,hi3+fi3+psi3}],Line[{{0.8,0},{1,0}}],Line[{0.8{Cos[hi3],Sin[hi3]},{Cos[hi3],Sin[hi3]}}],Line[{0.8{Cos[hi3+fi3],Sin[hi3+fi3]},{Cos[hi3+fi3],Sin[hi3+fi3]}}],Line[{0.8{Cos[hi3+fi3+psi3],Sin[hi3+fi3+psi3]},{Cos[hi3+fi3+psi3],Sin[hi3+fi3+psi3]}}]}},{Rotate[gr1,(Pi-(hi3+fi3+psi3))/2,{0,0}],Translate[Rotate[gr1,Pi+(Pi-(hi3+fi3+psi3))/2,{0,0}],{0.8,0.7}]}],PlotRange->{{-1.1,1.85},{-0.4,1.1}}];\ngraf12=Graphics[With[{gr1={Circle[{0,0},1,{0,4hi3}],Circle[{0,0},0.8,{0,4hi3}],Line[{{0.8,0},{1,0}}],Table[Line[{0.8{Cos[i hi3],Sin[i hi3]},{Cos[i hi3],Sin[i hi3]}}],{i,0,4}]}},{Rotate[gr1,(Pi-(4 hi3))/2,{0,0}],Translate[Rotate[gr1,Pi+(Pi-4(hi3))/2,{0,0}],{0.55,-0.04}]}],PlotRange->{{-1.1,1.7},{-1.1,1.1}}];\ngraf13=Graphics[With[{gr1={Circle[{0,0},1,{0,6fi3}],Circle[{0,0},0.8,{0,6fi3}],Line[{{0.8,0},{1,0}}],Table[Line[{0.8{Cos[i 2fi3],Sin[i 2fi3]},{Cos[i 2fi3],Sin[i 2fi3]}}],{i,0,3}]}},{Rotate[gr1,(Pi-(6 fi3))/2,{0,0}],Translate[Rotate[gr1,Pi+(Pi-6(fi3))/2,{0,0}],{0.654,-0.27}]}],PlotRange->{{-1.1,1.8},{-1.3,1.1}}];\ngraf14=Graphics[With[{gr1={Circle[{0,0},1,{0,6psi3}],Circle[{0,0},0.8,{0,6psi3}],Line[{{0.8,0},{1,0}}],Table[Line[{0.8{Cos[i 2psi3],Sin[i 2psi3]},{Cos[i 2psi3],Sin[i 2psi3]}}],{i,0,3}]}},{Rotate[gr1,(Pi-(6 psi3))/2,{0,0}],Translate[Rotate[gr1,Pi+(Pi-6(psi3))/2,{0,0}],{0.654,-0.27}]}],PlotRange->{{-1.1,1.8},{-1.3,1.1}}];\ngraf21=Graphics[With[{gr1={Circle[{0,0},1,{0,hiI+fiI+psiI}],Circle[{0,0},0.8,{0,hiI+fiI+psiI}],Line[{{0.8,0},{1,0}}],Line[{0.8{Cos[hiI],Sin[hiI]},{Cos[hiI],Sin[hiI]}}],Line[{0.8{Cos[hiI+fiI],Sin[hiI+fiI]},{Cos[hiI+fiI],Sin[hiI+fiI]}}],Line[{0.8{Cos[hiI+fiI+psiI],Sin[hiI+fiI+psiI]},{Cos[hiI+fiI+psiI],Sin[hiI+fiI+psiI]}}]}},{Rotate[gr1,(Pi-(hiI+fiI+psiI))/2,{0,0}],Translate[Rotate[gr1,Pi+(Pi-(hiI+fiI+psiI))/2,{0,0}],{0.6,1.3}]}],PlotRange->{{-0.8,1.4},{0.23,1.1}}];\ngraf22=Graphics[{With[{gr1={Circle[{0,0},1,{0,4hiI}],Circle[{0,0},0.8,{0,4hiI}],Line[{{0.8,0},{1,0}}],Table[Line[{0.8{Cos[i hiI],Sin[i hiI]},{Cos[i hiI],Sin[i hiI]}}],{i,0,4}]}},{Rotate[gr1,(Pi-(4 hiI))/2,{0,0}],Translate[Rotate[gr1,Pi+(Pi-4(hiI))/2,{0,0}],{0.8,0.95}]}]},PlotRange->{{-1.,1.8},{-.1,1.08}}];\ngraf23=Graphics[With[{gr1={Circle[{0,0},1,{0,6fiI}],Circle[{0,0},0.8,{0,6fiI}],Line[{{0.8,0},{1,0}}],Table[Line[{0.8{Cos[i 2fiI],Sin[i 2fiI]},{Cos[i 2fiI],Sin[i 2fiI]}}],{i,0,3}]}},{Rotate[gr1,(Pi-(6 fiI))/2,{0,0}],Translate[Rotate[gr1,Pi+(Pi-6(fiI))/2,{0,0}],{0.9,.65}]}],PlotRange->{{-1.1,1.95},{-0.4,1.1}}];\ngraf24=Graphics[With[{gr1={Circle[{0,0},1,{0,10psiI}],Circle[{0,0},0.8,{0,10psiI}],Line[{{0.8,0},{1,0}}],Table[Line[{0.8{Cos[i 2psiI],Sin[i 2psiI]},{Cos[i 2psiI],Sin[i 2psiI]}}],{i,0,5}]}},{Rotate[gr1,(Pi-(10 psiI))/2,{0,0}],Translate[Rotate[gr1,Pi+(Pi-10(psiI))/2,{0,0}],{0.84,0.47}]}],PlotRange->{{-1.1,1.9},{-.6,1.08}}];\nnets={{graf11,graf12,graf13,graf14},{graf31,graf32,graf33,graf34},{graf21,graf22,graf23,graf24}};"], "Input"],
Cell[CellGroupData[{
Cell[BoxData["Manipulate[Pane[Switch[nt, 2, Show[nets[[fam, 1]], ImageSize -> 300], 1, Graphics3D[{Red, EdgeForm[], Map[Polygon, grdcs[[fam]]]}, Boxed -> False, PlotRange -> 1.2, ImageSize -> 300]], Alignment -> Center, ImageSize -> 320],\n{{fam, 1, \"family\"}, {1 -> \"tetrahedral\", 2 -> \"octahedral\", 3 -> \"icosahedral\"}},\n{{nt, 2, \"view\"}, {1 -> \"solid\", 2 -> \"nets templates\"}},\nSaveDefinitions -> True]"], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`fam$$ = 1}, \"\\[Ellipsis]\"]"], "Output"]
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
      "the nets must build: {:?}",
      widget.error
    );
    assert!(widget.graphics_handle.is_some(), "the net must draw");
    let names: Vec<&str> = widget
      .controls
      .iter()
      .map(|c| match c {
        manipulate::ControlState::Discrete { name, .. } => name.as_str(),
        other => panic!("unexpected control: {other:?}"),
      })
      .collect();
    assert_eq!(names, ["fam", "nt"]);

    let render = |fam: u32, nt: u32| {
      woxi::interpret_with_stdout(&format!(
        "fam = {fam}; nt = {nt};\n{}",
        widget.body
      ))
      .expect("the body must render")
      .graphics
      .expect("the body must produce a graphic")
    };
    // The net template is drawn, not echoed as `Pane[GraphicsBox[]]`.
    let net = render(1, 2);
    assert!(!net.contains("GraphicsBox"), "{net}");
    assert!(net.contains("<path") || net.contains("<polyline"), "{net}");
    // Each family has its own net, and the solid view differs from it.
    assert_ne!(net, render(3, 2), "the family control must matter");
    assert_ne!(net, render(1, 1), "the view control must matter");
  }

  /// End-to-end regression for the "Chaos and Order in the Damped Forced
  /// Pendulum in a Plane" Demonstration: it integrates the damped driven
  /// pendulum `θ'' == -(g/l) Sin[θ] - γ θ' + a Cos[ω t]` from a grid of
  /// initial conditions and draws every trajectory in (θ', t, θ) space.
  ///
  /// The notebook is trimmed the way `tests/notebooks` trims the others —
  /// the author's comments are dropped and the published default of the
  /// "number of trajectories" picklist is its first choice (1) instead of
  /// 16, so the test integrates one trajectory rather than sixteen. Every
  /// structure the Demonstration relies on is kept: the typeset second
  /// derivative in the differential equation, the derivative of the
  /// InterpolatingFunction a part-extracted solution rule carries, and the
  /// `Show[Graphics3D[…], Axes -> True, PlotRange -> {…}, AxesLabel -> {…}]`
  /// frame.
  #[test]
  fn damped_forced_pendulum_notebook_draws_its_trajectories() {
    let nb_src = r##"Notebook[{
Cell[CellGroupData[{
Cell["Initialization Code", "Section"],
Cell[BoxData[
 RowBox[{
  RowBox[{"DFPPlot", "[", 
   RowBox[{
   "tmax_", ",", "ntray_", ",", "long_", ",", "mass_", ",", "grav_", ",", 
    "resis_", ",", "f0ext_", ",", "wext_"}], "]"}], ":=", 
  RowBox[{"Module", "[", 
   RowBox[{
    RowBox[{"{", 
     RowBox[{
      RowBox[{"tmaximun", "=", "tmax"}], ",", 
      RowBox[{"ntrayec", "=", "ntray"}], ",", "n", ",", 
      RowBox[{"g", "=", "grav"}], ",", 
      RowBox[{"l", "=", "long"}], ",", 
      RowBox[{"m", "=", "mass"}], ",", 
      RowBox[{"b", "=", "resis"}], ",", 
      RowBox[{"w", "=", "wext"}], ",", "w0", ",", "gamma", ",", "amp", ",", 
      "\[Theta]", ",", "\[Theta]0min", ",", "\[Theta]0max", ",", "v0min", ",",
       "v0max", ",", "\[Theta]aux", ",", "vaux", ",", "f", ",", "t", ",", "x",
       ",", "i", ",", "j", ",", "solutions", ",", "trayec"}], "}"}], ",", 
    RowBox[{
     RowBox[{"n", "=", 
      RowBox[{"Round", "[", 
       RowBox[{"Sqrt", "[", "ntrayec", "]"}], "]"}]}], ";", 
     RowBox[{"ntrayec", "=", 
      RowBox[{"n", "^", "2"}]}], ";", RowBox[{"gamma", "=", 
      RowBox[{"b", "/", "m"}]}], ";", RowBox[{"amp", " ", "=", 
      RowBox[{"f0ext", "/", 
       RowBox[{"(", 
        RowBox[{"m", "*", "l"}], ")"}]}]}], ";", RowBox[{"\[Theta]0min", "=", 
      RowBox[{"-", "0.15"}]}], ";", RowBox[{"\[Theta]0max", "=", "0.15"}], ";", RowBox[{"v0min", "=", 
      RowBox[{"-", "0.15"}]}], ";", RowBox[{"v0max", "=", "0.15"}], ";", RowBox[{"\[Theta]aux", "=", 
      RowBox[{
       RowBox[{"(", 
        RowBox[{"\[Theta]0max", "-", "\[Theta]0min"}], ")"}], "/", "n"}]}], 
     ";", " ", 
      RowBox[{"vaux", "=", 
      RowBox[{
       RowBox[{"(", 
        RowBox[{"v0max", "-", "v0min"}], ")"}], "/", "n"}]}], ";", "   ", 
      RowBox[{"solutions", "=", 
      RowBox[{"{", "}"}]}], ";", RowBox[{"Do", "[", RowBox[{
       RowBox[{"Do", "[", 
        RowBox[{
         RowBox[{"AppendTo", "[", 
          RowBox[{"solutions", ",", 
           RowBox[{"NDSolve", "[", 
            RowBox[{
             RowBox[{"{", 
              RowBox[{
               RowBox[{
                RowBox[{
                 SuperscriptBox[
                  SuperscriptBox["\[Theta]", "\[Prime]",
                   MultilineFunction->None], "\[Prime]",
                  MultilineFunction->None], "[", "t", "]"}], "\[Equal]", 
                RowBox[{
                 RowBox[{
                  RowBox[{"-", 
                   RowBox[{"(", 
                    RowBox[{"g", "/", "l"}], ")"}]}], " ", 
                  RowBox[{"Sin", "[", 
                   RowBox[{"\[Theta]", "[", "t", "]"}], "]"}]}], "-", 
                 RowBox[{"gamma", "*", 
                  RowBox[{
                   SuperscriptBox["\[Theta]", "\[Prime]",
                    MultilineFunction->None], "[", "t", "]"}]}], "+", 
                 RowBox[{"amp", "*", 
                  RowBox[{"Cos", "[", 
                   RowBox[{"w", "*", "t"}], "]"}]}]}]}], ",", 
               RowBox[{
                RowBox[{"\[Theta]", "[", "0", "]"}], "\[Equal]", 
                RowBox[{"\[Theta]0min", "+", 
                 RowBox[{"\[Theta]aux", "*", "i"}]}]}], ",", 
               RowBox[{
                RowBox[{
                 SuperscriptBox["\[Theta]", "\[Prime]",
                  MultilineFunction->None], "[", "0", "]"}], "\[Equal]", 
                RowBox[{"v0min", "+", 
                 RowBox[{"vaux", "*", "j"}]}]}]}], "}"}], ",", "\[Theta]", 
             ",", 
             RowBox[{"{", 
              RowBox[{"t", ",", "0", ",", "tmaximun"}], "}"}], ",", 
             RowBox[{"MaxSteps", "\[Rule]", "20000"}]}], "]"}]}], "]"}], ",", 
         RowBox[{"{", 
          RowBox[{"i", ",", "1", ",", "n"}], "}"}]}], "]"}], ",", 
       RowBox[{"{", 
        RowBox[{"j", ",", "1", ",", "n"}], "}"}]}], "]"}], ";", RowBox[{"solutions", "=", 
      RowBox[{"Flatten", "[", "solutions", "]"}]}], ";", 
     RowBox[{"trayec", "=", 
      RowBox[{"{", "}"}]}], ";", RowBox[{"Do", "[", RowBox[{
       RowBox[{"AppendTo", "[", 
        RowBox[{"trayec", ",", 
         RowBox[{
          RowBox[{"ParametricPlot3D", "[", 
           RowBox[{
            RowBox[{"Evaluate", "[", 
             RowBox[{"{", 
              RowBox[{
               RowBox[{
                RowBox[{
                 RowBox[{
                  RowBox[{"solutions", "[", 
                   RowBox[{"[", "i", "]"}], "]"}], "[", 
                  RowBox[{"[", "2", "]"}], "]"}], "'"}], "[", "t", "]"}], ",",
                "t", ",", 
               RowBox[{
                RowBox[{
                 RowBox[{"solutions", "[", 
                  RowBox[{"[", "i", "]"}], "]"}], "[", 
                 RowBox[{"[", "2", "]"}], "]"}], "[", "t", "]"}]}], "}"}], 
             "]"}], ",", 
            RowBox[{"{", 
             RowBox[{"t", ",", "0", ",", "tmaximun"}], "}"}], ",", 
            RowBox[{"PerformanceGoal", "\[Rule]", "\"\<Speed\>\""}], ",", 
            RowBox[{"Axes", "\[Rule]", "False"}], ",", 
            RowBox[{"PlotStyle", "\[Rule]", 
             RowBox[{"Hue", "[", 
              RowBox[{
               RowBox[{"1", "-", 
                RowBox[{"i", "/", "ntrayec"}]}], ",", "1", ",", "0.75"}], 
              "]"}]}]}], "]"}], "\[LeftDoubleBracket]", "1", 
          "\[RightDoubleBracket]"}]}], "]"}], ",", RowBox[{"{", 
        RowBox[{"i", ",", "1", ",", "ntrayec"}], "}"}]}], "]"}], ";", 
     RowBox[{"Show", "[", 
      RowBox[{
       RowBox[{"Graphics3D", "[", "trayec", "]"}], ",", 
       RowBox[{"ViewPoint", "\[Rule]", 
        RowBox[{"{", 
         RowBox[{"2.714", ",", " ", "1.876", ",", " ", "0.751"}], "}"}]}], 
       ",", 
       RowBox[{"Axes", "\[Rule]", "True"}], ",", 
       RowBox[{"PlotRange", "\[Rule]", 
        RowBox[{"{", 
         RowBox[{
          RowBox[{"{", 
           RowBox[{
            RowBox[{"-", "7"}], ",", "7"}], "}"}], ",", 
          RowBox[{"{", 
           RowBox[{"0", ",", "25"}], "}"}], ",", 
          RowBox[{"{", 
           RowBox[{
            RowBox[{"-", "20"}], ",", "20"}], "}"}]}], "}"}]}], ",", 
       RowBox[{"Background", "\[Rule]", 
        RowBox[{"GrayLevel", "[", "1", "]"}]}], ",", 
       RowBox[{"AxesLabel", "\[Rule]", 
        RowBox[{"{", 
         RowBox[{
         "\"\<\!\(\*FractionBox[\(d\[InvisibleSpace]\[Theta]\), \(d\
\[InvisibleSpace]t\)]\) (rad/s)\>\"", ",", " ", "\"\<t (s)\>\"", ",", " ", 
          "\"\<\[Theta] (rad)\>\""}], "}"}]}], ",", 
       RowBox[{"AspectRatio", "\[Rule]", "1"}], ",", 
       RowBox[{"ImageSize", "\[Rule]", 
        RowBox[{"{", 
         RowBox[{"300", ",", "300"}], "}"}]}]}], "]"}]}]}], "]"}]}]], "Input"]
}, Closed]],
Cell[CellGroupData[{
Cell["Manipulate", "Section"],
Cell[CellGroupData[{
Cell[BoxData[
 RowBox[{"Manipulate", "[", "\[IndentingNewLine]", 
  RowBox[{
   RowBox[{"DFPPlot", "[", 
    RowBox[{
    "time", ",", "ntrayec", ",", "length", ",", "mass", ",", "gravity", ",", 
     "resis", ",", "fext", ",", "wext"}], "]"}], ",", "\[IndentingNewLine]", 
   "\[IndentingNewLine]", 
   RowBox[{"Style", "[", 
    RowBox[{"\"\<parameters of the trajectories\>\"", ",", "Bold"}], "]"}], 
   ",", "\[IndentingNewLine]", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"time", ",", "25", ",", "\"\<time (s)\>\""}], "}"}], ",", "1", 
     ",", "25", ",", ".01", ",", 
     RowBox[{"Appearance", "\[Rule]", "\"\<Labeled\>\""}], ",", 
     RowBox[{"ImageSize", "\[Rule]", "Small"}]}], "}"}], ",", 
   "\[IndentingNewLine]", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"ntrayec", ",", "1", ",", "\"\<number of trajectories\>\""}], 
      "}"}], ",", 
     RowBox[{"{", 
      RowBox[{
      "1", ",", "4", ",", "9", ",", "16", ",", "25", ",", "36", ",", "49", 
       ",", "64"}], "}"}]}], "}"}], ",", "\[IndentingNewLine]", "Delimiter", 
   ",", "\[IndentingNewLine]", 
   RowBox[{"Style", "[", 
    RowBox[{"\"\<parameters of the pendulum\>\"", ",", "Bold"}], "]"}], ",", 
   "\[IndentingNewLine]", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"length", ",", "2", ",", "\"\<length ( m )\>\""}], "}"}], ",", 
     "1", ",", "10", ",", ".01", ",", 
     RowBox[{"Appearance", "\[Rule]", "\"\<Labeled\>\""}], ",", 
     RowBox[{"ImageSize", "\[Rule]", "Small"}]}], "}"}], ",", 
   "\[IndentingNewLine]", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"mass", ",", "1", ",", "\"\<mass ( kg )\>\""}], "}"}], ",", "1",
      ",", "10", ",", ".01", ",", 
     RowBox[{"Appearance", "\[Rule]", "\"\<Labeled\>\""}], ",", 
     RowBox[{"ImageSize", "\[Rule]", "Small"}]}], "}"}], ",", 
   "\[IndentingNewLine]", "Delimiter", ",", "\[IndentingNewLine]", 
   RowBox[{"Style", "[", 
    RowBox[{"\"\<acceleration due to gravity\>\"", ",", "Bold"}], "]"}], ",", 
   "\[IndentingNewLine]", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{
      "gravity", ",", "9.81", ",", 
       "\"\<gravity ( m/\!\(\*SuperscriptBox[\(s\), \(2\)]\) )\>\""}], "}"}], 
     ",", "0", ",", "12", ",", ".01", ",", 
     RowBox[{"Appearance", "\[Rule]", "\"\<Labeled\>\""}], ",", 
     RowBox[{"ImageSize", "\[Rule]", "Small"}]}], "}"}], ",", 
   "\[IndentingNewLine]", "Delimiter", ",", "\[IndentingNewLine]", 
   RowBox[{"Style", "[", 
    RowBox[{"\"\<resistance of the fluid\>\"", ",", "Bold"}], "]"}], ",", 
   "\[IndentingNewLine]", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"resis", ",", "1.11", ",", "\"\<resistance ( kg/s ) \>\""}], 
      "}"}], ",", "0", ",", "10", ",", ".01", ",", 
     RowBox[{"Appearance", "\[Rule]", "\"\<Labeled\>\""}], ",", 
     RowBox[{"ImageSize", "\[Rule]", "Small"}]}], "}"}], ",", 
   "\[IndentingNewLine]", "Delimiter", ",", "\[IndentingNewLine]", 
   RowBox[{"Style", "[", 
    RowBox[{"\"\<external force applied to pendulum\>\"", ",", "Bold"}], 
    "]"}], ",", "\[IndentingNewLine]", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"fext", ",", "11.46", ",", "\"\<amplitude ( N ) \>\""}], "}"}], 
     ",", "0", ",", "15", ",", ".01", ",", 
     RowBox[{"Appearance", "\[Rule]", "\"\<Labeled\>\""}], ",", 
     RowBox[{"ImageSize", "\[Rule]", "Small"}]}], "}"}], ",", 
   "\[IndentingNewLine]", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"wext", ",", "1.48", ",", "\"\<frequency ( rad/s ) \>\""}], 
      "}"}], ",", "0", ",", "4", ",", ".01", ",", 
     RowBox[{"Appearance", "\[Rule]", "\"\<Labeled\>\""}], ",", 
     RowBox[{"ImageSize", "\[Rule]", "Small"}]}], "}"}], ",", 
   "\[IndentingNewLine]", 
   RowBox[{"ControlPlacement", "\[Rule]", "Left"}], ",", 
   "\[IndentingNewLine]", 
   RowBox[{"SaveDefinitions", "\[Rule]", "True"}]}], "\[IndentingNewLine]", 
  "]"}]], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`time$$ = 25}, \"\\[Ellipsis]\"]"], "Output"]
}, Open]]
}, Closed]]
}]"##;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);
    let widget = editors
      .iter()
      .find_map(|e| e.manipulate_state.as_ref())
      .expect("the Manipulate cell must instantiate on load");
    assert!(
      widget.error.is_none(),
      "the pendulum must integrate: {:?}",
      widget.error
    );
    assert!(
      widget.graphics_handle.is_some(),
      "the trajectories must draw"
    );

    // Every control the Demonstration publishes, in order: five headings
    // separated by delimiters, seven sliders and one picklist.
    let labels: Vec<String> = widget
      .controls
      .iter()
      .map(|c| match c {
        manipulate::ControlState::Continuous { name, label, .. }
        | manipulate::ControlState::Discrete { name, label, .. } => {
          format!("{name}: {label}")
        }
        manipulate::ControlState::Heading { label, .. } => label.clone(),
        manipulate::ControlState::Divider => "-".to_string(),
        other => panic!("unexpected control: {other:?}"),
      })
      .collect();
    assert_eq!(
      labels,
      vec![
        "parameters of the trajectories",
        "time: time (s)",
        "ntrayec: number of trajectories",
        "-",
        "parameters of the pendulum",
        "length: length ( m )",
        "mass: mass ( kg )",
        "-",
        "acceleration due to gravity",
        // The unit's typeset exponent, not its box source.
        "gravity: gravity ( m/s\u{b2} )",
        "-",
        "resistance of the fluid",
        "resis: resistance ( kg/s ) ",
        "-",
        "external force applied to pendulum",
        "fext: amplitude ( N ) ",
        "wext: frequency ( rad/s ) ",
      ]
    );

    let render = |time: f64| {
      woxi::interpret_with_stdout(&format!(
        "time = {time}; ntrayec = 1; length = 2; mass = 1; gravity = 9.81; \
         resis = 1.11; fext = 11.46; wext = 1.48;\n{}",
        widget.body
      ))
      .expect("the body must render")
      .graphics
      .expect("the body must produce a graphic")
    };
    let published = render(25.0);
    // The frame the notebook asks for: `PlotRange -> {{-7, 7}, {0, 25},
    // {-20, 20}}` with all three axes labelled.
    for text in [">-5<", ">20<", ">-20<", "t (s)", "(rad/s)"] {
      assert!(published.contains(text), "expected {text} in the frame");
    }
    // A trajectory is a long polyline of coloured segments, not a handful.
    assert!(
      published.matches("<line").count() > 500,
      "expected a resolved trajectory, got {} segments",
      published.matches("<line").count()
    );
    // Integrating over a shorter interval draws a different trajectory.
    assert_ne!(render(5.0), published, "the time control must matter");
  }

  /// End-to-end regression for a Demonstration that draws a vector field:
  /// a grid of short arrows, each coloured by looking its value up in a
  /// named gradient through `ColorData[scheme, "ColorFunction"]`, inside a
  /// `Graphics` whose margins come from `ImagePadding`. All three used to
  /// fail together — the colour lookup was unimplemented so every arrow
  /// came out black, the padding was ignored, and the arrowheads were
  /// shrunk to fit shafts barely longer than themselves.
  #[test]
  fn vector_field_manipulate_colours_its_arrows() {
    let nb_src = r##"Notebook[{
Cell[BoxData["Manipulate[Module[{s = 2. r/n, pts}, pts = Table[{{x, y}, {y, -x}}, {x, -r, r, 2 r/n}, {y, -r, r, 2 r/n}]; Graphics[{Arrowheads[1./(2 n)], Map[{ColorData[scheme, \"ColorFunction\"][Norm[#[[2]]]/(Sqrt[2] r)], Arrow[{#[[1]], #[[1]] + s #[[2]]/Max[Norm[#[[2]]], 0.001]}]} &, pts, {2}]}, PlotRange -> All, Frame -> True, ImageSize -> {320, 300}, ImagePadding -> {{25, 25}, {25, 25}}]], {{scheme, \"RedBlueTones\", \"colors\"}, {\"RedBlueTones\", \"Rainbow\", \"SunsetColors\"}}, {{r, 3, \"domain size\"}, 1, 6}, {{n, 8, \"resolution\"}, 4, 12, 1}]"], "Input"]
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
    assert!(widget.graphics_handle.is_some(), "the field must draw");
    // One discrete gradient chooser and two sliders.
    assert_eq!(widget.controls.len(), 3, "{:?}", widget.controls);

    let svg = woxi::interpret_with_stdout(&format!(
      "scheme = \"RedBlueTones\"; r = 3; n = 8;\n{}",
      widget.body
    ))
    .expect("the body must render")
    .graphics
    .expect("the body must produce a graphic");
    // `ImagePadding -> {{25, 25}, {25, 25}}` puts the drawing area 25px in
    // from the top left, leaving 320-50 by 300-50 for the frame.
    assert!(
      svg.contains("translate(25,25)"),
      "the padding must place the drawing area: {svg:.400}"
    );
    // Each arrow is coloured from the gradient, so the fill colours are
    // many and none of them is the default black.
    let fills: std::collections::BTreeSet<&str> = svg
      .split("<polygon")
      .skip(1)
      .filter_map(|tag| tag.split_once("fill=\""))
      .filter_map(|(_, rest)| rest.split('"').next())
      .collect();
    assert!(
      fills.len() > 5,
      "the gradient must colour the arrows, got {fills:?}"
    );
    assert!(
      !fills.contains("rgb(0,0,0)"),
      "no arrow falls back to black: {fills:?}"
    );
    // `Arrowheads[1./(2 n)]` asks for heads a sixteenth of the plot wide;
    // they are drawn at that size even though each arrow is barely longer.
    let head_len = svg
      .split("<polygon points=\"")
      .nth(1)
      .and_then(|tag| tag.split('"').next())
      .map(|points| {
        let pts: Vec<(f64, f64)> = points
          .split_whitespace()
          .filter_map(|p| p.split_once(','))
          .filter_map(|(x, y)| Some((x.parse().ok()?, y.parse().ok()?)))
          .collect();
        let (tip, a, b) = (pts[0], pts[1], pts[2]);
        let mid = ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0);
        ((tip.0 - mid.0).powi(2) + (tip.1 - mid.1).powi(2)).sqrt()
      })
      .expect("an arrowhead must be drawn");
    // The drawing area is 320 less the 25px padding either side.
    let expected = (320.0 - 50.0) / 16.0;
    assert!(
      (head_len - expected).abs() < 2.0,
      "a 1/16 head spans {expected}px, got {head_len}"
    );
  }

  /// The SetterBar/PopupMenu split Wolfram's `Manipulate` makes on its own,
  /// pinned to the Demonstrations it was read off (see
  /// [`renders_as_setter_bar`]). The interesting pair is five phrases (a bar)
  /// against six single words (a dropdown) — the narrower row is the dropdown,
  /// so a width rule can't produce this and a count rule has to.
  #[test]
  fn setter_bar_is_chosen_the_way_wolfram_chooses_it() {
    let bar = |labels: &[&str]| {
      let labels: Vec<String> = labels.iter().map(|s| s.to_string()).collect();
      let svgs = vec![None; labels.len()];
      renders_as_setter_bar(&labels, &svgs)
    };

    // Up to five choices stay a bar even when every label is a phrase.
    assert!(bar(&["4", "20", "100", "500"]));
    assert!(bar(&["3", "4", "5", "6"]));
    assert!(bar(&["Poisson", "Gaussian", "gamma", "inverse Gaussian"]));
    assert!(bar(&[
      "prisoners dilemma",
      "battle of the sexes",
      "stag hunt",
      "coordination",
    ]));
    assert!(bar(&["2", "3", "4", "5", "6"]));
    // A dissection Demonstration's five target figures: 55 characters of
    // labels across five buttons is still a bar in Wolfram.
    assert!(bar(&[
      "quadrilateral",
      "Greek cross",
      "rhomboid",
      "rectangle",
      "right triangle",
    ]));

    // Past five, short labels keep the bar...
    assert!(bar(&["-3", "-2", "-1", "1", "2", "3", "4"]));
    assert!(bar(&["-3", "-2", "-1", "0", "1", "2", "3", "4"]));

    // ...and wide ones don't. Five sentence-long labels overflow the row
    // even though five phrase-long ones fit.
    assert!(!bar(&[
      "u(y)",
      "u'(y)",
      "u''(y)",
      "error in approximating u'(y)",
      "error in approximating u''(y)",
    ]));
    assert!(!bar(&[
      "triangle", "square", "pentagon", "hexagon", "heptagon", "octagon",
    ]));
    assert!(!bar(&[
      "Hue",
      "BlueGreenYellow",
      "BrightBands",
      "CMYKColors",
      "DarkBands",
      "GrayTones",
      "GrayYellowTones",
      "GreenPinkTones",
      "NeonColors",
      "Pastel",
      "Rainbow",
      "RedBlueTones",
      "RedGreenSplit",
      "SolarColors",
      "SunsetColors",
      "TemperatureMap",
      "ThermometerColors",
    ]));

    // Short labels buy a longer bar, not an unbounded one: 33 integers are a
    // dropdown even though every label is two or three characters.
    let many: Vec<String> = (-3..30).map(|n| n.to_string()).collect();
    assert!(!renders_as_setter_bar(&many, &vec![None; many.len()]));

    // An icon label is compact whatever its bound value reads as — it draws at
    // a fixed width, so a row of icons stays a bar.
    let icons: Vec<String> = (1..=6).map(|n| format!("choice {n}")).collect();
    assert!(!renders_as_setter_bar(&icons, &vec![None; icons.len()]));
    let handles: Vec<Option<svg::Handle>> = icons
      .iter()
      .map(|_| Some(svg::Handle::from_memory(Vec::new())))
      .collect();
    assert!(renders_as_setter_bar(&icons, &handles));
  }

  /// An explicit `ControlType -> RadioButtonBar` keeps its row of radio
  /// buttons even past the automatic SetterBar/PopupMenu split — the same
  /// forcing `ControlType -> SetterBar` already gets (see
  /// `demonstration_panel_with_escaped_glyphs_opens_live`). Six single-word
  /// choices are past that split (`renders_as_setter_bar` is false for this
  /// exact list above), so without the explicit type Woxi would draw a
  /// dropdown where Wolfram always draws the bar.
  #[test]
  fn radio_button_bar_control_type_forces_the_bar() {
    let expr = woxi::interpret_to_expr(
      "Manipulate[shape, {{shape, \"triangle\"}, \
       {\"triangle\", \"square\", \"pentagon\", \"hexagon\", \"heptagon\", \
       \"octagon\"}, ControlType -> RadioButtonBar}]",
    )
    .unwrap();
    let state = manipulate::ManipulateState::from_expr(&expr).unwrap();
    match &state.controls[..] {
      [
        manipulate::ControlState::Discrete {
          value_labels,
          setter_bar,
          popup,
          ..
        },
      ] => {
        assert!(
          !renders_as_setter_bar(value_labels, &vec![None; value_labels.len()]),
          "the automatic split must pick a dropdown for this list, so the \
           forced bar below is actually exercising ControlType"
        );
        assert!(
          *setter_bar && !*popup,
          "ControlType -> RadioButtonBar must force the bar layout"
        );
      }
      other => panic!("unexpected controls: {other:?}"),
    }
  }

  /// A control's caption: Wolfram writes the variable's own name when the
  /// spec gives no label, and writes nothing when the spec gives `""`. The
  /// two must not collapse into one another — a Demonstration suppresses a
  /// caption precisely by passing the empty string.
  #[test]
  fn an_explicitly_empty_control_label_stays_empty() {
    let control_labels = |code: &str| -> Vec<(String, usize)> {
      let expr = woxi::interpret_to_expr(code).expect("parses");
      let state =
        manipulate::ManipulateState::from_expr(&expr).expect("builds a widget");
      state
        .controls
        .iter()
        .map(|c| {
          let label = match c {
            manipulate::ControlState::Continuous { label, .. }
            | manipulate::ControlState::Discrete { label, .. } => label.clone(),
            other => panic!("unexpected control {other:?}"),
          };
          (label, manipulate_label_char_count(c))
        })
        .collect()
    };

    // No label of its own: the variable name captions the slider and sizes
    // the label column.
    assert_eq!(
      control_labels("Manipulate[x, {x, 0, 1}]"),
      [("x".into(), 1)]
    );
    // An initial value but still no label behaves the same way.
    assert_eq!(
      control_labels("Manipulate[x, {{x, 0.5}, 0, 1}]"),
      [("x".into(), 1)]
    );
    // An explicit label replaces the name.
    assert_eq!(
      control_labels("Manipulate[x, {{x, 0.5, \"move\"}, 0, 1}]"),
      [("move".into(), 4)]
    );
    // An explicit empty label suppresses the caption entirely, and claims no
    // width in the shared label column.
    assert_eq!(
      control_labels("Manipulate[x, {{x, 0.5, \"\"}, 0, 1}]"),
      [(String::new(), 0)]
    );
    assert_eq!(
      control_labels("Manipulate[n, {{n, 1, \"\"}, {1, 2, 3}}]"),
      [(String::new(), 0)]
    );
  }

  /// End-to-end regression for the "Regular Polygon Rolling on a Catenary"
  /// Demonstration: a `k`-gon rolls along a chain of catenary arches, its
  /// centre tracing the straight line the arches are cut for, with optional
  /// fans of earlier positions under constant angular and constant horizontal
  /// velocity.
  ///
  /// The body already evaluated; what diverged was the `polygon` control,
  /// which Woxi drew as a six-button SetterBar where Wolfram draws a dropdown.
  /// Checked against wolframscript's own rendering of the notebook at four
  /// control settings: the catenary/polygon/marker geometry agrees, and the
  /// three boolean controls are checkboxes in both.
  #[test]
  fn rolling_polygon_on_catenary_notebook_rolls_its_polygon() {
    let nb_src = r##"Notebook[{
Cell[BoxData[
 RowBox[{
  RowBox[{"positionoftangentang", "[", 
   RowBox[{"a_", ",", "q_"}], "]"}], ":=", 
  RowBox[{"2", " ", "a", " ", 
   RowBox[{"ArcTanh", "[", 
    RowBox[{"Tan", "[", 
     FractionBox["q", "2"], "]"}], "]"}]}]}]], "Input",
 InitializationCell->True],
Cell[BoxData[
 RowBox[{
  RowBox[{"aCatenary", "[", 
   RowBox[{"t_", ",", "a_"}], "]"}], ":=", 
  RowBox[{"a", " ", 
   RowBox[{"Cosh", "[", 
    RowBox[{"t", "/", "a"}], "]"}]}]}]], "Input",
 InitializationCell->True],
Cell[BoxData[
 RowBox[{
  RowBox[{"regularpolygon", "[", 
   RowBox[{"p_", ",", "w_", ",", "k_", ",", "R_"}], "]"}], ":=", 
  RowBox[{"Polygon", "[", 
   RowBox[{"Table", "[", 
    RowBox[{
     RowBox[{
      RowBox[{"R", 
       RowBox[{"{", 
        RowBox[{
         RowBox[{"Sin", "[", 
          RowBox[{"v", "-", 
           RowBox[{"w", " ", "2", 
            RowBox[{"Pi", "/", 
             RowBox[{"(", 
              RowBox[{"2", "k"}], ")"}]}]}]}], "  ", "]"}], ",", 
         RowBox[{"-", 
          RowBox[{"Cos", "[", 
           RowBox[{"v", "-", 
            RowBox[{"w", " ", "2", 
             RowBox[{"Pi", "/", 
              RowBox[{"(", 
               RowBox[{"2", "k"}], ")"}]}]}]}], "  ", "]"}]}]}], "}"}]}], "+",
       "p"}], ",", 
     RowBox[{"{", 
      RowBox[{"v", ",", "0", ",", 
       RowBox[{"2", "Pi"}], ",", 
       RowBox[{"2", 
        RowBox[{"Pi", "/", "k"}]}]}], "}"}]}], "]"}], "]"}]}]], "Input",
 InitializationCell->True],
Cell[CellGroupData[{
Cell[BoxData[
 RowBox[{"Manipulate", "[", 
  RowBox[{
   RowBox[{"With", "[", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"R", "=", "1"}], "}"}], ",", 
     RowBox[{"With", "[", 
      RowBox[{
       RowBox[{"{", 
        RowBox[{"a", "=", 
         RowBox[{"R", "*", " ", 
          RowBox[{"Cos", "[", 
           RowBox[{"Pi", "/", "k"}], "]"}]}]}], "}"}], ",", 
       RowBox[{"With", "[", 
        RowBox[{
         RowBox[{"{", 
          RowBox[{"tt", "=", 
           RowBox[{"positionoftangentang", "[", 
            RowBox[{"a", ",", 
             RowBox[{
              RowBox[{"(", 
               RowBox[{"2", 
                RowBox[{"Pi", "/", "k"}]}], ")"}], "/", "2"}]}], "]"}]}], 
          "}"}], ",", 
         RowBox[{"With", "[", 
          RowBox[{
           RowBox[{"{", 
            RowBox[{"hh", "=", 
             RowBox[{
              RowBox[{"tt", "*", "w"}], "-", 
              RowBox[{"tt", "*", 
               RowBox[{"Mod", "[", 
                RowBox[{"w", ",", "2"}], "]"}]}], "+", 
              RowBox[{"positionoftangentang", "[", 
               RowBox[{"a", ",", 
                RowBox[{
                 RowBox[{"-", 
                  FractionBox["\[Pi]", "k"]}], "+", 
                 FractionBox[
                  RowBox[{"\[Pi]", "  ", 
                   RowBox[{"Mod", "[", 
                    RowBox[{"w", ",", "2"}], "]"}]}], "k"]}]}], "]"}]}]}], 
            "}"}], ",", 
           RowBox[{"Graphics", "[", 
            RowBox[{
             RowBox[{"Flatten", "[", 
              RowBox[{"{", 
               RowBox[{
                RowBox[{"Table", "[", 
                 RowBox[{
                  RowBox[{"{", 
                   RowBox[{"Black", ",", 
                    RowBox[{"Line", "[", 
                    RowBox[{"Table", "[", 
                    RowBox[{
                    RowBox[{"{", 
                    RowBox[{
                    RowBox[{"t", "+", 
                    RowBox[{"2", "tt", "*", "per"}]}], ",", 
                    RowBox[{"-", 
                    RowBox[{"aCatenary", "[", 
                    RowBox[{"t", ",", "a"}], "]"}]}]}], "}"}], ",", 
                    RowBox[{"{", 
                    RowBox[{"t", ",", 
                    RowBox[{"-", "tt"}], ",", "tt", ",", 
                    RowBox[{"tt", "/", "60"}]}], "}"}]}], "]"}], "]"}]}], 
                   "}"}], ",", 
                  RowBox[{"{", 
                   RowBox[{"per", ",", 
                    RowBox[{"-", "1"}], ",", "2"}], "}"}]}], "]"}], ",", 
                "Gray", ",", 
                RowBox[{"regularpolygon", "[", 
                 RowBox[{
                  RowBox[{"{", 
                   RowBox[{
                    RowBox[{"+", "hh"}], ",", "0"}], "}"}], ",", "w", ",", 
                  "k", ",", "R"}], "]"}], ",", "Black", ",", 
                RowBox[{"Line", "[", 
                 RowBox[{"{", 
                  RowBox[{
                   RowBox[{"{", 
                    RowBox[{
                    RowBox[{
                    RowBox[{"-", "tt"}], "-", 
                    RowBox[{"2", "tt"}]}], ",", "0"}], "}"}], ",", 
                   RowBox[{"{", 
                    RowBox[{
                    RowBox[{
                    RowBox[{"4", "tt"}], "+", "tt"}], ",", "0"}], "}"}]}], 
                  "}"}], "]"}], ",", 
                RowBox[{"If", "[", 
                 RowBox[{"shocon", ",", 
                  RowBox[{"Table", "[", 
                   RowBox[{
                    RowBox[{"{", 
                    RowBox[{
                    RowBox[{"RGBColor", "[", 
                    RowBox[{"1", ",", ".21", ",", "0"}], "]"}], ",", 
                    RowBox[{"Line", "@@", 
                    RowBox[{"regularpolygon", "[", 
                    RowBox[{
                    RowBox[{"{", 
                    RowBox[{
                    RowBox[{
                    RowBox[{"t", "*", "tt"}], "+", "tt"}], ",", "0"}], "}"}], 
                    ",", "t", ",", "k", ",", "R"}], "]"}]}], ",", 
                    RowBox[{"Disk", "[", 
                    RowBox[{
                    RowBox[{"{", 
                    RowBox[{
                    RowBox[{
                    RowBox[{"t", "*", "tt"}], "+", "tt"}], ",", "0"}], "}"}], 
                    ",", ".02"}], "]"}]}], "}"}], ",", 
                    RowBox[{"{", 
                    RowBox[{"t", ",", 
                    RowBox[{"-", "3"}], ",", "1", ",", 
                    RowBox[{"1", "/", "8"}]}], "}"}]}], "]"}], ",", 
                  RowBox[{"{", "}"}]}], "]"}], ",", "\[IndentingNewLine]", 
                RowBox[{"If", "[", 
                 RowBox[{"shofle", ",", 
                  RowBox[{"Table", "[", 
                   RowBox[{
                    RowBox[{"With", "[", 
                    RowBox[{
                    RowBox[{"{", 
                    RowBox[{"hhx", "=", 
                    RowBox[{
                    RowBox[{"tt", " ", "*", " ", "t"}], "-", 
                    RowBox[{"tt", " ", 
                    RowBox[{"Mod", "[", 
                    RowBox[{"t", ",", "2"}], "]"}]}], "+", 
                    RowBox[{"positionoftangentang", "[", 
                    RowBox[{"a", ",", 
                    RowBox[{
                    RowBox[{"-", 
                    FractionBox["\[Pi]", "k"]}], "+", 
                    FractionBox[
                    RowBox[{"\[Pi]", "  ", 
                    RowBox[{"Mod", "[", 
                    RowBox[{"t", ",", "2"}], "]"}]}], "k"]}]}], "]"}]}]}], 
                    "}"}], ",", 
                    RowBox[{"{", 
                    RowBox[{
                    RowBox[{"RGBColor", "[", 
                    RowBox[{".11", ",", ".61", ",", ".79"}], "]"}], ",", 
                    RowBox[{"Line", "@@", 
                    RowBox[{"regularpolygon", "[", 
                    RowBox[{
                    RowBox[{"{", 
                    RowBox[{"hhx", ",", "0"}], "}"}], ",", "t", ",", "k", ",",
                     "R"}], "]"}]}], ",", 
                    RowBox[{"Disk", "[", 
                    RowBox[{
                    RowBox[{"{", 
                    RowBox[{"hhx", ",", "0"}], "}"}], ",", ".02"}], "]"}]}], 
                    "}"}]}], "]"}], ",", 
                    RowBox[{"{", 
                    RowBox[{"t", ",", "1", ",", "5", ",", 
                    RowBox[{"1", "/", "8"}]}], "}"}]}], "]"}], ",", 
                  RowBox[{"{", "}"}]}], "]"}], ",", "White", ",", 
                RowBox[{"Disk", "[", 
                 RowBox[{
                  RowBox[{"{", 
                   RowBox[{
                    RowBox[{"+", "hh"}], ",", "0"}], "}"}], ",", ".02"}], 
                 "]"}]}], "}"}], "]"}], ",", 
             RowBox[{"ImageSize", "\[Rule]", "500"}], ",", 
             RowBox[{"PlotRange", "\[Rule]", 
              RowBox[{"If", "[", 
               RowBox[{"zoo", ",", 
                RowBox[{"{", 
                 RowBox[{
                  RowBox[{
                   RowBox[{"{", 
                    RowBox[{
                    RowBox[{"(", 
                    RowBox[{
                    RowBox[{"-", "2"}], "-", 
                    RowBox[{"1", "/", "3"}]}], ")"}], ",", 
                    RowBox[{
                    RowBox[{"(", 
                    RowBox[{"3", "+", 
                    RowBox[{"1", "/", "2"}]}], ")"}], "/", "8"}]}], "}"}], 
                   "+", 
                   RowBox[{"(", 
                    RowBox[{
                    RowBox[{"w", "/", "2"}], "+", "1"}], ")"}]}], ",", 
                  RowBox[{"{", 
                   RowBox[{
                    RowBox[{
                    RowBox[{"-", "5"}], "/", "4"}], ",", ".1"}], "}"}]}], 
                 "}"}], ",", 
                RowBox[{"{", 
                 RowBox[{
                  RowBox[{"{", 
                   RowBox[{
                    RowBox[{
                    RowBox[{"-", "2"}], "-", 
                    RowBox[{"1", "/", "3"}]}], ",", 
                    RowBox[{"3", "+", 
                    RowBox[{"1", "/", "2"}]}]}], "}"}], ",", 
                  RowBox[{"{", 
                   RowBox[{
                    RowBox[{
                    RowBox[{"-", "5"}], "/", "4"}], ",", 
                    RowBox[{"5", "/", "4"}]}], "}"}]}], "}"}]}], "]"}]}]}], 
            "]"}]}], "]"}]}], "]"}]}], "]"}]}], "]"}], ",", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"w", ",", "0", ",", "\"\<rotation of polygon\>\""}], "}"}], ",", 
     RowBox[{"-", "1"}], ",", "5"}], "}"}], ",", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"k", ",", "4", ",", "\"\<polygon\>\""}], "}"}], ",", 
     RowBox[{"Thread", "[", 
      RowBox[{"Rule", "[", 
       RowBox[{
        RowBox[{"Range", "[", 
         RowBox[{"3", ",", "8"}], "]"}], ",", 
        RowBox[{"{", 
         RowBox[{
         "\"\<triangle\>\"", ",", "\"\<square\>\"", ",", "\"\<pentagon\>\"", 
          ",", "\"\<hexagon\>\"", ",", "\"\<heptagon\>\"", ",", 
          "\"\<octagon\>\""}], "}"}]}], "]"}], "]"}]}], "}"}], ",", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{
      "shofle", ",", "False", ",", "\"\<constant angular velocity\>\""}], 
      "}"}], ",", 
     RowBox[{"{", 
      RowBox[{"False", ",", "True"}], "}"}]}], "}"}], ",", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{
      "shocon", ",", "False", ",", 
       "\"\<constant angular and horizontal velocity\>\""}], "}"}], ",", 
     RowBox[{"{", 
      RowBox[{"False", ",", "True"}], "}"}]}], "}"}], ",", 
   "\[IndentingNewLine]", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"zoo", ",", "False", ",", "\"\<zoom\>\""}], "}"}], ",", 
     RowBox[{"{", 
      RowBox[{"False", ",", "True"}], "}"}]}], "}"}], ",", 
   "\[IndentingNewLine]", 
   RowBox[{"SaveDefinitions", "\[Rule]", "True"}]}], "]"}]], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`w$$ = 0}, \"\\[Ellipsis]\"]"], "Output"]
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
      "the body must evaluate: {:?}",
      widget.error
    );
    assert!(widget.graphics_handle.is_some(), "the polygon must draw");

    // The three initialization cells above the Manipulate define the helpers
    // its body calls; without them the body can't evaluate at all.
    let names: Vec<&str> = widget
      .controls
      .iter()
      .map(|c| match c {
        manipulate::ControlState::Continuous { name, .. } => name.as_str(),
        manipulate::ControlState::Discrete { name, .. } => name.as_str(),
        other => panic!("unexpected control: {other:?}"),
      })
      .collect();
    assert_eq!(names, ["w", "k", "shofle", "shocon", "zoo"]);

    // `Thread[Rule[Range[3, 8], {"triangle", …}]]` builds the polygon choices,
    // so the control binds 3..8 while showing the names. Six words is a
    // dropdown, matching Wolfram.
    match &widget.controls[1] {
      manipulate::ControlState::Discrete {
        values,
        value_labels,
        value_label_svgs,
        current_index,
        ..
      } => {
        assert_eq!(values, &["3", "4", "5", "6", "7", "8"]);
        assert_eq!(
          value_labels,
          &[
            "triangle", "square", "pentagon", "hexagon", "heptagon", "octagon",
          ]
        );
        assert_eq!(*current_index, 1, "the square is the default");
        assert!(
          !renders_as_setter_bar(value_labels, value_label_svgs),
          "Wolfram shows the polygon choices as a dropdown"
        );
      }
      other => panic!("expected the polygon control, got {other:?}"),
    }

    // The `{False, True}` controls are boolean domains, which render as
    // checkboxes rather than two-button setters.
    for idx in [2, 3, 4] {
      match &widget.controls[idx] {
        manipulate::ControlState::Discrete { values, .. } => {
          assert_eq!(values, &["False", "True"], "control {idx}");
        }
        other => panic!("expected a boolean control, got {other:?}"),
      }
    }

    let render = |w: &str, k: u32, shofle: &str, shocon: &str, zoo: &str| {
      woxi::interpret_with_stdout(&format!(
        "w = {w}; k = {k}; shofle = {shofle}; shocon = {shocon}; zoo = {zoo};\n{}",
        widget.body
      ))
      .expect("the body must render")
      .graphics
      .expect("the body must produce a graphic")
    };

    // At rest: four catenary arches (`per` runs -1..2) plus the straight line
    // the polygon rolls along, the polygon itself, and the white centre dot.
    let square = render("0", 4, "False", "False", "False");
    assert_eq!(square.matches("<polyline").count(), 5, "{square}");
    assert_eq!(square.matches("<polygon").count(), 1, "{square}");
    assert_eq!(square.matches("<ellipse").count(), 1, "{square}");
    // The rolling polygon is grey and the centre dot white.
    assert!(square.contains("fill=\"rgb(128,128,128)\""), "{square}");
    assert!(square.contains("fill=\"rgb(255,255,255)\""), "{square}");

    // Each overlay adds one outline and one marker per step of a 33-step
    // table (`t` from -3 to 1, and 1 to 5, by 1/8), in its own colour.
    let fans = render("2", 5, "True", "True", "False");
    assert_eq!(fans.matches("<polyline").count(), 5 + 33 + 33, "{fans}");
    assert_eq!(fans.matches("<ellipse").count(), 1 + 33 + 33, "{fans}");
    assert!(fans.contains("rgb(255,54,0)"), "constant-velocity fan");
    assert!(fans.contains("rgb(28,156,201)"), "angular-velocity fan");

    // Each control changes the picture: a different polygon, a different
    // rotation, and the zoom, which narrows the plot range around the polygon.
    assert_ne!(square, render("0", 5, "False", "False", "False"));
    assert_ne!(square, render("1", 4, "False", "False", "False"));
    assert_ne!(square, render("0", 4, "False", "False", "True"));
  }

  /// End-to-end regression for "Filling Cone, Hemisphere and Cylinder:
  /// Easy as 1:2:3": three vessels filling in the ratio 1:2:3, drawn out
  /// of `Tube` walls and a `RevolutionPlot3D` bowl lifted out of its plot,
  /// with a `SetterBar` switching between five views.
  ///
  /// Four things the notebook needs were missing: `Tube` collected no
  /// primitives at all, `CapForm` was not a directive, `First` could not
  /// reach inside a `RevolutionPlot3D`, and the water level came from
  /// `NSolve[eqn && 0 <= f <= 1]` — a one-argument call whose cubic roots
  /// were not filtered by the bound.
  #[test]
  fn filling_cone_hemisphere_cylinder_notebook_draws_every_view() {
    let nb_src = r##"Notebook[{
Cell[BoxData["myCone[h_] := {CapForm[\"Butt\"], Cone[{{0,0,h},{0,0,0}},h], Opacity[.2], CapForm[None], Tube[{{0,0,1},{0,0,h}},{1,h}], Black, CapForm[None], Tube[{{0,0,.99},{0,0,1}},{1,1}]}"], "Input"],
Cell[BoxData["myCylinder[h_] := {CapForm[\"Butt\"], Tube[{{0,0,0},{0,0,h}},{1,1}], Opacity[.2], CapForm[None], Tube[{{0,0,1},{0,0,h}},{1,1}], Black, CapForm[None], Tube[{{0,0,h-0.01},{0,0,h+0.01}},{1,1}], Black, CapForm[None], Tube[{{0,0,.99},{0,0,1}},{1,1}]}"], "Input"],
Cell[BoxData["myHemisphere[h_] := Module[{tf = -ArcSin[1-h]}, {If[h>0, First@RevolutionPlot3D[{Cos[th],1+Sin[th]},{th,-Pi/2,tf},Mesh->None,PerformanceGoal->\"Quality\"], Sequence[]], If[h<1, First@RevolutionPlot3D[{Cos[th],1+Sin[th]},{th,tf,0},Mesh->None,PerformanceGoal->\"Quality\",PlotStyle->Opacity[.2]], Sequence[]], Black, CapForm[None], Tube[{{0,0,.99},{0,0,1}},{1,1}]}]"], "Input"],
Cell[BoxData["heights[V_] := Module[{f}, (If[Length[#]>1 && Last[#]==0, Most[#], #]&) /@ {Flatten[{Table[1,{Quotient[V,1]}], (Mod[V,1])^(1/3)}], Flatten[{Table[1,{Quotient[V,2]}], Evaluate@NSolve[Mod[V,2]==f^2 (3-f) && 0<=f<=1, f][[1,1,2]]}], Flatten[{Table[1,{Quotient[V,3]}], Mod[V,3]/3}]}]"], "Input"],
Cell[BoxData["AnnotatedArrow[p_,q_,label_]:={Arrowheads[{{-Medium,0},{.1,.5,Graphics[Inset[Style[label,Medium,Italic],{Center,Top},{Center,Bottom}]]},{Medium,1}}],Arrow[{p,q}]}"], "Input"],
Cell[CellGroupData[{
Cell[BoxData["Manipulate[Switch[ChooseControlMode,\n1, tmax = 4;\nGraphics3D[MapIndexed[Switch[#2[[1]],\n1, Translate[myCone[#1], {-2, 0, #2[[2]] - 1}],\n2, Translate[myHemisphere[#1], {0, 0, #2[[2]] - 1}],\n3, Translate[myCylinder[#1], {2, 0, #2[[2]] - 1}]] &, heights[t], {2}],\nFaceGrids -> {{{0, 1, 0}, {{-3, -2, 0, 2, 3}, Range[0, 4, 1]}}, {{0, 0, -1}, {{-3, 3}, {-1, 1}}}},\nImageSize -> {450, 450},\nPlotRange -> {{-3, 3}, {-1, 1}, {0, 4}}, ViewPoint -> {-1.46, -2.96, 0.72},\nViewVertical -> {-0.03, -0.19, 0.98}, Boxed -> False,\nMethod -> {\"ShrinkWrap\" -> True}],\n\n2, Plot[{3 r, r^2 (3 - r), r^3}, {r, 0, 1}, PlotRange -> {0, 3},\nAspectRatio -> 1, GridLines -> {Range[0, 1, .2], Range[3]},\nImageSize -> {450, 395},\nPlotLegends -> Placed[{\nRow[{\"cylinder: 3\", Style[\"f\", Italic]}],\nRow[{\"hemisphere: \", Superscript[Style[\"f\", Italic], 2], \"(3-\", Style[\"f\", Italic], \")\"}],\nRow[{\"cone: \", Superscript[Style[\"f\", Italic], \"3\"]}]}, Below],\nAxesLabel -> (Style[#, \"Subsubtitle\"] & /@ {\nRow[{Style[\"f\", Italic], \" ≡ \", Style[\"h\", Italic], \"/\", Style[\"r\", Italic]}],\nRow[{Style[\"V\", Italic], \"/\", Subscript[Style[\"V\", Italic], \"full cone\"]}]}),\nPlotLabel -> Style[\"water volumes in three containers versus height\", \"Subsubtitle\"]],\n\n3 | 4, emb = 4 - ChooseControlMode; tmax = 2 - emb; t = Min[t, tmax];\nGraphics3D[{LightBlue, Opacity[0.6], myCylinder[heights[t][[2 - emb, 1]]],\nGray, Opacity[.9],\nRotate[Switch[emb, 0, myCone, 1, myHemisphere][1], Pi, {0, 1, 0}, {0, 0, .5}],\nLightBlue, Opacity[0.4],\nTranslate[Switch[emb, 0, myHemisphere, 1, myCone][heights[t][[2 - emb, 1]]], {2, 0, 0}]},\nViewPoint -> {-0.89, -3.22, 0.52}, ViewVertical -> {-0.04, -0.24, 0.97},\nImageSize -> {450, 450}],\n\n5, tmax = 0.8; t = Max[t, 0.2]; t = Min[t, tmax]; h = t;\nLabeled[Pane[\nGrid[{\n{Module[{r = 1, x, y}, y = -(r - h); x = Abs[y];\nGraphics[{Line[{{-1, 0}, {-1, -1}, {1, -1}, {1, 0}}],\nLine[{{-1, -1}, {0, 0}, {1, -1}}], Line[{{-1, y}, {1, y}}],\nAnnotatedArrow[{0, 0}, {1, 0}, \"r\"],\nAnnotatedArrow[{0, y}, {x, y}, \"r-h\"],\nAnnotatedArrow[{0, y}, {0, 0}, \"r-h\"],\nAnnotatedArrow[{0, -1}, {0, y}, \"h\"],\nLightBlue, Polygon[{{x, y}, {1, y}, {1, -1}}],\nPolygon[{{-x, y}, {-1, y}, {-1, -1}}], Thick, Blue,\nLine[{{-1, y}, {-x, y}}], Line[{{1, y}, {x, y}}]},\nBaseStyle -> {Medium, Italic},\nPlotRange -> 1.01 {{-1, 1}, {-1, 0.1}}, ImageSize -> {450, 160}]]},\n{Text@Row[{Subscript[Style[\"A\", Italic], \"slice\"], \" = Pi \",\nSuperscript[Style[\"x\", Italic], 2], \" = Pi (\",\nSuperscript[Style[\"r\", Italic], 2], \" - \",\nSuperscript[Row[{\"(\", Style[\"r\", Italic], \"-\", Style[\"h\", Italic], \")\"}], 2], \")\",\n\"= Pi \", Superscript[Style[\"r\", Italic], 2], \"-Pi \",\nSuperscript[Style[\"r\", Italic], 2], \" - \",\nSuperscript[Row[{\"Pi (\", Style[\"r\", Italic], \"-\", Style[\"h\", Italic], \")\"}], 2]}]},\n{},\n{Module[{r = 1, x, y}, y = -(r - h); x = Sqrt[r^2 - y^2];\nGraphics[{Circle[{0, 0}, 1, {Pi, 2 Pi}], LightBlue,\nDisk[{0, 0}, 1, {ArcTan[-x, y], ArcTan[x, y]}], White,\nPolygon[{{0, 0}, {x, y}, {-x, y}}], Black,\nLine[{{0, 0}, {x, y}}], Text[\"r\", {x, y}/2, {0, -1.5}],\nText[\"x\", {0.45 x, y}, {0, -1.2}],\nAnnotatedArrow[{0, 0}, {1, 0}, \"r\"],\nAnnotatedArrow[{0, y}, {0, 0}, \"r-h\"],\nAnnotatedArrow[{0, -1}, {0, y}, \"h\"], Thick, Blue,\nLine[{{-x, y}, {x, y}}]}, BaseStyle -> {Medium, Italic},\nPlotRange -> 1.01 {{-1, 1}, {-1, 0.1}}, ImageSize -> {450, 160}]]},\n{Text@Row[{Subscript[Style[\"A\", Italic], \"slice\"], \" = \",\nSubscript[Style[\"A\", Italic], \"big circle\"], \" - \",\nSubscript[Style[\"A\", Italic], \"small circle\"], \" = Pi \",\nSuperscript[Style[\"r\", Italic], 2], \" - \",\nSuperscript[Row[{\"Pi (\", Style[\"r\", Italic], \"-\", Style[\"h\", Italic], \")\"}], 2]}]}},\nItemSize -> Full], Center, ImageSize -> {450, 401}],\nText@Style[\"cone submerged in a cylinder versus a hemisphere\\n\\n\", \"Subsubtitle\"], Top]\n],\n{{ChooseControlMode, 1, \"view: \"}, {1 -> \"fill all\", 2 -> Row[{Style[\"V\", Italic], \" versus \", Style[\"h\", Italic], \" graph\"}], 3 -> \"submerged hemisphere\", 4 -> \"submerged cone\", 5 -> \"geometric explanation\"}, SetterBar},\nControl@{{t, 0.1, Style[\"t\", Italic]}, 0.1, tmax, 0.01, Appearance -> \"Labeled\", Enabled -> If[ChooseControlMode != 2, True, False]},\n{emb, 0, 1, ControlType -> None},\n{h, 0, 1, ControlType -> None},\n{{tmax, 4}, {1, 4}, ControlType -> None},\nSaveDefinitions -> True, Alignment -> Center, ControlPlacement -> Top]"], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`ChooseControlMode$$ = 1}, \"\\[Ellipsis]\"]"], "Output"]
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
      "the vessels must build: {:?}",
      widget.error
    );
    assert!(widget.graphics_handle.is_some(), "the vessels must draw");

    // The view picker and the fill slider get a row each; the three
    // `ControlType -> None` parameters are bound without one.
    let names: Vec<&str> = widget
      .controls
      .iter()
      .map(|c| match c {
        manipulate::ControlState::Discrete { name, .. }
        | manipulate::ControlState::Continuous { name, .. } => name.as_str(),
        other => panic!("unexpected control: {other:?}"),
      })
      .collect();
    assert_eq!(names, ["ChooseControlMode", "t"]);
    let hidden: Vec<&str> =
      widget.state.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(hidden, ["emb", "h", "tmax"]);

    let render = |mode: u32, t: &str| {
      let bindings: String = widget
        .state
        .iter()
        .map(|(n, v)| format!("{n} = {v};\n"))
        .collect();
      woxi::interpret_with_stdout(&format!(
        "{bindings}ChooseControlMode = {mode}; t = {t};\n{}",
        widget.body
      ))
      .expect("the body must render")
      .graphics
      .expect("the body must produce a graphic")
    };

    // View 1 fills all three vessels: the walls are tubes and the bowl is
    // a lifted revolution surface, so the picture is polygons throughout.
    let filled = render(1, "0.5");
    assert!(filled.contains("<polygon"), "the vessels must draw");
    assert_ne!(filled, render(1, "2.0"), "the fill level must matter");

    // View 2 is the volume-versus-height plot, and each of the remaining
    // views draws something of its own.
    let plot = render(2, "0.5");
    assert!(
      plot.contains("<polyline") || plot.contains("<path"),
      "the volume curves must draw"
    );
    let views: Vec<String> = (1..=5).map(|m| render(m, "0.5")).collect();
    for (i, a) in views.iter().enumerate() {
      for b in &views[i + 1..] {
        assert_ne!(a, b, "each view must draw something of its own");
      }
    }

    // The geometric explanation is a `Labeled[Pane[Grid[…]]]` of two
    // diagrams and their two formulas, composed rather than written out.
    let explanation = &views[4];
    assert!(!explanation.contains("GraphicsBox"), "{explanation:.400}");
    assert!(
      explanation.contains("cone submerged in a cylinder"),
      "the caption must typeset"
    );
  }

  /// End-to-end regression for the "Thermodynamic Consistency Test Based on
  /// Differential Residuals" Demonstration: four vapour-liquid equilibrium
  /// datasets, each shown as a pressure diagram, an x-y diagram, or the
  /// differential-residual consistency test, with the measured points drawn
  /// as `PlotMarkers` glyphs and the +/-0.1 acceptance band as an `Epilog`.
  ///
  /// The markers and the band are what the notebook needed: a list plot used
  /// to ignore both, so every dataset came out as identical default dots and
  /// the band was missing entirely. The rendered points match the
  /// Demonstration's own published snapshots.
  #[test]
  fn thermodynamic_consistency_notebook_marks_its_data() {
    let nb_src = r##"Notebook[{
Cell[CellGroupData[{
Cell[BoxData["Manipulate[\nModule[{xed,yed,Tbled,pltxye,a1,a2,GeOverRTe,GeOverRTx1x2e,lG1,lG2,b1,b2,A12,A21,T,G1,G2,plt1,plt2,plt1e,plt2e,pltxy,Tble,Tbl1,Tbl2,xe,ye,sole,sol,Pc,Pe,x1,PS1,PS2,x,GEoverRTx1x2,GEoverRT,GEoverRTx1x2vsx1,LG1vsx1,LG2vsx1,Px,Py,a,Pes},Switch[ctrl1,1,{A21=0.576900005340576;\nA12=0.612100005149841;\nT=58+273.15;\nA1=71.3031;B1=-5952.00;C1=0.0;D1=-8.53128;E1=7.82393 10^-6;F1=2.0;\nA2=59.8373;B2=-6282.89;C2=0.0;D2=-6.37873;E2=4.61746 10^-6;F2=2.0;\n\nPS1=Exp[A1+B1/(C1+T)+D1 Log[T]+E1 T^F1];\nPS2=Exp[A2+B2/(C2+T)+D2 Log[T]+E2 T^F2];\n\nG1[x_]:=Exp[(1-x)^2 (A12+2 (A21-A12) x)];\nG2[x_]:=Exp[x^2 (A21+2 (A12-A21) (1-x))];\n\ni=0;\nWhile[i< 101,{x=i 0.01,xe[i]=x,sole=FindRoot[G1[x] x PS1+G2[x] (1-x) PS2== P,{P, 100}],Pe[i]=(P/.sole),ye[i]=PS1 G1[x] xe[i]/P/.sole,i++}];\n\nTbl1=Table[{xe[i],Pe[i]},{i,0,100}];\nTbl2=Table[{ye[i],Pe[i]},{i,0,100}];\n\nplt1=ListPlot[Tbl1, PlotStyle->{Thick,RGBColor[1,0,0]},Joined-> True,GridLines-> Automatic,Frame-> True,AspectRatio-> 1,PlotRange-> {{0,1},Automatic}];\n\nplt2=ListPlot[Tbl2, PlotStyle->{Thick,RGBColor[0,0,1]},Joined-> True,GridLines-> Automatic,Frame-> True,AspectRatio-> 1,PlotRange-> {{0,1},Automatic}];\n\nTble=Table[{xe[i],ye[i]},{i,0,100}];\n\npltxy=ListPlot[Tble,PlotStyle->{Thick,RGBColor[0,1,0]},Joined-> True,GridLines-> Automatic,Frame-> True,AspectRatio-> 1,PlotRange-> {{0,1},{0,1}},Epilog-> {Black,Line[{{0,0},{1,1}}]}];i=0;Quiet@While[i< 101,{xed[i]=If[i==0,0,Min[Max[xe[i]+Wnoise1 Random[]+SysE1,0],1]],yed[i]=If[i==0,0,Min[Max[ye[i]+Wnoise2 Random[]+SysE2,0],1]],lG1[i]=Log[yed[i] Pe[i]/xed[i]/PS1],lG2[i]=Log[(1-yed[i]) Pe[i]/(1-xed[i])/PS2],GeOverRTe[i]=xed[i]lG1[i]+(1-xed[i])lG2[i],GeOverRTx1x2e[i]=GeOverRTe[i]/(xed[i] (1-xed[i])),i=i+10}];\n\nTbl1=Table[{xed[i],Pe[i]},{i,0,101,10}];\n\nTbl2=Table[{yed[i],Pe[i]},{i,0,101,10}];\n\nplt1e=ListPlot[Tbl1,PlotMarkers-> Style[\"◆\", Red,18],GridLines-> Automatic,Frame-> True,AspectRatio-> 1,PlotRange-> {{0,1},Automatic}];\n\nplt2e=ListPlot[Tbl2,PlotMarkers->Style[\"◆\", Blue,18],GridLines-> Automatic,Frame-> True,AspectRatio-> 1,PlotRange-> {{0,1},Automatic}];\n\nTbled=Table[{xed[i],yed[i]},{i,0,100,10}];\n\npltxye=ListPlot[Tbled,PlotMarkers-> Style[\"■\",16,Blue],GridLines-> Automatic,Frame-> True,AspectRatio-> 1,PlotRange-> {{0,1},{0,1}},Epilog-> {Black,Line[{{0,0},{1,1}}]}];\n\na1=ListPlot[Table[{xed[i],(Log[G1[xed[i]]]-Log[G2[xed[i]]])-(lG1[i]-lG2[i])},{i,10,91,10}],GridLines-> Automatic,Frame-> True,AspectRatio-> 1,PlotRange-> {{0,1},{-0.4,0.4}},PlotMarkers->Style[\"●\", 16, Red]];\n\na2=ListPlot[Table[{xed[i],(xed[i]Log[G1[xed[i]]]+(1-xed[i])Log[G2[xed[i]]])-GeOverRTe[i]},{i,10,91,10}],GridLines-> Automatic,Frame-> True,AspectRatio-> 1,PlotRange-> {{0,1},{-0.4,0.4}},PlotMarkers->Style[\"○\", 18,Blue]];},\n\n2,\n\n{a={{15.51`,0.0895`,0.2716`,0.266`,0.009`,0.032`,0.389`},{18.61`,0.1981`,0.4565`,0.172`,0.025`,0.054`,0.342`},{21.63`,0.3193`,0.5934`,0.108`,0.049`,0.068`,0.312`},{24.01`,0.4232`,0.6815`,0.069`,0.075`,0.072`,0.297`},{25.92`,0.5119`,0.744`,0.043`,0.1`,0.071`,0.283`},{27.96`,0.6096`,0.805`,0.023`,0.127`,0.063`,0.267`},{30.12`,0.7135`,0.8639`,0.01`,0.151`,0.051`,0.248`},{31.75`,0.7934`,0.9048`,0.003`,0.173`,0.038`,0.234`},{34.15`,0.9102`,0.959`,-0.003`,0.237`,0.019`,0.227`}};\nPe=Table[a[[i]][[1]],{i,1,9}];\nxed=Table[a[[i]][[2]],{i,1,9}];\nyed=Table[a[[i]][[3]],{i,1,9}];\nPx=Join[{{0,12.30}},Table[{xed[[i]],Pe[[i]]},{i,1,9}],{{1,36.09}}];\nPy=Join[{{0,12.30}},Table[{yed[[i]],Pe[[i]]},{i,1,9}],{{1,36.09}}];b1=ListPlot[{Px,Py},PlotMarkers->{Style[\"◆\", 18,Red],Style[\"◆\", 18,Blue]},Frame->True,GridLines->Automatic,PlotStyle->{{Thick,Red},{Thick,Blue}},AspectRatio->1];\nlG1=Table[a[[i]][[4]],{i,1,9}];\nlG2=Table[a[[i]][[5]],{i,1,9}];\nLG1vsx1=Table[{xed[[i]],lG1[[i]]},{i,1,9}];\nLG2vsx1=Table[{xed[[i]],lG2[[i]]},{i,1,9}];\nGEoverRTx1x2=Table[a[[i]][[7]],{i,1,9}];\nGEoverRTx1x2vsx1=Table[{xed[[i]],GEoverRTx1x2[[i]]},{i,1,9}];\n\nsol=FindFit[GEoverRTx1x2vsx1,A21 x1+A12 (1-x1),{A12,A21},x1];\n\nG1[x_]:=Exp[(1-x)^2 (A12+2 (A21-A12) x)]/.sol;\nG2[x_]:=Exp[x^2 (A21+2 (A12-A21) (1-x))]/.sol;\ni=0;\nT=50+273.15;\nPS1=36.09;\nPS2=12.30;\nWhile[i< 101,{x=i 0.01,xe[i]=x,sole=FindRoot[G1[x] x PS1+G2[x] (1-x) PS2== Pc,{Pc, 100}],Pes[i]=(Pc/.sole),ye[i]=PS1 G1[x] xe[i]/Pc/.sole,i++}];\nTbl1=Table[{xe[i],Pes[i]},{i,0,100}];\nTbl2=Table[{ye[i],Pes[i]},{i,0,100}];\nplt1=ListPlot[Tbl1, PlotStyle->{Thick,RGBColor[1,0,0]},Joined-> True,GridLines-> Automatic,Frame-> True,AspectRatio-> 1,PlotRange-> {{0,1},Automatic}];\nplt2=ListPlot[Tbl2, PlotStyle->{Thick,RGBColor[0,0,1]},Joined-> True,GridLines-> Automatic,Frame-> True,AspectRatio-> 1,PlotRange-> {{0,1},Automatic}];\nb2=Show[plt1,plt2];\nTble=Table[{xe[i],ye[i]},{i,0,100}];\nTbled=Table[{xed[[i]],yed[[i]]},{i,1,9}];\nplt2=Show[ListPlot[Tble,PlotStyle->{Thick,RGBColor[0,1,0]},Joined-> True,GridLines-> Automatic,Frame-> True,AspectRatio-> 1,PlotRange-> {{0,1},{0,1}},Epilog-> {Black,Line[{{0,0},{1,1}}]}],ListPlot[Tbled,PlotMarkers->{Style[\"◆\", 18,Green]},Frame->True,GridLines->Automatic,PlotStyle->{{Thick,Red},{Thick,Blue}},AspectRatio->1]];\n\nGEoverRTx1x2=Table[a[[i]][[7]],{i,1,9}];\nGEoverRT=Table[a[[i]][[6]],{i,1,9}];\na1=ListPlot[Table[{xed[[i]],(Log[G1[xed[[i]]]]-Log[G2[xed[[i]]]])-(lG1[[i]]-lG2[[i]])},{i,1,9}],GridLines-> Automatic,Frame-> True,AspectRatio-> 1,PlotRange-> {{0,1},{-0.3,0.3}},PlotMarkers->Style[\"●\", 16, Red]];\nQuiet@Table[{xed[[i]],(xed[[i]]Log[G1[xed[[i]]]]+(1-xed[[i]])Log[G2[xed[[i]]]])-GEoverRT[[i]]},{i,1,9}]//TableForm;\na2=ListPlot[Table[{xed[[i]],(xed[[i]]Log[G1[xed[[i]]]]+(1-xed[[i]])Log[G2[xed[[i]]]])-GEoverRT[[i]]},{i,1,9}],GridLines-> Automatic,Frame-> True,AspectRatio-> 1,PlotRange-> {{0,1},{-0.3,0.3}},PlotMarkers->Style[\"○\", 18,Blue]];},\n\n3,\n{a={{17.51`,0.0932`,0.1794`,-0.722`,0.004`,-0.064`,-0.758`},{18.15`,0.1248`,0.2383`,-0.694`,0.`,-0.086`,-0.79`},{19.3`,0.1757`,0.3302`,-0.648`,-0.007`,-0.12`,-0.825`},{19.89`,0.2`,0.3691`,-0.636`,-0.007`,-0.133`,-0.828`},{21.37`,0.2626`,0.4628`,-0.611`,-0.014`,-0.171`,-0.882`},{24.95`,0.3615`,0.6184`,-0.486`,-0.057`,-0.212`,-0.919`},{29.82`,0.475`,0.7552`,-0.38`,-0.127`,-0.248`,-0.992`},{34.8`,0.5555`,0.8378`,-0.279`,-0.218`,-0.252`,-1.019`},{42.1`,0.6718`,0.9137`,-0.192`,-0.355`,-0.245`,-1.113`},{60.38`,0.8782`,0.986`,-0.023`,-0.824`,-0.12`,-1.124`},{65.39`,0.9398`,0.9945`,-0.002`,-0.972`,-0.061`,-1.074`}};\nPe=Table[a[[i]][[1]],{i,1,11}];\nxed=Table[a[[i]][[2]],{i,1,11}];\nyed=Table[a[[i]][[3]],{i,1,11}];\nPx=Join[{{0,15.79}},Table[{xed[[i]],Pe[[i]]},{i,1,11}],{{1,69.36}}];Py=Join[{{0,15.79}},Table[{yed[[i]],Pe[[i]]},{i,1,11}],{{1,69.36}}];b1=ListPlot[{Px,Py},PlotMarkers->{Style[\"◆\", 18,Red],Style[\"◆\", 18,Blue]},Frame->True,GridLines->Automatic,PlotStyle->{{Thick,Red},{Thick,Blue}},AspectRatio->1];lG1=Table[a[[i]][[4]],{i,1,11}];\nlG2=Table[a[[i]][[5]],{i,1,11}];\nLG1vsx1=Table[{xed[[i]],lG1[[i]]},{i,1,11}];\nLG2vsx1=Table[{xed[[i]],lG2[[i]]},{i,1,11}];\nGEoverRTx1x2=Table[a[[i]][[7]],{i,1,11}];\nGEoverRTx1x2vsx1=Table[{xed[[i]],GEoverRTx1x2[[i]]},{i,1,11}];\nsol=FindFit[GEoverRTx1x2vsx1,A21 x1+A12 (1-x1),{A12,A21},x1];\nG1[x_]:=Exp[(1-x)^2 (A12+2 (A21-A12) x)]/.sol;\nG2[x_]:=Exp[x^2 (A21+2 (A12-A21) (1-x))]/.sol;\ni=0;\nT=50+273.15;\nPS1=69.36;\nPS2=15.79;\nWhile[i< 101,{x=i 0.01,xe[i]=x,sole=FindRoot[G1[x] x PS1+G2[x] (1-x) PS2== Pc,{Pc, 100}],Pes[i]=(Pc/.sole),ye[i]=PS1 G1[x] xe[i]/Pc/.sole,i++}];\nTbl1=Table[{xe[i],Pes[i]},{i,0,100}];\nTbl2=Table[{ye[i],Pes[i]},{i,0,100}];\nplt1=ListPlot[Tbl1, PlotStyle->{Thick,RGBColor[1,0,0]},Joined-> True,GridLines-> Automatic,Frame-> True,AspectRatio-> 1,PlotRange-> {{0,1},Automatic}];plt2=ListPlot[Tbl2, PlotStyle->{Thick,RGBColor[0,0,1]},Joined-> True,GridLines-> Automatic,Frame-> True,AspectRatio-> 1,PlotRange-> {{0,1},Automatic}];\nb2=Show[plt1,plt2];\nTble=Table[{xe[i],ye[i]},{i,0,100}];\nTbled=Table[{xed[[i]],yed[[i]]},{i,1,11}];\nplt2=Show[ListPlot[Tble,PlotStyle->{Thick,RGBColor[0,1,0]},Joined-> True,GridLines-> Automatic,Frame-> True,AspectRatio-> 1,PlotRange-> {{0,1},{0,1}},Epilog-> {Black,Line[{{0,0},{1,1}}]}],ListPlot[Tbled,PlotMarkers->{Style[\"◆\", 18,Green]},Frame->True,GridLines->Automatic,PlotStyle->{{Thick,Red},{Thick,Blue}},AspectRatio->1]];\n\nGEoverRTx1x2=Table[a[[i]][[7]],{i,1,11}];\nGEoverRT=Table[a[[i]][[6]],{i,1,11}];a1=ListPlot[Table[{xed[[i]],(Log[G1[xed[[i]]]]-Log[G2[xed[[i]]]])-(lG1[[i]]-lG2[[i]])},{i,1,11}],GridLines-> Automatic,Frame-> True,AspectRatio-> 1,PlotRange-> {{0,1},{-0.3,0.3}},PlotMarkers->Style[\"●\", 16, Red]];Quiet@Table[{xed[[i]],(xed[[i]]Log[G1[xed[[i]]]]+(1-xed[[i]])Log[G2[xed[[i]]]])-GEoverRT[[i]]},{i,1,11}]//TableForm;\n\na2=ListPlot[Table[{xed[[i]],(xed[[i]]Log[G1[xed[[i]]]]+(1-xed[[i]])Log[G2[xed[[i]]]])-GEoverRT[[i]]},{i,1,11}],GridLines-> Automatic,Frame-> True,AspectRatio-> 1,PlotRange-> {{0,1},{-0.3,0.3}},PlotMarkers->Style[\"○\", 18,Blue]];},\n\n4,\n{a={{91.78`,0.063`,0.049`,0.901`,0.033`,1.481`},{88.01`,0.248`,0.131`,0.472`,0.121`,1.114`},{81.67`,0.372`,0.182`,0.321`,0.166`,0.955`},{78.89`,0.443`,0.215`,0.278`,0.21`,0.972`},{76.82`,0.508`,0.248`,0.257`,0.264`,1.043`},{73.39`,0.561`,0.268`,0.19`,0.306`,0.977`},{66.45`,0.64`,0.316`,0.123`,0.337`,0.869`},{62.95`,0.702`,0.368`,0.129`,0.393`,0.993`},{57.7`,0.763`,0.412`,0.072`,0.462`,0.909`},{50.16`,0.834`,0.49`,0.016`,0.536`,0.74`},{45.7`,0.874`,0.57`,0.027`,0.548`,0.844`}};\nPe=Table[a[[i]][[1]],{i,1,11}];\nxed=Table[a[[i]][[2]],{i,1,11}];\nyed=Table[a[[i]][[3]],{i,1,11}];\nPx=Join[{{0,90.15}},Table[{xed[[i]],Pe[[i]]},{i,1,11}],{{1,29.00}}];\nPy=Join[{{0,90.15}},Table[{yed[[i]],Pe[[i]]},{i,1,11}],{{1,29.00}}];b1=ListPlot[{Px,Py},PlotMarkers->{Style[\"◆\", 18,Red],Style[\"◆\", 18,Blue]},Frame->True,GridLines->Automatic,PlotStyle->{{Thick,Red},{Thick,Blue}},AspectRatio->1];\nlG1=Table[a[[i]][[4]],{i,1,11}];\nlG2=Table[a[[i]][[5]],{i,1,11}];\nLG1vsx1=Table[{xed[[i]],lG1[[i]]},{i,1,11}];\nLG2vsx1=Table[{xed[[i]],lG2[[i]]},{i,1,11}];\nGEoverRTx1x2=Table[a[[i]][[6]],{i,1,11}];\nGEoverRTx1x2vsx1=Table[{xed[[i]],GEoverRTx1x2[[i]]},{i,1,11}];\n\nsol=FindFit[GEoverRTx1x2vsx1,A21 x1+A12 (1-x1),{A12,A21},x1];\n\nG1[x_]:=Exp[(1-x)^2 (A12+2 (A21-A12) x)]/.sol;\nG2[x_]:=Exp[x^2 (A21+2 (A12-A21) (1-x))]/.sol;\ni=0;\nT=65+273.15;\nPS1=29.00;\nPS2=90.15;\nWhile[i< 101,{x=i 0.01,xe[i]=x,sole=FindRoot[G1[x] x PS1+G2[x] (1-x) PS2== Pc,{Pc, 100}],Pes[i]=(Pc/.sole),ye[i]=PS1 G1[x] xe[i]/Pc/.sole,i++}];\nTbl1=Table[{xe[i],Pes[i]},{i,0,100}];\nTbl2=Table[{ye[i],Pes[i]},{i,0,100}];\nplt1=ListPlot[Tbl1, PlotStyle->{Thick,RGBColor[1,0,0]},Joined-> True,GridLines-> Automatic,Frame-> True,AspectRatio-> 1,PlotRange-> {{0,1},Automatic}];plt2=ListPlot[Tbl2, PlotStyle->{Thick,RGBColor[0,0,1]},Joined-> True,GridLines-> Automatic,Frame-> True,AspectRatio-> 1,PlotRange-> {{0,1},Automatic}];\nb2=Show[plt1,plt2];\nTble=Table[{xe[i],ye[i]},{i,0,100}];\nTbled=Table[{xed[[i]],yed[[i]]},{i,1,11}];\nplt2=Show[ListPlot[Tble,PlotStyle->{Thick,RGBColor[0,1,0]},Joined-> True,GridLines-> Automatic,Frame-> True,AspectRatio-> 1,PlotRange-> {{0,1},{0,1}},Epilog-> {Black,Line[{{0,0},{1,1}}]}],ListPlot[Tbled,PlotMarkers->{Style[\"◆\", 18,Green]},Frame->True,GridLines->Automatic,PlotStyle->{{Thick,Red},{Thick,Blue}},AspectRatio->1]];\n\nGEoverRTx1x2=Table[a[[i]][[6]],{i,1,11}];\nGEoverRT=Table[a[[i]][[6]] xed[[i]] (1-xed[[i]]),{i,1,11}];a1=ListPlot[Table[{xed[[i]],(Log[G1[xed[[i]]]]-Log[G2[xed[[i]]]])-(lG1[[i]]-lG2[[i]])},{i,1,11}],GridLines-> Automatic,Frame-> True,AspectRatio-> 1,PlotRange-> {{0,1},{-0.3,0.3}},PlotMarkers->Style[\"●\", 16, Red]];Quiet@Table[{xed[[i]],(xed[[i]]Log[G1[xed[[i]]]]+(1-xed[[i]])Log[G2[xed[[i]]]])-GEoverRT[[i]]},{i,1,11}]//TableForm;\na2=ListPlot[Table[{xed[[i]],(xed[[i]]Log[G1[xed[[i]]]]+(1-xed[[i]])Log[G2[xed[[i]]]])-GEoverRT[[i]]},{i,1,11}],GridLines-> Automatic,Frame-> True,AspectRatio-> 1,PlotRange-> {{0,1},{-0.3,0.3}},PlotMarkers->Style[\"○\", 18,Blue]];}];\n\n\nSwitch[ctrl1,1,Switch[ctrl,1,Show[plt1,plt2,plt1e,plt2e,FrameLabel-> {Style[\"acetone liquid and vapor‐phase mole fractions\",10],Style[\"pressure in kPa\",10]},ImageSize-> 400{1,1}],2,Show[pltxy,pltxye,FrameLabel-> {Style[\"acetone liquid‐phase mole fraction\",10],Style[\"acetone vapor‐phase mole fraction\",10]},ImageSize-> 400{1,1}],3,Show[a1,a2,FrameLabel-> {Style[\"acetone liquid‐phase mole fraction\",10],Style[Row[{TraditionalForm[δ Log[Subscript[γ, 1]/Subscript[γ, 2]]],\" and \",TraditionalForm[δ ((G)^(E)/(R Style[\"T\",Italic]))]}],10]},ImageSize-> 400{1,1},Epilog-> {Green,Thickness[0.0125],Line[{{0,0.1},{1,0.1}}],Line[{{0,-0.1},{1,-0.1}}]}]],\n\n2,Switch[ctrl,1,Show[b1,b2,FrameLabel-> {Style[\"MEK liquid and vapor‐phase mole fractions\",10],Style[\"pressure  in  kPa\",10]},ImageSize-> 400{1,1}],2,Show[plt2,FrameLabel-> {Style[\"MEK liquid‐phase mole fraction\",10],Style[\"MEK vapor‐phase  mole  fraction\",10]},ImageSize-> 400{1,1}],3,Show[a1,a2,FrameLabel-> {Style[\"MEK liquid‐phase mole fraction\",10],Style[Row[{TraditionalForm[δ Log[Subscript[γ, 1]/Subscript[γ, 2]]],\" and \",TraditionalForm[δ ((G)^(E)/(R Style[\"T\",Italic]))]}],10]},ImageSize-> 400{1,1},Epilog-> {Green,Thickness[0.0125],Line[{{0,0.1},{1,0.1}}],Line[{{0,-0.1},{1,-0.1}}]}]],\n\n3,Switch[ctrl,1,Show[b1,b2,FrameLabel-> {Style[\"chloroform liquid and vapor‐phase mole fractions\",10],Style[\"pressure in kPa\",10]},ImageSize-> 400{1,1}],2,Show[plt2,FrameLabel-> {Style[\"chloroform liquid‐phase mole fraction\",10],Style[\"chloroform vapor‐phase mole fraction\",10]},ImageSize-> 400{1,1}],3,Show[a1,a2,FrameLabel-> {Style[\"chloroform liquid‐phase mole fraction\",10],Style[Row[{TraditionalForm[δ Log[Subscript[γ, 1]/Subscript[γ, 2]]],\" and \",TraditionalForm[δ ((G)^(E)/(R Style[\"T\",Italic]))]}],10]},ImageSize-> 400{1,1},Epilog-> {Green,Thickness[0.0125],Line[{{0,0.1},{1,0.1}}],Line[{{0,-0.1},{1,-0.1}}]}]],\n\n4,Switch[ctrl,1,Show[b1,b2,FrameLabel-> {Style[\"diethyl ketone liquid and vapor‐phase mole fractions\",10],Style[\"pressure  in  kPa\",10]},ImageSize-> 400{1,1}],2,Show[plt2,FrameLabel-> {Style[\"diethyl ketone liquid‐phase mole fraction\",10],Style[\"diethyl ketone vapor‐phase mole fraction\",10]},ImageSize-> 400{1,1}],3,Show[a1,a2,FrameLabel-> {Style[\"diethyl ketone liquid‐phase mole fraction\",10],Style[Row[{TraditionalForm[δ Log[Subscript[γ, 1]/Subscript[γ, 2]]],\" and \",TraditionalForm[δ ((G)^(E)/(R Style[\"T\",Italic]))]}],10]},ImageSize-> 400{1,1},Epilog-> {Green,Thickness[0.0125],Line[{{0,0.1},{1,0.1}}],Line[{{0,-0.1},{1,-0.1}}]}]]]],\n\n{{ctrl1,1,\"\"},{1-> \"acetone‐methanol \\nat 58 °C\",2-> \"MEK‐toluene \\nat 50° C\",3-> \"chloroform‐1,4‐dioxane \\nat 50 °C\",4-> \"diethyl ketone‐n‐hexane \\nat 65° C\"},ControlType-> PopupMenu},\n{{ctrl,1,\"\"},{1-> \"isothermal VLE diagram\",2-> \"VLE data\",3-> \"consistency test\"},ControlType-> PopupMenu},\n\" \",Style[\"white noise for mole fractions\",Bold],\nColumn[{Control[{{Wnoise1,0.01,\"  liquid\"},0,0.05,0.01,Appearance-> \"Labeled\", ImageSize-> Tiny,Enabled-> If[ctrl1== 1,True,False]}],Spacer[20],\nControl[{{Wnoise2,0.01,\"  vapor\"},0,0.05,0.01,Appearance-> \"Labeled\", ImageSize-> Tiny,Enabled-> If[ctrl1== 1,True,False]}]}],\n\"  \",Style[\"systematic error for mole fractions\",Bold],\nColumn[{Control[{{SysE1,0.01,\"  liquid\"},0,0.05,0.01,Appearance-> \"Labeled\", ImageSize-> Tiny,Enabled-> If[ctrl1== 1,True,False]}],Spacer[20],\nControl[{{SysE2,0.01,\"  vapor\"},0,0.05,0.01,Appearance-> \"Labeled\", ImageSize-> Tiny,Enabled-> If[ctrl1== 1,True,False]}]}],\n\nTrackedSymbols:> {ctrl1,ctrl,SysE1,SysE2,Wnoise1,Wnoise2},\nControlPlacement->Left]"], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`ctrl1$$ = 1}, \"\\[Ellipsis]\"]"], "Output"]
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
      "the equilibrium must solve: {:?}",
      widget.error
    );
    assert!(widget.graphics_handle.is_some(), "the diagram must draw");

    // Two popup menus over the mixtures and the views, then the noise and
    // systematic-error sliders under their headings.
    let names: Vec<&str> = widget
      .controls
      .iter()
      .filter_map(|c| match c {
        manipulate::ControlState::Discrete { name, .. }
        | manipulate::ControlState::Continuous { name, .. } => {
          Some(name.as_str())
        }
        _ => None,
      })
      .collect();
    assert_eq!(
      names,
      ["ctrl1", "ctrl", "Wnoise1", "Wnoise2", "SysE1", "SysE2"]
    );

    // The mixtures other than the first are free of the noise term, so
    // their renders are reproducible.
    let render = |mixture: u32, view: u32| {
      woxi::interpret_with_stdout(&format!(
        "ctrl1 = {mixture}; ctrl = {view}; Wnoise1 = 0.01; Wnoise2 = 0.01; \
         SysE1 = 0.01; SysE2 = 0.01;\n{}",
        widget.body
      ))
      .expect("the body must render")
      .graphics
      .expect("the body must produce a graphic")
    };
    // Every occurrence of a marker glyph is one drawn data point: the
    // notebook's labels use none of them.
    let glyphs = |svg: &str, glyph: &str| svg.matches(glyph).count();

    // The pressure diagram marks the 9 measured points of the MEK-toluene
    // mixture plus the two pure-component ends, on both the liquid and the
    // vapour branch, with red and blue diamonds.
    let pressure = render(2, 1);
    assert_eq!(glyphs(&pressure, "\u{25c6}"), 22, "{pressure}");
    // The x-y diagram of the chloroform mixture marks its 11 points in
    // green over the computed curve.
    let xy = render(3, 2);
    assert_eq!(glyphs(&xy, "\u{25c6}"), 11, "{xy}");
    // The consistency test draws one filled red and one open blue circle
    // per point, inside the green +/-0.1 band the Epilog draws.
    let test = render(4, 3);
    assert_eq!(glyphs(&test, "\u{25cf}"), 11, "{test}");
    assert_eq!(glyphs(&test, "\u{25cb}"), 11, "{test}");
    assert_eq!(
      test.matches("stroke=\"rgb(0,255,0)\"").count(),
      2,
      "both edges of the acceptance band: {test}"
    );

    // Both controls change the picture.
    assert_ne!(pressure, render(3, 1), "the mixture control must matter");
    assert_ne!(pressure, xy, "the view control must matter");
  }

  /// The Demonstrations shape that offers several views of one computation:
  /// the body builds every plot, then a setter bar picks which to show. The
  /// widget must display the *picked* view — it used to show whichever plot
  /// the body drew last, so the setter looked like it did nothing. The
  /// stability-diagram view also exercises `ContourStyle`, `FrameLabel` and
  /// `Epilog` on a `ContourPlot`, and `ImageSize -> n {1, 1}`.
  #[test]
  fn view_switching_manipulate_shows_the_picked_view() {
    let code = "Manipulate[\n\
      Module[{p1, p2, p3},\n\
       p1 = Plot[Sin[k x], {x, 0, 10}, Frame -> True, \
         ImageSize -> 300 {1, 1}];\n\
       p2 = ParametricPlot[{Cos[k t], Sin[t]}, {t, 0, 2 Pi}, Frame -> True, \
         ImageSize -> 320 {1, 1}];\n\
       p3 = ContourPlot[y == k x, {x, 0, 4}, {y, 0, 4}, \
         ContourStyle -> {Thick, Blue}, ImageSize -> 340 {1, 1}, \
         FrameLabel -> {\"rate\", \"level\"}, \
         Epilog -> {Text[\"here\", {2, 3}], Red, PointSize[0.05], \
         Point[{k, 2}]}];\n\
       Switch[view, 1, p1, 2, p2, 3, p3]],\n\
      {{k, 1, \"rate\"}, 1, 3, 0.1, Appearance -> \"Labeled\"},\n\
      {{view, 1, \"\"}, {1 -> \"curve\", 2 -> \"phase\", 3 -> \"diagram\"}},\n\
      TrackedSymbols :> {k, view},\n\
      SynchronousUpdating -> False]";
    let widget = instantiate_stored_manipulate(code, "")
      .expect("the Manipulate must instantiate");
    assert!(
      widget.error.is_none(),
      "body must evaluate cleanly: {:?}",
      widget.error
    );
    assert!(widget.graphics_handle.is_some(), "a view must draw");
    let names: Vec<&str> = widget.controls.iter().map(|c| c.name()).collect();
    assert_eq!(names, vec!["k", "view"]);

    let render = |view: u32| {
      woxi::interpret_with_stdout(&format!(
        "k = 1; view = {view};\n{}",
        widget.body
      ))
      .expect("the body must render")
      .graphics
      .expect("the body must produce a graphic")
    };

    // Each view sizes its own picture, so the width names which one showed.
    for (view, width) in [(1, "300"), (2, "320"), (3, "340")] {
      let svg = render(view);
      assert!(
        svg.starts_with(&format!("<svg width=\"{width}\"")),
        "view {view} must show its own plot, got: {}",
        &svg[..svg.len().min(80)]
      );
    }

    // The diagram view draws through all three of its options.
    let diagram = render(3);
    assert!(
      diagram.contains("stroke=\"rgb(0,0,255)\""),
      "ContourStyle must colour the boundary"
    );
    for text in [">rate</text>", ">level</text>", ">here</text>"] {
      assert!(diagram.contains(text), "missing {text} in the diagram view");
    }
    assert!(
      diagram.contains("rgb(255,0,0)"),
      "the Epilog marker must be drawn"
    );
  }

  #[test]
  fn spectrum_band_manipulate_builds_with_nested_animate_and_graphics_row() {
    // Checked a randomly-sampled Wolfram Demonstrations Project notebook
    // (a wavelength-band visualizer) against Woxi Studio's Manipulate
    // pipeline. Its shape: a `PopupMenu`-controlled mode switch plus a
    // discrete-list "level" control drive a `Text[Which[...]]` body whose
    // branches are `Pane[Column[{...}]]` layouts mixing a `Module`-built
    // indicator bar (`Riffle`/`Partition`/`ConstantArray`/`Table` of
    // `Rectangle`s), a `Dynamic`-wrapped nested `Animate` showing an
    // orbiting `Disk`, and a `GraphicsRow` of two `Graphics` built from
    // `Table`-generated `Arrow`s `Append`ed onto a shared mark list. Colors
    // come from a piecewise `RGBColor` built with chained `If`.
    //
    // This is a self-authored construct-equivalent example, not the
    // notebook's own code or wording.
    woxi::interpret(
      "demoBandColor[v_] := If[v < 2, RGBColor[1, 0, 0], \
       If[v < 4, RGBColor[0, 1, 0], \
       If[v < 6, RGBColor[0, 0, 1], RGBColor[0, 0, 0]]]]; \
       demoVals = {1.5, 3.0, 5.5, 7.0}; \
       demoMarks = {{Line[{{0, -10}, {20, -10}}], \
         Inset[Style[\"A\", 8], {21, -10}]}, \
        {Line[{{0, -20}, {20, -20}}], Inset[Style[\"B\", 8], {21, -20}]}}; \
       demoBand[x_] := Module[{grid, pos}, \
         grid = Table[k, {k, 0, 8, 0.5}]; \
         pos = Table[Part[Position[grid, Part[x, k]], 1, 1], \
           {k, 1, Length[x]}]; \
         Graphics[Join[ \
           Partition[Riffle[ConstantArray[Black, 17], \
             Table[Rectangle[{k - 0.25, 0}, {k, 10}], {k, 0.5, 8.5, 0.5}]], \
            2], \
           Table[{demoBandColor[Part[x, k]], \
             Rectangle[{Part[pos, k]*0.5 - 0.5, 0}, \
              {Part[pos, k]*0.5 + 0.5, 10}]}, {k, 1, Length[x]}]], \
          ImageSize -> Large]]; \
       demoRing[n_] := Graphics[{Circle[{0, 0}, n*3], \
         Text[ToString[n], {0, n*3 + 2}]}]; \
       demoOrbit[n_] := Animate[ \
         Pane[Show[ \
           Graphics[{White, Opacity[0], Rectangle[{-40, -40}, {40, 40}]}], \
           demoRing[n], \
           Graphics[{demoBandColor[n], Arrow[{{4, 0}, {n^2/4, 0}}]}], \
           Graphics[{Blue, Disk[{n, 0}, 1.5]}]], {200, 180}], \
         {t, 0, 6 Pi, ControlPlacement -> Top}, \
         AnimationRunning -> False, Alignment -> Center]; \
       demoArrow1[i_, j_] := {demoBandColor[i], Arrowheads[0.075], \
         Arrow[{{15 - j*i, 0 - 150/i^2}, {15 - j*i, 0 - 150/4}}]}; \
       demoArrow2[i_, j_] := {demoBandColor[i], Arrowheads[0.075], \
         Arrow[{{15 - j*i, 0 - 150/4}, {15 - j*i, 0 - 150/i^2}}]};",
    )
    .expect("the helper definitions must evaluate");

    let expr = woxi::interpret_to_expr(
      "Manipulate[ \
         Text[Which[ \
           demoMode == 1, \
           Pane[Column[{\"low band\", demoBand[demoVals], \"single value\", \
              Dynamic[demoBandColor[demoLevel - 2]], \
              Row[{Column[{\"levels\", \
                  Pane[Dynamic[demoRing[demoLevel]], {200, 250}]}, \
                 Alignment -> Center], \
                Column[{\"orbit\", Dynamic[demoOrbit[demoLevel]]}, \
                 Alignment -> Center]}]}, Alignment -> Center], {600, 420}], \
           demoMode == 2, \
           Pane[Column[{\"high band\", demoBand[demoVals]}, \
             Alignment -> Center], {600, 420}], \
           demoMode == 3, \
           Pane[Column[{\"low band\", demoBand[demoVals], \"high band\", \
              demoBand[demoVals], \
              GraphicsRow[{ \
                Graphics[{Append[ \
                  Table[demoArrow1[demoLevel, 2], {demoLevel, 3, 6}], \
                  demoMarks]}], \
                Graphics[{Append[ \
                  Table[demoArrow2[demoLevel, 2], {demoLevel, 3, 6}], \
                  demoMarks]}]}, Spacings -> 300, \
               ImageSize -> {400, 400*2/3}]}, Alignment -> Center], \
            {600, 420}]]], \
         {{demoMode, 1, \"display\"}, \
          {1 -> \"low\", 2 -> \"high\", 3 -> \"both\"}, \
          ControlType -> PopupMenu}, \
         {{demoLevel, 3, Row[{\"level \", Style[\"n\", Italic]}]}, \
          {3, 4, 5, 6}}]",
    )
    .expect("the Manipulate source must parse and evaluate");
    let state = manipulate::ManipulateState::from_expr(&expr)
      .expect("the Manipulate must build a widget");

    assert!(
      state.error.is_none(),
      "initial body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the initial Which branch must render a graphic"
    );
    let names: Vec<&str> = state.controls.iter().map(|c| c.name()).collect();
    assert_eq!(names, vec!["demoMode", "demoLevel"]);
    match &state.controls[0] {
      manipulate::ControlState::Discrete { popup, .. } => {
        assert!(*popup, "ControlType -> PopupMenu must force the dropdown");
      }
      other => panic!("expected demoMode as a Discrete control: {other:?}"),
    }

    // Switching to the GraphicsRow branch (mode 3) must still render, and
    // the nested-Animate branch (mode 1, already the default) plus the
    // high-band-only branch (mode 2) must each render too.
    for mode in [1, 2, 3] {
      let render_code =
        format!("demoMode = {mode}; demoLevel = 4;\n{}", state.body);
      let render = woxi::interpret_with_stdout(&render_code)
        .unwrap_or_else(|e| panic!("mode {mode} must render: {e:?}"));
      assert!(
        render.graphics.is_some(),
        "mode {mode} must produce a graphic"
      );
    }
  }

  #[test]
  fn wave_pulse_manipulate_tracks_dynamic_animator_bound_and_which_regions() {
    // Checked a randomly-sampled Wolfram Demonstrations Project notebook
    // (a two-medium wave-pulse animation) against Woxi Studio's Manipulate
    // pipeline. Its shape: two `Appearance -> "Labeled"` sliders feed a
    // derived reflection/transmission pair, hidden `ControlType -> None`
    // state variables expose that pair to the body, and a `ControlType ->
    // Animator` time control whose *upper bound* is itself an `If` on one
    // of the sliders (so the animator's range must track a sibling
    // control's live value, not just its initial one) drives a `Which`
    // body that mixes bare comparisons with compound `a < t <= b`
    // (`Inequality`) conditions across four `Graphics` branches.
    //
    // This is a self-authored construct-equivalent example, not the
    // notebook's own code or wording.
    let expr = woxi::interpret_to_expr(
      "Manipulate[ \
         waveA = 1.0; \
         waveB = ratioD*ratioV; \
         waveR = (waveB - waveA)/(waveB + waveA); \
         waveT = 2*waveA/(waveB + waveA); \
         waveNull = Line[{{0, 0}, {0, 0}}]; \
         Which[ \
           waveT0 <= 4, \
           Graphics[{Line[{{-5, 0}, {5, 0}}], \
             Line[{{-5 + waveT0, 0}, {-5 + waveT0, 1}}]}, \
            ImageSize -> {400, 250}], \
           4 < waveT0 <= 4.5, \
           Graphics[{Line[{{-5, 0}, {5, 0}}], \
             Line[{{4 - waveT0, 1}, {4 - waveT0, 1 + waveR}}]}, \
            ImageSize -> {400, 250}], \
           4.5 < waveT0 <= 5, \
           Graphics[{Line[{{-5, 0}, {5, 0}}], \
             Line[{{waveT0 - 5, waveR}, {waveT0 - 5, 1 + waveR}}]}, \
            ImageSize -> {400, 250}], \
           waveT0 >= 5, \
           Graphics[{Line[{{-5, 0}, {5, 0}}], \
             If[waveT0 < 10, \
               Line[{{5 - waveT0, waveR}, {5 - waveT0, 0}}], waveNull]}, \
            ImageSize -> {400, 250}]], \
         {{ratioD, 1.5, \"density ratio\"}, 0, 2, Appearance -> \"Labeled\"}, \
         {{ratioV, 0.5, \"velocity ratio\"}, 0, 2, Appearance -> \"Labeled\"}, \
         {{waveT0, 0, \"time\"}, 0, \
          If[ratioV > 0, 5 + (5 - ratioV)/ratioV, 9], \
          ControlType -> Animator, AnimationRunning -> False}, \
         {{waveA, 1.0}, ControlType -> None}, \
         {waveB, ControlType -> None}, \
         {waveR, ControlType -> None}, \
         {waveT, ControlType -> None}]",
    )
    .expect("the Manipulate source must parse and evaluate");
    let state = manipulate::ManipulateState::from_expr(&expr)
      .expect("the Manipulate must build a widget");

    assert!(
      state.error.is_none(),
      "initial body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the initial Which branch must render a graphic"
    );
    let names: Vec<&str> = state.controls.iter().map(|c| c.name()).collect();
    assert_eq!(names, vec!["ratioD", "ratioV", "waveT0"]);
    match &state.controls[2] {
      manipulate::ControlState::Continuous { min, max, .. } => {
        // ratioV starts at 0.5: max = 5 + (5 - 0.5)/0.5 = 14.
        assert_eq!(*min, 0.0);
        assert!(
          (*max - 14.0).abs() < 1e-9,
          "the animator's max must resolve the If against ratioV's \
           initial value, got {max}"
        );
      }
      other => panic!("expected waveT0 as a Continuous control: {other:?}"),
    }

    // Each Which region (plain `<=`, both halves of the compound
    // `4 < t <= 4.5` / `4.5 < t <= 5`, and `>= 5`) must render.
    for wave_t0 in [0.0, 2.0, 4.25, 4.75, 6.0] {
      let render_code = format!(
        "ratioD = 1.5; ratioV = 0.5; waveT0 = {wave_t0};\n{}",
        state.body
      );
      let render = woxi::interpret_with_stdout(&render_code)
        .unwrap_or_else(|e| panic!("time {wave_t0} must render: {e:?}"));
      assert!(
        render.graphics.is_some(),
        "time {wave_t0} must produce a graphic"
      );
    }

    // Changing the density-ratio slider (the sibling the animator's upper
    // bound depends on) must move the resolved bound on re-evaluation, not
    // just at construction time.
    let mut state = state;
    if let manipulate::ControlState::Continuous { current, .. } =
      &mut state.controls[1]
    {
      *current = 1.0;
    }
    state.reevaluate();
    match &state.controls[2] {
      manipulate::ControlState::Continuous { max, .. } => {
        // ratioV = 1.0: max = 5 + (5 - 1)/1 = 9.
        assert!(
          (*max - 9.0).abs() < 1e-9,
          "the animator's max must track ratioV's live value, got {max}"
        );
      }
      other => panic!("expected waveT0 as a Continuous control: {other:?}"),
    }
  }

  /// End-to-end regression for the shape of the "Generating a Rotating
  /// Field by Superposition of Three Alternating Fields" Demonstration: a
  /// stored Manipulate whose `SaveDefinitions -> True` Initialization
  /// defines a family of pattern-matched helper functions built from
  /// complex exponentials (`Exp[I …]`, `Re`, `Im`, `Abs`), a `Module` body
  /// that combines a `ParametricPlot` with a `Show`'d `Graphics` overlay
  /// of `Dashed` `Circle`s and `Arrow`s, laid out side by side with
  /// `GraphicsRow`, driven by three `Appearance -> "Labeled"` sliders.
  #[test]
  fn demonstration_three_phase_field_manipulate_opens_with_its_widget() {
    let nb_src = r##"Notebook[{
Cell[CellGroupData[{
Cell[BoxData["Manipulate[
 Module[{trace, combined},
   trace = ParametricPlot[{Re[totalFieldErr[t, ampErr, phErr]], Im[totalFieldErr[t, ampErr, phErr]]}, {t, 0, 2 Pi}, PlotStyle -> Orange, PlotRange -> 3, Axes -> False];
   combined = Show[{trace, Graphics[{{Orange, Dashed, Circle[{0, 0}, Abs[totalField[angle]]]}, Black, Arrow[{{0, 0}, {Re[totalFieldErr[angle, ampErr, phErr]], Im[totalFieldErr[angle, ampErr, phErr]]}}]}]}, PlotLabel -> \"Combined field\"];
   GraphicsRow[{trace, combined}, ImageSize -> {600, 300}]
 ],
 {{angle, 0, \"phase angle\"}, 0, 2 Pi, 0.01, Appearance -> \"Labeled\"},
 {{ampErr, 0, \"amplitude error\"}, 0, 1, Appearance -> \"Labeled\"},
 {{phErr, 0, \"phase error\"}, 0, 1, Appearance -> \"Labeled\"},
 SaveDefinitions -> True]"], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`angle$$ = 0, $CellContext`ampErr$$ = 0, $CellContext`phErr$$ = 0}, DynamicBox[\[Ellipsis]], Initialization:>(($CellContext`compFwd[
     Pattern[$CellContext`t, Blank[]]] := Cos[$CellContext`t]; $CellContext`shift2 = \
2 (Pi/3); $CellContext`shift3 = 4 (Pi/3); $CellContext`rot2 = \
Exp[I $CellContext`shift2]; $CellContext`rot3 = \
Exp[I $CellContext`shift3]; $CellContext`phase2[
     Pattern[$CellContext`t, Blank[]]] := \
$CellContext`rot2 $CellContext`compFwd[$CellContext`t + $CellContext`shift2]; \
$CellContext`phase3[
     Pattern[$CellContext`t, Blank[]]] := \
$CellContext`rot3 $CellContext`compFwd[$CellContext`t + $CellContext`shift3]; \
$CellContext`totalField[
     Pattern[$CellContext`t, Blank[]]] := \
$CellContext`compFwd[$CellContext`t] + $CellContext`phase2[$CellContext`t] + \
$CellContext`phase3[$CellContext`t]; $CellContext`totalFieldErr[
     Pattern[$CellContext`t, Blank[]], Pattern[$CellContext`ampErr, Blank[]], \
Pattern[$CellContext`phErr, Blank[]]] := (
     1 + $CellContext`ampErr) $CellContext`compFwd[$CellContext`t + \
$CellContext`phErr Pi] + $CellContext`phase2[$CellContext`t] + \
$CellContext`phase3[$CellContext`t]; Null))]"], "Output"]
}, Open]]
}]"##;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);
    let mut widget = editors
      .into_iter()
      .find_map(|e| e.manipulate_state)
      .expect("the stored Manipulate must instantiate on load");
    assert!(
      widget.error.is_none(),
      "body must evaluate cleanly: {:?}",
      widget.error
    );
    assert!(
      widget.graphics_handle.is_some(),
      "the ParametricPlot/Show GraphicsRow must draw"
    );

    match &widget.controls[..] {
      [
        manipulate::ControlState::Continuous {
          name: angle,
          label: angle_label,
          min: angle_min,
          max: angle_max,
          current: angle_now,
          ..
        },
        manipulate::ControlState::Continuous {
          name: amp,
          label: amp_label,
          min: amp_min,
          max: amp_max,
          current: amp_now,
          ..
        },
        manipulate::ControlState::Continuous {
          name: ph,
          label: ph_label,
          min: ph_min,
          max: ph_max,
          current: ph_now,
          ..
        },
      ] => {
        assert_eq!(angle.as_str(), "angle");
        assert_eq!(angle_label.as_str(), "phase angle");
        assert_eq!(*angle_min, 0.0);
        assert!((*angle_max - std::f64::consts::TAU).abs() < 1e-9);
        assert_eq!(*angle_now, 0.0);
        assert_eq!(
          (
            amp.as_str(),
            amp_label.as_str(),
            *amp_min,
            *amp_max,
            *amp_now
          ),
          ("ampErr", "amplitude error", 0.0, 1.0, 0.0)
        );
        assert_eq!(
          (ph.as_str(), ph_label.as_str(), *ph_min, *ph_max, *ph_now),
          ("phErr", "phase error", 0.0, 1.0, 0.0)
        );
      }
      other => panic!("unexpected controls: {other:?}"),
    }

    // The three fields sum to a constant-magnitude phasor rotating with
    // `angle`; moving the slider must move the arrow and so the render.
    // The iced handle doesn't expose its bytes, so re-render the body
    // through the widget's own bindings to inspect the SVG.
    let render = |w: &manipulate::ManipulateState| {
      let bindings: Vec<(String, String)> = w
        .controls
        .iter()
        .filter(|c| c.binds_variable())
        .map(|c| (c.name().to_string(), c.current_code()))
        .collect();
      let code = match w.initialization.as_deref() {
        Some(init) => format!("{init}; {}", w.body),
        None => w.body.clone(),
      };
      woxi::with_scoped_globals(&bindings, || {
        woxi::interpret_with_stdout(&code)
      })
      .expect("body evaluates")
      .graphics
      .expect("the pieces must render")
    };

    let unmoved = render(&widget);
    match &mut widget.controls[0] {
      manipulate::ControlState::Continuous { current, .. } => {
        *current = std::f64::consts::PI / 2.0;
      }
      other => panic!("expected continuous control, got {other:?}"),
    }
    widget.reevaluate();
    assert!(widget.error.is_none());
    assert!(widget.graphics_handle.is_some());
    assert_ne!(
      unmoved,
      render(&widget),
      "rotating the phase angle must change the rendered graphic"
    );
  }

  /// Checked a randomly-sampled Wolfram Demonstrations Project notebook (a
  /// binary-mixture phase-equilibrium diagram visualizer) against Woxi
  /// Studio's Manipulate pipeline. Its shape: a `Module` body defines a
  /// couple of `Exp[…]`-based activity-style helper functions of one
  /// variable, then a `While[i < n, {…, i++}]` loop whose body is a
  /// comma-separated *list* (not a `CompoundExpression`) repeatedly calls
  /// `FindRoot` to solve a nonlinear equation and stores the results into
  /// indexed assignments (`arr[i] = …`); the resulting tables feed
  /// `Interpolation[…, InterpolationOrder -> 1]` to build a few
  /// `InterpolatingFunction`s; a `Switch` on an integer control then picks
  /// between two rendered views — a `Show` of a `Plot` of one interpolated
  /// curve overlaid with a `Graphics` of `Table`-generated tie-line
  /// `Line`s and `Style[Text[Row[{ToString[NumberForm[…]], "…"}], pos,
  /// Automatic, direction], size]` labels tilted to follow each tie line,
  /// versus a `Plot` of another interpolated curve with an `Epilog` marking
  /// a reference point — driven by a rule-list discrete control (`{{ctrl,
  /// 1, ""}, {1 -> "…", 2 -> "…"}}`) and a labeled slider, tracked via
  /// `TrackedSymbols`.
  ///
  /// This is a self-authored, construct-equivalent example (a made-up
  /// binary system with invented coefficients) — not the notebook's own
  /// code, data, or wording, which is copyrighted. It doubles as a
  /// regression test for `Text`'s fourth (`direction`) argument, which
  /// used to be silently dropped instead of tilting the label.
  #[test]
  fn demonstration_phase_diagram_manipulate_solves_and_interpolates_tables() {
    let nb_src = r##"Notebook[{
Cell[CellGroupData[{
Cell[BoxData["Manipulate[
 Module[{i, x, xL, pL, yV, fL, fV, fEQ, tblL, tblV, tblEQ, sol, p, kA, kB, mm, psA, psB, px, py},
  kA = 1.8; kB = 1.3; mm = 0.15;
  psA = 0.82 P; psB = 0.64 P;
  gA[u_] := Exp[kA (1 - u)^2 (1 + mm u)];
  gB[u_] := Exp[kB u^2 (1 - mm (1 - u))];
  i = 0;
  While[i < 21, {x = i*0.05, xL[i] = x, sol = FindRoot[gA[x] x psA + gB[x] (1 - x) psB == p, {p, 500}], pL[i] = (p /. sol), yV[i] = gA[x] psA x/pL[i], i++}];
  tblL = Table[{xL[i], pL[i]}, {i, 0, 20}];
  tblV = Table[{yV[i], pL[i]}, {i, 0, 20}];
  tblEQ = Table[{xL[i], yV[i]}, {i, 0, 20}];
  fL = Interpolation[tblL, InterpolationOrder -> 1];
  fV = Interpolation[tblV, InterpolationOrder -> 1];
  fEQ = Interpolation[tblEQ, InterpolationOrder -> 1];
  px = 0.5; py = fEQ[px];
  Switch[ctrl,
   1,
    Show[
     Plot[fL[xv], {xv, 0, 1}, Frame -> True, FrameLabel -> {Style[\"liquid mole fraction\", 14], Style[\"pressure (mmHg)\", 14]}, GridLines -> Automatic, ImageSize -> {600, 350}, PlotStyle -> Thick, PlotRange -> {{0, 1}, {0, 700}}],
     Graphics[{Thick, Green, Table[Line[{{a, fL[a]}, {fEQ[a], fV[fEQ[a]]}}], {a, 0.1, 0.9, 0.1}], Red, Table[Style[Text[Row[{ToString[NumberForm[fL[a], {4, 2}]], \" mmHg\"}], {(a + fEQ[a])/2, fL[a]}, Automatic, {1000, (fV[fEQ[a]] - fL[a])}], 10], {a, 0.1, 0.9, 0.1}]}]
    ],
   2,
    Plot[fEQ[t], {t, 0, 1}, Epilog -> {Green, Line[{{0, 0}, {1, 1}}], Red, PointSize[0.025], Point[{px, py}]}, ImageSize -> {600, 350}, Frame -> True, GridLines -> Automatic, PlotStyle -> Thick, FrameLabel -> {Style[\"liquid mole fraction\", 14], Style[\"vapor mole fraction\", 14]}]
  ]
 ],
 Control[{{ctrl, 1, \"\"}, {1 -> \"Pressure Curve\", 2 -> \"Equilibrium Curve\"}}],
 {{P, 760, \"system pressure (mmHg)\"}, 50, 1200, 10, Appearance -> \"Labeled\"},
 TrackedSymbols :> {P, ctrl}]"], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`ctrl$$ = 1, $CellContext`P$$ = 760}, DynamicBox[\[Ellipsis]]]"], "Output"]
}, Open]]
}]"##;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);
    let mut widget = editors
      .into_iter()
      .find_map(|e| e.manipulate_state)
      .expect("the stored Manipulate must instantiate on load");
    assert!(
      widget.error.is_none(),
      "body must evaluate cleanly: {:?}",
      widget.error
    );
    assert!(
      widget.graphics_handle.is_some(),
      "the Switch's first branch (Show of Plot + Graphics overlay) must draw"
    );

    match &widget.controls[..] {
      [
        manipulate::ControlState::Discrete {
          name: ctrl_name,
          values: ctrl_values,
          value_labels: ctrl_labels,
          current_index: ctrl_idx,
          ..
        },
        manipulate::ControlState::Continuous {
          name: p_name,
          label: p_label,
          min: p_min,
          max: p_max,
          step: p_step,
          current: p_now,
          ..
        },
      ] => {
        assert_eq!(ctrl_name.as_str(), "ctrl");
        assert_eq!(ctrl_values.as_slice(), ["1", "2"]);
        assert_eq!(
          ctrl_labels.as_slice(),
          ["Pressure Curve", "Equilibrium Curve"]
        );
        assert_eq!(*ctrl_idx, 0, "ctrl starts at its initial value 1");

        assert_eq!(p_name.as_str(), "P");
        assert_eq!(p_label.as_str(), "system pressure (mmHg)");
        assert_eq!(*p_min, 50.0);
        assert_eq!(*p_max, 1200.0);
        assert_eq!(*p_step, 10.0);
        assert_eq!(*p_now, 760.0);
      }
      other => panic!("unexpected controls: {other:?}"),
    }

    // Re-render the extracted body directly to inspect the two Switch
    // branches and the tilted tie-line labels the fourth Text argument
    // draws — the same technique the other demonstration regressions use.
    let render = |w: &manipulate::ManipulateState| {
      let bindings: Vec<(String, String)> = w
        .controls
        .iter()
        .filter(|c| c.binds_variable())
        .map(|c| (c.name().to_string(), c.current_code()))
        .collect();
      woxi::with_scoped_globals(&bindings, || {
        woxi::interpret_with_stdout(&w.body)
      })
      .expect("body evaluates")
      .graphics
      .expect("the pieces must render")
    };
    let pressure_view = render(&widget);
    assert!(
      pressure_view.contains("rotate("),
      "the tie-line labels' direction vector must tilt the text: \
       {pressure_view}"
    );

    // Moving the pressure slider re-solves the `FindRoot`/`Interpolation`
    // table and must change the rendered pressure curve.
    match &mut widget.controls[1] {
      manipulate::ControlState::Continuous { current, .. } => {
        *current = 1100.0;
      }
      other => panic!("expected P as a Continuous control, got {other:?}"),
    }
    widget.reevaluate();
    assert!(widget.error.is_none());
    assert!(widget.graphics_handle.is_some());
    assert_ne!(
      pressure_view,
      render(&widget),
      "raising the system pressure must change the re-solved curve"
    );

    // Switching `ctrl` to the second rule-list choice must render the
    // other Switch branch (the Plot with an Epilog point) instead.
    match &mut widget.controls[0] {
      manipulate::ControlState::Discrete { current_index, .. } => {
        *current_index = 1;
      }
      other => panic!("expected ctrl as a Discrete control, got {other:?}"),
    }
    widget.reevaluate();
    assert!(widget.error.is_none());
    assert!(widget.graphics_handle.is_some());
    assert_ne!(
      pressure_view,
      render(&widget),
      "the Switch must render a different picture for the other choice"
    );
  }

  /// Checked a randomly-sampled Wolfram Demonstrations Project notebook (a
  /// diffusion-driven release-profile visualizer) against Woxi Studio's
  /// Manipulate pipeline. Its shape: a `Setter` swaps between two `Plot`s
  /// selected by `Which`, a `RadioButtonBar` picks a discrete rate
  /// multiplier consumed by another `Which`, a `PaneSelector` shows a
  /// labeled `Slider` only on one of the two panes, an `Animator` drives
  /// elapsed time, and the body's helper functions lean on `Quiet[Sum[...
  /// Exp[...]]]` truncated-series formulas feeding `Plot`s styled with
  /// `PlotTheme`, dashed `PlotStyle`, `LabelStyle`, `FrameLabel`, `Filling`/
  /// `FillingStyle`, wrapped in a `Show` with `Frame`, `LabelStyle`,
  /// `ImageSize`, and `AspectRatio -> Full`.
  ///
  /// This is a self-authored, construct-equivalent example (a made-up
  /// "signal attenuation" scenario) — not the notebook's own code, data, or
  /// wording, which is copyrighted. It doubles as a regression test for the
  /// `LabelStyle` option, which used to be silently dropped by `Plot`.
  #[test]
  fn attenuation_manipulate_switches_panes_and_honors_label_style() {
    let expr = woxi::interpret_to_expr(
      "Manipulate[ \
         Module[{rateB, envFn, spatialFn, curve1, curve2}, \
           rateB = Which[bankPick == 1, 0.5, bankPick == 2, 1, bankPick == 3, 2]; \
           envFn[r_, k_, t_] := Quiet[100 Exp[-t (k Pi/r)^2]]; \
           spatialFn[r_, x_, t_] := Quiet[If[t < 0.01, 100, \
             100 (2/Pi) Sum[((-1)^(m + 1)/m) Sin[m Pi x/r] Exp[-t (m Pi/r)^2], {m, 40}]]]; \
           curve1 = Plot[{envFn[lenA, 4, elapsed], envFn[lenA, rateB, elapsed]}, \
             {elapsed, 0, 30}, PlotTheme -> \"Detailed\", \
             PlotStyle -> {{Gray, Thick}, {Orange, Thick, Dashed}}, \
             LabelStyle -> {17, Black}, PlotLegends -> False, \
             FrameLabel -> {\"elapsed (s)\", \"signal (%)\"}]; \
           curve2 = Plot[spatialFn[lenA, xPos, 0.05], {xPos, 0, lenA}, \
             PlotTheme -> \"Detailed\", PlotStyle -> {Gray, Thick}, \
             Filling -> Axis, FillingStyle -> Directive[Opacity[0.25]], \
             FrameLabel -> {\"position\", \"signal (%)\"}]; \
           Show[Which[viewPick == 1, curve1, viewPick == 2, curve2], \
             Frame -> True, LabelStyle -> {17, Black}, \
             ImageSize -> {600, 360}, AspectRatio -> Full] \
         ], \
         Control[{{viewPick, 1}, {1 -> \"vs time\", 2 -> \"vs position\"}, Setter}], \
         {{bankPick, 2}, {1 -> \"low\", 2 -> \"mid\", 3 -> \"high\"}, \
          ControlType -> RadioButtonBar}, \
         {{elapsed, 0, \"elapsed (s)\"}, 0, 30, 0.5, \
          ControlType -> Animator, AnimationRunning -> False}, \
         PaneSelector[{1 -> Control[{{lenA, 5, \"length A\"}, 1, 10, 0.5, \
           ControlType -> Slider, Appearance -> \"Labeled\"}], 2 -> \" \"}, \
           viewPick]]",
    )
    .expect("the Manipulate source must parse and evaluate");
    let mut state = manipulate::ManipulateState::from_expr(&expr)
      .expect("the Manipulate must build a widget");

    assert!(
      state.error.is_none(),
      "initial body must evaluate cleanly: {:?}",
      state.error
    );
    assert!(
      state.graphics_handle.is_some(),
      "the initial Which branch (curve1) must render a graphic"
    );
    let names: Vec<&str> = state.controls.iter().map(|c| c.name()).collect();
    assert_eq!(names, vec!["viewPick", "bankPick", "elapsed", "lenA"]);

    // viewPick starts at 1 ("vs time"): the PaneSelector shows lenA's row;
    // the always-visible bankPick/elapsed rows show regardless.
    assert_eq!(
      state.control_is_visible,
      vec![true, true, true, true],
      "pane 1 must show the length slider too"
    );

    // Switching the Setter to "vs position" swaps both the selected curve
    // and the PaneSelector's visible row.
    if let manipulate::ControlState::Discrete { current_index, .. } =
      &mut state.controls[0]
    {
      *current_index = 1;
    }
    state.reevaluate();
    assert_eq!(
      state.control_is_visible,
      vec![true, true, true, false],
      "pane 2 must hide the length slider"
    );
    assert!(
      state.graphics_handle.is_some(),
      "the second Which branch (curve2) must also render"
    );

    // The body's own text — not just Woxi Studio's live widget — must
    // reflect the LabelStyle fix: FrameLabel text drawn at the requested
    // size/color rather than silently defaulting to the theme's small gray.
    let render_code = format!(
      "viewPick = 1; bankPick = 2; elapsed = 12; lenA = 5;\n{}",
      state.body
    );
    let render = woxi::interpret_with_stdout(&render_code)
      .expect("the extracted body must re-render standalone");
    let svg = render.graphics.expect("Show[...] must produce a graphic");
    assert!(
      svg.contains("font-size=\"170\""),
      "LabelStyle size 17 must scale to font-size 170 at the default 10x \
       render scale: {svg}"
    );
    assert!(
      svg.contains("fill=\"rgb(0,0,0)\""),
      "LabelStyle color Black must reach the frame label: {svg}"
    );
  }

  /// End-to-end regression for the shape of the "Leverage Ratios"
  /// Wolfram Demonstration: a `Setter` whose choice rules are
  /// `key -> Tooltip[displayValue, hoverText]` (so the button shows the
  /// tooltip's *first* argument, not the whole `Tooltip[…]` source), a
  /// `Switch` that maps the setter's value onto one of several
  /// pre-defined reference records, and a body that lays out computed
  /// values next to those references in a `Grid` with a bold header row
  /// (`Style[…, Bold]`), `SpanFromLeft`, `Dividers`, `Spacings` and
  /// `ItemSize`, wrapped in a `Pane`. Heading rows built from
  /// `Row[{Style[…, Bold], " …"}]` and `Style[…, Medium]` sit between the
  /// sliders, the way the Demonstrations site captions groups of
  /// "enter values" controls.
  #[test]
  fn demonstration_setter_tooltip_choices_and_grid_reference_table() {
    woxi::interpret(
      "small = {1.1, 0.35}; medium = {1.3, 0.42}; large = {1.5, 0.5};",
    )
    .expect("the reference records must define");

    let code = "Manipulate[\
      Module[{ref, valRatio, spreadRatio}, \
        ref = Switch[batch, 1, small, 2, medium, 3, large]; \
        valRatio = N[flour/water, 3]; \
        spreadRatio = N[sugar/total, 3]; \
        Pane[\
          Grid[\
            Join[\
              {{Style[\"Recipe Ratios\", Bold], SpanFromLeft}, \
               {Style[\"parameter\", Bold], Style[\"value\", Bold], \
                Style[\"reference\", Bold]}}, \
              {{\"flour to water\", valRatio, ref[[1]]}, \
               {\"sugar share\", spreadRatio, ref[[2]]}}], \
            Dividers -> {{True, True, True, True}, \
              {True, True, True, True}}, \
            Spacings -> {Automatic, 1.2}, ItemSize -> 10], \
          ImageSize -> {300, 200}, Alignment -> {Center, Center}]\
      ], \
      {{batch, 1, \"select the recipe size\"}, \
       {3 -> Tooltip[301, \"large batch multiplier\"], \
        2 -> Tooltip[202, \"medium batch multiplier\"], \
        1 -> Tooltip[101, \"small batch multiplier\"]}, \
       ControlType -> Setter, ControlPlacement -> Top}, \
      Row[{Style[\"flour\", Bold], \" in (g)\"}], \
      Style[\"enter values from the recipe\", Medium], \
      {{flour, 300, \"flour amount\"}, 100, 1000, 0.01, \
       Appearance -> \"Labeled\", ImageSize -> Tiny}, \
      Row[{Style[\"water\", Bold], \" in (g)\"}], \
      Style[\"enter values from the recipe\", Medium], \
      {{water, 300, \"water amount\"}, 10, 1000, 0.01, \
       Appearance -> \"Labeled\", ImageSize -> Tiny}, \
      {{sugar, 500, \"sugar amount\"}, 100, 1000, 0.01, \
       Appearance -> \"Labeled\", ImageSize -> Tiny}, \
      {{total, 500, \"total weight\"}, 100, 1000, 0.01, \
       Appearance -> \"Labeled\", ImageSize -> Tiny}]";
    let mut state = instantiate_stored_manipulate(code, "")
      .expect("the recipe-ratios Manipulate must build a widget");
    assert!(
      state.error.is_none(),
      "body must evaluate cleanly: {:?}",
      state.error
    );
    // A `Pane[Grid[…]]` of plain text/numbers (no embedded picture) is left
    // to the text renderer rather than composed into an SVG — matching
    // `a_pane_without_pictures_is_not_composed` in svg_rendering.rs.
    assert!(
      state.graphics_handle.is_none(),
      "a text-only reference table composes no picture: {:?}",
      state.graphics_handle
    );
    let initial_text = state
      .text_output
      .clone()
      .expect("the reference table must render as text");
    assert!(
      initial_text.contains("flour to water")
        && initial_text.contains("sugar share"),
      "the ratio rows must appear in the rendered table: {initial_text}"
    );

    // The Setter's buttons show each choice's Tooltip *display* value
    // (532111-style numbers in the original), not the whole
    // `Tooltip[value, "…"]` source — the hover text never appears on the
    // button. Order follows the spec list, not numeric order.
    match &state.controls[0] {
      manipulate::ControlState::Discrete {
        values,
        value_labels,
        current_index,
        popup,
        ..
      } => {
        assert_eq!(values, &["3", "2", "1"]);
        assert_eq!(value_labels, &["301", "202", "101"]);
        assert_eq!(
          *current_index, 2,
          "initial batch value 1 is the third choice"
        );
        assert!(!popup, "ControlType -> Setter must not force a dropdown");
      }
      other => panic!("expected the batch Setter, got {other:?}"),
    }

    // Heading rows between the sliders carry their bold/medium styled text
    // as plain, readable labels.
    match &state.controls[1] {
      manipulate::ControlState::Heading { label, .. } => {
        assert_eq!(label, "flour in (g)");
      }
      other => panic!("expected a heading row, got {other:?}"),
    }
    match &state.controls[2] {
      manipulate::ControlState::Heading { label, .. } => {
        assert_eq!(label, "enter values from the recipe");
      }
      other => panic!("expected a heading row, got {other:?}"),
    }
    match &state.controls[3] {
      manipulate::ControlState::Continuous {
        name,
        min,
        max,
        current,
        ..
      } => {
        assert_eq!(name, "flour");
        assert_eq!((*min, *max, *current), (100.0, 1000.0, 300.0));
      }
      other => panic!("expected the flour slider, got {other:?}"),
    }

    // Picking a different industry-like reference (batch = "3", the first
    // Setter choice) must swap the Switch branch and re-render the table
    // without touching the ratios (the sliders haven't moved).
    if let manipulate::ControlState::Discrete { current_index, .. } =
      &mut state.controls[0]
    {
      *current_index = 0; // "3" -> large
    }
    state.reevaluate();
    assert!(state.error.is_none(), "re-render failed: {:?}", state.error);
    assert!(state.graphics_handle.is_none());
    let switched_text = state
      .text_output
      .expect("the reference table must still render as text");
    assert_ne!(
      initial_text, switched_text,
      "switching the reference record must change the rendered table"
    );
  }

  // A stored Manipulate whose body draws named Archimedean solids side by
  // side — the pattern a polyhedron-focused Demonstration's Initialization
  // Code + Manipulate cells reduce to once the boilerplate is stripped away.
  // Regression for the cubic-symmetry entities `PolyhedronData` was missing
  // (`TruncatedTetrahedron`, `TruncatedOctahedron`,
  // `SmallRhombicuboctahedron`): the Manipulate's body must evaluate and
  // render cleanly, and moving the slider must change the rendered scene.
  #[test]
  fn demonstration_polyhedron_manipulate_draws_cubic_archimedean_solids() {
    let nb_src = r##"Notebook[{
Cell[CellGroupData[{
Cell[BoxData["Manipulate[
 Module[{solids = {\"TruncatedTetrahedron\", \"TruncatedOctahedron\", \"SmallRhombicuboctahedron\"}},
  Graphics3D[
   Table[
    Translate[PolyhedronData[solids[[k]], \"Faces\"], {2.5 (k - 2), 0, 0}],
    {k, 1, gap}],
   Boxed -> False, ViewPoint -> {1.3, -2.4, 2}]],
 {{gap, 3, \"solids shown\"}, 1, 3, 1}]"], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`gap$$ = 3}, DynamicBox[\[Ellipsis]]]"], "Output"]
}, Open]]
}]"##;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);
    let mut widget = editors
      .into_iter()
      .find_map(|e| e.manipulate_state)
      .expect("the stored Manipulate must instantiate on load");
    assert!(
      widget.error.is_none(),
      "body must evaluate cleanly: {:?}",
      widget.error
    );
    assert!(
      widget.graphics_handle.is_some(),
      "PolyhedronData[...] must draw all three cubic Archimedean solids"
    );

    match &widget.controls[..] {
      [
        manipulate::ControlState::Continuous {
          name,
          min,
          max,
          current,
          ..
        },
      ] => {
        assert_eq!(name.as_str(), "gap");
        assert_eq!(*min, 1.0);
        assert_eq!(*max, 3.0);
        assert_eq!(*current, 3.0);
      }
      other => panic!("unexpected controls: {other:?}"),
    };

    let render = |w: &manipulate::ManipulateState| {
      let bindings: Vec<(String, String)> = w
        .controls
        .iter()
        .filter(|c| c.binds_variable())
        .map(|c| (c.name().to_string(), c.current_code()))
        .collect();
      woxi::with_scoped_globals(&bindings, || {
        woxi::interpret_with_stdout(&w.body)
      })
      .expect("body evaluates")
      .graphics
      .expect("the solids must render")
    };
    let three_solids_view = render(&widget);

    // Dropping to one solid must re-render a visibly different scene.
    match &mut widget.controls[0] {
      manipulate::ControlState::Continuous { current, .. } => *current = 1.0,
      other => panic!("expected gap as a Continuous control, got {other:?}"),
    }
    widget.reevaluate();
    assert!(widget.error.is_none());
    assert!(widget.graphics_handle.is_some());
    assert_ne!(
      three_solids_view,
      render(&widget),
      "showing only one solid must change the rendered scene"
    );
  }

  /// End-to-end regression for the shape of the "Primitive Relation for
  /// Elliptic Geometry" Demonstration: a separate `InitializationCell`
  /// defines a great-circle-arc helper built from `With`, `Normalize`,
  /// `Cross`, and `RotationTransform` composed with `ParametricPlot3D`
  /// (extracting its curve via `[[1]]`), and a stored `Manipulate` whose
  /// `Module` body assembles a `Graphics3D` of `Sphere[]`, `Opacity`,
  /// `Line`s and `Text[Style[...], point, offset]` labels, picking between
  /// two point-pair layouts with `Switch` and toggling a set of axis
  /// `Line`s on and off with `If`, driven by a continuous rotation slider
  /// alongside two discrete (`Setter`/checkbox) controls.
  #[test]
  fn demonstration_great_circle_pair_manipulate_switches_and_toggles() {
    let nb_src = r##"Notebook[{
Cell[BoxData["greatCircle3D[a_, b_] := With[{axis = Normalize[Cross[a, b]]}, ParametricPlot3D[RotationTransform[t, axis][a], {t, 0, 2 Pi}, PlotStyle -> {Purple}][[1]]]"], "Input",
 InitializationCell->True],
Cell[CellGroupData[{
Cell[BoxData["Manipulate[
 Module[{ptA = {Cos[theta], Sin[theta], 0}, ptB = {0, 1, 0}, ptC = {0, 0, 1}},
   Graphics3D[{
     Opacity[0.5], Sphere[],
     If[showAxes, {Line[{{-1, 0, 0}, {1, 0, 0}}], Line[{{0, -1, 0}, {0, 1, 0}}], Line[{{0, 0, -1}, {0, 0, 1}}]}, {}],
     Switch[view,
       1, {greatCircle3D[ptA, ptB], Line[{{0, 0, 0}, ptA}], Line[{{0, 0, 0}, ptB}],
           Text[Style[\"A\", 16], ptA, {0, 1}], Text[Style[\"B\", 16], ptB, {0, 1}]},
       2, {greatCircle3D[ptB, ptC], Line[{{0, 0, 0}, ptB}], Line[{{0, 0, 0}, ptC}],
           Text[Style[\"B\", 16], ptB, {0, 1}], Text[Style[\"C\", 16], ptC, {0, 1}]}]
   }]
 ],
 {{theta, 0, \"rotation\"}, 0, 2 Pi},
 {{view, 1, \"pair\"}, {1, 2}},
 {{showAxes, True, \"axes\"}, {True, False}}]"], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`theta$$ = 0, $CellContext`view$$ = 1, $CellContext`showAxes$$ = True}, DynamicBox[\[Ellipsis]]]"], "Output"]
}, Open]]
}]"##;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);
    let mut widget = editors
      .into_iter()
      .find_map(|e| e.manipulate_state)
      .expect("the stored Manipulate must instantiate on load");
    assert!(
      widget.error.is_none(),
      "body must evaluate cleanly: {:?}",
      widget.error
    );
    assert!(
      widget.graphics_handle.is_some(),
      "the Switch's first branch (Sphere + great circle + labels) must draw"
    );

    match &widget.controls[..] {
      [
        manipulate::ControlState::Continuous {
          name: theta_name,
          label: theta_label,
          min: theta_min,
          max: theta_max,
          current: theta_now,
          ..
        },
        manipulate::ControlState::Discrete {
          name: view_name,
          values: view_values,
          current_index: view_idx,
          ..
        },
        manipulate::ControlState::Discrete {
          name: axes_name,
          values: axes_values,
          current_index: axes_idx,
          ..
        },
      ] => {
        assert_eq!(theta_name.as_str(), "theta");
        assert_eq!(theta_label.as_str(), "rotation");
        assert_eq!(*theta_min, 0.0);
        assert!((*theta_max - std::f64::consts::TAU).abs() < 1e-9);
        assert_eq!(*theta_now, 0.0);
        assert_eq!(view_name.as_str(), "view");
        assert_eq!(view_values.as_slice(), ["1", "2"]);
        assert_eq!(*view_idx, 0);
        assert_eq!(axes_name.as_str(), "showAxes");
        assert_eq!(axes_values.as_slice(), ["True", "False"]);
        assert_eq!(*axes_idx, 0);
      }
      other => panic!("unexpected controls: {other:?}"),
    }

    // The great-circle helper was defined in the separate Initialization
    // cell (run once before the widget's body first evaluates), so the
    // stored body can be re-rendered on its own to inspect the SVG.
    let render = |w: &manipulate::ManipulateState| {
      let bindings: Vec<(String, String)> = w
        .controls
        .iter()
        .filter(|c| c.binds_variable())
        .map(|c| (c.name().to_string(), c.current_code()))
        .collect();
      woxi::with_scoped_globals(&bindings, || {
        woxi::interpret_with_stdout(&w.body)
      })
      .expect("body evaluates")
      .graphics
      .expect("the pieces must render")
    };

    let pair_one = render(&widget);

    // Switching `view` to the second pair swaps which two points the great
    // circle and its labels connect, so the render must change.
    match &mut widget.controls[1] {
      manipulate::ControlState::Discrete { current_index, .. } => {
        *current_index = 1;
      }
      other => panic!("expected discrete control, got {other:?}"),
    }
    widget.reevaluate();
    assert!(widget.error.is_none());
    assert!(widget.graphics_handle.is_some());
    let pair_two = render(&widget);
    assert_ne!(
      pair_one, pair_two,
      "switching the Switch selector must change the rendered pair"
    );

    // Toggling the axes checkbox off must drop the three axis Lines from
    // this same view, changing the render again.
    match &mut widget.controls[2] {
      manipulate::ControlState::Discrete { current_index, .. } => {
        *current_index = 1;
      }
      other => panic!("expected discrete control, got {other:?}"),
    }
    widget.reevaluate();
    assert!(widget.error.is_none());
    let pair_two_no_axes = render(&widget);
    assert_ne!(
      pair_two, pair_two_no_axes,
      "unchecking the axes toggle must hide the axis lines"
    );
  }

  /// End-to-end regression for the shape of the "Compound of Two
  /// Icosahedra" Demonstration: a stored `Manipulate` whose body is a
  /// flat sequence of `;`-separated assignments (no `Module`) building
  /// two `GraphicsComplex`es from `PolyhedronData[name, "VertexCoordinates"]`
  /// / `Polygon[PolyhedronData[name, "FaceIndices"]]`, rotating one by a
  /// fixed `ArcTan[...]` angle around one axis and then by a
  /// slider-controlled angle around another, scaling both by a
  /// `factor*{1, 1, 1}` triple (one a near-1 constant nudge, the other
  /// driven by a second slider), and combining the colored pieces in a
  /// `Graphics3D` with `ViewPoint`/`ViewAngle`/`SphericalRegion`/`Boxed`/
  /// `PlotRange` options. Driven by two continuous sliders with text
  /// labels, one ascending and one descending (`max` to `min`), plus a
  /// `TrackedSymbols` option.
  #[test]
  fn demonstration_polyhedron_compound_manipulate_rotates_and_scales() {
    let nb_src = r##"Notebook[{
Cell[CellGroupData[{
Cell[BoxData["Manipulate[p1 = GraphicsComplex[PolyhedronData[\"Tetrahedron\", \"VertexCoordinates\"], Polygon[PolyhedronData[\"Tetrahedron\", \"FaceIndices\"]]]; p1a = Rotate[p1, ArcTan[1/2], {1, 0, 0}]; p1b = Scale[Rotate[p1a, -spin, {0, 1, 0}], 1.002*{1, 1, 1}, {0, 0, 0}]; p2 = Scale[GraphicsComplex[PolyhedronData[\"Cube\", \"VertexCoordinates\"], Polygon[PolyhedronData[\"Cube\", \"FaceIndices\"]]], grow*{1, 1, 1}, {0, 0, 0}]; Graphics3D[{p1a, {Blue, p1b}, {Orange, p2}}, Boxed -> False, SphericalRegion -> True, ViewAngle -> Pi/24, ImageSize -> {380, 380}, ViewPoint -> {3, -8, 3}, PlotRange -> 1.3*{{-1, 1}, {-1, 1}, {-1, 1}}], {{grow, 1.5, \"cube size\"}, 0.4, 2.0}, {{spin, Pi/2, \"rotate tetrahedron\"}, Pi/2, 0}, TrackedSymbols :> {grow, spin}]"], "Input"],
Cell[BoxData["DynamicModuleBox[{$CellContext`grow$$ = 1.5, $CellContext`spin$$ = Pi/2}, DynamicBox[\[Ellipsis]]]"], "Output"]
}, Open]]
}]"##;
    let nb = woxi::notebook::parse_notebook(nb_src).unwrap();
    let editors = WoxiStudio::editors_from_notebook(&nb);
    let mut widget = editors
      .into_iter()
      .find_map(|e| e.manipulate_state)
      .expect("the stored Manipulate must instantiate on load");
    assert!(
      widget.error.is_none(),
      "body must evaluate cleanly: {:?}",
      widget.error
    );
    assert!(
      widget.graphics_handle.is_some(),
      "the two rotated/scaled polyhedron pieces must draw"
    );

    match &widget.controls[..] {
      [
        manipulate::ControlState::Continuous {
          name: grow_name,
          label: grow_label,
          min: grow_min,
          max: grow_max,
          current: grow_now,
          ..
        },
        manipulate::ControlState::Continuous {
          name: spin_name,
          label: spin_label,
          min: spin_min,
          max: spin_max,
          current: spin_now,
          ..
        },
      ] => {
        assert_eq!(grow_name.as_str(), "grow");
        assert_eq!(grow_label.as_str(), "cube size");
        assert_eq!(*grow_min, 0.4);
        assert_eq!(*grow_max, 2.0);
        assert_eq!(*grow_now, 1.5);
        assert_eq!(spin_name.as_str(), "spin");
        assert_eq!(spin_label.as_str(), "rotate tetrahedron");
        // The spin slider's spec range descends (max to min), matching
        // the source demonstration's rotation control.
        assert!((*spin_min - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
        assert_eq!(*spin_max, 0.0);
        assert!((*spin_now - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
      }
      other => panic!("unexpected controls: {other:?}"),
    }

    let render = |w: &manipulate::ManipulateState| {
      let bindings: Vec<(String, String)> = w
        .controls
        .iter()
        .filter(|c| c.binds_variable())
        .map(|c| (c.name().to_string(), c.current_code()))
        .collect();
      woxi::with_scoped_globals(&bindings, || {
        woxi::interpret_with_stdout(&w.body)
      })
      .expect("body evaluates")
      .graphics
      .expect("the pieces must render")
    };

    let initial = render(&widget);

    // Dragging the cube-size slider changes the cube's scale.
    match &mut widget.controls[0] {
      manipulate::ControlState::Continuous { current, .. } => *current = 2.0,
      other => panic!("expected continuous control, got {other:?}"),
    }
    widget.reevaluate();
    assert!(widget.error.is_none());
    let grown_render = render(&widget);
    assert_ne!(
      initial, grown_render,
      "moving the cube-size slider must change the rendered picture"
    );

    // Dragging the rotation slider to its other end rotates the
    // tetrahedron piece around a different axis, also changing the
    // picture again.
    match &mut widget.controls[1] {
      manipulate::ControlState::Continuous { current, .. } => *current = 0.0,
      other => panic!("expected continuous control, got {other:?}"),
    }
    widget.reevaluate();
    assert!(widget.error.is_none());
    let spun_render = render(&widget);
    assert_ne!(
      grown_render, spun_render,
      "moving the rotation slider must change the rendered picture"
    );
  }
}
