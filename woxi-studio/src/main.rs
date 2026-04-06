mod cell_type_dropdown;
mod highlighter;
mod notebook;

use iced::keyboard;
use iced::overlay::menu;
use iced::widget::operation::focus;
use iced::widget::{
  Column, button, column, container, image, pick_list, row, rule, scrollable,
  space, svg, text, text_editor,
};
use iced::{
  Background, Border, Center, Color, Element, Fill, Font, Subscription, Task,
  Theme,
};

use notebook::{Cell, CellEntry, CellGroup, CellStyle, Notebook};
use std::path::PathBuf;
use std::sync::Arc;

fn main() -> iced::Result {
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
  /// Warning messages from evaluation (e.g. unimplemented functions).
  warnings: Vec<String>,
  /// Undo stack: previous text snapshots.
  undo_stack: Vec<String>,
  /// Redo stack: snapshots restored via undo.
  redo_stack: Vec<String>,
  /// Whether the input has changed since the last evaluation.
  output_stale: bool,
}

// ── Messages ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[allow(dead_code)]
enum Message {
  // File operations
  NewNotebook,
  OpenFile,
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
  Undo(usize),
  Redo(usize),
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

  // Settings
  ThemeChanged(ThemeChoice),
  NewCellStyleChanged(CellStyle),

  // Cell type menu
  ToggleCellTypeMenu(usize),

  // Preview mode
  TogglePreview,

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

    let task = if let Some(path) = load_last_file_path() {
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
          db.load_system_fonts();
          Arc::new(db)
        },
      },
      task,
    )
  }

  /// Build editor state from a notebook.
  /// Output/Print cells within a group are attached to the
  /// preceding Input/Code cell rather than shown separately.
  fn editors_from_notebook(notebook: &Notebook) -> Vec<CellEditor> {
    let mut editors = Vec::new();

    for entry in &notebook.cells {
      match entry {
        CellEntry::Single(cell) => {
          if matches!(cell.style, CellStyle::Output | CellStyle::Print) {
            continue;
          }
          editors.push(CellEditor {
            content: text_editor::Content::with_text(&cell.content),
            style: cell.style,
            output: None,
            stdout: None,
            graphics_svg: None,
            graphics_handle: None,
            graphics_image: None,
            warnings: Vec::new(),
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            output_stale: false,
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
              editors.push(CellEditor {
                content: text_editor::Content::with_text(&cell.content),
                style: cell.style,
                output,
                stdout,
                graphics_svg: None,
                graphics_handle: None,
                graphics_image: None,
                warnings: Vec::new(),
                undo_stack: Vec::new(),
                redo_stack: Vec::new(),
                output_stale: false,
              });
              i = j;
            } else if matches!(cell.style, CellStyle::Output | CellStyle::Print)
            {
              // Skip standalone output/print in groups
              i += 1;
            } else {
              editors.push(CellEditor {
                content: text_editor::Content::with_text(&cell.content),
                style: cell.style,
                output: None,
                stdout: None,
                graphics_svg: None,
                graphics_handle: None,
                graphics_image: None,
                warnings: Vec::new(),
                undo_stack: Vec::new(),
                redo_stack: Vec::new(),
                output_stale: false,
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
      let cell = Cell::new(editor.style, content);

      // Group input cells with their output
      if editor.style == CellStyle::Input {
        if let Some(ref output) = editor.output {
          let output_cell = Cell::new(CellStyle::Output, output.clone());
          cells.push(CellEntry::Group(CellGroup {
            cells: vec![cell, output_cell],
            open: true,
          }));
          i += 1;
          continue;
        }
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
        self.file_path = None;
        self.notebook = Notebook::new();
        self
          .notebook
          .push_cell(Cell::new(CellStyle::Title, "Untitled Notebook"));
        self.notebook.push_cell(Cell::new(CellStyle::Input, ""));
        self.cell_editors = Self::editors_from_notebook(&self.notebook);
        self.focused_cell = Some(1);
        self.is_dirty = false;
        self.status = String::from("New notebook created");
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

      Message::FileOpened(result) => {
        self.is_loading = false;
        match result {
          Ok((path, contents)) => match notebook::parse_notebook(&contents) {
            Ok(nb) => {
              self.cell_editors = Self::editors_from_notebook(&nb);
              self.notebook = nb;
              self.status = format!("Opened: {}", path.display());
              save_last_file_path(&path);
              self.file_path = Some(path);
              self.is_dirty = false;
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

      Message::Undo(idx) => {
        if idx < self.cell_editors.len() {
          if let Some(prev) = self.cell_editors[idx].undo_stack.pop() {
            let current = self.cell_editors[idx].content.text();
            self.cell_editors[idx].redo_stack.push(current);
            self.cell_editors[idx].content =
              text_editor::Content::with_text(&prev);
            self.is_dirty = true;
            self.cell_editors[idx].output_stale = true;
          }
        }
        Task::none()
      }

      Message::Redo(idx) => {
        if idx < self.cell_editors.len() {
          if let Some(next) = self.cell_editors[idx].redo_stack.pop() {
            let current = self.cell_editors[idx].content.text();
            self.cell_editors[idx].undo_stack.push(current);
            self.cell_editors[idx].content =
              text_editor::Content::with_text(&next);
            self.is_dirty = true;
            self.cell_editors[idx].output_stale = true;
          }
        }
        Task::none()
      }

      Message::CellStyleChanged(idx, style) => {
        if idx < self.cell_editors.len() {
          self.cell_editors[idx].style = style;
          if style != CellStyle::Input {
            self.cell_editors[idx].output = None;
            self.cell_editors[idx].stdout = None;
            self.cell_editors[idx].graphics_svg = None;
            self.cell_editors[idx].graphics_handle = None;
            self.cell_editors[idx].graphics_image = None;
            self.cell_editors[idx].warnings.clear();
            self.cell_editors[idx].output_stale = false;
          }
          self.is_dirty = true;
        }
        self.cell_type_menu_open = None;
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

      Message::TogglePreview => {
        self.preview_mode = !self.preview_mode;
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
          }
        }
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
            warnings: Vec::new(),
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            output_stale: false,
          },
        );
        self.focused_cell = Some(insert_at);
        self.focused_divider = None;
        self.is_dirty = true;
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
            warnings: Vec::new(),
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            output_stale: false,
          },
        );
        self.focused_cell = Some(insert_at);
        self.focused_divider = None;
        self.is_dirty = true;
        focus(iced::widget::Id::from(format!("cell-{insert_at}")))
      }

      Message::DeleteCell(idx) => {
        if self.cell_editors.len() > 1 && idx < self.cell_editors.len() {
          self.cell_editors.remove(idx);
          self.is_dirty = true;
          if let Some(ref mut focused) = self.focused_cell {
            if *focused >= self.cell_editors.len() {
              *focused = self.cell_editors.len() - 1;
            }
          }
        }
        Task::none()
      }

      Message::MoveCellUp(idx) => {
        if idx > 0 && idx < self.cell_editors.len() {
          self.cell_editors.swap(idx, idx - 1);
          self.focused_cell = Some(idx - 1);
          self.is_dirty = true;
        }
        Task::none()
      }

      Message::MoveCellDown(idx) => {
        if idx + 1 < self.cell_editors.len() {
          self.cell_editors.swap(idx, idx + 1);
          self.focused_cell = Some(idx + 1);
          self.is_dirty = true;
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
            evaluate_cell_statements(
              &mut self.cell_editors[idx],
              &code,
              self.scale_factor,
              &self.fontdb,
            );
            self.status = format!("Evaluated cell {} successfully", idx + 1);
          }
        }
        Task::none()
      }

      Message::EvaluateAll => {
        woxi::clear_state();
        for idx in 0..self.cell_editors.len() {
          if matches!(
            self.cell_editors[idx].style,
            CellStyle::Input | CellStyle::Code
          ) {
            let code = self.cell_editors[idx].content.text().trim().to_string();
            if !code.is_empty() {
              evaluate_cell_statements(
                &mut self.cell_editors[idx],
                &code,
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
        if modifiers.control() {
          if let keyboard::Key::Character("d") = key.as_ref() {
            if let Some(idx) = self.focused_cell {
              self.cell_editors[idx]
                .content
                .perform(text_editor::Action::Edit(text_editor::Edit::Delete));
              self.is_dirty = true;
            }
            return Task::none();
          }
        }

        // Ctrl+W: delete previous word
        if modifiers.control() {
          if let keyboard::Key::Character("w") = key.as_ref() {
            if let Some(idx) = self.focused_cell {
              self.cell_editors[idx].content.perform(
                text_editor::Action::Select(text_editor::Motion::WordLeft),
              );
              self.cell_editors[idx].content.perform(
                text_editor::Action::Edit(text_editor::Edit::Backspace),
              );
              self.is_dirty = true;
            }
            return Task::none();
          }
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
    iced::event::listen_with(handle_event)
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

      if !self.preview_mode {
        // Add cell divider above the first cell
        col = col.push(self.view_add_cell_divider_above(0));
      }

      for (idx, editor) in self.cell_editors.iter().enumerate() {
        // Add cell divider between cells
        if !self.preview_mode && idx > 0 {
          col = col.push(self.view_add_cell_divider(idx.saturating_sub(1)));
        }

        let is_focused = self.focused_cell == Some(idx);
        col = col.push(self.view_cell(idx, editor, is_focused));
      }

      if !self.preview_mode {
        // Final add-cell divider after last cell
        col =
          col
            .push(self.view_add_cell_divider(
              self.cell_editors.len().saturating_sub(1),
            ));
      }

      scrollable(container(col.max_width(800)).center_x(Fill))
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

    // ── Layout ──
    column![
      toolbar,
      rule::horizontal(1).style(separator_style),
      cells,
      status_bar,
    ]
    .spacing(0)
    .into()
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
    }

    // ── Text editor ──
    let font_size = match editor.style {
      CellStyle::Title => 20.0,
      CellStyle::Subtitle => 16.0,
      CellStyle::Section => 15.0,
      CellStyle::Subsection => 14.0,
      CellStyle::Subsubsection => 13.0,
      _ => 13.0,
    };

    let cell_font = match editor.style {
      CellStyle::Title | CellStyle::Subtitle => Font {
        weight: iced::font::Weight::Bold,
        ..Font::MONOSPACE
      },
      _ => Font::MONOSPACE,
    };

    let cell_style = editor.style;
    let in_preview = self.preview_mode;
    let has_output = editor.stdout.is_some()
      || editor.graphics_svg.is_some()
      || editor.output.as_ref().map_or(false, |o| {
        let d = o
          .replace("-Graphics-", "")
          .replace("-Graphics3D-", "")
          .replace("-Image-", "");
        !d.trim().is_empty()
      });
    let is_grouped = is_input && has_output && !in_preview;
    let cursor_line = editor.content.cursor().position.line;
    let line_count = editor.content.line_count();
    let at_last_line = cursor_line >= line_count.saturating_sub(1);
    let at_first_line = cursor_line == 0;
    let cell_count = self.cell_editors.len();
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
        if modifiers.command() {
          match key.as_ref() {
            keyboard::Key::Character("z") if modifiers.shift() => {
              return Some(text_editor::Binding::Custom(Message::Redo(idx)));
            }
            keyboard::Key::Character("z") => {
              return Some(text_editor::Binding::Custom(Message::Undo(idx)));
            }
            _ => {}
          }
        }
        // Shift+Enter: evaluate the cell and move to the cell below.
        // If there is no cell below, insert a new one. Handled here
        // (before the text editor processes the key) so no stray newline
        // is inserted into the cell's content.
        if modifiers.shift() {
          if let keyboard::Key::Named(keyboard::key::Named::Enter) =
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
              bindings.push(text_editor::Binding::Custom(Message::FocusCell(
                idx + 1,
              )));
            }
            return Some(text_editor::Binding::Sequence(bindings));
          }
        }
        // Arrow key navigation between cells
        let no_mods =
          !modifiers.shift() && !modifiers.command() && !modifiers.control();
        if no_mods {
          if let keyboard::Key::Named(keyboard::key::Named::ArrowDown) =
            key.as_ref()
          {
            if at_last_line && idx < cell_count.saturating_sub(1) {
              return Some(text_editor::Binding::Sequence(vec![
                text_editor::Binding::Unfocus,
                text_editor::Binding::Custom(Message::FocusDividerBelow(idx)),
              ]));
            }
          }
          if let keyboard::Key::Named(keyboard::key::Named::ArrowUp) =
            key.as_ref()
          {
            if at_first_line && idx > 0 {
              return Some(text_editor::Binding::Sequence(vec![
                text_editor::Binding::Unfocus,
                text_editor::Binding::Custom(Message::FocusDividerAbove(idx)),
              ]));
            }
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
    content_col = content_col.push(cell_editor);

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
      if let Some(ref stdout) = editor.stdout {
        let mut stdout_text = text(stdout).size(12).font(Font::MONOSPACE);
        if stale {
          stdout_text = stdout_text.color(Color::from_rgba(0.5, 0.5, 0.5, 0.5));
        }
        let stdout_display = container(stdout_text).padding(6).width(Fill);

        output_col = output_col.push(stdout_display);
      }

      // Graphics rendering (pre-rasterized image, falls back to SVG)
      if let Some((ref img_handle, w, h)) = editor.graphics_image {
        let mut img_widget = image(img_handle.clone())
          .width(iced::Length::Fixed(w as f32))
          .height(iced::Length::Fixed(h as f32));
        if stale {
          img_widget = img_widget.opacity(0.3);
        }
        output_col = output_col.push(container(img_widget).padding(4));
      } else if let Some(ref handle) = editor.graphics_handle {
        let mut svg_widget =
          svg::Svg::new(handle.clone()).width(iced::Length::Shrink);
        if stale {
          svg_widget = svg_widget.opacity(0.3);
        }
        output_col = output_col.push(container(svg_widget).padding(4));
      }

      // Text output (filter out graphics placeholders)
      if let Some(ref output) = editor.output {
        let display = output
          .replace("-Graphics-", "")
          .replace("-Graphics3D-", "")
          .replace("-Image-", "");
        let display = display.trim().to_string();
        if !display.is_empty() {
          let mut output_text = text(display).size(12).font(Font::MONOSPACE);
          if stale {
            output_text =
              output_text.color(Color::from_rgba(0.5, 0.5, 0.5, 0.5));
          }
          let output_display = container(output_text).padding(6).width(Fill);

          output_col = output_col.push(output_display);
        }
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

      if let Some(ref stdout) = editor.stdout {
        let mut stdout_text = text(stdout).size(12).font(Font::MONOSPACE);
        if stale {
          stdout_text = stdout_text.color(Color::from_rgba(0.5, 0.5, 0.5, 0.5));
        }
        let stdout_display = container(stdout_text).padding(6).width(Fill);

        content_col = content_col.push(stdout_display);
      }

      if let Some((ref img_handle, w, h)) = editor.graphics_image {
        let mut img_widget = image(img_handle.clone())
          .width(iced::Length::Fixed(w as f32))
          .height(iced::Length::Fixed(h as f32));
        if stale {
          img_widget = img_widget.opacity(0.3);
        }
        content_col = content_col.push(container(img_widget).padding(4));
      } else if let Some(ref handle) = editor.graphics_handle {
        let mut svg_widget =
          svg::Svg::new(handle.clone()).width(iced::Length::Shrink);
        if stale {
          svg_widget = svg_widget.opacity(0.3);
        }
        content_col = content_col.push(container(svg_widget).padding(4));
      }

      if let Some(ref output) = editor.output {
        let display = output
          .replace("-Graphics-", "")
          .replace("-Graphics3D-", "")
          .replace("-Image-", "");
        let display = display.trim().to_string();
        if !display.is_empty() {
          let mut output_text = text(display).size(12).font(Font::MONOSPACE);
          if stale {
            output_text =
              output_text.color(Color::from_rgba(0.5, 0.5, 0.5, 0.5));
          }
          let output_display = container(output_text).padding(6).width(Fill);

          content_col = content_col.push(output_display);
        }
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
        right_col = right_col.push(
          button(text("\u{25B6}").size(14))
            .on_press(Message::EvaluateCell(idx))
            .padding([4, 8])
            .style(muted_button_style),
        );
      }
      right_col = right_col.push(trash_btn);
      right_col.into()
    } else {
      text("").into()
    };

    let cell_row = row![gutter, content_el, right_side]
      .spacing(0)
      .padding([1, 2]);

    container(cell_row).width(Fill).into()
  }
}

// ── Event handling ──────────────────────────────────────────────────

fn handle_event(
  event: iced::Event,
  status: iced::event::Status,
  _id: iced::window::Id,
) -> Option<Message> {
  if let iced::Event::Window(iced::window::Event::CloseRequested) = &event {
    return Some(Message::CloseRequested(_id));
  }

  if let iced::Event::Window(iced::window::Event::Rescaled(scale)) = &event {
    return Some(Message::ScaleFactorChanged(*scale));
  }

  if let iced::Event::Keyboard(keyboard::Event::KeyPressed {
    key,
    modifiers,
    ..
  }) = event
  {
    // When no widget captured the event (e.g. divider is focused),
    // handle arrow keys and Enter for navigation.
    if matches!(status, iced::event::Status::Ignored) {
      let no_mods =
        !modifiers.shift() && !modifiers.command() && !modifiers.control();
      if no_mods {
        match key.as_ref() {
          keyboard::Key::Named(
            keyboard::key::Named::ArrowDown
            | keyboard::key::Named::ArrowUp
            | keyboard::key::Named::Enter,
          ) => {
            return Some(Message::KeyPressed(key, modifiers));
          }
          _ => {}
        }
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

// ── Cell evaluation ─────────────────────────────────────────────────

/// Evaluate all statements in a cell and collect their results.
/// When a cell contains multiple newline-separated expressions,
/// each expression's output is included (matching Mathematica behavior).
fn evaluate_cell_statements(
  editor: &mut CellEditor,
  code: &str,
  scale_factor: f32,
  fontdb: &Arc<resvg::usvg::fontdb::Database>,
) {
  let statements = woxi::split_into_statements(code);

  let mut outputs: Vec<String> = Vec::new();
  let mut all_stdout = String::new();
  let mut last_graphics: Option<String> = None;
  let mut all_warnings: Vec<String> = Vec::new();
  let mut had_error = false;

  for stmt in &statements {
    match woxi::interpret_with_stdout(stmt) {
      Ok(result) => {
        if !result.stdout.is_empty() {
          all_stdout.push_str(&result.stdout);
        }
        all_warnings.extend(result.warnings);

        if let Some(svg) = result.graphics {
          if result.result != "\0" {
            last_graphics = Some(svg);
          }
        }

        if result.result != "\0" {
          outputs.push(result.result);
        }
      }
      Err(woxi::InterpreterError::EmptyInput) => {}
      Err(e) => {
        outputs.push(format!("Error: {e}"));
        had_error = true;
      }
    }
  }

  editor.output = if outputs.is_empty() {
    None
  } else {
    Some(outputs.join("\n"))
  };
  editor.stdout = if all_stdout.is_empty() {
    None
  } else {
    Some(all_stdout)
  };
  editor.graphics_svg = last_graphics;
  editor.graphics_handle = editor
    .graphics_svg
    .as_ref()
    .map(|s| svg::Handle::from_memory(s.as_bytes().to_vec()));
  editor.graphics_image = editor
    .graphics_svg
    .as_ref()
    .and_then(|s| rasterize_svg(s, scale_factor, fontdb));
  editor.warnings = all_warnings;
  editor.output_stale = false;
  let _ = had_error;
}

// ── Custom styles ───────────────────────────────────────────────────

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
      | CellStyle::Section
      | CellStyle::Subsection
      | CellStyle::Subsubsection
      | CellStyle::Text
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

// ── CellStyle display/picklist support ──────────────────────────────

const CELL_STYLES: &[CellStyle] = &[
  CellStyle::Title,
  CellStyle::Subtitle,
  CellStyle::Section,
  CellStyle::Subsection,
  CellStyle::Subsubsection,
  CellStyle::Text,
  CellStyle::Input,
  CellStyle::Output,
  CellStyle::Code,
];

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
          24.0,
          "bold",
          "sans-serif",
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
          16.0,
          "normal",
          "sans-serif",
          "#555",
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
          18.0,
          "bold",
          "sans-serif",
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
          15.0,
          "bold",
          "sans-serif",
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
          13.0,
          "bold",
          "sans-serif",
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
          12.0,
          "normal",
          "serif",
          "#000",
          16.0,
        );
        y += 8.0;
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
        for line in &lines {
          let _ = write!(
            elements,
            r##"<text x="{margin}" y="{y}" font-size="11" font-family="Courier Prime, monospace" fill="#333">{}</text>"##,
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
            11.0,
            "normal",
            "Courier Prime, monospace",
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
            11.0,
            "normal",
            "Courier Prime, monospace",
            "#888",
            14.0,
          );
          y += 4.0;
        }
      }

      if let Some(ref svg_data) = cell.graphics_svg {
        if let Some((svg_w, svg_h)) = parse_svg_dimensions(svg_data) {
          let scale = (content_width / svg_w).min(1.0);
          let rendered_w = svg_w * scale;
          let rendered_h = svg_h * scale;
          let _ = write!(
            elements,
            r#"<svg x="{margin}" y="{y}" width="{rendered_w}" height="{rendered_h}" viewBox="0 0 {svg_w} {svg_h}">"#,
          );
          elements.push_str(&strip_svg_wrapper(svg_data));
          elements.push_str("</svg>");
          y += rendered_h + 8.0;
        }
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
            11.0,
            "normal",
            "Courier Prime, monospace",
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
    include_bytes!("../../resources/CourierPrime-Regular.ttf").to_vec(),
  );
  fontdb.load_font_data(
    include_bytes!("../../resources/CourierPrime-Bold.ttf").to_vec(),
  );
  fontdb.set_monospace_family("Courier Prime");
  fontdb.load_system_fonts();

  let mut opt = svg2pdf::usvg::Options::default();
  opt.fontdb = StdArc::new(fontdb);

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
  font_size: f64,
  font_weight: &str,
  font_family: &str,
  fill: &str,
  line_height: f64,
) {
  use std::fmt::Write;
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
