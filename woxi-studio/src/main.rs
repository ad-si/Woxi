mod cell_type_dropdown;
mod highlighter;
mod notebook;

use iced::keyboard;
use iced::overlay::menu;
use iced::widget::operation::focus;
use iced::widget::{
  Column, button, column, container, image, mouse_area, opaque, pick_list, row,
  rule, scrollable, space, stack, svg, text, text_editor,
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
  /// Index of the cell whose graphic is shown in the fullscreen modal.
  graphics_modal_cell: Option<usize>,
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
  /// For Chapter/Subchapter cells: whether the section is collapsed,
  /// hiding all cells below it until the next same-or-higher heading.
  is_collapsed: bool,
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
  WrapSelection(usize, char, char),
  Undo(usize),
  Redo(usize),
  IndentLines(usize),
  UnindentLines(usize),
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

  // Collapse/expand Chapter or Subchapter
  ToggleCollapse(usize),

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

  // Graphics modal
  OpenGraphicsModal(usize),
  CloseGraphicsModal,
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
        graphics_modal_cell: None,
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
            is_collapsed: false,
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
                is_collapsed: false,
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
                is_collapsed: false,
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

      Message::WrapSelection(idx, open, close) => {
        if idx < self.cell_editors.len() {
          if let Some(sel) = self.cell_editors[idx].content.selection() {
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
              self.cell_editors[idx].content.perform(
                text_editor::Action::Edit(text_editor::Edit::Insert(c)),
              );
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

      Message::TogglePreview => {
        self.preview_mode = !self.preview_mode;
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
            is_collapsed: false,
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
            is_collapsed: false,
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
        // Escape closes the graphics modal
        if matches!(
          key.as_ref(),
          keyboard::Key::Named(keyboard::key::Named::Escape)
        ) && self.graphics_modal_cell.is_some()
        {
          self.graphics_modal_cell = None;
          return Task::none();
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

      let hidden = self.compute_hidden_cells();

      if !self.preview_mode {
        // Add cell divider above the first cell
        col = col.push(self.view_add_cell_divider_above(0));
      }

      let mut visible_count = 0usize;
      for (idx, editor) in self.cell_editors.iter().enumerate() {
        if hidden[idx] {
          continue;
        }
        // Add cell divider between cells
        if !self.preview_mode && visible_count > 0 {
          col = col.push(self.view_add_cell_divider(idx.saturating_sub(1)));
        }

        let is_focused = self.focused_cell == Some(idx);
        col = col.push(self.view_cell(idx, editor, is_focused));
        visible_count += 1;
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
    let main_view: Element<'_, Message> = column![
      toolbar,
      rule::horizontal(1).style(separator_style),
      cells,
      status_bar,
    ]
    .spacing(0)
    .into();

    // ── Graphics modal overlay ──
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

      let close_btn = button(text("Close").size(13))
        .on_press(Message::CloseGraphicsModal)
        .padding([6, 16])
        .style(muted_button_style);

      let modal_content =
        container(column![graphic, close_btn].spacing(12).align_x(Center))
          .center(Fill)
          .padding(40);

      let backdrop = mouse_area(
        container(opaque(modal_content))
          .width(Fill)
          .height(Fill)
          .style(graphics_modal_backdrop_style),
      )
      .on_press(Message::CloseGraphicsModal);

      stack![main_view, backdrop].into()
    } else {
      main_view
    }
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
      || editor.output.as_ref().map_or(false, |o| {
        let d = o
          .replace("-Graphics-", "")
          .replace("-Graphics3D-", "")
          .replace("-Image-", "");
        !d.trim().is_empty()
      });
    let is_grouped = is_input && has_output && !in_preview;
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
      text.lines().nth(cursor_line).map_or(true, |line| {
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
        // Wrap selection with matching brackets/quotes
        if has_selection {
          if let Some(ref text) = key_press.text {
            let pair = match text.as_ref() {
              "{" => Some(('{', '}')),
              "[" => Some(('[', ']')),
              "\"" => Some(('"', '"')),
              "'" => Some(('\'', '\'')),
              "(" => Some(('(', ')')),
              _ => None,
            };
            if let Some((open, close)) = pair {
              return Some(text_editor::Binding::Custom(
                Message::WrapSelection(idx, open, close),
              ));
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
      // Double-click opens a fullscreen modal for detailed inspection.
      if let Some((ref img_handle, w, h)) = editor.graphics_image {
        let mut img_widget = image(img_handle.clone())
          .width(iced::Length::Fixed(w as f32))
          .height(iced::Length::Fixed(h as f32));
        if stale {
          img_widget = img_widget.opacity(0.3);
        }
        let clickable = mouse_area(container(img_widget).padding(4))
          .on_double_click(Message::OpenGraphicsModal(idx));
        output_col = output_col.push(clickable);
      } else if let Some(ref handle) = editor.graphics_handle {
        let mut svg_widget =
          svg::Svg::new(handle.clone()).width(iced::Length::Shrink);
        if stale {
          svg_widget = svg_widget.opacity(0.3);
        }
        let clickable = mouse_area(container(svg_widget).padding(4))
          .on_double_click(Message::OpenGraphicsModal(idx));
        output_col = output_col.push(clickable);
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
        let clickable = mouse_area(container(img_widget).padding(4))
          .on_double_click(Message::OpenGraphicsModal(idx));
        content_col = content_col.push(clickable);
      } else if let Some(ref handle) = editor.graphics_handle {
        let mut svg_widget =
          svg::Svg::new(handle.clone()).width(iced::Length::Shrink);
        if stale {
          svg_widget = svg_widget.opacity(0.3);
        }
        let clickable = mouse_area(container(svg_widget).padding(4))
          .on_double_click(Message::OpenGraphicsModal(idx));
        content_col = content_col.push(clickable);
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
    if let Some(line) = lines.get(anchor_line) {
      if line.ends_with(first_sel_line) {
        let anchor_col = line.len() - first_sel_line.len();
        return ((anchor_line, anchor_col), (cursor_line, cursor_col));
      }
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

const ICON_CHEVRON_DOWN: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>"#;
const ICON_CHEVRON_RIGHT: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m9 18 6-6-6-6"/></svg>"#;

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
    {
      if let Some(level) = heading_level(style) {
        stack.push((i, level));
      }
    }
  }
  hidden
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
      CellStyle::Chapter => {
        y += 8.0;
        write_text_lines(
          &mut elements,
          &mut y,
          trimmed,
          margin,
          20.0,
          "bold",
          "sans-serif",
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
          17.0,
          "bold",
          "sans-serif",
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
      CellStyle::Item => {
        let wrapped = word_wrap(trimmed, 78);
        write_text_lines(
          &mut elements,
          &mut y,
          &format!("• {wrapped}"),
          margin + 8.0,
          12.0,
          "normal",
          "serif",
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
          12.0,
          "normal",
          "serif",
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
}
