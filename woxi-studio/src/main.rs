mod highlighter;
mod notebook;

use iced::keyboard;
use iced::widget::{
  button, column, container, focus_next, horizontal_rule,
  horizontal_space, pick_list, row, rule, scrollable, svg,
  text, text_editor, Column, Stack,
};
use iced::{
  Background, Border, Center, Color, Element, Fill, Font,
  Subscription, Task, Theme,
};
use iced::overlay::menu;

use notebook::{
  Cell, CellEntry, CellGroup, CellStyle, Notebook,
};
use std::path::PathBuf;
use std::sync::Arc;

fn main() -> iced::Result {
  iced::application(
    "Woxi Studio",
    WoxiStudio::update,
    WoxiStudio::view,
  )
  .subscription(WoxiStudio::subscription)
  .theme(|state: &WoxiStudio| state.theme.clone())
  .default_font(Font::MONOSPACE)
  .run_with(WoxiStudio::new)
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
  /// Style to use for new cells.
  new_cell_style: CellStyle,
  /// Whether preview mode is active (hides gutter, borders, etc).
  preview_mode: bool,
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
  /// Undo stack: previous text snapshots.
  undo_stack: Vec<String>,
  /// Redo stack: snapshots restored via undo.
  redo_stack: Vec<String>,
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

  // Export
  ExportAs(ExportFormat),
  FileExported(Result<PathBuf, FileError>),

  // Cell editing
  CellAction(usize, text_editor::Action),
  Undo(usize),
  Redo(usize),
  CellStyleChanged(usize, CellStyle),
  FocusCell(usize),

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

  // Keyboard
  KeyPressed(keyboard::Key, keyboard::Modifiers),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThemeChoice {
  Auto,
  Light,
  Dark,
}

impl std::fmt::Display for ThemeChoice {
  fn fmt(
    &self,
    f: &mut std::fmt::Formatter<'_>,
  ) -> std::fmt::Result {
    match self {
      ThemeChoice::Auto => write!(f, "Auto"),
      ThemeChoice::Light => write!(f, "Light"),
      ThemeChoice::Dark => write!(f, "Dark"),
    }
  }
}

impl ThemeChoice {
  const ALL: &'static [ThemeChoice] = &[
    ThemeChoice::Auto,
    ThemeChoice::Light,
    ThemeChoice::Dark,
  ];
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
}

impl std::fmt::Display for ExportFormat {
  fn fmt(
    &self,
    f: &mut std::fmt::Formatter<'_>,
  ) -> std::fmt::Result {
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
  ];
}

// ── Application Logic ───────────────────────────────────────────────

impl WoxiStudio {
  fn new() -> (Self, Task<Message>) {
    let mut notebook = Notebook::new();
    notebook.push_cell(Cell::new(
      CellStyle::Title,
      "Untitled Notebook",
    ));
    notebook.push_cell(Cell::new(CellStyle::Input, ""));

    let cell_editors = Self::editors_from_notebook(&notebook);

    let task = if let Some(path) = load_last_file_path() {
      Task::perform(
        open_file_path(path),
        Message::FileOpened,
      )
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
        new_cell_style: CellStyle::Input,
        preview_mode: false,
      },
      task,
    )
  }

  /// Build editor state from a notebook.
  /// Output/Print cells within a group are attached to the
  /// preceding Input/Code cell rather than shown separately.
  fn editors_from_notebook(
    notebook: &Notebook,
  ) -> Vec<CellEditor> {
    let mut editors = Vec::new();

    for entry in &notebook.cells {
      match entry {
        CellEntry::Single(cell) => {
          if matches!(
            cell.style,
            CellStyle::Output | CellStyle::Print
          ) {
            continue;
          }
          editors.push(CellEditor {
            content: text_editor::Content::with_text(
              &cell.content,
            ),
            style: cell.style,
            output: None,
            stdout: None,
            graphics_svg: None,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
          });
        }
        CellEntry::Group(group) => {
          let cells = &group.cells;
          let mut i = 0;
          while i < cells.len() {
            let cell = &cells[i];
            if matches!(
              cell.style,
              CellStyle::Input | CellStyle::Code
            ) {
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
                    output =
                      Some(cells[j].content.clone());
                  }
                  CellStyle::Print => {
                    stdout =
                      Some(cells[j].content.clone());
                  }
                  _ => {}
                }
                j += 1;
              }
              editors.push(CellEditor {
                content: text_editor::Content::with_text(
                  &cell.content,
                ),
                style: cell.style,
                output,
                stdout,
                graphics_svg: None,
                undo_stack: Vec::new(),
                redo_stack: Vec::new(),
              });
              i = j;
            } else if matches!(
              cell.style,
              CellStyle::Output | CellStyle::Print
            ) {
              // Skip standalone output/print in groups
              i += 1;
            } else {
              editors.push(CellEditor {
                content: text_editor::Content::with_text(
                  &cell.content,
                ),
                style: cell.style,
                output: None,
                stdout: None,
                graphics_svg: None,
                undo_stack: Vec::new(),
                redo_stack: Vec::new(),
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
      let content =
        editor.content.text().trim_end().to_string();
      let cell = Cell::new(editor.style, content);

      // Group input cells with their output
      if editor.style == CellStyle::Input {
        if let Some(ref output) = editor.output {
          let output_cell =
            Cell::new(CellStyle::Output, output.clone());
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
      Message::NewNotebook => {
        self.file_path = None;
        self.notebook = Notebook::new();
        self.notebook.push_cell(Cell::new(
          CellStyle::Title,
          "Untitled Notebook",
        ));
        self.notebook
          .push_cell(Cell::new(CellStyle::Input, ""));
        self.cell_editors =
          Self::editors_from_notebook(&self.notebook);
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
          Ok((path, contents)) => {
            match notebook::parse_notebook(&contents) {
              Ok(nb) => {
                self.cell_editors =
                  Self::editors_from_notebook(&nb);
                self.notebook = nb;
                self.status = format!(
                  "Opened: {}",
                  path.display()
                );
                save_last_file_path(&path);
                self.file_path = Some(path);
                self.is_dirty = false;
                self.focused_cell =
                  if self.cell_editors.is_empty() {
                    None
                  } else {
                    Some(0)
                  };
              }
              Err(e) => {
                self.status =
                  format!("Parse error: {e}");
              }
            }
          }
          Err(FileError::DialogClosed) => {
            self.status = String::from("Open cancelled");
          }
          Err(FileError::IoError(e)) => {
            self.status =
              format!("Error opening file: {e:?}");
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
        Task::perform(
          save_file(None, content),
          Message::FileSaved,
        )
      }

      Message::FileSaved(result) => {
        self.is_loading = false;
        match result {
          Ok(path) => {
            self.status =
              format!("Saved: {}", path.display());
            save_last_file_path(&path);
            self.file_path = Some(path);
            self.is_dirty = false;
          }
          Err(FileError::DialogClosed) => {
            self.status = String::from("Save cancelled");
          }
          Err(FileError::IoError(e)) => {
            self.status =
              format!("Error saving: {e:?}");
          }
        }
        Task::none()
      }

      Message::ExportAs(format) => {
        if self.is_loading {
          return Task::none();
        }
        self.sync_notebook_from_editors();
        let (content, filter_name, extension) = match format
        {
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
        };
        self.is_loading = true;
        self.status =
          format!("Exporting as {format}...");
        Task::perform(
          export_file(filter_name, extension, content),
          Message::FileExported,
        )
      }

      Message::FileExported(result) => {
        self.is_loading = false;
        match result {
          Ok(path) => {
            self.status = format!(
              "Exported: {}",
              path.display()
            );
          }
          Err(FileError::DialogClosed) => {
            self.status =
              String::from("Export cancelled");
          }
          Err(FileError::IoError(e)) => {
            self.status =
              format!("Error exporting: {e:?}");
          }
        }
        Task::none()
      }

      Message::CellAction(idx, action) => {
        if idx < self.cell_editors.len() {
          self.focused_cell = Some(idx);
          let is_edit = action.is_edit();
          if is_edit {
            // Snapshot current text for undo
            let snap =
              self.cell_editors[idx].content.text();
            self.cell_editors[idx]
              .undo_stack
              .push(snap);
            self.cell_editors[idx].redo_stack.clear();
          }
          self.cell_editors[idx].content.perform(action);
          if is_edit {
            self.is_dirty = true;
          }
        }
        Task::none()
      }

      Message::Undo(idx) => {
        if idx < self.cell_editors.len() {
          if let Some(prev) =
            self.cell_editors[idx].undo_stack.pop()
          {
            let current =
              self.cell_editors[idx].content.text();
            self.cell_editors[idx]
              .redo_stack
              .push(current);
            self.cell_editors[idx].content =
              text_editor::Content::with_text(&prev);
            self.is_dirty = true;
          }
        }
        Task::none()
      }

      Message::Redo(idx) => {
        if idx < self.cell_editors.len() {
          if let Some(next) =
            self.cell_editors[idx].redo_stack.pop()
          {
            let current =
              self.cell_editors[idx].content.text();
            self.cell_editors[idx]
              .undo_stack
              .push(current);
            self.cell_editors[idx].content =
              text_editor::Content::with_text(&next);
            self.is_dirty = true;
          }
        }
        Task::none()
      }

      Message::CellStyleChanged(idx, style) => {
        if idx < self.cell_editors.len() {
          self.cell_editors[idx].style = style;
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

      Message::FocusCell(idx) => {
        if idx < self.cell_editors.len() {
          self.focused_cell = Some(idx);
        }
        self.cell_type_menu_open = None;
        Task::none()
      }

      Message::AddCellBelow(idx) => {
        let insert_at =
          (idx + 1).min(self.cell_editors.len());
        self.cell_editors.insert(
          insert_at,
          CellEditor {
            content: text_editor::Content::new(),
            style: self.new_cell_style,
            output: None,
            stdout: None,
            graphics_svg: None,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
          },
        );
        self.focused_cell = Some(insert_at);
        self.is_dirty = true;
        focus_next()
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
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
          },
        );
        self.focused_cell = Some(insert_at);
        self.is_dirty = true;
        focus_next()
      }

      Message::DeleteCell(idx) => {
        if self.cell_editors.len() > 1
          && idx < self.cell_editors.len()
        {
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
          let code = self.cell_editors[idx]
            .content
            .text()
            .trim()
            .to_string();
          if !code.is_empty() {
            woxi::clear_state();
            match woxi::interpret_with_stdout(&code) {
              Ok(result) => {
                self.cell_editors[idx].output =
                  Some(result.result);
                self.cell_editors[idx].stdout =
                  if result.stdout.is_empty() {
                    None
                  } else {
                    Some(result.stdout)
                  };
                self.cell_editors[idx].graphics_svg =
                  result.graphics;
                self.status = format!(
                  "Evaluated cell {} successfully",
                  idx + 1
                );
              }
              Err(e) => {
                self.cell_editors[idx].output =
                  Some(format!("Error: {e}"));
                self.cell_editors[idx].stdout = None;
                self.cell_editors[idx].graphics_svg = None;
                self.status = format!(
                  "Cell {} evaluation error",
                  idx + 1
                );
              }
            }
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
            let code = self.cell_editors[idx]
              .content
              .text()
              .trim()
              .to_string();
            if !code.is_empty() {
              match woxi::interpret_with_stdout(&code) {
                Ok(result) => {
                  self.cell_editors[idx].output =
                    Some(result.result);
                  self.cell_editors[idx].stdout =
                    if result.stdout.is_empty() {
                      None
                    } else {
                      Some(result.stdout)
                    };
                  self.cell_editors[idx].graphics_svg =
                    result.graphics;
                }
                Err(e) => {
                  self.cell_editors[idx].output =
                    Some(format!("Error: {e}"));
                  self.cell_editors[idx].stdout = None;
                  self.cell_editors[idx].graphics_svg =
                    None;
                }
              }
            }
          }
        }
        self.status =
          String::from("All cells evaluated");
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
          if let keyboard::Key::Character("d") = key.as_ref()
          {
            if let Some(idx) = self.focused_cell {
              self.cell_editors[idx].content.perform(
                text_editor::Action::Edit(
                  text_editor::Edit::Delete,
                ),
              );
              self.is_dirty = true;
            }
            return Task::none();
          }
        }

        // Ctrl+W: delete previous word
        if modifiers.control() {
          if let keyboard::Key::Character("w") = key.as_ref()
          {
            if let Some(idx) = self.focused_cell {
              self.cell_editors[idx].content.perform(
                text_editor::Action::Select(
                  text_editor::Motion::WordLeft,
                ),
              );
              self.cell_editors[idx].content.perform(
                text_editor::Action::Edit(
                  text_editor::Edit::Backspace,
                ),
              );
              self.is_dirty = true;
            }
            return Task::none();
          }
        }

        // Shift+Enter to evaluate current cell
        if modifiers.shift() {
          if let keyboard::Key::Named(
            keyboard::key::Named::Enter,
          ) = key.as_ref()
          {
            if let Some(idx) = self.focused_cell {
              // Undo the newline the text editor just inserted
              self.cell_editors[idx].content.perform(
                text_editor::Action::Edit(
                  text_editor::Edit::Backspace,
                ),
              );
              return self
                .update(Message::EvaluateCell(idx));
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
    let eval_all_svg = svg::Handle::from_memory(
      PLAY_CIRCLE_SVG.as_bytes().to_vec(),
    );
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
        .on_press_maybe(
          (!self.is_loading).then_some(Message::OpenFile)
        )
        .padding([3, 8])
        .style(muted_button_style),
      button(text("Save").size(11))
        .on_press_maybe(
          self.is_dirty.then_some(Message::SaveFile)
        )
        .padding([3, 8])
        .style(muted_button_style),
      button(text("Save As").size(11))
        .on_press(Message::SaveFileAs)
        .padding([3, 8])
        .style(muted_button_style),
      pick_list(
        ExportFormat::ALL,
        None::<ExportFormat>,
        Message::ExportAs,
      )
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
      horizontal_space(),
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
    let cells: Element<'_, Message> = if self
      .cell_editors
      .is_empty()
    {
      container(
        text("Empty notebook. Click '+' to add a cell.")
          .size(13),
      )
      .center_x(Fill)
      .padding(40)
      .into()
    } else {
      let mut col = Column::new().spacing(0).width(Fill);

      if !self.preview_mode {
        // Add cell divider above the first cell
        col =
          col.push(self.view_add_cell_divider_above(0));
      }

      for (idx, editor) in
        self.cell_editors.iter().enumerate()
      {
        // Add cell divider between cells
        if !self.preview_mode && idx > 0 {
          col = col.push(self.view_add_cell_divider(
            idx.saturating_sub(1),
          ));
        }

        let is_focused = self.focused_cell == Some(idx);
        let cell_el = self.view_cell(idx, editor, is_focused);

        // Overlay the cell type dropdown on top of the cell
        if self.cell_type_menu_open == Some(idx) {
          let menu_el = container(
            self.view_cell_type_menu(idx, editor),
          )
          .width(Fill)
          .align_y(iced::alignment::Vertical::Bottom);

          col = col.push(
            Stack::new()
              .push(cell_el)
              .push(menu_el)
              .width(Fill),
          );
        } else {
          col = col.push(cell_el);
        }
      }

      if !self.preview_mode {
        // Final add-cell divider after last cell
        col = col.push(self.view_add_cell_divider(
          self.cell_editors.len().saturating_sub(1),
        ));
      }

      scrollable(
        container(col.max_width(800))
          .center_x(Fill),
      )
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

    let dirty_marker =
      if self.is_dirty { " [modified]" } else { "" };

    let status_bar = row![
      text(format!("{file_label}{dirty_marker}")).size(11),
      text("  |  ").size(11),
      text(&self.status).size(11),
      text("  |  ").size(11),
      text(format!("{} cells", self.cell_editors.len()))
        .size(11),
    ]
    .spacing(4)
    .padding([3, 8]);

    // ── Layout ──
    column![toolbar, horizontal_rule(1).style(separator_style), cells, status_bar,]
      .spacing(0)
      .into()
  }

  /// Small "+" divider above a cell (inserts before it).
  fn view_add_cell_divider_above(
    &self,
    idx: usize,
  ) -> Element<'_, Message> {
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
  fn view_add_cell_divider(
    &self,
    idx: usize,
  ) -> Element<'_, Message> {
    container(
      button(text("+").size(10))
        .on_press(Message::AddCellBelow(idx))
        .padding([0, 8])
        .style(add_cell_button_style),
    )
    .center_x(Fill)
    .padding([2, 0])
    .into()
  }

  fn view_cell<'a>(
    &'a self,
    idx: usize,
    editor: &'a CellEditor,
    _is_focused: bool,
  ) -> Element<'a, Message> {
    let is_input = editor.style == CellStyle::Input
      || editor.style == CellStyle::Code;

    // ── Left gutter: style picker + delete ──
    let mut gutter =
      Column::new().spacing(2).width(if self.preview_mode {
        iced::Length::Shrink
      } else {
        iced::Length::Fixed(60.0)
      });

    if !self.preview_mode {
      // Cell type: icon-only button; opens menu as overlay
      let current_icon_svg = svg::Handle::from_memory(
        cell_style_icon(editor.style).as_bytes().to_vec(),
      );
      gutter = gutter.push(
        button(
          svg::Svg::new(current_icon_svg)
            .width(16)
            .height(16)
            .style(gutter_icon_style),
        )
        .on_press(Message::ToggleCellTypeMenu(idx))
        .padding([3, 4])
        .style(cell_type_button_style),
      );

      let trash_svg = svg::Handle::from_memory(
        TRASH_ICON_SVG.as_bytes().to_vec(),
      );
      gutter = gutter.push(
        button(
          svg::Svg::new(trash_svg)
            .width(14)
            .height(14)
            .style(trash_icon_style),
        )
        .on_press_maybe(
          (self.cell_editors.len() > 1)
            .then_some(Message::DeleteCell(idx)),
        )
        .padding([2, 4])
        .style(trash_button_style),
      );
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
      || editor
        .output
        .as_ref()
        .map_or(false, |o| {
          let d = o
            .replace("-Graphics-", "")
            .replace("-Graphics3D-", "")
            .replace("-Image-", "");
          !d.trim().is_empty()
        });
    let is_grouped = is_input && has_output && !in_preview;
    let cell_editor = text_editor(&editor.content)
      .on_action(move |action| {
        Message::CellAction(idx, action)
      })
      .key_binding(move |key_press| {
        let text_editor::KeyPress {
          key, modifiers, ..
        } = &key_press;
        if modifiers.command() {
          match key.as_ref() {
            keyboard::Key::Character("z")
              if modifiers.shift() =>
            {
              return Some(
                text_editor::Binding::Custom(
                  Message::Redo(idx),
                ),
              );
            }
            keyboard::Key::Character("z") => {
              return Some(
                text_editor::Binding::Custom(
                  Message::Undo(idx),
                ),
              );
            }
            _ => {}
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
        } else if is_grouped {
          grouped_editor_style(theme, status, cell_style)
        } else {
          cell_editor_style(theme, status, cell_style)
        }
      })
      .highlight_with::<highlighter::WolframHighlighter>(
        highlighter::WolframSettings {
          enabled: is_input,
        },
        highlighter::format_highlight,
      );

    // ── Content column: editor + outputs ──

    let mut content_col =
      Column::new().spacing(0).width(Fill);
    content_col = content_col.push(cell_editor);

    if is_grouped {
      // Thin separator between input and output
      content_col = content_col.push(
        horizontal_rule(1).style(separator_style),
      );
    }

    // Stdout (Print output)
    if let Some(ref stdout) = editor.stdout {
      let stdout_display = container(
        text(stdout).size(12).font(Font::MONOSPACE),
      )
      .padding(6)
      .width(Fill);

      content_col = content_col.push(stdout_display);
    }

    // Graphics SVG rendering
    if let Some(ref svg_data) = editor.graphics_svg {
      let handle = svg::Handle::from_memory(
        svg_data.as_bytes().to_vec(),
      );
      let svg_widget = svg::Svg::new(handle)
        .width(iced::Length::Shrink);

      content_col = content_col.push(
        container(svg_widget).padding(4),
      );
    }

    // Text output (filter out graphics placeholders)
    if let Some(ref output) = editor.output {
      let display = output
        .replace("-Graphics-", "")
        .replace("-Graphics3D-", "")
        .replace("-Image-", "");
      let display = display.trim().to_string();
      if !display.is_empty() {
        let output_display = container(
          text(display).size(12).font(Font::MONOSPACE),
        )
        .padding(6)
        .width(Fill);

        content_col = content_col.push(output_display);
      }
    }

    // Wrap grouped input+output in a single rounded container
    let content_el: Element<'a, Message> = if is_grouped {
      container(content_col)
        .width(Fill)
        .style(grouped_cell_style)
        .into()
    } else {
      content_col.into()
    };

    // ── Right side: play button for input cells ──
    let right_side: Element<'a, Message> =
      if !self.preview_mode && is_input {
        container(
          button(text("\u{25B6}").size(14))
            .on_press(Message::EvaluateCell(idx))
            .padding([4, 8])
            .style(muted_button_style),
        )
        .into()
      } else {
        // Empty spacer
        text("").into()
      };

    let cell_row = row![gutter, content_el, right_side]
      .spacing(4)
      .padding([3, 6]);

    container(cell_row).width(Fill).into()
  }

  /// Build the cell type dropdown menu (shown below the cell).
  fn view_cell_type_menu<'a>(
    &'a self,
    idx: usize,
    editor: &'a CellEditor,
  ) -> Element<'a, Message> {
    let mut menu_col = Column::new().spacing(1).padding(4);
    for &style in CELL_STYLES {
      let icon_svg = svg::Handle::from_memory(
        cell_style_icon(style).as_bytes().to_vec(),
      );
      let is_selected = editor.style == style;
      menu_col = menu_col.push(
        button(
          row![
            svg::Svg::new(icon_svg)
              .width(12)
              .height(12)
              .style(if is_selected {
                gutter_icon_selected_style
              } else {
                gutter_icon_style
              }),
            text(style.as_str()).size(9),
          ]
          .align_y(Center)
          .spacing(4),
        )
        .on_press(Message::CellStyleChanged(idx, style))
        .padding([2, 6])
        .width(Fill)
        .style(if is_selected {
          cell_type_menu_selected_style
        } else {
          cell_type_menu_item_style
        }),
      );
    }

    container(
      container(menu_col)
        .style(cell_type_menu_container_style)
        .padding(2),
    )
    .padding(iced::Padding::default().left(12))
    .into()
  }
}

// ── Event handling ──────────────────────────────────────────────────

fn handle_event(
  event: iced::Event,
  _status: iced::event::Status,
  _id: iced::window::Id,
) -> Option<Message> {
  if let iced::Event::Keyboard(
    keyboard::Event::KeyPressed {
      key, modifiers, ..
    },
  ) = event
  {
    // Shift+Enter: always handle (even when text editor has focus)
    if modifiers.shift() {
      if let keyboard::Key::Named(
        keyboard::key::Named::Enter,
      ) = key.as_ref()
      {
        return Some(Message::KeyPressed(key, modifiers));
      }
    }

    // Ctrl shortcuts for text editing
    if modifiers.control() {
      match key.as_ref() {
        keyboard::Key::Character("d")
        | keyboard::Key::Character("w") => {
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

// ── Custom styles ───────────────────────────────────────────────────

fn separator_style(theme: &Theme) -> rule::Style {
  let is_dark = !matches!(theme, Theme::Light);
  rule::Style {
    color: if is_dark {
      Color::from_rgb(0.22, 0.22, 0.25)
    } else {
      Color::from_rgb(0.82, 0.82, 0.82)
    },
    width: 1,
    radius: 0.0.into(),
    fill_mode: rule::FillMode::Full,
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
    // Slightly brighter than the app background
    style.background =
      Background::Color(Color::from_rgb(0.16, 0.16, 0.18));
    if matches!(status, text_editor::Status::Focused) {
      style.border.color =
        Color::from_rgb(0.30, 0.30, 0.38);
    }
  } else {
    // Light mode: subtle off-white background for input cells
    style.background =
      Background::Color(Color::from_rgb(0.97, 0.97, 0.98));
    style.border.color = Color::from_rgb(0.82, 0.82, 0.85);
    if matches!(status, text_editor::Status::Focused) {
      style.border.color =
        Color::from_rgb(0.55, 0.55, 0.65);
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

fn grouped_cell_style(theme: &Theme) -> container::Style {
  let is_dark = !matches!(theme, Theme::Light);
  container::Style {
    background: None,
    border: Border {
      color: if is_dark {
        Color::from_rgb(0.22, 0.22, 0.25)
      } else {
        Color::from_rgb(0.78, 0.78, 0.80)
      },
      width: 1.0,
      radius: 6.0.into(),
    },
    ..container::Style::default()
  }
}

fn grouped_editor_style(
  theme: &Theme,
  status: text_editor::Status,
  cell_style: CellStyle,
) -> text_editor::Style {
  let mut style = cell_editor_style(theme, status, cell_style);
  // No border — the outer grouped container provides it
  style.border = Border {
    color: Color::TRANSPARENT,
    width: 0.0,
    radius: 0.0.into(),
  };
  style
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
    icon: Color::TRANSPARENT,
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

fn muted_button_style(
  theme: &Theme,
  status: button::Status,
) -> button::Style {
  let mut style = button::primary(theme, status);
  let is_dark = !matches!(theme, Theme::Light);
  if is_dark {
    style.background = Some(Background::Color(match status {
      button::Status::Active => {
        Color::from_rgb(0.18, 0.26, 0.40)
      }
      button::Status::Hovered => {
        Color::from_rgb(0.22, 0.32, 0.48)
      }
      button::Status::Pressed => {
        Color::from_rgb(0.15, 0.22, 0.35)
      }
      button::Status::Disabled => {
        Color::from_rgb(0.14, 0.16, 0.22)
      }
    }));
    style.text_color = Color::from_rgb(0.78, 0.82, 0.90);
  }
  style
}

fn trash_button_style(
  theme: &Theme,
  status: button::Status,
) -> button::Style {
  let mut style = button::text(theme, status);
  // Only show background on hover
  match status {
    button::Status::Hovered | button::Status::Pressed => {
      let is_dark = !matches!(theme, Theme::Light);
      style.background =
        Some(Background::Color(if is_dark {
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

fn eval_all_icon_style(
  theme: &Theme,
  _status: svg::Status,
) -> svg::Style {
  let is_dark = !matches!(theme, Theme::Light);
  svg::Style {
    color: Some(if is_dark {
      Color::from_rgb(0.45, 0.78, 0.45)
    } else {
      Color::from_rgb(0.20, 0.55, 0.20)
    }),
  }
}

fn cell_type_button_style(
  theme: &Theme,
  status: button::Status,
) -> button::Style {
  let is_dark = !matches!(theme, Theme::Light);
  let mut style = button::text(theme, status);
  style.border.radius = 4.0.into();
  match status {
    button::Status::Hovered | button::Status::Pressed => {
      style.background =
        Some(Background::Color(if is_dark {
          Color::from_rgba(1.0, 1.0, 1.0, 0.10)
        } else {
          Color::from_rgba(0.0, 0.0, 0.0, 0.08)
        }));
    }
    _ => {
      style.background = None;
    }
  }
  style
}

fn cell_type_menu_item_style(
  theme: &Theme,
  status: button::Status,
) -> button::Style {
  let is_dark = !matches!(theme, Theme::Light);
  let mut style = button::text(theme, status);
  style.border.radius = 3.0.into();
  style.text_color = if is_dark {
    Color::from_rgb(0.70, 0.72, 0.78)
  } else {
    Color::from_rgb(0.30, 0.30, 0.35)
  };
  match status {
    button::Status::Hovered | button::Status::Pressed => {
      style.background =
        Some(Background::Color(if is_dark {
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

fn cell_type_menu_selected_style(
  theme: &Theme,
  status: button::Status,
) -> button::Style {
  let is_dark = !matches!(theme, Theme::Light);
  let mut style = button::text(theme, status);
  style.border.radius = 3.0.into();
  style.text_color = if is_dark {
    Color::from_rgb(0.50, 0.70, 1.0)
  } else {
    Color::from_rgb(0.15, 0.40, 0.80)
  };
  style.background =
    Some(Background::Color(if is_dark {
      Color::from_rgba(0.30, 0.50, 1.0, 0.12)
    } else {
      Color::from_rgba(0.15, 0.40, 0.80, 0.08)
    }));
  style
}

fn cell_type_menu_container_style(
  theme: &Theme,
) -> container::Style {
  let is_dark = !matches!(theme, Theme::Light);
  container::Style {
    background: Some(Background::Color(if is_dark {
      Color::from_rgb(0.14, 0.14, 0.16)
    } else {
      Color::from_rgb(0.98, 0.98, 0.98)
    })),
    border: Border {
      color: if is_dark {
        Color::from_rgb(0.25, 0.25, 0.28)
      } else {
        Color::from_rgb(0.80, 0.80, 0.82)
      },
      width: 1.0,
      radius: 6.0.into(),
    },
    shadow: iced::Shadow {
      color: Color::from_rgba(0.0, 0.0, 0.0, 0.15),
      offset: iced::Vector::new(0.0, 2.0),
      blur_radius: 8.0,
    },
    ..container::Style::default()
  }
}

fn gutter_icon_selected_style(
  theme: &Theme,
  _status: svg::Status,
) -> svg::Style {
  let is_dark = !matches!(theme, Theme::Light);
  svg::Style {
    color: Some(if is_dark {
      Color::from_rgb(0.50, 0.70, 1.0)
    } else {
      Color::from_rgb(0.15, 0.40, 0.80)
    }),
  }
}

fn trash_icon_style(
  theme: &Theme,
  _status: svg::Status,
) -> svg::Style {
  let is_dark = !matches!(theme, Theme::Light);
  svg::Style {
    color: Some(if is_dark {
      Color::from_rgb(0.65, 0.65, 0.70)
    } else {
      Color::from_rgb(0.40, 0.40, 0.45)
    }),
  }
}

fn gutter_icon_style(
  theme: &Theme,
  _status: svg::Status,
) -> svg::Style {
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
  let is_dark = !matches!(theme, Theme::Light);
  let (bg, text_color, border_color) = if is_dark {
    match status {
      pick_list::Status::Hovered | pick_list::Status::Opened => (
        Color::from_rgb(0.22, 0.32, 0.48),
        Color::from_rgb(0.78, 0.82, 0.90),
        Color::from_rgb(0.22, 0.32, 0.48),
      ),
      _ => (
        Color::from_rgb(0.18, 0.26, 0.40),
        Color::from_rgb(0.78, 0.82, 0.90),
        Color::from_rgb(0.18, 0.26, 0.40),
      ),
    }
  } else {
    let base = pick_list::default(theme, status);
    return pick_list::Style {
      border: Border {
        radius: 2.0.into(),
        ..base.border
      },
      ..base
    };
  };
  pick_list::Style {
    text_color,
    placeholder_color: text_color,
    handle_color: text_color,
    background: Background::Color(bg),
    border: Border {
      color: border_color,
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
    style.background =
      Background::Color(Color::from_rgb(0.14, 0.14, 0.16));
    style.border.color = Color::from_rgb(0.22, 0.22, 0.25);
    if matches!(
      status,
      pick_list::Status::Hovered | pick_list::Status::Opened
    ) {
      style.border.color =
        Color::from_rgb(0.30, 0.30, 0.38);
    }
  }
  style
}

fn dropdown_menu_style(theme: &Theme) -> menu::Style {
  let mut style = menu::default(theme);
  let is_dark = !matches!(theme, Theme::Light);
  if is_dark {
    style.background =
      Background::Color(Color::from_rgb(0.14, 0.14, 0.16));
    style.border.color = Color::from_rgb(0.22, 0.22, 0.25);
  }
  style
}

const TRASH_ICON_SVG: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M3 6h18"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>"#;

const PLAY_CIRCLE_SVG: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 9.003a1 1 0 0 1 1.517-.859l4.997 2.997a1 1 0 0 1 0 1.718l-4.997 2.997A1 1 0 0 1 9 14.996z"/><circle cx="12" cy="12" r="10"/></svg>"#;

// ── Cell type icons (Lucide) ─────────────────────────────────────────

const ICON_HEADING_1: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12h8"/><path d="M4 18V6"/><path d="M12 18V6"/><path d="m17 12 3-2v8"/></svg>"#;

const ICON_HEADING_2: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12h8"/><path d="M4 18V6"/><path d="M12 18V6"/><path d="M21 18h-4c0-4 4-3 4-6 0-1.5-2-2.5-4-1"/></svg>"#;

const ICON_HEADING_3: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12h8"/><path d="M4 18V6"/><path d="M12 18V6"/><path d="M17.5 10.5c1.7-1 3.5 0 3.5 1.5a2 2 0 0 1-2 2"/><path d="M17 17.5c2 1.5 4 .3 4-1.5a2 2 0 0 0-2-2"/></svg>"#;

const ICON_HEADING_4: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 18V6"/><path d="M17 10v3a1 1 0 0 0 1 1h3"/><path d="M21 10v8"/><path d="M4 12h8"/><path d="M4 18V6"/></svg>"#;

const ICON_HEADING_5: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12h8"/><path d="M4 18V6"/><path d="M12 18V6"/><path d="M17 13v-3h4"/><path d="M17 17.7c.4.2.8.3 1.3.3 1.5 0 2.7-1.1 2.7-2.5S19.8 13 18.3 13H17"/></svg>"#;

const ICON_CODE: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m16 18 6-6-6-6"/><path d="m8 6-6 6 6 6"/></svg>"#;

const ICON_RECTANGLE_ELLIPSIS: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="12" x="2" y="6" rx="2"/><path d="M12 12h.01"/><path d="M17 12h.01"/><path d="M7 12h.01"/></svg>"#;

const ICON_TYPE: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4v16"/><path d="M4 7V5a1 1 0 0 1 1-1h14a1 1 0 0 1 1 1v2"/><path d="M9 20h6"/></svg>"#;

const ICON_FILE_BRACES: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 22a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h8a2.4 2.4 0 0 1 1.704.706l3.588 3.588A2.4 2.4 0 0 1 20 8v12a2 2 0 0 1-2 2z"/><path d="M14 2v5a1 1 0 0 0 1 1h5"/><path d="M10 12a1 1 0 0 0-1 1v1a1 1 0 0 1-1 1 1 1 0 0 1 1 1v1a1 1 0 0 0 1 1"/><path d="M14 18a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1 1 1 0 0 1-1-1v-1a1 1 0 0 0-1-1"/></svg>"#;

const ICON_TERMINAL: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19h8"/><path d="m4 17 6-6-6-6"/></svg>"#;

const ICON_EYE: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0"/><circle cx="12" cy="12" r="3"/></svg>"#;

const ICON_EYE_OFF: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.733 5.076a10.744 10.744 0 0 1 11.205 6.575 1 1 0 0 1 0 .696 10.747 10.747 0 0 1-1.444 2.49"/><path d="M14.084 14.158a3 3 0 0 1-4.242-4.242"/><path d="M17.479 17.499a10.75 10.75 0 0 1-15.417-5.151 1 1 0 0 1 0-.696 10.75 10.75 0 0 1 4.446-5.143"/><path d="m2 2 20 20"/></svg>"#;

fn cell_style_icon(style: CellStyle) -> &'static str {
  match style {
    CellStyle::Title => ICON_HEADING_1,
    CellStyle::Subtitle => ICON_HEADING_2,
    CellStyle::Section => ICON_HEADING_3,
    CellStyle::Subsection => ICON_HEADING_4,
    CellStyle::Subsubsection => ICON_HEADING_5,
    CellStyle::Text => ICON_TYPE,
    CellStyle::Input => ICON_CODE,
    CellStyle::Output => ICON_RECTANGLE_ELLIPSIS,
    CellStyle::Code => ICON_FILE_BRACES,
    CellStyle::Print => ICON_TERMINAL,
  }
}

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
  Some(
    PathBuf::from(home)
      .join(".config")
      .join("woxi-studio"),
  )
}

fn save_last_file_path(path: &std::path::Path) {
  if let Some(dir) = state_dir() {
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(
      dir.join("last_file"),
      path.display().to_string(),
    );
  }
}

fn load_last_file_path() -> Option<PathBuf> {
  let dir = state_dir()?;
  let content =
    std::fs::read_to_string(dir.join("last_file")).ok()?;
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

async fn open_file(
) -> Result<(PathBuf, Arc<String>), FileError> {
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
