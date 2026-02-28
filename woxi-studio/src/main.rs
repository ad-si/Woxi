mod highlighter;
mod notebook;

use iced::keyboard;
use iced::widget::{
  button, column, container, horizontal_rule, pick_list, row,
  rule, scrollable, svg, text, text_editor, Column,
};
use iced::{
  Border, Center, Color, Element, Fill, Font, Subscription,
  Task, Theme,
};

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
  /// Style to use for new cells.
  new_cell_style: CellStyle,
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

  // Keyboard
  KeyPressed(keyboard::Key, keyboard::Modifiers),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThemeChoice {
  Light,
  Dark,
}

impl std::fmt::Display for ThemeChoice {
  fn fmt(
    &self,
    f: &mut std::fmt::Formatter<'_>,
  ) -> std::fmt::Result {
    match self {
      ThemeChoice::Light => write!(f, "Light"),
      ThemeChoice::Dark => write!(f, "Dark"),
    }
  }
}

impl ThemeChoice {
  const ALL: &'static [ThemeChoice] =
    &[ThemeChoice::Light, ThemeChoice::Dark];
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

    (
      Self {
        file_path: None,
        notebook,
        cell_editors,
        focused_cell: Some(1),
        is_dirty: false,
        is_loading: false,
        status: String::from("Ready"),
        theme: Theme::Dark,
        new_cell_style: CellStyle::Input,
      },
      Task::none(),
    )
  }

  /// Build editor state from a notebook.
  fn editors_from_notebook(
    notebook: &Notebook,
  ) -> Vec<CellEditor> {
    let flat = notebook.flat_cells();
    flat
      .into_iter()
      .map(|(_, cell)| CellEditor {
        content: text_editor::Content::with_text(
          &cell.content,
        ),
        style: cell.style,
        output: None,
        stdout: None,
        graphics_svg: None,
      })
      .collect()
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
          self.cell_editors[idx].content.perform(action);
          if is_edit {
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
        Task::none()
      }

      Message::FocusCell(idx) => {
        if idx < self.cell_editors.len() {
          self.focused_cell = Some(idx);
        }
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
          },
        );
        self.focused_cell = Some(insert_at);
        self.is_dirty = true;
        Task::none()
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
          },
        );
        self.focused_cell = Some(insert_at);
        self.is_dirty = true;
        Task::none()
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
          && self.cell_editors[idx].style == CellStyle::Input
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
          if self.cell_editors[idx].style == CellStyle::Input
          {
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
        self.theme = match choice {
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

        // Shift+Enter to evaluate current cell
        if modifiers.shift() {
          if let keyboard::Key::Named(
            keyboard::key::Named::Enter,
          ) = key.as_ref()
          {
            if let Some(idx) = self.focused_cell {
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
    keyboard::on_key_press(|key, modifiers| {
      let msg = Message::KeyPressed(key, modifiers);
      Some(msg)
    })
  }

  fn view(&self) -> Element<'_, Message> {
    // ── Toolbar ──
    let toolbar = row![
      button("New")
        .on_press(Message::NewNotebook)
        .padding([3, 10]),
      button("Open")
        .on_press_maybe(
          (!self.is_loading).then_some(Message::OpenFile)
        )
        .padding([3, 10]),
      button("Save")
        .on_press_maybe(
          self.is_dirty.then_some(Message::SaveFile)
        )
        .padding([3, 10]),
      button("Save As")
        .on_press(Message::SaveFileAs)
        .padding([3, 10]),
      pick_list(
        ExportFormat::ALL,
        None::<ExportFormat>,
        Message::ExportAs,
      )
      .placeholder("Export")
      .text_size(12)
      .padding([2, 6]),
      text(" | ").size(12),
      button("Eval All")
        .on_press(Message::EvaluateAll)
        .padding([3, 10]),
      text(" | ").size(12),
      pick_list(
        ThemeChoice::ALL,
        Some(match self.theme {
          Theme::Light => ThemeChoice::Light,
          _ => ThemeChoice::Dark,
        }),
        Message::ThemeChanged,
      )
      .text_size(12)
      .padding([2, 6]),
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

      // Add cell divider above the first cell
      col = col.push(self.view_add_cell_divider_above(0));

      for (idx, editor) in
        self.cell_editors.iter().enumerate()
      {
        // Add cell divider between cells
        if idx > 0 {
          col = col.push(self.view_add_cell_divider(
            idx.saturating_sub(1),
          ));
        }

        let is_focused = self.focused_cell == Some(idx);
        col =
          col.push(self.view_cell(idx, editor, is_focused));
      }

      // Final add-cell divider after last cell
      col = col.push(self.view_add_cell_divider(
        self.cell_editors.len().saturating_sub(1),
      ));

      scrollable(col).height(Fill).into()
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
        .padding([0, 8]),
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
        .padding([0, 8]),
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
      Column::new().spacing(2).width(60);

    gutter = gutter.push(
      pick_list(
        CELL_STYLES,
        Some(editor.style),
        move |s| Message::CellStyleChanged(idx, s),
      )
      .text_size(10)
      .padding([1, 3]),
    );

    gutter = gutter.push(
      button(
        text("\u{1F5D1}") // 🗑
          .size(10),
      )
      .on_press_maybe(
        (self.cell_editors.len() > 1)
          .then_some(Message::DeleteCell(idx)),
      )
      .padding([1, 4]),
    );

    // ── Text editor ──
    let font_size = match editor.style {
      CellStyle::Title => 20.0,
      CellStyle::Subtitle => 16.0,
      CellStyle::Section => 15.0,
      CellStyle::Subsection => 14.0,
      CellStyle::Subsubsection => 13.0,
      _ => 13.0,
    };

    let cell_editor = text_editor(&editor.content)
      .on_action(move |action| {
        Message::CellAction(idx, action)
      })
      .height(iced::Length::Shrink)
      .padding(6)
      .size(font_size)
      .highlight_with::<highlighter::WolframHighlighter>(
        highlighter::WolframSettings {
          enabled: is_input,
        },
        highlighter::format_highlight,
      );

    // ── Content column: editor + outputs ──
    let mut content_col =
      Column::new().spacing(3).width(Fill);
    content_col = content_col.push(cell_editor);

    // Stdout (Print output)
    if let Some(ref stdout) = editor.stdout {
      let stdout_display = container(
        text(stdout).size(12).font(Font::MONOSPACE),
      )
      .padding(6)
      .width(Fill)
      .style(output_box_style);

      content_col = content_col.push(stdout_display);
    }

    // Graphics SVG rendering
    if let Some(ref svg_data) = editor.graphics_svg {
      let handle = svg::Handle::from_memory(
        svg_data.as_bytes().to_vec(),
      );
      let svg_widget =
        svg::Svg::new(handle).width(Fill);

      content_col = content_col.push(
        container(svg_widget).padding(4).width(Fill),
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
        .width(Fill)
        .style(output_box_style);

        content_col = content_col.push(output_display);
      }
    }

    // ── Right side: play button for input cells ──
    let right_side: Element<'a, Message> = if is_input {
      container(
        button(text("\u{25B6}").size(14))
          .on_press(Message::EvaluateCell(idx))
          .padding([4, 8]),
      )
      .into()
    } else {
      // Empty spacer
      text("").into()
    };

    let cell_row = row![gutter, content_col, right_side]
      .spacing(4)
      .padding([3, 6]);

    container(
      column![
        container(cell_row).width(Fill),
        horizontal_rule(1).style(separator_style)
      ]
      .spacing(0),
    )
    .width(Fill)
    .into()
  }
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

fn output_box_style(theme: &Theme) -> container::Style {
  let is_dark = !matches!(theme, Theme::Light);
  if is_dark {
    container::Style {
      border: Border {
        color: Color::from_rgb(0.22, 0.22, 0.25),
        width: 1.0,
        radius: 4.0.into(),
      },
      ..container::rounded_box(theme)
    }
  } else {
    container::rounded_box(theme)
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


// ── File I/O ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum FileError {
  DialogClosed,
  IoError(std::io::ErrorKind),
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
