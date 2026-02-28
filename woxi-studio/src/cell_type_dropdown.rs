//! A dropdown widget that shows an icon button (collapsed)
//! and a floating overlay menu with icon + text items (open).

use iced::advanced::layout::{self, Layout};
use iced::advanced::overlay;
use iced::advanced::renderer::{self, Quad};
use iced::advanced::text::{self, Text};
use iced::advanced::widget::{self, Widget};
use iced::advanced::{Clipboard, Renderer as _, Shell};
use iced::mouse;
use iced::widget::{button, row, svg};
use iced::{
  alignment, Background, Border, Center, Color, Element, Event,
  Font, Length, Pixels, Point, Rectangle, Size, Theme, Vector,
};

use iced::advanced::svg::{self as svg_core, Renderer as SvgRenderer};
use iced::advanced::text::Renderer as TextRenderer;

use crate::notebook::CellStyle;

// ── SVG icons ──────────────────────────────────────────────────────

const CHEVRON_DOWN_SVG: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>"#;
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

pub fn cell_style_icon(style: CellStyle) -> &'static str {
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

// ── Constants ──────────────────────────────────────────────────────

const ICON_SIZE: f32 = 12.0;
const TEXT_SIZE: f32 = 10.0;
const ITEM_PADDING_X: f32 = 8.0;
const ITEM_PADDING_Y: f32 = 4.0;
const ICON_TEXT_GAP: f32 = 6.0;
const ITEM_HEIGHT: f32 = ICON_SIZE + ITEM_PADDING_Y * 2.0;
const MENU_PADDING: f32 = 4.0;
const MENU_WIDTH: f32 = 120.0;

// ── Public API ─────────────────────────────────────────────────────

pub fn cell_type_dropdown<'a, Message: Clone + 'a>(
  current: CellStyle,
  is_open: bool,
  options: &'a [CellStyle],
  on_toggle: Message,
  on_select: impl Fn(CellStyle) -> Message + 'a,
) -> Element<'a, Message> {
  let underlay = make_trigger_button(current, on_toggle.clone());
  Element::new(CellTypeDropdownWidget {
    current,
    is_open,
    options,
    on_toggle,
    on_select: Box::new(on_select),
    underlay,
  })
}

struct CellTypeDropdownWidget<'a, Message> {
  current: CellStyle,
  is_open: bool,
  options: &'a [CellStyle],
  on_toggle: Message,
  on_select: Box<dyn Fn(CellStyle) -> Message + 'a>,
  underlay: Element<'a, Message>,
}

fn make_trigger_button<'a, Message: Clone + 'a>(
  current: CellStyle,
  on_toggle: Message,
) -> Element<'a, Message> {
  let icon_svg = svg::Handle::from_memory(
    cell_style_icon(current).as_bytes().to_vec(),
  );
  let chevron_svg =
    svg::Handle::from_memory(CHEVRON_DOWN_SVG.as_bytes().to_vec());

  button(
    row![
      svg::Svg::new(icon_svg)
        .width(14)
        .height(14)
        .style(gutter_icon_style),
      svg::Svg::new(chevron_svg)
        .width(8)
        .height(8)
        .style(gutter_icon_style),
    ]
    .align_y(Center)
    .spacing(2),
  )
  .on_press(on_toggle)
  .padding([3, 4])
  .style(trigger_button_style)
  .into()
}

impl<'a, Message: Clone + 'a> Widget<Message, Theme, iced::Renderer>
  for CellTypeDropdownWidget<'a, Message>
{
  fn size(&self) -> Size<Length> {
    Size::new(Length::Shrink, Length::Shrink)
  }

  fn layout(
    &mut self,
    tree: &mut widget::Tree,
    renderer: &iced::Renderer,
    limits: &layout::Limits,
  ) -> layout::Node {
    self
      .underlay
      .as_widget_mut()
      .layout(&mut tree.children[0], renderer, limits)
  }

  fn draw(
    &self,
    tree: &widget::Tree,
    renderer: &mut iced::Renderer,
    theme: &Theme,
    style: &renderer::Style,
    layout: Layout<'_>,
    cursor: mouse::Cursor,
    viewport: &Rectangle,
  ) {
    self.underlay.as_widget().draw(
      &tree.children[0],
      renderer,
      theme,
      style,
      layout,
      cursor,
      viewport,
    );
  }

  fn tag(&self) -> widget::tree::Tag {
    widget::tree::Tag::stateless()
  }

  fn state(&self) -> widget::tree::State {
    widget::tree::State::None
  }

  fn children(&self) -> Vec<widget::Tree> {
    vec![widget::Tree::new(self.underlay.as_widget())]
  }

  fn diff(&self, tree: &mut widget::Tree) {
    tree.diff_children(std::slice::from_ref(&self.underlay));
  }

  fn update(
    &mut self,
    tree: &mut widget::Tree,
    event: &Event,
    layout: Layout<'_>,
    cursor: mouse::Cursor,
    renderer: &iced::Renderer,
    clipboard: &mut dyn Clipboard,
    shell: &mut Shell<'_, Message>,
    viewport: &Rectangle,
  ) {
    self.underlay.as_widget_mut().update(
      &mut tree.children[0],
      event,
      layout,
      cursor,
      renderer,
      clipboard,
      shell,
      viewport,
    );
  }

  fn mouse_interaction(
    &self,
    tree: &widget::Tree,
    layout: Layout<'_>,
    cursor: mouse::Cursor,
    viewport: &Rectangle,
    renderer: &iced::Renderer,
  ) -> mouse::Interaction {
    self.underlay.as_widget().mouse_interaction(
      &tree.children[0],
      layout,
      cursor,
      viewport,
      renderer,
    )
  }

  fn overlay<'b>(
    &'b mut self,
    _tree: &'b mut widget::Tree,
    layout: Layout<'_>,
    _renderer: &iced::Renderer,
    _viewport: &Rectangle,
    translation: Vector,
  ) -> Option<overlay::Element<'b, Message, Theme, iced::Renderer>>
  {
    if !self.is_open {
      return None;
    }

    let bounds = layout.bounds();
    let position = Point::new(
      bounds.x + translation.x,
      bounds.y + bounds.height + translation.y + 2.0,
    );

    Some(overlay::Element::new(Box::new(DropdownOverlay {
      options: self.options,
      current: self.current,
      on_select: &self.on_select,
      on_toggle: self.on_toggle.clone(),
      position,
      hovered: None,
    })))
  }
}

// ── Overlay ────────────────────────────────────────────────────────

struct DropdownOverlay<'a, Message> {
  options: &'a [CellStyle],
  current: CellStyle,
  on_select: &'a dyn Fn(CellStyle) -> Message,
  on_toggle: Message,
  position: Point,
  hovered: Option<usize>,
}

impl<Message: Clone> DropdownOverlay<'_, Message> {
  fn menu_size(&self) -> Size {
    Size::new(
      MENU_WIDTH + MENU_PADDING * 2.0,
      ITEM_HEIGHT * self.options.len() as f32 + MENU_PADDING * 2.0,
    )
  }

  fn item_index_at(&self, pos: Point, bounds: Rectangle) -> Option<usize> {
    let relative_y = pos.y - bounds.y - MENU_PADDING;
    if relative_y < 0.0 {
      return None;
    }
    let idx = (relative_y / ITEM_HEIGHT) as usize;
    if idx < self.options.len() {
      Some(idx)
    } else {
      None
    }
  }
}

impl<Message: Clone>
  overlay::Overlay<Message, Theme, iced::Renderer>
  for DropdownOverlay<'_, Message>
{
  fn layout(
    &mut self,
    _renderer: &iced::Renderer,
    _bounds: Size,
  ) -> layout::Node {
    layout::Node::new(self.menu_size()).move_to(self.position)
  }

  fn update(
    &mut self,
    event: &Event,
    layout: Layout<'_>,
    cursor: mouse::Cursor,
    _renderer: &iced::Renderer,
    _clipboard: &mut dyn Clipboard,
    shell: &mut Shell<'_, Message>,
  ) {
    let bounds = layout.bounds();

    match event {
      Event::Mouse(mouse::Event::ButtonPressed(
        mouse::Button::Left,
      )) => {
        if let Some(pos) = cursor.position() {
          if bounds.contains(pos) {
            if let Some(idx) = self.item_index_at(pos, bounds) {
              shell.publish((self.on_select)(self.options[idx]));
            }
            shell.capture_event();
          } else {
            // Click outside: close the menu
            shell.publish(self.on_toggle.clone());
            shell.capture_event();
          }
        }
      }
      Event::Mouse(mouse::Event::CursorMoved { .. }) => {
        let new_hovered = cursor
          .position()
          .and_then(|pos| self.item_index_at(pos, bounds));
        if new_hovered != self.hovered {
          self.hovered = new_hovered;
          shell.request_redraw();
        }
      }
      _ => {}
    }
  }

  fn mouse_interaction(
    &self,
    layout: Layout<'_>,
    cursor: mouse::Cursor,
    _renderer: &iced::Renderer,
  ) -> mouse::Interaction {
    if cursor.is_over(layout.bounds()) {
      mouse::Interaction::Pointer
    } else {
      mouse::Interaction::default()
    }
  }

  fn draw(
    &self,
    renderer: &mut iced::Renderer,
    theme: &Theme,
    _style: &renderer::Style,
    layout: Layout<'_>,
    _cursor: mouse::Cursor,
  ) {
    let is_dark = !matches!(theme, Theme::Light);
    let bounds = layout.bounds();

    // Draw menu background
    renderer.fill_quad(
      Quad {
        bounds,
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
          offset: Vector::new(0.0, 2.0),
          blur_radius: 8.0,
        },
        snap: true,
      },
      Background::Color(if is_dark {
        Color::from_rgb(0.14, 0.14, 0.16)
      } else {
        Color::from_rgb(0.98, 0.98, 0.98)
      }),
    );

    let text_color = if is_dark {
      Color::from_rgb(0.78, 0.80, 0.85)
    } else {
      Color::from_rgb(0.20, 0.20, 0.25)
    };
    let selected_text_color = if is_dark {
      Color::from_rgb(0.50, 0.70, 1.0)
    } else {
      Color::from_rgb(0.15, 0.40, 0.80)
    };
    let icon_color = if is_dark {
      Color::from_rgb(0.65, 0.70, 0.78)
    } else {
      Color::from_rgb(0.35, 0.35, 0.40)
    };
    let selected_icon_color = selected_text_color;
    let hover_bg = if is_dark {
      Color::from_rgba(1.0, 1.0, 1.0, 0.08)
    } else {
      Color::from_rgba(0.0, 0.0, 0.0, 0.06)
    };
    let selected_bg = if is_dark {
      Color::from_rgba(0.30, 0.50, 1.0, 0.12)
    } else {
      Color::from_rgba(0.15, 0.40, 0.80, 0.08)
    };

    for (i, &style) in self.options.iter().enumerate() {
      let is_selected = style == self.current;
      let is_hovered = self.hovered == Some(i);

      let item_bounds = Rectangle {
        x: bounds.x + MENU_PADDING,
        y: bounds.y + MENU_PADDING + ITEM_HEIGHT * i as f32,
        width: bounds.width - MENU_PADDING * 2.0,
        height: ITEM_HEIGHT,
      };

      // Draw item background (selected or hovered)
      if is_selected || is_hovered {
        renderer.fill_quad(
          Quad {
            bounds: item_bounds,
            border: Border {
              radius: 3.0.into(),
              ..Border::default()
            },
            ..Quad::default()
          },
          Background::Color(if is_selected {
            selected_bg
          } else {
            hover_bg
          }),
        );
      }

      // Draw icon
      let icon_bounds = Rectangle {
        x: item_bounds.x + ITEM_PADDING_X,
        y: item_bounds.y + (ITEM_HEIGHT - ICON_SIZE) / 2.0,
        width: ICON_SIZE,
        height: ICON_SIZE,
      };

      let handle = svg::Handle::from_memory(
        cell_style_icon(style).as_bytes().to_vec(),
      );
      renderer.draw_svg(
        svg_core::Svg {
          handle,
          color: Some(if is_selected {
            selected_icon_color
          } else {
            icon_color
          }),
          rotation: iced::Radians(0.0),
          opacity: 1.0,
        },
        icon_bounds,
        bounds,
      );

      // Draw text
      let text_x =
        item_bounds.x + ITEM_PADDING_X + ICON_SIZE + ICON_TEXT_GAP;

      renderer.fill_text(
        Text {
          content: style.as_str().to_string(),
          bounds: Size::new(
            item_bounds.width - ITEM_PADDING_X * 2.0
              - ICON_SIZE
              - ICON_TEXT_GAP,
            item_bounds.height,
          ),
          size: Pixels(TEXT_SIZE),
          line_height: text::LineHeight::default(),
          font: Font::MONOSPACE,
          align_x: text::Alignment::Default,
          align_y: alignment::Vertical::Center,
          shaping: text::Shaping::default(),
          wrapping: text::Wrapping::default(),
        },
        Point::new(text_x, item_bounds.center_y()),
        if is_selected {
          selected_text_color
        } else {
          text_color
        },
        bounds,
      );
    }
  }
}

// ── Styles ─────────────────────────────────────────────────────────

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

fn trigger_button_style(
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
