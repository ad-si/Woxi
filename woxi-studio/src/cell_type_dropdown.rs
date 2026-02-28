//! A dropdown widget that shows an icon button (collapsed)
//! and a floating overlay menu with icon + text items (open).

use iced::advanced::layout::{self, Layout};
use iced::advanced::overlay;
use iced::advanced::renderer;
use iced::advanced::widget::{self, Widget};
use iced::advanced::{Clipboard, Shell};
use iced::mouse;
use iced::widget::{button, container, row, svg, text};
use iced::{
  Background, Border, Center, Color, Element, Event, Length, Point,
  Rectangle, Size, Theme, Vector,
};

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
}

impl<'a, Message: Clone>
  overlay::Overlay<Message, Theme, iced::Renderer>
  for DropdownOverlay<'a, Message>
{
  fn layout(
    &mut self,
    renderer: &iced::Renderer,
    _bounds: Size,
  ) -> layout::Node {
    let mut element = self.build_menu();
    let mut tree = widget::Tree::new(element.as_widget());
    let limits =
      layout::Limits::new(Size::ZERO, Size::new(200.0, 600.0));
    let node = element.as_widget_mut().layout(
      &mut tree, renderer, &limits,
    );
    node.move_to(self.position)
  }

  fn draw(
    &self,
    renderer: &mut iced::Renderer,
    theme: &Theme,
    style: &renderer::Style,
    layout: Layout<'_>,
    cursor: mouse::Cursor,
  ) {
    let element = self.build_menu();
    let tree = widget::Tree::new(element.as_widget());
    element.as_widget().draw(
      &tree, renderer, theme, style, layout, cursor,
      &layout.bounds(),
    );
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
    if let Event::Mouse(mouse::Event::ButtonPressed(
      mouse::Button::Left,
    )) = event
    {
      if let Some(pos) = cursor.position() {
        let bounds = layout.bounds();
        if bounds.contains(pos) {
          // Determine which item was clicked by position
          let relative_y = pos.y - bounds.y - 8.0; // outer + inner padding
          let item_height = 24.0;
          let idx = (relative_y / item_height) as usize;
          if idx < self.options.len() {
            shell.publish((self.on_select)(self.options[idx]));
          }
        } else {
          // Click outside: close the menu
          shell.publish(self.on_toggle.clone());
        }
      }
    }
  }
}

impl<'a, Message: Clone> DropdownOverlay<'a, Message> {
  fn build_menu(&self) -> Element<'a, Message> {
    let mut col =
      iced::widget::Column::new().spacing(1).padding(4);

    for &style in self.options {
      let icon_svg = svg::Handle::from_memory(
        cell_style_icon(style).as_bytes().to_vec(),
      );
      let is_selected = style == self.current;

      col = col.push(
        container(
          row![
            svg::Svg::new(icon_svg)
              .width(12)
              .height(12)
              .style(if is_selected {
                selected_icon_style
              } else {
                gutter_icon_style
              }),
            text(style.as_str()).size(10),
          ]
          .align_y(Center)
          .spacing(6),
        )
        .padding([3, 8])
        .width(Length::Fill)
        .style(if is_selected {
          selected_item_container_style
        } else {
          menu_item_container_style
        }),
      );
    }

    container(col)
      .style(menu_container_style)
      .into()
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

fn selected_icon_style(
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

fn menu_container_style(theme: &Theme) -> container::Style {
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
      offset: Vector::new(0.0, 2.0),
      blur_radius: 8.0,
    },
    ..container::Style::default()
  }
}

fn menu_item_container_style(
  _theme: &Theme,
) -> container::Style {
  container::Style {
    background: None,
    border: Border {
      radius: 3.0.into(),
      ..Border::default()
    },
    ..container::Style::default()
  }
}

fn selected_item_container_style(
  theme: &Theme,
) -> container::Style {
  let is_dark = !matches!(theme, Theme::Light);
  container::Style {
    background: Some(Background::Color(if is_dark {
      Color::from_rgba(0.30, 0.50, 1.0, 0.12)
    } else {
      Color::from_rgba(0.15, 0.40, 0.80, 0.08)
    })),
    border: Border {
      radius: 3.0.into(),
      ..Border::default()
    },
    ..container::Style::default()
  }
}
