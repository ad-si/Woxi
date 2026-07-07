mod miscellaneous_tests
{
  mod precision_display;
  mod wikidata;
  mod parser;
  mod high_level_functions;
  mod list;

  mod svg_rendering;
  #[path = "svg_rendering.rs"]
  mod box_representation;
  #[path = "svg_rendering.rs"]
  mod light_dark_theme_colors;
  #[path = "svg_rendering.rs"]
  mod triangle_rendering;
  #[path = "svg_rendering.rs"]
  mod traditional_form_boxes;
}
