//! `ExampleData` — example datasets, generated rather than bundled.
//!
//! Wolfram ships a large catalogue of example data grouped into collections
//! (`"TestImage"`, `"NetworkGraph"`, `"Statistics"`, …). Woxi serves the
//! same catalogue with the same call forms:
//!
//! - `ExampleData[]` — the collections.
//! - `ExampleData["coll"]` — the `{"coll", "name"}` entries of a collection.
//! - `ExampleData[{"coll", "name"}]` — the data itself.
//! - `ExampleData[{"coll", "name"}, "prop"]` — one property of it, with
//!   `"Properties"` listing them.
//!
//! The *structure* matches wolframscript exactly — the catalogue, property
//! names, the head and shape of every value, and the metadata (dimensions,
//! counts, sample rates, colour spaces, column names and types, …) — because
//! that is what scripts depend on. The *content* is invented: photographs
//! are crude drawings, 3D scans rough models, recordings synthesized sound,
//! books generated text, tables generated numbers with similar statistics.
//! Nothing but the generator code and a small metadata table
//! (`resources/example_data_meta.tsv.gz`, see [`meta`]) is bundled; every
//! value is generated on first request from a generator seeded with the
//! collection and entry name, so it is the same on every run. Descriptive
//! text (`"Description"`, `"Source"`, …) is written by Woxi.
//!
//! The exception is a handful of classic networks whose real data is
//! assembled from the original publications (see [`network_graph`]).
//!
//! Unknown collections report `ExampleData::notcoll`, unknown entries
//! `ExampleData::notent`, unknown properties `ExampleData::notpropx`.

use std::sync::LazyLock;

use super::*;

pub mod aerials;
pub mod animation;
pub mod audio;
pub mod dataset;
pub mod geometry_3d;
pub mod graph_models;
pub mod images;
pub mod instantiate;
pub mod linear_programming;
pub mod machine_learning;
pub mod matrix;
pub mod mesh;
pub mod meta;
pub mod network_graph;
pub mod photos;
pub mod raster;
pub mod rng;
pub mod scenes;
pub mod statistics;
pub mod synth;
pub mod tabular;
pub mod test_image_3d;
pub mod testimages;
pub mod text;
pub mod textures;
pub mod words;

use meta::{Cell, META, Row, Table};

/// One entry of a collection, as seen by a generator.
pub struct Entry<'a> {
  pub collection: &'a str,
  pub name: &'a str,
  pub table: &'a Table,
  pub row: &'a Row,
}

impl Entry<'_> {
  /// The parsed cell of `column`, if the entry has one.
  pub fn cell(&self, column: &str) -> Option<Cell> {
    self.table.raw(self.row, column).map(Cell::from_raw)
  }

  /// The exact value recorded for `column`, evaluated, if there is one.
  pub fn exact(&self, column: &str) -> Option<Expr> {
    match self.cell(column)? {
      Cell::Exact(e) => Some(evaluate(e)),
      Cell::Skeleton(_) => None,
    }
  }

  /// The skeleton recorded for `column`, if the value is not exact.
  pub fn skeleton(&self, column: &str) -> Option<Expr> {
    match self.cell(column)? {
      Cell::Skeleton(e) => Some(e),
      Cell::Exact(_) => None,
    }
  }

  /// An `@`-column (summary statistics and the like), parsed.
  pub fn stat(&self, column: &str) -> Option<Expr> {
    self
      .table
      .raw(self.row, column)
      .map(|raw| evaluate(meta::parse_value(raw)))
  }

  /// The entry's property names, in Wolfram's order.
  pub fn properties(&self) -> Vec<String> {
    match self.stat("@Properties") {
      Some(Expr::List(ref items)) => items
        .iter()
        .filter_map(|p| match p {
          Expr::String(s) => Some(s.clone()),
          _ => None,
        })
        .collect(),
      _ => Vec::new(),
    }
  }

  /// A random number generator seeded from this entry and `salt`.
  pub fn rng(&self, salt: &str) -> rng::Rng {
    rng::Rng::for_entry(self.collection, self.name, salt)
  }

  /// Every sub-entry is a row of its own; `{"TestImageSet", "Oxford2011_01"}`
  /// is requested like any other entry but not listed in the catalogue.
  fn listed(&self) -> bool {
    !matches!(self.stat("@Listed"), Some(Expr::Identifier(ref s)) if s == "False")
  }

  /// Entries Wolfram has retired from the catalogue but still serves with
  /// an `ExampleData::obsolete` warning (`{"TestImage", "Lena"}`).
  fn obsolete(&self) -> bool {
    matches!(self.stat("@Obsolete"), Some(Expr::Identifier(ref s)) if s == "True")
  }
}

fn evaluate(e: Expr) -> Expr {
  crate::evaluator::evaluate_expr_to_expr(&e).unwrap_or(e)
}

/// The generator of one collection: it produces the values the metadata
/// table cannot hold.
pub trait Collection: Sync {
  /// The value of `property` for `entry`, or of the entry itself when
  /// `property` is `None`. `None` means Woxi does not generate it (the
  /// call then stays unevaluated).
  fn value(&self, entry: &Entry, property: Option<&str>) -> Option<Expr>;

  /// Resolve an alternative spelling of an entry name.
  fn resolve<'n>(&self, name: &'n str) -> &'n str {
    name
  }
}

/// What `"Source"` says for generated data.
pub const GENERATED_SOURCE: &str = "Generated by Woxi: the structure follows \
  the original dataset, the values are synthetic.";

/// The generator for `collection`.
fn generator(collection: &str) -> Option<&'static dyn Collection> {
  Some(match collection {
    "NetworkGraph" => &network_graph::NetworkGraphs,
    "TestImage" | "AerialImage" | "Texture" | "ColorTexture" => &images::Images,
    "TestImageSet" => &images::ImageSets,
    "Text" => &text::Texts,
    "Geometry3D" => &geometry_3d::Geometry3D,
    "TestImage3D" => &test_image_3d::TestImages3D,
    "Statistics" => &statistics::StatisticsData,
    "Matrix" => &matrix::Matrices,
    "Audio" | "Sound" => &audio::Recordings,
    "MachineLearning" => &machine_learning::MachineLearning,
    "TestAnimation" => &animation::Animations,
    "Dataset" => &dataset::Datasets,
    "LinearProgramming" | "LinearOptimization" => {
      &linear_programming::LinearPrograms
    }
    _ => return None,
  })
}

/// A collection without generated values ("SpatialPointData", whose
/// catalogue is a fixed value): only what the table records is served.
struct MetadataOnly;

impl Collection for MetadataOnly {
  fn value(&self, _: &Entry, _: Option<&str>) -> Option<Expr> {
    None
  }
}

/// `ExampleData[]`.
static COLLECTIONS: LazyLock<Expr> =
  LazyLock::new(|| meta::parse_value(&META.collections));

/// `ExampleData[…]` — see the module documentation for the call forms.
pub fn example_data_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let echo = || Ok(unevaluated("ExampleData", args));

  if args.is_empty() {
    return Ok(COLLECTIONS.clone());
  }

  // ExampleData["coll"] — the listed entries, each a {coll, name} pair.
  if let Expr::String(collection) = &args[0] {
    if args.len() != 1 {
      return echo();
    }
    let Some(table) = META.table(collection) else {
      notcoll(&args[0]);
      return echo();
    };
    if let Some(value) = fixed_catalogue(table) {
      return Ok(value);
    }
    return Ok(catalogue(collection, table));
  }

  // ExampleData[{"coll", "name"}] / ExampleData[{"coll", "name"}, "prop"]
  let Expr::List(spec) = &args[0] else {
    return echo();
  };
  // Anything but a pair names no collection: `{"TestImage"}` and
  // `{"TestImage", "Moon", "x"}` are reported as a whole.
  let (Some(Expr::String(collection)), Some(name_expr), 2) =
    (spec.first(), spec.get(1), spec.len())
  else {
    notcoll(&args[0]);
    return echo();
  };
  let Some(table) = META.table(collection) else {
    notcoll(&spec[0]);
    return echo();
  };
  // A collection whose catalogue is a fixed value (`$Failed` for
  // "SpatialPointData", which needs Wolfram's server) answers every
  // request with it.
  if let Some(value) = fixed_catalogue(table) {
    return Ok(value);
  }
  let generator = generator(collection).unwrap_or(&MetadataOnly);
  let row = match name_expr {
    Expr::String(name) => table.row(generator.resolve(name)),
    _ => None,
  };
  let Some(row) = row else {
    crate::emit_message(&format!(
      "ExampleData::notent: {} is not a known entity for the \
       collection {collection:?}. Use ExampleData[{collection:?}] for a list \
       of entities.",
      crate::syntax::expr_to_input_form(name_expr)
    ));
    return echo();
  };
  let name = &row.name;
  let entry = Entry {
    collection,
    name: &row.name,
    table,
    row,
  };
  if entry.obsolete() {
    crate::emit_message(&format!(
      "ExampleData::obsolete: {{{collection}, {name}}} is being \
       deprecated, and may not be available in future versions of \
       Wolfram Language."
    ));
  }
  let property = match args.get(1) {
    None => None,
    Some(Expr::String(p)) => Some(p.as_str()),
    Some(_) => return echo(),
  };
  let Some(property) = property else {
    // The entry itself is one of its properties, or generated.
    return Ok(
      match entry.stat("@Default") {
        Some(Expr::String(ref p)) => property_value(generator, &entry, p),
        _ => generator.value(&entry, None),
      }
      .unwrap_or_else(|| unevaluated("ExampleData", args)),
    );
  };
  if property == "Properties" {
    return Ok(Expr::List(
      entry
        .properties()
        .iter()
        .map(|p| str_expr(p))
        .collect::<Vec<_>>()
        .into(),
    ));
  }
  if !entry.properties().iter().any(|p| p == property) {
    crate::emit_message(&format!(
      "ExampleData::notpropx: {property:?} is not a known property for \
       {{{collection:?}, {name:?}}}. Use ExampleData[{{{collection:?}, \
       {name:?}}}, \"Properties\"] for a list of properties."
    ));
    return echo();
  }
  Ok(
    property_value(generator, &entry, property)
      .unwrap_or_else(|| unevaluated("ExampleData", args)),
  )
}

/// A property's value: generated if the generator produces it, else the
/// exact value recorded in the metadata table.
fn property_value(
  generator: &dyn Collection,
  entry: &Entry,
  property: &str,
) -> Option<Expr> {
  generator
    .value(entry, Some(property))
    .or_else(|| entry.exact(property))
}

/// The recorded value of `ExampleData["coll"]` for a collection whose
/// catalogue is not a list of entries.
fn fixed_catalogue(table: &Table) -> Option<Expr> {
  if table.columns().first().map(String::as_str) != Some("@Catalogue") {
    return None;
  }
  let row = table.rows().first()?;
  Some(evaluate(meta::parse_value(table.raw(row, "@Catalogue")?)))
}

/// `ExampleData["coll"]`.
fn catalogue(collection: &str, table: &Table) -> Expr {
  Expr::List(
    table
      .rows()
      .iter()
      .filter(|row| {
        Entry {
          collection,
          name: &row.name,
          table,
          row,
        }
        .listed()
      })
      .map(|row| {
        Expr::List(vec![str_expr(collection), str_expr(&row.name)].into())
      })
      .collect::<Vec<_>>()
      .into(),
  )
}

/// Report `ExampleData::notcoll` for a collection Woxi does not know.
fn notcoll(collection: &Expr) {
  let shown = crate::syntax::expr_to_input_form(collection);
  crate::emit_message(&format!(
    "ExampleData::notcoll: {shown} is not a known collection for \
     ExampleData. Use ExampleData[] for a list of collections."
  ));
}

/// An `Expr::List` of strings.
pub fn string_list<S: AsRef<str>>(items: &[S]) -> Expr {
  Expr::List(
    items
      .iter()
      .map(|s| str_expr(s.as_ref()))
      .collect::<Vec<_>>()
      .into(),
  )
}
