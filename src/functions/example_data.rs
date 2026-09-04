//! `ExampleData` — the bundled example datasets.
//!
//! Wolfram ships a large catalogue of example data grouped by type
//! (`"NetworkGraph"`, `"Text"`, `"Matrix"`, …). Woxi implements the same
//! four call forms against the datasets it bundles:
//!
//! - `ExampleData[]` — the available types.
//! - `ExampleData["type"]` — the available `{"type", "name"}` entries.
//! - `ExampleData[{"type", "name"}]` — the data itself.
//! - `ExampleData[{"type", "name"}, "property"]` — one property of it.
//!
//! Only `"NetworkGraph"` data is bundled so far, and only with the classic
//! networks listed in `resources/network_graphs.txt.gz`; a catalogued name
//! whose data is not bundled stays unevaluated rather than returning wrong
//! data, and a name that is not in the catalogue at all is reported as
//! `ExampleData::notent` (a collection Woxi does not serve as
//! `ExampleData::notcoll`, a property the data does not have as
//! `ExampleData::notpropx`). The bundled networks are assembled from the
//! publications they come from and carry the names Wolfram's catalogue
//! lists them under, so a script that asks for one of them by name gets the
//! same data from either engine.
//!
//! `"TestImage"` is a partial exception: its name catalogue is bundled (see
//! `TEST_IMAGE_NAMES`) so scripts that build UI from the catalogue — a
//! `Control` popup, `Thread[...]` over the name list — see the real
//! entries, but no photographic data is bundled, so
//! `ExampleData[{"TestImage", name}]` stays unevaluated for every name.

use std::sync::LazyLock;

#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{Expr, unevaluated};

/// One bundled network dataset.
struct NetworkGraph {
  name: &'static str,
  description: String,
  source: String,
  /// The vertices in Wolfram's order — integers when the dataset numbers
  /// its nodes, strings when it names them.
  vertices: Vec<Expr>,
  /// Undirected edges as index pairs into `vertices`.
  edges: Vec<(usize, usize)>,
}

/// The bundled `"NetworkGraph"` datasets, in catalogue order.
///
/// The resource is a line-oriented text file: one `:name`, `:description`,
/// `:source`, `:vertices` (a `|`-separated list) and `:edges` (a
/// comma-separated list of `i-j` index pairs) per dataset.
static NETWORK_GRAPHS: LazyLock<Vec<NetworkGraph>> = LazyLock::new(|| {
  use flate2::read::GzDecoder;
  use std::io::Read;

  let compressed = include_bytes!("../../resources/network_graphs.txt.gz");
  let mut decoder = GzDecoder::new(&compressed[..]);
  let mut text = String::new();
  decoder
    .read_to_string(&mut text)
    .expect("failed to decompress the example network graphs");

  let mut out: Vec<NetworkGraph> = Vec::new();
  for line in text.lines() {
    let Some((key, value)) = line.split_once(' ') else {
      continue;
    };
    if key == ":name" {
      out.push(NetworkGraph {
        // Leaked so the name can be handed out as a `&'static str` for the
        // lifetime of the process, like the other bundled data tables.
        name: Box::leak(value.to_string().into_boxed_str()),
        description: String::new(),
        source: String::new(),
        vertices: Vec::new(),
        edges: Vec::new(),
      });
    } else {
      let Some(current) = out.last_mut() else {
        continue;
      };
      match key {
        ":description" => current.description = value.to_string(),
        ":source" => current.source = value.to_string(),
        ":vertices" => {
          current.vertices = value
            .split('|')
            .map(|v| match v.parse::<i128>() {
              Ok(n) => Expr::Integer(n),
              Err(_) => Expr::String(v.to_string()),
            })
            .collect();
        }
        ":edges" => {
          current.edges = value
            .split(',')
            .filter_map(|e| {
              let (a, b) = e.split_once('-')?;
              Some((a.parse().ok()?, b.parse().ok()?))
            })
            .collect();
        }
        _ => {}
      }
    }
  }
  out
});

/// A second spelling Wolfram resolves to a catalogue name, so that a script
/// written against either one runs: `ExampleData[{"NetworkGraph",
/// "ZacharysKarateClub"}]` and `…"ZacharyKarateClub"…` are the same network.
const NETWORK_GRAPH_ALIASES: &[(&str, &str)] = &[
  ("ZacharysKarateClub", "ZacharyKarateClub"),
  ("BooksAboutUSPolitics", "USPoliticsBooks"),
];

/// The catalogue name `name` stands for, resolving the alternative
/// spellings above.
fn resolve_network_graph_name(name: &str) -> &str {
  NETWORK_GRAPH_ALIASES
    .iter()
    .find(|(alias, _)| *alias == name)
    .map_or(name, |(_, catalogue)| *catalogue)
}

/// The dataset named by `name`, if it is bundled.
fn network_graph(name: &str) -> Option<&'static NetworkGraph> {
  let name = resolve_network_graph_name(name);
  NETWORK_GRAPHS.iter().find(|g| g.name == name)
}

/// The properties `ExampleData[{"NetworkGraph", …}, prop]` understands, in
/// the alphabetical order Wolfram reports them in.
const NETWORK_GRAPH_PROPERTIES: &[&str] = &[
  "AdjacencyMatrix",
  "Description",
  "EdgeCount",
  "EdgeRules",
  "Graph",
  "Name",
  "Source",
  "VertexCount",
  "VertexList",
];

/// The `Graph[…]` for a bundled network.
fn network_graph_expr(g: &NetworkGraph) -> Expr {
  let edges: Vec<Expr> = g
    .edges
    .iter()
    .map(|&(a, b)| {
      call(
        "UndirectedEdge",
        vec![g.vertices[a].clone(), g.vertices[b].clone()],
      )
    })
    .collect();
  Expr::FunctionCall {
    name: "Graph".to_string(),
    args: vec![
      Expr::List(g.vertices.clone().into()),
      Expr::List(edges.into()),
    ]
    .into(),
  }
}

/// One property of a bundled network, or `None` when the property is not
/// one this dataset type provides.
fn network_graph_property(g: &NetworkGraph, property: &str) -> Option<Expr> {
  let string = |s: &str| Expr::String(s.to_string());
  Some(match property {
    "Graph" => network_graph_expr(g),
    "Name" => string(g.name),
    "Description" => string(&g.description),
    "Source" => string(&g.source),
    "VertexCount" => Expr::Integer(g.vertices.len() as i128),
    "EdgeCount" => Expr::Integer(g.edges.len() as i128),
    "VertexList" => Expr::List(g.vertices.clone().into()),
    "EdgeRules" => Expr::List(
      g.edges
        .iter()
        .map(|&(a, b)| Expr::Rule {
          pattern: Box::new(g.vertices[a].clone()),
          replacement: Box::new(g.vertices[b].clone()),
        })
        .collect::<Vec<_>>()
        .into(),
    ),
    "AdjacencyMatrix" => {
      let n = g.vertices.len();
      let mut rows = vec![vec![0i128; n]; n];
      for &(a, b) in &g.edges {
        rows[a][b] = 1;
        rows[b][a] = 1;
      }
      Expr::List(
        rows
          .into_iter()
          .map(|row| {
            Expr::List(
              row
                .into_iter()
                .map(Expr::Integer)
                .collect::<Vec<_>>()
                .into(),
            )
          })
          .collect::<Vec<_>>()
          .into(),
      )
    }
    "Properties" => Expr::List(
      NETWORK_GRAPH_PROPERTIES
        .iter()
        .map(|p| string(p))
        .collect::<Vec<_>>()
        .into(),
    ),
    _ => return None,
  })
}

/// The `"NetworkGraph"` catalogue of names, in Wolfram's order. Woxi bundles
/// the data for only a handful of them (`resources/network_graphs.txt.gz`),
/// but the full catalogue is exposed so that `ExampleData["NetworkGraph"]`
/// lists the same entries Wolfram lists and `ExampleData::notent` fires for
/// exactly the names Wolfram rejects. A catalogued name Woxi has no data for
/// stays unevaluated, same as an un-bundled `"TestImage"`.
const NETWORK_GRAPH_NAMES: &[&str] = &[
  "AmericanCollegeFootball",
  "AskOpinionRecall",
  "AskOpinionRecognition",
  "AstrophysicsCollaborations",
  "BeAskedOpinionRecall",
  "BeAskedOpinionRecognition",
  "BipartiteDiseasomeNetwork",
  "Brock2001",
  "Brock2002",
  "Brock2003",
  "Brock2004",
  "Brock4001",
  "Brock4002",
  "Brock4003",
  "Brock4004",
  "Brock8001",
  "Brock8002",
  "Brock8003",
  "Brock8004",
  "BuddingYeast",
  "CellOntology",
  "CFat2001",
  "CFat2002",
  "CFat2005",
  "CFat5001",
  "CFat50010",
  "CFat5002",
  "CFat5005",
  "CoauthorshipsInNetworkScience",
  "CondensedMatterCollaborations",
  "CondensedMatterCollaborations2003",
  "CondensedMatterCollaborations2005",
  "DavisSouthernWomen",
  "DiscussionRecall",
  "DiscussionRecognition",
  "DiseaseGeneNetwork",
  "DolphinSocialNetwork",
  "EastAfricaEmbassyAttacks",
  "EmailListMathGroup",
  "EuclidElements",
  "EurovisionVotes",
  "ExpandedAbortion",
  "ExpandedComputationalComplexity",
  "ExpandedComputationalGeometry",
  "ExpandedDeathPenalty",
  "ExpandedGenetic",
  "ExpandedGunControl",
  "ExpandedMovies",
  "ExpandedNetCensorship",
  "FamilyGathering",
  "FlorentineFamilies",
  "FreeAssociationNormsAppendixA",
  "Friendship",
  "Hamming102",
  "Hamming104",
  "Hamming62",
  "Hamming64",
  "Hamming82",
  "Hamming84",
  "HighEnergyPhysicsPhenomenology",
  "HighEnergyPhysicsTheory",
  "HighEnergyTheoryCollaborations",
  "HumanDiseaseNetwork",
  "Internet",
  "JazzMusicians",
  "Johnson1624",
  "Johnson3224",
  "Johnson824",
  "Johnson844",
  "Keller4",
  "Keller5",
  "Keller6",
  "LesMiserables",
  "MannA27",
  "MannA45",
  "MannA81",
  "MannA9",
  "MarvelUniverseSocialGraph",
  "MetabolicNetworkActinobacillusActinomycetemcomitans",
  "MetabolicNetworkAeropyrumPernix",
  "MetabolicNetworkAquifexAeolicus",
  "MetabolicNetworkArabidopsisThaliana",
  "MetabolicNetworkArchaeoglobusFulgidus",
  "MetabolicNetworkBacillusSubtilis",
  "MetabolicNetworkBorreliaBurgdorferi",
  "MetabolicNetworkCaenorhabditisElegans",
  "MetabolicNetworkCampylobacterJejuni",
  "MetabolicNetworkChlamydiaPneumoniae",
  "MetabolicNetworkChlamydiaTrachomatis",
  "MetabolicNetworkChlorobiumTepidum",
  "MetabolicNetworkClostridiumAcetobutylicum",
  "MetabolicNetworkDeinococcusRadiodurans",
  "MetabolicNetworkEmericellaNidulans",
  "MetabolicNetworkEnterococcusFaecalis",
  "MetabolicNetworkEscherichiaColi",
  "MetabolicNetworkHaemophilusInfluenzae",
  "MetabolicNetworkHelicobacterPylori",
  "MetabolicNetworkMethanobacteriumThermoautotrophicum",
  "MetabolicNetworkMethanococcusJannaschii",
  "MetabolicNetworkMycobacteriumBovis",
  "MetabolicNetworkMycobacteriumLeprae",
  "MetabolicNetworkMycobacteriumTuberculosis",
  "MetabolicNetworkMycoplasmaGenitalium",
  "MetabolicNetworkMycoplasmaPneumoniae",
  "MetabolicNetworkNeisseriaGonorrhoeae",
  "MetabolicNetworkNeisseriaMeningitidis",
  "MetabolicNetworkOryzaSativa",
  "MetabolicNetworkPorphyromonasGingivalis",
  "MetabolicNetworkPseudomonasAeruginosa",
  "MetabolicNetworkPyrococcusFuriosus",
  "MetabolicNetworkPyrococcusHorikoshii",
  "MetabolicNetworkRhodobacterCapsulatus",
  "MetabolicNetworkRickettsiaProwazekii",
  "MetabolicNetworkSaccharomycesCerevisiae",
  "MetabolicNetworkSalmonellaTyphi",
  "MetabolicNetworkStreptococcusPneumoniae",
  "MetabolicNetworkStreptococcusPyogenes",
  "MetabolicNetworkSynechocystisSp",
  "MetabolicNetworkThermotogaMaritima",
  "MetabolicNetworkTreponemaPallidum",
  "MetabolicNetworkYersiniaPestis",
  "NationalHockeyLeague",
  "OnlineSocialNetwork",
  "PerlModuleAuthors",
  "PGPNetwork",
  "PHat10001",
  "PHat10002",
  "PHat10003",
  "PHat15001",
  "PHat15002",
  "PHat15003",
  "PHat3001",
  "PHat3002",
  "PHat3003",
  "PHat5001",
  "PHat5002",
  "PHat5003",
  "PHat7001",
  "PHat7002",
  "PHat7003",
  "PoliticalBlogs",
  "PowerGrid",
  "ProteinInteraction",
  "RefinedAbortion",
  "RefinedComputationalComplexity",
  "RefinedComputationalGeometry",
  "RefinedDeathPenalty",
  "RefinedGenetic",
  "RefinedGunControl",
  "RefinedMovies",
  "RefinedNetCensorship",
  "RegularAbortion",
  "RegularComputationalComplexity",
  "RegularComputationalGeometry",
  "RegularDeathPenalty",
  "RegularGenetic",
  "RegularGunControl",
  "RegularMovies",
  "RegularNetCensorship",
  "San1000",
  "San200071",
  "San200072",
  "San200091",
  "San200092",
  "San200093",
  "San400051",
  "San400071",
  "San400072",
  "San400073",
  "San400091",
  "Sanr20007",
  "Sanr20009",
  "Sanr40005",
  "Sanr40007",
  "September11Terrorists",
  "SimpleFoodWeb",
  "SloveneParliamentaryParties",
  "TaggedTestImages",
  "URVEmailNetwork",
  "USPoliticsBooks",
  "WholeNetworkActinobacillusActinomycetemcomitans",
  "WholeNetworkAeropyrumPernix",
  "WholeNetworkAquifexAeolicus",
  "WholeNetworkArabidopsisThaliana",
  "WholeNetworkArchaeoglobusFulgidus",
  "WholeNetworkBacillusSubtilis",
  "WholeNetworkBorreliaBurgdorferi",
  "WholeNetworkCaenorhabditisElegans",
  "WholeNetworkCampylobacterJejuni",
  "WholeNetworkChlamydiaPneumoniae",
  "WholeNetworkChlamydiaTrachomatis",
  "WholeNetworkChlorobiumTepidum",
  "WholeNetworkClostridiumAcetobutylicum",
  "WholeNetworkDeinococcusRadiodurans",
  "WholeNetworkEmericellaNidulans",
  "WholeNetworkEnterococcusFaecalis",
  "WholeNetworkEscherichiaColi",
  "WholeNetworkHaemophilusInfluenzae",
  "WholeNetworkHelicobacterPylori",
  "WholeNetworkMethanobacteriumThermoautotrophicum",
  "WholeNetworkMethanococcusJannaschii",
  "WholeNetworkMycobacteriumBovis",
  "WholeNetworkMycobacteriumLeprae",
  "WholeNetworkMycobacteriumTuberculosis",
  "WholeNetworkMycoplasmaGenitalium",
  "WholeNetworkMycoplasmaPneumoniae",
  "WholeNetworkNeisseriaGonorrhoeae",
  "WholeNetworkNeisseriaMeningitidis",
  "WholeNetworkOryzaSativa",
  "WholeNetworkPorphyromonasGingivalis",
  "WholeNetworkPseudomonasAeruginosa",
  "WholeNetworkPyrococcusFuriosus",
  "WholeNetworkPyrococcusHorikoshii",
  "WholeNetworkRhodobacterCapsulatus",
  "WholeNetworkRickettsiaProwazekii",
  "WholeNetworkSaccharomycesCerevisiae",
  "WholeNetworkSalmonellaTyphi",
  "WholeNetworkStreptococcusPneumoniae",
  "WholeNetworkStreptococcusPyogenes",
  "WholeNetworkSynechocystisSp",
  "WholeNetworkThermotogaMaritima",
  "WholeNetworkTreponemaPallidum",
  "WholeNetworkYersiniaPestis",
  "WikiVote",
  "WordAdjacencies",
  "WorldCup1988",
  "WorldWideWeb",
  "ZacharyKarateClub",
];

/// The example-data types Woxi bundles.
const TYPES: &[&str] = &["NetworkGraph", "TestImage"];

/// The `"TestImage"` catalogue of names. Woxi bundles no photographic data
/// (there is no license to redistribute the actual pixels), so only the
/// name catalogue is exposed: `ExampleData["TestImage"]` returns the same
/// `{"TestImage", name}` pairs Wolfram lists, which is enough for scripts
/// that build UI from the catalogue (a `Control` popup, `Thread[...]` over
/// the name list, …). `ExampleData[{"TestImage", name}]` itself stays
/// unevaluated for every name, same as an un-bundled `NetworkGraph`.
const TEST_IMAGE_NAMES: &[&str] = &[
  "Aerial",
  "Aerial2",
  "Airplane",
  "Airplane2",
  "Airport",
  "APC",
  "Apples",
  "Boat",
  "Bridge",
  "CarAndAPC",
  "CarAndAPC2",
  "ChemicalPlant",
  "Clock",
  "Couple",
  "Couple2",
  "Elaine",
  "F16",
  "Flower",
  "Girl",
  "Girl2",
  "Girl3",
  "Gray21",
  "House",
  "House2",
  "JellyBeans",
  "JellyBeans2",
  "Man",
  "Mandrill",
  "Marruecos",
  "Moon",
  "Peppers",
  "RadcliffeCamera",
  "ResolutionChart",
  "Ruler",
  "Sailboat",
  "Splash",
  "Stall",
  "Tank",
  "Tank2",
  "Tank3",
  "Tiffany",
  "Tree",
  "Truck",
  "TruckAndAPC",
  "TruckAndAPC2",
  "U2",
  "Volubilis",
];

/// `ExampleData[…]` — see the module documentation for the call forms.
pub fn example_data_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("ExampleData", args));

  // ExampleData[] — the available types.
  if args.is_empty() {
    return Ok(Expr::List(
      TYPES
        .iter()
        .map(|t| Expr::String(t.to_string()))
        .collect::<Vec<_>>()
        .into(),
    ));
  }

  // ExampleData["type"] — the entries of that type, each a {type, name} pair.
  if let Expr::String(kind) = &args[0]
    && args.len() == 1
  {
    let Some(names) = collection_names(kind) else {
      notcoll(kind);
      return unevaluated();
    };
    return Ok(Expr::List(
      names
        .iter()
        .map(|name| {
          Expr::List(
            vec![Expr::String(kind.clone()), Expr::String(name.to_string())]
              .into(),
          )
        })
        .collect::<Vec<_>>()
        .into(),
    ));
  }

  // ExampleData[{"type", "name"}] / ExampleData[{"type", "name"}, "prop"]
  let Expr::List(spec) = &args[0] else {
    return unevaluated();
  };
  let (Some(Expr::String(kind)), Some(Expr::String(name))) =
    (spec.first(), spec.get(1))
  else {
    return unevaluated();
  };
  let Some(names) = collection_names(kind) else {
    notcoll(kind);
    return unevaluated();
  };
  // A name outside the catalogue is what wolframscript reports as `notent`;
  // a catalogued name Woxi carries no data for stays quietly unevaluated.
  let catalogue_name = if kind == "NetworkGraph" {
    resolve_network_graph_name(name)
  } else {
    name.as_str()
  };
  if !names.contains(&catalogue_name) {
    crate::emit_message(&format!(
      "ExampleData::notent: {name:?} is not a known entity for the \
       collection {kind:?}. Use ExampleData[{kind:?}] for a list of entities."
    ));
    return unevaluated();
  }
  if kind != "NetworkGraph" || args.len() > 2 {
    return unevaluated();
  }
  let Some(graph) = network_graph(name) else {
    return unevaluated();
  };
  let property = match args.get(1) {
    None => "Graph",
    Some(Expr::String(p)) => p.as_str(),
    Some(_) => return unevaluated(),
  };
  if let Some(value) = network_graph_property(graph, property) {
    Ok(value)
  } else {
    crate::emit_message(&format!(
      "ExampleData::notpropx: {property:?} is not a known property for \
       {{{kind:?}, {name:?}}}. Use ExampleData[{{{kind:?}, {name:?}}}, \
       \"Properties\"] for a list of properties."
    ));
    unevaluated()
  }
}

/// The catalogue of entity names for `kind`, if it is a collection Woxi
/// serves.
fn collection_names(kind: &str) -> Option<&'static [&'static str]> {
  match kind {
    "NetworkGraph" => Some(NETWORK_GRAPH_NAMES),
    "TestImage" => Some(TEST_IMAGE_NAMES),
    _ => None,
  }
}

/// Report `ExampleData::notcoll` for a collection Woxi does not serve.
fn notcoll(kind: &str) {
  crate::emit_message(&format!(
    "ExampleData::notcoll: {kind:?} is not a known collection for \
     ExampleData. Use ExampleData[] for a list of collections."
  ));
}
