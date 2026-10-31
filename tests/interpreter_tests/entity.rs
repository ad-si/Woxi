use super::*;

mod entity_store_tests {
  use super::*;

  #[test]
  fn entity_store_creates_normalized_form() {
    clear_state();
    assert_eq!(
      interpret(
        "EntityStore[{\"Pet\" -> <|\"Entities\" -> <|\"cat1\" -> <|\"Name\" -> \"Mittens\"|>|>, \"Properties\" -> <|\"Name\" -> <||>|>|>}]"
      )
      .unwrap(),
      "EntityStore[<|Types -> <|Pet -> <|Entities -> <|cat1 -> <|Name -> Mittens|>|>, Properties -> <|Name -> <||>|>|>|>|>]"
    );
  }

  #[test]
  fn entity_register_returns_type_list() {
    clear_state();
    assert_eq!(
      interpret(
        "store = EntityStore[{\"Pet\" -> <|\"Entities\" -> <||>, \"Properties\" -> <||>|>}]; EntityRegister[store]"
      )
      .unwrap(),
      "{Pet}"
    );
  }

  #[test]
  fn entity_register_multiple_types() {
    clear_state();
    assert_eq!(
      interpret(
        "store = EntityStore[{\"Dog\" -> <|\"Entities\" -> <||>, \"Properties\" -> <||>|>, \"Cat\" -> <|\"Entities\" -> <||>, \"Properties\" -> <||>|>}]; EntityRegister[store]"
      )
      .unwrap(),
      "{Cat, Dog}"
    );
  }

  #[test]
  fn entity_value_single_property() {
    clear_state();
    assert_eq!(
      interpret(
        "store = EntityStore[{\"Pet\" -> <|\"Entities\" -> <|\"cat1\" -> <|\"Name\" -> \"Mittens\", \"Age\" -> 3|>|>, \"Properties\" -> <|\"Name\" -> <||>, \"Age\" -> <||>|>|>}]; EntityRegister[store]; EntityValue[Entity[\"Pet\", \"cat1\"], \"Name\"]"
      )
      .unwrap(),
      "Mittens"
    );
  }

  #[test]
  fn entity_value_integer_property() {
    clear_state();
    assert_eq!(
      interpret(
        "store = EntityStore[{\"Pet\" -> <|\"Entities\" -> <|\"cat1\" -> <|\"Name\" -> \"Mittens\", \"Age\" -> 3|>|>, \"Properties\" -> <|\"Name\" -> <||>, \"Age\" -> <||>|>|>}]; EntityRegister[store]; EntityValue[Entity[\"Pet\", \"cat1\"], \"Age\"]"
      )
      .unwrap(),
      "3"
    );
  }

  #[test]
  fn entity_value_multiple_properties() {
    clear_state();
    assert_eq!(
      interpret(
        "store = EntityStore[{\"Pet\" -> <|\"Entities\" -> <|\"cat1\" -> <|\"Name\" -> \"Mittens\", \"Age\" -> 3|>|>, \"Properties\" -> <|\"Name\" -> <||>, \"Age\" -> <||>|>|>}]; EntityRegister[store]; EntityValue[Entity[\"Pet\", \"cat1\"], {\"Name\", \"Age\"}]"
      )
      .unwrap(),
      "{Mittens, 3}"
    );
  }

  #[test]
  fn entity_value_multiple_entities() {
    clear_state();
    assert_eq!(
      interpret(
        "store = EntityStore[{\"Pet\" -> <|\"Entities\" -> <|\"cat1\" -> <|\"Name\" -> \"Mittens\"|>, \"dog1\" -> <|\"Name\" -> \"Rex\"|>|>, \"Properties\" -> <|\"Name\" -> <||>|>|>}]; EntityRegister[store]; EntityValue[{Entity[\"Pet\", \"cat1\"], Entity[\"Pet\", \"dog1\"]}, \"Name\"]"
      )
      .unwrap(),
      "{Mittens, Rex}"
    );
  }

  #[test]
  fn entity_value_unknown_type() {
    clear_state();
    assert_eq!(
      interpret("EntityValue[Entity[\"Nonexistent\", \"x\"], \"Name\"]")
        .unwrap(),
      "Missing[UnknownType, Nonexistent]"
    );
  }

  #[test]
  fn entity_value_type_level_properties() {
    clear_state();
    assert_eq!(
      interpret(
        "store = EntityStore[{\"Pet\" -> <|\"Entities\" -> <|\"cat1\" -> <|\"Name\" -> \"Mittens\", \"Age\" -> 3|>|>, \"Properties\" -> <|\"Name\" -> <||>, \"Age\" -> <||>|>|>}]; EntityRegister[store]; EntityValue[\"Pet\", \"Properties\"]"
      )
      .unwrap(),
      "{EntityProperty[Pet, Age], EntityProperty[Pet, Name]}"
    );
  }

  #[test]
  fn entity_list_basic() {
    clear_state();
    assert_eq!(
      interpret(
        "store = EntityStore[{\"Pet\" -> <|\"Entities\" -> <|\"cat1\" -> <|\"Name\" -> \"Mittens\"|>, \"dog1\" -> <|\"Name\" -> \"Rex\"|>|>, \"Properties\" -> <|\"Name\" -> <||>|>|>}]; EntityRegister[store]; EntityList[\"Pet\"]"
      )
      .unwrap(),
      "{Entity[Pet, cat1], Entity[Pet, dog1]}"
    );
  }

  #[test]
  fn entity_list_unknown_type() {
    clear_state();
    assert_eq!(
      interpret("EntityList[\"Nonexistent\"]").unwrap(),
      "Missing[UnknownType, Nonexistent]"
    );
  }

  #[test]
  fn entity_properties_basic() {
    clear_state();
    assert_eq!(
      interpret(
        "store = EntityStore[{\"Pet\" -> <|\"Entities\" -> <|\"cat1\" -> <|\"Name\" -> \"Mittens\", \"Age\" -> 3|>|>, \"Properties\" -> <|\"Name\" -> <||>, \"Age\" -> <||>|>|>}]; EntityRegister[store]; EntityProperties[\"Pet\"]"
      )
      .unwrap(),
      "{EntityProperty[Pet, Age], EntityProperty[Pet, Name]}"
    );
  }

  #[test]
  fn entity_property_access_shorthand() {
    clear_state();
    assert_eq!(
      interpret(
        "store = EntityStore[{\"Pet\" -> <|\"Entities\" -> <|\"cat1\" -> <|\"Name\" -> \"Mittens\", \"Age\" -> 3|>|>, \"Properties\" -> <|\"Name\" -> <||>, \"Age\" -> <||>|>|>}]; EntityRegister[store]; Entity[\"Pet\", \"cat1\"][\"Name\"]"
      )
      .unwrap(),
      "Mittens"
    );
  }

  #[test]
  fn entity_stores_lists_registered() {
    clear_state();
    assert_eq!(
      interpret(
        "store = EntityStore[{\"Pet\" -> <|\"Entities\" -> <|\"cat1\" -> <|\"Name\" -> \"Mittens\"|>|>, \"Properties\" -> <|\"Name\" -> <||>|>|>}]; EntityRegister[store]; Length[EntityStores[]]"
      )
      .unwrap(),
      "1"
    );
  }

  #[test]
  fn entity_stores_empty_initially() {
    clear_state();
    assert_eq!(interpret("EntityStores[]").unwrap(), "{}");
  }

  #[test]
  fn entity_unregister_by_type() {
    clear_state();
    assert_eq!(
      interpret(
        "store = EntityStore[{\"Pet\" -> <|\"Entities\" -> <|\"cat1\" -> <|\"Name\" -> \"Mittens\"|>|>, \"Properties\" -> <|\"Name\" -> <||>|>|>}]; EntityRegister[store]; EntityUnregister[\"Pet\"]; EntityList[\"Pet\"]"
      )
      .unwrap(),
      "Missing[UnknownType, Pet]"
    );
  }

  #[test]
  fn entity_unregister_returns_null() {
    clear_state();
    assert_eq!(
      interpret(
        "store = EntityStore[{\"Pet\" -> <|\"Entities\" -> <||>, \"Properties\" -> <||>|>}]; EntityRegister[store]; EntityUnregister[\"Pet\"]"
      )
      .unwrap(),
      "\0"
    );
  }

  #[test]
  fn entity_multiple_types_in_store() {
    clear_state();
    assert_eq!(
      interpret(
        "store = EntityStore[{\"Dog\" -> <|\"Entities\" -> <|\"d1\" -> <|\"Name\" -> \"Rex\"|>|>, \"Properties\" -> <|\"Name\" -> <||>|>|>, \"Cat\" -> <|\"Entities\" -> <|\"c1\" -> <|\"Name\" -> \"Mittens\"|>|>, \"Properties\" -> <|\"Name\" -> <||>|>|>}]; EntityRegister[store]; {EntityValue[Entity[\"Dog\", \"d1\"], \"Name\"], EntityValue[Entity[\"Cat\", \"c1\"], \"Name\"]}"
      )
      .unwrap(),
      "{Rex, Mittens}"
    );
  }

  #[test]
  fn entity_value_entity_count() {
    clear_state();
    assert_eq!(
      interpret(
        "store = EntityStore[{\"Pet\" -> <|\"Entities\" -> <|\"cat1\" -> <|\"Name\" -> \"Mittens\"|>, \"dog1\" -> <|\"Name\" -> \"Rex\"|>|>, \"Properties\" -> <|\"Name\" -> <||>|>|>}]; EntityRegister[store]; EntityValue[\"Pet\", \"EntityCount\"]"
      )
      .unwrap(),
      "2"
    );
  }

  #[test]
  fn entity_value_entities_query() {
    clear_state();
    assert_eq!(
      interpret(
        "store = EntityStore[{\"Pet\" -> <|\"Entities\" -> <|\"cat1\" -> <|\"Name\" -> \"Mittens\"|>|>, \"Properties\" -> <|\"Name\" -> <||>|>|>}]; EntityRegister[store]; EntityValue[\"Pet\", \"Entities\"]"
      )
      .unwrap(),
      "{Entity[Pet, cat1]}"
    );
  }

  #[test]
  fn entity_output_form_no_quotes() {
    clear_state();
    assert_eq!(
      interpret("Entity[\"Pet\", \"cat1\"]").unwrap(),
      "Entity[Pet, cat1]"
    );
  }

  #[test]
  fn entity_property_access_unregistered() {
    clear_state();
    assert_eq!(
      interpret("Entity[\"Pet\", \"cat1\"][\"Name\"]").unwrap(),
      "Missing[UnknownType, Pet]"
    );
  }

  #[test]
  fn entity_store_single_rule() {
    clear_state();
    assert_eq!(
      interpret(
        "store = EntityStore[\"Pet\" -> <|\"Entities\" -> <|\"cat1\" -> <|\"Name\" -> \"Mittens\"|>|>|>]; EntityRegister[store]; Entity[\"Pet\", \"cat1\"][\"Name\"]"
      )
      .unwrap(),
      "Mittens"
    );
  }

  #[test]
  fn entity_store_callable() {
    clear_state();
    assert_eq!(
      interpret(
        "store = EntityStore[\"Pet\" -> <|\"Entities\" -> <|\"Sam\" -> <|\"Age\" -> 5|>|>|>]; store[Entity[\"Pet\", \"Sam\"], \"Age\"]"
      )
      .unwrap(),
      "5"
    );
  }

  #[test]
  fn entity_class_list() {
    clear_state();
    assert_eq!(
      interpret(
        "store = EntityStore[{\"Pet\" -> <|\"Entities\" -> <||>, \"EntityClasses\" -> <|\"Dogs\" -> <|\"Entities\" -> {\"Rex\"}|>, \"Cats\" -> <|\"Entities\" -> {\"Mittens\"}|>|>|>}]; EntityRegister[store]; EntityClassList[\"Pet\"]"
      )
      .unwrap(),
      "{EntityClass[Pet, Cats], EntityClass[Pet, Dogs]}"
    );
  }

  #[test]
  fn entity_list_by_class() {
    clear_state();
    assert_eq!(
      interpret(
        "store = EntityStore[{\"Pet\" -> <|\"Entities\" -> <|\"Rex\" -> <|\"Type\" -> \"Dog\"|>, \"Mittens\" -> <|\"Type\" -> \"Cat\"|>|>, \"EntityClasses\" -> <|\"Dogs\" -> <|\"Entities\" -> {\"Rex\"}|>|>|>}]; EntityRegister[store]; EntityList[EntityClass[\"Pet\", \"Dogs\"]]"
      )
      .unwrap(),
      "{Entity[Pet, Rex]}"
    );
  }

  #[test]
  fn entity_computed_property() {
    clear_state();
    assert_eq!(
      interpret(
        "store = EntityStore[{\"Pet\" -> <|\"Entities\" -> <|\"Rex\" -> <|\"Score\" -> 10|>|>, \"Properties\" -> <|\"DoubleScore\" -> <|\"DefaultFunction\" -> (2 * #[\"Score\"] &)|>|>|>}]; EntityRegister[store]; Entity[\"Pet\", \"Rex\"][\"DoubleScore\"]"
      )
      .unwrap(),
      "20"
    );
  }

  #[test]
  fn entity_property_mutation() {
    clear_state();
    assert_eq!(
      interpret(
        "store = EntityStore[{\"Pet\" -> <|\"Entities\" -> <|\"Rex\" -> <|\"Age\" -> 3|>|>|>}]; EntityRegister[store]; Entity[\"Pet\", \"Rex\"][\"Age\"] = 5; Entity[\"Pet\", \"Rex\"][\"Age\"]"
      )
      .unwrap(),
      "5"
    );
  }

  #[test]
  fn entity_property_mutation_adds_entity() {
    clear_state();
    assert_eq!(
      interpret(
        "store = EntityStore[{\"Pet\" -> <|\"Entities\" -> <|\"Rex\" -> <|\"Age\" -> 3|>|>|>}]; EntityRegister[store]; Entity[\"Pet\", \"Scout\"][\"Age\"] = 2; EntityList[\"Pet\"]"
      )
      .unwrap(),
      "{Entity[Pet, Rex], Entity[Pet, Scout]}"
    );
  }

  #[test]
  fn association_curried_call_lookup() {
    clear_state();
    assert_eq!(
      interpret("a = <|\"x\" -> 42, \"y\" -> 99|>; f = (#[\"x\"] &); f[a]")
        .unwrap(),
      "42"
    );
  }

  #[test]
  fn association_function_slot_access() {
    clear_state();
    assert_eq!(
      interpret("f = (#[\"x\"] &); f[<|\"x\" -> 7|>]").unwrap(),
      "7"
    );
  }
}

mod country_data_tests {
  use super::*;

  #[test]
  fn interpreter_country_resolves_to_entity() {
    clear_state();
    assert_eq!(
      interpret("Interpreter[\"Country\"][\"Germany\"]").unwrap(),
      "Entity[Country, Germany]"
    );
  }

  #[test]
  fn interpreter_country_resolves_aliases() {
    clear_state();
    assert_eq!(
      interpret("Interpreter[\"Country\"][\"USA\"]").unwrap(),
      "Entity[Country, United States]"
    );
    clear_state();
    assert_eq!(
      interpret("Interpreter[\"Country\"][\"Bosnia & Herzegovina\"]").unwrap(),
      "Entity[Country, Bosnia and Herzegovina]"
    );
  }

  #[test]
  fn interpreter_country_unknown_is_not_an_entity() {
    clear_state();
    // Non-sovereign names don't resolve, so Head is Missing (not Entity).
    assert_eq!(
      interpret("Head[Interpreter[\"Country\"][\"Scotland\"]]").unwrap(),
      "Missing"
    );
  }

  #[test]
  fn entity_value_population_is_a_quantity() {
    clear_state();
    assert_eq!(
      interpret(
        "Head[EntityValue[Entity[\"Country\", \"Germany\"], \"Population\"]]"
      )
      .unwrap(),
      "Quantity"
    );
  }

  #[test]
  fn population_in_millions_is_numeric() {
    clear_state();
    // The notebook's popMillions kernel: Quantity magnitude / 10^6.
    assert_eq!(
      interpret(
        "NumericQ[QuantityMagnitude[EntityValue[Interpreter[\"Country\"][\"Japan\"], \"Population\"]]/10.^6]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn country_data_schematic_coordinates() {
    clear_state();
    // A list of landmasses, each a list of {lat, lon} points.
    assert_eq!(
      interpret(
        "c = CountryData[\"Japan\", \"SchematicCoordinates\"]; \
         {Head[c], MatchQ[c[[1, 1]], {_Real, _Real}], \
          Depth[c], 24 < c[[1, 1, 1]] < 46}"
      )
      .unwrap(),
      "{List, True, 4, True}"
    );
    assert_eq!(
      interpret(
        "Map[Length, CountryData[\"Belgium\", \"SchematicCoordinates\"]] \
         === {Length[CountryData[\"Belgium\", \"Polygon\"][[1]]]}"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn country_data_world_and_antarctica_schematic_coordinates() {
    clear_state();
    // The world is every landmass; Antarctica lies entirely south of -60°.
    assert_eq!(
      interpret(
        "w = CountryData[\"World\", \"SchematicCoordinates\"]; \
         a = CountryData[\"Antarctica\", \"SchematicCoordinates\"]; \
         {Head[w], Length[w] > 100, MatchQ[w[[1, 1]], {_Real, _Real}], \
          Max[a[[All, All, 1]] /. l_List :> Max[l]] < -60}"
      )
      .unwrap(),
      "{List, True, True, True}"
    );
  }

  #[test]
  fn country_data_bare_projection_name() {
    clear_state();
    assert_eq!(
      interpret(
        "CountryData[\"Belgium\", {\"Shape\", \"Mercator\"}] === \
         CountryData[\"Belgium\", {\"Shape\", {\"Mercator\", {0, 0}}}]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "p = CountryData[\"Japan\", {\"Polygon\", \"Equirectangular\"}]; \
         {Head[p], Head[p[[1]]], MatchQ[p[[1, 1, 1]], {_Real, _Real}]}"
      )
      .unwrap(),
      "{Polygon, List, True}"
    );
  }

  #[test]
  fn country_data_shape_projections() {
    clear_state();
    // Every projection yields a Graphics of polygons.
    assert_eq!(
      interpret(
        "Union[Table[Head[CountryData[\"World\", {\"Shape\", {p, {0, 0}}}]], \
         {p, {\"Albers\", \"Bonne\", {\"CylindricalEqualArea\", 0}, \
         {\"CylindricalEquidistant\", 0}, \"EckertIV\", \"EckertVI\", \
         \"Equirectangular\", \"LambertAzimuthal\", \"LambertConic\", \
         \"LambertCylindrical\", \"Mercator\", \"MillerCylindrical\", \
         \"Mollweide\", \"Orthographic\", \"Polyconic\", \"Robinson\", \
         \"Sinusoidal\", \"VanDerGrinten\", \"WinkelTripel\"}}]]"
      )
      .unwrap(),
      "{Graphics}"
    );
  }

  #[test]
  fn country_data_shape_equirectangular_is_plain_lon_lat() {
    clear_state();
    // Centered on the origin, the projected extent is in radians of lon/lat.
    assert_eq!(
      interpret(
        "pts = Flatten[CountryData[\"Egypt\", {\"Shape\", {\"Equirectangular\", {0, 0}}}][[1, All, 1]], 1]; \
         {0.4 < Min[pts[[All, 1]]] < 0.5, 0.45 < Max[pts[[All, 1]]] < 0.65}"
      )
      .unwrap(),
      "{True, True}"
    );
  }

  #[test]
  fn country_data_shape_recenters_across_the_antimeridian() {
    clear_state();
    // Centered on the Pacific, no point leaves the map and nothing is dropped
    // for a full-world cylindrical projection.
    assert_eq!(
      interpret(
        "pts = Flatten[CountryData[\"World\", {\"Shape\", {\"Equirectangular\", {0, 180}}}][[1, All, 1]], 1]; \
         {Max[Abs[pts[[All, 1]]]] <= Pi, Min[pts[[All, 1]]] < -3.1, Max[pts[[All, 1]]] > 3.1}"
      )
      .unwrap(),
      "{True, True, True}"
    );
  }

  #[test]
  fn country_data_shape_orthographic_hides_far_side() {
    clear_state();
    assert_eq!(
      interpret(
        "pts = Flatten[CountryData[\"World\", {\"Shape\", {\"Orthographic\", {0, 0}}}][[1, All, 1]], 1]; \
         Max[Norm /@ pts] <= 1.0000001"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn country_data_population() {
    clear_state();
    assert_eq!(
      interpret("CountryData[\"Qatar\", \"Population\"]").unwrap(),
      "Quantity[2695122, People]"
    );
  }

  #[test]
  fn country_data_language_speaking() {
    clear_state();
    assert_eq!(
      interpret("CountryData[\"France\", \"FrenchSpeaking\"]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("CountryData[\"France\", \"GermanSpeaking\"]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret(
        "CountryData[Entity[\"Country\", \"Brazil\"], \"PortugueseSpeaking\"]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn country_data_list_and_schematic_polygon() {
    clear_state();
    assert_eq!(
      interpret("MemberQ[CountryData[], Entity[\"Country\", \"Japan\"]]")
        .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "Head[CountryData[\"Japan\", \"SchematicPolygon\"]] === List || Head[CountryData[\"Japan\", \"SchematicPolygon\"]] === Polygon"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "StringContainsQ[ExportString[Graphics[{Orange, CountryData[\"Italy\", \"SchematicPolygon\"]}], \"SVG\"], \"<polygon\"]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn number_interpreter_threads_over_a_list() {
    assert_eq!(
      interpret("Interpreter[\"Number\"][{\"12.5\", \"3\"}]").unwrap(),
      "{12.5, 3}"
    );
    assert_eq!(interpret("Interpreter[\"Number\"][{}]").unwrap(), "{}");
  }

  #[test]
  fn country_data_classes_full_name_and_polygon() {
    clear_state();
    assert_eq!(interpret("Length[CountryData[\"EU\"]]").unwrap(), "27");
    assert_eq!(
      interpret("MemberQ[CountryData[\"EU\"], \"CzechRepublic\"]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "Complement[CountryData[\"EU\"], CountryData[\"Europe\"]] === {}"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("CountryData[\"UnitedKingdom\", \"Population\"] === CountryData[\"United Kingdom\", \"Population\"]")
        .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("CountryData[\"Italy\", \"FullName\"]").unwrap(),
      "Italy"
    );
    assert_eq!(
      interpret("CountryData[\"CzechRepublic\", \"FullName\"]").unwrap(),
      "CzechRepublic"
    );
    assert_eq!(
      interpret(
        "CountryData[\"Italy\", \"Polygon\"] === CountryData[\"Italy\", \"SchematicPolygon\"]"
      )
      .unwrap(),
      "True"
    );
  }
}

mod interpreter_scalar_tests {
  use super::*;

  #[test]
  fn integer_from_string() {
    assert_eq!(interpret(r#"Interpreter["Integer"]["42"]"#).unwrap(), "42");
    assert_eq!(interpret(r#"Interpreter["Integer"]["-7"]"#).unwrap(), "-7");
  }

  #[test]
  fn integer_handles_bignum() {
    assert_eq!(
      interpret(r#"Interpreter["Integer"]["99999999999999999999999999"]"#)
        .unwrap(),
      "99999999999999999999999999"
    );
  }

  #[test]
  fn number_keeps_integers_and_reals() {
    assert_eq!(interpret(r#"Interpreter["Number"]["42"]"#).unwrap(), "42");
    assert_eq!(
      interpret(r#"Interpreter["Number"]["3.14"]"#).unwrap(),
      "3.14"
    );
    // Scientific notation and signs.
    assert_eq!(
      interpret(r#"Interpreter["Number"]["1e3"]"#).unwrap(),
      "1000."
    );
    assert_eq!(
      interpret(r#"Interpreter["Number"]["-3.5"]"#).unwrap(),
      "-3.5"
    );
  }

  #[test]
  fn real_always_returns_a_real() {
    assert_eq!(interpret(r#"Interpreter["Real"]["3.14"]"#).unwrap(), "3.14");
    assert_eq!(interpret(r#"Interpreter["Real"]["42"]"#).unwrap(), "42.");
  }

  #[test]
  fn boolean_parses_case_insensitively() {
    assert_eq!(
      interpret(r#"Interpreter["Boolean"]["true"]"#).unwrap(),
      "True"
    );
    assert_eq!(
      interpret(r#"Interpreter["Boolean"]["False"]"#).unwrap(),
      "False"
    );
  }

  #[test]
  fn string_returns_input() {
    assert_eq!(
      interpret(r#"Interpreter["String"]["hello"]"#).unwrap(),
      "hello"
    );
  }

  #[test]
  fn integer_passthrough_for_numeric_input() {
    assert_eq!(interpret(r#"Interpreter["Integer"][42]"#).unwrap(), "42");
  }
}
