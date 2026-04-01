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
