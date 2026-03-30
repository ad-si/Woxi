//! AST-native entity store functions.
//!
//! Implements EntityStore, EntityRegister, EntityUnregister, EntityValue,
//! EntityList, EntityProperties, EntityStores, EntityClassList, and Entity property access.

use crate::InterpreterError;
use crate::syntax::{Expr, expr_to_string};

use std::cell::RefCell;

/// Registered entity stores: each store maps type_name -> EntityTypeData
#[derive(Clone, Debug)]
pub struct EntityTypeData {
  /// entity_name -> { property_name -> value }
  pub entities: Vec<(String, Vec<(String, Expr)>)>,
  /// property_name -> property metadata (currently just marker)
  pub properties: Vec<String>,
  /// entity_class_name -> list of entity names
  pub entity_classes: Vec<(String, Vec<String>)>,
  /// property_name -> DefaultFunction (an Expr, typically a Function/anonymous fn)
  pub computed_properties: Vec<(String, Expr)>,
}

thread_local! {
    /// Stack of registered entity stores.
    /// Each store is a Vec of (type_name, EntityTypeData).
    pub static ENTITY_STORES: RefCell<Vec<Vec<(String, EntityTypeData)>>> = const { RefCell::new(Vec::new()) };
}

/// Parse an EntityStore expression and return the internal representation.
/// Valid forms:
///   EntityStore[{type1 -> <|...|>, type2 -> <|...|>, ...}]
///   EntityStore["type" -> <|...|>]  (single type shorthand)
/// Returns the normalized EntityStore expression.
pub fn entity_store_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "EntityStore".to_string(),
      args: args.to_vec(),
    });
  }

  // Parse the list of type -> data rules
  // Accept either a List of rules or a single Rule
  let type_rules: Vec<&Expr> = match &args[0] {
    Expr::List(items) => items.iter().collect(),
    // Single rule: "Type" -> <|...|>
    Expr::Rule { .. } | Expr::RuleDelayed { .. } => vec![&args[0]],
    Expr::FunctionCall { name, args: fargs }
      if (name == "Rule" || name == "RuleDelayed") && fargs.len() == 2 =>
    {
      vec![&args[0]]
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "EntityStore".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let mut types_assoc: Vec<(Expr, Expr)> = Vec::new();

  for rule in &type_rules {
    let (type_name, type_data) = match extract_rule(rule) {
      Some(pair) => pair,
      None => {
        return Ok(Expr::FunctionCall {
          name: "EntityStore".to_string(),
          args: args.to_vec(),
        });
      }
    };

    let type_name_str = match &type_name {
      Expr::String(s) => s.clone(),
      _ => expr_to_string(&type_name),
    };

    // type_data should be an Association with "Entities" and "Properties" keys
    match &type_data {
      Expr::Association(_) => {}
      _ => {
        return Ok(Expr::FunctionCall {
          name: "EntityStore".to_string(),
          args: args.to_vec(),
        });
      }
    }

    types_assoc.push((Expr::String(type_name_str), type_data));
  }

  // Build the normalized form: EntityStore[<|Types -> <|...|>|>]
  let types_inner = Expr::Association(types_assoc);
  let outer =
    Expr::Association(vec![(Expr::String("Types".to_string()), types_inner)]);

  Ok(Expr::FunctionCall {
    name: "EntityStore".to_string(),
    args: vec![outer],
  })
}

/// Parse entity type data from an association.
/// Expected keys: "Entities" -> <|name -> <|prop -> val, ...|>, ...|>,
///                "Properties" -> <|propName -> <|"DefaultFunction" -> fn|>, ...|>
///                "EntityClasses" -> <|className -> <|"Entities" -> {name1, name2}|>, ...|>
fn parse_entity_type_data(data: &Expr) -> Option<EntityTypeData> {
  let pairs = match data {
    Expr::Association(pairs) => pairs,
    _ => return None,
  };

  let mut entities: Vec<(String, Vec<(String, Expr)>)> = Vec::new();
  let mut properties: Vec<String> = Vec::new();
  let mut entity_classes: Vec<(String, Vec<String>)> = Vec::new();
  let mut computed_properties: Vec<(String, Expr)> = Vec::new();

  for (key, value) in pairs {
    let key_str = match key {
      Expr::String(s) => s.clone(),
      _ => expr_to_string(key),
    };

    match key_str.as_str() {
      "Entities" => {
        if let Expr::Association(entity_pairs) = value {
          for (ename, edata) in entity_pairs {
            let entity_name = match ename {
              Expr::String(s) => s.clone(),
              _ => expr_to_string(ename),
            };
            let mut props = Vec::new();
            if let Expr::Association(prop_pairs) = edata {
              for (pname, pval) in prop_pairs {
                let prop_name = match pname {
                  Expr::String(s) => s.clone(),
                  _ => expr_to_string(pname),
                };
                props.push((prop_name, pval.clone()));
              }
            }
            entities.push((entity_name, props));
          }
        }
      }
      "Properties" => {
        if let Expr::Association(prop_pairs) = value {
          for (pname, pval) in prop_pairs {
            let prop_name = match pname {
              Expr::String(s) => s.clone(),
              _ => expr_to_string(pname),
            };
            properties.push(prop_name.clone());
            // Check for DefaultFunction in the property metadata
            if let Expr::Association(meta_pairs) = pval {
              for (mk, mv) in meta_pairs {
                let mk_str = match mk {
                  Expr::String(s) => s.clone(),
                  _ => expr_to_string(mk),
                };
                if mk_str == "DefaultFunction" {
                  computed_properties.push((prop_name.clone(), mv.clone()));
                }
              }
            }
          }
        }
      }
      "EntityClasses" => {
        if let Expr::Association(class_pairs) = value {
          for (cname, cdata) in class_pairs {
            let class_name = match cname {
              Expr::String(s) => s.clone(),
              _ => expr_to_string(cname),
            };
            let mut members = Vec::new();
            // cdata should be <|"Entities" -> {"name1", "name2"}|>
            if let Expr::Association(class_meta) = cdata {
              for (cmk, cmv) in class_meta {
                let cmk_str = match cmk {
                  Expr::String(s) => s.clone(),
                  _ => expr_to_string(cmk),
                };
                if cmk_str == "Entities" {
                  if let Expr::List(items) = cmv {
                    for item in items {
                      let name = match item {
                        Expr::String(s) => s.clone(),
                        _ => expr_to_string(item),
                      };
                      members.push(name);
                    }
                  }
                }
              }
            }
            entity_classes.push((class_name, members));
          }
        }
      }
      _ => {
        // Ignore other keys for now (Label, LabelPlural, etc.)
      }
    }
  }

  Some(EntityTypeData {
    entities,
    properties,
    entity_classes,
    computed_properties,
  })
}

/// EntityRegister[store] — register an entity store.
/// Returns a list of type names that were registered.
pub fn entity_register_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "EntityRegister".to_string(),
      args: args.to_vec(),
    });
  }

  let store = &args[0];

  // Must be EntityStore[<|Types -> <|...|>|>]
  let types_data = match store {
    Expr::FunctionCall { name, args }
      if name == "EntityStore" && args.len() == 1 =>
    {
      match &args[0] {
        Expr::Association(outer_pairs) => {
          // Find the "Types" key
          let mut types_assoc = None;
          for (k, v) in outer_pairs {
            let key_str = match k {
              Expr::String(s) => s.clone(),
              _ => expr_to_string(k),
            };
            if key_str == "Types" {
              types_assoc = Some(v);
              break;
            }
          }
          match types_assoc {
            Some(Expr::Association(type_pairs)) => type_pairs.clone(),
            _ => {
              return Err(InterpreterError::EvaluationError(
                "EntityRegister::invstore: not a valid entity store."
                  .to_string(),
              ));
            }
          }
        }
        _ => {
          return Err(InterpreterError::EvaluationError(
            "EntityRegister::invstore: not a valid entity store.".to_string(),
          ));
        }
      }
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "EntityRegister::invstore: not a valid entity store.".to_string(),
      ));
    }
  };

  let mut store_types: Vec<(String, EntityTypeData)> = Vec::new();
  let mut type_names: Vec<Expr> = Vec::new();

  for (type_key, type_val) in &types_data {
    let type_name = match type_key {
      Expr::String(s) => s.clone(),
      _ => expr_to_string(type_key),
    };

    if let Some(data) = parse_entity_type_data(type_val) {
      type_names.push(Expr::String(type_name.clone()));
      store_types.push((type_name, data));
    }
  }

  ENTITY_STORES.with(|stores| {
    stores.borrow_mut().push(store_types);
  });

  Ok(Expr::List(type_names))
}

/// EntityUnregister["type"] — unregister the first store containing that type.
pub fn entity_unregister_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "EntityUnregister".to_string(),
      args: args.to_vec(),
    });
  }

  match &args[0] {
    Expr::String(type_name) => {
      ENTITY_STORES.with(|stores| {
        let mut stores = stores.borrow_mut();
        // Find the first store that contains this type and remove it
        if let Some(idx) = stores
          .iter()
          .position(|store| store.iter().any(|(name, _)| name == type_name))
        {
          stores.remove(idx);
        }
      });
      Ok(Expr::Identifier("Null".to_string()))
    }
    _ => Ok(Expr::FunctionCall {
      name: "EntityUnregister".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// EntityStores[] — return a list of all registered entity stores.
pub fn entity_stores_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if !args.is_empty() {
    return Ok(Expr::FunctionCall {
      name: "EntityStores".to_string(),
      args: args.to_vec(),
    });
  }

  let stores = ENTITY_STORES.with(|stores| {
    let stores = stores.borrow();
    stores
      .iter()
      .map(|store| {
        // Reconstruct EntityStore[<|Types -> <|...|>|>]
        let type_pairs: Vec<(Expr, Expr)> = store
          .iter()
          .map(|(type_name, data)| {
            let entities_assoc = Expr::Association(
              data
                .entities
                .iter()
                .map(|(ename, props)| {
                  let prop_assoc = Expr::Association(
                    props
                      .iter()
                      .map(|(pname, pval)| {
                        (Expr::String(pname.clone()), pval.clone())
                      })
                      .collect(),
                  );
                  (Expr::String(ename.clone()), prop_assoc)
                })
                .collect(),
            );
            let properties_assoc = Expr::Association(
              data
                .properties
                .iter()
                .map(|pname| {
                  (Expr::String(pname.clone()), Expr::Association(vec![]))
                })
                .collect(),
            );
            let type_data = Expr::Association(vec![
              (Expr::String("Entities".to_string()), entities_assoc),
              (Expr::String("Properties".to_string()), properties_assoc),
            ]);
            (Expr::String(type_name.clone()), type_data)
          })
          .collect();
        let types_inner = Expr::Association(type_pairs);
        let outer = Expr::Association(vec![(
          Expr::String("Types".to_string()),
          types_inner,
        )]);
        Expr::FunctionCall {
          name: "EntityStore".to_string(),
          args: vec![outer],
        }
      })
      .collect::<Vec<_>>()
  });

  Ok(Expr::List(stores))
}

/// Look up an entity's property value from registered stores.
fn lookup_entity_property(
  type_name: &str,
  entity_name: &str,
  property: &str,
) -> Option<Expr> {
  ENTITY_STORES.with(|stores| {
    let stores = stores.borrow();
    for store in stores.iter() {
      for (tname, data) in store {
        if tname == type_name {
          for (ename, props) in &data.entities {
            if ename == entity_name {
              // First check explicit properties
              for (pname, pval) in props {
                if pname == property {
                  return Some(pval.clone());
                }
              }
              // Then check computed properties (DefaultFunction)
              for (cpname, func) in &data.computed_properties {
                if cpname == property {
                  // Build an association of entity properties for the function argument
                  let entity_assoc = Expr::Association(
                    props
                      .iter()
                      .map(|(k, v)| (Expr::String(k.clone()), v.clone()))
                      .collect(),
                  );
                  // Apply the function: func[entity_assoc]
                  // We return a marker so the caller can evaluate it
                  return Some(Expr::FunctionCall {
                    name: "__entity_computed__".to_string(),
                    args: vec![func.clone(), entity_assoc],
                  });
                }
              }
              // Entity found but property missing
              return Some(Expr::FunctionCall {
                name: "Missing".to_string(),
                args: vec![Expr::String("NotAvailable".to_string())],
              });
            }
          }
        }
      }
    }
    None
  })
}

/// Check if a type is registered
fn is_type_registered(type_name: &str) -> bool {
  ENTITY_STORES.with(|stores| {
    let stores = stores.borrow();
    stores
      .iter()
      .any(|store| store.iter().any(|(name, _)| name == type_name))
  })
}

/// EntityValue[entity, property] and related forms.
pub fn entity_value_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  match args.len() {
    2 => {
      let entity = &args[0];
      let property = &args[1];

      // EntityValue["type", "Properties"] — list properties for a type
      if let Expr::String(type_name) = entity {
        if let Expr::String(prop_query) = property {
          if prop_query == "Properties" {
            return entity_properties_for_type(type_name);
          }
          if prop_query == "Entities" {
            return entity_list_for_type(type_name);
          }
          if prop_query == "EntityCount" {
            return entity_count_for_type(type_name);
          }
        }
      }

      // EntityValue[Entity["type", "name"], "property"]
      if let Expr::FunctionCall {
        name,
        args: entity_args,
      } = entity
      {
        if name == "Entity" && entity_args.len() == 2 {
          let type_name = match &entity_args[0] {
            Expr::String(s) => s.clone(),
            _ => expr_to_string(&entity_args[0]),
          };
          let entity_name = match &entity_args[1] {
            Expr::String(s) => s.clone(),
            _ => expr_to_string(&entity_args[1]),
          };

          // Single property
          if let Expr::String(prop_name) = property {
            return resolve_entity_lookup(&type_name, &entity_name, prop_name);
          }

          // Multiple properties: EntityValue[entity, {"prop1", "prop2"}]
          if let Expr::List(prop_list) = property {
            let mut results = Vec::new();
            for p in prop_list {
              let prop_name = match p {
                Expr::String(s) => s.clone(),
                Expr::FunctionCall { name, args }
                  if name == "EntityProperty" && args.len() == 2 =>
                {
                  match &args[1] {
                    Expr::String(s) => s.clone(),
                    _ => expr_to_string(&args[1]),
                  }
                }
                _ => expr_to_string(p),
              };
              let val =
                resolve_entity_lookup(&type_name, &entity_name, &prop_name)?;
              results.push(val);
            }
            return Ok(Expr::List(results));
          }

          // EntityProperty form
          if let Expr::FunctionCall {
            name: pname,
            args: pargs,
          } = property
          {
            if pname == "EntityProperty" && pargs.len() == 2 {
              let prop_name = match &pargs[1] {
                Expr::String(s) => s.clone(),
                _ => expr_to_string(&pargs[1]),
              };
              return resolve_entity_lookup(
                &type_name,
                &entity_name,
                &prop_name,
              );
            }
          }
        }
      }

      // EntityValue[{entity1, entity2, ...}, property] — multiple entities
      if let Expr::List(entities) = entity {
        if let Expr::String(prop_name) = property {
          let mut results = Vec::new();
          for e in entities {
            if let Expr::FunctionCall {
              name,
              args: entity_args,
            } = e
            {
              if name == "Entity" && entity_args.len() == 2 {
                let type_name = match &entity_args[0] {
                  Expr::String(s) => s.clone(),
                  _ => expr_to_string(&entity_args[0]),
                };
                let entity_name = match &entity_args[1] {
                  Expr::String(s) => s.clone(),
                  _ => expr_to_string(&entity_args[1]),
                };
                let val =
                  resolve_entity_lookup(&type_name, &entity_name, prop_name)?;
                results.push(val);
              }
            }
          }
          return Ok(Expr::List(results));
        }
      }

      // Fallback: return unevaluated
      Ok(Expr::FunctionCall {
        name: "EntityValue".to_string(),
        args: args.to_vec(),
      })
    }
    _ => Ok(Expr::FunctionCall {
      name: "EntityValue".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Resolve an entity property lookup, evaluating computed properties if needed.
fn resolve_entity_lookup(
  type_name: &str,
  entity_name: &str,
  property: &str,
) -> Result<Expr, InterpreterError> {
  match lookup_entity_property(type_name, entity_name, property) {
    Some(
      ref val @ Expr::FunctionCall {
        ref name,
        args: ref comp_args,
      },
    ) if name == "__entity_computed__" && comp_args.len() == 2 => {
      let _ = val;
      // Evaluate the computed property: apply func to entity association
      let func_expr = Expr::CurriedCall {
        func: Box::new(comp_args[0].clone()),
        args: vec![comp_args[1].clone()],
      };
      crate::evaluator::evaluate_expr_to_expr(&func_expr)
    }
    Some(val) => Ok(val),
    None => Ok(Expr::FunctionCall {
      name: "Missing".to_string(),
      args: vec![
        Expr::String("UnknownType".to_string()),
        Expr::String(type_name.to_string()),
      ],
    }),
  }
}

/// EntityList["type"] — list all entities of a given type.
/// EntityList[Entity["type"]] — same.
/// EntityList[EntityClass["type", "class"]] — list entities in a class.
pub fn entity_list_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "EntityList".to_string(),
      args: args.to_vec(),
    });
  }

  match &args[0] {
    Expr::String(type_name) => entity_list_for_type(type_name),
    Expr::FunctionCall { name, args: fargs }
      if name == "Entity" && fargs.len() == 1 =>
    {
      // EntityList[Entity["type"]] form
      let type_name = match &fargs[0] {
        Expr::String(s) => s.clone(),
        _ => expr_to_string(&fargs[0]),
      };
      entity_list_for_type(&type_name)
    }
    Expr::FunctionCall { name, args: fargs }
      if name == "EntityClass" && fargs.len() == 2 =>
    {
      // EntityList[EntityClass["type", "class"]] form
      let type_name = match &fargs[0] {
        Expr::String(s) => s.clone(),
        _ => expr_to_string(&fargs[0]),
      };
      let class_name = match &fargs[1] {
        Expr::String(s) => s.clone(),
        _ => expr_to_string(&fargs[1]),
      };
      entity_list_for_class(&type_name, &class_name)
    }
    _ => Ok(Expr::FunctionCall {
      name: "EntityList".to_string(),
      args: args.to_vec(),
    }),
  }
}

fn entity_list_for_type(type_name: &str) -> Result<Expr, InterpreterError> {
  if !is_type_registered(type_name) {
    return Ok(Expr::FunctionCall {
      name: "Missing".to_string(),
      args: vec![
        Expr::String("UnknownType".to_string()),
        Expr::String(type_name.to_string()),
      ],
    });
  }

  let entities = ENTITY_STORES.with(|stores| {
    let stores = stores.borrow();
    let mut result = Vec::new();
    for store in stores.iter() {
      for (tname, data) in store {
        if tname == type_name {
          for (ename, _) in &data.entities {
            result.push(Expr::FunctionCall {
              name: "Entity".to_string(),
              args: vec![
                Expr::String(type_name.to_string()),
                Expr::String(ename.clone()),
              ],
            });
          }
        }
      }
    }
    result
  });

  Ok(Expr::List(entities))
}

fn entity_list_for_class(
  type_name: &str,
  class_name: &str,
) -> Result<Expr, InterpreterError> {
  if !is_type_registered(type_name) {
    return Ok(Expr::FunctionCall {
      name: "Missing".to_string(),
      args: vec![
        Expr::String("UnknownType".to_string()),
        Expr::String(type_name.to_string()),
      ],
    });
  }

  let entities = ENTITY_STORES.with(|stores| {
    let stores = stores.borrow();
    let mut result = Vec::new();
    for store in stores.iter() {
      for (tname, data) in store {
        if tname == type_name {
          for (cname, members) in &data.entity_classes {
            if cname == class_name {
              for member in members {
                result.push(Expr::FunctionCall {
                  name: "Entity".to_string(),
                  args: vec![
                    Expr::String(type_name.to_string()),
                    Expr::String(member.clone()),
                  ],
                });
              }
            }
          }
        }
      }
    }
    result
  });

  Ok(Expr::List(entities))
}

fn entity_count_for_type(type_name: &str) -> Result<Expr, InterpreterError> {
  if !is_type_registered(type_name) {
    return Ok(Expr::FunctionCall {
      name: "Missing".to_string(),
      args: vec![
        Expr::String("UnknownType".to_string()),
        Expr::String(type_name.to_string()),
      ],
    });
  }

  let count = ENTITY_STORES.with(|stores| {
    let stores = stores.borrow();
    let mut count = 0i128;
    for store in stores.iter() {
      for (tname, data) in store {
        if tname == type_name {
          count += data.entities.len() as i128;
        }
      }
    }
    count
  });

  Ok(Expr::Integer(count))
}

/// EntityClassList["type"] — list all entity classes for a given type.
pub fn entity_class_list_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "EntityClassList".to_string(),
      args: args.to_vec(),
    });
  }

  let type_name = match &args[0] {
    Expr::String(s) => s.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "EntityClassList".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if !is_type_registered(&type_name) {
    return Ok(Expr::FunctionCall {
      name: "Missing".to_string(),
      args: vec![
        Expr::String("UnknownType".to_string()),
        Expr::String(type_name),
      ],
    });
  }

  let classes = ENTITY_STORES.with(|stores| {
    let stores = stores.borrow();
    let mut class_names = Vec::new();
    for store in stores.iter() {
      for (tname, data) in store {
        if tname == &type_name {
          for (cname, _) in &data.entity_classes {
            class_names.push(cname.clone());
          }
        }
      }
    }
    class_names.sort();
    class_names
      .into_iter()
      .map(|cname| Expr::FunctionCall {
        name: "EntityClass".to_string(),
        args: vec![Expr::String(type_name.clone()), Expr::String(cname)],
      })
      .collect::<Vec<_>>()
  });

  Ok(Expr::List(classes))
}

/// EntityProperties["type"] — list all properties for a given type.
pub fn entity_properties_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "EntityProperties".to_string(),
      args: args.to_vec(),
    });
  }

  match &args[0] {
    Expr::String(type_name) => entity_properties_for_type(type_name),
    _ => Ok(Expr::FunctionCall {
      name: "EntityProperties".to_string(),
      args: args.to_vec(),
    }),
  }
}

fn entity_properties_for_type(
  type_name: &str,
) -> Result<Expr, InterpreterError> {
  if !is_type_registered(type_name) {
    return Ok(Expr::FunctionCall {
      name: "Missing".to_string(),
      args: vec![
        Expr::String("UnknownType".to_string()),
        Expr::String(type_name.to_string()),
      ],
    });
  }

  let properties = ENTITY_STORES.with(|stores| {
    let stores = stores.borrow();
    let mut seen = std::collections::BTreeSet::new();
    for store in stores.iter() {
      for (tname, data) in store {
        if tname == type_name {
          for prop_name in &data.properties {
            seen.insert(prop_name.clone());
          }
        }
      }
    }
    seen
      .into_iter()
      .map(|prop_name| Expr::FunctionCall {
        name: "EntityProperty".to_string(),
        args: vec![
          Expr::String(type_name.to_string()),
          Expr::String(prop_name),
        ],
      })
      .collect::<Vec<_>>()
  });

  Ok(Expr::List(properties))
}

/// Handle Entity["type", "name"]["property"] — property access via curried call.
pub fn entity_property_access(
  entity_args: &[Expr],
  access_args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if entity_args.len() != 2 || access_args.len() != 1 {
    return Ok(Expr::CurriedCall {
      func: Box::new(Expr::FunctionCall {
        name: "Entity".to_string(),
        args: entity_args.to_vec(),
      }),
      args: access_args.to_vec(),
    });
  }

  let type_name = match &entity_args[0] {
    Expr::String(s) => s.clone(),
    _ => expr_to_string(&entity_args[0]),
  };
  let entity_name = match &entity_args[1] {
    Expr::String(s) => s.clone(),
    _ => expr_to_string(&entity_args[1]),
  };

  let property = match &access_args[0] {
    Expr::String(s) => s.clone(),
    _ => {
      return Ok(Expr::CurriedCall {
        func: Box::new(Expr::FunctionCall {
          name: "Entity".to_string(),
          args: entity_args.to_vec(),
        }),
        args: access_args.to_vec(),
      });
    }
  };

  resolve_entity_lookup(&type_name, &entity_name, &property)
}

/// Handle EntityStore[...][Entity["type", "name"], "property"] — callable store form.
pub fn entity_store_property_access(
  store_args: &[Expr],
  call_args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if call_args.len() != 2 {
    return Ok(Expr::CurriedCall {
      func: Box::new(Expr::FunctionCall {
        name: "EntityStore".to_string(),
        args: store_args.to_vec(),
      }),
      args: call_args.to_vec(),
    });
  }

  // Extract Entity["type", "name"] from first arg
  let (type_name, entity_name) = match &call_args[0] {
    Expr::FunctionCall {
      name,
      args: entity_args,
    } if name == "Entity" && entity_args.len() == 2 => {
      let tn = match &entity_args[0] {
        Expr::String(s) => s.clone(),
        _ => expr_to_string(&entity_args[0]),
      };
      let en = match &entity_args[1] {
        Expr::String(s) => s.clone(),
        _ => expr_to_string(&entity_args[1]),
      };
      (tn, en)
    }
    _ => {
      return Ok(Expr::CurriedCall {
        func: Box::new(Expr::FunctionCall {
          name: "EntityStore".to_string(),
          args: store_args.to_vec(),
        }),
        args: call_args.to_vec(),
      });
    }
  };

  let property = match &call_args[1] {
    Expr::String(s) => s.clone(),
    _ => {
      return Ok(Expr::CurriedCall {
        func: Box::new(Expr::FunctionCall {
          name: "EntityStore".to_string(),
          args: store_args.to_vec(),
        }),
        args: call_args.to_vec(),
      });
    }
  };

  // Look up the property directly from the store's normalized form
  // The store is EntityStore[<|Types -> <|type -> <|Entities -> ...|>|>|>]
  if store_args.len() == 1 {
    if let Expr::Association(outer) = &store_args[0] {
      for (k, v) in outer {
        let ks = match k {
          Expr::String(s) => s.clone(),
          _ => expr_to_string(k),
        };
        if ks == "Types" {
          if let Expr::Association(types) = v {
            for (tk, tv) in types {
              let tks = match tk {
                Expr::String(s) => s.clone(),
                _ => expr_to_string(tk),
              };
              if tks == type_name {
                if let Some(data) = parse_entity_type_data(tv) {
                  for (ename, props) in &data.entities {
                    if ename == &entity_name {
                      for (pname, pval) in props {
                        if pname == &property {
                          return Ok(pval.clone());
                        }
                      }
                      return Ok(Expr::FunctionCall {
                        name: "Missing".to_string(),
                        args: vec![Expr::String("NotAvailable".to_string())],
                      });
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  Ok(Expr::FunctionCall {
    name: "Missing".to_string(),
    args: vec![
      Expr::String("UnknownType".to_string()),
      Expr::String(type_name),
    ],
  })
}

/// Handle Entity["type", "name"]["property"] = value — entity property mutation.
/// Sets/updates the property in the registered store.
pub fn entity_property_set(
  entity_args: &[Expr],
  access_args: &[Expr],
  value: &Expr,
) -> Result<Expr, InterpreterError> {
  if entity_args.len() != 2 || access_args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Entity property assignment requires Entity[type, name][property] = value"
        .to_string(),
    ));
  }

  let type_name = match &entity_args[0] {
    Expr::String(s) => s.clone(),
    _ => expr_to_string(&entity_args[0]),
  };
  let entity_name = match &entity_args[1] {
    Expr::String(s) => s.clone(),
    _ => expr_to_string(&entity_args[1]),
  };
  let property = match &access_args[0] {
    Expr::String(s) => s.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Entity property name must be a string".to_string(),
      ));
    }
  };

  let rhs_value = crate::evaluator::evaluate_expr_to_expr(value)?;

  ENTITY_STORES.with(|stores| {
    let mut stores = stores.borrow_mut();
    for store in stores.iter_mut() {
      for (tname, data) in store.iter_mut() {
        if tname == &type_name {
          // Try to find the entity
          for (ename, props) in &mut data.entities {
            if ename == &entity_name {
              // Update or add the property
              if let Some(pair) =
                props.iter_mut().find(|(pn, _)| pn == &property)
              {
                pair.1 = rhs_value.clone();
              } else {
                props.push((property.clone(), rhs_value.clone()));
              }
              return;
            }
          }
          // Entity not found in this type — add it
          data.entities.push((
            entity_name.clone(),
            vec![(property.clone(), rhs_value.clone())],
          ));
          return;
        }
      }
    }
  });

  Ok(rhs_value)
}

/// Extract key-value from a Rule expression
fn extract_rule(expr: &Expr) -> Option<(Expr, Expr)> {
  match expr {
    Expr::Rule {
      pattern,
      replacement,
    } => Some((*pattern.clone(), *replacement.clone())),
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => Some((*pattern.clone(), *replacement.clone())),
    Expr::FunctionCall { name, args }
      if (name == "Rule" || name == "RuleDelayed") && args.len() == 2 =>
    {
      Some((args[0].clone(), args[1].clone()))
    }
    _ => None,
  }
}

/// Clear all registered entity stores (used in test cleanup).
pub fn clear_entity_stores() {
  ENTITY_STORES.with(|stores| {
    stores.borrow_mut().clear();
  });
}
