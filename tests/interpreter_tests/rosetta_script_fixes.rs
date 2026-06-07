use super::*;

// Regression tests for engine fixes that unlocked several RosettaCode scripts
// (five_weekends, executable_library, bernoulli_numbers, append_a_record_...).
mod rosetta_script_fixes {
  use super::*;

  // `Join[lists…, n]` takes an optional trailing level specification. The
  // common `1` (e.g. `{y, m} ~Join~ 1`) is an ordinary join of the lists.
  mod join_level_spec {
    use super::*;

    #[test]
    fn trailing_one_is_level_spec() {
      assert_eq!(interpret("{1901, 3} ~Join~ 1").unwrap(), "{1901, 3}");
    }

    #[test]
    fn trailing_one_with_multiple_lists() {
      assert_eq!(interpret("Join[{1, 2}, {3}, 1]").unwrap(), "{1, 2, 3}");
    }

    #[test]
    fn ordinary_join_unaffected() {
      assert_eq!(interpret("Join[{1, 2}, {3, 4}]").unwrap(), "{1, 2, 3, 4}");
    }
  }

  // Composition `@*` (and `/*`) binds tighter than Map `/@`, so
  // `Length@*f /@ list` parses as `(Length@*f) /@ list`.
  mod composition_precedence {
    use super::*;

    #[test]
    fn composition_binds_tighter_than_map() {
      assert_eq!(
        interpret("f[x_] := {x, x}; Length@*f /@ {1, 2, 3}").unwrap(),
        "{2, 2, 2}"
      );
    }

    #[test]
    fn composition_still_applies_directly() {
      assert_eq!(
        interpret("g = Length@*Union; g[{1, 1, 2, 2, 3}]").unwrap(),
        "3"
      );
    }
  }

  // Times must combine rationals exactly even when numerator/denominator
  // exceed i128 (previously left an unreduced Times, which also panicked the
  // formatter via denominator_form on a non-Power factor).
  mod big_rational_times {
    use super::*;

    #[test]
    fn product_of_big_rationals_reduces() {
      assert_eq!(
        interpret(
          "(123456789012345678901234567890/7) * \
           (7/123456789012345678901234567890)"
        )
        .unwrap(),
        "1"
      );
    }

    #[test]
    fn product_of_big_rationals_no_panic() {
      // Two large rationals that don't cancel — must yield a single rational.
      assert_eq!(
        interpret(
          "(2479392929313226753685415739663229/870) * \
           (162766257435800426594527332711184631/57855)"
        )
        .unwrap(),
        "57651643973871427939970721018876416273348704855347199824144797233357/\
         7190550"
      );
    }

    #[test]
    fn small_products_unaffected() {
      assert_eq!(interpret("(7/3) * (2/5) * (3/7)").unwrap(), "2/5");
    }
  }

  // An association used as a function is a key lookup, so it works as the
  // function in Map: `assoc /@ {keys}` → the corresponding values.
  mod association_as_map_function {
    use super::*;

    #[test]
    fn map_association_over_keys() {
      assert_eq!(
        interpret("<|\"a\" -> 10, \"b\" -> 20|> /@ {\"b\", \"a\", \"a\"}")
          .unwrap(),
        "{20, 10, 10}"
      );
    }

    #[test]
    fn map_explicit_form() {
      assert_eq!(
        interpret("Map[<|1 -> \"x\", 2 -> \"y\"|>, {2, 1}]").unwrap(),
        "{y, x}"
      );
    }
  }

  // In a rule/condition context (the RHS of `->`, `/.`, or a List item) the
  // Map operator `/@` must be recognised before the bare arithmetic `/`,
  // otherwise `b /@ c` is mis-parsed as `b / (@c)` and the parse fails.
  // (ranking_methods)
  mod map_in_rule_context {
    use super::*;

    #[test]
    fn map_on_rule_rhs() {
      assert_eq!(
        interpret("{1, 2, 3} /. x_List :> (# + 1 & /@ x)").unwrap(),
        "{2, 3, 4}"
      );
    }

    #[test]
    fn map_inside_list_rule() {
      assert_eq!(
        interpret("f[x_] := x + 10; {a -> f /@ {1, 2}}").unwrap(),
        "{a -> {11, 12}}"
      );
    }

    #[test]
    fn map_on_replaceall_rhs() {
      assert_eq!(
        interpret("g[x_] := 2 x; 5 /. n_ :> Total[g /@ {n, n}]").unwrap(),
        "20"
      );
    }
  }

  // `Symbol["name"]` denotes the symbol `name` itself, so it must unify with
  // a plain `name` for both assignment and lookup. (dynamic_variable_names)
  mod symbol_from_string {
    use super::*;

    #[test]
    fn assign_via_symbol_then_read_bare() {
      assert_eq!(interpret("Set[Symbol[\"xy\"], 99]; xy").unwrap(), "99");
    }

    #[test]
    fn read_via_symbol_after_bare_assign() {
      assert_eq!(interpret("xy = 7; Symbol[\"xy\"]").unwrap(), "7");
    }

    #[test]
    fn unbound_symbol_returns_itself() {
      assert_eq!(interpret("Symbol[\"neverset\"]").unwrap(), "neverset");
    }
  }

  // `Import[file, "Text"]` returns the file contents as a string, the same
  // element wolframscript accepts. (letter_frequency, globally_replace_…)
  mod import_text_element {
    use super::*;

    #[test]
    fn text_element_reads_full_contents() {
      let path = std::env::temp_dir()
        .join("woxi_import_text_element.txt")
        .to_string_lossy()
        .into_owned();
      let _ = std::fs::remove_file(&path);
      let script = format!(
        "Export[\"{path}\", \"hello world\", \"Text\"]; \
         Print[Import[\"{path}\", \"Text\"]]"
      );
      let out = interpret_with_stdout(&script).unwrap().stdout;
      let _ = std::fs::remove_file(&path);
      assert_eq!(out, "hello world\n");
    }
  }
}
