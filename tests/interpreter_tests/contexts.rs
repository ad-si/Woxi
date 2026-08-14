//! Contexts: symbols are resolved when they are read, so a package's private
//! helpers stay private and its exported names become visible.
//!
//! Every expectation here was verified against wolframscript.

use super::*;

/// A package declaring `pub` and hiding `priv`, the standard idiom.
const PACKAGE: &str = "BeginPackage[\"P`\"]\n\
   ctxPub::usage = \"u\";\n\
   Begin[\"`Private`\"]\n\
   ctxPriv[] := 11\n\
   ctxPub[] := ctxPriv[] + 1\n\
   End[]\n\
   EndPackage[]\n";

mod package_idiom {
  use super::*;

  // The whole point: a symbol defined inside `Begin["`Private`"]` belongs to
  // the private context, so the name means nothing outside the package,
  // while one declared before it is exported.
  #[test]
  fn private_symbols_stay_private_and_public_ones_do_not() {
    clear_state();
    assert_eq!(interpret(&format!("{PACKAGE}ctxPub[]")).unwrap(), "12");
    // `ctxPriv` outside the package is a fresh Global` symbol with no value.
    assert_eq!(interpret("ctxPriv").unwrap(), "ctxPriv");
    assert_eq!(interpret("Context[ctxPriv]").unwrap(), "Global`");
    assert_eq!(interpret("Context[ctxPub]").unwrap(), "P`");
    // It is still reachable under its full name.
    assert_eq!(interpret("P`Private`ctxPriv[]").unwrap(), "11");
  }

  // Two packages that both define a helper called `helper` get one each.
  // With a flat namespace the second would silently overwrite the first.
  #[test]
  fn two_packages_do_not_share_a_private_helper() {
    clear_state();
    interpret(
      "BeginPackage[\"A1`\"]\na1f::usage = \"u\";\nBegin[\"`Private`\"]\n\
       helper[x_] := x + 1\na1f[x_] := helper[x]\nEnd[]\nEndPackage[]\n",
    )
    .unwrap();
    interpret(
      "BeginPackage[\"B1`\"]\nb1f::usage = \"u\";\nBegin[\"`Private`\"]\n\
       helper[x_] := x * 100\nb1f[x_] := helper[x]\nEnd[]\nEndPackage[]\n",
    )
    .unwrap();
    assert_eq!(interpret("a1f[1]").unwrap(), "2");
    assert_eq!(interpret("b1f[1]").unwrap(), "100");
    assert_eq!(
      interpret("ToString[Names[\"*`helper\"], InputForm]").unwrap(),
      r#"{"A1`Private`helper", "B1`Private`helper"}"#
    );
  }

  // A name already visible on `$ContextPath` is *not* re-created in the
  // current context — which is what makes the `f::usage` export idiom work:
  // the definition inside `Private` attaches to the public symbol.
  #[test]
  fn a_definition_attaches_to_the_declared_public_symbol() {
    clear_state();
    interpret(PACKAGE).unwrap();
    assert_eq!(
      interpret("ToString[Names[\"P`*\"], InputForm]").unwrap(),
      r#"{"ctxPub"}"#
    );
    assert_eq!(
      interpret("ToString[Names[\"P`Private`*\"], InputForm]").unwrap(),
      r#"{"P`Private`ctxPriv"}"#
    );
  }

  // `BeginPackage["P`", {"Q`"}]` needs `Q`` while reading, and both contexts
  // are on `$ContextPath` afterwards.
  #[test]
  fn end_package_puts_the_package_on_the_context_path() {
    clear_state();
    interpret(
      "BeginPackage[\"MyPackage`\", {\"VectorAnalysis`\"}]\n\
       Begin[\"`Private`\"]\nEnd[]\nEndPackage[]\n",
    )
    .unwrap();
    assert_eq!(
      interpret("ToString[$ContextPath, InputForm]").unwrap(),
      r#"{"MyPackage`", "VectorAnalysis`", "System`", "Global`"}"#
    );
  }
}

mod symbol_creation {
  use super::*;

  // `Begin` moves `$Context` without touching `$ContextPath`, so new names
  // land in the new context while everything visible stays visible.
  #[test]
  fn begin_creates_symbols_in_the_new_context() {
    clear_state();
    assert_eq!(
      interpret("Begin[\"A`\"]\nctxX = 5\nContext[ctxX]").unwrap(),
      "A`"
    );
    assert_eq!(interpret("End[]\nContext[ctxX]").unwrap(), "Global`");
    assert_eq!(interpret("A`ctxX").unwrap(), "5");
    // The name outside is a different symbol, with no value.
    assert_eq!(interpret("ctxX").unwrap(), "ctxX");
  }

  // Merely mentioning a name creates the symbol — that is how a package can
  // declare an export without giving it a value.
  #[test]
  fn mentioning_a_name_creates_it_in_the_current_context() {
    clear_state();
    interpret("Begin[\"A`\"]\nctxY\nEnd[]\n").unwrap();
    assert_eq!(
      interpret("ToString[Names[\"A`*\"], InputForm]").unwrap(),
      r#"{"A`ctxY"}"#
    );
  }

  #[test]
  fn begin_blocks_nest() {
    clear_state();
    interpret("Begin[\"N1`\"]\nctxN1 = 1\nBegin[\"N2`\"]\nctxN2 = 2\n")
      .unwrap();
    assert_eq!(interpret("Context[ctxN1]").unwrap(), "N2`");
    assert_eq!(interpret("Context[ctxN2]").unwrap(), "N2`");
    interpret("End[];").unwrap();
    assert_eq!(interpret("Context[ctxN2]").unwrap(), "N1`");
    interpret("End[];").unwrap();
  }

  // Pattern variables and scoping-construct locals are symbols of the
  // context they are read in, and `Names` reports them.
  #[test]
  fn pattern_variables_and_locals_belong_to_the_context() {
    clear_state();
    interpret(
      "BeginPackage[\"R`\"]\nrpub::usage = \"u\";\nBegin[\"`Private`\"]\n\
       rpub[n_] := Module[{loc = n}, Block[{bv = 1}, With[{wv = 2}, \
       loc + bv + wv]]]\nEnd[]\nEndPackage[]\n",
    )
    .unwrap();
    assert_eq!(interpret("rpub[3]").unwrap(), "6");
    // The `$`-suffixed names a scoping construct mints while it runs are
    // filtered out: Woxi numbers them (`loc$1`) where wolframscript does not
    // (`loc$`), an artefact of how each renames locals.
    assert_eq!(
      interpret(
        "ToString[Select[Names[\"R`Private`*\"],          !StringContainsQ[#, \"$\"] &], InputForm]"
      )
      .unwrap(),
      r#"{"R`Private`bv", "R`Private`loc", "R`Private`n", "R`Private`wv"}"#
    );
  }

  // A name given at runtime resolves against the contexts open at that
  // moment, exactly as one read from source does.
  #[test]
  fn symbol_and_to_expression_resolve_at_runtime() {
    clear_state();
    interpret(
      "BeginPackage[\"G1`\"]\ng1f::usage = \"u\";\nBegin[\"`Private`\"]\n\
       Symbol[\"dyn\"]\nToExpression[\"dynb\"]\nEnd[]\nEndPackage[]\n",
    )
    .unwrap();
    assert_eq!(
      interpret("ToString[Names[\"G1`Private`*\"], InputForm]").unwrap(),
      r#"{"G1`Private`dyn", "G1`Private`dynb"}"#
    );
  }

  // `Remove` makes a symbol stop existing, so its context no longer has it.
  #[test]
  fn remove_drops_the_symbol_from_its_context() {
    clear_state();
    interpret(PACKAGE).unwrap();
    interpret("Remove[ctxPub]").unwrap();
    assert_eq!(
      interpret("ToString[Names[\"P`*\"], InputForm]").unwrap(),
      "{}"
    );
  }
}

mod shadowing {
  use super::*;

  // Creating a symbol whose short name already lives in another visible
  // context reports `::shdw`, as wolframscript does.
  #[test]
  fn a_clashing_name_in_a_visible_context_is_reported() {
    clear_state();
    let result = interpret_with_stdout(
      "ctxShad = 1\nBeginPackage[\"Q`\"]\nctxShad::usage = \"u\";\n\
       Begin[\"`Private`\"]\nEnd[]\nEndPackage[]\n",
    )
    .unwrap();
    assert_eq!(
      result.stdout,
      "\nctxShad::shdw: Symbol ctxShad appears in multiple contexts \
       {Q`, Global`}; definitions in context Q` may shadow or be shadowed \
       by other definitions.\n"
    );
    // The package's symbol wins: it comes first on `$ContextPath`.
    assert_eq!(interpret("Context[ctxShad]").unwrap(), "Q`");
  }

  // A private context is on nobody's `$ContextPath`, so a `Global`` symbol
  // of the same name is not a clash and nothing is reported.
  #[test]
  fn a_private_name_does_not_clash() {
    clear_state();
    let result = interpret_with_stdout(&format!("{PACKAGE}ctxPriv")).unwrap();
    assert_eq!(result.stdout, "");
  }
}

mod display {
  use super::*;

  // A symbol prints under its short name wherever that reads back as the
  // same symbol, and under its full name otherwise.
  #[test]
  fn symbols_print_under_their_visible_name() {
    clear_state();
    interpret(PACKAGE).unwrap();
    assert_eq!(interpret("ctxPub").unwrap(), "ctxPub");
    assert_eq!(interpret("ToString[ctxPub, InputForm]").unwrap(), "ctxPub");
    assert_eq!(
      interpret("ToString[Hold[ctxPub[1]], InputForm]").unwrap(),
      "Hold[ctxPub[1]]"
    );
    // The private one is not visible here, so it prints in full.
    assert_eq!(
      interpret("ToString[Hold[P`Private`ctxPriv], InputForm]").unwrap(),
      "Hold[P`Private`ctxPriv]"
    );
  }

  #[test]
  fn definitions_show_visible_and_full_names() {
    clear_state();
    interpret(
      "BeginPackage[\"I1`\"]\ni1f::usage = \"d\";\nBegin[\"`Private`\"]\n\
       i1f[x_] := x\nEnd[]\nEndPackage[]\n",
    )
    .unwrap();
    assert_eq!(
      interpret("ToString[Definition[i1f], InputForm]").unwrap(),
      "i1f[I1`Private`x_] := I1`Private`x"
    );
    assert_eq!(
      interpret("ToString[DownValues[i1f], InputForm]").unwrap(),
      "{HoldPattern[i1f[I1`Private`x_]] :> I1`Private`x}"
    );
  }
}

mod names_and_contexts {
  use super::*;

  // A pattern's context part selects the context and its name part the
  // symbol, so `S`*` does not reach into `S`Private``.
  #[test]
  fn name_patterns_match_context_and_name_separately() {
    clear_state();
    interpret(
      "BeginPackage[\"S`\"]\nspub::usage = \"u\";\nBegin[\"`Private`\"]\n\
       sfun[a_] := a + 1\nEnd[]\nEndPackage[]\ngv = 3\n",
    )
    .unwrap();
    let names = |pattern: &str| {
      interpret(&format!("ToString[Names[\"{pattern}\"], InputForm]")).unwrap()
    };
    assert_eq!(names("S`*"), r#"{"spub"}"#);
    assert_eq!(names("S`Private`*"), r#"{"S`Private`a", "S`Private`sfun"}"#);
    assert_eq!(names("spub"), r#"{"spub"}"#);
    assert_eq!(names("Global`*"), r#"{"gv"}"#);
    assert_eq!(names("*`sfun"), r#"{"S`Private`sfun"}"#);
  }

  // Built-ins are `System`` symbols, which is why a pattern without a
  // context still finds them.
  #[test]
  fn builtins_are_system_symbols() {
    clear_state();
    assert_eq!(interpret("Names[\"List\"]").unwrap(), "{List}");
    assert_eq!(interpret("Context[Sin]").unwrap(), "System`");
    assert_eq!(
      interpret("ToString[Names[\"System`Sin\"], InputForm]").unwrap(),
      r#"{"Sin"}"#
    );
  }

  // wolframscript lists names case-insensitively with `$` after the letters.
  #[test]
  fn names_are_ordered_like_wolframscript() {
    clear_state();
    interpret(
      "zz$a::usage = \"u\"; zzb::usage = \"u\"; zz1::usage = \"u\"; \
       zzA::usage = \"u\"; zz$::usage = \"u\";",
    )
    .unwrap();
    assert_eq!(
      interpret("ToString[Names[\"zz*\"], InputForm]").unwrap(),
      r#"{"zz1", "zzA", "zzb", "zz$", "zz$a"}"#
    );
  }

  // `Context` reports the context of the *symbol*, not of its value.
  #[test]
  fn context_holds_its_argument() {
    clear_state();
    assert_eq!(
      interpret("Attributes[Context]").unwrap(),
      "{HoldFirst, Protected}"
    );
    assert_eq!(interpret("ctxV = 1; Context[ctxV]").unwrap(), "Global`");
    let result = interpret_with_stdout("Context[Symbol[\"dyn\"]]").unwrap();
    assert_eq!(result.result, "Context[Symbol[dyn]]");
    assert_eq!(
      result.stdout,
      "\nContext::ssle: Symbol or string expected at position 1 in \
       Context[Symbol[dyn]].\n"
    );
  }

  // A name given as a string resolves the way a read one would.
  #[test]
  fn context_of_a_string_resolves_it() {
    clear_state();
    interpret(PACKAGE).unwrap();
    assert_eq!(interpret("Context[\"ctxPub\"]").unwrap(), "P`");
  }

  #[test]
  fn information_reports_the_symbols_own_context() {
    clear_state();
    interpret(
      "BeginPackage[\"M1`\"]\nm1f::usage = \"m1f[x] does something\";\n\
       Begin[\"`Private`\"]\nm1f[x_] := x^2\nEnd[]\nEndPackage[]\n",
    )
    .unwrap();
    assert_eq!(
      interpret("Information[m1f, \"FullName\"]").unwrap(),
      "M1`m1f"
    );
    assert_eq!(
      interpret("Information[m1f, \"Usage\"]").unwrap(),
      "m1f[x] does something"
    );
    assert_eq!(interpret("Information[m1f, \"Attributes\"]").unwrap(), "{}");
    assert_eq!(
      interpret("ToString[Information[m1f, \"Bogus\"], InputForm]").unwrap(),
      r#"Missing["UnknownProperty", "Bogus"]"#
    );
  }
}
