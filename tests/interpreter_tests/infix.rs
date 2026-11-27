use super::*;

mod infix_text_mode {
  use super::*;

  /// The printed display (a bare echo, `Print`) keeps the wrappers as
  /// calls, as wolframscript's script mode does; only `ToString` lays
  /// them out.
  #[test]
  fn printed_display_keeps_the_call() {
    clear_state();
    assert_eq!(
      interpret(r#"Infix[{a, b, c}, "+"]"#).unwrap(),
      "Infix[{a, b, c}, +]"
    );
    assert_eq!(interpret("Infix[f[a, b, c]]").unwrap(), "Infix[f[a, b, c]]");
    assert_eq!(
      interpret(r#"{1, Infix[{a, b}, "+"]}"#).unwrap(),
      "{1, Infix[{a, b}, +]}"
    );
    assert_eq!(interpret("Prefix[f[x]]").unwrap(), "Prefix[f[x]]");
    assert_eq!(interpret("Postfix[f[x]]").unwrap(), "Postfix[f[x]]");
    assert_eq!(
      interpret_with_stdout(r#"Print[Infix[{a, b}, "+"]]"#)
        .unwrap()
        .stdout,
      "Infix[{a, b}, +]\n"
    );
  }

  #[test]
  fn string_operator_joins_list_elements() {
    clear_state();
    assert_eq!(
      interpret(r#"ToString[Infix[{a, b, c}, "+"]]"#).unwrap(),
      "a+b+c"
    );
    assert_eq!(
      interpret(r#"ToString[Infix[{1, 2}, " <> "]]"#).unwrap(),
      "1 <> 2"
    );
    assert_eq!(
      interpret(r#"ToString[{Infix[{a, b}, "+"]}]"#).unwrap(),
      "{a+b}"
    );
  }

  #[test]
  fn operands_of_any_head_are_joined() {
    clear_state();
    assert_eq!(
      interpret(r#"ToString[Infix[f[x, y, z], "|"]]"#).unwrap(),
      "x|y|z"
    );
    assert_eq!(
      interpret(r#"ToString[Infix[f[a, b], "+"]]"#).unwrap(),
      "a+b"
    );
  }

  #[test]
  fn bare_call_uses_its_head_between_tildes() {
    clear_state();
    assert_eq!(
      interpret("ToString[Infix[f[a, b, c]]]").unwrap(),
      "a ~f~ b ~f~ c"
    );
    // A bare list's head is `List`.
    assert_eq!(interpret("ToString[Infix[{a, b}]]").unwrap(), "a ~List~ b");
    assert_eq!(
      interpret("ToString[Infix[f[Infix[g[a, b]], c]]]").unwrap(),
      "a ~g~ b ~f~ c"
    );
  }

  #[test]
  fn non_string_operator_prints_in_output_form() {
    clear_state();
    assert_eq!(interpret("ToString[Infix[{a, b}, f]]").unwrap(), "afb");
    // A list of operators is shown as a list between each pair.
    assert_eq!(
      interpret(r#"ToString[Infix[{a, b, c}, {"+", "-"}]]"#).unwrap(),
      "a{+, -}b{+, -}c"
    );
  }

  #[test]
  fn fewer_than_two_operands_keep_the_call() {
    clear_state();
    assert_eq!(
      interpret(r#"ToString[Infix[{a}, "+"]]"#).unwrap(),
      "Infix[{a}, +]"
    );
    assert_eq!(interpret("ToString[Infix[f[a]]]").unwrap(), "Infix[f[a]]");
    assert_eq!(interpret("ToString[Infix[f[]]]").unwrap(), "Infix[f[]]");
  }

  /// An operand printing with an operator that binds no tighter than the
  /// precedence (`630`, that of `~`, by default) is bracketed.
  #[test]
  fn operands_are_bracketed_by_precedence() {
    clear_state();
    assert_eq!(
      interpret("ToString[Infix[f[a + b, c d]]]").unwrap(),
      "(a + b) ~f~ (c d)"
    );
    assert_eq!(
      interpret(
        r#"{ToString[Infix[{a b, c}, "+"]], ToString[Infix[{-a, b}, "+"]], ToString[Infix[{a -> b, c}, "+"]], ToString[Infix[{a!, c}, "+"]], ToString[Infix[{Derivative[1][a], c}, "+"]], ToString[Infix[{a . b, c}, "+"]], ToString[Infix[{-2, c}, "+"]], ToString[Infix[{1.5, c}, "+"]]}"#
      )
      .unwrap(),
      "{(a b)+c, (-a)+b, (a -> b)+c, (a!)+c, a'+c, (a . b)+c, (-2)+c, 1.5+c}"
    );
    assert_eq!(
      interpret(
        r#"{ToString[Infix[{a + b, c}, "+", 300]], ToString[Infix[{a + b, c}, "+", 310]], ToString[Infix[{a!, c}, "+", 610]], ToString[Infix[{a!, c}, "+", 609]]}"#
      )
      .unwrap(),
      "{a + b+c, (a + b)+c, (a!)+c, a!+c}"
    );
    // A nested `Infix` binds with its own precedence.
    assert_eq!(
      interpret(
        r#"{ToString[Infix[{Infix[{a, b}, "+", 100], c}, "*", 100]], ToString[Infix[{Infix[{a, b}, "+", 101], c}, "*", 100]]}"#
      )
      .unwrap(),
      "{(a+b)*c, a+b*c}"
    );
  }

  /// A precedence that is not a machine integer, or a grouping that is not
  /// a grouping specification, keeps the call and is reported.
  #[test]
  fn invalid_precedence_or_grouping_keeps_the_call() {
    clear_state();
    let result =
      interpret_with_stdout(r#"ToString[Infix[{a, b}, "+", x]]"#).unwrap();
    assert_eq!(result.result, "Infix[{a, b}, +, x]");
    assert!(
      result.warnings.iter().any(|w| w.contains(
        "Infix::intm: Machine-sized integer expected at position 3 in \
         Infix[{a, b}, +, x]."
      )),
      "{:?}",
      result.warnings
    );
    let result =
      interpret_with_stdout(r#"ToString[Infix[{a, b}, "+", 100, foo]]"#)
        .unwrap();
    assert_eq!(result.result, "Infix[{a, b}, +, 100, foo]");
    assert!(
      result.warnings.iter().any(|w| w.contains(
        "Infix::group: Grouping specification foo is not NonAssociative, \
         None, Left or Right."
      )),
      "{:?}",
      result.warnings
    );
    assert_eq!(
      interpret(r#"ToString[Infix[{a, b, c}, "*", 400, Left]]"#).unwrap(),
      "a*b*c"
    );
  }

  #[test]
  fn stays_inert() {
    clear_state();
    assert_eq!(interpret(r#"Head[Infix[{a, b}, "+"]]"#).unwrap(), "Infix");
    assert_eq!(
      interpret(r#"ToString[InputForm[Infix[{a, b}, "+"]]]"#).unwrap(),
      r#"Infix[{a, b}, "+"]"#
    );
    assert_eq!(
      interpret(r#"ToString[Infix[{a, b, c}, {"+", "-"}], InputForm]"#)
        .unwrap(),
      r#"Infix[{a, b, c}, {"+", "-"}]"#
    );
  }

  #[test]
  fn boxes_are_a_row_with_the_operator_text() {
    clear_state();
    assert_eq!(
      interpret(r#"ToBoxes[Infix[{a, b}, "+"]]"#).unwrap(),
      "InterpretationBox[RowBox[{a, +, b}], Infix[{a, b}, +], Editable -> False]"
    );
    assert_eq!(
      interpret(r#"ToString[ToBoxes[Infix[{a, b}, "+"]], InputForm]"#).unwrap(),
      r#"InterpretationBox[RowBox[{"a", "+", "b"}], Infix[{a, b}, "+"], Editable -> False]"#
    );
    // A sum operand is bracketed; `Infix[f[a, b, c]]` puts `~f~` between.
    assert_eq!(
      interpret(r#"ToString[{ToBoxes[Infix[{a + b, c}, "*"]], ToBoxes[Infix[f[a, b]]]}, InputForm]"#).unwrap(),
      r#"{InterpretationBox[RowBox[{RowBox[{"(", RowBox[{"a", "+", "b"}], ")"}], "*", "c"}], Infix[{a + b, c}, "*"], Editable -> False], InterpretationBox[RowBox[{"a", " ", "~", "f", "~", " ", "b"}], Infix[f[a, b]], Editable -> False]}"#
    );
    // With a precedence the wrapper boxes as a plain call.
    assert_eq!(
      interpret(r#"ToBoxes[Infix[{a, b}, "~", 300, Left]]"#).unwrap(),
      r#"RowBox[{Infix, [, RowBox[{RowBox[{{, RowBox[{a, ,, b}], }}], ,, "~", ,, 300, ,, Left}], ]}]"#
    );
    // So does one with fewer than two operands.
    assert_eq!(
      interpret("ToBoxes[Infix[f[a]]]").unwrap(),
      "RowBox[{Infix, [, RowBox[{f, [, a, ]}], ]}]"
    );
    assert_eq!(
      interpret("ToBoxes[Infix[{a, b}]]").unwrap(),
      "InterpretationBox[RowBox[{a,  , ~, List, ~,  , b}], Infix[{a, b}], Editable -> False]"
    );
  }

  /// In boxes an operand is bracketed when it binds no tighter than the
  /// operator token: `+` keeps a product bare, `*` brackets it.
  #[test]
  fn boxed_operands_are_bracketed_by_the_operator() {
    clear_state();
    let row = |input: &str| {
      let boxes =
        interpret(&format!("ToString[ToBoxes[{input}], InputForm]")).unwrap();
      let start = "InterpretationBox[".len();
      let end = boxes
        .rfind(", Infix[")
        .or_else(|| boxes.rfind(", Prefix["))
        .or_else(|| boxes.rfind(", Postfix["))
        .unwrap();
      boxes[start..end].to_string()
    };
    assert_eq!(
      row(r#"Infix[{a b, c}, "+"]"#),
      r#"RowBox[{RowBox[{"a", " ", "b"}], "+", "c"}]"#
    );
    assert_eq!(
      row(r#"Infix[{a b, c}, "*"]"#),
      r#"RowBox[{RowBox[{"(", RowBox[{"a", " ", "b"}], ")"}], "*", "c"}]"#
    );
    assert_eq!(
      row(r#"Infix[{a!, c}, "+"]"#),
      r#"RowBox[{RowBox[{"a", "!"}], "+", "c"}]"#
    );
    assert_eq!(
      row(r#"Infix[{-2, c}, "*"]"#),
      r#"RowBox[{RowBox[{"-", "2"}], "*", "c"}]"#
    );
    assert_eq!(
      row(r#"Infix[{-a, c}, "^"]"#),
      r#"RowBox[{RowBox[{"(", RowBox[{"-", "a"}], ")"}], "^", "c"}]"#
    );
    assert_eq!(
      row(r#"Infix[{a^b, c}, "^"]"#),
      r#"RowBox[{SuperscriptBox["a", "b"], "^", "c"}]"#
    );
    assert_eq!(
      row(r#"Infix[{a + b, c}, "&&"]"#),
      r#"RowBox[{RowBox[{"a", "+", "b"}], "&&", "c"}]"#
    );
    assert_eq!(
      row(r#"Infix[{a && b, c}, "&&"]"#),
      r#"RowBox[{RowBox[{"(", RowBox[{"a", "&&", "b"}], ")"}], "&&", "c"}]"#
    );
    assert_eq!(
      row("Infix[f[a!, c]]"),
      r#"RowBox[{RowBox[{"(", RowBox[{"a", "!"}], ")"}], " ", "~", "f", "~", " ", "c"}]"#
    );
    // An operator the FrontEnd does not know is a quoted string, binding
    // between `<>` and `!`.
    assert_eq!(
      row(r#"Infix[{a b, c}, "foo"]"#),
      r#"RowBox[{RowBox[{"(", RowBox[{"a", " ", "b"}], ")"}], "\"foo\"", "c"}]"#
    );
    assert_eq!(
      row(r#"Infix[{a!, c}, "foo"]"#),
      r#"RowBox[{RowBox[{"a", "!"}], "\"foo\"", "c"}]"#
    );
    assert_eq!(
      row(r#"Infix[{a + b, c}, f]"#),
      r#"RowBox[{RowBox[{"(", RowBox[{"a", "+", "b"}], ")"}], "f", "c"}]"#
    );
    // `Prefix` brackets for `@` (640), `Postfix` for `//` (70); an operator
    // given instead decides by its own precedence.
    assert_eq!(
      row("Prefix[f[a!]]"),
      r#"RowBox[{"f", "@", RowBox[{"(", RowBox[{"a", "!"}], ")"}]}]"#
    );
    assert_eq!(
      row(r#"Prefix[f[a + b], "!"]"#),
      r#"RowBox[{"!", RowBox[{"a", "+", "b"}]}]"#
    );
    assert_eq!(
      row("Postfix[f[a && b]]"),
      r#"RowBox[{RowBox[{"a", "&&", "b"}], "//", "f"}]"#
    );
    assert_eq!(
      row(r#"Postfix[f[a + b], "!"]"#),
      r#"RowBox[{RowBox[{"(", RowBox[{"a", "+", "b"}], ")"}], "!"}]"#
    );
  }

  #[test]
  fn prefix_and_postfix_boxes_and_text() {
    clear_state();
    assert_eq!(
      interpret(r#"ToString[{ToBoxes[Prefix[f[x]]], ToBoxes[Postfix[f[x], "!"]], ToBoxes[Prefix[f[a b]]]}, InputForm]"#).unwrap(),
      r#"{InterpretationBox[RowBox[{"f", "@", "x"}], Prefix[f[x]], Editable -> False], InterpretationBox[RowBox[{"x", "!"}], Postfix[f[x], "!"], Editable -> False], InterpretationBox[RowBox[{"f", "@", RowBox[{"(", RowBox[{"a", " ", "b"}], ")"}]}], Prefix[f[a*b]], Editable -> False]}"#
    );
    assert_eq!(
      interpret(r#"{ToString[Prefix[f[x]]], ToString[Postfix[f[a + b]]], ToString[Prefix[f[x], "!"]], ToString[Prefix[f[a + b]]]}"#).unwrap(),
      "{f@ x, a + b // f, !x, f@ (a + b)}"
    );
    assert_eq!(
      interpret(r#"{ToString[Prefix[f[a!]]], ToString[Prefix[f[a + b], "!"]], ToString[Postfix[f[a + b], "!"]], ToString[Prefix[f[a!], "-", 700]], ToString[Prefix[f[a + b], "!", 100]], ToString[Prefix[{a}]], ToString[Prefix[f[a], "!", 100, Left]], ToString[Postfix[f[a, b]]]}"#).unwrap(),
      "{f@ (a!), !(a + b), a + b!, -(a!), !a + b, List@ a, !a, Postfix[f[a, b]]}"
    );
    let result =
      interpret_with_stdout(r#"ToString[Prefix[f[x], "-", x]]"#).unwrap();
    assert_eq!(result.result, "Prefix[f[x], -, x]");
    assert!(
      result.warnings.iter().any(|w| w.contains(
        "Prefix::intm: Machine-sized integer expected at position 3 in \
         Prefix[f[x], -, x]."
      )),
      "{:?}",
      result.warnings
    );
  }

  #[test]
  fn tex_form_writes_the_operator_literally() {
    clear_state();
    assert_eq!(
      interpret(r#"ToString[TeXForm[Infix[{a, b}, "+"]]]"#).unwrap(),
      "a+b"
    );
    assert_eq!(
      interpret(r#"ToString[TeXForm[Infix[{a + b, c}, "*"]]]"#).unwrap(),
      "(a+b)*c"
    );
    assert_eq!(
      interpret("ToString[TeXForm[Infix[f[a, b]]]]").unwrap(),
      "a \\sim f\\sim  b"
    );
  }

  #[test]
  fn traditional_form_boxes_read_back_as_the_wrapper() {
    clear_state();
    assert_eq!(
      interpret(r#"ToBoxes[Infix[{a + b, c}, "*"], TraditionalForm]"#).unwrap(),
      "FormBox[InterpretationBox[RowBox[{RowBox[{(, RowBox[{a, +, b}], )}], *, c}], Infix[{a + b, c}, *], Editable -> False], TraditionalForm]"
    );
    assert_eq!(
      interpret("ToBoxes[Prefix[f[a b]], TraditionalForm]").unwrap(),
      "FormBox[InterpretationBox[RowBox[{f, @, RowBox[{(, RowBox[{a,  , b}], )}]}], Prefix[f[a*b]], Editable -> False], TraditionalForm]"
    );
  }

  /// A sum printed with its terms in a custom order — the operands of a
  /// `Plus` sorted by length and joined by `Infix` — keeps that order.
  #[test]
  fn reordered_sum_terms() {
    clear_state();
    assert_eq!(
      interpret(
        r#"pp[s_Plus] := Infix[Reverse@SortBy[List @@ s, Length], "+"];
           ToString[pp[Expand[(x + 1) (y + 1)]]]"#
      )
      .unwrap(),
      "(x y)+y+x+1"
    );
  }
}

mod infix_visual_mode {
  use super::*;

  /// `TraditionalForm` typesets the operands — subscripts included — with
  /// the operator text between them instead of echoing the `Infix` call.
  #[test]
  fn traditional_form_typesets_operands_and_operator() {
    clear_state();
    let result = interpret_with_stdout(
      r#"Text[Style[TraditionalForm[Infix[{Subscript[a, 1] Subscript[a, 2], Subscript[a, 3]}, "+"]], 21]]"#,
    )
    .unwrap();
    // A `Text`-wrapped typeset formula is drawn as a picture.
    let svg = result
      .graphics
      .or(result.output_svg)
      .expect("typeset output");
    assert!(
      !svg.contains(">Infix<"),
      "the call must not be echoed: {svg}"
    );
    assert!(svg.contains(">+<"), "the operator is drawn: {svg}");
    assert!(svg.contains(">3<"), "the subscripts are drawn: {svg}");
  }
}
