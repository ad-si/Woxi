use super::*;

mod infix_text_mode {
  use super::*;

  #[test]
  fn string_operator_joins_list_elements() {
    clear_state();
    assert_eq!(interpret(r#"Infix[{a, b, c}, "+"]"#).unwrap(), "a+b+c");
    assert_eq!(interpret(r#"Infix[{1, 2}, " <> "]"#).unwrap(), "1 <> 2");
  }

  #[test]
  fn operands_of_any_head_are_joined() {
    clear_state();
    assert_eq!(interpret(r#"Infix[f[x, y, z], "|"]"#).unwrap(), "x|y|z");
    // Operands keep their own OutputForm.
    assert_eq!(
      interpret(r#"Infix[{a*b*c, a*c, 1}, "+"]"#).unwrap(),
      "a*b*c+a*c+1"
    );
  }

  #[test]
  fn bare_call_uses_its_head_between_tildes() {
    clear_state();
    assert_eq!(interpret("Infix[f[a, b, c]]").unwrap(), "a ~ f ~ b ~ f ~ c");
  }

  #[test]
  fn non_string_operator_prints_in_output_form() {
    clear_state();
    assert_eq!(interpret("Infix[{a, b}, x^2]").unwrap(), "ax^2b");
  }

  #[test]
  fn single_operand_is_shown_alone() {
    clear_state();
    assert_eq!(interpret(r#"Infix[{a}, "+"]"#).unwrap(), "a");
  }

  #[test]
  fn stays_inert() {
    clear_state();
    assert_eq!(interpret(r#"Head[Infix[{a, b}, "+"]]"#).unwrap(), "Infix");
    assert_eq!(
      interpret(r#"ToString[InputForm[Infix[{a, b}, "+"]]]"#).unwrap(),
      r#"Infix[{a, b}, "+"]"#
    );
  }

  #[test]
  fn list_of_operators_is_left_as_the_call() {
    clear_state();
    assert_eq!(
      interpret(r#"ToString[Infix[{a, b, c}, {"+", "-"}], InputForm]"#)
        .unwrap(),
      r#"Infix[{a, b, c}, {"+", "-"}]"#
    );
  }

  #[test]
  fn precedence_and_grouping_arguments_are_accepted() {
    clear_state();
    assert_eq!(
      interpret(r#"Infix[{a, b, c}, "*", 400, Left]"#).unwrap(),
      "a*b*c"
    );
  }

  #[test]
  fn boxes_are_a_row_with_the_operator_text() {
    clear_state();
    assert_eq!(
      interpret(r#"ToBoxes[Infix[{a, b}, "+"]]"#).unwrap(),
      "InterpretationBox[RowBox[{a, +, b}], a+b, Editable -> False]"
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
  }

  #[test]
  fn tex_form_writes_the_operator_literally() {
    clear_state();
    assert_eq!(
      interpret(r#"ToString[TeXForm[Infix[{a, b}, "+"]]]"#).unwrap(),
      "a+b"
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
           pp[Expand[(x + 1) (y + 1)]]"#
      )
      .unwrap(),
      "x*y+y+x+1"
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
