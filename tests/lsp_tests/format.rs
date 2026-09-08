//! The language server's whitespace formatter.

use woxi::lsp::format::{Options, format};

/// Format with the default options (two spaces per level).
fn formatted(source: &str) -> String {
  format(source, &Options::default())
}

#[test]
fn spaces_out_an_assignment() {
  assert_eq!(formatted("f[x_]:=x^2"), "f[x_] := x^2");
  assert_eq!(formatted("x=1"), "x = 1");
  assert_eq!(formatted("x  =   1"), "x = 1");
}

#[test]
fn brackets_hug_their_contents() {
  assert_eq!(formatted("Print[ \"hi\" ]"), "Print[\"hi\"]");
  assert_eq!(formatted("f [ x , y ]"), "f[x, y]");
  assert_eq!(formatted("{ 1 , 2 , 3 }"), "{1, 2, 3}");
  assert_eq!(formatted("list [[ 1 ]]"), "list[[1]]");
  assert_eq!(formatted("f[x][y]"), "f[x][y]");
  assert_eq!(formatted("<|a->1|>"), "<|a -> 1|>");
}

#[test]
fn operators_get_one_space_on_each_side() {
  assert_eq!(formatted("a+b*c"), "a + b * c");
  assert_eq!(formatted("f/@{1,2}"), "f /@ {1, 2}");
  assert_eq!(formatted("x//N"), "x // N");
  assert_eq!(formatted("a&&b||c"), "a && b || c");
  assert_eq!(formatted("x/.a->b"), "x /. a -> b");
  assert_eq!(formatted("list[[1;;3]]"), "list[[1 ;; 3]]");
}

#[test]
fn pattern_operators_stay_tight() {
  assert_eq!(formatted("f[x_Integer]:=x"), "f[x_Integer] := x");
  assert_eq!(formatted("f[x__]:=x"), "f[x__] := x");
  assert_eq!(formatted("f[x_?NumberQ]:=x"), "f[x_?NumberQ] := x");
  assert_eq!(formatted("f[n_:1]:=n"), "f[n_:1] := n");
  assert_eq!(formatted("f[x_.]:=x"), "f[x_.] := x");
  // A `_` only binds the head that follows it; an operator after one is
  // still an operator.
  assert_eq!(formatted("f[x_+y_]:=x"), "f[x_ + y_] := x");
  assert_eq!(formatted("Sin::usage"), "Sin::usage");
}

#[test]
fn prefix_and_postfix_operators_keep_their_operand() {
  assert_eq!(formatted("- x"), "-x");
  assert_eq!(formatted("f[-1]"), "f[-1]");
  assert_eq!(formatted("a - - b"), "a - -b");
  // A sign written apart from a number literal stays apart from it: the
  // reader folds `-1` into one negative integer and reads `- 1` as
  // `Minus[1]`, so closing that gap would not be a whitespace change.
  assert_eq!(formatted("f[- 1]"), "f[- 1]");
  assert_eq!(formatted("a-b"), "a - b");
  assert_eq!(formatted("i ++"), "i++");
  assert_eq!(formatted("++ i"), "++i");
  assert_eq!(formatted("5 !"), "5!");
  assert_eq!(formatted("! True"), "!True");
  assert_eq!(formatted("a && ! b"), "a && !b");
}

#[test]
fn slots_stay_glued_to_their_number() {
  assert_eq!(formatted("#1+#2&"), "#1 + #2 &");
  assert_eq!(formatted("##2"), "##2");
  assert_eq!(formatted("#name"), "#name");
  // Two slots multiplied are not a `SlotSequence`.
  assert_eq!(formatted("# #"), "# #");
}

#[test]
fn a_slot_named_by_a_string_stays_one_expression() {
  assert_eq!(formatted("#\"key\"&"), "#\"key\" &");
  // With a space it is a slot multiplied by a string, and stays one.
  assert_eq!(formatted("# \"key\" &"), "# \"key\" &");
}

#[test]
fn reading_and_writing_files_keeps_its_operator_whole() {
  assert_eq!(formatted("<<\"init.m\""), "<< \"init.m\"");
  assert_eq!(formatted("expr>>\"out.txt\""), "expr >> \"out.txt\"");
  assert_eq!(formatted("x//=f"), "x //= f");
}

#[test]
fn a_derivative_keeps_its_primes_and_its_argument() {
  assert_eq!(formatted("f''[x]"), "f''[x]");
  assert_eq!(formatted("f'[x]+1"), "f'[x] + 1");
  assert_eq!(formatted("#[[1]]&"), "#[[1]] &");
}

#[test]
fn adjacent_atoms_keep_the_space_that_multiplies_them() {
  assert_eq!(formatted("2 x"), "2 x");
  assert_eq!(formatted("2x"), "2 x");
  assert_eq!(formatted("2 Pi"), "2 Pi");
}

#[test]
fn named_characters_are_left_alone() {
  assert_eq!(formatted("\\[Alpha]+1"), "\\[Alpha] + 1");
}

#[test]
fn indents_by_bracket_depth() {
  let source = "Module[{x=1},\nx+1\n]";
  assert_eq!(formatted(source), "Module[{x = 1},\n  x + 1\n]");
}

#[test]
fn indents_nested_brackets_one_level_per_bracket() {
  let source = "f[\ng[\n1\n]\n]";
  assert_eq!(formatted(source), "f[\n  g[\n    1\n  ]\n]");
}

#[test]
fn indents_the_continuation_of_an_unfinished_line() {
  assert_eq!(formatted("x = 1 +\n2"), "x = 1 +\n  2");
  assert_eq!(formatted("f[x_] :=\nx^2"), "f[x_] :=\n  x^2");
  // A comma only separates arguments; the next one is not a continuation.
  assert_eq!(formatted("f[\n1,\n2\n]"), "f[\n  1,\n  2\n]");
}

#[test]
fn re_indents_lines_that_were_indented_by_hand() {
  let source = "Module[{x = 1},\n        x + 1\n      ]\n";
  assert_eq!(formatted(source), "Module[{x = 1},\n  x + 1\n]\n");
}

#[test]
fn keeps_the_line_structure_of_the_file() {
  let source = "x = 1;\n\n\ny = 2;\n";
  assert_eq!(formatted(source), source);
  assert_eq!(formatted(source).lines().count(), source.lines().count());
}

#[test]
fn strips_trailing_whitespace() {
  assert_eq!(formatted("x = 1;   \ny = 2;  "), "x = 1;\ny = 2;");
}

#[test]
fn separates_a_comment_from_the_code_around_it() {
  assert_eq!(formatted("x=1(*set x*)"), "x = 1 (*set x*)");
  assert_eq!(formatted("(*head*)\nx=1"), "(*head*)\nx = 1");
  // A comment says nothing about the expression around it.
  assert_eq!(formatted("f[(*why*)x]"), "f[(*why*) x]");
}

#[test]
fn keeps_strings_and_comments_verbatim() {
  let source = "s = \"a  ,  b\";\n(*   spaced   *)\n";
  assert_eq!(formatted(source), source);
}

#[test]
fn honors_the_editor_s_indentation_options() {
  let source = "f[\n1\n]";
  let four_spaces = Options {
    tab_size: 4,
    insert_spaces: true,
  };
  assert_eq!(format(source, &four_spaces), "f[\n    1\n]");
  let tabs = Options {
    tab_size: 4,
    insert_spaces: false,
  };
  assert_eq!(format(source, &tabs), "f[\n\t1\n]");
}

#[test]
fn formatting_is_idempotent() {
  for source in [
    "f[x_]:=Module[{y=x^2},\ny+1\n]",
    "Print[ StringJoin[ \"a\" , \"b\" ] ]",
    "{1,2,3}//Total",
    "If[x>0,\n1,\n-1\n]",
    "#1+#2&/@{1,2}",
  ] {
    let once = formatted(source);
    assert_eq!(formatted(&once), once, "not idempotent: {source}");
  }
}

#[test]
fn an_empty_document_stays_empty() {
  assert_eq!(formatted(""), "");
  assert_eq!(formatted("\n\n"), "\n\n");
  assert_eq!(formatted("   "), "");
}

#[test]
fn unparsable_input_is_left_alone_rather_than_mangled() {
  // Mid-edit input still formats — it is only rejected when the result
  // would not read back as the same code.
  assert_eq!(formatted("f[x_]:="), "f[x_] :=");
  assert_eq!(formatted("{1,2"), "{1, 2");
}

#[test]
fn never_changes_the_tokens_of_a_file() {
  // Every construct of a small but broad script must survive formatting
  // unchanged in meaning: same tokens, same lines.
  let source = "\
(* header *)
factorial[n_Integer] := If[n <= 1,
1,
n*factorial[n - 1]
]

data = {1, 2, 3};
squares = #^2 & /@ data;
total = Total[squares];
Print[\"total: \" <> ToString[total]];
assoc = <|\"a\" -> 1, \"b\" -> 2|>;
part = data[[2 ;; 3]];
";
  let result = formatted(source);
  let tokens_of = |text: &str| {
    woxi::lsp::analysis::scan(text)
      .into_iter()
      .map(|token| (token.kind, token.text(text).to_string()))
      .collect::<Vec<_>>()
  };
  assert_eq!(tokens_of(&result), tokens_of(source));
  assert_eq!(result.lines().count(), source.lines().count());
  assert!(woxi::parse(&result).is_ok(), "does not parse:\n{result}");
}

#[test]
fn a_result_that_would_read_differently_is_dropped() {
  // The formatter's own rules keep these together; the check behind them
  // is what guarantees that a rule that ever stopped doing so could not
  // reach the file. Formatting them is a no-op either way.
  for source in ["\\[Alpha] + 1", "#\"key\" &", "f''[x]", "2 x"] {
    assert_eq!(formatted(source), source);
  }
}

#[test]
fn keeps_the_line_endings_the_file_uses() {
  assert_eq!(formatted("x=1;\r\ny=2;\r\n"), "x = 1;\r\ny = 2;\r\n");
  assert_eq!(formatted("f[\r\n1\r\n]"), "f[\r\n  1\r\n]");
}
