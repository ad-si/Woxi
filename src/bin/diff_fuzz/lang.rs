//! Language-construct cases for the differential fuzzer.
//!
//! The curated `FN_SPECS` table exercises builtins, but always as a FullForm
//! call on literal data: `Plus[1, 2]`, never `1 + 2`; `Map[(#^2 &), …]`,
//! never a pure function the fuzzer wrote itself. Most of what makes the
//! Wolfram Language a language — operator syntax, pure functions and slots,
//! scoping, pattern-based definitions, control flow, rules, Part syntax —
//! was out of its reach. This module generates small programs built from
//! exactly those constructs.
//!
//! Every program terminates and prints the same thing on every run:
//!
//! - Symbols that receive definitions (`f`, `g`, `a`, `b`, `s`) are cleared
//!   at the start of the program that assigns them, because the batch
//!   prefilter evaluates many cases in one session. Symbols used as free
//!   symbolic values (`p`, `q`, `x`, `y`) are never assigned at all;
//!   `Block` may rebind `x`, which it restores.
//! - Module variables are always initialized, so no `a$123` — whose number
//!   depends on `$ModuleNumber` — ever reaches the output.
//! - Recursion always has a decreasing step and a base case, and the parts
//!   that guarantee it are `Expr::Fixed`, so shrinking cannot remove them.

// `*rng.pick(…)` is load-bearing: auto-deref would shorten the picked
// `&'static str` to the slice borrow's lifetime, which `Expr::Sym` rejects.
#![allow(clippy::explicit_auto_deref)]

use super::*;

/// Symbols the programs here assign; cleared at the start of each program
/// that defines any of them.
const ASSIGNED_SYMS: &[&str] = &["f", "g", "a", "b", "s"];

fn sym(s: &'static str) -> Expr {
  Expr::Sym(s)
}

fn int(n: i64) -> Expr {
  Expr::Int(n)
}

fn call(head: &'static str, args: Vec<Expr>) -> Expr {
  Expr::Call(head, args)
}

fn infix(op: &'static str, args: Vec<Expr>) -> Expr {
  Expr::Infix(op, args)
}

fn fixed(e: Expr) -> Expr {
  Expr::Fixed(Box::new(e))
}

fn apply(head: Expr, args: Vec<Expr>) -> Expr {
  Expr::Apply(Box::new(head), args)
}

/// `body &`
fn pure(body: Expr) -> Expr {
  Expr::Unary("", Box::new(body), " &")
}

/// `ClearAll[f, g, a, b, s]; stmts…`
fn program(stmts: Vec<Expr>) -> Expr {
  let clear = call("ClearAll", ASSIGNED_SYMS.iter().map(|s| sym(s)).collect());
  Expr::Seq(std::iter::once(clear).chain(stmts).collect())
}

/// List length of a generated `Expr::List`.
fn list_len(e: &Expr) -> i64 {
  match e {
    Expr::List(xs) => xs.len() as i64,
    _ => unreachable!("list_len on a non-list"),
  }
}

/// Arithmetic over `vars` (slots like `#`, `#1`, or pattern names like `u`)
/// and small integers — the body of a generated pure function or
/// definition.
fn gen_body(rng: &mut Rng, vars: &[&'static str], depth: u32) -> Expr {
  if depth == 0 || rng.chance(1, 3) {
    return if rng.chance(3, 4) {
      sym(*rng.pick(vars))
    } else {
      int(rng.range(-5, 5))
    };
  }
  match rng.below(8) {
    0..=3 => {
      let op = *rng.pick(&[" + ", " - ", " * "]);
      infix(
        op,
        vec![
          gen_body(rng, vars, depth - 1),
          gen_body(rng, vars, depth - 1),
        ],
      )
    }
    4 => infix(
      " ^ ",
      vec![gen_body(rng, vars, depth - 1), int(rng.range(0, 3))],
    ),
    5 => call(
      "Mod",
      vec![gen_body(rng, vars, depth - 1), int(rng.range(2, 7))],
    ),
    6 => call(
      "If",
      vec![
        infix(
          *rng.pick(&[" > ", " < ", " == "]),
          vec![sym(*rng.pick(vars)), int(rng.range(-3, 3))],
        ),
        gen_body(rng, vars, depth - 1),
        gen_body(rng, vars, depth - 1),
      ],
    ),
    _ => call("Abs", vec![gen_body(rng, vars, depth - 1)]),
  }
}

/// One-argument pure function `(body(#) &)`.
fn gen_fn1(rng: &mut Rng) -> Expr {
  pure(gen_body(rng, &["#"], 2))
}

/// Two-argument pure function `(body(#1, #2) &)`.
fn gen_fn2(rng: &mut Rng) -> Expr {
  pure(gen_body(rng, &["#1", "#2"], 2))
}

/// Predicate pure function `(# > k &)`, `(EvenQ[#] &)`, …
fn gen_pred(rng: &mut Rng) -> Expr {
  match rng.below(3) {
    0 => pure(infix(
      *rng.pick(&[" > ", " < ", " >= ", " != "]),
      vec![sym("#"), int(rng.range(-3, 3))],
    )),
    1 => pure(call(
      *rng.pick(&["EvenQ", "OddQ", "Positive"]),
      vec![sym("#")],
    )),
    _ => pure(infix(
      " == ",
      vec![call("Mod", vec![sym("#"), int(rng.range(2, 4))]), int(0)],
    )),
  }
}

fn small_list(rng: &mut Rng) -> Expr {
  gen_int_list(rng, 1, 5, -5, 5)
}

fn pairs(rng: &mut Rng) -> Expr {
  let n = 1 + rng.below(3);
  Expr::List((0..n).map(|_| gen_int_list(rng, 2, 2, -5, 5)).collect())
}

/// A free boolean: `True`/`False`, a numeric comparison, or one of the
/// never-assigned symbols `p`, `q`.
fn gen_bool_leaf(rng: &mut Rng) -> Expr {
  match rng.below(4) {
    0 => sym("p"),
    1 => sym("q"),
    _ => Generator::gen_bool(rng, 1),
  }
}

/// Structural patterns for MatchQ/Cases/Count/Switch. They are printed
/// verbatim; `u` is the only pattern name, and nothing assigns it.
const PATTERNS: &[&str] = &[
  "_",
  "_Integer",
  "_Real",
  "_String",
  "_List",
  "_Rational",
  "_Plus",
  "_Times",
  "_Power",
  "_?EvenQ",
  "_?Negative",
  "_?NumberQ",
  "_Integer?Positive",
  "_Integer | _String",
  "Except[_Integer]",
  "{}",
  "{___}",
  "{_, _}",
  "{u_, u_}",
  "{__Integer}",
  "{___, 0, ___}",
  "{_Integer ..}",
  "u_ /; u > 2",
  "_?(# > 0 &)",
  "x^_",
  "_. x + _.",
];

/// Rules with a pattern left-hand side.
const PATTERN_RULES: &[&str] = &[
  "u_Integer :> u^2",
  "u_?EvenQ -> 0",
  "u_ /; u > 2 :> -u",
  "{u_, v_} :> u + v",
  "_Integer -> 0",
  "u_Integer?Negative :> Abs[u]",
  "x^u_ :> u",
  "x -> 2",
  "u_. x :> u",
  "{1 -> p, 2 -> q}",
];

/// String patterns for StringCases/StringMatchQ/StringReplace.
const STRING_PATTERNS: &[&str] = &[
  "\"a\"",
  "\"a\" | \"b\"",
  "LetterCharacter ..",
  "DigitCharacter ..",
  "\"a\" ~~ _",
  "___ ~~ \"a\" ~~ ___",
  "WordBoundary",
  "Whitespace",
  "RegularExpression[\"[a-c]+\"]",
  "StartOfString ~~ _",
  "u_ ~~ u_",
];

/// String over a small alphabet, so string patterns actually match.
fn gen_word_str(rng: &mut Rng) -> Expr {
  const CHARS: &[u8] = b"aabbc 12";
  let len = rng.below(8);
  Expr::Str(
    (0..len)
      .map(|_| CHARS[rng.below(CHARS.len() as u64) as usize] as char)
      .collect(),
  )
}

impl Generator {
  pub(super) fn gen_lang(&self, rng: &mut Rng) -> Expr {
    match rng.below(12) {
      0 => Self::gen_operator_arith(rng, 3),
      1 => gen_application(rng),
      2 => Self::gen_scoping(rng),
      3 | 4 => gen_definition(rng),
      5 => self.gen_control_flow(rng),
      6 | 7 => self.gen_rules_and_patterns(rng),
      8 => Self::gen_part_syntax(rng),
      9 => Self::gen_logic_and_strings(rng),
      _ => gen_held_syntax(rng),
    }
  }

  /// Arithmetic in operator syntax: `(a + b)`, `(2 x)`, `(-a)`, `(n!)`.
  fn gen_operator_arith(rng: &mut Rng, depth: u32) -> Expr {
    if depth == 0 || rng.chance(1, 4) {
      return match rng.below(5) {
        0 => sym(*rng.pick(&["x", "y"])),
        1 => gen_exact_num(rng),
        _ => gen_num_leaf(rng),
      };
    }
    match rng.below(9) {
      0..=3 => {
        let op = *rng.pick(&[" + ", " - ", " * ", " / "]);
        let n = 2 + rng.below(2);
        infix(
          op,
          (0..n)
            .map(|_| Self::gen_operator_arith(rng, depth - 1))
            .collect(),
        )
      }
      4 => {
        let exponent = match rng.below(3) {
          0 => Expr::Rational(1, 2),
          _ => int(rng.range(-2, 4)),
        };
        infix(
          " ^ ",
          vec![Self::gen_operator_arith(rng, depth - 1), exponent],
        )
      }
      // Juxtaposition is Times. Only a non-negative coefficient before a
      // symbol: `(3 -2)` would read as a subtraction.
      5 => infix(" ", vec![int(rng.range(0, 5)), sym(*rng.pick(&["x", "y"]))]),
      // The space keeps `- -5` from reading as a decrement, `--5`.
      6 => Expr::Unary(
        "- ",
        Box::new(Self::gen_operator_arith(rng, depth - 1)),
        "",
      ),
      7 => Expr::Unary("", Box::new(int(rng.range(0, 8))), "!"),
      _ => {
        let (lhs, rhs) = if rng.chance(1, 2) {
          (
            gen_int_list(rng, 2, 2, -4, 4),
            gen_int_list(rng, 2, 2, -4, 4),
          )
        } else {
          (gen_matrix(rng, 2), gen_int_list(rng, 2, 2, -4, 4))
        };
        infix(" . ", vec![lhs, rhs])
      }
    }
  }

  /// Module/With/Block, assignment operators and Part assignment.
  fn gen_scoping(rng: &mut Rng) -> Expr {
    let init = |rng: &mut Rng, name: &'static str| {
      infix(" = ", vec![sym(name), int(rng.range(-5, 5))])
    };
    match rng.below(6) {
      0 => {
        let vars = fixed(Expr::List(vec![init(rng, "a"), init(rng, "b")]));
        let n = 1 + rng.below(3);
        let mut stmts: Vec<Expr> = (0..n).map(|_| gen_update(rng)).collect();
        stmts.push(match rng.below(3) {
          0 => Expr::List(vec![sym("a"), sym("b")]),
          1 => gen_body(rng, &["a", "b"], 2),
          _ => infix(" ^ ", vec![sym("a"), int(2)]),
        });
        call("Module", vec![vars, Expr::Seq(stmts)])
      }
      1 => {
        let vars = fixed(Expr::List(vec![init(rng, "a")]));
        let mut body = gen_body(rng, &["a"], 2);
        if rng.chance(1, 3) {
          // With substitutes into held code too.
          body = Expr::List(vec![
            body,
            call("Hold", vec![infix(" + ", vec![sym("a"), int(1)])]),
          ]);
        }
        call("With", vec![vars, body])
      }
      2 => {
        let vars = fixed(Expr::List(vec![infix(
          " = ",
          vec![sym("x"), gen_exact_num(rng)],
        )]));
        let body = match rng.below(3) {
          0 => gen_poly(rng),
          1 => gen_trig_expr(rng, 1),
          _ => Self::gen_operator_arith(rng, 2),
        };
        call("Block", vec![vars, body])
      }
      3 => {
        // Shadowing: the inner `a` hides the outer one.
        let inner = call(
          "Module",
          vec![
            fixed(Expr::List(vec![init(rng, "a")])),
            gen_body(rng, &["a"], 1),
          ],
        );
        call(
          "Module",
          vec![
            fixed(Expr::List(vec![init(rng, "a")])),
            infix(" + ", vec![inner, sym("a")]),
          ],
        )
      }
      4 => {
        // Part assignment and in-place list modification.
        let list = gen_int_list(rng, 2, 5, -5, 5);
        let len = list_len(&list);
        let vars = fixed(Expr::List(vec![infix(" = ", vec![sym("a"), list])]));
        let idx = int(rng.range(1, len));
        let stmt = match rng.below(4) {
          0 => infix(
            " = ",
            vec![
              Expr::Index(Box::new(sym("a")), vec![idx]),
              int(rng.range(-9, 9)),
            ],
          ),
          1 => infix(
            " += ",
            vec![
              Expr::Index(Box::new(sym("a")), vec![idx]),
              int(rng.range(1, 9)),
            ],
          ),
          2 => call("AppendTo", vec![sym("a"), int(rng.range(-9, 9))]),
          _ => call("PrependTo", vec![sym("a"), int(rng.range(-9, 9))]),
        };
        call("Module", vec![vars, Expr::Seq(vec![stmt, sym("a")])])
      }
      _ => {
        // Function with named parameters, and nested pure functions.
        let f = call(
          "Function",
          vec![
            Expr::List(vec![sym("u"), sym("v")]),
            gen_body(rng, &["u", "v"], 2),
          ],
        );
        apply(f, vec![int(rng.range(-5, 5)), int(rng.range(-5, 5))])
      }
    }
  }

  fn gen_control_flow(&self, rng: &mut Rng) -> Expr {
    let n = rng.range(1, 6);
    let iter = |var: &'static str, n: i64| Expr::List(vec![sym(var), int(n)]);
    match rng.below(11) {
      0 => {
        let mut args = vec![
          gen_bool_leaf(rng),
          Self::gen_num(rng, 1),
          Self::gen_num(rng, 1),
        ];
        // Two-argument If returns Null when the test fails; four-argument
        // If has a branch for a test that is neither True nor False.
        match rng.below(3) {
          0 => {
            args.pop();
          }
          1 => args.push(int(0)),
          _ => {}
        }
        call("If", args)
      }
      1 => call(
        "Which",
        vec![
          Self::gen_bool(rng, 1),
          Self::gen_num(rng, 1),
          Self::gen_bool(rng, 1),
          Self::gen_num(rng, 1),
          sym("True"),
          int(0),
        ],
      ),
      2 => {
        let mut args = vec![self.gen_any(rng, 1)];
        for _ in 0..2 {
          args.push(sym(*rng.pick(PATTERNS)));
          args.push(int(rng.range(-9, 9)));
        }
        args.push(sym("_"));
        args.push(Expr::Str("other".into()));
        call("Switch", args)
      }
      3 => call(
        "Module",
        vec![
          fixed(Expr::List(vec![infix(" = ", vec![sym("s"), int(0)])])),
          Expr::Seq(vec![
            call(
              "Do",
              vec![
                infix(" += ", vec![sym("s"), gen_body(rng, &["i"], 2)]),
                iter("i", n),
              ],
            ),
            sym("s"),
          ]),
        ],
      ),
      4 => call(
        "Table",
        vec![
          gen_body(rng, &["i", "j"], 2),
          iter("i", rng.range(1, 3)),
          iter("j", rng.range(1, 3)),
        ],
      ),
      5 => call(
        "Catch",
        vec![call(
          "Do",
          vec![
            call(
              "If",
              vec![
                infix(
                  " > ",
                  vec![gen_body(rng, &["i"], 1), int(rng.range(0, 9))],
                ),
                call("Throw", vec![sym("i")]),
              ],
            ),
            Expr::List(vec![sym("i"), int(10)]),
          ],
        )],
      ),
      6 => call(
        "Reap",
        vec![call(
          "Do",
          vec![
            call(
              "If",
              vec![
                apply(gen_pred(rng), vec![sym("i")]),
                call("Sow", vec![sym("i")]),
              ],
            ),
            iter("i", n),
          ],
        )],
      ),
      7 => call("Scan", vec![sym("Print"), small_list(rng)]),
      8 => call(
        "Module",
        vec![
          fixed(Expr::List(vec![infix(
            " = ",
            vec![sym("s"), Expr::List(vec![])],
          )])),
          Expr::Seq(vec![
            call(
              "Do",
              vec![
                call("AppendTo", vec![sym("s"), gen_body(rng, &["i"], 2)]),
                iter("i", n),
              ],
            ),
            sym("s"),
          ]),
        ],
      ),
      9 => {
        // Break and Continue inside Do.
        let jump = if rng.chance(1, 2) {
          "Break"
        } else {
          "Continue"
        };
        call(
          "Module",
          vec![
            fixed(Expr::List(vec![infix(" = ", vec![sym("s"), int(0)])])),
            Expr::Seq(vec![
              call(
                "Do",
                vec![
                  Expr::Seq(vec![
                    call(
                      "If",
                      vec![
                        apply(gen_pred(rng), vec![sym("i")]),
                        call(jump, vec![]),
                      ],
                    ),
                    infix(" += ", vec![sym("s"), sym("i")]),
                  ]),
                  iter("i", 8),
                ],
              ),
              sym("s"),
            ]),
          ],
        )
      }
      _ => {
        // Return from inside a definition.
        let lhs = call("f", vec![sym("u_")]);
        let rhs = Expr::Seq(vec![
          call(
            "If",
            vec![
              infix(" > ", vec![sym("u"), int(rng.range(-2, 2))]),
              call("Return", vec![gen_body(rng, &["u"], 1)]),
            ],
          ),
          gen_body(rng, &["u"], 1),
        ]);
        program(vec![
          infix(" := ", vec![lhs, rhs]),
          infix(" /@ ", vec![sym("f"), small_list(rng)]),
        ])
      }
    }
  }

  fn gen_rules_and_patterns(&self, rng: &mut Rng) -> Expr {
    let list = small_list(rng);
    match rng.below(9) {
      0 => infix(" /. ", vec![list, sym(*rng.pick(PATTERN_RULES))]),
      1 => {
        let subject = match rng.below(3) {
          0 => gen_poly(rng),
          1 => gen_poly_prod(rng),
          _ => pairs(rng),
        };
        infix(" /. ", vec![subject, sym(*rng.pick(PATTERN_RULES))])
      }
      2 => {
        // Bubble sort by ReplaceRepeated. The rule is one atomic `Sym`,
        // so shrinking cannot drop its condition — without it the rule
        // would swap forever.
        let rule = sym("{u___, v_, w_, z___} /; v > w :> {u, w, v, z}");
        infix(" //. ", vec![list, rule])
      }
      3 => {
        let head = *rng.pick(&["Cases", "DeleteCases", "Count", "Position"]);
        call(
          head,
          vec![Self::gen_mixed_list(rng), sym(*rng.pick(PATTERNS))],
        )
      }
      4 => call(
        "Cases",
        vec![Self::gen_mixed_list(rng), sym(*rng.pick(PATTERN_RULES))],
      ),
      5 => {
        let subject = match rng.below(4) {
          0 => self.gen_any(rng, 2),
          1 => gen_poly(rng),
          2 => Self::gen_mixed_list(rng),
          _ => list,
        };
        call("MatchQ", vec![subject, sym(*rng.pick(PATTERNS))])
      }
      6 => call(
        "ReplaceList",
        vec![
          list,
          sym(*rng.pick(&[
            "{u___, v__} :> {{u}, {v}}",
            "{___, u_, ___} :> u",
            "{u__, v__} /; Length[{u}] > 1 :> {v}",
          ])),
        ],
      ),
      7 => {
        let pat = sym(*rng.pick(STRING_PATTERNS));
        let s = gen_word_str(rng);
        match rng.below(4) {
          0 => call("StringCases", vec![s, pat]),
          1 => call("StringMatchQ", vec![s, pat]),
          2 => call("StringCount", vec![s, pat]),
          _ => call(
            "StringReplace",
            vec![s, infix(" -> ", vec![pat, Expr::Str("X".into())])],
          ),
        }
      }
      _ => call(
        "Replace",
        vec![
          pairs(rng),
          sym(*rng.pick(PATTERN_RULES)),
          Expr::List(vec![int(rng.range(0, 2))]),
        ],
      ),
    }
  }

  /// List mixing numbers, strings, sublists and symbolic terms — a
  /// subject on which most patterns match something.
  fn gen_mixed_list(rng: &mut Rng) -> Expr {
    let n = rng.below(6);
    Expr::List(
      (0..n)
        .map(|_| match rng.below(6) {
          0 => gen_str(rng),
          1 => gen_int_list(rng, 0, 3, -3, 3),
          2 => gen_low_poly(rng),
          _ => gen_num_leaf(rng),
        })
        .collect(),
    )
  }

  fn gen_part_syntax(rng: &mut Rng) -> Expr {
    let list = gen_int_list(rng, 1, 6, -9, 9);
    let len = list_len(&list);
    let index = |list: Expr, idx: Vec<Expr>| Expr::Index(Box::new(list), idx);
    match rng.below(7) {
      0 => {
        let k = rng.range(1, len);
        index(list, vec![int(if rng.chance(1, 3) { -k } else { k })])
      }
      1 => {
        let i = rng.range(1, len);
        let j = rng.range(i, len);
        let mut span = vec![int(i), int(j)];
        if rng.chance(1, 3) {
          span.push(int(rng.range(1, 2)));
        }
        index(list, vec![infix(" ;; ", span)])
      }
      2 => {
        let m = gen_matrix(rng, 3);
        match rng.below(3) {
          0 => index(m, vec![sym("All"), int(rng.range(1, 3))]),
          1 => index(m, vec![int(rng.range(1, 3)), int(rng.range(1, 3))]),
          _ => index(
            m,
            vec![Expr::List(vec![int(rng.range(1, 3)), int(rng.range(1, 3))])],
          ),
        }
      }
      3 => {
        let assoc = Expr::AssocLit(
          ["a", "b", "c"]
            .iter()
            .map(|k| {
              infix(" -> ", vec![Expr::Str((*k).into()), int(rng.range(-9, 9))])
            })
            .collect(),
        );
        let key = Expr::Str((*rng.pick(&["a", "b", "c", "z"])).into());
        match rng.below(4) {
          0 => apply(assoc, vec![key]),
          1 => index(assoc, vec![call("Key", vec![key])]),
          2 => index(assoc, vec![int(rng.range(1, 3))]),
          _ => call("Lookup", vec![assoc, key, int(0)]),
        }
      }
      // Part of a canonicalized expression depends on the canonical order.
      4 => index(gen_poly(rng), vec![int(1)]),
      5 => index(
        Expr::List(vec![
          Expr::List(vec![int(1), Expr::List(vec![int(2), int(3)])]),
          int(4),
        ]),
        vec![int(1), int(2), int(rng.range(1, 2))],
      ),
      _ => {
        // Part assignment through a symbol.
        program(vec![
          infix(" = ", vec![sym("a"), list]),
          infix(
            " = ",
            vec![
              index(sym("a"), vec![int(rng.range(1, len))]),
              int(rng.range(-9, 9)),
            ],
          ),
          sym("a"),
        ])
      }
    }
  }

  fn gen_logic_and_strings(rng: &mut Rng) -> Expr {
    match rng.below(7) {
      0 | 1 => {
        let op = *rng.pick(&[" && ", " || "]);
        let n = 2 + rng.below(2);
        infix(op, (0..n).map(|_| gen_bool_leaf(rng)).collect())
      }
      2 => Expr::Unary("!", Box::new(gen_bool_leaf(rng)), ""),
      3 => {
        // Comparison chains: `a < b < c`, `a != b != c` (pairwise
        // distinct), `a === b === c`.
        let op = *rng.pick(&[" < ", " <= ", " == ", " != ", " === ", " =!= "]);
        let n = 2 + rng.below(2);
        infix(op, (0..n).map(|_| Self::gen_num(rng, 1)).collect())
      }
      4 => call(
        *rng.pick(&["Xor", "Implies", "Nand", "Nor", "Equivalent"]),
        vec![gen_bool_leaf(rng), gen_bool_leaf(rng)],
      ),
      5 => {
        let n = 2 + rng.below(2);
        infix(" <> ", (0..n).map(|_| gen_word_str(rng)).collect())
      }
      _ => call(
        *rng.pick(&["TrueQ", "Boole", "BooleanQ"]),
        vec![gen_bool_leaf(rng)],
      ),
    }
  }
}

/// Binary operators for held syntax trees: every precedence level from
/// `;;` and `=` up to `^` and `@`, so printing them back has to parenthesize
/// correctly. `.` is generated separately: next to an integer `1 . 2` would
/// read as a decimal point.
const HELD_INFIX: &[&str] = &[
  " + ",
  " - ",
  " * ",
  " / ",
  " ^ ",
  " == ",
  " != ",
  " < ",
  " >= ",
  " === ",
  " && ",
  " || ",
  " -> ",
  " :> ",
  " /. ",
  " //. ",
  " = ",
  " := ",
  " <> ",
  " ;; ",
  " @ ",
  " /@ ",
  " @@ ",
  " // ",
  " ~~ ",
  " | ",
  " ** ",
  " @* ",
  " /* ",
  " /; ",
  " ^= ",
  " += ",
  " @@@ ",
  " <-> ",
  " ^:= ",
  " //@ ",
  " -= ",
  " \\[Element] ",
  " \\[Implies] ",
  " >= ",
  " <= ",
  " =!= ",
];

/// A random operator-syntax tree, wrapped so that nothing evaluates:
/// `Hold[…]`, `HoldForm[…]` or `FullForm[Hold[…]]`. The output is the
/// printer's rendering of what the parser built, so a mismatch is a
/// precedence or parenthesization difference.
pub(super) fn gen_held_syntax(rng: &mut Rng) -> Expr {
  let tree = gen_held_tree(rng, 3);
  match rng.below(3) {
    0 => call("Hold", vec![tree]),
    1 => call("HoldForm", vec![tree]),
    _ => call("FullForm", vec![call("Hold", vec![tree])]),
  }
}

fn gen_held_tree(rng: &mut Rng, depth: u32) -> Expr {
  const LEAVES: &[&str] =
    &["a", "b", "c", "f", "x", "x_", "_Integer", "#", "#2"];
  if depth == 0 || rng.chance(1, 4) {
    return match rng.below(6) {
      0 | 1 | 2 => sym(*rng.pick(LEAVES)),
      3 => int(rng.range(-3, 9)),
      4 => Expr::Str((*rng.pick(&["s", "ab", ""])).into()),
      _ => Expr::Rational(rng.range(-3, 3), rng.range(2, 5)),
    };
  }
  match rng.below(12) {
    0..=5 => {
      let op = *rng.pick(HELD_INFIX);
      infix(
        op,
        vec![gen_held_tree(rng, depth - 1), gen_held_tree(rng, depth - 1)],
      )
    }
    6 => {
      let n = 3;
      let op = *rng.pick(&[" + ", " * ", " && ", " == ", " < "]);
      infix(op, (0..n).map(|_| gen_held_tree(rng, depth - 1)).collect())
    }
    7 => infix(
      " . ",
      vec![
        sym(*rng.pick(&["a", "b", "c"])),
        sym(*rng.pick(&["a", "b", "c"])),
      ],
    ),
    8 => {
      let (pre, post) = *rng.pick(&[
        ("- ", ""),
        ("!", ""),
        ("", "!"),
        ("", " &"),
        ("", "!!"),
        ("", "++"),
        ("", "--"),
        ("++", ""),
        ("--", ""),
        ("", "'"),
      ]);
      Expr::Unary(pre, Box::new(gen_held_tree(rng, depth - 1)), post)
    }
    9 => call(
      *rng.pick(&["f", "g", "Plus", "Times", "Power", "List", "Rule"]),
      (0..1 + rng.below(2))
        .map(|_| gen_held_tree(rng, depth - 1))
        .collect(),
    ),
    10 => Expr::Index(
      Box::new(gen_held_tree(rng, depth - 1)),
      vec![int(rng.range(1, 3))],
    ),
    _ => Expr::List(
      (0..rng.below(3))
        .map(|_| gen_held_tree(rng, depth - 1))
        .collect(),
    ),
  }
}

/// `a += k`, `b *= k`, `a++`, `--b`, `a = a + b`, …
fn gen_update(rng: &mut Rng) -> Expr {
  let var = sym(*rng.pick(&["a", "b"]));
  match rng.below(6) {
    0 => infix(
      *rng.pick(&[" += ", " -= ", " *= "]),
      vec![var, int(rng.range(-4, 4))],
    ),
    1 => Expr::Unary("", Box::new(var), *rng.pick(&["++", "--"])),
    2 => Expr::Unary(*rng.pick(&["++", "--"]), Box::new(var), ""),
    3 => infix(" = ", vec![var, gen_body(rng, &["a", "b"], 2)]),
    4 => infix(" /= ", vec![var, int(*rng.pick(&[1, 2, -3]))]),
    _ => infix(" = ", vec![var, Expr::List(vec![sym("a"), sym("b")])]),
  }
}

/// Operator syntax for function application and pure functions.
fn gen_application(rng: &mut Rng) -> Expr {
  match rng.below(14) {
    0 => infix(" /@ ", vec![gen_fn1(rng), small_list(rng)]),
    1 => infix(" @ ", vec![gen_fn1(rng), int(rng.range(-5, 5))]),
    2 => infix(" // ", vec![int(rng.range(-5, 5)), gen_fn1(rng)]),
    3 => infix(" @@ ", vec![sym(*rng.pick(OP_FNS)), small_list(rng)]),
    4 => infix(" @@@ ", vec![gen_fn2(rng), pairs(rng)]),
    5 => apply(gen_fn1(rng), vec![int(rng.range(-5, 5))]),
    6 => {
      let op = *rng.pick(&[" @* ", " /* "]);
      apply(
        infix(op, vec![gen_fn1(rng), gen_fn1(rng)]),
        vec![int(rng.range(-5, 5))],
      )
    }
    7 => apply(
      call("Function", vec![sym("u"), gen_body(rng, &["u"], 2)]),
      vec![int(rng.range(-5, 5))],
    ),
    8 => call(
      *rng.pick(&["Fold", "FoldList"]),
      vec![gen_fn2(rng), int(rng.range(-3, 3)), small_list(rng)],
    ),
    9 => {
      let n = rng.range(1, 4) as u64;
      call(
        "MapThread",
        vec![
          gen_fn2(rng),
          Expr::List(vec![
            gen_int_list(rng, n, n, -5, 5),
            gen_int_list(rng, n, n, -5, 5),
          ]),
        ],
      )
    }
    10 => call(
      "Array",
      vec![
        gen_fn2(rng),
        Expr::List(vec![int(rng.range(1, 3)), int(rng.range(1, 3))]),
      ],
    ),
    // Operator forms: `Map[f][list]`, `Select[pred][list]`, …
    11 => {
      let (head, arg) = match rng.below(4) {
        0 => ("Map", gen_fn1(rng)),
        1 => ("Select", gen_pred(rng)),
        2 => ("SortBy", gen_fn1(rng)),
        _ => ("Apply", sym(*rng.pick(OP_FNS))),
      };
      apply(call(head, vec![arg]), vec![small_list(rng)])
    }
    // `##` and `##2`: slot sequences.
    12 => {
      let f = pure(match rng.below(3) {
        0 => Expr::List(vec![sym("##")]),
        1 => call("Plus", vec![sym("##")]),
        _ => Expr::List(vec![sym("##2")]),
      });
      let n = rng.below(4);
      apply(f, (0..n).map(|_| int(rng.range(-5, 5))).collect())
    }
    _ => infix(
      " //@ ",
      vec![
        gen_fn1(rng),
        Expr::List(vec![int(1), Expr::List(vec![int(2), int(3)])]),
      ],
    ),
  }
}

/// Pattern-based definitions of `f` and `g`, each program starting from
/// cleared symbols.
fn gen_definition(rng: &mut Rng) -> Expr {
  let f_of = |pat: &'static str| call("f", vec![sym(pat)]);
  let define = |lhs: Expr, rhs: Expr| infix(" := ", vec![lhs, rhs]);
  let map_f = |rng: &mut Rng| {
    infix(" /@ ", vec![sym("f"), gen_int_list(rng, 1, 6, -4, 4)])
  };
  match rng.below(11) {
    0 => program(vec![
      define(f_of("u_"), gen_body(rng, &["u"], 3)),
      map_f(rng),
    ]),
    1 => {
      // Conditional clauses, on the pattern or on the right-hand side.
      let k = int(rng.range(-2, 2));
      let first = if rng.chance(1, 2) {
        define(
          call(
            "f",
            vec![infix(
              " /; ",
              vec![sym("u_"), infix(" > ", vec![sym("u"), k])],
            )],
          ),
          gen_body(rng, &["u"], 2),
        )
      } else {
        define(
          f_of("u_"),
          infix(
            " /; ",
            vec![gen_body(rng, &["u"], 2), infix(" < ", vec![sym("u"), k])],
          ),
        )
      };
      program(vec![
        first,
        define(
          f_of(*rng.pick(&["u_?EvenQ", "u_Integer", "u_"])),
          gen_body(rng, &["u"], 2),
        ),
        map_f(rng),
      ])
    }
    2 => program(vec![
      define(f_of("u_Integer"), gen_body(rng, &["u"], 2)),
      define(f_of("u_Real"), call("Round", vec![sym("u")])),
      define(f_of("u_String"), call("StringLength", vec![sym("u")])),
      define(f_of("_"), Expr::Str("other".into())),
      infix(
        " /@ ",
        vec![
          sym("f"),
          Expr::List(vec![
            int(rng.range(-5, 5)),
            Expr::RealTenths(rng.range(-50, 50)),
            gen_str(rng),
            Expr::Rational(1, 2),
            sym("p"),
          ]),
        ],
      ),
    ]),
    3 => {
      // Recursion with a base case. The step is fixed so shrinking cannot
      // turn `n - 1` into `n`.
      let (base, step, max_n): (&[(i64, i64)], &'static str, i64) =
        match rng.below(4) {
          0 => (&[(0, 1)], "n f[n - 1]", 8),
          1 => (&[(0, 0)], "f[n - 1] + n", 10),
          2 => (&[(0, 1)], "2 f[n - 1] + 1", 10),
          // Memoized: `f[n] = …` stores each value as it is computed.
          _ => (&[(0, 0), (1, 1)], "f[n] = f[n - 1] + f[n - 2]", 20),
        };
      let mut stmts: Vec<Expr> = base
        .iter()
        .map(|&(k, v)| {
          fixed(infix(" = ", vec![call("f", vec![int(k)]), int(v)]))
        })
        .collect();
      stmts.push(fixed(define(f_of("n_Integer?Positive"), sym(step))));
      stmts.push(match rng.below(3) {
        0 => call("f", vec![int(rng.range(0, max_n))]),
        1 => infix(
          " /@ ",
          vec![sym("f"), call("Range", vec![int(0), int(rng.range(0, 6))])],
        ),
        _ => call("Length", vec![call("DownValues", vec![sym("f")])]),
      });
      program(stmts)
    }
    4 => program(vec![
      define(
        call("f", vec![sym("u_"), sym("v_ : 2")]),
        gen_body(rng, &["u", "v"], 2),
      ),
      Expr::List(vec![
        call("f", vec![int(rng.range(-4, 4))]),
        call("f", vec![int(rng.range(-4, 4)), int(rng.range(-4, 4))]),
      ]),
    ]),
    5 => {
      // Sequence patterns.
      let (pat, rhs): (&[&'static str], Expr) = match rng.below(4) {
        0 => (&["u__"], Expr::List(vec![sym("u")])),
        1 => (
          &["u_", "v___"],
          Expr::List(vec![
            sym("u"),
            call("Length", vec![Expr::List(vec![sym("v")])]),
          ]),
        ),
        2 => (&["u__Integer"], call("Plus", vec![sym("u")])),
        _ => (
          &["u___", "v_"],
          Expr::List(vec![sym("v"), Expr::List(vec![sym("u")])]),
        ),
      };
      let calls = (0..4)
        .map(|n| call("f", (0..n).map(|_| int(rng.range(-4, 4))).collect()))
        .collect();
      program(vec![
        define(call("f", pat.iter().map(|p| sym(p)).collect()), rhs),
        Expr::List(calls),
      ])
    }
    6 => program(vec![
      define(
        call(
          "f",
          vec![sym(*rng.pick(&["{u_, v_}", "{u_, v__}", "{u_, u_}"]))],
        ),
        gen_body(rng, &["u"], 2),
      ),
      infix(" /@ ", vec![sym("f"), pairs(rng)]),
    ]),
    // `g` calls `f`; the definition order does not matter.
    7 => program(vec![
      define(
        call("g", vec![sym("u_")]),
        infix(" + ", vec![call("f", vec![sym("u")]), int(1)]),
      ),
      define(f_of("u_"), gen_body(rng, &["u"], 2)),
      infix(" /@ ", vec![sym("g"), small_list(rng)]),
    ]),
    // Set evaluates the right-hand side once, SetDelayed on every call.
    8 => program(vec![
      infix(" = ", vec![sym("a"), int(rng.range(-5, 5))]),
      infix(
        " = ",
        vec![f_of("u_"), infix(" + ", vec![sym("u"), sym("a")])],
      ),
      define(
        call("g", vec![sym("u_")]),
        infix(" + ", vec![sym("u"), sym("a")]),
      ),
      infix(" = ", vec![sym("a"), int(rng.range(-5, 5))]),
      Expr::List(vec![call("f", vec![int(1)]), call("g", vec![int(1)])]),
    ]),
    9 => program(vec![
      call(
        "TagSetDelayed",
        vec![sym("g"), call("f", vec![sym("g")]), int(rng.range(-5, 5))],
      ),
      Expr::List(vec![call("f", vec![sym("g")]), call("f", vec![int(1)])]),
    ]),
    _ => {
      let attr =
        *rng.pick(&["Listable", "Orderless", "Flat", "HoldAll", "OneIdentity"]);
      let n = 1 + rng.below(3);
      let args = (0..n)
        .map(|_| match rng.below(3) {
          0 => small_list(rng),
          1 => infix(" + ", vec![int(rng.range(1, 3)), int(rng.range(1, 3))]),
          _ => int(rng.range(-5, 5)),
        })
        .collect();
      program(vec![
        call("SetAttributes", vec![sym("f"), sym(attr)]),
        call("f", args),
      ])
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn generator() -> Generator {
    let implemented: HashSet<String> =
      FN_SPECS.iter().map(|s| s.name.to_string()).collect();
    Generator::new(&implemented, 3)
  }

  #[test]
  fn lang_programs_parse_and_stay_on_one_line() {
    let generator = generator();
    for seed in 0..3000u64 {
      let code = generator.gen_lang(&mut Rng::new(seed)).to_code();
      assert!(!code.contains('\n'), "seed {seed} spans lines: {code}");
      assert!(
        woxi::parse(&code).is_ok(),
        "seed {seed} generated unparsable code: {code}"
      );
    }
  }

  /// The batch prefilter evaluates many cases in one session, so a program
  /// that leaves a value on a symbol other cases use freely would change
  /// their output. Run programs back to back in one session, as a batch
  /// does, and check that the free symbols are still free afterwards.
  #[test]
  fn lang_programs_leave_free_symbols_unassigned() {
    let generator = generator();
    woxi::set_quiet_print(true);
    let free = "{x, y, p, q, u, v, w, z, i, j, n}";
    for seed in 0..300u64 {
      let code = generator.gen_lang(&mut Rng::new(seed)).to_code();
      let _ = woxi::interpret(&code);
      assert_eq!(
        woxi::interpret(free).unwrap(),
        free,
        "seed {seed} assigned a free symbol: {code}"
      );
    }
  }

  #[test]
  fn fixed_parts_are_not_shrunk() {
    let step = fixed(infix(
      " := ",
      vec![
        call("f", vec![sym("n_")]),
        infix(" - ", vec![sym("n"), int(1)]),
      ],
    ));
    assert_eq!(shrink_candidates(&step), Vec::<Expr>::new());
    let prog = program(vec![step.clone(), call("f", vec![int(3)])]);
    for cand in shrink_candidates(&prog) {
      let code = cand.to_code();
      assert!(
        !code.contains(":=") || code.contains("(f[n_] := (n - 1))"),
        "shrinking changed a fixed part: {code}"
      );
    }
  }
}
