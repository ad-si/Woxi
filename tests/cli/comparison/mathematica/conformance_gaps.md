---
icon: lucide/diff
---

# Wolfram Language conformance gaps

Known places where Woxi's output differs from `wolframscript`'s.
Every entry was found by diffing the two engines and verified against the
Wolfram Language; none of them is fixed yet.

Where an entry says **not reproducible**, the difference comes from a choice
Wolfram makes internally (an algorithm's tie-break, a library's rounding, a
front-end representation) that cannot be recovered from its output — those are
recorded so they are not investigated twice.


## Output form, ordering and rendering

Woxi and Wolfram usually agree on the *value* and disagree on how it is
printed. This is by far the largest group.

### InputForm does not normalize function spellings to operator syntax

`ToString[…, InputForm]` (and plain output) prints the function-call spelling
where WL prints the operator, so the two spellings of the same expression
render differently:

```sh
# In a result (an unevaluated ReplaceAll is what a failed replacement returns):
wolframscript -code 'ToString[ReplaceAll[x, 5], InputForm]'         # x /. 5
woxi eval 'ToString[ReplaceAll[x, 5], InputForm]'                   # ReplaceAll[x, 5]

# In a held expression:
wolframscript -code 'ToString[Unevaluated[ReplaceRepeated[x, 5]], InputForm]'   # x //. 5
woxi eval 'ToString[Unevaluated[ReplaceRepeated[x, 5]], InputForm]'             # ReplaceRepeated[x, 5]
```

The affected heads are `ReplaceAll` (`/.`) and `ReplaceRepeated` (`//.`).
`Map`, `Apply`, `MapApply`, `MapAll` and `Part` print with their operators.

### `InputForm` brackets more than the printed form does

wolframscript's `ToString[…, InputForm]` brackets things its top-level output
leaves bare, so the two disagree only when the InputForm *string* is compared:

```sh
wolframscript -code 'Hold[{1, 2}?f]'                          # Hold[{1, 2}?f]
wolframscript -code 'ToString[(Hold[{1, 2}?f]), InputForm]'   # Hold[({1, 2})?f]
```

Woxi prints `Hold[{1, 2}?f]` in both. A `Graph`'s options are the same story:
InputForm wraps them in a `List` (and reorders them into its own canonical
order), the printed form does not.

### A nested `Plus`/`Times` operand is not bracketed

`Times` and `Plus` only flatten once they evaluate, so a *held* expression can
carry one inside another. wolframscript brackets the inner one so the printed
form re-reads as the same tree; Woxi prints the flat spelling, which re-reads
as the flattened expression:

```sh
wolframscript -code 'ToString[Hold[Times[Times[a, b], c]], InputForm]' # Hold[(a*b)*c]
woxi eval 'ToString[Hold[Times[Times[a, b], c]], InputForm]'          # Hold[a*b*c]
wolframscript -code 'ToString[Hold[Plus[Plus[a, b], c]], InputForm]'  # Hold[(a + b) + c]
woxi eval 'ToString[Hold[Plus[Plus[a, b], c]], InputForm]'            # Hold[a + b + c]
```

The values agree — the two trees are equal once evaluated — and Woxi's reader
never builds a nested one, so this only shows on an expression written in
function form under a hold, or produced by a head replacement on a held
division (`Hold[a/(b c) d] /. Power -> pp`).

### Unary minus inside a pure function round-trips as a subtraction

```sh
wolframscript -code 'ToString[Unevaluated[-# &], InputForm]'  # -#1 &
woxi eval 'ToString[Unevaluated[-# &], InputForm]'            # 0 - #1 &
```

Same family as the known `Not[#2]` vs `!#2` rendering gap.

### Association inside Unevaluated echoes with the wrong spelling

```sh
wolframscript -code 'ToString[Unevaluated[<|1 -> a|>], InputForm]'  # Association[1 -> a]
woxi eval 'ToString[Unevaluated[<|1 -> a|>], InputForm]'            # <|1 -> a|>
```

Cosmetic, and confined to held echoes: `ToString[<|1 -> a|>, InputForm]` agrees
on both engines.

### A unary minus inside `Backslash` is not bracketed

`Backslash` (precedence 460) binds looser than unary minus (480), yet
wolframscript brackets a negated operand anyway; Woxi prints it bare. Both
re-parse to the same tree.

```sh
wolframscript -code 'ToString[Hold[Backslash[-a, b]], InputForm]'   # Hold[(-a) ∖ b]
woxi eval 'ToString[Hold[Backslash[-a, b]], InputForm]'             # Hold[-a ∖ b]
```

### `-a/b` groups differently, and a held reciprocal is not canonicalized

Woxi groups `-a/b` as `-(a/b)` where wolframscript groups it as `(-a)/b` — a
parse-precedence difference rather than a rendering one. `Hold[-1/x]` prints
`-1/x` where wolframscript canonicalises to `-x^(-1)`. `- -x` and `!!a` are
parse errors in Woxi (the grammar forbids a second prefix operator).

### `ToString` renders 1D where wolframscript renders 2D ASCII art

```sh
wolframscript -code 'ToString[a/b]'   # a over a bar over b
woxi eval 'ToString[a/b]'             # a/b

wolframscript -code 'ToString[x^2]'   # a raised 2 above x
woxi eval 'ToString[x^2]'             # x^2
```

The bare top-level echo is 1D in **both** engines — only `ToString` (and an
explicit `OutputForm` wrapper) triggers the 2D layout. `Sqrt[x]` and matrices
already agree. Switching the default `ToString` path to Woxi's existing 2D
renderer would move a great many snapshots at once, so fractions and powers are
still 1D.

Consequences elsewhere: `ToString[-48/2033]`, `ToString[-10/3]`,
`ToString[1.5*^10]` and 2D rationals inside most built-in message text
all print flat (see "Message repetition and 2D layouts").

### `NumberForm` does not switch to scientific notation

```sh
wolframscript -code 'ToString[NumberForm[123456789.]]'  # 1.23457 × 10^8
woxi eval 'ToString[NumberForm[123456789.]]'            # 123457000.
```

Also `NumberForm[1000000.]` (WL `1. × 10^6`) and `NumberForm[1.5*10^-8]`
(WL `1.5 × 10^-8`). In-range reals — roughly `10^-5 ≤ |x| < 10^6` — agree.

### `NumberForm[x, {n, f}]` with an `ExponentFunction`

`NumberForm[x, {n, f}]` now switches to scientific notation outside
`10^-5 ≤ |x| < 10^6` (`1.000 × 10^-7`), but the 3-argument form with an
`ExponentFunction` in Studio ignores the function (`1.×10^-7`) and `ToString`
prints `0.1 × 10^-6` without padding.

### `NumberForm`/`ScientificForm` round half-to-even, wolframscript rounds half-up

```sh
wolframscript -code 'ToString[ScientificForm[12345., 4]]'  # 1.235 × 10^4
woxi eval 'ToString[ScientificForm[12345., 4]]'            # 1.234 × 10^4
```

Only at an exact-half significant-figure boundary; anything with a non-zero
tail agrees. A latent edge in the same area: a decimal that rounds *up* across
`10^6` (`999999.6`) stays `1000000.` in the box path instead of switching to
`1.×10^6`.

### `NumberFormat` and `BaseForm` of a real are unimplemented

`NumberForm[…, NumberFormat -> f]` stays unevaluated where WL renders
`1234.5e`, and `BaseForm[0.5, 2]` stays unevaluated where WL gives `0.1` over a
subscript 2.

### `TextString` falls back to `ToString` for symbolic expressions

WL writes a `FullForm`-like linear form with numericized leaves — `x + y` →
`"Plus[x, y]"`, `Sqrt[x]` → `"Power[x, 0.5]"`, `a/b` →
`"Times[a, Power[b, -1]]"`, but `f[1/2]` → `"f[0.5]"`. Woxi keeps the ordinary
2D `ToString` form. (`TextString[100!]` is `"∞"` in wolframscript — a formatter
overflow that is deliberately not reproduced.)

### TeXForm orders the terms of a sum by monomial degree

```sh
wolframscript -code 'ToString[TeXForm[1 + x + x^2]]'      # x^2+x+1
woxi eval 'ToString[TeXForm[1 + x + x^2]]'                # x+x^2+1

wolframscript -code 'ToString[TeXForm[a x^2 + b x + c]]'  # a x^2+b x+c
woxi eval 'ToString[TeXForm[a x^2 + b x + c]]'            # c+b x+a x^2

wolframscript -code 'ToString[TeXForm[Sin[x] + Cos[y]]]'  # \sin (x)+\cos (y)
woxi eval 'ToString[TeXForm[Sin[x] + Cos[y]]]'            # \cos (y)+\sin (x)
```

Woxi keeps the canonical order and only moves numbers (and complex atoms) to
the end; Wolfram sorts by the monomial, highest degree first. Neither
"reverse the canonical order" nor "stable sort by total degree" reproduces
all the samples — `3 + a + b` stays `a+b+3` (not reversed) while
`Cos[y] + Sin[x]` does reverse, and `x^3 + x^2 y^2` keeps `x^3` first even
though its total degree is lower — so this needs WL's actual lexicographic
monomial comparison. Everything else in a 211-expression TeXForm sweep
agrees.

`TraditionalForm` boxes share the order, and a Demonstration that keys a
lookup table on `ToString[f, TraditionalForm]` (NewtonsMethod) then misses
the saved control value:

```sh
wolframscript -code 'ToString[Sin[x] + 3 Cos[x], TraditionalForm]'  # sin(x) + 3 cos(x) boxes
woxi eval 'ToString[Sin[x] + 3 Cos[x], TraditionalForm]'            # 3 cos(x) + sin(x) boxes
```

More samples of the order: `a + b + Sin[a]` is `a+sin(a)+b`, `3 + a + Cos[b]`
is `a+cos(b)+3`, and `Cos[x] + Sin[x] + Tan[x]` is `sin(x)+cos(x)+tan(x)`.

### TeXForm writes a rational exponent's quotient with a `\left.` bar

```sh
wolframscript -code 'ToString[TeXForm[x^(1/(2 y))]]'  # x^{\left.\frac{1}{2}\right/y}
woxi eval 'ToString[TeXForm[x^(1/(2 y))]]'            # x^{\frac{1}{2 y}}
```

(Whether a rational coefficient folds into its factor's fraction —
`\frac{f(x)}{2}` against `\frac{1}{2} \log \left(x^2+1\right)` — now follows
a box-count estimate of the factor that reproduces every probed case; see
`tex_box_weight` in `src/functions/string_ast.rs`.)

### TeXForm orders a product's factors by Woxi's own `Times` order

```sh
wolframscript -code 'ToString[TeXForm[Sin[x]^2 Star[a, b]]]'  # \sin ^2(x) (a*b)
woxi eval 'ToString[TeXForm[Sin[x]^2 Star[a, b]]]'            # (a*b) \sin ^2(x)

wolframscript -code 'ToString[TeXForm[(1 + x) y (a + b)]]'    # (x+1) y (a+b)
woxi eval 'ToString[TeXForm[(1 + x) y (a + b)]]'              # y (a+b) (x+1)

wolframscript -code 'ToString[x^2 Sqrt[a^2 - x^2], InputForm]'  # x^2*Sqrt[a^2 - x^2]
woxi eval 'ToString[x^2 Sqrt[a^2 - x^2], InputForm]'            # Sqrt[a^2 - x^2]*x^2
```

The renderer shows the factors in the order the evaluated `Times` holds
them (moving symbols ahead of a constant-free sum, as WL does). The
difference is upstream, in how Woxi sorts `Star[a, b]` against a power, one
sum against another, and a power of a sum against a power of a symbol; see
the `Times` ordering entries below.

### A minus written in front of a held rational coefficient

```sh
wolframscript -code 'ToString[TeXForm[HoldForm[-(4/3) Subst[a, b]]]]'  # \frac{1}{3} (-4) \text{Subst}(a,b)
woxi eval 'ToString[TeXForm[HoldForm[-(4/3) Subst[a, b]]]]'            # -\frac{4}{3} \text{Subst}(a,b)
```

WL boxes the held `Times[Times[-1, Times[4, Power[3, -1]]], Subst[a, b]]`
piecewise, leaving `(-4)` stranded behind the `\frac{1}{3}`; Woxi gathers
the coefficient into one signed fraction, as both engines do for the
evaluated `Times[Rational[-4, 3], Subst[a, b]]`. Not reproduced on purpose.

### A held `1/x` loses its explicit `1` in FullForm and Part

```sh
wolframscript -code 'ToString[FullForm[Hold[1/(2 y) z]]]'  # Hold[Times[Times[1, Power[Times[2, y], -1]], z]]
woxi eval 'ToString[FullForm[Hold[1/(2 y) z]]]'            # Hold[Times[Power[Times[2, y], -1], z]]

wolframscript -code 'Hold[1/(2 y)][[1, 1]]'  # 1
woxi eval 'Hold[1/(2 y)][[1, 1]]'            # 2*y
```

WL reads `1/(2 y)` as `Times[1, Power[Times[2, y], -1]]` and keeps the `1`
while the expression is held; Woxi's parse tree is the quotient itself, so
`Part` sees the divisor and the exponent as its two parts. The evaluated
expression is the same in both.

### TraditionalForm boxes are written out inline instead of as TemplateBoxes

`ToBoxes[TraditionalForm[…]]` differs in representation, not in picture:
wolframscript hands every special function to a named front-end template —
`TemplateBox[{"n","x"}, "LegendreP"]`, `{"x"}, "Gamma"`, `{"s"}, "Zeta"` —
while Woxi writes the same layout out as
`SubscriptBox`/`SubsuperscriptBox` rows, because it has no front end and its
own box renderers have to draw it. Only the outer
`TagBox[FormBox[…, TraditionalForm], …]` wrapper is common; `n!` and
`Subscript[a, b]` do conform.

`Row` is a template in both (`{"2","x","t"}, "RowDefault"`) — but only in the
box *escape* an `InputForm` string carries, which nothing has to draw.
The box builder that feeds Woxi's own renderers keeps flattening it:
`ToString[Row[{"a", 1}], TraditionalForm]` is `RowBox[{a, 1}]` against WL's
`TemplateBox[{"a", 1}, RowDefault]`, and `ToBoxes[Row[{"a", 1}]]` writes the
call's own source out as a `RowBox`. `ToBoxes` there also takes only one
argument where WL takes a form as the second.

Smaller TraditionalForm typesetter differences, all in representation:

- An inverse trig function is `arctan(x)`, WL writes `tan^-1(x)`
  (`SuperscriptBox["tan", RowBox[{"-", "1"}]]`).
- A held call keeps its StandardForm spelling in WL — `Hold[TraditionalForm[
  Style[x, Red]]]` boxes as `Style[x, Red]` and `Exp[x]` as `exp(x)`,
  `Column[…]` as its `GridBox` — where Woxi typesets the call.
- `Times[-1, Pi, a]` is `-π a`; WL writes `π (-a)`.
- A curried call `h[x][y, z]` gets WL's extra row around the arguments.
- `EulerGamma`, `GoldenRatio` and `Catalan` are `TagBox`es in WL, bare glyphs
  in Woxi, and keep their canonical place in a product where the numeric
  constants (numbers, `ⅈ`, `ⅇ`, `π`, `°`) move to the front in both.
- `ToString[TraditionalForm[…]]` and the InputForm of a held
  `TraditionalForm[…]` write `Row` as its template and a styled string with
  its quotes, as WL does; `ToString[…, TraditionalForm]` keeps the flat
  `RowBox`, the `U+2147`/`U+2148` letters and the `U+2009` gap Woxi's own
  renderers draw, where WL writes the template, `\[ExponentialE]` and `" "`.

### `ToBoxes` in StandardForm drops StyleBox and held TagBox

`ToBoxes[Style[x, Red]]` is `"x"` where wolframscript gives
`StyleBox["x", RGBColor[1, 0, 0], StripOnInput -> False]`. The TraditionalForm
builder learned both; the StandardForm one has not.

### A box segment mid-string is not shown as DisplayForm

WL renders a `\!\(\*…\)` box segment *anywhere* in a string as
`DisplayForm[<box>]` in terminal OutputForm; Woxi only does it when the string
*starts* with the segment. Generalising it would break plot labels, whose SVG
renderer typesets the raw markers — the real fix needs a "terminal text" vs
"text handed to a typesetter" distinction the form system does not have.

`\!\(…\)` **without** `\*` is full 2D linear syntax; WL parses it into a
`RowBox`, Woxi leaves it literal.

### Machine reals in grid cells render at six significant figures

`TableForm`/`Grid`/`MatrixForm` cells and typeset labels round machine reals to
6 significant figures with a `×10^exp` switch outside `[-5, 5]`, matching WL's
StandardForm. The CLI `eval` InputForm keeps full precision, as
`wolframscript -code` does. This is a deliberate display rule, listed here
because it makes the two layers disagree with each other.


## Canonical ordering of Plus and Times

### A coefficient times a sum of calls sorts before a lone call

```sh
wolframscript -code 'Cos[x] + 2 (Sin[x] + Sin[2 x])'        # Cos[x] + 2*(Sin[x] + Sin[2*x])
woxi eval 'Cos[x] + 2 (Sin[x] + Sin[2 x])'                  # 2*(Sin[x] + Sin[2*x]) + Cos[x]
wolframscript -code 'a/(a^2 + b^2) - 2 b/(a^2 + b^2)'       # a/(a^2 + b^2) - (2*b)/(a^2 + b^2)
woxi eval 'a/(a^2 + b^2) - 2 b/(a^2 + b^2)'                 # (-2*b)/(a^2 + b^2) + a/(a^2 + b^2)
```

Plus puts a multiple of a sum of calls ahead of a call, and orders two
quotients over the same denominator by their coefficient before their
numerator. `Sort` agrees with wolframscript on the stripped terms. It shows
in `ComplexExpand` (`I*(Sin[x] + Sin[2*x]) + Cos[x]`).

### `WaveletPsi[MexicanHatWavelet[1], t]` keeps the polynomial factor's sign

```sh
wolframscript -code 'WaveletPsi[MexicanHatWavelet[1], t]'  # (-2*(-1 + t^2))/(Sqrt[3]*E^(t^2/2)*Pi^(1/4))
woxi eval 'WaveletPsi[MexicanHatWavelet[1], t]'            # (2*(1 - t^2))/(Sqrt[3]*E^(t^2/2)*Pi^(1/4))
```

The values are equal, and Woxi's form is what both engines print for the
product typed in directly. WL builds it from the second derivative of a
Gaussian, which leaves `-2 (-1 + t^2)` unexpanded.

Woxi has three independent canonical-order comparators (one for `Sort`, one for
`Order`, one for `Plus` terms) and they drift from each other as well as from
Wolfram. Diagnose an ordering divergence by testing `Sort[…]`, `Order[a, b]`
and the bare `a + b` separately.

The `Plus` comparator follows Wolfram for every term led by a function call
(`plus_call_led_ordering` in `tests/interpreter_tests/arithmetic.rs`): calls
sort after the symbols, powers and products, by head name, then from the
last factor of a product by that call's arguments, its exponent and the
remaining factors. What is left are the entries below.

### `Sort` puts a symbol before a sum, and a call before its curried form

```sh
wolframscript -code 'Sort[{c, a + b}]'          # {a + b, c}
woxi eval 'Sort[{c, a + b}]'                    # {c, a + b}

wolframscript -code 'Sort[{f[x][y], f[x]}]'     # {f[x], f[x][y]}
woxi eval 'Sort[{f[x][y], f[x]}]'               # {f[x][y], f[x]}
```

Wolfram compares a sum against a symbol from the sum's last term (`b < c`)
and a curried call after the plain call with the same head. The `Plus`
comparator has both rules (`h[a + b] + h[c]`, `f[x] + f[x][y]`); the `Sort`
comparator still orders by expression class first.

### Cardano roots of a parametric cubic: term order and a shared denominator

```sh
wolframscript -code 'Solve[z^3 + a z^2 + b == 0, z][[1]]'
# {z -> (-a + (2^(1/3)*a^2)/(…)^(1/3) + (…)^(1/3)/2^(1/3))/3}
woxi eval 'Solve[z^3 + a z^2 + b == 0, z][[1]]'
# {z -> -1/3*a + (2^(1/3)*a^2)/(3*(…)^(1/3)) + (…)^(1/3)/(3*2^(1/3))}

wolframscript -code 'Solve[a z^3 + b z^2 + c z + d == 0, z][[1]]'
# {z -> -1/3*b/a - (2^(1/3)*(-b^2 + 3*a*c))/(3*a*C) + C/(3*2^(1/3)*a)}
woxi eval 'Solve[a z^3 + b z^2 + c z + d == 0, z][[1]]'
# {z -> -1/3*b/a + C/(3*2^(1/3)*a) - (2^(1/3)*(-b^2 + 3*a*c))/(3*a*C)}
```

The roots are the same (`solve_symbolic_cubic` in
`tests/interpreter_tests/algebra.rs` checks them numerically). Two things
differ. Some first roots WL writes over one denominator (`z^3 - z^2 + c`,
`2 z^3 + c z^2 + 1`, `z^3 + a z^2 + b`), and no rule for which ones has been
found yet. And the `Plus` comparator orders `p X^(-1/3)` after `X^(1/3)` when
the sum `X` mentions `p`, where WL compares the product from its last
factor (`X^(-1/3)` before `X^(1/3)`). It also orders a compound numeric factor
(`Sqrt[3]`, `2 + I`, `1 + I Sqrt[3]`) as a factor rather than as a coefficient:

```sh
wolframscript -code 'InputForm[(1 + c)^(1/3) + Sqrt[3]/(1 + c)^(1/3)]'  # Sqrt[3]/(1 + c)^(1/3) + (1 + c)^(1/3)
woxi eval 'InputForm[(1 + c)^(1/3) + Sqrt[3]/(1 + c)^(1/3)]'            # (1 + c)^(1/3) + Sqrt[3]/(1 + c)^(1/3)
wolframscript -code 'InputForm[x^2 + (1 + Sqrt[3]) x]'                  # (1 + Sqrt[3])*x + x^2
woxi eval 'InputForm[x^2 + (1 + Sqrt[3]) x]'                            # x^2 + (1 + Sqrt[3])*x
```

The discriminant is written expanded (with its integer content outside the
root) unless that makes it more than two leaves larger, a threshold fitted
to thirteen WL cubics.

### Two rational-function arguments of the same head

```sh
wolframscript -code 'ToString[Subst[f[(Sqrt[2] - 2 x)/(-1 + Sqrt[2] x - x^2)], x] + Subst[f[(Sqrt[2] + 2 x)/(-1 - Sqrt[2] x - x^2)], x], InputForm]'
# Subst[f[(Sqrt[2] + 2*x)/(-1 - Sqrt[2]*x - x^2)], x] + Subst[f[(Sqrt[2] - 2*x)/(-1 + Sqrt[2]*x - x^2)], x]
woxi eval 'ToString[Subst[f[(Sqrt[2] - 2 x)/(-1 + Sqrt[2] x - x^2)], x] + Subst[f[(Sqrt[2] + 2 x)/(-1 - Sqrt[2] x - x^2)], x], InputForm]'
# Subst[f[(Sqrt[2] - 2*x)/(-1 + Sqrt[2]*x - x^2)], x] + Subst[f[(Sqrt[2] + 2*x)/(-1 - Sqrt[2]*x - x^2)], x]
```

Wolfram decides this pair by the denominators (`-1 - Sqrt[2] x - x^2` before
`-1 + Sqrt[2] x - x^2`); Woxi's argument comparison reaches the numerators
first. Surfaces in one intermediate sum of Rubi's `Int[Sqrt[Tan[x]], x]`
step display.

### A symbolic multiple of `Infinity` cancels against itself

```sh
wolframscript -code 'a Infinity - a Infinity'   # a*-Infinity + a*Infinity
woxi eval 'a Infinity - a Infinity'             # 0
```

Woxi collects the two terms as like terms; WL leaves the sum (its two
`DirectedInfinity` products are not combined). Numeric multiples agree
(`Infinity - Infinity` is `Indeterminate` in both), though the `Infinity::indet`
message names `-Infinity + Infinity` where WL says
`ComplexInfinity + ComplexInfinity` for `ComplexInfinity - ComplexInfinity`.

### Held symbols compare by short name

```sh
wolframscript -code 'BeginPackage["Foo`"]; Int::usage = "x"; EndPackage[]; Hold[Foo`Int] === Hold[Global`Int]'  # False
woxi eval 'BeginPackage["Foo`"]; Int::usage = "x"; EndPackage[]; Hold[Foo`Int] === Hold[Global`Int]'            # True
```

`Foo`x === Global`x` is `False` in both, but inside a compound expression
Woxi's structural comparison sees the two symbols under their short name.

### Plus orders a reciprocal monomial before a power of a sum

```sh
wolframscript -code 'ToString[1/x + Sqrt[1 - x^(-2)], InputForm]'  # Sqrt[1 - x^(-2)] + x^(-1)
woxi eval 'ToString[1/x + Sqrt[1 - x^(-2)], InputForm]'            # x^(-1) + Sqrt[1 - x^(-2)]
```

Only the *reciprocal* monomial diverges — `x + Sqrt[1 - x^2]` agrees, and the
`Times` counterpart of the rule is implemented.

### A string atom sorts before a product in Plus

`"m" + 2 a` is `m + 2*a` in WL and `2*a + m` in Woxi. For *symbols* Woxi orders
atom-versus-product correctly; strings live in a different ordering class that
skips the name comparison. Strings inside `Plus` never fully evaluate, so this
is pathological input — recorded only so the comparator is not touched for it.

### `x^2 + (a + b) x` keeps Woxi's order

WL prints `(a + b) x + x^2`. The products-compare-from-the-last-factor rule is
implemented for `Sort` and `Order` but not for the `Plus` comparator, which is
the delicate one (it is not strictly transitive and its sort is wrapped in a
panic guard with a string-key fallback).

### A power of a function call sorts before everything

```sh
wolframscript -code 'Sort[{g[x]^2, x}]'   # {x, g[x]^2}
woxi eval 'Sort[{g[x]^2, x}]'             # {g[x]^2, x}
```

WL gives a `Power` its **base**'s place in the order, so `g[x]^2` sorts where
`g[x]` does — after `f[x]` and after every bare symbol. Woxi's `Power[g[x], 2]`
falls through to the generic "a function call sorts after atoms" rule and so
leads unconditionally. All three comparators are affected (`Sort`, `Order` and
the `Plus` one), which is why the divergence also shows up in printed sums:
`g[x]^2 + x*f[x]` is `x*f[x] + g[x]^2` in WL. A symbol base is unaffected
(`Sort[{y^2, x}]` agrees, via the string fallback).

Surfaces in `Variance[GompertzMakehamDistribution[l, x]]`, whose closed form
holds both an `E^x ExpIntegralEi[-x]^2` and an `x HypergeometricPFQ[…]` term:
WL puts the `ExpIntegralEi` one first (`E` < `H`), Woxi the other. Value
identical, display only — the unit test pins the value by substituting
parameters and rounding.

### A scaled sum is ordered by its last term, not its first

`FunctionExpand[Gamma[0, z]]` is `-ExpIntegralEi[-z] + (-Log[-z^(-1)] +
Log[-z])/2 - Log[z]` in WL and `(-Log[-z^(-1)] + Log[-z])/2 -
ExpIntegralEi[-z] - Log[z]` in Woxi — same three terms, different order. WL
compares a `c (p + q)` term against its neighbours using the sum's **last**
(greatest) element: `Aa[z] + (Log[w] + Log[y])/2` keeps `Aa[z]` first because
`Log[y] > Aa[z]`, while `ExpIntegralEi[-z] + (b + c)/2` puts the sum first
because `c < ExpIntegralEi[-z]`. Woxi's `Plus` comparator always puts the
scaled sum first, so it happens to agree with WL on the second shape and not
the first. Adopting the last-element key means reworking
`compare_plus_terms`'s hybrid `has_transcendental_subexpr` branch, which
currently keys on the *earliest* variable — the same delicate comparator the
`x^2 + (a + b) x` entry above describes. Value identical, display only.

The negative-order incomplete gammas carry that same group, and WL regroups
around it as well: `FunctionExpand[Gamma[-2, z]]` collects the elementary part
over the single denominator `E^z z^3` (`-((-z/2 + z^2/2)/(E^z z^3)) + …`) where
Woxi leaves the two reciprocal powers apart, and
`FunctionExpand[ExpIntegralE[2, z]]` distributes the leading `z^(n-1)` inward
(`E^(-z) - z (…)`) where Woxi keeps the product. The half-integer orders regroup
the same way once past the two WL writes bare — `Gamma[1/2, z]` and
`Gamma[3/2, z]` match exactly, `Gamma[-1/2, z]` differs only in whether the
leading `-2` is distributed. All are value-identical — the unit tests check the
`Gamma[a, z] = (Gamma[a + 1, z] - z^a E^-z)/a` recurrence and compare
numerically against `ExpIntegralE`.

### A bare symbol sorts before a scaled sum that ends in it

```sh
wolframscript -code 'ToString[x + 2 (-1 + x), InputForm]'      # 2*(-1 + x) + x
woxi eval 'ToString[x + 2 (-1 + x), InputForm]'                # x + 2*(-1 + x)
wolframscript -code 'ToString[y + 0.5 (-0.5 - x), InputForm]'  # 0.5*(-0.5 - x) + y
woxi eval 'ToString[y + 0.5 (-0.5 - x), InputForm]'            # y + 0.5*(-0.5 - x)
```

Every sign and coefficient (`2 (1 - x)`, `0.5 (-1 - x)`, `(-1/2 - x)/3`) puts
the product first in WL, while `x + 2 (y + z)` and `z + 0.5 (y + z)` keep the
symbol first in both. Same comparator as the entry above: WL keys the scaled
sum on its last term (`x` ties with `x`, and a sum then sorts before the bare
symbol, the `-1 + x < x` rule of `Sort`). Value identical, display only.

### A complex iterator with a fractional span

```sh
wolframscript -code 'Table[k, {k, I, I + 1.5}]'   # {I, 1 + I}
woxi eval 'Table[k, {k, I, I + 1.5}]'             # Table::iterb, unevaluated
```

`Table`, `Do`, `Range` and `Product` accept a complex start only when the
span `(max - min)/step` is an exact integer (`{k, I, I + 1}` works). WL takes
the floor of any non-negative real span. `Sum` handles it.

### A machine complex number inside a symbolic sum is not bracketed

```sh
wolframscript -code 'x + 1.5 I'   # (0. + 1.5*I) + x
woxi eval 'x + 1.5 I'             # 1.5*I + x
wolframscript -code 'a + 0. I'    # (0. + 0.*I) + a
woxi eval 'a + 0. I'              # 0. + 0.*I + a
```

In WL the machine `Complex` is one atom, printed with its `0.` real part and
bracketed among the other summands. Woxi stores it as `re + im*I` terms of
the surrounding `Plus`, so there is no atom to bracket. Values agree.

### Mixed sharing/non-sharing numerators over a shared denominator

`(5 + x)/(3 + x) + b/(3 + x)^2` — WL puts the `b` term first, Woxi keeps the
sharing term first. Same for `a*x/(1 - x) + b/(1 - x)^2`. The all-sharing and
all-non-sharing cases are decoded and implemented.

`c/x^2 + x^(-1)` (a **monomial** rather than sum base) also diverges: WL puts
`c/x^2` first.

### `Together` does not hoist a numerator monomial for multi-factor denominators

```sh
wolframscript -code 'ToString[Together[x/(1-x) + x/(2-x)], InputForm]'
# -((x*(-3 + 2*x))/((-2 + x)*(-1 + x)))
woxi eval 'Together[x/(1-x) + x/(2-x)]'
# keeps the numerator expanded: (3*x - 2*x^2)/(...)
```

Related: after a cancellation Woxi returns an expanded numerator and
denominator where WL reconstructs from the surviving factors —
`Together[(x^2 (1+x))/((1-x) x)]` is `-((x (1+x))/(-1+x))` in WL. And
`Together[(x^2+x)/((x-1)(x+1))]` stays put in Woxi where WL gives `x/(-1+x)`.

### Radical quotient canonicalization

WL folds integer coefficients into radicals of rationals and combines radical
quotients where Woxi keeps them split:

```sh
wolframscript -code 'ToString[Numerator[Sqrt[11]/Sqrt[7]], InputForm]'
woxi eval 'Numerator[Sqrt[11]/Sqrt[7]]'
```

Concretely `3*Sqrt[11/3]` → `Sqrt[33]`, `Sqrt[30]/Sqrt[6]` → `Sqrt[5]`, and
`Numerator[-3 Sqrt[16]/(Sqrt[2] Sqrt[11])]` is `-6 Sqrt[2]` in WL against
Woxi's `-12`. Woxi's `Simplify` is also non-confluent here: given the
already-distributed sum, `Simplify[-6 Sqrt[399] + 6 Sqrt[2261]]` returns
`-6*(Sqrt[399] - Sqrt[2261])`, while the product form it came from is fixed.

### `Sqrt[1/(2 Pi)]` and constants outside a radical

`Sqrt[1/(2 Pi)]` stays as written instead of `1/Sqrt[2 Pi]`, and a constant
sitting outside the radical does not fold in: `Pi/(2 Sqrt[2 Pi])` stays where
WL gives `Sqrt[Pi/2]/2`.

### A rational power's base is not kept primitive

Woxi normalizes `(3/11)^(2/3)` to `(9/121)^(1/3)`, where WL keeps the
primitive base and even rewrites the other direction.

### Powers of integer bases that share a prime do not merge

A coefficient is absorbed into a power of an integer base the way WL does it
(`2^x/12` → `2^(-2 + x)/3`, `6^x/4` → `2^(-2 + x)*3^x`), but two *powers*
sharing a prime are not split and merged:

```sh
wolframscript -code '6^x 2^y'      # 2^(x + y)*3^x
woxi eval '6^x 2^y'                # 2^y*6^x
wolframscript -code '2 6^x 10^y'   # 2^(1 + x + y)*3^x*5^y
woxi eval '2 6^x 10^y'             # 2^(1 + x)*3^x*10^y
wolframscript -code '6 2^x 3^x'    # 6^(1 + x)
woxi eval '6 2^x 3^x'              # 2^(1 + x)*3^(1 + x)
```

Rational bases are left out of the absorption: WL splits them into their
primes and recombines, choosing between a base and its reciprocal in a way
not yet pinned down (`4/9 (2/3)^x` → `(3/2)^(-2 - x)`, `2/3 (2/3)^x` →
`(2/3)^(1 + x)`, `3 (2/3)^x` → `2^x*3^(1 - x)`).

### A complex coefficient blocks the negative-exponent fold

```sh
wolframscript -code 'ToString[TrigToExp[Sin[x]], InputForm]'   # …I/2/E^(I*x)…
woxi eval 'ToString[TrigToExp[Sin[x]], InputForm]'             # …I/2*E^(-I*x)…
```

Real, integer and symbolic coefficients all fold correctly
(`1/2 E^(-I x)` → `1/(2*E^(I*x))`); only a complex or imaginary coefficient
fails. `Coth` and `Csch` are left unevaluated by `TrigToExp` for the related
reason that their denominators carry a negative *real* E-power, which Woxi
prints as `-(1/E^x)`. `ArcSin`/`ArcCos`/`ArcCsch`/`ArcSech` are omitted because
the `Log`-argument `Plus` order diverges.

### `Expand` leaves rational coefficients unfolded

```sh
wolframscript -code 'ToString[Expand[(x - 1/2)(x + 1/2)], InputForm]'  # -1/4 + x^2
woxi eval 'Expand[(x - 1/2)(x + 1/2)]'                                # -1/4 - x/2 + x/2 + x^2
```

Within a term the folding is fixed; **across** terms it is not, because the
sum-level combiner keys `Rational[1,2]*x` and `Rational[-1,2]*x` differently.
Two smaller residues: `Expand[-x/3]` prints `x*-1/3` where WL gives `-1/3*x`,
and `Expand[(a-b)^2/12]`'s middle term prints `(2*a*b)/12` instead of
`(a*b)/6`.

### Trig powers are not canonicalized to reciprocal functions

Wolfram rewrites `1/Sin[x]` → `Csc[x]`, `Cos[x]/Sin[x]` → `Cot[x]` and
`Sin[x]^2/Cos[x]` → `Sin[x] Tan[x]`. Woxi ships only the self-contained
reciprocal-pair subset (`Sin[x] Csc[x]` → 1 and friends); lone reciprocals and
the cross-function rewrites stay as written.

### `Simplify` factors where wolframscript keeps an expanded quotient

`Simplify[(-2 + x + 5x^2 + 2x^3)/(-5)]` gives `(-1/5)(1+x)(2+x)(-1+2x)` in Woxi
against WL's `(2 - x - 5x^2 - 2x^3)/5`. Wolfram also factors *numerators* of
combined fractions further than Woxi does (collecting in `x` with factored
coefficients), so Woxi keeps a sum of quotients where WL prints one fraction —
`Together` alone matches exactly on the same inputs.

### `Simplify` expands factored denominators that wolframscript keeps

```sh
wolframscript -code '{Simplify[1/(x (1 + y))], Simplify[(a + b)/(e^2 (2 + p)^2)], Simplify[1/(e (1 + p) (2 + p))]}'
# {1/(x*(1 + y)), (a + b)/(e^2*(2 + p)^2), 1/(e*(2 + 3*p + p^2))}
woxi eval '{Simplify[1/(x (1 + y))], Simplify[(a + b)/(e^2 (2 + p)^2)], Simplify[1/(e (1 + p) (2 + p))]}'
# {(x + x*y)^(-1), (a + b)/(4*e^2 + 4*e^2*p + e^2*p^2), (2*e + 3*e*p + e*p^2)^(-1)}
```

`simplify_division` works on the expanded numerator and denominator and only
restores a `(sum)^n` denominator afterwards. WL is not simply cost-minimal
here: `(x + x*y)^(-1)` has the lower `SimplifyCount` (8 against 11), yet
`Simplify[1/(x (1 + y))]` keeps the factored input. It expands only some
denominators: `1/((1 + x) (2 + y))` becomes `(2 + 2*x + y + x*y)^(-1)`, and
`x/(y (1 + y))` becomes `x/(y + y^2)`. Which candidates WL generates is not
decoded. `Together`, `Cancel`, `ExpandDenominator` and `Factor` all match on
these inputs. `DifferenceRoot` values with symbolic parameters go through
`Simplify` and inherit this gap.

### `Simplify` can be pathologically slow

`Simplify[(z+1)*(1/(z+1) + PolyGamma[0,2] + (z+1)*(PolyGamma[0,2]^2 + PolyGamma[1,2]))/(z+1)]`
takes tens of seconds where wolframscript is instant. The time is spent
re-evaluating the result tree after simplification completes, not inside
`Simplify` itself.

### Three-factor sign flip

`(-1/3)*(-2+x)*y` diverges because Woxi evaluates `BinaryOp` pairs
left-associatively and flips the inner pair, while WL sees a flat `Times`.
Needs parser-level `Times` flattening.

### `Eliminate` clears denominators differently

`Eliminate[{x + y == 1, x - y == 2}, y]` is `x == 3/2` in Woxi and `2*x == 3`
in WL, which keeps integer coefficients.


## Numeric precision and floating point

### `SpherePoints` differs in the last bit, and small `n` is missing

`SpherePoints[n]` lays the points on the golden spiral Wolfram uses (the same
z values exactly, longitude `k` times the golden angle), but the x and y
coordinates come out up to two ulps away from Wolfram's — its trigonometric
evaluation rounds differently. The `n` Wolfram takes from a table of optimal
arrangements (2–12, 14, 16, 18, 20, 22, 24, 30, 32, 40, 55, 60) are not
implemented and stay unevaluated.

### `NMinimize` ignores `WorkingPrecision`

`NMinimize[Total[EuclideanDistance[{a, b, c}, #] & /@ pts], {a, b, c},
WorkingPrecision -> 30]` gives 30-digit numbers in wolframscript; Woxi
answers with machine numbers (located to about 8 digits).

### Arbitrary-precision complex numbers print every working digit

`N[(1 + I)/3, 20]` is right but shows 58 digits per part,
`0.3333…3334`20.`, where wolframscript shows 38, `0.33333333333333333333333333333333333333`20.`.
Only a complex result of a one-argument function (`N[Sin[1 + I], 20]`)
gets wolframscript's per-part precision markers; others, including complex
powers like `N[(1 + 2 I)^(3/2), 20]`, carry a flat `20.` where WL tracks
`19.10381839395503` and `20.148756595204922`. So does a one-argument
result whose input is tiny next to its output (`N[Log[-10^-17 + I/10^40],
10]`, WL `10.149…` and `9.0536…`), and the imaginary part of `Log` of a
negative arbitrary-precision real (`Log[-2.5`20]`: WL `36.75…`, Woxi
`20.497…`, the accuracy of the real part).

### Inverse trigonometric functions of arbitrary-precision reals off their real domain stay unevaluated

```sh
wolframscript -code 'ArcSin[3`20]'  # 1.5707963267948966192313216916397514421`36.42616338166145 - 1.76274717403908605046521864995958461806`20.220613765883606*I
woxi eval 'ArcSin[3`20]'            # ArcSin[3.`20.]
```

`N[ArcSin[3], 20]` of an exact argument is complex as in Wolfram, but an
already arbitrary-precision argument (`ArcCos[-2.`30]`, `ArcCosh[0.5`20]`
except at 0) stays unevaluated: the functions' own arbitrary-precision
paths are real-only.

### Inexact `Solve` of `b^x == v` with negative `v` is 1 ULP off

```sh
wolframscript -code 'InputForm[Solve[2^x == -3., x]]'  # {{x -> 1.584962500721156 + 4.532360141827194*I}}
woxi eval 'InputForm[Solve[2^x == -3., x]]'            # {{x -> 1.5849625007211563 + 4.532360141827194*I}}
```

Woxi inverts through `Log[b, v]`, which matches wolframscript for positive
`v` (`Solve[2^(x - 2) == 15., x]`) and for `Log[2, -3.]` itself. Inside
`Solve`, wolframscript rounds the real part of a negative `v` differently
(`Log[3.]/Log[2.]`), but not consistently: `Solve[3^x == -15., x]` differs
in the last digit of the imaginary part instead. The same applies to the
complex roots of `Solve[2^b + 4^b == 6., b]`.

### `GaussianMatrix` differs in the last digits

```sh
wolframscript -code 'InputForm[GaussianMatrix[{{2}, 1}]]'  # {0.05088223571170066, 0.21183832321622456, 0.47455888214414965, …}
woxi eval 'InputForm[GaussianMatrix[{{2}, 1}]]'            # {0.05088223571170065, 0.2118383232162245, 0.4745588821441496, …}
```

Woxi builds the kernel from machine `BesselI` values; wolframscript's
round differently, and neither is the correctly rounded kernel
(`0.050882235711700655`). The derivative matrices (`GaussianMatrix[r, {n1,
n2}]`) follow the same construction and differ by up to about 1e-15
relative, 1e-14 at radius 50.

### `` NIntegrate`GaussRuleData `` differs in the last digits

Woxi finds the Gauss–Legendre nodes by Newton iteration in machine
arithmetic; wolframscript's own computation rounds differently from 3 points
on (`5/18` is `0.2777777777777776` there, `0.27777777777777635` in Woxi).
Only `MachinePrecision` is implemented — a numeric precision leaves the call
unevaluated — and the 1-point rule gives the weight `1.` where wolframscript
divides by zero (`{{0.5}, {ComplexInfinity}, {}}`).

### `FindMinimum`/`FindMaximum` with two starting values stop at a different last digit

```sh
wolframscript -code 'FindMaximum[Sin[x], {x, 1, 2}]'   # {1., {x -> 1.5707963267114806}}
woxi eval 'FindMaximum[Sin[x], {x, 1, 2}]'             # {1., {x -> 1.5707963267948966}}
```

`{x, x0, x1}` makes Wolfram use a derivative-free method that stops at its
own tolerance. Woxi runs its Newton search from `x0` and lands on the
extremum to machine precision. The extremum is the same; the digits past
about 1e-10 are not.

### `Erf`, `Erfc` and `Erfi` of complex machine numbers differ in the last digits

```sh
wolframscript -code 'Erf[1. + 0.5 I]'   # 0.9507097283189574 + 0.18797346722338326*I
woxi eval 'Erf[1. + 0.5 I]'             # 0.9507097283189571 + 0.18797346722338332*I
```

Woxi sums the Maclaurin series for |Re z| < 2 and uses the Erfc continued
fraction beyond that. Both are correct to about 1e-13 relative, but not
correctly rounded the way Wolfram's arbitrary-precision kernel is.

### `NSolve` lists the solutions of a polynomial system in a different order

```sh
wolframscript -code 'NSolve[x == 0.47 ss + 0.13 && y == 0.85 ss + 0.99 && (-0.039 + x)^2 + (-1.015 + y)^2 == 0.034, {x, y, ss}]'
# {{x -> 0.04187863964178423, …, ss -> -0.18749225608131015}, {x -> 0.19667891812798455, …}}
woxi eval 'NSolve[x == 0.47 ss + 0.13 && y == 0.85 ss + 0.99 && (-0.039 + x)^2 + (-1.015 + y)^2 == 0.034, {x, y, ss}]'
# {{x -> 0.19667891812798463, …}, {x -> 0.04187863964178421, …, ss -> -0.1874922560813102}}
```

**Not reproducible** from the output alone. The solution set is the same, but
Wolfram's order comes from its internal polynomial-system solver. It is
ascending in some samples and descending in others (`NSolve[x == 1 + ss && y
== 2 ss && x^2 + y^2 == 3., {x, y, ss}]` lists the larger `x` first), with no
consistent key. Code that picks a solution by position (`… [[2]]`) can see a
different root.

### `Rationalize[x, 0]` and `ContinuedFraction[x]` of a machine real stop one convergent off

```sh
wolframscript -code 'Rationalize[0.64894397210534, 0]'   # 204604909/315289020
woxi eval 'Rationalize[0.64894397210534, 0]'             # 25217905/38859911
wolframscript -code 'ContinuedFraction[Sqrt[2.]]'        # {1, 2, … (20 twos)}
woxi eval 'ContinuedFraction[Sqrt[2.]]'                  # {1, 2, … (19 twos)}
```

**Not reproducible** from the output alone. WL's results are always
continued-fraction convergents of the double's exact binary value, but
neither function stops by a fixed error bound. No symmetric interval
(relative or in ulps) and no float-equality test fits all the samples.
Woxi accepts the first convergent within `0.65 $MachineEpsilon |x|`.
That matched 41 of 42 random reals sampled across magnitudes.
`ContinuedFraction` usually equals the `Rationalize` list with its last term
dropped, which Woxi always returns. On the same sample it matched 35 of 42. In the misses WL's list is one term
longer or shorter (`Sqrt[2.]`), or is the full `Rationalize` list
(`469.4554167169332`).

### Inexact zero loses WL's accuracy tracking

```sh
wolframscript -code 'ToString[PascalBinomial[6.0, -2], InputForm]'  # 0``15.954589770191005
woxi eval 'PascalBinomial[6.0, -2]'                                 # 0.
```

`Binomial[6.0, -2]` has the same reference value but drifts further — Woxi
returns an exact `0` there, losing the inexactness too. WL returns an
arbitrary-precision zero whose *accuracy* is `$MachinePrecision` — not a
machine `0.`, whose accuracy is 323.6. Reproducing it needs precision/accuracy
tracking through the Gamma-ratio path starting from a machine-real argument.

### Machine floats overflow where wolframscript promotes to arbitrary precision

Overflow of `Exp[x.]`, `a.^b`, products, quotients and sums of machine reals
is promoted to an arbitrary-precision number (up to a decimal exponent of
10000), but not everywhere:

```sh
wolframscript -code 'Exp[1000] // N'   # 1.97007111401704699388887935224`12.95…*^434
woxi eval 'Exp[1000] // N'             # Infinity

wolframscript -code 'Exp[-1000.]'      # 5.07595889754946`12.95*^-435
woxi eval 'Exp[-1000.]'                # 0.  (underflow is not promoted)

wolframscript -code 'Exp[1000.] > 10^433'   # True
woxi eval 'Exp[1000.] > 10^433'             # False (comparison goes through f64)
```

Exact input made inexact by `N` and underflow still collapse to `Infinity` /
`0.`, and results beyond the 10000-digit bound stay `Infinity`.

### `Overflow[]` is not absorbing, and its head is not `Real`

Woxi returns `Overflow[]`/`Underflow[]` for a power past `$MaxNumber`, but the
object itself is inert rather than WL's absorbing one:

```sh
wolframscript -code 'ToString[Overflow[] + 1, InputForm]'   # Overflow[]
woxi eval 'ToString[Overflow[] + 1, InputForm]'             # 1 + Overflow[]

wolframscript -code 'Head[Overflow[]]'                      # Real
woxi eval 'Head[Overflow[]]'                                # Overflow
```

Between `10^5` and the `$MaxNumber` boundary Woxi also keeps a rational power
symbolic (`2^(10^10/3)` stays `2^(10000000000/3)`) where WL materialises the
extracted integer factor.

### Positive and Sign underflow on a rational below the double range

```sh
wolframscript -code 'Positive[10^-400]'   # True
woxi eval 'Positive[10^-400]'             # False

wolframscript -code 'Sign[10^-400]'       # 1
woxi eval 'Sign[10^-400]'                 # 0
```

`Power` and the comparison operators compare such values exactly, but
`Positive`, `Negative` and `Sign` still route through `f64`, where the value
underflows to `0.0`.

### An accuracy-form literal's precision differs in the last digit

`1.5``20` stores its precision as `20 + log10(1.5)`. Woxi computes that with
`f64::log10` and gets `20.17609125905568`; wolframscript computes it at
arbitrary precision and gets `20.176091259055685`. Visible whenever the
precision tag is printed. `Accuracy[1.5]` differs in the last ULP for the same
reason.

### A symbolic constant raised to a BigFloat exponent stays symbolic

`Pi^0.5`20`` stays `Pi^0.5`20.`, so `N[Gamma[1/2], 20]` comes out malformed and
`N[Gamma[3/2], 20]` gives `0.5`20.*Pi^0.5`20.`. Routing it through
`N[base, p]` produces numbers but with a **different precision tag**
(`20.30` against WL's `20.46`, which accounts for the base magnitude) and a
last mantissa digit off by one — an arbitrary-precision rounding problem rather
than a dispatch one.

### `N[I]` stays exact

`N[I]` is `I` in Woxi and `0. + 1.*I` in wolframscript, so `N[Gamma[I]]` and
`N[LogGamma[I]]` echo instead of numericizing. Fixing it needs
`scalar * (0. + c*I)` to fold first, and that non-fold is deliberate: a pure
imaginary product is left as the `c*I` monomial so it still merges inside an
enclosing `Plus`. Both changes were tried and reverted.

`BesselJ`/`PolyGamma` at complex numeric arguments are separately unimplemented.

### An inexact number does not numericize a complex-valued term

`Plus`/`Times` with a machine number numericize a `NumericQ` term only when it
is real-valued (`Zeta[3] + 0.`); a complex-valued one stays symbolic:

```sh
wolframscript -code 'ZetaZero[1] + 1.'     # 1.5 + 14.134725141734695*I
woxi eval 'ZetaZero[1] + 1.'               # 1. + ZetaZero[1]
wolframscript -code 'Zeta[1/2 + I] + 1.'   # 1.1439364270771886 - 0.7220997435316738*I
woxi eval 'Zeta[1/2 + I] + 1.'             # 1. + Zeta[1/2 + I]
```

The fallback in `try_eval_to_f64` only produces an `f64`.

### `ZetaZero` machine values and precision

`N[ZetaZero[k], p]` returns the machine value; WL computes `p` digits.
Woxi's machine values are the correctly rounded doubles, which WL's are not
everywhere — **not reproducible**: `N[ZetaZero[6709]]` is `7005.062866174921`
in Woxi and `7005.062866174927` in WL, whose own 25-digit value is
`7005.06286617492058…`, and `Im[ZetaZero[1]] - 14.` is off in WL's last digits
compared to `Im[N[ZetaZero[1]]] - 14.`.

### `Complex[Real, Real]` has the wrong head

```sh
wolframscript -code 'Head[Complex[0., 0.]]'   # Complex
woxi eval 'Head[Complex[0., 0.]]'             # Plus
```

Woxi has no dedicated `Complex` expression variant, so a real-real complex is
rewritten to `Plus[Real, Times[Real, I]]` during evaluation. The printed string
matches; the head does not.

Code that has to tell a complex with a zero imaginary part from a real must
therefore go by "extracts as complex but not as a real" rather than by the
head — see `reject_non_real_base` in `math_ast/elementary.rs`. Such a value
also renders with an explicit `*` where WL's messages use a space:

```sh
wolframscript -code 'CubeRoot[2. + 0. I]'   # …The parameter 2. + 0. I should be…
woxi eval 'CubeRoot[2. + 0. I]'             # …The parameter 2. + 0.*I should be…
```

### `Binomial[n, real]` carries Gamma-error noise

`Binomial[10, 3.]` is `119.99999999999987` in Woxi and `120.` in WL;
`Binomial[100, 3.]` is `161699.999999994` against `161700.0000000032`. WL's
exactness is **unpredictable** — it depends on the magnitudes of the Gamma
arguments, not of the result, so `Binomial[26, 13.]` is f64-exact there and
`Binomial[27, 13.]` is not. Returning the exact BigInt fixes the small cases
and introduces new divergences wherever WL is itself inexact.
**Not reproducible.**

### `Variance` and friends differ in the last ULP

`Variance[{1., 2., 3.}]` is `1.` in Woxi and `1.0000000000000002` in WL, whose
accumulation matches none of the textbook formulas (`Σ(x-m)²/(n-1)`,
`(Σx² − n m²)/(n-1)`, `(n Σx² − (Σx)²)/(n(n-1))`, mean-of-squared-deviations
scaled, or Welford; also not compensated or exactly rounded — a 40-sample
battery found every candidate off by up to 3 ULPs). Affects `Variance`,
`Correlation`, `Kurtosis`, `Cumulant`, `Variance` of an `Audio`, and the
machine bandwidth and `Variance` of a `KernelMixtureDistribution` of machine
data (built on the sample standard deviation). **Not reproducible.**

`PageRankCentrality[CycleGraph[4]]` is `{0.25000000000000006, …}` against WL's
clean `{0.25, …}` — power-iteration noise in the same class.

### FresnelF and FresnelG lose accuracy through composition

Woxi composes them from `FresnelS`/`FresnelC`; wolframscript uses dedicated
rational approximations. Real outputs differ in the last 1–2 digits for small
`|x|`, and the cancellation as `S, C → 1/2` degrades relative accuracy to
~1e-13 by `x = 10`. **Not reproducible** without WL's coefficients.

### Machine Bessel, Airy and Struve values are not correctly rounded in WL

Woxi rounds machine `BesselJ`, `BesselY`, `BesselI`, `BesselK`, `StruveH`,
`StruveL` and the four Airy functions from a 64-bit evaluation, so they are
correctly rounded (checked against 20-digit values of the exact double
arguments). wolframscript's agree on most probes but not all, and drift
further for large arguments: `BesselJ[0, 100.]` is `…223327` against the
correctly rounded `…223122`, `BesselJ[1, 3.8]` `…731658` against `…7317`,
`BesselY[0, 2.4]` `…7439` against `…7438`, `AiryAi[20.]` `…6705310*^-27`
against `…6705404*^-27` and `AiryAiPrime[-1.]` is off by 4e-14. Anything
built on them inherits the difference — `N[BesselJZero[1, 1]]` is `…5125`
(the root is `3.83170597020751231…`) against WL's `…512`, and
`GaussianFilter`'s kernel and `PDF[VonMisesDistribution[2, 3], 2.5]` differ
in the last digit. **Not reproducible.**

### Machine special-function values differ in the last digits

Woxi's machine `LogGamma` is msun's `lgamma` (accurate to an ulp); it matches
wolframscript on most arguments but not all — `LogGamma[1.5]` is `…4522` (the
correctly rounded value) against WL's `…4526`, `LogGamma[10.3]` `…355` against
`…359`, `LogGamma[4.0]` `…055` (`Log[6.]`, correctly rounded) against `…0552`.
The same 1–3 ULP class: `InverseErf[0.5]` (`…698` vs `…6993`, Woxi nearer).
**Not reproducible** without WL's own approximations.

### Machine exponential integrals are not correctly rounded in WL

Woxi rounds machine `ExpIntegralE`, `ExpIntegralEi`, `LogIntegral`,
`Gamma[a, z]` and `Gamma[x]` from a 64-bit evaluation, so they are correctly
rounded (checked against 40-digit values of the exact double arguments). WL's
`Gamma[x]` is correctly rounded too and the two agree, but its exponential
integrals are not: `LogIntegral[0.3]` is `…8946` against the correctly rounded
`…894`, `ExpIntegralE[2, 0.2]` `…031` against `…032`,
`ExpIntegralE[1, 0.7]` `…0923` against `…092`, `ExpIntegralEi[3.]` `…414`
against `…416`, `ExpIntegralE[3, 10.]` is off by 1.6e-14 and
`ExpIntegralEi[300.]` by 2e-14. Next to li's root WL loses whole digits:
`LogIntegral[1.4513692348]` is `-2.2383719991068265*^-10` in WL and
`-2.2383724226744812*^-10` (correct) in Woxi. **Not reproducible.**

The guard digits WL prints past the precision of `N[LogIntegral[3/10], 30]`,
`N[ExpIntegralE[2, 2/10], 30]` and `N[Gamma[0, 2/10], 30]` differ from
Woxi's from about the 35th digit; Woxi's are the true ones (they agree with
WL's own 60-digit values). For a complex result such as
`N[ExpIntegralE[1, -2], 20]` WL tags each part with its own precision
(`20.077…`, `19.879…`), Woxi with the requested one.

### Machine `PolyGamma` and Hurwitz `Zeta` values are not correctly rounded in WL

Woxi rounds machine `PolyGamma` and `Zeta[s, a]`/`HurwitzZeta` from an
arbitrary-precision evaluation, so they are correctly rounded (checked with
mpmath at 30 digits). wolframscript agrees on most arguments but not all:
`PolyGamma[-7.7]` is `…6933` against WL's `…7003`, `PolyGamma[2, 0.3]`
`…72604` against `…72617`, `Zeta[3, -2.5]` `…413456` against `…41346` and the
real part of `HurwitzZeta[0.5, -0.5]` `…216304` against `…21629`. Next to the
root of ψ WL loses digits: `PolyGamma[1.46]` is `-0.0015805619870834522`
(mpmath: `-0.00158056198708345210…`) against WL's `…0833398`. At a negative
`s` the drift reaches the third digit: `Zeta[-5.817415911509695,
0.1295065568974257]` is `-0.0033395080665649794` in Woxi and mpmath and
`-0.0033354818170026833` in WL. **Not reproducible.** The last-digit cases
are compared with a tolerance by the unit-test conformance check
(`NEAR_MATCH`); the negative-`s` one is skipped.

### `PolyGamma` of a fractional order is unimplemented

```sh
wolframscript -code 'PolyGamma[0.5, 1.9]'   # 0.7474244588681002
woxi eval 'PolyGamma[0.5, 1.9]'             # PolyGamma[0.5, 1.9]
```

WL continues ψ⁽ⁿ⁾ to non-integer orders by fractional calculus; neither
Espinosa–Moll's `e^(-γs) ∂s[e^(γs) ζ(s+1, z)/Γ(-s)]` (0.8274… here) nor
`Γ(ν+1) ζ(ν+1, z)` reproduces its values, so Woxi leaves the call unevaluated.

### Trig equations whose roots wolframscript leaves as ArcTan forms

```sh
wolframscript -code 'Solve[Sin[t] == Sin[t + 2 Pi/5], t]'
# {{t -> ConditionalExpression[ArcTan[Sqrt[2*(3 + Sqrt[5])]/(3*Sqrt[5 + 2*Sqrt[5]]
#   - Sqrt[5*(5 + 2*Sqrt[5])])] + 2*Pi*C[1], …]}, {t -> … -Pi - ArcTan[…] …}}
woxi eval 'Solve[Sin[t] == Sin[t + 2 Pi/5], t]'
# {{t -> ConditionalExpression[(-7*Pi)/10 + 2*Pi*C[1], …]},
#  {t -> ConditionalExpression[(3*Pi)/10 + 2*Pi*C[1], …]}}
```

The ArcTan expressions are 3π/10 and -7π/10; Woxi solves `Sin[A] == Sin[B]`
through `A == B + 2πk || A == π - B + 2πk` and gets the multiples of π
directly. `Reduce` over a bounded interval differs the same way.

### Reduce keeps a common equation inside each branch

```sh
wolframscript -code 'Reduce[z == Sin[t] && Sin[t] == 1/2 && 0 <= t < 2 Pi && z > 0, {z, t}, Reals]'
# z == 1/2 && (t == Pi/6 || t == (5*Pi)/6)
woxi eval 'Reduce[z == Sin[t] && Sin[t] == 1/2 && 0 <= t < 2 Pi && z > 0, {z, t}, Reals]'
# (z == 1/2 && t == Pi/6) || (z == 1/2 && t == (5*Pi)/6)
```

When to factor a shared equation out of the disjunction is not decoded:
`Reduce[{x == 2, y^2 == x + 2}, {x, y}]` factors (`x == 2 && (y == -2 ||
y == 2)`), but `Reduce[{x^2 == 1, y == 1}, {x, y}]` distributes and puts `y`
first (`(y == 1 && x == -1) || (y == 1 && x == 1)`), and Woxi does the
opposite in both.

### A definite integral up to a symbol carries no condition

```sh
wolframscript -code 'Integrate[2 (1 + 3 x)^(-1/2), {x, 0, t}]'
# ConditionalExpression[(4*(-1 + Sqrt[1 + 3*t]))/3, Re[t] > 0 || …]
woxi eval 'Integrate[2 (1 + 3 x)^(-1/2), {x, 0, t}]'
# -4/3 + (4*Sqrt[1 + 3*t])/3
```

Woxi returns `F[t] - F[0]` unconditionally and does not collect it.

### `x*(a*x)^n` orders its factors the other way round

```sh
wolframscript -code '{x (a x)^n, Sort[{(a x)^n, x}]}'   # {x*(a*x)^n, {x, (a*x)^n}}
woxi eval '{x (a x)^n, Sort[{(a x)^n, x}]}'             # {(a*x)^n*x, {(a*x)^n, x}}
```

A power of a product compares by its base, and `x` precedes `a*x` (the
last-factor rule), but `canonical_cmp` and `sort_symbolic_factors` put the
power first; `Order[x, (a x)^n]` is right. So `Integrate[(a x)^n, x]` prints
`((a*x)^n*x)/(1 + n)` where WL prints `(x*(a*x)^n)/(1 + n)`.

### RegionCentroid of a region difference

`RegionCentroid[RegionDifference[outer, hole]]` is worked out exactly only
when the hole lies inside `outer`. wolframscript also handles a hole that
sticks out (`Disk[{2, 2}, 1]` at a corner of `Rectangle[{0, 0}, {2, 2}]` gives
`{(2*(26 - 3*Pi))/(3*(16 - Pi)), …}`), where Woxi stays unevaluated; and for
an inexact disk it integrates over a discretized disk, so
`Disk[{0.38, 0.38}, 0.55]` gives `-0.11783686539341964` against the exact
`-0.11841487309694458`.

### HighlightImage returns the image at its own size

```sh
wolframscript -code 'ImageDimensions[HighlightImage[ConstantImage[Black, {200, 100}], {Red, Point[{5, 5}]}]]'
# {400, 200}
woxi eval 'ImageDimensions[HighlightImage[ConstantImage[Black, {200, 100}], {Red, Point[{5, 5}]}]]'
# {200, 100}
```

wolframscript rasterizes the highlighted picture through the front end, at
the display's scale — twice the pixels on a Retina Mac — and with a minimum
height (`{20, 10}` becomes `{40, 33}`). Woxi draws onto the image's own
pixels. **Not reproducible** across machines.

### Exact generalized eigenvectors are scaled differently

```sh
wolframscript -code 'Eigenvectors[{{{2, 0}, {0, 1}}, {{2, 1}, {1, 1}}}]'
# {{-1, Sqrt[2]}, {1, Sqrt[2]}}
woxi eval 'Eigenvectors[{{{2, 0}, {0, 1}}, {{2, 1}, {1, 1}}}]'
# {{-(1/Sqrt[2]), 1}, {1/Sqrt[2], 1}}
```

Woxi takes the eigenvectors of a pencil `{a, b}` from `Inverse[b] . a`, which
scales them so their last component is 1. The kernel's exact generalized
eigenvectors clear that radical denominator; the rule behind its scaling is not
known yet. Eigenvalues and machine eigenvectors agree.

### Machine eigenvectors have other signs

```sh
wolframscript -code 'Eigenvectors[{{2., 1.}, {1., 2.}}]'   # {{0.707…, 0.707…}, {-0.707…, 0.707…}}
woxi eval 'Eigenvectors[{{2., 1.}, {1., 2.}}]'             # {{-0.707…, -0.707…}, {-0.707…, 0.707…}}
```

An eigenvector's sign is LAPACK's choice in wolframscript and follows no
normalization of its components that Woxi could copy (`{{-1., 0.}, {0.,
-1.}}` for `{{4., 0.}, {0., 1.}}`, `{{0., 1.}, {-1., 0.}}` for `{{2., 0.},
{0., 3.}}`). Anything built on them inherits it, e.g. the directions of
`MultivariateStatistics`EllipsoidQuantile`. **Not reproducible.**

### Equations in a radical and its reciprocal

```sh
wolframscript -code 'Solve[Sqrt[x] + 1/Sqrt[x] == 3, x]'
# {{x -> (7 - 3*Sqrt[5])/2}, {x -> (7 + 3*Sqrt[5])/2}}
woxi eval 'Solve[Sqrt[x] + 1/Sqrt[x] == 3, x]'
# {{x -> Root[-1 + 18*#1 - 79*#1^2 + 18*#1^3 - #1^4 & , 2, 0]}, …}
```

Woxi does not substitute `u = Sqrt[x]` (or `u = (2 + y)^(1/3)`), so `Solve`
answers with `Root` objects or stays unevaluated, and `NSolve` falls back to
a numeric search that lands a few ULPs off (`159.83316151184493` for
`79 + 33 Sqrt[6]`, correctly `159.83316151184488`). wolframscript's
`NSolve` also lists such roots in the reverse of `Solve`'s order.

### `Sort` puts a sum before a lone monomial of a call

```sh
wolframscript -code 'Sort[{Pi + 2 Pi C[1], 2 Pi C[1], -Pi/3 + 2 Pi C[1]}]'
# {2*Pi*C[1], -1/3*Pi + 2*Pi*C[1], Pi + 2*Pi*C[1]}
woxi eval 'Sort[{Pi + 2 Pi C[1], 2 Pi C[1], -Pi/3 + 2 Pi C[1]}]'
# {-1/3*Pi + 2*Pi*C[1], Pi + 2*Pi*C[1], 2*Pi*C[1]}
```

`canonical_cmp`'s class rule for sums (decoded on polynomials in `x`) sorts
these sums before the monomial; `Order[2 Pi C[1], Pi + 2 Pi C[1]]` is `1`
in both. Solve's trig families sort with their own key for this reason.

### Piecewise integrals with parametric bounds under assumptions

```sh
wolframscript -code 'Integrate[Piecewise[{{1, 0 <= u <= 1}}, 0], {u, 0, w}, Assumptions -> w > 0]'
# Piecewise[{{1, w > 1}}, w]
woxi eval 'Integrate[Piecewise[{{1, 0 <= u <= 1}}, 0], {u, 0, w}, Assumptions -> w > 0]'
# Piecewise[{{1, w > 1}, {w, Inequality[0, Less, w, LessEqual, 1]}}, 0]
```

The values agree for every `w > 0`, but WL folds the last piece into the
default using the assumption. Which bound binds at a tie (a numeric lower
bound, else the first; the last upper bound, with the integration limits
counted after the condition) and the order of the pieces (the `Sort` order
of their values) are fitted to wolframscript's results for intervals
linear in one parameter. Nonlinear bounds (`u <= v^(1/3) - 1`) give
`Root` objects in WL and radicals in Woxi.

### Isolated powers of a linear expression

```sh
wolframscript -code 'Solve[(1 + y)^2 == c, y]'   # {{y -> -1 - Sqrt[c]}, {y -> -1 + Sqrt[c]}}
woxi eval 'Solve[(1 + y)^2 == c, y]'             # {{y -> (-2 - 2*Sqrt[c])/2}, {y -> (-2 + 2*Sqrt[c])/2}}
wolframscript -code 'Solve[(a + b y)^3 == v, y]' # {{y -> (-a + v^(1/3))/b}, …}
woxi eval 'Solve[(a + b y)^3 == v, y]'           # {{y -> -(a/b) + (b^6*v)^(1/3)/b^3}, …}
```

WL isolates the power and takes the root of the other side, simplified as
if the base were positive. Its fifth powers come back as `Root` objects and
its sixth powers as nested radicals. Woxi solves cubes by Cardano's formula
and the other degrees with roots of unity.

### NDSolve returns a complex solution as two real interpolations

```sh
wolframscript -code 'NDSolve[{x'"'"'[t] == I x[t], x[0] == 1}, x, {t, 0, 1}]'
# {{x -> InterpolatingFunction[{{0., 1.}}, <>]}} with complex values
woxi eval 'NDSolve[{x'"'"'[t] == I x[t], x[0] == 1}, x, {t, 0, 1}]'
# {{x -> Function[{NDSolve$z}, InterpolatingFunction[…][NDSolve$z] + I InterpolatingFunction[…][NDSolve$z]]}}
```

Woxi's integrator carries a real state vector, so a complex system is
solved as the real system of its parts (`ComplexExpand` of each equation)
and recombined. The values agree; the shape of the solution does not.

### `$MinNumber`, `$MaxNumber` and exponents past 32 bits

```sh
wolframscript -code '{$MinNumber, 1.5`20*^-4000000000}'
# {6.22968824967532196119819746965503015872`15.954589770191005*^-1355718576299610, 1.5`20.*^-4000000000}
woxi eval '{$MinNumber, 1.5`20*^-4000000000}'
# {$MinNumber, 1.5`20.}
```

A Woxi `BigFloat` keeps its digits with the decimal point shifted into
place, so it cannot hold the arbitrary-precision range limits. A literal
exponent outside `i32` is read as 0, which silently changes the value.
`$MinMachineNumber` and `$MaxMachineNumber` agree.

### CovarianceFunction of a concrete ARMA(1, 1) past lag 1

```sh
wolframscript -code 'CovarianceFunction[ARMAProcess[{1/2}, {3/10}, 2], 3]'   # 259/300
woxi eval 'CovarianceFunction[ARMAProcess[{1/2}, {3/10}, 2], 3]'             # 46/75
```

Woxi gives the textbook value `a^(h-1) (a + b) (1 + a b) v/(1 - a^2)`, which is
also what WL's own symbolic `CovarianceFunction[ARMAProcess[{a}, {b}, v], h]`
gives with these numbers substituted; WL's result for concrete parameters
differs from lag 2 on. Other orders agree exactly (an ARMA(2, 2) checked to
lag 4). For a nearly nonstationary machine process the value agrees to
about 8 digits only: WL's 1.2819544533776392*^8 against Woxi's
1.2819544632769972*^8 for `ARMAProcess[{1.997…, -0.997…}, {0.268…}, 1]`,
whose exact value is 1.28195445098…*^8.

### Guard digits of small arbitrary-precision numbers

WL shows the decimal digits of its binary value, which is rounded to whole
64-bit limbs below the binary point (see `wl_fraction_digits`). Woxi rounds
`N[rational, p]` to that grid too, which makes e.g. `N[3/700000000, 20]`
(`…428572`) match, but for some magnitudes WL's last displayed digit is one
lower than the nearest grid point gives: `N[10^-8/7, 20]` is `…285715` in
Woxi and `…285713` in WL, and `N[10^-9/7, 45]` `…715` against `…714`.
Truncating instead of rounding to the grid fixes those and breaks as many
others, so the rule is not decoded yet.

### Arbitrary-precision `Root` padding

`N[Root[#^3 - # - 1 &, 1], 10]` prints ~40 digits in both engines, but only the
first ~17 are the ones that were asked for; past that the two continuations
disagree. Woxi's is the correct one (checked against the plastic constant to 40
digits) — wolframscript pads with digits of its own. Same for
`N[Root[#^4 - # - 1 &, 1], 15]`.

### `MidDate` on sub-second instants

Woxi carries an instant as one `f64` count of seconds since the epoch, whose
resolution near 1.7e9 is ~2.4e-7 s, so averaging instants that carry fractional
seconds differs from wolframscript's exact mean in the 8th decimal. A midpoint
that lands on a whole second *is* reported exactly, as an integer, matching
wolframscript.

### Image filters accumulate in a different width

`RecurrenceFilter` and `Sharpen` on an `Image` differ from wolframscript by one
Real32 ULP in single taps (`0.7749999761581421` against `0.7750000357627869`):
Woxi accumulates in `f64` and snaps the result to the stored Real32 pixel,
wolframscript accumulates in Real32 throughout. **Not reproducible** without
matching its accumulation order.

### `N` does not push into `Cos[real*Pi]`

`N[Cos[Pi/28], 8]` leaves `Cos[0.0357*Pi]` unevaluated.

### `NIntegrate`, `NSum` and `NProduct` accuracy

`NIntegrate` never matches wolframscript digit for digit on any input — test
against the exact value with a tolerance, not against WL. Specific gaps:

- `NIntegrate[x^(-0.99), {x, 0, 1}]` gives `30.86` for 100. The tail decays
  like `distance^0.01`, past what double precision can represent;
  wolframscript does not converge either (it warns `::ncvb` and reports
  `100.0036`). **Not reproducible.**
- `NIntegrate[1/Sqrt[x], {x, 0, 1}]` is off by 1.8e-5.
- `NProduct` on a slowly convergent infinite product is ~2e-6 off where WL is
  ~3e-11 (`NProduct[1 + 1/n^2, {n, 1, Infinity}]` → 3.676076175562187 against
  3.6760779100585657). Finite products are exact. `NSum` is *more* accurate
  than WL.

### Numeric optimization lands on different points

Wolfram's interior-point method answers `FindMinimum[{x^2, x >= 1}, x]` with
`1.000000013282579` where Woxi returns the exact optimum, and
`NArgMax[{x + y, x^2 + y^2 <= 1}, {x, y}]` with `0.7071076183816036`. A
three-variable equality constraint lands within 1.5e-8 in Woxi rather than
exactly. Neither engine is wrong; bit-equality is not achievable.

```sh
wolframscript -code 'ToString[FindMaximum[{-((x - 2)^2 + (y - 3)^2), 0 < x < 10 && 0 < y < 10}, {{x, 5}, {y, 5}}], InputForm]'
# {1.7763568394002505*^-15, {x -> 2.0000000012019776, y -> 3.0000000006105285}}
woxi eval 'FindMaximum[{-((x - 2)^2 + (y - 3)^2), 0 < x < 10 && 0 < y < 10}, {{x, 5}, {y, 5}}]'
# {0., {x -> 2., y -> 3.}}
```

An objective the optimizer can only sample (a distribution's CDF, say)
diverges the same way: `FindMaximum[{CDF[NormalDistribution[], x - 3],
0 < x < 5}, {x, 1}]` stops at `x -> 4.999998363890543` in WL against Woxi's
`x -> 5.`. Assertions over either have to round.

### `StepMonitor` step counts for an unconstrained `FindMinimum`

Woxi fires the monitor once per accepted step of its own damped Newton
iteration, whatever `Method` is requested, so the *number* of steps (and the
intermediate points) can differ from WL's `"QuasiNewton"` or
`"ConjugateGradient"` runs. Quadratics, which WL also solves in one step,
agree.

### `Erf`'s last ULP, and the far tail of a correlated normal CDF

`Erf[0.7071067811865475]` is `0.6826894921370859` in WL (correctly rounded)
and `0.6826894921370857` here, so anything built on it — `Erfc`, a normal
CDF, `CDF[MultinormalDistribution[…]]` — can differ in the last printed
digit. Woxi's bivariate normal CDF (Genz's algorithm) otherwise agrees to
about 15 digits, but its *relative* error grows in the far tail, where WL
carries extra internal precision:
`CDF[MultinormalDistribution[{0, 0}, {{1., 0.2}, {0.2, 1.}}], {-3., 1.}]`
is `0.001288249693033831` in WL against `0.001288249693033744` here.

### `FindRoot` options and damping schedule

`WorkingPrecision`, `AccuracyGoal` and `PrecisionGoal` are ignored. WL's answer
under `WorkingPrecision` carries iteration residue *past* the requested
precision, so matching it needs its exact iteration and rounding schedule. Its
damping schedule for an *oscillating* Newton also differs
(`FindRoot[x^3 - 2x + 2, {x, 0}, MaxIterations -> 4]` is `0.816037292480469`
in WL, `0.8142908074786004` here), and the three-argument secant form differs
in the last digit.

### `FindRoot` with two starting points keeps WL's iteration residue

A two-point spec (`{x, x0, x1}`) makes WL build its Jacobian from the *secant*
through the two points rather than from a derivative, and stop as soon as the
step is under tolerance — so the answer carries a couple of ULP of that
Jacobian's error even when the exact root is representable:

```sh
wolframscript -code 'FindRoot[{x + y - 1 == 0, x - y - 0.5 == 0}, {x, 0.1, 0.2}, {y, 0.1, 0.2}]'
# {x -> 0.7500000000000002, y -> 0.2500000000000001}
woxi eval 'FindRoot[{x + y - 1 == 0, x - y - 0.5 == 0}, {x, 0.1, 0.2}, {y, 0.1, 0.2}]'
# {x -> 0.75, y -> 0.25}
```

The secant slope of `x + y - 1` between `x = 0.1` and `x = 0.2` is
`1.0000000000000009`, not `1`, which is exactly where WL's two low bits come
from; Woxi uses the symbolic Jacobian and lands on the exact root. The same
shows up on `FindRoot[5 == 50*x^0.6, {x, 0.001, 0.9}]`
(`0.021544346900318832` in WL, `0.021544346900318825` here) and on an opaque
`_?NumericQ` system, where Woxi's Broyden iterate is 1 ULP off WL's.
Matching it means reproducing WL's Broyden update, line search and
convergence test bit for bit — and being *less* accurate than the exact root.

### Arbitrary-precision `Tanh`, `Erf` and `Gamma` carry a different precision tag

Woxi propagates a precision tag through a transcendental with the first-order
relative condition number `|x f'(x) / f(x)|`, which is the rule WL's own
`Sin`/`Cos`/`Sinh`/`ArcTan` markers match to ~15 digits. Four functions do not
follow it in WL:

```sh
wolframscript -code 'ToString[Tanh[N[1/2, 20]], InputForm]'
# 0.46211715726000975850231848364367254873`19.881723602858166
woxi eval 'ToString[Tanh[N[1/2, 20]], InputForm]'
# 0.46211715726000975850231848364367254873`20.070112223892355
```

The *values* agree; only the tag differs. Two things cause it. The smaller one
is arithmetic: Woxi evaluates the condition number and its logarithm in `f64`,
so even where the rule *is* WL's the tag differs in the last digit or two
(`Cosh[N[1/2, 20]]` is `…`20.636277902572626` here against
`…`20.636277902572633`), exactly as for the accuracy-form literal above. The
larger one is the rule itself: WL's `Tanh` tag tracks whichever
internal route it took (it matches the condition number at `x = 1` and the
`Sinh`/`Cosh` quotient's error sum at `x = 1/2`), and its `Erf`/`Erfc` tags
report its own algorithm's error, not the conditioning — at `x = 1/2` it
claims 19.02 digits and its 21st digit onward is already wrong, where Woxi's
value is correct to the last digit shown. `Gamma`/`LogGamma` differ in the
third decimal of the tag for the same reason. **Not reproducible.**

`PolyGamma`, `Zeta` and `HurwitzZeta` at an arbitrary-precision argument
differ in more than the third decimal. Woxi tags `PolyGamma` with the
condition number `|z ψ⁽ⁿ⁺¹⁾(z)/ψ⁽ⁿ⁾(z)|` and `Zeta` with the lowest input
precision. WL's digamma tag even *exceeds* the input's, capped near
`p + log10(1024)`: `PolyGamma[1.9`20]` is tagged `22.45` (Woxi `19.44`), and
`Zeta[0.5`20]` is tagged `18.35` (Woxi `20.`). The values agree.

### Machine `PolyGamma` and `Zeta` are not correctly rounded in WL

Woxi rounds machine `PolyGamma`, `Zeta`, `HurwitzZeta` and `HarmonicNumber`
values from an arbitrary-precision evaluation, so they are correctly rounded.
Wolframscript's are not, and no f64 recurrence/series variant reproduces its
digits (best fit: 34% of 385 probes). Correct rounding matches it on about
half of the `PolyGamma` probes (`PolyGamma[5.5]` is `…751` in WL, `…7512`
correctly rounded) and on 50 of 80 `Zeta[s]` probes. Its machine Hurwitz
zeta is worse: `Zeta[3.5, 0.3]` is 40 ulps off and at negative `s` it loses
whole digits (`Zeta[-5.817415911509695, 0.1295065568974257]` is
`-0.0033354…` in WL, `-0.0033395…` in Woxi and mpmath). WL's machine
`Zeta[s]` at a negative `s` also returns a complex number with an imaginary
part of about `1e-18`, an artifact of its complex reflection formula.
**Not reproducible.**

### Fractional-order `PolyGamma`

`PolyGamma[0.5, 1.9]` is `0.7474244588681002` in WL (its fractional-calculus
continuation of the order) and unevaluated in Woxi. Only integral orders,
exact or inexact, are implemented.

### Display digits past the claimed precision differ

An arbitrary-precision number prints more digits than its precision tag
claims, and the surplus ones are not required to agree:

```sh
wolframscript -code 'ToString[N[Coth[1], 20], InputForm]'
# 1.31303528549933130363616124693084783292`20.
woxi eval 'ToString[N[Coth[1], 20], InputForm]'
# 1.31303528549933130363616124693084783291`20.
```

`coth(1)` is `1.313035285499331303636161246930847832912…`, so Woxi's digit is
the correctly rounded one. `Cot` at 40 digits is the same story.

### `Power` of two arbitrary-precision numbers keeps too many digits

```sh
wolframscript -code 'ToString[SetPrecision[Sqrt[2], 5], InputForm]'
# 1.4142135623730950488`5.301029995663981
woxi eval 'ToString[SetPrecision[Sqrt[2], 5], InputForm]'
# 1.41421356237309504880168872420969807857`5.301029995663981
```

`Power` computes at the operands' bit budget and does not re-truncate the
decimal string to the digit count the *result's* precision tier calls for, the
way `N[…, p]` does. A machine-real exponent is also not contagious:
`N[2, 20]^0.5` is `1.4142135623730951` (machine) in WL and an
arbitrary-precision number here.

### Cross-platform libm differences

The last ULP of `atanh`, `acos`, `asinh`, `tan`, `tanh` and friends differs
between macOS and Linux, so a full-precision string assertion is
platform-dependent: `Cot[0.3]` is `3.232728143765828` on macOS and
`3.2327281437658275` on the Linux of CI, and `Coth[0.8]` ends in `66` versus
`63`. The same 1-ULP FMA difference flips a single 8-bit colour channel in
`ComplexPlot` domain-colouring output at an exact `x.5` boundary.

A test that cares about the last bit therefore has to assert the *property*
rather than the digits: compare the reciprocal against the reciprocal
expression (`Cot[x]` vs `1/Tan[x]`, both evaluated by Woxi, so both carry the
same platform rounding) and pin the magnitude with a scaled integer
(`Round[1000*Cot[x]]`). See `reciprocal_trig_last_bit` in
`tests/interpreter_tests/math/numeric.rs`.

`Cot[-0.7]` is the third argument of that group: `-1.1872418321266796` on
macOS against `-1.1872418321266793` here. Neither library is the accurate one
throughout — against the correctly rounded value glibc wins at `Cot[0.3]`,
macOS at `Cot[-0.7]` and `Coth[0.8]` — so a divergence here is the libm's
last bit and never a formula to correct.

### `$MinNumber` and `$MaxNumber` are undefined

WL's arbitrary-precision range reaches decimal exponents of ±1.36·10^15
(`$MinNumber` is `6.229688249675322…*^-1355718576299610`); Woxi's
arbitrary-precision numbers (astro-float, a 32-bit binary exponent) end near
±6.5·10^8, so the two constants stay symbols. Code that uses `$MinNumber` as
a positive lower bound — `RandomReal[{$MinNumber, x}]` in Demonstration
1DAnd2DSingularWavefronts — gets `RandomReal::unifr` instead of a number.


## Algebra and calculus

### `NSolve` of a polynomial system: multiplicities and order of points

```sh
wolframscript -code 'NSolve[{x^2 y^2 - 2 x^2 - y^2 == 0, x^2 y^2 - x^2 - 3 y^2 == 0}, {x, y}]'
# {{x -> -2.236…, y -> -1.581…}, {x -> 2.236…, y -> 1.581…}, {x -> 2.236…, y -> -1.581…}, {x -> -2.236…, y -> 1.581…}, {x -> 0., y -> 0.} (4 times)}
woxi eval 'NSolve[{x^2 y^2 - 2 x^2 - y^2 == 0, x^2 y^2 - x^2 - 3 y^2 == 0}, {x, y}]'
# {{x -> 2.236…, y -> -1.581…}, {x -> 2.236…, y -> 1.581…}, {x -> 0., y -> 0.}, {x -> -2.236…, y -> -1.581…}, {x -> -2.236…, y -> 1.581…}}
```

The same points, but Wolfram lists the origin with its intersection
multiplicity 4 and orders the points differently.

### `Simplify` does not factor out powers whose exponents differ by an integer

```sh
wolframscript -code 'Simplify[c*(x - c)^(a - 1) - x*(x - c)^(a - 1) + (x - c)^a]'
# 0
woxi eval '…same…'   # c*(-c + x)^(-1 + a) + (-c + x)^a - x*(-c + x)^(-1 + a)
```

wolframscript takes `(x - c)^(a - 1)` out of all three terms, which leaves
`(c - x) + (x - c) = 0`. Woxi's `Times` merges `(x - c)*(x - c)^(a - 1)`
into `(x - c)^a`, but `Simplify` has no step that factors a symbolic
power out of a sum. Because of this, `D[Log[PDF[WeibullDistribution[a, b,
c], x]], b]` does not simplify to `a*(-1 + ((x - c)/b)^a)/b` either.

### `D` of `Min`/`Max` stays in `Derivative[…][Max]` form instead of a `Piecewise`

wolframscript differentiates `Min`/`Max` of a real variable into a
`Piecewise` with rationalized breakpoints, so a flat branch yields the exact
`0`. Woxi keeps `Derivative[0, 1][Max][…]` factors, which evaluate to `0`
numerically but multiply against machine reals into `0.`.

```sh
wolframscript -code 'g[t_] := Min[2.25, Max[1.25, t^3]]; Derivative[2][g][0.5]'
# 0
woxi eval 'g[t_] := Min[2.25, Max[1.25, t^3]]; Derivative[2][g][0.5]'
# 0.
```

### `FullSimplify` nests a sum of radicals under one root

```sh
wolframscript -code 'FullSimplify[(Sqrt[14] + Sqrt[22])*2*Sqrt[8]*2*Sqrt[26]]'
# 32*Sqrt[13*(9 + Sqrt[77])]
woxi eval 'FullSimplify[(Sqrt[14] + Sqrt[22])*2*Sqrt[8]*2*Sqrt[26]]'
# 16*(Sqrt[182] + Sqrt[286])
```

WL rewrites `√7 + √11` as `√(2(9 + √77))` (squaring the sum) and merges it
with the other radicals. Woxi has no such denesting-in-reverse transformation;
part of the open radical-canonicalization class.

### `FullSimplify` of a polynomial: WL's search is not cost-minimal

FullSimplify of a univariate integer polynomial is decoded as a greedy search
by SimplifyCount (`fs_univariate` in `polynomial_ast/simplify.rs`): a genuine
factorization when no more expensive, else the content times the simplified
primitive part, or the low-order terms plus the simplified remainder (nested
Horner form). It matches wolframscript on 175 of 181 probed polynomials. In
the rest wolframscript returns a form that its own `Simplify`SimplifyCount`
rates *more* expensive than the one Woxi finds, so the choice comes from the
order of its transformation search, not from the cost:

```sh
wolframscript -code 'FullSimplify[-6 + 2 x - 3 x^2 - 6 x^3 - 5 x^4]'
# -6 - x*(-2 + x*(3 + x*(6 + 5*x)))      (SimplifyCount 21)
woxi eval 'FullSimplify[-6 + 2 x - 3 x^2 - 6 x^3 - 5 x^4]'
# -6 + x*(2 - x*(3 + x*(6 + 5*x)))       (SimplifyCount 20)

wolframscript -code 'FullSimplify[1 - 5 x - 8 x^2 - x^3 + 6 x^5]'
# 1 + x*(-5 + x*(-8 - x + 6*x^3))        (21)
woxi eval 'FullSimplify[1 - 5 x - 8 x^2 - x^3 + 6 x^5]'
# 1 - x*(5 + x*(8 + x - 6*x^3))          (19)
```

The others: `3x - 5x^2 - 2x^3 - 3x^5`, `-8 - 8x - x^2 + 9x^4`,
`2 - 8x + 5x^3 + 2x^4 - 3x^5` (WL keeps it expanded) and
`-5 + 3x + 4x^2 + 9x^4 - 9x^5` (WL does not split off `9x^3(-1 + x)`). The
sign WL leaves on a split-off remainder follows neither the cost nor the
remainder's leading coefficient consistently; **not reproducible** from
outputs alone. Multivariate and rational-coefficient polynomials still go
through the older candidate pipeline.

### `FactorSquareFreeList` order of a lower-degree non-monic factor

An explicit product takes its factor order from `FactorSquareFree`, which
matches wolframscript. For an expanded polynomial the factors are ordered by
leading coefficient, then constant term, which matches 24 of 25 probes. The
exception: `FactorSquareFreeList[Expand[(x^2 + 3)(2x - 1)^2]]` lists
`{-1 + 2*x, 2}` before `{3 + x^2, 1}` in wolframscript.
Ordering by degree first fixes it but breaks `x` against `-1 + x^2` (WL puts
the latter first), and no single total order fits all 25 — the factor `x`
seems to be placed separately.

### `(-1)^r b^(-r)` does not merge in a product

```sh
wolframscript -code '(-1)^(2/3)*5^(-2/3)'   # (-1/5)^(2/3)
woxi eval '(-1)^(2/3)*5^(-2/3)'             # (-1)^(2/3)/5^(2/3)
```

`Times` merges `a^r b^r` but not a sign radical against a reciprocal one.
Wolframscript skips the merge when another radical is present
(`((-1)^(2/3)*2^(1/3))/5^(2/3)` stays), which the negative-base `Power` rule
mirrors (`(-8/5)^(2/3)` → `4*(-1/5)^(2/3)`).

### `Simplify` does not simplify function arguments

```sh
wolframscript -code 'Simplify[Sin[x^2 + 2 x + 1]]'   # Sin[(1 + x)^2]
woxi eval 'Simplify[Sin[x^2 + 2 x + 1]]'             # Sin[1 + 2*x + x^2]
```

`FullSimplify` does (`simplify_call_arguments`): every compound argument is
simplified and kept when the re-evaluated call gets strictly cheaper, so a
tie such as `f[6 - 5x + x^2]` keeps its argument, as in wolframscript.
`Simplify` has not been switched over.

### `Together` ignores an inexact exponent's sign

```sh
wolframscript -code 'ToString[{Together[E^(-0.5*t) + 1], Together[x^(-1.5) + 1]}, InputForm]'
# {(1 + E^(0.5*t))/E^(0.5*t), (1 + x^1.5)/x^1.5}
woxi eval '{Together[E^(-0.5*t) + 1], Together[x^(-1.5) + 1]}'
# {1 + E^(-0.5*t), 1 + x^(-1.5)}
```

An *exact* negative exponent is a denominator on both sides
(`Together[E^(-t) + E^t]` is `(1 + E^(2*t))/E^t` in each), and Woxi's
common-denominator arithmetic keeps the LCM of exponents that are rational
multiples of one symbol (`E^(-t) + E^(-2*t)` needs `E^(2*t)`, never
`E^(3*t)`). An inexact one is where they part: Woxi leaves it in the
numerator, since its exponent arithmetic is exact-rational. wolframscript
itself is not consistent about this — `Denominator[x^(-1.5)]` is `1` there
even though `Together[x^(-1.5) + 1]` puts `x^1.5` underneath.

### `Together` of an exponential sum sometimes factors the numerator in WL

```sh
wolframscript -code 'Together[1/2 E^(-t) - E^(-2 t) + 1/2 E^(-3 t)]'
# (-1 + E^t)^2/(2*E^(3*t))
woxi eval 'Together[1/2 E^(-t) - E^(-2 t) + 1/2 E^(-3 t)]'
# (1 - 2*E^t + E^(2*t))/(2*E^(3*t))
```

The same sum in a plain variable stays expanded in wolframscript
(`Together[1/(2x) - 1/x^2 + 1/(2x^3)]` is `(1 - 2*x + x^2)/(2*x^3)`), and
so does `Together[E^(-t) - 3 E^(-2 t) + 2 E^(-3 t)]`
(`(2 - 3*E^t + E^(2*t))/E^(3*t)`). The rule that picks the factored form is
not decoded. `Simplify` matches, through its exponential-regrouping candidate.

### Machine `Orthogonalize` differs in last digits

```sh
wolframscript -code 'InputForm[Orthogonalize[{{1., 2, 0}, {0, 1, 3}, {1, 0, 1}}]]'
# {{0.4472135954999579, 0.8944271909999159, 0.}, {-0.13187609467915742, 0.06593804733957871, 0.9890707100936806}, {0.8846517369293829, -0.44232586846469146, 0.14744195615489716}}
woxi eval 'InputForm[Orthogonalize[{{1., 2, 0}, {0, 1, 3}, {1, 0, 1}}]]'
# … 0.0659380473395787 …, -0.4423258684646914 …
```

Woxi runs Gram–Schmidt (twice for the default `Reorthogonalization`),
modified Gram–Schmidt or Householder QR in plain `f64`; wolframscript's
LAPACK-backed routines sum in another order, so a few coordinates differ in
the last place. Signs and zero vectors match.

### `TraditionalForm` orders the terms of a sum its own way

```sh
wolframscript -code 'ToBoxes[#, TraditionalForm] & /@ {1 + x + x^2, 3 Cos[x] + Sin[x], -x + y, Pi + x, a + b + c, x + y^2}'
# x^2 + x + 1, sin(x) + 3 cos(x), y - x, x + π, a + b + c, x + y^2
woxi eval 'ToBoxes[#, TraditionalForm] & /@ {1 + x + x^2, 3 Cos[x] + Sin[x], -x + y, Pi + x, a + b + c, x + y^2}'
# keeps the StandardForm order: 1 + x + x^2, 3 cos(x) + sin(x), -x + y, π + x, …
```

wolframscript puts a univariate polynomial in descending powers, numeric
constants last and a positive term first, yet leaves `a + b + c` and
`x + y^2` alone and puts `sin` before `cos`; no single rule fits all of
these. It matters wherever a TraditionalForm string is used as a key (a
control value saved as `ToString[f, TraditionalForm]`, as in the Newton's
Method Demonstration).

### `CirclePoints[n.]` differs in last digits

```sh
wolframscript -code 'InputForm[CirclePoints[4.]]'
# {{0.7071067811865476, -0.7071067811865475}, {0.7071067811865476, 0.7071067811865475}, {-0.7071067811865475, 0.7071067811865476}, {-0.7071067811865477, -0.7071067811865475}}
woxi eval 'InputForm[CirclePoints[4.]]'
# {{0.7071067811865475, -0.7071067811865475}, … every coordinate ±0.7071067811865475}
```

Woxi numericizes the exact points. wolframscript takes `Cos`/`Sin` of machine
angles starting at `-Pi/2 + Pi/n` in steps of `2 Pi/n`; a fused
multiply-add `k step + start` reproduces 97 of the 108 points for
`n = 8 … 16`, plain accumulation and `start + k step` fewer, so the exact
angle sequence is not decoded.

### `Together` does not pull a sign out of a numerator led by a negative kernel

```sh
wolframscript -code 'ToString[{Together[(-a^n + b)/z], Together[(-E^a + b)/z], Together[(-a + b)/z], Together[(-a^n + b)/2]}, InputForm]'
# {-((a^n - b)/z), -((-b + E^a)/z), (-a + b)/z, (-a^n + b)/2}
woxi eval '{Together[(-a^n + b)/z], Together[(-E^a + b)/z], Together[(-a + b)/z], Together[(-a^n + b)/2]}'
# {(-a^n + b)/z, (-E^a + b)/z, (-a + b)/z, (-a^n + b)/2}
```

With a non-numeric denominator, wolframscript makes the coefficient of the
numerator's leading transcendental kernel (`a^n`, `E^a`; not an algebraic
`Sqrt[a]`) positive and puts the `-1` outside; plain polynomial numerators
and purely numeric denominators are left alone. `(a^n - b^n)/z` stays put
while `(-a^n + b^n)/z` flips, so the first kernel in canonical order leads.
This is what makes `RSolve[{a[n+2] == q a[n+1] + s a[n], a[0] == 0,
a[1] == 1}, a[n], n]` come out as `(-(q - S)^n + (q + S)^n)/(2^n*S)`
instead of `-(((q - S)^n - (q + S)^n)/(2^n*S))`.

### A quotient with a symbolic negative exponent is split, not held

```sh
wolframscript -code 'ToString[{Cancel[(k/x)^(-a)], Simplify[(a*k^a*x^(-1 - a))/(k/x)^a]}, InputForm]'
# {(k/x)^(-a), (a*k^a*x^(-1 - a))/(k/x)^a}
woxi eval '{Cancel[(k/x)^(-a)], Simplify[(a*k^a*x^(-1 - a))/(k/x)^a]}'
# {x^a/k^a, a/x}
```

Woxi reads `(p/q)^(-a)` as `q^a/p^a`, which lets its cancellation see
through the power — the reason `HazardFunction[ParetoDistribution[k, a], x]`
reduces to `a/x` here and stays a ratio of powers there. Both forms are
correct; Woxi's is the simplified one.

### An antiderivative's common denominator is collected, not distributed

```sh
wolframscript -code 'ToString[Integrate[x^2/(2*x^2 + 1), x], InputForm]'
# x/2 - ArcTan[Sqrt[2]*x]/(2*Sqrt[2])
woxi eval 'Integrate[x^2/(2*x^2 + 1), x]'
# (x - ArcTan[Sqrt[2]*x]/Sqrt[2])/2
```

Same value, different common-denominator convention. wolframscript divides
over the rationals from the start; Woxi pseudo-divides over the integers and
takes the factor back out at the end, and `integrate_ast`'s simplify pass
re-collects it even when the rule distributes it. wolframscript is not
consistent about this either — it collects `Integrate[x^4/(3*x^2 + 1), x]`
into `(-3*x + 3*x^3 + Sqrt[3]*ArcTan[Sqrt[3]*x])/27` and distributes the
neighbouring cases — so there is no rule here to copy.

The same convention shows up whenever the result is a `Plus` with a shared
numeric content: `Integrate[(2 + 2*x)/(-3 + 3*x - 5*x^2), x]` is
`-2*((13*ArcTan[…])/(5*Sqrt[51]) + Log[3 - 3*x + 5*x^2]/10)` there and
`(-26*ArcTan[…])/(5*Sqrt[51]) - Log[3 - 3*x + 5*x^2]/5` here.

### `Integrate[f[a*x + b], x]` is not angle-expanded for Sin and Cos

```sh
wolframscript -code 'ToString[Integrate[Sin[x + y], x], InputForm]'
# -(Cos[x]*Cos[y]) + Sin[x]*Sin[y]
woxi eval 'Integrate[Sin[x + y], x]'
# -Cos[x + y]
```

Woxi integrates a shifted argument by substitution, so the antiderivative
keeps the argument it was given. wolframscript applies the angle-sum identity
to the `Sin`/`Cos` results — but not to `Tan`, `Cot` or `Sec`, which agree,
and not to `Integrate[Sin[2*x + 1], x]`, which it leaves as
`-1/2*Cos[1 + 2*x]` while expanding `Integrate[Sin[x + 1], x]`. The same
applies to `Integrate[Log[2*x + 3], x]` and `Integrate[ArcTan[2*x + 1], x]`,
where wolframscript expands the product and drops the resulting constant.


### `Integrate[Log[Sin[x]], …]` is unimplemented

```sh
wolframscript -code 'ToString[Integrate[Log[Sin[x]], {x, 0, 1}], InputForm]'
# (-1/12*I)*(6 + (-6 + Pi)*Pi - (12*I)*Log[2] - 6*PolyLog[2, E^(2*I)])
woxi eval 'Integrate[Log[Sin[x]], {x, 0, 1}]'
# Integrate[Log[Sin[x]], {x, 0, 1}]
```

The antiderivatives are known:

```wolfram
Integrate[Log[Sin[x]], x] ==
  -(x*Log[1 - E^((2*I)*x)]) + x*Log[Sin[x]] + (I/2)*(x^2 + PolyLog[2, E^((2*I)*x)])
Integrate[Log[Cos[x]], x] ==
  (I/2)*x^2 - x*Log[1 + E^((2*I)*x)] + x*Log[Cos[x]] + (I/2)*PolyLog[2, -E^((2*I)*x)]
```

The limit at `x -> 0` contributes `I Pi^2/12`. The blocker is not the value but
the printed form: WL's answer is `Simplify`-collected, and Woxi does not land
on that grouping.

### `Integrate[Sqrt[a ± x^2], x]` stays unevaluated

The antiderivative exists in Woxi already (the definite path uses it), and the
values verify — but the printer reorders `x*Sqrt[1-x^2]` to `Sqrt[1-x^2]*x`
whenever the radical contains the integration variable, and the `Plus` term
order flips too. WL keeps `(x*Sqrt[1-x^2] + ArcSin[x])/2`.

### Irreducible-quartic rational integrals

`Integrate[1/(1+x^4), x]` and its family stay unevaluated: the integrator does
not partial-fraction over quadratics, and WL's output uses ArcTan-reflection
forms (`ArcTan[2/x]` against `ArcTan[x/2]`) that would diverge anyway.

Also still unevaluated: `Integrate[x Log[x]^2, x]`, and
`Integrate[Log[3 x], {x, 1, 2}]` gives `-1 - Log[3] + 2 Log[6]` where WL
combines to `-1 + Log[12]`.

### `Integrate` under assumptions

`Integrate[Exp[-x t], {t, 0, Infinity}, Assumptions -> x > 0]` gives
`x^(-1) - 1/(E^(Infinity*x)*x)` instead of `1/x` — the boundary is an infinite
limit, not a refinement target. Without assumptions
`Integrate[x^n, {x, 0, 1}]` is `(1+n)^(-1) - 0^(1+n)/(1+n)` where WL emits
`ConditionalExpression[(1+n)^-1, Re[n] > -1]`, a form Woxi does not build.
`Sin[a x]/x` and one-sided `1/x^p` improper integrals stay unevaluated.
The Gaussian `Integrate[Exp[-a x^2], {x, -Infinity, Infinity}]` gives
`Sqrt[Pi/a]` against WL's `Sqrt[Pi]/Sqrt[a]`.

### `LaplaceTransform[t^n f(t), t, s]` is unimplemented

The rule `L[t^n f] = (-1)^n d^n/ds^n L[f]` is straightforward and the values
come out right, but WL canonicalizes an even power of a binomial by flipping
the base sign — `(s-a)^(-2)` prints as `(a-s)^(-2)` — keeps `(-1+s^2)^2` where
Woxi factors, and leaves an un-combined sum after the derivative. Plain
`Sin`/`Cos`/`Exp`/`t^n` transforms all match, as does the s-shifting theorem
`L[E^(c t) g(t)]`; `Cosh`/`Sinh` integrands are excluded from the shift because
WL canonicalizes the resulting difference-of-squares denominator inconsistently
(factored for `Cosh`, expanded for `Sinh`).

### `InverseLaplaceTransform` of a complex-pole rational function

A proper rational function with exact coefficients is inverted exactly, off
its partial-fraction decomposition, and the residue sum is regrouped as
`E^(k_min t)` times a polynomial in `E^t` — which reproduces WL for real
poles (`1/((s+1)(s+2)(s+3))` → `(-1 + E^t)^2/(2 E^(3 t))`, repeated poles
included). A **complex-conjugate pole pair** diverges in form only: Woxi
returns the real damped oscillation, WL a sum of complex exponentials.

```sh
wolframscript -code 'InverseLaplaceTransform[1/(s^3 + 2 s^2 + 5 s), s, t]'
# 1/5 + ((I/20)*((-1 + 2*I) + (1 + 2*I)*E^((4*I)*t)))/E^((1 + 2*I)*t)
woxi eval 'InverseLaplaceTransform[1/(s^3 + 2 s^2 + 5 s), s, t]'
# -1/10*(-2*E^t + 2*Cos[2*t] + Sin[2*t])/E^t
```

Both are the same function (`1/5 - E^-t (Cos[2t]/5 + Sin[2t]/10)`). WL's
choice is its `Simplify` acting on complex residues, and it is not
consistent — `(s+2)/((s+1)(s^2+4))` comes back in the real `Cos`/`Sin`
form, `1/(s^2+2 s+5)` in the complex-exponential one. `Simplify` on the
same input agrees with WL, so the divergence is in what each side feeds it.
A real-pole numerator can also land on a different-but-tied form:
`s/((s+1)(s+2)(s+3))` factors the polynomial in `E^t` where WL keeps it
expanded (both cost the same by WL's own `SimplifyCount`).

An improper fraction's `DiracDelta` derivative term also sorts differently
inside the `Plus`: `InverseLaplaceTransform[s^2/(s+1), s, t]` is
`Derivative[1][DiracDelta][t] + E^(-t) - DiracDelta[t]` where WL puts the
`Derivative` term last. Same terms, canonical-order divergence for a
curried head.

### `Fourier*Series` are unimplemented

`FourierSinSeries`, `FourierCosSeries`, `FourierTrigSeries` and `FourierSeries`
echo. The coefficients are easy; WL's output form is unpredictably factored —
`FourierTrigSeries[x, x, 3]` is expanded, `FourierTrigSeries[x^2, x, 2]` is
factored as `Pi^2/3 + 4*(-Cos[x] + Cos[2*x]/4)`, and the same expression
factors differently across the Sin/Cos/Trig variants.

### `Simplify` expands a numerator WL keeps factored per coefficient

```sh
# Simplify[InterpolatingPolynomial[{{0,1},…,{5,q},…,{10,321}}, x]]
wolframscript   # (14400 + 5760*(-10831 + 126*q)*x - … - (-86 + q)*x^10)/14400
woxi eval       # (14400 - 62386560*x + 725760*q*x + … - q*x^10)/14400
```

Same value and same common denominator, but WL pulls the repeated linear
factor `(-86 + q)` out of each coefficient where Woxi leaves the numerator
expanded. `Factor` on the same input returns the expanded form too, so the
missing step is recognizing a shared factor across coefficients rather than
anything about `Simplify` itself.

### `Factor` with `GaussianIntegers`, `Extension` or `Trig`

`GaussianIntegers -> True` (and `Extension -> I`) factors univariate
rational polynomials whose irreducible factors over Q are monic;
non-monic splitting factors, multivariate input, other extensions
(`Extension -> Sqrt[2]`, whose sign normalization is unverified) and
`Trig` stay unevaluated. `IrreduciblePolynomialQ` ignores the options. Composite-modulus `LinearSolve`, n-ary modular
`PolynomialLCM` and composite-modulus `GroebnerBasis` are also left
unevaluated (prime moduli work).

### `PolynomialGCD` is wrong for multivariate input

```sh
wolframscript -code 'PolynomialGCD[x*y, y]'   # y
woxi eval 'PolynomialGCD[x*y, y]'             # 1

wolframscript -code 'PolynomialGCD[2 x y, x^2]'  # x
woxi eval 'PolynomialGCD[2 x y, x^2]'            # 2*x*y
```

The third argument of the 3-argument form is effectively ignored. Multivariate
GCD needs a real algorithm (subresultant PRS over a polynomial ring, or a
modular method); the current implementation is univariate only. This blocks
multivariate `SquareFreeQ`, which stays unevaluated for the same reason.

### `Apart` residues

Two shapes remain: a missing polynomial division when the leading denominator
coefficient is negative (`1/4 + (9+17x+23x^2)/(4*(3-5x-3x^2+4x^3))`), and a
`0 + expr` artifact when the polynomial quotient is zero.

### `PartialFractions` is unimplemented

WL's `PartialFractions[expr, x]` (unlike `Apart`) always splits to **linear**
factors over the algebraic closure, so an irreducible quadratic produces
`(-1)^(1/3)`, `Sqrt` and three-argument `Root` coefficient forms. Only the
rational-factorable case coincides with `Apart`. Matching the coefficient forms
needs the roots-of-unity and `Root` display pipeline.

### `Series[x!, {x, 0, n}]` coefficient form is order dependent

The `x^2` coefficient is WL-canonical `(6*EulerGamma^2 + Pi^2)/12` when it is
the last one and re-folds to `(EulerGamma^2 + Pi^2/6)/2` once it becomes
interior; the `x^3` coefficient diverges similarly. Value correct, form only.

### Symbolic `Sum` is limited to monomials from 1

Non-unit lower bounds, linearity (`Sum[2k, …]`, `Sum[k^2+k, …]`) and an
explicit step with a symbolic bound are all unevaluated or refused. A general
Faulhaber engine is mathematically straightforward and was written, but its
output form does not match: `Sum[k+1, {k, 1, n}]` is `(3*n + n^2)/2` in WL,
while Woxi's `Simplify` gives `(n*(3+n))/2`, and factor ordering diverges
(`(1+Floor[n/2])*Floor[n/2]`).

Clean finite-sum targets still unevaluated:
`Sum[k Binomial[n,k], {k,0,n}]` = `2^(-1+n) n`,
`Sum[k^2 Binomial[n,k], …]` = `2^(-2+n) n (1+n)`,
`Sum[k 2^k, {k,1,n}]` = `2(1-2^n+2^n n)`, and `Sum[Cos[k x], {k,1,n}]`.
`Sum[(-1)^(2n)/n^2]` also fails, because `(-1)^(2n)` is not simplified to 1.

### Infinite-sum gaps

- Rational sums whose residues do not cancel per class have transcendental
  values Woxi cannot produce: `Sum[1/(9n^2-1)]` = `(9 - Sqrt[3] Pi)/18`,
  `Sum[1/(n(2n-1))]` = `Log[4]`. Needs `PolyGamma` at rational arguments.
- `PolyLog` sums with `n^k`, k ≥ 2: `Sum[1/(2^k k^2)]` = `Pi^2/12 - Log[2]^2/2`.
- Divergence messages: `Sum[2^n/n]` and `Sum[1/(n Log[n])]` are silent in Woxi
  where WL reports. WL's own boundary is inconsistent here (bare `2^n` silent,
  `2^n/n` messaged).
- `Product[R(n), {n, n0, Infinity}]` handles integer roots only; rational roots
  (`Product[1-1/(4n^2)]` = `2/Pi`) would need `Gamma` at half-integers.

### `Sum`/`Product` unevaluated echoes keep the source order

`Product[-n, …]` echoes as `0 - n` (a parse artifact) and Woxi keeps the source
`Plus` order (`n+2`) where WL canonicalizes the held body (`2+n`).

### `Limit` returns wrong values for factorial growth

```sh
wolframscript -code 'Limit[n!^(1/n), n -> Infinity]'      # Infinity
woxi eval 'Limit[n!^(1/n), n -> Infinity]'                # 1

wolframscript -code 'Limit[n!^(1/n)/n, n -> Infinity]'    # 1/E
woxi eval 'Limit[n!^(1/n)/n, n -> Infinity]'              # 0

wolframscript -code 'Limit[Log[n!]/n, n -> Infinity]'     # Infinity
woxi eval 'Limit[Log[n!]/n, n -> Infinity]'               # 0
```

Also `Limit[LogGamma[n]/n]` (WL `Infinity`, Woxi 0) and `Limit[(2n)!/n!]`
(WL `Infinity`, Woxi 1). The engine applies a `base^(1/n) → 1` /
`Log[g]/n → 0` shortcut that only holds for polynomial growth. This is the one
place in `Limit` that returns a **wrong value** rather than an unevaluated
call — everything else below only fails to evaluate.

### Three `Limit` shapes at infinity stay unevaluated

```sh
wolframscript -code 'Limit[x^100/Exp[x], x -> Infinity]'        # 0
woxi eval 'Limit[x^100/Exp[x], x -> Infinity]'                  # Limit[x^100/E^x, x -> Infinity]

wolframscript -code 'Limit[Log[Log[x]]/Log[x], x -> Infinity]'  # 0
woxi eval 'Limit[Log[Log[x]]/Log[x], x -> Infinity]'            # Limit[…] unevaluated

wolframscript -code 'Limit[x/(x + Sqrt[x]), x -> -Infinity]'    # 1
woxi eval 'Limit[x/(x + Sqrt[x]), x -> -Infinity]'              # Limit[…] unevaluated
```

Each has its own cause. A polynomial over an exponential resolves up to about
degree 45 and then hits the L'Hôpital depth guard. Nested logarithms need a
`u = Log[x]` substitution — `Limit[Log[u]/u, u -> Infinity]` is already 0.
The `-Infinity` case is excluded on purpose: the leading-order analysis is
gated to `+Infinity` because `x^p` for non-integer `p` is not real to the
left of zero, and extending it needs branch-cut care.

Consequences: `AsymptoticGreater[Exp[x], x^100, …]` and
`AsymptoticLess[Log[Log[x]], Log[x], …]` stay unevaluated too.

Other open limits: `Limit[Sin[1/x], x -> 0]` should be `Indeterminate`,
`Limit[HarmonicNumber[n]/Log[n]]` should be 1, and
`Limit[HarmonicNumber[2n] - HarmonicNumber[n]]` should be `Log[2]`.
`Limit[Binomial[2n,n], n -> Infinity]`, `Limit[1/Binomial[2n,n]]` and
`Limit[Binomial[2n,n]^(1/n)]` **hang** (the numeric probe evaluates `Binomial`
at n = 10^6), as does the literal-`Sum` spelling of the harmonic-minus-log
limit.

### `Residue` and `Simplify` around Gamma poles

A `Gamma` pole model multiplied by another pole at a non-zero point —
`Residue[Gamma[z]/(z+1), {z, -1}]`, WL `-1 + EulerGamma` — is refused rather
than risk the `Simplify` slowness above.

### `RSolve` first-order linear inhomogeneous

`RSolve[a[n] == c a[n-1] + d && a[k0] == v0, a[n], n]` with constant `c ≠ 1`
returns unevaluated. The closed form is trivial; WL's *display* varies
case by case through power folding, with no clean subset:

| recurrence | wolframscript |
| --- | --- |
| c=2, d=1, a[0]=2 | `-1 + 3*2^n` |
| c=2, d=1, a[0]=3 | `-1 + 2^(2 + n)` (the coefficient folded into the exponent) |
| c=2, d=1, a[0]=4 | `-1 + 5*2^n` |
| c=2, d=5, a[0]=0 | `5*(-1 + 2^n)` (common factor pulled out) |
| c=3, d=2, a[0]=5 | `-1 + 2*3^(1 + n)` (partial fold) |
| c=5, d=3, a[0]=10 | `(-3 + 43*5^n)/4` |

`RSolve` of the logistic map at r=2 is deliberately left unevaluated for the
same reason: WL displays an internally rewritten rational form
(`(-2^2^(1+n) + 5^2^n)/(2*5^2^n)`) rather than the clean `(1 - (1-2c)^2^n)/2`.

### `Casoratian` is unimplemented

The semantics are fully decoded, but every form needs exponential collapse
Woxi's `Simplify` lacks (`{2^n, 3^n}` → `6^n`, `{n!, 2^n}` → a Gamma form,
`{Sin[n], Cos[n]}` → `-Sin[1]`), and WL is internally inconsistent
(`{2^n,3^n,5^n}` → `5^n*6^(1+n)`, not `6*30^n`). Only the first-order-system
form (`Det[A]^n`) is form-stable.

### `FindGeneratingFunction` is unimplemented

Not a clean rational fit: `{1,1,1,1,1}` gives a Padé artifact, `{1×7}` and
`{1×10}` give *different families* of answer, and the display switches between
raw power and `Together` form. **Not reproducible.**

### `MultipleZeta` reduction

WL reduces multiple zeta values through at least weight 8 using
`MultipleZeta[{5,3}]` as a basis element. Depth 1, leading 1 and the empty
argument are easy; matching the reduction extent needs full MZV datamine
tables.

### Higher-rank `Curl`

Scalar in 3D, vector in 4D and rank-2 tensor in 3D — the valid higher-rank
antisymmetric forms — stay unevaluated. WL returns them as `SymmetrizedArray`
(or a collapsed 0). Needs exterior derivative plus Hodge dual.

### A coupled ODE system is solved numerically, so its constants are inexact

`DSolve[{eq1, …}, {y1[t], …}, t]` eliminates the system with an operator
determinant and finds the characteristic roots in machine floating point,
recovering exact-looking coefficients (`1/2`, `Sqrt[3]`) from them at the
end. For a *first-order homogeneous* system the answer is re-parametrized
to `Y(t) = MatrixExp[A·t]·C` — wolframscript's basis, in which `C[k]` is the
initial value of the k-th variable — and matches it term for term, secular
`t·E^(λt)` terms for defective systems included. Three shapes still differ:

- Coefficients that are not a small rational or the square root of one stay
  as machine reals, and a radical that both a numerator and a denominator
  carry is not cancelled (`(Sqrt[3]*E^(2*t) − …)/(3*Sqrt[3]*E^t)` where WL
  writes `(E^((3*t)/2) − …)/(3*E^(t/2))`).
- A system with inexact coefficients (a Foucault-pendulum-style rotational
  coupling with a `0.1` rate) gets a clean trigonometric answer here. WL's
  own is an unsimplified tangle of `4.4*^-17`-scale complex noise, so there
  is no form to match.
- A system with nonzero constant forcing gets its constant particular
  solution (a polynomial one when the characteristic root 0 makes `M(0)`
  singular, as for a charged particle in crossed fields) plus the
  eigenvector basis, where WL emits an unsimplified variation-of-parameters
  expression in its `MatrixExp` basis.
- A higher-order homogeneous system uses the eigenvector basis too; WL's
  constants are the initial values of `y1, y1', …, y2, …` in that order.

### Second-order PDEs with a lower-order term

`DSolve[u_xx + u_y == 0]` (the heat equation) is unevaluated; WL falls back to
a *particular* solution with `C[1] … C[8]` after `DSolve::lpdeprtclr`. With
`a == 0` or `c == 0` WL also writes the characteristics unnormalised
(`C[1][x - y] + C[2][x]`) rather than as `λ x + y`.

### `NDSolve`'s PDE branch is 1-D

This entry used to say "`NDSolve` covers ODEs only" — no longer accurate.
1-D parabolic PDEs (single or coupled reaction-diffusion-convection systems,
Dirichlet or Neumann boundaries) solve via the method of lines:

```wolfram
NDSolve[{D[u[x,t],t] == D[u[x,t],x,x], u[x,0] == Sin[Pi x],
         u[0,t] == 0, u[1,t] == 0}, u, {x,0,1}, {t,0,1}]
```

returns a real `InterpolatingFunction`. A second-order-in-time
(hyperbolic/wave) evolution equation — `D[u[t,x],t,t] == …`, needing an
extra initial *velocity* condition `D[u,t][t0,x] == g[x]` alongside the
ordinary initial value — is order-reduced to the first-order system
`D[u,t] == v`, `D[v,t] == w` and solved the same way, even when the
right-hand side itself contains an implicit/mixed derivative of the unknown
(e.g. `D[u[t,x],x,x,t,t]`): the acceleration field `w` is eliminated by a
tridiagonal solve each method-of-lines step rather than an explicit
formula, so long as the right-hand side is linear in it. This is what a
Wolfram Demonstration like
[*A Passive Cochlear Model*](https://demonstrations.wolfram.com/APassiveCochlearModel/)
needs, and Woxi Studio can now render its `Manipulate`. What's still
missing:

- The hyperbolic branch only supports a single dependent function — no
  coupled hyperbolic systems (the parabolic branch supports coupled
  systems).
- `DirichletCondition` exists only as a symbol; Dirichlet conditions must be
  written as plain equalities. `NeumannValue[g, x == x0]` terms are lowered
  to Neumann flux conditions (`n·(c ∇u) = g`), and the resulting solutions
  take the space variable first, then time (unverified against
  `wolframscript` for the `Method -> "FiniteElement"` form).

On the symbolic side `DSolve` recognises three first-order two-variable PDE
shapes; Laplace, which WL solves as `C[1][I x + y] + C[2][-I x + y]`, is not
among them.

`NDSolve`'s DAE support handles index-1 constraints: those that solve
explicitly for an unknown are eliminated, and the remaining derivative-free
equations are solved together by Newton iteration when they are as many as
the never-differentiated unknowns they mention. Index ≥ 2 constraints, and
constraints whose unknown is also differentiated, are not covered. The one exception is a
system whose algebraic equations leave a single degree of freedom, closed by
first-order derivative equations (a unit-speed condition): it is traced as an
implicit curve by arclength continuation, and ends early where the curve
leaves the region the equations are defined on.

### `NDSolve` of a DAE mixing algebraic unknowns into differential equations

A distillation-column model (Housam Binous's Demonstrations: holdup balances
with `T[i]'[t]` and `y[i]'[t]` terms, vapor-liquid equilibria, and flows
`L[t]`, `V[t]`, `D[t]` that only appear algebraically) returns `NDSolve`
unevaluated. Index reduction (differentiating an algebraic equation whose
unknowns are all differentiated elsewhere) and initial values for algebraic
unknowns are handled, but an equation combining differentiated unknowns with
the Newton-solved algebraic ones does not compile. Because `sol` then holds
the unevaluated `NDSolve`, the Demonstrations' `Plot[x[11][t] /. sol, …]`
reruns it at every sample and times out.

Status 2026-11-01. Columns without an energy balance now solve
(SolventSwapDistillation, TernaryBatchRectification, the residue-curve
models): the stiff Rosenbrock fallback runs when the right-hand side Newton
solves the stage temperatures. DistillationOfAnEthanolWaterMixture is still
open. Its temperatures are fixed by the bubble-point equations and also
differentiated in the energy balances (`T[0]'[t]`, `T[0]` not even listed as
an unknown), which makes it a higher-index DAE. WL reduces the index and
computes consistent initial values (`NDSolve::ivres`).

Two experiments, both reverted:
- Cancelling terms in the algebraic equations after the explicit
  eliminations exposes the overall balance `-10 + B[t] + Di[t] == 0`. Woxi
  had assigned it to `V2`, an unknown it no longer contains.
- Letting the stiff fallback run with order-0 unknowns then integrates. But
  from the inconsistent start it drifts away from WL's steady state
  (`x[1][50]` 0.0666 against 0.8375), so the result is wrong rather than
  missing.

### `NDSolve` uses a fixed grid

Woxi integrates on 1000 nominal RK4 steps with bisection refinement;
wolframscript adapts. `Length[(y /. s[[1]])[[2]]]` is 1001 against 13. Note
that stock wolframscript's *default* tolerances can be **less** accurate than
Woxi's here — get a converged reference before treating a difference as a bug.

### Zero-dimensional polynomial systems are expressed through one `Root`

When no equation of a polynomial system can be solved for an unknown in
closed form, `Solve` and `Reduce` take a lexicographic Groebner basis and
return `x == Root[R & , k, 0]` with every other unknown as a polynomial in that
root. The solutions are right (they satisfy the system to machine precision
once numericized), but wolframscript gives each unknown its own `Root` of a
univariate polynomial, so the exact printed form differs. Systems whose basis is
not in this shape (repeated roots, positive dimension) stay unevaluated. Machine
precision `N` of the polynomial-in-`Root` values loses about 7 digits to
cancellation (residuals around `1e-9`), where wolframscript raises its working
precision.

`Reduce[{x^3 + y^3 + x y == 3, x^2 + x y^2 == 2}, {x, y}, Reals]` returns a
single branch with a `Sqrt` that is not real for that root, although three real
solutions exist (`Solve[..., Reals]` finds them).

### `Reduce` does not eliminate quantifiers

```sh
wolframscript -code 'Reduce[Exists[y, x == y^2], x, Reals]'      # x >= 0
woxi eval 'Reduce[Exists[y, x == y^2], x, Reals]'                # Reduce[Exists[y, x == y^2], x]

wolframscript -code 'Reduce[ForAll[y, x + y^2 >= x], x, Reals]'  # True
woxi eval 'Reduce[ForAll[y, x + y^2 >= x], x, Reals]'            # Reduce[ForAll[y, x + y^2 >= x], x]
```

`Exists` and `ForAll` are parsed but never eliminated. Note the unevaluated
form also drops the `Reals` domain argument, which is a separate bug in the
echo path.

### Other `Reduce` and `FindInstance` divergences

- **Complex-domain `Abs` is a wrong answer**: `Reduce[Abs[x] < 3, x]` returns
  `False` although solutions exist. WL gives a `Re`/`Im` decomposition, since
  the default domain is `Complexes`.
- Multi-symbol parametric coefficients: `Reduce[a b x == c, x]` gives an
  incomplete `x == c/(a*b)` where WL enumerates three sub-cases. The
  single-symbol case is implemented, and only for the default domain.
- The two-argument `x ∈ Integers` membership form does not enumerate; the
  three-argument `Reduce[…, x, Integers]` form does.
- Disjunct ordering: `Reduce[x > 0 || x < -1, x]` — WL sorts the regions,
  Woxi keeps input order.
- `Reduce[Sin[x] == 1, x]` — WL puts the `Element` condition as a top-level
  `And`, Woxi wraps it in `ConditionalExpression`.
- `Reduce[…, Modulus -> n]` ignores the modulus for a multivariate **nonlinear**
  system (univariate and linear systems work). `Solve` refuses that case rather
  than answer wrongly.
- `FindInstance` picks a different, still-valid instance: `x^2 == 2` gives
  `Sqrt[2]` against WL's `-Sqrt[2]`; `x^2 + y^2 == 1` over the reals gives
  `{x -> -1, y -> 0}` against `{x -> 1, y -> 0}`; `x^2 > 4` gives `-100`
  against `-4`. **Not reproducible.**
- `FindInstance` for linear inequalities over the reals solves a linear
  program that maximizes the slack of the strict inequalities, so a bounded
  interval gives its midpoint (`3 < x < 10` → `13/2`, as in WL) and a
  non-strict bound its endpoint (`x >= 3` → `3`, as in WL). WL samples other
  regions pseudo-randomly: `x > 3` gives `30` (Woxi `4`), `1/3 < x < 1/2`
  gives `229/607` (Woxi `5/12`), and the extra instances of
  `FindInstance[…, n]` differ. **Not reproducible.**
- A single Boolean instance (`FindInstance[…, Booleans]`,
  `SatisfiabilityInstances[…, vars]` or `…, 1]`) comes from a different
  solver in WL than the instance lists: with several solutions, Woxi returns
  the first of the list for 2 or more (`Xor[p, q, r]` → `{True, True, True}`)
  where WL returns `{False, True, False}`. Lists of 2 or more instances, and
  any unique solution, match.
- `FindInstance` over the integers falls back to a bounded search when `Solve`
  cannot decide: every variable is walked outwards from zero (0, 1, −1, 2, −2,
  …) within a fixed evaluation budget. That reproduces wolframscript's answer
  for the small instances (`x^5 + y^5 + z^5 == w^5 && x > 0` → `{1, 0, 0, 1}`)
  but cannot reach a large one — Euler's sum-of-powers counterexample
  `27^5 + 84^5 + 110^5 + 133^5 == 144^5` stays unevaluated where wolframscript
  finds it.

### `Reduce` over `Rationals` omits the `Element` conjunct

Only a discrete solution set stands on its own over the rationals; an interval
has to say that just its rational points are meant, and wolframscript writes
that as a leading `Element`. Woxi drops it, and the surrounding rational
surface diverges with it:

```sh
wolframscript -code 'Reduce[x >= 2, x, Rationals]'       # Element[x, Rationals] && x >= 2
woxi eval 'Reduce[x >= 2, x, Rationals]'                 # x >= 2

wolframscript -code 'Reduce[True, x, Rationals]'         # Element[x, Rationals]
woxi eval 'Reduce[True, x, Rationals]'                   # True

wolframscript -code 'Reduce[x != 2, x, Rationals]'       # Element[x, Rationals] && (x < 2 || x > 2)
woxi eval 'Reduce[x != 2, x, Rationals]'                 # x != 2

wolframscript -code 'Reduce[x >= 2 && x <= 2, x, Rationals]'  # x == 2
woxi eval 'Reduce[x >= 2 && x <= 2, x, Rationals]'            # 2 <= x <= 2
```

Several targets share one `Element` over an `Alternatives`, which the integer
domain gets wrong the same way even though it does emit the conjunct:

```sh
wolframscript -code 'Reduce[x > 2 && y < 1, {x, y}, Integers]'
# Element[x | y, Integers] && x >= 3 && y <= 0
woxi eval 'Reduce[x > 2 && y < 1, {x, y}, Integers]'
# Element[x, Integers] && Element[y, Integers] && y <= 0 && x >= 3
```

The curated oracle case `Reduce[1/3 < x && x <= 5/7, x, Rationals]` in
`tests/reduce/oracle.rs` is the recorded instance; it is the one failure
`make test-slow` reports on a machine with `z3` installed and network access.

### `Reduce` leaves a joint multi-variable constraint unevaluated

Constraints that share a variable are reduced together, and the joint case is
only implemented for what the linear engines cover. A shared-variable group
that needs a cylindrical decomposition stays unevaluated:

```sh
wolframscript -code 'Reduce[a < b && b < 2 && x > 0, x]'
# a < 2 && Inequality[a, Less, b, Less, 2] && x > 0
woxi eval 'Reduce[a < b && b < 2 && x > 0, x]'
# Reduce[a < b && b < 2, {a, b}] && x > 0

wolframscript -code 'Reduce[a x > 0, x]'    # (a < 0 && x < 0) || (a > 0 && x > 0)
woxi eval 'Reduce[a x > 0, x]'              # Reduce[a*x + 0 > 0, x]
```

Independent constraints *are* separated and reduced one group at a time, so
`Reduce[x > 1 && a^2 < 4 && b^2 < 9, x]` matches.

### Solve of a system eliminated through a fractional power finds nothing

```sh
wolframscript -code 'Length[Solve[{1 == u*t/p, 1000*(t - 400) == (10000 - u^2)/2, t == p^(-2/7)}, {p, u, t}]]'
# 10
woxi eval 'Length[Solve[{1 == u*t/p, 1000*(t - 400) == (10000 - u^2)/2, t == p^(-2/7)}, {p, u, t}]]'
# 0
```

The kernel solves `t == p^(-2/7)` for `p`, substitutes, and reduces the system
to a degree-10 polynomial in `t`. Woxi's elimination does not invert the
fractional power. Written as `p == t^(-7/2)`, the system is solved, but through
a degree-20 polynomial in `Sqrt[t]` whose wrong-branch roots are not all
discarded (17 solutions instead of 10). This blocks the Demonstration
"Compressible Flow Through a Nozzle/Diffuser".

### Solve orders the solutions of `Norm` equations differently

```sh
wolframscript -code 'Solve[{Norm[{x, y}] == 1, Norm[{x, y} - {1, 1}] == 0.6}, {x, y}]'
# {{x -> 0.9137715508089904, y -> 0.4062284491910096}, {x -> 0.4062284491910096, y -> 0.9137715508089904}}
woxi eval 'Solve[{Norm[{x, y}] == 1, Norm[{x, y} - {1, 1}] == 0.6}, {x, y}]'
# {{x -> 0.40622844919100964, y -> 0.9137715508089904}, {x -> 0.9137715508089904, y -> 0.40622844919100953}}
```

Written as polynomials (`x^2 + y^2 == 1`), both list `x -> 0.406…` first.
The kernel solves the radical form by another route, which also skips the
`Solve::ratnz` message (Woxi matches that part). Woxi squares the radicals
away and keeps the polynomial order.

### Solve over a system of two Abs equations gives no solutions

```sh
wolframscript -code 'Solve[{Abs[x] == 2, Abs[y] == 3}, {x, y}]'
# {{x -> -2, y -> -3}, {x -> 2, y -> -3}, {x -> -2, y -> 3}, {x -> 2, y -> 3}}
woxi eval 'Solve[{Abs[x] == 2, Abs[y] == 3}, {x, y}]'
# {}
```

One `Abs` equation is solved correctly, and a single `Abs` equation narrowed by
an inequality is too. It is the multi-variable elimination that cannot take two
of them apart and reports the system unsatisfiable, so `ToRules` turns `False`
into an empty list.

### Solve orders the negative root first in a two-variable system

```sh
wolframscript -code 'Solve[{x^2 + y^2 == 1, y == 0}, {x, y}]'
# {{x -> 1, y -> 0}, {x -> -1, y -> 0}}
woxi eval 'Solve[{x^2 + y^2 == 1, y == 0}, {x, y}]'
# {{x -> -1, y -> 0}, {x -> 1, y -> 0}}
```

The single-variable case agrees (`Solve[x^2 == 4, x]` is `{{x -> -2}, {x -> 2}}`
in both), so the system path sorts where wolframscript does not.

### Integer lattices with congruences are printed in canonical form

`Reduce`/`Solve` over `Integers` parametrize a linear system with congruences
(`Mod[e, m] == r`) as one lattice in Hermite normal form. wolframscript
substitutes its way through the system and leaves the substitution visible:

```sh
wolframscript -code 'Reduce[Mod[x, 6] == 1 && x + y == 0, {x, y}, Integers]'
# Element[C[1], Integers] && x == 1 + 6*C[1] && y == 5 + 6*(-1 - C[1])
woxi eval 'Reduce[Mod[x, 6] == 1 && x + y == 0, {x, y}, Integers]'
# Element[C[1], Integers] && x == 1 + 6*C[1] && y == -1 - 6*C[1]
```

The sets agree, and so does the form whenever wolframscript's substitution
already ends in canonical form
(`Reduce[Mod[x + 2 y, 4] == 1 && x == 3 y + 1, {x, y}, Integers]`).
When the congruence leaves more than one parameter free, wolframscript
enumerates residue classes as a disjunction
(`Reduce[Mod[x + y + z, 5] == 2 && x == y, {x, y, z}, Integers]`); Woxi
has no such form and does not reduce these.

### `Sort` of a sum against a product with an imaginary or reciprocal factor

```sh
wolframscript -code 'Sort[{1 + I x, I x}]'
# {1 + I*x, I*x}
woxi eval 'Sort[{1 + I x, I x}]'
# {I*x, 1 + I*x}
```

Same for `1 + x/Log[3]` against `x/Log[3]`, but `Sort[{1 + 2 x, 2 x}]`
and `Sort[{1 + 2 I Pi x, 2 I Pi x}]` put the product first in both. It
shows in the order of `Solve`'s solution families
(`Solve[9^x - 4 3^x + 3 == 0, x]`).

### `N` of an overflowing exact expression comes back at machine precision

```sh
wolframscript -code 'N[Sqrt[10^400 + 1]]'
# 1.`16.255619765854984*^200
woxi eval 'N[Sqrt[10^400 + 1]]'
# 1.*^200
```

When a part of an exact expression overflows a machine number, Woxi
evaluates the whole expression with 30 digits and returns a machine real if
the value fits. wolframscript carries the overflowed part as a big number
of machine precision, so the result is a big number too.

### `Root` of a reducible polynomial keeps the whole polynomial

```sh
wolframscript -code 'Root[2 #^5 - 4 # - 2 &, 1]'
# -1
woxi eval 'Root[2 #^5 - 4 # - 2 &, 1]'
# Root[-1 - 2*#1 + #1^5 & , 1, 0]
```

The polynomial is `2 (1 + #) (#^4 - #^3 + #^2 - # - 1)`; wolframscript
reduces to the factor the `k`-th root belongs to. Woxi only does that when
every factor has degree 2 or less.

### `Solve` over `Integers` does not parametrize nonlinear equations

```sh
wolframscript -code 'Solve[x^2 == 4 y, {x, y}, Integers]'
# {{x -> ConditionalExpression[4*C[1], Element[C[1], Integers]], y -> ConditionalExpression[4*C[1]^2, Element[C[1], Integers]]}, {x -> ConditionalExpression[2 + 4*C[1], Element[C[1], Integers]], y -> ConditionalExpression[1 + 4*C[1] + 4*C[1]^2, Element[C[1], Integers]]}}
woxi eval 'Solve[x^2 == 4 y, {x, y}, Integers]'
# {}
```

`Reduce` gives the real solution `y == x^2/4` here, and `Solve` keeps only
integer-valued rules of the real solution, which this one is not.

### Solving for a variable divided by a symbol keeps a negated sum split

```sh
wolframscript -code 'Solve[-x/a == l - p, x]'   # {{x -> -(a*(l - p))}}
woxi eval 'Solve[-x/a == l - p, x]'             # {{x -> -(a*l) + a*p}}
wolframscript -code 'Solve[3^(-x/a) == 5, x]'
# {{x -> ConditionalExpression[a*(((2*I)*Pi*C[1])/Log[3] - Log[5]/Log[3]), …]}}
woxi eval 'Solve[3^(-x/a) == 5, x]'
# {{x -> ConditionalExpression[-(a*(((-2*I)*Pi*C[1])/Log[3] + Log[5]/Log[3])), …]}}
```

Numeric slopes (`-x`, `-2 x`, `-x/2`) and a symbolic one without the minus
agree. The exponential case is the same linear step: `b^(linear) == c` is
solved as `linear == Log[c]/Log[b] + 2 I Pi C[1]/Log[b]`, flipping `C[1]` for
a negative slope.

### `Roots` root ordering

`Roots[x^2 - 3x + 2 == 0, x]` is `x == 2 || x == 1` in Woxi and
`x == 1 || x == 2` in WL. But WL's order is not a clean sort — `x^2-9` gives
`3, -3`, `x^4-5x^2+4` gives `-2,-1,1,2`, `x^2-2` gives `Sqrt[2], -Sqrt[2]`,
while `x^2-2x-1` keeps `Solve`'s order. It depends on which internal
root-isolation path the degree takes. Woxi's `Solve` ordering does match
exactly; only `Roots` diverges. **Not reproducible.**

Related and still open: `Roots[x^n == c, x]` uses ω^k generation order starting
at k=1 when the radicand is not a perfect power, and `Root` of a reducible
polynomial with cyclotomic factors prints `a+bI` (`Root[x^6-1, 3]` =
`(-1 - I Sqrt[3])/2`) where Woxi keeps the `Solve` form `-(-1)^(1/3)`.

### Inexact polynomial solving

With any inexact coefficient WL's `Solve` **is** `NSolve` — purely numeric, no
exact factoring, so `Solve[x^3 - 1.5 x^2 + x - 1.5 == 0, x]` gives
`1.4999999999999996` rather than `3/2`. Woxi agrees on that model, and its
Durand–Kerner roots agree to 1–2 ULP — but never assert full digits. The one
case WL still gets exactly right is a **triple root**
(`x^3 - 2.25 x^2 + 1.6875 x - 0.421875` → `{0.75, 0.75, 0.75}`), which it
resolves at raised internal precision; f64 Durand–Kerner cannot.

### `Minimize` does not detect an unattained infimum

`Minimize[1/(1+x^2), x]` returns a bogus large-x numeric instead of WL's
`natt` message and 0.

Unbounded non-convex polynomials go the same way: `Minimize[x^2 - y^2,
{x, y}]` reports the saddle `{0, {x -> 0, y -> 0}}` where WL says
`-Infinity`. And where the minimum is attained along a whole curve
(`Minimize[(x y - 3)^2 + 1, {x, y}]`), the value matches but Woxi reports
the point with the free variable at 1 (`{x -> 1, y -> 3}`), WL
`{x -> -1, y -> -3}`.

### Symbolic `Fit` and `LinearModelFit`

`LinearModelFit[…]["BestFit"]` drops the `0. +` constant term wolframscript
keeps, and `Fit[{1,2,3,4},{1,x},x]` gives a clean `0. + 1. x` where WL keeps
`5.07*^-16 + 0.9999999999999999 x` — Woxi is the less noisy one.

`Fit` with a symbolic data value fits symbolically in wolframscript
(`Fit[{a, 1, 2}, {1, x}, x]` is linear in `a`); Woxi fails with "could not
convert data value to a number". `LinearModelFit` gives
`LinearModelFit::desmat` for such data in both. `LinearModelFit[…, Weights
-> w]` stays unevaluated in Woxi.

### `GeneralizedLinearModelFit` and `LogitModelFit` stop short of the estimate

```sh
wolframscript -code 'GeneralizedLinearModelFit[{{1,2},{2,3},{3,7},{4,9}}, x, x, ExponentialFamily -> "Poisson"]["BestFitParameters"]'
# {0.22480147381937685, 0.5104799964404124}
woxi eval '…same…'   # {0.22480147112192…, 0.51047999706…}, the maximum-likelihood estimate
```

wolframscript's iteration ends about 1e-9 (relative) away from the
estimate, or 1e-5 with `LinkFunction -> "IdentityLink"` or with
`ProbitModelFit` (O-ring data: wolframscript `{8.774897560207174,
-0.13509567688143528}` has log-likelihood -10.188869659392, Woxi's
`{8.7749543…, -0.1350965…}` -10.188869659266). Woxi iterates to
the estimate, which solving the score equations at 30 digits confirms.
With perfectly separated binomial data the estimate does not exist, and
both report large finite coefficients that differ (`LogitModelFit[{{0,
0}, {1, 0}, {2, 0}, {3, 1}, {4, 1}, {5, 1}}, {1, x}, x]`: wolframscript
`{-98.58, 39.43}`, Woxi `{-246.8, 98.71}`). Neither R's `glm.fit` starting
values nor its stopping rule reproduce wolframscript's digits.

### `GeneralizedLinearModelFit` has no `"NullDeviance"` for most families

```sh
wolframscript -code 'GeneralizedLinearModelFit[{{1,1.},{2,2.},{3,4.},{4,4.}}, x, x]["NullDeviance"]'
# 3.65505327700101
woxi eval '…same…'   # FittedModel::elmntav, the property is left out
```

The constant model's deviance is `Total[(y - Mean[y])^2] = 6.75`, which
no constant mean can bring down to 3.655; with `LinkFunction -> "LogLink"`
wolframscript reports yet another value (6.5635). The Poisson and binomial
null deviances are the constant model's and Woxi reports them; the
Gaussian, Gamma and inverse Gaussian ones are left out rather than
reported differently.

### `LinearModelFit` estimates differ in the last bits

```sh
wolframscript -code 'LinearModelFit[{{0, 1}, {1, 0}}, x, x]["BestFitParameters"]'
# {1., -1.}
woxi eval '…same…'   # {0.9999999999999999, -1.}
```

Woxi solves the least-squares problem by Householder QR. None of LAPACK's
`gelsd`, `gelsy` or `gelss`, the normal equations or the pseudo-inverse
reproduces wolframscript's digits on every input. With as many points as
parameters, the difference becomes visible: when wolframscript's residuals
come out exactly zero, its standard errors are `Indeterminate` (with
`Infinity::indet`), while Woxi's tiny nonzero residuals give
`ComplexInfinity`.

### Constrained and weighted fitting find different local minima

On a non-convex constrained problem the two engines land on *different valid*
local minima — for a sine fit Wolfram lands on the constraint boundary while
Woxi finds an interior point with a smaller residual. Compare the picture, not
the parameters.

### `RSolve` of a Riccati recurrence is written differently

`RSolve[k0 + k1 x[n] + k2 x[n + 1] + x[n] x[n + 1] == 0, x[n], n]` gives
the same function as wolframscript, with `C[1]` meaning the same number,
but not the same expression: wolframscript's carries large internal
scale factors (`4.596043240755094*^15*0.5447870971471407^n`) where Woxi
writes `(λ1 - s) b1^(1 + n)` with the bases `b = -1/λ` of the
characteristic roots. As in wolframscript, `C[1]` goes with one root for
numeric coefficients and with the other for symbolic ones.

### `NDEigensystem` digits and scope

Woxi discretizes a one-dimensional operator `a u'' + c(x) u` by finite
differences on 2000 intervals and refines each eigenvalue by its Rayleigh
quotient; wolframscript's default finite-element mesh is coarser, so its
values are less accurate (`{1., 4.0001, 9.0006}` for `-u''` with Dirichlet
ends on `[0, π]`, where Woxi has `{1., 4., 9.}`; both agree to 1e-4 with
wolframscript on a refined mesh). The eigenfunctions are
`InterpolatingFunction`s over the grid rather than over a finite-element
mesh. Operators with a `u'` term, a variable `u''` coefficient, more than
one dimension, or inhomogeneous conditions stay unevaluated.

### `TriangleCenter` of a symbolic triangle is not simplified

With numbers for vertices both bring each coordinate together and agree.
With symbols wolframscript cancels and factors further:

```sh
wolframscript -code 'TriangleCenter[{{0, 0}, {a, 0}, {b, c}}, "Orthocenter"]'  # {b, ((a - b)*b)/c}
woxi eval 'TriangleCenter[{{0, 0}, {a, 0}, {b, c}}, "Orthocenter"]'            # the uncancelled barycentric quotient
```

The rule is not `Together[ExpandAll[…]]` alone (that matches the
circumcenter and nine-point center but leaves `a*b - b^2` where
wolframscript writes `(a - b)*b`). It probably needs a `Factor` that
pulls algebraic content (`Factor[4 + 4 Sqrt[10]]` is `4*(1 + Sqrt[10])`
in wolframscript and unchanged in Woxi).

### Control systems: response forms and plot details

- `OutputResponse[model, u, t]` gives the inverse Laplace transform of
  `G(s) U(s)` through `Together`; wolframscript's forms come from its own
  state-space integration (`{1 + (2*(-1 + E^t))/E^t}` where Woxi has
  `{(1 + 3*E^t)/E^t}` for `(s + 3)/(s + 1)` and a unit step). Over a time
  span both are `{InterpolatingFunction[…][t]}`, but Woxi's samples a
  fixed-step Runge–Kutta simulation and interpolates linearly. An input
  step between two of its steps costs about 1e-4.
- An inexact delay is written exactly and the rest numericized; where
  the `1.` lands differs (`Exp[-0.5 s]/(s + 1)` gives `1. + s` as the
  denominator in wolframscript and `1. + 1.*s` in Woxi).
- `BodePlot` draws the whole unwrapped phase. With a delay, wolframscript
  cuts the phase curve at a phase range it chooses (`[φ0 - 180°, φ0]` for
  `E^-s/(s (s + 1))`). `NyquistPlot` has no arrows, and the frequency
  range of `NicholsPlot` differs.
- `ToDiscreteTimeModel`/`ToContinuousTimeModel` have the same values but
  write them differently (`0.1*(1 + z)` for wolframscript's `0.1*(1. +
  1.*z)`; with `"ForwardRectangularRule"` wolframscript scales to
  `{{{1}}, -9. + 10.*z}`, Woxi to `{{{0.1}}, {{-0.9 + 1.*z}}}`). The
  hold methods (`"ZeroOrderHold"`, `"FirstOrderHold"`) and
  `"ZeroPoleMapping"` stay unevaluated.

### `PowerMod[a, 1/n, m]` with multiple roots

When `gcd(a,m)=1` and `gcd(n, λ(m)) ≠ 1` with n ≥ 3, WL returns a specific root
whose choice depends on per-prime CRT lifting and follows no
smallest/largest rule — n=4 gives the largest for two probes and the smallest
for a third. Woxi leaves those unevaluated rather than return a possibly-wrong
root. The unique-root and no-root regimes match exactly, as does n=2.
**Not reproducible.**

### Boolean minimization

`BooleanConvert` ships every wolframscript form except `"ESOP"` and `"BFF"`,
but four differences remain:

- **DNF needs term merging**, not just redundancy removal:
  `(a&&!b)||(!a&&b)||(a&&b)` is `a || b` in WL; Woxi keeps three terms.
- **CNF clause strengthening**: `(!a||b||c)&&(a||!b)&&(b||!c)` — WL shortens the
  first clause to `(!a||b)`.
- **Clause order when clause lengths differ**: `Xor[a, Implies[b,c]]` — neither
  variables-first nor length-first reproduces WL together with `(a||b)&&c`.
- **Equivalent-but-different CNF choices**: `a&&b&&c||!a&&!b&&!c` — WL keys on
  `(a,b)`/`(b,c)` pairs, Woxi on `(a,c)`/`(b,c)`. WL also sometimes produces a
  resolution-redundant CNF and a non-minimal DNF cover that Woxi does not.

`BooleanConvert[Nand[a,b,c], "ANF"]` prints `Nand[a, b, c]` in WL (it does not
rewrite the head) against Woxi's `!(a && b && c)`, and
`BooleanConvert[Equivalent[Xor[a,b],c], "IF"]` branches on `c` first, so WL's
BDD variable order is not always alphabetical.

`BooleanFunction[k, n]` and `BooleanMinterms`/`BooleanMaxterms` with an integer
variable count normalise to the opaque `BooleanFunction["BDD" -> …]` object —
**implemented**, encoding verified against WL for every function of 1, 3 and 4
variables and a sample of 5. The only remaining divergence is the too-few-
arguments case: `BooleanFunction[7, 2][True]` and `BooleanFunction[7, 2, {a}]`
both emit `::argr`, but WL then leaks its internal `BooleanConvert[…]` wrapper
into the returned expression where Woxi leaves the call as it stands.

### `LogicalExpand` repeated literals (unverified)

`LogicalExpand` drops a conjunction holding `x && !x` and keeps a repeated
literal once while it expands, so `(a || b) && (a || c)` gives
`a || (a && c) || (b && a) || (b && c)`. This was written while the engine
was unactivated, so it is unverified: WL may write `a && a`, or absorb terms
further (`a || (b && c)`).

### `SatisfiabilityInstances` ordering

The default single instance (greedy True-first DFS) matches, and an
unsatisfiable problem gives `{}`. The **multi-instance and `All` ordering**
follows WL's internal BDD structure and differs per expression —
`a||b||c` orders 7,3,1,5,2,6,4 while `Majority[a,b,c]` orders 7,6,5,3.
**Not reproducible.**


### `Surd` with a degree past `i128`

```sh
wolframscript -code 'ToString[Surd[8, 10^40], InputForm]'   # 8^(1/10000000000000000000000000000000000000000)
woxi eval 'Surd[8, 10^40]'                                  # Surd[8, 10000000000000000000000000000000000000000]
```

`Surd` builds the exponent with `make_rational`, which is `i128`-only, so a
`BigInteger` degree leaves the call unevaluated. Degrees up to `10^38` work.


### `Convolve` of two Gaussians leaves the constant as an unrationalized root

```sh
wolframscript -code 'ToString[Convolve[PDF[NormalDistribution[0, 1], x],
                     PDF[NormalDistribution[0, 2], x], x, y], InputForm]'
# 1/(E^(y^2/10)*Sqrt[10*Pi])
woxi eval 'Convolve[PDF[NormalDistribution[0, 1], x],
           PDF[NormalDistribution[0, 2], x], x, y]'
# Sqrt[(2*Pi)/5]/(2*E^(y^2/10)*Pi)
```

Same value — `Sqrt[2 Pi/5]/(2 Pi)` squared is `1/(10 Pi)` — but Woxi does not
pull the leftover `Pi` under the root and rationalize the result. This is the
general `Sqrt[q Pi]/Pi` folding, not anything `Convolve` does; it only shows up
here when the two widths do not divide out (equal widths, the common case,
cancel exactly and agree). See also the `Sqrt` product-split note.

The **sign of the squared shift** used to diverge here too and no longer does:
wolframscript prints whichever of `(y - mu)^2` / `(mu - y)^2` has the smaller
`LeafCount` — its result has been through `Simplify` — with a tie going to the
form that does not open with a minus. That reproduces all 11 measured shift
shapes (`3`, `s`, `s + t`, `1 + s`, `Pi`, `2 s`, `2 s + 3 t`, `u v`, `s - t`,
`-s - t`, `a + b`).


### `Convolve` of two Gaussians needs a literal width

```sh
wolframscript -code 'ToString[Convolve[PDF[NormalDistribution[m, s], x],
                     PDF[NormalDistribution[n, s], x], x, y], InputForm]'
# Sqrt[s^(-2)]/(2*E^((m + n - y)^2/(4*s^2))*Sqrt[Pi])
woxi eval 'Convolve[PDF[NormalDistribution[m, s], x],
           PDF[NormalDistribution[n, s], x], x, y]'
# Convolve[1/(E^((-m + x)^2/(2*s^2))*Sqrt[2*Pi]*s), …]
```

`gaussian_shape` reads the quadratic coefficient as a literal rational, so a
symbolic standard deviation is not recognized and the call stays unevaluated.
Symbolic *means* are recognized; only the width has to be a number.


## Special functions

### `TrigToExp[Sin[x]^3]` keeps the power unexpanded

```sh
wolframscript -code 'ToString[TrigToExp[Sin[x]^3], InputForm]'
# (-1/8*I)*(E^(-I*x) - E^(I*x))^3
woxi eval 'TrigToExp[Sin[x]^3]'
# (I/2/E^(I*x) - I/2*E^(I*x))^3
```

Same value. wolframscript pulls the `(-I/2)^3` coefficient out of the cubed
difference; Woxi rewrites each `Sin` in place and leaves the coefficient
inside. Found by the differential fuzzer.

### `LegendreP` distributes but `LaguerreL` collects

Resolved for both (see `collect_or_distribute` in
`src/functions/math_ast/orthogonal_polynomials.rs`), noted here because the
split is wolframscript's and not derivable: `LegendreP[2, E]` is
`-1/2 + (3*E^2)/2` there but `LaguerreL[2, E]` is `(2 - 4*E + E^2)/2`, and
`LegendreP[2, x]` goes back to the collected `(-1 + 3*x^2)/2`. The rule Woxi
implements is "collect when the argument mentions a symbol" for `LegendreP`
and "always collect" for `LaguerreL`.


### `HypergeometricPFQ` residues

`HypergeometricPFQ[{}, {b}, z]` with a negative half-integer `b` below `-5/2`
diverges in shape: WL stops collecting and prints a nested Bessel form. Two
smaller divergences live in the core `Power` normalizer and reproduce without
any hypergeometric: `(64/729)^(1/4)` stays put where WL gives `(2 Sqrt[2/3])/3`,
and Woxi's prefactor prints `4^(1/2 - b/2)` where WL prints `2^(1 - b)`.

`Hypergeometric1F1[a, b, z]` with symbolic `z` and `1 ≤ a < b`:
`HypergeometricPFQ[{1},{2},x]` is `(-1+E^z)/z` in WL and unevaluated in Woxi.
The `a=1` family has a clean formula, but Woxi orders `E^z` after the z-power
terms where WL puts it right after the constant.

### Carlson elliptic integrals

`CarlsonRF[0, y, z]` with distinct non-zero arguments needs `CarlsonRK`, and
symbolic `CarlsonRC[y, y]` needs a `Piecewise` — both unimplemented, so Woxi
echoes. `CarlsonRK` and `CarlsonRM` are unimplemented outright.

`InverseWeierstrassP` is **broken**: `InverseWeierstrassP[0.5, {1,2}]` returns a
garbage real pair `{-109.5, -39.3}` where WL returns a scalar complex
`1.308 + 0.469 I`. The correct route is
`℘⁻¹(p) = CarlsonRF[p-e1, p-e2, p-e3]`, but WL's representative selection is not
a simple sign or conjugate of it — for one probe it is the conjugate, for
another it differs by a lattice period. Picking it needs the period lattice.

### Weierstrass and elliptic gaps

`WeierstrassHalfPeriods` covers only the real positive-discriminant case; the
rhombic case needs a complex `EllipticK`. `WeierstrassInvariants[{1,I}]` and
`WeierstrassHalfPeriods[{4,0}]` with exact arguments give Gamma closed forms in
WL and stay symbolic in Woxi. `WeierstrassZeta`, `WeierstrassSigma`,
`SiegelTheta` and `QPolyGamma` are unimplemented.

Neville theta values are 1–2 ULP off wolframscript on some points.

### Mathieu functions

`MathieuC[1, 2, 0.5]` is `0.26 + 1.17 I` in WL — the Mathieu functions are
complex for a general characteristic value. Woxi's `MathieuS` returns a wrong
**real** value, having confused the order-versus-characteristic convention.
A correct fix needs a full Floquet implementation (continued fractions for the
characteristic exponent plus Fourier coefficients).

### `SphericalHarmonicY` numericizes exact angles

`SphericalHarmonicY[l, m, Pi/2, 0]` returns a machine float (e.g. `2.99…*^-17`
instead of an exact 0) where wolframscript returns the exact symbolic form.
Two residual form divergences remain even on the symbolic path:
`Sqrt[Pi^(-1)]` against `1/Sqrt[Pi]`, and `E^(I k Pi)` against `(-1)^k`.

### `LaguerreL` with a fractional second argument

Value correct, form divergent: `(105/8 - (105*x)/4 + (21*x^2)/2 - x^3)/6` in
Woxi against `(105 - 210*x + 84*x^2 - 8*x^3)/48` in WL. WL's evaluator does not
hoist content out of `Times[1/6, Plus[…]]` when typed directly, so `LaguerreL`
must build the common denominator internally — but `LaguerreL[1, 1/2, x]` stays
`3/2 - x` rather than `(3 - 2*x)/2`, so it is not a plain `Together` either.
Integer second arguments match.

### `ArcCot[Interval[…]]` is unevaluated

WL converts endpoints to `ArcTan` form *inside* the interval (`ArcCot[2]` →
`ArcTan[1/2]`), which differs from its own scalar `ArcCot[2]`.

### `WignerD[{1,0,0}, Pi/2]`

Gives `2.22e-16` instead of an exact 0.

### `LerchPhi` outside the unit disc

The numeric path sums `Σ z^k/(k+a)^s` and converges only for `|z| < 1` (plus a
tail for `z = 1, s > 1`); outside that region the call is unevaluated.
wolframscript continues analytically —
`LerchPhi[2, 3, -1.5]` → `51.981861922538684 - 2.1345964981239467*I`. A correct
implementation needs Crandall's algorithm or a contour formula.

### `SyntaxLength` is unimplemented

```sh
wolframscript -code 'SyntaxLength["1+"]'   # 4
woxi eval 'SyntaxLength["1+"]'             # SyntaxLength[1+]  (+ "not yet implemented" warning)
```

`SyntaxLength[s]` is the length of the longest prefix of `s` that could still
begin a complete expression — 4 for `"1+"` (WL counts past the end of the
string, since more input could complete it).

### `Capitalize[str, "TitleCase"]` is unimplemented

The algorithm is verified (tokenize on whitespace, first and last words
capitalized, middle "small words" lowercased), but the small-word set is WL's
*complete internal* preposition/conjunction/article list, keyed on the literal
word. 130+ candidates were tested and it kept growing — common (a, the, of),
participial (considering, barring, notwithstanding, but not
excluding/following), and archaic (betwixt, qua, ere, sans, cum, nigh, outwith,
thru, contra, vice). Any finite list diverges on untested words.
**Not reproducible.** `SpokenString` and `TextCases` entity extraction are the
same class.

### `TextSentences` grouping

WL's segmenter groups short sentences oddly (`"One. Two. Three."` →
`{"One.", "Two. Three."}`) while handling `"Dr. Smith went home."` correctly.
Woxi's straightforward splitting is the sensible answer. **Not reproducible.**

### `Hash[expr]` for non-strings

wolframscript hashes an internal serialisation: `Hash[1, "CRC32"]` is
3017272578 and `Hash[{1,2}, "CRC32"]` is 1789268987, matching CRC32 of neither
the WXF, the InputForm text, the FullForm text, nor an integer encoding.
Strings and `ByteArray`s are special-cased to raw bytes and **do** match, as do
all the named algorithms on them. `Hash[expr]` with no type is WL's proprietary
64-bit expression hash. `Compress` output length differs too (their container
format). **Not reproducible.**


## Distributions and statistics

### Goodness-of-fit p-values other than Kolmogorov–Smirnov, Cramér–von Mises and Pearson

```sh
wolframscript -code '{AndersonDarlingTest[#, w], KuiperTest[#, w], WatsonUSquareTest[#, w]} &[{1.2, 2.3, 1.9, 3.1, 2.2}] /. w -> WeibullDistribution[2., 2.5]'
# {0.7774542220369554, 0.588057276340719, 0.4747782205540094}
woxi eval '…same…'   # {0.7785318445270717, 0.582206479184168, 0.4867170128931776}
```

The test statistics agree. Woxi's p-values come from Marsaglia's
finite-sample Anderson–Darling CDF, Stephens' modified Kuiper and
Watson statistics, and the asymptotic series; none of these, and no
exact-distribution variant tried, gives wolframscript's numbers, which
are deterministic. Kolmogorov–Smirnov uses Marsaglia, Tsang & Wang's
exact distribution as wolframscript does, and agrees except far in the
tail of large samples (n = 998, D = 0.1024: Woxi 1.4481e-9, equal to
twice the exact one-sided Smirnov tail; wolframscript 1.4208e-9).
`KolmogorovSmirnovTest[data1, data2]` (two samples) is not implemented.

Most distribution divergences are the `Plus`/`Times` ordering and `Sqrt`-split
classes above, reached through a moment formula. The recurring shapes:

- **`Sqrt` product split** — `Sqrt[2 nu]` against WL's `Sqrt[2] Sqrt[nu]`,
  reached through `StandardDeviation[ChiSquareDistribution[nu]]`,
  `Nakagami`'s numeric `Mean`, `Hoyt`'s numeric `Mean`
  (`4*Sqrt[1/(5*Pi)]*EllipticE[3/4]` against `(4*EllipticE[3/4])/Sqrt[5*Pi]`),
  `SkewNormal`'s `Mean[0,1,1]` (`Sqrt[Pi^(-1)]` against `1/Sqrt[Pi]`) and
  `VarianceGamma`'s integer-λ branches.
- **`Plus` term order in a two-term variance** — most distribution variances
  are a two-term difference and hit it. Named instances: `LogLogistic`,
  `LogSeries`, `Coxian`, `Hyperexponential`, `TukeyLambda`, `Wakeby`,
  `Weibull` (3-arg), `Suzuki`, `PERT`, `Benini`, `Dirichlet`.
- **`Times` factor order** — `Pi` sorts ahead of a function-containing sum in
  Woxi (`4 Pi (…)` against WL's `4 (…) Pi`), and
  `p^(-1+k)*(1-p)` against WL's `(1-p)*p^(-1+k)`.

### Distributions compute with out-of-range parameters

```sh
wolframscript -code 'PDF[NormalDistribution[0, -1], 0.3]'
# NormalDistribution::posprm: Parameter -1 at position 2 in
# NormalDistribution[0, -1] is expected to be positive.
woxi eval 'PDF[NormalDistribution[0, -1], 0.3]'
# -0.3813878…
```

A negative probability density. Constructing the distribution is fine in both
(`BetaDistribution[4, -1]` echoes silently); the divergence is in *using* one.
The same applies to `PDF`, `Quantile` and `Mean`. Woxi has a validator, but it
is wired only to `DistributionParameterQ` and returns a bare boolean — the
message needs the offending parameter's position and value.

### Raw moments are missing for most distributions

`Moment[dist, n]` for n ≥ 3 returns unevaluated except for Bernoulli,
Exponential, Uniform, Normal (symbolic n ≤ 4), Gamma, Poisson and ChiSquare,
which cascades into `Skewness`/`Kurtosis` not reducing. Two structural notes:
the order-2 raw moment is computed by a separate variance-plus-mean² path, so
`Moment[dist, 2]` prints expanded (`2k+k^2`) not WL's factored `k(2+k)`; and
raw `Moment[dist, n]` diverges anyway for the distributions whose WL answer is
an incomplete-Gamma form.

`Beta` raw moments are deliberately **not** added: its central moments are
rational functions in a,b that Woxi's simplifier cannot collapse, so
`Skewness`/`Kurtosis[Beta]` would explode into 100+ term expressions against
WL's compact form.

`Skewness` of a scale-parameter distribution is a positivity rabbit hole —
`Skewness[ExponentialDistribution[l]]` is `2/(l^(-2))^(3/2)*l^3`, value correct
but not simplified to `2` without knowing `l > 0`.

### Central moments of a distribution factor differently

```sh
wolframscript -code 'ToString[CentralMoment[BernoulliDistribution[p], 3], InputForm]'
# (1 - p)*p*((1 - p)^2 - p^2)
woxi eval 'CentralMoment[BernoulliDistribution[p], 3]'
# p - 3*p^2 + 2*p^3
```

Even constructing the factored form does not help: Woxi reorders the inner
`Plus` to `(1-p)*p*(-p^2 + (1-p)^2)`. `Skewness` and `Kurtosis` of such
distributions diverge for the same reason.

### `Probability` of a normal tail uses a different Erf form

```sh
wolframscript -code 'ToString[Probability[x > 1, x \[Distributed] NormalDistribution[0,1]], InputForm]'
# (1 - Erf[1/Sqrt[2]])/2
woxi eval 'Probability[x > 1, x \[Distributed] NormalDistribution[0,1]]'
# (2 - Erfc[-(1/Sqrt[2])])/2
```

`CDF` and `SurvivalFunction` of the same distribution **do** match. WL is
entangled with itself here — its `SurvivalFunction[Normal, 1]` is the
unsimplified `1 - Erfc[-(1/Sqrt[2])]/2` while `Probability[x > 1]` is the `Erf`
form, so there is no single canonical target for numeric arguments.

### Per-distribution scope gaps

Left unevaluated because WL's own output is an unreproducible form:

| distribution | what is missing |
| --- | --- |
| `Coxian` | mixed repeated/distinct rates, float and symbolic-rate PDF/CDF |
| `Hypoexponential` | repeated rates (Erlang-style terms) |
| `TsallisQGaussian` | CDF for numeric q ≠ 1 (WL collapses 2F1 per q), float params |
| `TukeyLambda` | any λ outside `{0, ±1, 1/2, 2}`; λ=3 is a `Root` form in WL |
| `Wakeby` | symbolic Mean/Variance and symbolic-q Quantile ordering |
| `VarianceGamma` | CDF (WL's forms are partially unexpanded) |
| `Hoyt` | CDF (needs `MarcumQ`) |
| `SkewNormal` | Skewness/Kurtosis; CDF at a=1 has a special squared form in WL |
| `NegativeMultinomial` | CDF with symbolic parameters; symbolic StandardDeviation |
| `HotellingTSquare` | 1-ULP coefficient gap for non-integer float parameters |
| `Wishart` | PDF (needs `MatrixGammaFunction`), `RandomVariate` |
| `FailureDistribution` | duplicated events, `Mean` (inclusion-exclusion sums) |
| `DiscreteMarkovProcess` | bare `StationaryDistribution`, `MarkovProcessProperties`, CDF, graph constructor |
| `FirstPassageTime` | symbolic-t PDF, probability-vector initial states |
| `PERT` | `Median` (WL gives `InverseBetaRegularized`/`Root` forms) |
| `LogSeries` | `RandomVariate`, `Median`, `Quantile` |
| `PolyaAeppli` | PDF, CDF |
| `Benford`, `WaringYule`, 4-arg `Gamma`, `Davis` | unimplemented |

Also: `DiscreteUniformDistribution`'s StandardDeviation is `Sqrt[X/12]` against
WL's `Sqrt[X]/(2 Sqrt[3])`, and `ExpGammaDistribution`'s PDF is unimplemented
(exponent `Plus` order).

### `MovingMap[Mean, …]` stays exact

```sh
wolframscript -code 'MovingMap[Mean, {1,2,3,4,5}, 2]'  # {2., 3., 4.}
woxi eval 'MovingMap[Mean, {1,2,3,4,5}, 2]'            # {2, 3, 4}
```

The trigger is the applied function, not the window: `MovingMap[Total, …]`
stays exact in both. `Mean[{1,2,3}]` is exact on its own, yet WL floats it
inside `MovingMap`. Which functions WL numericizes is not determinable without
a large probe; Woxi's exact behaviour is arguably the better one.
**Not reproducible.**

### Robust statistics that iterate

- `BiweightLocation` is iterative in machine floats in WL and returns
  unconverged trailing digits (`{1,2}` → `1.4999988566906757`) even for exact
  input. `BiweightMidvariance` is closed-form and matches.
- `SpatialMedian` uses an unconverged Weiszfeld iteration
  (`0.5773502623362886` against the true `1/Sqrt[3]`). Only 1-D inputs, which
  reduce to `Median`, are reproducible.

**Not reproducible.**

### `EstimatedDistribution` is unimplemented

### `Mean`, `Quantile` and `RootMeanSquare` of an `EventSeries` or `TemporalData`

Unevaluated.

### `FinancialDerivative` of an American option is unimplemented

Only European calls and puts (Black–Scholes–Merton price and Greeks) are
there; an `{"American", …}` contract stays unevaluated, so its
`"CriticalValue"` feeds a symbolic bound into the optimization that follows
(Demonstrations GeskeJohnsonMethod,
MaximizingABermudanPutWithTwoEarlyExerciseTemporalPoints).
WL prices these on a finite-difference grid (`"GridSize" -> {1000, Automatic}`):

```sh
wolframscript -code 'FinancialDerivative[{"American", "Put"}, {"StrikePrice" -> 45., "Expiration" -> 3.}, {"InterestRate" -> 0.07, "Volatility" -> 0.3, "CurrentPrice" -> 40., "Dividend" -> 0.}, "CriticalValue", "GridSize" -> {1000, Automatic}]'
# 30.855855855855857
```


## Linear algebra

### `SingularValueDecomposition` of an exact matrix stays unevaluated

```sh
wolframscript -code 'SingularValueDecomposition[{{1, 2}, {3, 4}, {5, 6}}]'  # exact radicals
woxi eval 'SingularValueDecomposition[{{1, 2}, {3, 4}, {5, 6}}]'            # unevaluated
```

Only machine-precision matrices are decomposed. Demonstration
ExploratoryFactorAnalysis normalizes exact data and runs its alternating
least squares on such exact matrices.

### Symbolic `Inverse` with rational entries normalizes one sign differently

```sh
wolframscript -code 'Inverse[{{1/2, x}, {x, 1}}]'  # {{-2/(-1 + 2*x^2), (2*x)/(-1 + 2*x^2)}, {(2*x)/(-1 + 2*x^2), -(-1 + 2*x^2)^(-1)}}
woxi eval 'Inverse[{{1/2, x}, {x, 1}}]'            # {{-2/(-1 + 2*x^2), (2*x)/(-1 + 2*x^2)}, {(2*x)/(-1 + 2*x^2), (1 - 2*x^2)^(-1)}}
```

Woxi brings each entry together on its own; Wolfram gives every entry the
denominator `-1 + 2 x^2`. Polynomial matrices without rationals agree.

### Float matrices with complex eigenvalues

Wolfram complexifies the **whole** result and orders each conjugate pair with
`+I` first: `{0. + 1.*I, 0. - 1.*I, 1. + 0.*I}`. Woxi complexifies the whole
result too (n ≥ 3, via QR iteration), but sorts by decreasing magnitude only,
so exact magnitude ties between a pair and a real value can order differently
(`{0. + 1.*I, 1. + 0.*I, 0. - 1.*I}`). Complex `Eigenvectors` use unit length
with a real largest component; LAPACK's phase/sign conventions are not
reproducible. 2×2 float matrices with complex eigenvalues stay as before.

Generic dense float matrices also differ in the last 1–2 digits from
WL/LAPACK.

### `SchurDecomposition` and `SmithDecomposition` factors are non-unique

Wolfram's LAPACK eigenvalue ordering along `t` and the column signs of `q` are
not reproducible independently (`{{1,2},{3,4}}` — WL puts −0.372 first, Woxi
5.372). Test the defining properties instead. `RealBlockDiagonalForm -> False`
(the complex Schur form) is unimplemented.

`SmithDecomposition`'s `u` and `v` differ the same way, and WL sometimes keeps
a **negative** diagonal entry in `s` (`{{1,0},{0,-2}}` stays) while Woxi always
normalises to the canonical non-negative SNF — six probe rounds produced no
rule. Woxi's answer is the mathematically canonical one. **Not reproducible.**

### `JordanReduce` scope

Left unevaluated: eigenvalues containing `Root` objects (WL's diagonal ordering
there is neither `Eigenvalues` order nor ascending), and inexact matrices with
n ≥ 3 and repeated eigenvalues (WL clusters with a tolerance).

### `LDLDecomposition` and `FrobeniusReduce` on inexact or symbolic input

Left unevaluated. WL's float LDL shows LAPACK-internal rounding
(`D = {4., 2.0000000000000004}` where plain f64 gives exactly 2.0), and its
`FrobeniusReduce` returns precision-tagged arbitrary-precision reals whose
precision comes from internal error tracking. Symbolic matrices come back with
unsimplified algorithm-internal quotients.

Also found: `FrobeniusReduce[m, Modulus -> -5]` reproducibly **crashes** the
wolframscript kernel (exit 139); Woxi returns unevaluated.

### `BunchKaufmanDecomposition` and `PopovDecomposition`

Both deferred. Bunch-Kaufman's `BlockDiagonalMatrix` payload has encoder quirks
(a whole-matrix antidiagonal 2×2 pivot is stored as two permuted 1×1 blocks)
plus LAPACK pivot-order replication in exact arithmetic. Popov normalizes over
ℤ[x] rather than ℚ[x] — constant matrices get integer Hermite normal form,
pivots are not made monic, and rational entries get content cleared.

### `BlockUpperTriangularMatrix` / `BlockLowerTriangularMatrix`

Not constructors: they find a matrix's finest block-triangular permutation and
return a structured array. The component ordering is a Dulmage–Mendelsohn
decomposition needing a perfect bipartite matching first, and WL's tie-breaking
would be high-risk to match.

### `MatrixFunction` and `MatrixExp` on irrational results

Both produce value-correct but unsimplified, form-divergent results for
non-diagonal matrices with irrational entries — `(3E-E^3)/2-E+E^3` against WL's
`E/2+E^3/2`, `(1+Sqrt[3])/2` against WL's split `1/2+Sqrt[3]/2`. Diagonal and
integer-result cases match. `MatrixExp` on complex/rotation matrices is
value-correct but Woxi simplifies to `Cos`/`Sin` where WL 15 keeps `E^±I`
combinations.

`MatrixExp[I t m]` for a symbolic `t` and an exact 3×3 Hermitian `m` (e.g. a
Gell-Mann matrix) now evaluates, but stays in the `(1 + E^(2 I t))/(2 E^(I t))`
form because `Simplify` does not fold it to `Cos[t]`/`I Sin[t]`.

### `FourierDCTMatrix` for n ≥ 5

Values agree to ~15 digits; the radical **display** diverges — WL canonicalizes
`Sqrt[(5/8+Sqrt[5]/8)/5]` where Woxi gives `Sqrt[5/8+Sqrt[5]/8]/Sqrt[5]`, and
WL folds `2/(2 Sqrt[3])` to `1/Sqrt[3]`.

### `SymmetricReduction` two-argument form

Value correct always, but the display order diverges for some higher-degree
two-variable cases: `x^3+y^3` gives `(x+y)^3-3*x*y*(x+y)` in Woxi against WL's
`-3*x*y*(x+y)+(x+y)^3`. The three-argument form is fully conformant.

### `Symmetrize` of a symbolic tensor

`Symmetrize[T, Symmetric[{1,2,3}]]` for a bare symbol `T` should give the
explicit `(T + TensorTranspose[T, {2,1}] + …)/6`. Woxi leaves it unevaluated:
`TensorTranspose` on a symbol errors, and wolframscript's term order truncates
trailing fixed points.

### `ArrayFlatten[a, r]`

The two-argument form is unimplemented (Woxi reports `argx`). Ragged
block-tensor assembly, rarely used.

### `RotationMatrix` / `RotationTransform` about a symbolic axis

Woxi leaves a 3D rotation about a non-numeric axis unevaluated
(`RotationTransform[Pi, {0, 0, x}, {1, 0, 0}]` echoes). Wolfram does return
something, but only as an unsimplified artifact of its internal Gram–Schmidt:

```sh
wolframscript -code 'RotationMatrix[Pi, {0, 0, x}]'
# {{-((x Conjugate[x])/Abs[x]^2), 0, 0}, {0, -1, 0}, {0, 0, (x Conjugate[x])/Abs[x]^2}}
```

Note the asymmetry — entry (1,1) keeps the un-cancelled `x Conjugate[x]/Abs[x]^2`
(which is just `1`) while entry (2,2) is a plain `-1`, so the two diagonal
entries of the *same* rotation print differently. A fully symbolic axis
`{a, b, c}` produces roughly 8 KB of nested `Conjugate[1/Sqrt[a Conjugate[a] +
b Conjugate[b]]]` terms, with `Sqrt[(a Conjugate[a] + b Conjugate[b])/(a
Conjugate[a])]` denominators that are singular exactly on the coordinate axes.
The output is not a canonical form (and is not a real rotation matrix for
complex components), so there is nothing stable to conform to. **Not
reproducible**; the affected unit test is listed in `EXACT_EXPR_SKIP` in
`tests/wolframscript/verify_unit_tests.ts`.

### `NumberFieldDiscriminant`

Non-maximal orders are unevaluated (`Sqrt[2]+Sqrt[3]` → WL 2304 via a full
conductor/Round-2 computation), as are non-monic minimal polynomials of degree
≥ 3.


### `LinearSolve[m]` is not a `LinearSolveFunction`

The operator form `LinearSolve[m]` / `LinearSolve[m, opts]` stays as that
unevaluated call in Woxi (applying it to a vector solves the system); WL
returns a `LinearSolveFunction[dims, data]` object holding the factorization:

```sh
wolframscript -code 'LinearSolve[{{2, 1}, {1, 3}}, Method -> "Cholesky"]'
# LinearSolveFunction[{2, 2}, {3, True, {{{Sqrt[2], 1/Sqrt[2]}, {0, Sqrt[5/2]}}, None, 0}, …}]
woxi eval 'LinearSolve[{{2, 1}, {1, 3}}, Method -> "Cholesky"]'
# LinearSolve[{{2, 1}, {1, 3}}, Method -> Cholesky]
```

Building that factorization is also where WL's operator form reports a
matrix that is not positive definite, under other names than the
three-argument form's `LinearSolve::npdef` (which Woxi uses for both):
`CholeskyDecomposition::npdef` for an exact matrix and, for a machine one,
`LinearSolve::posdef: -- Message text not found -- ({{1., 2.}, {2., 1.}})`.

## Expression structure and evaluation

### SameQ of reals nested more than 256 levels deep

```sh
wolframscript -code 'Nest[f, 1., 300] === Nest[f, 1.0000000000000002, 300]'   # True
woxi eval 'Nest[f, 1., 300] === Nest[f, 1.0000000000000002, 300]'             # False
```

SameQ identifies machine reals one unit in the last place apart anywhere in
an expression, but Woxi's structural comparison stops at depth 256 (to stay
off the stack limit) and calls deeper differently-spelled reals different.

### A held `Infinity` is not the symbol

```sh
wolframscript -code 'ToString[FullForm[Hold[-Infinity]]]'   # Hold[Times[-1, Infinity]]
woxi eval 'ToString[FullForm[Hold[-Infinity]]]'             # Hold[DirectedInfinity[-1]]
```

Woxi stores `Infinity` as one identifier whether or not it has evaluated, so
FullForm always shows the evaluated `DirectedInfinity[1]` (and `-Infinity`
the evaluated `DirectedInfinity[-1]`). Unheld, both agree.

### A package symbol answers before its package has been loaded

```sh
wolframscript -code 'ToString[InterpolatingFunctionDomain[
                     Interpolation[{1, 4, 9, 16}]], InputForm]'
# InterpolatingFunctionDomain[InterpolatingFunction[{{1, 4}}, …]]
woxi eval 'InterpolatingFunctionDomain[Interpolation[{1, 4, 9, 16}]]'
# {{1, 4}}
```

`InterpolatingFunctionDomain` and its `InterpolatingFunctionAnatomy` siblings
live in a package a fresh kernel does not autoload, so without
`Needs["DifferentialEquations`InterpolatingFunctionAnatomy`"]` wolframscript
reads the name into `Global`` and echoes the call back. Woxi keeps every
built-in in one namespace — `Needs` on a context that ships with the language
is a no-op — so the name works either way, which makes Woxi a superset here
rather than a mismatch. Writing the `Needs` makes the two agree; there is no
way to make Woxi *withhold* the symbol without giving up the flat namespace.

### `Unevaluated` comes off for some heads and not others

The wrapper is stripped where it was verified to matter — the arithmetic and
structural operators, `SameQ`/`UnsameQ`, `Print`, and the built-ins that
already took it (`Length`, `Head`, `Sqrt`, `Options`, `Information`) — but
not as the general rule Wolfram applies, which takes every `Unevaluated[…]`
off just before the head's own rules run and puts it back if nothing was
consumed:

```sh
wolframscript -code 'Print[Unevaluated[1] < 2]'
# True
woxi eval 'Unevaluated[1] < 2'
# Unevaluated[1] < 2
```

The arithmetic comparisons are the ones this reaches. Wolfram compares the
*unevaluated* content and answers `Unevaluated[1 + 1] == 2` with the
comparison itself; Woxi would have to evaluate the content to compare it at
all, so stripping there would answer `True` where Wolfram does not. The
wrapper is therefore left in place for `==`, `<`, `>` and their relatives.

### Rubi loads and integrates, but not everything it computes agrees

The [Rubi](../../rubi.md) rule base is the densest available exercise of the
pattern matcher and the definition store: 7000 rules, all read back out of
`DownValues` and rewritten before use. It loads unmodified, integrates, and
its step display (`$LoadShowSteps = True`, `Steps`) works. On a 30-integral
sample 27 answers are character-for-character `wolframscript`'s and the
other 3 are the same function written another way; the
*IntWithStepsOfTeXForm* notebook reproduces three of its four examples
exactly. What is left, all of it ordinary Woxi behaviour rather than
anything about the package:

- **Loading takes about a minute** against roughly twenty seconds under
  `wolframscript`, three with the step display.
- **`Int[Sin[x]^3*Cos[x]^2, x]` and `Int[Sin[x]*Cos[x]^3, x]`** run for
  minutes and exhaust memory. `wolframscript` answers both instantly.
- **`Int[E^x*x, x]`** is `-Gamma[2, -x]` where Rubi gives `-E^x + E^x*x`, and
  **`Int[1/(a + b*Cos[x]), x]`** reaches its arctangent through another
  substitution: a different rule fires first. Equivalent antiderivatives.
- **`x^2*Sqrt[a^2 - x^2]`** prints as `Sqrt[a^2 - x^2]*x^2` (the `Times`
  order of a power of a sum against a power of a symbol), and one
  intermediate sum of `Int[Sqrt[Tan[x]], x]` lists two `Subst` terms the
  other way round (see the ordering entries above).
- **The rule text recorded beside each step** keeps its `FreeQ` conditions
  and is written in linear box syntax rather than `DisplayForm`, and the
  rule numbers differ (7391 rules against 7300).

### A held `;`-sequence is not bracketed inside a tighter operator

```sh
wolframscript -code 'Hold[(a; b) + c]'
# Hold[(a; b) + c]
woxi eval 'Hold[(a; b) + c]'
# Hold[a; b + c]
```

`CompoundExpression` has the lowest precedence of any operator, so the
printed form re-parses as `CompoundExpression[a, Plus[b, c]]` — a different
expression. Inside a call it is right: `Hold[f[(a; b), c]]` prints as
`Hold[f[a; b, c]]`, as wolframscript does.

### Newline-separated statements become one `CompoundExpression`

```sh
wolframscript -code 'ToExpression["2\n3", InputForm, Hold]'
# Hold[2, 3]
woxi eval 'ToExpression["2\n3", InputForm, Hold]'
# Hold[2; 3]
```

The reader turns every statement separator into a `;` before parsing, so it
cannot tell a newline from one. A `;`-separated program agrees
(`ToExpression["2; 3", InputForm, Hold]` is `Hold[2; 3]` in both), and so
does a trailing `;` (`Hold[a := 1; ]`).

### `Derivative[n][f][x]` is stored flat, so structural functions see three parts

```sh
wolframscript -code 'Head[Derivative[1][g][x]]'      # Derivative[1][g]
woxi eval 'Head[Derivative[1][g][x]]'                # Derivative

wolframscript -code 'Length[Derivative[1][g][x]]'    # 1
woxi eval 'Length[Derivative[1][g][x]]'              # 3

wolframscript -code 'Apply[f, Derivative[1][g][x]]'  # f[x]
woxi eval 'Apply[f, Derivative[1][g][x]]'            # f[1, g, x]
```

Woxi stores it flat rather than as nested curried calls. The renderer prints it
correctly and `D` returns the right thing, so only structural introspection
diverges — `Head`, `Length`, `Part`, `Level`, `Map` and `Apply` all leak the
internal shape.

### `Unevaluated[…]` is not transparent to structural functions

```sh
wolframscript -code 'Unevaluated[1 + 1] // Head'   # Plus
woxi eval 'Unevaluated[1 + 1] // Head'             # Unevaluated

wolframscript -code 'Depth[Unevaluated[1 + 1]]'    # 2
woxi eval 'Depth[Unevaluated[1 + 1]]'              # 3
```

`Length`, `AtomQ` and `Head` now agree; `Depth`, `First` and `Part` still see
the wrapper (`Part[Unevaluated[1+2], 1]` gives 3 where WL gives 1). The
arithmetic consumers are also open: `Unevaluated[1+1] + 1` is
`1 + Unevaluated[1 + 1]` rather than 3, and `Total`/`Identity` behave the same
way. Woxi's model is per-consumer stripping because the real rule is not the
published "strip unless HoldAllComplete".

The same per-consumer stripping is why a wrapper a *pure function's body*
produced is stripped one level too eagerly. WL strips only wrappers written
literally in an argument list, so a produced one survives into any consumer:

```sh
wolframscript -code 'Head[(Unevaluated[Sequence[#, #^2]] &)[3]]'   # Unevaluated
woxi eval 'Head[(Unevaluated[Sequence[#, #^2]] &)[3]]'             # Sequence
```

Structural positions agree — `{0, (Unevaluated[Sequence[#, #^2]] &)[3], 9}` is
`{0, Unevaluated[Sequence[3, 3^2]], 9}` and `f[…]` keeps the wrapper in both —
and so do the strippers whose wrapper *was* literal. Only an
argument-consuming built-in (`ToString`, `Length`, `Head`) fed a produced
wrapper diverges; telling the two apart needs the literal-ness of each
argument threaded through to the built-in dispatch.

A loop body is an argument-consuming position of the same kind, so a wrapper
written there survives where WL strips it and evaluates what was inside:

```sh
wolframscript -code 'Table[Unevaluated[1 + 1], {i, 2}]'   # {2, 2}
woxi eval 'Table[Unevaluated[1 + 1], {i, 2}]'             # {Unevaluated[1 + 1], Unevaluated[1 + 1]}

wolframscript -code 'Do[Print[Unevaluated[1 + 1]], {i, 1}]'   # 2
woxi eval 'Do[Print[Unevaluated[1 + 1]], {i, 1}]'             # Unevaluated[1 + 1]
```

### A definition's bare symbol argument becomes a blank pattern

`SetDelayed` turns a plain symbol in the left-hand side's argument list into a
named pattern, so `f[x] := …` defines what `f[x_] := …` defines. WL takes the
symbol literally: the definition is stored for the single argument `x` and
matches nothing else.

```sh
wolframscript -code 'f[x] := x^2; f[5]'               # f[5]
woxi eval 'f[x] := x^2; f[5]'                         # 25

wolframscript -code 'f[x] := x^2; DownValues[f]'      # {HoldPattern[f[x]] :> x^2}
woxi eval 'f[x] := x^2; DownValues[f]'                # {HoldPattern[f[x_]] :> x^2}
```

WL also evaluates the arguments of the left-hand side before storing the
definition, which Woxi's pattern reading hides:

```sh
wolframscript -code 'i = 3; g[i] := i; DownValues[g]'  # {HoldPattern[g[3]] :> i}
woxi eval 'i = 3; g[i] := i; DownValues[g]'            # {HoldPattern[g[i_]] :> i}
```

`Set` (`=`) stores the literal argument as WL does. A symbol nested inside a
structured argument takes a third path and leaks an internal name into the
stored right-hand side:

```sh
wolframscript -code 'f[{x}] := x; DownValues[f]'      # {HoldPattern[f[{x}]] :> x}
woxi eval 'f[{x}] := x; DownValues[f]'                # {HoldPattern[f[{x}]] :> _lp0[[1]]}
```

### Sum and Product substitute their iterator instead of localizing it

`Table` and `Do` localize the iterator the way `Block` does — the symbol is
given a value for one iteration rather than replaced throughout the body — so
held positions keep it symbolic. `Sum` and `Product` still substitute, which
both burns the counter into held subexpressions and stops WL's own
recognition of a summand that does not depend on the index:

```sh
wolframscript -code 'Sum[Hold[i], {i, 2}]'      # 2*Hold[i]
woxi eval 'Sum[Hold[i], {i, 2}]'                # Hold[1] + Hold[2]

wolframscript -code 'Product[Hold[i], {i, 2}]'  # Hold[i]^2
woxi eval 'Product[Hold[i], {i, 2}]'            # Hold[1]*Hold[2]
```

### A non-terminating NestWhile returns a wrong answer instead of not terminating

```sh
wolframscript -code 'TimeConstrained[NestWhile[#/2 &, 16, UnsameQ, 2], 5, timeout]'   # timeout
woxi eval 'NestWhile[#/2 &, 16, UnsameQ, 2]'                                          # 1/2^10000
```

Successive halvings of 16 are never `SameQ`, so this iterates forever in
wolframscript. Woxi silently stops after about 10 000 iterations and returns
the value it had reached, which is neither wolframscript's behaviour nor an
error.

### `Return` unwinds through ordinary argument evaluation

Wolfram treats `Return[x]` as an ordinary expression that only three things
consume (a definition body, `Do`, `Scan`), with `CompoundExpression`, `While`,
`For`, `Module`, `Block` and `With` passing it upward intact. Everything else
keeps it as a **value**: `Table[Return[1], {2}]` is `{Return[1], Return[1]}`,
`Catch[Return[1]]` is `Return[1]`, `{Return[1], 2}` and `f[Return[1]]` and
`Return[1] + 2` all stand.

Woxi raises `Return` as a signal, so each boundary must be explicit. `Table`,
`Map`, `Select` and `With` are wired; a `Return` reached by ordinary argument
evaluation still unwinds — `{Return[1], 2}` gives 1, `Catch[Return[1]]` gives
1, `Function[Return[3]][]` gives 3.

### `x = 2 x` does not recurse

wolframscript evaluates the right-hand side, assigns the own-value `x → 2*x`,
and then re-evaluates it on every lookup until `$RecursionLimit`:

```text
$RecursionLimit::reclim: Recursion depth of 30 exceeded.
TerminatedEvaluation[RecursionLimit]
```

Woxi returns `2*x` and treats the own-value as a fixed-point substitution.

### `Format` upvalues are stored as UpValues rather than FormatValues

```sh
wolframscript -code 'c /: Format[c] := "see"; ToString[UpValues[c], InputForm]'  # {}
woxi eval 'ClearAll[c]; c /: Format[c] := "see"; UpValues[c]'
# {HoldPattern[Format[c]] :> see}
```

Wolfram files a `Format` definition made through `/:` under `FormatValues`,
not `UpValues`, so `UpValues[c]` comes back empty. Woxi has no separate
`FormatValues` table and reports the rule as an upvalue. The rest of the
`UpValues` / `TagSet` / `TagUnset` family agrees with wolframscript.

### `Format[…]` is neither held nor form-aware

```sh
wolframscript -code 'Head[Format[x + y]]'          # Format
woxi eval 'Head[Format[x + y]]'                    # Plus

wolframscript -code 'Head[Format[x + y, OutputForm]]'  # Format
woxi eval 'Head[Format[x + y, OutputForm]]'            # Symbol
```

Wolfram keeps `Format[expr]` and `Format[expr, form]` unevaluated — `Length`
is 1 or 2, part 1 is the expression — and only applies a display rule, the
same model `Definition` follows.

The second argument is also ignored, so every form but `OutputForm` renders
the plain expression:

```sh
wolframscript -code 'Format[Sqrt[x], TeXForm]'      # \sqrt{x}
woxi eval 'Format[Sqrt[x], TeXForm]'                # Sqrt[x]

wolframscript -code 'Format[x/y, TeXForm]'          # \frac{x}{y}
woxi eval 'Format[x/y, TeXForm]'                    # x/y

wolframscript -code 'Format[x + y, StandardForm]'   # RowBox[{x, +, y}]
woxi eval 'Format[x + y, StandardForm]'             # x + y

wolframscript -code 'Format["ab", InputForm]'       # "ab"
woxi eval 'Format["ab", InputForm]'                 # ab
```

`StandardForm` and `TraditionalForm` want boxes, which `ToBoxes` already
produces correctly, and `TeXForm` / `InputForm` want the corresponding
renderer. An unsupported form (`Format[x^2, FullForm]`) stays unevaluated in
WL and prints as `Format[x^2, FullForm]`.

### `CellularAutomaton` refuses an explicit window it would have to allocate

```sh
wolframscript -code 'CellularAutomaton[90, {1, 0, 1}, {{{1}}, {0, 1000000000}}]'
# a billion-cell list
woxi eval 'CellularAutomaton[90, {1, 0, 1}, {{{1}}, {0, 1000000000}}]'
# CellularAutomaton[90, {1, 0, 1}, {{{1}}, {0, 1000000000}}]
```

A cyclic initial condition never grows past its own size, so nothing bounds an
explicit cell window against it; Woxi checks the window's length
arithmetically and declines rather than collecting a billion indices. The
tspec grammar and every window that fits in memory agree with WL exactly.

### `DownValues` drops a `/;` guard on a compound left-hand side

`f[x_] := x^2 /; x > 0` round-trips, but
`k[(a_. + b_.*x_)^m_., x_Symbol] := … /; FreeQ[…]` reports
`{HoldPattern[k[…]] :> a + b + m}` with no `/;`. The guard *is* enforced at
call time — it lives in a per-parameter condition slot the reconstruction
never re-attaches. This is the single remaining blocker for running Rubi's
`FixIntRules[]`, which reads `DownValues[Int]`, sees unguarded rules and writes
them back, destroying the rule base.

A related display-only residue: a trailing-sequence list rule's `DownValues`
shows `{a_, __}` rather than `{a_, b__}`, because the trailing element's name
is not recoverable.

### `ToExpression` of a definition

`ToExpression["h[x__] := {x}; h[1, 2]"]` gives `h[1, 2]` while
`ToExpression["zz = 7; zz + 1"]` gives 8 — definitions made through
`ToExpression` do not take effect for the rest of the same string.

### `Context` of an unknown context

`Context["zzz`abc"]` returns `"zzz`"` in Woxi; wolframscript reports
`Context::notfound` and stays unevaluated.

### `Module` rename artefacts and `InformationData`

`Module` locals are named `loc$1` here and `loc$` in wolframscript (and the
counter value is a session number that can never be pinned), and
`InformationData` prints its keys unquoted. wolframscript also carries its own
`WolframScript`` context, which Woxi has no equivalent for — so a bare
`$ContextPath` can never match.

### Legacy package names

Any `Combinatorica`…`, `PolyhedronOperations`…` or `HypothesisTesting`…`
symbol Woxi implements (e.g. `UnrankPermutation`, `Truncate`, `Stellate`,
`MeanTest`, `MeanDifferenceTest`) evaluates where wolframscript leaves it
unevaluated, because the package is not loaded. Woxi has no package system for
`Needs` to load into, so the qualified names are always live. With the package
loaded the results agree.

`Needs` also does not put such a context on `$ContextPath`, so where
wolframscript prints `OneSidedPValue -> …` after
`Needs["HypothesisTesting`"]`, Woxi prints the qualified
`HypothesisTesting`OneSidedPValue -> …` — the same expression, spelled in
full. For the same reason a bare `MeanTest` never resolves to the package's
symbol here. Compare through `property /. …`, not on the printed rule.

`HypothesisTesting`MeanTest[data, mu0]` with a symbolic `mu0` (including
`Automatic`, which the package does *not* read as 0) or symbolic `data` stays
unevaluated in Woxi; wolframscript builds a symbolic `Piecewise` of
`BetaRegularized` branches out of it. `MeanDifferenceTest` with matrix samples
stays unevaluated too — the package's own matrix handling there is broken,
producing a `StudentTDistribution` over a *list* of degrees of freedom.

### `$VersionNumber` is a string

Woxi returns its version *string* (`v0.3.0-46-g…`) rather than a number, so
`ToString[$VersionNumber]` and any `$VersionNumber >= 9` test break.

### `Manipulate`'s `Initialization` is not scoped

`Manipulate[…, Initialization :> (f[x_] := …)]` leaves `f` defined in the
global scope, so a later cell sees it. wolframscript keeps it inside the
`DynamicModule` the Manipulate wraps itself in and `f[3]` echoes back. Woxi has
no such module, and its controls re-resolve the body on every frame, so the
definitions have to outlive the call.

Running `Initialization` at evaluation time also resolves control bounds that
reference what it defines. A text front end never displays the
`DynamicModule`, so wolframscript never runs `Initialization` at all and
evaluates such a bound against an undefined symbol:

```sh
wolframscript -code 'Manipulate[func[[n]], {{n, 1, "which"}, 1, Length[func], 1}, Initialization :> (func = {a, b, c, d})]'
# Manipulate[func[[n]], {{n, 1, which}, 1, 0, 1}, Initialization :> (func = {a, b, c, d})]
woxi eval 'Manipulate[func[[n]], {{n, 1, "which"}, 1, Length[func], 1}, Initialization :> (func = {a, b, c, d})]'
# Manipulate[func[[n]], {{n, 1, which}, 1, 4, 1}, Initialization :> (func = {a, b, c, d})]
```

Woxi matches the *notebook* here rather than the CLI: `4` is the slider
maximum a front end shows (this is the shape of the "Function Explorer 3D"
Demonstration), and Woxi Studio reads its controls out of exactly this
expression. A `0` conformant with wolframscript would leave that
Demonstration with an empty slider.

### `Get` returns raw text

`Get` returns an `Expr::Raw`, so `List @@ Get["PacletInfo.m"]` fails, and
`System`Private`$InputFileName` is never set, so a package cannot locate its
own directory. `DumpSave` is unimplemented, so a paclet's MX fast-load path
can never work.


## Lists, associations and structured objects

### `Permutations::toobig` triggers at a fixed size

WL refuses a `Permutations` result with `::toobig` when it would not fit in
the machine's free memory, and builds its packed array otherwise. Woxi's
owned `Expr` tree costs far more per element, so it refuses anything over
2^26 nodes. That means `Permutations[Range[11]]` (479M nodes) is refused
even where wolframscript, with enough RAM, returns all 39916800 permutations.
`::fac` and `::len` do not depend on memory, and those match.

### ListCorrelate / ListConvolve: the 7th argument (a level specification)

The overhang path (`k` / `{kL, kR}`, padding, generalized `g`/`h`) now
matches wolframscript for a kernel and array of equal rank ≥ 2 too — one
offset per dimension, each a scalar or `{kL, kR}` pair, with scalar padding
or cyclic wraparound extending the edges per dimension. Only a per-dimension
padding *array* and a rank mismatch between kernel and data still leave the
call unevaluated. The 7th argument, a level specification, remains
unimplemented for every rank:

```sh
wolframscript -code 'ToString[ListCorrelate[{x, y}, {a, b, c}, 1, p, Times, Plus, 1], InputForm]'
# {a*x + b*y, b*x + c*y, c*x + p*y}
woxi eval 'ListCorrelate[{x, y}, {a, b, c}, 1, p, Times, Plus, 1]'
# ListCorrelate::argb: called with 7 arguments; between 2 and 6 arguments are expected.
```

### `MusicScale` of a name stays raw

wolframscript resolves a named scale to its association form; Woxi leaves it
as written, so it also prints that way in messages:

```sh
wolframscript -code 'InputForm[MusicScale["CMajor"]]'
# MusicScale[<|Name -> Major, Tonic -> MusicPitch[<|Key -> C, Accidental -> 0|>]|>]
woxi eval 'InputForm[MusicScale["CMajor"]]'
# MusicScale["CMajor"]
```

`MusicPlot[MusicScale["CMajor"]]` therefore reports
`… instead of MusicScale[CMajor].` where wolframscript writes
`… instead of -MusicScale-.`

### `MusicNotation` is a Woxi extension

`MusicScore` and `MusicPlot` accept `MusicNotation -> "PianoRoll" |
"SheetMusic"` (default `Automatic`, the piano roll), which is not yet part of
the Wolfram Language. wolframscript answers it with `MusicScore::optx`, and
`Options[MusicScore]` lists it only in Woxi:

```sh
wolframscript -code 'Options[MusicScore]'   # {MusicTempo -> Automatic}
woxi eval 'Options[MusicScore]'             # {MusicTempo -> Automatic, MusicNotation -> Automatic}
```

### Total groups negative levels globally rather than per parent

```sh
wolframscript -code 'Total[{{1, 2}, {3, {4, 5}}}, {-1}]'   # {3, {3, 9}}
woxi eval 'Total[{{1, 2}, {3, {4, 5}}}, {-1}]'             # {{1, 2}, {3, 9}}
```

Woxi reads `{-1}` as the deepest level of the whole expression — level 3 here,
so it answers `Total[…, {3}]`. wolframscript sums the depth-1 parts grouped by
their immediate parent, which leaves the depth-1 atom `3` alone and reduces
only `{4, 5}`.

Every other head measures a negative level as the part's own depth; `Total`
is the exception because its traversal also carries `AllowedHeads` and the
head-preservation rules.

### AssociationThread does not take a scalar key

```sh
wolframscript -code 'AssociationThread[3, {1, 2}]'      # <|3 -> 2|>
wolframscript -code 'AssociationThread["ab", {1, 2}]'   # <|ab -> 2|>
woxi eval 'AssociationThread[3, {1, 2}]'                # AssociationThread[3, {1, 2}]
```

A scalar *value* is shared across the keys, but a scalar *key* against a list
of values keeps only the last value, which is peculiar enough that
generalising from two examples would be guessing. Left unevaluated on purpose;
it echoes rather than aborting.

### Mixed association/list set operations

`Union`, `Intersection` and `Complement` mixing an association with a list stay
unevaluated in both engines, but WL emits `::heads` where Woxi emits
`::normal`.

### `Take` with a zero start index

`Take[{1,2,3}, {0, -1}]` and `{0, 3}` are `{}` in wolframscript and an error in
Woxi. Every other `{0, n}` errors in WL too, so the rule there is obscure.

### SparseArray slices and `Select`

- `sa[[1]]` and `sa[[All, 2]]` return a dense list (or, for `[[All, 2]]`,
  garbage) where WL returns a **sparse** sub-array.
- `Select[sa, pred]` returns a broken `SparseArray[]`. WL returns a dense list
  when the background value fails the predicate and stays sparse when it
  passes — a predicate-dependent choice.
- `SparseArray op list` and `SparseArray + SparseArray` return a dense list in
  Woxi where WL keeps the result sparse. The values match through `Normal`.
- `Map[f, sa]` at rank ≥ 2 densifies; WL keeps the sub-rows sparse.
- `AdjacencyMatrix[g]` returns a dense matrix rather than a `SparseArray`, so
  `AdjacencyMatrix[g] // MatrixForm` prints differently.

### `CoefficientArrays` linear-term entries are stored in an unsorted order

**Not reproducible.** For a single polynomial WL stores the degree-1
`SparseArray` entries in an internal order that is not the variable order, so
the raw `SparseArray[…]` (e.g. its `InputForm`) differs while `Normal` agrees:

```sh
wolframscript -code 'CoefficientArrays[x + y + z, {x, y, z}] // InputForm'
# {0, SparseArray[Automatic, {3}, 0, {1, {{0, 3}, {{2}, {3}, {1}}}, {1, 1, 1}}]}
woxi eval 'CoefficientArrays[x + y + z, {x, y, z}] // InputForm'
# {0, SparseArray[Automatic, {3}, 0, {1, {{0, 3}, {{1}, {2}, {3}}}, {1, 1, 1}}]}
```

The order depends on the variable count and the order the variables are
listed in (`{x, y, z}` → `2, 3, 1`; `{z, x, y}` → `2, 1, 3`;
`{x, y, z, w}` → `2, 3, 1, 4`), not on the polynomial's terms. Woxi sorts.

### `Threaded[scalar]` gives garbage

```sh
wolframscript -code 'Threaded[1] + {1,2,3}'
# Threaded::rarray … and the operation unevaluated as {1, 2, 3} + Threaded[1]
woxi eval 'Threaded[1] + {1,2,3}'
# {Threaded[2], Threaded[3], Threaded[4]}
```

The `Threaded` argument must be rank ≥ 1; a scalar inner is invalid. A
two-argument special case was tried and reverted because it turned the
three-argument case from garbage into a crash.

### `Threaded` of a shorter list is not rejected

```sh
wolframscript -code '{1, 2} + Threaded[{1}]'
# Threaded::thrdims … and {1, 2} + Threaded[{1}] unevaluated
woxi eval 'Plus[{1, 2}, Threaded[{1}]]'
# {Threaded[{2}], Threaded[{3}]}
```

The infix form emits `Thread::tdlen` instead. `Threaded[list]` must match
the trailing dimensions of the other operand, and `try_threaded_op` does not
check that.

### An infix `Plus`/`Times` chain threads over lists pairwise

```sh
wolframscript -code 'd + {b, c} + {a}'
# Thread::tdlen … d + {a} + {b, c}
woxi eval 'd + {b, c} + {a}'
# Thread::tdlen … {a} + {b + d, c + d}
```

The parser keeps `a + b + c` as a chain of `BinaryOp` nodes and
`evaluate_binary_op_chain` combines them two at a time, so `d` is broadcast
into `{b, c}` before the length mismatch with `{a}` is seen. `Plus[d, {b, c},
{a}]` threads all three at once and matches. Fixing it means making the chain
evaluator n-ary for the Flat heads.

### `ReplaceAll` head replacement in two exotic corners

- Association under a symbol blank: Woxi descends into keys and values keeping
  the association form (`<|f[k] -> f[v]|>`); WL gives
  `f[Association][k -> f[v]]`, leaving keys and the inner `Rule` head alone.
- Held infix: `Hold[2+a] /. x_Symbol :> f[x]` leaves the inner *held* `Plus`
  head unrewritten (`f[Hold][2 + f[a]]` against WL's
  `f[Hold][f[Plus][2, f[a]]]`).

### A failed nested `Part` rebuilds the whole chain

wolframscript leaves `Part[m, 3]` unevaluated and applies the next index to
*that* expression, so `m[[3]][[1]]` returns `m` and `m[[1]][[1]][[1]]` reports
`Part::partd: Part specification 1[[1]]`. Woxi rebuilds the chain against the
base and prints `{{1, 2}, {3, 4}}[[1,1,1]]`. Both agree on every in-range read;
this shows only on the error path.

### `DateInterval` canonical form

Woxi canonicalizes to a 4-argument form with `"Day"` granularity; current
wolframscript produces a **6-argument** form and derives granularity from the
input (`Year` for a pair of year `DateObject`s). Both the arity and the
granularity diverge. The 2- and 3-argument constructor calls fall through
unevaluated.

### `Duration` of a `Video`, `DateInterval` or `Play`

Valid objects with no message, but unmeasured — silently unevaluated. An
interval-form `SoundNote` has absolute-placement semantics that are also not
modelled.

### `InterpolatingFunction`'s raw InputForm

WL prints its internal representation
(`InterpolatingFunction[{{1, 3}}, {5, 3, 0, {3}, {3}, 0, …}, …]`) while Woxi
prints `InterpolatingFunction[domain, data, order]`. The property queries
(`"Domain"`, `"Grid"`, `"ValuesOnGrid"`, …) all match.

### `Compile` produces a different object

Woxi's two-part `CompiledFunction[specs, body]` is nothing like WL's
eight-part bytecode object, so the printed form can never match. The binding
rules (untyped defaults to `_Real`, element-by-element conversion of rank-n
real arguments, result type following the signature) do conform.

### `TimeSeries` and `TemporalData`

The installed wolframscript (Tabular-backed internals) disagrees with Woxi's
verified tests on several points — 0-based `"Times"`, `Length` counting points,
`Part` giving a `{time, value}` pair, `"Values"` as a `TabularColumn`. These
are **version drift**, not Woxi bugs. Likewise `Mean`/`Variance`/`Median`/
`StandardDeviation` of a whole-valued series return machine reals in
wolframscript (its packed `Integer64` column being promoted) and stay exact in
Woxi.

Still genuinely open: WL keeps a `TemporalData[Automatic, {…}, False, version]`
internal form that Woxi normalizes to `TimeSeries`, and `MovingMap[f, x, 0]`
should report `MovingMap::wsizen` (WL then returns a mangled `DisplayForm`
value that is not worth reproducing, so Woxi's width check is silent).

### `Dataset` variance

`Variance[Dataset[{1., 2., 3.}]]` is `1` against WL's `1.0000000000000002` —
the float-noise class above.


## Patterns, strings and parsing

### A tagged assignment cannot have a curried left-hand side

```sh
wolframscript -code 'f /: g[a_][f[_]] := 1; UpValues[f]'
# {HoldPattern[g[a_][f[_]]] :> 1}
woxi eval 'f /: g[a_][f[_]] := 1; UpValues[f]'
# {HoldPattern[g[a_, f[_]]] :> 1}
```

`tag /: lhs := rhs` reads its left-hand side as a single call, so the two
bracket groups of `g[a_][f[_]]` are merged into one argument list and the
rule that gets stored fires on `g[1, f[2]]` instead. The upvalue store keys a
rule by one head and a flat parameter list, which a subvalue-shaped rule has
nowhere to go in; `g[a_][f[_]] := 1` *without* a tag is fine (it becomes a
`SubValues` entry). Reached through `Hold`/`ReleaseHold` the same definition
reports `TagSetDelayed::tagpos` and defines nothing.

### `x_ y_` is mis-parsed

```sh
wolframscript -code 'Hold[x_ y_] // FullForm'
# Hold[Times[Pattern[x, Blank[]], Pattern[y, Blank[]]]]
woxi eval 'Hold[x_ y_] // FullForm'
# Hold[Pattern[x, BlankSequence[y_]]]
```

The parser merges the two `_` across the space into a `BlankSequence` whose
head becomes `y_`. `x_ + y_` parses correctly; only the implicit-times
juxtaposition is broken, so any implicit-times product of patterns is affected
(`ReplaceList[a b c, x_ y_ :> {x,y}]` gives `{}` where WL gives six splits).
Writing `Times[x_, y_]` explicitly works.

### A bare `Span` cannot follow a `Rule`

`a -> 2 ;; 3` fails to parse ("expected SpanSep"); WL reads it as
`Rule[a, Span[2,3]]` because `;;` binds tighter than `->`. Parenthesizing
works. This blocks the documented `StringExtract[s, sep -> i ;; j]` form.
`X::y` on the right of `->`/`:>` fails the same way.

### `f[{1, 2} // Total -> 3]` does not parse

WL reads it as `{1,2} // (Total -> 3)`, i.e. the postfix *function* is the rule.
Making the rule's left side a postfix application parses it with the wrong
precedence, which is worse than the parse error.

### `(a|b)|c` loses its grouping

Explicit parentheses should give `Length` 2 (nested) in WL; Woxi's parser
produces the same flat 3-argument form as `a|b|c`.

### Parse-time messages do not exist

`{a,,b}` and `f[a,]` parse to `Null` as they should, but wolframscript also
prints `Syntax::com: Warning: comma encountered with no adjacent expression.`
once per omitted comma, at read time. Woxi has no parse-time message channel at
all — wiring one through the AST conversion would duplicate the warning,
because a cell is parsed more than once per evaluation. The same absence means
a `\!\(…\)` escape with a bad form tag is left literal rather than reported as
`MakeExpression::boxfmt`.

### Empty matches are dropped

```sh
wolframscript -code 'StringCases["abcd", ___]'                    # {abcd, }
woxi eval 'StringCases["abcd", ___]'                              # {abcd}

wolframscript -code 'StringCases["abcd", ___, Overlaps -> True]'  # {abcd, bcd, cd, d, }
```

`StringPosition["abcd", ___]` likewise drops WL's trailing `{5, 4}`. The
default and `Overlaps -> True` paths exclude empty regex matches and never
visit the position one past the last character; the `Overlaps -> All` path
handles them correctly.

### `Overlaps -> All` with zero-width assertions

Any zero-width assertion other than a leading `StartOfString` or trailing
`EndOfString` is evaluated against the probed slice rather than the whole
string, so `WordBoundary ~~ __` under `Overlaps -> All` diverges.

### String pattern back-references need backtracking

A repeated pattern name is a back-reference (`x_ ~~ x_` matches two equal
characters). The regex engine has none, so Woxi emits a duplicate capture and
compares the two after the match — which cannot backtrack to find an
*alternate* assignment. Fine for fixed-width single-character blanks;
variable-width (`x__ ~~ x__`) greedy assignment can miss a valid split.

### Look-around and back-references in `RegularExpression`

`RegularExpression["(?=[A-Z])"]` cannot compile (Woxi now reports
`RegularExpression::badregex` and leaves the call unevaluated rather than
aborting). WL supports look-around and `\1`.

### `DatePattern` element coverage

`AMPM`, `Quarter`, `YearShort` and other element names are unimplemented, as is
`DateDelimiters`-style probing beyond the four verified separators `[/-.:]`.

### `Longest`/`Shortest` in a definition

The rule path (`/.`, `Cases`, `MatchQ`) honours them; the **definition** path
strips them, so `f[Longest[x__], y__] := …` splits shortest-first.

### Unimplemented argument forms found by diffing against `SyntaxInformation`

Woxi's arity table can be diffed mechanically against wolframscript's own
declared signatures:

```wolfram
(* for each implemented function name *)
"ArgumentsPattern" /. SyntaxInformation[Symbol["System`" <> name]]
```

Counting the required and optional slots (skipping anything with
`OptionsPattern` or a `BlankSequence`, which are unbounded) and comparing
against Woxi's declared maximum turns up every documented argument form that
is missing. The sweep originally found 80; the 61 below are what is left.

Each one is a form wolframscript accepts and Woxi rejects on arity — so the
failure mode is an `::argt` / `::argb` / `::argx` message rather than a wrong
answer, which makes them safe but individually invisible.

| Function | Woxi | WL up to | `ArgumentsPattern` |
| --- | --- | --- | --- |
| `AiryAiZero` | 1–1 | 2 | `{_, _.}` |
| `AiryBiZero` | 1–1 | 2 | `{_, _.}` |
| `AngerJ` | 2–2 | 3 | `{_, _, _.}` |
| `ArcCurvature` | 2–2 | 3 | `{{__}, _, _.}` |
| `BesselYZero` | 2–2 | 3 | `{_, _, _.}` |
| `BooleanMaxterms` | 2–2 | 3 | `{_, _., _.}` |
| `BooleanMinterms` | 2–2 | 3 | `{_, _., _.}` |
| `CenterArray` | 1–3 | 4 | `{_, _., _., _.}` |
| `CharacterName` | 1–1 | 2 | `{_, _.}` |
| `CircleThrough` | 1–1 | 3 | `{{__}, _., _.}` |
| `ConstantArray` | 2–2 | 3 | `{_, _, _.}` |
| `CoordinateBoundingBoxArray` | 1–3 | 4 | `{{_, _}, _., _., _.}` |
| `CoordinateBoundsArray` | 1–3 | 4 | `{{__}, _., _., _.}` |
| `Correlation` | 1–2 | 3 | `{_, _., _.}` |
| `Counts` | 1–1 | 2 | `{_, _.}` |
| `Covariance` | 1–2 | 3 | `{_, _., _.}` |
| `DigitCount` | 1–3 | 4 | `{_, _., _., _.}` |
| `DigitSum` | 1–2 | 3 | `{_, _., _.}` |
| `FindLinearRecurrence` | 1–1 | 2 | `{_, _.}` |
| `FrenetSerretSystem` | 2–2 | 3 | `{{__}, _, _.}` |
| `GammaDistribution` | 2–2 | 4 | `{_, _, _., _.}` |
| `GompertzMakehamDistribution` | 2–2 | 4 | `{_, _, _., _.}` |
| `GroupOrbits` | 2–2 | 3 | `{_, Optional[{__}], _.}` |
| `GroupStabilizer` | 2–2 | 3 | `{_, {__}, _.}` |
| `Groupings` | 2–2 | 3 | `{_, _, _.}` |
| `HarmonicNumber` | 1–2 | 3 | `{_, _., _.}` |
| `Head` | 1–1 | 2 | `{_, _.}` |
| `ImageAdjust` | 1–2 | 4 | `{_, _., Optional[{_, _}], Optional[{_, _}]}` |
| `Inner` | 3–4 | 5 | `{_, _, _, _., _.}` |
| `InverseChiSquareDistribution` | 1–1 | 2 | `{_, _.}` |
| `InverseErf` | 1–1 | 2 | `{_, _.}` |
| `InverseGammaDistribution` | 2–2 | 4 | `{_, _, _., _.}` |
| `InverseGaussianDistribution` | 2–2 | 3 | `{_, _, _.}` |
| `KendallTau` | 2–2 | 3 | `{_, _., _.}` |
| `Latitude` | 1–1 | 2 | `{_., _.}` |
| `LatitudeLongitude` | 1–1 | 2 | `{_., _.}` |
| `LegendreP` | 2–3 | 4 | `{_, _, _., _.}` |
| `LegendreQ` | 2–3 | 4 | `{_, _, _., _.}` |
| `ListConvolve` | 2–6 | 7 | `{_, _, _., _., _., _., _.}` |
| `ListCorrelate` | 2–6 | 7 | `{_, _, _., _., _., _., _.}` |
| `Longitude` | 1–1 | 2 | `{_., _.}` |
| `MaximalBy` | 1–3 | 4 | `{_, _., _., _.}` |
| `MeijerG` | 3–3 | 4 | `{{{___}, {___}}, {{___}, {___}}, _, _.}` |
| `MinimalBy` | 1–3 | 4 | `{_, _., _., _.}` |
| `NotebookDirectory` | 0–0 | 1 | `{_.}` |
| `ParentDirectory` | 0–1 | 2 | `{_., _.}` |
| `PerfectNumber` | 1–1 | 2 | `{_, _.}` |
| `PolyLog` | 2–2 | 3 | `{_, _, _.}` |
| `PositionLargest` | 1–2 | 3 | `{_, _., _.}` |
| `Precedence` | 1–1 | 2 | `{_, _.}` |
| `QuantityQ` | 1–1 | 2 | `{_, _.}` |
| `RiceDistribution` | 2–2 | 3 | `{_, _, _.}` |
| `SpearmanRho` | 2–2 | 3 | `{_, _., _.}` |
| `Subsequences` | 1–2 | 3 | `{_, _., _.}` |
| `SubsetReplace` | 1–2 | 3 | `{_, _., _.}` |
| `Symmetrize` | 1–1 | 2 | `{_, _.}` |
| `SyntaxQ` | 1–1 | 2 | `{_, _.}` |
| `ToBoxes` | 1–1 | 2 | `{_, _.}` |
| `Uncompress` | 1–1 | 2 | `{_, _.}` |
| `Unique` | 0–1 | 2 | `{_., Optional[{__}]}` |
| `WeberE` | 2–2 | 3 | `{_, _, _.}` |

Notes on the ones already looked at:

- `MinimalBy` / `MaximalBy` / `DigitCount`: wolframscript appears to *ignore*
  the extra trailing argument (`MinimalBy[{{1,2},{3,1},{2,5}}, Last, 2, x]`
  gives the same answer as without `x`), so there may be nothing to match.
- `Counts[list, n]`: wolframscript rejects a non-list second argument with
  `Counts::invl`, so the slot is not a useful form either.
- `PositionLargest`: the remaining slot is an *orderfun*, but
  wolframscript rejects `Greater` and `Less` there with `::nord3` (it wants
  an `Order`-style comparison returning ±1/0). `PositionSmallest[list, n, p]`
  is implemented for such comparisons (e.g. `AlphabeticOrder`); `Greater` /
  `Less` stay unevaluated without the `::nord3` message.
- The generalized distribution constructors (`GammaDistribution` 2→4,
  `InverseGammaDistribution` 2→4, `GompertzMakehamDistribution` 2→4,
  `RiceDistribution` 2→3, `InverseGaussianDistribution` 2→3,
  `InverseChiSquareDistribution` 1→2) have real closed forms, but their
  output is `Piecewise` wrapping `GammaRegularized`, `LaguerreL` and nested
  `Gamma` quotients. Matching wolframscript there means matching its exact
  symbolic canonicalization.

```sh
wolframscript -code 'ToString[PDF[GammaDistribution[a, b, g, m], x], InputForm]'
# Piecewise[{{(g*((-m + x)/b)^(-1 + a*g))/(b*E^((-m + x)/b)^g*Gamma[a]), x > m}}, 0]
woxi eval 'PDF[GammaDistribution[a, b, g, m], x]'
# GammaDistribution::argrx: GammaDistribution called with 4 arguments; 2 arguments are expected.
```

A second sweep, comparing declared arities against the arities the dispatch
actually handles, found these still open: `Transliterate[s, scheme]`,
`SumConvergence[f, n, Assumptions -> …]`, `LexicographicSort[list, ord]`,
`SavitzkyGolayMatrix[r, order, deriv]`, and the 3-argument forms of
`ClusteringComponents`, `ComponentMeasurements`, `ImageValue`,
`WarpingCorrespondence`, plus `EventSeriesLookup` 3/4-argument.

### Declared but dead options

For each option a head declares in `Options[f]`, setting it should change the
answer. These do not:

- `NIntegrate`'s `MaxRecursion`, `MinRecursion`, `WorkingPrecision`,
  `MaxPoints`, `PrecisionGoal`, `AccuracyGoal`.
- `NSum` and `NProduct` accept `WorkingPrecision` and ignore it.
- `NDSolve`'s `WorkingPrecision` and `PrecisionGoal`: the solution is always
  machine precision. The Lorenz system with `WorkingPrecision -> 20` gives
  `x[1] = -12.49383128851448835…` in WL and `-12.493889543181242` here.
- `Root[…, ExactRootIsolation -> True]` stays unevaluated.
- `D[f[x] g[x], x, NonConstants -> {f}]` differentiates `f` anyway; WL holds it
  as `D[f[x], x, NonConstants -> {f}]`. (The plain
  `D[x y, x, NonConstants -> {y}]` case works.)
- `SetOptions[Plot, PlotStyle -> …]` maintains the option list but does not
  reach the renderer.
- `Options[ListPlot]` is `{}` against WL's 70 entries; `Plot3D` 85,
  `Histogram` 61, `BarChart` 64, `Graphics` 39. Only `Plot` has a table.
- `OptionValue[Plot, Axes]` gives `Axes` instead of `True`, and
  `$DisplayFunction`/`$PerformanceGoal` are undefined so
  `OptionValue[Plot, DisplayFunction]` returns the unresolved symbol.
- `OptionValue`'s `::rep` message for a non-rule element of the explicit option
  list is not emitted, and WL's own rule for when it fires is unclear
  (`OptionValue[Plot, Axes, Hold]` reports it, `OptionValue[Plot, 5, Frame]`
  does not).

### Exact `Eigenvectors` with `Root` eigenvalues stop at 5x5

Beyond 5x5, an exact integer matrix whose eigenvalues are not integers keeps
`Eigenvectors`/`Eigensystem` unevaluated: the elimination over algebraic
numbers is too slow in Woxi, while Wolfram returns `Root`-object vectors.
`Eigensystem[m, k]` with k ≤ 2 computes just those vectors, up to 8x8; their
entries are nested `Root` expressions, not Wolfram's reduced form (equal
under `N`).


### `Eigenvalues` of a matrix without radical roots

Woxi lists the eigenvalues as `Root` objects in `Root` order; wolframscript
writes them as radicals when it can and orders those its own way:

```sh
wolframscript -code 'Eigenvalues[{{0, 0, 0, 2}, {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 1}}]'
# {Root[-2 + 2*#1 - 2*#1^2 + #1^3 & , 1, 0], Root[… , 3, 0], Root[… , 2, 0], -1}
woxi eval 'Eigenvalues[{{0, 0, 0, 2}, {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 1}}]'
# {Root[-2 + 2*#1 - 2*#1^2 + #1^3 & , 1, 0], -1, Root[… , 2, 0], Root[… , 3, 0]}
```

`Eigenvectors` does follow Wolfram's order (decreasing absolute value, then
increasing real part, then decreasing imaginary part); `Eigenvalues` of the
`{{0, 0, 0, -1}, …}` shift matrix in wolframscript is in neither order.


### `Solve` of an arbitrary-precision polynomial answers at machine precision

```sh
wolframscript -code 'Solve[N[z^3 - 2 z + 1/3, 10] == 0, z]'
# {{z -> -1.4911540841057329132`9.544…}, …}
woxi eval 'Solve[N[z^3 - 2 z + 1/3, 10] == 0, z]'
# {{z -> -1.491154084105733}, …}
```

The roots are found numerically at machine precision; Wolfram carries the
coefficients' precision (and its loss) into them.


### `SmoothKernelDistribution` stays symbolic and is evaluated exactly

wolframscript turns `SmoothKernelDistribution[data, …]` into a binned
`DataDistribution["SmoothKernel", …]` and interpolates its `PDF`/`CDF` from
that grid; Woxi keeps the call and sums the kernels exactly, so values agree
to about six digits (`PDF[SmoothKernelDistribution[{1., 2., 3.5}, 0.5], 1.5]`
is `0.32271721782167967` there, `0.3227168528427011` here). A symbolic
point stays unevaluated.


### `Det` of a rational-coefficient matrix keeps a common factor in Wolfram

```sh
wolframscript -code 'Det[Table[1/(i + j + 1), {i, 3}, {j, 3}] - x IdentityMatrix[3]]'
# (2 - 9510*x + 511200*x^2 - 756000*x^3)/756000
woxi eval 'Det[Table[1/(i + j + 1), {i, 3}, {j, 3}] - x IdentityMatrix[3]]'
# (1 - 4755*x + 255600*x^2 - 378000*x^3)/378000
```

The value is the same; Wolfram's elimination leaves the numerator and
denominator with a common factor that `Together` removes.


### `Flat` without `OneIdentity` binds singletons bare

For a head that is `Flat` but not `OneIdentity`, Wolfram matches a blank that
takes a single argument as `h[arg]` first, then bare, even when the argument
counts are equal. Woxi always binds the bare argument. `Dot`,
`NonCommutativeMultiply`, `Plus` and `Times` are all `OneIdentity`, so this
only shows up for user heads.

```sh
wolframscript -code 'SetAttributes[h, Flat]; h[a, b, c] /. h[x_, y_] :> {x, y}'  # {h[a], h[b, c]}
woxi eval 'SetAttributes[h, Flat]; h[a, b, c] /. h[x_, y_] :> {x, y}'            # {a, h[b, c]}
wolframscript -code 'SetAttributes[h, Flat]; h[a, b] /. h[x_, y_] :> {x, y}'     # {h[a], h[b]}
wolframscript -code 'SetAttributes[h, Flat]; ReplaceList[h[a, b, c], h[x_, y_] :> {x, y}]'
# {{h[a], h[b, c]}, {h[a, b], h[c]}, {a, h[b, c]}, {h[a, b], c}}
```

With `Orderless` added, `ReplaceList` lists the wrapped readings first too,
but `/.` picks a bare one (`{b, g[a, c]}` for `g[a, b, c]`), so the rule is
not just "wrapped first".

### A minus right after `.` is accepted

Wolfram rejects `a . -b` (and `a . b . -c`) as a syntax error; Woxi reads it
as `Dot[a, -b]`.

## Messages and error handling

### `RecurrenceTable` of a transcendental system prints no `Solve` messages

A nonlinear system such as `RecurrenceTable[{r[n + 1] == 2 r[n] - r[n - 1]
- 20 Cos[t[n]]/(r[n] - 185.7)^2, t[n + 1] == t[n] + 2 Pi (185.7/r[n])^(3/2),
r[0] == 70, r[1] == r[0], t[0] == 0}, {r, t}, {n, 1, 4}]` gives the same
table in both, but wolframscript first tries to `Solve` the equations and
prints `Solve::incnst` and `Solve::ifun`. Woxi steps the recurrences
directly and prints nothing.

### Graph functions say nothing about a non-graph argument

`EdgeList[{1, 2}]` and `VertexList[Graph[{{a, b}}]]` (a `Graph` that did not
evaluate) come back unevaluated in both, but wolframscript also prints
`EdgeList::graph: A graph object is expected at position 1 in …`. A bare
symbol (`EdgeList[x]`) is silent in both. Woxi has no `::graph` message for
any graph function.

### `RealDigits` in a non-integer base stays unevaluated

`RealDigits[1/7, 2.5]` lists 39 digits of the greedy expansion in base 2.5 in
wolframscript; Woxi only implements integer bases and leaves the call
unevaluated.

### `RealDigits` of an arbitrary-precision number: the last digit

`RealDigits[N[Pi, 30]]` ends in `…3, 2, 8` in wolframscript, `…3, 2, 7` in
Woxi (Pi's 30th digit is 7, followed by 9); `RealDigits[N[999999/1000000, 3]]`
is `{{1, 0, 0}, 1}` against Woxi's `{{9, 9, 9}, 0}`. But `RealDigits[N[E, 20]]`
truncates in both (…3, followed by 6), so wolframscript neither always
rounds nor always truncates — the digit comes from its binary mantissa and
guard bits. **Not reproducible**; Woxi truncates, which agrees on most probes.

### `Min`/`Max` `::nord` count for nested lists and real infinities

`Min`/`Max` emit one `::nord` per evaluation that compares `ComplexInfinity`
(the call itself unless it is a lone list, plus the re-evaluation of a
changed result), which matches wolframscript on every flat probe. Two shapes
still differ:

```sh
# one extra pass for a list nested two deep:
wolframscript -code 'Min[{x, y}, {3, {ComplexInfinity}}]'   # 3 messages
woxi eval 'Min[{x, y}, {3, {ComplexInfinity}}]'             # 2 messages
# one fewer when a real infinity absorbs the numbers:
wolframscript -code 'Max[1, ComplexInfinity, Infinity]'      # 1 message
woxi eval 'Max[1, ComplexInfinity, Infinity]'                # 2 messages
```

### `Partition` reports `ilsmp` where the padded forms allow a zero block

With three or more arguments a block size of `0` is legal — the offset says
how many empty blocks come out — so wolframscript rejects a bad size there
with `ilsmn` ("non-negative"), keeping `ilsmp` ("positive") for the
two-argument form. Woxi says `ilsmp` for every arity, and rejects a literal
`0` outright once an alignment/padding argument follows:

```sh
wolframscript -code 'Partition[{a,b,c}, n, n, {1,1}, {}]'
# Partition::ilsmn: Single or list of non-negative machine-sized integers …
woxi eval 'Partition[{a,b,c}, n, n, {1,1}, {}]'
# Partition::ilsmp: Single or list of positive machine-sized integers …

wolframscript -code 'Partition[{a,b,c}, 0, 1, {1,1}, {}]'  # {{}, {}, {}}
woxi eval 'Partition[{a,b,c}, 0, 1, {1,1}, {}]'            # Partition::ilsmp
```

The unpadded `Partition[{a,b,c}, 0, 1]` agrees (`{{}, {}, {}, {}}`) — note
that the padded form yields one block *fewer*, so the two count differently.

### Expressions inside message text print in InputForm

WL formats a message's substituted expressions the way the front end shows
them (`Times` as a space), Woxi formats them like `InputForm` (`*`):

```sh
wolframscript -code '2 x = 5'   # Set::write: Tag Times in 2 x is Protected.
woxi eval '2 x = 5'             # Set::write: Tag Times in 2*x is Protected.
```

The message tag, its arguments and the returned value all agree; only the
rendering of an embedded expression differs, and it does so for every
message that embeds one (`Set::write`, `ImageAdjust::arg2`, …). Each
message site formats with `expr_to_string`; matching WL means routing them
through the OutputForm renderer instead.

### A refused base literal does not also report a syntax error

wolframscript treats `base^^digits` with a base outside 2–36, or with a digit
the base cannot spell, as text it could not read: it reports the reason and
then the syntax error that reading stopped at. Woxi reports the reason and
returns `$Failed` without the second message.

```sh
wolframscript -code '1^^111'
# General::base: Requested base 1 in 1^^111 should be between 2 and 36.
# ToExpression::sntx: Invalid syntax in or before "1^^111 ".
# $Failed
woxi eval '1^^111'
# General::base: Requested base 1 in 1^^111 should be between 2 and 36.
# $Failed
```

Same for `General::digit` (`2^^1012`). This is the general shape of Woxi's
read-time errors: input the grammar rejects reports a parser error of its own
rather than `ToExpression::sntx`/`::sntxi`, and in a script the lines before
it do not run.

### `Read` does not report unreadable input

```sh
wolframscript -code 's = StringToStream["1+"]; Read[s]'
# Read::readt: Invalid input found when reading 1+ from StringToStream[1+].
# $Failed
woxi eval 's = StringToStream["1+"]; Read[s]'
# $Failed
```

The value agrees; only the message is missing. The reader that produces it
sees the text and the requested type but not the stream expression the
message names, so emitting it means plumbing that through.

### Outer does not report mismatched heads

```sh
wolframscript -code 'Outer[f, h[1, 2], {a, b}]'
# Outer::heads: Heads List and h at positions 3 and 2 are expected to be the same.
# Outer[f, h[1, 2], {a, b}]
woxi eval 'Outer[f, h[1, 2], {a, b}]'
# Outer[f, h[1, 2], {a, b}]
```

The result is right — the call stays unevaluated either way — but the
`Outer::heads` message is missing. Every other `Outer` form checked
(per-list levels, general heads, operator form) agrees.

### `Options` does not report an unknown option name

```sh
wolframscript -code 'Options[Graphics[{Disk[]}], Frobnicate]'
# Options::optnf: Frobnicate is not a known option for Graphics.
# {}
woxi eval 'Options[Graphics[{Disk[]}], Frobnicate]'
# {}
```

The value agrees. On a head with no options at all the second argument is
dropped rather than answered, which does not:

```sh
wolframscript -code 'Options[f[1, 2], Axes]'   # Options::optnf: … ; {}
woxi eval 'Options[f[1, 2], Axes]'             # Options[f[1, 2]]
```

A list of option names is not accepted either — `Options[g, {Axes, PlotRange}]`
gives `{}` where WL answers both.

### Rasterize "BoundingBox" and "RasterSize" use Woxi's own layout

```sh
wolframscript -code 'Rasterize[Style["abg", FontSize -> 12], "BoundingBox"]'
# {21, 15, 11}
woxi eval 'Rasterize[Style["abg", FontSize -> 12], "BoundingBox"]'
# {22, 22, 17}
```

The sizes come from Woxi's SVG layout, so text and formulas measure
differently from the front end. The baseline is approximated: half the
height for graphics (WL: 182 of 359 for `Graphics[Disk[]]`) and three
quarters for everything else. WL's "RasterSize" is also at twice the
resolution (`{720, 719}` for `Graphics[Disk[]]`); Woxi rasterizes at 1x.

### Twenty-two heads raise a hard error for a bad first argument

```sh
wolframscript -code 'ImageCrop[{1, 2, 3}, -1]'
# ImageCrop::imgvinv: Expecting an image, graphics or video instead of {1, 2, 3}.
# ImageCrop[{1, 2, 3}, -1]
woxi eval 'ImageCrop[{1, 2, 3}, -1]'
# Error: Evaluation error: ImageCrop: first argument is not an Image
```

A sweep of the 855 heads whose arity admits two arguments, against nine
degenerate argument pairs, found 227 calls that abort the enclosing evaluation
instead of reporting. What remains is this one shape, spread over `While`,
`TuringMachine`, `Find`, `ReadList`, `Piecewise`, `ImageApply`,
`ImageAssemble`, `ImageCollage`, `ImageCompose`, `ImageCrop`, `NMinimize`,
`NMaximize`, `FindSequenceFunction`, `RealDigits`, `Dt`, `FrenetSerretSystem`,
`ArcCurvature`, `MinimalPolynomial`, `Root`, `NumberFieldSignature`,
`WordCounts` and `DivisorSigma`.

The argument is nonsense in every case, so no correct result is lost — but a
hard error takes the whole script down where a message does not.

### An operator form that fails names the flattened call, not what was written

```sh
wolframscript -code 'Select[EvenQ][5]'
# Select::normal: Nonatomic expression expected at position 1 in Select[EvenQ][5].
# Select[EvenQ][5]
woxi eval 'Select[EvenQ][5]'
# Select::normal: Nonatomic expression expected at position 1 in Select[5, EvenQ].
# Select[5, EvenQ]
```

`f[spec][data]` is rewritten to `f[data, spec]` before dispatch, and when the
call then fails both the message and the echo describe the rewrite. The same
happens for `SortBy`, `KeyTake` and `Nearest`, so this is one issue in the
operator-form dispatch rather than a per-function one.

### Iteration constructs substitute their variable instead of binding it

`Table`, `Sum` and `Product` rewrite the iterator symbol into the body rather
than binding it dynamically the way Wolfram does, so a held occurrence is
replaced instead of left standing: `Table[Hold[k], {k, 1, 2}]` gives
`{Hold[1], Hold[2]}` where Wolfram gives `{Hold[k], Hold[k]}`, and
`Sum[Hold[k], {k, 1, 2}]` gives `Hold[1] + Hold[2]` instead of `2 Hold[k]`.
The substitution is capture-avoiding, so a nested `Module`/`With`/`Function`
binder of the same name is left alone; only unbound held positions diverge.

### `$GeoLocation` and location-defaulting messages

Woxi returns `Missing["NotAvailable"]` — matching an offline wolframscript —
but does not emit `$GeoLocation::dloff` or the per-function `Fn::geoloc`.

### Message repetition and 2D layouts

- `DensityPlot`'s degenerate-range message is reported under its internal
  context name (``Visualization`Core`DensityPlot::plld``) and emitted twice;
  Woxi cannot produce that symbol name and leaves `DensityPlot` out of the
  check entirely.
- `PolarPlot`, `ContourPlot3D` and two-parameter `ParametricPlot` emit the
  message twice (a second internal pass); Woxi emits it once.
- `Hash::invhash` is printed twice by wolframscript for an unknown algorithm.
- `Quantity::compat` is emitted twice by wolframscript, without quotes, and its
  two unit names appear in an order that is not input order
  (`Kilograms + Meters` → "Meters and Kilograms"). Woxi quotes them and emits
  once.
- Messages that embed a fraction are rendered as 2D layouts by wolframscript.
  Woxi does the same for user `Message[sym::tag, args]` templates and for
  built-in messages composed with `syntax::format_message_with_expr` /
  `format_message_pieces` (`Select::normal`, `Surd::int`); built-in messages
  assembled with a flat `format!` (e.g. the distribution family, `Take::seqs`)
  still print 1D.
- Message **multiplicity** in general is not comparable: wolframscript
  re-evaluates a failing specification, so it prints some messages twice, and
  applies `General::stop` after three identical ones.

### A stored value is re-evaluated on every read, repeating its messages

Woxi evaluates an `OwnValue` again each time the symbol is read, so a value
that stays unevaluated with a message prints the message again on every use.
WL marks the stored expression as already evaluated and prints it once:

```sh
wolframscript -code 'op = LinearSolve[{{1, 2}, {3, 4}}, Method -> "Cholesky"]; op'
# LinearSolve::herm once
woxi eval 'op = LinearSolve[{{1, 2}, {3, 4}}, Method -> "Cholesky"]; op'
# LinearSolve::herm twice
```

The same missing mark shows when an unevaluated result is passed on: the
pure function in `Prepend[f[DeleteCases[l, #]], #] &@ l[[Quotient[n, 6] + 1]]`
gets `l[[1 + Quotient[n, 6]]]` as `#` and evaluates it again, so the first
`Part::pkspec1` prints twice. It also costs time: a recursion that nests an
unevaluated expression one level deeper per call re-evaluates the whole nest
on every call. A `fromrank[list_, n_]` whose `Mod[n, 6]` argument nests
until `$RecursionLimit` takes ~3s in Woxi and 0.03s in wolframscript. And a
nest built up step by step is re-evaluated whole at each step, which also
costs stack: `Depth[Nest[h[{#}] &, 0, 1500]]` is 3001 in WL but exceeds
`$RecursionLimit` in Woxi (as does the same `Fold` past ~500 steps).

### A too-deep `Part` on a packed array is `Part::partd1`

Which of the two messages wolframscript picks depends on whether the object is
a packed array, a storage detail Woxi has no equivalent of. Woxi always emits
`Part::partd`, which is the unpacked answer:

```sh
# Unpacked — both engines agree:
wolframscript -code '{{1, 2}, {3, 4}}[[1, 1, 1]]'
# Part::partd: Part specification …[[1,1,1]] is longer than depth of object.

# Packed (anything Range, Table or Permutations built) — WL switches message:
wolframscript -code 'Range[5][[1, 1]]'
# Part::partd1: Depth of object {1, 2, 3, 4, 5} is not sufficient for the
# given part specification.
woxi eval 'Range[5][[1, 1]]'
# Part::partd: Part specification {1, 2, 3, 4, 5}[[1,1]] is longer than depth …
```

Packing the literal with ``Developer`ToPackedArray`` and unpacking the
`Range` with ``Developer`FromPackedArray`` swap the two messages over, which
is how the rule was confirmed. The values agree — both engines return the
`Part` expression unevaluated.

### `ArgMax` with an invalid constraint form

`ArgMax[{list}, fn]` — WL emits `ArgMax::consf`, Woxi silently echoes.

### `Failure` prints its formatted message

In script mode wolframscript does not print a `Failure[…]` expression — it
prints the failure's *formatted message*, wrapping every substituted parameter
in `DisplayForm[TagBox[…, Short[#1, 3] &]]`. That is typeset internals leaking
into script mode; Woxi prints the expression and renders `["Message"]` cleanly.
**Deliberately not reproduced** — test a `Failure`'s keys, not its printed form.


### Compiled code reports no `CompiledFunction::cfn`

Woxi runs a `Compile`d body on machine numbers where it can and otherwise
evaluates it symbolically, silently. Wolfram prints `CompiledFunction::cfn:
Numerical error encountered at instruction …; proceeding with uncompiled
evaluation` when machine arithmetic overflows (an escape-time fractal whose
iterate grows without bound) and then continues uncompiled; Woxi keeps going
in machine arithmetic, so the values agree but the message is missing. The
instruction number has no counterpart in Woxi.

## Graphics and plotting

### `Attributes[GraphicsComplex]` (unverified)

Woxi gives `GraphicsComplex` `NHoldRest`, so `N` leaves the primitives'
point indices integers (a demonstration takes `Normal[N@PolyhedronData[…,
"Faces"]]`). The exact attribute list (`ReadProtected`?) was not checked
against wolframscript.

Woxi renders its own SVG; `wolframscript` prints `-Graphics-` and exports
cairo-rendered SVG, so pixel or markup equality is never the goal. What follows
is where the *drawn* result differs.

### Legacy `VectorFieldPlots`` plots carry fewer options

The package's `Graphics` gets `FilterRules[{opts, Options[ListVectorFieldPlot]},
Options[Graphics]]`: the caller's options followed by every default of
`Developer`GraphicsOptions[]` (twice over from `VectorFieldPlot`). Woxi keeps
the caller's options and appends only `AspectRatio -> Automatic` and
`PlotRange -> All`, the two defaults the package changes. The primitives — what
notebooks take apart with `[[1, 3]]` — are identical; only
`InputForm[plot]`'s option tail differs.

Loading the package also prints `General::obspkg` in wolframscript; Woxi
loads it silently.

### The default plot style is not Wolfram's

Every Woxi plot draws its first curve in `#5E81B5` at 1.5 px. Wolfram draws it
in `RGBColor[0.24, 0.6, 0.8]` = (61, 153, 204) at `AbsoluteThickness[2]`. Woxi
is using the `ColorData[97]` palette where `Plot` has its own
`DefaultPlotStyle`, readable from the kernel:

```wolfram
"DefaultPlotStyle" /. (Method /. Charting`ResolvePlotTheme[Automatic, Plot])
```

Woxi already writes the right values in its `.nb` `GraphicsBox` export; only
the SVG renderer disagrees. Changing it moves 153 snapshots and about 40 inline
colour assertions at once.

### `Graphics` with `Axes` clips its edge tick labels

`Graphics[…, Axes -> True, ImageSize -> {400, 400}]` lays its plotting area out
from x = 0 with only a right-hand margin, so the tick label at the range
minimum is centred on x = 0 and half of it falls outside the canvas; the
topmost y label is clipped the same way. Wolfram insets the area far enough for
both. This is the `Graphics` renderer, not the one `Plot` uses.

### `Graphics` ignores `AxesOrigin` and pads a `Full` range less

The `Graphics` renderer draws the x axis at `y = 0` when that is in range and
at the bottom edge otherwise, whatever `AxesOrigin` says, and pads a
`PlotRange -> Full` side by 4 %, where Wolfram extends it to include the axes
origin and pads more. In a `MusicPlot` this drops the pitch label just above
the highest note (the `C5` of `MusicPlot[MusicVoice[{"C4", "B4"}]]`), which
Wolfram shows.

### Tick step from the padded range

Woxi derives the tick step from the *padded* display range, so a `ListPlot`
whose data ends at 6e15 can pick a coarser step than Wolfram, which uses the
data range.

### `ParametricPlot` samples uniformly

`max(PlotPoints, 500)` points, no adaptive refinement (`Plot` has it). On a
51-term trig sum with `PlotPoints -> 100` wolframscript emits 6052 line points
against Woxi's 500. The values agree exactly and the shape is the same; only
small-scale wiggles are under-resolved.

### 2D field plots ignore `PlotPoints` and `MaxRecursion`

`DensityPlot`, `ContourPlot`, `RegionPlot` and the other field plots sample a
fixed 100×100 grid, whatever the options say. Wolfram starts from a
`PlotPoints` grid and refines it adaptively. Counting body evaluations of
`Sin[u v]` on `{u, 3, 4}, {v, 3, 4}`, wolframscript makes 973 by default, 210
with `PlotPoints -> 5`, and 27 with `PlotPoints -> 5, MaxRecursion -> 0`.
Woxi makes 10000 every time. The pictures look alike, but an expensive body
(a compiled iteration, say) runs about 10–50 times longer than in Wolfram.

### Plot parts nest differently

wolframscript wraps curves in `GraphicsComplex` / `Annotation` where Woxi emits
plain `Line`/`Arrow` lists; `SphericalPlot3D` samples a different grid;
`Graph[…, opts]` echoes its options unwrapped where wolframscript wraps them in
a list.

### Chart functions have no symbolic primitive list at all

`PieChart`, `BarChart` and friends render straight to SVG, so unlike `Plot`
they expose no `Graphics[{…}]` to index into: `Head[PieChart[{0.3, 0.7}][[1]]]`
is `Part` in Woxi (the `Part` stays unapplied on the opaque graphic) against
`List` in wolframscript, and a `/.` rule that matches nothing leaves the whole
expression alone rather than rewriting inside it. Making the charts symbolic
means routing them through the `Graphics` layer the plots use.

### `PointSize` in `PlotStyle` and `ListPlot`'s `PlotLabel`

`PointSize` inside `PlotStyle` is dropped — the three scatter renderers
hardcode the marker size. `ListPlot[…, PlotLabel -> "x"]` parses the option but
never draws it (`Show[…, PlotLabel -> …]` does work).

### An `AxesLabel` carrying a reciprocal power

A label's `Power[x, -1]` renders as `x⁻¹` in SVG markup where Wolfram stacks it
as the fraction `1/x`.

### Graphics3D lighting and projection

With the same scene both engines place every primitive identically, but:

- **Lighting.** Woxi applies a single diffuse term; Wolfram adds specular
  highlights and coloured light sources, so faces of one polygon come out at
  noticeably different brightness (its default face colour reads cream where
  Woxi's reads blue-grey). Woxi also clamps intensity to 1, so a face can never
  be brighter than its base colour, while Wolfram's ambient plus three
  directional lights routinely saturate a channel.
- **Projection.** `ViewAngle` drives a perspective camera in Wolfram; Woxi's
  projection stays effectively orthographic, so a lattice looks uniform instead
  of splaying toward the viewer. A wide angle (≳30°) leaves the silhouette a
  few percent narrow.

Compare geometry — vertex positions, which primitives appear, occlusion order —
not pixels.

### `Specularity` is accepted and ignored

Measured against wolframscript on a sphere under an explicit `Lighting` spec
matching Woxi's model, `Specularity[0.7]`, `Specularity[1]` and no
`Specularity` at all give **pixel-identical** output. The highlight only appears
under WL's *default* light set. So a specular term added to Woxi's single-light
model would be invented behaviour; it needs the default lighting first.

### `SphericalRegion -> True` shrinks about 17 % more than Wolfram

On a cylinder from `{0,0,0}` to `{10,0,0}` at `ImageSize -> {550, 350}` the
trimmed figure width is 324 px in Wolfram and 268 px in Woxi; **without** the
option both are 532. Working backwards, Wolfram behaves as if it fit the
enclosing sphere into 399 px of a 550×350 frame — neither dimension, nor their
mean, diagonal or geometric mean. What radius it normalizes by was not decoded.

### `Texture` and `VertexTextureCoordinates`

A textured convex `Polygon` in `Graphics3D` is drawn with its image (per-triangle
clip path and affine matrix), but the image is not lit: Wolfram shades it with
the scene lighting. Textured concave or holed polygons render as plain
default-blue faces. This cannot be checked against wolframscript by diffing SVG
(it rasterises 3D output).

### Some `Complex*` plot functions are unimplemented

`ComplexArrayPlot`, `ComplexContourPlot`, `ComplexVectorPlot` and
`ComplexStreamPlot` have no implementation; wolframscript renders all of them.
(`ComplexPlot`, `ComplexPlot3D`, `ComplexListPlot` and `ComplexRegionPlot`
are implemented.)

### `GeoGraphics` view and output form

The zoom level and single-point default span are Woxi heuristics and will not
match wolframscript's view. `GeoGraphics` text output is `-Graphics-` in Woxi
where wolframscript dumps the full `GeoGraphics[-Graphics-, GeoBackground -> …,
GeoCenter -> …]` options form, and `GeoRegionValuePlot` returns a bare
`Graphics` where WL wraps it in `Legended`.

### Geodesic values differ in the last 2–4 digits

`GeoDistance` and friends agree with WL to ~12 significant figures. Wolfram
displays the shortest round-trip of *its own* geodesic f64, which differs from
geographiclib (both the Rust crate and the canonical Python reference) by a few
to tens of ULP. **Not reproducible** with this library.

`Entity["City", …]` resolves to the region centroid, because the offline
country database has no city-level data — Munich resolves to Bavaria's centre
rather than WL's actual Munich coordinates. `GeoNearest` returns a human name
where WL returns its canonical entity id.

### `DiskSegment` and other region primitives are not drawn

`DiskSegment` has its measures but no graphics rendering.

### Buttons inside a picture are not clickable

`Inset[Button[label, action], pos]` inside a `Manipulate` body draws correctly
and the action is held, but pressing it does nothing: the renderer does not
report each inset button's pixel rectangle, and Woxi Studio has no hit-testing
against the rendered picture. (`Button[Graphics[…], action]` cells of a
`GraphicsGrid` are the exception: they carry hit regions, and Woxi Studio
overlays clickable areas on them.)

### `Manipulate` control gaps

- A `TabView` of controls is flattened — every tab's controls are shown at
  once, since Woxi's control panel is one flat list.
- One unrecognised control spec makes the **entire** `Manipulate` give up, so a
  single unsupported control takes every other control with it.
- `Manipulate[…, Initialization :> …]` leaks its definitions into the session.

### Iterated `NIntegrate` over a `Boole[…]` region can still be slow for
### some outer-parameter ranges

`NIntegrate[Boole[cond], {inner, a, b}]` integrates exactly (a boundary-
crossing search, not quadrature), and the iterated (2D+) form now detects
the common case where the *inner* region's measure touches exactly zero
partway across the outer range — a kink in the outer integrand's
derivative — and splits the outer integration there. That covers a region
disappearing (or appearing) as the outer variable sweeps past it, e.g. a
horizontal strip moving off the edge of a disk.

It does *not* detect a kink where the region's boundary switches which
underlying curve is active without the total number of boundary crossings
changing — e.g. cutting a disk with a horizontal plane whose height sweeps
past the disk's own top or bottom edge (as opposed to past the disk
entirely): the inner slice keeps exactly one crossing in, one out, on both
sides of that switch, so the crossing-count signal sees no kink there.
Outer sample points near that switch fall back to the shared evaluation
budget as a bounded-but-slow backstop (tens of seconds, not the multi-
minute hang before that budget was shared across the whole iterated call —
see `NINTEGRATE_BUDGET` in `src/functions/calculus_ast.rs`) rather than
hanging outright. Verified with:

```sh
woxi eval 'NIntegrate[Boole[p^2 + (q - 3)^2 < 4 && q < 4], {p, -2, 2}, {q, 1, 5}]'
# Correct (2.456739...+), but takes on the order of tens of seconds.
```

### Woxi's own SVG cannot be compared

`wolframscript`'s `ExportString[…, "SVG"]` is always cairo output — `pt` units,
text as glyph outlines, `rgb(%)` colours — for every graphics type. A test that
prints raw SVG can never match. Assert the meaning instead (`Area[Polygon[…]]`
rather than counting `<polygon>` tags).

Cairo also stamps per-run ids into the output, so even comparing wolframscript
with *itself* fails:

```sh
wolframscript -code 'ExportString[Graphics[Line[{{0, 0}, {1, 1}}]], "SVG"] ===
                     ExportString[Graphics[Line[{{0, 0}, {1, 1}}]], "SVG"]'
# False
```

Two pictures that are the same drawing therefore cannot be compared through
`ExportString` in a doc test either; pin those in
`tests/interpreter_tests/graphics.rs` instead.

### `Play` prints as `-Sound-`

```sh
wolframscript -code 'Play[Sin[2 Pi 440 t], {t, 0, 1}]'
# Sound[SampledSoundFunction[CompiledFunction[{11, 15., 5446}, …], 8000, 8000]]
woxi eval 'Play[Sin[2 Pi 440 t], {t, 0, 1}]'
# -Sound-
```

wolframscript compiles the amplitude function and prints the whole compiled
object, internal register layout and all. Woxi wraps the inert `Play` call in
a `Sound`, which reports the same `Head` and renders the same playable widget,
but prints as the short form. Test with `Head`, `AudioSampleRate` or
`AudioLength`, never against the printed form.

### `GraphPlot` accepts a `DirectedEdge`-keyed edge shape rule

```sh
wolframscript -code 'Head[GraphPlot[{1 -> 2, 2 -> 3},
                       EdgeShapeFunction -> {DirectedEdge[1, 2] -> (Line[#1] &)}]]'
# GraphPlot   (unevaluated, no message)
woxi eval 'Head[GraphPlot[{1 -> 2, 2 -> 3},
             EdgeShapeFunction -> {DirectedEdge[1, 2] -> (Line[#1] &)}]]'
# Graphics
```

wolframscript's `GraphPlot` builds an *undirected* graph out of a rule list, so
only an `UndirectedEdge`-keyed rule matches a part of it; a `DirectedEdge` key
matches nothing and the whole call is abandoned without a message. Woxi accepts
either key. Not reproduced: silently abandoning the plot looks like a defect,
and the same call with an `UndirectedEdge` key conforms on both sides.

### `StreamPlot` streamline spacing is an approximation (not reproducible)

Woxi places evenly spaced streamlines (Jobard–Lefer) about 1/22 of each axis'
range apart by default. Wolfram's spacing depends on the field: counting the
streamlines crossing a vertical line, a uniform flow
(`StreamPlot[{1, 0}, {x, -1, 1}, {y, -1, 1}]`) has 29, but the flow around a
cylinder (`{x, -3, 3}, {y, -2, 2}`) has 17 at its upstream edge. Its seeding
algorithm is not recoverable from the output, so Woxi's default sits between
the two, and `StreamPoints` settings keep Wolfram's proportions to it.

### `DensityPlot` ignores `AspectRatio`

`DensityPlot[x, {x, 0, 10}, {y, 0, 1}, AspectRatio -> 0.3]` is drawn square
in Woxi; Wolfram draws it 0.3 times as tall as wide.

### Merged graphics use a different tick font

A plot merged with other graphics by `Show` (say a `DensityPlot` under a
`StreamPlot`) is redrawn by the `Graphics` renderer, whose tick labels are a
monospace font, larger than the plot renderer's sans-serif ones. Wolfram draws
both alike.


## Images

### `HistogramTransform` equalizes on the bands, not on a spline

Woxi maps each pixel to the midpoint of the band its value occupies in the
cumulative distribution, rescaled from the 256 display levels to `k/255`. That
reproduces wolframscript exactly whenever the distinct values are few or evenly
spread — a four-value ramp, a two-value image, a constant channel, and the
256-level identity all agree to the last bit — but wolframscript actually
*interpolates* the cumulative distribution (its `Interpolation::inhr` message
leaks out for a constant channel), so for other value distributions the two
differ by up to about 3·10⁻³:

```sh
wolframscript -code 'ImageData[HistogramTransform[Image[{{0., 0.25, 0.5, 0.75, 1.}}]]]'
# {{0.1, 0.301961, 0.501961, 0.701961, 0.901961}}
woxi eval 'ImageData[HistogramTransform[Image[{{0., 0.25, 0.5, 0.75, 1.}}]]]'
# {{0.0984314, 0.299216, 0.5, 0.700784, 0.901569}}
```

Reproducing the rest needs wolframscript's exact spline through its 256-bin
histogram. Test rounded (`Round[100 …]`) or on the exact cases above.

### `ColorBalance` matches the Bradford model to ~3·10⁻⁴

Both engines run a von Kries adaptation in Bradford cone space over the same
D50 working matrix, dividing away the *chromaticity* (XYZ normalized to
`Y = 1`) of the reference. Woxi lands within about 3·10⁻⁴ of wolframscript,
which is enough for `Round[100 …]` to agree everywhere but not for a bare
`ImageData` comparison:

```sh
wolframscript -code 'ImageData[ColorBalance[Image[{{{0., 1., 0.}}}], Green]]'
# {{{0.863451, 0.863391, 0.863085}}}
woxi eval 'ImageData[ColorBalance[Image[{{{0., 1., 0.}}}], Green]]'
# {{{0.863393, 0.863393, 0.863393}}}
```

wolframscript's answer is not even neutral — the three channels differ in the
fourth decimal — so its matrix product is not the textbook one. Ruled out:
D65 primaries (an order of magnitude worse), the rounded Bradford inverse, and
CAT02/plain-von-Kries cone matrices.

A *two*-channel image (gray plus alpha) diverges outright: Woxi balances the
gray channel and drops the alpha, wolframscript mixes the two channels into
something that is not the balance of either. Same family as the four-channel
`ColorConvert` entry below — an unusual channel count with no colour space to
go by.

### `Pruning[image]` picks the other pixel of a final pair

The counted form `Pruning[image, n]` conforms exactly. The bare form, which
wears every thin arc down to a single pixel, agrees with wolframscript
everywhere except which of the last two pixels is kept:

```sh
wolframscript -code 'ImageData[Pruning[Image[{{0, 0, 1, 0, 0}, {0, 0, 1, 0, 0},
                                              {0, 0, 1, 0, 0}, {1, 1, 1, 1, 1}}]]]'
# the surviving pixel is in the bottom row
woxi eval '…'
# the surviving pixel is one row up
```

Woxi keeps the first pixel in raster order of a component that a pass would
empty. That matches wolframscript for horizontal, vertical and diagonal pairs
alike in every other case tried, so the rule wolframscript really uses is
something else again — it is not raster order, reverse raster order, or
distance from the component's centroid.

### `ColorConvert` of a four-channel image with no colour space

```sh
wolframscript -code 'ImageData[ColorConvert[Image[{{{1., 0., 0., 0.25}}}], "RGB"]]'
# {{{1., 0., 0.25}}}
woxi eval 'ImageData[ColorConvert[Image[{{{1., 0., 0., 0.25}}}], "RGB"]]'
# {{{1., 0., 0.}}}
```

With `ImageColorSpace -> Automatic` and four channels, wolframscript builds its
three output channels out of channels 1, 2 and **4** — it keeps the alpha and
drops the blue. Given `ColorSpace -> "RGB"` explicitly it agrees with Woxi
(alpha rides along untouched). Not reproduced: it looks like an off-by-one in
wolframscript's channel handling, and the explicit spelling conforms.

### ImageAdd and friends refuse mismatched dimensions

```sh
wolframscript -code 'ImageData[ImageAdd[Image[{{0.1, 0.2}}], Image[{{0.3}}]]]'
# {{0.4, 0.2}}
wolframscript -code 'ImageData[ImageAdd[Image[{{0.3}}], Image[{{0.1, 0.2}}]]]'
# {{0.5}}
woxi eval 'ImageData[ImageAdd[Image[{{0.1, 0.2}}], Image[{{0.3}}]]]'
# Error: Evaluation error: ImageAdd: images must have the same dimensions and channels
```

The result always takes the first image's dimensions, but the second image is
read differently depending on which is larger: a smaller second image is
applied at the top-left and the rest of the first is passed through, while a
larger one contributes its *last* element (`0.3 + 0.2 = 0.5`), not its first.
`ImageMultiply` and `ImageSubtract` behave the same way.

Refusing is at least honest, but it refuses with a hard error, which aborts the
enclosing evaluation.

### Four image heads accept a bad specification silently

```sh
wolframscript -code 'ImageCrop[Image[{{0.1, 0.2}}], x]'
# ImageCrop::arg2: x is not a positive integer, pair of integers, Full or Automatic.
woxi eval 'ImageCrop[Image[{{0.1, 0.2}}], x]'
# ImageCrop[-Image-, x]     (no message)

wolframscript -code 'ImagePad[Image[{{0.1, 0.2}}], x]'
# ImagePad::imgpadn: Expecting a number or a 2 by 2 matrix of numbers instead of x.
woxi eval 'ImagePad[Image[{{0.1, 0.2}}], x]'
# ImagePad[-Image-, x]      (no message)

wolframscript -code 'TotalVariationFilter[{1, 2, 3, 4, 100}, -1]'
# TotalVariationFilter::arg2: Expecting a non-negative real number, a vector of
# such numbers (for multi-channel images) or Automatic instead of -1.
woxi eval 'TotalVariationFilter[{1, 2, 3, 4, 100}, -1]'
# TotalVariationFilter[{1, 2, 3, 4, 100}, -1]   (no message)
```

These return the right expression and only the message is missing, so they are
conformance gaps rather than wrong answers. `Blur` and `ImageMultiply["x"]` are
missing `::bdrad` and `::bdarg` for the same reason. Note that
`TotalVariationFilter`'s second argument is a regularisation parameter, not a
neighbourhood range, so it does not share the other filters' validation.

More than the documented number of arguments gives `::argt` in Woxi where
wolframscript routes through its options machinery and gives `nonopt` —
systemic across `ImageTake`, `ImagePartition` and `DistanceTransform`.

### FindThreshold implements only the default method

`FindThreshold[image]` and `FindThreshold[data]` follow the default
"Cluster" method (Otsu): real levels in 256 bins over their own range,
integer levels on their raw values, the first maximizing bin as the threshold,
the midpoint when only two levels occur, and `1.` for a constant image. The
one-argument `MorphologicalBinarize` uses the same split binned on `[0, 1]`.
`Method` options are not supported.

### EdgeDetect keeps a whole flat gradient plateau

```sh
wolframscript -code 'ImageData[EdgeDetect[Image[{{0, 0, 1, 1, 0, 0, 1, 1, 0, 0}}]]]'
# {{0, 1, 0, 1, 0, 1, 0, 0, 1, 0}}
woxi eval 'ImageData[EdgeDetect[Image[{{0, 0, 1, 1, 0, 0, 1, 1, 0, 0}}]]]'
# {{0, 1, 0, 1, 1, 1, 1, 0, 1, 0}}
```

A period-4 square wave is resonant with the default radius-2 kernel and
produces a gradient magnitude that is *exactly* constant across six pixels.
wolframscript keeps alternating ones there; Woxi's non-maximum suppression
keeps the whole plateau.

No rule over the magnitudes of the two neighbours can tell index 3 from index 4
in that run — both have equal magnitudes on both sides — so the tie-break uses
information that could not be identified from the outputs. Every
non-degenerate case matches, over 15 reference cases spanning 1D and 2D shapes,
colour and Byte images, and explicit thresholds. **Not reproducible.**

### LaplacianGaussianFilter and DerivativeFilter are unimplemented

```sh
wolframscript -code 'First[ImageData[LaplacianGaussianFilter[Image[{{0., 0., 0., 1., 0., 0., 0.}}], 1]]]'
# {0., 0., 1.8418419230004717, -3.683683846000945, 1.8418419230004717, 0., 0.}
wolframscript -code 'First[ImageData[DerivativeFilter[Image[{{0., 0., 0., 1., 0., 0., 0.}}], {0, 1}]]]'
# {0.05771365940052183, -0.21539030917347252, 0.803847577293368, 0., …}
```

Both are feature gaps rather than divergences. Neither can reuse
`GradientFilter` directly: `DerivativeFilter` normalises its kernel
differently (the impulse responses do not agree up to a scale factor), and
`LaplacianGaussianFilter`'s three-tap response at radius 1 is a scaled second
difference rather than the second-derivative-of-Gaussian kernel the name
suggests.

### Colour-space tags do not propagate

`ImageTake` and `ImagePartition` propagate an image's colour-space tag;
`Blur`, `ImageAdjust`, blends, morphological operations and `ImageResize` all
construct an untagged image. Wolfram probably propagates — each needs its own
probe.

### `LABColor` and its family

`LABColor` is deliberately out of scope; `XYZColor`, `LUVColor` and `LCHColor`
belong to the same family, whose `ColorConvert` round-trips go through D50/D65
white points and WL-specific matrices that would all need to match bit for bit.

### `ColorData` named gradients

`ColorData["Rainbow", 0.5]` and `ColorData["Rainbow"]` (a `ColorDataFunction`)
are unimplemented. `ColorData[name, "ColorList"]` is
`Missing["NotApplicable"]` in WL, so the control points are not exposed — they
would have to be recovered by sampling and the interpolation scheme guessed.
The indexed schemes (1, 2, 3) are implemented but scheme 1 agrees to ~15 digits
rather than bit for bit.

### `ColorData["Named"]` collections (Studio: Select, View and Compare Named Colors demo)

`ColorData["Named"]`, `ColorData[scheme, "Range"]` and
`ColorData[scheme, "ColorList"]` stay unevaluated for the named collections,
so the Demonstration `Select, View and Compare Named Colors` initializes with
`Part::partw` errors and its color-dot panels are empty. Only `"HTML"` and
`"Legacy"` have lookup tables (by name). Making the notebook work needs the
exact `"Atoms"`, `"Crayola"`, `"GeologicAges"`, `"WebSafe"` and `"System"`
tables (names, RGB values and per-channel exactness) plus the shape of the
`"Range"` result, none of which can be reconstructed without `wolframscript`.

### `ColorQuantize` is a stub

Echoes unevaluated; a median-cut palette choice would likely be a
selection-artifact problem.

### `ImageMeasurements` intensity on colour images

WL takes the luma and carries its Real32 rounding (`MinIntensity` of pure red
is `0.29899999499320984`); those measures are left unevaluated in Woxi.

### `GaussianFilter` on an image

Differs in the last digits because wolframscript rounds its **intermediates**
to f32 while Woxi rounds only at the boundaries. `wolframscript` is also
inconsistent with itself here, giving `0.80123907` and `0.80123901` for the
same kernel tap across evaluation orders.

### `ImageEffect[img, {"MotionBlur", r, θ}]` kernel (unverified)

Woxi averages `2r + 1` bilinear samples on a symmetric line at `θ` degrees.
Not checked against wolframscript (engine unactivated when it was
written); WL's kernel shape, its direction convention and its edge handling
may differ.

### `AudioTrim` on tiny objects

wolframscript gives `{{1., 0.}}` for `{0., 1., 0.}` but `{{1.}}` for
`{1., 0., 1.}`, dropping a non-silent sample — minimum-window artifacts at
44.1 kHz, not a rule worth matching. `FourierDCT`/`FourierDST` differ in the
last ULP.

### `ImageDeconvolve` shifts by a pixel and iterates its own way

Woxi deconvolves with a regularized inverse filter in the frequency domain
(`"DampedLS"`, `"Tikhonov"` and `"Wiener"` share it), so a 1×1 identity
kernel gives the input back, scaled by the regularization. wolframscript's
result is shifted one pixel up and to the left, the last row and column
repeated — with the identity kernel `{{1}}` and with the centered
`{{0, 1, 0}}` alike:

```sh
wolframscript -code 'ImageData[ImageDeconvolve[Image[{{0.1, 0.2, 0.3, 0.4, 0.5}}], {{1}}, Method -> "RichardsonLucy"]]'
# {{0.2, 0.3, 0.4, 0.5, 0.5}}
wolframscript -code 'ImageData[ImageDeconvolve[Image[{{0.1, 0.2, 0.3, 0.4, 0.5}}], {{1}}, Method -> {"Tikhonov", 0.01}]]'
# {{0.19998, 0.29997, 0.39996, 0.49995, 0.49995}}   (shifted, times 1/(1 + 0.01^2))
woxi eval 'ImageData[ImageDeconvolve[Image[{{0.1, 0.2, 0.3, 0.4, 0.5}}], {{1}}, Method -> {"Tikhonov", 0.01}]]'
# {{0.0990099, 0.19802, 0.29703, 0.39604, 0.49505}}
```

The default `"DampedLS"` is an iterative method whose output
(`{{0.207692, 0.3, 0.392308, 0.430769, 0.369231}}` for the row above) cannot
be derived from these samples, and `"RichardsonLucy"`, `"TSVD"`, `"Hybrid"`,
`"SteepestDescent"` and `"TotalVariation"` are unimplemented (the call stays
unevaluated). **Not reproducible** without WL's implementation.

### `Export` of a multi-segment `Sound`

Single `Play` and `Audio` exports are byte-identical to wolframscript. For
`Sound[{Play, Play}]` segment 1 matches byte for byte, but WL applies an
unexplained affine requantization to segment 2 (residual ≈ 0.54·s − 0.54 LSB,
uncorrelated with the time shift; not float32 rounding, not floor-versus-round,
not global normalization). Maximum error ≤ 1 LSB of int16.
**Not reproducible.**


## Graphs

Wolfram's graph layer is igraph-backed, and two classes of its output are not
recoverable from outside.

### `GraphData[]` carries a slice of the atlas, not all 12474 entries

Woxi bundles the named entities it can answer every property for, plus the
parametrized families (`{"Complete", n}`, `{"Cycle", n}`, `{"Path", n}`,
`{"Star", n}`, `{"Wheel", n}`, `{"CompleteBipartite", {m, k}}`). Names,
vertex labelling and edge order follow Wolfram's, and an unknown entity gets
the same `GraphData::notent` message and unevaluated result — but
`Length[GraphData[]]` is 14 against WL's 12474, and any entity outside the
bundled set is a `notent` where WL answers. Only the ~760-entry property list
is likewise abridged: `GraphData["Properties"]` returns the nine properties
Woxi answers.

### Selection among equal optima

`FindVertexCover`, `FindEdgeCover`, `FindVertexCut` and `FindEdgeCut` each pick
one minimum solution via igraph internals. Three candidate models
(lexicographically smallest, complement of the lexicographically first or last
maximum independent set, greedy minimum-degree) are each contradicted by
probes: `CompleteGraph[4]` → `{1,2,3}`, `CycleGraph[6]` → `{2,4,6}` (not
`{1,3,5}`), `GridGraph[{2,3}]` → `{2,3,6}`. `FindVertexCut[PathGraph[{1..5}]]`
is `{4}` where `{2}` and `{3}` are equally small. **Not reproducible.**

`KnapsackSolve` is the same shape: value-optimal solutions are easy, but WL's
choice among ties follows its `LinearProgramming` simplex/branch-and-bound.
Reverse-index DFS fits some probes and max-count others, and the two
contradict.

### `FindHamiltonianCycle[g, k]` enumeration order

```sh
wolframscript -code 'FindHamiltonianCycle[Graph[{1<->2,2<->3,3<->4,4<->1,1<->3,2<->4}], All]'
# {{1<->3, 3<->2, 2<->4, 4<->1}, {1<->2, 2<->3, 3<->4, 4<->1}, {1<->2, 2<->4, 4<->3, 3<->1}}
woxi eval 'FindHamiltonianCycle[Graph[{1<->2,2<->3,3<->4,4<->1,1<->3,2<->4}], All]'
# {{1<->2, 2<->3, 3<->4, 4<->1}, {1<->2, 2<->4, 4<->3, 3<->1}, {1<->3, 3<->2, 2<->4, 4<->1}}
```

The set of cycles, their orientation and their count match (12 for
`CompleteGraph[5]`, 60 for `CompleteGraph[6]`), and `k = 1` is the
single-argument cycle on both sides; Woxi lists them in lexicographic DFS
order. WL's order is not lexicographic in any reading tried: on
`CompleteGraph[5]` it groups by last vertex descending (5, 4, 3), then by
second vertex descending, but the third vertex ascends in some groups and
descends in others. With `k` below the total, Woxi's first `k` are therefore
a different subset.

### Vertex order within a component

`ConnectedComponents` on an **undirected** graph: `Graph[{1<->2, 3<->4}]` gives
`{{1,2},{3,4}}` in Woxi and `{{2,1},{4,3}}` in WL. `WeaklyConnectedComponents`
already ships a divergent order for the same reason —
`{1→2}` lists as `{2, 1}` but the structurally identical `{7→8}` lists as
`{7, 8}`, and `{a→b}` lists as `{a, b}`. Every hypothesis tried (original
order, DFS pre/post, BFS from the target, degree-based start) is contradicted.
`WeaklyConnectedGraphComponents` is unimplemented for this reason.
**Not reproducible.**

### `TopologicalSort` vertex order

WL's vertex ordering is not reproducible by DFS reverse-postorder or by Kahn
with a min/max heap: `{1->3, 2->3}` → `{1,2,3}` needs Kahn-min, but
`{1->2, 1->3, 2->4, 3->4}` → `{1,3,2,4}` needs DFS, and the two contradict.
Also `{1->2, 3->4}` → `{3,4,1,2}` and `Graph[{1,2,3}, {1->3}]` → `{1,3,2}`.
A sweep of 115 random DAGs against 42 candidate algorithms (Kahn with a
min/max heap, FIFO and LIFO keyed on either `VertexList` index or vertex
value; DFS reverse-postorder over four start orders × four successor orders;
each of those on the transposed graph) tops out at 67/115 — no candidate fits.

The order in fact does not depend on the graph alone. `Graph[{4,1,2,7,3,6,5},
{6->7}]` and `Graph[{1,2,3,4,5,6,7}, {6->7}]` both give `{3,6,7,1,2,5,4}`, so
`VertexList` order is mostly ignored; but `Graph[{7,6,5,4,3,2,1}, {6->7}]`
gives `{3,6,7,1,5,2,4}`, and relabelling the same graph to the matching
strings gives `{c,d,e,f,g,b,a}`. That is WL's internal vertex hashing showing
through, and it is deterministic per run but not derivable. Woxi always
returns the lexicographically smallest topological order (Kahn's algorithm,
ties broken by `VertexList` index) — a valid ordering, just not always WL's.
**Not reproducible.**

### `FindHamiltonianPath` choice

`Graph[{1 <-> 2, 1 <-> 3}]` answers `{3, 1, 2}` while the structurally
identical `Graph[{2 <-> 1, 1 <-> 3}]` gives `{2, 1, 3}`. Woxi does a greedy DFS
from the first vertex. **Not reproducible.**

### Provenance-dependent edge order

Generator graphs (`CycleGraph`, `CompleteGraph`, `StarGraph`, `PetersenGraph`)
and explicitly constructed `Graph[{…},{…}]` objects produce *different* edge
orderings from the same operations (`Subgraph`, `LineGraph`,
`NeighborhoodGraph`). Woxi implements the generator convention. When a
neighborhood or subgraph covers the whole graph, WL emits igraph's BFS
traversal order.

### `GroupCentralizer` generator order

The returned generators are almost the structural centralizer generators, but
the ordering contradicts itself across cases — `S4/(12)(34)` gives rotations
ascending while `S6/(12)(34)(56)` reverses the first pair, and fixed points use
descending adjacent transpositions unlike the 2-cycle classes' star swaps.
It is Schreier–Sims discovery order. **Not reproducible.**

### `VertexCoordinates` annotations

An item-level `VertexCoordinates` read returns the laid-out coordinate, so
Woxi answers from its own `GraphEmbedding` — the values diverge because the
layout differs.

### `"PlanarEmbedding"` coordinates

`GraphLayout -> "PlanarEmbedding"` and `GraphEmbedding[g, "PlanarEmbedding"]`
give a crossing-free straight-line drawing in both, but not the same one. WL
places the vertices on an integer grid (`ImportString["Fxo[g", "Graph6"]`
spans `{0..6} × {0..6}`, a triangle `{0,0}, {6,0}, {0,6}` outside), the shape
of a canonical-ordering grid drawing whose exact variant is not known. Woxi
puts the largest face of a Demoucron–Malgrange–Pertuiset embedding on a
regular polygon and every other vertex at the barycenter of its neighbors
(Tutte), so the points are reals in `[-1, 1]`. A graph that is not
biconnected falls back to the default layout.

### Multi-entry rule lists inside a graph option

WL scrambles the order of a multi-entry rule list inside an option
(`VertexLabels -> {2 -> "b", 1 -> "a"}` for the input `{1 -> "a", 2 -> "b"}`) —
internal hash order. Woxi keeps input order.


## Knot data

### `KnotData` knows the knot table, not the curated geometry

Woxi names every knot of the Rolfsen table (`KnotData[All]`, up to ten
crossings) and answers what follows from the name alone — `CrossingNumber`,
`AlexanderBriggsList`/`AlexanderBriggsNotation`, `StandardName`, `Name`. A
`SpaceCurve` (and the `ImageData` tube swept around it) exists only for the
trefoil and the torus knots, from their textbook parametrizations; the
trefoil's is the one wolframscript uses too. wolframscript's curves for the
other knots are `InterpolatingFunction`s built from its curated data, which
Woxi does not bundle — so a table torus knot such as `SolomonSeal` (5_1) gets
the torus-knot formula instead of wolframscript's interpolated curve:

```sh
wolframscript -code 'KnotData["SolomonSeal", "SpaceCurve"][1.]'  # {2.16047, -1.44119, -0.497685}
woxi eval 'KnotData["SolomonSeal", "SpaceCurve"][1.]'            # {-0.950339, 2.07653, -0.958924}
```

`KnotData["Properties"]` lists only the properties above (wolframscript has
68), and `KnotData[name]` draws the curve rather than wolframscript's
pre-built tube mesh.

## Chemistry data

### `IsotopeData` carries NIST's isotope table, not the full ~3000-nuclide chart

Like `GraphData`'s atlas slice above, Woxi bundles the isotopes from NIST's
"Atomic Weights and Isotopic Compositions for All Elements" (the same table
`ElementData["…", "AtomicWeight"]` is built from): one or two naturally
occurring isotopes for every element that has any, plus the longest-lived
known isotope for every element that has none — 354 isotopes total, against
Wolfram's own curated chart of roughly 3000. An isotope outside that set is
unrecognized where WL answers it.

`ElementData[…, "StableIsotopes"]` draws from the same table and applies the
same coarser cut: an isotope counts as "stable" when NIST reports a measured
natural abundance for it, not by true nuclear-decay stability. Long-lived
primordial radionuclides that still occur naturally (potassium-40,
rubidium-87, thorium-232, the natural uranium isotopes, …) are included,
matching how these elements are conventionally described in chemistry
references, but diverging from a strict "does it decay" reading.

`IsotopeData[…, "BindingEnergy"]` is not looked up — it is computed from
NIST's atomic mass for the isotope via the standard mass-excess formula
`BE = (Z·m(¹H) + N·m(n) − M(A,Z))·c²`, using CODATA 2018 constants for the
neutron mass and the u→MeV conversion factor, then divided by the mass
number (wolframscript reports the binding energy *per nucleon*). This
reproduces textbook values (carbon-12's 92.16 MeV total, 7.68 MeV per
nucleon) but agrees with wolframscript's own tabulated value only to about
seven significant figures — `7.6801447` against WL's `7.6801446` for
carbon-12.

`IsotopeData[…, "AtomicMass"]` and `…, "IsotopeAbundance"` carry the same
units as wolframscript (`"AtomicMassUnit"`, `"Percent"`) but as machine
reals. WL tags its own values with the precision its measurement uncertainty
justifies, which is not reconstructible from NIST's published digits:

    Quantity[12.`9., "AtomicMassUnit"]
    Quantity[98.9399999999999946176`3.2172206556445184, "Percent"]

The abundances themselves are NIST's, which are not always Wolfram's: NIST
puts carbon-12 at 98.93 %, WL at 98.94 % (the difference between the
tabulated representative composition and the best-measurement range it sits
in).

`IsotopeData["Properties"]` lists the `EntityProperty["Isotope", …]` objects
Woxi answers — seven, against the roughly sixty of Wolfram's curated chart
(decay modes, half-lives, magnetic moments, medical applications, …). Same
story as `ElementData["Properties"]`.

## Geometry and regions

### `FindCurvePath` finds one curve where Wolfram may find several

Woxi orders the points along a single shortest tour, closed or cut at its
longest step. Wolfram splits scattered points into several curves:
`FindCurvePath[RandomSample[Table[{t, t^2}, {t, -1, 1, 0.25}], 9]]` gives
two curves there. A closed polygon and a sampled open curve match.

### Three region measures are missing for heads whose siblings have them

```sh
wolframscript -code 'ToString[{RegionDimension[Tube[{{0,0,0},{0,0,2}}]],
                     Area[Torus[{0,0,0},{1,3}]], SurfaceArea[Torus[{0,0,0},{1,3}]],
                     Volume[Hexahedron[]]}, InputForm]'
# {2, 8*Pi^2, Undefined, 1}
woxi eval '{RegionDimension[Tube[{{0,0,0},{0,0,2}}]], …}'
# each one echoed back unevaluated
```

`Tube` is absent from the intrinsic-dimension table (`Cylinder` and `Cone`
are there). `Torus` has `RegionMeasure` — `RegionMeasure[Torus[{0,0,0},{1,3}]]`
is `8*Pi^2`, correct — but `Area`/`SurfaceArea` do not route a 2-dimensional
region to it the way they do for `Sphere`. `Hexahedron`'s volume is only
computed from the centre-and-edge Platonic form, not from the eight-vertex
form its default expands to, which needs a general decomposition rather than
a table entry. Independent of how the region was written: the zero-argument
default forms diverge for the same reason the explicit ones do.

### `TriangleCenter` accepts a triangle embedded in 3D

`TriangleCenter[Triangle[{{0,0,0},{4,0,0},{0,3,0}}], "Circumcenter"]` is
`{2, 3/2, 0}` in Woxi; wolframscript handles only the planar case and leaves
the call unevaluated. Deliberate — the 3D centre is well defined.

### `ConvexHullMesh`'s 3D facet order and rotation is qhull's

3D point sets build a real hull (a standard incremental "beneath-beyond"
algorithm, not qhull) whose coplanar triangles are merged into the polygons
they tile, as qhull reports them — a cube's sides are six quads. Its facets
are the *same polygons with the same outward orientation* as wolframscript's;
what is not replicated is qhull's own bookkeeping:

```sh
wolframscript -code 'ToString[ConvexHullMesh[{{0,0,0},{1,0,0},{0,1,0},{0,0,1}}], InputForm]'
# BoundaryMeshRegion[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
#   {Polygon[{{3, 2, 1}, {2, 4, 1}, {4, 3, 1}, {3, 4, 2}}]},
#   Method -> {"SeparateBoundaries" -> False}, WorkingPrecision -> Infinity]
woxi eval 'ToString[ConvexHullMesh[{{0,0,0},{1,0,0},{0,1,0},{0,0,1}}], InputForm]'
# BoundaryMeshRegion[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
#   {Polygon[{{1, 3, 2}, {1, 2, 4}, {1, 4, 3}, {2, 3, 4}}]},
#   Method -> {"SeparateBoundaries" -> False}, WorkingPrecision -> Infinity]
```

Each face above is a cyclic rotation of the matching wolframscript face
(`{1,3,2}` rotates to `{3,2,1}`, `{1,2,4}` to `{2,4,1}`, and so on) — the same
triangle, the same winding, just listed starting at a different vertex. The
facet order and each facet's starting vertex follow no sort that explains the
samples below. Compare hulls with each face rotated to start at its lowest
vertex (which keeps the winding) and the list sorted:
`Sort[RotateLeft[#, First[Ordering[#, 1]] - 1] & /@ MeshCells[m, 2][[All, 1]]]`.

Reference outputs (all carry `Method -> {"SeparateBoundaries" -> False},
WorkingPrecision -> Infinity`):

| points | faces |
| --- | --- |
| `{{0,0,0},{1,0,0},{0,1,0},{0,0,1}}` | `{{3,2,1},{2,4,1},{4,3,1},{3,4,2}}` |
| the same plus `{1,1,1}` | `{{3,2,1},{2,4,1},{4,3,1},{3,5,2},{5,4,2},{4,5,3}}` |
| the eight unit-cube corners | `{{3,2,1,4},{1,2,6,5},{4,1,5,8},{2,3,7,6},{3,4,8,7},{5,6,7,8}}` |

### Mesh regions: constructor normalization, messages and `MeshCellMarker`

`ConvexHullMesh` normalizes its `MeshCellStyle`/`MeshCellHighlight`/
`MeshCellShapeFunction`/`MeshCellLabel` options into per-cell `Properties`
the way wolframscript does, but three related normalizations are open:

- The `MeshRegion[…]`/`BoundaryMeshRegion[…]` constructors themselves are
  echoed as given. wolframscript converts the coordinates to machine reals,
  splits a `Line[{1, 2, 3, 4, 1}]` boundary into its segments, wraps the cells
  in a list and normalizes the options into `Properties`:

  ```sh
  wolframscript -code 'ToString[BoundaryMeshRegion[{{0,0},{2,0},{2,2},{0,2}}, Line[{1,2,3,4,1}]], InputForm]'
  # BoundaryMeshRegion[{{0., 0.}, {2., 0.}, {2., 2.}, {0., 2.}},
  #   {Line[{{1, 2}, {2, 3}, {3, 4}, {4, 1}}]}, Method -> {"SeparateBoundaries" -> False}]
  woxi eval 'ToString[BoundaryMeshRegion[{{0,0},{2,0},{2,2},{0,2}}, Line[{1,2,3,4,1}]], InputForm]'
  # BoundaryMeshRegion[{{0, 0}, {2, 0}, {2, 2}, {0, 2}}, Line[{1, 2, 3, 4, 1}]]
  ```

- `MeshCellMarker -> {{0, 1} -> 3}` normalizes to
  `{0, {1}} -> MeshCellMarker -> {3}, {0, Default} -> MeshCellMarker -> 0` and
  sits between `MeshCellStyle` and `MeshCellHighlight`; Woxi keeps it as an
  ordinary trailing option.
- A mesh region in a message is summarized as `BoundaryMeshRegion[<2>, <2>]`
  (`Part::partd` on one, say); Woxi prints the whole object.

### Mesh cell order is qhull's

Which vertex a triangle starts at, and the order of the cells, follow qhull —
`{{1,2,3},{3,2,4}}` against Woxi's `{{1,2,3},{2,4,3}}`, the same two triangles
with the same orientation. Compare with `Sort[Sort /@ MeshCells[m, 2][[All,1]]]`.

Wolfram is also inconsistent with itself here:
`Perimeter[DelaunayMesh[…]]` of an exact square is `8.` while Woxi gives the
exact 8. `RegionMeasure[Polygon, spec]` in WL is a machine-precision quadrature
value (`3.366666189468263` for the exact `101/30`) while its
`MomentOfInertia[Polygon]` is exact — so Woxi leaves polygon `RegionMoment`
unevaluated on purpose.

### Symbolic region membership

`RegionMember` with a symbolic point is unevaluated for `HalfSpace`,
`StadiumShape` and others. WL emits a `Reduce`-normalized condition
`Element[x | y, Reals] && <inequality>` with non-trivial canonicalization:
rational bounds cleared to integers, division by the coefficient gcd,
negative-coefficient terms moved across, and an all-negative normal flipped to
a `>= 0` shape. Reproducing it is a mini-`Reduce`.

### Region scope gaps

- `RegionDisjoint` leaves ellipse `Disk[c, {a, b}]` (a quartic distance
  problem), `ImplicitRegion`, `Annulus` and `Line` unevaluated.
- `RegionMoment` for an ellipse `Disk` is open, as is `MomentOfInertia` beyond
  what is implemented.
- Simplices in a **higher embedding dimension** than their intrinsic dimension
  (a triangle given as three points in 3-space) need the Gram determinant and
  are unevaluated. The no-argument regular `Tetrahedron[]`/`Simplex[n]`
  defaults are only partly handled.
- `RegionBounds` and `RegionMember` for `Annulus` are unimplemented.
- `SphericalShell` and `CapsuleShape` `SurfaceArea` with **symbolic** radii is
  left unevaluated, because wolframscript itself hangs on it.
- `RegionMeasure[ImplicitRegion[…]]` is hard in general.

### `PolyhedronData` form divergences

Two entries in the hand-written `Dodecahedron` record print differently:
`Sqrt[5/8 + (11*Sqrt[5])/40]` against WL's `Sqrt[5/8 + 11/(8*Sqrt[5])]`, and a
`Sqrt[…]` where WL gives `Root[1 - 20*#1^2 + 80*#1^4 &, 3, 0]`. Re-evaluating
the first flips it to Wolfram's, so `===` against the stored value can be False
for the *same* value depending on how many evaluation passes each side saw.

### `Molecule` residues

`[13C]`-style isotope atoms get a WL `UnpairedElectronCount` property Woxi
omits, and the no-carbon Hill ordering diverges — WL writes ammonium as `NH4`
(N before H) where Woxi keeps strict alphabetical order.

### `SpherePoints` and `ShiftRegisterSequence`

`SpherePoints[n]` returns specific polyhedral configurations for small n, but
the machine values leak WL's construction pipeline (`SpherePoints[4]` starts
`0.9999999999999999`), and large n switch to an undocumented regularized
spiral. `ShiftRegisterSequence`'s explicit-taps form contradicts its own
default-form model: `{4, {3,4}}`, `{5, {2,5}}` and `{5, {3,5}}` all return
`{0}` even though those polynomials are primitive, yet the n=4 *default*
satisfies exactly the `{3,4}` recurrence. **Not reproducible.**

### `IsolatingInterval`

No decodable convention: `Sqrt[2]` → `{45/32, 91/64}` (width 2^-6), `Sqrt[3]` →
width 2^-5, but `2^(1/3)` and `Root[#^5-#-1&,1]` → `{0, 2}` (unrefined), and
`Sqrt[2]+Sqrt[3]` → `{1, 4}` (width 3, not from bisection at all). The
refinement depth comes from WL's internal interval-arithmetic convergence loop.
**Not reproducible.**

### `JuliaSetIterationCount`

Escaping orbits fit `R = 2 + |c|²` over four probes, and an undecided orbit
gives cap+1. But **bounded** orbits return small counts via an attractor/cycle
detection with internal tolerances — `[0, 1/2]` → 3, `[0, 0.9]` → 5,
`[0, 0.31]` → 2, yet the parabolic `[0.25, 0.1]` → 1001. A fixed inner
threshold needs ε ∈ (0.0343, 0.0625]; successive-difference and
Brent-checkpoint models both fail. `MandelbrotSetIterationCount` was decodable
and is implemented. **Not reproducible.**

### `FindClusters` without an explicit cluster count

`FindClusters[{1,2,10,11,20,21}]` gives `{{1,2,10,11},{20,21}}` in Woxi and
`{{1,2},{20,21},{10,11}}` in WL — a different k *and* a non-sorted emission
order, both from an internal silhouette/gap heuristic. The explicit-k and
`DistanceFunction` forms are correct. **Not reproducible.**

### `FindPeaks` with a non-zero smoothing scale

`FindPeaks[list, σ]` is *not* `FindPeaks[GaussianFilter[list, σ]]`: at σ=2.2 the
blurred list still has two peaks but `FindPeaks` reports one, and no blur radius
reconciles the counts across σ. It is a scale-space tracking pass. A bare
sharpness inherits σ as its scale and is not the plain second difference of the
blurred data either. Woxi leaves any non-zero scale unevaluated.

### Astronomy residues

`SunPosition`, `MoonPosition` and `SiderealTime` now agree closely, but:

- The Sun's near-meridian azimuth differs by ~20″, because Meeus's short solar
  series is only good to 0.01°.
- `Sunrise`/`Sunset` stay 13–30 s away: the half-day length implies
  wolframscript uses an effective horizon altitude near −0.885°, not the
  standard −50′ that Meeus documents. The offset is systematic across latitudes
  and seasons but has no physical derivation. **Do not tune the constant to
  fit it.**
- `SiderealTime` is 0.28 s off (a different nutation/precession model).


## Import, export, units and system

### A plain read after a pattern read still sees the rest of the stream

`ReadString[stream, patt]` returns the text before the first match of `patt`
and consumes the match. wolframscript implements that by pulling a
10 000-character chunk into a buffer: later *pattern* reads consume from the
buffer and continue correctly, but a later plain `ReadString[stream]` reads
from the underlying file position instead, which the chunk already moved
past. On anything under 10 000 characters that means the rest of the stream
is simply lost:

```sh
wolframscript -code 's = StringToStream["aaa%HEAD%rest"]; {ReadString[s, ___ ~~ "%HEAD%"], ReadString[s]}'
# {, EndOfFile}
woxi eval 's = StringToStream["aaa%HEAD%rest"]; {ReadString[s, ___ ~~ "%HEAD%"], ReadString[s]}'
# {, rest}
```

`StreamPosition` shows the same thing from the other side: after a pattern
read wolframscript reports the chunk's file offset (10000, or the file size
for a shorter file) where Woxi reports the logical position just past the
match. Every value a pattern read *returns* agrees, including a whole
sequence of them — WLJS Notebook's `.wln` reader is four pattern reads on one
stream and gives identical text on both. Only mixing a pattern read with a
plain read on the same stream diverges, which is why Woxi keeps the
self-consistent behaviour rather than emulating the buffer.

### `ExampleData` content is generated by design

Woxi serves Wolfram's whole `ExampleData` catalogue — all 19 collections and
their entries, the same property names, the same head and shape of every value
and the same metadata (image sizes, channels and types, sample rates and
lengths, vertex and face counts, matrix dimensions and nonzero counts, table
dimensions, column names and types, graph sizes and direction, …) — but none of
Wolfram's data. Every value is generated, deterministically, from the entry
name: photographs are crude drawings with the original's histogram statistics,
3D scans are rough models with exactly the original's vertex and face counts,
recordings are synthesized, books are generated text with the original's word,
line and paragraph counts, tables are drawn column by column from the original's
ranges, spreads, category frequencies and correlations. Descriptions and
sources are Woxi's own text. The structure is recorded from wolframscript by
`scripts/example_data_meta.wls` into `resources/example_data_meta.tsv.gz`.

```sh
wolframscript -code 'ImageDimensions[ExampleData[{"TestImage", "Mandrill"}]]'  # {512, 512}
woxi eval 'ImageDimensions[ExampleData[{"TestImage", "Mandrill"}]]'            # {512, 512}
wolframscript -code 'ExampleData[{"TestImage", "Mandrill"}]'                   # the photograph
woxi eval 'ExampleData[{"TestImage", "Mandrill"}]'                             # a drawn mandrill
```

So: test metadata (`ImageDimensions`, `Length`, `VertexCount`, `Dimensions`,
`Head`, …), never content. The exceptions are five classic networks that are
real, assembled from their publications (`ZacharyKarateClub`,
`DolphinSocialNetwork`, `LesMiserables`, `USPoliticsBooks`, `WordAdjacencies`);
their vertex names follow the publication (`"MlleBaptistine"`) rather than
Wolfram's spelling (`"Mlle Baptistine"`).

Known structural gaps:

- `"Matrix"` entries with more than 4 000 000 stored entries (92 of the
  2338, up to 99 million) stay unevaluated: Woxi holds sparse arrays as
  ordinary expressions. Their metadata properties are served.
- `"TestImage3D"` volumes are `Image3D[{slice, …}]` stacks of packed slices
  rather than `Image3D[NumericArray[…], "Byte", …]`, and `"TimeSeries"` /
  `"EventSeries"` values are Woxi's normalized `TimeSeries[…]` rather than
  `TemporalData[…]` (see the `TimeSeries` section).
- `Options` of an example image omit Wolfram's `MetaInformation` (file
  comments, creation dates).
- Synthetic test patterns (`"Gray21"`, `"ResolutionChart"`, `"Ruler"`,
  `"TestPattern"`) are drawn, not reproduced pixel for pixel.
- `"FilePath"` names a file Woxi exports on first request under the
  temporary directory, not a paclet file; `"AudioFile"` reads that file.
- `ExampleData["SpatialPointData"]` is `$Failed` in wolframscript (the
  collection needs a server), and is in Woxi too.
- The retired `"TestImage"` entries `"Lena"` and `"TestPattern"` (no longer
  listed, still served with `ExampleData::obsolete`) were added to the table
  by hand from the USC-SIPI originals they came from (4.2.04, 512 × 512 RGB;
  testpat.1k, 1024 × 1024 gray; both sizes and colour types confirmed
  against wolframscript). Their histogram statistics are estimates, not
  recorded from wolframscript.

### `ShortTimeFourier` partitions differently

Woxi's default partition offset is half the window; wolframscript's is 1, so
the same signal yields 3 frames against its 8. The property names differ too:
Woxi answers `"WindowSize"`, `"Offset"` and `"NumberOfFrames"` where
wolframscript's `ShortTimeFourierData` has `"PartitionSize"`,
`"PartitionOffset"`, `"SmoothingWindow"`, `"PartitionPadding"`,
`"FourierParameters"`, `"SampleRate"` and `"DataType"`.

### `UnitSimplify` is unimplemented

Two parts. The clean one: a compound dimension matching a named SI derived unit
becomes that unit (`N*m` → Joules, `kg*m/s^2` → Newtons, `J/s` → Watts), while
metric units and no-named-unit compounds stay (`1/s` stays `1/s`, not Hertz).
The **rabbit hole**: a single imperial unit converts to a magnitude-dependent
metric unit — Feet → `762/25` Centimeters, Miles → `25146/15625` Kilometers —
and the metric-prefix choice (cm vs m vs km) is WL-specific and unpredictable.
Implementing only the clean part would still diverge on the common cases.

### Temperature arithmetic is not affine

| expression | wolframscript | Woxi |
| --- | --- | --- |
| `Quantity[5,"Celsius"] + Quantity[3,"Celsius"]` | `Quantity[5543/10, Kelvins]` | `Quantity[8, DegreesCelsius]` |
| `Quantity[10,"Celsius"] - Quantity[3,"Celsius"]` | `Quantity[7, DegreesCelsiusDifference]` | `Quantity[7, DegreesCelsius]` |
| `Quantity[5,"Celsius"] + Quantity[3,"Kelvins"]` | `Quantity[8, DegreesCelsius]` | a `compat` error |
| `2 Quantity[5,"Celsius"]` | `Quantity[5563/10, Kelvins]` | `Quantity[10, DegreesCelsius]` |

Needs a full absolute-versus-difference temperature model with the
`DegreesCelsiusDifference` / `DegreesFahrenheitDifference` units.

### `MixedRadixQuantity` is unimplemented

Needs `MixedMagnitude` and `MixedUnit` first.

### WXF serialization bails

`BinarySerialize` deliberately leaves these unevaluated rather than write bytes
that would differ: a `Plus`/`Times` containing a numeric-complex term (WL sorts
`Complex` first in sums and folds numeric factors into it in products),
`Pattern`/`Function`/`Slot` expression kinds, `BigFloat`, and multi-operator
comparison chains. The deserializer treats the `NumericArray` token as corrupt,
since Woxi has no such object. `BinarySerialize[Range[3]]` is **packed** in WL —
only literal lists are comparable.

### CSV `ColumnTypes`

wolframscript types the last column of a labelled table as `"String"` when the
text does not end in a newline, contradicting its own import of the same field
as an integer. It also changes its CSV behaviour mid-session once any `"Table"`
import has run — `HeaderLines` is ignored in a cold kernel and honoured after,
and booleans are coerced in a cold kernel and left as strings after. Woxi
implements the cold-kernel behaviour and stays newline-agnostic.
**Not reproducible.**

Its `ColumnTypes` **values** are self-contradictory in the labelled two-column
case as well: `"a,b\n1,2"` types the last column as String although it holds
numbers, while `"a,b,c\n1,2,3"` types all three `Integer64`.

### `Import` element lists

WL's element list for CSV names `EventSeries` and `TimeSeries`, which Woxi does
not implement.

### Named characters and Unicode drift

wolframscript's Unicode database is older than Rust's. A sweep of
`CharacterNormalize` over the BMP and SMP found **zero** cases where
wolframscript normalizes and Woxi disagrees — every difference is the reverse,
Woxi normalizing a character wolframscript does not know yet (Latin Extended-D
additions, Vithkuqi, Latin Extended-F, Todhri, `㋿`). Woxi's output is the
standards-correct one. Never write a conformance test on a code point added
after Unicode ~13.

Note also that wolframscript stores most named characters at **private-use**
code points (`\[Rule]` is U+F522, `\[WarningSign]` U+F725), and mangles
non-ASCII input in both `-code` and `-file` — several apparent divergences in
this area are artifacts of that.

### `SeedRandom` streams differ

Wolfram uses a proprietary generator, Woxi uses ChaCha8; no seed produces the
same sequence. Verify determinism and distribution instead.
**Not reproducible.**

### Performance

Woxi's list operations are bound by expression-allocation throughput, not by
algorithmic complexity: `Expr::List` holds an owned vector with no structural
sharing, so building a large result deep-copies its elements.
`Tuples[{Range[4900], Range[70]}]` (343k pairs) takes ~1.5 s in release —
roughly 100× slower than wolframscript, which shares structure. A real fix
needs `Rc`/`Arc`-backed lists or a different allocator.

A `Table` of many `FindRoot` calls is slow the same way, for a different
reason: each Newton iteration re-substitutes the numeric guess into the whole
symbolic expression tree and walks it again with the general evaluator, and
every one of those recursive sub-evaluations pays `evaluate_expr_to_expr`'s
per-node bookkeeping (recursion-depth tracking, the `stacker` stack-growth
check, the termination latch). wolframscript's kernel instead compiles the
numeric expression once. As part of a scheduled QA routine, Woxi Studio was
tested against a randomly sampled Wolfram Demonstration notebook (a
chemical-engineering phase-equilibrium plot) whose `Manipulate` computed a
100×100 `Table` of `FindRoot` calls once in its `Initialization` to build an
`Interpolation`; this construct-equivalent example (invented constants and
variable names, similar expression nesting depth — not the Demonstration's
own code or data, which is copyrighted) reproduces both the shape and the
cost:

```wolfram
k1 = 3.2; k2 = 47.5; k3 = 6.1; m1 = 2.4; m2 = 33.8; m3 = 4.9; c1 = 1.3; c2 = 0.8;
a = 10^(k1 - k2/(k3 + T));
b = 10^(m1 - m2/(m3 + T));
g = Exp[c1 (c2 ((1 - u)/(c1 u + c2 (1 - u))))^2];
h = Exp[c2 (c1 (u/(c1 u + c2 (1 - u))))^2];
tbl = Table[
   FindRoot[u a g + (1 - u) b h - v == 0, {T, 1, 50}],
   {u, 0.01, 1, 0.01}, {v, 1, 100, 1}];
```

The 10,000 `FindRoot` calls take ~40 s in a release build against
wolframscript's sub-second evaluation — long enough that opening such a
notebook in Woxi Studio, which instantiates a stored `Manipulate`'s widget
(and so runs its `Initialization`) synchronously when the file loads,
freezes the UI for the whole computation. Like the `Tuples` case above, a
real fix is a faster/compiled numeric evaluation path rather than a
targeted change to `FindRoot` itself.

### `woxi eval` exits 0 on an evaluation error

The error goes to stderr but the exit code stays 0, so a shell check like
`woxi eval '…' >/dev/null 2>&1 && echo OK` reports OK even when the expression
failed. (`wolframscript -file` has the same property.)

### `woxi run` does not echo the value an aborted script ended on

`wolframscript -file` prints the terminal value of a script that stopped early
— `$Aborted` for `Abort[]`, `TerminatedEvaluation[RecursionLimit]` for a
recursion past `$RecursionLimit` — even though a script otherwise prints only
what `Print` sends:

```sh
printf 'Print["a"]\nAbort[]\nPrint["b"]\n' > /tmp/ab.wls
wolframscript -file /tmp/ab.wls  # a  + $Aborted
woxi run /tmp/ab.wls             # a
```

Both stop before `Print["b"]`, so only the echo is missing. `woxi eval` and
`wolframscript -code` agree (both print `$Aborted` /
`TerminatedEvaluation[RecursionLimit]`).

### `woxi repl` does not tag the output prompt with the result's form

When a REPL line evaluates to an output-form wrapper, wolframscript moves the
wrapper into the prompt and prints the wrapped expression:

```text
In[1]:= 7//FullForm
Out[1]//FullForm= 7      (* wolframscript *)
Out[1]= FullForm[7]      (* woxi repl     *)
```

Same for `MatrixForm`, `TableForm`, `InputForm` and the other form wrappers.
`woxi eval` is unaffected — script mode prints `FullForm[7]` on both sides.
The REPL gets only the formatted string back from `interpret`, so tagging the
prompt needs the result expression (and a per-wrapper rendering) plumbed
through to `repl.rs`.

### `balanced_ternary` computes a wrong value

`btmultiply["+-0++0+", btsubtract[tobt@-436, "+-++-"]]` gives `+--0` in Woxi
where wolframscript gives `----0+--0++0`. The divergence is inside `btadd`'s
`Fold` over a transposed, padded character matrix combined with a multi-clause
`StringReplace` over alternative patterns — a pattern-matching or accumulator
bug, not the (already fixed) `#0@` self-reference.

## Sockets

The TCP socket functions (`SocketOpen`, `SocketConnect`, `SocketListen`,
`SocketReadMessage`, `SocketReadyQ`, `SocketWaitNext`, `SocketWaitAll`,
`Sockets`) were implemented against behaviour recorded from wolframscript in
advance. The entries below are the parts of that surface **no wolframscript
run has confirmed**: the machine Woxi is developed on has no Wolfram Engine, so
these are Woxi's choices, not verified matches. Check each one against
wolframscript before writing a conformance test on it.

### Property *values* are unverified

The property *names* are the recorded ones —
`sock["Properties"]` gives `ConnectedClients`, `DestinationHostname`,
`DestinationIPAddress`, `DestinationPort`, `DirectionType`, `InprocQ`,
`Protocol`, `Scheme`, `SocketListener`, `Type`, `UUID`, and
`lis["Properties"]` gives `CharacterEncoding`, `HandlerFunctions`,
`HandlerFunctionsKeys`, `RecordSeparators`, `Socket` — but only
`DestinationPort` was ever seen with a value. Woxi answers `"TCP"` for
`Protocol`, `"tcp"` for `Scheme`, `"ZMQ_STREAM"` for `Type`, `"Server"` /
`"Client"` for `DirectionType`, `False` for `InprocQ`, `"UTF-8"` for
`CharacterEncoding` and `{}` for `RecordSeparators`. Wolfram's socket layer is
built on ZeroMQ, so the last three in particular may well read differently.

`c["SourcePort"]` returning `{}` *was* recorded, so Woxi answers `{}` for every
property it does not keep. Whether wolframscript does the same for an
arbitrary unknown property, or only for the ones it knows about but leaves
empty, is untested.

### Handler associations for events other than `"DataReceived"`

A `"DataReceived"` handler is called with the recorded seven keys —
`TimeStamp`, `SourceSocket`, `Socket`, `Data`, `DataBytes`, `DataByteArray`,
`MultipartComplete` — in that order. `"DataReceived"` is the only event name
wolframscript accepts: any other key makes `lis["HandlerFunctions"]` read
`Missing["NotAvailable", "HandlerFunctions"]`. Woxi keeps `"Accepted"`,
`"Closed"` and `"Error"` as extensions, and what wolframscript would pass such
a handler was never recorded; Woxi passes `TimeStamp`, `SourceSocket` and
`Socket`, plus a `Message` key for `"Error"`.

### A closed socket is closed again without complaint

wolframscript answers a second `Close` with a failure object naming the socket:

```sh
wolframscript -code 'srv = SocketOpen[0]; Close[srv]; Close[srv]'
# Failure["SocketsLink", <|"MessageTemplate" :> SocketObject::invalidSock,
#                          "MessageParameters" -> {SocketObject["TCPSERVER-…"]}|>]
woxi eval 'srv = SocketOpen[0]; Close[srv]; Close[srv]'
# "127.0.0.1:…"
```

The rest of Woxi's socket layer reports an unusable socket as `$Failed` plus a
free-text line rather than as a `Failure[…]`, so this is one instance of a
wider difference in how the two report a bad socket.

`Data` and `DataBytes` are delayed association entries (`key :> value`), so a
handler that only reads `DataByteArray` pays neither the text decoding nor one
`Expr` per byte. Which of the keys wolframscript delays is unverified; the
laziness itself is what the recorded behaviour called for.

### `SocketWaitNext` and `SocketWaitAll` on a timeout

The one recorded data point is that both stayed **unevaluated** for a client
socket with nothing pending and a one-second timeout. Woxi generalizes that:
each waits for activity on the listed sockets — an event a listener handled,
including one on a connection it accepted, or data waiting on the wire — and
leaves the call unevaluated when the timeout runs out. `SocketWaitNext`
returns the socket, `SocketWaitAll` the list it was given. Whether
wolframscript's rule is the same, or whether "unevaluated" was reporting
something narrower (a client socket being an invalid argument, say), is
unknown.

### `SocketReadMessage[sock, n]`

`SocketReadMessage[c, 2]` was recorded returning unevaluated with nothing
pending, which is equally consistent with wolframscript having no
two-argument form at all. Woxi reads up to `n` bytes of what has already
arrived and stays unevaluated when nothing has — agreeing with the
recorded case either way, but a superset if the form does not exist.

### An unusable socket is `$Failed`, not a `Failure[…]` object

Verified against wolframscript on 2026-09-10, and the root of most of the
entries in this section. Every operation on a socket wolframscript considers
unusable returns the *same object*:

```wolfram
Failure["SocketsLink",
  <|"MessageTemplate" :> SocketObject::invalidSock,
    "MessageParameters" -> {SocketObject["baef81c0-…"]}|>]
```

and prints nothing — reads, `Close` on an already closed socket, and every
property query, `"UUID"` and `"Properties"` included. (A read that fails for
another reason carries a different template, e.g.
`SocketReadMessage::zmqerror` with `{"Interrupted system call"}`.) Woxi prints
the free-text line and returns `$Failed` instead.

Displayed at top level the `Failure` renders as its message text with the
socket spelled as a front-end summary box (a `DisplayForm[TagBox[…]]` of
several kilobytes of `GraphicsBox`), so **no testcase that displays one can
conform** — which is why the closed-socket read lives in
`tests/interpreter_tests/sockets.rs` rather than in the doc tests.

*Resolved 2026-09-10:* Woxi used to keep answering property queries on a
**closed** socket (`Close[srv]; srv["DestinationPort"]` gave the port back
where wolframscript failed). It now rejects them all with `$Failed`, silently
— wolframscript prints nothing there either, since its failure is the
returned object. Only the shape of the failure value still differs.

### A refused connection blocks `SocketReadMessage` forever

Verified against wolframscript on 2026-09-10. Connecting to a port nothing
listens on still yields a socket, because Wolfram connects lazily:

```sh
wolframscript -code 'c = SocketConnect["127.0.0.1:1"]; Head[c]'   # SocketObject
```

`SocketReadMessage[c]` then waits for a first byte that never arrives and
never returns — there is no default timeout. Woxi detects the unusable socket
instead and answers `The socket object … is invalid or not open.` plus
`$Failed`.

`tests/cli/sockets/SocketReadMessage.md` used to run exactly that as a
testcase; it now shows it as an untested example and leaves the behaviour to
the unit tests, because no bound on the call makes it conform. Never run it
against wolframscript: a stuck `wolframscript`
holds the single license slot, so every *later* invocation fails with the
misleading `Your Wolfram Engine installation is not activated`, which turns
one hanging testcase into dozens of unrelated failed documents. `scrut`'s
per-document timeout does not prevent this — it reaps its own shell but not
the `wolframscript` below it. The `wo` wrapper caps each attempt at
`WO_TIMEOUT` seconds (300 by default) and kills the kernel with it for
exactly this reason.

### Error text

`The socket object … is invalid or not open.` and
`Failed socket operation: …` are free text, not tagged messages: they leave
`$MessageList` empty, which is what was recorded. The exact wording of the
second was recorded only for a privileged-port bind (`Permission denied`);
Woxi uses the operating system's message for every bind failure, and reports
`$Failed` as the value.

### A socket read races its own listener

Reading a connection directly (`SocketReadMessage`, `ReadString`, …) while a
`SocketListen` handler is attached to it means two readers on one socket, and
which of them gets a given chunk is a matter of timing. wolframscript polls
the same socket from both places and presumably has the same problem, but this
was not tested. Use one or the other, not both.

### Timing of handler calls

Handlers run on the thread that evaluates Wolfram code — the only thread that
may touch an `Expr` — so they fire where evaluation waits: `Pause`, a blocking
socket read, `SocketWaitNext`/`SocketWaitAll`, `SocketReadyQ` with a timeout,
and between statements. A long computation with no such point defers every
handler until it ends. The recorded wolframscript behaviour agrees (a handler
observably fired during a later blocking call rather than at write time), but
only for that one case.

### No transport but TCP

`SocketOpen[port, "UDP"]`, ZeroMQ schemes (`"inproc://…"`, `"ZMQ_*"` socket
types) and `SocketOpen`'s `"ZMQ_STREAM"`-family options are not implemented:
anything that is not TCP leaves the call unevaluated. `Sockets["TCP"]` is
accepted and is the same as `Sockets[]`.

### `Together[…, Modulus -> p]` only handles univariate rational functions

```sh
woxi eval 'Together[1/x + 1/y, Modulus -> 3]'   # stays unevaluated
```

The modular path cancels over GF(p) with univariate polynomial arithmetic,
so a multivariate fraction (or a composite modulus) is returned unevaluated.

### `BesselJ/I/Y/K` at complex arguments lose accuracy for large `|z|`

```sh
woxi eval 'BesselJ[0, 40. + 5. I]'   # power series, cancellation error
```

Complex (and negative-real `BesselY`/`BesselK`) arguments are evaluated with
the plain power series in machine precision, which is accurate for moderate
`|z|` (roughly below 12) but suffers cancellation beyond that; there is no
asymptotic-expansion branch yet.

### `ColorData[4, k]` indexed scheme is not tabulated

```sh
woxi eval 'ColorData[4, 9]'   # stays unevaluated
```

Only the indexed schemes 1, 2, 3, 30, 35 and 97 are tabulated; a graphic that
colors with scheme 4 renders those primitives with the default color.

### Named `VertexShapeFunction`/`EdgeShapeFunction` shapes are approximations

`VertexShapeFunction -> "Capsule"`, `"Star"`, `"Triangle"`, `"FiveDown"`,
`"ConcaveHexagon"`, `"Parallelogram"` and `"RoundedUpTrapezoid"` are drawn as
a plain polygon or rounded rectangle at roughly the right size; the exact
proportions differ from wolframscript. Named edge shapes other than
`"Line"`/`"Arrow"` (e.g. `"CarvedArrow"`, `"DashedLine"`, `"DottedLine"`,
`"DiamondLine"`, `"FilledArcArrow"`) still fall back to the plain edge.
`GraphData` also lacks most atlas names (e.g. `"PappusGraph"`,
`"HeawoodGraph"`), so Demonstrations that pick graphs by name still fail.

## Colorize

- `Colorize[m, ImageSize -> …]` ignores `ImageSize` (`Image` carries no
  display size), and the `Automatic` palette is Woxi's own distinct-hue
  scheme rather than Wolfram's exact colors.
- `ColorFunction -> "HypsometricTints"` (and other `ColorData` gradients
  not yet implemented) falls back to a gray ramp.
- On an image, a pixel at exactly 5/8, 6/8 or 7/8 lands one table entry
  (one byte) above wolframscript's: `Colorize[Image[{{0.75, 1.}}],
  ColorFunction -> (RGBColor[#, #, #] &)]` gives byte 192, wolframscript
  191. No index rule that fits these also fits a dense sweep of other
  values, where `min(floor(1024 x), 1023)` matches.
- On an integer-type image with no `ColorFunction` (label colouring)
  `Colorize` stays unevaluated.

## PolyhedronData["BilinskiDodecahedron", ...]

Geometry (unit edges, volume, surface area, face structure) is derived from
the golden-rhombus zonohedron; the vertex order, orientation and the
`"Classes"` list were not checked against `wolframscript`.

### `PolyhedronData` golden rhombic solids and `"NetFaces"` are unverified

`AcuteGoldenRhombohedron`, `ObtuseGoldenRhombohedron` and `RhombicIcosahedron`
are built as unit-edge zonohedra (z axis = the polar symmetry axis) without
access to `wolframscript`, so the vertex/face order, orientation and the printed
form of the exact metrics may differ from Wolfram's. Their `Circumradius` and
`Midradius` are `Missing["NotApplicable"]` (the vertices/edges are not
equidistant); the rhombic icosahedron's `Inradius` is too. `"NetFaces"` returns
a `GraphicsComplex` over de-duplicated 2D points (the shape a Demonstration
indexing `p[[1, #]] & /@ p[[2, 1]]` expects) — inferred from that usage, not
checked directly.

### `PolyhedronData["GyroelongatedSquareDipyramid", …]` ordering is unverified

The vertex/face order and orientation of the solid are built without access to
`wolframscript` and may differ from Wolfram's; `Circumradius`, `Inradius` and
`Midradius` are `Missing["NotApplicable"]`. The `"NetCoordinates"` lattice is
the strip layout a Demonstration caches, but `"NetFaceIndices"` (16 triangles:
an 8-triangle strip plus a row of 4 above and 4 below) was inferred from how
that Demonstration folds the net, so its face numbering is not checked directly.

### `PolyhedronData["SmallStellatedDodecahedron", …]` vertex/face ordering is unverified

The vertex order, face order/winding and the printed form of the exact
metrics were derived from the geometry (icosahedral vertices, {5/2, 5}
pentagram faces, unit pentagram chord) without access to `wolframscript`, so
they may be ordered or simplified differently from Wolfram's.

### `PolyhedronData["GreatIcosahedron", …]` ordering and volume are unverified

Vertices reuse the small stellated dodecahedron's order and faces are the 20
outward-wound unit triangles of mutual second neighbours, derived from the
geometry without access to `wolframscript`; Wolfram's vertex/face order may
differ. `"Volume"` is not derived and reports `Missing["NotAvailable"]`.

### `Solve` roots with a symbolic parameter keep the quadratic-formula form

The roots are right but not written the way wolframscript writes them.
`Solve[(x - 3)^2 == a, x]` gives `(6 - 2*Sqrt[a])/2` here and `3 - Sqrt[a]`
in WL; `Solve[(x/2 + 1)^2 == c, x]` is `(-4 + 4*Sqrt[c])/2` against
`2*(-1 + Sqrt[c])`; `Solve[y == 1/(x - 3)^2, x]` is
`(6*y - 2*Sqrt[9*y^2 - y*(-1 + 9*y)])/(2*y)` against
`(-1 + 3*Sqrt[y])/Sqrt[y]`, and the cube's principal root is
`3 + (y^(-1))^(1/3)` against `3 + y^(-1/3)`. WL's forms look like `Factor`
of the inverted single power (`Factor[3 - 1/Sqrt[y]]`), but the rule that
picks them is not pinned down; numeric coefficients already agree.

### `Solve` of a cubic with symbolic coefficients is unevaluated

`Solve[x^3 + b x + c == 0, x]` (and any cubic whose coefficients or
right-hand side are symbols, e.g. `1/(2 u^2) - k/u - k/u^3 == b`) stays
unevaluated; `wolframscript` returns the closed-form Cardano radicals. The
`Orbital Precession of Mercury` Demonstration depends on one such `Solve`
for its Schwarzschild window intercept, so that value is left symbolic (the
three plots still render).

### `Solve` through a linear denominator uses its own root form

When a rational equation depends on the unknown only through one denominator
sum linear in it (`a + b q`), the roots are found for that sum and mapped
back (`q -> (root - a)/b`). They are equal to Wolfram's but not guaranteed to
be printed in the same arrangement.

### `AxesLabel` on `ParametricPlot` is not drawn

`ParametricPlot[…, AxesLabel -> {Re[z], Im[z]}]` omits the axis labels (seen
in the `Logarithmic Integral on the Critical Line` Demonstration).

### `WordData` is not available

`WordData[word]` (parts of speech, definitions) ships no data in Woxi, so it
stays unevaluated. The `Oulipo: Wordshift + 7` Demonstration needs it (its
`Manipulate` is initialised from it together with an `ExampleData` text,
which Woxi generates), so only its static content and stored output render. `DictionaryLookup` works
against the bundled word list, which may differ from Wolfram's dictionary.

### `Texture` in `Plot3D` / `TextureCoordinateFunction` is ignored

`Plot3D[…, PlotStyle -> Directive[Texture[image]], TextureCoordinateFunction -> …]`
draws the surface with the default coloring instead of mapping the image onto
it (seen in the `Developing a Perspective Anamorphic Image` Demonstration).
`ImageForwardTransformation` and `HighlightImage`, which that notebook's flat
view uses, are implemented.

### `MorphologicalPerimeter` edge handling is unverified

`MorphologicalPerimeter[image, t]` marks foreground pixels that touch a
background pixel horizontally or vertically; pixels outside the image are
ignored (not treated as background). Whether Wolfram counts the image border
as background, and its exact connectivity, has not been checked against
`wolframscript` (seen in the `Segmenting a Medical Image` Demonstration).

### `NDSolve` with `WhenEvent` and `Sow` does not run

`NDSolve[{…, WhenEvent[t[s] == 1, Sow[{x[s], y[s]}]]}, …]` over an implicit
(differential-algebraic) system stays unevaluated, so `Reap` collects nothing.
The `All Real Roots of a Nonlinear System of Equations` Demonstration relies
on this to mark its roots: the contour curves render, but the red root points
and the root table are missing.

### `InverseLaplaceTransform` of a product of symbolic-pole factors

`InverseLaplaceTransform` of a product of many `1/(s - pole)` factors with
numeric complex poles returns unevaluated, so the time-response plot in the
`Poles and Zeros of Time Domain Response Functions` Demonstration stays empty
(its pole/zero plane renders fine).


### `ReplaceAll` of many call sites by a large `InterpolatingFunction`

`Table[x[i][t], {i, 80}, {t, 0, 388, 0.1}] /. sol[[1]]` with the NDSolve
result of an 80-equation system (the default view of the `Circulant Ring
Systems` Demonstration) deep-copies the tabulated data for every one of the
~310,000 replaced `x[i]` heads, so it takes minutes instead of Wolfram's
shared-structure instant. Applying an `InterpolatingFunction` itself is
O(log points) now; only the per-occurrence copy remains.

### Upvalues whose pattern ends in `OptionsPattern[]` need an explicit option

`Foo[c_Curve, OptionsPattern[]] ^:= …` is stored with a fixed argument count,
so `Foo[Curve[1]]` (no options) does not match it, while `Foo[Curve[1], a -> 2]`
does. In Wolfram `OptionsPattern[]` also matches zero arguments. The
`Turtle Fractals` Demonstration defines such a `Show` upvalue, which its
`Manipulate` never calls.

### `ExampleData[{"Text", name}]` is generated text

The texts exist with their titles, authors, languages and line structure, but
their words are generated (see the `ExampleData` section). The `Effect of
Keyword Length on the Letter Frequency in Encrypted Messages` Demonstration
builds its message popup from `StringTake[ExampleData[{"Text", #}], 5000]`, so
its chart shows the letter frequencies of the generated English text — close
to, but not the same as, the original's.

### `HamiltonianCycles` of an `UndirectedGraph[…]` result

`GraphUtilities`HamiltonianCycles` searches the graph's sparse adjacency matrix
in its stored order. `UndirectedGraph[g]` of a directed `g` keeps, in Wolfram,
the storage order of `Transpose[m] + m` — every vertex's in-neighbours of `g`
before its out-neighbours — while a `Graph` built from the same edges is stored
ascending. Woxi's graphs carry no storage order, so it searches the
`UndirectedGraph` result ascending and can report other cycles first (the same
set with `All`).

### `Rotate[Text[…], angle]` inside a `Grid` cell is not rotated

A `Grid` cell holding `Rotate[Text["Value"], 90 Degree]` (the usual vertical
axis label beside a `GraphicsGrid`, as in the `Intramarginal Rent`
Demonstration) is drawn unrotated. Rotation is honoured for plot labels
(`PlotLabel`/`AxesLabel`) but not for `Grid` cell content.

## Combinatorica graph functions and `{"Prism", n}` (unverified against wolframscript)

`Combinatorica`FromUnorderedPairs`, `V`, `ToAdjacencyLists`,
`ConnectedComponents`, `BreadthFirstTraversal[g, v]` and `ShortestPath` were
written from the package's documented behaviour (as needed by the `Two Points
on a Polyhedron` Demonstration) and could not be checked against a real kernel:
the circular-embedding vertex coordinates, the tie-breaking order of the
traversal/shortest path, and the `Level`/`Edge` flags of
`BreadthFirstTraversal` (left unevaluated) may differ. Likewise the vertex
order, face order and orientation of `PolyhedronData[{"Prism", n}, …]` are a
guess; `"InsphereRadius"`/`"Midradius"` report `Missing["NotApplicable"]`.


## `Mean` of a `TruncatedDistribution` over a discrete base

`Mean[TruncatedDistribution[{1, Infinity}, GeometricDistribution[0.25]]]`
integrates the density instead of summing the probability mass, so it gives
`5.186…` rather than `4` (found while sampling the distribution for the
`Block Bootstrap for Time Series` Demonstration). `Quantile`/`CDF` of a
truncated discrete distribution likewise exclude the lower bound's own mass.


## `ColorData[n]` for most indexed schemes

Only some indexed schemes are tabulated (see `ColorData` in `functions.csv`);
e.g. `ColorData[61][k]` stays unevaluated, so the `Manipulate` of the
`A Minimal Circumcircle Measure of District Compactness` Demonstration loses
its fill/line colors. The same holds for `ColorData[7, k]` and
`ColorData[13, k]` (the `Circumcircles of Two Midpoints and an Altitude`
Demonstration colors its triangle, discs and background with them). The
schemes' values could not be checked against a real kernel. `ColorData[8][k]`
(the row backgrounds of the `Grid` in the `The Perfect Venn Diagram`
Demonstration's plot title) is in the same state. `ComputationalGeometry`ConvexHull` starts its index list at the
leftmost corner, which is likewise unverified.


## Text width of `Row`/`Grid` captions is overestimated

Text is measured at a fixed 8.4 px per character, which is wider than the
proportional face the FrontEnd uses for digits and spaces. A `Pane[…, {450,
300}]` holding `Row[{Grid[…], Graphics[…], Grid[…]}]` captions (the
`Power versus Work` Demonstration) therefore lays out wider than the pane and
clips its right-hand column, where the FrontEnd fits it.


## Machine complex numbers with a `0.` imaginary part

```sh
wolframscript -code '{LogGamma[2. + 0. I], Erf[0.5 + 0. I], Zeta[2. + 0. I], BesselJ[0, 2. + 0. I]}'
# {3.552713678800501*^-15 + 0.*I, 0.5204998778130466 + 0.*I, 1.6449340668482264 + 0.*I, 0.22389077914123562 + 0.*I}
woxi eval '{LogGamma[2. + 0. I], Erf[0.5 + 0. I], Zeta[2. + 0. I], BesselJ[0, 2. + 0. I]}'
# {LogGamma[2. + 0.*I], 0.5204998778130465, 1.6449340668482264, BesselJ[0, 2. + 0.*I]}
```

Arithmetic, `Power`, `Sqrt`, `Exp`, `Log`, `Sign`, `Gamma` and the circular
and hyperbolic functions keep the `0.` imaginary part (they read the argument
through `machine_complex_parts`). Special functions such as `LogGamma`,
`PolyGamma`, `Zeta`, `Erf` and the Bessel family still either drop it or stay
unevaluated.


## `Sum` over symbolic bounds with a step stays unreduced

`Sum[k^i, {i, q h, 0, -q}]` is re-indexed to `Sum[k^(h q - q i), {i, 0, h}]`
but not summed in closed form (the kernel gives `(-1 + k^(q + h q))/(-1 +
k^q)`). Applying a definition stored from it to numbers evaluates correctly.
Steps that do not divide a symbolic span (`Sum[k^i, {i, 0, n, 2}]`) stay
unevaluated.

## Automatically sized `Graphics` in a `Row` inside a fixed `Pane`

A `Pane[Grid[{{Row[{Graphics[…], Graphics[…], …}]}, …}], {300, 300}]` cell
keeps each automatically sized picture at its default width, so the row is far
wider than the pane (arrows 330 points long). The FrontEnd shrinks those
pictures to fit the pane. `fit_grid_to_pane_width` only squeezes pictures that
are direct `Grid` cells or nested `Grid` columns, not pictures inside a `Row`
cell.

## `NDSolve` of complex periodic PDEs

A complex-valued solution of a 1-D PDE with periodic boundaries (e.g. the
nonlinear Schrödinger equation) is returned as
`u -> Function[{t, x}, re[t, x] + I im[t, x]]` built from two real
`InterpolatingFunction`s, where `wolframscript` returns a single complex
`InterpolatingFunction`. Values agree to the solver tolerance, but the
printed form of the result differs.

## HighlightMesh draws no highlighting

`HighlightMesh[mesh, spec]` evaluates to the bare mesh. wolframscript also
highlights the cells `spec` names and, for `Labeled[0, "Index"]`, writes each
vertex's index next to it; Woxi draws neither.

## `PolyhedronData` covers only a subset of the polyhedron entities

Only the polyhedra listed in `src/functions/polyhedron_data.rs` are known.
Further entities (e.g. `"SquashedDodecahedron"`, the Johnson solids,
`"Antiprism"` / `"Dipyramid"` / `"Pyramid"` families, `"TriakisTetrahedron"`)
give `PolyhedronData::notent`, so notebooks that offer them in a menu
(such as polyhedron projection demonstrations) fail on those entries.

### EdgeDetect only implements `Method -> "Canny"`

`EdgeDetect[image, r, t, Method -> m, "StraightEdges" -> s]` accepts the
options, but only `Method -> "Canny"` (or `Automatic`) with
`"StraightEdges" -> 0` is computed. The `"ShenCastan"` and `"Sobel"` methods
and non-zero `"StraightEdges"` leave the call unevaluated.


## `Solve` of a nonlinear numeric system returns spurious solutions first

`Solve[{eq1, eq2}, {x, y}]` for two rational equations with machine-number
coefficients (such as the equilibrium equations of a binary vapor–liquid
mixture) returns ten solutions, mostly complex, with an unphysical real one
(`x -> -1.2*^16`) first. `wolframscript` lists the physical solution first,
so `sol[[1]]` — the usual way to pick it — gives a different value.

### `DSolve` has no Riccati / Bessel-type solutions

`DSolve[α y'[x] + γ y[x]^2 == x^(k-2), y[x], x]` stays unevaluated. wolframscript
reduces it to the linear equation `u'' == (γ/α^2) x^(k-2) u` and returns ratios
of `Bessel*` / `Hypergeometric0F1Regularized` functions (elementary for special
`k`). Notebooks that wrap it in `y[x] /. DSolve[…]` (e.g. Riccati-equation
Manipulates) therefore only draw the equation panel, and `ReplaceAll::reps`
fires for the unevaluated `DSolve`.

### `ToString[TraditionalForm[…]]` is not a linear-syntax string

wolframscript returns the string `"\!\(TraditionalForm\`…\)"` (box markup in
linear syntax, e.g. `x\^\(k - 2\)`); Woxi returns an unevaluated
`DisplayForm[FormBox[…]]`, so `Text[ToString[TraditionalForm[…]]]` in a
Manipulate shows the raw box structure.

### `FindShortestTour` picks a different tour among equally short ones

The length always matches, and with at most 12 points the tour is optimal.
When several tours are equally short, Woxi can return a different one, or the
same one in the other direction. Examples:

- Collinear `{{0,0},{3,0},{1,0},{2,0}}`: WL `{1, 4, 2, 3, 1}`, Woxi `{1, 3, 4, 2, 1}`.
- A 7-point set where WL chose the direction whose second stop has the lower
  index. Woxi always reports the direction whose second stop has the higher
  index, which matches every other small case probed.
- Duplicate points: WL starts at point 2 (`{2, 4, 1, 3, 2}`).

Above 12 points Woxi uses iterated 2-opt/Or-opt local search. On random
instances the tour comes within about 0.4% of WL's length and is sometimes
shorter. `FindShortestTour[graph]` returns the cheapest Hamiltonian cycle along
the graph's edges (or `{}`); it has not been compared against WL.

### `RectangleChart` spacing and 3D layout are approximations

The default gap between rectangles in WL drifts slightly with the number of
rectangles: 0.098 for two, 0.097 for three, and 0.096 within datasets when
there are several. Woxi uses a fixed 0.0975. Woxi also renders neither WL's
hover tooltips nor the `"RoundingRadius"` option. `RectangleChart3D` stacks
plain cuboids from y = 0 and leaves out WL's shaded base plate, face grids,
`ViewPoint -> {2, -10, 2}` and box ratios.

### Grid `Interpolation` in two or more dimensions differs in the last digit

Woxi builds an n-D grid interpolation as a tensor product of the 1-D local
Lagrange pieces, with the last axis innermost. wolframscript returns the same
values but can round the last bit differently. For example, `x + y z` sampled
on a 3×3×3 grid gives `0.6400000000000001` at `{0.3, 1.7, 0.2}` where WL
gives `0.6399999999999999`. Partial derivatives (`Derivative[1, 0, 0][f]`)
can come out a few ULP away from WL's value, so a mixed derivative that WL
returns as exactly `0.` can be about `1*^-16` in Woxi.

### Symbolic values: `Interpolation` and `Fit` write their results differently

```sh
wolframscript -code 'Interpolation[{{1, a}, {2, b}, {3, c}, {5, d}}, InterpolationOrder -> 1][2.5]'
# 0.5*(b - c) + c
woxi eval '…same…'   # 0.5*b + 0.5*c
```

Values that are not numbers, such as symbols or `PlusMinus[v, u]`, are
interpolated as a linear combination of the values. Woxi writes it as each
value times its weight; wolframscript keeps the Newton divided-difference
form. `Fit` with such responses uses wolframscript's form, `s_j (P_j1 y_1 +
…)` with unit-length design columns, but the pseudo-inverse's last digits
(and tiny entries such as `-2.45*^-16 b` versus `2.2*^-16 b`) differ.

### `NDSolve` ignores the `"FixedStep"` sub-method

`Method -> {"FixedStep", Method -> "ExplicitEuler"}` (or any other
sub-method) is accepted but not honoured: Woxi integrates with its own
accurate scheme. For `y'[t] == -y[t], y[0] == 1` on `{t, 0, 1}` with
`StartingStepSize -> 0.1`, WL's explicit Euler gives `y[1] = 0.9^10 =
0.348678`, while Woxi returns `0.367879`. Step-size stability demonstrations
therefore do not show the instability WL does.

### `FindMinimum`/`FindMaximum` take Newton steps, wolframscript quasi-Newton

```sh
wolframscript -code 'FindMaximum[Log[x] - x, {x, 3}]'
# FindMaximum::nrnum: The function value -1. - 3.14159 I is not a real number at {x} = {-1.}.
# {-1.3068528194400546, {x -> 2.}}
woxi eval 'FindMaximum[Log[x] - x, {x, 3}]'
# FindMaximum::nrnum: The function value 4.09861 + 3.14159 I is not a real number at {x} = {-3.}.
# {-1.9013877113318902, {x -> 3.}}
```

wolframscript's first step is a unit step along the gradient and later
steps use a secant estimate of the curvature; Woxi's damped Newton uses the
exact second derivative from the start, so the two visit different points
and stop at different ones when a trial point leaves the real domain. Both
end with the same message.

### A fit with zero residual degrees of freedom has no parameter table

```sh
wolframscript -code 'lm = LinearModelFit[{{0, 1}, {1, 0}}, x, x]; Quiet[lm["ParameterTableEntries"]]'
# {{1., Indeterminate, Indeterminate, Indeterminate}, {-1., Indeterminate, Indeterminate, Indeterminate}}
woxi eval 'lm = LinearModelFit[{{0, 1}, {1, 0}}, x, x]; lm["ParameterTableEntries"]'
# FittedModel::elmntav: "ParameterTableEntries" is not an available property.
```

wolframscript divides by the zero degrees of freedom (with `Power::infy`
messages) and reports `Indeterminate` standard errors and statistics; Woxi
does not store the table for such a fit.

### `Style[Column[{Style[…, color], …}], size]` inside `Text` loses the per-item colors

```sh
wolframscript -code 'ExportString[Graphics[Text[Column[{Style["a", Blue], Style["b", Red]}], {0, 0}]], "SVG"]'
woxi eval 'ExportString[Graphics[Text[Column[{Style["a", Blue], Style["b", Red]}], {0, 0}]], "SVG"]'
```

wolframscript draws each line in its own color; Woxi flattens the `Column`
to one multi-line text label, which carries a single fill color.

### Inexact trig `Solve` and complex `ArcTan[x, y]` differ in the last digits

```sh
wolframscript -code 'InputForm[Solve[Sin[b] == 0.5 Cos[b], b]]'
# {{b -> -2.677945044588987}, {b -> 0.4636476090008061}}
woxi eval 'InputForm[Solve[Sin[b] == 0.5 Cos[b], b]]'
# {{b -> -2.6779450445889874}, {b -> 0.4636476090008059}}
wolframscript -code 'InputForm[ArcTan[1.2, 0.5 I]]'
# 0. + 0.44365159750045163*I
woxi eval 'InputForm[ArcTan[1.2, 0.5 I]]'
# 0. + 0.44365159750045147*I
```

The solutions are the same principal values in the same order; Woxi reaches
them through machine roots of the resolvent in `Cos[b]` and
`ArcTan[c, s]`, wolframscript through an evaluation order it does not
document, so a few ulps differ. Complex two-argument `ArcTan` is evaluated
as `-I Log[(x + I y)/Sqrt[x^2 + y^2]]`, which matches wolframscript's
branches but not its last digits.

### `N` evaluates inside `Hold`

```sh
wolframscript -code 'N[Hold[ArcTan[1, 2]]]'
# Hold[ArcTan[1., 2.]]
woxi eval 'N[Hold[ArcTan[1, 2]]]'
# Hold[1.1071487177940904]
```

wolframscript only numericizes the numbers inside a held expression; Woxi
evaluates the numericized calls as well.

### `CountryData[…, {"Shape", {projection, center}}]` coordinates

The shape is drawn from the bundled 1:110m outlines, projected on a unit
sphere with the center given as `{lat, lon}`. wolframscript uses finer
outlines and its own coordinate scale, so point counts and coordinates
differ. Hemispheric projections (`"Orthographic"`) drop the vertices on the
far side instead of clipping the outline at the horizon, and the conic
projections use fixed standard parallels (`"Albers"`) or the center
latitude (`"LambertConic"`, `"Bonne"`).

### Numeric radicals without a shared prime: WL splits some rational bases

`Sqrt[7/3] 5^(1/5)` is `(5^(1/5)*Sqrt[7])/Sqrt[3]` in WL but stays
`Sqrt[7/3]*5^(1/5)` in Woxi, and likewise `(7/3)^(1/3) 5^(1/5)`. WL does not
split `Sqrt[3/7] 5^(1/5)`, `(2/3)^(1/3) 5^(1/5)` or `Sqrt[7/3] 5^(1/5) x`, so
the rule (it seems to depend on the base being above 1 and the product being
purely numeric) is not pinned down. Radicals that share a prime are merged
like WL (`Sqrt[7/3] 3^(1/5)` is `Sqrt[7]/3^(3/10)`).

### `KernelMixtureDistribution` exact `Variance` and display

- `Variance[KernelMixtureDistribution[{1, 2, 4}]]` is
  `(7*(200 + 81*3^(3/5)))/900` in Woxi and the equal
  `(7*(243 + 200*3^(2/5)))/(900*3^(2/5))` in WL, whose bandwidth square keeps
  `3^(2/5)` in the denominator — evaluated alone, WL's own
  `189/(100 3^(2/5))` absorbs the 3 just like Woxi. Interval-rule bandwidths
  (`{1, 2, 3, 4, 5}`) match.
- `CDF[KernelMixtureDistribution[{1, 2, 4}], 3.]` is `0.6550092996714579`
  in Woxi (and in WL's own `N[CDF[…, 3]]`) but `0.6550092996714578` in WL.
- WL's `InputForm` prints the container's tags unquoted
  (`DataDistribution[KernelMixture, {1/3, Gaussian, …}, 1, 3]`); Woxi keeps
  them as strings, like the `"Empirical"` and `"Histogram"` containers.


### Weierstrass functions at machine precision: last digits and imaginary noise

Woxi evaluates `WeierstrassZeta`, `WeierstrassSigma` and complex
`WeierstrassP`/`WeierstrassPPrime` through the lattice's theta function; the
values agree with WL to about 14 significant digits, not to the last bit.
For real arguments WL attaches a rounding-noise imaginary part
(`WeierstrassZeta[1.5, {1., 2.}]` is `0.49214540663758793 +
4.6454696025598575*^-17*I`), where Woxi returns the real `0.492145406637588`.

### `ButterworthFilterModel`: exact highpass and bandpass normalization

The exact highpass and band filters are the same transfer functions as WL's
but scaled differently: `ButterworthFilterModel[{"Bandpass", 2, {1/3, 1}}]`
is `((4 s^2)/9)/((1/3 + …)(1/3 + …))` in Woxi and `(4 s^2)/((1 + … + 3 s^2)
(1 + … + 3 s^2))` in WL, which clears each factor's denominators (and for a
highpass multiplies each factor by a unit). Lowpass models, machine-number
models and the filtered values match. `ToDiscreteTimeModel` of a machine
model agrees with WL to the last one or two digits.

### `Short` only truncates integers

`Short[expr]` / `Short[expr, n]` on a list, sum or other compound
expression is returned unchanged instead of eliding the middle as
`<<k>>`, so `Panel[Short[Range[100]]]` in a `Manipulate` shows every
element. Only the digits of an over-long integer are shortened
(`shorten_for_display` in `src/functions/graphics.rs`).

## ImageRotate: exact pixel-corner ties at an arbitrary angle

`ImageRotate` at an angle that is not a right angle samples the source pixel
under each output pixel centre. When that point falls exactly on a source
pixel corner (the canvas centre of an even-sized image on an odd canvas),
wolframscript's choice depends on its transform's rounding noise:
`ImageRotate[Image[{{1, 0}, {0, 1}}, "Byte"], 1]` has a `0` in the centre,
Woxi takes the lower-right pixel (`1`). At `Pi/4` both pick the lower-right.

## Exact `Eigenvectors` with quadratic-surd components: unreduced form

```sh
wolframscript -code 'Eigenvectors[{{0, 0, 1, 0}, {1, 0, 1, 0}, {1, 1, 0, 0}, {1, 1, 0, 0}}]'
# {{-((-1 - Sqrt[5])/(3 + Sqrt[5])), 1, 1, 1}, {-1, 0, 1, 1}, {-((1 - Sqrt[5])/(-3 + Sqrt[5])), 1, 1, 1}, {0, 0, 0, 1}}
woxi eval 'Eigenvectors[{{0, 0, 1, 0}, {1, 0, 1, 0}, {1, 1, 0, 0}, {1, 1, 0, 0}}]'
# {{(7 + 3*Sqrt[5])/(11 + 5*Sqrt[5]), 1, 1, 1}, {-1, 0, 1, 1}, {(7 - 3*Sqrt[5])/(11 - 5*Sqrt[5]), 1, 1, 1}, {0, 0, 0, 1}}
```

The values are equal, but the elimination leaves a different quotient, so
`N` of the two differs in the last digits.

### A name repeated across list arguments is not checked for equality

```wolfram
h[{a_}, {a_}] := a;
h[{1}, {2}]
# Woxi:         1
# wolframscript: h[{1}, {2}]
```

List argument patterns are decomposed into per-element accessors, which
bind each name independently, so the repeated `a` never has to agree
(nested calls such as `f[g[i_], g[i_]]` are checked correctly).

## SparseArray: internal entry order of sums, `Select` keeping sparsity

- `Plus @@ SparseArray[{{1, 2}, {0, 5}}]` equals wolframscript's value, but
  wolframscript stores the sum's entries in a different order
  (`{{2}, {1}}` with values `{7, 1}`, Woxi `{{1}, {2}}` with `{1, 7}`), so the
  `InputForm`s differ.
- `Select[sparse, pred]` returns a dense list; wolframscript keeps a
  SparseArray when the background passes `pred`.
- `Fold`, `FoldList`, `Sort` and `Union` of a SparseArray emit `::normal` and
  stay unevaluated in wolframscript; Woxi does not reject them.

## DualLinearProgramming with `Method -> "InteriorPoint"`

wolframscript's interior-point solution carries solver noise
(`{1.0000000003499416, 1.0000000000555387}` for an optimum of `{1, 1}`);
Woxi solves exactly and converts, giving `{1., 1.}`.

## A conditional `:=` after a `=` value keeps the `=` value

wolframscript drops a symbol's `Set` value when a conditional own value is
added (`w = 5; w := {1} /; c == 1` leaves only the conditional rule, so `w`
stays `w` when the test fails), but keeps an unconditional `:=` value as the
fallback. Woxi does not record whether a symbol's value came from `=` or
`:=`, so it keeps either one as the fallback (`w` evaluates to `5`).

## Transcendental `Root[{f, x0}]` objects

- wolframscript stores `x0` as an arbitrary-precision number of about 20
  digits (`1.7292555589818595724706451954508223956`20.53`); Woxi keeps the
  correctly rounded machine number (`1.7292555589818597`).
- For an equation with negative exponents wolframscript first rewrites `f`
  (`2/2^t + 2/3^t == 1` gives `-2 - 2*E^(Log[2]*#1 - Log[3]*#1) +
  E^(-(Log[3]*#1) + (Log[2] + Log[3])*#1) &`); Woxi uses `lhs - rhs`
  (`-1 + 2^(1 - #1) + 2/3^#1 &`). The roots agree.
- `Root::invrt` is reported when the refined root is more than 10^-6
  (relative) away from `x0`; wolframscript's exact threshold is somewhere
  between 1.6*10^-7 and 4.8*10^-6 for the cases probed.

## FindGeometricTransform: last digits and noisy perspective fits

The transforms agree with wolframscript to about 1e-9, but not in the last
digits (wolframscript's exact-fit error is `2.29*^-15` where Woxi's is a
different tiny number). With more than four noisy plane points the
default (perspective) fit is Woxi's linear least-squares homography, which
need not be the minimizer wolframscript reports.

## Bates, Davis and Fisher z distributions

- The linear `1 - x` term of an upper `CDF[BatesDistribution[…]]` piece is
  written `c (-1 + x)` by Woxi; wolframscript does so for
  `BatesDistribution[4]` and `BatesDistribution[3, {-1, 2}]` but writes
  `-(4*(1 - x))/9` for `BatesDistribution[2, {-1, 2}]`.
- `BatesDistribution` with a symbolic `n` or interval keeps PDF/CDF
  unevaluated (wolframscript does the same for symbolic `n`, but gives a
  form for a symbolic interval with numeric `n`).
- `Quantile[DavisDistribution[…], p]` stays unevaluated: its CDF has no
  closed form (wolframscript leaves the symbolic CDF too, but inverts it
  numerically).
- In `Mean[FisherZDistribution[n, m]]` Woxi orders the factor
  `n^2*(m/n)^(n/2)`; wolframscript writes `(m/n)^(n/2)*n^2`.
- Numeric PDF/CDF/Quantile values can differ in the last digit.

## A left-hand-side `/;` guard is printed on the right-hand side

```sh
wolframscript -code 'f[p_] /; p == 1/2 := 2; DownValues[f]'
# {HoldPattern[f[p_] /; p == 1/2] :> 2}
woxi eval 'f[p_] /; p == 1/2 := 2; DownValues[f]'
# {HoldPattern[f[p_]] :> 2 /; p == 1/2}
```

A rule stores its guard in a trailing condition slot whichever side it was
written on. The two forms match the same calls; only the printed rule differs.

## Maximize of `(1 - x)/(Log[1/x] + 1)` on `[0, 1]` is numeric, not `ProductLog`

```sh
wolframscript -code 'Maximize[{(1 - x)/(Log[x^(-1)] + 1), 0 <= x <= 1}, x]'
# {(1 - E^(2 + ProductLog[-1, -E^(-2)]))/(-1 - ProductLog[-1, -E^(-2)]), {x -> E^(2 + ProductLog[-1, -E^(-2)])}}
woxi eval 'Maximize[{(1 - x)/(Log[x^(-1)] + 1), 0 <= x <= 1}, x]'
# {0.3178444328993727, {x -> 0.31784443289937303}}
```

The critical point solves `Log[1/x] + x == 0`-type equations that Woxi's
Solve does not invert to `ProductLog`, so the optimizer falls back to Newton.

## `RegionConvert[…, "Implicit"]` of a parallelogram is not reduced

```sh
wolframscript -code 'RegionConvert[Parallelogram[{1, 2}, {{1, 0}, {1, 1}}], "Implicit"] /. {\[FormalX] -> x, \[FormalY] -> y}'
# ImplicitRegion[x <= y && 1 + x >= y && Inequality[2, LessEqual, y, LessEqual, 3], {x, y}]
woxi eval 'RegionConvert[Parallelogram[{1, 2}, {{1, 0}, {1, 1}}], "Implicit"] /. {\[FormalX] -> x, \[FormalY] -> y}'
# ImplicitRegion[0 <= 1 + x - y <= 1 && 0 <= -2 + y <= 1, {x, y}]
```

Woxi bounds the parallelogram's coordinates `t = Inverse[{v1, v2}ᵀ] . (x - o)`
between 0 and 1. wolframscript reduces that system further for some vectors and
leaves it as is for others (`Parallelogram[{0, 0}, {{-1, -1}, {-1, 0}}]` gives
the same `0 <= -y <= 1 && 0 <= -x + y <= 1` in both). The regions are equal.
Only boxes, balls (round or axis-aligned ellipsoids) and parallelepipeds are
converted; other regions stay unevaluated.

## `CanonicalGraph` of a highly symmetric graph labels differently

```sh
wolframscript -code 'EdgeList[CanonicalGraph[PetersenGraph[]]][[;; 3]]'
# {1 <-> 3, 1 <-> 4, 1 <-> 7}
woxi eval 'EdgeList[CanonicalGraph[PetersenGraph[]]][[;; 3]]'
# {1 <-> 2, 1 <-> 3, 1 <-> 4}
```

Woxi orders vertices by their refined degree colouring and settles the ties
left over with the lexicographically smallest edge list. That reproduces
wolframscript's labels for graphs that degree refinement separates (paths,
stars, trees, cycles, complete bipartite graphs) but not for vertex-transitive
ones, where wolframscript's tie-breaking rule is unknown. Isomorphic graphs
still come out identical in Woxi.

## Generalized eigenvectors with a singular `b` can differ in sign

```sh
wolframscript -code 'Eigensystem[{{{1., 0}, {0, 2.}}, {{1., 0}, {0, 0}}}]'
# {{Infinity, 1.}, {{0., 1.}, {1., 0.}}}
woxi eval 'Eigensystem[{{{1., 0}, {0, 2.}}, {{1., 0}, {0, 0}}}]'
# {{Infinity, 1.}, {{0., -1.}, {-1., 0.}}}
```

With `b` singular Woxi takes the eigenpairs of `Inverse[a] . b` and inverts
the eigenvalues (`Infinity` for its zero ones). The eigenvalues agree, but the
vectors keep that reduced problem's sign choice and the finite eigenvalues can
differ in the last bit.

## Text with a Row of Style-sized labels

- `Graphics[Text[Row[{Style["12", Bold, 54], …}], pos]]` draws the row at the
  default font size: the `Style` sizes inside a `Row` passed to `Text` are
  not applied, so a barcode's printed digits come out much smaller than in
  wolframscript.
