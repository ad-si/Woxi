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
wolframscript -code 'ToString[Unevaluated[Map[f, g]], InputForm]'   # f /@ g
woxi eval 'ToString[Unevaluated[Map[f, g]], InputForm]'             # Map[f, g]
```

The affected heads are the ones with operator spellings: `ReplaceAll` (`/.`),
`ReplaceRepeated` (`//.`), `Map` (`/@`), `Apply` (`@@`), and `Part` (`[[…]]`).
`Part` is the mildest of these — it renders correctly on its own and only
diverges inside a hold (`ToString[Unevaluated[Part[a, 2]], InputForm]`).

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
`ToString[1.5*^10]` and 2D rationals inside message text all print flat.

### `NumberForm` does not switch to scientific notation

```sh
wolframscript -code 'ToString[NumberForm[123456789.]]'  # 1.23457 × 10^8
woxi eval 'ToString[NumberForm[123456789.]]'            # 123457000.
```

Also `NumberForm[1000000.]` (WL `1. × 10^6`) and `NumberForm[1.5*10^-8]`
(WL `1.5 × 10^-8`). In-range reals — roughly `10^-5 ≤ |x| < 10^6` — agree.

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

### TeXForm stacks a rational coefficient that wolframscript factors out

```sh
wolframscript -code 'ToString[TeXForm[(3 x^2 - 1)/2]]'  # \frac{1}{2} \left(3 x^2-1\right)
woxi eval 'ToString[TeXForm[(3 x^2 - 1)/2]]'            # \frac{3 x^2-1}{2}
```

Same for `LegendreP[2, x]`, which evaluates to that expression in both
engines. A sum without a numeric term is stacked by both
(`(a + b)/2` → `\frac{a+b}{2}`, `3 (a + b)/2` → `\frac{3 (a+b)}{2}`), so the
trigger looks like a numeric term inside the numerator; the rule was not
pinned down.

`x^(1/(2 y))` is a smaller instance of the same class: WL writes
`x^{\left.\frac{1}{2}\right/y}`, Woxi `x^{\frac{1}{2 y}}`.

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

Relatedly, `expr_to_boxes` typesets TraditionalForm with StandardForm glyphs:
`Sin[x]/2` is `FractionBox[RowBox[{Sin[, x, ]}], 2]` against WL's
`RowBox[{sin, (, x, )}]`, and `Pi` stays `Pi` instead of `π`.

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

Woxi has three independent canonical-order comparators (one for `Sort`, one for
`Order`, one for `Plus` terms) and they drift from each other as well as from
Wolfram. Diagnose an ordering divergence by testing `Sort[…]`, `Order[a, b]`
and the bare `a + b` separately.

### Plus orders a reciprocal monomial before a power of a sum

```sh
wolframscript -code 'ToString[1/x + Sqrt[1 - x^(-2)], InputForm]'  # Sqrt[1 - x^(-2)] + x^(-1)
woxi eval 'ToString[1/x + Sqrt[1 - x^(-2)], InputForm]'            # x^(-1) + Sqrt[1 - x^(-2)]
```

Only the *reciprocal* monomial diverges — `x + Sqrt[1 - x^2]` agrees, and the
`Times` counterpart of the rule is implemented.

### Plus puts a composite term before a bare symbol

`1 - n + Floor[k]` in WL against `1 + Floor[k] - n` in Woxi. Surfaces in
symbolic distribution CDFs with mixed symbol and `Floor` terms, e.g.
`CDF[PascalDistribution[n, p], k]`.

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

### Nested sum-versus-sum factor order

`(-1 + x)*(1 + (-2 + x)/2)` — WL emits the more-nested factor first. Value
identical, display only. Surfaced through `InterpolatingPolynomial`'s Newton
form.

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

### `(1/2)^k` is not rewritten to `2^(-k)`

WL rewrites a positive unit-fraction base with a **non-numeric** exponent:
`(1/2)^k` → `2^(-k)`, `(1/2)^Pi` → `2^(-Pi)`, `(1/2)^(k+1)` → `2^(-1-k)`.
Woxi keeps the unit fraction. Non-unit numerators (`(2/3)^k`) and negative
bases (`(-1/2)^k`) agree.

Doing it eagerly regresses `BinomialDistribution`'s PDF, which relies on
`(1/2)^x*(1/2)^(10-x)` merging to `(1/2)^10`; the rewrite has to happen after
same-base `Times` merging, or the integer-base merge has to combine
`2^a*2^b` → `2^(a+b)` for symbolic exponents.

Woxi also normalizes `(3/11)^(2/3)` to `(9/121)^(1/3)`, where WL keeps the
primitive base and even rewrites the other direction.

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

```sh
wolframscript -code 'Exp[1000] // N'   # 1.97007111401704699388887935224`12.95…*^434
woxi eval 'Exp[1000] // N'             # Infinity

wolframscript -code '1.0*^308 * 10'    # 1.00000000000000001097906362944`15.95…*^309
woxi eval '1.0*^308 * 10'              # Infinity
```

Anything that grows past ~1.8*^308 is affected, so `Exp`/`Power` of a few
hundred silently loses the answer rather than approximating it.

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
than a dispatch one. `N[Zeta[3], 25]` is a separate gap: there is no
arbitrary-precision `Zeta`.

### `N[I]` stays exact

`N[I]` is `I` in Woxi and `0. + 1.*I` in wolframscript, so `N[Gamma[I]]` and
`N[LogGamma[I]]` echo instead of numericizing. Fixing it needs
`scalar * (0. + c*I)` to fold first, and that non-fold is deliberate: a pure
imaginary product is left as the `c*I` monomial so it still merges inside an
enclosing `Plus`. Both changes were tried and reverted.

`BesselJ`/`PolyGamma` at complex numeric arguments are separately unimplemented.

### `Complex[Real, Real]` has the wrong head

```sh
wolframscript -code 'Head[Complex[0., 0.]]'   # Complex
woxi eval 'Head[Complex[0., 0.]]'             # Plus
```

Woxi has no dedicated `Complex` expression variant, so a real-real complex is
rewritten to `Plus[Real, Times[Real, I]]` during evaluation. The printed string
matches; the head does not.

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
scaled, or Welford). Affects `Variance`, `Correlation`, `Kurtosis`, `Cumulant`
and `Variance` of an `Audio`. **Not reproducible.**

`PageRankCentrality[CycleGraph[4]]` is `{0.25000000000000006, …}` against WL's
clean `{0.25, …}` — power-iteration noise in the same class.

### FresnelF and FresnelG lose accuracy through composition

Woxi composes them from `FresnelS`/`FresnelC`; wolframscript uses dedicated
rational approximations. Real outputs differ in the last 1–2 digits for small
`|x|`, and the cancellation as `S, C → 1/2` degrades relative accuracy to
~1e-13 by `x = 10`. **Not reproducible** without WL's coefficients.

### `BesselI` and `BesselJ` differ by 1–2 ULP

`BesselI[0, 3]` is `…5025` in Woxi and `…50235` in WL. `BesselK` and `BesselY`
are bit-exact. Affects e.g. `PDF[VonMisesDistribution[2, 3], 2.5]`'s last digit.

Also missing from Woxi's numeric folding: `Gamma[rational]`, so
`2. + Gamma[1/3]` stays symbolic where WL gives `5.357877069415496`.

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

### `StepMonitor` never fires for an unconstrained `FindMinimum`

Both engines report `noopmon` and monitor nothing for the *constrained*
`{f, cons}` form, but WL does fire the monitor once per step of the
unconstrained one, where Woxi's Newton iteration ignores it entirely:

```sh
wolframscript -code 'Length[Reap[FindMinimum[(x - 2)^2, {x, 5}, StepMonitor :> Sow[x]]][[2]]]'  # 1
woxi eval 'Length[Reap[FindMinimum[(x - 2)^2, {x, 5}, StepMonitor :> Sow[x]]][[2]]]'            # 0
```

Matching it needs Woxi's iteration count to match WL's as well, not just the
plumbing.

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

The last ULP of `atanh`, `acos`, `asinh` and friends differs between macOS and
Linux, so a full-precision string assertion is platform-dependent. The same
1-ULP FMA difference flips a single 8-bit colour channel in `ComplexPlot`
domain-colouring output at an exact `x.5` boundary.


## Algebra and calculus

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

### `Factor` with `GaussianIntegers`, `Extension` or `Trig`

Recognized as valid options but the call stays unevaluated; WL factors
`x^2+1` into `(-I + x)*(I + x)`. Same for `FactorList` and
`IrreduciblePolynomialQ`. Composite-modulus `LinearSolve`, n-ary modular
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

### Second-order PDEs with a lower-order term

`DSolve[u_xx + u_y == 0]` (the heat equation) is unevaluated; WL falls back to
a *particular* solution with `C[1] … C[8]` after `DSolve::lpdeprtclr`. With
`a == 0` or `c == 0` WL also writes the characteristics unnormalised
(`C[1][x - y] + C[2][x]`) rather than as `λ x + y`.

### `NDSolve` covers ODEs only

```wolfram
NDSolve[{D[u[x,t],t] == D[u[x,t],x,x], u[x,0] == Sin[Pi x],
         u[0,t] == 0, u[1,t] == 0}, u, {x,0,1}, {t,0,1}]
```

returns unevaluated. `NeumannValue` is out of scope, `DirichletCondition`
exists only as a symbol, and `Method -> {"MethodOfLines", …}` has nothing
behind it. On the symbolic side `DSolve` recognises three first-order
two-variable PDE shapes; Laplace, which WL solves as
`C[1][I x + y] + C[2][-I x + y]`, is not among them.

`NDSolve`'s DAE support handles an index-1 constraint that solves explicitly
for one unknown; quadratic, coupled or index ≥ 2 constraints, and constraints
whose unknown is also differentiated, are not covered.

### `NDSolve` uses a fixed grid

Woxi integrates on 1000 nominal RK4 steps with bisection refinement;
wolframscript adapts. `Length[(y /. s[[1]])[[2]]]` is 1001 against 13. Note
that stock wolframscript's *default* tolerances can be **less** accurate than
Woxi's here — get a converged reference before treating a difference as a bug.

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
- `FindInstance` over the integers falls back to a bounded search when `Solve`
  cannot decide: every variable is walked outwards from zero (0, 1, −1, 2, −2,
  …) within a fixed evaluation budget. That reproduces wolframscript's answer
  for the small instances (`x^5 + y^5 + z^5 == w^5 && x > 0` → `{1, 0, 0, 1}`)
  but cannot reach a large one — Euler's sum-of-powers counterexample
  `27^5 + 84^5 + 110^5 + 133^5 == 144^5` stays unevaluated where wolframscript
  finds it.

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

### Solve over the integers drops a range constraint

```sh
wolframscript -code 'Solve[Mod[x, 3] == 1 && 0 <= x < 10, x, Integers]'
# {{x -> 1}, {x -> 4}, {x -> 7}}
woxi eval 'Solve[Mod[x, 3] == 1 && 0 <= x < 10, x, Integers]'
# Solve[Mod[x, 3] == 1, x]
```

Bounded linear systems over the integers are already enumerated; a `Mod`
congruence with an explicit range is the same shape. The returned expression
having lost both the bound and the domain makes this look worse than a plain
unevaluated result.

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

### Symbolic `Fit` and `LinearModelFit`

`LinearModelFit[…]["BestFit"]` drops the `0. +` constant term wolframscript
keeps, and `Fit[{1,2,3,4},{1,x},x]` gives a clean `0. + 1. x` where WL keeps
`5.07*^-16 + 0.9999999999999999 x` — Woxi is the less noisy one.

For a **rank-deficient** design wolframscript is not minimum-norm:
`Fit[{{1,1},{2,2},{3,3}}, {1, x, x^2, x^3}, x]` gives coefficients of norm
0.7546 where the minimum-norm solution has norm 0.6134. Both interpolate;
WL switched to some other regularization there. **Not reproducible.**

### Constrained and weighted fitting find different local minima

On a non-convex constrained problem the two engines land on *different valid*
local minima — for a sine fit Wolfram lands on the constraint boundary while
Woxi finds an interior point with a smaller residual. Compare the picture, not
the parameters.

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

### `SatisfiabilityInstances` ordering

The default single instance (greedy True-first DFS) matches, and an
unsatisfiable problem gives `{}`. The **multi-instance and `All` ordering**
follows WL's internal BDD structure and differs per expression —
`a||b||c` orders 7,3,1,5,2,6,4 while `Majority[a,b,c]` orders 7,6,5,3.
**Not reproducible.**


## Special functions

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


## Linear algebra

### Float matrices with complex eigenvalues

Wolfram complexifies the **whole** result and orders each conjugate pair with
`+I` first: `{0. + 1.*I, 0. - 1.*I, 1. + 0.*I}`. Woxi gives the value-correct
but form-divergent `{0. - 1.*I, 0. + 1.*I, 1.}`, and non-block complex cases
stay unevaluated. Complex `Eigenvectors` for n ≥ 3 are unevaluated too, and
radical eigenvector components order differently (`(-Sqrt[5] + I)/3` against
`(I - Sqrt[5])/3`).

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


## Expression structure and evaluation

### Rubi loads and integrates, but not everything it computes agrees

The [Rubi](../../rubi.md) rule base is the densest available exercise of the
pattern matcher and the definition store: 7000 rules, all read back out of
`DownValues` and rewritten before use. It loads unmodified and integrates, and
on a 30-integral sample 20 answers are character-for-character
`wolframscript`'s, with 6 of the remaining 10 the same function written another
way. What is left, all of it ordinary Woxi behaviour rather than anything about
the package:

- **Loading takes about a minute** against roughly twenty seconds under
  `wolframscript`, and Rubi's step-display machinery (`$LoadShowSteps = True`,
  the default) rewrites all 7000 rules on load — which Woxi has not finished
  after half an hour and ten gigabytes. `Steps`, `Step` and `Stats` are
  therefore out of reach.
- **`Int[Sin[x]^3*Cos[x]^2, x]` and `Int[Sin[x]*Cos[x]^3, x]`** run for minutes
  and exhaust memory. `wolframscript` answers both instantly.
- **`Int[ArcSin[x], x]`** comes back as `Defer[Int][ArcSin[x], x]`. The rule is
  loaded — it sits at index 6964 of `DownValues[Int]` where wolframscript has
  it at 5130, behind the `Unintegrable` fallback that is meant to catch what it
  declines. Woxi orders two rules by how much structure each pattern carries;
  the language asks whether one's match set is contained in the other's. The
  two disagree for patterns that are simply incomparable
  (`f[(d_.*x_)^m_.*(a_. + b_.*g[x_])^n_.]` matches products that
  `f[(a_. + b_.*g[x_])^n_.]` never sees), and every later rule of that kind
  walks the general one further toward the back. Approximating containment
  with a cheap shape key was tried and reverted — it moved the errors rather
  than removing them, trading `Int[ArcTan[x], x]` for `Int[1/(1 + x^3), x]`.
  The fix is the containment test itself, made cheap enough to run at each of
  7000 insertions.
- **`Int[x/(a + b*x^2), x]`** is `Log[1 + b x^2/a]/(2 b)` where Rubi gives
  `Log[a + b x^2]/(2 b)` — off by a constant, from a different rule firing.
- **Equivalent but differently shaped answers** are common and not bugs as
  such: `Int[Sec[x]^2, x]` is `Sec[x]*Sin[x]` rather than `Tan[x]`,
  `Int[E^x*x, x]` is `-Gamma[2, -x]` rather than `E^x*(x - 1)`,
  `Int[x/(1 + x^4), x]` and `Int[1/(a + b*Cos[x]), x]` pick other valid
  antiderivatives, and sums come out in Woxi's own `Plus` order.

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

### ListCorrelate / ListConvolve have no multi-dimensional overhang

```sh
wolframscript -code 'ToString[ListCorrelate[{{1, 1}, {1, 1}}, {{a, b, c}, {d, e, f}, {g, h, i}}, 1], InputForm]'
# {{a + b + d + e, b + c + e + f, a + c + d + f},
#  {d + e + g + h, e + f + h + i, d + f + g + i},
#  {a + b + g + h, b + c + h + i, a + c + g + i}}
woxi eval 'ListCorrelate[{{1, 1}, {1, 1}}, {{a, b, c}, {d, e, f}, {g, h, i}}, 1]'
# ListCorrelate[{{1, 1}, {1, 1}}, {{a, b, c}, {d, e, f}, {g, h, i}}, 1]
```

The two-argument multi-dimensional form is correct; only the overhang path
(`k` / `{kL, kR}`, padding, generalized `g`/`h`) is one-dimensional, and it
stays unevaluated for a rank-2 kernel rather than answering.

The 7th argument, a level specification, is unimplemented for every rank:

```sh
wolframscript -code 'ToString[ListCorrelate[{x, y}, {a, b, c}, 1, p, Times, Plus, 1], InputForm]'
# {a*x + b*y, b*x + c*y, c*x + p*y}
woxi eval 'ListCorrelate[{x, y}, {a, b, c}, 1, p, Times, Plus, 1]'
# ListCorrelate::argb: called with 7 arguments; between 2 and 6 arguments are expected.
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
is missing. The sweep originally found 80; the 67 below are what is left.

Each one is a form wolframscript accepts and Woxi rejects on arity — so the
failure mode is an `::argt` / `::argb` / `::argx` message rather than a wrong
answer, which makes them safe but individually invisible.

| Function | Woxi | WL up to | `ArgumentsPattern` |
| --- | --- | --- | --- |
| `AbsoluteCorrelation` | 1–2 | 3 | `{_, _., _.}` |
| `AiryAiZero` | 1–1 | 2 | `{_, _.}` |
| `AiryBiZero` | 1–1 | 2 | `{_, _.}` |
| `AngerJ` | 2–2 | 3 | `{_, _, _.}` |
| `AngleBisector` | 1–1 | 2 | `{{_, _, _}, _.}` |
| `ArcCurvature` | 2–2 | 3 | `{{__}, _, _.}` |
| `BesselYZero` | 2–2 | 3 | `{_, _, _.}` |
| `BooleanMaxterms` | 2–2 | 3 | `{_, _., _.}` |
| `BooleanMinterms` | 2–2 | 3 | `{_, _., _.}` |
| `CellularAutomaton` | 1–3 | 4 | `{_, _., _., _.}` |
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
| `EulerAngles` | 1–1 | 2 | `{_, Optional[{_, _, _}]}` |
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
| `MultipleHarmonicNumber` | 1–2 | 3 | `{_, Optional[{__}], Optional[{__, _}]}` |
| `NotebookDirectory` | 0–0 | 1 | `{_.}` |
| `ParentDirectory` | 0–1 | 2 | `{_., _.}` |
| `PerfectNumber` | 1–1 | 2 | `{_, _.}` |
| `PolyLog` | 2–2 | 3 | `{_, _, _.}` |
| `PositionLargest` | 1–2 | 3 | `{_, _., _.}` |
| `PositionSmallest` | 1–2 | 3 | `{_, _., _.}` |
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
- `PositionLargest` / `PositionSmallest`: the remaining slot is an
  *orderfun*, but wolframscript rejects `Greater` and `Less` there with
  `::nord3` (it wants an `Order`-style comparison returning ±1/0), and
  `Order` itself is the default — so there is little behaviour to match.
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


## Messages and error handling

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

### Rasterize with an unknown element aborts the whole script

```sh
wolframscript -code 'Print[ToString[Rasterize[x, "Text"], InputForm]]; Print["after"]'
# Rasterize::elmntavl: "Text" is not an available element. Possible elements
# include "BoundingBox", "Data", "Graphics", "RasterSize", "Regions", and "Image".
# $Failed
# after
woxi eval 'Print[Rasterize[x, "Text"]]; Print["after"]'
# Error: Evaluation error: Rasterize: unsupported expression type
```

The hard error kills the run, so nothing after it evaluates. `Rasterize`
should emit `Rasterize::elmntavl` and return `$Failed`.

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
- Messages that embed a fraction are rendered as 2D layouts by wolframscript
  and 1D by Woxi. This is systemic across the distribution and `Select::normal`
  message families.
- Message **multiplicity** in general is not comparable: wolframscript
  re-evaluates a failing specification, so it prints some messages twice, and
  applies `General::stop` after three identical ones.

### `ArgMax` with an invalid constraint form

`ArgMax[{list}, fn]` — WL emits `ArgMax::consf`, Woxi silently echoes.

### `Failure` prints its formatted message

In script mode wolframscript does not print a `Failure[…]` expression — it
prints the failure's *formatted message*, wrapping every substituted parameter
in `DisplayForm[TagBox[…, Short[#1, 3] &]]`. That is typeset internals leaking
into script mode; Woxi prints the expression and renders `["Message"]` cleanly.
**Deliberately not reproduced** — test a `Failure`'s keys, not its printed form.


## Graphics and plotting

Woxi renders its own SVG; `wolframscript` prints `-Graphics-` and exports
cairo-rendered SVG, so pixel or markup equality is never the goal. What follows
is where the *drawn* result differs.

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

### Tick step from the padded range

Woxi derives the tick step from the *padded* display range, so a `ListPlot`
whose data ends at 6e15 can pick a coarser step than Wolfram, which uses the
data range.

### `ParametricPlot` samples uniformly

`max(PlotPoints, 500)` points, no adaptive refinement (`Plot` has it). On a
51-term trig sum with `PlotPoints -> 100` wolframscript emits 6052 line points
against Woxi's 500. The values agree exactly and the shape is the same; only
small-scale wiggles are under-resolved.

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

A textured solid renders as flat-shaded default-blue polygons. Implementing it
means carrying uv pairs per triangle and emitting the image once into `<defs>`
with per-triangle clip paths and an affine matrix — a renderer feature, and one
that cannot be checked against wolframscript by diffing SVG (it rasterises 3D
output).

### `Complex*` plot functions are unimplemented

`ComplexPlot`, `ComplexPlot3D`, `ComplexListPlot`, `ComplexArrayPlot`,
`ComplexContourPlot`, `ComplexRegionPlot`, `ComplexVectorPlot` and
`ComplexStreamPlot` are registered as graphics-producing heads (so `Head[…]`
behaves) but have no implementation; wolframscript renders all of them.

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
against the rendered picture.

### `Manipulate` control gaps

- A `TabView` of controls is flattened — every tab's controls are shown at
  once, since Woxi's control panel is one flat list.
- One unrecognised control spec makes the **entire** `Manipulate` give up, so a
  single unsupported control takes every other control with it.
- `Manipulate[…, Initialization :> …]` leaks its definitions into the session.

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

### FindThreshold is unimplemented and its return convention is unclear

```sh
wolframscript -code 'FindThreshold[Image[{{0., 1.}}]]'   # 0.498046875
woxi eval 'FindThreshold[Image[{{0., 1.}}]]'
# FindThreshold[-Image-] is a built-in Wolfram Language function not yet implemented in Woxi.
```

The Otsu machinery already exists — `EdgeDetect` uses it, and for the gradient
images it is applied to it reproduces wolframscript exactly: bin into 256 bins
spanning `[min, max]`, maximise the between-cluster variance, and return
`min + k (max - min)/256` for the winning bin index `k`.

What blocks exposing it as `FindThreshold` is that the reported value is not
always on that grid. `FindThreshold[Image[{{0., 1.}}]]` is `0.498046875`, which
is `127.5/256` — a bin *centre*, where the gradient cases return a bin *edge*.
Other probes give `0.4988281324331183` for `{{0.2, 0.8}}` but an exact `0.25`
for `{{0.25, 0.5, 0.75}}`, so the parity is inconsistent across inputs. This
also blocks the one-argument (Otsu-default) form of `MorphologicalBinarize`.

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

### `AudioTrim` on tiny objects

wolframscript gives `{{1., 0.}}` for `{0., 1., 0.}` but `{{1.}}` for
`{1., 0., 1.}`, dropping a non-silent sample — minimum-window artifacts at
44.1 kHz, not a rule worth matching. `FourierDCT`/`FourierDST` differ in the
last ULP.

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

### Multi-entry rule lists inside a graph option

WL scrambles the order of a multi-entry rule list inside an option
(`VertexLabels -> {2 -> "b", 1 -> "a"}` for the input `{1 -> "a", 2 -> "b"}`) —
internal hash order. Woxi keeps input order.


## Geometry and regions

### `TriangleCenter` accepts a triangle embedded in 3D

`TriangleCenter[Triangle[{{0,0,0},{4,0,0},{0,3,0}}], "Circumcenter"]` is
`{2, 3/2, 0}` in Woxi; wolframscript handles only the planar case and leaves
the call unevaluated. Deliberate — the 3D centre is well defined.

### `ConvexHullMesh` is unevaluated for 3D point sets

```sh
wolframscript -code 'ToString[ConvexHullMesh[{{0,0,0},{1,0,0},{0,1,0},{0,0,1}}], InputForm]'
# BoundaryMeshRegion[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
#   {Polygon[{{3, 2, 1}, {2, 4, 1}, {4, 3, 1}, {3, 4, 2}}]},
#   Method -> {"SeparateBoundaries" -> False}, WorkingPrecision -> Infinity]
woxi eval 'ConvexHullMesh[{{0,0,0},{1,0,0},{0,1,0},{0,0,1}}]'
# ConvexHullMesh[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}]
```

Computing the hull is the easy part; WL hands qhull's internal facet
bookkeeping straight through, and three things would have to be replicated to
match the printed `Polygon`:

1. **Facet order.** No sort explains all three samples below.
2. **Vertex rotation within a face.** Faces are outward-oriented, but the
   starting vertex varies.
3. **Coplanar merging.** Triangles that share a plane come back as one polygon,
   so the cube's six faces are quads, not twelve triangles.

Reference outputs (all carry `Method -> {"SeparateBoundaries" -> False},
WorkingPrecision -> Infinity`):

| points | faces |
| --- | --- |
| `{{0,0,0},{1,0,0},{0,1,0},{0,0,1}}` | `{{3,2,1},{2,4,1},{4,3,1},{3,4,2}}` |
| the same plus `{1,1,1}` | `{{3,2,1},{2,4,1},{4,3,1},{3,5,2},{5,4,2},{4,5,3}}` |
| the eight unit-cube corners | `{{3,2,1,4},{1,2,6,5},{4,1,5,8},{2,3,7,6},{3,4,8,7},{5,6,7,8}}` |

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

### `ExampleData` bundles its own data and properties

Woxi lists Wolfram's whole 228-entry `"NetworkGraph"` catalogue (so
`ExampleData["NetworkGraph"]` and `ExampleData::notent` agree), but ships the
data for only the handful of classic networks it assembled from the original
sources, and exposes the properties that data supports — `"VertexList"`, `"EdgeRules"`, `"AdjacencyMatrix"` — none of which
wolframscript knows (it answers `ExampleData::notpropx`). Its own list is
`ByteCount, Description, EdgeCount, EdgeProperty, FullGraph, Graph,
LongDescription, Name, Source, StandardName, VertexCount, VertexProperty`.
`ExampleData[]` likewise names only the two collections Woxi serves, against
wolframscript's nineteen, and the vertex *names* of a bundled network follow
the original publication (`"MlleBaptistine"`) rather than Wolfram's spelling
(`"Mlle Baptistine"`). Deliberate: the catalogue is Wolfram's. Write tests against shape and
presence, never against either side's catalogue.

### `ExampleData[{"TestImage", name}]` ships no pixels

```sh
wolframscript -code 'ExampleData[{"TestImage", "Couple"}]'  # Image[NumericArray[…]]
woxi eval 'ExampleData[{"TestImage", "Couple"}]'            # ExampleData[{TestImage, Couple}]
```

Deliberate: the photographs behind that catalogue are not Woxi's to
redistribute. The *name* catalogue is bundled, so `ExampleData["TestImage"]`
returns the same `{"TestImage", name}` pairs and a script that builds an image
picker from it (the "Histogram Equalization" Demonstration's popup, say) still
gets the real entries. Asking for the data itself stays unevaluated rather than
returning invented pixels.

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

### `woxi eval` exits 0 on an evaluation error

The error goes to stderr but the exit code stays 0, so a shell check like
`woxi eval '…' >/dev/null 2>&1 && echo OK` reports OK even when the expression
failed. (`wolframscript -file` has the same property.)

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
