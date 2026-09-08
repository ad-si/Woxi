---
icon: lucide/sigma
---

# Rubi

[Rubi] — the **Ru**le-**b**ased **I**ntegrator — is a Wolfram Language package
that computes indefinite integrals by applying a decision tree of about 7000
rewrite rules. It is not part of Woxi: it is an ordinary `.m` package that Woxi
reads and runs, and one of the largest Wolfram Language codebases there is, so
it doubles as a stress test for Woxi's pattern matcher and definition store.

[Rubi]: https://rulebasedintegration.org


## Installing

A [release] ships as a `.paclet` file, which is a ZIP archive:

```sh
curl -L -O \
  https://github.com/RuleBasedIntegration/Rubi/releases/download/4.17.3.0/Rubi-4.17.3.0.paclet
unzip Rubi-4.17.3.0.paclet
```

That unpacks a `Rubi-4.17.3.0/` directory holding `Rubi.m`, the
`IntegrationRules/` tree and a `PacletInfo.m`.
No patching is needed — Woxi reads the package as it ships.

[release]: https://github.com/RuleBasedIntegration/Rubi/releases


## Loading

Either read `Rubi.m` directly:

```wolfram
$LoadShowSteps = False;
Get["Rubi-4.17.3.0/Rubi.m"];
```

or point `PacletDirectoryLoad` at the directory the paclet was unpacked *into*
and load it by context:

```wolfram
PacletDirectoryLoad["."];
$LoadShowSteps = False;
Needs["Rubi`"];
```

Both read the same 200 rule files and end with about 7400 `Int` down-values.
`$RubiVersion` reports which version is loaded.

Loading takes roughly a minute and about a gigabyte of memory, and it happens
again in every process — the `.mx` fast-load path in the paclet's
`Kernel/init.m` needs `DumpSave`, which Woxi does not implement, so nothing is
cached between runs.

`$LoadShowSteps = False` switches off Rubi's step-display machinery. Left at
its default of `True`, Rubi additionally rewrites all 7400 rules to record
the steps they take, which costs about two more minutes of loading and makes
`Steps`, `Step` and `Stats` available (see below). `Int` itself is the same
either way.


## Integrating

`Int[expr, x]` is the antiderivative of `expr` with respect to `x`:

```wolfram
Int[1/(a + b*x), x]
(* Log[a + b*x]/b *)

Int[x^2*Sqrt[a^2 - x^2], x]
(* (Sqrt[a^2 - x^2]*x^3)/4 - (a^2*x*Sqrt[a^2 - x^2])/8
     + (a^4*ArcTan[x/Sqrt[a^2 - x^2]])/8 *)
```

A list of integrands is integrated term by term:

```wolfram
Int[{Sin[x], x*Log[x]}, x]
(* {-Cos[x], -1/4*x^2 + (x^2*Log[x])/2} *)
```

and an iterator gives the difference of the antiderivative's limits at the
endpoints:

```wolfram
Int[x^2, {x, 0, 1}]
(* 1/3 *)
```

Rubi's answers are usually not the ones `Integrate` gives — that is the point
of the package. Woxi's own `Integrate` is unaffected by loading Rubi; the two
live side by side.


## Showing the steps

With the step display loaded (`$LoadShowSteps` left at `True`),
`Steps[Int[expr, x]]` prints the rule applied at every step; with
`RubiPrintInformation -> False` it instead returns the recorded steps and the
result. Each intermediate expression arrives as a
`RubiIntermediateResult[HoldComplete[…]]`, the integrals still to be done
written as `Int[…]`:

```wolfram
{steps, result} = Steps[Int[(x^2 + x + 1)/(x^4 + x^3 + x + 1), x],
  RubiPrintInformation -> False];
Cases[steps, RubiIntermediateResult[HoldComplete[e_]] :> HoldForm[e], Infinity]
(* {Int[1/(3*(1 + x)^2) + 2/(3*(1 - x + x^2)), x],
    -1/(3*(1 + x)) + 2/3 ⋆ Int[1/(1 - x + x^2), x],
    -1/(3*(1 + x)) - 4/3 ⋆ Subst[Int[1/(-3 - x^2), x], x, -1 + 2*x],
    -1/(3*(1 + x)) - (4*ArcTan[(1 - 2*x)/Sqrt[3]])/(3*Sqrt[3])} *)
```

The `⋆` is Rubi's `Star`, a coefficient kept in front of an integral for
display. Rubi also defines how `Int` is typeset, so the steps convert to TeX
as integrals — `Convert`TeX`ExpressionToTeX[HoldForm[e]]` gives
`\int \frac{1}{3 (1+x)^2}+\frac{2}{3 \left(1-x+x^2\right)} \, dx` for the
first one — which is what the *IntWithStepsOfTeXForm* notebook builds its
`aligned` environment from.


## Typesetting the steps with MaTeX

[MaTeX] turns that TeX back into graphics by running LaTeX on it and importing
the resulting PDF, and it runs on Woxi as it ships. Its release paclet is a ZIP
archive like Rubi's; it needs `pdflatex` (with the `standalone` class) and
Ghostscript 9.15 or later, which it looks for in the usual macOS locations:

```wolfram
Get["MaTeX-1.7.10/MaTeX.m"];
ConfigureMaTeX[
  "pdfLaTeX" -> StringTrim@RunProcess[{"which", "pdflatex"}, "StandardOutput"],
  "Ghostscript" -> StringTrim@RunProcess[{"which", "gs"}, "StandardOutput"]];
MaTeX["\\int \\frac{1}{x} \\, \\mathrm{d}x = \\log (x)"]
(* -Graphics- *)
```

The result is ordinary graphics, so outside a front end it can be `Export`ed.
Running the *IntWithStepsOfTeXForm* notebook this way — `woxi run` on the
notebook, Rubi and MaTeX unpacked beside it — typesets all four of its
examples.

[MaTeX]: https://github.com/szhorvat/MaTeX


## What does not work yet

Rubi loads unmodified and integrates, but it is not fully supported. On a
30-integral sample, 27 answers are character-for-character what
`wolframscript` gets from the same package, and the other 3 are the same
function written another way. The *IntWithStepsOfTeXForm* pipeline above
reproduces three of its four notebook examples exactly; in the fourth
(`Sqrt[Tan[x]]`) one intermediate sum lists two `Subst` terms the other way
round. What is left:

- **Loading is slow.** About a minute against roughly twenty seconds under
  `wolframscript` (three minutes with the step display), and there is no
  `.mx` cache to make the second run faster. Rubi's progress bar is a
  `Monitor`, which Woxi evaluates without drawing, so the load also reports
  that no front end is available.
- **Two integrals exhaust memory.** `Int[Sin[x]^3*Cos[x]^2, x]` and
  `Int[Sin[x]*Cos[x]^3, x]` run for minutes and are killed;
  `wolframscript` answers both instantly.
- **A few rule paths differ.** `Int[E^x*x, x]` is `-Gamma[2, -x]` rather
  than `E^x*(x - 1)`, and `Int[1/(a + b*Cos[x]), x]` reaches its arctangent
  through a different substitution — same functions, other valid
  antiderivatives, from a different rule firing first.
- **Some products are ordered differently.** `x^2*Sqrt[a^2 - x^2]` prints as
  `Sqrt[a^2 - x^2]*x^2`, and a sum of two `Subst` terms whose integrands are
  `(Sqrt[2] + 2 x)/(-1 - Sqrt[2] x - x^2)` and `(Sqrt[2] - 2 x)/(-1 + Sqrt[2] x - x^2)`
  lists them the other way round. Value-identical, display-only.
- **The rule text recorded next to each step** keeps its `FreeQ` conditions
  and writes them in linear box syntax rather than `DisplayForm`, and the
  rule numbers differ (Woxi numbers 7391 rules where `wolframscript` has
  7300).
