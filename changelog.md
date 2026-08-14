# Changelog

# Unreleased

- Contexts are real: a symbol's name is resolved when its input is *read*,
    against `$Context` and `$ContextPath`, and the symbol is stored under its
    full name. A package's `Begin["`Private`"]` helpers are therefore private
    — two packages can each define a `helper` without colliding — while a name
    declared before it (the `f::usage` idiom) is exported and becomes visible
    once `EndPackage[]` puts the context on `$ContextPath`. Resolution happens
    per input unit, as in the Wolfram Language: a whole package written on one
    line resolves before any of it runs, the same way wolframscript does it.
    - `Context`, `Contexts`, `Names`, `Remove`, `Symbol`, `ToExpression`,
        `Definition`, `DownValues` and `Information` all speak contexts now.
        `Names["P`*"]` lists a package's exports without reaching into
        `P`Private``, and symbols print under their short name wherever that
        reads back as the same symbol and under their full name otherwise.
    - Creating a symbol whose short name already lives in another visible
        context reports `::shdw`, and `Context` reports `::ssle` / `::notfound`
        for arguments that name no symbol.
    - `Context` is `HoldFirst`, so it reports the context of a symbol rather
        than of its value.
    - `Information[sym, "Property"]` returns that one field (`Missing[
        "UnknownProperty", …]` for an unknown one), and a package symbol's
        `FullName` names its own context.
- `PacletDirectoryLoad` and `PacletDirectoryUnload` register the directories
    the paclet manager searches. A registered directory may be a paclet
    itself — a directory with a `PacletInfo.wl` — or a directory collecting
    several of them.
- `Needs["MyPaclet`"]` loads the file a context comes from instead of always
    returning `$Failed`: the `"Kernel"` extensions of the paclets in the
    loaded paclet directories are searched first, then `$Path`. `Get` and
    `FindFile` resolve a context the same way, and a context that no file
    provides reports `Get::noopen` and `Needs::nocont` as wolframscript does.
    `Needs` also validates its arguments now (`Needs::cxt`, `Needs::cxru`,
    `Needs::string`) and skips a context that is already loaded.
- `Get` on a file whose last expression is `Null` returns `Null` rather than
    the internal display-suppression marker.
- Fixes driven by a Wolfram Demonstration on cellular-automaton patch
    reentry, whose interactive widget sets a control's per-choice appearance
    with a small `Grid` and drives a step-counter button's label with a
    `Dynamic`:
    - Woxi Studio: a `Grid`/`TableForm`/`TextGrid` used as a setter's
        per-choice appearance function (rather than a plain string) now
        typesets its cells — joined by a space within a row, by a newline
        between rows — instead of falling through to the choice's raw
        source dump, options and all. `Framed` and `Labeled` wrappers around
        a choice's content are unwrapped the same way. A choice that
        legitimately renders to nothing (a blank `Grid` marking "no flag
        set" among a family of choices that each flip one cell on) now stays
        blank rather than falling back to its source, too.
    - Woxi Studio: a `Dynamic[…]` wrapped around a control's own label (a
        step-counter button framing a live `Dynamic[Row[…]]`) now typesets
        the wrapped content instead of the `Dynamic[…]` call's own source.
- Fixes driven by opening Wolfram's *Introduction to Calculus* ebook (41
    notebooks) in Woxi Studio. Every cell that could not be read or run is
    now read and run; what remains are computations Woxi does not carry out
    as far as Wolfram does.
    - `∫ f ⅆx` and `∫_a^b f ⅆx` are read back as `Integrate[f, x]` and
        `Integrate[f, {x, a, b}]`. The integral sign takes the rest of its
        row as the body, the way the `∑`/`∏` operators already did, and the
        `ⅆx` closing that body names the integration variable. The whole
        "Separable Differential Equations", "Indefinite Integrals" and
        "Average Value of a Function" chapters are written this way and
        every such cell used to fail to parse.
    - `ⅆx` on its own is the differential `DifferentialD[x]`, so the
        differentials the "Linear Approximations" chapter states its error
        estimates in (`ⅆarea == 2 π r ⅆr`, then `sol /. {r -> 50,
        ⅆr -> 0.1}`) stay symbolic and can be replaced. The character is a
        Unicode *letter*, so `ⅆarea` used to read as one symbol name.
    - A typeset `Piecewise` — the `⎧` brace, stored as a `GridBox` of
        value/condition rows — comes back as a `Piecewise[…]` call instead
        of a nested list, so `RevolutionPlot3D` of one has a function to
        sample.
    - A quote inside a cell's string literal stays escaped when the cell is
        read, so a `Plot` legend written as an inline cell
        (`"\!\(\*Cell[\"f[x]\", …]\)"`) stays one string instead of ending
        at its first inner quote.
    - `expr /. lhs -> rhs`, where `lhs` is a literal call rather than a
        pattern, rewrites every subexpression equal to `lhs` structurally.
        It used to fall through to a textual replacement that dropped the
        brackets the substituted expression needs, silently changing the
        result: `π r[t]^2 h[t]/3 /. r[t] -> h[t]/2` — the substitution the
        "Related Rates" chapter makes — came out as `π h[t]^2/12` instead
        of `π h[t]^3/12`, and `r[t]^2 /. r[t] -> q[t] + 1` as
        `1 + q[t]^2` instead of `(1 + q[t])^2`.
    - `c = 4; (* c = 6 *); c` evaluates: several `;` in a row with only
        space or a comment between them separate the same two statements,
        with the empty statement `Null` between them. `;;` still parses as
        `Span`.
    - `DifferenceDelta[Cos[f], …]` — and so `Limit[DifferenceQuotient[
        Cos[x], {x, h}], h -> 0]` — is the cosine's own. The sine of the
        mean was given the half-turn phase shift that only the `Sin` case
        needs (to write its cosine as a sine), which turned the result back
        into the *sine's* difference: the derivative of `Cos` came out as
        `Cos[x]`, and the intermediate quotient was numerically wrong.
    - Woxi Studio: a derivative sets as prime marks (`f′[x]`, `f″[x]`,
        `f⁽⁴⁾[x]`, `f⁽¹˒⁰⁾[x, y]`) in an evaluated cell, the way a notebook
        shows it. `TraditionalForm` already hid the `Derivative` head;
        StandardForm — what a cell's result is typeset with — did not, so
        every derivative of an undefined function read as
        `Derivative[1][f][x]`.

- Fixes driven by a Wolfram Demonstration on a cylindrical cavity resonator,
    which locates a Bessel function's stationary points and interpolates a
    field built on a 2-D grid:
    - `FindRoot[eqn, {var, x0}]` evaluates `eqn` once, with `var` still a
        free symbol, before substituting any numeric trial value. `FindRoot`
        is `HoldAll` (so `var` isn't looked up as an OwnValue before the
        search starts), so `eqn` used to arrive — and stay — with any held
        computation it wrote unevaluated, most commonly a derivative:
        `FindRoot[D[f[x], x] == 0, {x, x0}]`, the standard "extremum of f"
        idiom (and `FindRoot[D[BesselJ[m, r], r] == 0, {r,
        BesselJZero[n, k]}]` specifically), substituted its trial value
        straight into the raw `D[f[x], x]`, landing the number inside `D`'s
        variable slot (`D[f[3.8], 3.8]`) and failing with `D::ivar` on every
        iteration. The multivariate form (`FindRoot[{eqns}, {{x, x0}, …}]`)
        already evaluated its equations up front; the single-variable form
        now does the same.
    - `Interpolation[data]` accepts 2-D data given as a flat list of
        `{x, y, z}` triples — the shape
        `Flatten[Table[Table[{x, y, f[x, y]}, {y, ys}], {x, xs}], 1]` (or the
        equivalent built with `Join`) produces — recovering the grid from the
        distinct x/y coordinates and interpolating it the same way
        `ListInterpolation`'s implicit integer grid already did (tensor-
        product local Lagrange, any order 1–3, `InterpolationOrder ->
        {orderX, orderY}` honoured per axis). This data shape used to reach
        a generic numeric-conversion helper shared with `NDSolve` and fail
        with a message that named the wrong caller
        (`NDSolve: cannot convert {1., 1., 2.} to numeric value"`); that
        helper's error is now caller-agnostic.
    - Woxi Studio: `Control[{…, ControlType -> RadioButtonBar}]` forces the
        row of buttons the way `ControlType -> SetterBar` already did, so a
        `RadioButtonBar` with more choices than the automatic split allows
        keeps its bar instead of silently falling back to a dropdown.
- `$VersionNumber` is a `Real` — the Wolfram Language version Woxi aims to
    be compatible with (`15.`) — instead of the Woxi build's git version as
    a `String`. Scripts gate language features on it
    (`If[$VersionNumber >= 8, …]`), which only works for a number. The Woxi
    build stays available through `$Version`.
- `$InputFileName` names the file `Get` is currently reading, so a file
    pulled in with `Get` can locate its own directory
    (`DirectoryName[$InputFileName]`) instead of the including script's. The
    previous value is restored when the read finishes, so nested and
    sequential `Get`s each see their own file.
- `Get["relative/name.wl"]` resolves against the current directory reported
    by `Directory[]`, so a preceding `SetDirectory` is honoured.
- `ParallelSelect` and `ParallelCases` (new in Wolfram 14.2) are supported.
    Like the other `Parallel*` combinators in Woxi they delegate to the
    sequential implementation — `Select` and `Cases` respectively — so every
    form those accept works, including `ParallelSelect[list, crit, n]` and
    `ParallelCases[expr, patt, levelspec, n]`.
- Fixes driven by a Wolfram Demonstration that draws the shortest distance
    between two skew lines, a `Manipulate` whose `Graphics3D` is built out
    of unbounded objects:
    - `InfiniteLine`, `HalfLine`, `InfinitePlane` and `HalfPlane` are drawn
        in a `Graphics3D`, clipped to the picture's box: a line to the two
        points where it leaves the box, a plane to the cross section it cuts
        out of it. They used to draw nothing at all, so a scene built around
        a pair of lines arrived empty. They are clipped to exactly the range
        the finite contents ask for, so an unbounded object never widens the
        range it was measured against.
    - `Sphere[{p1, p2, …}, r]` is a set of spheres of radius `r`, one per
        centre — how a scene marks several points at once. The list of
        centres failed to parse as a point, leaving a single stray sphere at
        the origin. `Ball` draws in a `Graphics3D` too, the same way it
        already did in a 2-D `Graphics`.
    - `(a ⨯ b) ⨯ c` is `Cross[Cross[a, b], c]`. `Cross` has no `Flat`
        attribute, so parentheses group it like any other operator; the
        chain used to collapse through them into the three-argument
        `Cross[a, b, c]`, which wants vectors of length four and errored on
        3D ones. `((b - a) ⨯ (d - c)) ⨯ (b - a)` is how a Demonstration
        gets a second direction vector inside a plane.
    - `NumberForm[expr, spec]` formats the approximate reals *inside* a
        symbolic expression, so a plane equation shows its coefficients at
        the width asked for: `NumberForm[0.370991 x - 0.927478 y, {4, 3}]`
        reads `0.371 x - 0.927 y`.
    - A term with a negative *real* coefficient prints as a subtraction, the
        same as an integer one: `ToString[x - 0.5 y]` was `x + -0.5 y`.

- Fixes driven by a Wolfram Demonstration that pairs a plot of two sine
    waves and their sum with the sound of that sum:
    - `Play[f, {t, tmin, tmax}, opts…]` accepts its options. Only the
        two-argument form built a sound, so a `Play` carrying
        `SampleRate -> r` stayed an inert expression: no `-Sound-`, `Head`
        of `Play` instead of `Sound`, and nothing to play in the visual
        hosts.
    - `SampleRate -> r` now sets the rate the amplitude function is
        synthesized at, instead of every `Play` being fixed at 8000 Hz.
    - A `Sound` (or `Play`) inside a `Grid`, `Column` or `Row` draws the
        sound box a notebook shows for it — a play button beside the
        waveform — rather than printing the `Play[…]` source into the
        picture.
    - `Style[expr, FontFamily -> "Times"]` picks the face its text is set
        in inside those same layouts. The family was parsed but the layout
        renderers always emitted their own default.

- Fixes driven by a Wolfram Demonstration that quizzes the reader on the
    exact value of `f⁻¹(f(x))` for a trigonometric `f`, offering the
    candidate answers in a pick list drawn inside its own layout:
    - An inverse trigonometric function applied to its own forward function
        at an exact real argument reduces into the inverse's principal
        range: `ArcSin[Sin[2]]` is `Pi - 2`, `ArcSin[Sin[7]]` is
        `7 - 2 Pi`, `ArcCot[Cot[Pi/2]]` is `Pi/2`. The round trip used to
        stay unevaluated, which is precisely the answer the quiz asks for.
    - `\!\(\*boxes\)`, the escape `InputForm` writes for a typeset
        expression, reads back as that expression. Woxi Studio serializes a
        `Manipulate` body through `InputForm` and re-parses it on every
        frame, so a `TraditionalForm[…]` in the body came back as an opaque
        `HoldComplete` and printed as its own source.
    - A `Style` around a `Row`/`Column`/`Grid` sets the layout's text and
        leaves any picture inside it alone. The font directives used to be
        pushed onto the picture too, which hid it behind the `-Graphics-`
        placeholder — so a body written `Text@Style[Row[{…}], 18]` lost its
        plot.
    - `PopupMenu[Dynamic[var], choices]` written inside a `Manipulate` body
        becomes a real pick list for `var` instead of printing as source.
        The choice list is taken with the `With`/`Module`/`Block` scopes the
        body wraps it in, so a list built from the body's own locals still
        resolves, and it is re-resolved whenever the other controls change.

- `woxi repl` prints results the way wolframscript's terminal REPL does:
    a machine-precision real shows the 6 significant figures of OutputForm
    (`3203.60 - 2711.16` is `492.44`, not `492.44000000000005`) and an
    arbitrary-precision real shows its stored precision without the backtick
    marker (`N[Pi, 20]` is `3.1415926535897932385`). Only the printed text
    is rounded — `%` and `Out[]` still hold the full value, and `woxi eval`
    keeps the round-trip InputForm that `wolframscript -code` prints.

- Fixes driven by a Wolfram Demonstration that trisects an angle with a
    cubic curve, labelling its polar plot with the curve's equation and
    logarithmic derivative:
    - `TraditionalForm` sets a derivative with prime marks. `f'` holds as
        `Derivative[1][f]`, and that head was showing through as
        `Derivative(1, f)` instead of `f′`. Orders past three, and the
        multivariate `Derivative[1, 0][f]`, spell the order out as a
        parenthesised superscript.
    - `TraditionalForm` puts a negative exponent under a fraction bar, so
        `1/x` sets as a fraction rather than `x⁻¹`. A product already did
        this for its reciprocal factors; a lone power did not, which left
        `1/Cos[θ/3]^3` reading as `cos(θ/3)⁻³`.

- Fixes driven by a Wolfram Demonstration that highlights the smallest
    triangle among optimally placed points in the unit square, whose
    coordinates are tabulated as exact algebraic numbers:
    - A control bound written `Dynamic[…]` — a counter slider whose end
        follows another control (`{{k, 1}, 1, Dynamic[Binomial[n, 3]], 1}`)
        — is read as the dynamic bound it is. `Dynamic` never evaluates to
        a number, so the control failed to parse, and with it the whole
        `Manipulate`: no widget appeared at all.
    - A notebook cell holding a pasted `InputForm` result stands for the
        expression itself. Its `InterpretationBox` names the display form as
        the meaning, and keeping that wrapper left an inert one-element
        object where an array was expected, so `Dimensions`, `Map` and
        `Part` all saw a single opaque element.
    - `RootReduce` and `MinimalPolynomial` of a rational polynomial in one
        `Root` object stay inside that root's field: the minimal polynomial
        comes from the multiplication matrix there rather than from
        composing the terms pairwise by resultants, which multiplied the
        degrees together (3 · 3 · 3 for a three-term sum over a cubic) and
        then had to factor the result. Minutes per number became
        milliseconds.
    - The resultants those two functions still take are computed over big
        integers, by sampling and interpolation instead of cofactor
        expansion. A 9×9 Sylvester matrix overflowed `i128` — a panic in a
        debug build, a silently wrong polynomial in a release one.
    - `Factor` splits polynomials whose coefficients run into the thousands,
        such as `5184 x^6 - 153 x^4 + 130 x^2 - 9`. The lift target was
        capped where the *reached* modulus needed capping, so these were
        reported as irreducible.
    - `Style[expr, "Label", 12]` applies the explicit size over the named
        style's. Both were emitted, putting two `font-size` attributes on
        one SVG element.

- Fixes driven by a Wolfram Demonstration that draws the powers of a
    modular group as a directed graph:
    - `EdgeShapeFunction -> f` hands the drawing of each edge to `f`, which
        is applied as `f[{pt, …}, edge]` and whose result replaces the
        default line or arrow. The second argument is the edge itself — a
        `DirectedEdge` or `UndirectedEdge`, not a pair of vertices — so a
        `MemberQ` test on it picks the same edges Wolfram picks. Directives
        the function sets stay scoped to its own edge. `EdgeShapeFunction`
        also takes `None` (draw no edge), a shape name such as `"Line"` or
        `"Arrow"`, and the per-edge rule form `{DirectedEdge[u, v] -> f}`.
        The option used to be ignored altogether, so a plot that greys out
        most of its edges came out as a thicket of default arrows.
    - `GraphPlot[…, Method -> "CircularEmbedding"]` lays the vertices out on
        one circle. `Method` names the embedding for `GraphPlot` the way
        `GraphLayout` does for `Graph`, and is now read as such; an explicit
        `GraphLayout` still wins. A graph in several pieces used to be
        packed into a grid of clusters whatever the option said.

- Fixes driven by a Wolfram Demonstration that constructs the centre of a
    circle with compasses alone, each step drawing the circle through two
    points that two earlier circles cross at:
    - A system of polynomial equations is solved by eliminating one
        variable at a time, dividing one equation into another for as long
        as the divisor's leading coefficient is a number and falling back
        to a resultant when it is not. Solving one equation for one
        variable and substituting — what the solver used to do — turns a
        quadratic into a radical that then has to be eliminated all over
        again, and that step dropped solutions: two circles crossing twice
        reported one of the two crossings twice over, and with the inexact
        coordinates a slider produces the whole system came back
        unsolved. Touching circles still report the point they share
        twice, since it counts twice.
    - `Resultant` multiplies its result out, rather than reporting the
        products the Sylvester determinant is assembled from.
    - An equation carrying a root of the unknown — `Sqrt[2 x + 3] == x`,
        `Sqrt[x + 5] - Sqrt[x] == 1` — is solved by putting one root on its
        own, raising both sides to its index, and keeping the answers that
        survive the original equation. Such an equation used to be read as
        if the root were not there, which answered `Sqrt[2 x + 3] == x`
        with `x == 0`. Undoing a square root likewise keeps only the
        answers that survive, so `Sqrt[x] == -1`, which nothing solves,
        no longer reports `x == 1`.

- Fixes driven by a Wolfram Demonstration that draws a diatomic molecule as
    a three-dimensional schematic beside an energy plot:
    - `Scale[g, s]` in a `Graphics3D` scales about the centre of `g`'s
        bounding box, the way the two-dimensional renderer already did.
        Scaling about the origin dragged everything towards it, so two
        nuclei a bond length apart collapsed into one blob.
    - `Scale[g, {sx, sy, sz}]` with unequal factors bends a `Sphere`,
        `Cylinder` or `Cone` into the ellipsoidal shape it asks for. The
        radius used to take one averaged factor, which left a sphere a
        sphere.
    - `EdgeForm[colour]` outlines a face in that colour, and outlines a
        curved primitive's end circles rather than nothing at all —
        `{Opacity[0], EdgeForm[Black], Cylinder[…]}` is how a Demonstration
        draws an unfilled circle in space. Such an outline also stays
        opaque around a transparent face.
    - A `Style[…, size]` inside a plot label is sized in printer's points
        and a `Spacer[n]` is a gap of `n` of them, both scaled to the
        picture. The sizes used to be emitted verbatim into a viewBox
        twenty times larger, which left a styled run invisible and a spacer
        sized off the font instead of the page.
    - `\[InvisiblePrefixScriptBase]` and `\[InvisiblePostfixScriptBase]`
        set no type, and a script hung on one is a prefix script:
        `\!\(\*SuperscriptBox[\(\[InvisiblePrefixScriptBase]\), \(1\)]\)Σ`
        reads as `¹Σ`. The placeholder used to print its own name, and the
        script was read as an exponentiation — so `x^1` evaluated it away.
    - The `\[Raw…]` characters name the ASCII control codes, and the
        non-printing ones set no type in a notebook cell. A caption that
        opens an inline formula with a `\[RawEscape]` used to show the name.

- Fixes driven by a Wolfram Demonstration that solves a pair of coupled
    reaction equations and switches between three views of the result:
    - The picture a cell shows is the one its value *is*, not the last one
        drawn on the way there. A body that builds several plots and then
        picks one — `p1 = Plot[…]; p2 = ContourPlot[…]; Switch[view, 1, p1,
        2, p2]`, the standard Demonstrations "which view?" control — was
        displayed as whichever plot happened to be assigned last, because
        the visual hosts (Playground, Woxi Studio, the Jupyter kernel) read
        back the last graphic captured while evaluating.
    - `ContourStyle` draws the contour lines: `ContourStyle -> {Thick,
        Blue}` gives a thick blue curve instead of the default thin dark
        grey one. The option was read only into the plot's symbolic form,
        so the picture ignored it. It reaches the equation form
        (`ContourPlot[lhs == rhs, …]`) too, and travels with the curve so a
        `Show` merge keeps it.
    - `FrameLabel` and `Epilog` reach the density and contour plots, the
        way they already reached the function plots: the labels are written
        outside the frame (with room reserved for them) and the epilog
        primitives are drawn over the finished picture in data coordinates.
        A stability diagram marking the current parameters with a red
        `Point` came out as a bare unlabelled curve.
    - `ImageSize -> 400 {1, 1}` sizes a plot. A plot holds its arguments,
        so the option value arrived as an unevaluated product and was
        dropped, leaving the default width; it is now evaluated to the
        `{400, 400}` it describes.
- Fixes driven by a Wolfram Demonstration that plots the concentrations of
    a catalysed reaction and breaks them down in an inset pie chart:
    - `PieChart` honours `LabelingFunction -> f`, labelling every wedge with
        `f[value]`. A `Placed[label, position]` result picks the radius the
        text sits at — `"RadialInner"` and `"RadialOuter"` hug the hub or the
        rim, and `"RadialCallout"` puts the label outside the pie on a leader
        line, with the wedges giving up the room the text needs. The option
        used to be ignored, so a pie labelled only through it came out bare.
        A structured label such as `Row[{NumberForm[100 #, 2], "%"}, " "]`
        now typesets the way every other chart label does, so the number
        forms inside it are applied.
    - An axis label written through a typesetting wrapper still prints:
        `AxesLabel -> TraditionalForm /@ {t, y}` used to leave *both* axes
        unlabelled. A list is a label in its own right too, typeset as the
        list itself, and a wide one over an axis near the left edge slides
        right instead of running off the image.
    - Either half of an `Inset` anchor may be symbolic: `{0.8, Center}`
        means four-fifths along the x axis and halfway up the y one.
        An unresolved half used to drop the whole position, which parked
        the inset in the middle of the plot.
    - A notebook's prose renders a script written over a base
        (`OverscriptBox`) — a rate constant over a reaction arrow, or an
        accent such as `OverHat`, which reads as the accented letter. The
        case was missing entirely, so the raw box source was left in the
        cell. The long arrows those constants sit on (`\[LongRightArrow]`,
        `\[DoubleLongLeftRightArrow]` and the rest of the family, plus
        `\[Equilibrium]`) are named characters now, rather than printing as
        their own names.
- Fixes driven by a Wolfram Demonstration whose setter names each of ten
    conformal maps by its formula:
    - `\[WarningSign]` and its neighbours in Wolfram's pictograph block are
        characters like any other. A label warning that a choice is slow used
        to carry the escape through to the widget, so the button read
        `sn(z) (elliptical pairs—\[WarningSign] slow!)`. Added alongside the
        musical accidentals (`\[Sharp]`, `\[Flat]`, `\[Natural]`) and the
        astronomical symbols for the Sun and the planets.
    - A square root in a control label sets as the radical sign over its
        radicand — `√(eᶻ+1)`, not the source of the `Sqrt` head nor of the
        `x^(1/2)` power it normalizes to. Both spellings reach the radical
        now; a compound radicand is parenthesized, a single one is not.
    - Arithmetic joining typeset label pieces recurses into its operands, so
        a quotient of two `Row`s renders as the rows it lays out rather than
        printing their source next to a slash. Operands with nothing to
        typeset stay on the OutputForm path, which already knows this
        arithmetic's precedence and sign conventions.

# 2026-08-06 - 0.3.0

Between 0.2.0 and 0.3.0 the focus moved from growing the function library
to running real notebooks: Woxi Studio executes complete Wolfram
Demonstrations end to end, the plotters draw what wolframscript draws, and
hundreds of conformance and robustness fixes — many found by differential
fuzzing against wolframscript — closed gaps between the two engines.
Windows became a supported platform with prebuilt binaries. The list only
includes the most prominent changes; the Demonstration-driven fixes are
described in detail in the last section.

## Woxi Studio & Wolfram Demonstrations

- Woxi Studio opens and runs complete Wolfram Demonstration notebooks end
    to end — including definition notebooks and notebooks with embedded
    raster photos — and re-instantiates stored `Manipulate` widgets when a
    notebook is reopened. Demonstrations such as Kepler's Second Law,
    Parabolic Mirror, Doyle Spirals, Dedekind Cut, Damped Forced Pendulum
    and Merging Schools of Fish serve as end-to-end tests.
- `Manipulate` covers the Demonstrations control vocabulary: interactive
    `Locator`s (with `LocatorAutoCreate`), `Trigger`, `Button`,
    `ButtonBar`, `SetterBar`, `PopupMenu`, `Checkbox`/`CheckboxBar`,
    `Toggler`, `Opener`/`OpenerBar`, controls nested in `TabView`,
    `PaneSelector` or `Grid` layouts, per-control `ControlType` lists,
    dynamic control bounds, controls depending on other controls,
    `TrackedSymbols`, `Animate` bodies and `AnimationRunning`.
- Typeset notebook input is read back as code: inline box syntax, typeset
    `Sum`/`Product` boxes, the partial-derivative operator, `Part` written
    as a bracketed subscript, the radical prefix operators, and the `×`
    and `÷` characters.

## Plotting & graphics

- List plots grew `Around` error bars, `IntervalMarkers -> "Bands"`,
    `Callout` and `Labeled` wrappers, `PlotMarkers`, `Epilog`, `Filling`
    (including between datasets), `FillingStyle`, `DataRange`,
    `InterpolationOrder`, `Mesh -> Full`, `PlotLayout`, association keys
    as point labels, and `TimeSeries` input; `ComplexListPlot` and
    `ListLinePlot3D` are new.
- `Plot` supports `Background`, `Evaluated`, `EvaluationMonitor` and
    `Infinity` endpoints; automatic ticks match Wolfram's algorithm, tick
    labels switch to scientific notation outside `[10^-5, 10^6)`, frames
    take `FrameLabel` captions and `AxesLabel` sits at the end of its
    axis. Contour and density plots fix the marching-squares case table,
    draw their mesh, and keep their shading through `Show`.
- Graphics primitives and options: polygons with holes (and their
    measures), `Arrowheads`, `Tube`, `CapForm`, `Text` offsets,
    `Translate` and `Scale` in SVG export, `Framed`/`Highlighted`
    rendering, and `AspectRatio` applied to the plotting area rather than
    the whole image.
- Graphics3D draws curved surfaces without facet edges, triangulates
    concave polygons, honours `EdgeForm[]`, `SphericalRegion` and
    `ViewAngle`, and outlines flat faces the way Wolfram does.
    `PolyhedronData` gained face lists with Wolfram's vertex order and
    the icosahedral Archimedean solids.
- Exported SVG embeds the fonts it uses, renders `TableForm`/`MatrixForm`
    and the display-form wrappers as grids, and shows machine reals at
    six significant figures.

## Calculus, algebra & equation solving

- `Integrate` learned the antiderivatives of the error, Fresnel, inverse
    hyperbolic and exponential-integral families, `Sin`/`Cos[x^2]` to
    Fresnel integrals, `Exp[a x^2]` to `Erfi`, `1/Log[x]` to
    `LogIntegral`, trigonometric integrals over a linear term to
    `Si`/`Ci`, `u = x^n` substitution, Euler's log-trig definite
    integrals and `Abs` of a linear argument. `NIntegrate` handles
    endpoint singularities, multi-segment ranges and iterated
    multi-dimensional integrals (`NSum`/`NProduct` likewise).
- Differentiation covers `Piecewise`, `HeavisideTheta` (to `DiracDelta`,
    with its scaling law), `KroneckerDelta`, `Floor`/`Ceiling`, argument
    derivatives of the Bessel and Hankel functions, `PolyLog`,
    `ExpIntegralE`, the Airy primes and the incomplete elliptic
    integrals, plus the fractional `CaputoD`.
- `Series` expands `Gamma` at its pole, `Zeta` at 1 (Laurent) and
    algebraic expressions at `Infinity`; limits are decided from the
    leading exponent at infinity, and `MaxLimit`/`MinLimit` handle
    bounded trigonometric oscillations.
- Solving: `x^n == c` in radicals, polynomials with machine-real
    coefficients, systems over a modulus, `Modulus` and `MaxRoots`
    options, `NSolve` domain argument, `NSolveValues`, `SolveValues`
    result shaping, and `Reduce` over higher-degree polynomial
    inequalities. `DSolve` handles non-constant forcing and orders its
    fundamental pairs; `NDSolve` solves systems and constrained ODEs;
    `RSolve` solves the logistic map at `r = 4` and golden-ratio
    recurrences in the Fibonacci/Lucas basis.
- Optimization: `LinearProgramming` (exact simplex solver with variable
    bounds), constrained `FindMinimum`/`FindMaximum`/`NArgMin`/`NArgMax`,
    the domain argument, `NonlinearModelFit`, constrained and weighted
    `Fit`, and the minimum-norm fit for rank-deficient designs.
- Structural algebra: `RootReduce` and `RootApproximant`, `Root` objects
    treated as the numbers they are, polynomial reduction modulo
    polynomials, hyperbolic support in `TrigReduce`/`TrigFactor`, and
    many canonical-ordering fixes so sums, products and radicals print
    exactly as wolframscript prints them.

## Special functions, statistics & number theory

- Symbolic reductions for the hypergeometric families (`0F1`, `1F1`,
    `HypergeometricU`, `HypergeometricPFQ` Bessel forms), `PolyLog`,
    `LerchPhi`, the incomplete `Beta` and `Gamma` functions,
    `GegenbauerC[n, 1/2, x]`, parity of `Gudermannian`, `Haversine`,
    `Sinc`, `Erf` and the Fresnel functions, hyperbolic
    imaginary-period shifts, and exact values of the Carlson elliptic
    integrals. Factorial Taylor sums fold to `Sin`/`Cos`/`Sinh`/`Cosh`,
    `Sum[1/n^s]` gives `Zeta[s]`, and sums of `HarmonicNumber` give
    hyperharmonic closed forms.
- Numeric evaluation of the Coulomb wavefunctions at nonzero `eta`,
    `CarlsonRF` at complex arguments, two-argument `Erf`, `PolyLog` with
    real order, and inverse trigonometric functions at complex floats.
- Astronomy: `SunPosition`, `MoonPosition`, `MoonPhase`, eclipses and
    related functions compute from platform-independent ephemerides, as
    an observer on the ground sees them.
- Distributions: `SkewNormalDistribution`; `CDF`/`Quantile`/`Median` for
    the discrete families (`NegativeBinomial`, `Pascal`, `BetaBinomial`,
    `Zipf`, `Benford`, …); skewness, kurtosis, raw moments,
    characteristic functions and MGFs for many more; moments of
    `TruncatedDistribution` and `CensoredDistribution`; `RandomVariate`
    sampling for `BinomialDistribution` and hand-written distributions;
    and parameter-range validation instead of silent nonsense.

## Lists, arrays & structured data

- `SparseArray` behaves like the array it stores: arithmetic stays
    sparse, `Part` reaches into higher ranks, the structural list
    operations densify transparently, and the array predicates and grid
    queries see through it.
- `Dataset` answers queries through the `Query` engine instead of a few
    fixed shapes, and the statistics functions see through it.
    `TimeSeries` can be transformed, combined, resampled, windowed and
    rescaled; `MovingMap` supports padded and time-windowed forms.
- List operations rounded out: `Periodic`/`Reflected`/`Cyclic` padding
    (with nested padding arrays), two-dimensional `ListConvolve` and
    `ListCorrelate` with generalized operations, `MapAt` with `All` and
    `Span`, ordering functions in `SortBy`/`KeySort`, nested `GroupBy`
    classifiers, rule-based `StringSplit`/`SequenceSplit`, and the
    structured matrices (`CauchyMatrix`, `BlockDiagonalMatrix`,
    `CompanionMatrix`) plus `Symmetrize`/`SymmetrizedArray` and
    `TensorExpand`.

## Strings, dates, import & export

- XML imports to symbolic XML (namespaces included) and writes back out
    the way wolframscript does; CSV fixes cell quoting, empty fields and
    header inference; JSON export fails cleanly on unrepresentable
    values; `URLBuild`, `URLQueryEncode`/`URLQueryDecode`, `TextCases`
    for words and sentences, `CharacterName` and `CharacterNormalize`
    are new.
- Dates: `MaxDate`/`MinDate`, ISO and week-based date elements, exact
    rational `DateDifference` sub-day units, month rollover in
    `AbsoluteTime`, and `DateObject` granularity truncation.
- Output forms: `TeXForm` sets matrices as LaTeX arrays and matches
    wolframscript, `CForm`/`FortranForm` fix their operators and
    numbers, and the `NumberForm` display family shares one
    option-aware renderer (`NumberSigns`, `ExponentFunction`,
    `NumberPadding`, decimal-point alignment, `BaseForm` inner values).
- Patterns: `Verbatim` heads, `Longest`/`Shortest`/
    `OrderlessPatternSequence`, `Overlaps -> All` reporting every match
    at every start, string-pattern back-references, and implicitly named
    blanks in `StringSplit`.

## Images & audio

- Image processing: `ImagePad`, `ImageCrop`, `ImageFilter`,
    `MeanFilter`, `ImageDifference`, `ImageMeasurements`,
    `MorphologicalComponents`, `ComponentMeasurements`,
    `DeleteSmallComponents` and `MaxDetect`/`MinDetect` on matrices;
    `Blur` is the `GaussianFilter` it claims to be, `Sharpen` sharpens
    by Wolfram's amount, and `EdgeDetect` runs on the gradient Woxi
    already had.
- Audio: the accessors, `AudioNormalize`/`AudioReverse`/`AudioPad`, and
    `ListPlay` with playback in the Playground and Woxi Studio.

## Robustness & conformance

- Several rounds of differential fuzzing against wolframscript fixed
    dozens of silent divergences in arithmetic, ordering, `Simplify`,
    `Together` and friends.
- Edge cases that used to abort the evaluator or panic now return the
    message Wolfram reports: `FactorInteger` near `2^64`, degenerate
    random specifications, empty-list arguments, invalid permutation
    lists, out-of-range arguments and bad string positions across many
    heads.
- Evaluation-semantics fixes: the builtin attribute table, `Switch`
    pattern evaluation, `Return` no longer escaping `Table`/`Map`/
    `Select`, `Sequence` through postfix application, `Unevaluated`
    wrappers, chained `ReplaceAll`, `Condition` precedence, postfix `++`
    binding, and a parser call-limit fix for deeply nested brackets.
- `Enclose` and the `Confirm` family, `Success` and the `Failure`
    object family, and `$MessageList` (preserved around `Quiet`) are
    implemented.

## Platform, performance & tooling

- Windows is a supported platform: the unit tests pass and run in CI,
    and the nightly builds produce Windows binaries of both `woxi` and
    Woxi Studio.
- Prebuilt binaries for Linux, macOS and Windows are attached to every
    GitHub release, with a checksum file.
- The browser (WASM) build supports `Export`, including PNG/JPEG plot
    export via host rasterization.
- CI checks formatting and clippy (the codebase is clippy-clean outside
    three allowed lints), debug builds use `opt-level = 1`, and the
    crate metadata was fixed for crates.io publishing.

## Demonstration-driven fixes in detail

- Fixes driven by a Wolfram Demonstration that approximates derivatives on
    Chebyshev–Gauss–Lobatto points:
    - A pattern variable repeated on the left of a definition constrains the
        arguments to be equal: `f[i_, i_] := …` no longer matches `f[1, 2]`.
        The repeat used to be ignored, so a definition pair spelling out a
        matrix's diagonal and off-diagonal entries collapsed onto whichever
        rule came last — and a diagonal formula applied off the diagonal
        divided by zero. The constraint holds for a repeat inside a list
        pattern (`f[{a_, a_}]`) too, and alongside head constraints and
        `?test`s on the repeated slots.
    - `DownValues` and `Definition` print a nested argument pattern as it was
        written: `f[g[x_]] := x` used to read back as `f[__sp0_]`, exposing
        the placeholder such a slot is lowered to.
- Fixes driven by a Wolfram Demonstration that draws a rational cyclic
    polygon inside its circumcircle:
    - `Sphere` and `Ball` draw in a two-dimensional `Graphics`: in the plane
        a sphere is the circle bounding it and a ball the filled disk. Both
        used to be dropped, and since `Circumsphere` returns a `Sphere`
        whatever the dimension, a picture built around a circumcircle came
        out with no circle at all. A list of centres draws one per point,
        and `Ball[2]` is the unit disk at the origin.
    - A `Style` around a label paints its `Background` behind the text —
        `Style[Text[…], 12, Background -> White]` is what keeps a distance
        label readable over the line it sits on. Only a `Background` written
        on the `Text` itself was drawn before.
    - A square factor too large to reach by trial division comes out of a
        radical: `Sqrt[100003^2 * 115]` is `100003 Sqrt[115]`. `Sqrt` and the
        `c Sqrt[r]` merge used to look for square factors to different
        depths, so each rewrote what the other produced and halving such a
        coefficient — the `Mean` of two points carrying a radical — never
        returned.
- Fixes driven by the plane-geometry Wolfram Demonstrations that stack a
    numeric readout over a drawing:
    - A `Pane[…]` grid cell shows what it holds. `Pane` only reserves an
        area for its content, so a body laid out as
        `Grid[{{Pane[…]}, {Graphics[…]}}]` used to print the whole
        `Pane[…]` call as source text in place of the readout — and
        stretch the grid to the width of that source. `Item[…]` and
        `Text[…]` cells are peeled the same way, through the one helper
        every display pass already shares.
    - `Dividers -> All` rules every position of a grid, the outer edges
        included, so the grid is drawn as a closed box. Only the
        boundaries *between* cells were ruled before, which is what
        `Dividers -> Center` means; the two now differ, and either can be
        set per direction as `Dividers -> {colspec, rowspec}`.
    - `Spacings -> {{i -> s, …}, …}` sets the gap at individual column
        positions — position `i` is the gap to the left of column `i`, and
        position `ncols + 1` the margin after the last column. A
        list-valued horizontal spec used to be dropped, so a readout that
        groups its columns with wide and tight gaps came out evenly and
        far too widely spaced. A plain `{s1, s2, …}` names the positions
        in order, and positions left out keep the default gap.
- Manipulate improvements (driven by the "Recursive Exercises" Wolfram
    Demonstrations, which nest circles inside circles):
    - The leading assignments a body makes before anything else are
        evaluated with the control variables at their initial values, the
        way Wolfram evaluates the body before laying the controls out.
        A body opening with `u = {…, ss[-1., 1., n, dc]}`, where `ss`
        recurses down to a literal-`1` base case, used to recurse on a
        symbolic depth that never reached the base case, so extraction
        never returned and no widget appeared at all.
    - A choice list built from another control's variable
        (`Range[1, If[flat, 3, 6], 1]`) is re-resolved against the live
        bindings, so a level setter offering six levels flat narrows to
        three in 3D. A selected value the narrowed list no longer offers
        falls back to the last one it does, and the body is rendered again
        for it. Previously the list was fixed at build time.
    - `ControlType -> Slider` over a choice list renders a slider that
        steps through the choices, matching Wolfram; a twenty-entry
        colour-scheme list used to become a dropdown.
- Fixes driven by a Wolfram Demonstration that runs a cellular automaton
    over a dendrite and compares it with the linear case:
    - A layout that holds pictures — `Grid[{{plot1, plot2}, …}]`, a
        `Column` of them, a `Pane` around either — is composed into the one
        picture a notebook shows. The visual hosts (Playground, Woxi Studio)
        only ever reported the *last* graphic drawn while the layout was
        evaluated, so a Manipulate body that arranges several plots in a
        grid displayed a single one of them and dropped the rest.
    - A `Grid`'s columns line up when its rows hold different things: the
        caption under a picture is centred on that picture instead of
        being packed against the left edge of the grid.
    - `LayeredGraphPlot` and `TreePlot` draw the layered embedding they
        name, rather than the circular one every graph got. Vertices are
        placed by distance from a root, and the second argument (`Left`,
        `Right`, `Top`, `Bottom`) says which edge the roots go on.
    - `DirectedEdges -> False` draws a graph's edges as plain lines
        instead of arrows, and `ImageSize` sizes a graph plot — a
        Demonstration asking for a wide, short strip used to get the
        360-point default square.
    - `ArrayPlot` draws its `FrameLabel`, on any of the four edges, with
        the left and right labels rotated onto theirs.
- Fixes driven by the "Selectivity in a Semibatch Reactor" Wolfram
    Demonstration:
    - `Text` inside a `Graphics3D` scene is drawn — a labelled 3D
        schematic used to arrive with no lettering at all. It honours the
        `Style` size and colour, typesets `Subscript`/`Superscript`, and
        respects the alignment offset of `Text[expr, pos, offset]`.
    - `Scaled[{sx, sy}]` places a `Text` or `Inset` by fraction of the
        plot range, in `Graphics` and in a plot's `Epilog` alike. The
        fractions used to be unreadable, dropping every such label onto
        the origin.
    - `Inset[Graphics3D[…], pos]` embeds a three-dimensional picture in a
        flat one at its own size, instead of printing `-Graphics3D-`.
    - `Framed[…]` around a label draws its box — a border, and a panel
        when a `Background` is given — rather than printing its own
        source over the picture. `FrameStyle -> None` keeps the panel and
        drops the border.
    - `Subscript`/`Superscript` in a `Text` label typeset as scripts
        instead of falling through to the two-line `OutputForm` box.
    - `Graphics` accepts `Prolog` and `Epilog`, drawn under and over its
        content and taking no part in the plot range.
    - A `Manipulate` control panel written as a `Grid` drops its
        `SpanFromLeft` / `SpanFromAbove` cell markers; they used to
        survive as display elements and appear as literal text under the
        widget, one row per marker.
    - A control label computed with `Row[Flatten[{…, Riffle[…], …}]]` is
        evaluated before it is typeset, so the button shows the caption
        rather than the source of the computation.
- Fixes driven by the "Illustrative Performance Characteristics of Modern
    Motorcycles" Wolfram Demonstration:
    - `Piecewise` holds its pieces, so the value of a piece whose condition
        is `False` is never evaluated. This is what makes the construct a
        guard: `Piecewise[{{f[w], 0 <= w <= wmax}}, 0]` no longer calls `f`
        out of range, and the messages that call emitted are gone.
    - `FindRoot` falls back to a difference quotient when the symbolic
        derivative does not reduce to a number at the current iterate —
        differentiating a non-smooth function leaves `Derivative[1, 0][Max][…]`
        standing. It used to give up with `FindRoot::nlnum` and return
        unevaluated. Messages emitted by the discarded attempt (`D::ivar`
        and friends) no longer reach the user.
    - A `ControlType -> …` given to the `Manipulate` itself types every
        control that does not pick one, and a list value assigns one type per
        control in spec order. The Demonstrations idiom — controls laid out in
        a `Grid[…]`, typed once for the whole panel — now renders the dropdowns
        the author asked for instead of over-wide SetterBars.
- Fixes driven by a Wolfram Demonstration that draws a complex function as
    a colored vector field:
    - `ColorData[name, "ColorFunction"]` gives the gradient's color
        function, the same object `ColorData[name]` gives on its own. It
        used to be reported as unimplemented, so every primitive colored
        through it fell back to black. `ColorData[name, "Range"]` reads
        the parameter interval alongside it.
    - `Graphics[…, ImagePadding -> …]` honours the padding (it was only
        read by the plot functions), so the drawing area sits where the
        notebook puts it instead of at the automatic frame margins.
    - A named `Arrowheads` size is a fraction of the plot width and
        nothing else. It used to be capped at 45% of the arrow carrying
        it, which shrank the heads of a dense vector field — where each
        arrow is barely longer than its head — to unreadable specks.
- Fixes driven by the "Mass with a Spring and a Rubber Band" Wolfram
    Demonstration:
    - Substituting an `NDSolve` solution into a derivative
        (`y'[t] /. sol`) evaluates numerically instead of echoing a
        `Derivative[…]` expression, so a phase portrait of a numeric
        solution can be plotted.
    - A plot nested in a `Row` inside a `Column` keeps its own coordinate
        space and is drawn (it used to render blank).
    - `Frame -> True` labels its frame ticks even with `Axes -> False`.
    - `AspectRatio` together with `ImagePadding` sizes the image so the
        plot area spans the full width, instead of shrinking it into a
        canvas sized by the default ratio.
    - Labels written as a function of the plot variable typeset the way
        Wolfram draws them — `AxesLabel -> {t, y[t]}` reads `y(t)` and a
        derivative reads `y′(t)`, where both used to be dropped — and a
        `Manipulate` control labelled `Style["y", Italic]'` shows `y′`.
- Symbol names may contain `$` anywhere (`a$b`, `signal$1`), matching Wolfram.
- `PlotRange` bounds given in reversed order (e.g. `{3, -3}`) normalize to
    the same plot as the sorted form, matching Wolfram.
- Manipulate improvements (driven by the "Oscilloscope with Two Signal
    Inputs" Wolfram Demonstration):
    - `Style[…]`, bare-string, and `Delimiter` arguments are treated as
        static annotation rows between controls (no more spurious
        `Manipulate::vsform` messages) and render as headings/separators
        in Woxi Studio and the Playground.
    - Compound control variables such as `Subscript[signal, 1]` work: they
        are bound through synthesized symbols and keep a typeset label
        (`signal₁`).
    - Control specs may carry trailing options (`ControlType -> PopupMenu`,
        `ImageSize -> Tiny`, …); `ControlType -> PopupMenu` always renders
        a dropdown.
    - A `Manipulate` whose body is an `Animate[…]` renders as one combined
        widget, `AnimationRunning -> False` builds the widget paused, and
        an `Infinity` animation bound gets a finite looping window.
- Woxi Studio re-instantiates stored `Manipulate` widgets when opening a
    notebook (instead of showing the saved `DynamicModuleBox[…]` text dump).
- Manipulate improvements (driven by the "Center of Mass of a Polygon"
    Wolfram Demonstration):
    - `Locator` controls are interactive: a single point binds as a 2D
        slider, a point list becomes a per-point X/Y control (with
        add/remove when `LocatorAutoCreate -> True`, and a multi-handle
        drag pad in the Playground), instead of freezing the variable at
        its initial value.
    - Discrete choices whose rule label is a graphic (`"+" -> myIcon[2]`)
        render the icon in the SetterBar instead of dumping the label's
        InputForm.
    - A bare control-type shorthand in the range position
        (`{{p, init}, Locator}`) is no longer misread as a `Dynamic[…]`
        range.
- InputForm keeps required parentheses around loose operands of `@@`, `/@`,
    `@@@`, and `.` (`Plus @@ (x*y)/2`, `a . (b - c)`); previously the
    printed form re-parsed to a different expression, silently corrupting
    re-evaluated Manipulate bodies.
- `PlotLabel` works on plain `Graphics[…]` (rendered as a centered title),
    and labels may be arbitrary expressions such as
    `Row[{"center of mass: ", CM}]` on all plot types.
- An operator that binds tighter than `Times` now reaches into an adjacent
    implicit product: `2 Times @@ {3, 4}` is `2 (Times @@ {3, 4})` and
    `a b . c` is `a (b . c)`, matching Wolfram. Previously the whole product
    became one operand, so `lcm^n Times @@ Table[…]` applied the wrong head.
- Manipulate improvements (driven by the "Descartes's Rule of Signs"
    Wolfram Demonstration):
    - `Setter`, `Toggler`, `CheckboxBar`, `Opener` and `OpenerBar` are
        recognised as control types. A `Setter` spec used to be read as a
        slider bound, and since one unparsable spec aborts the whole
        extraction, the entire Manipulate fell back to a text echo.
    - `{v, domain, ControlType -> None}` starts `v` at the first choice of
        its domain rather than binding the choice list itself.
    - `TrackedSymbols :> {…}` limits which controls re-run the body; the
        others move without re-rendering, as in Wolfram.
- Display of `Text[content]` in the visual hosts (Woxi Studio, the
    Playground) shows the content, so a `Text@Pane[Column[{…}]]` body
    renders the whole column instead of only the picture inside it.
- `TraditionalForm` inside a layout is typeset in conventional notation
    rather than being stripped to StandardForm markup: `Equal` reads `=`
    (`≤`, `≥`, `≠` likewise), a `Row` of strings and `Style[…]`s is set as
    the text it displays, and a term with a negative coefficient carries
    its sign on the operator (`a - 130 x³`, not `a + -130 x³`).

# 2026-07-16 - 0.2.0

Between 0.1.0 and 0.2.0 Woxi grew from a minimal interpreter into a broad
computer algebra system covering a large subset of the Wolfram Language.
The list only includes the most prominent additions rather than every function.

## Calculus, algebra & equation solving

- Differentiation and integration: `D`, `Dt`, `Derivative` (including `f'[x]`
    prime notation, multi-index and pure-function derivatives), `Integrate`
    (trigonometric, Gaussian, u-substitution, multivariate/iterated, definite
    integrals), `NIntegrate` (including infinite bounds), `Grad`, `Div`, `Curl`,
    `Laplacian`, `Wronskian`, `ArcLength`, `FrenetSerretSystem`.
- Limits and series: `Limit` (directional, at infinity, finite points),
    `DiscreteLimit`, `MaxLimit`, `MinLimit`, `Series` (Taylor, Laurent, Puiseux,
    fractional powers, expansion at infinity), `SeriesData`, `Normal`, `O`,
    `Residue`, `PadeApproximant`, `ComposeSeries`, `InverseSeries`,
    `SeriesCoefficient`, `Asymptotic*` and `AsymptoticIntegrate`.
- Sums and products: symbolic `Sum` and `Product`, including geometric,
    exponential, alternating and p-series, harmonic-number and zeta closed forms,
    telescoping rational sums, multi-dimensional ranges, `NSum`, `NProduct`,
    `SumConvergence`, infinite rational products, and `GeneratingFunction` /
    `ExponentialGeneratingFunction`.
- Equation solving and optimization: `Solve`, `Reduce`, `FindInstance`,
    `FindRoot`, `Eliminate`, `SolveAlways`, `Roots`, `Root`, `RootSum`,
    `RSolve`, `RSolveValue`, `RecurrenceTable`, `FindLinearRecurrence`,
    `DSolve`, `DSolveValue`, `NDSolve`, `NDSolveValue`, `Minimize`, `Maximize`,
    `NMinimize`, `NMaximize`, `FindMinimum`, `FindMaximum`, `ArgMin`, `ArgMax`,
    `FunctionDomain`, `FunctionRange`.
- Simplification and manipulation: `Simplify`, `FullSimplify`, `Refine`,
    `Assuming`, `ConditionalExpression`, `Factor`, `Expand`, `ExpandAll`,
    `Together`, `Apart` (over irreducible quadratics and repeated factors),
    `ApartSquareFree`, `Cancel`, `Collect`, `PowerExpand`, `FunctionExpand`,
    `ComplexExpand`, `PiecewiseExpand`, `TrigExpand`, `TrigReduce`, `TrigFactor`,
    `TrigToExp`, `ExpToTrig`, `Variables`.

## Special functions

- Bessel family: `BesselJ`, `BesselY`, `BesselI`, `BesselK`,
    `SphericalBesselJ`/`SphericalBesselY`, `SphericalHankelH1`/`SphericalHankelH2`,
    `HankelH1`/`HankelH2`, `BesselJZero`, `KelvinBer`/`KelvinBei`,
    `StruveH`/`StruveL`, `AngerJ`, `WeberE`.
- Elliptic and Jacobi functions: `EllipticK`, `EllipticE`, `EllipticF`,
    `EllipticPi`, `EllipticTheta`, `EllipticNomeQ`, `JacobiSN`/`JacobiCN`/`JacobiDN`,
    `JacobiAmplitude`, `JacobiEpsilon`, `JacobiZeta`, all twelve inverse Jacobi
    functions (`InverseJacobiSN`, `InverseJacobiCN`, …), `WeierstrassP`,
    `WeierstrassInvariants`, `WeierstrassHalfPeriods`, the Neville theta
    functions, `ModularLambda`, `KleinInvariantJ`, `ArithmeticGeometricMean`.
- Gamma, zeta and related: `Gamma` (incomplete and regularized), `LogGamma`,
    `PolyGamma`, `Beta`/`BetaRegularized`, `Pochhammer`, `FactorialPower`,
    `BarnesG`, `LogBarnesG`, `Hyperfactorial`, `Zeta`, `HurwitzZeta`, `PrimeZetaP`,
    `RiemannR`, `RiemannSiegelZ`/`RiemannSiegelTheta`, `StieltjesGamma`, `LerchPhi`,
    `PolyLog`, `DirichletEta`/`DirichletBeta`/`DirichletLambda`/`DirichletL`.
- Hypergeometric functions: `Hypergeometric0F1`, `Hypergeometric1F1`,
    `Hypergeometric2F1`, `HypergeometricPFQ`, `HypergeometricU`, `MeijerG`,
    the Appell functions `AppellF1`–`F4`, and their regularized variants.
- Error, exponential-integral and Airy functions: `Erf`, `Erfc`, `Erfi`,
    `InverseErf`/`InverseErfc`, `FresnelS`/`FresnelC`/`FresnelF`/`FresnelG`,
    `ExpIntegralE`, `LogIntegral`, `SinIntegral`/`CosIntegral`,
    `ExpIntegralEi`, `SinhIntegral`/`CoshIntegral`, `DawsonF`, `OwenT`,
    `AiryAi`/`AiryBi` (and their derivatives `AiryAiPrime`/`AiryBiPrime`).
- Orthogonal polynomials and misc: `LegendreP`/`LegendreQ`, `HermiteH`,
    `LaguerreL`, `GegenbauerC`, `ChebyshevT`/`ChebyshevU`, `JacobiP`,
    `SphericalHarmonicY`, `ZernikeR`, `ClebschGordan`, `ThreeJSymbol`,
    `SixJSymbol`, `WignerD`, `MittagLefflerE`, `ChampernowneNumber`, `ThueMorse`,
    `RudinShapiro`, and the mathematical constants (`EulerGamma`, `Catalan`,
    `Glaisher`, `Khinchin`, `GoldenRatio`) at arbitrary precision.

## Statistics & probability

- Over 60 distributions with `PDF`, `CDF`, `Mean`, `Variance`,
    `StandardDeviation`, `Quantile`, `Moment`, `CharacteristicFunction`,
    `HazardFunction` and `SurvivalFunction` support, including
    `NormalDistribution`, `LogNormalDistribution`, `MultinormalDistribution`,
    `StudentTDistribution`, `ChiDistribution`/`ChiSquareDistribution`,
    `GammaDistribution`/`InverseGammaDistribution`,
    `BetaDistribution`/`BetaPrimeDistribution`/`BetaBinomialDistribution`,
    `CauchyDistribution`, `LaplaceDistribution`, `LogisticDistribution`,
    `WeibullDistribution`, `FrechetDistribution`,
    `ExtremeValueDistribution`/`GumbelDistribution`, `RayleighDistribution`,
    `MaxwellDistribution`, `ParetoDistribution`, `PoissonDistribution`,
    `BinomialDistribution`/`NegativeBinomialDistribution`, `GeometricDistribution`,
    `HypergeometricDistribution`, `ZipfDistribution`, `SkellamDistribution`,
    `PERTDistribution`, `DagumDistribution`, `RiceDistribution`,
    `InverseGaussianDistribution`, `MoyalDistribution`, `StableDistribution`,
    `VonMisesDistribution`, `HoytDistribution`, `NakagamiDistribution`,
    `LogLogisticDistribution`, `LogSeriesDistribution`, `MeixnerDistribution`,
    `TukeyLambdaDistribution`, `TsallisQGaussianDistribution`, `WakebyDistribution`,
    `SinghMaddalaDistribution`, `BenktanderWeibullDistribution`,
    `BenfordDistribution`, `PoissonConsulDistribution`,
    `CompoundPoissonDistribution`, `CoxianDistribution`,
    `HyperexponentialDistribution`, `HotellingTSquareDistribution`,
    `DirichletDistribution`, `WishartMatrixDistribution`,
    `NegativeMultinomialDistribution`, `HistogramDistribution` and
    many more, plus reliability distributions (`FailureDistribution`,
    `StandbyDistribution`, `FirstPassageTimeDistribution`) and meta-distributions
    (`TransformedDistribution`, `ProductDistribution`, `CensoredDistribution`,
    `EmpiricalDistribution`, `MixtureDistribution`, `SliceDistribution`,
    `QuantityDistribution`).
- Random processes: `WienerProcess`, `GeometricBrownianMotionProcess`,
    `OrnsteinUhlenbeckProcess`, `BrownianBridgeProcess`, `PoissonProcess`,
    `DiscreteMarkovProcess` and the Bernoulli/Binomial/WhiteNoise processes, with
    time slices, `CovarianceFunction`, `CorrelationFunction` and
    `AbsoluteCorrelationFunction`; plus `StateSpaceModel`, `ObservabilityMatrix`
    and `ControllabilityMatrix` for linear systems.
- Descriptive statistics: `Mean`, `Median`, `Commonest`, `Quantile`,
    `Quartiles`, `InterquartileRange`, `Variance`, `StandardDeviation`,
    `GeometricMean`, `HarmonicMean`, `ContraharmonicMean`, `RootMeanSquare`,
    `TrimmedMean`, `WinsorizedMean`, `Skewness`, `Kurtosis`, `CentralMoment`,
    `Cumulant`, `Covariance`, `Correlation`, `MeanDeviation`, `Standardize`,
    `MovingAverage`, `ExponentialMovingAverage`, `PrincipalComponents`.
- Fitting and inference: `Fit`, `LinearModelFit`, `FindFit`,
    `FindDistributionParameters`, `Expectation`, `Probability`, `LogLikelihood`,
    correlation and dissimilarity measures (`SpearmanRho`, `GoodmanKruskalGamma`,
    `HoeffdingD`, `BlomqvistBeta`, plus a family of distance functions), and
    random sampling via `RandomVariate`, `RandomChoice`, `RandomSample`,
    `RandomReal`, `RandomInteger`, `RandomComplex`, `RandomPrime`.

## Number theory

- Primes and factoring: `PrimeQ` (BigInteger Miller–Rabin), `NextPrime`,
    `PrimePi`, `PrimeOmega`, `PrimeNu`, `FactorInteger` (incl. Gaussian
    integers), `Divisors`, `DivisorSigma`, `DivisorSum`, `EulerPhi`,
    `CarmichaelLambda`, `MoebiusMu`, `PerfectNumber`/`PerfectNumberQ`,
    `MersennePrimeExponentQ`.
- Modular arithmetic and symbols: `Mod`, `PowerMod`/`PowerModList`,
    `ModularInverse`, `MultiplicativeOrder`, `PrimitiveRoot`, `ChineseRemainder`,
    `JacobiSymbol`, `KroneckerSymbol`, `CoprimeQ`.
- Integer sequences and digits: `Fibonacci`, `LucasL`, `CatalanNumber`,
    `BernoulliB`, `EulerE`, `StirlingS1`/`StirlingS2`, `BellB`,
    `PartitionsP`/`PartitionsQ`,
    `IntegerPartitions`, `Subfactorial`, `HarmonicNumber` (and hyper/multiple
    variants), `IntegerDigits`, `DigitCount`, `DigitSum`, `FromDigits`,
    `RealDigits` (arbitrary bases, repeating decimals), `ContinuedFraction`,
    `FareySequence`, `MinkowskiQuestionMark`, `RomanNumeral`, `IntegerName`.

## Polynomials

- `PolynomialGCD`/`PolynomialLCM`, `PolynomialQuotient`/`PolynomialRemainder`,
    `PolynomialExtendedGCD`, `PolynomialReduce`, `Resultant`, `Subresultants`,
    `Discriminant`, `GroebnerBasis`, `Cyclotomic`, `FactorList`,
    `FactorSquareFree`, `FactorTerms`, `Decompose`, `MonomialList`,
    `CoefficientList`/`CoefficientRules`/`FromCoefficientRules`,
    `InterpolatingPolynomial`, `CharacteristicPolynomial`, `HornerForm`,
    `SymmetricPolynomial`, `PowerSymmetricPolynomial`, `SymmetricReduction`,
    `SubresultantPolynomials`/`SubresultantPolynomialRemainders`, `ToRadicals`,
    `NumberFieldDiscriminant`, `AlgebraicNumber` norm/trace/`AlgebraicUnitQ`, and
    modular polynomial arithmetic over GF(p).

## Linear algebra & tensors

- Decompositions and solvers: `LinearSolve`, `LeastSquares`, `RowReduce`,
    `Inverse`, `PseudoInverse`, `Det`, `PfaffianDet`, `MatrixRank`, `NullSpace`,
    `Eigenvalues`, `Eigenvectors`, `Eigensystem`, `LUDecomposition`,
    `QRDecomposition`, `CholeskyDecomposition`, `LDLDecomposition`,
    `JordanDecomposition`, `SchurDecomposition`, `HermiteDecomposition`,
    `SmithDecomposition`, `FrobeniusReduce` (rational canonical form),
    `SingularValueList`, `Orthogonalize`, `LatticeReduce`, and `Modulus`-option
    solvers over GF(p).
- Matrix functions and constructors: `MatrixPower`, `MatrixExp`, `MatrixLog`,
    `MatrixFunction`, `DrazinInverse`, `Adjugate`, `RankDecomposition`,
    `LyapunovSolve`/`DiscreteLyapunovSolve`, `IdentityMatrix`, `DiagonalMatrix`,
    `HilbertMatrix`, `HankelMatrix`, `ToeplitzMatrix`, `HadamardMatrix`,
    `FourierMatrix`, `VandermondeMatrix`, `PauliMatrix`, `RotationMatrix`,
    various rotation/reflection/scaling/shearing matrices, and a full set of
    matrix predicates (`SymmetricMatrixQ`, `PositiveDefiniteMatrixQ`,
    `OrthogonalMatrixQ`, `UnitaryMatrixQ`, …).
- Vectors and tensors: `Dot`, `Cross`, `Norm`, `Normalize`, `Projection`,
    `VectorAngle`, `KroneckerProduct`, `Outer`, `Inner`, `TensorProduct`,
    `TensorWedge`, `TensorTranspose`, `ArrayDot`, `LeviCivitaTensor`,
    `SparseArray` and numerous distance functions.

## Integral transforms & signal processing

- `FourierTransform`/`InverseFourierTransform` (incl. sine/cosine variants),
    `LaplaceTransform`/`InverseLaplaceTransform`, `ZTransform`/`InverseZTransform`,
    `MellinTransform`/`InverseMellinTransform`, discrete `Fourier`/`InverseFourier`
    (Cooley–Tukey FFT), `Convolve`, `DiscreteConvolve`, `ListConvolve`,
    `ListCorrelate`, Fourier series coefficients, `DiscreteHadamardTransform`,
    `DiscreteHilbertTransform`.
- Filters and resampling: `LowpassFilter`, `HighpassFilter`, `BandpassFilter`,
    `BandstopFilter`, `WienerFilter`, `TotalVariationFilter`, `MeanFilter`,
    `MedianFilter`, `MinFilter`/`MaxFilter`, `Upsample`/`Downsample`,
    `PeakDetect`/`FindPeaks`, `CrossingDetect`, `SavitzkyGolayMatrix`, waveform
    generators (`SawtoothWave`, `SquareWave`, `TriangleWave`) and a full set of
    window functions.
- Wavelet analysis: the wavelet families (`HaarWavelet`, `DaubechiesWavelet`,
    `SymletWavelet`, `CoifletWavelet`, `MeyerWavelet`, `MexicanHatWavelet`,
    `MorletWavelet`, …), the transforms (`DiscreteWaveletTransform`,
    `StationaryWaveletTransform`, `LiftingWaveletTransform`,
    `ContinuousWaveletTransform` and inverses), the data objects
    (`DiscreteWaveletData`, `ContinuousWaveletData`), coefficient manipulation
    (`WaveletThreshold`, `WaveletBestBasis`), and the wavelet plots
    (`WaveletListPlot`, `WaveletScalogram`, …).

## Graph theory & permutations

- Graph construction and rendering: `Graph`, `GraphPlot`, `LayeredGraphPlot`,
    named graphs (`CompleteGraph`, `PathGraph`, `CycleGraph`, `WheelGraph`,
    `StarGraph`, `HypercubeGraph`, `PetersenGraph`, `KaryTree`, `TuranGraph`,
    `DeBruijnGraph`, `CirculantGraph`, …), adjacency/incidence conversions,
    `Subgraph`, `LineGraph`, `NeighborhoodGraph`, `DirectedGraph`,
    `TransitiveReductionGraph`, and edge/vertex editing.
- Metrics and algorithms: `DegreeCentrality`, `BetweennessCentrality`,
    `ClosenessCentrality`, `EigenvectorCentrality`, `KatzCentrality`,
    `PageRankCentrality`, `RadialityCentrality`, `GraphLinkEfficiency`,
    `GraphDistance`, `FindShortestPath`,
    `FindSpanningTree`, `FindCycle`, `FindMaximumFlow`, `FindMinimumCostFlow`,
    `FindClique`, `FindVertexCover`, `ConnectedComponents`,
    `WeaklyConnectedComponents`, `TuttePolynomial`, `ChromaticPolynomial`,
    `GraphDiameter`/`GraphRadius`/`GraphCenter`/`GraphPeriphery`, and a family of
    graph predicates.
- Group theory: `SymmetricGroup`, `AlternatingGroup`, `DihedralGroup`,
    `CyclicGroup`, the Mathieu groups (`M11`, `M12`, `M22`, `M23`, `M24`),
    `CycleIndexPolynomial`, `GroupMultiplicationTable`, `GroupStabilizer`,
    `GroupOrbits`, `GroupElementPosition`, and permutation operations
    (`PermutationProduct`, `PermutationPower`, `InversePermutation`,
    `FindPermutation`, `Cycles`).

## Lists, associations & functional programming

- Core list operations: `Table`, `Map`, `MapThread`, `MapIndexed`, `MapAt`,
    `Apply`, `Thread`, `Through`, `Fold`/`FoldList`/`FoldPair`, `Nest`/`NestList`,
    `NestWhile`/`NestWhileList` (with cycle detection), `Tuples`, `Subsets`,
    `Permutations`, `Flatten`, `Partition`, `Take`/`Drop`, `Part` (with `All`,
    `Span` and multi-index specs), `Cases`, `Count`, `Select`, `Pick`,
    `Sow`/`Reap`, `Gather`/`GatherBy`, `SortBy`, `Ordering`, `Subdivide`.
- Associations: the `<|…|>` constructor with key access, `AssociationThread`,
    `AssociationMap`, `Merge`, `KeyMap`, `KeySelect`, `KeyTake`/`KeyDrop`,
    `Keys`/`Values`, `KeySort`, `KeyValueMap`, `Lookup`, `GroupBy`, `Counts`,
    `Query` and `Dataset`.
- Higher-order helpers: `OperatorApplied`, `Comap`, `ReverseApplied`, `Curry`,
    `SequenceFold`, `ArrayReduce`, `SubsetMap`, `ReplaceAt`, `NearestTo`,
    `PositionLargest`/`PositionSmallest`, and the `AddSides`/`SubtractSides`/…
    equation-manipulation operators.

## Strings & text

- `StringJoin`, `StringSplit`, `StringReplace`, `StringCases`, `StringPosition`,
    `StringMatchQ`, `StringContainsQ`/`StringStartsQ`/`StringEndsQ`,
    `StringTake`/`StringDrop`, `StringInsert`/`StringDelete`, `StringPartition`,
    `StringRiffle`, `StringTemplate`,
    `StringForm`, `Capitalize`/`Decapitalize`, `ToUpperCase`/`ToLowerCase`,
    `Characters`, `CharacterRange`, `ToCharacterCode`/`FromCharacterCode`
    (with encodings), `IntegerString`, `Alphabet`, `Transliterate`, and full
    string-pattern support (`RegularExpression`, `Except`, `Repeated`, captures
    and backreferences) threaded over lists.
- Sequence alignment and similarity: `LongestCommonSubsequence`,
    `SequenceAlignment`, `NeedlemanWunschSimilarity`, `SmithWatermanSimilarity`,
    `DamerauLevenshteinDistance`, `WordCounts`, `TextSentences`.

## Dates, times, units & quantities

- Date/time: `DateObject`, `DateList`, `DateString`, `DateValue`, `DateRange`,
    `DatePlus`, `DayName`, `DayCount`, `DayRange`, `DateWithinQ`, `DateOverlapsQ`,
    `DateSelect`, `DatePattern`, `Duration`, `CalendarConvert`, `Now`, `Today`,
    `TimeObject`, `TimeSeries`/`TemporalData`, `AbsoluteTime`/`FromAbsoluteTime`,
    `UnixTime`, `JulianDate`, `TimeZoneConvert`/`TimeZoneOffset` (DST-aware named
    IANA zones), `$TimeZone`, and `DateObject` + `Quantity` arithmetic.
- Units: `Quantity`, `UnitConvert`, `UnitDimensions`, `QuantityUnit`,
    `KnownUnitQ`, compound-unit parsing and dimensional analysis, and affine
    temperature handling.

## Geometry & regions

- Regions and measures: `RegionMeasure`, `Area`, `Volume`, `Perimeter`,
    `SurfaceArea`, `ArcLength`, `RegionCentroid`, `RegionMoment`,
    `MomentOfInertia`, `RegionNearest`, `RegionDistance`, `RegionMember`,
    `RegionDisjoint`, `BoundingRegion`, `MeshRegion`, `VoronoiMesh`, `ArrayMesh`,
    `CantorMesh` and morphological operations.
- Constructors and transforms: `Triangle` (AAS/ASA/SAS/SSS, including symbolic
    angles), `TriangleCenter`/`TriangleMeasurement`, `Simplex`, `Ball`,
    `Ellipsoid`, `RegularPolygon`, the Platonic-solid primitives, `SphericalShell`,
    `CapsuleShape`, `StadiumShape`, `DiskSegment`, `HalfSpace`, `Insphere`,
    `AngleBisector`/`PerpendicularBisector`, the geometric predicates
    (`CollinearPoints`, `CoplanarPoints`, `ConvexPolygonQ`, `SimplePolygonQ`),
    coordinate-bounding utilities, and the affine
    transformation family (`TranslationTransform`, `RotationTransform`,
    `ScalingTransform`, `ShearingTransform`, `ReflectionTransform`,
    `AffineTransform`, `EulerMatrix`, `RollPitchYawMatrix`).
- Space-filling and fractal curves: `HilbertCurve`, `PeanoCurve`,
    `SierpinskiCurve`, `KochCurve`, `CantorStaircase`, `AnglePath`/`AnglePath3D`,
    `MandelbrotSetMemberQ` and `MandelbrotSetIterationCount`.

## Graphics & plotting

- Function plots: `Plot`, `Plot3D`, `ParametricPlot`/`ParametricPlot3D`,
    `PolarPlot`, `PolarCurve`/`FilledPolarCurve`, `ContourPlot`, `DensityPlot`,
    `RegionPlot`/`RegionPlot3D`, `RevolutionPlot3D`, `SphericalPlot3D`,
    `ComplexPlot`/`ComplexPlot3D`/`ComplexRegionPlot`, `LogPlot`/`LogLogPlot`/
    `LogLinearPlot`, `DiscretePlot`/`DiscretePlot3D`.
- List and chart visualizations: `ListPlot`, `ListLinePlot`, `ListPointPlot3D`,
    `ListContourPlot`, `ListDensityPlot`, `DateListPlot`, `NumberLinePlot`,
    `BarChart`/`BarChart3D`, `PieChart`/`SectorChart`, `BubbleChart`,
    `Histogram`, `BoxWhiskerChart`, `ArrayPlot`, `MatrixPlot`, `WordCloud`,
    `TimelinePlot`, `AngularGauge`, `PeriodicTablePlot`, `GeoGraphics`/
    `GeoHistogram`.
- Primitives, styling and output: `Graphics`/`Graphics3D`, the box-language
    pipeline, `GraphicsComplex`, `BezierCurve` and `BSplineCurve`, `Raster`, gradient
    fills, plot options (`PlotStyle`, `PlotLegends`, `PlotTheme`, `GridLines`,
    `Filling`, `Frame`, `Callout`, …), `Show`,
    `GraphicsRow`/`GraphicsColumn`/`GraphicsGrid`,
    light/dark-mode SVG, and rendering via `ExportString[expr, "SVG"]`.

## Images, audio & music

- Images: an `Image`/`Image3D` type with data access, arithmetic
    (`ImageAdd`/`ImageSubtract`/`ImageMultiply`, `Blend`), filters (`GaussianFilter`,
    `MedianFilter`, `ImageConvolve`, …), geometry (`ImageResize`, `ImageRotate`,
    `ImageReflect`, `ImageTrim`, `ImagePartition`, `Thumbnail`), color operations
    (`ColorConvert`, `ColorSeparate`, `ColorCombine`, `ColorNegate`,
    `ColorDistance`), analysis (`ImageValue`, `DistanceTransform`,
    `FillingTransform`, `MorphologicalBinarize`),
    `ImageCollage`/`ImageAssemble`, `Rasterize`, and image import/SVG export.
- Audio: the Audio Processing guide — editing (`AudioAmplify`, `AudioTrim`,
    `AudioJoin`, `AudioPitchShift`), analysis (`AudioMeasurements`,
    `AudioLocalMeasurements`, `AudioIntervals`), the short-time Fourier transform,
    spectral plots (`Spectrogram`, `Cepstrogram`, `Periodogram`), noise-removal
    filters, WAV import/export, and audible `Play`/`Sound`/`Audio` playback in
    the Playground and Studio.
- Music: computational-music objects (`MusicNote`, `MusicChord`, `MusicPitch`,
    `MusicDuration`) with canonicalization, pitch arithmetic, MIDI export and
    SMuFL staff rendering.

## Data, knowledge & I/O

- Knowledge and entities: `EntityStore`/`EntityRegister`/`EntityValue`,
    `ElementData` for all 118 elements, a country/planet knowledge base,
    `GeoPosition`/`Latitude`/`Longitude` and the geodesy functions
    (`GeoDistance`, `GeoPath`, `GeoDestination`, `GeoAntipode`), plus
    `Molecule`/`MoleculeValue` and `WikidataData`.
- Import/Export: `Import`/`ImportString` and `Export`/`ExportString` for CSV,
    TSV, Table, Text, JSON, XLSX, XML, image/SVG and CERN ROOT formats, `Dataset`,
    `BinarySerialize`/`BinaryDeserialize` (WXF), `$ImportFormats`/`$ExportFormats`,
    and `Hash` (MD5, SHA, CRC32, … with multiple output encodings),
    `BaseEncode`/`BaseDecode`.
- Files, streams and system: file-path utilities
    (`FileNameJoin`/`FileNameSplit`/`FileNameTake`, `DirectoryName`,
    `FileExistsQ`, `FileNames`, `SetDirectory`, `CreateFile`, `CopyFile`,
    `RenameFile`/`RenameDirectory`, `FileSize`), stream I/O
    (`OpenRead`/`OpenWrite`, `Read`/`Write`/`ReadList`, `BinaryRead`/`BinaryWrite`,
    `Put`/`Get`), and system variables (`$Version`, `$VersionNumber`,
    `$OperatingSystem`, `$SystemID`, and the memory/timing variables).
- Web: `URLRead`, `HTTPRequest`, `URLParse`, `URLBuild`, `URLEncode`/`URLDecode`.

## Language, patterns & evaluation

- Pattern matching: `Pattern`, `Blank`/`BlankSequence`/`BlankNullSequence`,
    `Optional` and defaults, `Alternatives`, `Except`, `Condition` (`/;`),
    `Verbatim`, `HoldPattern`, `KeyValuePattern`, `Repeated`/`RepeatedNull`,
    with `Flat`/`Orderless`/`OneIdentity` matching.
- Rules, definitions and attributes: `Set`/`SetDelayed`, `TagSet`/`TagSetDelayed`,
    `UpSet`/`UpSetDelayed`, `Unset`, `DownValues`/`UpValues`/`SubValues`,
    `Attributes` with the full attribute set, `Protect`/`Unprotect`,
    `Clear`/`ClearAll`/`Remove`, `Options`/`SetOptions`/`OptionValue`.
- Scoping and control flow: `Module`, `Block`, `With`, `If`, `Which`, `Switch`,
    `For`, `While`, `Do`, `Break`, `Continue`, `Return`, `Goto`/`Label`,
    `CompoundExpression`, `ApplyTo` (`//=`), `PrintTemporary`,
    `Hold`/`HoldForm`/`ReleaseHold`, `Evaluate`,
    `Sequence` flattening, `Catch`/`Throw`, `Quiet`, `Check`, `TimeConstrained`,
    `MemoryConstrained`, and `Message`/`MessageName` diagnostics matching
    `wolframscript`.
- Booleans and logic: `And`/`Or`/`Not`/`Nand`/`Nor`/`Xor`/`Xnor`/`Implies`/
    `Equivalent`, `Boole`, `BooleanConvert` (DNF/CNF), `BooleanMinimize`
    (Quine–McCluskey), `BooleanTable`, `SatisfiableQ`, `TautologyQ`, `Exists`,
    `ForAll`.

## Parser & syntax

- Operator support: implicit multiplication, `n!`/`n!!`, `*^` scientific
    literals, `..`/`...` repeated patterns, the `@`/`@@`/`@@@`/`//`/`/@`/`~f~`
    application operators with correct precedence, `|->` arrow functions,
    `;;` spans, `>>`/`>>>` `Put`/`PutAppend`, `^:=` `UpSetDelayed`, `::`
    message names, and many named infix operators (`CircleDot`, `CircleTimes`,
    `Wedge`, `CenterDot`, `Element`, `Distributed`, …).
- Unicode and box syntax: Unicode operators (`≤`, `≥`, `≠`, `→`, `∈`, `∑`, …),
    named characters (`\[Psi]`, `\[Element]`, …) as symbols and function heads,
    character escapes (`\.HH`, `\:HHHH`, `\OOO`), box-syntax escapes and
    multi-line continuation.

## Output & formatting

- Form functions: `InputForm`, `FullForm`, `OutputForm`, `TraditionalForm`
    (conventional TeX-like typesetting of sums, integrals, derivatives, matrices,
    radicals and special functions), `TeXForm`, `MathMLForm`, `CForm`,
    `FortranForm`, `TableForm`, `MatrixForm`, `TreeForm`, `Column`, `Grid`,
    `Row`, `Framed`, `Definition`/`FullDefinition`.
- Number formatting: `NumberForm`, `ScientificForm`, `EngineeringForm`,
    `AccountingForm`, `PaddedForm`, `PercentForm`, `BaseForm`, with correct
    scientific-notation thresholds, digit blocking and banker's rounding, and
    consistent 6-significant-figure machine-real rendering.

## Interactive manipulation (Playground & Studio)

- `Manipulate` renders as an interactive widget driving live graphics, with
    sliders, popup menus, `SetterBar`/`CheckboxBar`/`RadioButtonBar`, `Locator`
    controls, 2D sliders, interval sliders, discrete pick-lists and the
    standalone `Control[…]` expression. The remaining interactive-manipulation
    heads (`Animator`, `Trigger`, `ProgressIndicator`, `PopupView`,
    `PaneSelector`, `Slider2D`, …) stay symbolic in script mode, matching
    `wolframscript`.
- `Animate` and `ListAnimate` render as auto-playing widgets with a play/pause
    button; `LocatorPane` and `ClickPane` render as draggable/clickable pads that
    feed pointer positions to their handlers. `ControlActive` now evaluates to
    its inactive form outside an actively manipulated control.

## Woxi Studio

- New native `.nb` notebook editor (`woxi-studio` crate) built with `iced`:
    per-cell evaluation (Shift+Enter), cell-type dropdown, drag-and-drop cell
    reordering, undo/redo, preview mode, dark-mode styling, selectable output
    text, keyboard shortcuts and navigation, 3D-graphics and image modals, an
    interactive `Manipulate` pipeline, external-player audio play/pause, and
    export to Mathematica / Jupyter / Markdown / LaTeX / Typst.

## Woxi Playground & JupyterLite

- Playground: WASM interpreter with a CodeMirror editor, per-expression output
    boxes, SVG/graphics and `Dataset`/`TableForm` rendering, `?symbol`
    information lookup, an auto/light/dark theme toggle, and a share button that
    encodes the session into the URL.
- JupyterLite: an integrated Woxi kernel with graphical `Plot`/SVG output and
    `?symbol` support, embedded in the docs.

## Language bindings

- Woxi for Python: a PyO3/maturin package (published to PyPI as `woxi`) that
    wraps the interpreter, evaluating Wolfram Language expressions from Python.
- Node.js: an npm package with WebAssembly bindings for running Woxi in
    JavaScript/Node.js environments.

# 2025-05-08 - 0.1.0

- Render top-level `PolarCurve[…]` and `FilledPolarCurve[…]` as graphics in
    the playground and Woxi Studio (the CLI keeps the symbolic echo), and
    support the `FilledPolarCurve[r, θ]` bare-variable form
- Render `Region[Style[reg, directives…]]` with the style directives applied
- Render `DateObject[…]` results (e.g. from `RandomDate` or `Now`) as a
    framed date panel in the playground and Woxi Studio
- Implement `WikidataData` and `ExternalIdentifier`,
    including `Import` of SVG files and `URL[…]` sources
- Render `Audio[…]` objects (file-backed via `File[…]`/path strings, or from
    sample data) as a graphical audio player in the playground and Woxi Studio
- Add support for `HTTPRequest` objects including property extraction
- Add support for `QuestionObject`, `AssessmentFunction`, and `AssessmentResultObject`
- Implement `DateString` and `Now`
- Implement `StringStartsQ` and `StringEndsQ`
- Support executing Woxi as a shebang script
- Implement `RandomInteger` function
- Implement `AllTrue` function
- Add support for anonymous functions
- Implement `MemberQ` function
- Implement `NumberQ` function
- Add support for all comparison operators
- Add support for function declarations
- Add support for semicolon separated expressions, implement `Set`
- Add support for comments, implement `Sin`, `@`, and `//`
- Add support for associations
- Implement `Floor`, `Ceiling`, and `Round` functions
- Implement `Divide` function
- Implement several boolean functions
- Implement `Times` function
- Implement `Minus` function
- Implement `Plus` function
- Implement `Sqrt` function
- Implement several string functions
- Add support for `#^2& /@ …`
- Implement `Abs` function
- Implement `/@` (Map operator)
- Implement `Total` function
- Implement `Select` and `Flatten`
- Implement `Drop`, `Append`, and `Prepend`
- Implement `Rest`, `Most`, `Take`, and `Part`
- Implement `Map` function
- Implement `Sign` function
- Implement `Length` function
- Implement `Print` function
- Implement `EvenQ` and `OddQ` functions
- Implement `Prime` function
- Add CLI subcommands `run`
- Add subcommand `eval` for evaluating Wolfram Language expressions
