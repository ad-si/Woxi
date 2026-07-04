# Changelog

# Unreleased

- Add support for the remaining interactive-manipulation heads from the
    [InteractiveManipulation](https://reference.wolfram.com/language/guide/InteractiveManipulation.html)
    guide. `Animate`, `Animator`, `ListAnimate`, `ControllerManipulate`,
    `Trigger`, `SetterBar`, `CheckboxBar`, `TogglerBar`, `RadioButton`,
    `ProgressIndicator`, `ClickPane`, `PaneSelector`, `PopupView`,
    `ColorSetter`, `IntervalSlider`, and `Slider2D` now stay symbolic in
    script mode (matching `wolframscript`) instead of emitting a spurious
    "not yet implemented" warning, and the option symbols `Bookmarks`,
    `ContinuousAction`, and `AppearanceElements` do the same. `ControlActive`
    now evaluates: `ControlActive[active, normal]` returns its inactive form
    `normal` outside an actively-manipulated control.
- Render `TraditionalForm` mathematics in conventional, TeX-like notation in
    the Playground and Studio typeset SVG output. `HoldForm` is now invisible;
    `Sum`/`Product` become large `∑`/`∏` operators with limits above and below;
    `Integrate` uses a `∫` sign with bounds and a `ⅆx` differential; `D` renders
    as a Leibniz `∂ⁿ/∂xᵐ` fraction; `Det` and matrices get stretchy bars and
    parentheses; `Sqrt` is drawn as one connected radical-and-vinculum object
    that scales to its content; `Sin[x]^2` becomes `sin²(x)`; constants map to
    glyphs (`Pi`→`π`, `Infinity`→`∞`, `Exp[x]`→`ⅇˣ`); `Gamma`/`Zeta`/`BesselJ`,
    `Grad`/`Div`/`Curl`/`Laplacian` (`∇`), and `Limit` use their standard
    notation; binary operators and relations get proper spacing; and single
    Latin-letter variables are italicized. The parser now also accepts a
    named-character symbol as a function head (e.g. `\[Psi][x, t]`).
- Add support for the standalone `Control[…]` control expression in the
    Playground and Studio. `Control[{x, 0, 1}]` renders a slider,
    `Control[{x, {a, b, c}}]` a popup menu, `Control[{x, {0, 0}, {1, 1}}]`
    and `Control[{xy, 0, 1, ControlType -> Slider2D}]` a 2D slider, and
    `Control[{int, 0, 1, ControlType -> IntervalSlider}]` an interval
    slider. Initial values and labels (`Control[{{x, 0.5, "variable"}, 0, 1}]`)
    are honored, and the same `Slider2D` / `IntervalSlider` control types now
    work inside `Manipulate` too.
- Extend interactive `Manipulate` rendering (Playground + Studio) to
    support `Locator` controls and `ControlType -> None` (both bound to a
    frozen initial value), discrete pick lists whose value list is an
    expression that evaluates to a list (e.g. `{g, PolyhedronData[All]}`),
    and extra display arguments such as a trailing `Dynamic[Panel[…]]`
    (ignored). Also add `PolyhedronData[All]`, `PolyhedronData["Properties"]`,
    and `PolyhedronData["Classes"]`.
- Implement the audio processing functions: editing (`AudioAmplify`,
    `AudioTrim`, `AudioJoin`, `AudioPitchShift`), analysis
    (`AudioMeasurements`, `AudioLocalMeasurements`, `AudioIntervals`),
    the short-time Fourier transform (`ShortTimeFourier` with the
    `ShortTimeFourierData` object), the spectral plots (`Spectrogram`,
    `Cepstrogram`, `Periodogram`), the noise-removal filters
    (`WienerFilter`, `TotalVariationFilter`), Audio support for
    `LowpassFilter`, `MeanFilter`, `Mean`, `Median`, `Variance`, and
    `Quantile`, `Import` of audio files (WAV decoded to sample data),
    and headless `$Failed` stubs for `AudioCapture`/`WebAudioSearch`
- Implement the wavelet analysis functions: the wavelet families
    (`HaarWavelet`, `DaubechiesWavelet`, `SymletWavelet`, `CoifletWavelet`,
    `BattleLemarieWavelet`, `BiorthogonalSplineWavelet`,
    `ReverseBiorthogonalSplineWavelet`, `CDFWavelet`, `MeyerWavelet`,
    `ShannonWavelet`, `MexicanHatWavelet`, `MorletWavelet`, `GaborWavelet`,
    `DGaussianWavelet`, `PaulWavelet`), `WaveletFilterCoefficients`,
    the transforms (`DiscreteWaveletTransform`, `StationaryWaveletTransform`,
    `DiscreteWaveletPacketTransform`, `StationaryWaveletPacketTransform`,
    `LiftingWaveletTransform`, `InverseWaveletTransform`,
    `ContinuousWaveletTransform`, `InverseContinuousWaveletTransform`),
    the data objects (`DiscreteWaveletData`, `ContinuousWaveletData`,
    `LiftingFilterData`) with coefficient and property access,
    coefficient manipulation (`WaveletThreshold`, `WaveletMapIndexed`,
    `WaveletBestBasis`), the scaling/wavelet functions
    (`WaveletPhi`, `WaveletPsi`), and the wavelet plots
    (`WaveletListPlot`, `WaveletMatrixPlot`, `WaveletImagePlot`,
    `WaveletScalogram`)

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
- Add support for the `PolarCurve` and `FilledPolarCurve` graphics primitives
- Add support for `HTTPRequest` objects including property extraction
- Add support for `QuestionObject`, `AssessmentFunction`, and `AssessmentResultObject`
- Implement `DateString` and `Now`
- Implement `StringStartsQ` and `StringEndsQ`
- Support executing Woxi as a shebang script
- Implement `RandomInteger` function
- Implement `AllTrue` function
- Add support for anonymous functions
- Implement `AllTrue` function
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
