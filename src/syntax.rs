use num_bigint::{BigInt, Sign};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageType {
  Bit,
  Byte,
  Bit16,
  Real32,
  Real64,
}

/// Owned expression tree for storing parsed function bodies.
/// This avoids re-parsing function bodies on every call.
#[derive(Debug)]
pub enum Expr {
  /// Integer literal
  Integer(i128),
  /// Big integer (exceeds i128 range)
  BigInteger(BigInt),
  /// Real/float literal
  Real(f64),
  /// Arbitrary-precision real: (formatted_digits, precision_in_decimal_digits)
  BigFloat(String, f64),
  /// String literal (without quotes)
  String(String),
  /// Identifier/symbol
  Identifier(String),
  /// Slot (#, #1, #2, etc.)
  Slot(usize),
  /// SlotSequence (##, ##1, ##2, etc.) — represents a sequence of arguments
  SlotSequence(usize),
  /// List: {e1, e2, ...}
  List(crate::ExprList),
  /// Function call: f[e1, e2, ...]
  FunctionCall { name: String, args: crate::ExprList },
  /// Binary operator: e1 op e2
  BinaryOp {
    op: BinaryOperator,
    left: Box<Self>,
    right: Box<Self>,
  },
  /// Unary operator: op e
  UnaryOp {
    op: UnaryOperator,
    operand: Box<Self>,
  },
  /// Comparison chain: e1 op1 e2 op2 e3 ...
  Comparison {
    operands: Vec<Self>,
    operators: Vec<ComparisonOp>,
  },
  /// Compound expression: e1; e2; e3
  CompoundExpr(Vec<Self>),
  /// Association: <| key1 -> val1, key2 -> val2, ... |>
  Association(Vec<(Self, Self)>),
  /// Rule: pattern -> replacement
  Rule {
    pattern: Box<Self>,
    replacement: Box<Self>,
  },
  /// Delayed rule: pattern :> replacement
  RuleDelayed {
    pattern: Box<Self>,
    replacement: Box<Self>,
  },
  /// ReplaceAll: expr /. rules
  ReplaceAll { expr: Box<Self>, rules: Box<Self> },
  /// ReplaceRepeated: expr //. rules
  ReplaceRepeated { expr: Box<Self>, rules: Box<Self> },
  /// Map: f /@ list
  Map { func: Box<Self>, list: Box<Self> },
  /// Apply: f @@ list
  Apply { func: Box<Self>, list: Box<Self> },
  /// MapApply: f @@@ list (applies f to each sublist)
  MapApply { func: Box<Self>, list: Box<Self> },
  /// Prefix application: f @ x (equivalent to f[x])
  PrefixApply { func: Box<Self>, arg: Box<Self> },
  /// Postfix application: expr // f
  Postfix { expr: Box<Self>, func: Box<Self> },
  /// Part extraction: expr[[index]]
  Part { expr: Box<Self>, index: Box<Self> },
  /// Curried/chained function call: f[a][b] - func is f[a], args is {b}
  CurriedCall { func: Box<Self>, args: Vec<Self> },
  /// Anonymous function: body &
  Function { body: Box<Self> },
  /// Named-parameter function: Function[x, body] or Function[{x,y,...}, body].
  /// `bracketed` preserves how the parameters were written so the display
  /// stays faithful to the input: `Function[y, ...]` renders as
  /// `Function[y, ...]` and `Function[{y}, ...]` as `Function[{y}, ...]`.
  NamedFunction {
    params: Vec<String>,
    body: Box<Self>,
    bracketed: bool,
  },
  /// Pattern: name_ or name_Head or name__ (BlankSequence) or name___ (BlankNullSequence)
  /// blank_type: 1=Blank, 2=BlankSequence, 3=BlankNullSequence
  Pattern {
    name: String,
    head: Option<String>,
    blank_type: u8,
  },
  /// Optional pattern: name_ : default, name_Head : default, name_., or name_Head.
  PatternOptional {
    name: String,
    head: Option<String>,
    default: Option<Box<Self>>,
  },
  /// PatternTest: _?test or x_?test or x__?test — matches if test[x] is True
  /// blank_type: 1=Blank, 2=BlankSequence, 3=BlankNullSequence
  PatternTest {
    name: String,
    head: Option<String>,
    blank_type: u8,
    test: Box<Self>,
  },
  /// Constant like Pi, E, etc.
  Constant(String),
  /// Raw unparsed text (fallback)
  Raw(String),
  /// Image: raster image data
  Image {
    width: u32,
    height: u32,
    channels: u8,
    data: std::sync::Arc<Vec<f64>>,
    image_type: ImageType,
    /// Explicit color-space tag (e.g. from ColorCombine[imgs, "HSB"]).
    /// `None` reports as `Automatic` from ImageColorSpace.
    color_space: Option<&'static str>,
  },
  /// Graphics output: holds SVG string, displays as -Graphics- (or -Graphics3D- if is_3d).
  /// `source` optionally carries the raw plot series data so that
  /// `Show` can merge multiple pre-rendered plots by re-rendering via plotters.
  /// `head` overrides the symbolic head reported by `Head` (e.g. `GeoGraphics`),
  /// which otherwise defaults to `Graphics`/`Graphics3D`; the rendered SVG and
  /// the `-Graphics-` placeholder are unaffected.
  Graphics {
    svg: String,
    is_3d: bool,
    source: Option<Box<PlotSource>>,
    head: Option<String>,
    /// The symbolic expression this rendering was produced from (e.g.
    /// `Graphics3D[GraphicsComplex[…], opts]`), so structural operations
    /// like `Part` can look inside a rendered graphic the way Wolfram's
    /// `Graphics3D[…][[1]]` does. `None` for renderings with no useful
    /// symbolic form.
    structure: Option<Box<Self>>,
  },
}

/// Raw plot data stored alongside pre-rendered SVG so that `Show` can
/// merge multiple plots and re-render them together via plotters.
#[derive(Debug, Clone)]
pub struct PlotSource {
  pub series: Vec<PlotSeriesData>,
  pub x_range: (f64, f64),
  pub y_range: (f64, f64),
  pub image_size: (u32, u32),
  /// The option rules the plot was called with, verbatim. `Show` merges
  /// them into the combined graphic the way Wolfram does — the result
  /// keeps the options of the graphics it was given.
  pub options: Vec<Expr>,
}

/// Filling specification for a plot series (used by `Show` when merging
/// pre-rendered plots). Mirrors `functions::plot::Filling` without pulling
/// in the whole plot module as a dependency of `syntax`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SeriesFilling {
  None,
  Axis,
  Bottom,
  Top,
  Value(f64),
}

/// The glyph a series' points are drawn with (`PlotMarkers`). Lives here
/// so `Show` can carry it across a merge, like [`SeriesFilling`].
#[derive(Debug, Clone, PartialEq)]
pub struct PlotMarker {
  /// The character(s) drawn at every data point.
  pub glyph: String,
  /// Font size in printer's points, as `Style[marker, size]` gives it.
  pub size: f64,
  /// Colour from `Style[marker, colour]`; `None` = the series colour.
  pub color: Option<(u8, u8, u8)>,
}

/// A single data series within a plot.
#[derive(Debug, Clone)]
pub struct PlotSeriesData {
  pub points: Vec<(f64, f64)>,
  pub color: (u8, u8, u8),
  pub is_scatter: bool,
  pub filling: SeriesFilling,
  /// Fill color override from `FillingStyle` (`None` = use `color`).
  pub fill_color: Option<(u8, u8, u8)>,
  /// Fill opacity override from `FillingStyle` (`None` = the 0.2 default).
  pub fill_opacity: Option<f64>,
  /// `PlotMarkers` glyph for this series (`None` = plain round points).
  pub marker: Option<PlotMarker>,
  /// Line weight in display pixels from `PlotStyle` (`Thick`, `Thickness[…]`,
  /// …); `None` = the 1.5px default. Travels with the series so a curve keeps
  /// its weight when `Show` merges it into another graphic.
  pub thickness: Option<f64>,
  /// Dot size of a scatter series from `PlotStyle` — a fraction of the image
  /// width for `PointSize[…]`, stored negative for the absolute printer's
  /// points of `AbsolutePointSize[…]`, the same sign convention the graphics
  /// primitives use. `None` = the default dot.
  pub point_size: Option<f64>,
}

/// Convert a Wolfram named character name (e.g. "Pi", "Alpha", "Sum") to its
/// Unicode string. Returns None if the name is not recognized.
///
/// The code point returned is the one Wolfram itself stores, which for a
/// large part of the table is a *private-use* character rather than the
/// standard Unicode look-alike: `\[WarningSign]` is U+F725, not U+26A0, and
/// `ToCharacterCode` has to report exactly that. Only Wolfram's own fonts
/// draw those code points, so text that gets *drawn* runs through
/// [`substitute_private_use_glyphs`] first; the string itself keeps the
/// canonical code point.
pub fn named_char_to_unicode(name: &str) -> Option<&'static str> {
  Some(match name {
    // Constants / special identifiers (render as Unicode in strings)
    "ExponentialE" => "\u{F74D}",
    "Degree" => "\u{00B0}",
    "Infinity" => "\u{221E}",
    "ImaginaryI" => "\u{F74E}",
    "ImaginaryJ" => "\u{F74F}",
    // Lowercase Greek (Pi handled here too)
    "Alpha" => "\u{03B1}",
    "Beta" => "\u{03B2}",
    "Gamma" => "\u{03B3}",
    "Delta" => "\u{03B4}",
    "Epsilon" => "\u{03F5}",
    "Zeta" => "\u{03B6}",
    "Eta" => "\u{03B7}",
    "Theta" => "\u{03B8}",
    "Iota" => "\u{03B9}",
    "Kappa" => "\u{03BA}",
    "Lambda" => "\u{03BB}",
    "Mu" => "\u{03BC}",
    "Nu" => "\u{03BD}",
    "Xi" => "\u{03BE}",
    "Omicron" => "\u{03BF}",
    "Pi" => "\u{03C0}",
    "Rho" => "\u{03C1}",
    "Sigma" => "\u{03C3}",
    "FinalSigma" => "\u{03C2}",
    "Tau" => "\u{03C4}",
    "Upsilon" => "\u{03C5}",
    "Phi" => "\u{03D5}",
    "CurlyPhi" => "\u{03C6}",
    "CurlyEpsilon" => "\u{03B5}",
    "CurlyTheta" => "\u{03D1}",
    "CurlyKappa" => "\u{03F0}",
    "CurlyPi" => "\u{03D6}",
    "CurlyRho" => "\u{03F1}",
    "Chi" => "\u{03C7}",
    "Psi" => "\u{03C8}",
    "Omega" => "\u{03C9}",
    // Uppercase Greek
    "CapitalAlpha" => "\u{0391}",
    "CapitalBeta" => "\u{0392}",
    "CapitalGamma" => "\u{0393}",
    "CapitalDelta" => "\u{0394}",
    "CapitalEpsilon" => "\u{0395}",
    "CapitalZeta" => "\u{0396}",
    "CapitalEta" => "\u{0397}",
    "CapitalTheta" => "\u{0398}",
    "CapitalIota" => "\u{0399}",
    "CapitalKappa" => "\u{039A}",
    "CapitalLambda" => "\u{039B}",
    "CapitalMu" => "\u{039C}",
    "CapitalNu" => "\u{039D}",
    "CapitalXi" => "\u{039E}",
    "CapitalOmicron" => "\u{039F}",
    "CapitalPi" => "\u{03A0}",
    "CapitalRho" => "\u{03A1}",
    "CapitalSigma" => "\u{03A3}",
    "CapitalTau" => "\u{03A4}",
    "CapitalUpsilon" => "\u{03A5}",
    "CapitalPhi" => "\u{03A6}",
    "CapitalChi" => "\u{03A7}",
    "CapitalPsi" => "\u{03A8}",
    "CapitalOmega" => "\u{03A9}",
    // Script (calligraphic) letters. The letters whose script forms
    // predate the Mathematical Alphanumeric Symbols block live in
    // Letterlike Symbols (U+2100–214F) instead of U+1D49C–1D4CF.
    // Script letters. Wolfram keeps its own script alphabet in the private
    // use area (U+F770… and U+F6B2…) and only falls back to the standard
    // letterlike symbols where Unicode has one (ℬ, ℯ, ℓ, …) — the
    // Mathematical Alphanumeric Symbols block is *not* what it stores.
    "ScriptCapitalA" => "\u{F770}",
    "ScriptCapitalB" => "\u{212C}",
    "ScriptCapitalC" => "\u{F772}",
    "ScriptCapitalD" => "\u{F773}",
    "ScriptCapitalE" => "\u{2130}",
    "ScriptCapitalF" => "\u{2131}",
    "ScriptCapitalG" => "\u{F776}",
    "ScriptCapitalH" => "\u{210B}",
    "ScriptCapitalI" => "\u{2110}",
    "ScriptCapitalJ" => "\u{F779}",
    "ScriptCapitalK" => "\u{F77A}",
    "ScriptCapitalL" => "\u{2112}",
    "ScriptCapitalM" => "\u{2133}",
    "ScriptCapitalN" => "\u{F77D}",
    "ScriptCapitalO" => "\u{F77E}",
    "ScriptCapitalP" => "\u{F77F}",
    "ScriptCapitalQ" => "\u{F780}",
    "ScriptCapitalR" => "\u{211B}",
    "ScriptCapitalS" => "\u{F782}",
    "ScriptCapitalT" => "\u{F783}",
    "ScriptCapitalU" => "\u{F784}",
    "ScriptCapitalV" => "\u{F785}",
    "ScriptCapitalW" => "\u{F786}",
    "ScriptCapitalX" => "\u{F787}",
    "ScriptCapitalY" => "\u{F788}",
    "ScriptCapitalZ" => "\u{F789}",
    "ScriptA" => "\u{F6B2}",
    "ScriptB" => "\u{F6B3}",
    "ScriptC" => "\u{F6B4}",
    "ScriptD" => "\u{F6B5}",
    "ScriptE" => "\u{212F}",
    "ScriptF" => "\u{F6B7}",
    "ScriptG" => "\u{210A}",
    "ScriptH" => "\u{F6B9}",
    "ScriptI" => "\u{F6BA}",
    "ScriptJ" => "\u{F6BB}",
    "ScriptK" => "\u{F6BC}",
    "ScriptL" => "\u{2113}",
    "ScriptM" => "\u{F6BE}",
    "ScriptN" => "\u{F6BF}",
    "ScriptO" => "\u{2134}",
    "ScriptP" => "\u{F6C1}",
    "ScriptQ" => "\u{F6C2}",
    "ScriptR" => "\u{F6C3}",
    "ScriptS" => "\u{F6C4}",
    "ScriptT" => "\u{F6C5}",
    "ScriptU" => "\u{F6C6}",
    "ScriptV" => "\u{F6C7}",
    "ScriptW" => "\u{F6C8}",
    "ScriptX" => "\u{F6C9}",
    "ScriptY" => "\u{F6CA}",
    "ScriptZ" => "\u{F6CB}",
    // Common symbols
    "Euro" => "\u{20AC}",
    "Micro" => "\u{00B5}",
    "Angstrom" => "\u{212B}",
    "HBar" => "\u{210F}",
    // Math operators and symbols
    "Sum" => "\u{2211}",
    "Product" => "\u{220F}",
    "Integral" => "\u{222B}",
    "PartialD" => "\u{2202}",
    "Del" => "\u{2207}",
    "DifferentialD" => "\u{F74C}",
    "CapitalDifferentialD" => "\u{F74B}",
    "Sqrt" => "\u{221A}",
    "CubeRoot" => "\u{221B}",
    "Not" => "\u{00AC}",
    "And" => "\u{2227}",
    "Or" => "\u{2228}",
    "ForAll" => "\u{2200}",
    "Exists" => "\u{2203}",
    "NotExists" => "\u{2204}",
    "EmptySet" => "\u{2205}",
    "Element" => "\u{2208}",
    "NotElement" => "\u{2209}",
    "ReverseElement" => "\u{220B}",
    "Subset" => "\u{2282}",
    "Superset" => "\u{2283}",
    "SubsetEqual" => "\u{2286}",
    "SupersetEqual" => "\u{2287}",
    // Wolfram's `\[Union]`/`\[Intersection]` are the n-ary forms (⋃/⋂,
    // U+22C3/U+22C2), not the binary ∪/∩ (U+222A/U+2229).
    "Union" => "\u{22C3}",
    "Intersection" => "\u{22C2}",
    "Minus" => "\u{2212}",
    "PlusMinus" => "\u{00B1}",
    "MinusPlus" => "\u{2213}",
    "Times" => "\u{00D7}",
    "Divide" => "\u{00F7}",
    "CenterDot" => "\u{00B7}",
    // Unlike most named operator glyphs, `\[Backslash]` is not a private-use
    // look-alike: it is literally the ASCII backslash (its `Backslash[a, b]`
    // infix rendering is a separate display form, handled elsewhere).
    "Backslash" => "\\",
    // `\[Equal]` is the typeset `==`; it has its own private-use code point
    // rather than reusing the ASCII `=` (which is `Set`).
    "Equal" => "\u{F431}",
    "NotEqual" => "\u{2260}",
    "LessEqual" => "\u{2264}",
    "GreaterEqual" => "\u{2265}",
    // The slanted comparison family. `\[LessSlantEqual]` is the glyph a
    // Demonstration writes its ≤ relation with; the negated pair lives in
    // Wolfram's private use area, like the other negated operators.
    "LessSlantEqual" => "\u{2A7D}",
    "GreaterSlantEqual" => "\u{2A7E}",
    "NotLessSlantEqual" => "\u{F424}",
    "NotGreaterSlantEqual" => "\u{F429}",
    "LessFullEqual" => "\u{2266}",
    "GreaterFullEqual" => "\u{2267}",
    "NotLessEqual" => "\u{2270}",
    "NotGreaterEqual" => "\u{2271}",
    "LessTilde" => "\u{2272}",
    "GreaterTilde" => "\u{2273}",
    "LessEqualGreater" => "\u{22DA}",
    "GreaterEqualLess" => "\u{22DB}",
    "Proportional" => "\u{221D}",
    "Congruent" => "\u{2261}",
    "Tilde" => "\u{223C}",
    "TildeTilde" => "\u{2248}",
    "LeftArrow" => "\u{2190}",
    "RightArrow" => "\u{2192}",
    "UpArrow" => "\u{2191}",
    "DownArrow" => "\u{2193}",
    "LeftRightArrow" => "\u{2194}",
    "UpDownArrow" => "\u{2195}",
    "DoubleLeftArrow" => "\u{21D0}",
    "DoubleRightArrow" => "\u{21D2}",
    "DoubleUpArrow" => "\u{21D1}",
    "DoubleDownArrow" => "\u{21D3}",
    "DoubleLeftRightArrow" => "\u{21D4}",
    "DoubleUpDownArrow" => "\u{21D5}",
    // Long arrows — the reaction arrows a chemistry Demonstration writes
    // between reactants and products.
    "LongLeftArrow" => "\u{27F5}",
    "LongRightArrow" => "\u{27F6}",
    "LongLeftRightArrow" => "\u{27F7}",
    "DoubleLongLeftArrow" => "\u{27F8}",
    "DoubleLongRightArrow" => "\u{27F9}",
    "DoubleLongLeftRightArrow" => "\u{27FA}",
    "Equilibrium" => "\u{21CC}",
    "ReverseEquilibrium" => "\u{21CB}",
    "UpEquilibrium" => "\u{296E}",
    "ReverseUpEquilibrium" => "\u{296F}",
    "LeftTeeArrow" => "\u{21A4}",
    "RightTeeArrow" => "\u{21A6}",
    "UpTeeArrow" => "\u{21A5}",
    "DownTeeArrow" => "\u{21A7}",
    "Rule" => "\u{F522}",
    "RuleDelayed" => "\u{F51F}",
    "DirectedEdge" => "\u{F3D5}",
    "UndirectedEdge" => "\u{F3D4}",
    "Distributed" => "\u{F3D2}",
    "Conditioned" => "\u{F3D3}",
    "Cross" => "\u{F4A0}",
    "TensorProduct" => "\u{F3DA}",
    // Dots
    "Ellipsis" => "\u{2026}",
    "CenterEllipsis" => "\u{22EF}",
    "VerticalEllipsis" => "\u{22EE}",
    "AscendingEllipsis" => "\u{22F0}",
    "DescendingEllipsis" => "\u{22F1}",
    // Geometric and miscellaneous symbols (used as identifier-character
    // literals in Wolfram, e.g. `\[Angle]XYZ` is the symbol `∠XYZ`).
    "Angle" => "\u{2220}",
    "FilledSquare" => "\u{25A0}",
    "EmptySquare" => "\u{25A1}",
    "FilledSmallSquare" => "\u{25FC}",
    "EmptySmallSquare" => "\u{25FB}",
    "FilledVerySmallSquare" => "\u{25AA}",
    "EmptyVerySmallSquare" => "\u{25AB}",
    "FilledCircle" => "\u{25CF}",
    "EmptyCircle" => "\u{25CB}",
    "FilledSmallCircle" => "\u{F750}",
    "EmptySmallCircle" => "\u{25E6}",
    "FilledDiamond" => "\u{25C6}",
    "EmptyDiamond" => "\u{25C7}",
    "FilledUpTriangle" => "\u{25B2}",
    "EmptyUpTriangle" => "\u{25B3}",
    "FilledDownTriangle" => "\u{25BC}",
    "EmptyDownTriangle" => "\u{25BD}",
    "FilledLeftTriangle" => "\u{25C0}",
    "FilledRightTriangle" => "\u{25B6}",
    "Placeholder" => "\u{F528}",
    "SelectionPlaceholder" => "\u{F527}",
    // Pictographs and signs. Demonstrations reach for these inside prose
    // and control labels — a setter caption reading "…\[LongDash]\
    // \[WarningSign] slow!" must draw the sign, not the escape.
    "WarningSign" => "\u{F725}",
    "Checkmark" => "\u{2713}",
    "WatchIcon" => "\u{231A}",
    "FivePointedStar" => "\u{2605}",
    "SixPointedStar" => "\u{2736}",
    // Musical accidentals
    "Sharp" => "\u{266F}",
    "Flat" => "\u{266D}",
    "Natural" => "\u{266E}",
    // Astronomical symbols (the Sun and the planets, U+2609 and
    // U+263F-U+2647). Wolfram departs from that block twice: `\[Earth]` is a
    // private-use glyph and `\[Uranus]` is the astronomical ⛢ rather than the
    // astrological ♅.
    "Sun" => "\u{2609}",
    "Mercury" => "\u{263F}",
    "Venus" => "\u{2640}",
    "Earth" => "\u{F3DF}",
    "Mars" => "\u{2642}",
    "Jupiter" => "\u{2643}",
    "Saturn" => "\u{2644}",
    "Uranus" => "\u{26E2}",
    "Neptune" => "\u{2646}",
    "Pluto" => "\u{2647}",
    // Braces/brackets
    "LeftAngleBracket" => "\u{2329}",
    "RightAngleBracket" => "\u{232A}",
    "LeftCeiling" => "\u{2308}",
    "RightCeiling" => "\u{2309}",
    "LeftFloor" => "\u{230A}",
    "RightFloor" => "\u{230B}",
    "LeftDoubleBracket" => "\u{301A}",
    "RightDoubleBracket" => "\u{301B}",
    // Abs/Norm bars. Wolfram has no public Unicode code point for these
    // (unlike the ASCII `|` they visually resemble), so like `Rule` and
    // `Equal` above they live in the private use area.
    "LeftBracketingBar" => "\u{F603}",
    "RightBracketingBar" => "\u{F604}",
    "LeftDoubleBracketingBar" => "\u{F605}",
    "RightDoubleBracketingBar" => "\u{F606}",
    // Whitespace control characters (Wolfram treats these as the raw chars).
    // `\[IndentingNewLine]` is the FrontEnd's own newline, so it carries a
    // private-use code point instead of U+000A.
    "NewLine" => "\n",
    "IndentingNewLine" => "\u{F3A3}",
    "LineSeparator" => "\u{2028}",
    "ParagraphSeparator" => "\u{2029}",
    // Typographic punctuation
    "OpenCurlyQuote" => "\u{2018}",
    "CloseCurlyQuote" => "\u{2019}",
    "OpenCurlyDoubleQuote" => "\u{201C}",
    "CloseCurlyDoubleQuote" => "\u{201D}",
    "Hyphen" => "\u{2010}",
    "Dash" => "\u{2013}",
    "LongDash" => "\u{2014}",
    "LeftGuillemet" => "\u{00AB}",
    "RightGuillemet" => "\u{00BB}",
    "Prime" => "\u{2032}",
    "DoublePrime" => "\u{2033}",
    "Bullet" => "\u{2022}",
    "Dagger" => "\u{2020}",
    "DoubleDagger" => "\u{2021}",
    "Section" => "\u{00A7}",
    "Paragraph" => "\u{00B6}",
    "Copyright" => "\u{00A9}",
    "RegisteredTrademark" => "\u{00AE}",
    "Trademark" => "\u{2122}",
    "Continuation" => "\u{F3B1}",
    // The span characters fill the cells a `Grid` entry reaches over. They
    // draw nothing, but they are characters: `StringLength` counts them.
    "SpanFromLeft" => "\u{F3BA}",
    "SpanFromAbove" => "\u{F3BB}",
    "SpanFromBoth" => "\u{F3BC}",
    // The `\[Raw…]` family names the ASCII control characters, so that a
    // notebook can write one without embedding an untypeable byte.
    "RawEscape" => "\u{001B}",
    "RawTab" => "\t",
    "RawReturn" => "\r",
    "RawSpace" => " ",
    // Miscellaneous
    "Null" => "\u{F3A0}",
    "InvisibleSpace" => "\u{F360}",
    "ThinSpace" => "\u{2009}",
    "MediumSpace" => "\u{205F}",
    "ThickSpace" => "\u{2005}",
    "VeryThinSpace" => "\u{200A}",
    "NegativeVeryThinSpace" => "\u{F380}",
    "NegativeThinSpace" => "\u{F382}",
    "NegativeMediumSpace" => "\u{F383}",
    "NegativeThickSpace" => "\u{F384}",
    "InvisibleTimes" => "\u{2062}",
    "InvisibleComma" => "\u{F765}",
    "InvisibleApplication" => "\u{F76D}",
    // Placeholders the FrontEnd hangs a *prefix* or *postfix* script on, so
    // that `\!\(\*SuperscriptBox[\(\[InvisiblePrefixScriptBase]\), \(1\)]\)Σ`
    // typesets as `¹Σ`. They carry no glyph of their own, but they are
    // characters that `StringLength` counts — only drawing drops them.
    "InvisiblePrefixScriptBase" => "\u{F3B3}",
    "InvisiblePostfixScriptBase" => "\u{F3B4}",
    // Accented Latin letters (Latin-1 supplement). Wolfram names them by
    // base letter + diacritic, e.g. `\[CCedilla]` is ç, `\[ODoubleDot]` is
    // ö. Needed so imported text (e.g. "Curaçao") compares equal to source
    // written with the named-character escapes.
    "AGrave" => "\u{00E0}",
    "AAcute" => "\u{00E1}",
    "AHat" => "\u{00E2}",
    "ATilde" => "\u{00E3}",
    "ADoubleDot" => "\u{00E4}",
    "ARing" => "\u{00E5}",
    "AE" => "\u{00E6}",
    "CCedilla" => "\u{00E7}",
    "EGrave" => "\u{00E8}",
    "EAcute" => "\u{00E9}",
    "EHat" => "\u{00EA}",
    "EDoubleDot" => "\u{00EB}",
    "IGrave" => "\u{00EC}",
    "IAcute" => "\u{00ED}",
    "IHat" => "\u{00EE}",
    "IDoubleDot" => "\u{00EF}",
    "Eth" => "\u{00F0}",
    "NTilde" => "\u{00F1}",
    "OGrave" => "\u{00F2}",
    "OAcute" => "\u{00F3}",
    "OHat" => "\u{00F4}",
    "OTilde" => "\u{00F5}",
    "ODoubleDot" => "\u{00F6}",
    "OSlash" => "\u{00F8}",
    "UGrave" => "\u{00F9}",
    "UAcute" => "\u{00FA}",
    "UHat" => "\u{00FB}",
    "UDoubleDot" => "\u{00FC}",
    "YAcute" => "\u{00FD}",
    "Thorn" => "\u{00FE}",
    "YDoubleDot" => "\u{00FF}",
    "SZ" => "\u{00DF}",
    "CapitalAGrave" => "\u{00C0}",
    "CapitalAAcute" => "\u{00C1}",
    "CapitalAHat" => "\u{00C2}",
    "CapitalATilde" => "\u{00C3}",
    "CapitalADoubleDot" => "\u{00C4}",
    "CapitalARing" => "\u{00C5}",
    "CapitalAE" => "\u{00C6}",
    "CapitalCCedilla" => "\u{00C7}",
    "CapitalEGrave" => "\u{00C8}",
    "CapitalEAcute" => "\u{00C9}",
    "CapitalEHat" => "\u{00CA}",
    "CapitalEDoubleDot" => "\u{00CB}",
    "CapitalIGrave" => "\u{00CC}",
    "CapitalIAcute" => "\u{00CD}",
    "CapitalIHat" => "\u{00CE}",
    "CapitalIDoubleDot" => "\u{00CF}",
    "CapitalEth" => "\u{00D0}",
    "CapitalNTilde" => "\u{00D1}",
    "CapitalOGrave" => "\u{00D2}",
    "CapitalOAcute" => "\u{00D3}",
    "CapitalOHat" => "\u{00D4}",
    "CapitalOTilde" => "\u{00D5}",
    "CapitalODoubleDot" => "\u{00D6}",
    "CapitalOSlash" => "\u{00D8}",
    "CapitalUGrave" => "\u{00D9}",
    "CapitalUAcute" => "\u{00DA}",
    "CapitalUHat" => "\u{00DB}",
    "CapitalUDoubleDot" => "\u{00DC}",
    "CapitalYAcute" => "\u{00DD}",
    "CapitalThorn" => "\u{00DE}",
    _ => return None,
  })
}

/// Wolfram keeps a large part of its character set in the Unicode private use
/// area — `\[Rule]` is U+F522, `\[WarningSign]` is U+F725 — and only its own
/// fonts draw those code points, so a label written with one comes out as a
/// blank box everywhere else. Substitute the nearest standard character for
/// *drawing* only: the string itself, and everything `ToCharacterCode`
/// reports about it, keeps the code point Wolfram uses.
///
/// The characters that draw nothing at all in Wolfram (the invisible
/// operators, the script-base placeholders, the `Grid` span fillers)
/// substitute to the empty string, so they leave no artefact behind.
pub fn substitute_private_use_glyphs(s: &str) -> std::borrow::Cow<'_, str> {
  if !s.chars().any(|c| ('\u{E000}'..='\u{F8FF}').contains(&c)) {
    return std::borrow::Cow::Borrowed(s);
  }
  let mut out = String::with_capacity(s.len());
  for c in s.chars() {
    match private_use_glyph(c) {
      Some(sub) => out.push_str(sub),
      None => match script_letter_glyph(c) {
        Some(letter) => out.push(letter),
        None => out.push(c),
      },
    }
  }
  std::borrow::Cow::Owned(out)
}

/// The visible stand-in for a single private-use character, or `None` to
/// leave it alone. See [`substitute_private_use_glyphs`].
fn private_use_glyph(c: char) -> Option<&'static str> {
  Some(match c {
    '\u{F522}' | '\u{F3D5}' => "\u{2192}", // Rule, DirectedEdge: →
    '\u{F51F}' => "\u{29F4}",              // RuleDelayed: ⧴
    '\u{F3D4}' => "\u{2194}",              // UndirectedEdge: ↔
    '\u{F3D3}' => "\u{2223}",              // Conditioned: ∣
    '\u{F3D2}' => "\u{223C}",              // Distributed: ∼
    '\u{F4A0}' | '\u{F3C4}' => "\u{2A2F}", // Cross: ⨯
    '\u{F3DA}' => "\u{2297}",              // TensorProduct: ⊗
    '\u{F424}' => "\u{2270}",              // NotLessSlantEqual: ≰
    '\u{F429}' => "\u{2271}",              // NotGreaterSlantEqual: ≱
    '\u{F750}' => "\u{25CF}",              // FilledSmallCircle: ●
    '\u{F725}' => "\u{26A0}",              // WarningSign: ⚠
    '\u{F3DF}' => "\u{2641}",              // Earth: ♁
    '\u{F431}' => "=",                     // Equal
    '\u{F603}' | '\u{F604}' => "|",        // Left/RightBracketingBar: |
    '\u{F605}' | '\u{F606}' => "\u{2016}", // Left/RightDoubleBracketingBar: ‖
    '\u{F74B}' => "\u{2145}",              // CapitalDifferentialD: ⅅ
    '\u{F74C}' => "\u{2146}",              // DifferentialD: ⅆ
    '\u{F74D}' => "\u{212F}",              // ExponentialE: ℯ
    '\u{F74E}' => "\u{2148}",              // ImaginaryI: ⅈ
    '\u{F74F}' => "\u{2149}",              // ImaginaryJ: ⅉ
    '\u{F3A3}' => "\n",                    // IndentingNewLine
    // Characters with no glyph of their own.
    '\u{F360}'                            // InvisibleSpace
    | '\u{F380}' | '\u{F382}' | '\u{F383}' | '\u{F384}' // negative spaces
    | '\u{F765}'                          // InvisibleComma
    | '\u{F76D}'                          // InvisibleApplication
    | '\u{F3A0}'                          // Null
    | '\u{F3A2}'                          // invisible GridBox matrix bracket
    | '\u{F3B3}' | '\u{F3B4}'             // Invisible…ScriptBase
    | '\u{F3BA}' | '\u{F3BB}' | '\u{F3BC}' => "", // SpanFrom…
    _ => return None,
  })
}

/// The Mathematical Alphanumeric Symbols letter for one of Wolfram's
/// private-use script letters (`\[ScriptCapitalD]` is U+F773, not U+1D49F).
/// Both alphabets are contiguous, except that Unicode leaves the slots of
/// the script letters that already exist as letterlike symbols unassigned.
fn script_letter_glyph(c: char) -> Option<char> {
  const CAPITAL_GAPS: [(u32, char); 8] = [
    (1, 'ℬ'),
    (4, 'ℰ'),
    (5, 'ℱ'),
    (7, 'ℋ'),
    (8, 'ℐ'),
    (11, 'ℒ'),
    (12, 'ℳ'),
    (17, 'ℛ'),
  ];
  const SMALL_GAPS: [(u32, char); 4] =
    [(4, 'ℯ'), (6, 'ℊ'), (11, 'ℓ'), (14, 'ℴ')];
  let (offset, gaps, block) = match c {
    '\u{F770}'..='\u{F789}' => (c as u32 - 0xF770, &CAPITAL_GAPS[..], 0x1D49C),
    '\u{F6B2}'..='\u{F6CB}' => (c as u32 - 0xF6B2, &SMALL_GAPS[..], 0x1D4B6),
    _ => return None,
  };
  match gaps.iter().find(|(gap, _)| *gap == offset) {
    Some((_, letter)) => Some(*letter),
    None => char::from_u32(block + offset),
  }
}

/// Convert a Wolfram named character like `\[Phi]` to its Unicode equivalent.
fn named_char_to_expr(s: &str) -> Expr {
  // Direct Unicode character input
  match s {
    "\u{20AC}" => return Expr::Identifier("\u{20AC}".to_string()), // €
    "\u{03F5}" => return Expr::Identifier("\u{03F5}".to_string()), // ϵ
    "\u{03C0}" => return Expr::Constant("Pi".to_string()),         // π
    "\u{212F}" => return Expr::Constant("E".to_string()),          // ℯ
    "\u{2147}" => return Expr::Constant("E".to_string()), // ⅇ (DOUBLE-STRUCK ITALIC SMALL E)
    "\u{F74D}" => return Expr::Constant("E".to_string()), // Wolfram PUA \[ExponentialE]
    "\u{00B0}" => return Expr::Constant("Degree".to_string()), // °
    "\u{221E}" => return Expr::Identifier("Infinity".to_string()), // ∞
    "\u{2148}" => return Expr::Identifier("I".to_string()), // ⅈ
    "\u{F74E}" => return Expr::Identifier("I".to_string()), // Wolfram PUA \[ImaginaryI]
    "\u{F74F}" => return Expr::Identifier("I".to_string()), // Wolfram PUA \[ImaginaryJ]
    _ => {}
  }

  // Adjacent `\[Name]` segments (and trailing identifier chars)
  // concatenate into one identifier (e.g. `\[CapitalGamma]\[Beta]` → `Γβ`,
  // `\[Angle]XYZ` → `∠XYZ`), matching wolframscript's
  // identifier-character semantics. Detect any input with `\[`-segments
  // — including a single `\[Name]` followed by alphanumerics — and join
  // their Unicode mappings into one identifier.
  if s.starts_with("\\[") && (s.matches("\\[").count() > 1 || !s.ends_with(']'))
  {
    let mut joined = String::new();
    let mut rest = s;
    while !rest.is_empty() {
      if rest.starts_with("\\[") {
        let after = &rest[2..];
        if let Some(close) = after.find(']') {
          let name = &after[..close];
          if let Some(unicode) = named_char_to_unicode(name) {
            joined.push_str(unicode);
          } else {
            joined.push_str(name);
          }
          rest = &after[close + 1..];
        } else {
          break;
        }
      } else {
        // Take one char and continue
        let mut chars = rest.chars();
        if let Some(c) = chars.next() {
          joined.push(c);
          rest = chars.as_str();
        } else {
          break;
        }
      }
    }
    if !joined.is_empty() {
      return Expr::Identifier(joined);
    }
  }

  // Extract name from \[Name] syntax
  let name = if s.starts_with("\\[") && s.ends_with(']') {
    &s[2..s.len() - 1]
  } else {
    return Expr::Identifier(s.to_string());
  };

  // Special cases that map to constants
  match name {
    "Pi" => return Expr::Constant("Pi".to_string()),
    "ExponentialE" => return Expr::Constant("E".to_string()),
    "Degree" => return Expr::Constant("Degree".to_string()),
    _ => {}
  }

  // Special cases that map to known identifiers
  match name {
    "Infinity" => return Expr::Identifier("Infinity".to_string()),
    "ImaginaryI" | "ImaginaryJ" => return Expr::Identifier("I".to_string()),
    _ => {}
  }

  // Greek letters and other named characters → Unicode
  let unicode = match name {
    // Lowercase Greek
    "Alpha" => "\u{03B1}",
    "Beta" => "\u{03B2}",
    "Gamma" => "\u{03B3}",
    "Delta" => "\u{03B4}",
    "Epsilon" => "\u{03F5}",
    "Zeta" => "\u{03B6}",
    "Eta" => "\u{03B7}",
    "Theta" => "\u{03B8}",
    "Iota" => "\u{03B9}",
    "Kappa" => "\u{03BA}",
    "Lambda" => "\u{03BB}",
    "Mu" => "\u{03BC}",
    "Nu" => "\u{03BD}",
    "Xi" => "\u{03BE}",
    "Omicron" => "\u{03BF}",
    "Rho" => "\u{03C1}",
    "Sigma" => "\u{03C3}",
    "FinalSigma" => "\u{03C2}",
    "Tau" => "\u{03C4}",
    "Upsilon" => "\u{03C5}",
    "Phi" => "\u{03D5}",
    "CurlyPhi" => "\u{03C6}",
    "CurlyEpsilon" => "\u{03B5}",
    "CurlyTheta" => "\u{03D1}",
    "CurlyKappa" => "\u{03F0}",
    "CurlyPi" => "\u{03D6}",
    "CurlyRho" => "\u{03F1}",
    "Chi" => "\u{03C7}",
    "Psi" => "\u{03C8}",
    "Omega" => "\u{03C9}",
    // Uppercase Greek
    "CapitalAlpha" => "\u{0391}",
    "CapitalBeta" => "\u{0392}",
    "CapitalGamma" => "\u{0393}",
    "CapitalDelta" => "\u{0394}",
    "CapitalEpsilon" => "\u{0395}",
    "CapitalZeta" => "\u{0396}",
    "CapitalEta" => "\u{0397}",
    "CapitalTheta" => "\u{0398}",
    "CapitalIota" => "\u{0399}",
    "CapitalKappa" => "\u{039A}",
    "CapitalLambda" => "\u{039B}",
    "CapitalMu" => "\u{039C}",
    "CapitalNu" => "\u{039D}",
    "CapitalXi" => "\u{039E}",
    "CapitalOmicron" => "\u{039F}",
    "CapitalPi" => "\u{03A0}",
    "CapitalRho" => "\u{03A1}",
    "CapitalSigma" => "\u{03A3}",
    "CapitalTau" => "\u{03A4}",
    "CapitalUpsilon" => "\u{03A5}",
    "CapitalPhi" => "\u{03A6}",
    "CapitalChi" => "\u{03A7}",
    "CapitalPsi" => "\u{03A8}",
    "CapitalOmega" => "\u{03A9}",
    // Common symbols
    "Euro" => "\u{20AC}",
    "Micro" => "\u{00B5}",
    "Angstrom" => "\u{212B}",
    "HBar" => "\u{210F}",
    // Element/set operators (when used as identifiers)
    "Element" => "\u{2208}",
    "NotElement" => "\u{2209}",
    "ReverseElement" => "\u{220B}",
    // Edge operators (when used as identifiers)
    "DirectedEdge" => "\u{F3D5}",
    "UndirectedEdge" => "\u{F3D4}",
    "Distributed" => "\u{F3D2}",
    "Conditioned" => "\u{F3D3}",
    "Cross" => "\u{F4A0}",
    "TensorProduct" => "\u{F3DA}",
    // Any other named character that is a *letter* spells a symbol with
    // that letter, exactly as a Greek one does: `\[AAcute]` is the symbol
    // `á` and `\[ScriptX]` is the symbol Wolfram writes with its own script
    // x. Characters that are not letters (`\[Sum]`, `\[NewLine]`) are
    // operators or spacing and keep their name here.
    _ => {
      return match named_char_to_unicode(name) {
        Some(unicode)
          if !unicode.is_empty() && unicode.chars().all(is_symbol_letter) =>
        {
          Expr::Identifier(unicode.to_string())
        }
        _ => Expr::Identifier(name.to_string()),
      };
    }
  };
  Expr::Identifier(unicode.to_string())
}

/// Can this character spell a symbol on its own? Unicode letters can, and so
/// can the letters of Wolfram's own script alphabet, which live in the
/// private use area and are therefore not `Alphabetic`.
fn is_symbol_letter(c: char) -> bool {
  c.is_alphabetic()
    || ('\u{F6B2}'..='\u{F6CB}').contains(&c)
    || ('\u{F770}'..='\u{F789}').contains(&c)
    || is_letterlike_symbol_char(c)
}

/// Geometric shapes, pictographs, musical accidentals and astronomical
/// symbols that Wolfram treats as ordinary letterlike symbol names — e.g.
/// `Head[\[FilledSquare]]` is `Symbol`, not a syntax error — even though
/// Unicode files them under `So`/`Sm` rather than `L`. Mirrors
/// `LetterlikeSymbolChar` in wolfram.pest, which accepts the same
/// characters when they appear as the bare glyph rather than the `\[Name]`
/// escape.
fn is_letterlike_symbol_char(c: char) -> bool {
  matches!(
    c,
    '\u{2220}'
      | '\u{25A0}'
      | '\u{25A1}'
      | '\u{25FC}'
      | '\u{25FB}'
      | '\u{25AA}'
      | '\u{25AB}'
      | '\u{25CF}'
      | '\u{25CB}'
      | '\u{F750}'
      | '\u{25E6}'
      | '\u{25C6}'
      | '\u{25C7}'
      | '\u{25B2}'
      | '\u{25B3}'
      | '\u{25BC}'
      | '\u{25BD}'
      | '\u{25C0}'
      | '\u{25B6}'
      | '\u{F528}'
      | '\u{F527}'
      | '\u{F725}'
      | '\u{2713}'
      | '\u{231A}'
      | '\u{2605}'
      | '\u{2736}'
      | '\u{266F}'
      | '\u{266D}'
      | '\u{266E}'
      | '\u{2609}'
      | '\u{263F}'
      | '\u{2640}'
      | '\u{F3DF}'
      | '\u{2642}'
      | '\u{2643}'
      | '\u{2644}'
      | '\u{26E2}'
      | '\u{2646}'
      | '\u{2647}'
  )
}

/// Extract all child `Expr` nodes from a variant, leaving it childless.
/// Used by the iterative `Drop` implementation.
fn take_expr_children(expr: &mut Expr, stack: &mut Vec<Expr>) {
  match expr {
    // Leaf variants — no children
    Expr::Integer(_)
    | Expr::BigInteger(_)
    | Expr::Real(_)
    | Expr::BigFloat(_, _)
    | Expr::String(_)
    | Expr::Identifier(_)
    | Expr::Slot(_)
    | Expr::SlotSequence(_)
    | Expr::Pattern { .. }
    | Expr::Constant(_)
    | Expr::Raw(_)
    | Expr::Image { .. }
    | Expr::Graphics { .. } => {}

    // Vec<Expr> children
    Expr::CompoundExpr(children) => {
      stack.append(children);
    }
    Expr::List(children) => {
      stack.extend(std::mem::take(children));
    }
    Expr::FunctionCall { args, .. } => {
      stack.extend(std::mem::take(args));
    }
    Expr::Comparison { operands, .. } => {
      stack.append(operands);
    }

    // Box<Expr> children (two boxes)
    Expr::BinaryOp { left, right, .. } => {
      stack.push(*std::mem::replace(left, Box::new(Expr::Integer(0))));
      stack.push(*std::mem::replace(right, Box::new(Expr::Integer(0))));
    }
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      stack.push(*std::mem::replace(pattern, Box::new(Expr::Integer(0))));
      stack.push(*std::mem::replace(replacement, Box::new(Expr::Integer(0))));
    }
    Expr::ReplaceAll { expr, rules }
    | Expr::ReplaceRepeated { expr, rules } => {
      stack.push(*std::mem::replace(expr, Box::new(Expr::Integer(0))));
      stack.push(*std::mem::replace(rules, Box::new(Expr::Integer(0))));
    }
    Expr::Map { func, list }
    | Expr::Apply { func, list }
    | Expr::MapApply { func, list } => {
      stack.push(*std::mem::replace(func, Box::new(Expr::Integer(0))));
      stack.push(*std::mem::replace(list, Box::new(Expr::Integer(0))));
    }
    Expr::PrefixApply { func, arg } => {
      stack.push(*std::mem::replace(func, Box::new(Expr::Integer(0))));
      stack.push(*std::mem::replace(arg, Box::new(Expr::Integer(0))));
    }
    Expr::Postfix { expr, func } => {
      stack.push(*std::mem::replace(expr, Box::new(Expr::Integer(0))));
      stack.push(*std::mem::replace(func, Box::new(Expr::Integer(0))));
    }
    Expr::Part { expr, index } => {
      stack.push(*std::mem::replace(expr, Box::new(Expr::Integer(0))));
      stack.push(*std::mem::replace(index, Box::new(Expr::Integer(0))));
    }

    // Single Box<Expr>
    Expr::UnaryOp { operand, .. } => {
      stack.push(*std::mem::replace(operand, Box::new(Expr::Integer(0))));
    }
    Expr::Function { body } => {
      stack.push(*std::mem::replace(body, Box::new(Expr::Integer(0))));
    }
    Expr::NamedFunction { body, .. } => {
      stack.push(*std::mem::replace(body, Box::new(Expr::Integer(0))));
    }
    Expr::PatternOptional { default, .. } => {
      if let Some(d) = default {
        stack.push(*std::mem::replace(d, Box::new(Expr::Integer(0))));
      }
    }
    Expr::PatternTest { test, .. } => {
      stack.push(*std::mem::replace(test, Box::new(Expr::Integer(0))));
    }

    // Box<Expr> + Vec<Expr>
    Expr::CurriedCall { func, args } => {
      stack.push(*std::mem::replace(func, Box::new(Expr::Integer(0))));
      stack.append(args);
    }

    // Vec<(Expr, Expr)>
    Expr::Association(pairs) => {
      for (k, v) in pairs.drain(..) {
        stack.push(k);
        stack.push(v);
      }
    }
  }
}

impl Drop for Expr {
  fn drop(&mut self) {
    let mut work = Vec::new();
    take_expr_children(self, &mut work);
    while let Some(mut child) = work.pop() {
      take_expr_children(&mut child, &mut work);
      // child now has no Expr children, so its recursive drop is trivial
    }
  }
}

impl Clone for Expr {
  fn clone(&self) -> Self {
    // Fast path for leaf variants — avoid heap allocation
    match self {
      Self::Integer(n) => return Self::Integer(*n),
      Self::BigInteger(n) => return Self::BigInteger(n.clone()),
      Self::Real(f) => return Self::Real(*f),
      Self::BigFloat(s, p) => return Self::BigFloat(s.clone(), *p),
      Self::String(s) => return Self::String(s.clone()),
      Self::Identifier(s) => return Self::Identifier(s.clone()),
      Self::Slot(n) => return Self::Slot(*n),
      Self::SlotSequence(n) => return Self::SlotSequence(*n),
      Self::Pattern {
        name,
        head,
        blank_type,
      } => {
        return Self::Pattern {
          name: name.clone(),
          head: head.clone(),
          blank_type: *blank_type,
        };
      }
      Self::Constant(s) => return Self::Constant(s.clone()),
      Self::Raw(s) => return Self::Raw(s.clone()),
      Self::Image {
        color_space,
        width,
        height,
        channels,
        data,
        image_type,
      } => {
        return Self::Image {
          color_space: *color_space,
          width: *width,
          height: *height,
          channels: *channels,
          data: data.clone(),
          image_type: *image_type,
        };
      }
      Self::Graphics {
        svg,
        is_3d,
        source,
        head,
        structure,
      } => {
        return Self::Graphics {
          svg: svg.clone(),
          is_3d: *is_3d,
          source: source.clone(),
          head: head.clone(),
          structure: structure.clone(),
        };
      }
      _ => {} // fall through to iterative clone
    }

    // Iterative clone for non-leaf variants
    enum CloneTask<'a> {
      Visit(&'a Expr),
      Build(CloneFrame),
    }

    enum CloneFrame {
      List(usize),
      FunctionCall(String, usize),
      BinaryOp(BinaryOperator),
      UnaryOp(UnaryOperator),
      Comparison(Vec<ComparisonOp>, usize),
      CompoundExpr(usize),
      Association(usize),
      Rule,
      RuleDelayed,
      ReplaceAll,
      ReplaceRepeated,
      Map,
      Apply,
      MapApply,
      PrefixApply,
      Postfix,
      Part,
      CurriedCall(usize),
      Function,
      NamedFunction(Vec<String>, bool),
      PatternOptional(String, Option<String>, bool), // name, head, has_default
      PatternTest(String, Option<String>, u8),
    }

    let mut tasks: Vec<CloneTask> = vec![CloneTask::Visit(self)];
    let mut results: Vec<Self> = Vec::new();

    while let Some(task) = tasks.pop() {
      match task {
        CloneTask::Visit(expr) => match expr {
          // Leaf variants
          Self::Integer(n) => results.push(Self::Integer(*n)),
          Self::BigInteger(n) => results.push(Self::BigInteger(n.clone())),
          Self::Real(f) => results.push(Self::Real(*f)),
          Self::BigFloat(s, p) => results.push(Self::BigFloat(s.clone(), *p)),
          Self::String(s) => results.push(Self::String(s.clone())),
          Self::Identifier(s) => results.push(Self::Identifier(s.clone())),
          Self::Slot(n) => results.push(Self::Slot(*n)),
          Self::SlotSequence(n) => results.push(Self::SlotSequence(*n)),
          Self::Pattern {
            name,
            head,
            blank_type,
          } => results.push(Self::Pattern {
            name: name.clone(),
            head: head.clone(),
            blank_type: *blank_type,
          }),
          Self::Constant(s) => results.push(Self::Constant(s.clone())),
          Self::Raw(s) => results.push(Self::Raw(s.clone())),
          Self::Image {
            color_space,
            width,
            height,
            channels,
            data,
            image_type,
          } => results.push(Self::Image {
            color_space: *color_space,
            width: *width,
            height: *height,
            channels: *channels,
            data: data.clone(),
            image_type: *image_type,
          }),
          Self::Graphics {
            svg,
            is_3d,
            source,
            head,
            structure,
          } => results.push(Self::Graphics {
            svg: svg.clone(),
            is_3d: *is_3d,
            source: source.clone(),
            head: head.clone(),
            structure: structure.clone(),
          }),

          // Vec<Expr> children
          Self::List(children) => {
            let count = children.len();
            tasks.push(CloneTask::Build(CloneFrame::List(count)));
            for child in children.iter().rev() {
              tasks.push(CloneTask::Visit(child));
            }
          }
          Self::CompoundExpr(children) => {
            let count = children.len();
            tasks.push(CloneTask::Build(CloneFrame::CompoundExpr(count)));
            for child in children.iter().rev() {
              tasks.push(CloneTask::Visit(child));
            }
          }
          Self::FunctionCall { name, args } => {
            let count = args.len();
            tasks.push(CloneTask::Build(CloneFrame::FunctionCall(
              name.clone(),
              count,
            )));
            for child in args.iter().rev() {
              tasks.push(CloneTask::Visit(child));
            }
          }
          Self::Comparison {
            operands,
            operators,
          } => {
            let count = operands.len();
            tasks.push(CloneTask::Build(CloneFrame::Comparison(
              operators.clone(),
              count,
            )));
            for child in operands.iter().rev() {
              tasks.push(CloneTask::Visit(child));
            }
          }

          // Two Box<Expr> children
          Self::BinaryOp { op, left, right } => {
            tasks.push(CloneTask::Build(CloneFrame::BinaryOp(*op)));
            tasks.push(CloneTask::Visit(right));
            tasks.push(CloneTask::Visit(left));
          }
          Self::Rule {
            pattern,
            replacement,
          } => {
            tasks.push(CloneTask::Build(CloneFrame::Rule));
            tasks.push(CloneTask::Visit(replacement));
            tasks.push(CloneTask::Visit(pattern));
          }
          Self::RuleDelayed {
            pattern,
            replacement,
          } => {
            tasks.push(CloneTask::Build(CloneFrame::RuleDelayed));
            tasks.push(CloneTask::Visit(replacement));
            tasks.push(CloneTask::Visit(pattern));
          }
          Self::ReplaceAll { expr, rules } => {
            tasks.push(CloneTask::Build(CloneFrame::ReplaceAll));
            tasks.push(CloneTask::Visit(rules));
            tasks.push(CloneTask::Visit(expr));
          }
          Self::ReplaceRepeated { expr, rules } => {
            tasks.push(CloneTask::Build(CloneFrame::ReplaceRepeated));
            tasks.push(CloneTask::Visit(rules));
            tasks.push(CloneTask::Visit(expr));
          }
          Self::Map { func, list } => {
            tasks.push(CloneTask::Build(CloneFrame::Map));
            tasks.push(CloneTask::Visit(list));
            tasks.push(CloneTask::Visit(func));
          }
          Self::Apply { func, list } => {
            tasks.push(CloneTask::Build(CloneFrame::Apply));
            tasks.push(CloneTask::Visit(list));
            tasks.push(CloneTask::Visit(func));
          }
          Self::MapApply { func, list } => {
            tasks.push(CloneTask::Build(CloneFrame::MapApply));
            tasks.push(CloneTask::Visit(list));
            tasks.push(CloneTask::Visit(func));
          }
          Self::PrefixApply { func, arg } => {
            tasks.push(CloneTask::Build(CloneFrame::PrefixApply));
            tasks.push(CloneTask::Visit(arg));
            tasks.push(CloneTask::Visit(func));
          }
          Self::Postfix { expr, func } => {
            tasks.push(CloneTask::Build(CloneFrame::Postfix));
            tasks.push(CloneTask::Visit(func));
            tasks.push(CloneTask::Visit(expr));
          }
          Self::Part { expr, index } => {
            tasks.push(CloneTask::Build(CloneFrame::Part));
            tasks.push(CloneTask::Visit(index));
            tasks.push(CloneTask::Visit(expr));
          }

          // Single Box<Expr>
          Self::UnaryOp { op, operand } => {
            tasks.push(CloneTask::Build(CloneFrame::UnaryOp(*op)));
            tasks.push(CloneTask::Visit(operand));
          }
          Self::Function { body } => {
            tasks.push(CloneTask::Build(CloneFrame::Function));
            tasks.push(CloneTask::Visit(body));
          }
          Self::NamedFunction {
            params,
            body,
            bracketed,
          } => {
            tasks.push(CloneTask::Build(CloneFrame::NamedFunction(
              params.clone(),
              *bracketed,
            )));
            tasks.push(CloneTask::Visit(body));
          }
          Self::PatternOptional {
            name,
            head,
            default,
          } => {
            let has_default = default.is_some();
            tasks.push(CloneTask::Build(CloneFrame::PatternOptional(
              name.clone(),
              head.clone(),
              has_default,
            )));
            if let Some(d) = default {
              tasks.push(CloneTask::Visit(d));
            }
          }
          Self::PatternTest {
            name,
            head,
            blank_type,
            test,
          } => {
            tasks.push(CloneTask::Build(CloneFrame::PatternTest(
              name.clone(),
              head.clone(),
              *blank_type,
            )));
            tasks.push(CloneTask::Visit(test));
          }

          // Box<Expr> + Vec<Expr>
          Self::CurriedCall { func, args } => {
            let count = args.len();
            // Build needs: func first, then count args
            tasks.push(CloneTask::Build(CloneFrame::CurriedCall(count)));
            for child in args.iter().rev() {
              tasks.push(CloneTask::Visit(child));
            }
            tasks.push(CloneTask::Visit(func));
          }

          // Vec<(Expr, Expr)>
          Self::Association(pairs) => {
            let count = pairs.len();
            tasks.push(CloneTask::Build(CloneFrame::Association(count)));
            // Push pairs in reverse; each pair = (key, value)
            for (k, v) in pairs.iter().rev() {
              tasks.push(CloneTask::Visit(v));
              tasks.push(CloneTask::Visit(k));
            }
          }
        },

        CloneTask::Build(frame) => {
          let expr = match frame {
            CloneFrame::List(count) => {
              let children: crate::ExprList =
                results.drain(results.len() - count..).collect();
              Self::List(children)
            }
            CloneFrame::CompoundExpr(count) => {
              let children: Vec<Self> =
                results.drain(results.len() - count..).collect();
              Self::CompoundExpr(children)
            }
            CloneFrame::FunctionCall(name, count) => {
              let args: crate::ExprList =
                results.drain(results.len() - count..).collect();
              Self::FunctionCall { name, args }
            }
            CloneFrame::Comparison(operators, count) => {
              let operands: Vec<Self> =
                results.drain(results.len() - count..).collect();
              Self::Comparison {
                operands,
                operators,
              }
            }
            CloneFrame::BinaryOp(op) => {
              let right = Box::new(results.pop().unwrap());
              let left = Box::new(results.pop().unwrap());
              Self::BinaryOp { op, left, right }
            }
            CloneFrame::UnaryOp(op) => {
              let operand = Box::new(results.pop().unwrap());
              Self::UnaryOp { op, operand }
            }
            CloneFrame::Rule => {
              let replacement = Box::new(results.pop().unwrap());
              let pattern = Box::new(results.pop().unwrap());
              Self::Rule {
                pattern,
                replacement,
              }
            }
            CloneFrame::RuleDelayed => {
              let replacement = Box::new(results.pop().unwrap());
              let pattern = Box::new(results.pop().unwrap());
              Self::RuleDelayed {
                pattern,
                replacement,
              }
            }
            CloneFrame::ReplaceAll => {
              let rules = Box::new(results.pop().unwrap());
              let expr = Box::new(results.pop().unwrap());
              Self::ReplaceAll { expr, rules }
            }
            CloneFrame::ReplaceRepeated => {
              let rules = Box::new(results.pop().unwrap());
              let expr = Box::new(results.pop().unwrap());
              Self::ReplaceRepeated { expr, rules }
            }
            CloneFrame::Map => {
              let list = Box::new(results.pop().unwrap());
              let func = Box::new(results.pop().unwrap());
              Self::Map { func, list }
            }
            CloneFrame::Apply => {
              let list = Box::new(results.pop().unwrap());
              let func = Box::new(results.pop().unwrap());
              Self::Apply { func, list }
            }
            CloneFrame::MapApply => {
              let list = Box::new(results.pop().unwrap());
              let func = Box::new(results.pop().unwrap());
              Self::MapApply { func, list }
            }
            CloneFrame::PrefixApply => {
              let arg = Box::new(results.pop().unwrap());
              let func = Box::new(results.pop().unwrap());
              Self::PrefixApply { func, arg }
            }
            CloneFrame::Postfix => {
              let func = Box::new(results.pop().unwrap());
              let expr = Box::new(results.pop().unwrap());
              Self::Postfix { expr, func }
            }
            CloneFrame::Part => {
              let index = Box::new(results.pop().unwrap());
              let expr = Box::new(results.pop().unwrap());
              Self::Part { expr, index }
            }
            CloneFrame::Function => {
              let body = Box::new(results.pop().unwrap());
              Self::Function { body }
            }
            CloneFrame::NamedFunction(params, bracketed) => {
              let body = Box::new(results.pop().unwrap());
              Self::NamedFunction {
                params,
                body,
                bracketed,
              }
            }
            CloneFrame::PatternOptional(name, head, has_default) => {
              let default = if has_default {
                Some(Box::new(results.pop().unwrap()))
              } else {
                None
              };
              Self::PatternOptional {
                name,
                head,
                default,
              }
            }
            CloneFrame::PatternTest(name, head, blank_type) => {
              let test = Box::new(results.pop().unwrap());
              Self::PatternTest {
                name,
                head,
                blank_type,
                test,
              }
            }
            CloneFrame::CurriedCall(count) => {
              let args: Vec<Self> =
                results.drain(results.len() - count..).collect();
              let func = Box::new(results.pop().unwrap());
              Self::CurriedCall { func, args }
            }
            CloneFrame::Association(count) => {
              let mut pairs = Vec::with_capacity(count);
              let start = results.len() - count * 2;
              let flat: Vec<Self> = results.drain(start..).collect();
              let mut iter = flat.into_iter();
              for _ in 0..count {
                let k = iter.next().unwrap();
                let v = iter.next().unwrap();
                pairs.push((k, v));
              }
              Self::Association(pairs)
            }
          };
          results.push(expr);
        }
      }
    }

    results.pop().unwrap()
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
  Plus,
  Minus,
  Times,
  Divide,
  Power,
  And,
  Or,
  StringJoin,
  Alternatives,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOperator {
  Minus,
  Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOp {
  Equal,        // ==
  NotEqual,     // !=
  Less,         // <
  LessEqual,    // <=
  Greater,      // >
  GreaterEqual, // >=
  SameQ,        // ===
  UnsameQ,      // =!=
}

impl ComparisonOp {
  /// FullForm head name for this comparison operator (e.g. `!=` -> "Unequal").
  pub fn head_name(&self) -> &'static str {
    match self {
      Self::Equal => "Equal",
      Self::NotEqual => "Unequal",
      Self::Less => "Less",
      Self::LessEqual => "LessEqual",
      Self::Greater => "Greater",
      Self::GreaterEqual => "GreaterEqual",
      Self::SameQ => "SameQ",
      Self::UnsameQ => "UnsameQ",
    }
  }
}

/// Decompose a Comparison chain into its equivalent (head, args) exactly as WL
/// sees it: a uniform chain `a op b op c` is `Op[a, b, c]`, while a mixed chain
/// like `a < b <= c` is `Inequality[a, Less, b, LessEqual, c]` (operands and
/// operator symbols interleaved). Used by structural operations (Length, Part,
/// Apply) so they observe the same arity as wolframscript.
pub fn comparison_head_and_args(
  operands: &[Expr],
  operators: &[ComparisonOp],
) -> (String, Vec<Expr>) {
  let all_same = operators.windows(2).all(|w| w[0] == w[1]);
  if all_same {
    let head = operators
      .first()
      .map_or("Equal", ComparisonOp::head_name)
      .to_string();
    (head, operands.to_vec())
  } else {
    let mut args = Vec::with_capacity(operands.len() + operators.len());
    for (i, operand) in operands.iter().enumerate() {
      args.push(operand.clone());
      if let Some(op) = operators.get(i) {
        args.push(Expr::Identifier(op.head_name().to_string()));
      }
    }
    ("Inequality".to_string(), args)
  }
}

use crate::Rule;
use pest::iterators::Pair;

/// Extract the exponent expression from an `ImplicitPowerSuffix` pair.
/// Apply one `[[…]]` bracket group to a base expression, as one of a run of
/// groups. Within a group the specs nest as `Expr::Part`s, which is what the
/// evaluator reads as "each spec picks inside what the one before it picked"
/// — `m[[All, 2]]` is the second element of every row.
///
/// A group that is *followed* by another is emitted in the `Part[base, …]`
/// call form instead. That form evaluates to a value before the next group
/// sees it, which is the other reading: `m[[{1, 3}]][[2]]` is the second of
/// the two rows picked, not the second element of each. Both shapes parsed
/// the same before, so a chain of groups quietly took the first meaning.
fn apply_part_group(base: Expr, group: Pair<Rule>, is_last: bool) -> Expr {
  let specs: Vec<Expr> = group.into_inner().map(pair_to_expr).collect();
  if specs.is_empty() {
    return base;
  }
  if is_last {
    return specs.into_iter().fold(base, |acc, idx| Expr::Part {
      expr: Box::new(acc),
      index: Box::new(idx),
    });
  }
  let mut args = vec![base];
  args.extend(specs);
  Expr::FunctionCall {
    name: "Part".to_string(),
    args: args.into(),
  }
}

/// Apply every bracket group of a `PartIndexSuffix`, in order.
fn apply_part_index_suffix(base: Expr, suffix: Pair<Rule>) -> Expr {
  apply_part_groups(base, suffix.into_inner().collect())
}

/// Apply a run of `PartIndexGroup` pairs to a base expression.
fn apply_part_groups(base: Expr, groups: Vec<Pair<Rule>>) -> Expr {
  let last = groups.len();
  groups
    .into_iter()
    .enumerate()
    .fold(base, |acc, (i, g)| apply_part_group(acc, g, i + 1 == last))
}

/// Handles `^expr`, `^-expr`, and forms with a `PartIndexSuffix` after the
/// term (e.g. `^m[[1]]` → `Part[m, 1]`, `^-m[[1]]` → `-Part[m, 1]`).
fn implicit_power_exponent(pair: Pair<Rule>) -> Expr {
  let inners: Vec<_> = pair.into_inner().collect();
  let first = inners[0].clone();
  let mut expr = match first.as_rule() {
    Rule::NegSimpleTerm => {
      let neg_inners: Vec<_> = first.into_inner().collect();
      let mut base = pair_to_expr(neg_inners[0].clone());
      for p in neg_inners.iter().skip(1) {
        if p.as_rule() == Rule::PartIndexSuffix {
          base = apply_part_index_suffix(base, p.clone());
        }
      }
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand: Box::new(base),
      }
    }
    _ => pair_to_expr(first),
  };
  for p in inners.iter().skip(1) {
    if p.as_rule() == Rule::PartIndexSuffix {
      expr = apply_part_index_suffix(expr, p.clone());
    } else if p.as_rule() == Rule::FactorialSuffix {
      // `a^b!` parses as `a^(b!)` because Factorial binds tighter than Power.
      let func_name = if p.as_str() == "!!" {
        "Factorial2"
      } else {
        "Factorial"
      };
      expr = Expr::FunctionCall {
        name: func_name.to_string(),
        args: vec![expr].into(),
      };
    }
  }
  expr
}

/// True if an AssociationItem source slice uses `:>` as its separator
/// rather than `->`. The grammar accepts both but doesn't expose which one
/// matched, so we scan the raw text for a `:>` outside of brackets.
fn is_assoc_item_delayed(s: &str) -> bool {
  let bytes = s.as_bytes();
  let mut depth: i32 = 0;
  let mut i = 0;
  while i < bytes.len() {
    match bytes[i] as char {
      '[' | '{' | '(' | '<' => depth += 1,
      ']' | '}' | ')' | '>' => depth -= 1,
      ':' if depth == 0 && i + 1 < bytes.len() && bytes[i + 1] == b'>' => {
        return true;
      }
      _ => {}
    }
    i += 1;
  }
  false
}

/// Interpret the box source carried by a `\!\(\*boxes\)` escape back into
/// the expression those boxes typeset.
///
/// `InputForm` writes a typeset expression this way, so this is the read
/// side of that round trip — without it `TraditionalForm[expr]` serialized
/// into a string (as Woxi Studio does with a Manipulate body) parsed back
/// as an opaque `HoldComplete` of its own source.
///
/// The escape yields the expression the boxes *typeset*, with every
/// display-only wrapper gone: `\!\(\*FormBox[SqrtBox["x"], TraditionalForm]\)`
/// is `Sqrt[x]`, not `TraditionalForm[Sqrt[x]]`, just as `\!\(\*StyleBox["x",
/// Bold]\)` is plain `x`. The form tag only has to name a member of
/// `$BoxForms`; anything else is a `MakeExpression::boxfmt` error in
/// Wolfram, so the escape is left as literal source here.
fn box_escape_to_expr(box_src: &str) -> Option<Expr> {
  let src = box_src.trim();
  if let Some(rest) = src.strip_prefix("FormBox[")
    && let Some(inner) = rest.strip_suffix(']')
  {
    // The form name is the last top-level argument; everything before it
    // is the boxes being displayed.
    let (boxes, form) = split_last_top_level_comma(inner)?;
    if !matches!(form.trim(), "TraditionalForm" | "StandardForm") {
      return None;
    }
    return box_escape_to_expr(boxes)
      .or_else(|| string_to_expr(boxes.trim()).ok());
  }
  // A bare string box carries source *text*, not a string value: the box
  // `"3"` is the number 3, `"x + 1"` the sum, and only a box whose text is
  // itself quoted (`"\"3\""`) is the string "3". Text that does not parse
  // on its own falls through to the box readers below.
  if let Some(inner) = src.strip_prefix('"').and_then(|r| r.strip_suffix('"'))
    && let Ok(parsed) = string_to_expr(inner)
  {
    return Some(parsed);
  }
  let text = crate::notebook::box_source_to_expression(src)?;
  string_to_expr(&text).ok()
}

/// Is every `"` in this box source backslash-escaped?
///
/// That is the spelling `escape_string_for_input_form` produces when a box
/// escape is itself written out as the InputForm of a *string* — there the
/// box delimiters are escaped along with everything else, so what is left is
/// no longer box syntax and the reader rejects it. Box data as a real `.nb`
/// file writes it always leaves those delimiters bare and uses `\"` only
/// *inside* a box, to mark it as a string literal (`"\"a, b\""`) rather than
/// source text (`"a + b"`). So one bare `"` settles it.
fn box_source_is_double_escaped(s: &str) -> bool {
  let mut saw_quote = false;
  let mut backslashes = 0usize;
  for c in s.chars() {
    match c {
      '\\' => backslashes += 1,
      '"' => {
        if backslashes.is_multiple_of(2) {
          return false;
        }
        saw_quote = true;
        backslashes = 0;
      }
      _ => backslashes = 0,
    }
  }
  saw_quote
}

/// The `\!\(\*boxes\)` escape in `src` whose box source is not box syntax,
/// or `None` when every escape in it reads.
///
/// A box escape is read when the *source* is read, so an unreadable one is a
/// syntax error rather than a value — that is what makes writing a box
/// escape out as the InputForm of a string a one-way trip: the spelling that
/// comes back has its box delimiters escaped (`List[\"…\"]`) and no longer
/// parses. The escape is returned whole, to be quoted in the message.
pub(crate) fn double_escaped_box_escape(src: &str) -> Option<String> {
  let chars: Vec<char> = src.chars().collect();
  let mut i = 0;
  while i < chars.len() {
    let Some(open) = box_escape_opener(&chars, i) else {
      i += 1;
      continue;
    };
    let close = box_escape_closer(&chars, i + open)?;
    let box_src: String = chars[i + open..close].iter().collect();
    if box_source_is_double_escaped(&box_src) {
      let end = close + if chars[close] == '\\' { 2 } else { 1 };
      return Some(chars[i..end].iter().collect());
    }
    i = close;
  }
  None
}

/// The length in chars of the `\!\(\*` opener at `i`, in either of its
/// spellings (backslashes as read from source, private-use markers as a
/// string carries them), or `None` when no escape starts there.
fn box_escape_opener(chars: &[char], i: usize) -> Option<usize> {
  use crate::functions::string_ast::{BOX_OPEN, BOX_SEP, BOX_START};
  let mut len = 0;
  for (backslashed, marker) in
    [('!', BOX_START), ('(', BOX_OPEN), ('*', BOX_SEP)]
  {
    match chars.get(i + len) {
      Some(&c) if c == marker => len += 1,
      Some('\\') if chars.get(i + len + 1) == Some(&backslashed) => len += 2,
      _ => return None,
    }
  }
  Some(len)
}

/// The index of the `\)` that closes the escape opened before `from`.
fn box_escape_closer(chars: &[char], from: usize) -> Option<usize> {
  use crate::functions::string_ast::{BOX_CLOSE, BOX_OPEN};
  let mut depth = 0usize;
  let mut i = from;
  while i < chars.len() {
    let (opens, closes, width) = match (chars[i], chars.get(i + 1)) {
      (BOX_OPEN, _) => (true, false, 1),
      (BOX_CLOSE, _) => (false, true, 1),
      ('\\', Some('(')) => (true, false, 2),
      ('\\', Some(')')) => (false, true, 2),
      _ => (false, false, 1),
    };
    if closes {
      if depth == 0 {
        return Some(i);
      }
      depth -= 1;
    }
    if opens {
      depth += 1;
    }
    i += width;
  }
  None
}

/// Split `s` at its last top-level comma (one not nested in brackets or
/// inside a string). `None` when there is no such comma.
fn split_last_top_level_comma(s: &str) -> Option<(&str, &str)> {
  let mut depth = 0i32;
  let mut in_string = false;
  let mut prev_backslash = false;
  let mut last = None;
  for (i, c) in s.char_indices() {
    if in_string {
      if c == '"' && !prev_backslash {
        in_string = false;
      }
      prev_backslash = c == '\\' && !prev_backslash;
      continue;
    }
    match c {
      '"' => in_string = true,
      '[' | '{' | '(' => depth += 1,
      ']' | '}' | ')' => depth -= 1,
      ',' if depth == 0 => last = Some(i),
      _ => {}
    }
  }
  last.map(|i| (&s[..i], &s[i + 1..]))
}

/// Strip the `\(` … `\)` delimiters off a box-notation literal. Both
/// spellings of the pair are accepted: the backslash escapes source is
/// written with, and the private-use markers `ToString[InputForm[…]]`
/// writes for a typeset expression (see the `BoxLeft`/`BoxRight` rules in
/// wolfram.pest). Returns the input unchanged when it carries no
/// delimiters.
fn strip_box_delimiters(raw: &str) -> &str {
  use crate::functions::string_ast::{BOX_CLOSE, BOX_OPEN};
  let Some(inner) = raw
    .strip_prefix("\\(")
    .or_else(|| raw.strip_prefix(BOX_OPEN))
  else {
    return raw;
  };
  inner
    .strip_suffix("\\)")
    .or_else(|| inner.strip_suffix(BOX_CLOSE))
    .unwrap_or(raw)
}

/// Parse a `\(...\)` box-notation literal into an explicit *Box AST. Atoms
/// inside become String literals; `\^`, `\_`, `\+`, `\&`, `\@`, `\%`
/// translate to SuperscriptBox/SubscriptBox/UnderscriptBox/OverscriptBox/
/// SqrtBox/UnderoverscriptBox respectively. `\^` and `\_` are right-
/// associative; `\%` extends the immediately preceding `\+`/`\&` into an
/// UnderoverscriptBox. Returns None on unrecognised shapes so the caller
/// can fall back to a HoldComplete wrapper.
fn parse_box_notation_str(raw: &str) -> Option<Expr> {
  let stripped = strip_box_delimiters(raw);
  let inner = stripped.trim();
  if inner.is_empty() {
    return None;
  }
  let toks = tokenize_box(inner)?;
  // First try the operator-aware parser (SuperscriptBox/SubscriptBox/…) on a
  // single chain. If that consumes everything, return its result directly.
  if let Some((expr, idx)) = parse_box_chain(&toks, 0)
    && idx == toks.len()
  {
    return Some(expr);
  }
  // Otherwise fall through to a generic RowBox builder that handles plain
  // atom sequences like `\(c (1 + x)\)` → `RowBox[{c, RowBox[{(, RowBox[{1,
  // +, x}], )}]}]`. Returns `None` if the input isn't expressible as
  // RowBox-of-strings (e.g. unmatched parens or stray box operators).
  parse_box_rowbox(&toks)
}

/// Build a generic `RowBox[{…}]` from a token list. Recognises balanced `(` /
/// `)` Atom pairs as nested groups so the parens themselves stay as their own
/// string atoms inside the surrounding RowBox.
fn parse_box_rowbox(toks: &[BoxTok]) -> Option<Expr> {
  // Split the top-level token list at every balanced `(`/`)` pair, recursing
  // on the inside so each nested group becomes its own RowBox.
  let mut parts: Vec<Expr> = Vec::new();
  let mut i = 0;
  while i < toks.len() {
    match &toks[i] {
      BoxTok::Atom(s) if s == "(" => {
        // Find matching `)`.
        let mut depth = 1usize;
        let mut j = i + 1;
        while j < toks.len() && depth > 0 {
          match &toks[j] {
            BoxTok::Atom(s) if s == "(" => depth += 1,
            BoxTok::Atom(s) if s == ")" => depth -= 1,
            _ => {}
          }
          if depth == 0 {
            break;
          }
          j += 1;
        }
        if j >= toks.len() {
          return None;
        }
        let inner_toks = &toks[i + 1..j];
        let inner_expr = parse_box_rowbox(inner_toks)?;
        // Build the parenthesised group as its own RowBox of `(`, inner, `)`.
        let group = box_call(
          "RowBox",
          vec![Expr::List(
            vec![
              Expr::String("(".to_string()),
              inner_expr,
              Expr::String(")".to_string()),
            ]
            .into(),
          )],
        );
        parts.push(group);
        i = j + 1;
      }
      _ => {
        // Try `parse_box_chain` first so operators like `\/` (and
        // `\^`/`\_`) can combine with neighbouring units (e.g.
        // `x \/ y` inside `\(x \/ y + z\)` becomes
        // `FractionBox[x, y]`).
        if let Some((chained, new_i)) = parse_box_chain(toks, i)
          && new_i > i + 1
        {
          parts.push(chained);
          i = new_i;
          continue;
        }
        let unit = box_unit(&toks[i])?;
        parts.push(unit);
        i += 1;
      }
    }
  }
  if parts.is_empty() {
    return None;
  }
  if parts.len() == 1 {
    return Some(parts.into_iter().next().unwrap());
  }
  Some(box_call("RowBox", vec![Expr::List(parts.into())]))
}

#[derive(Debug)]
enum BoxTok {
  Atom(String),
  Op(char),
  Nested(Expr),
}

fn tokenize_box(s: &str) -> Option<Vec<BoxTok>> {
  let mut toks: Vec<BoxTok> = Vec::new();
  let bytes = s.as_bytes();
  let mut i = 0;
  while i < bytes.len() {
    let c = bytes[i];
    if (c as char).is_whitespace() {
      i += 1;
      continue;
    }
    if c == b'\\' && i + 1 < bytes.len() {
      let n = bytes[i + 1];
      if matches!(n, b'^' | b'_' | b'+' | b'&' | b'%' | b'@' | b'/' | b'`') {
        toks.push(BoxTok::Op(n as char));
        i += 2;
        continue;
      }
      if n == b'(' {
        // Find matching `\)` honouring nested `\(`.
        let start = i;
        let mut depth = 1;
        i += 2;
        while i < bytes.len() && depth > 0 {
          if i + 1 < bytes.len() && bytes[i] == b'\\' {
            if bytes[i + 1] == b'(' {
              depth += 1;
              i += 2;
              continue;
            }
            if bytes[i + 1] == b')' {
              depth -= 1;
              i += 2;
              continue;
            }
          }
          i += 1;
        }
        if depth != 0 {
          return None;
        }
        let nested = parse_box_notation_str(&s[start..i])?;
        toks.push(BoxTok::Nested(nested));
        continue;
      }
    }
    // Single-character punctuation that wolframscript prints inside the
    // RowBox sequence as its own string atom (e.g. `(`, `)`, `+`, `-`,
    // arithmetic operators in `\(c (1 + x)\)` → `RowBox[{c, RowBox[{(,
    // RowBox[{1, +, x}], )}]}]`).
    if matches!(c, b'(' | b')' | b'+' | b'-' | b'*' | b'/' | b'=' | b',') {
      toks.push(BoxTok::Atom((c as char).to_string()));
      i += 1;
      continue;
    }
    let start = i;
    while i < bytes.len() {
      let c = bytes[i];
      if (c as char).is_whitespace() {
        break;
      }
      if matches!(c, b'(' | b')' | b'+' | b'-' | b'*' | b'/' | b'=' | b',') {
        break;
      }
      if c == b'\\'
        && i + 1 < bytes.len()
        && matches!(
          bytes[i + 1],
          b'^' | b'_' | b'+' | b'&' | b'%' | b'@' | b'(' | b')'
        )
      {
        break;
      }
      i += 1;
    }
    if i > start {
      toks.push(BoxTok::Atom(s[start..i].to_string()));
    }
  }
  Some(toks)
}

fn box_unit(tok: &BoxTok) -> Option<Expr> {
  match tok {
    BoxTok::Atom(s) => Some(Expr::String(s.clone())),
    BoxTok::Nested(e) => Some(e.clone()),
    BoxTok::Op(_) => None,
  }
}

/// Consume one "box unit" from `toks[start..]`, expanding a balanced
/// `(...)` group into `RowBox[{"(", inner, ")"}]`. Returns the
/// resulting Expr together with the index just past the consumed
/// tokens.
fn box_unit_or_group(toks: &[BoxTok], start: usize) -> Option<(Expr, usize)> {
  if start >= toks.len() {
    return None;
  }
  if matches!(&toks[start], BoxTok::Atom(s) if s == "(") {
    // Find matching `)`.
    let mut depth = 1usize;
    let mut j = start + 1;
    while j < toks.len() && depth > 0 {
      match &toks[j] {
        BoxTok::Atom(s) if s == "(" => depth += 1,
        BoxTok::Atom(s) if s == ")" => {
          depth -= 1;
          if depth == 0 {
            break;
          }
        }
        _ => {}
      }
      j += 1;
    }
    if j >= toks.len() {
      return None;
    }
    let inner = parse_box_rowbox(&toks[start + 1..j])?;
    let group = box_call(
      "RowBox",
      vec![Expr::List(
        vec![
          Expr::String("(".to_string()),
          inner,
          Expr::String(")".to_string()),
        ]
        .into(),
      )],
    );
    return Some((group, j + 1));
  }
  let unit = box_unit(&toks[start])?;
  Some((unit, start + 1))
}

fn box_call(name: &str, args: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args: args.into(),
  }
}

fn parse_box_chain(toks: &[BoxTok], start: usize) -> Option<(Expr, usize)> {
  if start >= toks.len() {
    return None;
  }
  // Prefix `\@` (SqrtBox).
  if matches!(toks[start], BoxTok::Op('@')) {
    let (arg, end) = parse_box_chain(toks, start + 1)?;
    return Some((box_call("SqrtBox", vec![arg]), end));
  }
  let lhs = box_unit(&toks[start])?;
  parse_box_continued(lhs, toks, start + 1)
}

fn parse_box_continued(
  mut lhs: Expr,
  toks: &[BoxTok],
  mut idx: usize,
) -> Option<(Expr, usize)> {
  while idx < toks.len() {
    match &toks[idx] {
      BoxTok::Op('^' | '_') => {
        let op = match &toks[idx] {
          BoxTok::Op(c) => *c,
          _ => unreachable!(),
        };
        // Right-associative: full chain on the right.
        let (rhs, end) = parse_box_chain(toks, idx + 1)?;
        let head = if op == '^' {
          "SuperscriptBox"
        } else {
          "SubscriptBox"
        };
        return Some((box_call(head, vec![lhs, rhs]), end));
      }
      BoxTok::Op('/') => {
        // `\/` → FractionBox[lhs, rhs]. Binds tighter than the
        // surrounding RowBox so `\(x \/ y + z\)` parses as
        // `RowBox[{FractionBox["x", "y"], "+", "z"}]`. The rhs is a
        // single "box unit" — a token or a balanced `(...)` group
        // that recursively becomes `RowBox[{"(", inner, ")"}]`
        // (regression for `\(x \/ (y + z)\)`).
        let (rhs, end) = box_unit_or_group(toks, idx + 1)?;
        lhs = box_call("FractionBox", vec![lhs, rhs]);
        idx = end;
      }
      BoxTok::Op('`') => {
        // `\`` → FormBox[<rest>, lhs]. The body consumes the entire
        // remaining chain, e.g. `\(TraditionalForm \` a + b\)` →
        // `FormBox[RowBox[{a, +, b}], TraditionalForm]`. lhs is
        // converted from a quoted String token into the bare form
        // identifier (wolframscript's FormBox tag is unquoted).
        let body = parse_box_rowbox(&toks[idx + 1..])?;
        let form_tag = match &lhs {
          Expr::String(s) => Expr::Identifier(s.clone()),
          other => other.clone(),
        };
        return Some((box_call("FormBox", vec![body, form_tag]), toks.len()));
      }
      BoxTok::Op('+' | '&') => {
        let op = match &toks[idx] {
          BoxTok::Op(c) => *c,
          _ => unreachable!(),
        };
        // Read a single unit; `\%` may follow to extend into Underover.
        if idx + 1 >= toks.len() {
          return None;
        }
        let rhs = box_unit(&toks[idx + 1])?;
        let mut end = idx + 2;
        if end < toks.len() && matches!(toks[end], BoxTok::Op('%')) {
          if end + 1 >= toks.len() {
            return None;
          }
          let third = box_unit(&toks[end + 1])?;
          end += 2;
          let combined = if op == '+' {
            box_call("UnderoverscriptBox", vec![lhs, rhs, third])
          } else {
            box_call("UnderoverscriptBox", vec![lhs, third, rhs])
          };
          lhs = combined;
        } else {
          let head = if op == '+' {
            "UnderscriptBox"
          } else {
            "OverscriptBox"
          };
          lhs = box_call(head, vec![lhs, rhs]);
        }
        idx = end;
      }
      _ => break,
    }
  }
  Some((lhs, idx))
}

/// Convert a pest Pair to an owned Expr AST.
/// This is used to store function bodies without re-parsing.
pub fn pair_to_expr(pair: Pair<Rule>) -> Expr {
  // Handle some cases that recurse to avoid large stack usage.
  match pair.as_rule() {
    Rule::NumericValue | Rule::UnsignedNumericValue => {
      let inner = pair.into_inner().next().unwrap();
      pair_to_expr(inner)
    }
    // A definition whose left side carries a pattern (`f[x_] := rhs`) parses
    // as its own rule, handled at the top level by `store_function_definition`
    // on the pest pair. Nested uses (inside `ToExpression`, `ReleaseHold`, …)
    // reach `pair_to_expr` instead, so give it the `SetDelayed[lhs, rhs]` form
    // it means — without this it fell through to a shape that looped forever
    // when evaluated.
    Rule::FunctionDefinition => {
      let mut inner = pair.into_inner();
      let name = inner.next().map(|p| p.as_str().to_string());
      let mut parts: Vec<Expr> = inner.map(pair_to_expr).collect();
      match (name, parts.pop()) {
        (Some(name), Some(rhs)) => Expr::FunctionCall {
          name: "SetDelayed".to_string(),
          args: vec![
            Expr::FunctionCall {
              name,
              args: parts.into(),
            },
            rhs,
          ]
          .into(),
        },
        _ => Expr::Identifier("Null".to_string()),
      }
    }
    // `tag /: lhs := rhs` and friends parse as their own rules, handled at
    // the top level on the pest pair. Reaching `pair_to_expr` instead (a
    // notebook cell evaluated through `interpret_to_expr`, `ToExpression`,
    // …) used to fall back to `Expr::Raw`, which re-parsed to the same
    // `Raw` and looped forever. Give it the call form it means.
    Rule::TagSetDelayed | Rule::TagSet | Rule::TagUnset => {
      let name = match pair.as_rule() {
        Rule::TagSetDelayed => "TagSetDelayed",
        Rule::TagSet => "TagSet",
        _ => "TagUnset",
      };
      let args: Vec<Expr> = pair.into_inner().map(pair_to_expr).collect();
      Expr::FunctionCall {
        name: name.to_string(),
        args: args.into(),
      }
    }
    Rule::List => parse_list(pair),
    Rule::ListExtended => parse_list_extended(&pair),
    Rule::FunctionCallExtended => parse_function_call_extended(&pair),
    Rule::FunctionCall => parse_function_call(&pair),
    Rule::Expression | Rule::ExpressionNoImplicit | Rule::ConditionExpr => {
      parse_expression(pair)
    }
    _ => pair_to_expr_inner(pair),
  }
}

/// Multiply a decimal literal by `10^exponent` by moving its decimal point,
/// keeping every digit the literal was written with. Used for the `*^` form
/// of a precision-tagged real, where rounding through `f64` would throw away
/// the extra digits the precision tag exists to carry.
fn shift_decimal_point(literal: &str, exponent: i32) -> String {
  if exponent == 0 {
    return literal.to_string();
  }
  let (sign, digits) = match literal.strip_prefix('-') {
    Some(rest) => ("-", rest),
    None => ("", literal),
  };
  let (int_part, frac_part) = digits.split_once('.').unwrap_or((digits, ""));
  let mut all: String = int_part.to_string();
  all.push_str(frac_part);
  // Where the point sits in `all` after the shift.
  let point = int_part.len() as i32 + exponent;
  let body = if point <= 0 {
    format!("0.{}{all}", "0".repeat((-point) as usize))
  } else if (point as usize) >= all.len() {
    format!("{all}{}.", "0".repeat(point as usize - all.len()))
  } else {
    let (head, tail) = all.split_at(point as usize);
    format!("{head}.{tail}")
  };
  format!("{sign}{body}")
}

fn pair_to_expr_inner(pair: Pair<Rule>) -> Expr {
  match pair.as_rule() {
    Rule::Integer | Rule::UnsignedInteger => {
      let s = pair.as_str();
      match s.parse::<i128>() {
        Ok(n) => Expr::Integer(n),
        Err(_) => {
          // Overflows i128 — use BigInteger
          match s.parse::<BigInt>() {
            Ok(n) => Expr::BigInteger(n),
            Err(_) => Expr::Integer(0),
          }
        }
      }
    }
    Rule::IntegerScientific | Rule::UnsignedIntegerScientific => {
      let s = pair.as_str();
      let idx = s.find("*^").unwrap();
      let mantissa_str = &s[..idx];
      let exponent_str = &s[idx + 2..];
      let mantissa: i128 = mantissa_str.parse().unwrap_or(0);
      // `parse::<i32>` accepts a leading `+`, so `1*^+150` needs no stripping.
      let exponent: i32 = exponent_str.parse().unwrap_or(0);
      if exponent >= 0 {
        // Positive exponent: multiply mantissa by 10^exp → Integer
        let factor = 10_i128.checked_pow(exponent as u32);
        if let Some(n) = factor.and_then(|f| mantissa.checked_mul(f)) {
          Expr::Integer(n)
        } else {
          // Overflow: use BigInteger
          let m = BigInt::from(mantissa);
          let f = BigInt::from(10).pow(exponent as u32);
          Expr::BigInteger(m * f)
        }
      } else {
        // Negative exponent: mantissa / 10^|exp| → Rational
        let abs_exp = (-exponent) as u32;
        let denom = 10_i128.checked_pow(abs_exp);
        if let Some(d) = denom {
          // Simplify the fraction
          let (num, den) = crate::functions::math_ast::rat_reduce(mantissa, d);
          if den == 1 {
            Expr::Integer(num)
          } else {
            Expr::FunctionCall {
              name: "Rational".to_string(),
              args: vec![Expr::Integer(num), Expr::Integer(den)].into(),
            }
          }
        } else {
          // Overflow: keep the value exact with big integers, the way the
          // positive branch does. `1*^-300` is the rational 1/10^300 in
          // Wolfram, not a machine real.
          let m = BigInt::from(mantissa);
          let d = BigInt::from(10).pow(abs_exp);
          let zero = BigInt::from(0);
          let (mut a, mut b) =
            (if m < zero { -m.clone() } else { m.clone() }, d.clone());
          while b != zero {
            let r = &a % &b;
            a = b;
            b = r;
          }
          let (num, den) = (m / &a, d / &a);
          if den == BigInt::from(1) {
            Expr::BigInteger(num)
          } else {
            Expr::FunctionCall {
              name: "Rational".to_string(),
              args: vec![Expr::BigInteger(num), Expr::BigInteger(den)].into(),
            }
          }
        }
      }
    }
    Rule::Real | Rule::UnsignedReal => {
      let s = pair.as_str();
      // Handle Wolfram's *^ scientific notation (e.g. 2.7*^7 = 2.7e7).
      // Parse as a single f64 from the canonical `1.09e12` form rather than
      // computing `mantissa * 10^exp` — the latter re-rounds (e.g.
      // `1.09 * 1e12` → 1090000000000.0001, while `"1.09e12".parse()` gives
      // the exact nearest f64, 1090000000000.0).
      if let Some(idx) = s.find("*^") {
        let with_e = format!("{}e{}", &s[..idx], &s[idx + 2..]);
        Expr::Real(with_e.parse().unwrap_or(0.0))
      } else {
        // Wolfram switches a Real literal to a BigFloat once the total
        // digit count reaches 18.
        // - For an integer-zero part with all-zero fractional part
        //   (`0.000…0`), it becomes the accuracy form `0``N.` where N
        //   is the count of trailing zeros.
        // - For non-zero values like `10.000…0` it becomes a precision
        //   form with precision ≈ frac_len + log10(|value|).
        if let Some(dot_pos) = s.find('.') {
          let int_part = &s[..dot_pos];
          let frac_part = &s[dot_pos + 1..];
          let int_signless = int_part.trim_start_matches(['+', '-']);
          let int_zero =
            int_signless.is_empty() || int_signless.chars().all(|c| c == '0');
          let total_digits = int_signless.len() + frac_part.len();
          let value_f64: f64 = s.parse().unwrap_or(0.0);
          if int_zero
            && value_f64 == 0.0
            && frac_part.len() >= 18
            && frac_part.chars().all(|c| c == '0')
          {
            // Pure-zero literal with at least 18 fractional zeros becomes
            // the accuracy form `0``N.`.
            let acc = frac_part.len() as f64;
            return Expr::BigFloat("0".to_string(), acc);
          }
          // Significant-digit count: drop leading zeros for 0.xxx values.
          let significant_digits = if int_zero {
            let leading_zeros =
              frac_part.chars().take_while(|c| *c == '0').count();
            frac_part.len().saturating_sub(leading_zeros)
          } else {
            total_digits
          };
          if value_f64 != 0.0 && significant_digits >= 18 {
            // Non-zero literal with 18+ significant digits: store as a
            // BigFloat with that precision. Trailing zeros count toward
            // precision (Wolfram treats `0.7390…200…0` as a 25+ digit
            // PrecisionReal even though many digits are zero).
            let prec = if int_zero {
              significant_digits as f64
            } else {
              frac_part.len() as f64 + value_f64.abs().log10()
            };
            let prec = prec.max(1.0);
            let neg = int_part.starts_with('-');
            let digits = if neg {
              format!("-{int_signless}.{frac_part}")
            } else {
              format!("{int_signless}.{frac_part}")
            };
            return Expr::BigFloat(digits, prec);
          }
        }
        Expr::Real(s.parse().unwrap_or(0.0))
      }
    }
    Rule::PrecisionReal | Rule::UnsignedPrecisionReal => {
      let s = pair.as_str();
      // A `*^` exponent may follow the precision tag (`1.5`*^-16`). It scales
      // the value; the tag itself keeps its meaning, so an *accuracy* tag is
      // applied to the scaled value (hence `1.5``20*^3` has precision
      // 20 + log10(1500), not 20 + log10(1.5)).
      let (s, exponent) = match s.split_once("*^") {
        Some((mantissa, exp)) => (mantissa, exp.parse::<i32>().unwrap_or(0)),
        None => (s, 0),
      };
      let scale = |v: f64| v * 10f64.powi(exponent);
      // Detect double-backtick (accuracy) form. Single-backtick is precision.
      let double = s.contains("``");
      let backtick_pos = s.find('`').unwrap();
      let value_str = &s[..backtick_pos];
      let prec_str = if double {
        &s[backtick_pos + 2..]
      } else {
        &s[backtick_pos + 1..]
      };
      // The scaled literal, as the digit string the BigFloat path stores.
      let scaled_str = shift_decimal_point(value_str, exponent);
      // Integer-form `n`p` (no decimal point, with precision p) drops the
      // precision tag entirely and stays as an Integer (matches Wolfram:
      // `0`2 // Head` → Integer). The bare-backtick form `n`` still
      // promotes to Real to match Wolfram (`0` // Head` → Real).
      let int_form =
        !value_str.contains('.') && !prec_str.is_empty() && !double;
      if int_form && exponent == 0 {
        let n: i128 = value_str.parse().unwrap_or(0);
        Expr::Integer(n)
      } else if prec_str.is_empty() {
        // Bare backtick = machine precision, just parse as Real
        Expr::Real(scale(value_str.parse().unwrap_or(0.0)))
      } else {
        let value_f64: f64 = scale(value_str.parse().unwrap_or(0.0));
        if value_f64 == 0.0 {
          // For accuracy form `0.``α`, preserve the accuracy as a BigFloat
          // with value "0" and the parsed accuracy. The display path
          // formats BigFloat("0", α) as `0``α.`. For precision form
          // `0.`α`, the precision tag is meaningless on a zero value
          // and Wolfram drops it (output is just `0.`).
          if double {
            let acc: f64 = prec_str.parse().unwrap_or(0.0);
            let acc = acc.max(1.0);
            Expr::BigFloat("0".to_string(), acc)
          } else {
            Expr::Real(0.0)
          }
        } else {
          let raw_prec: f64 = prec_str.parse().unwrap_or(0.0);
          // Accuracy form: precision = accuracy + log10(|value|).
          let prec = if double {
            raw_prec + value_f64.abs().log10()
          } else {
            raw_prec
          };
          let prec = prec.max(1.0);
          Expr::BigFloat(scaled_str, prec)
        }
      }
    }
    Rule::BasePrefix => {
      let s = pair.as_str();
      // Parse base^^digits format (e.g. 16^^FF = 255, 2^^1010 = 10,
      // 2^^1.01 = 1.25, 16^^FF.A = 255.625)
      let parts: Vec<&str> = s.splitn(2, "^^").collect();
      let base: u32 = parts[0].parse().unwrap_or(10);
      let digits = parts[1];
      let lower = digits.to_lowercase();
      if let Some(dot_pos) = lower.find('.') {
        // Fractional base literal — produce Real(f64).
        let int_part = &lower[..dot_pos];
        let frac_part = &lower[dot_pos + 1..];
        let int_val: f64 = if int_part.is_empty() {
          0.0
        } else if let Ok(v) = i128::from_str_radix(int_part, base) {
          v as f64
        } else {
          use BigInt;
          use num_traits::{Num, ToPrimitive};
          BigInt::from_str_radix(int_part, base)
            .ok()
            .and_then(|n| n.to_f64())
            .unwrap_or(0.0)
        };
        let mut frac_val: f64 = 0.0;
        let mut divisor: f64 = base as f64;
        for c in frac_part.chars() {
          let digit = c.to_digit(base).unwrap_or(0) as f64;
          frac_val += digit / divisor;
          divisor *= base as f64;
        }
        Expr::Real(int_val + frac_val)
      } else if let Ok(val) = i128::from_str_radix(&lower, base) {
        Expr::Integer(val)
      } else {
        // Overflows i128 — try BigInteger
        use BigInt;
        use num_traits::Num;
        match BigInt::from_str_radix(&lower, base) {
          Ok(n) => Expr::BigInteger(n),
          Err(_) => Expr::Integer(0),
        }
      }
    }
    Rule::String => {
      let s = pair.as_str();
      // Remove surrounding quotes and process escape sequences
      let raw = &s[1..s.len() - 1];
      let mut result = String::with_capacity(raw.len());
      let mut chars = raw.chars().peekable();
      while let Some(c) = chars.next() {
        if c == '\\' {
          match chars.peek() {
            Some('[') => {
              // Potential named character \[Name]
              chars.next(); // consume '['
              let mut name = String::new();
              let mut found_close = false;
              while let Some(&ch) = chars.peek() {
                if ch == ']' {
                  chars.next();
                  found_close = true;
                  break;
                }
                if ch.is_ascii_alphabetic() {
                  name.push(ch);
                  chars.next();
                } else {
                  break;
                }
              }
              if found_close {
                if let Some(unicode) = named_char_to_unicode(&name) {
                  result.push_str(unicode);
                } else {
                  // Unknown named char: preserve original
                  result.push_str("\\[");
                  result.push_str(&name);
                  result.push(']');
                }
              } else {
                // Incomplete \[... — preserve as-is
                result.push_str("\\[");
                result.push_str(&name);
              }
            }
            Some(&'n') => {
              chars.next();
              result.push('\n');
            }
            Some(&'t') => {
              chars.next();
              result.push('\t');
            }
            Some(&'r') => {
              chars.next();
              result.push('\r');
            }
            Some(&'\\') => {
              chars.next();
              result.push('\\');
            }
            Some(&'"') => {
              chars.next();
              result.push('"');
            }
            Some(&'`') => {
              // Wolfram represents `\`` inside a string as a private-use
              // character (U+F7CD), distinct from a literal backtick. The
              // formatter renders it back as `\``.
              chars.next();
              result.push('\u{F7CD}');
            }
            // Box-syntax escapes — Wolfram maps each to a private-use codepoint
            // distinct from the literal two-character sequence. The formatter
            // renders these back as `\(`, `\)`, `\!`, `\*` in OutputForm.
            Some(&'(') => {
              chars.next();
              result.push(crate::functions::string_ast::BOX_OPEN);
            }
            Some(&')') => {
              chars.next();
              result.push(crate::functions::string_ast::BOX_CLOSE);
            }
            Some(&'!') => {
              chars.next();
              result.push(crate::functions::string_ast::BOX_START);
            }
            Some(&'*') => {
              chars.next();
              result.push(crate::functions::string_ast::BOX_SEP);
            }
            Some(&'\n') => {
              chars.next();
            } // line continuation
            Some(_) => {
              let other = chars.next().unwrap();
              result.push('\\');
              result.push(other);
            }
            None => result.push('\\'),
          }
        } else {
          result.push(c);
        }
      }
      Expr::String(result)
    }
    Rule::InformationQuery => {
      // `?symbol` is a REPL shortcut for symbol inspection. Information has
      // no Hold attribute, so a plain `Information[a]` would evaluate `a`
      // first; wrap the symbol in Unevaluated so `?a` keeps inspecting `a`
      // even after `a` was assigned a value. (String wildcards like `?Plot*`
      // need no wrapping — they're already non-evaluating.)
      let symbol_name = pair.into_inner().next().unwrap().as_str().to_string();
      if symbol_name.contains('*') {
        Expr::FunctionCall {
          name: "Information".to_string(),
          args: vec![Expr::String(symbol_name)].into(),
        }
      } else {
        Expr::FunctionCall {
          name: "Information".to_string(),
          args: vec![Expr::FunctionCall {
            name: "Unevaluated".to_string(),
            args: vec![Expr::Identifier(symbol_name)].into(),
          }]
          .into(),
        }
      }
    }
    Rule::FullInformationQuery => {
      // `??symbol` parses as Information[symbol, LongForm -> True]; like `?`
      // it must hold the symbol so post-assignment inspection works.
      let symbol_name = pair.into_inner().next().unwrap().as_str().to_string();
      let long_form_rule = Expr::Rule {
        pattern: Box::new(Expr::Identifier("LongForm".to_string())),
        replacement: Box::new(Expr::Identifier("True".to_string())),
      };
      if symbol_name.contains('*') {
        Expr::FunctionCall {
          name: "Information".to_string(),
          args: vec![Expr::String(symbol_name), long_form_rule].into(),
        }
      } else {
        Expr::FunctionCall {
          name: "Information".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Unevaluated".to_string(),
              args: vec![Expr::Identifier(symbol_name)].into(),
            },
            long_form_rule,
          ]
          .into(),
        }
      }
    }
    Rule::NamedCharIdentifier => {
      let s = pair.as_str();
      named_char_to_expr(s)
    }
    // An omitted element or argument (`{a,,b}`, `f[a,]`) is `Null`.
    Rule::OmittedArg | Rule::OmittedTrailingArg => {
      Expr::Identifier("Null".to_string())
    }
    Rule::Identifier => {
      // A leading backtick ` `x` ` is shorthand for the current context,
      // which is Global by default — collapse it to `x`. An explicit
      // `Global`x` is *not* collapsed here: inside a package it still names
      // the Global symbol, not the package's own `x`, so which symbol it is
      // has to be decided at read time by the context resolver (which maps
      // it back to the plain `x` that Woxi stores a Global symbol under).
      let s = pair.as_str();
      let name = s.strip_prefix('`').unwrap_or(s).to_string();
      // Expand any embedded `\[Name]` segments to their Unicode chars so
      // `Z\[Infinity]` becomes the single identifier `Z∞`, matching
      // wolframscript's identifier-character semantics.
      let name = if name.contains("\\[") {
        let mut out = String::with_capacity(name.len());
        let mut rest = name.as_str();
        while let Some(open) = rest.find("\\[") {
          out.push_str(&rest[..open]);
          let after = &rest[open + 2..];
          if let Some(close) = after.find(']') {
            let nm = &after[..close];
            if let Some(unicode) = named_char_to_unicode(nm) {
              out.push_str(unicode);
            } else {
              // Unknown name — keep the raw `\[Name]` form
              out.push_str(&rest[open..=(open + 2 + close)]);
            }
            rest = &after[close + 1..];
          } else {
            // Unterminated — keep the rest as-is
            out.push_str(&rest[open..]);
            rest = "";
            break;
          }
        }
        out.push_str(rest);
        out
      } else {
        name
      };
      Expr::Identifier(name)
    }
    Rule::DisplayedBoxes => {
      // `\!\(expr\)` — wolframscript's "DisplayedBoxes" form. When the
      // inner content is a regular expression (no box operators), the
      // `\!\(…\)` wrapper is a no-op: `\!\(2+2\)` → `4`. When the inner
      // is BoxNotation, translate the box operators to their math
      // equivalents so e.g. `\!\(x \^ 2\)` evaluates to `Power[x, 2]`.
      let raw_text = pair.as_str().to_string();
      let inner_pair = pair.into_inner().next().unwrap();
      if matches!(inner_pair.as_rule(), Rule::BoxNotation) {
        let raw = inner_pair.as_str();
        // Strip the leading `\(` and trailing `\)` and translate the
        // pairwise box-prefix operators to ordinary math operators.
        let inner = strip_box_delimiters(raw)
          .replace("\\^", "^")
          .replace("\\_", "_")
          .replace("\\+", "+")
          .replace("\\&", " ")
          .replace("\\%", " ")
          .replace("\\@", " ");
        if let Ok(expr) = string_to_expr(&inner) {
          return expr;
        }
        // `\!\(\*boxes\)` — the FrontEnd's "interpret these boxes" escape,
        // which is what `InputForm` writes for a typeset expression, in
        // either delimiter spelling (`\*` or the `BOX_SEP` marker). A `\"`
        // inside marks a *string literal* box (`"\"a, b\""`) apart from one
        // holding source text (`"a + b"`); an escape whose box delimiters
        // are themselves escaped is not box syntax at all (see
        // `box_source_is_double_escaped`) and is left unread.
        let trimmed = inner.trim_start();
        if let Some(box_src) = trimmed.strip_prefix("\\*").or_else(|| {
          trimmed.strip_prefix(crate::functions::string_ast::BOX_SEP)
        }) && !box_source_is_double_escaped(box_src)
          && let Some(expr) = box_escape_to_expr(box_src)
        {
          return expr;
        }
        // Fallback: surface the raw source as HoldComplete.
        return Expr::FunctionCall {
          name: "HoldComplete".to_string(),
          args: vec![Expr::String(raw_text)].into(),
        };
      }
      pair_to_expr(inner_pair)
    }
    Rule::BoxNotation => {
      // `\(... box content ...\)` — translate the box operators into
      // explicit *Box heads (SuperscriptBox, SubscriptBox, …). Atoms
      // inside the box are surfaced as String literals to mirror the
      // wolframscript box AST. Falls back to HoldComplete[<source>] if
      // the inner content can't be parsed as a recognised box form.
      let raw = pair.as_str().to_string();
      parse_box_notation_str(&raw).unwrap_or_else(|| Expr::FunctionCall {
        name: "HoldComplete".to_string(),
        args: vec![Expr::String(raw)].into(),
      })
    }
    Rule::GetShorthand => {
      // `<< filename` → Get["filename"]. The argument can be either a
      // quoted String or an unquoted path (consumed atomically by the
      // GetPath rule).
      let inner = pair.into_inner().next().unwrap();
      let path = if matches!(inner.as_rule(), Rule::String) {
        // Strip surrounding quotes; escape sequences inside the path are
        // unusual but handled the same as a regular string literal.
        let raw = inner.as_str();
        raw[1..raw.len() - 1].to_string()
      } else {
        inner.as_str().to_string()
      };
      Expr::FunctionCall {
        name: "Get".to_string(),
        args: vec![Expr::String(path)].into(),
      }
    }
    Rule::DerivativeIdentifier => {
      // Standalone f' → Derivative[1][f], f'' → Derivative[2][f], etc.
      let inner_pairs: Vec<_> = pair.into_inner().collect();
      let name = inner_pairs[0].as_str().to_string();
      let order = inner_pairs[1].as_str().len();
      Expr::CurriedCall {
        func: Box::new(Expr::FunctionCall {
          name: "Derivative".to_string(),
          args: vec![Expr::Integer(order as i128)].into(),
        }),
        args: vec![Expr::Identifier(name)],
      }
    }
    Rule::DerivativeNumeric => {
      // 1' → Derivative[1][1], (-1.4)' → Derivative[1][-1.4], etc.
      // The literal becomes the argument of the curried Derivative form.
      // Wolfram parses a leading minus on the LITERAL as outer unary minus,
      // not as part of the derivative's argument: `-1.4'` is
      // `Times[-1, Derivative[1][1.4]]`, not `Derivative[1][-1.4]`. The
      // underlying `Real`/`Integer` rules consume the sign, so undo it
      // here: peel off a negative literal value, build the curried
      // derivative on the positive value, then wrap with unary minus.
      let inner_pairs: Vec<_> = pair.into_inner().collect();
      let value = pair_to_expr(inner_pairs[0].clone());
      let order = inner_pairs[1].as_str().len();
      let (positive_value, negative) = match value {
        Expr::Real(f) if f < 0.0 => (Expr::Real(-f), true),
        Expr::Integer(n) if n < 0 => (Expr::Integer(-n), true),
        Expr::Constant(ref s) if s.starts_with('-') => {
          (Expr::Constant(s[1..].to_string()), true)
        }
        Expr::BigFloat(ref digits, prec) if digits.starts_with('-') => {
          (Expr::BigFloat(digits[1..].to_string(), prec), true)
        }
        other => (other, false),
      };
      let curried = Expr::CurriedCall {
        func: Box::new(Expr::FunctionCall {
          name: "Derivative".to_string(),
          args: vec![Expr::Integer(order as i128)].into(),
        }),
        args: vec![positive_value],
      };
      if negative {
        Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand: Box::new(curried),
        }
      } else {
        curried
      }
    }
    Rule::Slot => {
      let s = pair.as_str();
      // # is slot 1, #1 is slot 1, #2 is slot 2, etc.
      // #name is the named-slot form Slot["name"] — fills from the keys of
      // an Association passed as the first argument.
      let suffix = &s[1..];
      if suffix
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic())
      {
        return Expr::FunctionCall {
          name: "Slot".to_string(),
          args: vec![Expr::String(suffix.to_string())].into(),
        };
      }
      let num = if s.len() > 1 {
        suffix.parse().unwrap_or(1)
      } else {
        1
      };
      Expr::Slot(num)
    }
    Rule::SlotCall => {
      // #[args] or #n[args] — Slot used as a function head
      let mut inner = pair.into_inner();
      let slot_pair = inner.next().unwrap();
      let slot_expr = pair_to_expr(slot_pair);

      // Collect BracketArgs (supports chained calls like #[a][b]). A
      // `PartIndexSuffix` splits the run: brackets before it are calls on
      // the slot, the ones after are calls on the extracted part
      // (`#[k][[1]][x]`).
      let mut result = slot_expr;
      for child in inner {
        match child.as_rule() {
          Rule::BracketArgs => {
            result = Expr::CurriedCall {
              func: Box::new(result),
              args: child.into_inner().map(pair_to_expr).collect(),
            };
          }
          Rule::PartIndexSuffix => {
            result = apply_part_index_suffix(result, child);
          }
          _ => {}
        }
      }
      result
    }
    Rule::SlotSequence => {
      let s = pair.as_str();
      // ## is slot sequence starting at 1, ##2 starts at 2, etc.
      let num = if s.len() > 2 {
        s[2..].parse().unwrap_or(1)
      } else {
        1
      };
      Expr::SlotSequence(num)
    }
    Rule::OutShortcut => {
      // `%n` (digits) → Out[n]; bare `%`/`%%`/`%%%` → Out[$Line - k]. We
      // model `$Line` as a fresh-session counter that always reads as 1
      // (matching wolframscript's script-mode behaviour), so the parser
      // emits the resolved index directly. Negative indices stay as
      // Out[k] until evaluation collapses them to Out[0].
      let s = pair.as_str();
      let digit_start = s.find(|c: char| c.is_ascii_digit());
      let n: i128 = match digit_start {
        Some(i) => s[i..].parse().unwrap_or(0),
        None => 1 - (s.chars().filter(|c| *c == '%').count() as i128),
      };
      Expr::FunctionCall {
        name: "Out".to_string(),
        args: vec![Expr::Integer(n)].into(),
      }
    }
    Rule::Constant | Rule::UnsignedConstant => {
      Expr::Constant(pair.as_str().trim().to_string())
    }
    Rule::BaseFunctionCall => {
      let inner_pairs: Vec<_> = pair.into_inner().collect();
      let name_pair = &inner_pairs[0];
      let name = resolve_head_name(name_pair);
      let derivative_order = inner_pairs
        .iter()
        .find(|p| matches!(p.as_rule(), Rule::DerivativePrime))
        .map(|p| p.as_str().len());
      let args: Vec<Expr> = inner_pairs
        .iter()
        .skip(1)
        .filter(|p| {
          !matches!(p.as_rule(), Rule::DerivativePrime) && p.as_str() != ","
        })
        .map(|p| pair_to_expr(p.clone()))
        .collect();
      if let Some(order) = derivative_order {
        // f'[x] → Derivative[n][f][x]
        Expr::CurriedCall {
          func: Box::new(Expr::CurriedCall {
            func: Box::new(Expr::FunctionCall {
              name: "Derivative".to_string(),
              args: vec![Expr::Integer(order as i128)].into(),
            }),
            args: vec![Expr::Identifier(name)],
          }),
          args,
        }
      } else {
        Expr::FunctionCall {
          name,
          args: args.into(),
        }
      }
    }
    Rule::LeadingMinus => {
      // LeadingMinus is handled in parse_expression, not here directly
      // But if encountered standalone, treat as a sentinel
      Expr::Integer(0) // Should not be reached
    }
    Rule::CompoundExpression => parse_compound_expression(&pair),
    Rule::AssociationExtended => parse_association_extended(pair),
    Rule::Association => parse_association(pair),
    Rule::AssociationItem => parse_association_item(pair),
    Rule::ReplacementRule => {
      let pair_start = pair.as_span().start();
      let full_str = pair.as_str().to_string();
      let children: Vec<_> = pair.into_inner().collect();
      // The operator sits between the second-to-last and last children;
      // checking only that span keeps chained rules with mixed arrows
      // (a -> b :> c) classified correctly.
      let op_from = children[children.len() - 2].as_span().end() - pair_start;
      let op_to = children[children.len() - 1].as_span().start() - pair_start;
      let is_delayed = full_str[op_from..op_to].contains(":>");
      // Grammar: ConditionExpr ~ ("/;" ~ ConditionExpr)? ~ ("->" | ":>") ~ ConditionExpr
      // 2 children: pattern -> replacement
      // 3 children: pattern /; condition -> replacement
      let (pattern, replacement) = if children.len() == 3 {
        // pattern /; condition :> replacement
        // Build a proper Condition[pattern, test] AST node
        let pattern_expr = pair_to_expr(children[0].clone());
        let condition_expr = pair_to_expr(children[1].clone());
        (
          Expr::FunctionCall {
            name: "Condition".to_string(),
            args: vec![pattern_expr, condition_expr].into(),
          },
          pair_to_expr(children[2].clone()),
        )
      } else {
        (
          pair_to_expr(children[0].clone()),
          pair_to_expr(children[1].clone()),
        )
      };
      if is_delayed {
        Expr::RuleDelayed {
          pattern: Box::new(pattern),
          replacement: Box::new(replacement),
        }
      } else {
        Expr::Rule {
          pattern: Box::new(pattern),
          replacement: Box::new(replacement),
        }
      }
    }
    Rule::PatternSimple => {
      let full = pair.as_str();
      let inner = pair.into_inner();
      let children: Vec<_> = inner.collect();
      let name = if children.is_empty() {
        String::new()
      } else {
        children[0].as_str().to_string()
      };
      // Count underscores in the full text (reliable even with implicit whitespace)
      let blank_count = full.chars().filter(|&c| c == '_').count();
      let blank_type = blank_count.min(3) as u8;
      Expr::Pattern {
        name,
        head: None,
        blank_type,
      }
    }
    Rule::PatternWithHead => {
      let full = pair.as_str();
      let inner = pair.into_inner();
      // PatternName is optional; collect all children
      let children: Vec<_> = inner.collect();
      let (name, head_str) = if children.len() == 2 {
        // PatternName + Identifier(head)
        (
          children[0].as_str().to_string(),
          children[1].as_str().to_string(),
        )
      } else {
        // Just Identifier(head), no name
        (String::new(), children[0].as_str().to_string())
      };
      // Count underscores in the full text (reliable even with implicit whitespace)
      let blank_count = full.chars().filter(|&c| c == '_').count();
      let blank_type = blank_count.min(3) as u8;
      Expr::Pattern {
        name,
        head: Some(head_str),
        blank_type,
      }
    }
    Rule::PatternOptionalSimple => {
      // PatternOptionalSimple = { PatternName ~ "_" ~ ":" ~ Term }
      let mut inner = pair.into_inner();
      let name = inner.next().unwrap().as_str().to_string();
      let default_pair = inner.next().unwrap();
      let default = pair_to_expr(default_pair);
      Expr::PatternOptional {
        name,
        head: None,
        default: Some(Box::new(default)),
      }
    }
    Rule::PatternOptionalAnonBlank => {
      // PatternOptionalAnonBlank = { "_" ~ ":" ~ Term }
      // `_:default` — anonymous Optional[Blank[], default]. Same shape as
      // `name_:default` but without the name binding.
      let mut inner = pair.into_inner();
      let default = pair_to_expr(inner.next().unwrap());
      Expr::PatternOptional {
        name: String::new(),
        head: None,
        default: Some(Box::new(default)),
      }
    }
    Rule::PatternNamed => {
      // PatternNamed = { PatternName ~ ":" ~ Term }
      // `y : 1` is `Pattern[y, 1]` — a named pattern matching the literal
      // body. Distinct from `y_:1` (Optional with default), which keeps
      // its own rule.
      //
      // The grammar parses chained `:` right-associatively, so `a:b:c`
      // arrives here as `name=a, body=Pattern[b, c]`. Wolfram instead
      // treats the trailing `:c` as an Optional default on the already-
      // built Pattern, giving `Optional[Pattern[a, b], c]`. Restructure
      // here to match. Repeated chains (`a:b:c:d:e`) keep nesting the
      // Optional on the right (`Optional[Pattern[a, b], <inner>]`),
      // which mirrors wolframscript's `a:b:(c:d:e)` display.
      let mut inner = pair.into_inner();
      let name = inner.next().unwrap().as_str().to_string();
      let mut body = pair_to_expr(inner.next().unwrap());
      let mut rest = inner.peekable();
      if rest
        .peek()
        .is_some_and(|p| p.as_rule() == Rule::PatternTestSuffix)
      {
        let suffix = rest.next().unwrap();
        body = build_pattern_test(body, suffix.into_inner());
      }
      let mut inner = rest;
      // Optional trailing `..` / `...` postfix: wrap body in
      // Repeated[…] / RepeatedNull[…] before the Pattern so e.g.
      // `s:0..` parses as `Pattern[s, Repeated[0]]` (Wolfram's binding)
      // rather than `Repeated[Pattern[s, 0]]`.
      let mut named_default: Option<Expr> = None;
      for suffix in inner.by_ref() {
        let suffix_name = match suffix.as_rule() {
          Rule::RepeatedNullSuffix => "RepeatedNull",
          Rule::RepeatedSuffix => "Repeated",
          // `name : body : default` — the slot may be omitted.
          Rule::PatternNamedDefault => {
            named_default = suffix.into_inner().next().map(pair_to_expr);
            continue;
          }
          _ => "",
        };
        if !suffix_name.is_empty() {
          body = Expr::FunctionCall {
            name: suffix_name.to_string(),
            args: vec![body].into(),
          };
        }
      }
      if let Some(default) = named_default {
        return Expr::FunctionCall {
          name: "Optional".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Pattern".to_string(),
              args: vec![Expr::Identifier(name), body].into(),
            },
            default,
          ]
          .into(),
        };
      }
      if let Expr::FunctionCall {
        name: bn,
        args: bargs,
      } = &body
        && bn == "Pattern"
        && bargs.len() == 2
      {
        return Expr::FunctionCall {
          name: "Optional".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Pattern".to_string(),
              args: vec![Expr::Identifier(name), bargs[0].clone()].into(),
            },
            bargs[1].clone(),
          ]
          .into(),
        };
      }
      Expr::FunctionCall {
        name: "Pattern".to_string(),
        args: vec![Expr::Identifier(name), body].into(),
      }
    }
    Rule::PatternOptionalNamedBlank => {
      // PatternOptionalNamedBlank = { PatternName ~ ":" ~ "_" ~
      //                               Identifier? ~ ":" ~ Term }
      // Equivalent to `name_Head:default` / `name_:default`. `Term` is a
      // silent rule, so the inner iterator contains `PatternName`,
      // optionally the head `Identifier`, and then the substituted
      // default (also often an `Identifier`). Disambiguate by collecting
      // everything and deciding based on count.
      let children: Vec<_> = pair.into_inner().collect();
      let name = children[0].as_str().to_string();
      let (head, default_pair) = match children.len() {
        2 => (None, children[1].clone()),
        // 3: PatternName, head Identifier, default (from Term)
        _ => (Some(children[1].as_str().to_string()), children[2].clone()),
      };
      let default = pair_to_expr(default_pair);
      Expr::PatternOptional {
        name,
        head,
        default: Some(Box::new(default)),
      }
    }
    Rule::PatternOptionalDefaultSimple => {
      // PatternOptionalDefaultSimple = { PatternName? ~ "_" ~ "." }
      let mut inner = pair.into_inner();
      let name = inner
        .next()
        .map(|p| p.as_str().to_string())
        .unwrap_or_default();
      Expr::PatternOptional {
        name,
        head: None,
        default: None,
      }
    }
    Rule::PatternOptionalWithHead => {
      // PatternOptionalWithHead = { PatternName ~ "_" ~ Identifier ~ ":" ~ Term }
      let mut inner = pair.into_inner();
      let name = inner.next().unwrap().as_str().to_string();
      let head = Some(inner.next().unwrap().as_str().to_string());
      let default_pair = inner.next().unwrap();
      let default = pair_to_expr(default_pair);
      Expr::PatternOptional {
        name,
        head,
        default: Some(Box::new(default)),
      }
    }
    Rule::PatternTest => {
      // PatternTest has two alternatives:
      //   (a) pattern form:  x_?test, _?test, x_Head?test, _Head?test, ...
      //   (b) bare-infix form: a?b, a?b[c], a?(expr)[c], ...
      //       — `?` binds before any trailing `[args]`, producing
      //       PatternTest[a, b][c]. The first inner pair is a
      //       PatternTestLhsBare in this case.
      let full = pair.as_str();
      let inner_pairs: Vec<_> = pair.into_inner().collect();
      // (c) parenthesized form: `(r_)?Positive`, which is the same pattern
      //     as `r_?Positive` — the brackets only group.
      if matches!(
        inner_pairs.first().map(pest::iterators::Pair::as_rule),
        Some(Rule::PatternTestLhsParen)
      ) {
        let mut iter = inner_pairs.into_iter();
        let lhs = pair_to_expr(
          iter
            .next()
            .unwrap()
            .into_inner()
            .next()
            .expect("parenthesized left side"),
        );
        let test = pair_to_expr(iter.next().unwrap());
        return match &lhs {
          Expr::Pattern {
            name,
            head,
            blank_type,
          } => Expr::PatternTest {
            name: name.clone(),
            head: head.clone(),
            blank_type: *blank_type,
            test: Box::new(test),
          },
          _ => Expr::FunctionCall {
            name: "PatternTest".to_string(),
            args: vec![lhs, test].into(),
          },
        };
      }
      if matches!(
        inner_pairs.first().map(pest::iterators::Pair::as_rule),
        Some(Rule::PatternTestLhsBare)
      ) {
        let mut iter = inner_pairs.into_iter();
        let lhs = Expr::Identifier(iter.next().unwrap().as_str().to_string());
        return build_pattern_test(lhs, iter);
      }
      // Pattern form (existing): optional PatternName, optional PatternTestHead, then test
      let mut inner = inner_pairs.into_iter();
      let mut name = String::new();
      let mut head: Option<String> = None;
      let mut last = inner.next().unwrap();
      if last.as_rule() == Rule::PatternName {
        name = last.as_str().to_string();
        last = inner.next().unwrap();
      }
      if last.as_rule() == Rule::PatternTestHead {
        head = Some(last.as_str().to_string());
        last = inner.next().unwrap();
      }
      let test_pair = last;
      let blank_count = full.chars().filter(|&c| c == '_').count();
      let blank_type = blank_count.min(3) as u8;
      let test = pair_to_expr(test_pair);
      Expr::PatternTest {
        name,
        head,
        blank_type,
        test: Box::new(test),
      }
    }
    Rule::PatternCondition => {
      // Store the full pattern string as Raw to preserve test/condition info
      // The string-based pattern matching in apply_replace_all_direct handles these
      Expr::Raw(pair.as_str().to_string())
    }
    Rule::SimpleAnonymousFunction => {
      // Simple anonymous function like #^2& — never has BracketArgs
      let s = pair.as_str().trim().trim_end_matches('&');
      let body = parse_anonymous_body(s);
      Expr::Function {
        body: Box::new(body),
      }
    }
    Rule::ListCall => {
      // {f, g}[x] → CurriedCall with List as the func
      let inner_pairs: Vec<_> = pair.into_inner().collect();
      // First inner pair is the List, rest are BracketArgs
      let list_expr = pair_to_expr(inner_pairs[0].clone());
      let bracket_sequences: Vec<Vec<Expr>> = inner_pairs[1..]
        .iter()
        .filter(|p| matches!(p.as_rule(), Rule::BracketArgs))
        .map(|bracket| {
          bracket
            .clone()
            .into_inner()
            .filter(|p| {
              p.as_str() != "[" && p.as_str() != "]" && p.as_str() != ","
            })
            .map(pair_to_expr)
            .collect()
        })
        .collect();
      let mut result = Expr::CurriedCall {
        func: Box::new(list_expr),
        args: bracket_sequences[0].clone(),
      };
      for args in bracket_sequences.into_iter().skip(1) {
        result = Expr::CurriedCall {
          func: Box::new(result),
          args,
        };
      }
      result
    }
    Rule::NumericCall | Rule::StringCall => {
      // 1[2, 3] / "s"[x] → CurriedCall with the literal as the func (matches
      // the structure of `(1)[2, 3]`).
      let inner_pairs: Vec<_> = pair.into_inner().collect();
      let head_expr = pair_to_expr(inner_pairs[0].clone());
      let bracket_sequences: Vec<Vec<Expr>> = inner_pairs[1..]
        .iter()
        .filter(|p| matches!(p.as_rule(), Rule::BracketArgs))
        .map(|bracket| {
          bracket
            .clone()
            .into_inner()
            .filter(|p| {
              p.as_str() != "[" && p.as_str() != "]" && p.as_str() != ","
            })
            .map(pair_to_expr)
            .collect()
        })
        .collect();
      let mut result = Expr::CurriedCall {
        func: Box::new(head_expr),
        args: bracket_sequences[0].clone(),
      };
      for args in bracket_sequences.into_iter().skip(1) {
        result = Expr::CurriedCall {
          func: Box::new(result),
          args,
        };
      }
      result
    }
    Rule::RuleAnonymousFunction => {
      // Anonymous function with Rule body: {#, First@#2} -> "Q" &
      let inner = pair.into_inner().next().unwrap(); // The ReplacementRule
      let body = pair_to_expr(inner);
      Expr::Function {
        body: Box::new(body),
      }
    }
    Rule::ListItemRule | Rule::FunctionArgRule => {
      // Combined ReplacementRule + optional anonymous function suffix (&),
      // then any postfix applications: `f[a -> 5 // Head]` hands the whole
      // rule to Head, because `//` binds looser than `->`.
      let mut inner_pairs: Vec<_> = pair.into_inner().collect();
      let rule_expr = pair_to_expr(inner_pairs.remove(0));
      let anon_idx = inner_pairs
        .iter()
        .position(|p| matches!(p.as_rule(), Rule::RuleAnonSuffix));
      let boundary = anon_idx.unwrap_or(inner_pairs.len());
      let rest: Vec<_> = inner_pairs.split_off(boundary);

      // `/.`/`//.` (RuleReplaceSuffix) and a trailing `//` that appear
      // *before* the optional `&` bind to the bare rule and get absorbed
      // into the pure function's body — `pat -> repl /. rules &` — same as
      // when there is no `&` at all.
      let mut result = rule_expr;
      for p in inner_pairs {
        match p.as_rule() {
          Rule::RuleReplaceSuffix => {
            let repeated = p.as_str().trim_start().starts_with("//.");
            let rules = pair_to_expr(p.into_inner().next().unwrap());
            result = if repeated {
              Expr::ReplaceRepeated {
                expr: Box::new(result),
                rules: Box::new(rules),
              }
            } else {
              Expr::ReplaceAll {
                expr: Box::new(result),
                rules: Box::new(rules),
              }
            };
          }
          Rule::PostfixFunction => {
            result = Expr::Postfix {
              expr: Box::new(result),
              func: Box::new(parse_postfix_function(p)),
            };
          }
          _ => {}
        }
      }

      if anon_idx.is_some() {
        // Everything after the `&` is a continuation on the pure function
        // it just produced — operator/tilde-infix chains (`pat -> repl &
        // /@ list`, the idiom for building a PopupMenu's choices from a
        // lookup table), `/.`/`//.`, and trailing `//` — shared with a
        // plain Expression's own post-& handling.
        let post_pairs: Vec<_> = rest.into_iter().skip(1).collect();
        let func = Expr::Function {
          body: Box::new(result),
        };
        apply_anon_continuation(func, post_pairs)
      } else {
        result
      }
    }
    Rule::PartExtract => {
      let mut inner = pair.into_inner();
      let base_expr = pair_to_expr(inner.next().unwrap());
      // Chain multiple indices as nested Part: a[[1,2,3]] -> Part[Part[Part[a,1],2],3].
      // A trailing call suffix applies the Part result: a[[i]][x] -> (a[[i]])[x].
      // Extraction and application alternate, so the suffixes are walked in
      // written order — `t[[i]]["pos"][[2]]` extracts, looks up, extracts
      // again. Consecutive `[[…]]` groups are flushed as one run, which is
      // what makes `m[[a]][[b]]` differ from `m[[a, b]]`.
      let inner: Vec<_> = inner.collect();
      let mut result = base_expr;
      let mut pending_groups: Vec<Pair<Rule>> = Vec::new();
      for p in inner {
        if matches!(p.as_rule(), Rule::PartIndexGroup) {
          pending_groups.push(p);
          continue;
        }
        if !pending_groups.is_empty() {
          result =
            apply_part_groups(result, std::mem::take(&mut pending_groups));
        }
        if matches!(p.as_rule(), Rule::DerivativePrime) {
          // `a[[i]]'` differentiates what the part extracted, and any
          // bracket call after it applies the derivative: `a[[i]]'[t]`.
          let order = p.as_str().chars().filter(|c| *c == '\'').count();
          result = Expr::CurriedCall {
            func: Box::new(Expr::FunctionCall {
              name: "Derivative".to_string(),
              args: vec![Expr::Integer(order as i128)].into(),
            }),
            args: vec![result],
          };
        } else if matches!(p.as_rule(), Rule::BracketArgs) {
          let args: Vec<Expr> = p
            .into_inner()
            .filter(|c| {
              c.as_str() != "[" && c.as_str() != "]" && c.as_str() != ","
            })
            .map(pair_to_expr)
            .collect();
          result = Expr::CurriedCall {
            func: Box::new(result),
            args,
          };
        }
      }
      if !pending_groups.is_empty() {
        result = apply_part_groups(result, pending_groups);
      }
      result
    }
    Rule::ParenExtended => parse_paren_extended(pair),
    // `\[Sqrt]x` / `\[CubeRoot]x`: prefix radicals. `^`, `!` and `[[…]]`
    // bind tighter than the radical, so they are part of the operand.
    Rule::RadicalPrefix => {
      let inners: Vec<_> = pair.into_inner().collect();
      let op = inners[0].as_str();
      let mut operand = pair_to_expr(inners[1].clone());
      for suffix in &inners[2..] {
        match suffix.as_rule() {
          Rule::PartIndexSuffix => {
            operand = apply_part_index_suffix(operand, suffix.clone());
          }
          Rule::ImplicitPowerSuffix => {
            operand = Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(operand),
              right: Box::new(implicit_power_exponent(suffix.clone())),
            };
          }
          Rule::FactorialSuffix => {
            let name = if suffix.as_str().starts_with("!!") {
              "Factorial2"
            } else {
              "Factorial"
            };
            operand = Expr::FunctionCall {
              name: name.to_string(),
              args: vec![operand].into(),
            };
          }
          _ => {}
        }
      }
      // `\[CubeRoot]x` is `Surd[x, 3]`, the real-valued cube root — not
      // `x^(1/3)`, which is complex for a negative x.
      if op == "\\[CubeRoot]" || op == "\u{221B}" {
        return Expr::FunctionCall {
          name: "Surd".to_string(),
          args: vec![operand, Expr::Integer(3)].into(),
        };
      }
      Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![operand].into(),
      }
    }
    // `ⅆx`: the differential of the following factor. It has no value of its
    // own — `ⅆarea == 2 π r ⅆr` stays an equation between differentials — and
    // `∫ … ⅆx` is turned into `Integrate[…, x]` before it reaches the parser.
    Rule::DifferentialPrefix => {
      let inners: Vec<_> = pair.into_inner().collect();
      let mut operand = pair_to_expr(inners[1].clone());
      for suffix in &inners[2..] {
        match suffix.as_rule() {
          Rule::PartIndexSuffix => {
            operand = apply_part_index_suffix(operand, suffix.clone());
          }
          Rule::ImplicitPowerSuffix => {
            operand = Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(operand),
              right: Box::new(implicit_power_exponent(suffix.clone())),
            };
          }
          _ => {}
        }
      }
      Expr::FunctionCall {
        name: "DifferentialD".to_string(),
        args: vec![operand].into(),
      }
    }
    // `\[Piecewise]{{v1,c1},{v2,c2},…}` is the special-character input form
    // for `Piecewise[{{v1,c1},{v2,c2},…}]` (the way the front end writes the
    // piecewise brace notation). It binds to the following factor like a
    // prefix function application.
    Rule::PiecewisePrefix => {
      let inners: Vec<_> = pair.into_inner().collect();
      let operand = pair_to_expr(inners[1].clone());
      Expr::FunctionCall {
        name: "Piecewise".to_string(),
        args: vec![operand].into(),
      }
    }
    Rule::Increment => {
      // x++ -> Increment[x]; chained `x++++` -> Increment[Increment[x]].
      // Grammar emits one base pair followed by N `IncrementOp` pairs
      // (N ≥ 1). Wrap `Increment` around the base once per op pair.
      let mut inner = pair.into_inner();
      let base = pair_to_expr(inner.next().unwrap());
      let op_count = inner.count();
      let mut result = base;
      for _ in 0..op_count {
        result = Expr::FunctionCall {
          name: "Increment".to_string(),
          args: vec![result].into(),
        };
      }
      result
    }
    Rule::Decrement => {
      // x-- -> Decrement[x]; chained `x----` -> Decrement[Decrement[x]].
      let mut inner = pair.into_inner();
      let base = pair_to_expr(inner.next().unwrap());
      let op_count = inner.count();
      let mut result = base;
      for _ in 0..op_count {
        result = Expr::FunctionCall {
          name: "Decrement".to_string(),
          args: vec![result].into(),
        };
      }
      result
    }
    Rule::PreIncrement => {
      // ++x -> PreIncrement[x]
      let inner = pair.into_inner().next().unwrap();
      let var = pair_to_expr(inner);
      Expr::FunctionCall {
        name: "PreIncrement".to_string(),
        args: vec![var].into(),
      }
    }
    Rule::PreDecrement => {
      // --x -> PreDecrement[x]
      let inner = pair.into_inner().next().unwrap();
      let var = pair_to_expr(inner);
      Expr::FunctionCall {
        name: "PreDecrement".to_string(),
        args: vec![var].into(),
      }
    }
    Rule::Unset => {
      // x =. -> Unset[x]
      let inner = pair.into_inner().next().unwrap();
      let var = pair_to_expr(inner);
      Expr::FunctionCall {
        name: "Unset".to_string(),
        args: vec![var].into(),
      }
    }
    Rule::ApplyToOp => {
      // x //= f -> ApplyTo[x, f]
      let mut inner = pair.into_inner();
      let var = pair_to_expr(inner.next().unwrap());
      let func = pair_to_expr(inner.next().unwrap());
      Expr::FunctionCall {
        name: "ApplyTo".to_string(),
        args: vec![var, func].into(),
      }
    }
    Rule::AddTo => {
      // x += y -> AddTo[x, y]
      let mut inner = pair.into_inner();
      let var = pair_to_expr(inner.next().unwrap());
      let val = pair_to_expr(inner.next().unwrap());
      Expr::FunctionCall {
        name: "AddTo".to_string(),
        args: vec![var, val].into(),
      }
    }
    Rule::SubtractFrom => {
      // x -= y -> SubtractFrom[x, y]
      let mut inner = pair.into_inner();
      let var = pair_to_expr(inner.next().unwrap());
      let val = pair_to_expr(inner.next().unwrap());
      Expr::FunctionCall {
        name: "SubtractFrom".to_string(),
        args: vec![var, val].into(),
      }
    }
    Rule::TimesBy => {
      // x *= y -> TimesBy[x, y]
      let mut inner = pair.into_inner();
      let var = pair_to_expr(inner.next().unwrap());
      let val = pair_to_expr(inner.next().unwrap());
      Expr::FunctionCall {
        name: "TimesBy".to_string(),
        args: vec![var, val].into(),
      }
    }
    Rule::DivideBy => {
      // x /= y -> DivideBy[x, y]
      let mut inner = pair.into_inner();
      let var = pair_to_expr(inner.next().unwrap());
      let val = pair_to_expr(inner.next().unwrap());
      Expr::FunctionCall {
        name: "DivideBy".to_string(),
        args: vec![var, val].into(),
      }
    }
    Rule::PrefixApplySimple => {
      // f@x within implicit multiplication context → f[x]
      let mut inner = pair.into_inner();
      let func_pair = inner.next().unwrap();
      let arg_pair = inner.next().unwrap();
      let func_expr = pair_to_expr(func_pair);
      let arg_expr = pair_to_expr(arg_pair);
      match func_expr {
        Expr::Identifier(ref name) => Expr::FunctionCall {
          name: name.clone(),
          args: vec![arg_expr].into(),
        },
        Expr::FunctionCall { .. } => Expr::CurriedCall {
          func: Box::new(func_expr),
          args: vec![arg_expr],
        },
        _ => Expr::PrefixApply {
          func: Box::new(func_expr),
          arg: Box::new(arg_expr),
        },
      }
    }
    Rule::NegativeImplicitFirst => {
      // -N as the first factor in implicit multiplication → Integer(-N) or Real(-N)
      let inner = pair.into_inner().next().unwrap();
      let val = pair_to_expr(inner);
      match val {
        Expr::Integer(n) => Expr::Integer(-n),
        Expr::Real(r) => Expr::Real(-r),
        other => Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left: Box::new(Expr::Integer(0)),
          right: Box::new(other),
        },
      }
    }
    Rule::ImplicitTimes => {
      // Implicit multiplication: x y z -> Times[x, y, z]
      // Each factor can optionally have a Part suffix ([[...]]) and/or power suffix (ImplicitPowerSuffix)
      let inners: Vec<_> = pair.into_inner().collect();
      let mut factors: Vec<Expr> = Vec::new();
      let mut i = 0;
      while i < inners.len() {
        if inners[i].as_rule() == Rule::PartIndexSuffix {
          // Part suffix follows the previous factor: x[[i]] -> Part[x, i]
          if let Some(base) = factors.pop() {
            factors.push(apply_part_index_suffix(base, inners[i].clone()));
          }
        } else if inners[i].as_rule() == Rule::ImplicitPowerSuffix {
          // Power suffix follows the previous factor
          if let Some(base) = factors.pop() {
            let exponent = implicit_power_exponent(inners[i].clone());
            factors.push(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(base),
              right: Box::new(exponent),
            });
          }
        } else if matches!(
          inners[i].as_rule(),
          Rule::RepeatedSuffix | Rule::RepeatedNullSuffix
        ) {
          // `..` / `...` postfix wraps the previous factor in
          // `Repeated[…]` / `RepeatedNull[…]`, so `0..1` parses as
          // `Repeated[0] * 1` rather than splitting into two
          // statements.
          if let Some(base) = factors.pop() {
            let func_name = if inners[i].as_rule() == Rule::RepeatedNullSuffix {
              "RepeatedNull"
            } else {
              "Repeated"
            };
            factors.push(Expr::FunctionCall {
              name: func_name.to_string(),
              args: vec![base].into(),
            });
          }
        } else if inners[i].as_rule() == Rule::FactorialSuffix {
          // `!` / `!!` postfix wraps the previous factor in
          // `Factorial[…]` / `Factorial2[…]`. `a! b!` parses as
          // `Times[Factorial[a], Factorial[b]]`.
          if let Some(base) = factors.pop() {
            let func_name = if inners[i].as_str() == "!!" {
              "Factorial2"
            } else {
              "Factorial"
            };
            factors.push(Expr::FunctionCall {
              name: func_name.to_string(),
              args: vec![base].into(),
            });
          }
        } else {
          factors.push(pair_to_expr(inners[i].clone()));
        }
        i += 1;
      }
      if factors.len() == 1 {
        factors.into_iter().next().unwrap()
      } else {
        // Build nested Times
        let mut iter = factors.into_iter();
        let first = iter.next().unwrap();
        iter.fold(first, |acc, f| Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(acc),
          right: Box::new(f),
        })
      }
    }
    Rule::PostfixApplication => {
      let mut inner: Vec<_> = pair.into_inner().collect();
      if inner.is_empty() {
        return Expr::Raw(String::new());
      }
      let mut result = pair_to_expr(inner.remove(0));
      for func_pair in inner {
        let func = pair_to_expr(func_pair);
        result = Expr::Postfix {
          expr: Box::new(result),
          func: Box::new(func),
        };
      }
      result
    }
    Rule::PostfixBase => {
      let inner = pair.into_inner().next().unwrap();
      pair_to_expr(inner)
    }
    Rule::PostfixFunction => {
      let inner = pair.into_inner().next().unwrap();
      pair_to_expr(inner)
    }
    Rule::ReplaceAllExpr => {
      let mut inner = pair.into_inner();
      let expr = pair_to_expr(inner.next().unwrap());
      let rules = pair_to_expr(inner.next().unwrap());
      Expr::ReplaceAll {
        expr: Box::new(expr),
        rules: Box::new(rules),
      }
    }
    Rule::ReplaceRepeatedExpr => {
      let mut inner = pair.into_inner();
      let expr = pair_to_expr(inner.next().unwrap());
      let rules = pair_to_expr(inner.next().unwrap());
      Expr::ReplaceRepeated {
        expr: Box::new(expr),
        rules: Box::new(rules),
      }
    }
    _ => {
      // Fallback: store as raw text
      Expr::Raw(pair.as_str().to_string())
    }
  }
}

/// Parse a PostfixFunction pair into an Expr, wrapping in Function if & is present.
/// In Wolfram, `x // f &` parses as `(f &)[x]`, i.e. & binds tighter than //.
fn parse_postfix_function(pair: Pair<Rule>) -> Expr {
  let has_ampersand = pair.as_str().trim_end().ends_with('&');
  // The inner children are the same (BaseFunctionCall or Identifier);
  // the "&" is an anonymous literal in the grammar and doesn't appear as a child.
  let inner = pair.into_inner().next().unwrap();
  // SimpleAnonymousFunction (`#&`, `#+1&`) already parses itself into a
  // Function{body} and consumes its own trailing "&" — wrapping it again
  // would nest a second Slot scope and break `#` binding.
  if matches!(inner.as_rule(), Rule::SimpleAnonymousFunction) {
    return pair_to_expr(inner);
  }
  let func = pair_to_expr(inner);
  if has_ampersand {
    Expr::Function {
      body: Box::new(func),
    }
  } else {
    func
  }
}

/// Flatten a left-associative chain of `Times` BinaryOps into a Vec of
/// individual factors. Used when splicing the contents of an
/// `ImplicitTimes` term back into the parent operator chain so that
/// `a/b c d` parses as `(a*c*d)/b` rather than `a/(b*c*d)`.
fn flatten_times_chain(expr: &Expr) -> Vec<Expr> {
  fn walk(e: &Expr, out: &mut Vec<Expr>) {
    match e {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        walk(left, out);
        walk(right, out);
      }
      other => out.push(other.clone()),
    }
  }
  let mut out = Vec::new();
  walk(expr, &mut out);
  out
}

/// Whether `op` binds tighter than `Times`, so that it has to reach into an
/// adjacent implicit product rather than take the whole product as one
/// operand. `NEGATE` (the synthetic unary minus) and the `~f~` tilde infix
/// are excluded: both already bind around a whole product the way Wolfram
/// reads them.
fn splits_implicit_product(op: &str) -> bool {
  if op == "NEGATE"
    || (op.starts_with('~') && op.ends_with('~') && op.len() > 2)
  {
    return false;
  }
  operator_precedence(op) > operator_precedence("*")
}

/// An implicit product (`2 x`) is `Times` at ordinary multiplicative
/// precedence, but the grammar hands it to the tree builder as a single
/// term. When a neighbouring operator binds tighter than `Times` that
/// operator must reach the adjacent *factor* — Wolfram reads
/// `2 Times @@ {3, 4}` as `2 (Times @@ {3, 4})` and `a b . c` as
/// `a (b . c)` — so split such a term into its factors joined by explicit
/// `*` operators and let the precedence climb do the rest.
fn split_implicit_products(
  terms: &mut Vec<Expr>,
  operators: &mut Vec<String>,
  was_implicit: &[bool],
) {
  if !was_implicit.iter().any(|b| *b) {
    return;
  }
  let mut new_terms: Vec<Expr> = Vec::with_capacity(terms.len());
  let mut new_ops: Vec<String> = Vec::with_capacity(operators.len());
  for (i, term) in std::mem::take(terms).into_iter().enumerate() {
    if i > 0 {
      new_ops.push(operators[i - 1].clone());
    }
    let touches_tighter_op = (i > 0
      && splits_implicit_product(&operators[i - 1]))
      || operators
        .get(i)
        .is_some_and(|op| splits_implicit_product(op));
    if was_implicit.get(i).copied().unwrap_or(false) && touches_tighter_op {
      let mut factors = flatten_times_chain(&term).into_iter();
      if let Some(first) = factors.next() {
        new_terms.push(first);
        for factor in factors {
          new_ops.push("*".to_string());
          new_terms.push(factor);
        }
      }
    } else {
      new_terms.push(term);
    }
  }
  *terms = new_terms;
  *operators = new_ops;
}

fn parse_list(pair: Pair<Rule>) -> Expr {
  let items: Vec<Expr> = pair
    .into_inner()
    .filter(|p| p.as_str() != ",")
    .map(pair_to_expr)
    .collect();
  Expr::List(items.into())
}

fn parse_list_extended(pair: &Pair<Rule>) -> Expr {
  // Merged rule: List + optional suffix (PartIndexSuffix or ListCallSuffix)
  // Eliminates exponential backtracking for deeply nested lists.
  // Note: Anonymous function suffix (&) is NOT handled here — it is handled
  // at the Expression level via AnonymousFunctionSuffix so `&` gets the
  // correct low precedence.
  let inner_pairs: Vec<_> = pair.clone().into_inner().collect();

  // First inner pair is always List
  let list_expr = pair_to_expr(inner_pairs[0].clone());

  // Check for suffix types
  let has_part_index = inner_pairs
    .iter()
    .any(|p| matches!(p.as_rule(), Rule::PartIndexSuffix));
  let has_list_call = inner_pairs
    .iter()
    .any(|p| matches!(p.as_rule(), Rule::ListCallSuffix));

  if has_part_index {
    // List[[...]]: PartExtract with List base
    inner_pairs
      .iter()
      .filter(|p| matches!(p.as_rule(), Rule::PartIndexSuffix))
      .fold(list_expr, |acc, p| apply_part_index_suffix(acc, p.clone()))
  } else if has_list_call {
    // List[...]: ListCall
    let suffix_pair = inner_pairs
      .iter()
      .find(|p| matches!(p.as_rule(), Rule::ListCallSuffix))
      .unwrap();
    let bracket_sequences: Vec<Vec<Expr>> = suffix_pair
      .clone()
      .into_inner()
      .filter(|p| matches!(p.as_rule(), Rule::BracketArgs))
      .map(|bracket| bracket.into_inner().map(pair_to_expr).collect())
      .collect();
    let mut result = Expr::CurriedCall {
      func: Box::new(list_expr),
      args: bracket_sequences[0].clone(),
    };
    for args in bracket_sequences.into_iter().skip(1) {
      result = Expr::CurriedCall {
        func: Box::new(result),
        args,
      };
    }
    result
  } else {
    // Plain List
    list_expr
  }
}

fn parse_function_call_extended(pair: &Pair<Rule>) -> Expr {
  // Merged rule: FunctionCall + optional Part extraction + optional implicit multiplication suffix
  // Inner pairs: (Identifier|SimpleAnonymousFunction) DerivativePrime? BracketArgs+ DerivativePrime? [PartIndexSuffix | FunctionCallImplicitSuffix]
  // Note: Anonymous function suffix (&) is NOT handled here — it is handled
  // at the Expression level via AnonymousFunctionSuffix so that `&` gets the
  // correct low precedence (binds the whole infix chain, not just the call).
  let inner_pairs: Vec<_> = pair.clone().into_inner().collect();

  let name_pair = &inner_pairs[0];

  // A leading DerivativePrime (between Identifier and BracketArgs)
  // applies to the head symbol: `f'[x]` → `Derivative[1][f][x]`. A
  // trailing DerivativePrime (after the last BracketArgs) wraps the
  // entire call: `h[1]'` → `Derivative[1][h[1]]`.
  let first_bracket_idx_fce =
    inner_pairs.iter().enumerate().find_map(|(i, p)| {
      if matches!(p.as_rule(), Rule::BracketArgs) {
        Some(i)
      } else {
        None
      }
    });
  let derivative_order = first_bracket_idx_fce.and_then(|fb| {
    inner_pairs[..fb]
      .iter()
      .find(|p| matches!(p.as_rule(), Rule::DerivativePrime))
      .map(|p| p.as_str().len())
  });
  // The prime after the brackets, plus any arguments it is applied to:
  // `f[data]'[x]` is `Derivative[1][f[data]][x]`.
  let trailing_prime_idx = first_bracket_idx_fce.and_then(|fb| {
    inner_pairs[fb..]
      .iter()
      .position(|p| matches!(p.as_rule(), Rule::DerivativePrime))
      .map(|offset| fb + offset)
  });
  let trailing_prime_order_fce =
    trailing_prime_idx.map(|i| inner_pairs[i].as_str().len());
  let post_prime_bracket_args: Vec<Vec<Expr>> = trailing_prime_idx
    .map(|i| {
      inner_pairs[i + 1..]
        .iter()
        .take_while(|p| matches!(p.as_rule(), Rule::BracketArgs))
        .map(|bracket| bracket.clone().into_inner().map(pair_to_expr).collect())
        .collect()
    })
    .unwrap_or_default();

  // Collect the function call's BracketArgs (consecutive BracketArgs after name, before any suffix)
  let fc_bracket_args: Vec<Vec<Expr>> = inner_pairs
    .iter()
    .skip(1) // skip Identifier/SimpleAnonymousFunction
    .skip_while(|p| matches!(p.as_rule(), Rule::DerivativePrime)) // skip optional DerivativePrime
    .take_while(|p| matches!(p.as_rule(), Rule::BracketArgs))
    .map(|bracket| bracket.clone().into_inner().map(pair_to_expr).collect())
    .collect();

  // Detect suffix types via named rules
  let has_part_index = inner_pairs
    .iter()
    .any(|p| matches!(p.as_rule(), Rule::PartIndexSuffix));
  let has_implicit_suffix = inner_pairs
    .iter()
    .any(|p| matches!(p.as_rule(), Rule::FunctionCallImplicitSuffix));
  let has_implicit_power = inner_pairs
    .iter()
    .any(|p| matches!(p.as_rule(), Rule::ImplicitPowerSuffix));

  // Build the base function call expression
  let base_func =
    if matches!(name_pair.as_rule(), Rule::SimpleAnonymousFunction) {
      let anon_expr = pair_to_expr(name_pair.clone());
      let mut result = Expr::CurriedCall {
        func: Box::new(anon_expr),
        args: fc_bracket_args[0].clone(),
      };
      for args in fc_bracket_args.iter().skip(1) {
        result = Expr::CurriedCall {
          func: Box::new(result),
          args: args.clone(),
        };
      }
      result
    } else if let Some(order) = derivative_order {
      // f'[x] → Derivative[1][f][x], f''[x] → Derivative[2][f][x], etc.
      let name = resolve_head_name(name_pair);
      // Derivative[n][f]
      let mut result = Expr::CurriedCall {
        func: Box::new(Expr::FunctionCall {
          name: "Derivative".to_string(),
          args: vec![Expr::Integer(order as i128)].into(),
        }),
        args: vec![Expr::Identifier(name)],
      };
      // Apply bracket args: Derivative[n][f][x], then any further chained calls
      for args in &fc_bracket_args {
        result = Expr::CurriedCall {
          func: Box::new(result),
          args: args.clone(),
        };
      }
      result
    } else {
      let name = resolve_head_name(name_pair);
      if fc_bracket_args.len() == 1 {
        Expr::FunctionCall {
          name,
          args: fc_bracket_args[0].clone().into(),
        }
      } else {
        let mut result = Expr::FunctionCall {
          name,
          args: fc_bracket_args[0].clone().into(),
        };
        for args in fc_bracket_args.iter().skip(1) {
          result = Expr::CurriedCall {
            func: Box::new(result),
            args: args.clone(),
          };
        }
        result
      }
    };

  // A trailing DerivativePrime wraps the entire constructed call in
  // `Derivative[n][...]` (e.g. `h[1]'` → `Derivative[1][h[1]]`).
  let base_func = if let Some(order) = trailing_prime_order_fce {
    let mut wrapped = Expr::CurriedCall {
      func: Box::new(Expr::FunctionCall {
        name: "Derivative".to_string(),
        args: vec![Expr::Integer(order as i128)].into(),
      }),
      args: vec![base_func],
    };
    for args in &post_prime_bracket_args {
      wrapped = Expr::CurriedCall {
        func: Box::new(wrapped),
        args: args.clone(),
      };
    }
    wrapped
  } else {
    base_func
  };

  // Helper: parse FunctionCallImplicitSuffix inner pairs into multiplication factors
  // (same logic as ImplicitTimes handler)
  let parse_implicit_factors =
    |suffix_pair: &pest::iterators::Pair<Rule>| -> Vec<Expr> {
      let inners: Vec<_> = suffix_pair.clone().into_inner().collect();
      let mut factors: Vec<Expr> = Vec::new();
      let mut i = 0;
      while i < inners.len() {
        if inners[i].as_rule() == Rule::PartIndexSuffix {
          if let Some(base) = factors.pop() {
            // Delegate to the shared Part-suffix handler rather than
            // walking `PartIndexGroup` pairs by hand: `PartIndexGroup`'s
            // span is the whole bracketed group (`[[1]]`), not the index
            // expression, so feeding it straight to `pair_to_expr` fell
            // through to the "unknown rule" fallback and stored the raw
            // bracket text as the index instead of the parsed `1` — see
            // `apply_part_index_suffix`.
            factors.push(apply_part_index_suffix(base, inners[i].clone()));
          }
        } else if inners[i].as_rule() == Rule::ImplicitPowerSuffix {
          if let Some(base) = factors.pop() {
            let exponent = implicit_power_exponent(inners[i].clone());
            factors.push(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(base),
              right: Box::new(exponent),
            });
          }
        } else if inners[i].as_rule() == Rule::FactorialSuffix {
          // `!` / `!!` postfix wraps the previous factor in
          // `Factorial[…]` / `Factorial2[…]`, so `f[x] n!` parses as
          // `Times[f[x], Factorial[n]]`.
          if let Some(base) = factors.pop() {
            let func_name = if inners[i].as_str() == "!!" {
              "Factorial2"
            } else {
              "Factorial"
            };
            factors.push(Expr::FunctionCall {
              name: func_name.to_string(),
              args: vec![base].into(),
            });
          }
        } else if matches!(
          inners[i].as_rule(),
          Rule::RepeatedSuffix | Rule::RepeatedNullSuffix
        ) {
          // `..` / `...` postfix wraps the previous factor in
          // `Repeated[…]` / `RepeatedNull[…]`.
          if let Some(base) = factors.pop() {
            let func_name = if inners[i].as_rule() == Rule::RepeatedNullSuffix {
              "RepeatedNull"
            } else {
              "Repeated"
            };
            factors.push(Expr::FunctionCall {
              name: func_name.to_string(),
              args: vec![base].into(),
            });
          }
        } else {
          factors.push(pair_to_expr(inners[i].clone()));
        }
        i += 1;
      }
      factors
    };

  // Helper: fold a base expression with implicit multiplication factors into nested Times
  let fold_implicit_times = |base: Expr, factors: Vec<Expr>| -> Expr {
    factors.into_iter().fold(base, |acc, f| Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(acc),
      right: Box::new(f),
    })
  };

  // Apply the `[[…]]` extractions and the calls interleaved with them
  // (`f[x][[i]][y]`) in the order they were written, starting at the first
  // `PartIndexSuffix` so the head's own bracket args are not re-applied.
  let apply_part_and_call_suffixes = |base: Expr| -> Expr {
    let first = inner_pairs
      .iter()
      .position(|p| matches!(p.as_rule(), Rule::PartIndexSuffix))
      .unwrap();
    inner_pairs[first..]
      .iter()
      .fold(base, |acc, p| match p.as_rule() {
        Rule::PartIndexSuffix => apply_part_index_suffix(acc, p.clone()),
        Rule::BracketArgs => Expr::CurriedCall {
          func: Box::new(acc),
          args: p.clone().into_inner().map(pair_to_expr).collect(),
        },
        _ => acc,
      })
  };

  if has_part_index && has_implicit_suffix {
    // PartExtract with implicit multiplication: f[x][[i]]^2 y
    let mut result = apply_part_and_call_suffixes(base_func);
    if has_implicit_power {
      let exponent = implicit_power_exponent(
        inner_pairs
          .iter()
          .find(|p| matches!(p.as_rule(), Rule::ImplicitPowerSuffix))
          .unwrap()
          .clone(),
      );
      result = Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(result),
        right: Box::new(exponent),
      };
    }
    let suffix_pair = inner_pairs
      .iter()
      .find(|p| matches!(p.as_rule(), Rule::FunctionCallImplicitSuffix))
      .unwrap();
    let factors = parse_implicit_factors(suffix_pair);
    fold_implicit_times(result, factors)
  } else if has_part_index {
    // Plain PartExtract: f[x][[i]], and any call on what it extracted.
    apply_part_and_call_suffixes(base_func)
  } else if has_implicit_suffix {
    // Implicit multiplication after function call: f[x] g[y] or f[x]^2 y
    let mut result = base_func;
    if has_implicit_power {
      let exponent = implicit_power_exponent(
        inner_pairs
          .iter()
          .find(|p| matches!(p.as_rule(), Rule::ImplicitPowerSuffix))
          .unwrap()
          .clone(),
      );
      result = Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(result),
        right: Box::new(exponent),
      };
    }
    let suffix_pair = inner_pairs
      .iter()
      .find(|p| matches!(p.as_rule(), Rule::FunctionCallImplicitSuffix))
      .unwrap();
    let factors = parse_implicit_factors(suffix_pair);
    fold_implicit_times(result, factors)
  } else {
    // Plain FunctionCall
    base_func
  }
}

/// Resolve the display name of a function-call head pair. A plain
/// `Identifier` head uses its raw text; a `NamedCharIdentifier` head (e.g.
/// `\[Psi]`) is decoded to its Unicode symbol (`ψ`) so `\[Psi][x, t]` calls
/// the same symbol that `\[Psi]` denotes on its own.
fn resolve_head_name(name_pair: &Pair<Rule>) -> String {
  if matches!(name_pair.as_rule(), Rule::NamedCharIdentifier) {
    let decoded = pair_to_expr(name_pair.clone());
    if let Expr::Identifier(s) | Expr::Constant(s) = &decoded {
      return s.clone();
    }
  }
  name_pair.as_str().to_string()
}

fn parse_function_call(pair: &Pair<Rule>) -> Expr {
  // FunctionCall is a strict subset of FunctionCallExtended's inner shape
  // (no PartIndexSuffix / FunctionCallImplicitSuffix), so the extended
  // parser — which already handles a trailing prime followed by further
  // bracket calls, e.g. `h[i]'[t]` → `Derivative[1][h[i]][t]` — covers it
  // exactly. Delegate rather than duplicating that logic.
  parse_function_call_extended(pair)
}

/// Parse an expression with operators into an Expr
fn parse_expression(pair: Pair<Rule>) -> Expr {
  let mut inner: Vec<Pair<Rule>> = pair.into_inner().collect();

  if inner.is_empty() {
    return Expr::Raw(String::new());
  }

  // Find AnonymousFunctionSuffix position (if any) to split pre-& and post-& parts
  let anon_idx = inner
    .iter()
    .position(|p| p.as_rule() == Rule::AnonymousFunctionSuffix);

  // Split off continuation operators/postfix AFTER & (if any)
  let (anon_func_suffix, post_anon_pairs) = if let Some(idx) = anon_idx {
    let suffix = inner.remove(idx);
    let post_pairs: Vec<Pair<Rule>> = inner.drain(idx..).collect();
    let bracket_args: Vec<Vec<Expr>> = suffix
      .into_inner()
      .filter(|p| matches!(p.as_rule(), Rule::BracketArgs))
      .map(|bracket| bracket.into_inner().map(pair_to_expr).collect())
      .collect();
    (Some(bracket_args), post_pairs)
  } else {
    (None, Vec::new())
  };

  // Collect trailing PostfixFunction pairs from pre-& part
  let mut postfix_funcs: Vec<Pair<Rule>> = Vec::new();
  while inner
    .last()
    .is_some_and(|p| p.as_rule() == Rule::PostfixFunction)
  {
    postfix_funcs.push(inner.pop().unwrap());
  }
  postfix_funcs.reverse(); // restore left-to-right order

  // Check for trailing ReplaceAll/ReplaceRepeated suffixes:
  // Term (Operator Term)* (ReplaceAllSuffix | ReplaceRepeatedSuffix)*
  // A chain like `a /. b /. c` produces several suffixes that must all be
  // folded left-to-right (`ReplaceAll[ReplaceAll[a, b], c]`); popping only the
  // last one silently dropped the intermediate replacements.
  let mut replace_rules: Vec<(Pair<Rule>, bool)> = Vec::new();
  while inner.last().is_some_and(|p| {
    p.as_rule() == Rule::ReplaceAllSuffix
      || p.as_rule() == Rule::ReplaceRepeatedSuffix
  }) {
    let suffix = inner.pop().unwrap();
    let is_replace_repeated = suffix.as_rule() == Rule::ReplaceRepeatedSuffix;
    // The suffix contains the List or ReplacementRule as its child
    let rules_pair = suffix.into_inner().next().unwrap();
    replace_rules.push((rules_pair, is_replace_repeated));
  }
  // Popped from the end, so restore left-to-right application order.
  replace_rules.reverse();

  // Single term case (no operators, no replace, no post-& continuation).
  // A lone `;;` is not a term — it is `Span[1, All]`, built from the two
  // omitted operands by `parse_expression_inner`.
  if inner.len() == 1
    && inner[0].as_rule() != Rule::SpanNoOperandSep
    && replace_rules.is_empty()
    && post_anon_pairs.is_empty()
  {
    let mut result = pair_to_expr(inner.remove(0));
    for func_pair in postfix_funcs {
      let func = parse_postfix_function(func_pair);
      result = Expr::Postfix {
        expr: Box::new(result),
        func: Box::new(func),
      };
    }
    if let Some(bracket_args) = anon_func_suffix {
      result = Expr::Function {
        body: Box::new(result),
      };
      for args in bracket_args {
        result = Expr::CurriedCall {
          func: Box::new(result),
          args,
        };
      }
    }
    return result;
  }

  parse_expression_inner(
    inner,
    replace_rules,
    postfix_funcs,
    anon_func_suffix,
    post_anon_pairs,
  )
}

fn parse_expression_inner(
  inner: Vec<Pair<Rule>>,
  replace_rules: Vec<(Pair<Rule>, bool)>,
  postfix_funcs: Vec<Pair<Rule>>,
  anon_func_suffix: Option<Vec<Vec<Expr>>>,
  post_anon_pairs: Vec<Pair<Rule>>,
) -> Expr {
  // Parse operators: Term (Operator Term)*
  // Build expression with proper precedence
  let mut terms: Vec<Expr> = Vec::new();
  // Parallel to `terms`: whether each pushed term originated from an
  // `ImplicitTimes` (i.e. unparenthesised juxtaposition like `4z`). Used so
  // a trailing `!` can attach to the rightmost factor — `4z!` → `4*(z!)`,
  // matching wolframscript's binding while keeping `(4z)!` intact.
  let mut term_was_implicit_times: Vec<bool> = Vec::new();
  let mut operators: Vec<String> = Vec::new();
  let mut leading_minus = false;
  let mut leading_not = false;
  // True iff the *first* term was prefixed by `!`. Survives operator
  // boundaries (unlike pending_not_on_last, which is flushed eagerly).
  // Used to re-apply Not at the right precedence after the binary tree
  // is built.
  let mut leading_not_at_start = false;
  // Tracks a pending Not prefix that should wrap the *final* form of the
  // current term — i.e. after all suffixes (!, !!, \[Transpose], ..) have
  // been applied. This makes "!a!" parse as Not[Factorial[a]] rather than
  // Factorial[Not[a]].
  let mut pending_not_on_last = false;
  // Apply a pending Not to the top of `terms` and clear the flag. Used at
  // every boundary that ends a term block.
  fn flush_pending_not(
    pending: &mut bool,
    terms: &mut Vec<Expr>,
    origins: &mut Vec<bool>,
  ) {
    if *pending {
      if let Some(last) = terms.pop() {
        origins.pop();
        terms.push(Expr::FunctionCall {
          name: "Not".to_string(),
          args: vec![last].into(),
        });
        origins.push(false);
      }
      *pending = false;
    }
  }

  for item in inner {
    match item.as_rule() {
      Rule::LeadingPlus => {
        // Unary plus is a no-op: +expr is just expr. Simply skip.
      }
      Rule::LeadingMinus => {
        // Insert synthetic 0 and "NEGATE" operator so that -x^2 becomes 0 - x^2
        // NEGATE has higher precedence than * but lower than ^, matching
        // Wolfram Language's PreMinus precedence (between Times and Power).
        // This ensures ^ binds tighter than unary minus, and also works
        // correctly after other operators: a * -b^2 → a * (-(b^2))
        leading_minus = true;
      }
      Rule::LeadingNot => {
        // !expr becomes Not[expr]. Deferred until after suffixes so that
        // !a! → Not[Factorial[a]] rather than Factorial[Not[a]].
        leading_not = true;
      }
      Rule::Operator | Rule::ConditionOp => {
        flush_pending_not(
          &mut pending_not_on_last,
          &mut terms,
          &mut term_was_implicit_times,
        );
        operators.push(item.as_str().trim().to_string());
      }
      Rule::UnsetSuffix => {
        // f[x] =. → Unset[f[x]]: push "=." as an operator with a dummy
        // right operand so the precedence-climbing algorithm can handle it
        // like any other operator (at assignment precedence level 2).
        flush_pending_not(
          &mut pending_not_on_last,
          &mut terms,
          &mut term_was_implicit_times,
        );
        operators.push("=.".to_string());
        terms.push(Expr::Identifier("Null".to_string())); // dummy right operand
        term_was_implicit_times.push(false);
      }
      Rule::SpanNoOperandSep => {
        // `;;` on its own (`f[;;]`, `;; ;; 2`) — neither operand written.
        terms.push(Expr::Identifier(SPAN_OMITTED.to_string()));
        term_was_implicit_times.push(false);
        operators.push(";;".to_string());
        terms.push(Expr::Identifier(SPAN_OMITTED.to_string()));
        term_was_implicit_times.push(false);
      }
      Rule::SpanNoLhsSep => {
        // `;; 3` — a `;;` the chain has no left operand for. Stand the
        // omitted one in explicitly; `build_span` turns it into the `1`
        // default.
        terms.push(Expr::Identifier(SPAN_OMITTED.to_string()));
        term_was_implicit_times.push(false);
        operators.push(";;".to_string());
      }
      Rule::SpanNoRhsSep => {
        // `3 ;;`, and the gap in `1 ;;;; 3` — a `;;` with no right operand.
        flush_pending_not(
          &mut pending_not_on_last,
          &mut terms,
          &mut term_was_implicit_times,
        );
        operators.push(";;".to_string());
        terms.push(Expr::Identifier(SPAN_OMITTED.to_string()));
        term_was_implicit_times.push(false);
      }
      Rule::TildeInfix => {
        // a ~f~ b → f[a, b]: encode as "~funcName~" operator string
        // The inner pair is either BaseFunctionCall or Identifier
        flush_pending_not(
          &mut pending_not_on_last,
          &mut terms,
          &mut term_was_implicit_times,
        );
        let inner = item.into_inner().next().unwrap();
        let func_expr = pair_to_expr(inner);
        // Store as "~<encoded>~" for make_binary_op to handle
        let func_str = match &func_expr {
          Expr::Identifier(name) => format!("~{name}~"),
          _ => format!("~{}~", expr_to_string(&func_expr)),
        };
        operators.push(func_str);
      }
      Rule::FactorialSuffix => {
        // n! → Factorial[n], n!! → Factorial2[n]
        if let Some(last) = terms.pop() {
          let was_implicit = term_was_implicit_times.pop().unwrap_or(false);
          let func_name = if item.as_str() == "!!" {
            "Factorial2"
          } else {
            "Factorial"
          };
          // For an unparenthesised juxtaposition like `4z!`, attach the
          // factorial to the rightmost factor — `Times[4, Factorial[z]]` —
          // matching Wolfram's binding (factorial > implicit times). The
          // implicit-times builder yields a right-leaning BinaryOp Times
          // tree, so we descend into the right child.
          let mut new_term = last;
          let mut attached = false;
          if was_implicit {
            if let Expr::BinaryOp {
              op: BinaryOperator::Times,
              right,
              ..
            } = &mut new_term
            {
              let inner_right = std::mem::replace(
                right.as_mut(),
                Expr::Identifier(String::new()),
              );
              **right = Expr::FunctionCall {
                name: func_name.to_string(),
                args: vec![inner_right].into(),
              };
              attached = true;
            } else if let Expr::FunctionCall { name, args } = &mut new_term
              && name == "Times"
              && !args.is_empty()
            {
              let last_factor = args.pop().unwrap();
              args.push(Expr::FunctionCall {
                name: func_name.to_string(),
                args: vec![last_factor].into(),
              });
              attached = true;
            }
          }
          if !attached {
            new_term = Expr::FunctionCall {
              name: func_name.to_string(),
              args: vec![new_term].into(),
            };
          }
          terms.push(new_term);
          term_was_implicit_times.push(false);
        }
      }
      Rule::TransposeSuffix => {
        // expr \[Transpose] → Transpose[expr]
        if let Some(last) = terms.pop() {
          term_was_implicit_times.pop();
          terms.push(Expr::FunctionCall {
            name: "Transpose".to_string(),
            args: vec![last].into(),
          });
          term_was_implicit_times.push(false);
        }
      }
      Rule::PatternTestSuffix => {
        // expr?test → PatternTest[expr, test]. The left side is whatever
        // term precedes the `?` — a function call (`Except[0]?NumericQ`), a
        // list, a literal or a bracketed expression. Blank patterns
        // (`x_?test`) and the bare-infix form (`a?b`) never get here: their
        // own `PatternTest` rule consumes the `?` first.
        if let Some(last) = terms.pop() {
          term_was_implicit_times.pop();
          terms.push(build_pattern_test(last, item.into_inner()));
          term_was_implicit_times.push(false);
        }
      }
      Rule::ConjugateTransposeSuffix => {
        // expr \[ConjugateTranspose] → ConjugateTranspose[expr]
        if let Some(last) = terms.pop() {
          term_was_implicit_times.pop();
          terms.push(Expr::FunctionCall {
            name: "ConjugateTranspose".to_string(),
            args: vec![last].into(),
          });
          term_was_implicit_times.push(false);
        }
      }
      Rule::RepeatedSuffix | Rule::RepeatedNullSuffix => {
        // x.. → Repeated[x], x... → RepeatedNull[x]
        if let Some(last) = terms.pop() {
          term_was_implicit_times.pop();
          let func_name = if item.as_rule() == Rule::RepeatedNullSuffix {
            "RepeatedNull"
          } else {
            "Repeated"
          };
          terms.push(Expr::FunctionCall {
            name: func_name.to_string(),
            args: vec![last].into(),
          });
          term_was_implicit_times.push(false);
        }
      }
      _ => {
        if leading_minus {
          // When `^` is followed by `-`, replace the `^` operator with the
          // synthetic `^_NEG` (Power-with-negated-right) instead of inserting
          // a synthetic 0 NEGATE pair. This is required because Power has
          // higher precedence than NEGATE: `a^-b` would otherwise parse as
          // `(a^0) - b`. With `^_NEG`, the right operand of the chain
          // (e.g. `b^c` in `a^-b^c`) is wrapped in unary minus before being
          // used as the exponent — matching Wolfram's `Power[a, -(b^c)]`.
          if operators.last().is_some_and(|o| o == "^") {
            operators.pop();
            operators.push("^_NEG".to_string());
          } else {
            terms.push(Expr::Integer(0));
            term_was_implicit_times.push(false);
            operators.push("NEGATE".to_string());
          }
          leading_minus = false;
        }
        // `f[x] y` is an implicit product as much as `x y` is; the grammar
        // just reaches it through `FunctionCallExtended`'s implicit suffix
        // instead of `ImplicitTimes`. Both have to be spliceable, or a
        // neighbouring tighter operator takes the whole product as one
        // operand — `f[x] Times @@ list` would read as
        // `Apply[f[x] Times, list]`.
        let is_implicit_times = matches!(item.as_rule(), Rule::ImplicitTimes)
          || (matches!(item.as_rule(), Rule::FunctionCallExtended)
            && item
              .clone()
              .into_inner()
              .any(|p| p.as_rule() == Rule::FunctionCallImplicitSuffix));
        // When such a product follows a Divide operator (`/`), splice its
        // factors into the running term/operator vectors so that they stay
        // at multiplicative precedence — `a/b c` must parse as `(a*c)/b`,
        // not `a/(b*c)`. Without this, the whole product is consumed as a
        // single divisor.
        let split_implicit =
          is_implicit_times && operators.last().is_some_and(|o| o == "/");
        let expr = pair_to_expr(item);
        if split_implicit {
          let factors = flatten_times_chain(&expr);
          let mut iter = factors.into_iter();
          if let Some(first) = iter.next() {
            terms.push(first);
            term_was_implicit_times.push(false);
            for rest in iter {
              operators.push("*".to_string());
              terms.push(rest);
              term_was_implicit_times.push(false);
            }
          }
        } else {
          terms.push(expr);
          term_was_implicit_times.push(is_implicit_times);
        }
        if leading_not {
          // Defer the Not wrapping until any postfix suffixes have been
          // applied so that operators like `!` bind tighter than the prefix
          // Not, matching Wolfram precedence.
          pending_not_on_last = true;
          // If this is the first term (no operators yet), record the leading
          // `!` so we can re-apply it at the right precedence after the
          // binary tree is built. Mathematica's `!` (prec 230) is looser
          // than `+`, `==`, `@@`, `/@`, `@`, …, but tighter than `&&` (215)
          // and `||` (213). So `! Or @@ {…}` should parse as
          // `Not[Or @@ {…}]`, not `(Not[Or]) @@ {…}`.
          if terms.len() == 1 && operators.is_empty() {
            leading_not_at_start = true;
          }
          leading_not = false;
        }
      }
    }
  }
  // Decide whether to suppress the term-level Not in favour of wrapping
  // the whole tree after it is built. We only suppress when the *first*
  // infix operator binds tighter than `!`; otherwise the local wrap is
  // already correct (e.g. `!a && b` → `And[Not[a], b]`).
  let leading_not_on_first = leading_not_at_start && !operators.is_empty() && {
    let first_op_prec = operator_precedence(&operators[0]);
    first_op_prec > 18
  };
  if leading_not_on_first {
    // Undo the term-level wrap: terms[0] was set to Not[orig]; unwrap it.
    if let Expr::FunctionCall { name, args } = &terms[0]
      && name == "Not"
      && args.len() == 1
    {
      terms[0] = args[0].clone();
    }
    pending_not_on_last = false;
  }
  flush_pending_not(
    &mut pending_not_on_last,
    &mut terms,
    &mut term_was_implicit_times,
  );
  split_implicit_products(&mut terms, &mut operators, &term_was_implicit_times);

  let mut result = if terms.len() == 1 {
    terms.remove(0)
  } else {
    // Check if all operators are comparison operators
    let all_comparisons = operators.iter().all(|op| {
      matches!(
        op.as_str(),
        "=="
          | "\u{2A75}"
          | "\\[Equal]"
          | "\u{F431}"
          | "!="
          | "\u{2260}"
          | "\\[NotEqual]"
          | "<"
          | "<="
          | "\u{2264}"
          | "\u{2A7D}"
          | "\\[LessEqual]"
          | ">"
          | ">="
          | "\u{2265}"
          | "\u{2A7E}"
          | "\\[GreaterEqual]"
          | "==="
          | "=!="
      )
    });

    if all_comparisons && !operators.is_empty() {
      // Build comparison chain
      let comp_ops: Vec<ComparisonOp> = operators
        .iter()
        .map(|op| match op.as_str() {
          "==" | "\u{2A75}" | "\\[Equal]" | "\u{F431}" => ComparisonOp::Equal,
          "!=" | "\u{2260}" | "\\[NotEqual]" => ComparisonOp::NotEqual,
          "<" => ComparisonOp::Less,
          "<=" | "\u{2264}" | "\u{2A7D}" | "\\[LessEqual]" => {
            ComparisonOp::LessEqual
          }
          ">" => ComparisonOp::Greater,
          ">=" | "\u{2265}" | "\u{2A7E}" | "\\[GreaterEqual]" => {
            ComparisonOp::GreaterEqual
          }
          "===" => ComparisonOp::SameQ,
          "=!=" => ComparisonOp::UnsameQ,
          _ => ComparisonOp::Equal,
        })
        .collect();
      Expr::Comparison {
        operands: terms,
        operators: comp_ops,
      }
    } else {
      // Build binary operation tree (left-to-right for same precedence)
      build_binary_tree(terms, &operators)
    }
  };

  // If we deferred the leading `!` past tighter-binding operators, wrap
  // the (now-complete) sub-tree in Not now. For `! Or @@ {…}` this yields
  // `Not[Or @@ {…}]` rather than the wrong `(Not[Or]) @@ {…}`.
  if leading_not_on_first {
    result = Expr::FunctionCall {
      name: "Not".to_string(),
      args: vec![result].into(),
    };
  }

  // Apply ReplaceAll/ReplaceRepeated suffixes, folding a chain like
  // `a /. b /. c` left-to-right into `ReplaceAll[ReplaceAll[a, b], c]`.
  // In Wolfram Language, /. has higher precedence than = and :=, so:
  //   intA = expr /. rules  →  Set[intA, ReplaceAll[expr, rules]]
  // not: ReplaceAll[Set[intA, expr], rules]
  for (rules_pair, is_replace_repeated) in replace_rules {
    let rules = pair_to_expr(rules_pair);
    let make_replace = |e: Expr, r: Expr| -> Expr {
      if is_replace_repeated {
        Expr::ReplaceRepeated {
          expr: Box::new(e),
          rules: Box::new(r),
        }
      } else {
        Expr::ReplaceAll {
          expr: Box::new(e),
          rules: Box::new(r),
        }
      }
    };
    // If result is an assignment, push /. inside to the right-hand side. Only
    // the first suffix can meet a raw assignment; later folds wrap the
    // already-rewritten result.
    result = match result {
      Expr::FunctionCall { ref name, ref args }
        if (name == "Set" || name == "SetDelayed") && args.len() == 2 =>
      {
        let lhs = args[0].clone();
        let rhs = args[1].clone();
        Expr::FunctionCall {
          name: name.clone(),
          args: vec![lhs, make_replace(rhs, rules)].into(),
        }
      }
      _ => make_replace(result, rules),
    };
  }

  // Apply postfix functions. In Wolfram, `//` (postfix application) has
  // precedence 110 — higher than Set/SetDelayed (40) and TagSet/TagSetDelayed
  // (40), so `r = m // Grid` parses as `Set[r, Grid[m]]`, not as
  // `Grid[Set[r, m]]`. Push postfix wrappers into the RHS of the
  // assignment when one is present.
  for func_pair in postfix_funcs {
    let func = parse_postfix_function(func_pair);
    if let Expr::FunctionCall { name, args } = &mut result
      && matches!(
        name.as_str(),
        "Set" | "SetDelayed" | "UpSet" | "UpSetDelayed"
      )
      && args.len() == 2
    {
      let rhs = std::mem::replace(&mut args[1], Expr::Integer(0));
      args[1] = Expr::Postfix {
        expr: Box::new(rhs),
        func: Box::new(func),
      };
    } else {
      result = Expr::Postfix {
        expr: Box::new(result),
        func: Box::new(func),
      };
    }
  }

  // Apply AnonymousFunctionSuffix: expr &
  //
  // In Wolfram, `&` has precedence 90, which is higher (binds tighter) than
  // Set/SetDelayed/UpSet/UpSetDelayed (all 40) but lower than nearly every
  // other operator. For `f = body &` we must wrap only the RHS, producing
  // Set[f, Function[body]] — not Function[Set[f, body]]. For operators
  // tighter than `&` (e.g. `==`, `+`, `/.`, `->`) the whole infix chain is
  // already built in `result`, so wrapping it is correct.
  if let Some(bracket_args) = anon_func_suffix {
    // If `result` is a top-level assignment (Set/SetDelayed/UpSet/
    // UpSetDelayed), `&` (precedence 90) binds tighter than the assignment
    // (precedence 40), so we push the Function wrapper into the RHS.
    let pushed_into_assignment = if let Expr::FunctionCall { name, args } =
      &mut result
      && matches!(
        name.as_str(),
        "Set" | "SetDelayed" | "UpSet" | "UpSetDelayed"
      )
      && args.len() == 2
    {
      let rhs = std::mem::replace(&mut args[1], Expr::Integer(0));
      args[1] = Expr::Function {
        body: Box::new(rhs),
      };
      true
    } else {
      false
    };
    if !pushed_into_assignment {
      result = Expr::Function {
        body: Box::new(result),
      };
    }
    if pushed_into_assignment && !bracket_args.is_empty() {
      // `lhs = (body) & [args]` — both `&` and the `[args]` application
      // bind tighter than `=`, so the curried call belongs INSIDE the
      // assignment's RHS, not wrapped around the entire Set node.
      if let Expr::FunctionCall { args: set_args, .. } = &mut result {
        let mut rhs = std::mem::replace(&mut set_args[1], Expr::Integer(0));
        for args in bracket_args {
          rhs = Expr::CurriedCall {
            func: Box::new(rhs),
            args,
          };
        }
        set_args[1] = rhs;
      }
    } else {
      for args in bracket_args {
        result = Expr::CurriedCall {
          func: Box::new(result),
          args,
        };
      }
    }

    // Apply continuation operators after & (e.g., #^2& @ 3)
    if !post_anon_pairs.is_empty() {
      // If the pre-& result is an assignment (`a = body &`), the post-&
      // continuation operators (e.g. `/@ newlist`) almost always bind
      // tighter than `=`, so they should be applied to the RHS only.
      // Otherwise `a = body & /@ newlist` would wrongly parse as
      // `Map[Set[a, Function[body]], newlist]` and trigger Set::argrx.
      let assignment_lhs: Option<(String, Expr)> =
        if let Expr::FunctionCall { name, args } = &result
          && matches!(
            name.as_str(),
            "Set" | "SetDelayed" | "UpSet" | "UpSetDelayed"
          )
          && args.len() == 2
        {
          Some((name.clone(), args[0].clone()))
        } else {
          None
        };
      let starting_term = if let Some((_, _)) = &assignment_lhs {
        if let Expr::FunctionCall { args, .. } = &result {
          args[1].clone()
        } else {
          unreachable!()
        }
      } else {
        result
      };

      result = apply_anon_continuation(starting_term, post_anon_pairs);

      // Re-wrap the assignment around the now-extended RHS.
      if let Some((name, lhs)) = assignment_lhs {
        result = Expr::FunctionCall {
          name,
          args: vec![lhs, result].into(),
        };
      }
    }
  }

  result
}

/// Apply the continuation that can follow a `&`-terminated pure function:
/// operator/tilde-infix chains (`f & /@ list`), a further `&[args]`
/// application chained after (`f & [x] & [y]`), `/.`/`//.` replacements,
/// and trailing `// g` postfix, interleaved in any order. `starting_term`
/// is the pure function itself (already `Expr::Function`-wrapped);
/// `post_pairs` is everything the grammar captured after the `&`.
///
/// Shared by `Expression`'s own post-& handling and by
/// `ListItemRule`/`FunctionArgRule`, whose `pattern -> replacement &` is
/// exactly such a pure function and can take the same continuation —
/// `pattern -> replacement & /@ list` is the idiom for building a
/// `PopupMenu`'s choices from a lookup table.
fn apply_anon_continuation(
  starting_term: Expr,
  post_pairs: Vec<Pair<Rule>>,
) -> Expr {
  let mut post_terms: Vec<Expr> = Vec::new();
  let mut post_ops: Vec<String> = Vec::new();
  let mut post_postfix: Vec<Pair<Rule>> = Vec::new();

  // Collect postfix functions from the end of post-& pairs
  let mut post_pairs = post_pairs;
  while post_pairs
    .last()
    .is_some_and(|p| p.as_rule() == Rule::PostfixFunction)
  {
    post_postfix.push(post_pairs.pop().unwrap());
  }
  post_postfix.reverse();

  // Parse continuation as operator-term pairs
  post_terms.push(starting_term);
  // Each `&` chain may be followed by `/.` / `//.` suffixes, which
  // apply to the *result* of the application: `f & [x] /. rules`.
  #[allow(clippy::type_complexity)]
  let mut pending_anon_chains: Vec<(Vec<Vec<Expr>>, Vec<(Expr, bool)>)> =
    Vec::new();
  let mut leading_replaces: Vec<(Expr, bool)> = Vec::new();
  let mut iter = post_pairs.into_iter();
  while let Some(op_pair) = iter.next() {
    if op_pair.as_rule() == Rule::Operator {
      post_ops.push(op_pair.as_str().to_string());
      // Check for LeadingMinus after operator
      if let Some(next_pair) = iter.next() {
        if next_pair.as_rule() == Rule::LeadingMinus {
          // Use `^_NEG` for `^-` (see comment in main expression branch).
          if post_ops.last().is_some_and(|o| o == "^") {
            post_ops.pop();
            post_ops.push("^_NEG".to_string());
          } else {
            post_terms.push(Expr::Integer(0));
            post_ops.push("NEGATE".to_string());
          }
          if let Some(term_pair) = iter.next() {
            post_terms.push(pair_to_expr(term_pair));
          }
        } else {
          post_terms.push(pair_to_expr(next_pair));
        }
      }
    } else if op_pair.as_rule() == Rule::TildeInfix {
      let inner = op_pair.into_inner().next().unwrap();
      let func_expr = pair_to_expr(inner);
      let func_str = match &func_expr {
        Expr::Identifier(name) => format!("~{name}~"),
        _ => format!("~{}~", expr_to_string(&func_expr)),
      };
      post_ops.push(func_str);
      if let Some(next_pair) = iter.next() {
        if next_pair.as_rule() == Rule::LeadingMinus {
          post_terms.push(Expr::Integer(0));
          post_ops.push("NEGATE".to_string());
          if let Some(term_pair) = iter.next() {
            post_terms.push(pair_to_expr(term_pair));
          }
        } else {
          post_terms.push(pair_to_expr(next_pair));
        }
      }
    } else if op_pair.as_rule() == Rule::AnonymousFunctionSuffix {
      // (TildeInfix doesn't need `^_NEG` handling — Power is never the
      // operator immediately preceding a tilde infix.)
      // A second `&` after the first: wrap the accumulated infix chain
      // built so far (including any infix continuation) in a Function
      // and apply its BracketArgs. Matches Wolfram's `f & [x] & [y]`
      // ≡ `((f &)[x] &)[y]`.
      let bracket_args: Vec<Vec<Expr>> = op_pair
        .into_inner()
        .filter(|p| matches!(p.as_rule(), Rule::BracketArgs))
        .map(|bracket| bracket.into_inner().map(pair_to_expr).collect())
        .collect();
      pending_anon_chains.push((bracket_args, Vec::new()));
    } else if matches!(
      op_pair.as_rule(),
      Rule::ReplaceAllSuffix | Rule::ReplaceRepeatedSuffix
    ) {
      // `f & [x] /. rules` replaces in the *result* of the
      // application. Before any further `&`, that result is the term
      // chain built so far; after one, it is that chain's own result.
      let repeated = op_pair.as_rule() == Rule::ReplaceRepeatedSuffix;
      let rules = pair_to_expr(op_pair.into_inner().next().unwrap());
      match pending_anon_chains.last_mut() {
        Some((_, replaces)) => replaces.push((rules, repeated)),
        None => leading_replaces.push((rules, repeated)),
      }
    }
  }

  // Build expression tree with precedence
  let mut result = build_binary_tree(post_terms, &post_ops);

  // A `/.` between the first `&` application and any further one.
  for (rules, repeated) in leading_replaces {
    result = if repeated {
      Expr::ReplaceRepeated {
        expr: Box::new(result),
        rules: Box::new(rules),
      }
    } else {
      Expr::ReplaceAll {
        expr: Box::new(result),
        rules: Box::new(rules),
      }
    };
  }

  // Apply any additional `& [...]` chains that followed the first one.
  for (bracket_args, replaces) in pending_anon_chains {
    result = Expr::Function {
      body: Box::new(result),
    };
    for args in bracket_args {
      result = Expr::CurriedCall {
        func: Box::new(result),
        args,
      };
    }
    for (rules, repeated) in replaces {
      result = if repeated {
        Expr::ReplaceRepeated {
          expr: Box::new(result),
          rules: Box::new(rules),
        }
      } else {
        Expr::ReplaceAll {
          expr: Box::new(result),
          rules: Box::new(rules),
        }
      };
    }
  }

  // Apply post-& postfix functions
  for func_pair in post_postfix {
    let func = parse_postfix_function(func_pair);
    result = Expr::Postfix {
      expr: Box::new(result),
      func: Box::new(func),
    };
  }

  result
}

fn parse_compound_expression(pair: &Pair<Rule>) -> Expr {
  // Use each child Expression's span position to detect `;` separators
  // that have no Expression between them (Wolfram: `a ; ; c` →
  // CompoundExpression[a, Null, c]). Pest doesn't emit pairs for literal
  // `;` tokens, so we reconstruct the positions of omitted Expressions
  // by scanning the source string for top-level `;`s between children.
  let src = pair.as_str();
  let src_start = pair.as_span().start();
  let children: Vec<_> = pair.clone().into_inner().collect();
  // A `&` with no statement between it and the preceding `;` (the
  // `(a = #; t = 0; &)` idiom — see the grammar comment on
  // `CompoundExpression`) turns the statement sequence built so far into a
  // pure function body. It rides along as a `CompoundAnonSuffix` child,
  // together with whatever continuation followed it (`& /@ list`); the loop
  // below folds it in rather than treating it as a statement.
  let stmts_end = src_start + src.len();
  let mut exprs: Vec<Expr> = Vec::new();
  // Count the number of top-level `;` separators between `lo` and `hi`
  // (absolute offsets into the original input). `;;` is treated as a
  // Span separator and counted as zero semicolons.
  let count_separators = |lo: usize, hi: usize| -> usize {
    let local_lo = lo.saturating_sub(src_start);
    let local_hi = hi.saturating_sub(src_start);
    let slice = &src[local_lo.min(src.len())..local_hi.min(src.len())];
    let bytes = slice.as_bytes();
    let mut depth: i32 = 0;
    let mut i = 0;
    let mut count = 0usize;
    while i < bytes.len() {
      let c = bytes[i];
      match c {
        b'(' | b'[' | b'{' => depth += 1,
        b')' | b']' | b'}' => depth -= 1,
        b'"' => {
          // Skip string literal
          i += 1;
          while i < bytes.len() && bytes[i] != b'"' {
            if bytes[i] == b'\\' && i + 1 < bytes.len() {
              i += 1;
            }
            i += 1;
          }
        }
        b';' if depth == 0 => {
          // Skip `;;` (Span) — it's two chars, not two separators.
          if i + 1 < bytes.len() && bytes[i + 1] == b';' {
            i += 1;
          } else {
            count += 1;
          }
        }
        _ => {}
      }
      i += 1;
    }
    count
  };
  let mut prev_end = src_start;
  for (idx, child) in children.iter().enumerate() {
    let span = child.as_span();
    if child.as_rule() == Rule::CompoundAnonSuffix {
      // Every `;` between the last statement and the `&` is a statement of
      // its own — `a; &` is `Function[CompoundExpression[a, Null]]`, the
      // same `Null` any other trailing `;` appends.
      let trailing = count_separators(prev_end, span.start());
      for _ in 0..trailing {
        exprs.push(Expr::Identifier("Null".to_string()));
      }
      let body = if exprs.len() == 1 {
        exprs.pop().unwrap()
      } else {
        Expr::CompoundExpr(std::mem::take(&mut exprs))
      };
      exprs.push(parse_compound_anon_suffix(body, child.clone()));
      prev_end = span.end();
      continue;
    }
    if idx > 0 {
      // Missing expressions between the previous child and this one:
      // count separators minus 1 (one separator connects two expressions).
      let seps = count_separators(prev_end, span.start());
      for _ in 1..seps {
        exprs.push(Expr::Identifier("Null".to_string()));
      }
    }
    exprs.push(pair_to_expr(child.clone()));
    prev_end = span.end();
  }
  // Trailing separators after the last child → each is a Null.
  let trailing = count_separators(prev_end, stmts_end);
  for _ in 0..trailing {
    exprs.push(Expr::Identifier("Null".to_string()));
  }
  if exprs.len() == 1 {
    exprs.into_iter().next().unwrap()
  } else {
    Expr::CompoundExpr(exprs)
  }
}

/// Turn the statement sequence collected so far into the pure function a
/// `CompoundAnonSuffix` closes, then apply that suffix's own continuation.
///
/// `body` is everything to the left of the `&`; `pair` is the
/// `CompoundAnonSuffix` node holding the `&` (with any immediately-chained
/// `[args]`) and the operator/replace/postfix continuation that followed it,
/// so `a = #; & /@ list` maps the multi-statement function over `list`.
fn parse_compound_anon_suffix(body: Expr, pair: Pair<Rule>) -> Expr {
  let mut inner: Vec<Pair<Rule>> = pair.into_inner().collect();
  let mut result = Expr::Function {
    body: Box::new(body),
  };
  // The `&` itself, plus any `(…; &)[args]` application chained onto it.
  if !inner.is_empty() && inner[0].as_rule() == Rule::AnonymousFunctionSuffix {
    for bracket in inner
      .remove(0)
      .into_inner()
      .filter(|p| matches!(p.as_rule(), Rule::BracketArgs))
    {
      result = Expr::CurriedCall {
        func: Box::new(result),
        args: bracket.into_inner().map(pair_to_expr).collect(),
      };
    }
  }
  apply_anon_continuation(result, inner)
}

fn parse_association_extended(pair: Pair<Rule>) -> Expr {
  let inner_pairs: Vec<_> = pair.into_inner().collect();
  // First child is always the Association
  let base_expr = pair_to_expr(inner_pairs[0].clone());
  // Check whether this is a bracket call or Part extraction
  let has_call_suffix = inner_pairs
    .iter()
    .any(|p| matches!(p.as_rule(), Rule::AssociationCallSuffix));
  if has_call_suffix {
    // <|...|>[args] -> CurriedCall
    let bracket_sequences: Vec<Vec<Expr>> = inner_pairs
      .iter()
      .filter(|p| matches!(p.as_rule(), Rule::AssociationCallSuffix))
      .flat_map(|p| p.clone().into_inner())
      .filter(|p| matches!(p.as_rule(), Rule::BracketArgs))
      .map(|bracket| {
        bracket
          .into_inner()
          .filter(|p| {
            p.as_str() != "[" && p.as_str() != "]" && p.as_str() != ","
          })
          .map(pair_to_expr)
          .collect()
      })
      .collect();
    let mut result = Expr::CurriedCall {
      func: Box::new(base_expr),
      args: bracket_sequences[0].clone(),
    };
    for args in bracket_sequences.into_iter().skip(1) {
      result = Expr::CurriedCall {
        func: Box::new(result),
        args,
      };
    }
    result
  } else {
    // <|...|>[[index]] -> Part[assoc, index]
    let part_suffixes: Vec<Pair<Rule>> = inner_pairs
      .iter()
      .filter(|p| matches!(p.as_rule(), Rule::PartIndexSuffix))
      .cloned()
      .collect();
    part_suffixes
      .iter()
      .fold(base_expr, |acc, p| apply_part_index_suffix(acc, p.clone()))
  }
}

fn parse_association(pair: Pair<Rule>) -> Expr {
  // Association literal: <|item, item, ...|>.
  // If every item is a Rule (AssociationItem), produce Expr::Association.
  // Otherwise fall back to a FunctionCall so AssociationQ can return False
  // for malformed inputs like <|a, b|>.
  let inner_pairs: Vec<_> = pair.into_inner().collect();
  let all_rules = inner_pairs
    .iter()
    .all(|p| p.as_rule() == Rule::AssociationItem);
  if all_rules {
    let items: Vec<(Expr, Expr)> = inner_pairs
      .into_iter()
      .map(|item| {
        // Detect `key :> val` from the raw item text. `:>` keeps the
        // RuleDelayed marker (value is not evaluated eagerly) per the
        // Expr::Association convention used by the formatter.
        let item_text = item.as_str();
        let is_delayed = is_assoc_item_delayed(item_text);
        let mut inner = item.into_inner();
        let key = pair_to_expr(inner.next().unwrap());
        let val = pair_to_expr(inner.next().unwrap());
        if is_delayed {
          (
            key.clone(),
            Expr::RuleDelayed {
              pattern: Box::new(key),
              replacement: Box::new(val),
            },
          )
        } else {
          (key, val)
        }
      })
      .collect();
    Expr::Association(items)
  } else {
    let args: Vec<Expr> = inner_pairs
      .into_iter()
      .map(|p| match p.as_rule() {
        Rule::AssociationItem => {
          let mut inner = p.into_inner();
          let key = pair_to_expr(inner.next().unwrap());
          let val = pair_to_expr(inner.next().unwrap());
          Expr::Rule {
            pattern: Box::new(key),
            replacement: Box::new(val),
          }
        }
        _ => pair_to_expr(p.into_inner().next().unwrap()),
      })
      .collect();
    Expr::FunctionCall {
      name: "Association".to_string(),
      args: args.into(),
    }
  }
}

fn parse_association_item(pair: Pair<Rule>) -> Expr {
  let mut inner = pair.into_inner();
  let key = pair_to_expr(inner.next().unwrap());
  let val = pair_to_expr(inner.next().unwrap());
  Expr::Rule {
    pattern: Box::new(key),
    replacement: Box::new(val),
  }
}

fn parse_paren_extended(pair: Pair<Rule>) -> Expr {
  let inner_pairs: Vec<_> = pair.into_inner().collect();
  let base_expr = pair_to_expr(inner_pairs[0].clone());
  // Check for derivative prime first: (expr)' -> Derivative[n][expr]
  // and (expr)'[args] -> Derivative[n][expr][args].
  let prime_pair = inner_pairs
    .iter()
    .find(|p| matches!(p.as_rule(), Rule::DerivativePrime));
  if let Some(prime) = prime_pair {
    let order = prime.as_str().chars().filter(|c| *c == '\'').count();
    let derivative_head = Expr::FunctionCall {
      name: "Derivative".to_string(),
      args: vec![Expr::Integer(order as i128)].into(),
    };
    let derivative_call = Expr::CurriedCall {
      func: Box::new(derivative_head),
      args: vec![base_expr],
    };
    // Optional bracket call: (expr)'[args]
    let call_suffix = inner_pairs
      .iter()
      .find(|p| matches!(p.as_rule(), Rule::ParenCallSuffix));
    if let Some(suffix) = call_suffix {
      let bracket_sequences: Vec<Vec<Expr>> = suffix
        .clone()
        .into_inner()
        .filter(|p| matches!(p.as_rule(), Rule::BracketArgs))
        .map(|bracket| {
          bracket
            .into_inner()
            .filter(|p| {
              p.as_str() != "[" && p.as_str() != "]" && p.as_str() != ","
            })
            .map(pair_to_expr)
            .collect()
        })
        .collect();
      let mut result = derivative_call;
      for args in bracket_sequences {
        result = Expr::CurriedCall {
          func: Box::new(result),
          args,
        };
      }
      return result;
    }
    return derivative_call;
  }
  // Check whether this is a Part extraction or a bracket call
  let has_call_suffix = inner_pairs
    .iter()
    .any(|p| matches!(p.as_rule(), Rule::ParenCallSuffix));
  if has_call_suffix {
    // (expr)[args] -> CurriedCall: treat parenthesized expr as function head
    let bracket_sequences: Vec<Vec<Expr>> = inner_pairs
      .iter()
      .filter(|p| matches!(p.as_rule(), Rule::ParenCallSuffix))
      .flat_map(|p| p.clone().into_inner())
      .filter(|p| matches!(p.as_rule(), Rule::BracketArgs))
      .map(|bracket| {
        bracket
          .into_inner()
          .filter(|p| {
            p.as_str() != "[" && p.as_str() != "]" && p.as_str() != ","
          })
          .map(pair_to_expr)
          .collect()
      })
      .collect();
    let mut result = Expr::CurriedCall {
      func: Box::new(base_expr),
      args: bracket_sequences[0].clone(),
    };
    for args in bracket_sequences.into_iter().skip(1) {
      result = Expr::CurriedCall {
        func: Box::new(result),
        args,
      };
    }
    result
  } else {
    // (expr)[[index]] -> Part[expr, index]
    let part_suffixes: Vec<Pair<Rule>> = inner_pairs
      .iter()
      .filter(|p| matches!(p.as_rule(), Rule::PartIndexSuffix))
      .cloned()
      .collect();
    part_suffixes
      .iter()
      .fold(base_expr, |acc, p| apply_part_index_suffix(acc, p.clone()))
  }
}

/// Build a call to a Flat logical head, collapsing a same-head left
/// operand: Xor[a, b] \[Xor] c becomes Xor[a, b, c].
fn flat_logical_call(head: &str, left: &Expr, right: &Expr) -> Expr {
  let mut args: Vec<Expr> = match left {
    Expr::FunctionCall { name, args } if name == head => {
      args.iter().cloned().collect()
    }
    _ => vec![left.clone()],
  };
  args.push(right.clone());
  Expr::FunctionCall {
    name: head.to_string(),
    args: args.into(),
  }
}

/// Named-character operators that still need a right-hand operand, in both the
/// long `\[Name]` spelling and the character it stands for. A line ending in
/// one of these continues onto the next line, exactly like a trailing `+`.
///
/// Every infix operator of the grammar's `Operator` rule is here, plus the
/// `RadicalOp` prefixes. The postfix operators (`\[Transpose]`,
/// `\[ConjugateTranspose]`) are deliberately absent: they complete an
/// expression rather than open one, so a line ending in one is a whole
/// statement. Keep in sync with the grammar and [`operator_precedence`].
pub const CONTINUING_NAMED_OPERATORS: &[&str] = &[
  "\\[NotElement]",
  "\u{2209}",
  "\\[ReverseElement]",
  "\u{220B}",
  "\\[Element]",
  "\u{2208}",
  "\\[DirectedEdge]",
  "\u{F3D5}",
  "\\[UndirectedEdge]",
  "\u{F3D4}",
  "\\[Distributed]",
  "\u{F3D2}",
  "\\[Conditioned]",
  "\u{F3D3}",
  "\\[Function]",
  "\u{F4A1}",
  "\\[Cross]",
  "\u{F4A0}",
  "\u{F3C4}",
  "\u{2A2F}",
  "\\[TensorProduct]",
  "\u{F3DA}",
  "\\[CircleTimes]",
  "\u{2297}",
  "\\[CirclePlus]",
  "\u{2295}",
  "\\[CenterDot]",
  "\u{00B7}",
  "\\[Times]",
  "\u{00D7}",
  "\\[Divide]",
  "\u{00F7}",
  "\\[CircleMinus]",
  "\u{2296}",
  "\\[Star]",
  "\u{22C6}",
  "\\[Diamond]",
  "\u{22C4}",
  "\\[Backslash]",
  "\u{2216}",
  "\\[CircleDot]",
  "\u{2299}",
  "\\[SmallCircle]",
  "\u{2218}",
  "\\[Wedge]",
  "\u{22C0}",
  "\\[Vee]",
  "\u{22C1}",
  "\\[Cap]",
  "\u{2322}",
  "\\[Cup]",
  "\u{2323}",
  "\\[DoubleRightTee]",
  "\u{22A8}",
  "\\[RightTee]",
  "\u{22A2}",
  "\\[DoubleLeftTee]",
  "\u{2AE4}",
  "\\[LeftTee]",
  "\u{22A3}",
  "\\[Implies]",
  "\u{F523}",
  "\\[Equivalent]",
  "\u{29E6}",
  "\\[Nand]",
  "\u{22BC}",
  "\\[Nor]",
  "\u{22BD}",
  "\\[Xor]",
  "\u{22BB}",
  "\\[And]",
  "\u{2227}",
  "\\[Or]",
  "\u{2228}",
  "\\[Not]",
  "\u{00AC}",
  "\\[Sqrt]",
  "\u{221A}",
  "\\[CubeRoot]",
  "\u{221B}",
  "\\[Rule]",
  "\u{F522}",
  "\\[RuleDelayed]",
  "\u{F51F}",
  "\\[Equal]",
  "\u{F431}",
  "\\[NotEqual]",
  "\\[LessEqual]",
  "\\[GreaterEqual]",
];

/// Does this line's code (whitespace already squeezed out) end with a named
/// operator that is still waiting for its right operand? A single trailing
/// `]` is an ordinary closing bracket, so only a full `\[Name]` — or the
/// character it stands for — counts.
pub fn ends_with_continuing_named_operator(code_tail: &str) -> bool {
  CONTINUING_NAMED_OPERATORS
    .iter()
    .any(|op| code_tail.ends_with(op))
}

/// Get precedence of an operator (higher = binds tighter).
/// Matches Wolfram Language operator precedence ordering.
fn operator_precedence(op: &str) -> u8 {
  match op {
    ">>" | ">>>" => 0,      // Put/PutAppend (lowest precedence)
    "/:" => 3, // TagSet/TagSetDelayed (lower than assignment so RHS includes :=)
    "=" | ":=" | "=." => 6, // Assignment / Unset
    // Compound assignment binds like plain assignment (Wolfram gives all of
    // them precedence 70), so `f[x] += a + b` adds the whole sum.
    "+=" | "-=" | "*=" | "/=" => 6,
    "^=" | "^:=" => 6, // UpSet/UpSetDelayed (same as assignment)
    // Condition binds *tighter* than Rule/RuleDelayed in Wolfram (130 vs 120),
    // so `lhs /; test :> rhs` is `RuleDelayed[Condition[lhs, test], rhs]` and
    // `lhs :> rhs /; test` is `RuleDelayed[lhs, Condition[rhs, test]]`.
    "/;" => 12,
    "->" | "\u{2192}" | "\\[Rule]" | "\u{F522}" | ":>" | "\\[RuleDelayed]"
    | "\u{F51F}" => 11, // Rule/RuleDelayed (lower than boolean operators)
    "||" => 15,                              // Or
    "&&" => 18,                              // And
    "\\[And]" | "\u{2227}" => 18,            // \[And] (same as &&)
    "\\[Nand]" | "\u{22BC}" => 18,           // \[Nand] (And level)
    "\\[Xor]" | "\u{22BB}" => 16,            // \[Xor] (between Or and And)
    "\\[Or]" | "\u{2228}" => 15,             // \[Or] (same as ||)
    "\\[Nor]" | "\u{22BD}" => 15,            // \[Nor] (Or level)
    "\\[Equivalent]" | "\u{29E6}" => 14,     // \[Equivalent] (below Or)
    "\\[Implies]" | "\u{F523}" => 13, // \[Implies] (lowest logical, right-assoc)
    "\\[NotElement]" | "\u{2209}" => 21, // NotElement (same level as comparisons)
    "\\[ReverseElement]" | "\u{220B}" => 21, // ReverseElement (same level as comparisons)
    "\\[Element]" | "\u{2208}" => 21, // Element (same level as comparisons)
    "\\[DirectedEdge]" | "\u{F3D5}" => 21, // DirectedEdge (same level as comparisons)
    "\\[UndirectedEdge]" | "\u{F3D4}" => 21, // UndirectedEdge (same level as comparisons)
    "<->" => 21, // TwoWayRule (same level as comparisons, tighter than Rule)
    "\\[Distributed]" | "\u{F3D2}" => 21, // Distributed (same level as comparisons)
    "\\[Conditioned]" | "\u{F3D3}" => 10, // Conditioned (looser than Rule)
    // Cross and TensorProduct bind tighter than Dot in Wolfram
    // (Precedence 500 and 495 vs Dot's 490): a.b\[Cross]c is a.(b\[Cross]c).
    "\\[Cross]" | "\u{F4A0}" | "\u{F3C4}" | "\u{2A2F}" => 42, // Cross (above TensorProduct)
    "\\[TensorProduct]" | "\u{F3DA}" => 41, // TensorProduct (just above Dot)
    "\\[Cap]" | "\u{2322}" => 36,           // Cap (⌢, infix → Cap[a, b])
    "\\[Cup]" | "\u{2323}" => 36,           // Cup (⌣, infix → Cup[a, b])
    "\\[RightTee]" | "\u{22A2}" => 15, // RightTee (⊢, right-assoc, between -> and ==)
    "\\[DoubleRightTee]" | "\u{22A8}" => 15, // DoubleRightTee (⊨, right-assoc, same level)
    "\\[LeftTee]" | "\u{22A3}" => 15, // LeftTee (⊣, left-assoc, same level)
    "\\[DoubleLeftTee]" | "\u{2AE4}" => 15, // DoubleLeftTee (⫤, left-assoc, same level)
    // wolframscript: \[Function] is lower than Set, Condition, and Rule —
    // the right operand absorbs y, y = 1, y /; z, y -> 1. Place at TagSet
    // level so its rhs stays maximally permissive.
    "\\[Function]" | "\u{F4A1}" | "|->" => 3,
    "==" | "\u{2A75}" | "\\[Equal]" | "\u{F431}" | "!=" | "\u{2260}"
    | "\\[NotEqual]" | "<" | "<=" | "\u{2264}" | "\u{2A7D}"
    | "\\[LessEqual]" | ">" | ">=" | "\u{2265}" | "\u{2A7E}"
    | "\\[GreaterEqual]" | "===" | "=!=" => 21, // Comparisons
    // `;;` (Span, Wolfram precedence 305) sits just below `+` (310) and
    // above every operator Wolfram places under it — `|` (160), `~~` (135),
    // the comparisons, `&&`, `||`, `->`, `/;` and the assignments. Without a
    // slot here those operators end up *inside* the Span (issue #564).
    ";;" => 29,
    "~~" => 24, // StringExpression (lower than Alternatives)
    // Pattern/Optional `:` sits between StringExpression and Alternatives,
    // as in Wolfram (135 < 150 < 160): `x : a | b : v` is
    // `Optional[Pattern[x, Alternatives[a, b]], v]`.
    ":" => 26,
    "|" => 27, // Alternatives (higher than StringExpression, Or, And, Rule)
    "+" | "-" => 30, // Plus/Minus
    "*" | "/" => 33, // Times/Divide
    // `\[Times]` (×) and `\[Divide]` (÷) are the named infix spellings of
    // `*` and `/`, so they bind at the same level.
    "\\[Times]" | "\u{00D7}" => 33,
    "\\[Divide]" | "\u{00F7}" => 33,
    "<>" => 30, // StringJoin (same level as Plus)
    // Symbolic ring operators, ordered to match wolframscript:
    // Dot > CircleTimes > CenterDot > Times > … > CirclePlus > Plus.
    "\\[CirclePlus]" | "\u{2295}" => 31, // between Plus and Times
    "\\[CircleMinus]" | "\u{2296}" => 31, // same level as CirclePlus
    "\\[Star]" | "\u{22C6}" => 32,       // between CirclePlus and Times
    "\\[CenterDot]" | "\u{00B7}" => 34,  // just above Times
    "\\[CircleTimes]" | "\u{2297}" => 35, // above CenterDot, below Vee
    "\\[Vee]" | "\u{22C1}" => 36,        // above CircleTimes, below Wedge
    "\\[Wedge]" | "\u{22C0}" => 37,      // above Vee, below Diamond
    "\\[Diamond]" | "\u{22C4}" => 38,    // above Wedge, below Backslash
    "\\[Backslash]" | "\u{2216}" => 39,  // above Diamond, below Dot
    "." => 40,                           // Dot (higher than the ring ops)
    "\\[CircleDot]" | "\u{2299}" => 43,  // above Cross, below SmallCircle
    "\\[SmallCircle]" | "\u{2218}" => 44, // above CircleDot, below Apply
    "NEGATE" => 45, // Unary minus (PreMinus): between Times/Dot and Apply/Map,
    // so `-f @@ list` reads as `-(f @@ list)` and `-f /@ list` as
    // `-(f /@ list)`, matching Wolfram (Apply/Map bind tighter than a
    // leading unary minus).
    "@@@" | "@@" => 46,  // Apply/MapApply
    "/@" => 47,          // Map (higher than Apply)
    "^" | "^_NEG" => 50, // Power (`^_NEG` is `a^-b` with negated right operand)
    s if s.starts_with('~') && s.ends_with('~') && s.len() > 2 => 53, // Tilde infix: a ~f~ b (higher than ^, lower than @)
    "@" => 56, // Prefix application
    // Composition/RightComposition bind tighter than prefix application (so
    // `f @* g @ x` parses as `(f @* g) @ x`) and Map (`Length@*f /@ list`
    // parses as `(Length@*f) /@ list`), but looser than MessageName.
    "@*" | "/*" => 57,
    "::" => 59, // MessageName (highest — a::b binds tighter than everything)
    _ => 0,
  }
}

/// Placeholder term for a `Span` operand the source left out — the `1` of
/// `1 ;; 3` when it is written `;; 3`, or the `3` when it is written `1 ;;`.
/// Which value fills the gap depends on the slot it sits in, so the omission
/// has to survive precedence climbing and is resolved by [`build_span`]. The
/// leading control character keeps the marker distinct from every symbol a
/// program could spell.
const SPAN_OMITTED: &str = "\u{1}SpanOmitted";

/// Is this term an omitted `Span` operand?
fn is_span_omitted(expr: &Expr) -> bool {
  matches!(expr, Expr::Identifier(name) if name == SPAN_OMITTED)
}

/// The stand-in for an omitted `Span` operand that never reached
/// [`build_span`]; see the `;;` arm of [`make_binary_op`].
fn span_omitted_fallback(expr: &Expr) -> Expr {
  if is_span_omitted(expr) {
    Expr::Integer(1)
  } else {
    expr.clone()
  }
}

/// Assemble `Span[…]` from the operands of one `;;` chain, filling in the
/// default for every operand the source left out: `1` for the start, `All`
/// for the end and `1` for the step — `;;3` is `Span[1, 3]`, `3;;` is
/// `Span[3, All]` and `1;;;;3` is `Span[1, All, 3]`.
fn build_span(parts: Vec<Expr>) -> Expr {
  let args: Vec<Expr> = parts
    .into_iter()
    .enumerate()
    .map(|(slot, part)| {
      if is_span_omitted(&part) {
        if slot == 1 {
          Expr::Identifier("All".to_string())
        } else {
          Expr::Integer(1)
        }
      } else {
        part
      }
    })
    .collect();
  Expr::FunctionCall {
    name: "Span".to_string(),
    args: args.into(),
  }
}

/// Build a binary operation tree from terms and operators with correct precedence
fn build_binary_tree(terms: Vec<Expr>, operators: &[String]) -> Expr {
  if terms.len() == 1 {
    return terms.into_iter().next().unwrap();
  }
  if terms.is_empty() {
    return Expr::Raw(String::new());
  }

  // Use a precedence climbing algorithm
  build_expr_with_precedence(&terms, operators, 0, 0)
}

/// Build expression with precedence climbing
fn build_expr_with_precedence(
  terms: &[Expr],
  operators: &[String],
  term_start: usize,
  min_prec: u8,
) -> Expr {
  if term_start >= terms.len() {
    return Expr::Raw(String::new());
  }

  let mut result = terms[term_start].clone();
  let mut op_idx = term_start; // operators[i] is between terms[i] and terms[i+1]
  // Was the operator that built the current `result` the same ⨯ chain we are
  // extending? Only then may `a ⨯ b ⨯ c` collapse to `Cross[a, b, c]`; a
  // `Cross[…]` that came from parentheses or an explicit call is an operand.
  let mut in_cross_chain = false;

  while op_idx < operators.len() {
    let op_str = &operators[op_idx];
    let prec = operator_precedence(op_str);

    if prec < min_prec {
      break;
    }

    // `;;` groups n-ary: `1 ;; 2 ;; 3` is one `Span[1, 2, 3]`, never
    // `Span[Span[1, 2], 3]`. The whole run of `;;` operators at this level is
    // taken at once, which keeps a Span that came from parentheses nested —
    // `(1 ;; 2) ;; 3` has only one `;;` here, the other one is inside a term.
    if op_str == ";;" {
      // Left-associative, like every other operator without its own case.
      let next_min_prec = prec + 1;
      let mut parts = vec![result];
      loop {
        parts.push(build_expr_with_precedence(
          terms,
          operators,
          op_idx + 1,
          next_min_prec,
        ));
        // Skip past the terms the right operand consumed, as below.
        let mut right_terms = 1;
        let mut i = op_idx + 1;
        while i < operators.len()
          && operator_precedence(&operators[i]) >= next_min_prec
        {
          right_terms += 1;
          i += 1;
        }
        op_idx += right_terms;
        if operators.get(op_idx).map(String::as_str) != Some(";;") {
          break;
        }
      }
      result = build_span(parts);
      in_cross_chain = false;
      continue;
    }

    // For right-associative operators, use prec, otherwise use prec + 1
    let next_min_prec = if op_str == "^"
      || op_str == "^_NEG"
      || op_str == "@"
      || op_str == "="
      || op_str == ":="
      || op_str == "/:"
      || op_str == "@@"
      || op_str == "@@@"
      || op_str == "/@"
      || op_str == "@*"
      || op_str == "/*"
      || op_str == "->"
      || op_str == "\u{2192}"
      || op_str == "\\[Rule]"
      || op_str == "\u{F522}"
      || op_str == ":>"
      || op_str == "\\[RuleDelayed]"
      || op_str == "\u{F51F}"
      || op_str == "\\[RightTee]"
      || op_str == "\u{22A2}"
      || op_str == "\\[Implies]"
      || op_str == "\u{F523}"
      || op_str == "\\[DoubleRightTee]"
      || op_str == "\u{22A8}"
    {
      prec
    } else {
      prec + 1
    };

    // Build the right side with higher precedence
    let right =
      build_expr_with_precedence(terms, operators, op_idx + 1, next_min_prec);

    // Create the binary operation
    result = if is_cross_op(op_str) && !in_cross_chain {
      // First ⨯ of a chain: group binary, so `(a ⨯ b) ⨯ c` and
      // `Cross[a, b] ⨯ c` both stay `Cross[Cross[a, b], c]`.
      Expr::FunctionCall {
        name: "Cross".to_string(),
        args: vec![result.clone(), right.clone()].into(),
      }
    } else {
      make_binary_op(&result, op_str, &right)
    };
    in_cross_chain = is_cross_op(op_str);

    // Count how many terms were consumed on the right
    let mut right_terms = 1;
    let mut i = op_idx + 1;
    while i < operators.len()
      && operator_precedence(&operators[i]) >= next_min_prec
    {
      right_terms += 1;
      i += 1;
    }
    op_idx += right_terms;
  }

  result
}

/// Is this operator token one of the spellings of the `⨯` (`Cross`) infix
/// operator?
fn is_cross_op(op_str: &str) -> bool {
  matches!(op_str, "\\[Cross]" | "\u{F4A0}" | "\u{F3C4}" | "\u{2A2F}")
}

/// `p : v` where `p` is not a bare name: `Optional[p, v]`, and `sym : v`:
/// `Pattern[sym, v]`.
fn colon_node(left: &Expr, right: &Expr) -> Expr {
  let head = if matches!(left, Expr::Identifier(_)) {
    "Pattern"
  } else {
    "Optional"
  };
  Expr::FunctionCall {
    name: head.to_string(),
    args: vec![left.clone(), right.clone()].into(),
  }
}

/// Build the node for a bare `:` that reached the operator level — the left
/// side is a complete pattern rather than a bare name.
///
/// Wolfram's `:` binds *tightly* on its left: it takes only the term written
/// directly in front of it, even where that term is an operand of a
/// tighter-binding operator. `_Symbol | _Integer : 2` is
/// `Alternatives[_Symbol, Optional[_Integer, 2]]`, not an `Optional` around
/// the whole alternation, and `_?NumericQ : 2` is
/// `PatternTest[_, Pattern[NumericQ, 2]]` — the colon belongs to the test
/// operand of `?`. Woxi's precedence climbing has already folded those
/// operators, so reach back into the node and attach the default where
/// Wolfram would have put it.
///
/// The chained form `x : pat : v` is the one exception Wolfram makes: there
/// the trailing colon closes the whole `Pattern`, giving
/// `Optional[Pattern[x, pat], v]`. `Term` consumes the `x : pat` half before
/// this operator runs, so that case arrives here as a plain `Pattern` left
/// side and falls through to the default branch.
fn build_colon(left: &Expr, right: &Expr) -> Expr {
  match left {
    // The default attaches to the last alternative only.
    Expr::BinaryOp {
      op: BinaryOperator::Alternatives,
      left: l,
      right: r,
    } => Expr::BinaryOp {
      op: BinaryOperator::Alternatives,
      left: l.clone(),
      right: Box::new(build_colon(r, right)),
    },
    // `_?f : v` / `x_?f : v` — the colon binds inside the `?` test.
    Expr::PatternTest {
      name,
      head,
      blank_type,
      test,
    } => Expr::PatternTest {
      name: name.clone(),
      head: head.clone(),
      blank_type: *blank_type,
      test: Box::new(colon_node(test, right)),
    },
    Expr::FunctionCall { name, args }
      if name == "PatternTest" && args.len() == 2 =>
    {
      Expr::FunctionCall {
        name: "PatternTest".to_string(),
        args: vec![args[0].clone(), colon_node(&args[1], right)].into(),
      }
    }
    _ => colon_node(left, right),
  }
}

/// Create a binary operation from two expressions and an operator string
/// Build a flat (associative) binary operator, flattening chains so that
/// `a op b op c` collapses to `head[a, b, c]`.
fn build_flat_op(head: &str, left: &Expr, right: &Expr) -> Expr {
  let mut parts = Vec::new();
  for side in [left, right] {
    match side {
      Expr::FunctionCall { name, args } if name == head => {
        parts.extend(args.iter().cloned());
      }
      other => parts.push(other.clone()),
    }
  }
  Expr::FunctionCall {
    name: head.to_string(),
    args: parts.into(),
  }
}

fn make_binary_op(left: &Expr, op_str: &str, right: &Expr) -> Expr {
  match op_str {
    // A two-operand Span. Chains (`a ;; b ;; c`) are collected into a single
    // n-ary `Span` by `build_expr_with_precedence` before they get here.
    ";;" => build_span(vec![left.clone(), right.clone()]),
    // An omitted `;;` operand that an operator binding tighter than `;;`
    // claimed first — `a + ;; 3`, where `+` (310) takes the operand `;;`
    // (305) left out. Only `build_span` can fill such a gap in properly, so
    // read the omission as the `1` a written-out `Span[1, …]` would have.
    _ if is_span_omitted(left) || is_span_omitted(right) => make_binary_op(
      &span_omitted_fallback(left),
      op_str,
      &span_omitted_fallback(right),
    ),
    "=." => {
      // Unset (postfix): f[x] =. → Unset[f[x]], right operand is a dummy
      Expr::FunctionCall {
        name: "Unset".to_string(),
        args: vec![left.clone()].into(),
      }
    }
    "+" => Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(left.clone()),
      right: Box::new(right.clone()),
    },
    "-" => Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left: Box::new(left.clone()),
      right: Box::new(right.clone()),
    },
    // A leading minus is parsed with a synthetic `0` on the left so the
    // precedence climb can place it between Times and Power; the node itself
    // is the unary one, so that a held `-x` prints as `-x` and not `0 - x`.
    "NEGATE" => Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(right.clone()),
    },
    "*" => Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(left.clone()),
      right: Box::new(right.clone()),
    },
    // `a × b` and `a ÷ b` are `Times`/`Divide` written with their named
    // characters; a Demonstration's typeset formula uses them where it
    // would otherwise show a bare space.
    "\\[Times]" | "\u{00D7}" => Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(left.clone()),
      right: Box::new(right.clone()),
    },
    "\\[Divide]" | "\u{00F7}" => Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(left.clone()),
      right: Box::new(right.clone()),
    },
    "/" => Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(left.clone()),
      right: Box::new(right.clone()),
    },
    "^" => Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(left.clone()),
      right: Box::new(right.clone()),
    },
    "^_NEG" => Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(left.clone()),
      right: Box::new(Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand: Box::new(right.clone()),
      }),
    },
    "&&" => Expr::BinaryOp {
      op: BinaryOperator::And,
      left: Box::new(left.clone()),
      right: Box::new(right.clone()),
    },
    "||" => Expr::BinaryOp {
      op: BinaryOperator::Or,
      left: Box::new(left.clone()),
      right: Box::new(right.clone()),
    },
    "\\[And]" | "\u{2227}" => Expr::BinaryOp {
      op: BinaryOperator::And,
      left: Box::new(left.clone()),
      right: Box::new(right.clone()),
    },
    "\\[Or]" | "\u{2228}" => Expr::BinaryOp {
      op: BinaryOperator::Or,
      left: Box::new(left.clone()),
      right: Box::new(right.clone()),
    },
    // Flat named logical operators: chains collapse into one call
    // (a \[Xor] b \[Xor] c -> Xor[a, b, c]), matching wolframscript.
    "\\[Xor]" | "\u{22BB}" => flat_logical_call("Xor", left, right),
    "\\[Nand]" | "\u{22BC}" => flat_logical_call("Nand", left, right),
    "\\[Nor]" | "\u{22BD}" => flat_logical_call("Nor", left, right),
    "\\[Equivalent]" | "\u{29E6}" => {
      flat_logical_call("Equivalent", left, right)
    }
    "\\[Implies]" | "\u{F523}" => Expr::FunctionCall {
      name: "Implies".to_string(),
      args: vec![left.clone(), right.clone()].into(),
    },
    "<>" => Expr::BinaryOp {
      op: BinaryOperator::StringJoin,
      left: Box::new(left.clone()),
      right: Box::new(right.clone()),
    },
    ":" => build_colon(left, right),
    "|" => {
      // `:` (Pattern) binds looser than `|` (Alternatives) in Wolfram, so
      // `x : a | b` is `Pattern[x, Alternatives[a, b]]`. Woxi's parser
      // already consumed `x : a` into `Pattern[x, a]` before the `|` op,
      // so reach back into that Pattern's body and absorb the new
      // alternative there.
      if let Expr::FunctionCall { name, args } = left
        && name == "Pattern"
        && args.len() == 2
      {
        let inner_alts = match &args[1] {
          Expr::BinaryOp {
            op: BinaryOperator::Alternatives,
            left: l,
            right: r,
          } => Expr::BinaryOp {
            op: BinaryOperator::Alternatives,
            left: l.clone(),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Alternatives,
              left: r.clone(),
              right: Box::new(right.clone()),
            }),
          },
          body => Expr::BinaryOp {
            op: BinaryOperator::Alternatives,
            left: Box::new(body.clone()),
            right: Box::new(right.clone()),
          },
        };
        return Expr::FunctionCall {
          name: "Pattern".to_string(),
          args: vec![args[0].clone(), inner_alts].into(),
        };
      }
      Expr::BinaryOp {
        op: BinaryOperator::Alternatives,
        left: Box::new(left.clone()),
        right: Box::new(right.clone()),
      }
    }
    "\\[Element]" | "\u{2208}" => Expr::FunctionCall {
      name: "Element".to_string(),
      args: vec![left.clone(), right.clone()].into(),
    },
    "\\[NotElement]" | "\u{2209}" => Expr::FunctionCall {
      name: "NotElement".to_string(),
      args: vec![left.clone(), right.clone()].into(),
    },
    "\\[ReverseElement]" | "\u{220B}" => Expr::FunctionCall {
      name: "ReverseElement".to_string(),
      args: vec![left.clone(), right.clone()].into(),
    },
    "\\[DirectedEdge]" | "\u{F3D5}" => Expr::FunctionCall {
      name: "DirectedEdge".to_string(),
      args: vec![left.clone(), right.clone()].into(),
    },
    "\\[UndirectedEdge]" | "\u{F3D4}" => Expr::FunctionCall {
      name: "UndirectedEdge".to_string(),
      args: vec![left.clone(), right.clone()].into(),
    },
    "<->" => Expr::FunctionCall {
      name: "TwoWayRule".to_string(),
      args: vec![left.clone(), right.clone()].into(),
    },
    "\\[Distributed]" | "\u{F3D2}" => Expr::FunctionCall {
      name: "Distributed".to_string(),
      args: vec![left.clone(), right.clone()].into(),
    },
    "\\[Conditioned]" | "\u{F3D3}" => Expr::FunctionCall {
      name: "Conditioned".to_string(),
      args: vec![left.clone(), right.clone()].into(),
    },
    "\\[Function]" | "\u{F4A1}" | "|->" => Expr::FunctionCall {
      name: "Function".to_string(),
      args: vec![left.clone(), right.clone()].into(),
    },
    // Flat symbolic ring operators: a ⊕ b ⊕ c → CirclePlus[a, b, c], etc.
    "\\[CirclePlus]" | "\u{2295}" => build_flat_op("CirclePlus", left, right),
    "\\[CircleTimes]" | "\u{2297}" => build_flat_op("CircleTimes", left, right),
    "\\[CenterDot]" | "\u{00B7}" => build_flat_op("CenterDot", left, right),
    // CircleMinus is binary (not flat): a ⊖ b ⊖ c -> CircleMinus[(a ⊖ b), c].
    "\\[CircleMinus]" | "\u{2296}" => Expr::FunctionCall {
      name: "CircleMinus".to_string(),
      args: vec![left.clone(), right.clone()].into(),
    },
    "\\[Star]" | "\u{22C6}" => build_flat_op("Star", left, right),
    "\\[Diamond]" | "\u{22C4}" => build_flat_op("Diamond", left, right),
    "\\[Backslash]" | "\u{2216}" => build_flat_op("Backslash", left, right),
    "\\[CircleDot]" | "\u{2299}" => build_flat_op("CircleDot", left, right),
    "\\[SmallCircle]" | "\u{2218}" => build_flat_op("SmallCircle", left, right),
    // Wedge and Vee are flat: a ⋀ b ⋀ c -> Wedge[a, b, c].
    "\\[Wedge]" | "\u{22C0}" => build_flat_op("Wedge", left, right),
    "\\[Vee]" | "\u{22C1}" => build_flat_op("Vee", left, right),
    "\\[Cross]" | "\u{F4A0}" | "\u{F3C4}" | "\u{2A2F}" => {
      // Cross is Flat/associative — flatten chains: a ⨯ b ⨯ c → Cross[a, b, c].
      let mut parts = Vec::new();
      match left {
        Expr::FunctionCall { name, args } if name == "Cross" => {
          parts.extend(args.clone());
        }
        _ => parts.push(left.clone()),
      }
      match right {
        Expr::FunctionCall { name, args } if name == "Cross" => {
          parts.extend(args.clone());
        }
        _ => parts.push(right.clone()),
      }
      Expr::FunctionCall {
        name: "Cross".to_string(),
        args: parts.into(),
      }
    }
    "\\[TensorProduct]" | "\u{F3DA}" => {
      // TensorProduct is Flat: flatten chains a ⊗ b ⊗ c → TensorProduct[a, b, c].
      let mut parts = Vec::new();
      match left {
        Expr::FunctionCall { name, args } if name == "TensorProduct" => {
          parts.extend(args.clone());
        }
        _ => parts.push(left.clone()),
      }
      match right {
        Expr::FunctionCall { name, args } if name == "TensorProduct" => {
          parts.extend(args.clone());
        }
        _ => parts.push(right.clone()),
      }
      Expr::FunctionCall {
        name: "TensorProduct".to_string(),
        args: parts.into(),
      }
    }
    "\\[Cap]" | "\u{2322}" => {
      // Cap is Flat/associative — flatten chains: a ⌢ b ⌢ c → Cap[a, b, c].
      let mut parts = Vec::new();
      match left {
        Expr::FunctionCall { name, args } if name == "Cap" => {
          parts.extend(args.clone());
        }
        _ => parts.push(left.clone()),
      }
      match right {
        Expr::FunctionCall { name, args } if name == "Cap" => {
          parts.extend(args.clone());
        }
        _ => parts.push(right.clone()),
      }
      Expr::FunctionCall {
        name: "Cap".to_string(),
        args: parts.into(),
      }
    }
    "\\[Cup]" | "\u{2323}" => {
      // Cup is Flat/associative — flatten chains: a ⌣ b ⌣ c → Cup[a, b, c].
      let mut parts = Vec::new();
      match left {
        Expr::FunctionCall { name, args } if name == "Cup" => {
          parts.extend(args.clone());
        }
        _ => parts.push(left.clone()),
      }
      match right {
        Expr::FunctionCall { name, args } if name == "Cup" => {
          parts.extend(args.clone());
        }
        _ => parts.push(right.clone()),
      }
      Expr::FunctionCall {
        name: "Cup".to_string(),
        args: parts.into(),
      }
    }
    "\\[RightTee]" | "\u{22A2}" => Expr::FunctionCall {
      name: "RightTee".to_string(),
      args: vec![left.clone(), right.clone()].into(),
    },
    "\\[DoubleRightTee]" | "\u{22A8}" => Expr::FunctionCall {
      name: "DoubleRightTee".to_string(),
      args: vec![left.clone(), right.clone()].into(),
    },
    "\\[LeftTee]" | "\u{22A3}" => Expr::FunctionCall {
      name: "LeftTee".to_string(),
      args: vec![left.clone(), right.clone()].into(),
    },
    "\\[DoubleLeftTee]" | "\u{2AE4}" => Expr::FunctionCall {
      name: "DoubleLeftTee".to_string(),
      args: vec![left.clone(), right.clone()].into(),
    },
    "~~" => {
      // Flatten nested StringExpression (it's Flat/associative)
      let mut parts = Vec::new();
      match left {
        Expr::FunctionCall { name, args } if name == "StringExpression" => {
          parts.extend(args.clone());
        }
        _ => parts.push(left.clone()),
      }
      match right {
        Expr::FunctionCall { name, args } if name == "StringExpression" => {
          parts.extend(args.clone());
        }
        _ => parts.push(right.clone()),
      }
      Expr::FunctionCall {
        name: "StringExpression".to_string(),
        args: parts.into(),
      }
    }
    "/@" => Expr::Map {
      func: Box::new(left.clone()),
      list: Box::new(right.clone()),
    },
    "@@@" => Expr::MapApply {
      func: Box::new(left.clone()),
      list: Box::new(right.clone()),
    },
    "@@" => Expr::Apply {
      func: Box::new(left.clone()),
      list: Box::new(right.clone()),
    },
    "@*" => {
      // Flatten nested Composition: (f @* g) @* h -> Composition[f, g, h]
      let mut funcs = Vec::new();
      match left {
        Expr::FunctionCall { name, args } if name == "Composition" => {
          funcs.extend(args.clone());
        }
        _ => funcs.push(left.clone()),
      }
      match right {
        Expr::FunctionCall { name, args } if name == "Composition" => {
          funcs.extend(args.clone());
        }
        _ => funcs.push(right.clone()),
      }
      Expr::FunctionCall {
        name: "Composition".to_string(),
        args: funcs.into(),
      }
    }
    "/*" => {
      // Flatten nested RightComposition: (f /* g) /* h -> RightComposition[f, g, h]
      let mut funcs = Vec::new();
      match left {
        Expr::FunctionCall { name, args } if name == "RightComposition" => {
          funcs.extend(args.clone());
        }
        _ => funcs.push(left.clone()),
      }
      match right {
        Expr::FunctionCall { name, args } if name == "RightComposition" => {
          funcs.extend(args.clone());
        }
        _ => funcs.push(right.clone()),
      }
      Expr::FunctionCall {
        name: "RightComposition".to_string(),
        args: funcs.into(),
      }
    }
    // `f @ x` == `f[x]`. Use the FunctionCall form when the LHS is a plain
    // identifier (matching the Rule::PrefixApplySimple branch) so downstream
    // code that pattern-matches on FunctionCall (e.g. DownValue assignment,
    // `del2@banana = "phone"`) sees the canonical shape.
    "@" => match left {
      Expr::Identifier(name) => Expr::FunctionCall {
        name: name.clone(),
        args: vec![right.clone()].into(),
      },
      Expr::FunctionCall { .. } => Expr::CurriedCall {
        func: Box::new(left.clone()),
        args: vec![right.clone()],
      },
      _ => Expr::PrefixApply {
        func: Box::new(left.clone()),
        arg: Box::new(right.clone()),
      },
    },
    "." => Expr::FunctionCall {
      name: "Dot".to_string(),
      args: vec![left.clone(), right.clone()].into(),
    },
    "->" | "\u{2192}" | "\\[Rule]" | "\u{F522}" => Expr::Rule {
      pattern: Box::new(left.clone()),
      replacement: Box::new(right.clone()),
    },
    ":>" | "\\[RuleDelayed]" | "\u{F51F}" => Expr::RuleDelayed {
      pattern: Box::new(left.clone()),
      replacement: Box::new(right.clone()),
    },
    ">>" => Expr::FunctionCall {
      name: "Put".to_string(),
      args: vec![left.clone(), right.clone()].into(),
    },
    ">>>" => Expr::FunctionCall {
      name: "PutAppend".to_string(),
      args: vec![left.clone(), right.clone()].into(),
    },
    "=" => Expr::FunctionCall {
      name: "Set".to_string(),
      args: vec![left.clone(), right.clone()].into(),
    },
    // `AddTo` & co. reach the operator chain only when their target is a
    // function call — a symbol or part target is taken by the `AddTo`,
    // `SubtractFrom`, `TimesBy` and `DivideBy` terms in the grammar.
    "+=" => Expr::FunctionCall {
      name: "AddTo".to_string(),
      args: vec![left.clone(), right.clone()].into(),
    },
    "-=" => Expr::FunctionCall {
      name: "SubtractFrom".to_string(),
      args: vec![left.clone(), right.clone()].into(),
    },
    "*=" => Expr::FunctionCall {
      name: "TimesBy".to_string(),
      args: vec![left.clone(), right.clone()].into(),
    },
    "/=" => Expr::FunctionCall {
      name: "DivideBy".to_string(),
      args: vec![left.clone(), right.clone()].into(),
    },
    "^=" => Expr::FunctionCall {
      name: "UpSet".to_string(),
      args: vec![left.clone(), right.clone()].into(),
    },
    "^:=" => Expr::FunctionCall {
      name: "UpSetDelayed".to_string(),
      args: vec![left.clone(), right.clone()].into(),
    },
    ":=" => Expr::FunctionCall {
      name: "SetDelayed".to_string(),
      args: vec![left.clone(), right.clone()].into(),
    },
    "::" => {
      // MessageName[sym, "tag"]. The right-hand side is treated as a string tag:
      // identifiers become strings, integers become their decimal string form.
      let tag = match right {
        Expr::Identifier(name) => Expr::String(name.clone()),
        Expr::Integer(n) => Expr::String(n.to_string()),
        other => other.clone(),
      };
      Expr::FunctionCall {
        name: "MessageName".to_string(),
        args: vec![left.clone(), tag].into(),
      }
    }
    "/;" => Expr::FunctionCall {
      name: "Condition".to_string(),
      args: vec![left.clone(), right.clone()].into(),
    },
    "/:" => {
      // TagSet or TagSetDelayed or TagUnset:
      //   tag /: lhs = rhs   -> TagSet[tag, lhs, rhs]
      //   tag /: lhs := rhs  -> TagSetDelayed[tag, lhs, rhs]
      //   tag /: lhs =.      -> TagUnset[tag, lhs]
      // The right side has already been parsed with the = or := operator,
      // producing Set[lhs, rhs], SetDelayed[lhs, rhs], or Unset[lhs].
      match right {
        Expr::FunctionCall { name, args }
          if (name == "SetDelayed" || name == "Set") && args.len() == 2 =>
        {
          let tag_name = if name == "SetDelayed" {
            "TagSetDelayed"
          } else {
            "TagSet"
          };
          Expr::FunctionCall {
            name: tag_name.to_string(),
            args: vec![left.clone(), args[0].clone(), args[1].clone()].into(),
          }
        }
        Expr::FunctionCall { name, args }
          if name == "Unset" && args.len() == 1 =>
        {
          Expr::FunctionCall {
            name: "TagUnset".to_string(),
            args: vec![left.clone(), args[0].clone()].into(),
          }
        }
        _ => {
          // Fallback: wrap as Condition (x /: y without = or :=)
          Expr::FunctionCall {
            name: "Condition".to_string(),
            args: vec![left.clone(), right.clone()].into(),
          }
        }
      }
    }
    "==" | "\u{2A75}" | "\\[Equal]" | "\u{F431}" | "!=" | "\u{2260}"
    | "\\[NotEqual]" | "<" | "<=" | "\u{2264}" | "\u{2A7D}"
    | "\\[LessEqual]" | ">" | ">=" | "\u{2265}" | "\u{2A7E}"
    | "\\[GreaterEqual]" | "===" | "=!=" => {
      let comp_op = match op_str {
        "==" | "\u{2A75}" | "\\[Equal]" | "\u{F431}" => ComparisonOp::Equal,
        "!=" | "\u{2260}" | "\\[NotEqual]" => ComparisonOp::NotEqual,
        "<" => ComparisonOp::Less,
        "<=" | "\u{2264}" | "\u{2A7D}" | "\\[LessEqual]" => {
          ComparisonOp::LessEqual
        }
        ">" => ComparisonOp::Greater,
        ">=" | "\u{2265}" | "\u{2A7E}" | "\\[GreaterEqual]" => {
          ComparisonOp::GreaterEqual
        }
        "===" => ComparisonOp::SameQ,
        "=!=" => ComparisonOp::UnsameQ,
        _ => ComparisonOp::Equal,
      };
      // If the left side is already a Comparison, extend the chain
      {
        let mut cloned = left.clone();
        if let Expr::Comparison {
          operands: ref mut ops,
          operators: ref mut comp_ops,
        } = cloned
        {
          ops.push(right.clone());
          comp_ops.push(comp_op);
          cloned
        } else {
          Expr::Comparison {
            operands: vec![left.clone(), right.clone()],
            operators: vec![comp_op],
          }
        }
      }
    }
    s if s.starts_with('~') && s.ends_with('~') && s.len() > 2 => {
      // Tilde infix: a ~f~ b → f[a, b], a ~f[x]~ b → f[x][a, b]
      let func_name = &s[1..s.len() - 1];
      // Check if it's a function call like f[x] by looking for brackets
      if let Some(bracket_idx) = func_name.find('[') {
        // Parse as CurriedCall: f[x][a, b]
        let head = &func_name[..bracket_idx];
        let args_str = &func_name[bracket_idx + 1..func_name.len() - 1];
        // Build f[x] first, then apply [a, b]
        let func_call = if args_str.is_empty() {
          Expr::FunctionCall {
            name: head.to_string(),
            args: vec![].into(),
          }
        } else {
          // For simple cases, just re-parse through the evaluator
          Expr::FunctionCall {
            name: head.to_string(),
            args: vec![Expr::Identifier(args_str.to_string())].into(),
          }
        };
        Expr::CurriedCall {
          func: Box::new(func_call),
          args: vec![left.clone(), right.clone()],
        }
      } else {
        Expr::FunctionCall {
          name: func_name.to_string(),
          args: vec![left.clone(), right.clone()].into(),
        }
      }
    }
    _ => Expr::Raw(format!(
      "{} {} {}",
      expr_to_string(left),
      op_str,
      expr_to_string(right)
    )),
  }
}

/// Parse the body of an anonymous function using the pest parser
fn parse_anonymous_body(s: &str) -> Expr {
  let s = s.trim();

  if s.is_empty() {
    return Expr::Slot(1);
  }

  // Check for simple slot (fast path)
  if s == "#" {
    return Expr::Slot(1);
  }
  if s.starts_with('#')
    && s.len() > 1
    && s[1..].chars().all(|c| c.is_ascii_digit())
  {
    let num: usize = s[1..].parse().unwrap_or(1);
    return Expr::Slot(num);
  }

  // Use the pest parser to properly parse complex expressions
  match crate::parse(s) {
    Ok(pairs) => {
      for pair in pairs {
        for node in pair.into_inner() {
          if matches!(node.as_rule(), Rule::Expression) {
            return pair_to_expr(node);
          }
        }
      }
      // Fallback if parsing succeeded but no expression found
      Expr::Raw(s.to_string())
    }
    Err(_) => {
      // If parsing fails, fall back to Raw
      Expr::Raw(s.to_string())
    }
  }
}

/// Format a real number for output (matches Wolfram Language format)
pub fn format_real(f: f64) -> String {
  if f.is_infinite() {
    if f > 0.0 {
      return "Infinity".to_string();
    }
    return "-Infinity".to_string();
  }
  if f.is_nan() {
    return "Indeterminate".to_string();
  }
  let abs = f.abs();
  if abs == 0.0 {
    return "0.".to_string();
  }
  // Wolfram uses *^ scientific notation for |f| >= 1e6 or |f| < 1e-5
  if !(1e-5..1e6).contains(&abs) {
    format_real_scientific(f)
  } else if f.fract() == 0.0 {
    // Whole number in normal range - format with trailing dot
    format!("{}.", f as i64)
  } else {
    // IEEE 754 doubles need up to 17 significant digits for faithful
    // round-trip representation.  Wolfram displays all 17 when needed
    // (e.g. 0.1 + 0.2 → 0.30000000000000004).
    let s = format!("{f}");
    cap_significant_digits(&s, f, 17)
  }
}

/// Cap a formatted decimal number string to at most `max_sig` significant digits.
/// Uses `format!("{:.prec$e}", f)` for proper rounding when truncation is needed,
/// then reconstructs the decimal notation.
fn cap_significant_digits(s: &str, f: f64, max_sig: usize) -> String {
  // Count significant digits in the string
  let negative = s.starts_with('-');
  let abs_s = if negative { &s[1..] } else { s };
  let mut sig = 0usize;
  let mut started = false;
  for ch in abs_s.chars() {
    if ch == '.' {
      continue;
    }
    if !ch.is_ascii_digit() {
      break;
    }
    if ch != '0' {
      started = true;
    }
    if started {
      sig += 1;
    }
  }
  if sig <= max_sig {
    return s.to_string();
  }
  // Reformat with proper rounding via scientific notation
  let sci = format!("{:.prec$e}", f.abs(), prec = max_sig - 1);
  let (mantissa, exp_str) = sci.split_once('e').unwrap();
  let exp: i32 = exp_str.parse().unwrap();
  let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
  let digits = digits.trim_end_matches('0');
  if digits.is_empty() {
    return "0.".to_string();
  }
  let dot_offset = exp + 1;
  let mut result = String::new();
  if negative {
    result.push('-');
  }
  if dot_offset <= 0 {
    result.push_str("0.");
    for _ in 0..(-dot_offset) {
      result.push('0');
    }
    result.push_str(digits);
  } else {
    let dp = dot_offset as usize;
    if dp >= digits.len() {
      result.push_str(digits);
      for _ in 0..(dp - digits.len()) {
        result.push('0');
      }
      result.push('.');
    } else {
      result.push_str(&digits[..dp]);
      result.push('.');
      result.push_str(&digits[dp..]);
    }
  }
  result
}

/// Format a real number using Wolfram's *^ scientific notation.
/// E.g. 2.733467611516948*^33 or -1.5*^-6
///
/// Uses the shortest round-trip representation (matching Rust's `format!("{}", f)`)
/// to determine how many significant digits are needed, then converts to
/// Wolfram's *^ notation.
fn format_real_scientific(f: f64) -> String {
  let negative = f < 0.0;
  let abs = f.abs();
  let sign = if negative { "-" } else { "" };

  // Count significant digits from the shortest scientific-notation
  // round-trip representation. Using `{}` would emit the full integer
  // expansion for very large values (e.g. f64::MAX has 309 digits), which
  // then inflates the precision count and produces far more digits than
  // needed. Rust's `{:e}` already uses the shortest unambiguous mantissa.
  let sci_short = format!("{abs:e}");
  let mantissa_short = sci_short
    .split_once('e')
    .map_or(sci_short.as_str(), |(m, _)| m);
  let mut sig = 0usize;
  let mut started = false;
  for ch in mantissa_short.chars() {
    if ch == '.' || ch == '-' {
      continue;
    }
    if !ch.is_ascii_digit() {
      break;
    }
    if ch != '0' {
      started = true;
    }
    if started {
      sig += 1;
    }
  }
  let prec = sig.saturating_sub(1);

  // Format with exactly the right number of significant digits
  let sci = format!("{abs:.prec$e}");
  let (mantissa, exp_str) = sci.split_once('e').unwrap();
  let exp: i32 = exp_str.parse().unwrap();
  // Trim trailing zeros from mantissa, keeping the dot
  let mantissa = mantissa.trim_end_matches('0');
  // Ensure mantissa contains '.' (Wolfram uses "1.*^6" not "1*^6")
  let mantissa = if mantissa.contains('.') {
    mantissa.to_string()
  } else {
    format!("{mantissa}.")
  };
  format!("{sign}{mantissa}*^{exp}")
}

/// Format a BigFloat (arbitrary-precision real) for display.
/// Uses Wolfram's backtick notation: `digits`precision.`
/// Uses scientific notation (*^) for very large or very small numbers,
/// matching Wolfram's thresholds (|value| >= 1e6 or < 1e-5).
/// Format a precision value for backtick notation.
/// Integer precisions display as "2.", non-integer as full float "2.041392685158225".
fn format_precision(prec: f64) -> String {
  if prec == prec.floor() {
    format!("{}.", prec as i64)
  } else {
    // Format with full precision, trimming trailing zeros but keeping at least one decimal
    let s = format!("{prec}");
    s
  }
}

pub fn format_bigfloat(digits: &str, prec: f64) -> String {
  // Special case: BigFloat("0", α) is the accuracy form `0``α.` (zero with
  // accuracy α). Wolfram displays the integer 0 followed by a double
  // backtick and the accuracy.
  if digits == "0" {
    return format!("0``{}", format_precision(prec));
  }
  let (is_negative, abs_digits) = if let Some(rest) = digits.strip_prefix('-') {
    (true, rest)
  } else {
    (false, digits)
  };
  let prefix = if is_negative { "-" } else { "" };

  // Find the decimal point position
  let dot_pos = abs_digits.find('.');
  let int_part = if let Some(dp) = dot_pos {
    &abs_digits[..dp]
  } else {
    abs_digits
  };
  let frac_part = if let Some(dp) = dot_pos {
    if dp + 1 < abs_digits.len() {
      &abs_digits[dp + 1..]
    } else {
      ""
    }
  } else {
    ""
  };

  // Check if integer part has 6+ digits (value >= 1e6) → scientific notation
  // But skip if integer part is all zeros (like "0" or "00")
  let int_nonzero_len = int_part.trim_start_matches('0').len();
  if int_part.len() >= 6 && int_nonzero_len > 0 {
    // Collect all significant digits (integer + fractional)
    let all_digits: String =
      int_part.chars().chain(frac_part.chars()).collect();
    let sig_digits = all_digits.trim_end_matches('0');
    if sig_digits.is_empty() {
      return format!("{}0.`{}", prefix, format_precision(prec));
    }
    let exp = int_part.len() as i64 - 1;
    let mantissa = if sig_digits.len() > 1 {
      format!("{}{}.{}", prefix, &sig_digits[..1], &sig_digits[1..])
    } else {
      format!("{}{}.", prefix, &sig_digits[..1])
    };
    return format!("{}`{}*^{}", mantissa, format_precision(prec), exp);
  }

  // Check if number is very small: "0.00000..." with 5+ leading zeros
  if (int_part == "0" || int_part.is_empty()) && !frac_part.is_empty() {
    let leading_zeros = frac_part.chars().take_while(|&c| c == '0').count();
    if leading_zeros >= 5 {
      let sig_part = &frac_part[leading_zeros..];
      let sig_digits = sig_part.trim_end_matches('0');
      if sig_digits.is_empty() {
        return format!("{}0.`{}", prefix, format_precision(prec));
      }
      let exp = -(leading_zeros as i64 + 1);
      let mantissa = if sig_digits.len() > 1 {
        format!("{}{}.{}", prefix, &sig_digits[..1], &sig_digits[1..])
      } else {
        format!("{}{}.", prefix, &sig_digits[..1])
      };
      return format!("{}`{}*^{}", mantissa, format_precision(prec), exp);
    }
  }

  // Normal format (no scientific notation needed). Trim trailing zeros
  // from the fractional part — Wolfram normalizes `10.00`2` to `10.`2.`,
  // but keep the decimal point so `n.` stays distinct from integer `n`.
  let normalized = if let Some(dp) = abs_digits.find('.') {
    let int_p = &abs_digits[..dp];
    let frac_p = &abs_digits[dp + 1..];
    let trimmed_frac = frac_p.trim_end_matches('0');
    format!("{prefix}{int_p}.{trimmed_frac}")
  } else {
    digits.to_string()
  };
  format!("{}`{}", normalized, format_precision(prec))
}

/// If expr is Times[negative_coeff, rest...], return Some(Times[abs(coeff), rest...]).
/// Works for both BinaryOp{Times} and FunctionCall{Times} forms.
fn negate_leading_negative_in_times(expr: &Expr) -> Option<Expr> {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => match left.as_ref() {
      Expr::Integer(n) if *n < 0 => {
        if *n == -1 {
          Some(right.as_ref().clone())
        } else {
          Some(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(-n)),
            right: right.clone(),
          })
        }
      }
      Expr::BigInteger(n) if n.sign() == Sign::Minus => Some(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::BigInteger(-n)),
        right: right.clone(),
      }),
      Expr::FunctionCall { name, args }
        if name == "Rational"
          && args.len() == 2
          && matches!(&args[0], Expr::Integer(n) if *n < 0) =>
      {
        let n = if let Expr::Integer(n) = &args[0] {
          *n
        } else {
          return None;
        };
        let pos_rat = Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(-n), args[1].clone()].into(),
        };
        if -n == 1 {
          // Rational[-1, d] * x → x/d
          Some(Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: right.clone(),
            right: Box::new(args[1].clone()),
          })
        } else {
          Some(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(pos_rat),
            right: right.clone(),
          })
        }
      }
      _ => None,
    },
    Expr::FunctionCall { name, args } if name == "Times" && args.len() >= 2 => {
      match &args[0] {
        Expr::Integer(n) if *n < 0 => {
          if *n == -1 {
            let rest = args[1..].to_vec();
            Some(if rest.len() == 1 {
              rest[0].clone()
            } else {
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: rest.into(),
              }
            })
          } else {
            let mut new_args = vec![Expr::Integer(-n)];
            new_args.extend_from_slice(&args[1..]);
            Some(if new_args.len() == 1 {
              new_args[0].clone()
            } else {
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: new_args.into(),
              }
            })
          }
        }
        Expr::BigInteger(n) if n.sign() == Sign::Minus => {
          let mut new_args = vec![Expr::BigInteger(-n)];
          new_args.extend_from_slice(&args[1..]);
          Some(if new_args.len() == 1 {
            new_args[0].clone()
          } else {
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: new_args.into(),
            }
          })
        }
        Expr::FunctionCall { name: rn, args: ra }
          if rn == "Rational"
            && ra.len() == 2
            && matches!(&ra[0], Expr::Integer(n) if *n < 0) =>
        {
          let n = if let Expr::Integer(n) = &ra[0] {
            *n
          } else {
            return None;
          };
          let pos_rat = Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(-n), ra[1].clone()].into(),
          };
          let mut new_args = vec![pos_rat];
          new_args.extend_from_slice(&args[1..]);
          Some(if new_args.len() == 1 {
            new_args[0].clone()
          } else {
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: new_args.into(),
            }
          })
        }
        _ => None,
      }
    }
    _ => None,
  }
}

/// Format a Quantity unit expression for InputForm (strings are quoted).
/// Wolfram InputForm: "Meters"/"Seconds"^2
fn quantity_unit_to_input_form(unit: &Expr) -> String {
  // A unit is an ordinary expression, so the InputForm renderer writes it:
  // string names stay quoted, a negative exponent becomes a quotient
  // (`"Meters"/"Seconds"`) and a compound one is parenthesised.
  expr_to_input_form(unit)
}

/// Extract (base, negative_exponent) from a Power expression with a negative integer exponent.
/// Returns `Some((base, neg_exp))` for both FunctionCall and BinaryOp Power forms.
pub fn extract_neg_power_info(expr: &Expr) -> Option<(&Expr, i128)> {
  match expr {
    Expr::FunctionCall { name, args }
      if name == "Power"
        && args.len() == 2
        && matches!(&args[1], Expr::Integer(n) if *n < 0) =>
    {
      let Expr::Integer(n) = &args[1] else {
        unreachable!()
      };
      Some((&args[0], *n))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } if matches!(right.as_ref(), Expr::Integer(n) if *n < 0) => {
      let Expr::Integer(n) = right.as_ref() else {
        unreachable!()
      };
      Some((left.as_ref(), *n))
    }
    _ => None,
  }
}

/// True for atomic exponents that don't need parentheses inside a Quantity
/// unit display. Wolfram renders `Watts^0.24` (no parens) but parenthesizes
/// compound exponents like `Watts^(1+a)` and `Watts^(1/3)`. A Rational is
/// displayed as the infix expression `n/d` in InputForm, so it needs parens.
fn is_quantity_exp_atom(expr: &Expr) -> bool {
  matches!(
    expr,
    Expr::Integer(_)
      | Expr::BigInteger(_)
      | Expr::Real(_)
      | Expr::BigFloat(_, _)
      | Expr::Identifier(_)
      | Expr::Constant(_)
      | Expr::String(_)
  )
}

/// Format a Quantity unit expression without quoting.
/// Wolfram displays units unquoted: Meters, Miles/Hours, Meters/Seconds^2
fn quantity_unit_to_string(unit: &Expr) -> String {
  match unit {
    Expr::Identifier(s) | Expr::String(s) => s.clone(),
    // MixedUnit[{…}] carries the per-part units of a MixedMagnitude; each
    // one is a unit name and displays unquoted like any other.
    Expr::FunctionCall { name, args }
      if name == "MixedUnit"
        && args.len() == 1
        && matches!(&args[0], Expr::List(_)) =>
    {
      let Expr::List(parts) = &args[0] else {
        unreachable!()
      };
      let names: Vec<String> =
        parts.iter().map(quantity_unit_to_string).collect();
      format!("MixedUnit[{{{}}}]", names.join(", "))
    }
    // Sqrt (Power[x, 1/2]) → Sqrt[base_str]
    expr if crate::functions::is_sqrt(expr).is_some() => {
      let sqrt_arg = crate::functions::is_sqrt(expr).unwrap();
      format!("Sqrt[{}]", expr_to_string(sqrt_arg))
    }
    // Power must come before the general BinaryOp arm to avoid being shadowed
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      let exp_str = expr_to_string(right);
      let exp_fmt = if is_quantity_exp_atom(right) {
        exp_str
      } else {
        format!("({exp_str})")
      };
      format!("{}^{}", quantity_unit_to_string(left), exp_fmt)
    }
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      format!(
        "{}/{}",
        quantity_unit_to_string(left),
        quantity_unit_to_string(right)
      )
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      format!(
        "{}*{}",
        quantity_unit_to_string(left),
        quantity_unit_to_string(right)
      )
    }
    Expr::BinaryOp { .. } => expr_to_string(unit),
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      let exp_str = expr_to_string(&args[1]);
      let exp_fmt = if is_quantity_exp_atom(&args[1]) {
        exp_str
      } else {
        format!("({exp_str})")
      };
      format!("{}^{}", quantity_unit_to_string(&args[0]), exp_fmt)
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      // Check for fraction form: Times[..., Power[den, -1]]
      let mut numer_parts: Vec<String> = Vec::new();
      let mut denom_parts: Vec<String> = Vec::new();
      // A product base in the denominator (e.g. (m·s²) from
      // Power[Times[m, s^2], -1]) must be parenthesized, otherwise
      // a/(b*c) would mis-render as the differently-grouped a/b*c.
      let is_product = |e: &Expr| {
        matches!(e, Expr::FunctionCall { name, .. } if name == "Times")
          || matches!(
            e,
            Expr::BinaryOp {
              op: BinaryOperator::Times,
              ..
            }
          )
      };
      for a in args {
        if let Some((base, neg_exp)) = extract_neg_power_info(a) {
          let mut base_str = quantity_unit_to_string(base);
          if is_product(base) {
            base_str = format!("({base_str})");
          }
          if neg_exp == -1 {
            denom_parts.push(base_str);
          } else {
            denom_parts.push(format!("{}^{}", base_str, -neg_exp));
          }
        } else {
          numer_parts.push(quantity_unit_to_string(a));
        }
      }
      if denom_parts.is_empty() {
        numer_parts.join("*")
      } else {
        // A product numerator is parenthesized when a denominator follows:
        // (a*b)/c, matching Wolfram's display.
        let numer = if numer_parts.is_empty() {
          "1".to_string()
        } else if numer_parts.len() > 1 {
          format!("({})", numer_parts.join("*"))
        } else {
          numer_parts.join("*")
        };
        // Multiple denominator factors also need grouping: a/(b*c), not a/b*c.
        let denom = if denom_parts.len() > 1 {
          format!("({})", denom_parts.join("*"))
        } else {
          denom_parts.join("*")
        };
        format!("{numer}/{denom}")
      }
    }
    _ => expr_to_string(unit),
  }
}

/// Format a Quantity unit expression using abbreviated unit names for visual display.
/// E.g. `"Meters"/"Seconds"` → `m/s`, `"Meters"/"Seconds"^2` → `m/s^2`.
fn quantity_unit_to_abbrev(unit: &Expr) -> String {
  use crate::functions::quantity_ast::unit_to_abbreviation;
  match unit {
    Expr::Identifier(s) | Expr::String(s) => {
      // Try direct lookup, then plural form (since strings may not be normalized)
      if let Some(abbr) = unit_to_abbreviation(s) {
        abbr.to_string()
      } else {
        let plural = format!("{s}s");
        unit_to_abbreviation(&plural)
          .unwrap_or(s.as_str())
          .to_string()
      }
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      let exp_str = expr_to_string(right);
      let exp_fmt = if matches!(right.as_ref(), Expr::Integer(_)) {
        exp_str
      } else {
        format!("({exp_str})")
      };
      format!("{}^{}", quantity_unit_to_abbrev(left), exp_fmt)
    }
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      format!(
        "{}/{}",
        quantity_unit_to_abbrev(left),
        quantity_unit_to_abbrev(right)
      )
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      format!(
        "{}*{}",
        quantity_unit_to_abbrev(left),
        quantity_unit_to_abbrev(right)
      )
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      let exp_str = expr_to_string(&args[1]);
      let exp_fmt = if matches!(args[1], Expr::Integer(_)) {
        exp_str
      } else {
        format!("({exp_str})")
      };
      format!("{}^{}", quantity_unit_to_abbrev(&args[0]), exp_fmt)
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      // Check for fraction form: Times[..., Power[den, -1]]
      let mut numer_parts: Vec<String> = Vec::new();
      let mut denom_parts: Vec<String> = Vec::new();
      for a in args {
        if let Some((base, neg_exp)) = extract_neg_power_info(a) {
          let base_str = quantity_unit_to_abbrev(base);
          if neg_exp == -1 {
            denom_parts.push(base_str);
          } else {
            denom_parts.push(format!("{}^{}", base_str, -neg_exp));
          }
        } else {
          numer_parts.push(quantity_unit_to_abbrev(a));
        }
      }
      if denom_parts.is_empty() {
        numer_parts.join("*")
      } else {
        let numer = if numer_parts.is_empty() {
          "1".to_string()
        } else {
          numer_parts.join("*")
        };
        let denom = denom_parts.join("*");
        format!("{numer}/{denom}")
      }
    }
    _ => expr_to_string(unit),
  }
}

/// Format a Quantity expression for visual/notebook display.
/// `Quantity[12.345, "Meters"/"Seconds"]` → `"12.345 m/s"`.
pub fn quantity_to_visual_string(mag: &Expr, unit: &Expr) -> String {
  let mag_str = expr_to_output(mag);
  let unit_str = quantity_unit_to_abbrev(unit);
  let unit_str = singularize_unit_if_one(mag, &unit_str);
  format!("{mag_str} {unit_str}")
}

/// If the magnitude is exactly 1, singularize unit display names
/// that are full words (e.g. "days" → "day").
pub fn singularize_unit_if_one(mag: &Expr, unit_str: &str) -> String {
  let is_one = matches!(mag, Expr::Integer(1));
  if is_one && unit_str.ends_with('s') && unit_str.len() > 2 {
    // Only singularize word-like units (e.g. "days"), not abbreviations (e.g. "ms")
    if unit_str.chars().all(|c| c.is_ascii_lowercase()) {
      return unit_str[..unit_str.len() - 1].to_string();
    }
  }
  unit_str.to_string()
}

/// Check if an expression is Power[symbol, negative_integer].
/// Used to decide when to keep x^(-n) notation instead of fraction form.
/// Check if an expression is Power[base, negative_integer] where base is symbolic
/// (contains variables, not a purely numeric expression like Sqrt[2]).
/// Symbols and Plus/Times expressions containing identifiers are considered symbolic.
fn is_symbolic_neg_int_power(expr: &Expr) -> bool {
  // A negative exponent that is not a literal — `-x`, `-2 x`, `-y/2` — keeps
  // the power form whatever the base is: wolframscript writes `-2^(-x)` and
  // `-E^(-x)`, not `-(1/2^x)`.
  let symbolic_negative = |exponent: &Expr| -> bool {
    match exponent {
      Expr::FunctionCall { name, args }
        if name == "Times"
          && !args.is_empty()
          && (matches!(&args[0], Expr::Integer(n) if *n < 0)
            || matches!(&args[0], Expr::FunctionCall { name, args }
              if name == "Rational" && args.len() == 2
              && matches!(&args[0], Expr::Integer(n) if *n < 0))) =>
      {
        true
      }
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        ..
      } => matches!(left.as_ref(), Expr::Integer(n) if *n < 0),
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        ..
      } => true,
      _ => false,
    }
  };
  let (base, neg_exp) = match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      if symbolic_negative(right.as_ref()) {
        return true;
      }
      (
        left.as_ref(),
        matches!(right.as_ref(), Expr::Integer(n) if *n < 0),
      )
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      if symbolic_negative(&args[1]) {
        return true;
      }
      (&args[0], matches!(&args[1], Expr::Integer(n) if *n < 0))
    }
    _ => return false,
  };
  if !neg_exp {
    return false;
  }
  // Check if base contains any identifiers (symbolic, not purely numeric)
  fn has_identifier(e: &Expr) -> bool {
    match e {
      Expr::Identifier(_) => true,
      Expr::FunctionCall { args, .. } => args.iter().any(has_identifier),
      Expr::BinaryOp { left, right, .. } => {
        has_identifier(left) || has_identifier(right)
      }
      Expr::UnaryOp { operand, .. } => has_identifier(operand),
      _ => false,
    }
  }
  has_identifier(base)
}

/// Check if an expression is Power[base, negative_exponent] suitable for moving to denominator
/// in a Times expression. Handles both FunctionCall and BinaryOp representations.
fn is_denominator_factor(expr: &Expr) -> bool {
  // A Rational[n, d] with |d| > 1 contributes a denominator factor `d` and
  // triggers fraction formatting when n > 0: Wolfram uses `(n*X)/d` for
  // `|n| > 1` (e.g. `(-2*Pi)/3`) and `X/d` for `n == 1` (e.g. `Pi/4`). A
  // negative unit numerator is kept as `-1/d*X` (e.g. `-1/4*Pi`), so
  // `Rational[-1, d]` is NOT treated as a denominator factor here.
  if let Expr::FunctionCall { name, args } = expr
    && name == "Rational"
    && args.len() == 2
    && let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1])
    && (n.abs() > 1 || *n == 1)
    && d.abs() > 1
  {
    return true;
  }
  let exponent = match expr {
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      &args[1]
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      right,
      ..
    } => right.as_ref(),
    _ => return false,
  };
  match exponent {
    Expr::Integer(n) => *n < 0,
    Expr::Real(r) => *r < 0.0,
    Expr::FunctionCall { name: rn, args: ra }
      if rn == "Rational" && ra.len() == 2 =>
    {
      matches!(&ra[0], Expr::Integer(n) if *n < 0)
    }
    Expr::FunctionCall { name: tn, args: ta }
      if tn == "Times"
        && !ta.is_empty()
        && (matches!(&ta[0], Expr::Integer(n) if *n < 0)
          || matches!(&ta[0], Expr::Real(r) if *r < 0.0)) =>
    {
      true
    }
    Expr::FunctionCall { name: tn, args: ta }
      if tn == "Times"
        && !ta.is_empty()
        && matches!(&ta[0], Expr::FunctionCall { name: rn, args: ra }
          if rn == "Rational" && ra.len() == 2
          && matches!(&ra[0], Expr::Integer(n) if *n < 0)) =>
    {
      true
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      ..
    } => true,
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      ..
    } => matches!(left.as_ref(), Expr::Integer(-1)),
    _ => false,
  }
}

/// Given Power[base, negative_exp], return Power[base, positive_exp] for denominator display.
/// For exponent -1, returns just the base (since base^1 = base).
/// Handles both FunctionCall and BinaryOp representations.
fn denominator_form(expr: &Expr) -> Expr {
  // A Rational[n, d] denominator factor contributes its denominator `d`; the
  // numerator is emitted on the numerator side by the caller (and is 1 for the
  // `Rational[1, d]` unit case).
  if let Expr::FunctionCall { name, args } = expr
    && name == "Rational"
    && args.len() == 2
    && let Expr::Integer(d) = &args[1]
  {
    return Expr::Integer(d.abs());
  }
  let (base, exponent) = match expr {
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      (&args[0], &args[1])
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => (left.as_ref(), right.as_ref()),
    _ => unreachable!(),
  };
  let pos_exp = match exponent {
    Expr::Integer(n) => Expr::Integer(-n),
    Expr::Real(r) => Expr::Real(-r),
    Expr::FunctionCall { name: rn, args: ra }
      if rn == "Rational" && ra.len() == 2 =>
    {
      if let Expr::Integer(n) = &ra[0] {
        Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(-n), ra[1].clone()].into(),
        }
      } else {
        unreachable!()
      }
    }
    Expr::FunctionCall { name: tn, args: ta }
      if tn == "Times"
        && ta.len() >= 2
        && matches!(&ta[0], Expr::Real(r) if *r < 0.0) =>
    {
      if let Expr::Real(r) = &ta[0] {
        let mut new_args = vec![Expr::Real(-r)];
        new_args.extend_from_slice(&ta[1..]);
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: new_args.into(),
        }
      } else {
        unreachable!()
      }
    }
    Expr::FunctionCall { name: tn, args: ta }
      if tn == "Times"
        && ta.len() >= 2
        && matches!(&ta[0], Expr::Integer(n) if *n < 0) =>
    {
      if let Expr::Integer(n) = &ta[0] {
        let pos_coeff = Expr::Integer(-n);
        if matches!(pos_coeff, Expr::Integer(1)) {
          // Times[-1, a, b, ...] → Times[a, b, ...]
          if ta.len() == 2 {
            ta[1].clone()
          } else {
            unevaluated("Times", &ta[1..])
          }
        } else {
          // Times[-n, a, b, ...] → Times[n, a, b, ...]
          let mut new_args = vec![pos_coeff];
          new_args.extend_from_slice(&ta[1..]);
          if new_args.len() == 1 {
            new_args.remove(0)
          } else {
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: new_args.into(),
            }
          }
        }
      } else {
        unreachable!()
      }
    }
    Expr::FunctionCall { name: tn, args: ta }
      if tn == "Times"
        && ta.len() >= 2
        && matches!(&ta[0], Expr::FunctionCall { name: rn, args: ra }
          if rn == "Rational" && ra.len() == 2
          && matches!(&ra[0], Expr::Integer(n) if *n < 0)) =>
    {
      // Negate the leading Rational coefficient: Rational[-n, d] → Rational[n, d]
      let negated_rational = if let Expr::FunctionCall { args: ra, .. } = &ta[0]
      {
        if let Expr::Integer(n) = &ra[0] {
          Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(-n), ra[1].clone()].into(),
          }
        } else {
          unreachable!()
        }
      } else {
        unreachable!()
      };
      let mut new_args = vec![negated_rational];
      new_args.extend_from_slice(&ta[1..]);
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: new_args.into(),
      }
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => operand.as_ref().clone(),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } if matches!(left.as_ref(), Expr::Integer(-1)) => right.as_ref().clone(),
    _ => unreachable!(),
  };
  // If positive exponent is 1, return just the base
  if matches!(&pos_exp, Expr::Integer(1)) {
    base.clone()
  // If positive exponent is 1/2, return Sqrt[base]
  } else if matches!(&pos_exp, Expr::FunctionCall { name, args } if name == "Rational" && args.len() == 2 && matches!((&args[0], &args[1]), (Expr::Integer(1), Expr::Integer(2))))
  {
    crate::functions::math_ast::make_sqrt(base.clone())
  } else {
    Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![base.clone(), pos_exp].into(),
    }
  }
}

/// Format a list of Times factors, moving negative-exponent Powers to denominator.
/// `formatter` is the string formatting function (expr_to_string or expr_to_output).
fn format_times_with_denominator(
  args: &[Expr],
  formatter: fn(&Expr) -> String,
) -> Option<String> {
  // Check if any factor has negative exponent
  if !args.iter().any(is_denominator_factor) {
    return None;
  }

  let mut numer_factors_owned: Vec<Expr> = Vec::new();
  let mut denom_exprs: Vec<Expr> = Vec::new();
  // When a Rational[n, d] appears alongside denominator factors, split it:
  // n goes to the numerator and d goes to the denominator.
  // E.g. Times[Rational[1,3], Power[2,-1/2]] → 1/(3*Sqrt[2]) instead of (1/3)/Sqrt[2]
  for a in args {
    if let Expr::FunctionCall { name, args: rargs } = a
      && name == "Rational"
      && rargs.len() == 2
      && let (Expr::Integer(n), Expr::Integer(d)) = (&rargs[0], &rargs[1])
      && d.abs() > 1
    {
      let (rn, rd) = if *d > 0 { (*n, *d) } else { (-*n, -*d) };
      if rn != 1 {
        // Wolfram puts the rational's numerator first in the
        // numerator factors, e.g. `(-2*x*Sin[x])/3` not `(x*-2*Sin[x])/3`.
        numer_factors_owned.insert(0, Expr::Integer(rn));
      }
      denom_exprs.insert(0, Expr::Integer(rd));
    } else if is_denominator_factor(a) {
      denom_exprs.push(denominator_form(a));
    } else {
      numer_factors_owned.push(a.clone());
    }
  }
  let numer_factors: Vec<&Expr> = numer_factors_owned.iter().collect();

  // Format a single factor with parens around Plus
  let fmt_factor = |a: &Expr| -> String {
    let s = formatter(a);
    if matches!(a, Expr::FunctionCall { name, .. } if name == "Plus")
      || matches!(
        a,
        Expr::BinaryOp {
          op: BinaryOperator::Plus | BinaryOperator::Minus,
          ..
        }
      )
    {
      format!("({s})")
    } else {
      s
    }
  };

  // Format numerator
  let numer_str = if numer_factors.is_empty() {
    "1".to_string()
  } else if numer_factors.len() == 1 {
    fmt_factor(numer_factors[0])
  } else {
    let inner = numer_factors
      .iter()
      .map(|a| fmt_factor(a))
      .collect::<Vec<_>>()
      .join("*");
    format!("({inner})")
  };

  // Format denominator
  let needs_parens = |e: &Expr| -> bool {
    matches!(e, Expr::FunctionCall { name, .. } if name == "Plus" || name == "Times")
      || matches!(
        e,
        Expr::BinaryOp {
          op: BinaryOperator::Plus
            | BinaryOperator::Minus
            | BinaryOperator::Times,
          ..
        }
      )
  };
  let denom_str = if denom_exprs.len() == 1 {
    let s = formatter(&denom_exprs[0]);
    if needs_parens(&denom_exprs[0]) {
      format!("({s})")
    } else {
      s
    }
  } else {
    let inner = denom_exprs
      .iter()
      .map(|a| {
        let s = formatter(a);
        if needs_parens(a) { format!("({s})") } else { s }
      })
      .collect::<Vec<_>>()
      .join("*");
    format!("({inner})")
  };

  Some(format!("{numer_str}/{denom_str}"))
}

fn expr_to_part_index_string(expr: &Expr, form: ExprForm) -> String {
  // A Span index renders with the `;;` operator in InputForm
  // (l[[5 ;; 2]]) but keeps its functional head form in OutputForm and
  // FullForm (l[[Span[5, 2]]]), matching wolframscript. A nested Span
  // argument is parenthesised.
  if form == ExprForm::Input
    && let Expr::FunctionCall { name, args } = expr
    && name == "Span"
    && (args.len() == 2 || args.len() == 3)
  {
    return args
      .iter()
      .map(|a| {
        let s = format_expr(a, form);
        if matches!(a, Expr::FunctionCall { name: n, .. } if n == "Span") {
          format!("({s})")
        } else {
          s
        }
      })
      .collect::<Vec<_>>()
      .join(" ;; ");
  }
  format_expr(expr, form)
}

/// The form to use for formatting expressions.
#[derive(Clone, Copy, PartialEq)]
pub enum ExprForm {
  /// InputForm: strings are quoted, full round-trip syntax
  Input,
  /// OutputForm: strings are unquoted, display-oriented
  Output,
}

/// Detect the marker convention used by `Expr::Association` to indicate that
/// an entry was originally `key :> value` (RuleDelayed). The value side is
/// `RuleDelayed { pattern, replacement }` where `pattern` equals the key.
/// Used both by internally-constructed associations (Tabular failure
/// messages with `Expr::String` keys) and by associations parsed from
/// `<|key :> value|>` literals (any key type). `Expr` does not implement
/// `PartialEq`, so we compare via the canonical InputForm rendering.
pub(crate) fn assoc_marker_matches(key: &Expr, pattern: &Expr) -> bool {
  match (key, pattern) {
    (Expr::String(a), Expr::String(b)) => a == b,
    _ => expr_to_string(key) == expr_to_string(pattern),
  }
}

/// Recursively rewrite every `Expr::Association` inside `expr` into the
/// long-form `Association[…]` `FunctionCall`. Used when rendering an
/// outer unevaluated `Association[…]` to match wolframscript's display
/// rule: nested associations print in long form, never the `<|…|>`
/// shorthand. Other expression types pass through unchanged.
fn rewrite_assoc_to_long_form(expr: &Expr) -> Expr {
  match expr {
    Expr::Association(pairs) => {
      let mut converted_args: Vec<Expr> = Vec::with_capacity(pairs.len());
      for (k, v) in pairs {
        let kk = rewrite_assoc_to_long_form(k);
        let vv = rewrite_assoc_to_long_form(v);
        match v {
          Expr::RuleDelayed { pattern, .. }
            if assoc_marker_matches(k, pattern) =>
          {
            converted_args.push(Expr::FunctionCall {
              name: "RuleDelayed".to_string(),
              args: vec![kk, vv].into(),
            });
          }
          _ => {
            converted_args.push(Expr::FunctionCall {
              name: "Rule".to_string(),
              args: vec![kk, vv].into(),
            });
          }
        }
      }
      Expr::FunctionCall {
        name: "Association".to_string(),
        args: converted_args.into(),
      }
    }
    Expr::List(items) => {
      Expr::List(items.iter().map(rewrite_assoc_to_long_form).collect())
    }
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args.iter().map(rewrite_assoc_to_long_form).collect(),
    },
    Expr::Rule {
      pattern,
      replacement,
    } => Expr::Rule {
      pattern: Box::new(rewrite_assoc_to_long_form(pattern)),
      replacement: Box::new(rewrite_assoc_to_long_form(replacement)),
    },
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => Expr::RuleDelayed {
      pattern: Box::new(rewrite_assoc_to_long_form(pattern)),
      replacement: Box::new(rewrite_assoc_to_long_form(replacement)),
    },
    other => other.clone(),
  }
}

/// Unified expression formatter that handles both InputForm and OutputForm.
/// Render the argument of `PercentForm[...]`. A non-negative machine real
/// becomes `<x*100>%` (with any trailing decimal point dropped); a list is
/// rendered element-wise (recursing so nested lists also format); every other
/// value (integer, rational, negative real, symbolic) is shown unchanged.
fn format_percent_form(expr: &Expr, form: ExprForm) -> String {
  match expr {
    Expr::Real(f) if *f >= 0.0 => {
      let scaled = format_real(f * 100.0);
      let trimmed = scaled.strip_suffix('.').unwrap_or(&scaled);
      format!("{trimmed}%")
    }
    Expr::List(items) => {
      let parts: Vec<String> =
        items.iter().map(|e| format_percent_form(e, form)).collect();
      format!("{{{}}}", parts.join(", "))
    }
    other => format_expr(other, form),
  }
}

pub fn format_expr(expr: &Expr, form: ExprForm) -> String {
  // Grow the stack when running low so formatting a deeply nested expression
  // (e.g. SameQ/UnsameQ on Nest[f, x, 5000], which compares via expr_to_string)
  // doesn't overflow. Mirrors the guard on evaluate_expr_to_expr; the recursion
  // re-enters through this public entry, so every level is checked.
  stacker::maybe_grow(2 * 1024 * 1024, 4 * 1024 * 1024, || {
    // A symbol is stored under its full context name but written under the
    // short one wherever that reads back as the same symbol — `P`pub` is
    // `pub` while `P`` is on `$ContextPath`. Doing that once, at the
    // outermost call, keeps every renderer consistent without each of them
    // having to know about contexts. Formatting re-enters this function for
    // sub-expressions, so the guard keeps it to a single pass.
    with_display_names(expr, |e| format_expr_impl(e, form))
  })
}

thread_local! {
  /// Set while a formatting call is already showing context-resolved
  /// symbols under their visible names.
  static DISPLAY_PASS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Render `expr` with every symbol written the way it reads back: `P`pub`
/// as `pub` while `P`` is on `$ContextPath`, and under its full name once it
/// is not. Renderers re-enter each other, so the rewrite runs once, at the
/// outermost call, and the inner ones format what it produced.
pub(crate) fn with_display_names(
  expr: &Expr,
  render: impl FnOnce(&Expr) -> String,
) -> String {
  if !crate::evaluator::contexts::contexts_active()
    || DISPLAY_PASS.with(std::cell::Cell::get)
  {
    return render(expr);
  }
  struct DisplayPass;
  impl Drop for DisplayPass {
    fn drop(&mut self) {
      DISPLAY_PASS.with(|p| p.set(false));
    }
  }
  let displayed = crate::evaluator::contexts::to_display(expr);
  DISPLAY_PASS.with(|p| p.set(true));
  let _guard = DisplayPass;
  render(&displayed)
}

/// Whether `e` prints with a leading `!`. `Not` binds looser than arithmetic,
/// comparison, `;;`, `!` and function application, so wolframscript wraps it
/// in parens inside any of those — `( !a)^2` — but not inside `&&`, `||`,
/// `Xor`, a rule, a list or an argument, which bind looser still.
fn prints_as_not(e: &Expr) -> bool {
  matches!(
    e,
    Expr::UnaryOp {
      op: UnaryOperator::Not,
      ..
    }
  ) || matches!(e, Expr::FunctionCall { name, args }
    if name == "Not" && args.len() == 1)
}

/// Parse-side precedence (the `operator_precedence` scale) of the operator an
/// expression prints with at top level, or `None` for forms that print
/// self-delimiting (atoms, `f[…]`, `{…}`). Used to decide whether an operand
/// of an infix shorthand (`@@`, `/@`, `@@@`, `.`) needs parentheses so the
/// printed form re-parses to the same tree — e.g. `Plus @@ (x*y)` must not
/// print as `Plus @@ x*y`, which re-parses as `(Plus @@ x)*y`.
fn printed_infix_precedence(e: &Expr) -> Option<u8> {
  match e {
    Expr::BinaryOp { op, .. } => Some(match op {
      BinaryOperator::Or => 15,
      BinaryOperator::And => 18,
      BinaryOperator::Alternatives => 27,
      BinaryOperator::Plus
      | BinaryOperator::Minus
      | BinaryOperator::StringJoin => 30,
      BinaryOperator::Times | BinaryOperator::Divide => 33,
      BinaryOperator::Power => 48,
    }),
    Expr::Comparison { .. } => Some(21),
    Expr::Rule { .. } | Expr::RuleDelayed { .. } => Some(12),
    Expr::ReplaceAll { .. } | Expr::ReplaceRepeated { .. } => Some(7),
    Expr::CompoundExpr(_) => Some(1),
    // `body &` binds looser than every operator shorthand.
    Expr::Function { .. } => Some(5),
    Expr::Map { .. } => Some(47),
    Expr::Apply { .. } | Expr::MapApply { .. } => Some(46),
    // ` !x` sits between `&&` and the application shorthands.
    Expr::UnaryOp {
      op: UnaryOperator::Not,
      ..
    } => Some(20),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      ..
    } => Some(45),
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Not" if args.len() == 1 => Some(20),
      "Dot" if args.len() == 2 => Some(40),
      "Plus" if args.len() >= 2 => Some(30),
      "Times" if args.len() >= 2 => Some(33),
      // `=`, `:=`, `^=` and `^:=` print infix and bind looser than every
      // operator shorthand except `;` and `&`.
      "Set" | "SetDelayed" | "UpSet" | "UpSetDelayed" if args.len() == 2 => {
        Some(6)
      }
      // Heads that also print infix once evaluated (`a && b` keeps the head
      // `And` but still renders with the operator).
      "Or" if args.len() >= 2 => Some(15),
      "And" if args.len() >= 2 => Some(18),
      "Alternatives" if args.len() >= 2 => Some(27),
      "Rule" | "RuleDelayed" if args.len() == 2 => Some(12),
      // `/;` and `:` print infix and sit just above `->`, so both need
      // parens inside anything tighter: `a == (x_ /; y)`, `a == (b:c)`.
      "Condition" if args.len() == 2 => Some(13),
      "Pattern" if args.len() == 2 => Some(14),
      "StringJoin" if args.len() >= 2 => Some(30),
      "Power" if args.len() == 2 => Some(48),
      _ => None,
    },
    _ => None,
  }
}

/// Render the operands of a left-associative infix head in InputForm and
/// join them with `sep`. An operand that itself prints with a looser
/// operator than `prec` is parenthesized so the text re-parses to the same
/// tree — for an operand of the *same* precedence only the left one can
/// stay bare (`(a /; b) /; c` prints `a /; b /; c`, `a /; (b /; c)` keeps
/// its parentheses, both as in wolframscript). Forms that print
/// self-delimiting (atoms, `f[…]`, and heads like `Span` that print infix
/// only in InputForm but bind tighter than the heads using this helper)
/// never need parentheses.
fn input_form_infix(args: &[Expr], sep: &str, prec: u8) -> String {
  input_form_infix_forcing(args, sep, prec, |_| false)
}

/// `input_form_infix` with an extra escape hatch: `force_parens` brackets an
/// operand the precedence rule alone would leave bare. Needed for heads that
/// are *not* Flat, where the leading operand's own parentheses carry meaning
/// (`Alternatives[Alternatives[a, b], c]` must print `(a | b) | c`, since a
/// bare `a | b | c` re-parses to the flat three-argument form).
fn input_form_infix_forcing(
  args: &[Expr],
  sep: &str,
  prec: u8,
  force_parens: impl Fn(&Expr) -> bool,
) -> String {
  args
    .iter()
    .enumerate()
    .map(|(i, a)| {
      let s = expr_to_input_form(a);
      let looser = printed_infix_precedence(a)
        .is_some_and(|p| if i == 0 { p < prec } else { p <= prec });
      if looser || force_parens(a) {
        format!("({s})")
      } else {
        s
      }
    })
    .collect::<Vec<_>>()
    .join(sep)
}

/// An explicitly nested `Alternatives[a, b]` operand, i.e. the head form the
/// parser never produces for `a | b` chains (those are `BinaryOp`s, which are
/// genuinely flat and print bare).
fn is_nested_alternatives_call(e: &Expr) -> bool {
  matches!(e, Expr::FunctionCall { name, args }
    if name == "Alternatives" && args.len() >= 2)
}

/// Whether the body of `body &` has to print parenthesized.
///
/// `&` binds tighter than `;` and than every assignment operator, so a body
/// that prints as one of those re-parses the wrong way when printed bare:
/// `(a; b) &` would print as `a; b & ` and read back as
/// `CompoundExpression[a, b &]`, and `(a = 1) &` as `Set[a, 1 &]`. Both
/// change what the expression means — badly so when the body assigns a
/// slot-carrying value, which is how a Demonstration's `Initialization`
/// builds its lookup tables.
fn function_body_needs_parens(body: &Expr) -> bool {
  match body {
    Expr::CompoundExpr(_) => true,
    Expr::FunctionCall { name, args } => {
      args.len() == 2
        && matches!(
          name.as_str(),
          "Set"
            | "SetDelayed"
            | "UpSet"
            | "UpSetDelayed"
            | "AddTo"
            | "SubtractFrom"
            | "TimesBy"
            | "DivideBy"
        )
    }
    _ => false,
  }
}

/// Whether a `Part` base must be parenthesized so `base[[i]]` re-parses to
/// the same tree. `[[…]]` binds tighter than every infix operator, so any
/// base that prints with a top-level operator (`a /. b`, `a + b`, `x &`, …)
/// or as a negative literal needs parens: `(a /. b :> c)[[1]]` must not
/// print as `a /. b :> c[[1]]`.
fn part_base_needs_parens(base: &Expr) -> bool {
  printed_infix_precedence(base).is_some()
    || matches!(base, Expr::Integer(n) if *n < 0)
    || matches!(base, Expr::Real(f) if *f < 0.0)
}

/// Render a Part application as `base[[i,j,…]]`, parenthesizing the base when
/// it needs it. Part reaches the renderers in two shapes — `Expr::Part` (the
/// final bracket group of a Part suffix) and a `Part`-headed `FunctionCall`
/// (every earlier group, kept in call form so `m[[a]][[b]]` stays distinct
/// from `m[[a, b]]`) — and wolframscript prints both with double brackets.
fn format_part_brackets(
  base: &Expr,
  base_str: &str,
  indices: &[String],
) -> String {
  let joined = indices.join(",");
  if part_base_needs_parens(base) {
    format!("({base_str})[[{joined}]]")
  } else {
    format!("{base_str}[[{joined}]]")
  }
}

/// Whether an operand of an infix comparison (`==`, `<`, `=!=`, …) must be
/// parenthesized so the printed form re-parses to the same tree. Comparison
/// binds looser than `+`, `*`, `^`, `.`, `/@` and `@@`, so those print bare,
/// while everything looser — `&&`, `||`, `|`, `->`, `/.`, `=`, `body &`,
/// `;` and a nested comparison — needs parens: `a == (b < c)`, not
/// `a == b < c`.
/// A bare pattern also gets parens, matching wolframscript's `(a_) != (b_)`,
/// and so does anything printing with a leading `!`.
///
/// This is the single place the rule lives: both InputForm renderers
/// (`format_expr` and `expr_to_input_form`) call it for every operand, chained
/// comparisons included (`Hold[(a_) == (b_) == (c_)]`).
fn comparison_operand_needs_parens(e: &Expr) -> bool {
  matches!(
    e,
    Expr::Pattern { .. }
      | Expr::PatternOptional { .. }
      | Expr::PatternTest { .. }
  ) || prints_as_not(e)
    || printed_infix_precedence(e).is_some_and(|p| p < 30)
}

/// Whether the *leading* term of a `Plus`/`Subtract` chain must be
/// parenthesized in InputForm so the printed form re-parses to the same
/// tree. `+`/`-` bind tighter than every looser operator (`Rule`,
/// `ReplaceAll`, `Or`, `And`, `CompoundExpression`, …), so a term printing
/// with one of those needs parens: `(x /. sol) - 0.05`, not
/// `x /. sol - 0.05`, which re-parses as `x /. (sol - 0.05)` — a genuine
/// Wolfram Demonstrations regression this once caused, since `ReplaceAll`
/// (precedence 7) binds far looser than `Plus`/`Minus` (precedence 30).
///
/// A term at precedence exactly 30 (a nested `Plus`/`Minus`/`StringJoin`)
/// does *not* need parens here: text re-parses `+`/`-` left-associatively,
/// so a left-nested chain round-trips whether or not the leading term was
/// itself a `Plus` — unlike a *non*-leading term at the same precedence,
/// see `plus_nonleading_term_needs_parens`.
fn plus_term_needs_parens(e: &Expr) -> bool {
  printed_infix_precedence(e).is_some_and(|p| p < 30)
}

/// Whether a *non-leading* term of a `Plus`/`Subtract` chain — one printed
/// after a `+` or `-` — must be parenthesized in InputForm. Unlike the
/// leading term (`plus_term_needs_parens`), a term at precedence exactly 30
/// (a nested `Plus`/`Minus`/`StringJoin`) also needs parens here:
/// left-associative re-parsing only reconstructs a *left*-nested tree, so a
/// bare non-leading `Plus`/`Minus` term — which can only have survived
/// un-flattened under `Hold` — silently regroups. Following a `-` this is
/// not just a shape change but a value change: printing `Plus[a,
/// -(b + c)]` as `a - b + c` drops the sign that should apply to `c`
/// (`a - b + c` re-parses as `Plus[a, -b, c]`, not `Plus[a, -b, -c]`).
fn plus_nonleading_term_needs_parens(e: &Expr) -> bool {
  printed_infix_precedence(e).is_some_and(|p| p <= 30)
}

/// Render an application shorthand (`f /@ list`, `f @@ list`, `f @@@ list`)
/// in InputForm, parenthesizing either operand when it would otherwise bind
/// differently on re-parse. `prec` is the shorthand's own parse precedence
/// (`operator_precedence`): the left operand needs parens at equal precedence
/// too (the shorthands are right-associative), the right one only below it.
fn format_map_apply_shorthand(
  func: &Expr,
  list: &Expr,
  op: &str,
  prec: u8,
) -> String {
  let func_str = format_expr(func, ExprForm::Input);
  let func_needs_parens = matches!(func, Expr::NamedFunction { .. })
    || printed_infix_precedence(func).is_some_and(|p| p <= prec);
  let func_display = if func_needs_parens {
    format!("({func_str})")
  } else {
    func_str
  };
  let list_str = format_expr(list, ExprForm::Input);
  let list_display = if printed_infix_precedence(list).is_some_and(|p| p < prec)
  {
    format!("({list_str})")
  } else {
    list_str
  };
  format!("{func_display} {op} {list_display}")
}

/// Parenthesize an assignment's already-formatted RHS if the RHS is a
/// `;`-sequence: `CompoundExpression` has the lowest precedence of any
/// operator, so printing `lhs := a; b; c` without parens re-parses as
/// `(lhs := a); b; c`, silently dropping `b` and `c` from the delayed
/// definition — the Demonstrations idiom `f[x_] := (a = x; b = f[a]; b)`
/// for a multi-statement body would lose everything after the first
/// statement. Applies equally to `=`, `:=`, `^=` and `^:=`.
pub(crate) fn assignment_rhs_needs_parens(rhs: &Expr) -> bool {
  matches!(rhs, Expr::CompoundExpr(_))
}

fn format_expr_impl(expr: &Expr, form: ExprForm) -> String {
  let fmt = |e: &Expr| -> String { format_expr(e, form) };
  let fmt_fn: fn(&Expr) -> String = match form {
    ExprForm::Input => expr_to_string,
    ExprForm::Output => expr_to_output,
  };
  let is_output = form == ExprForm::Output;

  // `Not[x]` is the same node the `!` operator builds, and wolframscript
  // writes both with the operator. Held expressions keep the call form, so
  // route it through the operator branch below.
  if let Expr::FunctionCall { name, args } = expr
    && name == "Not"
    && args.len() == 1
  {
    return format_expr(
      &Expr::UnaryOp {
        op: UnaryOperator::Not,
        operand: Box::new(args[0].clone()),
      },
      form,
    );
  }

  match expr {
    Expr::Integer(n) => n.to_string(),
    Expr::BigInteger(n) => n.to_string(),
    Expr::Real(f) => format_real(*f),
    Expr::BigFloat(digits, prec) => format_bigfloat(digits, *prec),
    Expr::String(s) => {
      if is_output {
        // A string that opens with a box segment renders as
        // DisplayForm[<box expression>] in OutputForm (matching
        // wolframscript), with any trailing prose kept.
        //
        // A box segment further into the string is deliberately left alone:
        // the SVG label renderer (`box_string_to_svg`) typesets those
        // markers into sub/superscript tspans, and it reads the label text
        // that this function produces.
        if s.starts_with(crate::functions::string_ast::BOX_START) {
          return box_string_to_display_form(s);
        }
        // Render private-use escape codepoints back to their `\X` source form
        // (matching wolframscript). Inputs land here from `"\("` etc. or from
        // FromCharacterCode of the codepoint values.
        if s.chars().any(|c| {
          c == '\u{F7CD}'
            || c == crate::functions::string_ast::BOX_OPEN
            || c == crate::functions::string_ast::BOX_CLOSE
            || c == crate::functions::string_ast::BOX_START
            || c == crate::functions::string_ast::BOX_SEP
        }) {
          let mut out = String::with_capacity(s.len());
          for c in s.chars() {
            match c {
              '\u{F7CD}' => out.push_str("\\`"),
              c if c == crate::functions::string_ast::BOX_OPEN => {
                out.push_str("\\(");
              }
              c if c == crate::functions::string_ast::BOX_CLOSE => {
                out.push_str("\\)");
              }
              c if c == crate::functions::string_ast::BOX_START => {
                out.push_str("\\!");
              }
              c if c == crate::functions::string_ast::BOX_SEP => {
                out.push_str("\\*");
              }
              _ => out.push(c),
            }
          }
          return out;
        }
        s.clone() // No quotes for display
      } else if in_output_form() && !in_true_input_form() {
        // OutputForm-originated render that re-entered via the InputForm path
        // (held BinaryOp / operator shorthand). Strings stay unquoted — unless
        // we are inside genuine InputForm (`expr_to_input_form`), which itself
        // routes some nodes through `expr_to_output` and must keep quotes.
        s.clone()
      } else {
        let escaped = escape_string_for_input_form(s);
        format!("\"{escaped}\"")
      }
    }
    Expr::Identifier(s) => s.clone(),
    Expr::Slot(n) => {
      format!("#{n}")
    }
    Expr::SlotSequence(n) => {
      format!("##{n}")
    }
    Expr::List(items) => {
      let parts: Vec<String> = items.iter().map(&fmt).collect();
      format!("{{{}}}", parts.join(", "))
    }
    // `Definition[sym]` and `FullDefinition[sym]` stay held (like
    // wolframscript) and print as the symbol's definition text.
    Expr::FunctionCall { name, args }
      if (name == "Definition" || name == "FullDefinition")
        && args.len() == 1
        && matches!(&args[0], Expr::Identifier(_)) =>
    {
      match &args[0] {
        Expr::Identifier(sym) if name == "Definition" => {
          crate::evaluator::dispatch::complex_and_special::definition_text(sym)
            .unwrap_or_default()
        }
        Expr::Identifier(sym) => {
          crate::evaluator::dispatch::complex_and_special::full_definition_text(
            sym,
          )
          .unwrap_or_default()
        }
        _ => unreachable!(),
      }
    }
    Expr::FunctionCall { name, args } => {
      // Named slot Slot["name"] displays as #name (matching wolframscript).
      if name == "Slot"
        && args.len() == 1
        && let Expr::String(key) = &args[0]
        && key.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
        && key.chars().all(|c| c.is_ascii_alphanumeric())
      {
        return format!("#{key}");
      }
      // `Part[base, i, …]` prints with double brackets, not as a call.
      if name == "Part" && !args.is_empty() {
        let indices: Vec<String> = args[1..]
          .iter()
          .map(|a| expr_to_part_index_string(a, form))
          .collect();
        return format_part_brackets(
          &args[0],
          &format_expr(&args[0], form),
          &indices,
        );
      }
      // Sound[...] always renders as -Sound- (matching wolframscript REPL),
      // regardless of what primitives it wraps.
      if name == "Sound" && !args.is_empty() {
        return "-Sound-".to_string();
      }
      // A raw (held, unevaluated) Graphics[...] / Graphics3D[...] call still
      // summarizes to -Graphics-/-Graphics3D- in OutputForm, matching
      // wolframscript. This applies wherever the call sits — including a
      // Graphics argument held inside a symbolic wrapper such as
      // LocatorPane[Dynamic[p], Graphics[...]] or ClickPane[Graphics[...], f].
      // (InputForm/FullForm still print the full expression.)
      if is_output
        && !args.is_empty()
        && (name == "Graphics" || name == "Graphics3D")
      {
        return if name == "Graphics3D" {
          "-Graphics3D-".to_string()
        } else {
          "-Graphics-".to_string()
        };
      }
      // PercentForm[x] — wolframscript renders a non-negative machine real
      // as a percentage: x*100 with a trailing "%" (and the trailing decimal
      // point dropped, so 0.25 -> 25%, 0.999 -> 99.9%). Integers, rationals,
      // negative reals, and symbolic values display unchanged. PercentForm is
      // not Listable (Attributes are {NHoldRest, Protected}); instead the
      // renderer recurses into list structure, percent-formatting only the
      // real leaves (PercentForm[{1/4, 0.5, 2}] -> {1/4, 50%, 2}). The head
      // is preserved; only the rendering differs.
      if name == "PercentForm" && args.len() == 1 {
        return format_percent_form(&args[0], form);
      }
      // Graph[vertices, edges, opts...] — wolframscript prints the
      // summary `Graph[<n>, <m>]` (vertex count, edge count).
      if name == "Graph"
        && args.len() >= 2
        && let (Expr::List(verts), Expr::List(edges)) = (&args[0], &args[1])
      {
        return format!("Graph[<{}>, <{}>]", verts.len(), edges.len());
      }
      // Inequality[a, Op, b, Op, c] — always use head form. wolframscript
      // keeps the head in every form, including script-mode OutputForm:
      // Reduce[x^2 <= 4 && x > 0, x] prints Inequality[0, Less, x,
      // LessEqual, 2], not the chained 0 < x <= 2.
      if name == "Inequality" && args.len() >= 5 && args.len() % 2 == 1 {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        return format!("Inequality[{}]", parts.join(", "));
      }
      // Sequence[] (empty sequence) displays as nothing in Wolfram output (OutputForm only)
      if is_output && name == "Sequence" && args.is_empty() {
        return String::new();
      }
      // Special case: Quantity[n, unit] — unit shown as quoted string(s)
      if name == "Quantity" && args.len() == 2 {
        let mag_str = fmt(&args[0]);
        // A MixedUnit[{…}] holds one unit name per magnitude part. Only
        // OutputForm drops their quotes, like it does for any other
        // string; InputForm writes them out.
        let is_mixed_unit = matches!(
          &args[1],
          Expr::FunctionCall { name, args }
            if name == "MixedUnit"
              && args.len() == 1
              && matches!(&args[0], Expr::List(_))
        );
        let unit_str = if is_mixed_unit && !is_output {
          expr_to_input_form(&args[1])
        } else {
          quantity_unit_to_string(&args[1])
        };
        return format!("Quantity[{mag_str}, {unit_str}]");
      }
      // OutputForm-only: FullForm, CForm, TeXForm, FortranForm wrap inner in output form
      if is_output && name == "FullForm" && args.len() == 1 {
        return format!("FullForm[{}]", fmt(&args[0]));
      }
      if is_output && name == "CForm" && args.len() == 1 {
        return format!("CForm[{}]", fmt(&args[0]));
      }
      if is_output && name == "TeXForm" && args.len() == 1 {
        return format!("TeXForm[{}]", fmt(&args[0]));
      }
      if is_output && name == "FortranForm" && args.len() == 1 {
        return format!("FortranForm[{}]", fmt(&args[0]));
      }
      // Special case: URLRead's connection Failure. wolframscript renders
      // the failure object through its URLRead::iurl message template
      // ("Could not connect to `1`.") with the URL wrapped in the display
      // boxes the FE would use; script mode prints those boxes literally.
      if is_output
        && name == "Failure"
        && args.len() == 2
        && let Expr::Association(pairs) = &args[1]
        && pairs.iter().any(|(k, v)| {
          matches!(k, Expr::String(s) if s == "MessageTemplate")
            && matches!(v, Expr::RuleDelayed { replacement, .. }
              if matches!(&**replacement, Expr::FunctionCall { name, args }
                if name == "MessageName"
                  && matches!(&args[..],
                    [Expr::Identifier(s), Expr::String(t)]
                      if s == "URLRead" && t == "iurl")))
        })
        && let Some(Expr::String(url)) = pairs.iter().find_map(|(k, v)| match k
        {
          Expr::String(s) if s == "URL" => Some(v),
          _ => None,
        })
      {
        return format!(
          "Could not connect to DisplayForm[TagBox[\"{url}\", Short[#1, 3] & ]]."
        );
      }
      // Special case: ByteArray
      if name == "ByteArray" && args.len() == 1 && is_output {
        // OutputForm: ByteArray[<n>]
        if let Expr::String(b64) = &args[0] {
          use base64::Engine;
          let engine = base64::engine::general_purpose::STANDARD;
          if let Ok(decoded) = engine.decode(b64) {
            return format!("ByteArray[<{}>]", decoded.len());
          }
        }
        if let Expr::List(items) = &args[0] {
          return format!("ByteArray[<{}>]", items.len());
        }
      }
      // Special case: NumericArray. wolframscript hides the data and
      // displays only the dimensions, like `NumericArray[<2,2>, UnsignedInteger8]`.
      // Applies to the 2-arg form `NumericArray[list, dtype]` produced by
      // the dispatcher in evaluation_control.rs.
      if name == "NumericArray"
        && args.len() == 2
        && is_output
        && let Expr::List(_) = &args[0]
      {
        let dims = numeric_array_dims(&args[0]);
        let dtype = match &args[1] {
          Expr::String(s) => s.clone(),
          other => fmt(other),
        };
        let dim_str = dims
          .iter()
          .map(std::string::ToString::to_string)
          .collect::<Vec<_>>()
          .join(",");
        return format!("NumericArray[<{dim_str}>, {dtype}]");
      }
      // Special case: InterpolatingFunction[domain, data] — hide data with
      // <> for display only; InputForm falls through to the full literal
      // data so re-parsing it (e.g. Manipulate's per-frame state
      // round-trip) recovers the same object instead of hitting `<>`,
      // which isn't valid input syntax.
      if name == "InterpolatingFunction"
        && (args.len() == 2 || args.len() == 3)
        && is_output
      {
        return format!("InterpolatingFunction[{}, <>]", fmt(&args[0]));
      }
      // Special case: Skeleton[n] displays as <<n>>
      if name == "Skeleton" && args.len() == 1 {
        return format!("<<{}>>", fmt(&args[0]));
      }
      // Special case: StringSkeleton[n] displays as <<n>>
      if name == "StringSkeleton" && args.len() == 1 {
        return format!("<<{}>>", fmt(&args[0]));
      }
      // MessageName[sym, "tag"] shows as sym::tag under ToString, the same
      // infix form InputForm uses — `ToString[k::usage]` is "k::usage" in
      // wolframscript. The bare script-mode echo keeps the long head form
      // (see `IN_TO_STRING`).
      if is_output && in_to_string() && name == "MessageName" && args.len() == 2
      {
        let tag = match &args[1] {
          Expr::String(s) => s.clone(),
          Expr::Identifier(s) => s.clone(),
          other => fmt(other),
        };
        return format!("{}::{}", fmt(&args[0]), tag);
      }
      // Special case: Repeated[x] displays as x.. (or (x).. when x is a
      // Pattern or numeric atom, matching wolframscript's
      // parenthesisation of `_`-bearing patterns and numeric literals
      // — `Repeated[0]` prints as `(0)..`, not `0..`, since `0..`
      // would be lex-ambiguous with the `0.` Real literal).
      let is_pattern_arg = |e: &Expr| -> bool {
        matches!(
          e,
          Expr::Pattern { .. }
            | Expr::PatternOptional { .. }
            | Expr::PatternTest { .. }
            | Expr::Integer(_)
            | Expr::BigInteger(_)
            | Expr::Real(_)
            | Expr::BigFloat(_, _)
        ) || matches!(
          e,
          Expr::FunctionCall { name: pn, .. } if pn == "Pattern"
        )
      };
      // Special case: Pattern[name, body] displays as name:body
      // (e.g. `y : 1` → `Pattern[y, 1]` → `y:1`). When `body` is a
      // looser-binding operator like Condition/Rule/RuleDelayed/
      // ReplaceAll/ReplaceRepeated, wrap it in parens so `s:a /; b`
      // doesn't flip to `Condition[Pattern[s, a], b]` on re-parse.
      // When `body` is Blank/BlankSequence/BlankNullSequence, fold the
      // name into the underscore form Wolfram prefers: `Pattern[x,
      // Blank[]]` → `x_`, `Pattern[x, Blank[Integer]]` → `x_Integer`,
      // and so on for `__`/`___`.
      if name == "Pattern" && args.len() == 2 {
        if let Expr::Identifier(nm) = &args[0]
          && let Expr::FunctionCall {
            name: bn,
            args: bargs,
          } = &args[1]
          && bargs.len() <= 1
        {
          let underscores = match bn.as_str() {
            "Blank" => Some("_"),
            "BlankSequence" => Some("__"),
            "BlankNullSequence" => Some("___"),
            _ => None,
          };
          if let Some(under) = underscores {
            if bargs.is_empty() {
              return format!("{nm}{under}");
            }
            if let Expr::Identifier(h) = &bargs[0] {
              return format!("{nm}{under}{h}");
            }
          }
        }
        // Also fold when body is the typed `Expr::Pattern` form with an
        // empty name (the anonymous `_`/`__`/`___` parsed as a Term).
        if let Expr::Identifier(nm) = &args[0]
          && let Expr::Pattern {
            name: bname,
            head,
            blank_type,
          } = &args[1]
          && bname.is_empty()
        {
          let under = "_".repeat(*blank_type as usize);
          return match head {
            Some(h) => format!("{nm}{under}{h}"),
            None => format!("{nm}{under}"),
          };
        }
        let needs_parens = matches!(
          &args[1],
          Expr::Rule { .. }
            | Expr::RuleDelayed { .. }
            | Expr::ReplaceAll { .. }
            | Expr::ReplaceRepeated { .. }
        ) || matches!(
          &args[1],
          Expr::FunctionCall { name: bn, .. }
            if matches!(
              bn.as_str(),
              "Condition"
                | "Rule"
                | "RuleDelayed"
                | "ReplaceAll"
                | "ReplaceRepeated"
            )
        );
        let body_str = fmt(&args[1]);
        if needs_parens {
          return format!("{}:({})", fmt(&args[0]), body_str);
        }
        return format!("{}:{}", fmt(&args[0]), body_str);
      }
      if name == "Repeated" && args.len() == 1 {
        let inner = fmt(&args[0]);
        return if is_pattern_arg(&args[0]) {
          format!("({inner})..")
        } else {
          format!("{inner}..")
        };
      }
      // Special case: RepeatedNull[x] displays as x... (mirrors Repeated)
      if name == "RepeatedNull" && args.len() == 1 {
        let inner = fmt(&args[0]);
        return if is_pattern_arg(&args[0]) {
          format!("({inner})...")
        } else {
          format!("{inner}...")
        };
      }
      // Special case: Colon[a, b, ...] displays as a ∶ b ∶ ...
      if name == "Colon" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        return parts.join(" \u{2236} ");
      }
      // Special case: Cap[a, b, ...] displays as a ⌢ b ⌢ ...
      if name == "Cap" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        return parts.join(" \u{2322} ");
      }
      // Special case: Cup[a, b, ...] displays as a ⌣ b ⌣ ...
      if name == "Cup" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        return parts.join(" \u{2323} ");
      }
      // Special case: Congruent[a, b, ...] displays as a ≡ b ≡ ...
      if name == "Congruent" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        return parts.join(" \u{2261} ");
      }
      // Special case: RightTee[a, b, ...] displays as a ⊢ b ⊢ ...
      if name == "RightTee" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        return parts.join(" \u{22A2} ");
      }
      // Special case: DoubleRightTee[a, b, ...] displays as a ⊨ b ⊨ ...
      if name == "DoubleRightTee" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        return parts.join(" \u{22A8} ");
      }
      // Special case: LeftTee[a, b, ...] displays as a ⊣ b ⊣ ...
      if name == "LeftTee" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        return parts.join(" \u{22A3} ");
      }
      // Special case: DoubleLeftTee[a, b, ...] displays as a ⫤ b ⫤ ...
      if name == "DoubleLeftTee" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        return parts.join(" \u{2AE4} ");
      }
      // Special case: Style[expr, directives...] — OutputForm unwraps to
      // just the content expression (matching wolframscript's default
      // display). InputForm keeps Style[...] verbatim.
      if is_output && name == "Style" && !args.is_empty() {
        return fmt(&args[0]);
      }
      // Special case: LongRightArrow[a, b, ...] displays as a ⟶ b ⟶ ...
      if name == "LongRightArrow" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        return parts.join(" \u{27F6} ");
      }
      // Special case: Proportional[a, b, ...] displays as a ∝ b ∝ ...
      if name == "Proportional" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        return parts.join(" \u{221D} ");
      }
      // Blank[] → _, Blank[h] → _h
      if name == "Blank" {
        if args.is_empty() {
          return "_".to_string();
        }
        if args.len() == 1
          && let Expr::Identifier(h) = &args[0]
        {
          return format!("_{h}");
        }
      }
      // BlankSequence[] → __, BlankSequence[h] → __h
      if name == "BlankSequence" {
        if args.is_empty() {
          return "__".to_string();
        }
        if args.len() == 1
          && let Expr::Identifier(h) = &args[0]
        {
          return format!("__{h}");
        }
      }
      // BlankNullSequence[] → ___, BlankNullSequence[h] → ___h
      if name == "BlankNullSequence" {
        if args.is_empty() {
          return "___".to_string();
        }
        if args.len() == 1
          && let Expr::Identifier(h) = &args[0]
        {
          return format!("___{h}");
        }
      }
      // OutputForm-only: BaseForm[expr, base]
      if is_output && name == "BaseForm" && args.len() == 2 {
        return format!("BaseForm[{}, {}]", fmt(&args[0]), fmt(&args[1]));
      }
      // Special case: Rational[num, denom] displays as num/denom
      if name == "Rational" && args.len() == 2 {
        return format!("{}/{}", fmt(&args[0]), fmt(&args[1]));
      }
      // `Expr::Association` (the evaluated form) displays as `<|...|>`;
      // an unevaluated `Association[...]` FunctionCall (e.g. inside a
      // RuleDelayed RHS or Hold) keeps the long-form spelling, matching
      // Wolfram. Wolfram also propagates the long-form spelling to
      // any nested `Association` values inside the unevaluated form —
      // `Association[<|a -> v|>, {d}]` prints as
      // `Association[Association[a -> v], {d}]`, never the mixed
      // `Association[<|a -> v|>, {d}]`. Walk the args here and rewrite
      // every nested `Expr::Association` into the long-form FunctionCall
      // before emitting.
      if name == "Association" {
        let rewritten: Vec<Expr> =
          args.iter().map(rewrite_assoc_to_long_form).collect();
        let parts: Vec<String> = rewritten.iter().map(&fmt).collect();
        return format!("Association[{}]", parts.join(", "));
      }
      // Special case: Factorial[n] displays as n!, or (expr)! when the
      // argument is a Plus/Times or other operator-level expression so the
      // `!` suffix binds to the whole expression.
      if (name == "Factorial" || name == "Factorial2") && args.len() == 1 {
        let suffix = if name == "Factorial2" { "!!" } else { "!" };
        let arg_str = fmt(&args[0]);
        let needs_parens = match &args[0] {
          Expr::BinaryOp { op, .. } => !matches!(op, BinaryOperator::Power),
          Expr::UnaryOp {
            op: UnaryOperator::Minus,
            ..
          } => true,
          other if prints_as_not(other) => true,
          Expr::FunctionCall { name: n, args: a } => {
            (n == "Plus" || n == "Times" || n == "Minus") && a.len() >= 2
          }
          _ => false,
        };
        if needs_parens {
          return format!("({arg_str}){suffix}");
        }
        return format!("{arg_str}{suffix}");
      }
      if name == "Rule" && args.len() == 2 {
        return format!("{} -> {}", fmt(&args[0]), fmt(&args[1]));
      }
      if name == "TwoWayRule" && args.len() == 2 {
        return format!("{} <-> {}", fmt(&args[0]), fmt(&args[1]));
      }
      // Graph edges render as infix with their Wolfram private-use glyphs,
      // matching `wolframscript -code` (which prints U+F3D5/U+F3D4 directly).
      if name == "DirectedEdge" && args.len() == 2 {
        return format!("{} \u{F3D5} {}", fmt(&args[0]), fmt(&args[1]));
      }
      if name == "UndirectedEdge" && args.len() == 2 {
        return format!("{} \u{F3D4} {}", fmt(&args[0]), fmt(&args[1]));
      }
      if name == "RuleDelayed" && args.len() == 2 {
        // Parenthesize RHS if it's an assignment (Set/SetDelayed/Up*), so
        // that e.g. `Initialization :> (d[t_] := ...)` renders with the
        // parentheses required to disambiguate operator precedence.
        let rhs_str = fmt(&args[1]);
        let rhs_final = match &args[1] {
          Expr::FunctionCall { name: n, args: a }
            if matches!(
              n.as_str(),
              "Set" | "SetDelayed" | "UpSet" | "UpSetDelayed"
            ) && a.len() == 2 =>
          {
            format!("({rhs_str})")
          }
          _ => rhs_str,
        };
        return format!("{} :> {}", fmt(&args[0]), rhs_final);
      }
      if name == "Set" && args.len() == 2 {
        let rhs = fmt(&args[1]);
        let rhs = if assignment_rhs_needs_parens(&args[1]) {
          format!("({rhs})")
        } else {
          rhs
        };
        return format!("{} = {}", fmt(&args[0]), rhs);
      }
      if name == "SetDelayed" && args.len() == 2 {
        let rhs = fmt(&args[1]);
        let rhs = if assignment_rhs_needs_parens(&args[1]) {
          format!("({rhs})")
        } else {
          rhs
        };
        return format!("{} := {}", fmt(&args[0]), rhs);
      }
      if name == "UpSet" && args.len() == 2 {
        let rhs = fmt(&args[1]);
        let rhs = if assignment_rhs_needs_parens(&args[1]) {
          format!("({rhs})")
        } else {
          rhs
        };
        return format!("{} ^= {}", fmt(&args[0]), rhs);
      }
      if name == "UpSetDelayed" && args.len() == 2 {
        let rhs = fmt(&args[1]);
        let rhs = if assignment_rhs_needs_parens(&args[1]) {
          format!("({rhs})")
        } else {
          rhs
        };
        return format!("{} ^:= {}", fmt(&args[0]), rhs);
      }
      if name == "AddTo" && args.len() == 2 {
        return format!("{} += {}", fmt(&args[0]), fmt(&args[1]));
      }
      if name == "SubtractFrom" && args.len() == 2 {
        return format!("{} -= {}", fmt(&args[0]), fmt(&args[1]));
      }
      if name == "TimesBy" && args.len() == 2 {
        return format!("{} *= {}", fmt(&args[0]), fmt(&args[1]));
      }
      if name == "DivideBy" && args.len() == 2 {
        return format!("{} /= {}", fmt(&args[0]), fmt(&args[1]));
      }
      // Chained ++/-- variants follow Wolfram's display convention:
      //   * Postfix outers (Increment, Decrement) never parenthesize
      //     their inner expression — `Increment[Increment[a]]` prints
      //     as `a++++`, `Increment[PreIncrement[a]]` as `++a++`.
      //   * Prefix outers (PreIncrement, PreDecrement) always wrap
      //     inner ++/-- calls in parentheses for visual
      //     disambiguation — `PreIncrement[PreIncrement[a]]` prints
      //     as `++(++a)`, `PreIncrement[Increment[a]]` as `++(a++)`.
      // Round-trip parsing isn't preserved for mixed cases (Wolfram
      // also doesn't preserve them), but the display matches.
      let needs_inc_parens = |e: &Expr| -> bool {
        matches!(
          e,
          Expr::FunctionCall { name: n, args: a }
            if a.len() == 1 && (n == "Increment" || n == "Decrement"
              || n == "PreIncrement" || n == "PreDecrement")
        )
      };
      if name == "Increment" && args.len() == 1 {
        return format!("{}++", fmt(&args[0]));
      }
      if name == "Decrement" && args.len() == 1 {
        return format!("{}--", fmt(&args[0]));
      }
      if name == "PreIncrement" && args.len() == 1 {
        let inner = fmt(&args[0]);
        return if needs_inc_parens(&args[0]) {
          format!("++({inner})")
        } else {
          format!("++{inner}")
        };
      }
      if name == "PreDecrement" && args.len() == 1 {
        let inner = fmt(&args[0]);
        return if needs_inc_parens(&args[0]) {
          format!("--({inner})")
        } else {
          format!("--{inner}")
        };
      }
      if name == "Condition" && args.len() == 2 {
        // When the LHS is Plus (BinaryOp or FunctionCall), parenthesize
        // the last Pattern-like argument to avoid visual ambiguity:
        //   x + (y_) /; test  vs  x + y_ /; test
        // (Wolfram always adds these parens for readability)
        let is_pattern_like = |e: &Expr| -> bool {
          match e {
            Expr::Pattern { .. }
            | Expr::PatternOptional { .. }
            | Expr::PatternTest { .. } => true,
            Expr::FunctionCall { name: n, .. } => {
              n == "Pattern"
                || n == "Blank"
                || n == "BlankSequence"
                || n == "BlankNullSequence"
                || n == "Optional"
            }
            _ => false,
          }
        };
        // Handle BinaryOp Plus (the common case for parsed expressions)
        if let Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left,
          right,
        } = &args[0]
          && is_pattern_like(right)
        {
          return format!(
            "{} + ({}) /; {}",
            fmt(left),
            fmt(right),
            fmt(&args[1])
          );
        }
        // Handle FunctionCall Plus (normalized form)
        if let Expr::FunctionCall {
          name: ref pname,
          args: ref pargs,
        } = args[0]
          && pname == "Plus"
          && pargs.len() >= 2
        {
          let last = &pargs[pargs.len() - 1];
          if is_pattern_like(last) {
            // Re-format the Plus with the last arg parenthesized
            let mut plus_str = fmt(&pargs[0]);
            for (i, arg) in pargs.iter().enumerate().skip(1) {
              let s = fmt(arg);
              let s = if i == pargs.len() - 1 {
                format!("({s})")
              } else {
                s
              };
              if let Some(rest) = s.strip_prefix('-') {
                plus_str.push_str(" - ");
                plus_str.push_str(rest);
              } else if let Some(rest) = s.strip_prefix("(-") {
                plus_str.push_str(" - (");
                plus_str.push_str(rest);
              } else {
                plus_str.push_str(" + ");
                plus_str.push_str(&s);
              }
            }
            return format!("{} /; {}", plus_str, fmt(&args[1]));
          }
        }
        return format!("{} /; {}", fmt(&args[0]), fmt(&args[1]));
      }
      if name == "PatternTest" && args.len() == 2 {
        let pat = fmt(&args[0]);
        let test = fmt(&args[1]);
        // Atoms and lists print without parens, matching wolframscript:
        // `a?f`, `1?f`, `"s"?f`, `{1, 2}?f`. Everything else is wrapped —
        // `?` binds tighter than any infix operator, and wolframscript
        // brackets even a self-delimiting head: `(Except[0])?NumericQ`.
        let pat_atomic = pat == "_"
          || pat == "__"
          || pat == "___"
          || matches!(
            &args[0],
            Expr::Identifier(_)
              | Expr::Integer(_)
              | Expr::Real(_)
              | Expr::String(_)
              | Expr::Constant(_)
              | Expr::List(_)
          );
        if pat_atomic {
          return format!("{pat}?{test}");
        }
        return format!("({pat})?{test}");
      }
      // See the matching block above for the Wolfram display rule:
      // postfix outer never parenthesizes; prefix outer parenthesizes
      // any inner Increment/Decrement/PreIncrement/PreDecrement.
      let needs_inc_parens = |e: &Expr| -> bool {
        matches!(
          e,
          Expr::FunctionCall { name: n, args: a }
            if a.len() == 1 && (n == "Increment" || n == "Decrement"
              || n == "PreIncrement" || n == "PreDecrement")
        )
      };
      if name == "Increment" && args.len() == 1 {
        return format!("{}++", fmt(&args[0]));
      }
      if name == "Decrement" && args.len() == 1 {
        return format!("{}--", fmt(&args[0]));
      }
      if name == "PreIncrement" && args.len() == 1 {
        let inner = fmt(&args[0]);
        return if needs_inc_parens(&args[0]) {
          format!("++({inner})")
        } else {
          format!("++{inner}")
        };
      }
      if name == "PreDecrement" && args.len() == 1 {
        let inner = fmt(&args[0]);
        return if needs_inc_parens(&args[0]) {
          format!("--({inner})")
        } else {
          format!("--{inner}")
        };
      }
      if name == "Optional" && args.len() == 1 {
        // The `.` shorthand only exists for `Optional[Blank[]]`
        // (`_.`) and `Optional[Pattern[x, Blank[]]]` (`x_.`).
        // BlankSequence/BlankNullSequence inner patterns and
        // typed Blanks (`_Integer`, `x_Integer`) do not have a
        // valid `.` shorthand in wolframscript and must be
        // printed as `Optional[…]`.
        let inner_is_untyped_single_blank = match &args[0] {
          Expr::Pattern {
            blank_type: 1,
            head: None,
            ..
          } => true,
          Expr::FunctionCall {
            name: bn, args: ba, ..
          } if bn == "Blank" && ba.is_empty() => true,
          Expr::FunctionCall {
            name: pn,
            args: pargs,
            ..
          } if pn == "Pattern" && pargs.len() == 2 => {
            matches!(
              &pargs[1],
              Expr::FunctionCall { name: bn, args: ba, .. }
                if bn == "Blank" && ba.is_empty()
            )
          }
          _ => false,
        };
        if inner_is_untyped_single_blank {
          return format!("{}.", fmt(&args[0]));
        }
        let inner = fmt(&args[0]);
        return format!("Optional[{inner}]");
      }
      if name == "Optional" && args.len() == 2 {
        // Parenthesize the default when it's itself an Optional —
        // `Optional[Pattern[a, b], Optional[Pattern[c, d], e]]` prints as
        // `a:b:(c:d:e)`, not `a:b:c:d:e` (which would re-parse as a
        // single nested Optional).
        let needs_parens = matches!(
          &args[1],
          Expr::FunctionCall { name: dn, args: dargs }
            if dn == "Optional" && dargs.len() == 2
        );
        let body = fmt(&args[1]);
        if needs_parens {
          return format!("{}:({})", fmt(&args[0]), body);
        }
        return format!("{}:{}", fmt(&args[0]), body);
      }
      if name == "NonCommutativeMultiply" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        return parts.join("**");
      }
      // Special case: Minus[a, b, ...] with wrong arity displays with Unicode minus
      if name == "Minus" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        return parts.join(" \u{2212} ");
      }
      // Special case: ReverseElement[a, b] displays as a ∋ b
      if name == "ReverseElement" && args.len() == 2 {
        return format!("{} \u{220B} {}", fmt(&args[0]), fmt(&args[1]));
      }
      // Special case: Dot[a, b] displays as a . b (infix notation).
      // Operands that bind looser than `.` (Plus, Times, …) keep their
      // parens so e.g. `a . (b - c)` doesn't re-parse as `(a . b) - c`.
      if name == "Dot" && args.len() == 2 {
        let operand = |a: &Expr| {
          let s = fmt(a);
          if printed_infix_precedence(a).is_some_and(|p| p < 40) {
            format!("({s})")
          } else {
            s
          }
        };
        return format!("{} . {}", operand(&args[0]), operand(&args[1]));
      }
      if name == "Composition" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        return parts.join(" @* ");
      }
      if name == "RightComposition" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        return parts.join(" /* ");
      }
      // Special case: Therefore[a, b, ...] displays as a ∴ b ∴ ...
      if name == "Therefore" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        return parts.join(" \u{2234} ");
      }
      // Special case: Because[a, b, ...] displays as a ∵ b ∵ ...
      if name == "Because" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        return parts.join(" \u{2235} ");
      }
      // PlusMinus[x] displays as ±x, PlusMinus[a, b] displays as a ± b
      if name == "PlusMinus" {
        if args.len() == 1 {
          return format!("\u{00B1}{}", fmt(&args[0]));
        }
        if args.len() >= 2 {
          let parts: Vec<String> = args.iter().map(&fmt).collect();
          return parts.join(" \u{00B1} ");
        }
      }
      // MinusPlus[x] displays as ∓x, MinusPlus[a, b] displays as a ∓ b
      // (the sign-reversed partner of PlusMinus).
      if name == "MinusPlus" {
        if args.len() == 1 {
          return format!("\u{2213}{}", fmt(&args[0]));
        }
        if args.len() >= 2 {
          let parts: Vec<String> = args.iter().map(&fmt).collect();
          return parts.join(" \u{2213} ");
        }
      }
      // CircleTimes[a, b, ...] displays as a ⊗ b ⊗ ...
      if name == "CircleTimes" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        return parts.join(" \u{2297} ");
      }
      // TensorProduct[a, b, ...] displays with the U+F3DA operator (the glyph
      // wolframscript uses, distinct from CircleTimes' U+2297). Operands that
      // bind looser than the product (Plus, Times, …) are parenthesised:
      // TensorProduct[a + b, c] → (a + b) ⊗ c.
      if name == "TensorProduct" && args.len() >= 2 {
        let parts: Vec<String> = args
          .iter()
          .map(|a| {
            let s = fmt(a);
            if ring_operand_needs_parens(a) {
              format!("({s})")
            } else {
              s
            }
          })
          .collect();
        return parts.join(" \u{F3DA} ");
      }
      // CenterDot[a, b, ...] displays as a · b · ...
      if name == "CenterDot" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        return parts.join(" \u{00B7} ");
      }
      // CircleDot[a, b, ...] displays as a ⊙ b ⊙ ...
      // A nested CircleDot argument is parenthesized, matching
      // wolframscript: CircleDot[a, CircleDot[b, c]] -> a ⊙ (b ⊙ c)
      if name == "CircleDot" && args.len() >= 2 {
        let parts: Vec<String> = args
          .iter()
          .map(|a| {
            let s = fmt(a);
            if matches!(
              a,
              Expr::FunctionCall { name: n, args: ia }
                if n == "CircleDot" && ia.len() >= 2
            ) {
              format!("({s})")
            } else {
              s
            }
          })
          .collect();
        return parts.join(" \u{2299} ");
      }
      // Wedge[a, b, ...] displays as a ∧ b ∧ ...
      if name == "Wedge" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        return parts.join(" \u{22C0} ");
      }
      // Star[a, b, ...] displays as a ⋆ b ⋆ ...
      if name == "Star" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        return parts.join(" \u{22C6} ");
      }
      // Diamond[a, b, ...] displays as a ⋄ b ⋄ ...
      if name == "Diamond" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        return parts.join(" \u{22C4} ");
      }
      // Backslash[a, b, ...] displays as a ∖ b ∖ ...
      if name == "Backslash" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        return parts.join(" \u{2216} ");
      }
      // SmallCircle[a, b, ...] displays as a ∘ b ∘ ...
      if name == "SmallCircle" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        return parts.join(" \u{2218} ");
      }
      // Vee[a, b, ...] displays as a ⋁ b ⋁ ...
      if name == "Vee" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        return parts.join(" \u{22C1} ");
      }
      // Tilde[a, b, ...] displays as a ∼ b ∼ ...
      if name == "Tilde" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        return parts.join(" \u{223C} ");
      }
      // Del[f] displays as ∇f
      if name == "Del" && args.len() == 1 {
        return format!("\u{2207}{}", fmt(&args[0]));
      }
      // CirclePlus[a, b, ...] displays as a ⊕ b ⊕ ...
      if name == "CirclePlus" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        return parts.join(" \u{2295} ");
      }
      // CircleMinus[a, b, ...] displays as a ⊖ b ⊖ ...
      if name == "CircleMinus" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        return parts.join(" \u{2296} ");
      }
      // Subset[a, b] displays as a ⊂ b
      if name == "Subset" && args.len() == 2 {
        return format!("{} \u{2282} {}", fmt(&args[0]), fmt(&args[1]));
      }
      // LeftArrow[a, b] displays as a ← b
      if name == "LeftArrow" && args.len() == 2 {
        return format!("{} \u{2190} {}", fmt(&args[0]), fmt(&args[1]));
      }
      // DotEqual[a, b] displays as a ≐ b
      if name == "DotEqual" && args.len() == 2 {
        return format!("{} \u{2250} {}", fmt(&args[0]), fmt(&args[1]));
      }
      // AngleBracket[a, b, ...] displays as 〈a, b, ...〉
      if name == "AngleBracket" && !args.is_empty() {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        return format!("\u{2329} {} \u{232A}", parts.join(", "));
      }
      // Special case: Or[a, b, ...] displays as a || b || ...
      // Wolfram wraps And subterms in parens: (a && b) || (c && d)
      if name == "Or" && args.len() >= 2 {
        let parts: Vec<String> = args
          .iter()
          .map(|arg| {
            let s = fmt(arg);
            let is_and = matches!(
              arg,
              Expr::BinaryOp {
                op: BinaryOperator::And,
                ..
              }
            ) || matches!(arg, Expr::FunctionCall { name, .. } if name == "And");
            if is_and {
              format!("({s})")
            } else {
              s
            }
          })
          .collect();
        return parts.join(" || ");
      }
      // Special case: And[a, b, ...] displays as a && b && ...
      // Wolfram wraps Or subterms in parens: (a || b) && (a || c)
      if name == "And" && args.len() >= 2 {
        let parts: Vec<String> = args
          .iter()
          .map(|arg| {
            let s = fmt(arg);
            let is_or = matches!(
              arg,
              Expr::BinaryOp {
                op: BinaryOperator::Or,
                ..
              }
            ) || matches!(arg, Expr::FunctionCall { name, .. } if name == "Or");
            if is_or { format!("({s})") } else { s }
          })
          .collect();
        return parts.join(" && ");
      }
      // Equivalent[a, b, …] renders as the infix `a ⧦ b ⧦ c` (U+29E6) in
      // OutputForm, but as the functional `Equivalent[a, b, c]` in InputForm,
      // matching wolframscript.
      if name == "Equivalent" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        if is_output {
          return parts.join(" \u{29e6} ");
        }
        return format!("Equivalent[{}]", parts.join(", "));
      }
      // Special case: Alternatives[a, b, ...] displays as a | b | ...
      if name == "Alternatives" && args.len() >= 2 {
        let parts: Vec<String> = args
          .iter()
          .map(|a| {
            // Wrap nested Alternatives in parentheses
            if let Expr::FunctionCall {
              name: inner_name,
              args: inner_args,
            } = a
              && inner_name == "Alternatives"
              && inner_args.len() >= 2
            {
              return format!("({})", fmt(a));
            }
            fmt(a)
          })
          .collect();
        return parts.join(" | ");
      }
      // OutputForm: Entity[type, name] strips string quotes (matching wolframscript)
      if is_output && name == "Entity" {
        let parts: Vec<String> = args.iter().map(&fmt).collect();
        return format!("Entity[{}]", parts.join(", "));
      }
      // `Out[-k]` for k > 0 displays as the `%` shortcut — `Out[-1]` is
      // `%`, `Out[-2]` is `%%`, etc. Negative indices only show up here
      // when wrapped in a held context (Hold/HoldComplete/HoldPattern);
      // the standalone evaluator already collapses non-positive Out
      // arguments to `Out[0]`. `Out[0]` and positive indices render as
      // the literal `Out[k]` form, matching wolframscript.
      if name == "Out"
        && args.len() == 1
        && let Expr::Integer(n) = &args[0]
        && *n < 0
      {
        let count = (-*n) as usize;
        return "%".repeat(count);
      }
      // Notebook front-end only: Defer[expr] displays its argument without
      // the wrapper. wolframscript's script mode (and Woxi's CLI) print the
      // wrapper verbatim as `Defer[1 + 1]`, so only strip it in visual mode.
      // HoldForm is *not* stripped — wolframscript prints `Hold[HoldForm[1+2]]`
      // as `Hold[HoldForm[1 + 2]]` and `Trace[1+2]` as
      // `{HoldForm[1 + 2], HoldForm[3]}`.
      if is_output
        && name == "Defer"
        && args.len() == 1
        && crate::is_visual_mode()
      {
        return fmt(&args[0]);
      }
      // StringForm at top level renders as the literal `StringForm[…, args]`
      // wrapper in both InputForm and OutputForm — Wolfram substitutes only
      // when the user explicitly calls ToString. Fall through to the
      // general FunctionCall renderer below.
      // `Column[{…}]` has no plain-text typeset form without a front-end:
      // wolframscript's script mode prints it verbatim as `Column[{…}]`.
      // (Visual contexts render it as an SVG earlier, via
      // `render_column_if_needed`.) Fall through to the FunctionCall renderer.
      // OutputForm-only: Row[{exprs...}] concatenates; Row[{exprs...}, sep]
      // joins with separator. Trailing option rules (Alignment, ImageSize, …)
      // only affect notebook typesetting, so they are ignored here. A Rule
      // in separator position is an option, not a separator; any non-Rule
      // extra argument keeps the expression in its full form (matching
      // wolframscript, e.g. Row[{1, 2}, "|", "x"]).
      if is_output
        && name == "Row"
        && let Some(Expr::List(items)) = args.first()
      {
        let (sep_expr, opt_args) = match args.get(1) {
          Some(a) if !is_rule_expr(a) => (Some(a), &args[2..]),
          _ => (None, &args[1..]),
        };
        if opt_args.iter().all(is_rule_expr) {
          let parts: Vec<String> = items.iter().map(&fmt).collect();
          if let Some(sep_expr) = sep_expr {
            // wolframscript prints `Row[{}, sep]` as `{}` (but `Row[{}]`
            // as nothing).
            if items.is_empty() {
              return "{}".to_string();
            }
            // The separator prints in its plain OutputForm — wolframscript
            // shows even Spacer[w] literally (`aSpacer[7]b`) in script
            // mode; only visual contexts render it as a pixel gap.
            return parts.join(&fmt(sep_expr));
          }
          return parts.concat();
        }
      }
      // Special case: Times displays as infix with *
      if name == "Times" && args.len() >= 2 {
        // Flatten nested Times (Times is Flat)
        // Also decompose BinaryOp::Divide into numerator + Power[denom, -1]
        // so that format_times_with_denominator can render e.g. Times[a, d/(c+b*d)]
        // as (a*d)/(c + b*d).
        let mut flat_args: Vec<Expr> = Vec::with_capacity(args.len());
        for a in args {
          flatten_times_recursive(a, &mut flat_args);
        }
        // Normalize a pure-imaginary integer Complex coefficient
        // (`Complex[0, n]` = `n*I`) into explicit `[n, I]` factors so the
        // shared imaginary-coefficient handling below renders it uniformly
        // (`(-I)*x`, `(2*I)*x`). `Complex[0, 1]` collapses to plain `I`.
        let flat_args: Vec<Expr> = flat_args
          .iter()
          .flat_map(|a| match a {
            Expr::FunctionCall { name: cn, args: ca }
              if cn == "Complex"
                && ca.len() == 2
                && matches!(&ca[0], Expr::Integer(0)) =>
            {
              match &ca[1] {
                Expr::Integer(1) => vec![Expr::Identifier("I".to_string())],
                Expr::Integer(n) => {
                  vec![Expr::Integer(*n), Expr::Identifier("I".to_string())]
                }
                _ => vec![a.clone()],
              }
            }
            other => vec![other.clone()],
          })
          .collect();
        let args = &flat_args;
        // Handle Times[Rational[n, d], denom_factor] — form-specific formatting
        if args.len() == 2
          && let Expr::FunctionCall {
            name: rname,
            args: rargs,
          } = &args[0]
          && rname == "Rational"
          && rargs.len() == 2
          && let Expr::Integer(n) = &rargs[0]
          && let Expr::Integer(d) = &rargs[1]
          && *d > 0
          && is_denominator_factor(&args[1])
        {
          let denom_form = denominator_form(&args[1]);
          let denom_str = fmt(&denom_form);
          // When numerator is -1, Wolfram separates the rational
          // from the power term: -1/d*1/base^exp (e.g. -1/2*1/x^2)
          if *n == -1 {
            let denom_needs_parens = matches!(&denom_form, Expr::FunctionCall { name, .. } if name == "Plus" || name == "Times")
              || matches!(
                &denom_form,
                Expr::BinaryOp {
                  op: BinaryOperator::Plus
                    | BinaryOperator::Minus
                    | BinaryOperator::Times,
                  ..
                }
              );
            let power_str = if denom_needs_parens {
              format!("1/({denom_str})")
            } else {
              format!("1/{denom_str}")
            };
            return format!("-1/{d}*{power_str}");
          }
          let inner_needs_parens = matches!(&denom_form, Expr::FunctionCall { name, .. } if name == "Plus" || name == "Times")
            || matches!(
              &denom_form,
              Expr::BinaryOp {
                op: BinaryOperator::Plus
                  | BinaryOperator::Minus
                  | BinaryOperator::Times,
                ..
              }
            );
          let denom_str_parened = if inner_needs_parens {
            format!("({denom_str})")
          } else {
            denom_str
          };
          if is_output {
            // OutputForm: combine Rational denominator with Power denominator
            if *n == 1 {
              return format!("1/({d}*{denom_str_parened})");
            }
            return format!("{n}/({d}*{denom_str_parened})");
          }
          // For other numerators, combine into n/(d*base^exp)
          let full_denom = if *d > 1 {
            format!("({d}*{denom_str_parened})")
          } else {
            denom_str_parened
          };
          if *n == 1 {
            return format!("1/{full_denom}");
          }
          return format!("{n}/{full_denom}");
        }
        // Handle Times[Rational[1, d], expr] as "expr/d" (2-arg)
        if args.len() == 2
          && let Expr::FunctionCall {
            name: rname,
            args: rargs,
          } = &args[0]
          && rname == "Rational"
          && rargs.len() == 2
          && matches!((&rargs[0], &rargs[1]), (Expr::Integer(1), Expr::Integer(d)) if *d > 0)
          && let Expr::Integer(d) = &rargs[1]
        {
          if is_output {
            // OutputForm wraps Plus/Minus inner in parens
            let inner = fmt(&args[1]);
            let inner_str = if matches!(&args[1], Expr::FunctionCall { name, .. } if name == "Plus")
              || matches!(
                &args[1],
                Expr::BinaryOp {
                  op: BinaryOperator::Plus | BinaryOperator::Minus,
                  ..
                }
              ) {
              format!("({inner})")
            } else {
              inner
            };
            return format!("{inner_str}/{d}");
          }
          // InputForm: wrap Plus in parens like OutputForm
          let inner = fmt(&args[1]);
          let inner_str = if matches!(&args[1], Expr::FunctionCall { name, .. } if name == "Plus")
            || matches!(
              &args[1],
              Expr::BinaryOp {
                op: BinaryOperator::Plus | BinaryOperator::Minus,
                ..
              }
            ) {
            format!("({inner})")
          } else {
            inner
          };
          return format!("{inner_str}/{d}");
        }
        // Handle Times[Rational[-1, d], Plus[t1, t2, ...]] as
        // "(-t1 - t2 - ...)/d" by negating each summand (matching
        // wolframscript: (-1/2)(a + b) renders as (-a - b)/2, not -1/2*a + b).
        if args.len() == 2
          && let Expr::FunctionCall {
            name: rname,
            args: rargs,
          } = &args[0]
          && rname == "Rational"
          && rargs.len() == 2
          && matches!(&rargs[0], Expr::Integer(-1))
          && let Expr::Integer(d) = &rargs[1]
          && *d > 0
          && let Expr::FunctionCall {
            name: pname,
            args: pargs,
          } = &args[1]
          && pname == "Plus"
        {
          // Negate one summand for display, collapsing a leading -1.
          let negate = |t: &Expr| -> Expr {
            match t {
              Expr::Integer(k) => Expr::Integer(-k),
              Expr::FunctionCall { name, args: ra }
                if name == "Rational" && ra.len() == 2 =>
              {
                if let Expr::Integer(k) = &ra[0] {
                  Expr::FunctionCall {
                    name: "Rational".to_string(),
                    args: vec![Expr::Integer(-k), ra[1].clone()].into(),
                  }
                } else {
                  Expr::FunctionCall {
                    name: "Times".to_string(),
                    args: vec![Expr::Integer(-1), t.clone()].into(),
                  }
                }
              }
              // -1 * rest → rest (double negation cancels)
              Expr::FunctionCall { name, args: fa }
                if name == "Times"
                  && fa.len() >= 2
                  && matches!(&fa[0], Expr::Integer(-1)) =>
              {
                if fa.len() == 2 {
                  fa[1].clone()
                } else {
                  unevaluated("Times", &fa[1..])
                }
              }
              Expr::UnaryOp {
                op: UnaryOperator::Minus,
                operand,
              } => (**operand).clone(),
              _ => Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![Expr::Integer(-1), t.clone()].into(),
              },
            }
          };
          let neg_terms: Vec<Expr> = pargs.iter().map(negate).collect();
          let neg_plus = Expr::FunctionCall {
            name: "Plus".to_string(),
            args: neg_terms.into(),
          };
          return format!("({})/{}", fmt(&neg_plus), d);
        }
        // Handle Times[Rational[n, d], expr] as "(n*expr)/d" (Wolfram convention)
        if args.len() == 2
          && let Expr::FunctionCall {
            name: rname,
            args: rargs,
          } = &args[0]
          && rname == "Rational"
          && rargs.len() == 2
          && let Expr::Integer(n) = &rargs[0]
          && let Expr::Integer(d) = &rargs[1]
          && *n != 1
          && *n != -1
          && *d > 0
        {
          let inner = fmt(&args[1]);
          let inner_str = if matches!(&args[1], Expr::FunctionCall { name, .. } if name == "Plus")
            || matches!(
              &args[1],
              Expr::BinaryOp {
                op: BinaryOperator::Plus | BinaryOperator::Minus,
                ..
              }
            ) {
            format!("({inner})")
          } else {
            inner
          };
          return format!("({n}*{inner_str})/{d}");
        }
        // Handle Times[Rational[1, d], e1, e2, ...] as "(e1*e2*...)/d" (3+ args)
        // Only when no factor is I (imaginary unit, which pairs with the coefficient)
        if args.len() >= 3
          && let Expr::FunctionCall {
            name: rname,
            args: rargs,
          } = &args[0]
          && rname == "Rational"
          && rargs.len() == 2
          && matches!((&rargs[0], &rargs[1]), (Expr::Integer(1), Expr::Integer(d)) if *d > 0)
          && let Expr::Integer(d) = &rargs[1]
          && !args[1..].iter().any(is_denominator_factor)
          && !args[1..]
            .iter()
            .any(|a| matches!(a, Expr::Identifier(s) if s == "I"))
        {
          let fmt_factor = |a: &Expr| -> String {
            let s = fmt(a);
            if matches!(a, Expr::FunctionCall { name, .. } if name == "Plus")
              || matches!(
                a,
                Expr::BinaryOp {
                  op: BinaryOperator::Plus | BinaryOperator::Minus,
                  ..
                }
              )
            {
              format!("({s})")
            } else {
              s
            }
          };
          let rest: Vec<String> = args[1..].iter().map(fmt_factor).collect();
          let numer = rest.join("*");
          return format!("({numer})/{d}");
        }
        // Handle Times[Rational[n, d], e1, e2, ...] as "(n*e1*e2*...)/d"
        // for |n| > 1, d > 1 (Wolfram convention for general rationals).
        if args.len() >= 3
          && let Expr::FunctionCall {
            name: rname,
            args: rargs,
          } = &args[0]
          && rname == "Rational"
          && rargs.len() == 2
          && let (Expr::Integer(n), Expr::Integer(d)) = (&rargs[0], &rargs[1])
          && n.abs() > 1
          && *d > 1
          && !args[1..].iter().any(is_denominator_factor)
          && !args[1..]
            .iter()
            .any(|a| matches!(a, Expr::Identifier(s) if s == "I"))
        {
          let fmt_factor = |a: &Expr| -> String {
            let s = fmt(a);
            if matches!(a, Expr::FunctionCall { name, .. } if name == "Plus")
              || matches!(
                a,
                Expr::BinaryOp {
                  op: BinaryOperator::Plus | BinaryOperator::Minus,
                  ..
                }
              )
            {
              format!("({s})")
            } else {
              s
            }
          };
          let rest: Vec<String> = args[1..].iter().map(fmt_factor).collect();
          let numer = rest.join("*");
          return format!("({n}*{numer})/{d}");
        }
        // Handle Times[Rational[-1, d], e1, e2, ...] as "-1/d*(e1*e2*...)"
        // (Wolfram convention for negative reciprocal coefficients).
        if args.len() >= 2
          && let Expr::FunctionCall {
            name: rname,
            args: rargs,
          } = &args[0]
          && rname == "Rational"
          && rargs.len() == 2
          && matches!((&rargs[0], &rargs[1]), (Expr::Integer(-1), Expr::Integer(d)) if *d > 1)
          && let Expr::Integer(d) = &rargs[1]
          && !args[1..]
            .iter()
            .any(|a| matches!(a, Expr::Identifier(s) if s == "I"))
        {
          let rest = if args.len() == 2 {
            args[1].clone()
          } else {
            unevaluated("Times", &args[1..])
          };
          let rest_str = fmt(&rest);
          // If rest is a multi-factor Times (no denominator), wrap in parens.
          let needs_parens = if let Expr::FunctionCall {
            name: rname2,
            args: rargs2,
          } = &rest
          {
            rname2 == "Times"
              && rargs2.len() >= 2
              && !rargs2.iter().any(is_denominator_factor)
          } else {
            false
          };
          let formatted = if needs_parens {
            format!("({rest_str})")
          } else {
            rest_str
          };
          return format!("-1/{d}*{formatted}");
        }
        // Handle Times[-1, x, ...] as "-x*..."
        // In InputForm, `Times[-1, I, sym…]` is rendered by the imaginary-
        // coefficient block below as `(-I)*sym…` (parenthesised), so skip the
        // bare `-…` negation path here when an imaginary unit AND a symbolic
        // factor are both present. Pure imaginary coefficients (`-I`, with no
        // symbolic factor) keep the `-I` form, and OutputForm always does.
        let is_i_unit_factor = |a: &Expr| {
          matches!(a, Expr::Identifier(s) | Expr::Constant(s) if s == "I")
            || matches!(a, Expr::FunctionCall { name: cn, args: ca }
              if cn == "Complex" && ca.len() == 2
                && matches!((&ca[0], &ca[1]), (Expr::Integer(0), Expr::Integer(1))))
        };
        let leading_neg1_has_i = in_true_input_form()
          && args.iter().any(is_i_unit_factor)
          && args.iter().any(|a| {
            !matches!(a, Expr::Integer(_) | Expr::Real(_))
              && !is_i_unit_factor(a)
              && !matches!(a, Expr::FunctionCall { name, .. } if name == "Rational")
          });
        if matches!(&args[0], Expr::Integer(-1)) && !leading_neg1_has_i {
          // Special case: if Infinity is among the factors, wolframscript
          // merges the -1 into it and prints `-Infinity` inline, e.g.
          // Times[-1, a, b, Infinity] → "a*b*-Infinity" (not "-(a*b*Infinity)").
          let infinity_pos = args[1..]
            .iter()
            .position(|a| matches!(a, Expr::Identifier(s) if s == "Infinity"));
          if let Some(idx) = infinity_pos {
            let abs_pos = idx + 1; // position inside `args`
            let mut merged: Vec<Expr> = Vec::with_capacity(args.len() - 1);
            for (i, a) in args.iter().enumerate() {
              if i == 0 || i == abs_pos {
                continue;
              }
              merged.push(a.clone());
            }
            let rest_str = merged
              .iter()
              .map(|a| {
                let s = fmt(a);
                if matches!(a, Expr::FunctionCall { name, .. } if name == "Plus")
                  || matches!(
                    a,
                    Expr::BinaryOp {
                      op: BinaryOperator::Plus | BinaryOperator::Minus,
                      ..
                    }
                  )
                {
                  format!("({s})")
                } else {
                  s
                }
              })
              .collect::<Vec<_>>()
              .join("*");
            return if merged.is_empty() {
              "-Infinity".to_string()
            } else {
              format!("{rest_str}*-Infinity")
            };
          }
          // If the rest is a single Power[symbolic_base, negative_int], use -base^(-n)
          // notation instead of -(1/base^n), matching wolframscript output
          let is_single_symbolic_neg_power =
            args.len() == 2 && is_symbolic_neg_int_power(&args[1]);
          // Check if the rest of the factors need denominator formatting
          if !is_single_symbolic_neg_power
            && let Some(frac) =
              format_times_with_denominator(&args[1..], fmt_fn)
          {
            // wolframscript keeps a -I coefficient inline in quotients:
            // (-I*(-1 + E^(I*t)))/t, not -((I*(-1 + E^(I*t)))/t)
            if matches!(&args[1], Expr::Identifier(s) if s == "I")
              && frac.starts_with("(I*")
            {
              return format!("(-I*{}", &frac[3..]);
            }
            return format!("-({frac})");
          }
          let rest = args[1..]
            .iter()
            .map(|a| {
              let s = fmt(a);
              if matches!(a, Expr::FunctionCall { name, .. } if name == "Plus")
                || matches!(
                  a,
                  Expr::BinaryOp {
                    op: BinaryOperator::Plus | BinaryOperator::Minus,
                    ..
                  }
                )
              {
                format!("({s})")
              } else {
                s
              }
            })
            .collect::<Vec<_>>()
            .join("*");
          // Wolfram wraps negated products in parens when the factors are
          // all symbolic: -(a*b), -(x*Cos[x]), but -I*a, -2*a*b without parens.
          // The rule: parens needed when coefficient is -1 and ALL remaining
          // factors are symbolic (not I, numeric literals, or Rational).
          let rest_factors = &args[1..];
          let all_symbolic_factors = rest_factors.iter().all(|a| {
            !matches!(a, Expr::Integer(_) | Expr::Real(_))
              && !matches!(a, Expr::Identifier(n) | Expr::Constant(n) if n == "I")
              && !matches!(a, Expr::FunctionCall { name, .. } if name == "Rational")
          });
          let needs_neg_parens = (rest_factors.len() >= 2
            && all_symbolic_factors)
            || (args.len() == 2
              && (matches!(&args[1], Expr::FunctionCall { name, .. } if name == "Times")
                || matches!(
                  &args[1],
                  Expr::BinaryOp {
                    op: BinaryOperator::Times | BinaryOperator::Divide,
                    ..
                  }
                )
                // Anonymous functions need parens so the trailing `&`
                // doesn't get pulled into the unary-minus expression:
                // `-(0 & )` rather than `-0 & ` (which would parse as
                // `Function[-0]`).
                || matches!(&args[1], Expr::Function { .. })));
          if needs_neg_parens {
            return format!("-({rest})");
          }
          return format!("-{rest}");
        }
        // Complex number grouping: Times containing I with non-numeric remaining factors
        // e.g. Times[2, I, Sqrt[3]] → (2*I)*Sqrt[3], Times[Rational[1,2], I, Pi] → (I/2)*Pi
        // The imaginary unit can appear as either Identifier("I") or
        // Complex[0, 1] in the underlying AST.
        let is_i_unit = |a: &Expr| match a {
          Expr::Identifier(n) | Expr::Constant(n) => n == "I",
          Expr::FunctionCall { name: cn, args: ca } => {
            cn == "Complex"
              && ca.len() == 2
              && matches!(
                (&ca[0], &ca[1]),
                (Expr::Integer(0), Expr::Integer(1))
              )
          }
          _ => false,
        };
        let has_imaginary = args.iter().any(is_i_unit);
        if has_imaginary {
          let mut numeric_factors: Vec<&Expr> = Vec::new();
          let mut symbolic_factors: Vec<&Expr> = Vec::new();
          for arg in args {
            match arg {
              Expr::Integer(_) | Expr::Real(_) => numeric_factors.push(arg),
              _ if is_i_unit(arg) => {}
              Expr::FunctionCall { name: rn, .. } if rn == "Rational" => {
                numeric_factors.push(arg);
              }
              _ => symbolic_factors.push(arg),
            }
          }
          // When a BigFloat is present alongside the imaginary unit,
          // Wolfram does NOT use the `I*<rest>` grouping — it just emits
          // the canonical Times order, e.g. `<BigFloat>*I`. Fall through
          // to the default Times formatter so the BigFloat sorts before
          // I (its `times_factor_subpriority` is -3).
          let any_bigfloat = symbolic_factors
            .iter()
            .any(|a| matches!(a, Expr::BigFloat(_, _) | Expr::BigInteger(_)));
          if !symbolic_factors.is_empty() && !any_bigfloat {
            let i_part_opt: Option<String> = if numeric_factors.is_empty() {
              Some("I".to_string())
            } else if numeric_factors.len() == 1 {
              match numeric_factors[0] {
                Expr::Integer(1) => Some("I".to_string()),
                Expr::Integer(-1) => {
                  // InputForm: "(-I)", OutputForm: "-I"
                  if in_true_input_form() {
                    Some("(-I)".to_string())
                  } else {
                    Some("-I".to_string())
                  }
                }
                Expr::Integer(n) => Some(format!("({n}*I)")),
                Expr::FunctionCall { name: rn, args: ra }
                  if rn == "Rational" && ra.len() == 2 =>
                {
                  if let (Expr::Integer(num), Expr::Integer(den)) =
                    (&ra[0], &ra[1])
                  {
                    // InputForm: wolframscript parenthesises the imaginary
                    // coefficient (`(I/2)*x`). OutputForm uses the bare
                    // `I/2*x` 1D form. Negative rationals stay parenthesized
                    // (`(-1/2*I)*x`) so the leading `-` is not pulled out as
                    // unary minus by the parser.
                    if *num == 1 {
                      if in_true_input_form() {
                        Some(format!("(I/{den})"))
                      } else {
                        Some(format!("I/{den}"))
                      }
                    } else if *num == -1 {
                      Some(format!("(-1/{den}*I)"))
                    } else {
                      Some(format!("(({num}*I)/{den})"))
                    }
                  } else {
                    None
                  }
                }
                _ => None,
              }
            } else {
              None
            };
            if let Some(i_part) = i_part_opt {
              // If the symbolic side contains a denominator factor
              // (Power[base, negative]) split it out as `(<i_part>*<numer>)/<denom>`,
              // matching wolframscript's `Times[I, …, Power[d, -1]]` form. The
              // imaginary coefficient is folded into the numerator and the
              // whole numerator product is parenthesised when it has more than
              // one factor: `(I*y)/x`, `((-I)*(a + b))/c`, `I/x`.
              if symbolic_factors.iter().any(|a| is_denominator_factor(a)) {
                let numer_factors: Vec<&Expr> = symbolic_factors
                  .iter()
                  .copied()
                  .filter(|a| !is_denominator_factor(a))
                  .collect();
                let denom_exprs: Vec<Expr> = symbolic_factors
                  .iter()
                  .copied()
                  .filter(|a| is_denominator_factor(a))
                  .map(denominator_form)
                  .collect();
                let fmt_factor = |a: &Expr| -> String {
                  let s = fmt(a);
                  if matches!(a, Expr::FunctionCall { name, .. } if name == "Plus")
                    || matches!(
                      a,
                      Expr::BinaryOp {
                        op: BinaryOperator::Plus | BinaryOperator::Minus,
                        ..
                      }
                    )
                  {
                    format!("({s})")
                  } else {
                    s
                  }
                };
                let numer_str = if numer_factors.is_empty() {
                  i_part.clone()
                } else {
                  let mut parts = vec![i_part.clone()];
                  parts.extend(numer_factors.iter().map(|a| fmt_factor(a)));
                  format!("({})", parts.join("*"))
                };
                let needs_parens = |e: &Expr| -> bool {
                  matches!(e, Expr::FunctionCall { name, .. } if name == "Plus" || name == "Times")
                    || matches!(
                      e,
                      Expr::BinaryOp {
                        op: BinaryOperator::Plus
                          | BinaryOperator::Minus
                          | BinaryOperator::Times,
                        ..
                      }
                    )
                };
                let denom_str = if denom_exprs.len() == 1 {
                  let s = fmt(&denom_exprs[0]);
                  if needs_parens(&denom_exprs[0]) {
                    format!("({s})")
                  } else {
                    s
                  }
                } else {
                  let inner = denom_exprs
                    .iter()
                    .map(|a| {
                      let s = fmt(a);
                      if needs_parens(a) { format!("({s})") } else { s }
                    })
                    .collect::<Vec<_>>()
                    .join("*");
                  format!("({inner})")
                };
                return format!("{numer_str}/{denom_str}");
              }
              let rest: Vec<String> = symbolic_factors
                .iter()
                .map(|a| {
                  let s = fmt(a);
                  if matches!(a, Expr::FunctionCall { name, .. } if name == "Plus")
                    || matches!(
                      a,
                      Expr::BinaryOp {
                        op: BinaryOperator::Plus | BinaryOperator::Minus,
                        ..
                      }
                    )
                  {
                    format!("({s})")
                  } else {
                    s
                  }
                })
                .collect();
              return format!("{}*{}", i_part, rest.join("*"));
            }
          }
        }
        // Check for denominator factors (negative exponents) to format as fraction
        if let Some(frac) = format_times_with_denominator(args, fmt_fn) {
          return frac;
        }
        // Default Times join
        return args
          .iter()
          .map(|a| {
            let s = fmt(a);
            if matches!(a, Expr::FunctionCall { name, .. } if name == "Plus")
              || matches!(
                a,
                Expr::BinaryOp {
                  op: BinaryOperator::Plus | BinaryOperator::Minus,
                  ..
                }
              )
              // Complex numbers (except plain I = Complex[0,1]) need parens in Times (InputForm only)
              || (!is_output && matches!(a, Expr::FunctionCall { name, args } if name == "Complex"
                && args.len() == 2
                && !matches!((&args[0], &args[1]), (Expr::Integer(0), Expr::Integer(1)))))
              // Pattern expressions need parens in Times (Wolfram convention)
              || matches!(a, Expr::Pattern { .. } | Expr::PatternOptional { .. } | Expr::PatternTest { .. })
            {
              format!("({s})")
            } else {
              s
            }
          })
          .collect::<Vec<_>>()
          .join("*");
      }
      // Special case: Plus displays as infix with + (with spaces)
      if name == "Plus" && args.len() >= 2 {
        // Pattern/PatternOptional/PatternTest terms in Plus are
        // parenthesised in Wolfram (e.g. `B[...] + (a_.)`, never
        // `B[...] + a_.`). The `_` inside the term is otherwise
        // syntactically ambiguous next to an additive operator.
        let needs_pattern_parens = |a: &Expr| -> bool {
          matches!(
            a,
            Expr::Pattern { .. }
              | Expr::PatternOptional { .. }
              | Expr::PatternTest { .. }
          )
        };
        // `Condition[expr, test]` — printed with `/;`, which binds looser
        // than `+`. Wrap the whole thing so `p + (1 /; 2 > 1)` doesn't
        // surface as `p + 1 /; 2 > 1` (which would parse as
        // `Condition[p + 1, 2 > 1]`).
        let needs_condition_parens = |a: &Expr| -> bool {
          matches!(
            a,
            Expr::FunctionCall { name, args } if name == "Condition" && args.len() == 2
          )
        };
        // Nested Plus inside Plus needs parens to preserve grouping in held
        // expressions: `Plus[1, Plus[1, 1]]` → `1 + (1 + 1)`. Without parens
        // the displayed `1 + 1 + 1` would parse as the flat `Plus[1, 1, 1]`.
        let needs_nested_plus_parens = |a: &Expr| -> bool {
          matches!(
            a,
            Expr::FunctionCall { name, args } if name == "Plus" && args.len() >= 2
          ) || matches!(
            a,
            Expr::BinaryOp {
              op: BinaryOperator::Plus,
              ..
            }
          )
        };
        let fmt_plus_term = |a: &Expr| -> String {
          let s = fmt(a);
          if needs_pattern_parens(a)
            || needs_condition_parens(a)
            || needs_nested_plus_parens(a)
          {
            format!("({s})")
          } else {
            s
          }
        };
        let mut result = fmt_plus_term(&args[0]);
        for arg in args.iter().skip(1) {
          if let Expr::UnaryOp {
            op: UnaryOperator::Minus,
            operand,
          } = arg
          {
            result.push_str(" - ");
            result.push_str(&fmt_plus_term(operand));
          } else if let Expr::BinaryOp {
            op: BinaryOperator::Times,
            ..
          } = arg
          {
            // Flatten nested BinaryOp Times so we can inspect all factors,
            // not just the outermost left. An Expand result like
            // `Times[Rational[-1,3], Times[Cos[x], Sin[x]^2]]` was missing
            // this branch and printing as `+ -1/3*Cos[x]*Sin[x]^2` when it
            // should appear as `- (Cos[x]*Sin[x]^2)/3` inside a Plus.
            let mut factor_refs: Vec<&Expr> = Vec::new();
            flatten_binary_times(arg, &mut factor_refs);
            // Find a numeric factor whose negation flips the term's sign.
            let neg_idx = factor_refs.iter().position(|f| match f {
              Expr::Integer(n) if *n < 0 => true,
              Expr::BigInteger(n) if n.sign() == Sign::Minus => true,
              Expr::Real(v) if *v < 0.0 => true,
              Expr::FunctionCall { name, args }
                if name == "Rational"
                  && args.len() == 2
                  && matches!(&args[0], Expr::Integer(n) if *n < 0) =>
              {
                true
              }
              _ => false,
            });
            // Find a Rational factor whose denominator triggers fraction
            // formatting (positive Rational[n, d] with d > 1, anywhere in
            // the flattened factors). Used when there's no negation: e.g.
            // `Times[x, Rational[1, 3], Sin[x]^3]` should print as
            // `(x*Sin[x]^3)/3`, not `x*1/3*Sin[x]^3`.
            let pos_rat_idx = if neg_idx.is_some() {
              None
            } else {
              factor_refs.iter().position(|f| matches!(f, Expr::FunctionCall { name, args } if name == "Rational"
                    && args.len() == 2
                    && matches!(&args[0], Expr::Integer(n) if *n > 0)
                    && matches!(&args[1], Expr::Integer(d) if d.abs() > 1)))
            };
            if let Some(idx) = neg_idx {
              let pos_factor = match factor_refs[idx] {
                Expr::Integer(n) => {
                  if *n == -1 {
                    None // drop the -1 entirely
                  } else {
                    Some(Expr::Integer(-n))
                  }
                }
                Expr::BigInteger(n) => Some(Expr::BigInteger(-n)),
                Expr::Real(v) => Some(Expr::Real(-v)),
                Expr::FunctionCall { name, args }
                  if name == "Rational" && args.len() == 2 =>
                {
                  if let Expr::Integer(n) = &args[0] {
                    if *n == -1 {
                      Some(Expr::FunctionCall {
                        name: "Rational".to_string(),
                        args: vec![Expr::Integer(1), args[1].clone()].into(),
                      })
                    } else {
                      Some(Expr::FunctionCall {
                        name: "Rational".to_string(),
                        args: vec![Expr::Integer(-n), args[1].clone()].into(),
                      })
                    }
                  } else {
                    Some(factor_refs[idx].clone())
                  }
                }
                _ => Some(factor_refs[idx].clone()),
              };
              let mut new_args: Vec<Expr> = Vec::new();
              if let Some(f) = pos_factor {
                new_args.push(f);
              }
              for (i, f) in factor_refs.iter().enumerate() {
                if i != idx {
                  new_args.push((*f).clone());
                }
              }
              let pos_term = if new_args.len() == 1 {
                new_args.into_iter().next().unwrap()
              } else {
                Expr::FunctionCall {
                  name: "Times".to_string(),
                  args: new_args.into(),
                }
              };
              result.push_str(" - ");
              result.push_str(&fmt(&pos_term));
            } else if let Some(idx) = pos_rat_idx {
              // Reorder so the Rational is leading, then build a flat
              // FunctionCall Times so the standalone Times printer
              // recognises the fraction.
              let mut new_args: Vec<Expr> = Vec::new();
              new_args.push(factor_refs[idx].clone());
              for (i, f) in factor_refs.iter().enumerate() {
                if i != idx {
                  new_args.push((*f).clone());
                }
              }
              let canonical = Expr::FunctionCall {
                name: "Times".to_string(),
                args: new_args.into(),
              };
              result.push_str(" + ");
              result.push_str(&fmt(&canonical));
            } else {
              result.push_str(" + ");
              result.push_str(&fmt(arg));
            }
          } else if let Expr::FunctionCall {
            name: fn_name,
            args: fn_args,
          } = arg
          {
            if fn_name == "Times" && fn_args.len() >= 2 {
              // Check if leading factor is negative
              let neg_coeff = match &fn_args[0] {
                Expr::Integer(n) if *n < 0 => Some(if *n == -1 {
                  None // coefficient of -1 means just negate
                } else {
                  Some(Expr::Integer(-n))
                }),
                Expr::BigInteger(n) if n.sign() == Sign::Minus => {
                  Some(Some(Expr::BigInteger(-n)))
                }
                Expr::Real(r) if *r < 0.0 => Some(Some(Expr::Real(-r))),
                Expr::FunctionCall { name: rn, args: ra }
                  if rn == "Rational"
                    && ra.len() == 2
                    && matches!(&ra[0], Expr::Integer(n) if *n < 0) =>
                {
                  if let Expr::Integer(n) = &ra[0] {
                    if *n == -1 {
                      // Rational[-1, d] → just Rational[1, d] = 1/d
                      Some(Some(Expr::FunctionCall {
                        name: "Rational".to_string(),
                        args: vec![Expr::Integer(1), ra[1].clone()].into(),
                      }))
                    } else {
                      Some(Some(Expr::FunctionCall {
                        name: "Rational".to_string(),
                        args: vec![Expr::Integer(-n), ra[1].clone()].into(),
                      }))
                    }
                  } else {
                    None
                  }
                }
                _ => None,
              };
              if let Some(pos_coeff) = neg_coeff {
                result.push_str(" - ");
                let pos_term = match pos_coeff {
                  None => {
                    // Times[-1, rest...] → rest
                    let pos_args = fn_args[1..].to_vec();
                    if pos_args.len() == 1 {
                      pos_args[0].clone()
                    } else {
                      Expr::FunctionCall {
                        name: "Times".to_string(),
                        args: pos_args.into(),
                      }
                    }
                  }
                  Some(new_coeff) => {
                    let mut new_args = vec![new_coeff];
                    new_args.extend_from_slice(&fn_args[1..]);
                    if new_args.len() == 1 {
                      new_args[0].clone()
                    } else {
                      Expr::FunctionCall {
                        name: "Times".to_string(),
                        args: new_args.into(),
                      }
                    }
                  }
                };
                // Wolfram parenthesises a subtracted prefix MinusPlus so the
                // leading `-` doesn't fuse with the `∓`: a - MinusPlus[b] → a - (∓b).
                let pos_term_str = fmt_plus_term(&pos_term);
                let pos_term_str = if matches!(
                  &pos_term,
                  Expr::FunctionCall { name: n, args: a }
                    if n == "MinusPlus" && a.len() == 1
                ) {
                  format!("({pos_term_str})")
                } else {
                  pos_term_str
                };
                result.push_str(&pos_term_str);
              } else {
                result.push_str(" + ");
                result.push_str(&fmt(arg));
              }
            } else {
              result.push_str(" + ");
              result.push_str(&fmt_plus_term(arg));
            }
          } else if let Expr::Integer(n) = arg {
            if *n < 0 {
              result.push_str(" - ");
              result.push_str(&fmt(&Expr::Integer(-n)));
            } else {
              result.push_str(" + ");
              result.push_str(&fmt(arg));
            }
          } else if let Expr::BigInteger(n) = arg {
            if n.sign() == Sign::Minus {
              result.push_str(" - ");
              result.push_str(&fmt(&Expr::BigInteger(-n)));
            } else {
              result.push_str(" + ");
              result.push_str(&fmt(arg));
            }
          } else if !is_output {
            // InputForm: Fallback check if the rendered form starts with "-"
            let s = fmt_plus_term(arg);
            if let Some(rest) = s.strip_prefix('-') {
              result.push_str(" - ");
              result.push_str(rest);
            } else {
              result.push_str(" + ");
              result.push_str(&s);
            }
          } else {
            // OutputForm: no starts_with('-') check
            result.push_str(" + ");
            result.push_str(&fmt_plus_term(arg));
          }
        }
        return result;
      }
      // Special case: Power displays as infix with ^ (no spaces)
      if name == "Power" && args.len() == 2 {
        // Power[x, Rational[1, 2]] → Sqrt[x]
        if let Some(sqrt_arg) = crate::functions::is_sqrt(expr) {
          return format!("Sqrt[{}]", fmt(sqrt_arg));
        }
        // Power[base, Rational[-1, 2]] → 1/Sqrt[base] (wolframscript uses
        // this in InputForm too, e.g. Erfc[1/Sqrt[2]]^2)
        if let Expr::FunctionCall {
          name: rname,
          args: rargs,
        } = &args[1]
          && rname == "Rational"
          && rargs.len() == 2
          && matches!(&rargs[0], Expr::Integer(-1))
          && matches!(&rargs[1], Expr::Integer(2))
        {
          let base_str = fmt(&args[0]);
          return format!("1/Sqrt[{base_str}]");
        }
        let base_str = fmt(&args[0]);
        let exp_str = fmt(&args[1]);
        // Wrap base in parens if it's lower precedence than Power or is a negative number
        let base = if matches!(&args[0], Expr::FunctionCall { name, .. } if name == "Plus" || name == "Times")
          || matches!(
            &args[0],
            Expr::BinaryOp {
              op: BinaryOperator::Plus
                | BinaryOperator::Minus
                | BinaryOperator::Times
                | BinaryOperator::Divide,
              ..
            }
          )
          || matches!(&args[0], Expr::Integer(n) if *n < 0)
          || matches!(
            &args[0],
            Expr::Pattern { .. }
              | Expr::PatternOptional { .. }
              | Expr::PatternTest { .. }
          ) {
          format!("({base_str})")
        } else {
          base_str
        };
        // Wrap exponent in parens if it's a Plus, negative, Rational, or Times with negative coefficient
        let exp = if matches!(&args[1], Expr::FunctionCall { name, .. } if name == "Plus")
          || matches!(
            &args[1],
            Expr::BinaryOp {
              op: BinaryOperator::Plus | BinaryOperator::Minus,
              ..
            }
          )
          || matches!(&args[1], Expr::Integer(n) if *n < 0)
          || matches!(
            &args[1],
            Expr::UnaryOp {
              op: UnaryOperator::Minus,
              ..
            }
          )
          || matches!(&args[1], Expr::FunctionCall { name: tname, .. } if tname == "Times")
          || matches!(&args[1], Expr::FunctionCall { name: rname, .. } if rname == "Rational")
          || matches!(
            &args[1],
            Expr::BinaryOp {
              op: BinaryOperator::Divide,
              ..
            }
          )
          || matches!(
            &args[1],
            Expr::Pattern { .. }
              | Expr::PatternOptional { .. }
              | Expr::PatternTest { .. }
          ) {
          format!("({exp_str})")
        } else {
          exp_str
        };
        return format!("{base}^{exp}");
      }
      // Special case: Derivative[n, f, x] displays as Derivative[n][f][x]
      // and Derivative[n, f] displays as Derivative[n][f]
      // Only applies when args[1] is an Identifier (old flattened format from dispatch).
      // If all args are integers, it's the new CurriedCall-based multi-index format
      // and should display as Derivative[n1, n2, ...] (CurriedCall handles nesting).
      if name == "Derivative"
        && args.len() >= 2
        && matches!(&args[1], Expr::Identifier(_))
      {
        let n_str = fmt(&args[0]);
        let f_str = fmt(&args[1]);
        if args.len() == 3 {
          let x_str = fmt(&args[2]);
          return format!("Derivative[{n_str}][{f_str}][{x_str}]");
        }
        return format!("Derivative[{n_str}][{f_str}]");
      }
      let parts: Vec<String> = args.iter().map(&fmt).collect();
      format!("{}[{}]", call_head_display(name), parts.join(", "))
    }
    // BinaryOp::Times in OutputForm: flatten and check for denominator factors,
    // then fall through to InputForm for the rest
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      ..
    } if is_output => {
      let mut factor_refs = Vec::new();
      flatten_binary_times(expr, &mut factor_refs);
      if factor_refs.iter().any(|f| is_denominator_factor(f)) {
        let factors: Vec<Expr> =
          factor_refs.iter().map(|f| (*f).clone()).collect();
        if let Some(frac) =
          format_times_with_denominator(&factors, expr_to_output)
        {
          return frac;
        }
      }
      // Fall through to InputForm handling for non-denominator Times.
      // Mark the OutputForm re-entry so direct string factors render unquoted
      // (`Hold["a" "b"]` → `Hold[a*b]`, matching wolframscript); see the
      // generic BinaryOp arm below for the rationale.
      let _out_guard =
        OutputFormGuard(IN_OUTPUT_FORM.with(|c| c.replace(true)));
      format_expr(expr, ExprForm::Input)
    }
    // BinaryOp::Power with Rational[-1, 2] exponent → 1/Sqrt[base]
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } if matches!(
      right.as_ref(),
      Expr::FunctionCall { name, args }
        if name == "Rational"
          && args.len() == 2
          && matches!(&args[0], Expr::Integer(-1))
          && matches!(&args[1], Expr::Integer(2))
    ) =>
    {
      let base_str = expr_to_output(left);
      format!("1/Sqrt[{base_str}]")
    }
    // StringJoin renders as the function-call form `StringJoin[a, b, ...]`,
    // never as the infix `<>` operator. Wolfram's parser produces a flat
    // `StringJoin[a, b, c]`, and held instances stay in that form when
    // displayed (e.g. `Hold["a"<>"b"<>"c"]` → `Hold[StringJoin[a, b, c]]`).
    // Handle this before the generic OutputForm→InputForm fallthrough so
    // that strings inside render without quotes in OutputForm contexts.
    Expr::BinaryOp {
      op: BinaryOperator::StringJoin,
      left,
      right,
    } => {
      fn collect_string_join(e: &Expr, out: &mut Vec<Expr>) {
        match e {
          Expr::BinaryOp {
            op: BinaryOperator::StringJoin,
            left,
            right,
          } => {
            collect_string_join(left, out);
            collect_string_join(right, out);
          }
          other => out.push(other.clone()),
        }
      }
      let mut parts: Vec<Expr> = Vec::new();
      collect_string_join(left, &mut parts);
      collect_string_join(right, &mut parts);
      // Inside a genuine InputForm render (`expr_to_input_form`, which
      // routes some nodes through the OutputForm path) the operands must
      // keep their quotes — `StringJoin["z = ", ToString[z]]` — or the
      // result doesn't re-parse. Display forms keep the unquoted operands
      // (`StringJoin[z = , ToString[z]]`, matching wolframscript).
      let rendered: Vec<String> = if in_true_input_form() {
        parts.iter().map(expr_to_input_form).collect()
      } else {
        parts.iter().map(&fmt).collect()
      };
      format!("StringJoin[{}]", rendered.join(", "))
    }
    // All other BinaryOps in OutputForm fall through to InputForm
    Expr::BinaryOp { .. } if is_output => {
      // The InputForm path is reused as the 1D renderer for OutputForm
      // BinaryOps. Mark this re-entry so direct string operands of a held
      // Plus/Times render unquoted (`Hold["a" + "b"]` → `Hold[a + b]`,
      // matching wolframscript) while genuine InputForm still quotes. Scoped
      // to this fallthrough so it can't disturb nested renders that switch to
      // InputForm deliberately (e.g. Quantity units).
      let _out_guard =
        OutputFormGuard(IN_OUTPUT_FORM.with(|c| c.replace(true)));
      format_expr(expr, ExprForm::Input)
    }
    // InputForm BinaryOp handling
    Expr::BinaryOp { op, left, right } => {
      // Special case: (-x)/y should display as -(x/y) (Wolfram convention)
      // Only when the numerator is exactly Times[-1, x], not a negative integer coefficient
      if matches!(op, BinaryOperator::Divide) {
        let negated_inner = match left.as_ref() {
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: t_left,
            right: t_right,
          } if matches!(t_left.as_ref(), Expr::Integer(-1)) => {
            Some(t_right.as_ref())
          }
          Expr::UnaryOp {
            op: UnaryOperator::Minus,
            operand,
          } => Some(operand.as_ref()),
          Expr::FunctionCall { name, args }
            if name == "Times"
              && args.len() >= 2
              && matches!(&args[0], Expr::Integer(-1)) =>
          {
            None // handled below for FunctionCall variant
          }
          _ => None,
        };
        if let Some(inner) = negated_inner {
          let inner_div = Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(inner.clone()),
            right: right.clone(),
          };
          let inner_str = expr_to_string(&inner_div);
          return format!("-({inner_str})");
        }
        // Handle FunctionCall Times[-1, ...] as numerator
        if let Expr::FunctionCall { name, args } = left.as_ref()
          && name == "Times"
          && args.len() >= 2
          && matches!(&args[0], Expr::Integer(-1))
        {
          let pos_args = args[1..].to_vec();
          let pos_numerator = if pos_args.len() == 1 {
            pos_args[0].clone()
          } else {
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: pos_args.into(),
            }
          };
          let inner_div = Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(pos_numerator),
            right: right.clone(),
          };
          let inner_str = expr_to_string(&inner_div);
          return format!("-({inner_str})");
        }
        // Special case: 1/identifier → identifier^(-1) (Wolfram InputForm convention)
        if matches!(left.as_ref(), Expr::Integer(1))
          && let Expr::Identifier(s) = right.as_ref()
        {
          return format!("{s}^(-1)");
        }
        // Special case: 1/Plus[...] → (Plus[...])^(-1) (Wolfram InputForm convention)
        if matches!(left.as_ref(), Expr::Integer(1))
          && (matches!(right.as_ref(), Expr::FunctionCall { name, .. } if name == "Plus")
            || matches!(
              right.as_ref(),
              Expr::BinaryOp {
                op: BinaryOperator::Plus | BinaryOperator::Minus,
                ..
              }
            ))
        {
          let rhs = expr_to_string(right);
          return format!("({rhs})^(-1)");
        }
      }

      // Special case: BinaryOp Times with Rational[1, d] or Rational[-1, d]
      // Rational[1, d] * expr → "expr/d", Rational[-1, d] * expr → "-expr/d"
      if matches!(op, BinaryOperator::Times)
        && let Expr::FunctionCall {
          name: rname,
          args: rargs,
        } = left.as_ref()
        && rname == "Rational"
        && rargs.len() == 2
        && let (Expr::Integer(num), Expr::Integer(den)) = (&rargs[0], &rargs[1])
        && *num == 1
        && *den > 0
      {
        let inner = expr_to_string(right);
        return format!("{inner}/{den}");
      }

      // Special case: BinaryOp Times with Rational[n, d] * expr → "(n*expr)/d"
      // When expr contains I (imaginary unit), use complex grouping: "((n*I)/d)*rest"
      if matches!(op, BinaryOperator::Times)
        && let Expr::FunctionCall {
          name: rname,
          args: rargs,
        } = left.as_ref()
        && rname == "Rational"
        && rargs.len() == 2
        && let Expr::Integer(num) = &rargs[0]
        && let Expr::Integer(den) = &rargs[1]
        && *num != 1
        && *num != -1
        && *den > 0
      {
        // Check if the right side is a Times containing I
        let right_has_i = |e: &Expr| -> bool {
          match e {
            Expr::Identifier(n) if n == "I" => true,
            Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: l,
              right: r,
            } => {
              matches!(l.as_ref(), Expr::Identifier(n) if n == "I")
                || matches!(r.as_ref(), Expr::Identifier(n) if n == "I")
            }
            Expr::FunctionCall { name: tn, args: ta } if tn == "Times" => ta
              .iter()
              .any(|a| matches!(a, Expr::Identifier(n) if n == "I")),
            _ => false,
          }
        };
        if right_has_i(right) {
          // Collect symbolic factors (non-I) from right
          let mut symbolic: Vec<&Expr> = Vec::new();
          match right.as_ref() {
            Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: l,
              right: r,
            } => {
              if !matches!(l.as_ref(), Expr::Identifier(n) if n == "I") {
                symbolic.push(l);
              }
              if !matches!(r.as_ref(), Expr::Identifier(n) if n == "I") {
                symbolic.push(r);
              }
            }
            Expr::FunctionCall { name: tn, args: ta } if tn == "Times" => {
              for a in ta {
                if !matches!(a, Expr::Identifier(n) | Expr::Constant(n) if n == "I")
                {
                  symbolic.push(a);
                }
              }
            }
            _ => {}
          }
          if symbolic.is_empty() {
            return format!("({num}*I)/{den}");
          }
          let i_part = format!("(({num}*I)/{den})");
          let rest: Vec<String> =
            symbolic.iter().map(|a| expr_to_string(a)).collect();
          return format!("{}*{}", i_part, rest.join("*"));
        }
        let inner = expr_to_string(right);
        let inner_str = if matches!(right.as_ref(), Expr::FunctionCall { name, .. } if name == "Plus")
          || matches!(
            right.as_ref(),
            Expr::BinaryOp {
              op: BinaryOperator::Plus | BinaryOperator::Minus,
              ..
            }
          ) {
          format!("({inner})")
        } else {
          inner
        };
        return format!("({num}*{inner_str})/{den}");
      }

      // Special case: Times[-1, expr] should display as -expr
      if matches!(op, BinaryOperator::Times)
        && matches!(left.as_ref(), Expr::Integer(-1))
      {
        let right_str = expr_to_string(right);
        // Add parens for lower-precedence ops (Plus/Minus), products (Times), and divisions (Divide)
        // Wolfram displays -(a*b) not -a*b, and -(a/b) not -a/b
        return if matches!(
          right.as_ref(),
          Expr::BinaryOp {
            op: BinaryOperator::Plus
              | BinaryOperator::Minus
              | BinaryOperator::Times
              | BinaryOperator::Divide,
            ..
          }
        ) || matches!(right.as_ref(), Expr::FunctionCall { name, .. } if name == "Plus" || name == "Times")
        {
          format!("-({right_str})")
        } else {
          format!("-{right_str}")
        };
      }

      // Special case: a + (-b) should display as a - b
      if matches!(op, BinaryOperator::Plus)
        && let Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand,
        } = right.as_ref()
      {
        let left_str = expr_to_string(left);
        let operand_str = expr_to_string(operand);
        return format!("{left_str} - {operand_str}");
      }
      // Special case: a + Times[neg, ...] should display as a - abs(neg)*...
      // Handles both BinaryOp{Times} and FunctionCall{Times} forms
      if matches!(op, BinaryOperator::Plus) {
        let negated_term = negate_leading_negative_in_times(right);
        if let Some(abs_term) = negated_term {
          let left_str = expr_to_string(left);
          let right_str = expr_to_string(&abs_term);
          return format!("{left_str} - {right_str}");
        }
        // a + Divide[-n, d] displays as a - n/d (Wolfram never shows "+ -").
        if let Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: dnum,
          right: dden,
        } = right.as_ref()
        {
          let abs_num = match dnum.as_ref() {
            Expr::Integer(n) if *n < 0 => Some(Expr::Integer(-n)),
            Expr::Real(r) if *r < 0.0 => Some(Expr::Real(-r)),
            _ => None,
          };
          if let Some(abs_num) = abs_num {
            let abs_term = Expr::BinaryOp {
              op: BinaryOperator::Divide,
              left: Box::new(abs_num),
              right: dden.clone(),
            };
            return format!(
              "{} - {}",
              expr_to_string(left),
              expr_to_string(&abs_term)
            );
          }
        }
      }

      // BinaryOp::Times: flatten and check for denominator factors (negative exponents)
      if matches!(op, BinaryOperator::Times) {
        let mut factor_refs = Vec::new();
        flatten_binary_times(expr, &mut factor_refs);
        if factor_refs.iter().any(|f| is_denominator_factor(f)) {
          let factors: Vec<Expr> =
            factor_refs.iter().map(|f| (*f).clone()).collect();
          if let Some(frac) =
            format_times_with_denominator(&factors, expr_to_string)
          {
            return frac;
          }
        }
        // A product carrying the imaginary unit inside a nested product —
        // `Times[-1/2, Times[I, Sqrt[21]]]`, as Solve builds it — has to be
        // seen as one flat product for the coefficient grouping below to
        // apply, or it prints as `-1/2*I*Sqrt[21]` where wolframscript
        // writes `(-1/2*I)*Sqrt[21]`. Times is Flat, so re-rendering the
        // flattened form is the same expression.
        let mut flat: Vec<Expr> = Vec::with_capacity(factor_refs.len());
        for f in &factor_refs {
          flatten_times_recursive(f, &mut flat);
        }
        let is_i_unit = |a: &Expr| {
          matches!(a, Expr::Identifier(n) | Expr::Constant(n) if n == "I")
            || matches!(a, Expr::FunctionCall { name, args }
            if name == "Complex"
              && args.len() == 2
              && matches!(
                (&args[0], &args[1]),
                (Expr::Integer(0), Expr::Integer(1))
              ))
        };
        if flat.len() > factor_refs.len() && flat.iter().any(is_i_unit) {
          return fmt(&unevaluated("Times", &flat));
        }
      }

      // Mathematica uses no spaces for *, /, ^ but spaces for +, -, &&, ||
      let (op_str, needs_space) = match op {
        BinaryOperator::Plus => ("+", true),
        BinaryOperator::Minus => ("-", true),
        BinaryOperator::Times => ("*", false),
        BinaryOperator::Divide => ("/", false),
        BinaryOperator::Power => ("^", false),
        BinaryOperator::And => ("&&", true),
        BinaryOperator::Or => ("||", true),
        BinaryOperator::StringJoin => ("<>", false),
        BinaryOperator::Alternatives => ("|", true),
      };

      // Display Power[x, Rational[1, 2]] as Sqrt[x]
      if matches!(op, BinaryOperator::Power)
        && let Some(sqrt_arg) = crate::functions::is_sqrt(expr)
      {
        return format!("Sqrt[{}]", expr_to_string(sqrt_arg));
      }

      // Helper to check if expr is a lower-precedence additive expression
      let is_additive = |e: &Expr| -> bool {
        matches!(
          e,
          Expr::BinaryOp {
            op: BinaryOperator::Plus | BinaryOperator::Minus,
            ..
          }
        ) || matches!(e, Expr::FunctionCall { name, .. } if name == "Plus")
      };

      let is_multiplicative = matches!(
        op,
        BinaryOperator::Times | BinaryOperator::Divide | BinaryOperator::Power
      );

      let left_str = expr_to_string(left);
      let right_str = expr_to_string(right);

      // Special case: Or wraps And children in parens (Wolfram convention)
      if matches!(op, BinaryOperator::Or) {
        let is_and_expr = |e: &Expr| -> bool {
          matches!(
            e,
            Expr::BinaryOp {
              op: BinaryOperator::And,
              ..
            }
          ) || matches!(e, Expr::FunctionCall { name, .. } if name == "And")
        };
        let lf = if is_and_expr(left) {
          format!("({left_str})")
        } else {
          left_str
        };
        let rf = if is_and_expr(right) {
          format!("({right_str})")
        } else {
          right_str
        };
        return format!("{lf} || {rf}");
      }

      // Special case: And wraps Or children in parens (Wolfram convention)
      if matches!(op, BinaryOperator::And) {
        let is_or_expr = |e: &Expr| -> bool {
          matches!(
            e,
            Expr::BinaryOp {
              op: BinaryOperator::Or,
              ..
            }
          ) || matches!(e, Expr::FunctionCall { name, .. } if name == "Or")
        };
        let lf = if is_or_expr(left) {
          format!("({left_str})")
        } else {
          left_str
        };
        let rf = if is_or_expr(right) {
          format!("({right_str})")
        } else {
          right_str
        };
        return format!("{lf} && {rf}");
      }

      // Add parens when a lower-precedence expr is inside a higher-precedence one,
      // or when the numerator of a division is a product (Wolfram convention)
      let left_needs_parens = (matches!(op, BinaryOperator::Times)
        // Held right-nested Times chain on the left of another Times — i.e.
        // the inner Times is `BinaryOp[Times, Integer, BinaryOp[Times, …]]`.
        // This is the shape produced for held derivative chains like
        // `Derivative[2,1][#1^3*#2 &] → (3*(2*#1))*1 & `. Canonicalised
        // products are left-nested with an atomic right factor (e.g.
        // `Times[Times[2, c], d]` for `2*c*d`), so checking that the inner
        // right is itself a Times keeps us from over-wrapping those.
        && matches!(
          left.as_ref(),
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: ll,
            right: lr,
          } if matches!(ll.as_ref(), Expr::Integer(_))
            && matches!(
              lr.as_ref(),
              Expr::BinaryOp { op: BinaryOperator::Times, .. }
            )
        ))
        || (is_multiplicative && is_additive(left))
        || (matches!(op, BinaryOperator::Divide | BinaryOperator::Power)
          && (matches!(
            left.as_ref(),
            Expr::BinaryOp {
              op: BinaryOperator::Times,
              ..
            }
          ) || matches!(
            left.as_ref(),
            Expr::FunctionCall { name, .. } if name == "Times"
          )))
        || (matches!(op, BinaryOperator::Power)
          && matches!(
            left.as_ref(),
            Expr::BinaryOp {
              op: BinaryOperator::Divide,
              ..
            }
          ))
        || (matches!(op, BinaryOperator::Power)
          && matches!(left.as_ref(), Expr::Integer(n) if *n < 0))
        // A Power base that prints with a leading minus (e.g. -I, a Complex
        // such as Complex[0, -1]) must be parenthesized: `-I^k` reparses as
        // `-(I^k)` rather than `(-I)^k`.
        || (matches!(op, BinaryOperator::Power) && left_str.starts_with('-'))
        // Power with a Rational base must wrap the base in parens, since
        // Rational prints as `p/q` and unparenthesized `p/q^e` parses
        // as `p/(q^e)` rather than `(p/q)^e`.
        || (matches!(op, BinaryOperator::Power)
          && matches!(
            left.as_ref(),
            Expr::FunctionCall { name, args } if name == "Rational" && args.len() == 2
          ))
        // Power is right-associative: (x^a)^b must be parenthesized to avoid x^a^b = x^(a^b)
        || (matches!(op, BinaryOperator::Power)
          && (matches!(
            left.as_ref(),
            Expr::BinaryOp {
              op: BinaryOperator::Power,
              ..
            }
          ) || matches!(left.as_ref(), Expr::FunctionCall { name, .. } if name == "Power")))
        || (matches!(
          op,
          BinaryOperator::Power
            | BinaryOperator::Times
            | BinaryOperator::Plus
            | BinaryOperator::Minus
        ) && matches!(
          left.as_ref(),
          Expr::Pattern { .. }
            | Expr::PatternOptional { .. }
            | Expr::PatternTest { .. }
        ))
        // Rule/RuleDelayed/ReplaceAll/ReplaceRepeated have very low
        // precedence (`->`/`:>`/`/.`/`//.`); wrap in parens when they appear
        // inside any binary operator so the printed form re-parses to the
        // same structure (e.g. `(a -> b)^2`, not `a -> b^2`, and
        // `(x /. sol) - 0.05`, not `x /. sol - 0.05`, which re-parses as
        // `x /. (sol - 0.05)` since `/.` binds looser than `-`).
        || matches!(
          left.as_ref(),
          Expr::Rule { .. }
            | Expr::RuleDelayed { .. }
            | Expr::ReplaceAll { .. }
            | Expr::ReplaceRepeated { .. }
        )
        || (prints_as_not(left.as_ref())
          && !matches!(op, BinaryOperator::And | BinaryOperator::Or))
        // `(-a)*b`: a leading minus reaches past the product without them.
        || (matches!(op, BinaryOperator::Times)
          && matches!(
            left.as_ref(),
            Expr::UnaryOp {
              op: UnaryOperator::Minus,
              ..
            }
          ));
      let left_formatted = if left_needs_parens {
        format!("({left_str})")
      } else {
        left_str
      };
      let is_right_multiplicative = |e: &Expr| -> bool {
        matches!(
          e,
          Expr::BinaryOp {
            op: BinaryOperator::Times | BinaryOperator::Divide,
            ..
          }
        ) || matches!(e, Expr::FunctionCall { name, .. } if name == "Times")
      };
      // Check if right side of Power is a negative (Times[-1, ...]) or UnaryOp::Minus
      let is_negative_expr = |e: &Expr| -> bool {
        matches!(
          e,
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left,
            ..
          } if matches!(left.as_ref(), Expr::Integer(-1))
        ) || matches!(
          e,
          Expr::UnaryOp {
            op: UnaryOperator::Minus,
            ..
          }
        ) || matches!(e, Expr::Integer(n) if *n < 0)
          || matches!(e, Expr::FunctionCall { name, args } if name == "Times" && !args.is_empty() && matches!(&args[0], Expr::Integer(n) if *n < 0))
      };
      let needs_right_parens = (is_multiplicative && is_additive(right))
        || (matches!(op, BinaryOperator::Divide)
          && is_right_multiplicative(right))
        || (matches!(op, BinaryOperator::Power)
          && (matches!(
            right.as_ref(),
            Expr::FunctionCall { name, args } if name == "Rational" && args.len() == 2
          ) || matches!(
            right.as_ref(),
            Expr::BinaryOp {
              op: BinaryOperator::Divide,
              ..
            }
          ) || is_right_multiplicative(right)
            || is_negative_expr(right)))
        || (matches!(
          op,
          BinaryOperator::Power
            | BinaryOperator::Times
            | BinaryOperator::Plus
            | BinaryOperator::Minus
        ) && matches!(
          right.as_ref(),
          Expr::Pattern { .. }
            | Expr::PatternOptional { .. }
            | Expr::PatternTest { .. }
        ))
        // Plus + Plus (right-nested): preserve grouping in held expressions.
        // The parser produces left-nested chains, so right-nesting only
        // appears via Hold/HoldAll substitution where we want to display
        // `a + (b + c)` rather than auto-flattening to `a + b + c`.
        || (matches!(op, BinaryOperator::Plus)
          && (matches!(
            right.as_ref(),
            Expr::BinaryOp {
              op: BinaryOperator::Plus,
              ..
            }
          ) || matches!(
            right.as_ref(),
            Expr::FunctionCall { name, args } if name == "Plus" && args.len() >= 2
          )))
        // Minus with an additive right operand: subtraction is left-
        // associative, so a held `a - (b + c)` / `a - (b - c)` must keep
        // its parentheses — printing `a - b + c` changes the value.
        || (matches!(op, BinaryOperator::Minus) && is_additive(right))
        // Times * Times (right-nested) where the inner left factor is a
        // bare Integer: this is the literal chain produced for
        // `Derivative[n][# ^ k &]` (e.g. `3*(2*#1)`). Times normally
        // auto-flattens, so this nesting only survives inside Function
        // bodies that hold the structure, and wolframscript prints those
        // with the parens preserved.
        || (matches!(op, BinaryOperator::Times)
          && matches!(
            right.as_ref(),
            Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: rl,
              ..
            } if matches!(rl.as_ref(), Expr::Integer(_))
          ))
        // Rule/RuleDelayed/ReplaceAll/ReplaceRepeated have very low
        // precedence; wrap in parens when they appear inside any binary
        // operator so the printed form re-parses to the same structure.
        || matches!(
          right.as_ref(),
          Expr::Rule { .. }
            | Expr::RuleDelayed { .. }
            | Expr::ReplaceAll { .. }
            | Expr::ReplaceRepeated { .. }
        )
        || (prints_as_not(right.as_ref())
          && !matches!(op, BinaryOperator::And | BinaryOperator::Or))
        // `a*(-b)`, `a/(-b)`: the minus would otherwise swallow what follows.
        || (matches!(op, BinaryOperator::Times | BinaryOperator::Divide)
          && matches!(
            right.as_ref(),
            Expr::UnaryOp {
              op: UnaryOperator::Minus,
              ..
            }
          ));
      let right_formatted = if needs_right_parens {
        format!("({right_str})")
      } else {
        right_str
      };

      if needs_space {
        format!("{left_formatted} {op_str} {right_formatted}")
      } else {
        format!("{left_formatted}{op_str}{right_formatted}")
      }
    }
    // UnaryOp, Comparison, CompoundExpr, ReplaceAll, etc.: always use InputForm
    // (OutputForm falls through to InputForm for these in the original code)
    Expr::UnaryOp { op, operand } => {
      let inner = format_expr(operand, ExprForm::Input);
      if matches!(op, UnaryOperator::Not) {
        // Not: Wolfram formats as " !expr" (leading space) or " !(expr)".
        // The logical operators And/Or/Xor/Xnor/Nand/Nor have an operator
        // form whose precedence is below Not, so wolframscript parenthesises
        // them: `!(Xor[a, b])`, `!(a && b)`, etc.
        let needs_parens = matches!(
          operand.as_ref(),
          Expr::BinaryOp {
            op: BinaryOperator::And | BinaryOperator::Or,
            ..
          }
        ) || matches!(
          operand.as_ref(),
          Expr::FunctionCall { name, .. }
            if matches!(name.as_str(),
              "And" | "Or" | "Xor" | "Xnor" | "Nand" | "Nor" | "Not")
        ) || matches!(
          operand.as_ref(),
          Expr::UnaryOp {
            op: UnaryOperator::Not,
            ..
          }
        );
        if needs_parens {
          format!(" !({inner})")
        } else {
          format!(" !{inner}")
        }
      } else {
        // Minus needs parens around Plus, Minus, Times, Divide and around
        // anonymous functions (so e.g. `-1.4'` displays as `-(0 & )` rather
        // than `-0 & ` which would parse the trailing `&` as binding to
        // the entire wrapped expression).
        let needs_parens = matches!(
          operand.as_ref(),
          Expr::BinaryOp {
            op: BinaryOperator::Plus
              | BinaryOperator::Minus
              | BinaryOperator::Times
              | BinaryOperator::Divide,
            ..
          }
        ) || matches!(
          // `-(-a)`: a second minus needs them or the two run together.
          operand.as_ref(),
          Expr::UnaryOp {
            op: UnaryOperator::Minus,
            ..
          }
        ) || matches!(
          operand.as_ref(),
          Expr::FunctionCall { name, args } if (name == "Times" || name == "Plus") && args.len() >= 2
        ) || matches!(
          operand.as_ref(),
          Expr::Function { .. }
        ) || prints_as_not(operand.as_ref());
        if needs_parens {
          format!("-({inner})")
        } else {
          format!("-{inner}")
        }
      }
    }
    Expr::Comparison {
      operands,
      operators,
    } => {
      if operators.len() >= 2 {
        // When all operators are the same, use infix form: a <= b <= c
        // When operators differ, use Inequality[a, LessEqual, b, Less, c]
        let all_same = operators.windows(2).all(|w| w[0] == w[1]);
        if all_same {
          let op_str = match &operators[0] {
            ComparisonOp::Equal => " == ",
            ComparisonOp::NotEqual => " != ",
            ComparisonOp::Less => " < ",
            ComparisonOp::LessEqual => " <= ",
            ComparisonOp::Greater => " > ",
            ComparisonOp::GreaterEqual => " >= ",
            ComparisonOp::SameQ => " === ",
            ComparisonOp::UnsameQ => " =!= ",
          };
          let parts: Vec<String> = operands
            .iter()
            .map(|e| {
              let s = format_expr(e, form);
              if comparison_operand_needs_parens(e) {
                format!("({s})")
              } else {
                s
              }
            })
            .collect();
          parts.join(op_str)
        } else {
          let mut parts = Vec::with_capacity(operands.len() + operators.len());
          for (i, operand) in operands.iter().enumerate() {
            parts.push(format_expr(operand, form));
            if i < operators.len() {
              let op_name = match &operators[i] {
                ComparisonOp::Equal => "Equal",
                ComparisonOp::NotEqual => "Unequal",
                ComparisonOp::Less => "Less",
                ComparisonOp::LessEqual => "LessEqual",
                ComparisonOp::Greater => "Greater",
                ComparisonOp::GreaterEqual => "GreaterEqual",
                ComparisonOp::SameQ => "SameQ",
                ComparisonOp::UnsameQ => "UnsameQ",
              };
              parts.push(op_name.to_string());
            }
          }
          format!("Inequality[{}]", parts.join(", "))
        }
      } else {
        let fmt_operand = |e: &Expr| -> String {
          let s = format_expr(e, form);
          if comparison_operand_needs_parens(e) {
            format!("({s})")
          } else {
            s
          }
        };
        let mut result = fmt_operand(&operands[0]);
        for (i, op) in operators.iter().enumerate() {
          let op_str = match op {
            ComparisonOp::Equal => "==",
            ComparisonOp::NotEqual => "!=",
            ComparisonOp::Less => "<",
            ComparisonOp::LessEqual => "<=",
            ComparisonOp::Greater => ">",
            ComparisonOp::GreaterEqual => ">=",
            ComparisonOp::SameQ => "===",
            ComparisonOp::UnsameQ => "=!=",
          };
          if i + 1 < operands.len() {
            result = format!(
              "{} {} {}",
              result,
              op_str,
              fmt_operand(&operands[i + 1])
            );
          }
        }
        result
      }
    }
    Expr::CompoundExpr(exprs) => {
      let mut parts: Vec<String> = exprs
        .iter()
        .map(|e| format_expr(e, ExprForm::Input))
        .collect();
      // A trailing semicolon parses as a trailing `Null`, and that one
      // wolframscript writes back as nothing at all: `a; b;` echoes as
      // `a; b; `. A `Null` anywhere else is spelled out.
      if exprs
        .last()
        .is_some_and(|e| matches!(e, Expr::Identifier(n) if n == "Null"))
        && let Some(last) = parts.last_mut()
      {
        last.clear();
      }
      parts.join("; ")
    }
    Expr::Association(items) => {
      // Convention: a value of `RuleDelayed { pattern, replacement }` whose
      // `pattern` equals the key marks an entry that was originally `key :> v`,
      // since `Expr::Association` doesn't otherwise track Rule vs RuleDelayed.
      // A key that is itself a rule needs parentheses so the entry's own
      // `->` isn't misread (e.g. `<|(a -> b) -> 1|>`).
      let fmt_key = |k: &Expr| -> String {
        match k {
          Expr::Rule { .. } | Expr::RuleDelayed { .. } => {
            format!("({})", fmt(k))
          }
          _ => fmt(k),
        }
      };
      let parts: Vec<String> = items
        .iter()
        .map(|(k, v)| match v {
          Expr::RuleDelayed {
            pattern,
            replacement,
          } if assoc_marker_matches(k, pattern) => {
            format!("{} :> {}", fmt_key(k), fmt(replacement))
          }
          _ => format!("{} -> {}", fmt_key(k), fmt(v)),
        })
        .collect();
      format!("<|{}|>", parts.join(", "))
    }
    Expr::Rule {
      pattern,
      replacement,
    } => {
      // Parenthesize the LHS when it is itself a rule (`->` is
      // right-associative, so `(a -> b) -> c` must keep its parentheses) or a
      // pure function (`&` binds looser than `->`, so `(f & ) -> x`).
      let lhs = match pattern.as_ref() {
        Expr::Rule { .. }
        | Expr::RuleDelayed { .. }
        | Expr::Function { .. } => {
          format!("({})", fmt(pattern))
        }
        _ => fmt(pattern),
      };
      // Parenthesize RHS if it's a pure function (& has low precedence) or a
      // `;`-sequence (CompoundExpression binds looser than every operator,
      // including `->`, so `a -> b; c` is `(a -> b); c` without parens —
      // dropping them would silently turn `key -> (v1; v2)` into two
      // top-level statements instead of one rule).
      let rhs_str = fmt(replacement);
      let rhs_final = match replacement.as_ref() {
        Expr::Function { .. } | Expr::CompoundExpr(_) => format!("({rhs_str})"),
        _ => rhs_str,
      };
      format!("{lhs} -> {rhs_final}")
    }
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      // Parenthesize the LHS when it is itself a rule or a pure function
      // (see Expr::Rule).
      let lhs = match pattern.as_ref() {
        Expr::Rule { .. }
        | Expr::RuleDelayed { .. }
        | Expr::Function { .. } => {
          format!("({})", fmt(pattern))
        }
        _ => fmt(pattern),
      };
      // Parenthesize RHS if it's an assignment so the := operator is
      // correctly disambiguated from the :> operator visually, or if it's a
      // `;`-sequence — CompoundExpression has the lowest precedence of any
      // operator, so `cond :> a = 1; b = 2` is `(cond :> a = 1); b = 2`
      // without parens (the Demonstrations idiom `"event" :> (a = 1; b =
      // 2)` for a multi-statement event handler action would silently lose
      // its later statements from the rule).
      let rhs_str = fmt(replacement);
      let rhs_final = match replacement.as_ref() {
        Expr::FunctionCall { name: n, args: a }
          if matches!(
            n.as_str(),
            "Set" | "SetDelayed" | "UpSet" | "UpSetDelayed"
          ) && a.len() == 2 =>
        {
          format!("({rhs_str})")
        }
        Expr::CompoundExpr(_) => format!("({rhs_str})"),
        _ => rhs_str,
      };
      format!("{lhs} :> {rhs_final}")
    }
    // ReplaceAll, ReplaceRepeated, Map, Apply, etc.: always use InputForm
    Expr::ReplaceAll { expr, rules } => {
      format!(
        "{} /. {}",
        format_expr(expr, ExprForm::Input),
        format_expr(rules, ExprForm::Input)
      )
    }
    Expr::ReplaceRepeated { expr, rules } => {
      format!(
        "{} //. {}",
        format_expr(expr, ExprForm::Input),
        format_expr(rules, ExprForm::Input)
      )
    }
    Expr::Map { func, list } => {
      format_map_apply_shorthand(func, list, "/@", 47)
    }
    Expr::Apply { func, list } => {
      format_map_apply_shorthand(func, list, "@@", 46)
    }
    Expr::MapApply { func, list } => {
      format_map_apply_shorthand(func, list, "@@@", 46)
    }
    Expr::PrefixApply { func, arg } => {
      // f @ g is displayed as f[g] (Wolfram converts @ to function call notation)
      let func_str = format_expr(func, ExprForm::Input);
      let arg_str = format_expr(arg, ExprForm::Input);
      // Parenthesize func if it's complex (not a simple identifier or function call)
      let func_display = match func.as_ref() {
        head if prints_as_not(head) => format!("({func_str})"),
        Expr::Identifier(_)
        | Expr::FunctionCall { .. }
        | Expr::CurriedCall { .. } => func_str,
        _ => format!("({func_str})"),
      };
      format!("{func_display}[{arg_str}]")
    }
    Expr::Postfix { expr, func } => {
      // Display as function-call form (matching wolframscript): `x // f` is
      // semantically `f[x]`, so render as such inside Hold/FullForm output.
      let func_str = format_expr(func, ExprForm::Input);
      let arg_str = format_expr(expr, ExprForm::Input);
      let func_display = match func.as_ref() {
        head if prints_as_not(head) => format!("({func_str})"),
        Expr::Identifier(_)
        | Expr::FunctionCall { .. }
        | Expr::CurriedCall { .. } => func_str,
        _ => format!("({func_str})"),
      };
      format!("{func_display}[{arg_str}]")
    }
    Expr::Part { expr, index } => {
      // Flatten nested Part into a single [[i, j, k]] notation
      let mut indices = vec![expr_to_part_index_string(index, form)];
      let mut base = expr.as_ref();
      while let Expr::Part {
        expr: inner_expr,
        index: inner_index,
      } = base
      {
        indices.push(expr_to_part_index_string(inner_index, form));
        base = inner_expr.as_ref();
      }
      indices.reverse();
      format_part_brackets(base, &format_expr(base, form), &indices)
    }
    Expr::Function { body } => {
      // Wolfram shows anonymous functions with trailing space: "f & " (not "f &")
      // A body that prints with an operator binding looser than `&` — i.e.
      // a `;` compound — has to be parenthesized, or `(a; b) &` would print
      // as `a; b & ` and re-parse as `CompoundExpression[a, b &]`, which is
      // a different (and, once slots are involved, badly broken) tree.
      let body_str = format_expr(body, ExprForm::Input);
      if function_body_needs_parens(body) {
        format!("({body_str}) & ")
      } else {
        format!("{body_str} & ")
      }
    }
    Expr::NamedFunction {
      params,
      body,
      bracketed,
    } => {
      if params.len() == 1 && !bracketed {
        format!(
          "Function[{}, {}]",
          params[0],
          format_expr(body, ExprForm::Input)
        )
      } else {
        format!(
          "Function[{{{}}}, {}]",
          params.join(", "),
          format_expr(body, ExprForm::Input)
        )
      }
    }
    Expr::Pattern {
      name,
      head,
      blank_type,
    } => {
      let blanks = "_".repeat(*blank_type as usize);
      if let Some(h) = head {
        format!("{name}{blanks}{h}")
      } else {
        format!("{name}{blanks}")
      }
    }
    Expr::PatternOptional {
      name,
      head,
      default,
    } => match (head, default) {
      (Some(h), Some(d)) => {
        format!("{}_{}:{}", name, h, format_expr(d, ExprForm::Input))
      }
      (None, Some(d)) => {
        format!("{}_:{}", name, format_expr(d, ExprForm::Input))
      }
      (Some(h), None) => format!("{name}_{h}."),
      (None, None) => format!("{name}_."),
    },
    Expr::PatternTest {
      name,
      head,
      blank_type,
      test,
    } => {
      let blanks = "_".repeat(*blank_type as usize);
      let head_str = head.as_deref().unwrap_or("");
      let test_str = format_expr(test, ExprForm::Input);
      let pattern_str = format!("{name}{blanks}{head_str}");
      // An unnamed blank is the shape wolframscript writes without
      // parentheses — `_?test`, `___?test`, `_Integer?test`. A name makes
      // it a `Pattern`, which is wrapped: `(x_)?test`, `(x_Integer)?test`.
      let needs_parens = !name.is_empty();
      if !matches!(test.as_ref(), Expr::Identifier(_)) {
        // Non-atomic test needs parens around the test too.
        if needs_parens {
          format!("({pattern_str})?({test_str})")
        } else {
          format!("{pattern_str}?({test_str})")
        }
      } else if needs_parens {
        format!("({pattern_str})?{test_str}")
      } else {
        format!("{pattern_str}?{test_str}")
      }
    }
    Expr::Constant(s) => s.clone(),
    Expr::Raw(s) => s.clone(),
    Expr::Image { .. } => "-Image-".to_string(),
    Expr::Graphics { is_3d, .. } => {
      if *is_3d {
        "-Graphics3D-".to_string()
      } else {
        "-Graphics-".to_string()
      }
    }
    Expr::CurriedCall { func, args } => {
      // Display as nested calls: f[a][b, c]
      // When func is a Function (body &) or a non-atomic expression
      // (e.g. a Plus/Times like `1 + x + y + x*y`), wrap in parens so
      // `[x]` clearly applies to the whole head, matching Wolfram's
      // `(1 + x + y + x*y)[x]` rendering. Pattern/Optional heads
      // (`s:A[x]`, `x_:default`) also need parens so the `:` doesn't
      // re-associate with the trailing `[args]` — wolframscript
      // prints `(s:A[x])[t]`, not `s:A[x][t]`.
      let args_str: Vec<String> = args.iter().map(&fmt).collect();
      let func_str = fmt(func);
      let needs_parens = curried_head_needs_parens(func.as_ref());
      let func_display = if needs_parens {
        format!("({func_str})")
      } else {
        func_str
      };
      format!("{}[{}]", func_display, args_str.join(", "))
    }
  }
}

/// Convert an Expr back to a string representation (InputForm)
pub fn expr_to_string(expr: &Expr) -> String {
  format_expr(expr, ExprForm::Input)
}

/// Format expression for use in Wolfram-style messages (OutputForm-like).
/// Differences from expr_to_string:
/// - Times uses spaces instead of `*`
/// - Derivative[n][f][x] displays as f'[x], f''[x], etc.
pub fn expr_to_message_form(expr: &Expr) -> String {
  let s = expr_to_string(expr);
  // Replace Derivative[n][f][args] with shorthand notation
  let s = replace_derivative_shorthand(&s);
  // Replace * with space for OutputForm-style multiplication
  s.replace('*', " ")
}

/// Replace Derivative[n][f][args] patterns with short form: f'[args], f''[args], etc.
fn replace_derivative_shorthand(s: &str) -> String {
  let mut result = s.to_string();
  while let Some(start) = result.find("Derivative[") {
    let after_prefix = start + "Derivative[".len();
    // Parse the order n from Derivative[n]
    let Some(close1) = result[after_prefix..].find(']') else {
      break;
    };
    let n_str = &result[after_prefix..after_prefix + close1];
    let Ok(n) = n_str.parse::<i64>() else {
      break;
    };
    if n < 1 {
      break;
    }
    // Expect [f] after Derivative[n]
    let pos_after_n = after_prefix + close1 + 1;
    if !result[pos_after_n..].starts_with('[') {
      break;
    }
    let func_start = pos_after_n + 1;
    let Some(func_end) = find_matching_bracket(&result, func_start) else {
      break;
    };
    let func_name = result[func_start..func_end].to_string();
    // Expect [args] after [f]
    let pos_after_func = func_end + 1;
    if !result[pos_after_func..].starts_with('[') {
      break;
    }
    let args_start = pos_after_func + 1;
    let Some(args_end) = find_matching_bracket(&result, args_start) else {
      break;
    };
    let args_str = result[args_start..args_end].to_string();
    // Build shorthand: f'[x], f''[x], f'''[x], f^(4)[x], ...
    let primes = if n <= 3 {
      "'".repeat(n as usize)
    } else {
      format!("^({n})")
    };
    let replacement = format!("{func_name}{primes}[{args_str}]");
    result = format!(
      "{}{}{}",
      &result[..start],
      replacement,
      &result[args_end + 1..]
    );
  }
  result
}

/// Find the position of the matching closing bracket `]` for content starting at `start`.
/// Handles nested brackets. Returns the position of the closing `]`.
fn find_matching_bracket(s: &str, start: usize) -> Option<usize> {
  let mut depth = 1;
  for (i, ch) in s[start..].char_indices() {
    match ch {
      '[' => depth += 1,
      ']' => {
        depth -= 1;
        if depth == 0 {
          return Some(start + i);
        }
      }
      _ => {}
    }
  }
  None
}

/// True if `expr` is a rule (`a -> b` / `a :> b`), in either the dedicated
/// AST variants or the `Rule`/`RuleDelayed` FunctionCall forms. Used to
/// recognize trailing option rules in display wrappers like `Row`.
pub fn is_rule_expr(expr: &Expr) -> bool {
  matches!(expr, Expr::Rule { .. } | Expr::RuleDelayed { .. })
    || matches!(
      expr,
      Expr::FunctionCall { name, args }
        if (name == "Rule" || name == "RuleDelayed") && args.len() == 2
    )
}

/// Extract the width in printer's points from a `Spacer` expression.
/// Supports `Spacer[w]`, `Spacer[{w, ...}]`.
pub fn spacer_width_pts(expr: &Expr) -> Option<f64> {
  if let Expr::FunctionCall { name, args } = expr
    && name == "Spacer"
    && args.len() == 1
  {
    match &args[0] {
      Expr::Integer(v) => Some(*v as f64),
      Expr::Real(v) => Some(*v),
      // Spacer[{w, h}] or Spacer[{w, h, dh}] — extract w
      Expr::List(elems) if !elems.is_empty() => match &elems[0] {
        Expr::Integer(v) => Some(*v as f64),
        Expr::Real(v) => Some(*v),
        _ => None,
      },
      _ => None,
    }
  } else {
    None
  }
}

/// Render Expr for display output - strings are shown without quotes.
/// This is used for the final output in interpret(), not for round-tripping.
pub fn expr_to_output(expr: &Expr) -> String {
  format_expr(expr, ExprForm::Output)
}

/// Flatten a nested BinaryOp::Times tree into a list of factors.
fn flatten_binary_times<'a>(expr: &'a Expr, out: &mut Vec<&'a Expr>) {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      flatten_binary_times(left, out);
      flatten_binary_times(right, out);
    }
    _ => out.push(expr),
  }
}

/// Recursively flatten a Times product into its factor list, descending into
/// nested `Times` (both `FunctionCall` and `BinaryOp` forms) and decomposing
/// `BinaryOp::Divide` into `numerator` + `Power[denom, -1]`. Times is Flat, so
/// a one-level flatten can leave nested `BinaryOp::Times(-1, I)` factors that
/// hide the imaginary unit from coefficient handling.
fn flatten_times_recursive(expr: &Expr, out: &mut Vec<Expr>) {
  match expr {
    Expr::FunctionCall {
      name,
      args: inner_args,
    } if name == "Times" => {
      for a in inner_args {
        flatten_times_recursive(a, out);
      }
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      flatten_times_recursive(left, out);
      flatten_times_recursive(right, out);
    }
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      flatten_times_recursive(left, out);
      out.push(Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![(**right).clone(), Expr::Integer(-1)].into(),
      });
    }
    // `-x` inside a product is `(-1)*x`; split out the `-1` so an imaginary
    // operand (`-I`) is exposed to the coefficient handling. Don't split a
    // bare negative literal (`-2`) — that is already a numeric coefficient.
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } if !matches!(
      operand.as_ref(),
      Expr::Integer(_) | Expr::Real(_) | Expr::BigInteger(_)
    ) =>
    {
      out.push(Expr::Integer(-1));
      flatten_times_recursive(operand, out);
    }
    other => out.push(other.clone()),
  }
}

/// Check if an expression tree contains any Expr::String nodes.
fn contains_string(expr: &Expr) -> bool {
  match expr {
    Expr::String(_) => true,
    Expr::List(items) => items.iter().any(contains_string),
    Expr::FunctionCall { args, .. } => args.iter().any(contains_string),
    Expr::Association(items) => items
      .iter()
      .any(|(k, v)| contains_string(k) || contains_string(v)),
    Expr::BinaryOp { left, right, .. } => {
      contains_string(left) || contains_string(right)
    }
    Expr::UnaryOp { operand, .. } => contains_string(operand),
    _ => false,
  }
}

/// Escape a string for InputForm representation.
/// Wolfram private-use Unicode box characters are converted to their
/// escape-sequence equivalents (`\!`, `\(`, `\)`, `\*`).
/// Named-character sequences (`\[Name]`) are preserved as-is, while
/// other backslashes are doubled.
fn escape_string_for_input_form(s: &str) -> String {
  use crate::functions::string_ast::{BOX_CLOSE, BOX_OPEN, BOX_SEP, BOX_START};
  let mut escaped = String::with_capacity(s.len() + 10);
  let chars: Vec<char> = s.chars().collect();
  let mut i = 0;
  while i < chars.len() {
    match chars[i] {
      // Private-use Unicode box syntax chars → escape sequences
      c if c == BOX_START => escaped.push_str("\\!"),
      c if c == BOX_OPEN => escaped.push_str("\\("),
      c if c == BOX_SEP => escaped.push_str("\\*"),
      c if c == BOX_CLOSE => escaped.push_str("\\)"),
      '\\' if i + 1 < chars.len() => match chars[i + 1] {
        // \[Name] named character sequences — preserve as-is
        '[' => {
          escaped.push_str("\\[");
          i += 2;
          continue;
        }
        // Any other backslash (including a run of backslashes) — escape each
        // one individually, so `\\` becomes `\\\\` like wolframscript.
        _ => {
          escaped.push_str("\\\\");
        }
      },
      '\\' => {
        // Trailing backslash
        escaped.push_str("\\\\");
      }
      '"' => escaped.push_str("\\\""),
      '\n' => escaped.push_str("\\n"),
      '\t' => escaped.push_str("\\t"),
      '\r' => escaped.push_str("\\r"),
      '\u{F7CD}' => escaped.push_str("\\`"),
      c => escaped.push(c),
    }
    i += 1;
  }
  escaped
}

/// Render the left-hand side of a rule for InputForm, parenthesizing it when
/// it is itself a rule (`->` is right-associative, so `(a -> b) -> c` must
/// keep its parentheses).
fn input_form_rule_lhs(e: &Expr) -> String {
  match e {
    // A nested rule keeps its parens (`->` is right-associative) and a pure
    // function is parenthesised because `&` binds looser than `->`/`:>`
    // (`(#1 & ) -> x`), matching wolframscript.
    Expr::Rule { .. } | Expr::RuleDelayed { .. } | Expr::Function { .. } => {
      format!("({})", expr_to_input_form(e))
    }
    _ => expr_to_input_form(e),
  }
}

/// Parenthesize a rule's RHS when it is a pure function, since `&` binds
/// looser than `->`/`:>` (`x -> (#1 & )`), matching wolframscript. A
/// `;`-sequence is parenthesized too: `CompoundExpression` has the lowest
/// precedence of any operator, so `x -> a; b` is `(x -> a); b` without
/// parens — printing `x -> (a; b)` bare would silently split one rule into
/// a rule plus a stray top-level statement once re-parsed.
fn input_form_rule_rhs(e: &Expr) -> String {
  match e {
    Expr::Function { .. } | Expr::CompoundExpr(_) => {
      format!("({})", expr_to_input_form(e))
    }
    _ => expr_to_input_form(e),
  }
}

thread_local! {
  /// True while rendering *genuine* InputForm (via `expr_to_input_form`).
  /// `format_expr` reuses its InputForm path as the 1D renderer for OutputForm
  /// BinaryOps, so `is_output` alone cannot tell the two apart. This flag lets
  /// the imaginary-coefficient handling emit the parenthesised InputForm
  /// (`(-I)*x`, `(I/2)*x`) only when truly producing InputForm, and keep the
  /// bare OutputForm (`-I*x`, `I/2*x`) for the bare echo.
  static IN_TRUE_INPUT_FORM: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };

  /// True while rendering OutputForm. `format_expr` reuses its InputForm path
  /// as the 1D renderer for OutputForm BinaryOps and operator shorthands (`/@`,
  /// `@@`, `&`, …), which re-enters `format_expr` with `ExprForm::Input` and so
  /// loses the original `is_output` context. This flag survives that re-entry so
  /// string operands of held operators (e.g. `Hold["a" + "b"]`) render without
  /// quotes in OutputForm — matching wolframscript's `Hold[a + b]` — while
  /// genuine InputForm (`expr_to_input_form`, bare `expr_to_string`) still quotes.
  static IN_OUTPUT_FORM: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };

  /// True while `ToString` is rendering. wolframscript's OutputForm of
  /// `MessageName[f, "usage"]` is the short `f::usage` under `ToString`, but
  /// the bare script-mode echo (and `Print`) shows the long head form:
  /// `Solve[Abs[x] == 2, x, Complexes]; $MessageList` prints
  /// `{HoldForm[MessageName[Solve, ifun]]}` while
  /// `ToString[$MessageList, InputForm]` is `{HoldForm[Solve::ifun]}`. Both
  /// go through the same OutputForm renderer here, so this flag tells them
  /// apart.
  static IN_TO_STRING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };

  /// True while rendering the inner expression of a `FullForm[…]` wrapper.
  /// FullForm reuses `expr_to_input_form` to render its argument, but Span must
  /// stay in head form there (`FullForm[1 ;; 4]` prints `FullForm[Span[1, 4]]`)
  /// even though genuine InputForm renders it as `1 ;; 4`. This flag lets the
  /// Span branch fall through to the head-form catch-all inside FullForm.
  static IN_FULL_FORM: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the current render is genuine InputForm (see `IN_TRUE_INPUT_FORM`).
fn in_true_input_form() -> bool {
  IN_TRUE_INPUT_FORM.with(std::cell::Cell::get)
}

/// Whether the current render originated from OutputForm (see `IN_OUTPUT_FORM`).
fn in_output_form() -> bool {
  IN_OUTPUT_FORM.with(std::cell::Cell::get)
}

/// Whether `ToString` is the caller of the current render (see `IN_TO_STRING`).
pub(crate) fn in_to_string() -> bool {
  IN_TO_STRING.with(std::cell::Cell::get)
}

/// Run `render` with [`in_to_string`] reporting true, restoring the previous
/// value afterwards so a `ToString` reached from a bare echo (say through a
/// `Format` rule) does not leak the flag back out.
pub fn rendering_for_to_string<T>(render: impl FnOnce() -> T) -> T {
  struct Restore(bool);
  impl Drop for Restore {
    fn drop(&mut self) {
      IN_TO_STRING.with(|f| f.set(self.0));
    }
  }
  let _guard = Restore(IN_TO_STRING.with(std::cell::Cell::get));
  IN_TO_STRING.with(|f| f.set(true));
  render()
}

/// Whether the current render is the inner of a FullForm wrapper (see
/// `IN_FULL_FORM`). Used by the Span InputForm branch to keep the head form
/// (`Span[1, 4]`) inside `FullForm[…]` while rendering `1 ;; 4` elsewhere.
fn in_full_form() -> bool {
  IN_FULL_FORM.with(std::cell::Cell::get)
}

/// RAII guard that restores the previous `IN_FULL_FORM` value on drop.
struct FullFormGuard(bool);
impl Drop for FullFormGuard {
  fn drop(&mut self) {
    IN_FULL_FORM.with(|c| c.set(self.0));
  }
}

/// Render `expr` as the inner of a FullForm wrapper: like `expr_to_input_form`
/// but Span keeps its head form (`Span[1, 4]`, not `1 ;; 4`).
fn full_form_inner(expr: &Expr) -> String {
  let _guard = FullFormGuard(IN_FULL_FORM.with(|c| c.replace(true)));
  expr_to_input_form(expr)
}

/// RAII guard that restores the previous `IN_OUTPUT_FORM` value on drop.
struct OutputFormGuard(bool);
impl Drop for OutputFormGuard {
  fn drop(&mut self) {
    IN_OUTPUT_FORM.with(|c| c.set(self.0));
  }
}

/// RAII guard that restores the previous `IN_TRUE_INPUT_FORM` value on drop.
struct TrueInputFormGuard(bool);
impl Drop for TrueInputFormGuard {
  fn drop(&mut self) {
    IN_TRUE_INPUT_FORM.with(|c| c.set(self.0));
  }
}

/// True when `e` binds looser than the symbolic ring products
/// (TensorProduct/CircleTimes, precedence 36) and so must be parenthesised when
/// used as an operand, matching wolframscript (`TensorProduct[a + b, c]` →
/// `(a + b) ⊗ c`). Tighter-binding operands (Power, Dot, …) stay bare.
fn ring_operand_needs_parens(e: &Expr) -> bool {
  match e {
    Expr::BinaryOp { op, .. } => matches!(
      op,
      BinaryOperator::Plus
        | BinaryOperator::Minus
        | BinaryOperator::Times
        | BinaryOperator::Divide
        | BinaryOperator::And
        | BinaryOperator::Or
        | BinaryOperator::StringJoin
        | BinaryOperator::Alternatives
    ),
    Expr::Comparison { .. } => true,
    Expr::Rule { .. } | Expr::RuleDelayed { .. } => true,
    Expr::FunctionCall { name, args }
      if args.len() >= 2 && matches!(name.as_str(), "Plus" | "Times") =>
    {
      true
    }
    _ => false,
  }
}

/// Render Expr in InputForm - like expr_to_output but strings are quoted.
/// If `e` is a negative numeric coefficient (negative Real, or Rational with a
/// negative numerator), return its positive counterpart; otherwise None.
/// Used by the Plus InputForm renderer to turn `+ (-c)*x` into `- c*x`.
fn negate_neg_numeric_coeff(e: &Expr) -> Option<Expr> {
  match e {
    Expr::Real(r) if *r < 0.0 => Some(Expr::Real(-r)),
    Expr::BigInteger(n) if n.sign() == Sign::Minus => {
      Some(Expr::BigInteger(-n))
    }
    Expr::FunctionCall { name, args }
      if name == "Rational"
        && args.len() == 2
        && matches!(&args[0], Expr::Integer(n) if *n < 0) =>
    {
      if let Expr::Integer(n) = &args[0] {
        Some(Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(-n), args[1].clone()].into(),
        })
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Classify a `Times` factor as a negative numeric coefficient for InputForm
/// subtraction folding. Returns `None` if the factor is not a negative number;
/// `Some(None)` if it is exactly `-1` (the factor drops out entirely);
/// `Some(Some(pos))` with the sign-flipped coefficient otherwise. Handles
/// `Integer`, `Real`, `BigInteger`, and `Rational` numerator-negative factors.
fn neg_times_coeff(e: &Expr) -> Option<Option<Expr>> {
  match e {
    Expr::Integer(-1) => Some(None),
    Expr::Integer(n) if *n < 0 => Some(Some(Expr::Integer(-n))),
    _ => negate_neg_numeric_coeff(e).map(Some),
  }
}

/// Render a term that follows a ` - ` in a `Plus`, parenthesising a prefix
/// `MinusPlus` so the leading `-` doesn't fuse with the `∓`, the way Wolfram
/// prints it: `a - MinusPlus[b]` → `a - (∓b)`. Also parenthesises any term
/// that itself prints with an operator at or below `Plus`/`Minus`'s own
/// precedence — reached from the negative-`Times`-coefficient folding above,
/// e.g. `Plus[0.05, Times[-1, ReplaceAll[x, sol]]]` (ordinary evaluated
/// `0.05 - (x /. sol)` with `sol` unbound) must print `0.05 - (x /. sol)`,
/// not `0.05 - x /. sol`, which re-parses as `(0.05 - x) /. sol`.
fn input_form_subtracted_term(term: &Expr) -> String {
  let s = expr_to_input_form(term);
  if matches!(
    term,
    Expr::FunctionCall { name, args } if name == "MinusPlus" && args.len() == 1
  ) || plus_nonleading_term_needs_parens(term)
  {
    format!("({s})")
  } else {
    s
  }
}

pub fn expr_to_input_form(expr: &Expr) -> String {
  // Grow the stack when running low so rendering a deeply nested expression
  // (e.g. the script-mode display of Nest[f, x, 500], or FullForm of it) does
  // not overflow. The recursion re-enters this public entry, so every level is
  // checked. Mirrors the guard on format_expr / evaluate_expr_to_expr.
  stacker::maybe_grow(2 * 1024 * 1024, 4 * 1024 * 1024, || {
    with_display_names(expr, expr_to_input_form_impl)
  })
}

fn expr_to_input_form_impl(expr: &Expr) -> String {
  let _guard = TrueInputFormGuard(IN_TRUE_INPUT_FORM.with(|c| c.replace(true)));
  match expr {
    // `Definition[sym]` / `FullDefinition[sym]` print as the definition text
    // in every form, InputForm included (as wolframscript does).
    Expr::FunctionCall { name, args }
      if (name == "Definition" || name == "FullDefinition")
        && args.len() == 1
        && matches!(&args[0], Expr::Identifier(_)) =>
    {
      expr_to_string(expr)
    }
    Expr::String(s) => {
      let escaped = escape_string_for_input_form(s);
      format!("\"{escaped}\"")
    }
    Expr::List(items) => {
      let parts: Vec<String> = items.iter().map(expr_to_input_form).collect();
      format!("{{{}}}", parts.join(", "))
    }
    Expr::Association(items) => {
      let parts: Vec<String> = items
        .iter()
        .map(|(k, v)| match v {
          Expr::RuleDelayed {
            pattern,
            replacement,
          } if assoc_marker_matches(k, pattern) => {
            format!(
              "{} :> {}",
              expr_to_input_form(k),
              expr_to_input_form(replacement)
            )
          }
          _ => {
            format!("{} -> {}", input_form_rule_lhs(k), expr_to_input_form(v))
          }
        })
        .collect();
      format!("<|{}|>", parts.join(", "))
    }
    Expr::Rule {
      pattern,
      replacement,
    } => {
      format!(
        "{} -> {}",
        input_form_rule_lhs(pattern),
        input_form_rule_rhs(replacement)
      )
    }
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      // Parenthesize RHS if it's an assignment (Set/SetDelayed/Up*), so
      // that e.g. `Initialization :> (d[t_] := ...)` renders with the
      // parentheses required to disambiguate operator precedence. A pure
      // function on the RHS is also parenthesised (`&` binds looser than
      // `:>`), and so is a `;`-sequence — `CompoundExpression` has the
      // lowest precedence of any operator, so `cond :> a = 1; b = 2` is
      // `(cond :> a = 1); b = 2` without parens (the Demonstrations idiom
      // `"event" :> (a = 1; b = 2)` for a multi-statement action would
      // silently lose its later statements once re-parsed).
      let rhs_str = expr_to_input_form(replacement);
      let rhs_final = match replacement.as_ref() {
        Expr::FunctionCall { name: n, args: a }
          if matches!(
            n.as_str(),
            "Set" | "SetDelayed" | "UpSet" | "UpSetDelayed"
          ) && a.len() == 2 =>
        {
          format!("({rhs_str})")
        }
        Expr::Function { .. } | Expr::CompoundExpr(_) => format!("({rhs_str})"),
        _ => rhs_str,
      };
      format!("{} :> {}", input_form_rule_lhs(pattern), rhs_final)
    }
    Expr::FunctionCall { name, args }
      if name == "RuleDelayed" && args.len() == 2 =>
    {
      let rhs_str = expr_to_input_form(&args[1]);
      let rhs_final = match &args[1] {
        Expr::FunctionCall { name: n, args: a }
          if matches!(
            n.as_str(),
            "Set" | "SetDelayed" | "UpSet" | "UpSetDelayed"
          ) && a.len() == 2 =>
        {
          format!("({rhs_str})")
        }
        Expr::Function { .. } | Expr::CompoundExpr(_) => format!("({rhs_str})"),
        _ => rhs_str,
      };
      format!("{} :> {}", input_form_rule_lhs(&args[0]), rhs_final)
    }
    // Equivalent renders in functional form in InputForm: `Equivalent[a, b]`
    // (the infix `⧦` glyph is OutputForm-only), matching wolframscript.
    Expr::FunctionCall { name, args }
      if name == "Equivalent" && args.len() >= 2 =>
    {
      let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
      format!("Equivalent[{}]", parts.join(", "))
    }
    // TensorProduct[a, b, ...] renders with the U+F3DA operator in InputForm
    // (and inside FullForm), matching wolframscript: `a ⊗ b`, `(a + b) ⊗ c`.
    // wolframscript keeps the operator form even under FullForm, so this is
    // not guarded by `in_full_form()`.
    Expr::FunctionCall { name, args }
      if name == "TensorProduct" && args.len() >= 2 =>
    {
      args
        .iter()
        .map(|a| {
          let s = expr_to_input_form(a);
          if ring_operand_needs_parens(a) {
            format!("({s})")
          } else {
            s
          }
        })
        .collect::<Vec<_>>()
        .join(" \u{F3DA} ")
    }
    Expr::FunctionCall { name, args } if name == "Rule" && args.len() == 2 => {
      format!(
        "{} -> {}",
        input_form_rule_lhs(&args[0]),
        input_form_rule_rhs(&args[1])
      )
    }
    // Span[a, b] / Span[a, b, c] render with the `;;` operator in InputForm,
    // matching wolframscript's `ToString[5 ;; 2, InputForm]` -> "5 ;; 2"; a
    // nested Span argument is parenthesised. OutputForm and the bare echo keep
    // the head form (`Span[5, 2]`) via format_expr, and inside a FullForm
    // wrapper Span also stays in head form, so this branch is skipped there.
    Expr::FunctionCall { name, args }
      if name == "Span"
        && (args.len() == 2 || args.len() == 3)
        && !in_full_form() =>
    {
      args
        .iter()
        .map(|a| {
          let s = expr_to_input_form(a);
          if matches!(a, Expr::FunctionCall { name: n, .. } if n == "Span")
            || prints_as_not(a)
          {
            format!("({s})")
          } else {
            s
          }
        })
        .collect::<Vec<_>>()
        .join(" ;; ")
    }
    // Pattern[name, body] displays as name:body in InputForm; wrap
    // looser-binding bodies (Condition/Rule/RuleDelayed/ReplaceAll/
    // ReplaceRepeated) in parens so `s:a /; b` doesn't flip to
    // `Condition[Pattern[s, a], b]` on re-parse. When `body` is
    // Blank/BlankSequence/BlankNullSequence, fold into Wolfram's
    // underscore form: `Pattern[x, Blank[]]` → `x_`, etc.
    Expr::FunctionCall { name, args }
      if name == "Pattern" && args.len() == 2 =>
    {
      if let Expr::Identifier(nm) = &args[0]
        && let Expr::FunctionCall {
          name: bn,
          args: bargs,
        } = &args[1]
        && bargs.len() <= 1
      {
        let underscores = match bn.as_str() {
          "Blank" => Some("_"),
          "BlankSequence" => Some("__"),
          "BlankNullSequence" => Some("___"),
          _ => None,
        };
        if let Some(under) = underscores {
          if bargs.is_empty() {
            return format!("{nm}{under}");
          }
          if let Expr::Identifier(h) = &bargs[0] {
            return format!("{nm}{under}{h}");
          }
        }
      }
      // Also fold when body is the typed `Expr::Pattern` form with an
      // empty name (the anonymous `_`/`__`/`___` parsed as a Term).
      if let Expr::Identifier(nm) = &args[0]
        && let Expr::Pattern {
          name: bname,
          head,
          blank_type,
        } = &args[1]
        && bname.is_empty()
      {
        let under = "_".repeat(*blank_type as usize);
        return match head {
          Some(h) => format!("{nm}{under}{h}"),
          None => format!("{nm}{under}"),
        };
      }
      let needs_parens = matches!(
        &args[1],
        Expr::Rule { .. }
          | Expr::RuleDelayed { .. }
          | Expr::ReplaceAll { .. }
          | Expr::ReplaceRepeated { .. }
      ) || matches!(
        &args[1],
        Expr::FunctionCall { name: bn, .. }
          if matches!(
            bn.as_str(),
            "Condition"
              | "Rule"
              | "RuleDelayed"
              | "ReplaceAll"
              | "ReplaceRepeated"
          )
      );
      let body = expr_to_input_form(&args[1]);
      if needs_parens {
        format!("{}:({})", expr_to_input_form(&args[0]), body)
      } else {
        format!("{}:{}", expr_to_input_form(&args[0]), body)
      }
    }
    Expr::FunctionCall { name, args } if name == "Set" && args.len() == 2 => {
      let rhs = expr_to_input_form(&args[1]);
      let rhs = if assignment_rhs_needs_parens(&args[1]) {
        format!("({rhs})")
      } else {
        rhs
      };
      format!("{} = {}", expr_to_input_form(&args[0]), rhs)
    }
    // `Out[-k]` for k > 0 renders as `%` shorthand (`%`, `%%`, `%%%`, …)
    // when held; matches expr_to_string/format_expr behavior.
    Expr::FunctionCall { name, args }
      if name == "Out"
        && args.len() == 1
        && matches!(&args[0], Expr::Integer(n) if *n < 0) =>
    {
      if let Expr::Integer(n) = &args[0] {
        let count = (-*n) as usize;
        return "%".repeat(count);
      }
      unreachable!()
    }
    Expr::FunctionCall { name, args }
      if name == "SetDelayed" && args.len() == 2 =>
    {
      let rhs = expr_to_input_form(&args[1]);
      let rhs = if assignment_rhs_needs_parens(&args[1]) {
        format!("({rhs})")
      } else {
        rhs
      };
      format!("{} := {}", expr_to_input_form(&args[0]), rhs)
    }
    Expr::FunctionCall { name, args } if name == "UpSet" && args.len() == 2 => {
      let rhs = expr_to_input_form(&args[1]);
      let rhs = if assignment_rhs_needs_parens(&args[1]) {
        format!("({rhs})")
      } else {
        rhs
      };
      format!("{} ^= {}", expr_to_input_form(&args[0]), rhs)
    }
    Expr::FunctionCall { name, args }
      if name == "UpSetDelayed" && args.len() == 2 =>
    {
      let rhs = expr_to_input_form(&args[1]);
      let rhs = if assignment_rhs_needs_parens(&args[1]) {
        format!("({rhs})")
      } else {
        rhs
      };
      format!("{} ^:= {}", expr_to_input_form(&args[0]), rhs)
    }
    Expr::FunctionCall { name, args } if name == "AddTo" && args.len() == 2 => {
      format!(
        "{} += {}",
        expr_to_input_form(&args[0]),
        expr_to_input_form(&args[1])
      )
    }
    Expr::FunctionCall { name, args }
      if name == "SubtractFrom" && args.len() == 2 =>
    {
      format!(
        "{} -= {}",
        expr_to_input_form(&args[0]),
        expr_to_input_form(&args[1])
      )
    }
    Expr::FunctionCall { name, args }
      if name == "TimesBy" && args.len() == 2 =>
    {
      format!(
        "{} *= {}",
        expr_to_input_form(&args[0]),
        expr_to_input_form(&args[1])
      )
    }
    Expr::FunctionCall { name, args }
      if name == "DivideBy" && args.len() == 2 =>
    {
      format!(
        "{} /= {}",
        expr_to_input_form(&args[0]),
        expr_to_input_form(&args[1])
      )
    }
    Expr::FunctionCall { name, args }
      if name == "Increment" && args.len() == 1 =>
    {
      format!("{}++", expr_to_input_form(&args[0]))
    }
    Expr::FunctionCall { name, args }
      if name == "Decrement" && args.len() == 1 =>
    {
      format!("{}--", expr_to_input_form(&args[0]))
    }
    Expr::FunctionCall { name, args }
      if name == "PreIncrement" && args.len() == 1 =>
    {
      format!("++{}", expr_to_input_form(&args[0]))
    }
    Expr::FunctionCall { name, args }
      if name == "PreDecrement" && args.len() == 1 =>
    {
      format!("--{}", expr_to_input_form(&args[0]))
    }
    // Quantity: InputForm quotes string unit names
    Expr::FunctionCall { name, args }
      if name == "Quantity" && args.len() == 2 =>
    {
      let mag_str = expr_to_input_form(&args[0]);
      let unit_str = quantity_unit_to_input_form(&args[1]);
      format!("Quantity[{mag_str}, {unit_str}]")
    }
    // FunctionCall: handle FullForm specially in InputForm (keep the wrapper)
    Expr::FunctionCall { name, args }
      if name == "FullForm" && args.len() == 1 =>
    {
      // Span keeps its head form inside FullForm (see `full_form_inner`).
      format!("FullForm[{}]", full_form_inner(&args[0]))
    }
    // BaseForm: InputForm shows BaseForm[n, base] structure (not subscript notation)
    Expr::FunctionCall { name, args }
      if name == "BaseForm" && args.len() == 2 =>
    {
      format!(
        "BaseForm[{}, {}]",
        expr_to_input_form(&args[0]),
        expr_to_input_form(&args[1])
      )
    }
    // CForm: InputForm renders the expression as C code
    Expr::FunctionCall { name, args } if name == "CForm" && args.len() == 1 => {
      crate::functions::string_ast::expr_to_c(&args[0])
    }
    // TeXForm: InputForm renders the expression as TeX
    Expr::FunctionCall { name, args }
      if name == "TeXForm" && args.len() == 1 =>
    {
      crate::functions::string_ast::expr_to_tex(&args[0])
    }
    // FortranForm: InputForm renders the expression as Fortran code
    Expr::FunctionCall { name, args }
      if name == "FortranForm" && args.len() == 1 =>
    {
      crate::functions::string_ast::expr_to_fortran(&args[0])
    }
    // StringSkeleton[n]: InputForm shows n<<>> (content before <<>>)
    Expr::FunctionCall { name, args }
      if name == "StringSkeleton" && args.len() == 1 =>
    {
      format!("{}<<>>", expr_to_input_form(&args[0]))
    }
    // MessageName[sym, "tag"]: InputForm shows sym::tag
    Expr::FunctionCall { name, args }
      if name == "MessageName" && args.len() == 2 =>
    {
      let sym = expr_to_input_form(&args[0]);
      let tag = match &args[1] {
        Expr::String(s) => s.clone(),
        Expr::Identifier(s) => s.clone(),
        other => expr_to_input_form(other),
      };
      format!("{sym}::{tag}")
    }
    // StringExpression[a, b, c]: InputForm shows a~~b~~c with quoted strings.
    // A single-element StringExpression has no infix form, so it keeps the
    // head literal (`StringExpression[2]`), matching wolframscript.
    Expr::FunctionCall { name, args }
      if name == "StringExpression" && args.len() == 1 =>
    {
      format!("StringExpression[{}]", expr_to_input_form(&args[0]))
    }
    Expr::FunctionCall { name, args }
      if name == "StringExpression" && !args.is_empty() =>
    {
      let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
      parts.join("~~")
    }
    // StringForm: InputForm shows StringForm["template", args...] with quoted string
    Expr::FunctionCall { name, args }
      if name == "StringForm" && !args.is_empty() =>
    {
      let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
      format!("StringForm[{}]", parts.join(", "))
    }
    // Row, TableForm, MatrixForm: display directive wrappers, show structure in InputForm
    Expr::FunctionCall { name, args }
      if (name == "Row"
        || name == "TableForm"
        || name == "MatrixForm"
        || name == "Column")
        && !args.is_empty() =>
    {
      let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
      format!("{}[{}]", name, parts.join(", "))
    }
    // `TraditionalForm[expr]` becomes a box segment in InputForm text —
    // the *marker* form, not the `\!\(\*…\)` escape. Wolfram's
    // `ToString[InputForm[TraditionalForm[x + y]]]` is 53 characters
    // long and its first three are U+F7C1/F7C9/F7C8; the backslash
    // escape is what the *string* InputForm renderer writes for those
    // markers (`escape_string_for_input_form`, one arm below), so
    // producing it here escaped it twice over. A segment embeds inline,
    // so `Row[{1, TraditionalForm[x^2], "a"}]` keeps its surrounding
    // source text.
    Expr::FunctionCall { name, args }
      if name == "TraditionalForm" && args.len() == 1 =>
    {
      use crate::functions::string_ast::{
        BOX_CLOSE, BOX_OPEN, BOX_SEP, BOX_START,
      };
      let box_str = crate::functions::string_ast::expr_to_boxes(&args[0]);
      format!(
        "{BOX_START}{BOX_OPEN}{BOX_SEP}FormBox[{box_str}, TraditionalForm]{BOX_CLOSE}"
      )
    }
    // Or[a, b, ...] in InputForm: render as a || b || ... using InputForm for children
    Expr::FunctionCall { name, args } if name == "Or" && args.len() >= 2 => {
      let parts: Vec<String> = args
        .iter()
        .map(|arg| {
          let s = expr_to_input_form(arg);
          let is_and = matches!(
            arg,
            Expr::BinaryOp {
              op: BinaryOperator::And,
              ..
            }
          ) || matches!(arg, Expr::FunctionCall { name, .. } if name == "And");
          if is_and { format!("({s})") } else { s }
        })
        .collect();
      parts.join(" || ")
    }
    // And[a, b, ...] in InputForm: render as a && b && ... using InputForm for children
    Expr::FunctionCall { name, args } if name == "And" && args.len() >= 2 => {
      let parts: Vec<String> = args
        .iter()
        .map(|arg| {
          let s = expr_to_input_form(arg);
          let is_or = matches!(
            arg,
            Expr::BinaryOp {
              op: BinaryOperator::Or,
              ..
            }
          ) || matches!(arg, Expr::FunctionCall { name, .. } if name == "Or");
          if is_or { format!("({s})") } else { s }
        })
        .collect();
      parts.join(" && ")
    }
    // Alternatives[a, b, …] in InputForm: `a | b | …` with InputForm
    // children, so a `Span` operand prints with its own operator
    // (`ToString[a | 1 ;; 3, InputForm]` is "a | 1 ;; 3" in wolframscript,
    // not "a | Span[1, 3]"). Without this arm the whole expression fell
    // through to `expr_to_output`, which renders Span in head form.
    Expr::BinaryOp {
      op: BinaryOperator::Alternatives,
      left,
      right,
    } => input_form_infix_forcing(
      &[(**left).clone(), (**right).clone()],
      " | ",
      27,
      is_nested_alternatives_call,
    ),
    Expr::FunctionCall { name, args }
      if name == "Alternatives" && args.len() >= 2 =>
    {
      input_form_infix_forcing(args, " | ", 27, is_nested_alternatives_call)
    }
    // Condition[lhs, rhs] in InputForm: `lhs /; rhs`, again with InputForm
    // children (`ToString[x /; y ;; 3, InputForm]` is "x /; y ;; 3").
    Expr::FunctionCall { name, args }
      if name == "Condition" && args.len() == 2 =>
    {
      input_form_infix(args, " /; ", 13)
    }
    // Inequality[...] in InputForm always uses the head form Inequality[a, Less, b, Less, c],
    // even when all operators are the same (infix is only used in OutputForm).
    Expr::FunctionCall { name, args }
      if name == "Inequality" && args.len() >= 5 && args.len() % 2 == 1 =>
    {
      let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
      format!("Inequality[{}]", parts.join(", "))
    }
    // Unicode operator functions: use same Unicode formatting as OutputForm
    Expr::FunctionCall { name, args } if name == "Colon" && args.len() >= 2 => {
      let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
      parts.join(" \u{2236} ")
    }
    Expr::FunctionCall { name, args } if name == "Cap" && args.len() >= 2 => {
      let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
      parts.join(" \u{2322} ")
    }
    Expr::FunctionCall { name, args } if name == "Cup" && args.len() >= 2 => {
      let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
      parts.join(" \u{2323} ")
    }
    Expr::FunctionCall { name, args }
      if name == "Congruent" && args.len() >= 2 =>
    {
      let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
      parts.join(" \u{2261} ")
    }
    Expr::FunctionCall { name, args }
      if name == "RightTee" && args.len() >= 2 =>
    {
      let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
      parts.join(" \u{22A2} ")
    }
    Expr::FunctionCall { name, args }
      if name == "DoubleRightTee" && args.len() >= 2 =>
    {
      let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
      parts.join(" \u{22A8} ")
    }
    Expr::FunctionCall { name, args }
      if name == "LeftTee" && args.len() >= 2 =>
    {
      let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
      parts.join(" \u{22A3} ")
    }
    Expr::FunctionCall { name, args }
      if name == "DoubleLeftTee" && args.len() >= 2 =>
    {
      let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
      parts.join(" \u{2AE4} ")
    }
    Expr::FunctionCall { name, args }
      if name == "LongRightArrow" && args.len() >= 2 =>
    {
      let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
      parts.join(" \u{27F6} ")
    }
    Expr::FunctionCall { name, args }
      if name == "Proportional" && args.len() >= 2 =>
    {
      let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
      parts.join(" \u{221D} ")
    }
    Expr::FunctionCall { name, args } if name == "PlusMinus" => {
      if args.len() == 1 {
        format!("\u{00B1}{}", expr_to_input_form(&args[0]))
      } else if args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
        parts.join(" \u{00B1} ")
      } else {
        "PlusMinus[]".to_string()
      }
    }
    Expr::FunctionCall { name, args } if name == "MinusPlus" => {
      if args.len() == 1 {
        format!("\u{2213}{}", expr_to_input_form(&args[0]))
      } else if args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
        parts.join(" \u{2213} ")
      } else {
        "MinusPlus[]".to_string()
      }
    }
    Expr::FunctionCall { name, args }
      if name == "CircleTimes" && args.len() >= 2 =>
    {
      let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
      parts.join(" \u{2297} ")
    }
    Expr::FunctionCall { name, args }
      if name == "CenterDot" && args.len() >= 2 =>
    {
      let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
      parts.join(" \u{00B7} ")
    }
    Expr::FunctionCall { name, args }
      if name == "CircleDot" && args.len() >= 2 =>
    {
      let parts: Vec<String> = args
        .iter()
        .map(|a| {
          let s = expr_to_input_form(a);
          if matches!(
            a,
            Expr::FunctionCall { name: n, args: ia }
              if n == "CircleDot" && ia.len() >= 2
          ) {
            format!("({s})")
          } else {
            s
          }
        })
        .collect();
      parts.join(" \u{2299} ")
    }
    Expr::FunctionCall { name, args } if name == "Wedge" && args.len() >= 2 => {
      let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
      parts.join(" \u{22C0} ")
    }
    Expr::FunctionCall { name, args } if name == "Star" && args.len() >= 2 => {
      let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
      parts.join(" \u{22C6} ")
    }
    Expr::FunctionCall { name, args }
      if name == "Diamond" && args.len() >= 2 =>
    {
      let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
      parts.join(" \u{22C4} ")
    }
    Expr::FunctionCall { name, args }
      if name == "Backslash" && args.len() >= 2 =>
    {
      let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
      parts.join(" \u{2216} ")
    }
    Expr::FunctionCall { name, args }
      if name == "SmallCircle" && args.len() >= 2 =>
    {
      let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
      parts.join(" \u{2218} ")
    }
    Expr::FunctionCall { name, args } if name == "Vee" && args.len() >= 2 => {
      let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
      parts.join(" \u{22C1} ")
    }
    Expr::FunctionCall { name, args } if name == "Tilde" && args.len() >= 2 => {
      let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
      parts.join(" \u{223C} ")
    }
    Expr::FunctionCall { name, args } if name == "Del" && args.len() == 1 => {
      format!("\u{2207}{}", expr_to_input_form(&args[0]))
    }
    Expr::FunctionCall { name, args }
      if name == "CirclePlus" && args.len() >= 2 =>
    {
      let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
      parts.join(" \u{2295} ")
    }
    Expr::FunctionCall { name, args }
      if name == "CircleMinus" && args.len() >= 2 =>
    {
      let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
      parts.join(" \u{2296} ")
    }
    Expr::FunctionCall { name, args }
      if name == "Subset" && args.len() == 2 =>
    {
      format!(
        "{} \u{2282} {}",
        expr_to_input_form(&args[0]),
        expr_to_input_form(&args[1])
      )
    }
    Expr::FunctionCall { name, args }
      if name == "LeftArrow" && args.len() == 2 =>
    {
      format!(
        "{} \u{2190} {}",
        expr_to_input_form(&args[0]),
        expr_to_input_form(&args[1])
      )
    }
    Expr::FunctionCall { name, args }
      if name == "DotEqual" && args.len() == 2 =>
    {
      format!(
        "{} \u{2250} {}",
        expr_to_input_form(&args[0]),
        expr_to_input_form(&args[1])
      )
    }
    Expr::FunctionCall { name, args }
      if name == "AngleBracket" && !args.is_empty() =>
    {
      let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
      format!("\u{2329} {} \u{232A}", parts.join(", "))
    }
    // Named slot Slot["name"] displays as #name (matching wolframscript),
    // mirroring the format_expr arm. Must precede the generic FunctionCall
    // arm below, which would otherwise render it as `Slot["name"]`.
    Expr::FunctionCall { name, args }
      if name == "Slot"
        && args.len() == 1
        && matches!(&args[0], Expr::String(key)
          if key.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
            && key.chars().all(|c| c.is_ascii_alphanumeric())) =>
    {
      match &args[0] {
        Expr::String(key) => format!("#{key}"),
        _ => unreachable!(),
      }
    }
    // Generic FunctionCall: render as name[arg1, arg2, ...] with InputForm for args.
    // Known infix operators (Plus, Times, Power, etc.) fall through to expr_to_output
    // since they rarely contain string literals and need infix rendering.
    // `Part[base, i, …]` prints with double brackets, keeping string bases
    // and indices quoted (InputForm). Mirrors the format_expr arm.
    Expr::FunctionCall { name, args } if name == "Part" && !args.is_empty() => {
      let indices: Vec<String> = args[1..]
        .iter()
        .map(|a| expr_to_part_index_string(a, ExprForm::Input))
        .collect();
      format_part_brackets(&args[0], &expr_to_input_form(&args[0]), &indices)
    }
    Expr::FunctionCall { name, args }
      if !matches!(
        name.as_str(),
        "Plus"
          | "Times"
          | "Power"
          | "And"
          | "Or"
          | "Not"
          | "Equal"
          | "Unequal"
          | "Less"
          | "LessEqual"
          | "Greater"
          | "GreaterEqual"
          | "Inequality"
          | "Factorial"
          | "Factorial2"
          | "Increment"
          | "Decrement"
          | "PreIncrement"
          | "PreDecrement"
          | "Condition"
          | "PatternTest"
          | "Optional"
          | "Alternatives"
          | "Dot"
          | "Repeated"
          | "RepeatedNull"
          | "Derivative"
          | "Sequence"
          | "Rational"
          | "Complex"
          | "DirectedInfinity"
          | "Skeleton"
          | "Composition"
          | "RightComposition"
          | "NonCommutativeMultiply"
          | "Minus"
          | "Therefore"
          | "Because"
          | "Blank"
          | "BlankSequence"
          | "BlankNullSequence"
          | "ReverseElement"
          | "TwoWayRule"
      ) =>
    {
      if args.is_empty() {
        format!("{}[]", call_head_display(name))
      } else {
        let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
        format!("{}[{}]", call_head_display(name), parts.join(", "))
      }
    }
    // Chained comparison with 2+ operators:
    // When all operators are the same, use infix: a <= b <= c
    // When operators differ, use Inequality[a, LessEqual, b, Less, c]
    Expr::Comparison {
      operands,
      operators,
    } if operators.len() >= 2 => {
      let all_same = operators.windows(2).all(|w| w[0] == w[1]);
      if all_same {
        // Same operators: use infix form (e.g. 0 <= x <= 1)
        let op_str = match &operators[0] {
          ComparisonOp::Equal => " == ",
          ComparisonOp::NotEqual => " != ",
          ComparisonOp::Less => " < ",
          ComparisonOp::LessEqual => " <= ",
          ComparisonOp::Greater => " > ",
          ComparisonOp::GreaterEqual => " >= ",
          ComparisonOp::SameQ => " === ",
          ComparisonOp::UnsameQ => " =!= ",
        };
        let parts: Vec<String> = operands
          .iter()
          .map(|e| {
            let s = expr_to_input_form(e);
            if comparison_operand_needs_parens(e) {
              format!("({s})")
            } else {
              s
            }
          })
          .collect();
        parts.join(op_str)
      } else {
        // Mixed operators: use Inequality[...] head form
        let mut parts = Vec::with_capacity(operands.len() + operators.len());
        for (i, operand) in operands.iter().enumerate() {
          parts.push(expr_to_input_form(operand));
          if i < operators.len() {
            let op_name = match &operators[i] {
              ComparisonOp::Equal => "Equal",
              ComparisonOp::NotEqual => "Unequal",
              ComparisonOp::Less => "Less",
              ComparisonOp::LessEqual => "LessEqual",
              ComparisonOp::Greater => "Greater",
              ComparisonOp::GreaterEqual => "GreaterEqual",
              ComparisonOp::SameQ => "SameQ",
              ComparisonOp::UnsameQ => "UnsameQ",
            };
            parts.push(op_name.to_string());
          }
        }
        format!("Inequality[{}]", parts.join(", "))
      }
    }
    // Single-operator Comparison (e.g. `a == "A"`): use infix form,
    // but render operands via expr_to_input_form so strings stay quoted.
    Expr::Comparison {
      operands,
      operators,
    } if operators.len() == 1 && operands.len() == 2 => {
      let op_str = match &operators[0] {
        ComparisonOp::Equal => "==",
        ComparisonOp::NotEqual => "!=",
        ComparisonOp::Less => "<",
        ComparisonOp::LessEqual => "<=",
        ComparisonOp::Greater => ">",
        ComparisonOp::GreaterEqual => ">=",
        ComparisonOp::SameQ => "===",
        ComparisonOp::UnsameQ => "=!=",
      };
      let fmt_operand = |e: &Expr| -> String {
        let s = expr_to_input_form(e);
        if comparison_operand_needs_parens(e) {
          format!("({s})")
        } else {
          s
        }
      };
      format!(
        "{} {} {}",
        fmt_operand(&operands[0]),
        op_str,
        fmt_operand(&operands[1])
      )
    }
    // Plus in InputForm: render as infix but use expr_to_input_form for args
    Expr::FunctionCall { name, args } if name == "Plus" && args.len() >= 2 => {
      let first_str = expr_to_input_form(&args[0]);
      let mut result = if plus_term_needs_parens(&args[0]) {
        format!("({first_str})")
      } else {
        first_str
      };
      for arg in args.iter().skip(1) {
        // Check for negation patterns
        if let Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand,
        } = arg
        {
          result.push_str(" - ");
          let operand_str = expr_to_input_form(operand);
          if plus_nonleading_term_needs_parens(operand) {
            result.push_str(&format!("({operand_str})"));
          } else {
            result.push_str(&operand_str);
          }
        } else if let Expr::BinaryOp {
          op: BinaryOperator::Times,
          left,
          right,
        } = arg
        {
          // Pull out a negative numeric coefficient so the term renders as a
          // subtraction (`- (15*x)/2`) instead of an addition of a negative
          // coefficient (`+ (-15*x)/2`), matching wolframscript's InputForm.
          // Canonicalizers such as Expand can leave the coefficient on the
          // *right* factor (e.g. `Times[x^(-2), Rational[-2, 5]]`), so both
          // sides are checked.
          if let Some(pos_left) = neg_times_coeff(left.as_ref()) {
            result.push_str(" - ");
            match pos_left {
              None => result.push_str(&input_form_subtracted_term(right)),
              Some(pl) => {
                let pos = Expr::BinaryOp {
                  op: BinaryOperator::Times,
                  left: Box::new(pl),
                  right: right.clone(),
                };
                result.push_str(&expr_to_input_form(&pos));
              }
            }
          } else if let Some(pos_right) = neg_times_coeff(right.as_ref()) {
            result.push_str(" - ");
            match pos_right {
              None => result.push_str(&input_form_subtracted_term(left)),
              Some(pr) => {
                let pos = Expr::BinaryOp {
                  op: BinaryOperator::Times,
                  left: left.clone(),
                  right: Box::new(pr),
                };
                result.push_str(&expr_to_input_form(&pos));
              }
            }
          } else {
            result.push_str(" + ");
            result.push_str(&expr_to_input_form(arg));
          }
        } else if let Expr::FunctionCall {
          name: tname,
          args: targs,
        } = arg
          && tname == "Times"
          && !targs.is_empty()
        {
          // Check if leading factor is negative (Integer, Real, or Rational)
          let neg_coeff = match &targs[0] {
            Expr::Integer(n) if *n < 0 => Some(if *n == -1 {
              None // coefficient of -1 means just negate
            } else {
              Some(Expr::Integer(-n))
            }),
            Expr::BigInteger(n) if n.sign() == Sign::Minus => {
              Some(Some(Expr::BigInteger(-n)))
            }
            Expr::Real(r) if *r < 0.0 => Some(Some(Expr::Real(-r))),
            Expr::FunctionCall { name: rn, args: ra }
              if rn == "Rational"
                && ra.len() == 2
                && matches!(&ra[0], Expr::Integer(n) if *n < 0) =>
            {
              if let Expr::Integer(n) = &ra[0] {
                if *n == -1 {
                  Some(Some(Expr::FunctionCall {
                    name: "Rational".to_string(),
                    args: vec![Expr::Integer(1), ra[1].clone()].into(),
                  }))
                } else {
                  Some(Some(Expr::FunctionCall {
                    name: "Rational".to_string(),
                    args: vec![Expr::Integer(-n), ra[1].clone()].into(),
                  }))
                }
              } else {
                None
              }
            }
            _ => None,
          };
          if let Some(pos_coeff) = neg_coeff {
            // Negative Times: -n * ... → " - n*..."
            let pos_term = match pos_coeff {
              None => {
                // Times[-1, rest...] → rest
                if targs.len() == 2 {
                  targs[1].clone()
                } else {
                  unevaluated("Times", &targs[1..])
                }
              }
              Some(new_coeff) => {
                let mut new_args = vec![new_coeff];
                new_args.extend_from_slice(&targs[1..]);
                if new_args.len() == 1 {
                  new_args[0].clone()
                } else {
                  Expr::FunctionCall {
                    name: "Times".to_string(),
                    args: new_args.into(),
                  }
                }
              }
            };
            result.push_str(" - ");
            result.push_str(&input_form_subtracted_term(&pos_term));
          } else {
            result.push_str(" + ");
            result.push_str(&expr_to_input_form(arg));
          }
        } else if let Expr::Integer(n) = arg {
          if *n < 0 {
            result.push_str(" - ");
            result.push_str(&expr_to_input_form(&Expr::Integer(-n)));
          } else {
            result.push_str(" + ");
            result.push_str(&expr_to_input_form(arg));
          }
        } else if let Expr::BigInteger(n) = arg {
          if n.sign() == Sign::Minus {
            result.push_str(" - ");
            result.push_str(&expr_to_input_form(&Expr::BigInteger(-n)));
          } else {
            result.push_str(" + ");
            result.push_str(&expr_to_input_form(arg));
          }
        } else {
          result.push_str(" + ");
          let arg_str = expr_to_input_form(arg);
          if plus_nonleading_term_needs_parens(arg) {
            result.push_str(&format!("({arg_str})"));
          } else {
            result.push_str(&arg_str);
          }
        }
      }
      result
    }
    // Times in InputForm: use expr_to_output for structure, but render via
    // expr_to_input_form for args to preserve string quoting
    // A power whose base or exponent is a string: `expr_to_output` would drop
    // the quotes, so write it here. This is what a Quantity unit is made of.
    Expr::FunctionCall { name, args }
      if name == "Power" && args.len() == 2 && contains_string(expr) =>
    {
      let base = expr_to_input_form(&args[0]);
      let base_str = if matches!(&args[0], Expr::FunctionCall { name, .. }
        if name == "Times" || name == "Plus" || name == "Power")
        || matches!(
          &args[0],
          Expr::BinaryOp {
            op: BinaryOperator::Times
              | BinaryOperator::Plus
              | BinaryOperator::Minus
              | BinaryOperator::Power,
            ..
          }
        ) {
        format!("({base})")
      } else {
        base
      };
      let exp = expr_to_input_form(&args[1]);
      // A plain non-negative literal or symbol needs no parentheses.
      let exp_str = if matches!(&args[1], Expr::Integer(n) if *n >= 0)
        || matches!(&args[1], Expr::Real(r) if *r >= 0.0)
        || matches!(
          &args[1],
          Expr::Identifier(_) | Expr::Constant(_) | Expr::String(_)
        ) {
        exp
      } else {
        format!("({exp})")
      };
      format!("{base_str}^{exp_str}")
    }
    Expr::FunctionCall { name, args } if name == "Times" && args.len() >= 2 => {
      // Flatten nested Times so I and -1 are reachable at the top level
      // (Sin[-I] typically comes back as Times[-1, Times[I, Sinh[1]]]).
      let flat_args: Vec<Expr> = args
        .iter()
        .flat_map(|a| match a {
          Expr::FunctionCall {
            name: inner_name,
            args: inner_args,
          } if inner_name == "Times" => inner_args.clone(),
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left,
            right,
          } => vec![*left.clone(), *right.clone()].into(),
          other => vec![other.clone()].into(),
        })
        .collect();
      let args = &flat_args;
      // When the product contains a string literal, render factor-by-factor
      // with input_form so the string is quoted (expr_to_output / format_expr
      // would mis-quote embedded strings).
      let needs_input_form = args.iter().any(contains_string);
      if needs_input_form {
        // A negative-exponent factor still moves to a denominator — a string
        // among the factors only changes how each factor is written, not the
        // shape: `"a"*"b"^-1` is `"a"/"b"`.
        if matches!(&args[0], Expr::Integer(-1)) {
          if let Some(frac) =
            format_times_with_denominator(&args[1..], expr_to_input_form)
          {
            return format!("-({frac})");
          }
        } else if let Some(frac) =
          format_times_with_denominator(args, expr_to_input_form)
        {
          return frac;
        }
        // Handle Times[-1, ...] as negation
        if matches!(&args[0], Expr::Integer(-1)) {
          let rest: Vec<String> = args[1..]
            .iter()
            .map(|a| {
              let s = expr_to_input_form(a);
              if matches!(a, Expr::FunctionCall { name, .. } if name == "Plus")
                || matches!(
                  a,
                  Expr::BinaryOp {
                    op: BinaryOperator::Plus | BinaryOperator::Minus,
                    ..
                  }
                )
              {
                format!("({s})")
              } else {
                s
              }
            })
            .collect();
          format!("-{}", rest.join("*"))
        } else {
          let parts: Vec<String> = args.iter().map(|a| {
            let s = expr_to_input_form(a);
            if matches!(a, Expr::FunctionCall { name, .. } if name == "Plus")
              || matches!(a, Expr::BinaryOp { op: BinaryOperator::Plus | BinaryOperator::Minus, .. })
              || matches!(a, Expr::FunctionCall { name, args } if name == "Complex"
                && args.len() == 2
                && !matches!((&args[0], &args[1]), (Expr::Integer(0), Expr::Integer(1))))
            {
              format!("({s})")
            } else { s }
          }).collect();
          parts.join("*")
        }
      } else {
        // String-free Times: expr_to_output renders the structure (fractions,
        // Power, Sqrt, base parens) in 1D; the IN_TRUE_INPUT_FORM flag drives
        // format_expr's imaginary-coefficient parenthesisation so the result
        // is the InputForm `(-I)*x` / `(I/2)*x` rather than `-I*x` / `I/2*x`.
        expr_to_output(expr)
      }
    }
    // Image: produce NumericArray InputForm
    Expr::Image {
      color_space: _,
      width,
      height,
      channels,
      data,
      image_type,
    } => {
      let type_str = match image_type {
        ImageType::Bit => "Bit",
        ImageType::Byte => "UnsignedInteger8",
        ImageType::Bit16 => "UnsignedInteger16",
        ImageType::Real32 => "Real32",
        ImageType::Real64 => "Real64",
      };
      let interleaving = if *channels == 1 { "None" } else { "True" };

      // Build nested list representation
      let w = *width as usize;
      let h = *height as usize;
      let ch = *channels as usize;

      let mut rows = Vec::with_capacity(h);
      for y in 0..h {
        if ch == 1 {
          // Grayscale: {{v, v, v}, {v, v, v}}
          let mut row_vals = Vec::with_capacity(w);
          for x in 0..w {
            let v = data[y * w + x];
            row_vals.push(format_image_value(v, *image_type));
          }
          rows.push(format!("{{{}}}", row_vals.join(", ")));
        } else {
          // Color: {{{r, g, b}, {r, g, b}}, ...}
          let mut pixels = Vec::with_capacity(w);
          for x in 0..w {
            let base = (y * w + x) * ch;
            let mut comps = Vec::with_capacity(ch);
            for c in 0..ch {
              let v = data[base + c];
              comps.push(format_image_value(v, *image_type));
            }
            pixels.push(format!("{{{}}}", comps.join(", ")));
          }
          rows.push(format!("{{{}}}", pixels.join(", ")));
        }
      }
      let array_data = format!("{{{}}}", rows.join(", "));

      format!(
        "Image[NumericArray[{array_data}, \"{type_str}\"], \"{type_str}\", ColorSpace -> Automatic, Interleaving -> {interleaving}]"
      )
    }

    // CurriedCall: display as nested calls f[a][b, c] using InputForm.
    // Wrap the head in parens whenever it would otherwise re-associate
    // with the trailing `[args]` on re-parse — e.g. `(s:A[x])[t]` must
    // not collapse to `s:A[x][t]`, and `(x_A /; u > 0)[p]` must keep
    // its `(... /; ...)` group. Mirrors the InputForm logic in
    // format_expr's CurriedCall arm above.
    Expr::CurriedCall { func, args } => {
      let args_str: Vec<String> = args.iter().map(expr_to_input_form).collect();
      let func_str = expr_to_input_form(func);
      let needs_parens = curried_head_needs_parens(func.as_ref());
      let func_display = if needs_parens {
        format!("({func_str})")
      } else {
        func_str
      };
      format!("{}[{}]", func_display, args_str.join(", "))
    }
    // Part: flatten nested Part into a single [[i, j, k]] notation, keeping
    // string bases/indices quoted (InputForm). Mirrors the format_expr arm.
    Expr::Part { expr: e, index } => {
      let mut indices = vec![expr_to_part_index_string(index, ExprForm::Input)];
      let mut base = e.as_ref();
      while let Expr::Part {
        expr: inner_expr,
        index: inner_index,
      } = base
      {
        indices.push(expr_to_part_index_string(inner_index, ExprForm::Input));
        base = inner_expr.as_ref();
      }
      indices.reverse();
      format_part_brackets(base, &expr_to_input_form(base), &indices)
    }
    // For all other cases (infix operators, simple literals), delegate to expr_to_output
    _ => expr_to_output(expr),
  }
}

/// How a call's head is written when the head is itself a pattern.
///
/// A named head pattern is parenthesized — `(F_)[u_, t_]` — so the `_` cannot
/// be read as part of the following bracket group. An anonymous `_[u_]` needs
/// no parens, and an ordinary symbol head is written as it stands.
fn call_head_display(name: &str) -> std::borrow::Cow<'_, str> {
  match name.split_once('_') {
    Some((var, _)) if !var.is_empty() => format!("({name})").into(),
    _ => name.into(),
  }
}

/// Compute the dimensions of a (possibly nested) list payload of a
/// NumericArray, walking into the FIRST element of each level. Empty
/// lists terminate the descent. Used by OutputForm rendering of
/// NumericArray[<dim>, type].
fn numeric_array_dims(payload: &Expr) -> Vec<usize> {
  let mut dims = Vec::new();
  let mut cur = payload;
  while let Expr::List(items) = cur {
    dims.push(items.len());
    if items.is_empty() {
      break;
    }
    cur = &items[0];
  }
  dims
}

/// Format an image pixel value with proper precision.
/// For Real32 images, values are displayed with f32 precision (cast to f32 then back to f64).
fn format_image_value(v: f64, image_type: ImageType) -> String {
  match image_type {
    ImageType::Bit => {
      format!("{}", v.round() as i64)
    }
    ImageType::Real32 => {
      let f32_val = v as f32;
      let f64_val = f32_val as f64;
      format_real(f64_val)
    }
    ImageType::Byte => {
      format!("{}", (v * 255.0).round() as u8)
    }
    ImageType::Bit16 => {
      format!("{}", (v * 65535.0).round() as u16)
    }
    ImageType::Real64 => format_real(v),
  }
}

/// Parse a string into an Expr AST.
/// This is used when we need to convert external string input to AST form.
pub fn string_to_expr(s: &str) -> Result<Expr, crate::InterpreterError> {
  let trimmed = s.trim();

  // Handle empty string - return as empty quoted string literal
  if trimmed.is_empty() {
    return Ok(Expr::Raw(String::new()));
  }

  // Fast path for simple literals
  if let Ok(n) = trimmed.parse::<i128>() {
    return Ok(Expr::Integer(n));
  }
  // i128 overflow: try BigInteger before falling through to f64. Without
  // this branch a stored value like `2^200` round-trips through
  // `Set` (which stringifies the BigInteger then re-parses it from
  // `StoredValue::Raw`) and lossily collapses to a `Real` — which
  // breaks exact integer comparisons such as `2^200 < 2^200 + 1`.
  if !trimmed.contains(['.', 'e', 'E', '*', '/']) {
    let raw = trimmed.strip_prefix('+').unwrap_or(trimmed);
    let digits = raw.strip_prefix('-').unwrap_or(raw);
    if !digits.is_empty()
      && digits.chars().all(|c| c.is_ascii_digit())
      && let Ok(n) = trimmed.parse::<BigInt>()
    {
      return Ok(Expr::BigInteger(n));
    }
  }
  if let Ok(f) = trimmed.parse::<f64>() {
    return Ok(Expr::Real(f));
  }

  // Check for quoted string — but only when the entire input is one literal,
  // not e.g. `"a" -> "n"` which also starts and ends with `"`.
  if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2 {
    let bytes = trimmed.as_bytes();
    let mut is_single_literal = true;
    let mut i = 1;
    while i < bytes.len() - 1 {
      match bytes[i] {
        b'\\' if i + 1 < bytes.len() - 1 => i += 2,
        b'"' => {
          // Unescaped quote before the final char — input is not a single literal
          is_single_literal = false;
          break;
        }
        _ => i += 1,
      }
    }
    if is_single_literal {
      let inner = &trimmed[1..trimmed.len() - 1];
      return Ok(Expr::String(inner.to_string()));
    }
  }

  // Check for simple identifier
  if !trimmed.is_empty()
    && trimmed
      .chars()
      .next()
      .is_some_and(|c| c.is_ascii_alphabetic() || c == '$')
    && trimmed
      .chars()
      .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$')
  {
    return Ok(Expr::Identifier(trimmed.to_string()));
  }

  // Check for slot
  if trimmed == "#" {
    return Ok(Expr::Slot(1));
  }
  if trimmed.starts_with('#')
    && trimmed.len() > 1
    && let Ok(n) = trimmed[1..].parse::<usize>()
  {
    return Ok(Expr::Slot(n));
  }

  // Parse using pest
  let pairs = crate::parse(trimmed)?;
  let mut pairs_iter = pairs.into_iter();
  let program = pairs_iter
    .next()
    .ok_or(crate::InterpreterError::EmptyInput)?;

  // Any statement node, not just `Expression`: a span (`1 ;; 4`), a
  // definition or an upvalue is a statement of its own, and returning the
  // unparsed text for one meant evaluating it re-parsed to the same `Raw`
  // and spun forever.
  for node in program.into_inner() {
    if crate::is_statement_rule(node.as_rule()) {
      return Ok(pair_to_expr(node));
    }
  }

  // Fallback to Raw
  Ok(Expr::Raw(trimmed.to_string()))
}

/// Substitute slots (#, #1, #2, etc.) in an expression with values.
/// values[0] replaces #1 (or #), values[1] replaces #2, etc.
/// Substitute slots in a list of expressions, expanding SlotSequence into multiple args.
fn substitute_slots_expand(exprs: &[Expr], values: &[Expr]) -> Vec<Expr> {
  let mut result = Vec::new();
  for e in exprs {
    let substituted = substitute_slots(e, values);
    // Flatten Sequence[...] into the argument list
    if let Expr::FunctionCall { name, args } = &substituted
      && name == "Sequence"
    {
      result.extend(args.iter().cloned());
      continue;
    }
    result.push(substituted);
  }
  result
}

/// Whether a curried call's head has to be parenthesized so that `[args]`
/// does not re-associate when the printed form is read back.
///
/// Heads that print as an atom or as a self-delimiting `f[...]` need nothing;
/// heads that print in operator form (`u -> v`, `u ;; v`, `u!`, `u_`) would
/// otherwise bind the trailing bracket to their last operand.
fn curried_head_needs_parens(func: &Expr) -> bool {
  prints_as_not(func)
    || matches!(
      func,
      Expr::Function { .. }
        | Expr::PatternOptional { .. }
        | Expr::BinaryOp { .. }
        | Expr::UnaryOp { .. }
        | Expr::Comparison { .. }
        | Expr::Rule { .. }
        | Expr::RuleDelayed { .. }
    )
    || matches!(
      // A named pattern would re-parse as a pattern with a head (`u_[x]` is
      // `Pattern[u, Blank[x]]`), so it needs parens; a bare `_[x]` does not.
      func,
      Expr::Pattern { name, .. } if !name.is_empty()
    )
    || matches!(
      func,
      Expr::FunctionCall { name, args }
        if match name.as_str() {
          // Infix operator forms: `u + v`, `u -> v`, `u ;; v`, …
          "Plus" | "Times" | "Power" | "Pattern" | "Condition" | "Rule"
          | "RuleDelayed" | "And" | "Or" | "Alternatives" | "Span"
          | "Composition" | "RightComposition" => args.len() >= 2,
          // Prefix / postfix operator forms: `u!`, `u++`, `u..`. `Optional`
          // keeps its function spelling but still takes parens, matching
          // wolframscript.
          "Optional" | "Factorial" | "Factorial2" | "Increment" | "Decrement"
          | "PreIncrement" | "PreDecrement" | "Repeated" | "RepeatedNull" =>
            args.len() == 1,
          _ => false,
        }
    )
}

pub fn substitute_slots(expr: &Expr, values: &[Expr]) -> Expr {
  substitute_slots_impl(expr, values)
}

/// Returns `true` if `expr` contains a `Slot(0)` / `Slot[0]` reference
/// (i.e. uses `#0` for anonymous-function self-reference). Used as a
/// fast pre-check before calling `substitute_slot_zero_with_self`, which
/// would otherwise walk and clone the entire tree for nothing.
pub fn contains_slot_zero(expr: &Expr) -> bool {
  match expr {
    Expr::Slot(0) => true,
    Expr::Slot(_) | Expr::SlotSequence(_) => false,
    Expr::FunctionCall { name, args } if name == "Slot" && args.len() == 1 => {
      matches!(&args[0], Expr::Integer(0))
    }
    Expr::FunctionCall { args, .. } => args.iter().any(contains_slot_zero),
    Expr::CurriedCall { func, args } => {
      contains_slot_zero(func) || args.iter().any(contains_slot_zero)
    }
    Expr::List(items) => items.iter().any(contains_slot_zero),
    Expr::BinaryOp { left, right, .. } => {
      contains_slot_zero(left) || contains_slot_zero(right)
    }
    Expr::UnaryOp { operand, .. } => contains_slot_zero(operand),
    Expr::Comparison { operands, .. } => {
      operands.iter().any(contains_slot_zero)
    }
    Expr::CompoundExpr(items) => items.iter().any(contains_slot_zero),
    Expr::Association(items) => items
      .iter()
      .any(|(k, v)| contains_slot_zero(k) || contains_slot_zero(v)),
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => contains_slot_zero(pattern) || contains_slot_zero(replacement),
    // Application / replacement / mapping operators all carry sub-expressions
    // that can hold `#0` (e.g. the `@` in `#0@Quotient[#, 3]`). Without these
    // arms the self-substitution pass is skipped and `#0` is wrongly filled
    // with the first argument.
    Expr::PrefixApply { func, arg } => {
      contains_slot_zero(func) || contains_slot_zero(arg)
    }
    Expr::Postfix { expr: e, func } => {
      contains_slot_zero(e) || contains_slot_zero(func)
    }
    Expr::Map { func, list }
    | Expr::Apply { func, list }
    | Expr::MapApply { func, list } => {
      contains_slot_zero(func) || contains_slot_zero(list)
    }
    Expr::ReplaceAll { expr: e, rules }
    | Expr::ReplaceRepeated { expr: e, rules } => {
      contains_slot_zero(e) || contains_slot_zero(rules)
    }
    Expr::Part { expr: e, index } => {
      contains_slot_zero(e) || contains_slot_zero(index)
    }
    // Nested Function/NamedFunction bodies introduce their own #0 scope.
    Expr::Function { .. } | Expr::NamedFunction { .. } => false,
    _ => false,
  }
}

/// Collect the keys of every named slot (`#key` / `Slot["key"]`) inside
/// `expr` in occurrence order, without descending into nested Function
/// bodies (their slots bind to the inner function). Used by anonymous
/// function application to emit Function::slot1 / Function::slota messages.
pub fn collect_named_slot_keys(expr: &Expr, out: &mut Vec<String>) {
  match expr {
    Expr::FunctionCall { name, args } if name == "Slot" && args.len() == 1 => {
      if let Expr::String(key) = &args[0] {
        out.push(key.clone());
      }
    }
    Expr::FunctionCall { args, .. } => {
      for a in args {
        collect_named_slot_keys(a, out);
      }
    }
    Expr::CurriedCall { func, args } => {
      collect_named_slot_keys(func, out);
      for a in args {
        collect_named_slot_keys(a, out);
      }
    }
    Expr::List(items) => {
      for e in items {
        collect_named_slot_keys(e, out);
      }
    }
    Expr::CompoundExpr(items) => {
      for e in items {
        collect_named_slot_keys(e, out);
      }
    }
    Expr::BinaryOp { left, right, .. } => {
      collect_named_slot_keys(left, out);
      collect_named_slot_keys(right, out);
    }
    Expr::UnaryOp { operand, .. } => collect_named_slot_keys(operand, out),
    Expr::Comparison { operands, .. } => {
      for e in operands {
        collect_named_slot_keys(e, out);
      }
    }
    Expr::Association(items) => {
      for (k, v) in items {
        collect_named_slot_keys(k, out);
        collect_named_slot_keys(v, out);
      }
    }
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      collect_named_slot_keys(pattern, out);
      collect_named_slot_keys(replacement, out);
    }
    Expr::Part { expr: e, index } => {
      collect_named_slot_keys(e, out);
      collect_named_slot_keys(index, out);
    }
    // Nested Function/NamedFunction bodies introduce their own slot scope.
    Expr::Function { .. } | Expr::NamedFunction { .. } => {}
    _ => {}
  }
}

/// Replace every `Slot(0)` / `Slot[0]` inside `expr` with `self_fn`. Used by
/// anonymous-function application to support `#0` self-reference, enabling
/// recursive definitions like `If[#1 <= 1, 1, #1 #0[#1-1]] &`.
pub fn substitute_slot_zero_with_self(expr: &Expr, self_fn: &Expr) -> Expr {
  match expr {
    Expr::Slot(0) => self_fn.clone(),
    Expr::Slot(_) | Expr::SlotSequence(_) => expr.clone(),
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|e| substitute_slot_zero_with_self(e, self_fn))
        .collect(),
    ),
    Expr::FunctionCall { name, args } if name == "Slot" && args.len() == 1 => {
      if matches!(&args[0], Expr::Integer(0)) {
        self_fn.clone()
      } else {
        expr.clone()
      }
    }
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(|e| substitute_slot_zero_with_self(e, self_fn))
        .collect(),
    },
    Expr::CurriedCall { func, args } => Expr::CurriedCall {
      func: Box::new(substitute_slot_zero_with_self(func, self_fn)),
      args: args
        .iter()
        .map(|e| substitute_slot_zero_with_self(e, self_fn))
        .collect(),
    },
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(substitute_slot_zero_with_self(left, self_fn)),
      right: Box::new(substitute_slot_zero_with_self(right, self_fn)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(substitute_slot_zero_with_self(operand, self_fn)),
    },
    Expr::Comparison {
      operands,
      operators,
    } => Expr::Comparison {
      operands: operands
        .iter()
        .map(|e| substitute_slot_zero_with_self(e, self_fn))
        .collect(),
      operators: operators.clone(),
    },
    Expr::CompoundExpr(items) => Expr::CompoundExpr(
      items
        .iter()
        .map(|e| substitute_slot_zero_with_self(e, self_fn))
        .collect(),
    ),
    // Application / replacement / mapping operators carry sub-expressions that
    // can hold `#0` (mirrors the same arms in contains_slot_zero so the two
    // stay in lockstep).
    Expr::PrefixApply { func, arg } => Expr::PrefixApply {
      func: Box::new(substitute_slot_zero_with_self(func, self_fn)),
      arg: Box::new(substitute_slot_zero_with_self(arg, self_fn)),
    },
    Expr::Postfix { expr: e, func } => Expr::Postfix {
      expr: Box::new(substitute_slot_zero_with_self(e, self_fn)),
      func: Box::new(substitute_slot_zero_with_self(func, self_fn)),
    },
    Expr::Map { func, list } => Expr::Map {
      func: Box::new(substitute_slot_zero_with_self(func, self_fn)),
      list: Box::new(substitute_slot_zero_with_self(list, self_fn)),
    },
    Expr::Apply { func, list } => Expr::Apply {
      func: Box::new(substitute_slot_zero_with_self(func, self_fn)),
      list: Box::new(substitute_slot_zero_with_self(list, self_fn)),
    },
    Expr::MapApply { func, list } => Expr::MapApply {
      func: Box::new(substitute_slot_zero_with_self(func, self_fn)),
      list: Box::new(substitute_slot_zero_with_self(list, self_fn)),
    },
    Expr::ReplaceAll { expr: e, rules } => Expr::ReplaceAll {
      expr: Box::new(substitute_slot_zero_with_self(e, self_fn)),
      rules: Box::new(substitute_slot_zero_with_self(rules, self_fn)),
    },
    Expr::ReplaceRepeated { expr: e, rules } => Expr::ReplaceRepeated {
      expr: Box::new(substitute_slot_zero_with_self(e, self_fn)),
      rules: Box::new(substitute_slot_zero_with_self(rules, self_fn)),
    },
    Expr::Part { expr: e, index } => Expr::Part {
      expr: Box::new(substitute_slot_zero_with_self(e, self_fn)),
      index: Box::new(substitute_slot_zero_with_self(index, self_fn)),
    },
    Expr::Rule {
      pattern,
      replacement,
    } => Expr::Rule {
      pattern: Box::new(substitute_slot_zero_with_self(pattern, self_fn)),
      replacement: Box::new(substitute_slot_zero_with_self(
        replacement,
        self_fn,
      )),
    },
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => Expr::RuleDelayed {
      pattern: Box::new(substitute_slot_zero_with_self(pattern, self_fn)),
      replacement: Box::new(substitute_slot_zero_with_self(
        replacement,
        self_fn,
      )),
    },
    // Don't recurse into nested Function bodies — inner #0 refers to that
    // inner function, not this one.
    Expr::Function { .. } | Expr::NamedFunction { .. } => expr.clone(),
    _ => expr.clone(),
  }
}

fn substitute_slots_impl(expr: &Expr, values: &[Expr]) -> Expr {
  match expr {
    Expr::Slot(n) => {
      let index = if *n == 0 { 0 } else { n - 1 };
      if index < values.len() {
        values[index].clone()
      } else {
        expr.clone()
      }
    }
    Expr::SlotSequence(n) => {
      let start = if *n == 0 { 0 } else { n - 1 };
      if start < values.len() {
        let seq: Vec<Expr> = values[start..].to_vec();
        Expr::FunctionCall {
          name: "Sequence".to_string(),
          args: seq.into(),
        }
      } else {
        Expr::FunctionCall {
          name: "Sequence".to_string(),
          args: vec![].into(),
        }
      }
    }
    Expr::List(items) => {
      Expr::List(substitute_slots_expand(items, values).into())
    }
    Expr::FunctionCall { name, args } if name == "Slot" && args.len() == 1 => {
      if let Expr::Integer(n) = &args[0] {
        let index = if *n <= 0 { 0 } else { (*n as usize) - 1 };
        if index < values.len() {
          values[index].clone()
        } else {
          expr.clone()
        }
      } else if let Expr::String(key) = &args[0] {
        // Named slot #key fills from the keys of an Association first
        // argument; otherwise it stays unfilled (the application site emits
        // Function::slot1 / Function::slota messages).
        if let Some(Expr::Association(items)) = values.first()
          && let Some((_, v)) = items
            .iter()
            .find(|(k, _)| matches!(k, Expr::String(s) if s == key))
        {
          v.clone()
        } else {
          expr.clone()
        }
      } else {
        expr.clone()
      }
    }
    Expr::FunctionCall { name, args }
      if name == "SlotSequence" && args.len() == 1 =>
    {
      if let Expr::Integer(n) = &args[0] {
        let start = if *n <= 0 { 0 } else { (*n as usize) - 1 };
        if start < values.len() {
          let seq: Vec<Expr> = values[start..].to_vec();
          Expr::FunctionCall {
            name: "Sequence".to_string(),
            args: seq.into(),
          }
        } else {
          Expr::FunctionCall {
            name: "Sequence".to_string(),
            args: vec![].into(),
          }
        }
      } else {
        expr.clone()
      }
    }
    // `Unevaluated`, `Hold`, `HoldComplete` and other SequenceHold /
    // HoldAllComplete heads must keep a `Sequence[...]` produced by slot
    // substitution as a single argument rather than splicing it into their
    // own argument list — splicing is an evaluation-time rule that doesn't
    // apply to their (permanently unevaluated) contents. This matters for
    // the common Demonstrations idiom
    // `(If[cond, Unevaluated[Sequence[a, b]], {}]) & [x]`, where splicing
    // here would turn `Unevaluated[Sequence[a, b]]` into the wrong
    // `Unevaluated[a, b]`.
    Expr::FunctionCall { name, args }
      if crate::evaluator::listable::has_sequence_hold(name) =>
    {
      Expr::FunctionCall {
        name: name.clone(),
        args: args
          .iter()
          .map(|a| substitute_slots(a, values))
          .collect::<Vec<_>>()
          .into(),
      }
    }
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: substitute_slots_expand(args, values).into(),
    },
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(substitute_slots(left, values)),
      right: Box::new(substitute_slots(right, values)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(substitute_slots(operand, values)),
    },
    Expr::Comparison {
      operands,
      operators,
    } => Expr::Comparison {
      operands: operands
        .iter()
        .map(|e| substitute_slots(e, values))
        .collect(),
      operators: operators.clone(),
    },
    Expr::CompoundExpr(exprs) => Expr::CompoundExpr(
      exprs.iter().map(|e| substitute_slots(e, values)).collect(),
    ),
    Expr::Association(items) => Expr::Association(
      items
        .iter()
        .map(|(k, v)| {
          (substitute_slots(k, values), substitute_slots(v, values))
        })
        .collect(),
    ),
    Expr::Rule {
      pattern,
      replacement,
    } => Expr::Rule {
      pattern: Box::new(substitute_slots(pattern, values)),
      replacement: Box::new(substitute_slots(replacement, values)),
    },
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => Expr::RuleDelayed {
      pattern: Box::new(substitute_slots(pattern, values)),
      replacement: Box::new(substitute_slots(replacement, values)),
    },
    Expr::ReplaceAll { expr: e, rules } => Expr::ReplaceAll {
      expr: Box::new(substitute_slots(e, values)),
      rules: Box::new(substitute_slots(rules, values)),
    },
    Expr::ReplaceRepeated { expr: e, rules } => Expr::ReplaceRepeated {
      expr: Box::new(substitute_slots(e, values)),
      rules: Box::new(substitute_slots(rules, values)),
    },
    Expr::Map { func, list } => Expr::Map {
      func: Box::new(substitute_slots(func, values)),
      list: Box::new(substitute_slots(list, values)),
    },
    Expr::Apply { func, list } => Expr::Apply {
      func: Box::new(substitute_slots(func, values)),
      list: Box::new(substitute_slots(list, values)),
    },
    Expr::MapApply { func, list } => Expr::MapApply {
      func: Box::new(substitute_slots(func, values)),
      list: Box::new(substitute_slots(list, values)),
    },
    Expr::PrefixApply { func, arg } => Expr::PrefixApply {
      func: Box::new(substitute_slots(func, values)),
      arg: Box::new(substitute_slots(arg, values)),
    },
    Expr::Postfix { expr: e, func } => Expr::Postfix {
      expr: Box::new(substitute_slots(e, values)),
      func: Box::new(substitute_slots(func, values)),
    },
    Expr::Part { expr: e, index } => Expr::Part {
      expr: Box::new(substitute_slots(e, values)),
      index: Box::new(substitute_slots(index, values)),
    },
    Expr::Function { body } => {
      // Do NOT substitute slots inside nested Function bodies.
      // Slots (#, #2, etc.) bind to the innermost & (Function).
      Expr::Function { body: body.clone() }
    }
    Expr::NamedFunction {
      params,
      body,
      bracketed,
    } => {
      // Named functions don't use slots, so no substitution needed
      Expr::NamedFunction {
        params: params.clone(),
        body: body.clone(),
        bracketed: *bracketed,
      }
    }
    Expr::PatternOptional {
      name,
      head,
      default,
    } => Expr::PatternOptional {
      name: name.clone(),
      head: head.clone(),
      default: default
        .as_ref()
        .map(|d| Box::new(substitute_slots(d, values))),
    },
    Expr::PatternTest {
      name,
      head,
      blank_type,
      test,
    } => Expr::PatternTest {
      name: name.clone(),
      head: head.clone(),
      blank_type: *blank_type,
      test: Box::new(substitute_slots(test, values)),
    },
    Expr::CurriedCall { func, args } => Expr::CurriedCall {
      func: Box::new(substitute_slots(func, values)),
      args: substitute_slots_expand(args, values),
    },
    // Atoms that don't contain slots
    _ => expr.clone(),
  }
}

/// Collect every identifier that appears as `Expr::Identifier(name)` inside
/// `expr`, without attending to scoping. Used for capture-avoiding
/// substitution: before substituting `value` into a body that binds a
/// parameter P, we need to know whether P appears as a name somewhere in
/// `value` — if so, the binding must first be alpha-renamed.
pub fn collect_identifier_names<S: std::hash::BuildHasher>(
  expr: &Expr,
  out: &mut std::collections::HashSet<String, S>,
) {
  match expr {
    Expr::Identifier(name) => {
      out.insert(name.clone());
    }
    Expr::FunctionCall { args, .. } => {
      for a in args {
        collect_identifier_names(a, out);
      }
    }
    Expr::List(items) => {
      for it in items {
        collect_identifier_names(it, out);
      }
    }
    Expr::CompoundExpr(items) => {
      for it in items {
        collect_identifier_names(it, out);
      }
    }
    Expr::BinaryOp { left, right, .. } => {
      collect_identifier_names(left, out);
      collect_identifier_names(right, out);
    }
    Expr::UnaryOp { operand, .. } => collect_identifier_names(operand, out),
    Expr::Function { body } => collect_identifier_names(body, out),
    Expr::NamedFunction { body, .. } => collect_identifier_names(body, out),
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      collect_identifier_names(pattern, out);
      collect_identifier_names(replacement, out);
    }
    Expr::Comparison { operands, .. } => {
      for op in operands {
        collect_identifier_names(op, out);
      }
    }
    Expr::Part { expr: e, index } => {
      collect_identifier_names(e, out);
      collect_identifier_names(index, out);
    }
    _ => {}
  }
}

/// Substitute a variable name with a value in an expression.
///
/// Capture-avoiding, like [`substitute_variables`] it delegates to: an inner
/// `Function`/`With`/`Module` binding the same name shadows the substitution.
/// That is what keeps an iterator variable out of a nested binder, e.g.
/// `Table[Hold[Module[{k}, k]], {k, 1, 2}]` leaves both `k`s alone.
pub fn substitute_variable(expr: &Expr, var_name: &str, value: &Expr) -> Expr {
  substitute_variables(expr, &[(var_name, value)])
}

/// Rename every occurrence of a symbol — including function-call heads,
/// which stay plain FunctionCalls under the new name — used by Module's
/// lexical renaming of its local variables.
pub fn rename_symbol(expr: &Expr, var_name: &str, new_name: &str) -> Expr {
  substitute_variable_impl(
    expr,
    var_name,
    &Expr::Identifier(new_name.to_string()),
    true,
  )
}

fn substitute_variable_impl(
  expr: &Expr,
  var_name: &str,
  value: &Expr,
  rename_heads: bool,
) -> Expr {
  match expr {
    Expr::Identifier(name) if name == var_name => value.clone(),
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|e| substitute_variable_impl(e, var_name, value, rename_heads))
        .collect(),
    ),
    Expr::FunctionCall { name, args } => {
      let new_args: Vec<Expr> = args
        .iter()
        .map(|e| substitute_variable_impl(e, var_name, value, rename_heads))
        .collect();
      if name == var_name {
        // The function name matches the variable being substituted. When
        // renaming (Module locals), keep a plain FunctionCall under the
        // new name so definitions like f$1[x_] := ... still work;
        // otherwise transform into a CurriedCall applying the value.
        if rename_heads && let Expr::Identifier(new_name) = value {
          return Expr::FunctionCall {
            name: new_name.clone(),
            args: new_args.into(),
          };
        }
        Expr::CurriedCall {
          func: Box::new(value.clone()),
          args: new_args,
        }
      } else {
        Expr::FunctionCall {
          name: name.clone(),
          args: new_args.into(),
        }
      }
    }
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(substitute_variable_impl(
        left,
        var_name,
        value,
        rename_heads,
      )),
      right: Box::new(substitute_variable_impl(
        right,
        var_name,
        value,
        rename_heads,
      )),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(substitute_variable_impl(
        operand,
        var_name,
        value,
        rename_heads,
      )),
    },
    Expr::Comparison {
      operands,
      operators,
    } => Expr::Comparison {
      operands: operands
        .iter()
        .map(|e| substitute_variable_impl(e, var_name, value, rename_heads))
        .collect(),
      operators: operators.clone(),
    },
    Expr::CompoundExpr(exprs) => Expr::CompoundExpr(
      exprs
        .iter()
        .map(|e| substitute_variable_impl(e, var_name, value, rename_heads))
        .collect(),
    ),
    Expr::Association(items) => Expr::Association(
      items
        .iter()
        .map(|(k, v)| {
          (
            substitute_variable(k, var_name, value),
            substitute_variable(v, var_name, value),
          )
        })
        .collect(),
    ),
    Expr::Rule {
      pattern,
      replacement,
    } => Expr::Rule {
      pattern: Box::new(substitute_variable_impl(
        pattern,
        var_name,
        value,
        rename_heads,
      )),
      replacement: Box::new(substitute_variable_impl(
        replacement,
        var_name,
        value,
        rename_heads,
      )),
    },
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => Expr::RuleDelayed {
      pattern: Box::new(substitute_variable_impl(
        pattern,
        var_name,
        value,
        rename_heads,
      )),
      replacement: Box::new(substitute_variable_impl(
        replacement,
        var_name,
        value,
        rename_heads,
      )),
    },
    Expr::ReplaceAll { expr: e, rules } => Expr::ReplaceAll {
      expr: Box::new(substitute_variable_impl(
        e,
        var_name,
        value,
        rename_heads,
      )),
      rules: Box::new(substitute_variable_impl(
        rules,
        var_name,
        value,
        rename_heads,
      )),
    },
    Expr::ReplaceRepeated { expr: e, rules } => Expr::ReplaceRepeated {
      expr: Box::new(substitute_variable_impl(
        e,
        var_name,
        value,
        rename_heads,
      )),
      rules: Box::new(substitute_variable_impl(
        rules,
        var_name,
        value,
        rename_heads,
      )),
    },
    Expr::Map { func, list } => Expr::Map {
      func: Box::new(substitute_variable_impl(
        func,
        var_name,
        value,
        rename_heads,
      )),
      list: Box::new(substitute_variable_impl(
        list,
        var_name,
        value,
        rename_heads,
      )),
    },
    Expr::Apply { func, list } => Expr::Apply {
      func: Box::new(substitute_variable_impl(
        func,
        var_name,
        value,
        rename_heads,
      )),
      list: Box::new(substitute_variable_impl(
        list,
        var_name,
        value,
        rename_heads,
      )),
    },
    Expr::MapApply { func, list } => Expr::MapApply {
      func: Box::new(substitute_variable_impl(
        func,
        var_name,
        value,
        rename_heads,
      )),
      list: Box::new(substitute_variable_impl(
        list,
        var_name,
        value,
        rename_heads,
      )),
    },
    Expr::PrefixApply { func, arg } => Expr::PrefixApply {
      func: Box::new(substitute_variable_impl(
        func,
        var_name,
        value,
        rename_heads,
      )),
      arg: Box::new(substitute_variable_impl(
        arg,
        var_name,
        value,
        rename_heads,
      )),
    },
    Expr::Postfix { expr: e, func } => Expr::Postfix {
      expr: Box::new(substitute_variable_impl(
        e,
        var_name,
        value,
        rename_heads,
      )),
      func: Box::new(substitute_variable_impl(
        func,
        var_name,
        value,
        rename_heads,
      )),
    },
    Expr::Part { expr: e, index } => Expr::Part {
      expr: Box::new(substitute_variable_impl(
        e,
        var_name,
        value,
        rename_heads,
      )),
      index: Box::new(substitute_variable_impl(
        index,
        var_name,
        value,
        rename_heads,
      )),
    },
    Expr::Function { body } => Expr::Function {
      body: Box::new(substitute_variable_impl(
        body,
        var_name,
        value,
        rename_heads,
      )),
    },
    Expr::NamedFunction {
      params,
      body,
      bracketed,
    } => {
      // Don't substitute if var_name is one of the function's own parameters
      // (they are locally scoped)
      if params.contains(&var_name.to_string()) {
        Expr::NamedFunction {
          params: params.clone(),
          body: body.clone(),
          bracketed: *bracketed,
        }
      } else {
        // Capture-avoiding substitution: if any of this Function's params
        // appears as an identifier in `value`, naive substitution would
        // capture the new occurrence. Rename each such param to a fresh
        // name (with a `$` suffix) before substituting, matching Wolfram's
        // alpha-renaming behaviour: `Function[{x}, Function[{y}, f[x,y]]][y]`
        // ⇒ `Function[{y$}, f[y, y$]]`, not `Function[{y}, f[y, y]]`.
        let mut value_names = std::collections::HashSet::new();
        collect_identifier_names(value, &mut value_names);
        let mut new_params = Vec::with_capacity(params.len());
        let mut new_body = (**body).clone();
        for param in params {
          if value_names.contains(param) {
            let fresh = format!("{param}$");
            new_body = substitute_variable_impl(
              &new_body,
              param,
              &Expr::Identifier(fresh.clone()),
              rename_heads,
            );
            new_params.push(fresh);
          } else {
            new_params.push(param.clone());
          }
        }
        Expr::NamedFunction {
          params: new_params,
          body: Box::new(substitute_variable_impl(
            &new_body,
            var_name,
            value,
            rename_heads,
          )),
          bracketed: *bracketed,
        }
      }
    }
    // When renaming (Module locals), a pattern variable or pattern head
    // matching the local is renamed along with the body, keeping inner
    // definitions like `r[s_] := s^2` consistent after `s` becomes `s$n`
    // (Wolfram renames Pattern names the same way). Plain substitution
    // leaves pattern names alone — they shadow the substituted variable.
    Expr::Pattern {
      name,
      head,
      blank_type,
    } if rename_heads
      && (name == var_name || head.as_deref() == Some(var_name)) =>
    {
      match value {
        Expr::Identifier(new_name) => Expr::Pattern {
          name: if name == var_name {
            new_name.clone()
          } else {
            name.clone()
          },
          head: head.as_ref().map(|h| {
            if h == var_name {
              new_name.clone()
            } else {
              h.clone()
            }
          }),
          blank_type: *blank_type,
        },
        _ => expr.clone(),
      }
    }
    Expr::PatternOptional {
      name,
      head,
      default,
    } => {
      let (name, head) = rename_pattern_parts(
        name,
        head.as_deref(),
        var_name,
        value,
        rename_heads,
      );
      Expr::PatternOptional {
        name,
        head,
        default: default.as_ref().map(|d| {
          Box::new(substitute_variable_impl(d, var_name, value, rename_heads))
        }),
      }
    }
    Expr::PatternTest {
      name,
      head,
      blank_type,
      test,
    } => {
      let (name, head) = rename_pattern_parts(
        name,
        head.as_deref(),
        var_name,
        value,
        rename_heads,
      );
      Expr::PatternTest {
        name,
        head,
        blank_type: *blank_type,
        test: Box::new(substitute_variable_impl(
          test,
          var_name,
          value,
          rename_heads,
        )),
      }
    }
    Expr::CurriedCall { func, args } => Expr::CurriedCall {
      func: Box::new(substitute_variable_impl(
        func,
        var_name,
        value,
        rename_heads,
      )),
      args: args
        .iter()
        .map(|e| substitute_variable_impl(e, var_name, value, rename_heads))
        .collect(),
    },
    // Atoms that don't contain the variable
    _ => expr.clone(),
  }
}

/// Shared by the `PatternOptional`/`PatternTest` substitution arms: in
/// rename mode a pattern name (or head) matching the renamed local follows
/// the rename, like `Expr::Pattern`; otherwise both are kept as-is.
fn rename_pattern_parts(
  name: &str,
  head: Option<&str>,
  var_name: &str,
  value: &Expr,
  rename_heads: bool,
) -> (String, Option<String>) {
  if rename_heads && let Expr::Identifier(new_name) = value {
    let renamed_name = if name == var_name {
      new_name.clone()
    } else {
      name.to_string()
    };
    let renamed_head = head.map(|h| {
      if h == var_name {
        new_name.clone()
      } else {
        h.to_string()
      }
    });
    return (renamed_name, renamed_head);
  }
  (name.to_string(), head.map(str::to_string))
}

/// Perform simultaneous substitution of multiple variables.
/// Unlike calling `substitute_variable` in a loop, this replaces all
/// variables in a single pass so that substituted values cannot be
/// accidentally captured by later substitutions.
pub fn substitute_variables(expr: &Expr, bindings: &[(&str, &Expr)]) -> Expr {
  substitute_variables_impl(expr, bindings, false)
}

/// Substitute pattern-matched bindings into a *template*: the right-hand side
/// of a `:=` definition or of a `:>` rule.
///
/// Same as [`substitute_variables`], except that a `Function` parameter which
/// is itself one of the bound names counts as a template slot rather than as a
/// binder. `g[f_, s_] := Function[s, f]` called as `g[expr, s]` therefore
/// yields `Function[s, expr]`, so the `s` inside `expr` really is the pure
/// function's argument. Parameters that are *not* bound still shadow, and are
/// still alpha-renamed when an incoming value would otherwise be captured.
pub fn substitute_pattern_bindings(
  expr: &Expr,
  bindings: &[(&str, &Expr)],
) -> Expr {
  substitute_variables_impl(expr, bindings, true)
}

/// Work out a pure function's parameter list while substituting into it.
///
/// Returns the parameter names to use, the bindings that still reach the body,
/// and the body with any capture-avoiding renames already applied. `None` means
/// nothing reaches the body, so the function is left exactly as it was.
///
/// Two rules decide each parameter's fate:
///   * In template mode a parameter that is itself a bound name is a slot the
///     caller filled in, so it becomes that binding's symbol and its binding
///     keeps reaching the body.
///   * A parameter that stays a binder shadows its own binding, and is renamed
///     to `name$` when an incoming value mentions it — Wolfram's alpha-renaming,
///     e.g. `Function[{x}, Function[{y}, f[x, y]]][y]` ⇒ `Function[{y$}, f[y, y$]]`.
fn resolve_function_params<'a>(
  params: &[String],
  body: &Expr,
  bindings: &[(&'a str, &'a Expr)],
  template: bool,
) -> Option<(Vec<String>, Vec<(&'a str, &'a Expr)>, Expr)> {
  if bindings.is_empty() {
    return None;
  }
  // Parameter names the caller supplied (template mode only). A non-symbol
  // value can't name a parameter, so such a parameter keeps binding.
  let supplied: Vec<Option<String>> = params
    .iter()
    .map(|p| {
      if !template {
        return None;
      }
      bindings
        .iter()
        .find(|(var_name, _)| var_name == p)
        .and_then(|(_, value)| match value {
          Expr::Identifier(n) => Some(n.clone()),
          _ => None,
        })
    })
    .collect();
  let filtered: Vec<(&'a str, &'a Expr)> = bindings
    .iter()
    .filter(|&&(var_name, _)| {
      !params
        .iter()
        .zip(&supplied)
        .any(|(p, s)| s.is_none() && p == var_name)
    })
    .copied()
    .collect();
  if filtered.is_empty() {
    return None;
  }
  let mut value_names = std::collections::HashSet::new();
  for (_, value) in &filtered {
    collect_identifier_names(value, &mut value_names);
  }
  let mut new_params = Vec::with_capacity(params.len());
  let mut new_body = body.clone();
  for (param, supplied_name) in params.iter().zip(&supplied) {
    match supplied_name {
      Some(name) => new_params.push(name.clone()),
      None if value_names.contains(param) => {
        let fresh = format!("{param}$");
        new_body = substitute_variable(
          &new_body,
          param,
          &Expr::Identifier(fresh.clone()),
        );
        new_params.push(fresh);
      }
      None => new_params.push(param.clone()),
    }
  }
  Some((new_params, filtered, new_body))
}

/// The local names a `With`/`Module` variable specification binds.
/// Items that bind nothing (`With[{1 + 1}, …]`) contribute no name; the
/// scoping construct itself reports them.
fn scoping_binder_names(spec: &Expr) -> Vec<String> {
  let Expr::List(items) = spec else {
    return Vec::new();
  };
  items
    .iter()
    .filter_map(|item| match item {
      Expr::FunctionCall {
        name,
        args: set_args,
      } if name == "Set" && set_args.len() == 2 => match &set_args[0] {
        Expr::Identifier(var_name) => Some(var_name.clone()),
        _ => None,
      },
      Expr::Rule { pattern, .. } => match pattern.as_ref() {
        Expr::Identifier(var_name) => Some(var_name.clone()),
        _ => None,
      },
      Expr::Identifier(var_name) => Some(var_name.clone()),
      _ => None,
    })
    .collect()
}

/// Substitute into one local variable specification: the initializers see the
/// enclosing bindings, the bound names themselves are only alpha-renamed.
fn substitute_scoping_spec(
  spec: &Expr,
  bindings: &[(&str, &Expr)],
  renames: &[(String, String)],
) -> Expr {
  let Expr::List(items) = spec else {
    return substitute_variables_impl(spec, bindings, false);
  };
  let renamed = |var_name: &str| {
    renames
      .iter()
      .find(|(old, _)| old == var_name)
      .map_or_else(|| var_name.to_string(), |(_, new)| new.clone())
  };
  Expr::List(
    items
      .iter()
      .map(|item| match item {
        Expr::FunctionCall {
          name,
          args: set_args,
        } if name == "Set"
          && set_args.len() == 2
          && matches!(set_args[0], Expr::Identifier(_)) =>
        {
          let Expr::Identifier(var_name) = &set_args[0] else {
            unreachable!()
          };
          Expr::FunctionCall {
            name: "Set".to_string(),
            args: vec![
              Expr::Identifier(renamed(var_name)),
              substitute_variables_impl(&set_args[1], bindings, false),
            ]
            .into(),
          }
        }
        Expr::Rule {
          pattern,
          replacement,
        } if matches!(pattern.as_ref(), Expr::Identifier(_)) => {
          let Expr::Identifier(var_name) = pattern.as_ref() else {
            unreachable!()
          };
          Expr::Rule {
            pattern: Box::new(Expr::Identifier(renamed(var_name))),
            replacement: Box::new(substitute_variables_impl(
              replacement,
              bindings,
              false,
            )),
          }
        }
        Expr::Identifier(var_name) => Expr::Identifier(renamed(var_name)),
        other => substitute_variables_impl(other, bindings, false),
      })
      .collect(),
  )
}

/// Capture-avoiding substitution into `With[spec, …, body]` /
/// `Module[spec, body]`. Every leading specification of a multi-clause `With`
/// scopes the ones after it, so they are walked in order: each one's
/// initializers are substituted with the bindings that still reach it, its own
/// names then drop out of the bindings, and a name that would capture a value
/// still heading for the body is alpha-renamed to `name$` — Wolfram's own
/// rename, e.g. `With[{x = y}, With[{y = 2}, x + y]]` ⇒ `2 + y`.
fn substitute_into_scoping_construct(
  head: &str,
  args: &[Expr],
  bindings: &[(&str, &Expr)],
) -> Expr {
  // Only With takes more than one specification; Module's extra arguments
  // (DynamicModule's options) are part of the body region.
  let spec_count = if head == "With" { args.len() - 1 } else { 1 };
  let mut active: Vec<(&str, &Expr)> = bindings.to_vec();
  let mut tail: Vec<Expr> = args.to_vec();
  let mut new_args: Vec<Expr> = Vec::with_capacity(args.len());
  for i in 0..spec_count {
    let binder_names = scoping_binder_names(&tail[i]);
    // Bindings that survive this specification: the shadowed ones stop here.
    let inner: Vec<(&str, &Expr)> = active
      .iter()
      .filter(|(var_name, _)| !binder_names.iter().any(|b| b == var_name))
      .copied()
      .collect();
    let mut value_names = std::collections::HashSet::new();
    for (_, value) in &inner {
      collect_identifier_names(value, &mut value_names);
    }
    let renames: Vec<(String, String)> = binder_names
      .iter()
      .filter(|b| value_names.contains(b.as_str()))
      .map(|b| (b.clone(), format!("{b}$")))
      .collect();
    for expr in tail.iter_mut().skip(i + 1) {
      for (old, new) in &renames {
        *expr = rename_symbol(expr, old, new);
      }
    }
    new_args.push(substitute_scoping_spec(&tail[i], &active, &renames));
    active = inner;
  }
  for expr in tail.iter().skip(spec_count) {
    new_args.push(substitute_variables_impl(expr, &active, false));
  }
  Expr::FunctionCall {
    name: head.to_string(),
    args: new_args.into(),
  }
}

/// Statically resolve a `Part[…]` chain built by list-pattern lowering
/// (`build_list_pattern_match`/`collect_element_bindings` in
/// `evaluator::assignment`) down to the literal element it names, using
/// `bindings` to look up the synthetic `_lp{i}` base parameter those
/// functions introduce for an anonymous list-pattern argument (`f[{x_,
/// y_}] := …`).
///
/// A rule's pattern variables bind to the actual matched sub-expressions in
/// Wolfram — `x_` in that example becomes whatever `x` matched, a plain
/// value, never a `Part[…]` access. Woxi represents that binding
/// internally as `Part[_lp0, 1]` so the leaf pattern can be recovered once
/// the whole list argument is known, but that internal encoding must not
/// leak into the substituted body: left as `Part[…]`, a held argument
/// position (`Plot`'s iterator spec, `Hold`, …) would see the accessor
/// instead of the symbol/value it stands for. Resolving it here — before
/// the body is even substituted — closes that seam.
///
/// Returns `None` for anything that isn't one of these synthetic chains;
/// in particular a `Part[…]` the user actually wrote is left alone so it
/// keeps evaluating (or staying held) exactly like today.
fn resolve_synthetic_list_part(
  expr: &Expr,
  bindings: &[(&str, &Expr)],
) -> Option<Expr> {
  match expr {
    Expr::Identifier(name) if name.starts_with("_lp") => bindings
      .iter()
      .find(|&&(var_name, _)| var_name == name)
      .map(|&(_, value)| value.clone()),
    Expr::FunctionCall { name, args } if name == "Part" && args.len() == 2 => {
      let base = resolve_synthetic_list_part(&args[0], bindings)?;
      let Expr::List(items) = &base else {
        return None;
      };
      let Expr::Integer(n) = &args[1] else {
        return None;
      };
      let idx = if *n >= 1 {
        (*n - 1) as usize
      } else if *n < 0 {
        usize::try_from(items.len() as i128 + *n).ok()?
      } else {
        return None;
      };
      items.get(idx).cloned()
    }
    _ => None,
  }
}

fn substitute_variables_impl(
  expr: &Expr,
  bindings: &[(&str, &Expr)],
  template: bool,
) -> Expr {
  if bindings.is_empty() {
    return expr.clone();
  }
  match expr {
    Expr::Identifier(name) => {
      for &(var_name, value) in bindings {
        if name == var_name {
          return value.clone();
        }
      }
      expr.clone()
    }
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|e| substitute_variables_impl(e, bindings, template))
        .collect(),
    ),
    Expr::FunctionCall { name, args } if name == "Part" && args.len() == 2 => {
      match resolve_synthetic_list_part(expr, bindings) {
        Some(resolved) => resolved,
        None => Expr::FunctionCall {
          name: name.clone(),
          args: args
            .iter()
            .map(|e| substitute_variables_impl(e, bindings, template))
            .collect(),
        },
      }
    }
    Expr::FunctionCall { name, args } => {
      // Capture-avoiding substitution into an unevaluated
      // `Function[{params}, body]` FunctionCall. Function has HoldAll, so
      // substituting into an outer body before the inner Function is
      // evaluated leaves its binding as a FunctionCall. Without special
      // handling, `Function[{x}, Function[{y}, f[x,y]]][y]` would substitute
      // x→y in the body and capture the new y reference — producing
      // `Function[{y}, f[y,y]]` instead of Wolfram's `Function[{y$}, f[y, y$]]`.
      if name == "Function" && args.len() == 2 && !bindings.is_empty() {
        let (params_arg, body_arg) = (&args[0], &args[1]);
        let param_names: Option<Vec<String>> = match params_arg {
          Expr::Identifier(n) => Some(vec![n.clone()]),
          Expr::List(items) => items
            .iter()
            .map(|it| {
              if let Expr::Identifier(n) = it {
                Some(n.clone())
              } else {
                None
              }
            })
            .collect::<Option<Vec<_>>>(),
          _ => None,
        };
        if let Some(params) = param_names {
          let Some((new_params, filtered, new_body)) =
            resolve_function_params(&params, body_arg, bindings, template)
          else {
            return expr.clone();
          };
          let substituted_body =
            substitute_variables_impl(&new_body, &filtered, template);
          let new_params_arg = if matches!(params_arg, Expr::List(_)) {
            Expr::List(new_params.into_iter().map(Expr::Identifier).collect())
          } else {
            // Bare identifier form; must still be a list of 1 after rename.
            Expr::Identifier(new_params.into_iter().next().unwrap_or_default())
          };
          return Expr::FunctionCall {
            name: "Function".to_string(),
            args: vec![new_params_arg, substituted_body].into(),
          };
        }
      }
      // `With[{x = v}, …, body]` and `Module[{x = v}, body]` bind their local
      // names lexically, so an outer binding of the same name must not reach
      // the body: `With[{x = 5}, With[{x = x + 1}, x^2]]` is 36, not 25. The
      // initializers still evaluate in the enclosing scope and keep being
      // substituted. Block is deliberately absent — it scopes dynamically and
      // Wolfram substitutes straight through it, binder name included.
      // In template mode the same names are pattern slots the caller filled
      // in, and Wolfram substitutes those through as well
      // (`f[x_] := With[{x = 1}, x]` called as `f[7]` gives `With[{7 = 1}, 7]`).
      if !template
        && matches!(name.as_str(), "With" | "Module" | "DynamicModule")
        && args.len() >= 2
        && matches!(args[0], Expr::List(_))
      {
        return substitute_into_scoping_construct(name, args, bindings);
      }
      let new_args: Vec<Expr> = args
        .iter()
        .map(|e| substitute_variables_impl(e, bindings, template))
        .collect();
      // Check if the function name itself is being substituted
      for &(var_name, value) in bindings {
        if name == var_name {
          // A head that becomes a plain symbol is a call on that symbol,
          // not a curried application of it — `s[k]` with `s` bound to `q`
          // is `q[k]`, which is what carries `q`'s own definitions.
          if let Expr::Identifier(head) = value {
            return Expr::FunctionCall {
              name: head.clone(),
              args: new_args.into(),
            };
          }
          return Expr::CurriedCall {
            func: Box::new(value.clone()),
            args: new_args,
          };
        }
      }
      Expr::FunctionCall {
        name: name.clone(),
        args: new_args.into(),
      }
    }
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(substitute_variables_impl(left, bindings, template)),
      right: Box::new(substitute_variables_impl(right, bindings, template)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(substitute_variables_impl(operand, bindings, template)),
    },
    Expr::Comparison {
      operands,
      operators,
    } => Expr::Comparison {
      operands: operands
        .iter()
        .map(|e| substitute_variables_impl(e, bindings, template))
        .collect(),
      operators: operators.clone(),
    },
    Expr::CompoundExpr(exprs) => Expr::CompoundExpr(
      exprs
        .iter()
        .map(|e| substitute_variables_impl(e, bindings, template))
        .collect(),
    ),
    Expr::Association(items) => Expr::Association(
      items
        .iter()
        .map(|(k, v)| {
          (
            substitute_variables_impl(k, bindings, template),
            substitute_variables_impl(v, bindings, template),
          )
        })
        .collect(),
    ),
    Expr::Rule {
      pattern,
      replacement,
    } => Expr::Rule {
      pattern: Box::new(substitute_variables_impl(pattern, bindings, template)),
      replacement: Box::new(substitute_variables_impl(
        replacement,
        bindings,
        template,
      )),
    },
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => Expr::RuleDelayed {
      pattern: Box::new(substitute_variables_impl(pattern, bindings, template)),
      replacement: Box::new(substitute_variables_impl(
        replacement,
        bindings,
        template,
      )),
    },
    Expr::ReplaceAll { expr: e, rules } => Expr::ReplaceAll {
      expr: Box::new(substitute_variables_impl(e, bindings, template)),
      rules: Box::new(substitute_variables_impl(rules, bindings, template)),
    },
    Expr::ReplaceRepeated { expr: e, rules } => Expr::ReplaceRepeated {
      expr: Box::new(substitute_variables_impl(e, bindings, template)),
      rules: Box::new(substitute_variables_impl(rules, bindings, template)),
    },
    Expr::Map { func, list } => Expr::Map {
      func: Box::new(substitute_variables_impl(func, bindings, template)),
      list: Box::new(substitute_variables_impl(list, bindings, template)),
    },
    Expr::Apply { func, list } => Expr::Apply {
      func: Box::new(substitute_variables_impl(func, bindings, template)),
      list: Box::new(substitute_variables_impl(list, bindings, template)),
    },
    Expr::MapApply { func, list } => Expr::MapApply {
      func: Box::new(substitute_variables_impl(func, bindings, template)),
      list: Box::new(substitute_variables_impl(list, bindings, template)),
    },
    Expr::PrefixApply { func, arg } => Expr::PrefixApply {
      func: Box::new(substitute_variables_impl(func, bindings, template)),
      arg: Box::new(substitute_variables_impl(arg, bindings, template)),
    },
    Expr::Postfix { expr: e, func } => Expr::Postfix {
      expr: Box::new(substitute_variables_impl(e, bindings, template)),
      func: Box::new(substitute_variables_impl(func, bindings, template)),
    },
    Expr::Part { expr: e, index } => Expr::Part {
      expr: Box::new(substitute_variables_impl(e, bindings, template)),
      index: Box::new(substitute_variables_impl(index, bindings, template)),
    },
    Expr::Function { body } => Expr::Function {
      body: Box::new(substitute_variables_impl(body, bindings, template)),
    },
    Expr::NamedFunction {
      params,
      body,
      bracketed,
    } => match resolve_function_params(params, body, bindings, template) {
      None => expr.clone(),
      Some((new_params, filtered, new_body)) => Expr::NamedFunction {
        params: new_params,
        body: Box::new(substitute_variables_impl(
          &new_body, &filtered, template,
        )),
        bracketed: *bracketed,
      },
    },
    Expr::PatternOptional {
      name,
      head,
      default,
    } => Expr::PatternOptional {
      name: name.clone(),
      head: head.clone(),
      default: default
        .as_ref()
        .map(|d| Box::new(substitute_variables_impl(d, bindings, template))),
    },
    Expr::PatternTest {
      name,
      head,
      blank_type,
      test,
    } => Expr::PatternTest {
      name: name.clone(),
      head: head.clone(),
      blank_type: *blank_type,
      test: Box::new(substitute_variables_impl(test, bindings, template)),
    },
    Expr::CurriedCall { func, args } => Expr::CurriedCall {
      func: Box::new(substitute_variables_impl(func, bindings, template)),
      args: args
        .iter()
        .map(|e| substitute_variables_impl(e, bindings, template))
        .collect(),
    },
    // Atoms that don't contain the variable
    _ => expr.clone(),
  }
}

// ─── 2D OutputForm rendering ───────────────────────────────────────────

/// A rectangular block of text used for 2D layout.
#[derive(Debug, Clone)]
struct TextBox {
  lines: Vec<String>,
  baseline: usize, // 0-indexed row that is the "main" line
}

impl TextBox {
  /// Create a single-line box.
  fn atom(s: &str) -> Self {
    Self {
      lines: vec![s.to_string()],
      baseline: 0,
    }
  }

  /// Width of the widest line.
  fn width(&self) -> usize {
    self
      .lines
      .iter()
      .map(std::string::String::len)
      .max()
      .unwrap_or(0)
  }

  /// Height (number of lines).
  fn height(&self) -> usize {
    self.lines.len()
  }

  /// Pad all lines to the same width with trailing spaces.
  fn normalize(&mut self) {
    let w = self.width();
    for line in &mut self.lines {
      if line.len() < w {
        line.push_str(&" ".repeat(w - line.len()));
      }
    }
  }

  /// Horizontal concatenation of multiple boxes, aligned on baselines.
  fn hconcat(parts: &[Self]) -> Self {
    if parts.is_empty() {
      return Self::atom("");
    }
    if parts.len() == 1 {
      return parts[0].clone();
    }

    // Find max baseline and max height-above-baseline
    let max_above: usize = parts.iter().map(|b| b.baseline).max().unwrap_or(0);
    let max_below: usize = parts
      .iter()
      .map(|b| b.height().saturating_sub(b.baseline + 1))
      .max()
      .unwrap_or(0);
    let total_height = max_above + 1 + max_below;

    let mut result_lines: Vec<String> = vec![String::new(); total_height];

    for part in parts {
      let mut p = part.clone();
      p.normalize();
      let w = p.width();
      let offset = max_above - p.baseline; // lines of padding above this part

      for row in 0..total_height {
        if row >= offset && row < offset + p.height() {
          result_lines[row].push_str(&p.lines[row - offset]);
        } else {
          result_lines[row].push_str(&" ".repeat(w));
        }
      }
    }

    let baseline = max_above;
    Self {
      lines: result_lines,
      baseline,
    }
  }

  /// Place exponent as superscript to the right and above the base.
  fn superscript(base: &Self, exp: &Self) -> Self {
    let mut base = base.clone();
    let mut exp = exp.clone();
    base.normalize();
    exp.normalize();

    let bw = base.width();
    let ew = exp.width();

    // The exponent sits above the top of the base, shifted right by base width.
    // Total lines = exp.height() + base.height()
    // Baseline = exp.height() + base.baseline

    let total = exp.height() + base.height();
    let mut lines = Vec::with_capacity(total);

    // Exponent lines (shifted right by base width)
    for i in 0..exp.height() {
      lines.push(format!("{}{}", " ".repeat(bw), exp.lines[i]));
    }
    // Base lines (padded right by exponent width)
    for i in 0..base.height() {
      lines.push(format!("{}{}", base.lines[i], " ".repeat(ew)));
    }

    Self {
      baseline: exp.height() + base.baseline,
      lines,
    }
  }

  /// Render a fraction:  numerator / bar / denominator
  fn fraction(num: &Self, denom: &Self) -> Self {
    let mut num = num.clone();
    let mut denom = denom.clone();
    num.normalize();
    denom.normalize();

    let bar_width = num.width().max(denom.width());

    // Center numerator and denominator
    let num_pad = (bar_width.saturating_sub(num.width())) / 2;
    let denom_pad = (bar_width.saturating_sub(denom.width())) / 2;

    let mut lines = Vec::new();

    // Numerator lines (centered)
    for l in &num.lines {
      lines.push(format!("{}{}", " ".repeat(num_pad), l));
    }
    // Bar
    lines.push("-".repeat(bar_width));
    // Denominator lines (centered)
    for l in &denom.lines {
      lines.push(format!("{}{}", " ".repeat(denom_pad), l));
    }

    // Baseline is the bar line
    let baseline = num.height();
    Self { lines, baseline }
  }

  /// Place a script below and to the right of the base — the mirror of
  /// [`superscript`](Self::superscript), used by `SubscriptBox`.
  fn subscript(base: &Self, sub: &Self) -> Self {
    let mut base = base.clone();
    let mut sub = sub.clone();
    base.normalize();
    sub.normalize();

    let bw = base.width();
    let sw = sub.width();
    let mut lines = Vec::with_capacity(base.height() + sub.height());
    for l in &base.lines {
      lines.push(format!("{}{}", l, " ".repeat(sw)));
    }
    for l in &sub.lines {
      lines.push(format!("{}{}", " ".repeat(bw), l));
    }
    Self {
      baseline: base.baseline,
      lines,
    }
  }

  /// Stack boxes vertically, left-aligned, keeping the first row's baseline.
  fn vstack(parts: &[Self]) -> Self {
    let mut lines = Vec::new();
    let baseline = parts.first().map_or(0, |p| p.baseline);
    for part in parts {
      let mut p = part.clone();
      p.normalize();
      lines.extend(p.lines);
    }
    if lines.is_empty() {
      return Self::atom("");
    }
    Self { lines, baseline }
  }
}

/// Renders to the final string. Trailing whitespace is stripped per line
/// because a 2D layout pads its lines to a common width — but a single-line
/// box carries no such padding, so a string that genuinely ends in spaces
/// (`ToString[", "]`, the separator of a `Row`) keeps them.
impl std::fmt::Display for TextBox {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if self.lines.len() == 1 {
      return f.write_str(&self.lines[0]);
    }
    let joined = self
      .lines
      .iter()
      .map(|l| l.trim_end())
      .collect::<Vec<_>>()
      .join("\n");
    f.write_str(&joined)
  }
}

/// Render `Derivative[n][f][args]` / `Derivative[n][f]` as prime-shorthand
/// (`f'[args]`, `f'''[args]`, `f^(4)[args]`, `f'`, ...). Returns None if the
/// expression isn't in the expected derivative-application shape.
///
/// Woxi stores the applied derivative as a single flattened FunctionCall:
///   Derivative[1][f][x]  →  FunctionCall{name: "Derivative", args: [1, f, x]}
///   Derivative[1][f]     →  FunctionCall{name: "Derivative", args: [1, f]}
/// (see the `name == "Derivative"` branch in expr_to_output further up.)
fn derivative_shorthand(expr: &Expr) -> Option<String> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "Derivative" || args.len() < 2 {
    return None;
  }
  let Expr::Integer(n) = args[0] else {
    return None;
  };
  if n < 1 {
    return None;
  }
  // args[1] must be the function symbol (Identifier). args[2..] are call args.
  if !matches!(&args[1], Expr::Identifier(_)) {
    return None;
  }
  let func = expr_to_string(&args[1]);
  let primes = if n <= 3 {
    "'".repeat(n as usize)
  } else {
    format!("^({n})")
  };
  if args.len() == 2 {
    return Some(format!("{func}{primes}"));
  }
  let call_args: Vec<String> = args[2..].iter().map(expr_to_string).collect();
  Some(format!("{}{}[{}]", func, primes, call_args.join(", ")))
}

/// Convert an expression to a 2D TextBox for OutputForm rendering.
fn expr_to_textbox(expr: &Expr) -> TextBox {
  // Derivative[n][f][args] → f'[args], f''[args], ... in OutputForm.
  if let Some(short) = derivative_shorthand(expr) {
    return TextBox::atom(&short);
  }
  match expr {
    Expr::Integer(n) => TextBox::atom(&n.to_string()),
    Expr::BigInteger(n) => TextBox::atom(&n.to_string()),
    Expr::Real(f) => TextBox::atom(&format_real(*f)),
    Expr::String(s) => TextBox::atom(s),
    Expr::Identifier(s) | Expr::Constant(s) => TextBox::atom(s),
    Expr::Raw(s) => TextBox::atom(s),
    // `Definition[sym]` / `FullDefinition[sym]` print as the definition text.
    Expr::FunctionCall { name, args }
      if (name == "Definition" || name == "FullDefinition")
        && args.len() == 1
        && matches!(&args[0], Expr::Identifier(_)) =>
    {
      TextBox::atom(&expr_to_string(expr))
    }

    // Power[base, 1/2] → Sqrt[base] display
    expr if crate::functions::is_sqrt(expr).is_some() => {
      let sqrt_arg = crate::functions::is_sqrt(expr).unwrap();
      TextBox::atom(&format!("Sqrt[{}]", expr_to_string(sqrt_arg)))
    }

    // A standalone Power with exponent -1 or -1/2 renders as a fraction
    // (`x^-1` → 1/x, `x^(-1/2)` → 1/Sqrt[x]); other negative exponents
    // keep the superscript with the sign (`x^-2` shows ` -2` over `x`,
    // wolframscript-verified). Inside Times every negative power moves
    // into the display denominator instead (see render_times_textbox).
    expr
      if negative_power_parts(expr).is_some_and(|(_, pos)| {
        matches!(pos, Expr::Integer(1))
          || matches!(&pos, Expr::FunctionCall { name, args }
            if name == "Rational"
              && matches!(args.as_slice(), [Expr::Integer(1), Expr::Integer(2)]))
      }) =>
    {
      let (base, pos_exp) = negative_power_parts(expr).unwrap();
      let denom = if matches!(pos_exp, Expr::Integer(1)) {
        expr_to_textbox(&base)
      } else {
        expr_to_textbox(&Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![base, pos_exp].into(),
        })
      };
      TextBox::fraction(&TextBox::atom("1"), &denom)
    }

    // Power[base, exp]. The exponent renders in LINEAR form — Wolfram
    // keeps superscripts one line high (`x^(2/3)` shows ` 2/3` over `x`,
    // not a three-line fraction above the base).
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      let base = expr_to_textbox_base(left);
      let exp = TextBox::atom(&exponent_linear_form(right));
      TextBox::superscript(&base, &exp)
    }

    // FunctionCall Power
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      // Check for fraction: Power[denom, -1] inside Times will be handled by Times
      // Here handle standalone Power
      let base = expr_to_textbox_base(&args[0]);
      let exp = TextBox::atom(&exponent_linear_form(&args[1]));
      TextBox::superscript(&base, &exp)
    }

    // Plus[args...]. A term that is itself a Plus (possible with held
    // forms: HoldForm[1 + 1] + HoldForm[2 + 2]) parenthesizes.
    Expr::FunctionCall { name, args } if name == "Plus" && args.len() >= 2 => {
      let render_term = |e: &Expr| -> TextBox {
        let tb = expr_to_textbox(e);
        if is_plus_expr(e) {
          parenthesize(&tb)
        } else {
          tb
        }
      };
      let mut parts: Vec<TextBox> = Vec::new();
      parts.push(render_term(&args[0]));
      for arg in args.iter().skip(1) {
        // Check for negative terms
        let (sign, term) = extract_sign_for_plus(arg);
        parts.push(TextBox::atom(sign));
        parts.push(render_term(&term));
      }
      TextBox::hconcat(&parts)
    }

    // BinaryOp Plus
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      let l = expr_to_textbox(left);
      let (sign, term) = extract_sign_for_plus(right);
      let r = expr_to_textbox(&term);
      TextBox::hconcat(&[l, TextBox::atom(sign), r])
    }

    // Times[args...] - handle fractions and products
    Expr::FunctionCall { name, args }
      if name == "Times" && !args.is_empty() =>
    {
      render_times_textbox(args)
    }

    // BinaryOp Times
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => render_times_textbox(&[*left.clone(), *right.clone()]),

    // BinaryOp Divide
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      let num = expr_to_textbox(left);
      let denom = expr_to_textbox(right);
      TextBox::fraction(&num, &denom)
    }

    // Rational[num, denom]. A negative rational atom parenthesizes:
    // `-3/4` shows as `-(` 3/4 `)` with the sign on the bar row.
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let Expr::Integer(n) = &args[0]
        && *n < 0
      {
        let frac = TextBox::fraction(
          &expr_to_textbox(&Expr::Integer(-n)),
          &expr_to_textbox(&args[1]),
        );
        return TextBox::hconcat(&[TextBox::atom("-"), parenthesize(&frac)]);
      }
      let num = expr_to_textbox(&args[0]);
      let denom = expr_to_textbox(&args[1]);
      TextBox::fraction(&num, &denom)
    }

    // UnaryOp Minus. A multi-line operand (a fraction) parenthesizes so
    // the sign reads on the bar row: `-(x/y)`.
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      let inner = expr_to_textbox(operand);
      if inner.height() > 1 {
        TextBox::hconcat(&[TextBox::atom("-"), parenthesize(&inner)])
      } else {
        TextBox::hconcat(&[TextBox::atom("-"), inner])
      }
    }

    // List
    Expr::List(items) => {
      let mut parts: Vec<TextBox> = Vec::new();
      parts.push(TextBox::atom("{"));
      for (i, item) in items.iter().enumerate() {
        if i > 0 {
          parts.push(TextBox::atom(", "));
        }
        parts.push(expr_to_textbox(item));
      }
      parts.push(TextBox::atom("}"));
      TextBox::hconcat(&parts)
    }

    // Generic FunctionCall: recurse into the arguments so fractions render
    // 2D (wolframscript: f[{-(5/3), Pi}] spans three lines). Restricted to
    // heads with the default bracket rendering — a head with a special 1D
    // display (Rule → `a -> b`, …) keeps its flat form.
    Expr::FunctionCall { name, args } => {
      let flat = expr_to_output(expr);
      let default_flat = format!(
        "{}[{}]",
        name,
        args
          .iter()
          .map(expr_to_output)
          .collect::<Vec<_>>()
          .join(", ")
      );
      if flat != default_flat {
        return TextBox::atom(&flat);
      }
      let mut parts: Vec<TextBox> = Vec::new();
      parts.push(TextBox::atom(&format!("{name}[")));
      for (i, a) in args.iter().enumerate() {
        if i > 0 {
          parts.push(TextBox::atom(", "));
        }
        parts.push(expr_to_textbox(a));
      }
      parts.push(TextBox::atom("]"));
      TextBox::hconcat(&parts)
    }

    // For everything else, fall back to 1D rendering
    _ => TextBox::atom(&expr_to_output(expr)),
  }
}

/// Render a base expression for Power, adding parens if needed for precedence.
fn expr_to_textbox_base(expr: &Expr) -> TextBox {
  // Wrap the Power base in parens when the base would otherwise bind
  // less tightly than the implicit superscript: Plus/Minus, negative
  // numeric literals, and unary `-x` all need parens so that e.g.
  // `Power[-I, n]` prints as `(-I)^n`, not `-I^n` (the latter parses
  // as `-(I^n)`).
  // Also wrap `Complex[0, neg]` since wolframscript prints it as `-I`
  // (or `-k*I`) which has the same ambiguity.
  let complex_neg_im = matches!(expr, Expr::FunctionCall { name, args }
    if name == "Complex"
      && args.len() == 2
      && (matches!(&args[1], Expr::Integer(n) if *n < 0)
        || matches!(&args[1], Expr::Real(f) if *f < 0.0)));
  // Times with a leading negative coefficient (e.g. `Times[-1, I]`,
  // which prints as `-I` and needs to be parenthesised before `^n`).
  let times_neg_leading = match expr {
    Expr::FunctionCall { name, args }
      if name == "Times" && !args.is_empty() =>
    {
      matches!(&args[0], Expr::Integer(n) if *n < 0)
        || matches!(&args[0], Expr::Real(f) if *f < 0.0)
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      ..
    } => {
      matches!(left.as_ref(), Expr::Integer(n) if *n < 0)
        || matches!(left.as_ref(), Expr::Real(f) if *f < 0.0)
    }
    _ => false,
  };
  let needs_parens = matches!(
    expr,
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      ..
    } | Expr::BinaryOp {
      op: BinaryOperator::Minus,
      ..
    } | Expr::UnaryOp {
      op: UnaryOperator::Minus,
      ..
    }
  ) || matches!(expr, Expr::Integer(n) if *n < 0)
    || matches!(expr, Expr::Real(f) if *f < 0.0)
    || matches!(expr, Expr::FunctionCall { name, .. } if name == "Plus")
    || complex_neg_im
    || times_neg_leading;

  if needs_parens {
    let inner = expr_to_textbox(expr);
    TextBox::hconcat(&[TextBox::atom("("), inner, TextBox::atom(")")])
  } else {
    expr_to_textbox(expr)
  }
}

/// Extract sign and unsigned term for Plus rendering.
/// If `e` is a `Rational[n, d]` with n < 0, return `Rational[-n, d]`.
fn negate_negative_rational(e: &Expr) -> Option<Expr> {
  if let Expr::FunctionCall { name, args } = e
    && name == "Rational"
    && args.len() == 2
    && let Expr::Integer(n) = &args[0]
    && *n < 0
  {
    return Some(Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(-n), args[1].clone()].into(),
    });
  }
  None
}

fn extract_sign_for_plus(expr: &Expr) -> (&'static str, Expr) {
  match expr {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => (" - ", *operand.clone()),
    Expr::Integer(n) if *n < 0 => (" - ", Expr::Integer(-n)),
    Expr::Real(f) if *f < 0.0 => (" - ", Expr::Real(-f)),
    // Times with a leading negative Real coefficient: negate it, so
    // `x - 0.5 y` renders as a subtraction rather than `x + -0.5 y`.
    // Unlike an integer `-1`, a real `-1.` coefficient stays written out
    // (`x - 1. y`), which is what wolframscript shows.
    Expr::FunctionCall { name, args }
      if name == "Times"
        && !args.is_empty()
        && matches!(&args[0], Expr::Real(f) if *f < 0.0) =>
    {
      let mut new_args = args.to_vec();
      if let Expr::Real(f) = &args[0] {
        new_args[0] = Expr::Real(-f);
      }
      (
        " - ",
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: new_args.into(),
        },
      )
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } if matches!(left.as_ref(), Expr::Real(f) if *f < 0.0) => {
      let Expr::Real(f) = left.as_ref() else {
        unreachable!()
      };
      (
        " - ",
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Real(-f)),
          right: right.clone(),
        },
      )
    }
    // Negative rational atom: 1/2 - 1/3 x renders `- 1/3 x`, not `+ -(1/3) x`.
    _ if negate_negative_rational(expr).is_some() => {
      (" - ", negate_negative_rational(expr).unwrap())
    }
    // Times with a leading negative Rational coefficient: negate it, so
    // `1 - x/2` renders as a subtraction of `x/2`.
    Expr::FunctionCall { name, args }
      if name == "Times"
        && !args.is_empty()
        && negate_negative_rational(&args[0]).is_some() =>
    {
      let mut new_args = args.to_vec();
      new_args[0] = negate_negative_rational(&args[0]).unwrap();
      (
        " - ",
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: new_args.into(),
        },
      )
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } if negate_negative_rational(left).is_some() => (
      " - ",
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(negate_negative_rational(left).unwrap()),
        right: right.clone(),
      },
    ),
    Expr::BigInteger(n) if n.sign() == Sign::Minus => {
      (" - ", Expr::BigInteger(-n))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } if matches!(left.as_ref(), Expr::Integer(-1)) => (" - ", *right.clone()),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } if matches!(left.as_ref(), Expr::Integer(n) if *n < 0) => {
      if let Expr::Integer(n) = left.as_ref() {
        (
          " - ",
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(-n)),
            right: right.clone(),
          },
        )
      } else {
        (" + ", expr.clone())
      }
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } if matches!(left.as_ref(), Expr::BigInteger(n) if n.sign() == Sign::Minus) => {
      if let Expr::BigInteger(n) = left.as_ref() {
        (
          " - ",
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::BigInteger(-n)),
            right: right.clone(),
          },
        )
      } else {
        (" + ", expr.clone())
      }
    }
    Expr::FunctionCall { name, args }
      if name == "Times"
        && !args.is_empty()
        && matches!(&args[0], Expr::BigInteger(n) if n.sign() == Sign::Minus) =>
    {
      if let Expr::BigInteger(n) = &args[0] {
        let mut new_args = vec![Expr::BigInteger(-n)];
        new_args.extend_from_slice(&args[1..]);
        (
          " - ",
          if new_args.len() == 1 {
            new_args.into_iter().next().unwrap()
          } else {
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: new_args.into(),
            }
          },
        )
      } else {
        (" + ", expr.clone())
      }
    }
    Expr::FunctionCall { name, args }
      if name == "Times"
        && !args.is_empty()
        && matches!(&args[0], Expr::Integer(n) if *n < 0) =>
    {
      if let Expr::Integer(n) = &args[0] {
        if *n == -1 {
          let new_args = args[1..].to_vec();
          if new_args.len() == 1 {
            (" - ", new_args[0].clone())
          } else {
            (
              " - ",
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: new_args.into(),
              },
            )
          }
        } else {
          let mut new_args = vec![Expr::Integer(-n)];
          new_args.extend_from_slice(&args[1..]);
          (
            " - ",
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: new_args.into(),
            },
          )
        }
      } else {
        (" + ", expr.clone())
      }
    }
    _ => (" + ", expr.clone()),
  }
}

/// Is this expression a Plus node (either representation)?
fn is_plus_expr(e: &Expr) -> bool {
  matches!(e, Expr::FunctionCall { name, .. } if name == "Plus")
    || matches!(
      e,
      Expr::BinaryOp {
        op: BinaryOperator::Plus,
        ..
      }
    )
}

/// Linear (one-line) rendering of a superscript exponent. A negative
/// rational parenthesizes with the sign outside — `x^(-3/2)` shows
/// ` -(3/2)` over `x` in wolframscript.
fn exponent_linear_form(exp: &Expr) -> String {
  if let Expr::FunctionCall { name, args } = exp
    && name == "Rational"
    && args.len() == 2
    && let Expr::Integer(n) = &args[0]
    && *n < 0
  {
    return format!("-({}/{})", -n, expr_to_output(&args[1]));
  }
  expr_to_output(exp)
}

/// If `e` is `Power[base, exp]` (either node shape) with a NEGATIVE
/// numeric exponent, return `(base, -exp)` so the factor can move into a
/// display denominator: `x^-1` → denominator `x`, `x^-2` → `x^2`,
/// `x^(-1/2)` → `Sqrt[x]`.
fn negative_power_parts(e: &Expr) -> Option<(Expr, Expr)> {
  let (base, exp) = match e {
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      (&args[0], &args[1])
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => (left.as_ref(), right.as_ref()),
    _ => return None,
  };
  match exp {
    Expr::Integer(n) if *n < 0 => Some((base.clone(), Expr::Integer(-n))),
    Expr::FunctionCall { name, args }
      if name == "Rational"
        && args.len() == 2
        && matches!(&args[0], Expr::Integer(n) if *n < 0) =>
    {
      if let Expr::Integer(n) = &args[0] {
        Some((
          base.clone(),
          Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(-n), args[1].clone()].into(),
          },
        ))
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Join boxes with single-space separators, Wolfram's product row.
fn product_row(parts: &[TextBox]) -> TextBox {
  let mut row: Vec<TextBox> = Vec::new();
  for (i, p) in parts.iter().enumerate() {
    if i > 0 {
      row.push(TextBox::atom(" "));
    }
    row.push(p.clone());
  }
  TextBox::hconcat(&row)
}

/// Is this optional coefficient literally 1 (hidden in display)?
fn p_is_one_local(c: Option<&Expr>) -> bool {
  matches!(c, Some(Expr::Integer(1)))
}

/// Render the factors of a product row, parenthesizing sums — but only
/// when the row has company: ToString[2*(2 + y)] shows `2 (2 + y)`,
/// while a LONE sum filling a fraction's numerator stays bare
/// ((1 + x)/(1 - x) has no parens).
fn factor_boxes(factors: &[&Expr], extra_boxes: usize) -> Vec<TextBox> {
  let row_len = factors.len() + extra_boxes;
  factors
    .iter()
    .map(|e| {
      let tb = expr_to_textbox(e);
      if row_len > 1 && is_plus_expr(e) {
        parenthesize(&tb)
      } else {
        tb
      }
    })
    .collect()
}

/// Wrap a box in parentheses on the baseline row.
fn parenthesize(inner: &TextBox) -> TextBox {
  TextBox::hconcat(&[TextBox::atom("("), inner.clone(), TextBox::atom(")")])
}

/// Render Times arguments the way Wolfram's OutputForm does (all shapes
/// wolframscript-verified):
///   3 x/4     →  numerator `3 x` over `4`
///   -3 x/4    →  numerator `-3 x` over `4` (|p| ≠ 1 keeps the sign inline)
///   -x/3      →  `-(1/3) x` (p = -1 with a rational coefficient)
///   -x/y      →  `-(x/y)`   (p = -1, integer coefficient)
///   -2/x      →  numerator `-2` over `x`
///   1/(2 x)   →  `1` over `2 x`
///   x^-2      →  `1` over `x^2` (negative powers move to the denominator)
fn render_times_textbox(args: &[Expr]) -> TextBox {
  // Split into: exact numeric coefficient p/q, numerator factors, and
  // denominator factors (bases of negative powers, with |exp|).
  let mut coeff_num: Option<Expr> = None; // p (Integer or BigInteger)
  let mut coeff_den: Option<Expr> = None; // q
  let mut num_factors: Vec<Expr> = Vec::new();
  let mut denom_factors: Vec<Expr> = Vec::new();

  for arg in args {
    match arg {
      Expr::Integer(_) | Expr::BigInteger(_) if coeff_num.is_none() => {
        coeff_num = Some(arg.clone());
      }
      Expr::FunctionCall { name, args: rargs }
        if name == "Rational" && rargs.len() == 2 && coeff_num.is_none() =>
      {
        coeff_num = Some(rargs[0].clone());
        coeff_den = Some(rargs[1].clone());
      }
      _ => {
        if let Some((base, pos_exp)) = negative_power_parts(arg) {
          if matches!(pos_exp, Expr::Integer(1)) {
            denom_factors.push(base);
          } else {
            denom_factors.push(Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![base, pos_exp].into(),
            });
          }
        } else {
          num_factors.push(arg.clone());
        }
      }
    }
  }

  let p_is_minus_one = matches!(coeff_num, Some(Expr::Integer(-1)));
  let p_is_one = matches!(coeff_num, Some(Expr::Integer(1)));
  let has_den = coeff_den.is_some() || !denom_factors.is_empty();

  // Row lengths decide sum parenthesization: the numerator row also
  // carries the coefficient p when it is shown (p ≠ ±1).
  let p_shown = coeff_num.is_some() && !p_is_one_local(coeff_num.as_ref());
  let num_refs: Vec<&Expr> = num_factors.iter().collect();
  let num_boxes: Vec<TextBox> = factor_boxes(&num_refs, usize::from(p_shown));
  let den_refs: Vec<&Expr> = denom_factors.iter().collect();
  let mut den_boxes: Vec<TextBox> = Vec::new();
  if let Some(q) = &coeff_den {
    den_boxes.push(expr_to_textbox(q));
  }
  den_boxes.extend(factor_boxes(&den_refs, usize::from(coeff_den.is_some())));

  if !has_den {
    // Plain product row: `3 x`, `-2 x`, `-x`.
    let mut parts: Vec<TextBox> = Vec::new();
    if p_is_minus_one {
      return TextBox::hconcat(&[TextBox::atom("-"), product_row(&num_boxes)]);
    }
    if let Some(p) = &coeff_num
      && !p_is_one
    {
      parts.push(expr_to_textbox(p));
    }
    parts.extend(num_boxes);
    return product_row(&parts);
  }

  if p_is_minus_one {
    // The sign pulls out in front of a parenthesized fraction. With
    // numerator factors and a rational coefficient the factors stay
    // OUTSIDE the parens (`-x/3` → `-(1/3) x`); with an integer
    // coefficient they form the fraction numerator (`-x/y` → `-(x/y)`).
    if coeff_den.is_some() && !num_boxes.is_empty() {
      let frac =
        TextBox::fraction(&TextBox::atom("1"), &product_row(&den_boxes));
      let mut parts =
        vec![TextBox::hconcat(&[TextBox::atom("-"), parenthesize(&frac)])];
      parts.extend(num_boxes);
      return product_row(&parts);
    }
    let num_box = if num_boxes.is_empty() {
      TextBox::atom("1")
    } else {
      product_row(&num_boxes)
    };
    let frac = TextBox::fraction(&num_box, &product_row(&den_boxes));
    return TextBox::hconcat(&[TextBox::atom("-"), parenthesize(&frac)]);
  }

  // General fraction: numerator `p f1 f2` (p omitted when 1), denominator
  // `q d1 d2` (q omitted when absent).
  let mut num_parts: Vec<TextBox> = Vec::new();
  if let Some(p) = &coeff_num
    && !p_is_one
  {
    num_parts.push(expr_to_textbox(p));
  }
  num_parts.extend(num_boxes);
  let num_box = if num_parts.is_empty() {
    TextBox::atom("1")
  } else {
    product_row(&num_parts)
  };
  TextBox::fraction(&num_box, &product_row(&den_boxes))
}

/// Render an expression in 2D OutputForm.
pub fn expr_to_output_form_2d(expr: &Expr) -> String {
  with_display_names(expr, |e| expr_to_textbox(e).to_string())
}

/// Render a box tree (`RowBox`, `FractionBox`, `SuperscriptBox`, …) as the
/// same 2D text `expr_to_output_form_2d` produces for ordinary expressions.
/// `ToString[expr, TraditionalForm]` goes through here: the typesetter turns
/// the expression into boxes and this lays them out, so a fraction still
/// spans three lines while `Sin[x]` reads `sin(x)`.
pub fn boxes_to_output_form_2d(expr: &Expr) -> String {
  boxes_to_textbox(expr).to_string()
}

fn boxes_to_textbox(expr: &Expr) -> TextBox {
  let items = |arg: &Expr| -> Vec<Expr> {
    match arg {
      Expr::List(items) => items.to_vec(),
      other => vec![other.clone()],
    }
  };
  match expr {
    // Box atoms carry their text as strings, displayed unquoted.
    Expr::String(s) => TextBox::atom(s),
    Expr::FunctionCall { name, args } => match (name.as_str(), args.len()) {
      ("RowBox", 1) => {
        let parts: Vec<TextBox> =
          items(&args[0]).iter().map(boxes_to_textbox).collect();
        TextBox::hconcat(&parts)
      }
      ("SuperscriptBox", 2) => TextBox::superscript(
        &boxes_to_textbox(&args[0]),
        &boxes_to_textbox(&args[1]),
      ),
      ("SubscriptBox", 2) => TextBox::subscript(
        &boxes_to_textbox(&args[0]),
        &boxes_to_textbox(&args[1]),
      ),
      ("SubsuperscriptBox", 3) => TextBox::superscript(
        &TextBox::subscript(
          &boxes_to_textbox(&args[0]),
          &boxes_to_textbox(&args[1]),
        ),
        &boxes_to_textbox(&args[2]),
      ),
      ("FractionBox", 2) => TextBox::fraction(
        &boxes_to_textbox(&args[0]),
        &boxes_to_textbox(&args[1]),
      ),
      ("SqrtBox", 1) => TextBox::hconcat(&[
        TextBox::atom("\u{221A}"),
        boxes_to_textbox(&args[0]),
      ]),
      ("RadicalBox", 2) => TextBox::hconcat(&[
        TextBox::superscript(
          &TextBox::atom("\u{221A}"),
          &boxes_to_textbox(&args[1]),
        ),
        boxes_to_textbox(&args[0]),
      ]),
      // Over/under scripts stack their script on the far side of the base
      // (`lim` with its variable beneath, `∑` with its bounds).
      ("OverscriptBox", 2) => TextBox::vstack(&[
        boxes_to_textbox(&args[1]),
        boxes_to_textbox(&args[0]),
      ]),
      ("UnderscriptBox", 2) => TextBox::vstack(&[
        boxes_to_textbox(&args[0]),
        boxes_to_textbox(&args[1]),
      ]),
      ("UnderoverscriptBox", 3) => TextBox::vstack(&[
        boxes_to_textbox(&args[2]),
        boxes_to_textbox(&args[0]),
        boxes_to_textbox(&args[1]),
      ]),
      // A grid lays its rows out one per line, columns separated by two
      // spaces — the spacing `TableForm` uses.
      ("GridBox", n) if n >= 1 => {
        let rows: Vec<TextBox> = items(&args[0])
          .iter()
          .map(|row| {
            let cells: Vec<TextBox> = items(row)
              .iter()
              .enumerate()
              .flat_map(|(i, cell)| {
                let mut out = Vec::new();
                if i > 0 {
                  out.push(TextBox::atom("  "));
                }
                out.push(boxes_to_textbox(cell));
                out
              })
              .collect();
            TextBox::hconcat(&cells)
          })
          .collect();
        TextBox::vstack(&rows)
      }
      // Wrappers that only decorate what they hold.
      (
        "StyleBox" | "TagBox" | "InterpretationBox" | "FormBox"
        | "AdjustmentBox" | "FrameBox" | "BoxData" | "TooltipBox",
        n,
      ) if n >= 1 => boxes_to_textbox(&args[0]),
      _ => expr_to_textbox(expr),
    },
    other => expr_to_textbox(other),
  }
}

/// Compose a message line around an expression rendered in 2D OutputForm,
/// baseline-aligning the surrounding text. wolframscript prints message
/// arguments in OutputForm, so a rational like -5/3 spans three lines with
/// the message text on the middle (baseline) line:
/// ```text
///                                                    5
/// Select::normal: ... in Select[EvenQ, {-15.1, Pi, -(-)}].
///                                                    3
/// ```
pub fn format_message_with_expr(
  prefix: &str,
  expr: &Expr,
  suffix: &str,
) -> String {
  TextBox::hconcat(&[
    TextBox::atom(prefix),
    expr_to_textbox(expr),
    TextBox::atom(suffix),
  ])
  .to_string()
}

/// Convert a string containing Wolfram box-syntax Unicode markers to the
/// `DisplayForm[<box>]` representation used in OutputForm.
/// The box expression is shown with quoted atoms unquoted.
fn box_string_to_display_form(s: &str) -> String {
  // A box segment displays as `DisplayForm[<box expression>]`; the prose
  // around it (`"value to test against \!\(\*SubscriptBox[…]\)"`) stays put.
  crate::functions::string_ast::split_inline_boxes(s)
    .into_iter()
    .map(|seg| {
      if seg.is_box {
        format!("DisplayForm[{}]", seg.text)
      } else {
        seg.text
      }
    })
    .collect()
}

/// Top-level output: like expr_to_output but with special handling for
/// formatting wrappers. FullForm[expr] renders the inner expression in
/// canonical FullForm notation (matching wolframscript REPL behavior).
/// Sequence[a, b, ...] displays as concatenated elements.
/// Render a Box-form expression (`SuperscriptBox`, `RowBox`, …) the way
/// wolframscript prints them inside `FullForm` / `InputForm`: the head is
/// shown verbatim, but string arguments lose their surrounding quotes
/// (so `SuperscriptBox["a", "b"]` displays as `SuperscriptBox[a, b]`).
/// Recurses into `List` arguments (used by `RowBox`, `GridBox`, …).
fn render_box_form(expr: &Expr) -> String {
  fn render_box_arg(arg: &Expr) -> String {
    match arg {
      Expr::String(s) => s.clone(),
      Expr::List(items) => {
        let parts: Vec<String> = items.iter().map(render_box_arg).collect();
        format!("{{{}}}", parts.join(", "))
      }
      Expr::FunctionCall { name, .. }
        if matches!(
          name.as_str(),
          "SuperscriptBox"
            | "SubscriptBox"
            | "SubsuperscriptBox"
            | "OverscriptBox"
            | "UnderscriptBox"
            | "UnderoverscriptBox"
            | "FractionBox"
            | "SqrtBox"
            | "RadicalBox"
            | "FormBox"
            | "TagBox"
            | "RowBox"
            | "GridBox"
            | "PaneBox"
            | "InterpretationBox"
            | "StyleBox"
        ) =>
      {
        render_box_form(arg)
      }
      _ => expr_to_input_form(arg),
    }
  }
  if let Expr::FunctionCall { name, args } = expr {
    let parts: Vec<String> = args.iter().map(render_box_arg).collect();
    format!("{}[{}]", name, parts.join(", "))
  } else {
    expr_to_input_form(expr)
  }
}

pub fn top_level_output(expr: &Expr) -> String {
  match expr {
    // ColumnForm[{e1, e2, ...}] is a legacy display directive that renders at
    // top level (unlike Column, which stays symbolic): each element is stacked
    // on its own line. Trailing alignment arguments do not affect the text.
    Expr::FunctionCall { name, args }
      if name == "ColumnForm"
        && !args.is_empty()
        && matches!(&args[0], Expr::List(_)) =>
    {
      if let Expr::List(items) = &args[0] {
        let lines: Vec<String> = items.iter().map(expr_to_output).collect();
        return lines.join("\n");
      }
      expr_to_output(expr)
    }
    Expr::FunctionCall { name, args }
      if name == "FullForm" && args.len() == 1 =>
    {
      // Inside the FullForm wrapper, Span keeps its head form (`Span[1, 4]`,
      // not `1 ;; 4`); the flag makes `expr_to_input_form` skip the `;;` branch.
      let _full_form_guard =
        FullFormGuard(IN_FULL_FORM.with(|c| c.replace(true)));
      // SeriesData has a special box display in Wolfram: even inside a
      // FullForm wrapper, the coefficient List shows with `{}` braces and
      // any Rational coefficients use `n/d` notation. Match this specific
      // case so e.g. `Series[Cosh[x], {x, 0, 2}] // FullForm` lands on
      // `FullForm[SeriesData[x, 0, {1, 0, 1/2}, 0, 3, 1]]` (wrapper kept,
      // InputForm-style inside) rather than `SeriesData[…, List[…], …]`.
      if let Expr::FunctionCall { name: sn, args: sa } = &args[0]
        && sn == "SeriesData"
        && sa.len() == 6
      {
        let parts: Vec<String> = sa.iter().map(expr_to_output).collect();
        return format!("FullForm[SeriesData[{}]]", parts.join(", "));
      }
      // Pure-imaginary `Times[Real, I]` is the inexact-zero Complex form
      // and Wolfram's REPL prints `FullForm[0. ± r*I]` (not `FullForm[r*I]`).
      // Mirror the bare top-level rendering rule when this is the inner of
      // FullForm so `(1. I) // FullForm` lands on `FullForm[0. + 1.*I]`.
      if let Expr::FunctionCall { name: tn, args: ta } = &args[0]
        && tn == "Times"
        && ta.len() == 2
        && let Expr::Real(c) = &ta[0]
        && (matches!(&ta[1], Expr::Identifier(s) if s == "I")
          || matches!(&ta[1],
            Expr::FunctionCall { name: cn, args: ca }
              if cn == "Complex"
                && ca.len() == 2
                && matches!(&ca[0], Expr::Integer(0))
                && matches!(&ca[1], Expr::Integer(1))))
      {
        let coef_str = format_real(*c);
        let body = if *c < 0.0 {
          let abs_str = coef_str
            .strip_prefix('-')
            .map(str::to_string)
            .unwrap_or(coef_str);
          format!("0. - {abs_str}*I")
        } else {
          format!("0. + {coef_str}*I")
        };
        return format!("FullForm[{body}]");
      }
      // `MessageName[sym, "tag"]` shows as `MessageName[sym, tag]` inside the
      // FullForm wrapper in wolframscript's REPL: the head form is preserved
      // (instead of the `sym::tag` infix shown by InputForm), but the tag
      // string is rendered without surrounding quotes — matching what
      // `wolframscript -code 'FullForm[a::b]'` prints.
      if let Expr::FunctionCall { name: mn, args: ma } = &args[0]
        && mn == "MessageName"
        && ma.len() == 2
      {
        let head = expr_to_input_form(&ma[0]);
        let tag = match &ma[1] {
          Expr::String(s) => s.clone(),
          Expr::Identifier(s) => s.clone(),
          other => expr_to_input_form(other),
        };
        return format!("FullForm[MessageName[{head}, {tag}]]");
      }
      // Box-form heads (`SuperscriptBox`, `SubscriptBox`, `RowBox`, …)
      // render their string arguments without surrounding quotes inside
      // `FullForm` and `InputForm`, matching wolframscript:
      //   `wolframscript -code 'FullForm[ToBoxes[a^b]]'` →
      //   `FullForm[SuperscriptBox[a, b]]` (no quotes around `a`/`b`).
      if let Expr::FunctionCall { name: bn, .. } = &args[0]
        && matches!(
          bn.as_str(),
          "SuperscriptBox"
            | "SubscriptBox"
            | "SubsuperscriptBox"
            | "OverscriptBox"
            | "UnderscriptBox"
            | "UnderoverscriptBox"
            | "FractionBox"
            | "SqrtBox"
            | "RadicalBox"
            | "FormBox"
            | "TagBox"
            | "RowBox"
            | "GridBox"
            | "PaneBox"
            | "InterpretationBox"
            | "StyleBox"
        )
      {
        return format!("FullForm[{}]", render_box_form(&args[0]));
      }
      // For atomic inputs (Identifier/Constant/Number/String) and
      // Rational/Repeated/RepeatedNull/Pattern* pseudo-atoms, match
      // wolframscript's REPL display: keep the `FullForm[…]` wrapper
      // around the inner expression rendered in InputForm. For other
      // compound inputs (Plus, Times, generic FunctionCalls, …), fall
      // through to the canonical FullForm rendering — many existing
      // tests rely on `FullForm[<head>[…]]` returning the bare head
      // form (like `Plus[a, b]`) for AST inspection.
      let is_rational = matches!(
        &args[0],
        Expr::FunctionCall { name: rn, args: ra }
          if rn == "Rational" && ra.len() == 2
      );
      let is_repeat = matches!(
        &args[0],
        Expr::FunctionCall { name: rn, .. }
          if rn == "Repeated" || rn == "RepeatedNull"
      );
      let is_pattern = matches!(
        &args[0],
        Expr::Pattern { .. }
          | Expr::PatternOptional { .. }
          | Expr::PatternTest { .. }
      ) || matches!(
        &args[0],
        Expr::FunctionCall { name: pn, .. } if pn == "Pattern"
      );
      // `a > b > c` parses to `Expr::Comparison { … }` and renders as
      // `Greater[a, b, c]` in bare full form, but wolframscript shows
      // `FullForm[a > b > c]` with the natural operator chain in
      // InputForm. Treat the variant the same as the FunctionCall heads
      // listed above.
      let is_comparison = matches!(&args[0], Expr::Comparison { .. });
      // `Power`, `Plus`, `Times`, `Divide`, `Subtract` may appear as
      // BinaryOp variants rather than FunctionCalls. wolframscript still
      // shows them wrapped in FullForm.
      let is_binary_power = matches!(
        &args[0],
        Expr::BinaryOp {
          op: BinaryOperator::Power
            | BinaryOperator::Plus
            | BinaryOperator::Times
            | BinaryOperator::Divide
            | BinaryOperator::Minus,
          ..
        }
      );
      // `Expr::List` shows with `{…}` braces in wolframscript's REPL
      // FullForm display, so `FullForm[{1, 2, 3}]` prints as
      // `FullForm[{1, 2, 3}]` rather than the bare `List[1, 2, 3]`.
      let is_list = matches!(&args[0], Expr::List(_));
      // "Form-wrapper" heads — these stay as `Head[…]` in both InputForm
      // and FullForm renderings (no operator-shape transformation), so
      // wolframscript shows `FullForm[Head[…]]` verbatim. Adding them
      // here doesn't disturb AST-inspection tests that use `FullForm` on
      // math expressions.
      let is_form_wrapper = matches!(
        &args[0],
        Expr::FunctionCall { name: fn_, .. }
          if matches!(
            fn_.as_str(),
            "BaseForm"
              | "NumberForm"
              | "InputForm"
              | "OutputForm"
              | "TraditionalForm"
              | "StandardForm"
              | "TeXForm"
              | "CForm"
              | "FortranForm"
              | "MatrixForm"
              | "TableForm"
              | "Row"
              | "Column"
              // Comparison/inequality heads display with the natural
              // operator chain (`a > b > c`) in wolframscript's REPL view.
              | "Greater"
              | "Less"
              | "GreaterEqual"
              | "LessEqual"
              | "Equal"
              | "Unequal"
              | "Inequality"
              // `Equivalent` displays with its infix character (a ⧦ b)
              // even inside FullForm, matching wolframscript.
              | "Equivalent"
              // `Power` keeps its `^` operator chain inside FullForm in
              // wolframscript's REPL: `FullForm[x^2]` prints as
              // `FullForm[x^2]` (with `ToString[…]` reaching the bare
              // `Power[x, 2]` form).
              | "Power"
              // `Plus` and `Times` likewise keep their `+`/`*` operator
              // chains in wolframscript's REPL FullForm display:
              // `FullForm[a + b*c]` prints as `FullForm[a + b*c]`.
              | "Plus"
              | "Times"
              | "Divide"
              | "Subtract"
              // `Span` (`a ;; b`) preserves its head form inside FullForm:
              // wolframscript's `FullForm[1 ;; 4]` prints
              // `FullForm[Span[1, 4]]`, keeping the wrapper instead of
              // dropping to a bare `Span[1, 4]`.
              | "Span"
              // `Factorial`/`Factorial2` (`x!`/`x!!`) keep their
              // postfix-operator InputForm inside the FullForm wrapper.
              | "Factorial"
              | "Factorial2"
              // `Hold`/`HoldForm`/`HoldComplete`/`HoldCompleteForm`/
              // `HoldPattern`/`Defer`
              // preserve the FullForm wrapper and render their argument
              // in InputForm — wolframscript prints `FullForm[Hold[1+2]]`
              // as `FullForm[Hold[1 + 2]]`, not the bare
              // `Hold[Plus[1, 2]]` head form. The bare form remains
              // reachable via `ToString[FullForm[…]]`.
              | "Hold"
              | "HoldForm"
              | "HoldComplete"
              | "HoldCompleteForm"
              | "HoldPattern"
              | "Defer"
          )
      );
      // `Not` evaluates to `Expr::UnaryOp { Not, … }` rather than a
      // `FunctionCall`, but wolframscript still preserves the FullForm
      // wrapper and shows the InputForm `!x` (with a leading space when
      // disambiguation is needed, e.g. ` !a!`) inside it. Match the
      // UnaryOp::Not variant separately so `FullForm[!a!]` lands on
      // `FullForm[ !a!]` rather than `Not[Factorial[a]]`.
      let is_unary_not = matches!(
        &args[0],
        Expr::UnaryOp {
          op: UnaryOperator::Not,
          ..
        }
      );
      if is_rational
        || is_repeat
        || is_pattern
        || is_comparison
        || is_binary_power
        || is_list
        || is_form_wrapper
        || is_unary_not
        || matches!(
          &args[0],
          Expr::Integer(_)
            | Expr::BigInteger(_)
            | Expr::Real(_)
            | Expr::BigFloat(_, _)
            | Expr::String(_)
            | Expr::Identifier(_)
            | Expr::Constant(_)
            | Expr::Slot(_)
            | Expr::SlotSequence(_)
        )
      {
        return format!("FullForm[{}]", expr_to_input_form(&args[0]));
      }
      // Default: wolframscript always keeps the `FullForm[…]` wrapper at
      // the top level, with the inner expression rendered in InputForm
      // (e.g. `FullForm[Foo[x]]` → `FullForm[Foo[x]]`,
      // `FullForm[a <-> b]` → `FullForm[a <-> b]`). Stick with that.
      format!("FullForm[{}]", expr_to_input_form(&args[0]))
    }
    // `HoldForm[expr]` at the top level keeps its wrapper —
    // `wolframscript -code 'HoldForm[1 + 2 + 3]'` prints `HoldForm[1 + 2 + 3]`
    // verbatim. The OutputForm strip in `expr_to_output` (used for nested
    // contexts) still applies, but the outermost call should not.
    Expr::FunctionCall { name, args }
      if name == "HoldForm" && args.len() == 1 =>
    {
      format!("HoldForm[{}]", expr_to_output(&args[0]))
    }
    Expr::FunctionCall { name, args } if name == "Sequence" => {
      args.iter().map(expr_to_output).collect::<String>()
    }
    // `OutputForm[Times[Real, I]]` mirrors the bare pure-imaginary
    // display rule below — wolframscript prints
    // `OutputForm[0. + r*I]`, not `OutputForm[r*I]`. Splice the
    // `0. + ` (or `0. - |r|*I` for negative `r`) prefix back into the
    // wrapped inner expression.
    Expr::FunctionCall { name, args }
      if name == "OutputForm"
        && args.len() == 1
        && matches!(&args[0],
          Expr::FunctionCall { name: tn, args: ta }
            if tn == "Times"
              && ta.len() == 2
              && matches!(&ta[0], Expr::Real(_))
              && (matches!(&ta[1], Expr::Identifier(s) if s == "I")
                || matches!(&ta[1],
                  Expr::FunctionCall { name: cn, args: ca }
                    if cn == "Complex"
                      && ca.len() == 2
                      && matches!(&ca[0], Expr::Integer(0))
                      && matches!(&ca[1], Expr::Integer(1))))) =>
    {
      let Expr::FunctionCall { args: ta, .. } = &args[0] else {
        unreachable!()
      };
      let Expr::Real(c) = &ta[0] else {
        unreachable!()
      };
      let coef_str = format_real(*c);
      let body = if *c < 0.0 {
        let abs_str = coef_str
          .strip_prefix('-')
          .map(str::to_string)
          .unwrap_or(coef_str);
        format!("0. - {abs_str}*I")
      } else {
        format!("0. + {coef_str}*I")
      };
      format!("OutputForm[{body}]")
    }
    // Pure-imaginary `Times[Real, I]` at the top level displays as
    // `0. ± r*I` to match wolframscript's Complex output for
    // *machine-precision* Reals. Inside larger expressions the existing
    // Times display is used (so `2. + 3.*I` stays unchanged).
    // BigFloat * I keeps the literal `<BigFloat>*I` form — Wolfram does
    // NOT split arbitrary-precision factors into Re/Im at display time.
    Expr::FunctionCall { name, args } if name == "Times" && args.len() == 2 => {
      let real_coef = match &args[0] {
        Expr::Real(f) => Some(*f),
        _ => None,
      };
      let i_factor = matches!(&args[1], Expr::Identifier(s) if s == "I")
        || matches!(&args[1],
          Expr::FunctionCall { name: cn, args: ca }
            if cn == "Complex" && ca.len() == 2
              && matches!(&ca[0], Expr::Integer(0))
              && matches!(&ca[1], Expr::Integer(1)));
      if let (Some(c), true) = (real_coef, i_factor) {
        let coef_str = format_real(c);
        if c < 0.0 {
          let abs_str = coef_str
            .strip_prefix('-')
            .map(str::to_string)
            .unwrap_or(coef_str);
          return format!("0. - {abs_str}*I");
        }
        return format!("0. + {coef_str}*I");
      }
      expr_to_output(expr)
    }
    _ => expr_to_output(expr),
  }
}

/// Replace all occurrences of an identifier with a given expression.
pub fn replace_identifier_in_expr(
  expr: &Expr,
  name: &str,
  replacement: &Expr,
) -> Expr {
  match expr {
    Expr::Identifier(n) if n == name => replacement.clone(),
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(replace_identifier_in_expr(left, name, replacement)),
      right: Box::new(replace_identifier_in_expr(right, name, replacement)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(replace_identifier_in_expr(operand, name, replacement)),
    },
    Expr::FunctionCall {
      name: fname,
      args: fargs,
    } => Expr::FunctionCall {
      name: if fname == name {
        if let Expr::Identifier(n) = replacement {
          n.clone()
        } else {
          fname.clone()
        }
      } else {
        fname.clone()
      },
      args: fargs
        .iter()
        .map(|a| replace_identifier_in_expr(a, name, replacement))
        .collect(),
    },
    Expr::List(elems) => Expr::List(
      elems
        .iter()
        .map(|a| replace_identifier_in_expr(a, name, replacement))
        .collect(),
    ),
    _ => expr.clone(),
  }
}

pub fn unevaluated(name: &str, args: &[Expr]) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args: args.to_vec().into(),
  }
}

/// Build `PatternTest[lhs, test]` from the pairs following an infix `?`:
/// the test function, then any trailing `[args]` groups. `?` binds before
/// those brackets, so `a?b[c]` is `PatternTest[a, b][c]`, not
/// `PatternTest[a, b[c]]`.
fn build_pattern_test<'i>(
  lhs: Expr,
  mut rest: impl Iterator<Item = pest::iterators::Pair<'i, Rule>>,
) -> Expr {
  let test = pair_to_expr(rest.next().expect("pattern test function"));
  let mut result = Expr::FunctionCall {
    name: "PatternTest".to_string(),
    args: vec![lhs, test].into(),
  };
  for bracket in rest.filter(|p| p.as_rule() == Rule::BracketArgs) {
    let args = bracket
      .into_inner()
      .filter(|p| p.as_str() != "[" && p.as_str() != "]" && p.as_str() != ",")
      .map(pair_to_expr)
      .collect();
    result = Expr::CurriedCall {
      func: Box::new(result),
      args,
    };
  }
  result
}
