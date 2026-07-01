---
icon: lucide/music
---

# Computational Music

Woxi supports the symbolic music objects introduced in Wolfram Language 15
(the [ComputationalMusic](https://reference.wolfram.com/language/guide/ComputationalMusic.html)
guide). Notes, chords, rests, pitches and the other music objects are kept as
canonical symbolic expressions, just like `Sound` and `SoundNote`.


## MusicObjectQ

`MusicObjectQ` tests whether an expression is a music object.

```scrut
$ wo 'MusicObjectQ[MusicPitch["C4"]]'
True
```

```scrut
$ wo 'MusicObjectQ[MusicNote[MusicPitch["C4"], MusicDuration["Half"]]]'
True
```

```scrut
$ wo 'MusicObjectQ[42]'
False
```


## MusicPitch

A pitch can be given by name in scientific notation.

```scrut
$ wo 'MusicPitch["C4"]'
MusicPitch[C4]
```

An integer is interpreted as a MIDI note number and canonicalized to its
named pitch (middle C is MIDI 60 / C4, and A4 is MIDI 69).

```scrut
$ wo 'MusicPitch[60]'
MusicPitch[C4]
```

```scrut
$ wo 'MusicPitch[69]'
MusicPitch[A4]
```

A frequency is canonicalized to the nearest named pitch (A4 is tuned to
440 Hz).

```scrut
$ wo 'MusicPitch[Quantity[440, "Hertz"]]'
MusicPitch[A4]
```

The pitch of a `SoundNote` (numbered relative to middle C) or of a
`MusicNote` can also be used as a pitch specification.

```scrut
$ wo 'MusicPitch[SoundNote[0]]'
MusicPitch[C4]
```

```scrut
$ wo 'MusicPitch[MusicNote["G3"]]'
MusicPitch[G3]
```

Pitches can be added and subtracted. Both the diatonic staff position (which
letter, in which register) and the MIDI number are combined, then the result is
returned as a `<|"Accidental" -> …, "Key" -> …, "MIDINumber" -> …|>` association.
Names without an octave default to octave 4.

```scrut
$ wo 'MusicPitch["Bb"] + MusicPitch["A#"] - MusicPitch["C"]'
MusicPitch[<|Accidental -> 1, Key -> G, MIDINumber -> 80|>]
```

Enharmonic spellings denote the same pitch, so they compare equal.

```scrut
$ wo 'MusicPitch["C#"] == MusicPitch["Db"]'
True
```

Adding a `MusicInterval` transposes a pitch, spelling the result from the
interval's diatonic step: a minor third above C is Eb (not D#).

```scrut
$ wo 'MusicPitch["C"] + MusicInterval["MinorThird"]'
MusicPitch[<|Accidental -> -1, Key -> E, MIDINumber -> 63|>]
```

A bare-semitone interval such as `MusicInterval[7]` transposes by that many
semitones, spelled the simplest way (7 semitones above C is G).

```scrut
$ wo 'MusicPitch["C"] + MusicInterval[7]'
MusicPitch[<|Accidental -> 0, Key -> G, MIDINumber -> 67|>]
```


## Notes and chords

Music objects nest like any other symbolic expression.

```scrut
$ wo 'MusicChord[{MusicPitch["C"], MusicPitch["E"], MusicPitch["G"]}]'
MusicChord[{MusicPitch[C], MusicPitch[E], MusicPitch[G]}]
```

```scrut
$ wo 'Head[MusicNote[MusicPitch["C4"]]]'
MusicNote
```

`MusicNote[pitch, duration]` canonicalizes to an association exposing its
`"Pitch"` and `"Duration"`.

```scrut
$ wo 'MusicNote["A#", 1/2]'
MusicNote[<|Pitch -> MusicPitch[<|Accidental -> 1, Key -> A|>], Duration -> MusicDuration[<|Duration -> 1/2|>]|>]
```

```scrut
$ wo 'MusicNote["A#", 1/2]["Pitch"]'
MusicPitch[<|Accidental -> 1, Key -> A|>]
```

```scrut
$ wo 'MusicNote["A#", 1/2]["Duration"]'
MusicDuration[<|Duration -> 1/2|>]
```

Durations add up, scaled by any leading coefficient.

```scrut
$ wo '3 MusicDuration[<|"Duration" -> 1/2|>] + MusicDuration[1/4]'
MusicDuration[<|Duration -> 7/4|>]
```

A named chord canonicalizes to its `"Name"` and `"Root"`, and can report its
constituent pitches and the intervals between them.

```scrut
$ wo 'MusicChord["GMajor"]'
MusicChord[<|Name -> Major, Root -> MusicPitch[<|Key -> G, Accidental -> 0|>]|>]
```

```scrut
$ wo 'MusicChord["GMajor"]["PitchList"]'
{MusicPitch[<|Accidental -> 0, Octave -> 4, Key -> G|>], MusicPitch[<|Accidental -> 0, Octave -> 4, Key -> B|>], MusicPitch[<|Accidental -> 0, Octave -> 5, Key -> D|>]}
```

```scrut
$ wo 'MusicChord["GMajor"]["IntervalList"]'
{MusicInterval[<|Semitones -> 4, Name -> MajorThird, CompoundOctaves -> 0|>], MusicInterval[<|Semitones -> 3, Name -> MinorThird, CompoundOctaves -> 0|>]}
```

The name accepts the common jazz/pop chord symbols too. A bare root is a major
triad, and a space between the root and the quality is optional, so `"G"`,
`"GMajor"`, and `"G Major"` are the same chord.

```scrut
$ wo 'MusicChord["G"] === MusicChord["GMajor"] === MusicChord["G Major"]'
True
```

Short symbols such as `m`, `dim`, `+`, `7`, `maj7`, `m7b5`, and `sus4` resolve
to their canonical qualities; casing distinguishes `m` (minor) from `M` (major).

```scrut
$ wo 'MusicChord["Cm7"]["Name"]'
MinorSeventh
```

```scrut
$ wo 'MusicChord["Cm7b5"]["Name"]'
HalfDiminishedSeventh
```

```scrut
$ wo 'MusicChord["C7"]["PitchList"]'
{MusicPitch[<|Accidental -> 0, Octave -> 4, Key -> C|>], MusicPitch[<|Accidental -> 0, Octave -> 4, Key -> E|>], MusicPitch[<|Accidental -> 0, Octave -> 4, Key -> G|>], MusicPitch[<|Accidental -> -1, Octave -> 4, Key -> B|>]}
```

Adding a `MusicInterval` to an explicit-pitch chord transposes every tone by the
interval, so a C-major triad up a perfect fourth becomes an F-major triad. The
transposed chord is returned in the canonical `<|"PitchList" -> …|>` form, and a
bare-semitone interval spells each tone straight from its MIDI number.

```scrut
$ wo 'MusicChord[{MusicPitch["C4"], MusicPitch["E4"], MusicPitch["G4"]}] + MusicInterval[5]'
MusicChord[<|PitchList -> {MusicPitch[<|Accidental -> 0, Key -> F, MIDINumber -> 65|>], MusicPitch[<|Accidental -> 0, Key -> A, MIDINumber -> 69|>], MusicPitch[<|Accidental -> 0, Key -> C, MIDINumber -> 72|>]}|>]
```

A minor-third stack transposed by a bare semitone interval spells its tones from
MIDI (G#, not the enharmonic Ab), matching Wolfram's `InputForm`.

```scrut
$ wo 'InputForm[MusicChord[NestList[# + MusicInterval["MinorThird"] &, MusicPitch["C"], 4]] + MusicInterval[5]]'
InputForm[MusicChord[<|PitchList -> {MusicPitch[<|Accidental -> 0, Key -> F, MIDINumber -> 65|>], MusicPitch[<|Accidental -> 1, Key -> G, MIDINumber -> 68|>], MusicPitch[<|Accidental -> 0, Key -> B, MIDINumber -> 71|>], MusicPitch[<|Accidental -> 0, Key -> D, MIDINumber -> 74|>], MusicPitch[<|Accidental -> 0, Key -> F, MIDINumber -> 77|>]}|>]]
```


## Rendering as notation

In the visual hosts — the [Woxi Playground](https://woxi.dev) and Woxi Studio —
music objects (`MusicNote`, `MusicChord`, `MusicScale`, `MusicMeasure`,
`MusicVoice`, `MusicScore`, …) are drawn on a treble staff the way Mathematica
displays them, with note heads, stems, flags, accidentals, ledger lines, rests
and barlines. On the command line they stay symbolic (as shown above).

`MusicPlot` renders a music object explicitly and, like `Plot`, yields a
graphic:

```scrut
$ wo 'Head[MusicPlot[MusicScale["Major", MusicPitch["C4"]]]]'
Graphics
```

The same notation SVG is produced by `ExportString[obj, "SVG"]`. A chord built
from transposed pitches — here a stack of minor thirds above C — draws as one
staff with all its note heads and (double) accidentals.

```scrut
$ wo 'StringTake[ExportString[MusicChord[NestList[# + MusicInterval["MinorThird"] &, MusicPitch["C"], 4]], "SVG"], 4]'
<svg
```

Transposing that chord by a further interval still renders as a chord.

```scrut
$ wo 'StringTake[ExportString[MusicChord[NestList[# + MusicInterval["MinorThird"] &, MusicPitch["C"], 4]] + MusicInterval[5], "SVG"], 4]'
<svg
```

`MusicPlot` of a `MusicMeasure` draws its pitches as quarter notes on the staff.

```scrut
$ wo 'Head[MusicPlot[MusicMeasure[{"C", "G", "A", "C"}]]]'
Graphics
```

```scrut
$ wo 'StringTake[ExportString[MusicPlot[MusicMeasure[{"C", "G", "A", "C"}]], "SVG"], 4]'
<svg
```

A `MusicMeasure`, `MusicVoice` or `MusicScore` prints its time signature on the
staff — the default 4/4 here draws two digit glyphs (numerator and denominator).

```scrut
$ wo 'StringCount[ExportString[MusicMeasure[{"C", "G", "A", "C"}], "SVG"], "timesig"]'
2
```

The meter is re-printed only when it changes, so a voice that switches from 4/4
to 3/4 mid-stream shows two time signatures (four digit glyphs).

```scrut
$ wo 'StringCount[ExportString[MusicVoice[{"C", "D", "E", "F", MusicTimeSignature[3, 4], "G"}], "SVG"], "timesig"]'
4
```


## Exporting to MIDI

`Export[…, obj]` to a `.mid` file writes a Standard MIDI File. A `MusicScore`
becomes one track per `MusicVoice`; each voice's pitches fill a 4/4 measure, so
three notes play as quarter, quarter, half at 120 BPM.

```scrut
$ wo 'Export["score.mid", MusicScore[{MusicVoice[{"A", "G", "E"}], MusicVoice[{"F", "E", "C"}]}]]; FileByteCount["score.mid"]'
130
```
