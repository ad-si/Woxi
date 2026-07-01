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

The same notation SVG is produced by `ExportString[obj, "SVG"]`.
