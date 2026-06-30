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
