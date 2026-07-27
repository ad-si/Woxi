# `Snippet`

Gives the opening lines of a text, each cut to 80 characters. Without a
specification it gives the first line:

```scrut
$ wo 'Snippet["line one\nline two\nline three"]'
line one
```

The specification selects lines the way `Take` selects elements — a count
from the front, a negative count from the end, or a span:

```scrut
$ wo 'Snippet["line one\nline two\nline three", 2]'
line one
line two
```

```scrut
$ wo 'Snippet["a\nb\nc", -1]'
c
```

```scrut
$ wo 'Snippet["a\nb\nc\nd", 2 ;; 3]'
b
c
```

A long line is cut at 80 characters:

```scrut
$ wo 'StringLength[Snippet[StringRepeat["x", 200]]]'
80
```

Content that is not text, or a specification that is not a count or a span,
is reported:

```scrut {output_stream: combined}
$ wo 'Snippet[123]'

Snippet::invcnt: Content should be string, File, URL, or valid ContentObject.
$Failed
```
