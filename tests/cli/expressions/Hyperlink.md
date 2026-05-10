# `Hyperlink` - Represents a clickable hyperlink

`Hyperlink[uri]` represents a hyperlink that jumps to the specified URI when
clicked.

```scrut
$ wo 'Hyperlink["https://woxi.ad-si.com"]'
Hyperlink[https://woxi.ad-si.com]
```

`Hyperlink[label, uri]` represents a hyperlink to be displayed as `label`.

```scrut
$ wo 'Hyperlink["Woxi", "https://woxi.ad-si.com"]'
Hyperlink[Woxi, https://woxi.ad-si.com]
```

The two parts can be extracted with `Part`:

```scrut
$ wo 'Hyperlink["Woxi", "https://woxi.ad-si.com"][[2]]'
https://woxi.ad-si.com
```
