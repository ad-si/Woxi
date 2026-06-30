# `FileTemplate`

`FileTemplate[src]` reads a template file from disk and yields a
`TemplateObject` whose `` `slot` `` markers can later be filled in with
`TemplateApply`. `src` may be a path string or a `File["path"]` wrapper.

Loading a template file and rendering it as a `TemplateObject`:

```scrut
$ printf 'Hello `name`!' > "$TMPDIR/woxi_filetemplate.txt"; wo "FileTemplate[\"$TMPDIR/woxi_filetemplate.txt\"]"
TemplateObject[{Hello , TemplateSlot[name], !}, CombinerFunction -> StringJoin, InsertionFunction -> TextString, MetaInformation -> <||>]
```

Filling the slots with `TemplateApply`:

```scrut
$ printf 'Hello `name`!' > "$TMPDIR/woxi_filetemplate.txt"; wo "TemplateApply[FileTemplate[\"$TMPDIR/woxi_filetemplate.txt\"], <|\"name\" -> \"Ada\"|>]"
Hello Ada!
```
