# Basic String Tests

## `Print`

```scrut
$ wo 'Print["Hello World!"]'
Hello World!
Null
```


## `StringLength`

```scrut
$ wo 'StringLength["Hello World!"]'
12
```


## `StringTake`

```scrut
$ wo 'StringTake["Hello World!", 5]'
Hello
```


## `StringDrop`

```scrut
$ wo 'StringDrop["Hello World!", 6]'
World!
```


## `StringJoin`

```scrut
$ wo 'StringJoin["Hello", " ", "World!"]'
Hello World!
```


## `StringSplit`

```scrut
$ wo 'StringSplit["Hello World!", " "]'
{Hello, World!}
```


## `StringStartsQ`

```scrut
$ wo 'StringStartsQ["Hello World!", "Hello"]'
True
```

```scrut
$ wo 'StringStartsQ["Hello World!", "Bye"]'
False
```


## `StringEndsQ`

```scrut
$ wo 'StringEndsQ["Hello World!", "World!"]'
True
```

```scrut
$ wo 'StringEndsQ["Hello World!", "Moon!"]'
False
```
