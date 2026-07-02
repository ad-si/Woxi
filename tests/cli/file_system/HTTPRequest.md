# `HTTPRequest`

Builds a symbolic HTTP request object (no network access is performed
until the request is sent with `URLRead`).
The one-argument form canonicalizes to `HTTPRequest[url, <||>]`:

```scrut
$ wo 'HTTPRequest["https://example.com"]'
HTTPRequest[https://example.com, <||>]
```

Properties can be extracted from the request object:

```scrut
$ wo 'HTTPRequest["https://example.com"]["Method"]'
GET
```

```scrut
$ wo 'HTTPRequest["https://www.example.com:8080/api?a=1"]["Port"]'
8080
```

```scrut
$ wo 'HTTPRequest["https://example.com", <|"Method" -> "POST"|>]["Method"]'
POST
```

A list of property specifications yields an association.
The `Method` property can also be given as a symbol:

```scrut
$ wo 'req = HTTPRequest["https://www.wikipedia.org/"]; req[{"Scheme", "Domain", Method}]'
<|Scheme -> https, Domain -> www.wikipedia.org, Method -> GET|>
```

The URL is split into URLParse-style components:

```scrut
$ wo 'HTTPRequest["https://a.io/a/b.html?q=1"][{"Path", "PathString", "QueryString", "AbsoluteDomain"}]'
<|Path -> {, a, b.html}, PathString -> /a/b.html, QueryString -> q=1, AbsoluteDomain -> https://a.io|>
```
