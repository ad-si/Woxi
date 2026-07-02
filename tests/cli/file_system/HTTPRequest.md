# `HTTPRequest`

Builds a symbolic HTTP request object (no network access is performed).
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
