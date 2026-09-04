# `SocketReadMessage`

Reads whatever has arrived on a socket, as a `ByteArray`.
It waits for the first byte, so an echo server's answer
can be read straight back:

```scrut
$ wo 'srv = SocketOpen[0]; SocketListen[srv, WriteString[#["SourceSocket"], "Hello, " <> #["Data"] <> "!"] &]; c = SocketConnect[srv["DestinationPort"]]; WriteString[c, "sockets"]; ByteArrayToString[SocketReadMessage[c]]'
Hello, sockets!
```

```scrut
$ wo 'srv = SocketOpen[0]; SocketListen[srv, WriteString[#["SourceSocket"], "abcd"] &]; c = SocketConnect[srv["DestinationPort"]]; WriteString[c, "go"]; Head[SocketReadMessage[c]]'
ByteArray
```

Reading a socket that has been closed says so,
and gives `$Failed`:

```scrut
$ wo 'c = SocketConnect["127.0.0.1:1"]; SocketReadMessage[c]'

The socket object SocketObject\[[0-9a-f-]+\] is invalid or not open. (regex)
$Failed
```
