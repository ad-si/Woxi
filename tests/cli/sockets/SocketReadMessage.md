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

Reading a socket that has been closed says so, and gives `$Failed`:

```wolfram
c = SocketConnect["127.0.0.1:1"]; SocketReadMessage[c]
(* The socket object SocketObject[…] is invalid or not open. *)
(* $Failed *)
```

That one is deliberately not a testcase. Wolfram reports an unusable socket
as a `Failure[…]` object rather than as a printed line plus `$Failed`, and it
connects lazily, so `SocketReadMessage` on this very input waits for a first
byte that never comes and never returns at all — see "Sockets" in the
conformance gaps. The behaviour above is covered by the unit tests in
`tests/interpreter_tests/sockets.rs` instead.
