# `SocketListen`

Runs a handler for everything that arrives on a socket.
On a listening socket it also starts accepting connections:
each one becomes a `SocketObject` of its own,
handed to the handler as `"SourceSocket"`.

```scrut
$ wo 'Head[SocketListen[SocketOpen[0], Identity]]'
SocketListener
```

The handler is called with an association.
Its keys are the ones `"HandlerFunctionsKeys"` names:

```scrut
$ wo 'SocketListen[SocketOpen[0], Identity]["HandlerFunctionsKeys"]'
{TimeStamp, SourceSocket, Socket, Data, DataBytes, DataByteArray, MultipartComplete}
```

Handlers run on the thread that evaluates Wolfram code,
so they fire wherever evaluation waits — in a `Pause`,
in a blocking read, or between two statements:

```scrut
$ wo 'srv = SocketOpen[0]; got = ""; SocketListen[srv, (got = #["Data"]) &]; c = SocketConnect[srv["DestinationPort"]]; WriteString[c, "ping"]; Pause[0.5]; got'
ping
```

`DeleteObject` stops a listener:

```scrut
$ wo 'srv = SocketOpen[0]; DeleteObject[SocketListen[srv, Identity]]; srv["SocketListener"]'
{}
```
