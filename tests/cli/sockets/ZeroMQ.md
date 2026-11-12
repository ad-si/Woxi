# ZeroMQ

Passing `"ZMQ"` as the protocol to `SocketOpen` (bind) or `SocketConnect`
makes a ZeroMQ socket. Woxi implements the ZeroMQ wire protocol itself
(ZMTP 3.0), so it talks to any libzmq program (pyzmq, czmq, …)
without needing libzmq installed.

The socket type can be spelled several ways:

```scrut
$ wo 'Map[SocketOpen[{"127.0.0.1", 0}, #]["Type"] &, {"ZMQ", "ZMQ_PUB", {"ZMQ", "Subscribe"}, {"ZMQ", "Request"}}]'
{ZMQ_PAIR, ZMQ_PUB, ZMQ_SUB, ZMQ_REQ}
```

A request gets a reply:

```scrut
$ wo 'rep = SocketOpen[{"127.0.0.1", 0}, "ZMQ_REP"]; req = SocketConnect[{"127.0.0.1", rep["DestinationPort"]}, "ZMQ_REQ"]; WriteString[req, "ping"]; WriteString[rep, "pong:" <> ByteArrayToString[SocketReadMessage[rep]]]; ByteArrayToString[SocketReadMessage[req]]'
pong:ping
```

Each write is one message, and `SocketReadMessage` reads one message:

```scrut
$ wo 'pull = SocketOpen[{"127.0.0.1", 0}, "ZMQ_PULL"]; push = SocketConnect[{"127.0.0.1", pull["DestinationPort"]}, "ZMQ_PUSH"]; WriteString[push, "one"]; WriteString[push, "two"]; {ByteArrayToString[SocketReadMessage[pull]], ByteArrayToString[SocketReadMessage[pull]]}'
{one, two}
```

A subscriber chooses its topics with `SetOptions`:

```scrut
$ wo 'sub = SocketConnect[{"127.0.0.1", 5556}, {"ZMQ", "Subscribe"}]; SetOptions[sub, "Subscribe" -> "lmax"]'
{Subscribe -> lmax}
```

Socket options read back under their libzmq names or in CamelCase:

```scrut
$ wo 's = SocketOpen[{"127.0.0.1", 0}, "ZMQ_PUB"]; {s["Linger"], s["ZMQ_TYPE"], s["Protocol"]}'
{-1, 1, ZMQ}
```

A socket type refuses the functions it cannot serve:

```scrut
$ wo 'sub = SocketConnect[{"127.0.0.1", 5556}, "ZMQ_SUB"]; ToString[WriteString[sub, "x"], InputForm]'
Failure["SocketsLink", <|"MessageTemplate" :> SocketObject::fctnotsupported, "MessageParameters" -> {"ZMQ_SUB", WriteString}|>]
```
