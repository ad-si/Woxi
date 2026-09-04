# `SocketConnect`

Opens the near end of an outgoing TCP connection.
The endpoint may be a port, a `"host:port"` string,
a host and a port, or a `{host, port}` pair.

```scrut
$ wo 'srv = SocketOpen[0]; Head[SocketConnect[srv["DestinationPort"]]]'
SocketObject
```

```scrut
$ wo 'srv = SocketOpen[0]; port = srv["DestinationPort"]; SocketConnect["127.0.0.1:" <> ToString[port]]["DestinationPort"] == port'
True
```

A connection describes itself as a client, not a server:

```scrut
$ wo 'srv = SocketOpen[0]; SocketConnect[srv["DestinationPort"]]["DirectionType"]'
Client
```
