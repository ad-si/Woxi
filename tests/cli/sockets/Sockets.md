# `Sockets`

Lists the sockets open in this session, oldest first.

```scrut
$ wo 'srv = SocketOpen[0]; Sockets[] === {srv}'
True
```

`Close` takes a socket off the list,
and closing a listening socket takes the connections
it accepted with it:

```scrut
$ wo 'srv = SocketOpen[0]; Close[srv]; Sockets[]'
{}
```

`Close` answers with the endpoint it closed, not with the socket.
The port has to be read before the close: a closed socket answers no
property query at all, so asking it afterwards gives `$Failed`.

```scrut
$ wo 'srv = SocketOpen[0]; p = srv["DestinationPort"]; Close[srv] === "127.0.0.1:" <> ToString[p]'
True
```
