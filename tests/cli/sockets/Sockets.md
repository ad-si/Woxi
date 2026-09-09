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

`Close` answers with the endpoint it closed, not with the socket:

```scrut
$ wo 'srv = SocketOpen[0]; Close[srv] === "127.0.0.1:" <> ToString[srv["DestinationPort"]]'
True
```
