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

```scrut
$ wo 'srv = SocketOpen[0]; Close[srv] === srv'
True
```
