# `SocketOpen`

Binds a TCP port and starts listening on it.
Port `0` asks the operating system for a free one,
which the socket then reports as its `"DestinationPort"`.

```scrut
$ wo 'Head[SocketOpen[0]]'
SocketObject
```

A listening socket's UUID carries a `TCPSERVER-` prefix,
which is what tells it apart from a connection at a glance:

```scrut
$ wo 'StringStartsQ[SocketOpen[0]["UUID"], "TCPSERVER-"]'
True
```

```scrut
$ wo 'SocketOpen[0]["DestinationPort"] > 0'
True
```

Every socket answers the same set of properties:

```scrut
$ wo 'SocketOpen[0]["Properties"]'
{ConnectedClients, DestinationHostname, DestinationIPAddress, DestinationPort, DirectionType, InprocQ, Protocol, Scheme, SocketListener, Type, UUID}
```

```scrut
$ wo 'srv = SocketOpen[0, "TCP"]; {srv["Protocol"], srv["DirectionType"]}'
{TCP, Server}
```
