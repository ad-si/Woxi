---
icon: lucide/plug
---

# Sockets

Woxi implements the Wolfram Language's TCP socket functions natively,
over `std::net` — no external library and no event loop of its own.

A socket is a `SocketObject`, printed with its UUID:
`SocketOpen` binds and listens (its UUID carries a `TCPSERVER-` prefix),
`SocketConnect` opens the near end of an outgoing connection.
`SocketListen` attaches handler functions to a socket and returns a
`SocketListener`.

Handlers run on the thread that evaluates Wolfram code,
so they fire wherever evaluation waits:
inside `Pause`, inside a blocking read such as `SocketReadMessage`,
inside `SocketWaitNext` and `SocketWaitAll`,
and between two top-level statements.
Nothing runs a handler in the background behind a long computation,
which is also how wolframscript behaves.

- [`SocketOpen`](sockets/SocketOpen.md)
- [`SocketConnect`](sockets/SocketConnect.md)
- [`SocketListen`](sockets/SocketListen.md)
- [`SocketReadMessage`](sockets/SocketReadMessage.md)
- [`Sockets`](sockets/Sockets.md)

Sockets also work with the ordinary stream functions:
`WriteString`, `WriteLine`, `Write` and `BinaryWrite` send to one,
and `ReadString`, `ReadLine`, `Read` and `BinaryReadList` read from one.
`SocketReadyQ` asks whether a read would find anything,
`Close` shuts a socket down and `Sockets[]` lists what is still open.
