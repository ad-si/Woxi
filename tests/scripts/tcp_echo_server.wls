(* A TCP echo server and a client for it, in one script.

   The port, the UUIDs and the listener id all differ from run to run, so
   what is printed are the things that do not: the heads of the objects,
   the shape of the server's UUID, and the payload that came back. *)

server = SocketOpen[0, "TCP"];
port = server["DestinationPort"];
Print[Head[server]];
Print[StringStartsQ[server["UUID"], "TCPSERVER-"]];
Print[IntegerQ[port] && port > 0];
Print[server["Protocol"]];

(* Every connection the server accepts gets its own SocketObject, handed to
   the handler as "SourceSocket"; "Socket" stays the listening one. *)
listener = SocketListen[
  server,
  WriteString[#["SourceSocket"], StringReverse[#["Data"]]] &
];
Print[Head[listener]];
Print[listener["Socket"] === server];

client = SocketConnect[port];
Print[Head[client]];
Print[client["DestinationPort"] == port];

WriteString[client, "Hello, sockets!"];
Print[ByteArrayToString[SocketReadMessage[client]]];

Close[client];
Close[server];
Print[Sockets[]];
