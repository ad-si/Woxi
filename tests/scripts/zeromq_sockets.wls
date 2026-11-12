(* ZeroMQ sockets: a request-reply round trip, a push-pull pipeline and a
   publisher with a subscriber, all within one process. *)

rep = SocketOpen[{"127.0.0.1", 0}, {"ZMQ", "Reply"}];
req = SocketConnect[{"127.0.0.1", rep["DestinationPort"]}, {"ZMQ", "Request"}];
Print[{rep["Type"], req["Type"], rep["DirectionType"], req["DirectionType"]}];

Do[
  WriteString[req, "question " <> ToString[i]];
  q = ByteArrayToString[SocketReadMessage[rep]];
  WriteString[rep, "answer to " <> q];
  Print[ByteArrayToString[SocketReadMessage[req]]],
  {i, 3}
];

pull = SocketOpen[{"127.0.0.1", 0}, "ZMQ_PULL"];
push = SocketConnect[{"127.0.0.1", pull["DestinationPort"]}, "ZMQ_PUSH"];
Scan[WriteString[push, #] &, {"x", "y\nz", "\n", "q", "r\n"}];
Pause[0.3];
Print[ToString[{ReadLine[pull], ReadLine[pull], ReadLine[pull], ReadString[pull]}, InputForm]];

pub = SocketOpen[{"127.0.0.1", 0}, "ZMQ_PUB"];
sub = SocketConnect[{"127.0.0.1", pub["DestinationPort"]}, {"ZMQ", "Subscribe"}];
Print[SetOptions[sub, "Subscribe" -> "news"]];
Pause[0.5];
WriteString[pub, "weather: rain"];
WriteString[pub, "news: hello"];
Print[ByteArrayToString[SocketReadMessage[sub]]];
Print[SocketReadyQ[sub, 0.2]];

Print[ToString[WriteString[sub, "x"], InputForm]];
Print[Length /@ {Sockets["ZMQ"], Sockets[{"ZMQ", "PUB"}]}];
Print[StringStartsQ[#, "127.0.0.1:"] & /@ (Close /@ {rep, req, pull, push, pub, sub})];
