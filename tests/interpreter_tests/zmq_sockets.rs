//! ZeroMQ sockets: `SocketOpen` / `SocketConnect` with a `"ZMQ"` protocol.
//! Every expected value here was checked against wolframscript.

use super::*;

/// A bounded wait for something a background connection makes true.
fn wait_until(condition: &str) -> String {
  format!("Do[If[{condition}, Break[]]; Pause[0.02], {{250}}];")
}

/// A bound `server` socket of type `bound` on a free port, and a `client`
/// of type `connecting` connected to it.
fn pair(bound: &str, connecting: &str) -> String {
  format!(
    "server = SocketOpen[{{\"127.0.0.1\", 0}}, \"{bound}\"]; \
     port = server[\"DestinationPort\"]; \
     client = SocketConnect[{{\"127.0.0.1\", port}}, \"{connecting}\"]; "
  )
}

fn run(code: &str) -> String {
  clear_state();
  let result = interpret(code).unwrap();
  clear_state();
  result
}

mod protocol_specs {
  use super::*;

  #[test]
  fn every_spelling_of_a_type_names_it() {
    assert_eq!(
      run(
        "Map[(s = SocketOpen[{\"127.0.0.1\", 0}, #]; t = s[\"Type\"]; \
           Close[s]; t) &, \
           {\"ZMQ\", {\"ZMQ\", \"Publish\"}, {\"ZMQ\", \"PUB\"}, \
            {\"ZMQ\", \"pub\"}, \"zmq_pub\", {\"ZMQ\", \"Subscribe\"}, \
            {\"ZMQ\", \"Request\"}, {\"ZMQ\", \"Reply\"}, {\"ZMQ\", \"Push\"}, \
            {\"ZMQ\", \"Pull\"}, {\"ZMQ\", \"Dealer\"}, {\"ZMQ\", \"Router\"}, \
            {\"ZMQ\", \"XPub\"}, {\"ZMQ\", \"XSub\"}, \"ZMQ_REQ\", {\"ZMQ\"}}]"
      ),
      "{ZMQ_PAIR, ZMQ_PUB, ZMQ_PUB, ZMQ_PUB, ZMQ_PUB, ZMQ_SUB, ZMQ_REQ, \
       ZMQ_REP, ZMQ_PUSH, ZMQ_PULL, ZMQ_DEALER, ZMQ_ROUTER, ZMQ_XPUB, \
       ZMQ_XSUB, ZMQ_REQ, ZMQ_PAIR}"
    );
  }

  #[test]
  fn an_unknown_type_is_an_invproto_failure() {
    assert_eq!(
      run(
        "ToString[SocketOpen[{\"127.0.0.1\", 0}, {\"ZMQ\", \"Foo\"}], InputForm]"
      ),
      "Failure[\"SocketsLink\", <|\"MessageTemplate\" :> SocketOpen::invproto, \
       \"MessageParameters\" -> {\"ZMQ\", \"Foo\"}|>]"
    );
  }

  #[test]
  fn an_unknown_protocol_is_a_noproto_failure() {
    assert_eq!(
      run("ToString[SocketConnect[{\"127.0.0.1\", 1}, \"Foo\"], InputForm]"),
      "Failure[\"SocketsLink\", <|\"MessageTemplate\" :> SocketConnect::noproto, \
       \"MessageParameters\" -> {\"Foo\"}|>]"
    );
  }

  #[test]
  fn connecting_needs_a_real_port() {
    assert_eq!(
      run(
        "ToString[SocketConnect[\"tcp://127.0.0.1:*\", \"ZMQ_DEALER\"], InputForm]"
      ),
      "Failure[\"SocketsLink\", <|\"MessageTemplate\" :> SocketConnect::addrspec, \
       \"MessageParameters\" -> {\"tcp://127.0.0.1:*\", \
       \"the port must be a valid Integer when connecting a socket\"}|>]"
    );
    assert_eq!(
      run(
        "ToString[SocketConnect[\"tcp://127.0.0.1\", \"ZMQ_DEALER\"], InputForm]"
      ),
      "Failure[\"SocketsLink\", <|\"MessageTemplate\" :> SocketConnect::addrspec, \
       \"MessageParameters\" -> {\"tcp://127.0.0.1\", \
       \"it is missing the port specification\"}|>]"
    );
  }

  #[test]
  fn binding_a_busy_port_is_a_zmqinvcmd_failure() {
    let result = run(
      "s = SocketOpen[{\"127.0.0.1\", 0}, \"ZMQ_PUB\"]; \
       ToString[SocketOpen[{\"127.0.0.1\", s[\"DestinationPort\"]}, \"ZMQ_PUB\"], \
       InputForm]",
    );
    assert_eq!(
      result,
      "Failure[\"SocketsLink\", <|\"MessageTemplate\" :> SocketOpen::zmqinvcmd, \
       \"MessageParameters\" -> {\"Address already in use\"}|>]"
    );
  }
}

mod properties {
  use super::*;

  #[test]
  fn a_zeromq_socket_lists_its_options_as_properties() {
    assert_eq!(
      run(
        "s = SocketOpen[{\"127.0.0.1\", 0}, \"ZMQ_PUB\"]; \
         {Length[s[\"Properties\"]], s[\"Properties\"][[12]], \
          Last[s[\"Properties\"]]}"
      ),
      "{51, ZMQ_AFFINITY, ZMQ_ZAP_DOMAIN}"
    );
  }

  #[test]
  fn address_properties() {
    let result = run(&format!(
      "{} {{server[\"Protocol\"], server[\"Scheme\"], client[\"Scheme\"], \
        server[\"DirectionType\"], client[\"DirectionType\"], \
        client[\"DestinationIPAddress\"], server[\"ConnectedClients\"], \
        client[\"LastEndpoint\"] === \"tcp://127.0.0.1:\" <> ToString[port]}}",
      pair("ZMQ_PUB", "ZMQ_SUB")
    ));
    assert_eq!(
      result,
      "{ZMQ, tcp, TCP, Server, Client, IPAddress[127.0.0.1], None, True}"
    );
  }

  #[test]
  fn a_wildcard_bind_keeps_its_address_as_a_string() {
    assert_eq!(
      run(
        "s = SocketOpen[\"tcp://0.0.0.0:0\", \"ZMQ_PUB\"]; \
         {s[\"DestinationHostname\"], StringQ[s[\"DestinationIPAddress\"]], \
          s[\"Scheme\"]}"
      ),
      "{0.0.0.0, True, tcp}"
    );
  }

  #[test]
  fn option_getters_accept_both_spellings() {
    assert_eq!(
      run(
        "s = SocketOpen[{\"127.0.0.1\", 0}, \"ZMQ_PUB\"]; \
         {s[\"Linger\"], s[\"ZMQ_LINGER\"], s[\"ZMQ_TYPE\"], s[\"Rcvhwm\"], \
          s[\"ReconnectIvl\"], s[\"Rcvmore\"], s[\"Identity\"] === \"\"}"
      ),
      "{-1, -1, 1, 1000, 100, False, True}"
    );
  }

  #[test]
  fn subscribers_do_not_linger() {
    assert_eq!(
      run(
        "Map[SocketOpen[{\"127.0.0.1\", 0}, #][\"Linger\"] &, \
         {\"ZMQ_SUB\", \"ZMQ_XSUB\", \"ZMQ_PUB\", \"ZMQ_PAIR\"}]"
      ),
      "{0, 0, -1, -1}"
    );
  }

  #[test]
  fn events_say_whether_a_send_could_go_out() {
    assert_eq!(
      run(
        "Map[SocketOpen[{\"127.0.0.1\", 0}, #][\"Events\"] &, \
         {\"ZMQ_PUB\", \"ZMQ_SUB\", \"ZMQ_REP\", \"ZMQ_DEALER\", \"ZMQ_ROUTER\"}]"
      ),
      "{2, 0, 0, 0, 2}"
    );
  }

  #[test]
  fn several_properties_come_back_as_sorted_rules() {
    assert_eq!(
      run(
        "s = SocketOpen[{\"127.0.0.1\", 0}, \"ZMQ_SUB\"]; \
         {s[{\"Type\", \"Linger\"}], s[{\"Linger\"}], s[{}]}"
      ),
      "{{Linger -> 0, Type -> ZMQ_SUB}, 0, {}}"
    );
  }

  #[test]
  fn an_unknown_option_is_an_invalid_get_option_failure() {
    assert_eq!(
      run(
        "s = SocketOpen[{\"127.0.0.1\", 0}, \"ZMQ_PUB\"]; \
         ToString[s[\"Foo\"], InputForm]"
      ),
      "Failure[\"SocketsLink\", <|\"MessageTemplate\" :> \
       SocketObject::invalidGetOption, \"MessageParameters\" -> {\"Foo\"}|>]"
    );
  }

  #[test]
  fn options_lists_both_type_entries() {
    assert_eq!(
      run(
        "s = SocketOpen[{\"127.0.0.1\", 0}, \"ZMQ_PAIR\"]; \
         o = Options[s]; {Length[o], Cases[o, (\"Type\" -> t_) :> t]}"
      ),
      "{51, {0, ZMQ_PAIR}}"
    );
  }
}

mod set_options {
  use super::*;

  #[test]
  fn set_options_reports_what_it_set() {
    assert_eq!(
      run(
        "c = SocketConnect[{\"127.0.0.1\", 1}, {\"ZMQ\", \"Subscribe\"}]; \
         {SetOptions[c, \"Subscribe\" -> \"lmax\"], \
          SetOptions[c, \"ZMQ_SUBSCRIBE\" -> \"zz\", \"Linger\" -> 5], \
          SetOptions[c, \"Subscribe\" -> All], \
          SetOptions[c, \"Unsubscribe\" -> \"zz\"], c[\"Linger\"]}"
      ),
      "{{Subscribe -> lmax}, {Subscribe -> zz, Linger -> 5}, \
       {ZMQ_SUBSCRIBE -> All}, {Unsubscribe -> zz}, 5}"
    );
  }

  #[test]
  fn set_options_rejects_bad_names_and_values() {
    assert_eq!(
      run(
        "c = SocketOpen[{\"127.0.0.1\", 0}, \"ZMQ_DEALER\"]; \
         ToString[{SetOptions[c, \"Foo\" -> 1], SetOptions[c, \"Linger\" -> \"x\"], \
          SetOptions[c, \"Unsubscribe\" -> All], SetOptions[c, Linger -> 1]}, \
          InputForm]"
      ),
      "{Failure[\"SocketsLink\", <|\"MessageTemplate\" :> \
       SocketObject::invalidOptionName, \"MessageParameters\" -> {\"Foo\"}|>], \
       Failure[\"SocketsLink\", <|\"MessageTemplate\" :> \
       SocketObject::invalidOptionValue, \"MessageParameters\" -> \
       {\"Linger\", \"x\"}|>], Failure[\"SocketsLink\", <|\"MessageTemplate\" :> \
       SocketObject::invalidOptionValue, \"MessageParameters\" -> \
       {\"ZMQ_UNSUBSCRIBE\", All}|>], Failure[\"SocketsLink\", \
       <|\"MessageTemplate\" :> SocketObject::invalidOptionName, \
       \"MessageParameters\" -> {Linger}|>]}"
    );
  }

  #[test]
  fn an_identity_is_given_as_byte_values() {
    assert_eq!(
      run(
        "c = SocketOpen[{\"127.0.0.1\", 0}, \"ZMQ_DEALER\"]; \
         {SetOptions[c, \"Identity\" -> {65, 66}], c[\"Identity\"], \
          Head[SetOptions[c, \"Identity\" -> \"x\"]]}"
      ),
      "{{Identity -> AB}, AB, Failure}"
    );
  }
}

mod messaging {
  use super::*;

  #[test]
  fn push_pull_keeps_message_boundaries() {
    // WriteLine and Write send the text and the newline as two messages;
    // an empty write sends nothing.
    let result = run(&format!(
      "{} WriteLine[client, \"e\"]; Write[client, g[1]]; \
       WriteString[client, \"\"]; WriteString[client, \"end\"]; \
       Table[ByteArrayToString[SocketReadMessage[server]], 5]",
      pair("ZMQ_PULL", "ZMQ_PUSH")
    ));
    assert_eq!(result, "{e, \n, g[1], \n, end}");
  }

  #[test]
  fn a_non_blocking_read_of_nothing_is_an_empty_byte_array() {
    let result = run(&format!(
      "{} {{Length[SocketReadMessage[server, \"Blocking\" -> False]], \
        SocketReadyQ[server]}}",
      pair("ZMQ_PULL", "ZMQ_PUSH")
    ));
    assert_eq!(result, "{0, False}");
  }

  #[test]
  fn requests_get_replies() {
    let result = run(&format!(
      "{} WriteString[client, \"ping\"]; \
       m = ByteArrayToString[SocketReadMessage[server]]; \
       WriteString[server, \"pong:\" <> m]; \
       {{m, ByteArrayToString[SocketReadMessage[client]]}}",
      pair("ZMQ_REP", "ZMQ_REQ")
    ));
    assert_eq!(result, "{ping, pong:ping}");
  }

  #[test]
  fn a_subscriber_only_sees_its_topics() {
    let result = run(&format!(
      "{} SetOptions[client, \"Subscribe\" -> \"lmax\"]; \
       {wait} WriteString[server, \"other\"]; WriteString[server, \"lmax 1\"]; \
       ByteArrayToString[SocketReadMessage[client]]",
      pair("ZMQ_PUB", "ZMQ_SUB"),
      wait = wait_until(
        "(WriteString[server, \"lmax probe\"]; SocketReadyQ[client, 0.05])"
      )
    ));
    // The probes that raced the subscription are read off first.
    assert!(result.starts_with("lmax"), "{result}");
    assert_ne!(result, "other");
  }

  #[test]
  fn a_router_prefixes_the_sender_routing_id() {
    let result = run(&format!(
      "{} WriteString[client, \"hi\"]; \
       {{StringLength[ByteArrayToString[ \
           SocketReadMessage[server, \"Multipart\" -> False], \"ISOLatin1\"]], \
         server[\"Rcvmore\"], \
         ByteArrayToString[SocketReadMessage[server, \"Multipart\" -> False]], \
         server[\"Rcvmore\"]}}",
      pair("ZMQ_ROUTER", "ZMQ_DEALER")
    ));
    // A generated routing id is five bytes.
    assert_eq!(result, "{5, True, hi, False}");
  }

  #[test]
  fn multipart_reads_join_the_parts() {
    let result = run(&format!(
      "{} WriteString[client, \"hi\"]; \
       Length[SocketReadMessage[server]]",
      pair("ZMQ_ROUTER", "ZMQ_DEALER")
    ));
    assert_eq!(result, "7");
  }

  #[test]
  fn stream_reads_run_across_messages() {
    let result = run(&format!(
      "{} Scan[WriteString[client, #] &, \
         {{\"x\", \"y\\nz\", \"\\n\", \"q\", \"r\\n\", \"s\\nt\\n\"}}]; \
       {wait} ToString[{{ReadLine[server], ReadLine[server], ReadLine[server], \
         ReadLine[server], ReadString[server], ReadString[server]}}, InputForm]",
      pair("ZMQ_PULL", "ZMQ_PUSH"),
      wait = "Pause[0.3];"
    ));
    assert_eq!(
      result,
      "{\"xy\", \"z\", \"qr\", \"s\", \"\\nt\\n\", EndOfFile}"
    );
  }

  #[test]
  fn binary_write_answers_with_the_socket_output_stream() {
    let result = run(&format!(
      "{} a = BinaryWrite[client, {{65}}]; b = BinaryWrite[client, {{66}}]; \
       {{Head[a], a === b, First[a] === \"out-\" <> client[\"UUID\"]}}",
      pair("ZMQ_PULL", "ZMQ_PUSH")
    ));
    assert_eq!(result, "{OutputStream, True, True}");
  }
}

mod refusals {
  use super::*;

  #[test]
  fn read_only_and_write_only_types_refuse() {
    let result = run(&format!(
      "{} ToString[{{WriteString[client, \"x\"], ReadString[server], \
        SocketReadMessage[server]}}, InputForm]",
      pair("ZMQ_PUB", "ZMQ_SUB")
    ));
    assert_eq!(
      result,
      "{Failure[\"SocketsLink\", <|\"MessageTemplate\" :> \
       SocketObject::fctnotsupported, \"MessageParameters\" -> \
       {\"ZMQ_SUB\", WriteString}|>], Failure[\"SocketsLink\", \
       <|\"MessageTemplate\" :> SocketObject::fctnotsupported, \
       \"MessageParameters\" -> {\"ZMQ_PUB\", ReadString}|>], \
       Failure[\"SocketsLink\", <|\"MessageTemplate\" :> \
       SocketObject::zmqwriteonly, \"MessageParameters\" -> {\"ZMQ_PUB\"}|>]}"
    );
  }

  #[test]
  fn request_reply_refuses_two_message_writes() {
    let result = run(&format!(
      "{} Head /@ {{Write[client, 1], WriteLine[client, \"x\"]}}",
      pair("ZMQ_REP", "ZMQ_REQ")
    ));
    assert_eq!(result, "{Failure, Failure}");
  }
}

mod listening {
  use super::*;

  #[test]
  fn a_listener_gets_each_message() {
    let result = run(&format!(
      "{} got = {{}}; l = SocketListen[server, AppendTo[got, #] &]; \
       WriteString[client, \"hello\"]; {wait} \
       {{got[[1]][\"Data\"], got[[1]][\"MultipartComplete\"], \
         Head[got[[1]][\"TimeStamp\"]], got[[1]][\"Socket\"] === server, \
         server[\"SocketListener\"] === l}}",
      pair("ZMQ_PULL", "ZMQ_PUSH"),
      wait = wait_until("got =!= {}")
    ));
    assert_eq!(result, "{hello, True, DateObject, True, True}");
  }

  #[test]
  fn a_listener_gets_multipart_messages_part_by_part() {
    let result = run(&format!(
      "{} got = {{}}; SocketListen[server, AppendTo[got, #] &]; \
       WriteString[client, \"hi\"]; {wait} \
       {{Length[got], #[\"MultipartComplete\"] & /@ got, Last[got][\"Data\"]}}",
      pair("ZMQ_ROUTER", "ZMQ_DEALER"),
      wait = wait_until("Length[got] == 2")
    ));
    assert_eq!(result, "{2, {False, True}, hi}");
  }
}

mod sockets_filters {
  use super::*;

  #[test]
  fn sockets_filters_by_protocol_type_and_direction() {
    let result = run(&format!(
      "{} t = SocketOpen[0]; \
       Length /@ {{Sockets[], Sockets[\"ZMQ\"], Sockets[\"TCP\"], \
         Sockets[\"Server\"], Sockets[\"Client\"], Sockets[{{\"ZMQ\", \"PUB\"}}], \
         Sockets[\"ZMQ\", \"SUB\"]}}",
      pair("ZMQ_PUB", "ZMQ_SUB")
    ));
    assert_eq!(result, "{3, 2, 1, 2, 1, 1, 1}");
  }

  #[test]
  fn an_unknown_filter_is_a_noproto_failure() {
    assert_eq!(
      run("ToString[{Sockets[\"Foo\"], Sockets[\"ZMQ_SUB\"]}, InputForm]"),
      "{Failure[\"SocketsLink\", <|\"MessageTemplate\" :> Sockets::noproto, \
       \"MessageParameters\" -> {\"Foo\"}|>], Failure[\"SocketsLink\", \
       <|\"MessageTemplate\" :> Sockets::noproto, \"MessageParameters\" -> \
       {\"ZMQ_SUB\"}|>]}"
    );
  }

  #[test]
  fn close_answers_with_the_endpoint() {
    assert_eq!(
      run(
        "s = SocketOpen[\"tcp://0.0.0.0:0\", \"ZMQ_PUB\"]; p = s[\"DestinationPort\"]; \
         Close[s] === \"0.0.0.0:\" <> ToString[p]"
      ),
      "True"
    );
  }
}
