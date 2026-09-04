use super::*;

/// A bounded wait for a condition a socket handler makes true. Handlers run
/// on the evaluating thread, so `Pause` is what gives them their turn; the
/// loop keeps the tests from depending on how fast the machine is.
fn wait_until(condition: &str) -> String {
  format!("Do[If[{condition}, Break[]]; Pause[0.02], {{500}}];")
}

/// An echo server on a port the operating system picks, plus a client
/// connected to it. Leaves `srv`, `lis`, `port` and `c` bound.
fn echo_setup(handler: &str) -> String {
  format!(
    "srv = SocketOpen[0]; \
     port = srv[\"DestinationPort\"]; \
     lis = SocketListen[srv, {handler}]; \
     c = SocketConnect[port]; "
  )
}

mod sockets {
  use super::*;

  mod objects {
    use super::*;

    #[test]
    fn socket_open_returns_a_socket_object() {
      clear_state();
      let result = interpret("Head[SocketOpen[0]]").unwrap();
      assert_eq!(result, "SocketObject");
    }

    #[test]
    fn listening_socket_uuid_carries_the_tcpserver_prefix() {
      clear_state();
      let result =
        interpret("StringStartsQ[SocketOpen[0][\"UUID\"], \"TCPSERVER-\"]")
          .unwrap();
      assert_eq!(result, "True");
    }

    #[test]
    fn socket_object_prints_its_uuid_unquoted() {
      clear_state();
      // wolframscript shows `SocketObject[TCPSERVER-…]`, without the quotes
      // a plain string argument would otherwise print with.
      let result = interpret(
        "srv = SocketOpen[0]; \
         StringMatchQ[ToString[srv], \
           \"SocketObject[TCPSERVER-\" ~~ __ ~~ \"]\"]",
      )
      .unwrap();
      assert_eq!(result, "True");
    }

    #[test]
    fn socket_listener_is_an_integer_keyed_object() {
      clear_state();
      let result = interpret(
        "srv = SocketOpen[0]; lis = SocketListen[srv, Identity]; \
         {Head[lis], IntegerQ[First[lis]]}",
      )
      .unwrap();
      assert_eq!(result, "{SocketListener, True}");
    }

    #[test]
    fn port_zero_asks_the_system_for_a_free_port() {
      clear_state();
      let result = interpret("SocketOpen[0][\"DestinationPort\"] > 0").unwrap();
      assert_eq!(result, "True");
    }

    #[test]
    fn a_client_reports_the_port_it_connected_to() {
      clear_state();
      let result = interpret(
        "srv = SocketOpen[0]; port = srv[\"DestinationPort\"]; \
         SocketConnect[port][\"DestinationPort\"] == port",
      )
      .unwrap();
      assert_eq!(result, "True");
    }

    #[test]
    fn host_and_port_spellings_agree() {
      clear_state();
      // `SocketConnect` takes the endpoint as a string, as a host/port pair
      // and as a list; all three reach the same server.
      let result = interpret(
        "srv = SocketOpen[0]; port = srv[\"DestinationPort\"]; \
         a = SocketConnect[\"127.0.0.1:\" <> ToString[port]]; \
         b = SocketConnect[\"127.0.0.1\", port]; \
         d = SocketConnect[{\"127.0.0.1\", port}]; \
         Union[Map[#[\"DestinationPort\"] &, {a, b, d}]] === {port}",
      )
      .unwrap();
      assert_eq!(result, "True");
    }
  }

  mod properties {
    use super::*;

    #[test]
    fn socket_properties_are_the_documented_set() {
      clear_state();
      let result = interpret("SocketOpen[0][\"Properties\"]").unwrap();
      assert_eq!(
        result,
        "{ConnectedClients, DestinationHostname, DestinationIPAddress, \
         DestinationPort, DirectionType, InprocQ, Protocol, Scheme, \
         SocketListener, Type, UUID}"
      );
    }

    #[test]
    fn transport_properties_describe_tcp() {
      clear_state();
      let result = interpret(
        "srv = SocketOpen[0]; \
         {srv[\"Protocol\"], srv[\"Scheme\"], srv[\"InprocQ\"], \
          srv[\"DirectionType\"]}",
      )
      .unwrap();
      assert_eq!(result, "{TCP, tcp, False, Server}");
    }

    #[test]
    fn a_connection_reports_itself_as_a_client() {
      clear_state();
      let result = interpret(
        "srv = SocketOpen[0]; \
         SocketConnect[srv[\"DestinationPort\"]][\"DirectionType\"]",
      )
      .unwrap();
      assert_eq!(result, "Client");
    }

    #[test]
    fn destination_ip_address_is_an_ip_address_object() {
      clear_state();
      let result =
        interpret("SocketOpen[0][\"DestinationIPAddress\"]").unwrap();
      assert_eq!(result, "IPAddress[127.0.0.1]");
    }

    #[test]
    fn an_unknown_property_is_the_empty_list() {
      clear_state();
      // wolframscript answers `{}` for a property it does not keep, such as
      // `"SourcePort"`, rather than complaining.
      let result = interpret("SocketOpen[0][\"SourcePort\"]").unwrap();
      assert_eq!(result, "{}");
    }

    #[test]
    fn a_socket_points_back_at_its_listener() {
      clear_state();
      let result = interpret(
        "srv = SocketOpen[0]; lis = SocketListen[srv, Identity]; \
         srv[\"SocketListener\"] === lis",
      )
      .unwrap();
      assert_eq!(result, "True");
    }

    #[test]
    fn a_socket_without_a_listener_reports_none() {
      clear_state();
      let result = interpret("SocketOpen[0][\"SocketListener\"]").unwrap();
      assert_eq!(result, "{}");
    }

    #[test]
    fn listener_properties_are_the_documented_set() {
      clear_state();
      let result = interpret(
        "srv = SocketOpen[0]; SocketListen[srv, Identity][\"Properties\"]",
      )
      .unwrap();
      assert_eq!(
        result,
        "{CharacterEncoding, HandlerFunctions, HandlerFunctionsKeys, \
         RecordSeparators, Socket}"
      );
    }

    #[test]
    fn a_listener_points_back_at_its_socket() {
      clear_state();
      let result = interpret(
        "srv = SocketOpen[0]; lis = SocketListen[srv, Identity]; \
         lis[\"Socket\"] === srv",
      )
      .unwrap();
      assert_eq!(result, "True");
    }

    #[test]
    fn handler_functions_keys_name_the_association_a_handler_gets() {
      clear_state();
      let result = interpret(
        "srv = SocketOpen[0]; \
         SocketListen[srv, Identity][\"HandlerFunctionsKeys\"]",
      )
      .unwrap();
      assert_eq!(
        result,
        "{TimeStamp, SourceSocket, Socket, Data, DataBytes, DataByteArray, \
         MultipartComplete}"
      );
    }

    #[test]
    fn a_bare_handler_is_the_received_handler() {
      clear_state();
      let result = interpret(
        "srv = SocketOpen[0]; \
         Keys[SocketListen[srv, Identity][\"HandlerFunctions\"]]",
      )
      .unwrap();
      assert_eq!(result, "{Received}");
    }
  }

  mod transfer {
    use super::*;

    #[test]
    fn a_handler_sees_what_a_client_writes() {
      clear_state();
      let script = format!(
        "got = \"\"; {setup} WriteString[c, \"ping\"]; {wait} got",
        setup = echo_setup("(got = #[\"Data\"]) &"),
        wait = wait_until("got =!= \"\""),
      );
      assert_eq!(interpret(&script).unwrap(), "ping");
    }

    #[test]
    fn an_echo_server_answers_a_blocking_read() {
      clear_state();
      // The answer only exists once the handler has run, and the handler
      // runs on this thread — so the blocking read has to keep the event
      // queue moving while it waits, or this deadlocks.
      let script = format!(
        "{setup} WriteString[c, \"hi\"]; \
         ByteArrayToString[SocketReadMessage[c]]",
        setup = echo_setup(
          "WriteString[#[\"SourceSocket\"], \"echo:\" <> #[\"Data\"]] &"
        ),
      );
      assert_eq!(interpret(&script).unwrap(), "echo:hi");
    }

    #[test]
    fn socket_read_message_hands_back_a_byte_array() {
      clear_state();
      let script = format!(
        "{setup} WriteString[c, \"go\"]; Head[SocketReadMessage[c]]",
        setup = echo_setup("WriteString[#[\"SourceSocket\"], \"abcd\"] &"),
      );
      assert_eq!(interpret(&script).unwrap(), "ByteArray");
    }

    #[test]
    fn a_bounded_read_takes_only_what_was_asked_for() {
      clear_state();
      // `SocketReadMessage[sock, n]` never hands over more than `n` bytes
      // and never waits: with nothing pending it stays unevaluated, which
      // is what wolframscript does.
      let script = format!(
        "{setup} WriteString[c, \"go\"]; \
         parts = {{}}; \
         Do[r = SocketReadMessage[c, 2]; \
            If[Head[r] === ByteArray, \
              parts = Append[parts, ByteArrayToString[r]], \
              Pause[0.02]], \
           {{60}}]; \
         {{StringJoin[parts], Max[Map[StringLength, parts]]}}",
        setup = echo_setup("WriteString[#[\"SourceSocket\"], \"abcdef\"] &"),
      );
      assert_eq!(interpret(&script).unwrap(), "{abcdef, 2}");
    }

    #[test]
    fn read_line_reads_a_line_at_a_time() {
      clear_state();
      let script = format!(
        "{setup} WriteString[c, \"go\"]; {{ReadLine[c], ReadLine[c]}}",
        setup = echo_setup("WriteString[#[\"SourceSocket\"], \"a\\nb\\n\"] &"),
      );
      assert_eq!(interpret(&script).unwrap(), "{a, b}");
    }

    #[test]
    fn read_string_reads_until_the_peer_closes() {
      clear_state();
      let script = format!(
        "{setup} WriteString[c, \"go\"]; ReadString[c]",
        setup = echo_setup(
          "(WriteString[#[\"SourceSocket\"], \"done\"]; \
            Close[#[\"SourceSocket\"]]) &"
        ),
      );
      assert_eq!(interpret(&script).unwrap(), "done");
    }

    #[test]
    fn binary_read_list_reads_bytes_until_close() {
      clear_state();
      let script = format!(
        "{setup} WriteString[c, \"go\"]; BinaryReadList[c]",
        setup = echo_setup(
          "(BinaryWrite[#[\"SourceSocket\"], ByteArray[{1, 2, 255}]]; \
            Close[#[\"SourceSocket\"]]) &"
        ),
      );
      assert_eq!(interpret(&script).unwrap(), "{1, 2, 255}");
    }

    #[test]
    fn read_takes_one_byte_at_a_time() {
      clear_state();
      let script = format!(
        "{setup} WriteString[c, \"go\"]; {{Read[c, Byte], Read[c, Character]}}",
        setup = echo_setup("WriteString[#[\"SourceSocket\"], \"AB\"] &"),
      );
      assert_eq!(interpret(&script).unwrap(), "{65, B}");
    }

    #[test]
    fn a_handler_gets_the_documented_keys() {
      clear_state();
      let script = format!(
        "keys = {{}}; {setup} WriteString[c, \"x\"]; {wait} keys",
        setup = echo_setup("(keys = Keys[#]) &"),
        wait = wait_until("Length[keys] > 0"),
      );
      assert_eq!(
        interpret(&script).unwrap(),
        "{TimeStamp, SourceSocket, Socket, Data, DataBytes, DataByteArray, \
         MultipartComplete}"
      );
    }

    #[test]
    fn the_three_data_keys_describe_the_same_bytes() {
      clear_state();
      let script = format!(
        "seen = {{}}; {setup} WriteString[c, \"AB\"]; {wait} seen",
        setup = echo_setup(
          "(seen = {#[\"Data\"], #[\"DataBytes\"], \
             Normal[#[\"DataByteArray\"]], #[\"MultipartComplete\"]}) &"
        ),
        wait = wait_until("Length[seen] > 0"),
      );
      assert_eq!(
        interpret(&script).unwrap(),
        "{AB, {65, 66}, {65, 66}, True}"
      );
    }

    #[test]
    fn the_handler_is_told_which_connection_spoke() {
      clear_state();
      let script = format!(
        "seen = Null; {setup} WriteString[c, \"x\"]; {wait} \
         {{seen[[1]] === srv, MemberQ[srv[\"ConnectedClients\"], seen[[2]]]}}",
        setup = echo_setup("(seen = {#[\"Socket\"], #[\"SourceSocket\"]}) &"),
        wait = wait_until("seen =!= Null"),
      );
      assert_eq!(interpret(&script).unwrap(), "{True, True}");
    }

    #[test]
    fn a_large_payload_survives_being_chunked() {
      clear_state();
      // Sixteen mebibytes arrive as a few hundred reads, so this fails at
      // once if the read loop assumed a message fits in one chunk.
      let script = format!(
        "total = 0; {setup} \
         chunk = StringRepeat[\"x\", 1048576]; \
         Do[WriteString[c, chunk], {{16}}]; \
         {wait} total",
        setup = echo_setup("(total = total + Length[#[\"DataByteArray\"]]) &"),
        wait = wait_until("total >= 16777216"),
      );
      assert_eq!(interpret(&script).unwrap(), "16777216");
    }
  }

  mod waiting {
    use super::*;

    #[test]
    fn socket_ready_q_is_false_before_anything_arrives() {
      clear_state();
      let script =
        format!("{setup} SocketReadyQ[c]", setup = echo_setup("Identity"));
      assert_eq!(interpret(&script).unwrap(), "False");
    }

    #[test]
    fn socket_ready_q_waits_out_its_timeout() {
      clear_state();
      let script = format!(
        "{setup} WriteString[c, \"go\"]; SocketReadyQ[c, 5]",
        setup = echo_setup("WriteString[#[\"SourceSocket\"], \"back\"] &"),
      );
      assert_eq!(interpret(&script).unwrap(), "True");
    }

    #[test]
    fn socket_wait_next_names_the_socket_that_spoke() {
      clear_state();
      let script = format!(
        "{setup} WriteString[c, \"go\"]; SocketWaitNext[{{c}}, 5] === c",
        setup = echo_setup("WriteString[#[\"SourceSocket\"], \"back\"] &"),
      );
      assert_eq!(interpret(&script).unwrap(), "True");
    }

    #[test]
    fn socket_wait_next_stays_unevaluated_when_nothing_arrives() {
      clear_state();
      // wolframscript leaves the call unevaluated rather than returning a
      // failure when the timeout runs out.
      let script = format!(
        "{setup} Head[SocketWaitNext[{{c}}, 1]]",
        setup = echo_setup("Identity"),
      );
      assert_eq!(interpret(&script).unwrap(), "SocketWaitNext");
    }

    #[test]
    fn socket_wait_all_returns_the_sockets_it_was_given() {
      clear_state();
      let script = format!(
        "{setup} WriteString[c, \"go\"]; SocketWaitAll[{{c}}, 5] === {{c}}",
        setup = echo_setup("WriteString[#[\"SourceSocket\"], \"back\"] &"),
      );
      assert_eq!(interpret(&script).unwrap(), "True");
    }

    #[test]
    fn socket_wait_all_stays_unevaluated_when_nothing_arrives() {
      clear_state();
      let script = format!(
        "{setup} Head[SocketWaitAll[{{c}}, 1]]",
        setup = echo_setup("Identity"),
      );
      assert_eq!(interpret(&script).unwrap(), "SocketWaitAll");
    }
  }

  mod lifetime {
    use super::*;

    #[test]
    fn sockets_lists_what_is_open() {
      clear_state();
      let result =
        interpret("srv = SocketOpen[0]; Sockets[] === {srv}").unwrap();
      assert_eq!(result, "True");
    }

    #[test]
    fn closing_takes_a_socket_off_the_list() {
      clear_state();
      let result =
        interpret("srv = SocketOpen[0]; Close[srv]; Sockets[]").unwrap();
      assert_eq!(result, "{}");
    }

    #[test]
    fn close_hands_back_the_socket() {
      clear_state();
      let result =
        interpret("srv = SocketOpen[0]; Close[srv] === srv").unwrap();
      assert_eq!(result, "True");
    }

    #[test]
    fn reading_a_closed_socket_says_so() {
      clear_state();
      let script = format!(
        "{setup} Close[c]; SocketReadMessage[c]",
        setup = echo_setup("Identity"),
      );
      let result = interpret_with_stdout(&script).unwrap();
      assert_eq!(result.result, "$Failed");
      assert!(
        result
          .warnings
          .iter()
          .any(|w| w.contains("is invalid or not open.")),
        "expected the invalid-socket line, got {:?}",
        result.warnings
      );
    }

    #[test]
    fn the_invalid_socket_line_is_not_a_tagged_message() {
      clear_state();
      // wolframscript prints free text here and leaves `$MessageList` alone.
      let script = format!(
        "{setup} Close[c]; SocketReadMessage[c]; $MessageList",
        setup = echo_setup("Identity"),
      );
      assert_eq!(interpret(&script).unwrap(), "{}");
    }

    #[test]
    fn connecting_to_a_dead_port_fails_only_on_use() {
      clear_state();
      // `SocketConnect` says nothing about a refused connection; the
      // complaint comes when the socket is first used.
      let opened = interpret("Head[SocketConnect[\"127.0.0.1:1\"]]").unwrap();
      assert_eq!(opened, "SocketObject");
      clear_state();
      let result = interpret_with_stdout(
        "c = SocketConnect[\"127.0.0.1:1\"]; SocketReadMessage[c]",
      )
      .unwrap();
      assert_eq!(result.result, "$Failed");
      assert!(
        result
          .warnings
          .iter()
          .any(|w| w.contains("is invalid or not open.")),
        "expected the invalid-socket line, got {:?}",
        result.warnings
      );
    }

    #[test]
    fn deleting_a_listener_returns_null() {
      clear_state();
      let result = interpret(
        "srv = SocketOpen[0]; \
         DeleteObject[SocketListen[srv, Identity]] === Null",
      )
      .unwrap();
      assert_eq!(result, "True");
    }

    #[test]
    fn a_deleted_listener_is_off_its_socket() {
      clear_state();
      let result = interpret(
        "srv = SocketOpen[0]; lis = SocketListen[srv, Identity]; \
         DeleteObject[lis]; srv[\"SocketListener\"]",
      )
      .unwrap();
      assert_eq!(result, "{}");
    }

    #[test]
    fn closing_a_server_closes_the_connections_it_accepted() {
      clear_state();
      // Only the client end is left: the listening socket and the
      // connection it accepted both went with the close.
      let script = format!(
        "seen = Null; {setup} WriteString[c, \"x\"]; {wait} \
         Close[srv]; Sockets[] === {{c}}",
        setup = echo_setup("(seen = #[\"SourceSocket\"]) &"),
        wait = wait_until("seen =!= Null"),
      );
      assert_eq!(interpret(&script).unwrap(), "True");
    }

    #[test]
    fn closing_one_connection_leaves_the_listener_running() {
      clear_state();
      // An echo handler that closes the client it just answered must not
      // take the server down with it.
      let script = format!(
        "{setup} WriteString[c, \"one\"]; \
         first = ReadString[c]; \
         c2 = SocketConnect[port]; WriteString[c2, \"two\"]; \
         {{first, ReadString[c2]}}",
        setup = echo_setup(
          "(WriteString[#[\"SourceSocket\"], \"got:\" <> #[\"Data\"]]; \
            Close[#[\"SourceSocket\"]]) &"
        ),
      );
      assert_eq!(interpret(&script).unwrap(), "{got:one, got:two}");
    }
  }
}
