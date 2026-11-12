//! TCP sockets: `SocketOpen`, `SocketConnect`, `SocketListen` and the
//! blocking read/write API around `SocketObject`.
//!
//! # Shape
//!
//! Everything the evaluator sees lives on the thread that runs Wolfram
//! code: the registry, the listener table and the handler expressions are
//! all `thread_local!`, exactly like `ENV` and the stream registry. That is
//! not an accident — `Expr` is not `Send`, so a handler can only ever be
//! applied on the thread that defined it.
//!
//! Background threads therefore never touch an `Expr`. They only move
//! bytes: one accept thread per listening socket that has a
//! `SocketListen` attached, and one read thread per connection under such
//! a listener. Both push [`SocketEvent`]s into the owning thread's
//! channel, and the owning thread turns them into handler calls when it
//! next reaches a pump point ([`pump_socket_events`]).
//!
//! Sockets with no listener attached have no threads at all: their reads
//! are ordinary blocking reads made straight from the evaluator, buffered
//! through [`SocketEntry::read_buffer`] so a short read leaves the rest
//! addressable for the next one.

use super::*;

#[cfg(not(target_arch = "wasm32"))]
use std::cell::RefCell;
#[cfg(not(target_arch = "wasm32"))]
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::io::{Read, Write};
#[cfg(not(target_arch = "wasm32"))]
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc::{Receiver, Sender, channel};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Arc, Mutex};
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

/// The `TCPSERVER-` prefix wolframscript puts on a listening socket's UUID,
/// which is also how `SocketOpen`'s result is told from `SocketConnect`'s
/// at a glance.
#[cfg(not(target_arch = "wasm32"))]
pub const SERVER_UUID_PREFIX: &str = "TCPSERVER-";

/// How much a single read pulls from the socket at once. Reads are chunked
/// rather than sized to the request so a `SocketReadMessage` of a few bytes
/// does not turn a megabyte transfer into a million syscalls.
#[cfg(not(target_arch = "wasm32"))]
const READ_CHUNK: usize = 64 * 1024;

/// How long a read thread blocks before looking at its stop flag again.
/// Only the shutdown latency of `DeleteObject[listener]` depends on it —
/// a timed-out read consumes nothing, so throughput does not.
#[cfg(not(target_arch = "wasm32"))]
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// How long a blocking read waits before letting the socket handlers run.
/// Short, because a read whose answer depends on a handler is waiting on
/// exactly this.
#[cfg(not(target_arch = "wasm32"))]
const WAIT_SLICE: Duration = Duration::from_millis(5);

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SocketRole {
  /// Made by `SocketOpen`: bound and listening, never read or written.
  Server,
  /// Made by `SocketConnect`: the near end of an outgoing connection.
  Client,
  /// Handed over by a listening socket's accept thread.
  Accepted,
}

#[cfg(not(target_arch = "wasm32"))]
struct SocketEntry {
  role: SocketRole,
  /// The connection, for a client or accepted socket. Shared with the read
  /// thread when one is running; `&TcpStream` implements both `Read` and
  /// `Write`, so no cloning of the handle is needed to use it from here.
  stream: Option<Arc<TcpStream>>,
  listener: Option<Arc<TcpListener>>,
  /// Host as the caller spelled it (`"localhost"`, `"127.0.0.1"`, …).
  dest_host: String,
  dest_ip: String,
  dest_port: i128,
  /// The listening socket an accepted connection came in through.
  parent: Option<String>,
  /// Connections accepted through this listening socket, oldest first.
  children: Vec<String>,
  /// The `SocketListener[id]` attached to this socket, if any. Accepted
  /// sockets inherit their parent's id, which is what makes their reads go
  /// to the handler instead of to the evaluator.
  listener_id: Option<i128>,
  /// Bytes read from the socket but not yet handed to the caller.
  read_buffer: Vec<u8>,
  /// Set once the peer has closed and the buffer has run dry.
  eof: bool,
  closed: bool,
  /// Why a `SocketConnect` never got a connection. Reported on first use,
  /// because wolframscript's `SocketConnect` says nothing at the time.
  connect_error: Option<String>,
  /// The ZeroMQ side of a `"ZMQ"` socket. Such a socket has no `stream` or
  /// `listener` of its own: its connections live inside the
  /// [`zmtp::ZmqSocket`](super::zmtp::ZmqSocket).
  zmq: Option<ZmqEntry>,
}

/// What a ZeroMQ socket keeps on the evaluator's side.
#[cfg(not(target_arch = "wasm32"))]
struct ZmqEntry {
  socket: Arc<super::zmtp::ZmqSocket>,
  /// `"tcp"` or `"TCP"`, depending on how the address was spelled — the
  /// `"Scheme"` property reports it verbatim.
  scheme: String,
  /// `ZMQ_LAST_ENDPOINT`: `"tcp://host:port"`.
  last_endpoint: String,
  /// `"DestinationIPAddress"`: an `IPAddress[…]` when the host resolved,
  /// otherwise the host as given (`"0.0.0.0"` stays a plain string).
  ip: Expr,
  /// The listening descriptor of a bound socket, for `ZMQ_FD`.
  fd: Option<i128>,
  /// Option values set with `SetOptions`, by `ZMQ_…` name, as they read
  /// back through the getters.
  options: Vec<(String, Expr)>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
struct Registry {
  /// UUIDs in creation order, which is the order `Sockets[]` reports.
  order: Vec<String>,
  entries: HashMap<String, SocketEntry>,
}

/// A `SocketListen[…]` in force: what it watches and how to reach the
/// handler. The handler expressions never leave this thread.
#[cfg(not(target_arch = "wasm32"))]
struct ListenerEntry {
  socket: String,
  /// `HandlerFunctions -> <|…|>` broken out by event name. A bare
  /// `SocketListen[sock, f]` registers `f` under `"DataReceived"`.
  handlers: Vec<(String, Expr)>,
  /// The `HandlerFunctionsKeys` option: which keys a handler asked to be
  /// given. It is *not* the set of keys the handler association carries —
  /// wolframscript passes all seven of those whatever this says — it is only
  /// what the listener reports back for the option.
  handler_keys: Vec<String>,
  stop: Arc<AtomicBool>,
}

#[cfg(not(target_arch = "wasm32"))]
enum EventKind {
  Accepted,
  /// A chunk of incoming bytes, and whether it ends its message — always
  /// for TCP, and for the last part of a multipart ZeroMQ message.
  Data(Vec<u8>, bool),
  Closed,
  Error(String),
}

/// One thing that happened to a socket, as seen by a background thread.
#[cfg(not(target_arch = "wasm32"))]
struct SocketEvent {
  /// The connection it happened on. For `Accepted` this is the new
  /// connection, already registered by the accept thread.
  socket: String,
  kind: EventKind,
}

/// The owning thread's end of the event channel, plus the counter the
/// background threads bump so [`pump_socket_events`] can bail out without
/// touching thread-local storage.
#[cfg(not(target_arch = "wasm32"))]
struct Bus {
  tx: Sender<SocketEvent>,
  rx: Receiver<SocketEvent>,
}

/// Events queued across all threads. The pump's fast path is a single
/// relaxed load of this, so putting it on the statement loop costs
/// essentially nothing while no socket is listening.
#[cfg(not(target_arch = "wasm32"))]
static PENDING_EVENTS: AtomicUsize = AtomicUsize::new(0);

#[cfg(not(target_arch = "wasm32"))]
thread_local! {
  static SOCKETS: RefCell<Registry> = RefCell::new(Registry::default());
  static LISTENERS: RefCell<HashMap<i128, ListenerEntry>> =
    RefCell::new(HashMap::new());
  static BUS: Bus = {
    let (tx, rx) = channel();
    Bus { tx, rx }
  };
  /// Guards against a handler that reaches a pump point of its own
  /// (a `Pause`, another `SocketWaitNext`) re-entering the pump.
  static IN_PUMP: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(not(target_arch = "wasm32"))]
fn with_registry<R>(f: impl FnOnce(&mut Registry) -> R) -> R {
  SOCKETS.with(|r| f(&mut r.borrow_mut()))
}

/// Send an event to the owning thread and account for it globally.
#[cfg(not(target_arch = "wasm32"))]
fn post_event(tx: &Sender<SocketEvent>, event: SocketEvent) -> bool {
  PENDING_EVENTS.fetch_add(1, Ordering::Relaxed);
  if tx.send(event).is_ok() {
    true
  } else {
    // The owning thread is gone; undo the count so the pump's fast path
    // does not stay armed forever.
    PENDING_EVENTS.fetch_sub(1, Ordering::Relaxed);
    false
  }
}

/// Close every socket this thread opened and stop the threads behind them.
/// Called from `clear_state`, so one test cannot inherit another's server.
#[cfg(not(target_arch = "wasm32"))]
pub fn clear_sockets() {
  LISTENERS.with(|l| {
    for entry in l.borrow().values() {
      entry.stop.store(true, Ordering::Relaxed);
    }
    l.borrow_mut().clear();
  });
  with_registry(|reg| {
    for entry in reg.entries.values_mut() {
      shutdown_entry(entry);
    }
    reg.entries.clear();
    reg.order.clear();
  });
  // Whatever the stopped threads had already queued is now meaningless.
  BUS.with(|bus| {
    while bus.rx.try_recv().is_ok() {
      PENDING_EVENTS.fetch_sub(1, Ordering::Relaxed);
    }
  });
}

/// The wasm build has no sockets, so there is nothing to clear.
#[cfg(target_arch = "wasm32")]
pub fn clear_sockets() {}

#[cfg(not(target_arch = "wasm32"))]
fn shutdown_entry(entry: &mut SocketEntry) {
  if let Some(stream) = entry.stream.take() {
    let _ = stream.shutdown(std::net::Shutdown::Both);
  }
  if let Some(zmq) = &entry.zmq {
    zmq.socket.close();
  }
  entry.listener = None;
  entry.closed = true;
}

// ---------------------------------------------------------------------------
// Object identity
// ---------------------------------------------------------------------------

/// A fresh socket UUID. Listening sockets carry wolframscript's
/// `TCPSERVER-` prefix; connections get the bare UUID.
#[cfg(not(target_arch = "wasm32"))]
fn new_uuid(server: bool) -> String {
  let prefix = if server { SERVER_UUID_PREFIX } else { "" };
  let args = [str_expr(prefix)];
  let created =
    crate::evaluator::dispatch::predicate_functions::dispatch_predicate_functions(
      "CreateUUID",
      &args,
    );
  match created.as_ref() {
    Some(Ok(Expr::String(uuid))) => uuid.clone(),
    // `CreateUUID` cannot fail, but a UUID is not worth an unwrap.
    _ => format!("{prefix}00000000-0000-4000-8000-000000000000"),
  }
}

/// The `SocketObject["uuid"]` expression for a registry entry.
#[cfg(not(target_arch = "wasm32"))]
fn socket_expr(uuid: &str) -> Expr {
  call1("SocketObject", str_expr(uuid))
}

/// The UUID inside a `SocketObject["uuid"]`, whatever its registry state.
/// Recognizing the shape and finding it open are separate questions: a
/// closed socket still has to give the "invalid or not open" message
/// rather than fall through to a type error.
pub fn socket_object_uuid(expr: &Expr) -> Option<String> {
  match expr {
    Expr::FunctionCall { name, args }
      if name == "SocketObject" && args.len() == 1 =>
    {
      match &args[0] {
        Expr::String(uuid) => Some(uuid.clone()),
        _ => None,
      }
    }
    _ => None,
  }
}

/// The id inside a `SocketListener[id]`.
#[cfg(not(target_arch = "wasm32"))]
fn listener_object_id(expr: &Expr) -> Option<i128> {
  match expr {
    Expr::FunctionCall { name, args }
      if name == "SocketListener" && args.len() == 1 =>
    {
      match &args[0] {
        Expr::Integer(id) => Some(*id),
        _ => None,
      }
    }
    _ => None,
  }
}

/// wolframscript's free-text complaint about a socket that is closed, was
/// never connected, or belongs to another session. Not a tagged message:
/// it leaves `$MessageList` alone.
#[cfg(not(target_arch = "wasm32"))]
fn invalid_socket_message(uuid: &str) {
  crate::emit_message_to_stdout(&format!(
    "The socket object {} is invalid or not open.",
    expr_to_string(&socket_expr(uuid))
  ));
}

/// Report a socket operation that the operating system refused.
#[cfg(not(target_arch = "wasm32"))]
fn failed_operation_message(err: &std::io::Error) {
  crate::emit_message_to_stdout(&format!("Failed socket operation: {err}"));
}

// ---------------------------------------------------------------------------
// Address parsing
// ---------------------------------------------------------------------------

/// A socket endpoint as the socket functions accept it: `8000`,
/// `"127.0.0.1:8000"`, `"localhost", 8000` or `{"localhost", 8000}`.
#[cfg(not(target_arch = "wasm32"))]
struct Endpoint {
  host: String,
  port: u16,
}

/// What the protocol argument of `SocketOpen` / `SocketConnect` asks for.
#[cfg(not(target_arch = "wasm32"))]
enum SocketSpec {
  Tcp,
  /// A ZeroMQ socket of the given `zmq.h` type (`"ZMQ_PUB"`, …).
  Zmq(String),
}

/// The ZeroMQ socket types wolframscript accepts, `ZMQ_STREAM` included.
#[cfg(not(target_arch = "wasm32"))]
const ZMQ_TYPES: [&str; 12] = [
  "ZMQ_PAIR",
  "ZMQ_PUB",
  "ZMQ_SUB",
  "ZMQ_REQ",
  "ZMQ_REP",
  "ZMQ_DEALER",
  "ZMQ_ROUTER",
  "ZMQ_PULL",
  "ZMQ_PUSH",
  "ZMQ_XPUB",
  "ZMQ_XSUB",
  "ZMQ_STREAM",
];

/// Read a protocol specification the way wolframscript's `socketspecs`
/// does: `"TCP"`, `{"TCP"}`, `"ZMQ"` (a PAIR socket), `"ZMQ_SUB"`,
/// `"zmq_sub"`, `{"ZMQ", "Subscribe"}`, `{"ZMQ", "SUB"}`, … The type is
/// case-insensitive and may be spelled out (`"Publish"`, `"Request"`, …).
/// Anything else is a `noproto` failure, or `invproto` for a ZMQ type that
/// does not exist.
#[cfg(not(target_arch = "wasm32"))]
fn socket_spec(head: &str, protocol: &Expr) -> Result<SocketSpec, Expr> {
  match protocol {
    Expr::List(items) => match &items.iter().collect::<Vec<_>>()[..] {
      [Expr::String(single)] => socket_spec_string(head, single),
      [Expr::String(protocol), Expr::String(kind)] => {
        zmq_spec(head, protocol, kind)
      }
      _ => Err(no_protocol(head, &expr_to_string(protocol))),
    },
    Expr::String(spec) => socket_spec_string(head, spec),
    other => Err(no_protocol(head, &expr_to_string(other))),
  }
}

#[cfg(not(target_arch = "wasm32"))]
fn socket_spec_string(head: &str, spec: &str) -> Result<SocketSpec, Expr> {
  if spec.eq_ignore_ascii_case("TCP") {
    return Ok(SocketSpec::Tcp);
  }
  // Only a *leading* "ZMQ_" or "ZMQ" counts; anything else is not a
  // protocol at all.
  let upper = spec.to_ascii_uppercase();
  if upper.starts_with("ZMQ_") {
    zmq_spec(head, "ZMQ", &spec[4..])
  } else if upper.starts_with("ZMQ") {
    Ok(SocketSpec::Zmq("ZMQ_PAIR".to_string()))
  } else {
    Err(no_protocol(head, spec))
  }
}

/// `{protocol, type}`: the type, with any missing `ZMQ_` prefix added and
/// the long names folded to the `zmq.h` ones.
#[cfg(not(target_arch = "wasm32"))]
fn zmq_spec(
  head: &str,
  protocol: &str,
  kind: &str,
) -> Result<SocketSpec, Expr> {
  let joined = format!("{}_{kind}", protocol.to_ascii_uppercase());
  let upper = joined.to_ascii_uppercase();
  let full = if upper.starts_with("ZMQ_") {
    upper
  } else {
    format!("ZMQ_{upper}")
  };
  let full = match full.as_str() {
    "ZMQ_PUBLISH" => "ZMQ_PUB".to_string(),
    "ZMQ_SUBSCRIBE" => "ZMQ_SUB".to_string(),
    "ZMQ_REQUEST" => "ZMQ_REQ".to_string(),
    "ZMQ_REPLY" => "ZMQ_REP".to_string(),
    "ZMQ_XPUBLISH" => "ZMQ_XPUB".to_string(),
    "ZMQ_XSUBSCRIBE" => "ZMQ_XSUB".to_string(),
    _ => full,
  };
  if protocol.eq_ignore_ascii_case("ZMQ") && ZMQ_TYPES.contains(&full.as_str())
  {
    Ok(SocketSpec::Zmq(full))
  } else {
    Err(sockets_failure(
      head,
      "invproto",
      vec![str_expr(protocol), str_expr(kind)],
    ))
  }
}

/// `Failure["SocketsLink", <|"MessageTemplate" :> head::tag, …|>]`, the
/// object wolframscript's socket functions return instead of printing a
/// message.
#[cfg(not(target_arch = "wasm32"))]
fn sockets_failure(head: &str, tag: &str, params: Vec<Expr>) -> Expr {
  let template = call("MessageName", vec![id_expr(head), str_expr(tag)]);
  let key = str_expr("MessageTemplate");
  call(
    "Failure",
    vec![
      str_expr("SocketsLink"),
      Expr::Association(vec![
        (key.clone(), rule_delayed_expr(key, template)),
        (str_expr("MessageParameters"), Expr::List(params.into())),
      ]),
    ],
  )
}

/// The failure for a protocol specification that names no protocol.
#[cfg(not(target_arch = "wasm32"))]
fn no_protocol(head: &str, protocol: &str) -> Expr {
  sockets_failure(head, "noproto", vec![str_expr(protocol)])
}

/// Read a TCP endpoint: `8000`, `"host:port"`, `"tcp://host:port"` or
/// `{"host", 8000}`. The protocol is not part of it — that is the second
/// argument, so `SocketConnect["127.0.0.1", 8000]` is a protocol error, not
/// a host/port pair.
#[cfg(not(target_arch = "wasm32"))]
fn parse_endpoint(spec: &Expr, default_host: &str) -> Option<Endpoint> {
  match spec {
    Expr::Integer(port) => Some(Endpoint {
      host: default_host.to_string(),
      port: u16::try_from(*port).ok()?,
    }),
    Expr::String(spec) => parse_host_port(spec, default_host),
    Expr::List(items) => match &items.iter().collect::<Vec<_>>()[..] {
      [Expr::String(host), Expr::Integer(port)] => Some(Endpoint {
        host: host.clone(),
        port: u16::try_from(*port).ok()?,
      }),
      _ => None,
    },
    _ => None,
  }
}

/// A ZeroMQ endpoint. `port` is `None` for `*`, "any free port", which only
/// a bound socket may ask for.
#[cfg(not(target_arch = "wasm32"))]
struct ZmqAddress {
  /// As wolframscript records it: `"tcp"` when the address named its
  /// transport (or was a bare port), `"TCP"` when it had to be assumed.
  scheme: String,
  host: String,
  port: Option<u16>,
}

/// Read a ZeroMQ address: `5556`, `"tcp://host:5556"`, `"host:5556"`,
/// `{"host", 5556}`, `{IPAddress["…"], 5556}`, `Automatic`, or `*` for the
/// port of a socket being bound. `localhost` is recorded as `127.0.0.1`,
/// the way wolframscript's host parser rewrites it.
#[cfg(not(target_arch = "wasm32"))]
fn parse_zmq_address(
  head: &str,
  spec: &Expr,
  connecting: bool,
) -> Result<ZmqAddress, Expr> {
  let addrspec = |reason: &str| {
    sockets_failure(head, "addrspec", vec![spec.clone(), str_expr(reason)])
  };
  let invalid = || addrspec("the domain or port is invalid or missing");
  let port_of = |text: &str| -> Result<Option<u16>, Expr> {
    if text == "*" {
      return Ok(None);
    }
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
      return Err(invalid());
    }
    text.parse::<u16>().map(Some).map_err(|_| invalid())
  };
  let host_of = |host: &str| {
    if host.eq_ignore_ascii_case("localhost") {
      "127.0.0.1".to_string()
    } else {
      host.to_string()
    }
  };
  let address = match spec {
    Expr::Integer(port) => ZmqAddress {
      scheme: "tcp".to_string(),
      host: "127.0.0.1".to_string(),
      port: Some(u16::try_from(*port).map_err(|_| invalid())?),
    },
    Expr::Identifier(name) if name == "Automatic" => ZmqAddress {
      scheme: "tcp".to_string(),
      host: "127.0.0.1".to_string(),
      port: None,
    },
    Expr::String(text) => {
      let (scheme, rest) = match text.find("://") {
        Some(at) => (text[..at].to_string(), &text[at + 3..]),
        None => ("TCP".to_string(), text.as_str()),
      };
      if !scheme.eq_ignore_ascii_case("tcp") {
        // `inproc://` and `ipc://` have no TCP connection behind them.
        return Err(sockets_failure(
          head,
          "zmqinvcmd",
          vec![str_expr("Protocol not supported")],
        ));
      }
      let (host, port) = if let Some(inner) = rest.strip_prefix('[') {
        let close = inner.find(']').ok_or_else(invalid)?;
        (&inner[..close], inner[close + 1..].strip_prefix(':'))
      } else {
        match rest.rfind(':') {
          Some(colon) => (&rest[..colon], Some(&rest[colon + 1..])),
          None => (rest, None),
        }
      };
      let Some(port) = port else {
        return Err(addrspec("it is missing the port specification"));
      };
      if host.is_empty() {
        return Err(invalid());
      }
      ZmqAddress {
        scheme,
        host: host_of(host),
        port: port_of(port)?,
      }
    }
    Expr::List(items) => {
      let items: Vec<&Expr> = items.iter().collect();
      let [host, port] = items[..] else {
        return Err(invalid());
      };
      let host = match host {
        Expr::String(h) => h.clone(),
        Expr::FunctionCall { name, args }
          if name == "IPAddress" && args.len() == 1 =>
        {
          match &args[0] {
            Expr::String(h) => h.clone(),
            _ => return Err(invalid()),
          }
        }
        _ => return Err(invalid()),
      };
      let port = match port {
        Expr::Integer(p) => Some(u16::try_from(*p).map_err(|_| invalid())?),
        Expr::String(p) => port_of(p)?,
        _ => return Err(invalid()),
      };
      ZmqAddress {
        scheme: "TCP".to_string(),
        host: host_of(&host),
        port,
      }
    }
    _ => return Err(invalid()),
  };
  if connecting && address.port.is_none() {
    return Err(addrspec(
      "the port must be a valid Integer when connecting a socket",
    ));
  }
  Ok(address)
}

/// `"DestinationIPAddress"` of a ZeroMQ socket: the first address the host
/// resolves to, or the host itself when it does not resolve (which is how
/// the wildcard `0.0.0.0` comes out).
#[cfg(not(target_arch = "wasm32"))]
fn resolved_ip(host: &str) -> Expr {
  if host == "0.0.0.0" || host == "*" {
    return str_expr(host);
  }
  match (host, 0).to_socket_addrs().ok().and_then(|mut a| a.next()) {
    Some(addr) => call1(
      "IPAddress",
      Expr::String(addr.ip().to_canonical().to_string()),
    ),
    None => str_expr(host),
  }
}

/// The operating system's text for an error, without Rust's
/// `" (os error N)"` suffix — the way libzmq's `zmq_strerror` reads.
#[cfg(not(target_arch = "wasm32"))]
fn os_error_text(err: &std::io::Error) -> String {
  let text = err.to_string();
  match text.find(" (os error") {
    Some(at) => text[..at].to_string(),
    None => text,
  }
}

/// Split `"host:port"`, `"tcp://host:port"` or a bare `"port"`.
#[cfg(not(target_arch = "wasm32"))]
fn parse_host_port(spec: &str, default_host: &str) -> Option<Endpoint> {
  let spec = spec
    .strip_prefix("tcp://")
    .or_else(|| spec.strip_prefix("http://"))
    .unwrap_or(spec);
  // A bare number is a port on the default host, the way
  // `SocketConnect["8000"]` reads.
  if let Ok(port) = spec.parse::<u16>() {
    return Some(Endpoint {
      host: default_host.to_string(),
      port,
    });
  }
  // `[::1]:8000` keeps the brackets around a literal IPv6 host.
  let (host, port) = if let Some(rest) = spec.strip_prefix('[') {
    let close = rest.find(']')?;
    (&rest[..close], rest[close + 1..].strip_prefix(':')?)
  } else {
    let colon = spec.rfind(':')?;
    (&spec[..colon], &spec[colon + 1..])
  };
  Some(Endpoint {
    host: host.to_string(),
    port: port.parse::<u16>().ok()?,
  })
}

// ---------------------------------------------------------------------------
// Opening and connecting
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
fn register(uuid: &str, entry: SocketEntry) -> Expr {
  with_registry(|reg| {
    reg.order.push(uuid.to_string());
    reg.entries.insert(uuid.to_string(), entry);
  });
  socket_expr(uuid)
}

/// `SocketOpen[port]` — bind and start listening. Port 0 asks the operating
/// system for a free one, which `sock["DestinationPort"]` then reports.
#[cfg(not(target_arch = "wasm32"))]
fn socket_open(args: &[Expr]) -> Expr {
  if let Some(protocol) = args.get(1) {
    match socket_spec("SocketOpen", protocol) {
      Err(failure) => return failure,
      Ok(SocketSpec::Zmq(kind)) => return zmq_open(&args[0], &kind),
      Ok(SocketSpec::Tcp) => {}
    }
  }
  let Some(endpoint) = parse_endpoint(&args[0], "127.0.0.1") else {
    return unevaluated("SocketOpen", args);
  };
  let listener =
    match TcpListener::bind((endpoint.host.as_str(), endpoint.port)) {
      Ok(listener) => listener,
      Err(err) => {
        failed_operation_message(&err);
        return fail_expr();
      }
    };
  let local = listener.local_addr().ok();
  let port = local.map_or(i128::from(endpoint.port), |a| i128::from(a.port()));
  let ip = local.map_or_else(
    || endpoint.host.clone(),
    |a| a.ip().to_canonical().to_string(),
  );
  register(
    &new_uuid(true),
    SocketEntry {
      role: SocketRole::Server,
      stream: None,
      listener: Some(Arc::new(listener)),
      dest_host: endpoint.host,
      dest_ip: ip,
      dest_port: port,
      parent: None,
      children: Vec::new(),
      listener_id: None,
      read_buffer: Vec::new(),
      eof: false,
      closed: false,
      connect_error: None,
      zmq: None,
    },
  )
}

/// `SocketConnect[…]` — the near end of an outgoing TCP connection.
///
/// wolframscript hands back a `SocketObject` even for an address nothing is
/// listening on and says nothing about it; the complaint only comes when
/// something is read or written. The connection is attempted here all the
/// same — the alternative is to re-attempt it on every use — and a failure
/// is parked in `connect_error` until the first operation asks for it.
#[cfg(not(target_arch = "wasm32"))]
fn socket_connect(args: &[Expr]) -> Expr {
  if let Some(protocol) = args.get(1) {
    match socket_spec("SocketConnect", protocol) {
      Err(failure) => return failure,
      Ok(SocketSpec::Zmq(kind)) => return zmq_connect(&args[0], &kind),
      Ok(SocketSpec::Tcp) => {}
    }
  }
  let Some(endpoint) = parse_endpoint(&args[0], "127.0.0.1") else {
    return unevaluated("SocketConnect", args);
  };
  let resolved: Vec<SocketAddr> = (endpoint.host.as_str(), endpoint.port)
    .to_socket_addrs()
    .map_or_default(std::iter::Iterator::collect);
  let mut error = None;
  let mut stream = None;
  for addr in &resolved {
    match TcpStream::connect_timeout(addr, Duration::from_secs(10)) {
      Ok(s) => {
        stream = Some(s);
        break;
      }
      Err(err) => error = Some(err.to_string()),
    }
  }
  if resolved.is_empty() {
    error = Some(format!("cannot resolve {}", endpoint.host));
  }
  let peer = stream.as_ref().and_then(|s| s.peer_addr().ok());
  let ip = peer.map_or_else(
    || endpoint.host.clone(),
    |a| a.ip().to_canonical().to_string(),
  );
  register(
    &new_uuid(false),
    SocketEntry {
      role: SocketRole::Client,
      // `TCP_NODELAY` so a request written in one call is not held back
      // waiting for a second one; a socket API that coalesces small writes
      // makes every request/response exchange look like a hang.
      stream: stream.map(|s| {
        let _ = s.set_nodelay(true);
        Arc::new(s)
      }),
      dest_host: endpoint.host,
      dest_ip: ip,
      dest_port: i128::from(endpoint.port),
      listener: None,
      parent: None,
      children: Vec::new(),
      listener_id: None,
      read_buffer: Vec::new(),
      eof: false,
      closed: false,
      connect_error: error,
      zmq: None,
    },
  )
}

/// The ZeroMQ type to run, or the failure for `ZMQ_STREAM`, the one type
/// whose raw-TCP framing is not implemented.
#[cfg(not(target_arch = "wasm32"))]
fn zmq_type(head: &str, kind: &str) -> Result<super::zmtp::ZmqType, Expr> {
  super::zmtp::ZmqType::from_name(kind).ok_or_else(|| {
    sockets_failure(
      head,
      "zmqinvcmd",
      vec![str_expr("ZMQ_STREAM sockets are not supported")],
    )
  })
}

/// `SocketOpen[address, {"ZMQ", type}]` — bind a ZeroMQ socket.
#[cfg(not(target_arch = "wasm32"))]
fn zmq_open(spec: &Expr, kind: &str) -> Expr {
  let head = "SocketOpen";
  let zmq_kind = match zmq_type(head, kind) {
    Ok(k) => k,
    Err(failure) => return failure,
  };
  let address = match parse_zmq_address(head, spec, false) {
    Ok(a) => a,
    Err(failure) => return failure,
  };
  // libzmq's `*` host is every interface.
  let bind_host = if address.host == "*" {
    "0.0.0.0"
  } else {
    address.host.as_str()
  };
  let bound = match super::zmtp::ZmqSocket::bind(
    zmq_kind,
    bind_host,
    address.port.unwrap_or(0),
  ) {
    Ok(bound) => bound,
    Err(err) => {
      return sockets_failure(
        head,
        "zmqinvcmd",
        vec![Expr::String(os_error_text(&err))],
      );
    }
  };
  let port = bound.local.port();
  register(
    &new_uuid(false),
    SocketEntry {
      role: SocketRole::Server,
      stream: None,
      listener: None,
      dest_ip: address.host.clone(),
      dest_port: i128::from(port),
      parent: None,
      children: Vec::new(),
      listener_id: None,
      read_buffer: Vec::new(),
      eof: false,
      closed: false,
      connect_error: None,
      zmq: Some(ZmqEntry {
        socket: bound.socket,
        // A bound socket reports the scheme of its last endpoint, which
        // libzmq always spells in lower case.
        scheme: "tcp".to_string(),
        last_endpoint: format!("tcp://{bind_host}:{port}"),
        ip: resolved_ip(&address.host),
        fd: Some(bound.fd),
        options: Vec::new(),
      }),
      dest_host: address.host,
    },
  )
}

/// `SocketConnect[address, {"ZMQ", type}]` — connect a ZeroMQ socket. As in
/// libzmq the connection is made (and remade) in the background; a peer
/// that is already listening is connected by the time this returns.
#[cfg(not(target_arch = "wasm32"))]
fn zmq_connect(spec: &Expr, kind: &str) -> Expr {
  let head = "SocketConnect";
  let zmq_kind = match zmq_type(head, kind) {
    Ok(k) => k,
    Err(failure) => return failure,
  };
  let address = match parse_zmq_address(head, spec, true) {
    Ok(a) => a,
    Err(failure) => return failure,
  };
  let port = address.port.unwrap_or(0);
  let socket = super::zmtp::ZmqSocket::connect(zmq_kind, &address.host, port);
  register(
    &new_uuid(false),
    SocketEntry {
      role: SocketRole::Client,
      stream: None,
      listener: None,
      dest_ip: address.host.clone(),
      dest_port: i128::from(port),
      parent: None,
      children: Vec::new(),
      listener_id: None,
      read_buffer: Vec::new(),
      eof: false,
      closed: false,
      connect_error: None,
      zmq: Some(ZmqEntry {
        socket,
        scheme: address.scheme,
        last_endpoint: format!("tcp://{}:{port}", address.host),
        ip: resolved_ip(&address.host),
        fd: None,
        options: Vec::new(),
      }),
      dest_host: address.host,
    },
  )
}

// ---------------------------------------------------------------------------
// Listening
// ---------------------------------------------------------------------------

/// The event name wolframscript files an incoming chunk under.
#[cfg(not(target_arch = "wasm32"))]
const DATA_RECEIVED: &str = "DataReceived";

/// The default `HandlerFunctionsKeys` a listener reports.
#[cfg(not(target_arch = "wasm32"))]
const DEFAULT_HANDLER_FUNCTIONS_KEYS: [&str; 4] =
  ["Timestamp", "Socket", "SourceSocket", "Data"];

/// The handler functions a `SocketListen` call sets up, as
/// `(event name, function)` pairs, together with its `HandlerFunctionsKeys`.
///
/// `SocketListen[sock, f]` is the short form for
/// `HandlerFunctions -> <|"DataReceived" -> f|>`, the only event
/// wolframscript names; the long form may also name Woxi's `"Accepted"`,
/// `"Closed"` and `"Error"`.
#[cfg(not(target_arch = "wasm32"))]
fn parse_handlers(spec: &[Expr]) -> Option<(Vec<(String, Expr)>, Vec<String>)> {
  let mut handlers = Vec::new();
  let mut keys: Vec<String> = DEFAULT_HANDLER_FUNCTIONS_KEYS
    .iter()
    .map(|k| (*k).to_string())
    .collect();
  for arg in spec {
    match arg {
      // HandlerFunctionsKeys -> {…} only says which keys the handler wants;
      // it never becomes a handler itself.
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } if matches!(&**pattern,
          Expr::Identifier(n) | Expr::String(n)
            if n == "HandlerFunctionsKeys") =>
      {
        let Expr::List(items) = &**replacement else {
          return None;
        };
        keys = items
          .iter()
          .map(|item| match item {
            Expr::String(s) | Expr::Identifier(s) => Some(s.clone()),
            _ => None,
          })
          .collect::<Option<Vec<_>>>()?;
      }
      // HandlerFunctions -> <|"Received" :> f, …|>
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } if matches!(&**pattern,
          Expr::Identifier(n) | Expr::String(n) if n == "HandlerFunctions") =>
      {
        match &**replacement {
          Expr::Association(pairs) => {
            for (key, value) in pairs {
              let name = match key {
                Expr::String(s) => s.clone(),
                Expr::Identifier(s) => s.clone(),
                _ => return None,
              };
              handlers.push((
                name,
                crate::functions::association_ast::assoc_entry_value(
                  key, value,
                ),
              ));
            }
          }
          Expr::List(items) => {
            for item in items {
              let (Expr::Rule {
                pattern,
                replacement,
              }
              | Expr::RuleDelayed {
                pattern,
                replacement,
              }) = item
              else {
                return None;
              };
              let name = match &**pattern {
                Expr::String(s) => s.clone(),
                Expr::Identifier(s) => s.clone(),
                _ => return None,
              };
              handlers.push((name, (**replacement).clone()));
            }
          }
          _ => return None,
        }
      }
      // Anything else in the second position is the "DataReceived" handler.
      other => handlers.push((DATA_RECEIVED.to_string(), other.clone())),
    }
  }
  Some((handlers, keys))
}

/// `SocketListen[sock, handler]` — run `handler` for everything that
/// arrives on `sock`, and start accepting connections if it is a server.
#[cfg(not(target_arch = "wasm32"))]
fn socket_listen(args: &[Expr]) -> Expr {
  let Some(uuid) = socket_object_uuid(&args[0]) else {
    return unevaluated("SocketListen", args);
  };
  let Some((handlers, handler_keys)) = parse_handlers(&args[1..]) else {
    return unevaluated("SocketListen", args);
  };
  let listener_state = with_registry(|reg| {
    reg.entries.get(&uuid).map(|entry| {
      (
        entry.closed,
        entry.role,
        entry.listener.clone(),
        entry.stream.clone(),
        entry.zmq.as_ref().map(|z| z.socket.clone()),
      )
    })
  });
  let Some((closed, role, listener, stream, zmq)) = listener_state else {
    invalid_socket_message(&uuid);
    return fail_expr();
  };
  if closed {
    invalid_socket_message(&uuid);
    return fail_expr();
  }
  // Listening twice on one socket would leave two accept threads fighting
  // over it, so the second call replaces the first.
  if let Some(previous) =
    with_registry(|reg| reg.entries.get(&uuid).and_then(|e| e.listener_id))
  {
    stop_listener(previous);
  }

  // wolframscript's listener ids are large random integers, not a counter,
  // so nothing can depend on the order listeners were created in.
  let id = {
    let mut bytes = [0u8; 8];
    crate::with_rng(|rng| rand::Rng::fill_bytes(rng, &mut bytes));
    i128::from(u64::from_be_bytes(bytes) >> 1)
  };
  let stop = Arc::new(AtomicBool::new(false));
  LISTENERS.with(|l| {
    l.borrow_mut().insert(
      id,
      ListenerEntry {
        socket: uuid.clone(),
        handlers,
        handler_keys,
        stop: stop.clone(),
      },
    );
  });
  with_registry(|reg| {
    if let Some(entry) = reg.entries.get_mut(&uuid) {
      entry.listener_id = Some(id);
    }
  });

  let tx = BUS.with(|bus| bus.tx.clone());
  // A ZeroMQ socket delivers whole messages, part by part; its connection
  // threads post them as they arrive.
  if let Some(zmq) = zmq {
    let source = uuid.clone();
    zmq.set_listener(Box::new(move |bytes, complete| {
      post_event(
        &tx,
        SocketEvent {
          socket: source.clone(),
          kind: EventKind::Data(bytes, complete),
        },
      );
    }));
    return call1("SocketListener", Expr::Integer(id));
  }
  match role {
    SocketRole::Server => {
      if let Some(listener) = listener {
        spawn_accept_thread(uuid, listener, id, stop, tx);
      }
    }
    // Listening on a connection means "hand me its incoming bytes", so the
    // read thread starts straight away.
    SocketRole::Client | SocketRole::Accepted => {
      if let Some(stream) = stream {
        spawn_read_thread(uuid, stream, stop, tx);
      }
    }
  }
  call1("SocketListener", Expr::Integer(id))
}

/// Accept connections until the listener is deleted. Each one is registered
/// as a socket in its own right and gets a read thread, so the handler sees
/// `SourceSocket` for the connection and `Socket` for the server.
#[cfg(not(target_arch = "wasm32"))]
fn spawn_accept_thread(
  server_uuid: String,
  listener: Arc<TcpListener>,
  listener_id: i128,
  stop: Arc<AtomicBool>,
  tx: Sender<SocketEvent>,
) {
  // Accepting has to give up the CPU while it waits, and `accept` itself
  // has no timeout — so the listener is polled instead, which is also what
  // lets `DeleteObject` stop the thread promptly.
  let _ = listener.set_nonblocking(true);
  // Registering an accepted socket means touching the owning thread's
  // registry, which this thread cannot do. The pump does it instead, from
  // the parts handed over here.
  let pending: AcceptedQueue = Arc::new(Mutex::new(Vec::new()));
  ACCEPTED.with(|a| a.borrow_mut().push(pending.clone()));
  std::thread::spawn(move || {
    while !stop.load(Ordering::Relaxed) {
      match listener.accept() {
        Ok((stream, peer)) => {
          let _ = stream.set_nodelay(true);
          let _ = stream.set_nonblocking(false);
          let stream = Arc::new(stream);
          let uuid = uuid_for_accepted();
          if let Ok(mut pending) = pending.lock() {
            pending.push(AcceptedConn {
              uuid: uuid.clone(),
              stream: stream.clone(),
              peer_ip: peer.ip().to_canonical().to_string(),
              peer_port: i128::from(peer.port()),
              server: server_uuid.clone(),
              listener_id,
            });
          }
          if !post_event(
            &tx,
            SocketEvent {
              socket: uuid.clone(),
              kind: EventKind::Accepted,
            },
          ) {
            break;
          }
          spawn_read_thread(uuid, stream, stop.clone(), tx.clone());
        }
        Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
          std::thread::sleep(Duration::from_millis(2));
        }
        Err(err) => {
          post_event(
            &tx,
            SocketEvent {
              socket: server_uuid.clone(),
              kind: EventKind::Error(err.to_string()),
            },
          );
          break;
        }
      }
    }
  });
}

/// Feed everything a connection produces into the event channel until the
/// peer closes it or the listener is deleted.
#[cfg(not(target_arch = "wasm32"))]
fn spawn_read_thread(
  uuid: String,
  stream: Arc<TcpStream>,
  stop: Arc<AtomicBool>,
  tx: Sender<SocketEvent>,
) {
  // A read timeout, not a non-blocking socket: a timed-out read consumes
  // nothing and blocks in the kernel until data arrives, so checking the
  // stop flag costs nothing while bytes are flowing.
  let _ = stream.set_read_timeout(Some(POLL_INTERVAL));
  std::thread::spawn(move || {
    let mut buffer = vec![0u8; READ_CHUNK];
    while !stop.load(Ordering::Relaxed) {
      let read = (&*stream).read(&mut buffer);
      let event = match read {
        Ok(0) => EventKind::Closed,
        Ok(n) => EventKind::Data(buffer[..n].to_vec(), true),
        Err(err)
          if matches!(
            err.kind(),
            std::io::ErrorKind::WouldBlock
              | std::io::ErrorKind::TimedOut
              | std::io::ErrorKind::Interrupted
          ) =>
        {
          continue;
        }
        Err(err) => EventKind::Error(err.to_string()),
      };
      let last = !matches!(event, EventKind::Data(..));
      if !post_event(
        &tx,
        SocketEvent {
          socket: uuid.clone(),
          kind: event,
        },
      ) || last
      {
        break;
      }
    }
  });
}

/// A connection an accept thread has taken but the owning thread has not
/// registered yet.
#[cfg(not(target_arch = "wasm32"))]
struct AcceptedConn {
  uuid: String,
  stream: Arc<TcpStream>,
  peer_ip: String,
  peer_port: i128,
  server: String,
  listener_id: i128,
}

/// Connections accepted but not yet registered, one queue per accept
/// thread. Filled by the accept threads, drained by the pump.
#[cfg(not(target_arch = "wasm32"))]
type AcceptedQueue = Arc<Mutex<Vec<AcceptedConn>>>;

#[cfg(not(target_arch = "wasm32"))]
thread_local! {
  static ACCEPTED: RefCell<Vec<AcceptedQueue>> = const { RefCell::new(Vec::new()) };
}

/// UUIDs for accepted connections, generated off the owning thread.
///
/// `CreateUUID` draws on the evaluator's thread-local RNG, which an accept
/// thread has no access to, so these come from the operating system's
/// randomness instead. The result is the same shape: a version-4 UUID.
#[cfg(not(target_arch = "wasm32"))]
fn uuid_for_accepted() -> String {
  let mut bytes = [0u8; 16];
  {
    use rand::TryRng;
    rand::rngs::SysRng
      .try_fill_bytes(&mut bytes)
      .expect("operating system randomness unavailable");
  }
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const HEX: &[u8; 16] = b"0123456789abcdef";
  let mut hex = String::with_capacity(32);
  for byte in bytes {
    hex.push(char::from(HEX[(byte >> 4) as usize]));
    hex.push(char::from(HEX[(byte & 0x0f) as usize]));
  }
  format!(
    "{}-{}-{}-{}-{}",
    &hex[0..8],
    &hex[8..12],
    &hex[12..16],
    &hex[16..20],
    &hex[20..32]
  )
}

// ---------------------------------------------------------------------------
// The pump
// ---------------------------------------------------------------------------

/// Run the handlers for everything the background threads have delivered.
///
/// Called wherever evaluation would otherwise be busy or idle for a while —
/// `Pause`, the socket waits, and between top-level statements — because
/// handlers run on this thread and can run nowhere else. Cheap when nothing
/// is listening: one relaxed atomic load.
pub fn pump_socket_events() {
  #[cfg(not(target_arch = "wasm32"))]
  {
    drain_events();
  }
}

/// Take every queued event, run its handler, and report which sockets they
/// arrived on — the listening socket for a connection under a listener, so
/// a caller waiting on a server sees its clients' traffic.
#[cfg(not(target_arch = "wasm32"))]
fn drain_events() -> Vec<String> {
  let mut fired = Vec::new();
  if PENDING_EVENTS.load(Ordering::Relaxed) == 0 {
    return fired;
  }
  // A handler that pauses or waits reaches a pump point of its own; letting
  // it run the queue would reorder events behind the handler's own back.
  if IN_PUMP.with(std::cell::Cell::get) {
    return fired;
  }
  IN_PUMP.with(|f| f.set(true));
  loop {
    let event = BUS.with(|bus| bus.rx.try_recv().ok());
    let Some(event) = event else { break };
    PENDING_EVENTS.fetch_sub(1, Ordering::Relaxed);
    // An `Accepted` event names a socket the accept thread could not
    // register itself, so the registry catches up before the lookup.
    register_accepted();
    if let Some(source) = handle_event(&event) {
      fired.push(source);
    }
  }
  IN_PUMP.with(|f| f.set(false));
  fired
}

/// Move everything the accept threads have taken into the registry.
#[cfg(not(target_arch = "wasm32"))]
fn register_accepted() {
  let taken: Vec<AcceptedConn> = ACCEPTED.with(|queues| {
    queues
      .borrow()
      .iter()
      .filter_map(|queue| {
        queue.lock().ok().map(|mut q| std::mem::take(&mut *q))
      })
      .flatten()
      .collect()
  });
  if taken.is_empty() {
    return;
  }
  with_registry(|reg| {
    for conn in taken {
      if let Some(parent) = reg.entries.get_mut(&conn.server) {
        parent.children.push(conn.uuid.clone());
      }
      reg.order.push(conn.uuid.clone());
      reg.entries.insert(
        conn.uuid.clone(),
        SocketEntry {
          role: SocketRole::Accepted,
          stream: Some(conn.stream),
          listener: None,
          dest_host: conn.peer_ip.clone(),
          dest_ip: conn.peer_ip,
          dest_port: conn.peer_port,
          parent: Some(conn.server),
          children: Vec::new(),
          listener_id: Some(conn.listener_id),
          read_buffer: Vec::new(),
          eof: false,
          closed: false,
          connect_error: None,
          zmq: None,
        },
      );
    }
  });
}

/// The listening socket a connection belongs to — itself, for a socket
/// that `SocketListen` was called on directly.
#[cfg(not(target_arch = "wasm32"))]
fn owning_socket(uuid: &str) -> String {
  with_registry(|reg| {
    reg
      .entries
      .get(uuid)
      .and_then(|entry| entry.parent.clone())
      .unwrap_or_else(|| uuid.to_string())
  })
}

/// Apply the handler an event calls for. Returns the listening socket it
/// arrived on, so the socket waits can tell whose event it was.
#[cfg(not(target_arch = "wasm32"))]
fn handle_event(event: &SocketEvent) -> Option<String> {
  let listener_id = with_registry(|reg| {
    reg.entries.get(&event.socket).and_then(|e| e.listener_id)
  })?;
  let owner = owning_socket(&event.socket);
  if matches!(event.kind, EventKind::Closed) {
    with_registry(|reg| {
      if let Some(entry) = reg.entries.get_mut(&event.socket) {
        entry.eof = true;
      }
    });
  }
  let event_name = match &event.kind {
    EventKind::Accepted => "Accepted",
    EventKind::Data(..) => DATA_RECEIVED,
    EventKind::Closed => "Closed",
    EventKind::Error(_) => "Error",
  };
  let (listen_socket, handler) = LISTENERS.with(|l| {
    let listeners = l.borrow();
    let entry = listeners.get(&listener_id)?;
    let handler = entry
      .handlers
      .iter()
      .find(|(name, _)| name == event_name)
      .map(|(_, f)| f.clone());
    Some((entry.socket.clone(), handler))
  })?;
  let Some(handler) = handler else {
    // No handler for this event kind: the event still counts as activity,
    // it simply has nowhere to go.
    return Some(owner);
  };
  let assoc = handler_association(event, &listen_socket);
  // A handler that fails must not take the whole evaluation down with it:
  // it runs at an arbitrary point in some other expression, and
  // wolframscript keeps going too.
  match crate::evaluator::function_application::apply_function_to_arg(
    &handler, &assoc,
  ) {
    Ok(_) => {}
    Err(err) => crate::emit_message(&format!(
      "SocketListen: the handler for {event_name} failed: {err:?}"
    )),
  }
  Some(owner)
}

/// The association a handler is called with. The keys, and their order,
/// are the ones wolframscript passes.
#[cfg(not(target_arch = "wasm32"))]
fn handler_association(event: &SocketEvent, listen_socket: &str) -> Expr {
  // wolframscript stamps an event with the whole second it arrived in, as
  // a date: FromUnixTime[seconds].
  let seconds = std::time::SystemTime::now()
    .duration_since(std::time::UNIX_EPOCH)
    .map_or(0, |d| d.as_secs());
  let timestamp = crate::evaluator::evaluate_expr_to_expr(&call1(
    "FromUnixTime",
    Expr::Integer(i128::from(seconds)),
  ))
  .unwrap_or(Expr::Integer(i128::from(seconds)));
  let mut pairs = vec![
    (str_expr("TimeStamp"), timestamp),
    (str_expr("SourceSocket"), socket_expr(&event.socket)),
    (str_expr("Socket"), socket_expr(listen_socket)),
  ];
  match &event.kind {
    EventKind::Data(bytes, complete) => {
      use base64::Engine;
      let byte_array = call1(
        "ByteArray",
        Expr::String(base64::engine::general_purpose::STANDARD.encode(bytes)),
      );
      // `Data` and `DataBytes` are delayed: a handler that only looks at
      // `DataByteArray` should not pay to decode the chunk as text, nor to
      // build one `Expr` per byte of it.
      pairs.push(delayed(
        "Data",
        call(
          "ByteArrayToString",
          vec![byte_array.clone(), str_expr(LISTENER_ENCODING)],
        ),
      ));
      pairs.push(delayed("DataBytes", call1("Normal", byte_array.clone())));
      pairs.push((str_expr("DataByteArray"), byte_array));
      pairs.push((
        str_expr("MultipartComplete"),
        crate::helpers::bool_expr(*complete),
      ));
    }
    EventKind::Error(message) => {
      pairs.push((str_expr("Message"), Expr::String(message.clone())));
    }
    EventKind::Accepted | EventKind::Closed => {}
  }
  Expr::Association(pairs)
}

/// An association entry written `key :> value`, using the `RuleDelayed`
/// marker `Expr::Association` uses for one.
#[cfg(not(target_arch = "wasm32"))]
fn delayed(key: &str, value: Expr) -> (Expr, Expr) {
  let key = str_expr(key);
  (key.clone(), rule_delayed_expr(key, value))
}

// ---------------------------------------------------------------------------
// Blocking reads and writes
// ---------------------------------------------------------------------------

/// The ZeroMQ socket behind an open `"ZMQ"` socket object.
#[cfg(not(target_arch = "wasm32"))]
fn open_zmq(uuid: &str) -> Option<Arc<super::zmtp::ZmqSocket>> {
  with_registry(|reg| {
    let entry = reg.entries.get(uuid)?;
    if entry.closed {
      return None;
    }
    entry.zmq.as_ref().map(|z| z.socket.clone())
  })
}

/// Move the messages that have arrived on a ZeroMQ socket into its stream
/// buffer, back to back. With `block`, wait for at least one, letting the
/// socket handlers run meanwhile.
#[cfg(not(target_arch = "wasm32"))]
fn zmq_fill_buffer(
  uuid: &str,
  zmq: &super::zmtp::ZmqSocket,
  block: bool,
) -> usize {
  loop {
    let mut moved = 0;
    while let Some(message) = zmq.try_receive(true) {
      moved += message.len();
      with_registry(|reg| {
        if let Some(entry) = reg.entries.get_mut(uuid) {
          entry.read_buffer.extend_from_slice(&message);
        }
      });
    }
    if moved > 0 || !block || is_closed(uuid) {
      return moved;
    }
    zmq.wait_input(WAIT_SLICE);
    drain_events();
  }
}

/// `SocketReadMessage` on a ZeroMQ socket: one whole message (all its parts
/// joined) or, with `"Multipart" -> False`, one part of it. Without
/// blocking, an empty byte array says nothing was there.
#[cfg(not(target_arch = "wasm32"))]
fn zmq_read_message(
  zmq: &super::zmtp::ZmqSocket,
  blocking: bool,
  multipart: bool,
) -> Expr {
  if !zmq.kind.can_receive() {
    return sockets_failure(
      "SocketObject",
      "zmqwriteonly",
      vec![str_expr(zmq.kind.name())],
    );
  }
  loop {
    if let Some(message) = zmq.try_receive(multipart) {
      return byte_array(&message);
    }
    if !blocking {
      return byte_array(&[]);
    }
    zmq.wait_input(WAIT_SLICE);
    drain_events();
  }
}

/// The failure wolframscript gives for a stream function a ZeroMQ socket
/// type cannot serve: reading from PUB or PUSH, writing to SUB or PULL, and
/// `Write`/`WriteLine` on REQ and REP, whose two separate messages would
/// break the strict request-reply alternation.
#[cfg(not(target_arch = "wasm32"))]
pub fn socket_function_unsupported(uuid: &str, function: &str) -> Option<Expr> {
  let kind = open_zmq(uuid)?.kind;
  let writes = ["Write", "WriteString", "BinaryWrite", "WriteLine"];
  let reads = [
    "Read",
    "ReadString",
    "ReadByteArray",
    "ReadList",
    "BinaryRead",
    "BinaryReadList",
    "ReadLine",
  ];
  use super::zmtp::ZmqType;
  let refused = match kind {
    ZmqType::Sub | ZmqType::Pull => writes.contains(&function),
    ZmqType::Req | ZmqType::Rep => matches!(function, "Write" | "WriteLine"),
    ZmqType::Pub | ZmqType::Push => reads.contains(&function),
    _ => false,
  };
  refused.then(|| {
    sockets_failure(
      "SocketObject",
      "fctnotsupported",
      vec![str_expr(kind.name()), id_expr(function)],
    )
  })
}

/// Why a socket operation could not go ahead. Both cases print the same
/// free-text line wolframscript prints; they are kept apart only so the
/// caller knows whether anything was said.
#[cfg(not(target_arch = "wasm32"))]
struct SocketUnusable;

/// Whether `Close` (or the close of the listener that accepted it) has
/// already taken this socket down. A socket that was never registered counts
/// as open, so an unknown object still falls through to its own handling.
#[cfg(not(target_arch = "wasm32"))]
fn is_closed(uuid: &str) -> bool {
  with_registry(|reg| reg.entries.get(uuid).is_some_and(|entry| entry.closed))
}

/// The connection behind a socket, or the "invalid or not open" complaint.
#[cfg(not(target_arch = "wasm32"))]
fn usable_stream(uuid: &str) -> Result<Arc<TcpStream>, SocketUnusable> {
  let stream = with_registry(|reg| {
    let entry = reg.entries.get(uuid)?;
    if entry.closed || entry.connect_error.is_some() {
      return None;
    }
    entry.stream.clone()
  });
  stream.ok_or_else(|| {
    invalid_socket_message(uuid);
    SocketUnusable
  })
}

/// Pull one chunk off the wire into the socket's buffer. `Ok(0)` means the
/// peer has closed (or, when `block` is false, that nothing was waiting).
///
/// A blocking read keeps pumping the event queue while it waits. Without
/// that, a script that reads from a connection whose *answer* comes from a
/// `SocketListen` handler would deadlock: the handler runs on this thread,
/// and this thread would be sitting in `read`.
#[cfg(not(target_arch = "wasm32"))]
fn fill_buffer(uuid: &str, block: bool) -> Result<usize, SocketUnusable> {
  if let Some(zmq) = open_zmq(uuid) {
    return Ok(zmq_fill_buffer(uuid, &zmq, block));
  }
  let stream = usable_stream(uuid)?;
  if block {
    let _ = stream.set_read_timeout(Some(WAIT_SLICE));
  } else {
    let _ = stream.set_nonblocking(true);
  }
  let restore = || {
    if !block {
      let _ = stream.set_nonblocking(false);
    }
  };
  let mut chunk = vec![0u8; READ_CHUNK];
  loop {
    match (&*stream).read(&mut chunk) {
      Ok(0) => {
        restore();
        with_registry(|reg| {
          if let Some(entry) = reg.entries.get_mut(uuid) {
            entry.eof = true;
          }
        });
        return Ok(0);
      }
      Ok(n) => {
        restore();
        with_registry(|reg| {
          if let Some(entry) = reg.entries.get_mut(uuid) {
            entry.read_buffer.extend_from_slice(&chunk[..n]);
          }
        });
        return Ok(n);
      }
      // A signal interrupted the read: nothing was consumed, go round again.
      Err(err) if err.kind() == std::io::ErrorKind::Interrupted => {}
      Err(err)
        if matches!(
          err.kind(),
          std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
        ) =>
      {
        if block {
          // The wait is where the listeners get their turn.
          drain_events();
          continue;
        }
        restore();
        return Ok(0);
      }
      Err(_) => {
        restore();
        invalid_socket_message(uuid);
        return Err(SocketUnusable);
      }
    }
  }
}

/// Take up to `limit` buffered bytes (all of them when `limit` is `None`).
#[cfg(not(target_arch = "wasm32"))]
fn take_buffered(uuid: &str, limit: Option<usize>) -> Vec<u8> {
  with_registry(|reg| {
    let Some(entry) = reg.entries.get_mut(uuid) else {
      return Vec::new();
    };
    let take = limit
      .unwrap_or(entry.read_buffer.len())
      .min(entry.read_buffer.len());
    entry.read_buffer.drain(..take).collect()
  })
}

#[cfg(not(target_arch = "wasm32"))]
fn buffered_len(uuid: &str) -> usize {
  with_registry(|reg| reg.entries.get(uuid).map_or(0, |e| e.read_buffer.len()))
}

#[cfg(not(target_arch = "wasm32"))]
fn at_eof(uuid: &str) -> bool {
  with_registry(|reg| reg.entries.get(uuid).is_some_and(|e| e.eof))
}

/// Read until the peer closes, the way `ReadString` on a stream reads to
/// the end of the file.
#[cfg(not(target_arch = "wasm32"))]
fn read_to_close(uuid: &str) -> Result<Vec<u8>, SocketUnusable> {
  // A ZeroMQ socket never closes from the far end; its stream "ends"
  // wherever the messages that have arrived so far end.
  if open_zmq(uuid).is_some() {
    fill_buffer(uuid, false)?;
    return Ok(take_buffered(uuid, None));
  }
  while !at_eof(uuid) {
    if fill_buffer(uuid, true)? == 0 {
      break;
    }
  }
  Ok(take_buffered(uuid, None))
}

/// Read whatever is there, waiting for the first byte. Empty only once the
/// peer has closed.
#[cfg(not(target_arch = "wasm32"))]
fn read_available(uuid: &str) -> Result<Vec<u8>, SocketUnusable> {
  if buffered_len(uuid) == 0 && !at_eof(uuid) {
    fill_buffer(uuid, true)?;
  }
  Ok(take_buffered(uuid, None))
}

/// Send bytes down a socket.
#[cfg(not(target_arch = "wasm32"))]
pub fn socket_write_bytes(uuid: &str, bytes: &[u8]) -> bool {
  socket_write_parts(uuid, &[bytes])
}

/// Send what one write call produced, in the pieces it produced them. A TCP
/// socket gets the bytes back to back; a ZeroMQ socket gets one message per
/// non-empty piece, the way wolframscript's socket streams flush — `Write`
/// and `WriteLine` send the text and the newline as two messages, and an
/// empty write sends nothing.
#[cfg(not(target_arch = "wasm32"))]
pub fn socket_write_parts(uuid: &str, parts: &[&[u8]]) -> bool {
  if let Some(zmq) = open_zmq(uuid) {
    for part in parts.iter().filter(|p| !p.is_empty()) {
      zmq.send(vec![part.to_vec()]);
    }
    return true;
  }
  let bytes = parts.concat();
  let Ok(stream) = usable_stream(uuid) else {
    return false;
  };
  if let Ok(()) = (&*stream)
    .write_all(&bytes)
    .and_then(|()| (&*stream).flush())
  {
    true
  } else {
    invalid_socket_message(uuid);
    false
  }
}

/// Whether a read would find something without waiting: buffered bytes, a
/// peer that has closed (which a read reports at once), or data on the wire.
///
/// Asking is not itself an operation on the socket, so an unusable one is
/// simply not ready — no complaint. A listening socket is never ready:
/// what happens to it arrives as an event, not as bytes.
#[cfg(not(target_arch = "wasm32"))]
fn socket_ready(uuid: &str) -> bool {
  if buffered_len(uuid) > 0 || at_eof(uuid) {
    return true;
  }
  if let Some(zmq) = open_zmq(uuid) {
    return zmq.has_input();
  }
  let stream = with_registry(|reg| {
    let entry = reg.entries.get(uuid)?;
    if entry.closed
      || entry.connect_error.is_some()
      || entry.role == SocketRole::Server
    {
      return None;
    }
    entry.stream.clone()
  });
  let Some(stream) = stream else {
    return false;
  };
  let _ = stream.set_nonblocking(true);
  let mut probe = [0u8; 1];
  let ready = match stream.peek(&mut probe) {
    Ok(_) => true,
    Err(err) => !matches!(
      err.kind(),
      std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
    ),
  };
  let _ = stream.set_nonblocking(false);
  ready
}

// ---------------------------------------------------------------------------
// Waiting
// ---------------------------------------------------------------------------

/// The sockets a wait was asked about, as UUIDs.
#[cfg(not(target_arch = "wasm32"))]
fn wait_targets(expr: &Expr) -> Option<Vec<String>> {
  match expr {
    Expr::List(items) => items.iter().map(socket_object_uuid).collect(),
    other => socket_object_uuid(other).map(|uuid| vec![uuid]),
  }
}

/// Shared body of `SocketWaitNext` and `SocketWaitAll`.
///
/// "Activity" on a socket is an event its listener handled — including one
/// on a connection it accepted — or, for a socket read directly, data
/// waiting on the wire. Running out of time leaves the call unevaluated,
/// which is what wolframscript does when nothing arrives.
#[cfg(not(target_arch = "wasm32"))]
fn socket_wait(name: &str, args: &[Expr], all: bool) -> Expr {
  let Some(targets) = wait_targets(&args[0]) else {
    return unevaluated(name, args);
  };
  if targets.is_empty() {
    return unevaluated(name, args);
  }
  let timeout = match args.get(1) {
    None => None,
    Some(expr) => match crate::functions::math_ast::try_eval_to_f64(expr) {
      Some(seconds) if seconds >= 0.0 => Some(Duration::from_secs_f64(seconds)),
      _ => return unevaluated(name, args),
    },
  };
  let deadline = timeout.map(|t| Instant::now() + t);
  let mut seen: Vec<String> = Vec::new();
  loop {
    for source in drain_events() {
      if targets.contains(&source) && !seen.contains(&source) {
        seen.push(source);
      }
    }
    for target in &targets {
      if !seen.contains(target) && socket_ready(target) {
        seen.push(target.clone());
      }
    }
    if all && seen.len() == targets.len() {
      return Expr::List(
        targets
          .iter()
          .map(|u| socket_expr(u))
          .collect::<Vec<_>>()
          .into(),
      );
    }
    if !all && let Some(first) = seen.first() {
      return socket_expr(first);
    }
    if deadline.is_some_and(|d| Instant::now() >= d) {
      return unevaluated(name, args);
    }
    std::thread::sleep(Duration::from_millis(1));
  }
}

// ---------------------------------------------------------------------------
// Closing
// ---------------------------------------------------------------------------

/// `Close[sock]` — shut the connection down and stop any listener on it.
/// The entry stays in the registry marked closed, so a later operation
/// gives the "invalid or not open" message instead of looking like a
/// socket that never existed.
#[cfg(not(target_arch = "wasm32"))]
fn socket_close(uuid: &str) -> Expr {
  let known = with_registry(|reg| reg.entries.contains_key(uuid));
  if !known {
    invalid_socket_message(uuid);
    return fail_expr();
  }
  crate::evaluator::dispatch::io_functions::close_socket_streams(uuid);
  // Closing the socket a listener was set up on ends the listener; closing
  // one connection it accepted only ends that connection, which is what an
  // echo handler does when it is finished with a client.
  let own_listener = with_registry(|reg| {
    reg
      .entries
      .get(uuid)
      .filter(|entry| entry.parent.is_none())
      .and_then(|entry| entry.listener_id)
  });
  if let Some(id) = own_listener {
    stop_listener(id);
  }
  let children = with_registry(|reg| {
    let mut children = Vec::new();
    if let Some(entry) = reg.entries.get_mut(uuid) {
      children.clone_from(&entry.children);
      shutdown_entry(entry);
    }
    children
  });
  // Closing the server closes the connections it handed out, the way
  // dropping a listener does in wolframscript.
  for child in children {
    with_registry(|reg| {
      if let Some(entry) = reg.entries.get_mut(&child) {
        shutdown_entry(entry);
      }
    });
  }
  // wolframscript answers with the endpoint that was closed, spelled
  // `"host:port"`, not with the socket object.
  with_registry(|reg| {
    reg.entries.get(uuid).map_or_else(fail_expr, |entry| {
      Expr::String(format!("{}:{}", entry.dest_host, entry.dest_port))
    })
  })
}

/// Stop a listener's threads and forget it.
#[cfg(not(target_arch = "wasm32"))]
fn stop_listener(id: i128) -> bool {
  let entry = LISTENERS.with(|l| l.borrow_mut().remove(&id));
  let Some(entry) = entry else {
    return false;
  };
  entry.stop.store(true, Ordering::Relaxed);
  with_registry(|reg| {
    for socket in reg.entries.values_mut() {
      if socket.listener_id == Some(id) {
        socket.listener_id = None;
        if let Some(zmq) = &socket.zmq {
          zmq.socket.clear_listener();
        }
      }
    }
  });
  true
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

/// The property names `sock["Properties"]` reports, in wolframscript's
/// order (alphabetical).
#[cfg(not(target_arch = "wasm32"))]
const SOCKET_PROPERTIES: [&str; 11] = [
  "ConnectedClients",
  "DestinationHostname",
  "DestinationIPAddress",
  "DestinationPort",
  "DirectionType",
  "InprocQ",
  "Protocol",
  "Scheme",
  "SocketListener",
  "Type",
  "UUID",
];

#[cfg(not(target_arch = "wasm32"))]
const LISTENER_PROPERTIES: [&str; 5] = [
  "CharacterEncoding",
  "HandlerFunctions",
  "HandlerFunctionsKeys",
  "RecordSeparators",
  "Socket",
];

#[cfg(not(target_arch = "wasm32"))]
fn string_list(items: &[&str]) -> Expr {
  Expr::List(items.iter().map(|s| str_expr(s)).collect())
}

/// `SocketObject[…]["property"]`. A TCP socket answers an unknown property
/// with the empty list; a ZeroMQ socket also knows every gettable libzmq
/// option (`"Linger"`, `"ZMQ_LINGER"`, `"Rcvhwm"`, …) and rejects anything
/// else with an `invalidGetOption` failure.
///
/// A closed socket answers nothing at all: wolframscript rejects *every*
/// query on it — `"DestinationPort"` as much as `"UUID"` or `"Properties"` —
/// with the same invalid-socket failure a read gets, so the object cannot be
/// used to reconstruct what it was connected to. Woxi reports that as a bare
/// `$Failed`, without the free-text line a read prints: wolframscript prints
/// nothing either, because there the failure *is* the returned
/// `Failure["SocketsLink", …]` object (see the conformance gaps doc).
#[cfg(not(target_arch = "wasm32"))]
pub fn socket_property(uuid: &str, property: &str) -> Option<Expr> {
  if is_closed(uuid) {
    return Some(fail_expr());
  }
  if !with_registry(|reg| reg.entries.contains_key(uuid)) {
    return None;
  }
  if property == "Properties" {
    return Some(socket_properties_list(uuid));
  }
  Some(socket_options(uuid, &[property.to_string()]))
}

/// `SocketObject[…][{"p1", "p2", …}]`, also what `Options[sock, …]` asks.
#[cfg(not(target_arch = "wasm32"))]
pub fn socket_property_list(uuid: &str, names: &[Expr]) -> Option<Expr> {
  if is_closed(uuid) {
    return Some(fail_expr());
  }
  if !with_registry(|reg| reg.entries.contains_key(uuid)) {
    return None;
  }
  let names = names
    .iter()
    .map(|n| match n {
      Expr::String(s) => Some(s.clone()),
      _ => None,
    })
    .collect::<Option<Vec<_>>>()?;
  if names.is_empty() {
    return Some(Expr::List(Vec::new().into()));
  }
  Some(socket_options(uuid, &names))
}

/// `Options[sock]`: every property, as sorted rules.
#[cfg(not(target_arch = "wasm32"))]
pub fn socket_all_options(uuid: &str) -> Option<Expr> {
  if is_closed(uuid) {
    return Some(fail_expr());
  }
  if !with_registry(|reg| reg.entries.contains_key(uuid)) {
    return None;
  }
  let names = match &socket_properties_list(uuid) {
    Expr::List(items) => items
      .iter()
      .filter_map(|n| match n {
        Expr::String(s) => Some(s.clone()),
        _ => None,
      })
      .collect::<Vec<_>>(),
    _ => Vec::new(),
  };
  Some(socket_options(uuid, &names))
}

/// The ZeroMQ options `sock["Properties"]` lists on top of the TCP ones:
/// every gettable option except those libzmq only answers for particular
/// builds or socket types.
#[cfg(not(target_arch = "wasm32"))]
const ZMQ_PROPERTY_OPTIONS: [&str; 40] = [
  "ZMQ_AFFINITY",
  "ZMQ_BACKLOG",
  "ZMQ_CONNECT_TIMEOUT",
  "ZMQ_EVENTS",
  "ZMQ_FD",
  "ZMQ_HANDSHAKE_IVL",
  "ZMQ_IMMEDIATE",
  "ZMQ_INVERT_MATCHING",
  "ZMQ_IPV4ONLY",
  "ZMQ_IPV6",
  "ZMQ_LAST_ENDPOINT",
  "ZMQ_LINGER",
  "ZMQ_MAXMSGSIZE",
  "ZMQ_MECHANISM",
  "ZMQ_MULTICAST_HOPS",
  "ZMQ_MULTICAST_MAXTPDU",
  "ZMQ_PLAIN_PASSWORD",
  "ZMQ_PLAIN_SERVER",
  "ZMQ_PLAIN_USERNAME",
  "ZMQ_RATE",
  "ZMQ_RCVBUF",
  "ZMQ_RCVHWM",
  "ZMQ_RCVMORE",
  "ZMQ_RCVTIMEO",
  "ZMQ_RECONNECT_IVL",
  "ZMQ_RECONNECT_IVL_MAX",
  "ZMQ_RECOVERY_IVL",
  "ZMQ_SNDBUF",
  "ZMQ_SNDHWM",
  "ZMQ_SNDTIMEO",
  "ZMQ_SOCKS_PROXY",
  "ZMQ_TCP_KEEPALIVE",
  "ZMQ_TCP_KEEPALIVE_CNT",
  "ZMQ_TCP_KEEPALIVE_IDLE",
  "ZMQ_TCP_KEEPALIVE_INTVL",
  "ZMQ_TCP_MAXRT",
  "ZMQ_TOS",
  "ZMQ_TYPE",
  "ZMQ_USE_FD",
  "ZMQ_ZAP_DOMAIN",
];

/// `sock["Properties"]`: the TCP property names, and for a ZeroMQ socket
/// its option names after them (the two blocks are each sorted, and the
/// `ZMQ_` names sort after the others).
#[cfg(not(target_arch = "wasm32"))]
fn socket_properties_list(uuid: &str) -> Expr {
  let zmq =
    with_registry(|reg| reg.entries.get(uuid).is_some_and(|e| e.zmq.is_some()));
  let mut names: Vec<&str> = SOCKET_PROPERTIES.to_vec();
  if zmq {
    names.extend(ZMQ_PROPERTY_OPTIONS);
  }
  string_list(&names)
}

/// libzmq's spelling of an option name: `"ReconnectIvlMax"` becomes
/// `"ZMQ_RECONNECT_IVL_MAX"`; a name that already is one stays as it is.
#[cfg(not(target_arch = "wasm32"))]
fn zmq_option_name(name: &str) -> String {
  if name.starts_with("ZMQ_")
    && (zmq_get_default(name).is_some()
      || ZMQ_SET_OPTIONS.iter().any(|(n, _)| *n == name))
  {
    return name.to_string();
  }
  let mut out = String::from("ZMQ_");
  let mut previous_lower = false;
  for c in name.chars() {
    if previous_lower && c.is_ascii_uppercase() {
      out.push('_');
    }
    previous_lower = c.is_ascii_lowercase();
    out.extend(c.to_uppercase());
  }
  out
}

/// The CamelCase name an option is reported under: `"ZMQ_RECONNECT_IVL"`
/// becomes `"ReconnectIvl"`.
#[cfg(not(target_arch = "wasm32"))]
fn camel_option_name(name: &str) -> String {
  name
    .to_ascii_lowercase()
    .split('_')
    .skip(1)
    .map(|part| {
      let mut chars = part.chars();
      chars.next().map_or_else(String::new, |first| {
        first.to_ascii_uppercase().to_string() + chars.as_str()
      })
    })
    .collect()
}

/// What a gettable libzmq option reads as before anything set it — the
/// values wolframscript's bundled libzmq reports. `None` for a name that is
/// not a gettable option; options whose value depends on the socket's state
/// (`ZMQ_EVENTS`, `ZMQ_FD`, …) are filled in by [`zmq_option_value`].
#[cfg(not(target_arch = "wasm32"))]
fn zmq_get_default(name: &str) -> Option<Expr> {
  let int = |n: i128| Some(Expr::Integer(n));
  match name {
    "ZMQ_AFFINITY"
    | "ZMQ_CONNECT_TIMEOUT"
    | "ZMQ_IMMEDIATE"
    | "ZMQ_INVERT_MATCHING"
    | "ZMQ_IPV4ONLY"
    | "ZMQ_IPV6"
    | "ZMQ_MECHANISM"
    | "ZMQ_PLAIN_SERVER"
    | "ZMQ_RECONNECT_IVL_MAX"
    | "ZMQ_TCP_MAXRT"
    | "ZMQ_THREAD_SAFE"
    | "ZMQ_TOS" => int(0),
    "ZMQ_BACKLOG" | "ZMQ_RATE" | "ZMQ_RECONNECT_IVL" => int(100),
    "ZMQ_HANDSHAKE_IVL" => int(30000),
    "ZMQ_MAXMSGSIZE"
    | "ZMQ_RCVBUF"
    | "ZMQ_RCVTIMEO"
    | "ZMQ_SNDBUF"
    | "ZMQ_SNDTIMEO"
    | "ZMQ_TCP_KEEPALIVE"
    | "ZMQ_TCP_KEEPALIVE_CNT"
    | "ZMQ_TCP_KEEPALIVE_IDLE"
    | "ZMQ_TCP_KEEPALIVE_INTVL"
    | "ZMQ_USE_FD"
    | "ZMQ_LINGER" => int(-1),
    "ZMQ_MULTICAST_HOPS" => int(1),
    "ZMQ_MULTICAST_MAXTPDU" => int(1500),
    "ZMQ_RCVHWM" | "ZMQ_SNDHWM" => int(1000),
    "ZMQ_RECOVERY_IVL" => int(10000),
    "ZMQ_IDENTITY" | "ZMQ_PLAIN_PASSWORD" | "ZMQ_PLAIN_USERNAME"
    | "ZMQ_SOCKS_PROXY" | "ZMQ_ZAP_DOMAIN" => Some(str_expr("")),
    // CURVE keys read back Z85-encoded, all zeros while unset.
    "ZMQ_CURVE_PUBLICKEY" | "ZMQ_CURVE_SECRETKEY" | "ZMQ_CURVE_SERVERKEY" => {
      Some(Expr::String("0".repeat(40)))
    }
    "ZMQ_EVENTS" | "ZMQ_FD" | "ZMQ_LAST_ENDPOINT" | "ZMQ_RCVMORE"
    | "ZMQ_TYPE" => Some(Expr::Integer(0)),
    // Gettable in principle, but the bundled libzmq is built without
    // GSSAPI and VMCI, so asking fails.
    "ZMQ_GSSAPI_PLAINTEXT"
    | "ZMQ_GSSAPI_PRINCIPAL"
    | "ZMQ_GSSAPI_SERVER"
    | "ZMQ_GSSAPI_SERVICE_PRINCIPAL"
    | "ZMQ_VMCI_BUFFER_SIZE"
    | "ZMQ_VMCI_BUFFER_MIN_SIZE"
    | "ZMQ_VMCI_BUFFER_MAX_SIZE"
    | "ZMQ_VMCI_CONNECT_TIMEOUT" => Some(fail_expr()),
    _ => None,
  }
}

/// The current value of a gettable option, `Err` with the failure for one
/// that cannot be read, `None` for a name that is not a gettable option.
#[cfg(not(target_arch = "wasm32"))]
fn zmq_option_value(
  entry: &SocketEntry,
  name: &str,
) -> Option<Result<Expr, Expr>> {
  let zmq = entry.zmq.as_ref()?;
  let default = zmq_get_default(name)?;
  if matches!(default, Expr::Identifier(ref f) if f == "$Failed") {
    return Some(Err(sockets_failure(
      "SocketObject",
      "zmqinvcmd",
      vec![str_expr("Invalid argument")],
    )));
  }
  if let Some((_, value)) = zmq.options.iter().find(|(n, _)| n == name) {
    return Some(Ok(value.clone()));
  }
  use super::zmtp::ZmqType;
  let socket = &zmq.socket;
  Some(Ok(match name {
    "ZMQ_EVENTS" => Expr::Integer(
      i128::from(socket.has_input()) | (i128::from(socket.writable()) << 1),
    ),
    "ZMQ_FD" => {
      Expr::Integer(zmq.fd.or_else(|| socket.peer_fd()).unwrap_or(-1))
    }
    "ZMQ_LAST_ENDPOINT" => Expr::String(zmq.last_endpoint.clone()),
    "ZMQ_RCVMORE" => crate::helpers::bool_expr(socket.receive_more()),
    "ZMQ_TYPE" => Expr::Integer(socket.kind.code()),
    // wolframscript turns lingering off for subscribers.
    "ZMQ_LINGER" if matches!(socket.kind, ZmqType::Sub | ZmqType::XSub) => {
      Expr::Integer(0)
    }
    _ => default,
  }))
}

/// A property every socket has, whatever its protocol.
#[cfg(not(target_arch = "wasm32"))]
fn common_property(
  reg: &Registry,
  entry: &SocketEntry,
  uuid: &str,
  property: &str,
) -> Option<Expr> {
  let zmq = entry.zmq.as_ref();
  let clients = |server: &SocketEntry| {
    Expr::List(
      server
        .children
        .iter()
        .map(|c| socket_expr(c))
        .collect::<Vec<_>>()
        .into(),
    )
  };
  Some(match property {
    // A TCP server lists the connections it accepted, and so does each of
    // those connections (they share the server's registration); everything
    // else has none.
    "ConnectedClients" => match (zmq, entry.role) {
      (None, SocketRole::Server) => clients(entry),
      (None, SocketRole::Accepted) => entry
        .parent
        .as_ref()
        .and_then(|p| reg.entries.get(p))
        .map_or_else(|| id_expr("None"), clients),
      _ => id_expr("None"),
    },
    "DestinationHostname" => Expr::String(entry.dest_host.clone()),
    "DestinationIPAddress" => match zmq {
      Some(z) => z.ip.clone(),
      None => call1("IPAddress", Expr::String(entry.dest_ip.clone())),
    },
    "DestinationPort" => Expr::Integer(entry.dest_port),
    "DirectionType" => Expr::String(
      match entry.role {
        SocketRole::Server => "Server",
        SocketRole::Client | SocketRole::Accepted => "Client",
      }
      .to_string(),
    ),
    // True only for ZeroMQ's in-process transport, which has no TCP
    // socket behind it at all.
    "InprocQ" => crate::helpers::bool_expr(false),
    "Protocol" => str_expr(if zmq.is_some() { "ZMQ" } else { "TCP" }),
    "Scheme" => {
      Expr::String(zmq.map_or_else(|| "tcp".to_string(), |z| z.scheme.clone()))
    }
    // A socket nobody is listening on reports `None`, not the empty list
    // an unknown property gives.
    "SocketListener" => entry.listener_id.map_or_else(
      || id_expr("None"),
      |id| call1("SocketListener", Expr::Integer(id)),
    ),
    // A plain TCP socket has no ZeroMQ type.
    "Type" => {
      zmq.map_or_else(|| id_expr("None"), |z| str_expr(z.socket.kind.name()))
    }
    "UUID" => str_expr(uuid),
    _ => return None,
  })
}

/// wolframscript's `getSocketOptions`: look each name up as a common
/// property or (on a ZeroMQ socket) a libzmq option, and give a single value
/// for a single name, sorted rules otherwise. A TCP socket silently skips
/// names it does not know.
#[cfg(not(target_arch = "wasm32"))]
fn socket_options(uuid: &str, names: &[String]) -> Expr {
  let found = with_registry(|reg| -> Result<Vec<Expr>, Expr> {
    let Some(entry) = reg.entries.get(uuid) else {
      return Ok(Vec::new());
    };
    let mut rules = Vec::new();
    for name in names {
      if let Some(value) = common_property(reg, entry, uuid, name) {
        rules.push(rule_expr(str_expr(name), value));
      } else if entry.zmq.is_some() {
        let option = zmq_option_name(name);
        match zmq_option_value(entry, &option) {
          Some(Ok(value)) => {
            rules
              .push(rule_expr(Expr::String(camel_option_name(&option)), value));
          }
          Some(Err(failure)) => return Err(failure),
          None => {
            return Err(sockets_failure(
              "SocketObject",
              "invalidGetOption",
              vec![str_expr(name)],
            ));
          }
        }
      }
    }
    Ok(rules)
  });
  let mut rules = match found {
    Ok(rules) => rules,
    Err(failure) => return failure,
  };
  if rules.len() == 1 {
    if let Expr::Rule { replacement, .. } = &rules[0] {
      return (**replacement).clone();
    }
    return Expr::List(Vec::new().into());
  }
  rules.sort_by(crate::functions::list_helpers_ast::sorting::canonical_cmp);
  Expr::List(rules.into())
}

/// How `SetOptions` checks an option value before handing it to libzmq.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy)]
enum OptionKind {
  /// A machine integer.
  Int,
  /// An integer from 0 to 2^64 - 1.
  Unsigned,
  Text,
  /// A list of byte values.
  Bytes,
  /// `All`, a string, a byte array or a list of byte values.
  Topic,
}

/// Every option `SetOptions` accepts on a socket, by libzmq name.
#[cfg(not(target_arch = "wasm32"))]
const ZMQ_SET_OPTIONS: &[(&str, OptionKind)] = {
  use OptionKind::{Bytes, Int, Text, Topic, Unsigned};
  &[
    ("ZMQ_AFFINITY", Unsigned),
    ("ZMQ_BACKLOG", Int),
    ("ZMQ_CONNECT_RID", Bytes),
    ("ZMQ_CONFLATE", Int),
    ("ZMQ_CONNECT_TIMEOUT", Int),
    ("ZMQ_CURVE_PUBLICKEY", Bytes),
    ("ZMQ_CURVE_SECRETKEY", Bytes),
    ("ZMQ_CURVE_SERVER", Int),
    ("ZMQ_CURVE_SERVERKEY", Bytes),
    ("ZMQ_GSSAPI_PLAINTEXT", Int),
    ("ZMQ_GSSAPI_PRINCIPAL", Text),
    ("ZMQ_GSSAPI_SERVER", Int),
    ("ZMQ_GSSAPI_SERVICE_PRINCIPAL", Text),
    ("ZMQ_HANDSHAKE_IVL", Int),
    ("ZMQ_HEARTBEAT_IVL", Int),
    ("ZMQ_HEARTBEAT_TIMEOUT", Int),
    ("ZMQ_HEARTBEAT_TTL", Int),
    ("ZMQ_IDENTITY", Bytes),
    ("ZMQ_IMMEDIATE", Int),
    ("ZMQ_INVERT_MATCHING", Int),
    ("ZMQ_IPV6", Int),
    ("ZMQ_LINGER", Int),
    ("ZMQ_MAXMSGSIZE", Int),
    ("ZMQ_MULTICAST_HOPS", Int),
    ("ZMQ_MULTICAST_MAXTPDU", Int),
    ("ZMQ_PLAIN_PASSWORD", Text),
    ("ZMQ_PLAIN_SERVER", Int),
    ("ZMQ_PLAIN_USERNAME", Text),
    ("ZMQ_USE_FD", Int),
    ("ZMQ_PROBE_ROUTER", Int),
    ("ZMQ_RATE", Int),
    ("ZMQ_RCVBUF", Int),
    ("ZMQ_RCVHWM", Int),
    ("ZMQ_RCVTIMEO", Int),
    ("ZMQ_RECONNECT_IVL", Int),
    ("ZMQ_RECONNECT_IVL_MAX", Int),
    ("ZMQ_RECOVERY_IVL", Int),
    ("ZMQ_REQ_CORRELATE", Int),
    ("ZMQ_REQ_RELAXED", Int),
    ("ZMQ_ROUTER_HANDOVER", Int),
    ("ZMQ_ROUTER_MANDATORY", Int),
    ("ZMQ_ROUTER_RAW", Int),
    ("ZMQ_SNDBUF", Int),
    ("ZMQ_SNDHWM", Int),
    ("ZMQ_SNDTIMEO", Int),
    ("ZMQ_SOCKS_PROXY", Text),
    ("ZMQ_STREAM_NOTIFY", Int),
    ("ZMQ_SUBSCRIBE", Topic),
    ("ZMQ_TCP_KEEPALIVE", Int),
    ("ZMQ_TCP_KEEPALIVE_CNT", Int),
    ("ZMQ_TCP_KEEPALIVE_IDLE", Int),
    ("ZMQ_TCP_KEEPALIVE_INTVL", Int),
    ("ZMQ_TCP_MAXRT", Int),
    ("ZMQ_TOS", Int),
    ("ZMQ_UNSUBSCRIBE", Topic),
    ("ZMQ_XPUB_VERBOSE", Int),
    ("ZMQ_XPUB_VERBOSER", Int),
    ("ZMQ_XPUB_MANUAL", Int),
    ("ZMQ_XPUB_NODROP", Int),
    ("ZMQ_XPUB_WELCOME_MSG", Bytes),
    ("ZMQ_ZAP_DOMAIN", Text),
    ("ZMQ_TCP_ACCEPT_FILTER", Bytes),
    ("ZMQ_IPC_FILTER_GID", Int),
    ("ZMQ_IPC_FILTER_PID", Int),
    ("ZMQ_IPC_FILTER_UID", Int),
    ("ZMQ_IPV4ONLY", Int),
    ("ZMQ_VMCI_BUFFER_SIZE", Unsigned),
    ("ZMQ_VMCI_BUFFER_MIN_SIZE", Unsigned),
    ("ZMQ_VMCI_BUFFER_MAX_SIZE", Unsigned),
    ("ZMQ_VMCI_CONNECT_TIMEOUT", Int),
  ]
};

/// The bytes of a list of byte values, if that is what `expr` is.
#[cfg(not(target_arch = "wasm32"))]
fn byte_list(expr: &Expr) -> Option<Vec<u8>> {
  match expr {
    Expr::List(items) => items
      .iter()
      .map(|item| match item {
        Expr::Integer(b) => u8::try_from(*b).ok(),
        _ => None,
      })
      .collect(),
    _ => None,
  }
}

/// The bytes of a `ByteArray[…]`.
#[cfg(not(target_arch = "wasm32"))]
fn byte_array_bytes(expr: &Expr) -> Option<Vec<u8>> {
  match expr {
    Expr::FunctionCall { name, args }
      if name == "ByteArray" && args.len() == 1 =>
    {
      let Expr::String(b64) = &args[0] else {
        return None;
      };
      use base64::Engine;
      base64::engine::general_purpose::STANDARD.decode(b64).ok()
    }
    _ => None,
  }
}

/// Whether `value` passes the check wolframscript applies to `kind`.
#[cfg(not(target_arch = "wasm32"))]
fn option_value_ok(kind: OptionKind, value: &Expr) -> bool {
  match kind {
    OptionKind::Int => {
      matches!(value, Expr::Integer(n) if i64::try_from(*n).is_ok())
    }
    OptionKind::Unsigned => {
      matches!(value, Expr::Integer(n) if u64::try_from(*n).is_ok())
    }
    OptionKind::Text => matches!(value, Expr::String(_)),
    OptionKind::Bytes => byte_list(value).is_some(),
    OptionKind::Topic => {
      matches!(value, Expr::String(_))
        || matches!(value, Expr::Identifier(a) if a == "All")
        || byte_array_bytes(value).is_some()
        || byte_list(value).is_some()
    }
  }
}

/// The value as wolframscript passes it on, and reports it back: strings
/// and integers as they are, byte arrays and byte lists as the string of
/// those characters. `None` for a value that has no such form (`All` for
/// anything but `"Subscribe"`).
#[cfg(not(target_arch = "wasm32"))]
fn normalized_option_value(option: &str, value: &Expr) -> Option<Expr> {
  match value {
    Expr::Identifier(a) if a == "All" && option == "ZMQ_SUBSCRIBE" => {
      Some(str_expr(""))
    }
    Expr::String(_) | Expr::Integer(_) => Some(value.clone()),
    Expr::List(_) => {
      let codes = byte_list(value)?;
      Some(Expr::String(codes.iter().map(|b| char::from(*b)).collect()))
    }
    _ => {
      let bytes = byte_array_bytes(value)?;
      Some(Expr::String(bytes.iter().map(|b| char::from(*b)).collect()))
    }
  }
}

/// The bytes a subscription topic or identity stands for.
#[cfg(not(target_arch = "wasm32"))]
fn option_bytes(value: &Expr) -> Vec<u8> {
  match value {
    Expr::String(s) => s.as_bytes().to_vec(),
    other => byte_list(other)
      .or_else(|| byte_array_bytes(other))
      .unwrap_or_default(),
  }
}

/// `SetOptions[sock, name -> value, …]`. Every option is checked before
/// any is applied, so one bad option leaves the socket untouched. Returns
/// the options as set, or the failure. `"Subscribe"`, `"Unsubscribe"` and
/// `"Identity"` take effect; every other value is recorded and reads back
/// through the getters.
#[cfg(not(target_arch = "wasm32"))]
pub fn socket_set_options(uuid: &str, rules: &[Expr]) -> Option<Expr> {
  if is_closed(uuid) {
    return Some(fail_expr());
  }
  if !with_registry(|reg| reg.entries.contains_key(uuid)) {
    return None;
  }
  let head = "SocketObject";
  let mut accepted: Vec<(String, Expr, Expr)> = Vec::new();
  for rule in rules {
    let (Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    }) = rule
    else {
      return None;
    };
    let invalid_name =
      || sockets_failure(head, "invalidOptionName", vec![(**pattern).clone()]);
    let Expr::String(name) = &**pattern else {
      return Some(invalid_name());
    };
    let option = zmq_option_name(name);
    let Some((_, kind)) = ZMQ_SET_OPTIONS.iter().find(|(n, _)| *n == option)
    else {
      return Some(invalid_name());
    };
    if !option_value_ok(*kind, replacement) {
      return Some(sockets_failure(
        head,
        "invalidOptionValue",
        vec![(**pattern).clone(), (**replacement).clone()],
      ));
    }
    let Some(normalized) = normalized_option_value(&option, replacement) else {
      return Some(sockets_failure(
        head,
        "invalidOptionValue",
        vec![Expr::String(option), (**replacement).clone()],
      ));
    };
    accepted.push((option, normalized, (**replacement).clone()));
  }
  let zmq = open_zmq(uuid);
  let mut reported = Vec::new();
  for (option, normalized, original) in accepted {
    if let Some(zmq) = &zmq {
      match option.as_str() {
        "ZMQ_SUBSCRIBE" => zmq.subscribe(true, &option_bytes(&original)),
        "ZMQ_UNSUBSCRIBE" => zmq.subscribe(false, &option_bytes(&original)),
        "ZMQ_IDENTITY" => zmq.set_identity(option_bytes(&original)),
        _ => {}
      }
      if zmq_get_default(&option).is_some() {
        let stored = (option.clone(), normalized.clone());
        with_registry(|reg| {
          if let Some(z) =
            reg.entries.get_mut(uuid).and_then(|e| e.zmq.as_mut())
          {
            z.options.retain(|(n, _)| *n != option);
            z.options.push(stored);
          }
        });
      }
    }
    // An empty subscription — `All` or `""` — is reported under libzmq's
    // own name, with `All` as its value.
    let empty_topic = matches!(&normalized, Expr::String(t) if t.is_empty());
    reported.push(if option == "ZMQ_SUBSCRIBE" && empty_topic {
      rule_expr(str_expr("ZMQ_SUBSCRIBE"), id_expr("All"))
    } else {
      rule_expr(Expr::String(camel_option_name(&option)), normalized)
    });
  }
  Some(Expr::List(reported.into()))
}

/// `SocketListener[…]["property"]`.
#[cfg(not(target_arch = "wasm32"))]
pub fn listener_property(id: i128, property: &str) -> Option<Expr> {
  if property == "Properties" {
    return Some(string_list(&LISTENER_PROPERTIES));
  }
  let found = LISTENERS.with(|l| {
    let listeners = l.borrow();
    let entry = listeners.get(&id)?;
    Some(match property {
      "CharacterEncoding" => Ok(str_expr(LISTENER_ENCODING)),
      "HandlerFunctions" => Ok(Expr::Association(
        entry
          .handlers
          .iter()
          .map(|(name, f)| (Expr::String(name.clone()), f.clone()))
          .collect(),
      )),
      "HandlerFunctionsKeys" => Ok(Expr::List(
        entry
          .handler_keys
          .iter()
          .map(|k| Expr::String(k.clone()))
          .collect::<Vec<_>>()
          .into(),
      )),
      // Unset, records are whatever arrives in one piece.
      "RecordSeparators" => Ok(id_expr("None")),
      "Socket" => Ok(socket_expr(&entry.socket)),
      // Anything else is a question about the socket being listened to.
      _ => Err(entry.socket.clone()),
    })
  })?;
  match found {
    Ok(value) => Some(value),
    Err(socket) => socket_property(&socket, property),
  }
}

/// The options `Options[SocketListen]` declares, which is what
/// `Options[listener]` and `listener[{}]` report.
#[cfg(not(target_arch = "wasm32"))]
const LISTENER_OPTIONS: [&str; 4] = [
  "CharacterEncoding",
  "HandlerFunctions",
  "HandlerFunctionsKeys",
  "RecordSeparators",
];

/// `listener[{"p1", "p2", …}]`: a rule per name, keyed as asked; a single
/// name gives just its value. `listener[{}]` and `Options[listener]` give
/// the `SocketListen` options, keyed by their symbols.
#[cfg(not(target_arch = "wasm32"))]
pub fn listener_property_list(id: i128, names: &[Expr]) -> Option<Expr> {
  LISTENERS
    .with(|l| l.borrow().contains_key(&id))
    .then_some(())?;
  let names: Vec<Expr> = if names.is_empty() {
    LISTENER_OPTIONS.iter().map(|n| id_expr(n)).collect()
  } else {
    names.to_vec()
  };
  let value_of = |name: &Expr| match name {
    Expr::String(n) | Expr::Identifier(n) => listener_property(id, n),
    _ => None,
  };
  if let [only] = names.as_slice() {
    return value_of(only);
  }
  let rules = names
    .iter()
    .map(|name| Some(rule_expr(name.clone(), value_of(name)?)))
    .collect::<Option<Vec<_>>>()?;
  Some(Expr::List(rules.into()))
}

/// The character encoding a listener decodes `"Data"` with:
/// wolframscript's `$CharacterEncoding`.
#[cfg(not(target_arch = "wasm32"))]
const LISTENER_ENCODING: &str = "UTF8";

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The byte array expression for `bytes`.
#[cfg(not(target_arch = "wasm32"))]
fn byte_array(bytes: &[u8]) -> Expr {
  use base64::Engine;
  call1(
    "ByteArray",
    Expr::String(base64::engine::general_purpose::STANDARD.encode(bytes)),
  )
}

#[cfg(not(target_arch = "wasm32"))]
fn end_of_file() -> Expr {
  id_expr("EndOfFile")
}

/// Read one line, waiting for as much of it as the peer still owes. Lines
/// are read the way wolframscript reads them from any stream: blank lines
/// are skipped, and the separator ending the line is left unread unless
/// `consume` (`Read[sock, String]` takes it, `ReadLine` does not).
#[cfg(not(target_arch = "wasm32"))]
fn read_line(
  uuid: &str,
  consume: bool,
) -> Result<Option<String>, SocketUnusable> {
  loop {
    let found = with_registry(|reg| {
      let buffer = &reg.entries.get(uuid)?.read_buffer;
      let start = buffer.iter().position(|b| !matches!(b, b'\n' | b'\r'))?;
      let end = start
        + buffer[start..]
          .iter()
          .position(|b| matches!(b, b'\n' | b'\r'))?;
      let next = if buffer[end..].starts_with(b"\r\n") {
        end + 2
      } else {
        end + 1
      };
      Some((start, end, next))
    });
    if let Some((start, end, next)) = found {
      let line = take_buffered(uuid, Some(end));
      if consume {
        take_buffered(uuid, Some(next - end));
      }
      return Ok(Some(String::from_utf8_lossy(&line[start..]).into_owned()));
    }
    if at_eof(uuid) || fill_buffer(uuid, true)? == 0 {
      let rest = take_buffered(uuid, None);
      let text = String::from_utf8_lossy(&rest);
      let text = text.trim_start_matches(['\n', '\r']);
      return Ok((!text.is_empty()).then(|| text.to_string()));
    }
  }
}

/// The socket functions, plus the socket cases of the stream functions
/// (`Close`, `ReadString`, `Write`, …). Returns `None` for anything that is
/// not about a socket, which leaves it to the ordinary dispatch.
#[cfg(not(target_arch = "wasm32"))]
pub fn dispatch_socket_functions(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  dispatch_native(name, args)
}

/// The browser has no TCP stack to offer, so the socket functions stay
/// unevaluated there rather than pretending to work.
#[cfg(target_arch = "wasm32")]
pub fn dispatch_socket_functions(
  _name: &str,
  _args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  None
}

#[cfg(not(target_arch = "wasm32"))]
fn dispatch_native(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  // The socket argument the stream functions would be reading or writing,
  // so a file-shaped call never lands in the socket code by accident.
  let socket_arg = args.first().and_then(socket_object_uuid);
  // A ZeroMQ socket type that cannot serve this stream function says so
  // before anything is read.
  if let Some(uuid) = &socket_arg
    && matches!(name, "ReadString" | "ReadLine" | "BinaryReadList" | "Read")
    && let Some(failure) = socket_function_unsupported(uuid, name)
  {
    return Some(Ok(failure));
  }
  // SocketReadMessage[sock, "Blocking" -> b, "Multipart" -> m] on a
  // ZeroMQ socket reads one message.
  if name == "SocketReadMessage"
    && let Some(uuid) = &socket_arg
    && let Some(zmq) = open_zmq(uuid)
  {
    let mut blocking = true;
    let mut multipart = true;
    for option in &args[1..] {
      let (Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      }) = option
      else {
        return Some(Ok(unevaluated(name, args)));
      };
      let key = match &**pattern {
        Expr::String(k) | Expr::Identifier(k) => k.as_str(),
        _ => return Some(Ok(unevaluated(name, args))),
      };
      let value = match &**replacement {
        Expr::Identifier(v) if v == "True" => Some(true),
        Expr::Identifier(v) if v == "False" => Some(false),
        _ => None,
      };
      // A value that is not a boolean falls back to the default, as in
      // wolframscript (which also warns).
      match key {
        "Blocking" => blocking = value.unwrap_or(true),
        "Multipart" => multipart = value.unwrap_or(true),
        "MultipartComplete" => {}
        _ => return Some(Ok(unevaluated(name, args))),
      }
    }
    return Some(Ok(zmq_read_message(&zmq, blocking, multipart)));
  }
  match name {
    "SocketOpen" if (1..=2).contains(&args.len()) => {
      Some(Ok(socket_open(args)))
    }
    "SocketConnect" if (1..=2).contains(&args.len()) => {
      Some(Ok(socket_connect(args)))
    }
    "SocketListen" if (2..=3).contains(&args.len()) => {
      Some(Ok(socket_listen(args)))
    }
    // SocketReadMessage[sock] — everything that has arrived, waiting for
    // the first byte. SocketReadMessage[sock, n] — up to n bytes of what
    // is already there, and unevaluated when nothing is.
    "SocketReadMessage" if args.len() == 1 => {
      let uuid = socket_arg?;
      Some(Ok(match read_available(&uuid) {
        Err(SocketUnusable) => fail_expr(),
        Ok(bytes) if bytes.is_empty() => end_of_file(),
        Ok(bytes) => byte_array(&bytes),
      }))
    }
    "SocketReadMessage" if args.len() == 2 => {
      let uuid = socket_arg?;
      let Expr::Integer(limit) = &args[1] else {
        return Some(Ok(unevaluated(name, args)));
      };
      let limit = usize::try_from(*limit).ok()?;
      if buffered_len(&uuid) == 0
        && let Err(SocketUnusable) = fill_buffer(&uuid, false)
      {
        return Some(Ok(fail_expr()));
      }
      let bytes = take_buffered(&uuid, Some(limit));
      Some(Ok(if bytes.is_empty() {
        unevaluated(name, args)
      } else {
        byte_array(&bytes)
      }))
    }
    // SocketReadyQ[sock] asks now; SocketReadyQ[sock, t] waits up to t
    // seconds for the answer to become True.
    "SocketReadyQ" if (1..=2).contains(&args.len()) => {
      let uuid = socket_arg?;
      let deadline = match args.get(1) {
        None => None,
        Some(expr) => match crate::functions::math_ast::try_eval_to_f64(expr) {
          Some(seconds) if seconds >= 0.0 => {
            Some(Instant::now() + Duration::from_secs_f64(seconds))
          }
          _ => return Some(Ok(unevaluated(name, args))),
        },
      };
      loop {
        pump_socket_events();
        if socket_ready(&uuid) {
          return Some(Ok(crate::helpers::bool_expr(true)));
        }
        match deadline {
          Some(d) if Instant::now() < d => {
            std::thread::sleep(Duration::from_millis(1));
          }
          _ => return Some(Ok(crate::helpers::bool_expr(false))),
        }
      }
    }
    "SocketWaitNext" if (1..=2).contains(&args.len()) => {
      Some(Ok(socket_wait(name, args, false)))
    }
    "SocketWaitAll" if (1..=2).contains(&args.len()) => {
      Some(Ok(socket_wait(name, args, true)))
    }
    // Sockets[] — the sockets opened here, oldest first. Sockets[filter]
    // narrows them by "TCP", "ZMQ", "Client", "Server", "Local" or
    // "Remote"; Sockets[All] adds the connections servers accepted;
    // Sockets[{"ZMQ", type}] picks one ZeroMQ socket type.
    "Sockets" if args.len() <= 2 => Some(Ok(sockets(args))),
    "Close" if args.len() == 1 && socket_arg.is_some() => {
      Some(Ok(socket_close(&socket_arg?)))
    }
    // DeleteObject[SocketListener[id]] stops the listener; on a socket it
    // is the same as closing it.
    "DeleteObject" if args.len() == 1 => {
      if let Some(id) = listener_object_id(&args[0]) {
        if !stop_listener(id) {
          crate::emit_message_to_stdout(&format!(
            "The socket listener {} is invalid or not open.",
            expr_to_string(&args[0])
          ));
          return Some(Ok(fail_expr()));
        }
        return Some(Ok(null_expr()));
      }
      // On a socket `DeleteObject` *is* `Close`, down to the endpoint string
      // it answers with.
      let uuid = socket_arg?;
      Some(Ok(socket_close(&uuid)))
    }
    // ReadString[sock] — everything up to the peer's close.
    "ReadString" if args.len() == 1 && socket_arg.is_some() => {
      let uuid = socket_arg?;
      Some(Ok(match read_to_close(&uuid) {
        Err(SocketUnusable) => fail_expr(),
        Ok(bytes) if bytes.is_empty() => end_of_file(),
        Ok(bytes) => Expr::String(String::from_utf8_lossy(&bytes).into_owned()),
      }))
    }
    "ReadLine" if args.len() == 1 && socket_arg.is_some() => {
      let uuid = socket_arg?;
      Some(Ok(match read_line(&uuid, false) {
        Err(SocketUnusable) => fail_expr(),
        Ok(None) => end_of_file(),
        Ok(Some(line)) => Expr::String(line),
      }))
    }
    "BinaryReadList" if args.len() == 1 && socket_arg.is_some() => {
      let uuid = socket_arg?;
      Some(Ok(match read_to_close(&uuid) {
        Err(SocketUnusable) => fail_expr(),
        Ok(bytes) => Expr::List(
          bytes
            .iter()
            .map(|b| Expr::Integer(i128::from(*b)))
            .collect::<Vec<_>>()
            .into(),
        ),
      }))
    }
    // Read[sock, type] for the element types a socket can serve. Anything
    // that would need the expression parser (`Expression`, `Number`, …)
    // is left unevaluated rather than half-supported.
    "Read" if args.len() == 2 && socket_arg.is_some() => {
      let uuid = socket_arg?;
      let element = match &args[1] {
        Expr::Identifier(s) | Expr::String(s) => s.clone(),
        _ => return Some(Ok(unevaluated(name, args))),
      };
      Some(Ok(match element.as_str() {
        "Byte" => match read_one_byte(&uuid) {
          Err(SocketUnusable) => fail_expr(),
          Ok(None) => end_of_file(),
          Ok(Some(byte)) => Expr::Integer(i128::from(byte)),
        },
        "Character" => match read_one_byte(&uuid) {
          Err(SocketUnusable) => fail_expr(),
          Ok(None) => end_of_file(),
          Ok(Some(byte)) => {
            Expr::String(String::from_utf8_lossy(&[byte]).into_owned())
          }
        },
        "String" | "Record" => match read_line(&uuid, true) {
          Err(SocketUnusable) => fail_expr(),
          Ok(None) => end_of_file(),
          Ok(Some(line)) => Expr::String(line),
        },
        _ => unevaluated(name, args),
      }))
    }
    _ => None,
  }
}

/// `Sockets[…]`, filtered the way wolframscript's `getFilteredSockets`
/// does.
#[cfg(not(target_arch = "wasm32"))]
fn sockets(args: &[Expr]) -> Expr {
  register_accepted();
  // (protocol, ZeroMQ type) of each open socket, plus whether it is a
  // connection a server accepted ("Remote") and its direction.
  let filter: Box<dyn Fn(&SocketEntry) -> bool> = match args {
    [] => Box::new(|e| e.role != SocketRole::Accepted),
    [Expr::Identifier(all)] if all == "All" => Box::new(|_| true),
    [Expr::String(filter)] => match filter.as_str() {
      "Client" => Box::new(|e| e.role != SocketRole::Server),
      "Server" => Box::new(|e| e.role == SocketRole::Server),
      "TCP" => Box::new(|e| e.zmq.is_none()),
      "ZMQ" => Box::new(|e| e.zmq.is_some()),
      "WS" | "WSS" => Box::new(|_| false),
      "Local" => Box::new(|e| e.role != SocketRole::Accepted),
      "Remote" => Box::new(|e| e.role == SocketRole::Accepted),
      other => return no_protocol("Sockets", other),
    },
    _ => {
      let spec = if args.len() == 2 {
        Expr::List(args.to_vec().into())
      } else {
        args[0].clone()
      };
      match socket_spec("Sockets", &spec) {
        Err(failure) => return failure,
        Ok(SocketSpec::Tcp) => Box::new(|e| e.zmq.is_none()),
        Ok(SocketSpec::Zmq(kind)) => Box::new(move |e| {
          e.zmq.as_ref().is_some_and(|z| z.socket.kind.name() == kind)
        }),
      }
    }
  };
  let open = with_registry(|reg| {
    reg
      .order
      .iter()
      .filter(|uuid| {
        reg
          .entries
          .get(*uuid)
          .is_some_and(|entry| !entry.closed && filter(entry))
      })
      .map(|uuid| socket_expr(uuid))
      .collect::<Vec<_>>()
  });
  Expr::List(open.into())
}

/// One byte, waiting for it if need be. `None` once the peer has closed.
#[cfg(not(target_arch = "wasm32"))]
fn read_one_byte(uuid: &str) -> Result<Option<u8>, SocketUnusable> {
  while buffered_len(uuid) == 0 {
    if at_eof(uuid) || fill_buffer(uuid, true)? == 0 {
      return Ok(None);
    }
  }
  Ok(take_buffered(uuid, Some(1)).first().copied())
}

/// The browser build has no sockets, so a `SocketObject` there is just an
/// inert expression with no properties to report.
#[cfg(target_arch = "wasm32")]
pub fn socket_property(_uuid: &str, _property: &str) -> Option<Expr> {
  None
}

#[cfg(target_arch = "wasm32")]
pub fn listener_property(_id: i128, _property: &str) -> Option<Expr> {
  None
}

#[cfg(target_arch = "wasm32")]
pub fn socket_property_list(_uuid: &str, _names: &[Expr]) -> Option<Expr> {
  None
}

#[cfg(target_arch = "wasm32")]
pub fn socket_all_options(_uuid: &str) -> Option<Expr> {
  None
}

#[cfg(target_arch = "wasm32")]
pub fn socket_set_options(_uuid: &str, _rules: &[Expr]) -> Option<Expr> {
  None
}

#[cfg(target_arch = "wasm32")]
pub fn listener_property_list(_id: i128, _names: &[Expr]) -> Option<Expr> {
  None
}
