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

#[allow(unused_imports)]
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
  /// `SocketListen[sock, f]` registers `f` under `"Received"`.
  handlers: Vec<(String, Expr)>,
  handler_keys: Vec<String>,
  stop: Arc<AtomicBool>,
}

#[cfg(not(target_arch = "wasm32"))]
enum EventKind {
  Accepted,
  Data(Vec<u8>),
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
  let args = [Expr::String(prefix.to_string())];
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
  call1("SocketObject", Expr::String(uuid.to_string()))
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
    crate::syntax::expr_to_string(&socket_expr(uuid))
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

/// Read an endpoint out of the argument list, ignoring a trailing
/// `"TCP"` protocol argument. `None` for anything that is not an endpoint,
/// which leaves the call unevaluated.
#[cfg(not(target_arch = "wasm32"))]
fn parse_endpoint(args: &[Expr], default_host: &str) -> Option<Endpoint> {
  // The protocol argument is accepted but carries no information: TCP is
  // the only transport implemented, and the only one `SocketOpen`'s
  // default names.
  let args: Vec<&Expr> = args
    .iter()
    .filter(|a| !matches!(a, Expr::String(s) if s == "TCP"))
    .collect();
  match args.as_slice() {
    [Expr::Integer(port)] => Some(Endpoint {
      host: default_host.to_string(),
      port: u16::try_from(*port).ok()?,
    }),
    [Expr::String(spec)] => parse_host_port(spec, default_host),
    [Expr::String(host), Expr::Integer(port)] => Some(Endpoint {
      host: host.clone(),
      port: u16::try_from(*port).ok()?,
    }),
    [Expr::List(items)] => {
      let items: Vec<Expr> = items.iter().cloned().collect();
      let refs: Vec<&Expr> = items.iter().collect();
      match refs.as_slice() {
        [Expr::String(host), Expr::Integer(port)] => Some(Endpoint {
          host: host.clone(),
          port: u16::try_from(*port).ok()?,
        }),
        _ => None,
      }
    }
    _ => None,
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
  let Some(endpoint) = parse_endpoint(args, "127.0.0.1") else {
    return unevaluated("SocketOpen", args);
  };
  let listener =
    match TcpListener::bind((endpoint.host.as_str(), endpoint.port)) {
      Ok(listener) => listener,
      Err(err) => {
        failed_operation_message(&err);
        return Expr::Identifier("$Failed".to_string());
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
  let Some(endpoint) = parse_endpoint(args, "127.0.0.1") else {
    return unevaluated("SocketConnect", args);
  };
  let resolved: Vec<SocketAddr> = (endpoint.host.as_str(), endpoint.port)
    .to_socket_addrs()
    .map(std::iter::Iterator::collect)
    .unwrap_or_default();
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
    },
  )
}

// ---------------------------------------------------------------------------
// Listening
// ---------------------------------------------------------------------------

/// The handler functions a `SocketListen` call sets up, as
/// `(event name, function)` pairs.
///
/// `SocketListen[sock, f]` is the short form for
/// `HandlerFunctions -> <|"Received" -> f|>`; the long form may name any
/// of `"Accepted"`, `"Received"`, `"Closed"` and `"Error"`.
#[cfg(not(target_arch = "wasm32"))]
fn parse_handlers(spec: &[Expr]) -> Option<Vec<(String, Expr)>> {
  let mut handlers = Vec::new();
  for arg in spec {
    match arg {
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
      // Anything else in the second position is the "Received" handler.
      other => handlers.push(("Received".to_string(), other.clone())),
    }
  }
  Some(handlers)
}

/// `SocketListen[sock, handler]` — run `handler` for everything that
/// arrives on `sock`, and start accepting connections if it is a server.
#[cfg(not(target_arch = "wasm32"))]
fn socket_listen(args: &[Expr]) -> Expr {
  let Some(uuid) = socket_object_uuid(&args[0]) else {
    return unevaluated("SocketListen", args);
  };
  let Some(handlers) = parse_handlers(&args[1..]) else {
    return unevaluated("SocketListen", args);
  };
  let listener_state = with_registry(|reg| {
    reg.entries.get(&uuid).map(|entry| {
      (
        entry.closed,
        entry.role,
        entry.listener.clone(),
        entry.stream.clone(),
      )
    })
  });
  let Some((closed, role, listener, stream)) = listener_state else {
    invalid_socket_message(&uuid);
    return Expr::Identifier("$Failed".to_string());
  };
  if closed {
    invalid_socket_message(&uuid);
    return Expr::Identifier("$Failed".to_string());
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
    crate::with_rng(|rng| rand::RngCore::fill_bytes(rng, &mut bytes));
    i128::from(u64::from_be_bytes(bytes) >> 1)
  };
  let handler_keys = vec![
    "TimeStamp".to_string(),
    "SourceSocket".to_string(),
    "Socket".to_string(),
    "Data".to_string(),
    "DataBytes".to_string(),
    "DataByteArray".to_string(),
    "MultipartComplete".to_string(),
  ];
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
        Ok(n) => EventKind::Data(buffer[..n].to_vec()),
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
      let last = !matches!(event, EventKind::Data(_));
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
    use rand::RngCore;
    rand::rngs::OsRng.fill_bytes(&mut bytes);
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
    EventKind::Data(_) => "Received",
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
  let timestamp = crate::functions::datetime_ast::absolute_time_ast(&[])
    .unwrap_or(Expr::Real(0.0));
  let mut pairs = vec![
    (Expr::String("TimeStamp".to_string()), timestamp),
    (
      Expr::String("SourceSocket".to_string()),
      socket_expr(&event.socket),
    ),
    (
      Expr::String("Socket".to_string()),
      socket_expr(listen_socket),
    ),
  ];
  match &event.kind {
    EventKind::Data(bytes) => {
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
        call1("ByteArrayToString", byte_array.clone()),
      ));
      pairs.push(delayed("DataBytes", call1("Normal", byte_array.clone())));
      pairs.push((Expr::String("DataByteArray".to_string()), byte_array));
      pairs.push((
        Expr::String("MultipartComplete".to_string()),
        crate::helpers::bool_expr(true),
      ));
    }
    EventKind::Error(message) => {
      pairs.push((
        Expr::String("Message".to_string()),
        Expr::String(message.clone()),
      ));
    }
    EventKind::Accepted | EventKind::Closed => {}
  }
  Expr::Association(pairs)
}

/// An association entry written `key :> value`, using the `RuleDelayed`
/// marker `Expr::Association` uses for one.
#[cfg(not(target_arch = "wasm32"))]
fn delayed(key: &str, value: Expr) -> (Expr, Expr) {
  let key = Expr::String(key.to_string());
  (
    key.clone(),
    Expr::RuleDelayed {
      pattern: Box::new(key),
      replacement: Box::new(value),
    },
  )
}

// ---------------------------------------------------------------------------
// Blocking reads and writes
// ---------------------------------------------------------------------------

/// Why a socket operation could not go ahead. Both cases print the same
/// free-text line wolframscript prints; they are kept apart only so the
/// caller knows whether anything was said.
#[cfg(not(target_arch = "wasm32"))]
struct SocketUnusable;

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
  let Ok(stream) = usable_stream(uuid) else {
    return false;
  };
  if let Ok(()) = (&*stream)
    .write_all(bytes)
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
    return Expr::Identifier("$Failed".to_string());
  }
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
  socket_expr(uuid)
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
  Expr::List(
    items
      .iter()
      .map(|s| Expr::String((*s).to_string()))
      .collect::<Vec<_>>()
      .into(),
  )
}

/// `SocketObject[…]["property"]`. An unknown property gives the empty list,
/// which is what wolframscript answers for one it does not keep.
#[cfg(not(target_arch = "wasm32"))]
pub fn socket_property(uuid: &str, property: &str) -> Option<Expr> {
  if property == "Properties" {
    return Some(string_list(&SOCKET_PROPERTIES));
  }
  let empty = || Expr::List(Vec::new().into());
  with_registry(|reg| {
    let entry = reg.entries.get(uuid)?;
    Some(match property {
      "ConnectedClients" => Expr::List(
        entry
          .children
          .iter()
          .map(|c| socket_expr(c))
          .collect::<Vec<_>>()
          .into(),
      ),
      "DestinationHostname" => Expr::String(entry.dest_host.clone()),
      "DestinationIPAddress" => {
        call1("IPAddress", Expr::String(entry.dest_ip.clone()))
      }
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
      "Protocol" => Expr::String("TCP".to_string()),
      "Scheme" => Expr::String("tcp".to_string()),
      "SocketListener" => entry
        .listener_id
        .map_or_else(empty, |id| call1("SocketListener", Expr::Integer(id))),
      "Type" => Expr::String("ZMQ_STREAM".to_string()),
      "UUID" => Expr::String(uuid.to_string()),
      _ => empty(),
    })
  })
}

/// `SocketListener[…]["property"]`.
#[cfg(not(target_arch = "wasm32"))]
pub fn listener_property(id: i128, property: &str) -> Option<Expr> {
  if property == "Properties" {
    return Some(string_list(&LISTENER_PROPERTIES));
  }
  let empty = || Expr::List(Vec::new().into());
  LISTENERS.with(|l| {
    let listeners = l.borrow();
    let entry = listeners.get(&id)?;
    Some(match property {
      "CharacterEncoding" => Expr::String("UTF-8".to_string()),
      "HandlerFunctions" => Expr::Association(
        entry
          .handlers
          .iter()
          .map(|(name, f)| (Expr::String(name.clone()), f.clone()))
          .collect(),
      ),
      "HandlerFunctionsKeys" => Expr::List(
        entry
          .handler_keys
          .iter()
          .map(|k| Expr::String(k.clone()))
          .collect::<Vec<_>>()
          .into(),
      ),
      "RecordSeparators" => empty(),
      "Socket" => socket_expr(&entry.socket),
      _ => empty(),
    })
  })
}

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
  Expr::Identifier("EndOfFile".to_string())
}

/// Read one line, waiting for as much of it as the peer still owes.
#[cfg(not(target_arch = "wasm32"))]
fn read_line(uuid: &str) -> Result<Option<String>, SocketUnusable> {
  loop {
    let found = with_registry(|reg| {
      reg
        .entries
        .get(uuid)
        .and_then(|e| e.read_buffer.iter().position(|b| *b == b'\n'))
    });
    if let Some(index) = found {
      let mut line = take_buffered(uuid, Some(index + 1));
      line.pop();
      if line.last() == Some(&b'\r') {
        line.pop();
      }
      return Ok(Some(String::from_utf8_lossy(&line).into_owned()));
    }
    if at_eof(uuid) || fill_buffer(uuid, true)? == 0 {
      let rest = take_buffered(uuid, None);
      return Ok(if rest.is_empty() {
        None
      } else {
        Some(String::from_utf8_lossy(&rest).into_owned())
      });
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
        Err(SocketUnusable) => Expr::Identifier("$Failed".to_string()),
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
        return Some(Ok(Expr::Identifier("$Failed".to_string())));
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
    // Sockets[] — every socket still open, oldest first.
    // Sockets["TCP"] — the same, since TCP is the only transport.
    "Sockets" if args.len() <= 1 => {
      if let Some(arg) = args.first()
        && !matches!(arg, Expr::String(s) if s == "TCP")
      {
        return Some(Ok(unevaluated(name, args)));
      }
      register_accepted();
      let open = with_registry(|reg| {
        reg
          .order
          .iter()
          .filter(|uuid| {
            reg.entries.get(*uuid).is_some_and(|entry| !entry.closed)
          })
          .map(|uuid| socket_expr(uuid))
          .collect::<Vec<_>>()
      });
      Some(Ok(Expr::List(open.into())))
    }
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
            crate::syntax::expr_to_string(&args[0])
          ));
          return Some(Ok(Expr::Identifier("$Failed".to_string())));
        }
        return Some(Ok(Expr::Identifier("Null".to_string())));
      }
      let uuid = socket_arg?;
      socket_close(&uuid);
      Some(Ok(Expr::Identifier("Null".to_string())))
    }
    // ReadString[sock] — everything up to the peer's close.
    "ReadString" if args.len() == 1 && socket_arg.is_some() => {
      let uuid = socket_arg?;
      Some(Ok(match read_to_close(&uuid) {
        Err(SocketUnusable) => Expr::Identifier("$Failed".to_string()),
        Ok(bytes) if bytes.is_empty() => end_of_file(),
        Ok(bytes) => Expr::String(String::from_utf8_lossy(&bytes).into_owned()),
      }))
    }
    "ReadLine" if args.len() == 1 && socket_arg.is_some() => {
      let uuid = socket_arg?;
      Some(Ok(match read_line(&uuid) {
        Err(SocketUnusable) => Expr::Identifier("$Failed".to_string()),
        Ok(None) => end_of_file(),
        Ok(Some(line)) => Expr::String(line),
      }))
    }
    "BinaryReadList" if args.len() == 1 && socket_arg.is_some() => {
      let uuid = socket_arg?;
      Some(Ok(match read_to_close(&uuid) {
        Err(SocketUnusable) => Expr::Identifier("$Failed".to_string()),
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
          Err(SocketUnusable) => Expr::Identifier("$Failed".to_string()),
          Ok(None) => end_of_file(),
          Ok(Some(byte)) => Expr::Integer(i128::from(byte)),
        },
        "Character" => match read_one_byte(&uuid) {
          Err(SocketUnusable) => Expr::Identifier("$Failed".to_string()),
          Ok(None) => end_of_file(),
          Ok(Some(byte)) => {
            Expr::String(String::from_utf8_lossy(&[byte]).into_owned())
          }
        },
        "String" | "Record" => match read_line(&uuid) {
          Err(SocketUnusable) => Expr::Identifier("$Failed".to_string()),
          Ok(None) => end_of_file(),
          Ok(Some(line)) => Expr::String(line),
        },
        _ => unevaluated(name, args),
      }))
    }
    _ => None,
  }
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
