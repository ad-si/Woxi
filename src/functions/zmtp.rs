//! ZeroMQ sockets over TCP, spoken natively: the ZMTP 3.0 wire protocol
//! (<https://rfc.zeromq.org/spec/23/>) with the NULL security mechanism, and
//! the routing rules of each socket type (PUB/SUB, REQ/REP, PUSH/PULL, …).
//!
//! Nothing here knows about `Expr`. [`ZmqSocket`] is shared between the
//! evaluator thread, which sends and receives, and the background threads
//! that accept, connect and read — one per listening socket, one per
//! connection. Incoming messages queue up inside the socket until the
//! evaluator asks for them, or go straight to a listener callback when
//! `SocketListen` has attached one.
//!
//! Like libzmq, a connecting socket keeps reconnecting in the background
//! for as long as it is open, and messages sent while no peer is there are
//! queued for the first one that arrives (except on PUB, which drops them).
//!
//! Greeting minor version 0 is announced on purpose: a ZMTP 3.1 peer then
//! talks 3.0 to us, which means subscriptions travel as ordinary messages
//! (`\x01topic`). The 3.1 `SUBSCRIBE`/`CANCEL` commands are understood all
//! the same, in case a peer sends them regardless.

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// How long a background read blocks before looking at the stop flag.
const POLL_INTERVAL: Duration = Duration::from_millis(50);
/// How long a connection may take to finish its handshake (libzmq's
/// `ZMQ_HANDSHAKE_IVL` default).
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);
/// Pause between reconnection attempts (libzmq's `ZMQ_RECONNECT_IVL`).
const RECONNECT_INTERVAL: Duration = Duration::from_millis(100);
/// How long a single TCP connection attempt may take.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);
/// How long `connect` waits for the first attempt to settle, so that a
/// peer which is already listening is connected by the time
/// `SocketConnect` returns. A peer that is not there yet is simply retried
/// in the background, the way libzmq does.
const FIRST_ATTEMPT_WAIT: Duration = Duration::from_secs(1);

const FLAG_MORE: u8 = 0x01;
const FLAG_LONG: u8 = 0x02;
const FLAG_COMMAND: u8 = 0x04;

/// The socket types of `zmq.h`, minus the raw `ZMQ_STREAM`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ZmqType {
  Pair,
  Pub,
  Sub,
  Req,
  Rep,
  Dealer,
  Router,
  Pull,
  Push,
  XPub,
  XSub,
}

impl ZmqType {
  /// `"ZMQ_PUB"` → `Pub`. Exact, upper-case names only: spelling variants
  /// are the business of the socket-spec parser.
  pub fn from_name(name: &str) -> Option<Self> {
    Some(match name {
      "ZMQ_PAIR" => Self::Pair,
      "ZMQ_PUB" => Self::Pub,
      "ZMQ_SUB" => Self::Sub,
      "ZMQ_REQ" => Self::Req,
      "ZMQ_REP" => Self::Rep,
      "ZMQ_DEALER" => Self::Dealer,
      "ZMQ_ROUTER" => Self::Router,
      "ZMQ_PULL" => Self::Pull,
      "ZMQ_PUSH" => Self::Push,
      "ZMQ_XPUB" => Self::XPub,
      "ZMQ_XSUB" => Self::XSub,
      _ => return None,
    })
  }

  /// The `zmq.h` constant name, which is also what `sock["Type"]` reports.
  pub fn name(self) -> &'static str {
    match self {
      Self::Pair => "ZMQ_PAIR",
      Self::Pub => "ZMQ_PUB",
      Self::Sub => "ZMQ_SUB",
      Self::Req => "ZMQ_REQ",
      Self::Rep => "ZMQ_REP",
      Self::Dealer => "ZMQ_DEALER",
      Self::Router => "ZMQ_ROUTER",
      Self::Pull => "ZMQ_PULL",
      Self::Push => "ZMQ_PUSH",
      Self::XPub => "ZMQ_XPUB",
      Self::XSub => "ZMQ_XSUB",
    }
  }

  /// The `zmq.h` constant value, which is what `ZMQ_TYPE` reports.
  pub fn code(self) -> i128 {
    match self {
      Self::Pair => 0,
      Self::Pub => 1,
      Self::Sub => 2,
      Self::Req => 3,
      Self::Rep => 4,
      Self::Dealer => 5,
      Self::Router => 6,
      Self::Pull => 7,
      Self::Push => 8,
      Self::XPub => 9,
      Self::XSub => 10,
    }
  }

  /// The name the `Socket-Type` handshake property carries.
  fn wire_name(self) -> &'static str {
    &self.name()[4..]
  }

  /// Whether a peer announcing `peer` may talk to this type (ZMTP 3.0,
  /// "Socket Type Compatibility").
  fn accepts(self, peer: &[u8]) -> bool {
    let allowed: &[&str] = match self {
      Self::Pair => &["PAIR"],
      Self::Pub | Self::XPub => &["SUB", "XSUB"],
      Self::Sub | Self::XSub => &["PUB", "XPUB"],
      Self::Req => &["REP", "ROUTER"],
      Self::Rep => &["REQ", "DEALER"],
      Self::Dealer => &["REP", "DEALER", "ROUTER"],
      Self::Router => &["REQ", "DEALER", "ROUTER"],
      Self::Pull => &["PUSH"],
      Self::Push => &["PULL"],
    };
    allowed.iter().any(|name| name.as_bytes() == peer)
  }

  /// Whether messages can be received at all (PUB and PUSH only send).
  pub fn can_receive(self) -> bool {
    !matches!(self, Self::Pub | Self::Push)
  }

  /// Whether messages can be sent at all (SUB and PULL only receive).
  pub fn can_send(self) -> bool {
    !matches!(self, Self::Sub | Self::Pull)
  }

  /// Whether the handshake announces an `Identity` (libzmq does so for the
  /// types whose peers route by it).
  fn sends_identity(self) -> bool {
    matches!(self, Self::Req | Self::Dealer | Self::Router)
  }
}

// ---------------------------------------------------------------------------
// Wire format
// ---------------------------------------------------------------------------

/// The 64-byte ZMTP 3.0 greeting: signature, version 3.0, mechanism NULL,
/// as-server 0. The padding ends in 1 the way libzmq writes it, which keeps
/// ZMTP 1.0 peers from misreading the signature as a huge frame.
fn greeting() -> [u8; 64] {
  let mut g = [0u8; 64];
  g[0] = 0xFF;
  g[8] = 0x01;
  g[9] = 0x7F;
  g[10] = 3;
  g[11] = 0;
  g[12..16].copy_from_slice(b"NULL");
  g
}

fn encode_frame(out: &mut Vec<u8>, body: &[u8], more: bool, command: bool) {
  let mut flags = 0u8;
  if more {
    flags |= FLAG_MORE;
  }
  if command {
    flags |= FLAG_COMMAND;
  }
  if body.len() > 255 {
    out.push(flags | FLAG_LONG);
    out.extend_from_slice(&(body.len() as u64).to_be_bytes());
  } else {
    out.push(flags);
    out.push(body.len() as u8);
  }
  out.extend_from_slice(body);
}

fn encode_message(frames: &[Vec<u8>]) -> Vec<u8> {
  let mut out = Vec::with_capacity(frames.iter().map(|f| f.len() + 9).sum());
  for (i, frame) in frames.iter().enumerate() {
    encode_frame(&mut out, frame, i + 1 < frames.len(), false);
  }
  out
}

/// A command frame: name length, name, then the command's own data.
fn encode_command(name: &str, data: &[u8]) -> Vec<u8> {
  let mut body = Vec::with_capacity(1 + name.len() + data.len());
  body.push(name.len() as u8);
  body.extend_from_slice(name.as_bytes());
  body.extend_from_slice(data);
  let mut out = Vec::new();
  encode_frame(&mut out, &body, false, true);
  out
}

/// An ERROR command: a length-prefixed reason, after which the sender
/// closes the connection.
fn error_command(reason: &str) -> Vec<u8> {
  let reason = &reason.as_bytes()[..reason.len().min(255)];
  let mut data = vec![reason.len() as u8];
  data.extend_from_slice(reason);
  encode_command("ERROR", &data)
}

fn ready_command(kind: ZmqType, identity: &[u8]) -> Vec<u8> {
  let mut props = Vec::new();
  let mut property = |name: &str, value: &[u8]| {
    props.push(name.len() as u8);
    props.extend_from_slice(name.as_bytes());
    props.extend_from_slice(&(value.len() as u32).to_be_bytes());
    props.extend_from_slice(value);
  };
  property("Socket-Type", kind.wire_name().as_bytes());
  if kind.sends_identity() {
    property("Identity", identity);
  }
  encode_command("READY", &props)
}

/// The metadata properties of a READY command, as `(name, value)` pairs.
fn parse_properties(mut data: &[u8]) -> Option<Vec<(String, Vec<u8>)>> {
  let mut props = Vec::new();
  while !data.is_empty() {
    let name_len = usize::from(data[0]);
    let name = data.get(1..1 + name_len)?;
    data = &data[1 + name_len..];
    let len_bytes: [u8; 4] = data.get(..4)?.try_into().ok()?;
    let value_len = u32::from_be_bytes(len_bytes) as usize;
    let value = data.get(4..4 + value_len)?;
    props.push((String::from_utf8_lossy(name).into_owned(), value.to_vec()));
    data = &data[4 + value_len..];
  }
  Some(props)
}

/// A command split into its name and data.
fn parse_command(body: &[u8]) -> Option<(&[u8], &[u8])> {
  let len = usize::from(*body.first()?);
  Some((body.get(1..1 + len)?, &body[1 + len..]))
}

struct Frame {
  more: bool,
  command: bool,
  body: Vec<u8>,
}

/// Bytes read off a connection but not yet parsed into frames. Reads use a
/// timeout, so a frame may arrive in pieces across several of them; nothing
/// read is ever thrown away.
#[derive(Default)]
struct FrameReader {
  buf: Vec<u8>,
}

impl FrameReader {
  /// The next complete frame, if the buffer holds one. `Err` for a frame
  /// whose length cannot be real (it would not fit in memory).
  fn next_frame(&mut self) -> Result<Option<Frame>, ()> {
    let Some(&flags) = self.buf.first() else {
      return Ok(None);
    };
    let (header, len): (usize, usize) = if flags & FLAG_LONG != 0 {
      let Some(bytes) = self.buf.get(1..9) else {
        return Ok(None);
      };
      let len = u64::from_be_bytes(bytes.try_into().map_err(|_| ())?);
      (9, usize::try_from(len).map_err(|_| ())?)
    } else {
      let Some(&len) = self.buf.get(1) else {
        return Ok(None);
      };
      (2, usize::from(len))
    };
    // A length no buffer could hold is a broken (or hostile) peer.
    let end = header.checked_add(len).ok_or(())?;
    if self.buf.len() < end {
      return Ok(None);
    }
    let body = self.buf[header..end].to_vec();
    self.buf.drain(..end);
    Ok(Some(Frame {
      more: flags & FLAG_MORE != 0,
      command: flags & FLAG_COMMAND != 0,
      body,
    }))
  }

  /// Read whatever the connection has into the buffer. `Ok(false)` when the
  /// read timed out with nothing, `Err` once the peer is gone.
  fn fill(&mut self, stream: &TcpStream) -> Result<bool, ()> {
    // Read straight into the buffer's spare capacity rather than through
    // a 64 KiB array on the stack of every connection thread.
    let start = self.buf.len();
    self.buf.resize(start + 64 * 1024, 0);
    let read = (&*stream).read(&mut self.buf[start..]);
    self.buf.truncate(start + *read.as_ref().unwrap_or(&0));
    match read {
      Ok(0) => Err(()),
      Ok(_) => Ok(true),
      Err(err)
        if matches!(
          err.kind(),
          std::io::ErrorKind::WouldBlock
            | std::io::ErrorKind::TimedOut
            | std::io::ErrorKind::Interrupted
        ) =>
      {
        Ok(false)
      }
      Err(_) => Err(()),
    }
  }
}

/// What the peer told us about itself during the handshake.
struct PeerHello {
  identity: Vec<u8>,
}

/// Run the NULL-mechanism handshake on a fresh connection. On success the
/// reader keeps whatever followed the peer's READY command.
fn handshake(
  stream: &TcpStream,
  kind: ZmqType,
  identity: &[u8],
  stop: &AtomicBool,
  reader: &mut FrameReader,
) -> Result<PeerHello, ()> {
  let deadline = Instant::now() + HANDSHAKE_TIMEOUT;
  let _ = stream.set_read_timeout(Some(POLL_INTERVAL));
  let give_up = || stop.load(Ordering::Relaxed) || Instant::now() >= deadline;
  (&*stream).write_all(&greeting()).map_err(|_| ())?;
  while reader.buf.len() < 64 {
    if give_up() {
      return Err(());
    }
    reader.fill(stream)?;
  }
  let peer = reader.buf.drain(..64).collect::<Vec<u8>>();
  // Signature, then a major version of at least 3.
  if peer[0] != 0xFF || peer[9] & 0x01 == 0 || peer[10] < 3 {
    return Err(());
  }
  let mechanism: Vec<u8> = peer[12..32]
    .iter()
    .copied()
    .take_while(|b| *b != 0)
    .collect();
  if mechanism != b"NULL" {
    let _ =
      (&*stream).write_all(&error_command("unsupported security mechanism"));
    return Err(());
  }
  (&*stream)
    .write_all(&ready_command(kind, identity))
    .map_err(|_| ())?;
  let frame = loop {
    if let Some(frame) = reader.next_frame()? {
      break frame;
    }
    if give_up() {
      return Err(());
    }
    reader.fill(stream)?;
  };
  if !frame.command {
    return Err(());
  }
  let (name, data) = parse_command(&frame.body).ok_or(())?;
  if name != b"READY" {
    return Err(());
  }
  let props = parse_properties(data).ok_or(())?;
  let peer_type = props
    .iter()
    .find(|(n, _)| n.eq_ignore_ascii_case("Socket-Type"))
    .map(|(_, v)| v.clone())
    .ok_or(())?;
  if !kind.accepts(&peer_type) {
    let _ = (&*stream).write_all(&error_command("invalid socket type"));
    return Err(());
  }
  let identity = props
    .into_iter()
    .find(|(n, _)| n.eq_ignore_ascii_case("Identity"))
    .map(|(_, v)| v)
    .unwrap_or_default();
  Ok(PeerHello { identity })
}

// ---------------------------------------------------------------------------
// Socket state
// ---------------------------------------------------------------------------

/// Called for every frame of every incoming message once a listener is
/// attached: the frame, and whether it is the last of its message.
pub type FrameCallback = Box<dyn Fn(Vec<u8>, bool) + Send>;

struct Peer {
  id: u64,
  stream: Arc<TcpStream>,
  /// The routing id a ROUTER prepends to this peer's messages.
  routing_id: Vec<u8>,
  /// What this peer subscribed to, for PUB and XPUB.
  subscriptions: Vec<Vec<u8>>,
}

struct Incoming {
  peer: u64,
  /// The REP envelope — everything up to and including the empty
  /// delimiter — which the reply has to carry back.
  envelope: Vec<Vec<u8>>,
  frames: Vec<Vec<u8>>,
}

#[derive(Default)]
struct Inner {
  peers: Vec<Peer>,
  next_peer: u64,
  incoming: VecDeque<Incoming>,
  /// The frames still unread of a message being read one part at a time.
  partial: VecDeque<Vec<u8>>,
  /// Messages waiting for a peer to appear.
  outbox: VecDeque<Vec<Vec<u8>>>,
  /// This socket's own subscriptions (SUB, XSUB), one entry per subscribe.
  subscriptions: Vec<Vec<u8>>,
  round_robin: usize,
  /// The peer and envelope a REP socket owes its next reply to.
  reply_to: Option<(u64, Vec<Vec<u8>>)>,
  identity: Vec<u8>,
  next_routing_id: u32,
  listener: Option<FrameCallback>,
}

/// One ZeroMQ socket: its peers, its queues, and the stop flag its
/// background threads watch.
pub struct ZmqSocket {
  pub kind: ZmqType,
  /// Made by `connect` rather than `bind`. A connecting socket counts as
  /// writable before its peer is there, since messages queue for it.
  connecting: bool,
  inner: Mutex<Inner>,
  arrived: Condvar,
  stop: AtomicBool,
  /// Set once a connecting socket's first attempt has succeeded or failed.
  first_attempt: (Mutex<bool>, Condvar),
}

/// A bound socket, with the address it ended up on.
pub struct Bound {
  pub socket: Arc<ZmqSocket>,
  pub local: SocketAddr,
  pub fd: i128,
}

/// The raw descriptor behind a socket, for `ZMQ_FD`.
#[cfg(unix)]
fn raw_fd<T: std::os::fd::AsRawFd>(t: &T) -> i128 {
  i128::from(t.as_raw_fd())
}
#[cfg(windows)]
fn raw_fd<T: std::os::windows::io::AsRawSocket>(t: &T) -> i128 {
  i128::from(t.as_raw_socket())
}

impl ZmqSocket {
  fn new(kind: ZmqType, connecting: bool) -> Arc<Self> {
    let inner = Inner {
      // libzmq starts its generated routing ids at a random value.
      next_routing_id: {
        let mut bytes = [0u8; 4];
        use rand::TryRng;
        let _ = rand::rngs::SysRng.try_fill_bytes(&mut bytes);
        u32::from_be_bytes(bytes)
      },
      ..Inner::default()
    };
    Arc::new(Self {
      kind,
      connecting,
      inner: Mutex::new(inner),
      arrived: Condvar::new(),
      stop: AtomicBool::new(false),
      first_attempt: (Mutex::new(false), Condvar::new()),
    })
  }

  fn lock(&self) -> MutexGuard<'_, Inner> {
    self
      .inner
      .lock()
      .unwrap_or_else(std::sync::PoisonError::into_inner)
  }

  /// Bind to `host:port` (port 0 picks a free one) and accept peers in the
  /// background.
  pub fn bind(kind: ZmqType, host: &str, port: u16) -> std::io::Result<Bound> {
    let listener = TcpListener::bind((host, port))?;
    let local = listener.local_addr()?;
    let fd = raw_fd(&listener);
    listener.set_nonblocking(true)?;
    let socket = Self::new(kind, false);
    let accepting = socket.clone();
    std::thread::spawn(move || {
      while !accepting.stop.load(Ordering::Relaxed) {
        match listener.accept() {
          Ok((stream, _)) => {
            let _ = stream.set_nonblocking(false);
            let _ = stream.set_nodelay(true);
            let serving = accepting.clone();
            std::thread::spawn(move || {
              serving.serve(stream);
            });
          }
          Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
            std::thread::sleep(Duration::from_millis(2));
          }
          Err(_) => std::thread::sleep(Duration::from_millis(20)),
        }
      }
    });
    Ok(Bound { socket, local, fd })
  }

  /// Connect to `host:port`, retrying in the background until a peer is
  /// there and reconnecting whenever it goes away. Waits for the first
  /// attempt, so an endpoint that is already listening is connected on
  /// return.
  pub fn connect(kind: ZmqType, host: &str, port: u16) -> Arc<Self> {
    let socket = Self::new(kind, true);
    let connecting = socket.clone();
    let host = host.to_string();
    std::thread::spawn(move || {
      while !connecting.stop.load(Ordering::Relaxed) {
        let stream = (host.as_str(), port)
          .to_socket_addrs()
          .ok()
          .into_iter()
          .flatten()
          .find_map(|addr| {
            TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT).ok()
          });
        match stream {
          Some(stream) => {
            let _ = stream.set_nodelay(true);
            connecting.serve(stream);
          }
          None => connecting.first_attempt_done(),
        }
        let resume = Instant::now() + RECONNECT_INTERVAL;
        while Instant::now() < resume
          && !connecting.stop.load(Ordering::Relaxed)
        {
          std::thread::sleep(Duration::from_millis(5));
        }
      }
      connecting.first_attempt_done();
    });
    let (done, cond) = &socket.first_attempt;
    let guard = done
      .lock()
      .unwrap_or_else(std::sync::PoisonError::into_inner);
    let _ = cond.wait_timeout_while(guard, FIRST_ATTEMPT_WAIT, |done| !*done);
    socket
  }

  fn first_attempt_done(&self) {
    let (done, cond) = &self.first_attempt;
    if let Ok(mut done) = done.lock()
      && !*done
    {
      *done = true;
      cond.notify_all();
    }
  }

  /// Handshake, then read messages until the connection ends.
  fn serve(self: &Arc<Self>, stream: TcpStream) {
    let mut reader = FrameReader::default();
    let identity = self.lock().identity.clone();
    let hello =
      handshake(&stream, self.kind, &identity, &self.stop, &mut reader);
    let Ok(hello) = hello else {
      self.first_attempt_done();
      let _ = stream.shutdown(std::net::Shutdown::Both);
      return;
    };
    let stream = Arc::new(stream);
    let Some(id) = self.register(&stream, hello) else {
      self.first_attempt_done();
      let _ = stream.shutdown(std::net::Shutdown::Both);
      return;
    };
    self.first_attempt_done();
    let mut message: Vec<Vec<u8>> = Vec::new();
    'read: while !self.stop.load(Ordering::Relaxed) {
      loop {
        match reader.next_frame() {
          Ok(Some(frame)) if frame.command => {
            if !self.command(id, &stream, &frame.body) {
              break 'read;
            }
          }
          Ok(Some(frame)) => {
            message.push(frame.body);
            if !frame.more {
              self.deliver(id, std::mem::take(&mut message));
            }
          }
          Ok(None) => break,
          Err(()) => break 'read,
        }
      }
      if reader.fill(&stream).is_err() {
        break;
      }
    }
    let _ = stream.shutdown(std::net::Shutdown::Both);
    self.unregister(id);
  }

  /// Add a handshaken peer. `None` when this socket takes no more peers
  /// (a PAIR already has its one).
  fn register(&self, stream: &Arc<TcpStream>, hello: PeerHello) -> Option<u64> {
    let mut inner = self.lock();
    if self.stop.load(Ordering::Relaxed)
      || (self.kind == ZmqType::Pair && !inner.peers.is_empty())
    {
      return None;
    }
    let id = inner.next_peer;
    inner.next_peer += 1;
    let routing_id = if hello.identity.is_empty() {
      // libzmq's generated routing ids: a zero byte, then a counter.
      inner.next_routing_id = inner.next_routing_id.wrapping_add(1);
      let mut rid = vec![0u8];
      rid.extend_from_slice(&inner.next_routing_id.to_be_bytes());
      rid
    } else {
      hello.identity
    };
    // A subscriber tells every new publisher what it wants.
    if matches!(self.kind, ZmqType::Sub | ZmqType::XSub) {
      let mut out = Vec::new();
      for topic in &inner.subscriptions {
        out.extend(encode_message(&[subscription_message(true, topic)]));
      }
      if !out.is_empty() && (&**stream).write_all(&out).is_err() {
        return None;
      }
    }
    inner.peers.push(Peer {
      id,
      stream: stream.clone(),
      routing_id,
      subscriptions: Vec::new(),
    });
    // Whatever was sent while nobody was there goes to the first peer.
    if matches!(
      self.kind,
      ZmqType::Pair | ZmqType::Push | ZmqType::Dealer | ZmqType::Req
    ) {
      while let Some(message) = inner.outbox.pop_front() {
        if (&**stream).write_all(&encode_message(&message)).is_err() {
          break;
        }
      }
    }
    Some(id)
  }

  fn unregister(&self, id: u64) {
    let mut inner = self.lock();
    let Some(index) = inner.peers.iter().position(|p| p.id == id) else {
      return;
    };
    let peer = inner.peers.remove(index);
    // An XPUB hears about the subscriptions that left with the peer.
    if self.kind == ZmqType::XPub {
      for topic in peer.subscriptions {
        if !inner.peers.iter().any(|p| p.subscriptions.contains(&topic)) {
          let message = vec![subscription_message(false, &topic)];
          self.enqueue(&mut inner, id, Vec::new(), message);
        }
      }
    }
  }

  /// Handle a command frame. `false` to drop the connection.
  fn command(&self, peer: u64, stream: &TcpStream, body: &[u8]) -> bool {
    let Some((name, data)) = parse_command(body) else {
      return false;
    };
    match name {
      b"PING" => {
        // PONG echoes the ping's context, which follows its 2-byte TTL.
        let context = data.get(2..).unwrap_or(&[]);
        let _inner = self.lock();
        (&*stream)
          .write_all(&encode_command("PONG", context))
          .is_ok()
      }
      b"SUBSCRIBE" | b"CANCEL" => {
        if matches!(self.kind, ZmqType::Pub | ZmqType::XPub) {
          let mut inner = self.lock();
          self.peer_subscription(&mut inner, peer, name == b"SUBSCRIBE", data);
        }
        true
      }
      b"ERROR" => false,
      _ => true,
    }
  }

  /// Route a complete incoming message according to the socket type.
  fn deliver(&self, peer: u64, mut frames: Vec<Vec<u8>>) {
    let mut inner = self.lock();
    match self.kind {
      ZmqType::Pub | ZmqType::XPub => {
        // All a publisher ever hears are (un)subscriptions.
        if let [only] = frames.as_slice()
          && let Some((&flag, topic)) = only.split_first()
          && flag <= 1
        {
          let topic = topic.to_vec();
          self.peer_subscription(&mut inner, peer, flag == 1, &topic);
        }
      }
      ZmqType::Sub => {
        let first = frames.first().map_or(&[][..], Vec::as_slice);
        if inner.subscriptions.iter().any(|t| first.starts_with(t)) {
          self.enqueue(&mut inner, peer, Vec::new(), frames);
        }
      }
      ZmqType::Req => {
        // A reply starts with the empty delimiter REQ put on the request.
        if frames.first().is_some_and(Vec::is_empty) {
          frames.remove(0);
          self.enqueue(&mut inner, peer, Vec::new(), frames);
        }
      }
      ZmqType::Rep => {
        if let Some(split) = frames.iter().position(Vec::is_empty) {
          let body = frames.split_off(split + 1);
          self.enqueue(&mut inner, peer, frames, body);
        }
      }
      ZmqType::Router => {
        if let Some(rid) = inner
          .peers
          .iter()
          .find(|p| p.id == peer)
          .map(|p| p.routing_id.clone())
        {
          frames.insert(0, rid);
          self.enqueue(&mut inner, peer, Vec::new(), frames);
        }
      }
      _ => self.enqueue(&mut inner, peer, Vec::new(), frames),
    }
  }

  /// A (un)subscription from a peer of a PUB or XPUB. An XPUB passes on the
  /// first subscription to a topic and the last cancellation of one, the
  /// way libzmq does without `ZMQ_XPUB_VERBOSE`.
  fn peer_subscription(
    &self,
    inner: &mut Inner,
    peer: u64,
    subscribe: bool,
    topic: &[u8],
  ) {
    let before = inner
      .peers
      .iter()
      .any(|p| p.subscriptions.iter().any(|t| t == topic));
    let Some(p) = inner.peers.iter_mut().find(|p| p.id == peer) else {
      return;
    };
    if subscribe {
      p.subscriptions.push(topic.to_vec());
    } else if let Some(i) = p.subscriptions.iter().position(|t| t == topic) {
      p.subscriptions.remove(i);
    } else {
      return;
    }
    let after = inner
      .peers
      .iter()
      .any(|p| p.subscriptions.iter().any(|t| t == topic));
    if self.kind == ZmqType::XPub && before != after {
      let message = vec![subscription_message(subscribe, topic)];
      self.enqueue(inner, peer, Vec::new(), message);
    }
  }

  fn enqueue(
    &self,
    inner: &mut Inner,
    peer: u64,
    envelope: Vec<Vec<u8>>,
    frames: Vec<Vec<u8>>,
  ) {
    if let Some(callback) = &inner.listener {
      if self.kind == ZmqType::Rep {
        inner.reply_to = Some((peer, envelope));
      }
      let last = frames.len().saturating_sub(1);
      for (i, frame) in frames.into_iter().enumerate() {
        callback(frame, i == last);
      }
      return;
    }
    inner.incoming.push_back(Incoming {
      peer,
      envelope,
      frames,
    });
    self.arrived.notify_all();
  }

  // -------------------------------------------------------------------------
  // The evaluator's side
  // -------------------------------------------------------------------------

  /// Send one message. Never blocks on a missing peer: PUB drops what no
  /// subscriber wants, REP and ROUTER drop what has no addressee, and the
  /// load-balancing types queue it for the first peer to arrive.
  pub fn send(&self, mut frames: Vec<Vec<u8>>) {
    let mut inner = self.lock();
    match self.kind {
      ZmqType::Pub | ZmqType::XPub => {
        let first = frames.first().cloned().unwrap_or_default();
        let wire = encode_message(&frames);
        for peer in &inner.peers {
          if peer.subscriptions.iter().any(|t| first.starts_with(t)) {
            let _ = (&*peer.stream).write_all(&wire);
          }
        }
      }
      ZmqType::XSub => {
        if let [only] = frames.as_slice()
          && let Some((&flag, topic)) = only.split_first()
          && flag <= 1
        {
          let topic = topic.to_vec();
          update_subscriptions(&mut inner.subscriptions, flag == 1, &topic);
        }
        let wire = encode_message(&frames);
        for peer in &inner.peers {
          let _ = (&*peer.stream).write_all(&wire);
        }
      }
      ZmqType::Rep => {
        if let Some((peer, mut envelope)) = inner.reply_to.take()
          && let Some(p) = inner.peers.iter().find(|p| p.id == peer)
        {
          envelope.append(&mut frames);
          let _ = (&*p.stream).write_all(&encode_message(&envelope));
        }
      }
      ZmqType::Router => {
        if frames.len() < 2 {
          return;
        }
        let rid = frames.remove(0);
        if let Some(p) = inner.peers.iter().find(|p| p.routing_id == rid) {
          let _ = (&*p.stream).write_all(&encode_message(&frames));
        }
      }
      ZmqType::Sub | ZmqType::Pull => {}
      ZmqType::Pair | ZmqType::Push | ZmqType::Dealer | ZmqType::Req => {
        if self.kind == ZmqType::Req {
          frames.insert(0, Vec::new());
        }
        if inner.peers.is_empty() {
          inner.outbox.push_back(frames);
          return;
        }
        let index = inner.round_robin % inner.peers.len();
        inner.round_robin = inner.round_robin.wrapping_add(1);
        let _ =
          (&*inner.peers[index].stream).write_all(&encode_message(&frames));
      }
    }
  }

  /// Take the next message, or the next part of one. `multipart` joins all
  /// remaining parts of the message into one byte string, the way
  /// `SocketReadMessage` reads by default.
  pub fn try_receive(&self, multipart: bool) -> Option<Vec<u8>> {
    let mut inner = self.lock();
    if inner.partial.is_empty() {
      let message = inner.incoming.pop_front()?;
      if self.kind == ZmqType::Rep {
        inner.reply_to = Some((message.peer, message.envelope));
      }
      inner.partial.extend(message.frames);
      // An empty message is still a message: one empty part.
      if inner.partial.is_empty() {
        return Some(Vec::new());
      }
    }
    if multipart {
      Some(inner.partial.drain(..).flatten().collect())
    } else {
      inner.partial.pop_front()
    }
  }

  /// Whether a receive would find something.
  pub fn has_input(&self) -> bool {
    let inner = self.lock();
    !inner.partial.is_empty() || !inner.incoming.is_empty()
  }

  /// Whether more parts of the message being read are waiting
  /// (`ZMQ_RCVMORE`).
  pub fn receive_more(&self) -> bool {
    !self.lock().partial.is_empty()
  }

  /// Wait up to `timeout` for something to receive.
  pub fn wait_input(&self, timeout: Duration) -> bool {
    let inner = self.lock();
    if !inner.partial.is_empty() || !inner.incoming.is_empty() {
      return true;
    }
    let (inner, _) = self
      .arrived
      .wait_timeout_while(inner, timeout, |i| {
        i.partial.is_empty() && i.incoming.is_empty()
      })
      .unwrap_or_else(std::sync::PoisonError::into_inner);
    !inner.partial.is_empty() || !inner.incoming.is_empty()
  }

  /// Whether a send would go somewhere right now (`ZMQ_POLLOUT`).
  pub fn writable(&self) -> bool {
    let inner = self.lock();
    match self.kind {
      ZmqType::Pub | ZmqType::XPub | ZmqType::XSub | ZmqType::Router => true,
      ZmqType::Sub | ZmqType::Pull => false,
      ZmqType::Rep => inner.reply_to.is_some(),
      ZmqType::Pair | ZmqType::Push | ZmqType::Dealer | ZmqType::Req => {
        self.connecting || !inner.peers.is_empty()
      }
    }
  }

  /// Subscribe (or unsubscribe) to messages starting with `topic`. Only
  /// meaningful on SUB, where libzmq also accepts it.
  pub fn subscribe(&self, subscribe: bool, topic: &[u8]) {
    if self.kind != ZmqType::Sub {
      return;
    }
    let mut inner = self.lock();
    if !update_subscriptions(&mut inner.subscriptions, subscribe, topic) {
      return;
    }
    let wire = encode_message(&[subscription_message(subscribe, topic)]);
    for peer in &inner.peers {
      let _ = (&*peer.stream).write_all(&wire);
    }
  }

  /// The routing id announced to peers that connect from now on.
  pub fn set_identity(&self, identity: Vec<u8>) {
    self.lock().identity = identity;
  }

  /// Hand every incoming frame to `callback` instead of queueing it —
  /// including the ones already queued.
  pub fn set_listener(&self, callback: FrameCallback) {
    let mut inner = self.lock();
    let partial: Vec<Vec<u8>> = inner.partial.drain(..).collect();
    let last = partial.len().saturating_sub(1);
    for (i, frame) in partial.into_iter().enumerate() {
      callback(frame, i == last);
    }
    while let Some(message) = inner.incoming.pop_front() {
      if self.kind == ZmqType::Rep {
        inner.reply_to = Some((message.peer, message.envelope));
      }
      let last = message.frames.len().saturating_sub(1);
      for (i, frame) in message.frames.into_iter().enumerate() {
        callback(frame, i == last);
      }
    }
    inner.listener = Some(callback);
  }

  pub fn clear_listener(&self) {
    self.lock().listener = None;
  }

  /// The descriptor of the current connection, for `ZMQ_FD` on a
  /// connecting socket.
  pub fn peer_fd(&self) -> Option<i128> {
    self.lock().peers.first().map(|p| raw_fd(&*p.stream))
  }

  /// Stop the background threads and drop every connection.
  pub fn close(&self) {
    self.stop.store(true, Ordering::Relaxed);
    let mut inner = self.lock();
    for peer in inner.peers.drain(..) {
      let _ = peer.stream.shutdown(std::net::Shutdown::Both);
    }
    inner.listener = None;
    inner.outbox.clear();
    self.arrived.notify_all();
  }
}

/// The message form of a (un)subscription: a flag byte, then the topic.
fn subscription_message(subscribe: bool, topic: &[u8]) -> Vec<u8> {
  let mut message = Vec::with_capacity(topic.len() + 1);
  message.push(u8::from(subscribe));
  message.extend_from_slice(topic);
  message
}

/// Add or remove one subscription. Subscriptions count: subscribing twice
/// needs unsubscribing twice. `false` for an unsubscribe of nothing.
fn update_subscriptions(
  subscriptions: &mut Vec<Vec<u8>>,
  subscribe: bool,
  topic: &[u8],
) -> bool {
  if subscribe {
    subscriptions.push(topic.to_vec());
    true
  } else if let Some(i) = subscriptions.iter().position(|t| t == topic) {
    subscriptions.remove(i);
    true
  } else {
    false
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn frames_round_trip_short_and_long() {
    let long = vec![7u8; 300];
    let wire = encode_message(&[b"ab".to_vec(), long.clone()]);
    let mut reader = FrameReader { buf: wire };
    let first = reader.next_frame().unwrap().unwrap();
    assert!(first.more && !first.command);
    assert_eq!(first.body, b"ab");
    let second = reader.next_frame().unwrap().unwrap();
    assert!(!second.more);
    assert_eq!(second.body, long);
    assert!(reader.next_frame().unwrap().is_none());
  }

  #[test]
  fn partial_frames_wait_for_the_rest() {
    let wire = encode_message(&[b"hello".to_vec()]);
    let mut reader = FrameReader {
      buf: wire[..3].to_vec(),
    };
    assert!(reader.next_frame().unwrap().is_none());
    reader.buf.extend_from_slice(&wire[3..]);
    assert_eq!(reader.next_frame().unwrap().unwrap().body, b"hello");
  }

  #[test]
  fn ready_properties_parse() {
    let wire = ready_command(ZmqType::Dealer, b"me");
    let mut reader = FrameReader { buf: wire };
    let frame = reader.next_frame().unwrap().unwrap();
    assert!(frame.command);
    let (name, data) = parse_command(&frame.body).unwrap();
    assert_eq!(name, b"READY");
    let props = parse_properties(data).unwrap();
    assert_eq!(props[0], ("Socket-Type".to_string(), b"DEALER".to_vec()));
    assert_eq!(props[1], ("Identity".to_string(), b"me".to_vec()));
  }

  /// Whatever a peer sends, parsing it must not panic: a long frame whose
  /// announced length would overflow, truncated commands, property lists
  /// whose lengths run past the end.
  #[test]
  fn malformed_peer_input_never_panics() {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
      state ^= state << 13;
      state ^= state >> 7;
      state ^= state << 17;
      state
    };
    let mut reader = FrameReader {
      buf: vec![FLAG_LONG, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 1],
    };
    assert!(reader.next_frame().is_err());
    for _ in 0..20_000 {
      let len = (next() % 40) as usize;
      let bytes: Vec<u8> = (0..len).map(|_| next() as u8).collect();
      let mut reader = FrameReader { buf: bytes.clone() };
      while let Ok(Some(frame)) = reader.next_frame() {
        if let Some((_, data)) = parse_command(&frame.body) {
          let _ = parse_properties(data);
        }
      }
      let _ = parse_command(&bytes);
      let _ = parse_properties(&bytes);
    }
  }

  #[test]
  fn socket_type_compatibility() {
    assert!(ZmqType::Sub.accepts(b"PUB"));
    assert!(ZmqType::Sub.accepts(b"XPUB"));
    assert!(!ZmqType::Sub.accepts(b"PUSH"));
    assert!(ZmqType::Req.accepts(b"ROUTER"));
    assert!(!ZmqType::Pair.accepts(b"PUB"));
  }

  fn wait_until(cond: impl Fn() -> bool) -> bool {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
      if cond() {
        return true;
      }
      std::thread::sleep(Duration::from_millis(5));
    }
    false
  }

  #[test]
  fn push_pull_and_queue_before_connect() {
    let pull = ZmqSocket::bind(ZmqType::Pull, "127.0.0.1", 0).unwrap();
    let push =
      ZmqSocket::connect(ZmqType::Push, "127.0.0.1", pull.local.port());
    push.send(vec![b"one".to_vec()]);
    push.send(vec![b"a".to_vec(), b"b".to_vec()]);
    assert!(pull.socket.wait_input(Duration::from_secs(5)));
    assert_eq!(pull.socket.try_receive(true).unwrap(), b"one");
    assert!(wait_until(|| pull.socket.has_input()));
    assert_eq!(pull.socket.try_receive(false).unwrap(), b"a");
    assert!(pull.socket.receive_more());
    assert_eq!(pull.socket.try_receive(false).unwrap(), b"b");
    assert!(!pull.socket.receive_more());
    push.close();
    pull.socket.close();
  }

  #[test]
  fn pub_sub_filters_by_prefix() {
    let publisher = ZmqSocket::bind(ZmqType::Pub, "127.0.0.1", 0).unwrap();
    let sub =
      ZmqSocket::connect(ZmqType::Sub, "127.0.0.1", publisher.local.port());
    sub.subscribe(true, b"lmax");
    // The subscription reaches the publisher asynchronously.
    assert!(wait_until(|| {
      publisher
        .socket
        .lock()
        .peers
        .iter()
        .any(|p| !p.subscriptions.is_empty())
    }));
    publisher.socket.send(vec![b"other".to_vec()]);
    publisher.socket.send(vec![b"lmax 1".to_vec()]);
    assert!(sub.wait_input(Duration::from_secs(5)));
    assert_eq!(sub.try_receive(true).unwrap(), b"lmax 1");
    sub.close();
    publisher.socket.close();
  }

  #[test]
  fn req_rep_round_trip() {
    let rep = ZmqSocket::bind(ZmqType::Rep, "127.0.0.1", 0).unwrap();
    let req = ZmqSocket::connect(ZmqType::Req, "127.0.0.1", rep.local.port());
    req.send(vec![b"ping".to_vec()]);
    assert!(rep.socket.wait_input(Duration::from_secs(5)));
    assert_eq!(rep.socket.try_receive(true).unwrap(), b"ping");
    rep.socket.send(vec![b"pong".to_vec()]);
    assert!(req.wait_input(Duration::from_secs(5)));
    assert_eq!(req.try_receive(true).unwrap(), b"pong");
    req.close();
    rep.socket.close();
  }
}
