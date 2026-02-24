use anyhow::Result;
use jupyter_protocol::{
  CodeMirrorMode, ConnectionInfo, ExecuteResult, ExecutionCount, HelpLink, JupyterMessage,
  JupyterMessageContent, KernelInfoReply, LanguageInfo, ReplyStatus, ShutdownReply, Status,
  connection_info::Transport,
};
use log::{debug, error, info, trace, warn};
use runtimelib::{KernelIoPubConnection, RouterRecvConnection, RouterSendConnection};
use std::path::Path;
use uuid::Uuid;

pub fn run(connection_file: Option<&std::path::Path>) -> Result<()> {
  env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

  tokio::runtime::Runtime::new()?.block_on(async { run_impl(connection_file).await })
}

struct WoxiKernel {
  execution_count: ExecutionCount,
  iopub: KernelIoPubConnection,
  shell: RouterSendConnection,
}

impl WoxiKernel {
  pub async fn start(connection_info: &ConnectionInfo) -> Result<()> {
    let session_id = Uuid::new_v4().to_string();
    debug!("Starting kernel with session ID: {}", session_id);

    // Create all connections
    let mut heartbeat =
      runtimelib::create_kernel_heartbeat_connection(connection_info).await?;
    let shell_connection =
      runtimelib::create_kernel_shell_connection(connection_info, &session_id).await?;
    let (shell_writer, mut shell_reader) = shell_connection.split();
    let mut control_connection =
      runtimelib::create_kernel_control_connection(connection_info, &session_id).await?;
    let _stdin_connection =
      runtimelib::create_kernel_stdin_connection(connection_info, &session_id).await?;
    let iopub_connection =
      runtimelib::create_kernel_iopub_connection(connection_info, &session_id).await?;

    let mut kernel = Self {
      execution_count: ExecutionCount::default(),
      iopub: iopub_connection,
      shell: shell_writer,
    };

    // Send initial status: idle
    kernel
      .iopub
      .send(JupyterMessage::new(Status::idle(), None))
      .await?;

    // Heartbeat task
    let heartbeat_handle = tokio::spawn(async move {
      while let Ok(()) = heartbeat.single_heartbeat().await {}
    });

    // Control task
    let control_handle = tokio::spawn(async move {
      while let Ok(message) = control_connection.read().await {
        match &message.content {
          JupyterMessageContent::KernelInfoRequest(_) => {
            let reply = Self::kernel_info().as_child_of(&message);
            if let Err(err) = control_connection.send(reply).await {
              error!("Error on control: {}", err);
            }
          }
          JupyterMessageContent::ShutdownRequest(req) => {
            let reply = ShutdownReply {
              restart: req.restart,
              status: ReplyStatus::Ok,
              error: None,
            }
            .as_child_of(&message);
            let _ = control_connection.send(reply).await;
            std::process::exit(0);
          }
          _ => {}
        }
      }
    });

    // Shell task
    let shell_handle = tokio::spawn(async move {
      if let Err(err) = kernel.handle_shell(&mut shell_reader).await {
        error!("Shell error: {}", err);
      }
    });

    // Wait for all tasks
    tokio::select! {
      _ = heartbeat_handle => {}
      _ = control_handle => {}
      _ = shell_handle => {}
    }

    Ok(())
  }

  async fn handle_shell(&mut self, reader: &mut RouterRecvConnection) -> Result<()> {
    loop {
      let msg = reader.read().await?;
      if let Err(err) = self.handle_shell_message(&msg).await {
        error!("Error handling shell message: {}", err);
      }
    }
  }

  async fn handle_shell_message(&mut self, parent: &JupyterMessage) -> Result<()> {
    // Always send busy at the start
    self
      .iopub
      .send(Status::busy().as_child_of(parent))
      .await?;

    match &parent.content {
      JupyterMessageContent::ExecuteRequest(req) => {
        self.execution_count.0 += 1;
        self.execute(parent, req).await?;
      }
      JupyterMessageContent::KernelInfoRequest(_) => {
        self
          .shell
          .send(Self::kernel_info().as_child_of(parent))
          .await?;
      }
      JupyterMessageContent::IsCompleteRequest(req) => {
        trace!("is_complete_request: {}", req.code);
        let reply = jupyter_protocol::IsCompleteReply {
          status: jupyter_protocol::IsCompleteReplyStatus::Complete,
          indent: "".to_string(),
        };
        self.shell.send(reply.as_child_of(parent)).await?;
      }
      JupyterMessageContent::CommInfoRequest(_) => {
        self
          .shell
          .send(jupyter_protocol::CommInfoReply::default().as_child_of(parent))
          .await?;
      }
      JupyterMessageContent::HistoryRequest(_) => {
        self
          .shell
          .send(jupyter_protocol::HistoryReply::default().as_child_of(parent))
          .await?;
      }
      JupyterMessageContent::ShutdownRequest(req) => {
        info!("Shutdown request received");
        let reply = ShutdownReply {
          restart: req.restart,
          status: ReplyStatus::Ok,
          error: None,
        };
        self.shell.send(reply.as_child_of(parent)).await?;
        self
          .iopub
          .send(Status::idle().as_child_of(parent))
          .await?;
        std::process::exit(0);
      }
      _ => {
        warn!("Unhandled shell message: {:?}", parent.header.msg_type);
      }
    }

    // Always send idle at the end
    self
      .iopub
      .send(Status::idle().as_child_of(parent))
      .await?;

    Ok(())
  }

  async fn execute(
    &mut self,
    parent: &JupyterMessage,
    req: &jupyter_protocol::ExecuteRequest,
  ) -> Result<()> {
    debug!("Execute[{}]: {}", self.execution_count.0, req.code);

    // Send execute_input
    let execute_input = jupyter_protocol::ExecuteInput {
      code: req.code.clone(),
      execution_count: self.execution_count,
    };
    self
      .iopub
      .send(execute_input.as_child_of(parent))
      .await?;

    // Execute the code
    let (execution_result, graphics) = match woxi::interpret_with_stdout(&req.code) {
      Ok(result) => {
        if !result.stdout.is_empty() {
          self
            .iopub
            .send(
              jupyter_protocol::StreamContent::stdout(&result.stdout).as_child_of(parent),
            )
            .await?;
        }
        (result.result, result.graphics)
      }
      Err(e) => {
        self
          .iopub
          .send(
            jupyter_protocol::StreamContent::stderr(&format!("Error: {}\n", e))
              .as_child_of(parent),
          )
          .await?;
        (format!("Error: {}", e), None)
      }
    };

    // Send execute_result only if it's not "Null"
    if execution_result != "Null" {
      let mut media = jupyter_protocol::media::Media::default();
      media
        .content
        .push(jupyter_protocol::MediaType::Plain(execution_result));

      if let Some(svg) = graphics {
        media.content.push(jupyter_protocol::MediaType::Svg(svg));
      }

      let execute_result = ExecuteResult {
        execution_count: self.execution_count,
        data: media,
        metadata: Default::default(),
        transient: None,
      };
      self
        .iopub
        .send(execute_result.as_child_of(parent))
        .await?;
    }

    // Send execute_reply
    let execute_reply = jupyter_protocol::ExecuteReply {
      status: ReplyStatus::Ok,
      execution_count: self.execution_count,
      payload: vec![],
      user_expressions: Default::default(),
      error: None,
    };
    self.shell.send(execute_reply.as_child_of(parent)).await?;

    Ok(())
  }

  fn kernel_info() -> KernelInfoReply {
    KernelInfoReply {
      status: ReplyStatus::Ok,
      protocol_version: "5.3".to_string(),
      implementation: "woxi".to_string(),
      implementation_version: "0.1.0".to_string(),
      language_info: LanguageInfo {
        name: "wolfram".to_string(),
        version: "0.1.0".to_string(),
        mimetype: Some("application/vnd.wolfram.mathematica".to_string()),
        file_extension: Some(".wls".to_string()),
        pygments_lexer: Some("mathematica".to_string()),
        codemirror_mode: Some(CodeMirrorMode::Simple("mathematica".to_string())),
        nbconvert_exporter: Some("text".to_string()),
      },
      banner: "Woxi Jupyter Kernel - Wolfram Language Interpreter".to_string(),
      help_links: vec![HelpLink {
        text: "Woxi Documentation".to_string(),
        url: "https://github.com/ad-si/Woxi".to_string(),
      }],
      debugger: false,
      error: None,
    }
  }
}

async fn run_impl(connection_file: Option<&Path>) -> Result<()> {
  let connection_info = if let Some(file_path) = connection_file {
    debug!("Loading connection info from: {}", file_path.display());
    let content = tokio::fs::read_to_string(file_path).await?;
    serde_json::from_str(&content)?
  } else {
    // Create a new connection on localhost with random ports
    let ip = std::net::IpAddr::V4(std::net::Ipv4Addr::new(127, 0, 0, 1));
    let ports = runtimelib::peek_ports(ip, 5).await?;
    assert_eq!(ports.len(), 5);

    let connection_info = ConnectionInfo {
      transport: Transport::TCP,
      ip: ip.to_string(),
      stdin_port: ports[0],
      control_port: ports[1],
      hb_port: ports[2],
      shell_port: ports[3],
      iopub_port: ports[4],
      signature_scheme: "hmac-sha256".to_string(),
      key: Uuid::new_v4().to_string(),
      kernel_name: Some("woxi".to_string()),
    };

    // Write connection file for clients to connect
    let runtime_dir = runtimelib::dirs::runtime_dir();
    tokio::fs::create_dir_all(&runtime_dir).await?;
    let connection_path = runtime_dir.join("kernel-woxi.json");
    let content = serde_json::to_string(&connection_info)?;
    tokio::fs::write(&connection_path, content).await?;

    info!(
      "Started kernel with connection file: {}",
      connection_path.display()
    );
    info!(
      "Connect using: jupyter console --existing {}",
      connection_path.display()
    );

    connection_info
  };

  WoxiKernel::start(&connection_info).await
}
