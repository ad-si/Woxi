use jupyter_protocol::{
  ConnectionInfo, ExecuteResult, ExecutionCount, JupyterMessage,
  JupyterMessageContent, Status, connection_info::Transport,
};
use std::path::Path;
use uuid::Uuid;

pub fn run(connection_file: Option<&std::path::Path>) -> anyhow::Result<()> {
  tokio::runtime::Runtime::new()?
    .block_on(async { run_impl(connection_file).await })
}

async fn run_impl(connection_file: Option<&Path>) -> anyhow::Result<()> {
  let session_id = Uuid::new_v4().to_string();
  println!("Starting kernel with session ID: {}", session_id);

  // Create execution counter
  let mut execution_count: usize = 0;

  // If connection file is provided, load it
  let connection_info = if let Some(file_path) = connection_file {
    println!("Loading connection info from: {}", file_path.display());
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

    println!(
      "Started kernel with connection file: {}",
      connection_path.display()
    );
    println!(
      "Connect to this kernel using: jupyter console --existing {}",
      connection_path.display()
    );

    connection_info
  };

  // Set up ZMQ sockets
  let mut shell_socket =
    runtimelib::create_kernel_shell_connection(&connection_info, &session_id)
      .await?;
  let mut iopub_socket =
    runtimelib::create_kernel_iopub_connection(&connection_info, &session_id)
      .await?;
  let mut control_socket =
    runtimelib::create_kernel_control_connection(&connection_info, &session_id)
      .await?;
  let mut stdin_socket =
    runtimelib::create_kernel_stdin_connection(&connection_info, &session_id)
      .await?;
  let mut hb_socket =
    runtimelib::create_kernel_heartbeat_connection(&connection_info).await?;

  // Start heartbeat thread
  let hb_handle = tokio::spawn(async move {
    while let Ok(()) = hb_socket.single_heartbeat().await {}
  });

  // Send initial status: idle
  iopub_socket
    .send(JupyterMessage::new(Status::idle(), None))
    .await?;

  // Main loop
  loop {
    // Use tokio::select to handle multiple channels
    tokio::select! {
        // Shell messages (execute_request, kernel_info_request, etc.)
        shell_result = shell_socket.read() => {
            match shell_result {
                Ok(request) => {
                    match &request.content {
                        JupyterMessageContent::ExecuteRequest(execute_request) => {
                            // Increment the execution count for each new request
                            execution_count += 1;
                            handle_execute_request(&mut shell_socket, &mut iopub_socket, &request, execute_request, execution_count).await?;
                        },
                        JupyterMessageContent::KernelInfoRequest(_) => {
                            // Send busy status before handling kernel info
                            iopub_socket.send(Status::busy().as_child_of(&request)).await?;

                            // Handle the kernel info request
                            handle_kernel_info_request(&mut shell_socket, &request).await?;

                            // Send idle status after handling kernel info
                            iopub_socket.send(Status::idle().as_child_of(&request)).await?;
                        },
                        JupyterMessageContent::IsCompleteRequest(is_complete_request) => {
                            handle_is_complete_request(&mut shell_socket, &request, is_complete_request).await?;
                        },
                        JupyterMessageContent::ShutdownRequest(shutdown_request) => {
                            println!("Received shutdown request on shell channel");

                            // Send status: busy
                            iopub_socket.send(Status::busy().as_child_of(&request)).await?;

                            // Send the shutdown reply
                            let shutdown_reply = jupyter_protocol::ShutdownReply {
                                restart: shutdown_request.restart,
                                status: jupyter_protocol::ReplyStatus::Ok,
                                error: None,
                            };
                            shell_socket.send(shutdown_reply.as_child_of(&request)).await?;

                            // Send status: idle before exiting
                            iopub_socket.send(Status::idle().as_child_of(&request)).await?;

                            println!("Shutdown reply sent, exiting");
                            break;
                        },
                        _ => {
                            println!("Unhandled shell message: {:?}", request.header.msg_type);
                        }
                    }
                },
                Err(e) => {
                    eprintln!("Error reading from shell socket: {}", e);
                    break;
                }
            }
        },

        // Control messages
        control_result = control_socket.read() => {
            match control_result {
                Ok(request) => {
                    match &request.content {
                        JupyterMessageContent::ShutdownRequest(shutdown_request) => {
                            println!("Received shutdown request on control channel");

                            // Send status: busy
                            iopub_socket.send(Status::busy().as_child_of(&request)).await?;

                            // Send the shutdown reply
                            let shutdown_reply = jupyter_protocol::ShutdownReply {
                                restart: shutdown_request.restart,
                                status: jupyter_protocol::ReplyStatus::Ok,
                                error: None,
                            };
                            control_socket.send(shutdown_reply.as_child_of(&request)).await?;

                            // Send status: idle before exiting
                            iopub_socket.send(Status::idle().as_child_of(&request)).await?;

                            println!("Shutdown reply sent from control channel, exiting");
                            break;
                        },
                        JupyterMessageContent::KernelInfoRequest(_) => {
                            // Send busy status before handling kernel info
                            iopub_socket.send(Status::busy().as_child_of(&request)).await?;

                            // Handle the kernel info request
                            handle_kernel_info_request(&mut control_socket, &request).await?;

                            // Send idle status after handling kernel info
                            iopub_socket.send(Status::idle().as_child_of(&request)).await?;
                        },
                        _ => {
                            println!("Unhandled control message: {:?}", request.header.msg_type);
                        }
                    }
                },
                Err(e) => {
                    eprintln!("Error reading from control socket: {}", e);
                    break;
                }
            }
        },

        // Stdin messages (not handling for simplicity)
        stdin_result = stdin_socket.read() => {
            match stdin_result {
                Ok(request) => {
                    println!("Received stdin message: {:?}", request.header.msg_type);
                },
                Err(e) => {
                    eprintln!("Error reading from stdin socket: {}", e);
                    break;
                }
            }
        }
    }
  }

  // Print a message before exiting
  println!("Kernel is shutting down...");

  // Explicitly abort the heartbeat thread to ensure it stops
  hb_handle.abort();

  // Wait for the heartbeat thread to exit with a timeout
  if let Ok(result) =
    tokio::time::timeout(tokio::time::Duration::from_secs(5), hb_handle).await
  {
    match result {
      Ok(_) => println!("Heartbeat thread exited cleanly"),
      Err(e) => println!("Heartbeat thread exited with error: {}", e),
    }
  } else {
    println!("Heartbeat thread did not exit within timeout");
  }

  Ok(())
}

// Handle execute_request messages
async fn handle_execute_request(
  shell_socket: &mut runtimelib::KernelShellConnection,
  iopub_socket: &mut runtimelib::KernelIoPubConnection,
  request: &JupyterMessage,
  execute_request: &jupyter_protocol::ExecuteRequest,
  execution_count: usize,
) -> anyhow::Result<()> {
  println!(
    "Executing: {} (cell {})",
    execute_request.code, execution_count
  );

  // 1. Send status: busy
  iopub_socket.send(Status::busy().as_child_of(request)).await?;

  // 2. Send execute_input
  let execute_input = jupyter_protocol::ExecuteInput {
    code: execute_request.code.clone(),
    execution_count: ExecutionCount(execution_count),
  };
  iopub_socket.send(execute_input.as_child_of(request)).await?;

  // 3. Execute the code using woxi interpreter with the new stdout capture mode
  let (execution_result, graphics) =
    match woxi::interpret_with_stdout(&execute_request.code) {
      Ok(result) => {
        // Send captured stdout if there's any content
        if !result.stdout.is_empty() {
          iopub_socket
            .send(jupyter_protocol::StreamContent::stdout(&result.stdout).as_child_of(request))
            .await?;
        }

        // Return the result and any graphical output
        (result.result, result.graphics)
      }
      Err(e) => {
        // Send stderr stream message with error
        iopub_socket
          .send(jupyter_protocol::StreamContent::stderr(&format!("Error: {0}\n", e)).as_child_of(request))
          .await?;

        // Return error text for execute_result
        (format!("Error: {0}", e), None)
      }
    };

  // 4. Send execute_result only if it's not "Null" (from Print statements)
  if execution_result != "Null" {
    let mut media = jupyter_protocol::media::Media::default();
    media
      .content
      .push(jupyter_protocol::MediaType::Plain(execution_result));

    // If graphical output was produced, add SVG media type
    if let Some(svg) = graphics {
      media.content.push(jupyter_protocol::MediaType::Svg(svg));
    }

    let execute_result = ExecuteResult {
      execution_count: ExecutionCount(execution_count),
      data: media,
      metadata: Default::default(),
      transient: None,
    };
    iopub_socket
      .send(execute_result.as_child_of(request))
      .await?;
  }

  // 5. Send status: idle
  iopub_socket.send(Status::idle().as_child_of(request)).await?;

  // 6. Send execute_reply
  let execute_reply = jupyter_protocol::ExecuteReply {
    status: jupyter_protocol::ReplyStatus::Ok,
    execution_count: ExecutionCount(execution_count),
    payload: vec![],
    user_expressions: Default::default(),
    error: None,
  };
  shell_socket
    .send(execute_reply.as_child_of(request))
    .await?;

  Ok(())
}

async fn handle_is_complete_request(
  shell_socket: &mut runtimelib::KernelShellConnection,
  request: &JupyterMessage,
  is_complete_request: &jupyter_protocol::IsCompleteRequest,
) -> anyhow::Result<()> {
  println!(
    "Handling is_complete_request for code: {}",
    is_complete_request.code
  );

  // For simplicity, always say the code is complete
  let is_complete_reply = jupyter_protocol::IsCompleteReply {
    status: jupyter_protocol::IsCompleteReplyStatus::Complete,
    indent: "".to_string(),
  };
  shell_socket
    .send(is_complete_reply.as_child_of(request))
    .await?;

  Ok(())
}

// Handle kernel_info_request messages
async fn handle_kernel_info_request(
  shell_socket: &mut runtimelib::KernelShellConnection,
  request: &JupyterMessage,
) -> anyhow::Result<()> {
  println!(
    "Handling kernel info request from: {} (msg_id: {})",
    request.header.session, request.header.msg_id
  );

  // Create kernel info reply
  let language_info = jupyter_protocol::LanguageInfo {
    name: "wolfram".to_string(),
    version: "0.1.0".to_string(),
    mimetype: Some("application/vnd.wolfram.mathematica".to_string()),
    file_extension: Some(".wls".to_string()),
    pygments_lexer: Some("mathematica".to_string()),
    codemirror_mode: Some(jupyter_protocol::CodeMirrorMode::Simple(
      "mathematica".to_string(),
    )),
    nbconvert_exporter: Some("text".to_string()),
  };

  let kernel_info_reply = jupyter_protocol::KernelInfoReply {
    protocol_version: "5.3.0".to_string(),
    implementation: "woxi".to_string(),
    implementation_version: "0.1.0".to_string(),
    language_info,
    banner: "Woxi Jupyter Kernel - Wolfram Language Interpreter".to_string(),
    help_links: vec![jupyter_protocol::HelpLink {
      text: "Woxi Documentation".to_string(),
      url: "https://github.com/ad-si/Woxi".to_string(),
    }],
    status: jupyter_protocol::ReplyStatus::Ok,
    debugger: false,
    error: None,
  };

  println!(
    "Sending kernel info reply to msg_id: {} on identities: {:?}",
    request.header.msg_id, request.zmq_identities
  );
  shell_socket
    .send(kernel_info_reply.as_child_of(request))
    .await?;

  Ok(())
}
