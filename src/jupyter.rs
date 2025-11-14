use jupyter_protocol::{
  connection_info::Transport, ConnectionInfo, ExecuteResult, ExecutionCount,
  JupyterMessage, JupyterMessageContent, Status,
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

  // Start heartbeat thread with more aggressive polling
  let hb_handle = tokio::spawn(async move {
    let mut interval =
      tokio::time::interval(tokio::time::Duration::from_millis(500));
    loop {
      interval.tick().await; // Wait for the next interval tick

      // Check heartbeat more often
      match hb_socket.single_heartbeat().await {
        Ok(_) => {
          // Heartbeat successfully processed
        }
        Err(e) => {
          eprintln!("Heartbeat error: {}", e);
          break;
        }
      }
    }
  });

  // Send initial status: idle
  let status = Status::idle();
  // For initial messages with no parent, use our kernel's session ID
  let dummy_parent = jupyter_protocol::Header {
    msg_id: "".to_string(),
    session: session_id.clone(),
    username: "woxi".to_string(),
    date: chrono::Utc::now(),
    msg_type: "".to_string(),
    version: "5.3.0".to_string(),
  };
  let status_message = create_message(&status, Some(&dummy_parent), vec![]);
  iopub_socket.send(status_message).await?;

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
                            let busy_status = Status::busy();
                            let busy_msg = create_message(&busy_status, Some(&request.header), vec![]);
                            iopub_socket.send(busy_msg).await?;

                            // Handle the kernel info request
                            handle_kernel_info_request(&mut shell_socket, &request).await?;

                            // Send idle status after handling kernel info
                            let idle_status = Status::idle();
                            let idle_msg = create_message(&idle_status, Some(&request.header), vec![]);
                            iopub_socket.send(idle_msg).await?;
                        },
                        JupyterMessageContent::IsCompleteRequest(is_complete_request) => {
                            handle_is_complete_request(&mut shell_socket, &request, is_complete_request).await?;
                        },
                        JupyterMessageContent::ShutdownRequest(shutdown_request) => {
                            println!("Received shutdown request on shell channel");

                            // Send status: busy
                            let busy_status = Status::busy();
                            let busy_msg = create_message(&busy_status, Some(&request.header), vec![]);
                            iopub_socket.send(busy_msg).await?;

                            // Create shutdown reply
                            let shutdown_reply = jupyter_protocol::ShutdownReply {
                                restart: shutdown_request.restart,
                                status: jupyter_protocol::ReplyStatus::Ok,
                                error: None,
                            };

                            // Send the reply
                            let reply_msg = create_message(
                                &shutdown_reply,
                                Some(&request.header),
                                request.zmq_identities.iter().map(|b| b.to_vec()).collect(),
                            );
                            shell_socket.send(reply_msg).await?;

                            // Send status: idle before exiting
                            let idle_status = Status::idle();
                            let idle_msg = create_message(&idle_status, Some(&request.header), vec![]);
                            iopub_socket.send(idle_msg).await?;

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
                            let busy_status = Status::busy();
                            let busy_msg = create_message(&busy_status, Some(&request.header), vec![]);
                            iopub_socket.send(busy_msg).await?;

                            // Create shutdown reply
                            let shutdown_reply = jupyter_protocol::ShutdownReply {
                                restart: shutdown_request.restart,
                                status: jupyter_protocol::ReplyStatus::Ok,
                                error: None,
                            };

                            // Send the reply
                            let reply_msg = create_message(
                                &shutdown_reply,
                                Some(&request.header),
                                request.zmq_identities.iter().map(|b| b.to_vec()).collect(),
                            );
                            control_socket.send(reply_msg).await?;

                            // Send status: idle before exiting
                            let idle_status = Status::idle();
                            let idle_msg = create_message(&idle_status, Some(&request.header), vec![]);
                            iopub_socket.send(idle_msg).await?;

                            println!("Shutdown reply sent from control channel, exiting");
                            break;
                        },
                        JupyterMessageContent::KernelInfoRequest(_) => {
                            // Send busy status before handling kernel info
                            let busy_status = Status::busy();
                            let busy_msg = create_message(&busy_status, Some(&request.header), vec![]);
                            iopub_socket.send(busy_msg).await?;

                            // Handle the kernel info request
                            handle_kernel_info_request(&mut control_socket, &request).await?;

                            // Send idle status after handling kernel info
                            let idle_status = Status::idle();
                            let idle_msg = create_message(&idle_status, Some(&request.header), vec![]);
                            iopub_socket.send(idle_msg).await?;
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

// Helper function to create a jupyter message
fn create_message<T>(
  content: &T,
  parent: Option<&jupyter_protocol::Header>,
  identities: Vec<Vec<u8>>,
) -> JupyterMessage
where
  T: Clone,
  JupyterMessageContent: From<T>,
{
  let content_clone = content.clone();
  let jupyter_content: JupyterMessageContent = content_clone.into();

  let now = chrono::Utc::now();
  let header = jupyter_protocol::Header {
    msg_id: Uuid::new_v4().to_string(),
    session: parent
      .map_or_else(|| Uuid::new_v4().to_string(), |h| h.session.clone()),
    username: "woxi".to_string(),
    date: now,
    msg_type: get_message_type(&jupyter_content),
    version: "5.3.0".to_string(),
  };

  JupyterMessage {
    zmq_identities: identities.into_iter().map(Into::into).collect(),
    header,
    parent_header: parent.cloned(),
    metadata: Default::default(),
    content: jupyter_content,
    buffers: vec![],
    channel: None,
  }
}

// Helper function to get message type
fn get_message_type(content: &JupyterMessageContent) -> String {
  match content {
    JupyterMessageContent::Status(_) => "status".to_string(),
    JupyterMessageContent::ExecuteInput(_) => "execute_input".to_string(),
    JupyterMessageContent::ExecuteResult(_) => "execute_result".to_string(),
    JupyterMessageContent::StreamContent(_) => "stream".to_string(),
    JupyterMessageContent::ExecuteReply(_) => "execute_reply".to_string(),
    JupyterMessageContent::KernelInfoReply(_) => {
      "kernel_info_reply".to_string()
    }
    JupyterMessageContent::IsCompleteReply(_) => {
      "is_complete_reply".to_string()
    }
    _ => "unknown".to_string(),
  }
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
  let busy_status = Status::busy();
  let busy_msg = create_message(&busy_status, Some(&request.header), vec![]);
  iopub_socket.send(busy_msg).await?;

  // 2. Send execute_input
  let execute_input = jupyter_protocol::ExecuteInput {
    code: execute_request.code.clone(),
    execution_count: ExecutionCount(execution_count),
  };
  let input_msg = create_message(&execute_input, Some(&request.header), vec![]);
  iopub_socket.send(input_msg).await?;

  // 3. Execute the code using woxi interpreter with the new stdout capture mode

  // Use the new interpret_with_stdout function to get both stdout and the result
  let execution_result =
    match woxi::interpret_with_stdout(&execute_request.code) {
      Ok(result) => {
        // Send captured stdout if there's any content
        if !result.stdout.is_empty() {
          let stream_content = jupyter_protocol::StreamContent {
            name: jupyter_protocol::Stdio::Stdout,
            text: result.stdout,
          };
          let stream_msg =
            create_message(&stream_content, Some(&request.header), vec![]);
          iopub_socket.send(stream_msg).await?;
        }

        // Return the result
        result.result
      }
      Err(e) => {
        // Send stderr stream message with error
        let error_content = jupyter_protocol::StreamContent {
          name: jupyter_protocol::Stdio::Stderr,
          text: format!("Error: {0}\n", e),
        };
        let error_msg =
          create_message(&error_content, Some(&request.header), vec![]);
        iopub_socket.send(error_msg).await?;

        // Return error text for execute_result
        format!("Error: {0}", e)
      }
    };

  // 4. Send execute_result only if it's not "Null" (from Print statements)
  if execution_result != "Null" {
    let mut media = jupyter_protocol::media::Media::default();
    media
      .content
      .push(jupyter_protocol::MediaType::Plain(execution_result));

    let execute_result = ExecuteResult {
      execution_count: ExecutionCount(execution_count),
      data: media,
      metadata: Default::default(),
      transient: None,
    };

    let result_msg =
      create_message(&execute_result, Some(&request.header), vec![]);
    iopub_socket.send(result_msg).await?;
  }

  // 5. Send status: idle
  let idle_status = Status::idle();
  let idle_msg = create_message(&idle_status, Some(&request.header), vec![]);
  iopub_socket.send(idle_msg).await?;

  // 6. Send execute_reply
  let execute_reply = jupyter_protocol::ExecuteReply {
    status: jupyter_protocol::ReplyStatus::Ok,
    execution_count: ExecutionCount(execution_count),
    payload: vec![],
    user_expressions: Default::default(),
    error: None,
  };

  let reply_msg = create_message(
    &execute_reply,
    Some(&request.header),
    request.zmq_identities.iter().map(|b| b.to_vec()).collect(),
  );
  shell_socket.send(reply_msg).await?;

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

  let reply_msg = create_message(
    &is_complete_reply,
    Some(&request.header),
    request.zmq_identities.iter().map(|b| b.to_vec()).collect(),
  );
  shell_socket.send(reply_msg).await?;

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
    name: "woxi".to_string(),
    version: "0.1.0".to_string(),
    mimetype: Some("text/x-mathematica".to_string()),
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

  // Note: KernelInfoReply needs to be boxed for JupyterMessageContent
  let boxed_reply = Box::new(kernel_info_reply);

  // Create message manually for KernelInfoReply since it's a special case
  let now = chrono::Utc::now();
  let header = jupyter_protocol::Header {
    msg_id: Uuid::new_v4().to_string(),
    session: request.header.session.clone(),
    username: "woxi".to_string(),
    date: now,
    msg_type: "kernel_info_reply".to_string(),
    version: "5.3.0".to_string(),
  };

  // Create a reply message with proper metadata
  let reply_msg = JupyterMessage {
    zmq_identities: request.zmq_identities.clone(),
    header,
    parent_header: Some(request.header.clone()),
    metadata: Default::default(), // Use default metadata
    content: JupyterMessageContent::KernelInfoReply(boxed_reply),
    buffers: vec![],
    channel: None, // No channel needed
  };

  println!(
    "Sending kernel info reply to msg_id: {} with session: {} on identities: {:?}",
    request.header.msg_id, reply_msg.header.session, request.zmq_identities
  );
  shell_socket.send(reply_msg).await?;

  Ok(())
}
