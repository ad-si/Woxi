const http = require("http");
const { exec } = require("child_process");

const PORT = 3456;

const server = http.createServer((req, res) => {
  if (req.method === "POST" && req.url === "/exec") {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => {
      let cmd;
      try {
        cmd = JSON.parse(body).cmd;
      } catch {
        cmd = body.trim();
      }

      if (!cmd) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "Missing cmd" }));
        return;
      }

      exec(cmd, { timeout: 20000 }, (error, stdout, stderr) => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(
          JSON.stringify({
            exitCode: error ? error.code ?? 1 : 0,
            stdout,
            stderr,
          })
        );
      });
    });
    return;
  }

  res.writeHead(404, { "Content-Type": "text/plain" });
  res.end("POST /exec with {\"cmd\": \"...\"}\n");
});

server.listen(PORT, () => {
  console.log(`Listening on http://localhost:${PORT}`);
  console.log(`Usage: curl -X POST http://localhost:${PORT}/exec -d '{"cmd":"ls -la"}'`);
});
