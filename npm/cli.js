#!/usr/bin/env node
// Minimal CLI mirroring the native `woxi` binary:
//
//   woxi eval '<code>'   Evaluate an expression (pass `-` to read stdin)
//   woxi <file.wls>      Run a Wolfram Language script file
//
// Backed by the WebAssembly build, so it runs anywhere Node.js runs.

"use strict";

const fs = require("node:fs");
const woxi = require("./index.js");

function usage(exitCode) {
  const text = `Woxi - Interpreter for a subset of the Wolfram Language

Usage:
  woxi eval '<code>'   Evaluate an expression (pass \`-\` to read stdin)
  woxi <file>          Run a Wolfram Language script file
  woxi --help          Show this help
`;
  (exitCode === 0 ? process.stdout : process.stderr).write(text);
  process.exit(exitCode);
}

function run(code) {
  const items = woxi.evaluateAll(code);
  let sawError = false;
  for (const item of items) {
    switch (item.type) {
      case "error":
        sawError = true;
        process.stderr.write(`Error: ${item.text}\n`);
        break;
      case "warning":
        process.stderr.write(`${item.text}\n`);
        break;
      case "graphics":
        process.stdout.write(`${item.svg}\n`);
        break;
      case "sound":
        // No audio device in a terminal - print a placeholder like the echo
        // form of the expression would.
        process.stdout.write("-Sound-\n");
        break;
      default:
        if (item.text !== undefined && item.text !== "") {
          process.stdout.write(`${item.text}\n`);
        }
    }
  }
  process.exit(sawError ? 1 : 0);
}

const args = process.argv.slice(2);

if (args.length === 0 || args[0] === "--help" || args[0] === "-h") {
  usage(args.length === 0 ? 1 : 0);
} else if (args[0] === "eval") {
  if (args.length !== 2) {
    usage(1);
  }
  const code =
    args[1] === "-" ? fs.readFileSync(0, "utf8") : args[1];
  run(code);
} else if (args[0].startsWith("-")) {
  usage(1);
} else {
  let source;
  try {
    source = fs.readFileSync(args[0], "utf8");
  } catch (err) {
    process.stderr.write(`Error: cannot read ${args[0]}: ${err.message}\n`);
    process.exit(1);
  }
  // Strip a shebang line so `#!/usr/bin/env woxi` scripts work.
  if (source.startsWith("#!")) {
    source = source.slice(source.indexOf("\n") + 1);
  }
  run(source);
}
