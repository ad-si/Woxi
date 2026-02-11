import { KernelMessage } from '@jupyterlab/services';
import { BaseKernel, IKernel } from '@jupyterlite/kernel';

interface WoxiWasm {
  default: () => Promise<void>;
  evaluate: (code: string) => string;
  clear: () => void;
}

export class WoxiKernel extends BaseKernel implements IKernel {
  private _wasm: WoxiWasm | null = null;
  private _ready: Promise<void>;

  constructor(options: IKernel.IOptions) {
    super(options);
    this._ready = this._initWasm();
  }

  private async _initWasm(): Promise<void> {
    // Resolve the WASM package URL relative to the JupyterLite root.
    // JupyterLite serves from e.g. /jupyterlite/lab/index.html,
    // and the wasm dir is at /jupyterlite/wasm/.
    // Use '../wasm/woxi.js' relative to the current page location.
    const wasmUrl = new URL('../wasm/woxi.js', window.location.href).href;

    const module: WoxiWasm = await import(
      /* webpackIgnore: true */ wasmUrl
    );
    await module.default();
    module.clear();
    this._wasm = module;
  }

  async kernelInfoRequest(): Promise<KernelMessage.IInfoReplyMsg['content']> {
    return {
      implementation: 'Woxi',
      implementation_version: '0.1.0',
      language_info: {
        codemirror_mode: { name: 'mathematica' },
        file_extension: '.wls',
        mimetype: 'application/vnd.wolfram.mathematica',
        name: 'wolfram',
        version: '0.1.0',
      },
      protocol_version: '5.3',
      status: 'ok',
      banner: 'Woxi — Wolfram Language interpreter',
      help_links: [
        {
          text: 'Woxi Documentation',
          url: 'https://woxi.dev',
        },
      ],
    };
  }

  async executeRequest(
    content: KernelMessage.IExecuteRequestMsg['content'],
  ): Promise<KernelMessage.IExecuteReplyMsg['content']> {
    await this._ready;

    const code = content.code.trim();
    if (!code) {
      return {
        status: 'ok',
        execution_count: this.executionCount,
        user_expressions: {},
      };
    }

    try {
      const result = this._wasm!.evaluate(code);

      if (result) {
        // Detect SVG output (from Plot, etc.)
        if (result.trimStart().startsWith('<svg')) {
          this.publishExecuteResult({
            execution_count: this.executionCount,
            data: { 'image/svg+xml': result },
            metadata: {},
          });
        } else {
          this.publishExecuteResult({
            execution_count: this.executionCount,
            data: { 'text/plain': result },
            metadata: {},
          });
        }
      }

      return {
        status: 'ok',
        execution_count: this.executionCount,
        user_expressions: {},
      };
    } catch (e: any) {
      const errorText = e?.message ?? String(e);

      this.publishExecuteError({
        ename: 'EvaluationError',
        evalue: errorText,
        traceback: [errorText],
      });

      return {
        status: 'error',
        execution_count: this.executionCount,
        ename: 'EvaluationError',
        evalue: errorText,
        traceback: [errorText],
      };
    }
  }

  async completeRequest(
    _content: KernelMessage.ICompleteRequestMsg['content'],
  ): Promise<KernelMessage.ICompleteReplyMsg['content']> {
    return {
      matches: [],
      cursor_start: 0,
      cursor_end: 0,
      metadata: {},
      status: 'ok',
    };
  }

  async inspectRequest(
    _content: KernelMessage.IInspectRequestMsg['content'],
  ): Promise<KernelMessage.IInspectReplyMsg['content']> {
    return {
      status: 'ok',
      found: false,
      data: {},
      metadata: {},
    };
  }

  async isCompleteRequest(
    _content: KernelMessage.IIsCompleteRequestMsg['content'],
  ): Promise<KernelMessage.IIsCompleteReplyMsg['content']> {
    return { status: 'complete' };
  }

  async commInfoRequest(
    _content: KernelMessage.ICommInfoRequestMsg['content'],
  ): Promise<KernelMessage.ICommInfoReplyMsg['content']> {
    return {
      comms: {},
      status: 'ok',
    };
  }

  async inputReply(
    _content: KernelMessage.IInputReplyMsg['content'],
  ): Promise<void> {
    // No interactive input support
  }

  async commOpen(
    _msg: KernelMessage.ICommOpenMsg,
  ): Promise<void> {
    // No comm support
  }

  async commMsg(
    _msg: KernelMessage.ICommMsgMsg,
  ): Promise<void> {
    // No comm support
  }

  async commClose(
    _msg: KernelMessage.ICommCloseMsg,
  ): Promise<void> {
    // No comm support
  }
}
