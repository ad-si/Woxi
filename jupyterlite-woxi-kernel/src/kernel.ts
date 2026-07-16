import { KernelMessage } from '@jupyterlab/services';
import { BaseKernel, IKernel } from '@jupyterlite/kernel';

interface OutputItem {
  type: 'text' | 'graphics' | 'print' | 'warning' | 'error';
  text?: string;
  svg?: string;
}

interface WoxiWasm {
  default: () => Promise<void>;
  evaluate_all: (code: string) => string;
  clear: () => void;
  set_dark_mode: (enabled: boolean) => void;
}

export class WoxiKernel extends BaseKernel implements IKernel {
  private _wasm: WoxiWasm | null = null;
  private _ready: Promise<void>;
  // Bumped on every (re)load so the dynamic glue import URL is unique. Each
  // distinct module specifier gives a fresh module namespace with its own
  // `wasm` binding, which is the only way to obtain a brand-new WASM instance
  // — `module.default()` short-circuits once the cached `wasm` is set.
  private _loadCount = 0;
  // Captures the message forwarded by the Rust panic hook for the in-flight
  // evaluation; read after a trap to report a real cause instead of
  // "unreachable", then cleared.
  private _lastPanic: string | null = null;

  constructor(options: IKernel.IOptions) {
    super(options);
    this._installHostCallbacks();
    this._ready = this._initWasm();
  }

  /** Install the JS globals the WASM module imports (fetch + panic hook). */
  private _installHostCallbacks(): void {
    // Provide __woxi_fetch_url so Import["https://..."] works from WASM.
    (globalThis as any).__woxi_fetch_url = function (url: string): string {
      const xhr = new XMLHttpRequest();
      xhr.open('GET', url, false);
      xhr.overrideMimeType('text/plain; charset=x-user-defined');
      xhr.send();
      if (xhr.status < 200 || xhr.status >= 300) {
        throw new Error('HTTP ' + xhr.status + ' ' + xhr.statusText);
      }
      const text = xhr.responseText;
      const bytes = new Uint8Array(text.length);
      for (let i = 0; i < text.length; i++) {
        bytes[i] = text.charCodeAt(i) & 0xff;
      }
      let binary = '';
      for (let i = 0; i < bytes.length; i++) {
        binary += String.fromCharCode(bytes[i]);
      }
      return btoa(binary);
    };

    // The Rust panic hook calls this just before the `unreachable` trap, so we
    // can surface the real panic message and know an auto-restart is needed.
    (globalThis as any).__woxi_report_panic = (msg: string): void => {
      this._lastPanic = msg;
    };
  }

  /**
   * Load (or reload) the WASM glue and instantiate a fresh module. A unique
   * query string per load forces a new ES module namespace — without it a
   * re-import returns the cached (and, after a trap, poisoned) instance.
   */
  private async _loadWasm(): Promise<WoxiWasm> {
    // Resolve the WASM package URL relative to the JupyterLite root.
    // JupyterLite serves from e.g. /jupyterlite/lab/index.html,
    // and the wasm dir is at /jupyterlite/wasm/.
    const base = new URL('../wasm/woxi.js', window.location.href);
    base.searchParams.set('reload', String(this._loadCount++));
    const module: WoxiWasm = await import(
      /* webpackIgnore: true */ base.href
    );
    await module.default();
    module.clear();
    return module;
  }

  private async _initWasm(): Promise<void> {
    this._wasm = await this._loadWasm();
  }

  /**
   * Re-instantiate the WASM module after a trap and return a human-readable
   * description of what crashed. The previous instance is unrecoverable: a
   * trap leaves its globals corrupted so every later call re-traps.
   */
  private async _recover(fallback: string): Promise<string> {
    const cause = this._lastPanic ?? fallback;
    this._lastPanic = null;
    this._wasm = null;
    try {
      this._wasm = await this._loadWasm();
    } catch (e: any) {
      return `${cause}\n(failed to restart the Woxi kernel: ${e?.message ?? e})`;
    }
    return cause;
  }

  /** Publish an error and return the matching execute-reply content. */
  private _reportError(
    errorText: string,
  ): KernelMessage.IExecuteReplyMsg['content'] {
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

  private _isDarkTheme(): boolean {
    const attr = document.body.getAttribute('data-jp-theme-light');
    if (attr !== null) {
      return attr === 'false';
    }
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  }

  async kernelInfoRequest(): Promise<KernelMessage.IInfoReplyMsg['content']> {
    return {
      implementation: 'Woxi',
      implementation_version: '0.2.0',
      language_info: {
        codemirror_mode: { name: 'mathematica' },
        file_extension: '.wls',
        mimetype: 'application/vnd.wolfram.mathematica',
        name: 'wolfram',
        version: '0.2.0',
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

    if (!this._wasm) {
      // A previous recovery failed to reload; try once more before giving up.
      try {
        this._wasm = await this._loadWasm();
      } catch (e: any) {
        return this._reportError(`Woxi kernel is not loaded: ${e?.message ?? e}`);
      }
    }

    let json: string;
    try {
      this._lastPanic = null;
      this._wasm.set_dark_mode(this._isDarkTheme());
      json = this._wasm.evaluate_all(code);
    } catch (e: any) {
      // A WASM trap (e.g. a Rust panic compiled to `unreachable`) poisons the
      // instance: every later call would re-trap. Reinstantiate a fresh
      // module and tell the user their session state was cleared.
      const cause = await this._recover(e?.message ?? String(e));
      return this._reportError(
        'The Woxi kernel hit an internal error and was automatically ' +
          'restarted. All variable and function definitions have been ' +
          'cleared — re-run earlier cells to continue.\n\nCause: ' +
          cause,
      );
    }

    try {
      const items: OutputItem[] = JSON.parse(json);

      for (const item of items) {
        switch (item.type) {
          case 'print':
            this.publishExecuteResult({
              execution_count: this.executionCount,
              data: { 'text/plain': item.text ?? '' },
              metadata: {},
            });
            break;

          case 'warning':
            this.publishExecuteError({
              ename: 'Warning',
              evalue: item.text ?? '',
              traceback: [item.text ?? ''],
            });
            break;

          case 'error':
            this.publishExecuteError({
              ename: 'EvaluationError',
              evalue: item.text ?? '',
              traceback: [item.text ?? ''],
            });
            break;

          case 'graphics':
            this.publishExecuteResult({
              execution_count: this.executionCount,
              data: {
                'text/html': item.svg ?? '',
                'text/plain': '',
              },
              metadata: {},
            });
            break;

          case 'text': {
            if (item.svg) {
              this.publishExecuteResult({
                execution_count: this.executionCount,
                data: {
                  'text/html': item.svg,
                  'text/plain': item.text ?? '',
                },
                metadata: {},
              });
            } else {
              this.publishExecuteResult({
                execution_count: this.executionCount,
                data: { 'text/plain': item.text ?? '' },
                metadata: {},
              });
            }
            break;
          }
        }
      }

      return {
        status: 'ok',
        execution_count: this.executionCount,
        user_expressions: {},
      };
    } catch (e: any) {
      return this._reportError(e?.message ?? String(e));
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
