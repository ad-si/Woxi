/** One structured output item produced by evaluating a statement. */
export interface OutputItem {
  /** Kind of output this item carries. */
  type:
    | "text"
    | "graphics"
    | "print"
    | "warning"
    | "error"
    | "sound"
    | "manipulate";
  /** Textual content (present for text/print/warning/error items). */
  text?: string;
  /** SVG markup (present for graphics items and rich text renderings). */
  svg?: string;
  /** Base64-encoded audio data (present for sound items). */
  audio?: string;
  /** MIME type of the audio data (present for sound items). */
  mime?: string;
  /** Label shown next to the audio player (optional on sound items). */
  label?: string;
  /** Additional fields (e.g. the Manipulate widget spec). */
  [key: string]: unknown;
}

/**
 * Evaluate one or more Wolfram Language statements and return the combined
 * output (Print output followed by the final result) as a string.
 * Evaluation errors are returned as a string starting with "Error: ".
 */
export function evaluate(code: string): string;

/** Evaluate all top-level statements and return structured output items. */
export function evaluateAll(code: string): OutputItem[];

/** Split code into top-level statements (for progressive evaluation). */
export function splitStatements(code: string): string[];

/** Evaluate a single statement, returning structured output items. */
export function evaluateStatement(statement: string): OutputItem[];

/** SVG graphics captured by the last evaluate() call ("" when none). */
export function getGraphics(): string;

/** Base64-encoded audio captured by the last evaluate() call ("" when none). */
export function getSound(): string;

/** Warnings emitted by the last evaluate() call. */
export function getWarnings(): string[];

/** Clear all interpreter state (variables and function definitions). */
export function clear(): void;

/** Toggle dark-mode colors for SVG output. */
export function setDarkMode(enabled: boolean): void;

/** Register an in-memory file so `Import["name"]` can read it. */
export function setVirtualFile(
  name: string,
  data: string | Uint8Array | ArrayBuffer,
): void;

/** Remove all files registered with setVirtualFile(). */
export function clearVirtualFiles(): void;
