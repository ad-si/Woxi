#! /usr/bin/env bash

# Prune the libFuzzer `interpret` corpus down to the seeds that target can
# actually evaluate, quickly, and report what was dropped.
#
# Three kinds of seed are useless or harmful there:
#
#   - Oversized ones. The target returns immediately for inputs longer than
#     its length cap, so they only cost the fuzzer corpus slots.
#   - Denylisted ones. The target refuses inputs mentioning filesystem or
#     network heads, or the loop heads whose termination the program itself
#     decides, so those seeds never run either — and the timing pass below
#     must not run the side-effecting ones, as that *would* touch the disk.
#   - Slow ones. `tests/scripts/` holds programs that legitimately compute
#     for seconds; under ASan they take ~25× as long, and libFuzzer's
#     `-timeout` hang detector reports them as crashes. Interpreting a seed
#     has to stay well inside that timeout to be worth fuzzing.
#
# The length cap and the head denylist are read out of the fuzz target so
# this script cannot drift away from what the target actually skips.
#
# Usage: scripts/prune_fuzz_corpus.sh <corpus-dir> <seconds-per-seed>

set -euo pipefail

corpus="${1:?usage: prune_fuzz_corpus.sh <corpus-dir> <seconds-per-seed>}"
budget="${2:?usage: prune_fuzz_corpus.sh <corpus-dir> <seconds-per-seed>}"
target="fuzz/fuzz_targets/interpret.rs"
woxi="${WOXI:-./target/release/woxi}"

if [ ! -x "$woxi" ]
then
  echo "prune_fuzz_corpus: no woxi binary at $woxi" >&2
  exit 1
fi

# `if data.len() > 2048 { return; }` in the fuzz target.
max_len=$(grep -o 'data\.len() > [0-9]\+' "$target" | grep -o '[0-9]\+')

# Every denylist's entries, one head per line, as a grep pattern file — the
# target matches them as plain substrings, and so does `grep -F`.
denylist=$(mktemp)
trap 'rm -f "$denylist"' EXIT
awk '/_DENYLIST: &\[&str\]/, /\];/' "$target" \
  | grep -o '"[A-Za-z]\+"' \
  | tr -d '"' \
  > "$denylist"

dropped_size=0
dropped_denylisted=0
for seed in "$corpus"/*
do
  if [ "$(wc -c < "$seed")" -gt "$max_len" ]
  then
    rm -f "$seed"
    dropped_size=$((dropped_size + 1))
  elif grep -q -F -f "$denylist" "$seed"
  then
    rm -f "$seed"
    dropped_denylisted=$((dropped_denylisted + 1))
  fi
done

# Time what is left. `timeout` exits 124 only when the seed outlived its
# budget; any other non-zero exit is an ordinary evaluation error, which is
# cheap and stays in the corpus.
export WOXI_BUDGET="$budget"
export WOXI_BIN="$woxi"
dropped_slow=$(
  find "$corpus" -type f -print0 \
    | xargs -0 -P "$(nproc 2>/dev/null || echo 4)" -I {} \
      bash -c '
        timeout "$WOXI_BUDGET" "$WOXI_BIN" run "$1" \
          > /dev/null 2>&1 < /dev/null \
          || test $? -ne 124 \
          || { rm -f "$1"; echo "$1"; }
      ' _ {}
)

if [ -n "$dropped_slow" ]
then
  echo "$dropped_slow" | sed 's/^/  too slow to fuzz: /'
  dropped_slow_count=$(echo "$dropped_slow" | wc -l)
else
  dropped_slow_count=0
fi

echo "prune_fuzz_corpus: dropped $dropped_size oversized," \
  "$dropped_denylisted denylisted and" \
  "$dropped_slow_count seeds slower than ${budget}s;" \
  "$(find "$corpus" -type f | wc -l) left"
