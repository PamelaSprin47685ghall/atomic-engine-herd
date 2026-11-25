#!/bin/sh
# The three outcomes of the reader replacement switch, checked against a real product process.
#
# Documentation cannot hold this property: `HERD_READER_REPLACE` is fail-closed only if a run that
# asks for it can never exit 0. That is a statement about exit codes and about stdout, so it is
# checked by running the binary, not by reading the source.
#
# Usage: replace-failclosed-test.sh <tq-batch-compose> <gguf>
#
# Case 3 is the one that needs a process: the refusing code lives inside `build_attn_mha`, so a run
# whose attention is never built would never enter it. Only an exit-time check catches that, and
# only running such a process can show it works.

set -u

bin=$1
model=$2
fail=0

note() { printf 'FAIL: %s\n' "$1" >&2; fail=$((fail + 1)); }

if [ ! -x "$bin" ]; then
    note "binary not executable: $bin"
    exit 1
fi
if [ ! -f "$model" ]; then
    note "model missing: $model"
    exit 1
fi

work=$(mktemp -d) || exit 1
trap 'rm -rf "$work"' EXIT
printf 'a b c\n' > "$work/prompt.txt"

# 1. No switch: the stock path, byte-identical to a run without the reader lane at all. If this
#    were red, every other case would be measuring a broken binary.
"$bin" --model "$model" --prompts "$work/prompt.txt" --predict 2 \
    >"$work/plain.out" 2>"$work/plain.err"
plain_status=$?
[ "$plain_status" -eq 0 ] || note "plain run exited $plain_status, expected 0"
[ -s "$work/plain.out" ] || note "plain run produced no stdout"
grep -q 'reader\.replace' "$work/plain.err" && note "plain run printed a replacement message"

# 2. The switch, with attention reached: refuse and say why, with NO output. Zero bytes is the
#    assertion that matters -- a partial or stock answer on stdout would be a result this lane did
#    not compute, whatever the exit code said.
HERD_READER_REPLACE=1 "$bin" --model "$model" --prompts "$work/prompt.txt" --predict 2 \
    >"$work/refuse.out" 2>"$work/refuse.err"
refuse_status=$?
[ "$refuse_status" -eq 134 ] || note "replace run exited $refuse_status, expected 134 (SIGABRT)"
[ -s "$work/refuse.out" ] && note "replace run wrote $(wc -c <"$work/refuse.out") bytes to stdout"
grep -q 'REFUSED' "$work/refuse.err" || note "replace run did not report REFUSED"
grep -q 'emitter cannot express' "$work/refuse.err" ||
    note "replace run did not name the reason"

# 3. The same switch with no model present. The refusal happens at static-init time, so this run
#    must reach the SAME verdict as case 2 -- not a model-load error path, and not exit 0. That is
#    the property the earlier design got wrong: a refusal placed inside the attention seam is only
#    reached by runs that build attention, and this one never would.
HERD_READER_REPLACE=1 "$bin" --model "$work/absent.gguf" --prompts "$work/prompt.txt" --predict 2 \
    >"$work/absent.out" 2>"$work/absent.err"
absent_status=$?
[ "$absent_status" -eq 134 ] || note "unreached run exited $absent_status, expected 134"
grep -q 'REFUSED' "$work/absent.err" || note "the no-model run did not refuse at startup"
[ -s "$work/absent.out" ] && note "the no-model run wrote to stdout"

# 4. `--help` with the switch set. A startup refusal must not depend on what main was asked to do,
#    and it must not be reached at exit: an exit handler would also fire here *after* printing the
#    help, turning a successful `--help` into SIGABRT for no reason a reader could see.
HERD_READER_REPLACE=1 "$bin" --help >"$work/help.out" 2>"$work/help.err"
help_status=$?
[ "$help_status" -eq 134 ] || note "--help with the switch exited $help_status, expected 134"
[ -s "$work/help.out" ] && note "--help with the switch still printed a usage text"

if [ "$fail" -eq 0 ]; then
    printf 'replace fail-closed: OK\n' >&2
    exit 0
fi
printf 'replace fail-closed: %d failure(s)\n' "$fail" >&2
exit 1
