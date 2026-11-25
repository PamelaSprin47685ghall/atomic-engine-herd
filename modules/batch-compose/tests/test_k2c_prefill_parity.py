#!/usr/bin/env python3
"""K2c prefill regression: a replaced multi-token prefill must not change one byte of output.

The two halves of the physical-attention seam are disjoint by construction -- the prefill
emitter refuses `n_tokens == 1` by name, and K2 admits only that case -- so K2c needs its own
run over a prompt long enough to reach the multi-token path, and its own non-vacuity clause.

The slice is capped by the graph arena, not by correctness: the emission charges
`n_tokens * (13 + q_cast + 6 * spans)` tensors at `ggml_tensor_overhead()` bytes, and the
prompt graph's arena is fixed by the wide reserve. On the tiny fixture T=128 does not fit and
T<=13 does, so the run passes `--ubatch-size 8` -- a real replacement at a small T, which is
what proves the emitter rather than the scheduler.

A test that only compares stdout passes vacuously while every layer falls back; that is why
`REPLACED` is asserted FIRST, before parity.
"""

import argparse
import os
import pathlib
import re
import subprocess

REPLACED = re.compile(r'reader\.prefill: layer (\d+) REPLACED')
COVERAGE = re.compile(r'reader\.prefill: layer 0 mask coverage ok')
FALLBACK = re.compile(r'reader\.prefill: layer (\d+) falls back')
MULTI_FALLBACK = re.compile(r'reader\.prefill: layer \d+ falls back \(multi_token 1')

# 40 tokens: long enough that the prompt is issued as a multi-token ubatch, and every token is
# an ordinary word so the tokenizer does not collapse the prompt into fewer pieces.
PROMPT = ' '.join(['the'] * 40) + '\n'

# Enough decode steps that the parity clause compares digits of real logits rather than a single
# end-of-text byte: predict 1-3 on the tiny fixture emits ~17 bytes, which proves the path ran but
# says little about numerics. Sixteen steps widen the compared surface without touching the arena
# slice, which is set by the prefill construction alone.
PREDICT = 16


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def run(binary, model, prompts, output_dir, env_extra, name):
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [str(binary), '--model', str(model), '--prompts', str(prompts),
         '--predict', str(PREDICT), '--threads', '1', '--parallel', '1',
         '--ubatch-size', '8'],
        capture_output=True, timeout=300,
        env={**os.environ, **env_extra})
    require(completed.returncode == 0,
            f'{name} exited {completed.returncode}: '
            f'{completed.stderr[-2000:].decode("utf-8", "replace")}')
    (output_dir / f'{name}.stdout').write_bytes(completed.stdout)
    (output_dir / f'{name}.stderr').write_bytes(completed.stderr)
    return completed.stdout, completed.stderr.decode('utf-8', 'replace')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output-dir', required=True)
    arguments = parser.parse_args()

    output_dir = pathlib.Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = output_dir / 'prompts.txt'
    prompts.write_text(PROMPT)

    baseline_out, baseline_err = run(arguments.binary, arguments.model, prompts,
                                     output_dir, {}, 'baseline')
    k2c_out, k2c_err = run(arguments.binary, arguments.model, prompts, output_dir,
                           {'HERD_READER_GRAPH_K2C': '1'}, 'k2c')

    replaced = sorted({int(m) for m in REPLACED.findall(k2c_err)})

    # Non-vacuity first: without a replacement the parity below compares two fallback runs and
    # would pass while the emitter is never entered.
    require(replaced, 'K2c was enabled but no layer replaced the upstream prefill attention')
    require(replaced == [0, 1, 2],
            f'K2c replaced {replaced}; the whole prefill graph must be replaced or reported')
    require(k2c_out == baseline_out,
            'K2c changed the output stream: the prefill replacement is not numerically transparent')
    require('reader.prefill' not in baseline_err,
            'the replacement ran without being asked to: HERD_READER_GRAPH_K2C leaked into the baseline')
    require('REFUSED' not in k2c_err,
            f'a K2c layer refused its own emission: {k2c_err[-2000:]}')
    require(COVERAGE.search(k2c_err),
            'K2c replaced attention without proving per-token mask coverage')
    require(not MULTI_FALLBACK.search(k2c_err),
            'a multi-token prefill fell back while K2c was enabled: the arena slice is too large '
            f'for the emitter to be exercised: {MULTI_FALLBACK.search(k2c_err)}')
    # K2's own gate must stay untouched: the two switches describe disjoint token counts, and a
    # run that reports both would be reporting one implementation as the other.
    require('reader.k2' not in k2c_err,
            'K2c enabled the K2 single-token path: the two switches are not exclusive')

    fallbacks = sorted({int(m) for m in FALLBACK.findall(k2c_err)})
    print(f'k2c replaced {replaced}, prefill fallbacks {fallbacks}, '
          f'byte-identical output over {len(baseline_out)} bytes')


if __name__ == '__main__':
    main()
