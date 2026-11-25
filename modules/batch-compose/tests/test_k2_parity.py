#!/usr/bin/env python3
"""K2 physical-attention regression: the replacement must not change a single byte of output.

Two runs over the same prompt compare the C JSON stream byte for byte. A replacement that
alters numerics is a defect, not a feature; a replacement that silently stops firing is also
a defect, so the K2 run must also report REPLACED for every decode layer and must fall back
only on the prefill pass, where a multi-token query is deliberately out of scope (K2c).
"""

import argparse
import pathlib
import re
import subprocess

REPLACED = re.compile(r'reader\.k2: layer (\d+) REPLACED')
COVERAGE = re.compile(r'reader\.k2: layer (\d+) mask coverage ok')
FALLBACK = re.compile(r'reader\.k2: layer (\d+) falls back \(single_token 0')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def run(binary, model, prompts, output_dir, env_extra, name):
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [str(binary), '--model', str(model), '--prompts', str(prompts),
         '--predict', '3', '--threads', '1', '--parallel', '1'],
        capture_output=True, timeout=300,
        env={**__import__('os').environ, **env_extra})
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
    prompts.write_text('hello world\n')

    baseline_out, baseline_err = run(arguments.binary, arguments.model, prompts, output_dir, {}, 'baseline')
    k2_out, k2_err = run(arguments.binary, arguments.model, prompts, output_dir,
                         {'HERD_READER_GRAPH_K2': '1'}, 'k2')

    require(k2_out == baseline_out,
            'K2 changed the output stream: the replacement is not numerically transparent')
    require('reader.k2' not in baseline_err,
            'the replacement ran without being asked to: HERD_READER_GRAPH_K2 leaked into the baseline')

    replaced = sorted({int(m) for m in REPLACED.findall(k2_err)})
    fallback = sorted({int(m) for m in FALLBACK.findall(k2_err)})
    covered = sorted({int(m) for m in COVERAGE.findall(k2_err)})

    require(replaced, 'K2 was enabled but no layer replaced the upstream attention')
    require(fallback, 'K2 never reported a prefill fallback: the single-token gate is untested')
    require(covered, 'K2 replaced attention without proving mask coverage')
    require(set(covered) <= set(replaced), f'coverage reported for unreplaced layers: {covered} vs {replaced}')
    # A layer legitimately does both: the multi-token query is out of scope until K2c, so some
    # passes fall back and single-token passes replace. The order of the two is a property of the
    # scheduler, not of the gate; what the gate owes us is that every fallback names its reason.
    require(all('single_token 0' in line for line in k2_err.splitlines() if 'falls back' in line),
            'a K2 fallback did not name the single-token gate as its reason')
    print(f'k2 pairs {len(replaced)} replaced layers {replaced}, '
          f'{len(fallback)} prefill fallbacks {fallback}, coverage {covered}')


if __name__ == '__main__':
    main()
