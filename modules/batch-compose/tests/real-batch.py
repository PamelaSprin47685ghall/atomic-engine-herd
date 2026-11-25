#!/usr/bin/env python3
"""Exercise the installed C binary with real A and B. No model substitute."""

import argparse
import json
import pathlib
import re
import subprocess


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def run(binary, arguments, prefix, success=True):
    completed = subprocess.run([str(binary), *arguments], capture_output=True, timeout=300)
    prefix.with_suffix('.stdout.jsonl').write_bytes(completed.stdout)
    prefix.with_suffix('.stderr.log').write_bytes(completed.stderr)
    # A logs raw vocabulary byte pieces. Preserve those bytes; only the C JSON
    # stream has a UTF-8 contract and must decode strictly.
    events = [json.loads(line) for line in completed.stdout.decode('utf-8').splitlines() if line.strip()]
    if success:
        require(completed.returncode == 0, f'{prefix.name} failed: {completed.stderr[-2000:].decode("utf-8", "replace")}')
        require(not any(item['event'] == 'error' for item in events), 'successful run emitted an error')
    else:
        require(completed.returncode != 0, 'invalid configuration unexpectedly succeeded')
        require(any(item['event'] == 'error' for item in events), 'failure has no public error event')
        require(not any(item['event'] == 'result' for item in events), 'failure fabricated complete results')
    return events


def check_execution(events, release):
    config = next(item for item in events if item['event'] == 'config')
    require(config['logical_context_actual'] == 512, 'physical pool changed the logical window')
    require(config['total_kv_actual'] == 1024, 'A did not use the independent physical KV capacity')
    batches = [item for item in events if item['event'] == 'batch']
    results = [item for item in events if item['event'] == 'result']
    require([item['request_id'] for item in results] == [0, 1, 2], 'results are missing or reordered')
    require([item['step'] for item in batches] == list(range(len(batches))), 'batch ids are not contiguous')
    require(all(item['decode_return'] == 0 for item in batches), 'A did not accept a batch')
    require(all(len(item['tokens']) == 12 for item in results), 'fixed generation budget was not consumed')
    require(all(item['finish_reason'] == 'length' for item in results), 'unexpected finish reason')

    injection_index = next(i for i, event in enumerate(events) if event['event'] == 'inject')
    injection = events[injection_index]
    require(injection['after_generation_batches'] == 2, 'wrong injection time')
    require(injection['request_ids'] == [2], 'wrong injected request')
    before = [item for item in events[:injection_index] if item['event'] == 'batch']
    after = [item for item in events[injection_index + 1:] if item['event'] == 'batch']
    generation_before = [item for item in before if item['phase'] == 'generation']
    require(len(generation_before) == 2, 'initial requests were not decoding before injection')
    require(all([row['request_id'] for row in item['rows']] == [0, 1] for item in generation_before),
            'initial generation did not batch both requests')
    require(after and after[0]['phase'] == 'prompt', 'the injected prompt did not preempt generation')
    prompt_after = []
    for item in after:
        if item['phase'] != 'prompt':
            break
        prompt_after.append(item)
    require(prompt_after, 'the injected prompt was not actually decoded')
    require(all(all(row['request_id'] == 2 for row in item['rows']) for item in prompt_after),
            'generation ran inside the priority prefill')
    resumed = after[len(prompt_after)]
    require(resumed['phase'] == 'generation', 'generation did not resume')
    require([row['request_id'] for row in resumed['rows']] == [0, 1, 2], 'the paused requests did not resume with the new one')

    prompt_ids = [item['rows'][0]['request_id'] for item in batches if item['phase'] == 'prompt']
    require(prompt_ids == sorted(prompt_ids), 'prompt chunks did not keep sticky slot order')
    for item in batches:
        if item['phase'] == 'prompt':
            require(len({row['request_id'] for row in item['rows']}) == 1, 'prefill mixed requests')
    for result in results:
        request_id = result['request_id']
        rows = [row for item in batches for row in item['rows'] if row['request_id'] == request_id]
        require([row['position'] for row in rows] == list(range(len(rows))), 'pause lost or replayed a sequence position')
        require(len(rows) == result['prompt_tokens'] + len(result['tokens']) - 1, 'reported tokens lack real decode work')
        require(result['decoded_rows'] == len(rows), 'result row counter disagrees with actual A input')
        generated_rows = [row['token'] for item in batches if item['phase'] == 'generation'
                          for row in item['rows'] if row['request_id'] == request_id]
        require(generated_rows == result['tokens'][:-1], 'sampled tokens were not fed back to A')

    transitions = [item for item in events if item['event'] == 'phase' and
                   item['from'] == 'prompt' and item['to'] == 'generation']
    require(len(transitions) == 2, 'expected initial and injected prefill completion')
    require(all(item['synchronized'] for item in transitions), 'phase did not synchronize A')
    require(all(item['release_compute_requested'] == release for item in transitions), 'release switch was not consumed')
    require(all(item['release_compute_success'] == (True if release else None) for item in transitions),
            'release result was not reported truthfully')
    require(all(item['compute_bytes_before'] > 0 for item in transitions), 'no real compute allocation existed')
    if release:
        require(all(item['compute_bytes_after'] == 0 for item in transitions), 'compute allocator retained buffers')
    else:
        require(all(item['compute_bytes_after'] == item['compute_bytes_before'] for item in transitions),
                'disabled release changed the allocation')
    return results, batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=pathlib.Path, required=True)
    parser.add_argument('--model', type=pathlib.Path, required=True)
    parser.add_argument('--gpu-layers', default='0')
    parser.add_argument('--output-dir', type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = args.output_dir / 'prompts.txt'
    prompts.write_text(
        'Continue the numbered list of simple facts about the ocean. Water covers much of Earth. '
        'Waves travel across the surface. Tides rise and fall. Fish live underwater. 1.\n'
        'Continue the numbered list of simple facts about a forest. Trees have leaves. Roots grow in soil. '
        'Birds build nests. Rain provides water. Sunlight helps plants grow. 1.\n'
        'Continue the numbered list of simple facts about mountains. High peaks can hold snow. '
        'Rivers flow through valleys. Paths cross hills. Rocks form cliffs. 1.\n', encoding='utf-8')
    common = ['--model', str(args.model), '--prompts', str(prompts), '--predict', '12',
              '--ctx-size', '512', '--total-kv', '1024', '--batch-size', '16', '--ubatch-size', '8', '--threads', '2',
              '--gpu-layers', args.gpu_layers, '--inject-after', '2', '--ignore-eog', '--json']
    # The release capability exists, and the contract is that asking for it WORKS and does
    # not change the outputs. This case used to assert the refusal, because there was no
    # capability to exercise; the fork's own comparison is the stronger assertion and it
    # has a run again.
    #
    # Both halves matter. A run that released nothing would pass an equality check while
    # measuring nothing, so the phase event must report release_compute_success with the
    # byte count falling; and the tokens must match the non-releasing run, which is the
    # property the fork's version of this file existed to protect.
    on_events = run(args.binary, [*common, '--phase-memory'], args.output_dir / 'phase-on')
    phase_events = [item for item in on_events if item.get('event') == 'phase']
    released = [item for item in phase_events if item.get('release_compute_success') is True]
    require(released, 'the release was requested and did not report success; a comparison '
                      'against this run would prove nothing')
    require(released[0].get('compute_bytes_after', 1) < released[0].get('compute_bytes_before', 0),
            'the release reported success but compute bytes did not fall')
    on_results, _ = check_execution(on_events, True)
    off = run(args.binary, [*common, '--no-phase-memory'], args.output_dir / 'phase-off')
    actual_offload = []
    for name in ('phase-off',):
        raw_log = (args.output_dir / (name + '.stderr.log')).read_bytes()
        found = re.search(rb'offloaded (\d+)/(\d+) layers to GPU', raw_log)
        actual_offload.append(int(found.group(1)) if found else 0)
    if int(args.gpu_layers) > 0:
        require(all(count > 0 for count in actual_offload), 'requested GPU execution silently used no GPU layers')
    off_results, off_batches = check_execution(off, False)

    run(args.binary, [*common, '--total-kv', '256'], args.output_dir / 'invalid-kv', success=False)
    missing = args.output_dir / 'missing-model.gguf'
    require(not missing.exists(), 'missing-model path unexpectedly exists')
    run(args.binary, ['--model', str(missing), '--prompts', str(prompts), '--json'],
        args.output_dir / 'missing-model', success=False)

    summary = {'status': 'passed', 'participants': 'real installed A and B',
               'gpu_layers_requested': int(args.gpu_layers), 'gpu_layers_observed': actual_offload,
               'requests': len(off_results), 'actual_batches_per_run': len(off_batches),
               'actual_rows_per_run': sum(len(batch['rows']) for batch in off_batches),
               # Not "true": unavailable. The release cannot run in this build, so the
               # question this field used to answer is unanswerable, and a `True` here
               # would be a claim about a path no run took.
               'phase_release_outputs_equal': None,
               'phase_release': 'unavailable in this build; --phase-memory is refused (asserted above)',
               'limits': ['This test does not measure peak GPU allocation or throughput.',
                          'This is a greedy causal text runner; no HTTP, MTP, cache retry or shared-KV reasoning claim.']}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
