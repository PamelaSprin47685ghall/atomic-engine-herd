#!/usr/bin/env python3
"""Check queued inference against controls with the same submitted batch shape."""

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def reject_constant(value):
    raise ValueError(f'non-finite JSON number: {value}')


def execute(binary, arguments, output, name, success=True):
    completed = subprocess.run([str(binary), *arguments], capture_output=True, timeout=600)
    (output / f'{name}.jsonl').write_bytes(completed.stdout)
    (output / f'{name}.log').write_bytes(completed.stderr)
    events = [json.loads(line, parse_constant=reject_constant)
              for line in completed.stdout.decode('utf-8').splitlines() if line.strip()]
    if success:
        require(completed.returncode == 0, f'{name}: {completed.stderr[-2000:].decode("utf-8", "replace")}')
        require(not any(event['event'] == 'error' for event in events), 'successful run reported an error')
    else:
        require(completed.returncode != 0, f'{name}: invalid options succeeded')
        require(any(event['event'] == 'error' for event in events), 'failure has no public error')
        require(not any(event['event'] in ('batch', 'result') for event in events), 'invalid options performed model work')
    return events, completed.stderr


def inspect(events, expected_parallel, expected_capacity, request_count, phase_memory):
    configs = [event for event in events if event['event'] == 'config']
    require(len(configs) == 1, 'expected one configuration event')
    config = configs[0]
    require(config['parallel_actual'] == expected_parallel, 'resident request limit was not consumed')
    require(config['total_kv_actual'] == expected_capacity, 'physical token capacity changed')
    require(config['logical_context_actual'] == 256, 'physical capacity changed the logical window')
    require(config['phase_memory'] is phase_memory, 'phase-memory option was not consumed')
    require(config['requests'] == request_count, 'fixture request count changed')
    if request_count == 9:
        require(request_count > config['batch_actual'], 'fixture did not exceed one batch of requests')
    results = [event for event in events if event['event'] == 'result']
    require([event['request_id'] for event in results] == list(range(request_count)), 'requests were lost or reordered')
    active = {}
    completed = set()
    admitted = []
    positions = {i: [] for i in range(request_count)}
    feedback = {i: [] for i in range(request_count)}
    used_slots = set()
    reserved = 0
    generation = 0
    peak = 0
    token_waits = 0
    slot_waits = 0
    compute_releases = 0
    for event in events:
        kind = event['event']
        if kind == 'admit':
            request = event['request_id']
            require(request in positions and request not in active and request not in admitted, 'invalid or repeated admission')
            require(0 <= event['sequence_id'] < expected_parallel, 'request id escaped into A sequence ids')
            require(event['sequence_id'] not in [held['sequence_id'] for held in active.values()], 'two live requests share one slot')
            require(event['generation'] > generation, 'slot generation was reused')
            generation = event['generation']
            require(event['tokens_reserved'] == results[request]['prompt_tokens'] + 15, 'reservation did not cover full generation')
            active[request] = event
            admitted.append(request)
            used_slots.add(event['sequence_id'])
            reserved += event['tokens_reserved']
            require(reserved == event['total_reserved'] and reserved <= expected_capacity, 'admission exceeded the token budget')
            peak = max(peak, len(active))
            require(peak <= expected_parallel, 'admission exceeded the slot budget')
        elif kind == 'wait':
            require(event['request_id'] not in active, 'active request was queued again')
            require(event['tokens_available'] == expected_capacity - reserved, 'wait used stale token accounting')
            require(event['active_requests'] == len(active), 'wait used stale slot accounting')
            token_waits += event['tokens_required'] > event['tokens_available']
            slot_waits += event['active_requests'] == expected_parallel
            require(event['tokens_required'] > event['tokens_available'] or len(active) == expected_parallel,
                    'request waited without capacity pressure')
        elif kind == 'phase':
            require(event['synchronized'] is True, 'phase change did not report its fence')
            release = phase_memory and event['from'] == 'prompt' and event['to'] == 'generation'
            require(event['release_compute_requested'] is release, 'release did not follow the configured phase policy')
            if release:
                require(event['release_compute_success'] is True and event['compute_bytes_after'] == 0,
                        'compute memory was not released')
                require(event['compute_bytes_before'] > 0, 'release had no compute allocation to exercise')
                compute_releases += 1
            else:
                require(event['release_compute_success'] is None, 'an unrequested release reported success')
        elif kind == 'batch':
            require(event['decode_return'] == 0, 'A rejected an admitted batch')
            require(event['rows'], 'empty batch did not exercise A')
            for row in event['rows']:
                request = row['request_id']
                require(request in active, 'A executed an unreserved request')
                require(row['sequence_id'] == active[request]['sequence_id'], 'A used the wrong live sequence slot')
                positions[request].append(row['position'])
                if event['phase'] == 'generation':
                    feedback[request].append(row['token'])
            for sample in event['samples']:
                require(sample['request_id'] in active, 'A sampled an unreserved request')
                require(math.isfinite(sample['logit_best']), 'sample has a non-finite best logit')
                require(sample['logit_next'] is None or math.isfinite(sample['logit_next']), 'sample has a non-finite next logit')
                if sample['finished']:
                    completed.add(sample['request_id'])
        elif kind == 'release':
            request = event['request_id']
            require(request in active and request in completed, 'live request was released before completion')
            held = active.pop(request)
            require(event['sequence_id'] == held['sequence_id'] and event['generation'] == held['generation'],
                    'release named the wrong slot generation')
            reserved -= held['tokens_reserved']
            require(reserved == event['total_reserved'], 'release lost a reservation')
    require(admitted == list(range(request_count)), 'admission did not preserve input order')
    require(not active and reserved == 0, 'completed work retained a reservation')
    require(len(used_slots) < len(admitted), 'test did not reuse actual sequence slots')
    require(not phase_memory or compute_releases > 0, 'phase memory did not exercise a release')
    for result in results:
        request = result['request_id']
        require(len(result['tokens']) == 16 and result['finish_reason'] == 'length', 'request budget was not completed')
        require(positions[request] == list(range(result['decoded_rows'])), 'slot reuse lost, duplicated, or shifted positions')
        require(len(positions[request]) == result['prompt_tokens'] + 15, 'result counters lack actual model rows')
        require(feedback[request] == result['tokens'][:-1], 'sampled tokens were not fed back to A')
    return results, {'peak_resident_requests': peak, 'token_waits': token_waits, 'slot_waits': slot_waits,
                     'reused_slots': sorted(used_slots), 'requests': len(results), 'compute_releases': compute_releases}


def batch_rows(events, tokens=True):
    return [(event['phase'], [{key: value for key, value in row.items() if tokens or key != 'token'}
                             for row in event['rows']])
            for event in events if event['event'] == 'batch']


def result_value(result):
    return {key: value for key, value in result.items() if key not in ('event', 'request_id')}


def require_equal_execution(left, right, label):
    require(batch_rows(left['events'], tokens=False) == batch_rows(right['events'], tokens=False),
            f'{label}: controls have different actual row shapes')
    require(batch_rows(left['events']) == batch_rows(right['events']), f'{label}: submitted tokens changed')
    require([result_value(value) for value in left['results']] == [result_value(value) for value in right['results']],
            f'{label}: output tokens or text changed')


def cross_shape_diagnostic(left, right):
    differences = []
    for first, second in zip(left['results'], right['results']):
        if first['tokens'] != second['tokens']:
            index = next(i for i, (a, b) in enumerate(zip(first['tokens'], second['tokens'])) if a != b)
            differences.append({'request_id': first['request_id'], 'first_token_index': index,
                                'left_token': first['tokens'][index], 'right_token': second['tokens'][index]})
    return {'tokens_equal': not differences, 'different_requests': differences,
            'actual_row_shapes_equal': batch_rows(left['events'], tokens=False) == batch_rows(right['events'], tokens=False),
            'qualification': 'diagnostic only; this comparison changes the submitted batch shape'}


def wave_rows(events, first, count):
    result = []
    for phase, rows in batch_rows(events):
        selected = [row for row in rows if first <= row['request_id'] < first + count]
        if selected:
            require(len(selected) == len(rows), 'canary waves overlapped in a submitted batch')
            result.append((phase, [dict(row, request_id=row['request_id'] - first) for row in selected]))
    require(result, 'canary wave did not submit any rows')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--gpu-layers', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = Path(tempfile.mkdtemp(prefix='run-', dir=args.output_dir))
    prompts = args.output_dir / 'prompts.txt'
    # The fixture length is a measured constraint, not a style choice. With --ctx-size 256
    # and --predict 16 a prompt may use about 240 tokens, and the token budget must be
    # tight enough that two requests cannot be resident together (2*(p+15) > 256) or the
    # serial controls below have nothing to observe. For a byte-level tokenizer one token
    # is roughly one byte, so this sentence is repeated to about 150 bytes; the three-wide
    # case measures the real count and names what to change if another tokenizer moves it.
    body = 'Sea water is salt. Waves move sand. ' * 4
    fixture = [f'Case {i}. {body}1.\n' for i in range(9)]
    prompts.write_text(''.join(fixture), encoding='utf-8')
    common = ['--model', str(args.model), '--predict', '16', '--ctx-size', '256',
              '--batch-size', '8', '--ubatch-size', '8', '--threads', '2', '--gpu-layers', str(args.gpu_layers),
              '--ignore-eog', '--json']
    records = {}
    runs = {}

    def run_case(name, parallel, capacity, phase, source=prompts, count=9):
        events, log = execute(args.binary, [*common, '--prompts', str(source), '--parallel', str(parallel),
                                           '--total-kv', str(capacity), '--phase-memory' if phase else '--no-phase-memory'],
                              args.output_dir, name)
        results, record = inspect(events, parallel, capacity, count, phase)
        offload = re.search(rb'offloaded (\d+)/(\d+) layers to GPU', log)
        record['gpu_layers_observed'] = int(offload.group(1)) if offload else 0
        if args.gpu_layers > 0:
            require(record['gpu_layers_observed'] > 0, 'GPU request silently used CPU only')
        records[name] = record
        runs[name] = {'events': events, 'results': results}
        return runs[name]

    pairs = []
    # Each capacity case runs once with --no-phase-memory, then once with --phase-memory,
    # and the two runs are compared.
    #
    # The fork's version of this file did exactly that, to prove that releasing compute
    # memory at the prompt-to-generation boundary does not change the outputs. That
    # comparison was replaced here by a refusal assertion when the capability did not
    # exist; it exists now (release_compute honours the flag and reports
    # release_compute_success), so the stronger assertion is back and the refusal
    # assertion would be pinning an absence -- the kind of stale test that fails when the
    # capability is implemented and looks like a regression from the other side.
    for name, parallel, capacity in [('three-wide', 3, 2048), ('token-pressure', 3, 256), ('slot-pressure', 1, 2048)]:
        key = f'{name}-phase-off'
        run = run_case(key, parallel, capacity, False)
        record = records[key]
        if name == 'three-wide':
            require(record['peak_resident_requests'] == 3, 'reference never exercised concurrent requests')
            require(sum(result['prompt_tokens'] + 15 for result in run['results'][:3]) > 256,
                    'fixture cannot exercise token capacity pressure')
            lengths = [result['prompt_tokens'] for result in run['results']]
            require(all(114 <= value <= 240 for value in lengths),
                    f'fixture prompt length {min(lengths)}..{max(lengths)} is outside the 114..240 tokens '
                    'this case needs: below ~114 two requests fit the 256-cell budget together (no token '
                    'pressure), and above ~240 a prompt plus its generation no longer fits --ctx-size. '
                    'Adjust the sentence count in the fixture, not the assertions.')
        elif name == 'token-pressure':
            require(record['token_waits'] > 0 and record['peak_resident_requests'] == 1,
                    'token budget did not produce the required serial control')
        else:
            require(record['slot_waits'] > 0 and record['peak_resident_requests'] == 1,
                    'slot budget did not restrict admission')
        on = run_case(f'{name}-phase-on', parallel, capacity, True)
        # The release must actually happen, or this pair proves nothing about it: a run
        # that kept its buffers and produced the same tokens would pass an equality check
        # while measuring nothing. The phase event carries the answer.
        phase_events = [item for item in on['events'] if item.get('event') == 'phase']
        released = [item for item in phase_events if item.get('release_compute_success') is True]
        require(released, f'{name}: --phase-memory did not report a successful release '
                          f'(events: {[item.get("release_compute_success") for item in phase_events]})')
        require(released[0].get('compute_bytes_after', 1) < released[0].get('compute_bytes_before', 0),
                f'{name}: the release reported success but compute bytes did not fall')
        require_equal_execution(runs[key], on, f'{name}: releasing compute memory changed the outputs')
        pairs.append(name)

    for phase in ('off',):
        require_equal_execution(runs[f'token-pressure-phase-{phase}'], runs[f'slot-pressure-phase-{phase}'],
                                f'serial capacity pair, phase {phase}')

    canary_file = args.output_dir / 'repeat-wave-prompts.txt'
    canary_file.write_text(''.join(fixture[:3] * 2), encoding='utf-8')
    canary = run_case('repeat-wave', 3, 2048, False, canary_file, 6)
    require(records['repeat-wave']['peak_resident_requests'] == 3, 'canary did not fill three resident slots')
    original = runs['three-wide-phase-off']
    require(wave_rows(original['events'], 0, 3) == wave_rows(canary['events'], 0, 3),
            'canary first wave changed the original fresh control')
    require(wave_rows(canary['events'], 0, 3) == wave_rows(canary['events'], 3, 3),
            'reused canary slots changed submitted rows or feedback tokens')
    for lane in range(3):
        expected = result_value(original['results'][lane])
        require(result_value(canary['results'][lane]) == expected, 'canary changed a fresh-lane result')
        require(result_value(canary['results'][lane + 3]) == expected, 'dirty slot reuse changed a same-lane result')

    for name, extra in [('zero-parallel', ['--parallel', '0']),
                        ('too-many-slots', ['--parallel', '9']),
                        ('single-slot-injection', ['--parallel', '1', '--inject-after', '2'])]:
        execute(args.binary, [*common, '--prompts', str(prompts), *extra], args.output_dir, name, success=False)

    summary = {'status': 'passed', 'runs': records, 'same_configuration_cases': pairs,
               'serial_capacity_pairs_equal': ['phase-off'],
               'phase_memory': 'unavailable in this build; --phase-memory is refused (asserted per case)',
               'repeat_wave_equal': {'fresh_requests': [0, 1, 2], 'reused_requests': [3, 4, 5]},
               'cross_shape_diagnostics': {
                   phase: cross_shape_diagnostic(runs[f'three-wide-phase-{phase}'], runs[f'token-pressure-phase-{phase}'])
                   for phase in ('off',)},
               'output_dir': str(args.output_dir),
               'limits': ['Token reservations cover known request bounds; they do not measure VRAM.',
                          'Changing the submitted batch shape can change greedy tokens in the pinned upstream runtime.',
                          'The diagnostic does not establish the numerical cause of a cross-shape difference.',
                          'The runner owns greedy text requests; this does not qualify HTTP cancellation or MTP.']}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, allow_nan=False))


if __name__ == '__main__':
    main()
