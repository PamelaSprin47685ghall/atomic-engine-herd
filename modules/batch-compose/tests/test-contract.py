#!/usr/bin/env python3
"""Argument contract of tq-batch-compose.

Two fork features meet here, and they are in different states -- which is exactly
what this file has to keep straight:

  * releasing compute memory at the prompt-to-generation transition is still
    unavailable, so the default (which used to release) is refused with a message
    that names the way out;
  * a physical KV capacity larger than the logical context IS available now: it is
    applied by a scoped request around context creation, and the cache reports the
    larger pool while the logical window stays put.

The second one changed the contract, so this test changed with it. It used to
assert that any --total-kv differing from --ctx-size was refused; that was true
when nothing could honour it, and keeping it would have pinned the capability to
its absence. What survives is the part that is still a *safety* contract rather
than a capability one:

  * a capacity BELOW the window is refused: it would withhold cells the context
    legitimately uses. That is not a missing feature, it is a bad request;
  * equal capacities are the ordinary configuration;
  * a capacity above the window passes argument validation -- whether the cache
    can honour it is decided by the context, not by the parser, and asserting
    otherwise here would be asserting the wrong layer.

Without --model every check runs before the model is read, so no GGUF is needed:
a nonexistent path is enough, and its failure doubles as proof that argument
validation finished. With --model the file additionally checks the capability
end to end: the config event must report the decoupled capacity, and two requests
that could not be resident together at the window size must become resident
together -- which is the whole point of the feature, and is invisible in a single
request.
"""

import argparse
import json
import subprocess
import sys

REFUSED_PHASE = "phase-memory seam"
REFUSED_WINDOW = "cache below the logical window"


def run(binary, args):
    completed = subprocess.run([binary, *args], capture_output=True, text=True)
    return completed.returncode, (completed.stdout or "") + (completed.stderr or "")


def expect(condition, message, output, failures):
    if not condition:
        failures.append(f"{message}\n--- output ---\n{output.strip()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    # Optional: with a model the capability itself is checked, not only the parser.
    parser.add_argument("--model", default="")
    args = parser.parse_args()

    failures = []
    missing_model = ["--model", "/nonexistent/herd-test.gguf", "--prompts", "/dev/null"]

    # 1. The fork's default released compute memory, and this build can do it again, so
    #    the default is ACCEPTED at the argument layer.
    #
    #    These two cases used to assert a refusal, because there was nothing to accept:
    #    the seam was missing, so the parser refused before main could run. The refusal is
    #    gone and pinning it back would pin the feature to its absence. What is still worth
    #    asserting without a model is the ORDER of the layers: the run must get past
    #    argument validation and fail on the model. That is observable, it is the same
    #    property section 5 relies on, and it is the only thing this parser can be asked
    #    about a capability it no longer has an opinion on.
    #
    #    Refusing to run a *nonexistent model* is not a degradation: the release happens at
    #    prompt->generation, and a run that never loads a model never reaches it.
    rc, out = run(args.binary, missing_model)
    expect(rc != 0, "default invocation must fail (there is no model)", out, failures)
    expect(REFUSED_PHASE not in out,
           "the default must reach model loading: the release capability is present, so a "
           "phase-memory refusal here would be the absence being pinned again", out, failures)

    # 2. An explicit --phase-memory is the same acceptance, not a different answer.
    rc, out = run(args.binary, [*missing_model, "--phase-memory"])
    expect(rc != 0, "--phase-memory must still fail on the model", out, failures)
    expect(REFUSED_PHASE not in out, "--phase-memory must not be refused", out, failures)

    # 3. Opting into the supported mode gets past argument validation: the run
    #    then fails on the model, which is what proves the guard is not a wall.
    rc, out = run(args.binary, [*missing_model, "--no-phase-memory"])
    expect(rc != 0, "--no-phase-memory reaches model loading", out, failures)
    expect(REFUSED_PHASE not in out, "--no-phase-memory must not be refused", out, failures)

    # 4. Below the window: a safety refusal, not a capability one. The message has
    #    to say WHY, because "unsupported" would send the operator to look for a
    #    feature that exists.
    rc, out = run(args.binary, [*missing_model, "--no-phase-memory", "--ctx-size", "64", "--total-kv", "32"])
    expect(rc != 0, "--total-kv below the context must fail", out, failures)
    expect(REFUSED_WINDOW in out, "the refusal must say the cache is below the window", out, failures)
    expect("--ctx-size" in out, "and must name the number to compare against", out, failures)

    # 5. Above the window: argument validation passes. The run then fails because the
    #    model does not exist, which is what proves the parser let it through -- the
    #    layer that decides whether a cache can hold that capacity is the context, not
    #    this one. Asserting a refusal here would pin the feature to its absence.
    rc, out = run(args.binary, [*missing_model, "--no-phase-memory", "--ctx-size", "64", "--total-kv", "128"])
    expect(rc != 0, "--total-kv above the context still fails without a model", out, failures)
    expect(REFUSED_WINDOW not in out, "a larger capacity must not be refused as below the window", out, failures)
    expect("could not load the model" in out,
           "a larger capacity must reach model loading, which is where the parser's job ends", out, failures)

    # 6. Equal capacities are one capacity: the ordinary configuration.
    rc, out = run(args.binary, [*missing_model, "--no-phase-memory", "--ctx-size", "64", "--total-kv", "64"])
    expect(rc != 0, "equal --total-kv reaches model loading", out, failures)
    expect(REFUSED_WINDOW not in out, "equal --total-kv must not be refused", out, failures)

    # 7. Omitting --total-kv is not the same request as passing it, and neither is
    #    refused: the default is what the runner sizes for its largest concurrent set.
    rc, out = run(args.binary, [*missing_model, "--no-phase-memory", "--ctx-size", "64"])
    expect(rc != 0, "omitting --total-kv reaches model loading", out, failures)
    expect(REFUSED_WINDOW not in out, "omitting --total-kv must not be refused", out, failures)

    # 8. Help has to describe the mode the binary actually supports.
    rc, out = run(args.binary, ["--help"])
    expect(rc == 0, "--help exits 0", out, failures)
    expect("--no-phase-memory" in out, "--help mentions --no-phase-memory", out, failures)
    expect("--total-kv" in out, "--help mentions --total-kv", out, failures)

    # ---- end to end, when a model is supplied ----
    #
    # Everything above is a parser contract. The capability itself needs a context and
    # a decode, and its evidence is not "it did not fail": it is that two requests which
    # cannot be resident together at the window size become resident together with the
    # larger pool. A single-request run cannot show that difference at all, which is why
    # this was missing until now.
    if args.model:
        e2e(args.binary, args.model, failures)
    else:
        print("no --model given: the decoupled-capacity checks did not run", file=sys.stderr)

    if failures:
        print("\n\n".join(failures), file=sys.stderr)
        print(f"\n{len(failures)} contract violation(s)", file=sys.stderr)
        return 1
    print("batch-compose argument contract holds")
    return 0


def events(binary, model, prompts, extra):
    """Runs the runner with --json and returns (returncode, event list, raw text)."""
    completed = subprocess.run(
        [binary, "--model", model, "--prompts", prompts, "--no-phase-memory", "--json", *extra],
        capture_output=True, text=True, timeout=300)
    parsed = []
    for line in (completed.stdout or "").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                parsed.append(json.loads(line))
            except ValueError:
                pass  # a non-JSON line is not an event; the raw text is still reported
    return completed.returncode, parsed, (completed.stdout or "") + (completed.stderr or "")


def peak_reserved(sequence):
    """The largest total_reserved any admit event reported, or None if none.

    This is the residency claim stated as the number the runner itself publishes:
    each `admit` event carries `total_reserved`, the tokens committed across live
    requests at that moment. Two requests resident at once means that figure exceeds
    one request's share; serialised residency never does.

    Read from the counter rather than from event ordering on purpose. The previous
    version compared the index of the second admit against the first release, which is
    a property of scheduling: with two requests starting together, which one is
    admitted first and whether one finishes before the other is considered depends on
    per-request token counts, and it was observed to differ between runs of the same
    command. The counter cannot flip like that -- it either exceeded the single-request
    share or it did not.
    """
    values = [e.get("total_reserved") for e in sequence
              if e.get("event") == "admit" and isinstance(e.get("total_reserved"), int)]
    return max(values) if values else None


def result_count(sequence):
    return sum(1 for e in sequence if e.get("event") == "result")


def e2e(binary, model, failures):
    import os
    import tempfile

    def expect(condition, message, output=""):
        if not condition:
            failures.append(f"{message}\n--- output ---\n{str(output).strip()}")

    with tempfile.TemporaryDirectory() as work:
        prompts = os.path.join(work, "prompts.txt")
        # Two requests that each need the whole window: at a 256-cell capacity only one
        # can be resident, so the runner must release one before admitting the other.
        with open(prompts, "w", encoding="utf-8") as handle:
            handle.write("a b c d\nb c d a\n")

        base = ["--ctx-size", "256", "--parallel", "2", "--predict", "160"]

        # 1. Auto-expansion: the previous branch's contract, restored and reported.
        #
        # Two requests at 160 predicted tokens each need ~352 cells, and --ctx-size 256
        # provides 256. The fork sized its cache for the resident set without being asked
        # (fork/batch-compose/src/main.cpp:252-259), so this run must succeed -- a guard
        # that refused it was a feature removed, not prudence. What the guard was right
        # about is visibility, and that is what is asserted instead: the expansion has to
        # be named in the config event, not left to be inferred from two numbers.
        rc, seq, out = events(binary, model, prompts, base)
        expect(rc == 0, "a resident set larger than the window runs, as the fork did", out)
        config = next((e for e in seq if e.get("event") == "config"), {})
        expect(config.get("kv_auto_expanded") is True,
               "and the automatic expansion is reported, not silent", out)
        expect(config.get("kv_decoupled") is True,
               "the pool really is larger than the window", out)
        expect(int(config.get("total_kv_actual", 0)) > int(config.get("logical_context_actual", 0)),
               f"the pool exceeds the window: {config.get('total_kv_actual')} vs "
               f"{config.get('logical_context_actual')}", out)
        expect(result_count(seq) == 2,
               "both requests complete, which is what the larger pool is for", out)

        # 2. The control: the same work with the capacity stated explicitly. Equal to the
        # window, so nothing is decoupled -- the runner simply has 256 cells to share, and
        # two requests of this size cannot both be resident in them.
        rc, seq, out = events(binary, model, prompts, [*base, "--total-kv", "256"])
        expect(rc == 0, "the same work runs when the capacity is asked for", out)
        config = next((e for e in seq if e.get("event") == "config"), {})
        expect(config.get("kv_decoupled") is False,
               "a capacity equal to the window is not decoupled", out)
        expect(config.get("total_kv_actual") == 256,
               f"and the pool is the window's own size, got {config.get('total_kv_actual')}", out)
        expect(config.get("logical_context_actual") == 256,
               f"with the window unchanged at 256, got {config.get('logical_context_actual')}", out)
        expect(result_count(seq) == 2, "both requests complete", out)
        peak = peak_reserved(seq)
        expect(peak is not None and peak <= 256,
               f"at 256 cells the two requests cannot be resident together (peak {peak})", out)
        # Deliberately NOT asserted: which request was admitted first. That is a property
        # of scheduling -- observed to differ between runs of the same command -- so both
        # residency claims above are read from `total_reserved`, the number the runner
        # publishes, which cannot flip with timing.

        rc, seq, out = events(binary, model, prompts, [*base, "--total-kv", "512"])
        expect(rc == 0, "the decoupled run succeeds", out)
        config = next((e for e in seq if e.get("event") == "config"), {})
        expect(config.get("kv_decoupled") is True,
               "the run reports a decoupled cache", out)
        expect(config.get("total_kv_actual") == 512,
               f"the cache is the requested size, got {config.get('total_kv_actual')}", out)
        # The claim in one line: the cache grew and the window did not. Both figures
        # come from the context -- llama_n_ctx_seq() and the runner's own capacity
        # request -- so this is the decoupling itself, not a restatement of the flag.
        expect(config.get("logical_context_actual") == 256,
               f"the logical window stays 256 while the cache is 512, got "
               f"{config.get('logical_context_actual')}", out)
        expect(result_count(seq) == 2, "both requests complete with the larger pool", out)
        peak = peak_reserved(seq)
        expect(peak is not None and peak > 256,
               f"with 512 cells both requests are resident at once (peak {peak})", out)

        # The padded-window case, and the regression guard for a defect that shipped for
        # one round: a capacity request is compared against the window the context will
        # actually have, which upstream pads to a multiple of 256 (llama-context.cpp:456).
        #
        # Asking for --ctx-size 64 --total-kv 128 is a request BELOW the padded window of
        # 256, so nothing needs widening and the run must succeed with kv_decoupled false.
        # The version that compared against the requested --ctx-size opened a 128-cell
        # request, the hook saw 128 <= 256 and declined, and the run aborted with "this
        # memory shape cannot be decoupled" -- a wrong diagnosis of a legal configuration.
        #
        # No model-free check can catch it: the parser accepts the flags and the failure
        # needs a real context. That is why it lives here.
        small = ["--ctx-size", "64", "--parallel", "2", "--predict", "8"]
        rc, seq, out = events(binary, model, prompts, [*small, "--total-kv", "128"])
        expect(rc == 0,
               "a capacity below the padded window is honoured, not refused", out)
        config = next((e for e in seq if e.get("event") == "config"), {})
        expect(config.get("kv_decoupled") is False,
               "and it is reported as the ordinary, undecoupled cache", out)
        expect(config.get("logical_context_actual") == 256,
               f"the window is upstream's padded 256, got "
               f"{config.get('logical_context_actual')}", out)
        expect(config.get("total_kv_actual") == 256,
               f"the pool is that same 256, got {config.get('total_kv_actual')}", out)


if __name__ == "__main__":
    sys.exit(main())
