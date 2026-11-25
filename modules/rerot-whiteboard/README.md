# rerot-whiteboard — the shared board a body step publishes to

The RERoT lane's ledger. One `station`, fixed stage count, and `step`s that publish
all-or-nothing. No llama, no ggml, no sequence handle: which stage is bound to which sequence
is LEMMA 3's mapping and stays the caller's business, so this compiles and its rules are
measured without a model.

## The three sentences it implements

| Source | Rule | Where it is enforced |
|---|---|---|
| AXIOM 1 | each insertion point reads its peers' **last committed** token | `step::peers_of`, plus the epoch check in `commit` — a token computed from older peers is refused, not merged |
| AXIOM 3 | a failed forward must not publish **part** of an update | `commit` validates every submission, then writes every submission, under one lock |
| LEMMA 2 | one body step advances each **bound** stage by one token | one `step` per body step; a commit that lacks a bound stage is refused (`missing_stage`) and tainted steps cannot publish (`refused_submission`) |

## Shape

```cpp
tq::rerot::station board(/*n_stages=*/2);   // every stage bound
tq::rerot::station episode(/*n_stages=*/4, {0, 2});   // only 0 and 2 are insertion points

tq::rerot::step s = board.open();          // pins the epoch this step reads through
auto peers = s.peers_of(my_stage);         // last committed tokens, self excluded
int32_t next = compute(peers);
s.submit(my_stage, next);
switch (s.commit()) {
    case tq::rerot::step_status::ok:          break;
    case tq::rerot::step_status::stale_epoch: /* a peer committed; recompute */ break;
    default: throw std::runtime_error(tq::rerot::describe(status));
}
```

Destroying an open step **discards** it and counts the discard. An early return, a throw, or a
forgotten step cannot publish; it also cannot pass unnoticed, because `station::discards()`
reports it.

## Two APIs that exist because a test found them

**`snapshot()`.** The first version of the concurrency test compared stage 0 with stage 1
through two `peek` calls and reported a torn commit in roughly one run out of three. The board
was not at fault: two calls are two points in time, so a commit between them yields stage 0
from epoch *N* and stage 1 from epoch *N+1*. The publish was atomic; the comparison was not.
`snapshot()` reads every stage under one lock, which is the observation-side counterpart of
the commit's atomicity.

**`token_view` instead of `int32`.** AXIOM 1 is about the last *committed* token, which does
not exist before the first commit. A sentinel would make "no token yet" and "token 0" the same
answer, and a peer that cannot tell them apart starts its completion with the wrong token.

## Lifetime and threading

A `step` holds the board it came from, so a step that outlives its station is still safe to
read and to close: the board's lifetime is the last handle to it, not the station's. A
`station` is a handle, not an object with identity — copies name the same board.

Every read and every commit takes one mutex. Reads through a step are the ones AXIOM 1 needs,
because their commit validates them; reads outside a step are a hint. Both see the same data;
only the conclusion differs.

## Verification

- `tests/whiteboard-test.cpp` — 8 contract cases plus a 4-writer/1-reader concurrency case:
  both stages always written with the same value, so any observation of two different valid
  tokens would be a published mix. Result: 0 torn observations over ~1500–2400 snapshots,
  800/800 intended commits landed, `refusals == stale retries`.
- Two mutations of a **copy** in `/tmp` (epoch bump removed; stale check disabled) each fail
  5 and 6 assertions respectively — the test is not vacuous. The repository source was not
  modified for either.
- ThreadSanitizer run of the same test: no race reported.

## Not here

- No wiring into `rerot-compose` and no claim that reader inference uses this board. The
  reader's graph emission (`modules/reader`) is separately unimplemented, and this station is
  not a substitute for it.
- No retry policy, no timeout, no backoff: `stale_epoch` is reported, and recomputing is the
  caller's decision.
- No persistence: the board is process memory, as the in-process whiteboard always was.
