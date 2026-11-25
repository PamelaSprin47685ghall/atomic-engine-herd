// The release rule, exercised on the configurations the host cannot produce.
//
// The rule used to be `ggml_backend_sched_get_n_backends(sched) > 1`, under a comment that
// said the gate was "one buffer type". Two different predicates, and the code was the wrong
// one in both directions:
//
//   a single CUDA backend        passes n_backends == 1 and was never measured
//   two CPU backends, one buft   fails n_backends > 1 although the retry path in
//                                ggml_backend_sched_alloc_splits cannot tell them from one
//
// Neither the GPU case nor the two-buffer-type case can be built on this host, so a test
// that went through the device query would cover exactly one row of the table. This one
// calls the pure function, which is why the function exists as a separate thing rather than
// as an `if` inside release_compute().
//
// herd_phase.hpp is included, not herd_phase_station.hpp: the rule is declared in the
// phase header because it is part of the release's contract, while its definition sits in
// the station TU next to the other pure logic. The station header deliberately knows
// nothing about the release, and pulling the rule in there would have widened it.
#include "herd_phase.hpp"

#include <cstdio>
#include <cstring>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const std::string & what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

herd::phase::release_facts facts(bool known, int backends, int bufts, bool all_cpu,
                                 int null_backends = 0, int null_bufts = 0) {
    herd::phase::release_facts f;
    f.scheduler_known = known;
    f.n_backends = backends;
    f.n_buffer_types = bufts;
    f.all_backends_cpu = all_cpu;
    f.n_null_backends = null_backends;
    f.n_null_buffer_types = null_bufts;
    return f;
}

void a_refusal_names_its_cause() {
    // Every refusal carries a sentence. A null reason on a refused verdict would reach the
    // caller as "release_compute returned false" -- the message that hides the cause, which
    // is what the reason field exists to prevent.
    const herd::phase::release_facts refused[] = {
        facts(false, 1, 1, true),    // no scheduler
        facts(true, 0, 0, true),     // no backend
        facts(true, 1, 1, false),    // one GPU backend
        facts(true, 1, 2, true),     // two buffer types
    };
    for (const auto & f : refused) {
        const auto v = herd::phase::decide_release(f);
        check(!v.allowed, "the configuration is refused");
        check(v.reason != nullptr && std::strlen(v.reason) > 0,
              "a refusal names its cause instead of leaving null");
    }
}

void the_measured_configuration_is_allowed() {
    const auto v = herd::phase::decide_release(facts(true, 1, 1, true));
    check(v.allowed, "one CPU backend, one buffer type: the measured case is allowed");
    check(v.reason == nullptr, "an allowed verdict carries no reason to explain away");
}

void one_gpu_backend_is_not_the_measured_case() {
    // The defect this test was written for. n_backends == 1 was treated as "single buffer
    // type, therefore measured"; a lone CUDA device satisfies it exactly.
    const auto v = herd::phase::decide_release(facts(true, 1, 1, false));
    check(!v.allowed, "a single non-CPU backend is refused (n_backends == 1 is not CPU)");
    check(v.reason && std::strstr(v.reason, "non-CPU") != nullptr,
          "and says the reason is the backend kind, not the count");
}

void several_cpu_backends_are_refused_as_unmeasured() {
    // This row used to be ALLOWED, on a sound argument: two CPU backends over one buffer type
    // go through the same retry path in ggml_backend_sched_alloc_splits as a single backend, so
    // the mechanism is identical. The argument is still sound and the row is now refused,
    // because sound is not measured -- nothing on this host can build a two-backend scheduler,
    // so allowing it would have shipped an unexercised branch inside the function whose entire
    // purpose is to keep unexercised configurations away from a memory-freeing path. When
    // somebody measures it on real hardware, this assertion is the one to invert, and the
    // buffer-type check below is the condition that must still hold when they do.
    const auto v = herd::phase::decide_release(facts(true, 2, 1, true));
    check(!v.allowed, "two CPU backends sharing one buffer type are refused: unmeasured");
    check(v.reason && std::strstr(v.reason, "more than one backend") != nullptr,
          "and the reason is the backend count, not the buffer type");
}

void a_null_backend_fails_closed() {
    // The defect this test was written for. The gather loop used to `continue` past a null
    // slot, leaving all_backends_cpu true and n_buffer_types at the count of the slots that
    // answered -- so a scheduler it could not fully read arrived here looking like a verified
    // single-type CPU scheduler, and the release was allowed on facts nobody observed.
    //
    // The numbers below are exactly what that loop produced for such a scheduler: one backend,
    // one buffer type, all CPU. Only the count of unread slots distinguishes it from the
    // measured case, which is why the count had to become a fact.
    const auto hidden = herd::phase::decide_release(facts(true, 1, 1, true, /*null_backends=*/1));
    check(!hidden.allowed, "a null backend slot is refused even when the other facts look clean");
    check(hidden.reason && std::strstr(hidden.reason, "null backend") != nullptr,
          "and names the unread slot, not a downstream fact");

    const auto no_buft = herd::phase::decide_release(facts(true, 1, 1, true, /*nb=*/0, /*nbuft=*/1));
    check(!no_buft.allowed, "a null buffer type is refused: no allocator can be named");
    check(no_buft.reason && std::strstr(no_buft.reason, "buffer type") != nullptr,
          "with that cause");
}

void several_buffer_types_are_refused() {
    // Synthesised: after the one-backend rule this row cannot arise from a real scheduler, since
    // one backend reports one buffer type. It is kept because it is the mechanism's own limit --
    // the condition that must still hold if the one-backend refusal is ever relaxed -- and a
    // rule that no longer has a test is a rule the next change is free to delete.
    const auto v = herd::phase::decide_release(facts(true, 1, 2, true));
    check(!v.allowed, "more than one buffer type is refused, even with a single backend");
    check(v.reason && std::strstr(v.reason, "buffer type") != nullptr,
          "and the reason names the buffer types, which is the mechanism's actual limit");
}

void a_missing_scheduler_is_refused_before_anything_else() {
    // Order matters for the message: telling an operator "non-CPU backend" about a context
    // that never decoded would send them to look at their hardware.
    const auto v = herd::phase::decide_release(facts(false, 0, 0, false));
    check(!v.allowed, "an unknown scheduler is refused");
    check(v.reason && std::strstr(v.reason, "never been seen") != nullptr,
          "and the first cause reported is the missing scheduler, not a downstream one");
}

void no_backend_is_refused() {
    const auto v = herd::phase::decide_release(facts(true, 0, 0, true));
    check(!v.allowed, "a scheduler with no backend is refused");
    check(v.reason && std::strstr(v.reason, "no backend") != nullptr, "with that cause");
}

} // namespace

int main() {
    the_measured_configuration_is_allowed();
    one_gpu_backend_is_not_the_measured_case();
    several_cpu_backends_are_refused_as_unmeasured();
    a_null_backend_fails_closed();
    several_buffer_types_are_refused();
    a_missing_scheduler_is_refused_before_anything_else();
    no_backend_is_refused();
    a_refusal_names_its_cause();

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr, "release rule: CPU-only, one buffer type, every refusal named\n");
    return 0;
}
