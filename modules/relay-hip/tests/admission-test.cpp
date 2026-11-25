// The plugin's admission rule, on the machines the host cannot become.
//
// ggml declines a plugin at load time when its score is 0 (ggml-backend-reg.cpp:229), and it
// does so BEFORE calling ggml_backend_init. That ordering is the whole reason a machine with no
// GPU is safe to load the plugin on: nothing is registered, so nothing reports a backend with
// zero devices.
//
// This host has one GPU, so the "no device" row cannot be produced here. It does not have to
// be produced: score_from() takes the facts as a struct and returns the score, and the facts
// are trivially constructible. A rule that can only be exercised on the hardware it guards is
// a rule nobody tests -- the same reason the phase release decision was extracted.
//
// What this test does NOT establish: that HIP reports zero devices on a host without ROCm. That
// is HIP's behaviour, not this module's, and asserting it here would be asserting the runtime's
// contract from the outside. What it establishes is that when HIP does report no device, this
// plugin refuses to be loaded.
#include "tq/tp5/hip_backend.h"

#include <cstdio>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const std::string & what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

tq::tp5::hip_device_info device(bool present, int count) {
    tq::tp5::hip_device_info info;
    info.device_present = present;
    info.hip_device_count = count;
    info.memory_total_bytes = present ? (std::size_t(8) << 30) : 0;
    info.memory_free_bytes = info.memory_total_bytes;
    return info;
}

} // namespace

int main() {
    // The admission rule, in both directions, plus the two ways a device can be absent.
    check(tq::tp5::score_from(device(false, 0)) == 0,
          "no device: score 0, so the loader declines the plugin and registers nothing");
    check(tq::tp5::score_from(device(true, 0)) == 0,
          "a presence flag with a zero count is still 0: the count is the fact, not the flag");
    check(tq::tp5::score_from(device(false, 3)) == 0,
          "absent overrides a stale count: the two fields disagreeing is not permission");

    const int scored = tq::tp5::score_from(device(true, 1));
    check(scored > 0, "one device: a non-zero score, so the loader keeps the plugin");
    check(scored == tq::tp5::score_from(device(true, 5)),
          "the score does not grow with the device count: a plugin is admitted or not, and "
          "outbidding a vendor backend on the same hardware is not this plugin's decision");

    // A zero-byte round trip is refused rather than reported as a success. It is checked here
    // as well as in the plugin probe because it is the shape of claim this project keeps
    // having to remove: "it returned without error" is not "it verified something".
    const tq::tp5::roundtrip_result zero = tq::tp5::device_roundtrip(0);
    check(!zero.ok, "a zero-byte round trip reports failure, not success");
    check(zero.bytes == 0, "and reports the size it was asked for");

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr,
                 "TP5 admission: 0 without a device, non-zero with one; a zero-byte transfer "
                 "is not a success\n");
    return 0;
}
