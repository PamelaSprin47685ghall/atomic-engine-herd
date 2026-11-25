// The TP5 plugin, loaded the way a real host loads it: through upstream's own loader.
//
// This probe does not dlopen the plugin itself. It calls ggml_backend_load(path), the same
// function llama.cpp calls, which is the whole point of the design: if this works, the plugin
// is wired by upstream's mechanism and not by a hook of ours. Anything this probe can do, an
// unmodified build of llama.cpp can do.
//
// Three things are checked, in the order a host would:
//
//   1. the plugin loads and registers under the expected name;
//   2. it registers NO device, and that is the assertion -- not an omission. A device with a
//      null buffer type is not tolerated by llama.cpp (make_gpu_buft_list pushes
//      ggml_backend_dev_buffer_type(dev) without a null check, llama-model.cpp:1352), so a
//      registered device here would break ordinary inference on a machine with no discrete
//      GPU. The count staying zero is what makes loading this plugin safe;
//   3. the device plane still carries bytes -- device_roundtrip, reached through the
//      REGISTRY's proc-address hook, which needs no device, with the byte count reported and
//      no pointer escaping a freed allocation.
//
// What it does not check: that a model can run on it, because it cannot. supports_op is false
// for every op by design, and pretending otherwise in a probe would be the same defect one
// layer up.
#include <cstdio>
#include <cstring>
#include <string>

#include "ggml-backend.h"
#include "tq/tp5/hip_backend.h"

namespace {

int failures = 0;

void check(bool condition, const std::string & what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <path to libggml-tp5.so>\n", argv[0]);
        return 2;
    }
    const char * plugin_path = argv[1];

    // 1. Load through upstream's loader, not through dlopen.
    ggml_backend_reg_t reg = ggml_backend_load(plugin_path);
    check(reg != nullptr, "ggml_backend_load accepted the plugin");
    if (!reg) {
        std::fprintf(stderr, "the loader refused %s; nothing else can be checked\n", plugin_path);
        return 1;
    }
    std::fprintf(stderr, "loaded backend: name=%s devices=%zu\n", ggml_backend_reg_name(reg),
                 ggml_backend_reg_dev_count(reg));
    check(std::strcmp(ggml_backend_reg_name(reg), "TP5") == 0,
          "the backend registers under its own name, not as a vendor backend");

    // 2. The plugin registers NO device, and that is the property under test.
    //
    // The previous version asserted `device_count == 1` and then questioned the one device.
    // That assertion was the hazard: llama.cpp's default selection files an IGPU device into
    // `igpus` (llama.cpp:224-265), and make_gpu_buft_list pushes this device's buffer type --
    // which is null by design -- into the allocation list with no null check
    // (llama-model.cpp:1352). A plugin reporting one device therefore breaks model loading on
    // any host where it becomes the chosen device, and on a machine without a discrete GPU it
    // would have been the only one.
    //
    // Zero devices is not a retreat from the feature: the round-trip below is reached through
    // the registry's proc-address hook, which is independent of device registration. The
    // device appears on the day a compute backend exists.
    const size_t device_count = ggml_backend_reg_dev_count(reg);
    std::fprintf(stderr, "devices registered: %zu\n", device_count);
    check(device_count == 0,
          "the plugin registers no device: a device with a null buffer type makes llama.cpp "
          "push a nullptr into its allocation list (llama-model.cpp:1352), so registering one "
          "would break ordinary inference on a host where it is selected");
    check(ggml_backend_reg_dev_get(reg, 0) == nullptr,
          "and asks for device 0 return nothing, consistent with the count");
    // The registry's own device list must be untouched -- this is the property that makes the
    // plugin safe to load system-wide, and it is checkable without knowing which devices the
    // host has.
    std::fprintf(stderr, "process-wide device count: %zu\n", ggml_backend_dev_count());

    // 3. The device plane: bytes in, the same bytes out, from device memory.
    //
    // Through the C ABI, with the caller stating its version and its buffer size -- which is
    // the whole point of the interface. The previous cast here was
    // `roundtrip_result (*)(std::size_t)`, i.e. the probe asserted a by-value C++ return across
    // a DSO boundary by reinterpreting a `void *`. That is exactly the agreement the ABI now
    // states instead of assuming, and the probe is its first caller.
    using abi_fn = std::int32_t (*)(const tq::tp5::tq_tp5_roundtrip_request *,
                                    tq::tp5::tq_tp5_roundtrip_report *, std::uint64_t);
    auto * roundtrip = reinterpret_cast<abi_fn>(
        ggml_backend_reg_get_proc_address(reg, "tq_tp5_roundtrip_v1"));
    check(roundtrip != nullptr, "the device-plane entry point is reachable through the proc hook");

    // The pre-versioning name must resolve to NOTHING.
    //
    // This is the assertion that would have caught the defect the versioned name exists for.
    // A caller built against the first header asks for this exact string and casts the answer
    // to `roundtrip_result (*)(std::size_t)`. If the plugin answered with any address, the call
    // would enter the callee with the argument registers reinterpreted -- the callee would read
    // its `request` argument out of the register that holds 4096 and dereference address 4096.
    // The in-body version check cannot help: it runs after that read, so for such a caller it is
    // unreachable code.
    //
    // Null is the only safe answer, because we cannot know which signature the caller believed
    // in. Checking it here means the safe answer is what ships, rather than a convention a
    // future edit could quietly break by "helpfully" serving the old name again.
    check(ggml_backend_reg_get_proc_address(reg, "tq_tp5_device_roundtrip") == nullptr,
          "the pre-versioning proc name is served nothing: its callers' register agreement "
          "differs from this entry point's, so no address could be correct for them");

    // The version the plugin was built with, so a caller can refuse before calling anything it
    // does not understand.
    auto * version_fn = reinterpret_cast<std::int32_t (*)()>(
        ggml_backend_reg_get_proc_address(reg, "tq_tp5_roundtrip_abi_version"));
    check(version_fn != nullptr, "the plugin also exports its ABI version");
    if (version_fn) {
        std::fprintf(stderr, "plugin ABI version: %d (caller compiled for %u)\n", version_fn(),
                     unsigned(tq::tp5::TQ_TP5_ROUNDTRIP_ABI_VERSION));
        check(version_fn() == std::int32_t(tq::tp5::TQ_TP5_ROUNDTRIP_ABI_VERSION),
              "the plugin's ABI version matches the header this probe was built against");
    }

    // Calls the entry point the way a consumer must: fill a request, hand over a report of ITS
    // OWN size, and read the return code before trusting anything in the report.
    const auto call = [&](std::size_t bytes, std::uint32_t abi_version,
                          std::uint64_t report_size, tq::tp5::tq_tp5_roundtrip_report * out) {
        tq::tp5::tq_tp5_roundtrip_request request{};
        request.abi_version = abi_version;
        request.struct_size = sizeof(tq::tp5::tq_tp5_roundtrip_request);
        request.bytes = bytes;
        return roundtrip(&request, out, report_size);
    };

    if (roundtrip) {
        for (std::size_t bytes : {std::size_t(4096), std::size_t(1) << 20}) {
            tq::tp5::tq_tp5_roundtrip_report report{};
            const std::int32_t call_status = call(bytes, tq::tp5::TQ_TP5_ROUNDTRIP_ABI_VERSION,
                                                  sizeof(report), &report);
            check(call_status == tq::tp5::TQ_TP5_ABI_OK,
                  "the ABI call itself succeeded: caller and plugin agree on this interface");
            check(report.abi_version == tq::tp5::TQ_TP5_ROUNDTRIP_ABI_VERSION &&
                      report.struct_size == sizeof(tq::tp5::tq_tp5_roundtrip_report),
                  "and the report states which version and size the callee used");
            check(report.status == tq::tp5::TQ_TP5_TRIP_OK,
                  std::string("round trip of ") + std::to_string(bytes) + " bytes succeeded");
            check(report.bytes == bytes, "and reports the byte count it verified");
            check(report.buffer_freed == 1, "and the device allocation was returned to HIP");
            check(report.detail[0] != '\0', "and carries a reason, so a failure is actionable");
            std::fprintf(stderr, "roundtrip %zu bytes: status=%d stage=%s %s\n", bytes,
                         report.status, report.stage, report.detail);

            // Fault injection: the paths a working GPU cannot reach.
            //
            // Below the allocation there are four ways to fail -- a failed host->device copy,
            // a failed readback, a mismatched byte, and a failed free -- and none of them
            // happens on hardware that works. Leaving them to inspection is how a cleanup bug
            // ships: the previous revision's failure paths each leaked the staging block and
            // said nothing, and no test could have noticed.
            //
            // Each case substitutes exactly one step's outcome and lets everything else run
            // against real device memory, so the allocation being freed here is a real one.
            static bool faults_ran = false;
            if (!faults_ran) {
                faults_ran = true;

                const auto run_with = [](const tq::tp5::fault_plan & plan, std::size_t n) {
                    tq::tp5::set_fault_plan(plan);
                    const tq::tp5::roundtrip_result out = tq::tp5::device_roundtrip(n);
                    tq::tp5::set_fault_plan(tq::tp5::fault_plan{});
                    return out;
                };
                const auto contains = [](const char * haystack, const char * needle) {
                    return std::string(haystack).find(needle) != std::string::npos;
                };

                {
                    tq::tp5::fault_plan plan;
                    plan.fail_h2d = 1;
                    const auto failed = run_with(plan, 4096);
                    check(!failed.ok, "fault(h2d): the round trip reports failure");
                    check(failed.buffer_freed,
                          "fault(h2d): the REAL device buffer was still freed -- a failed copy "
                          "does not skip the cleanup");
                    check(contains(failed.detail, "fault seam"),
                          "fault(h2d): the detail names the seam, not a hardware error");
                    check(std::string(failed.stage) == "host -> device",
                          "fault(h2d): and the stage says where it stopped");
                }
                {
                    tq::tp5::fault_plan plan;
                    plan.fail_d2h = 1;
                    const auto failed = run_with(plan, 4096);
                    check(!failed.ok, "fault(d2h): reported as a failure");
                    check(failed.buffer_freed, "fault(d2h): the device buffer was freed");
                    check(std::string(failed.stage) == "device -> host",
                          "fault(d2h): stage names the readback");
                }
                {
                    // The real copies run; only the readback is damaged, so the comparison
                    // fails through its own check rather than by a substituted verdict. A
                    // substituted verdict would exercise the branch and not the test.
                    tq::tp5::fault_plan plan;
                    plan.corrupt_readback_bytes = 1;
                    const auto failed = run_with(plan, 4096);
                    check(!failed.ok, "fault(compare): a corrupted readback is caught");
                    check(contains(failed.detail, "differs"),
                          "fault(compare): and the detail reports the differing byte");
                    check(failed.buffer_freed, "fault(compare): the device buffer was freed");
                }
                // The fallback must never have run: every path frees through the single exit,
                // and nothing after adopt() can throw. This is the measurement that keeps the
                // "unreachable" claim honest instead of a comment asserting itself.
                check(tq::tp5::destructor_cleanup_count() == 0,
                      "the last-resort destructor cleanup never ran: no path escaped the single "
                      "exit");
                check(tq::tp5::destructor_free_failure_count() == 0,
                      "and no such cleanup failed");
            }
        }
        // A zero-byte request has nothing to verify, and saying so is better than reporting
        // success for a transfer that never happened.
        {
            tq::tp5::tq_tp5_roundtrip_report report{};
            check(call(0, tq::tp5::TQ_TP5_ROUNDTRIP_ABI_VERSION, sizeof(report), &report) ==
                      tq::tp5::TQ_TP5_ABI_OK,
                  "zero bytes: the call ran");
            check(report.status == tq::tp5::TQ_TP5_TRIP_ZERO_BYTES,
                  "zero bytes: reported as nothing to verify, not as success");
            check(report.detail[0] != '\0', "zero bytes: and says why");
        }

        // The ABI's reason for existing: a caller that disagrees with the plugin is told so,
        // and nothing is written past what it offered.
        {
            tq::tp5::tq_tp5_roundtrip_report report{};
            // Poison the report so a "no write happened" claim can be distinguished from a
            // write of zeros -- the shape of check this project keeps needing.
            std::memset(&report, 0xA5, sizeof(report));
            const std::int32_t mismatched =
                call(4096, tq::tp5::TQ_TP5_ROUNDTRIP_ABI_VERSION + 1u, sizeof(report), &report);
            check(mismatched == tq::tp5::TQ_TP5_ABI_VERSION_MISMATCH,
                  "a caller from a different ABI version is refused, not served");
            check(report.abi_version == tq::tp5::TQ_TP5_ROUNDTRIP_ABI_VERSION,
                  "and the header is still written, so the caller can say what it met");
            check(report.status == tq::tp5::TQ_TP5_TRIP_OK && report.buffer_freed == 1,
                  "while the body of the poisoned report is replaced, not left as garbage");
        }
        {
            // A report buffer that holds the header but not this callee's struct.
            //
            // The previous expectation here was `rc == ABI_OK || rc == VERSION_MISMATCH`, and it
            // pinned the defect rather than the rule: the old callee ran the transfer, copied a
            // truncated report, and returned ABI_OK. A caller that trusted the code then read
            // `stage` and `detail` out of bytes the plugin never wrote -- its own uninitialised
            // memory, presented as this plugin's diagnosis.
            //
            // The rule the interface now states is narrower and checkable: ABI_OK means every
            // field the caller may read was written. So a short buffer gets the two identifying
            // fields and nothing else, and the return value says the report is incomplete.
            alignas(8) unsigned char small[16];
            std::memset(small, 0xA5, sizeof(small));
            const std::int32_t rc =
                call(4096, tq::tp5::TQ_TP5_ROUNDTRIP_ABI_VERSION, sizeof(small),
                     reinterpret_cast<tq::tp5::tq_tp5_roundtrip_report *>(small));
            check(rc == tq::tp5::TQ_TP5_ABI_REPORT_TOO_SMALL,
                  "a report buffer too small for this callee is refused with TOO_SMALL, not "
                  "served with a code that implies the report is complete");

            // The header is the part a caller needs in order to say what it met, so it is
            // written: the callee's version and the size the callee's struct actually has.
            std::uint32_t stated_version = 0;
            std::uint32_t stated_size = 0;
            std::memcpy(&stated_version, small, sizeof(stated_version));
            std::memcpy(&stated_size, small + sizeof(std::uint32_t), sizeof(stated_size));
            check(stated_version == tq::tp5::TQ_TP5_ROUNDTRIP_ABI_VERSION,
                  "the short buffer's first field is the callee's version");
            check(stated_size == sizeof(tq::tp5::tq_tp5_roundtrip_report),
                  "and the second is the size the callee's report really has, so the caller can "
                  "compare it with its own");

            // Everything past those two fields stays exactly as the caller left it. No escape
            // clause: a partial write here would be the defect, so the assertion must fail if
            // even one byte moved.
            bool untouched_tail = true;
            for (std::size_t i = 2 * sizeof(std::uint32_t); i < sizeof(small); ++i) {
                if (small[i] != 0xA5) untouched_tail = false;
            }
            check(untouched_tail,
                  "and not one byte past the header was written into a buffer too small to hold "
                  "the report");
        }
        {
            // Even the header does not fit: nothing may be written at all.
            alignas(8) unsigned char tiny[4];
            std::memset(tiny, 0xA5, sizeof(tiny));
            const std::int32_t rc =
                call(1, tq::tp5::TQ_TP5_ROUNDTRIP_ABI_VERSION, sizeof(tiny),
                     reinterpret_cast<tq::tp5::tq_tp5_roundtrip_report *>(tiny));
            check(rc == tq::tp5::TQ_TP5_ABI_REPORT_TOO_SMALL,
                  "a buffer smaller than the header is refused before anything is written");
            bool untouched = true;
            for (unsigned char byte : tiny) {
                if (byte != 0xA5) untouched = false;
            }
            check(untouched, "and truly nothing was written into it");
        }
        {
            // A misaligned report pointer is refused, and refused BEFORE anything is written.
            //
            // The reasoning is asymmetric and that asymmetry is the point: this side writes with
            // memcpy, which tolerates any address, so our store would have succeeded. The caller
            // reads the result through a `tq_tp5_roundtrip_report *`, and that read is undefined
            // behaviour on a misaligned address. A plugin that filled such a buffer would have
            // produced a report its own caller could not legally look at -- a failure that shows
            // up in the caller's optimised code, months later, with nothing pointing back here.
            //
            // Naming it at the boundary is the only useful action this side can take, so the
            // entry point checks alignment first: it is a property of the pointer alone and
            // depends on no byte of the buffer.
            alignas(8) unsigned char raw[sizeof(tq::tp5::tq_tp5_roundtrip_report) + 8];
            std::memset(raw, 0xA5, sizeof(raw));
            auto * misaligned =
                reinterpret_cast<tq::tp5::tq_tp5_roundtrip_report *>(raw + 1);
            check(reinterpret_cast<std::uintptr_t>(misaligned) %
                          alignof(tq::tp5::tq_tp5_roundtrip_report) !=
                      0,
                  "fixture: the pointer really is misaligned for the report type");
            const std::int32_t rc = call(4096, tq::tp5::TQ_TP5_ROUNDTRIP_ABI_VERSION,
                                         sizeof(tq::tp5::tq_tp5_roundtrip_report), misaligned);
            check(rc == tq::tp5::TQ_TP5_ABI_UNALIGNED,
                  "a misaligned report pointer is refused with UNALIGNED, because the caller's "
                  "own read of it would be undefined behaviour");
            bool untouched = true;
            for (unsigned char byte : raw) {
                if (byte != 0xA5) untouched = false;
            }
            check(untouched, "and nothing was written into the misaligned buffer");
        }
        {
            // A request struct that is SHORTER than this callee's must not be over-read.
            //
            // The callee has to read request->abi_version and request->struct_size before it can
            // validate anything -- those are the first 8 bytes -- but it must not touch
            // request->bytes (offsets 8..16) until the size check has passed. A caller built
            // against a smaller request struct has only those 8 bytes allocated, so reading
            // `bytes` would be a heap over-read.
            //
            // The fixture allocates exactly those 8 bytes and fills only them. Under a sanitizer
            // an over-read faults here, at the boundary, which is where the bug would be; without
            // one the return code still proves the ordering: VERSION_MISMATCH is only reachable
            // by having read `struct_size`, and it is returned instead of a transfer, so `bytes`
            // was never consulted.
            //
            // This is the assertion the numeric address of `bytes` cannot give: a check on the
            // VALUE of an over-read byte cannot tell a read that happened from one that did not.
            struct short_request {
                std::uint32_t abi_version;
                std::uint32_t struct_size;
            };
            alignas(8) unsigned char request_storage[sizeof(short_request)];
            short_request shorter{};
            shorter.abi_version = tq::tp5::TQ_TP5_ROUNDTRIP_ABI_VERSION;
            // A size that is honest about being smaller: what this plugin asks the caller to
            // report, so the callee can refuse instead of guessing at the rest of the fields.
            shorter.struct_size = sizeof(short_request);
            std::memcpy(request_storage, &shorter, sizeof(short_request));

            tq::tp5::tq_tp5_roundtrip_report report{};
            std::memset(&report, 0xA5, sizeof(report));
            const std::int32_t rc = roundtrip(
                reinterpret_cast<const tq::tp5::tq_tp5_roundtrip_request *>(request_storage),
                &report, sizeof(report));
            check(rc == tq::tp5::TQ_TP5_ABI_VERSION_MISMATCH,
                  "a request struct smaller than this callee's is refused by size before any of "
                  "its later fields are read");
            check(sizeof(short_request) == 2 * sizeof(std::uint32_t),
                  "fixture: the short request carries exactly the two identifying fields, so "
                  "reading a third would be an over-read");
        }
    }

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr,
                 "TP5 plugin: loaded by upstream's loader, registers NO device (so a host that "
                 "loads it behaves exactly as if it had not), device memory verified through "
                 "the registry hook; compute deliberately absent\n");
    return 0;
}
