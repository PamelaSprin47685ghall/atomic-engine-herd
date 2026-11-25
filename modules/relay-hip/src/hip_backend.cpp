// The TP5 HIP device plane as a ggml backend plugin.
//
// Read hip_backend.h first: it says why this is a plugin rather than a patched translation
// unit, and what this file deliberately does not do (compute).
//
// Three claims this file makes, each with the call that backs it:
//
//   1. the device is real      hipGetDeviceCount / hipGetDeviceProperties / hipMemGetInfo
//   2. the memory is device memory, and bytes survive the trip
//                              hipMalloc / hipMemcpy / hipFree, verified by comparing what
//                              comes back with what went in
//   3. the arithmetic is absent
//                              supports_op returns false for every op, and init_backend
//                              returns nullptr with a message. A backend that accepted work
//                              it cannot do would fail inside a decode, far from here.
//
// The third is not modesty. llama.cpp schedules by supports_op; a plugin that said yes would
// take the transformer's nodes and then return GGML_STATUS_FAILED from a graph compute with
// no kernels behind it. Absent and honest beats present and broken.
#include "tq/tp5/hip_backend.h"

// <atomic>: the fault seam and the destructor counters.
// <algorithm>: std::min, for clamping the corruption seam to the buffer.
// <new>: std::bad_alloc, which the host-allocation guard catches so that a failed allocation
//        is reported in the result instead of unwinding through a live device allocation.
#include <algorithm>
#include <atomic>
#include <limits>
#include <new>

#include <cstdio>
#include <cstring>
#include <vector>

#include <hip/hip_runtime.h>

// The interface tables live in upstream's backend implementation header -- the one every
// backend in the tree includes (ggml-blas.cpp, ggml-cuda.cu, ggml-metal.m all do). It is a
// header a backend implements against, not private state of another module.
#include "ggml-backend-impl.h"

namespace tq::tp5 {

// The device plane's one measurable operation, defined before the interface tables that
// expose it through the proc-address hook.
//
// What it proves is exactly what a relay staging buffer needs: bytes the host writes arrive
// on the device unchanged, and bytes read back are the bytes that were there. It does not
// prove the collective works -- that is modules/relay-transport's business, on host memory,
// and the device-side kernels are the remaining work. It proves the ground the collective
// would stand on, with a pointer and a byte count rather than a claim.
namespace {

// Installed by the fault seam, read once at the top of device_roundtrip(). Inert by default:
// all three are zero, so a normal run takes the real HIP path with no branch taken and no
// outcome substituted. Atomics because the seam is called from the test thread and read from
// the same one only by convention; a race here would make the harness report a failure that
// did not happen, which is worse than the seam costing three relaxed loads.
std::atomic<int> g_fault_h2d{0};
std::atomic<int> g_fault_d2h{0};
std::atomic<int> g_fault_corrupt{0};

std::atomic<int> g_destructor_cleanups{0};
std::atomic<int> g_destructor_free_failures{0};

// Owns the device allocation for exactly one call, so that no path can return with a live
// handle, a stale handle, or a double free.
//
// The defect this replaces: each failure path after hipMalloc called hipFree() and then
// returned -- leaving the freed address in the result that a caller printed or could have
// dereferenced, and ignoring hipFree's own result so that a free which failed was a leak
// nobody could see. Two properties now hold by construction rather than by review: the
// allocation has one owner for its whole life, and its address never leaves this file.
class device_allocation {
public:
    device_allocation() = default;
    device_allocation(const device_allocation &) = delete;
    device_allocation & operator=(const device_allocation &) = delete;

    ~device_allocation() {
        // Unreachable on every path of this file, and that is the point of the arrangement
        // above it: all host allocation happens before adopt(), and nothing after adopt() can
        // throw, so no exception can unwind through this destructor. `noexcept` on
        // device_roundtrip() is what turns that arrangement into a compile-checked promise --
        // a later edit that introduces a throwing operation inside the window terminates at
        // the boundary rather than leaking quietly here.
        //
        // It is kept, and counted, because "unreachable" is a claim and a claim deserves a
        // measurement: destructor_cleanup_count() reports how many times it actually ran, and
        // the probe asserts zero. If that number is ever non-zero, a path escaped the single
        // exit and the count says so instead of a comment asserting otherwise.
        if (live_) {
            g_destructor_cleanups.fetch_add(1, std::memory_order_relaxed);
            if (hipFree(pointer_) != hipSuccess) {
                g_destructor_free_failures.fetch_add(1, std::memory_order_relaxed);
            }
            pointer_ = nullptr;
            live_ = false;
        }
    }

    void adopt(void * pointer) {
        pointer_ = pointer;
        live_ = pointer != nullptr;
    }

    // Frees now and reports the outcome. The handle is dropped even on failure: retrying a
    // failed free risks a double free, and a reported leak is worth more than a silent one.
    hipError_t release() {
        if (!live_) {
            return hipSuccess;
        }
        const hipError_t status = hipFree(pointer_);
        live_ = false;
        pointer_ = nullptr;
        return status;
    }

private:
    void * pointer_ = nullptr;
    bool   live_    = false;
};

} // namespace

roundtrip_result device_roundtrip(std::size_t bytes) noexcept {
    roundtrip_result r;
    r.bytes = bytes;

    if (bytes == 0) {
        r.stage = "argument check";
        std::snprintf(r.detail, sizeof(r.detail),
                      "a zero-byte round trip has nothing to verify");
        return r;
    }

    // The seam is read once, before any device work: a fault plan that could change mid-call
    // would make the measurement depend on when the test thread happened to write it.
    const int fault_h2d     = g_fault_h2d.load(std::memory_order_relaxed);
    const int fault_d2h     = g_fault_d2h.load(std::memory_order_relaxed);
    const int fault_corrupt = g_fault_corrupt.load(std::memory_order_relaxed);

    // EVERY host allocation happens here, before the first device call.
    //
    // This is what makes the destructor unreachable, and it was a defect: the readback buffer
    // used to be allocated after the device memory was adopted, so a std::bad_alloc there
    // unwound through ~device_allocation, whose only option is a hipFree whose result nobody
    // can report. The leak would have been real and silent -- and the caller would have
    // received an exception instead of a result describing what happened.
    //
    // Reserving both buffers first makes the exception window empty of owned device memory, so
    // the catch below is a clean refusal rather than a cleanup.
    std::vector<unsigned char> sent;
    std::vector<unsigned char> received;
    try {
        sent.resize(bytes);
        // A deterministic, position-dependent pattern: a constant fill would pass even if the
        // copy sent the same byte to every address, and the property under test is that the
        // mapping is positional.
        for (std::size_t i = 0; i < bytes; ++i) {
            sent[i] = static_cast<unsigned char>((i * 31u + 7u) & 0xFFu);
        }
        // A second buffer, so the comparison cannot be satisfied by the host copy outliving
        // the transfer: if the readback buffer were the send buffer, an implementation that
        // copied nothing would still compare equal.
        received.assign(bytes, 0);
    // std::exception, not std::bad_alloc.
    //
    // `sent.resize(bytes)` throws length_error when bytes exceeds the vector's max_size(),
    // which is reachable from this function's own parameter -- a caller passing SIZE_MAX asks
    // for it. The previous handler caught only bad_alloc, so a length_error would escape a
    // function declared noexcept: std::terminate. The declared contract was "never throws",
    // and the code kept it for one exception type out of two.
    //
    // Catching the base class also covers the allocation failure of `received.assign`, which
    // is bad_alloc proper, so one handler now matches the promise.
    } catch (const std::exception &) {
        r.stage = "host allocation";
        std::snprintf(r.detail, sizeof(r.detail),
                      "cannot allocate two %zu-byte host buffers; no device memory was "
                      "requested, so nothing was leaked",
                      bytes);
        return r;
    }

    device_allocation allocation;

    r.stage = "hipMalloc";
    void * device = nullptr;
    const hipError_t malloc_status = hipMalloc(&device, bytes);
    // Adopted even when the malloc reported failure.
    //
    // HIP is allowed to return a pointer together with an error, and that pointer must be
    // released. The previous version did it with a bare `(void) hipFree(device)` outside the
    // guarded path -- the same unreportable cleanup the guard exists to remove, in the one
    // place it did not reach, with its outcome discarded again. Routing it through adopt()
    // and release() gives it the same single exit and the same reporting as every other path.
    allocation.adopt(device);

    bool transferred = false;

    if (malloc_status != hipSuccess) {
        // `stage` stays "hipMalloc"; the free outcome is added by the single exit below.
        std::snprintf(r.detail, sizeof(r.detail), "hipMalloc(%zu) failed: %s", bytes,
                      hipGetErrorString(malloc_status));
    } else if (device == nullptr) {
        std::snprintf(r.detail, sizeof(r.detail),
                      "hipMalloc(%zu) reported success but returned no pointer", bytes);
    } else if (fault_h2d) {
        r.stage = "host -> device";
        std::snprintf(r.detail, sizeof(r.detail),
                      "host->device copy reported failed by the fault seam (not attempted; the "
                      "device buffer is the real one)");
    } else if (const hipError_t h2d = hipMemcpy(device, sent.data(), bytes, hipMemcpyHostToDevice);
               h2d != hipSuccess) {
        // The status hipMemcpy RETURNED, not hipGetLastError().
        //
        // hipMemcpy reports the failure itself; hipGetLastError() is a separate slot that the
        // NEXT HIP call overwrites. Asking for it after the copy can therefore name a
        // different call's error -- the message would read "invalid device pointer" for a
        // failed allocation. A diagnostic that names the wrong cause is worse than none: it
        // sends the reader to the wrong place with confidence.
        r.stage = "host -> device";
        std::snprintf(r.detail, sizeof(r.detail), "hipMemcpy H2D failed: %s",
                      hipGetErrorString(h2d));
    } else if (fault_d2h) {
        r.stage = "device -> host";
        std::snprintf(r.detail, sizeof(r.detail),
                      "device->host copy reported failed by the fault seam (not attempted; the "
                      "real copy did run the other way)");
    } else if (const hipError_t d2h = hipMemcpy(received.data(), device, bytes,
                                                hipMemcpyDeviceToHost);
               d2h != hipSuccess) {
        r.stage = "device -> host";
        std::snprintf(r.detail, sizeof(r.detail), "hipMemcpy D2H failed: %s",
                      hipGetErrorString(d2h));
    } else {
        // The corruption seam runs the REAL copies and only damages the readback, so the
        // comparison below fails through its own code rather than by a substituted verdict.
        // A substituted verdict would test the branch and not the check.
        if (fault_corrupt > 0) {
            const std::size_t n = std::min<std::size_t>(std::size_t(fault_corrupt), bytes);
            for (std::size_t i = 0; i < n; ++i) {
                received[i] = static_cast<unsigned char>(received[i] ^ 0xFFu);
            }
        }
        r.stage = "compare";
        transferred = true;
        for (std::size_t i = 0; i < bytes; ++i) {
            if (received[i] != sent[i]) {
                std::snprintf(r.detail, sizeof(r.detail),
                              "byte %zu of %zu differs: sent 0x%02x, read back 0x%02x", i, bytes,
                              sent[i], received[i]);
                transferred = false;
                break;
            }
        }
    }

    // THE single exit: free once, and fold the free's own outcome into the result. A destructor
    // cannot report, so the free is deliberate -- a failed free is a leak, and a leak nothing
    // says is a leak nobody finds. `stage` is left as the step that decided the outcome; only a
    // completed transfer moves it to "complete", so a failure names where it happened rather
    // than where the cleanup ran.
    const hipError_t freed = allocation.release();
    r.buffer_freed = freed == hipSuccess;
    r.ok = transferred && r.buffer_freed;

    if (!r.buffer_freed) {
        r.free_detail = "hipFree failed";
        char prior[sizeof(r.detail)];
        std::snprintf(prior, sizeof(prior), "%s", r.detail);
        std::snprintf(r.detail, sizeof(r.detail),
                      "%s; freeing the device buffer then failed (%s), so it may still be held",
                      *prior ? prior : "the round trip ended", hipGetErrorString(freed));
    } else if (transferred) {
        r.stage = "complete";
        std::snprintf(r.detail, sizeof(r.detail), "%zu bytes round-tripped through device memory",
                      bytes);
    }

    return r;
}

void set_fault_plan(const fault_plan & plan) noexcept {
    // Values are stored as 0/1 for the two booleans so that a caller passing 42 does not make
    // the branch depend on an int's magnitude.
    g_fault_h2d.store(plan.fail_h2d != 0 ? 1 : 0, std::memory_order_relaxed);
    g_fault_d2h.store(plan.fail_d2h != 0 ? 1 : 0, std::memory_order_relaxed);
    g_fault_corrupt.store(plan.corrupt_readback_bytes > 0 ? plan.corrupt_readback_bytes : 0,
                          std::memory_order_relaxed);
}

fault_plan get_fault_plan() noexcept {
    fault_plan plan;
    plan.fail_h2d = g_fault_h2d.load(std::memory_order_relaxed);
    plan.fail_d2h = g_fault_d2h.load(std::memory_order_relaxed);
    plan.corrupt_readback_bytes = g_fault_corrupt.load(std::memory_order_relaxed);
    return plan;
}

int destructor_cleanup_count() noexcept {
    return g_destructor_cleanups.load(std::memory_order_relaxed);
}

int destructor_free_failure_count() noexcept {
    return g_destructor_free_failures.load(std::memory_order_relaxed);
}

namespace {

// The device the plugin reports. Initialised once, from HIP, on first use.
//
// A function-local static rather than a namespace-scope object: this is a plugin loaded by
// dlopen at an arbitrary time, and a namespace-scope object with a HIP call in its
// constructor would run during that dlopen -- before the loader has finished registering
// anything, and with no chance to report a failure to anyone. Lazy and memoised means the
// first question ("is there a device?") is also the one that answers it.
const hip_device_info & cached_device_info() {
    static const hip_device_info info = [] {
        hip_device_info out;
        int count = 0;
        const hipError_t status = hipGetDeviceCount(&count);
        if (status != hipSuccess || count <= 0) {
            // No device, or no HIP runtime visible to this process. `device_present` stays
            // false, and score_from() turns that into a 0 so the loader declines the plugin
            // instead of registering a device that cannot be opened.
            return out;
        }
        out.device_present = true;
        out.hip_device_count = count;

        hipDeviceProp_t props{};
        if (hipGetDeviceProperties(&props, 0) == hipSuccess) {
            std::snprintf(out.name, sizeof(out.name), "%s", props.name);
            std::snprintf(out.arch, sizeof(out.arch), "%s", props.gcnArchName);
            // The fact, from the field HIP keeps for it (hip_runtime_api.h:145 "APU vs dGPU"),
            // rather than from the product name.
            out.integrated = props.integrated != 0;
        } else {
            std::snprintf(out.name, sizeof(out.name), "%s", "HIP device 0");
        }

        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
            out.memory_free_bytes = free_bytes;
            out.memory_total_bytes = total_bytes;
        }
        return out;
    }();
    return info;
}

// ---- backend (stream) -------------------------------------------------------
//
// There is no backend instance, because there is nothing to compute. init_backend reports
// that in words rather than returning a struct whose graph_compute would fail later.

ggml_backend_t hip_init_backend(ggml_backend_dev_t, const char * params) {
    // Three %s rather than two: the previous format opened a parenthesis and never closed it,
    // so a host that asked for this backend read "(requested: default" -- a small defect that
    // is worth removing precisely because it is the kind of thing a reader skims past.
    std::fprintf(stderr,
                 "[tp5] this plugin provides the device plane only: there are no arithmetic "
                 "kernels, so it does not initialise a compute backend%s%s%s\n",
                 params ? " (requested: " : "", params ? params : "", params ? ")" : "");
    return nullptr;
}

// ---- device -----------------------------------------------------------------

const char * hip_dev_name(ggml_backend_dev_t) {
    return "TP5";
}

const char * hip_dev_description(ggml_backend_dev_t) {
    return cached_device_info().name;
}

void hip_dev_memory(ggml_backend_dev_t, size_t * free_bytes, size_t * total_bytes) {
    // Re-read rather than reuse the cached figures: memory is the one property that changes
    // while the process runs, and a caller asking now wants now.
    std::size_t free_now = 0;
    std::size_t total_now = 0;
    if (hipMemGetInfo(&free_now, &total_now) != hipSuccess) {
        free_now = 0;
        total_now = 0;
    }
    if (free_bytes) *free_bytes = free_now;
    if (total_bytes) *total_bytes = total_now;
}

enum ggml_backend_dev_type hip_dev_type(ggml_backend_dev_t) {
    // The distinction is not cosmetic: llama.cpp uses it when deciding where weights may live,
    // and a plugin that claimed dedicated memory would invite an allocation the device cannot
    // back. Read from hipDeviceProp_t::integrated, not from the marketing name -- a name is a
    // label, and inferring "APU" from the words "Radeon" and "Graphics" classifying a card by
    // its product name is the same defect as reading a backend's kind out of its display name.
    //
    // Unreachable today: the registry reports zero devices (see hip_reg_device_count), so
    // nothing calls this. It stays because the device table is the scaffolding for the first
    // kernel, and because leaving it wrong while unregistered would make the day registration
    // resumes the day the classification bug ships.
    return cached_device_info().integrated ? GGML_BACKEND_DEVICE_TYPE_IGPU
                                           : GGML_BACKEND_DEVICE_TYPE_GPU;
}

void hip_dev_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    if (!props) return;
    const hip_device_info & info = cached_device_info();
    props->name = hip_dev_name(dev);
    props->description = info.name[0] ? info.name : "HIP device";
    std::size_t free_now = 0;
    std::size_t total_now = 0;
    hip_dev_memory(dev, &free_now, &total_now);
    props->memory_free = free_now;
    props->memory_total = total_now;
    props->type = hip_dev_type(dev);
    props->device_id = nullptr;   // how to name the PCI address is not known here
    props->caps.async = false;                 // nothing asynchronous is implemented
    props->caps.host_buffer = false;            // no pinned host buffer type
    props->caps.buffer_from_host_ptr = false;
    props->caps.events = false;
    props->caps.mmap_support = false;
}

bool hip_dev_supports_op(ggml_backend_dev_t, const ggml_tensor *) {
    // Every op, refused. The kernels are not here; saying otherwise would have llama.cpp
    // schedule the graph onto a backend whose compute path does not exist.
    return false;
}

bool hip_dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t) {
    // No buffer type is offered (see hip_dev_buffer_type), so no buffer type is supported.
    return false;
}

ggml_backend_buffer_type_t hip_dev_buffer_type(ggml_backend_dev_t) {
    // null, not a host-memory stand-in.
    //
    // A device buffer type here would have to be backed by hipMalloc to be truthful, and a
    // buffer type that is *only* hipMalloc is a lie in the other direction: llama.cpp would
    // allocate model weights into it and then find that no op can read them. The device plane
    // this plugin exists for -- a relay staging buffer -- is exposed through
    // device_roundtrip() and the proc address below, where its guarantees are stated and
    // measured, instead of through a ggml buffer contract it does not satisfy.
    return nullptr;
}

// The device table, kept as the scaffolding for the first kernel and not registered: the
// registry above reports no device, so nothing reaches these entries. Marked unused so the
// build stays warning-clean while the day it is wired back is a one-line change to the count.
[[maybe_unused]] const ggml_backend_device_i kDeviceInterface = {
    /* .get_name         = */ hip_dev_name,
    /* .get_description  = */ hip_dev_description,
    /* .get_memory       = */ hip_dev_memory,
    /* .get_type         = */ hip_dev_type,
    /* .get_props        = */ hip_dev_props,
    /* .init_backend     = */ hip_init_backend,
    /* .get_buffer_type  = */ hip_dev_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op      = */ hip_dev_supports_op,
    /* .supports_buft    = */ hip_dev_supports_buft,
    /* .offload_op       = */ nullptr,
    /* .event_new        = */ nullptr,
    /* .event_free       = */ nullptr,
    /* .event_synchronize = */ nullptr,
};

// ---- registry ---------------------------------------------------------------

const char * hip_reg_name(ggml_backend_reg_t) {
    return "TP5";
}

size_t hip_reg_device_count(ggml_backend_reg_t) {
    // ZERO, deliberately, and this is the most important line in the plugin.
    //
    // The previous version reported one device as soon as HIP saw a GPU. That device had a
    // null buffer type (correctly: no op can read hipMalloc'd weights), and llama.cpp does not
    // tolerate that. The chain, read out of upstream rather than imagined:
    //
    //   llama.cpp:224-265        default selection puts a GPU_TYPE_IGPU device into `igpus`
    //   llama-model.cpp:1802-1808 for every chosen device: make_gpu_buft_list(dev, ...)
    //   llama-model.cpp:1352     buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev))
    //                            -- NO null check
    //   llama-model.cpp:1815     the list is used to allocate model tensors
    //
    // So a registered device with a null buft lands a nullptr in the allocation list, and the
    // first model load walks into it. On this host there is no discrete GPU, which makes the
    // TP5 device the only igpu -- i.e. installing this plugin in the search path would have
    // broken ordinary CPU inference, and the crash would have landed in the model loader with
    // nothing pointing here.
    //
    // Reporting zero devices removes the hazard by construction instead of guarding it: no
    // device means nothing enters `gpus`/`igpus`, and ggml_backend_dev_count() is unchanged, so
    // a host that loads this plugin behaves exactly like one that does not. The device plane
    // stays reachable because the round-trip lives on the REGISTRY's proc-address hook, which
    // needs no device (that is what the probe checks).
    //
    // This number becomes non-zero on the day a compute backend exists -- i.e. when
    // get_buffer_type() can return a buffer type whose tensors a kernel can actually read.
    // Until then, a device here is a lie that costs a crash.
    return 0;
}

ggml_backend_dev_t hip_reg_device(ggml_backend_reg_t, size_t) {
    // No device is registered, so no index is valid. Returning a valid handle for index 0 --
    // as the previous version did, with a lazily-bound `reg` field -- would contradict the
    // count above and hand llama.cpp the device the count exists to withhold.
    return nullptr;
}

namespace {

// The report's `status`, derived from the in-process result.
//
// A mapping instead of a shared enum because `roundtrip_result` predates the ABI and its
// `stage` is what the failure paths set; adding a second status field to it would be a second
// place to keep true. The strings compared here are literals assigned in this same file, and
// the probe asserts the two directions agree (ok ⇔ status OK), so a renamed stage shows up as
// a test failure rather than as a silently wrong code.
//
// The fallback is -1, NOT a member of the enum: an unrecognised stage means this table is
// stale, and a caller testing `status != 0` still sees a failure. Inventing a specific code
// would name a cause the code did not observe.
std::int32_t trip_status_of(const roundtrip_result & r) {
    if (r.ok) return TQ_TP5_TRIP_OK;
    if (!r.buffer_freed) return TQ_TP5_TRIP_FREE;
    if (std::strcmp(r.stage, "argument check") == 0) return TQ_TP5_TRIP_ZERO_BYTES;
    if (std::strcmp(r.stage, "host allocation") == 0) return TQ_TP5_TRIP_HOST_ALLOC;
    if (std::strcmp(r.stage, "hipMalloc") == 0) return TQ_TP5_TRIP_MALLOC;
    if (std::strcmp(r.stage, "host -> device") == 0) return TQ_TP5_TRIP_COPY_H2D;
    if (std::strcmp(r.stage, "device -> host") == 0) return TQ_TP5_TRIP_COPY_D2H;
    if (std::strcmp(r.stage, "compare") == 0) return TQ_TP5_TRIP_MISMATCH;
    return -1;
}

} // namespace

// The exported boundary: fixed-width, self-describing, and bounded by the CALLER's buffer size.
//
// Every check here exists because the two sides may have been built at different times. The
// previous export -- `&device_roundtrip`, a C++ function returning a class by value -- assumed
// they had not, and the symptom was a crash inside the caller's own stack frame.
extern "C" std::int32_t tq_tp5_roundtrip_v1(const tq_tp5_roundtrip_request * request,
                                            tq_tp5_roundtrip_report * report,
                                            std::uint64_t report_size) {
    if (!request || !report) {
        return TQ_TP5_ABI_ARGUMENT;
    }
    // Alignment first, because it is a property of the POINTER the caller handed over and does
    // not depend on any byte of it. Our own write is memcpy and would tolerate any address; the
    // caller's read is through a struct pointer and would not. Refusing here turns its silent
    // undefined behaviour into a named fact at the boundary.
    if (report_size >= sizeof(std::uint32_t) &&
        reinterpret_cast<std::uintptr_t>(report) % alignof(tq_tp5_roundtrip_report) != 0) {
        return TQ_TP5_ABI_UNALIGNED;
    }
    // Fewer than the two identifying fields: nothing can be written at all, and the caller is
    // told so without its buffer being touched.
    if (report_size < 2 * sizeof(std::uint32_t)) {
        return TQ_TP5_ABI_REPORT_TOO_SMALL;
    }
    // Between the header and this callee's complete report.
    //
    // The previous version ran the transfer anyway and returned ABI_OK with a truncated report
    // copied in. A caller that trusted the return code then read `stage` and `detail` out of
    // bytes this function never wrote -- its own uninitialised memory, presented as this
    // plugin's diagnosis. So the call does not run: the header is written (it is what lets the
    // caller compare versions and sizes), and the return value says the report is incomplete.
    // ABI_OK now means "every field you may read was written", which is the only meaning a
    // caller can safely act on.
    if (report_size < sizeof(tq_tp5_roundtrip_report)) {
        const std::uint32_t header[2] = {TQ_TP5_ROUNDTRIP_ABI_VERSION,
                                         std::uint32_t(sizeof(tq_tp5_roundtrip_report))};
        std::memcpy(report, header, sizeof(header));
        return TQ_TP5_ABI_REPORT_TOO_SMALL;
    }

    // Filled locally and copied in one step at the end. Reaching this point means report_size
    // covers the whole struct (checked above), so that copy is exactly sizeof(filled) -- and
    // ABI_OK can therefore mean "every field you may read was written".
    tq_tp5_roundtrip_report filled{};
    filled.abi_version = TQ_TP5_ROUNDTRIP_ABI_VERSION;
    filled.struct_size = sizeof(tq_tp5_roundtrip_report);
    filled.status = TQ_TP5_TRIP_OK;
    filled.buffer_freed = 1;
    std::snprintf(filled.stage, sizeof(filled.stage), "%s", "not attempted");
    std::snprintf(filled.detail, sizeof(filled.detail), "%s",
                  "the call did not run: the caller and the plugin disagree about this interface");

    std::int32_t result = TQ_TP5_ABI_OK;

    if (request->abi_version != TQ_TP5_ROUNDTRIP_ABI_VERSION ||
        request->struct_size != sizeof(tq_tp5_roundtrip_request)) {
        // A distinct return code, not a status in the report: "we could not talk to each
        // other" is a different fact from "the transfer failed", and conflating them is what
        // made the old interface's failure unreadable.
        result = TQ_TP5_ABI_VERSION_MISMATCH;
    } else if (request->bytes > std::numeric_limits<std::size_t>::max()) {
        // A 64-bit byte count against a 32-bit size_t: the only honest answer is a refusal.
        // Truncating it would ask for a smaller transfer than the caller requested and then
        // report success.
        result = TQ_TP5_ABI_ARGUMENT;
    } else {
        const roundtrip_result r = device_roundtrip(static_cast<std::size_t>(request->bytes));
        filled.status = trip_status_of(r);
        filled.buffer_freed = r.buffer_freed ? 1 : 0;
        filled.bytes = r.bytes;
        // The prose is copied, not pointed at: the caller's buffer is the only storage that
        // both sides agree on, and a `const char *` into this image is a lifetime assumption
        // the interface must not make.
        std::snprintf(filled.stage, sizeof(filled.stage), "%s", r.stage ? r.stage : "");
        std::snprintf(filled.detail, sizeof(filled.detail), "%s%s%s", r.detail,
                      r.free_detail ? "; " : "", r.free_detail ? r.free_detail : "");
    }

    // One complete report, and ABI_OK says so. The bounded copy this replaces could return OK
    // with a partial struct written, which a caller would then read as its own garbage.
    std::memcpy(report, &filled, sizeof(filled));
    return result;
}

void * hip_reg_proc_address(ggml_backend_reg_t, const char * name) {
    if (!name) {
        return nullptr;
    }
    // The one extension this plugin offers, reachable through upstream's documented
    // "backend-specific function" hook (ggml_backend_reg_get_proc_address). A caller uses it
    // to exercise the device plane directly; nothing in llama.cpp calls it, so the plugin
    // stays inert for ordinary inference.
    //
    // THE NAME IS VERSIONED, and the unversioned old name is served NOTHING. That refusal is
    // the fix for the one defect a version *field* cannot address: a caller built against the
    // first header asks for "tq_tp5_device_roundtrip" and casts the answer to
    // `roundtrip_result (*)(std::size_t)`. If we answered with the new C function, the call
    // would enter the callee with the argument registers reinterpreted -- the callee would read
    // a request pointer from wherever the caller put 4096 and dereference it. The in-body
    // version check runs AFTER that read, so for a stale caller it is dead code. The mismatch
    // is in the call, not in the bytes, and only the name can express it.
    //
    // Null is the only safe answer: we cannot know which signature the caller believed in, so
    // no address could be correct. A caller that checks for null (any careful one does) reports
    // that the interface it was built for is absent; one that does not dereferences null in its
    // own frame, at the boundary, instead of writing through a stray pointer in here.
    if (std::strcmp(name, "tq_tp5_roundtrip_v1") == 0) {
        return reinterpret_cast<void *>(&tq_tp5_roundtrip_v1);
    }
    // The version, so a caller can refuse before it calls anything it does not understand.
    if (std::strcmp(name, "tq_tp5_roundtrip_abi_version") == 0) {
        return reinterpret_cast<void *>(
            static_cast<std::int32_t (*)()>([] { return std::int32_t(TQ_TP5_ROUNDTRIP_ABI_VERSION); }));
    }
    // The pre-versioning name. Recognised explicitly rather than left to the final `return
    // nullptr`, so that the refusal is a decision written down rather than a fall-through: a
    // future reader adding an entry point will see this branch and know the name is spent.
    if (std::strcmp(name, "tq_tp5_device_roundtrip") == 0) {
        return nullptr;
    }
    return nullptr;
}

const ggml_backend_reg_i kRegistryInterface = {
    /* .get_name         = */ hip_reg_name,
    /* .get_device_count = */ hip_reg_device_count,
    /* .get_device       = */ hip_reg_device,
    /* .get_proc_address = */ hip_reg_proc_address,
};

ggml_backend_reg_t hip_registry() {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ kRegistryInterface,
        /* .context     = */ nullptr,
    };
    return &reg;
}

// The two symbols upstream's loader looks for (ggml-backend-reg.cpp:229, :234). The macros
// emit them with the visibility and the C linkage the loader expects; writing them by hand
// would be the same thing with a chance of getting `extern "C"` wrong.
//
// The call sits inside the anonymous namespace, next to the function it names: members of an
// anonymous namespace have internal linkage that no outer scope can refer to, and `extern "C"`
// fixes the linkage NAME without moving the name into another scope. Emitted from here, the
// two symbols still leave the shared object with C linkage -- which is all dlsym asks for --
// while the definition stays where the name is visible.
GGML_BACKEND_DL_IMPL(hip_registry)

// A score function is the loader's other question (ggml-backend-reg.cpp:229): 0 means "not
// supported on this system", and the plugin is then declined BEFORE ggml_backend_init is
// called at all. Without it the loader takes the plugin's word that it is usable, so a
// machine with no visible GPU would register a backend with zero devices -- harmless in this
// process, and misleading to every tool that reports "a TP5 backend is present".
//
// Computed from cached_device_info(), through score_from(): a rule that can only be exercised
// on the hardware it guards is a rule nobody tests, and this one is testable everywhere
// because the fact it needs is a plain struct.
//
// score_from() itself is defined OUTSIDE this anonymous namespace, further down. It is part of
// the module's public header, so it must keep external linkage; defining it here would give it
// internal linkage, leave the shared object without the symbol, and turn the declaration in
// hip_backend.h into a promise nothing keeps -- exactly the unresolved-symbol link failure this
// cost once.
int hip_score() {
    return score_from(cached_device_info());
}

GGML_BACKEND_DL_SCORE_IMPL(hip_score)

} // namespace

// Public, external linkage, defined where hip_backend.h says it is.
//
// Upstream's scoring contract: higher is preferred, 0 means "not supported on this system", and
// the loader then declines the plugin WITHOUT calling init (ggml-backend-reg.cpp:229). So a host
// with no usable device must score 0: a plugin loaded into a machine with no GPU would enumerate
// zero devices and leave llama.cpp believing a GPU backend exists.
int score_from(const hip_device_info & info) {
    if (!info.device_present || info.hip_device_count <= 0) {
        return 0;
    }
    // Above the CPU backend's floor, below nothing in particular: this is a device backend, and
    // a device is preferred when it is present. It does not try to outbid a vendor backend for
    // the same hardware -- choosing between them is a llama.cpp decision (device type and
    // ordering), not a score war between plugins.
    return 800;
}

} // namespace tq::tp5
