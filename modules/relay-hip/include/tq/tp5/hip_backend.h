// tq/tp5/hip_backend.h — the TP5 HIP device plane, as a ggml backend plugin.
//
// Why a plugin and not a patch
// ----------------------------
// The fork carried its five-rank relay inside ggml-cuda: a copy of the CUDA backend with
// relay.hip added, kept alive by a maintained upstream exception. In this project upstream is
// a pinned submodule that nobody edits, so that arrangement is not available.
//
// It does not need to be. ggml loads backends as shared objects:
//
//   ggml-backend-reg.cpp:229   dlsym(handle, "ggml_backend_score")  -> int, 0 = unsupported
//   ggml-backend-reg.cpp:234   dlsym(handle, "ggml_backend_init")   -> ggml_backend_reg_t
//   ggml-backend-reg.cpp:242   reg->api_version != GGML_BACKEND_API_VERSION  -> refused
//
// and ggml_backend_load(path) does exactly that. Every backend upstream ships -- BLAS, CUDA,
// Vulkan, RPC -- is built through the same two symbols. So a TP5 device backend can be a
// separate .so that upstream loads without knowing what it is. No hook, no trampoline, no
// patched translation unit: the plugin is the mechanism, and upstream's own loader is the
// wiring.
//
// What this plugin is, and what it is not
// ---------------------------------------
// It IS the device plane a five-rank relay needs to stand on:
//
//   * real device enumeration through HIP, not a stub that reports one device;
//   * real device memory (hipMalloc / hipFree), because that is what a relay staging buffer
//     is -- the fork's relay.hip stages into device memory, and a host-memory stand-in would
//     be the CPU-simulation this project refuses to pass off as a GPU path;
//   * real host<->device copies, with the byte count and the pointer reported, so a caller
//     can check the transfer rather than trust it;
//   * a truthful capability report: supports_op() says no for every op, because the
//     arithmetic kernels are not here yet. A backend that claimed support it does not have
//     would make llama.cpp schedule work onto a device that cannot do it -- the failure would
//     land in the decode, far from this file.
//
// It is NOT the five-rank relay. The collective (generation/ready/ack over a shared control
// block, and the staged all-gather) is modules/relay-transport + modules/relay-plan, which
// are host-side and tested. Wiring those to this device plane -- device-side kernels that
// publish and poll the control block -- is the remaining work, and the reason this file says
// "supports_op: false" instead of pretending to compute.
//
// Why GGML_BACKEND_DL and ggml-backend-impl.h
// -------------------------------------------
// Both are upstream's public mechanism for exactly this: ggml-backend-impl.h is the header a
// backend implements against (CUDA, Vulkan and Metal all include it), and GGML_BACKEND_DL_IMPL
// is the macro that emits the two symbols the loader looks for. Including it is how a backend
// is written; it is not reaching into private state, and it does not modify upstream.
#pragma once

#include <cstddef>
#include <cstdint>

namespace tq::tp5 {

// What the device plane reports about itself. Filled by hip_backend.cpp from real HIP calls;
// every field is a measurement, and `device_present` is false when HIP reports no device (a
// host without ROCm, or a ROCm install with no GPU visible to this process).
struct hip_device_info {
    bool          device_present = false;
    int           hip_device_count = 0;
    std::size_t   memory_total_bytes = 0;
    std::size_t   memory_free_bytes = 0;
    // APU vs discrete, from hipDeviceProp_t::integrated -- the field HIP provides for exactly
    // this question. The earlier version inferred it from the marketing name ("Radeon" and
    // "Graphics" both appearing), which is the same mistake as reading "CPU" out of a backend's
    // display label: a name is a label, and a discrete card whose product name happens to
    // contain both words would have been reported as an APU. False when the properties call
    // failed, which errs toward "discrete" -- the conservative reading for a device whose
    // memory the host does not share.
    bool          integrated = false;
    char          name[64] = {};
    char          arch[32] = {};     // gcnArchName, e.g. "gfx1103"
};

// The facts the plugin's score is computed from. Pure: no HIP call, so the policy can be
// tested on a machine with no GPU -- the same reason release decisions keep their rule
// separate from the device query.
//
// Scoring is upstream's contract: 0 means "not supported on this system", and the loader
// then declines the plugin without calling init at all (ggml-backend-reg.cpp:229). So this
// must not be optimistic. A plugin that scores non-zero on a host with no device would be
// loaded, enumerate zero devices, and leave llama.cpp believing a GPU backend is present.
int score_from(const hip_device_info & info);

// The single entry point the probe uses to exercise the device plane without going through
// a model: allocate `bytes` of device memory, copy a caller-supplied pattern in, read it back,
// and report whether the round trip preserved every byte.
//
// This is the property a relay staging buffer needs: bytes written by the host arrive on the
// device intact, and bytes the device wrote arrive back intact. If that fails, no amount of
// collective protocol work matters.
struct roundtrip_result {
    bool        ok = false;
    const char * stage = "not attempted";   // where it stopped, for a caller's message
    std::size_t bytes = 0;
    // No device address is reported, and the field that used to carry one is gone.
    //
    // It could not be made safe by discipline. The allocation's owner is this function, so any
    // address handed out is either stale (freed before the return, which the failure paths
    // did) or a handle the caller must free but cannot -- it has no way to know whether the
    // free already happened. Both readings are wrong, and the previous fix (clear it on
    // success, keep it on failure) made the field truthful only for the paths nobody takes:
    // H2D, D2H and compare failures each called hipFree() and still reported the address.
    //
    // Removing the field designs the misuse out. What a caller needs from a failure is the
    // stage and the reason, both in `detail`; the address is not actionable by anyone outside
    // this translation unit.
    char        detail[192] = {};
    // Whether the device allocation was actually returned to HIP, and how that went.
    //
    // These are facts a caller can check, not diagnostics it must trust. Without them the
    // cleanup's outcome was invisible: a `hipFree` that failed inside the function left a
    // live allocation and nothing said so, and a caller could not tell "the transfer failed
    // and the memory came back" from "the transfer failed and the memory leaked". Both
    // readings produced the same `ok == false`.
    //
    // `buffer_freed` is true only when hipFree succeeded. When the malloc never produced a
    // pointer there was nothing to free, and it is true -- "nothing was owned" is a different
    // statement from "owned memory was released", and conflating them would make an
    // allocation that never happened look like a cleanup that worked.
    bool        buffer_freed = true;
    // Non-null only when buffer_freed is false: why the free failed, on the same terms as
    // `stage` (a literal, so it outlives the call).
    const char * free_detail = nullptr;
};

// ---------------------------------------------------------------------------
// The cross-object ABI
// ---------------------------------------------------------------------------
//
// `roundtrip_result` above is an IN-PROCESS type. It must never be returned through
// ggml_backend_reg_i::get_proc_address, and it no longer is: that hook was the defect.
//
// What went wrong, precisely. The proc address used to return &device_roundtrip -- a C++
// function returning roundtrip_result BY VALUE. A consumer built against an older header had
// its own idea of the struct's size (216 bytes, then 232 once the free-reporting fields were
// added) while the callee wrote according to the new one, so the callee's sret store ran past
// the end of the caller's object. It aborted. Rebuilding everything "fixes" it until the next
// field is added, and nothing in the interface said a rebuild was needed: the failure mode was
// heap corruption inside a plugin whose entire purpose is memory safety.
//
// A by-value C++ class across a shared-object boundary is not an ABI at all -- the two sides
// must agree on size, layout, and the calling convention for a non-trivial return. So the
// boundary is now C, fixed-width, self-describing, and refuses loudly:
//
//   * only uint32_t / int32_t / uint64_t / char[N] -- no bool, no size_t, no pointer, no C++
//     type, so there is no layout the two sides can compute differently;
//   * the caller states which version it was compiled against AND how big its buffer is;
//   * the callee writes the 8-byte header first whenever the buffer can hold it, so a
//     disagreement is diagnosable instead of silent or fatal;
//   * the return value is about the CALL; the report carries the round trip's outcome. That
//     separation is what makes "we could not talk to each other" distinguishable from "the
//     transfer failed" -- previously both were an `ok == false` in a struct of disputed shape.
constexpr std::uint32_t TQ_TP5_ROUNDTRIP_ABI_VERSION = 1u;

// THE ENTRY POINT IS VERSIONED IN ITS NAME. This is the second half of the same defect, and it
// was the half that still bit.
//
// The first revision fixed the struct but kept the proc name "tq_tp5_device_roundtrip". That
// name was already a promise made to older callers: a binary compiled against the first header
// asks for exactly that string and casts the result to `roundtrip_result (*)(std::size_t)`.
// Handing it the new C function does not fail a check -- it ENTERS the function with the
// caller's register agreement violated. The callee reads `request` from the first argument
// register, where the caller put 4096, and dereferences address 4096. The version test that is
// supposed to protect the caller runs after that read. For a stale caller it is unreachable
// code: the crash precedes the check that describes the mismatch.
//
// No runtime test can fix that, because the disagreement is about the CALL, not about the
// bytes. So the name carries the version, and the old name is served nothing:
//
//   "tq_tp5_roundtrip_v1"          -> this entry point
//   "tq_tp5_roundtrip_abi_version" -> int32_t (void), the same number
//   "tq_tp5_device_roundtrip"      -> nullptr, deliberately and permanently
//
// Null is the only safe answer for the old name. We cannot know which signature the caller
// believes in, so there is no address it could interpret correctly. A caller that checks for
// null -- the probe did, and any careful caller does -- declines cleanly and can report "this
// plugin does not offer the interface I was built for". A caller that does not check
// dereferences null at its own call site: a fault exactly at the boundary, in the frame that
// made the mistake, rather than a write through a stray pointer inside the callee.
//
// Return codes. TQ_TP5_ABI_OK now means PROPERTY, not just "the transfer ran": every field of
// the report was written. Every other value means the caller may read only the leading
// version/size pair, and nothing else in its buffer was touched.
enum : std::int32_t {
    TQ_TP5_ABI_OK               = 0,  // the whole report is valid; read report->status
    TQ_TP5_ABI_VERSION_MISMATCH = 1,  // caller and callee disagree; header only
    TQ_TP5_ABI_REPORT_TOO_SMALL = 2,  // the buffer cannot hold this callee's report; header only
    TQ_TP5_ABI_ARGUMENT         = 3,  // a null request or report pointer
    TQ_TP5_ABI_UNALIGNED        = 4,  // the report pointer is not aligned for its own type
};

// The round trip's own outcome, reported in report->status.
enum : std::int32_t {
    TQ_TP5_TRIP_OK         = 0,
    TQ_TP5_TRIP_HOST_ALLOC = 1,  // could not allocate the host buffers; no device memory asked
    TQ_TP5_TRIP_MALLOC     = 2,  // hipMalloc failed or returned nothing
    TQ_TP5_TRIP_COPY_H2D   = 3,
    TQ_TP5_TRIP_COPY_D2H   = 4,
    TQ_TP5_TRIP_MISMATCH   = 5,  // a byte differed
    TQ_TP5_TRIP_ZERO_BYTES = 6,  // nothing to verify
    TQ_TP5_TRIP_FREE       = 7,  // the transfer finished but the device buffer was not released
};

struct tq_tp5_roundtrip_request {
    std::uint32_t abi_version;
    std::uint32_t struct_size;
    std::uint64_t bytes;
};

struct tq_tp5_roundtrip_report {
    std::uint32_t abi_version;   // the callee's, always written
    std::uint32_t struct_size;   // the callee's, always written
    std::int32_t  status;        // TQ_TP5_TRIP_*
    std::int32_t  buffer_freed;  // 0 or 1; 1 also when there was nothing to free
    std::uint64_t bytes;
    char          stage[40];     // where it stopped, for a caller's message
    char          detail[192];   // why, for a caller's message
};

// The exported entry point, versioned in its name (see above). `report_size` is the caller's
// sizeof(tq_tp5_roundtrip_report) -- passed explicitly rather than inferred, because the whole
// point is that the callee may be newer than the caller and must not assume otherwise.
//
// The two pointer arguments carry contracts the language cannot check for us, so they are
// stated here instead of being assumed:
//
//   request       must point to a COMPLETE tq_tp5_roundtrip_request. The callee reads its
//                 first 8 bytes (version, struct_size) before it can validate anything, so a
//                 request object of fewer bytes is already undefined behaviour on the caller's
//                 side. There is no way to test this from inside: the test would have to read
//                 the pointer to decide whether reading it is allowed.
//
//   report        must be aligned for tq_tp5_roundtrip_report (alignof 8, because of the
//                 uint64_t member) and must have report_size writable bytes. The callee writes
//                 with memcpy, which tolerates any alignment, but the CALLER can only read the
//                 result by dereferencing a struct pointer -- and that is undefined behaviour
//                 on a misaligned address. So a misaligned pointer is refused with
//                 TQ_TP5_ABI_UNALIGNED: our write would have been safe, the caller's read would
//                 not, and turning its silent UB into a named refusal is the only useful thing
//                 this side can do.
extern "C" std::int32_t tq_tp5_roundtrip_v1(const tq_tp5_roundtrip_request * request,
                                            tq_tp5_roundtrip_report * report,
                                            std::uint64_t report_size);

// The in-process C++ wrapper, for callers in the same binary. It is deliberately NOT what the
// proc address returns: `roundtrip_result` carries a `const char * stage` pointing at a literal
// in whichever image defined it, which is fine while caller and callee are the same image and
// is exactly the assumption a plugin boundary must not make.
//
// Never throws, and that is a contract rather than a hope.
//
// The reason it matters: this function owns device memory between its malloc and its free.
// An exception escaping from inside that window would run the destructor, whose only
// remaining option is a hipFree whose result nobody can report -- the silent-leak shape this
// module has already had to remove twice. So the function is arranged so no exception can
// occur while it owns memory (every host allocation happens first, and is caught), and the
// `noexcept` here is what makes that arrangement a compile-checked promise: if a later edit
// introduces a throwing operation inside the window, the compiler terminates at the boundary
// instead of leaking quietly in a destructor.
roundtrip_result device_roundtrip(std::size_t bytes) noexcept;

// Test-only fault injection, inert in every normal run.
//
// The failure paths below the allocation -- a failed host->device copy, a failed readback, a
// mismatch -- cannot be reached on a working GPU, so they were "verified by inspection",
// which is another way of saying not verified. This seam makes them reachable: each field
// substitutes the outcome of one step, and `device_roundtrip` otherwise runs its real code on
// real device memory, so the cleanup under test is the cleanup that ships.
//
// Thread-safety: the fields are atomics because a seam that races is a defect in the test
// harness rather than in the code under test -- and a racing harness reports failures that
// did not happen, which costs more than the seam. Set the plan, call once, clear it.
struct fault_plan {
    // Non-zero: report the host->device copy as failed without attempting it.
    int fail_h2d = 0;
    // Non-zero: report the device->host copy as failed without attempting it.
    int fail_d2h = 0;
    // Non-zero: let the real copies run, then corrupt this many bytes of the readback so the
    // comparison fails through its own code path rather than by a substituted verdict.
    int corrupt_readback_bytes = 0;
};

void set_fault_plan(const fault_plan & plan) noexcept;
fault_plan get_fault_plan() noexcept;

// How many times the round trip's last-resort cleanup ran, and how many of those frees
// failed. Both should stay zero for the life of the process: the function has a single exit
// per path and cannot throw while it owns device memory, so the fallback is unreachable. A
// non-zero count means a path escaped that arrangement -- which is exactly the kind of claim
// worth measuring instead of asserting in a comment.
int destructor_cleanup_count() noexcept;
int destructor_free_failure_count() noexcept;

} // namespace tq::tp5
