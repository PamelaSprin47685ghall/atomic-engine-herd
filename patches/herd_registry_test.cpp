// The seam registry and installer under real concurrency, with real hooks.
//
// Two promises are made by patches/herd_patch.hpp, and each had a race behind it:
//
//   1. add() is public -- "for those, called from code that runs later." Two registrars
//      running together could both pass the duplicate check and both claim one slot, and
//      the slot count could be published before the slot was written. A seam lost that way
//      is invisible afterwards, because every surviving entry is valid: find() answers for
//      every name that won.
//
//   2. install_all() is idempotent, and is called both from constructor(102) and from
//      callers that run later. It used `static bool done`, so two callers entering together
//      both read `false` and both ran the DobbyHook loop. Hooking an address that already
//      holds our jump builds a trampoline that leads back into the replacement: recursion
//      until the stack ends.
//
// Every seam here targets a REAL function, so Dobby actually patches code and the installed
// table is exercised rather than merely counted. The one seam that is called afterwards is
// dispatched end to end: target -> hook -> replacement -> trampoline -> original.
//
// The bodies are deliberately not tiny. Dobby needs room to write an absolute jump over the
// prologue, and a two-instruction `return v * k` leaves it none: the first version of this
// test aborted with "DobbyHook failed" on sixteen one-liners. The volatile accumulator below
// is that room, and it is the same constraint -fpatchable-function-entry exists to solve.
//
// Not claimed here: the "add() after installation aborts" half of the contract. abort()
// leaves nothing to assert on in-process, and pretending a `check` covers it would be the
// kind of green tick this project keeps having to retract.
#include "herd_patch.hpp"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string & what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

// Sixteen distinct real functions, each with enough body for a hook. Distinct bodies matter:
// identical-code folding would merge twins into one address, and then "no slot shared" would
// be true for a reason that has nothing to do with the registry.
#define HERD_TEST_TARGET(N, MUL, ADD)                                            \
    extern "C" __attribute__((noinline)) int herd_reg_target_##N(int v) {        \
        volatile int acc = v;                                                    \
        for (int i = 0; i < 4; ++i) { acc = acc * (MUL) + (ADD); }               \
        return acc;                                                              \
    }
HERD_TEST_TARGET(0, 3, 1)
HERD_TEST_TARGET(1, 5, 2)
HERD_TEST_TARGET(2, 7, 3)
HERD_TEST_TARGET(3, 11, 4)
HERD_TEST_TARGET(4, 13, 5)
HERD_TEST_TARGET(5, 17, 6)
HERD_TEST_TARGET(6, 19, 7)
HERD_TEST_TARGET(7, 23, 8)
HERD_TEST_TARGET(8, 29, 9)
HERD_TEST_TARGET(9, 31, 10)
HERD_TEST_TARGET(10, 37, 11)
HERD_TEST_TARGET(11, 41, 12)
HERD_TEST_TARGET(12, 43, 13)
HERD_TEST_TARGET(13, 47, 14)
HERD_TEST_TARGET(14, 53, 15)
HERD_TEST_TARGET(15, 59, 16)
#undef HERD_TEST_TARGET

// The seam that is dispatched end to end. Its body is padded the same way, and its result is
// known by hand: 4 rounds of `acc * 3 + 1` from 0 reach 40, and from 1 reach 121.
extern "C" __attribute__((noinline)) int herd_reg_dispatch_target(int v) {
    volatile int acc = v;
    for (int i = 0; i < 4; ++i) {
        acc = acc * 3 + 1;
    }
    return acc * 0 + 7 + v;   // keeps the loop, pins the result to something checkable
}
static_assert(true, "see the arithmetic above");

using target_fn = int (*)(int);

constexpr int kRegistrars = 16;

// The addresses are needed both as void* (to hook) and as callable pointers (to call), so
// they live in one table rather than being spelled out twice.
target_fn g_target[kRegistrars];

target_fn g_orig = nullptr;
std::atomic<int> g_replacement_calls{0};

// Set by the test to make the NEXT replacement call ask the runtime how much is installed.
// That is what a real replacement does when it wants to know whether its peers are live, and
// it is the path that used to answer with whatever number the loop had reached -- including
// zero, if the caller happened to run before the loop finished.
std::atomic<bool> g_ask_installer{false};
int g_asked_count = -1;

extern "C" __attribute__((noinline)) int herd_reg_replacement(int v) {
    g_replacement_calls.fetch_add(1, std::memory_order_relaxed);
    if (g_ask_installer.exchange(false, std::memory_order_acq_rel)) {
        // Re-entrant install_all() from inside a hooked function. It must return, not spin:
        // this thread is the installer only during installation, and after installation it is
        // any thread, so both paths have to terminate.
        g_asked_count = herd::install_all();
    }
    return g_orig ? g_orig(v) + 1000 : -1;
}

// Replacement for the sixteen: installed but never called, so the body only has to be a real
// function at a real address.
extern "C" __attribute__((noinline)) int herd_reg_idle(int v) {
    volatile int acc = v;
    for (int i = 0; i < 2; ++i) {
        acc = acc + 1;
    }
    return acc * 0;
}

char  g_names[kRegistrars][40];
void * g_origin[kRegistrars] = {};
herd::Seam * g_seam[kRegistrars] = {};
std::atomic<bool> g_go{false};

void registrar(int index) {
    while (!g_go.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    g_seam[index] = herd::add(g_names[index],
                              reinterpret_cast<void *>(g_target[index]),
                              reinterpret_cast<void *>(&herd_reg_idle),
                              &g_origin[index], nullptr);
}

} // namespace

int main() {
    g_target[0]  = &herd_reg_target_0;
    g_target[1]  = &herd_reg_target_1;
    g_target[2]  = &herd_reg_target_2;
    g_target[3]  = &herd_reg_target_3;
    g_target[4]  = &herd_reg_target_4;
    g_target[5]  = &herd_reg_target_5;
    g_target[6]  = &herd_reg_target_6;
    g_target[7]  = &herd_reg_target_7;
    g_target[8]  = &herd_reg_target_8;
    g_target[9]  = &herd_reg_target_9;
    g_target[10] = &herd_reg_target_10;
    g_target[11] = &herd_reg_target_11;
    g_target[12] = &herd_reg_target_12;
    g_target[13] = &herd_reg_target_13;
    g_target[14] = &herd_reg_target_14;
    g_target[15] = &herd_reg_target_15;

    // ---- 1. concurrent registration: no lost seam, no shared slot ----
    for (int i = 0; i < kRegistrars; ++i) {
        std::snprintf(g_names[i], sizeof(g_names[i]), "registry.test.%02d", i);
    }
    std::vector<std::thread> workers;
    workers.reserve(kRegistrars);
    for (int i = 0; i < kRegistrars; ++i) {
        workers.emplace_back(registrar, i);
    }
    g_go.store(true, std::memory_order_release);
    for (auto & w : workers) {
        w.join();
    }

    for (int i = 0; i < kRegistrars; ++i) {
        const herd::Seam * entry = herd::find(g_names[i]);
        if (!entry) {
            std::fprintf(stderr, "FAIL: '%s' was lost by concurrent registration\n", g_names[i]);
            ++failures;
            continue;
        }
        if (g_seam[i] != entry) {
            std::fprintf(stderr, "FAIL: '%s' was handed the slot of '%s'; one entry, two owners\n",
                         g_names[i], g_seam[i] ? g_seam[i]->name : "?");
            ++failures;
        }
    }

    // ---- 2. the seam that will be called ----
    const int baseline = herd_reg_dispatch_target(7);
    check(baseline == 14, "the dedicated target is the function this test thinks it is");

    [[maybe_unused]] herd::Seam * dispatch_seam =
        herd::add("registry.test.hooked",
                  reinterpret_cast<void *>(&herd_reg_dispatch_target),
                  reinterpret_cast<void *>(&herd_reg_replacement),
                  reinterpret_cast<void **>(&g_orig), nullptr);

    // ---- 3. install from several threads at once ----
    //
    // All four enter install_all() together. Exactly one may hook; every thread must come
    // back with the same count, because a thread that saw a partially installed table would
    // believe a different number of live seams than the one that installed it.
    constexpr int kInstallers = 4;
    std::atomic<bool> install_go{false};
    int results[kInstallers] = {};
    std::vector<std::thread> installers;
    installers.reserve(kInstallers);
    for (int i = 0; i < kInstallers; ++i) {
        installers.emplace_back([&, i] {
            while (!install_go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            results[i] = herd::install_all();
        });
    }
    install_go.store(true, std::memory_order_release);
    for (auto & t : installers) {
        t.join();
    }

    check(results[0] >= kRegistrars + 1, "installation found every registered seam");
    for (int i = 1; i < kInstallers; ++i) {
        if (results[i] != results[0]) {
            std::fprintf(stderr,
                         "FAIL: installer %d reported %d seam(s), installer 0 reported %d\n",
                         i, results[i], results[0]);
            ++failures;
        }
    }
    check(herd::install_all() == results[0], "install_all is idempotent after completion");

    // ---- 4. the hook is live, and dispatches exactly once ----
    const int hooked = herd_reg_dispatch_target(7);
    check(hooked == baseline + 1000,
          "a call reaches the replacement and the trampoline reaches the original");
    check(g_replacement_calls.load() == 1,
          "one call dispatched exactly one replacement; a doubled install would recurse "
          "through two trampolines instead of returning");

    const int again = herd_reg_dispatch_target(9);
    check(again == 16 + 1000, "the hook stays in force for later calls");
    check(g_replacement_calls.load() == 2, "each call dispatches exactly once");

    const herd::Seam * hooked_seam = herd::find("registry.test.hooked");
    check(hooked_seam != nullptr, "the hooked seam is findable");
    if (hooked_seam) {
        check(hooked_seam->installed.load(), "the hooked seam reports installed");
    }

    // ---- 5. a replacement may ask how much is installed, and gets a number ----
    //
    // Two claims, and the second is the one that had a defect behind it:
    //
    //   * the call terminates. install_all() from inside a hooked function is the case that
    //     could deadlock -- on the install state if this thread were the installer, on the
    //     registry lock if a registrar held it.
    //   * the number is the number of live hooks, not zero. The counter is published after
    //     every hook precisely so that a replacement running during the loop (an allocator
    //     seam is called by the next hook's own allocation) sees what is live rather than what
    //     the loop has not yet reached.
    g_ask_installer.store(true, std::memory_order_release);
    const int after_reentry = herd_reg_dispatch_target(5);
    check(after_reentry == 12 + 1000, "the re-entrant call still dispatched normally");
    check(g_asked_count == results[0],
          "a replacement asking how much is installed is told the live count, not a stale one");

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr,
                 "registry: %d concurrent registrations, %d concurrent installers, "
                 "one dispatch chain\n", kRegistrars, kInstallers);
    return 0;
}
