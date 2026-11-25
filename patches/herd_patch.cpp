// herd/patch.cpp — seam runtime plus the first real seams.
//
// This translation unit is compiled twice: once into the herd_seams object
// library (libraries only), once into herd-probe with HERD_PROBE_MAIN (adds the
// smoke test). Keeping the probe in the same TU costs one `#ifdef` and buys the
// guarantee that the test exercises exactly the runtime that ships.
#include "herd_patch.hpp"

#include <llama.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <dobby.h>

extern "C" {
// The linker synthesises these for a section called `herd_seams`. Every entry is
// a constant-initialized SeamDesc, so the range is readable before any dynamic
// initialization runs; that is what lets install_all() work from a
// constructor(101) in a binary whose main() we never touch.
extern const herd::SeamDesc __start_herd_seams[];
extern const herd::SeamDesc __stop_herd_seams[];

// Forces the section to exist even when this translation unit declares no seam:
// GNU ld only defines the __start_/__stop_ pair for sections that are present, so
// without an anchor a seam-less build would fail to link. The walker skips a
// descriptor with a null name, which is exactly what this is.
__attribute__((used, aligned(8), section("herd_seams")))
const herd::SeamDesc herd_seam_section_anchor[1] = {{nullptr, nullptr, nullptr, nullptr, nullptr}};
}

namespace {

constexpr int kMaxSeams = 64;
herd::Seam    g_seams[kMaxSeams];
// Published, not merely shared: see add(). The table is filled from a
// constructor, but find() is called from working threads and from static
// destructors, so the counter is the hand-off between those lifetimes.
std::atomic<int> g_seam_count{0};
// Read by add() and by the exit reports, written by install_all(). Atomic because
// add() is a public entry point: a host that registers a seam from a worker thread
// is doing what the header invites, and reading a plain bool here was a race that
// could also let a registration slip past a concurrent install.
std::atomic<bool> g_registration_closed{false};

// Installation is a three-state machine, and the state is the only thing that decides who
// installs. The previous version used `static bool done` plus `static int installed_count`
// inside install_all(): two threads entering together both read done == false and both ran
// the DobbyHook loop over the same entries. Hooking an address that already holds our jump
// builds a trampoline that leads back into the replacement -- recursion until the stack
// ends -- so the guard against double installation was itself the race.
//
//   idle -> installing -> live
//
// The winner is the thread that moves idle to installing. A thread that arrives while
// installing waits for live and then reads the published count: a caller never sees a
// half-installed table, and never a zero that means "not yet" instead of "nothing".
enum install_state : int { kInstallIdle = 0, kInstallInstalling = 1, kInstallLive = 2 };
std::atomic<int> g_install_state{kInstallIdle};
// Published by the installer with release, read by everyone else with acquire. Separate
// from the state so a waiter that has seen `live` cannot observe a stale count.
std::atomic<int> g_installed_count{0};
// True only in the thread that is installing. A replacement that called install_all()
// would otherwise wait on the state its own frame set, which is a self-deadlock; this
// makes it return the published count instead.
thread_local bool t_installing = false;

// A spin lock, not std::mutex, and the reason is the caller's context.
//
// Registration is closed by install_all(), which runs from a constructor(102). A
// namespace-scope std::mutex is not constant-initialized by the standard, so using one
// there is static-initialization-order roulette: whether the mutex is usable depends on
// which translation unit's dynamic initialization ran first.
//
// The state is a plain `std::atomic<bool>` and NOT a member of the lock class, which is a
// correction: the previous version held the flag as `std::atomic_flag flag_ =
// ATOMIC_FLAG_INIT;` inside registry_lock, with the same argument about constant
// initialization written above it. That argument was false. A default member initializer
// makes the defaulted default constructor non-trivial ([class.default.ctor]), so
// registry_lock acquired a dynamic initialization, and a registrar running at a
// constructor priority lower than whatever order the linker happened to pick would use an
// unconstructed lock. `std::atomic<bool> x{false}` is constant-initialized (constexpr
// constructor, constant argument), and the lock class below has no data members at all,
// so neither object has a dynamic initialization to lose the race.
//
// Held for four field writes in add() and for one count read in install_all(), never
// across a DobbyHook or any other call out of this file, so contention is meaningless.
std::atomic<bool> g_registry_held{false};

class registry_lock {
public:
    void lock() noexcept {
        bool expected = false;
        // compare_exchange_weak may fail spuriously, so `expected` is reset on every
        // attempt: leaving it set would turn a spurious failure into a "the lock is taken"
        // observation and spin forever against a free lock.
        while (!g_registry_held.compare_exchange_weak(expected, true,
                                                      std::memory_order_acquire,
                                                      std::memory_order_relaxed)) {
            expected = false;
        }
    }
    void unlock() noexcept { g_registry_held.store(false, std::memory_order_release); }
};

registry_lock g_registry_lock;
constexpr const char * kLog = "[herd]";

void fail(const char * name, const char * why) {
    std::fprintf(stderr, "%s FATAL seam '%s': %s\n", kLog, name, why);
    std::abort();
}

// Per-process chatter, off by default. A CLI's stderr is part of its output
// contract, and a hook that prints on every call would be a defect in a hot path
// rather than a diagnostic. HERD_SEAM_TRACE=1 turns it on.
bool trace_enabled() {
    static const bool enabled = std::getenv("HERD_SEAM_TRACE") != nullptr;
    return enabled;
}

// The one failure this runtime cannot turn into an abort: a program that links
// the runtime but never calls install_all() (or calls it before the registrars
// ran) looks healthy while executing pristine upstream. Nothing can force that
// call, so it is reported at exit instead of staying invisible.
struct uninstalled_report {
    ~uninstalled_report() {
        // "Ran unpatched" is exactly: seams exist, and installation never reached `live`.
        // Asking the install state -- rather than a bool that add() also consults -- keeps
        // two different questions apart: registration being closed is not the same fact as
        // installation having finished, and conflating them is what let a stale snapshot
        // pass for an installed table.
        const int count = g_seam_count.load(std::memory_order_acquire);
        if (count > 0 && g_install_state.load(std::memory_order_acquire) != kInstallLive) {
            std::fprintf(stderr,
                         "%s WARNING: %d seam(s) registered but never installed; "
                         "this process ran unpatched upstream code\n",
                         kLog, count);
        }
    }
} g_uninstalled_report;

} // namespace

namespace herd {

Seam * add(const char * name, void * target, void * replace, void ** origin, int * hits) {
    if (!name || !target || !replace || !origin) {
        std::fprintf(stderr, "%s add() got a null field for '%s'\n", kLog, name ? name : "?");
        std::abort();
    }
    // The whole registration step is one critical section. Checking a name and then claiming
    // a slot are two observations of the same table, and a registrar that does them
    // separately can pass the check, lose the race, and publish over another registrar's
    // entry -- a seam that exists in the accounting and never in the hooks.
    //
    // Held across four field writes and released before any Dobby call, so the hook path
    // never contends with registration.
    struct registry_guard {
        registry_guard()  { g_registry_lock.lock(); }
        ~registry_guard() { g_registry_lock.unlock(); }
    } guard;

    if (find(name)) {
        fail(name, "duplicate seam name; one fact has one owner");
    }
    // A seam registered after the installer closed registration would sit in the table
    // forever with nothing hooking it: the table is walked once. That is a silent loss of
    // exactly the kind this project refuses, so it is an abort, and the message says how to
    // fix the ordering.
    if (g_registration_closed.load(std::memory_order_acquire)) {
        fail(name, "registered after seams were installed; register from a constructor with a "
                   "priority lower than the installer's (102)");
    }
    if (g_seam_count.load(std::memory_order_relaxed) >= kMaxSeams) {
        fail(name, "seam table full; raise kMaxSeams rather than drop a patch");
    }
    // Fill the slot, THEN publish the count.
    //
    // The previous order was `Seam & s = g_seams[g_seam_count++]; s = Seam{...};`,
    // which increments the count before the entry means anything. A concurrent
    // reader walking the table (find, hits, the exit reports) could then index a
    // slot whose `name` was still garbage and hand it to strcmp -- a read of an
    // uninitialised pointer, i.e. the crash would land in the reader, in a
    // different thread, far from the writer. Registration happens in a
    // constructor, but readers do not: a worker thread may call a seam-reporting
    // entry point at any time after main() starts.
    //
    // The release store pairs with the acquire load in find(): the entry is
    // visible before the count that announces it.
    const int slot = g_seam_count.load(std::memory_order_relaxed);
    Seam & s = g_seams[slot];
    // Field by field, not `s = Seam{...}`: the published struct holds an atomic, so
    // it is not copy-assignable, and that is the point -- the entry is written
    // before it is counted, never after.
    s.name      = name;
    s.target    = target;
    s.replace   = replace;
    s.origin    = origin;
    s.hits      = hits;
    s.installed.store(false, std::memory_order_relaxed);
    g_seam_count.store(slot + 1, std::memory_order_release);
    return &s;
}

const Seam * find(const char * name) {
    // Acquire pairs with the release store in add(): a slot counted here has its
    // fields assigned. Reading the counter non-atomically would be a data race
    // with a registrar, and reading it without the fence would let the loop see
    // the count before the entry.
    const int count = g_seam_count.load(std::memory_order_acquire);
    for (int i = 0; i < count; ++i) {
        if (std::strcmp(g_seams[i].name, name) == 0) {
            return &g_seams[i];
        }
    }
    return nullptr;
}

int hits(const char * name) {
    const Seam * s = find(name);
    return (s && s->hits) ? *s->hits : -1;
}

int install_all() {
    // Idempotent on purpose. Hooking an entry that already holds our jump would build a
    // trampoline that leads back into the replacement, which recurses until the stack ends
    // -- so this must run exactly once per process, and "exactly once" has to be decided
    // before the loop rather than by a flag read inside it.
    int expected = kInstallIdle;
    if (!g_install_state.compare_exchange_strong(expected, kInstallInstalling,
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_acquire)) {
        if (g_install_state.load(std::memory_order_acquire) == kInstallInstalling) {
            // The installer marks its own thread, so a replacement that called back into
            // install_all() returns instead of waiting on a state its own frame set.
            //
            // The number it returns is "installed so far", which can be less than the number
            // of seams in the table -- and it is published after every hook for exactly this
            // reason. A hook installed early in the loop is already live; if it is an
            // allocator hook, the next DobbyHook's own allocation calls it, and that
            // replacement may ask the runtime how much is installed. Returning a count that
            // only moves at the end of the loop would answer zero while live hooks dispatched,
            // which is the "exists in the accounting, absent in the hooks" failure this file
            // refuses, mirrored. A partial count is the honest answer here; a zero is not.
            if (t_installing) {
                return g_installed_count.load(std::memory_order_acquire);
            }
            // Waiting is the correct outcome: two callers racing to install -- a constructor
            // and a test, say -- is a legitimate configuration, and aborting would turn a
            // benign race into a crash. The loop ends when the installer publishes `live`,
            // and the installer never calls back into anything that could wait on this
            // thread, so there is no inversion to deadlock.
            while (g_install_state.load(std::memory_order_acquire) == kInstallInstalling) {
            }
        }
        return g_installed_count.load(std::memory_order_acquire);
    }

    // From here this thread is the installer. The mark is thread-local state so that a
    // re-entrant call returns the published count instead of spinning on itself.
    struct installer_mark {
        installer_mark()  { t_installing = true; }
        ~installer_mark() { t_installing = false; }
    } mark;

    int installed_count = 0;

    // Ingest the section first: those entries need no registration step, which is
    // why an overlay can patch a binary it does not compile.
    //
    // Counted after the null-name test: the section anchor is a real descriptor
    // with no fields, and counting it made the trace line read "3 seam(s) live
    // (4 from the herd_seams section)", which sends the next reader looking for a
    // seam that does not exist.
    int from_section = 0;
    for (const herd::SeamDesc * d = __start_herd_seams; d != __stop_herd_seams; ++d) {
        if (!d->name) {
            continue; // the section anchor
        }
        static_cast<void>(add(d->name, d->target, d->replace, d->origin, d->hits));
        ++from_section;
    }

    // Registration closes here, and the snapshot is taken in the SAME critical section.
    // That is the whole fix. The previous version read the count outside the lock, then
    // re-read it inside and closed registration only if the two matched; when they did not
    // match -- a registrar had appended in between -- it closed nothing and still installed
    // the stale count. That registrar's seam stayed in the table, was counted by nobody, and
    // was hooked by nothing, while find() answered for it and every reader saw a valid
    // entry. A seam that exists in the accounting and never in the hooks is the one outcome
    // this runtime exists to make impossible, so the window is removed, not narrowed.
    int registered = 0;
    {
        struct registry_guard {
            registry_guard()  { g_registry_lock.lock(); }
            ~registry_guard() { g_registry_lock.unlock(); }
        } guard;
        registered = g_seam_count.load(std::memory_order_acquire);
        // Closed inside the lock: an add() that runs after this point either took the lock
        // first -- and is inside `registered` -- or takes it later and aborts with the
        // message that names the ordering rule. There is no third case left to lose a seam
        // in.
        //
        // Closed even when `registered` is zero. A binary with no seams today can gain one
        // from a constructor that ran at a higher priority than this one, and that seam
        // would never be hooked. The zero case below returns 0 because there is nothing to
        // install, not because registration was allowed to continue.
        g_registration_closed.store(true, std::memory_order_release);
    }

    if (registered == 0) {
        // Legitimate now that each module owns its own seams: a binary can link the runtime
        // and no patch module, and then there is nothing to install. Failing here would make
        // "which modules did you link" a load-time crash instead of a fact.
        if (trace_enabled()) {
            std::fprintf(stderr, "%s no seams in this image; nothing to install\n", kLog);
        }
        g_installed_count.store(0, std::memory_order_release);
        g_install_state.store(kInstallLive, std::memory_order_release);
        return 0;
    }
    for (int i = 0; i < registered; ++i) {
        Seam & s = g_seams[i];
        int rc = DobbyHook(s.target, s.replace, s.origin);
        if (rc != 0 || !*s.origin) {
            fail(s.name, "DobbyHook failed; upstream symbol moved or TU was inlined");
        }
        s.installed.store(true, std::memory_order_relaxed);
        ++installed_count;
        // Published per hook, not once at the end: a hook installed earlier in this loop is
        // live NOW, and its replacement can run during a later hook's Dobby call (an
        // allocator seam is the obvious case). Such a replacement asking how much is
        // installed must not be told zero. Release, so a reader that sees the count also sees
        // the `installed` flag of every seam it counts.
        g_installed_count.store(installed_count, std::memory_order_release);
        if (trace_enabled()) {
            std::fprintf(stderr, "%s seam '%s' @ %p\n", kLog, s.name, s.target);
        }
    }
    // The final store is now redundant in value but not in ordering: it keeps the
    // count-before-state pair visible as one step to a reader of this code, and it is the
    // store that publishes the last hook for anyone who only reads at the end.
    g_installed_count.store(installed_count, std::memory_order_release);
    g_install_state.store(kInstallLive, std::memory_order_release);
    if (trace_enabled()) {
        std::fprintf(stderr, "%s %d seam(s) live (%d from the herd_seams section)\n",
                     kLog, installed_count, from_section);
    }
    return installed_count;
}

void * resolve_in(const char * image, const char * symbol) {
    return DobbySymbolResolver(image, symbol);
}

void * resolve(const char * symbol) {
    return DobbySymbolResolver(nullptr, symbol);
}

} // namespace herd

// ---------------------------------------------------------------------------
// Auto-install: what makes this a product runtime instead of a test fixture.
//
// A binary that links herd_seams gets its seams installed before main(), with no
// edit to that binary's source. That is the only route an overlay has into an
// upstream executable we do not own -- llama-cli, llama-server, and everything
// else built straight from the submodule.
//
// Priority 101 is the earliest user priority: hooks are live before every
// ordinary dynamic initializer (65535) and before main(), so a hooked function
// called from another translation unit's static init is already patched. This
// ordering is only possible because the seam table is a linker section of
// constants; a registering global would itself run at 65535, after this.
//
// HERD_NO_AUTOLOAD=1 keeps the seams out, for a test that needs pristine
// upstream in the same binary.
// ---------------------------------------------------------------------------
// Priority 102, not 101: registration is a priority-101 job (a module that cannot
// express its seam as a section constant registers from its own constructor), and
// hooks must be installed after every registrar has run. With both at 101 the order
// between constructors is unspecified, and because install_all() is idempotent, a
// module whose registrar lost the race had its seam registered but never hooked --
// observed, not hypothesised: the prefab lane reported three seams live and its own
// entry was not among them. 102 still runs before every ordinary dynamic
// initializer (65535) and before main().
#if !defined(HERD_NO_AUTOLOAD)
__attribute__((constructor(102))) static void herd_autoload() {
    if (std::getenv("HERD_NO_AUTOLOAD") != nullptr) {
        return;
    }
    herd::install_all();
}
#endif

// ---------------------------------------------------------------------------
// Seams. Each one: keep the upstream body authoritative, observe it, and change
// only what this project actually needs to change.
//
// The three below are the smoke test's own seams, not features, so they are
// compiled only into herd-probe. Leaving them in the runtime that products link
// meant every product hooked llama_backend_init/free/llama_max_devices and wrote
// "intercepted" to stderr -- measured at two lines per run, on a CLI whose stderr
// is part of its output contract. A demonstration seam in a shipping binary is
// indistinguishable from a real one at the console, which is the defect.
// ---------------------------------------------------------------------------
#ifdef HERD_PROBE_MAIN

namespace {

int      g_backend_init_hits = 0;
int      g_backend_free_hits = 0;
int      g_max_devices_hits  = 0;
int      g_max_devices_delta = 0;

void (*g_orig_backend_init)(void)   = nullptr;
void (*g_orig_backend_free)(void)   = nullptr;
size_t (*g_orig_max_devices)(void)  = nullptr;

void herd_backend_init(void) {
    ++g_backend_init_hits;
    if (trace_enabled()) {
        std::fprintf(stderr, "%s llama_backend_init intercepted\n", kLog);
    }
    g_orig_backend_init();
}

void herd_backend_free(void) {
    ++g_backend_free_hits;
    if (trace_enabled()) {
        std::fprintf(stderr, "%s llama_backend_free intercepted\n", kLog);
    }
    g_orig_backend_free();
}

size_t herd_max_devices(void) {
    ++g_max_devices_hits;
    return g_orig_max_devices() + static_cast<size_t>(g_max_devices_delta);
}

} // namespace

HERD_SEAM(backend_init, "llama_backend_init", &llama_backend_init, &herd_backend_init,
          &g_orig_backend_init, &g_backend_init_hits)
HERD_SEAM(backend_free, "llama_backend_free", &llama_backend_free, &herd_backend_free,
          &g_orig_backend_free, &g_backend_free_hits)
HERD_SEAM(max_devices, "llama_max_devices", &llama_max_devices, &herd_max_devices,
          &g_orig_max_devices, &g_max_devices_hits)

#endif // HERD_PROBE_MAIN

#ifdef HERD_PROBE_MAIN
// ---------------------------------------------------------------------------
// Smoke test. Not a unit test of herd: an end-to-end proof that (a) hooks are
// installed before upstream runs, (b) upstream still executes underneath, and
// (c) ordinary callers observe the replacement.
// ---------------------------------------------------------------------------

// File-local symbol for the resolver check. Itanium ABI mangles
// `static int herdprobe(int)` as _ZL9herdprobei (length-prefixed name, `i` for
// int). Dobby's resolver reaches .symtab, so a static function is visible even
// though no other translation unit can name it -- that is the whole reason the
// seam layer can patch upstream's anonymous-namespace code.
static int herdprobe(int value) { return value + 1; }

int main() {
    // Resolve every symbol that has no header declaration BEFORE installing
    // seams. Measured ordering constraint: with a resolve() call sitting between
    // install_all() and llama_backend_init(), the init seam stopped firing while
    // the other two kept working. Resolving first removes the interaction and
    // keeps the check honest.
    //
    // Three claims, all measured here: a linked symbol resolves to the very
    // address the linker gave it; an absent symbol yields null rather than an
    // invented address; and a file-local static is reachable by mangled name --
    // which is the only way a seam can reach upstream's anonymous-namespace code.
    int resolver_failures = 0;
    const void * resolved_init = herd::resolve("llama_backend_init");
    if (resolved_init != herd::raw(&llama_backend_init)) {
        std::fprintf(stderr, "probe FAIL: resolver returned %p, linked address is %p\n",
                     resolved_init, herd::raw(&llama_backend_init));
        ++resolver_failures;
    }
    if (herd::resolve("herd_no_such_symbol_exists_") != nullptr) {
        std::fprintf(stderr, "probe FAIL: resolver invented an address for an absent symbol\n");
        ++resolver_failures;
    }
#if defined(__GNUC__) || defined(__clang__)
    if (herd::resolve("_ZL9herdprobei") != herd::raw(&herdprobe)) {
        std::fprintf(stderr, "probe FAIL: file-local static not resolved by mangled name\n");
        ++resolver_failures;
    }
#endif
    if (resolver_failures) {
        return 1;
    }

    const int installed = herd::install_all();
    // Second call must be a no-op returning the same count: re-hooking would
    // make the trampoline jump back into our own replacement.
    const int again = herd::install_all();
    if (installed != 3 || again != installed) {
        std::fprintf(stderr, "probe FAIL: installed %d seams (%d on second call), expected 3 and 3\n",
                     installed, again);
        return 1;
    }

    int failures = 0;

    // This call is the seam's only trigger. It went missing during an earlier
    // investigation, and the resulting hits=0 was read as "the resolver breaks
    // the first seam" -- a wrong diagnosis that cost a cycle. A seam with no
    // caller is not a broken patch; it is a test that never exercised it.
    llama_backend_init();

    // The replacement must run, and the original must still run under it: a
    // device count of zero would mean we swallowed upstream instead of calling it.
    g_max_devices_delta = 100000;
    const size_t with_delta = llama_max_devices();
    g_max_devices_delta = 0;
    const size_t baseline = llama_max_devices();

    llama_backend_free();

    const int init_hits = herd::hits("llama_backend_init");
    const int free_hits = herd::hits("llama_backend_free");
    const int dev_hits  = herd::hits("llama_max_devices");

    std::fprintf(stderr, "probe: hits init=%d free=%d devices=%d baseline=%zu delta=%zu\n",
                 init_hits, free_hits, dev_hits, baseline, with_delta);
    if (init_hits != 1)  { std::fprintf(stderr, "probe FAIL: llama_backend_init hits=%d\n", init_hits); ++failures; }
    if (free_hits != 1)  { std::fprintf(stderr, "probe FAIL: llama_backend_free hits=%d\n", free_hits); ++failures; }
    if (dev_hits != 2)   { std::fprintf(stderr, "probe FAIL: llama_max_devices hits=%d\n", dev_hits); ++failures; }
    if (baseline == 0)   { std::fprintf(stderr, "probe FAIL: upstream llama_max_devices() returned 0; original not called\n"); ++failures; }
    if (with_delta != baseline + 100000) { std::fprintf(stderr, "probe FAIL: replacement value not observed\n"); ++failures; }

    if (failures) {
        return 1;
    }

    // Resolution must keep working with seams already installed, because a patch
    // module resolves file-local addresses at registration time -- after install.
    // The earlier "resolver breaks the first seam" reading was wrong, and this
    // check is what keeps that claim from being believed again without evidence.
#if defined(__GNUC__) || defined(__clang__)
    if (herd::resolve("_ZL9herdprobei") != herd::raw(&herdprobe)) {
        std::fprintf(stderr, "probe FAIL: resolver stopped working after seams were installed\n");
        ++failures;
    }
#endif
    if (failures) {
        return 1;
    }
    std::fprintf(stderr, "probe OK: 3 seams live, upstream bodies run underneath\n");
    return 0;
}
#endif
