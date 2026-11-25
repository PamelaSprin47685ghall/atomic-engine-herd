// herd/patch.hpp — seam declaration surface.
//
// A seam is one function-level patch: an upstream address, a replacement with
// the same ABI, and a slot where Dobby writes the trampoline to the original.
//
// Seams live in a linker section, not in registering globals. Each HERD_SEAM
// emits a constant-initialized SeamDesc into `herd_seams`, so the table is valid
// the instant the image is loaded, and the runtime installs it from a
// constructor at priority 101 -- before every ordinary dynamic initializer and
// before main(). A registering global would itself be a priority-65535
// initializer, which runs *after* the constructor that needs its entry; that is
// why an overlay cannot rely on static-init registration to patch a binary whose
// main() it does not own.
//
// Constant initialization also matters for what a descriptor may contain: only
// address constants (a function pointer, the address of a global, a string
// literal). A member function pointer does not qualify in the Itanium ABI, so
// `add()` remains for those, called from code that runs later.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <type_traits>

namespace herd {

struct Seam {
    const char * name;   // diagnostic identity, also the lookup key
    void *       target; // upstream entry point being replaced
    void *       replace;// our function, identical ABI
    void **      origin; // Dobby writes the callable original here
    int *        hits;   // incremented by the replacement, for tests
    // Atomic, because the writer and the readers are on different sides of a lock.
    // install_all() stores true under the registry mutex; the readers -- the exit
    // reports and tests polling seam state -- take no lock, and a plain bool read
    // there is a data race on a byte that decides whether a binary reports itself
    // as patched. The type is also why Seam cannot be copy-assigned once published:
    // add() fills the fields one at a time for that reason.
    std::atomic<bool> installed;
};

// One entry of the `herd_seams` section. Deliberately a separate type from Seam:
// this one must be constant-initializable, so it carries no bool and no computed
// state, only addresses. The runtime copies each descriptor into its Seam table
// at install time.
struct SeamDesc {
    const char * name;
    void *       target;
    void *       replace;
    void **      origin;
    int *        hits;
};

// Append a seam to the static table. Every rejected input is a programming
// error and aborts here: a seam that silently fails to register would leave the
// program running upstream code while claiming to be patched.
[[nodiscard]] Seam * add(const char * name, void * target, void * replace, void ** origin, int * hits);

// Install every registered seam. Call once, first thing in main(), before any
// upstream function that might be hooked. Aborts on failure: a seam that does
// not install means the upstream symbol moved, and continuing would run an
// unpatched engine while claiming to be patched. Idempotent: a second call is a
// no-op that returns the same count, because re-hooking an already patched entry
// would point the trampoline at our own replacement.
int install_all();

const Seam * find(const char * name);
int          hits(const char * name);

// Resolve an upstream symbol by name. Needed for file-local functions (static,
// anonymous namespace) and for anything without a header declaration. Returns
// null when the running image has no such symbol.
void * resolve(const char * symbol);
void * resolve_in(const char * image, const char * symbol);

// Convert "any callable" to the void* Dobby wants. Works for free functions,
// static members and non-virtual member functions. Virtual members are a vtable
// offset, not an address: resolve those by mangled name instead.
template <typename T>
inline void * raw(T fn) {
    static_assert(std::is_pointer_v<T> || std::is_member_function_pointer_v<T>,
                  "herd::raw needs a function pointer");
    if constexpr (std::is_member_function_pointer_v<T>) {
        static_assert(sizeof(T) == sizeof(void *), "member function pointer must fit a void*");
        union { T fn; void * ptr; } u{fn};
        // An Itanium-ABI pointer to a *virtual* member function is a vtable
        // offset (odd, or far below any mapped text), not a code address.
        // Handing that to DobbyHook writes over the vtable, which fails much
        // later and somewhere else; refuse it at the source.
        const auto address = reinterpret_cast<std::uintptr_t>(u.ptr);
        if (address != 0 && ((address & 1u) != 0 || address < 0x1000u)) {
            std::fprintf(stderr,
                         "[herd] FATAL: herd::raw got a virtual member function "
                         "(vtable offset %p); resolve it by mangled name instead\n",
                         u.ptr);
            std::abort();
        }
        return u.ptr;
    } else {
        return reinterpret_cast<void *>(fn);
    }
}

} // namespace herd

#define HERD_CAT_I(a, b) a##b
#define HERD_CAT(a, b) HERD_CAT_I(a, b)

// HERD_SEAM(tag, "name", &upstream_fn, &our_fn, &orig_slot, &hit_counter)
// The tag only names the generated variable; it must be unique per translation
// unit. `origin` is the address of the variable Dobby should fill in.
#define HERD_SEAM(tag, name_text, target, replace, origin, hits_ptr)                       \
    extern "C" {                                                                           \
    /* No namespace, no constructor: this is an initialized-at-link-time constant.        \
       The two casts are the only expressions, and both are address constants, so no      \
       dynamic initializer is emitted and nothing has to run before it can be read. */      \
    /* An array of one, explicitly aligned to pointer size. Without the array,    \
       each object got 64-byte section alignment on this toolchain, so the        \
       entries were 64 bytes apart while the walker stepped by sizeof(SeamDesc)   \
       = 40: __stop - __start counted wrong and the walk read padding as fields.  \
       An array makes the section's alignment the element's, and the stride       \
       sizeof(SeamDesc) again. */                                                  \
    __attribute__((used, aligned(8), section("herd_seams")))                        \
    const ::herd::SeamDesc HERD_CAT(herd_seam_, tag)[1] = {{                        \
        name_text,                                                                  \
        reinterpret_cast<void *>(target),                                           \
        reinterpret_cast<void *>(replace),                                          \
        reinterpret_cast<void **>(origin),                                          \
        hits_ptr,                                                                   \
    }};                                                                                      \
    }

// Member functions cannot go through HERD_SEAM: converting a member pointer to
// void* is not an address constant in the Itanium ABI, so the descriptor would
// need a dynamic initializer and the whole ordering guarantee above would be
// lost. Use herd::add() from a call site instead.

// The section macros above only expand inside a translation unit that includes
// this header; `herd_seams` is the section name the runtime walks.
#define HERD_SEAM_SECTION "herd_seams"
