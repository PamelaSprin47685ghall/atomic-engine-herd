#include <tq/relay/loopback.h>

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace tq::relay {
namespace {

// A small counting gate. It is the only waiting primitive in the transport and it
// exists for one reason: the reducer must not spin while five producers work, and
// a spin would burn a core per rank on a machine that has no GPU to wait for.
class gate {
  public:
    void arrive() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++arrived_;
        }
        cv_.notify_all();
    }

    // Blocks until `target` arrivals. Returns false if `stop` became true first,
    // so a failed rank can release the reducer instead of deadlocking it.
    bool wait(std::size_t target) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return arrived_ >= target || stop_; });
        return arrived_ >= target;
    }

    void abort() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
    }

  private:
    std::mutex              mutex_;
    std::condition_variable cv_;
    std::size_t             arrived_ = 0;
    bool                    stop_    = false;
};

} // namespace

epoch_report loopback_channel::run_epoch(std::uint64_t epoch, const producer_fn & producer,
                                         const consumer_fn & consumer) {
    epoch_report report;
    report.epoch = epoch;
    last_        = report;

    if (epoch == 0) {
        report.error = -1;
        last_        = report;
        return report;
    }

    // A plane still committing an older epoch is not reusable: publishing into it
    // would overwrite a message a consumer may be reading. The caller serialises
    // epochs; this check makes a serialisation bug loud instead of tearing data.
    if (const std::uint64_t committed = readable_epoch(plane_); committed >= epoch) {
        report.error = -2;
        last_        = report;
        return report;
    }

    gate published;   // producers -> reducer
    gate committed;   // reducer  -> consumers
    gate acked;       // consumers -> caller
    std::atomic<std::int32_t> first_error{0};
    std::atomic<std::uint64_t> early_generation{0};

    std::vector<std::thread> threads;
    threads.reserve(control_ranks * 2);

    for (std::size_t rank = 0; rank < control_ranks; ++rank) {
        threads.emplace_back([&, rank] {
            // Each rank writes into its own slice of the shared store. Slices do not
            // overlap, so five producers need no lock between them -- which is the
            // property the real transport relies on and this one must therefore
            // exercise rather than assume.
            std::uint32_t * slice = store_.words.data() + rank * (store_.words.size() / control_ranks);
            const std::size_t count = store_.words.size() / control_ranks;
            try {
                if (producer) {
                    producer(epoch, rank, slice, count);
                }
            } catch (...) {
                const std::int32_t code = -3;
                if (first_error.load() == 0) { first_error.store(code); }
                publish_error(plane_, rank, code);
                published.arrive();
                return;
            }
            if (!publish_input(plane_, rank, epoch)) {
                if (first_error.load() == 0) { first_error.store(-4); }
                publish_error(plane_, rank, -4);
            }
            published.arrive();
        });
    }

    threads.emplace_back([&] {
        // The reducer waits for all five. A rank that failed still arrives (with its
        // error word set), so this wait always terminates: a returning thread that
        // forgot to arrive would be the one bug this design cannot tolerate, and the
        // timeout in the test is what would catch it.
        const bool complete = published.wait(control_ranks);
        if (!complete) {
            committed.abort();
            return;
        }
        const readiness seen = observe(plane_, epoch);
        report.ready = seen.mask;
        if (seen.failed || !seen.complete) {
            if (first_error.load() == 0) { first_error.store(seen.error ? seen.error : -5); }
            // The reducer's own failure is NOT recorded in the plane, because the
            // plane has no slot for it: the fork's control block is exactly
            // generation + ready[5] + ack[5] + error[5] (relay.hip:98-105), and a
            // CPU-owned error word would have shifted every offset after it. The
            // device protocol expresses a reducer failure the only way it can: the
            // commit never happens, so no rank's wait can succeed. Here the report
            // carries the code, which is what an in-process caller needs.
            committed.abort();
            return;
        }

        // Reduce: rank i's words go to output[i * width .. (i+1) * width). The layout
        // is deliberately simple; what matters is that the whole output is written
        // before the commit, which is the ordering the plane enforces.
        const std::size_t width = store_.words.size() / control_ranks;
        for (std::size_t rank = 0; rank < control_ranks; ++rank) {
            std::copy_n(store_.words.begin() + std::ptrdiff_t(rank * width), width,
                        store_.output.begin() + std::ptrdiff_t(rank * width));
        }
        if (!commit_output(plane_, epoch)) {
            if (first_error.load() == 0) { first_error.store(-6); }
            committed.abort();
            return;
        }
        committed.arrive();
    });

    for (std::size_t rank = 0; rank < control_ranks; ++rank) {
        threads.emplace_back([&, rank] {
            if (!committed.wait(1)) {
                // No commit: nothing to read, and acking would be a lie about a
                // message that never existed.
                acked.arrive();
                return;
            }
            const std::uint64_t readable = readable_epoch(plane_);
            if (readable < epoch) {
                if (first_error.load() == 0) { first_error.store(-7); }
            }
            // A reader must never see a generation it cannot yet read: the plane's
            // commit word is what gates this, and recording the highest value seen
            // makes a violation visible even if the data happened to look right.
            const std::uint64_t seen = plane_.generation.load(std::memory_order_acquire);
            if (seen != 0) {
                std::uint64_t previous = early_generation.load();
                while (previous < seen && !early_generation.compare_exchange_weak(previous, seen)) {
                }
            }
            const std::size_t width  = store_.output.size() / control_ranks;
            const std::uint32_t * slice = store_.output.data() + rank * width;
            if (consumer) {
                consumer(epoch, rank, slice, width);
            }
            if (!ack_output(plane_, rank, epoch)) {
                if (first_error.load() == 0) { first_error.store(-8); }
            }
            acked.arrive();
        });
    }

    const bool all_acked = acked.wait(control_ranks);
    for (std::thread & thread : threads) {
        thread.join();
    }

    report.error                 = first_error.load();
    report.observed_before_commit = early_generation.load();
    report.acked                 = 0;
    for (std::size_t rank = 0; rank < control_ranks; ++rank) {
        if (plane_.ack[rank].load(std::memory_order_acquire) == epoch) {
            report.acked |= 1u << rank;
        }
    }
    report.ok = all_acked && report.error == 0 && report.ready == ((1u << control_ranks) - 1u) &&
                report.acked == ((1u << control_ranks) - 1u);

    // End the epoch: the only point at which error words may be cleared, run after
    // every thread has been joined so no rank can still be writing one.
    //
    // On a failed epoch this refuses, and refusal is the correct answer rather than a
    // leak: a failed epoch never commits, so it never earns acknowledgements, so its
    // error words stay set and the channel is visibly poisoned. Recovery is
    // initialise_plane -- mirroring the device, where a failed channel is torn down
    // instead of repaired.
    //
    // Said plainly, because it matters for anyone reading this as a fix: on the
    // success path the call is a no-op today, since a successful epoch has no error
    // words to clear. It is kept because it is the checked form of the only
    // legitimate clear, and because a protocol where a robust epoch may survive a
    // rank error would need exactly this call in exactly this place.
    const bool concluded = conclude_epoch(plane_, epoch);
    if (report.ok && !concluded) {
        // A contradiction: report.ok was computed from the acknowledgements this call
        // just re-read. Reporting it beats discarding it -- which is what the earlier
        // version did with the result of a call that could never succeed.
        report.error = report.error != 0 ? report.error : -9;
        report.ok    = false;
    }

    last_ = report;
    return report;
}

} // namespace tq::relay
