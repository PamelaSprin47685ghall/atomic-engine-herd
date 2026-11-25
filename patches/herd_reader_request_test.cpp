// What an explicit reader request does when the build cannot honour it.
//
// The property under test is one sentence: A REQUEST IS ANSWERED. Not honoured, not ignored --
// answered. Two variables carry the lane's requests, so all four combinations are run against the
// real product binary:
//
//   neither                 no lane, no output, exit 0
//   HERD_READER_ATTN only   the lane installs, exit 0
//   HERD_READER_EMIT_NOW    a request to replace attention in a program that does not run the
//                           lane: one line on stderr, exit 0 -- there is nothing here to replace.
//                           It USED to print only when HERD_SEAM_TRACE was also set, which made
//                           an ordinary request silent. That is the defect this case pins, and it
//                           is why the two --help cases below carry the weight: they need no model.
//   both                    the lane runs and is asked to replace: exit 2, no stdout, a message
//                           naming what the build lacks.
//
// Why the third and fourth differ, since it looks inconsistent: the constructor runs before `main`
// in EVERY program that links the runtime, including ones that never touch attention. Killing
// `--help` because an unrelated export is set would punish the wrong program for a request it
// cannot act on. So the refusal that stops the process belongs where the request would take
// effect -- in the hook, at the first layer -- and the constructor's job is to say what it saw.
//
// The two attention cases need a model AND a binary corpus (llama-corpus reads a directory of
// nodes-*.bin / requests-*.bin, not text; the first version of this file passed the model path and
// then a text directory, so both cases failed for a reason unrelated to the contract). They are
// skipped with a printed note when either is absent, so a run never reports a pass it did not earn.
//
//   usage: herd-reader-request-test <product-binary> [model.gguf [corpus-dir]]
#include <sys/wait.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string & what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

struct run_result {
    int exit_code = -1;
    std::string out;
    std::string err;
};

std::string slurp(const std::string & path) {
    std::string text;
    if (FILE * f = std::fopen(path.c_str(), "rb")) {
        char buffer[4096];
        std::size_t n = 0;
        while ((n = std::fread(buffer, 1, sizeof(buffer), f)) > 0) {
            text.append(buffer, n);
        }
        std::fclose(f);
    }
    return text;
}

// Runs `binary args...` with the reader switches cleared, then set as `env` says.
//
// Every case starts from a clean slate: a value inherited from whoever launched the test would
// otherwise leak into a case that means to run without it.
run_result run(const std::string & binary, const std::vector<std::string> & args,
               const std::vector<std::pair<std::string, std::string>> & env) {
    const std::string out_path = "/tmp/herd_req_" + std::to_string(::getpid()) + ".out";
    const std::string err_path = "/tmp/herd_req_" + std::to_string(::getpid()) + ".err";

    std::string command = "env -u HERD_READER_ATTN -u HERD_READER_EMIT_NOW -u HERD_SEAM_TRACE";
    for (const auto & entry : env) {
        command += " " + entry.first + "=" + entry.second;
    }
    command += " '" + binary + "'";
    for (const auto & arg : args) {
        command += " '" + arg + "'";
    }
    command += " >'" + out_path + "' 2>'" + err_path + "'";

    run_result result;
    const int status = std::system(command.c_str());
    result.exit_code = (status != -1 && WIFEXITED(status)) ? WEXITSTATUS(status) : -1;
    result.out = slurp(out_path);
    result.err = slurp(err_path);
    std::remove(out_path.c_str());
    std::remove(err_path.c_str());
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <product-binary> [model.gguf [corpus-dir]]\n", argv[0]);
        return 2;
    }
    const std::string binary = argv[1];
    const std::string model = argc > 2 ? argv[2] : "";
    const std::string corpus = argc > 3 ? argv[3] : "";

    // ---- 1. No request: nothing printed, exit 0. The baseline the others are measured against.
    {
        const run_result r = run(binary, {"--help"}, {});
        check(r.exit_code == 0, "no request: --help exits 0");
        check(r.err.find("[herd] reader") == std::string::npos,
              "no request: the reader lane says nothing at all");
    }

    // ---- 2. A replacement request in a program that does not run the lane: one line, exit 0.
    //
    // The case the previous build got wrong: it printed only under HERD_SEAM_TRACE, so an ordinary
    // invocation of a tool that cannot replace attention said nothing at all.
    {
        const run_result r = run(binary, {"--help"}, {{"HERD_READER_EMIT_NOW", "1"}});
        check(r.exit_code == 0,
              "EMIT_NOW alone: exit 0 -- this program has no attention to replace, so nothing is "
              "refused, but it must not be killed for a request it cannot act on");
        check(r.err.find("HERD_READER_EMIT_NOW asks for a layer's attention to be replaced") !=
                  std::string::npos,
              "EMIT_NOW alone: the request is ANSWERED on stderr, unconditionally -- not only "
              "under a trace switch, which is what made it silent");
        check(r.err.find("nothing was replaced") != std::string::npos,
              "EMIT_NOW alone: and the line says plainly that nothing was replaced");
        check(r.out.find("[herd]") == std::string::npos,
              "EMIT_NOW alone: nothing reaches stdout, so a piped consumer sees no phantom output");
    }

    // ---- 3. The lane without a replacement request: installs, exits 0, no refusal line.
    {
        const run_result r = run(binary, {"--help"}, {{"HERD_READER_ATTN", "1"}});
        check(r.exit_code == 0, "ATTN alone: --help still exits 0");
        check(r.err.find("HERD_READER_EMIT_NOW asks") == std::string::npos,
              "ATTN alone: no replacement request, so no refusal line");
    }

    // ---- 4 and 5 need a model and a real corpus: the request only becomes refusable when
    //      attention is reached, and reaching it needs a runnable input.
    if (model.empty() || corpus.empty()) {
        std::fprintf(stderr,
                     "model or corpus not given: cases 4-5 (the refusal at attention) were NOT "
                     "run; this run checked the constructor's behaviour, which is where the "
                     "previous defect lived\n");
        if (failures) {
            std::fprintf(stderr, "%d failure(s)\n", failures);
            return 1;
        }
        std::fprintf(stderr, "reader requests: constructor behaviour answered in all combinations\n");
        return 0;
    }

    // 4. Both switches: the lane is asked to replace, and refuses.
    {
        const run_result r = run(binary, {"score", "-m", model, "--corpus", corpus},
                                 {{"HERD_READER_ATTN", "1"}, {"HERD_READER_EMIT_NOW", "1"}});
        check(r.exit_code == 2,
              "both: exit 2 -- a request that cannot be honoured stops the run instead of "
              "producing stock numbers under a reader label");
        check(r.out.empty(),
              "both: no stdout at all, so no consumer reads a result from the refused "
              "configuration");
        check(r.err.find("was asked to REPLACE attention") != std::string::npos,
              "both: the refusal says which layer and what was asked");
        check(r.err.find("Missing, in build order") != std::string::npos,
              "both: and names what the build lacks, in the order it must be built, so the reader "
              "is not left guessing");
    }

    // 5. Neither switch, with a model: the ordinary run, unaffected.
    {
        const run_result plain = run(binary, {"score", "-m", model, "--corpus", corpus}, {});
        check(plain.exit_code == 0, "neither: the ordinary run exits 0");
        check(plain.err.find("[herd] reader") == std::string::npos,
              "neither: and the reader lane leaves no trace in the log");
    }

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr,
                 "reader requests: answered in every combination -- silent when nothing was asked, "
                 "refused where the refusal can act\n");
    return 0;
}
