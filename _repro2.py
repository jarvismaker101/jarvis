"""Run the research path the 'gpt 6 astra' question triggered, with a watchdog."""
import faulthandler, os, time
faulthandler.enable()
faulthandler.dump_traceback_later(float(os.getenv("REPRO_TIMEOUT", "90")), exit=True)

from backend.services.quick_search import run_quick_search

q = "current pricing for gpt 6 astra"
print("[REPRO2] run_quick_search(%r)" % q, flush=True)
t0 = time.time()
try:
    r = run_quick_search(q)
    print("[REPRO2] returned in %.2fs" % (time.time() - t0), flush=True)
    print("[REPRO2] keys:", sorted(r.keys()), flush=True)
    print("[REPRO2] spoken:", repr(r.get("spoken_summary"))[:300], flush=True)
    print("[REPRO2] overview_found:", r.get("overview_found"), flush=True)
except BaseException as exc:
    import traceback
    print("[REPRO2] RAISED after %.2fs: %s: %s" % (time.time()-t0, type(exc).__name__, exc), flush=True)
    traceback.print_exc()
