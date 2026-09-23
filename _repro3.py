"""Faithful repro: exactly how /ask/stream's worker calls process_message."""
import faulthandler, os, sys, time
faulthandler.enable()
faulthandler.dump_traceback_later(float(os.getenv("REPRO_TIMEOUT", "60")), exit=True)

from backend.core.brain import process_message
from backend.services import jobs as job_registry

MSG = sys.argv[1]
print("[REPRO3] asking: %r" % MSG, flush=True)

deltas = []
def on_delta(t):
    deltas.append(t)
def on_progress(message, **kw):
    print("   [progress] %s %s" % (message, kw or ""), flush=True)

job = job_registry.new_job(kind="request", label=MSG[:80])
t0 = time.time()
try:
    reply = process_message(
        MSG, from_voice=False, stream_reply=on_delta, progress=on_progress,
        request_id="repro-1", job=job,
    )
    print("[REPRO3] REPLY in %.2fs: %r" % (time.time()-t0, (reply or "")[:300]), flush=True)
    print("[REPRO3] deltas: %d  chars=%d" % (len(deltas), sum(len(d) for d in deltas)), flush=True)
except BaseException as exc:
    import traceback
    print("[REPRO3] RAISED after %.2fs: %s: %s" % (time.time()-t0, type(exc).__name__, exc), flush=True)
    traceback.print_exc()
