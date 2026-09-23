"""Reproduce the 'gpt 6 astra' failure in a subprocess with a hard timeout."""
import faulthandler, os, sys, threading, time

faulthandler.enable()
# Dump every thread's stack if we're still stuck after N seconds.
TIMEOUT = float(os.getenv("REPRO_TIMEOUT", "45"))
faulthandler.dump_traceback_later(TIMEOUT, exit=True)

from backend.core.brain import process_message

MSG = sys.argv[1] if len(sys.argv) > 1 else "whats the current pricing for gpt 6 astra"
print("[REPRO] asking: %r" % MSG, flush=True)
t0 = time.time()
try:
    reply = process_message(MSG, from_voice=False)
    print("[REPRO] REPLY in %.2fs: %r" % (time.time() - t0, (reply or "")[:400]), flush=True)
except BaseException as exc:
    import traceback
    print("[REPRO] RAISED after %.2fs: %s: %s" % (time.time() - t0, type(exc).__name__, exc), flush=True)
    traceback.print_exc()
