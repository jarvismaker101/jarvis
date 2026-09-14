// lib/retry.mjs
//
// Pure helper for mid-action browser-death recovery. Playwright reports a
// dead browser/page with a family of similar messages; isClosedError tells
// whether an error is one of them so the server can relaunch and retry
// instead of surfacing the failure to the model.

const CLOSED_PHRASES = [
  "target page, context or browser has been closed",
  "browser has been closed",
  "target closed",
  "target page or context has been closed",
  // Renderer-crash deaths: the page object lingers but is unusable; the
  // sane recovery is the same full relaunch.
  "page crashed",
  "page closed unexpectedly",
]

export function isClosedError(err) {
  if (!err || typeof err.message !== "string") return false
  const msg = err.message.toLowerCase()
  return CLOSED_PHRASES.some((phrase) => msg.includes(phrase))
}
