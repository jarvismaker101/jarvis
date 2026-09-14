# AGENTS.md — standing rules (auto-loaded every session)

## Context discipline
- Never auto-compact context at 50k tokens or any other token threshold — there is no auto-compaction rule.
- Keep full context for quality — do not auto-compact or summarize-and-drop history mid-task.
- Read only the needed slices, not whole threads/outputs.
- Prefer file redirection for large tool output.
