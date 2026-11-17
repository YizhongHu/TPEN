# Staged Task Orchestrator canonical skill bundle

This directory is the reviewable, **not deployed** Slice A payload.  It does
not change a live skill directory, plugin cache, settings, or the running
server.

Canonical v4 skill/reference content lives in the upstream
`jpicklyk/task-orchestrator` tree at immutable commit
`c25191bfa9e825882ac0da05a4f7373e543633cd` (annotated tag `v4.0.2`).  The
manifest records every v3 canonical skill's v4 counterpart and its pinned
upstream path.  Reviewers inspect this mapping together with the immutable
upstream files before approving a future cutover copy operation.

At cutover, an operator copies only reviewed manifest entries into the chosen
live skill distribution, then validates their actual calls on the new server.
No hand-edit of a plugin cache is permitted.

## Review procedure

1. Verify the upstream checkout is exactly the commit above.
2. For every row in `manifest.tsv`, inspect that exact source path and confirm
   it contains no v3 tool names or v3 REST call shape.
3. Run the scratch validation described in the PR receipt against a fresh
   `to-mig-scratch-*` v4.0.2 server.
4. Approve this repository PR only; activation remains operator-only cutover
   work.
