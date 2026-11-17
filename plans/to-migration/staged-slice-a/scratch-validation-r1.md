# Slice A scratch validation r1

- Image: `ghcr.io/jpicklyk/task-orchestrator@sha256:e2218297471b55e77b68c96c335cf6f7dde2f9beaf6cc53102624cfb21465e12`
- Isolated objects: `to-mig-scratch-a-402`, `to-mig-scratch-a-data`.
- Health: PASS, `{"status":"ok","dbReachable":true}`.
- MCP `tools/list`: 12 tools.  `items_read`, `notes_read`, and `notes_write`
  were present; legacy `manage_*`, `query_*`, `advance_item`, and
  `claim_item` were absent.
- Mapping gap: expected `claims_write` was absent.  Therefore the claim/release
  mapping is intentionally not asserted and Slice A is **not approval-ready**
  until the actual v4 claim operation is identified and added to the manifest.
- Teardown: PASS.  The scratch container and volume were removed immediately.
