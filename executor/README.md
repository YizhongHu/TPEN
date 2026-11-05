# tpen-executor

`tpen-executor` is the independent project boundary for TPEN execution
orchestration.  Its staged Hydra configuration preserves future-process
programs as unresolved specifications; owner processes materialize those
programs with their local environment.

Runtime dependencies are `hydra-core==1.3.2` and `omegaconf==2.3.0`.  The
development dependency is the same pinned pytest release used by the root
project.

The worker launches the selected science process through the `-c` shim
`<prepared-python> -c 'import sys; from tpen.run import main; raise SystemExit(main(sys.argv[1:]))' --config <path> [dotlist overrides...]`
and collects evidence with `<prepared-python> -m tpen.inspection <run_dir>`.
The core module guard for direct module execution is deferred to a core
follow-up. The executor never imports
`tpen` or `experiments`; all scientific behavior and artifact reads remain in
the child process, while the executor decodes only the inspection JSON output.

Program authoring rules are intentionally explicit. A program marker may have
several named concrete target slots; `materialize()` returns a frozen mapping
and attribute view of those slots. Nested program markers remain handles until
their owning process materializes them. Marker subtrees must not use key-taking
resolver arguments, references from a parent marker into a nested marker's
interior, or submitter-interpolated intermediate nodes on keys referenced by a
program. `${oc.env:NAME}` without a key-taking argument remains fine. Pass
values that exist only in the owner process through the `materialize()`
environment. Descriptor `env:` values must be strings; unquoted YAML integers
are rejected. The packaged config demonstrates independent
control and science environment slots; they may point at the same prepared
environment, but no descriptor provisions, writes, or existence-checks paths.

From this directory, install the locked environment and run the executor
tests with:

```text
cd executor && uv sync --locked && uv run pytest -q tests
```
