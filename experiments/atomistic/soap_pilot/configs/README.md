# Rendered pilot configuration directory

The composer writes resolved, invocation-specific YAML here (or to a caller's
output directory). These files are intentionally not hand-edited: each row
stores the exact bytes' SHA-256 in `row_extension.rendered_config`, while the
canonical Stage-1 train/eval and diagnostics YAML remain provenance inputs.
