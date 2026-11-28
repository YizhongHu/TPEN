# Shared path helpers for the Polaris preparation jobs.

canonicalize_path() {
  realpath -m -- "$1"
}

assert_results_path() {
  local allowed_canonical="$1" path="$2" label="$3" display_prefix="$4"
  local canonical
  canonical="$(canonicalize_path "$path")" || die "could not canonicalize $label: $path"
  [[ "$canonical" != "$allowed_canonical" ]] ||
    die "$label must name a child path"
  [[ "$canonical" == "$allowed_canonical"/* ]] ||
    die "$label is outside $display_prefix: $path"
}
