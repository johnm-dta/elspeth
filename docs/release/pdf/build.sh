#!/usr/bin/env bash
# build.sh — render the ELSPETH release PDF set from a clean checkout.
#
# Output:
#   out/elspeth-executive-summary.pdf      (PDF/UA-1)
#   out/elspeth-architecture.pdf           (PDF/UA-1)
#   out/elspeth-composer.pdf               (PDF/UA-1)
#   out/elspeth-guarantees.pdf             (PDF/UA-1)
#   out/elspeth-data-trust.pdf             (PDF/UA-1)
#
# Diagrams are drawn natively in Typst (cetz 0.4.2) — see theme.typ
# `diagram-sda-flow()` and `diagram-trust-tiers()`. The earlier
# Mermaid pipeline (mmdc / puppeteer / headless chromium) is no
# longer in the toolchain; vector output gives crisper rendering at
# any zoom, deterministic reproduction, and consistent fonts.
#
# Required tools (script fails loudly if missing):
#   typst   >= 0.14.0  (tagged-PDF by default; PDF/UA-1 export)
#
# Optional tools (script warns if missing, but build still proceeds):
#   pandoc  >= 3.2     (not used in the default build; reserved for a
#                       future markdown reflow path)
#   verapdf            (only consulted under --verify; authoritative
#                       PDF/UA-1 strict validator)
#
# Flags:
#   --verify   After Typst compilation, run veraPDF against each output
#              and fail the build on any PDF/UA-1 strict violation.
#              Requires either `verapdf` on PATH or a Docker runtime
#              (the verapdf/cli:v1.30.1 image is pulled if needed).
#
# Fonts:
#   Public Sans is bundled under fonts/public-sans/fonts/ttf. The build
#   passes --font-path so the system's font search does not get in the
#   way. If Public Sans is removed, Typst falls back to Lato per the
#   fallback chain in tokens.typ — `typst fonts` is consulted at
#   preflight so a partial-install silent-fallback is caught.

set -Eeuo pipefail

cd "$(dirname "$(readlink -f "$0")")"

# ---------------------------------------------------------------------------
# 0. Argument parsing.
# ---------------------------------------------------------------------------

verify=0
for arg in "$@"; do
  case "$arg" in
    --verify) verify=1 ;;
    --help|-h)
      sed -n '2,30p' "$0"
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $arg" >&2
      echo "Try: $0 --help" >&2
      exit 64
      ;;
  esac
done

# ---------------------------------------------------------------------------
# 1. Toolchain preflight.
# ---------------------------------------------------------------------------

# Minimum-required floors. Build FAILS below these. Locked versions
# (see versions.lock) are advisory: the documents typically build
# cleanly on any version >= floor, but visual reproduction is
# byte-stable only at the locked version.
typst_min="0.14.0"
pandoc_min="3.2"

# semver-ish compare: returns 0 iff $1 >= $2. Uses `sort -V` because
# the alternatives (full semver parser in pure bash) are gnarly and
# we only need monotonic ordering of "M.m.p"-shaped strings.
version_ge() {
  [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" == "$2" ]]
}

if ! command -v typst >/dev/null 2>&1; then
  echo "ERROR: required tool missing from PATH: typst" >&2
  echo "Install: https://typst.app (or: cargo install --git https://github.com/typst/typst typst-cli)" >&2
  exit 2
fi

# Optional tools: warn but do not fail.
if ! command -v pandoc >/dev/null 2>&1; then
  echo "INFO: pandoc not installed — default build does not use it (reserved for a future reflow path)." >&2
  pandoc_version="(absent)"
else
  pandoc_version="$(pandoc --version | awk 'NR==1 {print $2}')"
fi

typst_version="$(typst --version | awk '{print $2}')"
echo "[preflight] typst:  $typst_version"
echo "[preflight] pandoc: $pandoc_version"

# Minimum-version floor checks — these FAIL the build.
if ! version_ge "$typst_version" "$typst_min"; then
  echo "ERROR: typst $typst_version < required minimum $typst_min." >&2
  echo "       PDF/UA-1 export and tagged-PDF defaults require Typst >= $typst_min." >&2
  exit 2
fi
if [[ "$pandoc_version" != "(absent)" ]] && ! version_ge "$pandoc_version" "$pandoc_min"; then
  echo "WARNING: pandoc $pandoc_version < $pandoc_min (unused by default build; warning only)." >&2
fi

# Locked-version drift — WARNING only. Bundled outputs in out/ and
# diagrams/ are byte-stable at the locked versions; documents build
# cleanly on any version >= the minimum floor above.
lockfile="versions.lock"
if [[ -f "$lockfile" ]]; then
  while IFS='=' read -r key locked; do
    [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]] && continue
    key="${key// /}"
    locked="${locked// /}"
    case "$key" in
      typst)  actual="$typst_version"  ;;
      pandoc) actual="$pandoc_version" ;;
      universe.*) continue ;;  # Universe pins live in *.typ imports
      *) continue ;;
    esac
    if [[ "$actual" == "(absent)" ]]; then
      continue  # optional tool, not installed — already reported above
    fi
    if [[ "$actual" != "$locked" ]]; then
      echo "WARNING: $key version drift — locked=$locked, actual=$actual." >&2
      echo "         Bundled outputs in out/ were built at the locked version." >&2
    fi
  done < "$lockfile"
else
  echo "WARNING: versions.lock missing — cannot check for tool-version drift." >&2
fi

mkdir -p out

# ---------------------------------------------------------------------------
# 2. Typst compile (PDF/UA-1).
# ---------------------------------------------------------------------------
#
# --pdf-standard ua-1 makes Typst fail the build on detectable
# accessibility violations (missing alt text on a figure, skipped
# heading level, etc.). Tagged-PDF is on by default in Typst >= 0.14.

font_path="fonts/public-sans/fonts/ttf"
if [[ ! -d "$font_path" ]]; then
  echo "WARNING: $font_path missing; Typst will substitute Lato (or the next font in the body fallback chain)." >&2
  font_args=()
else
  font_args=(--font-path "$font_path")
  # Verify Typst actually discovers Public Sans inside the bundled
  # path. A partial install (missing weight, broken symlink) would
  # otherwise fall back silently per-weight and shift the visual
  # character of the documents without any build-time signal.
  if ! typst fonts --font-path "$font_path" 2>/dev/null | grep -q "^Public Sans$"; then
    echo "WARNING: Public Sans not discoverable under $font_path; Typst will fall back silently per weight." >&2
  fi
fi

for stem in executive-summary architecture composer guarantees data-trust; do
  src="${stem}.typ"
  case "$stem" in
    executive-summary) out_name="elspeth-executive-summary";;
    architecture)      out_name="elspeth-architecture";;
    composer)          out_name="elspeth-composer";;
    guarantees)        out_name="elspeth-guarantees";;
    data-trust)        out_name="elspeth-data-trust";;
  esac
  out_path="out/${out_name}.pdf"
  echo "[typst]   $src -> $out_path"
  typst compile --pdf-standard ua-1 "${font_args[@]}" "$src" "$out_path"
done

# ---------------------------------------------------------------------------
# 3. Optional external validation (veraPDF, opt-in via --verify).
# ---------------------------------------------------------------------------
#
# Typst's --pdf-standard ua-1 is best-effort at export time. veraPDF is
# the authoritative PDF/UA-1 strict validator and is recommended before
# any external sign-off. This stage is opt-in because veraPDF adds
# noticeable wall-time (and pulls a Docker image on first run); the
# default build emits without external validation.

if [[ "$verify" -eq 1 ]]; then
  echo
  echo "[verify] running veraPDF (PDF/UA-1 strict) against built outputs..."

  # Image override knob — useful for pinning to a different tag or
  # pointing at an internal mirror. The default `verapdf/cli:v1.30.1`
  # is the smaller CLI-only image (~50 MB) rather than the full REST
  # service image (~700 MB).
  verapdf_image="${VERAPDF_IMAGE:-verapdf/cli:v1.30.1}"

  if command -v verapdf >/dev/null 2>&1; then
    verapdf_cmd=(verapdf --flavour ua1 --format text)
    verapdf_runner="local"
  elif command -v docker >/dev/null 2>&1; then
    verapdf_cmd=(docker run --rm -v "$PWD/out:/work" "$verapdf_image"
      --flavour ua1 --format text)
    verapdf_runner="docker ($verapdf_image)"
  else
    echo "ERROR: --verify requested but neither verapdf nor docker is on PATH." >&2
    echo "       Install verapdf locally (https://verapdf.org/) or a Docker runtime." >&2
    exit 2
  fi

  echo "[verify] runner: $verapdf_runner"
  fail=0
  for f in out/*.pdf; do
    name="$(basename "$f")"
    echo "[verify] $name"
    # The Docker variant operates inside /work; the local variant needs
    # the host path. Build the per-file argument accordingly.
    if [[ "$verapdf_runner" == docker* ]]; then
      target="/work/$name"
    else
      target="$f"
    fi
    # veraPDF exits non-zero on hard failures and emits an
    # `isCompliant="false"` marker in its text output when the document
    # fails the flavour. Treat either as a verification failure.
    if ! out_lines="$("${verapdf_cmd[@]}" "$target" 2>&1)"; then
      echo "$out_lines" >&2
      echo "[verify] FAIL: $name (veraPDF exited non-zero)" >&2
      fail=1
    elif echo "$out_lines" | grep -qE 'isCompliant="false"|FAIL'; then
      echo "$out_lines" >&2
      echo "[verify] FAIL: $name (PDF/UA-1 strict non-compliance reported)" >&2
      fail=1
    else
      echo "[verify] PASS: $name"
    fi
  done

  if [[ "$fail" -ne 0 ]]; then
    echo
    echo "ERROR: veraPDF reported PDF/UA-1 strict violations. See output above." >&2
    exit 4
  fi
fi

# ---------------------------------------------------------------------------
# 4. Summary.
# ---------------------------------------------------------------------------

echo
echo "Built PDFs:"
for f in out/*.pdf; do
  size=$(stat -c '%s' "$f" 2>/dev/null || stat -f '%z' "$f" 2>/dev/null || echo "?")
  echo "  $f  ($size bytes)"
done
echo
echo "All generated PDFs are PDF/UA-1 conformant (Typst export-time check)."
if [[ "$verify" -eq 1 ]]; then
  echo "External validator (veraPDF) ran clean (--verify)."
else
  echo "External validator (veraPDF or PAC) recommended before sign-off — re-run with --verify."
fi
