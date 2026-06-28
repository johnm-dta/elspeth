# Makefile — incremental builds for the ELSPETH release PDF set.
#
# build.sh is the canonical reproducible build (toolchain preflight,
# version-lock checks, full re-render). Use this Makefile for tight
# edit-rebuild loops where you want only the changed PDF to recompile.
#
# Targets:
#   make             — build all five PDFs (incremental).
#   make all         — same as `make`.
#   make exec        — build the executive-summary PDF only.
#   make architecture — build the architecture reference PDF only.
#   make composer    — build the composer reference PDF only.
#   make guarantees  — build the assurance guarantees PDF only.
#   make data-trust  — build the data trust primer PDF only.
#   make verify      — full build via build.sh --verify (authoritative).
#   make clean       — delete out/.
#   make distclean   — alias for clean (kept for muscle-memory).
#
# This Makefile deliberately skips the toolchain preflight and the
# versions.lock advisory check that build.sh performs — use build.sh
# when entering or re-entering the project, or before any sign-off
# pass. The Makefile is a convenience for iterative authoring only.

TYPST       ?= typst
FONT_PATH   := fonts/public-sans/fonts/ttf
TYPST_FLAGS := --pdf-standard ua-1
ifneq ($(wildcard $(FONT_PATH)),)
TYPST_FLAGS += --font-path $(FONT_PATH)
endif

OUT          := out

# Every .typ document depends on the shared modules. Diagrams are now
# drawn natively in Typst (cetz) — no external diagram pipeline.
COMMON_DEPS  := tokens.typ theme.typ data.typ

PDFS := \
  $(OUT)/elspeth-executive-summary.pdf \
  $(OUT)/elspeth-architecture.pdf \
  $(OUT)/elspeth-composer.pdf \
  $(OUT)/elspeth-guarantees.pdf \
  $(OUT)/elspeth-data-trust.pdf

.PHONY: all exec architecture composer guarantees data-trust verify clean distclean

all: $(PDFS)

exec: $(OUT)/elspeth-executive-summary.pdf
architecture: $(OUT)/elspeth-architecture.pdf
composer: $(OUT)/elspeth-composer.pdf
guarantees: $(OUT)/elspeth-guarantees.pdf
data-trust: $(OUT)/elspeth-data-trust.pdf

# Use build.sh for verification — it owns the veraPDF wiring.
verify:
	./build.sh --verify

$(OUT):
	mkdir -p $(OUT)

# Per-document rules. Keep them explicit (rather than a pattern rule)
# so the stem-to-output-name mapping stays auditable in one place.
$(OUT)/elspeth-executive-summary.pdf: executive-summary.typ $(COMMON_DEPS) | $(OUT)
	$(TYPST) compile $(TYPST_FLAGS) $< $@

$(OUT)/elspeth-architecture.pdf: architecture.typ $(COMMON_DEPS) | $(OUT)
	$(TYPST) compile $(TYPST_FLAGS) $< $@

$(OUT)/elspeth-composer.pdf: composer.typ $(COMMON_DEPS) | $(OUT)
	$(TYPST) compile $(TYPST_FLAGS) $< $@

$(OUT)/elspeth-guarantees.pdf: guarantees.typ $(COMMON_DEPS) | $(OUT)
	$(TYPST) compile $(TYPST_FLAGS) $< $@

$(OUT)/elspeth-data-trust.pdf: data-trust.typ $(COMMON_DEPS) | $(OUT)
	$(TYPST) compile $(TYPST_FLAGS) $< $@

clean:
	rm -rf $(OUT)

distclean: clean
