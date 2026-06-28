> **Part of the [Tutorial Staged Recut plan](./00-overview.md).** Read the [overview](./00-overview.md) first — it holds the Global Constraints (§9.2 gate commands) and the "use VERBATIM" Shared Interfaces every task depends on. Phases execute **P0 → P7 in order**.

## Phase P4 — web_scrape recipe (D11) — re-polarized shield

> **Phase dependency note.** P4 introduces the `web-scrape-llm-rate-jsonl`
> `RecipeSpec` and the `_web_scrape_predicate`. Both are consumed by the
> **P2 completion-seam redirect** (`handle_step_2_5_recipe_apply` must leave
> `terminal=None` / `step=STEP_4_WIRE`) and by the **P6 advisor terminal gate**.
> P4 deliberately does NOT touch the completion seam, `STEP_4_WIRE`, or the wire
> stage — those land in P2/P3/P6. P4 ships the recipe so that, once P2 redirects
> the recipe-apply seam, the canonical pipeline composes deterministically.
> P4 also relies on the **already-shipped** raw-HTML cleanup contract
> (`composition_review_contract_error` → `raw_html_cleanup_review_contract_error`,
> `interpretation_state.py:186/226`) and the **already-shipped** prompt-shield
> advisory (`prompt_shield_recommendation_warning_pairs`,
> `interpretation_state.py:240`) — P4 adds no new contract, it satisfies the
> existing blocking one and preserves the existing advisory.

> **Resolved fact (pinned by Task P4.1).** The canonical URL-row source is an
> `inline_blob` in the wire payload, but it **materialises** to a real registered
> source plugin via `_MIME_TO_SOURCE` (`tools/sources.py:148`): a JSON URL-list →
> `json`, a CSV URL-list → `csv`, plain text → `text`. So `SourceResolved.plugin`
> is **`json`** (or `csv`) for the canonical source — never the literal string
> `"inline_blob"` and never `"web_scrape"` (web_scrape is a TRANSFORM,
> `plugins/transforms/web_scrape.py:146` `class WebScrapeConfig(TransformDataConfig)`).
> The predicate keys on that resolved plugin + a `url`-column signal + a single
> jsonl output, gated on `blob_ref` (mirroring `_classify_predicate`,
> `recipe_match.py:205`). The slot resolver must carry the resolved
> `source_plugin` into the recipe builder so JSON and CSV URL-row sources both
> build the same recipe without collapsing CSV into a JSON head source.
>
> **`blob_ref` IS present at match time (the load-bearing fact for §4.1).** When
> `set_pipeline` composes the canonical seed from `source.inline_blob`, it persists the
> inline blob as a real blob row AND **unconditionally writes
> `source.options["blob_ref"]` = the new blob UUID** (`composer/tools/sessions.py:425`,
> <!-- RECONCILE: Read tool shows :425; grep -n shows :427 — verify at implementation time -->
> inside the `if inline_blob is not None` branch; the same `SourceSpec.options` is what
> `match_recipe` reads). So the `blob_ref` gate is satisfied for the canonical source —
> the predicate matches, `match_recipe` returns the recipe, and §4.1's zero-LLM compose
> fires. The end-to-end proof is **Task P4.2 Step 4b**
> (`test_canonical_seed_materialised_source_matches_web_scrape_recipe`), which drives the
> exact materialised shape through `match_recipe`. (NB: this `source.options["blob_ref"]`
> — a plain UUID string — is a DIFFERENT object from the `{"mode": "inline_content",
> "blob_ref": ...}` widened marker the fork blob-rewrite recurses for
> (`sessions/routes/sessions.py:128`); see the spec's two-objects `blob_ref` note in
> §5/B4. The older fork-strip premise that the authoring alias lacks a `blob_ref`
> is rejected here — at recipe-match time the materialized source has `blob_ref`.)

> **Resolved fact (pinned by Task P4.2).** There is **no** registered `llm_rate`
> plugin (`grep -rn 'name = "llm_rate"'` → empty; only `llm` at
> `plugins/transforms/llm/transform.py:1124` and `field_mapper` at
> `plugins/transforms/field_mapper.py:117`). `llm_rate` is a cosmetic display
> label in `tutorial.spec.ts`. The recipe's rating node uses the **real `llm`
> plugin**, which is also what the shield advisory keys on
> (`prompt_shield_recommendation_warning_pairs` checks `node.plugin == "llm"`,
> `interpretation_state.py:246`).

---

### Task P4.1: Pin the resolved canonical-source plugin + add `_web_scrape_predicate`

**Files:**
- Modify `src/elspeth/web/composer/guided/recipe_match.py` (topology helpers after
  `_has_two_json_outputs` at :194; predicate after `_split_threshold_slot_resolver`
  at :346; `_RECIPE_PREDICATES` tuple at :353)
- Modify `tests/unit/web/composer/guided/test_recipe_match.py` (add a
  `TestWebScrapePredicate` class + a resolved-plugin-name pin test)

**Interfaces:**
- Consumes: `SourceResolved` (`guided/resolved.py:19`; fields `plugin`, `options`,
  `observed_columns`, `sample_rows`), `SinkResolved` (`resolved.py:89`),
  `SinkOutputResolved` (`resolved.py:54`; fields `plugin`, `options`,
  `required_fields`, `schema_mode`), `_MIME_TO_SOURCE`
  (`composer/tools/sources.py:148`).
- Produces (NEW, consumed by P4.2 + future phases):
  - `_web_scrape_predicate(source: SourceResolved, sink: SinkResolved) -> bool`
  - `_URL_ROW_SOURCE_PLUGINS: frozenset[str]` (module constant = `frozenset({"json", "csv"})`)
  - `_URL_COLUMN_NAMES: frozenset[str]` (module constant = `frozenset({"url"})`)
  - registry entry `(_web_scrape_predicate, "web-scrape-llm-rate-jsonl", _web_scrape_slot_resolver)`
    appended to `_RECIPE_PREDICATES` (slot resolver added in P4.2).

- [ ] **Step 1: Write the resolved-plugin-name pin test (run-to-fail).**
  Append to `tests/unit/web/composer/guided/test_recipe_match.py`:
  ```python
  class TestCanonicalSourcePluginIsResolved:
      """Pin the fact the predicate relies on: an inline_blob URL list
      materialises to a real registered source plugin, never the literal
      string 'inline_blob' and never 'web_scrape'.

      _MIME_TO_SOURCE is the single mapping that resolves a materialised
      inline_blob's MIME type to its concrete source plugin; the predicate
      must key on those resolved names, not on 'inline_blob'.
      """

      def test_mime_to_source_resolves_url_row_plugins(self) -> None:
          from elspeth.web.composer.tools.sources import _MIME_TO_SOURCE

          resolved_plugins = {plugin for plugin, _extra in _MIME_TO_SOURCE.values()}
          # The canonical URL list is JSON rows of {"url": ...}; CSV is the
          # other URL-row carrier. Both are real registered source plugins.
          assert "json" in resolved_plugins
          assert "csv" in resolved_plugins
          # web_scrape is a TRANSFORM, never a materialised source plugin.
          assert "web_scrape" not in resolved_plugins
          assert "inline_blob" not in resolved_plugins

      def test_url_row_source_plugins_constant_matches_resolved_names(self) -> None:
          from elspeth.web.composer.guided.recipe_match import _URL_ROW_SOURCE_PLUGINS

          assert _URL_ROW_SOURCE_PLUGINS == frozenset({"json", "csv"})
          assert "web_scrape" not in _URL_ROW_SOURCE_PLUGINS
  ```
  Run-to-fail:
  ```
  uv run python -m pytest tests/unit/web/composer/guided/test_recipe_match.py::TestCanonicalSourcePluginIsResolved -q
  ```
  Expected failure: `ImportError: cannot import name '_URL_ROW_SOURCE_PLUGINS' from 'elspeth.web.composer.guided.recipe_match'`.

- [ ] **Step 2: Add the module constants + `_web_scrape_predicate` (minimal impl).**
  In `recipe_match.py`, after `_has_two_json_outputs` (ends at :195), insert:
  ```python
  # ---------------------------------------------------------------------------
  # web-scrape-llm-rate-jsonl predicate
  #
  # web_scrape is a TRANSFORM (plugins/transforms/web_scrape.py); the predicate
  # keys on the URL-ROW SOURCE that feeds it, never on web_scrape itself. An
  # inline_blob URL list materialises to a real registered source plugin via
  # _MIME_TO_SOURCE (tools/sources.py): JSON rows -> "json", CSV -> "csv". The
  # predicate matches those resolved names + a "url" column signal + a single
  # jsonl output, gated on blob_ref (same blob-presence discipline as
  # _classify_predicate).
  # ---------------------------------------------------------------------------

  _URL_ROW_SOURCE_PLUGINS: frozenset[str] = frozenset({"json", "csv"})
  _URL_COLUMN_NAMES: frozenset[str] = frozenset({"url"})


  def _has_single_jsonl_output(sink: SinkResolved) -> bool:
      """Return True for a single ``json`` output configured as JSONL.

      The canonical web-scrape pipeline writes one JSONL file. ``json`` is the
      registered sink plugin; ``format: jsonl`` is the JSONL discriminator (an
      absent format is the json plugin's default object-array, not JSONL).
      """
      if not (len(sink.outputs) == 1 and sink.outputs[0].plugin == "json"):
          return False
      return sink.outputs[0].options.get("format") == "jsonl"


  def _source_has_url_column(source: SourceResolved) -> bool:
      """Return True iff the source surfaces a ``url`` column.

      The signal is the observed URL column that web_scrape's ``url_field``
      will read. Observed columns come from inspecting the materialised blob;
      a URL list always surfaces ``url``.
      """
      return any(col in _URL_COLUMN_NAMES for col in source.observed_columns)


  def _web_scrape_predicate(source: SourceResolved, sink: SinkResolved) -> bool:
      """Return True for a blob-backed URL-row source → single JSONL output.

      Matches the canonical tutorial shape: an inline_blob URL list that
      materialised to a ``json``/``csv`` source (NOT ``web_scrape`` — that is a
      downstream transform the recipe inserts) feeding a single JSONL sink, with
      an observed ``url`` column.

      Requires ``blob_ref`` in ``source.options`` for the same reason as
      ``_classify_predicate``: the slot resolver cannot derive ``source_blob_id``
      without it, and "no recipe match" (fall through to the live chain solver)
      is the correct outcome for a non-blob-backed URL source.
      """
      if source.plugin not in _URL_ROW_SOURCE_PLUGINS:
          return False
      if "blob_ref" not in source.options:
          return False
      if not _has_single_jsonl_output(sink):
          return False
      return _source_has_url_column(source)
  ```
  Run-to-pass:
  ```
  uv run python -m pytest tests/unit/web/composer/guided/test_recipe_match.py::TestCanonicalSourcePluginIsResolved -q
  ```
  Expected: `2 passed`.

- [ ] **Step 3: Write the predicate behaviour tests (run-to-fail).**
  Append to `test_recipe_match.py`:
  ```python
  def _make_url_json_source(
      blob_ref: str = "a1b2c3d4-0000-0000-0000-000000000099",
      *,
      with_blob: bool = True,
  ) -> SourceResolved:
      """A materialised inline_blob URL list: json plugin, url column, blob_ref."""
      options: dict[str, object] = {}
      if with_blob:
          options["blob_ref"] = blob_ref
      return SourceResolved(
          plugin="json",
          options=options,
          observed_columns=("url",),
          sample_rows=({"url": "https://dta.gov.au"},),
      )


  def _make_url_csv_source(
      blob_ref: str = "a1b2c3d4-0000-0000-0000-000000000099",
  ) -> SourceResolved:
      """A materialised inline_blob URL list: csv plugin, url column, blob_ref."""
      return SourceResolved(
          plugin="csv",
          options={"blob_ref": blob_ref},
          observed_columns=("url",),
          sample_rows=({"url": "https://dta.gov.au"},),
      )


  def _make_single_jsonl_sink(path: str = "outputs/ratings.jsonl") -> SinkResolved:
      return SinkResolved(
          outputs=(
              SinkOutputResolved(
                  plugin="json",
                  options={"path": path, "format": "jsonl"},
                  required_fields=(),
                  schema_mode="observed",
              ),
          )
      )


  class TestWebScrapePredicate:
      def test_matches_blob_backed_json_url_source(self) -> None:
          from elspeth.web.composer.guided.recipe_match import _web_scrape_predicate

          assert (
              _web_scrape_predicate(_make_url_json_source(), _make_single_jsonl_sink())
              is True
          )

      def test_matches_blob_backed_csv_url_source(self) -> None:
          from elspeth.web.composer.guided.recipe_match import _web_scrape_predicate

          assert (
              _web_scrape_predicate(_make_url_csv_source(), _make_single_jsonl_sink())
              is True
          )

      def test_does_not_reference_web_scrape_as_source(self) -> None:
          """A source whose plugin is literally 'web_scrape' must NOT match.

          web_scrape is a transform, not a source.
          """
          from elspeth.web.composer.guided.recipe_match import _web_scrape_predicate

          bad = SourceResolved(
              plugin="web_scrape",
              options={"blob_ref": "a1b2c3d4-0000-0000-0000-000000000099"},
              observed_columns=("url",),
              sample_rows=({"url": "https://dta.gov.au"},),
          )
          assert _web_scrape_predicate(bad, _make_single_jsonl_sink()) is False

      def test_no_match_without_blob_ref(self) -> None:
          from elspeth.web.composer.guided.recipe_match import _web_scrape_predicate

          assert (
              _web_scrape_predicate(
                  _make_url_json_source(with_blob=False),
                  _make_single_jsonl_sink(),
              )
              is False
          )

      def test_no_match_without_url_column(self) -> None:
          from elspeth.web.composer.guided.recipe_match import _web_scrape_predicate

          no_url = SourceResolved(
              plugin="json",
              options={"blob_ref": "a1b2c3d4-0000-0000-0000-000000000099"},
              observed_columns=("name",),
              sample_rows=({"name": "x"},),
          )
          assert (
              _web_scrape_predicate(no_url, _make_single_jsonl_sink()) is False
          )

      def test_no_match_for_non_jsonl_output(self) -> None:
          from elspeth.web.composer.guided.recipe_match import _web_scrape_predicate

          object_array = SinkResolved(
              outputs=(
                  SinkOutputResolved(
                      plugin="json",
                      options={"path": "outputs/ratings.json"},  # no format: jsonl
                      required_fields=(),
                      schema_mode="observed",
                  ),
              )
          )
          assert (
              _web_scrape_predicate(_make_url_json_source(), object_array) is False
          )
  ```
  Run-to-fail:
  ```
  uv run python -m pytest tests/unit/web/composer/guided/test_recipe_match.py::TestWebScrapePredicate -q
  ```
    Expected: `6 passed` (the predicate already exists from Step 2 — this step pins
    its behaviour; if any assertion is red, fix the predicate before proceeding).

- [ ] **Step 4: Register the predicate in `_RECIPE_PREDICATES` (run-to-fail).**
  Add a registry-shape test first. Append to `test_recipe_match.py`:
  ```python
  def test_web_scrape_predicate_registered_last() -> None:
      """The web-scrape predicate is registered, after the CSV recipes
      (most-specific-first ordering: the URL-row json source never collides
      with the CSV classify/split predicates, but order is asserted to keep
      registry edits intentional)."""
      from elspeth.web.composer.guided.recipe_match import _RECIPE_PREDICATES

      names = [name for _pred, name, _resolver in _RECIPE_PREDICATES]
      assert "web-scrape-llm-rate-jsonl" in names
      assert names[-1] == "web-scrape-llm-rate-jsonl"
  ```
  Run-to-fail:
  ```
  uv run python -m pytest tests/unit/web/composer/guided/test_recipe_match.py::test_web_scrape_predicate_registered_last -q
  ```
  Expected failure: `assert 'web-scrape-llm-rate-jsonl' in [...]` → KeyError/AssertionError
  (also note: `_web_scrape_slot_resolver` does not exist yet — that lands in P4.2;
  for this step register with a **temporary** resolver stub so the tuple is shape-valid).

- [ ] **Step 5: Append the registry entry with a slot-resolver stub (minimal impl).**
  In `recipe_match.py`, immediately after `_web_scrape_predicate`, add the stub
  resolver (the real one lands in P4.2):
  ```python
  def _web_scrape_slot_resolver(source: SourceResolved, sink: SinkResolved) -> Mapping[str, Any]:
      """Partial slot map for the web-scrape-llm-rate-jsonl recipe.

        Provides ``source_blob_id`` (the composer-canonical blob UUID),
        ``source_plugin`` (the real materialised source plugin: json or csv), and
        ``output_path`` (operator-set verbatim, else a rubber-stampable default).
        User-fillable: ``model``, ``api_key_secret``, ``provider``,
        ``rating_template``, ``abuse_contact``, and ``scraping_reason``.
      """
      if "blob_ref" not in source.options:
          raise InvariantError(
              "web-scrape recipe slot resolver requires source.options['blob_ref']; "
              f"source options present: {sorted(source.options.keys())}"
          )
      blob_ref = source.options["blob_ref"]
      sink_options = sink.outputs[0].options
      output_path = sink_options["path"] if "path" in sink_options else "outputs/ratings.jsonl"
      return {
          "source_blob_id": blob_ref,
          "source_plugin": source.plugin,
          "output_path": output_path,
      }
  ```
  Then append to `_RECIPE_PREDICATES` (after the split-threshold entry at :355,
  before the closing comment block):
  ```python
      (_web_scrape_predicate, "web-scrape-llm-rate-jsonl", _web_scrape_slot_resolver),
  ```
  Run-to-pass:
  ```
  uv run python -m pytest tests/unit/web/composer/guided/test_recipe_match.py -q
  ```
  Expected: all pass (existing classify/split tests + the new ones).
  Note: `match_recipe` (recipe_match.py:371) will now raise `InvariantError`
  ("Recipe '...' is in _RECIPE_PREDICATES but not registered in recipes.py") if
  invoked end-to-end — that is expected and resolved in P4.2 when the `RecipeSpec`
  is registered. The unit tests above call `_web_scrape_predicate` directly, so
  they pass now.

- [ ] **Step 6: Lint + commit.**
  ```
  uv run python -m ruff check src/elspeth/web/composer/guided/recipe_match.py tests/unit/web/composer/guided/test_recipe_match.py
  uv run python -m ruff format src/elspeth/web/composer/guided/recipe_match.py tests/unit/web/composer/guided/test_recipe_match.py
  git add src/elspeth/web/composer/guided/recipe_match.py tests/unit/web/composer/guided/test_recipe_match.py
  git commit -m "feat(composer/recipe-match): add web-scrape URL-row source predicate (D11/P4.1)

Predicate matches the materialised inline_blob URL source (json/csv via
_MIME_TO_SOURCE), never web_scrape (a transform) and never the literal
'inline_blob'. Keyed on blob_ref + a url column + a single JSONL output,
mirroring _classify_predicate's blob-presence discipline.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```
  Expected: ruff clean, commit succeeds.

---

### Task P4.2: Add the `web-scrape-llm-rate-jsonl` `RecipeSpec` + `_build_web_scrape_recipe`

**Files:**
- Modify `src/elspeth/web/composer/recipes.py` (slot table after `_RECIPE3_SLOTS`
  at :442; builder after `_build_fork_coalesce_truncate_recipe` at :561; registry
  `_RECIPES` at :569)
- Modify `tests/unit/web/composer/test_recipes.py` (registry assertion at :38; new
  `TestWebScrapeRecipeBuild` class)

**Interfaces:**
- Consumes: `RecipeSpec` (`recipes.py:26`), `SlotSpec` (`recipes.py:20`),
  `validate_slots` / `apply_recipe` (`recipes.py:119/643`),
  `RAW_HTML_CLEANUP_USER_TERM = "drop_raw_html_fields"` +
  `RAW_HTML_CLEANUP_REVIEW_DRAFT = "Drop the scraped raw HTML and fingerprint
  fields before saving the JSON output."` (`web/interpretation_state.py:31/32`),
  `INTERPRETATION_REQUIREMENTS_KEY = "interpretation_requirements"`
  (`interpretation_state.py:25`).
- Produces (NEW, consumed by P2 completion-seam + P4.3 contract test + P4.4 caching):
  - `_RECIPE_WEB_SCRAPE_SLOTS: Final[dict[str, SlotSpec]]`
  - `_build_web_scrape_recipe(slots: Mapping[str, Any]) -> dict[str, Any]`
  - registry entry `"web-scrape-llm-rate-jsonl"` in `_RECIPES`.
- Canonical chain produced (set_pipeline-compatible args), head source node named
  `"url_rows"`: `source(slots["source_plugin"], blob_id=…, on_success="rows")` →
  `web_scrape(input="rows", on_success="scraped")` →
  `llm(input="scraped", on_success="rated")` →
  `field_mapper(input="rated", on_success="clean", select_only=true, mapping
  drops content/fingerprint, interpretation_requirements stages the
  pipeline_decision)` → `json/jsonl` sink.

- [ ] **Step 1: Write the registry + structural build test (run-to-fail).**
  Update the registry assertion at `test_recipes.py:38`:
  ```python
          assert names == {
              "classify-rows-llm-jsonl",
              "split-by-numeric-threshold",
              "fork-coalesce-truncate-jsonl",
              "web-scrape-llm-rate-jsonl",
          }
  ```
  Append a new class:
  ```python
  class TestWebScrapeRecipeBuild:
      """The web-scrape recipe deterministically emits
      source → web_scrape → llm → field_mapper(cleanup) → jsonl."""

      _SLOTS = {
          "source_blob_id": str(uuid4()),
          "source_plugin": "json",
          "model": "anthropic/claude-sonnet-4.6",
          "api_key_secret": "OPENROUTER_API_KEY",
          "abuse_contact": "web-scrape-contact@dta.gov.au",
          "scraping_reason": "Tutorial exercise: fetch public pages for rating",
          "output_path": "outputs/ratings.jsonl",
      }

      def _build(self) -> dict:
          return apply_recipe("web-scrape-llm-rate-jsonl", self._SLOTS)

      def test_head_source_node_uses_resolved_source_plugin(self) -> None:
          args = self._build()
          assert args["source"]["plugin"] == "json"
          assert args["source"]["blob_id"] == self._SLOTS["source_blob_id"]
          assert args["source"]["on_success"] == "rows"

      def test_csv_url_source_stays_csv(self) -> None:
          slots = {**self._SLOTS, "source_plugin": "csv"}
          args = apply_recipe("web-scrape-llm-rate-jsonl", slots)
          assert args["source"]["plugin"] == "csv"
          assert args["source"]["blob_id"] == slots["source_blob_id"]

      def test_canonical_chain_order(self) -> None:
          args = self._build()
          plugins = [n["plugin"] for n in args["nodes"]]
          assert plugins == ["web_scrape", "llm", "field_mapper"]

      def test_chain_is_wired_by_connection_labels(self) -> None:
          args = self._build()
          by_plugin = {n["plugin"]: n for n in args["nodes"]}
          assert by_plugin["web_scrape"]["input"] == "rows"
          assert by_plugin["web_scrape"]["on_success"] == "scraped"
          assert by_plugin["llm"]["input"] == "scraped"
          assert by_plugin["llm"]["on_success"] == "rated"
          assert by_plugin["field_mapper"]["input"] == "rated"
          assert by_plugin["field_mapper"]["on_success"] == "clean"
          assert args["outputs"][0]["sink_name"] == "clean"
          assert args["outputs"][0]["plugin"] == "json"
          assert args["outputs"][0]["options"]["format"] == "jsonl"

      def test_field_mapper_select_only_excludes_raw_content_and_fingerprint(self) -> None:
          """Data-minimization: the cleanup sink field set EXCLUDES the raw
          web_scrape content/fingerprint fields (pin the actual output set)."""
          args = self._build()
          fm = next(n for n in args["nodes"] if n["plugin"] == "field_mapper")
          assert fm["options"]["select_only"] is True
          mapping = fm["options"]["mapping"]
          preserved = set(mapping) | set(mapping.values())
          assert "content" not in preserved
          assert "content_fingerprint" not in preserved
          # Positive pin: the rating + url ARE preserved (the user-facing output).
          assert "rating" in preserved
          assert "url" in preserved

      def test_field_mapper_stages_pipeline_decision_cleanup_requirement(self) -> None:
          """The raw-HTML cleanup pipeline_decision is staged on the field_mapper
          node so the blocking cleanup contract passes (tools/sessions.py:670 →
          raw_html_cleanup_review_contract_error)."""
          from elspeth.web.interpretation_state import (
              INTERPRETATION_REQUIREMENTS_KEY,
              RAW_HTML_CLEANUP_REVIEW_DRAFT,
              RAW_HTML_CLEANUP_USER_TERM,
          )

          args = self._build()
          fm = next(n for n in args["nodes"] if n["plugin"] == "field_mapper")
          reqs = fm["options"][INTERPRETATION_REQUIREMENTS_KEY]
          decision = next(r for r in reqs if r["kind"] == "pipeline_decision")
          assert decision["user_term"] == RAW_HTML_CLEANUP_USER_TERM
          assert decision["draft"] == RAW_HTML_CLEANUP_REVIEW_DRAFT
          assert decision["status"] == "pending"

      def test_web_scrape_node_declares_content_and_fingerprint_fields(self) -> None:
          """web_scrape must name content_field/fingerprint_field so
          _web_scrape_raw_fields can compute the raw set the cleanup drops."""
          args = self._build()
          ws = next(n for n in args["nodes"] if n["plugin"] == "web_scrape")
          assert ws["options"]["url_field"] == "url"
          assert ws["options"]["content_field"] == "content"
          assert ws["options"]["fingerprint_field"] == "content_fingerprint"

      def test_http_identity_options_are_operator_supplied_slots(self) -> None:
          args = self._build()
          ws = next(n for n in args["nodes"] if n["plugin"] == "web_scrape")
          assert ws["options"]["http"]["abuse_contact"] == self._SLOTS["abuse_contact"]
          assert ws["options"]["http"]["scraping_reason"] == self._SLOTS["scraping_reason"]

      def test_no_azure_prompt_shield_hard_node(self) -> None:
          """rev 4: omit the unbuildable azure_prompt_shield hard node
          (elspeth-abb2cb0931 — composer cannot instantiate it without
          configured endpoint+api_key secrets)."""
          args = self._build()
          plugins = {n["plugin"] for n in args["nodes"]}
          assert "azure_prompt_shield" not in plugins
  ```
  Run-to-fail:
  ```
  uv run python -m pytest tests/unit/web/composer/test_recipes.py::TestWebScrapeRecipeBuild tests/unit/web/composer/test_recipes.py::TestRecipeRegistry::test_registered_recipes -q
  ```
  Expected failure: `RecipeValidationError: recipe 'web-scrape-llm-rate-jsonl' is
  not registered` (and the registry-set assertion is red).

- [ ] **Step 2: Add the slot table + `_build_web_scrape_recipe` (minimal impl).**
  In `recipes.py`, after `_build_fork_coalesce_truncate_recipe` (ends at :561),
  insert:
  ```python
  # ---------------------------------------------------------------------------
  # Recipe 4: web-scrape-llm-rate-jsonl (D11)
  #
  #   json/csv URL-row source (blob)  →  web_scrape (fetch page content)
  #                                    →  llm (rate the page)
  #                                    →  field_mapper(select_only) cleanup
  #                                    →  jsonl sink (single output)
  #
  # web_scrape is a TRANSFORM, not a source: the head source is a json/csv blob
  # of {url: ...} rows. The field_mapper drops the raw scraped content/fingerprint
  # (data minimization) and stages the kind=pipeline_decision raw-HTML cleanup
  # requirement so the blocking cleanup contract (raw_html_cleanup_review_contract_error,
  # interpretation_state.py:186) passes deterministically.
  #
  # Prompt-injection shield (rev 4, re-polarized): the recipe OMITS an unbuildable
  # azure_prompt_shield hard node (the composer cannot instantiate it without
  # configured endpoint+api_key secrets — elspeth-abb2cb0931, a CONDITIONAL
  # security ticket, NOT a licence to remove all shield signal). It does NOT
  # suppress the existing medium-severity prompt-shield advisory warning
  # (prompt_shield_recommendation_warning_pairs, interpretation_state.py:240),
  # which surfaces at the wire stage. See test_no_azure_prompt_shield_hard_node
  # AND the P4.3 advisory-presence test.
  # ---------------------------------------------------------------------------

  _RECIPE_WEB_SCRAPE_SLOTS: Final[dict[str, SlotSpec]] = {
      "source_blob_id": SlotSpec(
          slot_type="blob_id",
          description="UUID of the operator-supplied URL-list blob (json/csv rows of {url: ...}; use create_blob to wrap inline content first)",
      ),
      "source_plugin": SlotSpec(
          slot_type="str",
          description="Resolved URL-row source plugin carried from match_recipe; must be 'json' or 'csv'. Direct apply callers pass the materialised source plugin explicitly.",
      ),
      "model": SlotSpec(
          slot_type="str",
          description="LLM model identifier (e.g., 'anthropic/claude-sonnet-4.6'); use list_models to discover",
      ),
      "api_key_secret": SlotSpec(
          slot_type="str",
          description=(
              "Name of an inventory secret to wire into the LLM 'api_key' option as "
              "a deferred {secret_ref} marker. Discover names via list_secret_refs; "
              "verify with validate_secret_ref. Literal credential strings are rejected."
          ),
      ),
      "provider": SlotSpec(
          slot_type="str",
          required=False,
          default="openrouter",
          description="LLM provider — 'openrouter' or 'azure'",
      ),
      "rating_template": SlotSpec(
          slot_type="str",
          required=False,
          default="Rate the appeal of this government web page from 1-10 and explain briefly:\n\n{{ row['content'] }}",
          description="Jinja2 template for the rating prompt; reference scraped content as {{ row['content'] }}",
      ),
      "abuse_contact": SlotSpec(
          slot_type="str",
          description=(
              "Operator-owned monitored contact address sent in web_scrape HTTP metadata. "
              "Do not default or invent this value; ask the operator if absent."
          ),
      ),
      "scraping_reason": SlotSpec(
          slot_type="str",
          description=(
              "Operator-authored reason for scraping, sent in web_scrape HTTP metadata. "
              "Do not default or infer this value from the tutorial prose."
          ),
      ),
      "output_path": SlotSpec(
          slot_type="str",
          required=False,
          default="outputs/ratings.jsonl",
          description="JSONL output path",
      ),
  }


  def _build_web_scrape_recipe(slots: Mapping[str, Any]) -> dict[str, Any]:
      """Build set_pipeline args for the web-scrape-llm-rate-jsonl recipe.

      Emits source → web_scrape → llm → field_mapper(cleanup) → jsonl, named by
      connection labels (NOT EdgeSpec objects — guided passes edges=[]). The
      field_mapper drops the raw scraped content/fingerprint and stages the
      kind=pipeline_decision raw-HTML cleanup requirement so the blocking cleanup
      contract passes. The unbuildable azure_prompt_shield hard node is omitted
      (elspeth-abb2cb0931); the existing medium-severity prompt-shield advisory
      is left to fire from validate() — the recipe MUST NOT suppress it.
      """
      from elspeth.web.composer.tools._common import _pending_interpretation_requirement
      from elspeth.contracts.composer_interpretation import InterpretationKind
      from elspeth.web.interpretation_state import (
          INTERPRETATION_REQUIREMENTS_KEY,
          RAW_HTML_CLEANUP_REVIEW_DRAFT,
          RAW_HTML_CLEANUP_USER_TERM,
      )

      content_field = "content"
      fingerprint_field = "content_fingerprint"
      cleanup_requirement = _pending_interpretation_requirement(
          requirement_id="drop_raw_html_review",
          kind=InterpretationKind.PIPELINE_DECISION,
          user_term=RAW_HTML_CLEANUP_USER_TERM,
          draft=RAW_HTML_CLEANUP_REVIEW_DRAFT,
      )
      return {
          "source": {
              "plugin": slots["source_plugin"],
              "blob_id": slots["source_blob_id"],
              "on_success": "rows",
              "options": {
                  "schema": {"mode": "observed"},
              },
              "on_validation_failure": "discard",
          },
          "nodes": [
              {
                  "id": "url_rows",
                  "node_type": "transform",
                  "plugin": "web_scrape",
                  "input": "rows",
                  "on_success": "scraped",
                  "on_error": "discard",
                  "options": {
                      "schema": {"mode": "observed"},
                      "url_field": "url",
                      "content_field": content_field,
                      "fingerprint_field": fingerprint_field,
                      "format": "markdown",
                      "http": {
                          # OPERATOR: these values are visible to scraped third
                          # parties. They are required slots, not tutorial
                          # defaults: use a monitored operator-owned inbox and an
                          # accurate reason, or do not apply the recipe.
                          "abuse_contact": slots["abuse_contact"],
                          "scraping_reason": slots["scraping_reason"],
                      },
                  },
              },
              {
                  "id": "rate_pages",
                  "node_type": "transform",
                  "plugin": "llm",
                  "input": "scraped",
                  "on_success": "rated",
                  "on_error": "discard",
                  "options": {
                      "provider": slots["provider"],
                      "model": slots["model"],
                      "api_key": {"secret_ref": slots["api_key_secret"]},
                      "prompt_template": slots["rating_template"],
                      "response_field": "rating",
                      "schema": {"mode": "observed"},
                      "required_input_fields": [content_field],
                  },
              },
              {
                  "id": "drop_raw_html",
                  "node_type": "transform",
                  "plugin": "field_mapper",
                  "input": "rated",
                  "on_success": "clean",
                  "on_error": "discard",
                  "options": {
                      "schema": {"mode": "observed"},
                      "select_only": True,
                      # mapping preserves ONLY the user-facing fields; the raw
                      # content/fingerprint are intentionally absent (dropped).
                      "mapping": {
                          "url": "url",
                          "rating": "rating",
                      },
                      INTERPRETATION_REQUIREMENTS_KEY: [cleanup_requirement],
                  },
              },
          ],
          "edges": [],
          "outputs": [
              {
                  "sink_name": "clean",
                  "plugin": "json",
                  "options": {
                      "path": slots["output_path"],
                      "format": "jsonl",
                      "schema": {"mode": "observed"},
                      "mode": "write",
                      "collision_policy": "auto_increment",
                  },
                  "on_write_failure": "discard",
              }
          ],
          "metadata": {
              "name": "web-scrape-llm-rate-jsonl",
              "description": (
                  f"Scrape each URL, rate the page with an LLM, drop the raw HTML/"
                  f"fingerprint, and write ratings to {slots['output_path']}"
              ),
          },
      }
  ```

- [ ] **Step 3: Register the `RecipeSpec` in `_RECIPES` (minimal impl).**
  In `recipes.py`, add to the `_RECIPES` dict (after the
  `fork-coalesce-truncate-jsonl` entry at :594-609, before the closing `}`):
  ```python
      "web-scrape-llm-rate-jsonl": RecipeSpec(
          name="web-scrape-llm-rate-jsonl",
          description=(
              "Fetch each URL in a blob of {url: ...} rows, rate the page with an "
              "LLM, drop the raw scraped HTML and fingerprint, and write a JSONL "
              "output of url + rating. Use for: 'scrape these pages and rate them', "
              "'fetch each site and score it'. The URL list must already be uploaded "
              "as a session blob (json or csv rows with a url column). The resolved "
              "source_plugin slot preserves whether the materialised source is json "
              "or csv. The raw-HTML "
              "cleanup is staged as a pipeline_decision so the data-minimization "
              "contract passes deterministically."
          ),
          slots=_RECIPE_WEB_SCRAPE_SLOTS,
          build=_build_web_scrape_recipe,
      ),
  ```
  Run-to-pass:
  ```
  uv run python -m pytest tests/unit/web/composer/test_recipes.py::TestWebScrapeRecipeBuild tests/unit/web/composer/test_recipes.py::TestRecipeRegistry -q
  ```
  Expected: all pass.

- [ ] **Step 4: Verify `match_recipe` end-to-end no longer raises (run-to-pass).**
  Add a regression to `test_recipe_match.py`:
  ```python
  def test_match_recipe_returns_web_scrape_match_for_url_source() -> None:
      """End-to-end: now that the RecipeSpec is registered, match_recipe returns
      a RecipeMatch (no InvariantError) for the canonical URL-row source."""
      from elspeth.web.composer.guided.recipe_match import match_recipe

      source = _make_url_json_source()
      sink = _make_single_jsonl_sink()
      result = match_recipe(source, sink)
      assert result is not None
      assert result.recipe_name == "web-scrape-llm-rate-jsonl"
      assert result.slots["source_blob_id"] == source.options["blob_ref"]
      assert result.slots["source_plugin"] == "json"
      # model/api_key_secret remain unsatisfied (operator fills them via recipe_offer).
      assert "model" in result.unsatisfied_slots
      assert "api_key_secret" in result.unsatisfied_slots
      assert "abuse_contact" in result.unsatisfied_slots
      assert "scraping_reason" in result.unsatisfied_slots

  def test_match_recipe_returns_web_scrape_match_for_csv_url_source() -> None:
      from elspeth.web.composer.guided.recipe_match import match_recipe

      source = _make_url_csv_source()
      sink = _make_single_jsonl_sink()
      result = match_recipe(source, sink)
      assert result is not None
      assert result.recipe_name == "web-scrape-llm-rate-jsonl"
      assert result.slots["source_blob_id"] == source.options["blob_ref"]
      assert result.slots["source_plugin"] == "csv"
  ```
  Run-to-pass:
  ```
  uv run python -m pytest tests/unit/web/composer/guided/test_recipe_match.py::test_match_recipe_returns_web_scrape_match_for_url_source tests/unit/web/composer/guided/test_recipe_match.py::test_match_recipe_returns_web_scrape_match_for_csv_url_source -q
  ```
  Expected: `2 passed`.

- [ ] **Step 4b: Trace the REAL canonical-seed materialised shape through `match_recipe`
  (run-to-pass).** Step 4 used the abstract `_make_url_json_source()` fixture; this test
  pins that the fixture's shape is the one the backend *actually* produces for the
  canonical `inline_blob` seed, so the §4.1 "zero-LLM canonical compose via recipe-match"
  lever provably fires for the real tutorial source. The canonical seed
  (`tutorial_cache.CANONICAL_SEED_PROMPT`; frontend `tutorial.spec.ts:56` `plugin:
  "inline_blob"`, rows of `{url: ...}`) is composed via `set_pipeline` with
  `source.inline_blob`; the backend persists the inline blob and binds a registered
  source plugin via `_MIME_TO_SOURCE` (`application/json → json`) **and unconditionally
  writes `source.options["blob_ref"]`** (`composer/tools/sessions.py:420-427`, inside the
  `if inline_blob is not None` branch). So the materialised `SourceResolved` is
  `plugin="json"`, `options={"blob_ref": <uuid>, "path": ...}`, observed `url` column —
  exactly the `_make_url_json_source()` shape. Append to `test_recipe_match.py`:
  ```python
  def test_canonical_seed_materialised_source_matches_web_scrape_recipe() -> None:
      """§4.1 zero-LLM lever: the REAL canonical tutorial seed, materialised by
      set_pipeline(source.inline_blob), matches the web_scrape recipe.

      Provenance pin — the materialised source shape this test encodes is what
      ``_execute_set_pipeline`` produces for the canonical ``inline_blob`` URL seed
      (``composer/tools/sessions.py:420-427``): an ``application/json`` inline blob
      binds the registered ``json`` source plugin via ``_MIME_TO_SOURCE``
      (``composer/tools/sources.py:148``) AND writes ``source.options["blob_ref"]``
      = the persisted blob UUID UNCONDITIONALLY in the ``if inline_blob is not None``
      branch. So ``SourceResolved.plugin == "json"`` (never the ``"inline_blob"``
      authoring alias, never ``"web_scrape"``) and ``blob_ref`` IS present at
      ``match_recipe`` time — the predicate's blob-presence gate is satisfied and the
      recipe fires. If this assertion ever flips to None, the zero-LLM canonical
      compose is broken; do NOT relax the predicate — fix the materialisation or the
      fixture so the two agree.
      """
      from elspeth.web.composer.guided.recipe_match import match_recipe

      # The materialised canonical source, byte-faithful to sessions.py:420-427:
      # json plugin + path + blob_ref overlay + observed url column.
      canonical_source = SourceResolved(
          plugin="json",  # _MIME_TO_SOURCE["application/json"] -> "json"
          options={
              "path": "composer_blobs/canonical-url-list.json",
              "blob_ref": "a1b2c3d4-0000-0000-0000-000000000099",  # sessions.py:425
          },
          observed_columns=("url",),
          sample_rows=({"url": "https://www.dta.gov.au"},),
      )
      canonical_sink = _make_single_jsonl_sink()

      result = match_recipe(canonical_source, canonical_sink)
      assert result is not None, "canonical seed must match the web_scrape recipe (zero-LLM §4.1)"
      assert result.recipe_name == "web-scrape-llm-rate-jsonl"
      # The slot resolver derives source_blob_id from the materialised blob_ref.
      assert result.slots["source_blob_id"] == canonical_source.options["blob_ref"]
      assert result.slots["source_plugin"] == "json"
  ```
  Run-to-pass:
  ```
  uv run python -m pytest tests/unit/web/composer/guided/test_recipe_match.py::test_canonical_seed_materialised_source_matches_web_scrape_recipe -q
  ```
  Expected: `1 passed`.

- [ ] **Step 5: Lint + mypy + commit.**
  ```
  uv run python -m ruff check src/elspeth/web/composer/recipes.py tests/unit/web/composer/test_recipes.py tests/unit/web/composer/guided/test_recipe_match.py
  uv run python -m ruff format src/elspeth/web/composer/recipes.py tests/unit/web/composer/test_recipes.py tests/unit/web/composer/guided/test_recipe_match.py
  uv run python -m mypy src/elspeth/web/composer/recipes.py
  git add src/elspeth/web/composer/recipes.py src/elspeth/web/composer/guided/recipe_match.py tests/unit/web/composer/test_recipes.py tests/unit/web/composer/guided/test_recipe_match.py
  git commit -m "feat(composer/recipes): add web-scrape-llm-rate-jsonl recipe (D11/P4.2)

Deterministically emits json-url-source → web_scrape → llm → field_mapper
(select_only cleanup, drops raw content/fingerprint) → jsonl, naming the head
source node and staging the kind=pipeline_decision raw-HTML cleanup so the
blocking cleanup contract passes. Omits the unbuildable azure_prompt_shield
hard node (elspeth-abb2cb0931) without suppressing the existing prompt-shield
advisory. Wires the slot resolver into the predicate registry.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```
  Expected: ruff + mypy clean, commit succeeds.

---

### Task P4.3: Contract + shield-advisory + data-minimization integration test (CompositionState)

**Files:**
- Create `tests/unit/web/composer/test_web_scrape_recipe_contract.py`

**Interfaces:**
- Consumes: `_build_web_scrape_recipe` (P4.2), `CompositionState` /
  `SourceSpec` / `NodeSpec` / `OutputSpec` / `PipelineMetadata`
  (`composer/state.py`), `composition_review_contract_error` +
  `prompt_shield_recommendation_warning_pairs` (`web/interpretation_state.py`),
  `interpretation_sites` (`interpretation_state.py:403`).
- Produces: a state-level pin that the **built recipe satisfies the blocking
  cleanup contract** AND **preserves the live prompt-shield advisory** AND
  **drops raw fields** — the load-bearing rev-4 re-polarized shield test.

> **Why a hand-built `CompositionState` (not `apply_recipe`).** `apply_recipe` →
> `_execute_set_pipeline` requires session+blob context to resolve `blob_id`. The
> contract/validate logic operates on `CompositionState`, so this test reconstructs
> the recipe's node graph as a `CompositionState` (mirroring
> `tests/unit/web/test_interpretation_state.py:85` `_state_with_web_scrape_cleanup_node`)
> and asserts directly. The set_pipeline-args shape (P4.2 tests) and this state-shape
> test together cover both halves.

- [ ] **Step 1: Write the state-builder helper + contract-pass test (run-to-fail).**
  Create the file:
  ```python
  """State-level contract + shield-advisory tests for the web-scrape recipe (D11).

  Pins the rev-4 re-polarized shield behaviour: the built pipeline omits the
  azure_prompt_shield HARD NODE but the medium-severity prompt-shield ADVISORY
  warning IS present (elspeth-abb2cb0931 is a conditional 'restore once plugins
  gate on secret availability' ticket, NOT a licence to hide the signal).
  """

  from __future__ import annotations

  from typing import Any

  from elspeth.web.composer.recipes import _build_web_scrape_recipe
  from elspeth.web.composer.state import (
      CompositionState,
      NodeSpec,
      OutputSpec,
      PipelineMetadata,
      SourceSpec,
  )
  from elspeth.web.interpretation_state import (
      composition_review_contract_error,
      interpretation_sites,
      prompt_shield_recommendation_warning_pairs,
  )

  _SLOTS = {
      "source_blob_id": "a1b2c3d4-0000-0000-0000-0000000000aa",
      "source_plugin": "json",
      "model": "anthropic/claude-sonnet-4.6",
      "api_key_secret": "OPENROUTER_API_KEY",
      "abuse_contact": "web-scrape-contact@dta.gov.au",
      "scraping_reason": "Tutorial exercise: fetch public pages for rating",
      "output_path": "outputs/ratings.jsonl",
  }


  def _node_from_args(node_args: dict[str, Any]) -> NodeSpec:
      return NodeSpec(
          id=node_args["id"],
          node_type=node_args["node_type"],
          plugin=node_args["plugin"],
          input=node_args["input"],
          on_success=node_args.get("on_success"),
          on_error=node_args.get("on_error"),
          options=node_args["options"],
          condition=None,
          routes=None,
          fork_to=None,
          branches=None,
          policy=None,
          merge=None,
      )


  def _state_from_recipe() -> CompositionState:
      args = _build_web_scrape_recipe(_SLOTS)
      src = args["source"]
      source = SourceSpec(
          plugin=src["plugin"],
          on_success=src["on_success"],
          options=src["options"],
          on_validation_failure=src["on_validation_failure"],
      )
      out = args["outputs"][0]
      output = OutputSpec(
          name=out["sink_name"],
          plugin=out["plugin"],
          options=out["options"],
          on_write_failure=out["on_write_failure"],
      )
      return CompositionState(
          source=source,
          nodes=tuple(_node_from_args(n) for n in args["nodes"]),
          edges=(),
          outputs=(output,),
          metadata=PipelineMetadata(),
          version=1,
      )


  def test_built_recipe_passes_blocking_cleanup_contract() -> None:
      """The staged pipeline_decision satisfies raw_html_cleanup_review_contract_error,
      so composition_review_contract_error (tools/sessions.py:670) is None."""
      state = _state_from_recipe()
      assert composition_review_contract_error(state) is None
  ```
  Run-to-fail (the helper imports may need adjusting if `OutputSpec`/`SourceSpec`
  fields differ — confirm at write time against `state.py:119/287`):
  ```
  uv run python -m pytest tests/unit/web/composer/test_web_scrape_recipe_contract.py::test_built_recipe_passes_blocking_cleanup_contract -q
  ```
  Expected: PASS once the helper compiles. If `composition_review_contract_error`
  returns a non-None string, the field_mapper requirement staging in P4.2 is wrong
  — fix `_build_web_scrape_recipe` (do NOT weaken the test).

- [ ] **Step 2: Write the re-polarized shield-advisory PRESENCE test (run-to-fail).**
  Append:
  ```python
  def test_prompt_shield_advisory_is_present_no_hard_node() -> None:
      """Re-polarized shield (rev 4): assert (a) no azure_prompt_shield HARD NODE,
      AND (b) the medium-severity prompt-shield ADVISORY warning IS present — pin
      the PRESENCE of the security signal, not its absence. The flagship example
      must not be the one web_scrape→llm pipeline that hides the warning the rest
      of the system shows.

      See elspeth-abb2cb0931 (conditional 'restore the shield advice once plugins
      gate on secret availability' ticket).
      """
      state = _state_from_recipe()

      # (a) no unbuildable hard node
      assert all(node.plugin != "azure_prompt_shield" for node in state.nodes)

      # (b) the advisory IS present (web_scrape → llm without a shield)
      warning_pairs = prompt_shield_recommendation_warning_pairs(state)
      assert warning_pairs, "expected the medium-severity prompt-shield advisory to fire"
      components = {component for component, _message in warning_pairs}
      assert "node:rate_pages" in components

  def test_prompt_shield_advisory_surfaces_in_validate_warnings() -> None:
      """The same advisory rides validate().warnings at 'medium' severity
      (state.py:2421), which is the payload the wire stage renders
      (_authoring_validation_payload['warnings'], tools/sessions.py:1166)."""
      state = _state_from_recipe()
      summary = state.validate()
      shield_warnings = [
          w for w in summary.warnings
          if "prompt-injection shield" in w.message and w.severity == "medium"
      ]
      assert shield_warnings, "prompt-shield advisory must ride validate().warnings at medium severity"
  ```
  Run-to-fail then run-to-pass:
  ```
  uv run python -m pytest tests/unit/web/composer/test_web_scrape_recipe_contract.py -q
  ```
  Expected: PASS. If the advisory is absent, the recipe accidentally suppressed it
  (e.g. wrong plugin name on the rating node, or it staged a
  prompt_injection_shield_recommendation requirement) — fix the recipe so the
  advisory fires; **do not** make the test green by asserting absence.
  (Confirm `ValidationEntry` exposes `.message` / `.severity` at write time; if the
  field is `.text`, adjust the comprehension accordingly.)

- [ ] **Step 3: Write the data-minimization site test (run-to-fail).**
  Append:
  ```python
  def test_cleanup_node_drops_raw_fields_no_orphan_site() -> None:
      """Data minimization: the raw content/fingerprint fields are NOT preserved,
      and because the pipeline_decision is staged, no missing-cleanup interpretation
      site orphans (interpretation_sites returns no raw-html-cleanup site for the
      field_mapper)."""
      state = _state_from_recipe()
      fm = next(n for n in state.nodes if n.plugin == "field_mapper")
      mapping = fm.options["mapping"]
      preserved = set(mapping) | set(mapping.values())
      assert "content" not in preserved
      assert "content_fingerprint" not in preserved

      from elspeth.contracts.composer_interpretation import InterpretationKind

      sites = interpretation_sites(state)
      raw_html_sites = [
          s for s in sites
          if s.kind is InterpretationKind.PIPELINE_DECISION and s.user_term == "drop_raw_html_fields"
      ]
      assert raw_html_sites == [], "raw-html cleanup must be staged (not orphaned)"
  ```
  Run-to-pass:
  ```
  uv run python -m pytest tests/unit/web/composer/test_web_scrape_recipe_contract.py -q
  ```
  Expected: all pass.

- [ ] **Step 3b: Add an end-to-end `execute_tool("apply_pipeline_recipe", ...)`
  proof with a real catalog and a seeded URL-row blob.**
  Extend the same file or add `tests/unit/web/composer/test_web_scrape_recipe_apply.py`
  using the existing `TestApplyRecipeEndToEnd` fixture style in
  `tests/unit/web/composer/test_recipes.py:784`. Seed a session blob containing
  URL rows, use `create_catalog_service()` (not a hand-rolled catalog), apply the
  recipe through the public dispatcher, and assert the resulting state preserves
  the advisory polarity:
  ```python
  from elspeth.web.composer.state import CompositionState, PipelineMetadata
  from elspeth.web.composer.tools import execute_tool
  from elspeth.web.dependencies import create_catalog_service
  from elspeth.web.interpretation_state import prompt_shield_recommendation_warning_pairs


  def test_apply_web_scrape_recipe_end_to_end_preserves_shield_advisory(_seeded_url_blob) -> None:
      engine, session_id, blob_id = _seeded_url_blob  # seeded body: [{"url": "https://www.dta.gov.au"}]
      empty = CompositionState(
          source=None,
          nodes=(),
          edges=(),
          outputs=(),
          metadata=PipelineMetadata(),
          version=1,
      )
      result = execute_tool(
          "apply_pipeline_recipe",
          {
              "recipe_name": "web-scrape-llm-rate-jsonl",
              "slots": {
                  "source_blob_id": blob_id,
                  "source_plugin": "json",
                  "model": "anthropic/claude-sonnet-4.6",
                  "api_key_secret": "OPENROUTER_API_KEY",
                  "abuse_contact": "web-scrape-contact@dta.gov.au",
                  "scraping_reason": "Tutorial exercise: fetch public pages for rating",
                  "output_path": "outputs/ratings.jsonl",
              },
          },
          empty,
          create_catalog_service(),
          session_engine=engine,
          session_id=session_id,
      )
      assert result.success, getattr(result, "data", result)
      state = result.updated_state
      assert all(node.plugin != "azure_prompt_shield" for node in state.nodes)
      warning_pairs = prompt_shield_recommendation_warning_pairs(state)
      assert warning_pairs, "prompt-shield advisory must remain warning/advisory"
      assert any(component == "node:rate_pages" for component, _ in warning_pairs)
  ```
  Add a CSV companion by seeding a `text/csv` URL-row blob and passing
  `"source_plugin": "csv"`; assert `result.updated_state.sources["source"].plugin
  == "csv"` and the same prompt-shield advisory/no-hard-node conditions hold.
  This is the full apply proof for the JSON/CSV resolver path; do not replace it
  with an isolated `_build_web_scrape_recipe` call.

  Run-to-pass:
  ```
  uv run python -m pytest tests/unit/web/composer/test_web_scrape_recipe_apply.py -q
  ```
  Expected: JSON and CSV apply tests pass; prompt-shield remains advisory/warning
  and no hard `azure_prompt_shield` node is inserted.

- [ ] **Step 4: Lint + commit.**
  ```
  uv run python -m ruff check tests/unit/web/composer/test_web_scrape_recipe_contract.py tests/unit/web/composer/test_web_scrape_recipe_apply.py
  uv run python -m ruff format tests/unit/web/composer/test_web_scrape_recipe_contract.py tests/unit/web/composer/test_web_scrape_recipe_apply.py
  git add tests/unit/web/composer/test_web_scrape_recipe_contract.py tests/unit/web/composer/test_web_scrape_recipe_apply.py
  git commit -m "test(composer/recipes): pin web-scrape recipe contract + re-polarized shield (D11/P4.3)

Builds the recipe into a CompositionState and asserts: (1) the staged
pipeline_decision satisfies the blocking raw-HTML cleanup contract;
(2) re-polarized shield — NO azure_prompt_shield hard node BUT the
medium-severity prompt-shield advisory IS present (presence, not absence;
refs elspeth-abb2cb0931); (3) data minimization — raw content/fingerprint
dropped with no orphaned interpretation site.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```
  Expected: ruff clean, commit succeeds.

---

### Task P4.4: Zero-LLM compose assertion (recipe-apply makes no provider call)

**Files:**
- Create `tests/unit/web/composer/test_web_scrape_recipe_zero_llm.py`

**Interfaces:**
- Consumes: `_build_web_scrape_recipe` / `apply_recipe` (P4.2),
  `validate_slots` (`recipes.py:119`). Patch target for the provider chokepoint:
  `elspeth.web.composer.service._litellm_acompletion` (the single LiteLLM
  chokepoint, per the existing `test_compose_loop_llm_audit.py:186` pattern).
- Produces: the gate for the §4.1 "zero-LLM canonical compose" claim — the recipe
  build path performs **no** LLM provider call.

> **Scope note.** P4 owns the recipe build, not the dispatch wiring (P2). This test
> proves the **recipe-build path itself** is provider-free: building the
> set_pipeline args and validating the slots calls the LLM zero times. The full
> dispatch-level zero-LLM assertion (`/api/tutorial/run` cache freeze) is asserted by P7
> once the recipe-apply seam redirects through STEP_4_WIRE.

- [ ] **Step 1: Write the zero-LLM build test (run-to-fail).**
  Create the file:
  ```python
  """Zero-LLM compose gate for the web-scrape recipe (D11, §4.1).

  Building the canonical recipe is a pure, deterministic function: it must make
  ZERO LLM provider calls. Pins the §4.1 claim that the canonical pipeline
  composes with no frontier round-trip at recipe-build time.
  """

  from __future__ import annotations

  from unittest.mock import AsyncMock, patch
  from uuid import uuid4

  from elspeth.web.composer.recipes import _build_web_scrape_recipe, apply_recipe

  _SLOTS = {
      "source_blob_id": str(uuid4()),
      "source_plugin": "json",
      "model": "anthropic/claude-sonnet-4.6",
      "api_key_secret": "OPENROUTER_API_KEY",
      "abuse_contact": "web-scrape-contact@dta.gov.au",
      "scraping_reason": "Tutorial exercise: fetch public pages for rating",
      "output_path": "outputs/ratings.jsonl",
  }


  def test_build_web_scrape_recipe_makes_zero_llm_calls() -> None:
      with patch(
          "elspeth.web.composer.service._litellm_acompletion",
          new_callable=AsyncMock,
      ) as mock_acomp:
          args = _build_web_scrape_recipe(_SLOTS)
          # llm node IS present in the COMPOSED pipeline (it runs at RUN time,
          # not compose time) — but the build itself called no provider.
          assert any(n["plugin"] == "llm" for n in args["nodes"])
      assert mock_acomp.call_count == 0


  def test_apply_web_scrape_recipe_makes_zero_llm_calls() -> None:
      with patch(
          "elspeth.web.composer.service._litellm_acompletion",
          new_callable=AsyncMock,
      ) as mock_acomp:
          args = apply_recipe("web-scrape-llm-rate-jsonl", _SLOTS)
          assert args["metadata"]["name"] == "web-scrape-llm-rate-jsonl"
      assert mock_acomp.call_count == 0
  ```
  Run-to-fail/pass:
  ```
  uv run python -m pytest tests/unit/web/composer/test_web_scrape_recipe_zero_llm.py -q
  ```
  Expected: PASS (the build is pure). If the patch target import path is wrong,
  fix the dotted path against `test_compose_loop_llm_audit.py:186`; do not skip.

- [ ] **Step 2: Lint + commit.**
  ```
  uv run python -m ruff check tests/unit/web/composer/test_web_scrape_recipe_zero_llm.py
  uv run python -m ruff format tests/unit/web/composer/test_web_scrape_recipe_zero_llm.py
  git add tests/unit/web/composer/test_web_scrape_recipe_zero_llm.py
  git commit -m "test(composer/recipes): zero-LLM gate for web-scrape recipe build (D11/P4.4)

Pins §4.1: building/applying the canonical recipe makes zero _litellm_acompletion
calls. The llm node runs at RUN time, never at compose time.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```
  Expected: ruff clean, commit succeeds.

---

### Task P4.5: Phase gate sweep + plugin-hash refresh

**Files:**
- Modify (if the CI plugin-hash gate flags it) `tests`/baseline only — no source
  change expected (P4 edits no plugin file; it edits `recipes.py` + `recipe_match.py`
  which are composer modules, not registered plugins).

**Interfaces:** none new.

- [ ] **Step 1: Run the full P4 test slice (run-to-pass).**
  ```
  uv run python -m pytest \
      tests/unit/web/composer/guided/test_recipe_match.py \
      tests/unit/web/composer/test_recipes.py \
      tests/unit/web/composer/test_web_scrape_recipe_contract.py \
      tests/unit/web/composer/test_web_scrape_recipe_apply.py \
      tests/unit/web/composer/test_web_scrape_recipe_zero_llm.py \
      -q
  ```
  Expected: all pass, `0 failed`.

- [ ] **Step 2: Lint + format + mypy over the touched source (run-to-pass).**
  ```
  uv run python -m ruff check src/elspeth/web/composer/recipes.py src/elspeth/web/composer/guided/recipe_match.py
  uv run python -m ruff format --check src/elspeth/web/composer/recipes.py src/elspeth/web/composer/guided/recipe_match.py
  uv run python -m mypy src/elspeth/web/composer/recipes.py src/elspeth/web/composer/guided/recipe_match.py
  ```
  Expected: `All checks passed!`, format clean, mypy `Success: no issues found`.

- [ ] **Step 3: trust-tier + wardline gate (recipe stages Tier-3 review text + LLM options).**
  The recipe authors `interpretation_requirements` (review text) and an
  `api_key` `{secret_ref}` — confirm no new trust-tier displacement and no taint
  finding:
  ```
    PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules trust_tier.tier_model,'composer/*' --root src/elspeth/web/composer/recipes.py
    wardline scan . --fail-on ERROR
  ```
  Expected: elspeth-lints check passes (the recipe builder is a pure function over
  already-validated slots — `apply_recipe` runs `validate_slots` first, so no new
  Tier-3 boundary is introduced); `wardline` exit 0. If trust-tier reports drift,
  it is a pre-existing operator-owned HMAC re-pin (state it in the commit, do not
  sign blind — see CLAUDE.md gate-debt doctrine).

- [ ] **Step 4: Final phase commit (only if Step 3 produced allowlist/test churn).**
  ```
    git add src/elspeth/web/composer/recipes.py src/elspeth/web/composer/guided/recipe_match.py tests/unit/web/composer/test_recipes.py tests/unit/web/composer/guided/test_recipe_match.py tests/unit/web/composer/test_web_scrape_recipe_contract.py tests/unit/web/composer/test_web_scrape_recipe_apply.py tests/unit/web/composer/test_web_scrape_recipe_zero_llm.py
  git commit -m "chore(composer/recipes): P4 gate sweep — web-scrape recipe slice green

Full P4 recipe slice (predicate + RecipeSpec + contract + zero-LLM) passes;
ruff/mypy clean; trust-tier + wardline checked. No plugin file edited (recipes.py
and recipe_match.py are composer modules, not registered plugins — no
source_file_hash refresh owed).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```
  Expected: commit succeeds (or "nothing to commit" if Step 3 was clean — that is
  acceptable, the slice is already committed by P4.1–P4.4).

---
