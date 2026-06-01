# DECORATOR BATCH 3 — DISPLACED SIGNED ENTRIES (operator re-sign worklist)
#
# 25 judge-signed allowlist entries in 3 files were staled by the
# `from elspeth.contracts.trust_boundary import trust_boundary` import added
# at each file's module top (shifts Module.body indices -> rotates every
# downstream entry's key fp). Per the operator-resign decision (batch 3), these
# were DELETED (pure text removal — forges nothing) and must be re-`justify`'d
# against their live (rotated) nodes. The keyless agent never re-signs.
#
# Re-sign each via: elspeth-lints justify --file-path <file> --rule <rule>
#   --symbol '<symbol>' --fingerprint <NEW_FP> --rationale '<...>' --owner <owner>
# (NEW_FP listed below = the live finding fp after the import shift; the judge
# re-runs against the current node, so the OLD rationale is a starting point,
# not a verbatim copy. AMBIGUOUS groups: the OLD->NEW fp pairing is NOT
# mechanically determinable — re-judge each live node; any bijection is valid.)


==============================================================================
## web/composer/source_inspection.py  — 4 displaced signed entries
==============================================================================

--- _declared_field_name | R5 | 1 entry ---
  NEW fingerprint(s) to re-sign against (1):
    - 468c6b8126cbe145   (live L774: isinstance() used: if isinstance(name_raw, str):)
  OLD entries (rationale + owner to carry into re-justify):
    * OLD_FP 684a0b8827dc898e | owner=web-composer | ast_path=body[49]/body[0]/test
      rationale: The isinstance check here is genuine union-type discrimination on `DeclaredFieldSpec = str | Mapping[str, Any]`, not a defensive shape guard against untrusted data. The two arms genuinely differ in how they extract the field name (string-split on ":" vs. mapping lookup of "name"), which is exactly the typed-dispatch pattern Python requires for tagged-union handling. This is not the R5 pattern the policy forbids (an isinstance used to silently coerce or recover from malformed Tier-3 input); it is mandatory control flow over a well-typed sum type defined in our own code. The decorator nudge does not apply — `field` is not rooted at an external function parameter carrying Tier-3 data; `DeclaredFieldSpec` is a typed contract.

--- _inspect_json | R6 | 1 entry ---
  NEW fingerprint(s) to re-sign against (1):
    - 7ac9c954381a870b   (live L504: Exception swallowed without re-raise or explicit error: except json.JS)
  OLD entries (rationale + owner to carry into re-justify):
    * OLD_FP 314ea348e863a6f4 | owner=web-composer | ast_path=body[38]/body[5]/handlers[0]
      rationale: This is a legitimate Tier-3 trust boundary: the function parses operator-supplied blob bytes that may be truncated at an 8 KiB peek or malformed. JSONDecodeError is an expected outcome of parsing untrusted external content, and the handler distinguishes the truncated-vs-malformed cases and surfaces both via warnings rather than silently swallowing. The exception is narrowly typed (json.JSONDecodeError only), the failure is recorded in operator-visible warnings, and loaded=None then flows through type-checked branches — this matches the 'validate at the boundary, record what we got' guidance for Tier-3. R6 (broad/silent exception handling) does not fit here because the catch is specific and observable.

--- _inspect_jsonl | R6 | 1 entry ---
  NEW fingerprint(s) to re-sign against (1):
    - 4e2ec0402671ae1e   (live L468: Exception swallowed without re-raise or explicit error: except json.JS)
  OLD entries (rationale + owner to carry into re-justify):
    * OLD_FP 327fe9a5a3e59ac5 | owner=web-composer | ast_path=body[37]/body[6]/body[3]/handlers[0]
      rationale: This is a legitimate Tier-3 trust boundary: the function parses operator-supplied JSONL sample bytes line-by-line during source inspection. Catching JSONDecodeError per-line is the correct boundary behavior — malformed lines are counted as parse failures and surfaced via a warning, while parsing continues for remaining lines. The exception is narrow (JSONDecodeError, not bare except), the failure is recorded rather than swallowed, and crashing the whole inspection on one bad line would defeat the purpose of inspecting untrusted blob content. R6 (silent exception handling) does not apply meaningfully here because the failure is explicitly counted and reported.

--- _facts_from_objects | R8 | 1 entry ---
  NEW fingerprint(s) to re-sign against (1):
    - baaedaff04f02989   (live L579: dict.setdefault() hides missing keys: seen_keys.setdefault(k, None))
  OLD entries (rationale + owner to carry into re-justify):
    * OLD_FP 03c72d42852801e5 | owner=web-composer | ast_path=body[39]/body[3]/body[0]/body[0]/value
      rationale: R8 (likely the dict[str, Any] / loosely-typed mapping rule) is being suppressed on a function whose explicit role is to ingest parsed JSON sample rows from external sources — genuine Tier-3 data with arbitrary operator-supplied schemas. A TypedDict cannot model arbitrary external JSON, and the function correctly performs offensive type-narrowing via isinstance dispatch on each value before use. The rationale is site-specific and accurately identifies the trust tier and the structural reason a stricter type cannot apply. R8 is not in the @trust_boundary decorator's suppressible set (R1/R5), so the decorator nudge does not apply.


==============================================================================
## web/composer/tools/blobs.py  — 4 displaced signed entries
==============================================================================

--- _session_blob_lock | R1 | 2 entries | AMBIGUOUS (no 1:1 old->new mapping) ---
  NEW fingerprint(s) to re-sign against (2):
    - c5b8692b1242fdb4   (live L1002: Potential dict.get() usage: lock = _SESSION_BLOB_LOCKS.get(session_id))
    - 1ff926b2bf15f5b6   (live L1006: Potential dict.get() usage: lock = _SESSION_BLOB_LOCKS.get(session_id))
  OLD entries (rationale + owner to carry into re-justify):
    * OLD_FP 1b80b2f85d944ac6 | owner=composer-tools-rearchitect | ast_path=body[63]/body[1]/value
      rationale: This is a legitimate get-or-create pattern on a process-local registry: `_SESSION_BLOB_LOCKS.get(session_id)` returns None on miss, which is the explicit signal to enter the registry mutex and install a new lock. The `.get()` is not defensive against malformed external data — it's the canonical double-checked-locking idiom where None-as-miss is load-bearing control flow. The data is not Tier-3 (the dict is a process-local lock registry we own), so the @trust_boundary decorator does not apply. Replacing with subscript access would force a try/except KeyError that is strictly worse than the current form. Rationale wording is duplicated from a prior entry but the code excerpt independently justifies the suppression.
    * OLD_FP 1572f749e0721c8d | owner=composer-tools-rearchitect | ast_path=body[63]/body[3]/body[0]/value
      rationale: This is a textbook double-checked locking pattern on a process-local registry dict (_SESSION_BLOB_LOCKS). The .get() call is not defensive handling of external/Tier-3 data — session_id is a string key, and .get() returning None is the correct semantic primitive for 'is there an existing lock?' that drives the create-if-absent branch. Using direct subscript would force a try/except KeyError that expresses exactly the same control flow more awkwardly. The decorator nudge does not apply: this is not a Tier-3 boundary function and the subject is a module-level dict, not an external parameter. Rationale wording is boilerplate from a refactor merge, but the code excerpt independently supports the suppression.

--- _execute_delete_blob | R6 | 1 entry ---
  NEW fingerprint(s) to re-sign against (1):
    - 6f346528de4eff44   (live L1491: Exception swallowed without re-raise or explicit error: except OSError)
  OLD entries (rationale + owner to carry into re-justify):
    * OLD_FP bf8563c850ca9a87 | owner=composer-tools-rearchitect | ast_path=body[68]/body[9]/handlers[0]/body[0]/body[0]/handlers[0]
      rationale: The flagged except OSError block does not silently swallow the rollback failure: it attaches a detailed note to primary_exc via add_note() (recording the rollback path, exception type/message, and the divergence warning for manual reconciliation), and the outer handler then re-raises primary_exc on line 1487. This is the correct pattern for a best-effort rollback inside an exception handler — re-raising rollback_exc would mask the primary failure, while annotating-and-re-raising preserves both. R6's 'swallowed without re-raise' heuristic misfires here because the re-raise is at the enclosing handler scope. The audit trail is preserved through exception chaining and the explicit note.

--- _execute_update_blob | R6 | 1 entry ---
  NEW fingerprint(s) to re-sign against (1):
    - 7ebc32f4e7eeeb5c   (live L1352: Exception swallowed without re-raise or explicit error: except OSError)
  OLD entries (rationale + owner to carry into re-justify):
    * OLD_FP 96d73b682ad0a226 | owner=composer-tools-rearchitect | ast_path=body[66]/body[10]/body[10]/body[1]/handlers[2]/body[0]/body[0]/handlers[0]
      rationale: The except OSError block does not silently swallow the error — it attaches a detailed note to primary_exc via add_note() and then re-raises via the bare 'raise' on line 1351. This is the correct offensive-programming pattern for rollback-failure handling at a filesystem boundary: the primary exception propagates with enriched diagnostic context about the storage/DB divergence, and the narrow OSError catch (rather than Exception) ensures programmer bugs still propagate. The R6 'exception swallowed without re-raise' rule is misfiring here because the re-raise is at the outer except's tail rather than inside the inner handler. The rationale accurately describes an AST-shift preservation of an already-correct pattern.


==============================================================================
## web/composer/tools/generation.py  — 17 displaced signed entries
==============================================================================

--- _numeric_aggregation_diagnostics_for_observed_csv | R1 | 1 entry ---
  NEW fingerprint(s) to re-sign against (1):
    - 216a69c122f772d8   (live L1105: Potential dict.get() usage: inferred_type = inferred_types.get(value_f)
  OLD entries (rationale + owner to carry into re-justify):
    * OLD_FP 9c1695c48f0efa04 | owner=composer-tools-generation | ast_path=body[60]/body[4]/body[2]/value
      rationale: The function reads aggregation contract options that originate from LLM-authored composition input — Tier-3 external data whose shape is not contractually guaranteed by an ELSPETH typed contract. The .get('value_field') is followed immediately by a strict type/non-empty check (``type(value_field) is not str or not value_field.strip()``) that abstains on absence or wrong shape rather than fabricating a default, which is the honest Tier-3 guard pattern. R1 (dict.get) is the flagged rule and the subject is rooted at options derived from LLM-authored node.options, so the suppression addresses exactly the pattern the rule targets. The decorator nudge does not cleanly apply here because ``options`` is not a direct function parameter but a value derived via get_aggregation_contract_options from state.nodes; the per-line allowlist is the appropriate granularity.

--- _source_schema_mode | R1 | 1 entry ---
  NEW fingerprint(s) to re-sign against (1):
    - 8009b82001221272   (live L896: Potential dict.get() usage: mode = schema.get("mode"))
  OLD entries (rationale + owner to carry into re-justify):
    * OLD_FP 23513e8813c297f6 | owner=composer-tools-generation | ast_path=body[53]/body[2]/value
      rationale: The helper operates on source.options['schema'], which is operator/LLM-authored composer state — Tier-3 external data whose shape is not contractually guaranteed (the preceding isinstance(schema, Mapping) guard at line 848 confirms the shape is untrusted). Within that Tier-3 mapping, 'mode' is genuinely optional, and the function honestly records absence by returning None rather than fabricating a default — consistent with the fabrication-decision test. The function is not @trust_boundary-decorated, but the parameter is a SourceSpec (a first-party typed contract), not the external dict itself; the externally-sourced value is reached via source.options.get('schema'), so source_param wouldn't cleanly apply here. The .get on the schema mapping is a legitimate Tier-3 optional-field read.

--- compute_proof_diagnostics | R1 | 3 entries | AMBIGUOUS (no 1:1 old->new mapping) ---
  NEW fingerprint(s) to re-sign against (3):
    - d687ac034ff293c0   (live L1264: Potential dict.get() usage: declared = schema.get("fields") or ())
    - 2d4dc010a1b6410f   (live L1266: Potential dict.get() usage: headerless_columns = source.plugin == "csv)
    - bfbecb913540fc00   (live L1327: Potential dict.get() usage: elif schema.get("mode") == "fixed" and not)
  OLD entries (rationale + owner to carry into re-justify):
    * OLD_FP 0945f207a98cbbff | owner=composer-tools-generation | ast_path=body[61]/body[4]/value
      rationale: The code accesses `source.options` which is LLM-authored composer state — the options dict is shaped by tool calls from the composer model, and `blob_ref` is schema-optional (present only when set_source_from_blob wired a blob, absent for path-based sources). The `.get()` returning None is exactly the legitimate Tier-3 boundary pattern: absence is recorded as None, the function abstains (returns empty diagnostics) rather than fabricating. The rationale correctly identifies the schema-optional nature and the abstain-on-absent behaviour, and the surrounding comments distinguish this from the Tier-1 BlobToolRecord access below (which correctly uses direct subscript).
    * OLD_FP f0e165ac927832cd | owner=composer-tools-generation | ast_path=body[61]/body[14]/body[1]/test/values[1]/left
      rationale: The flagged `.get('mode')` is on `schema`, a value extracted from `source.options` where the LLM (composer model) authored the configuration. While `source.options` itself is a Mapping by contract, the inner `schema` value's contents are LLM-authored and shape-unguaranteed — the code has already narrowed `schema` to `Mapping` via `isinstance`, and now probes for an optional `mode` discriminator whose presence and type are not guaranteed. This is the Tier-3 boundary pattern: guard the shape of externally-sourced unstructured nested data. The rationale correctly identifies the trust tier and addresses R1 specifically (discriminator probe on absent/non-string key).
    * OLD_FP 157afff4c5b89efd | owner=composer-tools-generation | ast_path=body[61]/body[14]/body[1]/body[0]/value/values[0]
      rationale: The excerpt's own comment establishes that source.options is Tier-1 but the value at 'schema' is LLM-authored unstructured data (Tier-3) — the schema mapping came from an LLM tool call, not a first-party typed contract. The surrounding code already shape-guards with isinstance(schema, Mapping) before any .get, and the flagged schema.get('fields') reads an optional declared-fields key from that untrusted mapping before the next isinstance check at line 1219 validates the shape. This is the legitimate Tier-3 boundary pattern: guard shape, then validate. The 'or ()' coerces a falsy/absent value to an empty tuple which still flows through the isinstance check, so no fabrication is smuggled into audit data. The function is not obviously a single-parameter trust_boundary candidate (schema is reached via source.options, not a direct external param), so the decorator nudge does not cleanly apply.

--- _row_fields_referenced_by_condition | R5 | 9 entries | AMBIGUOUS (no 1:1 old->new mapping) ---
  NEW fingerprint(s) to re-sign against (9):
    - d2ce1314623cf084   (live L919: isinstance() used: isinstance(node, ast.Subscript))
    - a853c4d37e0ab0f4   (live L920: isinstance() used: and isinstance(node.value, ast.Name))
    - 9e3396be69940de5   (live L922: isinstance() used: and isinstance(node.slice, ast.Constant))
    - 46721045f0f20130   (live L923: isinstance() used: and isinstance(node.slice.value, str))
    - d7042c58e9f9964c   (live L928: isinstance() used: isinstance(node, ast.Call))
    - 29bd2e75e6cdf05f   (live L929: isinstance() used: and isinstance(node.func, ast.Attribute))
    - f04db137fa0f3f6f   (live L931: isinstance() used: and isinstance(node.func.value, ast.Name))
    - 326106a6c4ebfdfa   (live L934: isinstance() used: and isinstance(node.args[0], ast.Constant))
    - 1ee8b77a6b6e4453   (live L935: isinstance() used: and isinstance(node.args[0].value, str))
  OLD entries (rationale + owner to carry into re-justify):
    * OLD_FP fb7e9f0ac5e38ba3 | owner=composer-tools-generation | ast_path=body[55]/body[2]/body[0]/test/values[0]
      rationale: The isinstance checks here are sum-type dispatch over ast.AST node variants produced by ast.parse — the canonical Python idiom for walking an AST and filtering by node kind. This is not a defensive type guard on untrusted data; ast.walk yields a heterogeneous union and isinstance is how you discriminate variants. The condition string itself is operator-supplied configuration, but the isinstance checks are about AST shape, not data trust. R5 is misfiring on legitimate structural pattern matching.
    * OLD_FP 1988212271415914 | owner=composer-tools-generation | ast_path=body[55]/body[2]/body[0]/test/values[1]
      rationale: The isinstance check is performing legitimate AST node-type discrimination, not a Tier-3 shape guard on external data. ast.walk yields heterogeneous node types and isinstance is the canonical, correct way to narrow ast.Subscript's .value field to ast.Name before accessing .id. This is structural pattern-matching on a typed AST, equivalent to a match/case discriminator — exactly the kind of offensive type narrowing the policy permits. R5 misfires on AST visitor patterns; suppression is site-appropriate.
    * OLD_FP 42dc2828b0e57306 | owner=composer-tools-generation | ast_path=body[55]/body[2]/body[0]/test/values[3]
      rationale: The isinstance checks are discriminating between AST node variants (ast.Constant vs other expression types in a Subscript slice), which is the canonical and correct use of isinstance for sum-type/variant discrimination over Python's ast module. This is not a defensive guard against malformed data — it's structural pattern matching on a typed AST to extract literal string field references from user-authored gate conditions. The R5 rule targets shape-guarding of external data; here the subject is an ast node tree produced by ast.parse, and isinstance is the standard idiom. No fabrication, no trust-tier confusion, no upward import.
    * OLD_FP d8ccc95d7f3fafec | owner=composer-tools-generation | ast_path=body[55]/body[2]/body[0]/test/values[4]
      rationale: The isinstance check is operating on Python AST nodes, which are a polymorphic union type by design — ast.Constant.value can be any literal type (str, int, float, bytes, None, etc.). Discriminating the variant via isinstance is the canonical, type-correct way to walk an AST; this is not a defensive shape-guard against malformed external data but offensive narrowing of a legitimately heterogeneous typed structure. R5's intent (suppressing Tier-3 boundary shape guards) does not apply to AST traversal of a parsed expression. The decorator nudge does not apply: the parameter is a condition string, not external row data, and the isinstance is on AST nodes derived from parsing, not on the parameter itself.
    * OLD_FP 3cf8de969f916272 | owner=composer-tools-generation | ast_path=body[55]/body[2]/body[1]/test/values[0]
      rationale: The isinstance checks here are AST node-type discrimination during ast.walk traversal, not defensive shape-guarding of external data. Selecting ast.Call vs ast.Subscript nodes from a parsed AST is exactly the offensive, structural pattern Python's ast module is designed for — there is no Tier-3 data flowing through these checks (the AST nodes are produced by ast.parse on a config-supplied condition string, and the discrimination is on node identity, not value coercion). R5's defensive-isinstance prohibition does not apply to AST visitor dispatch.
    * OLD_FP 788462ae239e2501 | owner=composer-tools-generation | ast_path=body[55]/body[2]/body[1]/test/values[1]
      rationale: The isinstance checks here are AST node-type discrimination on a parsed Python expression tree, not defensive programming on data values. Walking an ast tree and using isinstance to distinguish ast.Call/ast.Attribute/ast.Name/ast.Constant nodes is the canonical, type-correct way to use the ast module — these are sum-type tags, not trust-tier guards. R5's 'shape guard at boundary' framing doesn't apply: there is no Tier-3 data being shape-checked, only structural pattern-matching on a heterogeneous AST. The rationale accurately describes the code.
    * OLD_FP 8218c1e27d444b5d | owner=composer-tools-generation | ast_path=body[55]/body[2]/body[1]/test/values[3]
      rationale: The isinstance check is being used for AST node-shape discrimination during structural pattern matching of a parsed Python expression — this is the canonical legitimate use of isinstance (type-narrowing on a tagged-union AST), not defensive programming on data values. The R5 rule targets shape-guarding of external Tier-3 data; here the subject is an ``ast`` node from ``ast.parse``, and the isinstance is required to access ``.id`` safely on the union of possible ``func.value`` node types. There is no fabrication, no silent default, no trust-tier confusion — removing the check would either miscategorise expressions or crash on legitimate AST shapes.
    * OLD_FP 2780ab37f5f2d489 | owner=composer-tools-generation | ast_path=body[55]/body[2]/body[1]/test/values[6]
      rationale: The isinstance check is a legitimate AST node-shape discriminator, not a defensive guard on Tier-2/3 data. ast.walk yields heterogeneous node types and ast.Call.args is a list of arbitrary expression nodes; filtering for ast.Constant with a str value is the standard way to distinguish row.get('literal') from row.get(variable). This is offensive programming — narrowing to the exact AST shape the code can handle — and matches the policy's allowance for isinstance as type-narrowing on union types rather than as a trust-boundary shape guard on external data. Rule R5 is in the decorator-suppressible set, but the subject is an AST node from ast.parse on a project-controlled condition string, not an external parameter, so the decorator nudge does not apply.
    * OLD_FP 9481bb581f0a388f | owner=composer-tools-generation | ast_path=body[55]/body[2]/body[1]/test/values[7]
      rationale: The isinstance check is genuine AST node-shape discrimination, not a Tier-3 data shape guard. The code is walking a Python AST produced by ast.parse on a condition string, and ast.Constant.value can legitimately be any of str/int/float/bytes/None — narrowing to str is type discrimination on a sum-type union, which is the correct offensive use of isinstance. R5 (defensive shape guard at boundary) does not apply here; this is structural pattern matching against the AST grammar.

--- compute_proof_diagnostics | R5 | 2 entries | AMBIGUOUS (no 1:1 old->new mapping) ---
  NEW fingerprint(s) to re-sign against (2):
    - 5a62b94ef7c94b60   (live L1263: isinstance() used: if isinstance(schema, Mapping) and schema.get("mode)
    - dac3e3c207c20346   (live L1265: isinstance() used: if isinstance(declared, (list, tuple)):)
  OLD entries (rationale + owner to carry into re-justify):
    * OLD_FP 2e46bb891c34ebf7 | owner=composer-tools-generation | ast_path=body[61]/body[14]/body[1]/test/values[0]
      rationale: The excerpt shows ``schema = source.options.get("schema")`` where source.options is Tier-1 but the value at "schema" is LLM-authored unstructured config — genuinely Tier-3 at this nested level. The R5 isinstance(schema, Mapping) check is a legitimate shape probe at the boundary between trusted config keys and untrusted nested values. The rationale correctly identifies the narrow scope and explains the diagnostic's purpose. The decorator nudge doesn't apply cleanly here: the external data isn't a function parameter but a nested value pulled from a Tier-1 mapping, so @trust_boundary's source_param mechanism doesn't fit.
    * OLD_FP 69025a61f1c28d9f | owner=composer-tools-generation | ast_path=body[61]/body[14]/body[1]/body[1]/test
      rationale: The excerpt shows schema is read from source.options at an LLM-authored config boundary; the comment at lines 1213-1215 explicitly notes that the value at 'schema' is unstructured. The isinstance(declared, (list, tuple)) check is a Tier-3 shape probe on an LLM-authored value before iterating it as a sequence — exactly the boundary-validation pattern policy permits. R5 is in the decorator's suppressible set, but the function compute_proof_diagnostics does not take the external data as a direct parameter (it's reached via source.options on a parameter whose origin isn't visible), and the excerpt does not show the function signature, so a decorator nudge would be speculative. Accepting the per-line suppression on the visible evidence.

--- _gate_expression_type_diagnostics_for_observed_csv | R6 | 1 entry ---
  NEW fingerprint(s) to re-sign against (1):
    - dbe640b4339f7059   (live L981: Exception swallowed without re-raise or explicit error: except Express)
  OLD entries (rationale + owner to carry into re-justify):
    * OLD_FP 1a4634d0d4e540a1 | owner=composer-tools-generation | ast_path=body[56]/body[6]/body[6]/body[0]/handlers[0]
      rationale: The except clause catches a specific, narrowly-typed ExpressionEvaluationError raised when an LLM-authored gate condition is evaluated against a Tier-3 sampled CSV row. The exception is not silently swallowed: it is converted into a structured _blocking_diagnostic that records the original error text, node id, field, and sample row index — this is exactly the offensive-programming pattern of converting a runtime failure at a trust boundary into an informative audit record. The expression and the row are both external/untrusted (LLM-authored expression vs. observed-mode CSV with no declared types), so a typed-exception capture at this boundary is appropriate. R6 (broad-ish exception handling) is not in the decorator's suppressible set {R1, R5}, so the decorator nudge does not apply.


TOTAL displaced signed: 25
