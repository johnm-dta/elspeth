#!/usr/bin/env node
import { readFileSync, readdirSync, writeFileSync, appendFileSync, mkdirSync } from "node:fs";
import { execSync } from "node:child_process";

const batchId = process.argv[2];
if (!batchId) { console.error("usage: aggregate.mjs <batch_id>"); process.exit(1); }
const dir = `tests/e2e/.harness-results/${batchId}`;
const records = readdirSync(dir).filter((f) => /^run-\d+\.json$/.test(f)).map((f) => JSON.parse(readFileSync(`${dir}/${f}`, "utf-8")));

const n = records.length;
const passes = records.filter((r) => r.outcome === "pass");
const infra = records.filter((r) => r.outcome === "infra_fault");
const tutorialDenom = n - infra.length; // exclude infra noise from the rate we drive to 100%
const tutorialPassRate = tutorialDenom ? passes.length / tutorialDenom : 0;
const infraRate = n ? infra.length / n : 0;

const gitSha = execSync("git rev-parse --short HEAD").toString().trim();
const harnessVersion = "1.0.0";
const skillHash = records.find((r) => r.stamp.composer_skill_hash)?.stamp.composer_skill_hash ?? "unknown";
const modelId = records.find((r) => r.stamp.model_identifier)?.stamp.model_identifier ?? "unknown";

// failure table
const fails = records.filter((r) => r.outcome !== "pass");
const tableRows = fails.map((r) => `| ${r.run_index} | ${r.outcome} | ${r.fault_subclass ?? ""} | ${r.fix_target ?? ""} | turn ${r.turn_reached} | ${r.landscape.realsystem_failure ?? r.error ?? ""} |`).join("\n");

const report = `# Tutorial reliability batch \`${batchId}\`

- git: \`${gitSha}\` · harness: \`${harnessVersion}\` · model: \`${modelId}\` · skill_hash: \`${skillHash.slice(0,12)}\`
- runs: ${n} · **tutorial-pass-rate: ${passes.length}/${tutorialDenom} (${(tutorialPassRate*100).toFixed(0)}%)** · infra-noise: ${infra.length}/${n} (${(infraRate*100).toFixed(0)}%)
- dim pass counts — a:${records.filter(r=>r.dim_a_tutorial_completed).length} b:${records.filter(r=>r.dim_b_realsystem_passed).length} c:${records.filter(r=>r.dim_c_assumptions_ok).length} d:${records.filter(r=>r.dim_d_solution_quality.status==="pass").length}(judged)

## Failures
| run | outcome | subclass | fix-target | reached | detail |
|-----|---------|----------|-----------|---------|--------|
${tableRows || "_none_"}
`;

mkdirSync("../../../../notes/tutorial-reliability", { recursive: true });
writeFileSync(`../../../../notes/tutorial-reliability/${batchId}.md`, report);
const trend = { batch_id: batchId, git: gitSha, harness: harnessVersion, model: modelId, skill_hash: skillHash, n, tutorial_pass: passes.length, tutorial_denom: tutorialDenom, infra: infra.length };
appendFileSync("../../../../notes/tutorial-reliability/trend.jsonl", JSON.stringify(trend) + "\n");
console.log(report);
