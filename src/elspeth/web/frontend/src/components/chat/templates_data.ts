/**
 * Audit-domain exemplars sourced from README.md lines 560-571; the empty-state chat consumes this.
 * Update discipline — if README.md's table changes, update this file and
 * the snapshot in `templates_data.test.ts` in the same PR.
 */

export type RecommendedStartingPoint =
  | "dynamic_source_from_chat"
  | "csv_upload"
  | "api_source";

/**
 * Exhaustive list of starting points the templates know about.  Consumers
 * can iterate this constant in a `switch (sp)` and add a final default
 * `assertNever(sp)` arm so adding a new starting-point literal becomes
 * a compile error at every consumption site rather than a silent miss.
 * Update discipline: when extending `RecommendedStartingPoint`, extend
 * this array in the same edit — TypeScript's `satisfies` check below
 * fails if any union member is omitted.
 */
export const RECOMMENDED_STARTING_POINTS = [
  "dynamic_source_from_chat",
  "csv_upload",
  "api_source",
] as const satisfies ReadonlyArray<RecommendedStartingPoint>;

export interface ExampleUseCase {
  id: string;
  domain: string;
  description: string;
  sense: string;
  decide: string;
  act: string;
  seed_prompt: string;
  recommended_starting_point: RecommendedStartingPoint;
  icon: string;
}

export const TEMPLATES: ReadonlyArray<ExampleUseCase> = [
  {
    id: "tender-evaluation",
    domain: "Tender Evaluation",
    description: "Score procurement submissions; flag responses that need human review.",
    sense: "CSV of submissions",
    decide: "LLM classification + safety gates",
    act: "Results CSV, abuse review queue",
    seed_prompt:
      "I want to evaluate three tender submissions. Each has a vendor name, a price, and a 200-word capability statement. Use an LLM to score each on capability fit, and route anything mentioning offensive language to a review queue.",
    recommended_starting_point: "dynamic_source_from_chat",
    icon: "\u{1F4DD}",
  },
  {
    id: "document-qa",
    domain: "Document QA",
    description: "Extract answers from documents; escalate exceptions for human review.",
    sense: "PDF/text blobs",
    decide: "LLM extraction, rubric checks, statistical summaries",
    act: "Annotated outputs, exception queue",
    seed_prompt:
      "I have five short policy paragraphs. For each one, use an LLM to extract the key compliance obligation and check whether a deadline is mentioned. Output annotated results and flag any paragraph where the deadline is ambiguous.",
    recommended_starting_point: "csv_upload",
    icon: "\u{1F4C4}",
  },
  {
    id: "weather-monitoring",
    domain: "Weather Monitoring",
    description: "Classify sensor readings; escalate anomalies to alert channels.",
    sense: "Sensor API feed",
    decide: "Threshold + ML anomaly detection",
    act: "Routine log, warning, emergency alert",
    seed_prompt:
      "I have five temperature readings from a sensor: 22, 24, 21, 38, 23 degrees Celsius. Route values above 35 to an emergency alert sink and values between 30-35 to a warning sink. Log everything else as routine.",
    recommended_starting_point: "dynamic_source_from_chat",
    icon: "\u{1F324}",
  },
  {
    id: "satellite-operations",
    domain: "Satellite Operations",
    description: "Classify telemetry events; open investigation tickets for anomalies.",
    sense: "Telemetry stream",
    decide: "Anomaly classifier",
    act: "Routine log, investigation ticket",
    seed_prompt:
      "I have four satellite telemetry events: battery voltage 12.1V (nominal), attitude deviation 0.3 deg (nominal), solar panel output 47W (low), reaction wheel speed 8500 RPM (nominal). Classify each as routine or anomaly; open an investigation ticket for anomalies.",
    recommended_starting_point: "dynamic_source_from_chat",
    icon: "\u{1F6F0}",
  },
  {
    id: "financial-compliance",
    domain: "Financial Compliance",
    description: "Screen transactions for fraud; block high-risk and flag suspicious ones.",
    sense: "Transaction feed",
    decide: "Rules engine + ML fraud detection",
    act: "Approved, flagged, blocked",
    seed_prompt:
      "I have five financial transactions: a $20 coffee purchase, a $5,000 wire transfer to an overseas account at 3 AM, a $150 grocery shop, a $12,000 cash deposit, and a $45 streaming subscription. Apply a rules engine to approve routine transactions, flag suspicious ones, and block high-risk ones.",
    recommended_starting_point: "dynamic_source_from_chat",
    icon: "\u{1F4B3}",
  },
  {
    id: "content-moderation",
    domain: "Content Moderation",
    description: "Classify user submissions; route to publish, human review, or reject.",
    sense: "User submissions",
    decide: "Safety classifier",
    act: "Published, human review, rejected",
    seed_prompt:
      "I have four user-submitted comments: a helpful product review, a comment containing mild profanity, a post with a targeted personal attack, and a spam advertisement. Use a safety classifier to publish clean content, route borderline content to human review, and reject policy violations.",
    recommended_starting_point: "dynamic_source_from_chat",
    icon: "\u{1F6E1}",
  },
] as const;
