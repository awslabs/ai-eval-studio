// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @vitest-environment jsdom
 *
 * Report tests.
 *
 * Scoped to properties that have been wrong before, or that would fail
 * silently if they regressed:
 *
 * 1. "Checked and clean" and "not checked" must not render the same. The audit
 *    can fail to write, and no news reads as good news — the exact ambiguity the
 *    metric audit itself lived in. Carried by an explicit ``audit_available``
 *    flag rather than by the verdict being absent.
 * 2. The trust summary at the top is ONE line however many reasons exist. A
 *    variable-height block there pushes the result down, and real runs produce
 *    four reasons.
 * 3. The run report states its result from the FACTS. It used to lead with the
 *    narrator's prose and fall back to the computed line only if the narrator
 *    produced nothing, so vague narration left no reliable statement.
 * 4. Percentages truncate. Rounding showed "90%" for a 89.6% run beside a cell
 *    coloured red for missing its 0.90 target.
 * 5. Section ORDER: "What changed this run" sits above the failure sections,
 *    matching the executive report. It was the one thing telling the two
 *    reports' stories in opposite orders.
 * 6. The executive Recommendation is tinted and caveated from the same verdict,
 *    rather than staying confident green with the caveat five sections below.
 */

import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

import { DataQualitySection, TrustLine, worstTrustReason } from "./DataQuality";
import { FailureBucketsSection } from "./FailureBuckets";
import { RunReport } from "./RunReport";
import { ExecReport } from "./ExecReport";
import { fmtPct, fmtPctPrecise } from "@/lib/reports/format";
import type {
  ExecFacts,
  ModelFacts,
  RunFacts,
  RunTrustFacts,
} from "@/lib/reports/types";

afterEach(() => {
  cleanup();
});

/* ----------------------------- fixtures ----------------------------- */

function model(over: Partial<ModelFacts> = {}): ModelFacts {
  return {
    model_id: "vendor.model-a",
    display_name: "model-a",
    total: 208,
    passed: 112,
    pass_rate: 53.85,
    metric_rates: { Judge: 53.85 },
    avg_latency_ms: 800,
    latency_percentiles: {
      median: 800, p95: 1200, p99: 1500, max: 1800,
      over_budget_count: null, total: 208,
    },
    generation_cost_usd: 1,
    judge_cost_usd: 0,
    total_cost_usd: 1,
    cost_per_1k_gen: 0.42,
    judge_cost_available: false,
    pricing_available: true,
    avg_input_tokens: 500,
    avg_output_tokens: 100,
    cache_read_ratio: 0,
    pass_rate_delta: null,
    metric_rate_deltas: {},
    ...over,
  };
}

function runFacts(over: Partial<RunFacts> = {}): RunFacts {
  return {
    schema_version: 1,
    project_name: "Proj",
    project_description: "",
    run_filename: "eval_results_v1.json",
    run_timestamp: "2026-09-04T00:00:00Z",
    run_source: "optimization",
    prompt_label: "v1",
    prompt_iteration: 1,
    prompt_content: "prompt body",
    prompt_changes: "Tightened the schema rule",
    case_count: 208,
    category_count: 0,
    metric_names: ["Judge"],
    metric_thresholds: { Judge: 0.9 },
    metric_criteria: {},
    latency_budget_ms: null,
    has_latency_metric: false,
    hyperparameters: {},
    models: [model()],
    recommendations: [],
    recommended_model_id: "vendor.model-a",
    failed_cases: [],
    failure_buckets: { judge_false_negatives: 0, data_review: 0, model_gaps: 0 },
    run_integrity: {
      case_count: 208, errored: 0, dropped: 0, null_count: 0,
      unrecovered: 0, effective_total: 208, reliability: "ok",
    },
    run_trust: CLEAN,
    prior_step: null,
    is_winning_run: false,
    ...over,
  };
}

function execFacts(over: Partial<ExecFacts> = {}, winnerOver: Partial<RunFacts> = {}): ExecFacts {
  return {
    schema_version: 1,
    project_name: "Proj",
    project_description: "desc",
    case_count: 208,
    category_count: 0,
    metric_names: ["Judge"],
    metric_thresholds: { Judge: 0.9 },
    metric_criteria: {},
    latency_budget_ms: null,
    journey: [],
    winner_version: "v1",
    winner_run_filename: "eval_results_v1.json",
    winner_run: runFacts(winnerOver),
    generated_from_label: "v1 run",
    generated_at: "2026-09-04T00:00:00Z",
    ...over,
  } as ExecFacts;
}

/** Checked, and nothing found — the only state that earns an all-clear. */
const CLEAN: RunTrustFacts = { severity: "ok", reasons: [], audit_available: true };

/** The audit did not run. NOT the same as clean, and the distinction is the point. */
const UNCHECKED: RunTrustFacts = { severity: "ok", reasons: [], audit_available: false };

const FOUR_REASONS: RunTrustFacts = {
  severity: "warning",
  audit_available: true,
  reasons: [
    {
      id: "metric_non_discriminating", severity: "warning", metrics: ["Relevancy"],
      headline: "Relevancy scores 99% of cases the same",
      impact: "It cannot separate a good model from a bad one.",
    },
    {
      id: "gate_rejects_accepted", severity: "info", metrics: ["Judge"],
      headline: "Judge failed 22% of cases that Relevancy passed",
      impact: "One of the two is wrong on those cases.",
    },
    {
      id: "over_strict", severity: "info", metrics: ["Exact Match"],
      headline: "Exact Match failed 8% of cases that Judge passed",
      impact: "Two readings, and only you can tell which.",
    },
    {
      id: "metrics_contradict", severity: "info",
      metrics: ["Exact Match", "Relevancy"],
      headline: "Exact Match and Relevancy disagree on 31% of cases",
      impact: "Neither decides pass/fail.",
    },
  ],
};

const BROKEN_GATE: RunTrustFacts = {
  severity: "error",
  audit_available: true,
  reasons: [{
    id: "gate_saturated", severity: "error", metrics: ["Judge"],
    headline: "Judge decides pass/fail but scores 99% of cases the same",
    impact: "Treat the percentages as unproven.",
  }],
};

/* ------------------------- absence vs all-clear ------------------------- */

describe("an unchecked run is not a clean run", () => {
  it("says the cross-check did not run, rather than nothing", () => {
    render(<DataQualitySection trust={UNCHECKED} caseCount={208} />);
    expect(screen.getByText(/metric cross-check did not run/i)).toBeInTheDocument();
    expect(screen.queryByText(/scored cleanly/i)).not.toBeInTheDocument();
  });

  it("states the gap ALONGSIDE other findings, not instead of them", () => {
    // A run can have dropped cases recorded while the audit still failed to
    // write; suppressing either one would hide a real problem.
    render(
      <DataQualitySection
        trust={{ ...FOUR_REASONS, audit_available: false }}
        caseCount={208}
      />,
    );
    expect(screen.getByText(/metric cross-check did not run/i)).toBeInTheDocument();
    expect(screen.getByText(/Relevancy scores 99% of cases the same/)).toBeInTheDocument();
  });

  it("gives an explicit all-clear when the verdict is clean", () => {
    render(<DataQualitySection trust={CLEAN} caseCount={208} />);
    expect(screen.getByText(/208 cases scored cleanly/i)).toBeInTheDocument();
  });

  it("renders no trust line at the top in either of those states", () => {
    const { container } = render(
      <>
        <TrustLine trust={UNCHECKED} />
        <TrustLine trust={CLEAN} />
      </>,
    );
    expect(container).toBeEmptyDOMElement();
  });
});

/* --------------------------- the one-line rule --------------------------- */

describe("the trust line stays one line", () => {
  it("summarises four reasons as a count plus the worst one", () => {
    render(<TrustLine trust={FOUR_REASONS} />);
    expect(
      screen.getByText(/4 things to check before trusting these numbers/i),
    ).toBeInTheDocument();
    // The worst reason's headline appears; the other three do NOT.
    expect(screen.getByText(/Relevancy scores 99% of cases the same/)).toBeInTheDocument();
    expect(screen.queryByText(/disagree on 31% of cases/)).not.toBeInTheDocument();
  });

  it("uses the singular for one reason", () => {
    render(<TrustLine trust={BROKEN_GATE} />);
    expect(
      screen.getByText(/1 thing to check before trusting these numbers/i),
    ).toBeInTheDocument();
  });

  it("lists every reason in the section, worst first", () => {
    render(<DataQualitySection trust={FOUR_REASONS} caseCount={208} />);
    for (const r of FOUR_REASONS.reasons) {
      expect(screen.getByText(new RegExp(r.headline.slice(0, 24)))).toBeInTheDocument();
    }
  });
});

describe("worstTrustReason", () => {
  it("takes the head rather than re-ranking", () => {
    expect(worstTrustReason(FOUR_REASONS)?.id).toBe("metric_non_discriminating");
  });

  it("is null when there is nothing to report", () => {
    expect(worstTrustReason(CLEAN)).toBeNull();
    expect(worstTrustReason(UNCHECKED)).toBeNull();
  });
});

/* --------------------------- failure buckets --------------------------- */

describe("failure buckets", () => {
  it("renders nothing when no case failed on every model", () => {
    const { container } = render(
      <FailureBucketsSection
        buckets={{ judge_false_negatives: 0, data_review: 0, model_gaps: 0 }}
      />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("hides zero buckets and totals only the ones shown", () => {
    render(
      <FailureBucketsSection
        buckets={{ judge_false_negatives: 0, data_review: 2, model_gaps: 5 }}
      />,
    );
    expect(screen.getByText("7 cases no model passed, broken down by likely cause.")).toBeInTheDocument();
    expect(screen.getByText("Data review needed")).toBeInTheDocument();
    expect(screen.getByText("Model gaps")).toBeInTheDocument();
    expect(screen.queryByText("Likely scoring errors")).not.toBeInTheDocument();
  });

  it("uses the singular for one case", () => {
    render(
      <FailureBucketsSection
        buckets={{ judge_false_negatives: 1, data_review: 0, model_gaps: 0 }}
      />,
    );
    expect(screen.getByText(/^1 case no model passed/)).toBeInTheDocument();
  });
});

/* ----------------------------- truncation ----------------------------- */

describe("percentages truncate rather than round", () => {
  it("never claims a target was met that was not", () => {
    // 89.6 against a 0.90 target: rounding printed "90%" beside a cell this
    // same table colours RED for missing the target.
    expect(fmtPct(89.6)).toBe("89%");
    expect(fmtPctPrecise(89.96)).toBe("89.9%");
  });

  it("reserves 100% for a genuine 100", () => {
    expect(fmtPct(99.52)).toBe("99%");
    expect(fmtPct(100)).toBe("100%");
  });

  it("matches the Evals tab on the real value that disagreed", () => {
    expect(fmtPct(53.85)).toBe("53%");
  });
});

/* ------------------------- run report behaviour ------------------------- */

describe("the run report states its result from the facts", () => {
  it("names the model and rate with no narrative at all", () => {
    render(<RunReport facts={runFacts()} narrative={null} />);
    expect(screen.getByText("model-a reached 53% pass rate.")).toBeInTheDocument();
  });

  it("keeps the computed result even when the narrator says something else", () => {
    render(
      <RunReport
        facts={runFacts()}
        narrative={{ takeaway: "Results were mixed this run." } as never}
      />,
    );
    expect(screen.getByText("model-a reached 53% pass rate.")).toBeInTheDocument();
    expect(screen.getByText("Results were mixed this run.")).toBeInTheDocument();
  });

  it("puts 'What changed this run' ABOVE the failure sections", () => {
    render(
      <RunReport
        facts={runFacts({
          failure_buckets: { judge_false_negatives: 3, data_review: 0, model_gaps: 0 },
        })}
        narrative={null}
      />,
    );
    const headings = screen.getAllByRole("heading", { level: 2 }).map((h) => h.textContent);
    const changed = headings.indexOf("What changed this run");
    const blocking = headings.indexOf("What's blocking");
    expect(changed).toBeGreaterThanOrEqual(0);
    expect(blocking).toBeGreaterThanOrEqual(0);
    expect(changed).toBeLessThan(blocking);
  });

  it("shows the trust line under the takeaway and the detail in Data quality", () => {
    render(<RunReport facts={runFacts({ run_trust: BROKEN_GATE })} narrative={null} />);
    expect(screen.getByText(/1 thing to check/i)).toBeInTheDocument();
    // The impact text now appears in TWO places by design — beside the metric's
    // definition and in Data quality — so this asserts the Data quality copy
    // specifically rather than uniqueness, which was only ever a proxy for it.
    const section = screen.getByRole("heading", { name: "Data quality" })
      .closest("section");
    expect(section).not.toBeNull();
    expect(section!.textContent).toMatch(/Treat the percentages as unproven/);
  });
});

/* ------------------------ exec report behaviour ------------------------ */

describe("the executive recommendation carries its own confidence", () => {
  it("adds a caveat naming the worst reason when trust is compromised", () => {
    render(
      <ExecReport facts={execFacts({}, { run_trust: BROKEN_GATE })} narrative={null} />,
    );
    expect(screen.getByText(/Before acting on this:/)).toBeInTheDocument();
    expect(
      screen.getAllByText(/scores 99% of cases the same/).length,
    ).toBeGreaterThan(0);
  });

  it("stays uncaveated when the run is clean", () => {
    render(<ExecReport facts={execFacts({}, { run_trust: CLEAN })} narrative={null} />);
    expect(screen.queryByText(/Before acting on this:/)).not.toBeInTheDocument();
    expect(screen.getByText(/208 cases scored cleanly/i)).toBeInTheDocument();
  });

  it("puts the confidence clause in the Summary too", () => {
    render(
      <ExecReport facts={execFacts({}, { run_trust: FOUR_REASONS })} narrative={null} />,
    );
    expect(screen.getByText(/see Data quality/)).toBeInTheDocument();
  });
});

/* ------------------- per-metric marking (the audit's reader) ------------------- */

describe("a disputed metric is marked where its numbers are read", () => {
  /** Two metrics, one of which the verdict names. */
  const TWO_METRICS = {
    metric_names: ["Judge", "Exact Match"],
    metric_thresholds: { Judge: 0.9, "Exact Match": 1.0 },
  };

  /** The trust verdict names ONLY "Judge". */
  const JUDGE_DISPUTED: RunTrustFacts = {
    severity: "error",
    audit_available: true,
    reasons: [{
      id: "gate_saturated", severity: "error", metrics: ["Judge"],
      headline: "Judge decides pass/fail but scores 99% of cases the same",
      impact: "Treat the percentages as unproven.",
    }],
  };

  function definitionsSection(): HTMLElement {
    const section = screen.getByRole("heading", { name: "What was tested" })
      .closest("section");
    expect(section).not.toBeNull();
    return section as HTMLElement;
  }

  it("states the reason beside the metric's own definition", () => {
    render(
      <RunReport
        facts={runFacts({ ...TWO_METRICS, run_trust: JUDGE_DISPUTED })}
        narrative={null}
      />,
    );
    // In words, not just an icon: a printed report has no hover, and this is
    // the reader who has just been told what the metric measures.
    expect(definitionsSection().textContent).toMatch(
      /Judge decides pass\/fail but scores 99% of cases the same/,
    );
  });

  it("does not mark a metric the verdict never named", () => {
    render(
      <RunReport
        facts={runFacts({ ...TWO_METRICS, run_trust: JUDGE_DISPUTED })}
        narrative={null}
      />,
    );
    // The discrimination: marking every metric would be the same as marking
    // none. "Exact Match" is in the list; nothing about it is flagged.
    const text = definitionsSection().textContent ?? "";
    expect(text).toMatch(/Exact Match/);
    expect(text.match(/Treat the percentages as unproven/g) ?? []).toHaveLength(1);
  });

  it("marks the standings column, with the reason as its hover text", () => {
    render(
      <RunReport
        facts={runFacts({ ...TWO_METRICS, run_trust: JUDGE_DISPUTED })}
        narrative={null}
      />,
    );
    const judgeHeader = screen
      .getAllByRole("columnheader")
      .find((th) => th.textContent?.includes("Judge"));
    expect(judgeHeader).toBeDefined();
    expect(judgeHeader!.querySelector("[title]")?.getAttribute("title")).toMatch(
      /scores 99% of cases the same/,
    );

    const exactHeader = screen
      .getAllByRole("columnheader")
      .find((th) => th.textContent?.includes("Exact Match"));
    expect(exactHeader).toBeDefined();
    expect(exactHeader!.querySelector("[title]")).toBeNull();
  });

  it("marks the same metric in the executive report", () => {
    render(
      <ExecReport
        facts={execFacts(TWO_METRICS, { ...TWO_METRICS, run_trust: JUDGE_DISPUTED })}
        narrative={null}
      />,
    );
    // Both reports read one index. A marker in one and not the other is how the
    // two report renderers drifted before they shared components.
    expect(definitionsSection().textContent).toMatch(
      /Judge decides pass\/fail but scores 99% of cases the same/,
    );
    const judgeHeader = screen
      .getAllByRole("columnheader")
      .find((th) => th.textContent?.includes("Judge"));
    expect(judgeHeader?.querySelector("[title]")).not.toBeNull();
  });

  it("marks BOTH metrics when two diagnostics contradict each other", () => {
    // Neither can be singled out, so naming one would tell the reader the other
    // column is sound. This is why a reason carries a LIST of metrics.
    const contradiction: RunTrustFacts = {
      severity: "info",
      audit_available: true,
      reasons: [{
        id: "metrics_contradict", severity: "info",
        metrics: ["Judge", "Exact Match"],
        headline: "Judge and Exact Match disagree on 31% of cases",
        impact: "Neither decides pass/fail, so your headline numbers hold.",
      }],
    };
    render(
      <RunReport
        facts={runFacts({ ...TWO_METRICS, run_trust: contradiction })}
        narrative={null}
      />,
    );
    const marked = screen
      .getAllByRole("columnheader")
      .filter((th) => th.querySelector("[title]"))
      .map((th) => th.textContent ?? "");
    expect(marked).toHaveLength(2);
  });

  it("marks a metric with its WORST reason when several name it", () => {
    // The real shape: a saturated gate produces an error reason AND the
    // comparator disagreements that escalated it. The backend orders reasons
    // worst-first; a renderer that kept the last one would show an info-level
    // note for an error-level defect.
    const twoReasons: RunTrustFacts = {
      severity: "error",
      audit_available: true,
      reasons: [
        {
          id: "gate_saturated", severity: "error", metrics: ["Judge"],
          headline: "Judge decides pass/fail but scores 99% of cases the same",
          impact: "Treat the percentages as unproven.",
        },
        {
          id: "over_strict", severity: "info", metrics: ["Judge"],
          headline: "Judge failed 8% of cases that Exact Match passed",
          impact: "Read a few of these cases.",
        },
      ],
    };
    render(
      <RunReport
        facts={runFacts({ ...TWO_METRICS, run_trust: twoReasons })}
        narrative={null}
      />,
    );
    const text = definitionsSection().textContent ?? "";
    expect(text).toMatch(/Judge decides pass\/fail but scores 99%/);
    expect(text).not.toMatch(/failed 8% of cases that Exact Match passed/);
  });

  it("marks nothing on a clean run", () => {
    render(
      <RunReport facts={runFacts({ ...TWO_METRICS, run_trust: CLEAN })} narrative={null} />,
    );
    expect(
      screen.getAllByRole("columnheader").filter((th) => th.querySelector("[title]")),
    ).toHaveLength(0);
  });
});
