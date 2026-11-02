// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @vitest-environment jsdom
 *
 * The eval Stop's failure handling in `useEvalJobStream`.
 *
 * The backend answers a cancel it could not confirm with a 503 and leaves the
 * eval running on purpose. The hook used to swallow that, so "Stopping..." went
 * back to "Stop" with no word of why while the eval carried on — the same
 * refusal the optimization Stop shows as a toast.
 */
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ProjectActivityEval } from "@/lib/types";

const { toastError, cancelEvalJob } = vi.hoisted(() => ({
  toastError: vi.fn(),
  cancelEvalJob: vi.fn(),
}));

vi.mock("sonner", () => ({ toast: { error: toastError } }));
vi.mock("@/hooks/useWsStream", () => ({ useJobStream: () => {} }));
vi.mock("@/lib/api", () => ({ api: { cancelEvalJob } }));

import { useEvalJobStream } from "./useEvalJobStream";

const runningEval: ProjectActivityEval = {
  job_id: "job-1",
  status: "running",
  models_total: 2,
  progress: {
    current_model: "model-a",
    models_completed: 0,
    models_total: 2,
    current_case: 3,
    cases_total: 10,
    message: "Scoring: model-a (3/10)",
  },
};

/**
 * Mount with no eval, then deliver one — the order it happens in the app, where
 * the eval arrives from the provider's first ``/activity`` poll after mount.
 */
function mountWithRunningEval() {
  const hook = renderHook(
    ({ activeEval }: { activeEval: ProjectActivityEval | null }) =>
      useEvalJobStream("proj-1", activeEval),
    { initialProps: { activeEval: null as ProjectActivityEval | null } },
  );
  hook.rerender({ activeEval: runningEval });
  return hook;
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("eval Stop", () => {
  it("shows the refusal when the backend could not stop the eval", async () => {
    cancelEvalJob.mockRejectedValue(
      new Error("Could not stop the eval worker. The eval is still running — retry the cancel."),
    );
    const { result } = mountWithRunningEval();
    await waitFor(() => expect(result.current.jobId).toBe("job-1"));

    act(() => result.current.handleCancel());

    await waitFor(() =>
      expect(toastError).toHaveBeenCalledWith(
        "Could not stop the eval worker. The eval is still running — retry the cancel.",
      ),
    );
    await waitFor(() => expect(result.current.cancelling).toBe(false));
  });

  it("stays quiet when the cancel succeeds", async () => {
    cancelEvalJob.mockResolvedValue({ status: "cancelled", job_id: "job-1" });
    const { result } = mountWithRunningEval();
    await waitFor(() => expect(result.current.jobId).toBe("job-1"));

    act(() => result.current.handleCancel());

    await waitFor(() => expect(result.current.cancelling).toBe(false));
    expect(cancelEvalJob).toHaveBeenCalledWith("proj-1", "job-1");
    expect(toastError).not.toHaveBeenCalled();
  });
});
