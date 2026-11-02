// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @vitest-environment jsdom
 *
 * Replay-marking behavioral spec for useBackendEvents.
 *
 * Verifies that replayed envelopes suppress toasts but still deliver to state
 * consumers, and that momentary consumers can opt out of replays.
 */
import { useCallback } from "react";
import { render, cleanup, waitFor, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { toast } from "sonner";

// Mock the stream hook to control envelope delivery
let capturedProjectStreamHandler: ((envelope: any) => void) | null = null;

vi.mock("@/hooks/useWsStream", () => ({
  useProjectStream: (_projectId: string, handler: (env: any) => void) => {
    capturedProjectStreamHandler = handler;
  },
}));

// Mock sonner toast
vi.mock("sonner", () => ({
  toast: {
    error: vi.fn(),
    info: vi.fn(),
    success: vi.fn(),
    warning: vi.fn(),
  },
}));

// Import after mocks
const { useBackendEvents, useProjectEvent } = await import("@/hooks/useBackendEvents");

/**
 * Test harness that mounts useBackendEvents and a test consumer.
 *
 * Both hooks are called unconditionally. A test that supplies no
 * ``onEvent`` still subscribes, with a no-op handler — gating the
 * subscription behind a prop would make the hook order depend on
 * props, which is a rules-of-hooks violation even when the props
 * never change during a given mount.
 */
function Harness({
  projectId,
  onEvent,
  consumer = "state",
}: {
  projectId: string;
  onEvent?: (payload: any) => void;
  consumer?: "momentary" | "state";
}) {
  useBackendEvents(projectId);
  const handler = useCallback(
    (payload: any) => {
      onEvent?.(payload);
    },
    [onEvent],
  );
  useProjectEvent({ resource: "test", type: "event" }, handler, consumer);
  return <div>Test</div>;
}

function fireProjectEvent(projectId: string, eventData: any, replayed?: boolean) {
  if (!capturedProjectStreamHandler) {
    throw new Error("Handler not captured");
  }
  const envelope: any = {
    stream: `project:${projectId}`,
    seq: Date.now(),
    kind: "project_event",
    data: eventData,
  };
  if (replayed) {
    envelope.replayed = true;
  }
  act(() => {
    capturedProjectStreamHandler!(envelope);
  });
}

describe("useBackendEvents replay handling", () => {
  beforeEach(() => {
    capturedProjectStreamHandler = null;
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  it("Test 1: replayed error payload fires no toast, live error fires toast", async () => {
    const projectId = "proj-toast-1";

    render(<Harness projectId={projectId} />);

    // Wait for mount
    await waitFor(() => expect(capturedProjectStreamHandler).toBeTruthy());

    // Fire a replayed error
    fireProjectEvent(
      projectId,
      {
        resource: "eval_job",
        type: "error",
        code: "ERR_TEST",
        message: "Replayed error",
      },
      true // replayed
    );

    // Wait a beat
    await new Promise((resolve) => setTimeout(resolve, 50));

    // Assert: no toast fired
    expect(toast.error).not.toHaveBeenCalled();

    // Fire a LIVE error
    fireProjectEvent(projectId, {
      resource: "eval_job",
      type: "error",
      code: "ERR_LIVE",
      message: "Live error",
    });

    // Wait for toast
    await waitFor(() => expect(toast.error).toHaveBeenCalledTimes(1));
  });

  it("job executor {type: 'failed'} payloads surface a toast", async () => {
    /* The eval/harness executors publish terminal failures as
     * ``{type: "failed", ...ResourceError}`` — a distinct type from the
     * "error" most publishers use. No panel subscribes to job/failed, so
     * the toast layer is the ONLY surface; an earlier filter matched
     * "error" alone and a failed harness build was silent. */
    const projectId = "proj-failed-toast";
    render(<Harness projectId={projectId} />);
    await waitFor(() => expect(capturedProjectStreamHandler).toBeTruthy());

    fireProjectEvent(projectId, {
      resource: "job",
      type: "failed",
      job_type: "harness_build",
      code: "HARNESS_BUILD_FAILED",
      message: "Scaffold validation failed",
    });
    await waitFor(() => expect(toast.error).toHaveBeenCalledTimes(1));
  });

  it("Test 2: replayed payload delivered to state consumer", async () => {
    const projectId = "proj-state-2";
    const onEvent = vi.fn();

    render(<Harness projectId={projectId} onEvent={onEvent} consumer="state" />);

    await waitFor(() => expect(capturedProjectStreamHandler).toBeTruthy());

    // Fire a replayed event
    fireProjectEvent(
      projectId,
      {
        resource: "test",
        type: "event",
        data: "replayed-payload",
      },
      true
    );

    // Assert: handler was called (state consumers receive replays)
    await waitFor(() => expect(onEvent).toHaveBeenCalledTimes(1));
  });

  it("Test 3: replayed payload NOT delivered to momentary consumer", async () => {
    const projectId = "proj-momentary-3";
    const onEvent = vi.fn();

    render(<Harness projectId={projectId} onEvent={onEvent} consumer="momentary" />);

    await waitFor(() => expect(capturedProjectStreamHandler).toBeTruthy());

    // Fire a replayed event
    fireProjectEvent(
      projectId,
      {
        resource: "test",
        type: "event",
        data: "replayed-payload",
      },
      true
    );

    // Wait a beat
    await new Promise((resolve) => setTimeout(resolve, 50));

    // Assert: handler was NOT called (momentary consumers ignore replays)
    expect(onEvent).not.toHaveBeenCalled();
  });
});
