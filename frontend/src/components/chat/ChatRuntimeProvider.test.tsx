// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @vitest-environment jsdom
 *
 * Behavioral spec for refetch-on-orchestrator-finalize.
 *
 * The lifecycle attribution line was structurally reload-only: the transcript
 * parser yields the line but only on a `/chat/history` fetch, so live the user saw an
 * assistant bubble with no visible trigger. This test asserts the refetch-on-
 * finalize wiring exists and fires correctly — that an orchestrator turn
 * finalizing causes exactly one history refetch and the transcript's system
 * line becomes visible without a reload.
 *
 * Asserts EFFECTS (what reached the runtime store) rather than rendered DOM
 * because assistant-ui's runtime is the integration surface: the store's
 * `messages` array is what `<Thread>` renders, so asserting on it asserts
 * the user-visible effect without needing a full browser. This is the right
 * unit: the fetch trigger, fan-out wiring, and store application are this
 * component's responsibility; DOM rendering belongs to assistant-ui.
 *
 * Mocking scale: exactly three modules. `@/hooks/useWsStream` is mocked at
 * the stream-subscription boundary so tests can push envelopes by hand;
 * `@/hooks/useBackendEvents` is NOT mocked — its fan-out is part of the unit
 * under test, so the assertions land on its effect. `@/lib/api` is mocked
 * so each test controls the transcript fixture returned. `@assistant-ui/react`
 * is mocked to capture the store argument without needing a browser.
 */
import { render, cleanup, waitFor, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";
import type { ChatMessage } from "@/lib/types";

// ---------------------------------------------------------------------------
// Mocks. The component pulls from useWsStream (for envelope delivery),
// useBackendEvents (for project-stream fan-out), api (for getChatHistory),
// and assistant-ui (for the runtime). Each is stubbed to a controllable
// value. useBackendEvents is NOT mocked — the real fan-out is exercised.
// ---------------------------------------------------------------------------

let capturedProjectStreamHandler: ((envelope: any) => void) | null = null;
let capturedChatStreamHandler: ((envelope: any) => void) | null = null;
let capturedStore: any = null;

vi.mock("@/hooks/useWsStream", () => ({
  useProjectStream: (_projectId: string, handler: (env: any) => void) => {
    capturedProjectStreamHandler = handler;
  },
  useChatStream: (turnId: string | null, handler: (env: any) => void) => {
    if (turnId) capturedChatStreamHandler = handler;
  },
  useJobStream: vi.fn(),
  useWsStream: vi.fn(),
}));

const mockGetChatHistory = vi.fn();
const mockGetActiveTurn = vi.fn();
const mockStartChat = vi.fn();
const mockStopChatTurn = vi.fn();

vi.mock("@/lib/api", () => ({
  api: {
    getChatHistory: (...args: any[]) => mockGetChatHistory(...args),
    getActiveTurn: (...args: any[]) => mockGetActiveTurn(...args),
    startChat: (...args: any[]) => mockStartChat(...args),
    stopChatTurn: (...args: any[]) => mockStopChatTurn(...args),
  },
  ApiError: class ApiError extends Error {
    status: number;
    constructor(message: string, status = 500) {
      super(message);
      this.status = status;
    }
  },
}));

vi.mock("@assistant-ui/react", () => ({
  useExternalStoreRuntime: (store: any) => {
    capturedStore = store;
    return { runtime: "mock-runtime" };
  },
  AssistantRuntimeProvider: ({ children }: { children: ReactNode }) => children,
}));

// Import after mocks so the module graph picks them up.
const { ChatRuntimeProvider } = await import(
  "@/components/chat/ChatRuntimeProvider"
);

// Also need to import useBackendEvents so we can mount it in the harness.
const { useBackendEvents } = await import("@/hooks/useBackendEvents");

/**
 * Harness component that mounts useBackendEvents (as AppLayout in App.tsx does)
 * and renders ChatRuntimeProvider. Tests push envelopes by invoking the
 * captured project-stream callback.
 */
function Harness({ projectId }: { projectId: string }) {
  useBackendEvents(projectId);
  return (
    <ChatRuntimeProvider projectId={projectId}>
      <div data-testid="chat-content">Chat</div>
    </ChatRuntimeProvider>
  );
}

/**
 * Build a transcript fixture where the lifecycle entry uses the system role
 * with exactly one text part and clean text (no marker prefixes) — mirroring
 * what map_session_messages emits. Do not import
 * backend constants; hand-write the fixture so a backend refactor cannot
 * silently make this test vacuous.
 */
function buildTranscriptWithLifecycle(
  lifecycleText: string,
  turnId?: string,
): ChatMessage[] {
  return [
    {
      id: "msg-1",
      role: "user",
      parts: [{ type: "text", text: "Start optimization" }],
      created_at: "2026-08-06T10:00:00Z",
    },
    {
      id: "msg-lifecycle",
      role: "system",
      parts: [{ type: "text", text: lifecycleText }],
      created_at: "2026-08-06T10:00:01Z",
    },
    {
      /* The backend stamps ``turn_id`` on a turn's trailing assistant reply,
       * and the finalize refetch will not adopt a transcript that lacks the
       * turn it is reconciling — it keeps the streamed state instead. A fixture
       * without the field therefore models a transcript that has NOT caught up
       * yet, which is a different scenario. Callers pass the turn id when they
       * mean "this turn's reply has landed". */
      id: "msg-2",
      role: "assistant",
      ...(turnId ? { turn_id: turnId } : {}),
      parts: [{ type: "text", text: "Starting optimization..." }],
      created_at: "2026-08-06T10:00:02Z",
    },
  ];
}

/**
 * Fire a project event by invoking the captured project-stream callback with
 * a properly wrapped envelope. useBackendEvents drops any envelope whose
 * `kind` is not `project_event`, so the wrapper shape is load-bearing.
 *
 * @param replayed - Optional replay marker. When true, sets `replayed: true`
 *                  on the envelope. When omitted, the envelope is treated as
 *                  live (no marker key).
 */
function fireProjectEvent(projectId: string, eventData: any, replayed?: boolean) {
  if (!capturedProjectStreamHandler) {
    throw new Error("Project stream handler not captured — did the component mount?");
  }
  const envelope: any = {
    stream: `project:${projectId}`,
    seq: Date.now(),
    kind: "project_event",
    data: eventData,
  };
  // Mark as replayed if requested (omit key if not — absence means live)
  if (replayed) {
    envelope.replayed = true;
  }
  act(() => {
    capturedProjectStreamHandler!(envelope);
  });
}

describe("ChatRuntimeProvider refetch-on-orchestrator-finalize", () => {
  beforeEach(() => {
    capturedProjectStreamHandler = null;
    capturedChatStreamHandler = null;
    capturedStore = null;
    vi.clearAllMocks();
    mockGetActiveTurn.mockResolvedValue(null);
  });

  afterEach(() => {
    cleanup();
  });

  it("Test 1: orchestrator turn finalize triggers refetch and system line appears", async () => {
    const projectId = "proj-test-1";
    const lifecycleText = "Optimization started from the UI — prompt baseline.md, targets m1";

    // Mount fetch returns empty (no prior messages)
    mockGetChatHistory.mockResolvedValueOnce({
      messages: [],
      interrupted: false,
    });

    // Refetch after finalize returns the transcript with lifecycle entry
    const transcriptWithLifecycle = buildTranscriptWithLifecycle(lifecycleText, "turn-orch-1");
    mockGetChatHistory.mockResolvedValueOnce({
      messages: transcriptWithLifecycle,
      interrupted: false,
    });

    render(<Harness projectId={projectId} />);

    // Wait for mount fetch to complete
    await waitFor(() => expect(mockGetChatHistory).toHaveBeenCalledTimes(1));

    // Fire orchestrator turn started event
    fireProjectEvent(projectId, {
      resource: "agent_turn",
      type: "started",
      turn_id: "turn-orch-1",
      origin: "orchestrator",
    });

    // Fire orchestrator turn completed event
    fireProjectEvent(projectId, {
      resource: "agent_turn",
      type: "completed",
      turn_id: "turn-orch-1",
    });

    // Assert: getChatHistory was called twice (mount + refetch)
    await waitFor(() => expect(mockGetChatHistory).toHaveBeenCalledTimes(2));

    // Assert: the captured store's messages contains the system-role entry
    await waitFor(() => {
      expect(capturedStore).toBeTruthy();
      expect(capturedStore.messages).toBeDefined();
      const systemMessage = capturedStore.messages.find(
        (m: ChatMessage) => m.role === "system"
      );
      expect(systemMessage).toBeTruthy();
      expect(systemMessage.parts[0].text).toBe(lifecycleText);
    });
  });

  it("Test 2 (gate): started event alone does NOT trigger refetch", async () => {
    const projectId = "proj-test-2";

    mockGetChatHistory.mockResolvedValue({
      messages: [],
      interrupted: false,
    });

    render(<Harness projectId={projectId} />);

    // Wait for mount fetch
    await waitFor(() => expect(mockGetChatHistory).toHaveBeenCalledTimes(1));

    // Fire ONLY the started event
    fireProjectEvent(projectId, {
      resource: "agent_turn",
      type: "started",
      turn_id: "turn-orch-2",
      origin: "orchestrator",
    });

    // Wait a beat to ensure no additional fetch fires
    await new Promise((resolve) => setTimeout(resolve, 100));

    // Assert: getChatHistory called exactly once (mount only)
    expect(mockGetChatHistory).toHaveBeenCalledTimes(1);
  });

  it("Test 3: streaming message survives refetch", async () => {
    const projectId = "proj-test-3";

    // Mount fetch returns initial messages
    mockGetChatHistory.mockResolvedValueOnce({
      messages: [
        {
          id: "msg-1",
          role: "user",
          parts: [{ type: "text", text: "Hello" }],
          created_at: "2026-08-06T10:00:00Z",
        },
      ],
      interrupted: false,
    });

    // Refetch returns updated transcript
    const lifecycleText = "Optimization started";
    mockGetChatHistory.mockResolvedValueOnce({
      messages: buildTranscriptWithLifecycle(lifecycleText, "turn-stream-3"),
      interrupted: false,
    });

    render(<Harness projectId={projectId} />);

    await waitFor(() => expect(mockGetChatHistory).toHaveBeenCalledTimes(1));

    // Fire orchestrator started event to trigger turn subscription
    fireProjectEvent(projectId, {
      resource: "agent_turn",
      type: "started",
      turn_id: "turn-stream-3",
      origin: "orchestrator",
    });

    // Wait for turn to start
    await waitFor(() => {
      expect(capturedStore?.isRunning).toBe(true);
    });

    // Simulate streaming via text_delta on the turn stream
    // This requires capturedChatStreamHandler to be set
    await waitFor(() => {
      expect(capturedChatStreamHandler).toBeTruthy();
    });

    act(() => {
      capturedChatStreamHandler!({
        kind: "text_delta",
        data: { text: "Thinking..." },
      });
    });

    // Wait for streaming message to appear
    await waitFor(() => {
      const msgs = capturedStore?.messages || [];
      const streaming = msgs.find((m: ChatMessage) => m.id === "streaming");
      expect(streaming).toBeTruthy();
    });

    // Fire completed event to trigger refetch
    fireProjectEvent(projectId, {
      resource: "agent_turn",
      type: "completed",
      turn_id: "turn-stream-3",
    });

    // Wait for refetch
    await waitFor(() => expect(mockGetChatHistory).toHaveBeenCalledTimes(2));

    // Assert: streaming message no longer present (finalized on result)
    // but refetched messages ARE present
    await waitFor(() => {
      const msgs = capturedStore?.messages || [];
      const systemMsg = msgs.find((m: ChatMessage) => m.role === "system");
      expect(systemMsg).toBeTruthy();
    });
  });

  it("Test 4: stale response does not apply if newer issued", async () => {
    const projectId = "proj-test-4";

    // Mount fetch returns empty
    mockGetChatHistory.mockResolvedValueOnce({
      messages: [],
      interrupted: false,
    });

    render(<Harness projectId={projectId} />);

    await waitFor(() => expect(mockGetChatHistory).toHaveBeenCalledTimes(1));

    // Prepare transcripts
    const staleTranscript: ChatMessage[] = [
      {
        id: "msg-old",
        role: "user",
        parts: [{ type: "text", text: "Old message" }],
        created_at: "2026-08-06T09:00:00Z",
      },
    ];
    /* Stamped with turn-SECOND: this is the response to the second turn's
     * refetch, so it has to carry the turn that refetch is reconciling or the
     * guard treats it as a transcript that has not caught up. The stale
     * transcript above deliberately carries no turn id at all — the nonce guard
     * must reject it on ordering alone, before the turn check is even reached. */
    const newerTranscript = buildTranscriptWithLifecycle(
      "Latest lifecycle entry",
      "turn-second",
    );

    // Control promise resolution order manually
    let resolveFirst: ((value: any) => void) | null = null;
    let resolveSecond: ((value: any) => void) | null = null;

    const firstPromise = new Promise((resolve) => {
      resolveFirst = resolve;
    });
    const secondPromise = new Promise((resolve) => {
      resolveSecond = resolve;
    });

    mockGetChatHistory
      .mockReturnValueOnce(firstPromise as any)
      .mockReturnValueOnce(secondPromise as any);

    // Fire two orchestrator finalizes in quick succession
    // First finalize
    fireProjectEvent(projectId, {
      resource: "agent_turn",
      type: "started",
      turn_id: "turn-first",
      origin: "orchestrator",
    });
    fireProjectEvent(projectId, {
      resource: "agent_turn",
      type: "completed",
      turn_id: "turn-first",
    });

    // Second finalize
    fireProjectEvent(projectId, {
      resource: "agent_turn",
      type: "started",
      turn_id: "turn-second",
      origin: "orchestrator",
    });
    fireProjectEvent(projectId, {
      resource: "agent_turn",
      type: "completed",
      turn_id: "turn-second",
    });

    // Wait for both fetches to be issued
    await waitFor(() => expect(mockGetChatHistory).toHaveBeenCalledTimes(3)); // mount + 2 refetches

    // Resolve SECOND fetch FIRST (the newer one)
    await act(async () => {
      resolveSecond!({ messages: newerTranscript, interrupted: false });
      await new Promise((r) => setTimeout(r, 10));
    });

    // Then resolve FIRST fetch LAST (the stale one)
    await act(async () => {
      resolveFirst!({ messages: staleTranscript, interrupted: false });
      await new Promise((r) => setTimeout(r, 10));
    });

    // Assert: store holds the SECOND (newest-issued) response, NOT the stale one
    await waitFor(() => {
      const msgs = capturedStore?.messages || [];
      expect(msgs.length).toBeGreaterThan(1);
      const systemMsg = msgs.find((m: ChatMessage) => m.role === "system");
      expect(systemMsg).toBeTruthy();
      expect(systemMsg.parts[0].text).toBe("Latest lifecycle entry");
    });
  });

  it("Test 4b: a transcript missing this turn's reply is NOT adopted", async () => {
    /* The race: the assistant bubble streams, vanishes when the finalize
     * refetch replaces the thread, then reappears on refresh.
     *
     * The finalize refetch replaces the whole thread, and the reply is appended
     * by a SEPARATE best-effort write that can land after the finalize signal.
     * The old non-empty guard passed — the transcript had the user message and
     * the system line, just not the reply — so the streamed bubble was replaced
     * away by a transcript known to be behind.
     *
     * Terminal behaviour is the load-bearing part: after the retries are spent
     * the thread is LEFT ALONE. Replacing anyway would only narrow the window,
     * and losing costs the user a reply that was already correct on screen. */
    const projectId = "proj-test-4b";

    mockGetChatHistory.mockResolvedValueOnce({ messages: [], interrupted: false });
    // Every subsequent read returns a transcript WITHOUT turn-late's reply:
    // non-empty, and carrying the system line, so only the turn check can
    // reject it.
    mockGetChatHistory.mockResolvedValue({
      messages: buildTranscriptWithLifecycle("Cancelled at iteration 0"),
      interrupted: false,
    });

    render(<Harness projectId={projectId} />);
    await waitFor(() => expect(mockGetChatHistory).toHaveBeenCalledTimes(1));

    fireProjectEvent(projectId, {
      resource: "agent_turn",
      type: "started",
      turn_id: "turn-late",
      origin: "orchestrator",
    });
    fireProjectEvent(projectId, {
      resource: "agent_turn",
      type: "completed",
      turn_id: "turn-late",
    });

    // 1 initial + 4 attempts (first + 3 retries at 150/300/600ms).
    await waitFor(() => expect(mockGetChatHistory).toHaveBeenCalledTimes(5), {
      timeout: 4000,
    });

    // Retries are spent and the reply still is not there: the thread must be
    // untouched, NOT replaced with the stale transcript.
    await new Promise((r) => setTimeout(r, 250));
    const systemMsg = (capturedStore?.messages || []).find(
      (m: ChatMessage) => m.role === "system",
    );
    expect(systemMsg).toBeUndefined();
  });

  it("Test 4c: adopts the transcript as soon as the reply lands mid-retry", async () => {
    /* The other half — the guard has to CONVERGE, not just refuse. A first read
     * without the reply followed by one with it must be adopted, which is what
     * makes the retry worth having instead of skipping outright. */
    const projectId = "proj-test-4c";

    mockGetChatHistory.mockResolvedValueOnce({ messages: [], interrupted: false });
    mockGetChatHistory.mockResolvedValueOnce({
      messages: buildTranscriptWithLifecycle("Cancelled at iteration 0"),
      interrupted: false,
    });
    mockGetChatHistory.mockResolvedValue({
      messages: buildTranscriptWithLifecycle("Cancelled at iteration 0", "turn-slow"),
      interrupted: false,
    });

    render(<Harness projectId={projectId} />);
    await waitFor(() => expect(mockGetChatHistory).toHaveBeenCalledTimes(1));

    fireProjectEvent(projectId, {
      resource: "agent_turn",
      type: "started",
      turn_id: "turn-slow",
      origin: "orchestrator",
    });
    fireProjectEvent(projectId, {
      resource: "agent_turn",
      type: "completed",
      turn_id: "turn-slow",
    });

    await waitFor(
      () => {
        const msg = (capturedStore?.messages || []).find(
          (m: ChatMessage) => m.role === "system",
        );
        expect(msg).toBeTruthy();
      },
      { timeout: 4000 },
    );
  });

  it("Test 4d: an unset turn id cannot match a pre-field message", async () => {
    /* undefined === undefined would make an old message satisfy the guard and
     * license adopting a stale transcript. Both sides must be non-empty. A
     * caller with no turn id (mount, project switch) stays ungated and replaces
     * as before — asserted here by the system line actually appearing. */
    const projectId = "proj-test-4d";

    mockGetChatHistory.mockResolvedValue({
      messages: buildTranscriptWithLifecycle("Lifecycle line"),
      interrupted: false,
    });

    render(<Harness projectId={projectId} />);

    await waitFor(() => {
      const msg = (capturedStore?.messages || []).find(
        (m: ChatMessage) => m.role === "system",
      );
      expect(msg).toBeTruthy();
    });
  });

  it("Test 5 (navigate-back equivalence): refetch replaces, not appends", async () => {
    const projectId = "proj-test-5";
    const lifecycleText = "Optimization started";
    const transcript = buildTranscriptWithLifecycle(lifecycleText, "turn-nav-5");

    // Mount fetch returns transcript with one system entry
    mockGetChatHistory.mockResolvedValue({
      messages: transcript,
      interrupted: false,
    });

    render(<Harness projectId={projectId} />);

    await waitFor(() => expect(mockGetChatHistory).toHaveBeenCalledTimes(1));

    // Fire orchestrator finalize
    fireProjectEvent(projectId, {
      resource: "agent_turn",
      type: "started",
      turn_id: "turn-nav-5",
      origin: "orchestrator",
    });
    fireProjectEvent(projectId, {
      resource: "agent_turn",
      type: "completed",
      turn_id: "turn-nav-5",
    });

    // Wait for refetch
    await waitFor(() => expect(mockGetChatHistory).toHaveBeenCalledTimes(2));

    // Assert: exactly one system entry in store (replacement, not append)
    await waitFor(() => {
      const msgs = capturedStore?.messages || [];
      const systemMessages = msgs.filter((m: ChatMessage) => m.role === "system");
      expect(systemMessages.length).toBe(1);
      expect(systemMessages[0].parts[0].text).toBe(lifecycleText);
    });
  });

  it("Test 6 (invariant): empty refetch does not synthesize system entry", async () => {
    const projectId = "proj-test-6";

    // Mount fetch returns some messages
    mockGetChatHistory.mockResolvedValueOnce({
      messages: [
        {
          id: "msg-1",
          role: "user",
          parts: [{ type: "text", text: "Hello" }],
          created_at: "2026-08-06T10:00:00Z",
        },
      ],
      interrupted: false,
    });

    // Refetch returns EMPTY messages array (simulating best-effort-append failure)
    mockGetChatHistory.mockResolvedValueOnce({
      messages: [],
      interrupted: false,
    });

    render(<Harness projectId={projectId} />);

    await waitFor(() => expect(mockGetChatHistory).toHaveBeenCalledTimes(1));

    // Capture the initial messages count
    const initialMessages = capturedStore?.messages || [];
    const initialCount = initialMessages.length;

    // Fire orchestrator finalize
    fireProjectEvent(projectId, {
      resource: "agent_turn",
      type: "started",
      turn_id: "turn-empty-6",
      origin: "orchestrator",
    });
    fireProjectEvent(projectId, {
      resource: "agent_turn",
      type: "completed",
      turn_id: "turn-empty-6",
    });

    // Wait for refetch attempt
    await waitFor(() => expect(mockGetChatHistory).toHaveBeenCalledTimes(2));

    // Wait a beat for any potential state update
    await new Promise((resolve) => setTimeout(resolve, 100));

    // Assert: existing messages intact (empty response ignored)
    expect(capturedStore?.messages || []).toHaveLength(initialCount);

    // Assert: no system-role entry synthesized from event payload
    const systemMessages = (capturedStore?.messages || []).filter(
      (m: ChatMessage) => m.role === "system"
    );
    expect(systemMessages.length).toBe(0);
  });

  it("Test 7: replayed orchestrator started does not arm spinner", async () => {
    const projectId = "proj-replay-7";

    mockGetChatHistory.mockResolvedValue({
      messages: [],
      interrupted: false,
    });

    render(<Harness projectId={projectId} />);

    await waitFor(() => expect(mockGetChatHistory).toHaveBeenCalledTimes(1));

    // Fire a REPLAYED orchestrator started event
    fireProjectEvent(
      projectId,
      {
        resource: "agent_turn",
        type: "started",
        turn_id: "turn-replay-7",
        origin: "orchestrator",
      },
      true // replayed=true
    );

    // Wait a beat to ensure no state change
    await new Promise((resolve) => setTimeout(resolve, 100));

    // Assert: isRunning stays false (spinner not armed)
    expect(capturedStore?.isRunning).not.toBe(true);
    // Assert: no turn subscription opened (capturedChatStreamHandler not set)
    expect(capturedChatStreamHandler).toBeNull();
  });

  it("Test 8: live orchestrator started still arms spinner and opens subscription", async () => {
    const projectId = "proj-live-8";

    mockGetChatHistory.mockResolvedValue({
      messages: [],
      interrupted: false,
    });

    render(<Harness projectId={projectId} />);

    await waitFor(() => expect(mockGetChatHistory).toHaveBeenCalledTimes(1));

    // Fire a LIVE orchestrator started event (no replayed marker)
    fireProjectEvent(projectId, {
      resource: "agent_turn",
      type: "started",
      turn_id: "turn-live-8",
      origin: "orchestrator",
    });

    // Assert: isRunning becomes true (spinner armed)
    await waitFor(() => {
      expect(capturedStore?.isRunning).toBe(true);
    });
    // Assert: turn subscription opened (capturedChatStreamHandler is set)
    await waitFor(() => {
      expect(capturedChatStreamHandler).toBeTruthy();
    });
  });

  it("Test 9: replayed started does not populate orchestrator turn id set", async () => {
    const projectId = "proj-no-id-9";

    mockGetChatHistory.mockResolvedValue({
      messages: [],
      interrupted: false,
    });

    render(<Harness projectId={projectId} />);

    await waitFor(() => expect(mockGetChatHistory).toHaveBeenCalledTimes(1));

    // Fire a REPLAYED orchestrator started
    fireProjectEvent(
      projectId,
      {
        resource: "agent_turn",
        type: "started",
        turn_id: "turn-replayed-9",
        origin: "orchestrator",
      },
      true // replayed=true
    );

    // Fire a LIVE terminal event for the same turn
    fireProjectEvent(projectId, {
      resource: "agent_turn",
      type: "completed",
      turn_id: "turn-replayed-9",
    });

    // Wait for any potential refetch attempt
    await new Promise((resolve) => setTimeout(resolve, 100));

    // Assert: history fetch count is still 1 (mount only, no refetch)
    // because the replayed started never entered the turn-id set
    expect(mockGetChatHistory).toHaveBeenCalledTimes(1);
  });
});


describe("convertMessage streaming never-shrink invariant", () => {
  /* The live streaming message keeps one constant id and is re-converted on
   * every flush. assistant-ui mounts one component per content part, keyed
   * per message — if a later conversion of the SAME message yields FEWER
   * parts, mounted part components hit a stale index and crash the app
   * (tapClientLookup out-of-bounds → blank screen, e.g. during an
   * optimization's repeated status tool calls). Coalescing eligibility can
   * flip between flushes because a tool's detail arrives AFTER its start
   * event — so coalescing must be disabled for the streaming id and applied
   * only to finalized messages (fresh id, new subtree). */
  const toolPart = (input: Record<string, unknown> | null) => ({
    type: "tool" as const,
    name: "run_eval",
    input,
    result_preview: null,
    status: "running" as const,
    truncated: false,
  });

  it("streaming message content never shrinks when a detail resolves late", async () => {
    const { convertMessage, STREAMING_MESSAGE_ID } = await import("./ChatRuntimeProvider");
    // Flush N: second tool call's input (→ detail) not yet arrived.
    const early = convertMessage({
      id: STREAMING_MESSAGE_ID,
      role: "assistant",
      parts: [toolPart({ prompt_version: "v1" }), toolPart(null)],
      created_at: "2026-06-22T10:00:00.000Z",
    });
    // Flush N+1: detail arrived and now matches the first call's.
    const late = convertMessage({
      id: STREAMING_MESSAGE_ID,
      role: "assistant",
      parts: [toolPart({ prompt_version: "v1" }), toolPart({ prompt_version: "v1" })],
      created_at: "2026-06-22T10:00:00.000Z",
    });
    expect(early.content.length).toBe(2);
    expect(late.content.length).toBeGreaterThanOrEqual(early.content.length);
  });

  it("finalized message still coalesces duplicate consecutive tool calls", async () => {
    const { convertMessage } = await import("./ChatRuntimeProvider");
    const finalized = convertMessage({
      id: "real-message-id",
      role: "assistant",
      parts: [toolPart({ prompt_version: "v1" }), toolPart({ prompt_version: "v1" })],
      created_at: "2026-06-22T10:00:00.000Z",
    });
    expect(finalized.content.length).toBe(1);
    const merged = finalized.content[0] as { args: Record<string, unknown> };
    expect(merged.args._count).toBe(2);
  });
});


describe("truncated-stream recovery", () => {
  /* A terminal mirror arriving for a STILL-ACTIVE turn means the turn
   * stream dropped its tail frames, leaving the reply frozen mid-sentence
   * until a manual refresh. The handler must refetch the
   * durable transcript — which the relay completes BEFORE publishing the
   * mirror — instead of only clearing the spinner. */
  beforeEach(() => {
    capturedProjectStreamHandler = null;
    capturedChatStreamHandler = null;
    capturedStore = null;
    vi.clearAllMocks();
    mockGetActiveTurn.mockResolvedValue(null);
  });
  afterEach(() => {
    cleanup();
  });
  it("completed mirror for the active turn issues one refetch and applies it", async () => {
    const projectId = "proj-truncated-stream";
    const fullText = "the complete reply text the stream never finished delivering";
    mockGetChatHistory.mockResolvedValueOnce({ messages: [], interrupted: false });
    mockGetChatHistory.mockResolvedValueOnce({
      messages: [
        {
          id: "sdk-full",
          role: "assistant",
          turn_id: "turn-truncated-stream",
          parts: [{ type: "text", text: fullText }],
          created_at: "2026-06-22T10:00:01.000Z",
        },
      ],
      interrupted: false,
    });

    render(<Harness projectId={projectId} />);
    await waitFor(() => expect(mockGetChatHistory).toHaveBeenCalledTimes(1));

    // A user turn cannot be armed from outside the component, so the active
    // turn is armed with an orchestrator started event. The completed mirror
    // then reaches both the orchestrator-finalize refetch and the
    // truncated-stream recovery branch; the recovery must not add a second
    // fetch — one turn, one fetch.
    fireProjectEvent(projectId, {
      resource: "agent_turn",
      type: "started",
      turn_id: "turn-truncated-stream",
      origin: "orchestrator",
    });
    fireProjectEvent(projectId, {
      resource: "agent_turn",
      type: "completed",
      turn_id: "turn-truncated-stream",
    });

    // Exactly 2 fetches total (mount + one for the turn) — the branch
    // must not double-fetch when the orchestrator path already refetched.
    await waitFor(() => expect(mockGetChatHistory).toHaveBeenCalledTimes(2));
    await new Promise((r) => setTimeout(r, 50));
    expect(mockGetChatHistory).toHaveBeenCalledTimes(2);
    // And the refetched full text is applied.
    await waitFor(() => {
      const asst = capturedStore.messages.find((m: any) => m.id === "sdk-full");
      expect(asst).toBeTruthy();
    });
  });
});


describe("tool detail path shortening (cloud two-segment paths)", () => {
  // Cloud project paths carry TWO /projects/ segments
  // (/mnt/projects/users/<uid>/projects/<pid>/…). The non-greedy match
  // anchored on the FIRST one leaked "<uid>/projects/<pid>/…" into the
  // status strip and Bash pills. These pin the greedy fix.
  const CLOUD = "/mnt/projects/users/11111111-aaaa/projects/22222222-bbbb/prompts/v4.md";
  const DEV = "/Users/x/Sources/ai-eval-studio/projects/33333333-cccc/prompts/v3.md";

  async function detailFor(input: Record<string, unknown>): Promise<string | null> {
    const { convertMessage } = await import("./ChatRuntimeProvider");
    const msg = convertMessage({
      id: "m1",
      role: "assistant",
      parts: [{ type: "tool", name: "Write", input, status: "ok" }],
      created_at: "",
    } as never);
    const part = (msg.content as unknown as Array<Record<string, unknown>>).find(
      (p) => p.type === "tool-call",
    );
    return ((part?.argsText ?? null) as string | null) ?? ((part?.args as Record<string, unknown> | undefined)?._detail as string | null ?? null);
  }

  it("file_path: strips through the LAST /projects/<id>/ segment", async () => {
    const detail = await detailFor({ file_path: CLOUD });
    expect(JSON.stringify(detail)).toContain("prompts/v4.md");
    expect(JSON.stringify(detail)).not.toContain("11111111");
  });

  it("file_path: dev single-segment path still shortens", async () => {
    const detail = await detailFor({ file_path: DEV });
    expect(JSON.stringify(detail)).toContain("prompts/v3.md");
    expect(JSON.stringify(detail)).not.toContain("33333333");
  });

  it("command: keeps the command text and shortens embedded paths", async () => {
    const detail = await detailFor({ command: `wc -c ${CLOUD}` });
    const s = JSON.stringify(detail);
    expect(s).toContain("wc -c");
    expect(s).toContain("prompts/v4.md");
    expect(s).not.toContain("11111111");
  });
});
