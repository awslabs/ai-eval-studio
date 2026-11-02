// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Assistant-UI runtime provider that connects the chat interface to the
 * backend's unified WebSocket envelope channel. Handles message
 * streaming with RAF-batched updates, session persistence, status
 * event display, and proper error handling.
 *
 * Subscribes to the ``turn:<turn_id>`` stream while a turn is in
 * flight; the singleton ``ws-client`` handles reconnect / token
 * rotation transparently. There is no per-turn polling or active-turn
 * fallback — the page-reload-mid-turn case is detected via the
 * ``interrupted`` flag returned by ``/api/projects/{id}/chat/history``,
 * which renders the resume banner in the UI.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  useExternalStoreRuntime,
  AssistantRuntimeProvider,
  type ThreadMessageLike,
  type AppendMessage,
} from "@assistant-ui/react";
import type { ChatMessage, ChatMessagePart, AgentStatusInfo } from "@/lib/types";
import { api } from "@/lib/api";
import { useProjectEvent, type BackendEvent } from "@/hooks/useBackendEvents";
import { useChatStream } from "@/hooks/useWsStream";
import { wsClient, type Envelope } from "@/lib/ws-client";
import { TOOL_LABELS, TOOL_DETAIL_BLOCKLIST, cleanToolName } from "@/lib/tool-labels";
import { toast } from "sonner";

/**
 * Context for exposing agent status to child components.
 * Allows the chat UI to display connection/streaming/tool status.
 */
const AgentStatusContext = createContext<AgentStatusInfo>({ state: "idle" });

/**
 * Context for exposing session-interrupted state to child components.
 * True when the page was refreshed mid-turn and the session needs resuming.
 */
const SessionInterruptedContext = createContext<boolean>(false);

/**
 * Context exposing the chat-history load state + a manual retry hook.
 * Components (the failure banner, primarily) read this to decide
 * whether to render and to trigger a re-fetch on click.
 */
type HistoryLoadState = "loading" | "loaded" | "failed";
interface HistoryStateValue {
  state: HistoryLoadState;
  retry: () => void;
}
const HistoryStateContext = createContext<HistoryStateValue>({
  state: "loading",
  retry: () => {},
});

/**
 * Context exposing the active project UUID so chat-internal components
 * (e.g. the composer's model picker) can call project-scoped APIs
 * without threading props through every layer.
 */
const ProjectIdContext = createContext<string | null>(null);

/**
 * Hook to read the current agent status from ChatRuntimeProvider.
 */
export function useAgentStatus(): AgentStatusInfo {
  return useContext(AgentStatusContext);
}

/**
 * Hook to check if the session was interrupted (e.g., page refresh mid-turn).
 */
export function useSessionInterrupted(): boolean {
  return useContext(SessionInterruptedContext);
}

/**
 * Hook returning the current chat-history load state and a retry trigger.
 */
export function useHistoryState(): HistoryStateValue {
  return useContext(HistoryStateContext);
}

/**
 * Hook to read the active project UUID inside chat-internal components.
 */
export function useChatProjectId(): string | null {
  return useContext(ProjectIdContext);
}

/** Strip absolute project paths down to just the filename or relative tail. */
function shortenPath(fullPath: string): string {
  // Strip everything up to and including the LAST /projects/<segment>/ —
  // greedy ``.*`` is load-bearing. Cloud paths carry TWO such segments
  // (/mnt/projects/users/<user-id>/projects/<project-id>/prompts/x.md);
  // a non-greedy match anchored on the first one returned
  // "<user-id>/projects/<project-id>/prompts/x…", leaking the user and
  // project ids into the status strip and Bash pills.
  const projectMatch = fullPath.match(/^.*\/projects\/[^/]+\/(.+)$/);
  if (projectMatch) return projectMatch[1];
  // Fallback: just the filename
  const slash = fullPath.lastIndexOf("/");
  return slash >= 0 ? fullPath.slice(slash + 1) : fullPath;
}

/**
 * Shorten every absolute project path INSIDE a shell command while keeping
 * the command itself. ``shortenPath(command)`` discarded everything before
 * the matched path — "wc -c /mnt/…/prompts/v4.md" rendered as a bare
 * mangled path with the actual command gone.
 */
function shortenCommandPaths(command: string): string {
  return command.replace(/[^\s'"]*\/projects\/[^/]+\//g, "");
}

/**
 * Pick the most relevant human-readable field out of a parsed tool input.
 * Shared between streaming (partial JSON parse path) and rehydration
 * (already-parsed object from /chat/history).
 */
function detailFromParsedInput(input: Record<string, unknown>): string | null {
  // Pre-computed detail from streaming path — already shortened.
  const precomputed = input._detail;
  if (typeof precomputed === "string" && precomputed) return precomputed;
  const q = input.query;
  if (typeof q === "string" && q) return `"${q}"`;
  const u = input.url;
  if (typeof u === "string" && u) return u;
  const c = input.command;
  if (typeof c === "string" && c) return `$ ${shortenCommandPaths(c)}`;
  const fp = input.file_path;
  if (typeof fp === "string" && fp) return shortenPath(fp);
  const p = input.path;
  if (typeof p === "string" && p) return shortenPath(p);
  const pat = input.pattern;
  if (typeof pat === "string" && pat) return `pattern: ${pat}`;
  const fn = input.filename;
  if (typeof fn === "string" && fn) return fn;
  const desc = input.description;
  if (typeof desc === "string" && desc) return desc;
  return null;
}

/**
 * Extract a human-readable detail from partial tool input JSON.
 * Tries to parse the accumulated JSON and pull out the most relevant field.
 */
function extractToolDetail(_toolName: string, partialJson: string): string | null {
  try {
    return detailFromParsedInput(JSON.parse(partialJson));
  } catch {
    // JSON incomplete — try to extract partial strings with regex
    const queryMatch = partialJson.match(/"query"\s*:\s*"([^"]+)/);
    if (queryMatch) return `"${queryMatch[1]}"`;

    const urlMatch = partialJson.match(/"url"\s*:\s*"([^"]+)/);
    if (urlMatch) return urlMatch[1];

    const cmdMatch = partialJson.match(/"command"\s*:\s*"([^"]+)/);
    if (cmdMatch) return `$ ${shortenCommandPaths(cmdMatch[1])}`;

    const pathMatch = partialJson.match(/"(?:file_)?path"\s*:\s*"([^"]+)/);
    if (pathMatch) return shortenPath(pathMatch[1]);

    return null;
  }
}

// TOOL_LABELS imported from @/lib/tool-labels (single source of truth)

/** The constant id of the in-flight streaming message (one per turn). */
export const STREAMING_MESSAGE_ID = "streaming";

/**
 * Convert backend ChatMessage with parts to assistant-ui ThreadMessageLike.
 * Maps backend parts directly to assistant-ui's native part types:
 * - text → TextMessagePart
 * - thinking → ReasoningMessagePart
 * - tool → ToolCallMessagePart
 *
 * @param msg - Backend message format with ordered parts
 * @returns assistant-ui compatible message structure
 */
export function convertMessage(msg: ChatMessage): ThreadMessageLike {
  // The LIVE streaming message (constant id "streaming") is re-converted
  // on every flush, and a part's coalescing eligibility can CHANGE between
  // flushes — a tool's detail arrives via tool_input events after its
  // tool_start pushed the part, so a row that rendered separately can later
  // fold into its predecessor. That SHRINKS the converted content array under
  // assistant-ui's mounted per-part components, which crash on their stale
  // index (tapClientLookup out-of-bounds → blank screen, e.g. during an
  // optimization's repeated status tool calls). For the streaming message
  // the content list must be append-only, so coalescing is disabled; the
  // finalized message gets a FRESH id (finalizeStreamingMessage mints one),
  // mounts as a new subtree, and folds duplicates safely there.
  const coalesce = msg.id !== STREAMING_MESSAGE_ID;
  // Build assistant-ui content array directly from msg.parts IN ORDER.
  const content: Array<
    | { type: "text"; text: string }
    | { type: "reasoning"; text: string }
    | {
        type: "tool-call";
        toolCallId: string;
        toolName: string;
        args: Record<string, unknown>;
        result: string;
      }
  > = [];

  // Guard: a pre-parts-model record loaded from a stale transcript may lack
  // `parts` — skip it rather than crash. The backend self-heal rebuilds it.
  for (const part of (msg.parts ?? [])) {
    switch (part.type) {
      case "text":
        content.push({ type: "text" as const, text: part.text });
        break;
      case "thinking":
        content.push({ type: "reasoning" as const, text: part.text });
        break;
      case "tool": {
        // Map backend tool part to assistant-ui tool-call part.
        // Extract detail from input for the compact display.
        const detail = TOOL_DETAIL_BLOCKLIST.has(part.name)
          ? null
          : part.input && typeof part.input === "object"
            ? detailFromParsedInput(part.input as Record<string, unknown>)
            : null;
        // Use result_preview as the result field; fall back to status-based text.
        const result =
          part.result_preview ||
          (part.status === "ok"
            ? "Done"
            : part.status === "error"
              ? "Error"
              : "Running");

        // Coalesce a run of CONSECUTIVE identical tool calls (same tool +
        // same detail) into ONE row that carries a repeat count. The merged
        // part keeps its real toolName, so it still routes through the
        // specialized UI registry — every tool UI already renders the count
        // via args._count, so we only fold duplicates, never bypass the
        // dedicated renderer. Distinct details stay separate rows so we never
        // hide WHICH file/command ran; a text/thinking part between two tool
        // calls breaks the run (they're no longer consecutive).
        const prev = content[content.length - 1];
        if (
          coalesce &&
          prev &&
          prev.type === "tool-call" &&
          prev.toolName === part.name &&
          ((prev.args._detail as string | undefined) ?? null) === (detail ?? null)
        ) {
          prev.args._count = ((prev.args._count as number | undefined) ?? 1) + 1;
          // Reflect the latest call's outcome (e.g. the run's final status).
          prev.result = result;
          break;
        }

        const args: Record<string, unknown> = {};
        if (detail) args._detail = detail;
        content.push({
          type: "tool-call" as const,
          toolCallId: `${msg.id}-${part.name}-${content.length}`,
          toolName: part.name,
          args,
          result,
        });
        break;
      }
    }
  }

  return {
    id: msg.id,
    role: msg.role,
    content: content as ThreadMessageLike["content"],
    createdAt: msg.created_at ? new Date(msg.created_at) : undefined,
  };
}

/**
 * The messages assistant-ui is handed, in order: the finalized thread plus the
 * in-flight streaming message, minus assistant messages that have only tool
 * parts and no text or thinking (they appear as empty bubbles after reconnect).
 */
export function renderableMessages(
  messages: ChatMessage[],
  streaming: ChatMessage | null,
): ChatMessage[] {
  const base = streaming ? [...messages, streaming] : messages;
  return base.filter((msg) => {
    if (msg.role !== "assistant") return true;
    // Guard: a pre-parts-model record loaded from a stale transcript may
    // lack `parts` — treat as no content (backend self-heal rebuilds it).
    return (msg.parts ?? []).some((p) => p.type === "text" || p.type === "thinking");
  });
}


interface ChatRuntimeProviderProps {
  /** Project UUID to associate chat messages with */
  projectId: string;
  children: ReactNode;
}

/**
 * Provides assistant-ui runtime for chat functionality.
 * Manages message state, handles WebSocket envelope streaming from
 * the backend with RAF-batched updates, tracks session ID for
 * conversation continuity, and emits agent status for UI feedback.
 *
 * @param projectId - Project UUID for API requests
 * @param children - Chat UI components to wrap
 */
export function ChatRuntimeProvider({
  projectId,
  children,
}: ChatRuntimeProviderProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streamingMessage, setStreamingMessage] = useState<ChatMessage | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [agentStatus, setAgentStatus] = useState<AgentStatusInfo>({ state: "idle" });
  /* History-load state machine. ``loading`` covers initial fetch +
   * a single auto-retry; ``failed`` surfaces a retry banner; ``loaded``
   * gates the effect so we don't re-fetch on every render. Splitting
   * this from the previous boolean means a transient deploy-gap 502
   * doesn't leave the user staring at an empty thread that never tries
   * again — and a real failure shows a banner instead of pretending
   * the session has no messages. Mirrors the backend cleanup in
   * ``backend/app/platform/chat_history.py``: real errors must be
   * observable, not silently degrade to empty. */
  const [historyState, setHistoryState] = useState<
    "loading" | "loaded" | "failed"
  >("loading");
  const [historyRetryNonce, setHistoryRetryNonce] = useState(0);
  /* The session-interrupted banner is shown when the last persisted
   * message is user-only AND nothing is running right now. Both halves
   * matter: a refresh during a streaming agent reply must NOT show the
   * banner, even though the persisted transcript happens to end with
   * the user message at that moment. The visible value is derived
   * inline below from these two stored signals so a turn in flight
   * masks the banner; the effect below ALSO clears the stored flag
   * when a turn starts so a stale mount-time ``true`` doesn't
   * resurface after the turn ends.
   *
   * ``historyInterrupted`` reflects the backend's last word on the
   * persisted transcript shape; ``activeTurnId`` reflects whether
   * something's streaming now. */
  const [historyInterrupted, setHistoryInterrupted] = useState(false);

  /* ``activeTurnId`` drives the WebSocket subscription. When non-null,
   * ``useChatStream`` subscribes to ``turn:<id>``; setting it to null
   * unsubscribes. Refs alongside the state so synchronous handlers
   * (envelope arrival, RAF flush) can read the current value without
   * waiting for the next render. */
  const [activeTurnId, setActiveTurnId] = useState<string | null>(null);

  /* When a turn begins (user-initiated, orchestrator-initiated, or
   * recovered after refresh), the session is no longer "interrupted" —
   * something is actively responding. Clear the stored flag here so
   * after the turn completes and ``activeTurnId`` goes null again, the
   * banner doesn't resurface from the mount-time fetch. The render
   * gate (``historyInterrupted && activeTurnId === null``) still hides
   * it during the turn; this just ensures it stays hidden afterward. */
  useEffect(() => {
    if (activeTurnId !== null) setHistoryInterrupted(false);
  }, [activeTurnId]);

  /* Previously we dispatched an ``ai-eval-studio:agent-turn-complete``
   * window event here when the agent transitioned back to idle so
   * panels could refresh. That's now published server-side as
   * ``{resource:"agent_turn", type:"completed"}`` on the project event
   * bus — panels subscribe via ``useProjectEvent`` from
   * ``useBackendEvents``. One channel, typed payloads, no window
   * dispatcher to audit. */

  const sessionIdRef = useRef<string | null>(null);
  const activeTurnIdRef = useRef<string | null>(null);
  activeTurnIdRef.current = activeTurnId;

  // Mirror projectId into a ref so handlers that live inside a
  // useCallback([]) closure read the current value instead of the one
  // captured at mount.
  const projectIdRef = useRef(projectId);
  useEffect(() => { projectIdRef.current = projectId; }, [projectId]);

  /* Track orchestrator turn ids for refetch-on-finalize. Populated from
   * the "started" event (before startTurn's guard can reject the
   * subscription), so the cancel path's stranded-turn case still records
   * the id and refetches correctly. Bounded at 32 entries — a long-lived
   * loop must not grow this set without limit. */
  const orchestratorTurnIdsRef = useRef<Set<string>>(new Set());

  /* Monotonic request nonce for refetch ordering. api.getChatHistory
   * takes no AbortSignal (``getChatHistory`` in lib/api.ts), so overlapping
   * fetches cannot be
   * cancelled. A stale response landing last would roll the thread
   * backwards; the nonce guard prevents that. */
  const refetchNonceRef = useRef(0);

  // AgentCore turns run through the cancellable /ws relay (browser ↔ /ws ↔
  // coordinator ↔ runtime): POST /chat registers a backend turn_id in
  // _relay_tasks that /chat/stop/{turn_id} can address, so a user Stop issues
  // StopRuntimeSession on the microVM. The former direct browser→runtime SSE
  // path (localStorage "eval-studio-agentcore-chat") was removed: it had no
  // backend turn id, so its cancel could only abort the browser reader while
  // the microVM kept running (billing + mutating the tree). All AgentCore
  // turns are relay turns now — one transport, always cancellable.

  // Refs for RAF-batched streaming updates with ordered parts model
  const partsRef = useRef<ChatMessagePart[]>([]);
  const currentToolNameRef = useRef("");
  const currentToolInputRef = useRef("");
  const currentToolIdxRef = useRef(-1);
  const rafRef = useRef<number | null>(null);
  const flushIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const messageStartTimeRef = useRef("");

  // Track last flushed content to skip no-op updates that cause scroll jumps
  const lastFlushedPartsCountRef = useRef(0);
  const lastFlushedPartsLenRef = useRef(0);

  /** Schedule a RAF to push ref state into React state (deduplicated). */
  const scheduleFlush = useCallback(() => {
    if (rafRef.current) return;
    rafRef.current = requestAnimationFrame(() => {
      const partsCount = partsRef.current.length;
      // Compute a simple content length fingerprint: sum of all text/thinking
      // part lengths. Only update if parts changed — prevents assistant-ui
      // from interpreting no-op state updates as "new messages" and scrolling.
      let partsLen = 0;
      for (const p of partsRef.current) {
        if (p.type === "text" || p.type === "thinking") partsLen += p.text.length;
      }
      if (
        partsCount !== lastFlushedPartsCountRef.current ||
        partsLen !== lastFlushedPartsLenRef.current
      ) {
        lastFlushedPartsCountRef.current = partsCount;
        lastFlushedPartsLenRef.current = partsLen;
        setStreamingMessage({
          id: STREAMING_MESSAGE_ID,
          role: "assistant",
          parts: [...partsRef.current],
          created_at: messageStartTimeRef.current,
        });
      }
      rafRef.current = null;
    });
  }, []);

  /** Reset all per-turn streaming state — refs, batching, and metadata. */
  const resetStreamingState = useCallback(() => {
    partsRef.current = [];
    currentToolNameRef.current = "";
    currentToolInputRef.current = "";
    currentToolIdxRef.current = -1;
    lastFlushedPartsCountRef.current = 0;
    lastFlushedPartsLenRef.current = 0;
    messageStartTimeRef.current = new Date().toISOString();
  }, []);

  /**
   * Finalize the current streaming message into the messages list.
   * Used when a turn ends naturally (``result`` envelope) and on
   * cancellation. Idempotent — flushes the streaming buffer if any
   * content is pending and clears the agent state.
   */
  const finalizeStreamingMessage = useCallback(() => {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    const finalMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "assistant",
      parts: [...partsRef.current],
      created_at: messageStartTimeRef.current,
    };
    if (finalMessage.parts.length > 0) {
      setMessages((prev) => [...prev, finalMessage]);
    }
    setStreamingMessage(null);

    if (flushIntervalRef.current) {
      clearInterval(flushIntervalRef.current);
      flushIntervalRef.current = null;
    }
    setIsRunning(false);
    setAgentStatus({ state: "idle" });
  }, []);

  /** Guarded transcript refetch shared by the finalize paths.
   * Nonce guard (only the latest issued response applies), project guard
   * (a response for the previous project must not land after a switch),
   * non-empty guard (the backend append is best-effort; an unguarded
   * replacement would delete the locally-finalized reply on failure). */
  const refetchHistoryGuarded = useCallback(
    (reason: string, expectTurnId?: string) => {
      const nonce = ++refetchNonceRef.current;
      const capturedProjectId = projectIdRef.current;

      /* Has the turn we just finalized actually reached the transcript?
       *
       * Matched on the message's own ``turn_id``, which the backend stamps on
       * a turn's trailing reply AND on its stopped/failed marker — so a turn
       * that legitimately produces no reply still satisfies this and the
       * retry below converges instead of always exhausting.
       *
       * Both sides must be non-empty. Messages written before the field
       * existed have ``turn_id === undefined``, and an unset ``expectTurnId``
       * would make ``undefined === undefined`` match one of them — adopting a
       * stale transcript on the strength of an unrelated old message. */
      const transcriptHasTurn = (msgs: ChatMessage[]): boolean =>
        !!expectTurnId && msgs.some((m) => !!m.turn_id && m.turn_id === expectTurnId);

      /* Backoff for the finalize-vs-persistence race. The reply is appended by
       * a separate best-effort write that can land after the finalize signal,
       * so the first read can legitimately return a transcript without it. */
      const RETRY_DELAYS_MS = [150, 300, 600];

      const attempt = (retryIdx: number): void => {
        api
          .getChatHistory(capturedProjectId)
          .then((data) => {
            // Apply only if this is still the latest request (ordering guard)
            if (refetchNonceRef.current !== nonce) return;
            // Apply only if we haven't switched projects (project guard)
            if (projectIdRef.current !== capturedProjectId) return;
            // Apply only if the transcript is non-empty (best-effort-append guard)
            if (data.messages.length === 0) return;

            /* The turn we are reconciling has not landed yet. Retry briefly,
             * then GIVE UP WITHOUT REPLACING.
             *
             * Not replacing is the load-bearing half. Replacing after the last
             * retry would only narrow the race window rather than close it, and
             * what it costs when it loses is the user's reply disappearing from
             * a thread that was already showing it correctly. The streamed
             * content on screen IS this turn's reply; a transcript known to be
             * missing it is strictly worse information. The next natural
             * refetch reconciles, so nothing is lost permanently.
             *
             * Only gated when a turn id was supplied — callers without one
             * (mount, project switch) have nothing to wait for. */
            if (expectTurnId && !transcriptHasTurn(data.messages)) {
              if (retryIdx < RETRY_DELAYS_MS.length) {
                window.setTimeout(() => attempt(retryIdx + 1), RETRY_DELAYS_MS[retryIdx]);
                return;
              }
              console.warn(
                `[chat-history] refetch (${reason}): turn ${expectTurnId.slice(0, 8)} ` +
                  "not in transcript after retries — keeping streamed state",
              );
              return;
            }

            // A wholesale replacement: the transcript can split a reply into
            // fewer or different parts than the live view streamed.
            // ThreadMessages remounts the list when that would crash it in place.
            setMessages(data.messages);
            setHistoryInterrupted(data.interrupted);
          })
          .catch((err) => {
            console.warn(`[chat-history] refetch (${reason}) failed:`, err);
            // Failed refetch leaves the thread as-is — this is a display
            // refresh, not a load, so it gets no banner and no retry.
          });
      };

      attempt(0);
    },
    [],
  );

  /**
   * Re-fetch the durable transcript when an orchestrator turn finalizes.
   *
   * The lifecycle attribution line is owned by the transcript: the history
   * endpoint derives it, and the project event is an invalidation signal
   * only, never a payload. An orchestrator turn finalizing issues one
   * history refetch — retried briefly while the turn's reply has not yet
   * reached the transcript — and the transcript's system line becomes
   * visible without a reload.
   *
   * Refetching on finalize rather than on `started` is deliberate — at
   * `started` the synthetic prompt is not yet guaranteed written to the SDK
   * session, so a fetch there can race and return a transcript without the
   * lifecycle entry.
   *
   * The two finalize signals (`result` envelope on `turn:<id>` +
   * `{resource:"agent_turn", type:"completed"}` on `project:<id>`) both
   * route through this callback, and the first one consumes the turn id, so
   * a single turn issues one refetch. No dedupe key is needed because
   * setMessages is a full replacement, not an append. The resulting pop-in
   * (system line appears with/just after the assistant bubble) is accepted
   * rather than papered over with an optimistic append — the transcript
   * defines order.
   *
   * @returns true when a refetch was issued (lets the recovery skip a
   * duplicate fetch for orchestrator turns — one turn, one fetch).
   */
  const refetchOnOrchestratorFinalize = useCallback((turnId: string): boolean => {
    if (!orchestratorTurnIdsRef.current.has(turnId)) return false;
    orchestratorTurnIdsRef.current.delete(turnId);
    // Pass the turn id so the refetch waits for THIS turn's reply to reach the
    // transcript rather than replacing the thread with whatever is there now.
    refetchHistoryGuarded("orchestrator finalize", turnId);
    return true;
  }, [refetchHistoryGuarded]);

  /**
   * Handle a single envelope arriving on the active turn's WebSocket
   * stream. ``ws-client`` has already parsed the outer envelope, so the
   * dispatch is a switch on ``envelope.kind`` consuming ``envelope.data``
   * directly.
   */
  const handleEnvelope = useCallback(
    (envelope: Envelope) => {
      const kind = envelope.kind;
      const data = (envelope.data ?? {}) as Record<string, unknown>;

      switch (kind) {
        case "status":
          setAgentStatus({
            state: data.state as AgentStatusInfo["state"],
            detail: data.tool_name
              ? (TOOL_LABELS[data.tool_name as string] || cleanToolName(data.tool_name as string))
              : undefined,
          });
          break;

        case "text_delta": {
          const delta = (data.text as string) ?? "";
          // Append to (or extend) the trailing text part. If the last part is
          // not a text part, push a new one.
          const lastPart = partsRef.current[partsRef.current.length - 1];
          if (lastPart && lastPart.type === "text") {
            lastPart.text += delta;
          } else {
            partsRef.current.push({ type: "text", text: delta });
          }
          scheduleFlush();
          break;
        }

        case "thinking_start": {
          // Open a new reasoning segment.
          partsRef.current.push({ type: "thinking", text: "", truncated: false });
          scheduleFlush();
          break;
        }

        case "thinking_delta": {
          const delta = (data.text as string) ?? "";
          // Append to the trailing thinking part.
          const lastPart = partsRef.current[partsRef.current.length - 1];
          if (lastPart && lastPart.type === "thinking") {
            lastPart.text += delta;
          }
          scheduleFlush();
          break;
        }

        case "tool_start": {
          const name = (data.name as string) ?? "";
          // Push a new tool part with status "running".
          partsRef.current.push({
            type: "tool",
            name,
            input: null,
            result_preview: null,
            status: "running",
            truncated: false,
          });
          currentToolNameRef.current = name;
          currentToolInputRef.current = "";
          currentToolIdxRef.current = partsRef.current.length - 1;
          setAgentStatus({
            state: "tool_running",
            detail: TOOL_LABELS[name] || cleanToolName(name),
          });
          scheduleFlush();
          break;
        }

        case "tool_input": {
          currentToolInputRef.current += (data.partial_json as string) ?? "";
          if (TOOL_DETAIL_BLOCKLIST.has(currentToolNameRef.current)) break;
          const detail = extractToolDetail(
            currentToolNameRef.current,
            currentToolInputRef.current,
          );
          if (detail) {
            const label = TOOL_LABELS[currentToolNameRef.current]
              || cleanToolName(currentToolNameRef.current);
            setAgentStatus({
              state: "tool_running",
              detail: `${label} — ${detail}`,
            });
            const idx = currentToolIdxRef.current;
            if (idx >= 0 && idx < partsRef.current.length) {
              const part = partsRef.current[idx];
              if (part.type === "tool") {
                part.input = { _detail: detail };
              }
            }
            scheduleFlush();
          }
          break;
        }

        case "tool_end": {
          // content_block_stop for the tool_use block — model finished emitting
          // the tool call. Input tracking can reset now, but keep the tool name
          // around so the paired tool_result (below) knows which tool ran.
          // Backend may rewrite the tool name here (e.g. generic "Task" becomes
          // "subagent:data-analyzer" once the subagent_type arg has arrived).
          let renamedTo: string | null = null;
          if (typeof data.name === "string" && data.name) {
            renamedTo = data.name;
            const idx = currentToolIdxRef.current;
            if (idx >= 0 && idx < partsRef.current.length) {
              const part = partsRef.current[idx];
              if (part.type === "tool") {
                part.name = data.name;
              }
            }
            currentToolNameRef.current = data.name;
          }
          currentToolInputRef.current = "";
          currentToolIdxRef.current = -1;
          // For subagent delegations the tool_use block closes immediately
          // after the coordinator emits it — but the subagent keeps running
          // for potentially minutes. Keep the status bar showing the
          // subagent label ("Building harness…" etc.) until the actual
          // SubagentStop fires, instead of flipping to a misleading
          // "Processing…" that makes the UI look idle.
          if (renamedTo && renamedTo.startsWith("subagent:")) {
            setAgentStatus({
              state: "tool_running",
              detail: TOOL_LABELS[renamedTo] || cleanToolName(renamedTo),
            });
          } else {
            setAgentStatus({ state: "streaming", detail: "Processing…" });
          }
          // Don't flush immediately — text content hasn't changed, only tool status.
          // The 500ms interval flush or next text_delta will pick it up.
          // Flushing here causes scroll jumping on every tool completion.
          break;
        }

        case "tool_result": {
          /* Tool actually ran on the backend. Set the trailing tool part's
           * status + result_preview from the envelope. The backend
           * agent_runner already publishes
           * ``{resource:"filesystem", type:"changed"}`` on the project event
           * stream for FS-mutating tools, so panels refresh via
           * ``useProjectEvent`` — no frontend dispatch needed. */
          const status = (data.status as "ok" | "error") ?? "ok";
          const result_preview = (data.result_preview as string) ?? null;
          const truncated = (data.truncated as boolean) ?? false;
          // Find the trailing tool part and set its result fields.
          for (let i = partsRef.current.length - 1; i >= 0; i--) {
            const part = partsRef.current[i];
            if (part.type === "tool") {
              part.status = status;
              part.result_preview = result_preview;
              part.truncated = truncated;
              break;
            }
          }
          currentToolNameRef.current = "";
          setAgentStatus({ state: "streaming", detail: "Processing…" });
          scheduleFlush();
          break;
        }

        case "result": {
          if (typeof data.session_id === "string") {
            sessionIdRef.current = data.session_id;
          }
          /* Turn-complete notification is published by the backend as
           * ``{resource:"agent_turn", type:"completed"}``; no window
           * event dispatch needed here either. */
          /* End of turn — finalize the streaming message and clear
           * ``activeTurnId``, which unsubscribes us from the WS
           * stream. ``finalizeStreamingMessage`` handles flushing
           * the partial buffer and resetting agent status. */
          const finalizingTurnId = activeTurnIdRef.current;
          finalizeStreamingMessage();
          setActiveTurnId(null);
          /* Refetch on orchestrator finalize (first signal path). The
           * `result` envelope carries no turn id, so read it from
           * activeTurnIdRef BEFORE the above setActiveTurnId(null). */
          if (finalizingTurnId) refetchOnOrchestratorFinalize(finalizingTurnId);
          break;
        }
      }
    },
    [finalizeStreamingMessage, scheduleFlush, refetchOnOrchestratorFinalize],
  );

  /* Subscribe to the active turn's stream while one is in flight.
   * ``useChatStream`` no-ops when ``activeTurnId`` is null. */
  useChatStream(activeTurnId, handleEnvelope);

  /**
   * Begin tracking a new turn — reset streaming state, set initial
   * agent status, and subscribe via ``useChatStream``.
   */
  const startTurn = useCallback(
    (turnId: string) => {
      // Guard: don't clobber an in-flight stream for a different turn
      if (activeTurnIdRef.current && activeTurnIdRef.current !== turnId) {
        return;
      }

      resetStreamingState();
      setIsRunning(true);
      setAgentStatus({ state: "streaming" });

      if (flushIntervalRef.current) clearInterval(flushIntervalRef.current);
      flushIntervalRef.current = setInterval(scheduleFlush, 500);

      setActiveTurnId(turnId);
    },
    [resetStreamingState, scheduleFlush],
  );

  /** Clean up intervals and RAF on unmount to prevent leaks. */
  useEffect(() => {
    return () => {
      if (flushIntervalRef.current) {
        clearInterval(flushIntervalRef.current);
        flushIntervalRef.current = null;
      }
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, []);

  /**
   * Hydrate chat history on mount. The ``interrupted`` flag in the
   * response indicates a mid-turn page reload that ended with the
   * agent never replying — the UI shows a resume banner so the user
   * can retry. A separate effect below covers the *currently*
   * mid-turn case via ``/chat/active-turn``.
   *
   * On fetch failure: try once more after 1.5s (covers transient
   * 502/503 during a deploy roll), then surface the failed state so
   * ``HistoryFailedBanner`` can render a manual retry button. Never
   * silently degrade to ``messages: []`` — an empty thread reads as a
   * brand-new session and hides the failure.
   */
  useEffect(() => {
    if (historyState === "loaded") return;
    let cancelled = false;
    let retryTimer: ReturnType<typeof setTimeout> | undefined;

    const attempt = (isRetry: boolean) => {
      api
        .getChatHistory(projectId)
        .then((data) => {
          if (cancelled) return;
          if (data.messages.length > 0) {
            // The provider outlives a project switch, so this load can replace
            // another project's thread, not just an empty one. ThreadMessages
            // remounts the list when that replacement would crash it in place.
            setMessages(data.messages);
          }
          setHistoryInterrupted(data.interrupted);
          setHistoryState("loaded");
        })
        .catch((err) => {
          if (cancelled) return;
          if (!isRetry) {
            // Auto-retry once. A deploy roll typically returns 502/503
            // for a few seconds; surfacing the banner instantly would
            // train users to ignore it.
            retryTimer = setTimeout(() => attempt(true), 1500);
            return;
          }
          console.error("[chat-history] load failed after retry:", err);
          setHistoryState("failed");
        });
    };

    attempt(false);

    return () => {
      cancelled = true;
      if (retryTimer) clearTimeout(retryTimer);
    };
  }, [projectId, historyState, historyRetryNonce]);

  // Reset history state when the project changes — different project,
  // different session to load.
  useEffect(() => {
    setHistoryState("loading");
    setHistoryRetryNonce(0);
    orchestratorTurnIdsRef.current.clear();
  }, [projectId]);

  const retryHistory = useCallback(() => {
    setHistoryState("loading");
    setHistoryRetryNonce((n) => n + 1);
  }, []);

  /**
   * Mid-turn refresh recovery. After history hydrates, ask the
   * backend whether a turn is currently in flight for this project.
   * If yes, set ``activeTurnId`` — that triggers ``useChatStream`` to
   * subscribe to ``turn:<id>``, and the WS client's ``onopen`` /
   * subscribe path sends ``{action:"subscribe",since_seq:-1}`` so the
   * hub replays the running turn's full envelope buffer. Live
   * envelopes after the snapshot stream in via the same channel; no
   * client-side replay loop, no race window.
   *
   * Runs exactly once per (project, history-load) — gating on
   * ``historyState === "loaded"`` plus an ``activeTurnRecovered`` ref
   * so a re-render doesn't re-replay. The ref resets on project
   * change via the same effect that resets the history state above.
   */
  const activeTurnRecoveredRef = useRef(false);
  useEffect(() => {
    activeTurnRecoveredRef.current = false;
  }, [projectId]);

  useEffect(() => {
    if (historyState !== "loaded") return;
    if (activeTurnRecoveredRef.current) return;
    if (activeTurnIdRef.current) return; // turn already in flight from this session
    activeTurnRecoveredRef.current = true;

    let cancelled = false;
    api
      .getActiveTurn(projectId)
      .then((active) => {
        if (cancelled) return;
        if (!active || active.status !== "running") return;
        // Mirror startTurn's setup but skip ``resetStreamingState`` —
        // there's nothing to reset (we just hydrated history) and the
        // WS replay-on-subscribe will deliver the running turn's
        // buffered envelopes through the same handler used for live
        // frames.
        //
        // Cursor reset is LOAD-BEARING: the ws
        // client module survives SPA navigation, so without this the
        // resubscription asks for replay from the cursor advanced by the
        // PREVIOUS mount — the hub sends nothing earlier, this component
        // (which lost all its turn state at unmount) renders an empty
        // thread, and the turn's content only appears at finalize. Hard
        // refresh never hit it because a page load resets the module.
        wsClient.resetStreamCursor(`turn:${active.turn_id}`);
        setIsRunning(true);
        setAgentStatus({ state: "streaming" });
        if (flushIntervalRef.current) clearInterval(flushIntervalRef.current);
        flushIntervalRef.current = setInterval(scheduleFlush, 500);
        setActiveTurnId(active.turn_id);
      })
      .catch((err) => {
        console.warn("[active-turn] recovery failed:", err);
      });

    return () => {
      cancelled = true;
    };
  }, [historyState, projectId, scheduleFlush]);

  /* When an orchestrator completion turn starts, attach the WS
   * subscription to its turn id so the coordinator's response renders
   * live — same as user turns. The turn_id comes from the project
   * event stream. Also record the turn id for refetch-on-finalize —
   * record BEFORE startTurn's guard can reject the subscription, so the
   * stranded-turn case still refetches correctly.
   *
   * Classified as momentary (not state): arms a spinner, opens a turn
   * subscription, records the turn id. A replayed started would arm a
   * spinner and a subscription for a turn that has already ended. */
  useProjectEvent(
    { resource: "agent_turn", type: "started" },
    (event) => {
      if (event.turn_id && event.origin === "orchestrator") {
        orchestratorTurnIdsRef.current.add(event.turn_id);
        // Bound the set: after adding, if it exceeds 32, delete oldest entries.
        // Set iteration order is insertion order, so delete from the front.
        if (orchestratorTurnIdsRef.current.size > 32) {
          const toDelete: string[] = [];
          for (const id of orchestratorTurnIdsRef.current) {
            toDelete.push(id);
            if (toDelete.length >= orchestratorTurnIdsRef.current.size - 32) break;
          }
          toDelete.forEach((id) => orchestratorTurnIdsRef.current.delete(id));
        }
        startTurn(event.turn_id);
      }
    },
    "momentary"
  );

  /* Authoritative clear path for the running spinner / Stop button.
   *
   * The per-turn ``turn:<id>`` stream's ``result`` envelope is the
   * fast, detailed finalizer — but it can go missing (coordinator
   * restart kills the SDK subprocess before it emits ``result``; an
   * abnormally-torn-down turn never reaches it). When that happens the
   * composer is stuck showing Stop and the status bar keeps spinning
   * forever, because nothing else clears ``isRunning``.
   *
   * The backend ALSO publishes a terminal turn event
   * (``agent_turn`` / ``completed`` | ``cancelled`` | ``error``) on the
   * project stream — a separate channel that is always subscribed (via
   * ``useBackendEvents`` at the app layout) and is replayed on
   * reconnect. Treating that as the authoritative "turn ended" signal
   * means a dropped per-turn ``result`` no longer strands the UI: the
   * project event clears the same state the ``result`` handler would.
   * This is the chat-surface analogue of the eval bar's REST-poll
   * clear — riding an existing always-on channel rather than adding a
   * new poll.
   *
   * Guard on the tracked turn id so a stale event for a previous turn
   * can't tear down a freshly-started one. If ``result`` already
   * cleared us (``activeTurnIdRef`` is null), this is a harmless no-op. */
  const clearRunningState = useCallback(() => {
    if (flushIntervalRef.current) {
      clearInterval(flushIntervalRef.current);
      flushIntervalRef.current = null;
    }
    setIsRunning(false);
    setAgentStatus({ state: "idle" });
    setActiveTurnId(null);
  }, []);

  const handleTurnTerminal = useCallback(
    (event: BackendEvent) => {
      if (!event.turn_id) return;
      /* Refetch on orchestrator finalize (second signal path). Called
       * BEFORE the activeTurnIdRef guard below — the stranded-turn case
       * defeats that guard, but the refetch must still fire.
       * The orchestratorTurnIdsRef membership check ensures this only
       * refetches for orchestrator turns. */
      const orchestratorRefetched = refetchOnOrchestratorFinalize(event.turn_id);
      if (event.turn_id !== activeTurnIdRef.current) return;
      /* truncated-stream recovery. Reaching here means the terminal
       * mirror arrived while the turn is STILL active — the turn stream
       * never delivered its result envelope (dropped tail frames — the reply
       * freezes mid-sentence until a manual refresh). Finalize the partial locally
       * (preserved if the fetch fails), then refetch the transcript: the
       * relay appends the turn durably BEFORE publishing this mirror, so
       * the fetched history is guaranteed to carry the complete text. On
       * the normal path the result envelope already cleared
       * activeTurnIdRef and this branch never runs — no refetch per turn.
       * Orchestrator turns just refetched above — one turn, one fetch. */
      finalizeStreamingMessage();
      if (!orchestratorRefetched) {
        /* Deliberately NOT passed a turn id, unlike the orchestrator-finalize
         * path. The expect-turn guard's fallback is "keep the streamed state",
         * which is the safe direction there (local is complete; the transcript
         * only adds the lifecycle line) and the WRONG one here: reaching this
         * branch means the stream was truncated, so the local message is a
         * mid-sentence partial and the transcript is what completes it.
         * Keeping local on exhaustion would preserve the very partial this
         * recovery exists to replace. */
        refetchHistoryGuarded("terminal mirror on active turn");
      }
      clearRunningState();
    },
    [clearRunningState, refetchOnOrchestratorFinalize, finalizeStreamingMessage, refetchHistoryGuarded],
  );

  useProjectEvent({ resource: "agent_turn", type: "completed" }, handleTurnTerminal, "momentary");
  useProjectEvent({ resource: "agent_turn", type: "cancelled" }, handleTurnTerminal, "momentary");
  useProjectEvent({ resource: "agent_turn", type: "error" }, handleTurnTerminal, "momentary");

  /**
   * Handle new user message submission.
   *
   * Single transport: POST /chat (api.startChat) + startTurn + useChatStream
   * over /ws. When the relay flag is on for the project the backend runs the
   * turn in an AgentCore microVM and relays events over the same /ws stream;
   * the frontend is transport-agnostic. Every turn is addressable by its
   * backend turn_id so onCancel can stop it server-side.
   */
  const onNew = useCallback(
    async (message: AppendMessage) => {
      const textPart = message.content.find((p) => p.type === "text");
      if (!textPart || textPart.type !== "text") return;

      const userText = textPart.text.trim();
      if (!userText) return;

      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        parts: [{ type: "text", text: userText }],
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, userMsg]);
      messageStartTimeRef.current = new Date().toISOString();

      try {
        // Single transport: POST /chat. When the relay flag is on for the
        // project the backend runs the turn in an AgentCore microVM and relays
        // events over /ws; otherwise it runs in-process. Either way the turn is
        // addressable by the returned turn_id (in _relay_tasks for relay turns),
        // so onCancel can stop it server-side.
        const { turn_id, status } = await api.startChat(
          projectIdRef.current,
          userText,
          sessionIdRef.current ?? undefined,
        );

        // Handle busy rejection: the relay claim is held on the same project,
        // so the message is DROPPED (honest-busy). Do NOT subscribe to
        // a dead turn stream — clear spinner and surface a non-error notice.
        // ALSO remove the optimistic user bubble: the backend refused to
        // queue this message, so leaving it in the transcript shows a
        // message that never existed server-side. It would sit there with
        // no progress indicator and then silently vanish on the next
        // transcript refetch, which makes the send look like a no-op.
        if (status === "busy") {
          setMessages((prev) => prev.filter((m) => m.id !== userMsg.id));
          setIsRunning(false);
          setAgentStatus({ state: "idle" });
          toast.info("A turn is already running — retry in a moment.");
          return;
        }

        // For "running" or "queued" status, subscribe to the turn's WS stream
        // — the singleton ws-client handles reconnects + token rotation.
        startTurn(turn_id);
      } catch (error) {
        setIsRunning(false);
        setAgentStatus({ state: "idle" });
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant" as const,
            parts: [
              {
                type: "text",
                text: `Error: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            created_at: new Date().toISOString(),
          },
        ]);
      }
    },
    // Only ``startTurn`` is read directly; every other collaborator is reached
    // through a ref (projectIdRef and friends) precisely so this callback keeps a
    // stable identity across renders. Listing them anyway made the callback churn
    // whenever any of them was recreated, which re-registers downstream
    // subscriptions for no reason. ESLint flagged them as unnecessary — removed
    // rather than suppressed.
    [startTurn],
  );

  /**
   * Stop the background agent turn server-side, unsubscribe from the WS stream,
   * and clean up UI. The WS unsubscribe happens automatically when we set
   * ``activeTurnId`` to null below.
   */
  const onCancel = useCallback(async () => {
    // Every turn is addressable by its backend turn_id. POST /chat/stop/{id}
    // interrupts the in-process SDK client OR, for a relay turn, issues
    // StopRuntimeSession on the microVM (backend stop_turn branches on
    // _relay_tasks). This is the ONLY cancel path now that the uncancellable
    // direct browser→runtime SSE path is gone.
    const turnId = activeTurnIdRef.current;
    if (turnId) {
      api.stopChatTurn(turnId).catch(() => {});
    }

    // Finalize partial content as a message before unsubscribing.
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    if (partsRef.current.length > 0) {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant" as const,
          parts: [...partsRef.current],
          created_at: messageStartTimeRef.current,
        },
      ]);
    }
    setStreamingMessage(null);
    if (flushIntervalRef.current) {
      clearInterval(flushIntervalRef.current);
      flushIntervalRef.current = null;
    }
    setIsRunning(false);
    setAgentStatus({ state: "idle" });
    /* Setting ``activeTurnId`` to null triggers ``useChatStream`` to
     * unsubscribe from the turn's WS stream on the next render. */
    setActiveTurnId(null);
  }, []);

  // Merge finalized messages with streaming message for assistant-ui.
  const allMessages = useMemo(
    () => renderableMessages(messages, streamingMessage),
    [messages, streamingMessage],
  );

  const store = useMemo(
    () => ({
      isRunning,
      messages: allMessages,
      convertMessage,
      onNew,
      onCancel,
    }),
    [isRunning, allMessages, onNew, onCancel],
  );

  const runtime = useExternalStoreRuntime(store);

  const historyValue = useMemo<HistoryStateValue>(
    () => ({ state: historyState, retry: retryHistory }),
    [historyState, retryHistory],
  );

  return (
    <AgentStatusContext.Provider value={agentStatus}>
      <SessionInterruptedContext.Provider value={historyInterrupted && activeTurnId === null}>
        <HistoryStateContext.Provider value={historyValue}>
          <ProjectIdContext.Provider value={projectId}>
            <AssistantRuntimeProvider runtime={runtime}>
              {children}
            </AssistantRuntimeProvider>
          </ProjectIdContext.Provider>
        </HistoryStateContext.Provider>
      </SessionInterruptedContext.Provider>
    </AgentStatusContext.Provider>
  );
}
