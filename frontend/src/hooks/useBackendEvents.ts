// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Project-scoped backend event bus — the single channel the frontend
 * uses to react to anything that happens on the backend outside of the
 * chat / eval-job WS streams. One subscription on the
 * ``project:<id>`` WebSocket stream per project, opened at the app
 * layout level, fans events out to:
 *
 * - ``toast.error`` for ``{type: "error"}`` payloads so failures surface
 *   even when the user is on another tab.
 * - An in-process ``EventTarget`` (``projectEventTarget``) that
 *   component-level ``useProjectEvent`` hooks subscribe to. This avoids
 *   stacking redundant WebSocket subscribers per consumer and keeps the
 *   backend fan-out cheap.
 *
 * Replaces two prior ``window.dispatchEvent`` buses (``eval-studio:refresh``
 * and ``ai-eval-studio:project-fs-changed``) and the previous
 * ``EventSource`` transport — there is now exactly one channel from
 * backend to frontend for resource-change notifications, typed by
 * ``resource`` + ``type``.
 */
import { useEffect, useRef } from "react";
import { toast } from "sonner";

import { useProjectStream } from "@/hooks/useWsStream";
import { wsClient, type Envelope } from "@/lib/ws-client";
import type { ResourceError } from "@/lib/types";

/**
 * Event payload published on the ``project:<id>`` WebSocket stream as
 * the ``data`` field of a ``project_event`` envelope.
 *
 * ``resource`` is the short label of the resource that changed
 * (``optimization``, ``eval_job``, ``agent_turn``, ``filesystem``).
 * ``type`` is the change class — ``error``, ``completed``, ``changed``,
 * ``status``, etc. For errors, the ``ResourceError`` fields are spread
 * into the same object so the toast hook can render without a
 * secondary fetch.
 */
export interface BackendEvent extends Partial<ResourceError> {
  resource: string;
  type: string;
  /** Payloads frequently include a resource-specific id. */
  job_id?: string;
  job_type?: string;
  turn_id?: string;
  tool?: string;
  output_file?: string;
  /** Question text when a job enters awaiting_user state. */
  question?: string;
  /** Turn origin — "user" or "orchestrator". */
  origin?: string;
  /** Reason slug for terminal events (optimizer_stalled, plateau, etc.). */
  reason?: string;
  /** Report scope ("run" | "executive") on ``report/updated`` events. */
  scope?: string;
  /** Run filename on ``report/updated`` scope=run and ``eval_run/deleted`` events. */
  run_filename?: string;
}

const RESOURCE_LABELS: Record<string, string> = {
  optimization: "Optimization",
  eval_job: "Eval job",
  eval_run: "Eval run",
  job: "Job",
  agent_turn: "Chat turn",
  filesystem: "Project files",
};

function humanResource(name: string): string {
  return RESOURCE_LABELS[name] ?? name;
}

/**
 * Consumer classification for project-event subscriptions.
 *
 * **Momentary** consumers treat an event as "this is happening right now" —
 * they arm a spinner, open a subscription, or fire a toast. A buffered
 * envelope from minutes ago is actively wrong for these consumers, so they
 * ignore replayed envelopes.
 *
 * **State** consumers treat an event as "something changed, refetch it" — a
 * late envelope still carries correct information, which is why replay exists
 * at all. These consumers receive both live and replayed envelopes.
 */
export type ProjectEventConsumer = "momentary" | "state";

/**
 * In-process pub/sub target for project-scoped backend events. The
 * ``useBackendEvents`` hook (mounted once at app level) dispatches
 * incoming WebSocket payloads here; component-level ``useProjectEvent``
 * hooks subscribe. Using an ``EventTarget`` — rather than a React
 * context — means consumers don't re-render when unrelated events fire.
 */
const projectEventTarget = new EventTarget();
/** Name of the ``CustomEvent`` carried through ``projectEventTarget``. */
const PROJECT_EVENT = "project-event";

/**
 * Mount one WebSocket subscription per project and relay events to
 * both the toast layer and the in-process dispatcher. Call this at
 * the app layout level only; redundant calls would still work (they
 * subscribe to the same stream), but produce duplicate toasts.
 */
export function useBackendEvents(projectId: string | null): void {
  useProjectStream(projectId, (envelope: Envelope) => {
    if (envelope.kind !== "project_event") {
      return;
    }

    /* The envelope's ``data`` field carries the project-bus shape
     * (``{resource, type, ...}``) — see ``ws_hub.publish_project`` on
     * the backend, which wraps each call into a ``project_event``
     * envelope without restructuring the payload. */
    const payload = envelope.data as BackendEvent | null;
    if (!payload || typeof payload !== "object") {
      return;
    }

    /* Read the replay marker from the envelope (set by the server on buffered
     * envelopes). The marker is omitted on live envelopes, so absence means live. */
    const replayed = envelope.replayed === true;

    /* Fan out to in-process subscribers. The detail becomes a wrapper holding
     * the payload and the replay flag so useProjectEvent can filter by consumer
     * classification. */
    projectEventTarget.dispatchEvent(
      new CustomEvent<{ payload: BackendEvent; replayed: boolean }>(PROJECT_EVENT, {
        detail: { payload, replayed },
      }),
    );


    /* Guard all three toast paths on the replay marker. Momentary toasts
     * announce something that just happened — a replayed one is stale by
     * construction. Suppressing a replayed error loses nothing durable: the
     * inline resource-error banner is fed by the activity poll's server state,
     * so a still-relevant error is re-rendered on every mount and only the
     * transient toast is dropped. */
    if (replayed) return;

    /* "error" is the event-type most publishers use; the job executors
     * (eval, harness_build) publish their terminal failures as
     * ``{type: "failed"}`` with the same ResourceError fields spread in.
     * Before this matched both, a failed harness build reached no
     * subscriber and no toast — the failure was only discoverable in the
     * chat transcript. */
    if ((payload.type === "error" || payload.type === "failed") && payload.code && payload.message) {
      /* The toast is a one-line attention-grabber — the full
       * description lives on the inline ``<ResourceErrorBanner>`` on
       * the owning panel so we don't duplicate the same paragraph in
       * two places. Manual-dismiss only: a backend error is never a
       * passing status update; auto-hiding it would let the user
       * return to a silently-broken session.
       *
       * Stable id keyed on ``resource:code`` collapses the duplicate
       * publishes some paths emit by design (e.g. the optimization
       * loop crash branch in dispatch.py publishes both an ``error``
       * envelope for the toast AND a follow-up ``terminal`` envelope
       * for OptimizationStateProvider — sonner replaces a toast with
       * an existing id rather than stacking, so the second envelope
       * updates the first in place). */
      const errorId = `error:${payload.resource}:${payload.code}`;
      toast.error(`${humanResource(payload.resource)}: ${payload.message}`, {
        id: errorId,
        duration: Infinity,
      });
    }

    /* Surface "awaiting_user" job events as info toasts so the user
     * knows the system needs their input — even if they're on another
     * tab when it fires. */
    if (payload.type === "awaiting_user" && payload.resource === "job") {
      toast.info(
        payload.question
          ? `Input needed: ${payload.question.slice(0, 120)}`
          : "A job needs your input — check the chat.",
        { duration: 15000 },
      );
    }

    /* Terminal end states. A toast is for something that ended WITHOUT
     * the user driving it — convergence, plateau, max-iterations, an
     * optimizer stall, a crash, or a run orphaned by a coordinator
     * restart. The user should learn about those even if they're on
     * another tab.
     *
     * A user-initiated cancel (reason === "cancelled") is deliberately
     * NOT toasted: the user clicked Stop, so they already know. The bar
     * flipping to its resumable state IS the feedback. This is also why
     * there is no duplicate-cancel toast anymore — the cancel flow has
     * two terminal publishers by design (``api/optimization.py`` flips
     * the bar the instant Stop is clicked; ``mcp_tools/dispatch.py``
     * confirms it when the orchestrator loop actually exits, often >10s
     * later). Both still drive the bar refetch via
     * ``OptimizationStateProvider``'s own ``terminal`` subscription;
     * neither produces a toast, so the timing gap between them can't
     * surface as two toasts. */
    if (payload.type === "terminal" && payload.message && payload.reason !== "cancelled") {
      const text = `${humanResource(payload.resource)}: ${payload.message}`;
      if (payload.reason === "target_reached") {
        toast.success(text, { duration: 10000 });
      } else {
        toast.warning(text, { duration: 10000 });
      }
    }
  });
}

/**
 * Component-level subscription to backend events filtered by resource
 * (and optionally by event type). The handler fires for every matching
 * payload; it should be lightweight (e.g. a refetch trigger) — anything
 * heavier belongs in the corresponding query layer.
 *
 * The ``consumer`` classification is REQUIRED (not defaulted): a consumer added
 * later without classifying itself would silently inherit the wrong replay
 * behavior. TypeScript refuses to compile a call site that omits this
 * parameter, so the enforcement is a build gate rather
 * than a convention someone has to read. It also forces the author to do the
 * actual conceptual work — deciding what kind of consumer this is — instead of
 * describing a mechanism.
 *
 * Stable dependency story: handlers are intentionally not part of the
 * effect deps array — we keep the latest reference via a ref so that
 * callers can pass inline functions without re-subscribing on every
 * render. This matches the ergonomics of addEventListener.
 *
 * @param consumer - "momentary" for actions happening right now (spinner, toast,
 *                   subscription); "state" for refetch triggers. Momentary
 *                   consumers ignore replayed envelopes.
 */
export function useProjectEvent(
  match: { resource: string; type?: string },
  handler: (event: BackendEvent) => void,
  consumer: ProjectEventConsumer,
): void {
  /* Keep the latest handler in a ref so we can re-subscribe only when
   * the match predicate changes. Callers can pass inline closures —
   * the ref always points at the freshest version, so stale-closure
   * bugs don't surface here. */
  const handlerRef = useRef(handler);
  handlerRef.current = handler;

  useEffect(() => {
    const onEvent = (ev: Event) => {
      const detail = (ev as CustomEvent<{ payload: BackendEvent; replayed: boolean }>).detail;
      if (!detail) return;

      const { payload, replayed } = detail;
      if (!payload) return;
      if (payload.resource !== match.resource) return;
      if (match.type && payload.type !== match.type) return;

      /* Skip replayed envelopes for momentary consumers. State consumers
       * receive both live and replayed envelopes — a late envelope still
       * carries correct information (something changed, refetch it), which is
       * why replay exists at all. */
      if (replayed && consumer === "momentary") {
        return;
      }


      handlerRef.current(payload);
    };
    projectEventTarget.addEventListener(PROJECT_EVENT, onEvent);
    return () => projectEventTarget.removeEventListener(PROJECT_EVENT, onEvent);
  }, [match.resource, match.type, consumer]);
}

/**
 * React hook to fire a callback when a cursor gap is detected on a project stream.
 *
 * Fires on any of three conditions:
 *  - `kind: "replay_gap"` — server cannot serve the requested `since_seq` (restart,
 *    TTL, cap eviction)
 *  - `degraded` marker — server-side durable seq allocation failed
 *  - client-derived `gap` field — forward hole on the live socket (seq > cursor + 1)
 *
 * **Scoping decision:** This applies to PROJECT streams only. The scoping is structural
 * because this hook is the only subscriber to the signal and it subscribes to
 * `project:{id}` by construction. Project publishes serialize under a per-project lock
 * and draw from one durable counter, so a hole there is real. Turn and job streams
 * legitimately mix seq domains — `agent_runner` mints turn envelopes from the in-memory
 * counter, `jobs.__init__` falls back to the in-memory counter when a durable allocation
 * fails, and the domain-reset heuristic exists precisely because of that — so gap
 * detection there would fire constantly and has no consumer.
 *
 * **Bounded by design:** Exactly one signal per hole. The cursor advances on the frame
 * that carried the gap, so the next contiguous frame reports no gap. A refetch is
 * HTTP-only and cannot itself produce envelopes, so no refetch-per-frame loop is
 * reachable.
 *
 * @param projectId - Project GUID. Pass `null` to disable.
 * @param onGap - Callback invoked when any of the three gap conditions is detected.
 *                Typically a silent HTTP refetch (e.g., `refreshRunsSilently`).
 */
export function useCursorGapHandler(projectId: string | null, onGap: () => void) {
  const onGapRef = useRef(onGap);
  onGapRef.current = onGap;

  useEffect(() => {
    if (!projectId) return;

    const stream = `project:${projectId}`;
    const handler = (envelope: Envelope) => {
      // Fire on any of three conditions: replay_gap, degraded, or client-derived gap
      const isGap =
        envelope.kind === "replay_gap" ||
        envelope.degraded === true ||
        (envelope.gap !== undefined);

      if (isGap && envelope.stream === stream) {
        onGapRef.current();
      }
    };

    // ``subscribe`` returns its own unsubscribe closure — use it rather than a
    // separate removal call, so only this listener is detached.
    return wsClient.subscribe(stream, handler);
  }, [projectId]);
}
