// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * React bindings for the singleton WebSocket envelope client.
 *
 * Exposes a generic ``useWsStream(stream, onEnvelope)`` hook plus thin
 * wrappers for the three stream prefixes the SPA cares about:
 *  - ``useChatStream(turnId)`` — envelopes for an in-flight chat turn.
 *  - ``useJobStream(jobId)`` — progress/done for a background eval job.
 *  - ``useProjectStream(projectId)`` — project-scoped resource events.
 *
 * The handler identity is held in a ref so callers can pass inline
 * closures without re-subscribing on every render.
 */

import { useEffect, useRef } from "react";

import { wsClient, type Envelope } from "@/lib/ws-client";

/**
 * Subscribe a listener to a specific WebSocket stream.
 *
 * Re-subscribes only when ``stream`` changes, even if ``onEnvelope`` is
 * a new function reference each render. Pass ``null`` to disable
 * (e.g. while waiting for an id to resolve).
 *
 * @param stream - Stream id, or ``null`` to skip subscribing.
 * @param onEnvelope - Callback invoked for every envelope on the
 *   matching stream. The ref pattern keeps the latest version live so
 *   the callback identity doesn't matter.
 */
export function useWsStream(
  stream: string | null,
  onEnvelope: (env: Envelope) => void,
): void {
  const handlerRef = useRef(onEnvelope);
  handlerRef.current = onEnvelope;

  useEffect(() => {
    if (!stream) return;
    const unsubscribe = wsClient.subscribe(stream, (env) => {
      handlerRef.current(env);
    });
    return unsubscribe;
  }, [stream]);
}

/**
 * Subscribe to envelopes for a specific chat turn.
 *
 * @param turnId - Active turn id, or ``null`` if no turn is in flight.
 * @param onEnvelope - Envelope handler.
 */
export function useChatStream(
  turnId: string | null,
  onEnvelope: (env: Envelope) => void,
): void {
  useWsStream(turnId ? `turn:${turnId}` : null, onEnvelope);
}

/**
 * Subscribe to progress / done envelopes for a specific eval job.
 *
 * @param jobId - Active job id, or ``null`` if no job is running.
 * @param onEnvelope - Envelope handler.
 */
export function useJobStream(
  jobId: string | null,
  onEnvelope: (env: Envelope) => void,
): void {
  useWsStream(jobId ? `job:${jobId}` : null, onEnvelope);
}

/**
 * Subscribe to project-scoped resource events (filesystem changes,
 * job state transitions, etc.) for a specific project.
 *
 * @param projectId - Active project id, or ``null`` while no project
 *   is selected.
 * @param onEnvelope - Envelope handler.
 */
export function useProjectStream(
  projectId: string | null,
  onEnvelope: (env: Envelope) => void,
): void {
  useWsStream(projectId ? `project:${projectId}` : null, onEnvelope);
}
