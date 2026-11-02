// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * File-path formatter for tool-call displays.
 *
 * Chat pills only have ~300px for the detail, and the full absolute path is
 * noise the reader has to skim past to find what matters: the file name.
 * Returns just the basename (e.g. ``baseline.md``) — the caller should
 * preserve the raw path in a ``title`` attribute so hover still reveals
 * the full string for disambiguation.
 */
export function formatPath(rawPath: string): string {
  if (!rawPath) return rawPath;
  const basename = rawPath.split("/").filter(Boolean).pop();
  return basename || rawPath;
}
