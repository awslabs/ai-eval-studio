// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Themed wrapper around sonner's <Toaster>, following the shadcn/ui v4
 * recipe: pass theme tokens through sonner's ``style`` prop as inline
 * CSS custom properties so the library's built-in theme rules lose the
 * specificity battle.
 *
 * Also moves the close button from sonner's default top-left anchor to
 * the top-right of the card via a scoped ``<style>`` override keyed on
 * the library's own ``data-close-button`` attribute.
 */

import { Toaster as SonnerToaster, type ToasterProps } from "sonner";
import { useTheme } from "@/lib/theme-provider";

/** Scoped CSS: move the close button from top-left to top-right, tucked
 *  tight against the edge. */
const TOASTER_CSS = `
[data-sonner-toaster] [data-sonner-toast] [data-close-button] {
  left: auto !important;
  right: 4px !important;
  top: 4px !important;
  transform: none !important;
}
`;

/** Themed Toaster. Drop this at the app root in place of sonner's raw <Toaster>. */
export function ThemedToaster() {
  const { mode } = useTheme();

  return (
    <>
      <style>{TOASTER_CSS}</style>
      <SonnerToaster
        theme={mode as ToasterProps["theme"]}
        position="top-center"
        visibleToasts={5}
        closeButton
        richColors
        className="toaster group"
        style={
          {
            "--normal-bg": "var(--popover)",
            "--normal-text": "var(--popover-foreground)",
            "--normal-border": "var(--border)",
            "--border-radius": "var(--radius)",
          } as React.CSSProperties
        }
      />
    </>
  );
}
