// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * About page — app name, description, and version.
 *
 * Static informational page reached from the left-nav "About" item. The
 * version is injected at build time from package.json (see vite.config.ts).
 */

import { FlaskConical } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

/** Informational About page describing AI Eval Studio. */
export function AboutPage() {
  return (
    <div className="p-8 max-w-2xl mx-auto space-y-4">
      <h1 className="text-2xl font-bold tracking-tight">About</h1>

      <Card>
        <CardContent className="space-y-4">
          <div className="flex items-center gap-3">
            <div className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-primary/10">
              <FlaskConical className="size-5 text-primary" />
            </div>
            <div>
              <p className="text-base font-semibold leading-tight">AI Eval Studio</p>
              <p className="text-xs text-muted-foreground tabular-nums">
                Version {__APP_VERSION__}
              </p>
            </div>
          </div>

          <p className="text-sm leading-relaxed text-muted-foreground">
            AI Eval Studio is an end-to-end model evaluation platform that helps
            you choose the right LLM with an optimized prompt based on your test
            data. It combines a conversational AI interface with automated
            evaluation workflows, iterative prompt optimization, and rich
            analytics.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
