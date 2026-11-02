// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Settings page for application configuration.
 * Supports theme selection with color previews and a light/dark mode toggle.
 */

import { Moon, Sun, Check } from "lucide-react";
import { useTheme } from "@/lib/theme-provider";
import { themes } from "@/lib/themes";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
// Agent model is chosen per-project via the ModelSelector inline in the chat
// composer, so there's no global Agent Model control on this page.

/** Settings page with appearance and configuration options. */
export function SettingsPage() {
  const { mode, setMode, themeId, setThemeId } = useTheme();

  return (
    <div className="p-8 max-w-2xl mx-auto space-y-4">
      <h1 className="text-2xl font-bold tracking-tight mb-6">Settings</h1>

      {/* Appearance — combined mode + theme */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Appearance</CardTitle>
          <CardDescription>Choose how AI Eval Studio looks to you.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          {/* Mode toggle */}
          <div className="flex items-center gap-4">
            <span className="text-sm font-medium text-muted-foreground">Mode</span>
            <div className="flex gap-2">
              <Button variant={mode === "light" ? "default" : "outline"} size="sm" onClick={() => setMode("light")}>
                <Sun className="size-3.5 mr-1.5" /> Light
              </Button>
              <Button variant={mode === "dark" ? "default" : "outline"} size="sm" onClick={() => setMode("dark")}>
                <Moon className="size-3.5 mr-1.5" /> Dark
              </Button>
            </div>
          </div>

          {/* Theme picker with color swatches */}
          <div>
            <span className="text-sm font-medium text-muted-foreground">Theme</span>
            <div className="grid gap-3 sm:grid-cols-3 mt-2">
              {Object.entries(themes).map(([id, def]) => {
                const isActive = id === themeId;
                const preview = mode === "dark" ? def.dark : def.light;
                return (
                  <button
                    key={id}
                    onClick={() => setThemeId(id)}
                    className={`relative rounded-lg border-2 p-4 text-left transition-all outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background ${
                      isActive
                        ? "border-primary ring-2 ring-primary/20"
                        : "border-border hover:border-primary/40"
                    }`}
                  >
                    {isActive && (
                      <div className="absolute top-2 right-2 rounded-full bg-primary p-0.5">
                        <Check className="size-3 text-primary-foreground" />
                      </div>
                    )}
                    <div className="mb-3 font-medium text-sm text-foreground">{def.name}</div>
                    <div className="flex gap-1.5">
                      {["primary", "chart-1", "chart-2", "chart-3", "chart-4", "chart-5"].map((varName) => (
                        <div
                          key={varName}
                          className="size-5 rounded-full border border-border/50"
                          style={{ backgroundColor: preview[varName] }}
                        />
                      ))}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </CardContent>
      </Card>

    </div>
  );
}
