// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Header user menu (cloud mode only).
 *
 * Renders a person-icon button that opens a popover showing the signed-in
 * user's email and a Sign-out action. Replaces the bare logout button so
 * logging out is a deliberate two-step action and the user can see which
 * account they're in.
 */

import { useEffect, useState } from "react";
import { CircleUser, LogOut } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { getUserEmail, signOut } from "@/lib/auth";

/**
 * Account menu for the global header. Shows the current user's email and a
 * Sign-out item. Only rendered in cloud mode (the caller gates on
 * `isAuthRequired()`).
 */
export function UserMenu() {
  const [email, setEmail] = useState<string | null>(null);

  // Resolve the signed-in email once on mount. The OIDC user is already
  // loaded by AuthGate before the app shell renders, so this is a cheap
  // in-memory read in practice.
  useEffect(() => {
    let cancelled = false;
    void getUserEmail().then((e) => {
      if (!cancelled) setEmail(e);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <Popover>
      <PopoverTrigger
        render={
          <Button variant="ghost" size="icon" className="h-8 w-8" title="Account" />
        }
      >
        <CircleUser className="h-4 w-4" />
      </PopoverTrigger>
      <PopoverContent align="end" sideOffset={6} className="w-56 gap-0 p-0">
        <div className="px-3 py-2.5">
          <p className="text-2xs uppercase tracking-wider text-muted-foreground">
            Signed in as
          </p>
          <p className="mt-0.5 truncate text-sm font-medium" title={email ?? undefined}>
            {email ?? "Account"}
          </p>
        </div>
        <div className="border-t border-border p-1">
          <Button
            variant="ghost"
            size="sm"
            className="w-full justify-start gap-2"
            onClick={() => {
              void signOut();
            }}
          >
            <LogOut className="size-3.5" />
            Sign out
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  );
}
