// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Modified by Amazon. Original source: https://github.com/shadcn-ui/ui
// Portions Copyright (c) 2023 shadcn, MIT License. See THIRD-PARTY-LICENSES.

/**
 * Checkbox component built on @base-ui/react Checkbox primitive.
 * Supports checked, unchecked, and indeterminate states with proper styling.
 */

import { Checkbox as CheckboxPrimitive } from "@base-ui/react/checkbox";

import { cn } from "@/lib/class-utils";
import { CheckIcon, MinusIcon } from "lucide-react";

/** Styled checkbox with indeterminate support via the `indeterminate` prop. */
function Checkbox({
  className,
  indeterminate,
  ...props
}: CheckboxPrimitive.Root.Props) {
  return (
    <CheckboxPrimitive.Root
      data-slot="checkbox"
      indeterminate={indeterminate}
      className={cn(
        "peer relative flex size-4 shrink-0 items-center justify-center rounded-[4px] border border-input transition-colors outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 disabled:cursor-not-allowed disabled:opacity-50 data-checked:border-foreground/40 data-checked:bg-foreground/20 data-checked:text-foreground data-indeterminate:border-foreground/40 data-indeterminate:bg-foreground/20 data-indeterminate:text-foreground",
        className
      )}
      {...props}
    >
      <CheckboxPrimitive.Indicator
        data-slot="checkbox-indicator"
        className="grid place-content-center text-current transition-none [&>svg]:size-3.5"
      >
        {indeterminate ? <MinusIcon /> : <CheckIcon />}
      </CheckboxPrimitive.Indicator>
    </CheckboxPrimitive.Root>
  );
}

export { Checkbox };
