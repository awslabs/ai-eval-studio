// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Utility functions for the frontend application.
 */

import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

/**
 * Merge Tailwind CSS class names intelligently.
 * Combines clsx for conditional classes with tailwind-merge to resolve conflicts.
 *
 * @param inputs - Class names, conditionals, or arrays
 * @returns Merged class string with resolved Tailwind conflicts
 *
 * @example
 * cn("px-2 py-1", condition && "bg-primary", { "font-bold": isActive })
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
