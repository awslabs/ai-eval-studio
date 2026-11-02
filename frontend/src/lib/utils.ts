// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Re-export of the ``cn`` helper from ``class-utils`` under the
 * conventional ``@/lib/utils`` path that shadcn / assistant-ui CLI
 * components expect.
 *
 * Our canonical source is ``class-utils.ts`` (per the naming
 * conventions doc), but the CLI installs components hardcoded to
 * ``@/lib/utils``. This shim means ``npx assistant-ui add …`` and
 * ``npx shadcn add …`` work without a post-install patch step.
 */
export { cn } from "./class-utils";
