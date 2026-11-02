// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * The ONE frontend mirror of backend-owned default values.
 *
 * Contract: each constant here names its single backend owner. If a value
 * ever needs a second frontend copy, import from here — never re-type the
 * literal. If the backend owner changes, this file is the only frontend
 * edit.
 */

/**
 * Plateau spread threshold applied by the backend when the harness config
 * doesn't set one.
 *
 * Backend owner: ``app/services/optimization_state.py`` —
 * ``DEFAULT_PLATEAU_THRESHOLD``. (Unlike plateau_window/max_iterations,
 * which the backend REQUIRES and never defaults, the threshold is
 * a genuine backend default, so rendering it for an unset config reports
 * the effective value rather than inventing one.)
 */
export const DEFAULT_PLATEAU_THRESHOLD = 0.02;
