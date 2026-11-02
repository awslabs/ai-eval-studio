// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @vitest-environment jsdom
 *
 * The project name and description are edited in place in the page header. Both
 * were display-only before, so a project kept whatever label it was given in the
 * create dialog forever.
 *
 * The cases that matter are the ones where a naive implementation loses the
 * user's text: a failed save, a background refetch landing mid-edit, and a blank
 * value in a field that does not accept one. Each is asserted on the rendered
 * result rather than on internal state.
 */

import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { EditableText } from "./EditableText";

const toastError = vi.fn();
vi.mock("sonner", () => ({ toast: { error: (m: string) => toastError(m) } }));

afterEach(() => {
  cleanup();
  toastError.mockReset();
});

/** Click the edit button and return the resulting input. */
function openEditor(label: string): HTMLInputElement {
  fireEvent.click(screen.getByRole("button", { name: `Edit ${label.toLowerCase()}` }));
  return screen.getByRole("textbox", { name: label }) as HTMLInputElement;
}

describe("EditableText", () => {
  it("shows the value and no input until the edit button is clicked", () => {
    render(<EditableText value="Support Triage v2" label="Name" onSave={vi.fn()} />);

    expect(screen.getByText("Support Triage v2")).toBeInTheDocument();
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  });

  it("shows the placeholder when the value is empty", () => {
    render(
      <EditableText
        value=""
        label="Description"
        placeholder="Add a description"
        onSave={vi.fn()}
      />,
    );

    expect(screen.getByText("Add a description")).toBeInTheDocument();
  });

  it("saves the edited value on Enter", async () => {
    const onSave = vi.fn().mockResolvedValue(undefined);
    render(<EditableText value="Old" label="Name" onSave={onSave} />);

    const input = openEditor("Name");
    fireEvent.change(input, { target: { value: "New" } });
    fireEvent.keyDown(input, { key: "Enter" });

    await waitFor(() => expect(onSave).toHaveBeenCalledWith("New"));
  });

  it("saves on blur, so clicking away keeps the edit", async () => {
    const onSave = vi.fn().mockResolvedValue(undefined);
    render(<EditableText value="Old" label="Name" onSave={onSave} />);

    const input = openEditor("Name");
    fireEvent.change(input, { target: { value: "Clicked away" } });
    fireEvent.blur(input);

    await waitFor(() => expect(onSave).toHaveBeenCalledWith("Clicked away"));
  });

  it("discards the edit on Escape without saving", async () => {
    const onSave = vi.fn().mockResolvedValue(undefined);
    render(<EditableText value="Original" label="Name" onSave={onSave} />);

    const input = openEditor("Name");
    fireEvent.change(input, { target: { value: "Thrown away" } });
    fireEvent.keyDown(input, { key: "Escape" });

    expect(onSave).not.toHaveBeenCalled();
    await waitFor(() => expect(screen.getByText("Original")).toBeInTheDocument());
  });

  it("does not call onSave when the value is unchanged", async () => {
    const onSave = vi.fn().mockResolvedValue(undefined);
    render(<EditableText value="Same" label="Name" onSave={onSave} />);

    fireEvent.keyDown(openEditor("Name"), { key: "Enter" });

    await waitFor(() => expect(screen.getByText("Same")).toBeInTheDocument());
    expect(onSave).not.toHaveBeenCalled();
  });

  it("refuses a blank value when allowEmpty is false and keeps the editor open", async () => {
    const onSave = vi.fn().mockResolvedValue(undefined);
    render(<EditableText value="Named" label="Name" allowEmpty={false} onSave={onSave} />);

    const input = openEditor("Name");
    fireEvent.change(input, { target: { value: "   " } });
    fireEvent.keyDown(input, { key: "Enter" });

    await waitFor(() => expect(toastError).toHaveBeenCalledWith("Name cannot be empty"));
    expect(onSave).not.toHaveBeenCalled();
    // Still editing — the user has not lost their cursor or their text.
    expect(screen.getByRole("textbox", { name: "Name" })).toBeInTheDocument();
  });

  it("saves a blank value when allowEmpty is true, so a description can be cleared", async () => {
    const onSave = vi.fn().mockResolvedValue(undefined);
    render(<EditableText value="Remove me" label="Description" onSave={onSave} />);

    const input = openEditor("Description");
    fireEvent.change(input, { target: { value: "" } });
    fireEvent.keyDown(input, { key: "Enter" });

    await waitFor(() => expect(onSave).toHaveBeenCalledWith(""));
  });

  it("keeps the typed value in the editor when the save fails", async () => {
    const onSave = vi.fn().mockRejectedValue(new Error("boom"));
    render(<EditableText value="Original" label="Name" onSave={onSave} />);

    const input = openEditor("Name");
    fireEvent.change(input, { target: { value: "Attempted" } });
    fireEvent.keyDown(input, { key: "Enter" });

    await waitFor(() => expect(toastError).toHaveBeenCalled());
    // Dropping to display here would show "Original" and silently discard the
    // typed text, which reads as if the edit had been accepted.
    const stillEditing = screen.getByRole("textbox", { name: "Name" }) as HTMLInputElement;
    expect(stillEditing.value).toBe("Attempted");
  });

  it("does not overwrite an in-progress edit when the persisted value changes", () => {
    const { rerender } = render(
      <EditableText value="First" label="Name" onSave={vi.fn()} />,
    );

    const input = openEditor("Name");
    fireEvent.change(input, { target: { value: "Typing..." } });

    // A background refetch resolves with a different persisted value.
    rerender(<EditableText value="Refetched" label="Name" onSave={vi.fn()} />);

    expect((screen.getByRole("textbox", { name: "Name" }) as HTMLInputElement).value).toBe(
      "Typing...",
    );
  });

  it("adopts a new persisted value while not editing", () => {
    const { rerender } = render(
      <EditableText value="First" label="Name" onSave={vi.fn()} />,
    );
    rerender(<EditableText value="Second" label="Name" onSave={vi.fn()} />);

    expect(screen.getByText("Second")).toBeInTheDocument();
  });
});
