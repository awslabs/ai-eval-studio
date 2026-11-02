// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Chat panel state management context.
 * Controls open/close and full-screen state. Chat auto-opens when
 * entering a project page and auto-closes when navigating away.
 */

import {
  createContext,
  useCallback,
  useContext,
  useState,
  type ReactNode,
} from "react";

/** Chat panel state and controls available to all components. */
interface ChatContextValue {
  /** Whether the chat panel is visible */
  isOpen: boolean;
  /** Whether the chat panel is in full-screen mode (covers main content) */
  isFullScreen: boolean;
  /** Toggle chat panel open/closed */
  toggle: () => void;
  /** Open chat panel */
  open: () => void;
  /** Close chat panel (also exits full-screen) */
  close: () => void;
  /** Toggle full-screen mode */
  toggleFullScreen: () => void;
}

const ChatContext = createContext<ChatContextValue | null>(null);

export function ChatProvider({ children }: { children: ReactNode }) {
  const [isOpen, setIsOpen] = useState(true);
  const [isFullScreen, setIsFullScreen] = useState(false);

  const toggle = useCallback(() => setIsOpen((prev) => !prev), []);
  const open = useCallback(() => setIsOpen(true), []);
  const close = useCallback(() => {
    setIsOpen(false);
    setIsFullScreen(false);
  }, []);
  const toggleFullScreen = useCallback(() => setIsFullScreen((prev) => !prev), []);

  return (
    <ChatContext.Provider value={{ isOpen, isFullScreen, toggle, open, close, toggleFullScreen }}>
      {children}
    </ChatContext.Provider>
  );
}

/** Hook to access chat panel state. Must be used within ChatProvider. */
export function useChat() {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error("useChat must be used within ChatProvider");
  return ctx;
}
