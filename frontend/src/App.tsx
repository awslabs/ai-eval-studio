// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Root application component.
 * Provides routing, theme, chat panel, and the main layout with sidebar navigation.
 */

import { useEffect, useRef, useState } from "react";
import { BrowserRouter, Routes, Route, Link, useLocation } from "react-router-dom";
import {
  FlaskConical,
  Settings,
  Sun,
  Moon,
  MessageSquare,
  PanelLeftClose,
  PanelLeftOpen,
  Info,
} from "lucide-react";
import { cn } from "@/lib/class-utils";
import { ThemeProvider, useTheme } from "@/lib/theme-provider";
import { ThemedToaster } from "@/components/ui/themed-toaster";
import { ChatProvider, useChat } from "@/lib/chat-context";
import { ChatPanel } from "@/components/chat/ChatPanel";
import { Button } from "@/components/ui/button";
import { useBackendEvents } from "@/hooks/useBackendEvents";
import { HomePage } from "@/pages/HomePage";
import { ProjectPage } from "@/pages/ProjectPage";
import { SettingsPage } from "@/pages/SettingsPage";
import { AboutPage } from "@/pages/AboutPage";
import { isAuthRequired } from "@/lib/auth";
import { UserMenu } from "@/components/UserMenu";

/** localStorage key for the persisted left-nav collapsed state. */
const LS_NAV_COLLAPSED = "eval-studio-nav-collapsed";

/** Sidebar navigation link with active state highlighting. */
function NavLink({
  to,
  icon: Icon,
  label,
  collapsed,
}: {
  to: string;
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  /** When true, render icon-only (label hidden) with a tooltip. */
  collapsed: boolean;
}) {
  const location = useLocation();
  const isActive = location.pathname === to;

  return (
    <Link
      to={to}
      title={collapsed ? label : undefined}
      className={cn(
        "relative flex items-center gap-2 rounded-md py-2 text-sm transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring",
        collapsed ? "justify-center px-0" : "px-3",
        isActive
          ? "bg-sidebar-accent text-sidebar-accent-foreground font-medium before:absolute before:left-0 before:inset-y-1.5 before:w-0.5 before:rounded-full before:bg-primary"
          : "text-sidebar-foreground/70 hover:text-sidebar-foreground hover:bg-sidebar-accent/50",
      )}
    >
      <Icon className="size-4 shrink-0" />
      {!collapsed && label}
    </Link>
  );
}

/** Top bar with branding, project-scoped chat toggle, theme toggle, and (cloud-mode) sign-out. */
function TopBar({ hasActiveProject }: { hasActiveProject: boolean }) {
  const { mode, toggleMode } = useTheme();
  const { isOpen, toggle } = useChat();
  // `isAuthRequired()` reads cached `/api/config` set during `AuthGate`
  // bootstrap. False in dev mode -> hide the button entirely.
  const showSignOut = isAuthRequired();

  return (
    <header className="h-12 shrink-0 border-b border-border bg-card shadow-sm flex items-center justify-between px-4">
      <div className="flex items-center gap-2">
        <img src="/favicon.svg" alt="" className="size-8" />
        <h1 className="text-base font-semibold tracking-tight">AI Eval Studio</h1>
      </div>
      <div className="flex items-center gap-1">
        {/* Chat toggle — only shown when viewing a project */}
        {hasActiveProject && (
          <Button
            variant={isOpen ? "secondary" : "ghost"}
            size="icon"
            className="h-8 w-8"
            onClick={toggle}
            title={isOpen ? "Close chat" : "Open chat"}
          >
            <MessageSquare className="h-4 w-4" />
          </Button>
        )}
        {/* Theme mode toggle */}
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8"
          onClick={toggleMode}
          title={mode === "dark" ? "Switch to light mode" : "Switch to dark mode"}
        >
          {mode === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </Button>
        {/* Account menu — cloud mode only (shows email + Sign out). Hidden
            in dev mode since there's no session to manage. */}
        {showSignOut && <UserMenu />}
      </div>
    </header>
  );
}

/** Extract projectId from URL if on a project page. */
function useCurrentProjectId(): string | undefined {
  const location = useLocation();
  const match = location.pathname.match(/^\/projects\/([^/]+)/);
  return match?.[1];
}

/** Main application layout with top bar, sidebar, and project-scoped chat panel. */
function AppLayout() {
  const projectId = useCurrentProjectId();
  const { open, close } = useChat();

  // Left-nav collapse state, persisted across reloads (mirrors the theme
  // provider's localStorage pattern). Collapsed = icon-only rail.
  const [navCollapsed, setNavCollapsed] = useState(
    () => localStorage.getItem(LS_NAV_COLLAPSED) === "1",
  );
  const toggleNav = () => {
    setNavCollapsed((prev) => {
      const next = !prev;
      localStorage.setItem(LS_NAV_COLLAPSED, next ? "1" : "0");
      return next;
    });
  };

  // Subscribe to the project-scoped backend event bus so failures in
  // background work surface as toasts regardless of which tab is open.
  // Inline banners on the owning panels remain the primary error
  // surface — this is the safety net for when the user is elsewhere.
  useBackendEvents(projectId ?? null);

  // Auto-open chat when entering a project, auto-close when leaving
  const prevProjectIdRef = useRef(projectId);
  useEffect(() => {
    if (prevProjectIdRef.current && !projectId) {
      close();
    } else if (projectId) {
      open();
    }
    prevProjectIdRef.current = projectId;
  }, [projectId, open, close]);

  return (
    <div className="flex flex-col h-screen">
      <TopBar hasActiveProject={!!projectId} />
      <div className="flex flex-1 min-h-0">
        {/* Sidebar navigation — collapsible to an icon-only rail */}
        <nav
          className={cn(
            "shrink-0 border-r border-border bg-sidebar text-sidebar-foreground p-3 flex flex-col gap-1 transition-[width] duration-200",
            navCollapsed ? "w-14" : "w-48",
          )}
        >
          <NavLink to="/" icon={FlaskConical} label="Projects" collapsed={navCollapsed} />
          <NavLink to="/settings" icon={Settings} label="Settings" collapsed={navCollapsed} />
          <NavLink to="/about" icon={Info} label="About" collapsed={navCollapsed} />
          {/* Collapse toggle — pinned to the bottom (mt-auto), icon-only.
              Right-aligned when expanded; centered on the collapsed rail. */}
          <div className={cn("mt-auto flex pt-1", navCollapsed ? "justify-center" : "justify-end")}>
            <button
              type="button"
              onClick={toggleNav}
              title={navCollapsed ? "Expand sidebar" : "Collapse sidebar"}
              aria-label={navCollapsed ? "Expand sidebar" : "Collapse sidebar"}
              className="flex size-7 items-center justify-center rounded-md text-sidebar-foreground/70 transition-colors outline-none hover:text-sidebar-foreground hover:bg-sidebar-accent/50 focus-visible:ring-2 focus-visible:ring-ring"
            >
              {navCollapsed ? (
                <PanelLeftOpen className="size-4 shrink-0" />
              ) : (
                <PanelLeftClose className="size-4 shrink-0" />
              )}
            </button>
          </div>
        </nav>

        {/* Main content + chat panel (chat only renders on project pages) */}
        <div className="flex flex-1 min-w-0">
          <main className="flex-1 overflow-auto">
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/projects/:projectId" element={<ProjectPage />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="/about" element={<AboutPage />} />
            </Routes>
          </main>

          {/* Chat panel — only rendered when viewing a project */}
          {projectId && <ChatPanel projectId={projectId} />}
        </div>
      </div>
    </div>
  );
}

/** Root component wrapping all providers and router. */
export default function App() {
  return (
    <BrowserRouter>
      <ThemeProvider>
        <ChatProvider>
          <AppLayout />
          <ThemedToaster />
        </ChatProvider>
      </ThemeProvider>
    </BrowserRouter>
  );
}
