// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Recursive file tree component for browsing project files.
 * Renders a collapsible directory tree with file type icons
 * and selection highlighting.
 */

import { useState } from "react";
import { ChevronRight, Folder, FolderOpen, FileCode, FileText, FileJson } from "lucide-react";
import type { FileNode } from "@/lib/types";

/** Props for the FileTree component. */
interface FileTreeProps {
  /** Array of file/directory nodes to render */
  nodes: FileNode[];
  /** Currently selected file path (highlighted) */
  selectedPath: string | null;
  /** Callback when a file is clicked */
  onSelect: (path: string) => void;
  /**
   * Directory paths to start expanded in addition to the top-level nodes.
   * Used after an upload to reveal ``data/uploads/`` without surprising
   * the user by auto-selecting a file.
   */
  expandedPaths?: string[];
}

/**
 * Renders a recursive file tree with expand/collapse directories
 * and click-to-select files.
 *
 * @param nodes - Top-level file tree nodes
 * @param selectedPath - Path of the currently selected file
 * @param onSelect - Called when the user clicks a file node
 * @param expandedPaths - Optional directory paths to start expanded
 */
export function FileTree({ nodes, selectedPath, onSelect, expandedPaths }: FileTreeProps) {
  const expandedSet = new Set(expandedPaths ?? []);
  return (
    <div className="text-sm">
      {nodes.map((node) => (
        <FileTreeNode
          key={node.path}
          node={node}
          selectedPath={selectedPath}
          onSelect={onSelect}
          depth={0}
          defaultExpanded
          expandedPaths={expandedSet}
        />
      ))}
    </div>
  );
}

/** Props for a single file tree node (internal). */
interface FileTreeNodeProps {
  /** The file or directory node */
  node: FileNode;
  /** Currently selected file path */
  selectedPath: string | null;
  /** Selection callback */
  onSelect: (path: string) => void;
  /** Nesting depth for indentation */
  depth: number;
  /** Whether directories start expanded */
  defaultExpanded?: boolean;
  /** Set of directory paths to start expanded (passed through from FileTree). */
  expandedPaths?: Set<string>;
}

/**
 * Renders a single node in the file tree.
 * Directories are collapsible; files are selectable.
 *
 * @param node - File or directory node
 * @param selectedPath - Currently selected path for highlighting
 * @param onSelect - Click handler for file selection
 * @param depth - Current nesting depth
 * @param defaultExpanded - Whether to start expanded (true for first level)
 * @param expandedPaths - Extra directory paths to start expanded
 */
function FileTreeNode({ node, selectedPath, onSelect, depth, defaultExpanded = false, expandedPaths }: FileTreeNodeProps) {
  const [expanded, setExpanded] = useState(
    defaultExpanded || (expandedPaths?.has(node.path) ?? false),
  );
  const isDir = node.type === "dir";
  const isSelected = node.path === selectedPath;

  /** Handle click on a tree node. */
  const handleClick = () => {
    if (isDir) {
      setExpanded((prev) => !prev);
    } else {
      onSelect(node.path);
    }
  };

  /** Pick the appropriate icon based on file type and extension. */
  const getIcon = () => {
    if (isDir) {
      return expanded
        ? <FolderOpen className="size-4 text-muted-foreground shrink-0" />
        : <Folder className="size-4 text-muted-foreground shrink-0" />;
    }
    if (node.name.endsWith(".py")) {
      return <FileCode className="size-4 text-muted-foreground shrink-0" />;
    }
    if (node.name.endsWith(".json") || node.name.endsWith(".jsonl")) {
      return <FileJson className="size-4 text-muted-foreground shrink-0" />;
    }
    return <FileText className="size-4 text-muted-foreground shrink-0" />;
  };

  return (
    <div>
      <button
        onClick={handleClick}
        aria-current={isSelected || undefined}
        className={`flex items-center gap-1.5 w-full text-left py-1 px-1.5 rounded-sm transition-colors outline-none hover:bg-accent/50 focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-ring ${
          isSelected ? "bg-accent text-accent-foreground" : ""
        }`}
        style={{ paddingLeft: `${depth * 16 + 6}px` }}
      >
        {isDir && (
          <ChevronRight
            className={`size-3.5 text-muted-foreground shrink-0 transition-transform ${
              expanded ? "rotate-90" : ""
            }`}
          />
        )}
        {!isDir && <span className="w-3.5 shrink-0" />}
        {getIcon()}
        <span className="truncate">{node.name}</span>
      </button>
      {isDir && expanded && node.children && (
        <div>
          {node.children.map((child) => (
            <FileTreeNode
              key={child.path}
              node={child}
              selectedPath={selectedPath}
              onSelect={onSelect}
              depth={depth + 1}
              expandedPaths={expandedPaths}
            />
          ))}
        </div>
      )}
    </div>
  );
}
