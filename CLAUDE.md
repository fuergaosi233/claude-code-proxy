# Claude Code: Best Practices for Effective Collaboration

This document outlines best practices for working with Claude Code to ensure efficient and successful software development tasks.

## Task Management

For complex or multi-step tasks, Claude Code will use:
*   **TodoWrite**: To create a structured task list, breaking down the work into manageable steps. This provides clarity on the plan and allows for tracking progress.
*   **TodoRead**: To review the current list of tasks and their status, ensuring alignment and that all objectives are being addressed.

## File Handling and Reading

Understanding file content is crucial before making modifications.

1.  **Targeted Information Retrieval**:
    *   When searching for specific content, patterns, or definitions within a codebase, prefer using search tools like `Grep` or `Task` (with a focused search prompt). This is more efficient than reading entire files.

2.  **Reading File Content**:
    *   **Small to Medium Files**: For files where full context is needed or that are not excessively large, the `Read` tool can be used to retrieve the entire content.
    *   **Large File Strategy**:
        1.  **Assess Size**: Before reading a potentially large file, its size should be determined (e.g., using `ls -l` via the `Bash` tool or by an initial `Read` with a small `limit` to observe if content is truncated).
        2.  **Chunked Reading**: If a file is large (e.g., over a few thousand lines), it should be read in manageable chunks (e.g., 1000-2000 lines at a time) using the `offset` and `limit` parameters of the `Read` tool. This ensures all content can be processed without issues.
    *   Always ensure that the file path provided to `Read` is absolute.

## File Editing

Precision is key for successful file edits. The following strategies lead to reliable modifications:

1.  **Pre-Edit Read**: **Always** use the `Read` tool to fetch the content of the file *immediately before* attempting any `Edit` or `MultiEdit` operation. This ensures modifications are based on the absolute latest version of the file.

2.  **Constructing `old_string` (The text to be replaced)**:
    *   **Exact Match**: The `old_string` must be an *exact* character-for-character match of the segment in the file you intend to replace. This includes all whitespace (spaces, tabs, newlines) and special characters.
    *   **No Read Artifacts**: Crucially, do *not* include any formatting artifacts from the `Read` tool's output (e.g., `cat -n` style line numbers or display-only leading tabs) in the `old_string`. It must only contain the literal characters as they exist in the raw file.
    *   **Sufficient Context & Uniqueness**: Provide enough context (surrounding lines) in `old_string` to make it uniquely identifiable at the intended edit location. The "Anchor on a Known Good Line" strategy is preferred: `old_string` is a larger, unique block of text surrounding the change or insertion point. This is highly reliable.

3.  **Constructing `new_string` (The replacement text)**:
    *   **Exact Representation**: The `new_string` must accurately represent the desired state of the code, including correct indentation, whitespace, and newlines.
    *   **No Read Artifacts**: As with `old_string`, ensure `new_string` does *not* contain any `Read` tool output artifacts.

4.  **Choosing the Right Editing Tool**:
    *   **`Edit` Tool**: Suitable for a single, well-defined replacement in a file.
    *   **`MultiEdit` Tool**: Preferred when multiple changes are needed within the same file. Edits are applied sequentially, with each subsequent edit operating on the result of the previous one. This tool is highly effective for complex modifications.

5.  **Verification**:
    *   The success confirmation from the `Edit` or `MultiEdit` tool (especially if `expected_replacements` is used and matches) is the primary indicator that the change was made.
    *   If further visual confirmation is needed, use the `Read` tool with `offset` and `limit` parameters to view only the specific section of the file that was changed, rather than re-reading the entire file.

### Reliable Code Insertion with MultiEdit

When inserting larger blocks of new code (e.g., multiple functions or methods) where a simple `old_string` might be fragile due to surrounding code, the following `MultiEdit` strategy can be more robust:

1.  **First Edit - Targeted Insertion Point**: For the primary code block you want to insert (e.g., new methods within a class), identify a short, unique, and stable line of code immediately *after* your desired insertion point. Use this stable line as the `old_string`.
    *   The `new_string` will consist of your new block of code, followed by a newline, and then the original `old_string` (the stable line you matched on).
    *   Example: If inserting methods into a class, the `old_string` might be the closing brace `}` of the class, or a comment that directly follows the class.

2.  **Second Edit (Optional) - Ancillary Code**: If there's another, smaller piece of related code to insert (e.g., a function call within an existing method, or an import statement), perform this as a separate, more straightforward edit within the `MultiEdit` call. This edit usually has a more clearly defined and less ambiguous `old_string`.

**Rationale**:
*   By anchoring the main insertion on a very stable, unique line *after* the insertion point and prepending the new code to it, you reduce the risk of `old_string` mismatches caused by subtle variations in the code *before* the insertion point.
*   Keeping ancillary edits separate allows them to succeed even if the main insertion point is complex, as they often target simpler, more reliable `old_string` patterns.
*   This approach leverages `MultiEdit`'s sequential application of changes effectively.

**Example Scenario**: Adding new methods to a class and a call to one of these new methods elsewhere.
*   **Edit 1**: Insert the new methods. `old_string` is the class's closing brace `}`. `new_string` is `
    [new methods code]
    }`.
*   **Edit 2**: Insert the call to a new method. `old_string` is `// existing line before call`. `new_string` is `// existing line before call
    this.newMethodCall();`.

This method provides a balance between precise editing and handling larger code insertions reliably when direct `old_string` matches for the entire new block are problematic.

## Handling Large Files for Incremental Refactoring

When refactoring large files incrementally rather than rewriting them completely:

1. **Initial Exploration and Planning**:
   * Begin with targeted searches using `Grep` to locate specific patterns or sections within the file.
   * Use `Bash` commands like `grep -n "pattern" file` to find line numbers for specific areas of interest.
   * Create a clear mental model of the file structure before proceeding with edits.

2. **Chunked Reading for Large Files**:
   * For files too large to read at once, use multiple `Read` operations with different `offset` and `limit` parameters.
   * Read sequential chunks to build a complete understanding of the file.
   * Use `Grep` to pinpoint key sections, then read just those sections with targeted `offset` parameters.

3. **Finding Key Implementation Sections**:
   * Use `Bash` commands with `grep -A N` (to show N lines after a match) or `grep -B N` (to show N lines before) to locate function or method implementations.
   * Example: `grep -n "function findTagBoundaries" -A 20 filename.js` to see the first 20 lines of a function.

4. **Pattern-Based Replacement Strategy**:
   * Identify common patterns that need to be replaced across the file.
   * Use the `Bash` tool with `sed` for quick previews of potential replacements.
   * Example: `sed -n "s/oldPattern/newPattern/gp" filename.js` to preview changes without making them.

5. **Sequential Selective Edits**:
   * Target specific sections or patterns one at a time rather than attempting a complete rewrite.
   * Focus on clearest/simplest cases first to establish a pattern of successful edits.
   * Use `Edit` for well-defined single changes within the file.

6. **Batch Similar Changes Together**:
   * Group similar types of changes (e.g., all references to a particular function or variable).
   * Use `Bash` with `sed` to preview the scope of batch changes: `grep -n "pattern" filename.js | wc -l`
   * For systematic changes across a file, consider using `sed` through the `Bash` tool: `sed -i "s/oldPattern/newPattern/g" filename.js`

7. **Incremental Verification**:
   * After each set of changes, verify the specific sections that were modified.
   * For critical components, read the surrounding context to ensure the changes integrate correctly.
   * Validate that each change maintains the file's structure and logic before proceeding to the next.

8. **Progress Tracking for Large Refactors**:
   * Use the `TodoWrite` tool to track which sections or patterns have been updated.
   * Create a checklist of all required changes and mark them off as they're completed.
   * Record any sections that require special attention or that couldn't be automatically refactored.

## Commit Messages

When Claude Code generates commit messages on your behalf:
*   The `Co-Authored-By: Claude <noreply@anthropic.com>` line will **not** be included.
*   The `🤖 Generated with [Claude Code](https://claude.ai/code)` line will **not** be included.

## General Interaction

Claude Code will directly apply proposed changes and modifications using the available tools, rather than describing them and asking you to implement them manually. This ensures a more efficient and direct workflow.

## Recent Changes

- Updated MIDDLE_MODEL config to default to BIG_MODEL value for consistency