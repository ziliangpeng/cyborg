#!/usr/bin/env python3
"""
Lines of Code Counter

Scans repository and categorizes lines of code by type.
Usage: ./count_lines.py [path]
"""

import os
import re
import sys
import subprocess
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class Stats:
    """Statistics for lines of code and file counts"""
    # Line counts
    bazel: int = 0
    python: int = 0
    python_test: int = 0
    rust: int = 0
    rust_test: int = 0
    cuda: int = 0
    cpp: int = 0
    metal: int = 0
    shell: int = 0
    text: int = 0
    misc: int = 0

    # File counts
    bazel_files: int = 0
    python_files: int = 0
    python_test_files: int = 0
    rust_files: int = 0
    rust_test_files: int = 0
    cuda_files: int = 0
    cpp_files: int = 0
    metal_files: int = 0
    shell_files: int = 0
    text_files: int = 0
    misc_files: int = 0

    def total_lines(self) -> int:
        return sum([
            self.bazel,
            self.python,
            self.python_test,
            self.rust,
            self.rust_test,
            self.cuda,
            self.cpp,
            self.metal,
            self.shell,
            self.text,
            self.misc,
        ])

    def total_files(self) -> int:
        return sum([
            self.bazel_files,
            self.python_files,
            self.python_test_files,
            self.rust_files,
            self.rust_test_files,
            self.cuda_files,
            self.cpp_files,
            self.metal_files,
            self.shell_files,
            self.text_files,
            self.misc_files,
        ])


# Directories to skip during scanning
SKIP_DIRS = {
    '.git',
    'target',
    'node_modules',
    '.venv',
    'venv',
    '__pycache__',
    '.pytest_cache',
    '.mypy_cache',
    'build',
    'dist',
    'data',  # Skip data directories with large binary files
    '.cargo',
    '.cache',
}


def should_skip_path(path: Path) -> bool:
    """Check if path should be skipped"""
    # Skip if any parent directory is in SKIP_DIRS
    for parent in path.parts:
        if parent in SKIP_DIRS:
            return True
    return False


def get_gitignored_files(repo_root: Path) -> set:
    """Get set of all gitignored files in the repository"""
    try:
        # First, get all files in the repo
        result = subprocess.run(
            ['find', '.', '-type', 'f', '!', '-path', './.git/*'],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            text=True
        )

        if result.returncode != 0:
            return set()

        all_files = result.stdout.strip().split('\n')

        # Now batch check which ones are ignored
        # Use stdin to pass all files at once
        check_result = subprocess.run(
            ['git', 'check-ignore', '--stdin'],
            cwd=repo_root,
            input='\n'.join(all_files),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            text=True
        )

        # Git check-ignore outputs the ignored files
        if check_result.stdout:
            ignored_files = set(check_result.stdout.strip().split('\n'))
            # Convert to absolute paths
            return {(repo_root / f.lstrip('./')).resolve() for f in ignored_files if f}

        return set()
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        # If git is not available or command fails, return empty set
        return set()


def is_python_test(path: Path) -> bool:
    """Check if Python file is a test file"""
    # Check if in tests/ directory
    if 'tests' in path.parts:
        return True
    # Check if filename matches test_*.py pattern
    if path.name.startswith('test_') and path.suffix == '.py':
        return True
    return False


def count_rust_lines(file_path: Path) -> Tuple[int, int]:
    """
    Count Rust code lines separately from test lines.
    Returns (code_lines, test_lines)
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception:
        return 0, 0

    code_lines = 0
    test_lines = 0

    in_test_module = False
    brace_depth = 0
    test_start_depth = 0
    in_cfg_test = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Check if we're seeing #[cfg(test)]
        if stripped == '#[cfg(test)]':
            in_cfg_test = True
            test_lines += 1
            continue

        # Check if we're entering a test module (after #[cfg(test)])
        if in_cfg_test and stripped.startswith('mod ') and '{' in stripped:
            in_test_module = True
            test_start_depth = brace_depth
            in_cfg_test = False
            test_lines += 1
            brace_depth += stripped.count('{') - stripped.count('}')
            continue

        # Check for standalone #[test] functions
        if stripped == '#[test]':
            in_test_module = True
            test_start_depth = brace_depth
            test_lines += 1
            continue

        # Count the line
        if in_test_module:
            test_lines += 1
        else:
            code_lines += 1

        # Update brace depth
        brace_depth += stripped.count('{') - stripped.count('}')

        # Check if we're exiting test module
        if in_test_module and brace_depth <= test_start_depth:
            in_test_module = False

    return code_lines, test_lines


def categorize_and_count(file_path: Path) -> Tuple[str, int]:
    """
    Categorize file and count lines.
    Returns (category, line_count)
    """
    filename = file_path.name
    suffix = file_path.suffix.lower()

    # Bazel files
    if filename in ('BUILD', 'BUILD.bazel', 'MODULE.bazel', '.bazelrc') or suffix == '.bzl':
        return 'bazel', count_lines(file_path)

    # Python files
    if suffix == '.py':
        if is_python_test(file_path):
            return 'python_test', count_lines(file_path)
        else:
            return 'python', count_lines(file_path)

    # Rust files (special handling for tests)
    if suffix == '.rs':
        code_lines, test_lines = count_rust_lines(file_path)
        # Return both counts; we'll handle this specially
        return 'rust', (code_lines, test_lines)

    # CUDA files
    if suffix in ('.cu', '.cuh'):
        return 'cuda', count_lines(file_path)

    # C++ files
    if suffix in ('.cpp', '.cc', '.c', '.h', '.hpp', '.hh', '.cxx', '.hxx'):
        return 'cpp', count_lines(file_path)

    # Metal shader files
    if suffix == '.metal':
        return 'metal', count_lines(file_path)

    # Shell scripts
    if suffix in ('.sh', '.bash', '.zsh'):
        return 'shell', count_lines(file_path)

    # Text/config files
    if (suffix in ('.md', '.txt', '.rst', '.toml', '.yaml', '.yml', '.json') or
        filename.startswith('README') or
        filename.startswith('LICENSE')):
        return 'text', count_lines(file_path)

    # Everything else
    return 'misc', count_lines(file_path)


def count_lines(file_path: Path) -> int:
    """Count total lines in a file"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def scan_directory(root_path: Path, collect_files: bool = False) -> tuple:
    """
    Scan directory and count lines by category.

    Args:
        root_path: Directory to scan
        collect_files: If True, also collect file paths by category

    Returns:
        Tuple of (Stats, dict[str, list[Path]]) where dict maps category names to file paths
    """
    stats = Stats()
    file_dict = {
        'bazel': [],
        'python': [],
        'python_test': [],
        'rust': [],
        'rust_test': [],
        'cuda': [],
        'cpp': [],
        'metal': [],
        'shell': [],
        'text': [],
        'misc': [],
    } if collect_files else {}

    # Build set of gitignored files once upfront
    print("Checking for gitignored files...")
    gitignored_files = get_gitignored_files(root_path)
    print(f"Found {len(gitignored_files)} gitignored files to skip")

    for root, dirs, files in os.walk(root_path):
        # Remove skip directories from dirs list (modifies in-place)
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for filename in files:
            file_path = Path(root) / filename

            # Skip if path should be skipped
            if should_skip_path(file_path):
                continue

            # Skip symlinks
            if file_path.is_symlink():
                continue

            # Skip gitignored files
            if file_path.resolve() in gitignored_files:
                continue

            # Skip files larger than 10MB (likely binary/data files)
            try:
                if file_path.stat().st_size > 10 * 1024 * 1024:
                    continue
            except (OSError, PermissionError):
                continue

            category, count = categorize_and_count(file_path)

            # Special handling for Rust files (returns tuple)
            if category == 'rust' and isinstance(count, tuple):
                code_lines, test_lines = count
                stats.rust += code_lines
                stats.rust_test += test_lines
                # For Rust, we need to track if file has more code or test lines
                if code_lines >= test_lines:
                    stats.rust_files += 1
                    if collect_files:
                        file_dict['rust'].append(file_path)
                else:
                    stats.rust_test_files += 1
                    if collect_files:
                        file_dict['rust_test'].append(file_path)
            elif category == 'bazel':
                stats.bazel += count
                stats.bazel_files += 1
                if collect_files:
                    file_dict['bazel'].append(file_path)
            elif category == 'python':
                stats.python += count
                stats.python_files += 1
                if collect_files:
                    file_dict['python'].append(file_path)
            elif category == 'python_test':
                stats.python_test += count
                stats.python_test_files += 1
                if collect_files:
                    file_dict['python_test'].append(file_path)
            elif category == 'cuda':
                stats.cuda += count
                stats.cuda_files += 1
                if collect_files:
                    file_dict['cuda'].append(file_path)
            elif category == 'cpp':
                stats.cpp += count
                stats.cpp_files += 1
                if collect_files:
                    file_dict['cpp'].append(file_path)
            elif category == 'metal':
                stats.metal += count
                stats.metal_files += 1
                if collect_files:
                    file_dict['metal'].append(file_path)
            elif category == 'shell':
                stats.shell += count
                stats.shell_files += 1
                if collect_files:
                    file_dict['shell'].append(file_path)
            elif category == 'text':
                stats.text += count
                stats.text_files += 1
                if collect_files:
                    file_dict['text'].append(file_path)
            elif category == 'misc':
                stats.misc += count
                stats.misc_files += 1
                if collect_files:
                    file_dict['misc'].append(file_path)

    return stats, file_dict


def format_number(n: int) -> str:
    """Format number with comma separators"""
    return f"{n:,}"


def print_stats(stats: Stats):
    """Print statistics in a formatted table"""
    total_lines = stats.total_lines()
    total_files = stats.total_files()

    print("\nLines of Code by Category:")
    print("=" * 70)

    categories = [
        ("Bazel build files", stats.bazel, stats.bazel_files),
        ("Python code", stats.python, stats.python_files),
        ("Python test code", stats.python_test, stats.python_test_files),
        ("Rust code", stats.rust, stats.rust_files),
        ("Rust test code", stats.rust_test, stats.rust_test_files),
        ("CUDA code", stats.cuda, stats.cuda_files),
        ("C++ code", stats.cpp, stats.cpp_files),
        ("Metal shaders", stats.metal, stats.metal_files),
        ("Shell scripts", stats.shell, stats.shell_files),
        ("Text files", stats.text, stats.text_files),
        ("Misc", stats.misc, stats.misc_files),
    ]

    for name, line_count, file_count in categories:
        if total_lines > 0:
            percentage = (line_count / total_lines) * 100
            print(f"{name:20} {format_number(line_count):>10} lines ({percentage:5.1f}%)  {file_count:>4} files")
        else:
            print(f"{name:20} {format_number(line_count):>10} lines  {file_count:>4} files")

    print("=" * 70)
    print(f"{'Total':20} {format_number(total_lines):>10} lines {format_number(total_files):>10} files")
    print()


def cmd_stat(root_path: Path):
    """Show statistics by category (default command)"""
    print(f"Scanning directory: {root_path}")
    print("This may take a moment...")

    stats, _ = scan_directory(root_path, collect_files=False)
    print_stats(stats)


def cmd_find_misc(root_path: Path):
    """Find and list all files categorized as misc"""
    print(f"Scanning directory: {root_path}")
    print("This may take a moment...")

    _, file_dict = scan_directory(root_path, collect_files=True)
    misc_files = file_dict.get('misc', [])

    print(f"\nFound {len(misc_files)} misc files:")
    print("=" * 50)
    for file_path in sorted(misc_files):
        # Print relative path for readability
        try:
            rel_path = file_path.relative_to(root_path)
        except ValueError:
            rel_path = file_path
        print(rel_path)
    print()


def main():
    """Main entry point"""
    # Parse command and path
    command = 'stat'  # default
    root_path = Path.cwd()

    if len(sys.argv) > 1:
        # Check if first arg is a command or path
        first_arg = sys.argv[1].lower()
        if first_arg in ['stat', 'find_misc']:
            command = first_arg
            if len(sys.argv) > 2:
                root_path = Path(sys.argv[2])
        else:
            # Backward compatibility: treat as path
            root_path = Path(sys.argv[1])

    # Validate path
    if not root_path.exists():
        print(f"Error: Path '{root_path}' does not exist")
        sys.exit(1)

    if not root_path.is_dir():
        print(f"Error: Path '{root_path}' is not a directory")
        sys.exit(1)

    # Execute command
    if command == 'stat':
        cmd_stat(root_path)
    elif command == 'find_misc':
        cmd_find_misc(root_path)
    else:
        print(f"Error: Unknown command '{command}'")
        print("Available commands: stat, find_misc")
        sys.exit(1)


if __name__ == "__main__":
    main()
