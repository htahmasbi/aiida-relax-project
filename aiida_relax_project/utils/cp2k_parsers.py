"""Utilities for reading CP2K basis-set and potential files."""

from __future__ import annotations

import re


def parse_cp2k_data_file(path: str) -> dict[str, list[tuple[str, str]]]:
    """Parse a CP2K data file (basis set or potential) and return element→names.

    The format is:
        Element  Name
          <data lines>

    Returns:
        {element: [(name, raw_header), ...]} preserving order and
        allowing multiple entries per element.
    """
    entries: dict[str, list[tuple[str, str]]] = {}
    header_pattern = re.compile(r"^(\w+)\s+(\S+)")

    current_element: str | None = None
    current_name: str | None = None
    header_line: str | None = None
    expecting_data = False

    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            m = header_pattern.match(stripped)
            if m:
                # If we were building a previous entry, save it
                if current_element is not None and current_name is not None:
                    entries.setdefault(current_element, []).append(
                        (current_name, header_line)
                    )

                current_element = m.group(1)
                current_name = m.group(2)
                header_line = stripped
                expecting_data = True
            elif expecting_data:
                # data line — keep going
                pass

    if current_element is not None and current_name is not None:
        entries.setdefault(current_element, []).append((current_name, header_line))

    return entries


def resolve_basis_names(
    basis_file: str, element: str, pattern: str | None = None
) -> list[str]:
    """Return all basis set names for *element* in *basis_file*.

    If *pattern* is given, only names containing it are returned.
    """
    entries = parse_cp2k_data_file(basis_file)
    names = [name for name, _ in entries.get(element, [])]
    if pattern:
        names = [n for n in names if pattern in n]
    return names


def resolve_potential_name(potential_file: str, element: str) -> str | None:
    """Return the potential name for *element* in *potential_file*."""
    entries = parse_cp2k_data_file(potential_file)
    names = [name for name, _ in entries.get(element, [])]
    return names[0] if names else None


def resolve_ri_basis_name(ri_basis_file: str, element: str) -> str | None:
    """Return the RI auxiliary basis name for *element*.

    RI basis names typically contain the element symbol and are the only
    entry in the file for that element.
    """
    names = resolve_basis_names(ri_basis_file, element)
    return names[0] if names else None
