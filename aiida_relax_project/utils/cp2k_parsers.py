"""Utilities for reading CP2K basis-set and potential files."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class BasisEntry:
    """A single entry in a CP2K data file.

    Attributes:
        name: The basis set / potential name.
        header: The raw header line (e.g. ``"H  aug-SZV-MOLOPT-GTH-tier-1"``).
        comment: Concatenated comment lines associated with this entry.
        accuracy: Relative accuracy extracted from the comment or name, if any.
    """

    name: str
    header: str
    comment: str = ""
    accuracy: float | None = None

    def __post_init__(self) -> None:
        if self.accuracy is None:
            self.accuracy = _extract_accuracy(self.comment) or _extract_accuracy(self.name)


_ACCURACY_RE = re.compile(
    r"(?:relative\s+accuracy\s+of\s+RI-MP2|error)[:\s_]+([\d.eE+-]+)"
)
_HEADER_RE = re.compile(r"^([A-Z][a-z]?)\s+(\S+)")


def _extract_accuracy(text: str) -> float | None:
    """Extract a relative accuracy value from *text*.

    Supports two formats found in CP2K data files:
      - ``# RI basis set for H, GTH pseudo, relative accuracy of RI-MP2: 1.2e-06``
      - ``...error_1.1e-06`` (embedded in the basis-name itself).
    """
    if not text:
        return None
    m = _ACCURACY_RE.search(text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def parse_cp2k_data_file(path: str) -> dict[str, list[BasisEntry]]:
    """Parse a CP2K data file into element → list of *BasisEntry*.

    The format is::

        Element  Name
          <data lines>
        # comment with accuracy

    Comment lines appearing after a header (before the next header) are
    associated with that entry.
    """
    entries: dict[str, list[BasisEntry]] = {}

    current_element: str | None = None
    current_name: str | None = None
    current_header: str | None = None
    current_comment: list[str] = []
    in_entry = False

    def _save() -> None:
        if current_element is not None and current_name is not None:
            comment = " ".join(current_comment).strip()
            entries.setdefault(current_element, []).append(
                BasisEntry(
                    name=current_name,
                    header=current_header or "",
                    comment=comment,
                )
            )

    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith("#"):
                # Strip the leading "#" (and maybe whitespace) for storage
                comment_text = stripped.lstrip("#").strip()
                if comment_text:
                    current_comment.append(comment_text)
                continue

            m = _HEADER_RE.match(stripped)
            if m:
                _save()
                current_element = m.group(1)
                current_name = m.group(2)
                current_header = stripped
                current_comment = []
                in_entry = True

    if in_entry:
        _save()

    return entries


def list_basis_entries(
    basis_file: str, element: str
) -> list[BasisEntry]:
    """Return all *BasisEntry* objects for *element* in *basis_file*."""
    entries = parse_cp2k_data_file(basis_file)
    return entries.get(element, [])


def resolve_basis_names(
    basis_file: str, element: str, pattern: str | None = None
) -> list[str]:
    """Return all basis set names for *element* in *basis_file*.

    If *pattern* is given, only names containing it are returned.
    """
    entries = parse_cp2k_data_file(basis_file)
    names = [entry.name for entry in entries.get(element, [])]
    if pattern:
        names = [n for n in names if pattern in n]
    return names


def resolve_potential_name(
    potential_file: str,
    element: str,
    pattern: str | None = "GTH-",
) -> str | None:
    """Return the potential name for *element* in *potential_file*.

    When *pattern* is given (default ``"GTH-"``), only names containing
    it are considered — this avoids accidentally picking an all-electron
    entry when a pseudopotential is wanted.
    """
    entries = parse_cp2k_data_file(potential_file)
    names = [entry.name for entry in entries.get(element, [])]
    if pattern:
        names = [n for n in names if pattern in n]
    return names[0] if names else None


def resolve_ri_basis_name(
    ri_basis_file: str,
    element: str,
    accuracy_target: float | None = None,
    orb_basis: str | None = None,
) -> str | None:
    """Return the RI auxiliary basis name for *element*.

    When *orb_basis* is given (e.g. ``"aug-SZV-MOLOPT-GTH-tier-1"``),
    only entries whose name contains ``RI_{orb_basis}`` are considered,
    ensuring consistency with the ORB basis set.

    When *accuracy_target* is given (e.g. ``1e-5``), the best available
    basis set whose accuracy does not exceed the target is selected.
    Otherwise the first matching entry is returned.
    """
    entries = list_basis_entries(ri_basis_file, element)

    if orb_basis:
        prefix = f"RI_{orb_basis}"
        entries = [e for e in entries if prefix in e.name]

    if not entries:
        return None
    if accuracy_target is not None:
        return _select_basis_by_accuracy(entries, accuracy_target)
    return entries[0].name


def _select_basis_by_accuracy(
    entries: list[BasisEntry], target: float
) -> str | None:
    """Pick the entry with the largest accuracy value ≥ *target*.

    This selects the cheapest basis that still meets the target
    (largest accuracy value ≥ *target*).  If no entry meets the
    target, falls back to the best accuracy overall (smallest
    value).  If no entry has accuracy metadata, returns the first.
    """
    if not entries:
        return None
    candidates = [e for e in entries if e.accuracy is not None]
    if not candidates:
        return entries[0].name

    # Entries that meet the target — pick the cheapest (largest accuracy)
    within = [e for e in candidates if e.accuracy >= target]
    if within:
        return max(within, key=lambda e: e.accuracy).name

    # Fall back to the best accuracy overall (smallest error)
    return min(candidates, key=lambda e: e.accuracy).name
