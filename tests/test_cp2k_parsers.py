"""Tests for CP2K data file parsers."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiida_relax_project.utils.cp2k_parsers import (
    BasisEntry,
    _extract_accuracy,
    _select_basis_by_accuracy,
    list_basis_entries,
    parse_cp2k_data_file,
    resolve_basis_names,
    resolve_potential_name,
    resolve_ri_basis_name,
)


class TestExtractAccuracy:
    def test_comment_format(self):
        text = "RI basis set for H, GTH pseudo, relative accuracy of RI-MP2: 1.2e-06"
        assert _extract_accuracy(text) == pytest.approx(1.2e-06)

    def test_name_format(self):
        text = "RI_aug-SZV-MOLOPT-GTH-tier-1_H_RI_009_error_1.1e-06"
        assert _extract_accuracy(text) == pytest.approx(1.1e-06)

    def test_no_accuracy(self):
        assert _extract_accuracy("plain comment, no accuracy here") is None

    def test_empty_string(self):
        assert _extract_accuracy("") is None

    def test_invalid_number(self):
        assert _extract_accuracy("error: not_a_number") is None


class TestSelectBasisByAccuracy:
    def make_entry(self, name: str, accuracy: float | None) -> BasisEntry:
        return BasisEntry(name=name, header=name, accuracy=accuracy)

    def test_prefers_entries_within_target(self):
        entries = [
            self.make_entry("coarse", 1e-4),
            self.make_entry("medium", 1e-5),
            self.make_entry("fine", 1e-6),
        ]
        result = _select_basis_by_accuracy(entries, 1e-5)
        assert result == "fine"

    def test_no_accuracy_falls_back_to_first(self):
        entries = [
            self.make_entry("first", None),
            self.make_entry("second", None),
        ]
        assert _select_basis_by_accuracy(entries, 1e-5) == "first"

    def test_mixed_accuracy(self):
        entries = [
            self.make_entry("no_acc", None),
            self.make_entry("medium", 1e-5),
            self.make_entry("fine", 1e-6),
        ]
        result = _select_basis_by_accuracy(entries, 1e-5)
        assert result == "fine"

    def test_single_entry(self):
        entries = [self.make_entry("only", 1e-6)]
        assert _select_basis_by_accuracy(entries, 1e-5) == "only"

    def test_no_entries(self):
        assert _select_basis_by_accuracy([], 1e-5) is None


class TestParseCp2kDataFile:
    SAMPLE = """\
H  aug-SZV-MOLOPT-GTH-tier-1
  1    2
  3    4
# RI basis set for H, GTH pseudo, relative accuracy of RI-MP2: 1.2e-06
H  aug-TZV-MOLOPT-GTH-tier-2
  5    6
  7    8
# RI basis set for H, GTH pseudo, relative accuracy of RI-MP2: 8.5e-07
N  DZVP-MOLOPT-GTH
  9   10
"""

    def test_parse_multiple_entries(self, tmp_path: Path) -> None:
        f = tmp_path / "RI_BASIS"
        f.write_text(self.SAMPLE)
        entries = parse_cp2k_data_file(str(f))
        assert "H" in entries
        assert "N" in entries
        assert len(entries["H"]) == 2
        assert len(entries["N"]) == 1

    def test_entry_data(self, tmp_path: Path) -> None:
        f = tmp_path / "RI_BASIS"
        f.write_text(self.SAMPLE)
        entries = parse_cp2k_data_file(str(f))
        h_first = entries["H"][0]
        assert h_first.name == "aug-SZV-MOLOPT-GTH-tier-1"
        assert h_first.header == "H  aug-SZV-MOLOPT-GTH-tier-1"
        assert "1.2e-06" in h_first.comment
        assert h_first.accuracy == pytest.approx(1.2e-06)
        h_second = entries["H"][1]
        assert h_second.name == "aug-TZV-MOLOPT-GTH-tier-2"
        assert h_second.accuracy == pytest.approx(8.5e-07)

    def test_no_comment(self, tmp_path: Path) -> None:
        f = tmp_path / "POTENTIAL"
        f.write_text("B  GTH-PBE-q3\n  1   2\n")
        entries = parse_cp2k_data_file(str(f))
        entry = entries["B"][0]
        assert entry.name == "GTH-PBE-q3"
        assert entry.comment == ""
        assert entry.accuracy is None

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            parse_cp2k_data_file("/nonexistent/path")

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "EMPTY"
        f.write_text("")
        assert parse_cp2k_data_file(str(f)) == {}

    def test_comment_only_file(self, tmp_path: Path) -> None:
        f = tmp_path / "COMMENTS"
        f.write_text("# just a comment\n# another one\n")
        assert parse_cp2k_data_file(str(f)) == {}


class TestListBasisEntries:
    SAMPLE = """\
H  SZV
  1  2
H  DZVP
  3  4
"""

    def test_returns_all_entries(self, tmp_path: Path) -> None:
        f = tmp_path / "BASIS"
        f.write_text(self.SAMPLE)
        entries = list_basis_entries(str(f), "H")
        assert len(entries) == 2
        assert entries[0].name == "SZV"

    def test_nonexistent_element(self, tmp_path: Path) -> None:
        f = tmp_path / "BASIS"
        f.write_text(self.SAMPLE)
        assert list_basis_entries(str(f), "Xx") == []


class TestResolveBasisNames:
    SAMPLE = """\
H  SZV
  1  2
H  DZVP
  3  4
H  TZV
  5  6
"""

    def test_all_names(self, tmp_path: Path) -> None:
        f = tmp_path / "BASIS"
        f.write_text(self.SAMPLE)
        assert resolve_basis_names(str(f), "H") == ["SZV", "DZVP", "TZV"]

    def test_with_pattern(self, tmp_path: Path) -> None:
        f = tmp_path / "BASIS"
        f.write_text(self.SAMPLE)
        assert resolve_basis_names(str(f), "H", pattern="ZV") == ["SZV", "DZVP", "TZV"]

    def test_no_match(self, tmp_path: Path) -> None:
        f = tmp_path / "BASIS"
        f.write_text(self.SAMPLE)
        assert resolve_basis_names(str(f), "Xx") == []


class TestResolvePotentialName:
    def test_first_entry_returned(self, tmp_path: Path) -> None:
        f = tmp_path / "POTENTIAL"
        f.write_text("B  GTH-PBE-q3\n  1  2\nN  GTH-PBE-q5\n  3  4\n")
        assert resolve_potential_name(str(f), "B") == "GTH-PBE-q3"
        assert resolve_potential_name(str(f), "N") == "GTH-PBE-q5"

    def test_missing_element(self, tmp_path: Path) -> None:
        f = tmp_path / "POTENTIAL"
        f.write_text("B  GTH-PBE-q3\n")
        assert resolve_potential_name(str(f), "Xx") is None


class TestResolveRiBasisName:
    SAMPLE = """\
H  RI_aug-SZV-MOLOPT-GTH-tier-1_H_RI_001_error_1e-04
  1  2
# relative accuracy of RI-MP2: 1e-04
H  RI_aug-SZV-MOLOPT-GTH-tier-1_H_RI_002_error_1e-05
  3  4
# relative accuracy of RI-MP2: 1e-05
H  RI_aug-SZV-MOLOPT-GTH-tier-1_H_RI_003_error_1e-06
  5  6
# relative accuracy of RI-MP2: 1e-06
H  RI_aug-DZVP-GTH_H_RI_001_error_1e-07
  7  8
# relative accuracy of RI-MP2: 1e-07
"""

    def test_no_accuracy_target(self, tmp_path: Path) -> None:
        f = tmp_path / "RI_BASIS"
        f.write_text(self.SAMPLE)
        assert resolve_ri_basis_name(str(f), "H") == "RI_aug-SZV-MOLOPT-GTH-tier-1_H_RI_001_error_1e-04"

    def test_accuracy_target_picks_best(self, tmp_path: Path) -> None:
        f = tmp_path / "RI_BASIS"
        f.write_text(self.SAMPLE)
        result = resolve_ri_basis_name(
            str(f), "H", accuracy_target=1e-5,
        )
        assert result == "RI_aug-DZVP-GTH_H_RI_001_error_1e-07"

    def test_accuracy_target_coarse(self, tmp_path: Path) -> None:
        f = tmp_path / "RI_BASIS"
        f.write_text(self.SAMPLE)
        result = resolve_ri_basis_name(
            str(f), "H", accuracy_target=1e-3,
        )
        assert result == "RI_aug-DZVP-GTH_H_RI_001_error_1e-07"

    def test_no_accuracy_in_file(self, tmp_path: Path) -> None:
        f = tmp_path / "RI_BASIS"
        f.write_text("H  plain_basis\n  1  2\n")
        result = resolve_ri_basis_name(str(f), "H", accuracy_target=1e-5)
        assert result == "plain_basis"

    def test_missing_element(self, tmp_path: Path) -> None:
        f = tmp_path / "RI_BASIS"
        f.write_text(self.SAMPLE)
        assert resolve_ri_basis_name(str(f), "Xx") is None

    def test_orb_basis_filters_consistent(self, tmp_path: Path) -> None:
        f = tmp_path / "RI_BASIS"
        f.write_text(self.SAMPLE)
        result = resolve_ri_basis_name(
            str(f), "H",
            orb_basis="aug-SZV-MOLOPT-GTH-tier-1",
        )
        assert result == "RI_aug-SZV-MOLOPT-GTH-tier-1_H_RI_001_error_1e-04"

    def test_orb_basis_excludes_wrong_family(self, tmp_path: Path) -> None:
        f = tmp_path / "RI_BASIS"
        f.write_text(self.SAMPLE)
        result = resolve_ri_basis_name(
            str(f), "H",
            orb_basis="aug-DZVP-GTH",
        )
        assert result == "RI_aug-DZVP-GTH_H_RI_001_error_1e-07"

    def test_orb_basis_with_accuracy(self, tmp_path: Path) -> None:
        f = tmp_path / "RI_BASIS"
        f.write_text(self.SAMPLE)
        result = resolve_ri_basis_name(
            str(f), "H",
            accuracy_target=1e-5,
            orb_basis="aug-SZV-MOLOPT-GTH-tier-1",
        )
        assert result == "RI_aug-SZV-MOLOPT-GTH-tier-1_H_RI_003_error_1e-06"

    def test_orb_basis_no_match(self, tmp_path: Path) -> None:
        f = tmp_path / "RI_BASIS"
        f.write_text(self.SAMPLE)
        assert resolve_ri_basis_name(
            str(f), "H",
            orb_basis="nonexistent-basis",
        ) is None
