"""Tests for config loader."""

from __future__ import annotations

import pytest
import yaml

from aiida_relax_project.config import (
    CalculatorConfig,
    load_config,
    _deep_merge,
    _dict_to_config,
)


class TestDeepMerge:
    """Test dictionary deep merge functionality."""

    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        overlay = {"b": 3, "c": 4}
        result = _deep_merge(base, overlay)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}}
        overlay = {"a": {"y": 3, "z": 4}}
        result = _deep_merge(base, overlay)
        assert result == {"a": {"x": 1, "y": 3, "z": 4}}


class TestLoadConfig:
    """Test configuration loading."""

    def test_load_vasp_config(self, tmp_path):
        config_data = {
            "engine": "vasp",
            "description": "Test VASP config",
            "vasp": {
                "code_label": "vasp@localhost",
                "potential_family": "PBE.54",
                "parameters": {"ENCUT": 500},
            },
            "optimade": {"filter": None, "max_structures": None},
        }

        config_file = tmp_path / "test_vasp.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_file)

        assert config.engine == "vasp"
        assert config.vasp.code_label == "vasp@localhost"
        assert config.vasp.parameters["ENCUT"] == 500

    def test_load_cp2k_config(self, tmp_path):
        config_data = {
            "engine": "cp2k",
            "description": "Test CP2K config",
            "cp2k": {
                "code_label": "cp2k@localhost",
                "kinds": [
                    {
                        "element": "Si",
                        "basis_set": "DZVP-MOLOPT-SR-GTH-q4",
                        "potential": "GTH-PBE-q4",
                    },
                ],
                "parameters": {"CUTOFF": 500},
            },
            "optimade": {"filter": None, "max_structures": None},
        }

        config_file = tmp_path / "test_cp2k.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_file)

        assert config.engine == "cp2k"
        assert config.cp2k.code_label == "cp2k@localhost"
        assert len(config.cp2k.kinds) == 1
        assert config.cp2k.kinds[0].element == "Si"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")


class TestCalculatorConfigValidation:
    """Test configuration validation."""

    def test_valid_vasp_config(self):
        config_data = {
            "engine": "vasp",
            "optimade": {"filter": None, "max_structures": None},
            "vasp": {
                "code_label": "vasp@localhost",
            },
        }
        config = _dict_to_config(config_data)
        config.validate()
        assert config.engine == "vasp"

    def test_valid_cp2k_config(self):
        config_data = {
            "engine": "cp2k",
            "optimade": {"filter": None, "max_structures": None},
            "cp2k": {
                "code_label": "cp2k@localhost",
            },
        }
        config = _dict_to_config(config_data)
        config.validate()
        assert config.engine == "cp2k"

    def test_invalid_engine(self):
        config_data = {
            "engine": "quantum_espresso",
            "optimade": {"filter": None, "max_structures": None},
        }
        config = _dict_to_config(config_data)
        with pytest.raises(ValueError, match="engine must be 'vasp' or 'cp2k'"):
            config.validate()

    def test_vasp_without_vasp_section(self):
        config_data = {
            "engine": "vasp",
            "optimade": {"filter": None, "max_structures": None},
        }
        config = _dict_to_config(config_data)
        with pytest.raises(ValueError, match="engine=vasp requires vasp section"):
            config.validate()

    def test_no_structure_source(self):
        config_data = {
            "engine": "vasp",
            "optimade": {"filter": None, "max_structures": None},
            "vasp": {"code_label": "vasp@localhost"},
        }
        config = _dict_to_config(config_data)
        with pytest.raises(ValueError, match="must specify one of"):
            config.validate()