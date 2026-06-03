"""Tests for config loader (new Pydantic-based config)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
try:
    import tomllib
except ImportError:
    import tomli as tomllib
import tomli_w

from aiida_relax_project.core.config import (
    ProjectConfig,
    VaspConfig,
    Cp2kConfig,
    RelaxConfig,
    VolumeScanConfig,
    GwConfig,
    MetadataOptions,
    load_config,
    _merge_configs,
)


class TestMergeConfigs:
    """Test dictionary merge functionality."""

    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        overlay = {"b": 3, "c": 4}
        result = _merge_configs(base, overlay)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}}
        overlay = {"a": {"y": 3, "z": 4}}
        result = _merge_configs(base, overlay)
        assert result == {"a": {"x": 1, "y": 3, "z": 4}}


class TestProjectConfig:
    """Test ProjectConfig creation and validation."""

    def test_default_config(self):
        config = ProjectConfig()
        assert config.engine == "vasp"
        assert config.code_label == "localhost"
        assert config.vasp.potential_family == "PBE.54"
        assert config.vasp.kpoints_mesh == [4, 4, 4]
        assert config.cp2k.kpoints_mesh == [4, 1, 4]

    def test_cp2k_engine(self):
        config = ProjectConfig(engine="cp2k")
        assert config.engine == "cp2k"
        assert config.get_kpoints_mesh() == [4, 1, 4]

    def test_vasp_engine_kpoints(self):
        config = ProjectConfig(engine="vasp")
        assert config.get_kpoints_mesh() == [4, 4, 4]

    def test_invalid_engine_fails(self):
        with pytest.raises(ValueError):
            ProjectConfig(engine="quantum_espresso")

    def test_metadata_options_defaults(self):
        config = ProjectConfig()
        opts = config.metadata_options
        assert opts.num_machines == 1
        assert opts.num_mpiprocs_per_machine == 8
        assert opts.max_wallclock_seconds == 3600
        assert opts.withmpi is True

    def test_metadata_options_preset(self):
        config = ProjectConfig(resource_preset="high_memory")
        opts = config.metadata_options
        assert opts.num_machines == 2

    def test_to_dict(self):
        config = ProjectConfig(engine="vasp")
        d = config.to_dict()
        assert d["engine"] == "vasp"
        assert "vasp" in d
        assert "cp2k" in d
        assert "metadata_options" in d

    def test_get_potential_mapping(self):
        config = ProjectConfig()
        mapping = config.get_potential_mapping(["Si", "Ge"])
        assert mapping == {"Si": "Si", "Ge": "Ge"}


class TestVaspConfig:
    """Test VASP-specific configuration."""

    def test_defaults(self):
        cfg = VaspConfig()
        assert cfg.kpoints_mesh == [4, 4, 4]
        assert cfg.default_encut == 400
        assert cfg.default_isif == 3

    def test_kpoints_mesh_validates_length(self):
        with pytest.raises(ValueError):
            VaspConfig(kpoints_mesh=[4, 4])

    def test_kpoints_mesh_validates_positive(self):
        with pytest.raises(ValueError):
            VaspConfig(kpoints_mesh=[4, 0, 4])


class TestCp2kConfig:
    """Test CP2K-specific configuration."""

    def test_defaults(self):
        cfg = Cp2kConfig()
        assert cfg.kpoints_mesh == [4, 1, 4]
        assert cfg.default_cutoff == 400
        assert cfg.default_eps_scf == 1e-6
        assert cfg.ri_basis_set_mapping == {}
        assert cfg.ri_basis_set_file is None

    def test_ri_basis_set_mapping(self):
        cfg = Cp2kConfig(ri_basis_set_mapping={"B": "RI_DZVP", "N": "RI_DZVP"})
        assert cfg.ri_basis_set_mapping["B"] == "RI_DZVP"
        assert cfg.ri_basis_set_mapping["N"] == "RI_DZVP"

    def test_ri_basis_set_file(self):
        cfg = Cp2kConfig(ri_basis_set_file="/path/to/RI_BASIS")
        assert cfg.ri_basis_set_file == "/path/to/RI_BASIS"


class TestGwConfig:
    """Test GW-specific configuration."""

    def test_defaults(self):
        cfg = GwConfig()
        assert cfg.auto_resolve is False
        assert cfg.kpoints_mesh == [12, 1, 12]
        assert cfg.kpoints_w == [12, 1, 12]
        assert cfg.cutoff == 400
        assert cfg.periodic == "XZ"
        assert cfg.poisson_solver == "WAVELET"
        assert cfg.memory_per_proc == 600
        assert "B" in cfg.element_settings
        assert "N" in cfg.element_settings
        assert cfg.element_settings["B"].ri_basis.startswith("RI_")
        assert cfg.element_settings["N"].potential == "GTH-PBE-q5"

    def test_auto_resolve_flag(self):
        cfg = GwConfig(auto_resolve=True)
        assert cfg.auto_resolve is True

    def test_empty_element_settings(self):
        cfg = GwConfig(element_settings={})
        assert len(cfg.element_settings) == 0

    def test_show_config_roundtrip(self, tmp_path):
        import tomli_w
        config_data = {
            "engine": "cp2k",
            "gw": {
                "basis_set_file": "/custom/path/BASIS",
                "ri_basis_set_file": "/custom/path/RI_BASIS",
                "potential_file": "/custom/path/POTENTIAL",
                "element_settings": {
                    "C": {"ri_basis": "RI_C_basis", "potential": "GTH-PBE-q4"},
                    "N": {"ri_basis": "RI_N_basis", "potential": "GTH-PBE-q5"},
                },
            },
        }
        config_file = tmp_path / "config.toml"
        with open(config_file, "wb") as f:
            tomli_w.dump(config_data, f)

        from aiida_relax_project.core.config import load_config
        config = load_config(config_file)
        assert config.gw.basis_set_file == "/custom/path/BASIS"
        assert config.gw.ri_basis_set_file == "/custom/path/RI_BASIS"
        assert config.gw.potential_file == "/custom/path/POTENTIAL"
        assert config.gw.cutoff == 400  # default from field
        # element_settings should merge with TOML values (overriding defaults)
        assert "C" in config.gw.element_settings
        assert config.gw.element_settings["C"].ri_basis == "RI_C_basis"
        assert config.gw.element_settings["C"].potential == "GTH-PBE-q4"
        # N should also be updated (custom toml should fully replace the dict)
        assert config.gw.element_settings["N"].ri_basis == "RI_N_basis"
        assert "B" not in config.gw.element_settings  # replaced by custom


class TestMetadataOptions:
    """Test MetadataOptions validation."""

    def test_valid(self):
        opts = MetadataOptions(num_machines=2, num_mpiprocs_per_machine=16)
        assert opts.num_machines == 2

    def test_num_machines_must_be_positive(self):
        with pytest.raises(ValueError):
            MetadataOptions(num_machines=0)

    def test_min_wallclock(self):
        with pytest.raises(ValueError):
            MetadataOptions(max_wallclock_seconds=30)

    def test_to_dict(self):
        opts = MetadataOptions(num_machines=2, num_mpiprocs_per_machine=16)
        d = opts.to_dict()
        assert d["resources"]["num_machines"] == 2
        assert d["resources"]["num_mpiprocs_per_machine"] == 16
        assert d["max_wallclock_seconds"] == 3600


class TestLoadConfig:
    """Test config loading from TOML files."""

    def test_load_from_file(self, tmp_path):
        config_data = {
            "engine": "cp2k",
            "code_label": "mycluster",
            "vasp": {"potential_family": "PBE.54"},
            "cp2k": {
                "kpoints_mesh": [2, 1, 2],
                "default_cutoff": 500,
            },
        }
        config_file = tmp_path / "config.toml"
        with open(config_file, "wb") as f:
            tomli_w.dump(config_data, f)

        config = load_config(config_file)
        assert config.engine == "cp2k"
        assert config.code_label == "mycluster"
        assert config.cp2k.default_cutoff == 500
        assert config.cp2k.kpoints_mesh == [2, 1, 2]

    def test_file_not_found_fallback(self):
        config = load_config("/nonexistent/config.toml")
        assert isinstance(config, ProjectConfig)

    def test_merge_configs(self):
        base = {"engine": "vasp", "vasp": {"default_encut": 400}}
        overlay = {"vasp": {"default_encut": 500, "default_isif": 2}}
        merged = _merge_configs(base, overlay)
        assert merged["engine"] == "vasp"
        assert merged["vasp"]["default_encut"] == 500
        assert merged["vasp"]["default_isif"] == 2
