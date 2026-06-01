"""Workflow builders for constructing AiiDA workflow inputs."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aiida import orm
    from aiida.orm import StructureData, Dict, KpointsData, AbstractCode, Str
    from aiida_relax_project.core.enums import EngineType, RelaxType

from aiida import orm

from aiida_relax_project.core.engine import EngineFactory, BaseEngineAdapter
from aiida_relax_project.core.config import ProjectConfig, _merge_configs as _deep_merge
from aiida_relax_project.core.enums import EngineType

logger = logging.getLogger(__name__)


class BaseWorkflowBuilder(ABC):
    """Abstract base class for workflow builders."""

    def __init__(self, config: Optional[ProjectConfig] = None) -> None:
        self._config = config

    @property
    def config(self) -> ProjectConfig:
        """Get configuration, loading default if needed."""
        if self._config is None:
            from aiida_relax_project.core.config import get_config
            self._config = get_config()
        return self._config

    @abstractmethod
    def build_inputs(
        self,
        structure: "StructureData",
        code: "AbstractCode",
        parameters: "Dict",
        **kwargs,
    ) -> dict[str, Any]:
        """Build the complete input dictionary for the workflow."""
        raise NotImplementedError


class SinglePointBuilder(BaseWorkflowBuilder):
    """Builder for single-point calculation workflows."""

    def build_inputs(
        self,
        structure: "StructureData",
        code: "AbstractCode",
        parameters: "Dict",
        engine: Optional[EngineType] = None,
        kpoints: Optional["KpointsData"] = None,
        use_generic_params: bool = False,
        metadata_options: Optional[dict] = None,
        **extra_inputs,
    ) -> dict[str, Any]:
        """Build inputs for a single-point calculation.

        Args:
            structure: The structure to calculate
            code: AiiDA code for the calculation
            parameters: Calculation parameters
            engine: Engine type (uses config default if not specified)
            kpoints: K-points data
            use_generic_params: Whether to auto-translate parameters
            metadata_options: Scheduler options
            **extra_inputs: Additional engine-specific inputs

        Returns:
            Complete inputs dictionary for the workflow
        """
        engine = engine or self.config.engine
        adapter = EngineFactory.create(engine)

        adapter.validate_structure(structure)

        if use_generic_params:
            parameters = adapter.build_parameters(parameters.get_dict(), run_type="energy")

        kpoints = kpoints or adapter.build_kpoints(self.config.get_kpoints_mesh())

        inputs: dict[str, Any] = {
            "structure": structure,
            "code": code,
            "parameters": parameters,
            "kpoints": kpoints,
            "use_generic_params": orm.Bool(use_generic_params),
            "metadata_options": orm.Dict(dict=metadata_options or self.config.metadata_options.to_dict()),
        }

        if engine == "vasp":
            inputs["potential_family"] = orm.Str(self.config.vasp.potential_family)
            inputs["potential_mapping"] = orm.Dict(dict=self.config.vasp.potential_mapping)
            if use_generic_params and self.config.vasp.raw_incar:
                params_dict = inputs["parameters"].get_dict()
                params_dict.setdefault("raw_incar", {})
                params_dict["raw_incar"].update(self.config.vasp.raw_incar)
                inputs["parameters"] = orm.Dict(dict=params_dict)

        if engine == "cp2k" and use_generic_params:
            params_dict = inputs["parameters"].get_dict()
            if self.config.cp2k.basis_set_mapping:
                params_dict.setdefault("basis_set_mapping", {})
                params_dict["basis_set_mapping"].update(self.config.cp2k.basis_set_mapping)
            if self.config.cp2k.potential_mapping:
                params_dict.setdefault("potential_mapping", {})
                params_dict["potential_mapping"].update(self.config.cp2k.potential_mapping)
            params_dict.setdefault("basis_set_file", self.config.cp2k.basis_set_file)
            params_dict.setdefault("potential_file", self.config.cp2k.potential_file)
            if self.config.cp2k.raw_parameters:
                params_dict.setdefault("raw_parameters", {})
                params_dict["raw_parameters"] = _deep_merge(
                    params_dict["raw_parameters"], self.config.cp2k.raw_parameters
                )
            inputs["parameters"] = orm.Dict(dict=params_dict)

        inputs.update(extra_inputs)
        return inputs


class RelaxationBuilder(BaseWorkflowBuilder):
    """Builder for relaxation workflows."""

    def build_inputs(
        self,
        structure: "StructureData",
        code: "AbstractCode",
        parameters: "Dict",
        engine: Optional[EngineType] = None,
        relaxation_type: Optional["RelaxType"] = None,
        kpoints: Optional["KpointsData"] = None,
        use_generic_params: bool = False,
        convergence_criteria: Optional["Dict"] = None,
        metadata_options: Optional[dict] = None,
        **extra_inputs,
    ) -> dict[str, Any]:
        """Build inputs for a relaxation calculation.

        Args:
            structure: The structure to relax
            code: AiiDA code for the calculation
            parameters: Calculation parameters
            engine: Engine type (uses config default if not specified)
            relaxation_type: Type of relaxation (volume, shape, positions, cell)
            kpoints: K-points data
            use_generic_params: Whether to auto-translate parameters
            convergence_criteria: Custom convergence criteria
            metadata_options: Scheduler options
            **extra_inputs: Additional workflow inputs

        Returns:
            Complete inputs dictionary for the workflow
        """
        engine = engine or self.config.engine
        relaxation_type = relaxation_type or self.config.relax.relaxation_type
        adapter = EngineFactory.create(engine)

        adapter.validate_structure(structure)

        if use_generic_params:
            params_dict = parameters.get_dict()
            params_dict["run_type"] = "relax"
            if engine == "vasp" and self.config.vasp.raw_incar:
                params_dict.setdefault("raw_incar", {})
                params_dict["raw_incar"].update(self.config.vasp.raw_incar)
            if engine == "cp2k":
                if self.config.cp2k.basis_set_mapping:
                    params_dict.setdefault("basis_set_mapping", {})
                    params_dict["basis_set_mapping"].update(self.config.cp2k.basis_set_mapping)
                if self.config.cp2k.potential_mapping:
                    params_dict.setdefault("potential_mapping", {})
                    params_dict["potential_mapping"].update(self.config.cp2k.potential_mapping)
                params_dict.setdefault("basis_set_file", self.config.cp2k.basis_set_file)
                params_dict.setdefault("potential_file", self.config.cp2k.potential_file)
                if self.config.cp2k.raw_parameters:
                    params_dict.setdefault("raw_parameters", {})
                    params_dict["raw_parameters"] = _deep_merge(
                        params_dict["raw_parameters"], self.config.cp2k.raw_parameters
                    )
            parameters = adapter.build_parameters(params_dict, run_type="relax")

        kpoints = kpoints or adapter.build_kpoints(self.config.get_kpoints_mesh())

        inputs: dict[str, Any] = {
            "structure": structure,
            "code": code,
            "parameters": parameters,
            "engine": orm.Str(engine),
            "relaxation_type": orm.Str(relaxation_type),
            "use_generic_params": orm.Bool(use_generic_params),
            "metadata_options": orm.Dict(dict=metadata_options or self.config.metadata_options.to_dict()),
        }

        if kpoints is not None:
            inputs["kpoints"] = kpoints

        if convergence_criteria:
            inputs["convergence_criteria"] = convergence_criteria

        if engine == "vasp":
            inputs["potential_family"] = orm.Str(self.config.vasp.potential_family)
            inputs["potential_mapping"] = orm.Dict(dict=self.config.vasp.potential_mapping)

        inputs.update(extra_inputs)
        return inputs


class VolumeScanBuilder(BaseWorkflowBuilder):
    """Builder for volume scan workflows."""

    def build_inputs(
        self,
        structure_group: "orm.Group",
        code: "AbstractCode",
        parameters: "Dict",
        engine: Optional[EngineType] = None,
        kpoints: Optional["KpointsData"] = None,
        use_generic_params: bool = False,
        continue_on_failure: bool = False,
        metadata_options: Optional[dict] = None,
        **extra_inputs,
    ) -> dict[str, Any]:
        """Build inputs for a volume scan.

        Args:
            structure_group: Group containing structures to scan
            code: AiiDA code for calculations
            parameters: Calculation parameters
            engine: Engine type (uses config default if not specified)
            kpoints: K-points data
            use_generic_params: Whether to auto-translate parameters
            continue_on_failure: Continue processing if calculations fail
            metadata_options: Scheduler options
            **extra_inputs: Additional workflow inputs

        Returns:
            Complete inputs dictionary for the workflow
        """
        engine = engine or self.config.engine
        adapter = EngineFactory.create(engine)

        kpoints = kpoints or adapter.build_kpoints(self.config.get_kpoints_mesh())

        inputs: dict[str, Any] = {
            "structure_group": structure_group,
            "code": code,
            "parameters": parameters,
            "engine": orm.Str(engine),
            "use_generic_params": orm.Bool(use_generic_params),
            "continue_on_failure": orm.Bool(continue_on_failure),
            "metadata_options": orm.Dict(dict=metadata_options or self.config.metadata_options.to_dict()),
        }

        if kpoints is not None:
            inputs["kpoints"] = kpoints

        if engine == "vasp":
            inputs["potential_family"] = orm.Str(self.config.vasp.potential_family)
            inputs["potential_mapping"] = orm.Dict(dict=self.config.vasp.potential_mapping)
            if use_generic_params and self.config.vasp.raw_incar:
                vasp_params = inputs["parameters"].get_dict()
                vasp_params.setdefault("raw_incar", {})
                vasp_params["raw_incar"].update(self.config.vasp.raw_incar)
                inputs["parameters"] = orm.Dict(dict=vasp_params)

        if engine == "cp2k" and use_generic_params:
            params_dict = inputs["parameters"].get_dict()
            if self.config.cp2k.basis_set_mapping:
                params_dict.setdefault("basis_set_mapping", {})
                params_dict["basis_set_mapping"].update(self.config.cp2k.basis_set_mapping)
            if self.config.cp2k.potential_mapping:
                params_dict.setdefault("potential_mapping", {})
                params_dict["potential_mapping"].update(self.config.cp2k.potential_mapping)
            params_dict.setdefault("basis_set_file", self.config.cp2k.basis_set_file)
            params_dict.setdefault("potential_file", self.config.cp2k.potential_file)
            if self.config.cp2k.raw_parameters:
                params_dict.setdefault("raw_parameters", {})
                params_dict["raw_parameters"] = _deep_merge(
                    params_dict["raw_parameters"], self.config.cp2k.raw_parameters
                )
            inputs["parameters"] = orm.Dict(dict=params_dict)

        inputs.update(extra_inputs)
        return inputs


def create_example_structure(element: str = "Si") -> "orm.StructureData":
    """Create a simple example structure for testing.

    Args:
        element: Element symbol for the structure

    Returns:
        Stored StructureData instance
    """
    from ase.build import bulk

    from aiida.plugins import DataFactory

    StructureData = DataFactory("core.structure")

    ase_structure = bulk(element, a=5.43)
    structure = StructureData(ase=ase_structure)
    structure.label = f"Example {element} structure"
    structure.store()

    return structure


def fetch_structures_from_optimade(
    group_label: str,
    elements: Optional[list[str]] = None,
    max_structures: int = 5,
    modifier: Optional[callable] = None,
) -> "orm.Group":
    """Fetch structures from OPTIMADE and add to a group.

    Args:
        group_label: Label for the group
        elements: Required elements (e.g., ["B", "N"])
        max_structures: Maximum number of structures to fetch
        modifier: Optional function to modify fetched structures

    Returns:
        The group with added structures
    """
    from aiida import orm
    from aiida_relax_project.datasets.mc2d_optimade import fetch_mc2d_structures
    from aiida_relax_project.transformations.structures import rotate_xy_to_xz, make_supercell_3x3

    try:
        group = orm.Group.collection.get(label=group_label)
        logger.info(f"Found existing group '{group_label}' with {len(group.nodes)} nodes")
        return group
    except Exception:
        logger.info(f"Creating new group '{group_label}'")
        group, _ = orm.Group.collection.get_or_create(group_label)

    if elements:
        filter_str = f'elements HAS ALL "{'","'.join(elements)}" AND nelements={len(elements)}'
    else:
        filter_str = 'nelements=2'

    if modifier is None:
        modifier = lambda s: rotate_xy_to_xz(make_supercell_3x3(s), vacuum=20.0)

    try:
        structures = fetch_mc2d_structures(
            optimade_filter=filter_str,
            max_structures=max_structures,
            modifier=modifier,
        )

        for item in structures:
            from aiida.plugins import DataFactory
            StructureData = DataFactory("core.structure")

            pymatgen_structure = item["structure"]
            structure = StructureData(pymatgen=pymatgen_structure)
            structure.label = f"{item.get('formula', 'Unknown')} {item['id']}"
            structure.store()
            group.add_nodes(structure)

        logger.info(f"Added {len(structures)} structures to group '{group_label}'")

    except Exception as e:
        logger.warning(f"Failed to fetch from OPTIMADE: {e}")

    return group