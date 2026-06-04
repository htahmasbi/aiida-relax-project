"""Volume scan workflows supporting both VASP and CP2K engines."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiida import orm

from aiida import orm
from aiida.engine import ToContext, WorkChain

from aiida_relax_project.core.logging import get_logger
from aiida_relax_project.workflows.single_point import DynamicSinglePointWorkChain

logger = get_logger(__name__)


class DynamicVolumeScanWorkChain(WorkChain):
    """Engine-agnostic volume scan WorkChain for multiple structures.

    Submits single-point calculations for all structures in a group using
    either VASP or CP2K based on the specified engine.

    Features:
    - Automatic engine selection
    - Generic parameter translation
    - Continue-on-failure option for large batches
    - Comprehensive result collection

    Example usage:
        submit(DynamicVolumeScanWorkChain,
            engine=Str("vasp"),
            structure_group=group,
            code=code,
            parameters=Dict({"encut": 500}),
            use_generic_params=Bool(True),
            continue_on_failure=Bool(False),
        )
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input(
            "engine",
            valid_type=orm.Str,
            help="Calculation engine: 'vasp' or 'cp2k'",
        )
        spec.input("structure_group", valid_type=orm.Group)
        spec.input("code", valid_type=orm.AbstractCode)
        spec.input("parameters", valid_type=orm.Dict)
        spec.input("kpoints", valid_type=orm.KpointsData, required=False)
        spec.input("potential_family", valid_type=orm.Str, required=False)
        spec.input("potential_mapping", valid_type=orm.Dict, required=False)

        spec.input(
            "metadata_options",
            valid_type=orm.Dict,
            required=False,
            help="Scheduler resources and walltime options.",
        )

        spec.input(
            "use_generic_params",
            valid_type=orm.Bool,
            required=False,
            help="If True, parameters are treated as generic and auto-translated.",
        )

        spec.input(
            "continue_on_failure",
            valid_type=orm.Bool,
            required=False,
            help="If True, continue processing other structures if one fails.",
        )

        spec.outline(
            cls.setup,
            cls.run_calculations,
            cls.inspect_results,
            cls.collect_results,
        )

        spec.output("energies", valid_type=orm.Dict, required=False)
        spec.output("failed_structures", valid_type=orm.List, required=False)
        spec.output("engine_used", valid_type=orm.Str, required=False)

        spec.exit_code(
            500,
            "ERROR_SOME_CALCULATIONS_FAILED",
            message="Some calculations in the volume scan failed.",
        )

    def setup(self):
        """Parse engine and validate inputs."""
        engine_str = self.inputs.engine.value.lower().strip()

        if engine_str not in ["vasp", "cp2k"]:
            logger.error("Unknown engine: %s", engine_str)
            self.abort_nowait(f"Unknown engine: {engine_str}. Must be 'vasp' or 'cp2k'.")
            return

        self.ctx.engine = engine_str
        logger.info("Using engine: %s", self.ctx.engine)

        structures = list(self.inputs.structure_group.nodes)
        logger.info("Found %d structures to process with engine %s", len(structures), engine_str)

        if not structures:
            self.abort_nowait("Structure group is empty.")
            return

        self.ctx.structures = structures
        self.ctx.labels = []
        self.ctx.failed = []

    def run_calculations(self):
        """Submit one child workflow for each structure."""
        context = {}
        engine = self.ctx.engine

        for index, structure in enumerate(self.ctx.structures):
            label = f"calc_{index}"
            self.ctx.labels.append(label)

            inputs = {
                "engine": orm.Str(engine),
                "code": self.inputs.code,
                "structure": structure,
                "parameters": self.inputs.parameters,
            }

            use_generic = (
                self.inputs.use_generic_params.value
                if hasattr(self.inputs.use_generic_params, 'value')
                else self.inputs.use_generic_params
            )
            inputs["use_generic_params"] = orm.Bool(use_generic)

            if "kpoints" in self.inputs:
                inputs["kpoints"] = self.inputs.kpoints

            if "potential_family" in self.inputs:
                inputs["potential_family"] = self.inputs.potential_family

            if "potential_mapping" in self.inputs:
                inputs["potential_mapping"] = self.inputs.potential_mapping

            if "metadata_options" in self.inputs:
                inputs["metadata_options"] = self.inputs.metadata_options

            logger.info(
                "Submitting %s calculation %s for StructureData<%s>",
                str(engine).upper(),
                label,
                structure.pk,
            )
            context[label] = self.submit(DynamicSinglePointWorkChain, **inputs)

        return ToContext(**context)

    def inspect_results(self):
        """Check whether all child workflows finished successfully."""
        failed = []

        for label in self.ctx.labels:
            calculation = self.ctx[label]

            if not calculation.is_finished_ok:
                failed.append({
                    "label": label,
                    "pk": calculation.pk,
                    "exit_status": calculation.exit_status,
                    "exit_message": calculation.exit_message,
                })
                logger.warning(
                    "Calculation %s (pk=%s) failed: %s",
                    label,
                    calculation.pk,
                    calculation.exit_message,
                )

        self.ctx.failed = failed

        continue_on_failure = (
            self.inputs.continue_on_failure.value
            if hasattr(self.inputs.continue_on_failure, 'value')
            else self.inputs.continue_on_failure
        )

        if failed:
            if continue_on_failure:
                logger.info(
                    "%d calculations failed, continuing as requested",
                    len(failed),
                )
            else:
                logger.error("%d calculations failed", len(failed))
                return self.exit_codes.ERROR_SOME_CALCULATIONS_FAILED

        logger.info("All volume-scan calculations finished successfully")

    def collect_results(self):
        """Collect energies from child workflow outputs."""
        energies = {}

        for label in self.ctx.labels:
            calculation = self.ctx[label]
            structure = calculation.inputs.structure

            structure_label = structure.label or f"structure_{structure.pk}"

            energy = None

            if "output_parameters" in calculation.outputs:
                output_parameters = calculation.outputs.output_parameters.get_dict()
                energy = self._extract_energy(output_parameters)

            energies[structure_label] = {
                "structure_pk": structure.pk,
                "workflow_pk": calculation.pk,
                "energy": energy,
                "status": "ok" if calculation.is_finished_ok else "failed",
            }

        self.out("energies", orm.Dict(dict=energies))

        if self.ctx.failed:
            self.out(
                "failed_structures",
                orm.List(list=[
                    self.ctx[failed["label"]].inputs.structure.pk
                    for failed in self.ctx.failed
                ]),
            )

        self.out("engine_used", orm.Str(self.ctx.engine))
        logger.info("Collected results for %d structures", len(energies))

    def _extract_energy(self, output_parameters: dict) -> float | None:
        """Extract energy from output parameters (engine-agnostic)."""
        energy_keys = [
            ("energy", None),
            ("total_energy", None),
            ("energy_free", None),
            ("energy_total", None),
            ("total_energies", "energy_extrapolated"),
            ("total_energies", "energy_free"),
            ("Eh", None),
        ]

        for key, subkey in energy_keys:
            if subkey is None:
                if key in output_parameters:
                    return float(output_parameters[key])
            else:
                if key in output_parameters and isinstance(output_parameters[key], dict):
                    if subkey in output_parameters[key]:
                        return float(output_parameters[key][subkey])

        return None


class VaspVolumeScanWorkChain(WorkChain):
    """Legacy VASP-only volume scan (kept for backward compatibility)."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("structure_group", valid_type=orm.Group)
        spec.input("code", valid_type=orm.AbstractCode)
        spec.input("parameters", valid_type=orm.Dict)
        spec.input("kpoints", valid_type=orm.KpointsData)
        spec.input("potential_family", valid_type=orm.Str)
        spec.input("potential_mapping", valid_type=orm.Dict)
        spec.input("metadata_options", valid_type=orm.Dict, required=False)

        spec.outline(
            cls.run_calculations,
            cls.inspect_results,
            cls.collect_results,
        )

        spec.output("energies", valid_type=orm.Dict, required=False)

        spec.exit_code(
            500,
            "ERROR_SOME_CALCULATIONS_FAILED",
            message="At least one VASP calculation in the scan failed.",
        )

    def run_calculations(self):
        """Submit one child workflow for each structure."""
        from aiida_relax_project.workflows.single_point import VaspSinglePointWorkChain

        self.ctx.labels = []
        context = {}

        for index, structure in enumerate(self.inputs.structure_group.nodes):
            label = f"calc_{index}"
            self.ctx.labels.append(label)

            inputs = {
                "code": self.inputs.code,
                "structure": structure,
                "parameters": self.inputs.parameters,
                "kpoints": self.inputs.kpoints,
                "potential_family": self.inputs.potential_family,
                "potential_mapping": self.inputs.potential_mapping,
            }

            if "metadata_options" in self.inputs:
                inputs["metadata_options"] = self.inputs.metadata_options

            logger.info("Submitting VASP calculation %s for StructureData<%s>", label, structure.pk)
            context[label] = self.submit(VaspSinglePointWorkChain, **inputs)

        return ToContext(**context)

    def inspect_results(self):
        """Check whether all child workflows finished successfully."""
        failed = []

        for label in self.ctx.labels:
            calculation = self.ctx[label]

            if not calculation.is_finished_ok:
                failed.append(calculation)

        if failed:
            logger.error("%d calculations failed", len(failed))
            return self.exit_codes.ERROR_SOME_CALCULATIONS_FAILED

        logger.info("All VASP volume-scan calculations finished successfully")

    def collect_results(self):
        """Collect energies from child workflow outputs."""
        energies = {}

        for label in self.ctx.labels:
            calculation = self.ctx[label]
            structure = calculation.inputs.structure

            structure_label = structure.label or f"structure_{structure.pk}"

            energy = None

            if "output_parameters" in calculation.outputs:
                output_parameters = calculation.outputs.output_parameters.get_dict()
                energy = (
                    output_parameters.get("energy")
                    or output_parameters.get("total_energies", {}).get("energy_extrapolated")
                    or output_parameters.get("total_energies", {}).get("energy_free")
                )

            energies[structure_label] = {
                "structure_pk": structure.pk,
                "workflow_pk": calculation.pk,
                "energy": energy,
            }

        self.out("energies", orm.Dict(dict=energies))
