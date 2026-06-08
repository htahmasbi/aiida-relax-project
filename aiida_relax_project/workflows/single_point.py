"""Single-point calculation workflows supporting both VASP and CP2K engines."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiida import orm

from aiida import orm
from aiida.engine import ToContext, WorkChain
from aiida.plugins import CalculationFactory

from aiida_relax_project.core.engine import EngineFactory
from aiida_relax_project.core.exceptions import (
    StructureValidationError,
)
from aiida_relax_project.core.logging import get_logger

logger = get_logger(__name__)


class VaspSinglePointWorkChain(WorkChain):
    """VASP single-point calculation WorkChain.

    Legacy WorkChain for backward compatibility with existing VASP calculations.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("structure", valid_type=orm.StructureData)
        spec.input("code", valid_type=orm.AbstractCode)
        spec.input("parameters", valid_type=orm.Dict)
        spec.input("kpoints", valid_type=orm.KpointsData)
        spec.input("potential_family", valid_type=orm.Str)
        spec.input("potential_mapping", valid_type=orm.Dict)

        spec.input(
            "metadata_options",
            valid_type=orm.Dict,
            required=False,
            help="Scheduler resources and walltime options.",
        )

        spec.outline(
            cls.run_vasp,
            cls.inspect_vasp,
            cls.results,
        )

        spec.output("output_parameters", valid_type=orm.Dict, required=False)
        spec.output("retrieved", valid_type=orm.FolderData, required=False)

        spec.exit_code(
            400,
            "ERROR_VASP_FAILED",
            message="The underlying VASP calculation failed.",
        )

    def run_vasp(self):
        options = {
            "resources": {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 8,
            },
            "max_wallclock_seconds": 3600,
            "withmpi": True,
        }

        if "metadata_options" in self.inputs:
            options.update(self.inputs.metadata_options.get_dict())

        inputs = {
            "code": self.inputs.code,
            "structure": self.inputs.structure,
            "parameters": self.inputs.parameters,
            "kpoints": self.inputs.kpoints,
            "potential_family": self.inputs.potential_family,
            "potential_mapping": self.inputs.potential_mapping,
            "metadata": {"options": options},
        }

        logger.info("Submitting VASP calculation for structure %s", self.inputs.structure.pk)
        VaspCalculation = CalculationFactory("vasp.vasp")
        future = self.submit(VaspCalculation, **inputs)
        return ToContext(calc_job=future)

    def inspect_vasp(self):
        calculation = self.ctx.calc_job

        if not calculation.is_finished_ok:
            logger.error(
                "VASP calculation failed with exit status %s: %s",
                calculation.exit_status,
                calculation.exit_message,
            )
            return self.exit_codes.ERROR_VASP_FAILED

        logger.info("VASP calculation completed successfully")

    def results(self):
        calculation = self.ctx.calc_job

        if "output_parameters" in calculation.outputs:
            self.out("output_parameters", calculation.outputs.output_parameters)

        if "retrieved" in calculation.outputs:
            self.out("retrieved", calculation.outputs.retrieved)


class Cp2kSinglePointWorkChain(WorkChain):
    """CP2K single-point calculation WorkChain.

    WorkChain for CP2K single-point energy calculations.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("structure", valid_type=orm.StructureData)
        spec.input("code", valid_type=orm.AbstractCode)
        spec.input("parameters", valid_type=orm.Dict)
        spec.input("kpoints", valid_type=orm.KpointsData, required=False)

        spec.input(
            "metadata_options",
            valid_type=orm.Dict,
            required=False,
            help="Scheduler resources and walltime options.",
        )

        spec.outline(
            cls.run_cp2k,
            cls.inspect_cp2k,
            cls.results,
        )

        spec.output("output_parameters", valid_type=orm.Dict, required=False)
        spec.output("retrieved", valid_type=orm.FolderData, required=False)

        spec.exit_code(
            410,
            "ERROR_CP2K_FAILED",
            message="The underlying CP2K calculation failed.",
        )

    def run_cp2k(self):
        options = {
            "resources": {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 16,
            },
            "max_wallclock_seconds": 3600,
            "withmpi": True,
        }

        if "metadata_options" in self.inputs:
            options.update(self.inputs.metadata_options.get_dict())

        inputs = {
            "code": self.inputs.code,
            "structure": self.inputs.structure,
            "parameters": self.inputs.parameters,
            "metadata": {"options": options},
        }

        if "kpoints" in self.inputs:
            inputs["kpoints"] = self.inputs.kpoints

        logger.info("Submitting CP2K calculation for structure %s", self.inputs.structure.pk)
        Cp2kCalculation = CalculationFactory("cp2k")
        future = self.submit(Cp2kCalculation, **inputs)
        return ToContext(calc_job=future)

    def inspect_cp2k(self):
        calculation = self.ctx.calc_job

        if not calculation.is_finished_ok:
            logger.error(
                "CP2K calculation failed with exit status %s: %s",
                calculation.exit_status,
                calculation.exit_message,
            )
            return self.exit_codes.ERROR_CP2K_FAILED

        logger.info("CP2K calculation completed successfully")

    def results(self):
        calculation = self.ctx.calc_job

        if "output_parameters" in calculation.outputs:
            self.out("output_parameters", calculation.outputs.output_parameters)

        if "retrieved" in calculation.outputs:
            self.out("retrieved", calculation.outputs.retrieved)


class DynamicSinglePointWorkChain(WorkChain):
    """Engine-agnostic single-point WorkChain supporting VASP or CP2K.

    This WorkChain automatically selects the appropriate engine and translates
    parameters as needed. Supports both legacy per-engine parameters and
    generic parameters that are auto-converted to the correct format.

    Example usage:
        # With generic parameters (recommended)
        submit(DynamicSinglePointWorkChain,
            engine=Str("vasp"),
            structure=structure,
            code=code,
            parameters=Dict({"encut": 500, "max_steps": 100}),
            use_generic_params=Bool(True),
        )

        # With engine-specific parameters
        submit(DynamicSinglePointWorkChain,
            engine=Str("cp2k"),
            structure=structure,
            code=code,
            parameters=cp2k_specific_params,
            use_generic_params=Bool(False),
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
        spec.input("structure", valid_type=orm.StructureData)
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
            help="If True, parameters are treated as generic and auto-translated to engine format.",
        )

        spec.outline(
            cls.setup,
            cls.run_calculation,
            cls.inspect_calculation,
            cls.results,
        )

        spec.output("output_parameters", valid_type=orm.Dict, required=False)
        spec.output("retrieved", valid_type=orm.FolderData, required=False)
        spec.output("engine_used", valid_type=orm.Str, required=False)

        spec.exit_code(
            400,
            "ERROR_CALCULATION_FAILED",
            message="The underlying calculation failed.",
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

        adapter = EngineFactory.create(self.ctx.engine)
        try:
            adapter.validate_structure(self.inputs.structure)
        except StructureValidationError as e:
            logger.error("Structure validation failed: %s", e)
            self.abort_nowait(str(e))

    def _build_inputs(self):
        """Build engine-specific inputs."""
        engine = self.ctx.engine
        adapter = EngineFactory.create(engine)

        use_generic = (
            self.inputs.use_generic_params.value
            if hasattr(self.inputs.use_generic_params, 'value')
            else self.inputs.use_generic_params
        )

        if use_generic:
            logger.debug("Translating generic parameters to %s format", engine)
            params = adapter.build_parameters(
                self.inputs.parameters.get_dict(),
                run_type="energy"
            )
        else:
            params = self.inputs.parameters

        default_resources = adapter.get_default_resources()
        options = {
            "resources": default_resources,
            "max_wallclock_seconds": 3600,
            "withmpi": True,
        }

        if "metadata_options" in self.inputs:
            options.update(self.inputs.metadata_options.get_dict())

        inputs = {
            "code": self.inputs.code,
            "structure": self.inputs.structure,
            "parameters": params,
            "metadata": {"options": options},
        }

        if engine == "vasp":
            if "kpoints" in self.inputs:
                inputs["kpoints"] = self.inputs.kpoints
            if "potential_family" in self.inputs:
                inputs["potential_family"] = self.inputs.potential_family
            if "potential_mapping" in self.inputs:
                inputs["potential_mapping"] = self.inputs.potential_mapping
        elif engine == "cp2k":
            if "kpoints" in self.inputs:
                inputs["kpoints"] = self.inputs.kpoints

        return inputs

    def run_calculation(self):
        """Submit the calculation."""
        CalcClass = EngineFactory.create(self.ctx.engine).get_calculation_class()
        inputs = self._build_inputs()

        logger.info(
            "Submitting %s calculation for structure %s",
            self.ctx.engine.upper(),
            self.inputs.structure.pk,
        )
        future = self.submit(CalcClass, **inputs)
        return ToContext(calc_job=future)

    def inspect_calculation(self):
        """Check calculation status."""
        calculation = self.ctx.calc_job
        engine = self.ctx.engine.upper()

        if not calculation.is_finished_ok:
            logger.error(
                "%s calculation failed with exit status %s: %s",
                engine,
                calculation.exit_status,
                calculation.exit_message,
            )
            return self.exit_codes.ERROR_CALCULATION_FAILED

        logger.info("%s calculation completed successfully", engine)

    def results(self):
        """Collect results."""
        calculation = self.ctx.calc_job

        if "output_parameters" in calculation.outputs:
            self.out("output_parameters", calculation.outputs.output_parameters)

        if "retrieved" in calculation.outputs:
            self.out("retrieved", calculation.outputs.retrieved)

        self.out("engine_used", orm.Str(self.ctx.engine))
