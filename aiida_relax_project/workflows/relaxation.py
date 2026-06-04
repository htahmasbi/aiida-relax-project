"""Relaxation workflows supporting both VASP and CP2K engines."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiida import orm

from aiida import orm
from aiida.engine import ToContext, WorkChain
from aiida.plugins import WorkflowFactory

from aiida_relax_project.core.engine import EngineFactory
from aiida_relax_project.core.exceptions import StructureValidationError
from aiida_relax_project.core.logging import get_logger

logger = get_logger(__name__)


class MyRelaxLearningWorkChain(WorkChain):
    """Small wrapper around aiida-vasp RelaxWorkChain.

    Legacy WorkChain for backward compatibility with existing VASP relaxations.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("structure", valid_type=orm.StructureData)
        spec.input("vasp", valid_type=orm.Dict, help="Nested inputs for aiida-vasp relax workflow")

        spec.outline(
            cls.run_relax,
            cls.inspect_relax,
            cls.results,
        )

        spec.output("relaxed_structure", valid_type=orm.StructureData, required=False)
        spec.output("misc", valid_type=orm.Dict, required=False)

        spec.exit_code(
            600,
            "ERROR_RELAX_FAILED",
            message="The aiida-vasp relaxation workflow failed.",
        )

    def run_relax(self):
        inputs = self.inputs.vasp.get_dict()
        inputs["structure"] = self.inputs.structure

        logger.info("Submitting VASP relaxation for structure %s", self.inputs.structure.pk)
        VaspRelaxWorkChain = WorkflowFactory("vasp.relax")
        future = self.submit(VaspRelaxWorkChain, **inputs)
        return ToContext(relax=future)

    def inspect_relax(self):
        if not self.ctx.relax.is_finished_ok:
            logger.error("Relaxation failed with exit status %s", self.ctx.relax.exit_status)
            return self.exit_codes.ERROR_RELAX_FAILED
        logger.info("Relaxation completed successfully")

    def results(self):
        relax = self.ctx.relax

        if "relaxed_structure" in relax.outputs:
            self.out("relaxed_structure", relax.outputs.relaxed_structure)

        if "misc" in relax.outputs:
            self.out("misc", relax.outputs.misc)


class DynamicRelaxWorkChain(WorkChain):
    """Engine-agnostic relaxation WorkChain supporting VASP or CP2K.

    Automatically configures the appropriate relax workchain based on the
    specified engine, with optional generic parameter translation.

    Supports different relaxation types:
    - volume: Optimize cell volume (ISIF=3 for VASP, FULL for CP2K)
    - shape: Optimize cell shape and volume (ISIF=5 for VASP, ABC for CP2K)
    - positions: Optimize atomic positions only (ISIF=2 for VASP)
    - cell: Full cell optimization (ISIF=4 for VASP)

    Example usage:
        submit(DynamicRelaxWorkChain,
            engine=Str("vasp"),
            structure=structure,
            code=code,
            parameters=Dict({"encut": 500}),
            relaxation_type=Str("volume"),
            use_generic_params=Bool(True),
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
            "relaxation_type",
            valid_type=orm.Str,
            required=False,
            default=lambda: orm.Str("volume"),
            help="Relaxation type: 'volume', 'shape', 'positions', 'cell'",
        )

        spec.input(
            "convergence_criteria",
            valid_type=orm.Dict,
            required=False,
            help="Custom convergence criteria (engine-specific format).",
        )

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

        spec.outline(
            cls.setup,
            cls.run_relax,
            cls.inspect_relax,
            cls.results,
        )

        spec.output("relaxed_structure", valid_type=orm.StructureData, required=False)
        spec.output("misc", valid_type=orm.Dict, required=False)
        spec.output("engine_used", valid_type=orm.Str, required=False)

        spec.exit_code(
            600,
            "ERROR_RELAX_FAILED",
            message="The relaxation workflow failed.",
        )

    def setup(self):
        """Parse engine and validate inputs."""
        engine_str = self.inputs.engine.value.lower().strip()
        relax_type_str = self.inputs.relaxation_type.value.lower().strip()

        if engine_str not in ["vasp", "cp2k"]:
            logger.error("Unknown engine: %s", engine_str)
            self.abort_nowait(f"Unknown engine: {engine_str}. Must be 'vasp' or 'cp2k'.")
            return

        valid_relax_types = ["volume", "shape", "positions", "cell"]
        if relax_type_str not in valid_relax_types:
            self.abort_nowait(
                f"Unknown relaxation type: {relax_type_str}. "
                f"Must be one of: {', '.join(valid_relax_types)}"
            )
            return

        self.ctx.engine = engine_str
        self.ctx.relax_type = relax_type_str

        logger.info("Using engine: %s, relaxation type: %s", self.ctx.engine, self.ctx.relax_type)

        adapter = EngineFactory.create(self.ctx.engine)
        try:
            adapter.validate_structure(self.inputs.structure)
        except StructureValidationError as e:
            logger.error("Structure validation failed: %s", e)
            self.abort_nowait(str(e))

    def _build_inputs(self):
        """Build engine-specific inputs for the relax workchain."""
        engine = self.ctx.engine
        relax_type = self.ctx.relax_type
        adapter = EngineFactory.create(engine)

        use_generic = (
            self.inputs.use_generic_params.value
            if hasattr(self.inputs.use_generic_params, 'value')
            else self.inputs.use_generic_params
        )

        if use_generic:
            logger.debug("Translating generic parameters to %s relaxation format", engine)
            generic_params = self.inputs.parameters.get_dict()
            generic_params["run_type"] = "relax"
            params = adapter.build_parameters(generic_params, run_type="relax")
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

        relax_settings = adapter.get_relaxation_settings(relax_type)

        if engine == "vasp":
            inputs = {
                "structure": self.inputs.structure,
                "parameters": params,
                "potential_family": self.inputs.get("potential_family") or orm.Str("PBE.54"),
                "potential_mapping": self.inputs.get("potential_mapping") or orm.Dict(dict={}),
                "settings": orm.Dict(dict=relax_settings),
                "metadata": {"options": options},
            }

            if "kpoints" in self.inputs:
                inputs["kpoints"] = self.inputs.kpoints

        elif engine == "cp2k":
            inputs = {
                "structure": self.inputs.structure,
                "parameters": params,
                "metadata": {"options": options},
            }

            if "kpoints" in self.inputs:
                inputs["kpoints"] = self.inputs.kpoints

        if "convergence_criteria" in self.inputs:
            logger.debug("Using custom convergence criteria")

        return inputs

    def run_relax(self):
        """Submit the relaxation workchain."""
        WorkChainClass = EngineFactory.create(self.ctx.engine).get_workflow_class("relax")
        inputs = self._build_inputs()

        logger.info(
            "Submitting %s relaxation workflow for structure %s (type=%s)",
            self.ctx.engine.upper(),
            self.inputs.structure.pk,
            self.ctx.relax_type,
        )
        future = self.submit(WorkChainClass, **inputs)
        return ToContext(relax=future)

    def inspect_relax(self):
        """Check relaxation status."""
        relax = self.ctx.relax
        engine = self.ctx.engine.value.upper()

        if not relax.is_finished_ok:
            logger.error(
                "Relaxation failed with exit status %s: %s",
                relax.exit_status,
                relax.exit_message,
            )
            return self.exit_codes.ERROR_RELAX_FAILED

        logger.info("%s relaxation completed successfully", engine)

    def results(self):
        """Collect results."""
        relax = self.ctx.relax

        if "relaxed_structure" in relax.outputs:
            self.out("relaxed_structure", relax.outputs.relaxed_structure)

        if "misc" in relax.outputs:
            self.out("misc", relax.outputs.misc)

        self.out("engine_used", orm.Str(self.ctx.engine.value))
