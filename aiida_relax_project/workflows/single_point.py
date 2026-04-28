from aiida import orm
from aiida.engine import WorkChain, ToContext
from aiida.plugins import CalculationFactory

VaspCalculation = CalculationFactory("vasp.vasp")


class VaspSinglePointWorkChain(WorkChain):
    """Minimal educational WorkChain wrapping one aiida-vasp CalcJob."""

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
            "metadata": {
                "options": options,
            },
        }

        self.report("Submitting VASP calculation.")
        future = self.submit(VaspCalculation, **inputs)
        return ToContext(vasp_job=future)

    def inspect_vasp(self):
        calculation = self.ctx.vasp_job

        if not calculation.is_finished_ok:
            self.report(
                f"VASP calculation failed with exit status "
                f"{calculation.exit_status}: {calculation.exit_message}"
            )
            return self.exit_codes.ERROR_VASP_FAILED

        self.report("VASP calculation finished successfully.")

    def results(self):
        calculation = self.ctx.vasp_job

        if "output_parameters" in calculation.outputs:
            self.out("output_parameters", calculation.outputs.output_parameters)

        if "retrieved" in calculation.outputs:
            self.out("retrieved", calculation.outputs.retrieved)
