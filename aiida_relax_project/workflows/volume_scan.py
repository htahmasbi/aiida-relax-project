from aiida import orm
from aiida.engine import WorkChain, ToContext, append_

from aiida_relax_project.workflows.single_point import VaspSinglePointWorkChain


class VaspVolumeScanWorkChain(WorkChain):
    """Run VASP single-point calculations for a group of structures."""

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
        self.report("Submitting VASP calculations for all structures in group.")
    
        calculations = []
    
        for index, structure in enumerate(self.inputs.structure_group.nodes):
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
    
            future = self.submit(VaspSinglePointWorkChain, **inputs)
            calculations.append(append_(future))
    
        return ToContext(calculations=calculations)

    def inspect_results(self):
        failed = [
            calc for calc in self.ctx.calculations
            if not calc.is_finished_ok
        ]

        if failed:
            self.report(f"{len(failed)} calculations failed.")
            return self.exit_codes.ERROR_SOME_CALCULATIONS_FAILED

        self.report("All scan calculations finished successfully.")

    def collect_results(self):
        energies = {}

        for calc in self.ctx.calculations:
            label = calc.inputs.structure.label or str(calc.inputs.structure.pk)

            try:
                output_parameters = calc.outputs.output_parameters.get_dict()
                energy = output_parameters.get("energy", None)
            except Exception:
                energy = None

            energies[label] = {
                "workflow_pk": calc.pk,
                "energy": energy,
            }

        self.out("energies", orm.Dict(dict=energies))
