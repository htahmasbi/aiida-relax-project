from aiida import orm
from aiida.engine import WorkChain, ToContext

from aiida_relax_project.workflows.single_point import VaspSinglePointWorkChain


class VaspVolumeScanWorkChain(WorkChain):
    """Run one VASP single-point workflow for each structure in a group."""

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

            self.report(f"Submitting calculation {label} for StructureData<{structure.pk}>.")
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
            self.report(f"{len(failed)} calculations failed.")
            return self.exit_codes.ERROR_SOME_CALCULATIONS_FAILED

        self.report("All volume-scan calculations finished successfully.")

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

                # The exact key can depend on aiida-vasp version/parser output.
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
