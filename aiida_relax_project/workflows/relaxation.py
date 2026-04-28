from aiida import orm
from aiida.engine import WorkChain, ToContext
from aiida.plugins import WorkflowFactory

VaspRelaxWorkChain = WorkflowFactory("vasp.relax")


class MyRelaxLearningWorkChain(WorkChain):
    """Small wrapper around aiida-vasp RelaxWorkChain."""

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

        future = self.submit(VaspRelaxWorkChain, **inputs)
        return ToContext(relax=future)

    def inspect_relax(self):
        if not self.ctx.relax.is_finished_ok:
            return self.exit_codes.ERROR_RELAX_FAILED

    def results(self):
        relax = self.ctx.relax

        if "relaxed_structure" in relax.outputs:
            self.out("relaxed_structure", relax.outputs.relaxed_structure)

        if "misc" in relax.outputs:
            self.out("misc", relax.outputs.misc)
