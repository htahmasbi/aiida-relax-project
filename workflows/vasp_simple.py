from aiida.engine import WorkChain
from aiida.plugins import CalculationFactory
from aiida import orm

VaspCalculation = CalculationFactory('vasp.vasp')

class MyVaspWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        # Define inputs
        spec.input('structure', valid_type=orm.StructureData)
        spec.input('code', valid_type=orm.AbstractCode)
        spec.input('parameters', valid_type=orm.Dict, help='The INCAR dictionary')
        spec.input('kpoints', valid_type=orm.KpointsData)
        spec.input('potential_family', valid_type=orm.Str)
        spec.input('potential_mapping', valid_type=orm.Dict)
        
        # Outline of the workflow
        spec.outline(
            cls.run_vasp,
            cls.inspect_vasp,
        )
        
        spec.output('retrieved_folder', valid_type=orm.FolderData)
        spec.output('output_parameters', valid_type=orm.Dict)

    def run_vasp(self):
        # Assemble the inputs for the actual VASP calculation
        inputs = {
            'code': self.inputs.code,
            'structure': self.inputs.structure,
            'parameters': self.inputs.parameters,
            'kpoints': self.inputs.kpoints,
            'potential_family': self.inputs.potential_family,
            'potential_mapping': self.inputs.potential_mapping,
            'metadata': {
                'options': {
                    'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 8},
                    'max_wallclock_seconds': 3600,
                    'withmpi': True,
                }
            }
        }
        
        # We use 'submit' so the WorkChain can continue or wait
        future = self.submit(VaspCalculation, **inputs)
        return self.to_context(vasp_job=future)

    def inspect_vasp(self):
        if not self.ctx.vasp_job.is_finished_ok:
            self.report(f"VASP failed with exit status {self.ctx.vasp_job.exit_status}")
            return
        
        self.out('output_parameters', self.ctx.vasp_job.outputs.output_parameters)
        self.report("VASP completed successfully!")
