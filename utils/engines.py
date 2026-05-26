from aiida.plugins import WorkflowFactory, CalculationFactory
from aiida.orm import Dict

def get_engine_launcher(engine_name):
    """Returns the correct WorkChain and dynamic input generator."""
    engine = engine_name.lower().strip()
    
    if engine == 'vasp':
        return {
            'workchain': WorkflowFactory('vasp.vasp'),
            'builder_func': build_vasp_inputs
        }
    elif engine == 'cp2k':
        return {
            'workchain': WorkflowFactory('cp2k.cp2k'),
            'builder_func': build_cp2k_inputs
        }
    else:
        raise ValueError(f"Unknown electronic structure engine: {engine_name}")

def build_vasp_inputs(structure, parameters):
    """Translates generic parameters into VASP INCAR format."""
    return {
        'parameters': Dict(dict={
            'incar': {
                'IBRION': 2, 'ISIF': 3,
                'NSW': parameters.get('max_steps', 50),
                'EDIFF': parameters.get('energy_tolerance', 1e-5)
            }
        })
        # Note: You'll also append your KPOINTS and POTCAR maps here
    }

def build_cp2k_inputs(structure, parameters):
    """Translates generic parameters into CP2K nested blocks."""
    return {
        'parameters': Dict(dict={
            'GLOBAL': {'RUN_TYPE': 'GEO_OPT'},
            'FORCE_EVAL': {
                'METHOD': 'Quickstep',
                'DFT': {
                    'XC': {'XC_FUNCTIONAL': {'PBE': {}}},
                    'MGRID': {'CUTOFF': parameters.get('cutoff', 400)}
                }
            },
            'MOTION': {
                'GEO_OPT': {'MAX_ITER': parameters.get('max_steps', 50)}
            }
        })
    }
