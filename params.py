def create_param_dict(study):
    # partly clamped hyperbolic paraboloid (weak form)
    if study == 'hyperb_parab':
        param_dict = {
            'geometry': 'hyperb_parab', 
            'loading': 'gravity',
            'E': 1.,
            'thickness': 0.1,
            'shell_density': 1.,
            'nu': 0.3,
            'shear_factor': 5./6.,
            'bcs': 'ls',
            'N_col': 16384,
            'col_sampling': 'sobol',
            'epochs': 100,
            'opt_switch_epoch': 0,
            'FEM_sol_dir': 'FEM_sol/fenics_pred_hyperb_parab.csv',
            }
    # fully clamped hyperbolic paraboloid (strong form)
    elif study == 'hyperb_parab_strong':
        param_dict = {
            'geometry': 'hyperb_parab', 
            'loading': 'gravity',
            'E': 1.,
            'thickness': 0.1,
            'shell_density': 1.,
            'nu': 0.3,
            'shear_factor': 5./6.,
            'bcs': 'fc',
            'N_col': 2048,
            'col_sampling': 'sobol',
            # 'epochs': 100,
            'epochs': 10,
            'opt_switch_epoch': 0,
            'FEM_sol_dir': 'FEM_sol/fenics_pred_hyperb_parab_fully_clamped.csv',
            }
    # Scordelis-Lo roof (weak form)
    elif study == 'scordelis_lo':
        param_dict = {
            'geometry': 'scordelis_lo', 
            'loading': 'gravity',
            'E': 1.,
            'thickness': 0.005,
            'shell_density': 1.,
            'nu': 0.0,
            'shear_factor': 5./6.,
            'bcs': 'scordelis_lo',
            'N_col': 2**16,
            'col_sampling': 'sobol',
            'epochs': 1000,
            'opt_switch_epoch': 0,
            'FEM_sol_dir': 'FEM_sol/fenics_pred_scordelis_lo.csv',
            }
    # hemisphere under concentrated load (weak form)
    elif study == 'hemisphere':
        param_dict = {
            'geometry': 'hemisphere', 
            'loading': 'concentrated_load',
            'E': 1.,
            'thickness': 0.05,
            'shell_density': 1.,
            'nu': 0.3,
            'shear_factor': 5./6.,
            'bcs': 'fc_circular',
            'N_col': 280,
            'col_sampling': 'concentric',
            'epochs': 100,
            'opt_switch_epoch': 0,
            'FEM_sol_dir': 'FEM_sol/fenics_pred_hemisphere.csv',
            }
    else:
        raise ValueError('Invalid benchmark')
    return param_dict