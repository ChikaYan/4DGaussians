_base_ = './dnerf_default.py'

ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 32,
     'resolution': [64, 64, 64, 75]
    }
)


OptimizationParams = dict(

    coarse_iterations = 1500,
    deformation_lr_init = 0.00016,
    deformation_lr_final = 0.0000016,
    deformation_lr_delay_mult = 0.01,
    grid_lr_init = 0.0016,
    grid_lr_final = 0.000016,
    iterations = 90000,
    pruning_interval = 8000,
    percent_dense = 0.0,
    render_process=False,
    # no_do=False,
    no_dshs=True,
    densification_interval=200,
    densify_grad_threshold_fine_init=0.0004,
    densify_until_iter=15000,
    
    opacity_reset_interval=4116

)