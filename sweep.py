from argparse import ArgumentParser, Namespace
import torch
from train import setup_seed, scene_reconstruction
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from utils.general_utils import safe_state
from gaussian_renderer import render, network_gui
import sys, os
from scene import Scene, GaussianModel


import wandb
import shutil
from pathlib import Path
from copy import deepcopy
from utils.timer import Timer

from render import render_sets

from metrics import evaluate


PROJECT = 'phenoeye_static'
SWEEP_ID = 'c6hpf45w'
BASE_CONF = 'arguments/dnerf/long2.py'
SCENE = '/home/tw554/4DGaussians/data/china/cabbage3'

N_JOB = 50



def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname, args):
    conf = {}
    conf.update(dataset.__dict__)
    conf.update(hyper.__dict__)
    conf.update(opt.__dict__)

    wandb.init(
        project="phenoeye_static", 
        name=expname, 
        group=Path(dataset.source_path).name, 
        config=conf, 
        )
    



    # write wandb id for eval logging
    wandb_id_path = Path(args.model_path) / 'wandb_id.txt'
    wandb_id_path.parent.mkdir(exist_ok=True, parents=True)
    with wandb_id_path.open('w') as f:
        f.write(wandb.run.id)
    
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)
    timer.start()
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                            checkpoint_iterations, checkpoint, debug_from,
                            gaussians, scene, "coarse", None, opt.coarse_iterations,timer)
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                        checkpoint_iterations, checkpoint, debug_from,
                        gaussians, scene, "fine", None, opt.iterations,timer)



def train_agent(config=None):
    with wandb.init(config=config):

        torch.cuda.empty_cache()
        parser = ArgumentParser(description="Training script parameters")
        setup_seed(6666)
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)
        hp = ModelHiddenParams(parser)
        parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
        parser.add_argument("--quiet", action="store_true")
        parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
        parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
        parser.add_argument("--start_checkpoint", type=str, default = None)
        parser.add_argument('--debug_from', type=int, default=-1)
        parser.add_argument("--expname", type=str, default = "")
        parser.add_argument("--configs", type=str, default = "")
        
        args = parser.parse_args(sys.argv[1:])
        args.configs = BASE_CONF
        if args.configs:
            import mmcv
            from utils.params_utils import merge_hparams
            config = mmcv.Config.fromfile(args.configs)
            args = merge_hparams(args, config)
            # args = merge_hparams(args, wandb.config)

        for k in wandb.config:
            setattr(args, k, wandb.config[k])
        print("Optimizing " + args.model_path)
        args.save_iterations.append(args.iterations)

        args.source_path = SCENE
        

        # modify the confs
        args.expname = f"sweep/{SWEEP_ID}/{wandb.run.id}"

        if not args.model_path:
            unique_str = args.expname
            args.model_path = os.path.join("./output/", unique_str)

        model_path = args.model_path


        training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname, args)

        render_sets(lp.extract(args), hp.extract(args), -1, pp.extract(args), True, False, True)

        evaluate([model_path])




        # clean up logged files
        # shutil.rmtree(str(Path(conf['train']['exps_folder']) / conf['train']['methodname'] / 'all' / 'train' / 'checkpoints' / 'OptimizerParameters'))



if __name__ == "__main__":
    wandb.agent(f'{PROJECT}/{SWEEP_ID}', train_agent, count=N_JOB) 


