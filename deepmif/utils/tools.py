import multiprocessing
import os
import shutil
import sys
from datetime import datetime
from typing import List

import numpy as np
import open3d as o3d
import torch
from torch import optim
from torch.autograd import grad
from torch.optim.optimizer import Optimizer


# setup this run
def setup_experiment(conf, conf_path=None):
    os.environ["NUMEXPR_MAX_THREADS"] = str(multiprocessing.cpu_count())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(conf.setting.gpu_id)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # begining timestamp
    run_name = conf.setting.name + "_" + ts

    run_path = conf.setting.output_root
    if conf.eval.save.use:
        run_path = os.path.join(conf.setting.output_root, run_name)
        access = 0o755
        os.makedirs(run_path, access, exist_ok=True)
        assert os.access(run_path, os.W_OK)
        print(f"Start {run_path}")

        for p in [
            os.path.join(run_path, "mesh"),
            os.path.join(run_path, "map"),
            os.path.join(run_path, "model"),
        ]:
            os.makedirs(p, access, exist_ok=True)

        if conf_path:
            shutil.copy2(sys.argv[1], run_path)

    # Ensure deterministic behavior
    seed_hash = int(conf.setting.seed)

    torch.backends.cudnn.deterministic = True
    o3d.utility.random.seed(seed_hash)
    np.random.seed(seed_hash)
    torch.manual_seed(seed_hash)
    torch.cuda.manual_seed_all(seed_hash)

    return run_path


def setup_optimizer(
    opt_conf,
    octree_tree_level_feat,
    octree_feat,
    mif_decoder_param,
) -> Optimizer:
    lr_cur = opt_conf.learning_rate
    opt_setting = []
    if mif_decoder_param is not None:
        opt_setting.append(
            {
                "params": mif_decoder_param,
                "lr": lr_cur,
                "weight_decay": float(opt_conf.weight_decay),
            }
        )

    for i in range(octree_tree_level_feat):
        opt_setting.append(
            {
                "params": octree_feat[octree_tree_level_feat - i - 1],
                "lr": lr_cur,
            }
        )
        lr_cur *= opt_conf.lr_level_reduce_ratio

    opt = optim.AdamW(opt_setting, betas=(0.9, 0.99), eps=1e-15)

    return opt


def step_lr_decay(
    optimizer: Optimizer,
    learning_rate: float,
    iteration_number: int,
    steps: List,
    reduce: float = 1.0,
):
    if reduce > 1.0 or reduce <= 0.0:
        sys.exit("The decay reta should be between 0 and 1.")

    if iteration_number in steps:
        steps.remove(iteration_number)
        learning_rate *= reduce
        print("Reduce base learning rate to {}".format(learning_rate))

        for param in optimizer.param_groups:
            param["lr"] *= reduce

    return learning_rate


def get_gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return points_grad


def save_checkpoint(
    feature_octree,
    mif_decoder,
    optimizer,
    run_path,
    checkpoint_name,
    iters,
):
    torch.save(
        {
            "iters": iters,
            "feature_octree": feature_octree,  # save the whole NN module (the hierachical features and the indexing structure)
            "mif_decoder": mif_decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        os.path.join(run_path, f"{checkpoint_name}.pth"),
    )
    print(f"save the model to {run_path}/{checkpoint_name}.pth")
