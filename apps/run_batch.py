import sys

import torch
import yaml
from easydict import EasyDict as edict
from tqdm import tqdm

from deepmif.dataset import KittiOdometryDataset, KittiOdometryDatasetConfig
from deepmif.dataset.filters import (
    ApplyPose,
    FilterList,
    RangeFilter,
    StatisticalFilter,
    VoxelFilter,
)
from deepmif.dataset.lidar_dataset import LiDARDataset
from deepmif.model.feature_octree import FeatureOctree
from deepmif.model.mif_decoder import MIFDecoder
from deepmif.utils.loss import *
from deepmif.utils.mesher import Mesher
from deepmif.utils.tools import *


def create_kitti_dataset(ds_path: str, seq_num: int, conf):
    return KittiOdometryDataset(
        KittiOdometryDatasetConfig(
            correct_scan_calibration=True,
            dataset_path=ds_path,
            filters=FilterList(
                [
                    RangeFilter(conf.min_range_m, conf.pc_radius_m, is_centered=True),
                    VoxelFilter(conf.vox_down_m, is_centered=True),
                    StatisticalFilter(conf.sor_nn, conf.sor_std),
                    ApplyPose(),
                ]
            ),
        ),
        seq_num,
    )


def prepare_conf(conf):
    conf.world_size = conf.octree.leaf_vox_size * (
        2 ** (conf.octree.tree_level_world - 1)
    )
    conf.scale = 1.0 / conf.world_size
    conf.dtype = torch.float32
    conf.mc_query_level = conf.octree.tree_level_world - conf.octree.tree_level_feat + 1


def run_mapping_batch():
    if len(sys.argv) <= 1:
        sys.exit(
            "Please provide the path to the config file.\n"
            "Try: python run_batch.py xxx/xxx_config.yaml"
        )

    with open(os.path.abspath(sys.argv[1])) as f:
        conf = edict(yaml.safe_load(f))
        prepare_conf(conf)

    run_path = setup_experiment(conf, conf_path=sys.argv[1])

    raw_dataset = create_kitti_dataset(
        conf.setting.dataset_path,
        conf.setting.sequence_num,
        conf.dataset_process,
    )

    MAIN_DEVICE = conf.setting.device
    MAIN_DTYPE = conf.dtype

    # initialize the feature octree and mlp decoder
    octree = FeatureOctree(conf.octree, conf.scale, device=MAIN_DEVICE)
    mif_decoder = MIFDecoder(conf.network, conf.octree.feature_dim)
    mif_decoder.to(MAIN_DEVICE)

    # load the decoder model
    if conf.setting.load.use:
        loaded_model = torch.load(conf.setting.load.model_path)
        mif_decoder.load_state_dict(loaded_model["mif_decoder"])
        octree = loaded_model["feature_octree"]

    dataset = LiDARDataset(
        conf.lidar_sampler,
        conf.scale,
        conf.optimizer.batch_size,
        octree=octree,
        dtype=MAIN_DTYPE,
        device=MAIN_DEVICE,
        # pool_device=MAIN_DEVICE,
    )

    begin_frame = conf.setting.begin_frame
    end_frame = min(len(raw_dataset), conf.setting.end_frame)
    for frame_id in tqdm(range(begin_frame, end_frame, conf.setting.every_frame)):
        dataset.process_frame(raw_dataset[frame_id])

    mesher = Mesher(
        conf,
        octree,
        mif_decoder,
        conf.scale,
        device=MAIN_DEVICE,
        dtype=MAIN_DTYPE,
    )

    # loss
    dossr_loss = MIFLoss(conf.loss)
    dossr_loss.to(MAIN_DEVICE)

    if conf.eval.save.use:
        pc_map_path = run_path + "/map/pc_map_down.ply"
        dataset.write_merged_pc(pc_map_path)

    opt = setup_optimizer(
        conf.optimizer,
        conf.octree.tree_level_feat,
        list(octree.parameters()),
        list(mif_decoder.parameters()),
    )

    octree.print_detail()

    cur_base_lr = conf.optimizer.learning_rate
    lr_decay_step = conf.optimizer.lr_decay_step
    lr_iters_reduce_ratio = conf.optimizer.lr_iters_reduce_ratio

    p_bar = tqdm(range(conf.optimizer.iters), position=0)

    for iter in p_bar:
        step_lr_decay(opt, cur_base_lr, iter, lr_decay_step, lr_iters_reduce_ratio)

        decoder_input = dataset.get_batch()

        outputs = mif_decoder(decoder_input, octree)

        all_losses = dossr_loss(outputs)

        opt.zero_grad(set_to_none=True)

        all_losses["loss"].backward()
        opt.step()

        # save checkpoint model
        if conf.eval.save.use:
            if ((iter + 1) % conf.eval.save.model_iters) == 0:
                cp_name = "model/model_iter_" + str(iter + 1)
                save_checkpoint(octree, mif_decoder, opt, run_path, cp_name, iter)

            if ((iter + 1) % conf.eval.save.mesh_iters) == 0:
                mesh_path = "%s/mesh/mesh_iter_%05d.ply" % (run_path, (iter + 1))
                ql, res = conf.mc_query_level, conf.eval.marching_cubes.resolution_m
                mesher.recon_octree_mesh(ql, res, mesh_path)


if __name__ == "__main__":
    run_mapping_batch()
