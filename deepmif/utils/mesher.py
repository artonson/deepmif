import math

import numpy as np
import open3d as o3d
import skimage.measure
import torch
from tqdm import tqdm

from deepmif.model.feature_octree import FeatureOctree
from deepmif.model.mif_decoder import MIFDecoder


class Mesher:
    def __init__(
        self,
        config,
        octree: FeatureOctree,
        mif_decoder: MIFDecoder,
        scale,
        device="cuda",
        dtype=torch.float32,
    ):
        self.infer_bs = config.optimizer.batch_size * 16
        self.mc_mask_on = True
        self.mc_vis_level = 1

        self.octree = octree
        self.mif_decoder = mif_decoder

        self.device = device
        self.cur_device = self.device
        self.dtype = dtype
        self.world_scale = scale

        self.global_transform = np.eye(4)

    def query_points(
        self,
        coord,
        bs,
        query_mask=True,
    ):
        """query the sdf value, semantic label and marching cubes mask for points
        Args:
            coord: Nx3 torch tensor, the coordinates of all N (axbxc) query points in the scaled
                kaolin coordinate system [-1,1]
            bs: batch size for the inference
        Returns:
            sdf_pred: Ndim numpy array, signed distance value (scaled) at each query point
            sem_pred: Ndim numpy array, semantic label prediction at each query point
            mc_mask:  Ndim bool numpy array, marching cubes mask at each query point
        """
        # the coord torch tensor is already scaled in the [-1,1] coordinate system
        sample_count = coord.shape[0]
        iter_n = math.ceil(sample_count / bs)
        check_level = min(self.octree.featured_level_num, self.mc_vis_level) - 1
        sdf_pred = np.zeros(sample_count)
        mc_mask = np.zeros(sample_count) if query_mask else None

        with torch.no_grad():  # eval step
            if iter_n <= 1:
                c = coord / self.world_scale

                sdf = self.mif_decoder.implicit(c, self.octree)
                sdf_pred = sdf.flatten().detach().cpu().numpy()

                if query_mask:
                    # get the marching cubes mask
                    check_level_indices = self.octree.hierarchical_indices[check_level]
                    # if index is -1 for the level, then means the point is not valid under this level
                    mask_mc = check_level_indices >= 0
                    # all should be true (all the corner should be valid)
                    mc_mask = torch.all(mask_mc, dim=1).detach().cpu().numpy()
                return sdf_pred, mc_mask

            for n in tqdm(range(iter_n), position=2, leave=False):
                head = n * bs
                tail = min((n + 1) * bs, sample_count)
                batch_coord = coord[head:tail, :]

                if self.cur_device == "cpu" and self.device == "cuda":
                    batch_coord = batch_coord.detach().cuda()

                batch_sdf = self.mif_decoder.implicit(
                    batch_coord / self.world_scale, self.octree
                ).flatten()

                sdf_pred[head:tail] = batch_sdf.detach().cpu().numpy()

                if query_mask:
                    # get the marching cubes mask
                    # hierarchical_indices: bottom-up
                    check_level_indices = self.octree.hierarchical_indices[check_level]
                    # print(check_level_indices)
                    # if index is -1 for the level, then means the point is not valid under this level
                    mask_mc = check_level_indices >= 0
                    # print(mask_mc.shape)
                    # all should be true (all the corner should be valid)
                    mask_mc = torch.all(mask_mc, dim=1)
                    mc_mask[head:tail] = mask_mc.detach().cpu().numpy()
                    # but for scimage's marching cubes, the top right corner's mask should also be true to conduct marching cubes

            return sdf_pred, mc_mask

    def assign_to_bbx(self, sdf_pred, mc_mask, voxel_num_xyz):
        """assign the queried sdf, semantic label and marching cubes mask back to the 3D grids in the specified bounding box
        Args:
            sdf_pred: Ndim np.array
            sem_pred: Ndim np.array
            mc_mask:  Ndim bool np.array
            voxel_num_xyz: 3dim numpy array, the number of voxels on each axis for the bbx
        Returns:
            sdf_pred:  a*b*c np.array, 3d grids of sign distance values
            sem_pred:  a*b*c np.array, 3d grids of semantic labels
            mc_mask:   a*b*c np.array, 3d grids of marching cube masks, marching cubes only on where
                the mask is true
        """
        if sdf_pred is not None:
            sdf_pred = sdf_pred.reshape(
                voxel_num_xyz[0], voxel_num_xyz[1], voxel_num_xyz[2]
            )

        if mc_mask is not None:
            mc_mask = mc_mask.reshape(
                voxel_num_xyz[0], voxel_num_xyz[1], voxel_num_xyz[2]
            ).astype(dtype=bool)

        return sdf_pred, mc_mask

    def mc_mesh(self, mc_sdf, mc_mask, voxel_size, mc_origin):
        """use the marching cubes algorithm to get mesh vertices and faces
        Args:
            mc_sdf:  a*b*c np.array, 3d grids of sign distance values
            mc_mask: a*b*c np.array, 3d grids of marching cube masks, marching cubes only on where
                the mask is true
            voxel_size: scalar, marching cubes voxel size with unit m
            mc_origin: 3*1 np.array, the coordinate of the bottom-left corner of the 3d grids for
                marching cubes, in world coordinate system with unit m
        Returns:
            ([verts], [faces]), mesh vertices and triangle faces
        """
        print("Marching cubes ...")
        # the input are all already numpy arraies
        verts, faces = np.zeros((0, 3)), np.zeros((0, 3))
        try:
            verts, faces, _, _ = skimage.measure.marching_cubes(
                mc_sdf, level=0.0, allow_degenerate=False, mask=mc_mask
            )
        except:
            pass

        verts = mc_origin + verts * voxel_size
        return verts, faces

    def filter_isolated_vertices(self, mesh, filter_cluster_min_tri=300):
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        triangles_to_remove = (
            cluster_n_triangles[triangle_clusters] < filter_cluster_min_tri
        )
        mesh.remove_triangles_by_mask(triangles_to_remove)
        return mesh

    # reconstruct the map sparsely using the octree, only query the sdf at certain level ($query_level) of the octree
    # much faster and also memory-wise more efficient
    def recon_octree_mesh(
        self,
        query_level,
        mc_res_m,
        mesh_path,
        estimate_normal=True,
        filter_isolated_mesh=True,
    ):
        nodes_coord_scaled = self.octree.get_octree_nodes(
            query_level
        )  # query level top-down
        nodes_count = nodes_coord_scaled.shape[0]
        min_nodes = np.min(nodes_coord_scaled, 0)
        max_nodes = np.max(nodes_coord_scaled, 0)

        node_res_scaled = 2 ** (
            1 - query_level
        )  # voxel size for queried octree node in [-1,1] coordinate system
        # marching cube's voxel size should be evenly divisible by the queried octree node's size
        voxel_count_per_side_node = np.ceil(
            node_res_scaled / self.world_scale / mc_res_m
        ).astype(dtype=int)
        # assign coordinates for the queried octree node
        x = torch.arange(
            voxel_count_per_side_node, dtype=torch.int16, device=self.device
        )
        y = torch.arange(
            voxel_count_per_side_node, dtype=torch.int16, device=self.device
        )
        z = torch.arange(
            voxel_count_per_side_node, dtype=torch.int16, device=self.device
        )
        node_box_size = (np.ones(3) * voxel_count_per_side_node).astype(dtype=int)

        # order: [0,0,0], [0,0,1], [0,0,2], [0,1,0], [0,1,1], [0,1,2] ...
        x, y, z = torch.meshgrid(x, y, z, indexing="ij")
        # get the vector of all the grid point's 3D coordinates
        coord = (
            torch.stack((x.flatten(), y.flatten(), z.flatten())).transpose(0, 1).float()
        )
        mc_res_scaled = (
            node_res_scaled / voxel_count_per_side_node
        )  # voxel size for marching cubes in [-1,1] coordinate system
        # transform to [-1,1] coordinate system
        coord *= mc_res_scaled

        # the voxel count for the whole map
        voxel_count_per_side = (
            (max_nodes - min_nodes) / mc_res_scaled + voxel_count_per_side_node
        ).astype(int)

        # initialize the whole map
        query_grid_sdf = np.ones(
            (voxel_count_per_side[0], voxel_count_per_side[1], voxel_count_per_side[2]),
            dtype=np.float16,
        )  # use float16 to save memory

        query_grid_mask = np.zeros(
            (voxel_count_per_side[0], voxel_count_per_side[1], voxel_count_per_side[2]),
            dtype=bool,
        )  # mask off

        for node_idx in tqdm(range(nodes_count), position=1, leave=False):
            node_coord_scaled = nodes_coord_scaled[node_idx, :]
            cur_origin = torch.tensor(
                node_coord_scaled - 0.5 * (node_res_scaled - mc_res_scaled),
                device=self.device,
            )
            cur_coord = coord.clone()
            cur_coord += cur_origin
            cur_sdf_pred, cur_mc_mask = self.query_points(
                cur_coord, self.infer_bs, self.mc_mask_on
            )
            cur_sdf_pred, cur_mc_mask = self.assign_to_bbx(
                cur_sdf_pred, cur_mc_mask, node_box_size
            )
            shift_coord = (node_coord_scaled - min_nodes) / node_res_scaled
            shift_coord = (shift_coord * voxel_count_per_side_node).astype(int)
            query_grid_sdf[
                shift_coord[0] : shift_coord[0] + voxel_count_per_side_node,
                shift_coord[1] : shift_coord[1] + voxel_count_per_side_node,
                shift_coord[2] : shift_coord[2] + voxel_count_per_side_node,
            ] = cur_sdf_pred
            query_grid_mask[
                shift_coord[0] : shift_coord[0] + voxel_count_per_side_node,
                shift_coord[1] : shift_coord[1] + voxel_count_per_side_node,
                shift_coord[2] : shift_coord[2] + voxel_count_per_side_node,
            ] = cur_mc_mask

        mc_voxel_size = mc_res_scaled / self.world_scale
        mc_voxel_origin = (
            min_nodes - 0.5 * (node_res_scaled - mc_res_scaled)
        ) / self.world_scale

        verts, faces = self.mc_mesh(
            query_grid_sdf, query_grid_mask, mc_voxel_size, mc_voxel_origin
        )
        # directly use open3d to get mesh
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces)
        )

        if estimate_normal:
            mesh.compute_vertex_normals()

        if filter_isolated_mesh:
            mesh = self.filter_isolated_vertices(mesh)

        # global transform (to world coordinate system) before output
        mesh.transform(self.global_transform)

        # write the mesh to ply file
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print("save the mesh to %s\n" % (mesh_path))

        return mesh
