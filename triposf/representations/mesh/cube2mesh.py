# MIT License

# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 VAST-AI-Research and contributors.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

import torch
from ...modules.sparse import SparseTensor
from easydict import EasyDict as edict
from .utils_cube import *
from .flexicubes.flexicubes import FlexiCubes

class MeshExtractResult:
    def __init__(self,
        vertices,
        faces,
        vertex_attrs=None,
        res=64
    ):
        self.vertices = vertices
        self.faces = faces.long()
        self.vertex_attrs = vertex_attrs
        self.face_normal = self.compute_face_normals(vertices, faces)
        self.res = res
        self.success = (vertices.shape[0] != 0 and faces.shape[0] != 0)

        # training only
        self.tsdf_v = None
        self.tsdf_s = None
        
    def compute_face_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)
        
        return face_normals[:, None, :].repeat(1, 3, 1)
                
    def comput_v_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        v_normals = torch.zeros_like(verts)
        v_normals.scatter_add_(0, i0[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i1[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i2[..., None].repeat(1, 3), face_normals)

        v_normals = torch.nn.functional.normalize(v_normals, dim=1)
        return v_normals   


class SparseFeatures2Mesh:
    def __init__(self, device="cuda", res=64, use_color=False, use_sparse_flexicube=True, use_sparse_sparse_flexicube=False):
        '''
        a model to generate a mesh from sparse features structures using flexicube
        '''
        super().__init__()
        self.device=device
        self.res = res
        self.mesh_extractor = FlexiCubes(device=device)
        self.sdf_bias = -1.0 / res
        
        self.use_sparse_flexicube = use_sparse_flexicube
        self.use_sparse_sparse_flexicube = use_sparse_sparse_flexicube
        
        self.use_color = use_color
        self._calc_layout()
    
    def _calc_layout(self):
        LAYOUTS = {
            'sdf': {'shape': (8, 1), 'size': 8},
            'deform': {'shape': (8, 3), 'size': 8 * 3},
            'weights': {'shape': (21,), 'size': 21}
        }
        if self.use_color:
            '''
            6 channel color including normal map
            '''
            LAYOUTS['color'] = {'shape': (8, 6,), 'size': 8 * 6}
        self.layouts = edict(LAYOUTS)
        start = 0
        for k, v in self.layouts.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
        self.feats_channels = start
        
    def get_layout(self, feats : torch.Tensor, name : str):
        if name not in self.layouts:
            return None
        return feats[:, self.layouts[name]['range'][0]:self.layouts[name]['range'][1]].reshape(-1, *self.layouts[name]['shape'])
    
    def __call__(self, cubefeats : SparseTensor, training=False):
        """
        Generates a mesh based on the specified sparse voxel structures.
        Args:
            cube_attrs [Nx21] : Sparse Tensor attrs about cube weights
            verts_attrs [Nx10] : [0:1] SDF [1:4] deform [4:7] color [7:10] normal 
        Returns:
            return the success tag and ni you loss, 
        """
        assert self.use_color == False

        coords = cubefeats.coords[:, 1:]
        feats = cubefeats.feats
        
        sdf, deform, color, weights = [self.get_layout(feats, name) for name in ['sdf', 'deform', 'color', 'weights']]
        sdf = sdf * (4. / self.res)
        sdf += self.sdf_bias

        v_attrs = [sdf, deform, color] if self.use_color else [sdf, deform]
        v_pos, v_attrs, reg_loss = sparse_cube2verts(coords, torch.cat(v_attrs, dim=-1), training=training)
        
        res_v = self.res + 1
        v_attrs_d_sparse, v_pos_dilate = get_sparse_attrs(v_pos, v_attrs, res=res_v, sdf_init=True)
        weights_d_sparse, coords_dilate = get_sparse_attrs(coords, weights, res=self.res, sdf_init=False)
        
        sdf_d, deform_d = v_attrs_d_sparse[..., 0], v_attrs_d_sparse[..., 1:4]

        x_nx3 = get_defomed_verts(v_pos_dilate, deform_d, self.res)
        x_nx3 = torch.cat((x_nx3, torch.ones((1, 3), dtype=x_nx3.dtype, device=x_nx3.device) * 0.5))
        sdf_d = torch.cat((sdf_d, torch.ones((1), dtype=sdf_d.dtype, device=sdf_d.device)))
        
        mask_reg_c_sparse = (v_pos_dilate[..., 0] * res_v + v_pos_dilate[..., 1]) * res_v + v_pos_dilate[..., 2]
        reg_c_sparse = (coords_dilate[..., 0] * res_v + coords_dilate[..., 1]) * res_v + coords_dilate[..., 2]
        cube_corners_bias = (cube_corners[:, 0] * res_v + cube_corners[:, 1]) * res_v + cube_corners[:, 2]            
        reg_c_value = (reg_c_sparse.unsqueeze(1) + cube_corners_bias.unsqueeze(0).cuda()).reshape(-1)
        reg_c = torch.searchsorted(mask_reg_c_sparse, reg_c_value)
        exact_match_mask = mask_reg_c_sparse[reg_c] == reg_c_value
        reg_c[exact_match_mask == 0] = len(mask_reg_c_sparse)
        reg_c = reg_c.reshape(-1, 8)

        vertices, faces, L_dev, colors = self.mesh_extractor(
            voxelgrid_vertices=x_nx3,
            scalar_field=sdf_d,
            cube_idx=reg_c,
            resolution=self.res,
            beta=weights_d_sparse[:, :12],
            alpha=weights_d_sparse[:, 12:20],
            gamma_f=weights_d_sparse[:, 20],
            cube_index_map=coords_dilate,
            training=training)
                
        mesh = MeshExtractResult(vertices=vertices, faces=faces, vertex_attrs=colors, res=self.res)

        return mesh