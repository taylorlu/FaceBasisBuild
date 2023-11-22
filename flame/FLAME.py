"""
FLAME Layer: Implementation of the 3D Statistical Face model in PyTorch
It is designed in a way to directly plug in as a decoder layer in a 
Deep learning framework for training and testing
It can also be used for 2D or 3D optimisation applications
Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.
You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and liable to prosecution.
Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.
More information about FLAME is available at http://flame.is.tue.mpg.de.
For questions regarding the PyTorch implementation please contact soubhik.sanyal@tuebingen.mpg.de
"""
# Modified from smplx code [https://github.com/vchoutas/smplx] for FLAME

import numpy as np
import torch
import torch.nn as nn
import pickle
from flame.lbs_flame import lbs, to_tensor, to_np, Struct, vertices2landmarks


class FLAME(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 3D facial landmarks
    """
    def __init__(self, flame_model_path, flame_lmk_embedding_path, n_shape=300, n_exp=100):
        super(FLAME, self).__init__()
        print("creating the FLAME Decoder")
        with open(flame_model_path, 'rb') as f:
            flame_model = Struct(**pickle.load(f, encoding='latin1'))

        self.dtype = torch.float32
        self.faces = to_np(flame_model.f, dtype=np.int64)
        self.register_buffer('faces_tensor', to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long))
        # The vertices of the template model
        self.register_buffer('v_template', to_tensor(to_np(flame_model.v_template), dtype=self.dtype))
        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat([shapedirs[:,:,:n_shape], shapedirs[:,:,300:300+n_exp]], 2)
        self.register_buffer('shapedirs', shapedirs)
        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs), dtype=self.dtype))
        # 
        self.register_buffer('J_regressor', to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype))
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long(); parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', to_tensor(to_np(flame_model.weights), dtype=self.dtype))

        # Static and Dynamic Landmark embeddings for FLAME

        with open(flame_lmk_embedding_path, 'rb') as f:
            static_embeddings = Struct(**pickle.load(f, encoding='latin1'))

        lmk_faces_idx = (static_embeddings.lmk_face_idx).astype(np.int64)
        self.register_buffer('lmk_faces_idx',
                             torch.tensor(lmk_faces_idx, dtype=torch.long))
        lmk_bary_coords = static_embeddings.lmk_b_coords
        self.register_buffer('lmk_bary_coords',
                             torch.tensor(lmk_bary_coords, dtype=self.dtype))

    def forward(self, batch_size, shape_params=None, expression_params=None, pose_params=None):
        """
            Input:
                shape_params: N X number of shape parameters
                expression_params: N X number of expression parameters
                pose_params: N X number of pose parameters
            return:
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        betas = torch.cat([torch.tile(shape_params, [expression_params.shape[0], 1]), expression_params], dim=1)
        template_vertices = self.v_template.unsqueeze(0).repeat(batch_size, 1, 1)

        vertices, _ = lbs(betas, pose_params, template_vertices,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, dtype=self.dtype)

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).repeat(
            batch_size, 1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(
            batch_size, 1, 1)

        landmarks = vertices2landmarks(vertices, self.faces_tensor,
                                             lmk_faces_idx,
                                             lmk_bary_coords)

        return vertices, landmarks