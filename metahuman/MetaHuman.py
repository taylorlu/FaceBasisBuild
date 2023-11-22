import pickle
import numpy as np
import torch
import torch.nn as nn
from metahuman.lbs_metahuman import lbs, batch_rodrigues
from flame.lbs_flame import to_tensor, to_np, Struct


class MetaHuman(nn.Module):
    """
    Given metahuman parameters this class generates a differentiable metahuman function
    which outputs the a mesh and 3D facial landmarks
    """
    def __init__(self, metahuman_model_path):
        super(MetaHuman, self).__init__()
        print("creating the metahuman Decoder")
        with open(metahuman_model_path, 'rb') as f:
            metahuman_model = Struct(**pickle.load(f, encoding='latin1'))

        self.dtype = torch.float32
        self.register_buffer('faces_tensor', to_tensor(to_np(metahuman_model.f, dtype=np.int64), dtype=torch.long))
        # The vertices of the template model
        self.register_buffer('v_template', to_tensor(to_np(metahuman_model.v_template), dtype=self.dtype))
        # The expression components and expression
        self.register_buffer('expdirs', to_tensor(to_np(metahuman_model.expression_basis), dtype=self.dtype))
        # The jaw components
        self.register_buffer('jaw_pose_basis', to_tensor(to_np(metahuman_model.jaw_pose_basis), dtype=self.dtype))
        # The pose components
        self.register_buffer('posedirs', to_tensor(to_np(metahuman_model.pose_basis), dtype=self.dtype))
        self.register_buffer('J_coordinate', to_tensor(to_np(metahuman_model.J_coordinate), dtype=self.dtype))
        self.register_buffer('parents', to_tensor(to_np(metahuman_model.parents)).long())
        self.register_buffer('lbs_weights', to_tensor(to_np(metahuman_model.lbs_weights), dtype=self.dtype))
        self.register_buffer('lmks_idx', to_tensor(to_np(metahuman_model.lmks_idx), dtype=torch.long))
        self.register_buffer('global_matrix', to_tensor(torch.eye(3)))
        self.register_buffer('J_root', to_tensor(torch.tensor([[0, 0, 0]])))


    def forward(self, params):
        """
            Input:
                params: N X number of parameters
            return:
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        batch_size = params.shape[0]

        jaw_pose9 = batch_rodrigues(torch.multiply(self.jaw_pose_basis[0].unsqueeze(0).repeat(batch_size, 1), params[:, 9:10]))
        jaw_pose8 = batch_rodrigues(torch.multiply(self.jaw_pose_basis[1].unsqueeze(0).repeat(batch_size, 1), params[:, 8:9]))
        jaw_pose7 = batch_rodrigues(torch.multiply(self.jaw_pose_basis[2].unsqueeze(0).repeat(batch_size, 1), params[:, 7:8]))
        jaw_matrix = torch.matmul(torch.matmul(jaw_pose9, jaw_pose8), jaw_pose7).view(batch_size, 1, 9)
        
        full_pose = torch.concat([self.global_matrix.unsqueeze(0).repeat([batch_size, 1, 1]).view(batch_size, 1, 9), jaw_matrix], 1)

        template_vertices = self.v_template.unsqueeze(0).repeat(batch_size, 1, 1)

        exp_coeffs = torch.concat([params[:, :7], params[:, 10:]], dim=1)

        vertices, _ = lbs(exp_coeffs, full_pose, template_vertices,
                            self.expdirs, self.posedirs,
                            torch.concat([self.J_root, self.J_coordinate], dim=0).repeat(batch_size, 1, 1), self.parents,
                            self.lbs_weights, pose2rot=False)

        landmarks = vertices[:, self.lmks_idx, :]

        return vertices, landmarks

# params = torch.zeros([1, 35])
# model = MetaHuman('metahuman_model.pkl')
# vertices, landmarks = model(params)
