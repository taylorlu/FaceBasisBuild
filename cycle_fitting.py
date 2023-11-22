from tqdm import tqdm
import numpy as np
import torch
import copy
import os, trimesh
from torch.utils.data import Dataset, DataLoader
from pytorch3d.loss import point_mesh_edge_distance, point_mesh_face_distance
from pytorch3d.structures import Meshes, Pointclouds, packed_to_list
from metahuman.MetaHuman import MetaHuman
from flame.FLAME import FLAME, to_tensor
import pickle
cuda = torch.device('cuda:0')


class MyModel(torch.nn.Module):
    def __init__(self, obj_count):
        super(MyModel, self).__init__()

        flame_model_path = os.path.join('models', 'generic_model-2020.pkl') 
        flame_lmk_embedding_path = os.path.join('models', 'flame_static_embedding.pkl') 
        self.flame = FLAME(flame_model_path, flame_lmk_embedding_path)
        self.shape_embedding = torch.nn.Parameter(torch.zeros((1, 300)))
        self.exp_embeddings = torch.nn.Parameter(torch.zeros((obj_count, 100)))
        self.jaw_embeddings = torch.nn.Parameter(torch.zeros((obj_count, 3)))
        self.other_pose_embedding = torch.nn.Parameter(torch.zeros((1, 12)))
        self.scale = torch.nn.Parameter(torch.Tensor([900]))
        self.trans = torch.nn.Parameter(torch.zeros((obj_count, 1, 3)))

        metahuman_model_path = os.path.join('models', 'metahuman_model.pkl') 
        self.metahuman = MetaHuman(metahuman_model_path)
        self.exp_coeffs = torch.nn.Parameter(torch.zeros((obj_count, 35)))
        self.mh_scale = torch.nn.Parameter(torch.Tensor([900]))
        self.mh_trans = torch.nn.Parameter(torch.zeros((obj_count, 1, 3)))

    def forward(self, idx):
        other_pose_param = torch.tile(self.other_pose_embedding, [idx.shape[0], 1])
        jaw_param = torch.index_select(self.jaw_embeddings, dim=0, index=idx)
        pose_params = torch.cat([other_pose_param[:, :6], jaw_param, other_pose_param[:, 6:]], dim=1)
        fl_verts, fl_landmarks3d = self.flame(idx.shape[0], \
                                        shape_params=self.shape_embedding, \
                                        expression_params=torch.index_select(self.exp_embeddings, dim=0, index=idx), \
                                        pose_params=pose_params)
        fl_verts += self.trans
        fl_landmarks3d += self.trans
        fl_verts *= torch.abs(self.scale)
        fl_landmarks3d *= torch.abs(self.scale)

        mh_verts, mh_landmarks3d = self.metahuman(self.exp_coeffs)
        mh_verts += self.mh_trans
        mh_landmarks3d += self.mh_trans
        mh_verts *= torch.abs(self.mh_scale)
        mh_landmarks3d *= torch.abs(self.mh_scale)

        return fl_verts, fl_landmarks3d, mh_verts, mh_landmarks3d

    def infer(self, idx):
        other_pose_param = torch.tile(self.other_pose_embedding, [idx.shape[0], 1])
        jaw_param = torch.index_select(self.jaw_embeddings, dim=0, index=idx)
        pose_params = torch.cat([other_pose_param[:, :6], jaw_param, other_pose_param[:, 6:]], dim=1)
        verts, landmarks3d = self.flame(idx.shape[0], \
                                        shape_params=self.shape_embedding, \
                                        expression_params=torch.index_select(self.exp_embeddings, dim=0, index=idx), \
                                        pose_params=pose_params)
        verts += self.trans
        landmarks3d += self.trans
        verts *= torch.abs(self.scale)
        landmarks3d *= torch.abs(self.scale)

        mh_verts, mh_landmarks3d = self.metahuman(self.exp_coeffs)
        mh_verts += self.mh_trans
        mh_landmarks3d += self.mh_trans
        mh_verts *= torch.abs(self.mh_scale)
        mh_landmarks3d *= torch.abs(self.mh_scale)

        return verts, landmarks3d, mh_verts, mh_landmarks3d

    def infer2(self, idx):
        other_pose_param = torch.tile(self.other_pose_embedding, [idx.shape[0], 1])
        jaw_param = torch.index_select(self.jaw_embeddings, dim=0, index=idx)
        basic_jaw_param = torch.index_select(self.jaw_embeddings, dim=0, index=idx[-1:])
        pose_params = torch.cat([torch.zeros_like(other_pose_param[:, :6]), jaw_param-basic_jaw_param, torch.zeros_like(other_pose_param[:, 6:])], dim=1)
        basic_expression_params = torch.index_select(self.exp_embeddings, dim=0, index=idx[-1:])
        verts, landmarks3d = self.flame(idx.shape[0], \
                                        shape_params=torch.zeros_like(self.shape_embedding), \
                                        expression_params=torch.index_select(self.exp_embeddings, dim=0, index=idx)-basic_expression_params, \
                                        pose_params=pose_params)
        verts += self.trans
        landmarks3d += self.trans
        verts *= torch.abs(self.scale)
        landmarks3d *= torch.abs(self.scale)

        return verts, landmarks3d


class MetahumanDataset(Dataset):
    def __init__(self, mesh_root, label_file):
        lines = open(label_file).readlines()
        self.mesh_list = []
        self.labels = []
        for i, line in enumerate(tqdm(lines)):
            name = os.path.join(mesh_root, '{:04d}.obj'.format(i))
            myMesh = trimesh.load(name, process=False, maintain_order=True)
            self.mesh_list.append(myMesh.vertices)
            self.labels.append([float(x) for x in line.strip().split()])
        self.faces = myMesh.faces

    def __len__(self):
        return len(self.mesh_list)

    def __getitem__(self, idx):
        label = self.labels[idx]
        return to_tensor(self.mesh_list[idx])*900, to_tensor(label), to_tensor(idx, dtype=torch.long)


class Trainer():
    def __init__(self) -> None:
        self.lmks_idx = np.load('models/mh_lmks_idx.npy')

        def keep_face_area(vertices, faces, face_area):
            face_area = copy.deepcopy(face_area)
            face_area.sort()
            face_area = set(face_area)
            vertices_map = [0] * vertices.shape[0]
            acc = 0
            for i in range(len(vertices)):
                if(i in face_area):
                    vertices_map[i] = i - acc
                else:
                    acc += 1
            keep_face_indices = []
            for face in faces:
                if(face[0] in face_area and face[1] in face_area and face[2] in face_area):
                    keep_face_indices.append([vertices_map[face[0]], vertices_map[face[1]], vertices_map[face[2]]])
            return keep_face_indices

        self.mh_keep_vertices = np.load('models/keep_vertex_indices_metahuman.npy')
        mh_mesh = trimesh.load('models/metahuman_mesh.obj', process=False, maintain_order=True)
        self.mh_keep_faces = keep_face_area(mh_mesh.vertices, mh_mesh.faces, self.mh_keep_vertices)

        flame_mask = pickle.load(open('models/FLAME_masks.pkl', 'rb'), encoding='latin1')
        fl_mesh = trimesh.load('models/voca_mesh.obj', process=False, maintain_order=True)
        self.fl_keep_vertices = flame_mask['face']
        self.fl_keep_faces = keep_face_area(fl_mesh.vertices, fl_mesh.faces, self.fl_keep_vertices)

        mesh_root = r'/root/sample350'
        label_file = r'/root/lines350.txt'
        training_data = MetahumanDataset(mesh_root, label_file)
        self.batch_size = len(training_data)
        self.train_dataloader = DataLoader(training_data, batch_size=self.batch_size, shuffle=False, drop_last=True)

        self.model = MyModel(len(training_data)).to(cuda)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-3)

        self.start_epoch = 0
        # pths = os.listdir('checkpoints')
        # pths = [pth for pth in pths if pth.endswith('.pth')]
        # if(len(pths)!=0):
        #     pths = [int(x.split('.')[0]) for x in pths]
        #     pths.sort()
        #     lastest = f'{pths[-1]}.pth'
        #     checkpoint = torch.load(os.path.join('checkpoints', lastest))
        #     self.model.load_state_dict(checkpoint['model_state_dict'])
        #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #     self.start_epoch = checkpoint['epoch']

    def train(self):
        for epoch in range(self.start_epoch, 2000):
            self.model.train()
            for points, morphTarget, idxs in self.train_dataloader:
                points = points.to(cuda)
                morphTarget = morphTarget.to(cuda)
                idxs = idxs.to(cuda)
                mh_lmk3ds = points[:, self.lmks_idx, :]
                mh_pcls = Pointclouds(points[:, self.mh_keep_vertices])

                fl_verts, fl_landmarks3d, mt_verts, mt_landmarks3d = self.model(idxs)

                ################################
                fl_meshes_op = Meshes(verts=fl_verts[:, self.fl_keep_vertices], faces=torch.tile(torch.tensor(self.fl_keep_faces).unsqueeze(0).to(cuda), [self.batch_size, 1, 1]))
                fl_point_dist, fl_face_dist = point_mesh_face_distance(fl_meshes_op, mh_pcls)
                fl_point_dist = torch.reshape(fl_point_dist, [self.batch_size, -1])
                fl_face_dist = torch.reshape(fl_face_dist, [self.batch_size, -1])
                fl_loss_op = fl_point_dist.sum() + fl_face_dist.sum()

                criterion = torch.nn.MSELoss()
                fl_lmk_loss = criterion(mh_lmk3ds, fl_landmarks3d)
                fl_shape_loss = criterion(self.model.shape_embedding, torch.zeros_like(self.model.shape_embedding).to(cuda))
                fl_exp_loss = criterion(self.model.exp_embeddings, torch.zeros_like(self.model.exp_embeddings).to(cuda))
                fl_pose_loss = criterion(self.model.jaw_embeddings, torch.zeros_like(self.model.jaw_embeddings).to(cuda))
                fl_pose_loss += criterion(self.model.other_pose_embedding, torch.zeros_like(self.model.other_pose_embedding).to(cuda))

                ################################
                mh_meshes_op = Meshes(verts=mt_verts[:, self.mh_keep_vertices], faces=torch.tile(torch.tensor(self.mh_keep_faces).unsqueeze(0).to(cuda), [self.batch_size, 1, 1]))
                fl_pcls = Pointclouds(fl_verts[:, self.fl_keep_vertices])
                mh_point_dist, mh_face_dist = point_mesh_face_distance(mh_meshes_op, fl_pcls)
                mh_point_dist = torch.reshape(mh_point_dist, [self.batch_size, -1])
                mh_face_dist = torch.reshape(mh_face_dist, [self.batch_size, -1])
                mh_loss_op = mh_point_dist.sum() + mh_face_dist.sum()

                criterion = torch.nn.MSELoss()
                mh_lmk_loss = criterion(mh_lmk3ds, mt_landmarks3d)
                mh_exp_loss = criterion(self.model.exp_coeffs, torch.zeros_like(self.model.exp_coeffs).to(cuda))

                ################################
                loss = fl_loss_op * 1.0 + fl_lmk_loss * 20.0 + fl_shape_loss * 1.0 + fl_exp_loss * 4.0 + fl_pose_loss * 4.0 + mh_loss_op * 1.0 + mh_lmk_loss * 20.0 + mh_exp_loss * 4.0
                print(f'===========>>> {epoch}: {torch.abs(self.model.scale)}<<<===========')
                print('loss_op: {:.4f}, lmk_loss: {:.4f}, shape_loss: {:.4f}, exp_loss: {:.4f}, pose_loss: {:.4f}, mh_loss_op: {:.4f}, mh_lmk_loss: {:.4f}, mh_exp_loss: {:.4f}'.format(fl_loss_op, fl_lmk_loss, fl_shape_loss, fl_exp_loss, fl_pose_loss, mh_loss_op, mh_lmk_loss, mh_exp_loss))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.test()

    def test(self):
        self.model.eval()
        folder = 'evals'
        idxs = list(range(0, 36))
        fl_verts, _, mt_verts, _ = self.model.infer(torch.tensor(idxs).to(cuda))
        for i, idx in enumerate(idxs):
            hah = trimesh.Trimesh(fl_verts.detach().cpu().numpy()[i, ...], self.model.flame.faces_tensor.cpu().numpy(), process=False)
            hah.export('{}/{:04d}.obj'.format(folder, idx))
            hah = trimesh.Trimesh(mt_verts.detach().cpu().numpy()[i, ...], self.model.metahuman.faces_tensor.cpu().numpy(), process=False)
            hah.export('{}/mh_{:04d}.obj'.format(folder, idx))


if(__name__=='__main__'):
    trainer = Trainer()
    trainer.train()
