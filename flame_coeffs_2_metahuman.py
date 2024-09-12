import numpy as np
import torch
import copy
import os, trimesh
from scipy.spatial.transform import Rotation as RTT
from torch.utils.data import Dataset, DataLoader
from pytorch3d.loss import point_mesh_edge_distance, point_mesh_face_distance
from pytorch3d.structures import Meshes, Pointclouds, packed_to_list
from metahuman.MetaHuman import MetaHuman
from flame.FLAME import FLAME, to_tensor
import pickle
from render import render_sequence
cuda = torch.device('cuda:0')


class MyModel(torch.nn.Module):
    def __init__(self, obj_count):
        super(MyModel, self).__init__()
        self.obj_count = obj_count
        flame_model_path = os.path.join('models', 'generic_model-2020.pkl')
        flame_lmk_embedding_path = os.path.join('models', 'flame_static_embedding.pkl')
        self.flame = FLAME(flame_model_path, flame_lmk_embedding_path)

        self.shape_embedding1 = torch.nn.Parameter(torch.zeros((1, 300)))
        self.exp_embeddings1 = torch.nn.Parameter(torch.zeros((1, 100)))
        self.jaw_embeddings1 = torch.nn.Parameter(torch.zeros((1, 3)))

        metahuman_model_path = os.path.join('models', 'metahuman_model.pkl')
        self.metahuman = MetaHuman(metahuman_model_path)
        self.exp_coeffs = torch.nn.Parameter(-torch.ones((obj_count, 35))*30)
        self.last = torch.nn.Parameter(torch.zeros((1, 35)))

        # self.exp_coeffs = torch.nn.Parameter(-torch.ones((obj_count, 35))*30)
        # self.exp_coeffs = torch.nn.Parameter(torch.zeros((obj_count, 35)))
        self.mh_scale = torch.nn.Parameter(torch.Tensor([80]))
        self.mh_trans = torch.nn.Parameter(torch.zeros((1, 1, 3)))

    def forward(self, exp_embeddings, pose_params, idx):
        exp_embeddings += self.exp_embeddings1
        pose_params[:, 6:9] += self.jaw_embeddings1

        fl_verts1, fl_landmarks3d1 = self.flame(idx.shape[0], \
                                        shape_params=self.shape_embedding1, \
                                        expression_params=exp_embeddings, \
                                        pose_params=pose_params)
        fl_verts1 *= 90
        fl_landmarks3d1 *= 90
        # mh_verts, mh_landmarks3d = self.metahuman(torch.sigmoid(torch.index_select(self.exp_coeffs, dim=0, index=idx)))
        mh_verts, mh_landmarks3d = self.metahuman(torch.sigmoid(torch.index_select(self.exp_coeffs, dim=0, index=idx)) + self.last)
        # mh_verts2, mh_landmarks3d2 = self.metahuman(self.last)
        # mh_verts = torch.cat([mh_verts, mh_verts2], 0)
        # mh_landmarks3d = torch.cat([mh_landmarks3d, mh_landmarks3d2], 0)
        mh_verts += self.mh_trans
        mh_landmarks3d += self.mh_trans
        mh_verts *= self.mh_scale
        mh_landmarks3d *= self.mh_scale

        return fl_verts1, fl_landmarks3d1, mh_verts, mh_landmarks3d

    def forward_single(self, exp_embedding, pose_param):
        exp_embedding += self.exp_embeddings1
        pose_param[:, 6:9] += self.jaw_embeddings1

        fl_verts1, fl_landmarks3d1 = self.flame(1, \
                                        shape_params=self.shape_embedding1, \
                                        expression_params=exp_embedding, \
                                        pose_params=pose_param)
        fl_verts1 *= 90
        fl_landmarks3d1 *= 90
        # mh_verts, mh_landmarks3d = self.metahuman(torch.sigmoid(torch.index_select(self.exp_coeffs, dim=0, index=idx)))
        mh_verts, mh_landmarks3d = self.metahuman(self.last)
        mh_verts += self.mh_trans
        mh_landmarks3d += self.mh_trans
        mh_verts *= self.mh_scale
        mh_landmarks3d *= self.mh_scale

        return fl_verts1, fl_landmarks3d1, mh_verts, mh_landmarks3d

    def infer(self, idx):
        # a = torch.zeros([idx.shape[0], 35]).to(cuda)
        # a[:, 10] = 0.7
        # mh_verts, mh_landmarks3d = self.metahuman(torch.sigmoid(torch.index_select(self.exp_coeffs, dim=0, index=idx)) - torch.sigmoid(torch.index_select(self.exp_coeffs, dim=0, index=torch.tensor([self.obj_count-1]).to(cuda))))
        mh_verts, mh_landmarks3d = self.metahuman(torch.sigmoid(torch.index_select(self.exp_coeffs, dim=0, index=idx)))
        # mh_verts += self.mh_trans
        # mh_landmarks3d += self.mh_trans
        return mh_verts, mh_landmarks3d


# class FlameDataset(Dataset):
#     def __init__(self):
#         exp_jaw = np.load('models/exp_jaw.npy', 'r')#[:499]
#         # exp_jaw = np.concatenate([exp_jaw, np.zeros_like(exp_jaw)[-1:, :]], axis=0)
#         self.pose_params = torch.zeros([exp_jaw.shape[0], 15], dtype=torch.float32)
#         self.pose_params[:, 6:9] = to_tensor(exp_jaw[:, 50:])
#         self.expression_params = torch.zeros([exp_jaw.shape[0], 100], dtype=torch.float32)
#         self.expression_params[:, :50] = to_tensor(exp_jaw[:, :50])

#     def __len__(self):
#         return self.expression_params.shape[0]

#     def __getitem__(self, idx):
#         return self.expression_params[idx], self.pose_params[idx], idx

class FlameDataset(Dataset):
    def __init__(self):
        exp_fullpose = np.load('models/exp_fullpose.npy', allow_pickle=True).item()
        # self.pose_params = to_tensor(np.concatenate([exp_fullpose['full_pose'], np.zeros_like(exp_fullpose['full_pose'])[-1:, :]], axis=0))
        self.pose_params = to_tensor(exp_fullpose['full_pose'])
        self.pose_params[:, :6] = 0
        self.pose_params[:, 9:] = 0
        self.expression_params = torch.zeros([exp_fullpose['expcode'].shape[0], 100], dtype=torch.float32)
        self.expression_params[:, :50] = to_tensor(exp_fullpose['expcode'][:, :50])

    def __len__(self):
        return self.expression_params.shape[0]

    def __getitem__(self, idx):
        return self.expression_params[idx], self.pose_params[idx], idx

class Trainer():
    def __init__(self) -> None:
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
        self.fl_keep_vertices = flame_mask['face']
        fl_mesh = trimesh.load('models/voca_mesh.obj', process=False, maintain_order=True)
        self.fl_keep_faces = keep_face_area(fl_mesh.vertices, fl_mesh.faces, self.fl_keep_vertices)

        training_data = FlameDataset()
        self.total_size = len(training_data)
        self.train_dataloader = DataLoader(training_data, batch_size=500, shuffle=False, drop_last=False)
        self.model = MyModel(self.total_size).to(cuda)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-3)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.987)

        self.start_epoch = 0

    def train(self):
        exp_jaw_params = []
        last_params = []
        scale_params = []
        other_params = []
        for n, p in self.model.named_parameters():
            print(n)
            if('exp_coeffs' in n):
                exp_jaw_params.append(p)
            elif('last' in n):
                last_params.append(p)
            elif('mh_scale' in n):
                scale_params.append(p)
            else:
                other_params.append(p)

        self.optimizer = torch.optim.Adam([{'params': other_params, "lr": 3e-3}, {'params': scale_params, "lr": 3e-1}, {'params': exp_jaw_params, "lr": 5e-1}, {'params': last_params, "lr":5e-2}])

        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-3)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

        # tou5-1=5-1 >> loss_op: 0.1798, lmk_loss: 0.5315, mh_exp_loss: 0.0093
        # tou5-1=5-2 >> loss_op: 0.1396, lmk_loss: 0.4752, mh_exp_loss: 0.0057
        # tou5-1=5-3 >> loss_op: 0.1379, lmk_loss: 0.6054, mh_exp_loss: 0.0173
        # tou1+0=5-2 >> loss_op: 0.1402, lmk_loss: 0.4773, mh_exp_loss: 0.0027
        # tou1-1=5-2 >> loss_op: 0.1407, lmk_loss: 0.4745, mh_exp_loss: 0.0373
        for epoch in range(self.start_epoch, 200):
            self.model.train()
            if(epoch<20):
                self.model.shape_embedding1.requires_grad = True
                self.model.exp_embeddings1.requires_grad = True
                self.model.jaw_embeddings1.requires_grad = True
                self.model.exp_coeffs.requires_grad = False
                self.model.mh_trans.requires_grad = True
                self.model.mh_scale.requires_grad = True
            else:
                if(epoch==20):
                    self.model.exp_coeffs.copy_(torch.zeros_like(self.model.exp_coeffs))
                    # self.model.exp_coeffs = torch.nn.Parameter(torch.zeros_like(self.model.exp_coeffs))
                self.model.shape_embedding1.requires_grad = False
                self.model.exp_embeddings1.requires_grad = False
                self.model.jaw_embeddings1.requires_grad = False
                self.model.exp_coeffs.requires_grad = True
                self.model.mh_trans.requires_grad = False
                self.model.mh_scale.requires_grad = False

            for expression_params, pose_params, idxs in self.train_dataloader:
                expression_params = expression_params.to(cuda)
                pose_params = pose_params.to(cuda)
                idxs = idxs.to(cuda)

                fl_verts1, fl_landmarks3d1, mh_verts, mh_landmarks3d = self.model(expression_params, pose_params, idxs)

                expression_params = torch.zeros_like(expression_params[:1])
                pose_params = torch.zeros_like(pose_params[:1])
                fl_verts_last, fl_landmarks3d_last, mh_verts_last, mh_landmarks3d_last = self.model.forward_single(expression_params, pose_params)

                fl_verts1 = torch.cat([fl_verts1, fl_verts_last], 0)
                fl_landmarks3d1 = torch.cat([fl_landmarks3d1, fl_landmarks3d_last], 0)
                mh_verts = torch.cat([mh_verts, mh_verts_last], 0)
                mh_landmarks3d = torch.cat([mh_landmarks3d, mh_landmarks3d_last], 0)

                batch_size = fl_verts1.shape[0]

                ################################
                fl_meshes_op = Meshes(verts=fl_verts1[:, self.fl_keep_vertices], faces=torch.tile(torch.tensor(self.fl_keep_faces).unsqueeze(0).to(cuda), [batch_size, 1, 1]))
                mh_pcls = Pointclouds(mh_verts[:, self.mh_keep_vertices])
                mh_point_dist, mh_face_dist = point_mesh_face_distance(fl_meshes_op, mh_pcls)
                mh_point_dist = torch.reshape(mh_point_dist, [batch_size, -1])
                mh_face_dist = torch.reshape(mh_face_dist, [batch_size, -1])
                mh_loss_op = (mh_point_dist.sum() + mh_face_dist.sum())/batch_size
                # mh_loss_op = mh_loss_op + ((mh_point_dist[:-1].mean() + mh_face_dist[:-1].mean()) - (mh_point_dist[-1:].mean() + mh_face_dist[-1:].mean())) *5000
                # scale = 1
                # mh_loss_op = (mh_point_dist[:-1].sum() + mh_face_dist[:-1].sum() + (mh_point_dist[-1:].sum() + mh_face_dist[-1:].sum()) * scale)/scale /batch_size

                # criterion = torch.nn.MSELoss()
                # mh_lmk_loss = criterion(mh_landmarks3d, fl_landmarks3d1)

                criterion = torch.nn.MSELoss()
                mh_lmk_loss = criterion(mh_landmarks3d, fl_landmarks3d1)
                # mh_lmk_loss = mh_lmk_loss + torch.abs(torch.mean(criterion(mh_landmarks3d[:-1], fl_landmarks3d1[:-1])) - torch.mean(criterion(mh_landmarks3d[-1:], fl_landmarks3d1[-1:])))
                # mh_lmk_loss = (mh_lmk_loss[:-1].mean() + mh_lmk_loss[-1:].mean()*scale)/scale
                
                criterion = torch.nn.MSELoss()
                mh_exp_loss = criterion(torch.sigmoid(self.model.exp_coeffs), torch.zeros_like(self.model.exp_coeffs).to(cuda))
                # print(torch.sigmoid(self.model.exp_coeffs))

                ################################
                mh_meshes_op = Meshes(verts=mh_verts[:, self.mh_keep_vertices], faces=torch.tile(torch.tensor(self.mh_keep_faces).unsqueeze(0).to(cuda), [batch_size, 1, 1]))
                fl_pcls = Pointclouds(fl_verts1[:, self.fl_keep_vertices])
                fl_point_dist, fl_face_dist = point_mesh_face_distance(mh_meshes_op, fl_pcls)
                fl_point_dist = torch.reshape(fl_point_dist, [batch_size, -1])
                fl_face_dist = torch.reshape(fl_face_dist, [batch_size, -1])
                fl_loss_op = (fl_point_dist.sum() + fl_face_dist.sum())/batch_size
                # fl_loss_op = fl_loss_op + ((fl_point_dist[:-1].mean() + fl_face_dist[:-1].mean()) - (fl_point_dist[-1:].mean() + fl_face_dist[-1:].mean()))*5000
                # fl_loss_op = (fl_point_dist[:-1].sum() + fl_face_dist[:-1].sum() + (fl_point_dist[-1:].sum() + fl_face_dist[-1:].sum()) * scale)/scale/batch_size
                
                ################################
                loss = (fl_loss_op + mh_loss_op) * 1.0 + mh_lmk_loss * 10.0 + mh_exp_loss * 1.0
                print(f'===========>>> {epoch}: {torch.abs(self.model.mh_scale)}<<<===========')
                # print('loss_op: {:.4f}, lmk_loss: {:.4f}'.format(fl_loss_op * 1.0, mh_lmk_loss * 10.0))
                print('loss_op: {:.4f}, lmk_loss: {:.4f}, mh_exp_loss: {:.4f}'.format((fl_loss_op + mh_loss_op) * 1.0, mh_lmk_loss * 10.0, mh_exp_loss * 1.0))

                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

        self.test()

    def deal_eye_pose(self):
        up_down_range = 35
        left_right_range = 55
        exp_fullpose = np.load('models/exp_fullpose.npy', allow_pickle=True)
        full_pose = exp_fullpose.item()['full_pose']
        eye_coeffs = np.zeros([full_pose.shape[0], 8])
        smpl_euler = RTT.from_rotvec(full_pose[:, -6:-3]).as_euler('xyz', degrees=True)
        smpl_euler[:, 0] /= up_down_range
        smpl_euler[:, 1] /= left_right_range
        smpl_euler[:, 0] = np.clip(smpl_euler[:, 0], -1, 1)
        smpl_euler[:, 1] = np.clip(smpl_euler[:, 1], -1, 1)
        for i in range(smpl_euler.shape[0]):
            if(smpl_euler[i, 0]>=0 and smpl_euler[i, 1]>=0):
                eye_coeffs[i, 0] = smpl_euler[i, 0]
                eye_coeffs[i, 1] = smpl_euler[i, 1]
            elif(smpl_euler[i, 0]>=0 and smpl_euler[i, 1]<0):
                eye_coeffs[i, 0] = smpl_euler[i, 0]
                eye_coeffs[i, 3] = -smpl_euler[i, 1]
            elif(smpl_euler[i, 0]<0 and smpl_euler[i, 1]>=0):
                eye_coeffs[i, 2] = -smpl_euler[i, 0]
                eye_coeffs[i, 1] = smpl_euler[i, 1]
            elif(smpl_euler[i, 0]<0 and smpl_euler[i, 1]<0):
                eye_coeffs[i, 2] = -smpl_euler[i, 0]
                eye_coeffs[i, 3] = -smpl_euler[i, 1]

        smpl_euler = RTT.from_rotvec(full_pose[:, -3:]).as_euler('xyz', degrees=True)
        smpl_euler[:, 0] /= up_down_range
        smpl_euler[:, 1] /= left_right_range
        smpl_euler[:, 0] = np.clip(smpl_euler[:, 0], -1, 1)
        smpl_euler[:, 1] = np.clip(smpl_euler[:, 1], -1, 1)
        for i in range(smpl_euler.shape[0]):
            if(smpl_euler[i, 0]>=0 and smpl_euler[i, 1]>=0):
                eye_coeffs[i, 4] = smpl_euler[i, 0]
                eye_coeffs[i, 5] = smpl_euler[i, 1]
            elif(smpl_euler[i, 0]>=0 and smpl_euler[i, 1]<0):
                eye_coeffs[i, 4] = smpl_euler[i, 0]
                eye_coeffs[i, 7] = -smpl_euler[i, 1]
            elif(smpl_euler[i, 0]<0 and smpl_euler[i, 1]>=0):
                eye_coeffs[i, 6] = -smpl_euler[i, 0]
                eye_coeffs[i, 5] = smpl_euler[i, 1]
            elif(smpl_euler[i, 0]<0 and smpl_euler[i, 1]<0):
                eye_coeffs[i, 6] = -smpl_euler[i, 0]
                eye_coeffs[i, 7] = -smpl_euler[i, 1]

        avg_weight = 0.7
        eye_coeffs_ret = np.zeros_like(eye_coeffs)
        eye_coeffs_ret[:, 0] = (eye_coeffs[:, 0] + eye_coeffs[:, 4])/2 * avg_weight + eye_coeffs[:, 0] * (1 - avg_weight)
        eye_coeffs_ret[:, 4] = (eye_coeffs[:, 0] + eye_coeffs[:, 4])/2 * avg_weight + eye_coeffs[:, 4] * (1 - avg_weight)
        eye_coeffs_ret[:, 1] = (eye_coeffs[:, 1] + eye_coeffs[:, 5])/2 * avg_weight + eye_coeffs[:, 1] * (1 - avg_weight)
        eye_coeffs_ret[:, 5] = (eye_coeffs[:, 1] + eye_coeffs[:, 5])/2 * avg_weight + eye_coeffs[:, 5] * (1 - avg_weight)
        eye_coeffs_ret[:, 2] = (eye_coeffs[:, 2] + eye_coeffs[:, 6])/2 * avg_weight + eye_coeffs[:, 2] * (1 - avg_weight)
        eye_coeffs_ret[:, 6] = (eye_coeffs[:, 2] + eye_coeffs[:, 6])/2 * avg_weight + eye_coeffs[:, 6] * (1 - avg_weight)
        eye_coeffs_ret[:, 3] = (eye_coeffs[:, 3] + eye_coeffs[:, 7])/2 * avg_weight + eye_coeffs[:, 3] * (1 - avg_weight)
        eye_coeffs_ret[:, 7] = (eye_coeffs[:, 3] + eye_coeffs[:, 7])/2 * avg_weight + eye_coeffs[:, 7] * (1 - avg_weight)
        return eye_coeffs_ret

    def test(self):
        eye_coeffs = self.deal_eye_pose()
        self.model.eval()
        idxs = list(range(0, self.total_size))
        mh_verts, _ = self.model.infer(torch.tensor(idxs).to(cuda))
        exp_coeffs = torch.sigmoid(self.model.exp_coeffs).detach().cpu().numpy()
        np.save('output/morphTargets.npy', exp_coeffs)
        mFile = open('sampleMorTargets.txt', 'w')
        for i in range(exp_coeffs.shape[0]):
        # for i in range(200):
            mort = map(lambda x: '{:.3f}'.format(x), list(exp_coeffs[i]) + list(eye_coeffs[i]))
            line = ' '.join(list(mort)) +'\n'
            mFile.writelines(line)
        mFile.close()
        render_sequence('wav_clips/tou.wav', 'output', mh_verts.detach().cpu().numpy(), self.model.metahuman.faces_tensor.cpu().numpy())


if(__name__=='__main__'):
    trainer = Trainer()
    trainer.train()

    # metahuman_model_path = os.path.join('models', 'metahuman_model.pkl') 
    # metahuman = MetaHuman(metahuman_model_path)
    # dict = pickle.load(open(metahuman_model_path, 'rb'), encoding='latin1')
    # mh_verts = []
    # for i in range(35):
    #     exp_coeffs = torch.zeros([1, 35])
    #     exp_coeffs[0, i] = 1
    #     mh_vert, _ = metahuman(exp_coeffs)
    #     mh_verts.append(mh_vert[0].detach().cpu().numpy())
    # print(np.array(mh_verts).shape)
    # render_sequence('wav_clips/ObamaBiden.wav', 'output', np.array(mh_verts), metahuman.faces_tensor.cpu().numpy())


    # mesh_root = r'/root/sample10000'
    # mesh_list = []
    # for i in range(9963, 10000):
    #     name = os.path.join(mesh_root, 'mesh_{:04d}.obj'.format(i))
    #     myMesh = trimesh.load(name, process=False, maintain_order=True)
    #     mesh_list.append(myMesh.vertices)
    # render_sequence('wav_clips/ObamaBiden.wav', 'output', np.array(mesh_list), myMesh.faces)

    # angle = np.array([np.pi/2/np.sqrt(2), np.pi/2/np.sqrt(2), 0])
    # print(RTT.from_rotvec(angle).as_euler('xyz', degrees=True))
    