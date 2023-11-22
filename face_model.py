import numpy as np
from tqdm import tqdm
import trimesh, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from metahuman.lbs_metahuman import lbs, batch_rodrigues
cuda = torch.device('cuda:0')

def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


class MetahumanDataset(Dataset):
    def __init__(self, mesh_root, label_file):
        lines = open(label_file).readlines()
        self.mesh_list = []
        self.labels = []
        self.indices = []
        for i, line in enumerate(tqdm(lines)):
            self.indices.append(i)
            name = os.path.join(mesh_root, 'mesh_{:04d}.obj'.format(i))
            myMesh = trimesh.load(name, process=False, maintain_order=True)
            self.mesh_list.append(myMesh.vertices)
            self.labels.append([float(x) for x in line.strip().split()])
        self.faces = myMesh.faces

    def __len__(self):
        return len(self.mesh_list)

    def __getitem__(self, idx):
        label = self.labels[idx]
        return to_tensor(self.mesh_list[idx]), to_tensor(label), to_tensor(self.indices[idx], dtype=torch.long)


class MetahumanModel(nn.Module):
    NUM_JOINTS = 1
    EXPRESSION_SPACE_DIM = 32
    POSE_SPACE_DIM = 3
    NUM_VERTICES = 24533

    INIT_POINTS = set([3026, 3028, 3029, 3030, 3031, 3032, 3033, 3034, 3032, 3033, 3034, 3035, 3036, 3037, 3035, 3037, 3047, 3049, 3050, 3051, 3052, 3053, 3054, 3055, 3056, 3057, 3058, 3059, 3060, 3061, 3062, 3063, 3064, 3065, 3066, 3067, 3068, 3069, 3070, 3071, 3069, 3072, 3074, 3075, 3076, 3077, 3078, 3079, 3080, 3081, 3082, 3083, 3084, 3085, 3086, 3087, 3088, 3089, 3090, 3091, 3092, 3093, 3094, 3095, 3096, 3097, 3098, 3099, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3112, 3113, 3114, 3115, 3116, 3117, 3118, 3119, 3120, 3121, 3119, 3120, 3121, 3122, 3123, 3124, 3125, 3126, 3165, 3167, 3169, 3171, 3172, 3173, 3174, 3175, 3176, 3177, 3178, 3179, 3180, 3181, 3182, 3183, 3184, 3185, 3186, 3187, 3188, 3189, 3187, 3188, 3189, 3190, 3191, 3192, 3193, 3194, 3195, 3196, 3197, 3198, 3199, 3200, 3201, 3202, 3203, 3204, 3205, 3206, 3207, 3208, 3209, 3210, 3211, 3212, 3213, 3214, 3215, 3216, 3217, 3218, 3216, 3217, 3218, 3219, 3220, 3221, 3220, 3221, 3222, 3223, 3224, 3225, 3223, 7375, 7381, 7383, 7385, 7386, 7387, 7388, 7389, 7390, 7391, 7389, 7390, 7391, 7392, 7393, 7394, 7395, 7396, 7397, 7398, 7399, 7400, 7401, 7402, 7403, 7404, 7405, 7406, 7407, 7408, 7409, 7410, 7411, 7412, 7413, 7414, 7415, 7416, 7417, 7418, 7419, 7420, 7421, 7422, 7423, 7424, 7425, 7426, 7427, 7428, 7429, 7430, 7431, 7432, 7433, 7434, 7435, 7436, 7437, 7438, 7439, 7440, 7441, 7442, 7443, 7444, 7445, 7446, 7447, 7448, 7449, 7450, 7451, 7452, 7453, 7454, 7455, 7456, 7457, 7458, 7459, 7460, 7461, 7462, 7463, 7464, 7465, 7466, 7467, 7468, 7469, 7470, 7471, 7472, 7473, 7474, 7475, 7476, 7477, 7478, 7479, 7480, 7481, 7482, 7483, 7484, 7485, 7486, 7487, 7488, 7489, 7490, 7491, 7492, 7493, 7494, 7495, 7496, 7497, 7498, 7499, 7500, 7501, 7502, 7503, 7504, 7505, 7503, 7505, 7508, 7538, 7539, 7540, 7541, 7542, 7543, 7544, 7545, 7546, 7547, 7548, 7549, 7550, 7548, 7551, 7552, 7553, 7554, 7555, 7556, 7557, 7558, 7559, 7560, 7561, 7562, 7563, 7564, 7565, 7566, 7567, 7568, 7569, 7570, 7571, 7572, 7573, 7574, 7575, 7576, 7577, 7578, 7579, 7580, 7789, 7791, 7793, 7794, 7795, 7796, 7797, 7798, 7799, 7797, 7798, 7799, 7800, 7801, 7802, 7800, 7801, 7802, 7803, 7804, 7805, 7806, 7807, 7808, 7809, 7810, 7811, 7812, 7813, 7814, 7815, 7816, 7817, 7818, 7819, 7820, 7821, 7822, 7823, 7824, 7825, 7826, 7827, 7828, 7829, 7830, 7831, 7832, 7833, 7834, 7835, 7836, 7837, 7838, 7839, 7840, 7841, 7842, 7843, 7844, 7845, 7846, 7847, 7848, 7849, 7850, 7851, 7852, 7853, 7854, 7855, 7856, 7857, 11350, 11353, 11354, 11355, 11356, 11357, 11358, 11359, 11360, 11361, 11362, 11363, 11364, 11365, 11366, 11367, 11368, 11369, 11370, 11371, 11372, 11373, 11374, 11375, 11376, 11376, 11377, 11378, 11379, 11380, 11381, 11382, 11383, 11381, 11382, 11383, 11384, 11385, 11386, 11387, 11388, 11389, 11390, 11391, 11392, 11393, 11394, 11395, 11396, 11397, 11398, 11396, 11397, 11398, 11399, 11400, 11401, 11400, 11403, 11405, 11406, 11407, 11408, 11409, 11410, 11411, 11412, 11413, 11414, 11415, 11416, 11417, 11418, 11419, 11420, 11421, 11422, 11423, 11424, 11425, 11426, 11427, 11428, 11429, 11430, 11431, 11432, 11433, 11434, 11435, 11436, 11437, 11438, 11439, 11440, 11441, 11442, 11443, 11444, 11445, 11446, 11447, 11448, 11449, 11450, 11451, 11452, 11453, 11454, 11455, 11456, 11457, 11458, 11459, 11460, 11461, 11462, 11463, 11464, 11465, 11466, 11467, 11468, 11469, 11470, 11471, 11472, 11473, 11474, 11475, 11476, 11477, 11478, 11479, 11480, 11481, 11482, 11483, 11484, 11485, 11486, 11487, 11488, 11489, 11490, 11491, 11492, 11493, 11494, 11495, 11496, 11497, 11498, 11499, 11500, 11498, 11501, 11577, 11579, 11580, 11581, 11582, 11583, 11584, 11585, 11586, 11585, 11587, 11588, 11589, 11590, 11591, 11592, 11593, 11594, 11595, 11596, 11597, 11598, 11599, 11600, 11601, 11602, 11603, 11604, 11605, 11606, 11607, 11620, 21255, 21257])

    def __init__(self, template_path) -> None:
        super(MetahumanModel, self).__init__()
        self.dtype = torch.float32
        v_template = trimesh.load(template_path, process=False, maintain_order=True).vertices
        self.register_buffer('v_template', torch.tensor(v_template, dtype=self.dtype))
        self.register_buffer('parents', torch.tensor(np.array([-1, 0]), dtype=torch.long))
        self.register_parameter('expression_basis', nn.Parameter(torch.zeros([self.NUM_VERTICES, 3, self.EXPRESSION_SPACE_DIM], dtype=self.dtype)))
        self.register_parameter('pose_basis', nn.Parameter(torch.zeros([9*self.NUM_JOINTS, 3*self.NUM_VERTICES], dtype=self.dtype)))
        
        lll = torch.zeros([self.NUM_VERTICES, self.NUM_JOINTS+1], dtype=self.dtype)
        for index in range(self.NUM_VERTICES):
            if(index in self.INIT_POINTS):
                lll[index, 1] = 1
            else:
                lll[index, 0] = 1
        self.register_parameter('_lbs_weights', nn.Parameter(lll))
        self.register_buffer('J_root', to_tensor([[0, 0, 0]]))
        self.register_buffer('J_init', to_tensor((np.mean(v_template, axis=0)-np.array([0, 0.03, 0.02]))[np.newaxis, ...]))
        self.register_parameter('J_coordinate', nn.Parameter(self.J_init))
        self.register_buffer('jaw_pose_basis_init', to_tensor([[np.pi/4, 0, 0], [0, np.pi/10, 0], [0, -np.pi/10, 0]]))
        self.register_parameter('jaw_pose_basis', nn.Parameter(self.jaw_pose_basis_init))
        self.register_parameter('transl', nn.Parameter(to_tensor(np.zeros([10000, 3]))))
        self.register_parameter('global_matrix', nn.Parameter(torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat([10000,1,1])))
        self.register_parameter('scale', nn.Parameter(to_tensor(np.ones([10000, 1]))))
        # self.register_parameter('curve', nn.Parameter(to_tensor(np.ones([1, 35]))))

    def forward(self, coeffs, index):
        self.lbs_weights = F.softmax(self._lbs_weights, dim=-1)
        
        batch_size = coeffs.shape[0]
        jaw_pose9 = batch_rodrigues(torch.multiply(self.jaw_pose_basis[0].unsqueeze(0).repeat(batch_size, 1), coeffs[..., 9:10]))
        jaw_pose8 = batch_rodrigues(torch.multiply(self.jaw_pose_basis[1].unsqueeze(0).repeat(batch_size, 1), coeffs[..., 8:9]))
        jaw_pose7 = batch_rodrigues(torch.multiply(self.jaw_pose_basis[2].unsqueeze(0).repeat(batch_size, 1), coeffs[..., 7:8]))
        jaw_matrix = torch.matmul(torch.matmul(jaw_pose9, jaw_pose8), jaw_pose7).view(batch_size, 1, 9)
        # global_matrix = self.global_matrix.repeat(batch_size, 1, 1).view(batch_size, 1, 9)
        global_matrix = torch.index_select(self.global_matrix, dim=0, index=index).view(batch_size, 1, 9)
        full_pose = torch.concat([global_matrix, jaw_matrix], 1)
        exp_coeffs = torch.concat([coeffs[:, :7], coeffs[:, 10:]], dim=1)

        vertices, _ = lbs(exp_coeffs, full_pose, self.v_template.unsqueeze(0).repeat(batch_size, 1, 1),
                            self.expression_basis, self.pose_basis,
                            torch.concat([self.J_root, self.J_coordinate], dim=0).repeat(batch_size, 1, 1), self.parents,
                            self.lbs_weights, pose2rot=False)
        vertices += torch.index_select(self.transl.unsqueeze(1).repeat(1, self.NUM_VERTICES, 1), dim=0, index=index)
        vertices *= torch.index_select(self.scale.unsqueeze(1).repeat(1, self.NUM_VERTICES, 1), dim=0, index=index)
        return vertices


class Trainer(nn.Module):
    def __init__(self):
        super(Trainer, self).__init__()
        mesh_root = r'/root/sample10000'
        label_file = r'/root/lines.txt'
        template_path = r'/root/sample10000/mesh_9999.obj'
        self.training_data = MetahumanDataset(mesh_root, label_file)
        self.batch_size = 32
        self.train_dataloader = DataLoader(self.training_data, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.mtModel = MetahumanModel(template_path).to(cuda)
        self.optimizer = torch.optim.Adam(self.mtModel.parameters(), lr=1e-4)
        # self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=100)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)

    def train(self):
        max_epoch = 100
        for epoch in range(1, max_epoch+1):
            self.mtModel.train()

            for i, (verts, morphTarget, index) in enumerate(self.train_dataloader):
                verts = verts.to(cuda)
                morphTarget = morphTarget.to(cuda)
                index = index.to(cuda)

                vertices = self.mtModel(morphTarget, index)

                verts_loss = torch.sum(torch.square(verts - vertices))/self.batch_size
                exp_loss = torch.sum(torch.square(self.mtModel.expression_basis))/self.batch_size
                pose_loss = torch.sum(torch.square(self.mtModel.pose_basis))/self.batch_size
                J_loss = torch.sum(torch.square(self.mtModel.J_coordinate-self.mtModel.J_init))*self.mtModel.NUM_VERTICES
                jaw_basis_loss = torch.sum(torch.square(self.mtModel.jaw_pose_basis-self.mtModel.jaw_pose_basis_init))*self.mtModel.NUM_VERTICES
                # jaw_pose_scale_loss = torch.sum(torch.square(self.mtModel.jaw_pose_scale))
                # weight_loss = -torch.mean(torch.square(self.mtModel.lbs_weights - torch.ones_like(self.mtModel.lbs_weights)*0.5))
                verts_loss *= 1.0
                exp_loss *= 0.1
                pose_loss *= 0.1
                J_loss *= 1.0
                jaw_basis_loss *= 1.0
                # jaw_pose_scale_loss *= 10.0
                loss = verts_loss + exp_loss + pose_loss + J_loss + jaw_basis_loss

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                print('{}_{}: loss_op: {:.4f}, verts_loss: {:.4f}, exp_loss: {:.4f}, pose_loss: {:.4f}, J_loss: {:.4f}, jaw_basis_loss: {:.4f}'.format(epoch, i, loss, verts_loss, exp_loss, pose_loss, J_loss, jaw_basis_loss))
            if(epoch%50==0):
                os.makedirs(f'{epoch}', exist_ok=True)
                self.mtModel.eval()
                morphTarget = to_tensor(self.training_data.labels[:10])
                morphTarget = morphTarget.to(cuda)
                index = to_tensor(torch.arange(0, 10), dtype=torch.long).to(cuda)
                vertices = self.mtModel(morphTarget, index).detach().cpu().numpy()
                weights = F.softmax(self.mtModel._lbs_weights, dim=-1).detach().cpu().numpy()
                for iii in range(morphTarget.shape[0]):
                    self.draw_vertex_color(f'{epoch}/{iii}.obj', vertices[iii], self.training_data.faces, weights, self.mtModel.J_coordinate.detach().cpu().numpy())

                morphTarget = to_tensor(self.training_data.labels[-100:])
                morphTarget = morphTarget.to(cuda)
                index = to_tensor(torch.arange(10000-100, 10000), dtype=torch.long).to(cuda)
                vertices = self.mtModel(morphTarget, index).detach().cpu().numpy()
                weights = F.softmax(self.mtModel._lbs_weights, dim=-1).detach().cpu().numpy()
                for iii in range(morphTarget.shape[0]):
                    self.draw_vertex_color(f'{epoch}/{9900+iii}.obj', vertices[iii], self.training_data.faces, weights, self.mtModel.J_coordinate.detach().cpu().numpy())

            self.scheduler.step()

        torch.save(self.mtModel.state_dict(), f'models/last.pth')


    def draw_vertex_color(self, out_path, vertices, faces, weights, J_coordinate):
        vertex_colors = np.ones([vertices.shape[0]+1, 3])*0.75
        for i in range(len(vertices)):
            vertex_colors[i] = np.array([0.75, 0.75, 0.75]) * weights[i, 0]

        vertices = np.concatenate([vertices, J_coordinate], 0)
        vertex_colors[-1] = np.array([1, 0, 0])
        more = np.array([[24532, 24533, 24534]])
        faces = np.concatenate([faces, more], 0)

        hah = trimesh.Trimesh(vertices, faces, process=False, vertex_colors=vertex_colors)
        hah.export(out_path)


def pth2pkl_eval(pth_path, out_pkl):
    # dump to pkl
    import pickle
    weights = torch.load(pth_path, map_location='cpu')
    metahuman_model = {}
    for key in weights.keys():
        metahuman_model[key] = np.array(weights[key])

    lmks_idx = np.load('models/mh_lmks_idx.npy')
    template = trimesh.load('models/metahuman_mesh.obj', process=False, maintain_order=True)

    metahuman_model['f'] = template.faces
    metahuman_model['lmks_idx'] = lmks_idx
    metahuman_model['lbs_weights'] = F.softmax(torch.tensor(metahuman_model['_lbs_weights']), dim=-1).detach().numpy()
    metahuman_model['v_template'] = template.vertices
    metahuman_model['parents'] = metahuman_model['parents']
    del metahuman_model['_lbs_weights']
    del metahuman_model['transl']
    del metahuman_model['scale']
    del metahuman_model['global_matrix']
    del metahuman_model['J_root']
    del metahuman_model['J_init']
    del metahuman_model['jaw_pose_basis_init']
    pickle.dump(metahuman_model, open(out_pkl, 'wb'))
    print(metahuman_model.keys())

    # test coeffcient
    batch_size = 1
    exp_coeffs = torch.zeros([batch_size, 32])
    exp_coeffs[0, 0] = 1.0
    jaw_coeffs = to_tensor([[0.3, 0, 0]])
    jaw_pose_basis = to_tensor(metahuman_model['jaw_pose_basis'])
    v_template = to_tensor(metahuman_model['v_template'])
    J_coordinate = to_tensor(metahuman_model['J_coordinate'])
    expdirs = to_tensor(metahuman_model['expression_basis'])
    posedirs = to_tensor(metahuman_model['pose_basis'])
    lbs_weights = to_tensor(metahuman_model['lbs_weights'])
    parents = to_tensor(metahuman_model['parents'], dtype=torch.long)

    jaw_pose9 = batch_rodrigues(torch.multiply(jaw_pose_basis[0].unsqueeze(0).repeat(batch_size, 1), jaw_coeffs[:, 0:1]))
    jaw_pose8 = batch_rodrigues(torch.multiply(jaw_pose_basis[1].unsqueeze(0).repeat(batch_size, 1), jaw_coeffs[:, 1:2]))
    jaw_pose7 = batch_rodrigues(torch.multiply(jaw_pose_basis[2].unsqueeze(0).repeat(batch_size, 1), jaw_coeffs[:, 2:3]))
    jaw_matrix = torch.matmul(torch.matmul(jaw_pose9, jaw_pose8), jaw_pose7).view(batch_size, 1, 9)
    global_matrix = torch.eye(3).unsqueeze(0).repeat([batch_size, 1, 1]).view(batch_size, 1, 9)
    full_pose = torch.concat([global_matrix, jaw_matrix], 1)

    vertices, _ = lbs(exp_coeffs, full_pose, v_template.unsqueeze(0).repeat(batch_size, 1, 1),
                        expdirs, posedirs,
                        torch.concat([torch.tensor([[0, 0, 0]]), J_coordinate], dim=0).repeat(batch_size, 1, 1), parents,
                        lbs_weights, pose2rot=False)
    hah = trimesh.Trimesh(vertices[0].detach().numpy(), metahuman_model['f'], process=False)
    hah.export('test.obj')


if(__name__=='__main__'):
    # trainer = Trainer()
    # trainer.train()
    pth2pkl_eval('models/last.pth', 'models/metahuman_model.pkl')
