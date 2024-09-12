import numpy as np
import os, shutil
import cv2
from subprocess import call
os.environ['PYOPENGL_PLATFORM'] = 'egl' # egl
import pyrender
from psbody.mesh import Mesh
import trimesh


def render_mesh_helper(mesh, t_center, rot=np.zeros(3), z_offset=0):
    camera_params = {'c': np.array([256, 256]),
                        'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                        'f': np.array([3254.97941935 / 2, 3254.97941935 / 2])}

    frustum = {'near': 0.01, 'far': 3.0, 'height': 512, 'width': 512}

    mesh_copy = Mesh(mesh.v, mesh.f)
    mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v-t_center).T).T+t_center
    intensity = 2.0
    rgb_per_v = None

    primitive_material = pyrender.material.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorFactor=[0.3, 0.3, 0.3, 1.0],
                metallicFactor=0.8, 
                roughnessFactor=0.8 )

    tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f, vertex_colors=rgb_per_v)
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material,smooth=True)

    scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])

    camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                      fy=camera_params['f'][1],
                                      cx=camera_params['c'][0],
                                      cy=camera_params['c'][1],
                                      znear=frustum['near'],
                                      zfar=frustum['far'])

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3,3] = np.array([0, 0, 1.0-z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])

    angle = np.pi / 6.0
    pos = camera_pose[:3,3]
    light_color = np.array([1., 1., 1.])
    light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3,3] = pos
    scene.add(light, pose=light_pose.copy())
    
    light_pose[:3,3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    # light_pose[:3,3] =  cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    # scene.add(light, pose=light_pose.copy())

    # light_pose[:3,3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    # scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
        color, _ = r.render(scene, flags=flags)
    except:
        print('pyrender: Failed rendering frame')
        color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]


def render_sequence(wav_path, output_path, vertice_out, faces):
    if(wav_path is not None):
        test_name = os.path.basename(wav_path).split(".")[0]
    else:
        test_name = 'test'
    print("rendering: ", test_name)

    verts = np.reshape(vertice_out, (-1, vertice_out.shape[1], 3))

    num_frames = verts.shape[0]
    center = np.mean(verts[0], axis=0)

    from multiprocessing import Process
    import hashlib, time, random
    def threadRun(i_frames, tmp_folder):
        for i_frame in i_frames:
            render_mesh = Mesh(verts[i_frame], faces)
            pred_img = render_mesh_helper(render_mesh, center)
            pred_img = pred_img.astype(np.uint8)
            cv2.imwrite('{}/{:04d}.jpg'.format(tmp_folder, i_frame), pred_img)

    threads = []
    thread_count = 10
    length = num_frames//thread_count
    m = hashlib.md5()
    m.update(bytes(str(time.time())+str(random.random()), encoding='utf-8'))
    tmp_folder = f'tmp_{m.hexdigest()}'
    os.makedirs(tmp_folder, exist_ok=True)
    for i in range(thread_count):
        if(i==thread_count-1):
            i_frames = list(range(length*i, num_frames))
        else:
            i_frames = list(range(length*i, length*(i+1)))
        t = Process(target=threadRun, args=(i_frames, tmp_folder))
        threads.append(t)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    video_fname = os.path.join(output_path, test_name+'.mp4')
    if(wav_path is not None):
        cmd = ('ffmpeg' + ' -r 25 -i {}/%04d.jpg -i {} -pix_fmt yuv420p -qscale 0 -y {}'.format(tmp_folder, wav_path, video_fname)).split()
    else:
        cmd = ('ffmpeg' + ' -r 25 -i {}/%04d.jpg -pix_fmt yuv420p -qscale 0 -y {}'.format(tmp_folder, video_fname)).split()
    call(cmd)
    # shutil.rmtree(tmp_folder)
