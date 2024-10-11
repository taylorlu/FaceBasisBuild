import numpy as np
from scipy.spatial.transform import Rotation as RTT


def deal_eye_pose(eye_pose):
    up_down_range = 35
    left_right_range = 55

    eye_coeffs = np.zeros([eye_pose.shape[0], 8])
    smpl_euler = RTT.from_rotvec(eye_pose[:, -6:-3]).as_euler('xyz', degrees=True)
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

    smpl_euler = RTT.from_rotvec(eye_pose[:, -3:]).as_euler('xyz', degrees=True)
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


if(__name__=='__main__'):
    neutral = np.zeros([35])
    happiness = np.zeros([35])
    fear = np.zeros([35])
    anger = np.zeros([35])
    disgust = np.zeros([35])
    sadness = np.zeros([35])

    happiness[13] = 0.4
    happiness[14] = 0.4
    happiness[-4] = 0.8
    happiness[-3] = 0.8

    fear[4] = 0.5
    fear[5] = 0.5
    fear[19] = 0.5
    fear[20] = 0.5
    fear[27] = 0.5
    fear[28] = 0.5
    fear[29] = 0.5

    anger[4] = 0.5
    anger[5] = 0.5
    anger[21] = 0.5
    anger[22] = 0.5
    anger[25] = 0.5
    anger[26] = 0.5
    anger[-2] = 0.5
    anger[-1] = 0.5

    disgust[15] = 0.5
    disgust[16] = 0.5
    disgust[-2] = 0.5
    disgust[-1] = 0.5

    sadness[15] = 0.5
    sadness[16] = 0.5
    sadness[27] = 0.5
    sadness[28] = 0.5
    sadness[29] = 0.5
    sadness[-4] = 0.5
    sadness[-3] = 0.5

    emotions = np.concatenate([neutral[None, :], happiness[None, :], fear[None, :], anger[None, :], disgust[None, :], sadness[None, :]], axis=0)

    morphTargets = np.load('morphTargets.npy')
    trunk_size = morphTargets.shape[0] // emotions.shape[0]

    emo_morphTargets = []
    blink_list = [0.3, 0.6, 0.9, 0.6, 0.3]
    for i in range(emotions.shape[0]):
        emo_morphs = morphTargets[trunk_size*i:trunk_size*(i+1), :] + emotions[i][None, :]
        emo_morphs[0, :2] += blink_list[0]
        emo_morphs[1, :2] += blink_list[1]
        emo_morphs[2, :2] += blink_list[2]
        emo_morphs[3, :2] += blink_list[3]
        emo_morphs[4, :2] += blink_list[4]
        emo_morphTargets.append(emo_morphs)

    emo_morphTargets = np.concatenate(emo_morphTargets, axis=0)

    # 6 (emotion) * 25 (fps) = 150 frames
    # 150 + 35 + 1 (zero pose) + 4 (extra eye pose) = 190 frames
    # 190 * 25 (viewport) = 4750 images
    emo_morphTargets = np.concatenate([emo_morphTargets, np.eye(35), np.zeros([1, 35])], axis=0)

    eye_pose = np.array([[np.pi/6, 0, 0, np.pi/6, 0, 0], 
                         [0, np.pi/6, 0, 0, np.pi/6, 0], 
                         [-np.pi/6, 0, 0, -np.pi/6, 0, 0], 
                         [0, -np.pi/6, 0, 0, -np.pi/6, 0]])
    eye_morphTargets = deal_eye_pose(eye_pose)

    # eye_morphTargets = np.zeros([4, 8])
    # eye_morphTargets[0, 0] = eye_morphTargets[0, 4] = 0.5
    # eye_morphTargets[1, 1] = eye_morphTargets[1, 5] = 0.5
    # eye_morphTargets[2, 2] = eye_morphTargets[2, 6] = 0.5
    # eye_morphTargets[3, 3] = eye_morphTargets[3, 7] = 0.5

    eye_morphTargets = np.pad(eye_morphTargets, [[emo_morphTargets.shape[0], 0], [0, 0]], mode='constant', constant_values=0)
    emo_morphTargets = np.pad(emo_morphTargets, [[0, 4], [0, 0]], mode='constant', constant_values=0)
    print(eye_morphTargets.shape, emo_morphTargets.shape)

    # # we add 10 frames until warmup the scene
    eye_morphTargets = np.pad(eye_morphTargets, [[10, 0], [0, 0]], mode='constant', constant_values=0)
    emo_morphTargets = np.pad(emo_morphTargets, [[10, 0], [0, 0]], mode='constant', constant_values=0)
    print(eye_morphTargets.shape, emo_morphTargets.shape)

    with open('sampleMorTargets.txt', 'w') as mFile:
        for i in range(emo_morphTargets.shape[0]):
            mort = map(lambda x: '{:.3f}'.format(x), list(emo_morphTargets[i]) + list(eye_morphTargets[i]))
            line = ' '.join(list(mort)) +'\n'
            mFile.writelines(line)
