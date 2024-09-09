import os
import json
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


def vis_depth_map_and_joint(depth_map, depth_x, depth_y):
    plt.figure()
    plt.imshow(depth_map)
    plt.scatter(depth_x, depth_y, color='red',s=2)
    plt.savefig("output.jpg")
    print("Save to output.jpg")

dataset_path = 'data/data_3d_' + 'h36m' + '.npz'
from common.h36m_dataset import Human36mDataset
dataset = Human36mDataset(dataset_path)

print('Loading 2D detections...')
keypoints = np.load('data/data_2d_' + 'h36m' + '_' + 'cpn_ft_h36m_dbb' + '.npz', allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
# joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
keypoints = keypoints['positions_2d'].item()

depth_dic = {}
for subject in tqdm(keypoints.keys()):
    print("Dealing {}".format(subject))
    depth_dic[subject] = {}
    for action in tqdm(keypoints[subject]):
        print("Dealing {}".format(action))
        action1 = action.replace(" ", "_")
        depth_dic[subject][action1] = {}
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            mocap_length = len(dataset[subject][action]['positions'])
            if kps.shape[0] > mocap_length:
                kps = kps[:mocap_length]
            # Normalize camera frame

            cam = dataset.cameras()[subject][cam_idx]
            cam_idx = cam['id']
            depth_dic[subject][action1][cam_idx] = {}
            file_name = subject + '_' + action1 + '.' + cam_idx

            depth_file_name = os.path.join("depth", subject, file_name)
            seq_depth = []
            for i in range(len(kps)):
                if i % 5 == 1 :
                    depth_pos = kps[i]
                    depth_x = depth_pos[:, 0]
                    depth_y = depth_pos[:, 1]
                    image_path = depth_file_name + "_" + str(i).zfill(6) + ".jpg"
                    depth_map = Image.open(image_path)
                    depth_z = []
                    for j in range(depth_x.shape[0]):
                        depth_x0 = depth_x[j]
                        depth_y0 = depth_y[j]
                        depth_z.append(depth_map.getpixel((depth_x0, depth_y0))[0]/255)
                    depth_z = np.array(depth_z).astype(np.float32)
                    seq_depth.append(depth_z)
            seq_depth = np.stack(seq_depth)
            depth_dic[subject][action1][cam_idx] = seq_depth
            # with open("depth_dic.json", 'w') as json_file:
            #     json.dump(depth_dic, json_file, indent=4)
with open("depth_dic.pkl", 'wb') as file:
    pkl.dump(depth_dic, file)

print("Done")
            
