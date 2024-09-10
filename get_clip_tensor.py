import argparse
import cv2
import numpy as np
import os
import torch
import time
import clip
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
from PIL import Image
import pickle as pkl

def read_img(filelist, img_embedding, img_preprocess, device):
    output_image_features = []
    images = [Image.open(x).convert("RGB") for x in filelist]
    transformered_images =  torch.stack([img_preprocess(image) for image in images]).to(device)
    with torch.no_grad():
        image_features = img_embedding.encode_image(transformered_images)
        output_image_features.append(image_features.float())
    return torch.stack(output_image_features)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str, default="../h36m")
    parser.add_argument('--outdir', type=str, default='./clip_dic')
    parser.add_argument('--encoder', type=str, default='vitb', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--gpu', type=str, default='4')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    margin_width = 50
    caption_height = 60
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    
    DEVICE = 'cuda:{}'.format(args.gpu)
    # os.environ[]
    # DEVICE = 'cuda:5' if torch.cuda.is_available() else 'cpu'
    # depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(args.encoder)).to(
        # DEVICE).eval()
    model_clip, preprocess = clip.load("ViT-B/32", device=DEVICE)



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

    # clip_output_dic = {}
    target_folder = "clip_feature"
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for subject in tqdm(keypoints.keys()):
        print("Dealing {}".format(subject))
        # clip_output_dic[subject] = {}
        if subject not in ['S1']:
            continue

        subject_folder = os.path.join(target_folder, subject)
        if not os.path.exists(subject_folder):
            os.makedirs(subject_folder)
        for action in tqdm(keypoints[subject]):
            print("Dealing {}".format(action))
            action1 = action.replace(" ", "_")

            action_folder = os.path.join(subject_folder, action1)
            if not os.path.exists(action_folder):
                os.makedirs(action_folder)
            else:
                continue
            # clip_output_dic[subject][action1] = {}
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                mocap_length = len(dataset[subject][action]['positions'])
                if kps.shape[0] > mocap_length:
                    kps = kps[:mocap_length]
                cam = dataset.cameras()[subject][cam_idx]
                cam_idx = cam['id']
                # clip_output_dic[subject][action1][cam_idx] = {}
                file_name = subject + '_' + action1 + '.' + cam_idx
                clip_file_name = os.path.join("../h36m", subject, file_name)
                
                clip_filelist = []
                start_time = time.time()
                for i in range(len(kps)):
                    if i % 5 == 1 :
                        img_file_name = clip_file_name + "_" + str(i).zfill(6) + ".jpg"
                        clip_filelist.append(img_file_name)
                image_feature = read_img(clip_filelist, model_clip, preprocess, DEVICE)
                end_time = time.time()


                save_path = str(cam_idx) + ".pt"
                save_path = os.path.join(action_folder, save_path)
                torch.save(image_feature, save_path)
                # clip_output_dic[subject][action1][cam_idx] = image_feature
    # torch.save(clip_output_dic, "clip_dic.pt")
    # print("Done")
                # print("Done {}".format(end_time - start_time))
                # print("Done")
#                     # depth_map = Image.open(image_path)
#                     # image = Image.open(x)
#                     depth_z = []
#                     for j in range(depth_x.shape[0]):
#                         depth_x0 = depth_x[j]
#                         depth_y0 = depth_y[j]
#                         depth_z.append(depth_map.getpixel((depth_x0, depth_y0))[0]/255)
#                     depth_z = np.array(depth_z).astype(np.float32)
#                     seq_depth.append(depth_z)
#             seq_depth = np.stack(seq_depth)
#             clip_output_dic[subject][action1][cam_idx] = seq_depth
#             # with open("clip_output_dic.json", 'w') as json_file:
#             #     json.dump(clip_output_dic, json_file, indent=4)
# with open("clip_output_dic.pkl", 'wb') as file:
#     pkl.dump(clip_output_dic, file)
# with open("depth_dic.pkl", 'wb') as file:
#     pkl.dump(depth_dic, file)

    
 
