import os
import torch
import pickle as pkl
import numpy as np
from scipy.interpolate import interp1d

TrainSet = ['S1', 'S5', 'S6', 'S7', 'S8']
TestSet = ['S9', 'S11']

def parser_name(filename):
    subject = ""
    action = ""
    cam = ""
    idx = ""
    return subject, action, cam, idx
# input_file_name = "S11_Phoing_2.58860488"
def get_relative_depth(filename):
    # from file name to get: subject, action, cam, idx
    subject, action, cam, idx = parser_name(filename)
    z_depth_path = os.path.join(subject, action, cam + '.pkl')
    z_depth_path = os.path.join('z_depth', z_depth_path)
    with open(z_depth_path, 'rb') as file:
        data = pkl.load(file)
    use_idx = [idx[i/5] for i in idx]
    rel_z_depth = data[use_idx]
    return rel_z_depth

def get_image_clip_tensor(filename):
    data = torch.load(filename)
    subject, action, cam, idx = parser_name(filename)
    use_idx = [idx[i/5] for i in idx]
    image_tensor = data[use_idx]
    return image_tensor

def get_depth_dic(folder_path, type='train'):
    root_depth = "z_depth"
    target_dic = {}
    if type == 'train':
        Target_Set = TrainSet
    elif type == 'test':
        Target_Set = TestSet
    for file in Target_Set:
        print("Loading {}".format(file))
        target_dic[file] = {}
        file_path = os.path.join(root_depth, file)
        for action in os.listdir(file_path):
            print("Loading {}".format(action))
            target_dic[file][action] = {}
            action_path = os.path.join(file_path, action)
            for cam_file in os.listdir(action_path):
                cam = cam_file.split(".")[0]
                cam_path = os.path.join(action_path, cam_file)
                with open(cam_path, 'rb') as fl:
                    data = pkl.load(fl)
                target_dic[file][action][cam] = data
    # np.save("{}_depth.npy".format(type), target_dic)
    save_path = "{}_depth.pkl".format(type)
    with open(save_path, 'wb') as fl:
        pkl.dump(target_dic, fl)


def get_clip_tensor_dic(type='train'):
    root_depth = "clip_feature"
    target_dic = {}
    if type == 'train':
        Target_Set = TrainSet
    elif type == 'test':
        Target_Set = TestSet
    for file in Target_Set:
        print("Loading {}".format(file))
        target_dic[file] = {}
        file_path = os.path.join(root_depth, file)
        for action in os.listdir(file_path):
            print("Loading {}".format(action))
            target_dic[file][action] = {}
            action_path = os.path.join(file_path, action)
            for cam_file in os.listdir(action_path):
                cam = cam_file.split(".")[0]
                cam_path = os.path.join(action_path, cam_file)
                # with open(cam_path, 'rb') as fl:
                    # data = pkl.load(fl)
                data = torch.load(cam_path)
                target_dic[file][action][cam] = data[0].cpu().numpy()
    # np.save("{}_depth.npy".format(type), target_dic)
    save_path = "{}_clip_tensor.pkl".format(type)
    with open(save_path, 'wb') as fl:
        pkl.dump(target_dic, fl)

def parser_filename(filename):
    subject, action, cam_file = filename.split('/')[1:]
    return subject, action, cam_file

def interprate(input_array, type = 'linear'):
    x_original = np.arange(48)
    x_new = np.linspace(0, 47, 243)
    interpolation_func = interp1d(x_original, x_new.T, kind=type, axis=0)
    output_array = interpolation_func(x_new).T
    print(output_array.shape)

def test_tensor_from_clip(filename):
    # tensor = torch.load(filename)
    with open(filename, 'rb') as tar:
        data = pkl.load(tar)
    # print("Done")

def read_img(filelist, img_embedding, img_preprocess, device):
    output_image_features = []
    for i in range(len(filelist)):
        images = [Image.open(x).convert("RGB") for x in filelist[i]]
        transformered_images =  torch.stack([img_preprocess(image) for image in images]).to(device)
        with torch.no_grad():
            image_features = img_embedding.encode_image(transformered_images)
            output_image_features.append(image_features.float())
    return torch.stack(output_image_features)


if __name__ == "__main__":
    # get_depth_dic("./z_depth", 'train')
    # print("Done")

    # filename = 'z_depth/S1/Directions_1/54138969'
    # subject, action, cam_file = parser_filename(filename)
    # print(subject, action, cam_file)

    # input = np.random.rand(48, 17)
    # interprate(input)

    # filename = "train_clip_tensor.pkl"
    # test_tensor_from_clip(filename)

    get_clip_tensor_dic('test')