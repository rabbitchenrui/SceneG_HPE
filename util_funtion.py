import os
import torch
import pickle as pkl

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

