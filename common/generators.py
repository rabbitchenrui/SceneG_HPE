# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from itertools import zip_longest
import numpy as np
import pickle as pkl
import time

     
class ChunkedGenerator_Seq:
    """
    Batched data generator, used for training.
    The sequences are split into equal-length chunks and padded as necessary.
    
    Arguments:
    batch_size -- the batch size to use for training
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    chunk_length -- number of output frames to predict for each training example (usually 1)
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    shuffle -- randomly shuffle the dataset before each epoch
    random_seed -- initial seed to use for the random generator
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    def __init__(self, batch_size, cameras, poses_3d, poses_2d, filename_list,
                 chunk_length, pad=0, causal_shift=0,
                 shuffle=True, random_seed=1234,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert cameras is None or len(cameras) == len(poses_2d)
    
        # Build lineage info
        pairs = [] # (seq_idx, start_frame, end_frame, flip) tuples
        for i in range(len(poses_2d)):
            assert poses_3d is None or poses_2d[i].shape[0] == poses_3d[i].shape[0]
            n_chunks = (poses_2d[i].shape[0] + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - poses_2d[i].shape[0]) // 2
            bounds = np.arange(n_chunks+1)*chunk_length - offset
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], augment_vector)
            if augment:
                pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], ~augment_vector)

        # Initialize buffers
        if cameras is not None:
            self.batch_cam = np.empty((batch_size, cameras[0].shape[-1]))
        if poses_3d is not None:
            self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[0].shape[-2], poses_3d[0].shape[-1]))
        self.batch_2d = np.empty((batch_size, chunk_length, poses_2d[0].shape[-2], poses_2d[0].shape[-1]))
        self.filelist = [[] for i in range(batch_size)]
        self.z_depth = {}
        self.target_img_len = 48
        self.clip_feature_dim = 512
        self.batch_pesudo_depth = np.empty((batch_size, self.target_img_len, poses_2d[0].shape[-2]))
        self.batch_clip_tensor = np.empty((batch_size, self.target_img_len, self.clip_feature_dim))

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None
        
        self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d
        self.filename_list = filename_list
        self.train_depth_path = "train_depth.pkl" 
        self.train_depth = self.laod_z_depth()

        self.train_clip_feature_path = "train_clip_tensor.pkl"
        self.clip_feature = self.load_clip_feature()

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right

    def laod_z_depth(self):
        with open(self.train_depth_path, 'rb') as fl:
            data = pkl.load(fl)
        return data

    def load_clip_feature(self):
        with open(self.train_clip_feature_path, 'rb') as fl:
            data = pkl.load(fl)
        return data

    def parser_filename(self, filename):
        subject, action, cam_file = filename.split('/')[1:]
        return subject, action, cam_file

    def pad_with_left(self, lst):
        if len(lst) < self.target_img_len:
            first_element  = lst[0] if lst else None
            return [first_element] * (self.target_img_len - len(lst)) + lst
        return lst

    def pad_with_right(self, rst):
        if len(rst) < self.target_img_len:
            last_element  = rst[-1] if rst else None
            return  rst + [last_element] * (self.target_img_len - len(rst))
        return rst

    def num_frames(self):
        return self.num_batches * self.batch_size

    def batch_num(self):
        return self.num_batches
    
    def random_state(self):
        return self.random
    
    def set_random_state(self, random):
        self.random = random
        
    def augment_enabled(self):
        return self.augment
    
    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state
    
    def next_epoch(self):
        enabled = True
        while enabled:
            start_idx, pairs = self.next_pairs()
            for b_i in range(start_idx, self.num_batches):
                chunks = pairs[b_i*self.batch_size : (b_i+1)*self.batch_size]
                for i, (seq_i, start_3d, end_3d, flip) in enumerate(chunks):
                    # start_2d = start_3d - self.pad - self.causal_shift
                    start_2d = start_3d
                    start_fileidx = start_3d
                    # end_2d = end_3d + self.pad - self.causal_shift
                    end_2d = end_3d
                    end_fileidx = end_3d

                    # 2D poses
                    seq_2d = self.poses_2d[seq_i]
                    low_2d = max(start_2d, 0)
                    high_2d = min(end_2d, seq_2d.shape[0])
                    pad_left_2d = low_2d - start_2d
                    pad_right_2d = end_2d - high_2d
                    if pad_left_2d != 0 or pad_right_2d != 0:
                        self.batch_2d[i] = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
                    else:
                        self.batch_2d[i] = seq_2d[low_2d:high_2d]
                    
                    # get image idx
                    low_fileidx = max(start_fileidx, 0)
                    high_fileidx = min(end_fileidx, len(seq_2d))
                    pad_left_idx = low_fileidx - start_fileidx
                    pad_right_idx = end_fileidx - high_fileidx
                    fileidx = list(range(1, len(seq_2d), 5))
                    image_idx = [ x // 5 for x in fileidx if low_fileidx < x < high_fileidx]
                    if pad_left_idx != 0:
                        image_idx = self.pad_with_left(image_idx)
                    elif pad_right_idx != 0:
                        image_idx = self.pad_with_right(image_idx)
                    if len(image_idx) > self.target_img_len:
                        image_idx = image_idx[:-1]

                    #Pseudo-depth
                    subject, action, cam_file = self.parser_filename(self.filename_list[seq_i])
                    cam = cam_file.split(".")[0]
                    seq_depth = self.train_depth[subject][action][cam]
                    self.batch_pesudo_depth[i] = seq_depth[image_idx]

                    #Clip_image_feature
                    seq_clip_feature = self.clip_feature[subject][action][cam]
                    # print("feature.shape is", seq_clip_feature.shape)
                    self.batch_clip_tensor[i] = seq_clip_feature[image_idx]
                    # load_image
                    # seq_file_name = self.filename_list[seq_i]
                    # # image_filename = [ seq_file_name + "_" + str(x).zfill(6) + ".jpg" for x in image_idx]
                    # # image_filename = [ seq_file_name + ".pkl" for x in image_idx]
                    # image_filename = seq_file_name + ".pkl"
                    # self.filelist[i] = image_filename
                    # with open(self.filelist[i], 'rb') as file:
                    #     data = pkl.load(file)
                    

                    if flip:
                        # Flip 2D keypoints
                        self.batch_2d[i, :, :, 0] *= -1
                        self.batch_2d[i, :, self.kps_left + self.kps_right] = self.batch_2d[i, :, self.kps_right + self.kps_left]

                    # 3D poses
                    if self.poses_3d is not None:
                        seq_3d = self.poses_3d[seq_i]
                        low_3d = max(start_3d, 0)
                        high_3d = min(end_3d, seq_3d.shape[0])
                        pad_left_3d = low_3d - start_3d
                        pad_right_3d = end_3d - high_3d
                        if pad_left_3d != 0 or pad_right_3d != 0:
                            self.batch_3d[i] = np.pad(seq_3d[low_3d:high_3d], ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
                        else:
                            self.batch_3d[i] = seq_3d[low_3d:high_3d]

                        if flip:
                            # Flip 3D joints
                            self.batch_3d[i, :, :, 0] *= -1
                            self.batch_3d[i, :, self.joints_left + self.joints_right] = \
                                    self.batch_3d[i, :, self.joints_right + self.joints_left]

                    # Cameras
                    if self.cameras is not None:
                        self.batch_cam[i] = self.cameras[seq_i]
                        if flip:
                            # Flip horizontal distortion coefficients
                            self.batch_cam[i, 2] *= -1
                            self.batch_cam[i, 7] *= -1

                if self.endless:
                    self.state = (b_i + 1, pairs)
                if self.poses_3d is None and self.cameras is None:
                    yield None, None, self.batch_2d[:len(chunks)]
                elif self.poses_3d is not None and self.cameras is None:
                    yield None, self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]
                elif self.poses_3d is None:
                    yield self.batch_cam[:len(chunks)], None, self.batch_2d[:len(chunks)]
                else:
                    # yield self.batch_cam[:len(chunks)], self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)], self.filelist
                    yield self.batch_cam[:len(chunks)], self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)], self.batch_pesudo_depth, self.batch_clip_tensor
            
            if self.endless:
                self.state = None
            else:
                enabled = False


class UnchunkedGenerator_Seq:
    """
    Non-batched data generator, used for testing.
    Sequences are returned one at a time (i.e. batch size = 1), without chunking.
    
    If data augmentation is enabled, the batches contain two sequences (i.e. batch size = 2),
    the second of which is a mirrored version of the first.
    
    Arguments:
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    
    def __init__(self, cameras, poses_3d, poses_2d, filename_list, pad=0, causal_shift=0,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None):
        assert poses_3d is None or len(poses_3d) == len(poses_2d)
        assert cameras is None or len(cameras) == len(poses_2d)

        # self.augment = augment
        self.augment = False
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        
        self.pad = pad
        self.causal_shift = causal_shift
        self.cameras = [] if cameras is None else cameras
        self.poses_3d = [] if poses_3d is None else poses_3d
        self.poses_2d = poses_2d
        self.filename_list = filename_list
        # print("Done")
        self.TrainSet = ['S1', 'S5', 'S6', 'S7', 'S8']
        self.TestSet = ['S9', 'S11']
        self.depth_path = self.get_depth_path()
        self.depth = self.load_z_depth()

        self.clip_feature_path = self.get_clip_tensor_path()
        self.clip_feature = self.load_clip_feature()

    def parser_file_name(self, filename):
        subject, action, cam_file = filename.split("/")[1:]
        cam = cam_file.split(".")[0]
        return subject, action, cam

    def get_depth_path(self):
        if self.filename_list is not None:
            if self.filename_list[0].split("/")[1] in self.TrainSet:
                depth_path = "train_depth.pkl"
            elif self.filename_list[0].split("/")[1] in self.TestSet:
                depth_path =  "test_depth.pkl"
        return depth_path

    def get_clip_tensor_path(self):
        if self.filename_list is not None:
            if self.filename_list[0].split("/")[1] in self.TrainSet:
                clip_path = "train_clip_tensor.pkl"
            elif self.filename_list[0].split("/")[1] in self.TestSet:
                clip_path =  "test_clip_tensor.pkl"
        return clip_path

    def load_z_depth(self):
        with open(self.depth_path, 'rb') as fl:
            data = pkl.load(fl)
        return data

    def load_clip_feature(self):
        with open(self.clip_feature_path, 'rb') as fl:
            data = pkl.load(fl)
        return data

    def parser_filename(self, filename):
        subject, action, cam_file = filename.split('/')[1:]
        return subject, action, cam_file

    def num_frames(self):
        count = 0
        for p in self.poses_2d:
            count += p.shape[0]
        return count

    def batch_num(self):
        return self.num_batches

    def augment_enabled(self):
        return self.augment
    
    def set_augment(self, augment):
        self.augment = augment
    
    def next_epoch(self):
        for seq_cam, seq_3d, seq_2d, filename in zip_longest(self.cameras, self.poses_3d, self.poses_2d, self.filename_list):
            batch_cam = None if seq_cam is None else np.expand_dims(seq_cam, axis=0)
            batch_3d = None if seq_3d is None else np.expand_dims(seq_3d, axis=0)
            batch_2d = None if seq_2d is None else np.expand_dims(seq_2d, axis=0)
            subject, action, cam = self.parser_file_name(filename=filename)
            batch_depth = self.depth[subject][action][cam]
            batch_clip_feature = self.clip_feature[subject][action][cam]
            # batch_depth = None if sself.depth
            # batch_2d = np.expand_dims(np.pad(seq_2d,
            #                 ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
            #                 'edge'), axis=0)
            if self.augment:
                # Append flipped version
                if batch_cam is not None:
                    batch_cam = np.concatenate((batch_cam, batch_cam), axis=0)
                    batch_cam[1, 2] *= -1
                    batch_cam[1, 7] *= -1
                
                if batch_3d is not None:
                    batch_3d = np.concatenate((batch_3d, batch_3d), axis=0)
                    batch_3d[1, :, :, 0] *= -1
                    batch_3d[1, :, self.joints_left + self.joints_right] = batch_3d[1, :, self.joints_right + self.joints_left]

                batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
                batch_2d[1, :, :, 0] *= -1
                batch_2d[1, :, self.kps_left + self.kps_right] = batch_2d[1, :, self.kps_right + self.kps_left]
            # print(batch_2d.shape)
            yield batch_cam, batch_3d, batch_2d, batch_depth, batch_clip_feature 

class UnchunkedGenerator_Seq2Seq:
    """
    Non-batched data generator, used for testing.
    Sequences are returned one at a time (i.e. batch size = 1), without chunking.
    
    If data augmentation is enabled, the batches contain two sequences (i.e. batch size = 2),
    the second of which is a mirrored version of the first.
    
    Arguments:
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    
    def __init__(self, cameras, poses_3d, poses_2d, pad=0, causal_shift=0,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None):
        assert poses_3d is None or len(poses_3d) == len(poses_2d)
        assert cameras is None or len(cameras) == len(poses_2d)

        self.augment = False
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        
        self.pad = pad
        self.causal_shift = causal_shift
        self.cameras = [] if cameras is None else cameras
        self.poses_3d = [] if poses_3d is None else poses_3d
        self.poses_2d = poses_2d
        
    def num_frames(self):
        count = 0
        for p in self.poses_2d:
            count += p.shape[0]
        return count
    
    def augment_enabled(self):
        return self.augment

    def batch_num(self):
        return self.num_batches
    
    def set_augment(self, augment):
        self.augment = augment
    
    def next_epoch(self):
        for seq_cam, seq_3d, seq_2d in zip_longest(self.cameras, self.poses_3d, self.poses_2d):
            batch_cam = None if seq_cam is None else np.expand_dims(seq_cam, axis=0)
            batch_3d = None if seq_3d is None else np.expand_dims(np.pad(seq_3d,
                            ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
                            'edge'), axis=0)
            batch_2d = np.expand_dims(np.pad(seq_2d,
                            ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
                            'edge'), axis=0)
            if self.augment:
                # Append flipped version
                if batch_cam is not None:
                    batch_cam = np.concatenate((batch_cam, batch_cam), axis=0)
                    batch_cam[1, 2] *= -1
                    batch_cam[1, 7] *= -1
                
                if batch_3d is not None:
                    batch_3d = np.concatenate((batch_3d, batch_3d), axis=0)
                    batch_3d[1, :, :, 0] *= -1
                    batch_3d[1, :, self.joints_left + self.joints_right] = batch_3d[1, :, self.joints_right + self.joints_left]

                batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
                batch_2d[1, :, :, 0] *= -1
                batch_2d[1, :, self.kps_left + self.kps_right] = batch_2d[1, :, self.kps_right + self.kps_left]

            yield batch_cam, batch_3d, batch_2d
