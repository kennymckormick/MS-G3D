import sys
sys.path.extend(['../'])

import torch
import pickle
import numpy as np
import copy as cp
from torch.utils.data import Dataset
from mmcv import load

from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, dataset_name='gym_17p', maxlen=None,
                 manipulate_joints=[7, 8, 9, 10, 13, 14, 15, 16],
                 random_turb=False, random_turb_size=0., squeeze=False,
                 random_drop=False, drop_per_nframe=16, drop_njoints=2,
                 random_seed=1, kinetics_max_person=1, debug=False):
        """
        :param data_path:
        :param label_path:
        :param debug: If true, only use the first 100 samples
        """

        np.random.seed(random_seed)
        self.debug = debug
        self.data_path = data_path
        self.dataset_name = dataset_name
        # squeeze is an option for mmkpose
        self.kinetics_max_person = kinetics_max_person
        self.squeeze = squeeze

        # A list, each is a dictionary with keys 'kp', 'label', 'name'
        # The shape of 'kp' is 3 x T x K x M
        self.database = load(data_path)
        # remove None
        self.database = [x for x in self.database if x is not None]
        num_samples = len(self.database)
        print(f'{num_samples} annotations are valid! ')
        self.label = [x['label'] for x in self.database]
        self.sample_name = [x['frame_dir'] for x in self.database]
        self.maxlen = maxlen
        if self.maxlen is None:
            self.maxlen = max([item['kp'].shape[1] for item in self.database])

        assert isinstance(manipulate_joints, list)
        self.manipulate_joints = manipulate_joints

        self.random_drop = random_drop
        self.drop_per_nframe = drop_per_nframe
        self.drop_njoints = drop_njoints

        self.random_turb = random_turb
        if isinstance(random_turb_size, float):
            random_turb_size = [random_turb_size] * len(manipulate_joints)
        self.random_turb_size = random_turb_size

        ntu_groups = {}
        ntu_groups[10] = [11, 23, 24]
        ntu_groups[6] = [7, 21, 22]
        ntu_groups[18] = [19]
        ntu_groups[14] = [15]
        self.turb_groups = {'ntu': ntu_groups}
        if self.dataset_name in self.turb_groups:
            turb_group = self.turb_groups[self.dataset_name]
            for v in turb_group.values():
                for jt in v:
                    assert jt not in self.manipulate_joints

        self.jt2ind = {jt: i for i, jt in enumerate(self.manipulate_joints)}

    # Note that it only works for 2D dataset
    # Data is of the shape 3, T, V, M
    def _random_turb(self, data):
        # Use score to check
        mask = (data[2] > 0.01)
        # mask is of the shape: T, V, M (now)
        tight_mask = mask[:, self.manipulate_joints]
        # mask is of the shape: T, K, M (now)
        theta = np.random.random(tight_mask.shape) * np.pi
        delta_x, delta_y = np.cos(theta), np.sin(theta)
        delta = np.stack([delta_x, delta_y])
        # 2, T, K, M
        for i in range(delta.shape[2]):
            delta[:, :, i] *= self.random_turb_size[i]

        # 2, T, K, M
        tight_mask = np.stack([tight_mask] * 2)
        # 2, T, V, M
        mask = np.stack([mask] * 2)
        # By Group
        data[:2, :, self.manipulate_joints] += tight_mask * delta

        if self.dataset_name in self.turb_groups:
            turb_group = self.turb_groups[self.dataset_name]
            for k, v in turb_group:
                if k in self.manipulate_joints:
                    k_ind = self.jt2ind[k]
                    for jt in v:
                        data[:2, :, jt] += mask[:, :, jt] * delta[:, :, k_ind]
        return data

    # This is a relatively fix strategy
    def _random_drop(self, data):
        tlen = data.shape[1]
        for tidx in range(tlen):
            if np.random.random() < 1. / self.drop_per_nframe:
                jidxs = np.random.choice(self.manipulate_joints,
                                         size=self.drop_njoints,
                                         replace=False)
                for jidx in jidxs:
                    data[:, tidx, jidx, :] = 0.
        return data

    def _pad_to(self, data, length):
        tlen = data.shape[1]
        repeat_n = length // tlen
        remain = length - tlen * repeat_n
        return np.concatenate([data] * repeat_n + [data[:, :remain]], axis=1)

    def __len__(self):
        return len(self.database)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data = self.database[index]
        if 'kp' in data:
            data_numpy = cp.deepcopy(data['kp'])
            data_numpy = data_numpy.transpose([3, 1, 2, 0])
        else:
            # We are loading Kinetics, and the dataset is too large
            assert 'kinetics_kp' in data
            assert 'inds' in data
            kinetics_kp = data['kinetics_kp']
            inds = data['inds']
            num_frame = inds.shape[0]
            data_numpy = np.zeros([self.kinetics_max_person, num_frame, 17, 3],
                                   dtype=np.float16)
            cnt = 0
            for i in range(num_frame):
                st, ed = inds[i]
                num_person = min(self.kinetics_max_person, ed - st)
                if self.squeeze and num_person == 0:
                    continue
                data_numpy[:num_person, cnt] = kinetics_kp[st: st + num_person]
                cnt += 1
            data_numpy = data_numpy[:, :cnt]
            data_numpy = data_numpy.transpose([3, 1, 2, 0])

        label = data['label']
        if self.random_turb:
            data_numpy = self._random_turb(data_numpy)

        if self.random_drop:
            data_numpy = self._random_drop(data_numpy)

        data_numpy = self._pad_to(data_numpy, self.maxlen)
        if data_numpy.dtype != np.float32:
            data_numpy = data_numpy.astype(np.float32)
        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
