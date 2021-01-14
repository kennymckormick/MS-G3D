import sys
sys.path.extend(['../'])

import torch
import pickle
import numpy as np
from torch.utils.data import Dataset

from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path, manipulate_joints=[5, 6, 9, 10, 13, 14, 17, 18],
                 random_turb_size=0., random_choose=False, random_shift=False, turb_group=None,
                 random_move=False, window_size=-1, normalization=False, debug=False,
                 use_mmap=True, random_turb=False, random_seed=1, random_drop=False,
                 drop_per_nframe=16, original_len=None):
        """
        :param data_path:
        :param label_path:
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        np.random.seed(random_seed)
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.turb_group = turb_group
        self.random_drop = random_drop
        self.drop_per_nframe = drop_per_nframe
        if

        self.random_turb = random_turb
        assert isinstance(manipulate_joints, list)
        self.manipulate_joints = manipulate_joints
        if isinstance(random_turb_size, float):
            random_turb_size = [random_turb_size] * len(manipulate_joints)
        self.random_turb_size = random_turb_size
        ntu_groups = {}
        ntu_groups[10] = [11, 23, 24]
        ntu_groups[6] = [7, 21, 22]
        ntu_groups[18] = [19]
        ntu_groups[14] = [15]
        self.ntu_groups = ntu_groups
        if self.turb_group == 'ntu':
            for v in self.ntu_groups.values():
                for jt in v:
                    assert jt not in self.manipulate_joints

        self.jt2ind = {jt: i for i, jt in enumerate(self.manipulate_joints)}

        self.load_data()
        if normalization:
            self.get_mean_map()
        if random_turb:
            self._random_turb()
        if random_drop:
            self._random_drop()

    def load_data(self):
        # data: N(num_samples) C(3) T(temporal_length) V(num_kp) M(num_person)
        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')

        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    # Note that it only works for 2D dataset
    def _random_turb(self):
        mask = (self.data[:, :2] == 0) + (self.data[:, :2] <= -0.95)
        mask = (mask[:, 0] * mask[:, 1] == False)
        # mask is of the shape: N, T, V, M (now)
        tight_mask = mask[:, :, self.manipulate_joints]
        # mask is of the shape: N, T, K, M (now)
        theta = np.random.random(tight_mask.shape) * np.pi
        delta_x, delta_y = np.cos(theta), np.sin(theta)
        delta = np.concatenate([delta_x[:, np.newaxis],
                                delta_y[:, np.newaxis]], axis=1)
        # N, 2, T, K, M
        for i in range(delta.shape[3]):
            delta[:, :, :, i] *= self.random_turb_size[i]

        # N, 2, T, K, M
        tight_mask = np.concatenate([tight_mask[:, np.newaxis]] * 2, axis=1)
        # N, 2, T, V, M
        mask = np.concatenate([mask[:, np.newaxis]] * 2, axis=1)
        # By Group
        self.data[:, :2, :, self.manipulate_joints] += tight_mask * delta

        if self.turb_group == 'ntu':
            for k, v in self.ntu_groups:
                if k in self.manipulate_joints:
                    k_ind = self.jt2ind[k]
                    for jt in v:
                        self.data[:, :2, :, jt] += mask[:, :, :, jt] * delta[:, :, :, k_ind]

    def _random_drop(self):
        for i in range

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        if data_numpy.dtype != np.float32:
            data_numpy = data_numpy.astype(np.float32)

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def test(data_path, label_path, vid=None, graph=None, is_3d=False):
    '''
    vis the samples using matplotlib
    :param data_path:
    :param label_path:
    :param vid: the id of sample
    :param graph:
    :param is_3d: when vis NTU, set it True
    :return:
    '''
    import matplotlib.pyplot as plt
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path),
        batch_size=64,
        shuffle=False,
        num_workers=2)

    if vid is not None:
        sample_name = loader.dataset.sample_name
        sample_id = [name.split('.')[0] for name in sample_name]
        index = sample_id.index(vid)
        data, label, index = loader.dataset[index]
        data = data.reshape((1,) + data.shape)

        # for batch_idx, (data, label) in enumerate(loader):
        N, C, T, V, M = data.shape

        plt.ion()
        fig = plt.figure()
        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        if graph is None:
            p_type = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'k.', 'k.', 'k.']
            pose = [
                ax.plot(np.zeros(V), np.zeros(V), p_type[m])[0] for m in range(M)
            ]
            ax.axis([-1, 1, -1, 1])
            for t in range(T):
                for m in range(M):
                    pose[m].set_xdata(data[0, 0, t, :, m])
                    pose[m].set_ydata(data[0, 1, t, :, m])
                fig.canvas.draw()
                plt.pause(0.001)
        else:
            p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
            import sys
            from os import path
            sys.path.append(
                path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
            G = import_class(graph)()
            edge = G.inward
            pose = []
            for m in range(M):
                a = []
                for i in range(len(edge)):
                    if is_3d:
                        a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
                    else:
                        a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
                pose.append(a)
            ax.axis([-1, 1, -1, 1])
            if is_3d:
                ax.set_zlim3d(-1, 1)
            for t in range(T):
                for m in range(M):
                    for i, (v1, v2) in enumerate(edge):
                        x1 = data[0, :2, t, v1, m]
                        x2 = data[0, :2, t, v2, m]
                        if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                            pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                            pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                            if is_3d:
                                pose[m][i].set_3d_properties(data[0, 2, t, [v1, v2], m])
                fig.canvas.draw()
                # plt.savefig('/home/lshi/Desktop/skeleton_sequence/' + str(t) + '.jpg')
                plt.pause(0.01)


if __name__ == '__main__':
    import os
    os.environ['DISPLAY'] = 'localhost:10.0'
    data_path = "../data/ntu/xview/val_data_joint.npy"
    label_path = "../data/ntu/xview/val_label.pkl"
    graph = 'graph.ntu_rgb_d.Graph'
    test(data_path, label_path, vid='S004C001P003R001A032', graph=graph, is_3d=True)
    # data_path = "../data/kinetics/val_data.npy"
    # label_path = "../data/kinetics/val_label.pkl"
    # graph = 'graph.Kinetics'
    # test(data_path, label_path, vid='UOD7oll3Kqo', graph=graph)
